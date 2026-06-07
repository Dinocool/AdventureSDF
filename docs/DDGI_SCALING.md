# DDGI Scaling — research corpus, design, and techniques ledger

Living document for scaling our SDF DDGI (Dynamic Diffuse Global Illumination) to massive scenes while
staying memory-efficient and good-looking across a wide range of scene sizes. Update it as the
implementation lands and as new techniques are evaluated.

---

## 1. The problem

DDGI anchors one probe per occupied SDF brick (an 8×8 octahedral irradiance tile = 64 texels ×
`vec4<f32>` = **1 KB/probe**). At scale it broke two ways:

- **Memory / "GI doesn't work":** the irradiance buffer was sized by `atlas.tiles.high_water()` — *every
  resident brick at every clipmap LOD*. The clipmap keeps nested full discs, so a near surface is
  resident at LOD0…LOD7 at once; on a large spread scene each LOD disc is full of surface, so the buffer
  carried an ≈`lod_count`× redundant union and hit the ~2 GB storage-binding cap → far probes went
  inactive → distant GI holes.
- **Perf:** the trace dispatched `all_resident_chunks × 64` workgroups every frame (all LODs), even
  though only the finest-resident bricks do useful work.

The finest-resident "filter" only made the shader *early-out* — it never shrank the buffer or the
dispatch. The decoupled probe allocator that would have fixed it was dead code.

## 2. How the field scales probe GI (research)

- **Cascaded / clipmap volumes, cell size doubling per level** — the canonical large-scene approach.
  - *Godot SDFGI*: camera-following cascades, each doubling cell size, auto-linked min-cell/distance;
    cost dominated by cascade count + camera speed.
  - *NVIDIA "Scaling DDGI for Production"* (JCGT 10(2), 2021): multiresolution cascaded volumes →
    "practically unlimited view distance"; 1 km², 16k probes, ~2 ms (RTX 2080 Ti) with cascades.
  - **Our clipmap LOD already IS this hierarchy** — probes just need to ride it (finest-resident).
- **Bounded probe count is the whole game.** RTXGI caps it per cascade (e.g. 32×4×32 = 4096). Our
  per-brick scheme let it explode; finest-resident-LOD is our bound.
- **Memory arrangement.** RTXGI irradiance = 6×6+border octahedral in R11G11B10F (~590 kB/cascade),
  visibility 16×16 RG16F (~4 MB), <5 MB/cascade. Ours is `vec4<f32>` (16 B/texel) — ~4× heavier than a
  packed format.
- **Perf tricks.** Bounded rays/probe/frame (RTXGI default 192, range 100–300); partial/budgeted updates
  across frames (our `update_stride` round-robin); **probe classification** (off/dormant/active state
  machine) to skip ray work on probes that don't matter — RTXGI's main pruning trick. **Probe
  relocation** (push probes out of solids) — we already do this (`gs_origin` in the trace).
- Sources: NVIDIA Scaling DDGI (jcgt.org/published/0010/02/01), Majercik DDGI overview
  (morgan3d.github.io/articles/2019-04-01-ddgi), Godot SDFGI docs, RTXGI-DDGI `Algorithms.md`.

## 3. Chosen architecture

Probes **ride the existing clipmap LODs**: one probe per **finest-resident occupied brick** — dense
near (LOD0), coarse far (LOD7/8), whole scene covered. Count is bounded by the clipmap window, not the
scene's absolute size.

### A. Compact per-brick finest-resident allocation — `[DONE]`

- A free-list slot allocator (`chunk::SlotAllocator<K>`, generalized from the chunk slot allocator)
  keyed PER BRICK by `(ChunkKey, local)`, populated only for occupied bricks of **finest-resident**
  chunks. **One stable slot per finest brick** — exact (no intra-chunk waste, no all-LOD redundancy).
  Idempotent `alloc` → a brick that stays finest keeps the same slot = **boil-free** history.
- Each brick stores its slot in its tile-run record: **`BrickTile.probe_slot`** (widened 12→16 B).
  `ChunkLookup.probe_base` stays a per-chunk **flag** (`0` finest / `u32::MAX` non-finest) for a cheap
  whole-chunk/workgroup early-out. Probe slot for a brick = its `BrickTile.probe_slot`; sub-probe block =
  `probe_slot · subdiv³`.
- **Buffer sizing (the memory linchpin):** `chunk_cap.probe_slots = live.probe_high_water()`
  (`= probe_alloc.high_water()`, the per-brick finest count) — replaced `atlas.tiles.high_water()`.
- **Dispatch:** filtered to finest-resident chunks (`LiveChunkTables::finest_rows()`), so workgroup
  count is bounded by the clipmap window.
- **Lifecycle:** `refresh_probe_bases` (main world, on `topology_generation` change) allocs per occupied
  brick of finest chunks (sets `probe_slot` + the flag, marks the tile-run region dirty), releases on
  non-finest. Per-brick release on `clear_brick` + on chunk eviction (the `set_brick` overwrite belt).
- **Why per-brick, not 64-block-per-chunk:** a fixed 64-slot block per finest chunk wastes a slot for
  every unoccupied brick — thin surfaces (Cornell walls ~16/64, stress towers ~4/64) waste 4–16×, which
  on the stress field is *worse* than the original. Per-brick is exact.

Cost model: `probe_bytes = finest_occupied_bricks · subdiv³ · 64 · 16`.
Sites: `chunk.rs` (allocator, `refresh_probe_bases`, `probe_high_water`/`probe_count`/`finest_rows`,
`BrickTile`), `render/atlas_upload.rs` (sizing), `render/probe.rs` (finest dispatch),
`render/chunk_tables.rs` (16 B tile stride), `sdf/probe.wgsl` + `sdf_probe_trace.wgsl` (read
`BrickTile.probe_slot`), `sdf/bindings.wgsl` (WGSL `BrickTile`).

### B. Probe classification (state machine) — `[DONE]`

Cut steady-state ray work: a **converged** probe goes **dormant** — re-traces at `dormant_stride`
instead of `update_stride` (skips the ray-march, keeps its value). Implemented WITHOUT a new GPU buffer:
- **Per-probe state = the convergence count** already in the irradiance alpha (`N`, capped at
  `n_max = 1/(1-hysteresis)`). The trace reads texel-0's `N` (uniform across the workgroup): converged
  ⇔ `N ≥ n_cap`. `eff_stride = converged && classify ? dormant_stride : update_stride`
  (`sdf_probe_trace.wgsl`).
- **Safety via the settled gate (the key to no stale GI):** `classify` is only set once the scene has
  been UNCHANGED for a convergence window (`GiSettle.frames_unchanged > 192`). `track_gi_settle`
  (`mod.rs`) resets the counter on any topology / sun / GI-knob change. So a moving camera or changing
  light keeps every probe at full re-trace rate — and slot-churn (LOD paging) re-converges immediately,
  never showing a stale reused slot. Point-light moves aren't hashed (O(lights)); they're bounded by the
  `dormant_stride` re-trace (never permanently stale).
- Knobs: `DdgiParams.classify_enabled` + `dormant_stride` (default 32), with editor sliders.
- `ProbeParams` gained `dormant_stride` + `classify` (Rust + both WGSL copies + harness blob).

### C. Parametric Cornell scaling scenes + test bed — `[DONE]`

- `src/sdf_render/cornell.rs`: `cornell_room_volumes` + `spawn_cornell_grid(k)` tile `k×k` rooms;
  `generate_cornell_grid_scenes` writes `assets/scenes/cornell{2,4,8}.scene`.
- `tests/ddgi_harness.rs`: `cornell_grid_mini(k)` + `ddgi_scaling_gate` — bakes ever-larger grids and
  asserts (deterministically) that **probes/room DROPS as the world grows** (far rooms collapse to
  coarse LODs = probes ride the LODs), the compact buffer ≤ the all-LOD union, and GI covers near + far.
  Measured: probes/room 514 (k=1) → 154 (k=4); coverage centre + far corner both lit.

### D. Debug visualizations — `[DONE (probe-LOD + coverage); classification heatmap with B]`

`ShaderDebugRegistry` modes computed in `sdf_gi_resolve.wgsl` (it has the probe LOD walk), passed raw
through `sdf_deferred_lit.wgsl`:
- **Probe LOD** (`SDF_DEBUG_PROBE_LOD`): finest-resident probe LOD as a hue ramp (LOD0 red → coarse
  blue) — the clipmap annuli of the probe allocation; black = no probe.
- **Probe coverage** (`SDF_DEBUG_PROBE_COVERAGE`): green = covered, magenta = hole.
Live panel stats in the DDGI panel (`debug.rs`): finest probe count, irradiance MiB, redundancy removed
(`SdfAtlasStats.probe_redundancy = all-LOD bricks / finest probes`). The per-probe *classification*
heatmap lands with Part B (needs the dormancy state).

### E. Dispatch-level amortization — `[DONE]`

**Problem (real-scene profiling, cornell8):** the probe trace was **dispatch-bound**, not ray-bound — it
dispatched `finest_bricks × 64` workgroups EVERY frame (1 ms@cornell1 → 6 ms@cornell8). The
`update_stride` round-robin + dormancy skipped *rays* in-shader, but the workgroups still launched + read
occupancy + early-out — the 6 ms idle floor.

**Shipped: amortize at the DISPATCH level (`prepare_sdf_probe`, CPU).** Each frame only the SUBSET of
finest chunks whose turn it is is uploaded + dispatched. Rotation key = the chunk's stable tile-run slot
(`tile_run_base / TILE_RUN_SLOT`), so over `eff_stride` frames every chunk is covered once. `eff_stride`
= `update_stride` while settling, `dormant_stride` once CLASSIFY says the scene is settled (`GiSettle`).
The in-shader round-robin is disabled (pass `update_stride=1`, `classify=0`) — amortization is purely the
chunk subset. Result: idle dispatch ≈ `finest_chunks / dormant_stride` (6 ms → ~0.2 ms) + the edit
re-converge shrinks ~4× (active rate, not every brick). *(A per-probe GPU-compacted indirect dispatch
would lower the floor further — future refinement; the chunk-level version captured the win at far less
machinery.)*

### F. Localized per-region wake — `[DONE]`

**Problem:** any topology change reset `GiSettle` → classification off GLOBALLY → every finest probe
re-traced at full rate (FPS cliff on one cube move / cornell8 load).

**Shipped: localized wake.**
- `track_gi_settle` no longer resets on topology — only LIGHTING (sun, GI knobs) globally wakes (it
  affects all probes). Geometry changes are handled locally.
- `LiveChunkTables` records changed chunks (`set_brick`/`clear_brick` → `wake_keys`). `update_probe_wake`
  (main world) drains them, expands each to its 3×3×3 same-LOD neighbourhood (so adjacent contact
  shadows / colour bleed re-converge too), and keeps each woken chunk's tile-run slot active for
  `PROBE_WAKE_FRAMES` (90). The set is extracted (`ProbeWakeSet`) to `prepare_sdf_probe`.
- Dispatch: woken chunks rotate at `update_stride` (active); the rest at `dormant_stride`. So an edit
  re-converges only its neighbourhood (~0.3 ms bump) while the rest of a big static scene stays dormant —
  no global cliff. New/just-baked bricks (fresh `N=0`) converge fast on their own.
- *Residual:* woken probes re-converge via the temporal blend over ≈`n_max` active traces (~1 s local),
  not instant — a per-probe `N`-reset on wake would make it instant (future refinement; needs the
  scattered per-brick slot list).

### G. Central scene-switch eviction — `[DONE]`

**Problem:** switching scenes left the previous scene's data behind — the grow-with-headroom irradiance
buffer reused old slots' converged GI, and stale bricks/probe slots could linger — so a new scene
inherited the old one's GI until it slowly re-converged.

**Shipped: one central scene-switch signal + full SDF eviction.**
- `scene_manager::SceneSwitched` (a `Message`) is the single switch signal, fired by BOTH the editor
  tab swap (`editor::scene_tabs::load_doc_into_world`) and in-game `AppScene` transitions
  (`cleanup_scene_entities`). Reusable for any future in-game scene swap.
- `evict_on_scene_switch` (sdf_render, ungated) reacts: bumps `ProbeReset` (the render-world SDF
  cache-reset signal), resets `GiSettle` + the wake set, and calls `SdfAtlas::reset()` (clears bricks/
  tiles/chunk tables — and thus the probe slot allocator — + forces a full rebuild) and
  `BakeScheduler::reset()` (drops queued/in-flight bakes + clears `ring_chunk_origin` so the window
  re-bakes from scratch + bumps `edit_gen` to discard stale async results).
- The render world reacts to the `ProbeReset` bump in TWO prepares: `prepare_sdf_probe` recreates a
  ZEROED irradiance buffer (sized to the new scene), and **`prepare_sdf_atlas_gpu` reallocates fresh
  (zeroed) brick atlas PAGES** — the `dist` (R16Snorm) + `mat` (Rgba16Snorm) texel textures. This was
  the missed piece: the CPU tables + GPU lookup reset, but the texel pages persisted in VRAM, so a reused
  tile could show the previous scene's geometry (esp. when the new bake hash-skips it).
- Gap closed: closing the LAST editor scene now also fires `SceneSwitched` (it previously only despawned).
- Net: the incoming scene starts from a clean field — no stale GI, bricks, **atlas texels**, probe slots,
  or queued bakes.

## 4. Techniques ledger

| Technique | Status | Notes |
|---|---|---|
| Finest-resident per-brick compaction | DONE | The core fix (A). Memory + dispatch bounded by clipmap. |
| Probes ride clipmap LODs (dense near / coarse far) | DONE | Proven by the scaling gate (probes/room drops). |
| Probe relocation (push out of solids) | DONE | `gs_origin` in `sdf_probe_trace.wgsl`. |
| Round-robin partial updates (`update_stride`) | DONE | Pre-existing; amortizes ray cost across frames. |
| Probe classification / dormancy | DONE | Converged probes re-trace at `dormant_stride`; settled-gate (`GiSettle`) prevents stale GI. |
| Debug views (LOD / probe-state heatmaps) | TODO (D) | For inspecting + verifying the above. |
| Packed irradiance format (R11G11B10F / RGB9E5) | DEFERRED | ~4× memory; blocked on decoupling alpha=sample-count from rgb. |
| Dispatch-level amortization | DONE (E) | Rotate finest-chunk subset/frame (dormant_stride when settled); fixes the 6 ms idle. |
| Localized per-region wake | DONE (F) | Edits wake only the changed chunk + 3×3×3 neighbourhood; no global re-trace cliff. |
| Central scene-switch eviction | DONE (G) | `SceneSwitched` message → zero probes + reset atlas/chunks/scheduler; clean slate per scene. |
| Per-probe GPU-compacted indirect dispatch | IDEA | Lower the dispatch floor below chunk granularity (descriptor + atomic compaction + indirect args). |
| Instant wake N-reset | IDEA | Reset woken probes' sample count for instant local re-converge (needs per-brick slot list). |
| Visibility / Chebyshev moments (leak reduction) | IDEA | Separate planned stage; not part of scaling. |

## 5. Verification status

- `tests/ddgi_harness.rs`: 11/11 gates (incl. `ddgi_scaling_gate` — probes/room 514→154 across k 1→4,
  coverage near+far; `ddgi_buffer_bound_gate`).
- `cargo test --lib`: 168 pass (the one `instrument` failure under parallel runs is a pre-existing
  global-state race, passes single-threaded — unrelated). GPU rigs (`sdf_gpu_rig`, `sdf_shadow_harness`)
  pass with the 16 B tile stride. Shader compose (`shader_validation`): 11/11.
- Both build configs (`cargo build`, `--features editor`) + clippy (`--all-targets`, both feature sets):
  clean, zero warnings.
- Parts A–F all build (both configs) + clippy clean; the real engine boots clean on the default scene
  (the runtime caught + fixed a first-frame `GiSettle` extraction panic — now `Option`-guarded).
- **Pending (user runtime perf testing on real scenes):** confirm cornell8 idle drops from ~6 ms toward
  the dormant floor (E), and that moving a cube only re-converges its neighbourhood rather than the whole
  scene (F). The user drives this; harness numbers + clean boot cover correctness.
