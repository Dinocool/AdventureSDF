# Dynamic Large-Scene Residency — Execution Plan (current GPU front end)

Status: **PLAN OF RECORD** for scaling the streamed scene size. Worktree: `gpu-residency` (branch
`gpu-residency`, off `main`). Target: Bevy 0.18/0.19 + forked wgpu-trunk, RTX 4090 / Vulkan. This is the
*actionable execution plan on the current readback-free GPU residency front end*.

**Relationship to the other docs (read for background, do not re-derive):**
- `docs/VOXEL_LARGE_SCENE_PLAN.md` — the SOTA survey (§2: GigaVoxels / Aokana / Nanite / SVDAG / Teardown) and
  the cubic→quadratic brick-budget math (§3). Those sections are architecture-independent and remain valid.
  Its §1/§4/§5 *file references are the OLD CPU path* (`ResidencyManager`/`pack_resident_set`/`clip_half=8`/
  `max_resident=60k`) — superseded by this doc's §1.
- `docs/VOXEL_STORAGE_PLAN.md` — per-brick compression (R1 uniform-collapse, R2 palette, R3 dedup). Orthogonal:
  storage shrinks *bytes per brick*; this doc shrinks *brick count* and *makes the resident set budget-bounded*.
- `docs/GPU_VOXEL_WORLDGEN_PLAN.md` — the GPU-driven producer; demand/LRU (Phase 4 here) is its endgame.
- The non-negotiable invariant ([[feedback-one-gpu-residency-path]]): **ONE all-GPU, readback-free residency
  path for every scene.** Every change below is GPU-side; no CPU readback, no per-scene fork, no env gate.

---

## 1. Current ground truth (what the GPU front end already does)

The architecture has moved far past the old CPU plan. Verified against the code (2026-06-20):

- **Readback-free GPU pipeline.** `GpuResidencyFrontEnd` (`residency_front_end.rs`) +
  `StreamedResidencyPager` (`residency_pager.rs`) + `PagedBrickCoreStore` (`residency_gpu.rs`) run
  enumerate → diff/enter → pack → AABB → BLAS entirely on the GPU. The CPU only submits commands and does
  one-time pool allocation.
- **Surface classification is ALREADY live — at sector granularity.** The enumerate pass
  (`voxel_residency.wgsl` `classify_surface`, ~192-207) classifies surface-vs-interior using the per-sector
  occupancy mask + the per-sector FULL mask of the brick AND its 6 face-neighbours (`is_occupied`/`is_full`,
  ~177-180; sector masks built once at scene-load). **This is why Bistro is ~615k surface bricks resident, not
  its full volume.** The old plan's "Phase A surface-only" is *partly done* — coarsely.
- **Per-brick enclosed cull is ALREADY WIRED (Phase 1 is done).** Two layers, both live:
  - *Enter cull* — `classify_surface` (`voxel_residency.wgsl:192-207`) is a per-brick 6-face occlusion test
    using the per-brick `full` mask of the brick + its 6 neighbours; only bricks passing it enter
    `candidate_list` (the resident-target set, `:467`), and a brick that *becomes* enclosed is dropped
    (`:972-981`). Enclosed bricks **never get a residency slot**.
  - *AABB cull* — D3 `pack_build_commands` (`voxel_residency.wgsl:1396-1416`): even an entered surface brick
    with `has_air == 0` (`classify_brick` `voxel_pack.wgsl:510`) gets `buried = true` → degenerate AABB (no
    BLAS primitive), while staying resident to feed neighbours' halos.
  The resident set is therefore already Θ(H²) surface-only. (The original Phase-1 framing here was based on a
  research pass that read the CPU `incremental.rs` and missed D3 in `voxel_residency.wgsl`.) The remaining
  Phase-1-adjacent refinement — *freeing the pool slot* of a per-brick-buried dense brick (not just its BLAS
  primitive) — is entangled (the buried fact is known only after pack, and the core still feeds halos) and is
  low-value vs. C1/C2; deferred.
- **Halo = the 6 neighbours, already on-GPU.** `fill_halo` (`voxel_pack.wgsl:227-272`) reads all 27 same-LOD
  neighbours, with `NEIGHBOUR_ABSENT`→AIR and `NEIGHBOUR_SOLID`→synthetic-solid for occupied-but-unpaged
  neighbours. Shared SSOT between classify and pack. An enclosed predicate can ride this with no second fill.
- **AABB write is the cull point.** `write_aabb_dirty` (`voxel_pack.wgsl:375-427`) already branches on a
  per-slot `flag`: `1`→real `brick_aabb`, `0`→`degenerate_aabb` (BLAS non-candidate). Conditioning that flag is
  the lever.
- **Edit-exposure is already handled.** `neighbourhood_26` (`incremental.rs:573-587`) + `mark_rewritten`
  (~948-950) expand the dirty set to the owner + its 26 neighbours on every edit — *exactly* the set whose
  enclosed-ness can change. A dig that exposes a buried brick re-classifies it (now `any_air`) and re-packs it
  the same frame. Robust by construction.
- **Size-agnostic dispatch (landed, this branch).** The per-brick pack/classify passes are 2D-folded past the
  65535 workgroup-per-dimension cap (see [[voxel-rt-size-agnostic-dispatch]]). Bistro loads (`live≈521847`).
- **Current caps:** `max_resident=900_000` (clamp `1_048_576`, `streaming.rs`), `clip_half=160`. Bistro:
  ~143k cold-fill, ~615k surface peak, ~870k desired-shell superset — **~2.8× headroom** under the caps.

---

## 2. The concrete ceilings (current code, file:line)

| ID | Ceiling | Where | Trips when | Class |
|---|---|---|---|---|
| **C2** | `LIST_CAP = 1_000_000` transient per-frame lists (shell cells, cand/desired/enter/drop) + `would_overflow` bails the whole drive | `residency_front_end.rs:158`, `:699` | geometric shell-cell union `total_cells` > 1M, or `b0_wgs = total_cells/64 > 65535` — i.e. pushing `clip_half`/`MAX_LOD` past ~Bistro reach. Overflow → blank or permanent thrash. | **VIEW-DISTANCE WALL** |
| **C1** | ~2 GiB single-storage-buffer limit on the pools: core `MAX_CORE_BUFFER_CORES=900k`, index 512 w/brick, palette 256 w/brick, hard clamp `1_048_576` | `residency_pager.rs:41`, `incremental.rs:543,558`, `streaming.rs:~121` | the **surface set itself** exceeds ~900k bricks (city / forest / finer LOD0) | **RESIDENT-SIZE WALL** |
| C3 | core hash-table-full panic | `residency_gpu.rs:806` | invariant (table ≥ 2× cap); a *mis-sizing* guard, not a scene limit | invariant |
| C5 | `occ_capacity = next_pow2(2·max_resident)` | `residency_pager.rs:88-92` | conservative pre-size; per-region occupancy stays well under | invariant |
| C4 | per-brick dispatches | (pack/classify) | **already 2D-folded** (this branch); remaining 1D dispatches ≤ 32,768 WGs (safe) | done |

C1 and C2 are the only real scene-size walls. C2 caps *how far you can see*; C1 caps *how big the visible
surface can be*.

---

## 3. The sequenced plan (all five phases — bank cheap/safe first, endgame last)

Each phase: **independently shippable, GPU-side, with its own acceptance + benchmark gate.** Per the standing
mandates: implement in an interactive session ([[feedback-interactive-sessions-for-impl]]); each phase runs
specialist → ≥2 adversarial reviewers vs. the GPU ground truth → benchmark gate
([[feedback-agent-team-qa-per-stage]], [[feedback-benchmark-deliveries]]); design to the SOTA target, rip out
band-aids ([[feedback-plan-to-best-practice]]).

### Phase 1 — Per-brick enclosed cull (tighten surface-only) — **ALREADY DONE**
Verified live (see §1): the enter cull (`classify_surface`, per-brick `full` mask of brick + 6 neighbours,
`voxel_residency.wgsl:467`) keeps enclosed bricks out of the resident set, and D3's `has_air` test
(`voxel_residency.wgsl:1408`) gives any remaining buried surface brick a degenerate AABB. Resident set is
Θ(H²) surface-only; edit-exposure handled by the 26-neighbour dirty expansion (`incremental.rs:573`). No work
needed. Deferred refinement (low value): free the *pool slot* of a per-brick-buried dense brick, not just its
BLAS primitive — entangled (buried-ness known only post-pack; the core still feeds halos), revisit only if the
C1 pool wall proves tight after Phases 2–3.

### Phase 2 — Break C2 (decouple view distance from `LIST_CAP`) — 2a + 2b **DONE**

> **Implemented (2026-06-20):** 2a size-agnostic dispatch + 2b candidate-list-free **FUSED** enter (commits up to
> `748d1f2`). The original 2b sketch below was *windowed* enumeration; the shipped design is better — instead of
> windowing the candidate list, the enter side no longer MATERIALIZES candidates at all (the consumers are fused
> into the enumerate), so there's nothing to window. The view is bounded by the surface-CELL count
> (`shell_wg_indices`, OOB-guarded), not a per-frame candidate cap; `would_overflow` only trips past ~16.7M raw
> cells. Drop was first decoupled from enumeration (direct `present_contains`). Gates green; runs on 8 GB (no
> limit raise). Remaining tail (deferred): >`shell_wg_indices` solid cells clamps (would need windowing/enlarge);
> retire the stale `converge`/`diff_parity` harnesses; delete the residual dead `present_flag`/`candidate_list`
> buffers.

**MEMORY CONSTRAINT (user, 2026-06-20):** target devices include **8 GB VRAM** — be memory-efficient. This
*rules out* the naive "raise `LIST_CAP`" lever: the transient lists scale with it (`neighbour_indices` alone is
`LIST_CAP·108 B` — 432 MB at 4M). View distance must grow at **bounded transient memory**, i.e. by *tiling*,
not by a bigger cap.

**Phase 2a — Size-agnostic dispatch (DONE, memory-neutral).** The per-frame residency passes run at
`@workgroup_size(256)` and the indirect shell enumerate is 2D-folded (`finalize_shell_dispatch_2d`;
`enumerate_shells` recovers `wg = wid.x + wid.y·65535`), so **no dispatch hits the 65535-workgroup-per-dimension
cap regardless of brick/cell count**. This FIXES a real correctness ceiling — a dense scene with >65 535 solid
8³ WG-cells previously under-ran the 1D enumerate (silent holes) — at **zero extra memory** (it removes a
*dispatch* limit, not a size cap). `LIST_CAP` stays **1M** (108 MB `neighbour_indices`, fits the default 128 MB
`max_storage_buffer_binding_size` — runs on an 8 GB device with no limit raise). Gate: `enumerate_parity`
green; `paged_front_end_render` + `pack_parity` unchanged.

**Phase 2b — Tile the shell enumeration (the memory-efficient view-distance lever).** Process the shell union in
**bounded windows** so a far view (big `clip_half`/`MAX_LOD`, or a huge scene) needs **bounded** transient VRAM,
not a bigger cap — turning the current "candidate set > LIST_CAP ⇒ `would_overflow` skips the drive (blank/freeze)"
into "render the nearest pool-worth."

*Foundation already landed (commits `07d66ae`, `99bede5`):* the **drop** decision is now enumeration-independent
— `present_contains` computes `level_resident ∩ is_occupied` directly, so drop just scans the bounded slot_table;
no per-frame desired-set materialization. So tiling only has to window the **enter** (candidate) side.

*The enter side is already nearest-priority via a global distance histogram* (`enter_cap_histogram` →
`enter_cap_compute` cut from pool `room` → `diff_enter_scan` admits buckets below the cut). Tiling = run that in
two passes over windows, keeping the histogram (4096 buckets, tiny) GLOBAL:
  1. **Pass 1 (histogram):** for each window, enumerate its candidates into the LIST_CAP-sized `candidate_list`,
     fold into the GLOBAL `enter_hist` (do NOT clear between windows). After all windows the histogram is the full
     distance distribution.
  2. **`enter_cap_compute`** → the global cut bucket (≤ `room` nearest), unchanged.
  3. **Pass 2 (enter):** for each window, re-enumerate + `diff_enter_scan` admitting buckets below the global cut.
     Total entered = global count below the cut ≤ `room` — correct across windows by construction (no nearest
     brick dropped; contrast the simpler "near-first + clamp" heuristic, REJECTED — within-LOD flat order isn't
     distance order, so it can hole).
Window the **solid-cell list** (`shell_wg_indices`) in batches of `B` cells (each ≤ `B·512` candidates ≤ LIST_CAP
⇒ `B ≈ LIST_CAP/512`); the CPU drives the window loop (the permitted CPU↔GPU control). **Open tension to resolve
in design:** small `B` is overflow-safe but many dispatches (2 passes × ⌈cells/B⌉); validate the dispatch-count /
re-enumeration cost and consider a coarser `B` with a per-window candidate-overflow guard. Needs a NEW gate: a
small-`B` multi-window test (a modest scene spanning several windows) — the existing tests run a single window.

*Acceptance:* sweep `clip_half ∈ {160, 240, 320}` (and a >LIST_CAP-candidate scene), drive converges (no
thrash/blank) at **flat transient VRAM**, rendering the nearest pool-worth; resident count ~Θ(H²); report peak
transient + resident VRAM (8 GB gate). **Scope: a real subsystem (window loop + 2-pass + new test) — implement
with fresh focus, not as a tail-end edit.**

### Phase 3 — Break C1 (split/tiled pools past 2 GiB)
**Goal.** Let the resident *surface* set exceed ~900k bricks — the "3–10× Bistro, loaded whole" target.

**What changes.**
- Split each capped pool (core / index / palette) across **multiple storage buffers** behind a single logical
  addressing layer (slot → (buffer_index, local_offset)), lifting the single-buffer ~2 GiB wall. The fixed
  per-slot slab layout (this branch) makes the split clean: slab = `slot·stride` within its buffer.
- Carry the storage-plan R1 uniform-collapse **into VRAM** ([[voxel-storage-plan]]) so interior-adjacent
  surface bricks cost ~8 B not 4 KB — multiplies the per-buffer capacity. (Storage axis; do alongside.)
- Grow `max_resident` past `1_048_576`; size the core-store / occupancy hash from the multi-buffer capacity.

**Acceptance + benchmark.** A scene with a >900k surface set loads + renders correct; per-buffer high-water +
total VRAM reported; no slot-aliasing (the fixed-per-slot invariant holds across buffers). Peak-RAM/VRAM gate.

### Phase 4 — Demand / ray-guided residency + LRU (the unbounded answer)
**Goal.** Resident set bounded by a fixed VRAM **budget**, scene size unbounded — stream from disk/worldgen,
page in on demand, LRU-evict. GigaVoxels on our fixed-cap pool (`VOXEL_LARGE_SCENE_PLAN.md` §5).

**What changes (builds ON Phase 3's tiled pool — it becomes the LRU-managed pool).**
- **Ray-guided feedback buffer.** During the trace, a ray that wants a brick/LOD not resident appends
  `(coord, lod)` to a GPU request buffer (atomic counter; no readback). Consumed by the next streaming step.
- **Residency = requests + clipmap prefetch**, not a fixed cube. The clipmap becomes the near-field
  prefetch/seed; the far/detail fringe is demand-filled.
- **LRU eviction.** Each slot carries `last_used_frame` (written by the trace). Pool full → evict LRU surface
  slots (free slot + degenerate AABB; the per-chunk BLAS rebuild handles it).
- **Latency hiding (no black holes):** fall back to the coarser resident ancestor for a missing fine brick
  (the clipmap guarantees a coarse shell behind every fine one); keep-old-until-revealed; bound requests/frame.

**Correctness.** Live structure stays flat (BLAS + `voxel_offset` DDA unchanged — no tree on the hot path).
GI adapts locally, never resets ([[feedback-gi-adapt-not-reset]]).

**Acceptance + benchmark.** A scene far larger than the pool renders at a fixed VRAM budget with bounded
resident count; request count / pool occupancy / evictions-per-frame reported; a pathological all-surface fly
(forest/city) never exceeds the budget; no black holes under continuous motion.

### Phase 5 — Screen-error LOD selection (Nanite refinement)
**Goal.** Spend the far-field budget where pixel error is largest.

**What changes.** Replace the pure-distance shell test with a **projected-voxel-footprint** `want_lod`
(≈ one voxel ≈ one pixel), feeding Phase 4's request LOD. Optional polish once demand/LRU is live.

**Acceptance + benchmark.** Equal-or-better perceived detail at equal-or-lower resident count vs. the
distance-only LOD; report resident count + a perceptual/SSIM delta at matched budget.

---

## 4. Cross-cutting

- **Memory budget — must run on 8 GB VRAM (user, 2026-06-20).** Be memory-efficient: scaling levers must NOT
  balloon VRAM. This rules out "raise `LIST_CAP`" (transient lists scale with it) and "raise `max_resident`
  blindly" (pools scale with it). The memory-efficient levers are: surface-only residency (done — Θ(H²) not
  Θ(H³)), R1 uniform-collapse-into-VRAM (bytes/brick), TILING the enumeration (bounded transient), and
  demand/LRU (bounded resident set at a fixed budget). Every phase reports peak transient + pool VRAM and gates
  on an 8 GB ceiling, not just correctness. Size-agnostic *dispatch* (Phase 2a) is the model: remove a *limit*
  without growing a *buffer*.
- **One GPU path.** Every phase is readback-free and universal (no per-scene fork, no CPU fallback, no env
  gate) — [[feedback-one-gpu-residency-path]], [[feedback-gpu-readback-free-correct]]. Any slowness vs. the
  reference (GigaVoxels/Aokana) is OUR divergence, fixed by aligning, never a workaround.
- **Producer-agnostic.** Fed by the source/producer abstraction ([[voxel-residency-producer-abstraction]]) so
  `.vxo`, the GPU voxelizer (worldgen), and future producers all scale through the same path.
- **Benchmark harness.** Extend the residency perf rig with a **resident-count-vs-view-distance sweep** + a
  **VRAM-budget binary search** ("how far can we see / how big a surface at budget B"). Every phase gates on it
  ([[feedback-benchmark-deliveries]]). Measure the cubic→quadratic exponent, don't assert it.
- **Order recap + status:** **1 enclosed-cull = DONE** (surface-only, Θ(H²)); **2a size-agnostic dispatch =
  DONE**; **2b candidate-list-free fused enter = DONE** (far view at bounded transient VRAM) → **3** bigger-set
  (C1 tiled pools + R1-into-VRAM — the *actual* 8 GB-decisive lever: the ~4.6 GB pools) → **4** demand/LRU
  (unbounded, bounded VRAM, on the tiled pool) → **5** screen-error LOD (polish). Each de-risks the next; nothing
  is skipped; every step holds the 8 GB budget.

---

## 5. References
Current code: `src/voxel/{residency_front_end,residency_pager,residency_gpu,incremental,streaming,raytrace}.rs`,
`assets/shaders/{voxel_residency,voxel_pack,voxel_raytrace}.wgsl`. Background + SOTA + math:
`docs/VOXEL_LARGE_SCENE_PLAN.md` (§2–§3), `docs/VOXEL_STORAGE_PLAN.md`, `docs/GPU_VOXEL_WORLDGEN_PLAN.md`,
`docs/UNIFIED_GPU_RESIDENCY_PLAN.md`.
