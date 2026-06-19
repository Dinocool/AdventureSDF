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
- **Per-brick `any_air` is computed but UNUSED for residency.** `classify_brick` (`voxel_pack.wgsl:479,510`)
  scans all 1000 haloed cells and sets `any_air` (→ `has_air_bit`, bit 1 of `classify_out[0]`). A brick with
  `any_air == false` is provably enclosed (no ray reaches it). **This finer signal is currently never checked
  in the enter/AABB decision** — the gap Phase 1 closes.
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

### Phase 1 — Per-brick enclosed cull (tighten surface-only)
**Goal.** Don't make a provably-enclosed brick resident at all — finer than today's sector-level cull. Frees
C1 pool headroom *and* shrinks the BLAS (faster trace + faster dirty-chunk rebuild). The prerequisite
measurement for every later phase (you want the minimal surface set before sizing pools or deciding if
demand/LRU is even needed).

**What changes (all GPU-side, readback-free).**
- Promote the `any_air` signal from "computed but unused" to the **enter/AABB decision**: an entered brick with
  `any_air == false` AND all 6 face-neighbours full (via the halo's `NEIGHBOUR_SOLID`/occupancy info) →
  **degenerate AABB + no pool slot** rather than a real primitive. Reuse `write_aabb_dirty`'s existing `flag`
  and the `fill_halo` neighbour table; no new buffer, no CPU readback.
- Refinement over the sector cull: catches bricks the *sector* classifier called surface but are actually
  enclosed (e.g. a brick straddling a sector boundary). Unify with R1's uniform-incl-halo collapse on the one
  enclosed predicate (uniform-incl-halo ⊂ enclosed) per `VOXEL_LARGE_SCENE_PLAN.md` §4.2.
- Decide per-brick vs per-face fullness: start with the existing per-brick `is_full` (conservative toward
  keeping — never culls a possibly-exposed brick); upgrade to a per-face 64-bit plane test if the measured
  enclosed fraction warrants the extra ~6m²/m³.

**Correctness.** Exact for a first-hit DDA (enclosed ⇒ unreachable). Boundary safety: a missing/different-LOD
neighbour reads AIR → brick stays resident (conservative toward keeping). Edit-exposure: already covered by the
26-neighbour dirty expansion (§1).

**Acceptance + benchmark.** GPU oracle pixel-identical to today on Bistro + a solid building + Cornell; BLAS
primitive count + AABB/pool VRAM drop by the enclosed fraction; a dig exposes the interior brick in 1 frame
(promote test); a place that seals a face drops it (demote test). Report enclosed-fraction and resident-count
delta in the perf harness.

### Phase 2 — Break C2 (decouple view distance from `LIST_CAP`)
**Goal.** Let `clip_half`/`MAX_LOD` grow past Bistro reach so we can *see farther* without the drive bailing.

**What changes.**
- 2D-fold every remaining `LIST_CAP`-sized dispatch (same pattern as the landed pack/classify fix), and
  remove the `total_cells > LIST_CAP` / `b0_wgs > 65535` whole-drive bail (`residency_front_end.rs:699`).
- **Tile the shell enumeration** so `total_cells` is processed in bounded windows instead of one ≤1M list →
  the transient per-frame work is bounded by *window* size, not by view distance. Decouple `present_size`
  (the per-frame hash) from `LIST_CAP`.
- Keep the convergence guarantee: the desired/candidate lists must never silently truncate (that's the
  permanent-thrash BUG-2 the cap currently prevents) — tiling must be loss-less or explicitly logged
  ([[feedback-no-silent-layer-miss]]).

**Acceptance + benchmark.** Sweep `clip_half ∈ {160, 240, 320, …}` and confirm the drive converges (no thrash,
no blank) with resident count growing ~Θ(H²) (surface) not Θ(H³); fit the exponent (the cubic→quadratic claim
from `VOXEL_LARGE_SCENE_PLAN.md` §3, *measured*). Report max stable view distance.

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

- **One GPU path.** Every phase is readback-free and universal (no per-scene fork, no CPU fallback, no env
  gate) — [[feedback-one-gpu-residency-path]], [[feedback-gpu-readback-free-correct]]. Any slowness vs. the
  reference (GigaVoxels/Aokana) is OUR divergence, fixed by aligning, never a workaround.
- **Producer-agnostic.** Fed by the source/producer abstraction ([[voxel-residency-producer-abstraction]]) so
  `.vxo`, the GPU voxelizer (worldgen), and future producers all scale through the same path.
- **Benchmark harness.** Extend the residency perf rig with a **resident-count-vs-view-distance sweep** + a
  **VRAM-budget binary search** ("how far can we see / how big a surface at budget B"). Every phase gates on it
  ([[feedback-benchmark-deliveries]]). Measure the cubic→quadratic exponent, don't assert it.
- **Order recap:** 1 enclosed-cull (free headroom, smallest set) → 2 see-farther (C2) → 3 bigger-set (C1
  tiled pools) → 4 demand/LRU (unbounded, on the tiled pool) → 5 screen-error LOD (polish). Each de-risks the
  next; nothing is skipped.

---

## 5. References
Current code: `src/voxel/{residency_front_end,residency_pager,residency_gpu,incremental,streaming,raytrace}.rs`,
`assets/shaders/{voxel_residency,voxel_pack,voxel_raytrace}.wgsl`. Background + SOTA + math:
`docs/VOXEL_LARGE_SCENE_PLAN.md` (§2–§3), `docs/VOXEL_STORAGE_PLAN.md`, `docs/GPU_VOXEL_WORLDGEN_PLAN.md`,
`docs/UNIFIED_GPU_RESIDENCY_PLAN.md`.
