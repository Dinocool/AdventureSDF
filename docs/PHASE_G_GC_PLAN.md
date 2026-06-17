# Track B / Phase G "G-c" — GPU-Driven, Readback-Free Voxel-RT Streaming

> Design of record for the GPU-driven residency front end. Produced by the Track-B architecture
> pass (read-only audit vs re-flora `D:/tmp_test/re-flora` + our code) after the Nsight/chrome-trace
> divergence audit (see memory `voxel-rt-perf-divergence-audit`). All file:line cites are this repo
> unless prefixed re-flora. Hand each stage to an implementation agent with the named parity + bench gate.

## 0. Executive framing — what G-c actually is

The measured 445 ms `vox_residency_classify` and the perpetual `vox_blas_delta`/`vox_repack` churn are
**one** bug: the residency DECISION (enumerate shells → classify per brick → cap-sort) lives on the CPU
main thread, runs the full clipmap shell on each brick crossing, and — because the distance cap
(`select_nth` at `streaming.rs:769`) churns membership at the `clip_half=160` boundary — never converges,
so the downstream GPU-pack + BLAS-delta fire forever.

G-c moves the **decision** to the GPU, exactly as re-flora `make_surface_sparse` + `prepare_sparse_surface_dispatch`
+ `contree` build does. The landed G-a/G-b (`apply_gpu_pack`, `voxel_pack.wgsl`) already moved the *pack +
AABB-fill + BLAS-build* to the GPU readback-free — but they are still **driven by a CPU-built `GpuPackBatch`**
whose dirty set comes from the CPU `ResidencyManager`/`ResidentPacker`. G-c replaces that CPU driver with a GPU
enumerate→classify→compact→indirect-dispatch front end, so the *entire* pipeline from "camera moved" to "BLAS
built" is GPU, readback-free, and **idempotent for a static camera**. Names already in `PHASE_G_GALLERY_PLAN.md:54`
and `GPU_VOXEL_WORLDGEN_PLAN.md:26`.

**Hard constraints (must hold):** (1) readback-free per frame — the only CPU↔GPU sync is the out-of-band,
1-frame-late `change_count` mirror; (2) no indirect AS build on the fork — fixed-cap pool + degenerate AABBs
for free slots (G-b, already landed); (3) full clip_half=160 reach — bound the working set by the VISIBLE
SURFACE (re-flora), never a CPU cap; (4) static camera → idempotent active set → idle; (5) keep-old-until-revealed
survives; (6) reuse landed G-a/G-b verbatim; **(7) do NOT depend on a haloed-`.vxo` memcpy** — the `.vxo` stores
8³ CORES (`vxo/writer.rs:5,12-14`) and the packer re-halos via `fill_halo` (`voxel_pack.wgsl:201`); keep that
(it is gate-pinned by `voxel_gpu_pack_parity.rs`). Haloed-on-disk is a separate optional optimization.

## 1. The full GPU pipeline — one encoder, one submit, no readback between passes

CPU writes ONE small uniform per tick (`ResidencyParams`: per-LOD `cam_brick_coord[8]`, `clip_half`,
`frame_index`, `epoch`) — the only hot-path CPU→GPU traffic. Mirrors re-flora `submit_build_surface`
(`surface/mod.rs:535-701`).

- **Pass A — `clear_counters`** (GPU-timeline clear, never host — re-flora warns `surface/mod.rs:624-629`):
  zero `residency_counters` (active_brick_count, enter/drop/change), seed the indirect-dispatch buffers `(0,1,1)`.
- **Pass B0 — `prepare_shell_dispatch`**: one invocation per shell sub-box WG-cell (`shell_subboxes`,
  `streaming.rs:326`, ported). Test the coarse LOD occupancy mask (§2.2); atomic-append SOLID 8³ WG-cells +
  `atomicMax` the shell dispatch. = re-flora `prepare_sparse_surface_dispatch.comp:39-78`. **Bounds the working
  set by the occupied surface, not the H³ cube — the reach bound (constraint 3).**
- **Pass B — `enumerate_shells`** (`record_indirect` over Pass B0's count): per brick in a solid cell, do the
  6-face occlusion cull (re-flora `is_occluded` `make_surface_sparse.comp:116-130` ≡ our `StaticVoxSource::classify`
  N6 test `source.rs:471-485`) AND the `level_resident` predicate (`streaming.rs:211`, ported). Surface bricks →
  `atomicOr` present-flag + `atomicAdd` active count + write key to `candidate_list`. = re-flora
  `make_surface_sparse.comp:181-230`. **Replaces the 445 ms CPU classify.**
- **Pass C — diff vs the resident slot table** (our addition for the fixed-cap pool + idempotency):
  - C1 enter scan (per candidate): absent in `slot_table` → claim a free slot (GPU free-list §2.3), append
    `enter_list`, `atomicAdd enter_count`, stamp epoch.
  - C2 drop scan (per occupied slot): not desired AND `safe_to_drop` (§3.2, keep-old-until-revealed GPU predicate)
    → append `drop_list`, release slot, `atomicAdd drop_count`.
  - `change_count = enter_count + drop_count` is the idempotency signal.
- **Pass D — build pack-command list + GPU indirect dispatch**: from enter/drop lists, emit the SAME
  `PackCommand`/`AabbCommand`/`ClassifyCommand` structs the landed `voxel_pack.wgsl` consumes (`:61,87,143`), incl.
  the 26-neighbourhood halo expansion (`neighbourhood_26`, `incremental.rs:560`, ported) and the `neighbour_indices`
  indirection (`voxel_pack.wgsl:123`); enters get real AABB (`flag=1`), drops degenerate (`flag=0`). `atomicMax`
  the pack/aabb/classify dispatch-indirect (re-flora `prepare_sparse_surface_dispatch.comp:77`).
- **Passes E/F/G — `classify_brick` / `pack_brick` / `write_aabb`** (LANDED `voxel_pack.wgsl:391/237/333`,
  `record_indirect`): unchanged; now driven by GPU-built commands. The G4 classify readback (`raytrace.rs:3231`) is
  retired — the index/palette slab alloc it fed moves GPU-side (§2.3).
- **Pass H — AS build** (fill-then-build, same encoder, §4): on `change_count>0` only, rebuild dirty chunk BLASes +
  TLAS reading the just-filled `aabb_buf` (existing `apply_gpu_pack:3481-3517`). One `submit` closes the encoder.

## 2. GPU data structures

- **2.1 Clipmap** — NEVER materialized; the `level_box`/`level_hole`/`level_resident` math (`streaming.rs:168-220`)
  ported verbatim to WGSL, evaluated per-brick from `ResidencyParams` (port the `snap_even_odd` `& !1`/`| 1`
  `streaming.rs:159` + `div_euclid` `streaming.rs:590` EXACTLY). 8 LODs × ±160 bricks = full reach at zero
  per-frame buffer cost. **This is how constraint 3 (full reach, no CPU cap) is met.**
- **2.2 `.vxo` occupancy on GPU** (face-cull input) — per-LOD sparse occupancy using the dubiousconst282 64-bit
  sector alloc-mask + base-slot + popcount (`VoxelNotes.md:289`): gives the coarse "is this 8³ cell solid?" (Pass B0)
  AND the per-brick face test (Pass B) from one fetch, ~1 bit/brick. Uploaded ONCE per region on CPU disk-page (§5),
  not per frame.
- **2.3 Slot allocation** — **dubiousconst282 popcount free-list for the PERSISTENT pool** (our fixed-cap pool needs
  persistent `key→slot` identity, `incremental.rs:8`) + **re-flora atomic-append for the per-frame transient
  `candidate_list`**. The GPU `slot_table` (hash `key→slot`) + free-list ring replace `SlotAllocator`
  (`incremental.rs:580`); the `SlabArena` index/palette allocators (`incremental.rs:433`) become GPU bump+free-list
  per size class, pre-sized to `max_resident_bricks` (RESERVE_* `incremental.rs:540/545`) so no mid-frame grow → no
  re-snapshot → no readback. Maps onto `ResidentPacker::resident` (`incremental.rs:731`) + the SAME persistent pool
  buffers `apply_gpu_pack` writes (`raytrace.rs:3431`).
- **2.4 Core pool + neighbour table** — cores read from the persistent brick-voxel store (`.vxo`-decoded 8³ cores,
  keyed by `(coord,lod)`, uploaded per region with the occupancy). Pass D builds `neighbour_indices` by `slot_table`
  lookups. (Halo reconstructed by `fill_halo` — constraint 7, NOT a memcpy.)

## 3. Convergence / idempotency (static camera → idle), readback-free

- **3.1 No-change detection without readback:** Passes E/F/G are `record_indirect` over GPU-written counts; when
  `change_count==0` Pass D writes `(0,1,1)` → zero workgroups, ~0 GPU cost, no CPU branch, no readback (re-flora
  `record_indirect` semantics). The AS build (Pass H) is the one thing the CPU must conditionally *record* (no
  indirect AS). Solution: a **non-blocking, 1-frame-late `change_count` mirror** (`map_async` of 4 bytes, read
  out-of-band — explicitly permitted by constraint 1) decides whether NEXT frame records the AS section. Static
  camera: frame N `change_count=0` → frame N+1 CPU sees it (non-blocking) → stops recording AS build → fully idle.
  The AS build may run one extra frame after the last change (harmless — identical AABBs).
- **3.2 Keep-old-until-revealed on GPU:** port `safe_to_drop` (`streaming.rs:587-600`) into Pass C2 — coarsened: walk
  `coord.div_euclid(2), lod+1` to first desired ancestor, droppable iff resident; refined: bounded iterative descent
  (`REFINE_DESCENT_CAP=5`, `streaming.rs:76`) over the child subtree, pruned at each desired brick. The deferred-free
  quarantine (`incremental.rs:755`, one-frame latency so an in-flight TLAS never sees an overwritten slot) becomes a
  GPU `quarantine_slots` ring released at the TOP of next frame's Pass A — the GPU analogue of the atomic scene swap
  (`build_scene_full:3100`).

## 4. AS build under the no-indirect-AS-build constraint (already solved by G-b)

Fixed-cap pool; BLAS built over CPU-known `prim_count` per `CHUNK_SLOTS=512` band (`create_chunk_blas:2955`); GPU
decides per-slot live (real AABB) vs free (degenerate `incremental.rs:551`) by writing `aabb_buf` in Pass G; the
build reads `aabb_buf` at execution time (fill-then-build, `apply_gpu_pack:3519`; verified fork fact
`VOXEL_LARGE_SCENE_PLAN.md:78`). **Dirty-chunk set must be GPU-written** (the CPU no longer sees the commands): Pass
D `atomicOr`s a `chunk_dirty_mask` bit per `slot/CHUNK_SLOTS`. **First shippable: rebuild ALL chunks on any
`change_count>0` frame** (simplest, correct, only costs on real-change frames). Optimize to the GPU mask + out-of-band
mirror in G-c.4 if change-frame cost shows in the bench.

## 5. CPU↔GPU boundary

**DELETED (the hot-path CPU work behind the walls):** `ResidencyManager::update` (`streaming.rs:657`, the 445 ms pass),
`desired_clipmap_surface` + the parallel classify + the `select_nth` cap-sort (`streaming.rs:769`), `BrickSource::classify`/
`surface_bricks_in` on the hot path (`source.rs:457/497`; kept for tests/oracle), `ResidentPacker::update_gpu*` + the
`GpuPackBatch` build (`incremental.rs`) incl. the G4 classify readback, `drain_work_from` (`streaming.rs:817`, for `.vxo`
the cores come from the GPU store), the CPU command-building in `apply_gpu_pack`.

**RETAINED (CPU owns):** camera + clip params → the `ResidencyParams` uniform (the only per-frame write); `.vxo` region
paging from disk (`VxoSource` `RegionCache` LRU `vxo/source.rs:76` → upload occupancy + cores per newly-paged region,
NOT per frame — the constant-RAM spine `CONSTANT_RAM_BAKE_PLAN.md`, exactly `GPU_VOXEL_WORLDGEN_PLAN.md:81`); scene-switch
/epoch pool alloc (`build_scene_full` StreamSnapshot `raytrace.rs:2817`); the non-blocking `change_count` mirror;
GI/ReSTIR/DLSS untouched.

> FLAG — worldgen vs `.vxo`: G-c assumes brick cores are GPU-resident from the `.vxo` store (paged by region), true once
> §5's region upload lands. A future worldgen path must GPU-voxelize into the same store (`GPU_VOXEL_WORLDGEN_PLAN.md:102`,
> Stage G1) — out of G-c scope.

## 6. Phased plan (individually shippable, bench-gated, A/B behind `VoxelRtToggle.gpu_residency`)

Bench every stage: `ADVENTURE_BENCH_BISTRO` + a scripted MOVING-camera fly-through (crosses brick boundaries → exercises
enter/drop) + a static hold. **Two gates each:** (1) fly-through max-frame-time improves toward bounded (445 ms → <16 ms);
(2) static hold converges to idle (`change_count→0`, pack/AS passes dispatch 0 WGs). Per-stage QA: specialist → ≥2
adversarial reviewers → parity + perf + 3 zero-warning builds (`PHASE_G_GALLERY_PLAN.md:69`).

- **G-c.0 — Upload `.vxo` occupancy + brick-core store to GPU** (prereq, no behaviour change). Build per-LOD sparse
  occupancy (§2.2) + core store (§2.4) from `VxoSource` region decode; upload per region; no consumer yet. *Verify:* GPU
  occupancy bit == `StaticVoxSource` occupied/`classify` over a sample (GPU-vs-CPU oracle, mirror `voxel_gpu_pack_parity.rs`).
- **G-c.1 — GPU enumerate + face-cull (Pass B/B0), readback for PARITY only.** Port `level_resident`/`shell_subboxes`/N6
  to `voxel_residency.wgsl`; read back `candidate_list` in the test, assert set-equality with `desired_clipmap_surface` +
  CPU `classify` over the fly-through (lift `shell_first_resident_set_matches_cube_oracle` to GPU-vs-CPU). Live path still
  CPU. **Top risk (GPU classify ≠ oracle) is gated HERE before any consumer exists.**
- **G-c.2 — GPU slot table + free-list + diff (Pass C/D) driving the LANDED pack.** Replace `update_gpu`'s CPU slot/arena
  alloc with the GPU `slot_table`/free-list/slab allocators (§2.3); Pass D builds the SAME command buffers `apply_gpu_pack`
  consumes; keep-old-until-revealed (§3.2) ported. *Gate:* `voxel_gpu_pack_parity.rs` extended — GPU-driven commands produce
  byte-identical `meta/voxel/brick_palettes/aabb` buffers to the CPU `ResidentPacker` over a move sequence (the make-or-break
  anchor, `PHASE_G_GALLERY_PLAN.md:13`). *Bench:* the 445 ms classify is gone (update/update_gpu deleted).
- **G-c.3 — Readback-free convergence + indirect dispatch (THE HEADLINE).** Wire `atomicMax` GPU-written indirect dispatches
  so E/F/G self-gate to 0 WGs at `change_count==0`; add the non-blocking 1-frame-late `change_count` mirror gating the AS-build
  recording; delete per-frame CPU command-building. *Gate:* static hold converges to idle (trace shows no `vox_blas_*`/pack
  work); pixel-identical to G-c.2; a shell-shift fly-through asserts no LOD-seam holes. *Bench:* fly-through max frame bounded
  (the GI/render budget, not 445 ms) AND static idle — the deliverable.
- **G-c.4 — LIVE render-graph drive + change_count mirror + bench (LANDED).** The proven readback-free front end
  (`tests/voxel_gpu_residency_converge.rs` `GpuFrontEnd`) is ported to production `src/voxel/residency_front_end.rs`
  (`GpuResidencyFrontEnd`) with EXTERNAL pool buffers (the live scene's `meta/voxel/brick_palettes/aabb`), a caller's-encoder
  `record_frame` (so the dirty-chunk BLAS rides the same submit, fill-then-build), and the non-blocking 1-frame-late
  `change_count` staging-ring mirror (`poll_change_count`/`advance_ring`). Driven live in `raytrace.rs`
  `drive_gpu_residency_front_end` (per-frame, behind `gpu_residency`, INDEPENDENT of the CPU patch generation):
  poll mirror → record front-end frame into the live pool → (if `change_count>0` or first bound frame) rebuild ALL dirty
  chunk BLASes + TLAS from the GPU-written `aabb_buf` → one submit → advance ring. The CPU `apply_gpu_pack` arm is SKIPPED
  when the front end drove (no double-write); toggle OFF is byte-unchanged. Camera + clip_half cross via the new per-frame
  `VoxelRtResidencyParams` extract (the only per-frame CPU→GPU residency traffic). Bench: `ADVENTURE_GPU_RESIDENCY=1`,
  `ADVENTURE_CAM_PATH` moving fly-through, `max_frame_ms` (hitch) reported. **VALIDATED on the IN-RAM path (Sponza):** front
  end builds + binds to the live pool, renders the correct Sponza atrium (GI-lit, banners), `max_frame_ms` 18–23 ms (vs the
  CPU path's ~317 ms classify freeze), and a static hold CONVERGES to idle (patch_gen/resident flat after settle; the mirror
  gates the AS build off).

  **REMAINING (the streamed-`.vxo` Bistro goal):** the `MergedSource`/`VxoSource` Bistro is region-paged from disk, so its
  occupancy + core store are NOT GPU-resident (G-c.0 deferred the eager build to preserve constant-RAM; a whole-scene core
  store is ~GBs for Bistro — it MUST be demand-paged). The live drive correctly FALLS BACK to the CPU pack for Bistro today
  (graceful, gated on `gpu_residency && gpu_core_store.is_some()`), so the Bistro `max_frame_ms` still shows the classify
  freeze. The remaining piece is **per-region occupancy + core paging** wired to the `VxoSource` `RegionCache` LRU:
  * **Occupancy (small, ~1 bit/brick, do this FIRST):** a GROWABLE GPU sector hash (a mutable `SectorOccupancy` whose
    `entries` buffer is `COPY_DST` and re-uploaded incrementally). On a `decoded_region`, OR each brick's occ/full bit into
    its sector mask + upload the touched sectors; on LRU eviction, the occupancy STAYS (it is tiny + the clipmap re-demands
    it) — i.e. occupancy is a one-way accumulating index of EVER-SEEN sectors (a few MB for all of Bistro), built lazily as
    the camera reveals regions. No eviction needed.
  * **Core store (large, MUST evict):** the cores are only needed transiently to PACK a newly-entered brick's halo. Demand-page
    them keyed by `(coord,lod)` with an LRU bounded to ~the resident-set footprint, uploaded on region decode, evicted on
    region drop. The live pool already holds the packed resident cores, so the core store only spans the in-flight enter set.
  * Both upload from `VxoSource::decoded_region` (the single region-decode hook) and are bounded by the existing region LRU —
    constant-RAM preserved (one region's bricks at a time, no whole-scene decode).
- **G-c.4+ (optional) — GPU `chunk_dirty_mask`.** Replace "rebuild ALL dirty chunks on change" with the GPU-written
  per-chunk dirty mask gating per-band BLAS rebuilds. Only if the bench shows change-frame BLAS cost.

## 7. Critical files + risks

**Files:** `src/voxel/vxo/source.rs`+`reader.rs` (region→occupancy/core), new `src/voxel/residency_gpu.rs` (GPU store/upload),
new `assets/shaders/voxel_residency.wgsl` (Passes A/B0/B/C/D + clipmap math port), `src/voxel/incremental.rs` (GPU slot/arena
allocator), `src/voxel/raytrace.rs` (`VoxelRtResources` buffers ~1253; `apply_gpu_pack` GPU-driven ~3380; new `GpuResidency`
prepare arm ~2907; indirect-dispatch recording; out-of-band mirror), tests `voxel_gpu_residency_parity.rs` (new) +
`voxel_gpu_pack_parity.rs` (extend). `voxel_pack.wgsl` reused VERBATIM (only who builds the commands changes).

**Top risks → de-risk:** (1) GPU classify vs oracle → G-c.1 is parity-only, gated before consumers; direct port of
`level_resident` + N6 with exact integer ops. (2) no-indirect-AS + degenerate slots → unchanged from shipped G-b. (3) VRAM
for full-reach occupancy → sparse sector-mask ~1 bit/brick, bounded by paged regions not the cube (kilobytes); core store
Θ(H²) surface-only (`VOXEL_LARGE_SCENE_PLAN.md:141`); measure on the clip_half sweep rig. (4) cross-LOD shell holes →
identical tiling SSOT + keep-old-until-revealed + `BRICK_AABB_EPSILON` seam (`voxel_pack.wgsl:112`); shell-shift test. (5)
idempotency drift → Pass C diffs SETS (append order irrelevant); "static hold N frames → change_count==0 every frame after
the first" test.

**Extends prior docs:** `PHASE_G_GALLERY_PLAN.md:54-60` (G-c named), `GPU_VOXEL_WORLDGEN_PLAN.md:26-27,59,107` (readback-free
pipeline + no-indirect-AS + "CPU keeps only camera/clip"), `VOXEL_LARGE_SCENE_PLAN.md:130-167` (surface-only Θ(H²) residency),
`CONSTANT_RAM_BAKE_PLAN.md` (per-region paging spine), `incremental.rs:51-165` + `voxel_pack.wgsl` (the GPU interface Pass D targets).
