# Phase G (gallery) — GPU-driven readback-free PACK + AABB-fill + BLAS

Status: APPROVED design (2026-06-16), trace-driven. Worktree work off `main`. This is the gallery-focused
Phase G; the worldgen `NodeKind→WGSL` voxelize (GPU_VOXEL_WORLDGEN_PLAN.md Stage 1) is NOT on this path
(worldgen shelved — gallery bricks come from `.vxo` DECODE, not procedural voxelize).

## Why / target (instrumented live trace, after the LODS/grow-snapshot/off-origin/Tier-2 fixes)
Remaining gallery load freeze = `vox_pack_update` (CPU `ResidentPacker` pack into GPU buffers, 3.4 s/load,
~63 ms/call post-Tier-2) + `vox_blas_delta` (BLAS slot upload, 668 ms). Phase G moves the PACK + AABB-fill +
BLAS build to the GPU (readback-free), eliminating both. Enumeration (`vox_residency_classify` ~85 ms) is the
smallest cost → last.

## The one correctness anchor
The GPU pack MUST produce **byte-identical pool buffers** to the CPU `ResidentPacker`/`pack_one`. Every gate
keys on this (`delta_upload_matches_snapshot_buffers_over_sequence`, `incremental_matches_full_pack…`). Strategy:
`pack_one`/`encode_paletted`/`cell_block` (gpu.rs) stay the SSOT; the GPU reproduces them bit-for-bit, pinned by
a new headless GPU-vs-CPU byte-equality test reusing the existing comparators.

## Engine facts (verified)
AABB BLAS from a GPU buffer WORKS on the fork (a compute dispatch can fill `aabb_buffer` in the SAME submission
as the build). NO indirect AS build → fixed-cap pool (`N_capacity = max_resident_bricks`), real AABBs for
resident slots + **degenerate AABBs for free slots**. R2b paletted layout + the haloed brick + `BRICK_AABB_EPSILON`
+ the chunk-band BLAS are frozen.

## Stage G-a — GPU brick pool + GPU pack (kills `vox_pack_update`; FIRST)
**Split: allocation stays CPU, pure encode moves to GPU.**
- **CPU (incremental.rs):** dirty-set + 26-neighbourhood expansion, slot claim/release (`SlotAllocator`), `SlabArena`
  index+palette alloc, quarantine, the shadow byte-compare (keeps the delta O(actually-changed)). Emits per dirty
  brick `{slot, neighbour_indices[27], world_min, lod, index_word_offset, palette_word_offset, index_bits}` — the
  alloc decisions, NOT the bytes. New `ResidentPacker::update_gpu` alongside the Tier-2 `update`.
- **GPU (new `assets/shaders/voxel_pack.wgsl`):** one workgroup per brick — reproduce `pack_one`'s halo-fill (core
  + 26-neighbour border, AIR where absent, `halo_index` order) + `encode_paletted`, write the bit-packed index
  stream → `voxel_buf[index_word_offset]`, palette ids → `brick_palettes_buf[palette_word_offset]`,
  `GpuBrickMeta` → `meta_buf[slot·48]`. Inputs: a raw `8³`-core scratch upload for each dirty brick + its
  neighbours, and the per-brick command buffer. New `VoxelRtUpload::GpuPack` arm in `prepare_voxel_rt`.
- **Readback-free:** none on the hot path (CPU already knows `resident_count`/`grew()` from its own allocator).
- **HARDEST RISK — palette first-seen order:** `encode_paletted` appends ids in cell-iteration order; a parallel
  GPU encode permutes the palette → different bytes (decodes same, fails the byte gate). MITIGATION (designed-in):
  the palette-build step is SERIAL within the workgroup (one invocation walks the 1000 haloed cells in exact
  `halo_index` order building palette + local-index map in shared memory), then the workgroup bit-packs in
  parallel. `pow2_index_bits`-correct + order-identical by construction.
- **Gate:** `tests/voxel_gpu_pack_parity.rs` — CPU `update`+`snapshot_buffers` vs GPU pack (test-only readback),
  assert byte-identical `voxel`/`brick_palettes`/`meta` via the existing comparators (incremental/tests.rs) incl.
  PALETTE-BYTE equality (not just decode). Perf: GPU-path CPU cost <5 ms; `pack_one`+encode → 0 CPU.

## Stage G-b — GPU AABBs + fill-then-build in one submission (kills `vox_blas_delta`)
Fold the AABB write into the G-a shader (`brick_aabb(world_min, lod)`, GPU-portable; freed slots → `degenerate_aabb`
via a FREE-flag command). In `prepare_voxel_rt`'s GpuPack arm, ONE encoder: compute pass (fills aabb/meta/voxel/
palette) → if `topology_changed` `build_acceleration_structures` reading the same `aabb_buf` (the existing
dirty-chunk BLAS logic, lifted) → submit. DELETE the per-slot `queue_write_buffer(aabb)` loop. Gate: aabb_buf
byte-equal to CPU `SnapshotBuffers.aabbs` (incl. degenerate freed slots) + a render-level pixel-identity test
(voxel_render_headless on the GPU path). Fallback if a driver reorders: two submissions (still readback-free).

## Stage G-c — GPU residency enumeration + prefix-sum compaction (roadmap headline; LAST, smallest win)
The re-flora pattern (`D:\tmp_test\re-flora`: atomic-counter + compacted active-brick list + GPU-written indirect
dispatch). REUSE the engine's existing 3-pass prefix-sum compaction template (the world-cache
`wc_compact_single_block`/`_blocks`/`_write_active`). New `voxel_residency.wgsl` evaluates `desired_clipmap`
shells on GPU, queries occupancy (needs G6 GPU `classify` + the `.vxo` occupancy uploaded — the bigger
sub-project), atomic-compacts surviving `(coord,lod)` into the active list, writes the indirect dispatch for G-a.
Subsumes the cap-sort. Staged LAST: smallest measured win (~85 ms), biggest surface area.

## A/B gating + invariants
A `gpu_pack: bool` flag (default OFF until each stage's parity test is green); producer branches `update_gpu` vs
`update`, `prepare_voxel_rt` branches the `GpuPack` arm. Each stage flips independently. Preserved (every
reviewer): GI/ReSTIR untouched (only HOW the pool is written changes, not consumed); haloed-brick seam reproduced
(gate-pinned); AABB epsilon via the WGSL mirror; no indirect AS build (fixed-cap + degenerate free slots);
knobs-as-uniforms; 3 feature builds zero-warning; never git-restore world.graph.ron.

## Per-stage process (mandate)
specialist → ≥2 adversarial reviewers (byte-identity vs the CPU SSOT + the readback-free ordering + GI-untouched
regression) → QA gate (the parity test + the perf rig + 3 builds zero-warning, GPU tests with TMP=D:\tmp_test).
A/B-gated + parity-anchored + incremental — the most conservative path for the biggest architectural change.

## Critical files
`src/voxel/incremental.rs` (CPU alloc + the `update_gpu` split), `src/voxel/gpu.rs` (`pack_one`/`encode_paletted`/
`cell_block`/`brick_aabb` — the byte SSOT the shader matches), `src/voxel/raytrace.rs` (`prepare_voxel_rt`,
`build_scene_full`, `apply_delta`, the chunk-band BLAS, `VoxelRtPipelines`/`VoxelRtUpload`), new
`assets/shaders/voxel_pack.wgsl`, new `tests/voxel_gpu_pack_parity.rs`, the re-flora compaction reference.
