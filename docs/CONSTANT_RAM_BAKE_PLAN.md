# Constant-RAM `.vxo` Bake — Implementation Plan

Status: APPROVED (user: memory-agnostic/robust/fast bake; chose full constant-RAM 2026-06-16). Worktree `voxel-rt`.
The bake must have **constant peak RSS regardless of scene SURFACE size** (not just volume). One path for all
scenes; no monolithic-vs-tiled fork that OOMs. See memory `feedback-bake-memory-agnostic`.

## Why (the freeze taxonomy — fold everything)
Two DISTINCT bottlenecks, on different code paths, both real:
1. **Offline bake OOM (this doc).** Bistro's 1.4 GB `.vox` OOMs the full-RAM `write_vxo`; the tiled path holds
   the whole LOD0 surface brick-map resident (`assemble_vxo_streaming`, voxelize_scene.rs:376, ~700 MB Bistro,
   scales with surface AREA). Fixed here.
2. **Runtime pack freeze (SEPARATE — tracked as the runtime track, memory `voxel-rt-live-freeze-pack`).** The
   instrumented live trace (trace-1781585620402546.json) shows the live freeze is `vox_pack_update`
   (`ResidentPacker` incremental GPU pack) — 4.16 s/load, worst frame 482 ms = pack+snapshot. The coarse
   *source* is 37 ms (a non-issue live). This bake does NOT change live frame time — it's the GPU-pivot's (#141)
   target. Near-term: arena pre-size kills the ~200 ms grow-snapshots; rayon over `pack_one`; then GPU pack.

## Sequencing
**Stage 0 (stepping-stone, lands first, fixes cold-load + ships corpus LODS within the CURRENT tiled budget):**
extract the shared coarse-bake SSOT, drive it in the tiled path from the resident map (bounded, ~1/7), enforce
`max_lod==MAX_LOD`, add the perf/RSS harness, re-bake the corpus (incl. Bistro's FIRST LODS). **Stages 1-3
(constant-RAM) then SUPERSEDE** Stage 0's resident-map producer with disk-spill + windowed coarse — same
`add_region`/`add_lod_region` sinks, same byte-identity parity gate, no format/reader change.

## Stage 0 — shared coarse-bake SSOT + max_lod invariant + harness (bounded)
- `drive_coarse_lods(pyramid, k, emit)` in writer.rs: the region-bucket + `(z,y,x)` ordering loop (the SSOT used
  by BOTH writers). `build_lods_body` refactored to call it (emit→RAM blob); the tiled path calls it
  (emit→`writer.add_lod_region`). `build_coarse_pyramid` → `pub(crate)`/exported.
- Wire into `assemble_vxo_streaming` (voxelize_scene.rs:~401, after base regions, map still resident):
  `drive_coarse_lods(&build_coarse_pyramid(&map), k, |l,rc,b| writer.add_lod_region(l,rc,b))`.
- **max_lod==MAX_LOD invariant:** `build_coarse_pyramid` already runs to MAX_LOD for non-empty maps; routing
  the streaming path through it guarantees it. Add `finish` assert `lod_levels.len()==MAX_LOD` for non-empty.
  Test the read-side `coarse_level` clamp is a guaranteed no-op (Stage-2 reviewer finding: a short pyramid
  diverges from `StaticVoxSource::level`).
- `examples/bake_perf.rs`: cold-bake wall time + peak RSS (Win `GetProcessMemoryInfo`; Linux `/proc/self/status`
  VmHWM), skip-graceful on absent Bistro.
- Re-bake corpus: 3 small via `vox_to_vxo` (already LODS via the shared SSOT); Bistro via tiled `voxelize_scene
  --tiled` (first LODS). Verify `has_lods()`+`max_lod()==MAX_LOD`. Rerun g2_gallery_profile (source now O(1)).
- Gate: byte-identity (`stream_writer_lods_matches_encode_vxo` routed through the SSOT) + builds + clippy.

## Stage 1 — constant-RAM base pass (region-bucketed disk SPILL)
A sliding window is NOT viable: `stream_final` is tile-blocked (tx fastest, then ty, tz), the anchor shift
(voxelize_scene.rs:355) is non-aligned so bricks straddle tiles → a window would be surface-AREA-scaled.
**Disk spill instead:** Pass-2 rewrite of `assemble_vxo_streaming`:
- **Spill pass:** for each solid → `(bc, local_index, block)` appended to per-region file
  `scratch/base_region_{r}.spill` via an LRU pool of ≤64 BufWriters. RAM = pool buffers (constant) + a
  `FxHashSet<IVec3>` of seen region coords (the single sub-linear residual: region-COUNT, sub-MB).
- **Assemble pass:** per region (sorted z,y,x): read its `.spill`, group by `bc` into bricks, `Brick::from_voxels`,
  `writer.add_region(rc, &bricks)`, drop, delete the file. Resident = ONE region's bricks (≤K³, constant).
- Correctness: each `(bc,local)` written exactly once (deterministic routing); the straddle disappears
  (completeness is per-region-file, not per-window). BRIK add-order = sorted region order (deterministic).
- `add_region` accepts any feed order (BIDX sorted at finish) — confirmed writer.rs:540.

## Stage 2 — windowed constant-RAM coarse downsample
Re-spill each level to per-(next-coarser-)region files. Build level L (1..=MAX_LOD, dense, z,y,x) from L-1's
spills: per coarse region, load the ≤8 finer regions covering the children footprint `[2·crc·K, 2·(crc+1)·K)`
into a transient map, `gather_children`→`downsample_children` (the EXACT SSOT) per coarse brick, emit via
`add_lod_region`, re-spill for L+1, drop the window. Resident = the ≤8-finer-region neighbourhood (constant).
Bit-identity hazard = the cross-region gather (a high-face coarse brick's `2·cc+1` children in the adjacent
finer region) — pinned by the byte-identity parity gate + a boundary-straddle synthetic case.

## Stage 3 — the constant-RAM gate (proof + speed)
Extend `bake_perf.rs`: bake a synthetic scene at 1× and 2× SURFACE (same AABB/sparsity); **assert
peak_RSS(2×) ≤ peak_RSS(1×)+slack** (flat — the constant-RAM proof) while scratch high-water ~2× (sanity).
Byte-identity parity (windowed bake == resident `encode_vxo`/`build_coarse_pyramid`) on a small scene — pins
base-region completeness + the coarse cross-region gather. Rayon over regions (assemble + windowed coarse,
independent), collect→serial-add in (z,y,x) order for reproducible BRIK; the byte-identity gate guards it.

## Hardest risk
The Stage-2 windowed coarse cross-region gather (subtly-wrong boundary bricks, no crash). Pinned deterministically
by the byte-identity gate + a synthetic case with surface bricks on a coarse-region boundary (coords ≡K-1 and ≡0).

## Per-stage QA (mandate)
specialist → ≥2 adversarial reviewers (SSOT/bit-identity + format/RAM-bound) → QA gate (byte-identity, both
builds zero-warning, clippy, the surface-invariance RSS test). Re-bake corpus through the unified path at the end.

## Critical files
`examples/voxelize_scene.rs` (assemble_vxo_streaming, tiled mod stream_final/bake_tiled, anchor shift),
`src/voxel/vxo/writer.rs` (VxoStreamWriter add_region/add_lod_region/finish, drive_coarse_lods, build_coarse_pyramid,
region_of_brick, encode_region_bricks), `src/voxel/source.rs` (downsample_brickmap/children/gather — the SSOT),
`src/voxel/vxo/reader.rs` (decode_region_span/decode_lod_region), `examples/bake_perf.rs` (new).
