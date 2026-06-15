# Voxel-RT Execution Program — audit-refined, plan of record

Status: APPROVED (user, 2026-06: "we'll go with your sequencing"). This is THE execution plan — it supersedes the
ad-hoc stage lists in the other docs and synthesizes the 5-agent SOTA-alignment audit (Storage, GI, Streaming/LOD,
AS/DDA, Voxelization). Detail per area still lives in the owning doc (`VOXEL_STORAGE_PLAN`, `VOXEL_LARGE_SCENE_PLAN`,
`VOXEL_INSTANCING_PLAN`, `VOXEL_FINE_RESOLUTION_PLAN`, `SOTA_REFERENCE`); this ties them into one sequenced program.
Read `ENGINE_OVERVIEW.md` first for the architecture.

## Alignment baseline (the audit verdict — don't re-derive)
SOTA-aligned: the in-shader DDA/trace, the streaming *bookkeeping* (exact clipmap tiling, surface `classify`), the GI
*math* (faithful/improved bevy_solari port), and Tier-A VRAM storage (R1 uniform-collapse + R2 palette + R3 dedup +
surface-only residency — all landed). The GAPS, in priority order, are: (1) the **GPU execution layer** — the live path
still does an O(resident) full buffer re-create + full single-BLAS rebuild every move (the 58–103 ms hitch), and the
fixed-cap incremental upload that fixes it is **built but unwired**; (2) **Tier-B disk/load** — R5 `.vxo` is undesigned
beyond "compress + expand"; (3) **import fidelity + the tiled bake** for 0.05 m; (4) **GI breadth** — no screen-space
ReSTIR DI pass / light-tile presampling (quality, independent of scene-loading).

## Agreed sequencing
**Phase A (GPU execution) FIRST** — a standalone win at the current 0.2 m (kills the per-move hitch, makes the
multi-scene gallery smooth NOW), ~90% already built. **Then B (disk) → C (bake) → D (flip + reach).** Build the AS work
with the instance descriptor **from day one**. GI 3.0 **deferred**. Re-bake **Sponza/Sibenik/Conference at 0.05 m
first** (Sponza already is); **Bistro after** the tiled voxelizer + R5a/R6.

---

## Phase A — GPU execution layer (0.2 m; mostly wiring already-built code)
- **A1 [BLOCKER, ~90% built] Wire the O(changed) GPU upload.** Switch the render path (`raytrace.rs:repack_streamed_resident_set`/`prepare_voxel_rt`) from `snapshot_patch` (contiguous, re-based, full re-create) to `snapshot_buffers` (fixed-capacity, stable slots) + consume `RepackDelta.changed` via `queue_write_buffer` (meta/aabb at `slot·stride`, dense block at `voxel_word_offset`); degenerate AABBs for free slots. Allocate the fixed-cap buffers ONCE. *(streaming F1 + AS-b; the machinery already exists in `incremental.rs`, untested only on the GPU side.)*
- **A2 [MAJOR, small] Cap-after-classify.** Move the `max_resident_bricks` drop AFTER the surface `classify` so the cap bounds the surface SHELL (Θ(H²)), not the clip VOLUME (Θ(H³)) — otherwise the surface-only win is undone at the larger `clip_half` the flip forces. *(streaming F2)*
- **A3 [BLOCKER, the big rewrite] Per-chunk BLAS + dirty-chunk rebuild, built WITH the `InstanceDescriptor` + 3×4 transforms + descriptor-indexed hit path FROM DAY ONE.** Chunk = KxKxK bricks; only changed chunks rebuild; gate refit-vs-rebuild on `topology_changed`. The streamed world is descriptor 0 (identity, base offsets 0) — the degenerate case. **Acceptance: Cornell/Sponza render pixel-identical** (the GPU oracle). This makes the load-bearing hot-loop rewrite happen ONCE; multi-instance FEATURE work then needs no second AS refactor. *(AS-a + AS-b; `VOXEL_INSTANCING_PLAN §2`)*
- **A4 [MAJOR] Robustness.** Retire the bit-31 uniform-flag invariant on `voxel_offset`/`palette_base` (move the flag to a free `GpuBrickMeta` pad bit — it's release-unchecked + pinned to today's 60k cap → silent corruption when the flip raises the budget); make `BRICK_AABB_EPSILON` relative-per-LOD (it shrinks 4× at 0.05 m); persist the interner so `snapshot_*` is truly O(changed). *(storage MAJOR/MINOR + AS-e)*
- **Gate:** worldgen-perf rig shows per-move re-pack is O(changed) (no full rebuild); GPU oracle pixel-identical; the 3-scene gallery streams smooth at 0.2 m.

## Phase B — disk storage (`.vxo`, properly)
- **B1 [BLOCKER] R5a — the disk format.** Chunked + **spatially-indexed** (a `BIDX`/HEAD region→`(offset,len)` directory — "stream by region" is unimplementable without it) + **pointer-free / mmap-able**; `BRIK` body = the R2 `(palette,index_bits,indices)` triple so a decoded chunk is a memcpy into the arena, not a re-encode. `.vox` becomes import-only.
- **B2 [BLOCKER] R5b — the region-streamed loader.** A `.vxo` scene is a `BrickSource` backed by disk chunks, feeding the EXISTING `ResidencyManager` demand path — unifying static-scene load with worldgen streaming on one residency SSOT. **Acceptance: peak RAM during a Bistro load < budget** (NOT just disk ratio — a 2.6 GB scene must never fully expand in RAM). *(storage BLOCKER)*
- **B3 [MAJOR, after B1/B2] R6 — SVDAG asset transport.** An offline DAG-merged `BRIK` variant for IMMUTABLE imports, decoded per chunk to the R2 brick form. The biggest disk lever (~0.12 bits/voxel → Bistro tens of MB). Static/no-edit only (no COW hazard). *(storage MAJOR — namechecked, never planned)*

## Phase C — bake / import for 0.05 m
- **C1 [BLOCKER, currently UNDESIGNED] Tiled bounded-RAM voxelizer.** The hard part is the **out-of-core floodfill**: enclosure (interior-solid vs open-air) is globally coupled, so a per-tile flood can't classify it. Design = per-tile local floods recording boundary-face air-connectivity → **union-find across tile faces** → components touching the global boundary = exterior → second pass fills the rest; disk-backed tiles. Bistro @0.05 m (>1.5 B dense) cannot bake without it. Needs `docs/TILED_VOXELIZER_PLAN.md`. *(voxelization BLOCKER — "#125" is referenced, not designed)*
- **C2 [BLOCKER for GI-on-imports] `.vox` MATL emissive reader.** Today `load_vox` reads only the 256-RGBA palette and DROPS `data.materials` → imported lamps import dark; the GI/NEE stack (our headline) can't light from imported assets. Small, independently-shippable; pull forward. *(voxelization BLOCKER)*
- **C3 [MAJOR] Import fidelity.** CIELAB-space palette clustering (replaces sRGB median-cut; lands within the 255 cap) + area-averaged albedo (box-filter/supersample the triangle's texel footprint, kills the nearest-texel aliasing); the `u16` cap-lift rides the `.vxo` MATL chunk. *(voxelization MAJOR — the asset-gen ports, currently bundled)*

## Phase D — flip + reach
- **D1** `VOXEL_SIZE` 0.2→0.05 in `brickmap.rs` (`BRICK_WORLD_SIZE` derives 0.4); re-pin EVERY test asserting 0.2/1.6/`brick_span`/reach/scene-dims; re-bake Sibenik/Conference/Bistro → `.vxo` (Sponza already 0.05 m).
- **D2 [MAJOR] Screen-error LOD as the reach mechanism.** Make `want_lod` a function of projected-voxel-footprint (pixels/voxel) — the flip QUARTERS LOD0 reach (13 m→3.2 m) and a brute `clip_half`/`MAX_LOD` bump is FOV/res-blind. *(streaming F4; `VOXEL_LARGE_SCENE` Phase C, promoted)*

---

## Deferred / independent tracks (NOT gating the flip)
- **GI 3.0** — screen-space ReSTIR DI pass + light-tile presampling (the missing third of the Solari stack; GI quality, independent of scene-loading). **Latent hazard to fix WHEN DI lands:** the cache-fed GI reservoir's inline `direct+emissive` will double-count once a DI pass exists — pick one SSOT for "where the bounce surface's direct term lives." *(GI MAJOR×3)*
- **Demand / ray-guided residency + LRU** — behind a concrete **gallery-worst-case measurement gate** (camera mid-gallery, all scenes at 0.05 m, surface-only on; if peak surface count > cap, it becomes required). The A1 fixed-cap slot pool is its substrate. *(streaming F3)*
- **R4 occupancy-mask + compacted solid list** — DEMOTED to conditional/post-flip; surface-only + R2 largely consumed its win. Decide from a post-flip fill-fraction histogram. *(storage MAJOR)*
- **Multi-instance FEATURE work** (`.vox`-as-instances, off-axis rotation, per-instance COW destruction) — deferred; the A3 descriptor makes it a no-second-rewrite addition. *(VOXEL_INSTANCING Phases 2–6)*
- **Worldgen `classify` + coarse-LOD SSOT unification** — when 3D worldgen (caves/overhangs) lands, replace the 2.5D height cull with the static path's 6-neighbour predicate. *(streaming F5/F7)*
- **In-`dda_brick` occupancy-mask empty-space skip** + removing the redundant double-DDA — after Phase A; measure first (the trace is occupancy/register-bound, not memory-bound). *(AS-c)*

## Doc-accuracy fixes (apply with the next commit)
- `VOXEL_STORAGE_PLAN`: R4 (per-brick byte compression) ≠ surface-only residency (whole-brick cull) — they're orthogonal, currently conflated; R3's surface-brick dedup expectation is optimistic (halo variation limits hits to interior/strata).
- `SOTA_REFERENCE`: ReSTIR DI is listed ADOPTED but is NOT implemented (no DI pass); specular GI is silently deferred (mark it); the always-on interior floodfill is DONE (listed as "to port").
- Note in `VOXEL_LARGE_SCENE`: surface-only residency IS implemented (`source.rs` `classify` + `brickmap.rs` `is_full`), and the `RepackDelta` upload is built-but-unwired (A1).

## Status
R1 ✓ · **R2 ✓** (R2b landed, commit `d17f30d2`: `GpuBrickMeta` 48 B with `palette_base` + `index_bits` packed in `lod`; `voxels` is now the bit-packed index stream + a `brick_palettes` buffer at group0/binding12; SSOT decode = `GpuBrickPatch::cell_block`. **10.7× resident VRAM** on the worldgen slice, 41→3.8 MB; GPU oracle byte/pixel-identical) · R3 ✓ · surface-only `classify` ✓.

**A1 ✓ (LANDED) — the O(changed) GPU upload is WIRED.** Chose representation **(a)/A1-β**: a RAW fixed-block voxel arena (`SnapshotBuffers`) + an `index_bits == 0 ⇒ raw` decode branch in `cell_block` (CPU SSOT + WGSL mirror); the static `pack_brickmap`/`pack_resident_set` path keeps R2b (`index_bits >= 1`). `VoxelRtPatch` now carries a `VoxelRtUpload::{Snapshot, StreamSnapshot, Delta}` enum + a scene `epoch`. A streamed epoch ships ONE `StreamSnapshot` (the render path allocates the fixed-cap meta/aabb/voxel/palette buffers ONCE with `COPY_DST` + builds the BLAS over `capacity`, degenerate AABBs for free slots), then every later move ships a `Delta` (the render path `queue_write_buffer`s ONLY the changed slots — meta@`slot·48`, aabb@`slot·32`, the raw 4 KB block@`voxel_word_offset·4` — and rebuilds the BLAS in place ONLY on `topology_changed`). NEE lights ship whole per generation (free when the registry has no emitters). **Perf (worldgen slice, 10k resident):** per-move re-pack **1553 ms → 6.8 ms (228×)**; per-move GPU upload **~1.25 MB = 0.51% of the 245 MB fixed-cap arena the old path re-created every move**; the full per-generation BLAS rebuild is gone on non-topology moves. Tradeoff (documented): A1-β's raw arena gives up R2b's voxel-VRAM win on the STREAMED path (each dense block is a raw 4 KB `10³` grid, not a bit-packed stream) — recovered later via A4.4 (persistent interner). Gates green: `voxel_raytrace_gpu` oracle + the full GPU suite + the new `delta_upload_matches_snapshot_buffers_over_sequence` / `raw_arena_decodes_same_logical_cells_as_r2b` byte-identity tests + zero-warning default/editor build + clippy.

**NEXT = Phase A2** (cap-after-classify) per `PHASE_A_GPU_EXECUTION.md`.

**R2b reconciliation for A1 (IMPORTANT — the design doc predates R2b's final state):** R2b **removed `snapshot_buffers`** (the raw fixed-block arena) and made the per-brick voxel payload a **VARIABLE-size index stream** (`index_bits·1000/32` words: 32 for 1-bit … 500 for 16-bit). So A1's fixed-capacity O(changed) upload can no longer assume a 1000-u32 fixed block per slot. A1 must choose: **(a)** re-add a RAW fixed-block path + an `index_bits==0 ⇒ raw` shader decode branch (the doc's "A1-β" — keeps the fixed-block free-list + O(changed) `queue_write_buffer`, at the cost of R2's VRAM win on the *streamed* path; recover it later via the persistent-interner A4.4), or **(b)** a variable-size paletted index arena (keeps R2 VRAM, needs a non-fixed-block allocator). Recommend (a) first. The A1 agent reconciles against the post-R2b code, not the pre-R2b doc text.
