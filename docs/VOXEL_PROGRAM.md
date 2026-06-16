# Voxel-RT Execution Program — audit-refined, plan of record

Status: APPROVED (user, 2026-06: "we'll go with your sequencing"). This is THE execution plan — it supersedes the
ad-hoc stage lists in the other docs and synthesizes the 5-agent SOTA-alignment audit (Storage, GI, Streaming/LOD,
AS/DDA, Voxelization) + the later 4-agent SOTA-GAP audit (2026-06, incl. a re-flora deep-dive — see the
`## Committed roadmap` section below + the `voxel-rt-sota-gap-analysis` memory). Detail per area still lives in the
owning doc (`VOXEL_STORAGE_PLAN`, `VOXEL_LARGE_SCENE_PLAN`, `VOXEL_INSTANCING_PLAN`, `VOXEL_FINE_RESOLUTION_PLAN`,
`GPU_VOXEL_WORLDGEN_PLAN`, `TILED_VOXELIZER_PLAN`, `SOTA_REFERENCE`); this ties them into one sequenced program.
Read `ENGINE_OVERVIEW.md` first for the architecture.

**USER DECISION (2026-06): the GPU-driven pivot (Phase G below) is COMMITTED regardless of performance — it is the
CORRECT architecture, not a perf-gated choice ([[feedback-plan-to-best-practice]]).** The CPU-side D1d/A1 wins do
NOT change the go decision; any re-benchmark is informational only. We are doing the whole GPU-driven model + the
full committed roadmap below.

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

## Committed roadmap — the full SOTA-gap set (2026-06 4-agent audit; ALL committed)

The audit (storage / GI / streaming / re-flora deep-dive, code-grounded) confirmed we are SOTA-aligned and AHEAD
of OSS peers (incl. re-flora — a SW 64-tree, 1-bounce GI, no-LOD engine) on every RENDERING axis. The gaps are
execution + a convergent set, ALL now committed (✅ done · 🔜 to do · 🔬 research · ⛔ verified-reject). The single
biggest convergent finding (two independent agents): move the residency **ENUMERATION/COMPACTION** to GPU, not just
voxelize — the re-flora `contree/` build + gvox_engine GPU allocator are working references.

### Phase G — GPU-driven pipeline (the committed pivot; owning doc `GPU_VOXEL_WORLDGEN_PLAN.md`)
Headline = the residency enumeration/compaction on GPU (not merely GPU voxelize).
- 🔜 **G1 GPU residency enumeration + classify** — kill the per-camera-crossing CPU enumerate→classify→sort; evaluate `classify` per candidate on GPU.
- 🔜 **G2 Readback-free build** — workgroup prefix-sum **stream compaction** + atomic **sparse active-brick list** + **GPU-written indirect dispatch** (the re-flora pattern; zero render-path readback).
- 🔜 **G3 GPU voxelization** — Field/NodeKind graph → WGSL codegen + a GPU/CPU parity test.
- 🔜 **G4 GPU-resident pool allocator** — fixed-cap, degenerate-AABB free slots (extends the A1 substrate GPU-side); optional adaptive high-water capacity readback (no indirect AS build on the fork).
- 🔜 **G5 Occupancy-aware dispatch** — compact-then-dispatch (work-graphs are wgpu-unavailable; the manual prefix-sum-compact is the portable substitute; we are occupancy/register-bound per the SDF trace finding).
- 🔜 **G6 3D-capable surface enumeration** — GPU 3D-occupancy classify so shell-first survives caves/overhangs (today's column-walk is 2.5D-only → O(H³) fallback when 3D worldgen lands). Unifies worldgen `classify` with the static 6-neighbour predicate (was streaming F5/F7).
- 🔜 **G7 GPU edit path** — edit → re-extract surface → rebuild region on GPU (destructible at scale).

### Streaming / LOD (rest of `VOXEL_LARGE_SCENE_PLAN.md`)
- 🔜 **D2 Screen-error / projected-footprint LOD** — PROMOTED ahead of/with the flip (distance-only shells are FOV/res-blind; the 0.05 flip makes it near-mandatory).
- 🔜 **S-async Async-offload the `update` enumeration** — cheap interim before G1 (keep-old-until-revealed tolerates a frame of latency).
- 🔜 **S-prefetch Async region prefetch** — fetch the shell one ring out (the `.vxo` B2.3 deferred piece).
- 🔜 **S-lru Ray-guided demand paging + LRU** (GigaVoxels) — for all-surface views (forest/city); layers on the G4 pool.

### GI 3.0 (`SOTA_REFERENCE §2`)
- 🔜 **GI-DI Screen-space ReSTIR DI pass** — crisp, low-variance direct light from emissive voxels (today only laundered through the world cache → soft/laggy, no contact shadows).
- 🔜 **GI-tiles Light-tile presampling** (RTXDI) — scales NEE to thousands of emissive voxels (the destructible target).
- 🔜 **GI-regir ReGIR grid light reservoirs** — alternative/complement to light-tiles for many-lights (pick one, not both).
- 🔜 **GI-dc Double-count SSOT fix** — when DI lands, the cache-fed GI reservoir's inline direct+emissive double-counts; one SSOT for where the bounce-surface direct term lives.
- 🔜 **GI-thinwall Thin-wall ReSTIR-reuse cap** — `surfaces_dissimilar` `TODO(D-GI)`: the relative threshold leaks reuse across a thin wall >~16.7 m within the 64 m reach.
- 🔜 **GI-denoise Non-DLSS à-trous/SVGF fallback** — portability off NVIDIA/Vulkan (re-flora's SVGF is a reference).
- 🔜 **GI-spec Specular/glossy GI** — after a material-system expansion (wet/metal/glass voxels).

### Storage / acceleration tail
- 🔜 **B3 SVDAG transport** — `.vxo` `flags` bit1; ~0.12 bits/voxel; immutable assets only (no COW hazard); decode-to-R2b Tier-B.
- 🔜 **AS-c In-brick occupancy empty-space skip / two-level DDA** — 4³ sub-bitmask skip (measure-first; may be occupancy/register-bound not memory-bound).
- 🔜 **R4 Occupancy-mask compacted format** — demoted; decide from a post-flip fill-fraction histogram.
- 🔬 **Per-format auto-codegen** (Hybrid Voxel Formats, arXiv 2410.14128) — research; per-level heterogeneous format via codegen; read before hand-coding R4 (aligns with the G3 NodeKind→WGSL direction).

### Import / bake
- 🔜 **C1 Tiled out-of-core voxelizer** — Bistro @0.05 m bake <4 GiB (union-find tiled flood; `TILED_VOXELIZER_PLAN.md`). [C2 emissive ✅, C3 CIELAB+area-avg ✅]

### Minor / aesthetic (re-flora landscape)
- 🔜 Baked per-voxel gradient normals (opt-in SMOOTH-voxel material, computed in the G3 GPU build pass).
- 🔜 Per-voxel hash colour-variance; blue-noise sample textures for AO taps (both marginal — ReSTIR already low-discrepancy).

### ⛔ Verified rejections (do NOT pursue)
NanoVDB/OpenVDB/GVDB or SVDAG/64-tree as a **live trace** structure (we're HW-RT AABB-BLAS; SVDAG stays B3 disk
transport); NRC (tensor-core lock-in vs SHARC, our vendor-neutral sibling); DDGI / Radiance Cascades (dominated by
ReSTIR + world-cache for 3D voxels); RTX Mega Geometry / CLAS (triangle-cluster-only — N/A to procedural AABBs);
coverage-threshold (≥50 %) occupancy (drops sub-voxel-thin geometry); Atomontage server-side streaming (out of scope).

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

## Now committed (formerly "deferred / independent") — see the `## Committed roadmap` above
The 2026-06 gap audit + user decision moved these from "deferred" to COMMITTED; they live in the roadmap now:
- **GI 3.0** (ReSTIR DI + light-tiles/ReGIR + the double-count fix + thin-wall cap + non-DLSS denoiser + specular) → roadmap §GI 3.0.
- **Demand / ray-guided residency + LRU** → roadmap §Streaming (S-lru). The G4 GPU pool is its substrate.
- **R4 occupancy-mask** → roadmap §Storage tail (still post-flip-histogram-decided).
- **Worldgen `classify` + coarse-LOD SSOT unification** → roadmap §Phase G (G6, GPU 3D-occupancy classify).
- **In-`dda_brick` empty-space skip** → roadmap §Storage tail (AS-c, measure-first).
- **Multi-instance FEATURE work** (`.vox`-as-instances, off-axis rotation, per-instance COW destruction) — the only item STILL genuinely deferred; the A3 descriptor makes it a no-second-rewrite addition. *(VOXEL_INSTANCING Phases 2–6)*

## Doc-accuracy fixes (apply with the next commit)
- `VOXEL_STORAGE_PLAN`: R4 (per-brick byte compression) ≠ surface-only residency (whole-brick cull) — they're orthogonal, currently conflated; R3's surface-brick dedup expectation is optimistic (halo variation limits hits to interior/strata).
- `SOTA_REFERENCE`: ReSTIR DI is listed ADOPTED but is NOT implemented (no DI pass); specular GI is silently deferred (mark it); the always-on interior floodfill is DONE (listed as "to port").
- Note in `VOXEL_LARGE_SCENE`: surface-only residency IS implemented (`source.rs` `classify` + `brickmap.rs` `is_full`), and the `RepackDelta` upload is built-but-unwired (A1).

## Status
R1 ✓ · **R2 ✓** (R2b landed, commit `d17f30d2`: `GpuBrickMeta` 48 B with `palette_base` + `index_bits` packed in `lod`; `voxels` is now the bit-packed index stream + a `brick_palettes` buffer at group0/binding12; SSOT decode = `GpuBrickPatch::cell_block`. **10.7× resident VRAM** on the worldgen slice, 41→3.8 MB; GPU oracle byte/pixel-identical) · R3 ✓ · surface-only `classify` ✓.

**A1 ✓ (LANDED) — the O(changed) GPU upload is WIRED.** Chose representation **(a)/A1-β**: a RAW fixed-block voxel arena (`SnapshotBuffers`) + an `index_bits == 0 ⇒ raw` decode branch in `cell_block` (CPU SSOT + WGSL mirror); the static `pack_brickmap`/`pack_resident_set` path keeps R2b (`index_bits >= 1`). `VoxelRtPatch` now carries a `VoxelRtUpload::{Snapshot, StreamSnapshot, Delta}` enum + a scene `epoch`. A streamed epoch ships ONE `StreamSnapshot` (the render path allocates the fixed-cap meta/aabb/voxel/palette buffers ONCE with `COPY_DST` + builds the BLAS over `capacity`, degenerate AABBs for free slots), then every later move ships a `Delta` (the render path `queue_write_buffer`s ONLY the changed slots — meta@`slot·48`, aabb@`slot·32`, the raw 4 KB block@`voxel_word_offset·4` — and rebuilds the BLAS in place ONLY on `topology_changed`). NEE lights ship whole per generation (free when the registry has no emitters). **Perf (worldgen slice, 10k resident):** per-move re-pack **1553 ms → 6.8 ms (228×)**; per-move GPU upload **~1.25 MB = 0.51% of the 245 MB fixed-cap arena the old path re-created every move**; the full per-generation BLAS rebuild is gone on non-topology moves. Tradeoff (documented): A1-β's raw arena gives up R2b's voxel-VRAM win on the STREAMED path (each dense block is a raw 4 KB `10³` grid, not a bit-packed stream) — recovered later via A4.4 (persistent interner). Gates green: `voxel_raytrace_gpu` oracle + the full GPU suite + the new `delta_upload_matches_snapshot_buffers_over_sequence` / `raw_arena_decodes_same_logical_cells_as_r2b` byte-identity tests + zero-warning default/editor build + clippy.

**A2 ✓ · A3 ✓ · A4.1 ✓ · A4.2 ✓** (landed — see git log: cap-after-classify; per-chunk BLAS + multi-instance TLAS + dirty-chunk rebuild; retire the bit-31 uniform flag; per-LOD AABB eps).

**A4.4 ✓ (LANDED — Phase A COMPLETE) — the streamed arena is now R2b PALETTED, recovering the VRAM A1-β traded away.** Replaced the raw fixed-block arena with **size-class SLABS** via a generic `SlabArena` (per-class free-list + bump + grow-on-overflow), used as ONE SSOT for BOTH the index-stream arena (classes `{32,63,125,250,500}` words keyed by `index_bits`) AND the per-brick palette arena (power-of-2 ladder `{2..65536}`, variable — Checkpoint-2). `SnapshotBuffers`/`ChangedSlot` carry paletted index + palette blocks; the streamed metas now carry real `index_bits ≥ 1` + a variable `palette_base`, so the shader uses the EXISTING paletted `cell_block` decode (ZERO shader change; the raw `index_bits==0` branch is now streamed-unused). A grow in either arena forces a re-snapshot. **Measured (worldgen slice, 10k resident): resident streamed VRAM 245 MB → 11.6 MB (21.1×; index slabs 6.4 + palette 0.41 + meta/aabb 4.8); per-move upload ~140 KB; incremental re-pack 10.3 ms (153× the full pack).** The palette-slab win scales further on high-registry `.vox` scenes (Checkpoint-1's fixed palette alone would reserve ~61 MB for a 256-id registry). Gates green: `voxel_raytrace_gpu` oracle + the streamed byte-identity gates (`delta_upload_matches_snapshot_buffers_over_sequence`, `streamed_snapshot_decodes_same_logical_cells_as_r2b`) + the GPU suite (streaming/gi/seam/world_cache/gallery/sponza) + zero-warning default/editor build + clippy `--all-targets`.

**Phase B ✓ (DONE)** — `.vxo` format (B-i `517ed90` + review fix `45a0461`) + region-streamed `VxoSource` loader
(B-ii `731e2601` + coarse-LOD fix `1b0bf11`), each specialist→3-reviewer-panel→fix→verify. B3 SVDAG deferred to the
roadmap. **`.vox→.vxo` conversion ✓** (`afaacde5`: Sponza/Sibenik/Conference @0.05 m; Bistro→C1). **Phase C: C2
emissive ✓ (`72260ff`), C3 CIELAB+area-avg ✓ (`f2f871f` + review fix `cd889ed`); C1 tiled voxelizer remains** (in
the roadmap). **Phase D1 (early partial flip, user-chosen ahead of C1):** D1a `VOXEL_SIZE`→0.05 + `clip_half`→160
(64 m reach) + scale re-pin ✓ (`8ee9591`) + GI-blocker fix (production world-cache cell/bias made BRICK_WORLD_SIZE-
relative) ✓ (`def8e62`); D1c benchmark ✓ (`cceec457`) — found the 64 m reach was *fiction* (O(H³) cube enumeration
hit the 8 M ceiling, LOD0-only, ~38 s/crossing); **D1d shell-first O(H²) enumeration ✓ (`84b36112`, panel-verified
AIRTIGHT)** — `surface_bricks_in` + the exact `surface_by_band` SSOT + skip-classify for worldgen; restores ALL 8
coarse LODs at clip_half 160; cold `update` **38 s → 2.97 s (~13×)**, the residual ~3 s is the A2 distance-cap SORT
over 6.7 M candidates (a next-lever: `select_nth` partial-select — Phase G's GPU enumeration eliminates it entirely).
The early partial flip arc (D1a/c/d) is DONE + verified; worldgen + legacy scenes render at 0.05 m / 64 m reach.
D2 screen-error LOD remains (roadmap).

**REFOCUS (user 2026-06): worldgen SHELVED; priority = load the large classic scenes INTO THE WORLD; render/stream
techniques are SHARED with terrain (see `refocus-large-scene-load` memory).** Landed since: **G0** cap-select
`select_nth` (`07d5aeaf`, cold update 2.97→1.56 s); **streamed `.vxo` gallery wired live** (`0bc9262`,
Sponza/Sibenik/Conference into the world via `MergedSource`, bounded-RAM, replacing the full-RAM `.vox` path);
**C1 tiled out-of-core voxelizer** (`5f8605b2`, panel-verified ORACLE AIRTIGHT via a 2000-case differential fuzz)
+ **`bistro.vxo` produced** (13.17 B cells, 354 M solids, 82.7 MB, <4 GiB bake) + added to the gallery →
**the classic-scene corpus (Sponza/Sibenik/Conference/Bistro) is now complete + streamable in the world.**

**G2-pre LODS bake ✓ + the live-freeze TRACE (2026-06-16) — "streams in slowly" is now MEASURED.** The gallery
load stutter has been instrumented (`info_span!` on the voxel-RT systems, `c76838c`) and traced on the live
editor. Two of the three load costs are now FIXED, and the third is pinned:
- ✅ **Coarse-LOD source** — was the headless cold-load OOM (demand-downsample, ~64,000 s / 26 GB). FIXED by
  baking the coarse pyramid into the `.vxo` `LODS` chunk: format+writer+reader+read-path (Stages 1/2/0,
  `a3d2f6a`/`6d42588`/`947f09c9`) + **all 4 corpus scenes re-baked with LODS incl. Bistro @354 M voxels via the
  tiled path** (`14d8738f`). Live `vox_drain_source` now **23 ms** (was the freeze).
- ✅ **Grow-snapshots** — slab-arena growth forced ~200 ms full re-snapshots mid-load. FIXED by pre-sizing the
  arena to `max_resident_bricks` (`5aa64281`): grow-snapshots **6→0**, `vox_pack_snapshot` gone from the hotspots.
- ✅ **Off-origin coarse dispatch** — Sibenik/Conference lost far-LOD geometry (LOD0-vs-coarse coord mismatch).
  FIXED (`466d857d`, `offset_at_lod`/`lod_bounds` ÷2^L).
- 🔜 **THE REMAINING LIVE FREEZE = `vox_pack_update`** (the CPU `ResidentPacker` incremental pack of the streaming
  shell): **3.4 s/load, ~282 ms/call ×12** (the cold-load batch pack — A1/A4.4's O(changed) win is steady-state;
  the cold fill still pays O(shell)), + `vox_blas_delta` 668 ms. This is the Phase-G target. **Near-term #146
  Tier 2** = rayon two-phase `pack_one` split (Phase 1 `par_iter().map(pack_one)` pure → Phase 2 serial
  `emit_changed_slot`, byte-identical) — cuts ~282 ms/call ~Ncores× WITHOUT the full GPU pivot. **Structural
  #146 Tier 3 / Phase G** = readback-free GPU pack + BLAS (also kills `vox_blas_delta`).

**NEXT (parallel tracks, trace-driven):**
- **#146 Tier 2 — rayon `pack_one`** (runtime, `incremental.rs`/`gpu.rs`): the immediate live-freeze fix.
- **#144 — constant-RAM bake Stages 1-3** (offline, `voxelize_scene.rs`/`writer.rs`): disk-spill base + windowed
  coarse + flat-RSS gate + the scratch-location robustness fix (the bake defaulted scratch to the system drive C:
  and filled it — `voxelize_scene.rs:232`). The user's memory-agnostic-bake directive; also a Phase-G prerequisite.
- THEN **Phase G** (GPU pack/enumeration/compaction — supersedes Tier 2's CPU pack + `vox_blas_delta`) and **D2**
  screen-error LOD. Worldgen (parked 3D-density-field decision) follows.

**R2b reconciliation for A1 (IMPORTANT — the design doc predates R2b's final state):** R2b **removed `snapshot_buffers`** (the raw fixed-block arena) and made the per-brick voxel payload a **VARIABLE-size index stream** (`index_bits·1000/32` words: 32 for 1-bit … 500 for 16-bit). So A1's fixed-capacity O(changed) upload can no longer assume a 1000-u32 fixed block per slot. A1 must choose: **(a)** re-add a RAW fixed-block path + an `index_bits==0 ⇒ raw` shader decode branch (the doc's "A1-β" — keeps the fixed-block free-list + O(changed) `queue_write_buffer`, at the cost of R2's VRAM win on the *streamed* path; recover it later via the persistent-interner A4.4), or **(b)** a variable-size paletted index arena (keeps R2 VRAM, needs a non-fixed-block allocator). Recommend (a) first. The A1 agent reconciles against the post-R2b code, not the pre-R2b doc text.
