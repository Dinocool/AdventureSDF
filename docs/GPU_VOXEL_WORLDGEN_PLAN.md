# GPU-Driven Voxel Worldgen + Streaming — plan of record

## Decision (user-directed, 2026-06)
Pivot the worldgen/streaming hot path from CPU to the GPU — the SOTA "GPU-driven voxel" model — while
keeping the existing HW-RT + SHARC-style world-cache + ReSTIR + DLSS-RR GI **unchanged**. Chosen over an
incremental-CPU clipmap fix because it is the only path to "extremely fast generation + near-unbounded view
distance" the field actually achieves (Aokana, gvox_engine, GigaVoxels, Bonsai).

## 2026-06 REFRAME (post Phase A/B/C/D1 — read this first; it updates the bottleneck framing below)
Phases A (O(changed) upload + per-chunk BLAS + paletted slabs), B (`.vxo` streamed loader), the corpus
conversion, C2/C3, and **D1 (the 0.05 m flip + 64 m reach + shell-first O(H²) enumeration)** have all LANDED.
This changes WHICH part is the wall and re-orders the GPU work:
- The old "58–103 ms hitch = CPU voxelize + full re-pack/re-upload" is **mostly fixed CPU-side**: A1 killed the
  re-pack/upload (O(changed), ~140 KB/move); the per-brick voxelize is a **bounded drain** (D1c: 1.04 s cold-fill
  over 874 frames, max 7.95 ms/frame — fine).
- **The residual wall is the residency DECISION (`update`): the per-crossing ENUMERATE → classify → SORT.** D1d
  shell-first cut it 38 s → 2.97 s, and the remaining ~3 s is **the A2 distance-cap SORT over ~6.7 M candidates**
  (keep the nearest 400 k). User-confirmed 2026-06: "looks correct but pretty slow/unperformant."
- **USER DECISION: do the full GPU-driven pivot regardless of perf — it is the CORRECT architecture, not
  perf-gated ([[feedback-plan-to-best-practice]]).**
- **NEW HEADLINE (the SOTA-gap audit's convergent #1 — two independent agents): the fully-GPU, readback-free
  residency ENUMERATION + COMPACTION** (workgroup prefix-sum stream compaction + atomic sparse active-brick list +
  GPU-written indirect dispatch — the **re-flora `contree/` build** + gvox_engine GPU allocator are working
  references at `D:\tmp_test\re-flora`). This subsumes the cap-sort and the CPU enumeration entirely. The plan's
  "CPU keeps only a coarse residency structure" is the INTERIM; the END STATE moves the enumeration/compaction to
  GPU too (CPU keeps only the camera + clipmap params). The full readback-free pipeline = GPU classify/enumerate →
  prefix-sum compact → GPU voxelize → write pool slots → fill AABBs, all in one submission, zero hot-path readback.
- **Note: `MAX_LOD = 7`** now (the staging text below says 6 in places — stale; D1 set the 0.05 m / 64 m-reach
  clipmap). The staging order is updated: **G0 instrument + the cheap cap-sort `select_nth` win (immediate
  relief) → G1 GPU voxelize parity (the foundational de-risk, the pool needs it) → G2 GPU pool + GPU
  enumeration/compaction (the readback-free pipeline, the headline) → G3 per-chunk BLAS → G6 3D-occupancy
  enumeration → G7 GPU edit path.** See `VOXEL_PROGRAM.md` §"Committed roadmap" Phase G for the canonical list.

## Why (research synthesis — see also the survey in chat 2026-06)
The render + GI stack is already SOTA-aligned. **Every** modern large-world voxel engine diverges from us in
exactly one place — streaming — and that divergence is the literal cause of both reported bugs:
- **Movement hitch** = the CPU `update` schedule spiking **58–103 ms** on brick crossings (Chrome trace
  `trace-1781397412608464.json`); the GPU render schedule maxes **3.5 ms**. The hitch is 100% CPU
  voxelize + full resident-set re-pack/re-upload.
- **"View distance tiny"** = the flat 60k-brick residency cap (~38 m sphere), recomputed every move.

Field consensus we will adopt:
1. **Chunk the world** — many small acceleration structures in a top-level grid, never one monolith
   (Aokana SVDAG chunks; dubiousconst282 "many small trees"; gvox; GigaVoxels).
2. **CPU keeps only a coarse residency structure** (implicit octree / coord-map) that decides *which*
   chunks are resident; the **GPU** voxelizes + packs + writes AS data (Aokana, Bonsai, gvox).
3. **LOD by aggregation** of finer levels into coarser chunks → huge view distance at bounded VRAM
   (Aokana: 10 B voxels in ~400 MB / 6 ms, ~5% resident; Distant Horizons; GigaVoxels).
4. **Demand-driven residency** — async stream only the entering set; ideally ray-guided (GigaVoxels).
5. **GI is world-space and agnostic to how bricks arrive — do NOT change it to buy speed.**

## Engine facts that constrain the design (verified 2026-06)
- **AABB BLAS from a GPU buffer WORKS on our fork.** `raytrace.rs:1799` already calls
  `wgpu::BlasGeometries::AabbGeometries` with `aabb_buffer: &aabb_buf` (usage `BLAS_INPUT | STORAGE`);
  the buffer is read at build-execution time, so a prior compute dispatch in the same submission can fill
  it. (An Explore agent's "no AABB BLAS" claim was wrong — it inspected bevy_solari's *triangle* path.)
- **NO indirect AS build.** `primitive_count` / `max_instances` must be CPU-known at build time; there is
  no `build_acceleration_structures_indirect`. → use a **fixed-capacity brick pool**: always build
  `N_capacity` AABBs; the GPU writes real AABBs for resident slots and **degenerate/zero-extent AABBs for
  free slots**. No CPU readback of voxel data in the hot path. (A tiny high-water-mark readback is optional.)
- **The Field worldgen graph is WGSL-portable.** 20 pure-arithmetic + portable-noise opcodes (4 sources,
  8 unary, 5 binary, 1 ternary, + erosion + biome/material resolve), max 64 nodes (current graphs 9–15),
  no unbounded loops / recursion / CPU-only tables. A `NodeKind → WGSL` codegen pass keeps the node-graph
  as the **single SSOT** (one graph → CPU autodiff `eval_into` AND GPU WGSL). The hardest op is the
  monotone-cubic `Curve` spline (~30 lines WGSL).
- Current CPU path: `voxelize_brick` (voxelize.rs:116) = 8³=512 voxels/brick from 64 columns
  (one `ColumnSample::at` → `sample_world` per column); `pack_resident_set` (gpu.rs) builds the parallel
  AABB/meta/voxel(haloed)/palette buffers; `prepare_voxel_rt` (raytrace.rs:1740) re-allocs + re-uploads
  ALL of it and full-rebuilds the single BLAS on every `generation` bump. **This whole path is what moves
  to the GPU.** Note the **haloed brick** layout (`halo_edge = lod_edge+2`, gpu.rs) is the seam fix — the
  GPU voxelizer must reproduce the halo (neighbour boundary voxels, AIR where absent).

## Target architecture
"Aokana/gvox streaming + our SHARC/ReSTIR/DLSS-RR GI":
- **GPU brick pool** — fixed-capacity storage (voxels + meta + AABB + palette), with a GPU free-list
  allocator. Resident slot ↔ brick coord map. Free slots → degenerate AABBs.
- **GPU voxelization** — compute shader; `NodeKind→WGSL` codegen of the Field graph + erosion + biome +
  material resolve; writes voxels (haloed) + meta + AABB into the pool slot. Replaces CPU
  `voxelize_brick` + `pack_resident_set`.
- **CPU coarse residency** — an implicit octree / coord-map decides which chunk slots should be resident
  (clipmap rings around the camera); emits a small "fill these slots with (coord, lod)" command. Never
  touches voxel bytes. Replaces the O(region) `update()` recompute + the O(resident) re-pack.
- **Per-chunk BLAS + TLAS instances** (chunk = KxKxK bricks) so only changed chunks rebuild; the GPU
  builds changed chunks' AABB ranges. Replaces the single full-rebuild BLAS.
- **LOD by aggregation** + clipmap rings → view distance at bounded VRAM. Bricks store full-res; the
  GPU downsamples per `want_lod` at pack time (already the CPU model — port it).
- **GI unchanged** — the world cache + ReSTIR consume the GPU-resident pool exactly as today.

## Staged execution (each stage: specialist implementer → ≥2 adversarial reviewers → benchmark gate → commit)
Run via the `Workflow` tool. Every stage is independently shippable, benchmarked before/after with the
headless `tests/voxel_worldgen_perf.rs` rig (extend per stage), and must keep all 3 feature builds
zero-warning + the GPU/headless suite green.

- **Stage 0 — instrument + quick wins (no architecture change; ship immediately).**
  - Add the `debug_view` branch to `restir_dlss_p2` (DLSS path currently ignores it → "debug views stopped
    working"). Write the debug colour to `out_tex` + valid DLSS guides so RR passes it through.
  - Add **LOD debug view** (`debug_view == 7`): `trace` carries the hit brick's `lod`; colour by LOD ring
    (the instrument for validating Stages 3–4). Editor panel option + the existing panel drives it.
  - Confirm the FPS-camera worldgen reframe fix (uncommitted in `mod.rs`) builds; commit it.
  - Verify: `cargo check` (the user's running app holds `bevy_dylib.dll` → no relink), then user rebuilds.

- **Stage 1 — `NodeKind → WGSL` worldgen codegen + GPU voxelize compute (CORRECTNESS, not yet live).**
  Codegen the graph to WGSL; a compute shader voxelizes a brick (haloed) into a scratch buffer. Headless
  GPU-vs-CPU parity test: GPU brick == `voxelize_brick` bit-for-bit (or within a pinned tolerance) across
  a sample of coords/LODs/biomes. Live path UNCHANGED. This de-risks the single biggest piece first.

- **Stage 2 — GPU brick pool + allocator; GPU writes AABB/meta/voxel/palette into pool slots.** Fixed
  capacity, degenerate AABBs for free slots, BLAS built over the pool. CPU still decides residency but
  hands the GPU a fill-list; the per-frame CPU pack/upload is deleted. Benchmark: per-brick-crossing CPU
  cost → ~0; the `update` hitch gone. A/B-gated behind a flag vs the CPU path until parity-confirmed.

- **Stage 3 — per-chunk BLAS + multi-instance TLAS + dirty-chunk rebuild.** Only changed chunks rebuild.
  Benchmark: BLAS cost scales with *edited/streamed* chunks, not world size. (Folds in the old Phase-3
  plan + the #94 streaming glitch revisit.)

- **Stage 4 — LOD aggregation + clipmap rings → view distance. [LANDED — CPU clipmap]** A brick is always
  `8³` voxels; its world span scales with LOD (`brick_span(L) = BRICK_WORLD_SIZE · 2^L`, `MAX_LOD = 6`). The
  voxelizer samples each `(coord, lod)` directly at its coarse spacing (a true in-place 3D mip). Residency is
  re-keyed by `(coord, lod)` and built as NESTED CLIPMAP SHELLS (`desired_clipmap`: LOD0 fills the inner
  cube; each coarser level a shell `clip_half/2 < cheby ≤ clip_half`). View radius 45 m → **819 m (18.3×)**
  at `clip_half_bricks = 8`; per-single-brick-move STREAMING cost is **O(shell)** (~6.7 ms / 786 bricks vs a
  138 ms dense cold-fill — 21×). Cross-LOD seams handled by the existing `BRICK_AABB_EPSILON` overlap +
  nearest-hit DDA (no cross-LOD halo). The remaining per-move re-pack is O(resident) — the BLAS-rebuild cost
  Stage 3 amortizes, not the clipmap stutter. GPU mixed-LOD oracle + seam/show-through GPU tests pass.

- **Stage 5 (optional endgame) — demand / ray-guided residency.** Rays request bricks via a feedback
  buffer (GigaVoxels); CPU coarse octree shrinks to a seed. Only if Stages 2–4 leave residual stalls.

## Preserved invariants (every reviewer checks)
GI untouched (world cache + ReSTIR energy/convergence parity; store-before-visibility; one spatial
neighbour/frame; **edits ADAPT locally, never full-clear** — first-frame/res-change only); the haloed-brick
seam fix reproduced exactly on the GPU; the brick-AABB epsilon overlap rule (`BRICK_AABB_EPSILON`); knobs as
uniforms; DLSS depth-jittered/motion-unjittered contract; zero warnings + all 3 feature builds; no
self-launch (runtime visual checks are the user's); never git-restore `world.graph.ron`.

## Verification (every stage)
`cargo build` + `--features editor` + `--features dlss` (zero warnings); `cargo clippy --features editor
--all-targets`; `cargo test --lib`; the GPU harnesses; the worldgen perf rig before/after with numbers in
the commit. `TMP/TEMP=D:\tmp_test`. Re-pin worldgen parity on any SSOT touch. Re-tag after each user visual
confirmation.

## References
Aokana (arXiv 2505.02017, 2025); gvox_engine (GabeRundlett); GigaVoxels (INRIA CNLE09); Voxagon
(Teardown-successor) 2024 year summary; dubiousconst282 sparse-64-tree (2024-10); NVIDIA SHARC / RTXGI 2.0;
NanoVDB (Museth 2021); Distant Horizons. Our code: `src/voxel/{raytrace,gpu,voxelize,streaming,brickmap}.rs`,
`src/sdf_render/worldgen/{graph,layers,biome}`, `assets/shaders/voxel_raytrace.wgsl`.
