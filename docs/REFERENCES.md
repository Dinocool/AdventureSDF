# Voxel Ray-Tracing Engine — References Index

A maintainable index mapping each subsystem of the HW-ray-traced cubic-voxel engine
(Teardown-successor style: sparse brickmap of 8³ palette voxels, per-brick procedural
AABB in a BLAS + in-shader 3D-DDA via inline WGSL `ray_query`, custom single-bounce GI
+ emissive voxel lights + temporal accumulation, NVIDIA DLSS Ray Reconstruction, on a
vendored Bevy 0.19 + forked wgpu-trunk; physics via `rapier3d`) to the open-source
projects and papers it draws on.

Each entry lists: **what we borrow → citation/link → local checkout path** (if cloned).

## Local checkouts

Reference repos are shallow-cloned (`git clone --depth 1`) outside this repo, under
`D:/refs/`. Vendored engine dependencies live elsewhere on disk and are referenced
in place (not re-cloned).

| Local path | Repo | License | Cloned size | Role |
|---|---|---|---|---|
| `D:/refs/BrickMap` | [stijnherfst/BrickMap](https://github.com/stijnherfst/BrickMap) | MIT | 38 MB | Brickmap structure + GPU streaming reference (CUDA) |
| `D:/refs/VoxelRT` | [dubiousconst282/VoxelRT](https://github.com/dubiousconst282/VoxelRT) | (see repo) | 190 MB | Accel-structure benchmark suite: MultiDDA / brickmap / sparse 64-tree, with `docs/VoxelNotes.md` write-up |
| `D:/refs/shocovox` | [davids91/shocovox](https://github.com/davids91/shocovox) | MIT/Apache-2.0 | 194 MB | WGSL + Rust GPU sparse-voxel ray-marcher (closest stack match; archived — see note) |
| `D:/refs/RadianceCascadesPaper` | [Raikiri/RadianceCascadesPaper](https://github.com/Raikiri/RadianceCascadesPaper) | CC-BY-ND | 275 MB | Radiance Cascades paper source (LaTeX + figures) |
| `D:/bevy-fork/crates/bevy_solari` | Vendored fork of `bevy_solari` (Bevy 0.19) | MIT/Apache-2.0 | — (vendored, not cloned) | ReSTIR DI/GI + world-cache + DLSS-RR resolve reference |
| `D:/wgpu-fork` | Vendored fork of `wgpu` (trunk) | MIT/Apache-2.0 | — (vendored, not cloned) | `ray_query` / procedural-AABB BLAS API + examples/tests |

Engine-internal saved references (already in this repo):

| In-repo path | What it is |
|---|---|
| `docs/reference/*.txt` | Saved Shadertoy 3D Radiance Cascades reference (BufferA–D, CubeA, Image, common — the canonical projective-visibility merge port) |

---

## Subsystem → reference map

### 1. Brickmap / sparse voxel storage (8³ palette voxels)

- **What we borrow:** the brickmap layout itself — a sparse top-level grid of bricks,
  each brick an 8³ block of voxels; linear index-into-arena instead of pointers; doubling
  arena on fill; per-brick occupancy + LOD; GPU-resident with CPU streaming of surface
  bricks on ray miss. Palette/indexed voxel colors per brick.
- **References:**
  - stijnherfst/BrickMap — realtime CUDA brickmap path tracer. Index-based superchunks
    (16³ bricks), 12-bit indices, streaming request buffer, 3 LOD levels.
    <https://github.com/stijnherfst/BrickMap> — **local: `D:/refs/BrickMap`**
    (`README.md`, `src/kernel.cu`, `src/Scene.cpp`).
  - Source thesis behind BrickMap: van der Glas / "A real-time GPU brick-map renderer"
    (Utrecht University). <https://studenttheses.uu.nl/handle/20.500.12932/20460> — link-only.
  - dubiousconst282/VoxelRT — `eXtendedBrickMap`: 3-level grid (sectors → bricks → voxels)
    with 4³ occupancy bitmasks; `MultiDDA`: 2-level brick grid. Benchmarks brickmap vs
    other accel structures. <https://github.com/dubiousconst282/VoxelRT> —
    **local: `D:/refs/VoxelRT`** (see `README.md` table + `docs/VoxelNotes.md`).

### 2. Per-brick procedural AABB in a BLAS + HW ray traversal

- **What we borrow:** rather than meshing voxels, register one **procedural AABB per
  brick** as a custom-geometry BLAS, let the HW BVH cull/sort brick hits, then run a
  fine 3D-DDA inside the intersection shader. This is the Teardown approach.
- **References:**
  - Teardown / Dennis Gustafsson — talks/posts on Teardown's HW-RT voxel renderer
    (per-object voxel volumes, RT for GI/reflections/shadows). Blog: <https://blog.voxagon.se/>
    — link-only.
  - `wgpu` `ray_query` + procedural-AABB acceleration structures — the actual API we
    build on. **Vendored: `D:/wgpu-fork`.** Key references in the fork:
    - `examples/features/src/ray_aabb_compute/` — procedural AABB BLAS + `ray_query` compute.
    - `examples/features/src/ray_cube_normals/`, `ray_scene/`, `ray_shadows/` — `ray_query` usage.
    - `tests/tests/wgpu-gpu/ray_tracing/as_aabb.rs`, `as_build.rs`, `as_create.rs` — AABB
      acceleration-structure build/use tests (authoritative for the API surface).
    - `naga/src/back/spv/ray/query.rs` — naga's `ray_query` SPIR-V backend.
  - dubiousconst282/VoxelRT — `StdBVH`: "binary BVH + DDA over 8³ brick leafs" — the same
    BVH-over-bricks + intra-brick-DDA pattern. **local: `D:/refs/VoxelRT`.**

### 3. In-shader 3D-DDA voxel traversal (intra-brick)

- **What we borrow:** the incremental DDA loop (compare per-axis `tMax`, step the axis
  that has progressed least), in WGSL, inside the brick AABB intersection.
- **References:**
  - **Amanatides & Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing"** (Eurographics
    1987) — the canonical grid-DDA. <http://www.cse.yorku.ca/~amana/research/grid.pdf>
    — link-only. Reference impl: <https://github.com/cgyurgyik/fast-voxel-traversal-algorithm>.
  - **dubiousconst282, "A guide to fast voxel ray tracing using sparse 64-trees"** (2024) —
    DDA pitfalls at scale, parametric vs incremental traversal, 4³/64-bit occupancy masks
    for space-skipping + LODs. <https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/>
    — write-up mirrored at **`D:/refs/VoxelRT/docs/VoxelNotes.md`** + sketch
    `D:/refs/VoxelRT/docs/sketches/dda_vs_parametric.glsl`.
  - **"Branchless Voxel Raycasting"** Shadertoy (fizzer / Xor lineage) — compact branchless
    DDA used as the WGSL loop template. <https://www.shadertoy.com/view/4dX3zl> — link-only
    (also cross-referenced from `D:/refs/VoxelRT/docs/VoxelNotes.md`).
  - shocovox — a worked WGSL voxel ray-marcher: `assets/shaders/viewport_render.wgsl`.
    <https://github.com/davids91/shocovox> — **local: `D:/refs/shocovox`.**
    > Note: shocovox is archived in favor of its successor
    > [VoxelHex](https://github.com/Ministry-of-Voxel-Affairs/VoxelHex) (not cloned; check
    > there for the maintained version if extending this reference).

### 4. Voxel surface normals

- **What we borrow:** cube-face normal from the **crossed axis** of the last DDA step
  (the axis whose plane the ray crossed = the face it hit) — the standard branchless-DDA
  normal. Occupancy-gradient normals considered as an alternative for smoothed shading.
- **References:**
  - Amanatides & Woo (above) — the stepped axis gives the entry face directly.
    <http://www.cse.yorku.ca/~amana/research/grid.pdf>
  - **"Branchless Voxel Raycasting"** Shadertoy — derives the face normal as
    `mask * -sign(rayDir)` from the step mask, the exact pattern we use.
    <https://www.shadertoy.com/view/4dX3zl> — link-only.

### 5. Global illumination (single-bounce + emissive voxel lights + temporal accumulation)

- **What we borrow:** our own single-bounce GI is informed by these algorithms; emissive
  voxels act as area lights; results are temporally accumulated/resampled. The
  ReSTIR/world-cache plumbing is reused near-verbatim from `bevy_solari`.
- **References:**
  - **ReSTIR DI** — Bitterli et al., "Spatiotemporal reservoir resampling for real-time
    ray tracing with dynamic direct lighting" (SIGGRAPH 2020).
    <https://research.nvidia.com/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct>
    — link-only. Impl reference: `D:/bevy-fork/crates/bevy_solari/src/realtime/restir_di.wgsl`.
  - **ReSTIR GI** — Ouyang et al., "ReSTIR GI: Path Resampling for Real-Time Path Tracing"
    (HPG/CGF 2021). <https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing>
    — link-only. Impl reference: `D:/bevy-fork/crates/bevy_solari/src/realtime/restir_gi.wgsl`,
    plus `world_cache_{query,update,compact}.wgsl`.
  - **DDGI** — Majercik, Guertin, Nowrouzezahrai, McGuire, "Dynamic Diffuse Global
    Illumination with Ray-Traced Irradiance Fields", JCGT 8(2), 2019.
    <https://jcgt.org/published/0008/02/01/> — link-only (alternate prior DDGI work in this
    repo's history under `docs/DDGI_*`).
  - **Radiance Cascades** — Alexander Sannikov. Paper source (unpublished, JCGT template):
    Raikiri/RadianceCascadesPaper — **local: `D:/refs/RadianceCascadesPaper`**
    (PDF at `out_latexmk2/RadianceCascades.pdf`, source `RadianceCascades.tex`).
    <https://github.com/Raikiri/RadianceCascadesPaper>. The 3D RC merge math we use is
    additionally captured in **`docs/reference/*.txt`** (saved Shadertoy port — canonical
    projective-visibility merge weight).

### 6. DLSS Ray Reconstruction (denoise + upscale)

- **What we borrow:** feed our noisy single-bounce GI + G-buffer (albedo, normal, depth,
  motion vectors) to DLSS-RR for denoise + super-resolution. Reuse the texture-resolve
  pass from `bevy_solari`.
- **References:**
  - **NVIDIA Streamline / DLSS Ray Reconstruction SDK** — guide texture layout + integration.
    <https://github.com/NVIDIA-RTX/Streamline> and
    <https://developer.nvidia.com/rtx/dlss> — link-only.
  - `bevy_solari` DLSS-RR resolve — `D:/bevy-fork/crates/bevy_solari/src/realtime/resolve_dlss_rr_textures.wgsl`
    (+ `gbuffer_utils.wgsl`); shows exactly which guide buffers RR expects. **Vendored.**

### 7. Physics (engine-agnostic)

- **What we borrow:** rigid-body + collision via `rapier3d` directly (not the Bevy
  plugin wrapper), so physics is decoupled from the renderer.
- **References:**
  - **rapier3d** — Dimforge. <https://rapier.rs/> / <https://github.com/dimforge/rapier>
    — link-only (consumed as a crates.io dependency, not cloned).

---

## Notes / status

- **Cloned locally (`D:/refs/`):** BrickMap, VoxelRT, shocovox, RadianceCascadesPaper.
- **Vendored, referenced in place (not cloned):** `bevy_solari` (`D:/bevy-fork/crates/bevy_solari`),
  `wgpu` fork (`D:/wgpu-fork`).
- **Link-only (papers / large or upstream repos not cloned):** Amanatides & Woo PDF,
  dubiousconst282 blog (write-up mirrored inside VoxelRT), the UU brickmap thesis,
  Teardown blog, ReSTIR DI/GI papers, DDGI (JCGT), Streamline/DLSS SDK, rapier3d.
- shocovox is archived; its maintained successor is VoxelHex (link-only above).
- `RadianceCascadesPaper` is the largest checkout (275 MB) due to LaTeX build artifacts
  under `out_latexmk2/`; kept because it contains the only public copy of the paper text.
