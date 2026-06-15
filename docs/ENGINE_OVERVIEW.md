# Voxel-RT Engine — Overview (single source of truth)

> **This is THE entry point.** Read this first. It states the architecture as it actually is today,
> the current goal + active plan with status, and a map of every other doc (which is authoritative,
> which is superseded). When a detail doc and this overview disagree on *status*, this overview wins;
> when they disagree on *mechanism*, the linked detail doc owns the mechanism — fix the contradiction.
>
> Worktree: `voxel-rt`. Target: **Bevy 0.19 + forked wgpu-trunk, RTX 4090 / Vulkan.**
> Code: `src/voxel/{brickmap,gpu,streaming,raytrace,voxelize,vox,edits,palette,source}.rs`,
> `assets/shaders/voxel_raytrace.wgsl`, `examples/voxelize_scene.rs`,
> perf rig `tests/voxel_worldgen_perf.rs` (+ `voxel_sponza_pack`/`voxel_sponza_residency`).

---

## 1. Architecture (the pipeline, end to end)

A **SOTA hardware-ray-traced cubic-voxel engine** (Teardown-successor class). The world is a sparse
brickmap of palette voxels; the live render path is **hardware ray tracing** (procedural-AABB BLAS +
in-shader 3D-DDA), with a custom GI stack and DLSS Ray Reconstruction.

**Geometry / storage**
- A **brick** = `8³ = 512` voxels. Each voxel is a `BlockId` (`u16`) into a palette.
- `BrickStorage = Uniform(BlockId) | Dense(Box<[BlockId; 512]>)` on the CPU (`brickmap.rs`) — a
  uniform-brick fast path collapses an all-identical brick to one id + a `[u64; 8]` occupancy mask.
- Bricks are **haloed** to `10³` cells on the GPU (`halo_edge = lod_edge + 2`) — the cross-brick
  seam/normal fix (`gpu.rs`).
- A brick's world span scales with LOD: `brick_span(L) = BRICK_WORLD_SIZE · 2^L`.

**Residency / streaming**
- A **nested geometry-clipmap** of bricks around the camera (`streaming.rs`, `desired_clipmap`):
  LOD0 fills the inner cube; each coarser LOD is a Chebyshev shell. Per-single-brick-move streaming
  cost is O(shell), not a dense cold-fill. All-air bricks are dropped + memoized (sky is free).
- `ResidencyManager` is the SSOT for "what solid bricks exist"; `pack_resident_set` (`gpu.rs`) is the
  SSOT for "what the GPU traces" — it emits one AABB + one `GpuBrickMeta` (+ voxel data) per brick.

**Render (HW-RT)**
- `prepare_voxel_rt` (`raytrace.rs`) builds **one BLAS** over every resident brick's procedural AABB
  and a **single-instance identity TLAS**. A ray hits a brick AABB, then `dda_brick`
  (`voxel_raytrace.wgsl`) runs an **in-shader 3D-DDA** over the brick's cells, keeping the nearest
  first-solid hit. **The DDA must stay O(1)** (`id = voxels[offset + cell_index]`, no pointer chasing)
  — this is the hard constraint that classifies every storage method (Tier A traceable vs Tier B disk).
- The ray stops at the first surface, so a fully-enclosed (all-neighbours-solid) brick can never be
  hit — the structural basis for surface-only residency.

**GI**
- Custom **world-space ReSTIR + SHARC-style world-radiance-cache** (keyed on world position + normal),
  single-bounce + emissive voxel lights + temporal accumulation, resolved through **DLSS-RR**.
  Edits **adapt GI locally** — never full-clear the reservoirs / cache / DLSS history.

**Key constants (today)**
| Constant | Value | Where |
|---|---|---|
| `VOXEL_SIZE` (LOD0 edge) | **0.2 m** (migrating to 0.05 m — §2) | `brickmap.rs` |
| `BRICK_WORLD_SIZE` | **1.6 m** (= `VOXEL_SIZE · 8`) | `brickmap.rs` |
| brick dims | `8³` voxels, haloed to `10³` on GPU | `brickmap.rs` / `gpu.rs` |
| `MAX_LOD` | **7** | `streaming.rs` |
| `clip_half_bricks` | 8 | `StreamingConfig` |
| `max_resident_bricks` | ~60 000 | `StreamingConfig` |
| `GpuBrickMeta` / `GpuBrickAabb` | 32 B each | `gpu.rs` |

---

## 2. Current goal + active plan

### Goal
Load **all four classic demo scenes — Sponza, Sibenik, Conference, Bistro — into ONE world brick map**
(the **MERGE path**, *not* per-object instances), at a **fine 0.05 m LOD0 voxel size**, handled
correctly by SOTA storage. Architectural detail (Bistro signage/railings, Sponza relief) needs ~5 cm
voxels; 0.2 m is visibly coarse. Because scenes load *into the world brick map*, a scene's detail is
the world's detail → finer scenes ⇒ a finer **global** `VOXEL_SIZE`.

### Active plan (two coupled docs)
The **storage stack** (`VOXEL_STORAGE_PLAN.md`, R1–R5) feeds the **0.05 m migration**
(`VOXEL_FINE_RESOLUTION_PLAN.md`, S1–S4). 0.05 m is 64× more voxels per scene (cube law), so storage
**must** come first; the `VOXEL_SIZE` flip is the **last** step, not the first.

| Step | What | Doc | Status |
|---|---|---|---|
| **R1** | Uniform-brick collapse **into VRAM** (4 KB → ~8 B per buried brick) | STORAGE §R1 | ✅ **landed** |
| **R3** | Brick-level dedup (identical bricks share one slice; COW on edit) | STORAGE §R3 | ✅ **landed** (`719fab4`) |
| **R2a** | Palette **encode/decode** (CPU side of per-brick palette) | STORAGE §R2 | ✅ **landed** |
| **R2b** | Paletted **GPU buffers + shader bit-extract decode** in `dda_brick` | STORAGE §R2 / FINE §S1 | 🔄 **in progress** |
| **R5** | Native `.vxo` format — disk compression + **streamed** load (no full-RAM expand) | STORAGE §R5 / FINE §S2 / INSTANCING §1.5 | ⏭️ **next** |
| **Flip** | `VOXEL_SIZE` 0.2 → 0.05 (atomic) + re-bake Sibenik/Conference/Bistro to `.vxo` (Sponza already 0.05 m) | FINE §S3 | ⏭️ after R5 |
| **Reach/LOD** | re-tune `clip_half`/`MAX_LOD` so the fine band reaches far at bounded VRAM | FINE §S4 / LARGE-SCENE | ⏭️ last |

**Surface-only residency** (`VOXEL_LARGE_SCENE_PLAN.md`, Phase A) is the orthogonal residency axis that
makes the fine/far view affordable: cull provably-unreachable enclosed bricks from the BLAS, turning the
resident set from Θ(H³) → Θ(H²). It builds on R1–R3 and is the prerequisite for ~10× view reach at
bounded VRAM. (Demand/ray-guided residency = its Phase B, deferred.)

### Paused / deferred (do NOT present as active)
- **GPU-driven worldgen pivot** (`GPU_VOXEL_WORLDGEN_PLAN.md`) — Stage-1a `NodeKind→WGSL` codegen + GPU
  voxelize parity **landed**, then **PAUSED** per user redirect to focus on scene-loading. The fixed-cap
  GPU brick pool it specifies remains the substrate the storage/residency plans target. Stage 4 (CPU
  clipmap LOD aggregation) is already landed and lives in the active path.
- **Per-object instancing / multi-instance TLAS** (`VOXEL_INSTANCING_PLAN.md`) — fully designed,
  **DEFERRED**. Scenes load into the world brick map (merge path) for now. Its `.vxo` chunk spec (§1.5)
  and COW rule (§2.3 = storage R3) are reused by the active plan even though instancing itself is parked.

---

## 3. Doc map (authoritative source per topic)

| Doc | Purpose / owns | Status |
|---|---|---|
| **`ENGINE_OVERVIEW.md`** (this) | Master entry point: architecture, current goal, doc map | **active — master** |
| `VOXEL_FINE_RESOLUTION_PLAN.md` | The 0.05 m migration program (S1–S4); ties storage to the target; gating order | **active — plan of record** |
| `VOXEL_STORAGE_PLAN.md` | Per-brick compression R1–R5 (uniform-collapse, palette, dedup, occupancy, `.vxo`); Tier-A/Tier-B discipline | **active — owns storage** |
| `VOXEL_LARGE_SCENE_PLAN.md` | Residency/streaming axis: surface-only (Θ(H³)→Θ(H²)) + demand/ray-guided + screen-error LOD | **active — owns residency** |
| `VOXEL_INSTANCING_PLAN.md` | `.vox`/`.vxo` import, instancing, nested/off-axis sub-grids, per-instance destruction; owns the `.vxo` chunk spec (§1.5) | **paused/deferred** (`.vxo` §1.5 + COW §2.3 still authoritative for the active plan) |
| `GPU_VOXEL_WORLDGEN_PLAN.md` | GPU-driven worldgen + streaming + fixed-cap GPU brick pool (Stages 0–5) | **paused** (Stage-1a landed; Stage-4 CPU clipmap landed; rest parked) |
| `REFERENCES.md` | Subsystem → open-source/paper reference index + local checkout paths | **active — reference index** |
| `PERF_ROADMAP.md` | Pre-pivot **SDF-renderer** perf backlog (chrome-trace audit, 2026-06-02) | **superseded** (pre-voxel-RT pivot) |
| `MESH_BAKE_PLAN.md` | SDF → chunked-mesh-bake implementation plan | **superseded** (mesh-bake pivot, itself superseded by voxel-RT) |
| `MESH_BAKE_RESEARCH.md` | SDF → mesh-bake decision/research record | **superseded** (same) |
| `DDGI_SCALING.md` | SDF DDGI scaling research/ledger | **superseded** (DDGI dropped; GI is now ReSTIR + world cache) |
| `GPU_PREVIEW_RAYMARCH_PLAN.md` | GPU heightfield-preview raymarch for the **editor node graph** (Bevy 0.18 era) | **superseded/orphan** (SDF-worldgen editor tooling) |
| `WORLD_GEN_PLAN.md` | LayerProcGen-style procedural worldgen (SDF clipmap era) | **superseded** (worldgen now §GPU_VOXEL_WORLDGEN, itself paused) |
| `TERRAIN_MATERIALS_PLAN.md` | Volumetric biome strata / destruction-aware terrain materials (SDF era) | **superseded** (concept survives as voxel strata; doc is pre-pivot) |
| `BIOME_SHAPE_REGISTRY_PLAN.md` | Per-biome terrain-shape node graphs (SDF worldgen Phase 2) | **superseded** (SDF worldgen era) |
| `reference/*.txt` | Saved Shadertoy 3D Radiance Cascades reference buffers | **active — raw reference asset** |

> Note: `soft-coalescing-dolphin.md` is **referenced** by `VOXEL_INSTANCING_PLAN.md` (its "Phase 3" =
> per-chunk BLAS + multi-instance TLAS) but **does not exist in this worktree**. Treat that content as
> folded into `GPU_VOXEL_WORLDGEN_PLAN.md` Stage 3 (per-chunk BLAS) + `VOXEL_INSTANCING_PLAN.md` §2.4.

---

## 4. Pruning / consolidation recommendation

Goal: a clean hierarchy under this master — one active branch (the voxel-RT storage/residency/migration
stack) and a clearly-quarantined legacy set, with no dangling cross-references.

**Keep as active details (the live hierarchy under this overview):**
- `VOXEL_FINE_RESOLUTION_PLAN.md` — **keep** (plan of record; this overview's §2 links it).
- `VOXEL_STORAGE_PLAN.md` — **keep** (owns storage R1–R5).
- `VOXEL_LARGE_SCENE_PLAN.md` — **keep** (owns residency).
- `REFERENCES.md` — **keep** (already voxel-RT-aligned reference index).
- `reference/*.txt` — **keep** (raw GI reference assets).

**Keep but mark status (active concept, parked program):**
- `VOXEL_INSTANCING_PLAN.md` — **keep, mark DEFERRED** at the top; add a one-line pointer that §1.5
  (`.vxo`) + §2.3 (COW) are the authoritative specs the *active* storage plan consumes (so a reader
  isn't misled that instancing is being built now).
- `GPU_VOXEL_WORLDGEN_PLAN.md` — **keep, mark PAUSED** at the top (note Stage-1a + Stage-4 landed, rest
  parked); it's the GPU-pool substrate spec the storage/residency plans reference.

**Mark superseded (pre-pivot SDF / mesh-bake / terrain era — keep for history, banner at top):**
- `PERF_ROADMAP.md` → **mark SUPERSEDED** (SDF-renderer perf backlog; replaced by the per-stage
  `voxel_worldgen_perf` benchmark gate).
- `MESH_BAKE_PLAN.md` → **mark SUPERSEDED** (mesh-bake pivot abandoned for voxel-RT).
- `MESH_BAKE_RESEARCH.md` → **mark SUPERSEDED**; optionally **merge** its still-relevant citations into
  `REFERENCES.md`, then archive.
- `DDGI_SCALING.md` → **mark SUPERSEDED** (DDGI replaced by ReSTIR + world cache).
- `GPU_PREVIEW_RAYMARCH_PLAN.md` → **mark SUPERSEDED/ORPHAN** (Bevy-0.18 SDF-editor tooling).
- `WORLD_GEN_PLAN.md` → **mark SUPERSEDED** (worldgen now lives in the paused GPU worldgen plan).
- `TERRAIN_MATERIALS_PLAN.md` → **mark SUPERSEDED** (the volumetric-strata concept carries forward into
  voxel worldgen, but this doc is SDF-era; fold the surviving idea into the worldgen plan if revived).
- `BIOME_SHAPE_REGISTRY_PLAN.md` → **mark SUPERSEDED** (SDF worldgen Phase 2).

**Suggested physical layout (optional, for the agent who later does the move):** create
`docs/archive/` and move the eight superseded docs there with a one-line "superseded by ENGINE_OVERVIEW;
pre-voxel-RT-pivot" banner — leaving the top-level `docs/` to exactly this overview + the five active
voxel docs + `REFERENCES.md` + `reference/`. (Not done here: this overview only adds itself; the moves
are a separate, conflict-free follow-up.)

**Cross-reference cleanup:** the dangling `soft-coalescing-dolphin.md` reference in
`VOXEL_INSTANCING_PLAN.md` should be repointed to `GPU_VOXEL_WORLDGEN_PLAN.md` Stage 3 (per-chunk BLAS)
when that doc is next touched.
