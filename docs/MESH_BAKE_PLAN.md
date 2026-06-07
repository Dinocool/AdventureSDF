# SDF → Chunked Mesh Bake — Implementation Plan

> Companion to [`MESH_BAKE_RESEARCH.md`](./MESH_BAKE_RESEARCH.md) (the cited decision record).
> **Decision recap:** Surface Nets via off-the-shelf [`fast-surface-nets`](https://github.com/bonsairobo/fast-surface-nets-rs);
> cross-LOD by **skirts**; **sharp edges not required**; **CPU-async** meshing; render through Bevy's
> standard PBR `Mesh3d` pipeline (off-the-shelf, and Solari-ready). GI: custom DDGI **disabled now**
> (`DdgiParams.intensity = 0.0`), to be replaced by **Bevy Solari** once meshes land.

## Key architectural enabler — no GPU readback

The SDF is fully evaluable on the **CPU** from the edit list, the same representation the bake uses:
- `edits::fold_csg(edits, pos) -> EditSample` (distance + material) — `src/sdf_render/edits.rs:600`
- `edits::fold_csg_dist_indexed(edits, indices, pos) -> f32` (BVH-culled, fast) — `edits.rs:695`
- `edits::build_palette_indexed(edits, indices, sample_points) -> Palette` — `edits.rs:748`
- The bake scheduler already publishes `Arc<Vec<ResolvedEdit>>` + `Arc<Bvh>` each frame and culls
  edit indices per brick (`bake_scheduler`, `atlas::cull_edit_indices`).

⇒ The CPU mesher samples the field directly per voxel — **no GPU→CPU readback**, sidestepping the
entire readback concern. Brick = `BRICK_EDGE`³ (8³ = 512 voxels, `atlas.rs:20`); `BrickKey(lod, coord)`;
`SdfAtlas` holds the resident/occupied brick set.

## Render integration — off-the-shelf first

Each baked chunk becomes a Bevy **`Mesh3d` + material entity** (standard PBR pipeline). This is the
maximally off-the-shelf choice: no custom render-graph node for primary visibility, frustum culling /
shadows / depth come for free, and it is exactly the geometry **Bevy Solari** will trace for GI. The
existing SDF raymarch render path stays untouched and gated behind a flag so we can A/B them until the
mesh path wins.

---

## Phase 0 — Spike & de-risk (≈1–2 days) ✅ gate before building out
**Goal:** prove the premise + see edge-rounding on *our* content before investing.
1. Add deps: `fast-surface-nets`, `ndshape` (do **not** hard-depend on `bevy-sculpter` — early/breaking;
   reference only).
2. One-shot spike system: pick one resident finest-LOD gallery brick; sample `fold_csg_dist_indexed`
   over a `(BRICK_EDGE + 2)³` padded grid → `surface_nets()` → build a Bevy `Mesh` (positions/normals/
   indices) at the brick's world origin → spawn `Mesh3d` + a plain `StandardMaterial`.
3. **Eyeball:** is Surface Nets edge-rounding acceptable on the gallery's CSG shapes? (If not → revisit
   the optional constrained-QEF lever from the research doc.)
4. **Profile (premise check):** rough raymarch primary-visibility cost vs. the projected mesh path on
   `gallery.scene` + `stress.scene`. Confirm the pivot is worth it.
- **Exit gate:** rounding acceptable AND mesh path promising → proceed. Else reconsider (Transvoxel /
  defer).

## Phase 1 — Single-LOD CPU mesh bake (finest resident shell)
**Goal:** the gallery rendering as real meshes, finest LOD only, behind a flag.
- New module `src/sdf_render/mesh_bake/mod.rs` + `MeshBakePlugin` (registered in `main.rs`; register any
  `Reflect` types — invariant #4).
- `MeshBakeEnabled(bool)` resource (default true in this worktree) to A/B against raymarch.
- Per resident **finest-LOD surface brick**: sample a `(BRICK_EDGE + 1 + pad)³` grid via
  `fold_csg_dist_indexed` using that brick's BVH-culled edit indices (reuse `cull_edit_indices`), run
  `surface_nets`, emit a `Mesh` translated to the brick origin. Same-LOD seams are free (1-voxel pad,
  no faces on positive boundary).
- **Async:** run meshing on `AsyncComputeTaskPool`, poll across frames (Bevy pattern: vx_bevy /
  bevy_voxel_world); spawn/replace one `Mesh3d` entity per chunk. Re-mesh **only** dirty chunks — hook
  the scheduler's existing dirty set / `edit_gen` so drags are incremental.
- Material v1: a single `StandardMaterial` (or vertex color) — defer real material mapping to Phase 2.
- **Done when:** gallery renders as meshes, edits re-mesh interactively, zero warnings, both build configs.

## Phase 2 — Material & shading fidelity
- **v1 DONE (2026-06-08):** per-vertex material COLOUR — `mesh_chunk` resolves the material at each
  vertex (`edits::fold_csg(...).material_id`), looks it up in a `MaterialRegistry` snapshot
  (linear base + emissive, passed into the off-thread bake), writes it to the vertex COLOUR shaded by a
  cheap fixed-direction hemispheric term (form reads while the mesh stays unlit) + emissive added so
  glowing materials are bright. A material-COLOUR edit re-bakes (the per-chunk content hash keys on
  material *id*, so an appearance-hash check bumps the rebake epoch). `mesh_test.scene` gains a floating
  `emissive_orange` orb to exercise it; the row already uses sand/cobble/red_metal/white_gloss.
- **Deferred to Phase 2b (needs a custom shader — held until meshes are primary so it's verified live):**
  carry the 4-id palette + per-corner material *weights* onto vertices, L1-normalize, **triplanar-splat**
  the PBR maps in a `MaterialExtension` over `StandardMaterial` (bonsairobo "Smooth Voxel Mapping"), and
  true PBR lighting (metallic/roughness response). v1 shows one dominant material per vertex (no
  multi-material blend within a chunk) and bakes a fixed shade instead of real lights.

## Phase 3 — Cross-LOD (clipmap rings) + skirts  *(satisfies the locked crack-free requirement)*
- Mesh **all** resident LOD rings, not just finest.
- Design-in invariants (cheap now, invasive later): **2:1 ring ratios**, **per-face coarser-neighbor
  flags** (6 bits/chunk), **neighbor-aware dirty propagation** (extend the incremental dirty/BVH-refit
  path), **retain border Hermite** for shading.
- **Skirts:** per chunk, drop boundary skirts only on faces that border a coarser ring; length ≈
  `k · voxel_size(lod)` — tune `k` until fine/coarse ring boundaries stop cracking (godot_voxel's
  caveat: too-short skirts leak). Skirts are the accepted "true crack-free" mechanism.
- (Optional later: geomorph the LOD transition to kill popping — only if skirts pop visibly.)

## Phase 4 — Make meshes primary & retire the raymarch/GI stack
- Once the mesh path matches/beats raymarch primary visibility: switch primary visibility to meshes,
  then **remove** the SDF raymarch G-buffer/cone nodes and the now-dead **DDGI/RC/surfel** passes
  (the `intensity = 0.0` soft-disable becomes a full deletion).
- Keep the SDF as the **authoring/editing** representation + the mesh bake source.
- **Adopt Bevy Solari** for GI (separate effort) — meshes make it possible. Caveat: Solari needs
  RT hardware (NVIDIA/DLSS-RR in practice) + wgpu raytracing; confirm target platform. Baked irradiance
  volumes (`bevy-baked-gi`) are the broad-hardware fallback.

---

## Cross-cutting concerns / open questions
- **Editor re-mesh latency** on brush-drag: rely on incremental dirty + BVH refit + async meshing;
  measure. A single edited chunk's interior could later move to GPU compute if profiling demands (never
  the seams — Gildea kept GPU-DC seams on CPU).
- **Mesh storage:** start with one ECS `Mesh3d` entity per chunk (simplest/off-the-shelf); revisit a
  retained/instanced buffer only if entity churn is a cost.
- **LOD popping:** skirts hide cracks but not pops; add geomorph only if needed.
- **Solari gating:** RT-hardware/NVIDIA constraint — decide target before committing to Solari vs baked GI.

## Status (this worktree)
- ✅ Default scene = `mesh_test.scene` (`DEFAULT_SCENE_PATH`); DDGI disabled (`intensity = 0`); world
  terrain gated off.
- ✅ **Phase 0 spike** — validated Surface Nets quality as "almost perfect" on our CSG content; the only
  artifact (a grid-aligned sphere pinhole) was fixed with a sub-voxel iso-shift.
- ✅ **Phase 1 COMPLETE** — `src/sdf_render/mesh_bake.rs`: residency-driven Surface Nets bake, **async**
  on `AsyncComputeTaskPool` (sample+mesh off-thread, build `Mesh` on the main thread). Staleness is a
  **per-unit content hash** (`edits::bake_content_hash` of the overlapping edits — the SAME key the GPU
  bake scheduler uses): a unit re-bakes iff its current hash ≠ the displayed mesh's. Residency and
  staleness derive from ONE overlap test, so they can't diverge → stale/ghost geometry is structurally
  impossible (a key-stamped `ChunkMesh` reaper is the closed loop on residency departure). This replaced
  the earlier dirty-region/gen-stamp scheme that leaked move-remnants.
- ✅ **Configurable chunk unit (landed before Phase 2)** — the bake/render unit is a runtime-tunable
  **`K×K×K`-brick chunk** (`MeshBakeConfig::chunk_bricks`, default 2, slider 1..=8; `K=1` = per-brick).
  One contiguous mesh per chunk → atomic coherent swaps (no per-brick fragmentation during drags), far
  fewer draw calls/entities, and contiguous geometry for later weld/decimate/LOD. Grid edge =
  `K·cell_stride + 2` via `ndshape::RuntimeShape`; same 1-voxel apron keeps chunk seams crack-free. The
  whole content-hash design is `K`-parameterized (this was a coarsening, not a new aggregation layer).
  NOTE: this `chunk_bricks` is the mesh-bake unit, distinct from `chunk::CHUNK_BRICKS` (GPU-atlas residency).
- Controls in the **"Mesh Bake"** bottom editor panel: SDF-render toggle, wireframe, **Chunk bricks (K)**
  slider, chunk/in-flight counts, Rebake, Capture diagnostics. View: `cargo run --features editor`,
  uncheck "SDF raymarch render".
- ◑ Phase 2 — material colours (per-vertex base + emissive, shaded) DONE; triplanar multi-material splat
  + PBR lighting deferred to 2b (needs a custom shader, verified once meshes are primary). ☐ Phase 3 —
  cross-LOD rings + skirts. ☐ Phase 4 — make meshes primary, retire raymarch/DDGI, adopt Solari.

**Latent `main` bugs fixed here (port to main):** probe-trace `ChunkLookup` 24-vs-32 hand-pack crash;
`SdfRenderEnabled` not `ExtractResource` (F1 no-op).
