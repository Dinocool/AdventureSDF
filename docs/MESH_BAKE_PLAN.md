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

## Phase 2 — Material & shading fidelity  ✅ DONE (off-the-shelf PBR, 2026-06-08)
Off-the-shelf path per the decision ("try StandardMaterial before a custom material/pipeline"):
- **Per-vertex base colour** — `mesh_chunk` resolves the material at each vertex
  (`edits::fold_csg(...).material_id`) and writes its LINEAR base colour to the vertex COLOUR.
- **Real PBR lighting + per-material PBR params** — each chunk takes a lit `StandardMaterial` (cached by
  its DOMINANT material id, sampled at the surface centroid): `base_color = WHITE` (so the per-vertex
  COLOUR rules the albedo) + `metallic`/`perceptual_roughness`/`emissive` from the `MaterialRegistry`.
  So red_metal is a shiny metal, white_gloss is glossy, the `emissive_orange` orb self-glows, etc.,
  lit by the scene's directional light. (No `AmbientLight` — it's a per-camera component in 0.18; the
  bright scene directional suffices. Add one on the camera later if shadowed faces read too dark.)
- A material **appearance edit** (colour/metallic/roughness/emissive) re-bakes + rebuilds the cached
  `StandardMaterial`s (the per-chunk content hash keys on material *id*, so an appearance-hash check
  bumps the rebake epoch and clears `mat_cache`). `mesh_test.scene` gained a floating `emissive_orange`
  orb to exercise emissive; the row uses sand/cobble/red_metal/white_gloss.

**Known limitations → the Phase-4 CUSTOM MESH MATERIAL (one shader does all three).** The off-the-shelf
StandardMaterial path is per-mesh and untextured, so it can't do the three things the full vision needs;
they are NOT separate efforts — one custom mesh material/pipeline, fed by richer vertex data, carries all:
1. **Per-vertex PBR params** — metallic/roughness/emissive as vertex attributes (not one dominant material
   per chunk). Fixes the "metallic/roughness uniform per chunk" issue; does NOT need the render-world
   textures, so it's the tractable, verifiable first step.
2. **Multi-material (≤`PALETTE_K`=4) weighted BLEND** — per vertex sample `edits::build_palette_indexed`
   → the ≤4 nearest material ids + their **L1-normalized weights** (from the per-material sub-voxel
   distance fields, feathered by `MaterialDef::blend_softness`); carry ids+weights as vertex attributes;
   blend the 4 materials' PBR by weight in the fragment shader (bonsairobo "Smooth Voxel Mapping"). This
   is what gives a feathered seam between two materials (e.g. cube-on-sand) instead of the current hard
   dominant-material edge. The raymarch already does this per-pixel — reuse its blend logic.
3. **Triplanar PBR textures** — sand/cobble etc. The texels live in the SDF render world as `D2Array`
   `TextureView`s (`render/pbr_textures.rs`), NOT main-world `Handle<Image>`s, so they can't bind to an
   off-the-shelf StandardMaterial; this is the part that forces a custom pipeline (the decision
   deprioritised it). Metals ALSO need an `EnvironmentMapLight`/IBL to read as metal (a lighting-setup
   fix, not the shader). Best done once meshes are primary (Phase 4) and verifiable live.
Until then: one dominant material per chunk, no blend, textured materials render flat base colour
(sand/cobble base_color = white → light); solid-PBR materials are exact (modulo metals needing IBL).

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
- ✅ Phase 2 — off-the-shelf PBR: per-vertex base colour + lit per-material StandardMaterial
  (metallic/roughness/emissive). Triplanar TEXTURE detail needs the custom render path (render-world
  texture arrays) → folded into Phase 4 when meshes are primary + verifiable live. ☐ Phase 3 — cross-LOD
  rings + skirts. ☐ Phase 4 — make meshes primary, retire raymarch/DDGI, adopt Solari (+ triplanar textures).

**Latent `main` bugs fixed here (port to main):** probe-trace `ChunkLookup` 24-vs-32 hand-pack crash;
`SdfRenderEnabled` not `ExtractResource` (F1 no-op).
