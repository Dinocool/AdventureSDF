# Procedural World Generator — Design Plan

> Status: **DRAFT for review.** Architecture locked via brainstorm; phasing + implementation
> details open to redline. Inspired by [LayerProcGen](https://github.com/runevision/LayerProcGen)
> (layer-stack, deterministic, contextual generation), adapted from its CPU/2D-only model to this
> engine's GPU SDF clipmap renderer with a mixed 2D/3D layer graph.

> ## Implementation status — Phase-1 vertical slice ✅ COMPLETE
> The full slice — CPU producer → GPU height ring → bake `Terrain` primitive → Bevy wiring — is
> implemented, reviewed, and verified. Build + clippy clean; **80+ tests green** (47 worldgen lib + 17
> edits lib + 4 determinism gate + 2 GPU bake parity + 3 benches + 7 shader-validation), all via
> `CARGO_TARGET_DIR=...\target\claude` + `--features editor`. GPU bake parity proves the rendered
> terrain SDF matches the CPU height field (max err 1.2e-5 vs 9.8e-5 slack). Perf: fBm 0.23 ms/chunk,
> ring build 0.88 ms, cold-fill 11.7 ms/49 chunks.
> - ✅ `worldgen/noise.rs` — bit-portable integer-hash fBm + analytic gradient.
> - ✅ `worldgen/coord.rs` — f64/int identity, world↔chunk (float-floor parity), GPU key.
> - ✅ `worldgen/artifact.rs` — `ScalarField2D` (apron) + bilinear surface sampling.
> - ✅ `worldgen/store.rs` — toroidal `ArtifactStore` w/ dirty/dropped delta.
> - ✅ `worldgen/layer.rs` + `layers/height.rs` — `Layer` trait + CPU-authoritative height layer.
> - ✅ `worldgen/plan.rs` — `GenerationPlan` DAG + padded read-windows.
> - ✅ `worldgen/manager.rs` — `LayerManager::update(focus)` rolling residency + budget + param cascade.
> - ✅ `worldgen/upload.rs` — toroidal GPU height **ring** builder (`GpuHeightCell` dir + node buffer)
>   + `sample_ring` CPU mirror; CPU↔GPU sample parity test.
> - ✅ `tests/worldgen_parity.rs` — multiplayer-determinism CI gate (pinned bit-exact references).
>
> **Done in the GPU consumer + wiring:** render-world ring extract/prepare bound into the bake (7-entry
> layout, `render/bake.rs`); `sample_terrain_height` + `case 6u` Terrain in `sdf_brick_bake.wgsl` (mirror
> of `upload::sample_ring`, Lipschitz-normalised, band-mid miss fallback matching the CPU);
> `SdfPrimitive::Terrain` end-to-end in `edits.rs` (eval/AABB/`to_gpu_edit`/tag 6) sampling the global
> CPU ring for picking parity; `WorldGenPlugin` (`roll_worldgen` + `spawn_terrain_volume`) registered
> from `SdfScenePlugin`; `wgsl_terrain_constants_match_rust` pin; GPU bake parity rig + benches.
>
> **Intentionally deferred (documented, not hidden):** delta-slot ring upload (full rebuild on delta is
> fine at 8×8); GPU cosmetic-detail octaves (hook present, =0.0); parallel chunk generation (pure layers
> ⇒ drop-in later); the editor "World Generator" panel (Phase-3); disk cache (§2.3). A visual
> `cargo run --features editor` confirmation is the user's to do — automated proof is the GPU bake parity.
>
> ### Live integration (debugged in-editor via BRP)
> Brought up live and verified by BRP (`brp_extras/screenshot`, `world.query`, resource mutate). Fixes
> the first run-through exposed:
> - **Black terrain → material**: a bare `SdfMaterialSource::default()` (inline, all-`None` overrides)
>   resolves to a ZERO albedo. The worldgen `Terrain` volume now spawns with an explicit mossy-green
>   inline material. (Confirmed via BRP: `registry_id` resolved, `sun_color=[10000…]` reaching the shader.)
> - **No light → spawn a sun**: the SDF lit pass sources its sun from a scene `DirectionalLight` +
>   `SceneEntity`; with no scene it's unlit (black). `WorldGenPlugin` now spawns its own sun.
> - **Clean backdrop**: the heavy 3000-light stress scene is skipped when worldgen is enabled
>   (`load_default_gallery` gated on `WorldGenEnabled`).
> - **Camera buried in terrain**: the orbit camera starts at distance 8 (eye y≈3) — *below* the terrain.
>   A one-shot `reframe_worldgen_camera` (Update, after the camera exists) pulls it to a distance-320
>   overview; the zoom clamp was raised 50→4000 m. (`orbit_camera` only syncs while the pointer is in
>   the viewport, so the reframe writes the transform directly.)
> - **Horizon holes → focus on the look-target**: generation centers on the orbit *target*, not the
>   camera eye, so the viewed ground stays inside the resident window.
> - **Live tweak loop**: a **World Gen** editor panel (`debug.rs`) exposes the `HeightParams` sliders;
>   editing them (or mutating the reflected resource over BRP) regenerates + rebakes the terrain that
>   frame — verified live.
>
> ### Known issues / next refinements (not yet satisfactory for ship)
> - **Far-extent corruption — FIXED (bulk)**: the worst far corruption was *torn boundary bricks* —
>   the generation radius equalled the terrain volume half-extent, so coarse bricks at the boundary
>   sampled a mix of real height + the miss fallback. Now the generation **radius (480 m) exceeds the
>   volume half (384 m)** so every volume brick is backed by real height, and the vertical bake band
>   was tightened (±256→±96 m). Far extent now renders clean.
> - **Coarse-LOD holes — FIXED (height-ring mips)**: implemented LOD-aware band-limiting. The ring now
>   stores a per-chunk **mip pyramid** (`MAX_HEIGHT_MIP=6`, node counts 65²→2², offsets pinned
>   CPU↔WGSL), built by a separable 1-2-1 **tent downsample** with *reflection* boundaries (so a planar
>   field stays exact at every mip — corner nodes don't drift). The bake selects the mip whose node
>   spacing ≈ its voxel size and samples that low-passed sub-block, so a coarse brick resolves a clean
>   zero-crossing. This is the standard low-pass-mip technique for baking a heightfield into a voxel SDF;
>   **min/max "maximum-mipmaps"** (Tevs 2008, conservative) remain the fallback if the coarsest LOD ever
>   needs a guaranteed-no-miss envelope. Optional polish: **trilinear mip blend** by fractional `log2`
>   to remove LOD-shell popping (currently nearest-mip).
> - **Coarse-LOD aliasing (torn/black far extents) — FIXED (mip select rounds UP)**: the mip select was
>   `floor(log2(voxel/2 m))`, which picks a mip *finer* than the voxel at every non-power-of-two LOD
>   (LOD 5/6/7 voxels 3.2/6.4/12.8 m fell to the 2/4/8 m mips → spacing **<** voxel → the height field
>   still carried frequencies the voxels couldn't resolve → aliased zero-crossings = the corrupt coarse
>   extents). The anti-alias criterion is `mip_spacing ≥ voxel`, so the select now **rounds up**
>   (`ceil(log2(voxel/2 m))`, clamped) — the finest mip whose spacing still ≥ voxel (over-blurs by < 1
>   octave). Verified across the real LOD ladder: `terrain_mip_select_is_band_limited_across_lods` (CPU
>   invariant: spacing ≥ voxel at every LOD — the regression gate) + `gpu_terrain_bake_finds_surface_across_lods`
>   (GPU: each LOD resolves a crossing matching its band-limited mip surface; LOD 5/6/7 now use mip
>   1/2/3). **Eviction is NOT the cause** — `no_resident_brick_outside_its_lod_window_through_flight`
>   (+ the existing `exited_fine_chunk_evicts_while_coarser_stays_resident` / `continuous_motion_keeps_probe_covered`)
>   prove the scheduler evicts correctly on LOD transitions; the nested coarse shell stays resident under
>   finer ones and `resolve_march` serves the finest, so a transition is a clean LOD pop, never a hole.
> - **Coarse terrain surface poking through fine terrain — FIXED (ANNULAR residency)**: the clipmap was
>   fully NESTED (every LOD's ring a full box on the camera), so coarse-LOD bricks stayed resident under
>   the fine near-field and `resolve_march`'s fine→coarse fall-through served the coarse surface (a
>   different mip height) in the air above the fine terrain — coarse shelves intruding through detailed
>   terrain. Confirmed mip-INDEPENDENTLY (flat slab: served walked 0→1→3→4 up the air column). Fix: each
>   LOD is now resident only in its **annulus** (its ring minus the finer LOD's ring) + a 1-chunk overlap
>   for a hole-free handoff (`SdfGridConfig::annulus_overlap_chunks`, default 1; `>= R` ⇒ old nested
>   behaviour). Predicates `is_resident_chunk` / `is_owner_chunk` in `bake_scheduler/window.rs`; the
>   recenter gains an annulus enqueue-gate + a **supersede-evict** sweep (drop a coarse chunk a finer
>   ring now covers — the missing cross-LOD eviction). The shader `in_ring_chunk` is now the OWNER test
>   (ring(L) minus the finer ring) so the empty-skip's "absent + owned ⇒ empty" holds (else it'd jump a
>   finer LOD's terrain) — pinned by `gpu_in_ring_chunk_matches_cpu` (360 coords, 0 divergences). Now the
>   near-field is single-owner: `annular_residency_single_owner_in_near_field` shows the resident stack
>   `[0]` and served owner→None up the air column (no coarse intrusion); `annulus_residency_is_single_
>   owner_band_no_gaps` pins the contiguous ≤2-LOD band; `continuous_motion_keeps_probe_covered` guards
>   the handoff stays hole-free. (Orthogonal follow-ups, not done: trilinear mip blend; owner-authoritative
>   `resolve_march` to also kill the thin 1-chunk overlap-band seam.)
>
> - **Missing extents / terrain didn't stream — FIXED**: the `Terrain` volume was spawned at
>   `Transform::IDENTITY` and generation was focused on the orbit target, pinning a fixed ±384 m island
>   at the world origin (its edges "stopped generating"). Now: generation focus = camera **eye**, and
>   the volume **follows the camera** (translation snapped to the 128 m chunk grid, re-bake on each
>   crossing). Because the volume translates, the height lookup is **world-anchored**: GPU sets a
>   `terrain_world_xz` private global in `eval_world` (the un-transformed world pos, per-voxel + per
>   curvature tap), CPU adds a `cpu_terrain_offset` to the local XZ — so the moving footprint samples
>   the correct world height. Radius (480) > footprint (384) preserved → no torn boundary, clean
>   horizon beyond. Terrain now streams with the viewer (the clipmap's intended behavior).
>
> *(Debugging note: a temporary Bevy Remote (BRP) server was used to live-inspect the editor; it has
> since been removed from `main.rs` + `Cargo.toml`.)*
> - **Bake volume cost**: the world-spanning `Terrain` volume (±384 m XZ × 512 m band) bakes
>   ~300–640k bricks. Should bound the vertical AABB to a tight shell around the actual surface
>   (min/max height of the resident region) instead of a fixed ±256 band.
> - **Streaming horizon**: terrain ends at the resident-ring edge (a hard line at the top of an
>   overview) — atmosphere/fog (§6.1) + the far-horizon coarse raymarch (§0 "Far horizon") will hide it.
> - **Focus model**: tying the generation focus to the orbit target suits the orbit overview; FPS mode
>   needs the focus to follow the fly-camera position.
>
> **Next:** Phase-2 (erosion filter §3.1) builds directly on the height layer + gradient already in place.

---

## 0. Locked decisions (from brainstorm)

| Decision | Choice |
|---|---|
| **World scope** | Infinite, camera-streamed (world-anchored), mirroring the SDF clipmap. The editor edits *generation rules*, not a finite baked terrain. |
| **Generation compute** | **Split by authority (revised — see Determinism).** *Authoritative* gameplay-relevant gen runs **CPU, portable-deterministic** (shared-seed multiplayer must agree cross-platform). **GPU** is for the SDF bake (per-client, non-authoritative) + **cosmetic micro-detail only**. CPU authoritative fields upload as artifacts the bake samples. |
| **Determinism** | **Shared-seed multiplayer**: clients generate independently from the seed and must agree on all gameplay-relevant features → cross-platform **bit-determinism** for authoritative layers (no GPU floats, no fast-math). Cosmetic-only detail may differ harmlessly. |
| **Vertical extent** | **Effectively unbounded.** Depth axis = **depth-relative-to-surface** (near-surface/underworld) + absolute-Y deterministic functions for deep realms; no normalized height range. |
| **Chunk hierarchy** | **Tiered chunk sizes are required**, not optional — the size-difference + dependency **padding** is the contextual-generation mechanism (bigger chunks = higher abstraction; a layer reads a padded window of the coarser layer below). |
| **Biome borders** | **Smooth blend** (top-N weights, per-biome blend widths) **+ optional transition biomes** (beaches, cliff edges, fringes authored as real biomes). |
| **Climate** | **Hybrid**: latitude baseline (global temp gradient by world coord) + altitude lapse + noise; humidity from noise + water proximity. Planet-like bands with local variety. |
| **Erosion** | **runevision's analytical erosion filter** (NOT a sim) — a per-point noise filter over the base height returning eroded height + analytical derivatives; branching gullies/ridges; constant cost, chunkable. A stacked **height-filter layer**, biome-parameterized. |
| **Caves** | **Noise caverns + worm/tunnel carvers + surface-breaching entrances.** Macro structure CPU-authoritative. |
| **Streaming / cache** | **Pre-generate a ring ahead** of the focus + **disk-cache expensive artifacts where sensible** (contextual layers). Seed remains source of truth (+ diffs); cache is an invalidatable optimization (recipe/seed/gen-version keyed). |
| **Coordinates** | **f64 / integer world coords on CPU** (authoritative gen + collision; aids bit-determinism); **rendering rebases to camera-relative f32**. No f32 absolute world positions. |
| **Hydrology** | **Downhill gully-traced streams** — trace water down the erosion filter's gullies (no flow-accumulation/lakes yet). |
| **Mesh trees** | **Bevy built-in instanced meshes**, sharing the SDF pass depth buffer; mesh-LOD/impostors later. (Depth-integration with the deferred SDF output must be verified.) |
| **Resources** | **Each resource is its own data-driven asset** (`ResourceDef` RON, like biomes) with its own spawning controls (biome affinity, depth/altitude/slope, scatter-vs-vein distribution, rarity). A dedicated authoritative CPU layer reads the registry; editor-authorable/publishable. |
| **Activation** | Worldgen is a **toggle that drives the scene live**; placed `.scene` edits **compose on top** as overrides (one unified editor). |
| **Atmosphere** | **Atmospheric scattering + aerial perspective + fog**, **varying by biome/location** (params come from the biome/climate layers). Feeds the **DDGI-GI** system (sky as a GI source) and coexists with planned **volumetric clouds**. |
| **Ground cover** | **GPU-instanced grass/plants driven by biome** (distance-faded). Dedicated later-phase system, separate from the `InstanceStream` trees/rocks. |
| **Surface overlays** | **Climate-driven material modifiers** layered over biome surfaces — snow above a snowline, wetness near water, sand drifts — blended at bake time (cuts per-biome material duplication). |
| **Day-night** | **Dynamic day-night cycle** driving sun direction/color, atmosphere, and shadows; feeds DDGI relighting. |
| **Weather** | **Dynamic weather system (later)** — states (clear/rain/snow/fog/storm) modulating surface overlays, atmosphere, and clouds, gated by biome/climate. |
| **POI system** | **One general, data-driven POI system** (`POIDef` RON per type) is the umbrella; **settlements are a POI type** (with tiering/satellites/roads). Dungeons, ruins, landmarks, settlements = different POI types, each its own asset. Authoritative CPU contextual layer. |
| **Biome metadata** | Biomes carry an **extensible gameplay-metadata block** (mob/creature spawn tables, ambient audio/music, lighting/mood tags, gameplay flags) beyond visuals. |
| **Lattice dimensionality** | **Per-layer.** 2D (XZ) for continents/climate/hydrology; **3D** for biomes (sub-surface) and caves/overhangs. A 3D layer depending on a 2D layer samples the 2D field per column. |
| **Composition** | Procedural terrain is primary; hand-placed `SdfVolume` edits coexist for testing/runtime gameplay via the existing CSG fold + `SdfOrder`. |
| **Biome model** | Generic **named-field registry** seeded with Minecraft-style axes (temperature, humidity, continentalness, erosion, weirdness, depth/density), extensible. Biomes = noise **classified into discrete regions ("islands")**: **2D classification for overworld**, **3D for underworld**, blended across a depth transition. Surface-to-ground cave carvers supported. |
| **Scene instancing** | Authored editor `.scene` files act as **prefabs** placed by a settlement/POI layer; instancing injects the prefab's transformed edit records into the existing gather→BVH→bake stream. |
| **Terrain editing** | No freeform sculpt of the heightfield. Discrete **placed SDF edits** compose over procedural terrain (override stream) — primarily **at runtime in-game**, but fully possible in the editor. Persisted as diffs. |
| **Persistence** | **Re-derive from seed** + store only diffs (placed/override edits, modified instances). Infinite world, tiny saves. |
| **Water** | **Dedicated water pass** (screen-space over the depth buffer), not an SDF surface — waves/transparency/refraction/shorelines. **Rivers** later. Plus an **explorable deep-ocean realm** (Subnautica-style): ocean-floor terrain, an `Underwater` biome realm, and an **underwater atmosphere** mode (submerged volumetric fog, depth-based color absorption, caustics, god-rays) when the camera is below the surface. |
| **POI stamping** | A POI acts as a **local micro-biome that *overlays* (not replaces) the base biome** — it adjusts height variance (flatten/level), stamps structure, and blends materials/scatter masks over the existing biome via the biome-weight machinery (§5.1). |
| **Net model** | **Clients generate from the shared seed; only diffs sync** (placed edits, POI/resource runtime state) via a server relay. Bandwidth-light; rides entirely on the determinism guarantee. |
| **Cave/underworld lighting** | **DDGI ambient + emissive materials** (lava/crystal/fungi) **+ POI-placed point lights** (torches). |
| **Determinism verification** | **Reference-vector parity harness in CI** for authoritative layers (hashes at known coords/seeds) — mirrors the existing light-grid/chunk parity tests. |
| **Authoring** | **Purely rule-driven** — no hand-painted region overrides. The world comes from the recipe/rules; only discrete placed edits (§3 override stream) are manual, persisted as diffs. |
| **World map** | A **macro world-map panel** renders continents/biomes/climate/rivers/POIs at world scale, with click-to-teleport + authoring overlays (serves "visualize continents"). |
| **Terrain texturing** | **Triplanar projection + stochastic (hex) tiling-break**, biome-blended — SDF has no UVs; this hides seams/repetition at scale. |
| **Sub-generators** | **Nested procedural sub-generators** — a POIDef (or biome) can invoke a sub-generator (dungeon layout, city blocks, interiors) composing within its region, not just place static prefabs. Recursive worldgen. |
| **Far horizon** | **Raycast the coarse authoritative height field** beyond the brick clipmap's furthest LOD (an extended analytic SDF march), not an impostor mesh. Distant peaks/coastlines render; blends into fog. |
| **World-shaping = layers** | Sea level, continent scale, climate warmth, biome rarity, cave density, etc. are **individual layers** (each with params), not a separate global-knob block. The recipe selects + configures them. |
| **Destructibility** | **Full SDF destructibility at runtime**, diff-synced — players carve/build anywhere via the override edit stream; persists + syncs as diffs. Needs diff-volume management (region compaction/limits) at scale. |
| **Sky bodies** | **Sun + moon + stars (later)** driven by the day-night cycle + atmosphere; optional aurora. |
| **CPU queries / collision** | GPU is source of truth; **query GPU directly** where possible. Build a **low-res collision map near the player (LOD 0/1)** by GPU-generating + reading back a coarse height/density field for physics. Coarse regional fields stay CPU-cheap for contextual layers; **no full GPU/CPU parity of fine noise.** |
| **Layer authoring** | Layers are **fixed code** (Rust + WGSL kernels with reflected params), but **data-extensible** like biomes — params/sub-config driven by the world recipe + RON (swappable noise, biome-feature tables). No node-graph compiler. |
| **World definition** | A **world recipe asset** (seed + active layer stack + params + biome set). The layer registry is the palette; the recipe instantiates a world. Multiple worlds, shareable/publishable. |
| **LOD generation** | Generation is deterministic, so **LOD is a sampling choice**, not baked in. Continuous-field layers (height/climate) **downsample** gracefully and are sampled at whatever LOD a consumer needs; **discrete/placement layers (tree/rock scatter) don't LOD** — they generate within a range and are **culled**, not downsampled. |
| **Instance rendering** | Per-**template** render mode: **SDF** templates (rocks, structures) inject edits into the bake; **mesh** templates (trees) use a conventional instanced-mesh + impostor path. The placement/scatter system is shared; only the realization differs. |

---

## 1. Existing engine hooks (what we reuse)

The renderer already exposes nearly every mouth the generator needs to feed:

- **`SdfPrimitive::Heightmap { half_xz, max_height, freq, amp, seed }`** — baked as `p.y - height(xz)`,
  Lipschitz-normalized in `sdf_brick_bake.wgsl`. This is the **seed** of terrain; we generalize it
  from "one analytic noise" to "sample the layer artifacts + synth detail."
- **Edit pipeline**: `gather_sorted_edits → BVH → schedule_bakes → GpuEdit upload → fold_csg`.
  Procedural **features and instanced scenes inject here as ordinary edits.**
- **Material registry + paged PBR atlas**: `MaterialDef`, `MaterialRegistry`, `SdfMaterialSource`,
  `resolve_materials()`, ≤4-material brick **palettes** blended at shade time. Biome surface
  materials map into this.
- **Toroidal clipmap chunk system** (`chunk.rs`): absolute `ChunkKey`, `chunk_gpu_key()`, GPU binary
  search, camera-ring residency. The generator's artifact stores reuse this **world-anchored
  toroidal addressing** pattern.
- **Editor**: `register_panel()` + reflected-resource sliders (à la `SdfRaymarchParams`) → live
  param tweaking with no rebuild.

### Renderer changes required (kept small + enumerated)
1. **Artifact clipmap textures/buffers** (control maps + 3D fields) streamed with the chunk ring; bound into the bake.
2. **Generalize `Heightmap` → `Terrain`** primitive: sample artifact fields + synthesize fine detail (octaves/warp) in `sdf_brick_bake.wgsl`.
3. **3D density sampling** in bake (caves/overhangs/surface-to-ground carvers) folded via `fold_csg` (subtract).
4. **Terrain material-resolution path**: rule-based (slope/altitude/curvature/biome) blend → palette,
   with **triplanar projection + stochastic (hex) tiling-break** (SDF has no UVs). *(May extend
   brick/palette to carry blend weights — see Open Decisions.)*
5. **Feature / instanced-edit injection** into `gather_sorted_edits`.
6. **Dedicated water pass** — a screen-space pass after the deferred lit composite, reading the depth
   buffer + a sea-level/water-body field, shading waves/transparency/refraction/shorelines. New
   render-graph node; *not* part of the SDF bake.
7. *(no GPU collision readback)* — collision comes from the **CPU authoritative coarse field**
   directly (§2.8); the renderer needs no readback path for physics.
8. **Mesh-instance path** (trees / dense vegetation) — **Bevy's built-in instanced mesh/PBR**
   rendering driven by `InstanceStream` placements; mesh-LODs → impostors → cull later. **Must share
   the SDF pass depth buffer** (reverse-Z `Depth32Float`, `GreaterEqual`) so terrain/mesh occlude each
   other correctly — verify Bevy's mesh pass can target/read it. Later phase.
9. **Atmosphere/sky pass** — atmospheric scattering + aerial perspective + distance fog, with params
   **driven by the biome/climate layers** (location-varying). Hides the clipmap horizon; **feeds
   DDGI-GI** (sky radiance as a GI source). Later phase; coexists with planned **volumetric clouds**.
10. **GPU-instanced grass** (later) + **bake-time surface overlays** (snow/wetness modifiers in the
    terrain material resolution, §4.3).
11. **Far-horizon raymarch** — extend the primary march to sphere-trace the **coarse authoritative
    height field** beyond the brick clipmap's last LOD, blended into fog. No mesh impostor.

---

## 2. Core abstractions (the LayerProcGen port)

### 2.1 Artifacts
Every layer produces one or more **typed, world-anchored artifacts** that compose into the final
worldgen. Artifact kinds:

- `ScalarField2D` / `ScalarField3D` — named scalar (e.g. `temperature`, `continentalness`, `cave_density`).
- `Classification2D` / `Classification3D` — discrete region ids + per-id weights (biomes = this).
- `InstanceStream` — placements `{ template_ref, transform, params }`. **One unified type** spanning
  the whole complexity spectrum: a single-primitive **tree/rock/resource node** up to a multi-edit
  **prefab scene** (house, set-piece). The template is `1..N` `GpuEdit` records; instancing composes
  template × transform into the bake stream (see §5).
- `VectorGraph` — polylines/graphs (rivers, roads) — CPU-produced, rasterized to fields or edits downstream.

Artifacts live in **toroidal clipmap stores** keyed by absolute world coords (GPU textures for
field/classification artifacts; CPU buffers uploaded for graph/instance artifacts).

### 2.2 Layer
```text
trait Layer {
    type Params: Reflect + Default;       // auto-egui, hot-tweakable
    fn chunk_size(&self) -> ChunkSize;     // tier: bigger = higher abstraction (REQUIRED, see §2.7)
    fn dimensionality(&self) -> Dim;       // D2 | D3
    fn authority(&self) -> Authority;      // Authoritative(CPU, portable-det) | Cosmetic(GPU ok)
    fn dependencies(&self) -> &[LayerDependency]; // (layer id, padding) — padded read-window
    fn produces(&self) -> &[ArtifactDecl]; // named outputs + kinds
    fn generate(&self, ctx: GenCtx);       // pure f(chunk_coord, seed); CPU job or GPU dispatch
    fn visualize(&self, ctx: VizCtx);      // 2D map render + 3D overlay contribution
}
```
- **Determinism invariant** (see §2.8): `generate` is pure `f(chunk_coord, world_seed)`,
  order-independent.
  - **Authoritative layers** (gameplay-relevant) → **CPU, cross-platform bit-deterministic** (no GPU
    floats, no fast-math). Shared-seed multiplayer requires every client to agree.
  - **Cosmetic layers** (visual-only micro-detail) → may run on **GPU**, world-anchored toroidal
    addressing for flicker-free streaming (cross-platform exactness not required since gameplay
    doesn't depend on them).

### 2.3 LayerManager (Bevy resource)
- Holds artifact stores, the layer registry (dependency-ordered), and the **GenerationPlan**.
- Driven by focus points (camera); rolls **create/destroy** of artifact regions on
  `AsyncComputeTaskPool` (CPU layers) / render-graph compute (GPU layers).
- Recomputes the plan on camera snap (reuse the chunk-ring recenter trigger) and on **param edits**
  (dirty the affected layer's region → cascade to dependents → re-gen → re-bake).
- **Pre-generation**: the plan extends a **ring ahead** of the focus so motion rarely reaches
  ungenerated land; just-streamed regions still show coarse-LOD-first and refine async (no holes).
- **Disk cache**: expensive artifacts (contextual CPU layers — settlements, rivers, roads) may be
  cached to disk keyed by `(recipe_id, seed, gen_version, chunk_coord)`; cheap analytical layers
  (height, erosion, climate) re-derive. Cache is an optimization, **not** the source of truth — seed
  + diffs remain authoritative; a gen-version bump invalidates stale cache.

### 2.4 Field registry
Generic registry of named scalar fields. **Seeded** with the Minecraft axis set:
`temperature, humidity, continentalness, erosion, weirdness, depth/density`. Adding an axis = a new
field layer + a name; biome envelopes reference axes by name. No hard-coded axis list in the biome
resolver.

**Climate derivation (hybrid)**: `temperature` = latitude baseline (global gradient by world coord) −
altitude lapse + noise; `humidity` = noise + water-proximity falloff. Gives planet-like temperature
bands with local noise variety, rather than fully-scattered multi-noise climate.

### 2.5 World recipe + layer extensibility
- **World recipe asset** (`assets/worlds/*.world.ron`): `{ seed, layers: [{ type, params }], biome_set }`.
  The **layer registry** (code) is the palette of available layer types; the recipe selects + configures
  them and binds the biome set. Loading a recipe instantiates the `LayerManager` stack. Editor can
  author/save recipes ⇒ multiple worlds, shareable/publishable.
- **Layers are fixed code but data-extensible** (no node graph): a layer exposes reflected params and
  may take *data-driven config* — swappable noise function/octaves, lookup tables, the biome-feature
  scatter table — supplied by the recipe/RON. New layer *type* = code; new *world/biome/config* = data.
- **World-shaping lives in layers, not a global-knob block**: sea level, continent scale, climate
  warmth, biome rarity, cave density, etc. are each **their own (coarse) layer** with params. The
  recipe just selects + configures them — so "tuning the world" is tuning those layers, and the set is
  extensible the same way every other layer is.

### 2.6 LOD = a sampling choice (determinism dividend)
Because every layer is pure `f(world_coord, seed)`, **LOD is decided by the consumer at sample time,
not baked into the artifact**:
- **Continuous-field layers** (height, climate, density) downsample gracefully — the bake samples them
  at whatever clipmap LOD it's baking, coarse far away, fine near. Some fields may *always* need full
  detail (sharp features); that's a per-layer flag, not a global setting.
- **Discrete / placement layers** (tree/rock/resource scatter, instances) **do not LOD** — sampling a
  scatter layer "at low res" is meaningless. They generate instances within a **generation range** and
  are **culled** (or swapped to impostors, §5) past it, never downsampled.
- Classification/biome borders must be **stable across LOD** (snap to a consistent rule) to avoid
  popping as the camera moves between clipmap levels.

### 2.7 Tiered chunk hierarchy (required mechanism)
This is the heart of the LayerProcGen model — **not optional**:
- Each layer has a **chunk size**; **bigger chunks = higher abstraction**. Higher layers plan at scale
  (continents over km-sized chunks), lower layers add detail (biome/surface over small chunks).
- A dependency declares **padding**. Generating a chunk of layer L forces every chunk of its
  dependency within `(L_chunk_bounds + padding)` to exist first → L reads a **padded window** of the
  coarser layer. *This padded read-window is what makes contextual ops (blur, relaxation,
  pathfinding, road planning, biome smoothing) deterministic* — you generate slightly beyond your
  bounds so neighbor-dependent math has its inputs, with no seams and no order dependence.
- The **GenerationPlan** walks top-level requirements (focus points) down through the dependency DAG,
  computing exactly which chunks of which layers must exist and in what order, then rolls
  create/destroy as focus moves.
- Example tiers (sizes TBD, parameterized): `Continent` (huge) → `Region/Climate` (mid) →
  `Biome` (small-mid) → `SurfaceDetail`/`Scatter` (small). Uniform sizing would break "plan at scale"
  and the padded-context model.

### 2.8 Determinism & the CPU/GPU authority split
Shared-seed multiplayer is the hard constraint: **clients generate independently and must agree on
everything gameplay-relevant**, across GPU vendors/drivers/OSes. GPU floating-point is *not*
bit-portable, so:
- **Authoritative layers → CPU, portable-deterministic** (integer/fixed-point or strictly-specified
  IEEE, fast-math off, a portable hash/noise basis). Covers: coarse height/continent shape, biome
  **classification**, resource-node placement, settlements, rivers — anything clients must agree on
  or that drives collision/gameplay.
- **GPU is limited to** (a) the **SDF bake** (each client bakes its own view — non-authoritative; only
  needs to *look* the same, sourced from the same authoritative fields) and (b) **cosmetic
  micro-detail** that never affects gameplay/collision (sub-texel surface noise). Cosmetic divergence
  between clients is harmless.
- **Data flow**: CPU authoritative layers produce artifacts (height/climate/biome-weight fields,
  instance lists) → uploaded to GPU artifact stores → the bake samples them + adds cosmetic detail.
- **Collision** reads the **CPU authoritative coarse field directly** (it already exists CPU-side) —
  no GPU readback needed for authoritative height. (The earlier "GPU-first / readback collision map"
  idea is superseded by this split.)
- **Net model**: **clients generate the world independently from the shared seed; only diffs sync**
  (player/editor overrides, POI/resource runtime state) via a server relay. Bandwidth-light; correctness
  rides entirely on the bit-determinism above. Any server-driven RNG events (e.g. dynamic spawns) sync
  as diffs, not regenerated.
- **Verification**: a **reference-vector parity harness in CI** pins hashes of authoritative-layer
  outputs at known `(coord, seed)` points (and ideally across target platforms), mirroring the existing
  light-grid/chunk parity tests. This is the safety net — a silent determinism regression desyncs
  multiplayer, so it must fail loud in CI.

### 2.9 World coordinates (infinite-world precision)
f32 loses precision a few km from origin (vertex jitter, brick cracks, physics drift) — fatal for an
infinite world. So:
- **CPU authoritative side**: world coords are **f64 or integer** (chunk-index + intra-chunk offset).
  Exact everywhere; also helps cross-platform bit-determinism (integer math > float).
- **GPU / rendering side**: **camera-relative f32** — everything rebased so the camera sits near the
  origin, keeping coords small where precision matters.
- **Integration note**: the existing clipmap is world-anchored (absolute chunk keys, ±90 km). Audit
  the camera uniform + bake/march for any **absolute f32 world positions** and convert to
  camera-relative; the chunk-key system already isolates the integer lattice, so this is mostly the
  per-vertex/per-ray world-pos math, not the directory.

---

## 3. Terrain → SDF composition

In the bake, terrain distance is the CSG fold of:

```text
terrain_sdf(p) = (p.y - heightfield(p.xz))        // 2D surface from height artifact + synth detail
               ∘ cavern_density(p)                 // 3D subtract (caves, overhangs)
               ∘ surface_to_ground_carvers(p)      // 3D subtract (cave entrances/worms breaching surface)
               ∘ feature_edits(p)                   // scattered rocks/trees as GpuEdits
               ∘ scene_instance_edits(p)            // placed prefab edits
               ∘ override_edits(p)                   // placed/override SDF content (runtime or editor)
```

**Override stream**: discrete placed edits (runtime gameplay or editor) compose over procedural
terrain at high `SdfOrder`. These are the *only* persisted world state (re-derive everything else
from seed) — saved as diffs and replayed into the gather on load. Supports **full runtime SDF
destructibility** (carve/build anywhere), diff-synced in multiplayer; at scale this needs
**diff-volume management** — region compaction, coalescing, and/or budget limits so a heavily-edited
world's diff set stays bounded.

- **Heightfield** = `base_height(xz)` → **erosion filter** → cosmetic detail:
  - `base_height`: continents/regions/biome height-shaping, sampled from the CPU-authoritative height
    artifact (generalized from the current `Heightmap`).
  - **Erosion filter** (runevision, §3.1): an analytical per-point filter over the base height +
    gradient → eroded height + derivatives (gullies/ridges). Authoritative (shapes collision) →
    portable-deterministic; the same formula can refine on GPU for cosmetic micro-gullies.
  - Cosmetic octave/warp detail synthesized in-bake on top (GPU, sub-gameplay).
- **Density (3D)**: per-voxel subtract of **noise caverns** (cheese/spaghetti), **worm/tunnel
  carvers**, and **surface-breaching entrances** — macro structure CPU-authoritative.
- **Edits/instances**: reuse the entire existing edit path; nothing new in the fold beyond ordering.

### 3.1 Erosion filter (runevision)
Source: ["Fast and Gorgeous Erosion Filter"](https://blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html)
(builds on Clay John 2018 / Felix Westin 2023). **Not a simulation** — a special noise that produces
branching gullies/ridges while every point is evaluable in isolation (fast, GPU-friendly, chunkable,
constant cost per point).
- **Algorithm**: extract gradient from the input height → Worley-like cells with pivot points (avoids
  rotation distortion) → cosine/sine stripe pairs aligned to the local gradient → blend neighbor cells
  → stack octaves at decreasing scale (each accounts for prior slope changes) → fade at peaks/valleys
  to preserve sharp features. Returns modified height **+ analytical derivatives**.
- **I/O**: takes any height function + its gradient; the base height layer must expose (or allow
  computing) gradients. Returns eroded height + derivatives the bake uses for normals/slope.
- **Params** (reflected, biome-overridable): erosion strength/octaves, fade target (altitude-based
  peak/valley preservation), detail level (gate high-freq erosion to steeper slopes), gully weight
  (peak pointiness), ridge/crease rounding, input-slope pretending.
- **Bonus**: gully structure can guide **rivers/paths** downhill (a LayerProcGen sample does exactly
  this) — hydrology can read the erosion gradient instead of re-deriving flow.

---

## 4. Biomes

### 4.1 BiomeDef asset (`assets/biomes/*.biome.ron`, hot-reload, editor-authored)
```text
BiomeDef {
    name,
    // selection: where in named-field space this biome lives
    envelope: { <field_name>: range, ... },     // classification rule
    realm: Overworld | Underwater | Underworld | Both,  // surface / ocean-floor / sub-surface; 2D vs 3D classification
    transition: Option<{ between: [biome, biome], width }>,  // optional edge biome (beach, cliff, fringe)
    // shaping
    height: { base, octaves, amplitude, roughness, warp },
    // surface materials: ordered rules, slope/altitude/curvature gated, with blend widths
    surfaces: [ { material: SdfMaterialSource, when: {slope, altitude, curvature}, blend_width } ],
    // procedural content
    features: [ { primitive_or_scene_ref, density, jitter, material } ],
    // atmosphere (location-varying; feeds the atmosphere pass + DDGI)
    atmosphere: { fog_color, fog_density, sky_tint, haze, ... },
    // gameplay metadata (extensible; consumed by other systems, not gen)
    gameplay: { spawn_tables, ambient_audio, music, mood/lighting_tags, flags, ... },
}
```

### 4.2 Classification ("islands")
**CPU-authoritative** (§2.8) — biomes drive gameplay, so classification is computed cross-platform
deterministically and uploaded as a biome-weight artifact the bake reads (not selected per-voxel on GPU).
- **Overworld**: 2D classification — fields sampled at the surface column → biome id/weights map.
- **Underworld**: 3D classification — full 3D field point classified into 3D regions ("islands").
- **Blend**: a depth transition band where overworld 2D selection cross-fades into underworld 3D
  selection, so the surface and sub-surface biomes agree at the seam.
- Top-N biome **weights** (not just a hard id) so material/height blend across biome borders.
- **Transition biomes** (beaches, cliff edges, forest fringe) are authored as real biomes that win in
  the border band between two parents — smooth blend by default, art-directed edges where wanted.

### 4.3 Material resolution (bake-time)
1. Sample biome weights at the voxel (2D-projected above the transition, 3D below).
2. For each candidate biome, evaluate its surface rules vs **local slope/altitude/curvature** (from
   the SDF gradient — normals already available).
3. Blend the resulting materials (weighted by biome weight × rule weight) into the brick's ≤4-slot
   palette.
- Sub-surface case: a cave wall's biome comes from the 3D field at that depth; its rules pick the
  stone/moss/crystal blend.
4. **Surface overlays** (climate-driven modifiers, applied after biome rules): blend snow above a
   temperature/altitude **snowline**, wetness near water, sand drifts, etc. Global modifiers that
   override/tint the resolved material by climate state — cheaper than authoring snowy/wet variants
   per biome, and gives a consistent snowline across all biomes.

### 4.4 Registry + extensibility
- `BiomeRegistry` loads `assets/biomes/`, hot-reloadable, supports **editor authoring/saving**
  ("publish a new biome"). New biome = drop a RON file; no code change.

---

## 5. Instanced content (scatter → prefabs)

**One unified instancing system** covers everything placed into the world, from cheap scatter to
authored set-pieces:

- **Templates declare a render mode** — placement/scatter is shared; only realization differs:
  - **SDF templates** → `1..N` `GpuEdit` records injected into the bake (rocks, ore veins, props,
    full prefab scenes from authored `.scene` files — houses, ruins, set-pieces).
  - **Mesh templates** → conventional **instanced-mesh + impostor** rendering (trees and other dense
    vegetation), outside the SDF bake.
- **Complexity spectrum** (both modes): single object (a rock, a tree, a resource node) → small
  prefab (rock cluster, ore vein, campfire) → full prefab scene (building, set-piece).
- **Placement layers** emit `InstanceStream` artifacts `{ template_ref, transform, params }`:
  - **Scatter layers** (**CPU authoritative** — clients must agree on resource/tree/rock positions):
    density-driven Poisson/jitter scatter gated by biome + slope/altitude (biome `features` rules in §4.1).
  - **Structured placement layers** (CPU, contextual): settlements, POIs — neighbor-aware so
    instances don't overlap and align to roads/terrain (see §5.1).
- **Instancing mechanism** (SDF templates): inject each template's edit records, transformed, into
  `gather_sorted_edits`.
  - Preferred: **reference + transform** (don't duplicate ECS entities per instance) so thousands of
    rocks / hundreds of houses don't cost N× entities. The bake's edit upload composes template
    edits × instance transforms.
  - Reuses materials, CSG fold, picking, gizmos. Hand placement (testing) is the same path with a
    hand-authored transform.
- **Instancing mechanism** (mesh templates): feed the placement transforms to an instanced-mesh
  renderer with distance-based mesh LODs → impostors → cull. Shares the `InstanceStream` placement
  data; realization is a separate render path (new work, later phase).

### 5.1 POI system (umbrella — settlements are one POI type)
**One general, data-driven POI system**; settlements, dungeons, ruins, landmarks, unique set-pieces
are all **POI types**, each its own asset. Authoritative CPU contextual layer (clients must agree).
- **`POIDef` asset** (`assets/pois/*.poi.ron`, like biomes/resources):
  ```text
  POIDef {
      name, kind,                          // settlement | dungeon | ruin | landmark | ...
      template_ref | generator,            // prefab scene (§5) or a sub-generator
      // placement
      context: { biomes, altitude/slope, near_water, on_peak, ... },  // suitability
      rarity / spacing, seed_salt,
      // hierarchy (for settlements & nested POIs)
      tier: Option<{ rank, satellites: [poi_ref @ rate], connects_via: roads }>,
  }
  ```
- **Settlements** = POIDefs with a `tier` (hamlet → village → town → city): higher tiers are rarer,
  seed lower-tier **satellites**, and emit a road `VectorGraph` connecting them.
- **Stamping = a local micro-biome overlay** (not a replacement): a placed POI defines a small
  influence region that **overlays the base biome** via the same biome-weight machinery (§4) —
  reducing height variance (flatten/level its footprint), stamping foundations/paths, biasing surface
  materials, and emitting **scatter-suppression masks** (no trees in the town square). The base biome
  still shows through at the edges (blend), so a village reads as "this place, but settled."
- **Contextual placement** is the LayerProcGen padding sweet spot — a POI chunk reads its neighbors to
  enforce spacing, avoid overlap, and plan roads deterministically.
- Emits `InstanceStream` (the prefab buildings/structures via §5) + optional `VectorGraph` (roads,
  rasterized downstream). Editor: a **POI editor panel** to author/tune/publish POI types.
- **Nested sub-generators**: a POIDef's `generator` can invoke a **procedural sub-generator** that
  composes *within the POI's region* — a dungeon-layout generator, city-block planner, building
  interior, etc. — instead of (or alongside) a static prefab. This is **recursive worldgen**: the
  sub-generator is itself a small layer stack scoped to the POI bounds (same determinism rules,
  seeded from the POI's `seed_salt + coord`), emitting edits/instances back into the bake stream.
- Deferred to a later phase; §5 unified instancing is the foundation it builds on.

### 5.2 Resources (data-driven, each its own asset)
Mirrors the biome model — **every resource type is its own asset** with its own spawning controls,
hot-reloadable and editor-authorable/publishable. No code change to add a resource.
- **`ResourceDef` asset** (`assets/resources/*.resource.ron`):
  ```text
  ResourceDef {
      name,
      template_ref,                 // the SDF/mesh instance template to place (§5)
      realm: Surface | Subsurface | Both,
      // spawning controls
      biomes: { affinity per biome | any },     // where it's allowed / preferred
      depth: range,                 // for subsurface veins (uses the depth axis, §2.8)
      altitude/slope: ranges,       // surface gating
      distribution: Scatter { density, jitter } | Vein { noise, vein_size, branchiness },
      rarity / cluster_size, seed_salt,
  }
  ```
- **`ResourceRegistry`** loads `assets/resources/`, like `BiomeRegistry`. A **resource distribution
  layer** (CPU authoritative — clients must agree on every node/vein) reads the registry + the
  biome/depth/height artifacts and emits an `InstanceStream` per resource. Ore **veins** use 3D noise
  gated by depth+biome; **surface nodes** use biome-gated scatter.
- Editor: a **Resource editor panel** (sibling to the biome editor) to author/tune/publish resource
  defs and visualize their spawn distribution.

---

## 6. Environment passes (water, rivers, atmosphere, ground cover)

- **Dedicated water pass** (not an SDF surface): a render-graph node after the deferred lit composite.
  Inputs: scene depth/normal + a **water-body field** (sea level globally, per-body elevation for
  lakes). Per shoreline pixel: reconstruct world pos from depth, compare to water elevation, shade
  water where terrain is below it. Gives proper transparency, refraction (sample the lit color
  buffer offset by the water normal), depth-based absorption, and shoreline foam.
- **Sea level** falls out of the height/continentalness fields and feeds biome rules (beaches,
  coastal biomes) regardless of rendering.
- **Deep-ocean realm (Subnautica-style, explorable)**: low continentalness → deep ocean **basins and
  trenches** in the height field; the **ocean floor is real terrain** with **`Underwater` biomes**
  (reefs, kelp forests, trenches, hydrothermal vents) classified like any other realm. Underwater
  caves/overhangs use the same 3D density path. See §6.4 for submerged rendering.
- **Rivers (later)** — **downhill gully-traced streams**: a CPU layer traces water down the **erosion
  filter's gullies** (the erosion gradient already encodes flow direction, §3.1) into a river
  `VectorGraph`. Downstream: (a) **carve** channels into the height/density along the polyline, and
  (b) drive the water pass with a per-segment water-surface elevation. No flow-accumulation/lake
  modeling yet (rivers don't widen downstream; basins aren't flooded) — a later upgrade.

### 6.1 Atmosphere, sky & GI
- **Atmospheric scattering + aerial perspective + distance fog** (new render pass) — hides the clipmap
  horizon and grounds scale. Drives off the existing sun.
- **Location-varying**: atmosphere params (`fog_color`, `fog_density`, `sky_tint`, `haze`, …) come
  from the **biome/climate layers** (BiomeDef.atmosphere, §4.1), blended by the same biome weights →
  smoothly varying skies/fog as you cross regions.
- **Feeds DDGI-GI**: sky radiance is a GI source for the dynamic diffuse GI system; atmosphere output
  must be sample-able by the GI probes (design the pass DDGI-reuse-friendly).
- **Volumetric clouds** (planned, separate system) coexist with this pass.

### 6.2 Ground cover (grass) — later
**GPU-instanced grass/plants** over the terrain surface, **driven by biome** (species/density per
biome), distance-faded. Separate from the `InstanceStream` trees/rocks (its own dense-instancing
system). Deferred to a later phase.

### 6.3 Dynamics (day-night, weather)
- **Day-night cycle**: time-of-day drives sun direction/color → the existing shadows, the atmosphere
  pass (§6.1), and **DDGI relighting**. The GI/atmosphere must handle a moving sun gracefully.
- **Sky bodies (later)**: sun + moon disc, star field, optional aurora — driven by the day-night cycle
  and atmosphere. Polish that sells the dynamic sky.
- **Weather (later)**: dynamic states (clear/rain/snow/fog/storm) gated by biome/climate that modulate
  the **surface overlays** (wetness/snow accumulation), **atmosphere** (fog density/color), and
  **volumetric clouds** — the unifying layer over the overlay + atmosphere + cloud systems.

### 6.4 Underwater rendering (submerged atmosphere)
When the camera is **below the water surface**, swap the atmosphere for an **underwater mode**:
- **Volumetric water fog** with **depth-based color absorption** (warm wavelengths die first → blue/
  green deep), driving an exponential visibility falloff (the Subnautica look).
- **Caustics** projected onto the seafloor/objects, **god-rays** from the surface, suspended-particle
  scatter, and a refracted/animated **surface seen from below**.
- **Per-`Underwater`-biome params** (like §6.1's atmosphere block): water color, murkiness, caustic
  strength, bioluminescence ambient — so a kelp forest, a clear reef, and a black trench feel distinct.
- **DDGI underwater**: ambient comes from the attenuated surface light + emissive (vents,
  bioluminescence); the underworld cave-lighting story (DDGI + emissive + placed lights) applies here too.
- **Transition**: handle the camera crossing the surface (split-screen waterline, wet-lens) — a state
  the water pass already tracks (above/below test).

## 7. Collision & CPU queries (physics)

Per §2.8, **authoritative gen already lives CPU-side** (it must, for shared-seed multiplayer), so
collision needs no GPU readback:
- **Source**: the CPU authoritative height/density artifacts (f64/int coords, §2.9). Physics queries
  (ground height, slope, simple collision) sample these directly near the player (LOD 0/1 window).
- **Cosmetic GPU detail is non-physical** by construction — collision uses only authoritative fields,
  so the player never collides with something they can't deterministically agree on.
- **Contextual CPU layers** (scatter, settlements, rivers, resources) read these same regional
  authoritative fields. No GPU→CPU readback, no duplicate-noise parity problem (one CPU
  implementation is the authority; the GPU bake merely samples uploaded copies + adds cosmetic detail).

## 8. Editor & visualization

- **World Generator panel** (`register_panel`): a **master worldgen enable** (drives the scene live;
  placed `.scene` edits compose on top as overrides), the active **world recipe** selector, then
  layers in dependency order — per layer an enable toggle, auto-generated `Reflect` param sliders, and
  a **visualize** toggle.
- **Per-layer visualization**:
  - **2D map** — render the layer's artifact (or a slice of a 3D artifact) into a dockable panel /
    viewport overlay: continents colored, height ramp, climate heatmaps, biome ids, river/road graphs.
  - **3D overlay** — the layer's actual contribution to the baked terrain (isolate one layer).
- **Live re-gen**: editing a param dirties that layer's resident artifact region → cascades to
  dependents → re-gen → re-bake (reuse the existing dirty/rebake plumbing).
- **Biome editor panel**: author/save `BiomeDef` RON (envelope, realm, height, surface rules +
  blend widths, features) — the "publish new biomes in the editor" feature.
- **Resource editor panel** (§5.2): author/save `ResourceDef` RON + visualize spawn distribution.
- **POI editor panel** (§5.1): author/save `POIDef` RON (kind, context, tier/satellites) + visualize placements.
- **Prefab placement debug**: visualize instance points + bounds.
- **Macro world-map panel**: a zoomed-out world-scale 2D map (continents/biomes/climate/rivers/POIs)
  with click-to-teleport and toggleable authoring overlays — the "visualize continents" view.
- **Dynamics controls**: time-of-day scrubber (day-night) and weather-state override for testing.

---

## 9. Phasing

1. **Vertical slice** — Layer framework skeleton (`LayerManager` + tiered chunks + 1 **CPU
   authoritative** height layer) → upload height artifact → generalized `Terrain` primitive in bake
   samples it (+ cosmetic GPU detail) → noise terrain renders with one hardcoded biome material.
   *Proves the CPU-authoritative-field → artifact upload → bake data path end-to-end.* **Stand up the
   parity harness here** (§2.8) — the first authoritative layer is the moment to pin reference vectors.
2. **Erosion filter** — the runevision analytical filter (§3.1) stacked on base height (+ gradient
   exposure). Dramatically improves terrain looks early, cheaply; biome-parameterized later.
3. **Climate + overworld biomes** — hybrid climate (latitude+lapse+noise) field layers; 2D biome
   classification; `BiomeDef` RON + registry; rule-based surface materials with slope/altitude blending.
4. **Editor** — World Generator panel, per-layer 2D/3D visualization, biome editor panel, live re-gen.
5. **Underworld + caves** — 3D classification + depth-blend transition; noise caverns + worm carvers
   + surface-breaching entrances.
6. **Unified instancing + scatter** — the `InstanceStream` path end-to-end: scatter trees (Bevy mesh) /
   rocks (SDF) by biome rules, place single prefabs. Reference+transform injection; mesh-instance path.
7. **Resources** — `ResourceDef` registry + RON (§5.2); authoritative distribution layer (depth/biome
   veins + surface nodes); resource editor panel.
8. **Water pass** — dedicated screen-space water at sea level (transparency/refraction/shoreline).
8b. **Deep-ocean realm** — ocean basins/trenches in the height field, `Underwater` biomes, and the
   **underwater atmosphere** mode (§6.4, Subnautica-style). Follows the water pass.
9. **Surface overlays** — bake-time snow/wetness/sand material modifiers (§4.3); cheap, big payoff.
10. **Contextual CPU layers** — rivers (guided by erosion gullies → carve + water surface), roads;
   disk-cache where worth it.
11. **Collision** — expose the CPU authoritative coarse height/density field to physics near the
   player (no GPU readback; it already lives CPU-side per §2.8).
12. **Atmosphere + day-night** — scattering + aerial perspective + biome-varying fog (§6.1); dynamic
   sun/time-of-day (§6.3); wire both into DDGI.
13. **POI system** — general `POIDef`-driven placement (§5.1); settlements as a tiered POI type with
   roads/satellites; dungeons/ruins/landmarks as other POI types; POI editor panel.
14. **Ground cover** — GPU-instanced biome-driven grass (§6.2).
15. **Weather** — dynamic states modulating overlays/atmosphere/clouds (§6.3).
16. **Polish** — weirdness fidelity, pre-gen tuning, disk-cache invalidation, perf, world-recipe/seed UI,
   volumetric clouds integration.

---

## 10. Pitfalls / invariants

- **Cross-platform bit-determinism (authoritative layers)**: shared-seed multiplayer means
  authoritative gen must match across GPU vendors/CPUs/OSes. No GPU floats, no fast-math/FMA
  reordering, no platform-dependent transcendentals in authoritative paths. Pick a **portable noise
  basis** (integer-hash value/simplex, fixed-point where needed) and parity-test it. Cosmetic GPU
  detail is exempt.
- **Authority leak**: never let a gameplay-relevant decision depend on a GPU-only/cosmetic value. If
  collision/biome/placement reads it, it must be CPU-authoritative.
- **Order/streaming independence**: every layer pure `f(chunk_coord, seed)`, world-anchored toroidal
  addressing — no frame/camera/order dependence. (The reverted variable-rate lesson, at gen time:
  low-frequency streamed artifacts are fine; view-dependent values shimmer.)
- **CPU/GPU float-floor parity** for world→cell mapping (same trap as the light grid — plain float
  floor, not integer `floor_div`).
- **Padding correctness**: a layer must only read neighbor artifacts within its declared padding, or
  determinism breaks at region edges.
- **LOD = sampling choice**: continuous fields sampled at the consumer's LOD (some flagged
  always-full-detail); discrete/placement layers are culled, never downsampled; biome borders snapped
  stable across LOD (no popping).
- **Build constraint** (standing): all cargo commands use
  `$env:CARGO_TARGET_DIR='D:\Projects\bevy-setup\.claude\worktrees\editor-improvements\target\claude'`
  and `--features editor`. Only Rust *source* changes force rebuilds; no dep churn.

---

## 11. Implementation decisions

### Decided
- **Palette blend weights**: **extend the brick/palette to carry per-material weights** for faithful
  rule-based terrain blends (slope/altitude/biome), rather than the distance-field argmin approximation.
- **Instance residency**: **reference + transform composed at bake** (store `{template_ref, transform}`;
  bake composes template edits × transform) — memory-light at scale, cheap instance changes.
- **Portable noise basis**: **integer-hash value/gradient noise (fixed-point where needed)** for
  authoritative layers — fully portable, bit-deterministic. Cosmetic GPU detail may use floats freely.
- **Depth axis**: **depth-below-surface (near-surface) + absolute-Y bands (deep realms), blended** —
  consistent "N m down" biomes plus absolute strata, well-defined under unbounded vertical.

### Still open (non-blocking — redline)
- **3D artifact budget**: clipmap extent + resolution per 3D field; which fields are truly 3D vs
  2D-with-vertical-gradient (cheaper).
- **Authoritative-field resolution per tier**: cell sizes for the CPU height/climate/biome artifacts
  and how finely the bake samples before synthesizing cosmetic detail.
- **Water-body field representation**: single global sea level vs per-body elevation map (lakes at
  altitude) — affects the water pass input and river/lake integration.
- **Diff-volume management** (full destructibility): region compaction/coalescing/budgets so a
  heavily-edited multiplayer world's diff set stays bounded.
