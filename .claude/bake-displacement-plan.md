# Bake-time Height Displacement — Plan

## Goal
Move height-map relief from a per-pixel GPU march into the **baked SDF field**. At bake
time, subtract `(h-0.5)*relief_depth` from each voxel's `fold_csg` distance. Coarse relief
then lives in the field — shadows/reflections see it free, no per-pixel cost, no banding.
Voxel-resolution-limited by design (~0.1u; fine cobble detail stays in the normal map).

## Part A — Gate the GPU displacement pass OFF (default)
Keep it for A/B but inert by default.
- `SdfCameraData`/`SdfCameraUniform`: add `detail_params: Vec4` (`.x` = gpu_relief flag).
  - render.rs:55 struct, bindings.wgsl:9 struct, render.rs prepare packs it.
- `SdfRaymarchParams`: add `pub gpu_relief: bool` (default `false`).
- Shader: `fn gpu_relief() -> bool` accessor; gate the two sites:
  - sdf_raymarch.wgsl:232 `if (lod==0u && d<RELIEF_MAX_BAND && gpu_relief())`
  - sdf_raymarch.wgsl:518 `if (depth>0.0 && gpu_relief())`
- Editor raymarch panel: a checkbox for `gpu_relief`.

## Part B — Bake-time displacement

### B1. CPU height cache (new resource)
`HeightImageCache { images: Vec<Option<HeightImage>> }` indexed by **global material id**.
`HeightImage { w, h, data: Arc<[f32]> }` (grayscale 0..1, the `.r` channel).
- Built/rebuilt when `MaterialRegistry` changes. New system `build_height_cache` in
  `compile.rs` (or a small new module), reads `MaterialRegistry` + `MaterialTextureLibrary`:
  - For each material `id` with `tex_layers[3] != u32::MAX` and `parallax_scale > 0`:
    resolve layer → `variants[layer].{slug,dir}` → `assets/textures/{slug}/{dir}/height.png`
    → `image::open` → `to_luma8` (or `to_rgba8().r`) → `Vec<f32>`.
  - Reuse the load pattern from textures.rs:145 `write_rgba_map`.
- Store `parallax_scale` per material alongside (or read from registry at bake).

### B2. Triplanar height sampler (CPU, mirrors GPU exactly)
`fn sample_height(cache, mat_id, world_pos, normal) -> f32` returning signed offset
`(h-0.5)*parallax_scale`:
- `TEXTURE_WORLD_SCALE = 0.5`; tile = 2 world units; bilinear sample with wrap.
- Dominant axis from `normal` (matches `relief_axis`): X→uv=(z,y), Y→(x,z), Z→(x,y).
- Bilinear interp into the height image (GPU uses linear filtering — match it).

### B3. Wire into the bake
`bake_single_brick` (atlas.rs:245) gains a `height: &HeightSampler` param.
- Per voxel: after `fold_csg(edits, world_pos)`, compute the **gradient normal** from the
  6 neighbor `fold_csg` distances (cheap; needed for the triplanar axis), look up the
  winning `material_id`, subtract the height offset:
  `d = fold_csg(...).dist - sample_height(cache, mat_id, world_pos, normal)`.
- Then `dist_to_snorm_band` as today.
- **Important**: the `dist_band` clamp and the conservative march assume a valid SDF. The
  displacement makes the field slightly non-Euclidean — keep `parallax_scale` ≤ ~1 voxel so
  the field stays close to a true distance (IQ's guidance). Document this.

Thread the sampler `Arc` through the call chain:
- `bake_brick` (atlas.rs:316) + `bake_coord` — add param.
- `BakeScheduler` holds `Arc<HeightSampler>` (built from the cache each edit/registry change).
- Async task capture (bake_scheduler.rs:427) clones the `Arc` alongside `edits`/`bvh`.
- Sync path (bake_scheduler.rs:402) + `full_bake` (mod.rs sync_bake).

### B4. Rebake trigger on material/height change
Today only `SdfMaterial.registry_id` changes rebake; `parallax_scale`/height edits don't.
- When `MaterialRegistry` changes (compile step) AND any baked material's height/scale
  differs, set `atlas.rebake_all = true` so `schedule_bakes` re-dirties resident chunks.
- Gate so unrelated registry changes (color) don't force a full rebake — track a hash of
  the (tex_height, parallax_scale) columns; rebake only when that changes.

## Tests
- Unit: `sample_height` triplanar matches the GPU pairings on a known image (axis pick + uv).
- Unit: a brick baked with a flat (h=0.5) height = identical to no-height bake (no-op proof).
- Unit: a brick baked with a ramp height shows monotonic distance shift along the ramp.
- Existing: shader validation, both builds, clippy zero-warnings.

## Verification (runtime, user-driven)
- Default: bake displacement on, GPU march off. Cobble relief visible head-on AND in
  shadows/reflections, at full frame rate, no banding.
- Toggle GPU march on in the panel to A/B (should look similar near-surface, but the GPU one
  is the only one that can exceed voxel resolution).
