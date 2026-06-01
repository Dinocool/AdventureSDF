// World-space GI irradiance cache — FILL pass (compute).
//
// Stabilizes the screen-space radiance-cascade GI, which boils on camera rotation because its
// probes are SCREEN-locked. This pass re-bins the cascade's output into a WORLD-anchored 3D
// clipmap so the final GI read (in the combine pass) is keyed to world position, not screen
// position — a given world point keeps its cache cell across frames, so rotation can't re-dice it.
//
// EXPERIMENT scope: SH-L0 (flat per-cell irradiance, normal-independent) — validates that world
// anchoring kills the boil; upgrade to SH-L1 (directional) once proven. No temporal accumulation:
// the cache is recomputed every frame; stability comes from world-fixed cell identity, not blending.
//
// One thread per cache cell. The cell's world centre is projected to screen; if it lands on the
// visible surface there (depth/identity test), we gather cascade-0's radiance bins at that probe
// and store their mean × validity. Off-screen / sky / depth-mismatch cells store validity 0 (the
// combine's validity-weighted trilinear read ignores them, so no leak/black).
//
// Standalone (own camera uniform) — importing sdf::bindings would pull the atlas into group 1.

// Mirror of `SdfCameraData` (render.rs). Same per-view buffer (dynamic offset), bytes must match.
struct CacheCamera {
    inv_view_proj: mat4x4<f32>,
    clip_from_world: mat4x4<f32>,
    prev_clip_from_world: mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen_params: vec4<f32>,   // xy = screen size
    grid_origin: vec4<f32>,     // w = voxel_size
    grid_dims: vec4<f32>,       // z = brick_size (cell_stride = z - 1)
    debug_params: vec4<f32>,
    march_params: vec4<f32>,
    lod_params: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CacheCamera;

@group(1) @binding(0) var gbuf_albedo: texture_2d<f32>;   // a = camera distance to the surface
@group(1) @binding(1) var cascade0: texture_2d<f32>;      // finest radiance cascade (2× screen)
@group(1) @binding(2) var gi_cache: texture_storage_3d<rgba16float, write>;  // rgb = irr×valid, a = valid

// MUST match render.rs SDF_GI_CACHE_RES and the combine pass: the 3D clipmap is RES³ cells.
const CACHE_RES: i32 = 64;
// Must match SKY_DIST in sdf_raymarch.wgsl / the other passes.
const SKY_DIST: f32 = 1e8;
// Must match sdf_rc_cascade.wgsl / sdf_rc_composite.wgsl probe layout.
const PROBE_SPACING: i32 = 2;
const C0_RAYS_PER_DIM: i32 = 4;   // cascade 0: 4×4 = 16 octahedral direction bins

// Cell edge in world units = one LOD-0 brick = voxel_size · cell_stride. Derived from the camera
// uniform so the fill and the combine compute it identically (no separate uniform → no drift).
fn cell_size() -> f32 {
    return camera.grid_origin.w * (camera.grid_dims.z - 1.0);
}

// World-fixed clipmap corner cell: the camera's cell minus half the window. Steps in whole cells
// as the camera moves (round(world/cell) is a world-anchored lattice), so the cache never slides.
fn clipmap_origin_cell(cs: f32) -> vec3<i32> {
    return vec3<i32>(floor(camera.camera_pos.xyz / cs)) - vec3<i32>(CACHE_RES / 2);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = vec3<i32>(gid);
    if (any(cell >= vec3<i32>(CACHE_RES))) {
        return;
    }

    let cs = cell_size();
    let origin = clipmap_origin_cell(cs);
    // World centre of this cell.
    let cell_centre = (vec3<f32>(origin + cell) + 0.5) * cs;

    // Project to screen. Behind the camera or off-screen → invalid.
    let clip = camera.clip_from_world * vec4<f32>(cell_centre, 1.0);
    if (clip.w <= 0.0) {
        textureStore(gi_cache, cell, vec4<f32>(0.0));
        return;
    }
    let ndc = clip.xyz / clip.w;
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) {
        textureStore(gi_cache, cell, vec4<f32>(0.0));
        return;
    }
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
    let res = camera.screen_params.xy;
    let px = vec2<i32>(uv * res);

    // The G-buffer surface at that pixel. Sky / no surface → invalid.
    let dist = textureLoad(gbuf_albedo, px, 0).a;
    if (dist >= SKY_DIST) {
        textureStore(gi_cache, cell, vec4<f32>(0.0));
        return;
    }
    // Reconstruct the surface world position and check it actually sits in THIS cell (else the
    // cell is in empty space / occluded → its projection hit some other surface).
    let ndc_far = vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
    let wn = camera.inv_view_proj * ndc_far;
    let ray_dir = normalize(wn.xyz / wn.w - camera.camera_pos.xyz);
    let surface = camera.camera_pos.xyz + ray_dir * dist;
    if (length(surface - cell_centre) > cs) {
        textureStore(gi_cache, cell, vec4<f32>(0.0));
        return;
    }

    // Gather cascade-0's radiance bins at the probe covering this pixel; the mean is the cell's
    // flat (SH-L0) irradiance. Same probe/texel layout the composite uses.
    let probe_grid = (vec2<i32>(res) + vec2<i32>(PROBE_SPACING - 1)) / vec2<i32>(PROBE_SPACING);
    let probe = clamp(px / PROBE_SPACING, vec2<i32>(0), probe_grid - 1);
    var acc = vec3<f32>(0.0);
    for (var dy = 0; dy < C0_RAYS_PER_DIM; dy = dy + 1) {
        for (var dx = 0; dx < C0_RAYS_PER_DIM; dx = dx + 1) {
            acc += textureLoad(cascade0, vec2<i32>(dx, dy) * probe_grid + probe, 0).rgb;
        }
    }
    let irradiance = acc / f32(C0_RAYS_PER_DIM * C0_RAYS_PER_DIM);

    // Premultiplied by validity (1) so the combine's trilinear read does a validity-weighted
    // average: trilinear(rgb)/trilinear(a) ignores the invalid (0,0,0,0) cells.
    textureStore(gi_cache, cell, vec4<f32>(irradiance, 1.0));
}
