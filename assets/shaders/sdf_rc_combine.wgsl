// Combine pass: the final deferred-lighting step of the radiance-cascade GI (three-rc port).
//
// Reads the G-buffer + the GI texture (the isolated radiance-cascade indirect term from
// sdf_rc_composite.wgsl) and produces the lit pixel:
//   - BILATERAL BLUR of the GI: a depth/normal-aware screen-space blur. The GI is quarter-ish
//     resolution and, being a single-frame 16-direction trace, flickers as the camera moves
//     (angular undersampling). Blurring it spatially — but only across pixels on the SAME surface
//     (similar depth + normal) so edges stay crisp — turns that flicker into a stable soft glow.
//     This is why the GI lives on its own texture: we can blur it without smearing the sharp sun.
//   - ANALYTIC SUN: a sharp directional key light through the Frostbite BRDF, shadowed by the
//     sun-visibility the G-buffer pass marched into emissive.a. NOT blurred (it must stay sharp).
//   - EMISSIVE: self-lit surfaces pass their radiance through.
//
// Output is LINEAR HDR; Bevy's tonemapping pass converts to display.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import sdf::oct::oct_decode
#import sdf::brdf::frostbite_brdf

struct CombineCamera {
    inv_view_proj: mat4x4<f32>,
    clip_from_world: mat4x4<f32>,
    prev_clip_from_world: mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen_params: vec4<f32>,
    grid_origin: vec4<f32>,
    grid_dims: vec4<f32>,
    debug_params: vec4<f32>,
    march_params: vec4<f32>,
    lod_params: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CombineCamera;

@group(1) @binding(0) var gbuf_albedo: texture_2d<f32>;     // rgb = albedo, a = camera distance
@group(1) @binding(1) var gbuf_normal_mat: texture_2d<f32>; // rg = octN, b = metal, a = rough
@group(1) @binding(2) var gbuf_emissive: texture_2d<f32>;   // rgb = emissive, a = sun visibility
@group(1) @binding(3) var gbuf_sampler: sampler;
@group(1) @binding(4) var gi_tex: texture_2d<f32>;          // rgb = per-pixel GI (cache fallback)
// World-space GI irradiance cache (filled by sdf_gi_cache_fill.wgsl). rgb = irradiance×validity,
// a = validity. Sampled trilinearly with a LINEAR sampler → validity-weighted world-cell average.
@group(1) @binding(5) var gi_cache: texture_3d<f32>;
@group(1) @binding(6) var gi_cache_sampler: sampler;

const SKY_DIST: f32 = 1e8;
// MUST match sdf_gi_cache_fill.wgsl: the world clipmap is CACHE_RES³ cells, cell = one LOD-0 brick.
const CACHE_RES: f32 = 64.0;

// Cell edge in world units (= voxel_size · cell_stride), identical to the fill pass.
fn cell_size() -> f32 {
    return camera.grid_origin.w * (camera.grid_dims.z - 1.0);
}

// World-fixed clipmap corner cell — SAME formula as the fill, so cell coords agree exactly.
fn clipmap_origin_cell(cs: f32) -> vec3<f32> {
    return floor(camera.camera_pos.xyz / cs) - vec3<f32>(CACHE_RES * 0.5);
}

// Validity-weighted trilinear sample of the world cache at a surface world position. Returns the
// flat (SH-L0) irradiance, and `validity` (0 = no valid cells nearby → caller falls back).
fn sample_gi_cache(world_pos: vec3<f32>) -> vec4<f32> {
    let cs = cell_size();
    let origin = clipmap_origin_cell(cs);
    // `coord = world/cs - origin` equals i+0.5 at cell i's centre (the fill wrote cell i to texel
    // i); /RES maps that to texel i's centre (i+0.5)/RES, so trilinear interpolates between cells.
    let uvw = (world_pos / cs - origin) / CACHE_RES;
    let s = textureSampleLevel(gi_cache, gi_cache_sampler, uvw, 0.0);
    let validity = s.a;
    let irradiance = s.rgb / max(validity, 1e-4);
    return vec4<f32>(irradiance, validity);
}

@fragment
fn main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let albedo_d = textureSampleLevel(gbuf_albedo, gbuf_sampler, uv, 0.0);
    let dist = albedo_d.a;

    // Sky / miss: the G-buffer already holds the analytic sky in rgb — pass it straight through.
    if (dist >= SKY_DIST) {
        return vec4<f32>(albedo_d.rgb, 1.0);
    }

    let albedo = albedo_d.rgb;
    let nm = textureSampleLevel(gbuf_normal_mat, gbuf_sampler, uv, 0.0);
    let normal = oct_decode(nm.rg);
    let metallic = nm.b;
    let roughness = nm.a;
    let em = textureSampleLevel(gbuf_emissive, gbuf_sampler, uv, 0.0);
    let emissive = em.rgb;
    let sun_vis = em.a;

    // GI from the WORLD-SPACE cache (world-anchored → stable under camera rotation). Reconstruct
    // this surface's world position, sample the validity-weighted world-cell irradiance, multiply
    // by albedo·(1-metallic) (SH-L0: flat, normal-independent). Where the cache has no valid cell
    // (freshly revealed surface, off-screen neighbours), fall back to the composite's per-pixel
    // screen-space GI so the surface isn't black — blended by cache validity.
    let centre_px = vec2<i32>(uv * camera.screen_params.xy);
    let ndc0 = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    let wn0 = camera.inv_view_proj * ndc0;
    let rd0 = normalize(wn0.xyz / wn0.w - camera.camera_pos.xyz);
    let world_pos = camera.camera_pos.xyz + rd0 * dist;

    let cache = sample_gi_cache(world_pos);
    let cache_gi = cache.rgb * albedo * (1.0 - metallic);
    let fallback_gi = textureLoad(gi_tex, centre_px, 0).rgb;
    let gi = mix(fallback_gi, cache_gi, clamp(cache.a, 0.0, 1.0));

    // --- Deferred debug views ------------------------------------------------------------------
    // Each is a `#ifdef`-gated early return visualizing one G-buffer / GI channel. The defines are
    // toggled from the editor (debug.rs registers them); the combine pipeline rebuilds on def
    // change so these branches compile in/out. Sky pixels already returned above.
#ifdef SDF_DEBUG_ALBEDO
    return vec4<f32>(albedo, 1.0);
#endif
#ifdef SDF_DEBUG_NORMALS
    return vec4<f32>(normal * 0.5 + 0.5, 1.0);
#endif
#ifdef SDF_DEBUG_METALLIC
    return vec4<f32>(vec3<f32>(metallic), 1.0);
#endif
#ifdef SDF_DEBUG_ROUGHNESS
    return vec4<f32>(vec3<f32>(roughness), 1.0);
#endif
#ifdef SDF_DEBUG_EMISSIVE
    return vec4<f32>(emissive, 1.0);
#endif
#ifdef SDF_DEBUG_GI_RAW
    // The unblurred GI gather (cascade-0 → BRDF), so the probe-grid structure is visible.
    return vec4<f32>(textureLoad(gi_tex, centre_px, 0).rgb, 1.0);
#endif
#ifdef SDF_DEBUG_GI
    // The bilateral-blurred GI (what the lit result actually uses).
    return vec4<f32>(gi, 1.0);
#endif
#ifdef SDF_DEBUG_SUN_VIS
    return vec4<f32>(vec3<f32>(sun_vis), 1.0);
#endif
#ifdef SDF_DEBUG_DEPTH
    // Camera distance, scaled to a readable range (mid-grey ~ a few units out).
    return vec4<f32>(vec3<f32>(dist / (dist + 8.0)), 1.0);
#endif

    // --- Analytic sun (sharp, shadowed) ---
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    let world_near = camera.inv_view_proj * ndc;
    let ray_dir = normalize(world_near.xyz / world_near.w - camera.camera_pos.xyz);
    let view = -ray_dir;
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let sun = normalize(camera.sun_dir.xyz);
    let direct = frostbite_brdf(view, normal, sun, albedo, roughness, metallic, f0)
        * camera.sun_color.rgb * sun_vis;

    let lit = direct + gi + emissive;
    return vec4<f32>(lit, 1.0);
}
