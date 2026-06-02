// Deferred lit pass: the final deferred-lighting step of the SDF renderer.
//
// Reads the G-buffer and produces the lit pixel:
//   - ANALYTIC SUN: a sharp directional key light through the Frostbite BRDF, shadowed by the
//     sun-visibility the G-buffer pass marched into emissive.a.
//   - EMISSIVE: self-lit surfaces pass their radiance through.
//
// Indirect GI (the radiance cascade) has been removed; a world-anchored irradiance-probe volume
// will be added here later (see plans/sdf-ddgi-probe-volume.md) — its term will sum into `lit`
// alongside the sun + emissive.
//
// Output is LINEAR HDR; Bevy's tonemapping pass converts to display.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import sdf::oct::oct_decode
#import sdf::brdf::frostbite_brdf

struct LitCamera {
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

@group(0) @binding(0) var<uniform> camera: LitCamera;

@group(1) @binding(0) var gbuf_albedo: texture_2d<f32>;     // rgb = albedo, a = camera distance
@group(1) @binding(1) var gbuf_normal_mat: texture_2d<f32>; // rg = octN, b = metal, a = rough
@group(1) @binding(2) var gbuf_emissive: texture_2d<f32>;   // rgb = emissive, a = sun visibility
@group(1) @binding(3) var gbuf_sampler: sampler;

const SKY_DIST: f32 = 1e8;

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

    // --- Deferred debug views ------------------------------------------------------------------
    // Each is a `#ifdef`-gated early return visualizing one G-buffer channel. The defines are
    // toggled from the editor (debug.rs registers them); the lit pipeline rebuilds on def change
    // so these branches compile in/out. Sky pixels already returned above.
#ifdef SDF_DEBUG_ALBEDO
    return vec4<f32>(albedo, 1.0);
#endif
#ifdef SDF_DEBUG_LOD
    // The primary pass wrote the eff-LOD hue ramp into albedo; pass it straight through (unlit).
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

    let lit = direct + emissive;
    return vec4<f32>(lit, 1.0);
}
