// Deferred lit pass: the final deferred-lighting step of the SDF renderer.
//
// Reads the G-buffer (group 2) + the screen-space-denoised indirect-GI texture (group 1) and produces
// the lit pixel:
//   - ANALYTIC SUN: a sharp directional key light through the Frostbite BRDF, shadowed by the
//     sun-visibility the G-buffer pass marched into emissive.a.
//   - EMISSIVE: self-lit surfaces pass their radiance through.
//   - INDIRECT (DDGI): the probe irradiance — resolved per pixel and edge-aware blurred in the
//     gi_resolve / gi_blur passes (render/gi.rs), already scaled by intensity — composited as
//     `albedo × gi`. (Moved out of this shader so the coarse probe field can be denoised in screen
//     space; a bare probe lattice reads as blocks otherwise.)
//
// Output is LINEAR HDR; Bevy's tonemapping pass converts to display.
//
// Bind groups: 0 = camera, 1 = denoised GI texture + sampler, 2 = G-buffer.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import sdf::bindings::camera
#import sdf::oct::oct_decode
#import sdf::brdf::frostbite_brdf

@group(1) @binding(0) var gi_tex: texture_2d<f32>;   // denoised indirect irradiance (already × intensity)
@group(1) @binding(1) var gi_sampler: sampler;

@group(2) @binding(0) var gbuf_albedo: texture_2d<f32>;     // rgb = albedo, a = camera distance
@group(2) @binding(1) var gbuf_normal_mat: texture_2d<f32>; // rg = octN, b = metal, a = rough
@group(2) @binding(2) var gbuf_emissive: texture_2d<f32>;   // rgb = emissive, a = sun visibility
@group(2) @binding(3) var gbuf_sampler: sampler;

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
#ifdef SDF_DEBUG_STEP_COUNT
    // The primary pass wrote the march step-count heatmap into albedo; pass it straight through.
    // (Sky/miss pixels already returned above with the same heat, since their albedo.a = SKY_DIST.)
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

    // --- Indirect (DDGI): screen-space-denoised probe irradiance (already × intensity) ---
    let gi = textureSampleLevel(gi_tex, gi_sampler, uv, 0.0).rgb;

#ifdef SDF_DEBUG_GI
    return vec4<f32>(albedo * gi, 1.0);
#endif

    let lit = direct + emissive + albedo * gi;
    return vec4<f32>(lit, 1.0);
}
