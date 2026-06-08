// Deferred lit pass: the final compositing step of the SDF renderer.
//
// ALL direct lighting (the directional sun + every point light) is done in the G-buffer pass
// (`sdf_raymarch.wgsl`) via the shared `sdf::lights::direct_light` and summed into the emissive
// channel (physical, scene-referred). So this pass just:
//   - composites that pre-lit emissive radiance and applies the camera exposure, and
//   - passes the analytic sky through (exposed) on a ray miss.
//
// Indirect GI (DDGI/probe) was removed in the mesh-bake pivot; the indirect bounce is now zero.
//
// Output is LINEAR HDR; Bevy's tonemapping pass converts to display.
//
// Bind groups: 0 = camera, 1 = G-buffer.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import sdf::bindings::camera
#import sdf::oct::oct_decode

@group(1) @binding(0) var gbuf_albedo: texture_2d<f32>;     // rgb = albedo, a = camera distance
@group(1) @binding(1) var gbuf_normal_mat: texture_2d<f32>; // rg = octN, b = metal, a = rough
@group(1) @binding(2) var gbuf_emissive: texture_2d<f32>;   // rgb = emissive + ALL direct (sun+points), a = sun vis
@group(1) @binding(3) var gbuf_sampler: sampler;

const SKY_DIST: f32 = 1e8;

@fragment
fn main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let albedo_d = textureSampleLevel(gbuf_albedo, gbuf_sampler, uv, 0.0);
    let dist = albedo_d.a;

    // Sky / miss: the G-buffer holds the analytic sky (physical luminance) in rgb. Apply the camera
    // exposure (sun_color.w) so the sky maps to display alongside the exposed surfaces.
    if (dist >= SKY_DIST) {
        return vec4<f32>(albedo_d.rgb * camera.sun_color.w, 1.0);
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
    // NOTE: this channel now carries material emissive + the G-buffer pass's sun + point-light direct.
    return vec4<f32>(emissive, 1.0);
#endif
#ifdef SDF_DEBUG_SUN_VIS
    return vec4<f32>(vec3<f32>(sun_vis), 1.0);
#endif
#ifdef SDF_DEBUG_DEPTH
    // Camera distance, scaled to a readable range (mid-grey ~ a few units out).
    return vec4<f32>(vec3<f32>(dist / (dist + 8.0)), 1.0);
#endif

    // --- Indirect: removed in the mesh-bake pivot (DDGI/probe GI gone) ---
    let gi = vec3<f32>(0.0);

    // `emissive` already holds ALL direct lighting (sun + points) + material emissive — physical,
    // scene-referred. The indirect diffuse bounce (albedo × GI) is now zero. Apply the camera
    // exposure (sun_color.w = exp2(-ev100)/1.2) to map to the display range before tonemapping.
    let lit = (emissive + albedo * gi) * camera.sun_color.w;
    return vec4<f32>(lit, 1.0);
}
