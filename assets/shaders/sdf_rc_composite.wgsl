// GI pass for the screen-space radiance-cascade GI (three-rc port; Cody Bennett,
// github.com/CodyJasonBennett/three-rc).
//
// Computes ONLY the radiance-cascade indirect+ambient term and writes it to its own texture: a
// 4-probe depth-aware (bilateral) gather of cascade 0, each direction through the full Frostbite
// BRDF. The cascade carries soft sky ambient + emissive bounce with correct occlusion, so this
// single term IS the ambient + indirect lighting. The analytic sun + emissive + the spatial
// bilateral blur of THIS output are applied later in the combine pass (sdf_rc_combine.wgsl) — the
// GI is isolated on its own texture so it can be blurred without smearing the sharp direct light.
//
// Standalone: declares its OWN camera uniform + G-buffer/cascade bind groups (importing
// `sdf::bindings` would drag the whole atlas layout in). Only binding-free helpers are imported.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import sdf::oct::oct_decode
#import sdf::brdf::frostbite_brdf

// Mirror of `SdfCameraData` (render.rs). Reuses the SAME per-view uniform buffer (dynamic offset),
// so the byte layout must match exactly even though this pass reads only a few fields.
struct CompositeCamera {
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

@group(0) @binding(0) var<uniform> camera: CompositeCamera;

// G-buffer (written by sdf_raymarch.wgsl's MRT output). All Rgba16Float, non-filtering sampler.
@group(1) @binding(0) var gbuf_albedo: texture_2d<f32>;     // rgb = albedo, a = camera distance
@group(1) @binding(1) var gbuf_normal_mat: texture_2d<f32>; // rg = octN, b = metal, a = rough
@group(1) @binding(2) var gbuf_emissive: texture_2d<f32>;   // rgb = emissive, a = sun visibility
@group(1) @binding(3) var gbuf_sampler: sampler;
// Cascade 0 (finest radiance cascade). Each probe occupies a C0_RAYS_PER_DIM² block of direction
// bins laid out as `dir * probeGrid + probe`; this pass does the per-pixel bilinear gather.
@group(1) @binding(4) var cascade0: texture_2d<f32>;

// Must match `SKY_DIST` in sdf_raymarch.wgsl: albedo.a >= this ⇒ sky/miss pixel.
const SKY_DIST: f32 = 1e8;
// Must match `PROBE_SPACING` in sdf_rc_cascade.wgsl: cascade-0 probe tile edge in pixels (the GI
// spatial resolution). Independent of the per-probe DIRECTION count below.
const PROBE_SPACING: i32 = 2;
// Cascade-0 direction bins per axis = 1<<(0+2) = 4 (rays_per_dim at level 0). This is the
// direction-loop extent in the gather — NOT PROBE_SPACING (the two only coincided when spacing=4).
const C0_RAYS_PER_DIM: i32 = 4;

fn bilinear_weights(ratio: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(
        (1.0 - ratio.x) * (1.0 - ratio.y),
        ratio.x * (1.0 - ratio.y),
        (1.0 - ratio.x) * ratio.y,
        ratio.x * ratio.y,
    );
}

fn bilinear_offset(i: i32) -> vec2<i32> {
    var offs = array<vec2<i32>, 4>(
        vec2<i32>(0, 0), vec2<i32>(1, 0), vec2<i32>(0, 1), vec2<i32>(1, 1),
    );
    return offs[i];
}

// 4-probe bilinear gather of cascade 0, each direction through the Frostbite BRDF. This is the
// indirect + ambient lighting (the cascade already includes the sky dome + emissive bounce with
// occlusion). Returns the outgoing radiance contribution (BRDF already applied — NOT multiplied
// by albedo again at the call site).
// Cascade 0 resolves radiance over only PROBE_SPACING² = 16 directions on the sphere — a very
// coarse angular field. A narrow specular lobe (low roughness) sampled against 16 fixed bins
// aliases hard: the lobe lands on one bin (bright speckle) or between bins (near-zero), and which
// bin wins flickers across pixels → scattered specular points, worst on smooth metals (no diffuse
// to mask it). The cascade simply doesn't CONTAIN sharp-reflection detail. So the GI gather floors
// roughness to a value whose lobe spans several bins (smooth integration); sharp glossy highlights
// still come through the ANALYTIC sun (which isn't sampled from the cascade and keeps true
// roughness). This trades an unrepresentable mirror-GI for a stable rough-environment GI.
const GI_ROUGHNESS_FLOOR: f32 = 0.5;

fn gather_gi(
    uv: vec2<f32>, pixel_dist: f32, v: vec3<f32>, n: vec3<f32>,
    albedo: vec3<f32>, roughness: f32, metallic: f32, f0: vec3<f32>,
) -> vec3<f32> {
    let gi_rough = max(roughness, GI_ROUGHNESS_FLOOR);
    let res = vec2<i32>(camera.screen_params.xy);
    let probe_grid = (res + vec2<i32>(PROBE_SPACING - 1)) / vec2<i32>(PROBE_SPACING);

    // 2×2 nearest probes (three-rc: probeCoord = uv*grid - 0.5).
    let probe_coord = uv * vec2<f32>(probe_grid) - 0.5;
    let probe_base = vec2<i32>(floor(probe_coord));
    let bilinear = bilinear_weights(fract(probe_coord));

    // DEPTH-AWARE (bilateral) gather. Plain 2D bilinear blends a probe into this pixel even when
    // the probe sits on a DIFFERENT surface (across a depth edge): a box probe leaks onto the
    // floor behind it, and as the camera moves the screen-space probe slides across surface
    // boundaries → big patches of GI flip → the GI "swims" + smears. Multiplying each probe's
    // bilinear weight by a similarity in camera distance suppresses probes that aren't on this
    // pixel's surface, so the gather tracks the pixel's real geometry (three-rc does the same 3D-
    // aware weighting in its cascade MERGE; this brings it to the composite gather). The falloff
    // scale is relative to the pixel's distance so it's resolution/scene-scale independent.
    let depth_scale = max(pixel_dist * 0.05, 0.05);

    var radiance = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var p = 0; p < 4; p = p + 1) {
        let probe = clamp(probe_base + bilinear_offset(p), vec2<i32>(0), probe_grid - 1);
        // The probe's surface distance (G-buffer albedo.a at the probe centre pixel).
        let probe_uv = (vec2<f32>(probe) + 0.5) / vec2<f32>(probe_grid);
        let probe_px = vec2<i32>(probe_uv * camera.screen_params.xy);
        let probe_dist = textureLoad(gbuf_albedo, probe_px, 0).a;
        let depth_w = exp(-abs(probe_dist - pixel_dist) / depth_scale);
        let w = bilinear[p] * depth_w;

        var probe_radiance = vec3<f32>(0.0);
        // C0_RAYS_PER_DIM² direction bins (cascade 0: rays_per_dim = 1<<2 = 4).
        for (var dy = 0; dy < C0_RAYS_PER_DIM; dy = dy + 1) {
            for (var dx = 0; dx < C0_RAYS_PER_DIM; dx = dx + 1) {
                let dir2d = vec2<i32>(dx, dy);
                let dir_uv = (vec2<f32>(dir2d) + 0.5) / f32(C0_RAYS_PER_DIM);
                let l = oct_decode(dir_uv);
                let interval = textureLoad(cascade0, dir2d * probe_grid + probe, 0).rgb;
                probe_radiance += frostbite_brdf(v, n, l, albedo, gi_rough, metallic, f0) * interval;
            }
        }
        radiance += probe_radiance * w;
        wsum += w;
    }
    return select(vec3<f32>(0.0), radiance / wsum, wsum > 0.0);
}

// Outputs the GI term ONLY (RGB radiance) to its own texture. The combine pass adds the analytic
// sun + emissive and bilateral-blurs this. A sky/miss pixel writes 0 (the combine passes the sky
// G-buffer straight through there).
@fragment
fn main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let albedo_d = textureSampleLevel(gbuf_albedo, gbuf_sampler, uv, 0.0);
    let dist = albedo_d.a;
    if (dist >= SKY_DIST) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let albedo = albedo_d.rgb;
    let nm = textureSampleLevel(gbuf_normal_mat, gbuf_sampler, uv, 0.0);
    let normal = oct_decode(nm.rg);
    let metallic = nm.b;
    let roughness = nm.a;

    // World position from the stored camera distance + this pixel's ray → the view vector.
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    let world_near = camera.inv_view_proj * ndc;
    let ray_dir = normalize(world_near.xyz / world_near.w - camera.camera_pos.xyz);
    let view = -ray_dir;

    // Dielectric F0 0.04 (ior 1.5); metals take their albedo as F0.
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    let gi = gather_gi(uv, dist, view, normal, albedo, roughness, metallic, f0);
    return vec4<f32>(gi, 1.0);
}
