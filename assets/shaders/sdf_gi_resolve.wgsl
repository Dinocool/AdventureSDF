// SDF GI RESOLVE pass: compute the DDGI indirect irradiance per screen pixel into a texture, so it can
// be edge-aware blurred (the next pass) before being composited. This is the `sample_gi` that used to
// live inline in `sdf_deferred_lit.wgsl` — moved here so the low-frequency probe field can be screen-
// space denoised (the probe lattice is too coarse to look clean on bare walls otherwise).
//
// Output: rgb = indirect irradiance × DdgiParams.intensity (already scaled, so the blur + composite stay
// simple), a = camera distance (the blur uses it as the depth edge-stop signal). Sky/miss pixels = 0.
//
// Bind groups mirror the old combine GI path: 0 = camera, 1 = atlas (probe world→slot lookup via
// chunk_buf), 2 = G-buffer (normal + distance), 3 = probe irradiance + params.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import sdf::bindings::{camera, lod_count, cell_stride, voxel_size_at, floor_div, euclid_mod}
#import sdf::oct::{oct_decode, oct_encode}
#import sdf::probe::{probe_slot_at, PROBE_OCT_RES, PROBE_OCT_TEXELS}

struct ProbeParams {
    ray_count: u32,
    hysteresis: f32,
    intensity: f32,
    frame: u32,
    subdiv: u32,
    update_stride: u32,
    gi_range: f32,
    normal_bias: f32,
    view_bias: f32,
    sky_intensity: f32,  // unused here; kept so the layout matches the shared ProbeParams buffer
    bounce_shadows: f32, // unused here; kept so the layout matches the shared ProbeParams buffer
};

@group(2) @binding(0) var gbuf_albedo: texture_2d<f32>;     // a = camera distance
@group(2) @binding(1) var gbuf_normal_mat: texture_2d<f32>; // rg = octN
@group(2) @binding(2) var gbuf_emissive: texture_2d<f32>;   // (unused here; kept for layout parity)
@group(2) @binding(3) var gbuf_sampler: sampler;

@group(3) @binding(0) var<storage, read> irradiance: array<vec4<f32>>;
@group(3) @binding(1) var<uniform> probe_params: ProbeParams;

const SKY_DIST: f32 = 1e8;

// Sample probe `slot`'s octahedral IRRADIANCE map in direction `n` (bilinear, clamped at tile edges).
// Returns rgb (sqrt-encoded) + alpha (validity).
fn probe_oct_sample(slot: u32, n: vec3<f32>) -> vec4<f32> {
    let base = slot * PROBE_OCT_TEXELS;
    let res = f32(PROBE_OCT_RES);
    let uv = oct_encode(n) * res - vec2<f32>(0.5);
    let maxc = i32(PROBE_OCT_RES) - 1;
    let i0 = clamp(vec2<i32>(floor(uv)), vec2<i32>(0), vec2<i32>(maxc));
    let i1 = min(i0 + vec2<i32>(1), vec2<i32>(maxc));
    let f = clamp(uv - vec2<f32>(i0), vec2<f32>(0.0), vec2<f32>(1.0));
    let t00 = irradiance[base + u32(i0.y) * PROBE_OCT_RES + u32(i0.x)];
    let t10 = irradiance[base + u32(i0.y) * PROBE_OCT_RES + u32(i1.x)];
    let t01 = irradiance[base + u32(i1.y) * PROBE_OCT_RES + u32(i0.x)];
    let t11 = irradiance[base + u32(i1.y) * PROBE_OCT_RES + u32(i1.x)];
    return mix(mix(t00, t10, f.x), mix(t01, t11, f.x), f.y);
}

// Directional indirect irradiance at `world_pos` for a surface facing `normal`: trilinearly blend the 8
// surrounding sub-probes' octahedral maps (each toward `normal`), weighted by trilinear position ×
// validity × backface wrap. CROSS-LOD FADE: instead of hard-returning at the first LOD with any probe
// (which seams where neighbouring pixels resolve to different LODs), accumulate fine→coarse weighted by
// each LOD's remaining trilinear COVERAGE — a fine LOD that fully covers a pixel uses it alone; a
// partially-covered edge smoothly fills in from the coarser LOD. Stored sqrt (perceptual) space; the
// per-LOD irradiances are blended there and squared back to linear once at the end.
fn sample_gi(world_pos: vec3<f32>, normal: vec3<f32>, view: vec3<f32>) -> vec3<f32> {
    let nlods = lod_count();
    let s = cell_stride();
    let subdiv = max(probe_params.subdiv, 1u);
    let nsub = subdiv * subdiv * subdiv;
    let sd = i32(subdiv);
    var acc_hv = vec3<f32>(0.0); // accumulated sqrt-space irradiance across LODs
    var have = 0.0;             // accumulated coverage confidence in [0,1]
    for (var l = 0u; l < nlods; l = l + 1u) {
        let cell = f32(s) * voxel_size_at(l) / f32(subdiv);
        let p = world_pos + (normal * probe_params.normal_bias + view * probe_params.view_bias) * cell;
        let g = p / cell - vec3<f32>(0.5);
        let base = floor(g);
        // Smoothstep the trilinear fraction → C1-continuous across cells (no slope kink at boundaries).
        let f0 = g - base;
        let f = f0 * f0 * (vec3<f32>(3.0) - 2.0 * f0);
        let gi0 = vec3<i32>(base);
        var sum = vec3<f32>(0.0);
        var wsum = 0.0;
        var tricov = 0.0; // sum of trilinear weights over PRESENT corners = covered fraction [0,1]
        for (var c = 0u; c < 8u; c = c + 1u) {
            let off = vec3<i32>(i32(c & 1u), i32((c >> 1u) & 1u), i32((c >> 2u) & 1u));
            let gc = gi0 + off;
            let bli = vec3<i32>(floor_div(gc.x, sd), floor_div(gc.y, sd), floor_div(gc.z, sd));
            let sub = vec3<i32>(euclid_mod(gc.x, sd), euclid_mod(gc.y, sd), euclid_mod(gc.z, sd));
            let base_slot = probe_slot_at(bli * s, l);
            if (base_slot >= 0) {
                let sub_lin = u32(sub.z) * subdiv * subdiv + u32(sub.y) * subdiv + u32(sub.x);
                let pslot = u32(base_slot) * nsub + sub_lin;
                if ((pslot + 1u) * PROBE_OCT_TEXELS <= arrayLength(&irradiance)) {
                    let probe = probe_oct_sample(pslot, normal);
                    if (probe.a > 0.5) {
                        let tri = max(mix(1.0 - f.x, f.x, f32(off.x)), 0.001)
                            * max(mix(1.0 - f.y, f.y, f32(off.y)), 0.001)
                            * max(mix(1.0 - f.z, f.z, f32(off.z)), 0.001);
                        let probe_center = (vec3<f32>(gc) + vec3<f32>(0.5)) * cell;
                        let to_probe = probe_center - world_pos;
                        let wrap = max(dot(normalize(to_probe), normal) * 0.5 + 0.5, 0.0);
                        var w = tri * (wrap * wrap + 0.2);
                        if (w < 0.2) {
                            w = w * (w * w) * (1.0 / (0.2 * 0.2));
                        }
                        sum += w * max(probe.rgb, vec3<f32>(0.0));
                        wsum += w;
                        tricov += tri;
                    }
                }
            }
        }
        if (wsum > 1e-4) {
            let hv = sum / wsum;                      // this LOD's sqrt-space irradiance
            let take = (1.0 - have) * clamp(tricov, 0.0, 1.0);
            acc_hv += take * hv;
            have += take;
            if (have >= 0.999) {
                break; // fully covered by finer LODs — no need to read coarser
            }
        }
    }
    if (have > 1e-4) {
        let hv = acc_hv / have;
        return hv * hv; // square back to linear irradiance
    }
    return vec3<f32>(0.0);
}

@fragment
fn main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let albedo_d = textureSampleLevel(gbuf_albedo, gbuf_sampler, uv, 0.0);
    let dist = albedo_d.a;
    if (dist >= SKY_DIST) {
        return vec4<f32>(0.0, 0.0, 0.0, SKY_DIST); // sky: no GI, mark far for the blur's depth stop
    }
    let nm = textureSampleLevel(gbuf_normal_mat, gbuf_sampler, uv, 0.0);
    let normal = oct_decode(nm.rg);

    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    let world_near = camera.inv_view_proj * ndc;
    let ray_dir = normalize(world_near.xyz / world_near.w - camera.camera_pos.xyz);
    let view = -ray_dir;
    let world_pos = camera.camera_pos.xyz + ray_dir * dist;

    let gi = sample_gi(world_pos, normal, view) * probe_params.intensity;
    return vec4<f32>(gi, dist);
}
