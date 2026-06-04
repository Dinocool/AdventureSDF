// SDF GI BLUR pass: one iteration of an edge-avoiding à-trous filter on the resolved GI texture. The
// driving node runs this several times with a doubling `step` (1,2,4,8,16 px), which gives a very wide
// effective blur from a cheap 5×5 kernel per pass — enough to dissolve the ~0.3 m probe-lattice blocks
// (which can be ~100 px across on a near wall) without smearing across surfaces.
//
// Edge stops: a B3-spline 5×5 spatial kernel × a DEPTH weight (relative to camera distance, so it isn't
// fooled by grazing surfaces) × a NORMAL weight (so GI doesn't bleed across a 90° crease or a silhouette).
// rgb = blurred GI, a = camera distance carried through unchanged (sky stays at SKY_DIST).

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import sdf::oct::oct_decode

struct GiBlurParams {
    inv_size: vec2<f32>,   // 1 / (width, height) in pixels
    step: f32,             // à-trous tap spacing in pixels for this pass
    depth_sigma: f32,      // relative depth tolerance (× camera distance)
    normal_power: f32,     // normal edge-stop sharpness
};

@group(0) @binding(0) var gi_in: texture_2d<f32>;          // rgb = GI, a = camera distance
@group(0) @binding(1) var gbuf_normal_mat: texture_2d<f32>; // rg = octN
@group(0) @binding(2) var samp: sampler;                    // non-filtering (taps land on texel centers)
@group(0) @binding(3) var<uniform> bp: GiBlurParams;

const SKY_DIST: f32 = 1e8;

fn normal_at(uv: vec2<f32>) -> vec3<f32> {
    return oct_decode(textureSampleLevel(gbuf_normal_mat, samp, uv, 0.0).rg);
}

@fragment
fn main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let center = textureSampleLevel(gi_in, samp, uv, 0.0);
    let cdist = center.a;
    if (cdist >= SKY_DIST) {
        return center; // sky / miss — nothing to blur, keep the far marker
    }
    let n0 = normal_at(uv);
    let kw = array<f32, 5>(1.0, 4.0, 6.0, 4.0, 1.0); // B3-spline row

    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            let suv = uv + vec2<f32>(f32(dx), f32(dy)) * bp.step * bp.inv_size;
            let s = textureSampleLevel(gi_in, samp, suv, 0.0);
            if (s.a >= SKY_DIST) {
                continue; // don't pull sky into surfaces
            }
            let spatial = kw[dx + 2] * kw[dy + 2];
            let depth_w = exp(-abs(s.a - cdist) / (bp.depth_sigma * cdist + 1.0e-4));
            let ns = normal_at(suv);
            let normal_w = pow(max(dot(ns, n0), 0.0), bp.normal_power);
            let w = spatial * depth_w * normal_w;
            sum += w * s.rgb;
            wsum += w;
        }
    }
    let gi = select(center.rgb, sum / wsum, wsum > 1.0e-4);
    return vec4<f32>(gi, cdist);
}
