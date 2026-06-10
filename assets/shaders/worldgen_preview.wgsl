// Worldgen node-preview GPU raymarch (stages 2-3).
//
// The CPU bakes the graph's height + analytic normal into `height_tex` (R = height m, GBA = normal),
// the single Graph::eval source of truth — NO noise is re-implemented here. This shader only raymarches
// that heightfield with the orbit camera in `params`, so rotating is pure-GPU (rebake only on edit/zoom).

#import bevy_pbr::forward_io::VertexOutput

struct PreviewParams {
    eye: vec4<f32>,    // xyz = camera eye,  w = image-plane tan
    fwd: vec4<f32>,    // xyz = forward,     w = world half-extent (m)
    right: vec4<f32>,  // xyz = right,       w = height min (m)
    up: vec4<f32>,     // xyz = up,          w = height max (m)
    levels: vec4<f32>, // sea, snow, water-depth, res(px)
};

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: PreviewParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var height_tex: texture_2d<f32>;

// Manual bilinear fetch (textureLoad needs no filterable format → portable for Rgba32Float).
// Baked heightfield texture resolution — MUST match HEIGHTFIELD_RES in worldgen_gpu_preview.rs.
const HF_TEX_RES: f32 = 256.0;

fn sample_hf(uv: vec2<f32>) -> vec4<f32> {
    let res = HF_TEX_RES;
    let p = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * (res - 1.0);
    let i0 = floor(p);
    let f = p - i0;
    let x0 = i32(i0.x);
    let y0 = i32(i0.y);
    let x1 = min(x0 + 1, i32(res) - 1);
    let y1 = min(y0 + 1, i32(res) - 1);
    let a = textureLoad(height_tex, vec2<i32>(x0, y0), 0);
    let b = textureLoad(height_tex, vec2<i32>(x1, y0), 0);
    let c = textureLoad(height_tex, vec2<i32>(x0, y1), 0);
    let d = textureLoad(height_tex, vec2<i32>(x1, y1), 0);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Height + normal at a world XZ position (within the baked ±half window).
fn hf_at(world_xz: vec2<f32>) -> vec4<f32> {
    let half = params.fwd.w;
    let uv = (world_xz + vec2<f32>(half)) / (2.0 * half);
    return sample_hf(uv);
}

// Slab ray–AABB → (tmin, tmax); .z < 0 means miss.
fn ray_box(o: vec3<f32>, d: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec3<f32> {
    let inv = 1.0 / d;
    let t1 = (bmin - o) * inv;
    let t2 = (bmax - o) * inv;
    let tmn = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tmx = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    if (tmx >= max(tmn, 0.0)) { return vec3<f32>(tmn, tmx, 1.0); }
    return vec3<f32>(0.0, 0.0, -1.0);
}

// Absolute-height + sea-level colour ramp (mirrors the CPU height_color_rgb).
fn land_ramp(t: f32) -> vec3<f32> {
    let beach = vec3<f32>(0.76, 0.70, 0.50);
    let grass = vec3<f32>(0.24, 0.55, 0.35);
    let hill = vec3<f32>(0.36, 0.45, 0.26);
    let rock = vec3<f32>(0.48, 0.42, 0.36);
    let snow = vec3<f32>(0.95, 0.95, 0.97);
    if (t < 0.12) { return mix(beach, grass, t / 0.12); }
    if (t < 0.45) { return mix(grass, hill, (t - 0.12) / 0.33); }
    if (t < 0.72) { return mix(hill, rock, (t - 0.45) / 0.27); }
    return mix(rock, snow, clamp((t - 0.72) / 0.28, 0.0, 1.0));
}

fn height_colour(h: f32) -> vec3<f32> {
    let sea = params.levels.x;
    let snow = params.levels.y;
    let depth = params.levels.z;
    if (h < sea) {
        let t = clamp((sea - h) / depth, 0.0, 1.0);
        return mix(vec3<f32>(0.30, 0.52, 0.68), vec3<f32>(0.05, 0.12, 0.32), t);
    }
    return land_ramp(clamp((h - sea) / (snow - sea), 0.0, 1.0));
}

fn sky(ndcy: f32) -> vec3<f32> {
    let t = clamp(ndcy * 0.5 + 0.5, 0.0, 1.0);
    return mix(vec3<f32>(0.12, 0.15, 0.22), vec3<f32>(0.27, 0.36, 0.52), t);
}

// Earth cross-section: the absolute-height ramp banded into horizontal strata, so a box wall (where the
// solid terrain is sliced by the window edge) reads as layered ground rather than a hole.
fn strata(y: f32) -> vec3<f32> {
    let base = height_colour(y);
    let f = fract(y / 60.0);
    let line = (1.0 - smoothstep(0.0, 0.06, f)) + smoothstep(0.94, 1.0, f);
    return base * (1.0 - 0.4 * clamp(line, 0.0, 1.0));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let ndcx = in.uv.x * 2.0 - 1.0;
    let ndcy = 1.0 - in.uv.y * 2.0; // uv.y grows downward → flip so +y is up
    let tan = params.eye.w;
    let aspect = params.levels.w; // width/height — widen the horizontal fov so a non-square panel fills
    let eye = params.eye.xyz;
    let dir = normalize(params.fwd.xyz + params.right.xyz * (ndcx * tan * aspect) + params.up.xyz * (ndcy * tan));

    let half = params.fwd.w;
    let ymin = params.right.w;
    let ymax = params.up.w;
    let span = max(ymax - ymin, 1.0);
    let pad = span * 0.08 + 1.0;
    let bmin = vec3<f32>(-half, ymin - pad, -half);
    let bmax = vec3<f32>(half, ymax + pad, half);

    let hit = ray_box(eye, dir, bmin, bmax);
    if (hit.z < 0.0) {
        return vec4<f32>(sky(ndcy), 1.0);
    }
    let t0 = max(hit.x, 0.0);
    let t1 = hit.y;
    let light = normalize(vec3<f32>(0.4, 0.85, 0.3));

    // Solid earth: if the ray ENTERS the box already at/below the surface, it pierced a side/bottom wall
    // → shade the earth cross-section (strata) instead of marching into a hole.
    let pe = eye + dir * t0;
    if (pe.y - hf_at(pe.xz).x <= 0.0) {
        return vec4<f32>(strata(pe.y), 1.0);
    }

    // Adaptive march, but cap each step to ~2 texels HORIZONTALLY so thin ridges aren't tunnelled through
    // when viewed from the side (the classic heightfield-undersampling artefact).
    let res = params.levels.w;
    let texel = (2.0 * half) / res;
    let hspeed = max(length(dir.xz), 1e-4);
    let max_h = 2.0 * texel / hspeed;
    let descent = max(-dir.y, 0.02);
    let min_step = max((t1 - t0) / 8192.0, 0.02);
    var t = t0;
    var a_prev = pe.y - hf_at(pe.xz).x;

    for (var i = 0; i < 512; i = i + 1) {
        if (t >= t1) { break; }
        var step = max((max(a_prev, 0.0) / descent) * 0.45, min_step);
        step = min(step, max_h);
        let tn = min(t + step, t1);
        let pn = eye + dir * tn;
        let a_n = pn.y - hf_at(pn.xz).x;
        if (a_n <= 0.0) {
            // Bisect the crossing.
            var lo = t;
            var hi = tn;
            for (var k = 0; k < 20; k = k + 1) {
                let m = (lo + hi) * 0.5;
                let pm = eye + dir * m;
                if (pm.y - hf_at(pm.xz).x > 0.0) { lo = m; } else { hi = m; }
            }
            let pm = eye + dir * ((lo + hi) * 0.5);
            let s = hf_at(pm.xz);
            let n = normalize(s.yzw);
            let lamb = clamp(dot(n, light), 0.0, 1.0);
            let col = height_colour(s.x) * (0.28 + 0.72 * lamb);
            return vec4<f32>(col, 1.0);
        }
        a_prev = a_n;
        t = tn;
    }
    return vec4<f32>(sky(ndcy), 1.0);
}
