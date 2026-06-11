// Worldgen node-preview GPU raymarch (stages 2-3).
//
// The CPU bakes the graph's height + analytic normal into `height_tex` (R = height m, GBA = normal),
// the single Graph::eval source of truth — NO noise is re-implemented here. This shader only raymarches
// that heightfield with the orbit camera in `params`, so rotating is pure-GPU (rebake only on edit/zoom).
//
// BIOME / STRATA / WATER preview (TERRAIN_MATERIALS_PLAN "Biome preview"):
//   - The CPU also bakes a small `biome_tex` (R = primary biome id, G = secondary id, B = blend) by
//     running the Stage-1 Whittaker classifier per texel — NO Whittaker logic is ported to WGSL.
//   - The flattened per-biome strata table (each biome's surface/layer/bedrock `preview_color` +
//     cumulative layer-bottom depths) arrives in the `strata` uniform — the SAME table the Stage-3
//     in-world surface shader will index. The slice cut-face + the biome map read it here.

#import bevy_pbr::forward_io::VertexOutput

// Must match GPU_STRATA_MAX_LAYERS in src/sdf_render/worldgen/biome.rs.
const STRATA_MAX_LAYERS: u32 = 6u;
// Must match BiomeId::ALL.len() (the demo biome count).
const BIOME_COUNT: u32 = 5u;

struct PreviewParams {
    eye: vec4<f32>,    // xyz = camera eye,  w = image-plane tan
    fwd: vec4<f32>,    // xyz = forward,     w = world half-extent Z (m)
    right: vec4<f32>,  // xyz = right,       w = height min (m)
    up: vec4<f32>,     // xyz = up,          w = height max (m)
    levels: vec4<f32>, // sea, snow, water-depth-ramp, water-level (m)
    flags: vec4<f32>,  // x = mode (0 = 3D orbit, 1 = 2D top-down ortho), y = halfX (m), z = halfZ (m)
    modes: vec4<f32>,  // x = biome-map on, y = slice on, z = water on, w = (unused)
    slice: vec4<f32>,  // x = axis (0=X,1=Z,2=Y), y = world-plane coord (m), (z,w) unused
};

// One biome's flattened strata column (mirror of biome::GpuStrataColumn, std140-padded).
struct StrataColumn {
    surface_color: vec4<f32>,
    layer_color: array<vec4<f32>, STRATA_MAX_LAYERS>,
    // 6 cumulative layer bottoms packed into 2 vec4 lanes (only .xyzw of [0] + .xy of [1] used here for 6).
    layer_bottom: array<vec4<f32>, 2>,
    bedrock_color: vec4<f32>,
    layer_count: u32,
    _pad: vec3<u32>,
};

struct StrataTable {
    columns: array<StrataColumn, BIOME_COUNT>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: PreviewParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var height_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var biome_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var<uniform> strata: StrataTable;

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

// Height + normal at a world XZ position (within the baked ±halfX × ±halfZ window).
fn hf_at(world_xz: vec2<f32>) -> vec4<f32> {
    let half_xz = vec2<f32>(params.flags.y, params.flags.z);
    let uv = (world_xz + half_xz) / (2.0 * half_xz);
    return sample_hf(uv);
}

// World XZ → biome-texture UV; nearest-fetch the primary biome id (R), secondary (G), blend (B).
fn biome_at(world_xz: vec2<f32>) -> vec3<f32> {
    let half_xz = vec2<f32>(params.flags.y, params.flags.z);
    let uv = clamp((world_xz + half_xz) / (2.0 * half_xz), vec2<f32>(0.0), vec2<f32>(1.0));
    let res = HF_TEX_RES;
    let px = vec2<i32>(clamp(uv * (res - 1.0), vec2<f32>(0.0), vec2<f32>(res - 1.0)));
    let s = textureLoad(biome_tex, px, 0);
    return s.xyz;
}

// The cumulative bottom-depth of layer `i` (i < layer_count), unpacking the 2-vec4 packed array.
fn strata_bottom(col: StrataColumn, i: u32) -> f32 {
    let v = col.layer_bottom[i / 4u];
    let lane = i % 4u;
    if (lane == 0u) { return v.x; }
    if (lane == 1u) { return v.y; }
    if (lane == 2u) { return v.z; }
    return v.w;
}

// Walk one biome's strata column for `depth` (m below the original surface) → its `preview_color`.
// Mirror of biome::strata_material + preview_color (the Stage-3 / CPU SSOT).
fn strata_color_for(biome: u32, depth: f32) -> vec3<f32> {
    let b = min(biome, BIOME_COUNT - 1u);
    let col = strata.columns[b];
    if (depth <= 0.0) { return col.surface_color.rgb; }
    let n = min(col.layer_count, STRATA_MAX_LAYERS);
    for (var i = 0u; i < n; i = i + 1u) {
        if (depth < strata_bottom(col, i)) {
            return col.layer_color[i].rgb;
        }
    }
    return col.bedrock_color.rgb;
}

// Surface biome colour (depth 0) at a world XZ, blending primary↔secondary by the baked blend weight so
// boundaries read smoothly (matches the CPU classifier's BiomeSample.blend intent).
fn biome_surface_color(world_xz: vec2<f32>) -> vec3<f32> {
    let s = biome_at(world_xz);
    let prim = u32(s.x + 0.5);
    let sec = u32(s.y + 0.5);
    let blend = clamp(s.z, 0.0, 1.0);
    let cp = strata_color_for(prim, 0.0);
    let cs = strata_color_for(sec, 0.0);
    // blend → 1 at a border halves toward the neighbour (0.5 max mix so the primary still dominates).
    return mix(cp, cs, blend * 0.5);
}

// Slab ray–AABB → (tmin, tmax); .z < 0 means miss.
fn ray_box(o: vec3<f32>, d: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec3<f32> {
    // Safe inverse: a zero (axis-aligned) direction component would make 1/d = ±inf and then 0*inf = NaN
    // in the slab products. Floor |d| to a tiny epsilon, keeping the original sign (and +eps for exact 0),
    // so the slab for that axis is effectively unbounded instead of NaN.
    let eps = vec3<f32>(1e-6);
    let safe_d = select(d, max(abs(d), eps) * sign(d + vec3<f32>(1e-30)), abs(d) < eps);
    let inv = 1.0 / safe_d;
    let t1 = (bmin - o) * inv;
    let t2 = (bmax - o) * inv;
    let tmn = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tmx = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    if (tmx >= max(tmn, 0.0)) { return vec3<f32>(tmn, tmx, 1.0); }
    return vec3<f32>(0.0, 0.0, -1.0);
}

// Absolute-height + sea-level colour ramp. The default (non-biome) preview colour — the CPU bake only
// writes height + normal; nothing on the CPU re-implements this ramp.
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

// The surface colour the preview paints at a world XZ for the active mode: biome-map ON ⇒ the surface
// biome's preview_color, else the height ramp. Single chooser shared by the 2D path + the 3D surface hit.
fn surface_colour(world_xz: vec2<f32>, h: f32) -> vec3<f32> {
    if (params.modes.x >= 0.5) {
        return biome_surface_color(world_xz);
    }
    return height_colour(h);
}

fn sky(ndcy: f32) -> vec3<f32> {
    let t = clamp(ndcy * 0.5 + 0.5, 0.0, 1.0);
    return mix(vec3<f32>(0.12, 0.15, 0.22), vec3<f32>(0.27, 0.36, 0.52), t);
}

// Earth cross-section at a solid point `p` (the box wall / the slice cut face): depth below the ORIGINAL
// surface → the biome's strata `preview_color` (so grass→dirt→stone→bedrock bands show). NOT the height
// ramp — this is the volumetric strata view the slice exists to reveal.
fn strata_face(p: vec3<f32>) -> vec3<f32> {
    let surf = hf_at(p.xz).x;
    let depth = surf - p.y;
    let prim = u32(biome_at(p.xz).x + 0.5);
    let col = strata_color_for(prim, depth);
    // Subtle banding lines at each layer boundary so the strata read as distinct bands even where two
    // adjacent layers share a near colour.
    let f = fract(depth / 6.0);
    let line = (1.0 - smoothstep(0.0, 0.04, f)) + smoothstep(0.96, 1.0, f);
    return col * (1.0 - 0.18 * clamp(line, 0.0, 1.0));
}

// Water on the SLICE CUT FACE. Unlike `apply_water` (top-down: tint by the surface height), a vertical
// cross-section must tint by the POINT'S OWN height `p.y` vs the level, with a HARD top edge + a crisp
// waterline — otherwise the whole column tints uniformly and the level is invisible (the reported bug).
fn apply_water_face(col: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    if (params.modes.z < 0.5) { return col; }
    let wl = params.levels.w;
    var out = col;
    let under = wl - p.y;
    if (under > 0.0) {
        let shallow = vec3<f32>(0.20, 0.45, 0.62);
        let deep = vec3<f32>(0.02, 0.10, 0.26);
        let t = clamp(under / max(params.levels.z, 1.0), 0.0, 1.0);
        // Floor alpha 0.45 so the very top of the water column reads as water immediately below the line
        // (a HARD edge, not a fade-in), deepening to mostly-water below.
        out = mix(col, mix(shallow, deep, t), clamp(0.45 + 0.45 * t, 0.0, 0.9));
    }
    // Crisp bright waterline AT the level (~1.5 px via screen-space derivative of the cut-face height).
    let w = max(fwidth(p.y) * 1.5, 1.0e-4);
    let line = 1.0 - smoothstep(0.0, w, abs(p.y - wl));
    return mix(out, vec3<f32>(0.70, 0.88, 0.98), line);
}

// Flat WATER SURFACE over the submerged terrain `bg` seen at world `xz`. Tints by water DEPTH
// (`level - surface`) — shallows light, deeps dark — and draws a crisp bright shore line where the surface
// meets the level. Sampled on the smooth top-down / water-plane projection (NOT the jumpy terrain march),
// so the line is clean (no grain) and `fwidth` gives a ~constant SCREEN-width line at every zoom.
fn water_plane(bg: vec3<f32>, xz: vec2<f32>) -> vec3<f32> {
    if (params.modes.z < 0.5) { return bg; }
    let wl = params.levels.w;
    let depth = wl - hf_at(xz).x;
    var out = bg;
    if (depth > 0.0) {
        let t = clamp(depth / max(params.levels.z, 1.0), 0.0, 1.0);
        out = mix(bg, mix(vec3<f32>(0.20, 0.45, 0.62), vec3<f32>(0.02, 0.10, 0.26), t),
                  clamp(0.25 + 0.6 * t, 0.0, 0.9));
    }
    let w = max(fwidth(depth) * 1.5, 1.0e-5); // ~1.5 px, zoom-independent
    let line = 1.0 - smoothstep(0.0, w, abs(depth));
    return mix(out, vec3<f32>(0.70, 0.88, 0.98), clamp(line, 0.0, 1.0));
}

// Is the slice plane active and is world point `p` on the HIDDEN (near) side of it? Axis 0=X,1=Z,2=Y; the
// plane coord is params.slice.y. We hide the half with coordinate < plane (the near half toward -axis).
fn slice_hidden(p: vec3<f32>) -> bool {
    if (params.modes.y < 0.5) { return false; }
    let axis = params.slice.x;
    let plane = params.slice.y;
    var c = p.x;
    if (axis >= 1.5) { c = p.y; } else if (axis >= 0.5) { c = p.z; }
    return c < plane;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let halfx = params.flags.y;
    let halfz = params.flags.z;
    let ymin = params.right.w;
    let ymax = params.up.w;
    let light = normalize(vec3<f32>(0.4, 0.85, 0.3));

    // 2D top-down field map: biome-map ON ⇒ the climate/biome map; else flat absolute-height colour.
    if (params.flags.x >= 0.5) {
        let wx = (in.uv.x * 2.0 - 1.0) * halfx;
        let wz = (in.uv.y * 2.0 - 1.0) * halfz;
        let h = hf_at(vec2<f32>(wx, wz)).x;
        var col = surface_colour(vec2<f32>(wx, wz), h);
        col = water_plane(col, vec2<f32>(wx, wz));
        return vec4<f32>(col, 1.0);
    }

    let ndcx = in.uv.x * 2.0 - 1.0;
    let ndcy = 1.0 - in.uv.y * 2.0; // uv.y grows downward → flip so +y is up
    let tan = params.eye.w;          // square fov (the preview is drawn square)
    let eye = params.eye.xyz;
    let dir = normalize(params.fwd.xyz + params.right.xyz * (ndcx * tan) + params.up.xyz * (ndcy * tan));

    let span = max(ymax - ymin, 1.0);
    let pad = span * 0.08 + 1.0;
    let bmin = vec3<f32>(-halfx, ymin - pad, -halfz);
    let bmax = vec3<f32>(halfx, ymax + pad, halfz);

    let hit = ray_box(eye, dir, bmin, bmax);
    if (hit.z < 0.0) {
        return vec4<f32>(sky(ndcy), 1.0);
    }
    var t0 = max(hit.x, 0.0);
    let t1 = hit.y;

    // SLICE: advance the entry point to where the ray crosses the clip plane so the NEAR (hidden) half is
    // skipped and the cut face is the first thing the march sees. Where the ray enters already past the
    // plane into solid terrain, that entry pixel IS the cut face (strata).
    if (params.modes.y >= 0.5) {
        let axis = params.slice.x;
        let plane = params.slice.y;
        var o = eye.x; var d = dir.x;
        if (axis >= 1.5) { o = eye.y; d = dir.y; } else if (axis >= 0.5) { o = eye.z; d = dir.z; }
        // Hidden side = coord < plane. If we'd start hidden, jump to the plane crossing (if it's ahead).
        let pe0 = eye + dir * t0;
        if (slice_hidden(pe0)) {
            if (abs(d) > 1e-6) {
                let tp = (plane - o) / d;
                if (tp > t0 && tp < t1) { t0 = tp; }
                else { return vec4<f32>(sky(ndcy), 1.0); } // plane entirely behind/ahead of the box span
            } else {
                return vec4<f32>(sky(ndcy), 1.0); // ray parallel to plane, on the hidden side → nothing
            }
        }
    }

    // Solid earth: ray ENTERS (post-slice) at/below the surface → cut face / wall → strata cross-section.
    let pe = eye + dir * t0;
    let surf0 = hf_at(pe.xz).x;
    let wl = params.levels.w;
    let water_on = params.modes.z >= 0.5;
    let cut_on = params.modes.y >= 0.5;
    if (pe.y <= surf0) {
        return vec4<f32>(apply_water_face(strata_face(pe), pe), 1.0);
    }

    // Adaptive march for the terrain backdrop; cap each step to ~2 texels HORIZONTALLY so thin ridges aren't
    // tunnelled through when viewed from the side (the classic heightfield-undersampling artefact).
    let texel = (2.0 * min(halfx, halfz)) / HF_TEX_RES;
    let hspeed = max(length(dir.xz), 1e-4);
    let max_h = 2.0 * texel / hspeed;
    let descent = max(-dir.y, 0.02);
    let min_step = max((t1 - t0) / 8192.0, 0.02);
    var t = t0;
    var a_prev = pe.y - surf0;
    var bg = sky(ndcy);
    var t_hit = t1;
    for (var i = 0; i < 512; i = i + 1) {
        if (t >= t1) { break; }
        var step = max((max(a_prev, 0.0) / descent) * 0.45, min_step);
        step = min(step, max_h);
        let tn = min(t + step, t1);
        let pn = eye + dir * tn;
        let a_n = pn.y - hf_at(pn.xz).x;
        if (a_n <= 0.0) {
            var lo = t;
            var hi = tn;
            for (var k = 0; k < 20; k = k + 1) {
                let m = (lo + hi) * 0.5;
                let pm = eye + dir * m;
                if (pm.y - hf_at(pm.xz).x > 0.0) { lo = m; } else { hi = m; }
            }
            t_hit = (lo + hi) * 0.5;
            let pm = eye + dir * t_hit;
            let s = hf_at(pm.xz);
            let n = normalize(s.yzw);
            let lamb = clamp(dot(n, light), 0.0, 1.0);
            bg = surface_colour(pm.xz, s.x) * (0.28 + 0.72 * lamb);
            break;
        }
        a_prev = a_n;
        t = tn;
    }

    // WATER. Compute the water-plane crossing unconditionally (keeps `fwidth` in water_plane in uniform
    // control flow), then apply it only where the ray actually meets open water before the terrain.
    if (water_on) {
        let safe_dy = select(dir.y, -1.0e-6, abs(dir.y) < 1.0e-6);
        let tw = (wl - eye.y) / safe_dy;
        let pw = eye + dir * tw;
        let watered = water_plane(bg, pw.xz);
        if (cut_on && pe.y <= wl) {
            // The cut passes through the open-water COLUMN (solid already returned above): the cross
            // section's water body — backdrop seen through water + the top waterline, at the cut entry.
            bg = apply_water_face(bg, pe);
        } else if (dir.y < -1.0e-6 && tw >= t0 && tw <= t_hit && hf_at(pw.xz).x < wl) {
            bg = watered; // flat water surface over the submerged terrain
        }
    }
    return vec4<f32>(bg, 1.0);
}
