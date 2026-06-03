// SDF brick bake compute shader (Bevy 0.18).
//
// GPU half of the hybrid CPU→GPU bake (see `BakeBackend`). The CPU owns topology (which
// bricks exist, their material palette, their stable atlas tile); this shader runs the
// per-voxel CSG eval — the work that tanked the framerate on the main thread — and writes
// each brick's 512 distance + 512×4 material texels.
//
// It does NOT write the atlas textures directly: R16Snorm / Rgba16Snorm are not WGSL
// storage formats. Instead it writes two storage BUFFERS in the exact byte layout the atlas
// textures expect, and the bake node `copy_buffer_to_texture`s each tile's sub-rect into the
// persistent atlas — leaving the fragment reader's formats and sampling path untouched.
//
// Dispatch: one workgroup per brick job, `workgroup_id.x` = job index = the job's slot in
// both the header buffer and the output buffers. `@workgroup_size(4, 8, 1)` with an inner
// z-loop: each invocation owns an x-PAIR (x = 2·gid.x, 2·gid.x+1) at row gid.y, so the two
// adjacent-x R16 distances that share one packed u32 are written by a SINGLE invocation —
// no cross-invocation write race. 32 invocations, far under the 256 limit.

// Must match `edits::GpuEdit` (Rust) field order/layout exactly.
struct GpuEdit {
    inv_model: mat4x4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
    tag: u32,
    op_kind: u32,
    smoothing: f32,
    material_id: u32,
}

// Must match `bake_scheduler::GpuJobHeader` upload order. `coord` is the brick's
// stride-aligned origin on its LOD lattice; world voxel pos = (coord + voxel) · voxel_size.
struct JobHeader {
    coord: vec3<i32>,
    voxel_size: f32,
    dist_band: f32,
    edit_start: u32,
    edit_count: u32,
    pal01: u32,   // palette[0] | palette[1]<<16  (u16 global material ids)
    pal23: u32,   // palette[2] | palette[3]<<16
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> headers: array<JobHeader>;
@group(0) @binding(1) var<storage, read> edits: array<GpuEdit>;
// Distance output. Two R16 snorm texels per u32 (low = even x, high = odd x). Rows padded to
// DIST_ROW_U32 (= 256-byte copy alignment); only the first 32 u32 of each row carry data.
@group(0) @binding(2) var<storage, read_write> dist_out: array<u32>;
// Material output. One Rgba16Snorm texel = 2 u32 (u32_0 = r|g<<16, u32_1 = b|a<<16).
@group(0) @binding(3) var<storage, read_write> mat_out: array<u32>;

const EDGE: u32 = 8u;            // voxels per brick edge
const TILE_W: u32 = 64u;         // tile pixel width  (u = y*EDGE + x, 0..63)
const TILE_H: u32 = 8u;          // tile pixel height (v = z, 0..7)
const DIST_ROW_U32: u32 = 64u;   // padded to 256 bytes (32 real + 32 pad)
const DIST_TILE_U32: u32 = DIST_ROW_U32 * TILE_H;  // 512
const MAT_ROW_U32: u32 = TILE_W * 2u;              // 128 (no pad: 512 bytes, already aligned)
const MAT_TILE_U32: u32 = MAT_ROW_U32 * TILE_H;    // 1024

const PALETTE_K: u32 = 4u;
const PALETTE_EMPTY: u32 = 0xffffu;
const MATERIAL_FAR: f32 = 1.0;
const SNORM_CLAMP_DIST: f32 = 1.0;

const OP_UNION: u32 = 0u;
const OP_SUBTRACT: u32 = 1u;
const OP_INTERSECT: u32 = 2u;

const PRIM_SPHERE: u32 = 0u;
const PRIM_BOX: u32 = 1u;
const PRIM_TORUS: u32 = 2u;
const PRIM_CAPSULE: u32 = 3u;
const PRIM_CYLINDER: u32 = 4u;
const PRIM_HEIGHTMAP: u32 = 5u;

// --- Smooth min/max (iq polynomial) — ports edits::smin / edits::smax -----------------

fn smin(a: f32, b: f32, k: f32) -> f32 {
    if (k <= 0.0) { return min(a, b); }
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return b * (1.0 - h) + a * h - k * h * (1.0 - h);
}

fn smax(a: f32, b: f32, k: f32) -> f32 {
    if (k <= 0.0) { return max(a, b); }
    let h = clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0);
    return b * (1.0 - h) + a * h + k * h * (1.0 - h);
}

// --- Value-noise height — ports edits::hash2 / edits::height_sample --------------------

fn hash2(ix: i32, iy: i32, seed: u32) -> f32 {
    var h: u32 = u32(ix) * 374761393u;
    h = h + u32(iy) * 668265263u;
    h = h + seed * 2246822519u;
    h = h ^ (h >> 13u);
    h = h * 1274126177u;
    h = h ^ (h >> 16u);
    return (f32(h) / 4294967295.0) * 2.0 - 1.0;
}

fn height_sample(xz: vec2<f32>, freq: f32, amp: f32, seed: u32) -> f32 {
    let p = xz * freq;
    let i = floor(p);
    let f = p - i;
    let u = f * f * (vec2<f32>(3.0) - 2.0 * f);
    let ix = i32(i.x);
    let iy = i32(i.y);
    let a = hash2(ix, iy, seed);
    let b = hash2(ix + 1, iy, seed);
    let c = hash2(ix, iy + 1, seed);
    let d = hash2(ix + 1, iy + 1, seed);
    let ab = a + (b - a) * u.x;
    let cd = c + (d - c) * u.x;
    return (ab + (cd - ab) * u.y) * amp;
}

// Value-noise height AND its analytic XZ gradient (∂h/∂x, ∂h/∂z), as vec3(h, dh/dx, dh/dz). Same
// smoothstep value-noise as `height_sample` (KEEP IN SYNC with edits.rs::height_sample_grad). The
// gradient Lipschitz-normalises the heightmap field: the raw vertical gap `p.y - h` has
// |∇| = sqrt(1 + |∇h|²) ≥ 1 on slopes, so it OVER-states the true (perpendicular) distance and the
// sphere-trace overshoots steep crests; dividing by that factor restores a near-Euclidean field.
fn height_sample_grad(xz: vec2<f32>, freq: f32, amp: f32, seed: u32) -> vec3<f32> {
    let p = xz * freq;
    let i = floor(p);
    let f = p - i;
    let u = f * f * (vec2<f32>(3.0) - 2.0 * f);          // smoothstep
    let du = 6.0 * f * (vec2<f32>(1.0) - f);             // d(smoothstep)/df
    let ix = i32(i.x);
    let iy = i32(i.y);
    let a = hash2(ix, iy, seed);
    let b = hash2(ix + 1, iy, seed);
    let c = hash2(ix, iy + 1, seed);
    let d = hash2(ix + 1, iy + 1, seed);
    let ab = a + (b - a) * u.x;
    let cd = c + (d - c) * u.x;
    let h = (ab + (cd - ab) * u.y) * amp;
    // ∂(h/amp)/∂u.x and ∂(h/amp)/∂u.y, chained through smoothstep (du) and p = xz·freq.
    let dh_dx = ((b - a) + ((d - c) - (b - a)) * u.y) * du.x * freq * amp;
    let dh_dz = ((c - a) + ((d - c) - (b - a)) * u.x) * du.y * freq * amp;
    return vec3<f32>(h, dh_dx, dh_dz);
}

// --- Primitive eval — ports edits::eval_primitive (local space) ------------------------

fn eval_primitive(e: GpuEdit, p: vec3<f32>) -> f32 {
    switch (e.tag) {
        case 0u: {  // Sphere { radius }
            return length(p) - e.params.x;
        }
        case 1u: {  // Box { half_extents }
            let q = abs(p) - e.params.xyz;
            return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
        }
        case 2u: {  // Torus { major, minor } — ring in local XZ, axis Y
            let q = vec2<f32>(length(vec2<f32>(p.x, p.z)) - e.params.x, p.y);
            return length(q) - e.params.y;
        }
        case 3u: {  // Capsule { half_height, radius } — segment along local Y
            let half_height = e.params.x;
            let radius = e.params.y;
            var py = p.y;
            py = py - clamp(py, -half_height, half_height);
            return length(vec3<f32>(p.x, py, p.z)) - radius;
        }
        case 4u: {  // Cylinder { radius, half_height } — axis along local Y
            let radius = e.params.x;
            let half_height = e.params.y;
            let d = vec2<f32>(length(vec2<f32>(p.x, p.z)) - radius, abs(p.y) - half_height);
            return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
        }
        case 5u: {  // Heightmap (ONE-SIDED surface) { half_xz, max_height, freq, amp, seed }
            // Lipschitz-normalised signed distance to the noise surface — no box floor/walls, so only
            // the top shell bakes (interior + underside cull away). The raw vertical gap `p.y - h`
            // over-states the true (perpendicular) distance by sqrt(1+|∇h|²) on slopes, so the
            // sphere-trace overshoots steep crests; dividing by that factor restores a near-Euclidean,
            // march-safe field (the zero-crossing — the surface — is unchanged since the factor > 0).
            let max_height = e.params.z;
            let freq = e.params.w;
            let amp = e.params2.x;
            let seed = bitcast<u32>(e.params2.y);
            let xz = vec2<f32>(p.x, p.z);
            let hg = height_sample_grad(xz, freq, amp, seed);
            let h = hg.x + max_height * 0.5;
            // Lipschitz denominator from the NEIGHBOURHOOD-MAX slope, not the point gradient: a valid
            // bound must hold over the region a sphere-trace step can cross. The point gradient is
            // UNSAFE — at a convex peak the tip has |∇h|≈0 but its flanks are steep, so normalising by
            // the tip slope over-states the distance and the tracer over-steps into blocky "flat-pixel"
            // hits. Sample the slope half a noise cell out in ±x/±z (the feature scale is 1/freq) and
            // keep the steepest, so the tip inherits its flanks' slope. [Stefek, "Ray Marching with
            // Heightfields"; Bán & Valasek, Generalized Lipschitz Tracing, CGF 2025.]
            let r = clamp(0.5 / max(freq, 1e-4), 0.0, e.params.x);
            let gpx = height_sample_grad(xz + vec2<f32>(r, 0.0), freq, amp, seed);
            let gnx = height_sample_grad(xz - vec2<f32>(r, 0.0), freq, amp, seed);
            let gpz = height_sample_grad(xz + vec2<f32>(0.0, r), freq, amp, seed);
            let gnz = height_sample_grad(xz - vec2<f32>(0.0, r), freq, amp, seed);
            var g2 = dot(hg.yz, hg.yz);
            g2 = max(g2, dot(gpx.yz, gpx.yz));
            g2 = max(g2, dot(gnx.yz, gnx.yz));
            g2 = max(g2, dot(gpz.yz, gpz.yz));
            g2 = max(g2, dot(gnz.yz, gnz.yz));
            let lip = sqrt(1.0 + g2);
            return (p.y - h) / lip;
        }
        default: {
            return 1e30;
        }
    }
}

fn eval_world(e: GpuEdit, world_pos: vec3<f32>) -> f32 {
    let local = (e.inv_model * vec4<f32>(world_pos, 1.0)).xyz;
    return eval_primitive(e, local);
}

// --- CSG fold — ports edits::fold_csg (geometry only; material handled separately) -----

fn fold_csg(start: u32, count: u32, pos: vec3<f32>) -> f32 {
    var acc = 3.4e38;
    var started = false;
    for (var i = 0u; i < count; i = i + 1u) {
        let e = edits[start + i];
        let dn = eval_world(e, pos);
        let k = e.smoothing;
        if (!started) {
            if (e.op_kind == OP_UNION) {
                acc = dn;
                started = true;
            }
            continue;
        }
        switch (e.op_kind) {
            case 0u: { acc = smin(acc, dn, k); }            // Union
            case 1u: { acc = smax(acc, -dn, k); }           // Subtract
            default: { acc = smax(acc, dn, k); }            // Intersect
        }
    }
    return select(3.4e38, acc, started);
}

// --- snorm encode — ports SdfAtlas::dist_to_snorm{,_band} (truncating `as i16`) --------

fn snorm_bits(v: f32) -> u32 {
    let q = i32(clamp(v, -1.0, 1.0) * 32767.0);   // trunc toward 0, matches `as i16`
    return u32(q) & 0xffffu;
}

fn palette_id(pal01: u32, pal23: u32, slot: u32) -> u32 {
    switch (slot) {
        case 0u: { return pal01 & 0xffffu; }
        case 1u: { return pal01 >> 16u; }
        case 2u: { return pal23 & 0xffffu; }
        default: { return pal23 >> 16u; }
    }
}

// One voxel's 4 palette-slot material distances — ports edits::material_distances. Slot k
// holds the nearest distance of any non-subtract edit whose id == palette[k], else FAR.
fn material_slots(h: JobHeader, pos: vec3<f32>) -> vec4<f32> {
    var slots = vec4<f32>(MATERIAL_FAR);
    for (var i = 0u; i < h.edit_count; i = i + 1u) {
        let e = edits[h.edit_start + i];
        if (e.op_kind == OP_SUBTRACT) { continue; }
        // Map this edit's global id to its local palette slot, if present.
        var k = 0u;
        var found = false;
        loop {
            if (k >= PALETTE_K) { break; }
            if (palette_id(h.pal01, h.pal23, k) == e.material_id) { found = true; break; }
            k = k + 1u;
        }
        if (!found) { continue; }
        let d = eval_world(e, pos);
        if (d < slots[k]) { slots[k] = d; }
    }
    return slots;
}

// Width of the 2D workgroup dispatch grid. One workgroup per brick job; the dispatch is laid
// out 2D so the job count can exceed the 65535 single-dimension limit. MUST match
// `BAKE_DISPATCH_WIDTH` in render.rs.
const DISPATCH_WIDTH: u32 = 256u;

@compute @workgroup_size(4, 8, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let job = wid.y * DISPATCH_WIDTH + wid.x;
    if (job >= arrayLength(&headers)) { return; }
    let h = headers[job];

    let p = lid.x;          // x-pair index 0..3  → x = 2p, 2p+1
    let y = lid.y;          // 0..7
    let x_even = 2u * p;
    let x_odd = x_even + 1u;

    let dist_base = job * DIST_TILE_U32;
    let mat_base = job * MAT_TILE_U32;
    let origin = vec3<f32>(f32(h.coord.x), f32(h.coord.y), f32(h.coord.z));

    for (var z = 0u; z < EDGE; z = z + 1u) {
        let pe = (origin + vec3<f32>(f32(x_even), f32(y), f32(z))) * h.voxel_size;
        let po = (origin + vec3<f32>(f32(x_odd), f32(y), f32(z))) * h.voxel_size;

        // Distance: per-LOD band, two adjacent-x texels packed into one u32.
        let de = fold_csg(h.edit_start, h.edit_count, pe) / h.dist_band;
        let do_ = fold_csg(h.edit_start, h.edit_count, po) / h.dist_band;
        let row = z * DIST_ROW_U32 + (y * 4u + p);   // u/2 = (y*8 + 2p)/2 = y*4 + p
        dist_out[dist_base + row] = snorm_bits(de) | (snorm_bits(do_) << 16u);

        // Material: one Rgba16Snorm texel (2 u32) per voxel, fixed ±1.0 band.
        let se = material_slots(h, pe);
        let so = material_slots(h, po);
        let ue = y * EDGE + x_even;   // tile-local u for even x
        let uo = y * EDGE + x_odd;
        let mi_e = z * MAT_ROW_U32 + ue * 2u;
        let mi_o = z * MAT_ROW_U32 + uo * 2u;
        mat_out[mat_base + mi_e]      = snorm_bits(se.x) | (snorm_bits(se.y) << 16u);
        mat_out[mat_base + mi_e + 1u] = snorm_bits(se.z) | (snorm_bits(se.w) << 16u);
        mat_out[mat_base + mi_o]      = snorm_bits(so.x) | (snorm_bits(so.y) << 16u);
        mat_out[mat_base + mi_o + 1u] = snorm_bits(so.z) | (snorm_bits(so.w) << 16u);
    }
}
