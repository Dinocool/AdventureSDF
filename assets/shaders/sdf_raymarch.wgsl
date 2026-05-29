// SDF Raymarching Shader — Atlas-based (Bevy 0.18)
// Group 0 binding 0: SdfCameraUniform
// Group 1 binding 0: atlas texture (R8Snorm)
// Group 1 binding 1: sampler (Nearest)
// Group 1 binding 2: lookup storage buffer (sorted by brick_id)

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

// --- Structs ---

struct SdfCameraUniform {
    inv_view_proj: mat4x4<f32>,
    clip_from_world: mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen_params: vec4<f32>,
    grid_origin: vec4<f32>,
    grid_dims: vec4<f32>,
    debug_params: vec4<f32>,   // x = max_steps, y = max_dist, z = sdf_eps
    object_colors: array<vec4<f32>, 8u>,
};

struct BrickLookup {
    brick_id: u32,
    atlas_u: u32,
    atlas_v: u32,
    _pad: u32,
};

// BVH node: 32 bytes (two vec3<f32> + u32 rows). `count_or_right`'s high bit
// (0x80000000) marks an internal node — then the field is the right-child index
// and `left_or_first` is the left child. Clear high bit => leaf, where the field
// is the edit count and `left_or_first` the first edit index. The shader only
// needs the AABBs for empty-space skipping, so it ignores edit indices entirely.
struct BvhNode {
    aabb_min: vec3<f32>,
    left_or_first: u32,
    aabb_max: vec3<f32>,
    count_or_right: u32,
};

// --- Bindings ---

@group(0) @binding(0) var<uniform> camera: SdfCameraUniform;
@group(1) @binding(0) var atlas_tex: texture_2d<f32>;       // R8Snorm distance field
@group(1) @binding(1) var atlas_sampler: sampler;
@group(1) @binding(2) var<storage, read> lookup_buf: array<BrickLookup>;
@group(1) @binding(3) var mat_lo_tex: texture_2d<f32>;      // Rgba16Snorm material dist, ids 0..3
@group(1) @binding(4) var mat_hi_tex: texture_2d<f32>;      // Rgba16Snorm material dist, ids 4..7
@group(1) @binding(5) var<storage, read> bvh_buf: array<BvhNode>;  // edit-AABB BVH (empty-space skip)

fn num_bvh_nodes() -> u32 { return u32(camera.debug_params.w); }
const BVH_INTERNAL_FLAG: u32 = 0x80000000u;

// --- Raymarch params (driven live from camera.debug_params) ---

fn max_steps() -> u32 { return u32(camera.debug_params.x); }
fn max_dist() -> f32 { return camera.debug_params.y; }
fn sdf_eps() -> f32 { return camera.debug_params.z; }

// --- Brick coordinate helpers ---

// Brick spatial stride in voxels. Bricks hold `brick_size` samples (grid_dims.z,
// = 8) but span `brick_size - 1` cells (= 7) and duplicate the shared boundary
// plane (apron). Matches SdfGridConfig::cell_stride on the CPU side.
fn brick_stride() -> i32 {
    return i32(camera.grid_dims.z) - 1;
}

fn world_to_brick(world_pos: vec3<f32>) -> vec3<i32> {
    let grid_orig = camera.grid_origin.xyz;
    let voxel_size = camera.grid_origin.w;
    let s = brick_stride();
    let relative = world_pos - grid_orig;
    let vox = vec3<i32>(
        i32(relative.x / voxel_size),
        i32(relative.y / voxel_size),
        i32(relative.z / voxel_size),
    );
    return (vox / s) * s;
}

fn compute_brick_id(coord: vec3<i32>) -> u32 {
    let bpa = u32(camera.grid_dims.y);
    let s = brick_stride();
    return u32(coord.z / s) * bpa * bpa + u32(coord.y / s) * bpa + u32(coord.x / s);
}

// --- Binary search in lookup buffer ---

struct BrickLocation {
    atlas_u: u32,
    atlas_v: u32,
    found: bool,
};

// Binary search the sorted lookup buffer for a brick id.
fn find_brick_lookup(brick_id: u32) -> BrickLocation {
    let count = u32(camera.grid_dims.w);
    if (count == 0u) {
        return BrickLocation(0u, 0u, false);
    }

    var lo: i32 = 0;
    var hi: i32 = i32(count) - 1;

    while (lo <= hi) {
        let mid = (lo + hi) / 2;
        let entry = lookup_buf[u32(mid)];
        let mid_id = entry.brick_id;

        if (mid_id == brick_id) {
            return BrickLocation(entry.atlas_u, entry.atlas_v, true);
        } else if (mid_id < brick_id) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    return BrickLocation(0u, 0u, false);
}

// --- Sample brick SDF via trilinear interpolation ---

// Pixel coordinate of brick-local voxel (lx,ly,lz) within the atlas. Both the
// distance and object atlases share this layout: pixel = (base_u + ly*EDGE + lx, lz).
fn voxel_pixel(base_u: u32, lx: i32, ly: i32, lz: i32) -> vec2<i32> {
    let edge = i32(camera.grid_dims.z);  // samples per brick edge (8)
    let cx = clamp(lx, 0, edge - 1);
    let cy = clamp(ly, 0, edge - 1);
    let cz = clamp(lz, 0, edge - 1);
    return vec2<i32>(i32(base_u) + cy * edge + cx, cz);
}

// Fetch one distance sample. R8Snorm decodes to [-1,1] (the baked range).
fn load_voxel(base_u: u32, lx: i32, ly: i32, lz: i32) -> f32 {
    return textureLoad(atlas_tex, voxel_pixel(base_u, lx, ly, lz), 0).r;
}

// --- Dense per-material distance sampling ---
//
// Each voxel stores 8 signed distances (one per material), split across two
// Rgba16Snorm atlases (lo = ids 0..3, hi = 4..7). We trilinearly interpolate all
// 8 at the hit point; the material is the argmin (nearest material owns the
// surface), and the boundary between the two nearest materials is anti-aliased
// against the interpolated distance difference — a sub-voxel-sharp seam that does
// not depend on geometric smoothing.

// Fetch the 8 interpolated material distances at `world_pos`. Returns lo in
// `.x..w` of the first vec4 (ids 0..3) and hi in the second (ids 4..7).
struct MaterialDistances {
    lo: vec4<f32>,
    hi: vec4<f32>,
};

fn load_mat_texel(tex: texture_2d<f32>, base_u: u32, lx: i32, ly: i32, lz: i32) -> vec4<f32> {
    return textureLoad(tex, voxel_pixel(base_u, lx, ly, lz), 0);
}

// Trilinearly interpolate one material atlas (4 channels) at the brick-local
// fractional position.
fn sample_mat_tex(tex: texture_2d<f32>, base_u: u32, i0: vec3<i32>, f: vec3<f32>) -> vec4<f32> {
    let c000 = load_mat_texel(tex, base_u, i0.x,     i0.y,     i0.z);
    let c100 = load_mat_texel(tex, base_u, i0.x + 1, i0.y,     i0.z);
    let c010 = load_mat_texel(tex, base_u, i0.x,     i0.y + 1, i0.z);
    let c110 = load_mat_texel(tex, base_u, i0.x + 1, i0.y + 1, i0.z);
    let c001 = load_mat_texel(tex, base_u, i0.x,     i0.y,     i0.z + 1);
    let c101 = load_mat_texel(tex, base_u, i0.x + 1, i0.y,     i0.z + 1);
    let c011 = load_mat_texel(tex, base_u, i0.x,     i0.y + 1, i0.z + 1);
    let c111 = load_mat_texel(tex, base_u, i0.x + 1, i0.y + 1, i0.z + 1);

    let x00 = mix(c000, c100, f.x);
    let x10 = mix(c010, c110, f.x);
    let x01 = mix(c001, c101, f.x);
    let x11 = mix(c011, c111, f.x);
    let y0 = mix(x00, x10, f.y);
    let y1 = mix(x01, x11, f.y);
    return mix(y0, y1, f.z);
}

fn load_material_distances(base_u: u32, world_pos: vec3<f32>) -> MaterialDistances {
    let voxel_size = camera.grid_origin.w;
    let grid_orig = camera.grid_origin.xyz;
    let stride_f = f32(brick_stride());
    let voxel_f = (world_pos - grid_orig) / voxel_size;
    let brick_origin_voxel = floor(voxel_f / stride_f) * stride_f;
    let local_f = voxel_f - brick_origin_voxel;
    let i0 = vec3<i32>(floor(local_f));
    let f = local_f - floor(local_f);

    return MaterialDistances(
        sample_mat_tex(mat_lo_tex, base_u, i0, f),
        sample_mat_tex(mat_hi_tex, base_u, i0, f),
    );
}

// Read material distance slot `m` (0..7) from a sampled set.
fn mat_slot(md: MaterialDistances, m: u32) -> f32 {
    switch (m) {
        case 0u: { return md.lo.x; }
        case 1u: { return md.lo.y; }
        case 2u: { return md.lo.z; }
        case 3u: { return md.lo.w; }
        case 4u: { return md.hi.x; }
        case 5u: { return md.hi.y; }
        case 6u: { return md.hi.z; }
        default: { return md.hi.w; }
    }
}

// Resolved material at a point: the nearest (argmin) material id, the runner-up,
// and the signed distance gap between them (used for boundary anti-aliasing).
struct MaterialPick {
    id: u32,
    id2: u32,
    gap: f32,   // second_nearest - nearest, >= 0
};

fn pick_material(md: MaterialDistances) -> MaterialPick {
    var best = 0u;
    var best_d = mat_slot(md, 0u);
    var second = 0u;
    var second_d = 1e10;
    for (var m = 1u; m < 8u; m = m + 1u) {
        let d = mat_slot(md, m);
        if (d < best_d) {
            second = best;
            second_d = best_d;
            best = m;
            best_d = d;
        } else if (d < second_d) {
            second = m;
            second_d = d;
        }
    }
    return MaterialPick(best, second, second_d - best_d);
}

fn sample_brick_sdf(base_u: u32, world_pos: vec3<f32>) -> f32 {
    let voxel_size = camera.grid_origin.w;
    let grid_orig = camera.grid_origin.xyz;

    // Continuous voxel-space position, then split into the brick-local integer
    // corner and the sub-voxel fraction used for trilinear interpolation.
    // Brick stride is `stride` cells; sample `stride` (the apron) is shared with
    // the neighbour, so local coords stay in [0, stride) and i0+1 never exceeds it.
    let stride_f = f32(brick_stride());
    let voxel_f = (world_pos - grid_orig) / voxel_size;
    let brick_origin_voxel = floor(voxel_f / stride_f) * stride_f;
    let local_f = voxel_f - brick_origin_voxel;      // [0, stride)

    let i0 = vec3<i32>(floor(local_f));
    let f = local_f - floor(local_f);                // [0,1)
    let fx = f.x;
    let fy = f.y;
    let fz = f.z;

    let c000 = load_voxel(base_u, i0.x,     i0.y,     i0.z);
    let c100 = load_voxel(base_u, i0.x + 1, i0.y,     i0.z);
    let c010 = load_voxel(base_u, i0.x,     i0.y + 1, i0.z);
    let c110 = load_voxel(base_u, i0.x + 1, i0.y + 1, i0.z);
    let c001 = load_voxel(base_u, i0.x,     i0.y,     i0.z + 1);
    let c101 = load_voxel(base_u, i0.x + 1, i0.y,     i0.z + 1);
    let c011 = load_voxel(base_u, i0.x,     i0.y + 1, i0.z + 1);
    let c111 = load_voxel(base_u, i0.x + 1, i0.y + 1, i0.z + 1);

    let x00 = mix(c000, c100, fx);
    let x10 = mix(c010, c110, fx);
    let x01 = mix(c001, c101, fx);
    let x11 = mix(c011, c111, fx);
    let y0 = mix(x00, x10, fy);
    let y1 = mix(x01, x11, fy);
    return mix(y0, y1, fz);
}

// Trilinear SDF at any world position, resolving the brick by lookup. Returns a
// large positive value in empty (unbaked) space. Unlike sampling a fixed brick
// tile, this re-derives the brick per call, so it reads correct values across
// brick seams — essential for computing gradients near brick boundaries.
fn sample_sdf_world(world_pos: vec3<f32>) -> f32 {
    let loc = find_brick_lookup(compute_brick_id(world_to_brick(world_pos)));
    if (!loc.found) {
        return 1e10;
    }
    return sample_brick_sdf(loc.atlas_u, world_pos);
}

// --- Scene SDF (atlas-based union) ---

struct SceneSdfResult {
    dist: f32,
    object_id: u32,   // nearest material (argmin of the dense field)
    object_id2: u32,  // runner-up material (for seam anti-aliasing)
    gap: f32,         // runner_up_dist - nearest_dist, >= 0 (0 exactly on the seam)
    in_brick: bool,   // false => p lies in empty (unbaked) space
};

// NOTE: callable inside the raymarch loop. Must NOT use derivative ops (fwidth):
// control flow there is non-uniform. The seam anti-aliasing is done once, at the
// fragment level, from `object_id`/`object_id2`/`gap` (see `main`).
fn scene_sdf(p: vec3<f32>) -> SceneSdfResult {
    let brick_coord = world_to_brick(p);
    let brick_id = compute_brick_id(brick_coord);
    let loc = find_brick_lookup(brick_id);

    if (!loc.found) {
        return SceneSdfResult(1e10, 0u, 0u, 1e10, false);
    }

    let d = sample_brick_sdf(loc.atlas_u, p);

    // Material from the dense per-material distance field: the nearest material
    // owns the surface; the boundary against the runner-up is the bisector where
    // their interpolated distances are equal (`gap == 0`). Both distances are
    // continuous, so that bisector is sub-voxel sharp and independent of the
    // geometric smoothing — clean even at smoothing = 0.
    let md = load_material_distances(loc.atlas_u, p);
    let pick = pick_material(md);

    return SceneSdfResult(d, pick.id, pick.id2, pick.gap, true);
}

// Resolve the final surface colour with anti-aliased material seams. Safe to call
// fwidth here: `main` runs in uniform control flow. The seam (gap == 0) is widened
// to ~1 screen pixel via fwidth, so the boundary tracks projected size.
fn shade_material(res: SceneSdfResult) -> vec3<f32> {
    let col_a = camera.object_colors[i32(res.object_id)].rgb;
    let col_b = camera.object_colors[i32(res.object_id2)].rgb;
    let band = max(fwidth(res.gap), 1e-5);
    let w = clamp(0.5 + 0.5 * res.gap / band, 0.5, 1.0);  // 0.5 at seam → 1 away
    return mix(col_b, col_a, w);
}

// Distance along the ray to the far side of the brick containing `p`.
// Used for empty-space skipping (DDA-style): when `p` is in an unbaked brick,
// advance the ray to the next brick boundary instead of taking an infinite step.
fn dist_to_brick_exit(p: vec3<f32>, dir: vec3<f32>) -> f32 {
    let voxel_size = camera.grid_origin.w;
    let grid_orig = camera.grid_origin.xyz;
    let brick_world = voxel_size * f32(brick_stride());

    // Position within the current brick, in world units.
    let rel = p - grid_orig;
    let brick_min = floor(rel / brick_world) * brick_world + grid_orig;
    let brick_max = brick_min + vec3<f32>(brick_world);

    // Per-axis distance to the slab boundary in the ray's direction.
    var t = 1e10;
    for (var a = 0u; a < 3u; a = a + 1u) {
        let d = dir[a];
        if (abs(d) > 1e-6) {
            let bound = select(brick_min[a], brick_max[a], d > 0.0);
            let ta = (bound - p[a]) / d;
            if (ta > 0.0) {
                t = min(t, ta);
            }
        }
    }
    return t;
}

// --- Surface normal from the trilinear gradient ---

// Surface normal via the tetrahedron finite-difference technique, sampling the
// continuous cross-brick trilinear field. Probing the real interpolated field
// (rather than the piecewise-constant analytic gradient) gives normals that are
// continuous across cell *and* brick boundaries — no facets, no seam banding.
// The offset spans roughly one voxel so quantization in the snorm field doesn't
// dominate the difference.
fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let h = camera.grid_origin.w; // one voxel
    let k = vec2<f32>(1.0, -1.0);
    let n = k.xyy * sample_sdf_world(p + k.xyy * h)
          + k.yyx * sample_sdf_world(p + k.yyx * h)
          + k.yxy * sample_sdf_world(p + k.yxy * h)
          + k.xxx * sample_sdf_world(p + k.xxx * h);
    if (dot(n, n) > 1e-12) {
        return normalize(n);
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

// --- Analytic voxel intersection (paper §2) ---
//
// Inside one voxel the trilinearly-interpolated SDF is a cubic polynomial along
// the ray: f(t) = c3 t^3 + c2 t^2 + c1 t + c0. We solve it exactly per cell
// instead of sphere-tracing, which removes the stepping staircase from the
// surface and gives a crisp, correct silhouette of the trilinear isosurface.

struct CellCubic {
    c0: f32,
    c1: f32,
    c2: f32,
    c3: f32,
};

fn cubic_eval(c: CellCubic, t: f32) -> f32 {
    return c.c0 + t * (c.c1 + t * (c.c2 + t * c.c3));
}

// Build the cubic for the cell whose lower corner is `cell` (brick-local voxel
// index). `o` is the ray's entry point in this cell's canonical [0,1]^3 space
// and `d` is the ray direction in voxels-per-world-unit, so the cubic parameter
// is the world distance measured *from the cell entry* — keeping o and t small
// and the coefficients well-conditioned. Grouped per Eqs. (3),(6),(7).
fn build_cell_cubic(
    base_u: u32,
    cell: vec3<i32>,
    o: vec3<f32>,
    d: vec3<f32>,
) -> CellCubic {
    let s000 = load_voxel(base_u, cell.x,     cell.y,     cell.z);
    let s100 = load_voxel(base_u, cell.x + 1, cell.y,     cell.z);
    let s010 = load_voxel(base_u, cell.x,     cell.y + 1, cell.z);
    let s110 = load_voxel(base_u, cell.x + 1, cell.y + 1, cell.z);
    let s001 = load_voxel(base_u, cell.x,     cell.y,     cell.z + 1);
    let s101 = load_voxel(base_u, cell.x + 1, cell.y,     cell.z + 1);
    let s011 = load_voxel(base_u, cell.x,     cell.y + 1, cell.z + 1);
    let s111 = load_voxel(base_u, cell.x + 1, cell.y + 1, cell.z + 1);

    let k0 = s000;
    let k1 = s100 - s000;
    let k2 = s010 - s000;
    let k3 = s110 - s010 - k1;
    let a  = s101 - s001;
    let k4 = k0 - s001;
    let k5 = k1 - a;
    let k6 = k2 - (s011 - s001);
    let k7 = k3 - (s111 - s011 - a);

    let m0 = o.x * o.y;
    let m1 = d.x * d.y;
    let m2 = o.x * d.y + o.y * d.x;
    let m3 = k5 * o.z - k1;
    let m4 = k6 * o.z - k2;
    let m5 = k7 * o.z - k3;

    // Paper Eq (2) defines f_paper = z(...) - (...), which expands to the
    // NEGATED trilinear SDF. Negate so cubic_eval returns the true SDF and the
    // solver's "eval <= 0 means inside the surface" convention holds — otherwise
    // every ray false-hits at the first cell boundary (the shape renders boxy).
    let c0 = -((k4 * o.z - k0) + o.x * m3 + o.y * m4 + m0 * m5);
    let c1 = -(d.x * m3 + d.y * m4 + m2 * m5 + d.z * (k4 + k5 * o.x + k6 * o.y + k7 * m0));
    let c2 = -(m1 * m5 + d.z * (k5 * d.x + k6 * d.y + k7 * m2));
    let c3 = -(k7 * m1 * d.z);

    return CellCubic(c0, c1, c2, c3);
}

struct CellHit {
    hit: bool,
    t: f32,
};

// Refine a root known to lie in [a,b] (f(a),f(b) opposite signs) via regula
// falsi. Each subinterval is monotone so this converges reliably.
fn refine_root(c: CellCubic, a: f32, b: f32, fa: f32) -> f32 {
    var lo = a;
    var hi = b;
    var flo = fa;
    var tr = a;
    for (var k = 0u; k < 16u; k = k + 1u) {
        let fhi = cubic_eval(c, hi);
        let denom = fhi - flo;
        if (abs(denom) < 1e-20) {
            tr = 0.5 * (lo + hi);
        } else {
            tr = clamp(lo + (hi - lo) * (-flo) / denom, lo, hi);
        }
        let fr = cubic_eval(c, tr);
        if (fr * flo <= 0.0) {
            hi = tr;
        } else {
            lo = tr;
            flo = fr;
        }
    }
    return tr;
}

// First surface crossing of the cubic on a monotone subinterval [a,b].
fn test_subinterval(c: CellCubic, a: f32, b: f32) -> CellHit {
    if (b <= a) {
        return CellHit(false, 0.0);
    }
    let fa = cubic_eval(c, a);
    // Already inside the solid at the segment start.
    if (fa <= 0.0) {
        return CellHit(true, a);
    }
    let fb = cubic_eval(c, b);
    if (fa * fb <= 0.0) {
        return CellHit(true, refine_root(c, a, b, fa));
    }
    return CellHit(false, 0.0);
}

// Solve the cubic for the first root in [t0,t1]. The derivative is a quadratic
// whose (≤2) roots split [t0,t1] into ≤3 monotone segments; test them in order.
fn solve_cell_cubic(c: CellCubic, t0: f32, t1: f32) -> CellHit {
    let A = 3.0 * c.c3;
    let B = 2.0 * c.c2;
    let C = c.c1;

    var c_lo = t1;
    var c_hi = t1;
    if (abs(A) > 1e-10) {
        let disc = B * B - 4.0 * A * C;
        if (disc > 0.0) {
            let sq = sqrt(disc);
            let ra = (-B - sq) / (2.0 * A);
            let rb = (-B + sq) / (2.0 * A);
            c_lo = clamp(min(ra, rb), t0, t1);
            c_hi = clamp(max(ra, rb), t0, t1);
        }
    } else if (abs(B) > 1e-10) {
        c_lo = clamp(-C / B, t0, t1);
        c_hi = c_lo;
    }

    var r = test_subinterval(c, t0, c_lo);
    if (r.hit) { return r; }
    r = test_subinterval(c, c_lo, c_hi);
    if (r.hit) { return r; }
    return test_subinterval(c, c_hi, t1);
}

// Distance along the ray to the far face of the voxel cell containing `p`.
fn dist_to_cell_exit(p: vec3<f32>, dir: vec3<f32>) -> f32 {
    let vs = camera.grid_origin.w;
    let go = camera.grid_origin.xyz;
    let rel = p - go;
    let cell_min = floor(rel / vs) * vs + go;
    let cell_max = cell_min + vec3<f32>(vs);

    var t = 1e10;
    for (var a = 0u; a < 3u; a = a + 1u) {
        let dd = dir[a];
        if (abs(dd) > 1e-6) {
            let bound = select(cell_min[a], cell_max[a], dd > 0.0);
            let ta = (bound - p[a]) / dd;
            if (ta > 1e-6) {
                t = min(t, ta);
            }
        }
    }
    return t;
}

// --- BVH empty-space skipping ---

// Slab test: returns the entry distance t (>= t_min) if the ray hits the box
// within (t_min, t_max), else a negative sentinel.
fn ray_box_entry(lo: vec3<f32>, hi: vec3<f32>, o: vec3<f32>, inv_d: vec3<f32>, t_min: f32, t_max: f32) -> f32 {
    let t0 = (lo - o) * inv_d;
    let t1 = (hi - o) * inv_d;
    let tsmall = min(t0, t1);
    let tbig = max(t0, t1);
    let tn = max(max(tsmall.x, tsmall.y), max(tsmall.z, t_min));
    let tf = min(min(tbig.x, tbig.y), min(tbig.z, t_max));
    if (tf < tn) {
        return -1.0;
    }
    return tn;
}

// Distance to advance the ray from `p` so it reaches the next occupied region.
// Walks the BVH (bounded explicit stack, no recursion) for the nearest leaf-AABB
// the ray enters beyond a tiny epsilon. If none, returns a large skip so the march
// terminates quickly. Falls back to the brick DDA when the BVH is empty/degenerate.
fn bvh_ray_advance(p: vec3<f32>, dir: vec3<f32>) -> f32 {
    let count = num_bvh_nodes();
    if (count == 0u) {
        return dist_to_brick_exit(p, dir);
    }

    let inv_d = vec3<f32>(
        1.0 / select(dir.x, 1e-8, abs(dir.x) < 1e-8),
        1.0 / select(dir.y, 1e-8, abs(dir.y) < 1e-8),
        1.0 / select(dir.z, 1e-8, abs(dir.z) < 1e-8),
    );

    let MAXT = max_dist();
    var nearest = MAXT;
    var found = false;

    var stack: array<u32, 32>;
    var sp = 0u;
    stack[sp] = 0u; sp = sp + 1u;

    // The current point may already sit inside a leaf box; we want the distance to
    // the *entry* of the nearest box ahead. A small epsilon avoids re-detecting the
    // box we are leaving.
    let eps = camera.grid_origin.w * 0.5;

    while (sp > 0u) {
        sp = sp - 1u;
        let ni = stack[sp];
        if (ni >= count) { continue; }
        let node = bvh_buf[ni];

        let entry = ray_box_entry(node.aabb_min, node.aabb_max, p, inv_d, 0.0, nearest);
        if (entry < 0.0) {
            continue;  // ray misses this subtree within the current best bound
        }

        let is_internal = (node.count_or_right & BVH_INTERNAL_FLAG) != 0u;
        if (is_internal) {
            if (sp < 31u) { stack[sp] = node.left_or_first; sp = sp + 1u; }
            if (sp < 31u) { stack[sp] = node.count_or_right & ~BVH_INTERNAL_FLAG; sp = sp + 1u; }
        } else {
            // Leaf box: record its entry distance if it lies ahead of us.
            if (entry > eps && entry < nearest) {
                nearest = entry;
                found = true;
            }
        }
    }

    if (found) {
        return nearest - eps + camera.grid_origin.w * 0.01;
    }
    // Nothing ahead — skip far so the march ends.
    return MAXT;
}

// --- Raymarching ---

struct RaymarchResult {
    hit: bool,
    dist: f32,
    object_id: u32,
    steps: u32,
    hit_pos: vec3<f32>,
};

fn raymarch(origin: vec3<f32>, dir: vec3<f32>) -> RaymarchResult {
    var t = 0.0;
    var steps = 0u;
    var result = RaymarchResult(false, 0.0, 0u, 0u, vec3<f32>(0.0));

    let MAX_STEPS = max_steps();
    let MAX_DIST = max_dist();
    let SDF_EPS = sdf_eps();

    let voxel_size = camera.grid_origin.w;
    let grid_orig = camera.grid_origin.xyz;
    let edge = i32(camera.grid_dims.z);

    // Ray direction in voxels-per-world-unit. The per-cell cubic uses a local
    // entry point for its origin (computed inside the loop) so coefficients stay
    // well-conditioned; only the direction is precomputed here.
    let ray_d_voxel = dir / voxel_size;

    for (var i = 0u; i < MAX_STEPS; i = i + 1u) {
        steps = i + 1u;
        let p = origin + dir * t;

        if (t > MAX_DIST) {
            result.steps = steps;
            return result;
        }

        let scene = scene_sdf(p);

        if (scene.in_brick) {
            // Inside a baked brick: solve the cubic for the single voxel cell
            // containing `p`. This yields the exact ray/trilinear-surface
            // intersection rather than a sphere-traced approximation.
            let loc = find_brick_lookup(compute_brick_id(world_to_brick(p)));

            // Global voxel-space position and the integer cell (lower corner)
            // containing it. The cell's local frame is [0,1]^3 over that voxel.
            let gv = (p - grid_orig) / voxel_size;
            let cell_g = floor(gv);
            let o_local = gv - cell_g;   // entry point in [0,1]^3 (small, stable)

            // Brick-local cell index; clamp so cell+1 stays within stored samples
            // (0..edge-1, the last being the shared apron plane).
            let brick_origin_v = vec3<f32>(world_to_brick(p));
            let cell_local = clamp(
                vec3<i32>(cell_g - brick_origin_v),
                vec3<i32>(0),
                vec3<i32>(edge - 2),
            );

            let cubic = build_cell_cubic(loc.atlas_u, cell_local, o_local, ray_d_voxel);

            let advance = dist_to_cell_exit(p, dir);

            // Solve in the cell-local parameter [0, advance] (distance from `p`),
            // then offset by the global `t` to recover the true ray distance.
            let cell_hit = solve_cell_cubic(cubic, 0.0, advance);
            if (cell_hit.hit) {
                let t_hit = t + cell_hit.t;
                let hit_p = origin + dir * t_hit;
                result.hit = true;
                result.dist = t_hit;
                result.object_id = pick_material(load_material_distances(loc.atlas_u, hit_p)).id;
                result.steps = steps;
                result.hit_pos = hit_p;
                return result;
            }

            t += advance + voxel_size * 0.001;
        } else {
            // Empty space: jump straight to the next occupied edit-AABB using the
            // BVH (big skips across truly empty space), instead of stepping one
            // brick at a time. Falls back to the brick DDA when the BVH is empty.
            t += bvh_ray_advance(p, dir) + voxel_size * 0.01;
        }
    }

    result.steps = MAX_STEPS;
    return result;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

// --- Fragment shader ---

@fragment
fn main(in: FullscreenVertexOutput) -> FragmentOutput {
    let uv = in.uv;
    // Bevy/wgpu clip space is z in [0,1] with reverse-Z (near plane = 1.0).
    // Reconstruct the ray via the near-plane point — always finite, unlike the
    // far plane which sits at infinity for Bevy's infinite reverse-Z projection.
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    let world_near = camera.inv_view_proj * ndc;
    let world_pos = world_near.xyz / world_near.w;
    let ray_dir = normalize(world_pos - camera.camera_pos.xyz);
    let ray_origin = camera.camera_pos.xyz;

    // Background gradient
    let bg_color = mix(
        vec3<f32>(0.05, 0.05, 0.12),
        vec3<f32>(0.1, 0.1, 0.18),
        uv.y,
    );

    let rm = raymarch(ray_origin, ray_dir);

    if (!rm.hit) {
        return FragmentOutput(vec4<f32>(bg_color, 1.0), 0.0);
    }

    let hit_pos = rm.hit_pos;

    // True reverse-Z projection depth so the SDF surface shares the depth buffer
    // with normal geometry (wireframe, gizmos): project the world hit through the
    // forward view-proj and divide. Bevy clip space is z in [0,1], near = 1.
    let clip = camera.clip_from_world * vec4<f32>(hit_pos, 1.0);
    let ndc_depth = clip.z / clip.w;

    let obj_color = shade_material(scene_sdf(hit_pos));
    let normal = calc_normal(hit_pos);
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let diffuse = max(dot(normal, light_dir), 0.0);
    let ambient = 0.15;

    let shaded = obj_color * (ambient + diffuse * 0.85);

    // --- Debug output modes (toggled via shader_defs) ---

    #ifdef SDF_DEBUG_STEP_COUNT
    // Step count heatmap: blue (few) -> red (many)
    let t = f32(rm.steps) / f32(max_steps());
    let heatmap = vec3<f32>(t, 0.3 * (1.0 - t), 1.0 - t);
    if (rm.hit) {
        return FragmentOutput(vec4<f32>(heatmap, 1.0), ndc_depth);
    }
    return FragmentOutput(vec4<f32>(bg_color * 0.3, 1.0), 1.0);
    #endif

    #ifdef SDF_DEBUG_BVH_STEPS
    // Like the step heatmap, but colours *every* pixel (hit and miss) by march
    // cost so the empty-space traversal — which the BVH accelerates — is visible.
    // Compare against SDF_DEBUG_STEP_COUNT: with the BVH, background rays should
    // resolve in far fewer steps (deep blue) than brick-by-brick DDA.
    let bt = f32(rm.steps) / f32(max_steps());
    let bvh_heat = vec3<f32>(bt, 0.3 * (1.0 - bt), 1.0 - bt);
    let depth_out = select(1.0, ndc_depth, rm.hit);
    return FragmentOutput(vec4<f32>(bvh_heat, 1.0), depth_out);
    #endif

    #ifdef SDF_DEBUG_NORMALS
    if (rm.hit) {
        let debug_normal = normal * 0.5 + 0.5;
        return FragmentOutput(vec4<f32>(debug_normal, 1.0), ndc_depth);
    }
    return FragmentOutput(vec4<f32>(bg_color * 0.3, 1.0), 1.0);
    #endif

    #ifdef SDF_DEBUG_OBJECT_ID
    if (rm.hit) {
        // Generate distinct colors from object ID
        let hue = f32(rm.object_id) * 0.618033988749895;
        let h = fract(hue) * 6.0;
        let x = 1.0 - abs(h - 2.0) + 1.0;
        let sector = vec3<f32>(
            1.0 - abs(h - 3.0),
            1.0 - abs(h - 2.0),
            1.0 - abs(h - 1.0),
        );
        return FragmentOutput(vec4<f32>(sector, 1.0), ndc_depth);
    }
    return FragmentOutput(vec4<f32>(bg_color * 0.3, 1.0), 1.0);
    #endif

    #ifdef SDF_DEBUG_BRICK_BOUNDS
    if (rm.hit) {
        // Color the surface by the brick that contains the hit, and draw grid
        // lines on brick-cell boundaries to expose the sparse brick layout.
        let brick = world_to_brick(hit_pos);
        let brick_id = compute_brick_id(brick);
        let hue = f32(brick_id) * 0.618033988749895;
        let h = fract(hue) * 6.0;
        let tint = vec3<f32>(
            clamp(1.0 - abs(h - 3.0), 0.0, 1.0),
            clamp(1.0 - abs(h - 2.0), 0.0, 1.0),
            clamp(1.0 - abs(h - 1.0), 0.0, 1.0),
        );

        // Distance (in voxels) from the nearest brick-cell boundary plane.
        let voxel_size = camera.grid_origin.w;
        let s = f32(brick_stride());
        let rel = (hit_pos - camera.grid_origin.xyz) / voxel_size;
        let cell = rel / s;
        let frac3 = abs(fract(cell) - 0.5);
        let edge_dist = (0.5 - max(max(frac3.x, frac3.y), frac3.z)) * s;
        let line = select(1.0, 0.0, edge_dist < 0.15);

        let col = mix(vec3<f32>(0.05), tint, line * 0.85 + 0.15);
        return FragmentOutput(vec4<f32>(col, 1.0), ndc_depth);
    }
    return FragmentOutput(vec4<f32>(bg_color * 0.3, 1.0), 1.0);
    #endif

    return FragmentOutput(vec4<f32>(shaded, 1.0), ndc_depth);
}
