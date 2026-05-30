#define_import_path sdf::cubic

// Analytic per-voxel cubic intersection (JCGT 2022 §2). Inside one voxel the
// trilinearly-interpolated SDF is a cubic along the ray, f(t)=c3 t³+c2 t²+c1 t+c0;
// we solve it exactly per cell instead of sphere-tracing — crisp silhouettes, no
// stepping staircase. Also the per-cell exit-distance helper for the march.

#import sdf::bindings::{camera, voxel_size_at}
#import sdf::brick::load_voxel

// Cubic coefficients f(t) = const_term + t*(lin + t*(quad + t*cube)). Named (not
// c0..c3) because naga_oil rejects digit-suffixed struct field names in composable
// modules (its writeback-substitution check).
struct CellCubic {
    const_term: f32,
    lin: f32,
    quad: f32,
    cube: f32,
};

fn cubic_eval(c: CellCubic, t: f32) -> f32 {
    return c.const_term + t * (c.lin + t * (c.quad + t * c.cube));
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
    let A = 3.0 * c.cube;
    let B = 2.0 * c.quad;
    let C = c.lin;

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

// Distance along the ray to the far face of the voxel cell containing `p`, at LOD
// `lod`. The cell lattice is anchored at world 0 with the LOD's voxel size.
fn dist_to_cell_exit(p: vec3<f32>, dir: vec3<f32>, lod: u32) -> f32 {
    let vs = voxel_size_at(lod);
    let cell_min = floor(p / vs) * vs;
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
