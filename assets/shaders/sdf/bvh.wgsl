#define_import_path sdf::bvh

// GPU BVH traversal for empty-space skipping: a bounded-stack (no recursion) walk
// of the edit-AABB tree that jumps the ray to the next occupied region, instead of
// stepping brick-by-brick. Falls back to the brick DDA when the BVH is empty.

#import sdf::bindings::{camera, bvh_buf, BVH_INTERNAL_FLAG, num_bvh_nodes, max_dist, bake_reach, brick_world_at}
#import sdf::brick::{dist_to_brick_exit, finest_lod_window_at}

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

// Distance to advance the ray from `p` toward the next occupied region.
//
// The safe floor is one brick step (`dist_to_brick_exit`): stepping brick-by-brick
// never skips a baked voxel, which is why forcing pure brick-DDA renders correctly.
// The BVH is used ONLY to prove that a LARGER span ahead is empty, letting us skip
// the whole gap in one advance. Every node box is inflated by `bake_reach()` — the
// exact distance baked bricks extend beyond a tight edit AABB (see `bricks_in_aabb`)
// — so a skip can never land past a baked shell brick. Over-inflation only ever makes
// us skip less (a brick early), never overshoot. Falls back to the brick DDA when the
// BVH is empty/degenerate.
fn bvh_ray_advance(p: vec3<f32>, dir: vec3<f32>) -> f32 {
    // Always-safe one-brick advance. We never return less than this, and if the BVH
    // offers nothing better we return exactly this (== proven-correct pure brick-DDA).
    let t_brick = dist_to_brick_exit(p, dir);

    let count = num_bvh_nodes();
    if (count == 0u) {
        return t_brick;
    }

    let inv_d = vec3<f32>(
        1.0 / select(dir.x, 1e-8, abs(dir.x) < 1e-8),
        1.0 / select(dir.y, 1e-8, abs(dir.y) < 1e-8),
        1.0 / select(dir.z, 1e-8, abs(dir.z) < 1e-8),
    );

    let MAXT = max_dist();
    var nearest = MAXT;
    var found = false;
    // True when `p` is currently INSIDE an inflated box: occupied territory under the
    // ray right now. We must step one brick through it, never skip its surface.
    var inside = false;

    var stack: array<u32, 32>;
    var sp = 0u;
    stack[sp] = 0u; sp = sp + 1u;

    while (sp > 0u) {
        sp = sp - 1u;
        let ni = stack[sp];
        if (ni >= count) { continue; }
        let node = bvh_buf[ni];

        // Inflate by the full bake footprint so the skip lands at or before any baked
        // brick — for both internal and leaf nodes. The reach is LOD-dependent: a baked
        // brick at LOD L extends `~brick_world_at(L)` beyond the tight edit AABB stored
        // here, and L grows 2^L per level. Size the pad to the LOD actually resident at
        // this box (its centre), plus a 2-brick margin so over-inflation (harmless: skips
        // a hair early) covers the lattice snap. A single LOD-0 pad under-inflates coarse
        // rings → the skip jumps over their shells → escaped rays / grain at LOD 2+.
        let box_center = (node.aabb_min + node.aabb_max) * 0.5;
        let box_lod = finest_lod_window_at(box_center);
        let pad = vec3<f32>(bake_reach() + 2.0 * brick_world_at(box_lod));
        let lo = node.aabb_min - pad;
        let hi = node.aabb_max + pad;

        // Inside this box right now? Then there is occupied space on the ray here;
        // don't skip past it. (Cheap point-in-AABB; applies to internal nodes too,
        // which conservatively bounds their children.)
        if (all(p >= lo) && all(p <= hi)) {
            inside = true;
        }

        let entry = ray_box_entry(lo, hi, p, inv_d, 0.0, nearest);
        if (entry < 0.0) {
            continue;  // ray misses this subtree within the current best bound
        }

        let is_internal = (node.count_or_right & BVH_INTERNAL_FLAG) != 0u;
        if (is_internal) {
            if (sp < 31u) { stack[sp] = node.left_or_first; sp = sp + 1u; }
            if (sp < 31u) { stack[sp] = node.count_or_right & ~BVH_INTERNAL_FLAG; sp = sp + 1u; }
        } else if (entry > 0.0 && entry < nearest) {
            nearest = entry;
            found = true;
        }
    }

    // Inside an occupied box → step exactly one brick; never skip a surface.
    if (inside) {
        return t_brick;
    }
    // A box lies ahead and the span [p, p+nearest] is proven empty (no inflated box
    // overlaps it) → skip the whole gap, but never less than one safe brick step.
    if (found) {
        return max(t_brick, nearest);
    }
    // Nothing ahead — skip far so the march ends.
    return MAXT;
}
