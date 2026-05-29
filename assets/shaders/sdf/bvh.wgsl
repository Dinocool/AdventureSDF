#define_import_path sdf::bvh

// GPU BVH traversal for empty-space skipping: a bounded-stack (no recursion) walk
// of the edit-AABB tree that jumps the ray to the next occupied region, instead of
// stepping brick-by-brick. Falls back to the brick DDA when the BVH is empty.

#import sdf::bindings::{camera, bvh_buf, BVH_INTERNAL_FLAG, num_bvh_nodes, max_dist}
#import sdf::brick::dist_to_brick_exit

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
