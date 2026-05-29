#define_import_path sdf::brick

// Brick-atlas sampling: grid/brick coordinate math, the sorted-lookup binary
// search + palette unpack, trilinear distance & per-palette-slot material sampling,
// the combined `scene_sdf`, the cross-brick gradient normal, and the brick-DDA
// empty-space fallback.

#import sdf::bindings::{
    camera,
    BrickLookup,
    PALETTE_EMPTY,
    lookup_buf,
    atlas_tex,
    mat_tex,
}

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
    atlas_base: u32,     // packed tile origin (col_px | row_px<<16); see voxel_pixel
    palette: vec4<u32>,  // 4 global material ids (PALETTE_EMPTY = unused)
    found: bool,
};

fn unpack_palette(lo: u32, hi: u32) -> vec4<u32> {
    return vec4<u32>(
        lo & 0xffffu,
        (lo >> 16u) & 0xffffu,
        hi & 0xffffu,
        (hi >> 16u) & 0xffffu,
    );
}

// Binary search the sorted lookup buffer for a brick id.
fn find_brick_lookup(brick_id: u32) -> BrickLocation {
    let count = u32(camera.grid_dims.w);
    if (count == 0u) {
        return BrickLocation(0u, vec4<u32>(PALETTE_EMPTY), false);
    }

    var lo: i32 = 0;
    var hi: i32 = i32(count) - 1;

    while (lo <= hi) {
        let mid = (lo + hi) / 2;
        let entry = lookup_buf[u32(mid)];
        let mid_id = entry.brick_id;

        if (mid_id == brick_id) {
            return BrickLocation(entry.atlas_base, unpack_palette(entry.pal_lo, entry.pal_hi), true);
        } else if (mid_id < brick_id) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    return BrickLocation(0u, vec4<u32>(PALETTE_EMPTY), false);
}

// --- Sample brick SDF via trilinear interpolation ---

// Pixel coordinate of brick-local voxel (lx,ly,lz) in the 2D-tiled atlas. `base`
// packs the tile origin as col_px | row_px<<16. Within a tile, pixel =
// (col_px + ly*EDGE + lx, row_px + lz). Tiling keeps the atlas width bounded.
fn voxel_pixel(base: u32, lx: i32, ly: i32, lz: i32) -> vec2<i32> {
    let edge = i32(camera.grid_dims.z);  // samples per brick edge (8)
    let cx = clamp(lx, 0, edge - 1);
    let cy = clamp(ly, 0, edge - 1);
    let cz = clamp(lz, 0, edge - 1);
    let col_px = i32(base & 0xffffu);
    let row_px = i32(base >> 16u);
    return vec2<i32>(col_px + cy * edge + cx, row_px + cz);
}

// Fetch one distance sample. R16Snorm decodes to [-1,1] (the baked range).
fn load_voxel(base_u: u32, lx: i32, ly: i32, lz: i32) -> f32 {
    return textureLoad(atlas_tex, voxel_pixel(base_u, lx, ly, lz), 0).r;
}

// --- Per-palette-slot material distance sampling ---
//
// Each voxel stores K=4 signed distances in one Rgba16Snorm atlas — one per entry
// of the brick's material palette (NOT per global material). We trilinearly
// interpolate the 4 at the hit point; the nearest (argmin) palette slot owns the
// surface, mapped to a global id via the brick palette. The boundary between the
// two nearest slots is anti-aliased against the interpolated distance difference —
// sub-voxel sharp, independent of geometric smoothing. Per-pixel cost is O(K),
// independent of how many materials the world contains.

fn sample_mat_tex(base_u: u32, i0: vec3<i32>, f: vec3<f32>) -> vec4<f32> {
    let c000 = textureLoad(mat_tex, voxel_pixel(base_u, i0.x,     i0.y,     i0.z),     0);
    let c100 = textureLoad(mat_tex, voxel_pixel(base_u, i0.x + 1, i0.y,     i0.z),     0);
    let c010 = textureLoad(mat_tex, voxel_pixel(base_u, i0.x,     i0.y + 1, i0.z),     0);
    let c110 = textureLoad(mat_tex, voxel_pixel(base_u, i0.x + 1, i0.y + 1, i0.z),     0);
    let c001 = textureLoad(mat_tex, voxel_pixel(base_u, i0.x,     i0.y,     i0.z + 1), 0);
    let c101 = textureLoad(mat_tex, voxel_pixel(base_u, i0.x + 1, i0.y,     i0.z + 1), 0);
    let c011 = textureLoad(mat_tex, voxel_pixel(base_u, i0.x,     i0.y + 1, i0.z + 1), 0);
    let c111 = textureLoad(mat_tex, voxel_pixel(base_u, i0.x + 1, i0.y + 1, i0.z + 1), 0);

    let x00 = mix(c000, c100, f.x);
    let x10 = mix(c010, c110, f.x);
    let x01 = mix(c001, c101, f.x);
    let x11 = mix(c011, c111, f.x);
    let y0 = mix(x00, x10, f.y);
    let y1 = mix(x01, x11, f.y);
    return mix(y0, y1, f.z);
}

// The 4 interpolated palette-slot distances at `world_pos` (one Rgba16Snorm fetch
// set). `.x..w` correspond to palette slots 0..3.
fn load_material_distances(base_u: u32, world_pos: vec3<f32>) -> vec4<f32> {
    let voxel_size = camera.grid_origin.w;
    let grid_orig = camera.grid_origin.xyz;
    let stride_f = f32(brick_stride());
    let voxel_f = (world_pos - grid_orig) / voxel_size;
    let brick_origin_voxel = floor(voxel_f / stride_f) * stride_f;
    let local_f = voxel_f - brick_origin_voxel;
    let i0 = vec3<i32>(floor(local_f));
    let f = local_f - floor(local_f);
    return sample_mat_tex(base_u, i0, f);
}

// Resolved material at a point: nearest + runner-up GLOBAL material ids and the
// signed distance gap between them (for boundary anti-aliasing). `palette` maps
// local slots to global ids; empty slots are skipped via their MATERIAL_FAR dist.
struct MaterialPick {
    id: u32,
    id_b: u32,  // runner-up global material id
    gap: f32,   // second_nearest - nearest, >= 0
};

fn pick_material(slots: vec4<f32>, palette: vec4<u32>) -> MaterialPick {
    var best = 0u;
    var best_d = slots.x;
    var second = 0u;
    var second_d = 1e10;
    for (var k = 1u; k < 4u; k = k + 1u) {
        let d = slots[k];
        if (d < best_d) {
            second = best;
            second_d = best_d;
            best = k;
            best_d = d;
        } else if (d < second_d) {
            second = k;
            second_d = d;
        }
    }
    return MaterialPick(palette[best], palette[second], second_d - best_d);
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
    return sample_brick_sdf(loc.atlas_base, world_pos);
}

// --- Scene SDF (atlas-based union) ---

struct SceneSdfResult {
    dist: f32,
    object_id: u32,    // nearest material (argmin of the dense field)
    object_id_b: u32,  // runner-up material (for seam anti-aliasing)
    gap: f32,          // runner_up_dist - nearest_dist, >= 0 (0 exactly on the seam)
    in_brick: bool,    // false => p lies in empty (unbaked) space
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

    let d = sample_brick_sdf(loc.atlas_base, p);

    // Material from the per-palette-slot distance field: the nearest palette slot
    // owns the surface (mapped to a global id); the boundary against the runner-up
    // is the bisector where their interpolated distances are equal (`gap == 0`).
    // Both distances are continuous, so that bisector is sub-voxel sharp and
    // independent of geometric smoothing — clean even at smoothing = 0.
    let md = load_material_distances(loc.atlas_base, p);
    let pick = pick_material(md, loc.palette);

    return SceneSdfResult(d, pick.id, pick.id_b, pick.gap, true);
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
