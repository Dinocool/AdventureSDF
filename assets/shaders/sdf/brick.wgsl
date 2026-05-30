#define_import_path sdf::brick

// Brick-atlas sampling: grid/brick coordinate math, the sorted-lookup binary
// search + palette unpack, trilinear distance & per-palette-slot material sampling,
// the combined `scene_sdf`, the cross-brick gradient normal, and the brick-DDA
// empty-space fallback.

#import sdf::bindings::{
    camera,
    ChunkLookup,
    PALETTE_EMPTY,
    chunk_buf,
    chunk_tile_buf,
    atlas_tex,
    mat_tex,
    cell_stride,
    voxel_size_at,
    lod_count,
    brick_world_at,
    CHUNK_BRICKS,
    abs_chunk_key,
    local_brick_index,
}

// --- Brick coordinate helpers ---

// Brick spatial stride in voxels. Bricks hold `brick_size` samples (grid_dims.z,
// = 8) but span `brick_size - 1` cells (= 7) and duplicate the shared boundary
// plane (apron). Matches SdfGridConfig::cell_stride on the CPU side.
fn brick_stride() -> i32 {
    return i32(camera.grid_dims.z) - 1;
}

// Brick origin coord (stride-aligned voxel coords on LOD `lod`'s lattice, anchored at
// world 0) containing `world_pos`. `floor` + Euclidean snap so the lattice is
// continuous through the origin — mirrors SdfGridConfig::world_to_brick_lod.
fn world_to_brick_lod(world_pos: vec3<f32>, lod: u32) -> vec3<i32> {
    let s = cell_stride();
    let vs = voxel_size_at(lod);
    let vox = vec3<i32>(floor(world_pos / vs));
    // div_euclid for negative coords: floor(vox / s) * s.
    let snapped = vec3<i32>(
        i32(floor(f32(vox.x) / f32(s))) * s,
        i32(floor(f32(vox.y) / f32(s))) * s,
        i32(floor(f32(vox.z) / f32(s))) * s,
    );
    return snapped;
}

// --- Chunk lookup: binary search + occupancy resolve ---

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

// Lexicographic compare of two 64-bit keys (hi then lo): -1 a<b, 0 equal, 1 a>b.
fn key_cmp(a_hi: u32, a_lo: u32, b_hi: u32, b_lo: u32) -> i32 {
    if (a_hi < b_hi) { return -1; }
    if (a_hi > b_hi) { return 1; }
    if (a_lo < b_lo) { return -1; }
    if (a_lo > b_lo) { return 1; }
    return 0;
}

// Binary search the sorted chunk table for the chunk with key (key_hi,key_lo). Returns
// its index, or -1 if absent. `count` = camera.grid_dims.w (resident chunk count).
fn find_chunk(key_hi: u32, key_lo: u32) -> i32 {
    let count = i32(camera.grid_dims.w);
    var lo: i32 = 0;
    var hi: i32 = count - 1;
    while (lo <= hi) {
        let mid = (lo + hi) / 2;
        let e = chunk_buf[u32(mid)];
        let c = key_cmp(e.key_hi, e.key_lo, key_hi, key_lo);
        if (c == 0) { return mid; }
        else if (c < 0) { lo = mid + 1; }
        else { hi = mid - 1; }
    }
    return -1;
}

// Resolve the baked brick at `coord` on LOD `lod` via its chunk: find the chunk, test
// the occupancy bit for the brick's local slot, and if present index into the chunk's
// packed tile run (popcount of mask bits below the slot gives the offset).
fn find_brick_lookup(coord: vec3<i32>, lod: u32) -> BrickLocation {
    let count = u32(camera.grid_dims.w);
    if (count == 0u) {
        return BrickLocation(0u, vec4<u32>(PALETTE_EMPTY), false);
    }
    let key = abs_chunk_key(coord, lod);
    let ci = find_chunk(key.x, key.y);
    if (ci < 0) {
        return BrickLocation(0u, vec4<u32>(PALETTE_EMPTY), false);
    }
    let chunk = chunk_buf[u32(ci)];
    let li = local_brick_index(coord);   // 0..63

    // Occupancy bit test (mask split across two u32). Bit li set ⇒ brick resident.
    var bit: u32;
    if (li < 32u) { bit = (chunk.occ_lo >> li) & 1u; }
    else { bit = (chunk.occ_hi >> (li - 32u)) & 1u; }
    if (bit == 0u) {
        return BrickLocation(0u, vec4<u32>(PALETTE_EMPTY), false);
    }

    // Offset within the chunk's tile run = popcount of mask bits below li.
    var below_lo: u32;
    var below_hi: u32;
    if (li < 32u) {
        below_lo = chunk.occ_lo & ((1u << li) - 1u);
        below_hi = 0u;
    } else {
        below_lo = chunk.occ_lo;
        below_hi = chunk.occ_hi & ((1u << (li - 32u)) - 1u);
    }
    let off = countOneBits(below_lo) + countOneBits(below_hi);
    let tile = chunk_tile_buf[chunk.tile_run_base + off];
    return BrickLocation(tile.atlas_base, unpack_palette(tile.pal_lo, tile.pal_hi), true);
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
// set), sampling the brick at LOD `lod`. `.x..w` correspond to palette slots 0..3.
fn load_material_distances(base_u: u32, world_pos: vec3<f32>, lod: u32) -> vec4<f32> {
    let voxel_size = voxel_size_at(lod);
    let stride_f = f32(cell_stride());
    let voxel_f = world_pos / voxel_size;
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

fn sample_brick_sdf(base_u: u32, world_pos: vec3<f32>, lod: u32) -> f32 {
    let voxel_size = voxel_size_at(lod);

    // Continuous voxel-space position, then split into the brick-local integer
    // corner and the sub-voxel fraction used for trilinear interpolation.
    // Brick stride is `stride` cells; sample `stride` (the apron) is shared with
    // the neighbour, so local coords stay in [0, stride) and i0+1 never exceeds it.
    let stride_f = f32(cell_stride());
    let voxel_f = world_pos / voxel_size;
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

// Find the finest LOD with a baked brick at `world_pos`. Returns the brick location plus
// the LOD it was found at (via `out_lod`). Misses (`found == false`) when no LOD has a
// brick here (empty space).
fn find_brick_at(world_pos: vec3<f32>, out_lod: ptr<function, u32>) -> BrickLocation {
    let levels = lod_count();
    for (var lod = 0u; lod < levels; lod = lod + 1u) {
        let coord = world_to_brick_lod(world_pos, lod);
        let loc = find_brick_lookup(coord, lod);
        if (loc.found) {
            *out_lod = lod;
            return loc;
        }
    }
    *out_lod = 0u;
    return BrickLocation(0u, vec4<u32>(PALETTE_EMPTY), false);
}

// The finest LOD with a resident CHUNK at `world_pos` (whether or not the specific brick
// is occupied). This is the resolution at which a brick *could* exist at `p`, so the
// empty-space DDA must step by THIS LOD's brick size — never coarser, or it would jump
// over a thin baked surface near the camera (the "gaps in objects" bug). Near the camera
// LOD 0's chunk is resident → small steps; far out only coarse chunks reach `p` → big
// steps. Returns the coarsest LOD if no chunk covers `p`.
fn finest_lod_window_at(world_pos: vec3<f32>) -> u32 {
    let levels = lod_count();
    for (var lod = 0u; lod < levels; lod = lod + 1u) {
        let coord = world_to_brick_lod(world_pos, lod);
        let key = abs_chunk_key(coord, lod);
        if (find_chunk(key.x, key.y) >= 0) {
            return lod;
        }
    }
    return levels - 1u;
}

// Trilinear SDF at any world position, resolving the brick by finest-LOD lookup.
// Returns a large positive value in empty (unbaked) space. Re-derives the brick per
// call, so it reads correct values across brick seams — essential for gradients near
// brick boundaries.
fn sample_sdf_world(world_pos: vec3<f32>) -> f32 {
    var lod = 0u;
    let loc = find_brick_at(world_pos, &lod);
    if (!loc.found) {
        return 1e10;
    }
    return sample_brick_sdf(loc.atlas_base, world_pos, lod);
}

// --- Scene SDF (atlas-based union) ---

struct SceneSdfResult {
    dist: f32,
    object_id: u32,    // nearest material (argmin of the dense field)
    object_id_b: u32,  // runner-up material (for seam anti-aliasing)
    gap: f32,          // runner_up_dist - nearest_dist, >= 0 (0 exactly on the seam)
    in_brick: bool,    // false => p lies in empty (unbaked) space
    lod: u32,          // LOD level that served this sample (for cubic solve + debug)
    atlas_base: u32,   // tile origin of the serving brick (so the loop skips re-search)
    palette: vec4<u32>,// the serving brick's palette (for material pick at the hit)
};

// NOTE: callable inside the raymarch loop. Must NOT use derivative ops (fwidth):
// control flow there is non-uniform. The seam anti-aliasing is done once, at the
// fragment level, from `object_id`/`object_id2`/`gap` (see `main`).
fn scene_sdf(p: vec3<f32>) -> SceneSdfResult {
    var lod = 0u;
    let loc = find_brick_at(p, &lod);

    if (!loc.found) {
        return SceneSdfResult(1e10, 0u, 0u, 1e10, false, 0u, 0u, vec4<u32>(PALETTE_EMPTY));
    }

    let d = sample_brick_sdf(loc.atlas_base, p, lod);

    // Material from the per-palette-slot distance field: the nearest palette slot
    // owns the surface (mapped to a global id); the boundary against the runner-up
    // is the bisector where their interpolated distances are equal (`gap == 0`).
    // Both distances are continuous, so that bisector is sub-voxel sharp and
    // independent of geometric smoothing — clean even at smoothing = 0.
    let md = load_material_distances(loc.atlas_base, p, lod);
    let pick = pick_material(md, loc.palette);

    return SceneSdfResult(d, pick.id, pick.id_b, pick.gap, true, lod, loc.atlas_base, loc.palette);
}

// Distance along the ray to the far side of the brick containing `p`, at LOD `lod`.
// Used for empty-space skipping (DDA-style): when `p` is in an unbaked brick, advance
// the ray to the next brick boundary instead of taking an infinite step. The lattice
// is anchored at world 0 with the LOD's voxel size.
fn dist_to_brick_exit_lod(p: vec3<f32>, dir: vec3<f32>, lod: u32) -> f32 {
    let brick_world = brick_world_at(lod);

    let brick_min = floor(p / brick_world) * brick_world;
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

// Empty-space advance for a point in no baked brick: step to the far face of the brick
// at the FINEST LOD window covering `p`. Using the finest covering resolution (not the
// coarsest) guarantees the step never overshoots a thin baked surface near the camera,
// while still taking large steps far out where only coarse rings reach. This is the
// safe DDA floor the BVH skip builds on.
fn dist_to_brick_exit(p: vec3<f32>, dir: vec3<f32>) -> f32 {
    return dist_to_brick_exit_lod(p, dir, finest_lod_window_at(p));
}

// Voxel size at the finest LOD window covering `p` — the right scale for the
// post-advance epsilon nudge so it clears the brick boundary without overshooting.
fn step_voxel_at(p: vec3<f32>) -> f32 {
    return voxel_size_at(finest_lod_window_at(p));
}

// --- Surface normal from the trilinear gradient ---

// Surface normal via the tetrahedron finite-difference technique, sampling the
// continuous cross-brick trilinear field. Probing the real interpolated field
// (rather than the piecewise-constant analytic gradient) gives normals that are
// continuous across cell *and* brick boundaries — no facets, no seam banding.
// The offset spans roughly one voxel so quantization in the snorm field doesn't
// dominate the difference.
fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    // Probe offset ≈ one voxel at the LOD serving this point (coarser LODs need a
    // proportionally larger offset so the finite difference spans a real sample gap).
    var hit_lod = 0u;
    let loc = find_brick_at(p, &hit_lod);
    let h = voxel_size_at(hit_lod);
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
