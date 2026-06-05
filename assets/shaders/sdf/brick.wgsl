#define_import_path sdf::brick

// Brick-atlas sampling: grid/brick coordinate math, the direct toroidal-directory chunk
// lookup (`find_chunk` = `dir_index` + key-tag compare) + palette unpack, trilinear distance
// & per-palette-slot material sampling, the combined `scene_sdf`, the cross-brick gradient
// normal, and the brick-DDA empty-space fallback.

#import sdf::bindings::{
    camera,
    ChunkLookup,
    PALETTE_EMPTY,
    chunk_buf,
    chunk_tile_buf,
    atlas_pages,
    mat_pages,
    ATLAS_PAGE_HEIGHT_PX,
    cell_stride,
    voxel_size_at,
    lod_count,
    brick_world_at,
    CHUNK_BRICKS,
    abs_chunk_key,
    dir_index,
    local_brick_index,
    euclid_mod,
    floor_div,
    ring_bricks,
    recenter_snap,
    DIST_BAND_VOXELS,
}

// --- Brick coordinate helpers ---

// Brick spatial stride in voxels. Bricks hold `brick_size` samples (grid_dims.z,
// = 8) but span `brick_size - 1` cells (= 7) and duplicate the shared boundary
// plane (apron). Matches SdfGridConfig::cell_stride on the CPU side.
fn brick_stride() -> i32 {
    return i32(camera.grid_dims.z) - 1;
}

// Stride-align a voxel coord to its brick origin: subtract the euclidean remainder so the
// origin is the largest multiple of `s` that is <= coord. This is `floor(coord/s)*s` WITHOUT
// any division — `coord - euclid_mod(coord, s)`.
fn brick_snap(coord: i32, s: i32) -> i32 {
    return coord - euclid_mod(coord, s);
}

// Brick origin coord (stride-aligned voxel coords on LOD `lod`'s lattice, anchored at
// world 0) containing `world_pos`. `floor` + Euclidean snap so the lattice is
// continuous through the origin — mirrors SdfGridConfig::world_to_brick_lod.
fn world_to_brick_lod(world_pos: vec3<f32>, lod: u32) -> vec3<i32> {
    let s = cell_stride();
    let vs = voxel_size_at(lod);
    let vox = vec3<i32>(floor(world_pos / vs));
    let snapped = vec3<i32>(
        brick_snap(vox.x, s),
        brick_snap(vox.y, s),
        brick_snap(vox.z, s),
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

// Direct toroidal chunk lookup. The `chunk_buf` is the dense per-LOD DIRECTORY (chunk.rs
// LiveChunkTables): chunk `c` at `dir_index(c, lod)`. Index it directly and accept only if the
// stored key TAG equals `abs_chunk_key(coord, lod)` — empty/stale slots carry a sentinel key that
// never matches a real chunk, so a departed (cleared) or never-baked slot is a clean miss → the
// caller falls back to a coarser LOD. Returns the directory index (for `chunk_buf[ci]`), or -1.
// O(1) — no sort, no search. `arrayLength` is read directly so a stale-size read just misses.
fn find_chunk(coord: vec3<i32>, lod: u32) -> i32 {
    let idx = dir_index(coord, lod);
    if (idx >= arrayLength(&chunk_buf)) {
        return -1;
    }
    let e = chunk_buf[idx];
    let key = abs_chunk_key(coord, lod);
    if (e.key_hi == key.x && e.key_lo == key.y) {
        return i32(idx);
    }
    return -1;
}

// Resolve the brick at `coord` WITHIN an already-found chunk: test the occupancy bit for
// the brick's local slot, and if present index into the chunk's packed tile run (popcount
// of mask bits below the slot gives the offset). Split out of `find_brick_lookup` so the
// cached march resolve can reuse it after its own (cached) chunk search.
fn brick_in_chunk(chunk: ChunkLookup, coord: vec3<i32>) -> BrickLocation {
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

// Resolve the baked brick at `coord` on LOD `lod` via its chunk: find the chunk, then test
// the occupancy bit / index the tile run via `brick_in_chunk`.
fn find_brick_lookup(coord: vec3<i32>, lod: u32) -> BrickLocation {
    let ci = find_chunk(coord, lod);
    if (ci < 0) {
        return BrickLocation(0u, vec4<u32>(PALETTE_EMPTY), false);
    }
    return brick_in_chunk(chunk_buf[u32(ci)], coord);
}

// --- Sample brick SDF via trilinear interpolation ---

// Atlas location of a brick-local voxel: which PAGE texture, and the pixel WITHIN that page.
// `base` packs the tile origin as col_px | row_px<<16 (row_px is the GLOBAL tile row in pixels).
// Within a tile, the global pixel is (col_px + ly*EDGE + lx, row_px + lz); the paged pool splits
// the global row into page = global_y / ATLAS_PAGE_HEIGHT_PX and local_y = global_y % page_h. A
// tile is `edge`(8) px tall and the page height is a multiple of 8, so all 8 of a tile's rows live
// in the SAME page (the page index is constant across a tile's voxels).
struct AtlasLoc { page: u32, px: vec2<i32> };

fn voxel_loc(base: u32, lx: i32, ly: i32, lz: i32) -> AtlasLoc {
    let edge = i32(camera.grid_dims.z);  // samples per brick edge (8)
    let cx = clamp(lx, 0, edge - 1);
    let cy = clamp(ly, 0, edge - 1);
    let cz = clamp(lz, 0, edge - 1);
    let col_px = i32(base & 0xffffu);
    let row_px = i32(base >> 16u);
    let gy = row_px + cz;
    let page_h = ATLAS_PAGE_HEIGHT_PX;
    return AtlasLoc(u32(gy / page_h), vec2<i32>(col_px + cy * edge + cx, gy % page_h));
}

// Fetch one distance sample. R16Snorm decodes to [-1,1] (the baked range).
fn load_voxel(base_u: u32, lx: i32, ly: i32, lz: i32) -> f32 {
    let loc = voxel_loc(base_u, lx, ly, lz);
    return textureLoad(atlas_pages[loc.page], loc.px, 0).r;
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

fn load_mat(base_u: u32, lx: i32, ly: i32, lz: i32) -> vec4<f32> {
    let loc = voxel_loc(base_u, lx, ly, lz);
    return textureLoad(mat_pages[loc.page], loc.px, 0);
}

fn sample_mat_tex(base_u: u32, i0: vec3<i32>, f: vec3<f32>) -> vec4<f32> {
    let c000 = load_mat(base_u, i0.x,     i0.y,     i0.z);
    let c100 = load_mat(base_u, i0.x + 1, i0.y,     i0.z);
    let c010 = load_mat(base_u, i0.x,     i0.y + 1, i0.z);
    let c110 = load_mat(base_u, i0.x + 1, i0.y + 1, i0.z);
    let c001 = load_mat(base_u, i0.x,     i0.y,     i0.z + 1);
    let c101 = load_mat(base_u, i0.x + 1, i0.y,     i0.z + 1);
    let c011 = load_mat(base_u, i0.x,     i0.y + 1, i0.z + 1);
    let c111 = load_mat(base_u, i0.x + 1, i0.y + 1, i0.z + 1);

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
    let s = cell_stride();
    let voxel_f = world_pos / voxel_size;
    // Brick origin via the SAME exact integer floor-div as world_to_brick_lod — a float
    // `floor(voxel_f/stride)*stride` mis-snaps brick-boundary points (apron collapse).
    let vox = vec3<i32>(floor(voxel_f));
    let brick_origin_voxel = vec3<f32>(
        f32(brick_snap(vox.x, s)),
        f32(brick_snap(vox.y, s)),
        f32(brick_snap(vox.z, s)),
    );
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

// Resolve the material at `world_pos` given the serving brick's `palette`. A brick whose palette
// has only slot 0 filled (`palette[1] == PALETTE_EMPTY`) is SINGLE-MATERIAL — every voxel is
// `palette[0]` — so we skip the 8 material-distance texture loads + the argmin entirely and return
// it directly. Palettes are packed densely from slot 0 (edits::build_palette), so this test is
// exact. This is the common case (most bricks touch one material) and removes the dominant per-hit
// material cost there; multi-material bricks fall through to the per-palette-slot argmin. The
// returned `gap` (1.0 = the material "far" band) matches what the full per-slot path yields for a
// single-material brick (runner-up = MATERIAL_FAR), so `id_b == id` skips the blend and `fwidth(gap)`
// stays ~0 across a uniform↔multi material-brick boundary (a larger sentinel would blow up fwidth
// there and paint a 1px seam on the neighbouring multi-material pixel).
fn resolve_material(base_u: u32, world_pos: vec3<f32>, lod: u32, palette: vec4<u32>) -> MaterialPick {
    if (palette.y == PALETTE_EMPTY) {
        return MaterialPick(palette.x, palette.x, 1.0);
    }
    return pick_material(load_material_distances(base_u, world_pos, lod), palette);
}

fn sample_brick_sdf(base_u: u32, world_pos: vec3<f32>, lod: u32) -> f32 {
    let voxel_size = voxel_size_at(lod);

    // Continuous voxel-space position, then split into the brick-local integer
    // corner and the sub-voxel fraction used for trilinear interpolation.
    // Brick stride is `stride` cells; sample `stride` (the apron) is shared with
    // the neighbour, so local coords stay in [0, stride) and i0+1 never exceeds it.
    let s = cell_stride();
    let voxel_f = world_pos / voxel_size;
    // Brick origin via exact integer floor-div (matches world_to_brick_lod). A float
    // `floor(voxel_f/stride)*stride` rounds brick-boundary points one brick too low →
    // local index 7 (apron) → trilinear corners collapse onto the apron → seam fragments.
    let vox = vec3<i32>(floor(voxel_f));
    let brick_origin_voxel = vec3<f32>(
        f32(brick_snap(vox.x, s)),
        f32(brick_snap(vox.y, s)),
        f32(brick_snap(vox.z, s)),
    );
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
    // Decode the per-LOD voxel-unit clamp: the bake stored `d / (DIST_BAND_VOXELS·voxel_size)`
    // as snorm (atlas::dist_to_snorm_band), so multiply back to recover world distance. A
    // coarse LOD's large band lets the sphere-trace take big steps far from the surface.
    return mix(y0, y1, fz) * (DIST_BAND_VOXELS * voxel_size);
}

// Find the finest LOD with a baked brick at `world_pos`. Returns the brick location plus
// the LOD it was found at (via `out_lod`). Misses (`found == false`) when no LOD has a
// brick here (empty space).
fn find_brick_at(world_pos: vec3<f32>, out_lod: ptr<function, u32>) -> BrickLocation {
#ifdef SDF_DISABLE_LOD
    let levels = 1u;                 // diagnostic: LOD 0 only (match resolve_march)
#else
    let levels = lod_count();
#endif
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
    // independent of geometric smoothing — clean even at smoothing = 0. Single-material
    // bricks short-circuit the per-slot fetch (see `resolve_material`).
    let pick = resolve_material(loc.atlas_base, p, lod, loc.palette);

    return SceneSdfResult(d, pick.id, pick.id_b, pick.gap, true, lod, loc.atlas_base, loc.palette);
}

// --- Cached march resolve (per-ray accessor) ---
//
// The raymarch loop resolves the scene every step. Resolving the brick AND the empty-space
// window LOD separately walks the LODs twice and binary-searches the chunk table on every
// probe. `resolve_march` does it in ONE coarse→fine walk and memoises the chunk
// search through a per-ray `ChunkCache`: a marching ray stays in the same chunk for many
// consecutive steps, so the dominant (serving) chunk probe becomes O(1). Shading/normal
// paths still use the plain `scene_sdf`/`sample_sdf_world` (called once at the hit, not in
// the loop), so they are unaffected.

// Per-LOD memo of the last `find_chunk` query (key → table index, hit OR miss). Per-LOD
// (not single-entry) because each resolve probes one chunk key PER LOD; a single entry
// would be overwritten within a step. A marching ray stays in the same chunk at each LOD
// for many consecutive steps, so each LOD's probe becomes O(1) until it crosses that LOD's
// chunk boundary. `MAX_LODS` bounds the array; lod_count() <= this (DEFAULT_LOD_COUNT = 8).
const MAX_LODS: u32 = 8u;

struct ChunkCache {
    key_hi: array<u32, MAX_LODS>,
    key_lo: array<u32, MAX_LODS>,
    index: array<i32, MAX_LODS>,
    valid: array<bool, MAX_LODS>,
};

fn new_chunk_cache() -> ChunkCache {
    var c: ChunkCache;
    for (var i = 0u; i < MAX_LODS; i = i + 1u) {
        c.valid[i] = false;
    }
    return c;
}

fn find_chunk_cached(coord: vec3<i32>, lod: u32, cache: ptr<function, ChunkCache>) -> i32 {
    let key = abs_chunk_key(coord, lod);   // the chunk's tag — identifies it across march steps
#ifndef SDF_DISABLE_CHUNK_CACHE
    if ((*cache).valid[lod] && (*cache).key_hi[lod] == key.x && (*cache).key_lo[lod] == key.y) {
        return (*cache).index[lod];
    }
#endif
    let ci = find_chunk(coord, lod);
    (*cache).valid[lod] = true;
    (*cache).key_hi[lod] = key.x;
    (*cache).key_lo[lod] = key.y;
    (*cache).index[lod] = ci;
    return ci;
}

// Cached counterparts of find_brick_lookup / find_brick_at / sample_sdf_world. Identical
// logic, but route the per-LOD chunk probe through `find_chunk_cached(&cache)` so a SECONDARY
// ray (shadow, or the 4 normal taps) that re-evaluates the field every step gets the same
// O(1)-within-a-chunk memo the primary march enjoys (the uncached versions binary-search the
// whole chunk buffer EVERY call). The result is bit-identical — the cache is a pure memo.
fn find_brick_lookup_cached(coord: vec3<i32>, lod: u32, cache: ptr<function, ChunkCache>) -> BrickLocation {
    let ci = find_chunk_cached(coord, lod, cache);
    if (ci < 0) {
        return BrickLocation(0u, vec4<u32>(PALETTE_EMPTY), false);
    }
    return brick_in_chunk(chunk_buf[u32(ci)], coord);
}

fn find_brick_at_cached(world_pos: vec3<f32>, out_lod: ptr<function, u32>, cache: ptr<function, ChunkCache>) -> BrickLocation {
#ifdef SDF_DISABLE_LOD
    let levels = 1u;
#else
    let levels = lod_count();
#endif
    for (var lod = 0u; lod < levels; lod = lod + 1u) {
        let coord = world_to_brick_lod(world_pos, lod);
        let loc = find_brick_lookup_cached(coord, lod, cache);
        if (loc.found) {
            *out_lod = lod;
            return loc;
        }
    }
    *out_lod = 0u;
    return BrickLocation(0u, vec4<u32>(PALETTE_EMPTY), false);
}

fn sample_sdf_world_cached(world_pos: vec3<f32>, cache: ptr<function, ChunkCache>) -> f32 {
    var lod = 0u;
    let loc = find_brick_at_cached(world_pos, &lod, cache);
    if (!loc.found) {
        return 1e10;
    }
    return sample_brick_sdf(loc.atlas_base, world_pos, lod);
}

// Everything the march loop needs from one resolve at `p`. `in_brick` false ⇒ empty space;
// `window_lod` is then the finest LOD with a resident chunk at `p` (the DDA step scale),
// or the coarsest LOD if none. On a hit, `lod` is the serving (finest occupied) LOD.
struct MarchSample {
    dist: f32,
    in_brick: bool,
    window_lod: u32,
    lod: u32,
    atlas_base: u32,
    palette: vec4<u32>,
};

// One fine→coarse walk: capture the finest resident-chunk LOD (window) and return at the
// finest OCCUPIED brick (matching `find_brick_at` semantics — a coarse brick still shows
// through where a finer chunk is resident-but-empty). Uses the cached chunk search.
fn resolve_march(p: vec3<f32>, cache: ptr<function, ChunkCache>) -> MarchSample {
#ifdef SDF_DISABLE_LOD
    let levels = 1u;                 // diagnostic: LOD 0 only
#else
    let levels = lod_count();
#endif
    let count = arrayLength(&chunk_buf);   // buffer's own length, not the (possibly stale) uniform bound
    var window_lod = levels - 1u;
    var has_window = false;

    if (count != 0u) {
        for (var lod = 0u; lod < levels; lod = lod + 1u) {
            let coord = world_to_brick_lod(p, lod);
            let ci = find_chunk_cached(coord, lod, cache);
            if (ci >= 0) {
                if (!has_window) {
                    window_lod = lod;
                    has_window = true;
                }
                let loc = brick_in_chunk(chunk_buf[u32(ci)], coord);
                if (loc.found) {
                    let d = sample_brick_sdf(loc.atlas_base, p, lod);
                    return MarchSample(d, true, window_lod, lod, loc.atlas_base, loc.palette);
                }
            }
        }
    }
    return MarchSample(1e10, false, window_lod, 0u, 0u, vec4<u32>(PALETTE_EMPTY));
}

// Sample the conservative field at an ABSOLUTE target LOD, degrading to COARSER only.
//
// Unlike `resolve_march` (which serves the finest occupied LOD, so a finer occupancy
// ISLAND shows through inside a coarser region), this samples exactly `target` if it is
// occupied, else walks COARSER (target+1, target+2, …) until an occupied brick is found.
// It NEVER returns a finer-than-target sample. That is the whole point: the LOD cross-fade
// drives `target` purely from camera DISTANCE (continuous in screen space), so the level
// being sampled no longer depends on which LOD `resolve_march` happened to find occupied —
// killing the served-LOD-flip weight discontinuity that caused the hard blend seam. `found`
// is false only in genuine empty space (no LOD has a brick at `p`). Uses the per-ray cache.
fn sample_level_at_or_coarser(p: vec3<f32>, target_lod: u32, cache: ptr<function, ChunkCache>) -> MarchSample {
    let levels = lod_count();
    if (target_lod >= levels || arrayLength(&chunk_buf) == 0u) {
        return MarchSample(1e10, false, levels - 1u, 0u, 0u, vec4<u32>(PALETTE_EMPTY));
    }
    for (var lod = target_lod; lod < levels; lod = lod + 1u) {
        let coord = world_to_brick_lod(p, lod);
        let ci = find_chunk_cached(coord, lod, cache);
        if (ci >= 0) {
            let loc = brick_in_chunk(chunk_buf[u32(ci)], coord);
            if (loc.found) {
                let d = sample_brick_sdf(loc.atlas_base, p, lod);
                return MarchSample(d, true, lod, lod, loc.atlas_base, loc.palette);
            }
        }
    }
    return MarchSample(1e10, false, levels - 1u, 0u, 0u, vec4<u32>(PALETTE_EMPTY));
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

// Distance along the ray to the far side of the CHUNK containing `p`, at LOD `lod`. The
// chunk-DDA empty-space skip uses this to step across a whole provably-empty chunk box in
// one jump (a chunk is CHUNK_BRICKS bricks per axis). Identical slab test to
// `dist_to_brick_exit_lod`, scaled to chunk size.
fn dist_to_chunk_exit_lod(p: vec3<f32>, dir: vec3<f32>, lod: u32) -> f32 {
    let chunk_world = f32(CHUNK_BRICKS) * brick_world_at(lod);

    let chunk_min = floor(p / chunk_world) * chunk_world;
    let chunk_max = chunk_min + vec3<f32>(chunk_world);

    var t = 1e10;
    for (var a = 0u; a < 3u; a = a + 1u) {
        let d = dir[a];
        if (abs(d) > 1e-6) {
            let bound = select(chunk_min[a], chunk_max[a], d > 0.0);
            let ta = (bound - p[a]) / d;
            if (ta > 0.0) {
                t = min(t, ta);
            }
        }
    }
    return t;
}

// Occupancy-aware brick-DDA WITHIN one resident chunk. When the ray is in empty space yet inside an
// OCCUPIED (resident) chunk — air ABOVE the terrain that lives in the same chunk as the surface
// bricks below it — plain `dist_to_brick_exit_lod` crawls ONE brick per outer march step (the
// "horizon crawl" that exhausts the step budget on grazing rays). This walks the ray across the run
// of CONSECUTIVE EMPTY bricks in one shot, reading only the chunk's 64-bit occupancy mask (no field
// samples), and returns the distance to the near face of the next OCCUPIED brick, or to the chunk
// exit if none lie ahead on the ray (the caller re-resolves there, where the hierarchical chunk-DDA
// can skip further). Bounded to a chunk diagonal (≤ 3·CHUNK_BRICKS bricks).
//
// SAFE because the caller invokes it on the COARSEST resident chunk at `p`: a brick empty at a coarse
// LOD is empty at ALL finer LODs (the same geometry bakes at every level, so present-at-fine ⇒
// present-at-coarse ⇒ absent-at-coarse ⇒ absent-at-fine), so skipping an empty coarse brick can never
// step over a finer-LOD surface.
fn dist_over_empty_bricks(chunk: ChunkLookup, p: vec3<f32>, dir: vec3<f32>, lod: u32) -> f32 {
    let key_hi = chunk.key_hi;
    let key_lo = chunk.key_lo;
    let eps = voxel_size_at(lod) * 0.01;
    var adv = 0.0;
    let max_bricks = 3u * u32(CHUNK_BRICKS);
    for (var i = 0u; i < max_bricks; i = i + 1u) {
        let q = p + dir * (adv + eps);
        let coord = world_to_brick_lod(q, lod);
        let qkey = abs_chunk_key(coord, lod);
        if (qkey.x != key_hi || qkey.y != key_lo) {
            return adv;                        // left this chunk → caller re-resolves at the next one
        }
        if (brick_in_chunk(chunk, coord).found) {
            return adv;                        // occupied brick ahead → caller sphere-traces it
        }
        adv = adv + dist_to_brick_exit_lod(q, dir, lod) + eps;   // empty brick → step over it
    }
    return adv;
}

// True if the chunk containing brick `coord` at `lod` is inside that LOD's resident ring
// window. Mirrors bake_scheduler::ring_chunk_origin EXACTLY (camera chunk minus half-ring,
// snapped to the recenter_snap_chunks lattice) — must match or the chunk-DDA skip will
// mis-classify chunks near the ring edge. All integer math via floor_div (never raw `%`/`/`,
// the GPU signed-op hazard — see bindings.wgsl floor_div).
//
// Distinguishes "empty-culled" (in-ring + absent ⇒ provably empty ⇒ safe to skip) from
// "unbaked" (out-of-ring ⇒ unknown ⇒ a coarser LOD's ring covers it; not skipped here).
fn in_ring_chunk(coord: vec3<i32>, lod: u32) -> bool {
    let s = cell_stride();
    let c = CHUNK_BRICKS;
    let r = ring_bricks() / c;                  // ring chunks per axis

    // Camera chunk coord at this LOD (brick → brick-index → chunk-index, Euclidean).
    let cam_brick = world_to_brick_lod(camera.camera_pos.xyz, lod);
    let cam_cx = floor_div(floor_div(cam_brick.x, s), c);
    let cam_cy = floor_div(floor_div(cam_brick.y, s), c);
    let cam_cz = floor_div(floor_div(cam_brick.z, s), c);

    // Hysteresis snap to the coarse recenter lattice (mirrors ring_chunk_origin).
    let snap = recenter_snap();  // already max(,1)
    let scx = floor_div(cam_cx, snap) * snap;
    let scy = floor_div(cam_cy, snap) * snap;
    let scz = floor_div(cam_cz, snap) * snap;

    let half = r / 2;
    let ox = scx - half;
    let oy = scy - half;
    let oz = scz - half;

    // p's own chunk coord at this LOD.
    let cx = floor_div(floor_div(coord.x, s), c);
    let cy = floor_div(floor_div(coord.y, s), c);
    let cz = floor_div(floor_div(coord.z, s), c);

    let rx = cx - ox;
    let ry = cy - oy;
    let rz = cz - oz;
    return rx >= 0 && ry >= 0 && rz >= 0 && rx < r && ry < r && rz < r;
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
    // The 5 probes (center LOD + 4 tetrahedron taps) are spatially adjacent → almost always
    // the same chunk, so a shared per-call cache turns 5 chunk binary-searches into ~1.
    var cache = new_chunk_cache();
    var hit_lod = 0u;
    let loc = find_brick_at_cached(p, &hit_lod, &cache);
    let h = voxel_size_at(hit_lod);
    let k = vec2<f32>(1.0, -1.0);
    let n = k.xyy * sample_sdf_world_cached(p + k.xyy * h, &cache)
          + k.yyx * sample_sdf_world_cached(p + k.yyx * h, &cache)
          + k.yxy * sample_sdf_world_cached(p + k.yxy * h, &cache)
          + k.xxx * sample_sdf_world_cached(p + k.xxx * h, &cache);
    if (dot(n, n) > 1e-12) {
        return normalize(n);
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}
