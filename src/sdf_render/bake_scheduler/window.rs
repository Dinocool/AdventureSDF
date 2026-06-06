//! Pure integer chunk-ring window geometry: where each per-LOD window sits ([`ring_chunk_origin`]),
//! which chunks entered/exited on a recenter, which chunks an edit AABB touches (clamped to the
//! window), and a BVH occupancy probe. No `BakeScheduler`/`SdfAtlas` state — just `(config, coords,
//! bvh)` in, chunk coords out — so it is independently testable. Reaches the shared `sdf_render` types
//! (`SdfGridConfig`, `chunk`, `atlas`, `bvh`) via `use super::*`.

use super::*;

/// The chunk coord (per axis) of the chunk-ring window corner for `camera_pos` at `lod`:
/// the camera's chunk minus half the ring (in chunks) on each axis, so the ring is
/// centred on the camera. `ring_bricks / CHUNK_BRICKS` chunks per axis.
///
/// `pub` so the GPU rig (`tests/sdf_gpu_rig.rs`) and the editor LOD-ring overlay can assert/draw the
/// SAME source-of-truth window the shader's `in_ring_chunk` is hand-duplicated from — a silent
/// divergence would make the chunk-DDA skip step past real geometry. Re-exported from [`super`].
pub fn ring_chunk_origin(config: &SdfGridConfig, camera_pos: Vec3, lod: u32) -> IVec3 {
    let s = config.cell_stride();
    let cam_brick = config.world_to_brick_lod(camera_pos, lod);
    let cam_brick_idx = IVec3::new(
        cam_brick.x.div_euclid(s),
        cam_brick.y.div_euclid(s),
        cam_brick.z.div_euclid(s),
    );
    let cam_chunk = IVec3::new(
        cam_brick_idx.x.div_euclid(chunk::CHUNK_BRICKS),
        cam_brick_idx.y.div_euclid(chunk::CHUNK_BRICKS),
        cam_brick_idx.z.div_euclid(chunk::CHUNK_BRICKS),
    );
    // Hysteresis: snap the camera chunk to the coarse `recenter_snap_chunks` lattice so
    // the window only recenters on discrete jumps, not every chunk crossing. `div_euclid`
    // floors toward -inf so the snapped lattice is continuous across the world origin.
    let snap = config.recenter_snap_chunks.max(1);
    let cam_chunk_snapped = IVec3::new(
        cam_chunk.x.div_euclid(snap) * snap,
        cam_chunk.y.div_euclid(snap) * snap,
        cam_chunk.z.div_euclid(snap) * snap,
    );
    let half = config.ring_half_chunks();
    cam_chunk_snapped - IVec3::splat(half)
}

/// Whether any edit reaches chunk `ck`, reusing a caller-owned BVH traversal `stack` (cleared on
/// entry) so a recenter that runs thousands of these per snap frame does zero heap allocation per
/// query. A chunk's world AABB is exactly the union of its `CHUNK_BRICKS³` brick AABBs, so a BVH
/// miss here guarantees every brick in the chunk would bake to empty space — used to skip enqueuing
/// empty *entered* chunks (safe only for entered chunks: no resident bricks to evict yet). Uses the
/// BVH's EARLY-EXIT `any_overlap_with` (stops at the first overlapping leaf — an occupancy boolean
/// is all this needs), which roughly halves the per-query cost on tower-dense chunks.
pub(super) fn chunk_has_geometry_with(
    ck: chunk::ChunkKey,
    bvh: &bvh::Bvh,
    config: &SdfGridConfig,
    stack: &mut Vec<u32>,
) -> bool {
    let size = chunk::chunk_world_size(ck.lod, config);
    let min = chunk::chunk_min_world(ck, config);
    let aabb = bevy::math::bounding::Aabb3d::from_min_max(min, min + Vec3::splat(size));
    bvh.any_overlap_with(&aabb, stack)
}

/// The conservative 64-bit per-brick occupancy mask for chunk `ck`: bit `local` set iff the geometry's
/// SURFACE plausibly crosses local brick `local`. This is the empty-space DDA's traversal grid —
/// maintained for the FULL clipmap ring at every LOD, decoupled from baked residency.
///
/// Uses the EXACT same test as the bake's classify (`cull_edit_indices` → [`narrow_band_keep`]), NOT a
/// raw AABB overlap: a brick is occupied only if the folded analytic distance places the surface within
/// it (+ a conservative band), so the mask FOLLOWS the surface instead of filling a geometry's whole
/// bounding box. (An AABB test marked tall/large edits' empty box interior — e.g. the heightmap's
/// AABB far above its terrain — as occupied, stalling sky rays brick-by-brick.) Because it reuses the
/// bake's keep test, `cons_occ ⊇ baked occ` (the DDA never skips a sampled surface), and because it uses
/// the analytic field it has no trilinear-shrinkage gap, so coarse-empty ⇒ fine-empty. `scratch`/`stack`
/// are caller-owned reusable buffers (zero per-call alloc).
pub(super) fn chunk_conservative_mask(
    ck: chunk::ChunkKey,
    edits: &[edits::ResolvedEdit],
    bvh: &bvh::Bvh,
    config: &SdfGridConfig,
    scratch: &mut Vec<u32>,
    stack: &mut Vec<u32>,
) -> u64 {
    let c = chunk::CHUNK_BRICKS;
    let s = config.cell_stride();
    let base = ck.coord * c; // the chunk's brick-(0,0,0) in brick-index space
    let mut mask = 0u64;
    for lz in 0..c {
        for ly in 0..c {
            for lx in 0..c {
                let bi = base + IVec3::new(lx, ly, lz);
                let key = atlas::BrickKey::new(ck.lod, bi * s);
                // Cull edits to this brick via the BVH (fills `scratch`); None ⇒ no geometry reaches it.
                if atlas::SdfAtlas::cull_edit_indices_with(key, bvh, config, scratch, stack).is_none() {
                    continue;
                }
                if narrow_band_keep(edits, scratch, config, key) {
                    mask |= 1u64 << local_bit(lx, ly, lz);
                }
            }
        }
    }
    mask
}

/// Whether chunk coord `c` is inside the `R³` chunk window with corner `origin`.
pub(super) fn chunk_in_window(c: IVec3, origin: IVec3, r: i32) -> bool {
    let rel = c - origin;
    rel.x >= 0 && rel.y >= 0 && rel.z >= 0 && rel.x < r && rel.y < r && rel.z < r
}

/// Every chunk key in the `R³` window with corner `origin` at `lod`. Production no longer dirties
/// the whole window (it derives dirt from edit footprints — see `dirty_edit_footprints`); the
/// settle/cull unit tests still use it to dirty a known region directly.
#[cfg(test)]
pub(super) fn chunk_window_keys(origin: IVec3, r: i32, lod: u32) -> impl Iterator<Item = chunk::ChunkKey> {
    (0..r).flat_map(move |iz| {
        (0..r).flat_map(move |iy| {
            (0..r).map(move |ix| chunk::ChunkKey::new(lod, origin + IVec3::new(ix, iy, iz)))
        })
    })
}

/// Invoke `f` only for the chunk coords in the `R³` window at `new_origin` that are NOT in the
/// `R³` window at `old_origin` (the chunks that *entered* on a recenter). For two equal-size
/// axis-aligned windows the difference is a thin boundary slab, so this visits only the new shell
/// (a few hundred chunks) instead of the full `R³` interior (4096 at R=16) — the recenter only
/// cares about entered chunks, and the overlap is unchanged. Yields nothing when the windows are
/// identical; yields the whole new window when they don't overlap (a teleport). Each coord is
/// visited at most once (axis-partitioned: x-slabs, then the x-overlap's y-slabs, then the xy-
/// overlap's z-slabs), so no dedup is needed.
pub(super) fn for_each_entered_chunk(new_origin: IVec3, old_origin: IVec3, r: i32, mut f: impl FnMut(IVec3)) {
    // Overlap box of the two windows on each axis (empty if they don't overlap). Use saturating
    // arithmetic so a sentinel old_origin (i32::MIN on the first run) can't overflow `+ r`; it just
    // yields an empty overlap → the whole new window is "entered", which is the correct first-run
    // behaviour (and `for_each_exited_chunk` then evicts nothing, since the old window is degenerate).
    let new_end = IVec3::new(
        new_origin.x.saturating_add(r),
        new_origin.y.saturating_add(r),
        new_origin.z.saturating_add(r),
    );
    let old_end = IVec3::new(
        old_origin.x.saturating_add(r),
        old_origin.y.saturating_add(r),
        old_origin.z.saturating_add(r),
    );
    let ov_min = new_origin.max(old_origin);
    let ov_max = new_end.min(old_end);
    // x in new-window but outside the x-overlap → whole yz cross-section entered.
    let x_overlap_empty = ov_min.x >= ov_max.x;
    for x in new_origin.x..new_end.x {
        let x_entered = x_overlap_empty || x < ov_min.x || x >= ov_max.x;
        if x_entered {
            // Entire yz face at this x is new.
            for y in new_origin.y..new_origin.y + r {
                for z in new_origin.z..new_origin.z + r {
                    f(IVec3::new(x, y, z));
                }
            }
        } else {
            // x is shared; partition y the same way, then z within the xy-overlap.
            let y_overlap_empty = ov_min.y >= ov_max.y;
            for y in new_origin.y..new_origin.y + r {
                let y_entered = y_overlap_empty || y < ov_min.y || y >= ov_max.y;
                if y_entered {
                    for z in new_origin.z..new_origin.z + r {
                        f(IVec3::new(x, y, z));
                    }
                } else {
                    // x,y shared; only the z-slab outside the z-overlap entered.
                    let z_overlap_empty = ov_min.z >= ov_max.z;
                    for z in new_origin.z..new_origin.z + r {
                        if z_overlap_empty || z < ov_min.z || z >= ov_max.z {
                            f(IVec3::new(x, y, z));
                        }
                    }
                }
            }
        }
    }
}

/// Invoke `f` for the chunk coords that *exited* on a recenter — in the OLD window but not the
/// new. Symmetric to [`for_each_entered_chunk`] (args swapped), used to evict the trailing shell.
pub(super) fn for_each_exited_chunk(new_origin: IVec3, old_origin: IVec3, r: i32, f: impl FnMut(IVec3)) {
    for_each_entered_chunk(old_origin, new_origin, r, f);
}

// --- Hollow-shell clipmap residency ({native .. native+overlap_depth}) ----------------------------
//
// Each LOD's resident region is its outer ring box MINUS a central inner hole that two-or-more FINER
// LODs already cover, so every world point is resident at exactly its native LOD plus `overlap_depth`
// coarser levels — not the full LOD stack. The hole shares the ring's (snapped) camera centre and
// tracks it on recenter, so the per-frame entered/exited sets stay thin slabs (reusing the box-diff
// `for_each_entered_chunk` twice: outer rim + hole boundary).

/// Inner-hole half-extent (in chunks) for LOD `lod`: the central region a level `lod-overlap_depth-1`
/// (the coarsest level still finer than the kept `{native..native+overlap_depth}` set) already covers,
/// so LOD `lod` need not be resident there. Returns 0 for the finest `overlap_depth + 1` levels — they
/// stay solid boxes (LOD 0 keeps its ENTIRE ring), there being no finer level to defer to.
///
/// Derivation: that finer level's ring half-extent, expressed in LOD-`lod` chunks, is
/// `ring_half_chunks >> (overlap_depth + 1)` — each coarser level halves the chunk count for a fixed
/// world extent. The hole sits well inside LOD `lod-1`'s coverage (the `+overlap_depth` fallback), with
/// a margin far larger than the `recenter_snap_chunks` hysteresis, so the composed shells leave NO gap.
pub(super) fn inner_hole_half_chunks(config: &SdfGridConfig, lod: u32) -> i32 {
    if lod < config.overlap_depth + 1 {
        0
    } else {
        config.ring_half_chunks() >> (config.overlap_depth + 1)
    }
}

/// The corner of LOD `lod`'s inner hole, given its outer ring corner `outer_origin`. The hole is the
/// `2·hole_half` box concentric with the ring; degenerate (`hole_half == 0`) for the finest levels.
#[inline]
fn inner_hole_origin(config: &SdfGridConfig, lod: u32, outer_origin: IVec3) -> IVec3 {
    outer_origin + IVec3::splat(config.ring_half_chunks() - inner_hole_half_chunks(config, lod))
}

/// Whether chunk `ck` lies in its LOD's inner hole (covered by finer LODs ⇒ NOT resident). Used to keep
/// hole chunks out of the dirty/bake sets and the carry queue.
pub(super) fn chunk_in_inner_hole(config: &SdfGridConfig, camera_pos: Vec3, ck: chunk::ChunkKey) -> bool {
    let hole = inner_hole_half_chunks(config, ck.lod);
    if hole == 0 {
        return false;
    }
    let outer = ring_chunk_origin(config, camera_pos, ck.lod);
    chunk_in_window(ck.coord, inner_hole_origin(config, ck.lod, outer), 2 * hole)
}

/// Whether chunk coord `c` is in LOD `lod`'s resident SHELL = outer ring box `[outer_origin, +r)` MINUS
/// the inner hole. (With `hole_half == 0` the hole is empty and the shell is the full box.)
#[inline]
pub(super) fn chunk_in_shell(config: &SdfGridConfig, lod: u32, c: IVec3, outer_origin: IVec3) -> bool {
    let r = config.ring_chunks_per_axis();
    if !chunk_in_window(c, outer_origin, r) {
        return false;
    }
    let hole = inner_hole_half_chunks(config, lod);
    hole == 0 || !chunk_in_window(c, inner_hole_origin(config, lod, outer_origin), 2 * hole)
}

/// Invoke `f` for the chunks that ENTERED LOD `lod`'s resident SHELL when its window moved
/// `old_outer → new_outer` (the hole tracks the same centre). Entered = chunks the OUTER box gained
/// ∪ chunks the receding INNER hole uncovered — two disjoint slabs (outer rim vs. centre), each a
/// same-size box difference reusing [`for_each_entered_chunk`]. On `first_run` the whole new shell is
/// entered (sentinel `old_outer` has no meaningful hole).
pub(super) fn for_each_entered_shell(
    config: &SdfGridConfig,
    lod: u32,
    new_outer: IVec3,
    old_outer: IVec3,
    first_run: bool,
    mut f: impl FnMut(IVec3),
) {
    let r = config.ring_chunks_per_axis();
    let hole = inner_hole_half_chunks(config, lod);
    if hole == 0 {
        // Solid box (finest levels) — exactly the pre-shell behaviour (handles the first-run sentinel).
        for_each_entered_chunk(new_outer, old_outer, r, f);
        return;
    }
    let hsize = 2 * hole;
    let new_hole = inner_hole_origin(config, lod, new_outer);
    if first_run {
        for_each_shell_chunk(new_outer, r, new_hole, hsize, f);
        return;
    }
    let old_hole = inner_hole_origin(config, lod, old_outer);
    // Outer box gained — at the outer rim, never inside the small central hole.
    for_each_entered_chunk(new_outer, old_outer, r, &mut f);
    // Hole uncovered — in the OLD hole but not the NEW one ⇒ now back in the shell.
    for_each_entered_chunk(old_hole, new_hole, hsize, &mut f);
}

/// Invoke `f` for the chunks that EXITED LOD `lod`'s resident SHELL on `old_outer → new_outer`: chunks
/// the OUTER box lost ∪ chunks the advancing INNER hole newly covers (now redundant → evict). Symmetric
/// to [`for_each_entered_shell`]; never called on the first run (nothing was resident yet).
pub(super) fn for_each_exited_shell(
    config: &SdfGridConfig,
    lod: u32,
    new_outer: IVec3,
    old_outer: IVec3,
    mut f: impl FnMut(IVec3),
) {
    let r = config.ring_chunks_per_axis();
    let hole = inner_hole_half_chunks(config, lod);
    if hole == 0 {
        for_each_exited_chunk(new_outer, old_outer, r, f);
        return;
    }
    let hsize = 2 * hole;
    let new_hole = inner_hole_origin(config, lod, new_outer);
    let old_hole = inner_hole_origin(config, lod, old_outer);
    // Outer box lost.
    for_each_exited_chunk(new_outer, old_outer, r, &mut f);
    // Hole newly covered — in the NEW hole but not the OLD one ⇒ left the shell (redundant now).
    for_each_entered_chunk(new_hole, old_hole, hsize, &mut f);
}

/// Invoke `f` for every chunk in the shell `[outer_origin, +r)` minus the `[hole_origin, +hsize)` hole.
/// O(r³) — used only on the one-time first-run fill; the incremental recenter uses the slab diffs above.
fn for_each_shell_chunk(outer_origin: IVec3, r: i32, hole_origin: IVec3, hsize: i32, mut f: impl FnMut(IVec3)) {
    for iz in 0..r {
        for iy in 0..r {
            for ix in 0..r {
                let c = outer_origin + IVec3::new(ix, iy, iz);
                if hsize == 0 || !chunk_in_window(c, hole_origin, hsize) {
                    f(c);
                }
            }
        }
    }
}

/// All brick keys belonging to chunk `ck` (its `CHUNK_BRICKS³` local slots).
#[cfg(test)]
pub(super) fn chunk_brick_keys(ck: chunk::ChunkKey, config: &SdfGridConfig) -> Vec<atlas::BrickKey> {
    let mut keys = Vec::with_capacity(chunk::CHUNK_VOLUME as usize);
    for_each_brick_key(ck, config, |k| keys.push(k));
    keys
}

/// A per-chunk dirty-brick mask with every one of the chunk's `CHUNK_VOLUME` (64) local slots set.
/// Used when a whole chunk is dirtied (a recenter-entered chunk, or a resident chunk re-examined on a
/// structural rebake) — equivalent to the old whole-chunk `pending` membership.
pub(super) const FULL_CHUNK_MASK: u64 = u64::MAX;

/// The local 0..63 bit index of brick local-coord `(lx,ly,lz)` within its chunk — the SAME packing
/// `chunk::chunk_of` produces (`z·16 + y·4 + x`), so a mask bit and a `chunk_of` local slot agree.
#[inline]
fn local_bit(lx: i32, ly: i32, lz: i32) -> u32 {
    (lz * chunk::CHUNK_BRICKS * chunk::CHUNK_BRICKS + ly * chunk::CHUNK_BRICKS + lx) as u32
}

/// Invoke `f` for each brick key whose local slot is set in `mask` — the brick-granular counterpart of
/// [`for_each_brick_key`]. Iterates only the set bits (a moved edit's thin footprint touches a handful),
/// so an almost-empty mask costs almost nothing. Bit `i` ⇒ local `(x=i&3, y=(i>>2)&3, z=(i>>4)&3)`,
/// matching [`local_bit`]/`chunk_of`.
#[inline]
pub(super) fn for_each_brick_key_masked(
    ck: chunk::ChunkKey,
    mask: u64,
    config: &SdfGridConfig,
    mut f: impl FnMut(atlas::BrickKey),
) {
    let s = config.cell_stride();
    let c = chunk::CHUNK_BRICKS;
    let base = ck.coord * c; // brick-index space
    let mut bits = mask;
    while bits != 0 {
        let bit = bits.trailing_zeros() as i32;
        let lx = bit & (c - 1);
        let ly = (bit >> 2) & (c - 1);
        let lz = (bit >> 4) & (c - 1);
        let bi = base + IVec3::new(lx, ly, lz);
        f(atlas::BrickKey::new(ck.lod, bi * s)); // back to coord space
        bits &= bits - 1; // clear lowest set bit
    }
}

/// The 64-bit mask of every local brick in the inclusive local box `[lo, hi]` (each axis
/// `0..CHUNK_BRICKS`), bit `z·16 + y·4 + x` — the bricks a chunk-clipped AABB covers. A full box
/// short-circuits to [`FULL_CHUNK_MASK`] so an interior chunk costs O(1).
fn local_box_mask(lo: IVec3, hi: IVec3) -> u64 {
    if lo == IVec3::ZERO && hi == IVec3::splat(chunk::CHUNK_BRICKS - 1) {
        return FULL_CHUNK_MASK;
    }
    let mut m = 0u64;
    for z in lo.z..=hi.z {
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                m |= 1u64 << local_bit(x, y, z);
            }
        }
    }
    m
}

/// The BRICK-granular footprint of `aabb` at `lod`: for every chunk the padded AABB reaches (clamped
/// to the ring window `[win_origin, win_origin + r)`), the `u64` mask of exactly the bricks the AABB
/// overlaps — NOT all 64 bricks of the chunk. The brick-resolution counterpart of
/// [`chunks_in_aabb_windowed`]: the dirty set is the same world region, but tracked per brick so a
/// moved edit re-classifies only the bricks it actually touches instead of every brick of every
/// straddled chunk. The pad is identical to the chunk version, so the dirtied bricks are a strict
/// SUBSET of that version's chunks' bricks — no brick that should bake is ever dropped.
///
/// Iterates CHUNKS (not bricks): interior chunks get [`FULL_CHUNK_MASK`] in O(1), only the boundary
/// shell does per-brick bit work, so it is O(chunks) — a window-spanning AABB never enumerates its
/// millions of bricks. (Used for the structural/cold dirty; the hot MOVE path uses the surface-pruned
/// `dirty_moving_edit`, which dirties only the moving shell, not the solid interior.)
pub(super) fn bricks_in_aabb_windowed(
    config: &SdfGridConfig,
    aabb: &bevy::math::bounding::Aabb3d,
    lod: u32,
    win_origin: IVec3,
    r: i32,
) -> Vec<(chunk::ChunkKey, u64)> {
    let brick_world = config.brick_world_size(lod);
    let pad = Vec3::splat(atlas::SNORM_CLAMP_DIST + brick_world);
    let lo_w = Vec3::from(aabb.min) - pad;
    let hi_w = Vec3::from(aabb.max) + pad;
    let c = chunk::CHUNK_BRICKS;
    // Brick-index range (per axis: the brick containing lo through the one containing hi), clamped to
    // the window's brick range up front so a terrain-scale AABB costs O(window), not O(AABB).
    let win_lo = win_origin * c;
    let win_hi = (win_origin + IVec3::splat(r)) * c - IVec3::ONE;
    let bi_lo = ivec_floor_div(lo_w, brick_world).max(win_lo);
    let bi_hi = ivec_floor_div(hi_w, brick_world).min(win_hi);
    if bi_lo.x > bi_hi.x || bi_lo.y > bi_hi.y || bi_lo.z > bi_hi.z {
        return Vec::new();
    }
    // Enumerate the CHUNKS the brick-range spans; clip each to the range for its mask.
    let ci_lo = IVec3::new(bi_lo.x.div_euclid(c), bi_lo.y.div_euclid(c), bi_lo.z.div_euclid(c));
    let ci_hi = IVec3::new(bi_hi.x.div_euclid(c), bi_hi.y.div_euclid(c), bi_hi.z.div_euclid(c));
    let mut out: Vec<(chunk::ChunkKey, u64)> = Vec::new();
    let last = IVec3::splat(c - 1);
    for cz in ci_lo.z..=ci_hi.z {
        for cy in ci_lo.y..=ci_hi.y {
            for cx in ci_lo.x..=ci_hi.x {
                let cc = IVec3::new(cx, cy, cz);
                let chunk_b0 = cc * c; // this chunk's brick-(0,0,0) index
                let llo = bi_lo.max(chunk_b0) - chunk_b0; // local box lo, 0..c-1
                let lhi = bi_hi.min(chunk_b0 + last) - chunk_b0; // local box hi
                out.push((chunk::ChunkKey::new(lod, cc), local_box_mask(llo, lhi)));
            }
        }
    }
    out
}

/// Per-axis `floor(v / d)` as an `IVec3` — the brick index containing each world coordinate.
#[inline]
pub(super) fn ivec_floor_div(v: Vec3, d: f32) -> IVec3 {
    IVec3::new(
        (v.x / d).floor() as i32,
        (v.y / d).floor() as i32,
        (v.z / d).floor() as i32,
    )
}

/// Allocation-free counterpart of [`chunk_brick_keys`]: invoke `f` for each of a chunk's 64
/// brick keys without building a Vec. The bake emit's serial gather/apply loops run this over
/// the entire dirty set (thousands of chunks on a terrain-scale heightmap move), so avoiding a
/// per-chunk 64-element heap alloc there is a measurable win (emit phases 1+3 were ~20ms spikes).
#[inline]
pub(super) fn for_each_brick_key(ck: chunk::ChunkKey, config: &SdfGridConfig, mut f: impl FnMut(atlas::BrickKey)) {
    let s = config.cell_stride();
    let c = chunk::CHUNK_BRICKS;
    let base = ck.coord * c; // brick-index space
    for lz in 0..c {
        for ly in 0..c {
            for lx in 0..c {
                let bi = base + IVec3::new(lx, ly, lz);
                f(atlas::BrickKey::new(ck.lod, bi * s)); // back to coord space
            }
        }
    }
}

/// The chunks at `lod` whose world extent overlaps `aabb` (grown by the bake footprint
/// pad so a moved edit re-dirties every chunk that could fold it), **clamped to the resident
/// ring window** `[win_origin, win_origin + r)`. Computed directly in chunk-coord space — no
/// per-brick enumeration.
///
/// The clamp is essential for terrain-scale edits: a heightmap's AABB spans the whole world
/// in XZ, so the unclamped chunk range is millions of chunks. The caller only ever keeps the
/// in-window ones anyway, so intersecting the loop bounds with the window up front makes the
/// work O(AABB ∩ window) ≤ r³ instead of O(AABB volume) — the fix for the multi-hundred-ms
/// `schedule_bakes` freeze when a heightmap edit changes (it used to allocate + enumerate the
/// entire terrain-sized chunk volume per LOD, then discard 99.99% via `chunk_in_window`).
///
/// Production now dirties at BRICK granularity (see [`bricks_in_aabb_windowed`]); this whole-chunk
/// version is retained for the settle/cull unit tests + the small-edit perf scenario.
#[cfg(test)]
pub(super) fn chunks_in_aabb_windowed(
    config: &SdfGridConfig,
    aabb: &bevy::math::bounding::Aabb3d,
    lod: u32,
    win_origin: IVec3,
    r: i32,
) -> Vec<chunk::ChunkKey> {
    let chunk_world = chunk::chunk_world_size(lod, config);
    let pad = Vec3::splat(atlas::SNORM_CLAMP_DIST + config.brick_world_size(lod));
    let lo = (Vec3::from(aabb.min) - pad) / chunk_world;
    let hi = (Vec3::from(aabb.max) + pad) / chunk_world;
    // Intersect the AABB chunk-range with the window box BEFORE enumerating.
    let lo = IVec3::new(lo.x.floor() as i32, lo.y.floor() as i32, lo.z.floor() as i32)
        .max(win_origin);
    let hi = IVec3::new(hi.x.ceil() as i32, hi.y.ceil() as i32, hi.z.ceil() as i32)
        .min(win_origin + IVec3::splat(r - 1));

    let mut out = Vec::new();
    for z in lo.z..=hi.z {
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                out.push(chunk::ChunkKey::new(lod, IVec3::new(x, y, z)));
            }
        }
    }
    out
}
