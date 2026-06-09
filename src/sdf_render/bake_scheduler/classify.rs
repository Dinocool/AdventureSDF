//! The read-only classify core: turn candidate bricks into [`Verdict`]s (Empty / Drop / Skip / Keep)
//! by BVH-culling each brick's edits, a narrow-band keep test, a content-hash cache peek, and a
//! palette build. Reads ONLY snapshots (edit list, BVH, config, a resident-hash peek) — no `SdfAtlas`
//! borrow — so it is `Send` and runs either across the compute pool ([`classify_candidates`]) or
//! serially on a background task ([`classify_candidates_serial`]). [`apply_verdicts`](super) consumes
//! the verdicts on the main thread. Reaches shared types via `use super::*`.

use super::*;
// FxHash for the per-candidate classify maps: the resident-hash peek (one lookup per brick) and the
// content-hash memo (one lookup per unique culled edit-set). Integer/byte keys; std SipHash is waste.
use rustc_hash::FxHashMap;

/// One candidate brick's classification, produced by [`classify_candidates`] (read-only, can run
/// on a background task) and consumed by [`apply_verdicts`](super) (mutates the atlas, main thread only).
pub(crate) enum Verdict {
    /// No edit reaches it → evict.
    Empty,
    /// Narrow-band cull → evict.
    Drop,
    /// Resident with matching content hash → leave its texels as-is.
    Skip,
    /// Bake: palette + culled edit indices + content hash.
    Keep(edits::Palette, Vec<u32>, u64),
}

/// Whether brick `key`'s surface plausibly passes through it: keep if the folded center distance is
/// within a conservative reach (circumradius + the LOD distance band + a smoothing pad), OR — when
/// smoothing inflates the gradient — if the surface provably crosses the 9 palette sample points
/// (8 corners + center). The `dist_band` term only ever makes us KEEP, never drop, so it can't
/// introduce a hole; it also catches an enclosed cavity the center eval would miss.
///
/// `pub(super)` so the conservative-occupancy mask (`window::chunk_conservative_mask`) reuses the EXACT
/// same surface test the bake's classify uses — guaranteeing the empty-space DDA's `cons_occ` is a
/// superset of the baked occupancy (never skips a sampled surface) and is surface-following, not
/// AABB-over-inclusive (which made tall/large edit AABBs — e.g. the heightmap — stop sky rays).
pub(super) fn narrow_band_keep(
    edits: &[edits::ResolvedEdit],
    indices: &[u32],
    config: &SdfGridConfig,
    key: atlas::BrickKey,
) -> bool {
    let brick_world = config.brick_world_size(key.lod);
    let brick_min = config.brick_min_world(key.coord, key.lod);
    let center = brick_min + Vec3::splat(0.5 * brick_world);

    // Smoothing pad: additive Σ(kᵢ)/4 over the brick's smoothed candidate edits.
    let smooth_sum: f32 = indices
        .iter()
        .map(|&i| edits[i as usize].op.smoothing.max(0.0))
        .sum();
    // CONSERVATIVE reach: circumradius + the LOD's distance band + smoothing pad. The
    // `dist_band` term is a deliberate safety margin — it only ever makes us KEEP a brick the
    // center test would otherwise drop, never the reverse, so it cannot introduce a hole. (It is
    // largely subsumed by `cull_edit_indices` already culling on the raw brick AABB, but is kept
    // because proving it strictly redundant at coarse-LOD corners is fragile and the cost is one
    // add. The over-keep halo it allows is a one-time first-bake cost, not a per-frame drag cost.)
    let reach = brick_world * (0.5 * 3.0_f32.sqrt())
        + atlas::dist_band_world(config, key.lod)
        + 0.25 * smooth_sum;

    // Force-keep if the surface provably crosses the brick (sign change over the 9 samples).
    // Only meaningful when smoothing inflates the gradient; the common k=0 path stays 1 eval.
    if smooth_sum > 0.0 {
        let voxel_size = config.voxel_size_at(key.lod);
        let samples = atlas::SdfAtlas::brick_palette_samples(key, voxel_size);
        let mut neg = false;
        let mut pos = false;
        for p in samples {
            if edits::fold_csg_dist_indexed(edits, indices, p, 0.0) <= 0.0 {
                neg = true;
            } else {
                pos = true;
            }
            if neg && pos {
                return true;
            }
        }
    }

    edits::fold_csg_dist_indexed(edits, indices, center, 0.0).abs() <= reach
}

/// Classify ONE chunk's slice of candidate bricks into `Verdict`s (read-only). The shared core of
/// both the parallel (sync) and serial (async task) classify paths. `scratch`/`stack`/`hash_memo`
/// are caller-owned reusable buffers (the memo lets an edit-set folded by many bricks hash once per
/// unique set). Reads only the edit/BVH snapshots, config, and the `hash_peek` resident-hash
/// snapshot — no atlas borrow — so it is `Send` and safe on a background task.
#[expect(clippy::too_many_arguments)]
fn classify_chunk(
    chunk: &[(chunk::ChunkKey, atlas::BrickKey)],
    edits: &[edits::ResolvedEdit],
    bvh: &bvh::Bvh,
    config: &SdfGridConfig,
    hash_peek: &FxHashMap<atlas::BrickKey, u64>,
    scratch: &mut Vec<u32>,
    stack: &mut Vec<u32>,
    hash_memo: &mut FxHashMap<Box<[u32]>, u64>,
    out: &mut Vec<Verdict>,
) {
    for &(_ck, key) in chunk {
        if atlas::SdfAtlas::cull_edit_indices_with(key, bvh, config, scratch, stack).is_none() {
            out.push(Verdict::Empty);
            continue;
        }
        if !narrow_band_keep(edits, scratch, config, key) {
            out.push(Verdict::Drop);
            continue;
        }
        // Content hash of exactly the edits this brick folds — its bake-cache key. Memoised per
        // unique culled index-set (the costly part is the same for identical sets).
        let hash = *hash_memo
            .entry(scratch.clone().into_boxed_slice())
            .or_insert_with(|| edits::bake_content_hash(edits, scratch));
        // HASH-PEEK EARLY-OUT: a resident brick with the same content keeps valid texels — skip
        // BEFORE the (dominant-cost) palette build. This is what makes a sphere dragged over the
        // heightmap cheap. The peek reads a SNAPSHOT of resident hashes (no atlas borrow); a stale
        // snapshot is harmless — a wrong Skip re-bakes identically next frame, a wrong Keep is
        // overwritten by `insert_gpu_brick`'s authoritative hash.
        if hash_peek.get(&key).is_some_and(|&h| h == hash) {
            out.push(Verdict::Skip);
            continue;
        }
        let voxel_size = config.voxel_size_at(key.lod);
        let samples = atlas::SdfAtlas::brick_palette_samples(key, voxel_size);
        let palette = edits::build_palette_indexed(edits, scratch, &samples, 0.0);
        out.push(Verdict::Keep(palette, scratch.clone(), hash));
    }
}

/// Phase 2 (PARALLEL, READ-ONLY): classify every candidate via `par_chunk_map` across the compute
/// pool. The SYNCHRONOUS bake path. Reads only snapshots — see [`classify_chunk`].
pub(crate) fn classify_candidates(
    candidates: &[(chunk::ChunkKey, atlas::BrickKey)],
    edits: &[edits::ResolvedEdit],
    bvh: &bvh::Bvh,
    config: &SdfGridConfig,
    hash_peek: &FxHashMap<atlas::BrickKey, u64>,
) -> Vec<Verdict> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let _g = info_span!("emit_phase2_classify", candidates = candidates.len()).entered();
    let classify = |_idx: usize, chunk: &[(chunk::ChunkKey, atlas::BrickKey)]| -> Vec<Verdict> {
        let mut scratch: Vec<u32> = Vec::new();
        let mut stack: Vec<u32> = Vec::new();
        let mut hash_memo: FxHashMap<Box<[u32]>, u64> = FxHashMap::default();
        let mut out = Vec::with_capacity(chunk.len());
        classify_chunk(chunk, edits, bvh, config, hash_peek, &mut scratch, &mut stack, &mut hash_memo, &mut out);
        out
    };
    // `get_or_init` so headless tests (which don't boot the full app that sets up the pool) still
    // run; in production the pool already exists.
    let pool = ComputeTaskPool::get_or_init(bevy::tasks::TaskPool::default);
    let chunk_size = candidates.len().div_ceil(pool.thread_num().max(1)).max(1);
    candidates
        .par_chunk_map(pool, chunk_size, classify)
        .into_iter()
        .flatten()
        .collect()
}


/// Build the `hash_peek` snapshot: each candidate's resident `baked_hash` (absent → not resident).
/// Lets [`classify_candidates`] do the content-hash skip without borrowing the atlas, so it can run
/// on a background task.
pub(super) fn snapshot_hash_peek(
    atlas: &SdfAtlas,
    candidates: &[(chunk::ChunkKey, atlas::BrickKey)],
) -> FxHashMap<atlas::BrickKey, u64> {
    let mut map = FxHashMap::with_capacity_and_hasher(candidates.len(), Default::default());
    for &(_ck, key) in candidates {
        if let Some(b) = atlas.bricks.get(&key) {
            map.insert(key, b.baked_hash);
        }
    }
    map
}
