//! **Camera-following residency — a true nested CLIPMAP of voxel bricks.**
//!
//! The HW-RT path streams a brick set around the camera. A brick is ALWAYS `8³` voxels, but its WORLD SPAN
//! scales with LOD ([`super::brickmap::brick_span`]`(L) = BRICK_WORLD_SIZE · 2^L`), so COARSER levels cover
//! MORE world at the same resolution — a geometry-clipmap / GigaVoxels 3D-mipmap. This replaces the old
//! dense-cube residency (every brick a fixed `0.4 m`, so coarse LOD added no coverage and view distance was
//! hard-capped) with NESTED CLIPMAP SHELLS: LOD0 fills the inner cube, each coarser level is a thin SHELL
//! that doubles the reach. Total view radius = `clip_half · BRICK_WORLD_SIZE · 2^MAX_LOD`.
//!
//! This module owns the pure, headless-testable bookkeeping — no GPU, no Bevy systems — so the residency
//! scheme is proven in isolation and the render wiring ([`super::raytrace`]) just drives it:
//!
//! * [`brick_lod`] / [`desired_clipmap`] — given the camera world position, which `(coord, lod)` bricks
//!   should be resident: each level `L` fills a box of half-extent `clip_half` around the camera on grid `L`,
//!   snapped to the 2×-coarser grid, MINUS the finer level's footprint — so the levels tile EXACTLY (no
//!   overlap, no gap; the union telescopes to the outermost box). Each level is a bounded rectangular shell.
//! * [`ResidencyManager`] — the live set of resident bricks + a bounded WORK QUEUE, keyed by [`BrickKey`]
//!   `{coord, lod}` (coords now OVERLAP across LOD grids, so the lod is part of the key). Each `update`
//!   diffs the desired clipmap against the current set, ENQUEUES newly-entered / LOD-changed bricks, and
//!   DROPS exited ones; [`ResidencyManager::drain_work`] voxelizes at most `max_per_frame` per call so a big
//!   camera jump can't stall the frame. The packed list it exposes
//!   ([`ResidencyManager::resident_entries`]) feeds the SSOT [`super::gpu::pack_resident_set`].
//!
//! ## The stutter fix — incremental, O(shell) per move
//! Each level only changes when the camera crosses a LOD-`L` brick boundary (every `brick_span(L)` m), and a
//! coarse boundary is `2^L×` farther apart than LOD0's. So a small move shifts only the LOD0 shell (a thin
//! face-slab) and NOTHING coarse — the per-move enqueue/drop count is O(shell), not O(region). The
//! diff-reconcile `update` (drop-not-desired + enqueue-not-resident/lod-changed) gives this for free once
//! keyed by `(coord, lod)`.
//!
//! ## Keep-old-until-revealed
//! The manager only marks itself DIRTY (needing a re-pack + BLAS/TLAS rebuild) once a non-empty batch of
//! queued bricks has been voxelized. The previous resident set — and the TLAS the render path keeps bound —
//! stays valid until the new one is ready, so the camera never sees a hole/flash while a batch streams in.
//!
//! ## Cross-LOD seams
//! At a shell boundary a fine brick abuts a `2×`-coarser brick. We do NOT build a cross-LOD halo; the two
//! bricks are SEPARATE BLAS AABBs and the [`brick_aabb_epsilon`](super::gpu::brick_aabb_epsilon) overlap +
//! the nearest-solid-hit DDA commit the nearest surface across the LOD step. See the seam discussion in
//! [`super::gpu::pack_resident_set`].

use bevy::math::IVec3;
use rustc_hash::{FxHashMap, FxHashSet};

use super::brickmap::{Brick, MAX_LOD, brick_span};
use super::edits::{VoxelEdits, apply_edit_overlay};
use super::gpu::ResidentBrick;
use super::palette::BlockRegistry;
use super::source::{BrickClass, BrickSource, WorldgenSource};
use crate::sdf_render::worldgen::biome::BiomeLibrary;
use crate::sdf_render::worldgen::layers::height::HeightLayer;

/// A resident-brick key in the nested clipmap: the integer brick `coord` ON the LOD-`lod` grid. Coords now
/// OVERLAP across LOD grids — the same integer coord at two LODs is two DIFFERENT world bricks
/// (`world_min = coord · brick_span(lod)`) — so the `lod` MUST be part of the key. The SSOT key for the
/// resident map, the work queue, and the empty-memo.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BrickKey {
    /// Integer brick coordinate on the LOD-`lod` grid.
    pub coord: IVec3,
    /// The clipmap LOD level.
    pub lod: u32,
}

/// A purely-DEFENSIVE ceiling on the RAW [`desired_clipmap`] enumeration size (A2): a guard so a pathological
/// `clip_half_bricks` can't OOM the geometric tiling itself. This is NOT the resident cap (`max_resident_bricks`)
/// — it is the bound on the UNCLASSIFIED clip-VOLUME enumeration, deliberately ~orders of magnitude larger than
/// any sane resident set so it never binds in practice (the surface-shell cap in [`ResidencyManager::update`] is
/// the real, much tighter bound). At `clip_half = 26` the uncapped tiling is well under this.
pub const MAX_CLIP_ENUMERATION: usize = 8_000_000;

/// Tunable clipmap streaming knobs. Plain `Copy` data so it can be a Bevy resource or a test literal.
#[derive(Clone, Copy, Debug, bevy::prelude::Resource)]
pub struct StreamingConfig {
    /// The clipmap half-extent, in bricks: each nested level fills an `~(2·clip_half)³` box around the camera
    /// on ITS grid ([`level_box`]), MINUS the footprint of the finer level below it ([`level_hole`]). Because
    /// each box snaps to the 2×-coarser grid, the levels tile EXACTLY — adjacent levels abut with NO overlap
    /// and NO gap (the union telescopes to the outermost box). The total view radius is
    /// `clip_half · BRICK_WORLD_SIZE · 2^MAX_LOD`. Default 160 (D1a) ⇒ a view half-extent of
    /// `160 · 0.4 · 2^7 = 8192 m` at bounded VRAM (`MAX_LOD+1` thin rectangular shells, not a dense cube), with
    /// a LOD0 reach of `160 · 0.4 = 64 m`. This is ONE shared (UNIFORM) knob across all LODs — not a per-LOD
    /// ring. Use `>= 2` (below that the nested annuli degenerate). Push it UP to move the LOD transitions farther
    /// out (fine detail reaches farther), bounded by `max_resident_bricks`.
    pub clip_half_bricks: i32,
    /// Hard cap on resident bricks — a SAFETY bound so a mis-set `clip_half` can't blow VRAM. With the nested
    /// shells this should NOT bind (only NON-empty surface bricks are stored, a thin shell each level). If the
    /// desired set exceeds it the farthest bricks are dropped (logged). Default 400_000 — MEASURED (D1c
    /// benchmark, `examples/d1c_scaling.rs`): a cold fill at the origin-surface clipmap settles to **143_013**
    /// resident surface bricks (40.5 MB A4.4 VRAM), so 400_000 holds it with ~2.8× headroom and is NOT the
    /// binding constraint. (HISTORICAL — the D1c blocker, FIXED by D1d: the cube [`desired_clipmap`] hit
    /// [`MAX_CLIP_ENUMERATION`] at 8 M LOD0 keys before reaching the coarse shells, so the full 64 m + 8-shell
    /// reach was not streamed and a single cold `update` cost ~38 s — 9 height taps × 8 M single-threaded
    /// classify calls. **D1d — shell-first enumeration** ([`desired_clipmap_surface`] +
    /// [`BrickSource::surface_bricks_in`]) now enumerates the surface candidates DIRECTLY in `Θ(H²)`, so all
    /// coarse LODs enumerate and the cold `update` drops to milliseconds — see [`desired_clipmap_surface`].)
    pub max_resident_bricks: usize,
    /// Max bricks voxelized + enqueued→processed per `drain_work` call (per frame). Bounds the per-frame
    /// CPU cost of a big camera move; the rest carry in the queue to later frames. Default 256.
    pub max_bricks_per_frame: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            // D1a: 160 · 0.4 m = 64 m LOD0 reach; 160 · 0.4 · 2^7 = 8192 m total view (UNIFORM knob, all LODs).
            clip_half_bricks: 160,
            // MEASURED (D1c): the origin-surface cold fill settles to 143_013 resident bricks (40.5 MB VRAM),
            // so 400_000 holds it with ~2.8× headroom. The cap is NOT the bottleneck — the 38 s/`update` CPU
            // classify is (see the field doc + the GPU-voxel-worldgen pivot).
            max_resident_bricks: 400_000,
            max_bricks_per_frame: 256,
        }
    }
}

/// The brick coordinate the camera world position falls in on the LOD-`lod` grid: `floor(cam_world /
/// brick_span(lod))` per axis. DIFFERENT LODs are different coord grids, so the per-level clipmap centre
/// differs — this is the SSOT mapping camera world → the LOD-`lod` brick that contains it.
#[inline]
pub fn camera_brick_coord_lod(cam_world: [f32; 3], lod: u32) -> IVec3 {
    let span = brick_span(lod);
    IVec3::new(
        (cam_world[0] / span).floor() as i32,
        (cam_world[1] / span).floor() as i32,
        (cam_world[2] / span).floor() as i32,
    )
}

/// The LOD0 brick coordinate the camera falls in (`camera_brick_coord_lod(_, 0)`) — the SSOT "has the camera
/// crossed a brick?" key the render loop uses to decide when to re-`update`. A LOD0 crossing is the FINEST
/// boundary, so it strictly implies any coarser crossing; reconciling on it never misses a shell shift.
#[inline]
pub fn camera_brick_coord(cam_world: [f32; 3]) -> IVec3 {
    camera_brick_coord_lod(cam_world, 0)
}

/// The Chebyshev (L∞) distance in bricks between two brick coordinates. The exact tiling
/// ([`desired_clipmap`]) is AABB-based, not Chebyshev — this is only used by the residency tests to bound
/// resident bricks to their level's box.
#[cfg(test)]
#[inline]
fn cheby(a: IVec3, b: IVec3) -> i32 {
    (a.x - b.x).abs().max((a.y - b.y).abs()).max((a.z - b.z).abs())
}

/// Round an inclusive brick range `[lo, hi]` OUTWARD to align with the next-COARSER (2×) grid: `lo` DOWN to
/// the nearest even, `hi` UP to the nearest odd. The snapped range then spans WHOLE 2×-coarser bricks
/// (`[lo/2, (hi-1)/2]`) — the property that lets nested clipmap levels tile EXACTLY: a finer level's footprint
/// becomes an integer number of coarse bricks, so the coarse level can cede precisely that footprint with no
/// partial brick left covered (overlap) or uncovered (gap). The bit ops are correct for two's-complement
/// negatives: `& !1` floors to even, `| 1` ceils to odd.
#[inline]
fn snap_even_odd(lo: i32, hi: i32) -> (i32, i32) {
    (lo & !1, hi | 1)
}

/// Level `lod`'s resident AABB (INCLUSIVE, in LOD-`lod` brick coords): a cube of half-extent `half` around the
/// camera's brick on that grid, [`snap_even_odd`]-snapped per axis so the box boundary lies on the 2×-coarser
/// grid. Every level — and every level's hole — is derived from this one SSOT box, so the enumeration
/// ([`desired_clipmap`]) and the per-point query ([`brick_lod`]) can never disagree about the tiling.
#[inline]
fn level_box(cam_world: [f32; 3], lod: u32, half: i32) -> (IVec3, IVec3) {
    let c = camera_brick_coord_lod(cam_world, lod);
    let (lx, hx) = snap_even_odd(c.x - half, c.x + half);
    let (ly, hy) = snap_even_odd(c.y - half, c.y + half);
    let (lz, hz) = snap_even_odd(c.z - half, c.z + half);
    (IVec3::new(lx, ly, lz), IVec3::new(hx, hy, hz))
}

/// The INCLUSIVE AABB (on grid `lod`) that level `lod` cedes to the FINER level `lod - 1`: the finer level's
/// [`level_box`] footprint expressed on THIS grid (downsampled by 2). Because the finer box is
/// `[even, odd]`-snapped, the downsample is EXACT — `[flo/2, (fhi-1)/2]` are whole coarse bricks, and their
/// world extent equals the finer box's world extent EXACTLY (`flo·span_{lod-1} … (fhi+1)·span_{lod-1}`). The
/// coarse level is resident in `level_box \ hole`; that subtraction is where BOTH no-overlap (the finer
/// region is removed) AND no-gap (the removed region is EXACTLY what the finer level fills — the telescoping
/// property) come from. `None` for LOD0 (the finest level cedes nothing). For `half >= 2` the hole is always
/// strictly inside the box, so the annulus is a proper shell.
#[inline]
fn level_hole(cam_world: [f32; 3], lod: u32, half: i32) -> Option<(IVec3, IVec3)> {
    if lod == 0 {
        return None;
    }
    let (flo, fhi) = level_box(cam_world, lod - 1, half);
    // flo even, fhi odd ⇒ flo/2 and (fhi-1)/2 are exact (no truncation), for either sign.
    let hlo = IVec3::new(flo.x / 2, flo.y / 2, flo.z / 2);
    let hhi = IVec3::new((fhi.x - 1) / 2, (fhi.y - 1) / 2, (fhi.z - 1) / 2);
    Some((hlo, hhi))
}

/// True iff `coord` is inside the inclusive AABB `[lo, hi]`.
#[inline]
fn in_box(coord: IVec3, lo: IVec3, hi: IVec3) -> bool {
    coord.x >= lo.x
        && coord.x <= hi.x
        && coord.y >= lo.y
        && coord.y <= hi.y
        && coord.z >= lo.z
        && coord.z <= hi.z
}

/// True iff brick `coord` on grid `lod` is RESIDENT at that level: inside the level's [`level_box`] and NOT in
/// the [`level_hole`] ceded to the finer level. The single membership predicate both [`desired_clipmap`]
/// (which enumerates it) and [`brick_lod`] (which queries it per world point) share — one SSOT for the tiling.
#[inline]
fn level_resident(cam_world: [f32; 3], coord: IVec3, lod: u32, half: i32) -> bool {
    let (lo, hi) = level_box(cam_world, lod, half);
    if !in_box(coord, lo, hi) {
        return false;
    }
    match level_hole(cam_world, lod, half) {
        Some((hlo, hhi)) => !in_box(coord, hlo, hhi),
        None => true,
    }
}

/// The uncapped clipmap brick count — `Σ over levels of |level_box \ level_hole|` — the closed-form full
/// tiling size (an exact figure for tests). `level_hole ⊆ level_box` for `half >= 2`, so the subtraction is
/// exact. (A2: `desired_clipmap` is now itself uncapped, so `desired.len()` equals this; kept as the
/// independent closed-form oracle the residency / Θ-exponent tests check against.)
#[cfg(test)]
fn clipmap_uncapped_len(cam_world: [f32; 3], half: i32) -> usize {
    let vol = |lo: IVec3, hi: IVec3| {
        (hi.x - lo.x + 1).max(0) as usize
            * (hi.y - lo.y + 1).max(0) as usize
            * (hi.z - lo.z + 1).max(0) as usize
    };
    let mut n = 0usize;
    for lod in 0..=MAX_LOD {
        let (lo, hi) = level_box(cam_world, lod, half);
        let mut v = vol(lo, hi);
        if let Some((hlo, hhi)) = level_hole(cam_world, lod, half) {
            v = v.saturating_sub(vol(hlo, hhi));
        }
        n += v;
    }
    n
}

/// The clipmap LOD that COVERS a world position, given as a LOD0 brick coordinate `coord` (world centre =
/// `(coord + 0.5) · brick_span(0)`). Returns the FINEST level whose [`desired_clipmap`] region contains that
/// world position. Because the levels tile EXACTLY (no overlap, no gap — see [`level_hole`]), every world
/// point inside the outer box is covered by exactly ONE level and this returns it; a point past the outer box
/// clamps to [`MAX_LOD`]. The SSOT for "what resolution does the renderer see at this world point".
#[inline]
pub fn brick_lod(coord: IVec3, cam_world: [f32; 3], cfg: &StreamingConfig) -> u32 {
    let half = cfg.clip_half_bricks;
    let span0 = brick_span(0);
    let world = [
        (coord.x as f32 + 0.5) * span0,
        (coord.y as f32 + 0.5) * span0,
        (coord.z as f32 + 0.5) * span0,
    ];
    for lod in 0..=MAX_LOD {
        let here = camera_brick_coord_lod(world, lod);
        if level_resident(cam_world, here, lod, half) {
            return lod;
        }
    }
    MAX_LOD
}

/// The DESIRED clipmap residency: the set of `(coord, lod)` bricks that should be resident around the camera
/// at world position `cam_world`. EXACT NESTED TILING — for each level `L` in `0..=MAX_LOD`, level `L` is
/// resident in `level_box(L) \ level_hole(L)`:
/// * [`level_box`] is an `[even, odd]`-snapped cube of half-extent `clip_half` around the camera's brick on
///   grid `L`, so its boundary lies on the 2×-coarser grid;
/// * [`level_hole`] is the FINER level's box footprint on grid `L` — the region ceded to level `L-1`.
///
/// Because a level's box snaps to the coarser grid, the hole it carves equals the finer level's box world
/// EXACTLY: adjacent levels ABUT with NEITHER overlap NOR gap, and the union telescopes to the outermost box.
/// No band-aid one-ring overlap — a ray crossing a LOD boundary passes from a fine brick to the coarse brick
/// that shares its face (the [`BRICK_AABB_EPSILON`](super::gpu::brick_aabb_epsilon) + nearest-hit DDA commit
/// the surface across the seam). LOD0 has no hole (it is the solid inner box). The union reaches
/// `clip_half · BRICK_WORLD_SIZE · 2^MAX_LOD` from the camera. Returned keyed by [`BrickKey`]; iteration order
/// is not guaranteed (callers needing a stable `primitive_index` sort — the manager does).
///
/// **A2 — this is the UNCAPPED geometric tiling.** It is NO LONGER capped to `max_resident_bricks` here: the
/// `max_resident_bricks` cap is applied in [`ResidencyManager::update`] AFTER the surface [`classify`](BrickSource::classify)
/// split, so it bounds the surface SHELL (Θ(H²)) rather than the clip VOLUME (Θ(H³)) — a far buried-interior or
/// sky brick `classify` would prune anyway must NOT steal a slot from a near surface brick. A pure geometric
/// SAFETY ceiling ([`MAX_CLIP_ENUMERATION`]) still bounds the raw enumeration so a pathological `clip_half`
/// can't OOM the enumeration itself; that ceiling is NOT the resident cap (it is ~orders of magnitude larger).
pub fn desired_clipmap(cam_world: [f32; 3], cfg: &StreamingConfig) -> FxHashMap<BrickKey, ()> {
    let half = cfg.clip_half_bricks;
    let mut out: FxHashMap<BrickKey, ()> = FxHashMap::default();
    for lod in 0..=MAX_LOD {
        let (lo, hi) = level_box(cam_world, lod, half);
        let hole = level_hole(cam_world, lod, half);
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let coord = IVec3::new(x, y, z);
                    if let Some((hlo, hhi)) = hole
                        && in_box(coord, hlo, hhi)
                    {
                        continue; // ceded to the finer level — no overlap
                    }
                    out.insert(BrickKey { coord, lod }, ());
                    if out.len() > MAX_CLIP_ENUMERATION {
                        // Defensive enumeration ceiling (NOT the resident cap): a pathological `clip_half` can't
                        // OOM the raw tiling. The surface cap in `update` is the real bound; this only guards the
                        // enumeration's own memory. Bail out — the tiling is already absurdly large.
                        return out;
                    }
                }
            }
        }
    }
    out
}

/// **SHELL-FIRST desired clipmap (D1d)** — the candidate surface keys to consider for residency, enumerated
/// `Θ(H²)` DIRECTLY from the source's surface query instead of the `Θ(H³)` cube
/// ([`desired_clipmap`]). This is the SSOT enumeration the live [`ResidencyManager::update`] uses.
///
/// # Why (the D1c blocker this fixes)
/// [`desired_clipmap`] enumerates the full `level_box` CUBE per LOD. At `clip_half = 160` that is `321³ ≈
/// 33 M` keys on LOD0 — it BLOWS [`MAX_CLIP_ENUMERATION`] (8 M) and BAILS before reaching the coarse shells,
/// so the advertised 64 m + 8-shell reach was fiction; and `update` then classified all ~8 M keys
/// single-threaded at ~38 s per camera crossing. But the RESIDENT set is only the surface SHELL (`Θ(H²)`,
/// ~143 k bricks) — the work was `Θ(H³)` for an `Θ(H²)` result. Here we enumerate the surface candidates
/// directly: each level's `level_box \ level_hole` region is handed to
/// [`BrickSource::surface_bricks_in`], which yields a CONSERVATIVE SUPERSET of the bricks that
/// [`classify`](BrickSource::classify) could call `Surface` — `O(surface)` per level, so the coarse LODs now
/// enumerate (and [`MAX_CLIP_ENUMERATION`] becomes a slack `Θ(H²) ≪ 8 M` sanity bound that no longer binds).
///
/// # Identical TILING — only the ENUMERATION narrows
/// The tiling SSOT ([`level_box`]/[`level_hole`]/[`level_resident`]) is UNCHANGED. Each surface candidate is
/// then re-confirmed by [`classify`](BrickSource::classify) (pruning the superset's false positives) and
/// box-clipped against `level_box \ level_hole` (so a candidate the source over-yielded outside the shell —
/// or inside the finer level's hole — is dropped), EXACTLY as the cube path's keys were. So the resident set
/// after classify + cap is IDENTICAL to the cube path's — the D1d oracle test
/// ([`tests::shell_first_resident_set_matches_cube_oracle`]) asserts this set-equality on cliff / thin-wall /
/// LOD-seam / flat scenes. A source WITHOUT a fast surface query falls back to
/// [`BrickSource::surface_bricks_in`]'s default (the full box), so this degrades gracefully to the cube
/// behaviour for it (still correct, just not faster).
///
/// Returned keyed by [`BrickKey`] (deduped by the map); the per-level `level_hole` clip + the box clip make
/// the result the SAME `(coord, lod)` SET the cube path would, minus the buried/sky volume the source's
/// surface query never yields. The [`MAX_CLIP_ENUMERATION`] ceiling still guards the accumulated candidate
/// count (now `Θ(H²)`, so it does not bind at the shipping `clip_half`).
pub fn desired_clipmap_surface(
    cam_world: [f32; 3],
    cfg: &StreamingConfig,
    source: &dyn BrickSource,
) -> FxHashMap<BrickKey, ()> {
    let half = cfg.clip_half_bricks;
    let mut out: FxHashMap<BrickKey, ()> = FxHashMap::default();
    let mut candidates: Vec<IVec3> = Vec::new();
    for lod in 0..=MAX_LOD {
        let (lo, hi) = level_box(cam_world, lod, half);
        let hole = level_hole(cam_world, lod, half);
        // Ask the source for a conservative superset of the surface bricks in this level's box (Θ(H²)).
        candidates.clear();
        source.surface_bricks_in(lo, hi, lod, &mut candidates);
        for &coord in &candidates {
            // BOX-CLIP exactly as the cube path: a candidate the source over-yielded outside the level box —
            // or inside the finer level's hole — is dropped, so the per-level tiling is bit-identical.
            if !in_box(coord, lo, hi) {
                continue;
            }
            if let Some((hlo, hhi)) = hole
                && in_box(coord, hlo, hhi)
            {
                continue; // ceded to the finer level — no overlap
            }
            out.insert(BrickKey { coord, lod }, ());
            if out.len() > MAX_CLIP_ENUMERATION {
                // Defensive ceiling — now Θ(H²), so it does NOT bind at the shipping clip_half. A source whose
                // surface query degenerated to the full box (the default) could still hit it; bail as before.
                return out;
            }
        }
    }
    out
}

/// The approximate WORLD-metre centre distance of a clipmap brick `key` from `cam_world` — the cap-ranking key
/// shared by the surface cap (A2). Brick centre = `(coord + 0.5)·brick_span(lod)`, so a far coarse shell ranks
/// behind a near fine one only when it is genuinely farther in world metres.
#[inline]
fn brick_world_dist(key: &BrickKey, cam_world: [f32; 3]) -> f32 {
    let span = brick_span(key.lod);
    let cx = (key.coord.x as f32 + 0.5) * span - cam_world[0];
    let cy = (key.coord.y as f32 + 0.5) * span - cam_world[1];
    let cz = (key.coord.z as f32 + 0.5) * span - cam_world[2];
    (cx * cx + cy * cy + cz * cz).sqrt()
}

/// A queued unit of streaming work: voxelize the brick at clipmap key `key` (`(coord, lod)`). The LOD is part
/// of the key, so a brick that changes LOD (a shell shift) is a DIFFERENT key — enqueued + voxelized fresh at
/// the new LOD's coarse spacing, never silently re-tagged (the in-place mip means the voxel data differs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WorkItem {
    key: BrickKey,
}

/// The live clipmap residency state + bounded work queue. Holds the voxelized [`Brick`]s currently resident
/// (only NON-empty bricks are stored — empty/all-air bricks are skipped, the sparsity invariant), keyed by
/// [`BrickKey`] `(coord, lod)`, plus a FIFO queue of bricks awaiting voxelization. `update` recomputes the
/// desired clipmap and reconciles; `drain_work` does the bounded voxelization.
///
/// Robust-by-construction: the resident map is the single source of what's live; `dirty` is set only when a
/// drained batch actually changes it, so the render path re-packs exactly when (and only when) the GPU set
/// must change — keep-old-until-revealed falls out for free. Keying by `(coord, lod)` makes a LOD change a
/// DIFFERENT key (a fresh voxelize at the new coarse spacing), so the old "retag the same brick" confusion
/// is structurally impossible.
#[derive(Default)]
pub struct ResidencyManager {
    /// Resident, voxelized bricks: clipmap key → its `8³` brick (voxelized at the key's LOD). Empty bricks
    /// are never inserted.
    resident: FxHashMap<BrickKey, Brick>,
    /// Keys awaiting voxelization (enqueued by `update`, processed by `drain_work`). A set membership guard
    /// (`queued`) prevents a key from being enqueued twice while it waits.
    queue: std::collections::VecDeque<WorkItem>,
    queued: FxHashSet<BrickKey>,
    /// KNOWN-EMPTY (all-air) keys in the current clipmap: bricks that voxelized to empty (above the surface)
    /// are NEVER resident (sparsity), so without this memo `update` would find them absent and re-enqueue +
    /// re-voxelize them on EVERY camera move — and most of the desired clipmap (~2/3) is empty sky/air, the
    /// dominant streaming churn otherwise. Memoize them so each empty key is voxelized ONCE; bounded to the
    /// clipmap (`update` prunes keys that leave). Emptiness is per-`(coord, lod)` (a coarse brick samples the
    /// surface at coarse spacing, so it can differ from a finer one — hence the LOD is in the key).
    empty: FxHashSet<BrickKey>,
    /// CLASSIFY-PRUNED keys in the current clipmap: bricks the [`BrickSource::classify`] filter marked
    /// non-[`BrickClass::Surface`] (deep-buried interior or high sky), so they were NOT enqueued (the
    /// SURFACE-FOLLOWING RESIDENCY bound — only the surface shell is voxelized/kept resident). Memoized for the
    /// SAME reason as `empty`: without it `update` would re-classify the whole clipmap volume (most of it
    /// non-surface) on EVERY camera move. Bounded to the clipmap (`update` prunes keys that leave); a key
    /// re-evaluates when it re-enters. An EDIT clears keys here ([`requeue_keys`](Self::requeue_keys)) so a dig
    /// into a geometrically-Interior brick can still reveal it (the classify is edit-unaware).
    pruned: FxHashSet<BrickKey>,
    /// True iff the resident set CHANGED since the last `take_dirty` — the render path re-packs + rebuilds
    /// the BLAS/TLAS only then (otherwise it keeps the old, still-valid GPU scene).
    dirty: bool,
    /// Total bricks dropped by the resident cap over the manager's life (for logging).
    pub capped_total: usize,
}

impl ResidencyManager {
    /// A fresh, empty manager (no resident bricks, empty queue).
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of resident (voxelized, non-empty) bricks.
    #[inline]
    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    /// Number of keys PRUNED by the surface-following classify (deep-buried interior + high sky skipped at
    /// enqueue) and memoized in the current clipmap. The cull effectiveness — grows with `clip_half`. For the
    /// perf/stats panel.
    #[inline]
    pub fn pruned_count(&self) -> usize {
        self.pruned.len()
    }

    /// Resident brick count per LOD level (index = lod, length `MAX_LOD + 1`) — the clipmap's LOD distribution
    /// for the perf/stats panel. `O(resident)`; call only when displaying (e.g. the panel is open).
    pub fn resident_lod_counts(&self) -> Vec<usize> {
        let mut counts = vec![0usize; (MAX_LOD + 1) as usize];
        for k in self.resident.keys() {
            if let Some(c) = counts.get_mut(k.lod as usize) {
                *c += 1;
            }
        }
        counts
    }

    /// True iff `key` is currently resident (a non-empty brick is stored for it).
    #[inline]
    pub fn is_resident(&self, key: &BrickKey) -> bool {
        self.resident.contains_key(key)
    }

    /// Number of bricks waiting in the work queue.
    #[inline]
    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    /// True iff the resident set has changed and not yet been consumed (a re-pack is due).
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Reconcile the resident set toward the desired CLIPMAP around the camera at world position `cam_world`:
    /// * DROP every resident brick no longer in the desired clipmap (a shell shifted / the camera moved) —
    ///   marks dirty.
    /// * ENQUEUE every desired `(coord, lod)` that is NOT resident AND classifies [`BrickClass::Surface`] —
    ///   voxelized later by [`drain_work`]. Non-surface keys (deep-buried Interior / high sky Air) are PRUNED:
    ///   skipped + memoized in `pruned` (the SURFACE-FOLLOWING RESIDENCY bound — only the surface SHEET is
    ///   voxelized + kept resident, so residency + the per-frame voxelize are O(clip_half²), not O(clip_half³)).
    ///
    /// `source` supplies the cheap [`BrickSource::classify`] predicate AND (D1d) the `Θ(H²)`
    /// [`BrickSource::surface_bricks_in`] enumeration: the desired set is built by
    /// [`desired_clipmap_surface`] (surface candidates enumerated DIRECTLY), then `classify` re-confirms each
    /// candidate and prunes the superset's false positives. The classify is still an ADDITIVE FILTER on top of the
    /// exact-tiling residency ([`level_resident`] / [`level_box`] / [`level_hole`] — UNCHANGED, the
    /// no-overlap/no-gap tiling holds); it only removes provably-unhittable bricks. A finite/static source
    /// defaults to the full-box `surface_bricks_in` + `Surface` classify (never prune), so its behaviour is
    /// unchanged (the wholly-outside reject + the empty-memo already bound it).
    ///
    /// A LOD change is just a different [`BrickKey`] entering + the old one leaving (different coord grids),
    /// so there is NO retag path — each brick is voxelized at exactly one LOD. Does NOT itself voxelize, so a
    /// huge camera jump only enqueues here (cheap). The per-move enqueue/drop is O(shell): only the bricks
    /// whose key entered/left change, and a small move shifts only the LOD0 shell (coarse shells move
    /// `2^L×` less often). Returns the number of bricks dropped (so the caller can log churn).
    pub fn update(&mut self, cam_world: [f32; 3], cfg: &StreamingConfig, source: &dyn BrickSource) -> usize {
        // SHELL-FIRST (D1d): enumerate the desired set from the source's Θ(H²) surface query, NOT the Θ(H³)
        // cube. `desired` now holds the surface CANDIDATES (a conservative superset of `Surface` bricks, the
        // classify below re-confirms + prunes). Every resident brick was classified `Surface` to be
        // enqueued, so it is in this superset — the drop step below is correct (it never spuriously drops a
        // resident surface brick). The tiling is unchanged; only the enumeration narrowed from cube to shell,
        // restoring the coarse LODs (the cube path bailed at MAX_CLIP_ENUMERATION before reaching them) and
        // killing the 38 s/crossing classify (was 9 taps × 8 M keys; now 9 taps × Θ(H²) candidates).
        let desired = desired_clipmap_surface(cam_world, cfg, source);

        // Drop resident bricks that left the clipmap.
        let mut dropped = 0usize;
        let to_drop: Vec<BrickKey> =
            self.resident.keys().filter(|k| !desired.contains_key(*k)).copied().collect();
        for k in to_drop {
            self.resident.remove(&k);
            dropped += 1;
            self.dirty = true; // the GPU set shrank → must re-pack
        }
        // Prune the empty + classify-pruned memos to the current clipmap (bounds them as the camera roams; a
        // key that re-enters is cheaply re-evaluated + re-memoized). Deterministic terrain ⇒ an empty/pruned
        // key is stably so until an EDIT (which clears it via `requeue_keys`).
        self.empty.retain(|k| desired.contains_key(k));
        self.pruned.retain(|k| desired.contains_key(k));

        // CLASSIFY each desired brick that is NOT already resident (and not known-empty / known-pruned / already
        // queued). The classify FILTER prunes non-surface keys: a deep-buried Interior brick or a high-sky Air
        // brick is never voxelized nor kept resident (it can't be a primary-ray hit), so the resident set + the
        // per-frame drain track the SURFACE SHEET, not the clipmap volume. A `Surface` brick (the default, and
        // any straddle/uncertain case) is a candidate exactly as before — the prune is purely subtractive, so
        // the exact-tiling residency stays valid.
        //
        // A2 — CAP AFTER CLASSIFY: we collect the new SURFACE candidates here and apply `max_resident_bricks`
        // to the surface SHELL below (NOT the clip VOLUME in `desired_clipmap`). So a far buried-interior / sky
        // brick — which `classify` prunes anyway — can never steal a slot from a near surface brick; the cap
        // bounds Θ(H²) surface, not Θ(H³) volume.
        let mut surface_candidates: Vec<BrickKey> = Vec::new();
        for key in desired.keys() {
            if self.resident.contains_key(key)
                || self.queued.contains(key)
                || self.empty.contains(key)
                || self.pruned.contains(key)
            {
                continue;
            }
            match source.classify(key.coord, key.lod) {
                BrickClass::Surface => surface_candidates.push(*key),
                BrickClass::Air | BrickClass::Interior => {
                    // Provably unhittable — prune (memoized so we don't re-classify it every move).
                    self.pruned.insert(*key);
                }
            }
        }

        // A2 surface-shell cap: the resident set + the in-flight queue + the new surface candidates must not
        // exceed `max_resident_bricks`. When they would, drop the FARTHEST surface candidates (world-metre rank)
        // so the nearest surface shell is kept — the cap binds the surface SHEET, never the buried volume (those
        // were already pruned above and never count against it). With the nested shells this should rarely bind.
        let budget = cfg.max_resident_bricks;
        let already = self.resident.len() + self.queued.len();
        let room = budget.saturating_sub(already);
        if surface_candidates.len() > room {
            // Keep the nearest `room`; drop the rest (farthest first), deterministic tiebreak. Rank by world
            // distance so a far coarse shell drops before a near fine one only if it is genuinely farther.
            surface_candidates.sort_by(|a, b| {
                brick_world_dist(a, cam_world)
                    .partial_cmp(&brick_world_dist(b, cam_world))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then((a.lod, a.coord.z, a.coord.y, a.coord.x).cmp(&(b.lod, b.coord.z, b.coord.y, b.coord.x)))
            });
            self.capped_total += surface_candidates.len() - room;
            surface_candidates.truncate(room);
        }

        // Enqueue the (surviving) surface candidates for voxelization by `drain_work`.
        for key in surface_candidates {
            self.queue.push_back(WorkItem { key });
            self.queued.insert(key);
        }
        dropped
    }

    /// Process up to `cfg.max_bricks_per_frame` queued bricks from the WORLDGEN surface — the original
    /// worldgen drain (signature unchanged, so the streaming + perf harness tests are bit-identical). A thin
    /// wrapper over the source-generic [`drain_work_from`](Self::drain_work_from): it builds a
    /// [`WorldgenSource`] over `(layer, lib, seed)` and drains with NO edit overlay
    /// ([`VoxelEdits::is_empty`]), so the resident set is exactly what the direct `voxelize_brick` drain
    /// produced before the source abstraction.
    pub fn drain_work(
        &mut self,
        cfg: &StreamingConfig,
        layer: &HeightLayer,
        lib: &BiomeLibrary,
        registry: &BlockRegistry,
        seed: u64,
    ) -> usize {
        let source = WorldgenSource::new(layer, lib, seed);
        self.drain_work_from(cfg, &source, registry, &VoxelEdits::new())
    }

    /// Process up to `cfg.max_bricks_per_frame` queued bricks from ANY [`BrickSource`] (worldgen or a static
    /// `.vox`): SOURCE each at ITS key's LOD (the in-place mip — coarse keys sample at coarse spacing), apply
    /// the shared [`VoxelEdits`] overlay (so build/destroy editing works UNIFORMLY for every scene), store
    /// NON-empty results as resident, and drop empty ones (sparsity). Marks the set dirty iff at least one
    /// brick was actually added/removed — so a batch that produced only empty bricks does NOT trigger a
    /// needless re-pack, and the old GPU scene stays valid until a REVEALING batch lands
    /// (keep-old-until-revealed). Returns the number of bricks sourced this call.
    ///
    /// Bounded: never does more than `max_bricks_per_frame` sourcings; leftover queue items carry to the next
    /// call. Logs when it caps (leaves work pending). DETERMINISTIC: the source is [`Sync`] + pure and the
    /// per-brick overlay is pure, so the parallel drain yields a brick identical regardless of thread, applied
    /// in a fixed order — the resident set is bit-identical to a serial loop.
    pub fn drain_work_from(
        &mut self,
        cfg: &StreamingConfig,
        source: &dyn BrickSource,
        registry: &BlockRegistry,
        edits: &VoxelEdits,
    ) -> usize {
        use bevy::tasks::{ComputeTaskPool, ParallelSlice};
        let budget = cfg.max_bricks_per_frame;
        // Pop the per-frame batch first (serial, cheap): up to `budget` queued keys.
        let mut keys: Vec<BrickKey> = Vec::with_capacity(budget.min(self.queue.len()));
        while keys.len() < budget {
            let Some(item) = self.queue.pop_front() else { break };
            self.queued.remove(&item.key);
            keys.push(item.key);
        }
        let done = keys.len();
        if done > 0 {
            // Source the batch IN PARALLEL on the compute task pool. The source's `brick` is a pure function of
            // `(coord, lod, &registry)` (all shared + Sync), and the per-brick edit overlay is pure, so this is
            // determinism-preserving: each key yields an identical brick regardless of thread, and we apply the
            // results in a fixed order — the resident set is bit-identical to a serial loop. Chunked (~one
            // chunk per worker) so we spawn a handful of tasks, not one per brick.
            //
            // `get_or_init` (not `get`): the running app already initialized the ComputeTaskPool, but the
            // headless tests + perf harness call drain_work directly with no Bevy app — there `get()` panics,
            // so init a default pool on first use. (Same pool the live app uses when one exists.)
            //
            // The edit overlay is SKIPPED entirely when there are no edits — so a no-edit drain (the common
            // case, and EVERY worldgen-harness test) is the literal `source.brick(...)` path, bit-identical to
            // before the abstraction. When edits exist, each base brick is overlaid per-voxel via the shared
            // `apply_edit_overlay` SSOT (the same rule the static-scene + pick paths use).
            let has_edits = !edits.is_empty();
            let pool = ComputeTaskPool::get_or_init(bevy::tasks::TaskPool::default);
            let chunk = done.div_ceil(pool.thread_num().max(1)).max(1);
            let results: Vec<(BrickKey, Brick)> = keys
                .par_chunk_map(pool, chunk, |_, ks| {
                    ks.iter()
                        .map(|&k| {
                            let base = source.brick(k.coord, k.lod, registry);
                            // The overlay is keyed by world VOXEL coord on the LOD0 grid; it only affects LOD0
                            // bricks (a coarse brick's world-voxel footprint doesn't align with the override
                            // grid). Applying it unconditionally is still correct (a coarse base has no
                            // matching override key ⇒ unchanged), but skip non-LOD0 to keep coarse drains cheap.
                            let brick = if has_edits && k.lod == 0 {
                                apply_edit_overlay(k.coord, &base, edits)
                            } else {
                                base
                            };
                            (k, brick)
                        })
                        .collect::<Vec<_>>()
                })
                .into_iter()
                .flatten()
                .collect();
            // Apply serially (HashMap mutation): non-empty bricks become resident; an all-air brick is dropped.
            for (key, brick) in results {
                if brick.is_empty() {
                    // All-air → never resident; MEMOIZE so future moves don't re-source it (the churn fix +
                    // the static-scene clipmap BOUND: bricks outside the loaded map source empty once).
                    self.empty.insert(key);
                    if self.resident.remove(&key).is_some() {
                        self.dirty = true;
                    }
                } else {
                    self.empty.remove(&key); // defensive: a now-solid key must not stay memoized empty
                    self.resident.insert(key, brick);
                    self.dirty = true;
                }
            }
        }
        if !self.queue.is_empty() {
            bevy::log::debug!(
                "voxel streaming: capped at {budget} bricks/frame, {} still pending",
                self.queue.len()
            );
        }
        done
    }

    /// Force a RE-SOURCE of specific keys on the NEXT [`drain_work_from`](Self::drain_work_from): clear them
    /// from the empty-memo (so a now-solid edit isn't skipped as known-air) AND the classify-pruned memo (so a
    /// DIG into a geometrically-buried brick isn't skipped as known-Interior), then re-enqueue them. It does
    /// NOT drop the resident entry — the OLD voxelized brick stays resident + bound until the re-source
    /// overwrites it next drain (keep-old-until-revealed: the camera never sees a hole/flash while the edited
    /// brick re-sources).
    ///
    /// This is the DIG-REVEAL path, and it deliberately BYPASSES the classify FILTER that [`update`](Self::update)
    /// applies: the geometric [`BrickSource::classify`] is edit-UNAWARE (it samples the procedural surface, not
    /// the edit overlay), so a brick dug below the surface is still classified `Interior` and would never be
    /// enqueued by `update`. By FORCE-enqueueing the edit's owner + halo neighbours here (regardless of class)
    /// and clearing them from `pruned`, the dug shell — the newly-exposed brick AND its now-exposed neighbours —
    /// becomes resident, so the dig reveals SOLID interior (not a void). Digging deeper re-fires this per shell,
    /// progressively revealing the next layer.
    ///
    /// Used for UNIFORM editing — an edit names the affected LOD0 bricks (owner + halo neighbours) and this
    /// re-queues exactly those, so the edit re-sources + re-packs LOCALLY (it ADAPTS, never full-clears — the
    /// resident set, the GI reservoirs, and the world cache all stay; see [[feedback-gi-adapt-not-reset]]). Keys
    /// not currently resident are simply enqueued so a place into empty space still appears. A key already
    /// queued is left as-is (the membership guard avoids a double-enqueue). No-op for an empty set.
    pub fn requeue_keys(&mut self, keys: impl IntoIterator<Item = BrickKey>) {
        for key in keys {
            self.empty.remove(&key);
            self.pruned.remove(&key); // dig-reveal: a buried brick force-enqueued past the classify prune
            if !self.queued.contains(&key) {
                self.queue.push_back(WorkItem { key });
                self.queued.insert(key);
            }
        }
    }

    /// Take the dirty flag, clearing it. `true` ⇒ the resident set changed and the render path should
    /// re-pack + rebuild the BLAS/TLAS this frame; `false` ⇒ nothing changed, keep the old GPU scene.
    #[inline]
    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    /// The resident bricks as [`ResidentBrick`] entries in a DETERMINISTIC order (sorted by `(lod, z, y, x)`),
    /// ready for [`super::gpu::pack_resident_set`]. The stable order keeps each brick's `primitive_index`
    /// reproducible (the test oracle relies on it). Borrows `self`, so the returned entries live as long as
    /// the manager isn't mutated.
    pub fn resident_entries(&self) -> Vec<ResidentBrick<'_>> {
        let mut keys: Vec<BrickKey> = self.resident.keys().copied().collect();
        keys.sort_by_key(|k| (k.lod, k.coord.z, k.coord.y, k.coord.x));
        keys.into_iter()
            .map(|key| {
                let brick = self.resident.get(&key).expect("key came from keys");
                ResidentBrick { coord: key.coord, brick, lod: key.lod }
            })
            .collect()
    }
}

/// The world-metre AABB half-extent the resident CLIPMAP covers around the camera (for logging / framing):
/// the OUTERMOST shell reaches `clip_half · brick_span(MAX_LOD) = clip_half · BRICK_WORLD_SIZE · 2^MAX_LOD`.
/// This is the clipmap view radius — `2^MAX_LOD×` the old dense-cube reach at the same `clip_half`.
pub fn region_half_extent_m(cfg: &StreamingConfig) -> f32 {
    cfg.clip_half_bricks as f32 * brick_span(MAX_LOD)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_render::worldgen::biome::{
        BiomeDef, BiomeId, BiomeLibrary, StrataLayer, TerrainMatId, TerrainSurfaceMaterial,
    };
    use crate::sdf_render::worldgen::coord::LayerId;
    use crate::sdf_render::worldgen::layers::erosion::ErosionParams;
    use crate::sdf_render::worldgen::layers::height::HeightParams;

    const SEED: u64 = 0xA15E_C0DE_2026;

    fn test_layer() -> HeightLayer {
        HeightLayer::new(LayerId(0), HeightParams::default(), ErosionParams::default())
    }

    fn test_library() -> BiomeLibrary {
        let mat = |name: &str, c: [f32; 4]| TerrainSurfaceMaterial {
            name: name.into(),
            base_color: c,
            roughness: 0.9,
            blend: 0.0,
            texture: None,
            tiling: 4.0,
            ..Default::default()
        };
        let materials = vec![mat("surface", [0.1, 0.5, 0.1, 1.0]), mat("stone", [0.5, 0.5, 0.5, 1.0])];
        let column = |_| BiomeDef {
            name: "b".into(),
            surface: TerrainMatId(0),
            surface_rules: vec![],
            strata: vec![StrataLayer { material: TerrainMatId(0), thickness: 1000.0 }],
            bedrock: TerrainMatId(1),
        };
        let biomes = BiomeId::ALL.iter().map(column).collect();
        BiomeLibrary { materials, biomes }
    }

    fn registry() -> BlockRegistry {
        BlockRegistry::from_biome_library(&test_library())
    }

    /// A LOD-`L` brick's containing camera coord scales with `brick_span(L)`: a fixed world position maps to
    /// a coarser brick coord at coarser LODs (the per-level clipmap centres differ). Derived from the consts
    /// (not hardcoded) so it tracks the VOXEL_SIZE flip.
    #[test]
    fn camera_brick_coord_scales_with_lod() {
        // World position 5 m: LOD0 (span 0.4) → floor(5/0.4)=12; LOD1 (0.8) → 6; LOD2 (1.6) → 3.
        let w = [5.0, 5.0, 5.0];
        let expect = |lod: u32| IVec3::splat((5.0 / brick_span(lod)).floor() as i32);
        assert_eq!(camera_brick_coord_lod(w, 0), expect(0));
        assert_eq!(camera_brick_coord_lod(w, 1), expect(1));
        assert_eq!(camera_brick_coord_lod(w, 2), expect(2));
        // camera_brick_coord is the LOD0 alias.
        assert_eq!(camera_brick_coord(w), camera_brick_coord_lod(w, 0));
    }

    /// EXACT nested tiling (no overlap, no gap). Closed-form proof on the level boxes: each level's box is
    /// `[even, odd]`-snapped, its hole sits strictly inside its box (proper annulus), and the hole's WORLD
    /// extent equals the finer level's box WORLD extent EXACTLY (the telescoping property). Together these
    /// guarantee adjacent levels abut with neither overlap nor gap — proven for a spread of camera sub-cell
    /// offsets, not just the origin. Replaces the old one-ring-overlap scheme (the user requires NO overlap).
    #[test]
    fn clipmap_tiles_exactly_no_overlap_no_gap() {
        let half = 8;
        let cfg = StreamingConfig { clip_half_bricks: half, max_resident_bricks: usize::MAX, ..Default::default() };
        let cams = [
            [0.5_f32, 0.5, 0.5],
            [7.4, 14.5, 17.7],
            [-7.05, -13.97, 6.04],
            [3.3, 0.1, -9.9],
            [101.2, -50.7, 33.3],
        ];
        for cam in cams {
            // Every level present in the enumerated set.
            let d = desired_clipmap(cam, &cfg);
            for lod in 0..=MAX_LOD {
                assert!(d.keys().any(|k| k.lod == lod), "level {lod} present, cam={cam:?}");
            }
            for lod in 0..=MAX_LOD {
                let (lo, hi) = level_box(cam, lod, half);
                let (loa, hia) = (lo.to_array(), hi.to_array());
                for a in 0..3 {
                    assert_eq!(loa[a] & 1, 0, "box lo even (lod {lod} axis {a}, cam {cam:?})");
                    assert_eq!(hia[a] & 1, 1, "box hi odd (lod {lod} axis {a}, cam {cam:?})");
                }
                match level_hole(cam, lod, half) {
                    None => assert_eq!(lod, 0, "only LOD0 has no hole"),
                    Some((hlo, hhi)) => {
                        // Proper annulus: the hole is strictly inside the box.
                        assert!(
                            in_box(hlo, lo, hi) && in_box(hhi, lo, hi),
                            "hole ⊆ box (lod {lod}, cam {cam:?})"
                        );
                        // Telescoping: the hole's WORLD extent == the finer level's box WORLD extent exactly.
                        let span = brick_span(lod);
                        let span_f = brick_span(lod - 1);
                        let (flo, fhi) = level_box(cam, lod - 1, half);
                        let (hloa, hhia) = (hlo.to_array(), hhi.to_array());
                        let (floa, fhia) = (flo.to_array(), fhi.to_array());
                        for a in 0..3 {
                            assert!(
                                (hloa[a] as f32 * span - floa[a] as f32 * span_f).abs() < 1e-3,
                                "hole min == finer box min in world (lod {lod} axis {a})"
                            );
                            assert!(
                                ((hhia[a] + 1) as f32 * span - (fhia[a] + 1) as f32 * span_f).abs() < 1e-3,
                                "hole max == finer box max in world (lod {lod} axis {a})"
                            );
                        }
                    }
                }
            }
        }
    }

    /// `brick_lod(lod0_coord, cam_world, cfg)` reports the FINEST level whose tiled region covers that world
    /// position: inside the LOD0 box ⇒ LOD0; past it ⇒ the coarser level whose annulus it lands in.
    #[test]
    fn brick_lod_reports_covering_level() {
        let cfg = StreamingConfig { clip_half_bricks: 8, ..Default::default() };
        let cam = [0.5_f32, 0.5, 0.5];
        // The camera's own LOD0 brick is covered by LOD0 (LOD0 box at this cam is [-8, 9] per axis).
        assert_eq!(brick_lod(camera_brick_coord_lod(cam, 0), cam, &cfg), 0);
        // A LOD0 brick still inside the LOD0 box is LOD0.
        assert_eq!(brick_lod(IVec3::new(7, 0, 0), cam, &cfg), 0);
        // Just past the LOD0 box (x=12 ∉ [-8, 9]): its world centre (~5 m at 0.4 m span) lands in the LOD1 annulus.
        assert_eq!(brick_lod(IVec3::new(12, 0, 0), cam, &cfg), 1);
        // Far out (≈12 m) is a coarser level still.
        assert!(brick_lod(IVec3::new(30, 0, 0), cam, &cfg) >= 2);
        // Consistency: the level brick_lod reports is exactly the one that holds that point in desired_clipmap.
        let cfg_big = StreamingConfig { max_resident_bricks: usize::MAX, ..cfg };
        let d = desired_clipmap(cam, &cfg_big);
        let span0 = brick_span(0);
        for cx in [0, 5, 11, 13, 25, 60, 150] {
            let coord = IVec3::new(cx, 0, 0);
            let lod = brick_lod(coord, cam, &cfg_big);
            let world = [(cx as f32 + 0.5) * span0, 0.5 * span0, 0.5 * span0];
            let here = camera_brick_coord_lod(world, lod);
            assert!(
                lod == MAX_LOD || d.contains_key(&BrickKey { coord: here, lod }),
                "brick_lod({cx}) = {lod} must be the resident level for that point"
            );
        }
    }

    /// A test [`BrickSource`] that classifies EVERY brick as [`BrickClass::Surface`] (the default), so
    /// `ResidencyManager::update`'s classify split keeps all desired keys as surface candidates — letting the
    /// A2 surface cap be exercised directly (every desired brick competes for a slot). `brick` is never called
    /// by `update`, so it returns a trivial empty brick.
    struct AllSurfaceSource;
    impl BrickSource for AllSurfaceSource {
        fn brick(&self, _coord: IVec3, _lod: u32, _registry: &BlockRegistry) -> Brick {
            Brick::uniform(super::super::palette::BlockId::AIR)
        }
        // classify uses the trait default (always Surface) — exactly what we want.
    }

    /// A2 — the cap is applied AFTER the classify split, so it bounds the surface SHELL: a cold `update` at a
    /// small `max_resident_bricks` enqueues at most the cap, the resident+pending set never exceeds it, and the
    /// KEPT surface candidates are the NEAREST ones (a far surface brick is dropped before a near one). The
    /// `desired_clipmap` itself is now UNCAPPED — the cap lives in `update`.
    #[test]
    fn surface_cap_bounds_shell_keeping_nearest() {
        let cap = 50usize;
        let cfg = StreamingConfig { clip_half_bricks: 8, max_resident_bricks: cap, ..Default::default() };
        let cam = [0.5_f32, 0.5, 0.5];

        // desired_clipmap is no longer capped — it returns the full geometric tiling (≫ cap here).
        let big = StreamingConfig { max_resident_bricks: usize::MAX, ..cfg };
        let d = desired_clipmap(cam, &big);
        assert!(d.len() > cap, "the uncapped tiling is far larger than the cap");

        let src = AllSurfaceSource;
        let mut mgr = ResidencyManager::new();
        mgr.update(cam, &cfg, &src);
        // The cap binds the SURFACE shell: resident (0 yet) + pending must not exceed the cap.
        assert!(mgr.resident_count() + mgr.pending() <= cap, "the surface cap bounds resident+pending to the cap");
        assert_eq!(mgr.pending(), cap, "an all-surface cold fill enqueues exactly the cap (nearest kept)");
        assert!(mgr.capped_total > 0, "the cap dropped the farther surface candidates");

        // The KEPT (queued) candidates are the NEAREST surface bricks: the farthest queued brick is no farther
        // than the nearest DROPPED desired brick. Reconstruct the world-distance ranking over the full tiling.
        let mut all: Vec<BrickKey> = d.keys().copied().collect();
        all.sort_by(|a, b| {
            brick_world_dist(a, cam)
                .partial_cmp(&brick_world_dist(b, cam))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then((a.lod, a.coord.z, a.coord.y, a.coord.x).cmp(&(b.lod, b.coord.z, b.coord.y, b.coord.x)))
        });
        let kept: FxHashSet<BrickKey> = all.iter().take(cap).copied().collect();
        for item in &mgr.queue {
            assert!(kept.contains(&item.key), "every queued brick is among the nearest `cap` desired bricks");
        }
        // The camera's own LOD0 brick (nearest) is always kept.
        let cam0 = camera_brick_coord_lod(cam, 0);
        assert!(mgr.queued.contains(&BrickKey { coord: cam0, lod: 0 }), "the camera's LOD0 brick is always kept");
    }

    /// A2 — a deep-buried INTERIOR / high-sky AIR brick (which `classify` prunes) NEVER counts against the cap:
    /// the cap bounds only the surface candidates. With a source that prunes everything to Interior, even a
    /// tiny cap enqueues NOTHING and the resident set stays empty — the volume can't steal slots from a shell.
    #[test]
    fn surface_cap_ignores_pruned_interior() {
        struct AllInteriorSource;
        impl BrickSource for AllInteriorSource {
            fn brick(&self, _c: IVec3, _l: u32, _r: &BlockRegistry) -> Brick {
                Brick::uniform(super::super::palette::BlockId::AIR)
            }
            fn classify(&self, _c: IVec3, _l: u32) -> BrickClass {
                BrickClass::Interior
            }
        }
        let cfg = StreamingConfig { clip_half_bricks: 8, max_resident_bricks: 10, ..Default::default() };
        let cam = [0.5_f32, 0.5, 0.5];
        let mut mgr = ResidencyManager::new();
        mgr.update(cam, &cfg, &AllInteriorSource);
        assert_eq!(mgr.pending(), 0, "pruned interior bricks never become surface candidates");
        assert_eq!(mgr.capped_total, 0, "the cap never binds — nothing surface to drop");
    }

    /// Residency reconciliation: a simulated camera move enters new bricks (enqueued, then voxelized into
    /// resident) and drops exited ones; empty (sky) bricks are skipped; the keep-old invariant holds (the
    /// set isn't dirty until a revealing batch lands). Resident bricks always lie within the clipmap.
    #[test]
    fn residency_updates_as_camera_moves() {
        let layer = test_layer();
        let lib = test_library();
        let reg = registry();
        // Place the camera AT the surface so the inner LOD0 cube straddles terrain (non-empty bricks).
        let surf = layer.sample_world(0.0, 0.0, SEED).height;
        let cfg = StreamingConfig { clip_half_bricks: 2, max_resident_bricks: 100_000, max_bricks_per_frame: 100_000 };

        let src = WorldgenSource::new(&layer, &lib, SEED);
        let mut mgr = ResidencyManager::new();
        let cam0 = [0.0_f32, surf, 0.0];
        mgr.update(cam0, &cfg, &src);
        assert!(mgr.pending() > 0, "entering a fresh clipmap enqueues work");
        assert!(!mgr.is_dirty(), "no bricks voxelized yet → not dirty (keep-old)");

        mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
        assert!(mgr.is_dirty(), "voxelizing real terrain bricks reveals new geometry → dirty");
        assert!(mgr.take_dirty());
        assert!(mgr.resident_count() > 0, "some non-empty bricks resident");

        // Move the camera +5 m in X (crosses a few LOD0 bricks). New bricks enter, far ones drop.
        let cam1 = [5.0_f32, surf, 0.0];
        let dropped = mgr.update(cam1, &cfg, &src);
        assert!(dropped > 0, "moving away drops the bricks left behind");
        mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
        // Every resident brick lies within its level's clipmap shell around the new camera.
        // The snapped box has half-extent up to `half + 1` (snap_even_odd can extend one side by one brick).
        let half = cfg.clip_half_bricks;
        for e in mgr.resident_entries() {
            let cam_l = camera_brick_coord_lod(cam1, e.lod);
            assert!(cheby(e.coord, cam_l) <= half + 1, "resident bricks stay in the clipmap");
        }
    }

    /// The per-frame cap bounds work: a large fresh clipmap drains at most `max_bricks_per_frame` per call,
    /// carrying the rest in the queue across calls until empty.
    #[test]
    fn carry_queue_caps_per_frame_work() {
        let layer = test_layer();
        let lib = test_library();
        let reg = registry();
        let surf = layer.sample_world(0.0, 0.0, SEED).height;
        let cfg = StreamingConfig { clip_half_bricks: 3, max_resident_bricks: 1_000_000, max_bricks_per_frame: 50 };

        let src = WorldgenSource::new(&layer, &lib, SEED);
        let mut mgr = ResidencyManager::new();
        let cam = [0.0_f32, surf, 0.0];
        mgr.update(cam, &cfg, &src);
        let total = mgr.pending();
        assert!(total > 50, "the clipmap enqueues more than one frame's budget");

        let mut drains = 0;
        let mut voxelized = 0usize;
        while mgr.pending() > 0 {
            let n = mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
            assert!(n <= 50, "never exceeds the per-frame cap");
            voxelized += n;
            drains += 1;
            assert!(drains <= total / 50 + 5, "must terminate");
        }
        assert_eq!(voxelized, total, "every enqueued brick is eventually voxelized");
        assert_eq!(drains, total.div_ceil(50), "carries the rest across frames");
    }

    /// A LOD change is a DIFFERENT key: when the camera moves so a world region's covering level shifts (the
    /// shell boundary crosses it), the old `(coord, lod)` key leaves the clipmap and a new `(coord', lod')`
    /// key enters — voxelized fresh at the new LOD's coarse spacing, never silently re-tagged. We verify a
    /// move re-keys: a coord that was LOD0-resident is no longer LOD0-resident once it falls into the LOD1
    /// shell, and the manager enqueues the new coarse key.
    #[test]
    fn lod_change_is_a_fresh_key() {
        let layer = test_layer();
        let lib = test_library();
        let reg = registry();
        let surf = layer.sample_world(0.0, 0.0, SEED).height;
        let cfg = StreamingConfig { clip_half_bricks: 4, max_resident_bricks: 1_000_000, max_bricks_per_frame: 1_000_000 };

        let src = WorldgenSource::new(&layer, &lib, SEED);
        let mut mgr = ResidencyManager::new();
        let cam0 = [0.0_f32, surf, 0.0];
        mgr.update(cam0, &cfg, &src);
        mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
        mgr.take_dirty();
        // Every resident brick is in SOME shell of the desired clipmap (keys are well-formed).
        let d0 = desired_clipmap(cam0, &cfg);
        for e in mgr.resident_entries() {
            assert!(d0.contains_key(&BrickKey { coord: e.coord, lod: e.lod }), "resident keys are desired");
        }

        // Jump the camera far in +X so the inner cube fully shifts: the old keys leave, new ones enter and are
        // enqueued (a re-key, not a retag).
        let jump = brick_span(0) * (cfg.clip_half_bricks as f32 * 2.0 + 1.0);
        let cam1 = [jump, surf, 0.0];
        let dropped = mgr.update(cam1, &cfg, &src);
        assert!(dropped > 0, "the fully-shifted clipmap drops the old keys");
        assert!(mgr.pending() > 0, "and enqueues the new keys (fresh voxelize at their LOD)");
    }

    /// A2 — SURFACE-SHELL SCALING (`VOXEL_LARGE_SCENE_PLAN` §7). With the cap applied AFTER the classify split,
    /// the SURFACE candidate set the residency keeps scales ~Θ(H²) (a thin shell over the terrain), NOT Θ(H³)
    /// (the clip volume). We grow `clip_half` and fit the exponent of the surface-candidate count vs H: the
    /// classify-pruned volume (buried interior + high sky) must NOT inflate it to a cubic. The UNCAPPED tiling
    /// (`desired_clipmap`) by contrast grows ~cubically — so this also proves the cap-after-classify is what
    /// turns Θ(H³)→Θ(H²).
    #[test]
    fn surface_candidates_scale_quadratically_not_cubically() {
        let layer = test_layer();
        let lib = test_library();
        let surf = layer.sample_world(0.0, 0.0, SEED).height;
        let src = WorldgenSource::new(&layer, &lib, SEED);
        let cam = [0.0_f32, surf, 0.0];

        // The surface-candidate count for a given clip_half: a cold uncapped `update` enqueues exactly the
        // surface shell (classify prunes the buried/sky volume), so `pending()` IS the surface-candidate count.
        let surface_count = |half: i32| -> f64 {
            let cfg = StreamingConfig {
                clip_half_bricks: half,
                max_resident_bricks: usize::MAX, // uncapped — measure the true surface shell, not the cap
                max_bricks_per_frame: 1,
            };
            let mut mgr = ResidencyManager::new();
            mgr.update(cam, &cfg, &src);
            mgr.pending() as f64
        };
        // The full UNCAPPED geometric tiling (volume) for the same half — the Θ(H³) baseline the classify avoids.
        let volume_count = |half: i32| -> f64 { clipmap_uncapped_len(cam, half) as f64 };

        // Fit the power-law exponent p in count ≈ c·H^p from two clip_half samples: p = ln(n2/n1)/ln(h2/h1).
        let (h1, h2) = (8.0_f64, 16.0_f64);
        let exponent = |f: &dyn Fn(i32) -> f64| (f(h2 as i32).ln() - f(h1 as i32).ln()) / (h2 / h1).ln();

        let surf_p = exponent(&surface_count);
        let vol_p = exponent(&volume_count);
        // The surface shell scales ~quadratically: comfortably below cubic, around 2. (Allow slack for the
        // discrete shell + the band-limited terrain's gentle slope; the point is it is NOT ~3.)
        assert!(
            surf_p < 2.6,
            "surface candidates must scale sub-cubically (Θ(H²)), got exponent {surf_p:.2} (volume {vol_p:.2})"
        );
        // Sanity: the uncapped VOLUME really is ~cubic (so the surface sub-cubic result is meaningful).
        assert!(vol_p > 2.6, "the uncapped clip volume scales ~cubically, got {vol_p:.2}");
        // And the surface shell is materially SMALLER than the volume at the large half (the cull works).
        assert!(
            surface_count(h2 as i32) < 0.5 * volume_count(h2 as i32),
            "the surface shell is far smaller than the clip volume at clip_half={h2}"
        );
    }

    // ===== D1d — SHELL-FIRST ENUMERATION ORACLE =================================================
    // The candidate set MUST be a CONSERVATIVE SUPERSET of every `Surface` brick the full-cube path keeps; a
    // missed brick is a render hole. These tests pin that: the NEW surface-first resident set (after classify
    // + cap) is IDENTICAL to the OLD cube-enumerate-then-classify resident set, on scenes that STRESS the
    // worldgen superset — a CLIFF (a steep height step spanning many vertical bricks across one column
    // boundary), a THIN WALL, a surface crossing a LOD SEAM, and a FLAT plane. Plus: the coarse LODs now
    // actually enumerate at the shipping clip_half = 160 (the cube path bailed at MAX_CLIP_ENUMERATION).

    /// A synthetic [`BrickSource`] whose classify + `surface_bricks_in` are the WORLDGEN height-based logic
    /// (the exact `3×3`-tap envelope + the column Surface-band derivation) over a PROGRAMMABLE height field
    /// `h(wx, wz)`. This lets the oracle drive the worldgen surface predicate with adversarial heightfields a
    /// real `HeightLayer` can't easily produce on demand (a vertical cliff, a 1-brick-wide wall), so the
    /// superset guarantee is tested against the hard cases. The two methods share `surf_minmax`, exactly as
    /// `WorldgenSource` shares [`WorldgenSource::column_surf_minmax`] — so this faithfully mirrors the shipping
    /// source's contract (the classify formula here is copied verbatim from `source.rs`, with the SAME band
    /// derivation in `surface_bricks_in`). `brick` is unused by `update` (classify prunes/keeps), so it is a
    /// trivial air brick.
    struct HeightFnSource<F: Fn(f64, f64) -> f64 + Sync> {
        h: F,
    }
    impl<F: Fn(f64, f64) -> f64 + Sync> HeightFnSource<F> {
        /// The `(surf_min, surf_max)` envelope over column `(bx, bz)` — the SAME `3×3` taps over the
        /// `+1`-brick-expanded footprint that `WorldgenSource::column_surf_minmax` uses.
        fn surf_minmax(&self, bx: i32, bz: i32, span: f64) -> (f64, f64) {
            let wmx = bx as f64 * span;
            let wmz = bz as f64 * span;
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for iz in 0..3 {
                let wz = wmz - span + (iz as f64) * 1.5 * span;
                for ix in 0..3 {
                    let wx = wmx - span + (ix as f64) * 1.5 * span;
                    let h = (self.h)(wx, wz);
                    lo = lo.min(h);
                    hi = hi.max(h);
                }
            }
            (lo, hi)
        }
    }
    impl<F: Fn(f64, f64) -> f64 + Sync> BrickSource for HeightFnSource<F> {
        fn brick(&self, _c: IVec3, _l: u32, _r: &BlockRegistry) -> Brick {
            Brick::uniform(super::super::palette::BlockId::AIR)
        }
        fn classify(&self, coord: IVec3, lod: u32) -> BrickClass {
            let span = brick_span(lod) as f64;
            let (surf_min, surf_max) = self.surf_minmax(coord.x, coord.z, span);
            let bmin = coord.y as f64 * span;
            let bmax = bmin + span;
            if bmin >= surf_max + span {
                BrickClass::Air
            } else if surf_min >= bmax + span {
                BrickClass::Interior
            } else {
                BrickClass::Surface
            }
        }
        fn surface_bricks_in(&self, lo: IVec3, hi: IVec3, lod: u32, out: &mut Vec<IVec3>) {
            const PAD: i32 = 1;
            let span = brick_span(lod) as f64;
            for bz in lo.z..=hi.z {
                for bx in lo.x..=hi.x {
                    let (surf_min, surf_max) = self.surf_minmax(bx, bz, span);
                    let by_lo = ((surf_min / span).floor() as i32 - 1 - PAD).max(lo.y);
                    let by_hi = ((surf_max / span).floor() as i32 + PAD).min(hi.y);
                    for by in by_lo..=by_hi {
                        out.push(IVec3::new(bx, by, bz));
                    }
                }
            }
        }
    }

    /// The OLD resident-set ORACLE: enumerate the full `level_box \ level_hole` CUBE per LOD
    /// ([`desired_clipmap`]), classify every key, and keep the `(coord, lod)` of those that are `Surface`.
    /// This is the ground-truth set the surface-first path must reproduce EXACTLY. No cap (the oracle measures
    /// the geometric surface set, not the cap).
    fn cube_surface_set(cam: [f32; 3], cfg: &StreamingConfig, src: &dyn BrickSource) -> FxHashSet<BrickKey> {
        desired_clipmap(cam, cfg)
            .keys()
            .filter(|k| matches!(src.classify(k.coord, k.lod), BrickClass::Surface))
            .copied()
            .collect()
    }

    /// The NEW resident-set under test: run `update` (which enumerates via the shell-first
    /// [`desired_clipmap_surface`] + re-confirms with classify) at an UNCAPPED budget, and read the enqueued
    /// surface keys — exactly the set that becomes resident. `max_resident_bricks = usize::MAX` so the cap
    /// never perturbs the comparison (the oracle tests the ENUMERATION, not the cap, which has its own tests).
    fn shell_surface_set(cam: [f32; 3], cfg: &StreamingConfig, src: &dyn BrickSource) -> FxHashSet<BrickKey> {
        let mut mgr = ResidencyManager::new();
        mgr.update(cam, cfg, src);
        mgr.queued.iter().copied().collect()
    }

    /// THE ORACLE: surface-first ≡ cube-then-classify, on cliff / thin-wall / LOD-seam / flat. Uses a
    /// `clip_half` small enough that the cube path actually runs (no MAX_CLIP_ENUMERATION bail), so the two
    /// enumerations can be compared key-for-key. If `surface_bricks_in` ever MISSED a `Surface` brick the cube
    /// path keeps, the set-equality assert fails (the candidate set is not a superset).
    #[test]
    fn shell_first_resident_set_matches_cube_oracle() {
        // Small half so the cube path enumerates without bailing; an asymmetric cam to exercise sub-cell snap.
        let cfg = StreamingConfig { clip_half_bricks: 10, max_resident_bricks: usize::MAX, max_bricks_per_frame: 1 };
        let span0 = brick_span(0) as f64;

        // (a) FLAT plane at y = 3.3 m: the surface is one near-constant band; trivial but the baseline.
        let flat = HeightFnSource { h: |_x: f64, _z: f64| 3.3 };
        // (b) CLIFF: a steep step at world x = 0 — h jumps from 0 to 40 m across one column boundary, so a
        //     TALL vertical wall of surface bricks spans the seam between the two adjacent columns. This is the
        //     case the per-column `±1`-brick taps + the PAD must bracket (the worst case for a missed brick).
        let cliff = HeightFnSource { h: |x: f64, _z: f64| if x < 0.0 { 0.0 } else { 40.0 } };
        // (c) THIN WALL: a 1-brick-wide ridge at x ∈ [0, span0) rising to 30 m, flat elsewhere — a thin sheet
        //     of vertical surface bricks the column enumeration must not skip.
        let wall = HeightFnSource {
            h: move |x: f64, _z: f64| if (0.0..span0).contains(&x) { 30.0 } else { 1.0 },
        };
        // (d) LOD-SEAM crossing: a gentle slope so the surface threads through every LOD shell — the surface
        //     crosses the level boundaries, stressing the per-level box-clip + hole-clip of the candidate set.
        let slope = HeightFnSource { h: |x: f64, z: f64| 0.15 * x + 0.05 * z };

        // Cameras chosen so the surface passes through the clipmap (near the terrain) — place the cam at the
        // height the field takes near the origin so LOD0 straddles it.
        let cases: [(&str, &dyn BrickSource, [f32; 3]); 4] = [
            ("flat", &flat, [0.5, 3.3, 0.5]),
            ("cliff", &cliff, [0.3, 20.0, 0.3]),
            ("thin_wall", &wall, [0.3, 15.0, 0.3]),
            ("lod_seam", &slope, [0.7, 0.0, -0.4]),
        ];
        for (name, src, cam) in cases {
            let oracle = cube_surface_set(cam, &cfg, src);
            let shell = shell_surface_set(cam, &cfg, src);
            // SUPERSET (the load-bearing direction): every Surface brick the cube path keeps is in the shell
            // candidate set — a failure here is a render hole. Report the first miss for diagnosis.
            for k in &oracle {
                assert!(
                    shell.contains(k),
                    "[{name}] shell-first MISSED a Surface brick the cube keeps: {k:?} (a render hole)"
                );
            }
            // And IDENTICAL: the shell set introduces no spurious Surface key the cube path didn't keep (the
            // classify re-confirm + box-clip prunes the superset's extras), so the resident set is bit-equal.
            assert_eq!(oracle, shell, "[{name}] shell-first resident set must EQUAL the cube oracle set");
            assert!(!oracle.is_empty(), "[{name}] the surface set is non-empty (the surface is in the clipmap)");
        }
    }

    /// The SAME superset/equality oracle on the REAL [`WorldgenSource`] + the shipping height layer (not just
    /// the synthetic source) — proving the production worldgen `surface_bricks_in` reproduces its own
    /// `classify`-Surface set over the real band-limited terrain, at a comparable small `clip_half`.
    #[test]
    fn shell_first_matches_cube_oracle_real_worldgen() {
        let layer = test_layer();
        let lib = test_library();
        let src = WorldgenSource::new(&layer, &lib, SEED);
        let surf = layer.sample_world(0.0, 0.0, SEED).height;
        let cfg = StreamingConfig { clip_half_bricks: 12, max_resident_bricks: usize::MAX, max_bricks_per_frame: 1 };
        let cam = [0.0_f32, surf, 0.0];
        let oracle = cube_surface_set(cam, &cfg, &src);
        let shell = shell_surface_set(cam, &cfg, &src);
        for k in &oracle {
            assert!(shell.contains(k), "real worldgen: shell-first MISSED Surface brick {k:?} (a render hole)");
        }
        assert_eq!(oracle, shell, "real worldgen: shell-first resident set must EQUAL the cube oracle");
        assert!(!oracle.is_empty(), "the real worldgen surface set is non-empty");
    }

    /// COARSE LODS NOW ENUMERATE at the shipping `clip_half = 160` (the D1d payoff). The cube
    /// [`desired_clipmap`] bailed at [`MAX_CLIP_ENUMERATION`] on LOD0's `321³ ≈ 33 M` keys — so it was LOD0-only
    /// and the coarse reach was fiction. The shell-first [`desired_clipmap_surface`] enumerates `Θ(H²)`
    /// candidates, so EVERY level `0..=MAX_LOD` is present AND the candidate count stays well under
    /// `MAX_CLIP_ENUMERATION`. We use the synthetic flat source (a real surface in every shell) so each level
    /// has surface candidates to enumerate.
    #[test]
    fn shell_first_enumerates_all_coarse_lods_at_clip_half_160() {
        let cfg = StreamingConfig::default(); // clip_half = 160, the shipping config
        assert_eq!(cfg.clip_half_bricks, 160, "this test pins the shipping clip_half");
        // A flat plane near y = 0 so the surface threads EVERY shell (each level's box straddles it).
        let src = HeightFnSource { h: |_x: f64, _z: f64| 0.0 };
        let cam = [0.0_f32, 0.0, 0.0];

        // FIRST: the cube path is the broken baseline — it bails at the ceiling and is LOD0-only.
        let cube = desired_clipmap(cam, &cfg);
        assert!(cube.len() > MAX_CLIP_ENUMERATION, "the cube path hits the enumeration ceiling at clip_half 160");
        let cube_has_coarse = cube.keys().any(|k| k.lod > 0);
        assert!(!cube_has_coarse, "the cube path bailed before reaching ANY coarse shell (the D1c bug)");

        // SHELL-FIRST: every LOD enumerates, and the candidate count is far under the ceiling.
        let surf = desired_clipmap_surface(cam, &cfg, &src);
        assert!(
            surf.len() <= MAX_CLIP_ENUMERATION,
            "shell-first stays under the enumeration ceiling (Θ(H²)), got {}",
            surf.len()
        );
        for lod in 0..=MAX_LOD {
            assert!(
                surf.keys().any(|k| k.lod == lod),
                "shell-first must enumerate LOD{lod} at clip_half 160 (coarse reach restored); got levels {:?}",
                {
                    let mut ls: Vec<u32> = surf.keys().map(|k| k.lod).collect();
                    ls.sort_unstable();
                    ls.dedup();
                    ls
                }
            );
        }
    }
}
