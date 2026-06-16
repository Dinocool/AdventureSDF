//! **The brick SOURCE abstraction — one residency pipeline for every scene.**
//!
//! [`super::streaming::ResidencyManager`] streams a camera-following clipmap of `(coord, lod)` bricks. The
//! ONLY thing that differs between scenes is where a brick's `8³` voxels come from: the procedural worldgen
//! surface, or a baked static [`BrickMap`] (Sponza). This module captures that single degree of freedom in a
//! [`BrickSource`] trait so the streaming layer ([`super::streaming::ResidencyManager::drain_work_from`]) is
//! scene-agnostic — it sources a brick, applies the shared [`super::edits`] overlay, and stores the non-empty
//! result, identically for every source.
//!
//! A source returns the brick's `8³` CORE (no halo): the halo is the packer's job
//! ([`super::gpu::pack_resident_set`]) which reads SAME-LOD neighbour cores from the resident set, so
//! cross-brick face-exposure / seams work uniformly regardless of which source produced the neighbours.
//!
//! ## The two sources
//! * [`WorldgenSource`] wraps `(layer, lib, seed)` and is a thin pass-through to [`super::voxelize::
//!   voxelize_brick`] — the worldgen brick SSOT, NOT duplicated here. So worldgen residency is bit-identical
//!   to before the refactor.
//! * [`StaticVoxSource`] wraps a loaded fine [`BrickMap`] (the `0.05 m` Sponza voxels) and produces the same
//!   `(coord, lod)` brick the worldgen path would. To keep [`BrickSource::brick`] BOUNDED (sub-ms) at EVERY
//!   LOD — the coarse footprint of a big asset like Sponza is `(2^L)³` fine voxels per coarse cell, so a
//!   per-`brick()` brute scan was `O(512 · 8^L)` (seconds at LOD≥5) — `new` precomputes a MIP PYRAMID ONCE:
//!   `pyramid[0]` is the loaded fine map, `pyramid[L]` is `pyramid[L-1]` downsampled by `2³` (each coarser
//!   voxel aggregates its `2×2×2` children: SOLID-IF-ANY + the DOMINANT child block, deterministic). Then
//!   `brick(coord, L)` is just an `O(512)` extract of the `8³` core from `pyramid[L]` at the brick footprint
//!   — independent of LOD/distance. A brick whose footprint is entirely outside the loaded map is all-air —
//!   so the clipmap naturally BOUNDS the static scene (`drain_work` memoizes those empty bricks once).
//!
//! Every source is [`Sync`] + a PURE function of its inputs, so the parallel [`super::streaming::
//! ResidencyManager::drain_work_from`] stays deterministic (a brick is identical regardless of thread).

use bevy::math::IVec3;
use rustc_hash::FxHashMap;

use super::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, BrickMap, MAX_LOD, brick_span, voxel_index};
use super::palette::{BlockId, BlockRegistry};
use super::voxelize::voxelize_brick;
use crate::sdf_render::worldgen::biome::BiomeLibrary;
use crate::sdf_render::worldgen::layers::height::HeightLayer;

/// The coarse, CHEAP classification of a `(coord, lod)` brick relative to the scene surface — the
/// SURFACE-FOLLOWING RESIDENCY filter ([`super::streaming::ResidencyManager::update`] keeps only
/// [`BrickClass::Surface`] bricks resident, pruning the deep-buried interior + the high sky). It is a
/// conservative geometric predicate computed WITHOUT voxelizing the brick (no `8³` surface evals): a brick
/// is pruned ONLY when it is PROVABLY unhittable by any primary ray (occluded by the surface, or empty sky),
/// so the prune is hole-free for band-limited terrain.
///
/// The whole point is to bound residency + the per-frame voxelize to the SURFACE SHEET (O(clip_half²)) rather
/// than the clipmap VOLUME (O(clip_half³)) — the deep underground (millions of bricks at a large `clip_half`)
/// is never voxelized nor kept resident, because a primary ray hits the surface long before reaching it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrickClass {
    /// PROVABLY entirely above the surface (the whole brick AND its `+1`-brick margin are sky) — no solid
    /// voxel, never a ray hit. Pruned from residency.
    Air,
    /// PROVABLY fully buried — this brick AND the brick directly above it are entirely solid, so NO voxel of
    /// this brick has an exposed (air-adjacent) face. A primary ray hits the shallower surface first, so this
    /// brick is occluded and never the nearest hit. Pruned from residency.
    Interior,
    /// Straddles the surface, or is close enough to it that the conservative margin can't rule out an exposed
    /// face. KEPT resident + voxelized (the surface shell). The DEFAULT (never-prune) class — any uncertainty
    /// resolves here, so pruning is conservative by construction.
    Surface,
}

/// The single degree of freedom the streaming residency varies per scene: where a `(coord, lod)` brick's
/// `8³` voxels come from. Implementors return the brick's CORE grid (the halo is added later by the packer
/// reading neighbour cores), at the LOD's coarse spacing — a TRUE in-place mip, matching
/// [`super::voxelize::voxelize_brick`]'s contract so worldgen and static scenes are interchangeable.
///
/// MUST be a pure function of `(coord, lod, registry)` and [`Sync`], so [`super::streaming::
/// ResidencyManager::drain_work_from`] can voxelize the per-frame batch IN PARALLEL and remain deterministic
/// (every thread yields the identical brick, applied in a fixed order — the resident set is bit-identical to
/// a serial drain).
pub trait BrickSource: Sync {
    /// The `8³` core brick at clipmap key `(coord, lod)`. Spans world `[coord · brick_span(lod),
    /// +brick_span(lod))`; voxel `v`'s world centre is `world_min + (v + 0.5) · lod_voxel_size(lod)`. An
    /// all-air result is dropped by the residency (sparsity) and memoized so it isn't re-sourced.
    fn brick(&self, coord: IVec3, lod: u32, registry: &BlockRegistry) -> Brick;

    /// CHEAP, CONSERVATIVE classification of a `(coord, lod)` brick relative to the surface — the
    /// SURFACE-FOLLOWING RESIDENCY filter ([`BrickClass`]). MUST NOT voxelize the brick (no `8³` surface
    /// evals): it is a coarse geometric predicate the residency runs on EVERY desired key per camera move, so
    /// it has to be far cheaper than `brick`.
    ///
    /// The contract is CONSERVATIVE: return [`BrickClass::Surface`] unless the brick is PROVABLY unhittable by
    /// a primary ray ([`BrickClass::Air`] = entirely above the surface incl. a `+1`-brick margin;
    /// [`BrickClass::Interior`] = fully buried with the brick above it also fully buried, so no exposed face).
    /// A brick adjacent to ANY air MUST classify `Surface` — that is what keeps the prune hole-free.
    ///
    /// DEFAULT = [`BrickClass::Surface`] (never prune): correct for any finite/static source where the
    /// wholly-outside reject already bounds residency. Procedural sources (the unbounded worldgen volume)
    /// override this with a height-based predicate.
    #[inline]
    fn classify(&self, _coord: IVec3, _lod: u32) -> BrickClass {
        BrickClass::Surface
    }

    /// **D1d — is [`surface_bricks_in`](Self::surface_bricks_in) EXACT (not merely a superset)?** When `true`,
    /// the source GUARANTEES every coord it yields from `surface_bricks_in` would [`classify`](Self::classify)
    /// as exactly [`BrickClass::Surface`] — so [`super::streaming::ResidencyManager::update`] may SKIP the
    /// per-candidate `classify` re-confirm entirely (the candidates ARE the surface set). This removes the
    /// redundant second pass for sources whose enumeration already computed the classify verdict
    /// ([`WorldgenSource`]'s per-column band IS its classify band — same `column_surf_minmax` SSOT, so the
    /// candidates are precisely the `Surface` bricks, never `Air`/`Interior`).
    ///
    /// DEFAULT = `false`: the candidate set is a conservative SUPERSET and `update` MUST `classify`-re-confirm
    /// each (the safe contract — a source like [`super::source::StaticVoxSource`] yields occupied bricks
    /// including buried `Interior` ones that the enclosed-cull classify still prunes). A source overrides this
    /// to `true` ONLY when its `surface_bricks_in` is its `classify`-`Surface` set verbatim. The D1d oracle's
    /// set-equality (`surface_bricks_in` filtered by `classify` == cube filtered by `classify`) plus this flag
    /// being justified by a SHARED SSOT (the band) keeps it correct: skipping classify is then a NO-OP on the
    /// resulting set, just faster.
    #[inline]
    fn surface_bricks_are_exact(&self) -> bool {
        false
    }

    /// **SHELL-FIRST ENUMERATION (D1d)** — yield a CONSERVATIVE SUPERSET of the bricks in the inclusive
    /// AABB `[lo, hi]` (on the LOD-`lod` brick grid) that [`classify`](Self::classify) could return
    /// [`BrickClass::Surface`] for. The whole point: the camera-following clipmap's resident set is the
    /// SURFACE SHELL (`Θ(H²)`), but [`super::streaming::desired_clipmap`] enumerates the full clip VOLUME
    /// (`Θ(H³)`) and then classifies every cube key — at `clip_half = 160` that is `321³ ≈ 33 M` keys on
    /// LOD0 alone, which BLOWS the [`super::streaming::MAX_CLIP_ENUMERATION`] guard (so the coarse shells +
    /// the far reach never enumerate) and costs ~38 s of single-threaded classify per camera crossing
    /// (D1c). This method lets [`super::streaming::ResidencyManager::update`] enumerate the candidate
    /// surface bricks DIRECTLY — `O(surface) = Θ(H²)` — instead of the cube, restoring the coarse LODs and
    /// killing the 38 s update.
    ///
    /// # The conservative-superset CONTRACT (correctness is paramount — a missed brick is a render hole)
    /// `surface_bricks_in` MUST yield a SUPERSET of every brick in `[lo, hi]` that `classify(coord, lod)`
    /// returns `Surface` for. The downstream [`classify`](Self::classify) RE-CONFIRMS each candidate and
    /// prunes the superset's false positives (so over-inclusion only costs a few extra classify calls — it
    /// is always SAFE). A FALSE NEGATIVE — a `Surface` brick the candidate set omits — is NOT pruned by
    /// anything and renders as a hole, so when in doubt INCLUDE MORE. The exact-tiling residency
    /// ([`super::streaming::level_resident`]) is unchanged: the candidate is then `classify`-confirmed and
    /// box-clipped exactly as the cube path's keys were, so the resident set is IDENTICAL to enumerating the
    /// full cube and classifying it (the D1d oracle test asserts this set-equality).
    ///
    /// `out` is APPENDED to (not cleared) — the caller accumulates candidates across the per-level
    /// box-minus-hole regions. Candidates may be DUPLICATED or fall outside `[lo, hi]`; the caller
    /// box-clips + dedups via the resident/queued/memo guards, so an impl may be loose at the edges.
    ///
    /// # DEFAULT — the full box (today's behaviour, the graceful fallback)
    /// The default enumerates EVERY brick in `[lo, hi]` — a trivially-correct (if not cheap) superset for any
    /// source without a fast surface query. A source that CAN cheaply bound its surface (worldgen's
    /// heightfield, a static map's occupied keys) overrides this to yield only the `Θ(H²)` shell.
    fn surface_bricks_in(&self, lo: IVec3, hi: IVec3, _lod: u32, out: &mut Vec<IVec3>) {
        // Graceful fallback: the full box (a correct superset — every Surface brick is in the box).
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    out.push(IVec3::new(x, y, z));
                }
            }
        }
    }
}

/// The WORLDGEN brick source: a thin wrapper over `(layer, lib, seed)` that delegates to the
/// [`super::voxelize::voxelize_brick`] SSOT. Holds shared references (all [`Sync`]), so the parallel drain is
/// determinism-preserving and worldgen residency is BIT-IDENTICAL to the pre-refactor direct call.
pub struct WorldgenSource<'a> {
    /// The procedural height layer (the surface + materials SSOT).
    pub layer: &'a HeightLayer,
    /// The biome library (climate → strata materials).
    pub lib: &'a BiomeLibrary,
    /// The deterministic worldgen seed.
    pub seed: u64,
}

impl<'a> WorldgenSource<'a> {
    /// Wrap the worldgen sampling context. Cheap — just stores the shared references.
    pub fn new(layer: &'a HeightLayer, lib: &'a BiomeLibrary, seed: u64) -> Self {
        Self { layer, lib, seed }
    }
}

impl BrickSource for WorldgenSource<'_> {
    #[inline]
    fn brick(&self, coord: IVec3, lod: u32, registry: &BlockRegistry) -> Brick {
        voxelize_brick(coord, lod, self.layer, self.lib, registry, self.seed)
    }

    /// HEIGHT-BASED conservative classification of a worldgen brick (the SURFACE-FOLLOWING RESIDENCY prune).
    /// The worldgen surface is a height field: a voxel is solid iff its centre Y is `<= surface_height(x, z)`.
    /// So a brick's solidity is bounded by the surface min/max over its XZ footprint:
    ///
    /// * Sample the surface ([`HeightLayer::sample_world`]`.height`) over the brick's XZ footprint EXPANDED by
    ///   ONE brick on every side — a `3×3` grid of taps (corners + edge midpoints + centre of the expanded
    ///   footprint) — and take the conservative `surf_min` / `surf_max`. The `+1`-brick expansion makes the
    ///   bound cover the halo neighbours' columns too (face-exposure across a brick boundary reads the
    ///   neighbour), so a brick that is buried but ADJACENT to an exposed column is not mis-pruned.
    /// * The brick spans world Y `[brick_min_y, brick_max_y) = [coord.y·span, coord.y·span + span)`, `span =
    ///   brick_span(lod)`.
    /// * **AIR** iff `brick_min_y >= surf_max + span`: the WHOLE brick (and a `span` margin above the highest
    ///   surface in the footprint) is above the surface ⇒ every voxel is air ⇒ no ray hit.
    /// * **INTERIOR** iff `surf_min >= brick_max_y + span`: the LOWEST surface in the footprint is at least one
    ///   brick-span ABOVE the brick's top ⇒ this brick AND the brick directly above it are FULLY solid ⇒ no
    ///   voxel of this brick has an air-adjacent (exposed) face ⇒ occluded, never the nearest hit.
    /// * else **SURFACE** (the surface passes through the brick or its `+1` margin).
    ///
    /// The `+span` margins are what make the prune PROVABLY hole-free for band-limited terrain: a brick is
    /// pruned only when the surface is a full brick-span clear of it (above for Air, below for Interior), so no
    /// voxel that COULD be exposed is ever dropped. CAVEAT — a STEEP CLIFF whose surface varies by more than
    /// `span` BETWEEN the `3×3` taps could in principle slip a feature past the conservative min/max. The
    /// worldgen surface is band-limited (the height layer's tent finalize + the bounded octave amplitudes keep
    /// `|∇h|` finite), so over the `3·span` expanded footprint at any LOD the surface variation stays within
    /// the sampled min/max envelope — the taps bracket the true extrema and the `+span` margin absorbs the
    /// residual, keeping the shell hole-free. (For pathological near-vertical terrain, raise the tap density
    /// or the margin; the default is safe for the shipping band-limited graphs.)
    #[inline]
    fn classify(&self, coord: IVec3, lod: u32) -> BrickClass {
        let span = brick_span(lod) as f64;
        let (surf_min, surf_max) = self.column_surf_minmax(coord.x, coord.z, span);
        let brick_min_y = coord.y as f64 * span;
        let brick_max_y = brick_min_y + span;
        if brick_min_y >= surf_max + span {
            BrickClass::Air // wholly above the surface (+1-brick margin) ⇒ no solid voxel
        } else if surf_min >= brick_max_y + span {
            BrickClass::Interior // brick + the one above it fully buried ⇒ no exposed face
        } else {
            BrickClass::Surface // the surface (or its margin) passes through ⇒ keep resident
        }
    }

    /// **SHELL-FIRST candidate enumeration (D1d)** for the worldgen heightfield: yield ONLY the bricks in
    /// `[lo, hi]` (on grid `lod`) that [`classify`](Self::classify) could mark `Surface`, column by column —
    /// `O((hi.x-lo.x)·(hi.z-lo.z))` columns, each yielding a thin O(1) vertical span, so the whole region is
    /// `Θ(H²)` (the surface SHEET) instead of the `Θ(H³)` cube the default enumerates.
    ///
    /// PER-COLUMN EXACTNESS (the superset guarantee): for each brick column `(bx, bz)` this uses the IDENTICAL
    /// per-column `surf_min`/`surf_max` envelope that [`classify`](Self::classify) computes (the shared
    /// [`column_surf_minmax`](Self::column_surf_minmax) SSOT — same `3×3` taps over the `+1`-brick-expanded
    /// XZ footprint, which already brackets CLIFFS: a steep slope's tall vertical wall of surface bricks
    /// between adjacent columns is covered because each column's taps reach one brick into its neighbours, so
    /// `surf_min`/`surf_max` of BOTH bordering columns span the cliff face). The vertical Surface range for a
    /// column is then EXACTLY `classify`'s non-Air ∧ non-Interior `by`-band (solved for the integer `by` range
    /// in [`surface_by_band`](Self::surface_by_band) from the SAME `f64` quantities `classify` compares, so the
    /// two agree brick-for-brick with NO rounding drift — the candidate set EQUALS the `Surface` set, not
    /// merely a superset). This exactness is what lets [`surface_bricks_are_exact`](BrickSource::surface_bricks_are_exact)
    /// return `true` (the residency then SKIPS the redundant per-candidate `classify` re-confirm), and the D1d
    /// oracle test pins it: the surface-first candidate set == the cube path's `Surface` set on cliff / thin
    /// wall / LOD-seam / flat.
    fn surface_bricks_in(&self, lo: IVec3, hi: IVec3, lod: u32, out: &mut Vec<IVec3>) {
        let span = brick_span(lod) as f64;
        for bz in lo.z..=hi.z {
            for bx in lo.x..=hi.x {
                let (surf_min, surf_max) = self.column_surf_minmax(bx, bz, span);
                let (by_lo, by_hi) = Self::surface_by_band(surf_min, surf_max, span);
                let by_lo = by_lo.max(lo.y);
                let by_hi = by_hi.min(hi.y);
                for by in by_lo..=by_hi {
                    out.push(IVec3::new(bx, by, bz));
                }
            }
        }
    }

    /// `true` — worldgen's [`surface_bricks_in`](Self::surface_bricks_in) band IS its
    /// [`classify`](Self::classify) `Surface` band EXACTLY (both from the shared
    /// [`column_surf_minmax`](Self::column_surf_minmax) envelope; the `by`-band is `classify`'s OWN predicate
    /// solved for the integer range in [`surface_by_band`](Self::surface_by_band)). Every yielded coord
    /// classifies exactly `Surface` — never `Air`/`Interior` — so the residency may SKIP the redundant
    /// per-candidate `classify` re-confirm. Pinned brick-for-brick by `worldgen_surface_bricks_in_equals_classify`.
    #[inline]
    fn surface_bricks_are_exact(&self) -> bool {
        true
    }
}

impl WorldgenSource<'_> {
    /// The EXACT integer `by`-band `[by_lo, by_hi]` of LOD bricks (`span = brick_span(lod)`) a column with
    /// height envelope `(surf_min, surf_max)` classifies as [`BrickClass::Surface`] — solved directly from
    /// [`classify`](BrickSource::classify)'s float predicate so `surface_bricks_in` is EXACT (not a superset).
    /// A brick `by` is `Surface` iff it is neither Air nor Interior:
    /// * not **Air**      ⇔ `by·span < surf_max + span`     ⇔ `by < (surf_max + span)/span`,
    /// * not **Interior** ⇔ `surf_min < (by+1)·span + span` ⇔ `by > surf_min/span - 2`.
    ///
    /// The bounds below seed from the closed-form `by ∈ ( surf_min/span − 2 , (surf_max + span)/span )` (open),
    /// then SNAP each to satisfy `classify`'s EXACT MULTIPLICATIVE predicate (`by·span` compared to
    /// `surf_max + span` / `surf_min`), NOT the division reformulation — so the band agrees with `classify`
    /// bit-for-bit (a division `(surf_max+span)/span` is NOT bit-identical to the multiplication `by·span`, and
    /// that 1-ULP drift would otherwise drop a boundary brick `classify` keeps, a render hole at an LOD seam).
    /// Each snap converges in ≤1 step. `(by_lo, by_hi)` BEFORE the caller's level-box clamp; `by_lo > by_hi` ⇒
    /// no surface brick in this column.
    #[inline]
    pub(crate) fn surface_by_band(surf_min: f64, surf_max: f64, span: f64) -> (i32, i32) {
        // by_hi = largest integer with `by·span < surf_max + span` (classify's not-Air predicate, multiplicative).
        let air_top = surf_max + span;
        let mut by_hi = ((air_top) / span).floor() as i32; // seed
        while (by_hi as f64) * span >= air_top {
            by_hi -= 1; // too high — `by·span` reaches the Air threshold; step down
        }
        while ((by_hi + 1) as f64) * span < air_top {
            by_hi += 1; // one more brick is still below the threshold (not-Air)
        }
        // by_lo = smallest integer with `surf_min < (by+1)·span + span` (classify's not-Interior, multiplicative).
        let mut by_lo = (surf_min / span).floor() as i32 - 1; // seed
        while surf_min >= ((by_lo + 1) as f64) * span + span {
            by_lo += 1; // still Interior — `surf_min` is a full span above the brick top; step up
        }
        while surf_min < (by_lo as f64) * span + span {
            by_lo -= 1; // the brick below is ALSO not-Interior — include it
        }
        (by_lo, by_hi)
    }
}

impl WorldgenSource<'_> {
    /// The conservative `(surf_min, surf_max)` height envelope over brick COLUMN `(bx, bz)`'s XZ footprint at
    /// brick `span` — the SHARED SSOT for both [`classify`](BrickSource::classify) (the per-key prune) and
    /// [`surface_bricks_in`](BrickSource::surface_bricks_in) (the shell-first candidate band), so the two can
    /// NEVER disagree about which `by` rows are `Surface` (the superset guarantee falls out of using one
    /// formula). Samples a `3×3` grid over the footprint EXPANDED by one brick on each side
    /// (`[world_min - span, world_min + 2·span]`, taps at `world_min - span + i·1.5·span`): the `+1`-brick
    /// expansion makes the bound cover the halo neighbours' columns (face-exposure across a brick boundary
    /// reads the neighbour) AND brackets a CLIFF between adjacent columns — corners+centre is the conservative
    /// contract; the edge taps only tighten it. The worldgen surface is band-limited, so over the `3·span`
    /// expanded footprint the taps bracket the true extrema and the `±span` `classify` margins absorb the
    /// residual (the prune stays hole-free; see [`classify`](BrickSource::classify)'s cliff caveat).
    #[inline]
    fn column_surf_minmax(&self, bx: i32, bz: i32, span: f64) -> (f64, f64) {
        let world_min_x = bx as f64 * span;
        let world_min_z = bz as f64 * span;
        let mut surf_min = f64::INFINITY;
        let mut surf_max = f64::NEG_INFINITY;
        for iz in 0..3 {
            let wz = world_min_z - span + (iz as f64) * 1.5 * span;
            for ix in 0..3 {
                let wx = world_min_x - span + (ix as f64) * 1.5 * span;
                let h = self.layer.sample_world(wx, wz, self.seed).height as f64;
                surf_min = surf_min.min(h);
                surf_max = surf_max.max(h);
            }
        }
        (surf_min, surf_max)
    }
}

/// The STATIC `.vox` brick source: produces a `(coord, lod)` brick from a loaded fine [`BrickMap`] (the
/// `0.05 m` baked Sponza voxels, all at the LOD0 grid). Reproduces what [`super::voxelize::voxelize_brick`]
/// does for worldgen, but reads the stored voxels (via a precomputed MIP PYRAMID) instead of sampling a
/// surface:
///
/// * **LOD0** copies the fine voxels at the brick's world-voxel footprint (`coord · 8 .. +8`), AIR outside.
/// * **LOD L>0** reads `pyramid[L]`, where each coarser voxel aggregates its `2×2×2` finer children —
///   SOLID iff ANY child is solid (the occupancy/visibility invariant: a coarse cell is solid iff any fine
///   voxel under it is); its block is the DOMINANT (mode) child block (deterministic tie-break: lowest
///   [`BlockId`]) so the coarse colour/material is stable run-to-run. Built hierarchically (`pyramid[L]` from
///   `pyramid[L-1]`), so LOD `L`'s coarse block is the dominant-of-dominants over the `2^L` hierarchy rather
///   than the flat mode of the full footprint — both are valid coarse summaries; the hierarchical one is what
///   keeps `brick()` `O(512)` instead of `O(512 · 8^L)`.
///
/// A brick whose footprint is entirely outside the loaded map's voxel bounds is all-air, so the streaming
/// clipmap naturally BOUNDS the static scene to its actual extent (the empty bricks are memoized once by the
/// residency). The pyramid is built ONCE from a `&BrickMap` (a pure function of the [`Sync`] map), so the
/// parallel drain stays deterministic — every thread's `brick()` is a pure read of the shared pyramid.
pub struct StaticVoxSource {
    /// The per-LOD mip pyramid: `pyramid[0]` is the loaded fine map; `pyramid[L]` is `pyramid[L-1]`
    /// downsampled by `2³` (solid-if-any + dominant child). Indexed by LOD (length `MAX_LOD + 1`, shorter only
    /// if a level downsamples to EMPTY — then coarser LOD requests clamp to that last level, which the
    /// wholly-outside reject treats as all-air). Each level is keyed by ITS OWN brick-coord grid: a LOD-`L`
    /// brick at `coord` reads `pyramid[L]` at the world-voxel footprint `coord · 8 .. +8` (in level-`L` voxel
    /// coords).
    pyramid: Vec<BrickMap>,
    /// Per-LOD inclusive/exclusive world-voxel solid bounds (in that level's OWN voxel coords) for the fast
    /// "wholly outside ⇒ air" reject. `bounds[L]` is `Some((lo, hi))` iff `pyramid[L]` has any solid voxel.
    bounds: Vec<Option<(IVec3, IVec3)>>,
}

impl StaticVoxSource {
    /// Wrap a loaded fine [`BrickMap`] as a brick source, precomputing the MIP PYRAMID + per-level solid
    /// bounds ONCE so every `brick(coord, lod)` is a bounded `O(512)` extract (no per-brick scan of a
    /// `(2^lod)³` footprint). O(total voxels · 8/7) once — the pyramid is geometric (~`1/7` extra bricks over
    /// the fine map). Clones the fine map's bricks into `pyramid[0]`; the source then owns its pyramid (no
    /// borrow of the caller's map), so it is `'static` + trivially [`Sync`].
    pub fn new(map: &BrickMap) -> Self {
        // pyramid[0] = the fine map (cloned so the source owns it). pyramid[L] = pyramid[L-1] downsampled 2³.
        let mut pyramid: Vec<BrickMap> = Vec::with_capacity(MAX_LOD as usize + 1);
        let mut fine = BrickMap::new();
        for (c, b) in map.iter() {
            fine.insert(*c, b.clone());
        }
        pyramid.push(fine);
        for _ in 1..=MAX_LOD {
            let prev = pyramid.last().expect("non-empty pyramid");
            if prev.is_empty() {
                // An empty level downsamples to empty — stop; coarser LOD requests clamp to this empty level
                // (the wholly-outside reject then returns AIR for every brick). (Note: a NON-empty level that
                // already fits ONE brick must STILL be downsampled — its 8³ voxels reduce to a 4³ occupied
                // core, a genuinely different coarser brick — so we only short-circuit the truly-empty case.)
                break;
            }
            pyramid.push(downsample_brickmap(prev));
        }
        // Per-level solid world-voxel bounds (in each level's own voxel coords) for the wholly-outside reject.
        let bounds = pyramid.iter().map(brickmap_solid_bounds).collect();
        Self { pyramid, bounds }
    }

    /// The pyramid level a LOD request reads: `lod` clamped to the deepest level we built (a tiny asset
    /// collapses to one brick before `MAX_LOD`, so coarser requests reuse that smallest level).
    #[inline]
    fn level(&self, lod: u32) -> usize {
        (lod as usize).min(self.pyramid.len() - 1)
    }

    /// True iff the world-voxel AABB `[wmin, wmax)` (in `pyramid[level]`'s own voxel coords) cannot contain
    /// any solid voxel of that level (so the brick covering it is air). Cheap conservative reject using the
    /// precomputed per-level bounds.
    #[inline]
    fn wholly_outside(&self, level: usize, wmin: IVec3, wmax: IVec3) -> bool {
        let Some((lo, hi)) = self.bounds[level] else {
            return true; // empty level → everything is air
        };
        wmax.x <= lo.x
            || wmax.y <= lo.y
            || wmax.z <= lo.z
            || wmin.x >= hi.x
            || wmin.y >= hi.y
            || wmin.z >= hi.z
    }
}

impl BrickSource for StaticVoxSource {
    fn brick(&self, coord: IVec3, lod: u32, _registry: &BlockRegistry) -> Brick {
        // Read the matching pyramid level: a LOD-`lod` brick at `coord` is the 8³ core at level-`lod` voxel
        // footprint [coord·8, coord·8+8) — an O(512) extract, BOUNDED regardless of LOD/distance (the coarse
        // aggregation was paid ONCE in `new`, not per brick).
        let level = self.level(lod);
        let origin = coord * BRICK_EDGE;
        let extent = IVec3::splat(BRICK_EDGE);
        if self.wholly_outside(level, origin, origin + extent) {
            return Brick::uniform(BlockId::AIR); // outside the loaded scene → air (clipmap bound)
        }
        let src = &self.pyramid[level];
        let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                for x in 0..BRICK_EDGE {
                    voxels[voxel_index(x, y, z)] = src.voxel_block(origin + IVec3::new(x, y, z));
                }
            }
        }
        Brick::from_voxels(voxels)
    }

    /// STATIC enclosed-cull (always on): prune a fully-BURIED brick from residency. `Interior` iff the brick is
    /// fully solid AND all 6 same-LOD FACE-neighbours are fully solid — then no surface voxel is air-adjacent,
    /// so a primary ray is occluded by the surrounding solid and never reaches it (correct-by-construction,
    /// hole-free; a partial / absent neighbour ⇒ `Surface`). Only the 6 FACES matter for first-hit visibility
    /// (edge/corner neighbours only affect the render HALO, not whether a ray reaches the brick). Reads the
    /// pyramid occupancy ([`Brick::is_full`]) — an O(1) brick-map lookup × 7. For a CLAMPED coarse level (a
    /// tiny asset collapsed below `lod`, so the coord grid ≠ the pyramid level grid) don't prune (`Surface`) —
    /// the wholly-outside reject + empty-memo bound those. Absent brick ⇒ `Air`. Pixel-identical: an `Interior`
    /// brick is never the first hit, so dropping it from the packed set leaves the render unchanged.
    fn classify(&self, coord: IVec3, lod: u32) -> BrickClass {
        let level = self.level(lod);
        // Only the levels we built map a coord 1:1 to a brick on that grid; a clamped coarse lod does not.
        if level != lod as usize {
            return BrickClass::Surface;
        }
        let map = &self.pyramid[level];
        let Some(here) = map.get(coord) else {
            return BrickClass::Air; // absent ⇒ all-air outside the loaded scene
        };
        if !here.is_full() {
            return BrickClass::Surface; // an internal air voxel ⇒ an exposed surface
        }
        // Fully solid: buried iff all 6 FACE-neighbours are fully solid too (no air-adjacent face).
        const N6: [IVec3; 6] = [
            IVec3::new(1, 0, 0),
            IVec3::new(-1, 0, 0),
            IVec3::new(0, 1, 0),
            IVec3::new(0, -1, 0),
            IVec3::new(0, 0, 1),
            IVec3::new(0, 0, -1),
        ];
        for off in N6 {
            match map.get(coord + off) {
                Some(n) if n.is_full() => {}
                _ => return BrickClass::Surface, // a non-full / absent neighbour ⇒ this face is exposed
            }
        }
        BrickClass::Interior
    }

    /// **SHELL-FIRST candidate enumeration (D1d)** for a static map: yield ONLY the OCCUPIED bricks of the
    /// matching pyramid level that intersect `[lo, hi]` — a static source already KNOWS its occupied set (the
    /// `BrickMap` keys), so the candidate count is the asset's stored-brick count clipped to the box, never
    /// the box volume. This is a SUPERSET of every `Surface` brick: `classify` only ever returns `Surface`
    /// for a brick that is PRESENT in the map (an absent brick is `Air`), so iterating the present bricks
    /// covers every `Surface` one (and the buried `Interior` ones, which `classify` then prunes). For a
    /// CLAMPED coarse level (a tiny asset collapsed below `lod`, so the coord grid ≠ the pyramid level grid)
    /// we fall back to the FULL BOX (the trait default) — the coord-grid mismatch makes the level's keys
    /// non-comparable to `[lo, hi]`, and the wholly-outside reject + empty-memo bound that case as before.
    fn surface_bricks_in(&self, lo: IVec3, hi: IVec3, lod: u32, out: &mut Vec<IVec3>) {
        let level = self.level(lod);
        if level != lod as usize {
            // Clamped coarse level: the coord grid differs from the pyramid level grid, so we cannot intersect
            // its keys with `[lo, hi]`. Fall back to the full box (correct superset; the empty-memo bounds it).
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    for x in lo.x..=hi.x {
                        out.push(IVec3::new(x, y, z));
                    }
                }
            }
            return;
        }
        // Iterate the level's occupied bricks (sparse), keeping those inside the box. The candidate count is
        // the stored-brick count clipped to the shell — Θ(surface), not Θ(volume).
        for (coord, _brick) in self.pyramid[level].iter() {
            if in_box_incl(*coord, lo, hi) {
                out.push(*coord);
            }
        }
    }
}

/// True iff `coord` is inside the inclusive AABB `[lo, hi]` (the `source.rs`-local mirror of the streaming
/// module's `in_box`, for the shell-first candidate box-clip).
#[inline]
fn in_box_incl(coord: IVec3, lo: IVec3, hi: IVec3) -> bool {
    coord.x >= lo.x
        && coord.x <= hi.x
        && coord.y >= lo.y
        && coord.y <= hi.y
        && coord.z >= lo.z
        && coord.z <= hi.z
}

/// Downsample one pyramid level into the next-coarser one by `2³`: each coarse voxel aggregates its `2×2×2`
/// finer children — SOLID iff ANY child is solid (the occupancy/visibility invariant), and its block is the
/// DOMINANT (most-frequent) child solid block with a LOWEST-[`BlockId`] tie-break so the result is
/// DETERMINISTIC (independent of iteration order / thread). Iterates the SOURCE bricks only (sparse), writing
/// each into its coarse brick — so the cost is `O(source voxels)`, and over the whole pyramid the work is
/// geometric (`Σ 1/8^k`), bounded by `8/7 ×` the fine map. The coarse brick `c` covers source bricks
/// `[2c, 2c+2)` per axis; coarse voxel `v` (in `[0,8)`) aggregates source voxels `[2v, 2v+2)`.
///
/// Per coarse brick this routes through the [`downsample_children`] SSOT, so the in-RAM pyramid here and the
/// on-demand coarse-brick synthesis in [`super::vxo::source::VxoSource`] are GUARANTEED bit-identical (one
/// reducer, one octant layout — robust by construction). This map-level driver only gathers the 8 children of
/// each coarse brick from the sparse source and hands them off.
pub(crate) fn downsample_brickmap(fine: &BrickMap) -> BrickMap {
    // Every coarse brick that any source brick contributes to (a source brick `fc` feeds coarse `fc/2`).
    let mut coarse_coords: rustc_hash::FxHashSet<IVec3> = rustc_hash::FxHashSet::default();
    for (fc, _) in fine.iter() {
        coarse_coords.insert(IVec3::new(fc.x.div_euclid(2), fc.y.div_euclid(2), fc.z.div_euclid(2)));
    }
    let mut out = BrickMap::new();
    for cc in coarse_coords {
        // Gather the 8 children of this coarse brick (octant `(ox,oy,oz)` ⇒ source brick `2cc + (ox,oy,oz)`).
        let children = gather_children(cc, |c| fine.get(c).cloned());
        out.insert(cc, downsample_children(&children));
    }
    out
}

/// Gather the 8 children of coarse brick `cc` (on the next-finer grid) via `fetch`, indexed by OCTANT:
/// `children[ox + oy·2 + oz·4]` is the finer brick at `2·cc + (ox, oy, oz)` (or `None` if absent ⇒ all-air).
/// The single octant-layout SSOT shared by the in-RAM pyramid ([`downsample_brickmap`]) and the streamed
/// coarse-brick synthesis ([`super::vxo::source::VxoSource`]).
pub(crate) fn gather_children(cc: IVec3, mut fetch: impl FnMut(IVec3) -> Option<Brick>) -> [Option<Brick>; 8] {
    let mut children: [Option<Brick>; 8] = Default::default();
    for oz in 0..2 {
        for oy in 0..2 {
            for ox in 0..2 {
                let fc = IVec3::new(cc.x * 2 + ox, cc.y * 2 + oy, cc.z * 2 + oz);
                children[(ox + oy * 2 + oz * 4) as usize] = fetch(fc);
            }
        }
    }
    children
}

/// Downsample one coarse brick from its 8 finer children by `2³` — the SHARED DOWNSAMPLE SSOT (`source.rs`):
/// each coarse voxel `cv ∈ [0,8)³` aggregates the `2×2×2` finer voxels under it, SOLID iff ANY child voxel is
/// solid, its block the DOMINANT (most-frequent) solid child with a LOWEST-[`BlockId`] tie-break (deterministic
/// regardless of iteration order). `children[ox + oy·2 + oz·4]` is the finer brick at octant `(ox,oy,oz)` (the
/// [`gather_children`] layout); `None` ⇒ that octant is all-air. The coarse voxel `cv` lives in octant
/// `cv / 4`; within that child it covers finer voxels `[(cv % 4)·2, +2)` per axis. Both the in-RAM
/// [`downsample_brickmap`] and the on-demand streamed pyramid call THIS, so a coarse `.vxo` brick is bit-for-bit
/// the static-source coarse brick (the §B1.7 parity FIX).
pub(crate) fn downsample_children(children: &[Option<Brick>; 8]) -> Brick {
    let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
    for cz in 0..BRICK_EDGE {
        for cy in 0..BRICK_EDGE {
            for cx in 0..BRICK_EDGE {
                // The child octant this coarse voxel falls in, and its `2³` finer-voxel base within that child.
                let octant = (cx / 4 + (cy / 4) * 2 + (cz / 4) * 4) as usize;
                let Some(child) = &children[octant] else { continue }; // absent octant ⇒ all-air
                let base = IVec3::new((cx % 4) * 2, (cy % 4) * 2, (cz % 4) * 2);
                // Tally the 2×2×2 finer solid blocks under this coarse voxel; resolve the dominant.
                let mut tally: FxHashMap<u16, u32> = FxHashMap::default();
                for dz in 0..2 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let b = child.get(base.x + dx, base.y + dy, base.z + dz);
                            if !b.is_air() {
                                *tally.entry(b.0).or_insert(0) += 1;
                            }
                        }
                    }
                }
                voxels[voxel_index(cx, cy, cz)] = dominant_block(&tally);
            }
        }
    }
    Brick::from_voxels(voxels)
}

/// The DOMINANT (most-frequent) block in a `block id → count` tally, tie-broken by the LOWEST [`BlockId`] so
/// the result is DETERMINISTIC regardless of map iteration order. Empty tally → AIR (no solid child).
fn dominant_block(counts: &FxHashMap<u16, u32>) -> BlockId {
    let mut best: Option<(u32, u16)> = None; // (count, block id)
    for (&id, &c) in counts {
        best = match best {
            Some((bc, bid)) if bc > c || (bc == c && bid <= id) => Some((bc, bid)),
            _ => Some((c, id)),
        };
    }
    best.map(|(_, id)| BlockId(id)).unwrap_or(BlockId::AIR)
}

/// The inclusive/exclusive world-voxel solid bounds of a [`BrickMap`] (in its own voxel coords): a brick's
/// `8³` span `[bc·8, bc·8+8)` bounds its solids (stored bricks are never empty). `None` iff the map is empty.
/// Used per pyramid level for the cheap "wholly outside ⇒ air" reject.
fn brickmap_solid_bounds(map: &BrickMap) -> Option<(IVec3, IVec3)> {
    let mut lo = IVec3::splat(i32::MAX);
    let mut hi = IVec3::splat(i32::MIN);
    let mut any = false;
    for (bc, _brick) in map.iter() {
        let bmin = *bc * BRICK_EDGE;
        let bmax = bmin + IVec3::splat(BRICK_EDGE);
        lo = lo.min(bmin);
        hi = hi.max(bmax);
        any = true;
    }
    any.then_some((lo, hi))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_render::worldgen::biome::{
        BiomeDef, BiomeId, BiomeLibrary, StrataLayer, TerrainMatId, TerrainSurfaceMaterial,
    };
    use crate::sdf_render::worldgen::coord::LayerId;
    use crate::sdf_render::worldgen::layers::erosion::ErosionParams;
    use crate::sdf_render::worldgen::layers::height::{HeightLayer, HeightParams};

    const SEED: u64 = 0xA15E_C0DE_2026;

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

    /// Build a fine `BrickMap` from a closure `(world_voxel) -> BlockId` over a world-voxel AABB `[lo, hi)`.
    fn map_from_fn(lo: IVec3, hi: IVec3, f: impl Fn(IVec3) -> BlockId) -> BrickMap {
        use super::super::brickmap::{brick_coord_of_voxel, voxel_index as vi};
        let mut dense: FxHashMap<IVec3, Box<[BlockId; BRICK_VOXELS]>> = FxHashMap::default();
        for z in lo.z..hi.z {
            for y in lo.y..hi.y {
                for x in lo.x..hi.x {
                    let wv = IVec3::new(x, y, z);
                    let b = f(wv);
                    if b.is_air() {
                        continue;
                    }
                    let bc = brick_coord_of_voxel(wv);
                    let local = wv - bc * BRICK_EDGE;
                    let arr = dense.entry(bc).or_insert_with(|| Box::new([BlockId::AIR; BRICK_VOXELS]));
                    arr[vi(local.x, local.y, local.z)] = b;
                }
            }
        }
        let mut map = BrickMap::new();
        for (c, arr) in dense {
            map.insert(c, Brick::from_voxels(arr));
        }
        map
    }

    /// LOD0 EXTRACT: the StaticVoxSource LOD0 brick at `coord` reproduces the loaded fine voxels at the
    /// brick's world-voxel footprint exactly (core = `map.voxel_block(coord·8 + local)`).
    #[test]
    fn static_lod0_extracts_fine_voxels_verbatim() {
        // A solid slab in world voxels y∈[0,4): a brick straddling it has 4 solid layers + 4 air layers.
        let map = map_from_fn(IVec3::new(0, 0, 0), IVec3::new(8, 8, 8), |wv| {
            if wv.y < 4 { BlockId(1) } else { BlockId::AIR }
        });
        let src = StaticVoxSource::new(&map);
        let reg = registry();
        let b = src.brick(IVec3::ZERO, 0, &reg);
        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                for x in 0..BRICK_EDGE {
                    let want = map.voxel_block(IVec3::new(x, y, z));
                    assert_eq!(b.get(x, y, z), want, "LOD0 core must copy the fine voxel at ({x},{y},{z})");
                }
            }
        }
        assert!(!b.is_empty(), "the slab brick is non-empty");
    }

    /// A brick whose footprint is wholly OUTSIDE the loaded map is all-air at every LOD (the clipmap bound).
    #[test]
    fn static_outside_map_is_air() {
        let map = map_from_fn(IVec3::ZERO, IVec3::new(8, 8, 8), |_| BlockId(1));
        let src = StaticVoxSource::new(&map);
        let reg = registry();
        // Far away in +X at LOD0 and at a coarse LOD: both empty (memoized by drain_work).
        assert!(src.brick(IVec3::new(100, 0, 0), 0, &reg).is_empty(), "far LOD0 brick is air");
        assert!(src.brick(IVec3::new(100, 0, 0), 3, &reg).is_empty(), "far coarse brick is air");
        // An empty map → everything air.
        let empty = BrickMap::new();
        let esrc = StaticVoxSource::new(&empty);
        assert!(esrc.brick(IVec3::ZERO, 0, &reg).is_empty(), "empty map → air");
    }

    /// LOD>0 HIERARCHICAL DOWNSAMPLE (mip pyramid): each coarse voxel is SOLID-IF-ANY over its `2×2×2`
    /// children, and its block is the DOMINANT child block (deterministic). The pyramid is built level-by-level
    /// (`pyramid[L]` from `pyramid[L-1]`), so at LOD1 (one level down) a coarse cell aggregates its `2³` fine
    /// footprint directly. We assert: (a) solid-if-any — a cell with a single solid fine voxel is solid and
    /// carries a VALID PRESENT block (one that actually occurs in its footprint, never an invented id); (b) the
    /// dominant block wins when one block is the clear majority; (c) an all-air footprint stays air. We also
    /// drive LOD2 to confirm solid-if-any propagates UP the hierarchy (a single fine voxel keeps its coarse
    /// ancestors solid two levels up).
    #[test]
    fn static_lod_downsample_dominant_and_solid_if_any() {
        // LOD1 ⇒ each coarse cell aggregates a 2³ fine footprint; one LOD1 brick spans world voxels [0, 8·2).
        // World-voxel region for one LOD1 brick: [0, 8·2) = [0,16) per axis.
        // Cell (0,0,0) footprint [0,2)³: make it 7×block1 + 1×block2 ⇒ dominant = block1.
        // Cell (1,0,0) footprint x∈[2,4): make it a SINGLE solid fine voxel ⇒ solid-if-any, block = that one.
        // Cell (2,0,0) footprint x∈[4,6): all air ⇒ air.
        let map = map_from_fn(IVec3::splat(0), IVec3::splat(16), |wv| {
            // Cell (0,0,0): footprint [0,2)³ — block 1 everywhere except the single corner (1,1,1) = block 2.
            if (0..2).contains(&wv.x) && (0..2).contains(&wv.y) && (0..2).contains(&wv.z) {
                return if wv == IVec3::new(1, 1, 1) { BlockId(2) } else { BlockId(1) };
            }
            // Cell (1,0,0): footprint x∈[2,4), y,z∈[0,2) — exactly one solid fine voxel.
            if wv == IVec3::new(2, 0, 0) {
                return BlockId(3);
            }
            BlockId::AIR
        });
        let src = StaticVoxSource::new(&map);
        let reg = registry();
        let b = src.brick(IVec3::ZERO, 1, &reg);
        // Coarse cell (0,0,0): 7 block-1 + 1 block-2 ⇒ dominant block 1 (a valid present block).
        assert_eq!(b.get(0, 0, 0), BlockId(1), "dominant child block wins the coarse cell");
        // Coarse cell (1,0,0): one solid fine voxel ⇒ solid-if-any, block = that voxel's block (present).
        assert!(!b.get(1, 0, 0).is_air(), "solid-if-any: a single solid fine voxel makes the coarse cell solid");
        assert_eq!(b.get(1, 0, 0), BlockId(3), "the solid coarse cell carries a VALID present child block");
        // Coarse cell (2,0,0): all-air footprint ⇒ air.
        assert_eq!(b.get(2, 0, 0), BlockId::AIR, "an all-air coarse footprint stays air");

        // LOD2 (two levels up): solid-if-any must PROPAGATE — a single fine voxel keeps its coarse ancestor
        // solid two levels up. The fine voxel (2,0,0) → LOD1 cell (1,0,0) → LOD2 cell (0,0,0). The LOD2 brick
        // at coord 0 reads pyramid[2]; its voxel (0,0,0) covers LOD0 voxels [0,4)³, which include (2,0,0).
        let b2 = src.brick(IVec3::ZERO, 2, &reg);
        assert!(!b2.get(0, 0, 0).is_air(), "solid-if-any propagates up the hierarchy (solid two LODs coarser)");
        // And the block it carries is one that actually occurs under it (a present id, never invented).
        let present = [BlockId(1), BlockId(2), BlockId(3)];
        assert!(present.contains(&b2.get(0, 0, 0)), "the coarse-2 cell carries a VALID present block");
    }

    /// The dominant-block tie-break is the LOWEST BlockId, so the downsample is DETERMINISTIC regardless of
    /// hash iteration order: a coarse cell split 4/4 between block 5 and block 2 always resolves to block 2.
    #[test]
    fn static_downsample_tiebreak_is_deterministic() {
        // LOD1 cell (0,0,0): footprint [0,2)³ = 8 voxels; 4 of block 5, 4 of block 2 (a tie).
        let map = map_from_fn(IVec3::splat(0), IVec3::splat(2), |wv| {
            // Split by x: x==0 → block 5 (4 voxels), x==1 → block 2 (4 voxels).
            if wv.x == 0 { BlockId(5) } else { BlockId(2) }
        });
        let src = StaticVoxSource::new(&map);
        let reg = registry();
        // Run several times — the FxHashMap tally order is fixed, but the tie-break (lowest id) guarantees
        // block 2 regardless, and the result is identical run-to-run.
        for _ in 0..8 {
            let b = src.brick(IVec3::ZERO, 1, &reg);
            assert_eq!(b.get(0, 0, 0), BlockId(2), "a tied coarse cell resolves to the lowest BlockId");
        }
    }

    /// The same brick sourced twice is bit-identical (determinism the parallel drain relies on), at LOD0 and
    /// a coarse LOD.
    #[test]
    fn static_source_is_deterministic() {
        let map = map_from_fn(IVec3::splat(0), IVec3::splat(32), |wv| {
            if (wv.x + wv.y + wv.z) % 3 == 0 { BlockId(1) } else if wv.y < 4 { BlockId(2) } else { BlockId::AIR }
        });
        let src = StaticVoxSource::new(&map);
        let reg = registry();
        for &(c, lod) in &[(IVec3::ZERO, 0u32), (IVec3::new(1, 0, 0), 0), (IVec3::ZERO, 2)] {
            assert_eq!(src.brick(c, lod, &reg), src.brick(c, lod, &reg), "source must be deterministic");
        }
    }

    /// HALO correctness via the PACKER: pack a 2-brick static set with `pack_resident_set` and verify each
    /// brick's halo border reads the SAME-LOD neighbour's core voxel (the seam fix), exactly as it would for
    /// worldgen-sourced bricks — proving the static source is halo-interchangeable with worldgen.
    #[test]
    fn static_source_halo_matches_neighbour_core() {
        use super::super::gpu::{halo_index, pack_resident_set};
        use super::super::gpu::ResidentBrick;
        // Two adjacent LOD0 bricks (coords (0,0,0) and (1,0,0)) with DISTINCT solid blocks so a halo cell is
        // identifiable as coming from the neighbour. Fill both bricks fully solid.
        let map = map_from_fn(IVec3::ZERO, IVec3::new(16, 8, 8), |wv| {
            if wv.x < 8 { BlockId(1) } else { BlockId(2) }
        });
        let src = StaticVoxSource::new(&map);
        let reg = registry();
        let b0 = src.brick(IVec3::new(0, 0, 0), 0, &reg);
        let b1 = src.brick(IVec3::new(1, 0, 0), 0, &reg);
        let entries = vec![
            ResidentBrick { coord: IVec3::new(0, 0, 0), brick: &b0, lod: 0 },
            ResidentBrick { coord: IVec3::new(1, 0, 0), brick: &b1, lod: 0 },
        ];
        let patch = pack_resident_set(&entries, &reg);
        // Brick 0's +X halo ring (halo index hx == BRICK_EDGE+1 = 9) must DECODE to brick 1's core (block 2).
        // Read via the SSOT `GpuBrickPatch::cell_block` (R2b) so the oracle decodes the same way the shader does.
        let m0 = &patch.metas[0];
        for hz in 1..=BRICK_EDGE {
            for hy in 1..=BRICK_EDGE {
                let cell = patch.cell_block(m0, halo_index(BRICK_EDGE + 1, hy, hz, 0)).0;
                assert_eq!(cell, 2, "brick 0's +X halo reads the neighbour brick 1's core (block 2)");
            }
        }
    }

    /// PERF GUARD (the coarse-LOD hang anti-regression): a coarse `StaticVoxSource::brick()` over a
    /// Sponza-scale synthetic map must be SUB-MILLISECOND at EVERY LOD — the whole point of the mip pyramid.
    /// The pre-pyramid `downsample_cell` brute-scanned `(2^lod)³` fine voxels per coarse cell, so ONE coarse
    /// brick cost 76 ms (LOD5) / 543 ms (LOD6) / 4.3 s (LOD7) on the real `sponza.vox` — which stalled a drain
    /// thread for hundreds of ms when flying 300-800 m out put the building in a coarse shell. The pyramid
    /// makes `brick()` an `O(512)` extract regardless of LOD, so this holds with huge margin. (The `new` build
    /// cost is paid ONCE, off the per-frame path; it is not what this guards.)
    #[test]
    fn coarse_static_brick_is_sub_millisecond() {
        // A Sponza-scale fine map: a ~600×256×370 voxel solid-ish slab (the test is in VOXEL units, so it is
        // scale-invariant; at 0.05 m that is ~30 m × ~13 m × ~18 m, at the old 0.2 m it was ~120 m × ~50 m ×
        // ~75 m — the voxel COUNT, not the world size, is what stresses the coarse extract).
        // Build a coarse but dense-enough box (a floor slab + walls) so the coarse footprints
        // genuinely aggregate many fine voxels (the worst case for the OLD brute scan).
        let lo = IVec3::new(0, 0, 0);
        let hi = IVec3::new(600, 256, 370);
        let map = map_from_fn(lo, hi, |wv| {
            // Floor slab y∈[0,4), the four perimeter walls (2 voxels thick), and a sparse grid of columns —
            // enough solid mass that a LOD7 coarse cell (128³ fine voxels) is far from trivially empty.
            let on_wall = wv.x < 2 || wv.x >= hi.x - 2 || wv.z < 2 || wv.z >= hi.z - 2;
            let column = (wv.x % 32 < 2) && (wv.z % 32 < 2);
            if wv.y < 4 || on_wall || column { BlockId(1) } else { BlockId::AIR }
        });
        let src = StaticVoxSource::new(&map);
        let reg = registry();
        // Time a coarse brick() at the deepest LODs (the old hot path). A brick near the building centre so the
        // wholly-outside fast reject does NOT short-circuit it (we measure the real extract).
        for lod in [5u32, 6, 7] {
            // The coord on the LOD-`lod` grid covering the building centre (world voxel ~(300,128,185)).
            let span_vox = BRICK_EDGE * (1i32 << lod);
            let coord = IVec3::new(300 / span_vox, 128 / span_vox, 185 / span_vox);
            // Warm once (touch the level), then measure.
            let _ = src.brick(coord, lod, &reg);
            let t = std::time::Instant::now();
            let iters = 32;
            for _ in 0..iters {
                let _ = std::hint::black_box(src.brick(std::hint::black_box(coord), lod, &reg));
            }
            let per = t.elapsed().as_secs_f64() / iters as f64;
            assert!(
                per < 1.0e-3,
                "coarse StaticVoxSource::brick() at LOD{lod} must be sub-ms (was {:.3} ms) — the pyramid keeps \
                 it O(512); a regression here is the coarse-LOD hang returning",
                per * 1.0e3
            );
        }
    }

    /// The whole MIP PYRAMID is built deterministically: two `StaticVoxSource`s over the same map yield
    /// bit-identical coarse bricks at every LOD (the parallel drain relies on this — a coarse brick must be the
    /// same regardless of which thread/source produced it).
    #[test]
    fn static_pyramid_is_deterministic_across_lods() {
        let map = map_from_fn(IVec3::splat(0), IVec3::splat(64), |wv| {
            if (wv.x ^ wv.y ^ wv.z) % 5 == 0 { BlockId(1) } else if wv.y < 8 { BlockId(2) } else { BlockId::AIR }
        });
        let a = StaticVoxSource::new(&map);
        let b = StaticVoxSource::new(&map);
        let reg = registry();
        for lod in 0..=MAX_LOD {
            for &c in &[IVec3::ZERO, IVec3::new(1, 0, 0), IVec3::new(0, 1, 1)] {
                assert_eq!(a.brick(c, lod, &reg), b.brick(c, lod, &reg), "pyramid LOD{lod} must be deterministic");
            }
        }
    }

    /// `WorldgenSource` is a faithful pass-through to `voxelize_brick`: the brick it yields equals a direct
    /// `voxelize_brick` call for the same `(coord, lod)` — worldgen residency is unchanged by the abstraction.
    #[test]
    fn worldgen_source_matches_voxelize_brick() {
        let layer = HeightLayer::new(LayerId(0), HeightParams::default(), ErosionParams::default());
        let lib = test_library();
        let reg = registry();
        let src = WorldgenSource::new(&layer, &lib, SEED);
        for &(c, lod) in &[(IVec3::new(2, -1, 3), 0u32), (IVec3::new(0, 1, 0), 2)] {
            let via_src = src.brick(c, lod, &reg);
            let direct = voxelize_brick(c, lod, &layer, &lib, &reg, SEED);
            assert_eq!(via_src, direct, "WorldgenSource must equal voxelize_brick (SSOT pass-through)");
        }
    }

    /// **D1d EXACTNESS GATE** — `WorldgenSource::surface_bricks_in` yields EXACTLY its `classify`-`Surface`
    /// set, brick-for-brick, over a box: every coord it emits classifies `Surface`, AND every `Surface` brick
    /// in the box is emitted. This is the invariant that justifies `surface_bricks_are_exact() == true` (the
    /// residency skips the per-candidate `classify` re-confirm), so it must hold with NO off-by-one / float
    /// drift. Checked over a spread of LODs (so coarse spans, where the band is several bricks tall, are
    /// exercised) and a non-trivial XZ window of the real band-limited terrain (which includes slopes — the
    /// cliff-ish case for the per-column band).
    #[test]
    fn worldgen_surface_bricks_in_equals_classify() {
        let layer = HeightLayer::new(LayerId(0), HeightParams::default(), ErosionParams::default());
        let lib = test_library();
        let src = WorldgenSource::new(&layer, &lib, SEED);
        for lod in [0u32, 1, 3, 5, 7] {
            // A box around the surface at this LOD: find the surface brick-Y near the origin so the box straddles
            // it, then take a 12×(tall)×12 brick window.
            let span = brick_span(lod) as f64;
            let h0 = layer.sample_world(0.0, 0.0, SEED).height as f64;
            let cy = (h0 / span).floor() as i32;
            let lo = IVec3::new(-6, cy - 8, -6);
            let hi = IVec3::new(5, cy + 8, 5);
            // The emitted candidate set.
            let mut emitted: Vec<IVec3> = Vec::new();
            src.surface_bricks_in(lo, hi, lod, &mut emitted);
            let emitted: rustc_hash::FxHashSet<IVec3> = emitted.into_iter().collect();
            // The ground-truth Surface set: classify EVERY brick in the box.
            let mut classify_surface: rustc_hash::FxHashSet<IVec3> = rustc_hash::FxHashSet::default();
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    for x in lo.x..=hi.x {
                        let c = IVec3::new(x, y, z);
                        if matches!(src.classify(c, lod), BrickClass::Surface) {
                            classify_surface.insert(c);
                        }
                    }
                }
            }
            // EXACT equality — no Air/Interior emitted (exactness ⇒ skip-classify is sound) and no Surface missed
            // (no render hole). Diagnose the first divergence on either side.
            for c in &emitted {
                assert!(
                    classify_surface.contains(c),
                    "LOD{lod}: surface_bricks_in emitted {c:?} but classify says it is NOT Surface (exactness violated)"
                );
            }
            for c in &classify_surface {
                assert!(
                    emitted.contains(c),
                    "LOD{lod}: classify says {c:?} is Surface but surface_bricks_in did NOT emit it (a render hole)"
                );
            }
            assert_eq!(emitted, classify_surface, "LOD{lod}: surface_bricks_in must EQUAL classify's Surface set");
            assert!(!emitted.is_empty(), "LOD{lod}: the box straddles the surface, so the set is non-empty");
        }
    }

    /// STATIC enclosed-cull: a fully-buried brick (full + all 6 face-neighbours full) classifies `Interior`
    /// (prunable — never the first hit); a face/corner brick (an outward neighbour absent) or a partial brick
    /// classifies `Surface` (conservative, hole-free); an absent brick is `Air`.
    #[test]
    fn static_classify_prunes_only_fully_buried_bricks() {
        // A 3×3×3 block of fully-solid bricks at brick coords [0,3)³.
        let mut map = BrickMap::new();
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    map.insert(IVec3::new(x, y, z), Brick::uniform(BlockId(1)));
                }
            }
        }
        let src = StaticVoxSource::new(&map);
        // CENTRE (1,1,1): full + all 6 face-neighbours full ⇒ Interior (buried, occluded — never the first hit).
        assert!(matches!(src.classify(IVec3::new(1, 1, 1), 0), BrickClass::Interior));
        // FACE-centre (1,1,0): the −Z neighbour (1,1,−1) is absent ⇒ exposed face ⇒ Surface.
        assert!(matches!(src.classify(IVec3::new(1, 1, 0), 0), BrickClass::Surface));
        // CORNER (0,0,0): three outward neighbours absent ⇒ Surface.
        assert!(matches!(src.classify(IVec3::new(0, 0, 0), 0), BrickClass::Surface));
        // OUTSIDE the block ⇒ Air (absent in the map).
        assert!(matches!(src.classify(IVec3::new(9, 9, 9), 0), BrickClass::Air));

        // A PARTIAL centre brick (one air voxel) surrounded by full ⇒ Surface (it has an exposed internal face).
        let mut map2 = BrickMap::new();
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    map2.insert(IVec3::new(x, y, z), Brick::uniform(BlockId(1)));
                }
            }
        }
        let mut arr = Box::new([BlockId(1); BRICK_VOXELS]);
        arr[voxel_index(0, 0, 0)] = BlockId::AIR; // one air voxel ⇒ not full
        map2.insert(IVec3::new(1, 1, 1), Brick::from_voxels(arr));
        let src2 = StaticVoxSource::new(&map2);
        assert!(matches!(src2.classify(IVec3::new(1, 1, 1), 0), BrickClass::Surface));
    }
}
