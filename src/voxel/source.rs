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
//! * [`StaticVoxSource`] wraps a loaded fine [`BrickMap`] (the `0.2 m` Sponza voxels) and produces the same
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

use super::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, BrickMap, MAX_LOD, voxel_index};
use super::palette::{BlockId, BlockRegistry};
use super::voxelize::voxelize_brick;
use crate::sdf_render::worldgen::biome::BiomeLibrary;
use crate::sdf_render::worldgen::layers::height::HeightLayer;

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
}

/// The STATIC `.vox` brick source: produces a `(coord, lod)` brick from a loaded fine [`BrickMap`] (the
/// `0.2 m` baked Sponza voxels, all at the LOD0 grid). Reproduces what [`super::voxelize::voxelize_brick`]
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
}

/// Downsample one pyramid level into the next-coarser one by `2³`: each coarse voxel aggregates its `2×2×2`
/// finer children — SOLID iff ANY child is solid (the occupancy/visibility invariant), and its block is the
/// DOMINANT (most-frequent) child solid block with a LOWEST-[`BlockId`] tie-break so the result is
/// DETERMINISTIC (independent of iteration order / thread). Iterates the SOURCE bricks only (sparse), writing
/// each into its coarse brick — so the cost is `O(source voxels)`, and over the whole pyramid the work is
/// geometric (`Σ 1/8^k`), bounded by `8/7 ×` the fine map. The coarse brick `c` covers source bricks
/// `[2c, 2c+2)` per axis; coarse voxel `v` (in `[0,8)`) aggregates source voxels `[2v, 2v+2)`.
fn downsample_brickmap(fine: &BrickMap) -> BrickMap {
    // Accumulate per coarse brick: the 512 coarse voxels' (count-map, best) tallies. Keyed by coarse coord.
    let mut coarse_voxels: FxHashMap<IVec3, Box<[BlockId; BRICK_VOXELS]>> = FxHashMap::default();
    // For each coarse voxel we need the dominant child block; tally children then resolve. To stay sparse +
    // cheap we walk the FINE bricks and, for each solid fine voxel, bump the count for its coarse voxel.
    let mut counts: FxHashMap<(IVec3, usize), FxHashMap<u16, u32>> = FxHashMap::default();
    for (fc, brick) in fine.iter() {
        // The coarse brick this fine brick contributes to: each axis halves (Euclidean for negatives), and the
        // fine brick's 8 voxels map to coarse voxels [fc&1 ? 4 : 0 .. +4) along that axis.
        let cc = IVec3::new(fc.x.div_euclid(2), fc.y.div_euclid(2), fc.z.div_euclid(2));
        // The fine brick's offset (0 or 1) within its coarse brick, ×4 = the coarse-voxel base it writes into.
        let base = IVec3::new(fc.x.rem_euclid(2), fc.y.rem_euclid(2), fc.z.rem_euclid(2)) * (BRICK_EDGE / 2);
        for fz in 0..BRICK_EDGE {
            for fy in 0..BRICK_EDGE {
                for fx in 0..BRICK_EDGE {
                    let b = brick.get(fx, fy, fz);
                    if b.is_air() {
                        continue;
                    }
                    // The coarse voxel this fine voxel falls in: base + fine/2.
                    let cv = base + IVec3::new(fx / 2, fy / 2, fz / 2);
                    let ci = voxel_index(cv.x, cv.y, cv.z);
                    *counts.entry((cc, ci)).or_default().entry(b.0).or_insert(0) += 1;
                }
            }
        }
    }
    // Resolve each coarse voxel's dominant block (lowest-id tie-break) into its coarse brick array.
    for ((cc, ci), tally) in &counts {
        let block = dominant_block(tally);
        let arr = coarse_voxels.entry(*cc).or_insert_with(|| Box::new([BlockId::AIR; BRICK_VOXELS]));
        arr[*ci] = block;
    }
    let mut out = BrickMap::new();
    for (cc, arr) in coarse_voxels {
        out.insert(cc, Brick::from_voxels(arr));
    }
    out
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
        // Brick 0's +X halo ring (halo index hx == BRICK_EDGE+1 = 9) must read brick 1's core (block 2).
        let off0 = patch.metas[0].voxel_offset as usize;
        for hz in 1..=BRICK_EDGE {
            for hy in 1..=BRICK_EDGE {
                let cell = patch.voxels[off0 + halo_index(BRICK_EDGE + 1, hy, hz, 0)];
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
        // A Sponza-scale fine map: a ~120 m × ~50 m × ~75 m solid-ish slab in 0.2 m voxels ⇒ ~600×250×370
        // world voxels. Build a coarse but dense-enough box (a floor slab + walls) so the coarse footprints
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
}
