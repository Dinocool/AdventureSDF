//! A sparse brick store of voxels.
//!
//! A [`Brick`] is an `8³` block of voxels (each `VOXEL_SIZE` = 0.2 m → a `1.6 m` brick). Bricks are
//! keyed by their integer BRICK coordinate in an [`FxHashMap`]; an absent key is fully-empty (all air)
//! space, so the store stays sparse — empty regions cost nothing.
//!
//! A brick carries an OCCUPANCY bitmask (one bit per voxel: solid vs air) for fast neighbour/exposure
//! queries, plus a per-voxel [`BlockId`]. A UNIFORM fast path (`Brick::uniform`) represents a brick whose
//! voxels are all the same block with no per-voxel allocation — the common case for fully-buried interior
//! bricks. Empty bricks are simply absent from the map (never stored).

use bevy::math::IVec3;
use rustc_hash::FxHashMap;

use super::palette::BlockId;

/// Voxels per brick edge. `8³ = 512` voxels per brick.
pub const BRICK_EDGE: i32 = 8;
/// Voxels per brick (`BRICK_EDGE³`).
pub const BRICK_VOXELS: usize = (BRICK_EDGE * BRICK_EDGE * BRICK_EDGE) as usize;
/// Edge length of one voxel, in world metres.
pub const VOXEL_SIZE: f32 = 0.2;
/// World-metre edge of a brick (`BRICK_EDGE · VOXEL_SIZE` = 1.6 m).
pub const BRICK_WORLD_SIZE: f32 = BRICK_EDGE as f32 * VOXEL_SIZE;

/// The maximum LOD level a brick can be stored at. `BRICK_EDGE = 8 = 2³`, so LOD0 = 8³ (full res),
/// LOD1 = 4³, LOD2 = 2³, LOD3 = 1³ (a single voxel). Beyond `MAX_LOD` the brick would be sub-voxel, so
/// LOD selection clamps here — the SSOT cap shared by streaming, packing, the shader, and the tests.
pub const MAX_LOD: u32 = 3;

/// The voxel-grid EDGE of a brick stored at LOD `lod`: `BRICK_EDGE >> lod` (8,4,2,1 for lod 0..=3). The
/// brick's world AABB is unchanged across LODs — only the grid resolution (and therefore the DDA stride)
/// changes. Clamped at [`MAX_LOD`] so the edge never drops below 1. The single SSOT for LOD→resolution,
/// shared by the downsampler, the GPU packing, and the WGSL DDA.
#[inline]
pub fn lod_edge(lod: u32) -> i32 {
    BRICK_EDGE >> lod.min(MAX_LOD)
}

/// The world-metre size of ONE voxel cell in a brick stored at LOD `lod`: `VOXEL_SIZE << lod`. A coarse
/// brick's cells are larger (the brick spans the same world AABB with fewer, bigger cells), so its DDA
/// crosses fewer boundaries. SSOT for the per-LOD cell size.
#[inline]
pub fn lod_voxel_size(lod: u32) -> f32 {
    VOXEL_SIZE * (1i32 << lod.min(MAX_LOD)) as f32
}

/// Linear voxel index for a local `(x, y, z)` in `[0, BRICK_EDGE)` — +X fastest, then +Y, then +Z. The
/// single SSOT for the brick voxel layout, shared by storage and the occupancy bitmask.
#[inline]
pub fn voxel_index(x: i32, y: i32, z: i32) -> usize {
    debug_assert!((0..BRICK_EDGE).contains(&x) && (0..BRICK_EDGE).contains(&y) && (0..BRICK_EDGE).contains(&z));
    (x + y * BRICK_EDGE + z * BRICK_EDGE * BRICK_EDGE) as usize
}

/// The per-voxel storage of a brick: either a UNIFORM block (all voxels identical — no per-voxel alloc)
/// or a dense `BRICK_VOXELS`-long array of block ids. Empty bricks are never stored (absent from the map),
/// so a `Uniform(AIR)` brick should not normally exist in the store; the variant is kept total for
/// in-place edits that may clear a brick.
#[derive(Clone, Debug, PartialEq)]
enum BrickStorage {
    /// Every voxel is this block. The fast path for fully-buried interior bricks.
    Uniform(BlockId),
    /// Per-voxel block ids, indexed by [`voxel_index`].
    Dense(Box<[BlockId; BRICK_VOXELS]>),
}

/// One `8³` brick of voxels: a per-voxel block store (uniform or dense) plus a cached occupancy bitmask
/// (`solid` bit per voxel) for fast exposure/neighbour queries.
#[derive(Clone, Debug, PartialEq)]
pub struct Brick {
    storage: BrickStorage,
    /// Occupancy: bit `voxel_index(x,y,z)` set ⇒ that voxel is SOLID (non-air). `512` bits = `8 × u64`.
    occupancy: [u64; BRICK_VOXELS / 64],
}

impl Brick {
    /// A uniform brick — all `BRICK_VOXELS` voxels are `block`. No per-voxel allocation. Occupancy is all
    /// set when `block` is solid, all clear when it is air (`Brick::is_empty` then true).
    pub fn uniform(block: BlockId) -> Self {
        let occ = if block.is_air() { 0u64 } else { u64::MAX };
        Self { storage: BrickStorage::Uniform(block), occupancy: [occ; BRICK_VOXELS / 64] }
    }

    /// Build a brick from a per-voxel block array. If every voxel is identical it COLLAPSES to the uniform
    /// fast path (no dense allocation retained); otherwise it stores the dense array. The occupancy bitmask
    /// is derived from the voxels (solid = non-air). The single constructor the voxelizer uses.
    pub fn from_voxels(voxels: Box<[BlockId; BRICK_VOXELS]>) -> Self {
        let first = voxels[0];
        let uniform = voxels.iter().all(|&b| b == first);
        let mut occupancy = [0u64; BRICK_VOXELS / 64];
        for (i, &b) in voxels.iter().enumerate() {
            if !b.is_air() {
                occupancy[i / 64] |= 1u64 << (i % 64);
            }
        }
        let storage = if uniform { BrickStorage::Uniform(first) } else { BrickStorage::Dense(voxels) };
        Self { storage, occupancy }
    }

    /// The block at local `(x, y, z)` in `[0, BRICK_EDGE)`.
    #[inline]
    pub fn get(&self, x: i32, y: i32, z: i32) -> BlockId {
        match &self.storage {
            BrickStorage::Uniform(b) => *b,
            BrickStorage::Dense(v) => v[voxel_index(x, y, z)],
        }
    }

    /// True iff local `(x, y, z)` is SOLID (non-air), via the occupancy bitmask (no storage deref).
    #[inline]
    pub fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        let i = voxel_index(x, y, z);
        (self.occupancy[i / 64] >> (i % 64)) & 1 == 1
    }

    /// True iff the brick is entirely air (no solid voxel). Such bricks are never stored in [`BrickMap`].
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.occupancy.iter().all(|&w| w == 0)
    }

    /// True iff every voxel is the SAME solid block (the uniform fast path with a non-air block) — a
    /// fully-buried interior brick. Used to skip exposed-surface meshing of solid interiors.
    #[inline]
    pub fn is_uniform_solid(&self) -> bool {
        matches!(self.storage, BrickStorage::Uniform(b) if !b.is_air())
    }

    /// Downsample this `8³` brick to its LOD-`lod` grid (`lod_edge(lod)³` cells; LOD0 returns the full
    /// `8³` block ids unchanged). Each coarse cell aggregates the `2^lod`-cubed block of fine voxels it
    /// covers via [`downsample_cell`]: air loses to solid, the most-common non-air [`BlockId`] wins, and a
    /// cell becomes solid iff at least `solid_keep_k` of its children are solid (the thin-feature rule).
    ///
    /// Returns a `Vec<BlockId>` of length `lod_edge(lod)³` in [`voxel_index`]-style order at the COARSE
    /// edge (+X fastest, then +Y, then +Z), so the same linear-index convention applies at every LOD — the
    /// shader reconstructs the index identically with the coarse edge. The single SSOT for mip generation.
    pub fn downsample(&self, lod: u32, solid_keep_k: u32) -> Vec<BlockId> {
        let lod = lod.min(MAX_LOD);
        let cedge = lod_edge(lod);
        if lod == 0 {
            // Full resolution — copy the fine grid out in voxel_index order.
            let mut out = Vec::with_capacity(BRICK_VOXELS);
            for z in 0..BRICK_EDGE {
                for y in 0..BRICK_EDGE {
                    for x in 0..BRICK_EDGE {
                        out.push(self.get(x, y, z));
                    }
                }
            }
            return out;
        }
        let factor = 1i32 << lod; // fine voxels per coarse cell per axis (2, 4, 8)
        let mut out = Vec::with_capacity((cedge * cedge * cedge) as usize);
        for cz in 0..cedge {
            for cy in 0..cedge {
                for cx in 0..cedge {
                    out.push(self.downsample_cell(cx, cy, cz, factor, solid_keep_k));
                }
            }
        }
        out
    }

    /// Aggregate the `factor³` fine voxels under coarse cell `(cx,cy,cz)` into one [`BlockId`].
    ///
    /// Rule (documented thin-feature handling): we tally each NON-AIR child block id. If the number of
    /// solid children is `>= solid_keep_k` the cell is SOLID and takes the most-common solid block id (ties
    /// broken by the smallest id for determinism); otherwise the cell is AIR. Setting `solid_keep_k = 1`
    /// gives "keep solid if ANY child is solid" — the conservative rule used for the NEAR rings so a thin
    /// one-voxel surface survives a downsample instead of vanishing. Larger `k` (e.g. half the children)
    /// erodes thin features but matches majority occupancy — used for far rings where erosion is invisible.
    fn downsample_cell(&self, cx: i32, cy: i32, cz: i32, factor: i32, solid_keep_k: u32) -> BlockId {
        // Small dense tally: solid block ids in a brick are few, so a linear (id,count) scan is cheap and
        // avoids a hashmap allocation per cell.
        let mut counts: Vec<(BlockId, u32)> = Vec::new();
        let mut solid_children = 0u32;
        let (ox, oy, oz) = (cx * factor, cy * factor, cz * factor);
        for dz in 0..factor {
            for dy in 0..factor {
                for dx in 0..factor {
                    let b = self.get(ox + dx, oy + dy, oz + dz);
                    if b.is_air() {
                        continue;
                    }
                    solid_children += 1;
                    match counts.iter_mut().find(|(id, _)| *id == b) {
                        Some((_, c)) => *c += 1,
                        None => counts.push((b, 1)),
                    }
                }
            }
        }
        if solid_children < solid_keep_k.max(1) {
            return BlockId::AIR;
        }
        // Most-common solid block; ties → smallest id (deterministic).
        counts
            .into_iter()
            .max_by(|(ida, ca), (idb, cb)| ca.cmp(cb).then(idb.cmp(ida)))
            .map(|(id, _)| id)
            .unwrap_or(BlockId::AIR)
    }
}

/// A sparse store of [`Brick`]s keyed by integer BRICK coordinate (world brick = brick_coord ·
/// `BRICK_EDGE` voxels = brick_coord · `BRICK_WORLD_SIZE` metres). Absent keys are fully-empty space.
#[derive(Default, Debug)]
pub struct BrickMap {
    bricks: FxHashMap<IVec3, Brick>,
}

impl BrickMap {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `brick` at `coord`, UNLESS it is entirely air — empty bricks are never stored (the sparsity
    /// invariant). Returns `true` if a brick was stored. Replaces any existing brick at `coord`.
    pub fn insert(&mut self, coord: IVec3, brick: Brick) -> bool {
        if brick.is_empty() {
            self.bricks.remove(&coord);
            false
        } else {
            self.bricks.insert(coord, brick);
            true
        }
    }

    /// The brick at `coord`, or `None` if that brick is empty/absent.
    #[inline]
    pub fn get(&self, coord: IVec3) -> Option<&Brick> {
        self.bricks.get(&coord)
    }

    /// The block at a WORLD voxel coordinate (in voxel units, not metres). Resolves the owning brick + the
    /// local voxel; an absent brick is AIR. The SSOT for cross-brick neighbour queries (exposure tests).
    #[inline]
    pub fn voxel_block(&self, world_voxel: IVec3) -> BlockId {
        let bc = brick_coord_of_voxel(world_voxel);
        match self.bricks.get(&bc) {
            Some(brick) => {
                let local = world_voxel - bc * BRICK_EDGE;
                brick.get(local.x, local.y, local.z)
            }
            None => BlockId::AIR,
        }
    }

    /// True iff the WORLD voxel is solid (via [`voxel_block`](Self::voxel_block)).
    #[inline]
    pub fn voxel_is_solid(&self, world_voxel: IVec3) -> bool {
        !self.voxel_block(world_voxel).is_air()
    }

    /// Number of stored (non-empty) bricks.
    #[inline]
    pub fn len(&self) -> usize {
        self.bricks.len()
    }

    /// True iff no bricks are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bricks.is_empty()
    }

    /// Iterate `(brick_coord, brick)` over every stored brick.
    pub fn iter(&self) -> impl Iterator<Item = (&IVec3, &Brick)> {
        self.bricks.iter()
    }
}

/// The BRICK coordinate owning a WORLD voxel coordinate (Euclidean floor-division by `BRICK_EDGE`, so it
/// is correct for negative coordinates). The SSOT for voxel→brick addressing.
#[inline]
pub fn brick_coord_of_voxel(world_voxel: IVec3) -> IVec3 {
    IVec3::new(
        world_voxel.x.div_euclid(BRICK_EDGE),
        world_voxel.y.div_euclid(BRICK_EDGE),
        world_voxel.z.div_euclid(BRICK_EDGE),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(n: u16) -> BlockId {
        BlockId(n)
    }

    /// A uniform solid brick: every voxel reads the block, occupancy is full, not empty, and it reports
    /// as a uniform solid (the fully-buried fast path).
    #[test]
    fn uniform_solid_fast_path() {
        let b = Brick::uniform(solid(3));
        assert!(!b.is_empty());
        assert!(b.is_uniform_solid());
        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                for x in 0..BRICK_EDGE {
                    assert_eq!(b.get(x, y, z), solid(3));
                    assert!(b.is_solid(x, y, z));
                }
            }
        }
    }

    /// A uniform AIR brick is empty (occupancy all clear) and never stored by the map.
    #[test]
    fn uniform_air_is_empty_and_unstored() {
        let b = Brick::uniform(BlockId::AIR);
        assert!(b.is_empty());
        assert!(!b.is_uniform_solid());
        let mut map = BrickMap::new();
        assert!(!map.insert(IVec3::ZERO, b));
        assert!(map.get(IVec3::ZERO).is_none());
        assert!(map.is_empty());
    }

    /// `from_voxels` collapses an all-identical array to the uniform fast path (bit-identical to
    /// `Brick::uniform`), and a mixed array stays dense with the right per-voxel reads + occupancy.
    #[test]
    fn from_voxels_collapses_uniform() {
        let all_same = Box::new([solid(5); BRICK_VOXELS]);
        let b = Brick::from_voxels(all_same);
        assert_eq!(b, Brick::uniform(solid(5)));
        assert!(b.is_uniform_solid());

        // Mixed: one air voxel embedded in solids → dense, not uniform, occupancy reflects the hole.
        let mut voxels = Box::new([solid(2); BRICK_VOXELS]);
        voxels[voxel_index(1, 2, 3)] = BlockId::AIR;
        let b = Brick::from_voxels(voxels);
        assert!(!b.is_uniform_solid());
        assert!(!b.is_empty());
        assert_eq!(b.get(1, 2, 3), BlockId::AIR);
        assert!(!b.is_solid(1, 2, 3));
        assert!(b.is_solid(0, 0, 0));
    }

    /// A coarse linear index at the LOD edge (+X fastest), mirroring the shader's reconstruction.
    fn lindex(x: i32, y: i32, z: i32, edge: i32) -> usize {
        (x + y * edge + z * edge * edge) as usize
    }

    /// LOD0 downsample is the identity: it returns the brick's `8³` block ids unchanged in voxel_index
    /// order.
    #[test]
    fn downsample_lod0_is_identity() {
        let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
        voxels[voxel_index(1, 2, 3)] = solid(7);
        voxels[voxel_index(4, 5, 6)] = solid(9);
        let b = Brick::from_voxels(voxels);
        let lod0 = b.downsample(0, 1);
        assert_eq!(lod0.len(), BRICK_VOXELS);
        assert_eq!(lod0[voxel_index(1, 2, 3)], solid(7));
        assert_eq!(lod0[voxel_index(4, 5, 6)], solid(9));
        assert_eq!(lod0[voxel_index(0, 0, 0)], BlockId::AIR);
    }

    /// Majority block wins per coarse cell: fill the lower-X half of the brick with block A and the upper
    /// half with block B; at LOD1 (4³, factor 2) every coarse cell is fully one block, so the coarse grid
    /// is a clean A/B split.
    #[test]
    fn downsample_majority_block() {
        let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                for x in 0..BRICK_EDGE {
                    voxels[voxel_index(x, y, z)] = if x < 4 { solid(1) } else { solid(2) };
                }
            }
        }
        let b = Brick::from_voxels(voxels);
        let lod1 = b.downsample(1, 1);
        assert_eq!(lod1.len(), 4 * 4 * 4);
        // Coarse cells cx∈{0,1} (cover fine x 0..4) are block 1; cx∈{2,3} (fine x 4..8) are block 2.
        assert_eq!(lod1[lindex(0, 0, 0, 4)], solid(1));
        assert_eq!(lod1[lindex(1, 1, 1, 4)], solid(1));
        assert_eq!(lod1[lindex(2, 0, 0, 4)], solid(2));
        assert_eq!(lod1[lindex(3, 3, 3, 4)], solid(2));
    }

    /// Mixed-count cell takes the most-common solid id (5 of block 3 vs 3 of block 4 in an 8-child cell).
    #[test]
    fn downsample_picks_most_common_solid() {
        let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
        // Coarse cell (0,0,0) at LOD1 covers fine voxels x,y,z ∈ {0,1}: 8 children. Make 5 block-3, 3 block-4.
        let cells = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1), // block 3 ×5
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1), // block 4 ×3
        ];
        for (i, &(x, y, z)) in cells.iter().enumerate() {
            voxels[voxel_index(x, y, z)] = if i < 5 { solid(3) } else { solid(4) };
        }
        let b = Brick::from_voxels(voxels);
        let lod1 = b.downsample(1, 1);
        assert_eq!(lod1[lindex(0, 0, 0, 4)], solid(3), "majority solid block (5 vs 3) wins");
    }

    /// Thin-feature rule: a single solid voxel in an otherwise-air coarse cell SURVIVES with the
    /// conservative `solid_keep_k = 1` ("any child solid") but is ERODED to air with a higher threshold —
    /// the documented near-vs-far behaviour.
    #[test]
    fn downsample_thin_feature_threshold() {
        let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
        voxels[voxel_index(0, 0, 0)] = solid(5); // one solid child of coarse cell (0,0,0) at LOD1
        let b = Brick::from_voxels(voxels);

        // Conservative (k=1): the thin voxel survives the downsample.
        let keep = b.downsample(1, 1);
        assert_eq!(keep[lindex(0, 0, 0, 4)], solid(5), "k=1 keeps a thin solid voxel");
        // Aggressive (k=5, majority of 8): the lone voxel erodes to air.
        let erode = b.downsample(1, 5);
        assert_eq!(erode[lindex(0, 0, 0, 4)], BlockId::AIR, "k=5 erodes a 1/8 sliver");
    }

    /// A thin SURFACE (a one-voxel-thick solid slab) survives one downsample as a solid (thinner) slab with
    /// `k=1`: every coarse cell straddling the slab has at least one solid child, so the coarse slab is
    /// continuous (no holes) — it thins but does not vanish.
    #[test]
    fn downsample_thin_surface_survives() {
        let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
        // A horizontal slab one voxel thick at y=3 across all x,z.
        for z in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                voxels[voxel_index(x, 3, z)] = solid(6);
            }
        }
        let b = Brick::from_voxels(voxels);
        let lod1 = b.downsample(1, 1);
        // Coarse cells with cy=1 cover fine y∈{2,3}; the slab at y=3 makes every such cell solid.
        for cz in 0..4 {
            for cx in 0..4 {
                assert_eq!(lod1[lindex(cx, 1, cz, 4)], solid(6), "slab survives continuously at coarse cy=1");
                // The neighbouring coarse layer (cy=0 covers y∈{0,1}) is air — the slab thinned, not spread.
                assert_eq!(lod1[lindex(cx, 0, cz, 4)], BlockId::AIR);
            }
        }
    }

    /// LOD2 collapses the whole brick to `2³` and LOD3 to a single cell; a uniform solid brick stays solid
    /// at every LOD (no spurious erosion of a full interior).
    #[test]
    fn downsample_deep_lods() {
        let b = Brick::uniform(solid(2));
        let lod2 = b.downsample(2, 1);
        assert_eq!(lod2.len(), 2 * 2 * 2);
        assert!(lod2.iter().all(|&v| v == solid(2)));
        let lod3 = b.downsample(3, 1);
        assert_eq!(lod3.len(), 1);
        assert_eq!(lod3[0], solid(2));
    }

    /// Cross-brick world-voxel addressing: a voxel in the next brick over reads that brick; negative
    /// coordinates floor-divide correctly; absent bricks read air.
    #[test]
    fn world_voxel_addressing() {
        let mut map = BrickMap::new();
        map.insert(IVec3::new(0, 0, 0), Brick::uniform(solid(1)));
        map.insert(IVec3::new(-1, 0, 0), Brick::uniform(solid(2)));

        // Brick (0,0,0) owns world voxels [0,8); a voxel inside it is solid(1).
        assert_eq!(map.voxel_block(IVec3::new(3, 4, 5)), solid(1));
        // World voxel -1 belongs to brick -1 (floor-div), which is solid(2).
        assert_eq!(map.voxel_block(IVec3::new(-1, 0, 0)), solid(2));
        assert_eq!(brick_coord_of_voxel(IVec3::new(-1, 0, 0)), IVec3::new(-1, 0, 0));
        assert_eq!(brick_coord_of_voxel(IVec3::new(-8, 0, 0)), IVec3::new(-1, 0, 0));
        assert_eq!(brick_coord_of_voxel(IVec3::new(-9, 0, 0)), IVec3::new(-2, 0, 0));
        // An absent brick reads air.
        assert_eq!(map.voxel_block(IVec3::new(100, 100, 100)), BlockId::AIR);
        assert!(!map.voxel_is_solid(IVec3::new(100, 100, 100)));
    }
}
