//! A sparse brick store of voxels.
//!
//! A [`Brick`] is an `8³` block of voxels (its world span scales with LOD — see below). Bricks are
//! keyed by their integer BRICK coordinate in an [`FxHashMap`]; an absent key is fully-empty (all air)
//! space, so the store stays sparse — empty regions cost nothing.
//!
//! ## Clipmap LOD: a brick is ALWAYS `8³` voxels; only its WORLD SPAN scales with LOD
//! A LOD-`L` brick is `8³` voxels of edge [`lod_voxel_size`]`(L) = VOXEL_SIZE · 2^L`, so it spans
//! [`brick_span`]`(L) = BRICK_WORLD_SIZE · 2^L` metres. LOD0 = a `1.6 m` brick of `0.2 m` voxels; LOD1 = a
//! `3.2 m` brick of `0.4 m` voxels; … This is a true nested CLIPMAP (geometry-clipmaps / GigaVoxels 3D
//! mipmap): coarser levels cover MORE world at the SAME `8³` resolution, so view distance grows with `2^L`
//! at bounded VRAM. DIFFERENT LODs are DIFFERENT coord grids — the same integer coord at two LODs is two
//! different world bricks (`world_min(c, L) = c · brick_span(L)`). The voxelizer samples each LOD brick
//! DIRECTLY at its own (coarse) spacing — a true in-place mip, not a downsample of a finer brick.
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
/// World-metre edge of a LOD0 brick (`BRICK_EDGE · VOXEL_SIZE` = 1.6 m). This is the FINEST brick span;
/// a LOD-`L` brick spans [`brick_span`]`(L) = BRICK_WORLD_SIZE · 2^L`.
pub const BRICK_WORLD_SIZE: f32 = BRICK_EDGE as f32 * VOXEL_SIZE;

/// The maximum LOD level a brick can be stored at. A brick is ALWAYS [`BRICK_EDGE`]³ voxels at every LOD;
/// only its world SPAN scales (`brick_span(L) = BRICK_WORLD_SIZE · 2^L`). MAX_LOD = 7 ⇒ the coarsest level
/// covers `2^7 = 128×` the LOD0 span per axis, so a clipmap of `MAX_LOD+1 = 8` nested shells reaches a view
/// radius of `clip_half · BRICK_WORLD_SIZE · 2^7` — e.g. `8 · 1.6 · 128 ≈ 1640 m` half-extent at
/// `clip_half_bricks = 8` (~36× the old 45 m dense reach). The coarsest voxel is `0.2 · 2^7 = 25.6 m`,
/// comfortably sub-pixel at >1 km, so no detail is wasted; each level stays a thin `(2·clip_half+1)³ − inner`
/// shell and the coarse shells are sparse, so the extra level costs few bricks. **If you change this, change
/// the WGSL mirror `MAX_LOD` in `voxel_raytrace.wgsl` too** (the per-LOD `brick_span`/clamp must agree). The
/// SSOT cap shared by streaming, packing, the shader, and the tests; LOD selection clamps here.
pub const MAX_LOD: u32 = 7;

/// The voxel-grid EDGE of a brick at LOD `lod`: a CONSTANT [`BRICK_EDGE`] (8) at EVERY LOD. The clipmap
/// keeps resolution fixed and scales the world SPAN instead (see [`brick_span`] / [`lod_voxel_size`]), so a
/// coarse brick is the same `8³` grid covering more world — the AABB is NO LONGER LOD-invariant. Kept as a
/// function (not a bare const) so every call site stays the single SSOT for LOD→resolution if the policy
/// ever changes, and the WGSL DDA mirrors it exactly.
#[inline]
pub fn lod_edge(_lod: u32) -> i32 {
    BRICK_EDGE
}

/// The world-metre size of ONE voxel cell in a brick at LOD `lod`: `VOXEL_SIZE · 2^lod`. A coarse brick's
/// cells are larger (it spans `2^lod×` more world at the same `8³` resolution), so its DDA crosses fewer,
/// wider boundaries. SSOT for the per-LOD cell size, shared by the voxelizer, the GPU packing, and the DDA.
#[inline]
pub fn lod_voxel_size(lod: u32) -> f32 {
    VOXEL_SIZE * (1u32 << lod.min(MAX_LOD)) as f32
}

/// The world-metre SPAN of a brick at LOD `lod`: `BRICK_WORLD_SIZE · 2^lod` (= [`BRICK_EDGE`] ·
/// [`lod_voxel_size`]`(lod)`). The fundamental clipmap quantity — a LOD-`lod` brick at integer coord `c`
/// spans world `[c · brick_span(lod), (c+1) · brick_span(lod))` per axis, so `world_min = c · brick_span`.
/// The single SSOT for the per-LOD world span, shared by the Rust packers ([`super::gpu::brick_aabb`]) and
/// the WGSL DDA (`brick_span` in `voxel_raytrace.wgsl`), so they can never disagree on a brick's extent.
#[inline]
pub fn brick_span(lod: u32) -> f32 {
    BRICK_WORLD_SIZE * (1u32 << lod.min(MAX_LOD)) as f32
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

    /// The single block id of a UNIFORM brick (every voxel the same block), or `None` for a dense brick.
    /// Returns `Some(AIR)` only for the degenerate uniform-air brick (which is never stored in the map). The
    /// GPU packer ([`super::gpu::pack_resident_set`]) uses this to collapse a fully-buried uniform brick whose
    /// HALO also matches into a flag + block id in the meta — no per-voxel array in VRAM (storage plan R1).
    #[inline]
    pub fn uniform_block(&self) -> Option<BlockId> {
        match self.storage {
            BrickStorage::Uniform(b) => Some(b),
            BrickStorage::Dense(_) => None,
        }
    }
}

/// A sparse store of [`Brick`]s keyed by integer BRICK coordinate on the LOD0 grid (world brick =
/// brick_coord · `BRICK_EDGE` voxels = brick_coord · `BRICK_WORLD_SIZE` metres). Absent keys are
/// fully-empty space. Used for the static, all-LOD0 Cornell scene + cross-brick neighbour queries; the
/// streamed worldgen set is keyed by `(coord, lod)` in [`super::streaming`], not stored here.
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

    /// The clipmap SSOT: a brick is ALWAYS `8³` voxels at every LOD ([`lod_edge`] constant), the voxel cell
    /// size doubles per LOD ([`lod_voxel_size`] = `VOXEL_SIZE · 2^lod`), and the brick world span doubles
    /// likewise ([`brick_span`] = `BRICK_WORLD_SIZE · 2^lod` = `BRICK_EDGE · lod_voxel_size`). So a LOD-`L`
    /// brick covers `2^L×` more world per axis at the SAME resolution — the nested-clipmap invariant the
    /// packers + the WGSL DDA share. Clamps at [`MAX_LOD`].
    #[test]
    fn clipmap_span_scales_with_lod() {
        for lod in 0..=MAX_LOD {
            assert_eq!(lod_edge(lod), BRICK_EDGE, "resolution is fixed 8³ at every LOD");
            let scale = (1u32 << lod) as f32;
            assert!((lod_voxel_size(lod) - VOXEL_SIZE * scale).abs() < 1e-6, "voxel size doubles per LOD");
            assert!((brick_span(lod) - BRICK_WORLD_SIZE * scale).abs() < 1e-4, "brick span doubles per LOD");
            // The fundamental identity: span = edge · cell size (the DDA walks 8 cells of lod_voxel_size).
            assert!((brick_span(lod) - BRICK_EDGE as f32 * lod_voxel_size(lod)).abs() < 1e-4);
        }
        // Clamps past MAX_LOD (never sub-voxel / never blows up).
        assert_eq!(brick_span(MAX_LOD + 5), brick_span(MAX_LOD));
        assert_eq!(lod_voxel_size(MAX_LOD + 5), lod_voxel_size(MAX_LOD));
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
