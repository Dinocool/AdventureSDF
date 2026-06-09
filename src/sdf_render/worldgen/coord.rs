//! Coordinate + identity types for the worldgen layer framework.
//!
//! Per WORLD_GEN_PLAN §2.9, the CPU (authoritative) side uses **f64 / integer** world coordinates —
//! exact everywhere, and integer lattice math is trivially bit-portable. Each layer has its own chunk
//! lattice (its *tier*, §2.7); a [`ChunkCoord`] is the absolute integer index on that lattice,
//! anchored at world 0 and **never** relative to the camera (that camera-relative-id mistake is what
//! caused the renderer's historic "world shifts / disappears" bugs — see `chunk.rs`).
//!
//! World→chunk mapping uses a plain `f64` floor (NOT integer `div_euclid` on a pre-divided coord),
//! matching the WGSL `floor()` the GPU sampler will use — the CPU/GPU float-floor parity the
//! light-grid lesson pins (WORLD_GEN_PLAN §10).

use bevy::math::{DVec2, DVec3, IVec3};

/// Stable identity of a layer within a recipe's stack (index into the `LayerManager`'s layer list).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct LayerId(pub u32);

/// Lattice dimensionality of a layer (WORLD_GEN_PLAN §2: lattice is per-layer). D2 layers ignore the
/// Y component of their [`ChunkCoord`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dim {
    /// 2D (XZ) lattice — continents, climate, the Phase-1 height layer.
    D2,
    /// 3D (XYZ) lattice — caves, sub-surface biomes (later phases).
    D3,
}

/// CPU authority class (WORLD_GEN_PLAN §2.8). `Authoritative` layers MUST be cross-platform
/// bit-deterministic (they drive gameplay/collision and clients must agree); `Cosmetic` layers may
/// use non-portable math (GPU detail).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Authority {
    Authoritative,
    Cosmetic,
}

/// A layer's chunk size = its tier. Stored as a whole number of base cells so the lattice is exact
/// (no float chunk size to drift). Bigger = higher abstraction (WORLD_GEN_PLAN §2.7).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ChunkSize {
    /// Chunk edge length in whole base cells. Power-of-two recommended so tiers nest cleanly.
    pub cells: u32,
}

impl ChunkSize {
    /// World metres per base cell. The recipe may scale the world later; for now 1 cell = 1 metre.
    pub const BASE_CELL_METRES: f64 = 1.0;

    pub const fn new(cells: u32) -> Self {
        Self { cells }
    }

    /// World-metre edge length of one chunk on this tier.
    #[inline]
    pub fn world_size(self) -> f64 {
        self.cells as f64 * Self::BASE_CELL_METRES
    }
}

/// Absolute integer chunk index on a layer's own lattice. XZ for [`Dim::D2`] (`xyz.y == 0`), XYZ for
/// [`Dim::D3`]. World-0 anchored, camera-independent — this is the determinism + toroidal-store key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ChunkCoord {
    pub layer: LayerId,
    pub xyz: IVec3,
}

impl ChunkCoord {
    pub fn new(layer: LayerId, xyz: IVec3) -> Self {
        Self { layer, xyz }
    }
}

// Total order for deterministic iteration (e.g. a `BTreeSet` dirty set, or sorted uploads): by layer,
// then lexicographically by (x, y, z). `IVec3` isn't `Ord`, so order on its components explicitly.
impl PartialOrd for ChunkCoord {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ChunkCoord {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.layer
            .cmp(&other.layer)
            .then(self.xyz.x.cmp(&other.xyz.x))
            .then(self.xyz.y.cmp(&other.xyz.y))
            .then(self.xyz.z.cmp(&other.xyz.z))
    }
}

/// The chunk on `layer`'s lattice that contains world position `world_xz` (2D layers). Uses a plain
/// `f64` floor to match the GPU sampler's `floor()` (CPU/GPU float-floor parity, §10).
#[inline]
pub fn chunk_of_world(layer: LayerId, size: ChunkSize, world_xz: DVec2) -> ChunkCoord {
    let s = size.world_size();
    ChunkCoord::new(
        layer,
        IVec3::new(
            (world_xz.x / s).floor() as i32,
            0,
            (world_xz.y / s).floor() as i32,
        ),
    )
}

/// World-space minimum corner (f64) of a chunk — its node-(0,0) origin.
#[inline]
pub fn chunk_min_world(c: ChunkCoord, size: ChunkSize) -> DVec3 {
    let s = size.world_size();
    DVec3::new(c.xyz.x as f64 * s, c.xyz.y as f64 * s, c.xyz.z as f64 * s)
}

/// World-space centre (f64) of a chunk — used for focus/range tests.
#[inline]
pub fn chunk_center_world(c: ChunkCoord, size: ChunkSize) -> DVec3 {
    let s = size.world_size();
    chunk_min_world(c, size) + DVec3::splat(s * 0.5)
}

/// Bias added to each signed chunk-axis index so it fits an unsigned 16-bit key field. ±32768
/// chunks/axis. Mirrors `chunk::KEY_BIAS`; pinned by a parity test so any GPU mirror can't drift.
pub const WG_KEY_BIAS: i32 = 1 << 15;

/// Order-preserving 64-bit GPU key for a worldgen chunk coord, packed lexicographically so sorting /
/// binary-searching by `(key_hi, key_lo)` orders by (x, y, z). Mirrors the `chunk::chunk_gpu_key`
/// scheme (minus the LOD field — worldgen tiers are separate layers, not LODs of one lattice). Used
/// by key-indexed GPU artifact stores; the slice's dense-ring path is origin-relative and doesn't
/// need it, but it's the extensibility hook for sparse 3D artifact directories.
#[inline]
pub fn chunk_gpu_key(xyz: IVec3) -> (u32, u32) {
    let cx = ((xyz.x + WG_KEY_BIAS) as u32) & 0xffff;
    let cy = ((xyz.y + WG_KEY_BIAS) as u32) & 0xffff;
    let cz = ((xyz.z + WG_KEY_BIAS) as u32) & 0xffff;
    let key_hi = (cx << 16) | cy;
    let key_lo = cz << 16;
    (key_hi, key_lo)
}

/// Inverse of [`chunk_gpu_key`]: decode a packed `(key_hi, key_lo)` back to its chunk XYZ coord. The
/// 16-bit fields are biased by `WG_KEY_BIAS`; sign-extend by subtracting the bias. Used by the height
/// ring's residency-bounds report (`upload::ring_resident_bounds`) to turn directory key-tags back
/// into chunk coords for the fail-loud sampler's diagnostics.
#[inline]
pub fn chunk_coord_from_gpu_key(key_hi: u32, key_lo: u32) -> IVec3 {
    let cx = ((key_hi >> 16) & 0xffff) as i32 - WG_KEY_BIAS;
    let cy = (key_hi & 0xffff) as i32 - WG_KEY_BIAS;
    let cz = ((key_lo >> 16) & 0xffff) as i32 - WG_KEY_BIAS;
    IVec3::new(cx, cy, cz)
}

#[cfg(test)]
mod tests {
    use super::*;

    const L: LayerId = LayerId(0);

    /// World→chunk→world-min round-trips: a point inside a chunk maps to that chunk, whose world box
    /// encloses the point on both axes (`min ≤ p < min + size`).
    #[test]
    fn world_chunk_roundtrip_contains_point() {
        let size = ChunkSize::new(256);
        let s = size.world_size();
        for &(px, pz) in &[(0.0, 0.0), (5.5, 9.1), (300.0, -10.0), (-260.0, -513.0), (1000.25, 777.75)] {
            let c = chunk_of_world(L, size, DVec2::new(px, pz));
            let min = chunk_min_world(c, size);
            assert!(px >= min.x && px < min.x + s, "x {px} not in chunk [{}, {})", min.x, min.x + s);
            assert!(pz >= min.z && pz < min.z + s, "z {pz} not in chunk [{}, {})", min.z, min.z + s);
        }
    }

    /// Negative coords use a floor (chunk below), not truncation toward zero — the float-floor trap.
    #[test]
    fn negative_world_floors_to_lower_chunk() {
        let size = ChunkSize::new(100);
        // x = -1 m with 100 m chunks → chunk -1 (floor(-0.01) = -1), NOT 0.
        let c = chunk_of_world(L, size, DVec2::new(-1.0, -0.0));
        assert_eq!(c.xyz.x, -1, "negative x must floor to chunk -1");
        // Exactly on a boundary belongs to the upper chunk.
        let c0 = chunk_of_world(L, size, DVec2::new(0.0, 0.0));
        assert_eq!(c0.xyz.x, 0);
        let cn = chunk_of_world(L, size, DVec2::new(-100.0, 0.0));
        assert_eq!(cn.xyz.x, -1, "x=-100 with 100m chunks is the top of chunk -1");
    }

    /// Adjacent chunks tile exactly (no gap/overlap) — the next chunk's min is one full size further.
    #[test]
    fn adjacent_chunks_tile_without_gaps() {
        let size = ChunkSize::new(64);
        let s = size.world_size();
        let base = ChunkCoord::new(L, IVec3::new(2, 0, -3));
        let min = chunk_min_world(base, size);
        let nx = chunk_min_world(ChunkCoord::new(L, base.xyz + IVec3::X), size);
        let nz = chunk_min_world(ChunkCoord::new(L, base.xyz + IVec3::Z), size);
        assert!((nx.x - (min.x + s)).abs() < 1e-9 && (nx.z - min.z).abs() < 1e-9);
        assert!((nz.z - (min.z + s)).abs() < 1e-9 && (nz.x - min.x).abs() < 1e-9);
    }

    /// The GPU key is order-preserving: sorting by `(key_hi, key_lo)` orders by (x, y, z) — the
    /// precondition for any binary-searched GPU artifact directory.
    #[test]
    fn gpu_key_is_order_preserving() {
        let mut coords = vec![
            IVec3::new(0, 0, 0),
            IVec3::new(0, 0, 1),
            IVec3::new(0, 1, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(-1, 0, 0),
            IVec3::new(-5, -5, -5),
            IVec3::new(3, -2, 7),
        ];
        let mut by_packed = coords.clone();
        by_packed.sort_by_key(|c| chunk_gpu_key(*c));
        coords.sort_by_key(|c| (c.x, c.y, c.z));
        assert_eq!(by_packed, coords);
    }

    /// Distinct coords in range never collide on the packed key.
    #[test]
    fn gpu_key_no_collision_in_range() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for x in -4..=4 {
            for y in -4..=4 {
                for z in -4..=4 {
                    assert!(seen.insert(chunk_gpu_key(IVec3::new(x, y, z))), "collision at ({x},{y},{z})");
                }
            }
        }
    }

    /// `ChunkCoord` total order is layer-major then lexicographic on (x, y, z) — needed for
    /// deterministic dirty-set iteration.
    #[test]
    fn chunk_coord_total_order() {
        let mut v = [
            ChunkCoord::new(LayerId(1), IVec3::new(0, 0, 0)),
            ChunkCoord::new(LayerId(0), IVec3::new(5, 0, 0)),
            ChunkCoord::new(LayerId(0), IVec3::new(0, 0, 9)),
            ChunkCoord::new(LayerId(0), IVec3::new(0, 0, 0)),
        ];
        v.sort();
        assert_eq!(v[0], ChunkCoord::new(LayerId(0), IVec3::new(0, 0, 0)));
        assert_eq!(v[1], ChunkCoord::new(LayerId(0), IVec3::new(0, 0, 9)));
        assert_eq!(v[2], ChunkCoord::new(LayerId(0), IVec3::new(5, 0, 0)));
        assert_eq!(v[3], ChunkCoord::new(LayerId(1), IVec3::new(0, 0, 0)));
    }
}
