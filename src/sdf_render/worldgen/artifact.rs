//! Artifacts — the typed, world-anchored outputs layers produce and consumers read.
//!
//! Every layer emits one or more artifacts (WORLD_GEN_PLAN §2.1). The Phase-1 slice needs only
//! [`ScalarField2D`] (a chunk-local height field), but the [`Artifact`] trait + [`ArtifactKind`] are
//! the generic seam so 3D fields, biome classification, instance streams, and vector graphs slot in
//! later with no framework change (each is a new `Artifact` impl + a `super::store::ArtifactStore` of
//! that type).

use bevy::math::{DVec2, DVec3};

use super::coord::{ChunkCoord, ChunkSize, chunk_min_world};

/// Discriminant of an artifact's data shape — declared by a layer's outputs and used to key stores /
/// visualizers. The full set is fixed now (extensibility is in the data, not new kinds), even though
/// the slice only constructs `ScalarField2D`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArtifactKind {
    ScalarField2D,
    ScalarField3D,
    Classification2D,
    Classification3D,
    InstanceStream,
    VectorGraph,
}

/// A produced, world-anchored chunk artifact. Concrete kinds implement this; stores are generic over
/// it. `Send + Sync + 'static` so artifacts live across the compute task pool and in `Arc`s shared
/// into both the GPU upload and the CPU `eval_primitive` surface query.
pub trait Artifact: Send + Sync + 'static {
    /// The kind discriminant for this artifact type.
    fn kind() -> ArtifactKind
    where
        Self: Sized;
}

/// One height-field node: world-metre surface height plus its analytic world-space XZ gradient.
/// `f32` is ample for the rendered surface; the authoritative f64 generation is narrowed here (the
/// field is stored relative to the chunk, so the f32 has full sub-mm precision within a chunk).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HeightNode {
    pub height: f32,
    pub dh_dx: f32,
    pub dh_dz: f32,
}

/// A chunk-local 2D scalar height field with the analytic XZ gradient at each node.
///
/// Stored at `res` cells per axis ⇒ `(res + 1)²` nodes, row-major with **+X fastest then +Z**. The
/// extra "+1" node on each high edge is an **apron**: it duplicates the value the neighbouring chunk
/// owns at the shared boundary, so bilinear sampling anywhere in `[min, min + world_size]` reads only
/// this chunk's nodes and agrees with the neighbour at the seam (no cracks — the §10 "padding
/// correctness" property at the consumer's sampling level).
pub struct ScalarField2D {
    pub coord: ChunkCoord,
    pub size: ChunkSize,
    /// Cells per axis. Nodes per axis = `res + 1`.
    pub res: u32,
    /// World metres between adjacent nodes = `size.world_size() / res`.
    pub node_spacing: f64,
    /// Chunk min corner (cached; `chunk_min_world(coord, size)`).
    pub min_world: DVec3,
    /// `(res + 1)²` nodes, row-major (+X fastest, then +Z).
    pub nodes: Vec<HeightNode>,
}

impl Artifact for ScalarField2D {
    fn kind() -> ArtifactKind {
        ArtifactKind::ScalarField2D
    }
}

impl ScalarField2D {
    /// Number of nodes per axis (`res + 1`, including the high-edge apron).
    #[inline]
    pub fn nodes_per_axis(res: u32) -> u32 {
        res + 1
    }

    /// Allocate a zeroed field for `coord` at `res` cells/axis. Fill via [`set`](Self::set).
    pub fn zeroed(coord: ChunkCoord, size: ChunkSize, res: u32) -> Self {
        assert!(res >= 1, "ScalarField2D needs res >= 1");
        let n = Self::nodes_per_axis(res) as usize;
        Self {
            coord,
            size,
            res,
            node_spacing: size.world_size() / res as f64,
            min_world: chunk_min_world(coord, size),
            nodes: vec![HeightNode::default(); n * n],
        }
    }

    /// Row-major node index for `(i, j)` with `i, j ∈ [0, res]`.
    #[inline]
    pub fn index(&self, i: u32, j: u32) -> usize {
        let n = Self::nodes_per_axis(self.res);
        debug_assert!(i < n && j < n, "node ({i},{j}) out of range for res {}", self.res);
        (j * n + i) as usize
    }

    /// Set node `(i, j)`.
    #[inline]
    pub fn set(&mut self, i: u32, j: u32, node: HeightNode) {
        let idx = self.index(i, j);
        self.nodes[idx] = node;
    }

    /// Get node `(i, j)`.
    #[inline]
    pub fn node(&self, i: u32, j: u32) -> HeightNode {
        self.nodes[self.index(i, j)]
    }

    /// World XZ of node `(i, j)`.
    #[inline]
    pub fn node_world_xz(&self, i: u32, j: u32) -> DVec2 {
        DVec2::new(
            self.min_world.x + i as f64 * self.node_spacing,
            self.min_world.z + j as f64 * self.node_spacing,
        )
    }

    /// Bilinearly sample the field at world `world_xz`, returning interpolated height + gradient.
    /// The single source of truth for "the surface height/normal at this world point", shared by the
    /// CPU `eval_primitive` (picking) and mirrored by the GPU bake — so picking matches the render.
    ///
    /// Clamps the local coordinate into the valid node window, so sampling exactly on (or a hair
    /// past) the far boundary reads the apron node rather than indexing out of bounds.
    pub fn sample(&self, world_xz: DVec2) -> HeightNode {
        let lx = (world_xz.x - self.min_world.x) / self.node_spacing;
        let lz = (world_xz.y - self.min_world.z) / self.node_spacing;
        // Cell index clamped to [0, res-1] so the +1 tap stays within the [0, res] node range
        // (the apron node at `res` is the last valid tap). Fractional weight clamped to [0, 1] so
        // out-of-window samples saturate to the boundary node instead of extrapolating.
        let last_cell = (self.res - 1) as f64;
        let fi = lx.floor().clamp(0.0, last_cell);
        let fj = lz.floor().clamp(0.0, last_cell);
        let i = fi as u32;
        let j = fj as u32;
        let tx = (lx - fi).clamp(0.0, 1.0);
        let tz = (lz - fj).clamp(0.0, 1.0);

        let n00 = self.node(i, j);
        let n10 = self.node(i + 1, j);
        let n01 = self.node(i, j + 1);
        let n11 = self.node(i + 1, j + 1);

        let lerp = |a: f32, b: f32, t: f64| a + (b - a) * t as f32;
        let bilerp = |a: f32, b: f32, c: f32, d: f32| {
            let ab = lerp(a, b, tx);
            let cd = lerp(c, d, tx);
            lerp(ab, cd, tz)
        };
        HeightNode {
            height: bilerp(n00.height, n10.height, n01.height, n11.height),
            dh_dx: bilerp(n00.dh_dx, n10.dh_dx, n01.dh_dx, n11.dh_dx),
            dh_dz: bilerp(n00.dh_dz, n10.dh_dz, n01.dh_dz, n11.dh_dz),
        }
    }

    /// Min/max stored node height — the vertical extent this chunk's terrain spans, for the bake's
    /// edit AABB so the world-spanning Terrain volume bounds tightly per region.
    pub fn height_range(&self) -> (f32, f32) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for n in &self.nodes {
            lo = lo.min(n.height);
            hi = hi.max(n.height);
        }
        (lo, hi)
    }
}

#[cfg(test)]
mod tests {
    use super::super::coord::LayerId;
    use super::*;
    use bevy::math::IVec3;

    fn coord() -> ChunkCoord {
        ChunkCoord::new(LayerId(0), IVec3::new(2, 0, -1))
    }

    /// Node layout: `(res+1)²` nodes, indexable, round-tripping set/get.
    #[test]
    fn node_layout_and_set_get() {
        let mut f = ScalarField2D::zeroed(coord(), ChunkSize::new(64), 8);
        assert_eq!(f.nodes.len(), 9 * 9);
        f.set(3, 5, HeightNode { height: 1.5, dh_dx: 0.25, dh_dz: -0.5 });
        assert_eq!(f.node(3, 5).height, 1.5);
        assert_eq!(f.node(3, 5).dh_dx, 0.25);
        // distinct indices don't alias
        assert_eq!(f.node(5, 3).height, 0.0);
    }

    /// Sampling exactly at a node returns that node's value (bilinear weights collapse to it).
    #[test]
    fn sample_at_nodes_is_exact() {
        let size = ChunkSize::new(40);
        let mut f = ScalarField2D::zeroed(coord(), size, 4);
        // Fill with a recognizable per-node value.
        for j in 0..=4 {
            for i in 0..=4 {
                f.set(i, j, HeightNode { height: (i * 10 + j) as f32, dh_dx: 0.0, dh_dz: 0.0 });
            }
        }
        for j in 0..4 {
            for i in 0..4 {
                let wp = f.node_world_xz(i, j);
                let s = f.sample(wp);
                assert!((s.height - (i * 10 + j) as f32).abs() < 1e-4, "node ({i},{j}) sample {s:?}");
            }
        }
    }

    /// A planar height field `h = a·x + b·z + c` is reproduced exactly by bilinear sampling, and the
    /// stored constant gradient is returned — the property the Lipschitz/normal path relies on.
    #[test]
    fn sample_reproduces_planar_field_and_gradient() {
        let size = ChunkSize::new(100);
        let res = 10u32;
        let mut f = ScalarField2D::zeroed(coord(), size, res);
        let (a, b, c) = (0.3f64, -0.7f64, 12.0f64);
        for j in 0..=res {
            for i in 0..=res {
                let wp = f.node_world_xz(i, j);
                f.set(i, j, HeightNode {
                    height: (a * wp.x + b * wp.y + c) as f32,
                    dh_dx: a as f32,
                    dh_dz: b as f32,
                });
            }
        }
        // Sample at several interior off-node points; bilinear is exact for a linear field.
        let min = f.min_world;
        for &(u, v) in &[(0.13, 0.41), (0.5, 0.5), (0.87, 0.22), (0.99, 0.99)] {
            let wp = DVec2::new(min.x + u * size.world_size(), min.z + v * size.world_size());
            let s = f.sample(wp);
            let expect = (a * wp.x + b * wp.y + c) as f32;
            assert!((s.height - expect).abs() < 1e-2, "planar sample at ({u},{v}): {} vs {expect}", s.height);
            assert!((s.dh_dx - a as f32).abs() < 1e-5 && (s.dh_dz - b as f32).abs() < 1e-5);
        }
    }

    /// Sampling on/just past the far boundary clamps into the apron rather than panicking.
    #[test]
    fn sample_far_boundary_uses_apron() {
        let size = ChunkSize::new(50);
        let mut f = ScalarField2D::zeroed(coord(), size, 5);
        for j in 0..=5 {
            for i in 0..=5 {
                f.set(i, j, HeightNode { height: 7.0, dh_dx: 0.0, dh_dz: 0.0 });
            }
        }
        let min = f.min_world;
        // Exactly at the far corner, and a hair beyond — must not panic and returns the edge value.
        let s = f.sample(DVec2::new(min.x + size.world_size(), min.z + size.world_size()));
        assert!((s.height - 7.0).abs() < 1e-4);
        let s2 = f.sample(DVec2::new(min.x + size.world_size() + 0.001, min.z));
        assert!((s2.height - 7.0).abs() < 1e-4);
    }

    /// `height_range` reports the min/max node height.
    #[test]
    fn height_range_reports_extent() {
        let mut f = ScalarField2D::zeroed(coord(), ChunkSize::new(16), 2);
        f.set(0, 0, HeightNode { height: -3.0, dh_dx: 0.0, dh_dz: 0.0 });
        f.set(2, 2, HeightNode { height: 9.0, dh_dx: 0.0, dh_dz: 0.0 });
        let (lo, hi) = f.height_range();
        assert_eq!(lo, -3.0);
        assert_eq!(hi, 9.0);
    }
}
