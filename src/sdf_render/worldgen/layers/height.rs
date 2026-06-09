//! The Phase-1 base-height layer: CPU-authoritative, 2D, integer-hash fBm over the XZ plane.
//!
//! This is the root of the dependency DAG (no dependencies) and the terrain seed — it generalizes the
//! old analytic `SdfPrimitive::Heightmap` noise into an artifact the bake samples (WORLD_GEN_PLAN §9
//! phase 1 / §3). Authoritative ⇒ generated with the bit-portable [`super::super::noise`] basis so
//! shared-seed multiplayer clients agree (§2.8); the `worldgen_parity` harness pins its outputs.

use bevy::prelude::*;

use super::super::artifact::{ArtifactKind, HeightNode, ScalarField2D};
use super::super::coord::{Authority, ChunkSize, Dim, LayerId};
use super::super::layer::{ArtifactDecl, GenCtx, GenOutput, Layer};
use super::super::noise::{FbmParams, fbm_height_grad};

/// Bump when the height layer's *output* intentionally changes (algorithm/constants). It keys the
/// disk cache (WORLD_GEN_PLAN §2.3) and the parity reference vectors — a change here forces
/// regenerating reference values, making "I meant to change the terrain" explicit and review-visible.
pub const HEIGHT_GEN_VERSION: u32 = 2;

/// Chunk edge in base cells (= metres) for the height layer's tier.
pub const HEIGHT_CHUNK_CELLS: u32 = 128;
/// Cells per axis the height field is sampled at within a chunk (nodes = res + 1). `128 / 64 = 2 m`
/// node spacing — the authoritative base resolution; the GPU adds finer cosmetic detail on top.
pub const HEIGHT_FIELD_RES: u32 = 64;

/// Editor-tweakable height-layer parameters (mirrors the `SdfRaymarchParams` reflected-resource
/// idiom). A change dirties the layer → regen cascade (handled by the manager).
#[derive(Resource, Reflect, Clone, Copy, Debug, PartialEq)]
#[reflect(Resource)]
pub struct HeightParams {
    /// fBm octave count.
    pub octaves: u32,
    /// Octave-0 spatial frequency, cycles per world metre.
    pub base_freq: f32,
    /// Frequency multiplier per octave.
    pub lacunarity: f32,
    /// Amplitude multiplier per octave.
    pub gain: f32,
    /// Octave-0 world-metre amplitude.
    pub amplitude: f32,
    /// Reference plane added to the noise (sea level / base elevation), world metres.
    pub sea_level: f32,
    /// Per-layer salt mixed with the world seed so this layer has an independent stream.
    pub seed_salt: u32,
}

impl Default for HeightParams {
    fn default() -> Self {
        // Tuned GENTLE: large-wavelength, few octaves, fast amplitude falloff. High-frequency octaves
        // create near-sub-voxel cliffs that distant (coarse-LOD) bricks can't resolve → empty bricks
        // → holes. Keeping the relief broad makes the surface marchable at every clipmap LOD.
        Self {
            octaves: 4,
            base_freq: 1.0 / 512.0,
            lacunarity: 2.0,
            gain: 0.45,
            amplitude: 40.0,
            sea_level: 0.0,
            seed_salt: 0,
        }
    }
}

/// The base-height layer. Holds its params + per-tier chunk size + cached (empty) deps and (single)
/// output decl.
///
/// CLIPMAP TIERS — the same layer serves every clipmap tier. Tier `t` is just a `HeightLayer` whose
/// `chunk_cells = HEIGHT_CHUNK_CELLS · 2^t` (so a coarser node grid: `HEIGHT_FIELD_RES` nodes still,
/// but spread over a bigger chunk). CRUCIALLY, `sample_world` is UNCHANGED across tiers — every tier
/// evaluates the SAME continuous, world-anchored fBm `f(world_xz)`, only on a coarser grid. The fBm is
/// already band-limited (gentle params: ~64 m finest feature), so coarse tiers DON'T alias, and because
/// all tiers represent the SAME surface, cross-tier height values agree → per-voxel tier selection
/// produces NO seams and NO cross-LOD cracks. A `HeightLayer` is therefore a pure `f(world_xz)`; tier
/// `t`'s layer just samples it on a `HEIGHT_CHUNK_CELLS·2^t` chunk grid.
pub struct HeightLayer {
    pub id: LayerId,
    pub params: HeightParams,
    /// Chunk edge in base cells for THIS tier (`HEIGHT_CHUNK_CELLS·2^t`). Tier 0 = `HEIGHT_CHUNK_CELLS`.
    chunk_cells: u32,
    decls: [ArtifactDecl; 1],
}

impl HeightLayer {
    /// Tier-0 layer (`chunk_cells = HEIGHT_CHUNK_CELLS`). Convenience for the single-tier callers /
    /// tests; multi-tier clipmap construction uses [`new_tier`](Self::new_tier).
    pub fn new(id: LayerId, params: HeightParams) -> Self {
        Self::new_tier(id, params, HEIGHT_CHUNK_CELLS)
    }

    /// A layer for an arbitrary clipmap tier: `chunk_cells` = the tier's chunk edge in base cells
    /// (`HEIGHT_CHUNK_CELLS·2^t`). `HEIGHT_FIELD_RES` nodes per chunk regardless of tier (so the node
    /// spacing scales with the tier). The fBm (`sample_world`) is identical across tiers.
    pub fn new_tier(id: LayerId, params: HeightParams, chunk_cells: u32) -> Self {
        Self {
            id,
            params,
            chunk_cells,
            decls: [ArtifactDecl { name: Self::OUTPUT, kind: ArtifactKind::ScalarField2D }],
        }
    }

    /// The name of this layer's single produced artifact.
    pub const OUTPUT: &'static str = "height";

    /// This tier's chunk size (`chunk_cells` base cells).
    pub fn chunk_size(&self) -> ChunkSize {
        ChunkSize::new(self.chunk_cells)
    }

    /// Fold the world seed with this layer's salt into the fBm parameter block. Pure / deterministic.
    pub fn fbm_params(&self, world_seed: u64) -> FbmParams {
        let p = &self.params;
        // Mix both halves of the 64-bit world seed with the layer salt → a stable u32 noise seed.
        let seed = (world_seed as u32) ^ ((world_seed >> 32) as u32) ^ p.seed_salt;
        FbmParams {
            octaves: p.octaves,
            base_freq: p.base_freq as f64,
            lacunarity: p.lacunarity as f64,
            gain: p.gain as f64,
            amplitude: p.amplitude as f64,
            seed,
        }
    }

    /// Authoritative surface height + analytic XZ gradient at world `(wx, wz)`. Single source of
    /// truth for "the terrain surface here", shared by chunk generation, the parity harness, and
    /// (later) the CPU `eval_primitive` picking path. Deterministic & bit-portable.
    #[inline]
    pub fn sample_world(&self, wx: f64, wz: f64, world_seed: u64) -> HeightNode {
        let fbm = self.fbm_params(world_seed);
        let (h, gx, gz) = fbm_height_grad(wx, wz, &fbm);
        HeightNode {
            height: (h + self.params.sea_level as f64) as f32,
            dh_dx: gx as f32,
            dh_dz: gz as f32,
        }
    }
}

impl Layer for HeightLayer {
    fn id(&self) -> LayerId {
        self.id
    }
    fn chunk_size(&self) -> ChunkSize {
        HeightLayer::chunk_size(self)
    }
    fn dimensionality(&self) -> Dim {
        Dim::D2
    }
    fn authority(&self) -> Authority {
        Authority::Authoritative
    }
    fn produces(&self) -> &[ArtifactDecl] {
        &self.decls
    }

    fn generate(&self, ctx: &GenCtx, out: &mut GenOutput) {
        let res = HEIGHT_FIELD_RES;
        let mut field = ScalarField2D::zeroed(ctx.coord, ctx.size, res);
        // Sample every node (incl. the high-edge apron) at its exact f64 world coord — no camera
        // rebase, so two chunks sharing a boundary node compute the identical value (seam-free).
        for j in 0..=res {
            for i in 0..=res {
                let wp = field.node_world_xz(i, j); // DVec2(world_x, world_z)
                field.set(i, j, self.sample_world(wp.x, wp.y, ctx.seed));
            }
        }
        out.produce(Self::OUTPUT, field);
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::coord::{ChunkCoord, chunk_min_world};
    use super::super::super::layer::GenCtx;
    use super::*;
    use bevy::math::{DVec2, IVec3};

    fn layer() -> HeightLayer {
        HeightLayer::new(LayerId(0), HeightParams::default())
    }

    /// `generate` produces a height field of the right shape, sampled consistently with the public
    /// `sample_world` (the single surface-truth function).
    #[test]
    fn generate_fills_field_matching_sample_world() {
        let l = layer();
        let coord = ChunkCoord::new(l.id(), IVec3::new(1, 0, -2));
        let ctx = GenCtx { coord, seed: 7, size: l.chunk_size() };
        let mut out = GenOutput::default();
        l.generate(&ctx, &mut out);
        let field = out.take::<ScalarField2D>(HeightLayer::OUTPUT).unwrap();
        assert_eq!(field.res, HEIGHT_FIELD_RES);
        assert_eq!(field.nodes.len(), ((HEIGHT_FIELD_RES + 1) * (HEIGHT_FIELD_RES + 1)) as usize);
        // Every node equals sample_world at its world coord (no rebase / per-node drift).
        for &(i, j) in &[(0u32, 0u32), (10, 20), (HEIGHT_FIELD_RES, HEIGHT_FIELD_RES)] {
            let wp = field.node_world_xz(i, j);
            let expect = l.sample_world(wp.x, wp.y, ctx.seed);
            assert_eq!(field.node(i, j), expect);
        }
    }

    /// Seam-free across a chunk boundary: the far apron node of chunk C equals the near node of chunk
    /// C+1 at the same world XZ — because both come from `sample_world` at the identical world coord.
    /// This is the §10 "padding correctness" property at the field level.
    #[test]
    fn adjacent_chunks_agree_on_shared_boundary() {
        let l = layer();
        let size = l.chunk_size();
        let seed = 123;
        let ca = ChunkCoord::new(l.id(), IVec3::new(0, 0, 0));
        let cb = ChunkCoord::new(l.id(), IVec3::new(1, 0, 0)); // +X neighbour

        let mut oa = GenOutput::default();
        l.generate(&GenCtx { coord: ca, seed, size }, &mut oa);
        let fa = oa.take::<ScalarField2D>(HeightLayer::OUTPUT).unwrap();
        let mut ob = GenOutput::default();
        l.generate(&GenCtx { coord: cb, seed, size }, &mut ob);
        let fb = ob.take::<ScalarField2D>(HeightLayer::OUTPUT).unwrap();

        // Chunk A's far-X apron column (i = res) is chunk B's near column (i = 0), same world X.
        let res = HEIGHT_FIELD_RES;
        for j in 0..=res {
            let a = fa.node(res, j);
            let b = fb.node(0, j);
            assert_eq!(a.height.to_bits(), b.height.to_bits(), "boundary height seam at j={j}");
            assert_eq!(a.dh_dx.to_bits(), b.dh_dx.to_bits(), "boundary ∂x seam at j={j}");
        }
        // Sanity: the shared world X is exactly chunk B's min.x.
        let bmin = chunk_min_world(cb, size);
        let shared_x = fa.node_world_xz(res, 0).x;
        assert!((shared_x - bmin.x).abs() < 1e-9);
        let _ = DVec2::ZERO;
    }

    /// Different seeds give different terrain (the seed actually drives the field).
    #[test]
    fn seed_changes_terrain() {
        let l = layer();
        let a = l.sample_world(12.0, 34.0, 1);
        let b = l.sample_world(12.0, 34.0, 2);
        assert_ne!(a.height.to_bits(), b.height.to_bits());
    }
}
