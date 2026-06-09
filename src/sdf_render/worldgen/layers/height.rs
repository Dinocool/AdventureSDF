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
use super::super::noise::{FbmParams, fbm_height, fbm_height_grad};
use super::erosion::{ErosionParams, erode_height};

/// Bump when the height layer's *output* intentionally changes (algorithm/constants). It keys the
/// disk cache (WORLD_GEN_PLAN §2.3) and the parity reference vectors — a change here forces
/// regenerating reference values, making "I meant to change the terrain" explicit and review-visible.
pub const HEIGHT_GEN_VERSION: u32 = 3;

/// Central-difference step (world metres) for the eroded/ridged gradient in [`HeightLayer::sample_world`].
/// Small enough to resolve the finest erosion feature, large enough to stay well clear of `f64` noise.
pub const EROSION_GRAD_EPS: f64 = 0.5;

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
    /// Ridged-multifractal blend in `[0, 1]`: `0` = plain fBm, `1` = fully ridged (the fBm octave sum
    /// folded toward `1 - |fbm|`-style sharp peaks). Folds the SAME octave sum, so the field stays
    /// band-limited; the (now non-analytic) gradient is taken by central difference in `sample_world`.
    pub ridge: f32,
    /// Per-layer salt mixed with the world seed so this layer has an independent stream.
    pub seed_salt: u32,
}

impl Default for HeightParams {
    fn default() -> Self {
        // Tuned DRAMATIC: wide mountains (low base_freq), more octaves for branching detail, and a
        // large octave-0 amplitude for hundreds-of-metres relief. A partial `ridge` fold sharpens the
        // peaks into ridgelines; the erosion layer then carves gullies into the slopes. The relief is
        // kept broad enough (octave-0 wavelength = 1536 m) that every clipmap LOD can still march it.
        Self {
            octaves: 6,
            base_freq: 1.0 / 1536.0,
            lacunarity: 2.0,
            gain: 0.5,
            amplitude: 280.0,
            sea_level: 0.0,
            ridge: 0.5,
            seed_salt: 0,
        }
    }
}

impl HeightParams {
    /// Geometric sum of the per-octave amplitudes (`amplitude·Σ gain^o`) — the maximum fBm swing
    /// magnitude (value noise ∈ [-1, 1] per octave). The vertical-band derivation and the parity tests
    /// both use it to bound the surface. Pure basic ops.
    pub fn amplitude_sum(&self) -> f64 {
        let mut amp = self.amplitude as f64;
        let mut sum = 0.0;
        for _ in 0..self.octaves {
            sum += amp;
            amp *= self.gain as f64;
        }
        sum
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
    /// The erosion-filter params applied on top of the base height (Layer #2; see [`super::erosion`]).
    pub erosion: ErosionParams,
    /// Chunk edge in base cells for THIS tier (`HEIGHT_CHUNK_CELLS·2^t`). Tier 0 = `HEIGHT_CHUNK_CELLS`.
    chunk_cells: u32,
    decls: [ArtifactDecl; 1],
}

impl HeightLayer {
    /// Tier-0 layer (`chunk_cells = HEIGHT_CHUNK_CELLS`). Convenience for the single-tier callers /
    /// tests; multi-tier clipmap construction uses [`new_tier`](Self::new_tier).
    pub fn new(id: LayerId, params: HeightParams, erosion: ErosionParams) -> Self {
        Self::new_tier(id, params, erosion, HEIGHT_CHUNK_CELLS)
    }

    /// A layer for an arbitrary clipmap tier: `chunk_cells` = the tier's chunk edge in base cells
    /// (`HEIGHT_CHUNK_CELLS·2^t`). `HEIGHT_FIELD_RES` nodes per chunk regardless of tier (so the node
    /// spacing scales with the tier). The carved surface (`sample_world`) is identical across tiers.
    pub fn new_tier(id: LayerId, params: HeightParams, erosion: ErosionParams, chunk_cells: u32) -> Self {
        Self {
            id,
            params,
            erosion,
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

    /// Full carved SCALAR surface height at world `(wx, wz)`: fBm, optionally folded toward a ridged
    /// multifractal by `params.ridge`, plus sea level, then carved by the erosion filter. Pure /
    /// deterministic / bit-portable (basic `f64` ops + the portable noise basis). This is the function
    /// the central difference in [`sample_world`](Self::sample_world) differentiates for the gradient.
    #[inline]
    fn height_world(&self, wx: f64, wz: f64, fbm: &FbmParams, world_seed: u64) -> f64 {
        let ridge = self.params.ridge as f64;
        let h_base = if ridge == 0.0 {
            // Plain fBm value (value-only variant — no gradient needed here).
            fbm_height(wx, wz, fbm)
        } else {
            // Ridged fold of the SAME octave sum: blend `h` toward `amplitude_sum·(1 - |fbm|/amp_sum)`,
            // i.e. fold the signed fBm to its ridged form scaled to the same swing. Keeps it band-limited
            // (same octaves) and parameter-driven (no magic constants).
            let h = fbm_height(wx, wz, fbm);
            let amp_sum = self.params.amplitude_sum();
            // `1 - |h|/amp_sum ∈ [0, 1]`; rescale to the fBm swing and recentre so ridge=1 still spans a
            // comparable band. `ridged = amp_sum·(1 - 2·|h|/amp_sum) = amp_sum - 2·|h|` (mapped to
            // roughly `[-amp_sum, amp_sum]`, peaks where the fBm crosses zero → sharp ridgelines).
            let ah = if h < 0.0 { -h } else { h };
            let ridged = amp_sum - 2.0 * ah;
            h + ridge * (ridged - h)
        };
        let h_sea = h_base + self.params.sea_level as f64;
        // Carve. `erode_height` needs the base XZ gradient to seed its slope-damp feedback; recompute it
        // analytically from the fBm (the ridge fold's own derivative is folded in by the central
        // difference in `sample_world`, so the seed gradient here is the smooth fBm slope — fine).
        let (_, gx, gz) = fbm_height_grad(wx, wz, fbm);
        erode_height(h_sea, gx, gz, wx, wz, world_seed, &self.erosion)
    }

    /// Authoritative carved surface height + XZ gradient at world `(wx, wz)`. Single source of truth
    /// for "the terrain surface here", shared by chunk generation, the parity harness, and the CPU
    /// `eval_primitive` picking path. Deterministic & bit-portable.
    ///
    /// FAST PATH — plain fBm (`ridge == 0` AND erosion disabled): the closed-form `fbm_height_grad`
    /// gradient (exact, cheap). OTHERWISE the ridge fold (`1 - |fbm|`) and the erosion carve are not
    /// closed-form differentiable through this layer (they'd need the noise Hessian), so the gradient is
    /// taken by a CENTRAL DIFFERENCE of [`height_world`](Self::height_world) at `±EROSION_GRAD_EPS` in
    /// wx and wz — 5 height evals/node, fine for per-chunk (infrequent) generation; the bake reads the
    /// stored gradient. Still a pure deterministic `f(wx, wz, seed)`.
    #[inline]
    pub fn sample_world(&self, wx: f64, wz: f64, world_seed: u64) -> HeightNode {
        let fbm = self.fbm_params(world_seed);

        if self.params.ridge == 0.0 && !self.erosion.enabled {
            // Exact analytic path (unchanged behaviour for plain-fBm configs).
            let (h, gx, gz) = fbm_height_grad(wx, wz, &fbm);
            return HeightNode {
                height: (h + self.params.sea_level as f64) as f32,
                dh_dx: gx as f32,
                dh_dz: gz as f32,
            };
        }

        let e = EROSION_GRAD_EPS;
        let h = self.height_world(wx, wz, &fbm, world_seed);
        let hxp = self.height_world(wx + e, wz, &fbm, world_seed);
        let hxm = self.height_world(wx - e, wz, &fbm, world_seed);
        let hzp = self.height_world(wx, wz + e, &fbm, world_seed);
        let hzm = self.height_world(wx, wz - e, &fbm, world_seed);
        let inv2e = 1.0 / (2.0 * e);
        HeightNode {
            height: h as f32,
            dh_dx: ((hxp - hxm) * inv2e) as f32,
            dh_dz: ((hzp - hzm) * inv2e) as f32,
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
        HeightLayer::new(LayerId(0), HeightParams::default(), ErosionParams::default())
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

    /// The stored gradient (from the `EROSION_GRAD_EPS` central difference) is finite and approximately
    /// matches a FINER-eps central difference of the carved height — guards the terrain normals the bake
    /// reconstructs from this gradient.
    #[test]
    fn stored_gradient_matches_finer_central_difference() {
        let l = layer();
        let seed = 4242u64;
        // Sloped, mid-altitude points where erosion + ridge bite.
        for &(wx, wz) in &[(321.0, -123.0), (-560.0, 880.0), (1500.5, 700.25)] {
            let n = l.sample_world(wx, wz, seed);
            assert!(n.dh_dx.is_finite() && n.dh_dz.is_finite(), "gradient not finite at ({wx},{wz})");
            // Reference: a finer central difference of the full carved height.
            let fbm = l.fbm_params(seed);
            let e = 0.05f64;
            let hxp = l.height_world(wx + e, wz, &fbm, seed);
            let hxm = l.height_world(wx - e, wz, &fbm, seed);
            let hzp = l.height_world(wx, wz + e, &fbm, seed);
            let hzm = l.height_world(wx, wz - e, &fbm, seed);
            let fd_x = (hxp - hxm) / (2.0 * e);
            let fd_z = (hzp - hzm) / (2.0 * e);
            // Generous tol: the two epsilons see slightly different band-limits of the same field.
            assert!((n.dh_dx as f64 - fd_x).abs() < 1.0, "∂x at ({wx},{wz}): {} vs {fd_x}", n.dh_dx);
            assert!((n.dh_dz as f64 - fd_z).abs() < 1.0, "∂z at ({wx},{wz}): {} vs {fd_z}", n.dh_dz);
        }
    }

    /// With erosion disabled AND `ridge == 0`, `sample_world` takes the exact analytic fBm path — its
    /// gradient equals `fbm_height_grad` exactly (bit-for-bit), unchanged from the pre-erosion behaviour.
    #[test]
    fn plain_fbm_path_is_exact_analytic() {
        let params = HeightParams { ridge: 0.0, ..Default::default() };
        let erosion = ErosionParams { enabled: false, ..Default::default() };
        let l = HeightLayer::new(LayerId(0), params, erosion);
        let seed = 7u64;
        let fbm = l.fbm_params(seed);
        for &(wx, wz) in &[(0.0, 0.0), (123.5, -456.25), (-789.0, 1011.0)] {
            let n = l.sample_world(wx, wz, seed);
            let (h, gx, gz) = fbm_height_grad(wx, wz, &fbm);
            assert_eq!(n.height.to_bits(), ((h + params.sea_level as f64) as f32).to_bits());
            assert_eq!(n.dh_dx.to_bits(), (gx as f32).to_bits());
            assert_eq!(n.dh_dz.to_bits(), (gz as f32).to_bits());
        }
    }
}
