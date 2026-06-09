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
use super::super::noise::{FbmParams, fbm_height_grad, fbm_height_grad_hess};
use super::erosion::{ErosionParams, erode_with_grad};

/// Bump when the height layer's *output* intentionally changes (algorithm/constants). It keys the
/// disk cache (WORLD_GEN_PLAN §2.3) and the parity reference vectors — a change here forces
/// regenerating reference values, making "I meant to change the terrain" explicit and review-visible.
pub const HEIGHT_GEN_VERSION: u32 = 4;

/// Chunk edge in base cells (= metres) for the height layer's tier.
pub const HEIGHT_CHUNK_CELLS: u32 = 128;
/// Cells per axis the height field is sampled at within a chunk (nodes = res + 1). `128 / 64 = 2 m`
/// node spacing — the authoritative base resolution; the GPU adds finer cosmetic detail on top.
pub const HEIGHT_FIELD_RES: u32 = 64;

/// ANTI-ALIAS supersample factor per axis used by [`HeightLayer::generate`] when sharp features are
/// present (ridge fold / erosion). The ridge fold `1−|fbm|` and erosion creases are INFINITELY sharp
/// (sub-node), so POINT-sampling them at the node grid ALIASES the crest — captured only where a node
/// lands on it → the mesh renders degenerate SPIKES. Box-filtering each node over its cell with an
/// `N×N` grid band-limits those creases to the node Nyquist, so the mesh sees a clean representable
/// ridge. The `N²` cost is paid ONCE per node into the CACHED artifact (reused by every bake sample —
/// generation is separate from sampling), so it's fully amortised. Plain fBm (smooth) skips this.
pub const HEIGHT_GEN_SUPERSAMPLE: u32 = 3;

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
    /// band-limited; the fold's gradient is taken CLOSED-FORM (value-noise gradient + base Hessian) in
    /// `sample_world` / `carved_grad` — no central difference.
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

    /// Full carved surface at world `(wx, wz)`: fBm, optionally folded toward a ridged multifractal by
    /// `params.ridge`, plus sea level, then carved by the erosion filter — returning the height AND its
    /// CLOSED-FORM XZ gradient `(H, ∂H/∂wx, ∂H/∂wz)`. Pure / deterministic / bit-portable (basic `f64`
    /// ops + the portable noise basis + Hessian). ONE eval/node — replaces the old 5-tap central
    /// difference (the gen-perf + FD-smoothing regression).
    ///
    /// The gradient is exact: the fBm carries an analytic gradient+Hessian; the ridge fold `1−|fbm|`
    /// differentiates with the value-noise gradient (its piecewise-constant `sign` is measure-zero, as
    /// for the FD it replaces); and [`erode_with_grad`] differentiates the erosion carve using the base
    /// Hessian (the slope-damp term needs `d(∇h)`). The erosion's slope-damp is now seeded with the TRUE
    /// (ridge-folded) surface gradient — so the carved surface VALUE differs slightly from the old
    /// smooth-fBm-seeded form (re-pinned parity, `HEIGHT_GEN_VERSION` 3 → 4).
    #[inline]
    fn carved_grad(&self, wx: f64, wz: f64, fbm: &FbmParams, world_seed: u64) -> (f64, f64, f64) {
        // Base fBm value + analytic gradient + Hessian (one eval).
        let (h, gx, gz, hxx, hxz, hzz) = fbm_height_grad_hess(wx, wz, fbm);

        let ridge = self.params.ridge as f64;
        let (h_base, gx_b, gz_b, hxx_b, hxz_b, hzz_b) = if ridge == 0.0 {
            (h, gx, gz, hxx, hxz, hzz) // plain fBm — exact analytic, no fold.
        } else {
            // Ridged fold of the SAME octave sum (band-limited, parameter-driven):
            //   h_base = h + ridge·((amp_sum − 2|h|) − h) = h·(1−ridge) + ridge·amp_sum − 2·ridge·|h|.
            // ⇒ ∂h_base = ∂h·[(1−ridge) − 2·ridge·sign(h)] and the Hessian scales by the SAME factor `k`
            //   (sign(h)'s jump at h=0 is measure-zero — matches the FD this replaces).
            let amp_sum = self.params.amplitude_sum();
            let ah = if h < 0.0 { -h } else { h };
            let ridged = amp_sum - 2.0 * ah;
            let h_base = h + ridge * (ridged - h);
            let sgn = if h < 0.0 { -1.0 } else { 1.0 };
            let k = (1.0 - ridge) - 2.0 * ridge * sgn;
            (h_base, gx * k, gz * k, hxx * k, hxz * k, hzz * k)
        };
        let h_sea = h_base + self.params.sea_level as f64;
        // Carve: analytic eroded height + gradient, seeded by the TRUE (ridge-folded) base gradient +
        // Hessian. `enabled = false` ⇒ exact identity `(h_sea, gx_b, gz_b)`.
        erode_with_grad(h_sea, gx_b, gz_b, hxx_b, hxz_b, hzz_b, wx, wz, world_seed, &self.erosion)
    }

    /// Authoritative carved surface height + analytic XZ gradient at world `(wx, wz)`. Single source of
    /// truth for "the terrain surface here", shared by chunk generation, the parity harness, and the CPU
    /// `eval_primitive` picking path. Deterministic & bit-portable.
    ///
    /// FAST PATH — plain fBm (`ridge == 0` AND erosion disabled): the closed-form `fbm_height_grad`
    /// (exact, cheap, bit-for-bit unchanged from the pre-erosion behaviour). OTHERWISE the ridge fold +
    /// erosion carve are differentiated CLOSED-FORM via [`carved_grad`](Self::carved_grad) (one eval, the
    /// noise Hessian) — no central-difference taps. Still a pure deterministic `f(wx, wz, seed)`.
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

        let (h, gx, gz) = self.carved_grad(wx, wz, &fbm, world_seed);
        HeightNode { height: h as f32, dh_dx: gx as f32, dh_dz: gz as f32 }
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
        // ANTI-ALIAS: with the ridge fold / erosion the surface has infinitely-sharp creases — point
        // sampling them at the node grid ALIASES the crest into degenerate mesh spikes. Box-filter each
        // node over its own cell (an `N×N` grid centred on the node, ±½ spacing) to band-limit those
        // creases to the node Nyquist. The filter is a pure function of (world position, spacing), centred
        // and symmetric, so a chunk's far-edge APRON node box-filters the SAME world point as the
        // neighbour's near node ⇒ still seam-free. Plain fBm (smooth, no ridge/erosion) needs no AA → the
        // single-sample fast path. The `N²` cost is paid ONCE per node into the CACHED artifact (reused by
        // every bake sample), so it never touches the per-sample hot path.
        let aa = if self.params.ridge != 0.0 || self.erosion.enabled { HEIGHT_GEN_SUPERSAMPLE.max(1) } else { 1 };
        let spacing = field.node_spacing;
        for j in 0..=res {
            for i in 0..=res {
                let wp = field.node_world_xz(i, j); // DVec2(world_x, world_z)
                if aa == 1 {
                    field.set(i, j, self.sample_world(wp.x, wp.y, ctx.seed));
                    continue;
                }
                let (mut h, mut gx, mut gz) = (0.0f64, 0.0f64, 0.0f64);
                for sj in 0..aa {
                    for si in 0..aa {
                        let ox = ((si as f64 + 0.5) / aa as f64 - 0.5) * spacing;
                        let oz = ((sj as f64 + 0.5) / aa as f64 - 0.5) * spacing;
                        let n = self.sample_world(wp.x + ox, wp.y + oz, ctx.seed);
                        h += n.height as f64;
                        gx += n.dh_dx as f64;
                        gz += n.dh_dz as f64;
                    }
                }
                let inv = 1.0 / (aa * aa) as f64;
                field.set(i, j, HeightNode {
                    height: (h * inv) as f32,
                    dh_dx: (gx * inv) as f32,
                    dh_dz: (gz * inv) as f32,
                });
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

    /// A PLAIN-fBm layer (no ridge, no erosion) — `generate` takes the point-sample fast path
    /// (`aa == 1`), so its nodes equal `sample_world` exactly.
    fn plain_layer() -> HeightLayer {
        HeightLayer::new(
            LayerId(0),
            HeightParams { ridge: 0.0, ..Default::default() },
            ErosionParams { enabled: false, ..Default::default() },
        )
    }

    /// For a PLAIN layer (no sharp features ⇒ no AA), `generate` produces a height field of the right
    /// shape, every node equal to `sample_world` (the single surface-truth function) at its world coord.
    #[test]
    fn generate_fills_field_matching_sample_world() {
        let l = plain_layer();
        let coord = ChunkCoord::new(l.id(), IVec3::new(1, 0, -2));
        let ctx = GenCtx { coord, seed: 7, size: l.chunk_size() };
        let mut out = GenOutput::default();
        l.generate(&ctx, &mut out);
        let field = out.take::<ScalarField2D>(HeightLayer::OUTPUT).unwrap();
        assert_eq!(field.res, HEIGHT_FIELD_RES);
        assert_eq!(field.nodes.len(), ((HEIGHT_FIELD_RES + 1) * (HEIGHT_FIELD_RES + 1)) as usize);
        for &(i, j) in &[(0u32, 0u32), (10, 20), (HEIGHT_FIELD_RES, HEIGHT_FIELD_RES)] {
            let wp = field.node_world_xz(i, j);
            let expect = l.sample_world(wp.x, wp.y, ctx.seed);
            assert_eq!(field.node(i, j), expect);
        }
    }

    /// With ridge/erosion on, `generate` ANTI-ALIASES: each node is the `N×N` box-filter average over its
    /// cell (band-limiting the sharp crest), NOT the raw point sample — and it stays a seam-free pure
    /// function of world position (asserted by `adjacent_chunks_agree_on_shared_boundary` below, which
    /// uses the AA `layer()`).
    #[test]
    fn generate_anti_aliases_sharp_features() {
        let l = layer(); // default: ridge + erosion ⇒ aa > 1
        let coord = ChunkCoord::new(l.id(), IVec3::new(1, 0, -2));
        let ctx = GenCtx { coord, seed: 7, size: l.chunk_size() };
        let mut out = GenOutput::default();
        l.generate(&ctx, &mut out);
        let field = out.take::<ScalarField2D>(HeightLayer::OUTPUT).unwrap();
        let (i, j) = (10u32, 20u32);
        let wp = field.node_world_xz(i, j);
        // Recompute the expected box-filter average independently.
        let aa = HEIGHT_GEN_SUPERSAMPLE;
        let spacing = field.node_spacing;
        let (mut h, mut gx, mut gz) = (0.0f64, 0.0f64, 0.0f64);
        for sj in 0..aa {
            for si in 0..aa {
                let ox = ((si as f64 + 0.5) / aa as f64 - 0.5) * spacing;
                let oz = ((sj as f64 + 0.5) / aa as f64 - 0.5) * spacing;
                let n = l.sample_world(wp.x + ox, wp.y + oz, ctx.seed);
                h += n.height as f64;
                gx += n.dh_dx as f64;
                gz += n.dh_dz as f64;
            }
        }
        let inv = 1.0 / (aa * aa) as f64;
        let node = field.node(i, j);
        assert!((node.height - (h * inv) as f32).abs() < 1e-3, "node is the box-filter average");
        assert!((node.dh_dx - (gx * inv) as f32).abs() < 1e-3);
        // And it actually differs from the raw point sample (AA is doing something).
        let point = l.sample_world(wp.x, wp.y, ctx.seed);
        assert_ne!(node.height.to_bits(), point.height.to_bits(), "AA node differs from the point sample");
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

    /// The CLOSED-FORM carved gradient (now stored by `sample_world`) matches a central difference of the
    /// carved height — the correctness guard for the analytic erosion/ridge gradient (it replaced the FD
    /// taps). The carved height value is `carved_grad(...).0` (its value lane is bit-identical to the old
    /// `erode_height` path). Tolerance ~1e-2 (the FD's own truncation error at this eps).
    #[test]
    fn analytic_gradient_matches_central_difference() {
        let l = layer();
        let seed = 4242u64;
        let fbm = l.fbm_params(seed);
        // Sloped, mid-altitude points where erosion + ridge bite.
        for &(wx, wz) in &[(321.0, -123.0), (-560.0, 880.0), (1500.5, 700.25)] {
            let (_, gx, gz) = l.carved_grad(wx, wz, &fbm, seed);
            assert!(gx.is_finite() && gz.is_finite(), "gradient not finite at ({wx},{wz})");
            // Reference: a central difference of the full carved height (value lane of carved_grad).
            let e = 0.01f64;
            let hxp = l.carved_grad(wx + e, wz, &fbm, seed).0;
            let hxm = l.carved_grad(wx - e, wz, &fbm, seed).0;
            let hzp = l.carved_grad(wx, wz + e, &fbm, seed).0;
            let hzm = l.carved_grad(wx, wz - e, &fbm, seed).0;
            let fd_x = (hxp - hxm) / (2.0 * e);
            let fd_z = (hzp - hzm) / (2.0 * e);
            assert!((gx - fd_x).abs() < 1e-2, "∂x at ({wx},{wz}): analytic {gx} vs FD {fd_x}");
            assert!((gz - fd_z).abs() < 1e-2, "∂z at ({wx},{wz}): analytic {gz} vs FD {fd_z}");
        }
    }

    /// `sample_world`'s stored gradient (the analytic carved gradient narrowed to f32) is finite and
    /// matches `carved_grad` — guards the terrain normals the bake reconstructs from this gradient.
    #[test]
    fn sample_world_stores_analytic_gradient() {
        let l = layer();
        let seed = 4242u64;
        let fbm = l.fbm_params(seed);
        for &(wx, wz) in &[(321.0, -123.0), (-560.0, 880.0), (1500.5, 700.25)] {
            let n = l.sample_world(wx, wz, seed);
            assert!(n.dh_dx.is_finite() && n.dh_dz.is_finite(), "gradient not finite at ({wx},{wz})");
            let (h, gx, gz) = l.carved_grad(wx, wz, &fbm, seed);
            assert_eq!(n.height.to_bits(), (h as f32).to_bits());
            assert_eq!(n.dh_dx.to_bits(), (gx as f32).to_bits());
            assert_eq!(n.dh_dz.to_bits(), (gz as f32).to_bits());
        }
    }

    /// Terrain-GEN microbench: the analytic gradient path (`sample_world`, ONE carved eval/node) vs the
    /// old 5-tap central-difference path (one carved eval + 4 offset taps/node). `#[ignore]` — run with
    /// `--release --ignored --nocapture`. Reports the per-node gen time + the speedup, the direct measure
    /// of the gen-perf regression the analytic path recovers.
    #[test]
    #[ignore = "gen-perf microbench; run with --release --ignored --nocapture"]
    fn bench_analytic_vs_fd_gradient() {
        let l = layer();
        let seed = 4242u64;
        let fbm = l.fbm_params(seed);
        // A grid of distinct world points (HEIGHT_FIELD_RES² ≈ one chunk's worth, several chunks over).
        let n = 256usize;
        let mut pts = Vec::with_capacity(n * n);
        for j in 0..n {
            for i in 0..n {
                pts.push((i as f64 * 2.0 - 256.0, j as f64 * 2.0 + 100.0));
            }
        }
        let e = 0.5f64; // the old EROSION_GRAD_EPS

        // Analytic: 1 carved eval/node (value + gradient together).
        let t0 = std::time::Instant::now();
        let mut acc = 0.0f64;
        for &(wx, wz) in &pts {
            let (h, gx, gz) = l.carved_grad(wx, wz, &fbm, seed);
            acc += h + gx + gz;
        }
        let analytic_ns = t0.elapsed().as_nanos() as f64 / pts.len() as f64;

        // Old FD: 1 value eval + 4 offset value taps/node (the regression this replaced).
        let t1 = std::time::Instant::now();
        let mut acc2 = 0.0f64;
        for &(wx, wz) in &pts {
            let h = l.carved_grad(wx, wz, &fbm, seed).0;
            let hxp = l.carved_grad(wx + e, wz, &fbm, seed).0;
            let hxm = l.carved_grad(wx - e, wz, &fbm, seed).0;
            let hzp = l.carved_grad(wx, wz + e, &fbm, seed).0;
            let hzm = l.carved_grad(wx, wz - e, &fbm, seed).0;
            acc2 += h + (hxp - hxm) / (2.0 * e) + (hzp - hzm) / (2.0 * e);
        }
        let fd_ns = t1.elapsed().as_nanos() as f64 / pts.len() as f64;

        eprintln!(
            "TERRAIN-GEN-BENCH: analytic {analytic_ns:.1} ns/node | 5-tap FD {fd_ns:.1} ns/node | \
             speedup {:.2}x  (sink {acc:.3}/{acc2:.3})",
            fd_ns / analytic_ns
        );
        assert!(analytic_ns < fd_ns, "analytic must be faster than the 5-tap FD");
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
