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
pub const HEIGHT_GEN_VERSION: u32 = 5;

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
    /// band-limited; the fold's gradient is taken CLOSED-FORM (value-noise gradient + base Hessian) in
    /// `sample_world` / `carved_grad` — no central difference.
    pub ridge: f32,
    /// SURFACE BAND-LIMIT radius, in node spacings — the single finalize stage applied over the WHOLE
    /// composed surface (fBm + ridge + erosion + any future layer) in [`HeightLayer::generate`]. A sharp
    /// ridge crest is a sub-voxel-sharp convex crease that the regular Transvoxel grid can't represent
    /// (→ torn/degenerate triangles) and whose gradient flips at the crest (→ serrated normals). A
    /// separable TENT low-pass of this radius rounds the crest over `~2·radius` nodes so it becomes
    /// grid-representable AND its gradient transition smooths — fixing both the shapes and the shading at
    /// once. Filters height AND gradient with the SAME kernel, so the stored gradient stays the exact
    /// gradient of the band-limited height (`∇(K∗h) = K∗∇h`). `0` = no band-limit (point sample); higher
    /// = rounder crests (the editor slider). Bit-portable (rational tent weights, no transcendentals).
    pub band_limit: f32,
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
            // Round sharp crests over ~3 nodes (≈6 m at tier 0) so the Transvoxel grid meshes them with
            // well-formed triangles and continuous normals. Tunable live via the World Gen panel.
            band_limit: 1.5,
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

        // SURFACE BAND-LIMIT — the single finalize stage over the WHOLE composed surface. A sharp ridge
        // crest / erosion crease is sub-voxel-sharp: point-sampling it at the node grid ALIASES it into
        // degenerate mesh triangles AND a discontinuous (serrated) gradient. A separable TENT low-pass of
        // radius `band_limit` nodes rounds it so it's grid-representable with continuous normals. Disabled
        // (`band_limit == 0`) OR a smooth field (plain fBm, no ridge/erosion) ⇒ the single-tap fast path.
        let sharp = self.params.ridge != 0.0 || self.erosion.enabled;
        let radius = if sharp { self.params.band_limit.max(0.0) } else { 0.0 };
        if radius <= 0.0 {
            for j in 0..=res {
                for i in 0..=res {
                    let wp = field.node_world_xz(i, j);
                    field.set(i, j, self.sample_world(wp.x, wp.y, ctx.seed));
                }
            }
            out.produce(Self::OUTPUT, field);
            return;
        }
        self.generate_band_limited(ctx, &mut field, radius);
        out.produce(Self::OUTPUT, field);
    }
}

impl HeightLayer {
    /// Apply the separable TENT band-limit (radius in node spacings) over the chunk's node grid, writing
    /// the low-passed `(h, dh/dx, dh/dz)` into `field`. The KEY properties:
    ///
    /// - **Seam-free**: it resamples the world-anchored, continuous [`sample_world`] on a FINE grid that
    ///   extends an APRON of `⌈radius⌉` nodes past every chunk edge, with a symmetric kernel — so a chunk's
    ///   boundary node convolves the identical world samples as the neighbour's, bit-for-bit (the §10
    ///   padding-correctness property, asserted by `adjacent_chunks_agree_on_shared_boundary`).
    /// - **Gradient-consistent**: height AND gradient are filtered by the SAME kernel; since convolution
    ///   commutes with differentiation (`∇(K∗h) = K∗∇h`), the stored gradient stays the exact gradient of
    ///   the band-limited height — so the terrain normals the bake reconstructs from it match the meshed
    ///   surface (no shading/geometry mismatch).
    /// - **Bit-portable**: the tent weights are rationals (`(K+1−|t|)/(K+1)²`) and the accumulation order
    ///   is fixed — no transcendentals, no fast-math — so shared-seed clients agree (the parity contract).
    /// - **Cheap**: `sample_world` is evaluated once per apron-padded NODE (`(res+2·apron+1)²` ≈ the plain
    ///   point-sample count, NOT a multiple of it — the analytic field needs no supersampling), with
    ///   precomputed tent weights and a separable two-pass convolution. Far below the old 9× box supersample.
    fn generate_band_limited(&self, ctx: &GenCtx, field: &mut ScalarField2D, radius: f32) {
        // Sample at NODE resolution (no sub-node supersampling): `sample_world` returns ANALYTIC values +
        // gradients, so there's no finite-difference aliasing to supersample away — a tent average of the
        // analytic field over `±kf` nodes IS the band-limit. This keeps the `sample_world` count at
        // `~(res+2·apron)²` (≈ the plain point-sample count), not a multiple of it.
        let res = HEIGHT_FIELD_RES as i32;
        let spacing = field.node_spacing; // already f64
        let kf = radius.ceil() as i32; // tent half-width, in nodes
        let ap = kf; // apron = kernel support (seam-free: a boundary node reads only genuine apron samples)

        // PRECOMPUTED tent weights `(K+1−|t|)/(K+1)²` over `t ∈ [−kf, kf]` — bit-portable rationals summing
        // to 1, computed ONCE (not a division per tap, which dominated the old cost).
        let denom = ((kf + 1) * (kf + 1)) as f64;
        let w: Vec<f64> = (-kf..=kf).map(|t| (((kf + 1) - t.abs()) as f64) / denom).collect();

        let npa = (res + 2 * ap + 1) as usize; // node samples per axis incl. apron
        let n00 = field.node_world_xz(0, 0); // chunk's mip-0 node (0,0) world XZ
        let ox = n00.x - ap as f64 * spacing;
        let oz = n00.y - ap as f64 * spacing;

        // Sample the composed surface at every (apron-padded) node position — packed `[h, gx, gz]`.
        let count = npa * npa;
        let mut grid = vec![[0.0f64; 3]; count];
        for gz in 0..npa {
            let wz = oz + gz as f64 * spacing;
            for gx in 0..npa {
                let n = self.sample_world(ox + gx as f64 * spacing, wz, ctx.seed);
                grid[gz * npa + gx] = [n.height as f64, n.dh_dx as f64, n.dh_dz as f64];
            }
        }

        // Separable pass 1 — convolve along X into a temporary (edges clamp but are never read by node
        // positions, which sit `ap = kf` samples in from each edge ⇒ node results only ever read genuine
        // apron samples, never a clamped value ⇒ seam-free).
        let mut tx = vec![[0.0f64; 3]; count];
        for gz in 0..npa {
            let row = gz * npa;
            for gx in 0..npa {
                let mut acc = [0.0f64; 3];
                for (wi, t) in (-kf..=kf).enumerate() {
                    let sx = (gx as i32 + t).clamp(0, npa as i32 - 1) as usize;
                    let s = grid[row + sx];
                    let ww = w[wi];
                    acc[0] += s[0] * ww;
                    acc[1] += s[1] * ww;
                    acc[2] += s[2] * ww;
                }
                tx[row + gx] = acc;
            }
        }

        // Separable pass 2 — convolve along Z, evaluated ONLY at node positions. Node (i,j) ↔ grid (i+ap, j+ap).
        for j in 0..=HEIGHT_FIELD_RES {
            let gz0 = j as i32 + ap;
            for i in 0..=HEIGHT_FIELD_RES {
                let gx = (i as i32 + ap) as usize;
                let mut acc = [0.0f64; 3];
                for (wi, t) in (-kf..=kf).enumerate() {
                    let sz = (gz0 + t).clamp(0, npa as i32 - 1) as usize;
                    let s = tx[sz * npa + gx];
                    let ww = w[wi];
                    acc[0] += s[0] * ww;
                    acc[1] += s[1] * ww;
                    acc[2] += s[2] * ww;
                }
                field.set(i, j, HeightNode { height: acc[0] as f32, dh_dx: acc[1] as f32, dh_dz: acc[2] as f32 });
            }
        }
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

    /// With ridge/erosion on, `generate` BAND-LIMITS the composed surface (the finalize tent low-pass): a
    /// node is a CONVEX COMBINATION (weighted average) of `sample_world` over its kernel neighbourhood, so
    /// it (a) differs from the raw point sample (the low-pass is doing something) and (b) lies within the
    /// local [min, max] of the surface (a low-pass can never overshoot — the property that rounds the sharp
    /// crest instead of spiking it). Seam-freeness is asserted separately by
    /// `adjacent_chunks_agree_on_shared_boundary` (which uses this same `layer()`).
    #[test]
    fn generate_band_limits_sharp_features() {
        let l = layer(); // default: ridge + erosion + band_limit > 0
        assert!(l.params.band_limit > 0.0, "default layer must band-limit");
        let coord = ChunkCoord::new(l.id(), IVec3::new(1, 0, -2));
        let ctx = GenCtx { coord, seed: 7, size: l.chunk_size() };
        let mut out = GenOutput::default();
        l.generate(&ctx, &mut out);
        let field = out.take::<ScalarField2D>(HeightLayer::OUTPUT).unwrap();
        let (i, j) = (10u32, 20u32);
        let wp = field.node_world_xz(i, j);
        let node = field.node(i, j);
        assert!(node.height.is_finite() && node.dh_dx.is_finite() && node.dh_dz.is_finite());

        // Bound the band-limited node by the local surface range over the kernel footprint (±radius nodes).
        let spacing = field.node_spacing; // already f64
        let r = l.params.band_limit.ceil() as i32;
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for sj in -r..=r {
            for si in -r..=r {
                let s = l.sample_world(wp.x + si as f64 * spacing, wp.y + sj as f64 * spacing, ctx.seed);
                lo = lo.min(s.height);
                hi = hi.max(s.height);
            }
        }
        assert!(
            node.height >= lo - 1e-2 && node.height <= hi + 1e-2,
            "band-limited node {} must lie within local surface range [{lo}, {hi}] (a low-pass can't overshoot)",
            node.height,
        );
        // And it actually differs from the raw point sample (the band-limit is active).
        let point = l.sample_world(wp.x, wp.y, ctx.seed);
        assert_ne!(node.height.to_bits(), point.height.to_bits(), "band-limited node differs from point sample");
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

    /// GEN-PERF: full-chunk [`HeightLayer::generate`] cost — the band-limit finalize stage vs the plain
    /// point-sample fast path — plus the `sample_world` call count each makes. `#[ignore]` — run with
    /// `--release --ignored --nocapture`. Confirms the separable band-limit is NOT a gen-perf regression:
    /// it evaluates `sample_world` once per FINE sample (`((res+2·apron)·D+1)²`, independent of kernel
    /// radius), which for the default radius is COMPARABLE to (and below the old 9× box supersample's)
    /// `9·(res+1)²` — the convolution itself is cheap float work on the cached grid.
    #[test]
    #[ignore = "gen-perf bench; run with --release --ignored --nocapture"]
    fn bench_generate_chunk() {
        let coord = ChunkCoord::new(LayerId(0), IVec3::new(3, 0, -5));
        let seed = 4242u64;
        let nodes = ((HEIGHT_FIELD_RES + 1) * (HEIGHT_FIELD_RES + 1)) as f64;

        let run = |label: &str, l: &HeightLayer| {
            let size = l.chunk_size();
            // Warm + timed runs (generate allocates the fine grid; amortise the allocator).
            for _ in 0..2 {
                let mut o = GenOutput::default();
                l.generate(&GenCtx { coord, seed, size }, &mut o);
            }
            let reps = 8;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                let mut o = GenOutput::default();
                l.generate(&GenCtx { coord, seed, size }, &mut o);
            }
            let us = t.elapsed().as_micros() as f64 / reps as f64;
            eprintln!("GEN-PERF [{label}]: {us:.0} µs/chunk ({:.1} ns/node)", us * 1000.0 / nodes);
            us
        };

        let band = run("band-limit (default)", &layer());
        let plain = run("plain point-sample", &plain_layer());
        // The old box supersample evaluated `sample_world` 9× per node; the band-limit's fine grid is fewer
        // evals than that, so the band-limit must stay within a small multiple of the plain path (NOT the
        // ~9× the old supersample cost). Generous bound — this is a regression tripwire, not a tight gate.
        assert!(band < plain * 6.0, "band-limit gen {band:.0}µs must stay well under 9× plain {plain:.0}µs");
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
