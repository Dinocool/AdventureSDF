//! The Phase-1 base-height layer: CPU-authoritative, 2D, integer-hash fBm over the XZ plane.
//!
//! This is the root of the dependency DAG (no dependencies) and the terrain seed — it generalizes the
//! old analytic `SdfPrimitive::Heightmap` noise into an artifact the bake samples (WORLD_GEN_PLAN §9
//! phase 1 / §3). Authoritative ⇒ generated with the bit-portable [`super::super::noise`] basis so
//! shared-seed multiplayer clients agree (§2.8); the `worldgen_parity` harness pins its outputs.

use bevy::prelude::*;

use std::sync::Arc;

use super::super::artifact::{ArtifactKind, HeightNode, ScalarField2D};
use super::super::coord::{Authority, ChunkSize, Dim, LayerId};
use super::super::graph::preset::MAX_GRAPH_NODES;
use super::super::graph::node::GridOut;
use super::super::graph::{Field, Graph};
use super::super::layer::{ArtifactDecl, GenCtx, GenOutput, Layer};
use super::super::noise::{FbmParams, fbm_height_grad, fbm_height_grad_hess};
use super::erosion::{ErosionParams, erode_with_grad};

/// Bump when the height layer's *output* intentionally changes (algorithm/constants). It keys the
/// disk cache (WORLD_GEN_PLAN §2.3) and the parity reference vectors — a change here forces
/// regenerating reference values, making "I meant to change the terrain" explicit and review-visible.
pub const HEIGHT_GEN_VERSION: u32 = 8;

/// Chunk edge in base cells (= metres) for the height layer's tier.
pub const HEIGHT_CHUNK_CELLS: u32 = 128;
/// Cells per axis the height field is sampled at within a chunk (nodes = res + 1). `128 / 64 = 2 m`
/// node spacing — the authoritative base resolution; the GPU adds finer cosmetic detail on top.
pub const HEIGHT_FIELD_RES: u32 = 64;

/// FIXED world tap step of the surface band-limit (= the tier-0 node spacing, `128 / 64 = 2 m`). The
/// band-limit kernel samples [`HeightLayer::sample_world`] at multiples of THIS step at EVERY tier and
/// in the point-evaluable hi-fi normal — so all of them evaluate the IDENTICAL band-limited world
/// function (the cross-tier no-seam invariant; see [`HeightLayer::generate`]). Single source of truth.
pub const HEIGHT_BAND_LIMIT_TAP: f64 = HEIGHT_CHUNK_CELLS as f64 / HEIGHT_FIELD_RES as f64;

/// The separable TENT kernel weights `(K+1−|t|)/(K+1)²` over `t ∈ [−kf, kf]` — bit-portable rationals
/// summing to 1, in FIXED `t = −kf..=kf` order. THE single source of truth for the band-limit kernel:
/// [`HeightLayer::generate_band_limited`] (the chunk-grid finalize, applied only when the `band_limit`
/// slider is > 0) builds the kernel here. Deterministic / bit-portable (rational weights, fixed order, no
/// transcendentals).
#[inline]
pub fn band_limit_weights(kf: i32) -> Vec<f64> {
    let denom = ((kf + 1) * (kf + 1)) as f64;
    (-kf..=kf).map(|t| (((kf + 1) - t.abs()) as f64) / denom).collect()
}

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
            // SURFACE BAND-LIMIT OFF by default (`0` = point-sample the raw `sample_world` surface, no tent
            // finalize). The knob is retained + runtime-tunable via the World Gen panel's "Crest band-limit"
            // (it still rounds sharp crests over `~2·radius` nodes when raised), but the default no longer
            // low-passes the meshed height — the terrain meshes the raw surface. Reversible: set > 0 to
            // re-enable. The triangle-quality harness still shows crest normal-spread falling monotonically
            // with the radius when it is raised (dense ridge: 143°→91°→59°→30° at radius 0→2→4→8).
            band_limit: 0.0,
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
    /// The active biome terrain node-graph. When `Some`, [`sample_world`](Self::sample_world) evaluates
    /// it (the new biome-driven surface) instead of the legacy fBm+ridge+erosion path. Shared `Arc` so
    /// every tier samples the SAME graph (the cross-tier-agreement invariant). `None` ⇒ legacy path
    /// (tests / pre-graph fallback). Set by the `LayerManager` from the `WorldGraph` resource.
    graph: Option<Arc<Graph>>,
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
            graph: None,
            decls: [ArtifactDecl { name: Self::OUTPUT, kind: ArtifactKind::ScalarField2D }],
        }
    }

    /// Attach (or clear) the biome terrain graph this tier samples (builder style; see the `graph`
    /// field). A graph with more than [`MAX_GRAPH_NODES`] nodes is rejected (kept `None`) since the
    /// per-sample evaluator uses a fixed stack scratch — callers validate/size-check before this.
    pub fn with_graph(mut self, graph: Option<Arc<Graph>>) -> Self {
        self.graph = graph.filter(|g| g.nodes.len() <= MAX_GRAPH_NODES);
        self
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
        // GRAPH PATH: when a biome terrain graph is attached, it IS the surface — evaluate it with
        // forward-mode autodiff (the output `Field` is `(height, dh_dx, dh_dz)` directly). Pure +
        // bit-portable; a fixed stack scratch keeps the bake hot path alloc-free.
        if let Some(g) = &self.graph {
            let n = g.nodes.len();
            debug_assert!(n <= MAX_GRAPH_NODES);
            let mut scratch = [Field::constant(0.0); MAX_GRAPH_NODES];
            let f = g.eval_into(wx, wz, world_seed, &mut scratch[..n]);
            return HeightNode { height: f.v as f32, dh_dx: f.dx as f32, dh_dz: f.dz as f32 };
        }

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

    /// Batch-evaluate the surface over the world coordinate columns `xs`/`zs` into `out` (parallel
    /// slices, all the same length). When a biome graph is attached this uses the COLUMNAR
    /// [`Graph::eval_grid`] (the gen hot path — match the node kinds once, loop per point), which is
    /// **bit-for-bit identical** to calling [`sample_world`](Self::sample_world) per point (the graph
    /// branch of `sample_world` is exactly `Field` → f32). Without a graph it falls back to the
    /// per-point `sample_world` (the legacy fBm/erosion path stays scalar — it's not the production
    /// path and the per-point closed-form already amortizes well).
    fn sample_world_grid(&self, xs: &[f64], zs: &[f64], world_seed: u64, out: &mut [HeightNode]) {
        debug_assert_eq!(xs.len(), zs.len());
        debug_assert_eq!(xs.len(), out.len());
        if let Some(g) = &self.graph {
            let n = g.nodes.len();
            let npts = xs.len();
            // Node-major column scratch (reused for the whole grid; one alloc per chunk).
            let mut scratch = vec![Field::constant(0.0); n * npts];
            let mut v = vec![0.0f64; npts];
            let mut dx = vec![0.0f64; npts];
            let mut dz = vec![0.0f64; npts];
            g.eval_grid(xs, zs, world_seed, &mut scratch, GridOut { v: &mut v, dx: &mut dx, dz: &mut dz });
            for p in 0..npts {
                out[p] = HeightNode { height: v[p] as f32, dh_dx: dx[p] as f32, dh_dz: dz[p] as f32 };
            }
            return;
        }
        for p in 0..xs.len() {
            out[p] = self.sample_world(xs[p], zs[p], world_seed);
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

        // SURFACE BAND-LIMIT — the single finalize stage over the WHOLE composed surface. A sharp ridge
        // crest / erosion crease is sub-voxel-sharp: point-sampling it at the node grid ALIASES it into
        // degenerate mesh triangles AND a discontinuous (serrated) gradient. A separable TENT low-pass
        // rounds it so it's grid-representable with continuous normals.
        //
        // CROSS-TIER CONSISTENCY (FOOLPROOF, no LOD seam by construction): the kernel is sampled at a FIXED
        // 2 m world tap step (the tier-0 node spacing) at EVERY tier, with the SAME world half-width
        // (`band_limit` × 2 m). So every clipmap tier evaluates the IDENTICAL band-limited world function
        // `f_bl(x,z)` at its nodes — and at any node two tiers SHARE they agree bit-for-bit (same world taps
        // ⇒ same f64 convolution ⇒ same value). That restores the pre-band-limit invariant ("all tiers
        // sample one world function") that the seam-free design rests on. `kf` is the half-width in 2 m
        // taps, CONSTANT across tiers.
        //
        // BOUNDED COST: applied only where the band-limit is meaningful — a tier whose node spacing already
        // exceeds `2·width` smooths ≥ the band-limit by its own (coarse) sampling, so `f_bl ≈ raw` there →
        // point-sample. This caps the fixed-2 m grid to the near tiers (it would explode on the huge far
        // chunks otherwise). Plain fBm (no ridge/erosion) also point-samples.
        let tap = HEIGHT_BAND_LIMIT_TAP; // FIXED 2 m tap step (tier-0 spacing) — the shared SSOT
        let kf = self.band_limit_kf(); // SHARED `sharp`/`radius` gate (also used by the hi-fi gradient)
        let world_w = kf as f64 * tap;
        if kf < 1 || field.node_spacing >= 2.0 * world_w {
            // Build the (res+1)² node coordinate columns, batch-evaluate the surface once, write back.
            let side = (res + 1) as usize;
            let count = side * side;
            let mut xs = vec![0.0f64; count];
            let mut zs = vec![0.0f64; count];
            for j in 0..=res {
                for i in 0..=res {
                    let wp = field.node_world_xz(i, j);
                    let idx = j as usize * side + i as usize;
                    xs[idx] = wp.x;
                    zs[idx] = wp.y;
                }
            }
            let mut nodes = vec![HeightNode::default(); count];
            self.sample_world_grid(&xs, &zs, ctx.seed, &mut nodes);
            for j in 0..=res {
                for i in 0..=res {
                    field.set(i, j, nodes[j as usize * side + i as usize]);
                }
            }
            out.produce(Self::OUTPUT, field);
            return;
        }
        self.generate_band_limited(ctx, &mut field, kf);
        out.produce(Self::OUTPUT, field);
    }
}

impl HeightLayer {
    /// The half-width of the surface band-limit in FIXED 2 m taps (`kf`), or `0` when the surface has no
    /// sharp features (plain fBm, no ridge/erosion ⇒ no band-limit) — exactly the `sharp`/`radius` gate
    /// [`generate`](Self::generate) applies. THE single decision of "does this surface band-limit, and how
    /// wide", shared by `generate` and the hi-fi gradient so the rendered normal matches the meshed height.
    #[inline]
    fn band_limit_kf(&self) -> i32 {
        let sharp = self.params.ridge != 0.0 || self.erosion.enabled;
        let radius = if sharp { self.params.band_limit.max(0.0) } else { 0.0 };
        if radius > 0.0 { radius.round() as i32 } else { 0 }
    }

    /// Apply the separable TENT band-limit at a FIXED 2 m world tap step over `±kf` taps, writing the
    /// low-passed `(h, dh/dx, dh/dz)` into `field`. `kf` (half-width in 2 m taps) is CONSTANT across tiers,
    /// so every tier evaluates the identical band-limited world function (see `generate`). KEY properties:
    ///
    /// - **Seam-free, ALL boundaries**: the kernel samples the world-anchored, continuous [`sample_world`]
    ///   at fixed 2 m world positions with a symmetric kernel. A node shared by two CHUNKS (same tier) OR
    ///   two TIERS (a coarse tier's node coincides with a finer tier's) lands on the SAME world taps ⇒ the
    ///   SAME f64 convolution ⇒ bit-for-bit equal — no within-tier seam AND no cross-tier LOD seam.
    /// - **Gradient-consistent**: height AND gradient are filtered by the SAME kernel; since convolution
    ///   commutes with differentiation (`∇(K∗h) = K∗∇h`), the stored gradient stays the exact gradient of
    ///   the band-limited height — so the terrain normals the bake reconstructs from it match the surface.
    /// - **Bit-portable**: tent weights are rationals (`(K+1−|t|)/(K+1)²`), fixed accumulation order — no
    ///   transcendentals/fast-math — so shared-seed clients agree (the parity contract).
    /// - **Bounded cost**: a fine 2 m grid over the chunk + `kf`-tap apron, separable two-pass convolution,
    ///   then the tier's nodes (each a multiple of 2 m) read straight off it. The caller only invokes this
    ///   where the band-limit is meaningful (node spacing < 2·width), so the 2 m grid stays small (≤ tier 2
    ///   for the default radius); coarser tiers point-sample.
    fn generate_band_limited(&self, ctx: &GenCtx, field: &mut ScalarField2D, kf: i32) {
        let res = HEIGHT_FIELD_RES as i32;
        let tap = HEIGHT_BAND_LIMIT_TAP; // FIXED 2 m tap step (tier-0 spacing) — the shared SSOT
        let step = (field.node_spacing / tap).round() as i32; // fine 2 m samples per node = 2^tier (≥ 1)
        let ap = kf; // apron in FINE (2 m) taps — exactly the kernel support ⇒ node results never clamp

        // PRECOMPUTED tent weights — built by the SHARED [`band_limit_weights`] (the same kernel the
        // point-evaluable hi-fi gradient uses, so they can never drift). Bit-portable rationals summing to 1.
        let w: Vec<f64> = band_limit_weights(kf);

        let npa = (res * step + 2 * ap + 1) as usize; // FINE (2 m) samples per axis over chunk + apron
        let n00 = field.node_world_xz(0, 0); // chunk's node (0,0) world XZ
        let ox = n00.x - ap as f64 * tap;
        let oz = n00.y - ap as f64 * tap;

        // Sample the composed surface on the fixed 2 m world grid — packed `[h, gx, gz]`. Build the
        // fine-grid coordinate columns once and batch-evaluate the surface (columnar graph eval).
        let count = npa * npa;
        let mut xs = vec![0.0f64; count];
        let mut zs = vec![0.0f64; count];
        for fz in 0..npa {
            let wz = oz + fz as f64 * tap;
            for fx in 0..npa {
                let idx = fz * npa + fx;
                xs[idx] = ox + fx as f64 * tap;
                zs[idx] = wz;
            }
        }
        let mut nodes = vec![HeightNode::default(); count];
        self.sample_world_grid(&xs, &zs, ctx.seed, &mut nodes);
        let mut grid = vec![[0.0f64; 3]; count];
        for (g, n) in grid.iter_mut().zip(nodes.iter()) {
            *g = [n.height as f64, n.dh_dx as f64, n.dh_dz as f64];
        }

        // Separable pass 1 — convolve along X into a temporary (edges clamp but are never read by node
        // positions, which sit `ap = kf` taps in from each edge ⇒ node results only ever read genuine apron
        // samples ⇒ seam-free).
        let mut tx = vec![[0.0f64; 3]; count];
        for fz in 0..npa {
            let row = fz * npa;
            for fx in 0..npa {
                let mut acc = [0.0f64; 3];
                for (wi, t) in (-kf..=kf).enumerate() {
                    let sx = (fx as i32 + t).clamp(0, npa as i32 - 1) as usize;
                    let s = grid[row + sx];
                    let ww = w[wi];
                    acc[0] += s[0] * ww;
                    acc[1] += s[1] * ww;
                    acc[2] += s[2] * ww;
                }
                tx[row + fx] = acc;
            }
        }

        // Separable pass 2 — convolve along Z, evaluated ONLY at node positions. Node (i,j) ↔ fine sample
        // (i·step + ap, j·step + ap) (the node lands exactly on a 2 m tap since node spacing is a multiple).
        for j in 0..=HEIGHT_FIELD_RES {
            let fz0 = j as i32 * step + ap;
            for i in 0..=HEIGHT_FIELD_RES {
                let fx = (i as i32 * step + ap) as usize;
                let mut acc = [0.0f64; 3];
                for (wi, t) in (-kf..=kf).enumerate() {
                    let sz = (fz0 + t).clamp(0, npa as i32 - 1) as usize;
                    let s = tx[sz * npa + fx];
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

    /// CROSS-TIER CONSISTENCY GUARD — the band-limit must NOT introduce an LOD seam. Adjacent clipmap
    /// tiers (tier 0 @ 2 m nodes, tier 1 @ 4 m nodes) must agree on the band-limited surface at the world
    /// nodes they SHARE (multiples of 4 m), or a tier-coverage boundary shows a crack + shading kink. This
    /// is the guard for the fixed-WORLD band-limit width (a node-relative width — the seam bug — would make
    /// tier 1 smooth ~2× wider than tier 0, blowing up the mismatch here). Uses dense sharp ridges (worst
    /// case). The existing `terrain_2to1_*` harness only tests cross-MIP within ONE tier, so it missed this.
    #[test]
    fn tiers_agree_on_shared_nodes_after_band_limit() {
        use bevy::math::Vec3;
        let params = HeightParams {
            ridge: 1.0,
            base_freq: 1.0 / 256.0,
            amplitude: 100.0,
            octaves: 4,
            band_limit: 3.0,
            ..Default::default()
        };
        let erosion = ErosionParams { enabled: false, ..Default::default() };
        let seed = 7u64;
        let t0 = HeightLayer::new_tier(LayerId(0), params, erosion, HEIGHT_CHUNK_CELLS);
        let t1 = HeightLayer::new_tier(LayerId(0), params, erosion, HEIGHT_CHUNK_CELLS * 2);
        let gen_chunk0 = |l: &HeightLayer| {
            let mut o = GenOutput::default();
            l.generate(&GenCtx { coord: ChunkCoord::new(l.id(), IVec3::ZERO), seed, size: l.chunk_size() }, &mut o);
            o.take::<ScalarField2D>(HeightLayer::OUTPUT).unwrap()
        };
        let (f0, f1) = (gen_chunk0(&t0), gen_chunk0(&t1));
        // Shared world nodes: tier 0 chunk spans [0,128] (nodes every 2 m), tier 1 spans [0,256] (every
        // 4 m). A world coord that is a multiple of 4 m in [4,124] is a node in BOTH (skip the very edge so
        // both have the band-limit apron). tier-0 node index = w/2, tier-1 = w/4.
        let amp = (params.amplitude_sum()) as f32;
        let (mut worst_h, mut worst_dot) = (0.0f32, 1.0f32);
        for kz in 1..=30 {
            for kx in 1..=30 {
                let (wx, wz) = (4 * kx, 4 * kz);
                let n0 = f0.node((wx / 2) as u32, (wz / 2) as u32);
                let n1 = f1.node((wx / 4) as u32, (wz / 4) as u32);
                worst_h = worst_h.max((n0.height - n1.height).abs());
                let g0 = Vec3::new(-n0.dh_dx, 1.0, -n0.dh_dz).normalize();
                let g1 = Vec3::new(-n1.dh_dx, 1.0, -n1.dh_dz).normalize();
                worst_dot = worst_dot.min(g0.dot(g1));
            }
        }
        println!(
            "CROSS-TIER: worst |Δh|={worst_h:.3}m ({:.2}% of amp {amp:.0}m), worst normal dot={worst_dot:.4} \
             ({:.1}deg)",
            100.0 * worst_h / amp,
            worst_dot.clamp(-1.0, 1.0).acos().to_degrees(),
        );
        // FIXED 2 m world taps at every tier ⇒ tier 0 and tier 1 evaluate the IDENTICAL band-limited world
        // function at the nodes they share ⇒ they agree to f32 round-off (NOT just "closely"). This is the
        // foolproof no-seam guarantee; a node-relative or per-tier-tap band-limit would smash both
        // (Δh ~ metres+, dot well below 1). Tolerances are f32-epsilon-tight.
        assert!(worst_h < 0.05, "cross-tier height seam: |Δh|={worst_h:.4}m (tiers must agree at shared nodes)");
        assert!(worst_dot > 0.9999, "cross-tier normal seam: worst dot {worst_dot:.5} (tiers must agree)");
        let _ = amp;
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

    /// A layer with the band-limit finalize EXPLICITLY enabled (`band_limit = 3`). The DEFAULT is now `0`
    /// (point-sample the raw surface), so this explicitly exercises the still-supported `generate_band_limited`
    /// finalize the slider drives — without depending on the default.
    fn band_limited_layer() -> HeightLayer {
        HeightLayer::new(
            LayerId(0),
            HeightParams { band_limit: 3.0, ..Default::default() },
            ErosionParams::default(),
        )
    }

    /// With ridge/erosion on AND `band_limit > 0`, `generate` BAND-LIMITS the composed surface (the finalize
    /// tent low-pass): a node is a CONVEX COMBINATION (weighted average) of `sample_world` over its kernel
    /// neighbourhood, so it (a) differs from the raw point sample (the low-pass is doing something) and (b)
    /// lies within the local [min, max] of the surface (a low-pass can never overshoot — the property that
    /// rounds the sharp crest instead of spiking it). Uses an EXPLICITLY band-limited layer (the default is
    /// now `band_limit = 0` / raw); this pins the still-supported slider finalize path.
    #[test]
    fn generate_band_limits_sharp_features() {
        let l = band_limited_layer(); // ridge + erosion + band_limit = 3 (slider on)
        assert!(l.params.band_limit > 0.0, "this test must use a band-limited layer");
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
