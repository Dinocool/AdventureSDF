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
use super::super::biome::{BIOME_COUNT, BiomeId, biome_blend_weight_grad};
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
    /// Per-biome SHAPE override graphs (index = `BiomeId as usize`); `None` ⇒ use the default [`graph`]. When
    /// ANY override is set, [`sample_world`](Self::sample_world) blends the primary+secondary biomes' shape
    /// graphs by the climate weight ([`biome_blend_weight_grad`]) — "biomes own their terrain shape". When ALL
    /// are `None` ([`is_single`](Self::is_single)) the layer behaves EXACTLY as the single-graph path
    /// (bit-identical), so attaching the registry with no overrides is a no-op. Shared `Arc`s ⇒ every tier
    /// blends the SAME set (cross-tier agreement).
    biome_shapes: [Option<Arc<Graph>>; BIOME_COUNT],
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
            biome_shapes: [const { None }; BIOME_COUNT],
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

    /// Attach the per-biome SHAPE override graphs (builder style; see [`biome_shapes`](Self::biome_shapes)).
    /// Index = `BiomeId as usize`; `None` keeps the default [`graph`]. Over-size graphs are rejected (kept
    /// `None`). All-`None` ⇒ [`is_single`](Self::is_single) ⇒ the single-graph path (bit-identical).
    pub fn with_biome_shapes(mut self, shapes: [Option<Arc<Graph>>; BIOME_COUNT]) -> Self {
        self.biome_shapes = shapes.map(|g| g.filter(|g| g.nodes.len() <= MAX_GRAPH_NODES));
        self
    }

    /// True iff no per-biome shape override is set — the layer is a single graph (or legacy fBm), evaluated
    /// WITHOUT the biome blend (bit-identical to the pre-registry behaviour). The hot-path fast case.
    #[inline]
    fn is_single(&self) -> bool {
        self.biome_shapes.iter().all(Option::is_none)
    }

    /// The shape graph for `biome`: its override if set, else the default [`graph`].
    #[inline]
    fn shape_of(&self, biome: BiomeId) -> Option<&Arc<Graph>> {
        self.biome_shapes[biome as usize].as_ref().or(self.graph.as_ref())
    }

    /// Evaluate one shape graph at world `(wx, wz)` → its output [`Field`] (`height, dh/dx, dh/dz`). The
    /// per-sample stack scratch keeps the bake hot path alloc-free.
    #[inline]
    fn eval_graph_field(g: &Graph, wx: f64, wz: f64, world_seed: u64) -> Field {
        let n = g.nodes.len();
        debug_assert!(n <= MAX_GRAPH_NODES);
        let mut scratch = [Field::constant(0.0); MAX_GRAPH_NODES];
        g.eval_into(wx, wz, world_seed, &mut scratch[..n])
    }

    /// BLENDED surface: the climate-weighted blend of the primary+secondary biomes' shape graphs at
    /// `(wx, wz)` (only reached when a per-biome override exists — see [`biome_shapes`](Self::biome_shapes)).
    /// `Field::mix` carries the weight gradient (product rule) so the terrain normal stays analytic. If the
    /// two biomes resolve to the SAME graph (or the weight is 0) it evals ONCE — bit-identical to single.
    fn sample_blended(&self, wx: f64, wz: f64, world_seed: u64) -> HeightNode {
        let (prim, sec, w, dwdx, dwdz) = biome_blend_weight_grad(wx, wz, world_seed);
        let Some(ga) = self.shape_of(prim) else {
            // No graph at all (default None + no override) → legacy fBm fallback for this point.
            return self.sample_world_legacy(wx, wz, world_seed);
        };
        let fa = Self::eval_graph_field(ga, wx, wz, world_seed);
        let blended = match self.shape_of(sec) {
            Some(gb) if w > 0.0 && !Arc::ptr_eq(ga, gb) => {
                let fb = Self::eval_graph_field(gb, wx, wz, world_seed);
                fa.mix(fb, Field { v: w, dx: dwdx, dz: dwdz })
            }
            _ => fa, // same graph or zero weight → primary only (bit-identical to single)
        };
        HeightNode { height: blended.v as f32, dh_dx: blended.dx as f32, dh_dz: blended.dz as f32 }
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
        // BIOME-SHAPE BLEND: when a per-biome shape override is set, blend the primary+secondary biomes'
        // shape graphs by the climate weight ("biomes own their terrain shape"). All-`None` ⇒ the single
        // path below (bit-identical to pre-registry).
        if !self.is_single() {
            return self.sample_blended(wx, wz, world_seed);
        }
        // GRAPH PATH: when a biome terrain graph is attached, it IS the surface — evaluate it with
        // forward-mode autodiff (the output `Field` is `(height, dh_dx, dh_dz)` directly). Pure +
        // bit-portable; a fixed stack scratch keeps the bake hot path alloc-free.
        if let Some(g) = &self.graph {
            let f = Self::eval_graph_field(g, wx, wz, world_seed);
            return HeightNode { height: f.v as f32, dh_dx: f.dx as f32, dh_dz: f.dz as f32 };
        }
        self.sample_world_legacy(wx, wz, world_seed)
    }

    /// The legacy (no-graph) fBm/ridge/erosion surface — the closed-form analytic path. Factored out so the
    /// blend path can fall back to it for a biome with no graph at all.
    #[inline]
    fn sample_world_legacy(&self, wx: f64, wz: f64, world_seed: u64) -> HeightNode {
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
        // BIOME-SHAPE BLEND: a per-biome override is set → no single graph spans the column; blend per point
        // (`sample_blended`). Most chunks are single-biome (km-scale climate), so this is the rare path; the
        // common single case keeps the columnar fast path below.
        if !self.is_single() {
            for p in 0..xs.len() {
                out[p] = self.sample_blended(xs[p], zs[p], world_seed);
            }
            return;
        }
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
mod tests;
