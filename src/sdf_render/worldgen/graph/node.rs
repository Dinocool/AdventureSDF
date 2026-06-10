//! The node library + graph evaluator. A [`Graph`] is a DAG of [`Node`]s; each node computes a
//! [`Field`] (value + analytic gradient) at one world point from its inputs, so evaluating the graph
//! forward-mode-autodiffs to the terrain `(height, dh_dx, dh_dz)`. Nodes are stored in topological
//! order (each input references an EARLIER node), so a single forward pass over a scratch buffer
//! evaluates the whole graph.
//!
//! Bit-portable/deterministic: every node is `Field` basic ops + the portable noise basis, fixed
//! order. Sources (`WorldX/Z`, `Const`, `Fbm`) ignore their inputs; `Fbm` reads the eval point + folds
//! the world seed with its salt (the exact idiom in `layers::height::fbm_params`).

use super::super::noise::{FbmParams, fbm_height_grad, fbm_height_grad_x4};
use super::super::spline::Spline;
use super::Field;

/// A low-frequency fBm "climate axis" source (continentalness/temperature/humidity/… ) OR the
/// high-frequency carrier detail — the only difference is the params. Samples [`fbm_height_grad`] at
/// the eval point; the world seed is folded with `seed_salt` so each instance is an independent stream.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, bevy::reflect::Reflect)]
pub struct FbmAxis {
    pub octaves: u32,
    pub base_freq: f64,
    pub lacunarity: f64,
    pub gain: f64,
    pub amplitude: f64,
    pub seed_salt: u32,
}

impl FbmAxis {
    #[inline]
    fn params(&self, world_seed: u64) -> FbmParams {
        let seed = (world_seed as u32) ^ ((world_seed >> 32) as u32) ^ self.seed_salt;
        FbmParams {
            octaves: self.octaves,
            base_freq: self.base_freq,
            lacunarity: self.lacunarity,
            gain: self.gain,
            amplitude: self.amplitude,
            seed,
        }
    }
}

/// What a node computes. Source kinds (`WorldX/Z`, `Const`, `Fbm`) take 0 inputs; the rest take 1, 2,
/// or 3 (see [`NodeKind::arity`]). Most are thin wrappers over [`Field`] ops, so their gradient comes
/// from the autodiff for free; `Fbm`/`Curve`/`Ridge` are the load-bearing ones.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, bevy::reflect::Reflect)]
pub enum NodeKind {
    // --- sources (arity 0) ---
    WorldX,
    WorldZ,
    Const(f64),
    Fbm(FbmAxis),
    // --- unary (arity 1) ---
    Curve(Spline),
    /// Ridged-multifractal fold of the input: `in + ridge·((amp_sum − 2|in|) − in)` (`amp_sum` = the
    /// input's expected swing). Sharpens peaks; gradient via the `Field` ops (autodiff).
    Ridge { ridge: f64, amp_sum: f64 },
    Clamp { lo: f64, hi: f64 },
    Smoothstep { edge0: f64, edge1: f64 },
    Scale(f64),
    Offset(f64),
    Abs,
    Neg,
    // --- binary (arity 2) ---
    Add,
    Sub,
    Mul,
    Min,
    Max,
    // --- ternary (arity 3) ---
    /// `mix(a, b, t)` — linear blend of inputs 0,1 by input 2 (the placement/biome-blend op).
    Mix,
}

impl NodeKind {
    /// How many inputs this node consumes.
    pub const fn arity(&self) -> usize {
        match self {
            NodeKind::WorldX | NodeKind::WorldZ | NodeKind::Const(_) | NodeKind::Fbm(_) => 0,
            NodeKind::Curve(_)
            | NodeKind::Ridge { .. }
            | NodeKind::Clamp { .. }
            | NodeKind::Smoothstep { .. }
            | NodeKind::Scale(_)
            | NodeKind::Offset(_)
            | NodeKind::Abs
            | NodeKind::Neg => 1,
            NodeKind::Add | NodeKind::Sub | NodeKind::Mul | NodeKind::Min | NodeKind::Max => 2,
            NodeKind::Mix => 3,
        }
    }
}

/// One node: its kind + the indices of its input nodes (each strictly less than this node's index —
/// topological order). Unused input slots are ignored per [`NodeKind::arity`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, bevy::reflect::Reflect)]
pub struct Node {
    pub kind: NodeKind,
    pub inputs: [u32; 3],
}

impl Node {
    pub fn source(kind: NodeKind) -> Self {
        Self { kind, inputs: [0, 0, 0] }
    }
    pub fn unary(kind: NodeKind, a: u32) -> Self {
        Self { kind, inputs: [a, 0, 0] }
    }
    pub fn binary(kind: NodeKind, a: u32, b: u32) -> Self {
        Self { kind, inputs: [a, b, 0] }
    }
    pub fn ternary(kind: NodeKind, a: u32, b: u32, c: u32) -> Self {
        Self { kind, inputs: [a, b, c] }
    }
}

/// The three parallel output columns the columnar [`Graph::eval_grid`] writes — the output node's
/// value + its world-XZ gradient, one entry per input point. Bundled so the batched evaluator stays a
/// readable call (and under the arg-count lint). All three slices must be the same length as the input
/// coordinate columns.
pub struct GridOut<'a> {
    /// Output node value per point.
    pub v: &'a mut [f64],
    /// ∂value/∂(world x) per point.
    pub dx: &'a mut [f64],
    /// ∂value/∂(world z) per point.
    pub dz: &'a mut [f64],
}

/// A field node-graph: topologically-ordered nodes + the index of the output node. Evaluated per world
/// point to a [`Field`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, bevy::reflect::Reflect)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub output: u32,
}

/// Why a graph is structurally invalid (caught once at build, not per-sample).
#[derive(Debug, PartialEq, Eq)]
pub enum GraphError {
    Empty,
    OutputOutOfRange,
    /// Node `node` references input `input` which is not strictly earlier (breaks topological order /
    /// would need a cycle).
    NonTopologicalInput { node: usize, input: u32 },
}

impl Graph {
    /// Validate topological order + output range. Cheap; call once before evaluating in the hot path.
    pub fn validate(&self) -> Result<(), GraphError> {
        if self.nodes.is_empty() {
            return Err(GraphError::Empty);
        }
        if self.output as usize >= self.nodes.len() {
            return Err(GraphError::OutputOutOfRange);
        }
        for (i, node) in self.nodes.iter().enumerate() {
            for &inp in &node.inputs[..node.kind.arity()] {
                if inp as usize >= i {
                    return Err(GraphError::NonTopologicalInput { node: i, input: inp });
                }
            }
        }
        Ok(())
    }

    /// CONSERVATIVE bound on `|output value|` via interval propagation over the (topological) nodes —
    /// each node's output magnitude is bounded from its inputs' bounds. Used to size the terrain vertical
    /// AABB band so the tallest peaks the graph can produce don't clip (an over-estimate is safe: the
    /// narrow-band cull still bakes only the thin surface shell, the band just bounds the BVH).
    pub fn value_bound(&self) -> f64 {
        let mut b = vec![0.0f64; self.nodes.len()];
        for (i, node) in self.nodes.iter().enumerate() {
            let a = [
                b[node.inputs[0] as usize],
                b[node.inputs[1] as usize],
                b[node.inputs[2] as usize],
            ];
            b[i] = match node.kind {
                // Raw/scaled coordinates are unbounded; sane terrain graphs gate them through a Curve/
                // Smoothstep/Clamp (all bounded below) before the output, so this only matters if a graph
                // outputs a coord directly — give a large finite bound rather than ∞.
                NodeKind::WorldX | NodeKind::WorldZ => 1.0e6,
                NodeKind::Const(c) => c.abs(),
                NodeKind::Fbm(ax) => {
                    let mut amp = ax.amplitude.abs();
                    let mut sum = 0.0;
                    for _ in 0..ax.octaves {
                        sum += amp;
                        amp *= ax.gain.abs();
                    }
                    sum
                }
                NodeKind::Curve(s) => s.max_abs_y(),
                NodeKind::Ridge { ridge, amp_sum } => {
                    // |in + ridge·((amp_sum − 2|in|) − in)| ≤ |in|(1+3|ridge|) + |ridge|·|amp_sum|.
                    a[0] * (1.0 + 3.0 * ridge.abs()) + ridge.abs() * amp_sum.abs()
                }
                NodeKind::Clamp { lo, hi } => lo.abs().max(hi.abs()),
                NodeKind::Smoothstep { .. } => 1.0,
                NodeKind::Scale(k) => a[0] * k.abs(),
                NodeKind::Offset(k) => a[0] + k.abs(),
                NodeKind::Abs | NodeKind::Neg => a[0],
                NodeKind::Add | NodeKind::Sub => a[0] + a[1],
                NodeKind::Mul => a[0] * a[1],
                NodeKind::Min | NodeKind::Max => a[0].max(a[1]),
                NodeKind::Mix => a[0].max(a[1]), // mix(a,b,t) with a gate t∈[0,1] stays within [a,b]
            };
        }
        b[self.output as usize]
    }

    /// Evaluate the graph at world `(wx, wz)` for `world_seed`, returning the output node's [`Field`].
    /// Assumes [`validate`](Self::validate) passed (topological order); `scratch` is a reusable buffer
    /// of length `nodes.len()` to avoid per-sample allocation in the hot path.
    pub fn eval_into(&self, wx: f64, wz: f64, world_seed: u64, scratch: &mut [Field]) -> Field {
        debug_assert_eq!(scratch.len(), self.nodes.len());
        for (i, node) in self.nodes.iter().enumerate() {
            let a = scratch[node.inputs[0] as usize];
            let b = scratch[node.inputs[1] as usize];
            let c = scratch[node.inputs[2] as usize];
            scratch[i] = eval_node(&node.kind, wx, wz, world_seed, a, b, c);
        }
        scratch[self.output as usize]
    }

    /// Convenience: allocate scratch + evaluate. Tests / cold paths; the hot bake path reuses a buffer.
    pub fn eval(&self, wx: f64, wz: f64, world_seed: u64) -> Field {
        let mut scratch = vec![Field::constant(0.0); self.nodes.len()];
        self.eval_into(wx, wz, world_seed, &mut scratch)
    }

    /// COLUMNAR batched evaluator: evaluate the graph ONCE over a whole column of points
    /// `(xs[p], zs[p])`, writing the OUTPUT node's `(v, dx, dz)` into `out_v/out_dx/out_dz`.
    ///
    /// This is the gen hot path: instead of walking all `n` nodes per point (re-matching every
    /// `NodeKind` and re-touching the per-point scratch for each point), we walk the nodes ONCE,
    /// matching each `NodeKind` a single time, then run a tight per-point loop that computes that
    /// node's `Field` for every point. The match + node-walk overhead is amortized over the column.
    ///
    /// **Bit-for-bit identical to per-point [`eval_into`]** (the determinism contract): every node's
    /// per-point arithmetic dispatches to the SAME [`Field`] ops as the scalar [`eval_node`] — the
    /// `Field` methods are the single source of truth for the math, this just hoists the *dispatch*
    /// (the `match`) out of the per-point loop. No reassociation, no FMA, no f32 accumulation.
    ///
    /// `scratch` is a reusable column buffer of length `nodes.len() * xs.len()` laid out node-major
    /// (`scratch[node * npts + p]`), so a node reads its inputs' already-computed columns. All output
    /// slices and `zs` must have the same length as `xs`.
    pub fn eval_grid(&self, xs: &[f64], zs: &[f64], world_seed: u64, scratch: &mut [Field], out: GridOut<'_>) {
        let npts = xs.len();
        debug_assert_eq!(zs.len(), npts);
        debug_assert_eq!(out.v.len(), npts);
        debug_assert_eq!(out.dx.len(), npts);
        debug_assert_eq!(out.dz.len(), npts);
        debug_assert_eq!(scratch.len(), self.nodes.len() * npts);
        if npts == 0 {
            return;
        }

        for (i, node) in self.nodes.iter().enumerate() {
            // Input column base offsets (sources ignore them per arity, but indexing is harmless since
            // all inputs are strictly-earlier valid node indices once `validate` passed).
            let ia = node.inputs[0] as usize * npts;
            let ib = node.inputs[1] as usize * npts;
            let ic = node.inputs[2] as usize * npts;
            let base = i * npts;

            // Match the kind ONCE, then a tight per-point loop. Each arm calls the SAME `Field` op (or
            // the same source/closed-form) the scalar `eval_node` calls — that's the SSOT for the math.
            // Sources read xs/zs/seed; the rest read their input columns.
            match node.kind {
                NodeKind::WorldX => {
                    for p in 0..npts {
                        scratch[base + p] = Field::world_x(xs[p]);
                    }
                }
                NodeKind::WorldZ => {
                    for p in 0..npts {
                        scratch[base + p] = Field::world_z(zs[p]);
                    }
                }
                NodeKind::Const(v) => {
                    let f = Field::constant(v);
                    for p in 0..npts {
                        scratch[base + p] = f;
                    }
                }
                NodeKind::Fbm(axis) => {
                    // The gen hot path: ~13 octaves of value-noise per grid point. Process the grid 4
                    // points at a time with the bit-identical SIMD octave sum (`fbm_height_grad_x4`),
                    // then the `npts % 4` tail with the scalar `fbm_height_grad`. The x4 path is pinned
                    // `to_bits()`-equal to the scalar one (noise.rs `x4_matches_scalar`), so this stays
                    // bit-for-bit identical to the per-point `eval_into` (the determinism contract).
                    let params = axis.params(world_seed);
                    let chunks = npts / 4;
                    for c in 0..chunks {
                        let p = c * 4;
                        let wx = wide::f64x4::new([xs[p], xs[p + 1], xs[p + 2], xs[p + 3]]);
                        let wz = wide::f64x4::new([zs[p], zs[p + 1], zs[p + 2], zs[p + 3]]);
                        let (h, gx, gz) = fbm_height_grad_x4(wx, wz, &params);
                        let ha = h.to_array();
                        let gxa = gx.to_array();
                        let gza = gz.to_array();
                        for l in 0..4 {
                            scratch[base + p + l] = Field::new(ha[l], gxa[l], gza[l]);
                        }
                    }
                    for p in (chunks * 4)..npts {
                        let (h, gx, gz) = fbm_height_grad(xs[p], zs[p], &params);
                        scratch[base + p] = Field::new(h, gx, gz);
                    }
                }
                NodeKind::Curve(spline) => {
                    for p in 0..npts {
                        let a = scratch[ia + p];
                        let (y, dy) = spline.eval(a.v);
                        scratch[base + p] = Field::new(y, dy * a.dx, dy * a.dz);
                    }
                }
                NodeKind::Ridge { ridge, amp_sum } => {
                    for p in 0..npts {
                        let a = scratch[ia + p];
                        let ridged = Field::constant(amp_sum).sub(a.abs().scale(2.0));
                        scratch[base + p] = a.add(ridged.sub(a).scale(ridge));
                    }
                }
                NodeKind::Clamp { lo, hi } => {
                    for p in 0..npts {
                        scratch[base + p] = scratch[ia + p].clamp(lo, hi);
                    }
                }
                NodeKind::Smoothstep { edge0, edge1 } => {
                    for p in 0..npts {
                        scratch[base + p] = scratch[ia + p].smoothstep(edge0, edge1);
                    }
                }
                NodeKind::Scale(k) => {
                    for p in 0..npts {
                        scratch[base + p] = scratch[ia + p].scale(k);
                    }
                }
                NodeKind::Offset(k) => {
                    for p in 0..npts {
                        scratch[base + p] = scratch[ia + p].offset(k);
                    }
                }
                NodeKind::Abs => {
                    for p in 0..npts {
                        scratch[base + p] = scratch[ia + p].abs();
                    }
                }
                NodeKind::Neg => {
                    for p in 0..npts {
                        scratch[base + p] = scratch[ia + p].neg();
                    }
                }
                NodeKind::Add => {
                    for p in 0..npts {
                        scratch[base + p] = scratch[ia + p].add(scratch[ib + p]);
                    }
                }
                NodeKind::Sub => {
                    for p in 0..npts {
                        scratch[base + p] = scratch[ia + p].sub(scratch[ib + p]);
                    }
                }
                NodeKind::Mul => {
                    for p in 0..npts {
                        scratch[base + p] = scratch[ia + p].mul(scratch[ib + p]);
                    }
                }
                NodeKind::Min => {
                    for p in 0..npts {
                        scratch[base + p] = scratch[ia + p].min(scratch[ib + p]);
                    }
                }
                NodeKind::Max => {
                    for p in 0..npts {
                        scratch[base + p] = scratch[ia + p].max(scratch[ib + p]);
                    }
                }
                NodeKind::Mix => {
                    for p in 0..npts {
                        scratch[base + p] = scratch[ia + p].mix(scratch[ib + p], scratch[ic + p]);
                    }
                }
            }
        }

        let out_base = self.output as usize * npts;
        for p in 0..npts {
            let f = scratch[out_base + p];
            out.v[p] = f.v;
            out.dx[p] = f.dx;
            out.dz[p] = f.dz;
        }
    }
}

/// Evaluate a single node from its (already-computed) input fields.
#[inline]
fn eval_node(kind: &NodeKind, wx: f64, wz: f64, seed: u64, a: Field, b: Field, c: Field) -> Field {
    match *kind {
        NodeKind::WorldX => Field::world_x(wx),
        NodeKind::WorldZ => Field::world_z(wz),
        NodeKind::Const(v) => Field::constant(v),
        NodeKind::Fbm(axis) => {
            let (h, gx, gz) = fbm_height_grad(wx, wz, &axis.params(seed));
            Field::new(h, gx, gz)
        }
        NodeKind::Curve(spline) => {
            let (y, dy) = spline.eval(a.v);
            Field::new(y, dy * a.dx, dy * a.dz) // chain rule through the input field
        }
        NodeKind::Ridge { ridge, amp_sum } => {
            // in + ridge·((amp_sum − 2|in|) − in) — gradient via the Field ops (autodiff).
            let ridged = Field::constant(amp_sum).sub(a.abs().scale(2.0));
            a.add(ridged.sub(a).scale(ridge))
        }
        NodeKind::Clamp { lo, hi } => a.clamp(lo, hi),
        NodeKind::Smoothstep { edge0, edge1 } => a.smoothstep(edge0, edge1),
        NodeKind::Scale(k) => a.scale(k),
        NodeKind::Offset(k) => a.offset(k),
        NodeKind::Abs => a.abs(),
        NodeKind::Neg => a.neg(),
        NodeKind::Add => a.add(b),
        NodeKind::Sub => a.sub(b),
        NodeKind::Mul => a.mul(b),
        NodeKind::Min => a.min(b),
        NodeKind::Max => a.max(b),
        NodeKind::Mix => a.mix(b, c),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central-difference the graph output's gradient and assert the autodiff `(dx,dz)` matches.
    fn assert_graph_grad(g: &Graph, pts: &[(f64, f64)], seed: u64) {
        g.validate().expect("valid graph");
        let e = 1e-3;
        for &(wx, wz) in pts {
            let f = g.eval(wx, wz, seed);
            let cd_x = (g.eval(wx + e, wz, seed).v - g.eval(wx - e, wz, seed).v) / (2.0 * e);
            let cd_z = (g.eval(wx, wz + e, seed).v - g.eval(wx, wz - e, seed).v) / (2.0 * e);
            assert!((f.dx - cd_x).abs() < 1e-2, "∂x at ({wx},{wz}): {} vs CD {cd_x}", f.dx);
            assert!((f.dz - cd_z).abs() < 1e-2, "∂z at ({wx},{wz}): {} vs CD {cd_z}", f.dz);
        }
    }

    const PTS: &[(f64, f64)] = &[(10.0, -20.0), (123.5, 456.0), (-300.0, 50.0), (1000.0, -700.0)];

    fn fbm(seed_salt: u32, base_freq: f64, amplitude: f64) -> NodeKind {
        NodeKind::Fbm(FbmAxis { octaves: 4, base_freq, lacunarity: 2.0, gain: 0.5, amplitude, seed_salt })
    }

    #[test]
    fn single_fbm_graph_grad_matches_fbm() {
        let g = Graph { nodes: vec![Node::source(fbm(0, 1.0 / 256.0, 100.0))], output: 0 };
        assert_graph_grad(&g, PTS, 7);
    }

    #[test]
    fn ridge_curve_mix_graph_grad() {
        // A realistic mini terrain graph: carrier fBm → ridge; a climate axis → curve → smoothstep gates
        // a mix between flat (const) and the ridged carrier. Exercises Fbm/Ridge/Curve/Smoothstep/Mix.
        let carrier = fbm(1, 1.0 / 512.0, 120.0);
        let climate = fbm(2, 1.0 / 4096.0, 1.0);
        let nodes = vec![
            Node::source(carrier),                                                  // 0
            Node::unary(NodeKind::Ridge { ridge: 0.7, amp_sum: 200.0 }, 0),         // 1 ridged carrier
            Node::source(climate),                                                  // 2
            Node::unary(NodeKind::Curve(Spline::new(&[(-1.0, 0.0), (0.0, 0.3), (1.0, 1.0)])), 2), // 3
            Node::unary(NodeKind::Smoothstep { edge0: 0.2, edge1: 0.8 }, 3),        // 4 placement gate
            Node::source(NodeKind::Const(5.0)),                                     // 5 plains
            Node::ternary(NodeKind::Mix, 5, 1, 4),                                  // 6 mix(plains, mtn, gate)
        ];
        let g = Graph { nodes, output: 6 };
        assert_graph_grad(&g, PTS, 42);
    }

    #[test]
    fn eval_is_deterministic_and_buffer_reuse_matches_alloc() {
        let g = Graph {
            nodes: vec![
                Node::source(fbm(3, 1.0 / 300.0, 80.0)),
                Node::unary(NodeKind::Ridge { ridge: 0.5, amp_sum: 160.0 }, 0),
            ],
            output: 1,
        };
        let a = g.eval(12.5, -7.25, 9);
        let b = g.eval(12.5, -7.25, 9);
        assert_eq!(a.v.to_bits(), b.v.to_bits());
        let mut scratch = vec![Field::constant(0.0); g.nodes.len()];
        let c = g.eval_into(12.5, -7.25, 9, &mut scratch);
        assert_eq!(a.v.to_bits(), c.v.to_bits());
        assert_eq!(a.dx.to_bits(), c.dx.to_bits());
    }

    /// SSOT GUARD for the columnar path: [`Graph::eval_grid`] must produce `to_bits()`-IDENTICAL
    /// `(v, dx, dz)` to the per-point [`Graph::eval`]/[`Graph::eval_into`] for a representative
    /// multi-node graph (the shipped mountains+plains biome graph — Fbm/Curve/Smoothstep/Ridge/
    /// Offset/Mix, the full op set) over a set of points. This is the bit-for-bit determinism contract
    /// for the batched evaluator; any drift between the scalar dispatch and the columnar dispatch fails.
    #[test]
    fn eval_grid_is_bit_identical_to_per_point_eval() {
        use super::super::preset::mountains_plains_graph;
        let g = mountains_plains_graph(700.0);
        g.validate().expect("valid graph");
        let seed = 1234u64;

        // A column of varied points (incl. negatives, sub-metre fractions, large coords).
        let pts: &[(f64, f64)] = &[
            (0.0, 0.0),
            (10.0, -20.0),
            (123.5, 456.25),
            (-300.0, 50.75),
            (1000.0, -700.0),
            (-2500.5, 3000.0),
            (4096.0, 4096.0),
            (-1.25, 0.5),
        ];
        let xs: Vec<f64> = pts.iter().map(|&(x, _)| x).collect();
        let zs: Vec<f64> = pts.iter().map(|&(_, z)| z).collect();
        let npts = pts.len();

        let n = g.nodes.len();
        let mut scratch = vec![Field::constant(0.0); n * npts];
        let mut v = vec![0.0f64; npts];
        let mut dx = vec![0.0f64; npts];
        let mut dz = vec![0.0f64; npts];
        g.eval_grid(&xs, &zs, seed, &mut scratch, GridOut { v: &mut v, dx: &mut dx, dz: &mut dz });

        for (p, &(wx, wz)) in pts.iter().enumerate() {
            let f = g.eval(wx, wz, seed);
            assert_eq!(v[p].to_bits(), f.v.to_bits(), "v mismatch at ({wx},{wz})");
            assert_eq!(dx[p].to_bits(), f.dx.to_bits(), "dx mismatch at ({wx},{wz})");
            assert_eq!(dz[p].to_bits(), f.dz.to_bits(), "dz mismatch at ({wx},{wz})");
        }
    }

    /// Also guard the smaller realistic graph from `ridge_curve_mix_graph_grad` (different topology /
    /// output index) so a node-walk/output-indexing bug in the columnar path can't hide.
    #[test]
    fn eval_grid_bit_identical_mini_graph() {
        let carrier = fbm(1, 1.0 / 512.0, 120.0);
        let climate = fbm(2, 1.0 / 4096.0, 1.0);
        let nodes = vec![
            Node::source(carrier),
            Node::unary(NodeKind::Ridge { ridge: 0.7, amp_sum: 200.0 }, 0),
            Node::source(climate),
            Node::unary(NodeKind::Curve(Spline::new(&[(-1.0, 0.0), (0.0, 0.3), (1.0, 1.0)])), 2),
            Node::unary(NodeKind::Smoothstep { edge0: 0.2, edge1: 0.8 }, 3),
            Node::source(NodeKind::Const(5.0)),
            Node::ternary(NodeKind::Mix, 5, 1, 4),
        ];
        let g = Graph { nodes, output: 6 };
        g.validate().expect("valid graph");
        let seed = 42u64;
        let xs: Vec<f64> = PTS.iter().map(|&(x, _)| x).collect();
        let zs: Vec<f64> = PTS.iter().map(|&(_, z)| z).collect();
        let npts = PTS.len();
        let mut scratch = vec![Field::constant(0.0); g.nodes.len() * npts];
        let (mut v, mut dx, mut dz) = (vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]);
        g.eval_grid(&xs, &zs, seed, &mut scratch, GridOut { v: &mut v, dx: &mut dx, dz: &mut dz });
        for (p, &(wx, wz)) in PTS.iter().enumerate() {
            let f = g.eval(wx, wz, seed);
            assert_eq!(v[p].to_bits(), f.v.to_bits());
            assert_eq!(dx[p].to_bits(), f.dx.to_bits());
            assert_eq!(dz[p].to_bits(), f.dz.to_bits());
        }
    }

    #[test]
    fn validate_rejects_bad_graphs() {
        assert_eq!(Graph { nodes: vec![], output: 0 }.validate(), Err(GraphError::Empty));
        let g = Graph { nodes: vec![Node::source(NodeKind::WorldX)], output: 5 };
        assert_eq!(g.validate(), Err(GraphError::OutputOutOfRange));
        // node 0 references input 0 (itself) on a binary op → non-topological.
        let g = Graph { nodes: vec![Node::binary(NodeKind::Add, 0, 0)], output: 0 };
        assert_eq!(g.validate(), Err(GraphError::NonTopologicalInput { node: 0, input: 0 }));
    }
}
