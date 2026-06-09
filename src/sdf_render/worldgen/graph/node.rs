//! The node library + graph evaluator. A [`Graph`] is a DAG of [`Node`]s; each node computes a
//! [`Field`] (value + analytic gradient) at one world point from its inputs, so evaluating the graph
//! forward-mode-autodiffs to the terrain `(height, dh_dx, dh_dz)`. Nodes are stored in topological
//! order (each input references an EARLIER node), so a single forward pass over a scratch buffer
//! evaluates the whole graph.
//!
//! Bit-portable/deterministic: every node is `Field` basic ops + the portable noise basis, fixed
//! order. Sources (`WorldX/Z`, `Const`, `Fbm`) ignore their inputs; `Fbm` reads the eval point + folds
//! the world seed with its salt (the exact idiom in `layers::height::fbm_params`).

use super::super::noise::{FbmParams, fbm_height_grad};
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
