//! `NodeKind → WGSL` codegen — compiles a [`Graph`] to a WGSL `wg_eval_graph` function so the GPU
//! voxelizer evaluates the SAME node-graph the CPU `eval_into` does (Stage 1a of the GPU-voxel-worldgen
//! pivot, docs/GPU_VOXEL_WORLDGEN_PLAN.md). The node-graph stays the single source of truth: one graph →
//! CPU autodiff (`graph::node::eval_into`) AND GPU WGSL (this pass).
//!
//! The emitted function mirrors [`Graph::eval_into`] line-for-line: it walks the topologically-ordered
//! nodes once, computing each node's [`WgField`](crate) into a local `var n{i}`, each calling the matching
//! `wg_*` helper in `assets/shaders/worldgen_gpu.wgsl`. The output node's `WgField` is returned.
//!
//! ```ignore
//! fn wg_eval_graph(wx: f32, wz: f32, world_seed: u32) -> WgField {
//!     let n0 = wg_fbm_node(wx, wz, world_seed, WgFbmParams(/* axis */));
//!     let n1 = wg_ridge(n0, 0.5, 551.25);
//!     ...
//!     return n{output};
//! }
//! ```
//!
//! ## Exhaustiveness (robust by construction)
//! The `match` over [`NodeKind`] has NO catch-all arm — adding a future opcode is a compile error here
//! until it gets a WGSL emission, so the GPU path can never silently drop an op the CPU has.
//!
//! ## f32 vs f64
//! Every f64 graph constant is narrowed to an f32 WGSL literal ([`wgsl_f32`]); the math then runs in f32
//! (the CPU runs f64). See the `worldgen_gpu.wgsl` header for the full divergence list. The integer
//! `seed_salt` / octave counts are emitted as exact `u32`, so the hash entropy still matches bit-for-bit.

use super::node::{FbmAxis, Graph, NodeKind};
use super::super::spline::Spline;

/// The WGSL function name the codegen emits and the GPU voxelizer calls.
pub const EVAL_FN_NAME: &str = "wg_eval_graph";

/// Format an `f64` graph constant as a finite, parseable WGSL `f32` literal. WGSL requires a float literal
/// to carry a decimal point (or exponent); this always emits one with the `f` suffix so the literal is
/// unambiguously `f32` (the value round-trips through `{:?}`'s shortest exact decimal).
///
/// Non-finite values (a malformed graph) are clamped to the largest finite f32 with the sign preserved —
/// the GPU can't represent inf/NaN literals and the CPU `value_bound` already keeps sane graphs finite.
fn wgsl_f32(x: f64) -> String {
    let v = x as f32;
    let v = if v.is_finite() {
        v
    } else if v > 0.0 {
        f32::MAX
    } else {
        -f32::MAX
    };
    // `{:?}` on f32 prints the shortest decimal that round-trips and always includes a `.` or `e`,
    // e.g. `0.5`, `-1.0`, `1378.125`, `3.4028235e38` — all valid WGSL f32 literals once suffixed.
    format!("{v:?}f")
}

/// Emit a `WgFbmParams(...)` struct literal for an fBm source axis. The `seed` field is left as the axis
/// `seed_salt` (an exact `u32`); the WGSL `wg_fbm_node` folds it with the runtime `world_seed` — matching
/// the CPU `FbmAxis::params` seed fold (see the worldgen_gpu.wgsl header for the u64→u32 collapse note).
fn fbm_params_literal(ax: &FbmAxis) -> String {
    format!(
        "WgFbmParams({octaves}u, {base_freq}, {lacunarity}, {gain}, {amplitude}, {seed}u)",
        octaves = ax.octaves,
        base_freq = wgsl_f32(ax.base_freq),
        lacunarity = wgsl_f32(ax.lacunarity),
        gain = wgsl_f32(ax.gain),
        amplitude = wgsl_f32(ax.amplitude),
        seed = ax.seed_salt,
    )
}

/// Emit the per-Curve spline control-point arrays + the `wg_curve(...)` call into `body`, binding the
/// node result to `dst` (`n{i}`). The spline's active `(x, y)` points are written into fixed `array<f32, 8>`
/// `var`s (SPLINE_MAX_POINTS = 8, matching `wg_curve`'s pointer args); unused slots are padded with `0.0`
/// (never read — `wg_spline_eval` only indexes `< len`).
fn emit_curve(body: &mut String, dst: &str, input: &str, spline: &Spline) {
    let pts: Vec<(f64, f64)> = spline.points().collect();
    let len = pts.len();
    debug_assert!((1..=super::super::spline::SPLINE_MAX_POINTS).contains(&len), "spline len {len} out of range");
    let mut xs = [0.0f64; super::super::spline::SPLINE_MAX_POINTS];
    let mut ys = [0.0f64; super::super::spline::SPLINE_MAX_POINTS];
    for (i, &(x, y)) in pts.iter().enumerate() {
        xs[i] = x;
        ys[i] = y;
    }
    let xs_lit = xs.iter().map(|&x| wgsl_f32(x)).collect::<Vec<_>>().join(", ");
    let ys_lit = ys.iter().map(|&y| wgsl_f32(y)).collect::<Vec<_>>().join(", ");
    // Unique array names per node so multiple Curve nodes don't collide.
    let xs_name = format!("{dst}_cx");
    let ys_name = format!("{dst}_cy");
    body.push_str(&format!("    var {xs_name} = array<f32, 8>({xs_lit});\n"));
    body.push_str(&format!("    var {ys_name} = array<f32, 8>({ys_lit});\n"));
    body.push_str(&format!("    let {dst} = wg_curve({input}, &{xs_name}, &{ys_name}, {len}u);\n"));
}

/// Compile a validated [`Graph`] to a WGSL `wg_eval_graph(wx: f32, wz: f32, world_seed: u32) -> WgField`
/// function whose body evaluates the graph's nodes in topological order (each node a local `let`/`var`
/// `n{i}` calling the matching `wg_*` op from `worldgen::gpu`), returning the output node's `WgField`.
///
/// The caller composes this string AFTER `#import worldgen::gpu::*` (or the whole library source) so the
/// `wg_*` helpers + `WgField`/`WgFbmParams` types resolve. The graph SHOULD be [`Graph::validate`]'d first
/// (topological order); an out-of-range output index is clamped to the last node so the emitted WGSL still
/// compiles (the validator is the real gate).
pub fn graph_to_wgsl(graph: &Graph) -> String {
    let n = graph.nodes.len();
    let mut body = String::new();
    body.push_str(&format!("fn {EVAL_FN_NAME}(wx: f32, wz: f32, world_seed: u32) -> WgField {{\n"));

    if n == 0 {
        // Degenerate (empty) graph — emit a constant-zero field so the WGSL still compiles. `validate`
        // rejects this; this is the structural safety net.
        body.push_str("    return wg_const(0.0f);\n}\n");
        return body;
    }

    for (i, node) in graph.nodes.iter().enumerate() {
        let dst = format!("n{i}");
        let a = format!("n{}", node.inputs[0]);
        let b = format!("n{}", node.inputs[1]);
        let c = format!("n{}", node.inputs[2]);
        // EXHAUSTIVE match — no catch-all: a new NodeKind must add a WGSL emission here or it won't compile.
        match &node.kind {
            // --- sources (arity 0) ---
            NodeKind::WorldX => {
                body.push_str(&format!("    let {dst} = wg_world_x(wx);\n"));
            }
            NodeKind::WorldZ => {
                body.push_str(&format!("    let {dst} = wg_world_z(wz);\n"));
            }
            NodeKind::Const(v) => {
                body.push_str(&format!("    let {dst} = wg_const({});\n", wgsl_f32(*v)));
            }
            NodeKind::Fbm(ax) => {
                body.push_str(&format!(
                    "    let {dst} = wg_fbm_node(wx, wz, world_seed, {});\n",
                    fbm_params_literal(ax),
                ));
            }
            // --- unary (arity 1) ---
            NodeKind::Curve(spline) => {
                emit_curve(&mut body, &dst, &a, spline);
            }
            NodeKind::Ridge { ridge, amp_sum } => {
                body.push_str(&format!(
                    "    let {dst} = wg_ridge({a}, {}, {});\n",
                    wgsl_f32(*ridge),
                    wgsl_f32(*amp_sum),
                ));
            }
            NodeKind::Clamp { lo, hi } => {
                body.push_str(&format!(
                    "    let {dst} = wg_clamp_field({a}, {}, {});\n",
                    wgsl_f32(*lo),
                    wgsl_f32(*hi),
                ));
            }
            NodeKind::Smoothstep { edge0, edge1 } => {
                body.push_str(&format!(
                    "    let {dst} = wg_smoothstep_field({a}, {}, {});\n",
                    wgsl_f32(*edge0),
                    wgsl_f32(*edge1),
                ));
            }
            NodeKind::Scale(k) => {
                body.push_str(&format!("    let {dst} = wg_scale({a}, {});\n", wgsl_f32(*k)));
            }
            NodeKind::Offset(k) => {
                body.push_str(&format!("    let {dst} = wg_offset({a}, {});\n", wgsl_f32(*k)));
            }
            NodeKind::Abs => {
                body.push_str(&format!("    let {dst} = wg_abs({a});\n"));
            }
            NodeKind::Neg => {
                body.push_str(&format!("    let {dst} = wg_neg({a});\n"));
            }
            // --- binary (arity 2) ---
            NodeKind::Add => {
                body.push_str(&format!("    let {dst} = wg_add({a}, {b});\n"));
            }
            NodeKind::Sub => {
                body.push_str(&format!("    let {dst} = wg_sub({a}, {b});\n"));
            }
            NodeKind::Mul => {
                body.push_str(&format!("    let {dst} = wg_mul({a}, {b});\n"));
            }
            NodeKind::Min => {
                body.push_str(&format!("    let {dst} = wg_min({a}, {b});\n"));
            }
            NodeKind::Max => {
                body.push_str(&format!("    let {dst} = wg_max({a}, {b});\n"));
            }
            // --- ternary (arity 3) ---
            NodeKind::Mix => {
                body.push_str(&format!("    let {dst} = wg_mix({a}, {b}, {c});\n"));
            }
        }
    }

    let out = (graph.output as usize).min(n - 1);
    body.push_str(&format!("    return n{out};\n}}\n"));
    body
}

#[cfg(test)]
mod tests {
    use super::super::preset::{default_terrain_graph, mountains_plains_graph};
    use super::super::node::{FbmAxis, Graph, Node, NodeKind};
    use super::super::super::spline::Spline;
    use super::*;

    fn carrier() -> FbmAxis {
        FbmAxis { octaves: 6, base_freq: 1.0 / 1536.0, lacunarity: 2.0, gain: 0.5, amplitude: 280.0, seed_salt: 0 }
    }

    /// A graph exercising EVERY NodeKind variant — the exhaustiveness guard for the codegen (if a new
    /// opcode is added without a WGSL emission, `graph_to_wgsl`'s match won't compile, and this graph
    /// makes sure each op actually appears in the output).
    fn all_ops_graph() -> Graph {
        let nodes = vec![
            Node::source(NodeKind::WorldX),                                            // 0
            Node::source(NodeKind::WorldZ),                                            // 1
            Node::source(NodeKind::Const(3.0)),                                        // 2
            Node::source(NodeKind::Fbm(carrier())),                                    // 3
            Node::unary(NodeKind::Curve(Spline::new(&[(-1.0, 0.0), (0.0, 0.3), (1.0, 1.0)])), 3), // 4
            Node::unary(NodeKind::Ridge { ridge: 0.5, amp_sum: 100.0 }, 3),            // 5
            Node::unary(NodeKind::Clamp { lo: -10.0, hi: 10.0 }, 4),                   // 6
            Node::unary(NodeKind::Smoothstep { edge0: 0.1, edge1: 0.7 }, 4),           // 7
            Node::unary(NodeKind::Scale(2.0), 5),                                      // 8
            Node::unary(NodeKind::Offset(5.0), 8),                                     // 9
            Node::unary(NodeKind::Abs, 0),                                             // 10
            Node::unary(NodeKind::Neg, 1),                                             // 11
            Node::binary(NodeKind::Add, 9, 2),                                         // 12
            Node::binary(NodeKind::Sub, 12, 10),                                       // 13
            Node::binary(NodeKind::Mul, 13, 11),                                       // 14
            Node::binary(NodeKind::Min, 14, 6),                                        // 15
            Node::binary(NodeKind::Max, 15, 6),                                        // 16
            Node::ternary(NodeKind::Mix, 16, 7, 7),                                    // 17
        ];
        Graph { nodes, output: 17 }
    }

    #[test]
    fn emits_function_with_signature_and_return() {
        let g = default_terrain_graph(carrier(), 0.5, 551.25, 0.0);
        let src = graph_to_wgsl(&g);
        assert!(src.contains("fn wg_eval_graph(wx: f32, wz: f32, world_seed: u32) -> WgField"));
        assert!(src.trim_end().ends_with('}'));
        assert!(src.contains("return n2;"), "default graph output is node 2");
        // The three ops the default graph uses.
        assert!(src.contains("wg_fbm_node("));
        assert!(src.contains("wg_ridge("));
        assert!(src.contains("wg_offset("));
    }

    #[test]
    fn every_op_appears_in_output() {
        let src = graph_to_wgsl(&all_ops_graph());
        for needle in [
            "wg_world_x(", "wg_world_z(", "wg_const(", "wg_fbm_node(", "wg_curve(", "wg_ridge(",
            "wg_clamp_field(", "wg_smoothstep_field(", "wg_scale(", "wg_offset(", "wg_abs(", "wg_neg(",
            "wg_add(", "wg_sub(", "wg_mul(", "wg_min(", "wg_max(", "wg_mix(",
        ] {
            assert!(src.contains(needle), "codegen output missing {needle}\n{src}");
        }
    }

    #[test]
    fn fbm_seed_salt_is_exact_u32() {
        // The seed salt must be emitted as an exact integer (bit-exact hash entropy), not a float.
        let ax = FbmAxis { octaves: 3, base_freq: 0.001, lacunarity: 2.0, gain: 0.5, amplitude: 50.0, seed_salt: 0x00C0_0001 };
        let g = Graph { nodes: vec![Node::source(NodeKind::Fbm(ax))], output: 0 };
        let src = graph_to_wgsl(&g);
        assert!(src.contains("12582913u"), "seed_salt 0x00C00001 must appear as 12582913u\n{src}");
        assert!(src.contains("3u,"), "octaves must be an exact u32 literal\n{src}");
    }

    #[test]
    fn mountains_plains_emits_all_its_ops_topologically() {
        let g = mountains_plains_graph(700.0);
        g.validate().expect("valid preset");
        let src = graph_to_wgsl(&g);
        // Mix is the output (node 8).
        assert!(src.contains("return n8;"));
        assert!(src.contains("wg_curve("));
        assert!(src.contains("wg_smoothstep_field("));
        assert!(src.contains("wg_mix(n4, n7, n2)"), "mix wiring (inputs 4,7,2) must be preserved\n{src}");
    }

    #[test]
    fn float_literals_are_wgsl_valid() {
        // Every emitted float literal ends in `f` and has a `.` or exponent (WGSL f32 literal form).
        let g = mountains_plains_graph(700.0);
        let src = graph_to_wgsl(&g);
        assert!(!src.contains("inf") && !src.contains("nan"), "no non-finite literals\n{src}");
        // Spot-check a known constant: Offset(120.0) → wg_offset(.., 120.0f).
        assert!(src.contains("120.0f"), "Offset(120.0) must emit 120.0f\n{src}");
    }
}
