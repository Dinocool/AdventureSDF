//! Conversion + compilation of the editor [`Snarl`] into the flat engine [`Graph`]. Biomes are inlined
//! (their climate `Input` sentinels rewired to the parent edges feeding the biome's pins), so the engine
//! form is unchanged by the editor-side grouping.

use egui_snarl::{NodeId, Snarl};

use crate::sdf_render::worldgen::graph::node::{Graph, Node, NodeKind};
use crate::sdf_render::worldgen::graph::preset::MAX_GRAPH_NODES;

use super::{CLIMATE_INPUTS, EdNode, climate_name};

/// Find the node whose output feeds the single `Output` sink of `snarl` (the (sub)graph's root).
pub(super) fn output_root(snarl: &Snarl<EdNode>) -> Result<NodeId, String> {
    let mut sink = None;
    for (id, node) in snarl.node_ids() {
        if matches!(node, EdNode::Output) {
            if sink.is_some() {
                return Err("graph has more than one Output node".into());
            }
            sink = Some(id);
        }
    }
    let sink = sink.ok_or("graph has no Output node")?;
    for (out, inp) in snarl.wires() {
        if inp.node == sink && inp.input == 0 {
            return Ok(out.node);
        }
    }
    Err("the Output node has no input wired".into())
}

/// Convert the editor Snarl (possibly nested with biomes) to a flat engine [`Graph`] by **inlining**:
/// each biome's sub-graph is spliced in with its climate-`Input` sentinels rewired to the parent edges
/// feeding that biome's pins. Errors on missing/duplicate Output, a cycle, an unconnected required
/// input, or >[`MAX_GRAPH_NODES`] nodes.
pub fn snarl_to_graph(snarl: &Snarl<EdNode>) -> Result<Graph, String> {
    compile_root(snarl, false)
}

/// Compile the sub-graph feeding `root` into a flat engine [`Graph`] rooted at that node — used by the
/// per-node 2D/3D preview. Tolerant of unbound climate `Input`s (treats them as 0) so a node inside a
/// biome still previews in isolation.
pub fn graph_rooted_at(snarl: &Snarl<EdNode>, root: NodeId) -> Result<Graph, String> {
    let mut out = Vec::new();
    let output = CompileCtx::new(snarl, &[], &mut out, true).compile_subgraph(root)?;
    finish_graph(out, output)
}

/// Compile the whole `snarl` from its single `Output` sink.
fn compile_root(snarl: &Snarl<EdNode>, input_fallback: bool) -> Result<Graph, String> {
    let root = output_root(snarl)?;
    let mut out = Vec::new();
    let output = CompileCtx::new(snarl, &[], &mut out, input_fallback).compile_subgraph(root)?;
    finish_graph(out, output)
}

fn finish_graph(nodes: Vec<Node>, output: u32) -> Result<Graph, String> {
    if nodes.len() > MAX_GRAPH_NODES {
        return Err(format!("graph has {} nodes (max {MAX_GRAPH_NODES})", nodes.len()));
    }
    let graph = Graph { nodes, output };
    graph.validate().map_err(|e| format!("{e:?}"))?;
    Ok(graph)
}

/// The state threaded through one sub-graph's compilation. `out` (the flat engine node list being built)
/// and `input_fallback` are shared across all inlined sub-graphs (a biome reuses the parent's `out`); the
/// rest (`snarl`/`src`/`binds`/`memo`/`on_stack`) are per-sub-graph — a fresh `CompileCtx` is built for
/// each inlined biome (see [`CompileCtx::compile_node`]'s `Biome` arm).
struct CompileCtx<'a> {
    /// The sub-graph being compiled.
    snarl: &'a Snarl<EdNode>,
    /// Wiring of `snarl`: (dst node, input slot) → source node.
    src: std::collections::HashMap<(NodeId, usize), NodeId>,
    /// `binds[k]` is the engine index a climate `Input(k)` resolves to (the parent edge feeding pin `k`).
    binds: &'a [Option<u32>],
    /// The flat engine node list being built (shared across inlined sub-graphs).
    out: &'a mut Vec<Node>,
    /// node → its engine index, memoised so shared upstreams compile once.
    memo: std::collections::HashMap<NodeId, u32>,
    /// Nodes on the current recursion path, for cycle detection.
    on_stack: std::collections::HashSet<NodeId>,
    /// Substitute 0 for an unbound-but-used climate input (preview only).
    input_fallback: bool,
}

impl<'a> CompileCtx<'a> {
    fn new(snarl: &'a Snarl<EdNode>, binds: &'a [Option<u32>], out: &'a mut Vec<Node>, input_fallback: bool) -> Self {
        let mut src = std::collections::HashMap::new();
        for (o, i) in snarl.wires() {
            src.insert((i.node, i.input), o.node);
        }
        Self { snarl, src, binds, out, memo: std::collections::HashMap::new(), on_stack: std::collections::HashSet::new(), input_fallback }
    }

    /// Compile the sub-graph rooted at `root`, appending engine nodes to `self.out` and returning `root`'s
    /// engine index.
    fn compile_subgraph(&mut self, root: NodeId) -> Result<u32, String> {
        self.compile_node(root)
    }

    fn compile_node(&mut self, id: NodeId) -> Result<u32, String> {
        if let Some(&i) = self.memo.get(&id) {
            return Ok(i);
        }
        if !self.on_stack.insert(id) {
            return Err("graph has a cycle".into());
        }
        let res = self.compile_node_inner(id);
        self.on_stack.remove(&id);
        if let Ok(i) = res {
            self.memo.insert(id, i);
        }
        res
    }

    fn compile_node_inner(&mut self, id: NodeId) -> Result<u32, String> {
        match self.snarl.get_node(id) {
            Some(EdNode::Op { kind, .. }) => {
                let kind = *kind;
                let mut inputs = [0u32; 3];
                for (slot, inp) in inputs.iter_mut().enumerate().take(kind.arity()) {
                    let up = *self
                        .src
                        .get(&(id, slot))
                        .ok_or_else(|| format!("node input {slot} is not connected"))?;
                    *inp = self.compile_node(up)?;
                }
                self.out.push(Node { kind, inputs });
                Ok((self.out.len() - 1) as u32)
            }
            Some(EdNode::Input(k)) => match self.binds.get(*k).copied().flatten() {
                Some(i) => Ok(i),
                None if self.input_fallback => {
                    self.out.push(Node { kind: NodeKind::Const(0.0), inputs: [0; 3] });
                    Ok((self.out.len() - 1) as u32)
                }
                None => Err(format!("biome input '{}' is not connected", climate_name(*k))),
            },
            Some(EdNode::Biome { graph, .. }) => {
                // Resolve the parent edges feeding this biome's climate pins, then inline its sub-graph.
                let mut sub_binds: Vec<Option<u32>> = Vec::with_capacity(CLIMATE_INPUTS.len());
                for slot in 0..CLIMATE_INPUTS.len() {
                    match self.src.get(&(id, slot)).copied() {
                        Some(up) => sub_binds.push(Some(self.compile_node(up)?)),
                        None => sub_binds.push(None),
                    }
                }
                let sub_root = output_root(graph)?;
                // Fresh per-sub-graph ctx (new snarl/src/binds/memo/on_stack) sharing the same `out`.
                CompileCtx::new(graph, &sub_binds, self.out, self.input_fallback).compile_subgraph(sub_root)
            }
            Some(EdNode::Output) => Err("the Output node cannot be used as an input".into()),
            None => Err("dangling node reference".into()),
        }
    }
}
