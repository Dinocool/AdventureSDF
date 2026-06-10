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
    let root = output_root(snarl)?;
    let mut out = Vec::new();
    let output = compile_subgraph(snarl, root, &[], &mut out, false)?;
    finish_graph(out, output)
}

/// Compile the sub-graph feeding `root` into a flat engine [`Graph`] rooted at that node — used by the
/// per-node 2D/3D preview. Tolerant of unbound climate `Input`s (treats them as 0) so a node inside a
/// biome still previews in isolation.
pub fn graph_rooted_at(snarl: &Snarl<EdNode>, root: NodeId) -> Result<Graph, String> {
    let mut out = Vec::new();
    let output = compile_subgraph(snarl, root, &[], &mut out, true)?;
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

/// Compile one (sub)snarl rooted at `root`, appending engine nodes to `out` and returning `root`'s engine
/// index. `binds[k]` is the engine index a climate `Input(k)` resolves to (the parent edge feeding pin
/// `k`); `input_fallback` substitutes 0 for an unbound-but-used input (preview only).
fn compile_subgraph(
    snarl: &Snarl<EdNode>,
    root: NodeId,
    binds: &[Option<u32>],
    out: &mut Vec<Node>,
    input_fallback: bool,
) -> Result<u32, String> {
    use std::collections::{HashMap, HashSet};
    let mut src: HashMap<(NodeId, usize), NodeId> = HashMap::new();
    for (o, i) in snarl.wires() {
        src.insert((i.node, i.input), o.node);
    }
    let mut memo: HashMap<NodeId, u32> = HashMap::new();
    let mut on_stack: HashSet<NodeId> = HashSet::new();
    compile_node(root, snarl, &src, binds, out, &mut memo, &mut on_stack, input_fallback)
}

#[allow(clippy::too_many_arguments)]
fn compile_node(
    id: NodeId,
    snarl: &Snarl<EdNode>,
    src: &std::collections::HashMap<(NodeId, usize), NodeId>,
    binds: &[Option<u32>],
    out: &mut Vec<Node>,
    memo: &mut std::collections::HashMap<NodeId, u32>,
    on_stack: &mut std::collections::HashSet<NodeId>,
    input_fallback: bool,
) -> Result<u32, String> {
    if let Some(&i) = memo.get(&id) {
        return Ok(i);
    }
    if !on_stack.insert(id) {
        return Err("graph has a cycle".into());
    }
    let res = (|| match snarl.get_node(id) {
        Some(EdNode::Op(kind)) => {
            let kind = *kind;
            let mut inputs = [0u32; 3];
            for (slot, inp) in inputs.iter_mut().enumerate().take(kind.arity()) {
                let up = *src
                    .get(&(id, slot))
                    .ok_or_else(|| format!("node input {slot} is not connected"))?;
                *inp = compile_node(up, snarl, src, binds, out, memo, on_stack, input_fallback)?;
            }
            out.push(Node { kind, inputs });
            Ok((out.len() - 1) as u32)
        }
        Some(EdNode::Input(k)) => match binds.get(*k).copied().flatten() {
            Some(i) => Ok(i),
            None if input_fallback => {
                out.push(Node { kind: NodeKind::Const(0.0), inputs: [0; 3] });
                Ok((out.len() - 1) as u32)
            }
            None => Err(format!("biome input '{}' is not connected", climate_name(*k))),
        },
        Some(EdNode::Biome { graph, .. }) => {
            // Resolve the parent edges feeding this biome's climate pins, then inline its sub-graph.
            let mut sub_binds: Vec<Option<u32>> = Vec::with_capacity(CLIMATE_INPUTS.len());
            for slot in 0..CLIMATE_INPUTS.len() {
                match src.get(&(id, slot)) {
                    Some(&up) => sub_binds
                        .push(Some(compile_node(up, snarl, src, binds, out, memo, on_stack, input_fallback)?)),
                    None => sub_binds.push(None),
                }
            }
            let sub_root = output_root(graph)?;
            compile_subgraph(graph, sub_root, &sub_binds, out, input_fallback)
        }
        Some(EdNode::Output) => Err("the Output node cannot be used as an input".into()),
        None => Err("dangling node reference".into()),
    })();
    on_stack.remove(&id);
    if let Ok(i) = res {
        memo.insert(id, i);
    }
    res
}
