//! Conversion between the engine [`Graph`] and the editor [`Snarl`], the built-in default world graph,
//! and biome-navigation helpers (drill into a biome's sub-graph / breadcrumb out). The on-disk
//! load/save chain lives in `persist.rs`.

use bevy_egui::egui;
use egui_snarl::{InPinId, NodeId, OutPinId, Snarl};

use crate::sdf_render::worldgen::graph::node::{FbmAxis, Graph, NodeKind};

use super::{CLIMATE_INPUTS, EdNode};

/// Name of climate input `k` (falls back gracefully past the vocabulary).
pub(super) fn climate_name(k: usize) -> &'static str {
    CLIMATE_INPUTS.get(k).copied().unwrap_or("input")
}

// ===================================================================================================
// Conversion: engine Graph <-> editor Snarl
// ===================================================================================================

/// Build an editor Snarl from an engine [`Graph`]: one Snarl node per engine node (laid out in a column),
/// wired per the engine inputs, plus an `Output` sink wired to the engine output node.
pub fn graph_to_snarl(graph: &Graph) -> Snarl<EdNode> {
    let mut snarl = Snarl::new();
    let mut ids: Vec<NodeId> = Vec::with_capacity(graph.nodes.len());
    for (i, node) in graph.nodes.iter().enumerate() {
        let pos = egui::pos2(220.0 * (i % 4) as f32, 140.0 * (i / 4) as f32);
        ids.push(snarl.insert_node(pos, EdNode::Op(node.kind)));
    }
    // Wire inputs (skip self-referential placeholder slots beyond each node's arity).
    for (i, node) in graph.nodes.iter().enumerate() {
        for (slot, &src) in node.inputs[..node.kind.arity()].iter().enumerate() {
            snarl.connect(
                OutPinId { node: ids[src as usize], output: 0 },
                InPinId { node: ids[i], input: slot },
            );
        }
    }
    // The Output sink, wired to the engine output node.
    let out = snarl.insert_node(egui::pos2(220.0 * 4.0, 0.0), EdNode::Output);
    snarl.connect(OutPinId { node: ids[graph.output as usize], output: 0 }, InPinId { node: out, input: 0 });
    snarl
}

// ===================================================================================================
// Biome navigation (drill into a biome's sub-graph; breadcrumb back out)
// ===================================================================================================

/// How many leading `path` entries still resolve to live biome nodes (trailing stale ids dropped).
pub(super) fn valid_depth(root: &Snarl<EdNode>, path: &[NodeId]) -> usize {
    let mut s = root;
    for (i, &id) in path.iter().enumerate() {
        match s.get_node(id) {
            Some(EdNode::Biome { graph, .. }) => s = graph,
            _ => return i,
        }
    }
    path.len()
}

/// Biome names along `path` (for the breadcrumb).
pub(super) fn breadcrumb_names(root: &Snarl<EdNode>, path: &[NodeId]) -> Vec<String> {
    let mut names = Vec::with_capacity(path.len());
    let mut s = root;
    for &id in path {
        match s.get_node(id) {
            Some(EdNode::Biome { name, graph }) => {
                names.push(name.clone());
                s = graph;
            }
            _ => break,
        }
    }
    names
}

/// Resolve the snarl at `nav` (read-only), or `None` if any step no longer points to a biome (e.g. a
/// popped-out preview whose biome was deleted).
pub(super) fn resolve_snarl<'a>(root: &'a Snarl<EdNode>, nav: &[NodeId]) -> Option<&'a Snarl<EdNode>> {
    let mut s = root;
    for &id in nav {
        match s.get_node(id) {
            Some(EdNode::Biome { graph, .. }) => s = graph,
            _ => return None,
        }
    }
    Some(s)
}

/// The snarl shown at the current `path` (mutable). `path` must be valid (see [`valid_depth`]).
pub(super) fn current_snarl_mut<'a>(root: &'a mut Snarl<EdNode>, path: &[NodeId]) -> &'a mut Snarl<EdNode> {
    let mut s = root;
    for &id in path {
        s = match &mut s[id] {
            EdNode::Biome { graph, .. } => graph.as_mut(),
            _ => unreachable!("path is validated to biome nodes before use"),
        };
    }
    s
}

/// A fresh biome sub-graph: the four climate `Input` sentinels (available to wire) + a `Const(0)` wired
/// to an `Output`, so a new biome is valid (flat height 0) until the user shapes it.
pub(super) fn new_biome_subgraph() -> Snarl<EdNode> {
    let mut s = Snarl::new();
    for k in 0..CLIMATE_INPUTS.len() {
        s.insert_node(egui::pos2(0.0, 60.0 * k as f32), EdNode::Input(k));
    }
    let c = s.insert_node(egui::pos2(260.0, 0.0), EdNode::Op(NodeKind::Const(0.0)));
    let o = s.insert_node(egui::pos2(520.0, 0.0), EdNode::Output);
    s.connect(OutPinId { node: c, output: 0 }, InPinId { node: o, input: 0 });
    s
}

/// Sibling path the editor saves the **hierarchical** snarl (with biomes) to, alongside the flat engine
/// `.graph.ron` the worldgen loads. e.g. `…/mountains_plains.graph.ron` → `…/mountains_plains.worldgraph.ron`.
pub(super) fn worldgraph_path(graph_path: &str) -> String {
    let stem = graph_path.strip_suffix(".graph.ron").or_else(|| graph_path.strip_suffix(".ron"));
    match stem {
        Some(stem) => format!("{stem}.worldgraph.ron"),
        None => format!("{graph_path}.worldgraph.ron"),
    }
}

// Pin-id shorthands (every node has one output, pin 0).
fn opin(n: NodeId) -> OutPinId {
    OutPinId { node: n, output: 0 }
}
fn ipin(n: NodeId, i: usize) -> InPinId {
    InPinId { node: n, input: i }
}

// ===================================================================================================
// Default multi-biome "World" graph (the Phase-2 classifier example)
// ===================================================================================================

/// Build the default **multi-biome** world graph: low-frequency climate axes place + shape two biomes
/// (Plains in low continentalness, Mountains in high) blended by a continentalness gate — the Phase-2
/// architecture end-to-end (classifier on top, biomes own their shape, climate piped into each).
pub(super) fn world_biome_snarl() -> Snarl<EdNode> {
    fn climate(salt: u32, wavelength: f64) -> EdNode {
        EdNode::Op(NodeKind::Fbm(FbmAxis {
            octaves: 2,
            base_freq: 1.0 / wavelength,
            lacunarity: 2.0,
            gain: 0.5,
            amplitude: 1.0,
            seed_salt: salt,
        }))
    }

    // Plains biome: gentle rolling hills, nudged up a little by continentalness.
    let plains = {
        let mut s = Snarl::new();
        let cont = s.insert_node(egui::pos2(0.0, 0.0), EdNode::Input(0));
        let lift = s.insert_node(egui::pos2(220.0, 0.0), EdNode::Op(NodeKind::Scale(25.0)));
        s.connect(opin(cont), ipin(lift, 0));
        let hills = s.insert_node(
            egui::pos2(0.0, 140.0),
            EdNode::Op(NodeKind::Fbm(FbmAxis { octaves: 4, base_freq: 1.0 / 500.0, lacunarity: 2.0, gain: 0.5, amplitude: 30.0, seed_salt: 11 })),
        );
        let add = s.insert_node(egui::pos2(440.0, 0.0), EdNode::Op(NodeKind::Add));
        s.connect(opin(hills), ipin(add, 0));
        s.connect(opin(lift), ipin(add, 1));
        let o = s.insert_node(egui::pos2(660.0, 0.0), EdNode::Output);
        s.connect(opin(add), ipin(o, 0));
        s
    };

    // Mountains biome: ridged peaks on a continentalness-raised base.
    let mountains = {
        let mut s = Snarl::new();
        let cont = s.insert_node(egui::pos2(0.0, 0.0), EdNode::Input(0));
        let base = s.insert_node(egui::pos2(220.0, 0.0), EdNode::Op(NodeKind::Scale(220.0)));
        s.connect(opin(cont), ipin(base, 0));
        let fbm = s.insert_node(
            egui::pos2(0.0, 140.0),
            EdNode::Op(NodeKind::Fbm(FbmAxis { octaves: 5, base_freq: 1.0 / 1300.0, lacunarity: 2.0, gain: 0.5, amplitude: 1.0, seed_salt: 12 })),
        );
        let ridge = s.insert_node(egui::pos2(220.0, 140.0), EdNode::Op(NodeKind::Ridge { ridge: 0.9, amp_sum: 2.0 }));
        s.connect(opin(fbm), ipin(ridge, 0));
        let peaks = s.insert_node(egui::pos2(440.0, 140.0), EdNode::Op(NodeKind::Scale(620.0)));
        s.connect(opin(ridge), ipin(peaks, 0));
        let add = s.insert_node(egui::pos2(660.0, 0.0), EdNode::Op(NodeKind::Add));
        s.connect(opin(peaks), ipin(add, 0));
        s.connect(opin(base), ipin(add, 1));
        let off = s.insert_node(egui::pos2(880.0, 0.0), EdNode::Op(NodeKind::Offset(80.0)));
        s.connect(opin(add), ipin(off, 0));
        let o = s.insert_node(egui::pos2(1100.0, 0.0), EdNode::Output);
        s.connect(opin(off), ipin(o, 0));
        s
    };

    let mut s = Snarl::new();
    let cont = s.insert_node(egui::pos2(0.0, 0.0), climate(5, 8000.0));
    let temp = s.insert_node(egui::pos2(0.0, 150.0), climate(6, 7000.0));
    let humid = s.insert_node(egui::pos2(0.0, 300.0), climate(7, 6500.0));
    let weird = s.insert_node(egui::pos2(0.0, 450.0), climate(8, 5000.0));
    let bp = s.insert_node(egui::pos2(340.0, 0.0), EdNode::Biome { name: "Plains".into(), graph: Box::new(plains) });
    let bm = s.insert_node(egui::pos2(340.0, 320.0), EdNode::Biome { name: "Mountains".into(), graph: Box::new(mountains) });
    for b in [bp, bm] {
        s.connect(opin(cont), ipin(b, 0));
        s.connect(opin(temp), ipin(b, 1));
        s.connect(opin(humid), ipin(b, 2));
        s.connect(opin(weird), ipin(b, 3));
    }
    // Classifier: blend plains↔mountains by a continentalness gate (low ⇒ plains, high ⇒ mountains).
    let gate = s.insert_node(egui::pos2(340.0, 620.0), EdNode::Op(NodeKind::Smoothstep { edge0: 0.0, edge1: 0.5 }));
    s.connect(opin(cont), ipin(gate, 0));
    let mix = s.insert_node(egui::pos2(700.0, 160.0), EdNode::Op(NodeKind::Mix));
    s.connect(opin(bp), ipin(mix, 0)); // a = plains
    s.connect(opin(bm), ipin(mix, 1)); // b = mountains
    s.connect(opin(gate), ipin(mix, 2)); // t = gate
    let o = s.insert_node(egui::pos2(980.0, 160.0), EdNode::Output);
    s.connect(opin(mix), ipin(o, 0));
    s
}
