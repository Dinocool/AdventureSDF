//! The **biome node-graph editor** — a visual `egui-snarl` panel for authoring the worldgen terrain
//! graph. Nodes are the engine [`NodeKind`] library (plus an [`EdNode::Output`] sink); editing rebuilds
//! the engine [`Graph`] and republishes it into the [`WorldGraph`] resource, which `roll_worldgen`
//! re-meshes live. Load/save go through the same RON asset pipeline as materials.
//!
//! `Snarl<EdNode>` is the editor's working graph; [`snarl_to_graph`]/[`graph_to_snarl`] convert to/from
//! the engine [`Graph`] (the bake samples the engine form). Gated behind `editor`.

use std::sync::Arc;

use bevy::prelude::*;
use bevy_egui::egui;
use egui_snarl::ui::{PinInfo, SnarlStyle, SnarlViewer, SnarlWidget};
use egui_snarl::{InPin, InPinId, NodeId, OutPin, OutPinId, Snarl};

use crate::sdf_render::worldgen::WorldGraph;
use crate::sdf_render::worldgen::graph::node::{FbmAxis, Graph, Node, NodeKind};
use crate::sdf_render::worldgen::graph::preset::{MAX_GRAPH_NODES, mountains_plains_graph};
use crate::sdf_render::worldgen::spline::Spline;

/// A node in the editor graph: an engine operation, or the single graph OUTPUT sink (1 input, 0
/// outputs) that designates which node's value is the terrain height.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum EdNode {
    Op(NodeKind),
    Output,
}

/// Editor state: the working Snarl graph + whether it's been seeded from the live `WorldGraph` yet.
#[derive(Resource, Default)]
pub struct WorldGraphEditor {
    snarl: Snarl<EdNode>,
    seeded: bool,
}

/// Plugin: registers the editor state + the dockable "Biome Graph" panel.
pub struct WorldgenGraphEditorPlugin;

impl Plugin for WorldgenGraphEditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldGraphEditor>();
        super::panels::register_panel(
            app,
            "worldgen/graph",
            "Biome Graph",
            super::panels::DockSide::Right,
            30,
            graph_panel,
        );
    }
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

/// Convert the editor Snarl back to an engine [`Graph`]: topologically order the Op nodes feeding the
/// `Output` sink, map Snarl ids → engine indices, resolve each node's inputs from the wires. Errors on a
/// missing/duplicate Output, a cycle, an unconnected required input, or >[`MAX_GRAPH_NODES`] nodes.
pub fn snarl_to_graph(snarl: &Snarl<EdNode>) -> Result<Graph, String> {
    use std::collections::HashMap;

    // Source feeding (node, input slot) → the upstream node id (every node has one output, pin 0).
    let mut src: HashMap<(NodeId, usize), NodeId> = HashMap::new();
    for (out, inp) in snarl.wires() {
        src.insert((inp.node, inp.input), out.node);
    }

    // Find the single Output sink.
    let mut output_sink = None;
    for (id, node) in snarl.node_ids() {
        if matches!(node, EdNode::Output) {
            if output_sink.is_some() {
                return Err("graph has more than one Output node".into());
            }
            output_sink = Some(id);
        }
    }
    let output_sink = output_sink.ok_or("graph has no Output node")?;
    let root = *src.get(&(output_sink, 0)).ok_or("the Output node has no input wired")?;

    // Post-order DFS from the root over input edges → topological order; detect cycles.
    let mut order: Vec<NodeId> = Vec::new();
    let mut index: HashMap<NodeId, u32> = HashMap::new();
    let mut on_stack: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    fn visit(
        id: NodeId,
        snarl: &Snarl<EdNode>,
        src: &std::collections::HashMap<(NodeId, usize), NodeId>,
        order: &mut Vec<NodeId>,
        index: &mut std::collections::HashMap<NodeId, u32>,
        on_stack: &mut std::collections::HashSet<NodeId>,
    ) -> Result<(), String> {
        if index.contains_key(&id) {
            return Ok(());
        }
        if !on_stack.insert(id) {
            return Err("graph has a cycle".into());
        }
        let kind = match snarl.get_node(id) {
            Some(EdNode::Op(k)) => *k,
            _ => return Err("Output node cannot be an input".into()),
        };
        for slot in 0..kind.arity() {
            let up = *src
                .get(&(id, slot))
                .ok_or_else(|| format!("node input {slot} is not connected"))?;
            visit(up, snarl, src, order, index, on_stack)?;
        }
        on_stack.remove(&id);
        index.insert(id, order.len() as u32);
        order.push(id);
        Ok(())
    }
    visit(root, snarl, &src, &mut order, &mut index, &mut on_stack)?;

    if order.len() > MAX_GRAPH_NODES {
        return Err(format!("graph has {} nodes (max {MAX_GRAPH_NODES})", order.len()));
    }

    // Emit engine nodes in topological order; inputs reference the (earlier) engine indices.
    let mut nodes = Vec::with_capacity(order.len());
    for &id in &order {
        let kind = match snarl.get_node(id) {
            Some(EdNode::Op(k)) => *k,
            _ => unreachable!(),
        };
        let mut inputs = [0u32; 3];
        for (slot, inp) in inputs.iter_mut().enumerate().take(kind.arity()) {
            *inp = index[&src[&(id, slot)]];
        }
        nodes.push(Node { kind, inputs });
    }
    let graph = Graph { nodes, output: index[&root] };
    graph.validate().map_err(|e| format!("{e:?}"))?;
    Ok(graph)
}

// ===================================================================================================
// SnarlViewer — the node UI
// ===================================================================================================

struct Viewer;

impl SnarlViewer<EdNode> for Viewer {
    fn title(&mut self, node: &EdNode) -> String {
        match node {
            EdNode::Output => "Output".into(),
            EdNode::Op(k) => node_kind_name(k).into(),
        }
    }

    fn inputs(&mut self, node: &EdNode) -> usize {
        match node {
            EdNode::Output => 1,
            EdNode::Op(k) => k.arity(),
        }
    }

    fn outputs(&mut self, node: &EdNode) -> usize {
        match node {
            EdNode::Output => 0,
            EdNode::Op(_) => 1,
        }
    }

    fn show_input(&mut self, _pin: &InPin, _ui: &mut egui::Ui, _snarl: &mut Snarl<EdNode>) -> impl SnarlPin + 'static {
        PinInfo::circle().with_fill(egui::Color32::from_rgb(120, 160, 220))
    }

    fn show_output(&mut self, pin: &OutPin, ui: &mut egui::Ui, snarl: &mut Snarl<EdNode>) -> impl SnarlPin + 'static {
        // Edit the node's params on its (single) output row.
        if let EdNode::Op(kind) = &mut snarl[pin.id.node] {
            node_params_ui(ui, kind);
        }
        PinInfo::circle().with_fill(egui::Color32::from_rgb(160, 210, 140))
    }

    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<EdNode>) {
        // An input takes a single wire: replace any existing connection on the target pin.
        snarl.drop_inputs(to.id);
        snarl.connect(from.id, to.id);
    }

    fn has_graph_menu(&mut self, _pos: egui::Pos2, _snarl: &mut Snarl<EdNode>) -> bool {
        true
    }

    fn show_graph_menu(&mut self, pos: egui::Pos2, ui: &mut egui::Ui, snarl: &mut Snarl<EdNode>) {
        ui.label("Add node");
        for kind in node_catalog() {
            if ui.button(node_kind_name(&kind)).clicked() {
                snarl.insert_node(pos, EdNode::Op(kind));
                ui.close();
            }
        }
    }
}

/// Editable params for a node kind (drawn on its output row).
fn node_params_ui(ui: &mut egui::Ui, kind: &mut NodeKind) {
    match kind {
        NodeKind::Const(v) => {
            ui.add(egui::DragValue::new(v).speed(1.0));
        }
        NodeKind::Scale(k) | NodeKind::Offset(k) => {
            ui.add(egui::DragValue::new(k).speed(1.0));
        }
        NodeKind::Ridge { ridge, amp_sum } => {
            ui.add(egui::DragValue::new(ridge).speed(0.01).range(0.0..=1.0).prefix("ridge "));
            ui.add(egui::DragValue::new(amp_sum).speed(1.0).prefix("amp_sum "));
        }
        NodeKind::Smoothstep { edge0, edge1 } => {
            ui.add(egui::DragValue::new(edge0).speed(0.01).prefix("e0 "));
            ui.add(egui::DragValue::new(edge1).speed(0.01).prefix("e1 "));
        }
        NodeKind::Clamp { lo, hi } => {
            ui.add(egui::DragValue::new(lo).speed(1.0).prefix("lo "));
            ui.add(egui::DragValue::new(hi).speed(1.0).prefix("hi "));
        }
        NodeKind::Fbm(ax) => {
            ui.add(egui::DragValue::new(&mut ax.amplitude).speed(1.0).prefix("amp "));
            let mut wavelength = if ax.base_freq != 0.0 { 1.0 / ax.base_freq } else { 0.0 };
            if ui.add(egui::DragValue::new(&mut wavelength).speed(8.0).prefix("λ ")).changed() && wavelength > 0.0 {
                ax.base_freq = 1.0 / wavelength;
            }
            ui.add(egui::DragValue::new(&mut ax.octaves).range(1..=8).prefix("oct "));
        }
        NodeKind::Curve(_) => {
            ui.label("curve");
        }
        // WorldX/WorldZ/Abs/Neg/Add/Sub/Mul/Min/Max/Mix — no scalar params.
        _ => {}
    }
}

/// The palette of node kinds offered by the add-node menu (sensible defaults).
fn node_catalog() -> Vec<NodeKind> {
    vec![
        NodeKind::Fbm(FbmAxis { octaves: 4, base_freq: 1.0 / 1024.0, lacunarity: 2.0, gain: 0.5, amplitude: 100.0, seed_salt: 1 }),
        NodeKind::Ridge { ridge: 0.85, amp_sum: 200.0 },
        NodeKind::Curve(Spline::new(&[(-1.0, 0.0), (0.0, 0.5), (1.0, 1.0)])),
        NodeKind::Smoothstep { edge0: 0.0, edge1: 1.0 },
        NodeKind::Mix,
        NodeKind::Add,
        NodeKind::Sub,
        NodeKind::Mul,
        NodeKind::Min,
        NodeKind::Max,
        NodeKind::Clamp { lo: -1000.0, hi: 1000.0 },
        NodeKind::Scale(1.0),
        NodeKind::Offset(0.0),
        NodeKind::Abs,
        NodeKind::Neg,
        NodeKind::Const(0.0),
        NodeKind::WorldX,
        NodeKind::WorldZ,
    ]
}

fn node_kind_name(k: &NodeKind) -> &'static str {
    match k {
        NodeKind::WorldX => "WorldX",
        NodeKind::WorldZ => "WorldZ",
        NodeKind::Const(_) => "Const",
        NodeKind::Fbm(_) => "Fbm",
        NodeKind::Curve(_) => "Curve",
        NodeKind::Ridge { .. } => "Ridge",
        NodeKind::Clamp { .. } => "Clamp",
        NodeKind::Smoothstep { .. } => "Smoothstep",
        NodeKind::Scale(_) => "Scale",
        NodeKind::Offset(_) => "Offset",
        NodeKind::Abs => "Abs",
        NodeKind::Neg => "Neg",
        NodeKind::Add => "Add",
        NodeKind::Sub => "Sub",
        NodeKind::Mul => "Mul",
        NodeKind::Min => "Min",
        NodeKind::Max => "Max",
        NodeKind::Mix => "Mix",
    }
}

// `SnarlPin` is the trait the show_input/show_output return values implement (PinInfo does).
use egui_snarl::ui::SnarlPin;

// ===================================================================================================
// Panel
// ===================================================================================================

fn graph_panel(world: &mut World, ui: &mut egui::Ui) {
    // Seed the editor graph from the live WorldGraph once.
    world.resource_scope::<WorldGraphEditor, ()>(|world, mut editor| {
        if !editor.seeded {
            let g = world.resource::<WorldGraph>().0.clone();
            editor.snarl = graph_to_snarl(&g);
            editor.seeded = true;
        }

        ui.horizontal(|ui| {
            if ui.button("Apply").clicked() {
                match snarl_to_graph(&editor.snarl) {
                    Ok(g) => {
                        world.resource_mut::<WorldGraph>().0 = Arc::new(g);
                    }
                    Err(e) => {
                        warn!("biome graph invalid: {e}");
                    }
                }
            }
            if ui.button("Reset to mountains/plains").clicked() {
                let g = mountains_plains_graph(
                    crate::sdf_render::worldgen::graph::preset::MOUNTAINS_PLAINS_AMPLITUDE,
                );
                editor.snarl = graph_to_snarl(&g);
            }
            // Live validity hint.
            match snarl_to_graph(&editor.snarl) {
                Ok(g) => ui.label(format!("{} nodes ✓", g.nodes.len())),
                Err(e) => ui.colored_label(egui::Color32::from_rgb(220, 120, 120), e),
            };
        });

        SnarlWidget::new()
            .id(egui::Id::new("worldgen-biome-graph"))
            .style(SnarlStyle::new())
            .show(&mut editor.snarl, &mut Viewer, ui);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_render::worldgen::graph::preset::{default_terrain_graph, mountains_plains_graph};

    /// engine Graph → Snarl → engine Graph must round-trip to an EVALUATION-equivalent graph. (Not
    /// structurally equal: `snarl_to_graph` topologically re-orders nodes, so indices differ — but the
    /// DAG + params are preserved, so it evaluates bit-for-bit identically.)
    #[test]
    fn graph_snarl_round_trip() {
        for g in [
            mountains_plains_graph(700.0),
            default_terrain_graph(
                FbmAxis { octaves: 6, base_freq: 1.0 / 1536.0, lacunarity: 2.0, gain: 0.5, amplitude: 280.0, seed_salt: 0 },
                0.5,
                551.25,
                0.0,
            ),
        ] {
            let snarl = graph_to_snarl(&g);
            let back = snarl_to_graph(&snarl).expect("convert back");
            for &(x, z) in &[(0.0, 0.0), (123.0, -456.0), (2000.0, 1000.0), (-800.0, 300.0)] {
                let (a, b) = (g.eval(x, z, 7), back.eval(x, z, 7));
                assert_eq!(a.v.to_bits(), b.v.to_bits(), "value at ({x},{z}) after Snarl round-trip");
                assert_eq!(a.dx.to_bits(), b.dx.to_bits(), "∂x at ({x},{z}) after Snarl round-trip");
                assert_eq!(a.dz.to_bits(), b.dz.to_bits(), "∂z at ({x},{z}) after Snarl round-trip");
            }
        }
    }

    #[test]
    fn missing_output_is_an_error() {
        let mut snarl = Snarl::new();
        snarl.insert_node(egui::pos2(0.0, 0.0), EdNode::Op(NodeKind::Const(1.0)));
        assert!(snarl_to_graph(&snarl).is_err());
    }
}
