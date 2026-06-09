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

use crate::assets::Asset as _;
use crate::sdf_render::worldgen::WorldGraph;
use crate::sdf_render::worldgen::graph::GraphAsset;
use crate::sdf_render::worldgen::graph::node::{FbmAxis, Graph, Node, NodeKind};
use crate::sdf_render::worldgen::graph::preset::{MAX_GRAPH_NODES, mountains_plains_graph};
use crate::sdf_render::worldgen::spline::Spline;

/// Default on-disk path the editor saves/loads the active biome graph to (the production graph the
/// worldgen loads — see `WorldGenPlugin`'s asset hot-reload). Relative to the app's `assets/` root.
const DEFAULT_GRAPH_PATH: &str = "assets/worldgen/mountains_plains.graph.ron";

/// A node in the editor graph: an engine operation, or the single graph OUTPUT sink (1 input, 0
/// outputs) that designates which node's value is the terrain height.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum EdNode {
    Op(NodeKind),
    Output,
}

/// Editor state: the working Snarl graph, whether it's been seeded from the live `WorldGraph` yet, and
/// the RON save/load path.
#[derive(Resource)]
pub struct WorldGraphEditor {
    snarl: Snarl<EdNode>,
    seeded: bool,
    path: String,
    /// Last save/load status message (shown in the toolbar).
    status: String,
    /// Per-node preview textures, recomputed each frame a node's preview is open (so param edits are
    /// reflected live) and dropped when collapsed. Keyed by the Snarl node id.
    previews: std::collections::HashMap<NodeId, egui::TextureHandle>,
    /// Which nodes have their preview COLLAPSED. Previews are on by default, so absence ⇒ open.
    collapsed: std::collections::HashSet<NodeId>,
    /// Per-node preview zoom: half-extent (metres) of the sampled world window. Absence ⇒ default. Shared
    /// by the 2D heatmap (grid extent) and the 3D surface (camera framing).
    zoom_half_m: std::collections::HashMap<NodeId, f64>,
    /// Which nodes show the 3D SDF-raymarched surface instead of the 2D heatmap. Absence ⇒ 2D.
    surface: std::collections::HashSet<NodeId>,
    /// Per-node 3D-preview orbit camera (yaw, pitch) in radians. Absence ⇒ default angle.
    cam: std::collections::HashMap<NodeId, (f32, f32)>,
}

impl Default for WorldGraphEditor {
    fn default() -> Self {
        Self {
            snarl: Snarl::new(),
            seeded: false,
            path: DEFAULT_GRAPH_PATH.to_string(),
            status: String::new(),
            previews: std::collections::HashMap::new(),
            collapsed: std::collections::HashSet::new(),
            zoom_half_m: std::collections::HashMap::new(),
            surface: std::collections::HashSet::new(),
            cam: std::collections::HashMap::new(),
        }
    }
}

/// Default 3D orbit camera (yaw, pitch) in radians.
const CAM_DEFAULT: (f32, f32) = (0.7, 0.6);

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
    graph_rooted_at(snarl, root)
}

/// Convert the sub-graph feeding `root` (which must be an Op node) into an engine [`Graph`] whose output
/// is that node. Shared by [`snarl_to_graph`] (root = the node wired to `Output`) and the per-node 2D
/// preview (root = the previewed node). Errors on a cycle, an unconnected input, or too many nodes.
pub fn graph_rooted_at(snarl: &Snarl<EdNode>, root: NodeId) -> Result<Graph, String> {
    use std::collections::HashMap;

    // Source feeding (node, input slot) → the upstream node id (every node has one output, pin 0).
    let mut src: HashMap<(NodeId, usize), NodeId> = HashMap::new();
    for (out, inp) in snarl.wires() {
        src.insert((inp.node, inp.input), out.node);
    }

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

/// The Snarl UI viewer. Borrows the editor's per-node preview caches for the frame so each node can
/// draw a (default-on, collapsible, resizable, zoomable) 2D heatmap of its sub-graph (see
/// [`Viewer::show_body`]).
struct Viewer<'a> {
    previews: &'a mut std::collections::HashMap<NodeId, egui::TextureHandle>,
    collapsed: &'a mut std::collections::HashSet<NodeId>,
    zoom_half_m: &'a mut std::collections::HashMap<NodeId, f64>,
    surface: &'a mut std::collections::HashSet<NodeId>,
    cam: &'a mut std::collections::HashMap<NodeId, (f32, f32)>,
}

impl SnarlViewer<EdNode> for Viewer<'_> {
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

    // Every Op node gets a (default-on) collapsible 2D preview in its body.
    fn has_body(&mut self, node: &EdNode) -> bool {
        matches!(node, EdNode::Op(_))
    }

    fn show_body(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<EdNode>,
    ) {
        // Toggle row (sits ABOVE the preview): collapse/expand + zoom. Previews are on by default.
        let open = !self.collapsed.contains(&node);
        ui.horizontal(|ui| {
            if ui
                .small_button(if open { "▾ Preview" } else { "▸ Preview" })
                .on_hover_text("2D top-down heatmap of THIS node's output — drag the ⤢ corner to resize, set the zoom, updates live.")
                .clicked()
            {
                if open {
                    self.collapsed.insert(node);
                } else {
                    self.collapsed.remove(&node);
                }
            }
            if open {
                let half = self.zoom_half_m.entry(node).or_insert(PREVIEW_HALF_M);
                let mut km = *half * 2.0 / 1000.0; // window width in km
                if ui
                    .add(egui::DragValue::new(&mut km).speed(0.25).range(0.25..=128.0).suffix(" km"))
                    .on_hover_text("Zoom: width of the sampled world window (smaller = zoomed in)")
                    .changed()
                {
                    *half = (km * 1000.0 / 2.0).max(1.0);
                }
                // 2D heatmap ⇆ 3D SDF-raymarched surface.
                let is3d = self.surface.contains(&node);
                if ui
                    .selectable_label(is3d, "3D")
                    .on_hover_text("Toggle a 3D SDF-raymarched surface — drag to orbit, scroll or the km field to zoom/scale")
                    .clicked()
                {
                    if is3d {
                        self.surface.remove(&node);
                    } else {
                        self.surface.insert(node);
                    }
                }
            }
        });
        if !open {
            self.previews.remove(&node); // free the GPU texture while collapsed
            return;
        }
        // Re-evaluate the sub-graph rooted at this node every frame (so edits show immediately) → texture.
        // An unconnected input just shows a hint instead of a preview.
        let half = *self.zoom_half_m.get(&node).unwrap_or(&PREVIEW_HALF_M);
        match graph_rooted_at(snarl, node) {
            Ok(g) => {
                let img = if self.surface.contains(&node) {
                    let (yaw, pitch) = *self.cam.get(&node).unwrap_or(&CAM_DEFAULT);
                    render_surface_preview(&g, half, yaw, pitch)
                } else {
                    render_field_preview(&g, half)
                };
                let handle =
                    ui.ctx().load_texture(format!("wg-preview-{node:?}"), img, egui::TextureOptions::LINEAR);
                // Resizable region → dragging its corner grows the node. Image fills it (kept square).
                let resp = egui::Resize::default()
                    .id_salt(("wg-prev", node))
                    .default_size([130.0, 130.0])
                    .min_size([64.0, 64.0])
                    .show(ui, |ui| {
                        let s = ui.available_size();
                        let d = s.x.min(s.y).max(48.0);
                        ui.image(egui::load::SizedTexture::new(handle.id(), egui::vec2(d, d)))
                    });
                // 3D camera interaction: drag to orbit, scroll-over to zoom (scale the framed window).
                if self.surface.contains(&node) {
                    let cam = self.cam.entry(node).or_insert(CAM_DEFAULT);
                    if resp.dragged() {
                        let d = resp.drag_delta();
                        cam.0 += d.x * 0.01;
                        cam.1 = (cam.1 - d.y * 0.01).clamp(0.05, 1.5);
                    }
                    if resp.hovered() {
                        let scroll = ui.input(|i| i.raw_scroll_delta.y);
                        if scroll != 0.0 {
                            let h = self.zoom_half_m.entry(node).or_insert(PREVIEW_HALF_M);
                            *h = (*h * (1.0 - scroll as f64 * 0.0015)).clamp(64.0, 200_000.0);
                        }
                    }
                }
                self.previews.insert(node, handle); // keep alive past paint; drops the prior frame's texture
            }
            Err(e) => {
                self.previews.remove(&node);
                ui.colored_label(egui::Color32::from_rgb(200, 150, 120), format!("connect inputs ({e})"));
            }
        }
    }

    fn show_input(&mut self, pin: &InPin, ui: &mut egui::Ui, snarl: &mut Snarl<EdNode>) -> impl SnarlPin + 'static {
        ui.label(input_label(&snarl[pin.id.node], pin.id.input));
        PinInfo::circle().with_fill(egui::Color32::from_rgb(120, 160, 220))
    }

    fn show_output(&mut self, pin: &OutPin, ui: &mut egui::Ui, snarl: &mut Snarl<EdNode>) -> impl SnarlPin + 'static {
        // Edit the node's params on its (single) output row, then label the output pin.
        if let EdNode::Op(kind) = &mut snarl[pin.id.node] {
            node_params_ui(ui, kind);
        }
        ui.label("out");
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
            ui.add(egui::DragValue::new(ridge).speed(0.01).range(0.0..=1.0).prefix("ridge "))
                .on_hover_text("Ridge fold strength: 0 = smooth fBm, 1 = sharp ridged peaks (folds toward 1−|n|). Lower this to calm over-prominent ridgelines.");
            ui.add(egui::DragValue::new(amp_sum).speed(1.0).prefix("amp_sum "))
                .on_hover_text("Expected swing of the input (the fBm's amplitude·Σgain^o). Sets where the fold reflects.");
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
            ui.add(egui::DragValue::new(&mut ax.amplitude).speed(1.0).prefix("amp "))
                .on_hover_text("Height of the biggest (octave-0) wave, in metres — the overall vertical scale.");
            let mut wavelength = if ax.base_freq != 0.0 { 1.0 / ax.base_freq } else { 0.0 };
            if ui
                .add(egui::DragValue::new(&mut wavelength).speed(8.0).prefix("λ "))
                .on_hover_text("Wavelength (m) of the biggest feature: larger = broader, gentler shapes.")
                .changed()
                && wavelength > 0.0
            {
                ax.base_freq = 1.0 / wavelength;
            }
            ui.add(egui::DragValue::new(&mut ax.octaves).range(1..=8).prefix("oct "))
                .on_hover_text("How many noise layers to sum — more octaves = finer detail (each half as tall, twice as fine).");
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
// Auto-arrange
// ===================================================================================================

/// Lay the graph out left→right by dependency depth: each node's column = the longest input-chain to a
/// leaf, rows stack within a column. Pure function of the wiring, so the layout is stable + readable.
fn auto_arrange(snarl: &mut Snarl<EdNode>) {
    use std::collections::{HashMap, HashSet};
    const COL: f32 = 280.0;
    const ROW: f32 = 300.0;

    // Upstream nodes feeding each node (over all input slots).
    let mut up: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for (out, inp) in snarl.wires() {
        up.entry(inp.node).or_default().push(out.node);
    }

    fn depth(
        id: NodeId,
        up: &HashMap<NodeId, Vec<NodeId>>,
        memo: &mut HashMap<NodeId, i32>,
        on_stack: &mut HashSet<NodeId>,
    ) -> i32 {
        if let Some(&d) = memo.get(&id) {
            return d;
        }
        if !on_stack.insert(id) {
            return 0; // cycle guard (validation rejects cycles elsewhere)
        }
        let d = match up.get(&id) {
            Some(parents) if !parents.is_empty() => {
                parents.iter().map(|&p| depth(p, up, memo, on_stack)).max().unwrap_or(-1) + 1
            }
            _ => 0,
        };
        on_stack.remove(&id);
        memo.insert(id, d);
        d
    }

    let mut ids: Vec<NodeId> = snarl.node_ids().map(|(id, _)| id).collect();
    ids.sort(); // stable order within a column
    let mut memo = HashMap::new();
    let mut on_stack = HashSet::new();
    let mut row_in_col: HashMap<i32, f32> = HashMap::new();
    for id in ids {
        let d = depth(id, &up, &mut memo, &mut on_stack);
        let row = row_in_col.entry(d).or_insert(0.0);
        if let Some(node) = snarl.get_node_info_mut(id) {
            node.pos = egui::pos2(d as f32 * COL, *row * ROW);
        }
        *row += 1.0;
    }
}

// ===================================================================================================
// Per-node 2D preview
// ===================================================================================================

/// Heatmap resolution (px per side) of a node preview — small + recomputed every frame, so keep modest.
const PREVIEW_RES: usize = 48;
/// Default half-extent (metres) of the world window a preview samples, centred on the origin.
const PREVIEW_HALF_M: f64 = 2048.0;
/// Seed used for previews (matches the default world seed so the heatmap mirrors the live terrain).
const PREVIEW_SEED: u64 = 7;

/// Human label for a node's input pin `slot` (shown beside the pin).
fn input_label(node: &EdNode, slot: usize) -> &'static str {
    match node {
        EdNode::Output => "height",
        EdNode::Op(k) => match k {
            NodeKind::Add | NodeKind::Sub | NodeKind::Mul | NodeKind::Min | NodeKind::Max => {
                if slot == 0 { "a" } else { "b" }
            }
            NodeKind::Mix => match slot {
                0 => "a",
                1 => "b",
                _ => "t",
            },
            NodeKind::Ridge { .. }
            | NodeKind::Curve(_)
            | NodeKind::Smoothstep { .. }
            | NodeKind::Clamp { .. }
            | NodeKind::Scale(_)
            | NodeKind::Offset(_)
            | NodeKind::Abs
            | NodeKind::Neg => "x",
            _ => "in",
        },
    }
}

/// Evaluate a node's sub-[`Graph`] over a top-down grid spanning ±`half_m` metres, normalise to its own
/// min/max, and colour-map it into an [`egui::ColorImage`] (terrain ramp: low = blue/green → high = white).
fn render_field_preview(g: &Graph, half_m: f64) -> egui::ColorImage {
    let n = PREVIEW_RES;
    let mut vals = vec![0.0f64; n * n];
    let span_m = 2.0 * half_m;
    for j in 0..n {
        for i in 0..n {
            let wx = -half_m + (i as f64 + 0.5) / n as f64 * span_m;
            let wz = -half_m + (j as f64 + 0.5) / n as f64 * span_m;
            vals[j * n + i] = g.eval(wx, wz, PREVIEW_SEED).v;
        }
    }
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in &vals {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    let span = if hi > lo { hi - lo } else { 1.0 };
    let pixels: Vec<egui::Color32> =
        vals.iter().map(|&v| terrain_ramp((((v - lo) / span).clamp(0.0, 1.0)) as f32)).collect();
    egui::ColorImage::new([n, n], pixels)
}

/// Map a normalised height `t∈[0,1]` to a terrain colour ramp (deep → grass → rock → snow) as linear rgb.
fn ramp_rgb(t: f32) -> [f32; 3] {
    const STOPS: [(f32, [f32; 3]); 5] = [
        (0.0, [0.12, 0.24, 0.47]),
        (0.35, [0.24, 0.55, 0.35]),
        (0.6, [0.47, 0.59, 0.27]),
        (0.8, [0.55, 0.47, 0.35]),
        (1.0, [0.94, 0.94, 0.96]),
    ];
    let mut c = STOPS[STOPS.len() - 1].1;
    for w in STOPS.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 {
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            c = [c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, c0[2] + (c1[2] - c0[2]) * f];
            break;
        }
    }
    c
}

/// Map a normalised height `t∈[0,1]` to a terrain ramp colour.
fn terrain_ramp(t: f32) -> egui::Color32 {
    let c = ramp_rgb(t);
    egui::Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8)
}

// ===================================================================================================
// 3D SDF-raymarched surface preview
// ===================================================================================================

/// Surface-preview render resolution (px per side) — opt-in per node, so a touch sharper than the 2D map.
const SURFACE_RES: usize = 64;
/// Max march steps per ray through the surface's bounding box.
const SURFACE_STEPS: usize = 80;

/// Render the node's height field as a 3D **SDF-raymarched** surface (heightfield ray–surface
/// intersection) into an [`egui::ColorImage`]. The camera orbits the ±`half_m` window at (`yaw`,`pitch`),
/// framing the field's own height range; shading uses the analytic gradient normal + a terrain ramp.
fn render_surface_preview(g: &Graph, half_m: f64, yaw: f32, pitch: f32) -> egui::ColorImage {
    use bevy::math::Vec3;
    let res = SURFACE_RES;

    // Coarse pass: the field's height range over the window (for camera framing + colour normalisation).
    let (mut hmin, mut hmax) = (f64::INFINITY, f64::NEG_INFINITY);
    let cg = 16usize;
    for j in 0..cg {
        for i in 0..cg {
            let wx = -half_m + (i as f64 + 0.5) / cg as f64 * 2.0 * half_m;
            let wz = -half_m + (j as f64 + 0.5) / cg as f64 * 2.0 * half_m;
            let v = g.eval(wx, wz, PREVIEW_SEED).v;
            if v.is_finite() {
                hmin = hmin.min(v);
                hmax = hmax.max(v);
            }
        }
    }
    if !hmin.is_finite() {
        hmin = 0.0;
        hmax = 1.0;
    }
    let span = (hmax - hmin).max(1.0);

    let half = half_m as f32;
    let (ymin, ymax) = (hmin as f32, hmax as f32);
    let pad = span as f32 * 0.08 + 1.0;
    let (bmin, bmax) = (Vec3::new(-half, ymin - pad, -half), Vec3::new(half, ymax + pad, half));

    // Orbit camera framing the box centre.
    let centre = Vec3::new(0.0, (ymin + ymax) * 0.5, 0.0);
    let dist = half * 2.4 + span as f32;
    let (sp, cp) = (pitch.sin(), pitch.cos());
    let (sy, cyaw) = (yaw.sin(), yaw.cos());
    let eye = centre + Vec3::new(cp * cyaw, sp, cp * sy) * dist;
    let fwd = (centre - eye).normalize();
    let right = fwd.cross(Vec3::Y).normalize_or_zero();
    let up = right.cross(fwd);
    let tan = (0.6f32 * 0.5).tan() * 2.0; // ~ vertical half-extent at the image plane
    let light = Vec3::new(0.4, 0.85, 0.3).normalize();

    let mut pixels = Vec::with_capacity(res * res);
    for py in 0..res {
        for px in 0..res {
            let ndcx = (px as f32 + 0.5) / res as f32 * 2.0 - 1.0;
            let ndcy = 1.0 - (py as f32 + 0.5) / res as f32 * 2.0;
            let dir = (fwd + right * (ndcx * tan) + up * (ndcy * tan)).normalize();
            let col = match ray_box(eye, dir, bmin, bmax) {
                Some((t0, t1)) => march_surface(g, eye, dir, t0.max(0.0), t1, hmin, span, light, ndcy),
                None => sky(ndcy),
            };
            pixels.push(col);
        }
    }
    egui::ColorImage::new([res, res], pixels)
}

/// Slab ray–AABB intersection → entry/exit `t` (or `None`).
fn ray_box(o: bevy::math::Vec3, d: bevy::math::Vec3, bmin: bevy::math::Vec3, bmax: bevy::math::Vec3) -> Option<(f32, f32)> {
    let inv = bevy::math::Vec3::ONE / d;
    let t1 = (bmin - o) * inv;
    let t2 = (bmax - o) * inv;
    let tmin = t1.min(t2).max_element();
    let tmax = t1.max(t2).min_element();
    if tmax >= tmin.max(0.0) { Some((tmin, tmax)) } else { None }
}

/// March a ray through the heightfield between `t0..t1`; on the first downward crossing shade with the
/// analytic-gradient normal + terrain ramp, else return sky.
#[allow(clippy::too_many_arguments)]
fn march_surface(
    g: &Graph,
    eye: bevy::math::Vec3,
    dir: bevy::math::Vec3,
    t0: f32,
    t1: f32,
    hmin: f64,
    span: f64,
    light: bevy::math::Vec3,
    ndcy: f32,
) -> egui::Color32 {
    use bevy::math::Vec3;
    let diff = |t: f32| -> f64 {
        let p = eye + dir * t;
        p.y as f64 - g.eval(p.x as f64, p.z as f64, PREVIEW_SEED).v
    };
    let dt = (t1 - t0) / SURFACE_STEPS as f32;
    if dt <= 0.0 {
        return sky(ndcy);
    }
    let mut t = t0;
    let mut prev = diff(t);
    for _ in 0..SURFACE_STEPS {
        let tn = t + dt;
        let cur = diff(tn);
        if cur <= 0.0 && prev > 0.0 {
            // Bisect the crossing for a crisp silhouette.
            let (mut a, mut b) = (t, tn);
            for _ in 0..6 {
                let m = (a + b) * 0.5;
                if diff(m) > 0.0 {
                    a = m;
                } else {
                    b = m;
                }
            }
            let pm = eye + dir * ((a + b) * 0.5);
            let f = g.eval(pm.x as f64, pm.z as f64, PREVIEW_SEED);
            let n = Vec3::new(-f.dx as f32, 1.0, -f.dz as f32).normalize();
            let lamb = n.dot(light).clamp(0.0, 1.0);
            let base = ramp_rgb((((f.v - hmin) / span).clamp(0.0, 1.0)) as f32);
            let lit = 0.28 + 0.72 * lamb;
            return egui::Color32::from_rgb(
                (base[0] * lit * 255.0) as u8,
                (base[1] * lit * 255.0) as u8,
                (base[2] * lit * 255.0) as u8,
            );
        }
        prev = cur;
        t = tn;
    }
    sky(ndcy)
}

/// Background sky gradient for ray misses (darker low, lighter high).
fn sky(ndcy: f32) -> egui::Color32 {
    let t = (ndcy * 0.5 + 0.5).clamp(0.0, 1.0);
    let r = (30.0 + 40.0 * t) as u8;
    let g = (38.0 + 55.0 * t) as u8;
    let b = (55.0 + 75.0 * t) as u8;
    egui::Color32::from_rgb(r, g, b)
}

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
            // APPLY — rebuild the engine graph + push it live into the world (roll_worldgen re-meshes).
            if ui.button("Apply").on_hover_text("Rebuild + drive the live world terrain from this graph").clicked() {
                match snarl_to_graph(&editor.snarl) {
                    Ok(g) => {
                        world.resource_mut::<WorldGraph>().0 = Arc::new(g);
                        editor.status = "applied to world".into();
                    }
                    Err(e) => editor.status = format!("invalid: {e}"),
                }
            }
            // SAVE — write the graph to its .ron (the production graph the worldgen hot-reloads).
            if ui.button("Save").on_hover_text("Write to the .ron asset (the world hot-reloads it)").clicked() {
                editor.status = match snarl_to_graph(&editor.snarl) {
                    Ok(g) => match (GraphAsset { graph: g }).save(std::path::Path::new(&editor.path)) {
                        Ok(()) => format!("saved {}", editor.path),
                        Err(e) => format!("save failed: {e}"),
                    },
                    Err(e) => format!("invalid: {e}"),
                };
            }
            // LOAD — read the .ron back into the editor.
            if ui.button("Load").clicked() {
                editor.status = match std::fs::read_to_string(&editor.path) {
                    Ok(s) => match ron::de::from_str::<GraphAsset>(&s) {
                        Ok(asset) => {
                            editor.snarl = graph_to_snarl(&asset.graph);
                            format!("loaded {}", editor.path)
                        }
                        Err(e) => format!("parse failed: {e}"),
                    },
                    Err(e) => format!("read failed: {e}"),
                };
            }
            if ui.button("Reset").on_hover_text("Restore the built-in mountains/plains graph").clicked() {
                let g = mountains_plains_graph(
                    crate::sdf_render::worldgen::graph::preset::MOUNTAINS_PLAINS_AMPLITUDE,
                );
                editor.snarl = graph_to_snarl(&g);
                editor.status = "reset to default".into();
            }
            if ui.button("Auto-arrange").on_hover_text("Lay nodes out left→right by dependency depth").clicked() {
                auto_arrange(&mut editor.snarl);
                editor.status = "arranged".into();
            }
        });
        ui.horizontal(|ui| {
            ui.label("Path:");
            // Borrow path mutably without conflicting with the snarl borrow below.
            let path = &mut editor.path;
            ui.add(egui::TextEdit::singleline(path).desired_width(360.0));
        });
        // Live validity hint + last status.
        ui.horizontal(|ui| {
            match snarl_to_graph(&editor.snarl) {
                Ok(g) => ui.colored_label(egui::Color32::from_rgb(140, 200, 140), format!("{} nodes ✓", g.nodes.len())),
                Err(e) => ui.colored_label(egui::Color32::from_rgb(220, 120, 120), e),
            };
            if !editor.status.is_empty() {
                ui.label(format!("· {}", editor.status));
            }
        });

        // Disjoint borrows of the editor's fields: the snarl is the working graph; the rest are the
        // per-node preview caches the Viewer drives.
        let WorldGraphEditor { snarl, previews, collapsed, zoom_half_m, surface, cam, .. } = &mut *editor;
        let mut viewer = Viewer { previews, collapsed, zoom_half_m, surface, cam };
        SnarlWidget::new()
            .id(egui::Id::new("worldgen-biome-graph"))
            .style(SnarlStyle::new())
            .show(snarl, &mut viewer, ui);
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
