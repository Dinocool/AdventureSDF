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

/// The climate axes a biome can read from its parent (its input pins, in order). Expandable: add an
/// axis here and biomes gain a pin for it. The parent graph drives these (low-freq Fbm / derived math)
/// and they place + shape biomes.
pub const CLIMATE_INPUTS: [&str; 4] = ["continentalness", "temperature", "humidity", "weirdness"];

/// Name of climate input `k` (falls back gracefully past the vocabulary).
fn climate_name(k: usize) -> &'static str {
    CLIMATE_INPUTS.get(k).copied().unwrap_or("input")
}

/// A node in the editor graph. Biomes are a purely **editor-side** grouping: a biome owns its own
/// sub-graph and is *inlined* into the flat engine [`Graph`] at compile time (climate input pins → the
/// parent edges feeding them; one height out), so the engine, determinism, and parity are unchanged.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum EdNode {
    Op(NodeKind),
    /// A biome group node: climate inputs in ([`CLIMATE_INPUTS`]), one height out; its `graph` is the
    /// biome's terrain shape, inlined at compile.
    Biome { name: String, graph: Box<Snarl<EdNode>> },
    /// Inside a biome's sub-graph: the Nth climate input piped down from the parent biome node's pins.
    Input(usize),
    /// The single graph OUTPUT sink (1 input, 0 outputs) — its input is the terrain height.
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
    /// Last-frame body content size per node (egui can't expose the node rect), used by `auto_arrange`
    /// to pack columns/rows by real size instead of a fixed grid.
    body_size: std::collections::HashMap<NodeId, egui::Vec2>,
    /// Last-frame on-screen preview square side (points) per node, used to pick the render resolution so
    /// previews stay crisp as the node is resized.
    disp_px: std::collections::HashMap<NodeId, f32>,
    /// Navigation stack of biome nodes we've descended into (empty ⇒ the top "World" graph). The shown
    /// snarl is `snarl` walked through each biome's sub-graph. (Distinct from `path`, the save file path.)
    nav: Vec<NodeId>,
    /// Set by the Viewer when the user clicks a biome's "Open"; the panel descends into it after the show.
    enter: Option<NodeId>,
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
            body_size: std::collections::HashMap::new(),
            disp_px: std::collections::HashMap::new(),
            nav: Vec::new(),
            enter: None,
        }
    }
}

impl WorldGraphEditor {
    /// Drop all per-node UI caches — called on navigation, since `NodeId`s are per-snarl-level (a fresh
    /// id namespace each level) so caches must not bleed between levels.
    fn clear_node_caches(&mut self) {
        self.previews.clear();
        self.collapsed.clear();
        self.zoom_half_m.clear();
        self.surface.clear();
        self.cam.clear();
        self.body_size.clear();
        self.disp_px.clear();
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

/// Find the node whose output feeds the single `Output` sink of `snarl` (the (sub)graph's root).
fn output_root(snarl: &Snarl<EdNode>) -> Result<NodeId, String> {
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

// ===================================================================================================
// Biome navigation (drill into a biome's sub-graph; breadcrumb back out)
// ===================================================================================================

/// How many leading `path` entries still resolve to live biome nodes (trailing stale ids dropped).
fn valid_depth(root: &Snarl<EdNode>, path: &[NodeId]) -> usize {
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
fn breadcrumb_names(root: &Snarl<EdNode>, path: &[NodeId]) -> Vec<String> {
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

/// The snarl shown at the current `path` (mutable). `path` must be valid (see [`valid_depth`]).
fn current_snarl_mut<'a>(root: &'a mut Snarl<EdNode>, path: &[NodeId]) -> &'a mut Snarl<EdNode> {
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
fn new_biome_subgraph() -> Snarl<EdNode> {
    let mut s = Snarl::new();
    for k in 0..CLIMATE_INPUTS.len() {
        s.insert_node(egui::pos2(0.0, 60.0 * k as f32), EdNode::Input(k));
    }
    let c = s.insert_node(egui::pos2(260.0, 0.0), EdNode::Op(NodeKind::Const(0.0)));
    let o = s.insert_node(egui::pos2(520.0, 0.0), EdNode::Output);
    s.connect(OutPinId { node: c, output: 0 }, InPinId { node: o, input: 0 });
    s
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
    body_size: &'a mut std::collections::HashMap<NodeId, egui::Vec2>,
    disp_px: &'a mut std::collections::HashMap<NodeId, f32>,
    /// Set to a biome node id when the user clicks its "Open" — the panel descends after the show.
    enter: &'a mut Option<NodeId>,
}

impl SnarlViewer<EdNode> for Viewer<'_> {
    fn title(&mut self, node: &EdNode) -> String {
        match node {
            EdNode::Output => "Output".into(),
            EdNode::Op(k) => node_kind_name(k).into(),
            EdNode::Biome { name, .. } => format!("🌱 {name}"),
            EdNode::Input(k) => format!("⮡ {}", climate_name(*k)),
        }
    }

    fn inputs(&mut self, node: &EdNode) -> usize {
        match node {
            EdNode::Output => 1,
            EdNode::Op(k) => k.arity(),
            EdNode::Biome { .. } => CLIMATE_INPUTS.len(),
            EdNode::Input(_) => 0,
        }
    }

    fn outputs(&mut self, node: &EdNode) -> usize {
        match node {
            EdNode::Output => 0,
            EdNode::Op(_) | EdNode::Biome { .. } | EdNode::Input(_) => 1,
        }
    }

    // Op + Biome nodes get a body (preview / biome controls); Input + Output don't.
    fn has_body(&mut self, node: &EdNode) -> bool {
        matches!(node, EdNode::Op(_) | EdNode::Biome { .. })
    }

    fn show_body(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<EdNode>,
    ) {
        ui.set_max_width(NODE_BODY_MAX_W);
        // Node params, stacked VERTICALLY at the top of the body (keeps nodes narrow); preview below.
        match &mut snarl[node] {
            EdNode::Op(kind) => node_params_ui(ui, kind),
            EdNode::Biome { name, .. } => {
                ui.add(egui::TextEdit::singleline(name).desired_width(120.0).hint_text("biome name"));
            }
            _ => {}
        }
        if matches!(snarl.get_node(node), Some(EdNode::Biome { .. }))
            && ui.button("Open ▸").on_hover_text("Edit this biome's sub-graph").clicked()
        {
            *self.enter = Some(node);
        }
        // Divider between the node params (above) and the preview section (options row + preview below).
        ui.separator();
        // Preview-options row (sits ABOVE the preview): collapse/expand + zoom + 2D/3D. Previews on by default.
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
            self.body_size.insert(node, ui.min_rect().size());
            return;
        }
        // Render resolution tracks the displayed (physical-pixel) size from last frame, so the preview
        // gains real detail as the node is resized — capped per mode (3D raymarch is far costlier).
        let is3d = self.surface.contains(&node);
        let disp = self.disp_px.get(&node).copied().unwrap_or(130.0);
        let ppp = ui.ctx().pixels_per_point();
        let res = ((disp * ppp).round() as usize).clamp(32, if is3d { SURFACE_RES_MAX } else { PREVIEW_RES_MAX });

        // Re-evaluate the sub-graph rooted at this node every frame (so edits show immediately) → texture.
        // An unconnected input just shows a hint instead of a preview.
        let half = *self.zoom_half_m.get(&node).unwrap_or(&PREVIEW_HALF_M);
        match graph_rooted_at(snarl, node) {
            Ok(g) => {
                let img = if is3d {
                    let (yaw, pitch) = *self.cam.get(&node).unwrap_or(&CAM_DEFAULT);
                    render_surface_preview(&g, half, yaw, pitch, res)
                } else {
                    render_field_preview(&g, half, res)
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
                        let r = ui.image(egui::load::SizedTexture::new(handle.id(), egui::vec2(d, d)));
                        (r, d)
                    });
                let (resp, d) = resp;
                self.disp_px.insert(node, d); // feeds next frame's render resolution
                // 3D camera interaction: drag to orbit, scroll-over to zoom (scale the framed window).
                if is3d {
                    let cam = self.cam.entry(node).or_insert(CAM_DEFAULT);
                    if resp.dragged() {
                        let dd = resp.drag_delta();
                        cam.0 += dd.x * 0.01;
                        cam.1 = (cam.1 - dd.y * 0.01).clamp(0.05, 1.5);
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
        self.body_size.insert(node, ui.min_rect().size());
    }

    fn show_input(&mut self, pin: &InPin, ui: &mut egui::Ui, snarl: &mut Snarl<EdNode>) -> impl SnarlPin + 'static {
        ui.label(input_label(&snarl[pin.id.node], pin.id.input));
        PinInfo::circle().with_fill(egui::Color32::from_rgb(120, 160, 220))
    }

    fn show_output(&mut self, _pin: &OutPin, ui: &mut egui::Ui, _snarl: &mut Snarl<EdNode>) -> impl SnarlPin + 'static {
        // Params now live in the body (stacked vertically) to keep nodes narrow; the pin just gets a label.
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
        ui.separator();
        if ui.button("🌱 Biome").on_hover_text("A nested biome sub-graph (climate in, height out)").clicked() {
            snarl.insert_node(pos, EdNode::Biome { name: "biome".into(), graph: Box::new(new_biome_subgraph()) });
            ui.close();
        }
        ui.menu_button("Climate input", |ui| {
            for (k, name) in CLIMATE_INPUTS.iter().enumerate() {
                if ui.button(*name).on_hover_text("A climate value piped in from the parent biome").clicked() {
                    snarl.insert_node(pos, EdNode::Input(k));
                    ui.close();
                }
            }
        });
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
/// leaf, rows stack within a column. Columns are spaced by their widest node and rows by each node's real
/// height (measured last frame in `body_size`), so preview-laden nodes don't overlap. Pure function of
/// the wiring + measured sizes ⇒ stable + readable.
fn auto_arrange(snarl: &mut Snarl<EdNode>, body_size: &std::collections::HashMap<NodeId, egui::Vec2>) {
    use std::collections::{HashMap, HashSet};
    const GAP_X: f32 = 64.0;
    const GAP_Y: f32 = 34.0;
    const HEADER: f32 = 30.0; // header bar
    const PIN_ROW: f32 = 24.0; // per input/output pin row
    const FRAME: f32 = 22.0; // node frame padding

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

    // Depth + estimated full node size per node (body measured last frame + header/pin/frame allowance).
    let mut depth_of: HashMap<NodeId, i32> = HashMap::new();
    let mut size_of: HashMap<NodeId, (f32, f32)> = HashMap::new();
    let mut max_depth = 0i32;
    for &id in &ids {
        let d = depth(id, &up, &mut memo, &mut on_stack);
        depth_of.insert(id, d);
        max_depth = max_depth.max(d);
        let arity = snarl
            .get_node(id)
            .map(|n| match n {
                EdNode::Output => 1,
                EdNode::Op(k) => k.arity().max(1),
                EdNode::Biome { .. } => CLIMATE_INPUTS.len(),
                EdNode::Input(_) => 1,
            })
            .unwrap_or(1);
        let body = body_size.get(&id).copied().unwrap_or(egui::vec2(120.0, 56.0));
        let w = body.x.max(120.0) + FRAME;
        let h = HEADER + arity as f32 * PIN_ROW + body.y + FRAME;
        size_of.insert(id, (w, h));
    }

    // Column x = prefix sum of each column's widest node + gap.
    let cols = (max_depth + 1) as usize;
    let mut col_w = vec![0.0f32; cols];
    for &id in &ids {
        let w = size_of[&id].0;
        let c = depth_of[&id] as usize;
        if w > col_w[c] {
            col_w[c] = w;
        }
    }
    let mut col_x = vec![0.0f32; cols];
    let mut acc = 0.0;
    for c in 0..cols {
        col_x[c] = acc;
        acc += col_w[c] + GAP_X;
    }

    // Stack rows within each column by real height.
    let mut col_y: HashMap<i32, f32> = HashMap::new();
    for &id in &ids {
        let d = depth_of[&id];
        let h = size_of[&id].1;
        let y = col_y.entry(d).or_insert(0.0);
        if let Some(node) = snarl.get_node_info_mut(id) {
            node.pos = egui::pos2(col_x[d as usize], *y);
        }
        *y += h + GAP_Y;
    }
}

// ===================================================================================================
// Per-node 2D preview
// ===================================================================================================

/// Max heatmap resolution (px per side) of a 2D node preview — actual res tracks the displayed size.
const PREVIEW_RES_MAX: usize = 256;
/// Max render resolution (px per side) of a 3D surface preview (raymarch is far costlier than the heatmap).
const SURFACE_RES_MAX: usize = 160;
/// Max width (points) of a node body — keeps nodes narrow regardless of param/preview content.
const NODE_BODY_MAX_W: f32 = 168.0;
/// Default half-extent (metres) of the world window a preview samples, centred on the origin.
const PREVIEW_HALF_M: f64 = 2048.0;
/// Seed used for previews (matches the default world seed so the heatmap mirrors the live terrain).
const PREVIEW_SEED: u64 = 7;

/// Human label for a node's input pin `slot` (shown beside the pin).
fn input_label(node: &EdNode, slot: usize) -> &'static str {
    match node {
        EdNode::Output => "height",
        EdNode::Input(_) => "",
        EdNode::Biome { .. } => climate_name(slot),
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
fn render_field_preview(g: &Graph, half_m: f64, res: usize) -> egui::ColorImage {
    let n = res.max(2);
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

/// Max march steps per ray through the surface's bounding box.
const SURFACE_STEPS: usize = 80;

/// Render the node's height field as a 3D **SDF-raymarched** surface (heightfield ray–surface
/// intersection) into an [`egui::ColorImage`]. The camera orbits the ±`half_m` window at (`yaw`,`pitch`),
/// framing the field's own height range; shading uses the analytic gradient normal + a terrain ramp.
fn render_surface_preview(g: &Graph, half_m: f64, yaw: f32, pitch: f32, res: usize) -> egui::ColorImage {
    use bevy::math::Vec3;
    let res = res.max(2);

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
                            editor.nav.clear();
                            editor.clear_node_caches();
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
                editor.nav.clear();
                editor.clear_node_caches();
                editor.status = "reset to default".into();
            }
            if ui.button("Auto-arrange").on_hover_text("Lay nodes out left→right by dependency depth").clicked() {
                let WorldGraphEditor { snarl, body_size, .. } = &mut *editor;
                auto_arrange(snarl, body_size);
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

        // Drop any stale tail of the nav path (e.g. a biome was deleted), then a breadcrumb to walk out.
        let valid = valid_depth(&editor.snarl, &editor.nav);
        if valid != editor.nav.len() {
            editor.nav.truncate(valid);
            editor.clear_node_caches();
        }
        let mut nav_to: Option<usize> = None;
        let crumbs = breadcrumb_names(&editor.snarl, &editor.nav);
        ui.horizontal(|ui| {
            if ui.selectable_label(editor.nav.is_empty(), "🌍 World").clicked() {
                nav_to = Some(0);
            }
            for (i, name) in crumbs.iter().enumerate() {
                ui.label("›");
                if ui.selectable_label(i + 1 == editor.nav.len(), format!("🌱 {name}")).clicked() {
                    nav_to = Some(i + 1);
                }
            }
        });
        if let Some(d) = nav_to.filter(|&d| d != editor.nav.len()) {
            editor.nav.truncate(d);
            editor.clear_node_caches();
        }
        ui.separator();

        // Show the snarl at the current nav depth. Disjoint borrows: `snarl`+`nav` resolve the level;
        // the rest are the per-node preview caches the Viewer drives.
        editor.enter = None;
        {
            let WorldGraphEditor {
                snarl, nav, previews, collapsed, zoom_half_m, surface, cam, body_size, disp_px, enter, ..
            } = &mut *editor;
            let current = current_snarl_mut(snarl, nav);
            let mut viewer = Viewer {
                previews, collapsed, zoom_half_m, surface, cam, body_size, disp_px, enter,
            };
            SnarlWidget::new()
                .id(egui::Id::new("worldgen-biome-graph"))
                .style(SnarlStyle::new())
                .show(current, &mut viewer, ui);
        }
        // Descend into a biome the user opened this frame.
        if let Some(id) = editor.enter.take() {
            editor.nav.push(id);
            editor.clear_node_caches();
        }
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

    // -- biome (nested sub-graph) inlining ----------------------------------------------------------
    fn p() -> egui::Pos2 {
        egui::pos2(0.0, 0.0)
    }
    fn out(n: NodeId) -> OutPinId {
        OutPinId { node: n, output: 0 }
    }
    fn inn(n: NodeId, i: usize) -> InPinId {
        InPinId { node: n, input: i }
    }

    /// A biome wrapping a sub-graph must inline to a graph that evaluates bit-for-bit like the same
    /// sub-graph placed flat (no biome) — biomes are pure authoring grouping.
    #[test]
    fn biome_inlines_to_flat_equivalent() {
        let axis = FbmAxis { octaves: 3, base_freq: 1.0 / 512.0, lacunarity: 2.0, gain: 0.5, amplitude: 100.0, seed_salt: 2 };

        let mut flat = Snarl::new();
        let f = flat.insert_node(p(), EdNode::Op(NodeKind::Fbm(axis)));
        let o = flat.insert_node(p(), EdNode::Output);
        flat.connect(out(f), inn(o, 0));
        let flat = snarl_to_graph(&flat).expect("flat");

        let mut sub = Snarl::new();
        let sf = sub.insert_node(p(), EdNode::Op(NodeKind::Fbm(axis)));
        let so = sub.insert_node(p(), EdNode::Output);
        sub.connect(out(sf), inn(so, 0));
        let mut top = Snarl::new();
        let b = top.insert_node(p(), EdNode::Biome { name: "Mountains".into(), graph: Box::new(sub) });
        let o = top.insert_node(p(), EdNode::Output);
        top.connect(out(b), inn(o, 0));
        let nested = snarl_to_graph(&top).expect("nested");

        for &(x, z) in &[(0.0, 0.0), (123.0, -456.0), (2000.0, 1000.0)] {
            let (a, c) = (flat.eval(x, z, 7), nested.eval(x, z, 7));
            assert_eq!(a.v.to_bits(), c.v.to_bits(), "value at ({x},{z})");
            assert_eq!(a.dx.to_bits(), c.dx.to_bits(), "∂x at ({x},{z})");
            assert_eq!(a.dz.to_bits(), c.dz.to_bits(), "∂z at ({x},{z})");
        }
    }

    /// A climate value wired into a biome pin must reach the biome's `Input` sentinel through inlining.
    #[test]
    fn biome_climate_input_is_piped() {
        let mut sub = Snarl::new();
        let inp = sub.insert_node(p(), EdNode::Input(0)); // continentalness
        let c5 = sub.insert_node(p(), EdNode::Op(NodeKind::Const(5.0)));
        let add = sub.insert_node(p(), EdNode::Op(NodeKind::Add));
        sub.connect(out(inp), inn(add, 0));
        sub.connect(out(c5), inn(add, 1));
        let so = sub.insert_node(p(), EdNode::Output);
        sub.connect(out(add), inn(so, 0));

        let mut top = Snarl::new();
        let c10 = top.insert_node(p(), EdNode::Op(NodeKind::Const(10.0)));
        let b = top.insert_node(p(), EdNode::Biome { name: "B".into(), graph: Box::new(sub) });
        top.connect(out(c10), inn(b, 0)); // feed continentalness pin
        let o = top.insert_node(p(), EdNode::Output);
        top.connect(out(b), inn(o, 0));

        let g = snarl_to_graph(&top).expect("piped");
        assert_eq!(g.eval(0.0, 0.0, 7).v, 15.0); // Input(0)=10 + Const 5
    }

    /// A biome `Input` that is used but its parent pin is unconnected is a hard error (no silent 0).
    #[test]
    fn unconnected_used_biome_input_errors() {
        let mut sub = Snarl::new();
        let inp = sub.insert_node(p(), EdNode::Input(0));
        let so = sub.insert_node(p(), EdNode::Output);
        sub.connect(out(inp), inn(so, 0));
        let mut top = Snarl::new();
        let b = top.insert_node(p(), EdNode::Biome { name: "B".into(), graph: Box::new(sub) });
        let o = top.insert_node(p(), EdNode::Output);
        top.connect(out(b), inn(o, 0)); // continentalness pin left unconnected
        assert!(snarl_to_graph(&top).is_err());
    }
}
