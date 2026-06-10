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
use egui_phosphor::regular as icon;
use egui_snarl::ui::{PinInfo, SnarlStyle, SnarlViewer, SnarlWidget};
use egui_snarl::{InPin, InPinId, NodeId, OutPin, OutPinId, Snarl};

use crate::assets::Asset as _;
use crate::sdf_render::worldgen::WorldGraph;
use crate::sdf_render::worldgen::graph::GraphAsset;
use crate::sdf_render::worldgen::graph::node::{FbmAxis, Graph, Node, NodeKind};
use crate::sdf_render::worldgen::graph::preset::MAX_GRAPH_NODES;
use crate::sdf_render::worldgen::spline::Spline;
use super::worldgen_gpu_preview::{GpuPreviewRequest, GpuPreviewRequests, GpuPreviewTextures};

/// Default on-disk path the editor saves/loads the active biome graph to (the production graph the
/// worldgen loads — see `WorldGenPlugin`'s asset hot-reload). Relative to the app's `assets/` root.
const DEFAULT_GRAPH_PATH: &str = "assets/worldgen/world.graph.ron";

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
    /// Per-node pan: world-XZ centre offset of the sampled window (drag-pan / scroll over the preview).
    pan: std::collections::HashMap<NodeId, (f64, f64)>,
    /// Which inline preview image the pointer was over last frame — so `graph_panel` can intercept the
    /// scroll-zoom for it BEFORE egui-snarl applies its own (graph) zoom.
    hovered_preview: Option<NodeId>,
    /// Navigation stack of biome nodes we've descended into (empty ⇒ the top "World" graph). The shown
    /// snarl is `snarl` walked through each biome's sub-graph. (Distinct from `path`, the save file path.)
    nav: Vec<NodeId>,
    /// Set by the Viewer when the user clicks a biome's "Open"; the panel descends into it after the show.
    enter: Option<NodeId>,
    /// Previews "popped out" into floating windows (drag anywhere, incl. over the top panel). Each is
    /// self-contained so it survives navigation and doesn't clash with the in-graph preview caches.
    popped: Vec<PoppedPreview>,
    /// Set by the Viewer when the user clicks a node's pop-out button; the panel snapshots it after show.
    pop_request: Option<NodeId>,
    /// Set by the Viewer when the user clicks "→ panel"; the panel retargets the dockable preview panel.
    to_panel: Option<NodeId>,
    /// Monotonic id source for popped windows (their stable GPU pool key).
    next_pop_id: u64,
    /// Set after a graph is seeded/loaded; the panel auto-arranges once the nodes have been measured.
    needs_arrange: bool,
}

/// The dockable, viewport-located preview panel's state: which node it shows + its own view.
#[derive(Resource)]
pub struct WorldgenPreviewPanel {
    target: Option<(Vec<NodeId>, NodeId)>,
    half: f64,
    cam: (f32, f32),
    pan: (f64, f64),
    is3d: bool,
    /// Set by "→ panel"; a system outside the dock render ensures + focuses the tab (the dock state is
    /// taken OUT of the World while the dock renders, so it can't be touched from a panel callback).
    pending_open: bool,
}

impl Default for WorldgenPreviewPanel {
    fn default() -> Self {
        Self {
            target: None,
            half: PREVIEW_HALF_M,
            cam: CAM_DEFAULT,
            pan: (0.0, 0.0),
            is3d: true,
            pending_open: false,
        }
    }
}

/// Fixed GPU pool key for the dockable preview panel (distinct from inline high-bit keys + pop-out ids).
const PANEL_GPU_KEY: u64 = 7;

/// A node preview detached into its own floating window — carries its own nav path, view state, and
/// texture so it stays live across navigation independently of the in-graph preview.
struct PoppedPreview {
    /// Stable id (the GPU pool slot key for this window — unchanged across rotate/zoom/nav).
    id: u64,
    nav: Vec<NodeId>,
    node: NodeId,
    half: f64,
    /// World-XZ pan centre (offset X/Y).
    cx: f64,
    cz: f64,
    size: f32,
    is3d: bool,
    cam: (f32, f32),
    open: bool,
}

impl Default for WorldGraphEditor {
    fn default() -> Self {
        Self {
            snarl: Snarl::new(),
            seeded: false,
            path: DEFAULT_GRAPH_PATH.to_string(),
            status: String::new(),
            collapsed: std::collections::HashSet::new(),
            zoom_half_m: std::collections::HashMap::new(),
            surface: std::collections::HashSet::new(),
            cam: std::collections::HashMap::new(),
            body_size: std::collections::HashMap::new(),
            disp_px: std::collections::HashMap::new(),
            pan: std::collections::HashMap::new(),
            hovered_preview: None,
            nav: Vec::new(),
            enter: None,
            popped: Vec::new(),
            pop_request: None,
            to_panel: None,
            next_pop_id: 1000,
            needs_arrange: true,
        }
    }
}

impl WorldGraphEditor {
    /// Drop all per-node UI caches — called on navigation, since `NodeId`s are per-snarl-level (a fresh
    /// id namespace each level) so caches must not bleed between levels.
    fn clear_node_caches(&mut self) {
        self.collapsed.clear();
        self.zoom_half_m.clear();
        self.surface.clear();
        self.cam.clear();
        self.body_size.clear();
        self.disp_px.clear();
        self.pan.clear();
    }

    /// Auto-arrange the top-level snarl (plain `&mut self` so the disjoint snarl/body_size borrows don't
    /// alias through `Mut`'s deref).
    fn rearrange(&mut self) {
        auto_arrange(&mut self.snarl, &self.body_size);
    }
}

/// Default 3D orbit camera (yaw, pitch) in radians.
const CAM_DEFAULT: (f32, f32) = (0.7, 0.6);

/// Plugin: registers the editor state + the dockable "Biome Graph" panel.
pub struct WorldgenGraphEditorPlugin;

impl Plugin for WorldgenGraphEditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldGraphEditor>();
        app.init_resource::<WorldgenPreviewPanel>();
        // Deferred dock manipulation (the dock state is removed from the World during its own render).
        app.add_systems(Update, open_preview_panel);
        super::panels::register_panel(
            app,
            "worldgen/graph",
            "Biome Graph",
            super::panels::DockSide::Right,
            30,
            graph_panel,
        );
        // A viewport-located preview panel; "→ panel" on a node targets it.
        super::panels::register_panel(
            app,
            "worldgen/node-preview",
            "Node Preview",
            super::panels::DockSide::Center,
            10,
            preview_panel,
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

/// Resolve the snarl at `nav` (read-only), or `None` if any step no longer points to a biome (e.g. a
/// popped-out preview whose biome was deleted).
fn resolve_snarl<'a>(root: &'a Snarl<EdNode>, nav: &[NodeId]) -> Option<&'a Snarl<EdNode>> {
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

/// Sibling path the editor saves the **hierarchical** snarl (with biomes) to, alongside the flat engine
/// `.graph.ron` the worldgen loads. e.g. `…/mountains_plains.graph.ron` → `…/mountains_plains.worldgraph.ron`.
fn worldgraph_path(graph_path: &str) -> String {
    let stem = graph_path.strip_suffix(".graph.ron").or_else(|| graph_path.strip_suffix(".ron"));
    match stem {
        Some(stem) => format!("{stem}.worldgraph.ron"),
        None => format!("{graph_path}.worldgraph.ron"),
    }
}

/// Load the editor graph for `graph_path`: prefer the hierarchical `.worldgraph.ron` (keeps biomes), then
/// the flat `.graph.ron`, then the built-in default. Used by the startup seed + the Load button so the
/// editor reflects what's actually on disk rather than a hard-coded graph.
fn load_editor_snarl(graph_path: &str) -> Snarl<EdNode> {
    let wg = worldgraph_path(graph_path);
    if let Ok(s) = std::fs::read_to_string(&wg)
        && let Ok(snarl) = ron::de::from_str::<Snarl<EdNode>>(&s)
    {
        return snarl;
    }
    if let Ok(s) = std::fs::read_to_string(graph_path)
        && let Ok(asset) = ron::de::from_str::<GraphAsset>(&s)
    {
        return graph_to_snarl(&asset.graph);
    }
    world_biome_snarl()
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
fn world_biome_snarl() -> Snarl<EdNode> {
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
    collapsed: &'a mut std::collections::HashSet<NodeId>,
    zoom_half_m: &'a mut std::collections::HashMap<NodeId, f64>,
    surface: &'a mut std::collections::HashSet<NodeId>,
    cam: &'a mut std::collections::HashMap<NodeId, (f32, f32)>,
    body_size: &'a mut std::collections::HashMap<NodeId, egui::Vec2>,
    disp_px: &'a mut std::collections::HashMap<NodeId, f32>,
    /// Set to a biome node id when the user clicks its "Open" — the panel descends after the show.
    enter: &'a mut Option<NodeId>,
    /// Set to a node id when the user clicks its pop-out button — the panel opens a window after the show.
    pop_request: &'a mut Option<NodeId>,
    /// Set to a node id when the user clicks "→ panel" — retargets the dockable preview panel.
    to_panel: &'a mut Option<NodeId>,
    /// Last frame's GPU preview textures (key → egui id) read by 3D inline previews.
    gpu_tex: &'a std::collections::HashMap<u64, egui::TextureId>,
    /// This frame's GPU preview requests, pushed by 3D inline previews; drained by the panel.
    gpu_reqs: &'a mut Vec<GpuPreviewRequest>,
    /// Per-node pan (world-XZ centre offset).
    pan: &'a mut std::collections::HashMap<NodeId, (f64, f64)>,
    /// Set to the node whose preview image the pointer is over (for next-frame scroll interception).
    hovered_preview: &'a mut Option<NodeId>,
    /// Hash of the current nav path — combined with the node id into a stable GPU pool key per preview.
    level_salt: u64,
}

impl SnarlViewer<EdNode> for Viewer<'_> {
    fn title(&mut self, node: &EdNode) -> String {
        match node {
            EdNode::Output => "Output".into(),
            EdNode::Op(k) => node_kind_name(k).into(),
            EdNode::Biome { name, .. } => format!("{} {name}", icon::PLANT),
            EdNode::Input(k) => format!("{} {}", icon::ARROW_ELBOW_DOWN_RIGHT, climate_name(*k)),
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
        // Node params / biome header, stacked vertically at the top of the body.
        match &mut snarl[node] {
            EdNode::Op(kind) => node_params_ui(ui, kind),
            EdNode::Biome { name, .. } => {
                ui.add(egui::TextEdit::singleline(name).desired_width(120.0).hint_text("biome name"));
            }
            _ => {}
        }
        if matches!(snarl.get_node(node), Some(EdNode::Biome { .. }))
            && ui.button(format!("{} Open", icon::CARET_RIGHT)).on_hover_text("Edit this biome's sub-graph").clicked()
        {
            *self.enter = Some(node);
        }
        // Divider between the node params (above) and the preview section (below).
        ui.separator();

        // Collapsed: just an expand toggle.
        if self.collapsed.contains(&node) {
            if ui
                .small_button(format!("{} Preview", icon::CARET_RIGHT))
                .on_hover_text("Show this node's 2D/3D preview")
                .clicked()
            {
                self.collapsed.remove(&node);
            }
            self.body_size.insert(node, ui.min_rect().size());
            return;
        }

        // Open: the preview IMAGE on the LEFT, its controls in a column on the RIGHT (no overlap).
        let is3d = self.surface.contains(&node);
        let size = self.disp_px.get(&node).copied().unwrap_or(DEFAULT_PREVIEW_PX);
        // Render at the displayed size in physical pixels (no cap) so the preview is always crisp.
        let ppp = ui.ctx().pixels_per_point();
        let res = ((size * ppp).round() as usize).max(32);
        let half = *self.zoom_half_m.get(&node).unwrap_or(&PREVIEW_HALF_M);
        let (yaw, pitch) = *self.cam.get(&node).unwrap_or(&CAM_DEFAULT);
        let (cx, cz) = *self.pan.get(&node).unwrap_or(&(0.0, 0.0));

        match graph_rooted_at(snarl, node) {
            Ok(g) => {
                // Both 2D and 3D render on the GPU pool (one shader, one `height_colour` SSOT). Push a
                // request and draw last frame's pool texture.
                let gkey = gpu_inline_key(self.level_salt, node);
                self.gpu_reqs.push(GpuPreviewRequest {
                    key: gkey,
                    graph: g,
                    half,
                    center: (cx, cz),
                    is3d,
                    yaw,
                    pitch,
                    res_w: res as u32,
                    res_h: res as u32,
                });
                let tex = self.gpu_tex.get(&gkey).copied();
                ui.horizontal_top(|ui| {
                    // LEFT — the preview image (a flat placeholder for the ~1 frame before the GPU texture
                    // warms up) with on-image gestures: scroll = zoom, drag = orbit (3D) / pan (2D),
                    // right-drag = pan (3D). The scroll is consumed so the graph doesn't also zoom.
                    let img_resp = preview_image(ui, tex, egui::vec2(size, size));
                    {
                        let h = self.zoom_half_m.entry(node).or_insert(PREVIEW_HALF_M);
                        let cam = self.cam.entry(node).or_insert(CAM_DEFAULT);
                        let pan = self.pan.entry(node).or_insert((0.0, 0.0));
                        handle_preview_gestures(ui, &img_resp, is3d, size, h, &mut pan.0, &mut pan.1, &mut cam.0, &mut cam.1);
                    }
                    // Record hover so the panel can intercept this preview's scroll-zoom next frame
                    // (before egui-snarl applies its own graph zoom).
                    if img_resp.hovered() {
                        *self.hovered_preview = Some(node);
                    }
                    // RIGHT — controls column (collapse, pop-out, zoom, 2D/3D, size).
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            if ui.small_button(icon::CARET_DOWN).on_hover_text("Collapse preview").clicked() {
                                self.collapsed.insert(node);
                            }
                            if ui.small_button(icon::ARROWS_OUT).on_hover_text("Pop out into a movable window").clicked() {
                                *self.pop_request = Some(node);
                            }
                            if ui.small_button(icon::PICTURE_IN_PICTURE).on_hover_text("Show in the dockable preview panel (by the viewport)").clicked() {
                                *self.to_panel = Some(node);
                            }
                        });
                        let h = self.zoom_half_m.entry(node).or_insert(PREVIEW_HALF_M);
                        let mut km = *h * 2.0 / 1000.0;
                        if ui
                            .add(egui::DragValue::new(&mut km).speed(0.25).range(0.05..=512.0).suffix(" km"))
                            .on_hover_text("Zoom: width of the sampled world window")
                            .changed()
                        {
                            *h = (km * 1000.0 / 2.0).max(1.0);
                        }
                        if ui
                            .selectable_label(is3d, "3D")
                            .on_hover_text("3D SDF-raymarched surface (drag the image to orbit)")
                            .clicked()
                        {
                            if is3d {
                                self.surface.remove(&node);
                            } else {
                                self.surface.insert(node);
                            }
                        }
                        let sz = self.disp_px.entry(node).or_insert(DEFAULT_PREVIEW_PX);
                        ui.add(egui::DragValue::new(sz).speed(2.0).range(64.0..=1024.0).suffix(" px"))
                            .on_hover_text("Preview size");
                    });
                });
            }
            Err(e) => {
                ui.colored_label(egui::Color32::from_rgb(200, 150, 120), format!("connect inputs ({e})"));
            }
        }
        self.body_size.insert(node, ui.min_rect().size());
    }

    fn show_input(&mut self, pin: &InPin, ui: &mut egui::Ui, snarl: &mut Snarl<EdNode>) -> impl SnarlPin + 'static {
        ui.label(input_label(&snarl[pin.id.node], pin.id.input));
        PinInfo::circle().with_fill(egui::Color32::from_rgb(120, 160, 220))
    }

    fn show_output(&mut self, _pin: &OutPin, _ui: &mut egui::Ui, _snarl: &mut Snarl<EdNode>) -> impl SnarlPin + 'static {
        // Single output pin — self-evident, and an "out" label here overlaps the pin (bad right margin).
        // Params live in the body (stacked vertically) to keep nodes narrow.
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
        if ui.button(format!("{} Biome", icon::PLANT)).on_hover_text("A nested biome sub-graph (climate in, height out)").clicked() {
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
    const GAP_X: f32 = 80.0;
    const GAP_Y: f32 = 44.0;
    const HEADER: f32 = 40.0; // title bar
    const PIN_ROW: f32 = 26.0; // per input/output pin row
    const FRAME: f32 = 34.0; // node frame margins (top+bottom / left+right)
    const OUT_PIN: f32 = 18.0; // output-pin column on the right

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
        let node = snarl.get_node(id);
        let arity = node
            .map(|n| match n {
                EdNode::Output => 1,
                EdNode::Op(k) => k.arity().max(1),
                EdNode::Biome { .. } => CLIMATE_INPUTS.len(),
                EdNode::Input(_) => 1,
            })
            .unwrap_or(1);
        // Op + Biome nodes carry a (default-on) preview, so they're tall; Input/Output are tiny. Use a
        // realistic per-kind default for any node not yet measured this session (e.g. off-screen at load),
        // so a single Auto-arrange already clears them instead of collapsing to a tiny default.
        let has_preview = matches!(node, Some(EdNode::Op(_)) | Some(EdNode::Biome { .. }));
        let default_body =
            if has_preview { egui::vec2(210.0, DEFAULT_PREVIEW_PX + 96.0) } else { egui::vec2(70.0, 6.0) };
        let body = body_size.get(&id).copied().unwrap_or(default_body);
        // Input pins flank the body on the LEFT (header on top, output pin on the right), so the node's
        // height is `header + max(body, pin-rows)` — NOT the sum — and its width includes the input-label
        // column (biome climate labels are long) + the output pin. Summing body+pins over-estimated height,
        // spreading nodes (and shrinking the fit-to-view zoom).
        let max_label = node.map(|n| (0..arity).map(|s| input_label(n, s).len()).max().unwrap_or(0)).unwrap_or(0);
        let in_col = if max_label > 0 { 20.0 + max_label as f32 * 7.0 } else { 14.0 };
        let w = in_col + body.x.max(96.0) + OUT_PIN + FRAME;
        let h = HEADER + body.y.max(arity as f32 * PIN_ROW) + FRAME;
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

/// Stable GPU pool key for an inline preview = nav-level salt ⊕ node id, with the top bit set so it can
/// never collide with the small pop-out window ids.
fn gpu_inline_key(level_salt: u64, node: NodeId) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    level_salt.hash(&mut h);
    node.hash(&mut h);
    h.finish() | (1u64 << 63)
}

/// Hash of a nav path (the per-level salt for inline preview keys).
fn nav_hash(nav: &[NodeId]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    nav.hash(&mut h);
    h.finish()
}

/// Apply (and CONSUME) scroll-zoom over a hovered preview image: zooms `half`, zeroes the ctx scroll so
/// the surrounding window/panel doesn't also scroll. (Inline snarl previews intercept scroll BEFORE the
/// snarl reads it — see `graph_panel` — because egui-snarl applies its own zoom before drawing nodes.)
fn scroll_zoom_consume(ui: &egui::Ui, resp: &egui::Response, half: &mut f64) {
    if !resp.hovered() {
        return;
    }
    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
    if scroll != 0.0 {
        ui.ctx().input_mut(|i| {
            i.smooth_scroll_delta = egui::Vec2::ZERO;
            i.raw_scroll_delta = egui::Vec2::ZERO;
        });
        *half = (*half * (1.0 - scroll as f64 * 0.0015)).clamp(20.0, 1_000_000.0);
    }
}

/// On-image drag gestures: left-drag = orbit (3D) / pan (2D), right-drag = pan (3D). `size` is the
/// on-screen image side (px). (Scroll-zoom is handled separately — see [`scroll_zoom_consume`].)
#[allow(clippy::too_many_arguments)]
fn handle_preview_gestures(
    ui: &egui::Ui,
    resp: &egui::Response,
    is3d: bool,
    size: f32,
    half: &mut f64,
    cx: &mut f64,
    cz: &mut f64,
    yaw: &mut f32,
    pitch: &mut f32,
) {
    let _ = ui;
    let wpp = (2.0 * *half) / size.max(1.0) as f64; // world units per display pixel
    if is3d {
        if resp.dragged_by(egui::PointerButton::Primary) {
            let d = resp.drag_delta();
            *yaw += d.x * 0.01;
            *pitch = (*pitch - d.y * 0.01).clamp(0.05, 1.5);
        }
        if resp.dragged_by(egui::PointerButton::Secondary) {
            let d = resp.drag_delta();
            *cx -= d.x as f64 * wpp;
            *cz -= d.y as f64 * wpp;
        }
    } else if resp.dragged_by(egui::PointerButton::Primary) {
        let d = resp.drag_delta();
        *cx -= d.x as f64 * wpp;
        *cz -= d.y as f64 * wpp;
    }
}

// ===================================================================================================
// Per-node 2D preview
// ===================================================================================================

/// Default on-screen size (points) of a node preview; adjustable per node via the size control.
const DEFAULT_PREVIEW_PX: f32 = 120.0;
/// Default half-extent (metres) of the world window a preview samples, centred on the origin.
const PREVIEW_HALF_M: f64 = 2048.0;

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

/// Draw a preview image at `size`, or a flat "baking…" placeholder for the ~1 frame before the GPU pool
/// texture is ready. Returns the (click-and-drag-sensing) response so on-image gestures work either way.
fn preview_image(ui: &mut egui::Ui, tex: Option<egui::TextureId>, size: egui::Vec2) -> egui::Response {
    match tex {
        Some(t) => ui.add(egui::Image::new(egui::load::SizedTexture::new(t, size)).sense(egui::Sense::click_and_drag())),
        None => {
            let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
            ui.painter().rect_filled(rect, 4.0, egui::Color32::from_gray(20));
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "baking…",
                egui::FontId::proportional(12.0),
                egui::Color32::from_gray(90),
            );
            resp
        }
    }
}

// ===================================================================================================
// Panel
// ===================================================================================================

fn graph_panel(world: &mut World, ui: &mut egui::Ui) {
    // Seed the editor once by LOADING the graph from disk (the saved .worldgraph.ron / .graph.ron, falling
    // back to the built-in default), and drive the live terrain from it.
    world.resource_scope::<WorldGraphEditor, ()>(|world, mut editor| {
        if !editor.seeded {
            editor.snarl = load_editor_snarl(&editor.path);
            editor.seeded = true;
            if let Ok(g) = snarl_to_graph(&editor.snarl) {
                world.resource_mut::<WorldGraph>().0 = Arc::new(g);
            }
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
            // SAVE — write BOTH the compiled flat engine graph (.graph.ron, the world hot-reloads it) AND
            // the hierarchical editor snarl with biomes (.worldgraph.ron, so the hierarchy survives reload).
            if ui.button("Save").on_hover_text("Write the flat .graph.ron (world reloads it) + the .worldgraph.ron hierarchy").clicked() {
                editor.status = match snarl_to_graph(&editor.snarl) {
                    Ok(g) => {
                        let flat = (GraphAsset { graph: g }).save(std::path::Path::new(&editor.path));
                        let wg = worldgraph_path(&editor.path);
                        let hier = ron::ser::to_string_pretty(&editor.snarl, ron::ser::PrettyConfig::default())
                            .map_err(|e| e.to_string())
                            .and_then(|s| std::fs::write(&wg, s).map_err(|e| e.to_string()));
                        match (flat, hier) {
                            (Ok(()), Ok(())) => format!("saved {} (+hierarchy)", editor.path),
                            (Err(e), _) => format!("save failed: {e}"),
                            (_, Err(e)) => format!("flat saved; hierarchy failed: {e}"),
                        }
                    }
                    Err(e) => format!("invalid: {e}"),
                };
            }
            // LOAD — prefer the hierarchical .worldgraph.ron (restores biomes); else the flat .graph.ron.
            if ui.button("Load").clicked() {
                let wg = worldgraph_path(&editor.path);
                editor.status = match std::fs::read_to_string(&wg) {
                    Ok(s) => match ron::de::from_str::<Snarl<EdNode>>(&s) {
                        Ok(snarl) => {
                            editor.snarl = snarl;
                            editor.nav.clear();
                            editor.clear_node_caches();
                            editor.needs_arrange = true;
                            format!("loaded {wg}")
                        }
                        Err(e) => format!("hierarchy parse failed: {e}"),
                    },
                    Err(_) => match std::fs::read_to_string(&editor.path) {
                        Ok(s) => match ron::de::from_str::<GraphAsset>(&s) {
                            Ok(asset) => {
                                editor.snarl = graph_to_snarl(&asset.graph);
                                editor.nav.clear();
                                editor.clear_node_caches();
                                editor.needs_arrange = true;
                                format!("loaded {} (flat)", editor.path)
                            }
                            Err(e) => format!("parse failed: {e}"),
                        },
                        Err(e) => format!("read failed: {e}"),
                    },
                };
            }
            if ui.button("Reset").on_hover_text("Restore the default multi-biome world graph").clicked() {
                editor.snarl = world_biome_snarl();
                editor.nav.clear();
                editor.clear_node_caches();
                editor.needs_arrange = true;
                editor.status = "reset to biome world".into();
            }
            if ui.button("Auto-arrange").on_hover_text("Lay nodes out left→right by dependency depth").clicked() {
                // Arrange the CURRENTLY shown level (inside a biome, not the top graph).
                let WorldGraphEditor { snarl, nav, body_size, .. } = &mut *editor;
                let vd = valid_depth(snarl, nav);
                nav.truncate(vd);
                auto_arrange(current_snarl_mut(snarl, nav), body_size);
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
            if ui.selectable_label(editor.nav.is_empty(), format!("{} World", icon::GLOBE)).clicked() {
                nav_to = Some(0);
            }
            for (i, name) in crumbs.iter().enumerate() {
                ui.label(icon::CARET_RIGHT);
                if ui.selectable_label(i + 1 == editor.nav.len(), format!("{} {name}", icon::PLANT)).clicked() {
                    nav_to = Some(i + 1);
                }
            }
        });
        if let Some(d) = nav_to.filter(|&d| d != editor.nav.len()) {
            editor.nav.truncate(d);
            editor.clear_node_caches();
        }
        ui.separator();

        // Intercept scroll-zoom for the inline preview hovered last frame — egui-snarl applies its own
        // graph zoom BEFORE drawing nodes, so consume the scroll here (before the show) and route it to
        // the preview instead.
        if let Some(node) = editor.hovered_preview.take() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                ui.ctx().input_mut(|i| {
                    i.smooth_scroll_delta = egui::Vec2::ZERO;
                    i.raw_scroll_delta = egui::Vec2::ZERO;
                });
                let h = editor.zoom_half_m.entry(node).or_insert(PREVIEW_HALF_M);
                *h = (*h * (1.0 - scroll as f64 * 0.0015)).clamp(20.0, 1_000_000.0);
            }
        }

        // GPU preview plumbing: read last frame's textures + gather this frame's requests (shared by the
        // inline 3D previews below + the pop-out windows).
        let gpu_tex = world.get_resource::<GpuPreviewTextures>().map(|t| t.0.clone()).unwrap_or_default();
        let mut gpu_reqs: Vec<GpuPreviewRequest> = Vec::new();
        let level_salt = nav_hash(&editor.nav);

        // Show the snarl at the current nav depth. Disjoint borrows: `snarl`+`nav` resolve the level;
        // the rest are the per-node preview caches the Viewer drives.
        editor.enter = None;
        editor.pop_request = None;
        editor.to_panel = None;
        {
            let WorldGraphEditor {
                snarl,
                nav,
                collapsed,
                zoom_half_m,
                surface,
                cam,
                body_size,
                disp_px,
                pan,
                hovered_preview,
                enter,
                pop_request,
                to_panel,
                ..
            } = &mut *editor;
            let current = current_snarl_mut(snarl, nav);
            let mut viewer = Viewer {
                collapsed,
                zoom_half_m,
                surface,
                cam,
                body_size,
                disp_px,
                enter,
                pop_request,
                to_panel,
                gpu_tex: &gpu_tex,
                gpu_reqs: &mut gpu_reqs,
                pan,
                hovered_preview,
                level_salt,
            };
            // Keep nodes readable on load: egui-snarl's initial view auto-fits the graph clamped to
            // [min_scale, max_scale], so the floor doubles as the default zoom — 0.75 keeps a freshly-loaded
            // graph legible (the compact auto-arrange usually fits above this). Allow zooming in to 3×.
            let style = SnarlStyle { min_scale: Some(0.75), max_scale: Some(3.0), ..SnarlStyle::new() };
            SnarlWidget::new()
                .id(egui::Id::new("worldgen-biome-graph"))
                .style(style)
                .show(current, &mut viewer, ui);
        }
        // After a seed/load, auto-arrange once the nodes have been measured this frame (so the layout uses
        // real sizes). Applies on the next frame.
        if std::mem::take(&mut editor.needs_arrange) {
            editor.rearrange();
        }
        // Descend into a biome the user opened this frame.
        if let Some(id) = editor.enter.take() {
            editor.nav.push(id);
            editor.clear_node_caches();
        }
        // Retarget the dockable preview panel (snapshotting the node's nav + view state).
        if let Some(node) = editor.to_panel.take() {
            let nav = editor.nav.clone();
            let half = editor.zoom_half_m.get(&node).copied().unwrap_or(PREVIEW_HALF_M);
            let cam = editor.cam.get(&node).copied().unwrap_or(CAM_DEFAULT);
            let pan = editor.pan.get(&node).copied().unwrap_or((0.0, 0.0));
            let is3d = editor.surface.contains(&node);
            if let Some(mut panel) = world.get_resource_mut::<WorldgenPreviewPanel>() {
                panel.target = Some((nav, node));
                panel.half = half;
                panel.cam = cam;
                panel.pan = pan;
                panel.is3d = is3d;
                // Ensure/focus the dock tab — but only OUTSIDE the dock render (the dock state isn't in
                // the World here). `open_preview_panel` handles it next frame.
                panel.pending_open = true;
            }
        }
        // Pop a node's preview out into a movable window (snapshotting its current view state + nav path).
        if let Some(node) = editor.pop_request.take() {
            let half = editor.zoom_half_m.get(&node).copied().unwrap_or(PREVIEW_HALF_M);
            let is3d = editor.surface.contains(&node);
            let cam = editor.cam.get(&node).copied().unwrap_or(CAM_DEFAULT);
            let size = editor.disp_px.get(&node).copied().unwrap_or(DEFAULT_PREVIEW_PX).max(260.0);
            let nav = editor.nav.clone();
            let id = editor.next_pop_id;
            editor.next_pop_id += 1;
            editor.popped.push(PoppedPreview {
                id,
                nav,
                node,
                half,
                cx: 0.0,
                cz: 0.0,
                size,
                is3d,
                cam,
                open: true,
            });
        }
        // Render the popped-out preview windows (float above everything; drag anywhere incl. top panel).
        // 3D pop-outs render on the GPU via the same shared request/texture buffers as the inline previews.
        {
            let WorldGraphEditor { snarl, popped, .. } = &mut *editor;
            for p in popped.iter_mut() {
                popped_preview_window(ui, p, snarl, &gpu_tex, &mut gpu_reqs);
            }
            popped.retain(|p| p.open);
        }
        if !gpu_reqs.is_empty()
            && let Some(mut reqs) = world.get_resource_mut::<GpuPreviewRequests>()
        {
            reqs.0.append(&mut gpu_reqs);
        }
    });
}

/// The dockable, viewport-located **Node Preview** panel — shows whichever node was sent via "→ panel",
/// large, with its own 2D/3D + zoom/pan/orbit (both rendered on the shared GPU pool).
fn preview_panel(world: &mut World, ui: &mut egui::Ui) {
    let Some((nav, node)) = world.resource::<WorldgenPreviewPanel>().target.clone() else {
        ui.label("No preview targeted. In the Biome Graph, click a node preview's ▢ button to show it here.");
        return;
    };
    // Compile the targeted node's sub-graph from the editor snarl.
    let g = world.resource_scope::<WorldGraphEditor, Option<Graph>>(|_w, ed| {
        resolve_snarl(&ed.snarl, &nav).and_then(|s| graph_rooted_at(s, node).ok())
    });
    let Some(g) = g else {
        ui.label("the targeted node no longer exists");
        return;
    };

    world.resource_scope::<WorldgenPreviewPanel, ()>(|world, mut panel| {
        let panel = &mut *panel; // reborrow once so disjoint field borrows don't alias through Mut's deref
        ui.horizontal(|ui| {
            if ui.selectable_label(panel.is3d, "3D").on_hover_text("GPU 3D surface").clicked() {
                panel.is3d = !panel.is3d;
            }
            let mut km = panel.half * 2.0 / 1000.0;
            if ui.add(egui::DragValue::new(&mut km).speed(0.5).range(0.05..=512.0).suffix(" km")).changed() {
                panel.half = (km * 1000.0 / 2.0).max(1.0);
            }
            ui.add(egui::DragValue::new(&mut panel.pan.0).speed(10.0).prefix("X ").suffix(" m"));
            ui.add(egui::DragValue::new(&mut panel.pan.1).speed(10.0).prefix("Y ").suffix(" m"));
            ui.label("· drag orbit · right-drag pan · scroll zoom");
        });
        let ppp = ui.ctx().pixels_per_point();
        // Fill the panel non-square (drag the dock edge to resize); render res tracks the on-screen size.
        // Square preview sized to fit the panel (drag the dock edge to resize), centred in the leftover space.
        let avail = ui.available_size();
        let side = avail.x.min(avail.y).max(64.0);
        let res = ((side * ppp).round() as usize).max(32);
        world.resource_mut::<GpuPreviewRequests>().0.push(GpuPreviewRequest {
            key: PANEL_GPU_KEY,
            graph: g,
            half: panel.half,
            center: panel.pan,
            is3d: panel.is3d,
            yaw: panel.cam.0,
            pitch: panel.cam.1,
            res_w: res as u32,
            res_h: res as u32,
        });
        let tex = world.resource::<GpuPreviewTextures>().0.get(&PANEL_GPU_KEY).copied();
        ui.vertical_centered(|ui| {
            let resp = preview_image(ui, tex, egui::vec2(side, side));
            scroll_zoom_consume(ui, &resp, &mut panel.half);
            let WorldgenPreviewPanel { half, pan, cam, is3d, .. } = &mut *panel;
            handle_preview_gestures(ui, &resp, *is3d, side, half, &mut pan.0, &mut pan.1, &mut cam.0, &mut cam.1);
        });
    });
}

/// Outside the dock render (when `EditorDockState` is back in the World), ensure + focus the dockable
/// Node Preview tab if "→ panel" was requested this/last frame.
fn open_preview_panel(world: &mut World) {
    if !world.resource::<WorldgenPreviewPanel>().pending_open {
        return;
    }
    world.resource_mut::<WorldgenPreviewPanel>().pending_open = false;
    if !world.contains_resource::<super::dock::EditorDockState>() {
        return;
    }
    let tab = super::dock::EditorTab::Registered("worldgen/node-preview".into());
    super::layout::set_panel_present(world, tab.clone(), super::panels::DockSide::Center, true);
    if let Some(mut dock) = world.get_resource_mut::<super::dock::EditorDockState>()
        && let Some((n, t)) = dock.state.find_main_surface_tab(&tab)
    {
        dock.state.set_active_tab((egui_dock::SurfaceIndex::main(), n, t));
    }
}

/// Draw one popped-out preview as a floating, resizable `egui::Window`. Both 2D and 3D render on the
/// shared GPU pool (push a request, draw last frame's texture). `gpu_tex` is last frame's pool output.
fn popped_preview_window(
    ui: &egui::Ui,
    p: &mut PoppedPreview,
    root: &Snarl<EdNode>,
    gpu_tex: &std::collections::HashMap<u64, egui::TextureId>,
    gpu_reqs: &mut Vec<GpuPreviewRequest>,
) {
    let mut open = p.open;
    egui::Window::new(format!("Preview {}", p.id))
        .id(egui::Id::new(("wg-pop", p.id)))
        .open(&mut open)
        .resizable(true)
        .default_size([p.size + 80.0, p.size + 60.0])
        .show(ui.ctx(), |ui| {
            let g = match resolve_snarl(root, &p.nav).map(|s| graph_rooted_at(s, p.node)) {
                Some(Ok(g)) => g,
                _ => {
                    ui.colored_label(egui::Color32::from_rgb(200, 150, 120), "node no longer exists");
                    return;
                }
            };
            ui.horizontal(|ui| {
                if ui.selectable_label(p.is3d, "3D").on_hover_text("GPU 3D surface (drag to orbit)").clicked() {
                    p.is3d = !p.is3d;
                }
                let mut km = p.half * 2.0 / 1000.0;
                if ui.add(egui::DragValue::new(&mut km).speed(0.25).range(0.05..=512.0).suffix(" km")).changed() {
                    p.half = (km * 1000.0 / 2.0).max(1.0);
                }
            });
            if p.is3d {
                ui.horizontal(|ui| {
                    ui.label("offset");
                    ui.add(egui::DragValue::new(&mut p.cx).speed(10.0).prefix("X ").suffix(" m"));
                    ui.add(egui::DragValue::new(&mut p.cz).speed(10.0).prefix("Y ").suffix(" m"));
                    if ui.button("center").clicked() {
                        p.cx = 0.0;
                        p.cz = 0.0;
                    }
                });
            }

            let ppp = ui.ctx().pixels_per_point();
            // Square preview sized to fit the window (drag its edge to resize), centred in the leftover space.
            let avail = ui.available_size();
            let side = avail.x.min(avail.y).max(64.0);
            let res = ((side * ppp).round() as usize).max(32);
            // GPU path (2D + 3D): request a render for next frame; draw last frame's pool texture.
            gpu_reqs.push(GpuPreviewRequest {
                key: p.id,
                graph: g,
                half: p.half,
                center: (p.cx, p.cz),
                is3d: p.is3d,
                yaw: p.cam.0,
                pitch: p.cam.1,
                res_w: res as u32,
                res_h: res as u32,
            });
            let tex = gpu_tex.get(&p.id).copied();
            ui.vertical_centered(|ui| {
                let resp = preview_image(ui, tex, egui::vec2(side, side));
                scroll_zoom_consume(ui, &resp, &mut p.half);
                handle_preview_gestures(ui, &resp, p.is3d, side, &mut p.half, &mut p.cx, &mut p.cz, &mut p.cam.0, &mut p.cam.1);
            });
        });
    p.open = open;
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

    /// A hierarchical snarl (with a biome) must survive a RON round-trip and compile identically — this
    /// is what persists the biome hierarchy across save/reload (`.worldgraph.ron`).
    #[test]
    fn biome_hierarchy_ron_round_trips() {
        let mut sub = Snarl::new();
        let inp = sub.insert_node(p(), EdNode::Input(0));
        let c = sub.insert_node(p(), EdNode::Op(NodeKind::Const(3.0)));
        let add = sub.insert_node(p(), EdNode::Op(NodeKind::Add));
        sub.connect(out(inp), inn(add, 0));
        sub.connect(out(c), inn(add, 1));
        let so = sub.insert_node(p(), EdNode::Output);
        sub.connect(out(add), inn(so, 0));
        let mut top = Snarl::new();
        let c10 = top.insert_node(p(), EdNode::Op(NodeKind::Const(10.0)));
        let b = top.insert_node(p(), EdNode::Biome { name: "Hills".into(), graph: Box::new(sub) });
        top.connect(out(c10), inn(b, 0));
        let o = top.insert_node(p(), EdNode::Output);
        top.connect(out(b), inn(o, 0));

        let s = ron::ser::to_string(&top).expect("serialize");
        let back: Snarl<EdNode> = ron::de::from_str(&s).expect("deserialize");
        let (g1, g2) = (snarl_to_graph(&top).unwrap(), snarl_to_graph(&back).unwrap());
        for &(x, z) in &[(0.0, 0.0), (123.0, -456.0)] {
            assert_eq!(g1.eval(x, z, 7).v.to_bits(), g2.eval(x, z, 7).v.to_bits());
        }
        assert_eq!(g2.eval(0.0, 0.0, 7).v, 13.0); // Input(0)=10 + Const 3
    }

    /// The default multi-biome world graph compiles, evaluates finite, and shows BOTH gentle (plains) and
    /// tall (mountains) terrain over a region — the climate classifier actually places both biomes.
    #[test]
    fn biome_world_has_plains_and_mountains() {
        let g = snarl_to_graph(&world_biome_snarl()).expect("compile biome world");
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..40 {
            for j in 0..40 {
                let x = i as f64 * 450.0 - 9000.0;
                let z = j as f64 * 450.0 - 9000.0;
                let v = g.eval(x, z, 7).v;
                assert!(v.is_finite(), "non-finite at ({x},{z})");
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        assert!(hi - lo > 300.0, "expected plains+mountains spread, got {lo}..{hi}");
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

    /// Regenerate the shipped default world assets from `world_biome_snarl` (run on purpose):
    /// `cargo test --features editor write_world_biome_assets -- --ignored`.
    #[test]
    #[ignore]
    fn write_world_biome_assets() {
        let snarl = world_biome_snarl();
        let g = snarl_to_graph(&snarl).expect("compile");
        (GraphAsset { graph: g })
            .save(std::path::Path::new("assets/worldgen/world.graph.ron"))
            .expect("save flat");
        let s = ron::ser::to_string_pretty(&snarl, ron::ser::PrettyConfig::default()).expect("ser hierarchy");
        std::fs::write("assets/worldgen/world.worldgraph.ron", s).expect("write hierarchy");
    }

    /// The shipped `world.graph.ron` must match the compiled `world_biome_snarl` (catches drift after the
    /// default graph changes — re-run `write_world_biome_assets`).
    #[test]
    fn shipped_world_graph_matches_snarl() {
        let s = std::fs::read_to_string("assets/worldgen/world.graph.ron").expect("read shipped world graph");
        let shipped: GraphAsset = ron::de::from_str(&s).expect("parse shipped");
        let built = snarl_to_graph(&world_biome_snarl()).unwrap();
        for &(x, z) in &[(0.0, 0.0), (1234.0, -987.0), (5000.0, 5000.0)] {
            assert_eq!(
                shipped.graph.eval(x, z, 7).v.to_bits(),
                built.eval(x, z, 7).v.to_bits(),
                "shipped world.graph.ron is stale — re-run write_world_biome_assets"
            );
        }
    }
}
