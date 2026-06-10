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
use egui_snarl::{InPin, NodeId, OutPin, Snarl};

use crate::assets::Asset as _;
use crate::sdf_render::worldgen::WorldGraph;
use crate::sdf_render::worldgen::graph::GraphAsset;
use crate::sdf_render::worldgen::graph::node::NodeKind;
use super::worldgen_gpu_preview::{GpuPreviewRequest, GpuPreviewRequests, GpuPreviewTextures};

mod arrange;
mod compile;
mod convert;
mod node;
mod preview;
#[cfg(test)]
mod tests;

use arrange::auto_arrange;
pub use compile::{graph_rooted_at, snarl_to_graph};
pub use convert::graph_to_snarl;
use convert::{
    breadcrumb_names, climate_name, current_snarl_mut, load_editor_snarl, new_biome_subgraph, resolve_snarl,
    valid_depth, world_biome_snarl, worldgraph_path,
};
use node::{input_label, node_catalog, node_kind_name, node_params_ui};
use preview::{
    CAM_DEFAULT, DEFAULT_PREVIEW_PX, PoppedPreview, PREVIEW_HALF_M, WorldgenPreviewPanel, apply_scroll_zoom,
    gpu_inline_key, handle_preview_gestures, nav_hash, open_preview_panel, popped_preview_window, preview_image,
    preview_panel,
};

/// Default on-disk path the editor saves/loads the active biome graph to (the production graph the
/// worldgen loads — see `WorldGenPlugin`'s asset hot-reload). Relative to the app's `assets/` root.
const DEFAULT_GRAPH_PATH: &str = "assets/worldgen/world.graph.ron";

/// The climate axes a biome can read from its parent (its input pins, in order). Expandable: add an
/// axis here and biomes gain a pin for it. The parent graph drives these (low-freq Fbm / derived math)
/// and they place + shape biomes.
pub const CLIMATE_INPUTS: [&str; 4] = ["continentalness", "temperature", "humidity", "weirdness"];

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

// `SnarlPin` is the trait the show_input/show_output return values implement (PinInfo does).
use egui_snarl::ui::SnarlPin;

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
                let h = editor.zoom_half_m.entry(node).or_insert(PREVIEW_HALF_M);
                apply_scroll_zoom(ui, scroll, h);
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
