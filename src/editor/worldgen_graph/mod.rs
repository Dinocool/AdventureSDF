//! The **biome node-graph editor** — a visual `egui-snarl` panel for authoring the worldgen terrain
//! graph. Nodes are the engine [`NodeKind`] library (plus an [`EdNode::Output`] sink); editing rebuilds
//! the engine [`Graph`] and republishes it into the [`WorldGraph`] resource, which `roll_worldgen`
//! re-meshes live. Load/save go through the same RON asset pipeline as materials.
//!
//! `Snarl<EdNode>` is the editor's working graph; [`snarl_to_graph`]/[`graph_to_snarl`] convert to/from
//! the engine [`Graph`] (the bake samples the engine form). Gated behind `editor`.

use bevy::prelude::*;
use bevy_egui::egui;
use egui_snarl::{NodeId, Snarl};

use crate::sdf_render::worldgen::graph::node::NodeKind;

mod arrange;
mod compile;
mod convert;
mod node;
mod panel;
mod preview;
mod viewer;
#[cfg(test)]
mod tests;

use arrange::auto_arrange;
pub use compile::{graph_rooted_at, snarl_to_graph};
pub use convert::graph_to_snarl;
// Re-exported so child modules can reach them as `super::…` (viewer/preview/tests).
use convert::{climate_name, resolve_snarl};
use panel::graph_panel;
use preview::{PoppedPreview, WorldgenPreviewPanel, open_preview_panel, preview_panel};

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
