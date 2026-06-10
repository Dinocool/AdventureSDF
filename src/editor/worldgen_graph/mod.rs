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
mod persist;
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
use preview::{
    CAM_DEFAULT, DEFAULT_PREVIEW_PX, PREVIEW_HALF_M, PoppedPreview, WorldgenPreviewPanels, open_preview_panel,
};

/// Render the dynamic Node Preview tab for instance `id` (the dock `EditorTab::WorldgenPreview` arm calls
/// this). Thin crate-visible wrapper over the module-private `preview::preview_panel_impl`.
pub(crate) fn preview_panel_for(world: &mut World, ui: &mut bevy_egui::egui::Ui, id: u64) {
    preview::preview_panel_impl(world, ui, id);
}

/// Drop the closed preview instance `id`'s state (the dock `on_close` arm calls this), so a closed tab
/// doesn't leak or reopen.
pub(crate) fn close_preview(world: &mut World, id: u64) {
    if let Some(mut panels) = world.get_resource_mut::<WorldgenPreviewPanels>() {
        panels.map.remove(&id);
    }
}

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

/// One clipboard entry: a copied node's kind + canvas position + the internal wires (among the copied
/// set) that feed it. `wires_in` is `(src_clip_index, src_out_pin, this_in_pin)` — by clipboard-LOCAL
/// index, so paste can re-wire the copied subgraph after inserting fresh nodes (mapping clip-index → new
/// `NodeId`). Transient (never persisted). See [`copy_selection`]/[`paste_clipboard`].
#[derive(Clone)]
pub(super) struct ClipNode {
    pub(super) kind: EdNode,
    pub(super) pos: egui::Pos2,
    pub(super) wires_in: Vec<(usize, usize, usize)>,
}

/// A node's persisted preview view-state — ONE struct per node (the single source of truth for every
/// per-node preview setting). This consolidation is the extensibility win: a new per-node preview
/// setting is one new field here (give it a `Default`), and it persists + resumes for free.
///
/// `#[serde(default)]` so an older `.worldgraph.ron` missing a field still loads (the field defaults).
/// All fields default to the "fresh node" semantics: preview OPEN (`collapsed = false`), 2D
/// (`surface = false`), default zoom/camera/pan/size.
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub(super) struct NodeView {
    /// Preview COLLAPSED (default `false` = open). Previews are on by default.
    pub collapsed: bool,
    /// Show the 3D SDF-raymarched surface (`true`) instead of the 2D heatmap (`false`, the default).
    pub surface: bool,
    /// Preview zoom: half-extent (metres) of the sampled world window. Shared by the 2D heatmap (grid
    /// extent) and the 3D surface (camera framing).
    pub zoom_half_m: f64,
    /// 3D-preview orbit camera (yaw, pitch) in radians.
    pub cam: (f32, f32),
    /// Pan: world-XZ centre offset of the sampled window (drag-pan / scroll over the preview).
    pub pan: (f64, f64),
    /// On-screen preview square side (points), used to pick the render resolution so previews stay
    /// crisp as the node is resized.
    pub disp_px: f32,
}

impl Default for NodeView {
    fn default() -> Self {
        Self {
            collapsed: false,
            surface: false,
            zoom_half_m: PREVIEW_HALF_M,
            cam: CAM_DEFAULT,
            pan: (0.0, 0.0),
            disp_px: DEFAULT_PREVIEW_PX,
        }
    }
}

/// Per-node UI caches the Viewer drives, keyed by `NodeId`. `NodeId`s are per-snarl-level (a fresh id
/// namespace each level), so the whole set is cleared on navigation — see [`WorldGraphEditor::
/// clear_node_caches`] (one assignment, can't miss a map).
#[derive(Default)]
struct NodeCaches {
    /// Persisted per-node preview settings (collapsed/surface/zoom/cam/pan/size). Absence ⇒ defaults
    /// (see [`NodeView::default`]). This is what `gather_view`/`apply_view` snapshot + restore.
    views: std::collections::HashMap<NodeId, NodeView>,
    /// Last-frame body content size per node (egui can't expose the node rect), used by `auto_arrange`
    /// to pack columns/rows by real size instead of a fixed grid. TRANSIENT — never persisted.
    body_size: std::collections::HashMap<NodeId, egui::Vec2>,
}

/// One-shot signals the Viewer raises during a graph show, drained by `graph_panel` right after (each is
/// a "the user clicked X this frame" request). Reset to default before every show.
#[derive(Default)]
struct ViewerSignals {
    /// Set to a biome node id when the user clicks its "Open" — the panel descends after the show.
    enter: Option<NodeId>,
    /// Set to a node id when the user clicks its pop-out button — the panel opens a window after the show.
    pop_request: Option<NodeId>,
    /// Set to a node id when the user clicks "→ panel" — the panel retargets the dockable preview panel.
    to_panel: Option<NodeId>,
    /// Raised when a node's `collapsed` flag is toggled this frame — the panel ORs it into
    /// `needs_arrange` so the layout re-packs (collapsed nodes shrink, so the columns tighten up).
    needs_arrange: bool,
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
    /// Per-node UI caches (the persisted [`NodeView`] settings + the transient body_size), all cleared
    /// on navigation.
    caches: NodeCaches,
    /// Which inline preview image the pointer was over last frame — so `graph_panel` can intercept the
    /// scroll-zoom for it BEFORE egui-snarl applies its own (graph) zoom.
    hovered_preview: Option<NodeId>,
    /// Navigation stack of biome nodes we've descended into (empty ⇒ the top "World" graph). The shown
    /// snarl is `snarl` walked through each biome's sub-graph. (Distinct from `path`, the save file path.)
    nav: Vec<NodeId>,
    /// One-shot Viewer→panel signals (Open / pop-out / → panel), drained each frame after the show.
    signals: ViewerSignals,
    /// Previews "popped out" into floating windows (drag anywhere, incl. over the top panel). Each is
    /// self-contained so it survives navigation and doesn't clash with the in-graph preview caches.
    popped: Vec<PoppedPreview>,
    /// Monotonic id source for popped windows (their stable GPU pool key).
    next_pop_id: u64,
    /// Set after a graph is seeded/loaded; the panel auto-arranges once the nodes have been measured.
    needs_arrange: bool,
    /// Cut/copy clipboard for node select+copy+paste (transient; not part of the persist doc). Holds the
    /// last copied selection (kinds + positions + internal wires) so paste reproduces the subgraph.
    clipboard: Vec<ClipNode>,
}

impl Default for WorldGraphEditor {
    fn default() -> Self {
        Self {
            snarl: Snarl::new(),
            seeded: false,
            path: DEFAULT_GRAPH_PATH.to_string(),
            status: String::new(),
            caches: NodeCaches::default(),
            hovered_preview: None,
            nav: Vec::new(),
            signals: ViewerSignals::default(),
            popped: Vec::new(),
            next_pop_id: 1000,
            needs_arrange: true,
            clipboard: Vec::new(),
        }
    }
}

impl WorldGraphEditor {
    /// Drop all per-node UI caches — called on navigation, since `NodeId`s are per-snarl-level (a fresh
    /// id namespace each level) so caches must not bleed between levels.
    fn clear_node_caches(&mut self) {
        self.caches = NodeCaches::default();
    }

    /// Auto-arrange the CURRENTLY-NAVIGATED level's snarl (plain `&mut self` so the disjoint
    /// snarl/nav/body_size borrows don't alias through `Mut`'s deref). Truncates a stale nav tail first
    /// so `current_snarl_mut` only ever walks live biome nodes.
    fn rearrange(&mut self) {
        let WorldGraphEditor { snarl, nav, caches, .. } = self;
        let vd = convert::valid_depth(snarl, nav);
        nav.truncate(vd);
        auto_arrange(convert::current_snarl_mut(snarl, nav), &caches.body_size);
    }
}

/// Plugin: registers the editor state + the dockable "Biome Graph" panel.
pub struct WorldgenGraphEditorPlugin;

impl Plugin for WorldgenGraphEditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldGraphEditor>();
        app.init_resource::<WorldgenPreviewPanels>();
        // Deferred dock manipulation (the dock state is removed from the World during its own render).
        app.add_systems(Update, open_preview_panel);
        // Auto-persist the editor session on exit so reopening resumes it WITHOUT an explicit Save — same
        // `Last`-schedule pattern as the dock layout's auto-persist.
        app.add_systems(Last, save_worldgen_session_on_exit);
        super::panels::register_panel(
            app,
            "worldgen/graph",
            "Biome Graph",
            super::panels::DockSide::Right,
            30,
            graph_panel,
        );
        // The viewport-located Node Preview tabs are DYNAMIC center tabs (`EditorTab::WorldgenPreview`),
        // not registered panels: "→ panel" spawns a fresh one per click (unbounded), `open_preview_panel`
        // creates its dock tab, and closing the tab drops its state. So nothing is registered here.
    }
}

/// On app exit, snapshot the live editor session (graph + preview view-state) to the auto-persist file so
/// the next launch resumes it — see [`persist::save_session`]. Resources are `Option` so a headless run
/// without the editor resources is a no-op.
fn save_worldgen_session_on_exit(
    mut exit: MessageReader<AppExit>,
    editor: Option<Res<WorldGraphEditor>>,
    panels: Option<Res<WorldgenPreviewPanels>>,
) {
    if exit.read().next().is_none() {
        return;
    }
    if let (Some(editor), Some(panels)) = (editor, panels) {
        persist::save_session(&editor, &panels);
    }
}
