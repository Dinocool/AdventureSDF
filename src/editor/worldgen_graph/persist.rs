//! **Versioned editor-state persistence** for the biome graph. The on-disk `.worldgraph.ron` is a
//! [`WorldGraphDoc`]: the working [`Snarl`] PLUS an [`EditorView`] capturing where the user was and how
//! their previews were configured, so opening the editor resumes exactly where they left off.
//!
//! Every field is `#[serde(default)]`, so adding a persisted thing later is a one-line plug AND older
//! files (missing the new field, or even a bare pre-doc `Snarl`) still load. The load chain
//! ([`load_editor_doc`]) degrades gracefully: doc → bare snarl → flat engine graph → built-in default.

use std::collections::HashMap;

use bevy::log::warn;
use egui_snarl::{NodeId, Snarl};

use crate::sdf_render::worldgen::graph::GraphAsset;

use super::convert::{graph_to_snarl, valid_depth, world_biome_snarl, worldgraph_path};
use super::preview::{PoppedPreview, PreviewView, WorldgenPreviewPanels};
use super::{EdNode, NodeView, WorldGraphEditor};

/// Current persistence-document schema version. Bump when the doc shape changes incompatibly; a
/// migration can then branch on the loaded `version` (older files default to `1`, see below).
fn doc_version() -> u32 {
    1
}

/// The full on-disk editor document: the working graph + the editor's view-state. Serialized as
/// `.worldgraph.ron` (the hierarchical save; the flat `.graph.ron` the world hot-reloads is written
/// separately).
#[derive(serde::Serialize, serde::Deserialize)]
pub(super) struct WorldGraphDoc {
    /// Schema version (defaults to `1` for files written before the field existed).
    #[serde(default = "doc_version")]
    pub version: u32,
    /// The editor's working graph (with biomes), the SSOT the rest compiles from.
    pub snarl: Snarl<EdNode>,
    /// The editor's resumable view-state (per-node settings, nav, panel target, pop-outs).
    #[serde(default)]
    pub view: EditorView,
}

/// The resumable editor view-state. Every field `#[serde(default)]` so each is an independent,
/// optional plug — a future addition (e.g. graph pan/zoom) is one new field and old files still load.
#[derive(Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub(super) struct EditorView {
    /// TOP-LEVEL per-node preview settings (the [`NodeView`]s of the root snarl). Keyed by `NodeId`,
    /// which round-trips stably through `Snarl`'s `Slab` serde, so settings re-attach to their node.
    pub nodes: HashMap<NodeId, NodeView>,
    /// Which biome the user was navigated into (restored on load, clamped to live biomes).
    pub nav: Vec<NodeId>,
    /// The dockable Node Preview panels' targets + views — one entry per OCCUPIED pool slot (empty slots
    /// aren't persisted). Restored into the first free slots on load.
    pub panels: Vec<PanelView>,
    /// DEPRECATED backward-compat alias: the single pre-pool panel field. Older `.worldgraph.ron` files
    /// wrote `panel: Some(..)`; on load `apply_view` folds it into `panels`. Never written anymore.
    #[serde(skip_serializing)]
    pub panel: Option<PanelView>,
    /// Floating pop-out preview windows to re-open.
    pub popped: Vec<PoppedView>,
}

/// The dockable Node Preview panel's persisted target + view.
#[derive(serde::Serialize, serde::Deserialize)]
pub(super) struct PanelView {
    pub nav: Vec<NodeId>,
    pub node: NodeId,
    pub is3d: bool,
    pub view: PreviewView,
}

/// One floating pop-out preview window's persisted target + view.
#[derive(serde::Serialize, serde::Deserialize)]
pub(super) struct PoppedView {
    pub nav: Vec<NodeId>,
    pub node: NodeId,
    pub is3d: bool,
    pub size: f32,
    pub view: PreviewView,
}

/// Snapshot the editor + preview panel SET into an [`EditorView`] (the resumable state half of the doc).
/// Only TOP-LEVEL node settings are captured — `caches.views` holds the current nav level's nodes,
/// which at save time (the toolbar is on the top graph) is the root. Each live preview instance with a
/// target becomes one `panels` entry.
pub(super) fn gather_view(editor: &WorldGraphEditor, panels: &WorldgenPreviewPanels) -> EditorView {
    let panels = panels
        .map
        .values()
        .filter_map(|p| {
            p.target.as_ref().map(|(nav, node)| PanelView {
                nav: nav.clone(),
                node: *node,
                is3d: p.is3d,
                view: p.view(),
            })
        })
        .collect();
    let popped = editor
        .popped
        .iter()
        .map(|p| PoppedView { nav: p.nav.clone(), node: p.node, is3d: p.is3d, size: p.size, view: p.view() })
        .collect();
    EditorView { nodes: editor.caches.views.clone(), nav: editor.nav.clone(), panels, panel: None, popped }
}

/// Restore an [`EditorView`] into the editor + preview panel SET: per-node settings, nav (clamped so a
/// stale path can't panic), each dockable panel (spawned as a fresh instance + queued tab via
/// [`WorldgenPreviewPanels::open`] so its tab reopens populated), and the pop-out windows. A legacy single
/// `panel` field (old files) is folded in front of `panels`.
pub(super) fn apply_view(view: EditorView, editor: &mut WorldGraphEditor, panels: &mut WorldgenPreviewPanels) {
    editor.caches.views = view.nodes;
    // Clamp the saved nav to the live biome chain so a deleted/renamed biome can't desync or panic.
    let depth = valid_depth(&editor.snarl, &view.nav);
    editor.nav = view.nav;
    editor.nav.truncate(depth);

    // Fold the deprecated single `panel` field (old saves) in front of the `panels` list, then spawn a
    // fresh preview instance + tab for each (allocates an id, queues `to_open`).
    for p in view.panel.into_iter().chain(view.panels) {
        panels.open((p.nav, p.node), p.view, p.is3d);
    }

    editor.popped = view
        .popped
        .into_iter()
        .map(|p| {
            let id = editor.next_pop_id;
            editor.next_pop_id += 1;
            PoppedPreview {
                id,
                nav: p.nav,
                node: p.node,
                half: p.view.half,
                cx: p.view.cx,
                cz: p.view.cz,
                size: p.size,
                is3d: p.is3d,
                cam: (p.view.yaw, p.view.pitch),
                open: true,
            }
        })
        .collect();
}

/// Load the editor document for `graph_path`, degrading gracefully:
/// 1. parse the hierarchical `.worldgraph.ron` as a [`WorldGraphDoc`] (snarl + view);
/// 2. else parse it as a BARE pre-doc `Snarl<EdNode>` (old format → default view);
/// 3. else parse the flat `.graph.ron` as a [`GraphAsset`] → snarl (default view);
/// 4. else the built-in default world (default view).
///
/// A file that is PRESENT but fails BOTH doc + bare-snarl parses is `warn!`ed (then falls through), so
/// a corrupt save surfaces rather than silently vanishing.
pub(super) fn load_editor_doc(graph_path: &str) -> (Snarl<EdNode>, EditorView) {
    let wg = worldgraph_path(graph_path);
    if let Ok(s) = std::fs::read_to_string(&wg) {
        match ron::de::from_str::<WorldGraphDoc>(&s) {
            Ok(doc) => return (doc.snarl, doc.view),
            Err(doc_err) => match ron::de::from_str::<Snarl<EdNode>>(&s) {
                // Backward compat: a pre-doc `.worldgraph.ron` was a bare snarl — load it, view defaults.
                Ok(snarl) => return (snarl, EditorView::default()),
                Err(_) => warn!("worldgen: hierarchical graph '{wg}' is corrupt ({doc_err}); trying the flat graph"),
            },
        }
    }
    if let Ok(s) = std::fs::read_to_string(graph_path) {
        match ron::de::from_str::<GraphAsset>(&s) {
            Ok(asset) => return (graph_to_snarl(&asset.graph), EditorView::default()),
            Err(e) => warn!("worldgen: flat graph '{graph_path}' is corrupt ({e}); using the default world"),
        }
    }
    (world_biome_snarl(), EditorView::default())
}

/// Write the hierarchical editor document (versioned snarl + gathered view-state) to the
/// `.worldgraph.ron` sibling of `editor.path`. The flat `.graph.ron` is written separately by the Save
/// button (the engine asset the world hot-reloads).
pub(super) fn save_editor_doc(editor: &WorldGraphEditor, panels: &WorldgenPreviewPanels) -> Result<(), String> {
    let doc = WorldGraphDoc { version: doc_version(), snarl: editor.snarl.clone(), view: gather_view(editor, panels) };
    let s = ron::ser::to_string_pretty(&doc, ron::ser::PrettyConfig::default()).map_err(|e| e.to_string())?;
    std::fs::write(worldgraph_path(&editor.path), s).map_err(|e| e.to_string())
}

// --- Auto-persisted session (resume on reopen, no explicit Save) --------------------------------------

/// Auto-persisted session file, under the editor's gitignored `.soul` dir beside the dock `layout.ron`.
/// It holds the LIVE editor doc (snarl + view), written on app exit and PREFERRED by the startup seed —
/// so closing and reopening the editor resumes exactly where you left off WITHOUT an explicit Save (Save
/// stays reserved for committing to the asset `.worldgraph.ron` / `.graph.ron`). Mirrors how the dock
/// layout auto-persists, the same robust pattern for "editor session UI state".
fn session_path() -> std::path::PathBuf {
    std::path::Path::new(".soul").join("worldgraph_session.ron")
}

/// Write the live editor doc to the session file (called on app exit). Best-effort: a write failure just
/// means the next launch falls back to the on-disk asset.
pub(super) fn save_session(editor: &WorldGraphEditor, panels: &WorldgenPreviewPanels) {
    let doc = WorldGraphDoc { version: doc_version(), snarl: editor.snarl.clone(), view: gather_view(editor, panels) };
    if let Ok(s) = ron::ser::to_string_pretty(&doc, ron::ser::PrettyConfig::default()) {
        let _ = std::fs::create_dir_all(std::path::Path::new(".soul"));
        let _ = std::fs::write(session_path(), s);
    }
}

/// Load the auto-persisted session doc (snarl + view), if present and valid — preferred over the on-disk
/// asset by the startup seed. A corrupt session is `warn!`ed and ignored (falls back to the asset).
pub(super) fn load_session() -> Option<(Snarl<EdNode>, EditorView)> {
    let s = std::fs::read_to_string(session_path()).ok()?;
    match ron::de::from_str::<WorldGraphDoc>(&s) {
        Ok(doc) => Some((doc.snarl, doc.view)),
        Err(e) => {
            warn!("worldgen: session '{}' is corrupt ({e}); ignoring", session_path().display());
            None
        }
    }
}
