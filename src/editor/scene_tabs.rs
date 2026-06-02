//! Multi-scene document model. The editor can hold several scenes open at once, one tab
//! each (named after the file), but only ONE is live in the world — the global SDF atlas /
//! bake state can't represent two scenes simultaneously. Switching tabs therefore
//! *swaps*: the outgoing scene is serialized to an in-memory snapshot (so unsaved edits
//! survive), despawned, and the incoming scene is loaded from its snapshot or disk.
//!
//! This module owns the orchestration (open / new / save / save-as / activate / close +
//! dirty tracking). The low-level scene I/O lives in `soul_scene`; the tab widgets live in
//! `dock`. Everything is driven from `dock::show_editor_dock` each frame, where the dock
//! state and `&mut World` are both available.

use std::path::{Path, PathBuf};

use bevy::prelude::*;
use bevy_egui::egui;
use egui_dock::SurfaceIndex;

use crate::scene_manager::{EditorEntity, SceneEntity};
use crate::sdf_render::{DEFAULT_SCENE_PATH, SdfOrbitCamera};
use crate::soul_scene::{
    EditorCamera, LoadedEditorCamera, despawn_scene_content, load_scene, load_scene_from_str,
    save_scene_to_string, save_scene_to_string_with_camera,
};

use super::dock::{center_leaf, EditorDockState, EditorTab};
use super::menu_bar::{CurrentScenePath, EditorRequests};
use super::scene_browser::{SCENES_ROOT, SaveSceneDialog};

/// Stable per-session identifier for an open scene tab.
pub type SceneId = u32;

/// The scene loaded at startup (the gallery). The dock seeds its first tab with this id and
/// [`OpenScenes::default`] makes the matching document, so the two agree from frame one.
pub const INITIAL_SCENE_ID: SceneId = 0;

/// A per-scene snapshot of the orbit camera, so each tab restores its own view on activate.
#[derive(Clone, Copy)]
struct CameraState {
    target: Vec3,
    distance: f32,
    yaw: f32,
    pitch: f32,
}

impl CameraState {
    fn capture(c: &SdfOrbitCamera) -> Self {
        Self {
            target: c.target,
            distance: c.distance,
            yaw: c.yaw,
            pitch: c.pitch,
        }
    }

    fn restore(self, c: &mut SdfOrbitCamera) {
        c.target = self.target;
        c.distance = self.distance;
        c.yaw = self.yaw;
        c.pitch = self.pitch;
    }

    /// The persistable form written into a `.scene` file.
    fn to_editor_camera(self) -> EditorCamera {
        EditorCamera {
            target: self.target.to_array(),
            distance: self.distance,
            yaw: self.yaw,
            pitch: self.pitch,
        }
    }

    fn from_editor_camera(e: EditorCamera) -> Self {
        Self {
            target: Vec3::from_array(e.target),
            distance: e.distance,
            yaw: e.yaw,
            pitch: e.pitch,
        }
    }
}

/// One open scene document.
pub struct SceneDoc {
    pub id: SceneId,
    /// Disk path, or `None` for a never-saved scene.
    pub path: Option<PathBuf>,
    /// Display name for the tab (file stem, or `untitled-N`).
    pub title: String,
    /// Serialized contents for an *inactive* doc (carries unsaved edits across a swap). The
    /// active doc's truth is the world, so its snapshot is stale until it's swapped out.
    snapshot: Option<String>,
    /// Contents as last saved/loaded — the clean baseline that dirty-detection compares
    /// against. `None` until first captured (right after a load).
    baseline: Option<String>,
    /// This scene's saved camera view, restored on activate.
    camera: Option<CameraState>,
    /// Cached "has unsaved edits" flag (drives the tab's `*` marker and the close prompt).
    pub dirty: bool,
}

/// All open scenes + the swap/close request channel the dock writes to. `active` is `None`
/// when every scene has been closed — the dock then shows the [`EditorTab::NoScene`]
/// placeholder and the world holds no scene content.
#[derive(Resource)]
pub struct OpenScenes {
    docs: Vec<SceneDoc>,
    active: Option<SceneId>,
    next_id: SceneId,
    /// The scene tab whose `ui()` ran this frame (i.e. the visible one). The dock sets this;
    /// [`handle_activation`] swaps the world to it when it differs from `active`.
    pub rendered: Option<SceneId>,
    /// A tab's close button was clicked. Drained by [`handle_close`].
    pub close_request: Option<SceneId>,
    /// While `Some`, the unsaved-changes confirm dialog is showing for this (now-active) doc.
    confirm_close: Option<SceneId>,
    /// Elapsed-seconds gate for the dirty re-check, so we don't serialize the whole scene
    /// every frame just to refresh a cosmetic `*` marker.
    next_dirty_check: f32,
}

impl Default for OpenScenes {
    fn default() -> Self {
        let path = PathBuf::from(DEFAULT_SCENE_PATH);
        let title = stem(&path);
        Self {
            docs: vec![SceneDoc {
                id: INITIAL_SCENE_ID,
                path: Some(path),
                title,
                snapshot: None,
                baseline: None,
                camera: None,
                dirty: false,
            }],
            active: Some(INITIAL_SCENE_ID),
            next_id: INITIAL_SCENE_ID + 1,
            rendered: None,
            close_request: None,
            confirm_close: None,
            next_dirty_check: 0.0,
        }
    }
}

impl OpenScenes {
    fn index_of(&self, id: SceneId) -> Option<usize> {
        self.docs.iter().position(|d| d.id == id)
    }

    fn active_index(&self) -> Option<usize> {
        self.active.and_then(|a| self.index_of(a))
    }

    /// Title for the tab labelled `id` (with a `*` when dirty), or a fallback.
    pub fn tab_title(&self, id: SceneId) -> String {
        match self.docs.iter().find(|d| d.id == id) {
            Some(d) if d.dirty => format!("{}*", d.title),
            Some(d) => d.title.clone(),
            None => "Scene".to_string(),
        }
    }
}

/// File stem (no extension) as the tab name, e.g. `assets/scenes/gallery.scene` → `gallery`.
fn stem(path: &Path) -> String {
    path.file_stem()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "scene".to_string())
}

/// Serialize the live scene to a `.scene` RON string (the active doc's current contents).
fn serialize_world(world: &mut World, registry: &AppTypeRegistry) -> Option<String> {
    let reg = registry.read();
    save_scene_to_string(world, &reg).ok()
}

/// Whether the world currently holds any loaded scene content (excludes editor infra like
/// the persistent camera). Used to avoid baselining a path-backed scene before it's loaded.
fn scene_has_content(world: &mut World) -> bool {
    world
        .query_filtered::<(), (With<SceneEntity>, Without<EditorEntity>)>()
        .iter(world)
        .next()
        .is_some()
}

/// Allocate a fresh scene id.
fn alloc_id(world: &mut World) -> SceneId {
    let mut open = world.resource_mut::<OpenScenes>();
    let id = open.next_id;
    open.next_id += 1;
    id
}

/// Mirror the active doc's path into `CurrentScenePath` so the menu/status read-outs and any
/// path-based UI stay in step with the focused tab. Empty when no scene is open.
fn sync_current_path(world: &mut World) {
    let path = {
        let open = world.resource::<OpenScenes>();
        match open.active_index() {
            Some(i) => open.docs[i]
                .path
                .clone()
                .unwrap_or_else(|| PathBuf::from(&open.docs[i].title)),
            None => PathBuf::new(),
        }
    };
    world.resource_mut::<CurrentScenePath>().0 = path;
}

// --- Swap primitives ---------------------------------------------------------------------

/// Serialize the current active scene into its own doc (snapshot + camera + dirty), so a
/// later activate can restore it exactly, edits and view included.
fn snapshot_active(world: &mut World, registry: &AppTypeRegistry) {
    if world.resource::<OpenScenes>().active.is_none() {
        return; // nothing live to snapshot
    }
    let ron = serialize_world(world, registry);
    let cam = CameraState::capture(world.resource::<SdfOrbitCamera>());
    let mut open = world.resource_mut::<OpenScenes>();
    let Some(i) = open.active_index() else {
        return;
    };
    let doc = &mut open.docs[i];
    if let Some(ron) = ron {
        doc.dirty = doc.baseline.as_ref().is_none_or(|b| *b != ron);
        doc.snapshot = Some(ron);
    }
    doc.camera = Some(cam);
}

/// Replace the world's scene content with document `doc_index`'s: despawn the current scene,
/// spawn from snapshot (preferred, carries edits) or disk path, restore its camera, and
/// capture a clean baseline on first load.
fn load_doc_into_world(world: &mut World, registry: &AppTypeRegistry, doc_index: usize) {
    let (snapshot, path, camera, has_baseline) = {
        let open = world.resource::<OpenScenes>();
        let d = &open.docs[doc_index];
        (d.snapshot.clone(), d.path.clone(), d.camera, d.baseline.is_some())
    };

    despawn_scene_content(world);

    {
        let reg = registry.read();
        let result = match (&snapshot, &path) {
            (Some(s), _) => Some(load_scene_from_str(world, s, &reg)),
            (None, Some(p)) => Some(load_scene(world, p, &reg)),
            (None, None) => None, // brand-new empty scene: nothing to spawn
        };
        if let Some(Err(e)) = result {
            error!("scene load failed: {e}");
        }
    }

    // Camera to apply: the in-memory per-tab camera (set on swap-away) takes priority; on a
    // first load from disk there's none, so fall back to the camera saved in the file.
    let applied = camera.or_else(|| {
        world
            .resource::<LoadedEditorCamera>()
            .0
            .map(CameraState::from_editor_camera)
    });
    if let Some(cam) = applied {
        cam.restore(&mut world.resource_mut::<SdfOrbitCamera>());
        // Push the restored orbit state to the camera transform immediately — `orbit_camera`
        // only runs while the pointer is in the viewport, so otherwise the view wouldn't
        // update until the cursor re-entered it (a delayed "jump").
        crate::sdf_render::sync_orbit_camera_transform(world);
        // Persist it on the doc so later tab swaps reuse it.
        world.resource_mut::<OpenScenes>().docs[doc_index].camera = Some(cam);
    }

    // First time we materialize this doc, the freshly-loaded world IS the clean baseline.
    // (Comparing serialize-of-loaded against serialize-of-current is like-for-like, so no
    // spurious dirty from RON formatting differences.)
    if !has_baseline
        && let Some(base) = serialize_world(world, registry)
    {
        let mut open = world.resource_mut::<OpenScenes>();
        open.docs[doc_index].baseline = Some(base);
        open.docs[doc_index].dirty = false;
    }
}

/// Make `target` the active scene: snapshot the outgoing one, swap the world to `target`,
/// focus its tab, and drop the empty-state placeholder. A no-op swap (already active) still
/// re-focuses the tab.
fn activate(world: &mut World, dock: &mut EditorDockState, registry: &AppTypeRegistry, target: SceneId) {
    let current = world.resource::<OpenScenes>().active;
    if current != Some(target) {
        snapshot_active(world, registry);
        if let Some(ti) = world.resource::<OpenScenes>().index_of(target) {
            load_doc_into_world(world, registry, ti);
            world.resource_mut::<OpenScenes>().active = Some(target);
            sync_current_path(world);
        }
    }
    set_dock_active(dock, target);
    remove_no_scene_tab(dock);
}

/// The ids of all open scenes in dock-tab order, plus the active id (for re-injecting the
/// scene box when a layout is applied). Empty list ⇒ the empty-state placeholder.
pub fn scene_tab_ids(world: &World) -> (Vec<SceneId>, Option<SceneId>) {
    let open = world.resource::<OpenScenes>();
    (open.docs.iter().map(|d| d.id).collect(), open.active)
}

/// Append `tab` into the center (scene) leaf, falling back to the first leaf.
fn add_center_tab(dock: &mut EditorDockState, tab: EditorTab) {
    match center_leaf(dock) {
        Some(node) => dock.state.main_surface_mut()[node].append_tab(tab),
        None => dock.state.push_to_first_leaf(tab),
    }
}

/// Drop the empty-state placeholder tab, if present.
fn remove_no_scene_tab(dock: &mut EditorDockState) {
    if let Some((node, tab)) = dock.state.find_main_surface_tab(&EditorTab::NoScene) {
        dock.state.remove_tab((SurfaceIndex::main(), node, tab));
    }
}

/// Focus the tab for scene `id` (selects it within its leaf).
fn set_dock_active(dock: &mut EditorDockState, id: SceneId) {
    if let Some((node, tab)) = dock.state.find_main_surface_tab(&EditorTab::Scene(id)) {
        dock.state.set_active_tab((SurfaceIndex::main(), node, tab));
    }
}

// --- Request handling (called from show_editor_dock) -------------------------------------

/// Interval (seconds) between dirty re-checks. Serializing the scene every frame just to
/// keep the tab `*` marker live is wasteful; once a second is plenty.
const DIRTY_CHECK_INTERVAL: f32 = 1.0;

/// Recompute the active doc's dirty flag from the live world, throttled to a few times a
/// second. Keeps the tab `*` marker live without a per-frame full-scene serialize.
pub fn refresh_active_dirty(world: &mut World, registry: &AppTypeRegistry) {
    let now = world.resource::<Time>().elapsed_secs();
    if now < world.resource::<OpenScenes>().next_dirty_check {
        return;
    }
    world.resource_mut::<OpenScenes>().next_dirty_check = now + DIRTY_CHECK_INTERVAL;

    let Some(current) = serialize_world(world, registry) else {
        return;
    };
    let loaded = scene_has_content(world);
    let mut open = world.resource_mut::<OpenScenes>();
    let Some(i) = open.active_index() else {
        return; // no scene open
    };
    let doc = &mut open.docs[i];
    match &doc.baseline {
        // First sight of the startup scene (loaded by sdf_render, never through us): adopt
        // the current serialization as the clean baseline — but not before a path-backed
        // scene has actually loaded, else its first frame would look spuriously dirty.
        None => {
            if doc.path.is_some() && !loaded {
                return;
            }
            doc.baseline = Some(current);
            doc.dirty = false;
        }
        Some(base) => doc.dirty = *base != current,
    }
}

/// Drain File-menu requests (new / open / save / save-as) into document operations.
pub fn drain_requests(world: &mut World, dock: &mut EditorDockState, registry: &AppTypeRegistry) {
    let (do_new, open_path, do_save, save_as) = {
        let mut req = world.resource_mut::<EditorRequests>();
        (
            std::mem::take(&mut req.new_scene),
            req.open.take(),
            std::mem::replace(&mut req.save, false),
            req.save_as.take(),
        )
    };

    if let Some(dest) = save_as {
        save_active_to(world, registry, &dest);
    }

    if do_save {
        let path = {
            let open = world.resource::<OpenScenes>();
            open.active_index().map(|i| open.docs[i].path.clone())
        };
        match path {
            Some(Some(p)) => save_active_to(world, registry, &p),
            // Never-saved scene: route Save through the Save As browser.
            Some(None) => open_save_as_dialog(world),
            None => {} // no scene open; nothing to save
        }
    }

    if let Some(p) = open_path {
        open_path_as_tab(world, dock, registry, &p);
    }

    if do_new {
        new_scene_tab(world, dock, registry);
    }
}

/// Serialize the active world scene, write it to `dest` (with the current editor camera
/// embedded), and adopt it as the doc's path + clean baseline. The baseline is camera-free
/// so that merely moving the camera doesn't flag the scene dirty.
fn save_active_to(world: &mut World, registry: &AppTypeRegistry, dest: &Path) {
    let camera = CameraState::capture(world.resource::<SdfOrbitCamera>()).to_editor_camera();
    let (file_ron, baseline_ron) = {
        let reg = registry.read();
        (
            save_scene_to_string_with_camera(world, &reg, Some(camera)),
            save_scene_to_string(world, &reg),
        )
    };
    let (Ok(file_ron), Ok(baseline_ron)) = (file_ron, baseline_ron) else {
        error!("scene save failed: could not serialize world");
        notify_error(world, "Save failed: could not serialize scene");
        return;
    };
    if let Some(parent) = dest.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        error!("scene save failed: {e}");
        notify_error(world, format!("Save failed: {e}"));
        return;
    }
    if let Err(e) = std::fs::write(dest, &file_ron) {
        error!("scene save failed: {e}");
        notify_error(world, format!("Save failed: {e}"));
        return;
    }

    {
        let mut open = world.resource_mut::<OpenScenes>();
        let Some(i) = open.active_index() else {
            return;
        };
        let doc = &mut open.docs[i];
        doc.path = Some(dest.to_path_buf());
        doc.title = stem(dest);
        doc.baseline = Some(baseline_ron);
        doc.dirty = false;
    }
    sync_current_path(world);
    // Capture a viewport screenshot for this scene's asset-browser thumbnail.
    world
        .resource_mut::<crate::editor::assets_browser::PendingSceneThumbnail>()
        .0 = Some(dest.to_path_buf());
    info!("saved scene to {}", dest.display());
    world
        .resource_mut::<crate::editor::notifications::Notifications>()
        .success(format!("Saved {}", stem(dest)));
}

/// Push an error toast (used on save failures).
fn notify_error(world: &mut World, message: impl Into<String>) {
    world
        .resource_mut::<crate::editor::notifications::Notifications>()
        .error(message);
}

/// Open `path` in a tab: focus it if already open, else create a new doc + tab and activate.
fn open_path_as_tab(world: &mut World, dock: &mut EditorDockState, registry: &AppTypeRegistry, path: &Path) {
    let existing = world
        .resource::<OpenScenes>()
        .docs
        .iter()
        .find(|d| d.path.as_deref() == Some(path))
        .map(|d| d.id);
    if let Some(id) = existing {
        activate(world, dock, registry, id);
        return;
    }

    let id = alloc_id(world);
    world.resource_mut::<OpenScenes>().docs.push(SceneDoc {
        id,
        path: Some(path.to_path_buf()),
        title: stem(path),
        snapshot: None,
        baseline: None,
        camera: None,
        dirty: false,
    });
    add_center_tab(dock, EditorTab::Scene(id));
    activate(world, dock, registry, id);
}

/// Create a fresh empty scene in a new tab and activate it.
fn new_scene_tab(world: &mut World, dock: &mut EditorDockState, registry: &AppTypeRegistry) {
    let id = alloc_id(world);
    world.resource_mut::<OpenScenes>().docs.push(SceneDoc {
        id,
        path: None,
        title: format!("untitled-{id}"),
        snapshot: None, // no snapshot + no path ⇒ loads as an empty world
        baseline: None,
        camera: None,
        dirty: false,
    });
    add_center_tab(dock, EditorTab::Scene(id));
    activate(world, dock, registry, id);
}

/// React to a scene tab the user switched to (set by the dock during render).
pub fn handle_activation(world: &mut World, dock: &mut EditorDockState, registry: &AppTypeRegistry) {
    let rendered = world.resource_mut::<OpenScenes>().rendered.take();
    let active = world.resource::<OpenScenes>().active;
    if let Some(id) = rendered
        && Some(id) != active
    {
        activate(world, dock, registry, id);
    }
}

/// Drain a close request and run the close (with an unsaved-changes prompt when needed).
pub fn handle_close(
    world: &mut World,
    dock: &mut EditorDockState,
    registry: &AppTypeRegistry,
    ctx: &egui::Context,
) {
    if let Some(id) = world.resource_mut::<OpenScenes>().close_request.take() {
        // The dirty flag is throttled, so for the active doc force a fresh check before
        // deciding whether to prompt — otherwise a just-edited scene could close silently.
        if world.resource::<OpenScenes>().active == Some(id) {
            world.resource_mut::<OpenScenes>().next_dirty_check = 0.0;
            refresh_active_dirty(world, registry);
        }
        let dirty = world
            .resource::<OpenScenes>()
            .docs
            .iter()
            .find(|d| d.id == id)
            .is_some_and(|d| d.dirty);
        if dirty {
            // Bring the doc into the world first, so the prompt's "Save" acts on live state.
            activate(world, dock, registry, id);
            world.resource_mut::<OpenScenes>().confirm_close = Some(id);
        } else {
            close_doc(world, dock, registry, id);
        }
    }

    let confirm = world.resource::<OpenScenes>().confirm_close;
    if let Some(id) = confirm {
        confirm_close_dialog(world, dock, registry, ctx, id);
    }
}

/// The Save / Discard / Cancel modal for closing a scene with unsaved edits. The doc is
/// guaranteed active (see [`handle_close`]).
fn confirm_close_dialog(
    world: &mut World,
    dock: &mut EditorDockState,
    registry: &AppTypeRegistry,
    ctx: &egui::Context,
    id: SceneId,
) {
    #[derive(PartialEq)]
    enum Choice {
        Save,
        Discard,
        Cancel,
    }

    let title = world.resource::<OpenScenes>().tab_title(id);
    let mut choice = None;
    egui::Window::new("Unsaved changes")
        .id(egui::Id::new("scene_close_confirm"))
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.label(format!("\u{201C}{title}\u{201D} has unsaved changes."));
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    choice = Some(Choice::Save);
                }
                if ui.button("Discard").clicked() {
                    choice = Some(Choice::Discard);
                }
                if ui.button("Cancel").clicked() {
                    choice = Some(Choice::Cancel);
                }
            });
        });

    let clear = |world: &mut World| world.resource_mut::<OpenScenes>().confirm_close = None;
    match choice {
        Some(Choice::Save) => {
            let path = {
                let open = world.resource::<OpenScenes>();
                open.active_index().and_then(|i| open.docs[i].path.clone())
            };
            clear(world);
            match path {
                Some(p) => {
                    save_active_to(world, registry, &p);
                    close_doc(world, dock, registry, id);
                }
                // No path: send the user through Save As and abort the close for now.
                None => open_save_as_dialog(world),
            }
        }
        Some(Choice::Discard) => {
            clear(world);
            close_doc(world, dock, registry, id);
        }
        Some(Choice::Cancel) => clear(world),
        None => {}
    }
}

/// Open the Save As browser, pre-filled from the active doc.
fn open_save_as_dialog(world: &mut World) {
    let suggested = {
        let open = world.resource::<OpenScenes>();
        match open.active_index() {
            Some(i) => {
                let doc = &open.docs[i];
                doc.path.clone().unwrap_or_else(|| {
                    PathBuf::from(SCENES_ROOT).join(format!("{}.scene", doc.title))
                })
            }
            None => PathBuf::from(SCENES_ROOT).join("untitled.scene"),
        }
    };
    world.resource_mut::<SaveSceneDialog>().show_for(&suggested);
}

/// Remove document `id`: drop its tab. If it was the active scene, load a neighbour in its
/// place — or, if it was the last scene, blank the world and show the empty-state placeholder.
fn close_doc(world: &mut World, dock: &mut EditorDockState, registry: &AppTypeRegistry, id: SceneId) {
    let (idx, was_active, neighbor) = {
        let open = world.resource::<OpenScenes>();
        let Some(idx) = open.index_of(id) else {
            return;
        };
        // Neighbour to fall back to, or `None` when this is the last open scene.
        let neighbor = if open.docs.len() <= 1 {
            None
        } else if idx > 0 {
            Some(open.docs[idx - 1].id)
        } else {
            Some(open.docs[idx + 1].id)
        };
        (idx, open.active == Some(id), neighbor)
    };

    world.resource_mut::<OpenScenes>().docs.remove(idx);

    // Last scene closing: add the placeholder into the scene leaf BEFORE removing the scene
    // tab, so the (about-to-be-empty) leaf survives the removal.
    if neighbor.is_none() {
        add_center_tab(dock, EditorTab::NoScene);
    }
    if let Some((node, tab)) = dock.state.find_main_surface_tab(&EditorTab::Scene(id)) {
        dock.state.remove_tab((SurfaceIndex::main(), node, tab));
    }

    if was_active {
        match neighbor {
            Some(nid) => {
                world.resource_mut::<OpenScenes>().active = Some(nid);
                if let Some(ni) = world.resource::<OpenScenes>().index_of(nid) {
                    load_doc_into_world(world, registry, ni);
                    sync_current_path(world);
                }
                set_dock_active(dock, nid);
            }
            None => {
                // No scenes left: blank the world and focus the placeholder.
                despawn_scene_content(world);
                world.resource_mut::<OpenScenes>().active = None;
                sync_current_path(world);
                if let Some((node, tab)) = dock.state.find_main_surface_tab(&EditorTab::NoScene)
                {
                    dock.state.set_active_tab((SurfaceIndex::main(), node, tab));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stem_drops_extension() {
        assert_eq!(stem(Path::new("assets/scenes/gallery.scene")), "gallery");
        assert_eq!(stem(Path::new("level.scn.ron")), "level.scn");
        assert_eq!(stem(Path::new("noext")), "noext");
    }

    #[test]
    fn default_has_one_active_doc() {
        // Title derives from DEFAULT_SCENE_PATH, so this tracks whatever scene is the default.
        let expect = stem(Path::new(DEFAULT_SCENE_PATH));
        let open = OpenScenes::default();
        assert_eq!(open.docs.len(), 1);
        assert_eq!(open.active, Some(INITIAL_SCENE_ID));
        assert_eq!(open.next_id, INITIAL_SCENE_ID + 1);
        assert_eq!(open.docs[0].title, expect);
    }

    #[test]
    fn tab_title_marks_dirty_and_falls_back() {
        let expect = stem(Path::new(DEFAULT_SCENE_PATH));
        let mut open = OpenScenes::default();
        assert_eq!(open.tab_title(INITIAL_SCENE_ID), expect);
        open.docs[0].dirty = true;
        assert_eq!(open.tab_title(INITIAL_SCENE_ID), format!("{expect}*"));
        assert_eq!(open.tab_title(999), "Scene");
    }
}
