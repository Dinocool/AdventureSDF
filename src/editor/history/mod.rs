//! Undo/redo + clipboard for the scene editor.
//!
//! ## Model
//! A per-scene **command stack**. Every undoable mutation is an [`EditCommand`] with
//! `revert` (undo) and `reapply` (redo). UI-driven edits have *already* mutated the world by
//! the time they're recorded, so recording just pushes the command (carrying before/after);
//! programmatic ops (paste) use [`EditHistories::apply`] which executes then records.
//!
//! ## Stable identity
//! Commands reference entities by [`LocalId`] (the scene's stable per-entity id), never the
//! raw `Entity` — so a delete→undo that re-spawns an entity with a fresh `Entity` still
//! resolves. [`ensure_local_ids`] assigns a `LocalId` to every `SceneEntity` that lacks one
//! (newly spawned nodes), which ALSO makes them serializable (save filters on `LocalId`).
//!
//! ## Scope
//! History is **per scene tab** (keyed by [`SceneId`]); only one scene is live in the world at
//! a time, and `LocalId`s survive the snapshot round-trip when tabs swap, so a scene's stack
//! stays valid across activation. Selection/camera/panel changes are deliberately NOT recorded.
//!
//! Submodules: [`commands`] (the concrete command types), [`clipboard`] (copy/cut/paste).

mod clipboard;
mod commands;

#[cfg(test)]
mod tests;

pub use clipboard::{EditorClipboard, copy, cut, delete_selected, paste};
pub use commands::{ComponentEdit, DeleteCommand, RenameCommand, ReparentCommand, SpawnCommand};

use std::any::Any;
use std::collections::HashMap;

use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy_egui::{EguiContext, PrimaryEguiContext};

use crate::editor::scene_tabs::{SceneId, scene_tab_ids};
use crate::scene_manager::SceneEntity;
use crate::sdf_render::SdfSelection;
use crate::sdf_render::gizmo::GizmoState;
use crate::soul_scene::LocalId;

/// Hard cap on retained undo steps per scene, so a long session can't grow history unbounded.
const MAX_UNDO: usize = 256;

/// A single reversible scene edit. Object-safe so heterogeneous commands share one stack.
///
/// At record time the world already holds the *after* state (the UI mutation happened), so
/// neither hook is called on record — `revert` rolls back to before, `reapply` re-does after.
pub trait EditCommand: Send + Sync + 'static {
    /// Undo: restore the pre-edit state.
    fn revert(&self, world: &mut World);
    /// Redo: re-apply the edit.
    fn reapply(&self, world: &mut World);
    /// Short human label (for the Edit menu / tooltips).
    fn label(&self) -> String;
    /// Downcast hooks for coalescing (see [`SceneHistory::record`]).
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Per-scene undo/redo stacks + the in-flight coalescing target.
#[derive(Default)]
struct SceneHistory {
    undo: Vec<Box<dyn EditCommand>>,
    redo: Vec<Box<dyn EditCommand>>,
    /// While a pointer drag is active, consecutive edits to this `(LocalId, type_path)` fold
    /// into the top undo entry (one drag = one step). Sealed (cleared) on pointer release.
    open: Option<(LocalId, String)>,
}

impl SceneHistory {
    /// Push a command, clearing the redo branch. Coalesces consecutive same-target
    /// [`ComponentEdit`]s while `open` matches (a continuous drag).
    fn record(&mut self, cmd: Box<dyn EditCommand>) {
        if let Some(ce) = cmd.as_any().downcast_ref::<ComponentEdit>() {
            let key = (ce.local_id, ce.type_path.clone());
            if self.open.as_ref() == Some(&key)
                && let Some(top) = self.undo.last_mut()
                && let Some(top_ce) = top.as_any_mut().downcast_mut::<ComponentEdit>()
                && (top_ce.local_id, &top_ce.type_path) == (ce.local_id, &ce.type_path)
            {
                // Same target, same drag: extend the existing step's "after".
                top_ce.after = ce.after.clone();
                self.redo.clear();
                return;
            }
            self.open = Some(key);
        } else {
            self.open = None; // a structural edit breaks any open drag-coalescing
        }
        self.undo.push(cmd);
        self.redo.clear();
        if self.undo.len() > MAX_UNDO {
            self.undo.remove(0);
        }
    }
}

/// All scenes' histories + the active scene routing. A resource so any system can record.
#[derive(Resource, Default)]
pub struct EditHistories {
    scenes: HashMap<SceneId, SceneHistory>,
    /// Scene the recorder routes to — mirrored from `OpenScenes::active` each frame.
    active: Option<SceneId>,
}

impl EditHistories {
    fn active_mut(&mut self) -> Option<&mut SceneHistory> {
        let id = self.active?;
        Some(self.scenes.entry(id).or_default())
    }

    /// Record an already-applied command onto the active scene's stack.
    pub fn record(&mut self, cmd: Box<dyn EditCommand>) {
        if let Some(h) = self.active_mut() {
            h.record(cmd);
        }
    }

    /// Whether undo/redo are currently available (for menu enable state).
    pub fn can_undo(&self) -> bool {
        self.active
            .and_then(|id| self.scenes.get(&id))
            .is_some_and(|h| !h.undo.is_empty())
    }
    pub fn can_redo(&self) -> bool {
        self.active
            .and_then(|id| self.scenes.get(&id))
            .is_some_and(|h| !h.redo.is_empty())
    }
}

/// Resolve a stable [`LocalId`] to its live `Entity` (linear scan — history ops are rare and
/// user-driven, so a cached map isn't worth the invalidation surface).
pub(crate) fn resolve(world: &mut World, id: LocalId) -> Option<Entity> {
    world
        .query::<(Entity, &LocalId)>()
        .iter(world)
        .find(|(_, l)| **l == id)
        .map(|(e, _)| e)
}

/// The `LocalId` of an entity, if it has one.
pub(crate) fn local_id_of(world: &World, entity: Entity) -> Option<LocalId> {
    world.get::<LocalId>(entity).copied()
}

/// RON-serialize a reflected value with the world's type registry (for capturing the
/// before/after of a [`ComponentEdit`] from an already-cloned inspector value).
pub(crate) fn reflect_to_ron(world: &World, value: &dyn bevy::reflect::PartialReflect) -> Option<String> {
    let registry = world.resource::<AppTypeRegistry>().read();
    let ser = bevy::reflect::serde::ReflectSerializer::new(value, &registry);
    ron::ser::to_string(&ser).ok()
}

/// Ensure `entity` has a `LocalId` right now (rather than waiting for [`ensure_local_ids`] next
/// frame), so a command recorded this frame can reference it. Returns the id.
pub(crate) fn ensure_id(world: &mut World, entity: Entity) -> LocalId {
    if let Some(id) = local_id_of(world, entity) {
        return id;
    }
    let id = LocalId(next_free_local_id(world));
    world.entity_mut(entity).insert(id);
    id
}

/// Smallest unused `LocalId` value in the world (max existing + 1). Used both by
/// [`ensure_local_ids`] and clipboard paste remapping.
pub(crate) fn next_free_local_id(world: &mut World) -> u64 {
    world
        .query::<&LocalId>()
        .iter(world)
        .map(|l| l.0)
        .max()
        .map_or(0, |m| m + 1)
}

pub struct EditHistoryPlugin;

impl Plugin for EditHistoryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EditHistories>()
            .init_resource::<EditorClipboard>()
            .init_resource::<GizmoDragTracker>()
            .add_systems(
                Update,
                (
                    ensure_local_ids,
                    sync_active_scene,
                    seal_on_release,
                    shortcuts,
                    record_gizmo_edits,
                )
                    .run_if(in_state(crate::scene_manager::AppScene::SdfEditor)),
            );
    }
}

/// Tracks an in-progress gizmo drag so the whole drag becomes ONE undo step (start→end),
/// recorded on release. The gizmo's per-frame transform writes live in core (`sdf_render`),
/// so we observe `GizmoState.drag` here on the editor side rather than instrument them.
#[derive(Resource, Default)]
struct GizmoDragTracker {
    entity: Option<Entity>,
    start: Option<Transform>,
}

/// Record a gizmo move/rotate/scale as a single [`ComponentEdit`] on the dragged entity's
/// `Transform`, captured on drag release.
fn record_gizmo_edits(world: &mut World) {
    let start_xf = world.resource::<GizmoState>().drag.as_ref().map(|d| d.start_xf);
    if let Some(start) = start_xf {
        // Drag active: latch the start pose + target the first frame we see it.
        let sel = world.resource::<SdfSelection>().entity;
        let mut tr = world.resource_mut::<GizmoDragTracker>();
        if tr.entity.is_none() {
            tr.entity = sel;
            tr.start = Some(start);
        }
        return;
    }

    // Not dragging: if a drag just ended, record start→current.
    let (entity, start) = {
        let tr = world.resource::<GizmoDragTracker>();
        (tr.entity, tr.start)
    };
    let (Some(entity), Some(start)) = (entity, start) else {
        return;
    };
    {
        let mut tr = world.resource_mut::<GizmoDragTracker>();
        tr.entity = None;
        tr.start = None;
    }
    let Some(current) = world.get::<Transform>(entity).copied() else {
        return;
    };
    if current == start {
        return;
    }
    if let (Some(before), Some(after)) = (
        reflect_to_ron(world, start.as_partial_reflect()),
        reflect_to_ron(world, current.as_partial_reflect()),
    ) {
        let id = ensure_id(world, entity);
        if let Some(cmd) = ComponentEdit::new(id, Transform::type_path(), before, after) {
            world.resource_mut::<EditHistories>().record(Box::new(cmd));
        }
    }
}

/// Undo/redo/copy/cut/paste/delete keyboard shortcuts. Exclusive-world (the command ops need
/// full mutation). Suppressed while egui is capturing the keyboard (text field focused) so the
/// same chords edit text instead.
fn shortcuts(world: &mut World) {
    let wants_kb = world
        .query_filtered::<&mut EguiContext, With<PrimaryEguiContext>>()
        .single_mut(world)
        .map(|mut c| c.get_mut().wants_keyboard_input())
        .unwrap_or(false);
    if wants_kb {
        return;
    }

    let kb = world.resource::<ButtonInput<KeyCode>>();
    let ctrl = kb.pressed(KeyCode::ControlLeft) || kb.pressed(KeyCode::ControlRight);
    let shift = kb.pressed(KeyCode::ShiftLeft) || kb.pressed(KeyCode::ShiftRight);
    let (z, y, c, x, v) = (
        kb.just_pressed(KeyCode::KeyZ),
        kb.just_pressed(KeyCode::KeyY),
        kb.just_pressed(KeyCode::KeyC),
        kb.just_pressed(KeyCode::KeyX),
        kb.just_pressed(KeyCode::KeyV),
    );
    // X with no modifier deletes; Ctrl+X cuts. (In the non-ctrl branch below, `!ctrl` holds.)
    let delete = kb.just_pressed(KeyCode::Delete) || x;
    // `kb` (immutable resource borrow) ends here; the dispatch below needs `&mut World`.

    if ctrl {
        if z && shift || y {
            redo(world);
        } else if z {
            undo(world);
        } else if c {
            clipboard::copy(world);
        } else if x {
            clipboard::cut(world);
        } else if v {
            clipboard::paste(world);
        }
    } else if delete {
        clipboard::delete_selected(world);
    }
}

/// Give every `SceneEntity` that lacks a [`LocalId`] a fresh unique one. Runs each frame but
/// only touches newly-spawned entities. This is what lets history reference new nodes by a
/// stable id — and (separately) what lets `save` include them, since it filters on `LocalId`.
fn ensure_local_ids(
    mut commands: Commands,
    missing: Query<Entity, (With<SceneEntity>, Without<LocalId>)>,
    existing: Query<&LocalId>,
) {
    if missing.is_empty() {
        return;
    }
    let base = existing.iter().map(|l| l.0).max().map_or(0, |m| m + 1);
    for (i, entity) in missing.iter().enumerate() {
        commands.entity(entity).insert(LocalId(base + i as u64));
    }
}

/// Mirror the active scene id into [`EditHistories`] so recorders route to the right stack.
fn sync_active_scene(world: &mut World) {
    let active = scene_tab_ids(world).1;
    world.resource_mut::<EditHistories>().active = active;
}

/// Seal drag-coalescing when the (left) mouse button isn't held, so a new click-drag on the
/// same field starts a fresh undo step instead of folding into the previous one.
fn seal_on_release(mouse: Res<ButtonInput<MouseButton>>, mut hist: ResMut<EditHistories>) {
    if !mouse.pressed(MouseButton::Left)
        && let Some(id) = hist.active
        && let Some(h) = hist.scenes.get_mut(&id)
    {
        h.open = None;
    }
}

/// Undo one step on the active scene (no-op if the stack is empty). Exclusive-world so commands
/// get full mutation access.
pub fn undo(world: &mut World) {
    let Some(cmd) = pop_undo(world) else { return };
    cmd.revert(world);
    let mut hist = world.resource_mut::<EditHistories>();
    if let Some(h) = hist.active_mut() {
        h.redo.push(cmd);
    }
}

/// Redo one step on the active scene.
pub fn redo(world: &mut World) {
    let Some(cmd) = pop_redo(world) else { return };
    cmd.reapply(world);
    let mut hist = world.resource_mut::<EditHistories>();
    if let Some(h) = hist.active_mut() {
        h.undo.push(cmd);
    }
}

fn pop_undo(world: &mut World) -> Option<Box<dyn EditCommand>> {
    let mut hist = world.resource_mut::<EditHistories>();
    let h = hist.active_mut()?;
    h.open = None;
    h.undo.pop()
}

fn pop_redo(world: &mut World) -> Option<Box<dyn EditCommand>> {
    let mut hist = world.resource_mut::<EditHistories>();
    let h = hist.active_mut()?;
    h.open = None;
    h.redo.pop()
}
