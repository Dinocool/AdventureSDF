//! Clipboard + the structural editor ops (copy / cut / paste / delete) that produce undoable
//! [`SpawnCommand`] / [`DeleteCommand`]s. All reuse the `soul_scene` subtree serialization.

use bevy::prelude::*;

use crate::sdf_render::SdfSelection;
use crate::soul_scene::{LocalId, instantiate_scene_str, save_subtree_to_string};

use super::commands::{DeleteCommand, SpawnCommand};
use super::{EditHistories, ensure_id, local_id_of, next_free_local_id};

/// Holds the most recently copied/cut subtree, serialized to a `.scene` RON string. A `String`
/// (not live entities) so it survives scene swaps and repeated pastes.
#[derive(Resource, Default)]
pub struct EditorClipboard {
    pub content: Option<String>,
}

/// The currently selected entity, if any.
fn selected(world: &World) -> Option<Entity> {
    world.resource::<SdfSelection>().entity
}

/// Copy the selected subtree into the clipboard (serialized RON). No-op without a selection.
pub fn copy(world: &mut World) {
    let Some(e) = selected(world) else { return };
    ensure_id(world, e);
    let registry = world.resource::<AppTypeRegistry>().clone();
    let ron = {
        let reg = registry.read();
        save_subtree_to_string(world, &reg, e).ok()
    };
    if let Some(ron) = ron {
        world.resource_mut::<EditorClipboard>().content = Some(ron);
    }
}

/// Cut = copy then delete (one undo step for the delete).
pub fn cut(world: &mut World) {
    copy(world);
    delete_selected(world);
}

/// Paste the clipboard subtree as a child of the current selection (or a root if nothing is
/// selected), with fresh `LocalId`s so it doesn't collide with the original. Records a
/// [`SpawnCommand`]. No-op if the clipboard is empty.
pub fn paste(world: &mut World) {
    let Some(ron) = world.resource::<EditorClipboard>().content.clone() else {
        return;
    };
    let registry = world.resource::<AppTypeRegistry>().clone();
    let roots = {
        let reg = registry.read();
        match instantiate_scene_str(world, &ron, &reg) {
            Ok(r) if !r.is_empty() => r,
            _ => return,
        }
    };

    // The pasted entities carry the clipboard's `LocalId`s (used only transiently to wire
    // ChildOf during load); reassign fresh ids so they don't collide with the source/world.
    let mut next = next_free_local_id(world);
    for &root in &roots {
        reassign_ids(world, root, &mut next);
    }

    let root = roots[0];
    let parent = selected(world).filter(|&p| !roots.contains(&p));
    if let Some(pe) = parent {
        world.entity_mut(root).insert(ChildOf(pe));
    }
    world.resource_mut::<SdfSelection>().entity = Some(root);

    // Record for undo: serialize the now-fresh-id subtree so redo restores the same ids.
    let root_id = local_id_of(world, root);
    let subtree = {
        let reg = registry.read();
        save_subtree_to_string(world, &reg, root).ok()
    };
    let parent_id = parent.and_then(|pe| local_id_of(world, pe));
    if let (Some(root_id), Some(subtree)) = (root_id, subtree) {
        world
            .resource_mut::<EditHistories>()
            .record(Box::new(SpawnCommand::new(root_id, subtree, parent_id)));
    }
}

/// Delete the selected subtree, recording a [`DeleteCommand`] (undo restores it with the same
/// ids + parent). No-op without a selection.
pub fn delete_selected(world: &mut World) {
    let Some(e) = selected(world) else { return };
    let root_id = ensure_id(world, e);
    let parent_id = world
        .get::<ChildOf>(e)
        .map(|c| c.parent())
        .and_then(|p| local_id_of(world, p));
    let registry = world.resource::<AppTypeRegistry>().clone();
    let ron = {
        let reg = registry.read();
        save_subtree_to_string(world, &reg, e).ok()
    };
    let Some(ron) = ron else { return };

    world.entity_mut(e).despawn();
    world.resource_mut::<SdfSelection>().entity = None;
    world
        .resource_mut::<EditHistories>()
        .record(Box::new(DeleteCommand::new(root_id, ron, parent_id)));
}

/// Assign fresh sequential `LocalId`s to every entity in the subtree rooted at `root`,
/// advancing `next`. ChildOf links (which reference `Entity`, not `LocalId`) are unaffected.
fn reassign_ids(world: &mut World, root: Entity, next: &mut u64) {
    let mut stack = vec![root];
    while let Some(e) = stack.pop() {
        world.entity_mut(e).insert(LocalId(*next));
        *next += 1;
        if let Some(children) = world.get::<Children>(e) {
            stack.extend(children.iter());
        }
    }
}
