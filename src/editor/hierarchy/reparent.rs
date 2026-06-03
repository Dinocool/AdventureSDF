//! Action application for the hierarchy pass: rename, reparent (cycle-guarded, world-
//! transform-preserving), focus, and selection. The egui render pass only *accumulates*
//! [`Actions`]; they're applied to the world here afterward so we never mutate while a
//! tree/query borrow is live.

use bevy::prelude::*;

use crate::sdf_render::{OrbitFocus, SdfOrbitCamera, SdfSelection};

/// In-progress inline rename, stashed in egui temp memory between frames.
#[derive(Clone, Default)]
pub(super) struct RenameState {
    pub(super) entity: Option<Entity>,
    pub(super) buf: String,
}

/// Actions accumulated during the egui pass, applied to the world afterward (so we
/// never mutate while a query/tree borrow is live).
#[derive(Default)]
pub(super) struct Actions {
    pub(super) clicked: Option<Entity>,
    pub(super) double_clicked: Option<Entity>,
    pub(super) start_rename: Option<(Entity, String)>,
    /// `(child, new_parent)` — `None` parent unparents to a root.
    pub(super) reparent: Option<(Entity, Option<Entity>)>,
    /// Snap the editor viewport to look through this node (a scene camera).
    pub(super) look_through: Option<Entity>,
    /// Clicked empty space in the tree → clear the selection.
    pub(super) deselect: bool,
}

/// Apply the egui pass's accumulated actions to the world.
pub(super) fn apply_actions(
    world: &mut World,
    rename: &mut RenameState,
    commit_rename: bool,
    actions: Actions,
) {
    if let Some((entity, name)) = actions.start_rename {
        rename.entity = Some(entity);
        rename.buf = name;
    }

    if commit_rename {
        if let Some(entity) = rename.entity {
            let new = rename.buf.trim().to_string();
            let old = world
                .get::<Name>(entity)
                .map(|n| n.as_str().to_string())
                .unwrap_or_default();
            if !new.is_empty() && new != old {
                if let Ok(mut e) = world.get_entity_mut(entity) {
                    e.insert(Name::new(new.clone()));
                }
                let id = crate::editor::history::ensure_id(world, entity);
                world
                    .resource_mut::<crate::editor::history::EditHistories>()
                    .record(Box::new(crate::editor::history::RenameCommand::new(
                        id, old, new,
                    )));
            }
        }
        *rename = RenameState::default();
    }

    // Reparent (cycle-guarded): never parent a node under itself or a descendant.
    if let Some((child, new_parent)) = actions.reparent {
        let old_parent = world.get::<ChildOf>(child).map(|c| c.parent());
        let old_local = world.get::<Transform>(child).copied().unwrap_or_default();
        let mut applied = false;
        match new_parent {
            Some(parent) if parent != child && !is_descendant(world, parent, child) => {
                reparent_preserving_world(world, child, parent);
                applied = true;
            }
            Some(_) => {} // self or cycle — ignore
            None => {
                // Unparented: local transform becomes the former world transform.
                let cg = world.get::<GlobalTransform>(child).copied();
                if let Ok(mut e) = world.get_entity_mut(child) {
                    if let Some(cg) = cg {
                        e.insert(cg.compute_transform());
                    }
                    e.remove::<ChildOf>();
                }
                applied = true;
            }
        }
        if applied {
            let new_parent_e = world.get::<ChildOf>(child).map(|c| c.parent());
            let new_local = world.get::<Transform>(child).copied().unwrap_or_default();
            let cid = crate::editor::history::ensure_id(world, child);
            let old_pid = old_parent.and_then(|p| crate::editor::history::local_id_of(world, p));
            let new_pid = new_parent_e.and_then(|p| crate::editor::history::local_id_of(world, p));
            world
                .resource_mut::<crate::editor::history::EditHistories>()
                .record(Box::new(crate::editor::history::ReparentCommand::new(
                    cid, old_pid, old_local, new_pid, new_local,
                )));
        }
    }

    // Double-click focuses the orbit camera on the node (if it has a Transform).
    if let Some(entity) = actions.double_clicked {
        let pos = world.get::<Transform>(entity).map(|t| t.translation);
        if let Some(pos) = pos {
            world.resource_mut::<OrbitFocus>().target = Some(pos);
        }
    }

    // "Look through": snap the editor orbit camera to a node's pose. The orbit camera sits
    // at `target - dir*distance` (dir = the orbit offset) and looks at `target`. To match
    // the node's view (position `p`, forward `fwd`), we put `target` ahead of the node and
    // set the orbit dir = `-fwd`, then derive yaw/pitch the same way `orbit_camera` reads them.
    if let Some(entity) = actions.look_through
        && let Some(gt) = world.get::<GlobalTransform>(entity).copied()
    {
        let (_, rot, p) = gt.to_scale_rotation_translation();
        let fwd = (rot * Vec3::NEG_Z).normalize_or_zero();
        let mut orbit = world.resource_mut::<SdfOrbitCamera>();
        let dir = -fwd; // orbit offset direction (camera = target - dir*distance)
        orbit.target = p + fwd * orbit.distance;
        orbit.yaw = dir.z.atan2(dir.x);
        orbit.pitch = dir.y.clamp(-1.0, 1.0).asin();
    }

    if let Some(entity) = actions.clicked.or(actions.double_clicked) {
        world.resource_mut::<SdfSelection>().entity = Some(entity);
    } else if actions.deselect {
        // Clicked empty tree space → select nothing.
        world.resource_mut::<SdfSelection>().entity = None;
    }
}

/// True if `candidate` is `ancestor` or appears in `ancestor`'s `Children` subtree.
/// Used to reject reparent operations that would create a cycle.
fn is_descendant(world: &World, candidate: Entity, ancestor: Entity) -> bool {
    if candidate == ancestor {
        return true;
    }
    let mut stack = vec![ancestor];
    while let Some(e) = stack.pop() {
        if let Some(children) = world.get::<Children>(e) {
            for child in children.iter() {
                if child == candidate {
                    return true;
                }
                stack.push(child);
            }
        }
    }
    false
}

/// Parent `child` under `parent`, preserving the child's world transform. Bevy keeps
/// the child's *local* Transform across a reparent, so under a non-identity parent the
/// node would visually jump; recompute the local transform via `reparented_to`.
/// Caller must have already rejected cycles (`is_descendant`).
pub(super) fn reparent_preserving_world(world: &mut World, child: Entity, parent: Entity) {
    let cg = world.get::<GlobalTransform>(child).copied();
    let pg = world.get::<GlobalTransform>(parent).copied();
    if let (Some(cg), Some(pg)) = (cg, pg) {
        let local = cg.reparented_to(&pg);
        if let Ok(mut e) = world.get_entity_mut(child) {
            e.insert((ChildOf(parent), local));
        }
    } else if let Ok(mut e) = world.get_entity_mut(child) {
        e.insert(ChildOf(parent));
    }
}
