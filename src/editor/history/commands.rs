//! Concrete [`EditCommand`]s. The generic [`ComponentEdit`] covers *any* component value edit
//! via reflection (so one type serves transform/primitive/op/material/... edits); the rest are
//! structural (spawn / delete / reparent / rename). All reference entities by [`LocalId`].

use std::any::Any;

use bevy::ecs::reflect::ReflectComponent;
use bevy::prelude::*;
use bevy::reflect::serde::ReflectDeserializer;
use serde::de::DeserializeSeed;

use crate::sdf_render::SdfSelection;
use crate::soul_scene::{LocalId, instantiate_scene_str};

use super::{EditCommand, resolve};

// --- Reflection round-trip helpers -------------------------------------------------------

/// Apply a RON-serialized component value (from [`super::reflect_to_ron`]) back onto `entity`.
fn apply_ron_to_component(world: &mut World, entity: Entity, type_path: &str, ron: &str) {
    // Clone the registry handle so the read guard doesn't borrow `world` across get_entity_mut.
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    let Some(reg) = registry.get_with_type_path(type_path) else {
        return;
    };
    let Some(rc) = reg.data::<ReflectComponent>().cloned() else {
        return;
    };
    let de = ReflectDeserializer::new(&registry);
    let Ok(mut ron_de) = ron::Deserializer::from_str(ron) else {
        return;
    };
    let Ok(value) = de.deserialize(&mut ron_de) else {
        return;
    };
    drop(registry);
    if let Ok(mut em) = world.get_entity_mut(entity) {
        rc.apply(&mut em, &*value);
    }
}

// --- ComponentEdit (generic value edit) --------------------------------------------------

/// A value change to one component, stored as before/after RON. Reverting/redoing just
/// reflection-applies the respective snapshot — works for every reflected component with no
/// per-type code.
pub struct ComponentEdit {
    pub(crate) local_id: LocalId,
    pub(crate) type_path: String,
    pub(crate) before: String,
    pub(crate) after: String,
}

impl ComponentEdit {
    /// Build from before/after RON (the inspector already has the cloned values). Returns
    /// `None` if the two are identical (no-op edit) — nothing to record.
    pub fn new(local_id: LocalId, type_path: impl Into<String>, before: String, after: String) -> Option<Self> {
        if before == after {
            return None;
        }
        Some(Self { local_id, type_path: type_path.into(), before, after })
    }
}

impl EditCommand for ComponentEdit {
    fn revert(&self, world: &mut World) {
        if let Some(e) = resolve(world, self.local_id) {
            apply_ron_to_component(world, e, &self.type_path, &self.before);
        }
    }
    fn reapply(&self, world: &mut World) {
        if let Some(e) = resolve(world, self.local_id) {
            apply_ron_to_component(world, e, &self.type_path, &self.after);
        }
    }
    fn label(&self) -> String {
        let short = self.type_path.rsplit("::").next().unwrap_or(&self.type_path);
        format!("Edit {short}")
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// --- Structural helpers ------------------------------------------------------------------

/// Despawn the subtree rooted at `id` (Bevy `despawn` is recursive). Clears selection if the
/// removed root was selected.
fn despawn_subtree(world: &mut World, id: LocalId) {
    if let Some(e) = resolve(world, id) {
        if world.resource::<SdfSelection>().entity == Some(e) {
            world.resource_mut::<SdfSelection>().entity = None;
        }
        world.entity_mut(e).despawn();
    }
}

/// Instantiate a serialized subtree, reparent its root under `parent` (if any), select it, and
/// return the new root entity.
fn instantiate_subtree(world: &mut World, ron: &str, parent: Option<LocalId>) -> Option<Entity> {
    let registry = world.resource::<AppTypeRegistry>().clone();
    let roots = {
        let reg = registry.read();
        instantiate_scene_str(world, ron, &reg).ok()?
    };
    let root = *roots.first()?;
    if let Some(pid) = parent
        && let Some(pe) = resolve(world, pid)
    {
        // The stored root transform is its LOCAL transform under the original parent, so a
        // plain ChildOf re-link restores it in place.
        world.entity_mut(root).insert(ChildOf(pe));
    }
    world.resource_mut::<SdfSelection>().entity = Some(root);
    Some(root)
}

// --- SpawnCommand / DeleteCommand --------------------------------------------------------

/// A node (subtree) was added — from the Create-Node catalog or a paste. Undo despawns it,
/// redo re-instantiates it (same `LocalId`s, same parent).
pub struct SpawnCommand {
    root: LocalId,
    subtree_ron: String,
    parent: Option<LocalId>,
}

impl SpawnCommand {
    pub fn new(root: LocalId, subtree_ron: String, parent: Option<LocalId>) -> Self {
        Self { root, subtree_ron, parent }
    }
}

impl EditCommand for SpawnCommand {
    fn revert(&self, world: &mut World) {
        despawn_subtree(world, self.root);
    }
    fn reapply(&self, world: &mut World) {
        instantiate_subtree(world, &self.subtree_ron, self.parent);
    }
    fn label(&self) -> String {
        "Add node".into()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A node (subtree) was deleted. Undo re-instantiates it (restoring its `LocalId`s + parent so
/// later history still resolves it), redo despawns it again. The inverse of [`SpawnCommand`].
pub struct DeleteCommand {
    root: LocalId,
    subtree_ron: String,
    parent: Option<LocalId>,
}

impl DeleteCommand {
    pub fn new(root: LocalId, subtree_ron: String, parent: Option<LocalId>) -> Self {
        Self { root, subtree_ron, parent }
    }
}

impl EditCommand for DeleteCommand {
    fn revert(&self, world: &mut World) {
        instantiate_subtree(world, &self.subtree_ron, self.parent);
    }
    fn reapply(&self, world: &mut World) {
        despawn_subtree(world, self.root);
    }
    fn label(&self) -> String {
        "Delete node".into()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// --- ReparentCommand ---------------------------------------------------------------------

/// A node moved in the hierarchy. Stores both endpoints (parent + local transform) so undo and
/// redo each restore the matching side — the local transform is captured because reparenting
/// rewrites it to preserve world position.
pub struct ReparentCommand {
    child: LocalId,
    old_parent: Option<LocalId>,
    old_local: Transform,
    new_parent: Option<LocalId>,
    new_local: Transform,
}

impl ReparentCommand {
    pub fn new(
        child: LocalId,
        old_parent: Option<LocalId>,
        old_local: Transform,
        new_parent: Option<LocalId>,
        new_local: Transform,
    ) -> Self {
        Self { child, old_parent, old_local, new_parent, new_local }
    }

    fn set(world: &mut World, child: Entity, parent: Option<Entity>, local: Transform) {
        let Ok(mut em) = world.get_entity_mut(child) else {
            return;
        };
        match parent {
            Some(p) => {
                em.insert((ChildOf(p), local));
            }
            None => {
                em.remove::<ChildOf>();
                em.insert(local);
            }
        }
    }
}

impl EditCommand for ReparentCommand {
    fn revert(&self, world: &mut World) {
        if let Some(c) = resolve(world, self.child) {
            let p = self.old_parent.and_then(|id| resolve(world, id));
            Self::set(world, c, p, self.old_local);
        }
    }
    fn reapply(&self, world: &mut World) {
        if let Some(c) = resolve(world, self.child) {
            let p = self.new_parent.and_then(|id| resolve(world, id));
            Self::set(world, c, p, self.new_local);
        }
    }
    fn label(&self) -> String {
        "Reparent".into()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// --- RenameCommand -----------------------------------------------------------------------

/// A node's `Name` changed.
pub struct RenameCommand {
    entity: LocalId,
    old: String,
    new: String,
}

impl RenameCommand {
    pub fn new(entity: LocalId, old: String, new: String) -> Self {
        Self { entity, old, new }
    }
    fn set(world: &mut World, entity: Entity, name: &str) {
        if let Ok(mut em) = world.get_entity_mut(entity) {
            em.insert(Name::new(name.to_owned()));
        }
    }
}

impl EditCommand for RenameCommand {
    fn revert(&self, world: &mut World) {
        if let Some(e) = resolve(world, self.entity) {
            Self::set(world, e, &self.old);
        }
    }
    fn reapply(&self, world: &mut World) {
        if let Some(e) = resolve(world, self.entity) {
            Self::set(world, e, &self.new);
        }
    }
    fn label(&self) -> String {
        format!("Rename to {}", self.new)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
