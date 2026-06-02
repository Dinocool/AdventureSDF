//! soul-engine custom scene format (`.scene`). Engine-core (not editor-gated): the
//! runtime needs to *load* scenes; the editor feature adds the save UI wiring.
//!
//! Scenes support Godot-style **nested instances**: a scene can instance another
//! `.scene` as a subtree with per-sub-entity overrides, with a live link to the
//! source file. See [`format`] (schema), [`load`] (instantiate + override-merge),
//! and [`save`] (world-walk + diff).

use std::path::PathBuf;

use bevy::prelude::*;

pub mod format;
pub mod load;
pub mod save;

#[cfg(test)]
mod tests;

pub use format::{LocalId, SceneFile, SceneRecord};
pub use load::{SceneLoadError, load_scene, load_scene_from_str};
pub use save::{SceneSaveError, save_scene, save_scene_to_string};

use format::ComponentMap;

/// On an instance root: the source `.scene` and the per-sub-entity overrides that
/// were applied on top of it. Lets save re-emit the ref + diffs losslessly instead
/// of the expanded subtree.
#[derive(Component, Reflect, Clone, Default)]
#[reflect(Component)]
pub struct SceneInstance {
    pub source: PathBuf,
    /// Overrides keyed by the *source* scene's [`LocalId`].
    #[reflect(ignore)]
    pub overrides: std::collections::HashMap<LocalId, ComponentMap>,
}

/// On every entity materialized from an instance's source subtree. Save prunes
/// these (the instance root's [`SceneInstance`] represents them).
#[derive(Component, Reflect, Clone, Copy)]
#[reflect(Component)]
pub struct InstanceChild {
    pub root: Entity,
}

/// Entities that must never be serialized — runtime-rebuilt children (e.g. baked
/// meshes) reconstructed from their parent's data on load. (jackdaw `NonSerializable`.)
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct NonSerializable;

/// Editor-time visual indicators (gizmo geometry, overlays) that render in the
/// viewport but never land in a `.scene`. (jackdaw `SkipSerialization`.)
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct SkipSerialization;

/// Hides an entity from editor-facing surfaces (hierarchy) and from save.
/// (jackdaw `EditorHidden`.)
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct EditorHidden;

pub struct SoulScenePlugin;

impl Plugin for SoulScenePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<LocalId>()
            .register_type::<SceneInstance>()
            .register_type::<InstanceChild>()
            .register_type::<NonSerializable>()
            .register_type::<SkipSerialization>()
            .register_type::<EditorHidden>();

        // Scene-file I/O is exposed as plain functions (load/save/snapshot + despawn);
        // the editor's multi-scene tab manager drives them (see `editor::scene_tabs`).
    }
}

/// Despawn all loaded scene content (`SceneEntity`), sparing editor infrastructure
/// (`EditorEntity` — the persistent viewport camera). Used before loading another scene
/// so the new one replaces the current one rather than stacking on top.
#[cfg(feature = "editor")]
pub fn despawn_scene_content(world: &mut World) {
    use crate::scene_manager::{EditorEntity, SceneEntity};

    let to_despawn: Vec<Entity> = world
        .query_filtered::<Entity, (With<SceneEntity>, Without<EditorEntity>)>()
        .iter(world)
        .collect();
    for entity in to_despawn {
        if let Ok(e) = world.get_entity_mut(entity) {
            e.despawn();
        }
    }
}
