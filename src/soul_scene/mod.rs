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
pub use load::{SceneLoadError, load_scene};
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

        // Drain editor File-menu requests into scene I/O. The request resource and
        // its menu live behind the editor feature.
        #[cfg(feature = "editor")]
        app.add_systems(Update, drain_editor_scene_requests);
    }
}

/// Bridge: turn editor menu/keybind requests into save/load against the world,
/// using the app's `AppTypeRegistry`. Exclusive system (reflection + spawn need
/// `&mut World`).
#[cfg(feature = "editor")]
fn drain_editor_scene_requests(world: &mut World) {
    use crate::editor::menu_bar::{CurrentScenePath, EditorRequests};

    let (do_save, open_path) = {
        let mut req = world.resource_mut::<EditorRequests>();
        let save = req.save || req.save_as.is_some();
        let save_as = req.save_as.take();
        let open = req.open.take();
        let new_scene = std::mem::take(&mut req.new_scene);
        req.save = false;
        if let Some(p) = save_as.clone() {
            world.resource_mut::<CurrentScenePath>().0 = p;
        }
        let _ = new_scene; // New-scene handling (despawn + reset) deferred.
        (save, open)
    };

    let registry = world.resource::<AppTypeRegistry>().clone();

    if do_save {
        let path = world.resource::<CurrentScenePath>().0.clone();
        let registry = registry.read();
        match save_scene(world, &path, &registry) {
            Ok(()) => info!("saved scene to {}", path.display()),
            Err(e) => error!("scene save failed: {e}"),
        }
    }

    if let Some(path) = open_path {
        let registry = registry.read();
        match load_scene(world, &path, &registry) {
            Ok(roots) => info!(
                "loaded {} root entities from {}",
                roots.len(),
                path.display()
            ),
            Err(e) => error!("scene load failed: {e}"),
        }
    }
}
