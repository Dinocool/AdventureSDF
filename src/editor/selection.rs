//! Unified editor selection. The Inspector inspects whichever of {scene entity, asset
//! file} was selected last — the two are mutually exclusive. Scene-node selection is
//! still owned by [`SdfSelection`] (the viewport gizmo/picking source of truth); this
//! layer mirrors it and adds asset (file-path) selection on top.

use std::path::{Path, PathBuf};

use bevy::prelude::*;

use crate::sdf_render::SdfSelection;

/// What the Inspector is currently inspecting.
#[derive(Resource, Default, Debug, Clone, PartialEq, Eq)]
pub enum EditorSelection {
    #[default]
    None,
    /// A scene entity (mirrors [`SdfSelection`]).
    Entity(Entity),
    /// An asset file, path relative to the working dir (e.g. `assets/materials/x.material.ron`).
    Asset(PathBuf),
}

impl EditorSelection {
    /// Select an asset, clearing any entity selection (mutual exclusion). Returns true
    /// if the selection changed.
    pub fn select_asset(&mut self, path: impl Into<PathBuf>) -> bool {
        let next = EditorSelection::Asset(path.into());
        let changed = *self != next;
        *self = next;
        changed
    }

    /// The selected asset path, if an asset is currently selected.
    pub fn asset(&self) -> Option<&Path> {
        match self {
            EditorSelection::Asset(p) => Some(p),
            _ => None,
        }
    }
}

/// Keep [`EditorSelection`] in step with the entity-side [`SdfSelection`]. Uses the
/// *previous* `SdfSelection.entity` to detect a genuinely new scene-node pick, so:
/// - A new entity pick (viewport/hierarchy) supersedes any selection — even an asset
///   one (clicking a node deselects an asset).
/// - While an asset stays selected, `SdfSelection` is held cleared so the viewport
///   gizmo doesn't keep targeting an entity that's no longer inspected. (The asset
///   selection itself is set by the Assets panel via `EditorSelection::select_asset`.)
pub fn sync_selection(
    mut selection: ResMut<EditorSelection>,
    mut sdf: ResMut<SdfSelection>,
    mut prev_entity: Local<Option<Entity>>,
) {
    // Did the entity-side selection change to a new entity since last frame? That's a
    // real pick and wins over whatever we currently show.
    let picked_new = sdf.entity.is_some() && sdf.entity != *prev_entity;

    if picked_new {
        *selection = EditorSelection::Entity(sdf.entity.unwrap());
    } else {
        match *selection {
            // Asset selected → keep the entity side cleared.
            EditorSelection::Asset(_) => sdf.entity = None,
            // Entity selected → follow deselection (entity despawned / cleared).
            EditorSelection::Entity(_) if sdf.entity.is_none() => {
                *selection = EditorSelection::None;
            }
            _ => {}
        }
    }

    *prev_entity = sdf.entity;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_asset_replaces_and_reports_change() {
        let mut s = EditorSelection::None;
        assert!(s.select_asset("assets/a.png"));
        assert_eq!(s.asset(), Some(Path::new("assets/a.png")));
        // Selecting the same asset again is not a change.
        assert!(!s.select_asset("assets/a.png"));
        // A different asset is a change.
        assert!(s.select_asset("assets/b.png"));
    }

    #[test]
    fn asset_accessor_only_for_asset_variant() {
        assert_eq!(EditorSelection::None.asset(), None);
        assert_eq!(EditorSelection::Entity(Entity::PLACEHOLDER).asset(), None);
        assert_eq!(
            EditorSelection::Asset(PathBuf::from("x")).asset(),
            Some(Path::new("x"))
        );
    }
}
