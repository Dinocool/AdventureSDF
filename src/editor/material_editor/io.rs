//! Save/load helpers for material + PBR-texture assets: resolve asset paths to handles
//! and write edited assets back to their `.ron` files.

use bevy::prelude::*;

use crate::assets::MaterialAsset;

/// Save the material behind `handle` to `assets/materials/<name>.material.ron`. Shared
/// by the Resources panel and the asset inspector.
pub fn save_material(world: &World, handle: &Handle<MaterialAsset>, name: &str) {
    let path = std::path::PathBuf::from(format!("assets/materials/{name}.material.ron"));
    if let Some(asset) = world.resource::<Assets<MaterialAsset>>().get(handle) {
        match crate::assets::Asset::save(asset, &path) {
            Ok(()) => info!("saved material to {}", path.display()),
            Err(e) => error!("material save failed: {e}"),
        }
    }
}

/// Resolve a `.material.ron` asset path to its loaded `MaterialAsset` handle. Used by
/// the asset/entity inspectors. Returns the table's handle when the path is already a
/// registered material (so edits drive the same live-recompiled asset), else loads it.
pub fn handle_for_path(world: &World, path: &std::path::Path) -> Option<Handle<MaterialAsset>> {
    let rel = crate::editor::fs_util::relative_to_assets(path)?;
    let server = world.resource::<AssetServer>();
    // Prefer an already-loaded handle (the demo scene loads its materials at startup);
    // fall back to a fresh load so any `.material.ron` is inspectable.
    server
        .get_handle::<MaterialAsset>(rel.clone())
        .or_else(|| Some(server.load::<MaterialAsset>(rel)))
}


/// Resolve a `.pbrtex.ron` path to its loaded handle (or load it). For the inspector.
pub fn pbrtex_handle_for_path(
    world: &mut World,
    path: &std::path::Path,
) -> Option<Handle<crate::assets::PbrTextureAsset>> {
    let rel = crate::editor::fs_util::relative_to_assets(path)?;
    // Cache a strong handle so the bundle stays loaded (a fresh `load` each frame would
    // never finish loading → the inspector shows "still loading…" forever).
    let server = world.resource::<AssetServer>().clone();
    Some(
        world
            .resource_mut::<crate::assets::PbrTextureHandles>()
            .ensure(&rel, &server),
    )
}

/// Save a `PbrTextureAsset` bundle to `assets/<rel>` (rel relative to assets root).
pub fn save_pbr_texture(
    world: &World,
    handle: &Handle<crate::assets::PbrTextureAsset>,
    rel_path: &std::path::Path,
) {
    let path = std::path::Path::new("assets").join(rel_path);
    if let Some(asset) = world.resource::<Assets<crate::assets::PbrTextureAsset>>().get(handle) {
        match crate::assets::Asset::save(asset, &path) {
            Ok(()) => info!("saved pbr texture to {}", path.display()),
            Err(e) => error!("pbr texture save failed: {e}"),
        }
    }
}
