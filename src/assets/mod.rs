//! soul-engine asset framework: Godot-style **resources** — editable, savable disk
//! assets. In code the abstraction is the [`Asset`] trait; on disk they are RON
//! resource files. Built on top of `bevy::asset` (load + hot-reload via
//! `Handle`/`AssetServer`/`AssetLoader`) plus a custom [`save`] layer (bevy has no
//! save path).
//!
//! This pass implements **materials** (and the demand-driven texture library they
//! pull from). Fonts / 3D models / sounds slot in later as more [`Asset`] impls +
//! loaders — the trait and plugin are written to extend.

use std::path::Path;

use bevy::prelude::*;

pub mod compile;
pub mod material;
pub mod save;
pub mod texture_lib;

#[cfg(test)]
mod tests;

pub use material::{MaterialAsset, TexRef};
pub use texture_lib::{MAX_TEXTURE_LAYERS, MaterialAssetTable, MaterialTextureLibrary};

/// A soul-engine resource: a concrete, serde + reflect type that loads via a
/// `bevy::asset` loader and saves back to disk as RON. Implementors get [`save`]
/// for free.
pub trait Asset: bevy::asset::Asset + serde::Serialize {
    /// On-disk extension for this resource (e.g. `"material.ron"`).
    const EXTENSION: &'static str;

    /// Write this resource to `path` as pretty RON.
    fn save(&self, path: &Path) -> Result<(), save::AssetSaveError>
    where
        Self: Sized,
    {
        save::save_ron(self, path)
    }
}

/// Registers asset types, loaders, the demand-driven texture library, and the
/// material compile systems.
pub struct AssetsPlugin;

impl Plugin for AssetsPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<MaterialAsset>()
            .register_asset_loader(material::MaterialAssetLoader)
            .register_type::<MaterialAsset>()
            .register_type::<TexRef>()
            .init_resource::<MaterialTextureLibrary>()
            .init_resource::<MaterialAssetTable>();

        compile::register(app);
    }
}
