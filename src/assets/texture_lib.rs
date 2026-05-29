//! Demand-driven texture library + the material-asset id table.
//!
//! Replaces the old hardcoded `LIBRARY_SLUGS` / `build_texture_library`. Texture
//! array layers are assigned **on demand**: as the material compile step resolves
//! the `TexRef`s of materials actually used by the scene, each unique `(slug, dir)`
//! gets the next free GPU array layer. Layers are **grow-only** up to
//! [`MAX_TEXTURE_LAYERS`] so indices stay stable as materials change.
//!
//! The set of currently-needed `(slug, dir)` pairs ([`MaterialTextureLibrary`]) is
//! also the residency input a future virtual-texture system will consume.

use std::collections::HashMap;

use bevy::prelude::*;

use super::MaterialAsset;
use crate::sdf_render::textures::LibraryVariant;

/// Physical texture-array layer cap. Demand-driven assignment fills slots up to
/// this; the arrays are created once at this size (no recreation). A placeholder
/// for the physical layer set that virtual texturing will later replace with a
/// page cache + indirection table.
pub const MAX_TEXTURE_LAYERS: u32 = 64;

/// The demand-driven texture library: a grow-only map from a texture variant
/// (`slug`, `dir`) to its GPU array layer, plus the ordered variant list (index =
/// layer) that feeds BC7 streaming.
#[derive(Resource, Default)]
pub struct MaterialTextureLibrary {
    layer_of: HashMap<(String, String), u32>,
    /// Index = layer. Cloned into the render world to drive streaming.
    pub variants: Vec<LibraryVariant>,
    /// Set when a new layer is assigned, so the render world re-extracts + streams.
    pub dirty: bool,
}

impl MaterialTextureLibrary {
    /// Resolve a texture variant to its GPU layer, assigning the next free layer on
    /// first use. Returns `u32::MAX` if the cap is reached (renders as fallback).
    pub fn resolve_layer(&mut self, slug: &str, dir: &str) -> u32 {
        let key = (slug.to_string(), dir.to_string());
        if let Some(&layer) = self.layer_of.get(&key) {
            return layer;
        }
        let layer = self.variants.len() as u32;
        if layer >= MAX_TEXTURE_LAYERS {
            warn!(
                "texture library at MAX_TEXTURE_LAYERS ({MAX_TEXTURE_LAYERS}); \
                 '{slug}/{dir}' falls back"
            );
            return u32::MAX;
        }
        self.variants.push(LibraryVariant {
            slug: slug.to_string(),
            dir: dir.to_string(),
            display_name: format!("{slug} {dir}"),
        });
        self.layer_of.insert(key, layer);
        self.dirty = true;
        layer
    }
}

/// Maps a [`MaterialAsset`] to a stable `registry_id` (= its row in the GPU material
/// table / `MaterialRegistry::defs`). Grow-only: ids are never reindexed, since the
/// id IS the GPU row and per-volume `SdfMaterial { registry_id }` references it.
/// Id 0 is the default fallback (no handle).
#[derive(Resource, Default)]
pub struct MaterialAssetTable {
    /// `handles[id]` is the asset for `registry_id == id`. Index 0 = fallback (a
    /// default/weak handle that may be unset).
    pub handles: Vec<Handle<MaterialAsset>>,
    id_of: HashMap<AssetId<MaterialAsset>, u32>,
}

impl MaterialAssetTable {
    /// Ensure id 0 exists as the fallback slot (a default handle). Call once at init.
    pub fn ensure_fallback(&mut self) {
        if self.handles.is_empty() {
            self.handles.push(Handle::default());
        }
    }

    /// Register a material handle, returning its stable `registry_id`. Idempotent:
    /// the same handle always maps to the same id.
    pub fn register(&mut self, handle: Handle<MaterialAsset>) -> u32 {
        if let Some(&id) = self.id_of.get(&handle.id()) {
            return id;
        }
        self.ensure_fallback();
        let id = self.handles.len() as u32;
        self.id_of.insert(handle.id(), id);
        self.handles.push(handle);
        id
    }

    /// The `registry_id` for an already-registered handle, if any.
    pub fn id_for(&self, handle: &Handle<MaterialAsset>) -> Option<u32> {
        self.id_of.get(&handle.id()).copied()
    }
}
