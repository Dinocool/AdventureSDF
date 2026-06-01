//! Image-file thumbnail backend: loads raster files via the `AssetServer` and registers
//! them with egui once the GPU image exists. Keyed by asset path; the panel only reads
//! the resulting `egui::TextureId`.

use std::path::{Path, PathBuf};

use bevy::image::Image;
use bevy::prelude::*;
use bevy_egui::{EguiTextureHandle, EguiUserTextures, egui};

use super::super::{Thumbnail, ThumbnailProvider};

/// Cache of image-file thumbnails: handle (keeps the image resident) + egui id once
/// the GPU image is ready.
#[derive(Resource, Default)]
pub struct ImageThumbnailCache {
    entries: std::collections::HashMap<PathBuf, ImageSlot>,
}

struct ImageSlot {
    handle: Handle<Image>,
    tex_id: Option<egui::TextureId>,
}

/// Provider for raster image files.
pub struct ImageThumbnailProvider;

impl ThumbnailProvider for ImageThumbnailProvider {
    fn matches(&self, path: &Path) -> bool {
        crate::editor::fs_util::is_image_file(path)
    }

    fn thumbnail(&self, world: &mut World, path: &Path) -> Thumbnail {
        match ensure_image_texture(world, path) {
            ImageTexture::Ready { tex_id, .. } => Thumbnail::Texture(tex_id),
            ImageTexture::Loading => Thumbnail::Pending,
            ImageTexture::Invalid => Thumbnail::Icon("\u{1F5BC}"),
        }
    }
}

/// State of a cached image thumbnail/preview.
pub enum ImageTexture {
    /// Loaded + registered with egui.
    Ready {
        tex_id: egui::TextureId,
        handle: Handle<Image>,
    },
    /// Still loading.
    Loading,
    /// Not an asset under `assets/` (bad path).
    Invalid,
}

/// Ensure an image at `path` (working-dir-relative, under `assets/`) is loaded and
/// registered with egui; cache by path. Reused by the grid provider and the asset
/// inspector's texture preview. Idempotent.
pub fn ensure_image_texture(world: &mut World, path: &Path) -> ImageTexture {
    let Some(asset_path) = crate::editor::fs_util::relative_to_assets(path) else {
        return ImageTexture::Invalid;
    };

    let key = path.to_path_buf();
    if !world.resource::<ImageThumbnailCache>().entries.contains_key(&key) {
        let handle = world.resource::<AssetServer>().load::<Image>(asset_path);
        world
            .resource_mut::<ImageThumbnailCache>()
            .entries
            .insert(key.clone(), ImageSlot { handle, tex_id: None });
    }

    let slot_handle = world.resource::<ImageThumbnailCache>().entries[&key].handle.clone();
    if let Some(id) = world.resource::<ImageThumbnailCache>().entries[&key].tex_id {
        return ImageTexture::Ready { tex_id: id, handle: slot_handle };
    }

    if world
        .resource::<AssetServer>()
        .is_loaded_with_dependencies(&slot_handle)
    {
        let id = world
            .resource_mut::<EguiUserTextures>()
            .add_image(EguiTextureHandle::Strong(slot_handle.clone()));
        world
            .resource_mut::<ImageThumbnailCache>()
            .entries
            .get_mut(&key)
            .unwrap()
            .tex_id = Some(id);
        return ImageTexture::Ready { tex_id: id, handle: slot_handle };
    }
    ImageTexture::Loading
}
