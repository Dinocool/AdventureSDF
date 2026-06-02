//! Thumbnail backends for the assets browser. Two independent caches, both keyed by
//! asset path and filled by Bevy systems (the panel only reads the resulting
//! `egui::TextureId`):
//! - [`image`] — loads image files via the `AssetServer` and registers them with egui.
//! - [`material`] — captures a lit PBR sphere per `*.material.ron` (offscreen render +
//!   GPU readback + disk cache).

use bevy::prelude::*;

pub mod image;
pub mod material;
pub mod scene;

pub use image::{ImageThumbnailCache, ImageThumbnailProvider, ImageTexture, ensure_image_texture};
pub use material::{
    MaterialThumbnailCache, MaterialThumbnailProvider, PbrTextureThumbnailProvider,
};
pub(crate) use material::standard_from_material;
pub use scene::{PendingSceneThumbnail, SceneThumbnailProvider};

/// Plugin: registers the thumbnail caches + the offscreen material-sphere render rig.
/// Image thumbnails need no systems (loaded on demand in `ensure_image_texture`); the
/// material backend registers its own capture systems via [`material::register`].
pub struct ThumbnailRenderPlugin;

impl Plugin for ThumbnailRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ImageThumbnailCache>();
        material::register(app);
        scene::register(app);
    }
}
