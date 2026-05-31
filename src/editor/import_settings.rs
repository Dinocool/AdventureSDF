//! Per-texture import settings, persisted as a Godot-style sidecar `<file>.import.ron`
//! next to the image. Editor-facing this pass: the settings drive how the editor loads
//! and samples the image for preview. Wiring them into the SDF BC7 import pipeline (a
//! fixed 1024² path) is a deliberate follow-up.

use std::path::{Path, PathBuf};

use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// Texture filtering applied when sampling the image.
#[derive(Serialize, Deserialize, Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ImageFilter {
    #[default]
    Linear,
    Nearest,
}

/// Whether the image's color data is sRGB-encoded (color maps) or linear (data maps
/// like normals/roughness).
#[derive(Serialize, Deserialize, Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ColorSpace {
    #[default]
    Srgb,
    Linear,
}

/// Address mode at UV edges.
#[derive(Serialize, Deserialize, Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum WrapMode {
    #[default]
    Repeat,
    Clamp,
}

/// Editable import settings for one texture asset.
#[derive(Serialize, Deserialize, Reflect, Clone, PartialEq, Debug, Default)]
pub struct TextureImportSettings {
    pub filter: ImageFilter,
    pub color_space: ColorSpace,
    pub wrap: WrapMode,
}

impl TextureImportSettings {
    /// Sidecar path for an image: `<image>.import.ron`.
    pub fn sidecar_path(image: &Path) -> PathBuf {
        let mut s = image.as_os_str().to_os_string();
        s.push(".import.ron");
        PathBuf::from(s)
    }

    /// Load settings for `image` from its sidecar, or defaults if absent/unparseable.
    pub fn load_for(image: &Path) -> Self {
        let sidecar = Self::sidecar_path(image);
        let Ok(text) = std::fs::read_to_string(&sidecar) else {
            return Self::default();
        };
        ron::from_str(&text).unwrap_or_else(|e| {
            warn!("import settings: cannot parse {}: {e}", sidecar.display());
            Self::default()
        })
    }

    /// Write settings to `image`'s sidecar as pretty RON (reuses the asset save helper,
    /// which creates parent dirs).
    pub fn save_for(&self, image: &Path) -> Result<(), crate::assets::save::AssetSaveError> {
        crate::assets::save::save_ron(self, &Self::sidecar_path(image))
    }

    /// Translate to a Bevy `ImageSampler` so the editor can apply these settings to a
    /// preview image.
    pub fn to_sampler(&self) -> ImageSampler {
        let filter = match self.filter {
            ImageFilter::Linear => ImageFilterMode::Linear,
            ImageFilter::Nearest => ImageFilterMode::Nearest,
        };
        let address = match self.wrap {
            WrapMode::Repeat => ImageAddressMode::Repeat,
            WrapMode::Clamp => ImageAddressMode::ClampToEdge,
        };
        ImageSampler::Descriptor(ImageSamplerDescriptor {
            mag_filter: filter,
            min_filter: filter,
            mipmap_filter: filter,
            address_mode_u: address,
            address_mode_v: address,
            address_mode_w: address,
            ..default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_path_appends_import_ron() {
        assert_eq!(
            TextureImportSettings::sidecar_path(Path::new("assets/t/diffuse.png")),
            PathBuf::from("assets/t/diffuse.png.import.ron")
        );
    }

    #[test]
    fn settings_round_trip_through_ron() {
        let s = TextureImportSettings {
            filter: ImageFilter::Nearest,
            color_space: ColorSpace::Linear,
            wrap: WrapMode::Clamp,
        };
        let text = ron::ser::to_string(&s).unwrap();
        let back: TextureImportSettings = ron::from_str(&text).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn missing_sidecar_yields_defaults() {
        let s = TextureImportSettings::load_for(Path::new("does/not/exist.png"));
        assert_eq!(s, TextureImportSettings::default());
    }
}
