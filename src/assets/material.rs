//! [`MaterialAsset`] — a Godot-style material *resource*: an editable, savable disk
//! asset (RON) that compiles into the GPU-facing [`MaterialRegistry`].
//!
//! Textures are referenced **by path** ([`TexRef`] = slug/dir), resolved to a GPU
//! texture-array layer at compile time (see `compile.rs`). This indirection is the
//! seam a future virtual-texture system slots into — materials never hold raw layer
//! indices.

use bevy::asset::{AssetLoader, LoadContext, io::Reader};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::sdf_render::edits::MATERIAL_TEX_MAPS;

/// Reference to a texture variant on disk: `assets/textures/<slug>/<dir>`. Resolved
/// to a GPU array layer on demand by the material compile step.
#[derive(Reflect, Serialize, Deserialize, Clone, PartialEq, Eq, Debug, Default)]
pub struct TexRef {
    pub slug: String,
    pub dir: String,
}

/// An authored material resource. The editable source of truth on disk; the compile
/// step flattens it (+ resolved texture layers) into a `MaterialDef` row in the
/// `MaterialRegistry` that the GPU table mirrors.
///
/// `base_color` is stored as `[f32; 4]` (linear RGBA) rather than `Color` so the RON
/// is stable across engine versions and trivially serde-able.
#[derive(Asset, Reflect, Serialize, Deserialize, Clone, Debug)]
pub struct MaterialAsset {
    /// Linear RGBA tint multiplied into the sampled diffuse.
    pub base_color: [f32; 4],
    /// Shading-time seam cross-fade width (world units). See `MaterialDef`.
    pub blend_softness: f32,
    /// Scalar metallic/roughness fallbacks, used when no MRA texture is set (`maps[2]`
    /// is `None`). `#[serde(default)]` so older RON without these fields still loads
    /// (defaulting to a fully-rough dielectric, the prior behaviour).
    #[serde(default)]
    pub metallic: f32,
    #[serde(default = "default_roughness")]
    pub roughness: f32,
    /// One texture reference per PBR map (diffuse, normal, mra, height, edge);
    /// `None` = no texture for that map.
    pub maps: [Option<TexRef>; MATERIAL_TEX_MAPS],
}

/// serde default for `roughness` (1.0 = fully diffuse). A bare `Default` would give 0.0
/// (a mirror), which is the wrong fallback for un-annotated legacy material RON.
fn default_roughness() -> f32 {
    1.0
}

impl Default for MaterialAsset {
    fn default() -> Self {
        Self {
            base_color: [0.8, 0.8, 0.8, 1.0],
            blend_softness: 0.0,
            metallic: 0.0,
            roughness: 1.0,
            maps: std::array::from_fn(|_| None),
        }
    }
}

impl MaterialAsset {
    /// `base_color` as a Bevy linear `Color` (for the GPU table / UI).
    pub fn color(&self) -> Color {
        let [r, g, b, a] = self.base_color;
        Color::linear_rgba(r, g, b, a)
    }

    /// Set `base_color` from a Bevy `Color`.
    pub fn set_color(&mut self, color: Color) {
        let l = color.to_linear();
        self.base_color = [l.red, l.green, l.blue, l.alpha];
    }
}

impl super::Asset for MaterialAsset {
    const EXTENSION: &'static str = "material.ron";
}

/// Loads [`MaterialAsset`] from a RON resource file. `MaterialAsset` is a concrete
/// serde type, so this is plain `ron` deserialization — no reflection registry
/// needed at load time (unlike the type-erased scene components in `soul_scene`).
#[derive(Default, bevy::reflect::TypePath)]
pub struct MaterialAssetLoader;

/// Errors surfaced while loading a material resource.
#[derive(Debug)]
pub enum MaterialLoadError {
    Io(std::io::Error),
    Ron(ron::error::SpannedError),
}

impl std::fmt::Display for MaterialLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MaterialLoadError::Io(e) => write!(f, "material io: {e}"),
            MaterialLoadError::Ron(e) => write!(f, "material ron: {e}"),
        }
    }
}

impl std::error::Error for MaterialLoadError {}

impl From<std::io::Error> for MaterialLoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<ron::error::SpannedError> for MaterialLoadError {
    fn from(e: ron::error::SpannedError) -> Self {
        Self::Ron(e)
    }
}

impl AssetLoader for MaterialAssetLoader {
    type Asset = MaterialAsset;
    type Settings = ();
    type Error = MaterialLoadError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _ctx: &mut LoadContext<'_>,
    ) -> Result<MaterialAsset, MaterialLoadError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let asset = ron::de::from_bytes::<MaterialAsset>(&bytes)?;
        Ok(asset)
    }

    fn extensions(&self) -> &[&str] {
        // Bevy matches on the final extension; the `.material.ron` convention still
        // ends in `ron`, so we claim `ron` and rely on the materials/ directory.
        &["material.ron", "ron"]
    }
}
