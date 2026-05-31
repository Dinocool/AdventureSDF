//! [`PbrTextureAsset`] — a bundled PBR texture *resource*: per-role source image files
//! (diffuse / normal / metallic / roughness / ao / height / edge), each an arbitrary
//! image path under `assets/`. A [`MaterialAsset`](super::MaterialAsset) references one
//! bundle and may override individual roles; the compile step merges them into a
//! [`MapSet`] that the texture library encodes into one GPU array layer.

use std::path::{Path, PathBuf};

use bevy::asset::{AssetLoader, LoadContext, io::Reader};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// Per-role source files for one PBR texture, each relative to `assets/`. Any image
/// format the importer can decode (png/jpg/jpeg/bmp/tga). `None` = neutral fallback for
/// that role at encode time.
#[derive(Asset, Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct PbrTextureAsset {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diffuse: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metallic: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roughness: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ao: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge: Option<PathBuf>,
}

impl PbrTextureAsset {
    /// Whether every role is empty (so a material referencing this has no textures).
    pub fn is_empty(&self) -> bool {
        *self == PbrTextureAsset::default()
    }

    /// Merge `over` onto `self`: any role set in `over` wins, else keep `self`'s. Used
    /// to apply a material's per-role overrides on top of its bundle.
    pub fn merge(&self, over: &PbrTextureAsset) -> PbrTextureAsset {
        let pick = |b: &Option<PathBuf>, o: &Option<PathBuf>| o.clone().or_else(|| b.clone());
        PbrTextureAsset {
            diffuse: pick(&self.diffuse, &over.diffuse),
            normal: pick(&self.normal, &over.normal),
            metallic: pick(&self.metallic, &over.metallic),
            roughness: pick(&self.roughness, &over.roughness),
            ao: pick(&self.ao, &over.ao),
            height: pick(&self.height, &over.height),
            edge: pick(&self.edge, &over.edge),
        }
    }

    /// The resolved, encode-ready [`MapSet`] (same role files; the distinct type marks
    /// "this is the effective, override-merged set" handed to the texture library).
    pub fn to_map_set(&self) -> MapSet {
        MapSet {
            diffuse: self.diffuse.clone(),
            normal: self.normal.clone(),
            metallic: self.metallic.clone(),
            roughness: self.roughness.clone(),
            ao: self.ao.clone(),
            height: self.height.clone(),
            edge: self.edge.clone(),
        }
    }
}

/// Holds strong [`Handle`]s to every `.pbrtex.ron` bundle referenced by a material,
/// keyed by its `assets/`-relative path. Without this, a handle from `AssetServer::load`
/// is dropped each frame and the bundle never stays loaded — so `Assets::get` is forever
/// `None`. Populated on demand by the compile step + the editor.
#[derive(Resource, Default)]
pub struct PbrTextureHandles {
    by_path: std::collections::HashMap<PathBuf, Handle<PbrTextureAsset>>,
}

impl PbrTextureHandles {
    /// Strong handle for `rel_path` (relative to `assets/`), loading + caching on first
    /// use so it stays resident.
    pub fn ensure(&mut self, rel_path: &Path, server: &AssetServer) -> Handle<PbrTextureAsset> {
        if let Some(h) = self.by_path.get(rel_path) {
            return h.clone();
        }
        let h = server.load::<PbrTextureAsset>(rel_path.to_path_buf());
        self.by_path.insert(rel_path.to_path_buf(), h.clone());
        h
    }

    /// Already-cached handle for `rel_path`, if any (no load).
    pub fn get(&self, rel_path: &Path) -> Option<Handle<PbrTextureAsset>> {
        self.by_path.get(rel_path).cloned()
    }
}

/// A resolved set of role files for one texture layer — the override-merged result the
/// texture library keys + encodes. `Hash`/`Eq` give it a stable identity, so two
/// materials with the same effective maps share a GPU layer.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct MapSet {
    pub diffuse: Option<PathBuf>,
    pub normal: Option<PathBuf>,
    pub metallic: Option<PathBuf>,
    pub roughness: Option<PathBuf>,
    pub ao: Option<PathBuf>,
    pub height: Option<PathBuf>,
    pub edge: Option<PathBuf>,
}

impl MapSet {
    /// No role files set → nothing to encode (material renders untextured).
    pub fn is_empty(&self) -> bool {
        *self == MapSet::default()
    }

    /// Absolute (working-dir) path of a role file, joining the `assets/` root.
    pub fn role_abs(role: &Option<PathBuf>) -> Option<PathBuf> {
        role.as_ref().map(|p| Path::new("assets").join(p))
    }

    /// A short display label (the diffuse stem, else the first set role's stem).
    pub fn label(&self) -> String {
        let first = [
            &self.diffuse,
            &self.normal,
            &self.metallic,
            &self.roughness,
            &self.ao,
            &self.height,
            &self.edge,
        ]
        .into_iter()
        .flatten()
        .next();
        first
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "(empty)".to_string())
    }
}

impl super::Asset for PbrTextureAsset {
    const EXTENSION: &'static str = "pbrtex.ron";
}

/// Loads [`PbrTextureAsset`] from a RON resource file. Plain `ron` deserialization, like
/// [`MaterialAssetLoader`](super::material::MaterialAssetLoader).
#[derive(Default, bevy::reflect::TypePath)]
pub struct PbrTextureAssetLoader;

/// Errors surfaced while loading a PBR-texture resource.
#[derive(Debug)]
pub enum PbrTextureLoadError {
    Io(std::io::Error),
    Ron(ron::error::SpannedError),
}

impl std::fmt::Display for PbrTextureLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PbrTextureLoadError::Io(e) => write!(f, "pbrtex io: {e}"),
            PbrTextureLoadError::Ron(e) => write!(f, "pbrtex ron: {e}"),
        }
    }
}

impl std::error::Error for PbrTextureLoadError {}

impl From<std::io::Error> for PbrTextureLoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<ron::error::SpannedError> for PbrTextureLoadError {
    fn from(e: ron::error::SpannedError) -> Self {
        Self::Ron(e)
    }
}

impl AssetLoader for PbrTextureAssetLoader {
    type Asset = PbrTextureAsset;
    type Settings = ();
    type Error = PbrTextureLoadError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _ctx: &mut LoadContext<'_>,
    ) -> Result<PbrTextureAsset, PbrTextureLoadError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let asset = ron::de::from_bytes::<PbrTextureAsset>(&bytes)?;
        Ok(asset)
    }

    fn extensions(&self) -> &[&str] {
        &["pbrtex.ron", "ron"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_override_wins() {
        let base = PbrTextureAsset {
            diffuse: Some("a/diffuse.png".into()),
            normal: Some("a/normal.png".into()),
            ..Default::default()
        };
        let over = PbrTextureAsset {
            normal: Some("b/normal.png".into()),
            height: Some("b/height.png".into()),
            ..Default::default()
        };
        let m = base.merge(&over);
        assert_eq!(m.diffuse, Some("a/diffuse.png".into())); // from base
        assert_eq!(m.normal, Some("b/normal.png".into())); // override wins
        assert_eq!(m.height, Some("b/height.png".into())); // from override
    }

    #[test]
    fn map_set_key_stable_and_distinct() {
        let a = PbrTextureAsset {
            diffuse: Some("x/diffuse.png".into()),
            ..Default::default()
        };
        let b = a.clone();
        assert_eq!(a.to_map_set(), b.to_map_set());
        let c = PbrTextureAsset {
            diffuse: Some("y/diffuse.png".into()),
            ..Default::default()
        };
        assert_ne!(a.to_map_set(), c.to_map_set());
    }

    #[test]
    fn pbrtex_ron_round_trips() {
        let t = PbrTextureAsset {
            diffuse: Some("t/1/diffuse.png".into()),
            height: Some("t/1/height.png".into()),
            ..Default::default()
        };
        let text = ron::ser::to_string(&t).unwrap();
        let back: PbrTextureAsset = ron::from_str(&text).unwrap();
        assert_eq!(t, back);
    }
}
