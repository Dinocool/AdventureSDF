//! Save layer for asset resources. `bevy::asset` only loads; this writes assets
//! back to disk as RON. Asset types are concrete serde types, so this is plain
//! `ron` pretty-serialization (the type-erased `ReflectSerializer` dance in
//! `soul_scene` is only needed there because scene components are `dyn Reflect`).

use std::path::Path;

use serde::Serialize;

/// Errors raised while saving an asset resource.
#[derive(Debug)]
pub enum AssetSaveError {
    Io(std::io::Error),
    Ron(ron::Error),
}

impl std::fmt::Display for AssetSaveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssetSaveError::Io(e) => write!(f, "asset save io: {e}"),
            AssetSaveError::Ron(e) => write!(f, "asset save ron: {e}"),
        }
    }
}

impl std::error::Error for AssetSaveError {}

/// Serialize `value` to a pretty RON resource file at `path`, creating parent dirs.
pub fn save_ron<T: Serialize>(value: &T, path: &Path) -> Result<(), AssetSaveError> {
    let cfg = ron::ser::PrettyConfig::new()
        .struct_names(true)
        .indentor("  ".to_string());
    let text = ron::ser::to_string_pretty(value, cfg).map_err(AssetSaveError::Ron)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(AssetSaveError::Io)?;
    }
    std::fs::write(path, text).map_err(AssetSaveError::Io)?;
    Ok(())
}
