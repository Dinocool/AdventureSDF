//! On-disk schema for soul-engine `.scene` files (RON).
//!
//! A scene is a flat list of records keyed by a stable [`LocalId`]. Two record
//! kinds: a plain `Entity` (its components), or an `Instance` of another `.scene`
//! with per-sub-entity override deltas (Godot-style nested scenes).
//!
//! Component values are stored as **independently reflection-serialized RON
//! strings**, keyed by Bevy type path. This keeps the container plain `serde`
//! (no `TypeRegistry` threading through `Deserialize`); the load/save layer owns
//! the reflection round-trip per component (see `load.rs` / `save.rs`).

use std::collections::BTreeMap;
use std::path::PathBuf;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use super::ReflectSerializeSkip;

/// Stable per-entity id within a scene file. Survives re-saves (never reindexed)
/// and is the key override deltas target onto an instanced scene's sub-entities.
#[derive(Component, Reflect, Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[reflect(Component, SerializeSkip)]
pub struct LocalId(pub u64);

/// A component value, reflection-serialized to RON, keyed by its Bevy type path
/// (e.g. `"adventure::sdf_render::edits::SdfPrimitive"`).
pub type ComponentMap = BTreeMap<String, String>;

/// One record in a `.scene` file.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum SceneRecord {
    /// A locally-authored entity and its components.
    Entity {
        id: LocalId,
        /// `LocalId` of this entity's parent node, if any. The on-disk hierarchy link
        /// (Bevy's `ChildOf` holds a raw `Entity`, which isn't stable across re-save,
        /// so we serialize the stable parent id instead). `#[serde(default)]` keeps
        /// pre-hierarchy `.scene` files loadable.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent: Option<u64>,
        components: ComponentMap,
    },
    /// An instance of another `.scene`, with per-sub-entity component overrides.
    /// `overrides` is keyed by the *source* scene's `LocalId`; each value is the
    /// subset of components whose values differ from the source.
    Instance {
        id: LocalId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent: Option<u64>,
        source: PathBuf,
        overrides: BTreeMap<u64, ComponentMap>,
    },
}

impl SceneRecord {
    pub fn id(&self) -> LocalId {
        match self {
            SceneRecord::Entity { id, .. } | SceneRecord::Instance { id, .. } => *id,
        }
    }

    pub fn parent(&self) -> Option<u64> {
        match self {
            SceneRecord::Entity { parent, .. } | SceneRecord::Instance { parent, .. } => *parent,
        }
    }
}

/// The editor camera pose saved alongside a scene, so reopening restores the view that was
/// framed when it was saved. Plain data (no engine coupling); the editor maps it to/from its
/// orbit camera. `#[serde(default)]` on the field keeps camera-less `.scene` files loadable.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq)]
pub struct EditorCamera {
    pub target: [f32; 3],
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
}

/// A parsed `.scene` file: a monotonic id counter (so re-save never reuses ids)
/// plus the records.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct SceneFile {
    /// Next free [`LocalId`] value. Persisted so ids stay stable across re-saves.
    pub next_id: u64,
    pub records: Vec<SceneRecord>,
    /// Editor camera pose at save time, restored on load. Absent in older files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub editor_camera: Option<EditorCamera>,
}

impl SceneFile {
    /// Pretty-print to RON for on-disk storage.
    pub fn to_ron(&self) -> Result<String, ron::Error> {
        let cfg = ron::ser::PrettyConfig::new()
            .struct_names(true)
            .indentor("  ".to_string());
        ron::ser::to_string_pretty(self, cfg)
    }

    /// Parse from RON text.
    pub fn from_ron(text: &str) -> Result<Self, ron::error::SpannedError> {
        ron::from_str(text)
    }
}
