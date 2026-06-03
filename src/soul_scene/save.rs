//! Scene saving: walk the live editor world and emit a `.scene`. Locally-authored
//! entities emit their components; instanced subtrees emit only the instance ref +
//! re-diffed overrides (the expanded children are pruned). Editor-only and
//! runtime-rebuilt entities are excluded via marker components.

use std::collections::BTreeMap;

use bevy::ecs::reflect::ReflectComponent;
use bevy::prelude::*;
use bevy::reflect::TypeRegistry;
use bevy::reflect::serde::ReflectSerializer;

use crate::scene_manager::{EditorEntity, SceneEntity};

use super::format::{ComponentMap, LocalId, SceneFile, SceneRecord};
use super::{
    EditorHidden, InstanceChild, NonSerializable, ReflectSerializeSkip, SceneInstance,
    SkipSerialization,
};

/// Foreign (engine) type paths never written to a `.scene`: runtime-derived/bookkeeping
/// components on `bevy_*` crates that we can't annotate. Our *own* components opt out at
/// their definition with `#[reflect(SerializeSkip)]` instead (see [`ReflectSerializeSkip`]).
const SKIP_TYPE_PATHS: &[&str] = &[
    // Hierarchy is persisted as a stable parent `LocalId` on each record (see
    // `parent_local_id`), not as the raw `Entity` in `ChildOf`.
    "bevy_ecs::hierarchy::ChildOf",
    "bevy_ecs::hierarchy::Children",
    // Render-world sync bookkeeping. `RenderEntity` holds a render-world entity id that is
    // only valid for the live run; serializing + restoring it makes Bevy try to sync an
    // already-synced entity ("Attempting to synchronize an entity that has already been
    // synchronized!"). Both are re-added automatically as required components on load.
    "bevy_render::sync_world::RenderEntity",
    "bevy_render::sync_world::SyncToRenderWorld",
    // Transform-derived: `GlobalTransform` is recomputed from `Transform` by propagation,
    // and `TransformTreeChanged` is a per-frame dirty flag. Saving them is redundant and
    // restoring a stale `GlobalTransform` would fight propagation for a frame.
    "bevy_transform::components::global_transform::GlobalTransform",
    "bevy_transform::components::transform::TransformTreeChanged",
    // Light/visibility runtime state, auto-added as required components of lights/cameras
    // and rebuilt every frame — never authored, never restored.
    "bevy_camera::primitives::CascadesFrusta",
    "bevy_camera::visibility::CascadesVisibleEntities",
    "bevy_camera::visibility::InheritedVisibility",
    "bevy_camera::visibility::ViewVisibility",
    // Auto-added required component of `Visibility`; holds `TypeId`s, which have no
    // `ReflectSerialize` — serializing it only spams warnings (it's never authored).
    "bevy_camera::visibility::VisibilityClass",
    "bevy_light::cascade::Cascades",
];

/// Errors raised while saving a `.scene`.
#[derive(Debug)]
pub enum SceneSaveError {
    Io(String),
    Serialize(String),
}

impl std::fmt::Display for SceneSaveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SceneSaveError::Io(e) => write!(f, "scene io error: {e}"),
            SceneSaveError::Serialize(e) => write!(f, "scene serialize error: {e}"),
        }
    }
}

/// Serialize the world's scene entities to a `.scene` RON string (no camera).
pub fn save_scene_to_string(
    world: &mut World,
    registry: &TypeRegistry,
) -> Result<String, SceneSaveError> {
    save_scene_to_string_with_camera(world, registry, None)
}

/// Like [`save_scene_to_string`] but embeds `camera` (the editor's saved view) in the file.
pub fn save_scene_to_string_with_camera(
    world: &mut World,
    registry: &TypeRegistry,
    camera: Option<super::EditorCamera>,
) -> Result<String, SceneSaveError> {
    let mut file = build_scene_file(world, registry);
    file.editor_camera = camera;
    file.to_ron()
        .map_err(|e| SceneSaveError::Serialize(e.to_string()))
}

/// Serialize and write the world's scene entities to `path`.
pub fn save_scene(
    world: &mut World,
    path: &std::path::Path,
    registry: &TypeRegistry,
) -> Result<(), SceneSaveError> {
    let ron = save_scene_to_string(world, registry)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| SceneSaveError::Io(e.to_string()))?;
    }
    std::fs::write(path, ron).map_err(|e| SceneSaveError::Io(e.to_string()))
}

/// Build the in-memory [`SceneFile`] from the live world.
fn build_scene_file(world: &mut World, registry: &TypeRegistry) -> SceneFile {
    // Candidate roots: scene entities that are NOT part of an instanced subtree and
    // NOT excluded by markers. Instanced subtrees are represented by their root's
    // SceneInstance, never by their expanded children.
    let mut entity_ids: Vec<(Entity, LocalId)> = world
        .query_filtered::<(Entity, &LocalId), (
            With<SceneEntity>,
            Without<EditorEntity>,
            Without<InstanceChild>,
            Without<NonSerializable>,
            Without<SkipSerialization>,
            Without<EditorHidden>,
        )>()
        .iter(world)
        .map(|(e, id)| (e, *id))
        .collect();
    entity_ids.sort_by_key(|(_, id)| id.0);

    let mut records = Vec::new();
    let mut max_id = 0u64;

    for (entity, id) in entity_ids {
        max_id = max_id.max(id.0);

        let parent = parent_local_id(world, entity);

        // An instance root re-emits its ref + a freshly re-diffed override map so
        // edits made to the instanced subtree since load are captured (the plan's
        // "re-capture on save" pitfall).
        if let Some(instance) = world.get::<SceneInstance>(entity).cloned() {
            records.push(SceneRecord::Instance {
                id,
                parent,
                source: instance.source.clone(),
                overrides: rediff_overrides(world, &instance, registry),
            });
            continue;
        }

        let components = serialize_entity_components(world, entity, registry);
        records.push(SceneRecord::Entity {
            id,
            parent,
            components,
        });
    }

    SceneFile {
        next_id: max_id + 1,
        records,
        editor_camera: None,
    }
}

/// Serialize `root` and its descendants (via `Children`) to a `.scene` RON string — the
/// subtree snapshot the editor uses for copy/paste and for undo of spawn/delete. The root is
/// emitted with no parent (`parent: None`) so it re-instantiates as a standalone root that the
/// caller reparents; descendant parent links are preserved within the subtree. Entities are
/// expected to carry `LocalId` (the editor's `ensure_local_ids` guarantees this for live scene
/// nodes); any without one are skipped.
pub fn save_subtree_to_string(
    world: &mut World,
    registry: &TypeRegistry,
    root: Entity,
) -> Result<String, SceneSaveError> {
    let mut members = Vec::new();
    collect_subtree(world, root, &mut members);

    let mut records = Vec::new();
    let mut max_id = 0u64;
    for entity in &members {
        let Some(id) = world.get::<LocalId>(*entity).copied() else {
            continue;
        };
        max_id = max_id.max(id.0);
        // Root detaches from its real parent; descendants keep links inside the subtree.
        let parent = if *entity == root {
            None
        } else {
            parent_local_id(world, *entity)
        };
        if let Some(instance) = world.get::<SceneInstance>(*entity).cloned() {
            records.push(SceneRecord::Instance {
                id,
                parent,
                source: instance.source.clone(),
                overrides: rediff_overrides(world, &instance, registry),
            });
        } else {
            records.push(SceneRecord::Entity {
                id,
                parent,
                components: serialize_entity_components(world, *entity, registry),
            });
        }
    }

    let file = SceneFile {
        next_id: max_id + 1,
        records,
        editor_camera: None,
    };
    file.to_ron()
        .map_err(|e| SceneSaveError::Serialize(e.to_string()))
}

/// Depth-first collect `root` + descendants that are serializable scene entities (same marker
/// filters as [`build_scene_file`]). Excluded nodes (editor-only / runtime-rebuilt) are skipped
/// but their children are still visited.
fn collect_subtree(world: &World, entity: Entity, out: &mut Vec<Entity>) {
    let serializable = world.get::<SceneEntity>(entity).is_some()
        && world.get::<EditorEntity>(entity).is_none()
        && world.get::<InstanceChild>(entity).is_none()
        && world.get::<NonSerializable>(entity).is_none()
        && world.get::<SkipSerialization>(entity).is_none()
        && world.get::<EditorHidden>(entity).is_none();
    if serializable {
        out.push(entity);
    }
    if let Some(children) = world.get::<Children>(entity) {
        for child in children.iter() {
            collect_subtree(world, child, out);
        }
    }
}

/// Resolve an entity's parent (via `ChildOf`) to the parent's stable `LocalId`, if
/// the parent is itself a saved scene entity. Returns `None` for roots or parents
/// outside the scene set.
fn parent_local_id(world: &World, entity: Entity) -> Option<u64> {
    let parent = world.get::<ChildOf>(entity)?.parent();
    world.get::<LocalId>(parent).map(|id| id.0)
}

/// Re-derive the override map for an instance by diffing each overridden
/// sub-entity's *live* component values against the recorded overrides. For this
/// pass we re-emit the stored overrides (they were applied at load and live edits
/// to instance children are not yet tracked back to their source id); this keeps
/// save lossless for the common "instance, don't touch children" case.
fn rediff_overrides(
    _world: &mut World,
    instance: &SceneInstance,
    _registry: &TypeRegistry,
) -> BTreeMap<u64, ComponentMap> {
    instance
        .overrides
        .iter()
        .map(|(k, v)| (k.0, v.clone()))
        .collect()
}

/// Serialize all reflected components present on `entity` (minus the skip list) to
/// a `type_path -> RON` map. Iterates the registry's `ReflectComponent` entries and
/// keeps those actually present on the entity.
fn serialize_entity_components(
    world: &World,
    entity: Entity,
    registry: &TypeRegistry,
) -> ComponentMap {
    let entity_ref = world.entity(entity);
    let mut map = ComponentMap::new();

    for registration in registry.iter() {
        let type_path = registration.type_info().type_path();
        // Our own components opt out via the `SerializeSkip` tag; foreign engine components
        // (which we can't annotate) are covered by the path list.
        if registration.data::<ReflectSerializeSkip>().is_some()
            || SKIP_TYPE_PATHS.contains(&type_path)
        {
            continue;
        }
        let Some(reflect_component) = registration.data::<ReflectComponent>() else {
            continue;
        };
        let Some(value) = reflect_component.reflect(entity_ref) else {
            continue;
        };

        let serializer = ReflectSerializer::new(value.as_partial_reflect(), registry);
        match ron::ser::to_string(&serializer) {
            Ok(ron) => {
                map.insert(type_path.to_string(), ron);
            }
            Err(e) => warn!("scene save: failed to serialize `{type_path}`: {e}"),
        }
    }

    map
}
