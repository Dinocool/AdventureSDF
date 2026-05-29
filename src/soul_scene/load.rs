//! Scene loading: parse a `.scene`, spawn entities, deserialize components via
//! reflection, and recursively instantiate nested scenes applying per-instance
//! overrides (Godot-style). Outermost overrides win; nesting composes A→B→C.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bevy::ecs::reflect::ReflectComponent;
use bevy::prelude::*;
use bevy::reflect::TypeRegistry;
use bevy::reflect::serde::ReflectDeserializer;
use serde::de::DeserializeSeed;

use crate::scene_manager::SceneEntity;

use super::format::{ComponentMap, LocalId, SceneFile, SceneRecord};
use super::{InstanceChild, SceneInstance};

/// Errors raised while loading a `.scene`.
#[derive(Debug)]
pub enum SceneLoadError {
    Io(String),
    Parse(String),
    /// A nested instance referenced a scene already on the load stack (cycle).
    Cycle(PathBuf),
}

impl std::fmt::Display for SceneLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SceneLoadError::Io(e) => write!(f, "scene io error: {e}"),
            SceneLoadError::Parse(e) => write!(f, "scene parse error: {e}"),
            SceneLoadError::Cycle(p) => write!(f, "cyclic scene instance: {}", p.display()),
        }
    }
}

/// Spawn a `.scene` file's contents into the world. Returns the spawned root
/// entities (one per top-level record), all tagged [`SceneEntity`].
pub fn load_scene(
    world: &mut World,
    path: &Path,
    registry: &TypeRegistry,
) -> Result<Vec<Entity>, SceneLoadError> {
    let mut stack = Vec::new();
    instantiate(world, path, registry, &mut stack)
}

/// Recursively instantiate a scene file. `stack` carries the in-progress source
/// paths so cycles are detected instead of looping forever.
fn instantiate(
    world: &mut World,
    path: &Path,
    registry: &TypeRegistry,
    stack: &mut Vec<PathBuf>,
) -> Result<Vec<Entity>, SceneLoadError> {
    let canonical = path.to_path_buf();
    if stack.contains(&canonical) {
        return Err(SceneLoadError::Cycle(canonical));
    }
    stack.push(canonical);

    let text = std::fs::read_to_string(path).map_err(|e| SceneLoadError::Io(e.to_string()))?;
    let file = SceneFile::from_ron(&text).map_err(|e| SceneLoadError::Parse(e.to_string()))?;

    let mut roots = Vec::new();
    for record in &file.records {
        match record {
            SceneRecord::Entity { id, components } => {
                let entity = world.spawn((SceneEntity, *id)).id();
                apply_components(world, entity, components, registry);
                roots.push(entity);
            }
            SceneRecord::Instance {
                id,
                source,
                overrides,
            } => {
                let root = instantiate_nested(world, id, source, overrides, registry, stack)?;
                roots.push(root);
            }
        }
    }

    stack.pop();
    Ok(roots)
}

/// Materialize a nested instance: load the source subtree, tag it as instance
/// children, then apply this instance's per-sub-entity overrides on top.
fn instantiate_nested(
    world: &mut World,
    id: &LocalId,
    source: &Path,
    overrides: &std::collections::BTreeMap<u64, ComponentMap>,
    registry: &TypeRegistry,
    stack: &mut Vec<PathBuf>,
) -> Result<Entity, SceneLoadError> {
    // Dedicated instance root: carries SceneInstance + LocalId, NOT InstanceChild,
    // so the save walk emits it (the ref) and prunes its expanded children.
    let root = world.spawn((SceneEntity, *id)).id();

    let children = instantiate(world, source, registry, stack)?;

    // Map source LocalId -> spawned entity so overrides can target sub-entities,
    // and tag every materialized child so save can prune the expanded subtree.
    let mut by_local: HashMap<u64, Entity> = HashMap::new();
    for &child in &children {
        if let Some(local) = world.get::<LocalId>(child) {
            by_local.insert(local.0, child);
        }
        world.entity_mut(child).insert(InstanceChild { root });
    }

    // Apply overrides keyed by the source's LocalId.
    for (local_id, comp_map) in overrides {
        match by_local.get(local_id) {
            Some(&target) => apply_components(world, target, comp_map, registry),
            None => warn!(
                "scene instance {:?}: override targets source id {local_id} which no \
                 longer exists in {} — dropping override",
                id,
                source.display()
            ),
        }
    }

    world.entity_mut(root).insert(SceneInstance {
        source: source.to_path_buf(),
        overrides: overrides
            .iter()
            .map(|(k, v)| (LocalId(*k), v.clone()))
            .collect(),
    });

    Ok(root)
}

/// Deserialize each `(type_path -> RON)` component and insert it onto `entity`.
/// Uses `ReflectComponent::insert` (overwrite-or-insert) so applying overrides on
/// top of an instanced subtree is idempotent and never panics on an absent
/// component (unlike `apply`).
fn apply_components(
    world: &mut World,
    entity: Entity,
    components: &ComponentMap,
    registry: &TypeRegistry,
) {
    for (type_path, ron_value) in components {
        let Some(registration) = registry.get_with_type_path(type_path) else {
            warn!("scene load: type `{type_path}` not registered — skipping component");
            continue;
        };
        let Some(reflect_component) = registration.data::<ReflectComponent>() else {
            warn!("scene load: type `{type_path}` has no ReflectComponent — skipping");
            continue;
        };

        let mut de = ron::Deserializer::from_str(ron_value)
            .expect("override RON should be valid (written by our serializer)");
        let reflect_de = ReflectDeserializer::new(registry);
        let value = match reflect_de.deserialize(&mut de) {
            Ok(v) => v,
            Err(e) => {
                warn!("scene load: failed to deserialize `{type_path}`: {e}");
                continue;
            }
        };

        let mut entity_mut = world.entity_mut(entity);
        reflect_component.insert(&mut entity_mut, value.as_partial_reflect(), registry);
    }
}
