//! Round-trip tests for the `.scene` format: plain entities, nested instances
//! with overrides, deep nesting composition, and cycle detection.

use std::path::PathBuf;

use bevy::prelude::*;
use bevy::reflect::TypeRegistry;

use crate::scene_manager::SceneEntity;
use crate::sdf_render::{SdfMaterial, SdfOp, SdfOrder, SdfPrimitive, SdfVolume};

use super::format::{LocalId, SceneFile, SceneRecord};
use super::{load::load_scene, save::save_scene_to_string};

/// A registry with every type a scene round-trip touches. Mirrors what the real
/// app registers across its plugins.
fn test_registry() -> TypeRegistry {
    let mut r = TypeRegistry::new();
    r.register::<Transform>();
    r.register::<SceneEntity>();
    r.register::<crate::node::SceneNode>();
    r.register::<crate::node::Node3D>();
    r.register::<SdfVolume>();
    r.register::<SdfPrimitive>();
    r.register::<SdfOp>();
    r.register::<SdfOrder>();
    r.register::<SdfMaterial>();
    r.register::<LocalId>();
    r.register::<super::SceneInstance>();
    r.register::<super::InstanceChild>();
    r.register::<super::NonSerializable>();
    r.register::<super::SkipSerialization>();
    r.register::<super::EditorHidden>();
    r.register::<ChildOf>();
    r.register::<Children>();
    // Field types reached by reflection serialization.
    r.register::<Vec3>();
    r.register::<Quat>();
    r
}

/// Unique temp path so parallel test runs don't collide.
fn temp_scene(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "soul_scene_test_{}_{}.scene",
        name,
        std::process::id()
    ));
    p
}

fn spawn_sphere(world: &mut World, id: u64, order: u32) -> Entity {
    world
        .spawn((
            SceneEntity,
            SdfVolume,
            LocalId(id),
            Transform::from_xyz(1.0, 2.0, 3.0),
            SdfPrimitive::Sphere { radius: 0.5 },
            SdfOp::default(),
            SdfOrder(order),
            SdfMaterial::default(),
        ))
        .id()
}

#[test]
fn plain_entity_round_trip_is_stable() {
    let registry = test_registry();

    // Author a world, save it.
    let mut world = World::new();
    spawn_sphere(&mut world, 0, 0);
    let first = save_scene_to_string(&mut world, &registry).expect("save");

    // Write it, load into a fresh world, re-save: text must be byte-identical.
    let path = temp_scene("plain");
    std::fs::write(&path, &first).unwrap();

    let mut world2 = World::new();
    load_scene(&mut world2, &path, &registry).expect("load");
    let second = save_scene_to_string(&mut world2, &registry).expect("re-save");

    assert_eq!(first, second, "save→load→save must be byte-stable");

    // And the geometry survived.
    let prim = world2
        .query::<&SdfPrimitive>()
        .iter(&world2)
        .next()
        .cloned();
    assert!(matches!(prim, Some(SdfPrimitive::Sphere { .. })));
    std::fs::remove_file(&path).ok();
}

#[test]
fn nested_instance_applies_override_and_resaves_only_diff() {
    let registry = test_registry();

    // Source scene: one sphere, radius 0.5, order 0.
    let src = temp_scene("nested_src");
    let mut src_world = World::new();
    spawn_sphere(&mut src_world, 0, 0);
    let src_ron = save_scene_to_string(&mut src_world, &registry).unwrap();
    std::fs::write(&src, &src_ron).unwrap();

    // Parent scene: instance the source, override sub-entity 0's SdfOrder to 7.
    let override_order = ron::ser::to_string(&bevy::reflect::serde::ReflectSerializer::new(
        SdfOrder(7).as_partial_reflect(),
        &registry,
    ))
    .unwrap();
    let mut overrides = std::collections::BTreeMap::new();
    let mut comp = std::collections::BTreeMap::new();
    comp.insert(
        "adventure::sdf_render::edits::SdfOrder".to_string(),
        override_order,
    );
    overrides.insert(0u64, comp);

    let parent = SceneFile {
        next_id: 1,
        records: vec![SceneRecord::Instance {
            id: LocalId(0),
            parent: None,
            source: src.clone(),
            overrides,
        }],
        editor_camera: None,
    };
    let parent_path = temp_scene("nested_parent");
    std::fs::write(&parent_path, parent.to_ron().unwrap()).unwrap();

    // Load the parent: the override must have been applied to the instanced sphere.
    let mut world = World::new();
    load_scene(&mut world, &parent_path, &registry).expect("load nested");

    let max_order = world
        .query::<&SdfOrder>()
        .iter(&world)
        .map(|o| o.0)
        .max()
        .unwrap();
    assert_eq!(max_order, 7, "override SdfOrder(7) must be applied");

    // Re-save: the parent must still be ONE Instance record (subtree pruned), not
    // the expanded sphere.
    let resaved = save_scene_to_string(&mut world, &registry).unwrap();
    let reparsed = SceneFile::from_ron(&resaved).unwrap();
    assert_eq!(reparsed.records.len(), 1, "instance subtree must be pruned");
    assert!(
        matches!(reparsed.records[0], SceneRecord::Instance { .. }),
        "re-saved record must remain an Instance"
    );

    std::fs::remove_file(&src).ok();
    std::fs::remove_file(&parent_path).ok();
}

#[test]
fn deep_nesting_composes() {
    let registry = test_registry();

    // C: a plain sphere.
    let c = temp_scene("deep_c");
    let mut cw = World::new();
    spawn_sphere(&mut cw, 0, 0);
    std::fs::write(&c, save_scene_to_string(&mut cw, &registry).unwrap()).unwrap();

    // B: instances C.
    let b = temp_scene("deep_b");
    let b_file = SceneFile {
        next_id: 1,
        records: vec![SceneRecord::Instance {
            id: LocalId(0),
            parent: None,
            source: c.clone(),
            overrides: Default::default(),
        }],
        editor_camera: None,
    };
    std::fs::write(&b, b_file.to_ron().unwrap()).unwrap();

    // A: instances B.
    let a = temp_scene("deep_a");
    let a_file = SceneFile {
        next_id: 1,
        records: vec![SceneRecord::Instance {
            id: LocalId(0),
            parent: None,
            source: b.clone(),
            overrides: Default::default(),
        }],
        editor_camera: None,
    };
    std::fs::write(&a, a_file.to_ron().unwrap()).unwrap();

    // Loading A must materialize C's sphere through B.
    let mut world = World::new();
    load_scene(&mut world, &a, &registry).expect("load A→B→C");
    let spheres = world
        .query::<&SdfPrimitive>()
        .iter(&world)
        .filter(|p| matches!(p, SdfPrimitive::Sphere { .. }))
        .count();
    assert_eq!(spheres, 1, "deep nesting must materialize the leaf sphere");

    for p in [&a, &b, &c] {
        std::fs::remove_file(p).ok();
    }
}

#[test]
fn parent_child_hierarchy_round_trips() {
    let registry = test_registry();

    // Parent sphere (LocalId 0) with a child sphere (LocalId 1) under it.
    let mut world = World::new();
    let parent = spawn_sphere(&mut world, 0, 0);
    let child = spawn_sphere(&mut world, 1, 1);
    world.entity_mut(child).insert(ChildOf(parent));

    let path = temp_scene("hierarchy");
    std::fs::write(
        &path,
        save_scene_to_string(&mut world, &registry).expect("save"),
    )
    .unwrap();

    // Load into a fresh world; the child's ChildOf must resolve to the reloaded
    // parent (the one carrying LocalId 0).
    let mut world2 = World::new();
    load_scene(&mut world2, &path, &registry).expect("load");

    let parent2 = world2
        .query_filtered::<(Entity, &LocalId), ()>()
        .iter(&world2)
        .find(|(_, id)| id.0 == 0)
        .map(|(e, _)| e)
        .expect("parent reloaded");
    let (child2, child_parent) = world2
        .query::<(Entity, &LocalId, &ChildOf)>()
        .iter(&world2)
        .find(|(_, id, _)| id.0 == 1)
        .map(|(e, _, c)| (e, c.parent()))
        .expect("child reloaded with ChildOf");

    assert_ne!(child2, parent2);
    assert_eq!(child_parent, parent2, "child must re-link to its parent node");

    // Bevy auto-builds Children on the parent.
    let children: Vec<Entity> = world2
        .get::<Children>(parent2)
        .map(|c| c.iter().collect())
        .unwrap_or_default();
    assert_eq!(children, vec![child2], "parent must list the child");

    std::fs::remove_file(&path).ok();
}

#[test]
fn cyclic_instance_errors() {
    let registry = test_registry();

    // Two scenes that instance each other.
    let a = temp_scene("cycle_a");
    let b = temp_scene("cycle_b");

    let a_file = SceneFile {
        next_id: 1,
        records: vec![SceneRecord::Instance {
            id: LocalId(0),
            parent: None,
            source: b.clone(),
            overrides: Default::default(),
        }],
        editor_camera: None,
    };
    let b_file = SceneFile {
        next_id: 1,
        records: vec![SceneRecord::Instance {
            id: LocalId(0),
            parent: None,
            source: a.clone(),
            overrides: Default::default(),
        }],
        editor_camera: None,
    };
    std::fs::write(&a, a_file.to_ron().unwrap()).unwrap();
    std::fs::write(&b, b_file.to_ron().unwrap()).unwrap();

    let mut world = World::new();
    let result = load_scene(&mut world, &a, &registry);
    assert!(
        matches!(result, Err(super::SceneLoadError::Cycle(_))),
        "cyclic instances must error, not loop: got {result:?}"
    );

    std::fs::remove_file(&a).ok();
    std::fs::remove_file(&b).ok();
}
