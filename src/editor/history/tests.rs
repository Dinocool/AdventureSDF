//! Undo/redo unit tests on a minimal world with reflection registered so the
//! reflection-based commands round-trip.

use bevy::prelude::*;
use bevy::reflect::TypePath;

use crate::soul_scene::LocalId;

use super::*;

/// A bare world carrying an `AppTypeRegistry` (the components the commands touch) + the
/// selection resource the structural ops read.
fn test_world() -> World {
    let mut world = World::new();
    let registry = AppTypeRegistry::default();
    {
        let mut w = registry.write();
        w.register::<Transform>();
        w.register::<Vec3>();
        w.register::<Quat>();
        w.register::<LocalId>();
        w.register::<crate::sdf_render::SdfPrimitive>();
    }
    world.insert_resource(registry);
    world.init_resource::<crate::sdf_render::SdfSelection>();
    world
}

#[test]
fn next_free_local_id_is_max_plus_one() {
    let mut world = test_world();
    assert_eq!(next_free_local_id(&mut world), 0);
    world.spawn(LocalId(4));
    world.spawn(LocalId(2));
    assert_eq!(next_free_local_id(&mut world), 5);
}

#[test]
fn resolve_finds_entity_by_local_id() {
    let mut world = test_world();
    let e = world.spawn(LocalId(7)).id();
    assert_eq!(resolve(&mut world, LocalId(7)), Some(e));
    assert_eq!(resolve(&mut world, LocalId(99)), None);
}

#[test]
fn component_edit_reverts_and_reapplies_transform() {
    let mut world = test_world();
    let tp = Transform::type_path();
    let e = world.spawn((LocalId(1), Transform::from_xyz(1.0, 0.0, 0.0))).id();

    let before =
        reflect_to_ron(&world, world.get::<Transform>(e).unwrap().as_partial_reflect())
            .expect("serialize before");
    world.entity_mut(e).insert(Transform::from_xyz(9.0, 0.0, 0.0));
    let after = reflect_to_ron(&world, world.get::<Transform>(e).unwrap().as_partial_reflect())
        .expect("serialize after");

    let cmd = ComponentEdit::new(LocalId(1), tp, before, after).expect("non-noop edit");

    cmd.revert(&mut world);
    assert_eq!(world.get::<Transform>(e).unwrap().translation.x, 1.0);
    cmd.reapply(&mut world);
    assert_eq!(world.get::<Transform>(e).unwrap().translation.x, 9.0);
}

#[test]
fn rename_command_round_trips() {
    let mut world = test_world();
    let e = world.spawn((LocalId(3), Name::new("old"))).id();
    let cmd = RenameCommand::new(LocalId(3), "old".into(), "new".into());
    cmd.reapply(&mut world);
    assert_eq!(world.get::<Name>(e).unwrap().as_str(), "new");
    cmd.revert(&mut world);
    assert_eq!(world.get::<Name>(e).unwrap().as_str(), "old");
}

#[test]
fn reparent_command_swaps_parent_and_local() {
    let mut world = test_world();
    let p1 = world.spawn(LocalId(10)).id();
    let p2 = world.spawn(LocalId(11)).id();
    let c = world.spawn((LocalId(12), Transform::from_xyz(1.0, 0.0, 0.0))).id();
    world.entity_mut(c).insert(ChildOf(p1));

    let cmd = ReparentCommand::new(
        LocalId(12),
        Some(LocalId(10)),
        Transform::from_xyz(1.0, 0.0, 0.0),
        Some(LocalId(11)),
        Transform::from_xyz(4.0, 0.0, 0.0),
    );
    cmd.reapply(&mut world);
    assert_eq!(world.get::<ChildOf>(c).unwrap().parent(), p2);
    assert_eq!(world.get::<Transform>(c).unwrap().translation.x, 4.0);
    cmd.revert(&mut world);
    assert_eq!(world.get::<ChildOf>(c).unwrap().parent(), p1);
    assert_eq!(world.get::<Transform>(c).unwrap().translation.x, 1.0);
}

#[test]
fn scene_history_coalesces_consecutive_same_target_edits() {
    let mut h = SceneHistory::default();
    let mk = |before: &str, after: &str| {
        Box::new(
            ComponentEdit::new(LocalId(1), "T", before.to_string(), after.to_string()).unwrap(),
        ) as Box<dyn EditCommand>
    };
    // Same target, drag "open": second edit folds into the first (one undo step, before kept).
    h.open = Some((LocalId(1), "T".to_string()));
    h.record(mk("a", "b"));
    h.record(mk("b", "c"));
    assert_eq!(h.undo.len(), 1);
    let top = h.undo[0].as_any().downcast_ref::<ComponentEdit>().unwrap();
    assert_eq!(top.before, "a");
    assert_eq!(top.after, "c");
}
