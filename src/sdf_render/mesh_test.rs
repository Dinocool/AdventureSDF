//! Minimal, legible SDF scene for the **mesh-bake migration** — a handful of CSG shapes chosen so the
//! Surface Nets meshing behaviour is immediately readable:
//!
//! - a **sharp cube** — the edge-rounding test (Surface Nets bevels 90° edges);
//! - a **sphere** — the smooth baseline (Surface Nets is ideal here);
//! - a **smooth box∪sphere blend** — a wide-smoothing union (Surface Nets' happy path);
//! - a **box−sphere subtraction** — a concavity + a thin lip (stresses thin features).
//!
//! Every shape is ≥ one brick across, so cross-brick same-LOD seams are exercised inherently.
//!
//! Like [`super::gallery`], this is a pure scene GENERATOR (test-only): the runtime loads the
//! serialized `assets/scenes/mesh_test.scene`. Regenerate with:
//! `cargo test -- generate_mesh_test_scene --nocapture`.

use bevy::prelude::*;

use crate::node::{EditorGizmo, Node3D};
use crate::scene_manager::SceneEntity;
use crate::soul_scene::LocalId;

use super::{CsgKind, SdfMaterialSource, SdfOp, SdfOrder, SdfPrimitive, SdfVolume};

/// Ground-plane half-height (thin slab → reads like a plane). Its top face sits at y = 0.
const PLANE_HALF_Y: f32 = 0.15;

/// A material file path relative to `assets/`, for [`SdfMaterialSource::asset`].
fn mat(name: &str) -> Option<std::path::PathBuf> {
    Some(std::path::PathBuf::from(format!("materials/{name}.material.ron")))
}

/// Spawn the mesh-test scene (CSG volumes + a directional light) with stable `LocalId`s, ready for
/// serialization. Volumes fold into one CSG field in `SdfOrder`, so the subtraction sphere must come
/// after the box it carves; each carve/blend is positioned to only overlap its intended neighbour.
fn spawn_mesh_test(world: &mut World) {
    // (local_id, order, transform, primitive, op, material file name)
    let volumes: [(u64, u32, Transform, SdfPrimitive, SdfOp, &str); 7] = [
        // Ground plane: wide + thin, top face at y = 0 (centre at y = -half_y).
        (
            0,
            0,
            Transform::from_xyz(0.0, -PLANE_HALF_Y, 0.0),
            SdfPrimitive::Box {
                half_extents: Vec3::new(4.0, PLANE_HALF_Y, 2.5),
            },
            SdfOp {
                kind: CsgKind::Union,
                smoothing: 0.0,
            },
            "sand",
        ),
        // Sharp cube — THE edge-rounding test (Surface Nets bevels these to ~one voxel).
        (
            1,
            1,
            Transform::from_xyz(-2.2, 0.5, 0.0),
            SdfPrimitive::Box {
                half_extents: Vec3::splat(0.5),
            },
            SdfOp {
                kind: CsgKind::Union,
                smoothing: 0.0,
            },
            "cobble",
        ),
        // Sphere — smooth baseline (Surface Nets reproduces this cleanly).
        (
            2,
            2,
            Transform::from_xyz(-0.7, 0.5, 0.0),
            SdfPrimitive::Sphere { radius: 0.5 },
            SdfOp {
                kind: CsgKind::Union,
                smoothing: 0.0,
            },
            "red_metal",
        ),
        // Smooth blend: a box, then an overlapping sphere unioned with a wide smoothing band.
        (
            3,
            3,
            Transform::from_xyz(0.95, 0.45, 0.0),
            SdfPrimitive::Box {
                half_extents: Vec3::splat(0.45),
            },
            SdfOp {
                kind: CsgKind::Union,
                smoothing: 0.0,
            },
            "white_gloss",
        ),
        (
            4,
            4,
            Transform::from_xyz(1.45, 0.6, 0.0),
            SdfPrimitive::Sphere { radius: 0.42 },
            SdfOp {
                kind: CsgKind::Union,
                smoothing: 0.3,
            },
            "white_gloss",
        ),
        // Subtraction: a box, then a sphere carved out of it (concavity + a thin lip).
        (
            5,
            5,
            Transform::from_xyz(2.9, 0.5, 0.0),
            SdfPrimitive::Box {
                half_extents: Vec3::splat(0.5),
            },
            SdfOp {
                kind: CsgKind::Union,
                smoothing: 0.0,
            },
            "cobble",
        ),
        (
            6,
            6,
            Transform::from_xyz(3.15, 0.85, 0.25),
            SdfPrimitive::Sphere { radius: 0.38 },
            SdfOp {
                kind: CsgKind::Subtract,
                smoothing: 0.05,
            },
            "cobble",
        ),
    ];

    for (local, order, transform, prim, op, mat_name) in volumes {
        world.spawn((
            LocalId(local),
            transform,
            prim,
            op,
            SdfOrder(order),
            SdfMaterialSource {
                asset: mat(mat_name),
                ..default()
            },
            SdfVolume,
            Node3D,
            SceneEntity,
        ));
    }

    // Directional light so the geometry (and debug wireframes) are visible.
    world.spawn((
        LocalId(7),
        Name::new("Directional Light"),
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_x(-0.5)),
        Node3D,
        EditorGizmo::DirectionalLight { scale: 1.0 },
        SceneEntity,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::reflect::TypeRegistry;

    /// Type registry covering every component + field type the scene serializes (mirrors
    /// `gallery`'s registry).
    fn mesh_test_registry() -> TypeRegistry {
        let mut r = TypeRegistry::new();
        r.register::<Transform>();
        r.register::<Name>();
        r.register::<SceneEntity>();
        r.register::<crate::node::SceneNode>();
        r.register::<Node3D>();
        r.register::<EditorGizmo>();
        r.register::<SdfVolume>();
        r.register::<SdfPrimitive>();
        r.register::<SdfOp>();
        r.register::<SdfOrder>();
        r.register::<SdfMaterialSource>();
        r.register::<crate::sdf_render::MaterialFields>();
        r.register::<CsgKind>();
        r.register::<DirectionalLight>();
        r.register::<LocalId>();
        r.register::<crate::soul_scene::SceneInstance>();
        r.register::<crate::soul_scene::InstanceChild>();
        r.register::<crate::soul_scene::NonSerializable>();
        r.register::<crate::soul_scene::SkipSerialization>();
        r.register::<crate::soul_scene::EditorHidden>();
        r.register::<ChildOf>();
        r.register::<Children>();
        r.register::<Vec3>();
        r.register::<Quat>();
        r.register::<Color>();
        r
    }

    /// Generate the mesh-bake test scene file. Run with:
    /// `cargo test -- generate_mesh_test_scene --nocapture`
    #[test]
    fn generate_mesh_test_scene() {
        let registry = mesh_test_registry();
        let mut world = World::new();
        spawn_mesh_test(&mut world);

        let ron = crate::soul_scene::save_scene_to_string(&mut world, &registry)
            .expect("serialize mesh_test scene");

        std::fs::create_dir_all("assets/scenes").expect("create assets/scenes");
        std::fs::write("assets/scenes/mesh_test.scene", &ron).expect("write mesh_test.scene");
        println!("wrote assets/scenes/mesh_test.scene:\n{ron}");
    }
}
