//! The demo gallery scene authored as data: a wide flat ground plane plus a spread of
//! distinct SDF primitives resting on it, and a directional light. Spawned into a world
//! with stable [`LocalId`]s so it can be serialized to `assets/scenes/gallery.scene` (the
//! editor's default scene). Each volume references its material **by file path** via
//! [`SdfMaterialSource`]; the runtime `registry_id` is derived on load by `resolve_materials`,
//! so nothing here depends on material load order.

use bevy::prelude::*;

use crate::node::{EditorGizmo, Node3D};
use crate::scene_manager::SceneEntity;
use crate::soul_scene::LocalId;

use super::{CsgKind, SdfMaterialSource, SdfOp, SdfOrder, SdfPrimitive, SdfVolume};

/// Ground-plane half-height (thin slab → reads like a plane). Its top face sits at y = 0.
#[cfg(test)]
const PLANE_HALF_Y: f32 = 0.15;

/// A material file path relative to `assets/`, for [`SdfMaterialSource::asset`].
#[cfg(test)]
fn mat(name: &str) -> Option<std::path::PathBuf> {
    Some(std::path::PathBuf::from(format!("materials/{name}.material.ron")))
}

/// Spawn the gallery (7 SDF volumes + a directional light) into `world` with stable
/// `LocalId`s, ready for serialization. The volumes are all plain unions; each object's centre is
/// placed so it rests on the ground plane (top face at y = 0). Test-only: the runtime loads the
/// serialized `assets/scenes/gallery.scene`; this builder only regenerates that file (see the test).
#[cfg(test)]
pub fn spawn_gallery(world: &mut World) {
    // (local_id, order, transform, primitive, material file name)
    let volumes: [(u64, u32, Transform, SdfPrimitive, &str); 10] = [
        // Ground plane: wide + thin, top face at y = 0 (centre at y = -half_y).
        (
            0,
            0,
            Transform::from_xyz(0.0, -PLANE_HALF_Y, 0.0),
            SdfPrimitive::Box {
                half_extents: Vec3::new(4.0, PLANE_HALF_Y, 3.0),
            },
            "sand",
        ),
        // Box resting on the plane (half-height 0.4 → centre at y = 0.4).
        (
            1,
            1,
            Transform::from_xyz(-2.4, 0.4, 0.4),
            SdfPrimitive::Box {
                half_extents: Vec3::splat(0.4),
            },
            "cobble",
        ),
        // Headline exemplar: deep-red mirror metal sphere.
        (
            2,
            2,
            Transform::from_xyz(-1.1, 0.55, -0.3),
            SdfPrimitive::Sphere { radius: 0.55 },
            "red_metal",
        ),
        // Torus lies flat: its half-thickness above centre is `minor` (0.18).
        (
            3,
            3,
            Transform::from_xyz(0.2, 0.18, 0.5),
            SdfPrimitive::Torus {
                major: 0.5,
                minor: 0.18,
            },
            "ground",
        ),
        // Rough gold metal exemplar.
        (
            4,
            4,
            Transform::from_xyz(1.3, 0.68, -0.4),
            SdfPrimitive::Capsule {
                half_height: 0.4,
                radius: 0.28,
            },
            "gold_rough",
        ),
        // Cylinder standing up: half-height above centre.
        (
            5,
            5,
            Transform::from_xyz(2.4, 0.5, 0.3),
            SdfPrimitive::Cylinder {
                radius: 0.4,
                half_height: 0.5,
            },
            "cobble",
        ),
        // Glossy white dielectric exemplar.
        (
            6,
            6,
            Transform::from_xyz(0.6, 0.45, -1.1),
            SdfPrimitive::Sphere { radius: 0.45 },
            "white_gloss",
        ),
        // Emissive orange "lamp" sphere, near the cobble box + red sphere so its warm glow
        // bounces onto them through the GI.
        (
            7,
            7,
            Transform::from_xyz(-1.8, 0.35, 0.9),
            SdfPrimitive::Sphere { radius: 0.35 },
            "emissive_orange",
        ),
        // Emissive cyan capsule standing up, lighting the gold/white exemplars from the side.
        (
            8,
            8,
            Transform::from_xyz(1.9, 0.5, -1.0),
            SdfPrimitive::Capsule {
                half_height: 0.3,
                radius: 0.2,
            },
            "emissive_cyan",
        ),
        // Emissive green torus lying flat on the plane, a low ambient glow patch.
        (
            9,
            9,
            Transform::from_xyz(0.0, 0.14, 1.6),
            SdfPrimitive::Torus {
                major: 0.4,
                minor: 0.14,
            },
            "emissive_green",
        ),
    ];

    for (local, order, transform, prim, mat_name) in volumes {
        world.spawn((
            LocalId(local),
            transform,
            prim,
            SdfOp {
                kind: CsgKind::Union,
                smoothing: 0.0,
            },
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

    // Directional light so 3D geometry (and debug wireframes) are visible.
    world.spawn((
        LocalId(10),
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

    /// Type registry covering every component + field type the gallery scene serializes.
    fn gallery_registry() -> TypeRegistry {
        let mut r = TypeRegistry::new();
        // Components on the authored entities.
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
        // Field types reached by reflection serialization.
        r.register::<Vec3>();
        r.register::<Quat>();
        r.register::<Color>();
        r
    }

    /// Generate the default gallery scene file. Run with:
    /// `cargo test -- generate_gallery_scene --nocapture`
    #[test]
    fn generate_gallery_scene() {
        let registry = gallery_registry();
        let mut world = World::new();
        spawn_gallery(&mut world);

        let ron = crate::soul_scene::save_scene_to_string(&mut world, &registry)
            .expect("serialize gallery scene");

        std::fs::create_dir_all("assets/scenes").expect("create assets/scenes");
        std::fs::write("assets/scenes/gallery.scene", &ron).expect("write gallery.scene");
        println!("wrote assets/scenes/gallery.scene:\n{ron}");
    }
}
