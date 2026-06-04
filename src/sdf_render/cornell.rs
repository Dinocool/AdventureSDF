//! A classic Cornell-box-style GLOBAL-ILLUMINATION demo scene, authored as data (like [`super::gallery`]).
//!
//! An open-front white room (matte white floor / ceiling / back + side walls) lit ONLY by a white
//! emissive panel on the ceiling, with three saturated red / green / blue diffuse objects inside. It's
//! built to show off the DDGI path: the white surfaces pick up the objects' colour (colour bleeding),
//! the objects cast soft contact shadows on the floor, and the single ceiling light fills the room
//! entirely through bounces. Serialized to `assets/scenes/cornell.scene`; regenerate via the test.
//!
//! Each volume references its material by file path via [`SdfMaterialSource`]; the runtime `registry_id`
//! is derived on load, so nothing here depends on material load order.

use bevy::prelude::*;

use crate::node::{EditorGizmo, Node3D};
use crate::scene_manager::SceneEntity;
use crate::soul_scene::LocalId;

use super::{CsgKind, SdfMaterialSource, SdfOp, SdfOrder, SdfPrimitive, SdfVolume};

fn mat(name: &str) -> Option<std::path::PathBuf> {
    Some(std::path::PathBuf::from(format!("materials/{name}.material.ron")))
}

/// Spawn the Cornell GI box into `world` with stable `LocalId`s. Room interior spans x∈[-2,2],
/// y∈[0,4], z∈[-2,2] (open front toward +z so the camera looks in). Wall slabs are 0.1 thick.
fn spawn_cornell(world: &mut World) {
    let b = |h: Vec3| SdfPrimitive::Box { half_extents: h };
    // (local_id, order, transform, primitive, material file name)
    let volumes: [(u64, u32, Transform, SdfPrimitive, &str); 9] = [
        // --- White room (matte) ---
        // Floor (top face at y=0) and ceiling (bottom face at y=4).
        (0, 0, Transform::from_xyz(0.0, -0.1, 0.0), b(Vec3::new(2.2, 0.1, 2.2)), "cornell_white"),
        (1, 1, Transform::from_xyz(0.0, 4.1, 0.0), b(Vec3::new(2.2, 0.1, 2.2)), "cornell_white"),
        // Back wall (z=-2) + left/right side walls (x=∓2).
        (2, 2, Transform::from_xyz(0.0, 2.0, -2.1), b(Vec3::new(2.2, 2.1, 0.1)), "cornell_white"),
        (3, 3, Transform::from_xyz(-2.1, 2.0, 0.0), b(Vec3::new(0.1, 2.1, 2.2)), "cornell_white"),
        (4, 4, Transform::from_xyz(2.1, 2.0, 0.0), b(Vec3::new(0.1, 2.1, 2.2)), "cornell_white"),
        // --- The light: a white emissive panel just below the ceiling, the room's only source ---
        (5, 5, Transform::from_xyz(0.0, 3.92, 0.0), b(Vec3::new(0.7, 0.06, 0.7)), "cornell_light"),
        // --- Coloured diffuse objects (bleed their colour onto the white surfaces) ---
        // Tall red box, slightly turned, back-left.
        (
            6,
            6,
            Transform::from_xyz(-0.85, 0.75, -0.55)
                .with_rotation(Quat::from_rotation_y(0.32)),
            b(Vec3::new(0.55, 0.75, 0.55)),
            "cornell_red",
        ),
        // Green cube, front-centre.
        (
            7,
            7,
            Transform::from_xyz(0.05, 0.42, 0.55).with_rotation(Quat::from_rotation_y(-0.35)),
            b(Vec3::splat(0.42)),
            "cornell_green",
        ),
        // Blue sphere, right.
        (8, 8, Transform::from_xyz(0.95, 0.55, -0.1), SdfPrimitive::Sphere { radius: 0.55 }, "cornell_blue"),
    ];

    for (local, order, transform, prim, mat_name) in volumes {
        world.spawn((
            LocalId(local),
            transform,
            prim,
            SdfOp { kind: CsgKind::Union, smoothing: 0.0 },
            SdfOrder(order),
            SdfMaterialSource { asset: mat(mat_name), ..default() },
            SdfVolume,
            Node3D,
            SceneEntity,
        ));
    }

    // A dim directional fill so the scene isn't pitch-black before GI converges / when GI is off; the
    // ceiling emissive + bounces are the intended look, so keep it low.
    world.spawn((
        LocalId(9),
        Name::new("Fill Light"),
        DirectionalLight { illuminance: 1200.0, shadows_enabled: false, ..default() },
        Transform::from_rotation(Quat::from_rotation_x(-1.1)),
        Node3D,
        EditorGizmo::DirectionalLight { scale: 1.0 },
        SceneEntity,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::reflect::TypeRegistry;

    fn cornell_registry() -> TypeRegistry {
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

    /// Generate the Cornell GI demo scene file. Run with:
    /// `cargo test -- generate_cornell_scene --nocapture`
    #[test]
    fn generate_cornell_scene() {
        let registry = cornell_registry();
        let mut world = World::new();
        spawn_cornell(&mut world);

        let ron = crate::soul_scene::save_scene_to_string(&mut world, &registry)
            .expect("serialize cornell scene");

        std::fs::create_dir_all("assets/scenes").expect("create assets/scenes");
        std::fs::write("assets/scenes/cornell.scene", &ron).expect("write cornell.scene");
        println!("wrote assets/scenes/cornell.scene:\n{ron}");
    }
}
