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
    spawn_cornell_rooms(world, &[Vec3::ZERO]);
}

/// World-space pitch between adjacent Cornell rooms in the scaling grid. The room footprint is
/// x,z ∈ [-2.2, 2.2] (walls included), so a pitch of 5.0 leaves a ~0.6 gap between rooms — they read as
/// distinct boxes while staying close enough that a moderate `k` spans several clipmap LODs.
const ROOM_PITCH: f32 = 5.0;

/// One Cornell room's 9 volumes, translated to `origin` with ids/orders offset by the room's bases so a
/// grid keeps globally-unique [`LocalId`]s. Returns `(local_id, order, transform, primitive, material)`.
/// The single source of truth for the room geometry — both [`spawn_cornell`] and [`spawn_cornell_grid`]
/// build from it, so the scaling scenes are byte-identical rooms tiled out.
fn cornell_room_volumes(
    origin: Vec3,
    id_base: u64,
    order_base: u32,
) -> Vec<(u64, u32, Transform, SdfPrimitive, &'static str)> {
    let b = |h: Vec3| SdfPrimitive::Box { half_extents: h };
    // (local_id, order, transform, primitive, material file name) — room interior x∈[-2,2], y∈[0,4],
    // z∈[-2,2] (open front toward +z). Wall slabs 0.1 thick.
    let room: [(u64, u32, Transform, SdfPrimitive, &str); 9] = [
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
    room.into_iter()
        .map(|(local, order, t, prim, mat_name)| {
            // Preserve rotation/scale, just shift the room to its grid cell.
            let t = Transform { translation: t.translation + origin, ..t };
            (id_base + local, order_base + order, t, prim, mat_name)
        })
        .collect()
}

/// Spawn one or more Cornell rooms (each at its `origin`) plus a single shared directional fill. Each
/// room reserves a block of 16 [`LocalId`]s (9 volumes use 0..8) so ids never collide across the grid.
fn spawn_cornell_rooms(world: &mut World, origins: &[Vec3]) {
    for (room_idx, &origin) in origins.iter().enumerate() {
        let id_base = room_idx as u64 * 16;
        let order_base = room_idx as u32 * 16;
        for (local, order, transform, prim, mat_name) in
            cornell_room_volumes(origin, id_base, order_base)
        {
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
    }

    // A dim directional fill so the scene isn't pitch-black before GI converges / when GI is off; the
    // ceiling emissives + bounces are the intended look, so keep it low. One fill for the whole grid,
    // its id placed past every room's reserved block.
    world.spawn((
        LocalId(origins.len() as u64 * 16 + 1),
        Name::new("Fill Light"),
        DirectionalLight { illuminance: 1200.0, shadows_enabled: false, ..default() },
        Transform::from_rotation(Quat::from_rotation_x(-1.1)),
        Node3D,
        EditorGizmo::DirectionalLight { scale: 1.0 },
        SceneEntity,
    ));
}

/// Tile a `k × k` grid of Cornell rooms in the x–z plane (centred on the origin, [`ROOM_PITCH`] apart)
/// for DDGI scaling tests — bigger `k` = bigger world. Every room is a full GI box (its own ceiling
/// emitter and colour-bleed objects); with the camera in the middle, near rooms resolve at LOD0 and far
/// rooms at progressively coarser clipmap LODs, exercising the LOD-scaled probe allocation.
fn spawn_cornell_grid(world: &mut World, k: u32) {
    let off = (k as f32 - 1.0) * 0.5 * ROOM_PITCH;
    let mut origins = Vec::with_capacity((k * k) as usize);
    for gz in 0..k {
        for gx in 0..k {
            origins.push(Vec3::new(gx as f32 * ROOM_PITCH - off, 0.0, gz as f32 * ROOM_PITCH - off));
        }
    }
    spawn_cornell_rooms(world, &origins);
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

    /// Generate the Cornell SCALING grid scenes (`cornell{k}.scene` for k = 2, 4, 8, 16, 32) — each a
    /// `k×k` tiling of the room (k=32 ⇒ 1024 rooms), the ever-larger test bed for DDGI scaling. Run with:
    /// `cargo test -- generate_cornell_grid_scenes --nocapture`
    #[test]
    fn generate_cornell_grid_scenes() {
        let registry = cornell_registry();
        std::fs::create_dir_all("assets/scenes").expect("create assets/scenes");
        for k in [2u32, 4, 8, 16, 32] {
            let mut world = World::new();
            spawn_cornell_grid(&mut world, k);
            let ron = crate::soul_scene::save_scene_to_string(&mut world, &registry)
                .unwrap_or_else(|e| panic!("serialize cornell{k} scene: {e:?}"));
            let path = format!("assets/scenes/cornell{k}.scene");
            std::fs::write(&path, &ron).unwrap_or_else(|e| panic!("write {path}: {e:?}"));
            println!("wrote {path} ({k}×{k} = {} rooms)", k * k);
        }
    }

    /// The grid tiles unique `LocalId`s (16 per room) and produces `k×k` rooms' worth of volumes.
    #[test]
    fn cornell_grid_ids_are_unique() {
        use std::collections::HashSet;
        let mut world = World::new();
        spawn_cornell_grid(&mut world, 3);
        let mut ids = HashSet::new();
        let mut q = world.query::<&LocalId>();
        let mut count = 0;
        for id in q.iter(&world) {
            assert!(ids.insert(id.0), "duplicate LocalId {} in cornell grid", id.0);
            count += 1;
        }
        // 3×3 rooms × 9 volumes + 1 shared fill light.
        assert_eq!(count, 3 * 3 * 9 + 1, "unexpected entity count for 3×3 cornell grid");
    }
}
