//! **LOD showcase scene** for the mesh-bake clipmap (Phase 3). A spiral of varied primitives at
//! exponentially increasing distance from the origin AND exponentially increasing size, so each object
//! sits in a successive 2:1 LOD ring: small, detailed objects near the camera (LOD 0) grading out to a
//! huge, coarse object far away (≈ LOD 8). Because size ∝ distance, every object subtends roughly the
//! same screen angle — the whole point of a clipmap (constant screen-space detail across LODs).
//!
//! Spiralled (golden angle) rather than co-linear so the objects don't occlude each other from the
//! centre. Fly the editor camera out along the spiral and toggle **Colour by LOD** in the Mesh Bake
//! panel to watch the rings; tune **LOD-0 radius** to slide the LOD transitions across the objects.
//!
//! Test-only GENERATOR (like [`super::gallery`] / [`super::mesh_test`]): the runtime loads the
//! serialized `assets/scenes/lod_test.scene`. Regenerate with:
//! `cargo test -- generate_lod_test_scene --nocapture`.

use bevy::prelude::*;

use crate::node::{EditorGizmo, Node3D};
use crate::scene_manager::SceneEntity;
use crate::soul_scene::LocalId;

use super::{CsgKind, SdfMaterialSource, SdfOp, SdfOrder, SdfPrimitive, SdfVolume};

/// Number of objects = LOD levels showcased (LOD 0..=8).
const LEVELS: u32 = 9;

/// A material file path relative to `assets/`, for [`SdfMaterialSource::asset`].
fn mat(name: &str) -> Option<std::path::PathBuf> {
    Some(std::path::PathBuf::from(format!("materials/{name}.material.ron")))
}

/// Spawn the LOD showcase (a baseline heightmap + one object per LOD ring + a directional light) with
/// stable `LocalId`s.
fn spawn_lod_test(world: &mut World) {
    // Baseline terrain: a large procedural heightmap spanning every LOD ring — a single CONTINUOUS surface
    // crossing all fine↔coarse boundaries, so it's the strongest cross-LOD seam stress-test (far better than
    // the isolated spiral objects). `SdfPrimitive::Heightmap` evaluates through the CPU `fold_csg` path the
    // mesh bake uses (see `tower_field.rs`). half_xz covers d≈1024 + r≈256; amp ≤ max_height/2 keeps the
    // surface inside the primitive's [0,max_height] AABB.
    world.spawn((
        LocalId(0),
        Name::new("Terrain"),
        Transform::from_xyz(0.0, -20.0, 0.0),
        SdfPrimitive::Heightmap {
            half_xz: Vec2::new(1300.0, 1300.0),
            max_height: 40.0,
            freq: 0.01,
            amp: 15.0,
            seed: 1337,
        },
        SdfOp { kind: CsgKind::Union, smoothing: 0.0 },
        SdfOrder(0),
        SdfMaterialSource { asset: mat("sand"), ..default() },
        SdfVolume,
        Node3D,
        SceneEntity,
    ));

    // Per-level material (cycled), chosen for visible PBR variety: metal / gloss / gold / textured / emissive.
    let mats = [
        "red_metal",
        "white_gloss",
        "gold_rough",
        "cobble",
        "sand",
        "emissive_orange",
        "red_metal",
        "white_gloss",
        "gold_rough",
    ];

    for l in 0..LEVELS {
        let scale = (1u32 << l) as f32; // 2^l
        let d = 4.0 * scale; // centre distance from origin → ~ring L (with default lod0_radius)
        let r = 1.0 * scale; // size ∝ distance → constant screen angle; fits inside ring L's annulus
        let a = l as f32 * 2.399_963_2; // golden angle → consecutive objects well separated, no occlusion
        let pos = Vec3::new(d * a.cos(), 0.0, d * a.sin());

        // Cycle primitive types for variety.
        let prim = match l % 5 {
            0 => SdfPrimitive::Sphere { radius: r },
            1 => SdfPrimitive::Box { half_extents: Vec3::splat(r) },
            2 => SdfPrimitive::Cylinder { radius: r, half_height: r },
            3 => SdfPrimitive::Torus { major: r, minor: r * 0.4 },
            _ => SdfPrimitive::Capsule { half_height: r, radius: r * 0.5 },
        };

        // ids 1..=LEVELS (terrain took id 0); ordered after the terrain so objects union on top of it.
        world.spawn((
            LocalId(l as u64 + 1),
            Transform::from_translation(pos),
            prim,
            SdfOp { kind: CsgKind::Union, smoothing: 0.0 },
            SdfOrder(l + 1),
            SdfMaterialSource { asset: mat(mats[l as usize]), ..default() },
            SdfVolume,
            Node3D,
            SceneEntity,
        ));
    }

    // Directional light so the lit PBR meshes (and debug wireframes) read.
    world.spawn((
        LocalId(LEVELS as u64 + 1),
        Name::new("Directional Light"),
        DirectionalLight { illuminance: 10000.0, shadows_enabled: false, ..default() },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.6, 0.5, 0.0)),
        Node3D,
        EditorGizmo::DirectionalLight { scale: 1.0 },
        SceneEntity,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::reflect::TypeRegistry;

    /// Type registry covering every component + field type the scene serializes (mirrors `mesh_test`).
    fn lod_test_registry() -> TypeRegistry {
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
        r.register::<Vec2>();
        r.register::<Vec3>();
        r.register::<Quat>();
        r.register::<Color>();
        r
    }

    /// Generate the LOD showcase scene file. Run with:
    /// `cargo test -- generate_lod_test_scene --nocapture`
    #[test]
    fn generate_lod_test_scene() {
        let registry = lod_test_registry();
        let mut world = World::new();
        spawn_lod_test(&mut world);

        let ron = crate::soul_scene::save_scene_to_string(&mut world, &registry)
            .expect("serialize lod_test scene");

        std::fs::create_dir_all("assets/scenes").expect("create assets/scenes");
        std::fs::write("assets/scenes/lod_test.scene", &ron).expect("write lod_test.scene");
        println!("wrote assets/scenes/lod_test.scene:\n{ron}");
    }
}
