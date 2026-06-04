//! The SDF stress scene: a procedural cube-tower field for perf/correctness testing. Authored as
//! a single [`TowerSpawner`] node (plus a light) rather than thousands of serialized volumes — the
//! spawner carries the scatter parameters and expands into the full tower field at load time, so
//! `assets/scenes/stress.scene` stays tiny while the runtime world holds ~3000 towers.
//!
//! The same [`super::tower_field::tower_field_edits`] builder feeds the bake-cache regression test, so the
//! stressed scene and the test exercise byte-identical geometry.

use bevy::prelude::*;

use crate::node::Node3D;
use crate::scene_manager::SceneEntity;
// `EditorGizmo` + `LocalId` are only used by the test-only `spawn_stress` scene generator below.
#[cfg(test)]
use crate::node::EditorGizmo;
#[cfg(test)]
use crate::soul_scene::LocalId;

use super::edits::{MaterialFields, SdfMaterialSource, SdfOp};
use super::tower_field::{tower_field_edits, TowerFieldParams, TowerRole};
use super::{CsgKind, SdfVolume};

/// A node that procedurally spawns a scattered cube-tower field on scene load. Holds the
/// [`TowerFieldParams`] (flattened into reflectable scalar fields) plus the material file names for
/// each [`TowerRole`]. An expansion system ([`expand_tower_spawners`]) detects a newly-added
/// spawner and spawns the tower volumes as its children, then marks it expanded so it never
/// double-spawns. The spawner itself is the serialized truth in `stress.scene`; the towers are
/// runtime-derived (never serialized — they carry [`NonSerializable`](crate::soul_scene::NonSerializable)).
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component)]
#[require(Node3D)]
pub struct TowerSpawner {
    pub ground_y: f32,
    pub max_height: f32,
    pub freq: f32,
    pub amp: f32,
    pub seed: u32,
    pub half_extent: f32,
    pub spacing: f32,
    pub jitter: f32,
    pub cubes_per_tower: u32,
    pub cube_half: f32,
    /// Material file (relative to `assets/`) for the ground / cube / cap roles, e.g. `sand`.
    pub ground_mat: String,
    pub cube_mat: String,
    pub cap_mat: String,
}

impl Default for TowerSpawner {
    fn default() -> Self {
        let p = TowerFieldParams::default();
        Self {
            ground_y: p.ground_y,
            max_height: p.max_height,
            freq: p.freq,
            amp: p.amp,
            seed: p.seed,
            half_extent: p.half_extent,
            spacing: p.spacing,
            jitter: p.jitter,
            cubes_per_tower: p.cubes_per_tower,
            cube_half: p.cube_half,
            ground_mat: "sand".to_string(),
            cube_mat: "cobble".to_string(),
            cap_mat: "red_metal".to_string(),
        }
    }
}

impl TowerSpawner {
    fn field_params(&self) -> TowerFieldParams {
        TowerFieldParams {
            ground_y: self.ground_y,
            max_height: self.max_height,
            freq: self.freq,
            amp: self.amp,
            seed: self.seed,
            half_extent: self.half_extent,
            spacing: self.spacing,
            jitter: self.jitter,
            cubes_per_tower: self.cubes_per_tower,
            cube_half: self.cube_half,
        }
    }

    fn mat_for(&self, role: TowerRole) -> &str {
        match role {
            TowerRole::Ground => &self.ground_mat,
            TowerRole::Cube => &self.cube_mat,
            TowerRole::Cap => &self.cap_mat,
        }
    }
}

/// Marks a [`TowerSpawner`] whose field has already been spawned, so the expansion system is
/// idempotent across frames / re-runs. Not serialized (the spawner re-expands on each load).
#[derive(Component)]
pub struct TowerSpawnerExpanded;

/// Expand every not-yet-expanded [`TowerSpawner`] into its tower-field volumes, parented to the
/// spawner, then tag it [`TowerSpawnerExpanded`] so it never re-expands. Runs in `Update` so it
/// picks up spawners a scene load inserts (the `Without` filter makes it idempotent across frames).
/// The spawned volumes carry [`SdfMaterialSource`] (resolved to a GPU material id by
/// `resolve_materials`) and are tagged [`NonSerializable`] so re-saving the scene keeps just the
/// compact spawner node, not the thousands of expanded towers.
pub fn expand_tower_spawners(
    mut commands: Commands,
    spawners: Query<(Entity, &TowerSpawner), Without<TowerSpawnerExpanded>>,
) {
    for (entity, spawner) in &spawners {
        let edits = tower_field_edits(&spawner.field_params());
        // One point light per tower (at its cap), to stress the point-light path + the world-space
        // light grid at scale (~3000 lights). Per-tower hue makes the per-tower pools visually
        // distinct so the culling is easy to verify. NonSerializable, like the towers.
        let mut cap_index: u32 = 0;
        for (order, transform, prim, role) in edits {
            let asset = Some(std::path::PathBuf::from(format!(
                "materials/{}.material.ron",
                spawner.mat_for(role)
            )));
            commands.spawn((
                transform,
                prim,
                SdfOp { kind: CsgKind::Union, smoothing: 0.0 },
                order,
                SdfMaterialSource { asset, overrides: MaterialFields::default() },
                SdfVolume,
                Node3D,
                SceneEntity,
                crate::soul_scene::NonSerializable,
                ChildOf(entity),
            ));
            if role == TowerRole::Cap {
                let idx = cap_index;
                cap_index += 1;
                // Golden-ratio hue walk for an even rainbow spread across the field.
                let hue = (idx as f32 * 0.618_034).fract() * 360.0;
                // Deterministic per-tower offset: push the light off to a random side + above the
                // cap (not straight overhead) so each tower throws a real DIRECTIONAL shadow, in a
                // different direction per tower. Hash the cap index so it's stable across reloads.
                let h = idx.wrapping_mul(2_654_435_761);
                let ang = (h & 0xffff) as f32 / 65_535.0 * std::f32::consts::TAU;
                let rad = 2.0 + ((h >> 16) & 0xff) as f32 / 255.0 * 2.5; // 2.0 .. 4.5 m to the side
                let up = 2.5 + ((h >> 24) & 0xff) as f32 / 255.0 * 2.0; //  2.5 .. 4.5 m above the cap
                let offset = Vec3::new(ang.cos() * rad, up, ang.sin() * rad);
                commands.spawn((
                    Name::new("Tower Light"),
                    PointLight {
                        color: Color::hsl(hue, 0.85, 0.55),
                        // Punchy on purpose — this is a stress DEMO, so each tower should throw an
                        // obvious coloured pool (physical lumens; candela = intensity/4π). Range
                        // ~tower-spacing so a light reaches its tower + the ground around it.
                        intensity: 1_000_000.0,
                        range: 10.0,
                        // Source SIZE (sphere): soft shadow edge + no inverse-square singularity.
                        radius: 1.5,
                        shadows_enabled: false,
                        ..default()
                    },
                    // Offset off to a side + above the cap so the tower casts a directional shadow.
                    Transform::from_translation(transform.translation + offset),
                    // Full editor node: `Node3D` (→ SceneNode) puts it in the scene tree;
                    // `EditorGizmo::PointLight` makes it viewport-pickable + draws the bulb/range
                    // rings. The gizmo DRAW is camera-distance-culled (node_gizmos) so thousands of
                    // these don't tank the editor; picking is click-time so far lights still select.
                    Node3D,
                    crate::node::EditorGizmo::PointLight { scale: 1.0 },
                    SceneEntity,
                    crate::soul_scene::NonSerializable,
                    ChildOf(entity),
                ));
            }
        }
        // The spawned `PointLight` children carry `InheritedVisibility` (a `PointLight` required
        // component); Bevy's visibility propagation warns (B0004) if their parent lacks the
        // visibility chain. Give the spawner a default `Visibility` so the parent is a valid
        // visibility root for its light children.
        commands
            .entity(entity)
            .insert((TowerSpawnerExpanded, Visibility::default()));
    }
}

/// Spawn the stress scene (one [`TowerSpawner`] node + a directional light) into `world` with
/// stable `LocalId`s, ready for serialization. The towers themselves are materialized at load time
/// by [`expand_tower_spawners`], not stored here. Test-only: the runtime loads the serialized
/// `assets/scenes/stress.scene`; this builder only regenerates that file (see the test).
#[cfg(test)]
pub fn spawn_stress(world: &mut World) {
    world.spawn((
        LocalId(0),
        Name::new("Tower Field"),
        TowerSpawner::default(),
        Transform::IDENTITY,
        Node3D,
        EditorGizmo::Axes { scale: 1.0 },
        SceneEntity,
    ));

    world.spawn((
        LocalId(1),
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

    /// Type registry covering every component the stress scene serializes.
    fn stress_registry() -> TypeRegistry {
        let mut r = TypeRegistry::new();
        r.register::<Transform>();
        r.register::<Name>();
        r.register::<SceneEntity>();
        r.register::<crate::node::SceneNode>();
        r.register::<Node3D>();
        r.register::<EditorGizmo>();
        r.register::<TowerSpawner>();
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
        r.register::<String>();
        r
    }

    /// Generate the default stress scene file. Run with:
    /// `cargo test -- generate_stress_scene --nocapture`
    #[test]
    fn generate_stress_scene() {
        let registry = stress_registry();
        let mut world = World::new();
        spawn_stress(&mut world);

        let ron = crate::soul_scene::save_scene_to_string(&mut world, &registry)
            .expect("serialize stress scene");

        std::fs::create_dir_all("assets/scenes").expect("create assets/scenes");
        std::fs::write("assets/scenes/stress.scene", &ron).expect("write stress.scene");
        println!("wrote assets/scenes/stress.scene:\n{ron}");
    }

    /// The expansion is deterministic and produces the ground + the expected tower-edit count.
    #[test]
    fn tower_field_has_ground_and_towers() {
        let edits = tower_field_edits(&TowerFieldParams::default());
        assert!(edits.len() > 1000, "stress field should be large (got {})", edits.len());
        assert_eq!(edits[0].3, TowerRole::Ground, "first edit must be the ground");
        assert!(edits.iter().any(|(_, _, _, r)| *r == TowerRole::Cap), "must have capping spheres");
    }

    /// Adding a `TowerSpawner` and running `expand_tower_spawners` materializes the full tower
    /// field as child volumes, exactly once (idempotent on re-run).
    #[test]
    fn spawner_expands_into_volumes_once() {
        use super::SdfVolume;
        let mut app = App::new();
        app.add_systems(Update, expand_tower_spawners);

        // A small spawner so the test is fast but still multi-tower.
        let small = TowerSpawner {
            half_extent: 30.0,
            spacing: 10.0,
            ..default()
        };
        let expected = tower_field_edits(&small.field_params()).len();
        let spawner = app.world_mut().spawn(small).id();

        app.update();
        let volumes_after_first = app
            .world_mut()
            .query::<&SdfVolume>()
            .iter(app.world())
            .count();
        assert_eq!(volumes_after_first, expected, "expansion must spawn one volume per tower edit");
        assert!(
            app.world().get::<TowerSpawnerExpanded>(spawner).is_some(),
            "spawner must be marked expanded"
        );

        // Re-run: the `Without<Expanded>` guard must prevent a second expansion.
        app.update();
        let volumes_after_second = app
            .world_mut()
            .query::<&SdfVolume>()
            .iter(app.world())
            .count();
        assert_eq!(volumes_after_second, expected, "expansion must be idempotent");
    }

    /// Expansion spawns exactly one `PointLight` per tower (per `TowerRole::Cap`), tagged
    /// `SceneEntity` (so the SDF light extraction picks it up) and positioned at the cap.
    #[test]
    fn spawner_spawns_a_point_light_per_tower() {
        let mut app = App::new();
        app.add_systems(Update, expand_tower_spawners);

        let small = TowerSpawner {
            half_extent: 30.0,
            spacing: 10.0,
            ..default()
        };
        let caps = tower_field_edits(&small.field_params())
            .iter()
            .filter(|(_, _, _, r)| *r == TowerRole::Cap)
            .count();
        assert!(caps > 0, "the small field must have towers");
        app.world_mut().spawn(small);
        app.update();

        let lights = app
            .world_mut()
            .query_filtered::<&PointLight, With<SceneEntity>>()
            .iter(app.world())
            .count();
        assert_eq!(lights, caps, "one SceneEntity point light per tower cap");
    }
}
