//! Stage 6 — **voxel physics: walk the cubes.**
//!
//! The player drops into the voxel world in **first-person** and walks on the cubes, colliding with
//! them. We use the engine-agnostic [`rapier3d`] solver directly (no `bevy_rapier` — it has no Bevy-0.19
//! build and would be another vendored fork): a kinematic character cuboid is moved through a static set
//! of colliders built from the voxel occupancy.
//!
//! # Why first-person (no player mesh)
//! `StandardMaterial`'s bindless PBR shader is broken on the wgpu-trunk fork, so a capsule mesh wouldn't
//! render anyway. First-person sidesteps that entirely: the [`SdfCamera`] IS the player's eyes, so there
//! is no mesh to draw — and walking through the Cornell box demonstrates collide-and-slide cleanly.
//!
//! # Colliders from voxels — greedy cuboids (not one box per voxel)
//! A Cornell box is ~222k solid voxels; one collider each would be absurd. [`greedy_boxes`] decomposes a
//! brick's `8³` occupancy into a small set of **non-overlapping axis-aligned boxes** that exactly cover
//! the solid voxels (a uniform-solid brick → a single box). Each box becomes one world-space cuboid
//! collider. This is the plan's "greedy cuboid compound colliders", bounded + edit-friendly (only the
//! edited bricks need re-boxing — currently we rebuild the whole static Cornell set on an edit, which is
//! cheap for the small box).
//!
//! # SSOT
//! The collider world is rebuilt from the SAME geometry SSOT the renderer traces:
//! [`build_cornell_with_edits`](super::cornell::build_cornell_with_edits)`(registry, edits)`. There is no
//! second copy of the world — the colliders are a pure function of `(registry, edits)`, exactly like the
//! packed GPU bricks. (Worldgen physics — colliders from the streamed resident set — is a later
//! extension; for now walk mode is gated to the static Cornell scene.)

use bevy::math::IVec3;
use bevy::prelude::*;

use rapier3d::control::{CharacterAutostep, CharacterLength, KinematicCharacterController};
use rapier3d::geometry::BroadPhaseBvh;
use rapier3d::math::{Pose, Vector as RVec};
use rapier3d::parry::shape::Cuboid as RCuboid;
use rapier3d::prelude::{
    CCDSolver, ColliderBuilder, ColliderSet, ImpulseJointSet, IntegrationParameters, IslandManager,
    MultibodyJointSet, NarrowPhase, PhysicsPipeline, QueryFilter, RigidBodySet,
};

use super::VoxelScene;
use super::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, BrickMap, VOXEL_SIZE};
use super::cornell::build_cornell_with_edits;
use super::edits::VoxelEdits;
use super::palette::BlockRegistry;
use crate::sdf_render::editor_camera::SdfCameraMode;
use crate::sdf_render::{SdfCamera, SdfOrbitCamera};

// --- Tunables (first-person walk) ---
const GRAVITY: f32 = 22.0; // m/s²
const JUMP_SPEED: f32 = 7.0; // m/s
const WALK_SPEED: f32 = 6.0; // m/s
const EYE_HEIGHT: f32 = 1.6; // camera height above the player's feet (m)
/// Character cuboid half-extents: 0.5 m wide × 1.8 m tall (Minecraft-ish AABB body).
const PLAYER_HALF: Vec3 = Vec3::new(0.25, 0.9, 0.25);

// =====================================================================================================
// Greedy box decomposition of a brick's occupancy
// =====================================================================================================

/// A box of solid voxels in a brick's LOCAL voxel grid: inclusive `min..=max` on each axis.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VoxelBox {
    pub min: IVec3,
    pub max: IVec3,
}

/// Decompose a brick's `8³` occupancy into a small set of **non-overlapping** axis-aligned boxes that
/// EXACTLY cover its solid voxels. Greedy growth per seed voxel: extend along +X as far as solid, then
/// grow that row along +Y while the whole row stays solid, then grow that slab along +Z while the whole
/// slab stays solid; mark the box consumed and continue. A uniform-solid brick collapses to ONE box.
///
/// Guarantees (the physics-equivalence contract, asserted in tests): the boxes are pairwise disjoint and
/// their union is exactly the set of solid voxels — so the cuboid colliders neither miss a solid cell nor
/// double-cover one.
pub fn greedy_boxes(brick: &Brick) -> Vec<VoxelBox> {
    let e = BRICK_EDGE;
    let idx = |x: i32, y: i32, z: i32| (x + y * e + z * e * e) as usize;
    let mut consumed = [false; BRICK_VOXELS];
    let mut boxes = Vec::new();

    for z in 0..e {
        for y in 0..e {
            for x in 0..e {
                if consumed[idx(x, y, z)] || !brick.is_solid(x, y, z) {
                    continue;
                }
                // Grow +X across solid, unconsumed voxels.
                let mut x1 = x;
                while x1 + 1 < e && !consumed[idx(x1 + 1, y, z)] && brick.is_solid(x1 + 1, y, z) {
                    x1 += 1;
                }
                // Grow +Y while the whole x-row [x..=x1] is solid + unconsumed.
                let mut y1 = y;
                'gy: while y1 + 1 < e {
                    for xx in x..=x1 {
                        if consumed[idx(xx, y1 + 1, z)] || !brick.is_solid(xx, y1 + 1, z) {
                            break 'gy;
                        }
                    }
                    y1 += 1;
                }
                // Grow +Z while the whole xy-slab [x..=x1]×[y..=y1] is solid + unconsumed.
                let mut z1 = z;
                'gz: while z1 + 1 < e {
                    for yy in y..=y1 {
                        for xx in x..=x1 {
                            if consumed[idx(xx, yy, z1 + 1)] || !brick.is_solid(xx, yy, z1 + 1) {
                                break 'gz;
                            }
                        }
                    }
                    z1 += 1;
                }
                // Consume the box and emit it.
                for zz in z..=z1 {
                    for yy in y..=y1 {
                        for xx in x..=x1 {
                            consumed[idx(xx, yy, zz)] = true;
                        }
                    }
                }
                boxes.push(VoxelBox {
                    min: IVec3::new(x, y, z),
                    max: IVec3::new(x1, y1, z1),
                });
            }
        }
    }
    boxes
}

// =====================================================================================================
// The static collider world
// =====================================================================================================

/// The rapier collider world built from the resident voxels: a [`ColliderSet`] of greedy cuboids plus the
/// broad/narrow-phase acceleration the character shape-casts query. Static — rebuilt only when the
/// geometry changes (a build/destroy edit), never stepped per frame (nothing in the set moves; only the
/// character does, via [`move_character`](Self::move_character)).
#[derive(Resource)]
pub struct VoxelColliders {
    bodies: RigidBodySet,
    colliders: ColliderSet,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    /// The `(scene, edit-generation)` the current colliders were built for; `None` until first build.
    /// We rebuild only when this changes, so a still player on unedited geometry does zero work.
    built_for: Option<(VoxelScene, u64)>,
    /// Number of cuboid colliders (greedy boxes) in the current world — diagnostics.
    pub box_count: usize,
}

impl Default for VoxelColliders {
    fn default() -> Self {
        Self {
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            built_for: None,
            box_count: 0,
        }
    }
}

impl VoxelColliders {
    /// Rebuild the static collider world from a brickmap: greedy-box every brick's occupancy into
    /// world-space cuboid colliders, then run ONE zero-gravity physics step to populate the broad/narrow
    /// phase so scene queries (the character's shape-casts) work. Clears any previous world first.
    pub fn rebuild_from_bricks(&mut self, map: &BrickMap) {
        self.bodies = RigidBodySet::new();
        self.colliders = ColliderSet::new();
        self.broad_phase = BroadPhaseBvh::new();
        self.narrow_phase = NarrowPhase::new();

        let mut count = 0usize;
        for (bc, brick) in map.iter() {
            let brick_origin_voxel = *bc * BRICK_EDGE; // world voxel of the brick's min corner
            for b in greedy_boxes(brick) {
                // World-metre AABB of the box: voxel span [vmin, vmax + 1) → metres.
                let vmin = brick_origin_voxel + b.min;
                let vmax = brick_origin_voxel + b.max;
                let min_m = vmin.as_vec3() * VOXEL_SIZE;
                let max_m = (vmax + IVec3::ONE).as_vec3() * VOXEL_SIZE;
                let he = (max_m - min_m) * 0.5;
                let center = (min_m + max_m) * 0.5;
                self.colliders.insert(
                    ColliderBuilder::cuboid(he.x, he.y, he.z)
                        .translation(RVec::new(center.x, center.y, center.z)),
                );
                count += 1;
            }
        }
        self.box_count = count;

        // One step (zero gravity) populates the broad-phase BVH + narrow-phase so `as_query_pipeline`
        // returns a usable query system. Nothing is dynamic, so this neither moves nor wakes anything.
        PhysicsPipeline::new().step(
            RVec::ZERO,
            &IntegrationParameters::default(),
            &mut IslandManager::new(),
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut ImpulseJointSet::new(),
            &mut MultibodyJointSet::new(),
            &mut CCDSolver::new(),
            &(),
            &(),
        );
    }

    /// Move a character cuboid through the static world with collide-and-slide, returning the new FEET
    /// position + whether the character is grounded. `feet` is the world position of the cuboid's base;
    /// `half` its half-extents; `desired` the attempted translation this step.
    pub fn move_character(
        &self,
        controller: &KinematicCharacterController,
        feet: Vec3,
        half: Vec3,
        desired: Vec3,
        dt: f32,
    ) -> (Vec3, bool) {
        let shape = RCuboid::new(RVec::new(half.x, half.y, half.z));
        // The cuboid is centred on its origin, so the character centre is half a body above the feet.
        let center = feet + Vec3::Y * half.y;
        let pose = Pose::from_translation(RVec::new(center.x, center.y, center.z));
        let qp = self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.bodies,
            &self.colliders,
            QueryFilter::default(),
        );
        let mv = controller.move_shape(
            dt,
            &qp,
            &shape,
            &pose,
            RVec::new(desired.x, desired.y, desired.z),
            |_| {},
        );
        (feet + Vec3::new(mv.translation.x, mv.translation.y, mv.translation.z), mv.grounded)
    }
}

/// The kinematic controller config shared by the move system + tests: slide along walls, auto-step one
/// 0.2 m voxel, snap to ground. (Cheap to build per call — it is plain data.)
pub fn walk_controller() -> KinematicCharacterController {
    KinematicCharacterController {
        offset: CharacterLength::Absolute(0.01),
        slide: true,
        autostep: Some(CharacterAutostep {
            // One voxel is 0.2 m; allow stepping up a single voxel ledge.
            max_height: CharacterLength::Absolute(0.25),
            min_width: CharacterLength::Absolute(0.05),
            include_dynamic_bodies: false,
        }),
        snap_to_ground: Some(CharacterLength::Absolute(0.1)),
        ..default()
    }
}

// =====================================================================================================
// First-person walk player (drives the SdfCamera)
// =====================================================================================================

/// First-person walk state. While [`SdfCameraMode::player`] is on, the [`SdfCamera`] is driven from this:
/// `feet` is the player's world base position, `vy` the integrated vertical velocity (gravity + jump), and
/// `yaw`/`pitch` the look direction (hold RMB + drag to look).
#[derive(Resource, Default)]
pub struct VoxelWalk {
    pub feet: Vec3,
    pub vy: f32,
    pub yaw: f32,
    pub pitch: f32,
}

/// `P` toggles first-person walk mode (Cornell scene only — see module docs). On enable it drops the
/// player onto the floor under the orbit target and seeds the look heading; on disable the orbit camera
/// resumes. Refused (with a one-shot log) outside the Cornell scene until worldgen physics lands.
pub fn toggle_walk_mode(
    keyboard: Res<ButtonInput<KeyCode>>,
    scene: Res<VoxelScene>,
    mut mode: ResMut<SdfCameraMode>,
    mut walk: ResMut<VoxelWalk>,
    orbit: Res<SdfOrbitCamera>,
) {
    if !keyboard.just_pressed(KeyCode::KeyP) {
        return;
    }
    if mode.player {
        mode.player = false;
        info!("voxel walk mode: OFF (orbit camera)");
        return;
    }
    if !scene.is_cornell() {
        info!("voxel walk mode: only the Cornell scene has physics for now — press V for Cornell first");
        return;
    }
    // Drop in above the floor under the current orbit target (the box interior); gravity settles it.
    walk.feet = Vec3::new(orbit.target.x, super::cornell::INTERIOR as f32 * VOXEL_SIZE * 0.5, orbit.target.z);
    walk.vy = 0.0;
    // Seed the look heading from the orbit yaw so the view doesn't snap.
    walk.yaw = orbit.yaw;
    walk.pitch = -0.1;
    mode.player = true;
    info!("voxel walk mode: ON (WASD walk, Space jump, hold RMB to look, P to exit)");
}

/// Rebuild the collider world when walk mode is active and the geometry changed (first activation or a
/// build/destroy edit). Pure function of `(scene, registry, edits)` — the same SSOT the renderer traces.
pub fn rebuild_walk_colliders(
    mode: Res<SdfCameraMode>,
    scene: Res<VoxelScene>,
    edits: Res<VoxelEdits>,
    mut colliders: ResMut<VoxelColliders>,
) {
    if !mode.player || !scene.is_cornell() {
        return;
    }
    let want = (*scene, edits.generation());
    if colliders.built_for == Some(want) {
        return; // up to date
    }
    let map = build_cornell_with_edits(&BlockRegistry::cornell(), &edits);
    colliders.rebuild_from_bricks(&map);
    colliders.built_for = Some(want);
    debug!("voxel physics: rebuilt {} cuboid colliders (edit gen {})", colliders.box_count, edits.generation());
}

/// First-person walk: hold RMB to look (yaw/pitch), WASD to move relative to the look heading, Space to
/// jump, gravity always pulling — moved through the voxel colliders with collide-and-slide. Drives the
/// [`SdfCamera`] transform to the eye position each frame.
#[allow(clippy::too_many_arguments)] // a Bevy system; an arg struct would hurt readability here
pub fn walk_player(
    mode: Res<SdfCameraMode>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut motion: MessageReader<bevy::input::mouse::MouseMotion>,
    time: Res<Time>,
    colliders: Res<VoxelColliders>,
    mut walk: ResMut<VoxelWalk>,
    mut cam: Query<&mut Transform, With<SdfCamera>>,
) {
    if !mode.player {
        motion.clear();
        return;
    }
    let Ok(mut cam_t) = cam.single_mut() else {
        return;
    };

    // --- Look (hold RMB + drag) ---
    if mouse.pressed(MouseButton::Right) {
        let d: Vec2 = motion.read().map(|m| m.delta).sum();
        walk.yaw -= d.x * 0.005;
        walk.pitch = (walk.pitch + d.y * 0.005).clamp(-1.5, 1.5);
    } else {
        motion.clear();
    }

    let dt = time.delta_secs();

    // --- Gravity + jump (vy integrated; reset on ground) ---
    let grounded_now = colliders_grounded(&colliders, walk.feet);
    if keyboard.just_pressed(KeyCode::Space) && grounded_now {
        walk.vy = JUMP_SPEED;
    }
    if grounded_now && walk.vy < 0.0 {
        walk.vy = 0.0;
    }
    walk.vy -= GRAVITY * dt;

    // --- Horizontal WASD relative to the look yaw (XZ plane) ---
    let (s, c) = walk.yaw.sin_cos();
    let forward = Vec3::new(-s, 0.0, -c); // matches forward_from_yaw convention
    let right = Vec3::new(-forward.z, 0.0, forward.x);
    let mut dir = Vec3::ZERO;
    if keyboard.pressed(KeyCode::KeyW) {
        dir += forward;
    }
    if keyboard.pressed(KeyCode::KeyS) {
        dir -= forward;
    }
    if keyboard.pressed(KeyCode::KeyD) {
        dir += right;
    }
    if keyboard.pressed(KeyCode::KeyA) {
        dir -= right;
    }
    if dir.length_squared() > 0.0 {
        dir = dir.normalize();
    }

    let desired = dir * WALK_SPEED * dt + Vec3::Y * walk.vy * dt;
    let (new_feet, grounded) = colliders.move_character(&walk_controller(), walk.feet, PLAYER_HALF, desired, dt);
    walk.feet = new_feet;
    if grounded && walk.vy < 0.0 {
        walk.vy = 0.0;
    }

    // --- Drive the camera to the eye position, looking along yaw/pitch ---
    let eye = walk.feet + Vec3::Y * EYE_HEIGHT;
    let cp = walk.pitch.cos();
    let look = Vec3::new(-s * cp, walk.pitch.sin(), -c * cp);
    cam_t.translation = eye;
    cam_t.look_to(look, Vec3::Y);
}

/// Cheap "is the player resting on something" probe: try a tiny downward move and see if the controller
/// reports grounded. Used to gate jump + zero out downward velocity before integrating the real move.
fn colliders_grounded(colliders: &VoxelColliders, feet: Vec3) -> bool {
    let (_, grounded) = colliders.move_character(&walk_controller(), feet, PLAYER_HALF, Vec3::new(0.0, -0.01, 0.0), 1.0 / 60.0);
    grounded
}

#[cfg(test)]
mod tests;
