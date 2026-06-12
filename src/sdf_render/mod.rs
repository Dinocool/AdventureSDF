//! # SDF clipmap scene (voxel-RT rebuild — slimmed)
//!
//! The app is being rebuilt as a voxel ray-tracing engine. The SDF GPU render path, the mesh-bake
//! terrain renderer, the SDF editor, picking/gizmos/overlays, and the legacy scene generators have
//! all been pruned from compilation. What remains here is the reusable core:
//!
//! - [`editor_camera`] — the free-fly + orbit editor cameras.
//! - [`edits`] — the SDF edit primitives + material model. Still consumed by the `assets` material
//!   compile pipeline and by `worldgen` (the `Terrain` primitive), so it stays compiled.
//! - [`worldgen`] — the procedural generation stack (the valuable, reusable logic).
//!
//! [`SdfScenePlugin`] is now minimal: it spawns the persistent editor camera, drives the editor
//! cameras, and adds [`worldgen::WorldGenPlugin`].
//!
//! (`chunk.rs` — the clipmap chunk-addressing math — is retained on disk but no longer compiled:
//! it was tightly coupled to the removed `atlas` / `SdfGridConfig` clipmap, and the future voxel-RT
//! residency scheme will define its own addressing. Re-add the `mod chunk;` decl when that lands.)

pub(crate) mod editor_camera;
pub mod edits;
pub mod worldgen;

use bevy::prelude::*;

use crate::scene_manager::AppScene;

// The editor viewport cameras (orbit + free-fly) live in `editor_camera`. Their public types are
// re-exported here so cross-module consumers keep the stable `sdf_render::` path.
pub use editor_camera::{
    CameraInput, OrbitFocus, SdfCameraMode, SdfOrbitCamera, sync_orbit_camera_transform,
};

// --- Components ---

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
#[require(crate::node::Node3D)]
pub struct SdfVolume;

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct SdfCamera;

/// Whether viewport input (orbit/pick/gizmo-drag) is allowed this frame. The editor sets this from
/// the pointer-in-viewport test so clicks on dock panels don't fall through to the 3D scene.
/// Defaults to `true` so the non-editor build (full-window viewport, no panels) keeps working.
#[derive(Resource)]
pub struct ViewportInputAllowed(pub bool);

impl Default for ViewportInputAllowed {
    fn default() -> Self {
        Self(true)
    }
}

// --- Plugin ---

pub struct SdfScenePlugin;

impl Plugin for SdfScenePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<OrbitFocus>()
            .init_resource::<SdfOrbitCamera>()
            .init_resource::<SdfCameraMode>()
            .init_resource::<ViewportInputAllowed>()
            .register_type::<SdfVolume>()
            .register_type::<SdfCamera>()
            // The viewport camera persists across scene-state transitions (editor infra),
            // spawned once at startup and activated only while in the SDF editor.
            .add_systems(Startup, editor_camera::spawn_editor_camera)
            .add_systems(Update, editor_camera::sync_editor_camera_active)
            // Camera control: skipped when the pointer is over a dock panel (editor
            // sets ViewportInputAllowed). Non-editor build leaves it true.
            .add_systems(
                Update,
                (
                    editor_camera::orbit_camera.run_if(|m: Res<SdfCameraMode>| !m.fps && !m.player),
                    editor_camera::fps_camera.run_if(|m: Res<SdfCameraMode>| m.fps && !m.player),
                )
                    .run_if(in_state(AppScene::SdfEditor))
                    .run_if(|allowed: Res<ViewportInputAllowed>| allowed.0),
            )
            // Focus easing runs even while the pointer is over a dock panel, so a
            // Hierarchy double-click animates the camera without re-entering the viewport.
            .add_systems(
                Update,
                editor_camera::ease_orbit_focus
                    .run_if(in_state(AppScene::SdfEditor))
                    .run_if(|m: Res<SdfCameraMode>| !m.fps && !m.player),
            )
            // Procedural worldgen: owns the LayerManager and rolls the streamed CPU height ring
            // around the camera. The voxel-RT renderer that consumes it will be wired up later.
            .add_plugins(worldgen::WorldGenPlugin);
    }
}
