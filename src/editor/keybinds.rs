//! Editor keyboard shortcuts (jackdaw `keybinds`). Drives the transform-gizmo
//! manipulator's [`GizmoOptions`]: W/E/R restrict the gizmo to translate/rotate/
//! scale, Q shows all modes at once; Ctrl toggles snapping. Runs only in the SDF
//! editor scene.

use bevy::prelude::*;
use transform_gizmo_bevy::{GizmoMode, GizmoOptions};

/// Register editor keybinds.
pub fn plugin(app: &mut App) {
    app.add_systems(
        Update,
        gizmo_mode_keys.run_if(in_state(crate::scene_manager::AppScene::SdfEditor)),
    );
}

/// W = translate, E = rotate, R = scale, Q = all modes. Ctrl held = snapping on.
fn gizmo_mode_keys(keyboard: Res<ButtonInput<KeyCode>>, mut options: ResMut<GizmoOptions>) {
    if keyboard.just_pressed(KeyCode::KeyW) {
        options.gizmo_modes = GizmoMode::all_translate();
    } else if keyboard.just_pressed(KeyCode::KeyE) {
        options.gizmo_modes = GizmoMode::all_rotate();
    } else if keyboard.just_pressed(KeyCode::KeyR) {
        options.gizmo_modes = GizmoMode::all_scale();
    } else if keyboard.just_pressed(KeyCode::KeyQ) {
        options.gizmo_modes = GizmoMode::all();
    }

    // Snapping engaged while either Ctrl is held (Blender-style).
    options.snapping =
        keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight);
}
