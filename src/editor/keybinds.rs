//! Editor keyboard shortcuts (jackdaw `keybinds`). Drives the in-tree gizmo's
//! [`GizmoState`]: W/E/R restrict the gizmo to translate/rotate/scale, Q shows all
//! modes; Ctrl toggles snapping. Runs only in the SDF editor scene.

use bevy::prelude::*;

use crate::sdf_render::gizmo::{GizmoModes, GizmoState};

/// Register editor keybinds.
pub fn plugin(app: &mut App) {
    app.add_systems(
        Update,
        gizmo_mode_keys.run_if(in_state(crate::scene_manager::AppScene::SdfEditor)),
    );
}

/// W = translate, E = rotate, R = scale, Q = all modes. Ctrl held = snapping on.
fn gizmo_mode_keys(keyboard: Res<ButtonInput<KeyCode>>, mut state: ResMut<GizmoState>) {
    if keyboard.just_pressed(KeyCode::KeyW) {
        state.modes = GizmoModes::TRANSLATE;
    } else if keyboard.just_pressed(KeyCode::KeyE) {
        state.modes = GizmoModes::ROTATE;
    } else if keyboard.just_pressed(KeyCode::KeyR) {
        state.modes = GizmoModes::SCALE;
    } else if keyboard.just_pressed(KeyCode::KeyQ) {
        state.modes = GizmoModes::all();
    }

    // Snapping engaged while either Ctrl is held (Blender-style).
    state.snap = keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight);
}
