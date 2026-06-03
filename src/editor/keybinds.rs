//! Editor keyboard shortcuts (jackdaw `keybinds`). Drives the in-tree gizmo's
//! [`GizmoState`]: W/E/R restrict the gizmo to translate/rotate/scale, Q shows all
//! modes; Ctrl toggles snapping. X/Delete despawns the selection. Runs only in the
//! SDF editor scene.

use bevy::prelude::*;

use crate::editor::menu_bar::EditorRequests;
use crate::sdf_render::gizmo::{GizmoModes, GizmoState};
use crate::sdf_render::SdfSelection;

/// Register editor keybinds.
pub struct KeybindsPlugin;

impl Plugin for KeybindsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (gizmo_mode_keys, delete_selection, save_shortcut)
                .run_if(in_state(crate::scene_manager::AppScene::SdfEditor)),
        );
    }
}

/// Ctrl+S saves the active scene (same path as File ▸ Save — the scene-tab manager drains
/// the request, writing to the scene's path or prompting Save As if it has none).
fn save_shortcut(keyboard: Res<ButtonInput<KeyCode>>, mut requests: ResMut<EditorRequests>) {
    let ctrl = keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight);
    if ctrl && keyboard.just_pressed(KeyCode::KeyS) {
        requests.save = true;
    }
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

    // Effective snap = sticky toolbar toggle OR a momentary Ctrl-hold (Blender-style).
    let ctrl = keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight);
    state.snap = state.snap_sticky || ctrl;
}

/// X or Delete despawns the selected SDF volume (replaces the old Spawn panel's
/// "Delete selected" button). Despawning changes the edit set → full rebake.
fn delete_selection(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    mut selection: ResMut<SdfSelection>,
) {
    if !(keyboard.just_pressed(KeyCode::KeyX) || keyboard.just_pressed(KeyCode::Delete)) {
        return;
    }
    if let Some(e) = selection.entity.take() {
        commands.entity(e).despawn();
    }
}
