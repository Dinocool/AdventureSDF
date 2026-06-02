//! In-editor acknowledgement for Nsight GPU-Trace captures.
//!
//! When the editor is launched under NVIDIA Nsight Graphics (the root `run-worktree.ps1`
//! "Nsight profiling" / `S` option does this), the GPU Trace activity is armed with
//! `--start-after-hotkey`, so pressing **F11** in-game makes Nsight capture the live frame
//! and auto-export it to `.soul/ngfx` (then `rdoc/scripts/ngfx/parse.py` -> `perf.json`).
//!
//! Nsight performs the capture itself out-of-process via its injected layer — the app never
//! calls into it. This module exists only to surface a toast + log line so the trigger is
//! acknowledged inside the editor (the "notification when it happens"). It is compiled only
//! into `shader-debug` builds, which is exactly when the editor is run under Nsight.

use bevy::prelude::*;

use super::notifications::Notifications;

pub struct NsightCapturePlugin;

impl Plugin for NsightCapturePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, acknowledge_on_f11);
    }
}

/// F11 → toast + log that an Nsight GPU-Trace capture was triggered. This does NOT perform
/// the capture (Nsight's injected layer does, watching the same key); it only confirms the
/// trigger in-editor, since the app can't observe Nsight's out-of-process export.
fn acknowledge_on_f11(keyboard: Res<ButtonInput<KeyCode>>, mut notes: ResMut<Notifications>) {
    if keyboard.just_pressed(KeyCode::F11) {
        info!(
            "Nsight GPU-Trace: F11 trigger pressed — capturing the live frame and exporting \
             to .soul/ngfx (run rdoc/scripts/ngfx/parse.py for perf.json)."
        );
        notes.info("GPU-Trace capture triggered (F11) — exporting to .soul/ngfx\u{2026}");
    }
}
