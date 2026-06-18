//! **Dev-only fly + screenshot harness** (env-gated; zero impact unless the env vars are set).
//!
//! Lets the engine reproduce + capture MOTION artifacts headlessly without a human at the controls — fly the
//! `SdfCamera` along +X at a fixed speed, screenshot the primary window to disk every N frames, and force a
//! `debug_view`. Used to diagnose streaming/residency artifacts that only appear while moving (e.g. the gallery
//! black-cube during flight). All three knobs are independent:
//!
//! * `ADVENTURE_FLY=<m/s>`      — advance the camera +X at this speed each frame (overrides the controller by
//!                                accumulating into `Transform.translation`, which `fps_camera` preserves with
//!                                no key input). 0 / unset ⇒ no auto-fly.
//! * `ADVENTURE_SHOT_EVERY=<n>` — every `n` frames, save a screenshot to `D:/tmp_test/shot_<frame>.png`.
//! * `ADVENTURE_DEBUG_VIEW=<i>` — force `LightingUniformData.debug_view` to `i` (0=Lit,1=Normals,2=Depth,
//!                                3=Albedo,4=AO,5=GI,6=FaceOrient,7=LOD) so a capture can classify an artifact.

use bevy::prelude::*;
use bevy::render::view::window::screenshot::{Screenshot, save_to_disk};

use crate::sdf_render::SdfCamera;
use crate::voxel::raytrace::VoxelRtLighting;

/// Adds the dev fly/screenshot system, but ONLY when at least one of its env vars is set (so a normal run pays
/// nothing — the system isn't even registered).
pub struct DevFlycamPlugin;

impl Plugin for DevFlycamPlugin {
    fn build(&self, app: &mut App) {
        let on = std::env::var("ADVENTURE_FLY").is_ok()
            || std::env::var("ADVENTURE_SHOT_EVERY").is_ok()
            || std::env::var("ADVENTURE_DEBUG_VIEW").is_ok();
        if on {
            app.add_systems(Update, dev_flycam);
            info!("dev_flycam: ENABLED (ADVENTURE_FLY/SHOT_EVERY/DEBUG_VIEW set)");
        }
    }
}

fn dev_flycam(
    time: Res<Time>,
    mut frame: Local<u64>,
    mut cam: Query<&mut Transform, With<SdfCamera>>,
    lighting: Option<ResMut<VoxelRtLighting>>,
    mut commands: Commands,
) {
    *frame += 1;

    // Force a debug view (each frame — cheap; lets a later capture toggle without a rebuild).
    if let (Ok(v), Some(mut l)) = (std::env::var("ADVENTURE_DEBUG_VIEW"), lighting) {
        if let Ok(iv) = v.parse::<u32>() {
            l.data.debug_view = iv;
        }
    }

    // Auto-fly +X (accumulates into the camera Transform; `fps_camera` preserves it with no key input).
    if let Some(speed) = std::env::var("ADVENTURE_FLY").ok().and_then(|s| s.parse::<f32>().ok()) {
        if speed != 0.0 {
            let dt = time.delta_secs();
            for mut t in cam.iter_mut() {
                t.translation.x += speed * dt;
            }
        }
    }

    // Periodic screenshot to disk.
    if let Some(every) = std::env::var("ADVENTURE_SHOT_EVERY").ok().and_then(|s| s.parse::<u64>().ok()) {
        if every > 0 && *frame % every == 0 {
            let path = format!("D:/tmp_test/shot_{:05}.png", *frame);
            commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path));
        }
    }
}
