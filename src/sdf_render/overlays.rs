//! Editor viewport overlays drawn with immediate-mode gizmos (NOT the SDF shader): the gizmo config
//! groups, the Godot-style ground grid, and the per-LOD clipmap ring boxes. These are editor-
//! interaction concerns that sit alongside the render path but never touch it. The gizmo groups are
//! re-exported from [`super`] so cross-module consumers keep the stable `sdf_render::` path; they are
//! still `init_gizmo_group`'d in `SdfScenePlugin::build`.

use bevy::prelude::*;

use super::{SdfCamera, SdfGridConfig, SdfVolume};
use super::{bake_scheduler, chunk};

/// Gizmo config group for editor overlays (transform handles, bounds). Uses
/// `depth_bias = -1.0` so overlays always draw on top of the SDF surface — the
/// editor convention. Drawn via immediate-mode gizmos, not the SDF shader.
#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct SdfOverlayGizmos;

/// Gizmo config group for the editor ground grid. Separate from the overlay group
/// so it keeps default depth (the SDF surface and geometry occlude grid lines
/// behind them) rather than always drawing on top.
#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct SdfGridGizmos;

/// Gizmo config group for node editor glyphs (light suns, empty-node axes). Uses
/// default depth (`depth_bias = 0.0`) so the SDF surface and other geometry occlude a
/// glyph that sits behind them — unlike the always-on-top transform handles in
/// [`SdfOverlayGizmos`].
#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct SdfNodeGizmos;

/// Whether the per-LOD clipmap ring wire boxes are drawn (toggled with F8). Off by
/// default so the overlay stays clean; see [`draw_lod_rings`].
#[derive(Resource, Default)]
pub struct LodRingsVisible(pub bool);

/// Push the overlay gizmo group in front of everything (always-on-top handles).
pub fn configure_overlay_gizmos(mut store: ResMut<GizmoConfigStore>) {
    let (config, _) = store.config_mut::<SdfOverlayGizmos>();
    config.depth_bias = -1.0;
    config.line.width = 3.0;

    // Grid uses default depth (occluded by geometry) and thin lines.
    let (grid, _) = store.config_mut::<SdfGridGizmos>();
    grid.depth_bias = 0.0;
    grid.line.width = 1.0;

    // Node glyphs (light suns, empties) depth-test against the SDF surface: a glyph
    // behind geometry is occluded, so it reads as being in the scene.
    let (nodes, _) = store.config_mut::<SdfNodeGizmos>();
    nodes.depth_bias = 0.0;
    nodes.line.width = 2.0;
}

/// Draw a Godot-style infinite ground grid on the XZ plane: faint minor lines
/// every unit, brighter major lines every `MAJOR` units, and colored X (red) /
/// Z (blue) axis lines through the world origin. Centred on the camera EYE snapped to
/// the grid so it reads as infinite as the view pans — including FPS/free-fly mode,
/// where the orbit target is stale (mirrors `draw_lod_rings`, which also follows the
/// camera eye). The axis lines stay at the world origin (keyed on the absolute grid
/// index), so the origin axes remain meaningful while the grid extent follows the camera.
pub fn draw_ground_grid(
    mut gizmos: Gizmos<SdfGridGizmos>,
    camera: Query<&Transform, (With<SdfCamera>, Without<SdfVolume>)>,
) {
    const HALF: i32 = 50; // lines each side of centre
    const STEP: f32 = 1.0; // grid spacing in world units (Godot-style 1m cells)
    let step = STEP;

    let minor = Color::srgba(0.35, 0.35, 0.38, 0.5);
    let major = Color::srgba(0.55, 0.55, 0.60, 0.8);
    let x_axis = Color::srgb(0.86, 0.24, 0.24);
    let z_axis = Color::srgb(0.26, 0.49, 0.92);

    // Snap the grid centre to the camera eye so lines stay put as the camera moves (orbit OR FPS).
    let cam_pos = camera.iter().next().map(|t| t.translation).unwrap_or(Vec3::ZERO);
    let cx = (cam_pos.x / step).round() as i32;
    let cz = (cam_pos.z / step).round() as i32;
    let extent = HALF as f32 * step;

    for i in -HALF..=HALF {
        let gx = cx + i;
        let gz = cz + i;
        let wx = gx as f32 * step;
        let wz = gz as f32 * step;

        // Line parallel to Z at x = wx. At gx == 0 this lies on the Z axis (blue).
        let color = line_color(gx, z_axis, major, minor);
        gizmos.line(
            Vec3::new(wx, 0.0, cz as f32 * step - extent),
            Vec3::new(wx, 0.0, cz as f32 * step + extent),
            color,
        );
        // Line parallel to X at z = wz. At gz == 0 this lies on the X axis (red).
        let color = line_color(gz, x_axis, major, minor);
        gizmos.line(
            Vec3::new(cx as f32 * step - extent, 0.0, wz),
            Vec3::new(cx as f32 * step + extent, 0.0, wz),
            color,
        );
    }
}

/// Pick a grid line's colour: the axis colour at index 0 (the origin line), else a
/// major or minor tone depending on divisibility by `MAJOR`.
fn line_color(index: i32, axis: Color, major: Color, minor: Color) -> Color {
    const MAJOR: i32 = 10;
    if index == 0 {
        axis
    } else if index % MAJOR == 0 {
        major
    } else {
        minor
    }
}

/// Draw each LOD clipmap ring's world-AABB as a wire box, colour-matched to the
/// `SDF_DEBUG_LOD` shader ramp (green = fine/near, red = coarse/far). Makes the nested
/// ring extents and their camera-centred recentering directly visible. Derives each box from
/// `bake_scheduler::ring_chunk_origin` — the SAME snapped chunk-space origin the bake centres each
/// ring on (with `recenter_snap_chunks` hysteresis) — so the boxes track the actual resident set.
pub fn draw_lod_rings(
    mut gizmos: Gizmos<SdfOverlayGizmos>,
    config: Res<SdfGridConfig>,
    camera: Query<&Transform, (With<SdfCamera>, Without<SdfVolume>)>,
) {
    let Some(cam) = camera.iter().next() else {
        return;
    };
    let cam_pos = cam.translation;

    for lod in 0..config.lod_count {
        let origin_chunk = bake_scheduler::ring_chunk_origin(&config, cam_pos, lod);
        let min = chunk::chunk_min_world(chunk::ChunkKey::new(lod, origin_chunk), &config);
        // The ring spans `ring_bricks` bricks per axis at this LOD's voxel size.
        let extent = Vec3::splat(config.brick_world_size(lod) * config.ring_bricks as f32);
        let center = min + extent * 0.5;

        // Discrete colours matching the SDF_DEBUG_LOD shader: 0 white, 1 green,
        // 2 blue, 3 red, 4+ yellow.
        let color = match lod {
            0 => Color::srgb(1.0, 1.0, 1.0),
            1 => Color::srgb(0.0, 1.0, 0.0),
            2 => Color::srgb(0.0, 0.4, 1.0),
            3 => Color::srgb(1.0, 0.0, 0.0),
            _ => Color::srgb(1.0, 1.0, 0.0),
        };
        gizmos.primitive_3d(
            &Cuboid::new(extent.x, extent.y, extent.z),
            Isometry3d::from_translation(center),
            color,
        );
    }
}
