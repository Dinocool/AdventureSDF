//! Viewport toolbar strip. Rendered across the top of the dock's Viewport tab (see
//! [`super::dock`]); split out because it's the cluster that reaches into the
//! `sdf_render` gizmo/camera resources, separate from the dock layout host itself.

use bevy::prelude::*;
use bevy_egui::egui;

use crate::node::GizmoKind;
use crate::sdf_render::gizmo::{GizmoModes, GizmoState};
use crate::sdf_render::{GizmoVisibility, SdfCameraMode, SdfOrbitCamera, WireframeBoundsVisible};
use crate::voxel::{SceneReframed, VoxelScene};

/// One entry in the viewport scene-selector dropdown: a display label, the [`VoxelScene`] it selects, and —
/// for the BAKED `.vox` scenes that may not be on disk yet — the asset path to probe so an un-baked scene
/// shows DISABLED with a "bake it" hint instead of silently failing at switch time. `vox_path: None` means a
/// built-in scene (Cornell box / procedural worldgen) that is always available.
///
/// This is the SSOT name→scene(+path) table. To wire a NEW baked scene (San Miguel, Sibenik, …): add its
/// `VoxelScene` variant, its `pack` branch in `stream_voxel_rt_residency`, and ONE row here — the selector,
/// the availability probe, and the "bake via …" tooltip all follow from this table.
struct SceneOption {
    label: &'static str,
    scene: VoxelScene,
    /// `Some(path)` for a baked `.vox` scene (probed for existence); `None` for an always-available built-in.
    vox_path: Option<&'static str>,
}

/// The scene-selector table. Cornell + Worldgen are built-in (always available); Sponza is the baked default;
/// the Gallery is the side-by-side MERGED row (always available — its merge skips any unbaked row with a warn,
/// so the entry never hard-fails even with nothing baked, falling back to a Cornell box). San Miguel + Sibenik
/// are listed but NOT yet baked — their `.vox` files don't exist, so the selector shows them disabled with a
/// "bake via `cargo run --example voxelize_scene`" tooltip until the asset is produced. Their `VoxelScene`
/// variants don't exist yet either, so they surface in [`UNBAKED_SCENES`] as a roadmap placeholder until baked.
fn scene_options() -> [SceneOption; 4] {
    [
        SceneOption { label: "Sponza", scene: VoxelScene::Sponza, vox_path: Some(crate::voxel::raytrace::SPONZA_VOX_PATH) },
        SceneOption { label: "Cornell", scene: VoxelScene::Cornell, vox_path: None },
        SceneOption { label: "Worldgen", scene: VoxelScene::Worldgen, vox_path: None },
        SceneOption { label: "Gallery (side-by-side)", scene: VoxelScene::Gallery, vox_path: None },
    ]
}

/// Baked `.vox` scenes that are NOT yet wired into the [`VoxelScene`] enum — shown DISABLED in the selector
/// with a "bake via …" tooltip so the path to adding them is discoverable in the UI. `(label, vox_path)`.
/// When one is baked: add a [`VoxelScene`] variant, its pack branch, and a row to [`scene_options`]; drop it
/// from here.
const UNBAKED_SCENES: [(&str, &str); 2] = [
    ("San Miguel", "assets/models/san_miguel.vox"),
    ("Sibenik", "assets/models/sibenik.vox"),
];

/// The viewport scene selector: an egui ComboBox driving the SSOT [`VoxelScene`] resource (the same resource
/// the **`V`** key cycles). Switching resets the [`SceneReframed`] latch so the camera re-frames onto the new
/// scene next frame — exactly what the keyboard toggle does (the two entry points stay in lock-step). Baked
/// `.vox` scenes missing from disk are shown disabled with a "bake it" hint.
fn scene_selector(world: &mut World, ui: &mut egui::Ui) {
    let current = *world.resource::<VoxelScene>();
    let options = scene_options();
    let cur_label = options
        .iter()
        .find(|o| o.scene == current)
        .map(|o| o.label)
        .unwrap_or("?");
    let mut pick: Option<VoxelScene> = None;
    egui::ComboBox::from_id_salt("viewport_scene_selector")
        .selected_text(format!("{} Scene: {cur_label}", egui_phosphor::regular::STACK))
        .show_ui(ui, |ui| {
            for opt in &options {
                // A baked scene whose `.vox` is missing is selectable=false (it would just fall back to
                // Cornell at switch time) with a hint; a built-in (None) or a present `.vox` is selectable.
                let available = opt.vox_path.is_none_or(|p| std::path::Path::new(p).exists());
                let resp = ui.add_enabled(
                    available,
                    egui::SelectableLabel::new(opt.scene == current, opt.label),
                );
                let resp = if available {
                    resp
                } else {
                    resp.on_disabled_hover_text(
                        "not baked — run `cargo run --example voxelize_scene` to produce it",
                    )
                };
                if resp.clicked() && opt.scene != current {
                    pick = Some(opt.scene);
                }
            }
            // Roadmap scenes not yet in the enum: always disabled, with the bake hint.
            for (label, _path) in UNBAKED_SCENES {
                ui.add_enabled(false, egui::SelectableLabel::new(false, label))
                    .on_disabled_hover_text(
                        "not baked — run `cargo run --example voxelize_scene` to produce it",
                    );
            }
        });
    if let Some(scene) = pick {
        *world.resource_mut::<VoxelScene>() = scene;
        world.resource_mut::<SceneReframed>().0 = false; // re-frame onto the new scene (mirrors the V toggle)
        info!("voxel scene (editor selector): {scene:?}");
    }
}

/// Toolbar strip rendered across the top of the Viewport tab: camera-mode toggle
/// (Orbit ⇄ FPS), gizmo transform tools (mode + snap), and a view-options dropdown.
/// Drawn with a `TopBottomPanel::top` scoped to the tab's `ui`, so the 3D camera's
/// reserved rect (captured after) sits below it.
pub(crate) fn viewport_toolbar(world: &mut World, ui: &mut egui::Ui) {
    egui::TopBottomPanel::top("viewport_toolbar")
        .exact_height(28.0)
        .show_inside(ui, |ui| {
            ui.horizontal_centered(|ui| {
                let fps = world.resource::<SdfCameraMode>().fps;

                // Orbit / FPS segmented toggle.
                let orbit = format!("{} Orbit", egui_phosphor::regular::ARROWS_CLOCKWISE);
                if ui.selectable_label(!fps, orbit).clicked() && fps {
                    world.resource_mut::<SdfCameraMode>().fps = false;
                }
                let fps_label = format!("{} FPS", egui_phosphor::regular::GAME_CONTROLLER);
                if ui.selectable_label(fps, fps_label).clicked() && !fps {
                    // Seed the free-fly yaw/pitch from the orbit view so toggling in
                    // doesn't snap the camera to a new orientation.
                    // Orbit stores the target→camera offset (view looks along -dir),
                    // while FPS stores the look direction (+dir); they differ by π in
                    // yaw and a negated pitch.
                    let orbit = world.resource::<SdfOrbitCamera>();
                    let (yaw, pitch) = (orbit.yaw, orbit.pitch);
                    let mut mode = world.resource_mut::<SdfCameraMode>();
                    mode.fps = true;
                    mode.yaw = yaw + std::f32::consts::PI;
                    mode.pitch = -pitch;
                }
                // PLAYER toggle — drop a first-person walker on the terrain for a true sense of scale (or `P`).
                let player = world.resource::<SdfCameraMode>().player;
                let player_label = format!("{} Player", egui_phosphor::regular::PERSON_SIMPLE_WALK);
                if ui.selectable_label(player, player_label).on_hover_text("First-person walk on the terrain — sense of scale (P). RMB look · WASD · Space jump").clicked() {
                    world.resource_mut::<SdfCameraMode>().player = !player;
                }

                ui.separator();

                // Voxel scene selector (Sponza / Cornell / Worldgen; San Miguel + Sibenik disabled until
                // baked). Drives the same SSOT `VoxelScene` the `V` key cycles.
                scene_selector(world, ui);

                ui.separator();

                // Camera-mode instructions, to the left of the gizmo tools.
                if fps {
                    let speed = world.resource::<SdfCameraMode>().speed;
                    ui.label(format!("Fly: RMB look · WASD · Space/Ctrl · {speed:.0} u/s"));
                } else {
                    ui.label("Orbit: MMB rotate · Shift+MMB pan · wheel zoom");
                }

                ui.separator();

                // Gizmo transform tools: mode icon buttons + snap magnet. Disabled in
                // FPS mode (the gizmo only operates under the orbit camera). Icons are
                // Phosphor glyphs (installed via `install_phosphor_font`).
                ui.add_enabled_ui(!fps, |ui| {
                    use egui_phosphor::regular as icon;
                    let cur = world.resource::<GizmoState>().modes;
                    // (mode, icon glyph, tooltip incl. keybind).
                    for (modes, glyph, tip) in [
                        (GizmoModes::TRANSLATE, icon::ARROWS_OUT_CARDINAL, "Move (W)"),
                        (GizmoModes::ROTATE, icon::ARROW_CLOCKWISE, "Rotate (E)"),
                        (GizmoModes::SCALE, icon::ARROWS_OUT, "Scale (R)"),
                        (GizmoModes::all(), icon::CUBE, "All modes (Q)"),
                    ] {
                        if ui
                            .selectable_label(cur == modes, glyph)
                            .on_hover_text(tip)
                            .clicked()
                        {
                            world.resource_mut::<GizmoState>().modes = modes;
                        }
                    }

                    // Snap magnet: sticky toggle (Ctrl-hold still forces snap on too).
                    let sticky = world.resource::<GizmoState>().snap_sticky;
                    if ui
                        .selectable_label(sticky, icon::MAGNET)
                        .on_hover_text("Snap (toggle; or hold Ctrl)")
                        .clicked()
                    {
                        world.resource_mut::<GizmoState>().snap_sticky = !sticky;
                    }

                    // Snap settings dropdown: per-axis snap step sizes.
                    egui::ComboBox::from_id_salt("viewport_snap_settings")
                        .selected_text(format!("{} Snap settings", icon::GEAR))
                        .show_ui(ui, |ui| {
                            let mut g = world.resource_mut::<GizmoState>();
                            ui.add(egui::Slider::new(&mut g.snap_move, 0.0..=2.0).text("Move"));
                            ui.add(
                                egui::Slider::new(
                                    &mut g.snap_angle,
                                    0.0..=std::f32::consts::FRAC_PI_2,
                                )
                                .text("Rotate (rad)"),
                            );
                            ui.add(egui::Slider::new(&mut g.snap_scale, 0.0..=1.0).text("Scale"));
                        });
                });

                // Blender-style view-options dropdown, right-aligned. Display toggles.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    egui::ComboBox::from_id_salt("viewport_view_options")
                        .selected_text("View")
                        .show_ui(ui, |ui| {
                            let mut wf = world.resource::<WireframeBoundsVisible>().0;
                            if ui.checkbox(&mut wf, "Bounds wireframe").changed() {
                                world.resource_mut::<WireframeBoundsVisible>().0 = wf;
                            }

                            // Per-type gizmo visibility (+ master "All"). Driven by
                            // `GizmoKind::ALL`, so a new gizmo type appears here automatically.
                            ui.separator();
                            let all_on = GizmoKind::ALL
                                .iter()
                                .all(|k| world.resource::<GizmoVisibility>().is_visible(*k));
                            let mut all = all_on;
                            if ui.checkbox(&mut all, "All gizmos").changed() {
                                let mut vis = world.resource_mut::<GizmoVisibility>();
                                for k in GizmoKind::ALL {
                                    vis.0.insert(k, all);
                                }
                            }
                            for kind in GizmoKind::ALL {
                                let mut on = world.resource::<GizmoVisibility>().is_visible(kind);
                                if ui.checkbox(&mut on, kind.label()).changed() {
                                    world.resource_mut::<GizmoVisibility>().0.insert(kind, on);
                                }
                            }
                        });
                });
            });
        });
}
