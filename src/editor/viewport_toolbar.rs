//! Viewport toolbar strip. Rendered across the top of the dock's Viewport tab (see
//! [`super::dock`]); split out because it's the cluster that reaches into the
//! `sdf_render` gizmo/camera resources, separate from the dock layout host itself.

use bevy::prelude::*;
use bevy_egui::egui;

use crate::node::GizmoKind;
use crate::sdf_render::gizmo::{GizmoModes, GizmoState};
use crate::sdf_render::{GizmoVisibility, SdfCameraMode, SdfOrbitCamera, WireframeBoundsVisible};

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
