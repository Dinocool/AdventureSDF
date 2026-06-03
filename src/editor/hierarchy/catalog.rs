//! Create-Node catalog + dialog: a Godot-style searchable, category-nested tree of
//! creatable node types. Adding a node type is a one-line [`NODE_CATALOG`] entry plus a
//! `spawn_*` helper — no dialog code changes.

use bevy::prelude::*;
use bevy_egui::egui;

use crate::sdf_render::debug::{
    spawn_camera, spawn_directional_light, spawn_empty_node, spawn_point_light, spawn_sdf_primitive,
};
use crate::sdf_render::{SdfPrimitive, SdfSelection};

use super::reparent::reparent_preserving_world;

/// A single creatable node type: how it's labelled in the Create-Node dialog and the
/// closure that spawns it (returning the new entity, which the dialog reparents).
struct NodeType {
    label: &'static str,
    icon: &'static str,
    /// One-line description shown under the dialog's selection.
    hint: &'static str,
    spawn: fn(&mut World) -> Entity,
}

/// A named group of node types in the Create-Node dialog (Godot-style nesting).
struct NodeCategory {
    label: &'static str,
    types: &'static [NodeType],
}

/// The full node-type tree offered by the Create-Node dialog. Adding a new node type
/// is a one-line entry here plus a `spawn_*` helper — no dialog code changes.
const NODE_CATALOG: &[NodeCategory] = &[
    NodeCategory {
        label: "Node3D",
        types: &[
            NodeType {
                label: "Node3D",
                icon: egui_phosphor::regular::CUBE_TRANSPARENT,
                hint: "Empty spatial node — a transform-only group / locator.",
                spawn: spawn_empty_node,
            },
            NodeType {
                label: "Camera",
                icon: egui_phosphor::regular::VIDEO_CAMERA,
                hint: "Scene camera node (serialized); the editor can look through it.",
                spawn: spawn_camera,
            },
        ],
    },
    NodeCategory {
        label: "SDF Primitives",
        types: &[
            NodeType {
                label: "Sphere",
                icon: egui_phosphor::regular::CUBE,
                hint: "SDF sphere volume.",
                spawn: |w| spawn_sdf_primitive(w, SdfPrimitive::Sphere { radius: 0.5 }),
            },
            NodeType {
                label: "Box",
                icon: egui_phosphor::regular::CUBE,
                hint: "SDF box volume.",
                spawn: |w| {
                    spawn_sdf_primitive(w, SdfPrimitive::Box {
                        half_extents: Vec3::splat(0.5),
                    })
                },
            },
            NodeType {
                label: "Torus",
                icon: egui_phosphor::regular::CUBE,
                hint: "SDF torus volume.",
                spawn: |w| {
                    spawn_sdf_primitive(w, SdfPrimitive::Torus {
                        major: 0.5,
                        minor: 0.18,
                    })
                },
            },
            NodeType {
                label: "Capsule",
                icon: egui_phosphor::regular::CUBE,
                hint: "SDF capsule volume.",
                spawn: |w| {
                    spawn_sdf_primitive(w, SdfPrimitive::Capsule {
                        half_height: 0.4,
                        radius: 0.28,
                    })
                },
            },
            NodeType {
                label: "Cylinder",
                icon: egui_phosphor::regular::CUBE,
                hint: "SDF cylinder volume.",
                spawn: |w| {
                    spawn_sdf_primitive(w, SdfPrimitive::Cylinder {
                        radius: 0.4,
                        half_height: 0.5,
                    })
                },
            },
        ],
    },
    NodeCategory {
        label: "Lights",
        types: &[
            NodeType {
                label: "Directional Light",
                icon: egui_phosphor::regular::SUN,
                hint: "Sun-style directional light (editor gizmo shows its direction).",
                spawn: spawn_directional_light,
            },
            NodeType {
                label: "Point Light",
                icon: egui_phosphor::regular::LIGHTBULB,
                hint: "Omnidirectional point light; drag the ring handle to set its radius.",
                spawn: spawn_point_light,
            },
        ],
    },
];

/// Create-Node dialog state, stashed in egui temp memory between frames.
#[derive(Clone, Default)]
pub(super) struct CreateDialog {
    pub(super) open: bool,
    /// Filter text matched against type labels.
    filter: String,
    /// `(category, type)` indices of the highlighted entry, if any.
    selected: Option<(usize, usize)>,
    /// Entity the new node is parented under (the selection when the dialog opened);
    /// `None` spawns at the scene root.
    pub(super) parent: Option<Entity>,
}

impl CreateDialog {
    /// Open the dialog with the new node parented under `parent` (the current selection).
    pub(super) fn opened_under(parent: Option<Entity>) -> Self {
        Self {
            open: true,
            parent,
            ..Default::default()
        }
    }
}

/// The Godot-style Create Node dialog: a searchable, category-nested tree of node
/// types. Picking one (double-click or the Create button) spawns it via its catalog
/// closure, parents it under `dialog.parent` (preserving world position), selects it,
/// and closes. State persists in `dialog` between frames.
pub(super) fn show_create_dialog(world: &mut World, ui: &mut egui::Ui, dialog: &mut CreateDialog) {
    if !dialog.open {
        return;
    }

    let mut open = true;
    let mut spawn_choice: Option<(usize, usize)> = None;

    egui::Window::new("Create Node")
        .id(ui.make_persistent_id("create_node_window"))
        .collapsible(false)
        .resizable(true)
        .default_size([320.0, 420.0])
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .open(&mut open)
        .show(ui.ctx(), |ui| {
            ui.add(
                egui::TextEdit::singleline(&mut dialog.filter)
                    .hint_text("Search nodes…")
                    .desired_width(f32::INFINITY),
            );
            ui.separator();

            let needle = dialog.filter.trim().to_lowercase();
            egui::ScrollArea::vertical()
                .max_height(300.0)
                .show(ui, |ui| {
                    for (ci, cat) in NODE_CATALOG.iter().enumerate() {
                        // Hide a category whose every type is filtered out.
                        let any = cat.types.iter().any(|t| {
                            needle.is_empty() || t.label.to_lowercase().contains(&needle)
                        });
                        if !any {
                            continue;
                        }
                        egui::CollapsingHeader::new(cat.label)
                            .default_open(true)
                            .id_salt(("cat", ci))
                            .show(ui, |ui| {
                                for (ti, ty) in cat.types.iter().enumerate() {
                                    if !needle.is_empty()
                                        && !ty.label.to_lowercase().contains(&needle)
                                    {
                                        continue;
                                    }
                                    let is_sel = dialog.selected == Some((ci, ti));
                                    let resp = ui.selectable_label(
                                        is_sel,
                                        format!("{}  {}", ty.icon, ty.label),
                                    );
                                    if resp.clicked() {
                                        dialog.selected = Some((ci, ti));
                                    }
                                    if resp.double_clicked() {
                                        spawn_choice = Some((ci, ti));
                                    }
                                }
                            });
                    }
                });

            ui.separator();
            // Hint line for the highlighted type.
            if let Some((ci, ti)) = dialog.selected {
                ui.weak(NODE_CATALOG[ci].types[ti].hint);
            } else {
                ui.weak("Select a node type to create.");
            }

            ui.horizontal(|ui| {
                let can_create = dialog.selected.is_some();
                if ui
                    .add_enabled(can_create, egui::Button::new("Create"))
                    .clicked()
                {
                    spawn_choice = dialog.selected;
                }
                if ui.button("Cancel").clicked() {
                    dialog.open = false;
                }
            });
        });

    // Window close button (the `open` bool) also dismisses the dialog.
    if !open {
        dialog.open = false;
    }

    if let Some((ci, ti)) = spawn_choice {
        let entity = (NODE_CATALOG[ci].types[ti].spawn)(world);
        if let Some(parent) = dialog.parent
            && world.get_entity(parent).is_ok()
            && parent != entity
        {
            reparent_preserving_world(world, entity, parent);
        }
        world.resource_mut::<SdfSelection>().entity = Some(entity);
        dialog.open = false;
    }
}
