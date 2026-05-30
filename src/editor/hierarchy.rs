//! Scene panel (jackdaw `hierarchy`): the scene tree. Lists scene entities and
//! syncs click-selection with the SDF editor's [`SdfSelection`]. The `+` button adds
//! a new SDF node; the filter box narrows the list by label. Rows show each node's
//! [`Name`] and can be renamed inline (right-click → Rename, or F2 on the selection).

use bevy::prelude::*;
use bevy_egui::egui;

use crate::scene_manager::SceneEntity;
use crate::sdf_render::debug::spawn_default_sdf;
use crate::sdf_render::{OrbitFocus, SdfPrimitive, SdfSelection, SdfVolume};

/// In-progress inline rename, stashed in egui temp memory between frames.
#[derive(Clone, Default)]
struct RenameState {
    entity: Option<Entity>,
    buf: String,
}

/// One collected row: entity, display name (Name or primitive label), whether it
/// has a real Name, and the lowercased text the filter matches against.
struct Row {
    entity: Entity,
    name: String,
    /// True if the node has a `Name` component (vs. a primitive-kind fallback). When
    /// named, the row hides the `#index` identifier.
    named: bool,
    filter_key: String,
}

/// Render the scene tree. SDF volumes are clickable rows that drive selection;
/// other scene entities (camera, light) are listed for context.
pub fn hierarchy_ui(world: &mut World, ui: &mut egui::Ui) {
    let selected = world.resource::<SdfSelection>().entity;

    let filter_id = ui.make_persistent_id("scene_filter");
    let rename_id = ui.make_persistent_id("scene_rename");
    let mut filter: String =
        ui.memory_mut(|m| m.data.get_temp::<String>(filter_id).unwrap_or_default());
    let mut rename: RenameState =
        ui.memory_mut(|m| m.data.get_temp::<RenameState>(rename_id).unwrap_or_default());

    // Toolbar: add-node button + filter box.
    let mut spawn_now = false;
    ui.horizontal(|ui| {
        if ui
            .button("+")
            .on_hover_text("Add a new SDF node")
            .clicked()
        {
            spawn_now = true;
        }
        ui.add(
            egui::TextEdit::singleline(&mut filter)
                .hint_text("Filter")
                .desired_width(f32::INFINITY),
        );
    });
    ui.memory_mut(|m| m.data.insert_temp(filter_id, filter.clone()));

    if spawn_now {
        let e = spawn_default_sdf(world);
        world.resource_mut::<SdfSelection>().entity = Some(e);
    }

    // Collect rows first so we don't hold a query borrow across the UI closures. A
    // node's display name is its `Name` if set, else the primitive kind.
    let needle = filter.trim().to_lowercase();
    let mut rows: Vec<Row> = world
        .query_filtered::<(Entity, &SdfPrimitive, Option<&Name>), With<SdfVolume>>()
        .iter(world)
        .map(|(e, p, name)| {
            let named = name.is_some();
            let name = name
                .map(|n| n.as_str().to_string())
                .unwrap_or_else(|| primitive_label(p).to_string());
            let filter_key = format!("{name} #{}", e.index()).to_lowercase();
            Row {
                entity: e,
                name,
                named,
                filter_key,
            }
        })
        .filter(|r| needle.is_empty() || r.filter_key.contains(&needle))
        .collect();
    rows.sort_by_key(|r| r.entity.index());

    let other = world
        .query_filtered::<(), (With<SceneEntity>, Without<SdfVolume>)>()
        .iter(world)
        .count();

    // F2 starts renaming the current selection (if not already renaming).
    let f2 = ui.input(|i| i.key_pressed(egui::Key::F2));
    if f2
        && rename.entity.is_none()
        && let Some(sel) = selected
        && let Some(row) = rows.iter().find(|r| r.entity == sel)
    {
        rename.entity = Some(sel);
        rename.buf = row.name.clone();
    }

    let mut clicked: Option<Entity> = None;
    let mut double_clicked: Option<Entity> = None;
    let mut commit_rename = false;
    let mut start_rename: Option<(Entity, String)> = None;

    egui::ScrollArea::vertical().show(ui, |ui| {
        egui::CollapsingHeader::new(format!("SDF Volumes ({})", rows.len()))
            .default_open(true)
            .show(ui, |ui| {
                for row in &rows {
                    if rename.entity == Some(row.entity) {
                        // Inline rename field: autofocus, commit on Enter or focus loss.
                        let resp = ui.add(
                            egui::TextEdit::singleline(&mut rename.buf)
                                .desired_width(f32::INFINITY),
                        );
                        resp.request_focus();
                        let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
                        if enter || resp.lost_focus() {
                            commit_rename = true;
                        }
                        continue;
                    }

                    let is_sel = selected == Some(row.entity);
                    // Named nodes show just the name; unnamed fall back to "Kind #index".
                    let label = if row.named {
                        row.name.clone()
                    } else {
                        format!("{}  #{}", row.name, row.entity.index())
                    };
                    let resp = ui.selectable_label(is_sel, label);
                    if resp.clicked() {
                        clicked = Some(row.entity);
                    }
                    if resp.double_clicked() {
                        double_clicked = Some(row.entity);
                    }
                    resp.context_menu(|ui| {
                        if ui.button("Rename").clicked() {
                            start_rename = Some((row.entity, row.name.clone()));
                            ui.close();
                        }
                    });
                }
            });
        ui.weak(format!(
            "{other} other scene entit{}",
            if other == 1 { "y" } else { "ies" }
        ));
    });

    if let Some((entity, name)) = start_rename {
        rename.entity = Some(entity);
        rename.buf = name;
    }

    if commit_rename {
        if let Some(entity) = rename.entity {
            let new = rename.buf.trim();
            if !new.is_empty()
                && let Ok(mut e) = world.get_entity_mut(entity)
            {
                e.insert(Name::new(new.to_string()));
            }
        }
        rename = RenameState::default();
    }
    ui.memory_mut(|m| m.data.insert_temp(rename_id, rename));

    // Double-click focuses the orbit camera on the volume, mirroring a viewport
    // double-click. `orbit_camera` eases `SdfOrbitCamera.target` toward this point.
    if let Some(entity) = double_clicked {
        let pos = world.get::<Transform>(entity).map(|t| t.translation);
        if let Some(pos) = pos {
            world.resource_mut::<OrbitFocus>().target = Some(pos);
        }
    }

    if let Some(entity) = clicked.or(double_clicked) {
        world.resource_mut::<SdfSelection>().entity = Some(entity);
    }
}

fn primitive_label(prim: &SdfPrimitive) -> &'static str {
    match prim {
        SdfPrimitive::Sphere { .. } => "Sphere",
        SdfPrimitive::Box { .. } => "Box",
        SdfPrimitive::Torus { .. } => "Torus",
        SdfPrimitive::Capsule { .. } => "Capsule",
        SdfPrimitive::Cylinder { .. } => "Cylinder",
        SdfPrimitive::Heightmap { .. } => "Heightmap",
    }
}
