//! Hierarchy panel (jackdaw `hierarchy`): the scene tree. Lists scene entities and
//! syncs click-selection with the SDF editor's [`SdfSelection`].

use bevy::prelude::*;
use bevy_egui::egui;

use crate::scene_manager::SceneEntity;
use crate::sdf_render::{SdfPrimitive, SdfSelection, SdfVolume};

/// Render the scene tree. SDF volumes are clickable rows that drive selection;
/// other scene entities (camera, light) are listed for context.
pub fn hierarchy_ui(world: &mut World, ui: &mut egui::Ui) {
    let selected = world.resource::<SdfSelection>().entity;

    // Collect rows first so we don't hold a query borrow across the UI closures.
    let mut volumes: Vec<(Entity, String)> = world
        .query_filtered::<(Entity, &SdfPrimitive), With<SdfVolume>>()
        .iter(world)
        .map(|(e, p)| (e, primitive_label(p).to_string()))
        .collect();
    volumes.sort_by_key(|(e, _)| e.index());

    let other = world
        .query_filtered::<(), (With<SceneEntity>, Without<SdfVolume>)>()
        .iter(world)
        .count();

    let mut clicked: Option<Entity> = None;
    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.collapsing(format!("SDF Volumes ({})", volumes.len()), |ui| {
            for (entity, label) in &volumes {
                let is_sel = selected == Some(*entity);
                if ui
                    .selectable_label(is_sel, format!("{label}  #{}", entity.index()))
                    .clicked()
                {
                    clicked = Some(*entity);
                }
            }
        });
        ui.weak(format!(
            "{other} other scene entit{}",
            if other == 1 { "y" } else { "ies" }
        ));
    });

    if let Some(entity) = clicked {
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
