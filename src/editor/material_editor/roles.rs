//! Per-role override editing for a `PbrTextureAsset`: the 7 image-file role pickers
//! (diffuse, normal, metallic, roughness, AO, height, edge), shared by the material
//! editor's Overrides section and the PBR-texture asset inspector.

use bevy::prelude::*;
use bevy_egui::egui;

use super::discovery::{image_file_entry, image_file_picker_entries};

/// Accessor pair for one override role on a `PbrTextureAsset`, so the override loop can
/// get/set each role generically.
struct RoleAccess {
    get: fn(&crate::assets::PbrTextureAsset) -> Option<std::path::PathBuf>,
    set: fn(&mut crate::assets::PbrTextureAsset, Option<std::path::PathBuf>),
}
impl RoleAccess {
    fn get(&self, t: &crate::assets::PbrTextureAsset) -> Option<std::path::PathBuf> {
        (self.get)(t)
    }
    fn set(&self, t: &mut crate::assets::PbrTextureAsset, v: Option<std::path::PathBuf>) {
        (self.set)(t, v)
    }
}

/// The 7 override roles, label + get/set, in editor display order.
const OVERRIDE_ROLES: [(&str, RoleAccess); 7] = [
    ("Diffuse", RoleAccess { get: |t| t.diffuse.clone(), set: |t, v| t.diffuse = v }),
    ("Normal", RoleAccess { get: |t| t.normal.clone(), set: |t, v| t.normal = v }),
    ("Metallic", RoleAccess { get: |t| t.metallic.clone(), set: |t, v| t.metallic = v }),
    ("Roughness", RoleAccess { get: |t| t.roughness.clone(), set: |t, v| t.roughness = v }),
    ("AO", RoleAccess { get: |t| t.ao.clone(), set: |t, v| t.ao = v }),
    ("Height", RoleAccess { get: |t| t.height.clone(), set: |t, v| t.height = v }),
    ("Edge", RoleAccess { get: |t| t.edge.clone(), set: |t, v| t.edge = v }),
];

/// The 7 per-role image-file pickers for a `PbrTextureAsset`, editing `tex` in place.
/// Shared by the material editor's Overrides section and the PBR-texture inspector.
pub fn pbr_texture_roles_ui(
    world: &mut World,
    ui: &mut egui::Ui,
    id: egui::Id,
    tex: &mut crate::assets::PbrTextureAsset,
) {
    for (i, (label, role)) in OVERRIDE_ROLES.iter().enumerate() {
        ui.label(*label);
        let cur = role.get(tex);
        let entry = cur.as_ref().map(|p| image_file_entry(p));
        match crate::editor::resource_picker::resource_picker(
            world,
            ui,
            id.with(("role", i)),
            entry.as_ref(),
            true,
            image_file_picker_entries,
        ) {
            Some(crate::editor::resource_picker::PickResult::Key(key)) => {
                role.set(tex, Some(std::path::PathBuf::from(key)));
            }
            Some(crate::editor::resource_picker::PickResult::None) => role.set(tex, None),
            None => {}
        }
    }
}
