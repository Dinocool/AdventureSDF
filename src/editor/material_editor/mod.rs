//! Shared material editor: one `material_editor_ui` used by both the Resources panel
//! and the asset/entity inspectors. Renders an interactive live preview (orbit + shape
//! switch, see [`crate::editor::material_preview`]) above the editable fields (base
//! color, blend softness, metallic/roughness, the texture maps). Edits go through
//! `Assets::<MaterialAsset>::get_mut`, firing `AssetEvent::Modified` → live recompile.
//!
//! Submodules: [`roles`] (per-role override pickers), [`discovery`] (filesystem → picker
//! entries + library variants), [`io`] (save/load + path↔handle resolution).

mod discovery;
mod io;
mod roles;

pub use discovery::{discover_variants, material_picker_entries};
pub use io::{
    handle_for_path, pbrtex_handle_for_path, save_material,
    save_pbr_texture,
};
pub use roles::pbr_texture_roles_ui;

use bevy::prelude::*;
use bevy_egui::egui;

use crate::assets::MaterialAsset;
use crate::editor::material_preview::{MaterialPreviewState, PreviewShape};

use discovery::{pbrtex_entry, pbrtex_picker_entries};

/// Edit the `MaterialAsset` behind `handle`: interactive preview + field editors. The
/// preview's live `StandardMaterial` is (re)built from the asset each frame and pointed
/// at by [`MaterialPreviewState`], so edits show instantly. Returns nothing — edits are
/// applied in place via `get_mut` (which marks the asset Modified for live recompile).
pub fn material_editor_ui(world: &mut World, handle: &Handle<MaterialAsset>, ui: &mut egui::Ui) {
    // --- Interactive preview ------------------------------------------------------
    // Rebuild the preview StandardMaterial from the current asset and hand it to the
    // preview rig. Built every frame (cheap) so field edits are reflected live.
    let preview_mat = {
        let server = world.resource::<AssetServer>().clone();
        let bundles = world.resource::<Assets<crate::assets::PbrTextureAsset>>();
        world
            .resource::<Assets<MaterialAsset>>()
            .get(handle)
            .map(|asset| {
                crate::editor::assets_browser::thumbnail::standard_from_material(
                    asset, &server, bundles,
                )
                .0
            })
    };
    if let Some(mat) = preview_mat {
        world.resource_scope(|world, mut state: Mut<MaterialPreviewState>| {
            state.set_material(&mut world.resource_mut::<Assets<StandardMaterial>>(), mat);
        });
    }

    preview_widget(world, ui);
    ui.separator();

    // --- Field editors ------------------------------------------------------------
    // Edit against a CLONE, writing back only on a real change (see `edit_asset`): a bare
    // `get_mut` every frame would fire `AssetEvent::Modified` even with no edit and
    // needlessly re-render the material thumbnail just from viewing the inspector.
    if world.resource::<Assets<MaterialAsset>>().get(handle).is_none() {
        ui.weak("Material asset still loading…");
        return;
    }
    let base = ui.make_persistent_id(("mat_tex_picker", handle.id()));
    crate::editor::fs_util::edit_asset(world, handle, |edited, world| {
        let mut rgb = [edited.base_color[0], edited.base_color[1], edited.base_color[2]];
        ui.horizontal(|ui| {
            ui.label("Base color");
            if ui.color_edit_button_rgb(&mut rgb).changed() {
                edited.base_color[0] = rgb[0];
                edited.base_color[1] = rgb[1];
                edited.base_color[2] = rgb[2];
            }
        });
        ui.add(egui::Slider::new(&mut edited.blend_softness, 0.0..=1.0).text("Blend softness"));
        ui.add(egui::Slider::new(&mut edited.metallic, 0.0..=1.0).text("Metallic"));
        ui.add(egui::Slider::new(&mut edited.roughness, 0.0..=1.0).text("Roughness"));

        // Emissive (self-lit) colour + intensity. The shader emits `color × intensity` as
        // radiance — this is also the GI light source, so a glowing material
        // lights its surroundings. `color_edit_button_rgb` edits the `[f32; 3]` in place.
        ui.horizontal(|ui| {
            ui.label("Emissive");
            ui.color_edit_button_rgb(&mut edited.emissive_color);
        });
        ui.add(
            egui::Slider::new(&mut edited.emissive_intensity, 0.0..=1000.0)
                .logarithmic(true)
                .text("Emissive intensity"),
        );

        ui.separator();
        ui.label("PBR Texture");

        // ONE bundle picker: choose the `.pbrtex.ron` this material uses (grid of bundles).
        let current_bundle = edited.texture.as_ref().map(|p| pbrtex_entry(p));
        match crate::editor::resource_picker::resource_picker(
            world,
            ui,
            base.with("bundle"),
            current_bundle.as_ref(),
            true,
            pbrtex_picker_entries,
        ) {
            Some(crate::editor::resource_picker::PickResult::Key(key)) => {
                edited.texture = Some(std::path::PathBuf::from(key));
            }
            Some(crate::editor::resource_picker::PickResult::None) => edited.texture = None,
            None => {}
        }

        // Per-role overrides (a role set here replaces the bundle's for that role).
        let overrides_id = base.with("overrides");
        egui::CollapsingHeader::new("Overrides")
            .id_salt(overrides_id)
            .default_open(false)
            .show(ui, |ui| {
                pbr_texture_roles_ui(world, ui, overrides_id, &mut edited.overrides);
            });
    });
}

/// Edit a `PbrTextureAsset` bundle: a live preview (the bundle applied to the preview
/// sphere) + the 7 role pickers. Edits write back only on change. Used by the PBR-texture
/// asset inspector.
pub fn pbr_texture_editor_ui(
    world: &mut World,
    handle: &Handle<crate::assets::PbrTextureAsset>,
    ui: &mut egui::Ui,
) {
    // Live preview: build a StandardMaterial from the bundle's diffuse/normal.
    let preview_mat = {
        let server = world.resource::<AssetServer>().clone();
        world
            .resource::<Assets<crate::assets::PbrTextureAsset>>()
            .get(handle)
            .map(|tex| {
                let load = |p: &Option<std::path::PathBuf>| {
                    p.as_ref().map(|p| server.load::<Image>(p.clone()))
                };
                StandardMaterial {
                    base_color_texture: load(&tex.diffuse),
                    normal_map_texture: load(&tex.normal),
                    perceptual_roughness: 0.8,
                    ..default()
                }
            })
    };
    if let Some(mat) = preview_mat {
        world.resource_scope(|world, mut state: Mut<MaterialPreviewState>| {
            state.set_material(&mut world.resource_mut::<Assets<StandardMaterial>>(), mat);
        });
    }
    preview_widget(world, ui);
    ui.separator();

    if world
        .resource::<Assets<crate::assets::PbrTextureAsset>>()
        .get(handle)
        .is_none()
    {
        ui.weak("PBR texture still loading…");
        return;
    }
    let id = ui.make_persistent_id(("pbrtex_editor", handle.id()));
    crate::editor::fs_util::edit_asset(world, handle, |edited, world| {
        pbr_texture_roles_ui(world, ui, id, edited);
    });
}

/// Draw the interactive preview image (orbit on drag, scroll to zoom) + the shape
/// selector. Reads/writes [`MaterialPreviewState`].
fn preview_widget(world: &mut World, ui: &mut egui::Ui) {
    let tex = world.resource::<MaterialPreviewState>().tex_id;
    let Some(tex) = tex else {
        ui.weak("Preview initializing…");
        return;
    };

    let resp = ui.add(
        egui::Image::new(egui::load::SizedTexture::new(tex, egui::vec2(220.0, 220.0)))
            .sense(egui::Sense::click_and_drag()),
    );
    if resp.dragged() {
        let d = resp.drag_delta();
        let mut state = world.resource_mut::<MaterialPreviewState>();
        state.yaw -= d.x * 0.01;
        state.pitch = (state.pitch + d.y * 0.01).clamp(-1.4, 1.4);
    }
    if resp.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            let mut state = world.resource_mut::<MaterialPreviewState>();
            state.distance = (state.distance - scroll * 0.01).clamp(1.5, 8.0);
        }
    }

    ui.horizontal(|ui| {
        let cur = world.resource::<MaterialPreviewState>().shape;
        let mut pick = cur;
        ui.selectable_value(&mut pick, PreviewShape::Sphere, "Sphere");
        ui.selectable_value(&mut pick, PreviewShape::Cube, "Cube");
        ui.selectable_value(&mut pick, PreviewShape::Torus, "Torus");
        if pick != cur {
            world.resource_mut::<MaterialPreviewState>().shape = pick;
        }
    });
}
