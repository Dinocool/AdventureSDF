//! Shared material editor: one `material_editor_ui` used by both the Resources panel
//! and the asset/entity inspectors. Renders an interactive live preview (orbit + shape
//! switch, see [`crate::editor::material_preview`]) above the editable fields (base
//! color, blend softness, metallic/roughness, the 5 texture maps). Edits go through
//! `Assets::<MaterialAsset>::get_mut`, firing `AssetEvent::Modified` → live recompile.

use bevy::prelude::*;
use bevy_egui::egui;

use crate::assets::{MaterialAsset, MaterialAssetTable};
use crate::editor::material_preview::{MaterialPreviewState, PreviewShape};
use crate::sdf_render::textures::{LibraryVariant, TEXTURE_ROOT, read_manifest};

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
        let mat_handle = world.resource_mut::<Assets<StandardMaterial>>().add(mat);
        world.resource_mut::<MaterialPreviewState>().material = Some(mat_handle);
    }

    preview_widget(world, ui);
    ui.separator();

    // --- Field editors ------------------------------------------------------------
    // Edit against a CLONE, then write back only if something changed. Crucial: a bare
    // `get_mut` every frame fires `AssetEvent::Modified` even with no edit, which would
    // invalidate the material thumbnail (→ "…" / needless re-render) just by viewing the
    // inspector.
    let Some(mut edited) = world.resource::<Assets<MaterialAsset>>().get(handle).cloned() else {
        ui.weak("Material asset still loading…");
        return;
    };
    let before = edited.clone();

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

    ui.separator();
    ui.label("PBR Texture");

    // ONE bundle picker: choose the `.pbrtex.ron` this material uses (grid of bundles).
    let base = ui.make_persistent_id(("mat_tex_picker", handle.id()));
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

    // Write back only if a field actually changed (avoids spurious `Modified` events).
    if !material_eq(&edited, &before)
        && let Some(asset) = world.resource_mut::<Assets<MaterialAsset>>().get_mut(handle)
    {
        *asset = edited;
    }
}

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
        let h = world.resource_mut::<Assets<StandardMaterial>>().add(mat);
        world.resource_mut::<MaterialPreviewState>().material = Some(h);
    }
    preview_widget(world, ui);
    ui.separator();

    let Some(mut edited) = world
        .resource::<Assets<crate::assets::PbrTextureAsset>>()
        .get(handle)
        .cloned()
    else {
        ui.weak("PBR texture still loading…");
        return;
    };
    let before = edited.clone();

    let id = ui.make_persistent_id(("pbrtex_editor", handle.id()));
    pbr_texture_roles_ui(world, ui, id, &mut edited);

    if edited != before
        && let Some(asset) = world
            .resource_mut::<Assets<crate::assets::PbrTextureAsset>>()
            .get_mut(handle)
    {
        *asset = edited;
    }
}

/// Resolve a `.pbrtex.ron` path to its loaded handle (or load it). For the inspector.
pub fn pbrtex_handle_for_path(
    world: &mut World,
    path: &std::path::Path,
) -> Option<Handle<crate::assets::PbrTextureAsset>> {
    let rel = path
        .strip_prefix(crate::editor::assets_browser::ASSETS_ROOT)
        .ok()?
        .to_path_buf();
    // Cache a strong handle so the bundle stays loaded (a fresh `load` each frame would
    // never finish loading → the inspector shows "still loading…" forever).
    let server = world.resource::<AssetServer>().clone();
    Some(
        world
            .resource_mut::<crate::assets::PbrTextureHandles>()
            .ensure(&rel, &server),
    )
}

/// Save a `PbrTextureAsset` bundle to `assets/<rel>` (rel relative to assets root).
pub fn save_pbr_texture(
    world: &World,
    handle: &Handle<crate::assets::PbrTextureAsset>,
    rel_path: &std::path::Path,
) {
    let path = std::path::Path::new("assets").join(rel_path);
    if let Some(asset) = world.resource::<Assets<crate::assets::PbrTextureAsset>>().get(handle) {
        match crate::assets::Asset::save(asset, &path) {
            Ok(()) => info!("saved pbr texture to {}", path.display()),
            Err(e) => error!("pbr texture save failed: {e}"),
        }
    }
}

/// Field equality for `MaterialAsset` (it isn't `PartialEq`), to detect real edits.
fn material_eq(a: &MaterialAsset, b: &MaterialAsset) -> bool {
    a.base_color == b.base_color
        && a.blend_softness == b.blend_softness
        && a.metallic == b.metallic
        && a.roughness == b.roughness
        && a.parallax_scale == b.parallax_scale
        && a.texture == b.texture
        && a.overrides == b.overrides
}

/// A `.pbrtex.ron` bundle path → `PickerEntry` (key/label = the path; thumbnail = the
/// bundle file itself, rendered by the material-sphere thumbnail provider... actually a
/// generic icon, since the bundle's diffuse is what we want — use the bundle path tile).
fn pbrtex_entry(path: &std::path::Path) -> crate::editor::resource_picker::PickerEntry {
    use crate::editor::assets_browser::TileThumb;
    let label = path
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.trim_end_matches(".pbrtex.ron").to_string())
        .unwrap_or_default();
    crate::editor::resource_picker::PickerEntry {
        key: path.to_string_lossy().into_owned(),
        label,
        // Working-dir path so the assets-browser thumbnail provider can resolve it.
        thumb: TileThumb::Path(
            std::path::Path::new(crate::editor::assets_browser::ASSETS_ROOT).join(path),
        ),
    }
}

/// Picker entries for every `*.pbrtex.ron` under `assets/` (recursive).
fn pbrtex_picker_entries() -> Vec<crate::editor::resource_picker::PickerEntry> {
    let root = std::path::Path::new(crate::editor::assets_browser::ASSETS_ROOT);
    let mut out: Vec<_> = walk_files(root)
        .into_iter()
        .filter(|p| p.to_string_lossy().to_lowercase().ends_with(".pbrtex.ron"))
        .filter_map(|p| p.strip_prefix(root).ok().map(pbrtex_entry))
        .collect();
    out.sort_by_key(|e| e.label.to_lowercase());
    out
}

/// An image file (path relative to `assets/`) → `PickerEntry` (thumbnail = the image).
fn image_file_entry(path: &std::path::Path) -> crate::editor::resource_picker::PickerEntry {
    use crate::editor::assets_browser::TileThumb;
    let label = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    crate::editor::resource_picker::PickerEntry {
        key: path.to_string_lossy().into_owned(),
        label,
        thumb: TileThumb::Path(
            std::path::Path::new(crate::editor::assets_browser::ASSETS_ROOT).join(path),
        ),
    }
}

/// Picker entries for every image file under `assets/textures/` (recursive). Used by the
/// per-role override pickers — any supported image format.
fn image_file_picker_entries() -> Vec<crate::editor::resource_picker::PickerEntry> {
    let root = std::path::Path::new(crate::editor::assets_browser::ASSETS_ROOT);
    let tex_root = root.join("textures");
    let mut out: Vec<_> = walk_files(&tex_root)
        .into_iter()
        .filter(|p| is_image_file(p))
        .filter_map(|p| p.strip_prefix(root).ok().map(image_file_entry))
        .collect();
    out.sort_by_key(|e| e.label.to_lowercase());
    out
}

/// Whether `path` has a supported image extension.
fn is_image_file(path: &std::path::Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()).as_deref(),
        Some("png" | "jpg" | "jpeg" | "bmp" | "tga")
    )
}

/// Recursively collect all files under `dir`.
fn walk_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out
}

/// Save the material behind `handle` to `assets/materials/<name>.material.ron`. Shared
/// by the Resources panel and the asset inspector.
pub fn save_material(world: &World, handle: &Handle<MaterialAsset>, name: &str) {
    let path = std::path::PathBuf::from(format!("assets/materials/{name}.material.ron"));
    if let Some(asset) = world.resource::<Assets<MaterialAsset>>().get(handle) {
        match crate::assets::Asset::save(asset, &path) {
            Ok(()) => info!("saved material to {}", path.display()),
            Err(e) => error!("material save failed: {e}"),
        }
    }
}

/// Resolve a `.material.ron` asset path to its loaded `MaterialAsset` handle. Used by
/// the asset/entity inspectors. Returns the table's handle when the path is already a
/// registered material (so edits drive the same live-recompiled asset), else loads it.
pub fn handle_for_path(world: &World, path: &std::path::Path) -> Option<Handle<MaterialAsset>> {
    let rel = path
        .strip_prefix(crate::editor::assets_browser::ASSETS_ROOT)
        .ok()?
        .to_path_buf();
    let server = world.resource::<AssetServer>();
    // Prefer an already-loaded handle (the demo scene loads its materials at startup);
    // fall back to a fresh load so any `.material.ron` is inspectable.
    server
        .get_handle::<MaterialAsset>(rel.clone())
        .or_else(|| Some(server.load::<MaterialAsset>(rel)))
}

/// List every `assets/materials/*.material.ron` as a [`PickerEntry`] (key = working-dir
/// path string, label = file stem, thumb = the file path so the registry renders its
/// sphere). For the material resource picker.
pub fn material_picker_entries() -> Vec<crate::editor::resource_picker::PickerEntry> {
    use crate::editor::assets_browser::{ASSETS_ROOT, TileThumb};
    use crate::editor::resource_picker::PickerEntry;

    let dir = std::path::Path::new(ASSETS_ROOT).join("materials");
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<PickerEntry> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.to_string_lossy().to_lowercase().ends_with(".material.ron"))
        .map(|p| {
            let label = p
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.trim_end_matches(".material.ron").to_string())
                .unwrap_or_default();
            PickerEntry {
                key: p.to_string_lossy().into_owned(),
                label,
                thumb: TileThumb::Path(p),
            }
        })
        .collect();
    out.sort_by_key(|e| e.label.to_lowercase());
    out
}

/// The working-dir material file path backing a registry id, if any. Resolves the
/// table handle → its asset path → `assets/<...>`. Used to show the current selection
/// in the material picker.
pub fn material_path_for_registry_id(world: &World, registry_id: u32) -> Option<std::path::PathBuf> {
    let table = world.resource::<MaterialAssetTable>();
    let handle = table.handles.get(registry_id as usize)?;
    if handle.id() == Handle::<MaterialAsset>::default().id() {
        return None;
    }
    let asset_path = world.resource::<AssetServer>().get_path(handle.id())?;
    Some(std::path::Path::new(crate::editor::assets_browser::ASSETS_ROOT).join(asset_path.path()))
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

/// Discover all texture variants by scanning `assets/textures/<slug>/material.ron`.
pub fn discover_variants() -> Vec<LibraryVariant> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(TEXTURE_ROOT) else {
        return out;
    };
    let mut slugs: Vec<String> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter(|e| e.path().join("material.ron").is_file())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    slugs.sort();
    for slug in slugs {
        out.extend(read_manifest(&slug));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn pbrtex_entry_key_is_path_label_is_stem() {
        let e = pbrtex_entry(Path::new("pbrtextures/cobble_stone_3.pbrtex.ron"));
        assert_eq!(e.key, "pbrtextures/cobble_stone_3.pbrtex.ron");
        assert_eq!(e.label, "cobble_stone_3");
    }

    #[test]
    fn image_file_entry_key_is_path_label_is_name() {
        let e = image_file_entry(Path::new("textures/sand/1/height.png"));
        assert_eq!(e.key, "textures/sand/1/height.png");
        assert_eq!(e.label, "height.png");
    }

    #[test]
    fn is_image_file_matches_supported_formats() {
        assert!(is_image_file(Path::new("a.png")));
        assert!(is_image_file(Path::new("a.JPG")));
        assert!(is_image_file(Path::new("a.tga")));
        assert!(!is_image_file(Path::new("a.ron")));
        assert!(!is_image_file(Path::new("a")));
    }
}
