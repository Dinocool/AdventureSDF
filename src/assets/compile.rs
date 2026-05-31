//! Material compile step: flatten loaded [`MaterialAsset`]s (+ resolved texture
//! layers) into the GPU-facing [`MaterialRegistry`]. This is the one place the
//! editable asset world meets the render world's flat material table — the GPU
//! upload path (`prepare_sdf_camera_data`) is untouched and just reacts to
//! `MaterialRegistry` change detection.

use bevy::prelude::*;

use super::{
    MaterialAsset, MaterialAssetTable, MaterialTextureLibrary, PbrTextureAsset, PbrTextureHandles,
};
use crate::sdf_render::edits::{MATERIAL_TEX_MAPS, MaterialDef, MaterialRegistry};

/// Register the compile systems. Runs in `Update`: rebuild the registry whenever a
/// material asset is added/modified or the asset table grows. Ordered before
/// `prepare_sdf_camera_data` (which lives in `render.rs`, also `Update`) via the
/// `is_changed()` reaction — no explicit ordering needed since the registry change
/// is picked up the same or next frame.
pub fn register(app: &mut App) {
    app.add_systems(Update, compile_materials);
}

/// Rebuild `MaterialRegistry::defs` from the registered material assets. Each
/// `registry_id` (table index) maps to one `MaterialDef` row; unresolved (still
/// loading) assets keep the fallback so the scene renders during async load.
#[allow(clippy::too_many_arguments)]
fn compile_materials(
    mut mat_events: MessageReader<AssetEvent<MaterialAsset>>,
    mut tex_events: MessageReader<AssetEvent<PbrTextureAsset>>,
    table: Res<MaterialAssetTable>,
    assets: Res<Assets<MaterialAsset>>,
    pbr_textures: Res<Assets<PbrTextureAsset>>,
    mut pbr_handles: ResMut<PbrTextureHandles>,
    asset_server: Res<AssetServer>,
    mut library: ResMut<MaterialTextureLibrary>,
    mut registry: ResMut<MaterialRegistry>,
) {
    // Recompile when a material OR a referenced PBR-texture (re)loaded/changed, the
    // table grew, or on first run while the registry is unpopulated.
    let asset_changed = mat_events.read().count() > 0 || tex_events.read().count() > 0;
    let table_grew = table.is_changed();
    if !asset_changed && !table_grew && !registry.defs.is_empty() {
        return;
    }

    // Row 0 is the fallback; rows 1.. mirror the asset table by stable id.
    let mut defs = vec![MaterialDef::default()];
    for handle in table.handles.iter().skip(1) {
        let def = match assets.get(handle) {
            Some(asset) => {
                let layer = resolve_layer(
                    asset,
                    &pbr_textures,
                    &mut pbr_handles,
                    &asset_server,
                    &mut library,
                );
                MaterialDef {
                    base_color: asset.color(),
                    blend_softness: asset.blend_softness,
                    metallic: asset.metallic,
                    roughness: asset.roughness,
                    parallax_scale: asset.parallax_scale,
                    tex_layers: [layer; MATERIAL_TEX_MAPS],
                }
            }
            // Still loading: leave a fallback row; an AssetEvent::Added re-runs us.
            None => MaterialDef::default(),
        };
        defs.push(def);
    }

    // Only write through (triggering `is_changed`) if something actually differs,
    // so we don't force a GPU re-upload every frame the table is marked changed.
    if registry.defs.len() != defs.len() || !defs_equal(&registry.defs, &defs) {
        registry.defs = defs;
    }
}

/// Resolve a material's effective texture to a single GPU array layer (or `u32::MAX`):
/// load its `.pbrtex.ron` bundle, merge its per-role overrides on top, and hand the
/// resulting [`MapSet`](super::MapSet) to the library. While the bundle is still loading
/// we resolve with the overrides alone (so an override-only material still textures, and
/// the bundle layers in once `AssetEvent::Added` re-runs us).
fn resolve_layer(
    asset: &MaterialAsset,
    pbr_textures: &Assets<PbrTextureAsset>,
    pbr_handles: &mut PbrTextureHandles,
    asset_server: &AssetServer,
    library: &mut MaterialTextureLibrary,
) -> u32 {
    let bundle = asset.texture.as_ref().and_then(|path| {
        // Cache a STRONG handle so the bundle stays resident (a load-and-drop here would
        // unload it every frame → `get` forever `None`).
        let handle = pbr_handles.ensure(path, asset_server);
        pbr_textures.get(&handle).cloned()
    });
    let effective = bundle.unwrap_or_default().merge(&asset.overrides);
    library.resolve_layer(&effective.to_map_set())
}

/// Field-wise equality for the registry rows (avoids a needless GPU re-upload when
/// the recompiled table is identical). `MaterialDef` is `Copy` with simple fields.
fn defs_equal(a: &[MaterialDef], b: &[MaterialDef]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.base_color == y.base_color
                && x.blend_softness == y.blend_softness
                && x.metallic == y.metallic
                && x.roughness == y.roughness
                && x.parallax_scale == y.parallax_scale
                && x.tex_layers == y.tex_layers
        })
}

const _: () = assert!(MATERIAL_TEX_MAPS == 5);
