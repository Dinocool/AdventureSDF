//! Material compile step: flatten loaded [`MaterialAsset`]s (+ resolved texture
//! layers) into the GPU-facing [`MaterialRegistry`]. This is the one place the
//! editable asset world meets the render world's flat material table — the GPU
//! upload path (`prepare_sdf_camera_data`) is untouched and just reacts to
//! `MaterialRegistry` change detection.

use bevy::prelude::*;

use super::{MaterialAsset, MaterialAssetTable, MaterialTextureLibrary};
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
fn compile_materials(
    mut events: MessageReader<AssetEvent<MaterialAsset>>,
    table: Res<MaterialAssetTable>,
    assets: Res<Assets<MaterialAsset>>,
    mut library: ResMut<MaterialTextureLibrary>,
    mut registry: ResMut<MaterialRegistry>,
) {
    // Recompile when an asset (re)loaded/changed, or the table grew (new material
    // registered this frame), or on first run while the registry is unpopulated.
    let asset_changed = events.read().count() > 0;
    let table_grew = table.is_changed();
    if !asset_changed && !table_grew && !registry.defs.is_empty() {
        return;
    }

    // Row 0 is the fallback; rows 1.. mirror the asset table by stable id.
    let mut defs = vec![MaterialDef::default()];
    for handle in table.handles.iter().skip(1) {
        let def = match assets.get(handle) {
            Some(asset) => MaterialDef {
                base_color: asset.color(),
                blend_softness: asset.blend_softness,
                tex_layers: std::array::from_fn(|m| resolve(&mut library, asset, m)),
            },
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

/// Resolve PBR map `m` of `asset` to a texture-array layer (or `u32::MAX`).
fn resolve(library: &mut MaterialTextureLibrary, asset: &MaterialAsset, m: usize) -> u32 {
    match &asset.maps[m] {
        Some(t) => library.resolve_layer(&t.slug, &t.dir),
        None => u32::MAX,
    }
}

/// Field-wise equality for the registry rows (avoids a needless GPU re-upload when
/// the recompiled table is identical). `MaterialDef` is `Copy` with simple fields.
fn defs_equal(a: &[MaterialDef], b: &[MaterialDef]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.base_color == y.base_color
                && x.blend_softness == y.blend_softness
                && x.tex_layers == y.tex_layers
        })
}

const _: () = assert!(MATERIAL_TEX_MAPS == 5);
