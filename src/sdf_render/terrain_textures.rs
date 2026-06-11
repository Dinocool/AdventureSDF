//! Terrain PBR **texture arrays** — the per-material `diffuse`/`normal`/`MRA` `texture_2d_array`s the
//! terrain-surface shader triplanar-samples by baked material id (Stage 5 of the terrain-materials feature).
//!
//! ONE array layer per [`TerrainSurfaceMaterial`] (layer index == `TerrainMatId.0`), assembled from the
//! material's authored `texture` set (`assets/textures/<slug>/<variant>/{diffuse,normal,metallic,roughness,
//! ao}.png`). A material with no `texture` gets a flat fallback layer — the shader reads its `has_tex` palette
//! flag and uses the flat `base_color` instead. Mirrors the mesh material's [`crate::sdf_render::mesh_material`]
//! array pipeline (reusing its generic assembly helpers): a 1×1 fallback array is bound immediately so the
//! material is valid before the PNGs finish loading, then the real arrays assemble + swap in once ready.

use bevy::prelude::*;

use super::mesh_material::{LayerSrc, array_image, fallback_array, layer_mra, layer_rgba, load_src};
use super::worldgen::biome::BiomeLibrary;

/// The shared terrain texture arrays (one layer per palette material), rebuilt when the library changes.
/// `ready` flips true once the assembled arrays (not the fallback) are live — the material-sync system then
/// pushes the real handles into every live `TerrainMaterial`.
#[derive(Resource, Default)]
pub struct TerrainTextureArrays {
    pub diffuse: Handle<Image>,
    pub normal: Handle<Image>,
    pub mra: Handle<Image>,
    /// Fingerprint of the material texture set the arrays were built from / are loading for.
    hash: u64,
    /// True once the assembled (non-fallback) arrays are live.
    pub ready: bool,
    /// Source PNG handles being loaded (one [`LayerSrc`] per material), or empty when idle/assembled.
    pending: Vec<LayerSrc>,
}

/// Fingerprint the per-material texture assignments (slug/variant + tiling) so the arrays only rebuild when
/// the authored textures change (a `biomes.ron` edit), not every frame.
fn tex_hash(lib: &BiomeLibrary) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write_usize(lib.materials.len());
    for m in &lib.materials {
        m.texture.hash(&mut h);
    }
    h.finish()
}

/// Build the terrain diffuse/normal/MRA arrays from the live [`BiomeLibrary`] (async: phase 1 starts the
/// source PNG loads, phase 2 assembles the `texture_2d_array`s once every source is in). One layer per
/// material in palette order; an untextured material contributes a flat fallback layer.
pub fn build_terrain_texture_arrays(
    lib: Res<BiomeLibrary>,
    assets: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    mut arr: ResMut<TerrainTextureArrays>,
) {
    // Seed the fallback arrays once so the material binds before any real texture loads.
    if arr.diffuse == Handle::default() {
        arr.diffuse = fallback_array(&mut images, [255, 255, 255, 255], true);
        arr.normal = fallback_array(&mut images, [128, 128, 255, 255], false);
        arr.mra = fallback_array(&mut images, [0, 255, 255, 255], false);
    }

    let hash = tex_hash(&lib);
    // Phase 1: a new/changed texture set → start loading each material's source maps.
    if hash != arr.hash && !lib.materials.is_empty() {
        arr.hash = hash;
        arr.ready = false;
        arr.pending = lib
            .materials
            .iter()
            .map(|m| {
                // `texture` is "<slug>/<variant>" under assets/textures/; None ⇒ all-fallback layer.
                let dir = m.texture.as_ref().map(|t| format!("textures/{t}"));
                let load = |map: &str, srgb: bool| {
                    dir.as_ref().map(|d| load_src(&assets, std::path::Path::new(&format!("{d}/{map}.png")), srgb))
                };
                LayerSrc {
                    diffuse: load("diffuse", true),
                    normal: load("normal", false),
                    metallic: load("metallic", false),
                    roughness: load("roughness", false),
                    ao: load("ao", false),
                }
            })
            .collect();
    }
    if arr.pending.is_empty() {
        return;
    }
    // Phase 2: assemble once every source image is loaded.
    let loaded = |h: &Option<Handle<Image>>| h.as_ref().is_none_or(|h| images.get(h).is_some());
    let all = arr.pending.iter().all(|l| {
        loaded(&l.diffuse) && loaded(&l.normal) && loaded(&l.metallic) && loaded(&l.roughness) && loaded(&l.ao)
    });
    if !all {
        return;
    }
    // Common size = the largest source diffuse, clamped + power-of-two (resize_exact handles the rest).
    let size = arr
        .pending
        .iter()
        .filter_map(|l| l.diffuse.as_ref().and_then(|h| images.get(h)))
        .map(|i: &Image| i.width().max(i.height()))
        .max()
        .unwrap_or(512)
        .clamp(64, 2048)
        .next_power_of_two();

    let pending = std::mem::take(&mut arr.pending);
    let (mut diff, mut norm, mut mra) = (Vec::new(), Vec::new(), Vec::new());
    for l in &pending {
        diff.push(layer_rgba(&images, &l.diffuse, size, [255, 255, 255, 255]));
        norm.push(layer_rgba(&images, &l.normal, size, [128, 128, 255, 255]));
        mra.push(layer_mra(&images, l, size));
    }
    arr.diffuse = array_image(&mut images, &diff, size, true);
    arr.normal = array_image(&mut images, &norm, size, false);
    arr.mra = array_image(&mut images, &mra, size, false);
    arr.ready = true;
}
