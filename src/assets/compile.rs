//! Material resolve step: walk every SDF volume's authored [`SdfMaterialSource`], compute
//! one effective [`MaterialDef`] per *unique* source (base file + per-field overrides, or a
//! fully-inline material), dedupe identical sources to a single GPU row, and write the
//! derived [`SdfMaterial`] `registry_id` back onto each volume. This is the one place the
//! editable asset/scene world meets the render world's flat material table — the GPU upload
//! (`prepare_sdf_camera_data`) just reacts to [`MaterialRegistry`] change detection.
//!
//! Replaces the old fixed `compile_materials` (which mirrored a hardcoded asset table 1:1):
//! materials are now driven entirely by what the scene contains, so adding/removing/editing
//! a volume's material reshapes the registry dynamically.

use std::collections::HashMap;
use std::path::PathBuf;

use bevy::prelude::*;

use super::{
    MaterialAsset, MaterialAssetTable, MaterialTextureLibrary, PbrTextureAsset, PbrTextureHandles,
};
use crate::sdf_render::edits::{
    MATERIAL_TEX_MAPS, MaterialDef, MaterialFields, MaterialRegistry, SdfMaterial, SdfMaterialSource,
};

/// Register the resolve system.
pub fn register(app: &mut App) {
    app.add_systems(Update, resolve_materials);
}

/// Dedup key for a material source. Floats aren't `Hash`/`Eq`, so override scalars are
/// keyed by their bit pattern; identical sources (same file, same overrides) collapse to
/// one registry row (e.g. the gallery's cobble used on two volumes).
#[derive(Clone, PartialEq, Eq, Hash)]
struct SourceKey {
    asset: Option<PathBuf>,
    /// `[base_color(4) + metallic + roughness + blend_softness + parallax_scale]`, each
    /// `Some(bits)` or `None`.
    color: [Option<u32>; 4],
    scalars: [Option<u32>; 4],
    /// Emissive override RGB bit-pattern (`Some` when set), keyed so two volumes with
    /// different emissive overrides don't collapse to one registry row.
    emissive: [Option<u32>; 3],
}

impl SourceKey {
    fn of(src: &SdfMaterialSource) -> Self {
        let f = &src.overrides;
        let color = match f.base_color {
            Some(c) => [Some(c[0].to_bits()), Some(c[1].to_bits()), Some(c[2].to_bits()), Some(c[3].to_bits())],
            None => [None; 4],
        };
        let emissive = match f.emissive {
            Some(e) => [Some(e[0].to_bits()), Some(e[1].to_bits()), Some(e[2].to_bits())],
            None => [None; 3],
        };
        SourceKey {
            asset: src.asset.clone(),
            color,
            scalars: [
                f.metallic.map(f32::to_bits),
                f.roughness.map(f32::to_bits),
                f.blend_softness.map(f32::to_bits),
                f.parallax_scale.map(f32::to_bits),
            ],
            emissive,
        }
    }
}

/// Rebuild `MaterialRegistry::defs` from every volume's `SdfMaterialSource` and write the
/// derived `registry_id` onto each volume. Re-runs when a volume's source changes, a
/// volume with a source is removed, a referenced asset (re)loaded/changed, or on first run.
#[allow(clippy::too_many_arguments)]
fn resolve_materials(
    mut commands: Commands,
    mut mat_events: MessageReader<AssetEvent<MaterialAsset>>,
    mut tex_events: MessageReader<AssetEvent<PbrTextureAsset>>,
    removed: RemovedComponents<SdfMaterialSource>,
    volumes: Query<(Entity, &SdfMaterialSource, Option<&SdfMaterial>)>,
    changed: Query<(), Changed<SdfMaterialSource>>,
    assets: Res<Assets<MaterialAsset>>,
    pbr_textures: Res<Assets<PbrTextureAsset>>,
    mut pbr_handles: ResMut<PbrTextureHandles>,
    asset_server: Res<AssetServer>,
    mut table: ResMut<MaterialAssetTable>,
    mut library: ResMut<MaterialTextureLibrary>,
    mut registry: ResMut<MaterialRegistry>,
) {
    let _span = crate::instrument::span("material compile");
    // Dirty if: any volume's source changed, a sourced volume was removed, a material/texture
    // asset event fired, or the registry is still unpopulated (first run).
    let asset_changed = mat_events.read().count() > 0 || tex_events.read().count() > 0;
    let dirty =
        asset_changed || !removed.is_empty() || !changed.is_empty() || registry.defs.len() <= 1;
    if !dirty {
        return;
    }

    table.ensure_fallback();

    // One registry row per unique source. Row 0 stays the fallback.
    let mut defs = vec![MaterialDef::default()];
    let mut id_of: HashMap<SourceKey, u32> = HashMap::new();
    // Per-entity target id, applied after the borrow on `volumes` ends.
    let mut assignments: Vec<(Entity, u32, Option<u32>)> = Vec::new();

    for (entity, source, current) in &volumes {
        let key = SourceKey::of(source);
        let id = if let Some(&id) = id_of.get(&key) {
            id
        } else {
            let id = defs.len() as u32;
            let def = resolve_def(
                source,
                &mut table,
                &assets,
                &pbr_textures,
                &mut pbr_handles,
                &asset_server,
                &mut library,
            );
            defs.push(def);
            id_of.insert(key, id);
            id
        };
        assignments.push((entity, id, current.map(|m| m.registry_id)));
    }

    // Write the derived id back ONLY when it changed — `schedule_bakes` keys rebakes off
    // `Changed<SdfMaterial>`, so a spurious insert every dirty tick would force needless
    // rebakes.
    for (entity, id, current) in assignments {
        if current != Some(id) {
            commands.entity(entity).insert(SdfMaterial { registry_id: id });
        }
    }

    // Only write through the registry (triggering `is_changed` → GPU re-upload) when the
    // rows actually differ.
    if registry.defs.len() != defs.len() || !defs_equal(&registry.defs, &defs) {
        registry.defs = defs;
    }
}

/// Build one [`MaterialDef`] for a source: start from its base file (if any), then apply
/// the per-field overrides. An inline source (`asset: None`) starts from defaults.
#[allow(clippy::too_many_arguments)]
fn resolve_def(
    source: &SdfMaterialSource,
    table: &mut MaterialAssetTable,
    assets: &Assets<MaterialAsset>,
    pbr_textures: &Assets<PbrTextureAsset>,
    pbr_handles: &mut PbrTextureHandles,
    asset_server: &AssetServer,
    library: &mut MaterialTextureLibrary,
) -> MaterialDef {
    // Base: the file material if present + loaded, else defaults (+ no texture).
    let mut def = match &source.asset {
        Some(path) => {
            // Load + register the file so a STRONG handle keeps it resident (a load-and-drop
            // would unload it every frame → `assets.get` forever None). Registration is now
            // residency-only; the registry id comes from the source dedup, not the table.
            let handle = asset_server.load::<MaterialAsset>(path.clone());
            table.register(handle.clone());
            match assets.get(&handle) {
                Some(asset) => {
                    let layer =
                        resolve_layer(asset, pbr_textures, pbr_handles, asset_server, library);
                    MaterialDef {
                        base_color: asset.color(),
                        blend_softness: asset.blend_softness,
                        metallic: asset.metallic,
                        roughness: asset.roughness,
                        parallax_scale: asset.parallax_scale,
                        // Premultiply intensity here so the shader just adds `emissive`.
                        emissive: Vec3::from(asset.emissive_color) * asset.emissive_intensity,
                        tex_layers: [layer; MATERIAL_TEX_MAPS],
                    }
                }
                // Still loading: defaults for now; an `AssetEvent::Added` re-runs us.
                None => MaterialDef::default(),
            }
        }
        None => MaterialDef::default(),
    };

    apply_overrides(&mut def, &source.overrides);
    def
}

/// Apply scalar/colour overrides on top of a base [`MaterialDef`]. Texture layers are not
/// overridable (they always come from the base file) — see [`MaterialFields`].
fn apply_overrides(def: &mut MaterialDef, o: &MaterialFields) {
    if let Some(c) = o.base_color {
        def.base_color = Color::linear_rgba(c[0], c[1], c[2], c[3]);
    }
    if let Some(v) = o.metallic {
        def.metallic = v;
    }
    if let Some(v) = o.roughness {
        def.roughness = v;
    }
    if let Some(v) = o.blend_softness {
        def.blend_softness = v;
    }
    if let Some(v) = o.parallax_scale {
        def.parallax_scale = v;
    }
    if let Some(e) = o.emissive {
        def.emissive = Vec3::from(e);
    }
}

/// Resolve a material's effective texture to a single GPU array layer (or `u32::MAX`):
/// load its `.pbrtex.ron` bundle, merge its per-role overrides on top, and hand the
/// resulting [`MapSet`](super::MapSet) to the library. While the bundle is still loading we
/// resolve with the overrides alone (so an override-only material still textures, and the
/// bundle layers in once `AssetEvent::Added` re-runs us).
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

/// Field-wise equality for the registry rows (avoids a needless GPU re-upload when the
/// recompiled table is identical). `MaterialDef` is `Copy` with simple fields.
fn defs_equal(a: &[MaterialDef], b: &[MaterialDef]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.base_color == y.base_color
                && x.blend_softness == y.blend_softness
                && x.metallic == y.metallic
                && x.roughness == y.roughness
                && x.parallax_scale == y.parallax_scale
                && x.emissive == y.emissive
                && x.tex_layers == y.tex_layers
        })
}

const _: () = assert!(MATERIAL_TEX_MAPS == 5);
