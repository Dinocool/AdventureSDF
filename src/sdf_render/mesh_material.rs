//! Custom baked-mesh material — **triplanar PBR** (the Transvoxel meshes have no UVs).
//!
//! An `ExtendedMaterial<StandardMaterial, MeshTriplanarExt>` so Bevy's PBR lighting (directional + ambient +
//! the planned Bevy GI) shades it for free; the extension fragment samples each material's diffuse + normal
//! TRIPLANAR (three world-axis projections blended by the surface normal — ported from the retired raymarch
//! `sdf::material` shader) and writes them into the StandardMaterial `PbrInput` before lighting. Metallic /
//! roughness / emissive / base-colour tint come from the StandardMaterial half (from `MaterialDef`).
//!
//! V1 is PER-CHUNK (each chunk's dominant material → one handle, cached in `MeshMaterials`); per-vertex
//! multi-material blend (top-2 from [`crate::sdf_render::edits::fold_csg_top2`]) over a shared texture array
//! is the next step.

use std::hash::Hasher;

use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

use crate::assets::MaterialTextureLibrary;
use crate::sdf_render::edits::MaterialRegistry;

/// World units → texture-tile scale (one tile per 2 world units), matching the raymarch `TEXTURE_WORLD_SCALE`.
const TEXTURE_WORLD_SCALE: f32 = 0.5;

/// The baked-mesh material type: StandardMaterial (PBR scalars + lighting) extended with triplanar texturing.
pub type MeshMaterial = ExtendedMaterial<StandardMaterial, MeshTriplanarExt>;

/// Per-material uniform for the extension (std140 16-byte aligned).
#[derive(ShaderType, Clone, Copy, Default, Debug, Reflect)]
pub struct MeshExtParams {
    /// World→UV scale for the triplanar projection.
    pub world_scale: f32,
    /// 1 if the diffuse texture is real (else the fragment keeps the StandardMaterial base colour).
    pub has_diffuse: u32,
    /// 1 if the normal texture is real (else the fragment keeps the geometric normal).
    pub has_normal: u32,
    pub _pad: u32,
}

/// The triplanar extension: a per-material uniform + a diffuse and normal texture (a 1×1 fallback when the
/// material has none — `has_*` gates their use).
#[derive(Asset, AsBindGroup, Reflect, Clone)]
pub struct MeshTriplanarExt {
    #[uniform(100)]
    pub params: MeshExtParams,
    #[texture(101)]
    #[sampler(102)]
    pub diffuse: Handle<Image>,
    #[texture(103)]
    #[sampler(104)]
    pub normal: Handle<Image>,
}

impl MaterialExtension for MeshTriplanarExt {
    fn fragment_shader() -> ShaderRef {
        "shaders/mesh_pbr.wgsl".into()
    }
    // The fragment is FORWARD-only (no `PREPASS_PIPELINE` branch), so keep it out of the depth/normal prepass
    // pipeline. These chunked meshes don't need to write the prepass yet (no SSAO/TAA dependency); shadows use
    // the separate shadow pass and still work. Re-enable + add the prepass branch when a prepass effect needs it.
    fn enable_prepass() -> bool {
        false
    }
}

/// Cache of one `MeshMaterial` handle per material id (its dominant-material handle), rebuilt when the
/// material registry changes. The mesh-bake commit picks a chunk's handle by its dominant material id.
#[derive(Resource, Default)]
pub struct MeshMaterials {
    /// `by_id[id]` = the handle for material id `id`.
    pub by_id: Vec<Handle<MeshMaterial>>,
    /// Shared unlit material for the "Colour by LOD" debug view (vertex colour = the per-LOD tint).
    pub debug: Handle<MeshMaterial>,
    /// Hash of the registry the cache was built from — rebuild when it changes.
    built_hash: u64,
}

/// Plugin: registers the `MeshMaterial` pipeline, the cache, the rebuild system, and a modest ambient light
/// so shadowed/indirect areas aren't black until Bevy GI lands.
pub struct MeshMaterialPlugin;

impl Plugin for MeshMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<MeshMaterial>::default())
            .register_type::<MeshExtParams>()
            .init_resource::<MeshMaterials>()
            .add_systems(Update, (rebuild_mesh_materials, ensure_camera_ambient));
    }
}

/// `AmbientLight` is a per-camera component in Bevy 0.18 — give every 3D camera a modest ambient term so
/// shadowed / indirect areas aren't pure black until the planned Bevy GI lands. (Idempotent: skips cameras
/// that already have one, so a hand-authored value wins.)
fn ensure_camera_ambient(mut commands: Commands, cams: Query<Entity, (With<Camera3d>, Without<AmbientLight>)>) {
    for e in &cams {
        commands.entity(e).insert(AmbientLight { brightness: 250.0, ..default() });
    }
}

/// Quantised hash of the registry's material-relevant fields (so a colour/PBR/texture edit rebuilds the
/// handles, but per-frame transform jitter doesn't).
fn registry_hash(reg: &MaterialRegistry) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write_usize(reg.defs.len());
    for d in &reg.defs {
        let l = d.base_color.to_linear();
        for v in [l.red, l.green, l.blue, d.metallic, d.roughness, d.emissive.x, d.emissive.y, d.emissive.z] {
            h.write_i64((v as f64 * 1.0e4) as i64);
        }
        for t in d.tex_layers {
            h.write_u32(t);
        }
    }
    h.finish()
}

/// (Re)build the per-material `MeshMaterial` handles when the registry changes. Loads each material's source
/// diffuse/normal PNGs (resolved via `MaterialTextureLibrary.variants[layer]`) as Bevy Images; an absent map
/// uses a 1×1 fallback gated off by `has_*`.
/// Load a texture with a TILING (Repeat) linear sampler — triplanar samples at `world_pos·scale`, far outside
/// `[0,1]`, so the maps must wrap; the default ClampToEdge sampler stretches one edge texel across the surface.
fn load_tiling(assets: &AssetServer, path: &std::path::Path) -> Handle<Image> {
    use bevy::image::{
        ImageAddressMode, ImageFilterMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor,
    };
    assets.load_with_settings(path.to_path_buf(), |s: &mut ImageLoaderSettings| {
        s.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::Repeat,
            address_mode_v: ImageAddressMode::Repeat,
            address_mode_w: ImageAddressMode::Repeat,
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            mipmap_filter: ImageFilterMode::Linear,
            ..default()
        });
    })
}

pub(crate) fn rebuild_mesh_materials(
    reg: Res<MaterialRegistry>,
    library: Res<MaterialTextureLibrary>,
    assets: Res<AssetServer>,
    mut mats: ResMut<Assets<MeshMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut cache: ResMut<MeshMaterials>,
) {
    let hash = registry_hash(&reg);
    if !cache.by_id.is_empty() && cache.built_hash == hash {
        return;
    }
    cache.built_hash = hash;

    // 1×1 white fallback image used when a material lacks a map (the fragment gates it off via `has_*`).
    let white = images.add(Image::new_fill(
        bevy::render::render_resource::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        bevy::render::render_resource::TextureDimension::D2,
        &[255, 255, 255, 255],
        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
        bevy::asset::RenderAssetUsages::RENDER_WORLD,
    ));

    cache.by_id.clear();
    for def in &reg.defs {
        // Source texture paths for this material (its MapSet, via the diffuse layer index).
        let map_set = (def.tex_layers[0] != u32::MAX)
            .then(|| library.variants.get(def.tex_layers[0] as usize))
            .flatten();
        // Load with a REPEAT sampler — triplanar UVs are `world_pos·scale` (far outside [0,1]), so the
        // textures must tile; the default ClampToEdge sampler would stretch one texel across the surface.
        let diffuse = map_set.and_then(|m| m.diffuse.as_ref()).map(|p| load_tiling(&assets, p));
        let normal = map_set.and_then(|m| m.normal.as_ref()).map(|p| load_tiling(&assets, p));

        let base = StandardMaterial {
            // White when textured (the texture supplies colour); else the authored tint.
            base_color: if diffuse.is_some() { Color::WHITE } else { def.base_color },
            metallic: def.metallic,
            perceptual_roughness: def.roughness.max(0.045),
            emissive: LinearRgba::rgb(def.emissive.x, def.emissive.y, def.emissive.z),
            double_sided: true,
            cull_mode: None,
            ..default()
        };
        let ext = MeshTriplanarExt {
            params: MeshExtParams {
                world_scale: TEXTURE_WORLD_SCALE,
                has_diffuse: diffuse.is_some() as u32,
                has_normal: normal.is_some() as u32,
                _pad: 0,
            },
            diffuse: diffuse.unwrap_or_else(|| white.clone()),
            normal: normal.unwrap_or_else(|| white.clone()),
        };
        cache.by_id.push(mats.add(MeshMaterial { base, extension: ext }));
    }

    // Debug (Colour by LOD): unlit white StandardMaterial; the per-LOD tint rides on the vertex colour.
    cache.debug = mats.add(MeshMaterial {
        base: StandardMaterial { base_color: Color::WHITE, unlit: true, double_sided: true, cull_mode: None, ..default() },
        extension: MeshTriplanarExt {
            params: MeshExtParams { world_scale: TEXTURE_WORLD_SCALE, has_diffuse: 0, has_normal: 0, _pad: 0 },
            diffuse: white.clone(),
            normal: white,
        },
    });
}
