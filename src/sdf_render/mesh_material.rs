//! Custom baked-mesh material — **per-vertex multi-material triplanar PBR** (the Transvoxel meshes have no
//! UVs). ONE shared `ExtendedMaterial<StandardMaterial, MeshBlendExt>` for every chunk: each vertex carries
//! its top-2 material ids (`fold_csg_top2`) in `ATTRIBUTE_UV_0` and a blend weight in the COLOUR alpha; the
//! fragment biplanar-samples both materials from shared texture ARRAYS and cross-fades, then writes albedo /
//! normal / metallic / roughness into the StandardMaterial `PbrInput` so Bevy PBR (directional + ambient +
//! future Bevy GI) shades it. The triplanar math is ported from the retired raymarch `sdf::material` shader.
//!
//! Pipeline: a build system loads each resolved material's source PNGs (`MaterialTextureLibrary` map-sets),
//! resizes + (for MRA) packs + MIP-chains them into three `texture_2d_array` `Image`s (diffuse sRGB, normal
//! + MRA linear), and a per-material storage table {layer, base_color, emissive, metallic, roughness,
//! texture_scale, blend_softness}. The shared material binds all of it; the bake just stamps vertex ids.

use std::hash::Hasher;
use std::path::Path;

use bevy::asset::RenderAssetUsages;
use bevy::image::{
    ImageAddressMode, ImageFilterMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor,
};
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat, TextureViewDescriptor,
    TextureViewDimension,
};
use bevy::render::storage::ShaderStorageBuffer;
use bevy::shader::ShaderRef;

use crate::assets::MaterialTextureLibrary;
use crate::sdf_render::edits::MaterialRegistry;
use crate::sdf_render::mesh_bake::MeshBakeConfig;

/// `u32::MAX` material-table sentinel = "no texture for this map" (the fragment falls back to scalars).
const NO_LAYER: u32 = u32::MAX;

/// The baked-mesh material type: StandardMaterial (PBR lighting) extended with the per-vertex triplanar blend.
pub type MeshMaterial = ExtendedMaterial<StandardMaterial, MeshBlendExt>;

/// Per-material GPU row in the material table (std430; 64 B, 16-aligned).
#[derive(ShaderType, Clone, Copy, Default, Debug)]
pub struct MeshMatGpu {
    /// Linear base colour (tints the diffuse texture, or IS the colour when untextured).
    base_color: Vec4,
    /// Linear emissive radiance (xyz; w spare).
    emissive: Vec4,
    /// Texture-array layer for all three maps (`NO_LAYER` = untextured → use scalars).
    layer: u32,
    has_diffuse: u32,
    has_normal: u32,
    has_mra: u32,
    metallic: f32,
    roughness: f32,
    /// Triplanar UV tile density (world→UV).
    texture_scale: f32,
    /// Blend band width (world units) for the A/B cross-fade.
    blend_softness: f32,
}

/// Per-draw uniform for the blend extension.
#[derive(ShaderType, Clone, Copy, Default, Debug, Reflect)]
pub struct BlendParams {
    /// 1 = "Colour by LOD" debug (unlit, vertex-colour tint); 0 = lit PBR.
    debug_lod: u32,
    /// Pad to 16 B (uniform min binding). A `vec3<u32>` (NOT `[u32; 3]`) — a uniform array element's stride
    /// must be 16-aligned, which a `u32` array can't satisfy (encase panics); `UVec3` is one 16-aligned vec.
    _pad: UVec3,
}

/// The blend extension: the three shared texture arrays + sampler + the per-material table + a debug flag.
/// `TypePath` (not full `Reflect`) satisfies `Asset` without forcing every bound type (`MeshMatGpu`) to be
/// `Reflect` — the GPU table is plain `ShaderType` data, never reflected.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
pub struct MeshBlendExt {
    #[uniform(100)]
    pub params: BlendParams,
    #[texture(101, dimension = "2d_array")]
    #[sampler(102)]
    pub diffuse: Handle<Image>,
    #[texture(103, dimension = "2d_array")]
    pub normal: Handle<Image>,
    #[texture(104, dimension = "2d_array")]
    pub mra: Handle<Image>,
    #[storage(105, read_only)]
    pub table: Handle<ShaderStorageBuffer>,
}

impl MaterialExtension for MeshBlendExt {
    fn fragment_shader() -> ShaderRef {
        "shaders/mesh_pbr.wgsl".into()
    }
    // Forward-only fragment (no PREPASS_PIPELINE branch) → keep it out of the depth/normal prepass pipeline.
    fn enable_prepass() -> bool {
        false
    }
}

/// The single shared mesh-material handle every chunk uses + the empty-array placeholder.
#[derive(Resource, Default)]
pub struct MeshMaterials {
    pub handle: Handle<MeshMaterial>,
    /// Registry hash the material table was built from (rebuild on change).
    table_hash: u64,
}

/// The shared 1-layer fallback + the assembled texture arrays, rebuilt when the material library changes.
#[derive(Resource, Default)]
pub(crate) struct MeshTextureArrays {
    diffuse: Handle<Image>,
    normal: Handle<Image>,
    mra: Handle<Image>,
    /// Hash of the library variants the arrays were built from / are loading for.
    hash: u64,
    /// True once the assembled arrays (not the fallback) are live.
    ready: bool,
    /// Source PNG handles being loaded (one set per layer), or empty when idle/assembled.
    pending: Vec<LayerSrc>,
}

/// The five source-PNG handles for one texture-array layer (a resolved material map-set).
struct LayerSrc {
    diffuse: Option<Handle<Image>>,
    normal: Option<Handle<Image>>,
    metallic: Option<Handle<Image>>,
    roughness: Option<Handle<Image>>,
    ao: Option<Handle<Image>>,
}

/// Plugin: the material pipeline + the array/table/material build systems + a per-camera ambient light.
pub struct MeshMaterialPlugin;

impl Plugin for MeshMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<MeshMaterial>::default())
            .register_type::<BlendParams>()
            .init_resource::<MeshMaterials>()
            .init_resource::<MeshTextureArrays>()
            .add_systems(
                Update,
                (
                    // ORDER MATTERS: `build_texture_arrays` seeds the `D2Array` fallback array images; if
                    // `rebuild_mesh_material` ran first it would bind `Handle::default()` (Bevy's builtin D2
                    // white image) into the `2d_array` slots → a wgpu D2-vs-D2Array validation panic.
                    build_texture_arrays,
                    rebuild_mesh_material.after(build_texture_arrays),
                    ensure_camera_ambient,
                ),
            );
    }
}

/// `AmbientLight` is a per-camera component in Bevy 0.18 — give every 3D camera a modest ambient so shadowed
/// / indirect areas aren't pure black until Bevy GI lands. (Idempotent; a hand-authored value wins.)
fn ensure_camera_ambient(mut commands: Commands, cams: Query<Entity, (With<Camera3d>, Without<AmbientLight>)>) {
    for e in &cams {
        commands.entity(e).insert(AmbientLight { brightness: 250.0, ..default() });
    }
}

// ─────────────────────────── texture-array assembly ───────────────────────────

/// Force an `Image`'s GPU view to be `2d_array` — a 2D texture (even with >1 layer) otherwise defaults to a
/// plain `D2` view, which mismatches the `texture_2d_array` binding (a wgpu validation panic).
fn as_d2_array_view(img: &mut Image) {
    img.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
}

/// A 1×1 layer-0 fallback array image (so the material is bindable before the real arrays finish loading).
fn fallback_array(images: &mut Assets<Image>, fill: [u8; 4], srgb: bool) -> Handle<Image> {
    let fmt = if srgb { TextureFormat::Rgba8UnormSrgb } else { TextureFormat::Rgba8Unorm };
    let mut img = Image::new(
        Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        TextureDimension::D2,
        fill.to_vec(),
        fmt,
        RenderAssetUsages::RENDER_WORLD,
    );
    img.sampler = tiling_sampler();
    as_d2_array_view(&mut img);
    images.add(img)
}

/// A tiling (Repeat) trilinear sampler — triplanar UVs leave `[0,1]`, and mipmaps need linear mip filtering.
fn tiling_sampler() -> ImageSampler {
    ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        address_mode_w: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    })
}

/// Load a source PNG readable in the main world (so we can assemble it into an array), with the right colour
/// space (`srgb` for diffuse, linear for normal/MRA channels).
fn load_src(assets: &AssetServer, path: &Path, srgb: bool) -> Handle<Image> {
    assets.load_with_settings(path.to_path_buf(), move |s: &mut ImageLoaderSettings| {
        s.is_srgb = srgb;
        s.asset_usage = RenderAssetUsages::all();
    })
}

fn variants_hash(library: &MaterialTextureLibrary) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write_usize(library.variants.len());
    for v in &library.variants {
        h.write(v.label().as_bytes());
    }
    h.finish()
}

/// Read a loaded image as RGBA8 resized to `size`, or `fill` repeated if absent/unreadable.
fn layer_rgba(images: &Assets<Image>, h: &Option<Handle<Image>>, size: u32, fill: [u8; 4]) -> Vec<u8> {
    let n = (size * size) as usize;
    let resized = h.as_ref().and_then(|h| images.get(h)).and_then(|img| {
        img.clone()
            .try_into_dynamic()
            .ok()
            .map(|d| d.resize_exact(size, size, image::imageops::FilterType::Triangle).to_rgba8().into_raw())
    });
    resized.unwrap_or_else(|| fill.iter().copied().cycle().take(n * 4).collect())
}

/// Build a layer's MRA (metal/rough/AO) by packing the R channel of three source maps (defaults 0/255/255).
fn layer_mra(images: &Assets<Image>, l: &LayerSrc, size: u32) -> Vec<u8> {
    let m = layer_rgba(images, &l.metallic, size, [0, 0, 0, 255]);
    let r = layer_rgba(images, &l.roughness, size, [255, 255, 255, 255]);
    let a = layer_rgba(images, &l.ao, size, [255, 255, 255, 255]);
    let n = (size * size) as usize;
    let mut out = vec![0u8; n * 4];
    for i in 0..n {
        out[i * 4] = m[i * 4];
        out[i * 4 + 1] = r[i * 4];
        out[i * 4 + 2] = a[i * 4];
        out[i * 4 + 3] = 255;
    }
    out
}

/// Box-downsample one RGBA8 mip level (`w`×`h`) to half size.
fn downsample(src: &[u8], w: u32, h: u32) -> Vec<u8> {
    let (hw, hh) = ((w / 2).max(1), (h / 2).max(1));
    let mut out = vec![0u8; (hw * hh * 4) as usize];
    for y in 0..hh {
        for x in 0..hw {
            for c in 0..4 {
                let mut s = 0u32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let sx = (x * 2 + dx).min(w - 1);
                        let sy = (y * 2 + dy).min(h - 1);
                        s += src[((sy * w + sx) * 4 + c) as usize] as u32;
                    }
                }
                out[((y * hw + x) * 4 + c) as usize] = (s / 4) as u8;
            }
        }
    }
    out
}

/// Assemble `layers` of RGBA8 (each `size`×`size`) into a MIP-chained `texture_2d_array` image. Data layout is
/// `TextureDataOrder::LayerMajor`: each layer's full mip chain contiguous.
fn array_image(images: &mut Assets<Image>, layers: &[Vec<u8>], size: u32, srgb: bool) -> Handle<Image> {
    let mips = (32 - size.leading_zeros()).max(1); // floor(log2)+1 mip levels
    let mut data = Vec::new();
    for layer in layers {
        let (mut cur, mut w, mut h) = (layer.clone(), size, size);
        data.extend_from_slice(&cur);
        for _ in 1..mips {
            cur = downsample(&cur, w, h);
            w = (w / 2).max(1);
            h = (h / 2).max(1);
            data.extend_from_slice(&cur);
        }
    }
    let fmt = if srgb { TextureFormat::Rgba8UnormSrgb } else { TextureFormat::Rgba8Unorm };
    // `Image::new` debug-asserts data == ONE mip level's size; ours is the full mip chain, so build uninit
    // (no validation) and assign the mip-chained data + level count directly.
    let mut img = Image::new_uninit(
        Extent3d { width: size, height: size, depth_or_array_layers: layers.len() as u32 },
        TextureDimension::D2,
        fmt,
        RenderAssetUsages::RENDER_WORLD,
    );
    img.data = Some(data);
    img.texture_descriptor.mip_level_count = mips;
    img.data_order = bevy::render::render_resource::TextureDataOrder::LayerMajor;
    img.sampler = tiling_sampler();
    as_d2_array_view(&mut img);
    images.add(img)
}

/// Build the shared diffuse/normal/MRA texture arrays from the resolved material library (async: phase 1 kicks
/// off the source loads, phase 2 assembles once they're all in).
fn build_texture_arrays(
    library: Res<MaterialTextureLibrary>,
    assets: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    mut arr: ResMut<MeshTextureArrays>,
) {
    // Seed the fallback arrays once so the material is always bindable.
    if arr.diffuse == Handle::default() {
        arr.diffuse = fallback_array(&mut images, [255, 255, 255, 255], true);
        arr.normal = fallback_array(&mut images, [128, 128, 255, 255], false);
        arr.mra = fallback_array(&mut images, [0, 255, 255, 255], false);
    }

    let hash = variants_hash(&library);
    // Phase 1: a new/changed set of textured materials → start loading their sources.
    if hash != arr.hash && !library.variants.is_empty() {
        arr.hash = hash;
        arr.ready = false;
        arr.pending = library
            .variants
            .iter()
            .map(|m| LayerSrc {
                diffuse: m.diffuse.as_ref().map(|p| load_src(&assets, p, true)),
                normal: m.normal.as_ref().map(|p| load_src(&assets, p, false)),
                metallic: m.metallic.as_ref().map(|p| load_src(&assets, p, false)),
                roughness: m.roughness.as_ref().map(|p| load_src(&assets, p, false)),
                ao: m.ao.as_ref().map(|p| load_src(&assets, p, false)),
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
    // Common size = the largest source diffuse (power-of-two-ish; resize_exact handles the rest), capped.
    let size = arr
        .pending
        .iter()
        .filter_map(|l| l.diffuse.as_ref().and_then(|h| images.get(h)))
        .map(|i| i.width().max(i.height()))
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

// ─────────────────────────── shared material + table ───────────────────────────

fn registry_hash(reg: &MaterialRegistry, debug: bool, arrays_ready: bool) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write_u8(debug as u8);
    h.write_u8(arrays_ready as u8);
    h.write_usize(reg.defs.len());
    for d in &reg.defs {
        let l = d.base_color.to_linear();
        for v in
            [l.red, l.green, l.blue, d.metallic, d.roughness, d.texture_scale, d.blend_softness, d.emissive.x]
        {
            h.write_i64((v as f64 * 1.0e4) as i64);
        }
        h.write_u32(d.tex_layers[0]);
    }
    h.finish()
}

/// (Re)build the per-material table + the single shared `MeshMaterial` handle whenever the registry, the debug
/// flag, or the texture arrays change. The table is `MaterialDef` flattened to GPU rows; layer = the library
/// layer when the assembled arrays are live (else `NO_LAYER` → scalars, so untextured materials still shade).
pub(crate) fn rebuild_mesh_material(
    reg: Res<MaterialRegistry>,
    arrays: Res<MeshTextureArrays>,
    cfg: Res<MeshBakeConfig>,
    mut mats: ResMut<Assets<MeshMaterial>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut cache: ResMut<MeshMaterials>,
) {
    // Wait until `build_texture_arrays` has seeded the `D2Array` fallback arrays — never bind a default
    // (D2) image into the `2d_array` slots (a wgpu validation panic). Belt-and-braces with the `.after`.
    if arrays.diffuse == Handle::default() {
        return;
    }
    let hash = registry_hash(&reg, cfg.debug_lod_colour, arrays.ready);
    if cache.handle != Handle::default() && cache.table_hash == hash {
        return;
    }
    cache.table_hash = hash;

    let table: Vec<MeshMatGpu> = reg
        .defs
        .iter()
        .map(|d| {
            // Texture layer only once the real arrays are assembled (the fallback array has a single layer).
            let layer = if arrays.ready && d.tex_layers[0] != u32::MAX { d.tex_layers[0] } else { NO_LAYER };
            let textured = layer != NO_LAYER;
            let l = d.base_color.to_linear();
            MeshMatGpu {
                base_color: Vec4::new(l.red, l.green, l.blue, 1.0),
                emissive: d.emissive.extend(0.0),
                layer,
                has_diffuse: textured as u32,
                has_normal: textured as u32,
                has_mra: textured as u32,
                metallic: d.metallic,
                roughness: d.roughness.max(0.045),
                texture_scale: d.texture_scale,
                blend_softness: d.blend_softness,
            }
        })
        .collect();

    // The table is a storage buffer asset (Bevy 0.18 `#[storage]` binds a `Handle<ShaderStorageBuffer>`);
    // `ShaderStorageBuffer::from` encase-encodes the `Vec<MeshMatGpu>` as a runtime-sized `array<MeshMat>`.
    // encase can't encode a ZERO-length runtime array, and the registry is momentarily empty before the
    // first material resolve — so guarantee at least one (fallback) row. The shader clamps ids into range.
    let mut table = table;
    if table.is_empty() {
        table.push(MeshMatGpu::default());
    }
    let table_buf = buffers.add(ShaderStorageBuffer::from(table));

    let material = MeshMaterial {
        base: StandardMaterial {
            base_color: Color::WHITE,
            double_sided: true,
            cull_mode: None,
            ..default()
        },
        extension: MeshBlendExt {
            params: BlendParams { debug_lod: cfg.debug_lod_colour as u32, _pad: UVec3::ZERO },
            diffuse: arrays.diffuse.clone(),
            normal: arrays.normal.clone(),
            mra: arrays.mra.clone(),
            table: table_buf,
        },
    };
    // MUTATE the existing asset in place behind a STABLE handle. Every chunk mesh clones `cache.handle` at
    // commit; replacing it with a fresh `add()` would orphan all already-spawned meshes on the OLD asset, so
    // material-property edits (colour, metallic, blend_softness, …) would never show. `get_mut` keeps the
    // handle stable so the live edit reaches every chunk; only the first build allocates.
    if let Some(existing) = mats.get_mut(&cache.handle) {
        *existing = material;
    } else {
        cache.handle = mats.add(material);
    }
}
