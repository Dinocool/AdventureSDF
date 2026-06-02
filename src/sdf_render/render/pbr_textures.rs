//! PBR texture-array streaming: extract the demand-driven material texture library, allocate the
//! fixed-cap BC7 array textures once, and stream each variant's encoded layers in on background
//! tasks (so first-run BC7 encoding never blocks the render thread). Writes the array views +
//! sampler onto the shared `SdfGpuAtlas` (from [`super`]).

use super::super::{bc7, edits, textures};
use super::*;

/// One variant's encoded BC7 maps + its destination array layer, produced by a
/// background task and consumed by the upload poll system.
struct EncodedVariant {
    layer: u32,
    maps: textures::VariantBc7,
}

/// Render-world streaming state for the PBR texture arrays: the fallback-filled,
/// full-size destination textures and the in-flight per-variant encode tasks. Layers
/// are `write_texture`d in as their tasks finish, so first-run BC7 encoding never
/// blocks the render thread — materials show the magenta fallback until their layer
/// arrives.
#[derive(Resource, Default)]
pub(super) struct TextureStreamState {
    /// Destination BC7 array textures (kept alive so layer uploads stay valid).
    textures: Vec<Texture>,
    /// Background encode tasks, drained as they complete.
    tasks: Vec<Task<EncodedVariant>>,
    /// Whether the (fixed-cap) arrays were allocated (one-shot allocation guard).
    allocated: bool,
    /// How many variants have had an encode task spawned. Grows as the demand-driven
    /// library appends variants; we spawn tasks for `[spawned_layers, variants.len())`.
    spawned_layers: u32,
}

/// The texture library extracted from the main world. `variants` grows on demand as
/// materials reference new textures; index = GPU array layer. The render world
/// streams any layers that appear beyond what it has already uploaded.
#[derive(Resource, Default)]
pub(super) struct ExtractedTextureLibrary {
    variants: Vec<crate::assets::MapSet>,
}

pub(super) fn extract_texture_library(
    library: Extract<Res<crate::assets::MaterialTextureLibrary>>,
    mut commands: Commands,
) {
    commands.insert_resource(ExtractedTextureLibrary {
        variants: library.variants.clone(),
    });
}

/// BC7 array formats per `MapArray`: sRGB for diffuse (0), linear for the rest.
const PBR_ARRAY_FORMATS: [TextureFormat; edits::MATERIAL_TEX_MAPS] = [
    TextureFormat::Bc7RgbaUnormSrgb,
    TextureFormat::Bc7RgbaUnorm,
    TextureFormat::Bc7RgbaUnorm,
    TextureFormat::Bc7RgbaUnorm,
    TextureFormat::Bc7RgbaUnorm,
];

/// One-shot: once the extracted library is available, create the 5 EMPTY BC7 arrays
/// at full size, point the bind-group views at them, and spawn one background encode
/// task per variant. No GPU upload here — layers stream in via `upload_texture_layers`
/// as tasks finish, so the first-run BC7 encode never blocks the render thread.
pub(super) fn init_texture_streaming(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    extracted: Option<Res<ExtractedTextureLibrary>>,
    mut gpu_atlas: ResMut<SdfGpuAtlas>,
    mut stream: ResMut<TextureStreamState>,
) {
    use crate::assets::MAX_TEXTURE_LAYERS;
    use textures::TEXTURE_SIZE;

    // 1) Allocate the fixed-cap arrays once (the moment the render device is up). The
    // arrays are sized to MAX_TEXTURE_LAYERS so the demand-driven library can append
    // variants without ever recreating the textures or rebuilding the bind group.
    if !stream.allocated {
        let mips = bc7::mip_count(TEXTURE_SIZE);
        let labels = [
            "sdf_tex_diffuse",
            "sdf_tex_normal",
            "sdf_tex_mra",
            "sdf_tex_height",
            "sdf_tex_edge",
        ];
        // Per-map fallback fill shown until a layer streams in: magenta diffuse (an
        // obvious "loading" colour), NEUTRAL data maps so lit surfaces still look sane
        // (flat normal, mid-rough/unoccluded MRA, zero height, no edge wear).
        let fallback: [[u8; 4]; edits::MATERIAL_TEX_MAPS] = [
            [255, 0, 255, 255],
            [128, 128, 255, 255],
            [0, 255, 255, 255],
            [0, 0, 0, 255],
            [0, 0, 0, 255],
        ];

        let mut textures = Vec::with_capacity(edits::MATERIAL_TEX_MAPS);
        let views: [TextureView; edits::MATERIAL_TEX_MAPS] = std::array::from_fn(|i| {
            let fill = bc7::solid_fill_bc7(fallback[i], TEXTURE_SIZE, MAX_TEXTURE_LAYERS);
            let tex = device.create_texture_with_data(
                &queue,
                &TextureDescriptor {
                    label: Some(labels[i]),
                    size: Extent3d {
                        width: TEXTURE_SIZE,
                        height: TEXTURE_SIZE,
                        depth_or_array_layers: MAX_TEXTURE_LAYERS,
                    },
                    mip_level_count: mips,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: PBR_ARRAY_FORMATS[i],
                    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                TextureDataOrder::LayerMajor,
                &fill.data,
            );
            let view = tex.create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::D2Array),
                ..default()
            });
            textures.push(tex);
            view
        });

        gpu_atlas.tex_sampler = Some(device.create_sampler(&SamplerDescriptor {
            label: Some("sdf_tex_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            ..default()
        }));
        gpu_atlas.tex_array_views = Some(views);
        stream.textures = textures;
        stream.allocated = true;
    }

    // 2) Spawn encode tasks for any variants the library appended since last frame
    // (demand-driven: a variant appears when a used material first references it).
    let Some(extracted) = extracted else { return };
    let want = (extracted.variants.len() as u32).min(MAX_TEXTURE_LAYERS);
    if want <= stream.spawned_layers {
        return;
    }
    let pool = AsyncComputeTaskPool::get();
    for layer in stream.spawned_layers..want {
        let map_set = extracted.variants[layer as usize].clone();
        stream.tasks.push(pool.spawn(async move {
            let maps = textures::encode_mapset_bc7(&map_set);
            EncodedVariant { layer, maps }
        }));
    }
    info!(
        "SDF textures: streaming layers {}..{}",
        stream.spawned_layers, want
    );
    stream.spawned_layers = want;
}

/// Each frame, drain any finished encode tasks and `write_texture` their BC7 mip
/// chains into the destination array layer (per map, per mip). Non-blocking poll —
/// unfinished tasks are left for next frame.
pub(super) fn upload_texture_layers(queue: Res<RenderQueue>, mut stream: ResMut<TextureStreamState>) {
    if stream.tasks.is_empty() {
        return;
    }
    use textures::TEXTURE_SIZE;

    let mut i = 0;
    while i < stream.tasks.len() {
        let Some(done) = block_on(poll_once(&mut stream.tasks[i])) else {
            i += 1;
            continue;
        };
        // Upload every map's single-layer mip chain into `done.layer`. Clamp to the
        // texture's actual mip count — a stale cache blob claiming more levels than
        // the texture has would otherwise over-run it (wgpu fatal). The cache key's
        // ENCODER_VERSION normally prevents this; the clamp is belt-and-suspenders.
        let tex_mips = bc7::mip_count(TEXTURE_SIZE);
        for (map, arr) in done.maps.iter().enumerate() {
            let texture = &stream.textures[map];
            let mut offset = 0usize;
            let mut size = TEXTURE_SIZE;
            for mip in 0..arr.mip_levels.min(tex_mips) {
                let blocks_w = size.div_ceil(4);
                let blocks_h = size.div_ceil(4);
                let bytes_per_row = blocks_w * 16; // BC7 = 16 bytes/block
                let level_len = (bytes_per_row * blocks_h) as usize;
                queue.write_texture(
                    TexelCopyTextureInfo {
                        texture,
                        mip_level: mip,
                        origin: Origin3d {
                            x: 0,
                            y: 0,
                            z: done.layer,
                        },
                        aspect: TextureAspect::All,
                    },
                    &arr.data[offset..offset + level_len],
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(blocks_h),
                    },
                    Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: 1,
                    },
                );
                offset += level_len;
                size = (size / 2).max(4); // BC7 mip chain stops at the 4×4 block min
            }
        }
        let done_layer = done.layer;
        // Task already produced its result via poll_once; drop the finished handle.
        drop(stream.tasks.swap_remove(i));
        let remaining = stream.tasks.len();
        debug!("SDF textures: layer {done_layer} uploaded ({remaining} remaining)");
        // don't advance `i` — swap_remove moved a new task into this slot.
    }
}
