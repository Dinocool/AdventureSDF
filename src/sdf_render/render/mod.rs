//! GPU SDF-volume bake + atlas plumbing.
//!
//! The on-screen SDF *surface* raymarch was removed in the mesh-bake pivot — the baked
//! meshes (`mesh_bake`) render the surfaces now. What remains here is the GPU brick-bake
//! compute path: the analytic CSG field is sampled into a sparse R16Snorm distance atlas
//! (+ material / gradient channels) on the GPU, kept resident in a paged atlas, and a chunk
//! directory lets a sampler look it up. Nothing here draws to the screen anymore.
//!
//! This is retained as a compilable, gated-off foundation for a FUTURE volumetric-cloud
//! raymarcher: the bake fills a 3D distance field the cloud pass would march. It is gated by
//! [`SdfRenderEnabled`] (default OFF), so the volume bake costs nothing during normal mesh
//! editing — `bake_scheduler::schedule_bakes` skips when the toggle is off.
//!
//! Kept render-world systems: per-view [`SdfCameraData`] extraction, the brick-bake compute
//! node, atlas-page upload, chunk-table upload, material upload, and PBR texture streaming.

use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin};
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel,
};
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only, storage_buffer_sized,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderStartup};
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};

// Re-exported via the submodules' `use super::*`: the atlas layout constants + types the bake /
// atlas-upload / chunk-table code reaches without re-importing.
use super::atlas::{ATLAS_TILES_PER_ROW, BRICK_EDGE, SdfAtlas};
use super::{SdfCamera, SdfGridConfig};

// Concern-specific submodules of the bake path (bake compute, atlas paging, chunk tables, PBR
// texture streaming); each reaches the shared render types here via `use super::*`.
mod atlas_pages;
mod atlas_upload;
mod bake;
mod chunk_tables;
mod gpu;
mod pbr_textures;

use atlas_pages::AtlasPages;
use chunk_tables::ChunkTableBuffers;

// --- GPU Types ---

/// Per-view camera data, extracted into the render world and uploaded as a dynamic-offset
/// uniform. Kept as the foundation for a future volumetric-cloud raymarch pass (which would
/// need the inverse-view-projection + camera position to cast rays into the baked SDF volume).
/// No surface pass consumes it today.
#[derive(Component, Clone, Copy, ShaderType, Default, ExtractComponent, Reflect)]
#[reflect(Component)]
struct SdfCameraData {
    inv_view_proj: Mat4,
    /// Forward view-projection (clip from world).
    clip_from_world: Mat4,
    camera_pos: Vec4,
    /// xy = screen_size; z = overlap_depth (u32); w = unused.
    screen_params: Vec4,
    /// xyz = grid origin, w = voxel_size.
    grid_origin: Vec4,
    /// z = brick_size (samples per edge); x/y/w unused.
    grid_dims: Vec4,
    /// x = lod_count, y = ring_bricks, z = base voxel_size, w = cell_stride.
    lod_params: Vec4,
}

/// GPU mirror of a [`super::edits::MaterialDef`], one per global material id, in a storage buffer
/// indexed by id. Carries the PBR texture-array layer for each map (`u32::MAX` = none). 80 bytes,
/// 16-byte aligned for std430. The three `_pad*` words align `emissive` (a `vec4`) to offset 64.
/// Built each frame and uploaded; retained as part of the bake-foundation (no surface pass reads
/// it today).
#[repr(C)]
#[derive(ShaderType, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSdfMaterial {
    base_color: Vec4,
    blend_softness: f32,
    tex_diffuse: u32,
    tex_normal: u32,
    tex_mra: u32,
    tex_height: u32,
    tex_edge: u32,
    /// Scalar metallic/roughness fallbacks.
    metallic: f32,
    roughness: f32,
    /// Parallax-occlusion relief depth (UV units) for this material's height map. 0 = flat.
    parallax_scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    /// Emissive radiance, linear RGB in `xyz`; `w` spare. `vec4` so it's 16-byte aligned at offset 64.
    emissive: Vec4,
}

// --- GPU Atlas ---

#[derive(Resource, Default)]
struct SdfGpuAtlas {
    /// Paged distance + material atlases (R16Snorm + Rgba16Snorm), grown one fixed-size page at a
    /// time with NO copy (see [`AtlasPages`]). `None` until `init_sdf_atlas` creates the pool. The
    /// bake writes tiles straight into the live pages. This is the 3D distance field a future cloud
    /// raymarch would sample.
    pages: Option<AtlasPages>,
    /// Nearest sampler for the atlas pages. Read only by the removed surface bind group; retained
    /// for the future cloud-raymarch atlas bind group.
    #[allow(dead_code)]
    sampler: Option<Sampler>,
    /// Chunk lookup directory + packed per-chunk tile runs, as the shared growable-storage-buffer
    /// pool (`ChunkTableBuffers`). `None` buffers until `init_sdf_atlas`.
    tables: ChunkTableBuffers,
    /// Material table (storage buffer of `GpuSdfMaterial`, indexed by material id).
    material_buffer: Option<Buffer>,
    /// PBR texture-array views (one per `MapArray`: diffuse, normal, mra, height, edge), filled by
    /// the texture streamer. Retained for the future cloud/PBR sampling path.
    tex_array_views: Option<[TextureView; super::edits::MATERIAL_TEX_MAPS]>,
    /// Filtering+mip sampler for the PBR arrays (distinct from the nearest atlas one).
    tex_sampler: Option<Sampler>,
}

/// Material table extracted from the main world for GPU upload.
#[derive(Resource, Default)]
struct ExtractedSdfMaterials {
    materials: Vec<GpuSdfMaterial>,
}

// --- Render Graph (bake only) ---

pub struct SdfRenderPlugin;

impl Plugin for SdfRenderPlugin {
    fn build(&self, app: &mut App) {
        let asset_server = app.world().resource::<AssetServer>();
        let bake_shader_handle: Handle<Shader> = asset_server.load(bake::SDF_BAKE_SHADER_PATH);

        app.init_resource::<SdfMaterialTable>()
            .register_type::<SdfCameraData>()
            // These plugins internally find the render app via get_sub_app_mut(RenderApp).
            .add_plugins((
                ExtractComponentPlugin::<SdfCameraData>::default(),
                UniformComponentPlugin::<SdfCameraData>::default(),
                bevy::render::extract_resource::ExtractResourcePlugin::<super::SdfRenderEnabled>::default(),
                // ProbeReset drives the scene-switch atlas-page reset (atlas_upload).
                bevy::render::extract_resource::ExtractResourcePlugin::<super::ProbeReset>::default(),
            ))
            .add_systems(
                Update,
                prepare_sdf_camera_data
                    .run_if(in_state(crate::scene_manager::AppScene::SdfEditor))
                    .after(super::editor_camera::orbit_camera)
                    // Run after bake scheduling so the camera uniform reflects this frame's
                    // post-bake state.
                    .after(super::bake_scheduler::schedule_bakes),
            );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.insert_resource(bake::SdfBakeShaderHandle(bake_shader_handle));

        render_app
            .add_systems(ExtractSchedule, atlas_upload::extract_sdf_atlas)
            .add_systems(ExtractSchedule, extract_sdf_materials)
            .add_systems(ExtractSchedule, pbr_textures::extract_texture_library)
            .add_systems(ExtractSchedule, bake::extract_brick_bakes)
            .init_resource::<pbr_textures::TextureStreamState>()
            .init_resource::<atlas_upload::LastAtlasGen>()
            .init_resource::<chunk_tables::ChunkBufCapacity>()
            .init_resource::<bake::SdfBakeBuffers>()
            .add_systems(
                Render,
                bake::prepare_brick_bake_buffers.before(atlas_upload::prepare_sdf_atlas_gpu),
            )
            .add_systems(Render, atlas_upload::prepare_sdf_atlas_gpu)
            .add_systems(Render, prepare_sdf_materials_gpu)
            .add_systems(Render, pbr_textures::init_texture_streaming)
            .add_systems(
                Render,
                pbr_textures::upload_texture_layers.after(pbr_textures::init_texture_streaming),
            )
            .add_systems(RenderStartup, init_sdf_atlas)
            .add_systems(RenderStartup, bake::init_bake_pipeline.after(init_sdf_atlas))
            .add_render_graph_node::<bake::SdfBrickBakeNode>(Core3d, bake::SdfBrickBakeLabel)
            // The brick-bake compute pass fills the shared atlas. It's a standalone (non-view)
            // node, scheduled right after the opaque pass; nothing downstream reads the atlas
            // on-screen yet (the future cloud pass would), so it has a single edge into the graph.
            .add_render_graph_edges(
                Core3d,
                (Node3d::MainOpaquePass, bake::SdfBrickBakeLabel),
            );
    }
}

// --- Main World: Prepare Camera Data ---

/// Main-world material table, rebuilt each frame from the material registry in resolved id order.
/// Extracted into the render world and uploaded as a storage buffer. Retained as part of the bake
/// foundation (a future cloud/PBR pass would read it); no surface pass consumes it today.
#[derive(Resource, Default)]
pub struct SdfMaterialTable {
    materials: Vec<GpuSdfMaterial>,
}

/// Physical-luminance scale applied to authored (display-referred) material emissive at GPU upload.
/// Retained with the material table for the future cloud/PBR pass.
const EMISSIVE_NITS_SCALE: f32 = 4000.0;

fn prepare_sdf_camera_data(
    mut commands: Commands,
    cameras: Query<(Entity, &Camera, &Transform), With<SdfCamera>>,
    config: Res<SdfGridConfig>,
    registry: Res<super::edits::MaterialRegistry>,
    mut material_table: ResMut<SdfMaterialTable>,
) {
    let _span = info_span!("prepare_sdf_camera_data").entered();

    // The GPU material table mirrors the global registry verbatim: row i = the material with global
    // id i. Rebuilt only when the registry changes (the single source of truth).
    if registry.is_changed() || material_table.materials.is_empty() {
        material_table.materials.clear();
        for def in &registry.defs {
            let lin = def.base_color.to_linear();
            let t = def.tex_layers;
            material_table.materials.push(GpuSdfMaterial {
                base_color: Vec4::new(lin.red, lin.green, lin.blue, 1.0),
                blend_softness: def.blend_softness,
                tex_diffuse: t[0],
                tex_normal: t[1],
                tex_mra: t[2],
                tex_height: t[3],
                tex_edge: t[4],
                metallic: def.metallic,
                roughness: def.roughness,
                parallax_scale: def.parallax_scale,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                emissive: (def.emissive * EMISSIVE_NITS_SCALE).extend(0.0),
            });
        }
    }

    for (entity, camera, transform) in &cameras {
        let view_from_world = transform.to_matrix().inverse();
        let clip_from_world = camera.clip_from_view() * view_from_world;
        let inv_view_proj = clip_from_world.inverse();

        let size = camera
            .physical_viewport_size()
            .unwrap_or(UVec2::new(1920, 1080));

        commands.entity(entity).insert(SdfCameraData {
            inv_view_proj,
            clip_from_world,
            camera_pos: transform.translation.extend(0.0),
            screen_params: Vec4::new(size.x as f32, size.y as f32, config.overlap_depth as f32, 0.0),
            grid_origin: Vec4::new(
                config.world_origin().x,
                config.world_origin().y,
                config.world_origin().z,
                config.voxel_size,
            ),
            grid_dims: Vec4::new(0.0, 0.0, config.brick_size as f32, 0.0),
            lod_params: Vec4::new(
                config.lod_count as f32,
                config.ring_bricks as f32,
                config.voxel_size,
                config.cell_stride() as f32,
            ),
        });
    }
}

// --- Extract: Material table ---

fn extract_sdf_materials(table: Extract<Res<SdfMaterialTable>>, mut commands: Commands) {
    // Always carry at least one row so the storage buffer is never zero-sized.
    let mut materials = table.materials.clone();
    if materials.is_empty() {
        materials.push(GpuSdfMaterial::default());
    }
    commands.insert_resource(ExtractedSdfMaterials { materials });
}

fn prepare_sdf_materials_gpu(
    device: Res<RenderDevice>,
    extracted: Option<Res<ExtractedSdfMaterials>>,
    mut gpu_atlas: ResMut<SdfGpuAtlas>,
) {
    let Some(extracted) = extracted else { return };
    gpu_atlas.material_buffer = Some(gpu::storage_buffer_init(
        &device,
        "sdf_material_buffer",
        &extracted.materials,
    ));
}

// --- Render World: Atlas Init ---

/// Seed the GPU atlas pool (paged distance/material/gradient textures + chunk-table buffers +
/// dummy material/PBR resources) so the bake path has valid resources before the first bake.
/// Runs at `RenderStartup`.
fn init_sdf_atlas(mut commands: Commands, device: Res<RenderDevice>, queue: Res<RenderQueue>) {
    // Nearest sampler for the atlas pages (one texel per sample, no interpolation of packed data).
    let dummy_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("sdf_atlas_sampler"),
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        mipmap_filter: FilterMode::Nearest,
        ..default()
    });
    // One zeroed 80-byte GpuSdfMaterial row so the material buffer meets the struct's minimum size
    // before the real table uploads.
    let dummy_material = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_dummy_material"),
        contents: &[0u8; 80],
        usage: BufferUsages::STORAGE,
    });
    // Dummy 1×1×1 PBR arrays + a filtering sampler so the atlas's tex slots are valid before the
    // texture library uploads. Each is a D2Array view of one zeroed layer.
    let dummy_tex_views: [TextureView; super::edits::MATERIAL_TEX_MAPS] =
        std::array::from_fn(|i| {
            let tex = device.create_texture_with_data(
                &queue,
                &TextureDescriptor {
                    label: Some("sdf_dummy_tex_array"),
                    size: Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: if i == 0 {
                        TextureFormat::Rgba8UnormSrgb
                    } else {
                        TextureFormat::Rgba8Unorm
                    },
                    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                TextureDataOrder::LayerMajor,
                &[0u8; 4],
            );
            tex.create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::D2Array),
                ..default()
            })
        });
    let dummy_tex_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("sdf_dummy_tex_sampler"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });

    commands.insert_resource(SdfGpuAtlas {
        // Empty page pool (its own 1×1 dummies fill the binding array until the first bake grows it).
        pages: Some(AtlasPages::new(&device)),
        sampler: Some(dummy_sampler),
        // Directory + tile-run buffers with 1-record dummies until the first bake fills them.
        tables: ChunkTableBuffers::new(&device),
        material_buffer: Some(dummy_material),
        tex_array_views: Some(dummy_tex_views),
        tex_sampler: Some(dummy_tex_sampler),
    });
}
