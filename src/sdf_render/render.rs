use bevy::core_pipeline::FullscreenShader;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::extract_component::{
    ComponentUniforms, ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin,
};
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only, texture_2d, texture_2d_array, uniform_buffer,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::view::{ViewDepthTexture, ViewTarget};
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderStartup};
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};

use super::atlas::{BRICK_EDGE, SdfAtlas};
use super::edits::PALETTE_K;
use super::{SdfCamera, SdfGridConfig, SdfRenderEnabled};

// --- GPU Types ---

/// One entry in the chunk lookup buffer (20 bytes, std430), sorted by `(key_hi,key_lo)`
/// and binary-searched by the shader. `key_*` = the absolute chunk key (see
/// `super::chunk`), independent of camera so CPU and GPU agree. `occ_*` = 64-bit
/// occupancy mask (bit i ⇒ local brick i resident); `tile_run_base` indexes the packed
/// `chunk_tile_table` where this chunk's `popcount(occ)` brick `atlas_base`s live.
#[derive(ShaderType, Clone, Copy, Default)]
struct GpuChunkLookup {
    key_hi: u32,
    key_lo: u32,
    occ_lo: u32,
    occ_hi: u32,
    tile_run_base: u32,
}

/// One resident brick's record in the packed chunk tile-run buffer (12 bytes): atlas
/// tile origin (`col_px | row_px<<16`) + packed 4-entry material palette.
#[derive(ShaderType, Clone, Copy, Default)]
struct GpuBrickTile {
    atlas_base: u32,
    pal01: u32,
    pal23: u32,
}

/// Atlas tiles per row. Width = this × 64 px. 256 → 16384 px wide, half the 32768
/// wgpu limit, so it never overflows while keeping the texture reasonably square.
const ATLAS_TILES_PER_ROW: u32 = 256;

#[derive(Component, Clone, Copy, ShaderType, Default, ExtractComponent, Reflect)]
#[reflect(Component)]
struct SdfCameraData {
    inv_view_proj: Mat4,
    /// Forward view-projection. Used to write true reverse-Z projection depth from
    /// the raymarch hit, so the SDF surface occludes/are-occluded-by other passes
    /// (wireframe, gizmos) through the normal depth buffer.
    clip_from_world: Mat4,
    camera_pos: Vec4,
    screen_params: Vec4, // xy = screen_size, zw = unused
    grid_origin: Vec4,   // xyz = grid origin, w = voxel_size
    grid_dims: Vec4, // x = grid_size, y = bricks_per_axis, z = brick_size (8.0), w = num_lookups
    debug_params: Vec4, // x = max_steps, y = max_dist, z = sdf_eps, w = unused
    /// x = pixel_cone (world radius per unit ray distance per pixel), y = cubic_band,
    /// z = over_relax, w unused.
    march_params: Vec4,
    /// x = lod_count, y = ring_bricks, z = base voxel_size, w = cell_stride.
    lod_params: Vec4,
}

/// GPU mirror of a [`super::edits::MaterialDef`], one per global material id, in a
/// storage buffer indexed by id. Carries the PBR texture-array layer for each map
/// (`u32::MAX` = none); the shader samples those layers via triplanar projection.
/// 48 bytes, 16-byte aligned for std430.
#[derive(ShaderType, Clone, Copy, Default)]
struct GpuSdfMaterial {
    base_color: Vec4,
    blend_softness: f32,
    tex_diffuse: u32,
    tex_normal: u32,
    tex_mra: u32,
    tex_height: u32,
    tex_edge: u32,
    _pad0: u32,
    _pad1: u32,
}

// --- Extracted Atlas ---

/// One brick's texels for a partial upload: a 64×8 sub-rect at the tile's pixel
/// origin. `dist` is BRICK_VOXELS i16; `mat` is BRICK_VOXELS×4 i16 (the 4 palette
/// slots). Laid out tile-local (the same y*EDGE+x / z mapping the full path uses).
struct TileTexels {
    /// Pixel origin of the tile in the atlas (`col_px | row_px<<16`, as packed into
    /// `atlas_base`). Split in `prepare` for the `write_texture` origin.
    atlas_base: u32,
    dist: Vec<i16>,
    mat: Vec<i16>,
}

#[derive(Resource, Default)]
struct ExtractedSdfAtlas {
    /// Full atlas pixel buffers, present only on a realloc (full rebuild / grow).
    /// R16Snorm distance + Rgba16Snorm 4-slot material, whole-texture sized.
    dist_data: Vec<i16>,
    mat_data: Vec<i16>,
    /// Per-tile deltas for an in-place partial upload (only the bricks that changed
    /// this bake). Empty on a realloc.
    changed_tiles: Vec<TileTexels>,
    /// Sorted chunk lookup table + packed per-chunk tile runs (see `super::chunk`).
    /// Populated only when `tables_dirty` (the atlas topology changed); empty otherwise,
    /// so the GPU prepare keeps the existing buffers instead of rebuilding them.
    chunk_data: Vec<GpuChunkLookup>,
    tile_run_data: Vec<GpuBrickTile>,
    /// Whether `chunk_data`/`tile_run_data` carry a fresh table this frame. False on a
    /// texel-only re-bake — the lookup buffers are reused as-is.
    tables_dirty: bool,
    texture_width: u32,
    texture_height: u32,
    /// Recreate the textures + views (full rebuild or capacity grow). When false,
    /// `prepare` keeps the existing textures and only `write_texture`s `changed_tiles`.
    realloc: bool,
    dirty: bool,
}

/// Render-world memo of the last atlas generation uploaded, so `extract_sdf_atlas`
/// only flags `dirty` (and `prepare_sdf_atlas_gpu` only re-uploads) when the
/// main-world bake actually changed something. Without this the atlas was rebuilt
/// every frame.
#[derive(Resource, Default)]
struct LastAtlasGen(u64);

/// Render-world memo of the last atlas *topology* generation whose chunk lookup / tile-run
/// tables were uploaded. `extract_sdf_atlas` rebuilds those (O(bricks)) tables and flags
/// them for re-upload only when this differs, so a frame that only re-baked texels in place
/// (camera idle, an edit nudged) skips the table rebuild entirely. A realloc always rebuilds
/// the tables (the texture and its tile layout were recreated).
#[derive(Resource, Default)]
struct LastAtlasTopology(u64);

/// Render-world record of how many tile rows the persistent atlas texture currently
/// spans. `extract_sdf_atlas` reads it to decide grow-vs-partial-upload; the texture
/// only grows (never shrinks except on a full rebuild), so a tile origin assigned
/// once stays valid until the next full bake.
#[derive(Resource, Default)]
struct AtlasCapacity {
    rows: u32,
}

// --- GPU Atlas ---

#[derive(Resource, Default)]
struct SdfGpuAtlas {
    /// Persistent distance atlas (R16Snorm). Kept (not just its view) so partial
    /// bakes can `write_texture` only the changed tiles instead of recreating it.
    dist_tex: Option<Texture>,
    dist_view: Option<TextureView>,
    /// Persistent per-palette-slot distance atlas (Rgba16Snorm, 4 channels). The
    /// shader argmins the 4 slots for the local material index, then maps it via the
    /// brick palette. Kept across frames for the same partial-upload reason.
    mat_tex: Option<Texture>,
    mat_view: Option<TextureView>,
    sampler: Option<Sampler>,
    /// Chunk lookup table (binding 2) + packed per-chunk tile runs (binding 11).
    lookup_buffer: Option<Buffer>,
    chunk_tile_buffer: Option<Buffer>,
    /// Material table (storage buffer of `GpuSdfMaterial`, indexed by material id).
    material_buffer: Option<Buffer>,
    /// PBR texture-array views (one per `MapArray`: diffuse, normal, mra, height,
    /// edge), each `texture_2d_array` indexed by a material's tex layer. A 1×1×1
    /// dummy keeps the bind group valid until the real arrays are created; the real
    /// arrays are pre-filled with a fallback (magenta diffuse + neutral data maps)
    /// and each layer is overwritten via `write_texture` when its encode finishes,
    /// so a not-yet-streamed material shows magenta rather than black.
    tex_array_views: Option<[TextureView; super::edits::MATERIAL_TEX_MAPS]>,
    /// Filtering+mip sampler for the PBR arrays (distinct from the nearest atlas one).
    tex_sampler: Option<Sampler>,
}

/// One variant's encoded BC7 maps + its destination array layer, produced by a
/// background task and consumed by the upload poll system.
struct EncodedVariant {
    layer: u32,
    maps: super::textures::VariantBc7,
}

/// Render-world streaming state for the PBR texture arrays: the fallback-filled,
/// full-size destination textures and the in-flight per-variant encode tasks. Layers
/// are `write_texture`d in as their tasks finish, so first-run BC7 encoding never
/// blocks the render thread — materials show the magenta fallback until their layer
/// arrives.
#[derive(Resource, Default)]
struct TextureStreamState {
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

/// Material table extracted from the main world for GPU upload.
#[derive(Resource, Default)]
struct ExtractedSdfMaterials {
    materials: Vec<GpuSdfMaterial>,
}

/// The texture library extracted from the main world. `variants` grows on demand as
/// materials reference new textures; index = GPU array layer. The render world
/// streams any layers that appear beyond what it has already uploaded.
#[derive(Resource, Default)]
struct ExtractedTextureLibrary {
    variants: Vec<super::textures::LibraryVariant>,
}

// --- Pipeline ---

// BISECT: minimal shader while building features back up after the division-free fix.
const SDF_SHADER_PATH: &str = "shaders/sdf_raymarch.wgsl";

#[derive(Resource)]
struct SdfPipeline {
    pipeline_id: CachedRenderPipelineId,
    layout_0: BindGroupLayoutDescriptor,
    layout_1: BindGroupLayoutDescriptor,
    #[expect(dead_code)]
    shader_handle: Handle<Shader>,
}

#[derive(Resource, Default)]
pub struct SdfShaderDefs {
    pub defs: Vec<String>,
}

// --- Render Graph ---

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct SdfLabel;

fn create_dummy_bg0(device: &RenderDevice, layout: &BindGroupLayout) -> BindGroup {
    let buf = device.create_buffer(&BufferDescriptor {
        label: Some("sdf_dummy_uniform"),
        size: 512,
        usage: BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    device.create_bind_group(
        "sdf_bind_group_0_empty",
        layout,
        &BindGroupEntries::sequential((buf.as_entire_buffer_binding(),)),
    )
}

#[derive(Default)]
struct SdfNode;

impl ViewNode for SdfNode {
    type ViewQuery = (&'static ViewTarget, &'static ViewDepthTexture);

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, depth): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // Only run on SDF cameras — skip all other views
        let view_entity = graph.view_entity();
        if world.get::<SdfCameraData>(view_entity).is_none() {
            return Ok(());
        }

        // Skip SDF pass when toggled off (F1)
        if let Some(enabled) = world.get_resource::<SdfRenderEnabled>()
            && !enabled.0
        {
            return Ok(());
        }

        let pipeline_res = world.resource::<SdfPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let device = render_context.render_device();

        let pipeline = pipeline_cache.get_render_pipeline(pipeline_res.pipeline_id);

        if pipeline.is_none() {
            use std::sync::atomic::{AtomicBool, Ordering};
            static LOGGED: AtomicBool = AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::Relaxed)
                && let bevy::render::render_resource::CachedPipelineState::Err(err) =
                    pipeline_cache.get_render_pipeline_state(pipeline_res.pipeline_id)
            {
                bevy::log::error!("SDF pipeline error: {err}");
            }
        }
        let layout_0 = pipeline_cache.get_bind_group_layout(&pipeline_res.layout_0);
        let layout_1 = pipeline_cache.get_bind_group_layout(&pipeline_res.layout_1);

        // Bind group 0: camera uniform or fallback
        let has_camera = world
            .get_resource::<ComponentUniforms<SdfCameraData>>()
            .and_then(|u| u.binding())
            .is_some();
        if !has_camera {
            use std::sync::atomic::{AtomicBool, Ordering};
            static LOGGED: AtomicBool = AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::Relaxed) {
                warn!("SDF: no camera uniform — using dummy data");
            }
        }
        let bind_group_0 = if let Some(camera_uniforms) =
            world.get_resource::<ComponentUniforms<SdfCameraData>>()
        {
            if let Some(binding) = camera_uniforms.binding() {
                device.create_bind_group(
                    "sdf_bind_group_0",
                    &layout_0,
                    &BindGroupEntries::sequential((binding.clone(),)),
                )
            } else {
                create_dummy_bg0(device, &layout_0)
            }
        } else {
            create_dummy_bg0(device, &layout_0)
        };

        // Bind group 1: atlas (always available — dummy in init)
        let gpu_atlas = world.resource::<SdfGpuAtlas>();
        let tex_views = gpu_atlas.tex_array_views.as_ref().unwrap();
        let bind_group_1 = device.create_bind_group(
            "sdf_bind_group_1",
            &layout_1,
            &BindGroupEntries::sequential((
                gpu_atlas.dist_view.as_ref().unwrap(),
                gpu_atlas.sampler.as_ref().unwrap(),
                gpu_atlas
                    .lookup_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_buffer_binding(),
                gpu_atlas.mat_view.as_ref().unwrap(),
                gpu_atlas
                    .material_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_buffer_binding(),
                gpu_atlas.tex_sampler.as_ref().unwrap(),
                &tex_views[0],
                &tex_views[1],
                &tex_views[2],
                &tex_views[3],
                &tex_views[4],
                gpu_atlas
                    .chunk_tile_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_buffer_binding(),
            )),
        );

        let post_process = view_target.post_process_write();

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("sdf_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination,
                resolve_target: None,
                depth_slice: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        if let Some(pipeline) = pipeline {
            render_pass.set_render_pipeline(pipeline);
            render_pass.set_bind_group(0, &bind_group_0, &[0]);
            render_pass.set_bind_group(1, &bind_group_1, &[]);
            render_pass.draw(0..3, 0..1);
        }

        Ok(())
    }
}

// --- Plugin ---

#[derive(Resource)]
struct SdfShaderHandle(Handle<Shader>);

/// The `sdf::*` shader modules imported by the entry shader. They use
/// `#define_import_path` (Custom names), which Bevy's asset pipeline does NOT
/// auto-load — so we load them here and keep the handles alive for the app's
/// lifetime. Dropping a handle would unload the module and break composition.
#[derive(Resource)]
struct SdfShaderModules(#[expect(dead_code)] Vec<Handle<Shader>>);

/// The `#define_import_path` module files the entry shader composes.
const SDF_SHADER_MODULES: [&str; 5] = [
    "shaders/sdf/bindings.wgsl",
    "shaders/sdf/brick.wgsl",
    "shaders/sdf/cubic.wgsl",
    "shaders/sdf/material.wgsl",
    "shaders/sdf/pbr.wgsl",
];

pub struct SdfRenderPlugin;

impl Plugin for SdfRenderPlugin {
    fn build(&self, app: &mut App) {
        // Load shader asset in main world so it's available for extraction
        let asset_server = app.world().resource::<AssetServer>();
        let shader_handle = asset_server.load(SDF_SHADER_PATH);
        // Load + retain the imported modules (Custom-path imports aren't auto-loaded).
        let module_handles: Vec<Handle<Shader>> = SDF_SHADER_MODULES
            .iter()
            .map(|p| asset_server.load(*p))
            .collect();
        app.insert_resource(SdfShaderModules(module_handles))
            .insert_resource(SdfShaderHandle(shader_handle))
            .init_resource::<SdfShaderDefs>()
            .init_resource::<SdfMaterialTable>()
            .register_type::<SdfCameraData>()
            // These plugins must be added to the main app — they internally
            // find the render app via get_sub_app_mut(RenderApp)
            .add_plugins((
                ExtractComponentPlugin::<SdfCameraData>::default(),
                UniformComponentPlugin::<SdfCameraData>::default(),
            ))
            .add_systems(
                Update,
                prepare_sdf_camera_data
                    .run_if(in_state(crate::scene_manager::AppScene::SdfEditor))
                    .after(super::orbit_camera)
                    // MUST run after the bake apply: the shader's binary-search bound
                    // (`grid_dims.w = atlas.bricks.len()`) has to match the lookup
                    // buffer `extract_sdf_atlas` builds from the *same* post-bake
                    // brick set. Reading the count before the bake desyncs them while
                    // dragging (count too high → past-end reads = phantom geometry;
                    // too low → missed bricks = gaps).
                    .after(super::bake_scheduler::apply_bakes),
            );

        #[cfg(feature = "editor")]
        {
            app.add_systems(Update, sync_sdf_shader_defs);
        }

        // Get shader handle before mutable borrow of render app
        let shader_handle = app.world().resource::<SdfShaderHandle>().0.clone();

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        // Pass shader handle directly to render app (RenderStartup runs before Extract)
        render_app.insert_resource(SdfShaderHandle(shader_handle));

        render_app
            .add_systems(ExtractSchedule, extract_sdf_atlas)
            .add_systems(ExtractSchedule, extract_sdf_materials)
            .add_systems(ExtractSchedule, extract_texture_library)
            .add_systems(ExtractSchedule, extract_shader_defs)
            .init_resource::<TextureStreamState>()
            .init_resource::<LastAtlasGen>()
            .init_resource::<LastAtlasTopology>()
            .init_resource::<AtlasCapacity>()
            .add_systems(Render, prepare_sdf_atlas_gpu)
            .add_systems(Render, prepare_sdf_materials_gpu)
            .add_systems(Render, init_texture_streaming)
            .add_systems(Render, upload_texture_layers.after(init_texture_streaming))
            .add_systems(Render, rebuild_pipeline_on_def_change)
            .add_systems(RenderStartup, init_sdf_pipeline)
            .add_render_graph_node::<ViewNodeRunner<SdfNode>>(Core3d, SdfLabel)
            // Run the SDF fullscreen pass between the opaque and transparent
            // passes. Gizmos (transform handles, bounds) draw in the Transparent3d
            // phase, so the SDF surface must be composited *before* them — otherwise
            // the SDF pass (which fills background on a ray miss) paints over the
            // gizmos. Their negative depth_bias then keeps them on top.
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::MainOpaquePass,
                    SdfLabel,
                    Node3d::MainTransparentPass,
                ),
            );
    }
}

// --- Main World: Prepare Camera Data ---

/// Main-world material table, rebuilt each frame from the SDF volumes in resolved
/// material-id order. Extracted into the render world and uploaded as a storage
/// buffer. The PBR workflow extends [`GpuSdfMaterial`]; this builder just copies
/// the extra fields across.
#[derive(Resource, Default)]
pub struct SdfMaterialTable {
    materials: Vec<GpuSdfMaterial>,
}

#[allow(clippy::too_many_arguments)] // Bevy system params; splitting is artificial.
fn prepare_sdf_camera_data(
    mut commands: Commands,
    cameras: Query<(Entity, &Camera, &Transform), With<SdfCamera>>,
    atlas: Res<SdfAtlas>,
    config: Res<SdfGridConfig>,
    raymarch: Res<super::SdfRaymarchParams>,
    registry: Res<super::edits::MaterialRegistry>,
    mut material_table: ResMut<SdfMaterialTable>,
) {
    // The GPU material table mirrors the global registry verbatim: row i = the
    // material with global id i. Bricks index it by their palette ids. Rebuilt only
    // when the registry changes (it is the single source of truth, not per-volume).
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
                _pad0: 0,
                _pad1: 0,
            });
        }
    }

    // grid_dims.w = the shader's chunk-table binary-search bound = distinct resident
    // chunks (NOT brick count). Must match `chunk_data.len()` extract uploads.
    let num_chunks = super::chunk::resident_chunks(&atlas, &config).len() as u32;
    let bpa = config.bricks_per_axis();
    let grid_size = config.grid_size;

    for (entity, camera, transform) in &cameras {
        let view_from_world = transform.to_matrix().inverse();
        let clip_from_world = camera.clip_from_view() * view_from_world;
        let inv_view_proj = clip_from_world.inverse();

        let size = camera
            .physical_viewport_size()
            .unwrap_or(UVec2::new(1920, 1080));

        // Pixel cone half-width per unit ray distance: the world radius one pixel covers
        // at distance 1. For a perspective projection `proj.y_axis.y = cot(fov_y/2)`, the
        // full vertical world extent at distance 1 is `2·tan(fov_y/2)`, so one pixel spans
        // `2·tan(fov_y/2)/height` and its half-width (radius) is `tan(fov_y/2)/height`.
        // Scaled by `cone_scale` so the surface-within-a-pixel test can be tuned.
        let proj = camera.clip_from_view();
        let tan_half_fov_y = 1.0 / proj.y_axis.y.max(1e-6);
        let pixel_cone = (tan_half_fov_y / size.y.max(1) as f32) * raymarch.cone_scale;

        commands.entity(entity).insert(SdfCameraData {
            inv_view_proj,
            clip_from_world,
            camera_pos: transform.translation.extend(0.0),
            screen_params: Vec4::new(size.x as f32, size.y as f32, 0.0, 0.0),
            grid_origin: Vec4::new(
                config.world_origin().x,
                config.world_origin().y,
                config.world_origin().z,
                config.voxel_size,
            ),
            grid_dims: Vec4::new(
                grid_size as f32,
                bpa as f32,
                config.brick_size as f32,
                num_chunks as f32,
            ),
            debug_params: Vec4::new(
                raymarch.max_steps as f32,
                raymarch.max_dist,
                raymarch.sdf_eps,
                0.0,
            ),
            // March tuning: the pixel cone half-width per unit ray distance drives the
            // screen-space termination (a surface within a pixel ends the march, so far
            // geometry resolves at coarse LOD); `cubic_band` is the near-surface distance
            // within which a LOD-0 sample switches to the exact analytic cubic.
            march_params: Vec4::new(
                pixel_cone,
                raymarch.cubic_band,
                raymarch.over_relax,
                0.0,
            ),
            lod_params: Vec4::new(
                config.lod_count as f32,
                config.ring_bricks as f32,
                config.voxel_size,
                config.cell_stride() as f32,
            ),
        });
    }
}

// --- Bridge: Sync debug state to shader defs ---

#[cfg(feature = "editor")]
fn sync_sdf_shader_defs(
    registry: Res<crate::editor::registry::ShaderDebugRegistry>,
    state: Res<crate::editor::registry::ShaderDebugState>,
    mut defs: ResMut<SdfShaderDefs>,
) {
    let active = state.active_defines_for_prefix(&registry, "sdf/");
    if defs.defs != active {
        defs.defs = active;
    }
}

// --- Extract: Shader Defs ---

#[derive(Resource, Default)]
struct ExtractedShaderDefs {
    defs: Vec<String>,
    changed: bool,
}

fn extract_shader_defs(
    defs: Extract<Res<SdfShaderDefs>>,
    mut commands: Commands,
    existing: Option<ResMut<ExtractedShaderDefs>>,
) {
    let new_defs = defs.defs.clone();
    match existing {
        Some(mut existing) => {
            if existing.defs != new_defs {
                existing.defs = new_defs;
                existing.changed = true;
            } else {
                existing.changed = false;
            }
        }
        None => {
            commands.insert_resource(ExtractedShaderDefs {
                defs: new_defs,
                changed: false,
            });
        }
    }
}

fn rebuild_pipeline_on_def_change(
    mut pipeline: ResMut<SdfPipeline>,
    extracted: Option<Res<ExtractedShaderDefs>>,
    shader_handle: Res<SdfShaderHandle>,
    pipeline_cache: Res<PipelineCache>,
    fullscreen_shader: Res<FullscreenShader>,
) {
    let Some(extracted) = extracted else { return };
    if !extracted.changed {
        return;
    }

    let shader_defs: Vec<_> = extracted.defs.iter().map(|s| s.as_str().into()).collect();
    let shader = shader_handle.0.clone();
    let vertex_state = fullscreen_shader.to_vertex_state();

    let new_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("sdf_pipeline".into()),
        layout: vec![pipeline.layout_0.clone(), pipeline.layout_1.clone()],
        vertex: vertex_state,
        fragment: Some(FragmentState {
            shader,
            shader_defs,
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::bevy_default(),
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: CompareFunction::GreaterEqual,
            stencil: default(),
            bias: default(),
        }),
        ..default()
    });

    pipeline.pipeline_id = new_id;
}

// --- Extract: Pack Atlas for GPU ---

/// Pixel origin of `tile` in the 2D-tiled atlas. Tiles wrap into rows so the width
/// stays bounded (a single strip overflows wgpu's 32768 max once brick count passes
/// ~512). Returns `(col_px, row_px)`; `atlas_base = col_px | row_px<<16`.
fn tile_origin(tile: u32) -> (u32, u32) {
    let edge = BRICK_EDGE as u32;
    let tile_width = edge * edge; // 64
    let col_px = (tile % ATLAS_TILES_PER_ROW) * tile_width;
    let row_px = (tile / ATLAS_TILES_PER_ROW) * edge;
    (col_px, row_px)
}

/// Pack one brick's voxels into a tile-local `(dist[512], mat[2048])` pair, in the
/// same `(y*EDGE+x, z)` pixel layout the atlas uses. Shared by the full and partial
/// upload paths so they're byte-identical.
fn pack_tile_texels(packed: &super::atlas::PackedBrick) -> (Vec<i16>, Vec<i16>) {
    let edge = BRICK_EDGE as u32;
    let tile_width = (edge * edge) as usize; // 64 px wide, EDGE tall
    let mut dist = vec![0i16; tile_width * edge as usize];
    let mut mat = vec![i16::MAX; tile_width * edge as usize * 4];
    for z in 0..edge {
        for y in 0..edge {
            for x in 0..edge {
                let src_idx = (z * edge * edge + y * edge + x) as usize;
                // Tile-local destination: u in [0,64), v in [0,EDGE).
                let local = (z * tile_width as u32 + y * edge + x) as usize;
                dist[local] = packed.dist[src_idx];
                let mat_base = src_idx * PALETTE_K;
                for k in 0..PALETTE_K {
                    mat[local * 4 + k] = packed.mat_dist[mat_base + k];
                }
            }
        }
    }
    (dist, mat)
}

fn extract_sdf_atlas(
    atlas: Extract<Res<SdfAtlas>>,
    config: Extract<Res<SdfGridConfig>>,
    mut last_gen: ResMut<LastAtlasGen>,
    mut last_topology: ResMut<LastAtlasTopology>,
    mut capacity: ResMut<AtlasCapacity>,
    mut commands: Commands,
) {
    // Nothing changed since the last upload — skip the rebuild entirely so idle
    // frames cost no extract/prepare work. `prepare_sdf_atlas_gpu` keeps last
    // frame's GPU resources because the inserted resource has `dirty = false`.
    if atlas.generation == last_gen.0 {
        commands.insert_resource(ExtractedSdfAtlas::default()); // dirty = false
        return;
    }
    last_gen.0 = atlas.generation;

    let num_bricks = atlas.bricks.len() as u32;
    if num_bricks == 0 {
        commands.insert_resource(ExtractedSdfAtlas::default());
        return;
    }

    let edge = BRICK_EDGE as u32;
    let tile_width = edge * edge; // 64
    let texture_width = ATLAS_TILES_PER_ROW * tile_width;

    // Tile origins come from the stable allocator (its high-water mark), NOT brick
    // iteration order — so a re-baked brick keeps its sub-rect across frames.
    let required_rows = atlas.tiles.high_water().div_ceil(ATLAS_TILES_PER_ROW).max(1);
    let texture_height = required_rows * edge;

    // Realloc when a full bake happened or the atlas must grow taller. The texture
    // never shrinks except on a full bake, so a tile origin stays valid until then.
    let realloc = atlas.last_bake_was_full || required_rows > capacity.rows;

    // The chunk lookup / tile-run tables only need rebuilding when the atlas topology
    // changed (a brick entered/exited, or a palette changed) or on a realloc. A frame
    // that only re-baked texels in place reuses last frame's buffers — skipping the
    // O(bricks) table rebuild + full lookup re-upload that ran every applied frame before.
    let tables_dirty = realloc || atlas.topology_generation != last_topology.0;
    last_topology.0 = atlas.topology_generation;

    let (chunk_data, tile_run_data) = if tables_dirty {
        // Build the sorted chunk lookup table + packed per-chunk brick runs from the
        // resident bricks. Absolute chunk keys (no ring origin) → CPU/GPU agree by
        // construction. `chunk::build_chunk_tables` owns the grouping + sort.
        let tables = super::chunk::build_chunk_tables(&atlas, &config, |key| {
            let tile = atlas
                .tiles
                .tile(key)
                .expect("baked brick must have an allocated tile");
            let (col_px, row_px) = tile_origin(tile);
            let p = atlas.bricks[key].palette;
            super::chunk::BrickTile {
                atlas_base: col_px | (row_px << 16),
                pal01: p[0] as u32 | ((p[1] as u32) << 16),
                pal23: p[2] as u32 | ((p[3] as u32) << 16),
            }
        });
        let chunk_data: Vec<GpuChunkLookup> = tables
            .chunks
            .iter()
            .map(|c| GpuChunkLookup {
                key_hi: c.key_hi,
                key_lo: c.key_lo,
                occ_lo: c.occ_lo,
                occ_hi: c.occ_hi,
                tile_run_base: c.tile_run_base,
            })
            .collect();
        let tile_run_data: Vec<GpuBrickTile> = tables
            .tile_run
            .iter()
            .map(|b| GpuBrickTile {
                atlas_base: b.atlas_base,
                pal01: b.pal01,
                pal23: b.pal23,
            })
            .collect();
        (chunk_data, tile_run_data)
    } else {
        (Vec::new(), Vec::new())
    };

    if realloc {
        capacity.rows = required_rows;
        let pixels = (texture_width * texture_height) as usize;
        let mut dist_data = vec![0i16; pixels];
        let mut mat_data = vec![i16::MAX; pixels * 4]; // far sentinel loses the argmin
        for (key, packed) in atlas.bricks.iter() {
            let tile = atlas.tiles.tile(key).unwrap();
            let (col_px, row_px) = tile_origin(tile);
            let (dist, mat) = pack_tile_texels(packed);
            // Blit the tile-local buffers into the full texture at (col_px,row_px).
            for v in 0..edge {
                for u in 0..tile_width {
                    let local = (v * tile_width + u) as usize;
                    let dst = ((row_px + v) * texture_width + col_px + u) as usize;
                    dist_data[dst] = dist[local];
                    mat_data[dst * 4..dst * 4 + 4].copy_from_slice(&mat[local * 4..local * 4 + 4]);
                }
            }
        }
        commands.insert_resource(ExtractedSdfAtlas {
            dist_data,
            mat_data,
            changed_tiles: Vec::new(),
            chunk_data,
            tile_run_data,
            tables_dirty: true,
            texture_width,
            texture_height,
            realloc: true,
            dirty: true,
        });
        return;
    }

    // Partial upload: only the tiles the incremental bake touched.
    let mut changed_tiles = Vec::with_capacity(atlas.changed_tiles.len());
    // Map tile → coord once so we can pull the baked brick. (changed_tiles holds tile
    // indices; bricks are keyed by coord.)
    for (key, packed) in atlas.bricks.iter() {
        let tile = atlas.tiles.tile(key).unwrap();
        if !atlas.changed_tiles.contains(&tile) {
            continue;
        }
        let (col_px, row_px) = tile_origin(tile);
        let (dist, mat) = pack_tile_texels(packed);
        changed_tiles.push(TileTexels {
            atlas_base: col_px | (row_px << 16),
            dist,
            mat,
        });
    }

    commands.insert_resource(ExtractedSdfAtlas {
        dist_data: Vec::new(),
        mat_data: Vec::new(),
        changed_tiles,
        chunk_data,
        tile_run_data,
        tables_dirty,
        texture_width,
        texture_height,
        realloc: false,
        dirty: true,
    });
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
    // std430: each GpuSdfMaterial is 48 bytes (vec4 + f32 + 5×u32 + 2×u32 pad).
    let mut bytes = Vec::with_capacity(extracted.materials.len() * 48);
    for m in &extracted.materials {
        for c in [
            m.base_color.x,
            m.base_color.y,
            m.base_color.z,
            m.base_color.w,
        ] {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
        bytes.extend_from_slice(&m.blend_softness.to_le_bytes());
        bytes.extend_from_slice(&m.tex_diffuse.to_le_bytes());
        bytes.extend_from_slice(&m.tex_normal.to_le_bytes());
        bytes.extend_from_slice(&m.tex_mra.to_le_bytes());
        bytes.extend_from_slice(&m.tex_height.to_le_bytes());
        bytes.extend_from_slice(&m.tex_edge.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
    }
    let buffer = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_material_buffer"),
        contents: &bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    gpu_atlas.material_buffer = Some(buffer);
}

// --- Extract + upload: PBR texture-array library ---

fn extract_texture_library(
    library: Extract<Res<crate::assets::MaterialTextureLibrary>>,
    mut commands: Commands,
) {
    commands.insert_resource(ExtractedTextureLibrary {
        variants: library.variants.clone(),
    });
}

/// BC7 array formats per `MapArray`: sRGB for diffuse (0), linear for the rest.
const PBR_ARRAY_FORMATS: [TextureFormat; super::edits::MATERIAL_TEX_MAPS] = [
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
fn init_texture_streaming(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    extracted: Option<Res<ExtractedTextureLibrary>>,
    mut gpu_atlas: ResMut<SdfGpuAtlas>,
    mut stream: ResMut<TextureStreamState>,
) {
    use super::textures::TEXTURE_SIZE;
    use crate::assets::MAX_TEXTURE_LAYERS;

    // 1) Allocate the fixed-cap arrays once (the moment the render device is up). The
    // arrays are sized to MAX_TEXTURE_LAYERS so the demand-driven library can append
    // variants without ever recreating the textures or rebuilding the bind group.
    if !stream.allocated {
        let mips = super::bc7::mip_count(TEXTURE_SIZE);
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
        let fallback: [[u8; 4]; super::edits::MATERIAL_TEX_MAPS] = [
            [255, 0, 255, 255],
            [128, 128, 255, 255],
            [0, 255, 255, 255],
            [0, 0, 0, 255],
            [0, 0, 0, 255],
        ];

        let mut textures = Vec::with_capacity(super::edits::MATERIAL_TEX_MAPS);
        let views: [TextureView; super::edits::MATERIAL_TEX_MAPS] = std::array::from_fn(|i| {
            let fill = super::bc7::solid_fill_bc7(fallback[i], TEXTURE_SIZE, MAX_TEXTURE_LAYERS);
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
        let v = &extracted.variants[layer as usize];
        let slug = v.slug.clone();
        let dir = v.dir.clone();
        stream.tasks.push(pool.spawn(async move {
            let maps = super::textures::encode_variant_bc7(&slug, &dir);
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
fn upload_texture_layers(queue: Res<RenderQueue>, mut stream: ResMut<TextureStreamState>) {
    if stream.tasks.is_empty() {
        return;
    }
    use super::textures::TEXTURE_SIZE;

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
        let tex_mips = super::bc7::mip_count(TEXTURE_SIZE);
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

// --- Prepare: Upload to GPU ---

fn prepare_sdf_atlas_gpu(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    extracted: Option<Res<ExtractedSdfAtlas>>,
    mut gpu_atlas: ResMut<SdfGpuAtlas>,
) {
    let Some(extracted) = extracted else { return };
    if !extracted.dirty {
        return;
    }

    // Rebuild the chunk lookup + tile-run buffers only when the atlas topology changed
    // (`tables_dirty`). A texel-only frame (camera idle / an edit nudged in place) reuses
    // last frame's buffers untouched, skipping the per-frame O(bricks) re-upload.
    if extracted.tables_dirty {
        if extracted.chunk_data.is_empty() {
            // No resident chunks (atlas empty / fully evicted). Leave last frame's buffers;
            // the camera uniform's chunk count (grid_dims.w) is 0 so the shader searches
            // nothing and renders background. Avoids a zero-sized storage buffer.
            return;
        }

        // Chunk lookup table (std430: 5 × u32 per entry), sorted by absolute key — the
        // shader binary-searches it.
        let mut chunk_bytes = Vec::with_capacity(extracted.chunk_data.len() * 20);
        for c in &extracted.chunk_data {
            chunk_bytes.extend_from_slice(&c.key_hi.to_le_bytes());
            chunk_bytes.extend_from_slice(&c.key_lo.to_le_bytes());
            chunk_bytes.extend_from_slice(&c.occ_lo.to_le_bytes());
            chunk_bytes.extend_from_slice(&c.occ_hi.to_le_bytes());
            chunk_bytes.extend_from_slice(&c.tile_run_base.to_le_bytes());
        }
        gpu_atlas.lookup_buffer = Some(device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("sdf_chunk_lookup_buffer"),
            contents: &chunk_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        }));

        // Packed per-chunk brick tile-run buffer (std430: 3 × u32 per resident brick).
        let mut tile_bytes = Vec::with_capacity(extracted.tile_run_data.len() * 12);
        for b in &extracted.tile_run_data {
            tile_bytes.extend_from_slice(&b.atlas_base.to_le_bytes());
            tile_bytes.extend_from_slice(&b.pal01.to_le_bytes());
            tile_bytes.extend_from_slice(&b.pal23.to_le_bytes());
        }
        gpu_atlas.chunk_tile_buffer = Some(device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("sdf_chunk_tile_buffer"),
            contents: &tile_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        }));
    }

    if extracted.realloc {
        // Full rebuild / grow: recreate both textures sized to the new height and
        // upload everything. Keep the Texture handles so later partial bakes can
        // write_texture into them.
        let size = Extent3d {
            width: extracted.texture_width,
            height: extracted.texture_height,
            depth_or_array_layers: 1,
        };
        let mut dist_bytes = Vec::with_capacity(extracted.dist_data.len() * 2);
        for v in &extracted.dist_data {
            dist_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let dist_tex = device.create_texture_with_data(
            &queue,
            &TextureDescriptor {
                label: Some("sdf_dist_atlas"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R16Snorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            },
            TextureDataOrder::default(),
            &dist_bytes,
        );
        let mat_tex = device.create_texture_with_data(
            &queue,
            &TextureDescriptor {
                label: Some("sdf_mat_atlas"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Snorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            },
            TextureDataOrder::default(),
            &i16s_to_le_bytes(&extracted.mat_data),
        );

        gpu_atlas.dist_view = Some(dist_tex.create_view(&TextureViewDescriptor::default()));
        gpu_atlas.mat_view = Some(mat_tex.create_view(&TextureViewDescriptor::default()));
        gpu_atlas.dist_tex = Some(dist_tex);
        gpu_atlas.mat_tex = Some(mat_tex);
        if gpu_atlas.sampler.is_none() {
            gpu_atlas.sampler = Some(device.create_sampler(&SamplerDescriptor {
                label: Some("sdf_atlas_sampler"),
                mag_filter: FilterMode::Nearest,
                min_filter: FilterMode::Nearest,
                mipmap_filter: FilterMode::Nearest,
                ..default()
            }));
        }
        return;
    }

    // Partial upload: write_texture only the changed tiles into the existing
    // textures (64×8 sub-rects). No realloc, no view change → bind group unaffected.
    let (Some(dist_tex), Some(mat_tex)) = (&gpu_atlas.dist_tex, &gpu_atlas.mat_tex) else {
        return; // never reallocated yet — nothing to patch into
    };
    let edge = BRICK_EDGE as u32;
    let tile_width = edge * edge; // 64
    for t in &extracted.changed_tiles {
        let col_px = t.atlas_base & 0xffff;
        let row_px = t.atlas_base >> 16;
        let tile_extent = Extent3d {
            width: tile_width,
            height: edge,
            depth_or_array_layers: 1,
        };
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: dist_tex,
                mip_level: 0,
                origin: Origin3d {
                    x: col_px,
                    y: row_px,
                    z: 0,
                },
                aspect: TextureAspect::All,
            },
            &i16s_to_le_bytes(&t.dist),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(tile_width * 2), // R16: 2 bytes/texel
                rows_per_image: Some(edge),
            },
            tile_extent,
        );
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: mat_tex,
                mip_level: 0,
                origin: Origin3d {
                    x: col_px,
                    y: row_px,
                    z: 0,
                },
                aspect: TextureAspect::All,
            },
            &i16s_to_le_bytes(&t.mat),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(tile_width * 8), // Rgba16: 8 bytes/texel
                rows_per_image: Some(edge),
            },
            tile_extent,
        );
    }
    debug!(
        "SDF atlas: patched {} changed tile(s)",
        extracted.changed_tiles.len()
    );
}

/// Flatten an i16 slice to little-endian bytes for texture upload.
fn i16s_to_le_bytes(data: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

// --- Render World: Pipeline Init ---

fn init_sdf_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    fullscreen_shader: Res<FullscreenShader>,
    shader_handle: Res<SdfShaderHandle>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout_0 = BindGroupLayoutDescriptor::new(
        "sdf_bind_group_0",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (uniform_buffer::<SdfCameraData>(true),),
        ),
    );
    let layout_1 = BindGroupLayoutDescriptor::new(
        "sdf_bind_group_1",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                // binding 0: distance atlas (R8Snorm, filterable)
                texture_2d(TextureSampleType::Float { filterable: true }),
                // binding 1: nearest sampler
                sampler(SamplerBindingType::Filtering),
                // binding 2: chunk lookup table (sorted, binary-searched)
                storage_buffer_read_only::<GpuChunkLookup>(false),
                // binding 3: per-palette-slot distance atlas (Rgba16Snorm, 4 slots)
                texture_2d(TextureSampleType::Float { filterable: false }),
                // binding 4: material table (indexed by global material id)
                storage_buffer_read_only::<GpuSdfMaterial>(false),
                // binding 5: PBR-array filtering+mip sampler
                sampler(SamplerBindingType::Filtering),
                // bindings 6..10: PBR texture arrays (diffuse, normal, mra, height, edge)
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                // binding 11: packed per-chunk brick tile runs
                storage_buffer_read_only::<GpuBrickTile>(false),
            ),
        ),
    );

    let shader = shader_handle.0.clone();
    let vertex_state = fullscreen_shader.to_vertex_state();

    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("sdf_pipeline".into()),
        layout: vec![layout_0.clone(), layout_1.clone()],
        vertex: vertex_state,
        fragment: Some(FragmentState {
            shader: shader.clone(),
            shader_defs: vec![],
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::bevy_default(),
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: CompareFunction::GreaterEqual,
            stencil: default(),
            bias: default(),
        }),
        ..default()
    });

    // Create minimal dummy atlas so bind group 1 is always valid
    let dummy_tex = device.create_texture_with_data(
        &queue,
        &TextureDescriptor {
            label: Some("sdf_dummy_atlas"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R16Snorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        TextureDataOrder::default(),
        &[0u8, 0u8],
    );
    let dummy_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("sdf_dummy_atlas_sampler"),
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        mipmap_filter: FilterMode::Nearest,
        ..default()
    });
    // One zeroed 20-byte chunk lookup entry so binding 2 is valid pre-bake.
    let dummy_lookup = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_dummy_chunk_lookup"),
        contents: &[0u8; 20],
        usage: BufferUsages::STORAGE,
    });
    // One zeroed 12-byte brick-tile entry so binding 12 is valid pre-bake.
    let dummy_chunk_tile = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_dummy_chunk_tile"),
        contents: &[0u8; 12],
        usage: BufferUsages::STORAGE,
    });
    // One zeroed 48-byte GpuSdfMaterial row so binding 4 meets the struct's minimum
    // size before the real table uploads.
    let dummy_material = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_dummy_material"),
        contents: &[0u8; 48],
        usage: BufferUsages::STORAGE,
    });
    // Matching dummy material atlas (Rgba16Snorm = 8 bytes/texel) so bind group 1 is
    // always valid before the first bake.
    let dummy_mat_tex = device.create_texture_with_data(
        &queue,
        &TextureDescriptor {
            label: Some("sdf_dummy_mat_atlas"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Snorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        TextureDataOrder::default(),
        &[0u8; 8],
    );

    // Dummy 1×1×1 PBR arrays + a filtering sampler so bind group 1 is valid before
    // the texture library uploads. Each is a D2Array view of one zeroed layer.
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

    commands.insert_resource(SdfPipeline {
        pipeline_id,
        layout_0,
        layout_1,
        shader_handle: shader,
    });
    commands.insert_resource(SdfGpuAtlas {
        dist_tex: None,
        dist_view: Some(dummy_tex.create_view(&TextureViewDescriptor::default())),
        mat_tex: None,
        mat_view: Some(dummy_mat_tex.create_view(&TextureViewDescriptor::default())),
        sampler: Some(dummy_sampler),
        lookup_buffer: Some(dummy_lookup),
        chunk_tile_buffer: Some(dummy_chunk_tile),
        material_buffer: Some(dummy_material),
        tex_array_views: Some(dummy_tex_views),
        tex_sampler: Some(dummy_tex_sampler),
    });
}
