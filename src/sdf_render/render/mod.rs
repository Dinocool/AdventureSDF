use bevy::core_pipeline::FullscreenShader;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_component::{
    ComponentUniforms, ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin,
};
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only, storage_buffer_sized, texture_2d, texture_2d_array,
    texture_storage_2d, uniform_buffer,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::view::{ViewDepthTexture, ViewTarget};
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderStartup};
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};

use super::atlas::{BRICK_EDGE, SdfAtlas};
use super::{SdfCamera, SdfGridConfig, SdfRenderEnabled};

// Concern-specific submodules of the render path (bake compute, cone prepass, PBR texture
// streaming); each reaches the shared render types here via `use super::*`.
mod atlas_pages;
mod atlas_upload;
mod bake;
mod chunk_tables;
mod cone;
mod pbr_textures;

use atlas_pages::AtlasPages;
use chunk_tables::ChunkTableBuffers;

// --- GPU Types ---

/// Atlas tiles per row. Width = this × 64 px. 256 → 16384 px wide, half the 32768
/// wgpu limit, so it never overflows while keeping the texture reasonably square.
/// Defined in `atlas` so the GPU-bake realloc mirror agrees on the layout.
use super::atlas::ATLAS_TILES_PER_ROW;

#[derive(Component, Clone, Copy, ShaderType, Default, ExtractComponent, Reflect)]
#[reflect(Component)]
struct SdfCameraData {
    inv_view_proj: Mat4,
    /// Forward view-projection. Used to write true reverse-Z projection depth from
    /// the raymarch hit, so the SDF surface occludes/are-occluded-by other passes
    /// (wireframe, gizmos) through the normal depth buffer.
    clip_from_world: Mat4,
    /// LAST frame's `clip_from_world`. Reprojects a reflected world point into the previous
    /// frame's screen for the SSR reflection path.
    prev_clip_from_world: Mat4,
    camera_pos: Vec4,
    screen_params: Vec4, // xy = screen_size; zw unused (was surface_bias — iso-offset removed)
    grid_origin: Vec4,   // xyz = grid origin, w = voxel_size
    grid_dims: Vec4, // z = brick_size (8.0); x/y/w unused (chunk count = arrayLength(&chunk_buf))
    debug_params: Vec4, // x = max_steps, y = max_dist, z = sdf_eps, w = unused
    /// x = pixel_cone (world radius per unit ray distance per pixel), y = reserved
    /// (was cubic_band), z = over_relax, w = lod_blend_band.
    march_params: Vec4,
    /// x = lod_count, y = ring_bricks, z = base voxel_size, w = cell_stride.
    lod_params: Vec4,
    /// xyz = world-space direction toward the key light; w unused.
    sun_dir: Vec4,
    /// rgb = key-light radiance; w unused.
    sun_color: Vec4,
}

/// GPU mirror of a [`super::edits::MaterialDef`], one per global material id, in a
/// storage buffer indexed by id. Carries the PBR texture-array layer for each map
/// (`u32::MAX` = none); the shader samples those layers via triplanar projection.
/// 80 bytes, 16-byte aligned for std430. The three `_pad*` words align `emissive` (a
/// `vec4`) to its 16-byte boundary at offset 64.
#[derive(ShaderType, Clone, Copy, Default)]
struct GpuSdfMaterial {
    base_color: Vec4,
    blend_softness: f32,
    tex_diffuse: u32,
    tex_normal: u32,
    tex_mra: u32,
    tex_height: u32,
    tex_edge: u32,
    /// Scalar metallic/roughness fallbacks (used by the shader when `tex_mra` is absent).
    metallic: f32,
    roughness: f32,
    /// Parallax-occlusion relief depth (UV units) for this material's height map. 0 = flat.
    parallax_scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    /// Emissive radiance, linear RGB in `xyz` (intensity premultiplied in `MaterialDef`);
    /// `w` is spare. `vec4` so it's 16-byte aligned at offset 64 (struct = 80 bytes).
    emissive: Vec4,
}

// --- GPU Atlas ---

#[derive(Resource, Default)]
struct SdfGpuAtlas {
    /// Paged distance + material atlases (R16Snorm + Rgba16Snorm), grown one fixed-size page at a
    /// time with NO copy (see [`AtlasPages`]). `None` until `init_sdf_pipeline` creates the pool.
    /// The bake writes tiles straight into the live pages; the fragment shader reads them as a
    /// `binding_array`. Replaces the old single-texture atlas whose taller-realloc + full-copy
    /// spiked VRAM ~2× and cost O(N²) during a fill.
    pages: Option<AtlasPages>,
    sampler: Option<Sampler>,
    /// Chunk lookup directory (binding 2) + packed per-chunk tile runs (binding 11), as the shared
    /// growable-storage-buffer pool (`ChunkTableBuffers`). `None` buffers until `init_sdf_pipeline`.
    tables: ChunkTableBuffers,
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

/// Material table extracted from the main world for GPU upload.
#[derive(Resource, Default)]
struct ExtractedSdfMaterials {
    materials: Vec<GpuSdfMaterial>,
}

// --- Pipeline ---

// BISECT: minimal shader while building features back up after the division-free fix.
const SDF_SHADER_PATH: &str = "shaders/sdf_raymarch.wgsl";

#[derive(Resource)]
struct SdfPipeline {
    pipeline_id: CachedRenderPipelineId,
    layout_0: BindGroupLayoutDescriptor,
    layout_1: BindGroupLayoutDescriptor,
    /// Cone-prepass seed texture, read (textureLoad) by the fragment march to start each
    /// pixel at its tile's seed distance instead of 0.
    layout_2: BindGroupLayoutDescriptor,
    #[expect(dead_code)]
    shader_handle: Handle<Shader>,
    /// The shader defs the current `pipeline_id` was queued with. Rebuild compares the
    /// extracted defs against this (not a per-frame `changed` flag, which is fragile at
    /// startup when the defs haven't synced yet) and re-queues only on a real mismatch.
    current_defs: Vec<String>,
}

/// The G-buffer's three MRT colour formats (all linear HDR). Shared by the pipeline target
/// list and the texture allocation in `prepare_sdf_gbuffer` so they can't drift.
const GBUFFER_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Combine pipeline: the final deferred-lighting pass. Reads the G-buffer (group 1) + camera
/// (group 0), evaluates the analytic sun + emissive, and writes the lit result to the HDR view
/// target. Rebuilt on shader-def change so its `#ifdef` debug views (SDF_DEBUG_*) recompile when
/// toggled in the editor.
#[derive(Resource)]
struct SdfCombinePipeline {
    pipeline_id: CachedRenderPipelineId,
    /// G-buffer (3 tex) + sampler read layout.
    layout: BindGroupLayoutDescriptor,
    /// The shader defs the current `pipeline_id` was queued with (rebuild on mismatch).
    current_defs: Vec<String>,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct SdfCombineLabel;

#[derive(Resource)]
struct SdfCombineShaderHandle(Handle<Shader>);

/// Deferred lit-pass shader (analytic sun + emissive → view target).
const SDF_COMBINE_SHADER_PATH: &str = "shaders/sdf_deferred_lit.wgsl";

/// Deferred G-buffer: the three per-view `Rgba16Float` targets the primary SDF pass writes
/// (replacing the old forward-lit single colour). `albedo` carries rgb albedo + camera distance
/// in alpha; `normal_mat` carries the octahedral world normal + metallic/roughness; `emissive`
/// carries premultiplied emissive radiance. Re-created lazily to match the viewport size. The
/// deferred lit pass samples all three. The matching `sampler` is a non-filtering nearest sampler
/// (one G-buffer texel per pixel — no interpolation wanted).
#[derive(Resource, Default)]
struct SdfGBuffer {
    albedo: Option<Texture>,
    albedo_view: Option<TextureView>,
    normal_mat: Option<Texture>,
    normal_mat_view: Option<TextureView>,
    emissive: Option<Texture>,
    emissive_view: Option<TextureView>,
    sampler: Option<Sampler>,
    size: UVec2,
}


#[derive(Resource, Default)]
pub struct SdfShaderDefs {
    pub defs: Vec<String>,
}

// --- Render Graph ---

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct SdfGBufferLabel;

fn create_dummy_bg0(device: &RenderDevice, layout: &BindGroupLayout) -> BindGroup {
    let camera_buf = device.create_buffer(&BufferDescriptor {
        label: Some("sdf_dummy_camera_uniform"),
        size: 512,
        usage: BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    device.create_bind_group(
        "sdf_bind_group_0_empty",
        layout,
        &BindGroupEntries::sequential((camera_buf.as_entire_buffer_binding(),)),
    )
}

/// Build the 12-entry atlas bind group (group 1): dist view + sampler, chunk-lookup buffer, mat
/// view + material buffer, the PBR texture sampler + 5 array views, and the packed chunk-tile
/// buffer. Shared VERBATIM by the G-buffer fragment pass ([`SdfGBufferNode`]) and the cone prepass
/// ([`cone::SdfConeNode`]) — was copy-pasted in both. Every binding is required; panics if the atlas
/// isn't fully initialized (the nodes already early-out before this when resources are missing).
fn atlas_bind_group_1(
    device: &RenderDevice,
    layout: &BindGroupLayout,
    gpu_atlas: &SdfGpuAtlas,
    label: &str,
) -> BindGroup {
    let tex_views = gpu_atlas.tex_array_views.as_ref().unwrap();
    let pages = gpu_atlas.pages.as_ref().unwrap();
    // Live page views + dummy fill to ATLAS_MAX_PAGES, bound as the `binding_array`s at 0 and 3.
    let dist_refs = pages.dist_refs();
    let mat_refs = pages.mat_refs();
    device.create_bind_group(
        label,
        layout,
        &BindGroupEntries::sequential((
            &dist_refs[..],
            gpu_atlas.sampler.as_ref().unwrap(),
            gpu_atlas.tables.lookup_buffer().as_entire_buffer_binding(),
            &mat_refs[..],
            gpu_atlas.material_buffer.as_ref().unwrap().as_entire_buffer_binding(),
            gpu_atlas.tex_sampler.as_ref().unwrap(),
            &tex_views[0],
            &tex_views[1],
            &tex_views[2],
            &tex_views[3],
            &tex_views[4],
            gpu_atlas.tables.tile_buffer().as_entire_buffer_binding(),
        )),
    )
}

#[derive(Default)]
struct SdfGBufferNode;

impl ViewNode for SdfGBufferNode {
    type ViewQuery = &'static ViewDepthTexture;

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        depth: QueryItem<Self::ViewQuery>,
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

        // The G-buffer textures must be allocated (prepare_sdf_gbuffer runs each frame). If not
        // yet (no view this frame), skip — the deferred lit pass will also skip.
        let gbuffer = world.resource::<SdfGBuffer>();
        let (Some(albedo_view), Some(normal_view), Some(emissive_view)) = (
            &gbuffer.albedo_view,
            &gbuffer.normal_mat_view,
            &gbuffer.emissive_view,
        ) else {
            return Ok(());
        };

        // During a window resize the shared view-depth texture can re-size a frame before
        // `prepare_sdf_gbuffer` re-sizes the colour targets (they're driven off different view
        // resources). A render pass requires ALL attachments to share one size, so skip the frame on
        // a mismatch — prepare re-sizes the G-buffer next frame and rendering resumes (invisible
        // during a drag-resize). Without this, wgpu aborts with "Attachments have differing sizes".
        if gbuffer.size.x != depth.texture.width() || gbuffer.size.y != depth.texture.height() {
            return Ok(());
        }

        let layout_0 = pipeline_cache.get_bind_group_layout(&pipeline_res.layout_0);
        let layout_1 = pipeline_cache.get_bind_group_layout(&pipeline_res.layout_1);
        let layout_2 = pipeline_cache.get_bind_group_layout(&pipeline_res.layout_2);

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
        let bind_group_1 = atlas_bind_group_1(device, &layout_1, gpu_atlas, "sdf_bind_group_1");

        // Bind group 2: cone-prepass seed texture (per-tile start distance).
        let prepass = world.resource::<cone::SdfConePrepass>();
        let bind_group_2 = device.create_bind_group(
            "sdf_bind_group_2",
            &layout_2,
            &BindGroupEntries::sequential((&prepass.read_view,)),
        );

        // Render into the three G-buffer MRT targets + the shared depth attachment. Clear the
        // colour targets (a miss writes the sky sentinel anyway, but a clean clear avoids stale
        // data leaking where the fullscreen triangle doesn't cover). Depth keeps Load so the SDF
        // surface shares the buffer with prior opaque geometry.
        // Per-pass GPU timing (no-op unless RenderDiagnosticsPlugin is present — editor builds).
        // Obtained before begin_tracked_render_pass (which mut-borrows render_context); the recorder
        // is owned, so it coexists with the pass borrow below. Records render/sdf_gbuffer_pass/*.
        let diagnostics = render_context.diagnostic_recorder();
        let clear = LoadOp::Clear(LinearRgba::NONE.into());
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("sdf_gbuffer_pass"),
            color_attachments: &[
                Some(RenderPassColorAttachment {
                    view: albedo_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: Operations { load: clear, store: StoreOp::Store },
                }),
                Some(RenderPassColorAttachment {
                    view: normal_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: Operations { load: clear, store: StoreOp::Store },
                }),
                Some(RenderPassColorAttachment {
                    view: emissive_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: Operations { load: clear, store: StoreOp::Store },
                }),
            ],
            depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        let span = diagnostics.pass_span(&mut render_pass, "sdf_gbuffer_pass");
        if let Some(pipeline) = pipeline {
            render_pass.set_render_pipeline(pipeline);
            render_pass.set_bind_group(0, &bind_group_0, &[0]);
            render_pass.set_bind_group(1, &bind_group_1, &[]);
            render_pass.set_bind_group(2, &bind_group_2, &[]);
            render_pass.draw(0..3, 0..1);
        }
        span.end(&mut render_pass);

        Ok(())
    }
}

/// Combine pass: the final deferred-lighting step — evaluates the analytic sun + emissive from the
/// G-buffer and writes the lit result into the HDR view target.
#[derive(Default)]
struct SdfCombineNode;

impl ViewNode for SdfCombineNode {
    type ViewQuery = &'static ViewTarget;

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        view_target: QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.view_entity();
        if world.get::<SdfCameraData>(view_entity).is_none() {
            return Ok(());
        }
        if let Some(enabled) = world.get_resource::<SdfRenderEnabled>()
            && !enabled.0
        {
            return Ok(());
        }

        let combine = world.resource::<SdfCombinePipeline>();
        let sdf = world.resource::<SdfPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let device = render_context.render_device();

        let Some(pipeline) = pipeline_cache.get_render_pipeline(combine.pipeline_id) else {
            use std::sync::atomic::{AtomicBool, Ordering};
            static LOGGED: AtomicBool = AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::Relaxed)
                && let bevy::render::render_resource::CachedPipelineState::Err(err) =
                    pipeline_cache.get_render_pipeline_state(combine.pipeline_id)
            {
                bevy::log::error!("SDF combine pipeline error: {err}");
            }
            return Ok(());
        };

        let gbuffer = world.resource::<SdfGBuffer>();
        let (Some(albedo_view), Some(normal_view), Some(emissive_view), Some(sampler)) = (
            &gbuffer.albedo_view,
            &gbuffer.normal_mat_view,
            &gbuffer.emissive_view,
            &gbuffer.sampler,
        ) else {
            return Ok(());
        };

        let Some(camera_uniforms) = world.get_resource::<ComponentUniforms<SdfCameraData>>() else {
            return Ok(());
        };
        let Some(camera_binding) = camera_uniforms.binding() else {
            return Ok(());
        };
        let layout_0 = pipeline_cache.get_bind_group_layout(&sdf.layout_0);
        let layout = pipeline_cache.get_bind_group_layout(&combine.layout);

        let bind_group_0 = device.create_bind_group(
            "sdf_combine_bind_group_0",
            &layout_0,
            &BindGroupEntries::sequential((camera_binding.clone(),)),
        );
        let bind_group_1 = device.create_bind_group(
            "sdf_combine_gbuffer",
            &layout,
            &BindGroupEntries::sequential((albedo_view, normal_view, emissive_view, sampler)),
        );

        let diagnostics = render_context.diagnostic_recorder();
        let post_process = view_target.post_process_write();
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("sdf_combine_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination,
                resolve_target: None,
                depth_slice: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        let span = diagnostics.pass_span(&mut render_pass, "sdf_combine_pass");
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group_0, &[0]);
        render_pass.set_bind_group(1, &bind_group_1, &[]);
        render_pass.draw(0..3, 0..1);
        span.end(&mut render_pass);

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

/// The `#define_import_path` module files the entry shader composes, in dependency order. The single
/// source of truth for the SDF import graph — `tests/shader_validation.rs` composes the SAME list
/// (prefixing `assets/`) so a new `sdf/*.wgsl` module can't be added to the pipeline without the
/// validation rig also seeing it. Paths are asset-server-relative (the `assets/` root is implicit).
pub const SDF_SHADER_MODULES: [&str; 9] = [
    "shaders/sdf/bindings.wgsl",
    "shaders/sdf/brick.wgsl",
    "shaders/sdf/material.wgsl",
    // march must register before shadows: `soft_shadow` now imports `sdf::march::lod_crossfade`
    // so the shadow ray samples the SAME LOD-blended field the primary march renders.
    "shaders/sdf/march.wgsl",
    "shaders/sdf/shadows.wgsl",
    "shaders/sdf/sky.wgsl",
    "shaders/sdf/pbr.wgsl",
    "shaders/sdf/oct.wgsl",
    "shaders/sdf/brdf.wgsl",
];

pub struct SdfRenderPlugin;

impl Plugin for SdfRenderPlugin {
    fn build(&self, app: &mut App) {
        // Load shader asset in main world so it's available for extraction
        let asset_server = app.world().resource::<AssetServer>();
        let shader_handle = asset_server.load(SDF_SHADER_PATH);
        let cone_shader_handle: Handle<Shader> = asset_server.load(cone::SDF_CONE_SHADER_PATH);
        let bake_shader_handle: Handle<Shader> = asset_server.load(bake::SDF_BAKE_SHADER_PATH);
        let combine_shader_handle: Handle<Shader> = asset_server.load(SDF_COMBINE_SHADER_PATH);
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
                    .after(super::editor_camera::orbit_camera)
                    // Run after the bake scheduling so the camera uniform reflects this frame's
                    // post-bake state. (The shader's chunk-search bound no longer comes from
                    // this uniform — it reads `arrayLength(&chunk_buf)` — so this ordering is
                    // for tidiness, not the old bound/table consistency requirement.)
                    .after(super::bake_scheduler::schedule_bakes),
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

        // Pass shader handles directly to render app (RenderStartup runs before Extract)
        render_app.insert_resource(SdfShaderHandle(shader_handle));
        render_app.insert_resource(cone::SdfConeShaderHandle(cone_shader_handle));
        render_app.insert_resource(bake::SdfBakeShaderHandle(bake_shader_handle));
        render_app.insert_resource(SdfCombineShaderHandle(combine_shader_handle));
        render_app.init_resource::<SdfGBuffer>();

        render_app
            .add_systems(ExtractSchedule, atlas_upload::extract_sdf_atlas)
            .add_systems(ExtractSchedule, extract_sdf_materials)
            .add_systems(ExtractSchedule, pbr_textures::extract_texture_library)
            .add_systems(ExtractSchedule, extract_shader_defs)
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
            .add_systems(Render, rebuild_pipeline_on_def_change)
            .add_systems(Render, prepare_sdf_gbuffer)
            .add_systems(RenderStartup, init_sdf_pipeline)
            .add_systems(RenderStartup, cone::init_cone_pipeline.after(init_sdf_pipeline))
            .add_systems(RenderStartup, bake::init_bake_pipeline.after(init_sdf_pipeline))
            .add_systems(RenderStartup, init_combine_pipeline.after(init_sdf_pipeline))
            .add_render_graph_node::<bake::SdfBrickBakeNode>(Core3d, bake::SdfBrickBakeLabel)
            .add_render_graph_node::<ViewNodeRunner<cone::SdfConeNode>>(Core3d, cone::SdfConeLabel)
            .add_render_graph_node::<ViewNodeRunner<SdfGBufferNode>>(Core3d, SdfGBufferLabel)
            .add_render_graph_node::<ViewNodeRunner<SdfCombineNode>>(Core3d, SdfCombineLabel)
            // The brick-bake compute pass writes the atlas BEFORE the view passes read it,
            // so it runs first (after the opaque pass, before the cone prepass + G-buffer).
            // It's a standalone (non-view) node — it fills the shared atlas once per frame,
            // not per view.
            //
            // Order: opaque → bake → cone prepass → G-buffer → combine(sun+emissive) → transparent.
            // The combine pass evaluates the analytic sun + emissive from the G-buffer (and fills
            // the sky on a miss) and writes the view target. It runs BEFORE the transparent pass so
            // gizmos (Transparent3d, negative depth_bias) draw on top. (Indirect GI is being
            // replaced by a world-anchored probe volume — see plans/sdf-ddgi-probe-volume.md.)
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::MainOpaquePass,
                    bake::SdfBrickBakeLabel,
                    cone::SdfConeLabel,
                    SdfGBufferLabel,
                    SdfCombineLabel,
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
    config: Res<SdfGridConfig>,
    raymarch: Res<super::SdfRaymarchParams>,
    registry: Res<super::edits::MaterialRegistry>,
    // The active scene's key light, read directly here (this system runs in the MAIN
    // world, so the light entity is available). Filtered to `SceneEntity` so the editor's
    // offscreen thumbnail / preview rig lights are excluded.
    sun_light: Query<(&GlobalTransform, &DirectionalLight), With<crate::scene_manager::SceneEntity>>,
    mut material_table: ResMut<SdfMaterialTable>,
    // Per-camera last-frame `clip_from_world`, for SSR reprojection. Persists across frames in
    // the main world via Local; seeded to this frame's matrix on the first sighting (so frame 0
    // reprojects to itself — harmless, the history buffer is also invalid that frame).
    mut prev_clip: Local<bevy::platform::collections::HashMap<Entity, Mat4>>,
) {
    let sun = sun_light
        .iter()
        .next()
        .map(|(xf, light)| {
            let forward = xf.rotation() * Vec3::NEG_Z;
            let c = light.color.to_linear();
            let intensity = (light.illuminance / 10_000.0).clamp(0.0, 8.0) * 3.0;
            (
                (-forward).normalize_or_zero(),
                Vec3::new(c.red, c.green, c.blue) * intensity,
            )
        })
        // Default sun (matches the old hardcoded constants) when the scene has no light.
        .unwrap_or((Vec3::new(0.5, 1.0, 0.3).normalize(), Vec3::splat(3.0)));
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
                metallic: def.metallic,
                roughness: def.roughness,
                parallax_scale: def.parallax_scale,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                emissive: def.emissive.extend(0.0),
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

        // Pixel cone half-width per unit ray distance: the world radius one pixel covers
        // at distance 1. For a perspective projection `proj.y_axis.y = cot(fov_y/2)`, the
        // full vertical world extent at distance 1 is `2·tan(fov_y/2)`, so one pixel spans
        // `2·tan(fov_y/2)/height` and its half-width (radius) is `tan(fov_y/2)/height`.
        // Scaled by `cone_scale` so the surface-within-a-pixel test can be tuned.
        let proj = camera.clip_from_view();
        let tan_half_fov_y = 1.0 / proj.y_axis.y.max(1e-6);
        let pixel_cone = (tan_half_fov_y / size.y.max(1) as f32) * raymarch.cone_scale;

        // Last frame's matrix (this frame's on first sighting); then stash this frame's for next.
        let prev_clip_from_world = *prev_clip.entry(entity).or_insert(clip_from_world);
        prev_clip.insert(entity, clip_from_world);

        commands.entity(entity).insert(SdfCameraData {
            inv_view_proj,
            clip_from_world,
            prev_clip_from_world,
            camera_pos: transform.translation.extend(0.0),
            // z = second-order grazing-step curvature `k` (sdf::march, SDF_SECOND_ORDER_STEP);
            // w unused (was surface_bias — iso-offset removed).
            screen_params: Vec4::new(
                size.x as f32,
                size.y as f32,
                raymarch.second_order_k,
                0.0,
            ),
            grid_origin: Vec4::new(
                config.world_origin().x,
                config.world_origin().y,
                config.world_origin().z,
                config.voxel_size,
            ),
            // Only `.z` (brick_size / samples-per-edge) is read by the shader. `.x`/`.y`/`.w`
            // are unused: the chunk-search bound is now `arrayLength(&chunk_buf)` in the shader
            // (not `.w`), which is consistent with the bound lookup buffer by construction — see
            // `find_chunk`. Kept as a vec4 for std140 alignment of the following fields.
            grid_dims: Vec4::new(0.0, 0.0, config.brick_size as f32, 0.0),
            // `w` carries `recenter_snap_chunks` so the shader can recompute the chunk-
            // snapped ring centre (the LOD cross-fade must key off the true resident-ring
            // boundary, which is hysteresis-snapped — see bake_scheduler::ring_chunk_origin).
            debug_params: Vec4::new(
                raymarch.max_steps as f32,
                raymarch.max_dist,
                raymarch.sdf_eps,
                config.recenter_snap_chunks as f32,
            ),
            // March tuning: the pixel cone half-width per unit ray distance drives the
            // screen-space termination (a surface within a pixel ends the march, so far
            // geometry resolves at coarse LOD); `y` is the soft-shadow penumbra hardness `k`
            // (lower = softer; blurs coarse-LOD faceting + the penumbra→umbra edge); `w` is the
            // LOD cross-fade band (fraction of each ring's half-extent; 0 = hard seams).
            march_params: Vec4::new(
                pixel_cone,
                raymarch.shadow_softness,
                raymarch.over_relax,
                raymarch.lod_blend_band,
            ),
            lod_params: Vec4::new(
                config.lod_count as f32,
                config.ring_bricks as f32,
                config.voxel_size,
                config.cell_stride() as f32,
            ),
            sun_dir: sun.0.extend(0.0),
            sun_color: sun.1.extend(0.0),
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
}

fn extract_shader_defs(defs: Extract<Res<SdfShaderDefs>>, mut commands: Commands) {
    commands.insert_resource(ExtractedShaderDefs {
        defs: defs.defs.clone(),
    });
}

#[allow(clippy::too_many_arguments)] // Bevy system params; rebuilds two def-gated pipelines.
fn rebuild_pipeline_on_def_change(
    mut pipeline: ResMut<SdfPipeline>,
    mut combine: ResMut<SdfCombinePipeline>,
    extracted: Option<Res<ExtractedShaderDefs>>,
    shader_handle: Res<SdfShaderHandle>,
    combine_shader: Res<SdfCombineShaderHandle>,
    pipeline_cache: Res<PipelineCache>,
    fullscreen_shader: Res<FullscreenShader>,
) {
    let Some(extracted) = extracted else { return };
    let shader_defs: Vec<_> = extracted.defs.iter().map(|s| s.as_str().into()).collect();
    let vertex_state = fullscreen_shader.to_vertex_state();
    // The combine pipeline reuses the camera layout (group 0). Clone it up front so the combine
    // rebuild below doesn't re-borrow `pipeline` (which is mutated by the primary rebuild).
    let camera_layout = pipeline.layout_0.clone();

    // Primary (G-buffer) pipeline. Rebuild whenever the extracted defs differ from what the live
    // pipeline was built with — timing-independent, so the startup case (defs sync a frame or two
    // after the pipeline was first queued with empty defs) rebuilds without a manual toggle.
    if extracted.defs != pipeline.current_defs {
        let new_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("sdf_gbuffer_pipeline".into()),
            layout: vec![
                pipeline.layout_0.clone(),
                pipeline.layout_1.clone(),
                pipeline.layout_2.clone(),
            ],
            vertex: vertex_state.clone(),
            fragment: Some(FragmentState {
                shader: shader_handle.0.clone(),
                shader_defs: shader_defs.clone(),
                targets: gbuffer_targets(),
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
        pipeline.current_defs = extracted.defs.clone();
    }

    // Combine pipeline (carries the SDF_DEBUG_* G-buffer/GI visualizer `#ifdef` branches).
    if extracted.defs != combine.current_defs {
        let new_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("sdf_combine_pipeline".into()),
            layout: vec![camera_layout, combine.layout.clone()],
            vertex: vertex_state,
            fragment: Some(FragmentState {
                shader: combine_shader.0.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: ViewTarget::TEXTURE_FORMAT_HDR,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        });
        combine.pipeline_id = new_id;
        combine.current_defs = extracted.defs.clone();
    }
}

/// The three G-buffer MRT colour-target states (albedo+dist, normal+material, emissive). All
/// `GBUFFER_FORMAT`, no blend (the fullscreen pass fully overwrites each covered pixel).
fn gbuffer_targets() -> Vec<Option<ColorTargetState>> {
    let one = || {
        Some(ColorTargetState {
            format: GBUFFER_FORMAT,
            blend: None,
            write_mask: ColorWrites::ALL,
        })
    };
    vec![one(), one(), one()]
}

/// (Re)allocate the three G-buffer textures to match the SDF view target's size. Runs each
/// frame; only recreates on a size change (or first run). The primary SDF pass renders into
/// these (RENDER_ATTACHMENT) and the deferred lit pass samples them (TEXTURE_BINDING).
fn prepare_sdf_gbuffer(
    device: Res<RenderDevice>,
    mut gbuffer: ResMut<SdfGBuffer>,
    views: Query<&ViewTarget, With<SdfCameraData>>,
) {
    // One SDF camera; take its target size. (Multiple SDF views would need per-view G-buffers —
    // not a case this editor hits.)
    let Some(view) = views.iter().next() else {
        return;
    };
    let size = view.main_texture().size();
    let dims = UVec2::new(size.width, size.height);

    if gbuffer.albedo.is_some() && gbuffer.size == dims {
        return; // already sized correctly
    }

    let make = |label: &str| {
        let tex = device.create_texture(&TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: GBUFFER_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&TextureViewDescriptor::default());
        (tex, view)
    };
    let (albedo, albedo_view) = make("sdf_gbuffer_albedo");
    let (normal_mat, normal_mat_view) = make("sdf_gbuffer_normal_mat");
    let (emissive, emissive_view) = make("sdf_gbuffer_emissive");

    if gbuffer.sampler.is_none() {
        gbuffer.sampler = Some(device.create_sampler(&SamplerDescriptor {
            label: Some("sdf_gbuffer_sampler"),
            // Nearest: one G-buffer texel per pixel; no interpolation of packed normals/distance.
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            ..default()
        }));
    }
    gbuffer.albedo = Some(albedo);
    gbuffer.albedo_view = Some(albedo_view);
    gbuffer.normal_mat = Some(normal_mat);
    gbuffer.normal_mat_view = Some(normal_mat_view);
    gbuffer.emissive = Some(emissive);
    gbuffer.emissive_view = Some(emissive_view);
    gbuffer.size = dims;
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
    // std430: each GpuSdfMaterial is 80 bytes (vec4 + f32 + 5×u32 + 3×f32 + 3×u32 pad +
    // vec4 emissive). The pads align `emissive` to its 16-byte boundary at offset 64.
    let mut bytes = Vec::with_capacity(extracted.materials.len() * 80);
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
        bytes.extend_from_slice(&m.metallic.to_le_bytes());
        bytes.extend_from_slice(&m.roughness.to_le_bytes());
        bytes.extend_from_slice(&m.parallax_scale.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // _pad0
        bytes.extend_from_slice(&0u32.to_le_bytes()); // _pad1
        bytes.extend_from_slice(&0u32.to_le_bytes()); // _pad2
        for c in [m.emissive.x, m.emissive.y, m.emissive.z, m.emissive.w] {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
    }
    let buffer = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_material_buffer"),
        contents: &bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    gpu_atlas.material_buffer = Some(buffer);
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
    // Visible to FRAGMENT (the raymarch pass) AND COMPUTE (the cone prepass reuses the
    // same camera + atlas bind groups), so both pipelines share one layout source.
    let vis = ShaderStages::FRAGMENT | ShaderStages::COMPUTE;
    let layout_0 = BindGroupLayoutDescriptor::new(
        "sdf_bind_group_0",
        &BindGroupLayoutEntries::sequential(
            vis,
            (
                // binding 0: per-view camera uniform (dynamic offset)
                uniform_buffer::<SdfCameraData>(true),
            ),
        ),
    );
    let layout_1 = BindGroupLayoutDescriptor::new(
        "sdf_bind_group_1",
        &BindGroupLayoutEntries::sequential(
            vis,
            (
                // binding 0: distance atlas — PAGED `binding_array` (R16Snorm pages, see atlas_pages)
                texture_2d(TextureSampleType::Float { filterable: true })
                    .count(core::num::NonZero::new(atlas_pages::ATLAS_MAX_PAGES).unwrap()),
                // binding 1: nearest sampler
                sampler(SamplerBindingType::Filtering),
                // binding 2: chunk lookup table (sorted, binary-searched)
                storage_buffer_read_only::<atlas_upload::GpuChunkLookup>(false),
                // binding 3: per-palette-slot distance atlas — PAGED `binding_array` (Rgba16Snorm pages)
                texture_2d(TextureSampleType::Float { filterable: false })
                    .count(core::num::NonZero::new(atlas_pages::ATLAS_MAX_PAGES).unwrap()),
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
                storage_buffer_read_only::<atlas_upload::GpuBrickTile>(false),
            ),
        ),
    );
    // group 2: cone-prepass seed texture (read in the fragment march as a per-tile start-t
    // via textureLoad — no sampler). R32Float, non-filterable.
    let layout_2 = BindGroupLayoutDescriptor::new(
        "sdf_bind_group_2",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (texture_2d(TextureSampleType::Float { filterable: false }),),
        ),
    );
    let shader = shader_handle.0.clone();
    let vertex_state = fullscreen_shader.to_vertex_state();

    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("sdf_gbuffer_pipeline".into()),
        layout: vec![layout_0.clone(), layout_1.clone(), layout_2.clone()],
        vertex: vertex_state,
        fragment: Some(FragmentState {
            shader: shader.clone(),
            shader_defs: vec![],
            targets: gbuffer_targets(),
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

    // The distance + material atlases are the paged pool (`AtlasPages`, created in the resource
    // below); its own 1×1 dummy pages keep the `binding_array`s valid before the first bake.
    let dummy_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("sdf_dummy_atlas_sampler"),
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        mipmap_filter: FilterMode::Nearest,
        ..default()
    });
    // The chunk directory (binding 2) + tile-run (binding 11) buffers are the ChunkTableBuffers pool
    // (created in the resource below); its 1-record dummies keep both bindings valid pre-bake.
    // One zeroed 80-byte GpuSdfMaterial row so binding 4 meets the struct's minimum
    // size before the real table uploads.
    let dummy_material = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_dummy_material"),
        contents: &[0u8; 80],
        usage: BufferUsages::STORAGE,
    });
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
        layout_2,
        shader_handle: shader,
        // Queued above with empty shader_defs; rebuild fires once the synced defs differ.
        current_defs: Vec::new(),
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

/// Queue the combine render pipeline (final deferred-lighting pass: analytic sun + emissive →
/// view target). Reuses `layout_0` (camera) + declares group 1 = 3 G-buffer textures + sampler.
/// Runs after `init_sdf_pipeline`.
fn init_combine_pipeline(
    mut commands: Commands,
    fullscreen_shader: Res<FullscreenShader>,
    combine_shader: Res<SdfCombineShaderHandle>,
    sdf_pipeline: Res<SdfPipeline>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "sdf_combine_gbuffer",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                // 0..2: G-buffer albedo, normal-material, emissive
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                // 3: non-filtering sampler
                sampler(SamplerBindingType::NonFiltering),
            ),
        ),
    );

    let vertex_state = fullscreen_shader.to_vertex_state();
    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("sdf_combine_pipeline".into()),
        layout: vec![sdf_pipeline.layout_0.clone(), layout.clone()],
        vertex: vertex_state,
        fragment: Some(FragmentState {
            shader: combine_shader.0.clone(),
            shader_defs: vec![],
            targets: vec![Some(ColorTargetState {
                format: ViewTarget::TEXTURE_FORMAT_HDR,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    });

    commands.insert_resource(SdfCombinePipeline {
        pipeline_id,
        layout,
        // Queued above with empty defs; rebuild fires once the synced defs differ.
        current_defs: Vec::new(),
    });
}
