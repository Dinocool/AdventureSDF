use bevy::core_pipeline::FullscreenShader;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
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

// --- GPU Types ---

/// One entry in the chunk lookup buffer (20 bytes, std430), sorted by `(key_hi,key_lo)`
/// and binary-searched by the shader. `key_*` = the absolute chunk key (see
/// `super::chunk`), independent of camera so CPU and GPU agree. `occ_*` = 64-bit
/// occupancy mask (bit i ⇒ local brick i resident); `tile_run_base` indexes the packed
/// `chunk_tile_table` where this chunk's `popcount(occ)` brick `atlas_base`s live.
///
/// Exists SOLELY as the std430 `ShaderType` for the chunk-lookup storage buffer's binding layout /
/// min-binding-size (see `init_*_pipeline`). The actual data flows as `chunk::ChunkLookup` and is
/// serialized by [`encode_lookup`]; this mirror is kept here, not on `chunk::ChunkLookup`, to
/// preserve `chunk.rs`'s render-free purity. Its fields MUST match `chunk::ChunkLookup` byte-for-byte.
#[derive(ShaderType, Clone, Copy, Default)]
struct GpuChunkLookup {
    key_hi: u32,
    key_lo: u32,
    occ_lo: u32,
    occ_hi: u32,
    tile_run_base: u32,
}

/// std430 `ShaderType` for the tile-run storage buffer's binding layout / min-binding-size only
/// (12 bytes: atlas tile origin `col_px | row_px<<16` + packed 4-entry palette). Like
/// [`GpuChunkLookup`], the data flows as `chunk::BrickTile` (serialized by [`encode_tile`]); this
/// mirror keeps the GPU derive out of the pure `chunk.rs`. Fields MUST match `chunk::BrickTile`.
#[derive(ShaderType, Clone, Copy, Default)]
struct GpuBrickTile {
    atlas_base: u32,
    pal01: u32,
    pal23: u32,
}

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

// --- Extracted Atlas ---

#[derive(Resource, Default)]
struct ExtractedSdfAtlas {
    /// FULL-rebuild payload (`full_rebuild`): the entire sorted chunk lookup table
    /// (`chunk_data`, one row per resident chunk) + the whole packed tile-run buffer
    /// (`tile_run_data`, capacity-sized, each chunk's 64-entry region at `slot*64`). Used the
    /// first frame, on a capacity grow, and on the empty-atlas sentinel. See `super::chunk`.
    chunk_data: Vec<super::chunk::ChunkLookup>,
    tile_run_data: Vec<super::chunk::BrickTile>,
    /// DELTA payload (`tables_dirty && !full_rebuild`): only the directory slots and tile-run
    /// regions (slot → 64-entry region) that changed this frame. The toroidal directory is fixed-
    /// position, so each is an in-place `write_buffer` — no row shift, no sentinel tail.
    chunk_row_updates: Vec<(u32, super::chunk::ChunkLookup)>,
    tile_run_updates: Vec<(u32, Vec<super::chunk::BrickTile>)>,
    /// Directory length (= `R³ × lod_count`); the GPU direct-indexes it (no logical-count bound).
    new_chunk_len: u32,
    /// Buffer capacities (rows / tile-run entries) this frame's table needs. `prepare` grows the
    /// buffers (with headroom) when these exceed the current allocation.
    chunk_cap_needed: u32,
    tile_cap_needed: u32,
    /// True ⇒ upload `chunk_data`/`tile_run_data` wholesale; false ⇒ apply the delta updates.
    full_rebuild: bool,
    /// Whether the chunk lookup / tile-run buffers changed at all this frame. False on a
    /// texel-only re-bake — the lookup buffers are reused as-is.
    tables_dirty: bool,
    texture_width: u32,
    texture_height: u32,
    /// Grow the atlas texture taller this frame: `prepare` recreates the dist+mat textures at
    /// the new height and `copy_texture_to_texture`s the old content in (the GPU owns the
    /// texels — there is no CPU upload), then the bake node fills the genuinely-new tiles. When
    /// false, `prepare` keeps the existing textures and the bake node patches tiles in place.
    realloc: bool,
    dirty: bool,
}

/// Render-world memo of the last atlas generation uploaded, so `extract_sdf_atlas`
/// only flags `dirty` (and `prepare_sdf_atlas_gpu` only re-uploads) when the
/// main-world bake actually changed something. Without this the atlas was rebuilt
/// every frame.
#[derive(Resource, Default)]
struct LastAtlasGen(u64);

/// Render-world record of how many tile rows the persistent atlas texture currently
/// spans. `extract_sdf_atlas` reads it to decide grow-vs-partial-upload; the texture
/// only grows (never shrinks except on a full rebuild), so a tile origin assigned
/// once stays valid until the next full bake.
#[derive(Resource, Default)]
struct AtlasCapacity {
    rows: u32,
}

/// Render-world record of the allocated chunk-lookup + tile-run buffer capacities (in rows /
/// tile-run entries), so `prepare_sdf_atlas_gpu` knows when an incremental delta needs the
/// buffer grown (recreate larger + full re-upload) versus a plain in-place `write_buffer`.
/// Both buffers are over-sized with headroom on a rebuild so most frames stay in the cheap
/// delta path.
#[derive(Resource, Default)]
struct ChunkBufCapacity {
    chunk_rows: u32,
    tile_slots: u32,
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
    variants: Vec<crate::assets::MapSet>,
}

// --- Pipeline ---

// BISECT: minimal shader while building features back up after the division-free fix.
const SDF_SHADER_PATH: &str = "shaders/sdf_raymarch.wgsl";
/// Cone-prepass compute shader (per-tile seed-distance march).
const SDF_CONE_SHADER_PATH: &str = "shaders/sdf_cone_prepass.wgsl";
/// Brick-bake compute shader (per-voxel CSG eval → atlas tile buffers). The GPU half of the
/// hybrid bake; see `BakeBackend` and `bake_scheduler::emit_gpu_bakes`.
const SDF_BAKE_SHADER_PATH: &str = "shaders/sdf_brick_bake.wgsl";

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

/// Compute pipeline + layouts for the cone prepass (one tile-cone march per 8×8 tile,
/// writing per-tile seed distances). Reuses the SDF camera (layout_0) and atlas (layout_1)
/// bind groups; `layout_2` is the write-only storage texture.
#[derive(Resource)]
struct SdfConePipeline {
    pipeline_id: CachedComputePipelineId,
    layout_2: BindGroupLayoutDescriptor,
}

/// Per-tile seed-distance texture written by the cone prepass and read by the fragment
/// march. R32Float, one texel per 8×8 screen tile. Sized for 4K (480×270 tiles); the
/// compute shader and fragment both bounds-check against the actual viewport.
#[derive(Resource)]
struct SdfConePrepass {
    /// Storage-write view for the compute pass.
    storage_view: TextureView,
    /// Sampled (textureLoad) view for the fragment pass — same texture.
    read_view: TextureView,
}

/// Screen-tile edge in pixels. MUST match `TILE` in sdf_cone_prepass.wgsl and the divisor
/// the fragment pass uses to index the seed texture.
const CONE_TILE: u32 = 8;
/// Seed-texture capacity in tiles (covers 4K: ceil(3840/8) × ceil(2160/8)).
const CONE_TEX_TILES_X: u32 = 480;
const CONE_TEX_TILES_Y: u32 = 270;

#[derive(Resource)]
struct SdfConeShaderHandle(Handle<Shader>);

#[derive(Resource)]
struct SdfBakeShaderHandle(Handle<Shader>);

// --- GPU brick bake (compute) ---

/// Width of the 2D bake dispatch grid in workgroups. The compute dispatch uses one workgroup
/// per brick job; a single dimension caps at 65535 (wgpu/Vulkan limit), which a large edit can
/// blow past (a big sphere dirties 70k+ bricks). So we lay the jobs out in a 2D grid of this
/// width and reconstruct the linear job index in the shader as `wg.y * DISPATCH_WIDTH + wg.x`.
/// 256² = 65536 jobs per "page"; the Y extent then carries the rest, well under the limit.
/// Must match `DISPATCH_WIDTH` in sdf_brick_bake.wgsl.
const BAKE_DISPATCH_WIDTH: u32 = 256;

/// u32s per distance tile in the bake output buffer. Each tile is 64×8 R16 texels = 512
/// texels = 256 u32 (two R16 packed per u32), but rows are padded to 64 u32 so each tile row
/// is 256 bytes — `copy_buffer_to_texture` requires `bytes_per_row` to be a multiple of 256.
/// 64 u32/row × 8 rows = 512 u32 per tile (32 real + 32 pad per row). Must match the bake
/// shader's `DIST_ROW_U32`/`DIST_TILE_U32`.
const BAKE_DIST_ROW_U32: u32 = 64;
const BAKE_DIST_TILE_U32: u32 = BAKE_DIST_ROW_U32 * 8;
/// u32s per material tile: 64×8 Rgba16 texels, 2 u32 per texel, 128 u32/row × 8 = 1024.
/// Row stride = 128 u32 = 512 bytes (already a multiple of 256). Matches `MAT_TILE_U32`.
const BAKE_MAT_ROW_U32: u32 = 128;
const BAKE_MAT_TILE_U32: u32 = BAKE_MAT_ROW_U32 * 8;

/// One brick bake job's header, std430. Mirror of the WGSL `JobHeader` in
/// `sdf_brick_bake.wgsl` and built from `bake_scheduler::GpuBakeJob`.
#[derive(ShaderType, Clone, Copy, Default)]
struct GpuJobHeader {
    coord: IVec3,
    voxel_size: f32,
    dist_band: f32,
    edit_start: u32,
    edit_count: u32,
    pal01: u32,
    pal23: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Render-world copy of this frame's GPU bake jobs (extracted from
/// `bake_scheduler::PendingGpuBakes`). `tiles` parallels `headers`: job i writes the atlas
/// tile `tiles[i]`. Empty on frames with no bake work (the node early-outs).
#[derive(Resource, Default)]
struct ExtractedBrickBakes {
    headers: Vec<GpuJobHeader>,
    edits: Vec<super::edits::GpuEdit>,
    /// Destination atlas tile index per job (drives the `copy_buffer_to_texture` origin).
    tiles: Vec<u32>,
}

/// The bake compute pipeline + the storage buffers the dispatch writes (sized to the job
/// count each frame). The buffers are re-created when the job count grows; the per-tile
/// `copy_buffer_to_texture` into the persistent atlas happens in the bake node.
#[derive(Resource)]
struct SdfBakePipeline {
    pipeline_id: CachedComputePipelineId,
    layout: BindGroupLayoutDescriptor,
}

#[derive(Resource, Default)]
struct SdfBakeBuffers {
    header_buffer: Option<Buffer>,
    edit_buffer: Option<Buffer>,
    dist_buffer: Option<Buffer>,
    mat_buffer: Option<Buffer>,
    /// Number of jobs prepared this frame (workgroup dispatch count + copy loop bound).
    job_count: u32,
    /// Destination atlas tiles for this frame's jobs (parallels the dispatch order).
    tiles: Vec<u32>,
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

        // Bind group 2: cone-prepass seed texture (per-tile start distance).
        let prepass = world.resource::<SdfConePrepass>();
        let bind_group_2 = device.create_bind_group(
            "sdf_bind_group_2",
            &layout_2,
            &BindGroupEntries::sequential((&prepass.read_view,)),
        );

        // Render into the three G-buffer MRT targets + the shared depth attachment. Clear the
        // colour targets (a miss writes the sky sentinel anyway, but a clean clear avoids stale
        // data leaking where the fullscreen triangle doesn't cover). Depth keeps Load so the SDF
        // surface shares the buffer with prior opaque geometry.
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

        if let Some(pipeline) = pipeline {
            render_pass.set_render_pipeline(pipeline);
            render_pass.set_bind_group(0, &bind_group_0, &[0]);
            render_pass.set_bind_group(1, &bind_group_1, &[]);
            render_pass.set_bind_group(2, &bind_group_2, &[]);
            render_pass.draw(0..3, 0..1);
        }

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
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group_0, &[0]);
        render_pass.set_bind_group(1, &bind_group_1, &[]);
        render_pass.draw(0..3, 0..1);

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
const SDF_SHADER_MODULES: [&str; 9] = [
    "shaders/sdf/bindings.wgsl",
    "shaders/sdf/brick.wgsl",
    "shaders/sdf/material.wgsl",
    "shaders/sdf/shadows.wgsl",
    "shaders/sdf/sky.wgsl",
    "shaders/sdf/pbr.wgsl",
    "shaders/sdf/oct.wgsl",
    "shaders/sdf/march.wgsl",
    "shaders/sdf/brdf.wgsl",
];

pub struct SdfRenderPlugin;

impl Plugin for SdfRenderPlugin {
    fn build(&self, app: &mut App) {
        // Load shader asset in main world so it's available for extraction
        let asset_server = app.world().resource::<AssetServer>();
        let shader_handle = asset_server.load(SDF_SHADER_PATH);
        let cone_shader_handle: Handle<Shader> = asset_server.load(SDF_CONE_SHADER_PATH);
        let bake_shader_handle: Handle<Shader> = asset_server.load(SDF_BAKE_SHADER_PATH);
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
                    .after(super::orbit_camera)
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
        render_app.insert_resource(SdfConeShaderHandle(cone_shader_handle));
        render_app.insert_resource(SdfBakeShaderHandle(bake_shader_handle));
        render_app.insert_resource(SdfCombineShaderHandle(combine_shader_handle));
        render_app.init_resource::<SdfGBuffer>();

        render_app
            .add_systems(ExtractSchedule, extract_sdf_atlas)
            .add_systems(ExtractSchedule, extract_sdf_materials)
            .add_systems(ExtractSchedule, extract_texture_library)
            .add_systems(ExtractSchedule, extract_shader_defs)
            .add_systems(ExtractSchedule, extract_brick_bakes)
            .init_resource::<TextureStreamState>()
            .init_resource::<LastAtlasGen>()
            .init_resource::<AtlasCapacity>()
            .init_resource::<ChunkBufCapacity>()
            .init_resource::<SdfBakeBuffers>()
            .add_systems(Render, prepare_brick_bake_buffers.before(prepare_sdf_atlas_gpu))
            .add_systems(Render, prepare_sdf_atlas_gpu)
            .add_systems(Render, prepare_sdf_materials_gpu)
            .add_systems(Render, init_texture_streaming)
            .add_systems(Render, upload_texture_layers.after(init_texture_streaming))
            .add_systems(Render, rebuild_pipeline_on_def_change)
            .add_systems(Render, prepare_sdf_gbuffer)
            .add_systems(RenderStartup, init_sdf_pipeline)
            .add_systems(RenderStartup, init_cone_pipeline.after(init_sdf_pipeline))
            .add_systems(RenderStartup, init_bake_pipeline.after(init_sdf_pipeline))
            .add_systems(RenderStartup, init_combine_pipeline.after(init_sdf_pipeline))
            .add_render_graph_node::<SdfBrickBakeNode>(Core3d, SdfBrickBakeLabel)
            .add_render_graph_node::<ViewNodeRunner<SdfConeNode>>(Core3d, SdfConeLabel)
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
                    SdfBrickBakeLabel,
                    SdfConeLabel,
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
            // zw unused (was surface_bias — iso-offset removed).
            screen_params: Vec4::new(size.x as f32, size.y as f32, 0.0, 0.0),
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
            // geometry resolves at coarse LOD); `y` is reserved (was the removed cubic band);
            // `w` is the LOD cross-fade band (fraction of each ring's half-extent; 0 = hard
            // seams).
            march_params: Vec4::new(
                pixel_cone,
                0.0,
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

// --- Extract: Pack Atlas for GPU ---

fn extract_sdf_atlas(
    atlas: Extract<Res<SdfAtlas>>,
    mut last_gen: ResMut<LastAtlasGen>,
    mut capacity: ResMut<AtlasCapacity>,
    mut chunk_cap: ResMut<ChunkBufCapacity>,
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

    let live = &atlas.live_chunks;
    let num_bricks = atlas.bricks.len() as u32;
    if num_bricks == 0 {
        // Fully evicted (roamed into empty space). Signal a full rebuild with EMPTY chunk data
        // so `prepare_sdf_atlas_gpu` replaces the lookup buffer with a miss-only sentinel. The
        // shader bounds its search by `arrayLength(&chunk_buf)`, so leaving the old buffer bound
        // would search stale entries and render ghost geometry. Reset capacity so the next
        // re-entry triggers a fresh full rebuild rather than a delta against a dropped buffer.
        chunk_cap.chunk_rows = 0;
        chunk_cap.tile_slots = 0;
        commands.insert_resource(ExtractedSdfAtlas {
            tables_dirty: true,
            full_rebuild: true,
            dirty: true,
            ..Default::default()
        });
        return;
    }

    let edge = BRICK_EDGE as u32;
    let tile_width = edge * edge; // 64
    let texture_width = ATLAS_TILES_PER_ROW * tile_width;

    // Tile origins come from the stable allocator (its high-water mark), NOT brick
    // iteration order — so a re-baked brick keeps its sub-rect across frames.
    let required_rows = atlas.tiles.high_water().div_ceil(ATLAS_TILES_PER_ROW).max(1);
    let texture_height = required_rows * edge;

    // Realloc when the atlas TEXTURE must grow taller (the GPU bake never shrinks it). This is now
    // INDEPENDENT of the chunk table: a brick's `atlas_base` is derived from its stable tile index
    // and is unaffected by the texture's height, so a texture grow never forces a table rebuild.
    let realloc = required_rows > capacity.rows;
    if realloc {
        capacity.rows = required_rows;
    }

    let mut extracted = ExtractedSdfAtlas {
        texture_width,
        texture_height,
        realloc,
        dirty: true,
        ..Default::default()
    };

    // Full-rebuild-vs-delta + the tile-run headroom policy live on `LiveChunkTables::upload` (the
    // SINGLE source of truth, mirrored by the churn + recenter-lifecycle differential tests). Extract
    // passes its render-world buffer capacities in and maps the returned NATIVE records onto the GPU
    // mirror; the directory is fixed-size so a Full only happens on first upload / a tile-run grow.
    match live.upload(chunk_cap.chunk_rows, chunk_cap.tile_slots) {
        super::chunk::ChunkUpload::Full { rows, tile_run, cap_rows, cap_slots } => {
            chunk_cap.chunk_rows = cap_rows;
            chunk_cap.tile_slots = cap_slots;
            extracted.chunk_data = rows;
            extracted.tile_run_data = tile_run;
            extracted.new_chunk_len = cap_rows;
            extracted.chunk_cap_needed = cap_rows;
            extracted.tile_cap_needed = cap_slots;
            extracted.full_rebuild = true;
            extracted.tables_dirty = true;
        }
        super::chunk::ChunkUpload::Delta { row_updates, region_updates } => {
            // Fixed-position directory → every dirty entry is an in-place index→value write.
            extracted.chunk_row_updates = row_updates;
            extracted.tile_run_updates =
                region_updates.into_iter().map(|(s, reg)| (s, reg.to_vec())).collect();
            extracted.tables_dirty =
                !extracted.chunk_row_updates.is_empty() || !extracted.tile_run_updates.is_empty();
            extracted.new_chunk_len = live.row_count();
        }
    }

    commands.insert_resource(extracted);
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
        let map_set = extracted.variants[layer as usize].clone();
        stream.tasks.push(pool.spawn(async move {
            let maps = super::textures::encode_mapset_bc7(&map_set);
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

// --- Prepare: chunk-table upload (full rebuild + incremental delta) ---

/// 20-byte std430 encoding of one chunk lookup row.
fn encode_lookup(c: &super::chunk::ChunkLookup, out: &mut Vec<u8>) {
    out.extend_from_slice(&c.key_hi.to_le_bytes());
    out.extend_from_slice(&c.key_lo.to_le_bytes());
    out.extend_from_slice(&c.occ_lo.to_le_bytes());
    out.extend_from_slice(&c.occ_hi.to_le_bytes());
    out.extend_from_slice(&c.tile_run_base.to_le_bytes());
}

/// 12-byte std430 encoding of one tile-run brick record.
fn encode_tile(b: &super::chunk::BrickTile, out: &mut Vec<u8>) {
    out.extend_from_slice(&b.atlas_base.to_le_bytes());
    out.extend_from_slice(&b.pal01.to_le_bytes());
    out.extend_from_slice(&b.pal23.to_le_bytes());
}

/// The 20-byte `(u32::MAX, u32::MAX, 0, 0, 0)` chunk-lookup sentinel. Its key sorts after every
/// real chunk key, so binary search over the fixed physical buffer never matches a tail slot.
fn sentinel_row_bytes() -> [u8; 20] {
    let mut b = [0u8; 20];
    b[0..4].copy_from_slice(&u32::MAX.to_le_bytes());
    b[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    b
}

/// Full (re)allocation + upload of both chunk-table buffers, sized to CAPACITY (with headroom)
/// so later frames can `write_buffer` deltas in place. The chunk-lookup buffer is filled with
/// `new_chunk_len` live rows followed by sentinel rows to capacity; the tile-run buffer is the
/// capacity-sized `tile_run_data` (each live slot's region at `slot*64`, gaps zero). Used on the
/// first upload, a capacity grow, and the empty-atlas case (zero live rows → all sentinel).
fn upload_tables_full(
    device: &RenderDevice,
    gpu_atlas: &mut SdfGpuAtlas,
    extracted: &ExtractedSdfAtlas,
) {
    // Chunk lookup buffer: live rows then sentinel tail to capacity. Capacity is always ≥1 so the
    // storage buffer is never zero-sized (an empty atlas yields a single sentinel — the prior
    // dedicated empty path, now folded in here).
    let cap_rows = extracted.chunk_cap_needed.max(1);
    let live = extracted.new_chunk_len.min(extracted.chunk_data.len() as u32);
    let mut chunk_bytes = Vec::with_capacity(cap_rows as usize * 20);
    for c in extracted.chunk_data.iter().take(live as usize) {
        encode_lookup(c, &mut chunk_bytes);
    }
    let sentinel = sentinel_row_bytes();
    for _ in live..cap_rows {
        chunk_bytes.extend_from_slice(&sentinel);
    }
    gpu_atlas.lookup_buffer = Some(device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_chunk_lookup_buffer"),
        contents: &chunk_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    }));

    // Tile-run buffer: capacity-sized (extract already laid out `tile_run_data` to the slot
    // high-water; pad to `tile_cap_needed` so deltas into freshly-grown slots have room).
    let cap_slots = extracted.tile_cap_needed.max(super::chunk::TILE_RUN_SLOT) as usize;
    let mut tile_bytes = Vec::with_capacity(cap_slots * 12);
    for b in &extracted.tile_run_data {
        encode_tile(b, &mut tile_bytes);
    }
    tile_bytes.resize(cap_slots * 12, 0);
    gpu_atlas.chunk_tile_buffer = Some(device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_chunk_tile_buffer"),
        contents: &tile_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    }));
}

/// Incremental upload: `write_buffer` only the chunk rows + tile-run regions that changed this
/// frame, plus sentinel-blank the rows a removed chunk vacated. The buffers keep their (capacity)
/// allocation — only the changed byte ranges are touched, so a coarse-LOD snap pages the handful
/// of dirty chunks instead of recreating the whole ~1 MB table.
fn upload_tables_delta(
    queue: &RenderQueue,
    gpu_atlas: &SdfGpuAtlas,
    extracted: &ExtractedSdfAtlas,
) {
    let (Some(lookup), Some(tiles)) = (&gpu_atlas.lookup_buffer, &gpu_atlas.chunk_tile_buffer)
    else {
        return; // no buffers yet (shouldn't happen — first frame is a full rebuild)
    };

    // Changed chunk-lookup rows (20 B each, at row*20). A structural change marks a contiguous
    // suffix `[R..end)` dirty (every row at/after an insert/remove shifts), so coalesce consecutive
    // rows into one `write_buffer` to avoid a long burst of 20-byte writes on a snap frame.
    let mut run_start: Option<u32> = None;
    let mut run_bytes: Vec<u8> = Vec::new();
    let flush = |start: u32, bytes: &[u8]| {
        if !bytes.is_empty() {
            queue.write_buffer(lookup, (start as u64) * 20, bytes);
        }
    };
    for (row, c) in &extracted.chunk_row_updates {
        match run_start {
            Some(s) if *row == s + (run_bytes.len() as u32 / 20) => {}
            _ => {
                if let Some(s) = run_start {
                    flush(s, &run_bytes);
                }
                run_start = Some(*row);
                run_bytes.clear();
            }
        }
        encode_lookup(c, &mut run_bytes);
    }
    if let Some(s) = run_start {
        flush(s, &run_bytes);
    }
    // No sentinel tail: the directory is fixed-size and an emptied chunk's slot was already reset to
    // the sentinel tag in `clear_brick` (it shows up as a normal dirty-row write above).

    // Changed tile-run regions (64 entries × 12 B = 768 B each, at slot*64*12).
    let mut region_bytes = Vec::with_capacity(super::chunk::TILE_RUN_SLOT as usize * 12);
    for (slot, region) in &extracted.tile_run_updates {
        region_bytes.clear();
        for b in region {
            encode_tile(b, &mut region_bytes);
        }
        let base = (*slot as u64) * super::chunk::TILE_RUN_SLOT as u64 * 12;
        queue.write_buffer(tiles, base, &region_bytes);
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

    if extracted.tables_dirty {
        if extracted.full_rebuild {
            upload_tables_full(&device, &mut gpu_atlas, &extracted);
        } else {
            upload_tables_delta(&queue, &gpu_atlas, &extracted);
        }
    }

    if extracted.realloc {
        // Grow the atlas taller. The GPU owns the texels (the CPU has only palette-only
        // placeholders), so create EMPTY textures and copy any prior content into the taller
        // replacement; the bake node fills the genuinely-new tiles this same frame. On the
        // very first bake there's no prior texture — just the empty allocation, no copy. All
        // atlas textures carry COPY_SRC (for this grow copy) + COPY_DST (the bake node's
        // per-tile copy_buffer_to_texture).
        let usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::COPY_SRC;
        let size = Extent3d {
            width: extracted.texture_width,
            height: extracted.texture_height,
            depth_or_array_layers: 1,
        };
        let dist_tex = device.create_texture(&TextureDescriptor {
            label: Some("sdf_dist_atlas"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R16Snorm,
            usage,
            view_formats: &[],
        });
        let mat_tex = device.create_texture(&TextureDescriptor {
            label: Some("sdf_mat_atlas"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Snorm,
            usage,
            view_formats: &[],
        });
        // Copy prior content (full width, old height) into the new taller textures, if any.
        if let (Some(old_dist), Some(old_mat)) = (&gpu_atlas.dist_tex, &gpu_atlas.mat_tex) {
            let old_h = old_dist.height().min(extracted.texture_height);
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("sdf_atlas_grow_copy"),
            });
            let copy_extent = Extent3d {
                width: extracted.texture_width,
                height: old_h,
                depth_or_array_layers: 1,
            };
            for (src, dst) in [(old_dist, &dist_tex), (old_mat, &mat_tex)] {
                encoder.copy_texture_to_texture(
                    TexelCopyTextureInfo {
                        texture: src,
                        mip_level: 0,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    TexelCopyTextureInfo {
                        texture: dst,
                        mip_level: 0,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    copy_extent,
                );
            }
            queue.submit([encoder.finish()]);
        }

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
    }
    // Non-grow frames: the existing textures are kept; the bake node patches changed tiles in
    // place via copy_buffer_to_texture. Nothing to upload here.
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
    // One zeroed 80-byte GpuSdfMaterial row so binding 4 meets the struct's minimum
    // size before the real table uploads.
    let dummy_material = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_dummy_material"),
        contents: &[0u8; 80],
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
        layout_2,
        shader_handle: shader,
        // Queued above with empty shader_defs; rebuild fires once the synced defs differ.
        current_defs: Vec::new(),
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

/// Allocate the per-tile seed texture (storage-write + sampled views) and queue the cone-
/// prepass compute pipeline. Runs after `init_sdf_pipeline` so the shared camera/atlas
/// layouts (layout_0/1) already exist on `SdfPipeline`.
fn init_cone_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    sdf_pipeline: Res<SdfPipeline>,
    cone_shader: Res<SdfConeShaderHandle>,
) {
    // group 2 for the COMPUTE side: write-only R32Float storage texture (one texel/tile).
    let layout_2 = BindGroupLayoutDescriptor::new(
        "sdf_cone_bind_group_2",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (texture_storage_2d(
                TextureFormat::R32Float,
                StorageTextureAccess::WriteOnly,
            ),),
        ),
    );

    let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("sdf_cone_pipeline".into()),
        layout: vec![
            sdf_pipeline.layout_0.clone(),
            sdf_pipeline.layout_1.clone(),
            layout_2.clone(),
        ],
        shader: cone_shader.0.clone(),
        ..default()
    });

    // The seed texture: STORAGE_BINDING (compute writes) + TEXTURE_BINDING (fragment reads).
    let seed_tex = device.create_texture(&TextureDescriptor {
        label: Some("sdf_cone_seed"),
        size: Extent3d {
            width: CONE_TEX_TILES_X,
            height: CONE_TEX_TILES_Y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R32Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let storage_view = seed_tex.create_view(&TextureViewDescriptor::default());
    let read_view = seed_tex.create_view(&TextureViewDescriptor::default());

    commands.insert_resource(SdfConePipeline {
        pipeline_id,
        layout_2,
    });
    commands.insert_resource(SdfConePrepass {
        storage_view,
        read_view,
    });
}

// --- Render Graph: cone prepass node ---

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct SdfConeLabel;

#[derive(Default)]
struct SdfConeNode;

impl ViewNode for SdfConeNode {
    // Run on the SDF camera view; we only need to gate on the view entity (camera uniform)
    // and read the viewport size off the camera uniform itself.
    type ViewQuery = &'static ViewTarget;

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        _view_target: QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // Only run on SDF cameras.
        let view_entity = graph.view_entity();
        if world.get::<SdfCameraData>(view_entity).is_none() {
            return Ok(());
        }
        if let Some(enabled) = world.get_resource::<SdfRenderEnabled>()
            && !enabled.0
        {
            return Ok(());
        }

        let cone = world.resource::<SdfConePipeline>();
        let sdf = world.resource::<SdfPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let device = render_context.render_device();

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(cone.pipeline_id) else {
            return Ok(());
        };

        // The compute pass reuses the fragment camera + atlas bind groups, so it needs the
        // same camera uniform (with dynamic offset) and the atlas bind group 1.
        let Some(camera_uniforms) = world.get_resource::<ComponentUniforms<SdfCameraData>>()
        else {
            return Ok(());
        };
        let Some(camera_binding) = camera_uniforms.binding() else {
            return Ok(());
        };
        // Dynamic offset for this view's camera uniform.
        let Some(dyn_off) = world.get::<bevy::render::extract_component::DynamicUniformIndex<SdfCameraData>>(view_entity)
        else {
            return Ok(());
        };

        let layout_0 = pipeline_cache.get_bind_group_layout(&sdf.layout_0);
        let layout_1 = pipeline_cache.get_bind_group_layout(&sdf.layout_1);
        let layout_2 = pipeline_cache.get_bind_group_layout(&cone.layout_2);

        let bind_group_0 = device.create_bind_group(
            "sdf_cone_bind_group_0",
            &layout_0,
            &BindGroupEntries::sequential((camera_binding.clone(),)),
        );

        let gpu_atlas = world.resource::<SdfGpuAtlas>();
        let tex_views = gpu_atlas.tex_array_views.as_ref().unwrap();
        let bind_group_1 = device.create_bind_group(
            "sdf_cone_bind_group_1",
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

        let prepass = world.resource::<SdfConePrepass>();
        let bind_group_2 = device.create_bind_group(
            "sdf_cone_bind_group_2",
            &layout_2,
            &BindGroupEntries::sequential((&prepass.storage_view,)),
        );

        // Viewport in tiles → workgroup count (workgroup is 8×8 = one tile per invocation).
        let size = world
            .get::<SdfCameraData>(view_entity)
            .map(|c| UVec2::new(c.screen_params.x as u32, c.screen_params.y as u32))
            .unwrap_or(UVec2::new(1920, 1080));
        let tiles_x = size.x.div_ceil(CONE_TILE);
        let tiles_y = size.y.div_ceil(CONE_TILE);
        let wg_x = tiles_x.div_ceil(8);
        let wg_y = tiles_y.div_ceil(8);

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("sdf_cone_prepass"),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group_0, &[dyn_off.index()]);
        pass.set_bind_group(1, &bind_group_1, &[]);
        pass.set_bind_group(2, &bind_group_2, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);

        Ok(())
    }
}

// --- GPU brick bake: extract / prepare / pipeline / node ---

/// Extract this frame's GPU bake jobs from the main world into the render world, converting
/// each `GpuBakeJob` into its `GpuJobHeader`. The flat `GpuEdit` list is shared by all jobs
/// (each job's `edit_start..edit_start+edit_count` indexes it). Empty when not in GPU mode.
fn extract_brick_bakes(
    pending: Extract<Res<super::bake_scheduler::PendingGpuBakes>>,
    mut commands: Commands,
) {
    if pending.jobs.is_empty() {
        commands.insert_resource(ExtractedBrickBakes::default());
        return;
    }
    let mut headers = Vec::with_capacity(pending.jobs.len());
    let mut tiles = Vec::with_capacity(pending.jobs.len());
    for j in &pending.jobs {
        headers.push(GpuJobHeader {
            coord: j.coord,
            voxel_size: j.voxel_size,
            dist_band: j.dist_band,
            edit_start: j.edit_start,
            edit_count: j.edit_count,
            pal01: j.palette[0] as u32 | ((j.palette[1] as u32) << 16),
            pal23: j.palette[2] as u32 | ((j.palette[3] as u32) << 16),
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        });
        tiles.push(j.tile);
    }
    commands.insert_resource(ExtractedBrickBakes {
        headers,
        edits: pending.edits.clone(),
        tiles,
    });
}

/// Upload this frame's bake job headers + edits into storage buffers and (re)size the
/// dist/mat output buffers to the job count. The actual dispatch + per-tile copy into the
/// atlas happens in `SdfBrickBakeNode`. Runs before `prepare_sdf_atlas_gpu` so a realloc that
/// recreates the atlas texture this frame is followed by our bake filling it.
fn prepare_brick_bake_buffers(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    extracted: Option<Res<ExtractedBrickBakes>>,
    mut buffers: ResMut<SdfBakeBuffers>,
) {
    let Some(extracted) = extracted else { return };
    let n = extracted.headers.len() as u32;
    buffers.job_count = n;
    buffers.tiles = extracted.tiles.clone();
    if n == 0 {
        return;
    }
    let _span = info_span!("sdf_prepare_bake_buffers", jobs = n).entered();

    // Headers (std430, GpuJobHeader = 48 bytes).
    let mut header_bytes: Vec<u8> = Vec::with_capacity(extracted.headers.len() * 48);
    for h in &extracted.headers {
        header_bytes.extend_from_slice(&h.coord.x.to_le_bytes());
        header_bytes.extend_from_slice(&h.coord.y.to_le_bytes());
        header_bytes.extend_from_slice(&h.coord.z.to_le_bytes());
        header_bytes.extend_from_slice(&h.voxel_size.to_le_bytes());
        header_bytes.extend_from_slice(&h.dist_band.to_le_bytes());
        header_bytes.extend_from_slice(&h.edit_start.to_le_bytes());
        header_bytes.extend_from_slice(&h.edit_count.to_le_bytes());
        header_bytes.extend_from_slice(&h.pal01.to_le_bytes());
        header_bytes.extend_from_slice(&h.pal23.to_le_bytes());
        header_bytes.extend_from_slice(&0u32.to_le_bytes());
        header_bytes.extend_from_slice(&0u32.to_le_bytes());
        header_bytes.extend_from_slice(&0u32.to_le_bytes());
    }
    buffers.header_buffer = Some(device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_bake_headers"),
        contents: &header_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    }));

    // Edits (std430, GpuEdit = 96 bytes: mat4 + 2×vec4 + 4×u32). Always ≥1 row so the
    // storage binding is never zero-sized.
    let mut edit_bytes: Vec<u8> = Vec::with_capacity(extracted.edits.len().max(1) * 96);
    for e in &extracted.edits {
        for col in e.inv_model.to_cols_array() {
            edit_bytes.extend_from_slice(&col.to_le_bytes());
        }
        for v in [e.params.x, e.params.y, e.params.z, e.params.w] {
            edit_bytes.extend_from_slice(&v.to_le_bytes());
        }
        for v in [e.params2.x, e.params2.y, e.params2.z, e.params2.w] {
            edit_bytes.extend_from_slice(&v.to_le_bytes());
        }
        edit_bytes.extend_from_slice(&e.tag.to_le_bytes());
        edit_bytes.extend_from_slice(&e.op_kind.to_le_bytes());
        edit_bytes.extend_from_slice(&e.smoothing.to_le_bytes());
        edit_bytes.extend_from_slice(&e.material_id.to_le_bytes());
    }
    if edit_bytes.is_empty() {
        edit_bytes.resize(96, 0);
    }
    buffers.edit_buffer = Some(device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_bake_edits"),
        contents: &edit_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    }));

    // Output buffers (STORAGE write target + COPY_SRC for the per-tile blit into the atlas).
    let dist_size = (n * BAKE_DIST_TILE_U32) as u64 * 4;
    let mat_size = (n * BAKE_MAT_TILE_U32) as u64 * 4;
    let needs_dist = buffers.dist_buffer.as_ref().is_none_or(|b| b.size() < dist_size);
    let needs_mat = buffers.mat_buffer.as_ref().is_none_or(|b| b.size() < mat_size);
    if needs_dist {
        buffers.dist_buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("sdf_bake_dist_out"),
            size: dist_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }
    if needs_mat {
        buffers.mat_buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("sdf_bake_mat_out"),
            size: mat_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }
    let _ = &queue; // (kept for parity with sibling prepare systems; no immediate write here)
}

/// Queue the brick-bake compute pipeline. Standalone bind group (no camera/atlas-read): two
/// read-only storage buffers (headers, edits) + two read-write storage buffers (dist, mat
/// output). Runs at `RenderStartup` after `init_sdf_pipeline` (no dependency, just ordering).
fn init_bake_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    bake_shader: Res<SdfBakeShaderHandle>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "sdf_bake_bind_group",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only::<GpuJobHeader>(false),
                storage_buffer_read_only::<super::edits::GpuEdit>(false),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    );
    let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("sdf_bake_pipeline".into()),
        layout: vec![layout.clone()],
        shader: bake_shader.0.clone(),
        ..default()
    });
    commands.insert_resource(SdfBakePipeline {
        pipeline_id,
        layout,
    });
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct SdfBrickBakeLabel;

#[derive(Default)]
struct SdfBrickBakeNode;

impl Node for SdfBrickBakeNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let buffers = world.resource::<SdfBakeBuffers>();
        if buffers.job_count == 0 {
            return Ok(());
        }
        let _span = info_span!("sdf_brick_bake_node", jobs = buffers.job_count).entered();
        let bake = world.resource::<SdfBakePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(bake.pipeline_id) else {
            return Ok(());
        };
        let (Some(header_buf), Some(edit_buf), Some(dist_buf), Some(mat_buf)) = (
            buffers.header_buffer.as_ref(),
            buffers.edit_buffer.as_ref(),
            buffers.dist_buffer.as_ref(),
            buffers.mat_buffer.as_ref(),
        ) else {
            return Ok(());
        };
        // The atlas textures must already exist (a prior bake/realloc created them). If not,
        // there's nothing to copy into yet — skip this frame.
        let gpu_atlas = world.resource::<SdfGpuAtlas>();
        let (Some(dist_tex), Some(mat_tex)) = (&gpu_atlas.dist_tex, &gpu_atlas.mat_tex) else {
            return Ok(());
        };

        let device = render_context.render_device();
        let layout = pipeline_cache.get_bind_group_layout(&bake.layout);
        let bind_group = device.create_bind_group(
            "sdf_bake_bind_group",
            &layout,
            &BindGroupEntries::sequential((
                header_buf.as_entire_buffer_binding(),
                edit_buf.as_entire_buffer_binding(),
                dist_buf.as_entire_buffer_binding(),
                mat_buf.as_entire_buffer_binding(),
            )),
        );

        {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("sdf_brick_bake"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per brick job, laid out in a 2D grid so the count can exceed the
            // 65535 single-dimension dispatch limit (a large edit dirties 70k+ bricks). The
            // shader reconstructs the linear job index from (wg.x, wg.y).
            let wg_x = buffers.job_count.min(BAKE_DISPATCH_WIDTH);
            let wg_y = buffers.job_count.div_ceil(BAKE_DISPATCH_WIDTH);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Blit each job's tile from the output buffers into the persistent atlas textures.
        // The buffer layout matches the texture sub-rect (dist rows padded to 256 bytes,
        // mat rows already 512). `copy_buffer_to_texture` requires bytes_per_row % 256 == 0.
        let edge = BRICK_EDGE as u32;
        let tile_width = edge * edge; // 64
        let encoder = render_context.command_encoder();
        for (i, &tile) in buffers.tiles.iter().enumerate() {
            // Same packing as the lookup rows (single source: `chunk::tile_atlas_base`); unpack the
            // `col_px | row_px<<16` it returns into the sub-rect origin for the texture blit.
            let base = super::chunk::tile_atlas_base(tile);
            let (col_px, row_px) = (base & 0xFFFF, base >> 16);
            let tile_extent = Extent3d {
                width: tile_width,
                height: edge,
                depth_or_array_layers: 1,
            };
            let dist_offset = (i as u32 * BAKE_DIST_TILE_U32) as u64 * 4;
            encoder.copy_buffer_to_texture(
                TexelCopyBufferInfo {
                    buffer: dist_buf,
                    layout: TexelCopyBufferLayout {
                        offset: dist_offset,
                        bytes_per_row: Some(BAKE_DIST_ROW_U32 * 4), // 256 bytes
                        rows_per_image: Some(edge),
                    },
                },
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
                tile_extent,
            );
            let mat_offset = (i as u32 * BAKE_MAT_TILE_U32) as u64 * 4;
            encoder.copy_buffer_to_texture(
                TexelCopyBufferInfo {
                    buffer: mat_buf,
                    layout: TexelCopyBufferLayout {
                        offset: mat_offset,
                        bytes_per_row: Some(BAKE_MAT_ROW_U32 * 4), // 512 bytes
                        rows_per_image: Some(edge),
                    },
                },
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
                tile_extent,
            );
        }

        Ok(())
    }
}
