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
    sampler, storage_buffer_read_only, texture_2d, uniform_buffer,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::view::{ViewDepthTexture, ViewTarget};
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderStartup};

use super::atlas::{BRICK_EDGE, SdfAtlas};
use super::edits::MATERIAL_SLOTS;
use super::bvh::Bvh;
use super::{SdfCamera, SdfColor, SdfGridConfig, SdfOrder, SdfRenderEnabled, SdfVolume};

// --- GPU Types ---

/// One entry in the brick lookup buffer: maps a brick id to its column in the
/// atlas. Object IDs now live per-voxel in the object texture, so this is purely
/// a spatial index. `_pad` keeps the struct 16-byte aligned for std430.
#[derive(ShaderType, Clone, Copy, Default)]
struct GpuBrickLookup {
    brick_id: u32,
    atlas_u: u32,
    atlas_v: u32,
    _pad: u32,
}

/// GPU mirror of [`super::bvh::BvhNode`] for the storage-buffer layout (32 bytes:
/// two `vec3<f32> + u32` rows). Only used to declare the bind-group layout; the
/// actual node bytes come straight from `Bvh::to_gpu_bytes`.
#[derive(ShaderType, Clone, Copy, Default)]
struct GpuBvhNode {
    aabb_min: Vec3,
    left_or_first: u32,
    aabb_max: Vec3,
    count_or_right: u32,
}

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
    debug_params: Vec4, // x = max_steps, y = max_dist, z = sdf_eps, w = bvh_node_count
    object_colors: [Vec4; 8],
}

// --- Extracted Atlas ---

#[derive(Resource, Default)]
struct ExtractedSdfAtlas {
    /// R16Snorm distance values, one i16 per voxel (the CSG-combined surface).
    dist_data: Vec<i16>,
    /// Dense per-material distance field, Rgba16Snorm. Two textures of 4 channels
    /// each cover the 8 material slots: `mat_lo` = materials 0..3, `mat_hi` =
    /// materials 4..7. Same tile layout as `dist_data` (4 i16 per texel).
    mat_lo_data: Vec<i16>,
    mat_hi_data: Vec<i16>,
    lookup_data: Vec<GpuBrickLookup>,
    texture_width: u32,
    texture_height: u32,
    dirty: bool,
}

/// Flattened BVH nodes (raw std430 bytes, 32B each) extracted from the main world
/// for GPU upload. Used by the raymarch only to accelerate empty-space skipping —
/// the surface is still sampled from the atlas textures.
#[derive(Resource, Default)]
struct ExtractedSdfBvh {
    node_bytes: Vec<u8>,
    node_count: u32,
    dirty: bool,
}

// --- GPU Atlas ---

#[derive(Resource, Default)]
struct SdfGpuAtlas {
    dist_view: Option<TextureView>,
    /// Dense per-material distance atlases (Rgba16Snorm): lo = materials 0..3,
    /// hi = materials 4..7. The shader argmins across all 8 for the material id.
    mat_lo_view: Option<TextureView>,
    mat_hi_view: Option<TextureView>,
    sampler: Option<Sampler>,
    lookup_buffer: Option<Buffer>,
    bvh_buffer: Option<Buffer>,
    bvh_node_count: u32,
}

// --- Pipeline ---

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
        if let Some(enabled) = world.get_resource::<SdfRenderEnabled>() {
            if !enabled.0 {
                return Ok(());
            }
        }

        let pipeline_res = world.resource::<SdfPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let device = render_context.render_device();

        let pipeline = pipeline_cache.get_render_pipeline(pipeline_res.pipeline_id);

        if pipeline.is_none() {
            use std::sync::atomic::{AtomicBool, Ordering};
            static LOGGED: AtomicBool = AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::Relaxed) {
                match pipeline_cache.get_render_pipeline_state(pipeline_res.pipeline_id) {
                    bevy::render::render_resource::CachedPipelineState::Err(err) => {
                        bevy::log::error!("SDF pipeline error: {err}");
                    }
                    _ => {}
                }
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
                create_dummy_bg0(&device, &layout_0)
            }
        } else {
            create_dummy_bg0(&device, &layout_0)
        };

        // Bind group 1: atlas (always available — dummy in init)
        let gpu_atlas = world.resource::<SdfGpuAtlas>();
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
                gpu_atlas.mat_lo_view.as_ref().unwrap(),
                gpu_atlas.mat_hi_view.as_ref().unwrap(),
                gpu_atlas
                    .bvh_buffer
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

pub struct SdfRenderPlugin;

impl Plugin for SdfRenderPlugin {
    fn build(&self, app: &mut App) {
        // Load shader asset in main world so it's available for extraction
        let shader_handle = app.world().resource::<AssetServer>().load(SDF_SHADER_PATH);
        app.insert_resource(SdfShaderHandle(shader_handle))
            .init_resource::<SdfShaderDefs>()
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
                    .after(super::orbit_camera),
            );

        #[cfg(feature = "debug_toolkit")]
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
            .add_systems(ExtractSchedule, extract_sdf_bvh)
            .add_systems(ExtractSchedule, extract_shader_defs)
            .add_systems(Render, prepare_sdf_atlas_gpu)
            .add_systems(Render, prepare_sdf_bvh_gpu)
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

fn prepare_sdf_camera_data(
    mut commands: Commands,
    cameras: Query<(Entity, &Camera, &Transform), With<SdfCamera>>,
    volumes: Query<(Entity, &SdfColor, &SdfOrder), With<SdfVolume>>,
    atlas: Res<SdfAtlas>,
    config: Res<SdfGridConfig>,
    raymarch: Res<super::SdfRaymarchParams>,
    bvh: Res<Bvh>,
) {
    let mut object_colors = [Vec4::ZERO; 8];

    // Material id = position in SdfOrder order (ties by entity index), matching
    // `gather_sorted_edits` so colours line up with the baked per-voxel ids.
    let mut ordered: Vec<(Entity, Color, SdfOrder)> =
        volumes.iter().map(|(e, c, o)| (e, c.0, *o)).collect();
    ordered.sort_by(|a, b| a.2.cmp(&b.2).then(a.0.index().cmp(&b.0.index())));

    for (i, (_, color, _)) in ordered.iter().take(8).enumerate() {
        let linear = color.to_linear();
        object_colors[i] = Vec4::new(linear.red, linear.green, linear.blue, 1.0);
    }

    let num_lookups = atlas.bricks.len() as u32;
    let bpa = config.bricks_per_axis();
    let grid_size = config.grid_size;

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
                num_lookups as f32,
            ),
            debug_params: Vec4::new(
                raymarch.max_steps as f32,
                raymarch.max_dist,
                raymarch.sdf_eps,
                bvh.nodes.len() as f32,
            ),
            object_colors,
        });
    }
}

// --- Bridge: Sync debug state to shader defs ---

#[cfg(feature = "debug_toolkit")]
fn sync_sdf_shader_defs(
    registry: Res<crate::debug_toolkit::registry::ShaderDebugRegistry>,
    state: Res<crate::debug_toolkit::registry::ShaderDebugState>,
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

fn extract_sdf_atlas(
    atlas: Extract<Res<SdfAtlas>>,
    config: Extract<Res<SdfGridConfig>>,
    mut commands: Commands,
) {
    let num_bricks = atlas.bricks.len() as u32;
    if num_bricks == 0 {
        commands.insert_resource(ExtractedSdfAtlas::default());
        return;
    }

    // Atlas is a 2D texture: each brick occupies an EDGE*EDGE-wide, EDGE-tall
    // tile. Within a tile pixel (u,v) = (y*EDGE + x, z) addresses voxel (x,y,z).
    let edge = BRICK_EDGE as u32;
    let tile_width = edge * edge; // 64
    let texture_width = num_bricks * tile_width;
    let texture_height = edge; // 8
    let pixels = (texture_width * texture_height) as usize;
    let mut dist_data = vec![0i16; pixels];
    // Rgba16Snorm: 4 material-distance channels per texel. lo = materials 0..3,
    // hi = materials 4..7. Far sentinel (+1.0 snorm) so empty slots lose the argmin.
    let far = i16::MAX;
    let mut mat_lo_data = vec![far; pixels * 4];
    let mut mat_hi_data = vec![far; pixels * 4];
    let mut lookups = Vec::with_capacity(num_bricks as usize);

    for (i, (coord, packed)) in atlas.bricks.iter().enumerate() {
        let base_u = i as u32 * tile_width;

        for z in 0..edge {
            for y in 0..edge {
                for x in 0..edge {
                    let src_idx = (z * edge * edge + y * edge + x) as usize;
                    let dst_u = base_u + y * edge + x;
                    let dst_idx = (z * texture_width + dst_u) as usize;
                    dist_data[dst_idx] = packed.dist[src_idx];

                    // Split the 8 per-material slots across the two RGBA textures.
                    let mat_base = src_idx * MATERIAL_SLOTS;
                    for m in 0..4 {
                        mat_lo_data[dst_idx * 4 + m] = packed.mat_dist[mat_base + m];
                        mat_hi_data[dst_idx * 4 + m] = packed.mat_dist[mat_base + 4 + m];
                    }
                }
            }
        }

        lookups.push(GpuBrickLookup {
            brick_id: config.brick_id(*coord),
            atlas_u: base_u,
            atlas_v: 0,
            _pad: 0,
        });
    }

    // Sorted so the shader can binary-search by brick_id.
    lookups.sort_by_key(|l| l.brick_id);

    commands.insert_resource(ExtractedSdfAtlas {
        dist_data,
        mat_lo_data,
        mat_hi_data,
        lookup_data: lookups,
        texture_width,
        texture_height,
        dirty: true,
    });
}

// --- Extract: Flatten BVH for GPU ---

fn extract_sdf_bvh(bvh: Extract<Res<Bvh>>, mut commands: Commands) {
    // A single empty node keeps the storage buffer non-zero-sized so the bind
    // group is always valid even before the first bake.
    let (node_bytes, node_count) = if bvh.nodes.is_empty() {
        (vec![0u8; 32], 0)
    } else {
        (bvh.to_gpu_bytes(), bvh.nodes.len() as u32)
    };
    commands.insert_resource(ExtractedSdfBvh {
        node_bytes,
        node_count,
        dirty: true,
    });
}

// --- Prepare: Upload to GPU ---

fn prepare_sdf_bvh_gpu(
    device: Res<RenderDevice>,
    extracted: Option<Res<ExtractedSdfBvh>>,
    mut gpu_atlas: ResMut<SdfGpuAtlas>,
) {
    let Some(extracted) = extracted else { return };
    if !extracted.dirty {
        return;
    }
    let buffer = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_bvh_buffer"),
        contents: &extracted.node_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    gpu_atlas.bvh_buffer = Some(buffer);
    gpu_atlas.bvh_node_count = extracted.node_count;
}

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

    let num_lookups = extracted.lookup_data.len() as u32;
    if num_lookups == 0 {
        return;
    }

    // Lookup buffer (std430: 4 x u32 per entry).
    let mut buffer_bytes = Vec::with_capacity(extracted.lookup_data.len() * 16);
    for l in &extracted.lookup_data {
        buffer_bytes.extend_from_slice(&l.brick_id.to_le_bytes());
        buffer_bytes.extend_from_slice(&l.atlas_u.to_le_bytes());
        buffer_bytes.extend_from_slice(&l.atlas_v.to_le_bytes());
        buffer_bytes.extend_from_slice(&l._pad.to_le_bytes());
    }

    let lookup_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_lookup_buffer"),
        contents: &buffer_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    let size = Extent3d {
        width: extracted.texture_width,
        height: extracted.texture_height,
        depth_or_array_layers: 1,
    };

    // Distance atlas: R16Snorm, the trilinearly-interpolated SDF. i16 values
    // are uploaded as little-endian bytes (2 per texel).
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

    // Dense per-material distance atlases: Rgba16Snorm, materials 0..3 (lo) and
    // 4..7 (hi). The shader trilinearly interpolates these and argmins for the
    // material id, so the boundary is the exact sub-voxel bisector.
    let mat_lo_tex = device.create_texture_with_data(
        &queue,
        &TextureDescriptor {
            label: Some("sdf_mat_lo_atlas"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Snorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        TextureDataOrder::default(),
        &i16s_to_le_bytes(&extracted.mat_lo_data),
    );
    let mat_hi_tex = device.create_texture_with_data(
        &queue,
        &TextureDescriptor {
            label: Some("sdf_mat_hi_atlas"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Snorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        TextureDataOrder::default(),
        &i16s_to_le_bytes(&extracted.mat_hi_data),
    );

    let atlas_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("sdf_atlas_sampler"),
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        mipmap_filter: FilterMode::Nearest,
        ..default()
    });

    gpu_atlas.dist_view = Some(dist_tex.create_view(&TextureViewDescriptor::default()));
    gpu_atlas.mat_lo_view = Some(mat_lo_tex.create_view(&TextureViewDescriptor::default()));
    gpu_atlas.mat_hi_view = Some(mat_hi_tex.create_view(&TextureViewDescriptor::default()));
    gpu_atlas.sampler = Some(atlas_sampler);
    gpu_atlas.lookup_buffer = Some(lookup_buffer);
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
                // binding 2: brick lookup buffer
                storage_buffer_read_only::<GpuBrickLookup>(false),
                // binding 3: material-distance atlas lo (Rgba16Snorm, materials 0..3)
                texture_2d(TextureSampleType::Float { filterable: false }),
                // binding 4: material-distance atlas hi (Rgba16Snorm, materials 4..7)
                texture_2d(TextureSampleType::Float { filterable: false }),
                // binding 5: BVH nodes (empty-space-skip acceleration)
                storage_buffer_read_only::<GpuBvhNode>(false),
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
    let dummy_lookup = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_dummy_lookup"),
        contents: &[0u8; 16],
        usage: BufferUsages::STORAGE,
    });
    // One zeroed 32-byte BVH node so binding 5 is always valid pre-bake.
    let dummy_bvh = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_dummy_bvh"),
        contents: &[0u8; 32],
        usage: BufferUsages::STORAGE,
    });
    // Matching dummy material atlases (Rgba16Snorm = 8 bytes/texel) so bind group
    // 1 is always valid before the first bake.
    let dummy_mat = |label: &'static str| {
        device.create_texture_with_data(
            &queue,
            &TextureDescriptor {
                label: Some(label),
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
        )
    };
    let dummy_mat_lo = dummy_mat("sdf_dummy_mat_lo");
    let dummy_mat_hi = dummy_mat("sdf_dummy_mat_hi");

    commands.insert_resource(SdfPipeline {
        pipeline_id,
        layout_0,
        layout_1,
        shader_handle: shader,
    });
    commands.insert_resource(SdfGpuAtlas {
        dist_view: Some(dummy_tex.create_view(&TextureViewDescriptor::default())),
        mat_lo_view: Some(dummy_mat_lo.create_view(&TextureViewDescriptor::default())),
        mat_hi_view: Some(dummy_mat_hi.create_view(&TextureViewDescriptor::default())),
        sampler: Some(dummy_sampler),
        lookup_buffer: Some(dummy_lookup),
        bvh_buffer: Some(dummy_bvh),
        bvh_node_count: 0,
    });
}
