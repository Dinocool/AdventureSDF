//! The cone-prepass compute pass: one tile-cone march per 8×8 screen tile, writing a per-tile seed
//! distance the fragment march reads to skip empty space. Reuses the shared SDF camera (layout_0)
//! and atlas (layout_1) bind groups from [`SdfPipeline`]; `layout_2` is its write-only seed texture.

use super::*;

pub(super) const SDF_CONE_SHADER_PATH: &str = "shaders/sdf_cone_prepass.wgsl";

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
pub(super) struct SdfConePrepass {
    /// Storage-write view for the compute pass.
    storage_view: TextureView,
    /// Sampled (textureLoad) view for the fragment pass — same texture. Read by the G-buffer node.
    pub(super) read_view: TextureView,
}

/// Screen-tile edge in pixels. MUST match `TILE` in sdf_cone_prepass.wgsl and the divisor
/// the fragment pass uses to index the seed texture.
const CONE_TILE: u32 = 8;
/// Seed-texture capacity in tiles (covers 4K: ceil(3840/8) × ceil(2160/8)).
const CONE_TEX_TILES_X: u32 = 480;
const CONE_TEX_TILES_Y: u32 = 270;

#[derive(Resource)]
pub(super) struct SdfConeShaderHandle(pub(super) Handle<Shader>);

/// Allocate the per-tile seed texture (storage-write + sampled views) and queue the cone-
/// prepass compute pipeline. Runs after `init_sdf_pipeline` so the shared camera/atlas
/// layouts (layout_0/1) already exist on `SdfPipeline`.
pub(super) fn init_cone_pipeline(
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

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(super) struct SdfConeLabel;

#[derive(Default)]
pub(super) struct SdfConeNode;

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
        let bind_group_1 = atlas_bind_group_1(device, &layout_1, gpu_atlas, "sdf_cone_bind_group_1");

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
