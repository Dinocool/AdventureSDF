//! Screen-space GI denoise: resolve the DDGI indirect irradiance to a texture, then run an edge-aware
//! à-trous blur over it, so the coarse probe lattice (one sample per ~0.3 m) stops reading as blocks on
//! flat walls. The combine pass composites the blurred result instead of evaluating `sample_gi` inline.
//!
//! Graph order: … → SdfProbeTrace → SdfGiResolve → SdfGiBlur → SdfCombine → …
//!
//! - Resolve (`sdf_gi_resolve.wgsl`): per pixel, `sample_gi(world_pos, normal) * intensity` → `gi_a`
//!   (rgb = GI, a = camera distance, the blur's depth edge-stop signal). Reuses the combine GI binding
//!   set (camera + atlas + G-buffer + probe).
//! - Blur (`sdf_gi_blur.wgsl`): `BLUR_PASSES` à-trous iterations with a doubling pixel step, ping-ponging
//!   `gi_a`/`gi_b`, final pass → `gi_out`. Depth + normal edge stops keep GI from crossing surfaces.

use super::*;
use bevy::render::render_resource::encase::UniformBuffer as EncaseUniformBuffer;

pub(super) const SDF_GI_RESOLVE_SHADER_PATH: &str = "shaders/sdf_gi_resolve.wgsl";
pub(super) const SDF_GI_BLUR_SHADER_PATH: &str = "shaders/sdf_gi_blur.wgsl";

pub(super) const GI_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
/// À-trous iterations. Steps double each pass (1,2,4,8,16 px) → ~31 px effective radius, enough to
/// dissolve the probe-lattice blocks without smearing across edges.
const BLUR_PASSES: usize = 5;

#[derive(Resource)]
pub(super) struct SdfGiResolveShaderHandle(pub(super) Handle<Shader>);
#[derive(Resource)]
pub(super) struct SdfGiBlurShaderHandle(pub(super) Handle<Shader>);

/// Per-pass blur knobs (mirrors `sdf_gi_blur.wgsl::GiBlurParams`).
#[derive(ShaderType, Clone, Copy, Default)]
struct GiBlurParams {
    inv_size: Vec2,
    step: f32,
    depth_sigma: f32,
    normal_power: f32,
}

/// The GI resolve/blur textures (full-screen) + the per-pass blur uniform buffers. `out` holds the
/// final blurred GI the combine pass samples.
#[derive(Resource, Default)]
pub(super) struct SdfGiTextures {
    a: Option<TextureView>,
    b: Option<TextureView>,
    out: Option<TextureView>,
    sampler: Option<Sampler>,
    params: Vec<Buffer>, // one per à-trous pass
    size: UVec2,
}

/// The blurred-GI apply group the combine pass binds at group 1 (replacing the old atlas group): the
/// final GI texture + a sampler. Shared descriptor so the combine pipeline + bind group agree.
pub(super) fn gi_apply_layout_desc() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "sdf_gi_apply",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                sampler(SamplerBindingType::NonFiltering),
            ),
        ),
    )
}

#[derive(Resource)]
pub(super) struct SdfGiPipelines {
    pub(super) resolve_id: CachedRenderPipelineId,
    blur_id: CachedRenderPipelineId,
    /// Blur group: gi_in tex + gbuffer normal tex + sampler + params uniform.
    blur_layout: BindGroupLayoutDescriptor,
    /// Shader defs the live `resolve_id` was built with — the resolve carries the
    /// `SDF_DEBUG_PROBE_LOD` / `SDF_DEBUG_PROBE_COVERAGE` `#ifdef` branches, so it must rebuild when
    /// the active debug set changes (mirrors the primary/combine rebuild). Empty at first queue.
    pub(super) resolve_defs: Vec<String>,
}

/// Queue the GI-resolve render pipeline with the given shader defs. Shared by the initial queue and the
/// def-change rebuild so the descriptor (layout/targets) is defined once. The resolve binds 0 = camera,
/// 1 = atlas (probe lookup), 2 = G-buffer (combine's layout), 3 = probe irradiance.
pub(super) fn queue_resolve_pipeline(
    pipeline_cache: &PipelineCache,
    sdf_pipeline: &SdfPipeline,
    combine: &SdfCombinePipeline,
    resolve_shader: &SdfGiResolveShaderHandle,
    fullscreen_shader: &FullscreenShader,
    shader_defs: Vec<bevy::shader::ShaderDefVal>,
) -> CachedRenderPipelineId {
    pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("sdf_gi_resolve_pipeline".into()),
        layout: vec![
            sdf_pipeline.layout_0.clone(),
            sdf_pipeline.layout_1.clone(),
            combine.layout.clone(),
            probe::probe_apply_layout_desc(),
        ],
        vertex: fullscreen_shader.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: resolve_shader.0.clone(),
            shader_defs,
            targets: vec![Some(ColorTargetState {
                format: GI_FORMAT,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    })
}

/// Whether a probe-state debug overlay (LOD / coverage) is active — these are computed in the resolve
/// pass and must reach the screen UNBLURRED, so the combine binds the pre-blur `gi_a` for them.
pub(super) fn probe_debug_active(defs: &[String]) -> bool {
    defs.iter().any(|d| d == "SDF_DEBUG_PROBE_LOD" || d == "SDF_DEBUG_PROBE_COVERAGE")
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(super) struct SdfGiResolveLabel;
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(super) struct SdfGiBlurLabel;

/// Queue the resolve + blur pipelines. Runs after `init_combine_pipeline` (reuses its G-buffer layout).
pub(super) fn init_gi_pipelines(
    mut commands: Commands,
    fullscreen_shader: Res<FullscreenShader>,
    resolve_shader: Res<SdfGiResolveShaderHandle>,
    blur_shader: Res<SdfGiBlurShaderHandle>,
    sdf_pipeline: Res<SdfPipeline>,
    combine: Res<SdfCombinePipeline>,
    pipeline_cache: Res<PipelineCache>,
) {
    // Resolve: 0 = camera, 1 = atlas (probe lookup), 2 = G-buffer (reuse combine's), 3 = probe.
    // Queued with empty defs; `rebuild_pipeline_on_def_change` re-queues it with the active debug set
    // (so the SDF_DEBUG_PROBE_LOD / _COVERAGE `#ifdef` branches in the resolve actually compile in).
    let resolve_id = queue_resolve_pipeline(
        &pipeline_cache,
        &sdf_pipeline,
        &combine,
        &resolve_shader,
        &fullscreen_shader,
        vec![],
    );

    let blur_layout = BindGroupLayoutDescriptor::new(
        "sdf_gi_blur",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: false }), // gi_in
                texture_2d(TextureSampleType::Float { filterable: false }), // gbuffer normal
                sampler(SamplerBindingType::NonFiltering),
                uniform_buffer::<GiBlurParams>(false),
            ),
        ),
    );
    let blur_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("sdf_gi_blur_pipeline".into()),
        layout: vec![blur_layout.clone()],
        vertex: fullscreen_shader.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: blur_shader.0.clone(),
            shader_defs: vec![],
            targets: vec![Some(ColorTargetState {
                format: GI_FORMAT,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    });

    commands.insert_resource(SdfGiPipelines {
        resolve_id,
        blur_id,
        blur_layout,
        resolve_defs: vec![],
    });
    commands.insert_resource(SdfGiTextures::default());
}

/// (Re)size the GI textures to the view + (re)write the per-pass blur uniforms. Mirrors
/// `prepare_sdf_gbuffer`.
pub(super) fn prepare_sdf_gi(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    params: Res<super::super::DdgiParams>,
    mut gi: ResMut<SdfGiTextures>,
    views: Query<&ViewTarget, With<SdfCameraData>>,
) {
    let Some(view) = views.iter().next() else {
        return;
    };
    let extent = view.main_texture().size();
    let dims = UVec2::new(extent.width, extent.height);

    if gi.a.is_none() || gi.size != dims {
        let make = |label: &str| {
            device
                .create_texture(&TextureDescriptor {
                    label: Some(label),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: GI_FORMAT,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&TextureViewDescriptor::default())
        };
        gi.a = Some(make("sdf_gi_a"));
        gi.b = Some(make("sdf_gi_b"));
        gi.out = Some(make("sdf_gi_out"));
        gi.sampler = Some(device.create_sampler(&SamplerDescriptor {
            label: Some("sdf_gi_sampler"),
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            ..default()
        }));
        gi.params = (0..BLUR_PASSES)
            .map(|_| {
                device.create_buffer(&BufferDescriptor {
                    label: Some("sdf_gi_blur_params"),
                    size: GiBlurParams::min_size().get(),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        gi.size = dims;
    }

    let inv_size = Vec2::new(1.0 / dims.x.max(1) as f32, 1.0 / dims.y.max(1) as f32);
    for (i, buf) in gi.params.iter().enumerate() {
        let p = GiBlurParams {
            inv_size,
            step: (1u32 << i) as f32, // 1,2,4,8,16
            depth_sigma: params.gi_blur_depth_sigma.max(1.0e-3),
            normal_power: params.gi_blur_normal_power.max(0.0),
        };
        let mut bytes = EncaseUniformBuffer::new(Vec::<u8>::new());
        bytes.write(&p).unwrap();
        queue.write_buffer(buf, 0, bytes.as_ref());
    }
}

/// Resolve `sample_gi` into the `gi_a` texture (same binding set as the old inline combine GI path).
#[derive(Default)]
pub(super) struct SdfGiResolveNode;

impl ViewNode for SdfGiResolveNode {
    type ViewQuery = ();

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        _view: (),
        world: &World,
    ) -> Result<(), NodeRunError> {
        if world.get::<SdfCameraData>(graph.view_entity()).is_none() {
            return Ok(());
        }
        if let Some(enabled) = world.get_resource::<SdfRenderEnabled>()
            && !enabled.0
        {
            return Ok(());
        }
        let pipelines = world.resource::<SdfGiPipelines>();
        let sdf = world.resource::<SdfPipeline>();
        let combine = world.resource::<SdfCombinePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let device = render_context.render_device().clone();
        let gi = world.resource::<SdfGiTextures>();

        let (Some(pipeline), Some(gi_a)) =
            (pipeline_cache.get_render_pipeline(pipelines.resolve_id), &gi.a)
        else {
            return Ok(());
        };

        let gbuffer = world.resource::<SdfGBuffer>();
        let (Some(albedo), Some(normal), Some(emissive), Some(samp)) = (
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
        let gpu_atlas = world.resource::<SdfGpuAtlas>();
        if gpu_atlas.pages.is_none()
            || gpu_atlas.material_buffer.is_none()
            || gpu_atlas.tex_array_views.is_none()
        {
            return Ok(());
        }
        let Some(probe_bg3) = probe::probe_apply_bind_group(&device, pipeline_cache, world) else {
            return Ok(());
        };

        let layout_0 = pipeline_cache.get_bind_group_layout(&sdf.layout_0);
        let layout_1 = pipeline_cache.get_bind_group_layout(&sdf.layout_1);
        let gbuf_layout = pipeline_cache.get_bind_group_layout(&combine.layout);

        let bg0 = device.create_bind_group(
            "sdf_gi_resolve_bg0",
            &layout_0,
            &BindGroupEntries::sequential((camera_binding.clone(),)),
        );
        let bg1 = atlas_bind_group_1(&device, &layout_1, gpu_atlas, "sdf_gi_resolve_bg1");
        let bg2 = device.create_bind_group(
            "sdf_gi_resolve_gbuffer",
            &gbuf_layout,
            &BindGroupEntries::sequential((albedo, normal, emissive, samp)),
        );

        let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("sdf_gi_resolve"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: gi_a,
                resolve_target: None,
                depth_slice: None,
                ops: Operations { load: LoadOp::Clear(LinearRgba::NONE.into()), store: StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_render_pipeline(pipeline);
        pass.set_bind_group(0, &bg0, &[0]);
        pass.set_bind_group(1, &bg1, &[]);
        pass.set_bind_group(2, &bg2, &[]);
        pass.set_bind_group(3, &probe_bg3, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

/// Edge-aware à-trous blur: `BLUR_PASSES` iterations ping-ponging `gi_a`/`gi_b`, final → `gi_out`.
#[derive(Default)]
pub(super) struct SdfGiBlurNode;

impl ViewNode for SdfGiBlurNode {
    type ViewQuery = ();

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        _view: (),
        world: &World,
    ) -> Result<(), NodeRunError> {
        if world.get::<SdfCameraData>(graph.view_entity()).is_none() {
            return Ok(());
        }
        if let Some(enabled) = world.get_resource::<SdfRenderEnabled>()
            && !enabled.0
        {
            return Ok(());
        }
        let pipelines = world.resource::<SdfGiPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let device = render_context.render_device().clone();
        let gi = world.resource::<SdfGiTextures>();
        let gbuffer = world.resource::<SdfGBuffer>();

        let (Some(pipeline), Some(a), Some(b), Some(out), Some(samp), Some(normal)) = (
            pipeline_cache.get_render_pipeline(pipelines.blur_id),
            &gi.a,
            &gi.b,
            &gi.out,
            &gi.sampler,
            &gbuffer.normal_mat_view,
        ) else {
            return Ok(());
        };
        if gi.params.len() != BLUR_PASSES {
            return Ok(());
        }
        let layout = pipeline_cache.get_bind_group_layout(&pipelines.blur_layout);

        for i in 0..BLUR_PASSES {
            let src = if i % 2 == 0 { a } else { b };
            let dst = if i == BLUR_PASSES - 1 {
                out
            } else if i % 2 == 0 {
                b
            } else {
                a
            };
            let bg = device.create_bind_group(
                "sdf_gi_blur_bg",
                &layout,
                &BindGroupEntries::sequential((
                    src,
                    normal,
                    samp,
                    gi.params[i].as_entire_buffer_binding(),
                )),
            );
            let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("sdf_gi_blur"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: dst,
                    resolve_target: None,
                    depth_slice: None,
                    ops: Operations {
                        load: LoadOp::Clear(LinearRgba::NONE.into()),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_render_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        Ok(())
    }
}

/// The combine pass's group-1 bind group: the final blurred GI texture + sampler.
pub(super) fn gi_apply_bind_group(device: &RenderDevice, world: &World) -> Option<BindGroup> {
    let gi = world.get_resource::<SdfGiTextures>()?;
    let pipeline_cache = world.resource::<PipelineCache>();
    // A probe-state overlay (LOD / coverage) is written by the resolve into `gi_a`; bind that PRE-blur
    // so the lit pass shows it crisp (the à-trous blur would smear the hue / coverage edges). Normal GI
    // binds the blurred `gi_out`.
    let debug_probe = world
        .get_resource::<super::ExtractedShaderDefs>()
        .is_some_and(|d| probe_debug_active(&d.defs));
    let tex = if debug_probe { gi.a.as_ref()? } else { gi.out.as_ref()? };
    let samp = gi.sampler.as_ref()?;
    let layout = pipeline_cache.get_bind_group_layout(&gi_apply_layout_desc());
    Some(device.create_bind_group(
        "sdf_combine_gi",
        &layout,
        &BindGroupEntries::sequential((tex, samp)),
    ))
}
