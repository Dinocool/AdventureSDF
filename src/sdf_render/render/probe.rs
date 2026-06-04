//! DDGI probe trace pass (P1 MVP): flat per-probe irradiance, performance-amortized.
//!
//! A compute pass enumerates probes by dispatching one workgroup per RESIDENT chunk (a compact list
//! extracted from [`crate::sdf_render::chunk::LiveChunkTables::resident_rows`] — NOT the full
//! `R³·lod_count` toroidal directory, which is millions of empty slots). Each occupied brick owns a
//! contiguous block of `subdiv³` probe slots; each is a probe that traces rays through the shared
//! `raymarch` (reusing the default march's empty-space skipping at a coarse LOD floor) and writes flat
//! irradiance into a single in-place buffer, indexed by the brick's tile-run slot — the SAME index the
//! deferred-lit apply resolves a world position to. Round-robin (`update_stride`) re-traces only
//! `1/stride` of probe slots per frame; the rest retain their value and converge via temporal blend.
//!
//! Reuses `SdfPipeline::layout_0` (camera) + `layout_1` (atlas) like [`super::cone`]; `layout_3` is the
//! probe group (irradiance R/W + params + resident chunks), with a read-only `layout_3_apply` variant
//! the lit pass binds.

use super::chunk_tables::ChunkBufCapacity;
use super::*;
use crate::sdf_render::atlas::SdfAtlas;
use crate::sdf_render::chunk::ChunkLookup;
use crate::sdf_render::DdgiParams;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized,
};

pub(super) const SDF_PROBE_TRACE_SHADER_PATH: &str = "shaders/sdf_probe_trace.wgsl";

/// The deferred-lit apply group-3 layout: irradiance (read) + params (uniform). Shared descriptor so
/// the combine pipeline's layout and the bind group resolve to the same cached layout.
pub(super) fn probe_apply_layout_desc() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "sdf_probe_apply_bind_group_3",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                storage_buffer_read_only_sized(false, None), // irradiance (read)
                uniform_buffer::<ProbeParams>(false),        // params
            ),
        ),
    )
}

/// Per-frame probe-trace knobs (mirrors `sdf_probe_trace.wgsl::ProbeParams`).
#[derive(ShaderType, Clone, Copy, Default)]
struct ProbeParams {
    ray_count: u32,
    hysteresis: f32,
    intensity: f32,
    frame: u32,
    subdiv: u32,
    update_stride: u32,
    gi_range: f32,
    normal_bias: f32,
    view_bias: f32,
}

#[derive(Resource)]
pub(super) struct SdfProbeShaderHandle(pub(super) Handle<Shader>);

/// The compact resident-chunk directory rows, extracted from the main world each frame. The trace
/// dispatches one workgroup per row.
#[derive(Resource, Default)]
pub(super) struct ExtractedResidentChunks {
    rows: Vec<ChunkLookup>,
}

/// Extract the resident-chunk list (one row per non-empty chunk) for the compact trace dispatch.
pub(super) fn extract_resident_chunks(atlas: Extract<Res<SdfAtlas>>, mut commands: Commands) {
    commands.insert_resource(ExtractedResidentChunks { rows: atlas.live_chunks.resident_rows() });
}

/// Single in-place per-probe irradiance buffer + params uniform + the resident-chunk buffer the trace
/// dispatches over. The lit apply binds the same `irr` buffer read-only.
#[derive(Resource)]
pub(super) struct SdfProbeBuffers {
    irr: Buffer,
    params: Buffer,
    resident: Buffer,
    /// Element capacity (number of `vec4<f32>` slots = tile-run capacity × subdiv³).
    capacity: u32,
    /// Resident-chunk count = trace workgroup count.
    resident_count: u32,
    /// Frame counter (drives round-robin phase + frame-0 history reset).
    frame: u32,
}

#[derive(Resource)]
struct SdfProbeTracePipeline {
    pipeline_id: CachedComputePipelineId,
    /// Trace group: irradiance (read_write) + params (uniform) + resident chunks (read).
    layout_3: BindGroupLayoutDescriptor,
    /// Apply group (lit pass): irradiance (read) + params (uniform).
    layout_3_apply: BindGroupLayoutDescriptor,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(super) struct SdfProbeTraceLabel;

/// Allocate the probe-buffer layouts + queue the trace compute pipeline. Runs after `init_sdf_pipeline`.
pub(super) fn init_probe_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    sdf_pipeline: Res<SdfPipeline>,
    probe_shader: Res<SdfProbeShaderHandle>,
) {
    let layout_3 = BindGroupLayoutDescriptor::new(
        "sdf_probe_bind_group_3",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_sized(false, None),           // irradiance (read_write, in place)
                uniform_buffer::<ProbeParams>(false),        // params
                storage_buffer_read_only_sized(false, None), // resident chunks
            ),
        ),
    );
    let layout_3_apply = probe_apply_layout_desc();

    let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("sdf_probe_trace_pipeline".into()),
        layout: vec![
            sdf_pipeline.layout_0.clone(), // camera
            sdf_pipeline.layout_1.clone(), // atlas (chunk_buf + raymarch)
            layout_3.clone(),
        ],
        shader: probe_shader.0.clone(),
        ..default()
    });

    let dummy = |label: &str| {
        device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: 16,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let params = device.create_buffer(&BufferDescriptor {
        label: Some("sdf_probe_params"),
        size: ProbeParams::min_size().get(),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    commands.insert_resource(SdfProbeTracePipeline { pipeline_id, layout_3, layout_3_apply });
    commands.insert_resource(SdfProbeBuffers {
        irr: dummy("sdf_probe_irr"),
        params,
        resident: dummy("sdf_probe_resident"),
        capacity: 0,
        resident_count: 0,
        frame: 0,
    });
}

/// (Re)size the irradiance buffer to the tile-run capacity × subdiv³, upload the resident-chunk list +
/// params uniform. Runs in the Render schedule after the atlas extract set [`ChunkBufCapacity`].
pub(super) fn prepare_sdf_probe(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cap: Res<ChunkBufCapacity>,
    params_res: Res<DdgiParams>,
    resident: Res<ExtractedResidentChunks>,
    mut bufs: ResMut<SdfProbeBuffers>,
) {
    let subdiv = params_res.subdiv.clamp(1, 4);
    // Each probe slot holds an octahedral tile (PROBE_OCT_TEXELS vec4s).
    let need = cap.tile_slots.max(1)
        * subdiv
        * subdiv
        * subdiv
        * crate::sdf_render::probe::PROBE_OCT_TEXELS;
    if need != bufs.capacity {
        // Fresh (zeroed) buffer — history starts empty, so frame 0 takes the traced value directly.
        bufs.irr = device.create_buffer(&BufferDescriptor {
            label: Some("sdf_probe_irr"),
            size: need as u64 * 16,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        bufs.capacity = need;
        bufs.frame = 0;
    }

    // Resident-chunk list → buffer (recreated each frame; it tracks the moving clipmap window and is
    // small — hundreds–thousands of 20-byte rows). One dummy row keeps the binding valid when empty.
    let mut bytes = Vec::with_capacity(resident.rows.len() * 20);
    for c in &resident.rows {
        for v in [c.key_hi, c.key_lo, c.occ_lo, c.occ_hi, c.tile_run_base] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    if bytes.is_empty() {
        bytes.resize(20, 0xff); // a sentinel-keyed row (never matches) so the trace finds no probes
    }
    bufs.resident = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_probe_resident"),
        contents: &bytes,
        usage: BufferUsages::STORAGE,
    });
    bufs.resident_count = resident.rows.len().max(1) as u32;

    let p = ProbeParams {
        ray_count: params_res.ray_count.max(1),
        hysteresis: params_res.hysteresis.clamp(0.0, 0.99),
        // Effective apply scale: zero when GI is off, so the lit pass adds nothing even if the
        // irradiance buffer still holds values traced while it was on (no stale-GI leak).
        intensity: if params_res.enabled { params_res.intensity.max(0.0) } else { 0.0 },
        frame: bufs.frame,
        subdiv,
        update_stride: params_res.update_stride.max(1),
        gi_range: params_res.gi_range.max(1.0),
        normal_bias: params_res.normal_bias.max(0.0),
        view_bias: params_res.view_bias.max(0.0),
    };
    let mut ubytes = bevy::render::render_resource::encase::UniformBuffer::new(Vec::<u8>::new());
    ubytes.write(&p).unwrap();
    queue.write_buffer(&bufs.params, 0, ubytes.as_ref());

    bufs.frame = bufs.frame.saturating_add(1);
}

#[derive(Default)]
pub(super) struct SdfProbeTraceNode;

impl ViewNode for SdfProbeTraceNode {
    type ViewQuery = &'static ViewTarget;

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        _view_target: QueryItem<Self::ViewQuery>,
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
        let Some(params_res) = world.get_resource::<DdgiParams>() else {
            return Ok(());
        };
        if !params_res.enabled {
            return Ok(());
        }

        let probe = world.resource::<SdfProbeTracePipeline>();
        let sdf = world.resource::<SdfPipeline>();
        let bufs = world.resource::<SdfProbeBuffers>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let device = render_context.render_device();

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(probe.pipeline_id) else {
            return Ok(());
        };
        let gpu_atlas = world.resource::<SdfGpuAtlas>();
        if gpu_atlas.pages.is_none()
            || gpu_atlas.material_buffer.is_none()
            || gpu_atlas.tex_array_views.is_none()
        {
            return Ok(());
        }
        if bufs.capacity == 0 || bufs.resident_count == 0 {
            return Ok(());
        }

        let Some(camera_uniforms) = world.get_resource::<ComponentUniforms<SdfCameraData>>() else {
            return Ok(());
        };
        let Some(camera_binding) = camera_uniforms.binding() else {
            return Ok(());
        };
        let Some(dyn_off) = world
            .get::<bevy::render::extract_component::DynamicUniformIndex<SdfCameraData>>(view_entity)
        else {
            return Ok(());
        };

        let layout_0 = pipeline_cache.get_bind_group_layout(&sdf.layout_0);
        let layout_1 = pipeline_cache.get_bind_group_layout(&sdf.layout_1);
        let layout_3 = pipeline_cache.get_bind_group_layout(&probe.layout_3);

        let bind_group_0 = device.create_bind_group(
            "sdf_probe_bind_group_0",
            &layout_0,
            &BindGroupEntries::sequential((camera_binding.clone(),)),
        );
        let bind_group_1 = atlas_bind_group_1(device, &layout_1, gpu_atlas, "sdf_probe_bind_group_1");
        let bind_group_3 = device.create_bind_group(
            "sdf_probe_bind_group_3",
            &layout_3,
            &BindGroupEntries::sequential((
                bufs.irr.as_entire_buffer_binding(),
                bufs.params.as_entire_buffer_binding(),
                bufs.resident.as_entire_buffer_binding(),
            )),
        );

        // One workgroup per (resident chunk × 64 local bricks) — the thread-per-texel trace puts a
        // whole probe-brick on one workgroup (64 threads = octahedral texels). Empty bricks early-out
        // on the occupancy-bit check. Tiled into a 2D grid (X capped at 65535).
        let rows = bufs.resident_count.max(1) * 64;
        let wg_x = rows.min(65535);
        let wg_y = rows.div_ceil(wg_x);

        let diagnostics = render_context.diagnostic_recorder();
        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("sdf_probe_trace"),
                timestamp_writes: None,
            });
        let span = diagnostics.pass_span(&mut pass, "sdf_probe_trace");
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group_0, &[dyn_off.index()]);
        pass.set_bind_group(1, &bind_group_1, &[]);
        pass.set_bind_group(2, &bind_group_3, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
        span.end(&mut pass);

        Ok(())
    }
}

/// Build the lit-pass apply bind group (group 3): the irradiance buffer (read) + params. Returned to
/// [`super::SdfCombineNode`] so the deferred lit shader can fold indirect light into the result.
pub(super) fn probe_apply_bind_group(
    device: &RenderDevice,
    pipeline_cache: &PipelineCache,
    world: &World,
) -> Option<BindGroup> {
    let probe = world.get_resource::<SdfProbeTracePipeline>()?;
    let bufs = world.get_resource::<SdfProbeBuffers>()?;
    let layout = pipeline_cache.get_bind_group_layout(&probe.layout_3_apply);
    Some(device.create_bind_group(
        "sdf_probe_apply_bind_group_3",
        &layout,
        &BindGroupEntries::sequential((
            bufs.irr.as_entire_buffer_binding(),
            bufs.params.as_entire_buffer_binding(),
        )),
    ))
}
