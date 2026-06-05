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
use crate::sdf_render::chunk::{ChunkLookup, TILE_RUN_SLOT};
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
    sky_intensity: f32,
    bounce_shadows: f32,
    /// Re-trace rate for converged (dormant) probes when `classify != 0` (1/dormant_stride per frame).
    dormant_stride: u32,
    /// 1 = classification active (converged probes go dormant); 0 = every probe traces at `update_stride`.
    /// Set 0 while the scene is unsettled (recent topology/lighting change) so nothing goes stale.
    classify: u32,
    /// LOD ≥ this → the probe traces `distant_ray_count` rays instead of `ray_count` (far field needs less).
    ray_falloff_lod: u32,
    distant_ray_count: u32,
}

#[derive(Resource)]
pub(super) struct SdfProbeShaderHandle(pub(super) Handle<Shader>);

/// The compact resident-chunk directory rows, extracted from the main world each frame. The trace
/// dispatches one workgroup per row.
#[derive(Resource, Default)]
pub(super) struct ExtractedResidentChunks {
    rows: Vec<ChunkLookup>,
}

/// Extract the FINEST-RESIDENT chunk list (one row per finest-resident chunk, those owning probe
/// blocks) for the compact trace dispatch. Filtering to finest here — not shipping all-LOD resident
/// rows and early-outing on the GPU — bounds the workgroup count to the clipmap window, not the
/// all-LOD union (the perf half of the scaling fix).
pub(super) fn extract_resident_chunks(atlas: Extract<Res<SdfAtlas>>, mut commands: Commands) {
    commands.insert_resource(ExtractedResidentChunks { rows: atlas.live_chunks.finest_rows() });
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
    /// Effective probe density the current buffer was sized for — recreate (+ reset history) when the
    /// adaptive `subdiv` changes, since it re-strides every brick's probe block.
    subdiv: u32,
    /// Set once the probe count was clamped to the budget (so the warning fires only once).
    clamped: bool,
    /// Last-seen [`ProbeReset`] counter — when it changes (a scene switch), the irradiance buffer is
    /// recreated (zeroed) so the new scene never inherits the previous scene's converged GI.
    last_reset: u32,
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
            layout_3.clone(),              // g2: irradiance + params + resident chunks
            sdf_pipeline.layout_3.clone(), // g3: point lights + light grid (shared w/ the G-buffer pass)
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
        subdiv: 0,
        clamped: false,
        last_reset: 0,
    });
}

/// Largest probe density `subdiv ∈ [1, desired]` whose buffer (`tiles × subdiv³ × oct` vec4 slots)
/// fits `cap_slots`. This is the graceful-degradation rule: a huge scene drops to a coarser probe
/// density so GI still covers the WHOLE scene, instead of clamping the buffer (which leaves entire
/// regions with no GI). Returns 1 if even subdiv 1 overflows — the caller then clamps as a last resort.
fn fit_probe_subdiv(tiles: u64, desired: u32, oct: u32, cap_slots: u64) -> u32 {
    let mut s = desired.max(1);
    while s > 1 && tiles * (s * s * s) as u64 * oct as u64 > cap_slots {
        s -= 1;
    }
    s
}

/// (Re)size the irradiance buffer to the tile-run capacity × subdiv³, upload the resident-chunk list +
/// params uniform. Runs in the Render schedule after the atlas extract set [`ChunkBufCapacity`].
#[allow(clippy::too_many_arguments)] // a render-world system: each param is a distinct resource/buffer
pub(super) fn prepare_sdf_probe(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cap: Res<ChunkBufCapacity>,
    params_res: Res<DdgiParams>,
    // Optional: extracted from the main world, but absent on the very first render frame before the
    // first `ExtractResourcePlugin` pass — treat as "not settled" (classify off) until it arrives.
    settle: Option<Res<crate::sdf_render::GiSettle>>,
    // Chunk slots whose region recently changed — these re-converge at the active rate even in a settled
    // (dormant) scene, so an edit only wakes its neighbourhood. Optional: absent before the first extract.
    wake_set: Option<Res<crate::sdf_render::ProbeWakeSet>>,
    // Bumped on a scene switch → recreate (zero) the irradiance buffer so the new scene never inherits
    // the previous scene's GI. Optional: absent before the first extract.
    reset_res: Option<Res<crate::sdf_render::ProbeReset>>,
    resident: Res<ExtractedResidentChunks>,
    mut bufs: ResMut<SdfProbeBuffers>,
) {
    // Probes may go dormant only once the scene has been SETTLED (no topology/lighting change) for at
    // least the convergence window — long enough for every probe to reach its sample cap at the active
    // re-trace rate. While unsettled, classification is off (full re-trace), so a change re-converges
    // immediately and slot-churn never shows stale GI.
    const CONVERGE_WINDOW: u32 = 192;
    let frames_unchanged = settle.map(|s| s.frames_unchanged).unwrap_or(0);
    let classify = params_res.classify_enabled && frames_unchanged > CONVERGE_WINDOW;
    let desired = params_res.subdiv.clamp(1, 4);
    let oct = crate::sdf_render::probe::PROBE_OCT_TEXELS;
    // Probe slots sized by the FINEST-RESIDENT probe high-water (`finest_chunks · CHUNK_VOLUME`) — one
    // compact block per finest chunk, brick at `probe_base + local`. Bounded by the clipmap window, not
    // the all-LOD atlas tile union. (`cap.probe_slots` is now `live.probe_high_water()`.)
    let tiles = cap.probe_slots.max(1) as u64;
    let budget_bytes = (params_res.probe_budget_bytes as u64)
        .min(device.limits().max_storage_buffer_binding_size as u64);
    let cap_slots = (budget_bytes / 16 / oct as u64).max(1) * oct as u64;
    // GRACEFUL DEGRADATION: largest density fitting the budget — whole-scene GI at lower density vs
    // clamping into GI holes. 310k bricks @ subdiv 2 = ~2.5 GB > the ~2 GB binding limit → subdiv 1.
    let subdiv = fit_probe_subdiv(tiles, desired, oct, cap_slots);
    let want = (tiles * (subdiv * subdiv * subdiv) as u64 * oct as u64).min(u32::MAX as u64) as u32;
    // Even at subdiv 1 over budget → clamp as the last resort (far probes inactive); the trace + apply
    // bounds-check `arrayLength(&irradiance)`, so it can't crash.
    let need = (want as u64).min(cap_slots) as u32;
    if subdiv < desired && !bufs.clamped {
        warn!(
            "DDGI auto-reduced probe density subdiv {desired}→{subdiv} to fit the {} MiB budget \
             ({tiles} occupied bricks) — GI covers the whole scene at lower density. Raise \
             DdgiParams.probe_budget_bytes for more.",
            budget_bytes / (1 << 20),
        );
        bufs.clamped = true;
    } else if want > need && !bufs.clamped {
        warn!(
            "DDGI probe buffer clamped even at subdiv 1: {tiles} occupied bricks exceed the {} MiB \
             budget — far probes inactive. Raise DdgiParams.probe_budget_bytes.",
            budget_bytes / (1 << 20),
        );
        bufs.clamped = true;
    }
    // A scene switch bumps `ProbeReset` → force a fresh (zeroed) buffer so the new scene never inherits
    // the old scene's converged irradiance from a reused slot (the grow-with-headroom buffer otherwise
    // keeps it). Sized to the new scene's `need` (so it also shrinks back from a big previous scene).
    let reset_id = reset_res.map(|r| r.0).unwrap_or(0);
    let reset_changed = reset_id != bufs.last_reset;
    // GROW WITH HEADROOM: DDGI is always on, and the atlas tile count GROWS every frame during a
    // multi-frame bake. Recreating the (potentially hundreds-of-MB) irradiance buffer every frame would
    // be a ~250 ms/frame hitch. So only recreate when the buffer is actually too small OR the adaptive
    // density (`subdiv`) changed (which re-strides every brick's block), allocating +25% slack (clamped
    // to the budget) so growth happens in a handful of steps, never shrinking — UNLESS a scene switch
    // asked for a clean zeroed buffer.
    if need > bufs.capacity || subdiv != bufs.subdiv || reset_changed {
        let new_cap = (need as u64 + need as u64 / 4).min(cap_slots).max(need as u64) as u32;
        // Fresh (zeroed) buffer — history starts empty, so frame 0 takes the traced value directly.
        bufs.irr = device.create_buffer(&BufferDescriptor {
            label: Some("sdf_probe_irr"),
            size: new_cap as u64 * 16,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        bufs.capacity = new_cap;
        bufs.subdiv = subdiv;
        bufs.frame = 0;
        bufs.last_reset = reset_id;
    }

    // DISPATCH-LEVEL AMORTIZATION (the dispatch-bound idle-cost fix): instead of dispatching a workgroup
    // for EVERY finest brick every frame (and round-robin-skipping in-shader, which still pays the
    // workgroup launch + occupancy read), only the SUBSET of finest chunks whose turn it is this frame is
    // uploaded + dispatched. Rotation key = the chunk's stable tile-run slot (`tile_run_base /
    // TILE_RUN_SLOT`), so over `eff_stride` frames every chunk is covered exactly once. `eff_stride` is
    // the active `update_stride` while the scene is settling, and the much larger `dormant_stride` once
    // CLASSIFY has determined the scene is converged + unchanged — so a static scene dispatches only
    // `finest_chunks / dormant_stride` workgroups per frame (the 6 ms → ~0.2 ms idle win).
    let active_stride = params_res.update_stride.max(1);
    let dormant_stride = if classify { params_res.dormant_stride.max(1) } else { active_stride };
    // Woken chunks (recently-changed region) re-converge at the active rate; everything else rotates at
    // the dormant rate once the scene is settled. So an edit's neighbourhood traces fast while the rest of
    // a big static scene stays cheap — localized wake, no global FPS cliff.
    let wake: std::collections::HashSet<u32> =
        wake_set.map(|w| w.slots.iter().copied().collect()).unwrap_or_default();
    let frame = bufs.frame;
    let mut bytes = Vec::with_capacity(resident.rows.len() / dormant_stride as usize * 24 + 24);
    let mut dispatched = 0u32;
    for c in &resident.rows {
        let slot = c.tile_run_base / TILE_RUN_SLOT;
        let s = if wake.contains(&slot) { active_stride } else { dormant_stride };
        if slot % s != frame % s {
            continue; // not this chunk's turn — it keeps its in-place irradiance (covered another frame)
        }
        for v in [c.key_hi, c.key_lo, c.occ_lo, c.occ_hi, c.tile_run_base, c.probe_base] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        dispatched += 1;
    }
    if bytes.is_empty() {
        bytes.resize(24, 0xff); // a sentinel-keyed row (never matches) so the trace finds no probes
    }
    bufs.resident = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_probe_resident"),
        contents: &bytes,
        usage: BufferUsages::STORAGE,
    });
    bufs.resident_count = dispatched.max(1);

    let p = ProbeParams {
        ray_count: params_res.ray_count.max(1),
        hysteresis: params_res.hysteresis.clamp(0.0, 0.99),
        intensity: params_res.intensity.max(0.0),
        frame: bufs.frame,
        subdiv,
        // Amortization is now at the dispatch level (the chunk subset above), so the in-shader
        // round-robin / dormancy is disabled (every dispatched probe traces): pass stride 1, classify 0.
        update_stride: 1,
        gi_range: params_res.gi_range.max(1.0),
        normal_bias: params_res.normal_bias.max(0.0),
        view_bias: params_res.view_bias.max(0.0),
        sky_intensity: params_res.gi_sky_intensity.max(0.0),
        bounce_shadows: if params_res.gi_bounce_shadows { 1.0 } else { 0.0 },
        dormant_stride: 1,
        classify: 0,
        ray_falloff_lod: params_res.ray_falloff_lod,
        distant_ray_count: params_res.distant_ray_count.max(1),
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
        let layout_2 = pipeline_cache.get_bind_group_layout(&probe.layout_3);
        let layout_lights = pipeline_cache.get_bind_group_layout(&sdf.layout_3);

        let bind_group_0 = device.create_bind_group(
            "sdf_probe_bind_group_0",
            &layout_0,
            &BindGroupEntries::sequential((camera_binding.clone(),)),
        );
        let bind_group_1 = atlas_bind_group_1(device, &layout_1, gpu_atlas, "sdf_probe_bind_group_1");
        let bind_group_2 = device.create_bind_group(
            "sdf_probe_bind_group_2",
            &layout_2,
            &BindGroupEntries::sequential((
                bufs.irr.as_entire_buffer_binding(),
                bufs.params.as_entire_buffer_binding(),
                bufs.resident.as_entire_buffer_binding(),
            )),
        );
        // Group 3: the SAME point-light + world-grid buffers the G-buffer pass binds, so the bounce
        // shades point lights identically (the layout is declared FRAGMENT|COMPUTE for exactly this).
        // Dummies are seeded in init, so these are Some before the first trace dispatch.
        let gpu_lights = world.resource::<SdfGpuLights>();
        let bind_group_lights = device.create_bind_group(
            "sdf_probe_bind_group_lights",
            &layout_lights,
            &BindGroupEntries::sequential((
                gpu_lights.point_buffer.as_ref().unwrap().as_entire_buffer_binding(),
                gpu_lights.cell_buffer.as_ref().unwrap().as_entire_buffer_binding(),
                gpu_lights.index_buffer.as_ref().unwrap().as_entire_buffer_binding(),
            )),
        );

        // One workgroup per (FINEST-resident chunk × 64 local bricks) — the thread-per-texel trace puts
        // a whole probe-brick on one workgroup (64 threads = octahedral texels). Empty bricks early-out
        // on the occupancy-bit check. The list is already filtered to finest-resident chunks (the
        // dispatch is bounded by the clipmap window). Tiled into a 2D grid (X capped at 65535).
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
        pass.set_bind_group(2, &bind_group_2, &[]);
        pass.set_bind_group(3, &bind_group_lights, &[]);
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

#[cfg(test)]
mod tests {
    use super::fit_probe_subdiv;
    use crate::sdf_render::probe::PROBE_OCT_TEXELS;

    /// Graceful degradation: probe density drops to fit the budget (GI stays whole) rather than the
    /// buffer clamping (GI holes). Mirrors the LOD-8 reality found on the live 310k-brick scene.
    #[test]
    fn adaptive_subdiv_degrades_to_fit_budget() {
        let oct = PROBE_OCT_TEXELS;
        let cap_1gib = (1u64 << 30) / 16; // budget in vec4 slots

        // 310k bricks (the live gallery): subdiv 2 = ~2.5 GiB > 1 GiB → auto-drop to subdiv 1 (~317 MiB).
        assert_eq!(fit_probe_subdiv(310_292, 2, oct, cap_1gib), 1);
        // A small scene keeps the desired density.
        assert_eq!(fit_probe_subdiv(1_000, 4, oct, cap_1gib), 4);
        // Never below 1 (the caller clamps the buffer as the last resort beyond this).
        assert_eq!(fit_probe_subdiv(100_000_000, 4, oct, cap_1gib), 1);
        // A bigger budget keeps a higher density on the same scene.
        assert_eq!(fit_probe_subdiv(310_292, 2, oct, (4u64 << 30) / 16), 2);
    }
}
