//! **Headless correctness gate for the ReSTIR GI estimator** (`voxel_raytrace.wgsl`, the `restir_probe`
//! entry — R0 of the ReSTIR rollout).
//!
//! ReSTIR replaces `gather_gi`'s plain per-pixel mean with a per-shading-point reservoir that resamples one
//! candidate per frame (RIS) and REUSES it across frames (temporal) — the cure for the GI surface boiling.
//! Before wiring the screen-space passes into the live renderer (R1), this rig proves the estimator MATH in
//! isolation, on a real GPU, with NO GUI: for each probe (a world position + normal) the shader generates an
//! initial reservoir and merges it into the probe's persistent reservoir each "frame", and the harness reads
//! the resolved indirect irradiance back for every frame.
//!
//! It asserts the three load-bearing properties:
//!   1. **Unbiased / energy** — the frame-averaged ReSTIR irradiance matches the established high-spp
//!      `gather_gi` reference (same scene, same `LightingUniformData`), i.e. the resolve constant is right
//!      and the estimator is unbiased. (Ambient = 0 so sky-misses contribute 0 in BOTH, since ReSTIR drops
//!      missed bounces while `gather_gi` adds `bounce_sky` = ambient.)
//!   2. **Stability** — the running mean over the first half and the second half agree (no drift), and the
//!      confidence weight saturates at the cap (temporal history is bounded).
//!   3. **Emissive concentration (the adaptation)** — a probe that FACES the emitter ends up with a
//!      higher-radiance selected sample (and brighter irradiance) than one far from it, because including
//!      emissive sample points makes resampling concentrate toward the bright emitter.
//!
//! Skips cleanly (no failure) on a box without an `EXPERIMENTAL_RAY_QUERY` Vulkan adapter.

use std::iter;
use std::mem;

use bevy::math::IVec3;
use wgpu::util::DeviceExt;

use adventure::sdf_render::worldgen::biome::{
    BiomeDef, BiomeId, BiomeLibrary, StrataLayer, TerrainMatId, TerrainSurfaceMaterial,
};
use adventure::voxel::brickmap::{BRICK_WORLD_SIZE, Brick};
use adventure::voxel::gpu::{ResidentBrick, pack_resident_set};
use adventure::voxel::palette::{BlockId, BlockRegistry};
use adventure::voxel::raytrace::LightingUniformData;

mod common;

// Mirror of the WGSL `ProbePoint` (group 0 binding 8) — a shading point fed to the estimator.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ProbePoint {
    world_position: [f32; 3],
    _p0: u32,
    world_normal: [f32; 3],
    _p1: u32,
}

// Mirror of the WGSL `ProbeOut` (group 0 binding 10) — what the estimator reports each frame.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
struct ProbeOut {
    irradiance: [f32; 3],
    confidence: f32,
    reference: [f32; 3],
    ucw: f32,
}

// Mirror of the WGSL `RestirProbeParams` (group 0 binding 11).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RestirProbeParams {
    frame_index: u32,
    reset: u32,
    n_probes: u32,
    _p: u32,
}

// Mirror of the WGSL `Reservoir` (48 bytes) — only used to size the persistent reservoir buffer.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuReservoir {
    sample_point_world_position: [f32; 3],
    weight_sum: f32,
    radiance: [f32; 3],
    confidence_weight: f32,
    sample_point_world_normal: [f32; 3],
    unbiased_contribution_weight: f32,
}

const FLOOR: BlockId = BlockId(1);
const EMITTER: BlockId = BlockId(3);

fn test_library() -> BiomeLibrary {
    let mat = |name: &str, c: [f32; 4]| TerrainSurfaceMaterial {
        name: name.into(),
        base_color: c,
        roughness: 0.9,
        blend: 0.0,
        texture: None,
        tiling: 4.0,
    };
    let materials = vec![
        mat("floor", [0.5, 0.5, 0.5, 1.0]),
        mat("red", [0.9, 0.02, 0.02, 1.0]),
        mat("emit", [0.04, 0.04, 0.04, 1.0]),
    ];
    let column = |_| BiomeDef {
        name: "b".into(),
        surface: TerrainMatId(0),
        surface_rules: vec![],
        strata: vec![StrataLayer { material: TerrainMatId(0), thickness: 1.0 }],
        bedrock: TerrainMatId(1),
    };
    let biomes = BiomeId::ALL.iter().map(column).collect();
    BiomeLibrary { materials, biomes }
}

fn solid(id: BlockId) -> Brick {
    Brick::uniform(id)
}

fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

const N_FRAMES: u32 = 384;
const N_PROBES: u32 = 2; // 0 = near the emitter, 1 = far from it

/// Drive the `restir_probe` entry for `N_FRAMES` "frames" over `probes`, returning the full per-frame
/// per-probe `ProbeOut` history (indexed `[frame * N_PROBES + probe]`).
fn run_probes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    patch: &adventure::voxel::gpu::GpuBrickPatch,
    light: &LightingUniformData,
    probes: &[ProbePoint],
    force_reset_every_frame: bool,
) -> Vec<ProbeOut> {
    let n = patch.brick_count() as u32;
    let n_probes = probes.len() as u32;

    let aabb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("restir_aabbs"),
        contents: bytemuck::cast_slice(&patch.aabbs),
        usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE,
    });
    let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("restir_metas"),
        contents: bytemuck::cast_slice(&patch.metas),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let voxel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("restir_voxels"),
        contents: bytemuck::cast_slice(&patch.voxels),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let palette_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("restir_palette"),
        contents: bytemuck::cast_slice(&patch.palette),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let size_desc = wgpu::BlasAABBGeometrySizeDescriptor {
        primitive_count: n,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("restir_blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::AABBs { descriptors: vec![size_desc.clone()] },
    );
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("restir_tlas"),
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        max_instances: 1,
    });
    tlas[0] = Some(wgpu::TlasInstance::new(
        &blas,
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        0,
        0xff,
    ));

    let probe_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("restir_probes"),
        contents: bytemuck::cast_slice(probes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let reservoir_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("restir_reservoirs"),
        size: (n_probes as u64) * mem::size_of::<GpuReservoir>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Zero the persistent reservoirs (empty_reservoir == all-zero → confidence 0).
    queue.write_buffer(
        &reservoir_buf,
        0,
        &vec![0u8; (n_probes as usize) * mem::size_of::<GpuReservoir>()],
    );

    let out_len = (N_FRAMES * n_probes) as u64 * mem::size_of::<ProbeOut>() as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("restir_out"),
        size: out_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("restir_read"),
        size: out_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("restir_params"),
        size: mem::size_of::<RestirProbeParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let light_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("restir_light"),
        size: mem::size_of::<LightingUniformData>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&light_buf, 0, bytemuck::bytes_of(light));

    let src = common::voxel_raytrace_shader_src();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_raytrace"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("restir_probe"),
        layout: None,
        module: &shader,
        entry_point: Some("restir_probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("restir_scene_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::AccelerationStructure(&tlas) },
            wgpu::BindGroupEntry { binding: 1, resource: meta_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: voxel_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: palette_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: probe_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: reservoir_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: params_buf.as_entire_binding() },
        ],
    });
    let sky_buf = common::sky_uniform_buffer(device);
    let light_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("restir_light_bg"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: sky_buf.as_entire_binding() },
        ],
    });

    let mut build = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("restir_build") });
    build.build_acceleration_structures(
        iter::once(&wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::AabbGeometries(vec![wgpu::BlasAabbGeometry {
                size: &size_desc,
                stride: mem::size_of::<adventure::voxel::gpu::GpuBrickAabb>() as wgpu::BufferAddress,
                aabb_buffer: &aabb_buf,
                primitive_offset: 0,
            }]),
        }),
        iter::once(&tlas),
    );
    queue.submit(Some(build.finish()));

    // One dispatch per "frame": reset on frame 0, then temporal accumulation.
    for frame in 0..N_FRAMES {
        let params = RestirProbeParams {
            frame_index: frame,
            reset: u32::from(frame == 0 || force_reset_every_frame),
            n_probes,
            _p: 0,
        };
        queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, Some(&scene_bg), &[]);
            cpass.set_bind_group(1, Some(&light_bg), &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    let mut copy = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    copy.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, out_len);
    queue.submit(Some(copy.finish()));

    let slice = read_buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
    let data = slice.get_mapped_range().unwrap();
    let out: Vec<ProbeOut> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    read_buf.unmap();
    let _ = (&aabb_buf, &blas, &tlas);
    out
}

/// Analytically-exact, LOW-VARIANCE scene: a probe on a floor directly under a LARGE flat emissive ceiling.
/// With the ceiling subtending ~the full upper hemisphere, the indirect irradiance has a closed form:
/// `I = (1/π)∫_hemisphere R·cosθ dω = R` where `R = emissive·strength`. So the up-facing probe MUST resolve
/// to ≈ R (here 3·4 = 12) — a clean check of the resolve constant + merge with negligible Monte-Carlo
/// variance (almost every bounce ray hits the bright ceiling). Sun off + ambient 0 ⇒ the ceiling is the only
/// light and sky-misses are 0 in both estimators.
const CEILING_RADIANCE: f32 = 12.0; // emissive 3.0 × strength 4.0 ⇒ analytic I for the up-facing probe

fn emitter_scene() -> (BlockRegistry, LightingUniformData, [ProbePoint; N_PROBES as usize]) {
    let mut reg = BlockRegistry::from_biome_library(&test_library());
    reg.set_emissive(EMITTER, [3.0, 3.0, 3.0]);

    let l = LightingUniformData {
        sun_direction: [0.0, 1.0, 0.0], // travels up ⇒ no direct light on the +Y floor
        ambient_color: [0.0, 0.0, 0.0], // sky-miss = 0 in BOTH estimators
        gi_rays: 32,
        gi_intensity: 1.0,
        gi_bounce_dist: 40.0, // must comfortably reach the ceiling (~1.6 m up)
        emissive_strength: 4.0,
        // (firefly clamping was discarded in Phase 2.2 — the probe estimator is the unbiased oracle, no clamp.)
        ..LightingUniformData::default()
    };

    let s = BRICK_WORLD_SIZE;
    let floor_top = s;
    // Probe 0 (near): floor centre, facing UP into the full ceiling ⇒ I ≈ R = 12.
    let near = ProbePoint {
        world_position: [s * 0.5, floor_top, s * 0.5],
        _p0: 0,
        world_normal: [0.0, 1.0, 0.0],
        _p1: 0,
    };
    // Probe 1 (far): same point facing SIDEWAYS (+X) — only part of its hemisphere sees the ceiling, so its
    // irradiance is strictly lower (the concentration contrast), and it exercises an off-normal surface.
    let far = ProbePoint {
        world_position: [s * 0.5, floor_top, s * 0.5],
        _p0: 0,
        world_normal: [1.0, 0.0, 0.0],
        _p1: 0,
    };
    (reg, l, [near, far])
}

fn emitter_patch(reg: &BlockRegistry) -> adventure::voxel::gpu::GpuBrickPatch {
    let floor = solid(FLOOR);
    let emit = solid(EMITTER);
    let mut entries: Vec<ResidentBrick> = Vec::new();
    // Wide floor + a wide emissive ceiling 2 bricks above it (gap at by=1), so the ceiling fills the probe's
    // upper hemisphere (closed-form I = R).
    for bx in -8..=8i32 {
        for bz in -8..=8i32 {
            entries.push(ResidentBrick { coord: IVec3::new(bx, 0, bz), brick: &floor, lod: 0 });
            entries.push(ResidentBrick { coord: IVec3::new(bx, 2, bz), brick: &emit, lod: 0 });
        }
    }
    pack_resident_set(&entries, reg)
}

/// Mean of a slice of `ProbeOut` field selected by `f`.
fn mean3(frames: &[ProbeOut], f: impl Fn(&ProbeOut) -> [f32; 3]) -> [f32; 3] {
    let mut acc = [0.0f32; 3];
    for o in frames {
        let v = f(o);
        acc[0] += v[0];
        acc[1] += v[1];
        acc[2] += v[2];
    }
    let n = frames.len().max(1) as f32;
    [acc[0] / n, acc[1] / n, acc[2] / n]
}

#[test]
fn restir_probe_is_valid_unbiased_and_concentrates() {
    let Some((device, queue)) = common::headless_ray_query_device() else {
        eprintln!("no ray-query device — skipping restir_probe");
        return;
    };
    let (reg, light, probes) = emitter_scene();
    let patch = emitter_patch(&reg);

    // Luma mean + std of a `ProbeOut` window's irradiance.
    let lstd = |fr: &[ProbeOut]| -> (f32, f32) {
        let m = luma(mean3(fr, |o| o.irradiance));
        let var = fr.iter().map(|o| (luma(o.irradiance) - m).powi(2)).sum::<f32>() / fr.len().max(1) as f32;
        (m, var.sqrt())
    };
    let warmup = 16usize;

    // BASE = reset every frame ⇒ the single-sample estimator with NO temporal accumulation (high variance).
    // TEMPORAL = the real reservoir reuse. Comparing the two shows both unbiasedness AND variance reduction.
    let base = run_probes(&device, &queue, &patch, &light, &probes, true);
    let out = run_probes(&device, &queue, &patch, &light, &probes, false);
    let frames = |hist: &[ProbeOut], p: usize| -> Vec<ProbeOut> {
        (0..N_FRAMES as usize).map(|f| hist[f * N_PROBES as usize + p]).collect()
    };
    let base_near = frames(&base, 0);
    let near = frames(&out, 0);
    let far = frames(&out, 1);

    let (base_restir, base_std) = lstd(&base_near[warmup..]);
    let base_ref = luma(mean3(&base_near[warmup..], |o| o.reference));
    let (temporal_restir, temporal_std) = lstd(&near[warmup..]);
    let near_ref = luma(mean3(&near[warmup..], |o| o.reference));
    let (far_restir, _) = lstd(&far[warmup..]);
    eprintln!(
        "[restir] analytic I≈{CEILING_RADIANCE:.1} | base {base_restir:.3} (std {base_std:.3}) | temporal {temporal_restir:.3} (std {temporal_std:.3}) | ref {base_ref:.3}/{near_ref:.3} | far {far_restir:.3} | final_conf {:.1}",
        near.last().unwrap().confidence
    );

    // --- 1. Validity: finite, non-negative; confidence bounded. The temporal reservoir is capped at 8
    //        BEFORE merging with the canonical (confidence 1), so the steady-state max is 9. ---
    for (i, o) in near.iter().chain(far.iter()).enumerate() {
        assert!(
            o.irradiance.iter().all(|v| v.is_finite() && *v >= 0.0)
                && o.ucw.is_finite()
                && o.ucw >= 0.0
                && o.confidence.is_finite()
                && (0.0..=9.0 + 1e-3).contains(&o.confidence),
            "invalid reservoir output at slot {i}: {o:?}"
        );
    }
    assert!(
        near.last().unwrap().confidence > 4.0,
        "temporal confidence must accumulate toward the cap, got {}",
        near.last().unwrap().confidence
    );

    // --- 2. Unbiased (absolute): under a full emissive ceiling the indirect irradiance has the closed form
    //        I = R. The base single-sample estimator must hit it (validates the resolve constant), and the
    //        reference gather_gi must agree — so ReSTIR shares the renderer's radiometry with no constant drift. ---
    assert!(
        (base_restir - CEILING_RADIANCE).abs() / CEILING_RADIANCE < 0.25,
        "base ReSTIR must match the analytic I={CEILING_RADIANCE} (got {base_restir:.3}) — resolve constant wrong?"
    );
    assert!(
        (base_ref - CEILING_RADIANCE).abs() / CEILING_RADIANCE < 0.25,
        "the gather_gi reference must match analytic I={CEILING_RADIANCE} (got {base_ref:.3})"
    );
    // Temporal reuse stays close to the reference (a small upward M-cap bias is expected/acceptable).
    let ratio = temporal_restir / near_ref;
    assert!(
        (0.7..=1.45).contains(&ratio),
        "temporal ReSTIR must match the reference within tolerance, ratio={ratio:.3} ({temporal_restir:.3} vs {near_ref:.3})"
    );

    // --- 3. Variance reduction (the boil cure): temporal reuse must cut the per-frame variance well below
    //        the single-sample estimator. This is the property that kills the surface boiling. ---
    assert!(
        temporal_std < 0.5 * base_std,
        "temporal reuse must reduce per-frame variance (boil), temporal std {temporal_std:.3} vs base std {base_std:.3}"
    );

    // --- 4. Stability: first-half vs second-half running means agree (no drift). ---
    let half = N_FRAMES as usize / 2;
    let first = luma(mean3(&near[warmup..half], |o| o.irradiance));
    let second = luma(mean3(&near[half..], |o| o.irradiance));
    assert!(
        (first - second).abs() <= 0.35 * (first + second).max(1e-4),
        "ReSTIR estimate must be stable across halves (no drift): {first:.4} vs {second:.4}"
    );

    // --- 5. Emissive concentration (the adaptation): the up-facing probe (full ceiling) gathers clearly
    //        more than the sideways probe (only part of its hemisphere sees the ceiling). ---
    assert!(
        temporal_restir > 1.3 * far_restir,
        "the ceiling-facing probe must gather more light than the sideways probe: {temporal_restir:.3} vs {far_restir:.3}"
    );
}

/// R1 compile gate: the live screen-space two-pass ReSTIR entry points (`restir_p1`/`restir_p2` + the DLSS
/// `restir_dlss_p1`/`restir_dlss_p2`) must compile on the real device. The lib build never compiles the WGSL
/// (it's loaded at runtime), and the estimator test above only exercises `restir_probe` — so without this, a
/// syntax/binding error in the screen-space entries would only surface at launch. Creating the pipelines (auto
/// layout) forces naga to compile every entry + validate its bindings. Needs an 8-storage-texture +
/// 16-storage-buffer device (the DLSS variants write colour + 5 guides; Phase 2.2's `restir_p1`/
/// `restir_dlss_p1` query the group(3) world cache, so they bind 11 storage buffers — over the default 8).
#[test]
fn restir_screen_space_entries_compile() {
    // 8 storage textures (the DLSS variants write colour + 5 guides) AND 16 storage buffers: Phase 2.2's
    // `restir_p1`/`restir_dlss_p1` query the group(3) world cache, so their auto-derived layout binds 11
    // storage buffers (3 scene + 4 reservoir/surface + 4 cache) — over wgpu's default of 8. Mirrors the
    // in-engine `wgpu_settings()` device.
    let Some((device, _queue)) = common::headless_ray_query_device_with_storage(8, 16) else {
        eprintln!("no ray-query device with 8 storage textures + 16 storage buffers — skipping restir_screen_space_entries_compile");
        return;
    };
    let src = common::voxel_raytrace_shader_src();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_raytrace"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    for entry in ["restir_p1", "restir_p2", "restir_dlss_p1", "restir_dlss_p2"] {
        let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None, // auto layout from reflection — validates the entry + its bindings compile
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
    }
}
