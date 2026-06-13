//! **Headless correctness gate for the WORLD-SPACE RADIANCE CACHE subsystem** (Phase 2.1;
//! `voxel_raytrace.wgsl` `world_cache_*` entries + `query_world_cache`).
//!
//! The cache stores pre-accumulated outgoing radiance per (quantized world position + normal) in a GPU hash
//! grid, refreshed each frame by the six-pass compute loop ported from `bevy_solari::world_cache_*` and
//! adapted to our tracer (no light list → the update pass traces ONE cosine bounce and gathers
//! `direct_lighting + emissive` / `sky`). In Phase 2.1 the cache RUNS but is NOT read by the live image, so
//! this rig is the proof the subsystem is correct end-to-end, WITHOUT a GUI.
//!
//! It drives the FULL pass sequence for N frames against a known scene — a floor cell directly under a LARGE
//! flat emissive ceiling, the same analytically-exact setup the ReSTIR probe oracle uses — and asserts:
//!   1. **Insert + probe** — the seeded cell's checksum becomes non-empty and its life is the full lifetime
//!      (the lazy-insert + alive-mark worked, and the linear probe re-finds the SAME slot every frame).
//!   2. **Becomes non-zero** — the cell's stored radiance rises above 0 (the update + blend ran).
//!   3. **Stabilises** — the frame-to-frame variance of the radiance falls (the temporal blend converges).
//!   4. **Matches the analytic single-bounce irradiance** — under a full emissive ceiling the outgoing
//!      radiance of an up-facing floor cell has the closed form `R = emissive·strength`, so the converged
//!      cell radiance ≈ R (here 3·4 = 12) within tolerance — the same oracle convention as `restir_probe` /
//!      a high-spp `gather_gi` gather.
//!
//! Uses a SMALL hash table (2^12) via the production `voxel_raytrace_shader_src(size)` SSOT loader so the
//! whole-table decay/compaction passes stay fast. Skips cleanly on a box without an `EXPERIMENTAL_RAY_QUERY`
//! Vulkan adapter (or one that can't reach the storage-buffer count the cache binds).

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
use adventure::voxel::raytrace::{LightingUniformData, SkyUniformData, WorldCacheUniformData};

mod common;

/// A small cache hash table for the test — four 1024-blocks, exercising the two-level prefix-sum compaction
/// (single-block + block-scan) at a fraction of the live 2^20 cost.
const TEST_WORLD_CACHE_SIZE: u32 = 1 << 12;

const FLOOR: BlockId = BlockId(1);
const EMITTER: BlockId = BlockId(3);

/// Analytic outgoing radiance of an up-facing floor cell under a full emissive ceiling: with the ceiling
/// subtending ~the whole upper hemisphere and `direct_lighting` of it ≈ 0 (sun off, ambient 0), every cosine
/// bounce returns `emissive·strength`, so the cosine-weighted gather = `R = 3·4 = 12`. Same convention as the
/// ReSTIR probe oracle.
const CEILING_RADIANCE: f32 = 12.0;

fn test_library() -> BiomeLibrary {
    let mat = |name: &str, c: [f32; 4]| TerrainSurfaceMaterial {
        name: name.into(),
        base_color: c,
        roughness: 0.9,
        blend: 0.0,
        texture: None,
        tiling: 4.0,
        ..Default::default()
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

// Mirror of the WGSL `WcQueryPoint` (group 3 binding 12).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WcQueryPoint {
    world_position: [f32; 3],
    _p0: u32,
    world_normal: [f32; 3],
    _p1: u32,
}

// Mirror of the WGSL `WcQueryOut` (group 3 binding 13).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
struct WcQueryOut {
    radiance: [f32; 3],
    cell_index: u32,
    checksum: u32,
    life: u32,
    _p0: u32,
    _p1: u32,
}

// Mirror of the WGSL `WcQueryParams` (group 3 binding 14).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WcQueryParams {
    view_position: [f32; 3],
    n_points: u32,
    frame_index: u32,
    /// The bounce distance fed to `query_world_cache` (drives the first-bounce light-leak-prevention clamp).
    /// Zero for the convergence/energy fill loops (they use the no-jitter LOD0 seed, which ignores it).
    ray_t: f32,
    _p1: u32,
    _p2: u32,
}

const N_FRAMES: u32 = 64;

/// Build the wide-floor + wide-emissive-ceiling scene (a 2-brick gap) so the probe cell's upper hemisphere is
/// filled by the emitter (closed-form outgoing radiance = R). Mirrors the ReSTIR oracle scene.
fn emitter_patch(reg: &BlockRegistry) -> adventure::voxel::gpu::GpuBrickPatch {
    let floor = solid(FLOOR);
    let emit = solid(EMITTER);
    let mut entries: Vec<ResidentBrick> = Vec::new();
    for bx in -8..=8i32 {
        for bz in -8..=8i32 {
            entries.push(ResidentBrick { coord: IVec3::new(bx, 0, bz), brick: &floor, lod: 0 });
            entries.push(ResidentBrick { coord: IVec3::new(bx, 2, bz), brick: &emit, lod: 0 });
        }
    }
    pack_resident_set(&entries, reg)
}

#[test]
fn world_cache_converges_to_single_bounce_irradiance() {
    // The cache binds 3 scene storage buffers (group 0) + 12 cache storage buffers (group 3, including the
    // two test-only seed buffers) in one pipeline layout = 15 in a single stage; raise the limit accordingly.
    let Some((device, queue)) = common::headless_ray_query_device_with_storage_buffers(16) else {
        eprintln!("no ray-query device with 16 storage buffers — skipping world_cache test");
        return;
    };

    let mut reg = BlockRegistry::from_biome_library(&test_library());
    reg.set_emissive(EMITTER, [3.0, 3.0, 3.0]);
    let patch = emitter_patch(&reg);
    let n = patch.brick_count() as u32;

    // Lighting: sun off (travels up ⇒ no direct on the +Y floor), ambient 0 ⇒ the ceiling is the only light,
    // so the cell's gathered radiance is exactly the emissive. Bounce reaches the ceiling (~1.6 m up).
    let light = LightingUniformData {
        sun_direction: [0.0, 1.0, 0.0],
        ambient_color: [0.0, 0.0, 0.0],
        gi_rays: 1,
        gi_intensity: 1.0,
        gi_bounce_dist: 40.0,
        emissive_strength: 4.0,
        // (firefly clamping discarded in Phase 2.2 — the cache update is unclamped, matching Solari sample_gi.)
        ..LightingUniformData::default()
    };
    // Dark sky so a (rare) sideways/over-the-edge bounce miss adds nothing — the radiance is the ceiling alone.
    let sky = SkyUniformData {
        horizon_color: [0.0, 0.0, 0.0],
        zenith_color: [0.0, 0.0, 0.0],
        ground_color: [0.0, 0.0, 0.0],
        sun_size: 0.0,
        intensity: 0.0,
        gi_sky_intensity: 0.0,
        sun_tint: [0.0, 0.0, 0.0],
        _pad: 0.0,
    };
    // Cache knobs: a generous bounce distance (must reach the ceiling) + a small cell so the floor probe maps
    // to one stable cell. cell_lifetime comfortably exceeds 1 so the seeded cell never decays between frames.
    let wc_defaults = WorldCacheUniformData {
        cell_base_size: 0.3,
        gi_ray_distance: 40.0,
        cell_lifetime: 8,
        ..WorldCacheUniformData::default()
    };

    let s = BRICK_WORLD_SIZE;
    let floor_top = s;
    // The probe: a floor cell facing UP into the full emissive ceiling ⇒ outgoing radiance ≈ R.
    let probe = WcQueryPoint {
        world_position: [s * 0.5, floor_top, s * 0.5],
        _p0: 0,
        world_normal: [0.0, 1.0, 0.0],
        _p1: 0,
    };
    let probes = [probe];
    let n_points = probes.len() as u32;
    let view_position = [s * 0.5, floor_top + 3.0, s * 0.5]; // a near camera (LOD 0)

    // --- Scene (group 0) GPU objects ---
    let aabb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wc_aabbs"),
        contents: bytemuck::cast_slice(&patch.aabbs),
        usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE,
    });
    let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wc_metas"),
        contents: bytemuck::cast_slice(&patch.metas),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let voxel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wc_voxels"),
        contents: bytemuck::cast_slice(&patch.voxels),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let palette_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wc_palette"),
        contents: bytemuck::cast_slice(&patch.palette),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let size_desc = wgpu::BlasAABBGeometrySizeDescriptor {
        primitive_count: n,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("wc_blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::AABBs { descriptors: vec![size_desc.clone()] },
    );
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("wc_tlas"),
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

    // --- Persistent cache buffers (zero-initialised → all cells empty) ---
    let tsz = TEST_WORLD_CACHE_SIZE as u64;
    let zeroed = |label: &str, bytes: u64, indirect: bool| {
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        if indirect {
            usage |= wgpu::BufferUsages::INDIRECT;
        }
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, &vec![0u8; bytes as usize]);
        buf
    };
    let checksums = zeroed("wc_checksums", tsz * 4, false);
    let life = zeroed("wc_life", tsz * 4, false);
    let radiance = zeroed("wc_radiance", tsz * 16, false);
    let geometry = zeroed("wc_geometry", tsz * 32, false);
    let luminance_deltas = zeroed("wc_luminance_deltas", tsz * 4, false);
    let new_radiance = zeroed("wc_new_radiance", tsz * 16, false);
    let a = zeroed("wc_a", tsz * 4, false);
    let b = zeroed("wc_b", 1024 * 4, false);
    let active_cell_indices = zeroed("wc_active_cell_indices", tsz * 4, false);
    let active_cells_count = zeroed("wc_active_cells_count", 4, false);
    let active_cells_dispatch = zeroed("wc_active_cells_dispatch", 12, true);

    // --- Per-frame uniforms ---
    let wc_uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wc_uniform"),
        size: mem::size_of::<WorldCacheUniformData>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let light_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wc_light"),
        contents: bytemuck::bytes_of(&light),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let sky_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wc_sky"),
        contents: bytemuck::bytes_of(&sky),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // --- Seed (test) buffers ---
    let query_points_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wc_query_points"),
        contents: bytemuck::cast_slice(&probes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let query_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wc_query_out"),
        size: (n_points as u64) * mem::size_of::<WcQueryOut>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let query_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wc_query_params"),
        size: mem::size_of::<WcQueryParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wc_read"),
        size: (n_points as u64) * mem::size_of::<WcQueryOut>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // --- Pipelines (one shared pipeline layout: scene(0) + view(1: light,sky) + empty(2) + cache(3)) ---
    let src = adventure::voxel::raytrace::voxel_raytrace_shader_src(TEST_WORLD_CACHE_SIZE);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_raytrace"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });

    let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("wc_scene_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
                count: None,
            },
            storage_ro(1),
            storage_ro(2),
            storage_ro(3),
        ],
    });
    let view_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("wc_view_layout"),
        entries: &[uniform(2), uniform(11)],
    });
    // group(2): the indirect-dispatch buffer — bound for seed+compaction, UNBOUND for update/blend (it can't
    // be bound storage AND used as indirect args in one scope). Mirrors the production split.
    let dispatch_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("wc_dispatch_layout"),
        entries: &[storage_rw(0)],
    });
    let cache_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("wc_cache_layout"),
        entries: &[
            uniform(0),
            storage_rw(1),
            storage_rw(2),
            storage_rw(3),
            storage_rw(4),
            storage_rw(5),
            storage_rw(6),
            storage_rw(7),
            storage_rw(8),
            storage_rw(9),
            storage_rw(10),
            storage_ro(12),
            storage_rw(13),
            uniform(14),
        ],
    });
    // Layout A — seed + decay + 3 compaction passes (group 2 = dispatch present).
    let compact_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wc_compact_pl"),
        bind_group_layouts: &[Some(&scene_layout), Some(&view_layout), Some(&dispatch_layout), Some(&cache_layout)],
        immediate_size: 0,
    });
    // Layout B — update + blend (group 2 absent so the dispatch buffer is unbound when used as indirect args).
    let update_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wc_update_pl"),
        bind_group_layouts: &[Some(&scene_layout), Some(&view_layout), None, Some(&cache_layout)],
        immediate_size: 0,
    });
    let mk = |entry: &str, layout: &wgpu::PipelineLayout| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let p_seed = mk("world_cache_query_seed", &compact_pl);
    let p_decay = mk("world_cache_decay", &compact_pl);
    let p_csb = mk("world_cache_compact_single_block", &compact_pl);
    let p_cb = mk("world_cache_compact_blocks", &compact_pl);
    let p_cwa = mk("world_cache_compact_write_active", &compact_pl);
    let p_update = mk("world_cache_update", &update_pl);
    let p_blend = mk("world_cache_blend", &update_pl);

    // --- Bind groups ---
    let scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wc_scene_bg"),
        layout: &scene_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::AccelerationStructure(&tlas) },
            wgpu::BindGroupEntry { binding: 1, resource: meta_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: voxel_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: palette_buf.as_entire_binding() },
        ],
    });
    let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wc_view_bg"),
        layout: &view_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: sky_buf.as_entire_binding() },
        ],
    });
    let dispatch_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wc_dispatch_bg"),
        layout: &dispatch_layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: active_cells_dispatch.as_entire_binding() }],
    });
    let cache_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wc_cache_bg"),
        layout: &cache_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wc_uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: checksums.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: life.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: radiance.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: geometry.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: luminance_deltas.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: new_radiance.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: active_cell_indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: active_cells_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: query_points_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 13, resource: query_out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 14, resource: query_params_buf.as_entire_binding() },
        ],
    });

    let mut build = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wc_build") });
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

    let table_groups = TEST_WORLD_CACHE_SIZE / 1024;

    // Drive the full pass sequence one frame at a time, reading back the seeded cell each frame.
    let mut history: Vec<WcQueryOut> = Vec::with_capacity(N_FRAMES as usize);
    for frame in 0..N_FRAMES {
        // Per-frame uniforms. `reset == 1` only on the first frame (mirrors the production one-shot clear).
        let mut wc = wc_defaults;
        wc.frame_index = frame.wrapping_mul(5782582).wrapping_add(1);
        wc.reset = u32::from(frame == 0);
        queue.write_buffer(&wc_uniform, 0, bytemuck::bytes_of(&wc));
        let qp = WcQueryParams {
            view_position,
            n_points,
            frame_index: wc.frame_index,
            ray_t: 0.0,
            _p1: 0,
            _p2: 0,
        };
        queue.write_buffer(&query_params_buf, 0, bytemuck::bytes_of(&qp));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_bind_group(0, Some(&scene_bg), &[]);
            cpass.set_bind_group(1, Some(&view_bg), &[]);
            cpass.set_bind_group(2, Some(&dispatch_bg), &[]); // group 2 = the indirect-dispatch buffer
            cpass.set_bind_group(3, Some(&cache_bg), &[]);
            // SEED the query points FIRST (insert / re-find + alive-mark) — this is where the live reservoir
            // query will sit in 2.2 — then the six-pass loop, in order.
            cpass.set_pipeline(&p_seed);
            cpass.dispatch_workgroups(n_points.div_ceil(64), 1, 1);
            cpass.set_pipeline(&p_decay);
            cpass.dispatch_workgroups(table_groups, 1, 1);
            cpass.set_pipeline(&p_csb);
            cpass.dispatch_workgroups(table_groups, 1, 1);
            cpass.set_pipeline(&p_cb);
            cpass.dispatch_workgroups(1, 1, 1);
            cpass.set_pipeline(&p_cwa);
            cpass.dispatch_workgroups(table_groups, 1, 1);
            // UNBIND group 2 before the indirect dispatches (the update/blend layout omits it) — the dispatch
            // buffer can't be bound storage AND used as the indirect-args source in one usage scope.
            cpass.set_bind_group(2, None, &[]);
            cpass.set_pipeline(&p_update);
            cpass.dispatch_workgroups_indirect(&active_cells_dispatch, 0);
            cpass.set_pipeline(&p_blend);
            cpass.dispatch_workgroups_indirect(&active_cells_dispatch, 0);
        }
        encoder.copy_buffer_to_buffer(
            &query_out_buf,
            0,
            &read_buf,
            0,
            (n_points as u64) * mem::size_of::<WcQueryOut>() as u64,
        );
        queue.submit(Some(encoder.finish()));

        let slice = read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
        let data = slice.get_mapped_range().unwrap();
        let out: WcQueryOut = *bytemuck::from_bytes(&data[..mem::size_of::<WcQueryOut>()]);
        drop(data);
        read_buf.unmap();
        history.push(out);
    }
    let _ = (&aabb_buf, &blas, &tlas);

    let last = *history.last().unwrap();
    eprintln!(
        "[world-cache] analytic R≈{CEILING_RADIANCE:.1} | final radiance={:?} (luma {:.3}) cell={} checksum={} life={}",
        last.radiance,
        luma(last.radiance),
        last.cell_index,
        last.checksum,
        last.life
    );

    // --- 1. Insert + probe: the seeded cell is occupied (non-empty checksum) and re-found at the SAME slot
    //        every frame, with the full lifetime re-stamped. ---
    let first_cell = history[0].cell_index;
    for (f, o) in history.iter().enumerate() {
        assert_ne!(o.checksum, 0, "frame {f}: seeded cell must have a non-empty checksum (insert/probe failed)");
        assert_eq!(o.cell_index, first_cell, "frame {f}: the probe must re-find the SAME cell slot every frame");
        assert!(o.life >= 1, "frame {f}: a just-queried cell must be alive, life={}", o.life);
        assert!(
            o.radiance.iter().all(|v| v.is_finite() && *v >= 0.0),
            "frame {f}: cell radiance must be finite + non-negative, got {:?}",
            o.radiance
        );
    }

    // --- 2. Becomes non-zero: after a few frames the cell's stored radiance is meaningfully above 0 (the
    //        update + blend ran and accumulated the gathered bounce). ---
    assert!(
        luma(last.radiance) > 1e-2,
        "the cache cell must accumulate non-zero radiance, got {:?} (luma {})",
        last.radiance,
        luma(last.radiance)
    );

    // --- 3. Stabilises: the frame-to-frame variance of the radiance in the SECOND half is far below the FIRST
    //        half (the adaptive temporal blend converges). ---
    let half = N_FRAMES as usize / 2;
    let warmup = 4usize; // skip the first few frames while the cell first fills
    let lvar = |slice: &[WcQueryOut]| -> f32 {
        let m = slice.iter().map(|o| luma(o.radiance)).sum::<f32>() / slice.len().max(1) as f32;
        slice.iter().map(|o| (luma(o.radiance) - m).powi(2)).sum::<f32>() / slice.len().max(1) as f32
    };
    let first_var = lvar(&history[warmup..half]);
    let second_var = lvar(&history[half..]);
    eprintln!("[world-cache] first-half var={first_var:.4} second-half var={second_var:.4}");
    assert!(
        second_var <= first_var + 1e-4,
        "radiance variance must not grow as the cache converges: first {first_var:.4} → second {second_var:.4}"
    );

    // --- 4. Matches the analytic single-bounce irradiance: the converged cell radiance ≈ R (the ceiling
    //        radiance), the same oracle a high-spp gather_gi / restir_probe would resolve. ---
    let conv = {
        let tail = &history[(N_FRAMES as usize - 8)..];
        tail.iter().map(|o| luma(o.radiance)).sum::<f32>() / tail.len() as f32
    };
    eprintln!("[world-cache] converged radiance luma={conv:.3} vs analytic {CEILING_RADIANCE:.1}");
    assert!(
        (conv - CEILING_RADIANCE).abs() / CEILING_RADIANCE < 0.3,
        "the cache cell must converge to the analytic single-bounce radiance R={CEILING_RADIANCE} (got {conv:.3})"
    );
}

// === Phase 2.2 energy gate: `reservoir_from_bounce_cached` adds exactly `albedo·cache` ================
//
// The convergence test above proves the cache FILLS to the analytic incoming radiance (cache(floor) ≈ 12); the
// restir_probe test proves the resolve constant. NEITHER drove `reservoir_from_bounce_cached` (the live
// cache-fed initial reservoir) through the resolve — the only other coverage was a compile gate, so the 2.2
// wrong-energy bug (it read the cache RAW, dropping the bounce surface's albedo AND its own direct+emissive)
// would not have been caught. This test pins the corrected rendering-equation relation on the SAME floor /
// emissive-ceiling scene the cache test fills:
//   * fills the cache for N frames (seeding a small grid of up-facing floor cells so `query_world_cache`'s
//     tangent-plane jitter still lands on a filled ≈12 cell), then
//   * runs `world_cache_energy_probe`, which builds BOTH `reservoir_from_bounce` (cache OFF) and
//     `reservoir_from_bounce_cached` (cache ON) for one shading point whose fixed straight-down bounce hits the
//     filled floor, resolves each, and reports the raw radiances + the deterministic cache value + the floor
//     albedo, and
//   * asserts:  cache_on.radiance ≈ cache_off.radiance + floor_albedo·cache(floor)
//     i.e. the cache adds ~ albedo(0.5)·12 = 6 of reflected indirect — NOT 12 (the bug read it raw) and NOT
//     replacing the fresh direct+emissive (a prior reviewer's "* albedo only" mistake would have dropped those).

// Mirror of the WGSL `EnergyProbeParams` (group 0 binding 8) — the shading point + a fixed bounce direction.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EnergyProbeParams {
    shading_position: [f32; 3],
    _p0: u32,
    shading_normal: [f32; 3],
    _p1: u32,
    bounce_dir: [f32; 3],
    _p2: u32,
}

// Mirror of the WGSL `EnergyProbeOut` (group 0 binding 9). Field order + padding MUST match the shader struct.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug, Default)]
struct EnergyProbeOut {
    cache_off_radiance: [f32; 3],
    _p0: u32,
    cache_on_radiance: [f32; 3],
    _p1: u32,
    cache_off_irradiance: [f32; 3],
    _p2: u32,
    cache_on_irradiance: [f32; 3],
    _p3: u32,
    hit_albedo: [f32; 3],
    _p4: u32,
    cache_value: [f32; 3],
    _p5: u32,
    hit: u32,
    _p6: u32,
    _p7: u32,
    _p8: u32,
}

// Mirror of the WGSL `CameraUniform` (group 1 binding 0): `world_from_clip`(64) + `cam_pos`(12) + `t_max`(4) +
// `viewport`(8) + `accum_weight`(4) + pad(4) + `prev_clip_from_world`(64) = 160 bytes. The energy probe reads
// only `cam_pos` (for the cache LOD), so the rest stays zero.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniformMirror {
    world_from_clip: [[f32; 4]; 4],
    cam_pos: [f32; 3],
    t_max: f32,
    viewport: [u32; 2],
    accum_weight: f32,
    _pad: u32,
    prev_clip_from_world: [[f32; 4]; 4],
}

/// The floor albedo (material 0 = "floor", base_color 0.5) — the receiver/bounce reflectance the relation uses.
const FLOOR_ALBEDO: f32 = 0.5;
/// A small floor self-emissive (linear radiance) so the FRESH path carries a non-zero `emissive·strength`
/// term — this is what lets the energy gate distinguish the correct form (keeps direct+emissive) from the
/// "* albedo only" mistake (drops it). Kept small so it doesn't perturb cache(floor) (it isn't gathered into
/// the floor's own cell). Fresh `cache_off.radiance` ≈ FLOOR_EMISSIVE · emissive_strength(4) = 2.0.
const FLOOR_EMISSIVE: f32 = 0.5;

#[test]
fn cached_initial_reservoir_adds_albedo_times_cache() {
    // The fill passes bind 16 storage buffers in one stage (3 scene + 11 cache + query_out + dispatch); the
    // energy probe adds ONE more (`energy_out` on group 0) → 17, over the convergence test's 16.
    let Some((device, queue)) = common::headless_ray_query_device_with_storage_buffers(17) else {
        eprintln!("no ray-query device with 17 storage buffers — skipping energy test");
        return;
    };

    let mut reg = BlockRegistry::from_biome_library(&test_library());
    reg.set_emissive(EMITTER, [3.0, 3.0, 3.0]);
    // Give the FLOOR a small self-emissive so the FRESH path's `direct_lighting + emissive` term is NON-ZERO
    // (= FLOOR_EMISSIVE · emissive_strength). This is what makes the gate ALSO reject the "* albedo only"
    // mistake (which drops the fresh direct+emissive): with a non-zero fresh term, the correct form gives
    // `cache_on = fresh + albedo·cache` while the albedo-only form gives just `albedo·cache` — distinguishable.
    // The floor's own emissive does NOT feed its own cache cell (that cell's +Y hemisphere bounce gathers the
    // CEILING, not itself), so cache(floor) stays ≈ R = 12.
    reg.set_emissive(FLOOR, [FLOOR_EMISSIVE, FLOOR_EMISSIVE, FLOOR_EMISSIVE]);
    let patch = emitter_patch(&reg);
    let n = patch.brick_count() as u32;

    // Same lighting/sky/cache knobs as the convergence test (sun off, ambient 0, dark sky ⇒ the ceiling is the
    // ONLY external light, so cache(floor) ≈ R = 12). The floor's small self-emissive is the only fresh term.
    let light = LightingUniformData {
        sun_direction: [0.0, 1.0, 0.0],
        ambient_color: [0.0, 0.0, 0.0],
        gi_rays: 1,
        gi_intensity: 1.0,
        gi_bounce_dist: 40.0,
        emissive_strength: 4.0,
        ..LightingUniformData::default()
    };
    let sky = SkyUniformData {
        horizon_color: [0.0, 0.0, 0.0],
        zenith_color: [0.0, 0.0, 0.0],
        ground_color: [0.0, 0.0, 0.0],
        sun_size: 0.0,
        intensity: 0.0,
        gi_sky_intensity: 0.0,
        sun_tint: [0.0, 0.0, 0.0],
        _pad: 0.0,
    };
    let wc_defaults = WorldCacheUniformData {
        cell_base_size: 0.3,
        gi_ray_distance: 40.0,
        cell_lifetime: 8,
        ..WorldCacheUniformData::default()
    };

    let s = BRICK_WORLD_SIZE;
    let floor_top = s;
    let cx = s * 0.5;
    let cz = s * 0.5;
    let view_position = [cx, floor_top + 3.0, cz]; // a near camera ⇒ cache LOD 0 (matches the cell_base_size)

    // SEED a 3×3 grid of up-facing floor cells around the bounce-hit cell. Filling the neighbours means
    // `query_world_cache`'s tangent-plane jitter (inside `reservoir_from_bounce_cached`) still lands on a
    // filled ≈12 cell regardless of which neighbour it dithers into — so the energy relation is robust, not a
    // lucky single-cell hit. The grid step is one cache cell (`cell_base_size`).
    let step = wc_defaults.cell_base_size;
    let mut probes: Vec<WcQueryPoint> = Vec::new();
    for dz in -1..=1i32 {
        for dx in -1..=1i32 {
            probes.push(WcQueryPoint {
                world_position: [cx + dx as f32 * step, floor_top, cz + dz as f32 * step],
                _p0: 0,
                world_normal: [0.0, 1.0, 0.0],
                _p1: 0,
            });
        }
    }
    let n_points = probes.len() as u32;

    // The energy probe: a shading point in the gap ABOVE the floor, facing DOWN, firing a fixed straight-down
    // bounce so it deterministically hits the (filled) floor centre cell with normal +Y. Facing down (the
    // sample point is straight below) makes the resolve cosine = 1, so the irradiance relation is clean too.
    let energy_params = EnergyProbeParams {
        shading_position: [cx, floor_top + 0.6, cz],
        _p0: 0,
        shading_normal: [0.0, -1.0, 0.0],
        _p1: 0,
        bounce_dir: [0.0, -1.0, 0.0],
        _p2: 0,
    };

    // --- Scene (group 0) GPU objects ---
    let aabb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("e_aabbs"),
        contents: bytemuck::cast_slice(&patch.aabbs),
        usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE,
    });
    let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("e_metas"),
        contents: bytemuck::cast_slice(&patch.metas),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let voxel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("e_voxels"),
        contents: bytemuck::cast_slice(&patch.voxels),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let palette_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("e_palette"),
        contents: bytemuck::cast_slice(&patch.palette),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let size_desc = wgpu::BlasAABBGeometrySizeDescriptor {
        primitive_count: n,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("e_blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::AABBs { descriptors: vec![size_desc.clone()] },
    );
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("e_tlas"),
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

    // --- Persistent cache buffers (zero-initialised → all cells empty) ---
    let tsz = TEST_WORLD_CACHE_SIZE as u64;
    let zeroed = |label: &str, bytes: u64, indirect: bool| {
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        if indirect {
            usage |= wgpu::BufferUsages::INDIRECT;
        }
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, &vec![0u8; bytes as usize]);
        buf
    };
    let checksums = zeroed("e_checksums", tsz * 4, false);
    let life = zeroed("e_life", tsz * 4, false);
    let radiance = zeroed("e_radiance", tsz * 16, false);
    let geometry = zeroed("e_geometry", tsz * 32, false);
    let luminance_deltas = zeroed("e_luminance_deltas", tsz * 4, false);
    let new_radiance = zeroed("e_new_radiance", tsz * 16, false);
    let a = zeroed("e_a", tsz * 4, false);
    let b = zeroed("e_b", 1024 * 4, false);
    let active_cell_indices = zeroed("e_active_cell_indices", tsz * 4, false);
    let active_cells_count = zeroed("e_active_cells_count", 4, false);
    let active_cells_dispatch = zeroed("e_active_cells_dispatch", 12, true);

    // --- Per-frame + uniform buffers ---
    let wc_uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e_wc_uniform"),
        size: mem::size_of::<WorldCacheUniformData>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let light_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("e_light"),
        contents: bytemuck::bytes_of(&light),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let sky_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("e_sky"),
        contents: bytemuck::bytes_of(&sky),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let camera = CameraUniformMirror {
        world_from_clip: [[0.0; 4]; 4],
        cam_pos: view_position,
        t_max: 1.0e4,
        viewport: [1, 1],
        accum_weight: 1.0,
        _pad: 0,
        prev_clip_from_world: [[0.0; 4]; 4],
    };
    let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("e_camera"),
        contents: bytemuck::bytes_of(&camera),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // --- Seed (test) buffers — drive the cache fill (group 3 bindings 12/13/14) ---
    let query_points_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("e_query_points"),
        contents: bytemuck::cast_slice(&probes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let query_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e_query_out"),
        size: (n_points as u64) * mem::size_of::<WcQueryOut>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let query_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e_query_params"),
        size: mem::size_of::<WcQueryParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // --- Energy-probe I/O (group 0 bindings 8/9) ---
    let energy_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("e_energy_params"),
        contents: bytemuck::bytes_of(&energy_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let energy_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e_energy_out"),
        size: mem::size_of::<EnergyProbeOut>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let energy_read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e_energy_read"),
        size: mem::size_of::<EnergyProbeOut>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // --- Bind-group layouts ---
    let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    // group 0: scene (0-3) for the FILL passes + energy I/O (8 uniform, 9 storage) for the probe. One layout
    // shared by both pipelines (the fill passes simply don't touch bindings 8/9).
    let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("e_scene_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
                count: None,
            },
            storage_ro(1),
            storage_ro(2),
            storage_ro(3),
            uniform(8),
            storage_rw(9),
        ],
    });
    // group 1: camera (0) for the probe's cache LOD + light (2) + sky (11). The fill passes ignore camera.
    let view_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("e_view_layout"),
        entries: &[uniform(0), uniform(2), uniform(11)],
    });
    let dispatch_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("e_dispatch_layout"),
        entries: &[storage_rw(0)],
    });
    let cache_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("e_cache_layout"),
        entries: &[
            uniform(0),
            storage_rw(1),
            storage_rw(2),
            storage_rw(3),
            storage_rw(4),
            storage_rw(5),
            storage_rw(6),
            storage_rw(7),
            storage_rw(8),
            storage_rw(9),
            storage_rw(10),
            storage_ro(12),
            storage_rw(13),
            uniform(14),
        ],
    });
    // Layout A — seed + decay + 3 compaction passes (group 2 = the indirect-dispatch buffer present).
    let compact_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("e_compact_pl"),
        bind_group_layouts: &[Some(&scene_layout), Some(&view_layout), Some(&dispatch_layout), Some(&cache_layout)],
        immediate_size: 0,
    });
    // Layout B — update + blend + the energy probe (group 2 absent so the dispatch buffer is free as the
    // indirect-args source; the energy probe doesn't touch group 2 either).
    let update_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("e_update_pl"),
        bind_group_layouts: &[Some(&scene_layout), Some(&view_layout), None, Some(&cache_layout)],
        immediate_size: 0,
    });

    let src = adventure::voxel::raytrace::voxel_raytrace_shader_src(TEST_WORLD_CACHE_SIZE);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_raytrace"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let mk = |entry: &str, layout: &wgpu::PipelineLayout| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let p_seed = mk("world_cache_query_seed", &compact_pl);
    let p_decay = mk("world_cache_decay", &compact_pl);
    let p_csb = mk("world_cache_compact_single_block", &compact_pl);
    let p_cb = mk("world_cache_compact_blocks", &compact_pl);
    let p_cwa = mk("world_cache_compact_write_active", &compact_pl);
    let p_update = mk("world_cache_update", &update_pl);
    let p_blend = mk("world_cache_blend", &update_pl);
    let p_energy = mk("world_cache_energy_probe", &update_pl);

    // --- Bind groups ---
    let scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("e_scene_bg"),
        layout: &scene_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::AccelerationStructure(&tlas) },
            wgpu::BindGroupEntry { binding: 1, resource: meta_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: voxel_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: palette_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: energy_params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: energy_out_buf.as_entire_binding() },
        ],
    });
    let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("e_view_bg"),
        layout: &view_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: sky_buf.as_entire_binding() },
        ],
    });
    let dispatch_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("e_dispatch_bg"),
        layout: &dispatch_layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: active_cells_dispatch.as_entire_binding() }],
    });
    let cache_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("e_cache_bg"),
        layout: &cache_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wc_uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: checksums.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: life.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: radiance.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: geometry.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: luminance_deltas.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: new_radiance.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: active_cell_indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: active_cells_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: query_points_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 13, resource: query_out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 14, resource: query_params_buf.as_entire_binding() },
        ],
    });

    let mut build = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("e_build") });
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

    let table_groups = TEST_WORLD_CACHE_SIZE / 1024;

    // FILL the cache: run the full fill loop for N frames (no energy probe yet — let the floor cells converge).
    for frame in 0..N_FRAMES {
        let mut wc = wc_defaults;
        wc.frame_index = frame.wrapping_mul(5782582).wrapping_add(1);
        wc.reset = u32::from(frame == 0);
        queue.write_buffer(&wc_uniform, 0, bytemuck::bytes_of(&wc));
        let qp = WcQueryParams {
            view_position,
            n_points,
            frame_index: wc.frame_index,
            ray_t: 0.0,
            _p1: 0,
            _p2: 0,
        };
        queue.write_buffer(&query_params_buf, 0, bytemuck::bytes_of(&qp));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_bind_group(0, Some(&scene_bg), &[]);
            cpass.set_bind_group(1, Some(&view_bg), &[]);
            cpass.set_bind_group(2, Some(&dispatch_bg), &[]);
            cpass.set_bind_group(3, Some(&cache_bg), &[]);
            cpass.set_pipeline(&p_seed);
            cpass.dispatch_workgroups(n_points.div_ceil(64), 1, 1);
            cpass.set_pipeline(&p_decay);
            cpass.dispatch_workgroups(table_groups, 1, 1);
            cpass.set_pipeline(&p_csb);
            cpass.dispatch_workgroups(table_groups, 1, 1);
            cpass.set_pipeline(&p_cb);
            cpass.dispatch_workgroups(1, 1, 1);
            cpass.set_pipeline(&p_cwa);
            cpass.dispatch_workgroups(table_groups, 1, 1);
            cpass.set_bind_group(2, None, &[]);
            cpass.set_pipeline(&p_update);
            cpass.dispatch_workgroups_indirect(&active_cells_dispatch, 0);
            cpass.set_pipeline(&p_blend);
            cpass.dispatch_workgroups_indirect(&active_cells_dispatch, 0);
        }
        queue.submit(Some(encoder.finish()));
    }

    // RUN the energy probe ONCE on the filled cache. group 2 is unbound (update_pl), and the probe touches only
    // groups 0/1/3 — so the same scene/view/cache bind groups apply.
    {
        let mut wc = wc_defaults;
        wc.frame_index = 0xABCDEF; // any non-zero stream
        wc.reset = 0;
        queue.write_buffer(&wc_uniform, 0, bytemuck::bytes_of(&wc));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_bind_group(0, Some(&scene_bg), &[]);
            cpass.set_bind_group(1, Some(&view_bg), &[]);
            cpass.set_bind_group(3, Some(&cache_bg), &[]);
            cpass.set_pipeline(&p_energy);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&energy_out_buf, 0, &energy_read_buf, 0, mem::size_of::<EnergyProbeOut>() as u64);
        queue.submit(Some(encoder.finish()));
    }

    let slice = energy_read_buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
    let data = slice.get_mapped_range().unwrap();
    let out: EnergyProbeOut = *bytemuck::from_bytes(&data[..mem::size_of::<EnergyProbeOut>()]);
    drop(data);
    energy_read_buf.unmap();
    let _ = (&aabb_buf, &blas, &tlas);

    let cache_off = luma(out.cache_off_radiance);
    let cache_on = luma(out.cache_on_radiance);
    let cache_val = luma(out.cache_value);
    let albedo = luma(out.hit_albedo);
    let delta = cache_on - cache_off;
    let expected_delta = albedo * cache_val;
    eprintln!(
        "[energy] hit={} albedo={:.3} cache(floor)={:.3} | cache_off.radiance={:.3} cache_on.radiance={:.3} \
         delta={:.3} vs albedo*cache={:.3} | irradiance off={:.3} on={:.3}",
        out.hit,
        albedo,
        cache_val,
        cache_off,
        cache_on,
        delta,
        expected_delta,
        luma(out.cache_off_irradiance),
        luma(out.cache_on_irradiance),
    );

    // --- Sanity: the bounce hit the floor, with the expected albedo, and the cache there filled to ≈ R. ---
    assert_eq!(out.hit, 1, "the fixed straight-down bounce must hit the floor (cache cell to read)");
    assert!(
        (albedo - FLOOR_ALBEDO).abs() < 0.02,
        "the bounce must hit the FLOOR (albedo {FLOOR_ALBEDO}), got {albedo:.3}"
    );
    assert!(
        (cache_val - CEILING_RADIANCE).abs() / CEILING_RADIANCE < 0.3,
        "the floor cache cell must have filled to the analytic R={CEILING_RADIANCE} (got {cache_val:.3})"
    );

    // --- Cache OFF == the FRESH single bounce (direct+emissive). With sun off + ambient 0 the floor's
    //     direct_lighting is 0, so the fresh radiance is its self-emissive ≈ FLOOR_EMISSIVE·emissive_strength.
    //     This being NON-ZERO is what lets the relation distinguish "keeps direct+emissive" from "albedo only". ---
    let fresh_expected = FLOOR_EMISSIVE * light.emissive_strength; // 0.5 · 4 = 2.0
    assert!(
        (cache_off - fresh_expected).abs() / fresh_expected < 0.1,
        "cache-OFF must equal the fresh direct+emissive ≈ {fresh_expected:.3} (floor self-emissive), got {cache_off:.3}"
    );

    // --- THE ENERGY RELATION (the bug gate): cache_on.radiance ≈ cache_off.radiance + albedo·cache(floor).
    //     The cache adds exactly one reflected indirect bounce (albedo·cache ≈ 0.5·12 = 6), ON TOP of the fresh
    //     direct+emissive (≈ 2). This FAILS for BOTH wrong-energy forms:
    //       * the original bug (raw cache → cache_on ≈ 12): delta ≈ 12 = the un-weighted cache — caught below;
    //       * the "* albedo only" mistake (cache REPLACES direct+emissive → cache_on ≈ albedo·cache ≈ 6):
    //         then delta = cache_on - cache_off ≈ 6 - 2 = 4 ≠ albedo·cache ≈ 6 — caught by the relation. ---
    let denom = expected_delta.max(1e-3);
    assert!(
        (delta - expected_delta).abs() / denom < 0.15,
        "cache_on.radiance must equal cache_off.radiance + albedo·cache(floor): delta={delta:.3} vs \
         expected={expected_delta:.3} (cache_off={cache_off:.3}, albedo={albedo:.3}, cache={cache_val:.3})"
    );
    // Reject the RAW-cache bug: the cache contribution is ALBEDO-WEIGHTED (~6), not the raw cache (~12).
    assert!(
        delta < 0.75 * cache_val,
        "the cache contribution must be albedo-weighted (≈{:.3}), not the RAW cache ({cache_val:.3}) — the 2.2 bug",
        expected_delta
    );
    // Reject the "* albedo only" mistake: cache_on must KEEP the fresh direct+emissive (so it exceeds the bare
    // albedo·cache by ≈ the fresh term). If the fix dropped direct+emissive, cache_on would be ≈ albedo·cache.
    assert!(
        cache_on > expected_delta + 0.5 * fresh_expected,
        "cache_on.radiance must INCLUDE the fresh direct+emissive on top of albedo·cache (≈{:.3}+{:.3}), got \
         {cache_on:.3} — does the cached path drop direct+emissive?",
        fresh_expected,
        expected_delta
    );

    // --- The resolve carries the SAME relation through (the live path resolves these to irradiance): the
    //     extra irradiance is the resolve factor times albedo·cache, and is strictly positive (cache helps). ---
    assert!(
        luma(out.cache_on_irradiance) > luma(out.cache_off_irradiance) + 1e-3,
        "resolved cache-ON irradiance must exceed cache-OFF (the cache adds reflected indirect): on={:.4} off={:.4}",
        luma(out.cache_on_irradiance),
        luma(out.cache_off_irradiance)
    );
}

// === Phase 2.2.1 thin-wall LIGHT-LEAK regression gate ==================================================
//
// Reproduces, headlessly, the user-reported cache leak (light from UNDER the closed Cornell box bleeding onto
// interior cube faces) and pins the first-bounce light-leak-prevention clamp that fixes it
// (`voxel_raytrace.wgsl` `query_world_cache`: `if (ray_t < cell_size) { cell_size = wc.cell_base_size; }`).
//
// HOW THE CLAMP WORKS (verified against the ported Solari code): the clamp does NOT shrink the final
// quantization cell (that is re-derived from the LOD AFTER the jitter, line ~1832, exactly as Solari does) —
// it shrinks the TANGENT-PLANE JITTER amplitude (`offset = ±0.5·cell_size`, line ~1830). The leak is the jitter
// stochastically pushing a near-wall query ACROSS a thin wall into the cell on the far side; clamping the cell
// to the small base size (0.15 m) for a SHORT bounce keeps the jitter sub-wall so it cannot cross. (This is why
// the leak is "infrequent" — only the fraction of jitter offsets that cross leak.)
//
// SCENE (matches that mechanism exactly): viewer at distance == `lod_scale` (15 m) ⇒ `lod_f = log2(1+15/15) =
// 1` with fract 0 ⇒ a DETERMINISTIC LOD-1 cell of 0.15·2 = 0.3 m (no stochastic round-up). Two UP-FACING (+Y)
// points under the emissive ceiling (so a +Y cosine bounce reaches it ⇒ a seeded cell fills to R≈12), with the
// straddle along X (so it lies in the +Y tangent plane the jitter moves in):
//   * EXTERIOR x=0.45 — in the LOD-1 X-bucket [0.30,0.60); SEEDED + filled bright every frame.
//   * INTERIOR x=0.70 — in the next bucket [0.60,0.90), 0.10 m above the bucket boundary at x=0.60; NEVER
//     seeded, so its own cell is empty (0). It is the leak target.
// Un-jittered the two are in different buckets (no quantization-collapse leak). The jitter (in X) bridges them:
//   * WITHOUT the clamp: jitter ±0.5·0.3 = ±0.15 m ⇒ the interior query reaches x as low as 0.55 < 0.60 ⇒ a
//     fraction of samples quantize into the EXTERIOR bucket and read its bright radiance (the LEAK).
//   * WITH the clamp (short bounce ⇒ cell = 0.15): jitter ±0.5·0.15 = ±0.075 m ⇒ interior x stays ≥ 0.625 >
//     0.60 ⇒ NO sample crosses ⇒ the interior reads its own empty cell ⇒ ~0.
// The probe fires the REAL `query_world_cache` (256 jittered samples averaged) via the new
// `world_cache_leak_probe` entry, so the clamp is exercised exactly as the live ReSTIR path does.
//
// ASSERTS: with the clamp, the SHORT-`ray_t` interior read is ≪ the bright exterior cell. MUTATION CHECK (in
// the verification report): deleting the clamp lets the ±0.15 jitter cross, so the interior read jumps to a
// meaningful fraction of R and the assert fails.

/// View distance == `WorldCacheUniformData::lod_scale` (15 m) ⇒ `lod_f = log2(1 + 15/15) = 1.0`, fract 0 ⇒ a
/// DETERMINISTIC LOD-1 cell of `0.15·2^1 = 0.3 m` (the stochastic round-up term `rand < fract³` is 0 at fract
/// 0). A small jitter perturbs the recomputed distance by ≤0.15 m ⇒ fract ≈ 1e-2 ⇒ round-up prob ≈ 1e-6, so the
/// cell stays 0.3 m for essentially every sample. This is the regime where the clamp (jitter-amplitude) bites.
const LEAK_VIEW_DIST: f32 = 15.0;

#[test]
fn thin_wall_no_exterior_leak_with_clamp() {
    let Some((device, queue)) = common::headless_ray_query_device_with_storage_buffers(16) else {
        eprintln!("no ray-query device with 16 storage buffers — skipping thin-wall leak test");
        return;
    };

    let mut reg = BlockRegistry::from_biome_library(&test_library());
    reg.set_emissive(EMITTER, [3.0, 3.0, 3.0]);
    let patch = emitter_patch(&reg);
    let n = patch.brick_count() as u32;

    // Same single-light setup as the convergence test: sun off, ambient 0, dark sky ⇒ the emissive ceiling is
    // the ONLY light, so a filled up-facing cell holds R = emissive·strength = 12.
    let light = LightingUniformData {
        sun_direction: [0.0, 1.0, 0.0],
        ambient_color: [0.0, 0.0, 0.0],
        gi_rays: 1,
        gi_intensity: 1.0,
        gi_bounce_dist: 40.0,
        emissive_strength: 4.0,
        ..LightingUniformData::default()
    };
    let sky = SkyUniformData {
        horizon_color: [0.0, 0.0, 0.0],
        zenith_color: [0.0, 0.0, 0.0],
        ground_color: [0.0, 0.0, 0.0],
        sun_size: 0.0,
        intensity: 0.0,
        gi_sky_intensity: 0.0,
        sun_tint: [0.0, 0.0, 0.0],
        _pad: 0.0,
    };
    // PRODUCTION base cell (0.15 m): the clamp target. The leak depends on the LOD cell exceeding the wall, so
    // we keep the real base size and force a large LOD via the far viewer instead.
    let wc_defaults = WorldCacheUniformData {
        cell_base_size: 0.15,
        lod_scale: 15.0,
        gi_ray_distance: 40.0,
        cell_lifetime: 8,
        ..WorldCacheUniformData::default()
    };

    let cz = BRICK_WORLD_SIZE * 0.5;
    // Both points sit in the open gap below the emissive ceiling (gap y∈[1.6,3.2]), UP-FACING, at the same y so
    // a +Y cosine bounce reaches the ceiling and a seeded cell fills to R≈12. The straddle is along X (in the
    // +Y tangent plane the jitter perturbs). The LOD-1 cell is 0.3 m with X-bucket boundaries at multiples of
    // 0.3; the boundary at x=0.60 separates the two points.
    let y_pt = 2.0;
    let x_ext = 0.45; // bucket [0.30,0.60) — seeded + filled bright
    let x_int = 0.70; // bucket [0.60,0.90), 0.10 m above the x=0.60 boundary — never seeded (the leak target)
    // The viewer is directly above (distance == LEAK_VIEW_DIST regardless of the small X offset), pinning LOD 1.
    let view_position = [(x_ext + x_int) * 0.5, y_pt + LEAK_VIEW_DIST, cz];
    let exterior = WcQueryPoint {
        world_position: [x_ext, y_pt, cz],
        _p0: 0,
        world_normal: [0.0, 1.0, 0.0],
        _p1: 0,
    };
    let interior = WcQueryPoint {
        world_position: [x_int, y_pt, cz],
        _p0: 0,
        world_normal: [0.0, 1.0, 0.0],
        _p1: 0,
    };
    // index 0 = exterior (seeded/filled bright every frame), index 1 = interior (never seeded; the leak target).
    let probes = [exterior, interior];
    let n_points = probes.len() as u32;

    // --- Scene (group 0) ---
    let aabb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lk_aabbs"),
        contents: bytemuck::cast_slice(&patch.aabbs),
        usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE,
    });
    let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lk_metas"),
        contents: bytemuck::cast_slice(&patch.metas),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let voxel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lk_voxels"),
        contents: bytemuck::cast_slice(&patch.voxels),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let palette_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lk_palette"),
        contents: bytemuck::cast_slice(&patch.palette),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let size_desc = wgpu::BlasAABBGeometrySizeDescriptor {
        primitive_count: n,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("lk_blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::AABBs { descriptors: vec![size_desc.clone()] },
    );
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("lk_tlas"),
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

    // --- Persistent cache buffers (zero ⇒ all cells empty) ---
    let tsz = TEST_WORLD_CACHE_SIZE as u64;
    let zeroed = |label: &str, bytes: u64, indirect: bool| {
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        if indirect {
            usage |= wgpu::BufferUsages::INDIRECT;
        }
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, &vec![0u8; bytes as usize]);
        buf
    };
    let checksums = zeroed("lk_checksums", tsz * 4, false);
    let life = zeroed("lk_life", tsz * 4, false);
    let radiance = zeroed("lk_radiance", tsz * 16, false);
    let geometry = zeroed("lk_geometry", tsz * 32, false);
    let luminance_deltas = zeroed("lk_luminance_deltas", tsz * 4, false);
    let new_radiance = zeroed("lk_new_radiance", tsz * 16, false);
    let a = zeroed("lk_a", tsz * 4, false);
    let b = zeroed("lk_b", 1024 * 4, false);
    let active_cell_indices = zeroed("lk_active_cell_indices", tsz * 4, false);
    let active_cells_count = zeroed("lk_active_cells_count", 4, false);
    let active_cells_dispatch = zeroed("lk_active_cells_dispatch", 12, true);

    // --- Uniforms + test buffers ---
    let wc_uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lk_wc_uniform"),
        size: mem::size_of::<WorldCacheUniformData>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let light_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lk_light"),
        contents: bytemuck::bytes_of(&light),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let sky_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lk_sky"),
        contents: bytemuck::bytes_of(&sky),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let query_points_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lk_query_points"),
        contents: bytemuck::cast_slice(&probes),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, // rewritten per read in `read_one`
    });
    let query_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lk_query_out"),
        size: (n_points as u64) * mem::size_of::<WcQueryOut>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let query_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lk_query_params"),
        size: mem::size_of::<WcQueryParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lk_read"),
        size: (n_points as u64) * mem::size_of::<WcQueryOut>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // --- Layouts (identical to the convergence test) ---
    let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lk_scene_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
                count: None,
            },
            storage_ro(1),
            storage_ro(2),
            storage_ro(3),
        ],
    });
    let view_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lk_view_layout"),
        entries: &[uniform(2), uniform(11)],
    });
    let dispatch_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lk_dispatch_layout"),
        entries: &[storage_rw(0)],
    });
    let cache_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lk_cache_layout"),
        entries: &[
            uniform(0),
            storage_rw(1),
            storage_rw(2),
            storage_rw(3),
            storage_rw(4),
            storage_rw(5),
            storage_rw(6),
            storage_rw(7),
            storage_rw(8),
            storage_rw(9),
            storage_rw(10),
            storage_ro(12),
            storage_rw(13),
            uniform(14),
        ],
    });
    let compact_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("lk_compact_pl"),
        bind_group_layouts: &[Some(&scene_layout), Some(&view_layout), Some(&dispatch_layout), Some(&cache_layout)],
        immediate_size: 0,
    });
    let update_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("lk_update_pl"),
        bind_group_layouts: &[Some(&scene_layout), Some(&view_layout), None, Some(&cache_layout)],
        immediate_size: 0,
    });

    let src = adventure::voxel::raytrace::voxel_raytrace_shader_src(TEST_WORLD_CACHE_SIZE);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_raytrace"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let mk = |entry: &str, layout: &wgpu::PipelineLayout| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let p_seed = mk("world_cache_query_seed", &compact_pl);
    let p_decay = mk("world_cache_decay", &compact_pl);
    let p_csb = mk("world_cache_compact_single_block", &compact_pl);
    let p_cb = mk("world_cache_compact_blocks", &compact_pl);
    let p_cwa = mk("world_cache_compact_write_active", &compact_pl);
    let p_update = mk("world_cache_update", &update_pl);
    let p_blend = mk("world_cache_blend", &update_pl);
    let p_leak = mk("world_cache_leak_probe", &update_pl);

    // --- Bind groups ---
    let scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lk_scene_bg"),
        layout: &scene_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::AccelerationStructure(&tlas) },
            wgpu::BindGroupEntry { binding: 1, resource: meta_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: voxel_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: palette_buf.as_entire_binding() },
        ],
    });
    let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lk_view_bg"),
        layout: &view_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: sky_buf.as_entire_binding() },
        ],
    });
    let dispatch_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lk_dispatch_bg"),
        layout: &dispatch_layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: active_cells_dispatch.as_entire_binding() }],
    });
    let cache_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lk_cache_bg"),
        layout: &cache_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wc_uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: checksums.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: life.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: radiance.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: geometry.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: luminance_deltas.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: new_radiance.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: active_cell_indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: active_cells_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: query_points_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 13, resource: query_out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 14, resource: query_params_buf.as_entire_binding() },
        ],
    });

    let mut build = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("lk_build") });
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

    let table_groups = TEST_WORLD_CACHE_SIZE / 1024;

    // FILL: each frame, leak-probe ONLY the EXTERIOR point (`n_points = 1`) with a LARGE `ray_t` (no clamp ⇒
    // the large LOD cell) so its lazy-insert + alive-mark land on the LARGE-cell key; the six fill passes then
    // bounce it up to the emissive ceiling and accumulate R≈12 into that exterior cell. The interior cell is
    // NEVER touched here — its only path to non-zero radiance is the (clamp-defeated) straddle, which is the leak.
    let fill_params = |frame: u32| WcQueryParams {
        view_position,
        n_points: 1, // exterior only
        frame_index: frame.wrapping_mul(5782582).wrapping_add(1),
        ray_t: 1.0e4, // huge ⇒ ray_t >= cell_size ⇒ NO clamp ⇒ fills the LARGE cell
        _p1: 0,
        _p2: 0,
    };
    for frame in 0..N_FRAMES {
        let mut wc = wc_defaults;
        wc.frame_index = frame.wrapping_mul(5782582).wrapping_add(1);
        wc.reset = u32::from(frame == 0);
        queue.write_buffer(&wc_uniform, 0, bytemuck::bytes_of(&wc));
        queue.write_buffer(&query_params_buf, 0, bytemuck::bytes_of(&fill_params(frame)));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_bind_group(0, Some(&scene_bg), &[]);
            cpass.set_bind_group(1, Some(&view_bg), &[]);
            cpass.set_bind_group(3, Some(&cache_bg), &[]);
            // Leak-probe the exterior point (group 2 unbound = update_pl) — lazy-inserts the LARGE-cell key.
            cpass.set_pipeline(&p_leak);
            cpass.dispatch_workgroups(1, 1, 1);
            // Then the six fill passes (group 2 bound for decay/compaction).
            cpass.set_bind_group(2, Some(&dispatch_bg), &[]);
            cpass.set_pipeline(&p_seed); // no-op here (n_points stays 1 = the exterior; harmless re-mark)
            cpass.dispatch_workgroups(1, 1, 1);
            cpass.set_pipeline(&p_decay);
            cpass.dispatch_workgroups(table_groups, 1, 1);
            cpass.set_pipeline(&p_csb);
            cpass.dispatch_workgroups(table_groups, 1, 1);
            cpass.set_pipeline(&p_cb);
            cpass.dispatch_workgroups(1, 1, 1);
            cpass.set_pipeline(&p_cwa);
            cpass.dispatch_workgroups(table_groups, 1, 1);
            cpass.set_bind_group(2, None, &[]);
            cpass.set_pipeline(&p_update);
            cpass.dispatch_workgroups_indirect(&active_cells_dispatch, 0);
            cpass.set_pipeline(&p_blend);
            cpass.dispatch_workgroups_indirect(&active_cells_dispatch, 0);
        }
        queue.submit(Some(encoder.finish()));
    }

    // MEASURE: two final leak-probe reads on the filled cache (no fill passes — pure reads):
    //   * EXTERIOR with a LARGE ray_t (no clamp, large cell) — anchors that the exterior cell IS bright (so a
    //     "both zero" pass can't sneak through).
    //   * INTERIOR with a SHORT ray_t (clamp ARMED) — the leak target; must stay ~0.
    let read_one = |point_index: u32, ray_t: f32| -> [f32; 3] {
        let mut wc = wc_defaults;
        wc.frame_index = 0x1234567;
        wc.reset = 0;
        queue.write_buffer(&wc_uniform, 0, bytemuck::bytes_of(&wc));
        // Point the probe at a single chosen query point by writing it alone to slot 0 + n_points = 1.
        let pt = probes[point_index as usize];
        queue.write_buffer(&query_points_buf, 0, bytemuck::bytes_of(&pt));
        let qp = WcQueryParams {
            view_position,
            n_points: 1,
            frame_index: 0x1234567,
            ray_t,
            _p1: 0,
            _p2: 0,
        };
        queue.write_buffer(&query_params_buf, 0, bytemuck::bytes_of(&qp));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_bind_group(0, Some(&scene_bg), &[]);
            cpass.set_bind_group(1, Some(&view_bg), &[]);
            cpass.set_bind_group(3, Some(&cache_bg), &[]);
            cpass.set_pipeline(&p_leak);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&query_out_buf, 0, &read_buf, 0, mem::size_of::<WcQueryOut>() as u64);
        queue.submit(Some(encoder.finish()));
        let slice = read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
        let data = slice.get_mapped_range().unwrap();
        let out: WcQueryOut = *bytemuck::from_bytes(&data[..mem::size_of::<WcQueryOut>()]);
        drop(data);
        read_buf.unmap();
        out.radiance
    };

    // The SHORT ray_t (a cube-face→adjacent-floor bounce, ~one voxel) — strictly below the 0.3 m LOD-1 cell so
    // `ray_t < cell_size` fires the clamp; comfortably below the sub-0.4 m wall the production guard targets.
    let short_ray_t = 0.2_f32;
    let exterior_rad = luma(read_one(0, 1.0e4)); // large ray_t ⇒ reads the bright LARGE-cell exterior value
    let interior_rad = luma(read_one(1, short_ray_t)); // short ray_t ⇒ clamp ⇒ small cell ⇒ no straddle ⇒ ~0
    let _ = (&aabb_buf, &blas, &tlas);

    eprintln!(
        "[leak] exterior cell luma={exterior_rad:.3} (analytic R≈{CEILING_RADIANCE:.1}) | interior (short ray_t, \
         clamp ARMED) luma={interior_rad:.4} | leak ratio={:.4}",
        interior_rad / exterior_rad.max(1e-6)
    );

    // Anchor: the exterior cell actually filled bright (else a both-zero pass would be meaningless).
    assert!(
        exterior_rad > 0.5 * CEILING_RADIANCE,
        "the EXTERIOR cache cell must have filled bright (≈R={CEILING_RADIANCE}); got {exterior_rad:.3} — fill failed, \
         the leak assertion below would be vacuous"
    );

    // THE LEAK GATE: with the clamp armed, the SHORT-ray_t interior query maps to its OWN (empty) base cell and
    // does NOT collapse onto the bright exterior cell ⇒ ~0. MUTATION CHECK: deleting
    // `if (ray_t < cell_size) { cell_size = wc.cell_base_size; }` makes the short-ray_t interior query use the
    // LARGE cell, collapse onto the exterior cell, and read ≈ R — this assert then FAILS (interior ≈ exterior).
    assert!(
        interior_rad < 0.1 * exterior_rad,
        "thin-wall LEAK: the interior receiver (short ray_t={short_ray_t}) must NOT pick up the exterior cell's \
         radiance with the leak-prevention clamp armed — interior luma {interior_rad:.4} vs exterior {exterior_rad:.3} \
         (the clamp `if (ray_t < cell_size) {{ cell_size = wc.cell_base_size; }}` was likely removed)"
    );
}
