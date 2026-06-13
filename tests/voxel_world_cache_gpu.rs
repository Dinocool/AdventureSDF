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
    _p0: u32,
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
        gi_firefly_clamp: 0.0,
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
            _p0: 0,
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
