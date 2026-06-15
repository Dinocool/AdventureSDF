//! Real-GPU execution of the hardware-ray-traced voxel path (`voxel_raytrace.wgsl`, `trace_one` entry).
//!
//! This is the correctness ORACLE for the whole HW-RT voxel path, proven WITHOUT the GUI:
//!   1. Voxelize a small known region into a [`BrickMap`] (the SAME `voxelize`/brickmap CPU code the
//!      renderer uses).
//!   2. Pack it into the SSOT GPU layout (`voxel::gpu::pack_brickmap`).
//!   3. Build a per-brick procedural-AABB BLAS + a TLAS of brick instances (the proven wgpu-trunk AABB
//!      `ray_query` API from `D:/spike-aabb`).
//!   4. Run the ACTUAL `voxel_raytrace.wgsl` compute shader for ONE known ray; read back the hit.
//!   5. Assert the GPU hit (block id + brick + world-t) matches a CPU DDA ground truth through the same
//!      brickmap. Same ray, same first-solid voxel — the GPU path is correct iff they agree.
//!
//! Skips cleanly (no failure) on a box without an `EXPERIMENTAL_RAY_QUERY` Vulkan adapter.

use std::iter;
use std::mem;

use bevy::math::{IVec3, Vec3};
use wgpu::util::DeviceExt;

use adventure::sdf_render::worldgen::biome::{
    BiomeDef, BiomeId, BiomeLibrary, StrataLayer, TerrainMatId, TerrainSurfaceMaterial,
};
use adventure::sdf_render::worldgen::coord::LayerId;
use adventure::sdf_render::worldgen::layers::erosion::ErosionParams;
use adventure::sdf_render::worldgen::layers::height::{HeightLayer, HeightParams};
use adventure::voxel::brickmap::{
    BRICK_EDGE, BRICK_WORLD_SIZE, Brick, BrickMap, VOXEL_SIZE, brick_coord_of_voxel, brick_span, lod_edge,
    lod_voxel_size,
};
use adventure::voxel::gpu::{GpuBrickPatch, ResidentBrick, pack_brickmap, pack_resident_set};
use adventure::voxel::palette::{BlockId, BlockRegistry};
use adventure::voxel::voxelize::voxelize_brick;

mod common;

const SEED: u64 = 0xA15E_C0DE_2026;

// Mirror of the WGSL `Hit` struct (binding 5): hit, block_id, prim, t, a 16-byte vec4 colour, the face
// normal (vec3 → 16-byte aligned/padded) + `shadowed`, then the GI oracle terms (direct/indirect/emissive,
// each vec3 padded to 16 in std430). This rig only reads the geometry fields but must size the buffer to the
// full struct the shader now writes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
struct GpuHit {
    hit: u32,
    block_id: u32,
    prim: u32,
    t: f32,
    color: [f32; 4],
    normal: [f32; 3],
    shadowed: u32,
    direct: [f32; 3],
    _p0: u32,
    indirect: [f32; 3],
    _p1: u32,
    emissive_out: [f32; 3],
    _p2: u32,
}

/// Mirror of the WGSL `LightingUniform` (group 1, binding 2) — 64 bytes. `trace_one` references the lighting
/// global for its shadow ray, so this rig must bind it even though it only checks geometry. Re-uses the SSOT
/// `LightingUniformData` from the crate so the layout cannot drift.
fn lighting_uniform() -> adventure::voxel::raytrace::LightingUniformData {
    adventure::voxel::raytrace::LightingUniformData::default()
}

// Mirror of the WGSL `RayUniform` (binding 4): origin+t_min, dir+t_max (std140: vec3 then trailing f32).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RayUniform {
    origin: [f32; 3],
    t_min: f32,
    dir: [f32; 3],
    t_max: f32,
}

/// A small worldgen library with distinct strata so a hit's block id is meaningful, mirroring the
/// voxelizer test library (surface / sub / stone / bedrock).
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
        mat("surface", [0.1, 0.5, 0.1, 1.0]),
        mat("sub", [0.3, 0.2, 0.1, 1.0]),
        mat("stone", [0.5, 0.5, 0.5, 1.0]),
        mat("bedrock", [0.0, 0.0, 0.0, 1.0]),
    ];
    let column = |_| BiomeDef {
        name: "b".into(),
        surface: TerrainMatId(0),
        surface_rules: vec![],
        strata: vec![
            StrataLayer { material: TerrainMatId(0), thickness: 1.0 },
            StrataLayer { material: TerrainMatId(1), thickness: 4.0 },
            StrataLayer { material: TerrainMatId(2), thickness: 20.0 },
        ],
        bedrock: TerrainMatId(3),
    };
    let biomes = BiomeId::ALL.iter().map(column).collect();
    BiomeLibrary { materials, biomes }
}

fn test_layer() -> HeightLayer {
    HeightLayer::new(LayerId(0), HeightParams::default(), ErosionParams::default())
}

/// Voxelize the bricks spanned by world-voxel AABB `[vmin, vmax]` into a [`BrickMap`].
fn voxelize_region(
    vmin: IVec3,
    vmax: IVec3,
    layer: &HeightLayer,
    lib: &BiomeLibrary,
    reg: &BlockRegistry,
) -> BrickMap {
    let bc_min = brick_coord_of_voxel(vmin);
    let bc_max = brick_coord_of_voxel(vmax);
    let mut map = BrickMap::new();
    for bz in bc_min.z..=bc_max.z {
        for by in bc_min.y..=bc_max.y {
            for bx in bc_min.x..=bc_max.x {
                let coord = IVec3::new(bx, by, bz);
                map.insert(coord, voxelize_brick(coord, 0, layer, lib, reg, SEED));
            }
        }
    }
    map
}

/// CPU ground truth: DDA-march the brickmap along the world ray and return the FIRST solid voxel —
/// `(block_id, world_voxel, world_t)`. Mirrors the shader's per-voxel stepping (centre-plane entry t),
/// at world-voxel granularity (the shader walks the same `VOXEL_SIZE` grid). Returns `None` on a clean miss
/// within `t_max`.
fn cpu_first_solid(
    map: &BrickMap,
    ro: Vec3,
    rd: Vec3,
    t_max: f32,
) -> Option<(BlockId, IVec3, f32)> {
    let rd = rd.normalize();
    // Standard world-grid 3D-DDA over VOXEL_SIZE (0.05 m) voxels.
    let step = IVec3::new(rd.x.signum() as i32, rd.y.signum() as i32, rd.z.signum() as i32);
    let inv = Vec3::new(1.0 / rd.x, 1.0 / rd.y, 1.0 / rd.z);
    let mut vox = IVec3::new(
        (ro.x / VOXEL_SIZE).floor() as i32,
        (ro.y / VOXEL_SIZE).floor() as i32,
        (ro.z / VOXEL_SIZE).floor() as i32,
    );
    let next_boundary = Vec3::new(
        (vox.x + step.x.max(0)) as f32 * VOXEL_SIZE,
        (vox.y + step.y.max(0)) as f32 * VOXEL_SIZE,
        (vox.z + step.z.max(0)) as f32 * VOXEL_SIZE,
    );
    // Axis-aligned rays (rd component == 0) must never step/terminate on that axis: give it +inf.
    let big = f32::MAX;
    let pick = |z: bool, v: f32| if z { big } else { v };
    let mut t_max_axis = Vec3::new(
        pick(rd.x.abs() < 1e-12, (next_boundary.x - ro.x) * inv.x),
        pick(rd.y.abs() < 1e-12, (next_boundary.y - ro.y) * inv.y),
        pick(rd.z.abs() < 1e-12, (next_boundary.z - ro.z) * inv.z),
    );
    let t_delta = Vec3::new(
        pick(rd.x.abs() < 1e-12, (VOXEL_SIZE * inv.x).abs()),
        pick(rd.y.abs() < 1e-12, (VOXEL_SIZE * inv.y).abs()),
        pick(rd.z.abs() < 1e-12, (VOXEL_SIZE * inv.z).abs()),
    );
    let mut t_cur = 0.0f32;

    for _ in 0..4096 {
        if t_cur > t_max {
            return None;
        }
        let block = map.voxel_block(vox);
        if !block.is_air() {
            return Some((block, vox, t_cur));
        }
        // Advance across the nearest axis boundary.
        if t_max_axis.x < t_max_axis.y && t_max_axis.x < t_max_axis.z {
            t_cur = t_max_axis.x;
            t_max_axis.x += t_delta.x;
            vox.x += step.x;
        } else if t_max_axis.y < t_max_axis.z {
            t_cur = t_max_axis.y;
            t_max_axis.y += t_delta.y;
            vox.y += step.y;
        } else {
            t_cur = t_max_axis.z;
            t_max_axis.z += t_delta.z;
            vox.z += step.z;
        }
    }
    None
}

#[test]
fn gpu_ray_query_hit_matches_cpu_ground_truth() {
    let Some((device, queue)) = common::headless_ray_query_device() else {
        eprintln!("no ray-query device — skipping gpu_ray_query_hit_matches_cpu_ground_truth");
        return;
    };

    // --- CPU: voxelize a small known region around the origin surface ---
    let layer = test_layer();
    let lib = test_library();
    let reg = BlockRegistry::from_biome_library(&lib);

    // A region a few bricks wide around the origin column's surface. Find the surface to centre the Y band.
    let surf_h = layer.sample_world(0.0, 0.0, SEED).height;
    let surf_vy = (surf_h / VOXEL_SIZE).floor() as i32;
    // ~3×3 bricks in XZ, a Y band straddling the surface (a few bricks tall).
    let span = BRICK_EDGE; // one brick to each side in XZ
    let vmin = IVec3::new(-span, surf_vy - 2 * BRICK_EDGE, -span);
    let vmax = IVec3::new(span, surf_vy + BRICK_EDGE, span);
    let map = voxelize_region(vmin, vmax, &layer, &lib, &reg);
    assert!(!map.is_empty(), "the region must contain terrain bricks");

    let patch = pack_brickmap(&map, &reg);
    assert!(!patch.is_empty());

    let t_max = 100.0f32;

    // --- GPU: build the per-brick AABB BLAS + a TLAS of one identity instance (AABBs are world-space) ---
    let n = patch.brick_count() as u32;

    let aabb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("brick_aabbs"),
        contents: bytemuck::cast_slice(&patch.aabbs),
        usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE,
    });
    let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("brick_metas"),
        contents: bytemuck::cast_slice(&patch.metas),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let voxel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("brick_voxels"),
        contents: bytemuck::cast_slice(&patch.voxels),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let palette_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("brick_palette"),
        contents: bytemuck::cast_slice(&patch.palette),
        usage: wgpu::BufferUsages::STORAGE,
    });
    // Storage plan R2b — the per-brick palettes the bit-packed index stream indirects through.
    let brick_palettes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("brick_palette_brick_palettes"),
        contents: bytemuck::cast_slice(&patch.brick_palettes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    // A3 — ONE identity descriptor 0 (the whole scene = the streamed-world degenerate case).
    let descriptors_buf = common::instance_descriptors_buffer(&device);

    let size_desc = wgpu::BlasAABBGeometrySizeDescriptor {
        primitive_count: n,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("brick_blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::AABBs { descriptors: vec![size_desc.clone()] },
    );
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("brick_tlas"),
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

    // Ray uniform (rewritten per ray) + output buffers, reused across the rays we test.
    let ray_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ray"),
        size: mem::size_of::<RayUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hit"),
        size: mem::size_of::<GpuHit>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read"),
        size: mem::size_of::<GpuHit>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Pipeline from the real shader, `trace_one` entry.
    let src = common::voxel_raytrace_shader_src();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_raytrace"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("trace_one"),
        layout: None,
        module: &shader,
        entry_point: Some("trace_one"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::AccelerationStructure(&tlas) },
            wgpu::BindGroupEntry { binding: 1, resource: meta_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: voxel_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: palette_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: brick_palettes_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 13, resource: descriptors_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: ray_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
        ],
    });
    // `trace_one` references the group-1 lighting uniform for its shadow ray; bind the SSOT defaults.
    let light_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lighting"),
        contents: bytemuck::bytes_of(&lighting_uniform()),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let sky_buf = common::sky_uniform_buffer(&device);
    let light_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lighting_bg"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: sky_buf.as_entire_binding() },
        ],
    });

    // Build the BLAS/TLAS ONCE (the scene is static across the rays).
    let mut build = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("build") });
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

    // Run one ray through the GPU shader and read back its `GpuHit`.
    let run_ray = |ro: Vec3, rd: Vec3| -> GpuHit {
        let ray = RayUniform { origin: ro.into(), t_min: 0.0, dir: rd.into(), t_max };
        queue.write_buffer(&ray_buf, 0, bytemuck::bytes_of(&ray));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, Some(&bind_group), &[]);
            cpass.set_bind_group(1, Some(&light_bg), &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, mem::size_of::<GpuHit>() as u64);
        queue.submit(Some(encoder.finish()));
        let slice = read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
        let data = slice.get_mapped_range().unwrap();
        let gpu: GpuHit = *bytemuck::from_bytes(&data);
        drop(data);
        read_buf.unmap();
        gpu
    };

    // Assert the GPU hit for `(ro, rd)` matches the CPU ground truth (block id, world-t within a voxel,
    // and the committed palette colour). The oracle: same ray, same first-solid voxel ⇒ same answer.
    let assert_matches = |label: &str, ro: Vec3, rd: Vec3| {
        let rd = rd.normalize();
        let (cpu_block, cpu_vox, cpu_t) =
            cpu_first_solid(&map, ro, rd, t_max).expect("CPU ray must hit the terrain surface");
        assert!(!cpu_block.is_air());
        let gpu = run_ray(ro, rd);
        eprintln!(
            "[{label}] GPU: hit={} block_id={} prim={} t={:.4} color={:?}",
            gpu.hit, gpu.block_id, gpu.prim, gpu.t, gpu.color
        );
        eprintln!("[{label}] CPU: block_id={} vox={:?} t={:.4}", cpu_block.0, cpu_vox, cpu_t);

        assert_eq!(gpu.hit, 1, "[{label}] GPU ray must hit (CPU did at t={cpu_t:.3})");
        assert_eq!(gpu.block_id, cpu_block.0 as u32, "[{label}] GPU first-solid block id must match CPU");
        assert!(
            (gpu.t - cpu_t).abs() <= VOXEL_SIZE + 1e-3,
            "[{label}] GPU hit-t {} must match CPU hit-t {} within one voxel",
            gpu.t,
            cpu_t
        );
        let expected = reg.color(cpu_block);
        for (c, &exp) in expected.iter().enumerate() {
            assert!(
                (gpu.color[c] - exp).abs() < 1e-4,
                "[{label}] GPU committed colour channel {c} ({}) must equal palette ({})",
                gpu.color[c],
                exp
            );
        }
    };

    let ro = Vec3::new(BRICK_WORLD_SIZE * 0.5, surf_h + 5.0, BRICK_WORLD_SIZE * 0.5);
    // Case 1: a slightly-tilted downward ray (general DDA path).
    assert_matches("tilted_down", ro, Vec3::new(0.02, -1.0, 0.015));
    // Case 2: a perfectly axis-aligned straight-DOWN ray (the degenerate zero-direction DDA case — must
    // be handled robustly, not skip the surface layer).
    assert_matches("straight_down", ro, Vec3::new(0.0, -1.0, 0.0));

    // Keep the GPU scene objects alive until all rays have run.
    let _ = (&aabb_buf, &meta_buf, &voxel_buf, &palette_buf, &blas, &tlas);
}

/// CPU ground truth over a PACKED [`GpuBrickPatch`] with per-brick LOD — the exact mirror of the WGSL
/// `trace`: for every resident brick whose world AABB the ray crosses, coarse-DDA its `lod_edge(lod)³` grid
/// (cells of `lod_voxel_size(lod)` m) and keep the NEAREST first-solid hit across all bricks. Returns
/// `(block_id, world_t)` of the committed hit, or `None` on a miss. This walks the SAME coarse grid the GPU
/// does, so a coarse-LOD brick is validated at its coarse resolution.
fn cpu_first_solid_packed(patch: &GpuBrickPatch, ro: Vec3, rd: Vec3, t_max: f32) -> Option<(u32, f32)> {
    let rd = rd.normalize();
    let mut best: Option<(u32, f32)> = None;
    for (bi, m) in patch.metas.iter().enumerate() {
        let bmin = Vec3::from(m.world_min);
        let bmax = bmin + Vec3::splat(brick_span(m.lod())); // clipmap: coarse bricks span 2^lod× more world
        // Ray/AABB slab test → [t_enter, t_exit].
        let inv = Vec3::new(1.0 / rd.x, 1.0 / rd.y, 1.0 / rd.z);
        let ta = (bmin - ro) * inv;
        let tb = (bmax - ro) * inv;
        let tmin3 = ta.min(tb);
        let tmax3 = ta.max(tb);
        let t_enter = tmin3.x.max(tmin3.y).max(tmin3.z);
        let t_exit = tmax3.x.min(tmax3.y).min(tmax3.z);
        if t_enter > t_exit || t_exit < 0.0 || t_enter > t_max {
            continue;
        }
        let lod = m.lod();
        let edge = lod_edge(lod);
        let csize = lod_voxel_size(lod);
        // Coarse-DDA exactly like dda_brick.
        let t0 = t_enter.max(0.0);
        let p_enter = ro + rd * (t0 + 1e-4);
        let local = (p_enter - bmin) / csize;
        let mut vox = IVec3::new(local.x.floor() as i32, local.y.floor() as i32, local.z.floor() as i32);
        vox = vox.clamp(IVec3::ZERO, IVec3::splat(edge - 1));
        let step = IVec3::new(rd.x.signum() as i32, rd.y.signum() as i32, rd.z.signum() as i32);
        let next_boundary = bmin
            + Vec3::new(
                (vox.x + step.x.max(0)) as f32,
                (vox.y + step.y.max(0)) as f32,
                (vox.z + step.z.max(0)) as f32,
            ) * csize;
        let big = f32::MAX;
        let pick = |z: bool, v: f32| if z { big } else { v };
        let nz = |c: f32| c.abs() < 1e-12;
        let mut t_axis = Vec3::new(
            pick(nz(rd.x), (next_boundary.x - ro.x) * inv.x),
            pick(nz(rd.y), (next_boundary.y - ro.y) * inv.y),
            pick(nz(rd.z), (next_boundary.z - ro.z) * inv.z),
        );
        let t_delta = Vec3::new(
            pick(nz(rd.x), (csize * inv.x).abs()),
            pick(nz(rd.y), (csize * inv.y).abs()),
            pick(nz(rd.z), (csize * inv.z).abs()),
        );
        let mut t_cur = t0;
        for _ in 0..(3 * BRICK_EDGE) {
            if vox.x < 0 || vox.x >= edge || vox.y < 0 || vox.y >= edge || vox.z < 0 || vox.z >= edge {
                break;
            }
            // Read the cell via the production SSOT `GpuBrickPatch::cell_block` (storage plan R2b) — a UNIFORM
            // brick returns its single meta id; a DENSE brick decodes the bit-packed index + per-brick palette,
            // EXACTLY as the GPU `cell_block` does. The core cell `vox` lives at haloed index `vox + 1` (the
            // `(edge+2)³` haloed grid, core cells at halo index [1, edge]).
            let hedge = edge + 2;
            let idx = ((vox.x + 1) + (vox.y + 1) * hedge + (vox.z + 1) * hedge * hedge) as usize;
            let id = patch.cell_block(m, idx).0 as u32;
            if id != 0 {
                if best.map(|(_, bt)| t_cur < bt).unwrap_or(true) {
                    best = Some((id, t_cur));
                }
                break;
            }
            if t_axis.x < t_axis.y && t_axis.x < t_axis.z {
                t_cur = t_axis.x;
                t_axis.x += t_delta.x;
                vox.x += step.x;
            } else if t_axis.y < t_axis.z {
                t_cur = t_axis.y;
                t_axis.y += t_delta.y;
                vox.y += step.y;
            } else {
                t_cur = t_axis.z;
                t_axis.z += t_delta.z;
                vox.z += step.z;
            }
            if t_cur > t_exit {
                break;
            }
        }
        let _ = bi;
    }
    best
}

/// **The multi-LOD CLIPMAP GPU oracle.** Builds a brick set with MIXED per-brick LODs placed at their TRUE
/// clipmap world positions — different LODs are DIFFERENT coord grids (`world_min = coord · brick_span(lod)`,
/// a coarse brick spans `2^lod×` more world). Packs it via the SSOT `pack_resident_set`, builds the
/// BLAS/TLAS, runs the REAL `voxel_raytrace.wgsl` for several rays, and asserts each GPU hit (block id +
/// world-t) matches a CPU DDA over the SAME LOD'd packed grids. The key proof that a coarse-LOD brick is
/// marched over its coarse-cell grid (covering more world) correctly on the GPU.
#[test]
fn gpu_mixed_lod_matches_cpu_ground_truth() {
    let Some((device, queue)) = common::headless_ray_query_device() else {
        eprintln!("no ray-query device — skipping gpu_mixed_lod_matches_cpu_ground_truth");
        return;
    };

    let reg = {
        let lib = test_library();
        BlockRegistry::from_biome_library(&lib)
    };

    // Three side-by-side bands along +X, each at its OWN LOD's world span (the clipmap layout). A fully-solid
    // floor brick (block 1) fills each band's X cells; the LOD1 pillar brick gets a block-2 top half so a
    // coarse-cell read is observable. Each band is placed at an explicit GRID-ALIGNED coord range so the
    // bricks' world bounds (`coord · brick_span(lod)`) are exact (no rounding gaps).
    // (world-X annotations at the 0.05 m flip; ranges = coord·brick_span(lod), derived in the asserts below):
    //   LOD0: coords {0,1} → world X [0, 0.8)   (span 0.4)
    //   LOD1: coords {2,3} → world X [1.6, 3.2)  (span 0.8; pillar in coord 3 = [2.4, 3.2))
    //   LOD2: coords {2,3} → world X [3.2, 6.4)  (span 1.6)
    let mut entries_owned: Vec<(IVec3, Brick, u32)> = Vec::new();
    let solid_floor = |pillar: bool| {
        let mut v = Box::new([BlockId(1); BRICK_EDGE as usize * BRICK_EDGE as usize * BRICK_EDGE as usize]);
        if pillar {
            for zz in 0..BRICK_EDGE {
                for yy in 4..BRICK_EDGE {
                    for xx in 0..BRICK_EDGE {
                        v[(xx + yy * BRICK_EDGE + zz * BRICK_EDGE * BRICK_EDGE) as usize] = BlockId(2);
                    }
                }
            }
        }
        Brick::from_voxels(v)
    };
    entries_owned.push((IVec3::new(0, 0, 0), solid_floor(false), 0));
    entries_owned.push((IVec3::new(1, 0, 0), solid_floor(false), 0));
    entries_owned.push((IVec3::new(2, 0, 0), solid_floor(false), 1));
    entries_owned.push((IVec3::new(3, 0, 0), solid_floor(true), 1)); // pillar
    entries_owned.push((IVec3::new(2, 0, 0), solid_floor(false), 2));
    entries_owned.push((IVec3::new(3, 0, 0), solid_floor(false), 2));

    let entries: Vec<ResidentBrick> =
        entries_owned.iter().map(|(c, b, l)| ResidentBrick { coord: *c, brick: b, lod: *l }).collect();
    let patch = pack_resident_set(&entries, &reg);
    assert!(!patch.is_empty());
    // Sanity: mixed LODs actually present in the packed metas.
    let lods: std::collections::BTreeSet<u32> = patch.metas.iter().map(|m| m.lod()).collect();
    assert!(lods.contains(&0) && lods.contains(&1) && lods.contains(&2), "mixed LODs packed: {lods:?}");

    let t_max = 100.0f32;
    let n = patch.brick_count() as u32;

    // --- build GPU scene from the packed patch ---
    let aabb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mlod_aabbs"),
        contents: bytemuck::cast_slice(&patch.aabbs),
        usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE,
    });
    let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mlod_metas"),
        contents: bytemuck::cast_slice(&patch.metas),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let voxel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mlod_voxels"),
        contents: bytemuck::cast_slice(&patch.voxels),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let palette_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mlod_palette"),
        contents: bytemuck::cast_slice(&patch.palette),
        usage: wgpu::BufferUsages::STORAGE,
    });
    // Storage plan R2b — the per-brick palettes the bit-packed index stream indirects through.
    let brick_palettes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mlod_palette_brick_palettes"),
        contents: bytemuck::cast_slice(&patch.brick_palettes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    // A3 — ONE identity descriptor 0 (the whole scene = the streamed-world degenerate case).
    let descriptors_buf = common::instance_descriptors_buffer(&device);

    let size_desc = wgpu::BlasAABBGeometrySizeDescriptor {
        primitive_count: n,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("mlod_blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::AABBs { descriptors: vec![size_desc.clone()] },
    );
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("mlod_tlas"),
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

    let ray_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mlod_ray"),
        size: mem::size_of::<RayUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mlod_hit"),
        size: mem::size_of::<GpuHit>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mlod_read"),
        size: mem::size_of::<GpuHit>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let src = common::voxel_raytrace_shader_src();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_raytrace"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("trace_one"),
        layout: None,
        module: &shader,
        entry_point: Some("trace_one"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::AccelerationStructure(&tlas) },
            wgpu::BindGroupEntry { binding: 1, resource: meta_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: voxel_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: palette_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: brick_palettes_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 13, resource: descriptors_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: ray_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
        ],
    });
    let light_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mlod_lighting"),
        contents: bytemuck::bytes_of(&lighting_uniform()),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let sky_buf = common::sky_uniform_buffer(&device);
    let light_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mlod_lighting_bg"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: sky_buf.as_entire_binding() },
        ],
    });

    let mut build = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("mlod_build") });
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

    let run_ray = |ro: Vec3, rd: Vec3| -> GpuHit {
        let ray = RayUniform { origin: ro.into(), t_min: 0.0, dir: rd.normalize().into(), t_max };
        queue.write_buffer(&ray_buf, 0, bytemuck::bytes_of(&ray));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, Some(&bind_group), &[]);
            cpass.set_bind_group(1, Some(&light_bg), &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, mem::size_of::<GpuHit>() as u64);
        queue.submit(Some(encoder.finish()));
        let slice = read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
        let data = slice.get_mapped_range().unwrap();
        let gpu: GpuHit = *bytemuck::from_bytes(&data);
        drop(data);
        read_buf.unmap();
        gpu
    };

    // Band world-X layout (the clipmap, grid-aligned coords; 0.05 m flip): LOD0 coords {0,1} = X [0, 0.8); LOD1
    // coords {2,3} = X [1.6, 3.2) (pillar brick coord 3 = [2.4, 3.2)); LOD2 coords {2,3} = X [3.2, 6.4). Each
    // band's bricks span their LOD's world height [0, brick_span(lod)) in Y/Z. All ray origins derive from these.
    let s0 = BRICK_WORLD_SIZE; // 0.4
    let s1 = brick_span(1); // 0.8
    let s2 = brick_span(2); // 1.6
    let assert_matches = |label: &str, ro: Vec3, rd: Vec3| {
        let (cpu_id, cpu_t) =
            cpu_first_solid_packed(&patch, ro, rd, t_max).expect("CPU ray must hit a brick");
        let gpu = run_ray(ro, rd);
        eprintln!("[{label}] GPU hit={} id={} t={:.4} | CPU id={cpu_id} t={cpu_t:.4}", gpu.hit, gpu.block_id, gpu.t);
        assert_eq!(gpu.hit, 1, "[{label}] GPU must hit (CPU did at t={cpu_t:.3})");
        assert_eq!(gpu.block_id, cpu_id, "[{label}] block id must match the coarse-LOD CPU oracle");
        assert!(
            (gpu.t - cpu_t).abs() <= lod_voxel_size(2) + 1e-3,
            "[{label}] GPU t {} vs CPU t {} within one (coarse) cell",
            gpu.t,
            cpu_t
        );
    };

    // Ray 1: straight DOWN into the LOD0 band (full-res floor, block 1), from above its top (Y = s0).
    assert_matches("down_lod0", Vec3::new(s0 * 0.5, s0 + 2.0, s0 * 0.5), Vec3::new(0.0, -1.0, 0.0));
    // Ray 2: straight DOWN into the LOD1 PILLAR brick (coord 3, world X [2.4, 3.2)): its top half (world Y
    // [0.4, 0.8)) is block 2, marched on the LOD1 (0.1 m) coarse grid — the surface hit must be block 2.
    assert_matches("down_lod1_pillar", Vec3::new(3.0 * s1 + s1 * 0.5, s1 + 2.0, s1 * 0.5), Vec3::new(0.0, -1.0, 0.0));
    // Ray 3: straight DOWN into the LOD2 band (coord 2, world X [3.2, 4.8), 0.2 m coarse grid, block 1).
    assert_matches("down_lod2", Vec3::new(2.0 * s2 + s2 * 0.5, s2 + 2.0, s2 * 0.5), Vec3::new(0.0, -1.0, 0.0));
    // Ray 4: a horizontal ray skimming +X from before the LOD0 band, into the LOD0 floor — the nearest solid
    // hit (the first LOD0 brick's −X wall) must win the TLAS merge across mixed-LOD AABBs.
    assert_matches("across_mixed", Vec3::new(-2.0, s0 * 0.5, s0 * 0.5), Vec3::new(1.0, 0.0, 0.0));

    let _ = (&aabb_buf, &meta_buf, &voxel_buf, &palette_buf, &blas, &tlas);
}

/// **A3 Stage 3 acceptance — the 2-INSTANCE scene oracle (multi-instance TLAS + per-instance object-local DDA).**
///
/// Builds a TLAS with TWO instances: the streamed WORLD (descriptor 0, identity transform, meta_base 0) + a
/// translated CUBE OBJECT (descriptor 1, a pure-translation `world_from_object`, meta_base = the world's slot
/// count). Each instance is its own BLAS reading its SLICE of the single shared aabb_buf via `primitive_offset`
/// (Stage 2 pinned that `primitive_index` is geometry-relative, so the descriptor's meta_base resolves the
/// global brick). The cube's bricks live in OBJECT space (origin at the cube's local 0); the TLAS instance
/// places it in the world, and the shader transforms the world ray into the cube's object space (via
/// descriptor.object_from_world = the inverse translation), DDA-marches the cube's bricks, and commits the
/// world-t. This proves the descriptor indirection + object-local DDA + world-t commit ACROSS instances —
/// `VOXEL_INSTANCING_PLAN` §7 Phase-1 acceptance — the load-bearing multi-instance correctness gate.
///
/// The CPU oracle traces EACH instance independently (world in world space, cube in its object space with the
/// ray pre-transformed) and asserts the GPU's nearest-across-instances hit matches.
#[test]
fn gpu_two_instance_world_plus_cube_matches_cpu() {
    let Some((device, queue)) = common::headless_ray_query_device() else {
        eprintln!("no ray-query device — skipping gpu_two_instance_world_plus_cube_matches_cpu");
        return;
    };

    let reg = {
        let lib = test_library();
        BlockRegistry::from_biome_library(&lib)
    };

    // --- WORLD: a few LOD0 floor bricks (block 1) at coords (0..2, 0, 0..2), placed at the clipmap origin. ---
    let world_brick = Brick::from_voxels(Box::new(
        [BlockId(1); BRICK_EDGE as usize * BRICK_EDGE as usize * BRICK_EDGE as usize],
    ));
    let mut world_entries_owned: Vec<(IVec3, Brick)> = Vec::new();
    for bz in 0..2 {
        for bx in 0..2 {
            world_entries_owned.push((IVec3::new(bx, 0, bz), world_brick.clone()));
        }
    }
    let world_entries: Vec<ResidentBrick> =
        world_entries_owned.iter().map(|(c, b)| ResidentBrick { coord: *c, brick: b, lod: 0 }).collect();
    let world_patch = pack_resident_set(&world_entries, &reg);
    let n_world = world_patch.brick_count() as u32;
    assert!(n_world > 0, "world must have bricks");

    // --- CUBE OBJECT: a single solid LOD0 brick (block 2) at OBJECT coord (0,0,0). Its world AABBs (in object
    // space) are [0, 0.4)³. The TLAS instance translates it to `cube_t` in the world. ---
    let cube_brick = Brick::from_voxels(Box::new(
        [BlockId(2); BRICK_EDGE as usize * BRICK_EDGE as usize * BRICK_EDGE as usize],
    ));
    let cube_entries = vec![ResidentBrick { coord: IVec3::new(0, 0, 0), brick: &cube_brick, lod: 0 }];
    let cube_patch = pack_resident_set(&cube_entries, &reg);
    let n_cube = cube_patch.brick_count() as u32;
    assert!(n_cube > 0, "cube must have bricks");

    // Place the cube well away from the world floor (which spans world X/Z [0, 0.8) at the 0.05 m flip). Put it
    // at X = 10 m so a ray down into X≈10 hits ONLY the cube and a ray down into X≈0.2 hits ONLY the world.
    let cube_t = Vec3::new(10.0, 0.0, 0.0); // world_from_object translation
    let span = BRICK_WORLD_SIZE; // 0.4 — the cube's object-local extent per axis

    // --- Concatenate the world + cube into GLOBAL buffers. The cube's metas are REBASED so their voxel_offset /
    // palette_base point past the world's slices (the descriptor's meta_base selects the cube's brick range; the
    // voxel/palette bases are folded into the metas here so the Stage-1 shader's `voxel_indices[m.voxel_offset+…]`
    // and `brick_palettes[m.palette_base+…]` address the concatenated buffers with no shader change). ---
    let voxel_rebase = world_patch.voxels.len() as u32;
    let pal_rebase = world_patch.brick_palettes.len() as u32;
    let cube_metas_rebased: Vec<adventure::voxel::gpu::GpuBrickMeta> = cube_patch
        .metas
        .iter()
        .map(|m| {
            let mut m = *m;
            if !m.is_uniform() {
                m.voxel_offset = m.voxel_offset.wrapping_add(voxel_rebase);
                m.palette_base = m.palette_base.wrapping_add(pal_rebase);
            }
            m
        })
        .collect();

    let mut metas = world_patch.metas.clone();
    metas.extend_from_slice(&cube_metas_rebased);
    let mut aabbs = world_patch.aabbs.clone();
    aabbs.extend_from_slice(&cube_patch.aabbs);
    // CLONE the cube's voxel/palette slices into the concatenation (do NOT drain `cube_patch` — the CPU oracle
    // below DDA-marches `cube_patch` in object space, so it still needs its own voxels/brick_palettes intact).
    let mut voxels = world_patch.voxels.clone();
    voxels.extend_from_slice(&cube_patch.voxels);
    let mut brick_palettes = world_patch.brick_palettes.clone();
    brick_palettes.extend_from_slice(&cube_patch.brick_palettes);
    // The palette is the shared registry (both instances index it directly) — use the world's.
    let palette = world_patch.palette.clone();

    // --- Descriptors: 0 = world (identity, meta_base 0); 1 = cube (translation, meta_base = n_world). ---
    // object_from_world for a pure translation `T` is `translate(-T)` (row-major 3×4): rows
    // [1,0,0,-Tx; 0,1,0,-Ty; 0,0,1,-Tz]. world_from_object_rot is the rotation (identity here — pure translation,
    // so the normal is unchanged in world). inv_scale = 1 (rigid). meta_base = n_world (the cube's slot base).
    let object_from_world = [
        1.0, 0.0, 0.0, -cube_t.x,
        0.0, 1.0, 0.0, -cube_t.y,
        0.0, 0.0, 1.0, -cube_t.z,
    ];
    let cube_desc = adventure::voxel::gpu::GpuInstanceDescriptor {
        object_from_world,
        world_from_object_rot: adventure::voxel::gpu::IDENTITY_3X4,
        meta_base: n_world,
        voxel_base: 0, // folded into the rebased cube metas
        palette_base: 0, // folded into the rebased cube metas
        inv_scale: 1.0,
        edit_base: adventure::voxel::gpu::INSTANCE_EDIT_NONE,
        mask: 0xff,
        _pad: [0; 2],
    };
    let descriptors =
        [adventure::voxel::gpu::GpuInstanceDescriptor::world_identity(0), cube_desc];

    // --- Upload the global buffers. ---
    let mk = |label, bytes: &[u8], extra| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE | extra,
        })
    };
    let aabb_buf = mk("ti_aabbs", bytemuck::cast_slice(&aabbs), wgpu::BufferUsages::BLAS_INPUT);
    let meta_buf = mk("ti_metas", bytemuck::cast_slice(&metas), wgpu::BufferUsages::empty());
    let voxel_buf = mk("ti_voxels", bytemuck::cast_slice(&voxels), wgpu::BufferUsages::empty());
    let palette_buf = mk("ti_palette", bytemuck::cast_slice(&palette), wgpu::BufferUsages::empty());
    let brick_palettes_buf =
        mk("ti_brick_palettes", bytemuck::cast_slice(&brick_palettes), wgpu::BufferUsages::empty());
    let descriptors_buf =
        mk("ti_descriptors", bytemuck::cast_slice(&descriptors), wgpu::BufferUsages::empty());

    // --- TWO BLASes over slices of the shared aabb_buf, and a 2-instance TLAS. ---
    let stride = mem::size_of::<adventure::voxel::gpu::GpuBrickAabb>() as wgpu::BufferAddress;
    let world_size = wgpu::BlasAABBGeometrySizeDescriptor {
        primitive_count: n_world,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let cube_size = wgpu::BlasAABBGeometrySizeDescriptor {
        primitive_count: n_cube,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let world_blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("ti_world_blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::AABBs { descriptors: vec![world_size.clone()] },
    );
    let cube_blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("ti_cube_blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::AABBs { descriptors: vec![cube_size.clone()] },
    );
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("ti_tlas"),
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        max_instances: 2,
    });
    // Instance 0 = world (identity transform, custom_index 0 → descriptor 0).
    tlas[0] = Some(wgpu::TlasInstance::new(
        &world_blas,
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        0,
        0xff,
    ));
    // Instance 1 = cube (world_from_object = translate(cube_t), custom_index 1 → descriptor 1).
    tlas[1] = Some(wgpu::TlasInstance::new(
        &cube_blas,
        [1.0, 0.0, 0.0, cube_t.x, 0.0, 1.0, 0.0, cube_t.y, 0.0, 0.0, 1.0, cube_t.z],
        1,
        0xff,
    ));

    // --- Pipeline + bind groups (the production trace_one entry). ---
    let ray_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ti_ray"),
        size: mem::size_of::<RayUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ti_hit"),
        size: mem::size_of::<GpuHit>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ti_read"),
        size: mem::size_of::<GpuHit>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let src = common::voxel_raytrace_shader_src();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_raytrace"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("trace_one"),
        layout: None,
        module: &shader,
        entry_point: Some("trace_one"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::AccelerationStructure(&tlas) },
            wgpu::BindGroupEntry { binding: 1, resource: meta_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: voxel_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: palette_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: brick_palettes_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 13, resource: descriptors_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: ray_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
        ],
    });
    let light_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ti_lighting"),
        contents: bytemuck::bytes_of(&lighting_uniform()),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let sky_buf = common::sky_uniform_buffer(&device);
    let light_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ti_lighting_bg"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: sky_buf.as_entire_binding() },
        ],
    });

    // Build BOTH BLASes (each over its slice of the shared aabb_buf via primitive_offset) + the 2-instance TLAS.
    let mut build = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ti_build") });
    let build_entries = [
        wgpu::BlasBuildEntry {
            blas: &world_blas,
            geometry: wgpu::BlasGeometries::AabbGeometries(vec![wgpu::BlasAabbGeometry {
                size: &world_size,
                stride,
                aabb_buffer: &aabb_buf,
                primitive_offset: 0, // world bricks occupy slots [0, n_world)
            }]),
        },
        wgpu::BlasBuildEntry {
            blas: &cube_blas,
            geometry: wgpu::BlasGeometries::AabbGeometries(vec![wgpu::BlasAabbGeometry {
                size: &cube_size,
                stride,
                aabb_buffer: &aabb_buf,
                primitive_offset: n_world * stride as u32, // cube bricks occupy slots [n_world, …)
            }]),
        },
    ];
    build.build_acceleration_structures(build_entries.iter(), iter::once(&tlas));
    queue.submit(Some(build.finish()));

    let run_ray = |ro: Vec3, rd: Vec3| -> GpuHit {
        let ray = RayUniform { origin: ro.into(), t_min: 0.0, dir: rd.normalize().into(), t_max: 100.0 };
        queue.write_buffer(&ray_buf, 0, bytemuck::bytes_of(&ray));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, Some(&bind_group), &[]);
            cpass.set_bind_group(1, Some(&light_bg), &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, mem::size_of::<GpuHit>() as u64);
        queue.submit(Some(encoder.finish()));
        let slice = read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
        let data = slice.get_mapped_range().unwrap();
        let gpu: GpuHit = *bytemuck::from_bytes(&data);
        drop(data);
        read_buf.unmap();
        gpu
    };

    // Ray A: straight DOWN into the WORLD floor (block 1) at world X≈0.2 (inside the world's [0,0.8) span,
    // away from the cube at X=10). The CPU oracle is the world patch in WORLD space.
    {
        let ro = Vec3::new(span * 0.5, span + 2.0, span * 0.5);
        let rd = Vec3::new(0.0, -1.0, 0.0);
        let (cpu_id, cpu_t) = cpu_first_solid_packed(&world_patch, ro, rd, 100.0).expect("world ray must hit");
        let gpu = run_ray(ro, rd);
        eprintln!("[world] GPU hit={} id={} t={:.4} | CPU id={cpu_id} t={cpu_t:.4}", gpu.hit, gpu.block_id, gpu.t);
        assert_eq!(gpu.hit, 1, "[world] must hit the floor");
        assert_eq!(gpu.block_id, cpu_id, "[world] block id matches the descriptor-0 world DDA");
        assert!((gpu.t - cpu_t).abs() <= VOXEL_SIZE + 1e-3, "[world] t {} vs CPU {}", gpu.t, cpu_t);
    }

    // Ray B: straight DOWN into the CUBE (block 2) at world (cube_t + half a brick). The GPU transforms the
    // world ray into cube-object space; the CPU oracle does the SAME (subtract cube_t) then DDAs the cube patch
    // in object space. World-t == object-t (pure translation), so they compare directly.
    {
        let ro = Vec3::new(cube_t.x + span * 0.5, cube_t.y + span + 2.0, cube_t.z + span * 0.5);
        let rd = Vec3::new(0.0, -1.0, 0.0);
        // CPU object-space ray = world ray shifted by -cube_t (the descriptor's object_from_world).
        let ro_obj = ro - cube_t;
        let (cpu_id, cpu_t) =
            cpu_first_solid_packed(&cube_patch, ro_obj, rd, 100.0).expect("cube ray must hit");
        let gpu = run_ray(ro, rd);
        eprintln!("[cube] GPU hit={} id={} t={:.4} | CPU id={cpu_id} t={cpu_t:.4}", gpu.hit, gpu.block_id, gpu.t);
        assert_eq!(gpu.hit, 1, "[cube] must hit the translated cube");
        assert_eq!(gpu.block_id, cpu_id, "[cube] block id matches the descriptor-1 OBJECT-LOCAL DDA");
        assert!((gpu.t - cpu_t).abs() <= VOXEL_SIZE + 1e-3, "[cube] world-t {} vs CPU object-t {}", gpu.t, cpu_t);
        // The cube hit must NOT be the world block — proving the descriptor indirection selected the cube's
        // slot range (meta_base = n_world), not the world's.
        assert_eq!(gpu.block_id, 2, "[cube] the cube is block 2 (distinct from the world's block 1)");
    }

    // Ray C: a ray that would MISS the world but hits the cube — confirms the cube instance is independently
    // reachable (the TLAS merges both instances). Fire down at X = cube centre, far outside the world span.
    {
        let ro = Vec3::new(cube_t.x + span * 0.5, 50.0, cube_t.z + span * 0.5);
        let rd = Vec3::new(0.0, -1.0, 0.0);
        let gpu = run_ray(ro, rd);
        assert_eq!(gpu.hit, 1, "[cube-far] the distant cube is reachable via its TLAS instance");
        assert_eq!(gpu.block_id, 2, "[cube-far] hits the cube (block 2)");
    }

    let _ = (&aabb_buf, &meta_buf, &voxel_buf, &palette_buf, &brick_palettes_buf, &world_blas, &cube_blas, &tlas);
}
