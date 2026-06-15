//! **The brick-seam correctness ORACLE.**
//!
//! DEFECT under test: rays slipping BETWEEN adjacent bricks at their shared faces, hitting nothing — the
//! "black lines along brick boundaries" the user sees on the Cornell box. This rig builds a CONTINUOUS solid
//! wall out of a block of ADJACENT fully-solid bricks (so there is no real gap anywhere — every world point
//! in the wall volume is solid), then fires a DENSE grid of primary rays straight at the wall face, with the
//! grid deliberately spanning several brick boundaries. Every ray MUST hit a solid voxel and read back the
//! wall colour: a miss (or a wrong colour) at a brick boundary is the seam bug.
//!
//! It runs the REAL `voxel_raytrace.wgsl` `trace_one` entry on a real ray-query device, so it proves the
//! fix end-to-end through the actual shader + the SSOT packing. Skips cleanly without a ray-query adapter.

use std::iter;
use std::mem;

use bevy::math::{IVec3, Vec3};
use wgpu::util::DeviceExt;

use adventure::sdf_render::worldgen::biome::{
    BiomeDef, BiomeId, BiomeLibrary, StrataLayer, TerrainMatId, TerrainSurfaceMaterial,
};
use adventure::voxel::brickmap::{BRICK_WORLD_SIZE, Brick, BrickMap};
use adventure::voxel::gpu::pack_brickmap;
use adventure::voxel::palette::{BlockId, BlockRegistry};

mod common;

// Mirror of the WGSL `Hit` struct (binding 5) — same as the other GPU rigs.
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

// Mirror of the WGSL `RayUniform` (binding 4).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RayUniform {
    origin: [f32; 3],
    t_min: f32,
    dir: [f32; 3],
    t_max: f32,
}

/// A minimal two-material library so the wall block (id 1) has a distinct, known colour.
fn wall_library() -> BiomeLibrary {
    let mat = |name: &str, c: [f32; 4]| TerrainSurfaceMaterial {
        name: name.into(),
        base_color: c,
        roughness: 0.9,
        blend: 0.0,
        texture: None,
        tiling: 4.0,
        ..Default::default()
    };
    let materials = vec![mat("air_unused", [0.0, 0.0, 0.0, 1.0]), mat("wall", [0.8, 0.2, 0.05, 1.0])];
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

/// Build a CONTINUOUS solid wall from a `nx × ny × nz` block of fully-solid bricks (block id 1), brick
/// coordinates `[0,nx) × [0,ny) × [0,nz)`. Every voxel in the block is solid, so the wall has NO gaps — any
/// ray entering the block's world AABB and travelling through it must hit a solid voxel.
fn solid_wall_map(nx: i32, ny: i32, nz: i32, wall: BlockId) -> BrickMap {
    let mut map = BrickMap::new();
    for bz in 0..nz {
        for by in 0..ny {
            for bx in 0..nx {
                map.insert(IVec3::new(bx, by, bz), Brick::uniform(wall));
            }
        }
    }
    map
}

#[test]
fn gpu_continuous_wall_has_no_brick_seams() {
    let Some((device, queue)) = common::headless_ray_query_device() else {
        eprintln!("no ray-query device — skipping gpu_continuous_wall_has_no_brick_seams");
        return;
    };

    let lib = wall_library();
    let reg = BlockRegistry::from_biome_library(&lib);
    let wall = BlockId(1);
    let wall_color = reg.color(wall);

    // A 3×3×3 block of solid bricks → a cube of solid voxels spanning [0, 3·S) on each axis (S = brick size).
    // The −Z face of the cube (z = 0) is the wall face we fire rays at; the X/Y grid of rays spans the
    // internal brick boundaries (at x,y = S and 2S) so every seam between adjacent bricks is sampled.
    let (nx, ny, nz) = (3, 3, 3);
    let map = solid_wall_map(nx, ny, nz, wall);
    let patch = pack_brickmap(&map, &reg);
    assert!(!patch.is_empty());
    assert_eq!(patch.brick_count(), (nx * ny * nz) as usize);

    let s = BRICK_WORLD_SIZE;
    let span_x = nx as f32 * s; // wall face extent in X
    let span_y = ny as f32 * s; // wall face extent in Y
    let t_max = 100.0f32;
    let n = patch.brick_count() as u32;

    // --- GPU scene ---
    let aabb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("seam_aabbs"),
        contents: bytemuck::cast_slice(&patch.aabbs),
        usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE,
    });
    let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("seam_metas"),
        contents: bytemuck::cast_slice(&patch.metas),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let voxel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("seam_voxels"),
        contents: bytemuck::cast_slice(&patch.voxels),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let palette_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("seam_palette"),
        contents: bytemuck::cast_slice(&patch.palette),
        usage: wgpu::BufferUsages::STORAGE,
    });
    // Storage plan R2b — the per-brick palettes the bit-packed index stream indirects through.
    let brick_palettes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("seam_palette_brick_palettes"),
        contents: bytemuck::cast_slice(&patch.brick_palettes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let size_desc = wgpu::BlasAABBGeometrySizeDescriptor {
        primitive_count: n,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("seam_blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::AABBs { descriptors: vec![size_desc.clone()] },
    );
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("seam_tlas"),
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
        label: Some("seam_ray"),
        size: mem::size_of::<RayUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("seam_hit"),
        size: mem::size_of::<GpuHit>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("seam_read"),
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
    let descriptors_buf = common::instance_descriptors_buffer(&device); // A3: one identity descriptor 0
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
        label: Some("seam_lighting"),
        contents: bytemuck::bytes_of(&adventure::voxel::raytrace::LightingUniformData::default()),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let sky_buf = common::sky_uniform_buffer(&device);
    let light_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("seam_lighting_bg"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: sky_buf.as_entire_binding() },
        ],
    });

    let mut build = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("seam_build") });
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
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
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

    // Fire a DENSE grid of rays straight at the wall's −Z face (rays go +Z). The grid spans the full wall
    // face including the internal brick boundaries (x,y = S and 2S). The origins sit just in front of the
    // wall (z = -1) so each ray travels +Z into the solid cube. Every ray MUST hit the wall colour.
    let grid = 41usize; // odd → samples land exactly on the boundary planes
    let mut misses = 0usize;
    let mut wrong_color = 0usize;
    let mut boundary_misses = 0usize;
    let eps = 1e-3f32;
    for iy in 0..grid {
        for ix in 0..grid {
            // Sample across [eps, span - eps] so we stay inside the face but still hit the internal seams.
            let fx = ix as f32 / (grid - 1) as f32;
            let fy = iy as f32 / (grid - 1) as f32;
            let x = eps + fx * (span_x - 2.0 * eps);
            let y = eps + fy * (span_y - 2.0 * eps);
            let ro = Vec3::new(x, y, -1.0);
            let rd = Vec3::new(0.0, 0.0, 1.0);
            let gpu = run_ray(ro, rd);
            // Is this ray aimed at (within FP) an internal brick boundary plane?
            let on_boundary = (1..nx).any(|b| (x - b as f32 * s).abs() < 1e-3)
                || (1..ny).any(|b| (y - b as f32 * s).abs() < 1e-3);
            if gpu.hit != 1 {
                misses += 1;
                if on_boundary {
                    boundary_misses += 1;
                }
                if misses <= 12 {
                    eprintln!("MISS at ({x:.4},{y:.4}) on_boundary={on_boundary}");
                }
            } else if (gpu.color[0] - wall_color[0]).abs() > 1e-4
                || (gpu.color[1] - wall_color[1]).abs() > 1e-4
                || (gpu.color[2] - wall_color[2]).abs() > 1e-4
            {
                wrong_color += 1;
            }
        }
    }
    eprintln!(
        "seam grid {grid}×{grid}: misses={misses} (on-boundary {boundary_misses}) wrong_color={wrong_color}"
    );
    assert_eq!(misses, 0, "rays hit NOTHING at {misses} points on a continuous solid wall — brick seam(s)!");
    assert_eq!(wrong_color, 0, "{wrong_color} rays hit the wrong colour (should all be the wall colour)");

    // Belt-and-braces: a handful of rays aimed EXACTLY at the internal brick-boundary planes (the worst case
    // for a seam) must hit. These coordinates land precisely on x = S / x = 2S / y = S / y = 2S.
    for &(bx, by) in &[(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (2.0, 2.0), (1.5, 1.0), (1.0, 1.5)] {
        let ro = Vec3::new(bx * s, by * s, -1.0);
        let gpu = run_ray(ro, Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(
            gpu.hit, 1,
            "a ray aimed EXACTLY at a brick-boundary plane ({:.3},{:.3}) must still hit the wall",
            bx * s,
            by * s
        );
    }

    // A ray skimming ALONG the wall surface across X (parallel to the face, crossing every X brick boundary)
    // at a depth one voxel inside the wall must stay hit the whole way — no gaps where bricks abut.
    for k in 0..=20 {
        let x0 = 0.05 + k as f32 * (span_x - 0.1) / 20.0;
        let ro = Vec3::new(x0, span_y * 0.5, s * 0.5); // start inside the cube
        let gpu = run_ray(ro, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(gpu.hit, 1, "a ray inside the solid cube must hit (skim x0={x0:.3})");
    }

    let _ = (&aabb_buf, &meta_buf, &voxel_buf, &palette_buf, &blas, &tlas);
}
