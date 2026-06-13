//! **Real-GPU correctness oracle for HW-RT voxel DIRECT LIGHTING** (`voxel_raytrace.wgsl`, `trace_one`).
//!
//! Stage 4 increment 1 adds physically-plausible direct lighting (Lambert sun + traced hard shadow +
//! traced AO) on top of the proven `ray_query` DDA core. This rig proves the two load-bearing new pieces in
//! ISOLATION, WITHOUT a GUI:
//!   1. **Face normal**: a primary ray straight DOWN onto a floor's top face must report `N == (0, +1, 0)`;
//!      a ray going +X into a wall's −X face must report `N == (−1, 0, 0)`.
//!   2. **Traced hard shadow**: with the sun pointing straight down, a ground point UNDER a floating roof
//!      voxel must report `shadowed == 1` (its sun ray hits the roof) while an adjacent OPEN ground point
//!      must report `shadowed == 0` (its sun ray escapes to the sky).
//!
//! The shader's `trace_one` entry writes the committed hit's normal AND traces the same sun shadow ray
//! `shade()` uses, so this rig reads back exactly the values the render path lights with — no oracle drift.
//!
//! Skips cleanly (no failure) on a box without an `EXPERIMENTAL_RAY_QUERY` Vulkan adapter.

use std::iter;
use std::mem;

use bevy::math::{IVec3, Vec3};
use wgpu::util::DeviceExt;

use adventure::sdf_render::worldgen::biome::{
    BiomeDef, BiomeId, BiomeLibrary, StrataLayer, TerrainMatId, TerrainSurfaceMaterial,
};
use adventure::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, BRICK_WORLD_SIZE, Brick};
use adventure::voxel::gpu::{ResidentBrick, pack_resident_set};
use adventure::voxel::palette::{BlockId, BlockRegistry};
use adventure::voxel::raytrace::LightingUniformData;

mod common;

// Mirror of the WGSL `Hit` struct (binding 5): hit, block_id, prim, t, vec4 colour, vec3 normal, shadowed,
// then the GI oracle terms (direct/indirect/emissive_out, each a vec3 padded to vec4 in std430).
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

/// A minimal two-material library so packed bricks carry valid block ids / colours.
fn test_library() -> BiomeLibrary {
    let mat = |name: &str, c: [f32; 4]| TerrainSurfaceMaterial {
        name: name.into(),
        base_color: c,
        roughness: 0.9,
        blend: 0.0,
        texture: None,
        tiling: 4.0,
    };
    let materials = vec![mat("floor", [0.4, 0.4, 0.4, 1.0]), mat("wall", [0.6, 0.3, 0.2, 1.0])];
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

/// A fully-solid brick of `id`.
fn solid(id: BlockId) -> Brick {
    Brick::uniform(id)
}

#[test]
fn gpu_lighting_normals_and_shadows() {
    let Some((device, queue)) = common::headless_ray_query_device() else {
        eprintln!("no ray-query device — skipping gpu_lighting_normals_and_shadows");
        return;
    };

    let reg = BlockRegistry::from_biome_library(&test_library());

    // --- Scene (brick coordinates; each brick spans S = BRICK_WORLD_SIZE world metres) ---
    // A flat solid floor of block-1 bricks at brick-y = 0 across a small XZ patch. A single FLOATING "roof"
    // brick of block-2 high above the column (bx=0, bz=0), so the ground point there is in shadow when the
    // sun points straight down, while the neighbouring column (bx=2, bz=0) has open sky. Also a tall block-2
    // WALL brick at bx=3, bz=0 (on top of the floor) whose −X face we hit with a sideways ray to check the
    // X-axis normal.
    let s = BRICK_WORLD_SIZE;
    let floor1 = solid(BlockId(1));
    let roof = solid(BlockId(2));
    let wall = solid(BlockId(2));
    let mut entries: Vec<ResidentBrick> = Vec::new();
    // Floor row bx = 0..=3 at by = 0.
    for bx in 0..=3i32 {
        entries.push(ResidentBrick { coord: IVec3::new(bx, 0, 0), brick: &floor1, lod: 0 });
    }
    // Floating roof 4 bricks up over the bx=0 column.
    entries.push(ResidentBrick { coord: IVec3::new(0, 4, 0), brick: &roof, lod: 0 });
    // Tall wall directly on the floor at bx=3, one brick up (by=1), so a sideways ray hits its −X face.
    entries.push(ResidentBrick { coord: IVec3::new(3, 1, 0), brick: &wall, lod: 0 });

    let patch = pack_resident_set(&entries, &reg);
    assert!(!patch.is_empty());

    // Sun straight DOWN for a clean shadow setup: toward-sun = (0,+1,0). The floor top face normal is
    // (0,+1,0) so N·L = 1 everywhere on the floor — the only variable is the traced shadow.
    let light = LightingUniformData { sun_direction: [0.0, -1.0, 0.0], ..Default::default() };

    let t_max = 1000.0f32;
    let n = patch.brick_count() as u32;

    // --- build GPU scene from the packed patch (same path as the other rigs) ---
    let aabb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lit_aabbs"),
        contents: bytemuck::cast_slice(&patch.aabbs),
        usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE,
    });
    let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lit_metas"),
        contents: bytemuck::cast_slice(&patch.metas),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let voxel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lit_voxels"),
        contents: bytemuck::cast_slice(&patch.voxels),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let palette_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lit_palette"),
        contents: bytemuck::cast_slice(&patch.palette),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let size_desc = wgpu::BlasAABBGeometrySizeDescriptor {
        primitive_count: n,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("lit_blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::AABBs { descriptors: vec![size_desc.clone()] },
    );
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("lit_tlas"),
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
        label: Some("lit_ray"),
        size: mem::size_of::<RayUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lit_hit"),
        size: mem::size_of::<GpuHit>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lit_read"),
        size: mem::size_of::<GpuHit>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let light_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lit_lighting"),
        contents: bytemuck::bytes_of(&light),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let src = std::fs::read_to_string("assets/shaders/voxel_raytrace.wgsl").expect("read shader");
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
            wgpu::BindGroupEntry { binding: 4, resource: ray_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
        ],
    });
    let light_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lit_lighting_bg"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() }],
    });

    let mut build = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("lit_build") });
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

    // Floor top is at world Y = S (one brick tall). The roof sits at by=4 (Y in [4S, 5S]).
    let floor_top = s;

    // --- 1. Normal of the floor top face: ray straight down onto the OPEN column (bx=2). ---
    let lit_ground = Vec3::new(s * 2.5, floor_top + 2.0, s * 0.5);
    let lit = run_ray(lit_ground, Vec3::new(0.0, -1.0, 0.0));
    eprintln!("[lit_ground] hit={} id={} t={:.3} N={:?} shadowed={}", lit.hit, lit.block_id, lit.t, lit.normal, lit.shadowed);
    assert_eq!(lit.hit, 1, "lit-ground ray must hit the floor");
    assert!((lit.normal[1] - 1.0).abs() < 1e-3, "floor top face normal must be +Y, got {:?}", lit.normal);
    assert!(lit.normal[0].abs() < 1e-3 && lit.normal[2].abs() < 1e-3, "floor normal must be axis-aligned +Y, got {:?}", lit.normal);
    // N·L: N=(0,1,0), toward-sun=(0,1,0) ⇒ ndotl=1 (the known face normal aligns with the sun).
    let ndotl_lit = lit.normal[1]; // dot with (0,1,0)
    assert!((ndotl_lit - 1.0).abs() < 1e-3, "N·L on the sun-facing floor must be 1, got {ndotl_lit}");
    // Open column: sun ray escapes ⇒ NOT shadowed.
    assert_eq!(lit.shadowed, 0, "open ground must be unshadowed (sun ray escapes to the sky)");

    // --- 2. Traced hard shadow: ray straight down onto the column UNDER the floating roof (bx=0). ---
    let shadow_ground = Vec3::new(s * 0.5, floor_top + 2.0, s * 0.5);
    let shadowed = run_ray(shadow_ground, Vec3::new(0.0, -1.0, 0.0));
    eprintln!(
        "[shadow_ground] hit={} id={} t={:.3} N={:?} shadowed={}",
        shadowed.hit, shadowed.block_id, shadowed.t, shadowed.normal, shadowed.shadowed
    );
    assert_eq!(shadowed.hit, 1, "shadow-ground ray must hit the floor");
    assert!((shadowed.normal[1] - 1.0).abs() < 1e-3, "floor normal under the roof must still be +Y, got {:?}", shadowed.normal);
    // Same N·L as the lit point — the ONLY difference is the traced shadow.
    assert_eq!(shadowed.shadowed, 1, "ground under the floating roof must be shadowed (sun ray hits the roof)");

    // The pair is the lighting contrast: same albedo, same N·L, but lit vs. shadowed ⇒ different final
    // colour. Make that explicit so the oracle states the lit-vs-shadow invariant outright.
    assert!(
        lit.shadowed == 0 && shadowed.shadowed == 1,
        "lit point must be unshadowed AND roofed point shadowed — that contrast is the whole feature"
    );

    // --- 3. X-axis face normal: ray going +X into the wall's −X face (wall at bx=3, by=1). ---
    // Wall world X spans [3S, 4S]; fire from before it at the wall's mid height (Y in [S, 2S]).
    let wall_probe = Vec3::new(s * 2.0, s * 1.5, s * 0.5);
    let wall_hit = run_ray(wall_probe, Vec3::new(1.0, 0.0, 0.0));
    eprintln!("[wall] hit={} id={} t={:.3} N={:?}", wall_hit.hit, wall_hit.block_id, wall_hit.t, wall_hit.normal);
    assert_eq!(wall_hit.hit, 1, "the +X ray must hit the wall");
    assert!((wall_hit.normal[0] + 1.0).abs() < 1e-3, "wall −X face normal must be (−1,0,0), got {:?}", wall_hit.normal);
    assert!(wall_hit.normal[1].abs() < 1e-3 && wall_hit.normal[2].abs() < 1e-3, "wall normal must be axis-aligned −X, got {:?}", wall_hit.normal);

    // Keep GPU scene objects alive until all rays have run.
    let _ = (&aabb_buf, &meta_buf, &voxel_buf, &palette_buf, &blas, &tlas, BRICK_EDGE, BRICK_VOXELS);
}
