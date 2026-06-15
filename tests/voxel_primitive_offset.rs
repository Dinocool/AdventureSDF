//! **A3 Stage 2 — pin the wgpu-trunk fork's `primitive_offset` convention.**
//!
//! Per-chunk BLAS (Stage 3) builds one BLAS per chunk over a SLICE of the single shared `aabb_buf` via
//! `BlasAabbGeometry::primitive_offset` (a BYTE offset into the buffer). The open question the A1/A3 docs
//! flagged: does the `primitive_index` the ray query reports for a hit on such a sliced geometry come back
//! **RELATIVE to the geometry** (0-based within the chunk's slice) or **ABSOLUTE** (the index into the whole
//! buffer)? The shader resolves the global brick as `metas[descriptor.meta_base + primitive_index]`, so:
//!   - RELATIVE ⇒ `meta_base` must be the chunk's slot base (the shader adds it). ← what the design assumes.
//!   - ABSOLUTE ⇒ `meta_base` must be 0 and the slice offset is already folded into `primitive_index`.
//!
//! This test PINS the answer empirically on this exact fork/GPU, so Stage 3's world split can't silently
//! mis-index. It builds TWO solid bricks at distinct world X positions in ONE `aabb_buf`, then builds a BLAS
//! whose single AABB GEOMETRY reads ONLY the SECOND brick (`primitive_offset = 1·stride`, `primitive_count =
//! 1`). A ray fired straight down into the second brick MUST hit it; the reported `prim` (the WGSL
//! `primitive_index`) is then 0 (relative) or 1 (absolute). The shader resolves `metas[meta_base + prim]`, so
//! the test sets `meta_base` to make `meta_base + prim == 1` (the global slot of the brick the BLAS holds)
//! and asserts the recovered block id is brick 1's — i.e. it proves the convention by REQUIRING the correct
//! `meta_base` for the buffer-slice BLAS to resolve the right brick.
//!
//! Skips cleanly (no failure) without an `EXPERIMENTAL_RAY_QUERY` Vulkan adapter.

use std::iter;
use std::mem;

use bevy::math::{IVec3, Vec3};
use wgpu::util::DeviceExt;

use adventure::voxel::brickmap::{BRICK_EDGE, BRICK_WORLD_SIZE, Brick};
use adventure::voxel::gpu::{GpuBrickAabb, GpuInstanceDescriptor, ResidentBrick, pack_resident_set};
use adventure::voxel::palette::{BlockId, BlockRegistry};
use adventure::sdf_render::worldgen::biome::{
    BiomeDef, BiomeId, BiomeLibrary, StrataLayer, TerrainMatId, TerrainSurfaceMaterial,
};

mod common;

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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RayUniform {
    origin: [f32; 3],
    t_min: f32,
    dir: [f32; 3],
    t_max: f32,
}

/// Two distinct solid blocks so a hit's block id identifies WHICH brick was hit.
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
        mat("a", [0.8, 0.1, 0.1, 1.0]),
        mat("b", [0.1, 0.1, 0.8, 1.0]),
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

#[test]
fn primitive_offset_makes_primitive_index_geometry_relative() {
    let Some((device, queue)) = common::headless_ray_query_device() else {
        eprintln!("no ray-query device — skipping primitive_offset convention test");
        return;
    };

    let reg = BlockRegistry::from_biome_library(&test_library());
    let block_a = BlockId(1);
    let block_b = BlockId(2);

    // Two solid LOD0 bricks: brick 0 (block A) at coord (0,0,0); brick 1 (block B) at coord (4,0,0) (well
    // separated in X). `pack_resident_set` lays them out in the SSOT `(lod,z,y,x)` order, so coord (0,..) is
    // slot 0 and coord (4,..) is slot 1 — giving aabbs[0]=brick A, aabbs[1]=brick B.
    let brick_a = Brick::uniform(block_a);
    let brick_b = Brick::uniform(block_b);
    let entries = vec![
        ResidentBrick { coord: IVec3::new(0, 0, 0), brick: &brick_a, lod: 0 },
        ResidentBrick { coord: IVec3::new(4, 0, 0), brick: &brick_b, lod: 0 },
    ];
    let patch = pack_resident_set(&entries, &reg);
    assert_eq!(patch.brick_count(), 2, "two bricks packed");
    // Confirm the slot order: slot 1's meta must be brick B (coord (4,0,0) → voxel_origin (32,0,0)). The bricks
    // pack in `(lod,z,y,x)` order, so coord (0,..) = slot 0 and coord (4,..) = slot 1. (Both bricks are DENSE
    // here, not uniform — their halo neighbours are air, so the full haloed grid is not one block — which is
    // exactly why a hit recovers the CORE block id via the DDA, the thing the test asserts below.)
    assert_eq!(
        patch.metas[1].voxel_origin,
        [4 * BRICK_EDGE, 0, 0],
        "slot 1 must be brick B (coord (4,0,0) ⇒ voxel_origin (32,0,0))"
    );

    // --- Upload the WHOLE 2-brick patch (metas/voxels/palette are global; the BLAS reads only a SLICE). ---
    let aabb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("po_aabbs"),
        contents: bytemuck::cast_slice(&patch.aabbs),
        usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE,
    });
    let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("po_metas"),
        contents: bytemuck::cast_slice(&patch.metas),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let voxel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("po_voxels"),
        contents: bytemuck::cast_slice(&patch.voxels),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let palette_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("po_palette"),
        contents: bytemuck::cast_slice(&patch.palette),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let brick_palettes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("po_brick_palettes"),
        contents: bytemuck::cast_slice(&patch.brick_palettes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // The descriptor 0 carries `meta_base = 1` — the slot base of the slice the BLAS holds (brick B). The
    // shader resolves `metas[meta_base + primitive_index]`. If the fork reports primitive_index RELATIVE to
    // the sliced geometry, it is 0, so `meta_base + 0 == 1` lands on brick B (correct). If it were ABSOLUTE
    // (== 1), `meta_base + 1 == 2` would be out of bounds / wrong — so a correct hit on brick B PROVES the
    // relative convention with this `meta_base`.
    let descriptors = [GpuInstanceDescriptor::world_identity(1)];
    let descriptors_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("po_descriptors"),
        contents: bytemuck::cast_slice(&descriptors),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // --- BLAS over ONLY the SECOND brick's AABB (a slice of the shared buffer via primitive_offset). ---
    let stride = mem::size_of::<GpuBrickAabb>() as wgpu::BufferAddress; // 32 (multiple of 8 ✓)
    let size_desc = wgpu::BlasAABBGeometrySizeDescriptor {
        primitive_count: 1, // ONLY brick B
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("po_blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::AABBs { descriptors: vec![size_desc.clone()] },
    );
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("po_tlas"),
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        max_instances: 1,
    });
    // custom_index = 0 → descriptor 0 (meta_base = 1).
    tlas[0] = Some(wgpu::TlasInstance::new(
        &blas,
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        0,
        0xff,
    ));

    let ray_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("po_ray"),
        size: mem::size_of::<RayUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("po_hit"),
        size: mem::size_of::<GpuHit>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("po_read"),
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
        label: Some("po_lighting"),
        contents: bytemuck::bytes_of(&adventure::voxel::raytrace::LightingUniformData::default()),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let sky_buf = common::sky_uniform_buffer(&device);
    let light_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("po_lighting_bg"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: sky_buf.as_entire_binding() },
        ],
    });

    // Build the BLAS reading ONLY brick B's AABB via the byte offset `1·stride` into the shared buffer.
    let mut build = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("po_build") });
    build.build_acceleration_structures(
        iter::once(&wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::AabbGeometries(vec![wgpu::BlasAabbGeometry {
                size: &size_desc,
                stride,
                aabb_buffer: &aabb_buf,
                primitive_offset: stride as u32, // SKIP brick 0; the geometry begins at brick 1's AABB
            }]),
        }),
        iter::once(&tlas),
    );
    queue.submit(Some(build.finish()));

    // Fire straight down into brick B's world column (coord (4,0,0) → world X [6.4, 8.0)), from above its top.
    let bx = 4.0 * BRICK_WORLD_SIZE + BRICK_WORLD_SIZE * 0.5; // centre of brick B in X
    let bz = BRICK_WORLD_SIZE * 0.5;
    let ro = Vec3::new(bx, BRICK_WORLD_SIZE + 2.0, bz);
    let rd = Vec3::new(0.0, -1.0, 0.0);
    let ray = RayUniform { origin: ro.into(), t_min: 0.0, dir: rd.into(), t_max: 100.0 };
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

    // `gpu.prim` is the shader's RESOLVED GLOBAL index `r.prim = meta_base + primitive_index`. With
    // `meta_base = 1`: a GEOMETRY-RELATIVE primitive_index (0) ⇒ `gpu.prim == 1` and the re-walk reads
    // `metas[1]` = brick B; an ABSOLUTE primitive_index (1) ⇒ `gpu.prim == 2` and the re-walk reads
    // `metas[2]` (out of bounds) ⇒ NOT brick B. So `gpu.prim == 1 && block_id == B` pins RELATIVE; the
    // raw primitive_index the fork delivered is `gpu.prim - meta_base`.
    let raw_primitive_index = gpu.prim.wrapping_sub(descriptors[0].meta_base);
    eprintln!(
        "[primitive_offset] hit={} block_id={} resolved_prim(meta_base+primitive_index)={} \
         raw_primitive_index={} t={:.3}  (meta_base={}, block_a={}, block_b={})",
        gpu.hit, gpu.block_id, gpu.prim, raw_primitive_index, gpu.t, descriptors[0].meta_base, block_a.0, block_b.0
    );

    // THE PIN.
    assert_eq!(gpu.hit, 1, "the ray must hit brick B (the only geometry in the sliced BLAS)");
    assert_eq!(
        raw_primitive_index, 0,
        "FORK CONVENTION: primitive_index for a primitive_offset-sliced AABB geometry is GEOMETRY-RELATIVE \
         (0-based within the geometry), NOT absolute. Stage 3 sets the chunk descriptor's meta_base to the \
         chunk's slot base and the shader adds primitive_index. (Got raw primitive_index={}, expected 0.)",
        raw_primitive_index
    );
    assert_eq!(
        gpu.block_id, block_b.0 as u32,
        "with meta_base=1 + relative primitive_index=0, metas[1] resolves to brick B (block {}) — confirming \
         primitive_offset shifts the DATA read, not the reported primitive_index",
        block_b.0
    );

    let _ = (&aabb_buf, &meta_buf, &voxel_buf, &palette_buf, &blas, &tlas);
}
