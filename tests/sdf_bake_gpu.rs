//! Real-GPU execution of the brick-bake compute shader (`sdf_brick_bake.wgsl`).
//!
//! Runs the ACTUAL bake shader on a headless wgpu device for a known scene (one sphere),
//! reads back the distance output buffer, decodes the packed R16-snorm texels, and checks
//! the baked SDF matches the analytic sphere distance. This verifies the whole GPU bake
//! contract end-to-end — dispatch shape, 2D job indexing, snorm packing, row padding, buffer
//! layout — without a window or the render graph. A bug here would show as wrong/empty
//! geometry in the live renderer.

use std::borrow::Cow;

use bevy::math::{IVec3, Vec3};
use naga_oil::compose::{Composer, NagaModuleDescriptor};

use adventure::sdf_render::atlas::{BRICK_EDGE, dist_band_world};
use adventure::sdf_render::edits::{
    GPU_OP_UNION, GPU_PRIM_SPHERE, GpuEdit, ResolvedEdit, SdfOp, SdfPrimitive, to_gpu_edit,
};
use adventure::sdf_render::SdfGridConfig;

mod common;

// The bake compute path needs no special features.
fn device_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
    common::headless_device(wgpu::Features::empty())
}

fn compose_bake() -> naga::Module {
    // The bake shader is self-contained (no sdf::* imports).
    let src = std::fs::read_to_string("assets/shaders/sdf_brick_bake.wgsl")
        .expect("read sdf_brick_bake.wgsl");
    Composer::default()
        .make_naga_module(NagaModuleDescriptor {
            source: &src,
            file_path: "sdf_brick_bake.wgsl",
            ..Default::default()
        })
        .expect("compose bake shader")
}

const DIST_ROW_U32: u32 = 64;
const DIST_TILE_U32: u32 = DIST_ROW_U32 * 8;
const MAT_TILE_U32: u32 = 128 * 8;

// Mirror of bake_scheduler::GpuJobHeader upload order (48 bytes, 12 u32).
fn header_bytes(coord: IVec3, voxel_size: f32, dist_band: f32, edit_count: u32) -> Vec<u8> {
    let mut b = Vec::with_capacity(48);
    b.extend_from_slice(&coord.x.to_le_bytes());
    b.extend_from_slice(&coord.y.to_le_bytes());
    b.extend_from_slice(&coord.z.to_le_bytes());
    b.extend_from_slice(&voxel_size.to_le_bytes());
    b.extend_from_slice(&dist_band.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes()); // edit_start
    b.extend_from_slice(&edit_count.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes()); // pal01
    b.extend_from_slice(&0u32.to_le_bytes()); // pal23
    b.extend_from_slice(&0u32.to_le_bytes()); // pad
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b
}

fn edit_bytes(e: &GpuEdit) -> Vec<u8> {
    let mut b = Vec::with_capacity(96);
    for col in e.inv_model.to_cols_array() {
        b.extend_from_slice(&col.to_le_bytes());
    }
    for v in [e.params.x, e.params.y, e.params.z, e.params.w] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for v in [e.params2.x, e.params2.y, e.params2.z, e.params2.w] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(&e.tag.to_le_bytes());
    b.extend_from_slice(&e.op_kind.to_le_bytes());
    b.extend_from_slice(&e.smoothing.to_le_bytes());
    b.extend_from_slice(&e.material_id.to_le_bytes());
    b
}

/// Tile-local pixel for voxel (x,y,z): u = y*EDGE + x in [0,64), v = z in [0,8). The dist
/// buffer packs two adjacent-x R16 per u32; row stride is DIST_ROW_U32 (padded).
fn decode_dist(dist_u32: &[u32], x: u32, y: u32, z: u32, band: f32) -> f32 {
    let u = y * 8 + x; // 0..63
    let word = z * DIST_ROW_U32 + u / 2; // u/2 packs the x-pair
    let packed = dist_u32[word as usize];
    let half = if u.is_multiple_of(2) { packed & 0xffff } else { packed >> 16 };
    let snorm = half as u16 as i16; // reinterpret low 16 bits as i16
    (snorm as f32 / 32767.0) * band
}

#[test]
fn gpu_bake_sphere_matches_analytic_distance() {
    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter — skipping");
        return;
    };
    use wgpu::util::DeviceExt;

    let cfg = SdfGridConfig::default();
    let lod = 0u32;
    let voxel_size = cfg.voxel_size_at(lod);
    let band = dist_band_world(&cfg, lod);

    // One sphere at the origin. Bake the brick whose origin is the stride-aligned coord at
    // world 0 — it straddles the sphere so its voxels span inside/surface/outside.
    let radius = 0.3f32;
    let edit = ResolvedEdit::new(
        SdfPrimitive::Sphere { radius },
        bevy::prelude::Transform::IDENTITY,
        SdfOp::default(),
        0,
    );
    let gpu_edit = to_gpu_edit(&edit);
    assert_eq!(gpu_edit.tag, GPU_PRIM_SPHERE);
    assert_eq!(gpu_edit.op_kind, GPU_OP_UNION);

    let coord = cfg.world_to_brick_lod(Vec3::ZERO, lod);

    // --- GPU resources -----------------------------------------------------------------
    let module = compose_bake();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bake"),
        source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
    });

    let headers = header_bytes(coord, voxel_size, band, 1);
    let header_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("headers"),
        contents: &headers,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let edits = edit_bytes(&gpu_edit);
    let edit_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("edits"),
        contents: &edits,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dist_size = (DIST_TILE_U32 * 4) as u64;
    let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dist_out"),
        size: dist_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mat_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mat_out"),
        size: (MAT_TILE_U32 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bake_bgl"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            storage_entry(3, false),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bake_pl"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bake"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bake_bg"),
        layout: &layout,
        entries: &[
            bind(0, &header_buf),
            bind(1, &edit_buf),
            bind(2, &dist_buf),
            bind(3, &mat_buf),
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1); // one job
    }
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: dist_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&dist_buf, 0, &readback, 0, dist_size);
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let data = slice.get_mapped_range();
    let dist_u32: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback.unmap();

    // --- Verify: every voxel's decoded distance ≈ analytic sphere SDF (clamped to band) ---
    let slack = band / 32767.0 * 4.0; // a few snorm LSBs
    let mut max_err = 0.0f32;
    let mut any_inside = false;
    let mut any_outside = false;
    for z in 0..BRICK_EDGE as u32 {
        for y in 0..BRICK_EDGE as u32 {
            for x in 0..BRICK_EDGE as u32 {
                let world = Vec3::new(
                    (coord.x + x as i32) as f32,
                    (coord.y + y as i32) as f32,
                    (coord.z + z as i32) as f32,
                ) * voxel_size;
                let analytic = (world.length() - radius).clamp(-band, band);
                let baked = decode_dist(&dist_u32, x, y, z, band);
                let err = (baked - analytic).abs();
                max_err = max_err.max(err);
                if analytic < 0.0 { any_inside = true; }
                if analytic > 0.0 { any_outside = true; }
                assert!(
                    err <= slack,
                    "voxel ({x},{y},{z}) world={world:?}: baked={baked} analytic={analytic} err={err} > slack={slack}"
                );
            }
        }
    }
    assert!(any_inside && any_outside, "brick must straddle the sphere surface (in {any_inside}, out {any_outside})");
    eprintln!("GPU bake OK: max_err={max_err} (slack={slack}, band={band})");
}

/// Bake a sphere brick, then `copy_buffer_to_texture` it into an R16Snorm atlas texture at a
/// non-trivial tile origin (exactly as the live `SdfBrickBakeNode` does), read the TEXTURE
/// back, and check the decoded distances still match the analytic sphere. This covers the
/// copy stage the buffer-only test skips: the 256-byte-padded `bytes_per_row` into the snorm
/// texture and the `tile_origin` placement. A bug here = geometry written to the wrong place
/// or corrupted on copy → the live "chunk goes missing" hole.
#[test]
fn gpu_bake_copy_to_atlas_texture_roundtrips() {
    use wgpu::util::DeviceExt;

    // R16Snorm (the atlas format) needs TEXTURE_FORMAT_16BIT_NORM; skip rather than fail if absent.
    let Some((device, queue)) = common::headless_device(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
    else {
        eprintln!("adapter lacks TEXTURE_FORMAT_16BIT_NORM — skipping texture roundtrip");
        return;
    };

    let cfg = SdfGridConfig::default();
    let lod = 0u32;
    let voxel_size = cfg.voxel_size_at(lod);
    let band = dist_band_world(&cfg, lod);
    let radius = 0.3f32;
    let edit = ResolvedEdit::new(
        SdfPrimitive::Sphere { radius },
        bevy::prelude::Transform::IDENTITY,
        SdfOp::default(),
        0,
    );
    let gpu_edit = to_gpu_edit(&edit);
    let coord = cfg.world_to_brick_lod(Vec3::ZERO, lod);

    // Bake into the dist buffer (same as the first test).
    let module = compose_bake();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bake"),
        source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
    });
    let header_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("headers"),
        contents: &header_bytes(coord, voxel_size, band, 1),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let edit_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("edits"),
        contents: &edit_bytes(&gpu_edit),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dist_out"),
        size: (DIST_TILE_U32 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mat_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mat_out"),
        size: (MAT_TILE_U32 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            storage_entry(3, false),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            bind(0, &header_buf),
            bind(1, &edit_buf),
            bind(2, &dist_buf),
            bind(3, &mat_buf),
        ],
    });

    // A small multi-tile atlas at a non-zero tile so col/row offsets are exercised. (The
    // live atlas is 256 tiles/row × 64px = 16384 wide, but that exceeds the default 8192
    // texture-dim limit on some adapters; the copy math only depends on the tile origin, so a
    // narrower test atlas exercises it identically.)
    let edge = BRICK_EDGE as u32; // 8
    let tile_w = edge * edge; // 64
    let test_tiles_per_row = 64u32; // 64*64 = 4096 wide, within the 8192 limit
    let atlas_w = test_tiles_per_row * tile_w;
    let atlas_h = edge * 4; // 4 tile rows
    let tile_index = test_tiles_per_row + 1; // row 1, col 1 → non-trivial col_px AND row_px
    let col_px = (tile_index % test_tiles_per_row) * tile_w;
    let row_px = (tile_index / test_tiles_per_row) * edge;

    let atlas = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("atlas_r16snorm"),
        size: wgpu::Extent3d { width: atlas_w, height: atlas_h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R16Snorm,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    // copy_buffer_to_texture — EXACTLY as SdfBrickBakeNode does (padded bytes_per_row).
    enc.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: &dist_buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(DIST_ROW_U32 * 4), // 256
                rows_per_image: Some(edge),
            },
        },
        wgpu::TexelCopyTextureInfo {
            texture: &atlas,
            mip_level: 0,
            origin: wgpu::Origin3d { x: col_px, y: row_px, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d { width: tile_w, height: edge, depth_or_array_layers: 1 },
    );

    // Read the tile sub-rect back out of the texture. R16Snorm = 2 bytes/texel; pad the
    // readback row to 256 bytes (copy alignment).
    let rb_row_bytes = (tile_w * 2).div_ceil(256) * 256; // 256 (64*2=128 → padded to 256)
    let rb_size = (rb_row_bytes * edge) as u64;
    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tex_readback"),
        size: rb_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &atlas,
            mip_level: 0,
            origin: wgpu::Origin3d { x: col_px, y: row_px, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &rb,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(rb_row_bytes),
                rows_per_image: Some(edge),
            },
        },
        wgpu::Extent3d { width: tile_w, height: edge, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);

    let slice = rb.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let data = slice.get_mapped_range().to_vec();

    // Decode: texture tile is 64px wide × 8 tall; pixel (u=y*8+x, v=z) holds the R16 snorm.
    let slack = band / 32767.0 * 4.0;
    let mut max_err = 0.0f32;
    for z in 0..edge {
        let row = &data[(z * rb_row_bytes) as usize..((z * rb_row_bytes) + tile_w * 2) as usize];
        let texels: &[i16] = bytemuck::cast_slice(row);
        for y in 0..edge {
            for x in 0..edge {
                let u = (y * edge + x) as usize; // 0..63
                let baked = (texels[u] as f32 / 32767.0) * band;
                let world = Vec3::new(
                    (coord.x + x as i32) as f32,
                    (coord.y + y as i32) as f32,
                    (coord.z + z as i32) as f32,
                ) * voxel_size;
                let analytic = (world.length() - radius).clamp(-band, band);
                let err = (baked - analytic).abs();
                max_err = max_err.max(err);
                assert!(
                    err <= slack,
                    "TEX voxel ({x},{y},{z}): baked={baked} analytic={analytic} err={err} > slack={slack}"
                );
            }
        }
    }
    eprintln!("GPU bake→texture roundtrip OK: max_err={max_err}");
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}
