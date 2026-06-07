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
// Material tile: Rgba16Snorm, 2 u32/texel, 128 u32/row (= 512 bytes) × 8 rows. Mirrors the bake
// shader's `MAT_TILE_U32` and `render::bake::BAKE_MAT_TILE_U32`.
const MAT_ROW_U32: u32 = 128;
const MAT_TILE_U32: u32 = MAT_ROW_U32 * 8;
// Gradient tile: Rgba8Snorm, 1 u32/texel, 64 u32/row x 8 = 512. Matches the bake shader.
const GRAD_TILE_U32: u32 = 64 * 8;

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

/// CPU mirror of the bake shader's curvature compensation (`sdf_brick_bake.wgsl`
/// `curvature_correct` + the `|f| < 2.5·voxel` near-surface gate): near the surface the stored
/// field is the analytic SDF pre-biased by −(h²/8)·∇²f (with a reliability taper) so trilinear
/// reconstruction doesn't erode convex features at coarse LOD. The reference recomputes the SAME
/// 6-tap Laplacian stencil and the SAME smoothstep taper so it stays within a few snorm LSBs of
/// the GPU. `sdf` is the UNCLAMPED analytic field; the result is band-clamped to match the encode.
fn corrected_ref(sdf: impl Fn(Vec3) -> f32, world: Vec3, voxel_size: f32, band: f32) -> f32 {
    let f = sdf(world);
    let e = voxel_size;
    let corrected = if f.abs() < 2.5 * e {
        let s = sdf(world + Vec3::X * e) + sdf(world - Vec3::X * e)
            + sdf(world + Vec3::Y * e) + sdf(world - Vec3::Y * e)
            + sdf(world + Vec3::Z * e) + sdf(world - Vec3::Z * e);
        let corr = (s - 6.0 * f) / 8.0;
        // smoothstep(0.15, 0.4, |corr|/e): full when small/reliable, →0 when large/unreliable.
        let t = (((corr.abs() / e) - 0.15) / (0.4 - 0.15)).clamp(0.0, 1.0);
        let reliab = 1.0 - t * t * (3.0 - 2.0 * t);
        f - corr * reliab
    } else {
        f
    };
    corrected.clamp(-band, band)
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
    let grad_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grad_out"),
        size: (GRAD_TILE_U32 * 4) as u64,
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
            storage_entry(4, false),
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
            bind(4, &grad_buf),
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

    // --- Verify: every voxel's decoded distance ≈ curvature-compensated sphere SDF ---
    // The bake stores f − (h²/8)∇²f (reliability-tapered) near the surface to un-shrink, so the
    // reference is the analytic sphere run through the SAME correction, not the raw distance.
    let slack = band / 32767.0 * 8.0; // a few snorm LSBs (+ the correction's extra arithmetic)
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
                let reference = corrected_ref(|p| p.length() - radius, world, voxel_size, band);
                let baked = decode_dist(&dist_u32, x, y, z, band);
                let err = (baked - reference).abs();
                max_err = max_err.max(err);
                if reference < 0.0 { any_inside = true; }
                if reference > 0.0 { any_outside = true; }
                assert!(
                    err <= slack,
                    "voxel ({x},{y},{z}) world={world:?}: baked={baked} reference={reference} err={err} > slack={slack}"
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
    let grad_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grad_out"),
        size: (GRAD_TILE_U32 * 4) as u64,
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
            storage_entry(4, false),
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
            bind(4, &grad_buf),
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
    let slack = band / 32767.0 * 8.0;
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
                let reference = corrected_ref(|p| p.length() - radius, world, voxel_size, band);
                let err = (baked - reference).abs();
                max_err = max_err.max(err);
                assert!(
                    err <= slack,
                    "TEX voxel ({x},{y},{z}): baked={baked} reference={reference} err={err} > slack={slack}"
                );
            }
        }
    }
    eprintln!("GPU bake→texture roundtrip OK: max_err={max_err}");
}

/// As [`header_bytes`] but with an explicit material palette (pal01/pal23), so the bake's
/// per-palette-slot material eval can be driven with more than one material.
fn header_bytes_pal(
    coord: IVec3,
    voxel_size: f32,
    dist_band: f32,
    edit_count: u32,
    pal: [u16; 4],
) -> Vec<u8> {
    let mut b = Vec::with_capacity(48);
    b.extend_from_slice(&coord.x.to_le_bytes());
    b.extend_from_slice(&coord.y.to_le_bytes());
    b.extend_from_slice(&coord.z.to_le_bytes());
    b.extend_from_slice(&voxel_size.to_le_bytes());
    b.extend_from_slice(&dist_band.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes()); // edit_start
    b.extend_from_slice(&edit_count.to_le_bytes());
    b.extend_from_slice(&((pal[0] as u32) | ((pal[1] as u32) << 16)).to_le_bytes()); // pal01
    b.extend_from_slice(&((pal[2] as u32) | ((pal[3] as u32) << 16)).to_le_bytes()); // pal23
    b.extend_from_slice(&0u32.to_le_bytes()); // pad
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b
}

/// Bake a TWO-material brick, copy the material buffer into an `Rgba16Snorm` atlas texture exactly
/// as the live bake node does, read the texture back, and assert the per-voxel argmin material id
/// matches the analytic nearest of the two spheres. Proves the material snorm pack, the
/// `Rgba16Snorm` 256-byte-aligned-row copy, and the per-palette-slot material path end-to-end.
/// (The single-material sphere tests cover only the distance atlas.)
#[test]
fn gpu_bake_material_atlas_roundtrips() {
    // Rgba16Snorm (the material atlas format) needs TEXTURE_FORMAT_16BIT_NORM; skip if absent.
    let Some((device, queue)) = common::headless_device(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
    else {
        eprintln!("adapter lacks TEXTURE_FORMAT_16BIT_NORM — skipping material roundtrip");
        return;
    };
    use wgpu::util::DeviceExt;

    let cfg = SdfGridConfig::default();
    let lod = 0u32;
    let voxel_size = cfg.voxel_size_at(lod);
    let band = dist_band_world(&cfg, lod);

    // Two overlapping spheres with DISTINCT material ids, offset in ±x so material splits across x.
    let (ca, ra) = (Vec3::new(-0.2, 0.0, 0.0), 0.35f32);
    let (cb, rb) = (Vec3::new(0.2, 0.0, 0.0), 0.35f32);
    let ea = ResolvedEdit::new(
        SdfPrimitive::Sphere { radius: ra },
        bevy::prelude::Transform::from_translation(ca),
        SdfOp::default(),
        1,
    );
    let eb = ResolvedEdit::new(
        SdfPrimitive::Sphere { radius: rb },
        bevy::prelude::Transform::from_translation(cb),
        SdfOp::default(),
        2,
    );
    let pal = [1u16, 2, 0xffff, 0xffff]; // densely filled from slot 0 (build_palette invariant)
    let coord = cfg.world_to_brick_lod(Vec3::ZERO, lod);

    let module = compose_bake();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bake"),
        source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
    });
    let header_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("headers"),
        contents: &header_bytes_pal(coord, voxel_size, band, 2, pal),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mut edit_data = edit_bytes(&to_gpu_edit(&ea));
    edit_data.extend_from_slice(&edit_bytes(&to_gpu_edit(&eb)));
    let edit_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("edits"),
        contents: &edit_data,
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
    let grad_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grad_out"),
        size: (GRAD_TILE_U32 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            storage_entry(3, false),
            storage_entry(4, false),
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
            bind(4, &grad_buf),
        ],
    });

    let edge = BRICK_EDGE as u32; // 8
    let tile_w = edge * edge; // 64
    let mat_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("atlas_rgba16snorm"),
        size: wgpu::Extent3d { width: tile_w, height: edge, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Snorm,
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
    let row_bytes = MAT_ROW_U32 * 4; // 64 texels × 8 B = 512 (aligned)
    enc.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: &mat_buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(edge),
            },
        },
        wgpu::TexelCopyTextureInfo {
            texture: &mat_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d { width: tile_w, height: edge, depth_or_array_layers: 1 },
    );
    let rb_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mat_readback"),
        size: (row_bytes * edge) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &mat_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &rb_buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(edge),
            },
        },
        wgpu::Extent3d { width: tile_w, height: edge, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);

    let slice = rb_buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let data = slice.get_mapped_range().to_vec();

    let mut checked = 0u32;
    for z in 0..edge {
        let row = &data[(z * row_bytes) as usize..((z * row_bytes) + tile_w * 8) as usize];
        let texels: &[i16] = bytemuck::cast_slice(row);
        for y in 0..edge {
            for x in 0..edge {
                let u = (y * edge + x) as usize; // 0..63
                let s = [
                    texels[u * 4] as f32 / 32767.0,
                    texels[u * 4 + 1] as f32 / 32767.0,
                    texels[u * 4 + 2] as f32 / 32767.0,
                    texels[u * 4 + 3] as f32 / 32767.0,
                ];
                let world = Vec3::new(
                    (coord.x + x as i32) as f32,
                    (coord.y + y as i32) as f32,
                    (coord.z + z as i32) as f32,
                ) * voxel_size;
                let da = (world - ca).length() - ra;
                let db = (world - cb).length() - rb;
                // Only assert where the nearest is inside the ±1 material band and the two are
                // clearly separated (> a few snorm16 LSBs), avoiding boundary ties / clamped slots.
                if da.min(db).abs() >= 0.9 || (da - db).abs() <= 4.0 / 32767.0 {
                    continue;
                }
                let mut best = 0usize;
                for k in 1..4 {
                    if s[k] < s[best] {
                        best = k;
                    }
                }
                let gpu_id = pal[best];
                let expected = if da <= db { 1u16 } else { 2u16 };
                assert_eq!(
                    gpu_id, expected,
                    "voxel ({x},{y},{z}) world={world:?}: da={da} db={db} slots={s:?} gpu_id={gpu_id} expected={expected}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 32, "too few clearly-separated voxels checked ({checked}) — geometry degenerate");
    eprintln!("GPU material Rgba16Snorm roundtrip OK: {checked} voxels");
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

/// PERF rig (ignored): GPU brick-bake dispatch time, A/B'ing the 6-tap curvature-correction cost
/// (`correct_and_grad`) against a variant where its near-surface gate never fires. Confirms +
/// quantifies the bake regression headlessly — no editor, no Nsight, no scene.
/// `cargo test --release --test sdf_bake_gpu -- --ignored --nocapture bake_perf`
#[test]
#[ignore = "perf rig — run with --ignored --nocapture; needs a real GPU"]
fn bake_perf_curvature_ab() {
    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter — skipping");
        return;
    };
    use wgpu::util::DeviceExt;

    let cfg = SdfGridConfig::default();
    let lod = 0u32;
    let voxel_size = cfg.voxel_size_at(lod);
    let band = dist_band_world(&cfg, lod);

    // Representative near-surface brick: several overlapping spheres so `edit_count > 1` (the
    // per-voxel `fold_csg` cost the curvature taps multiply). The brick at world 0 straddles them.
    for n_edits in [8u32, 32, 128, 512, 2048] {
    let mut edits = Vec::new();
    for i in 0..n_edits {
        let off = i as f32 * 0.015;
        let e = ResolvedEdit::new(
            SdfPrimitive::Sphere { radius: 0.30 + off },
            bevy::prelude::Transform::from_xyz(off, 0.0, 0.0),
            SdfOp::default(),
            i as u16,
        );
        edits.extend_from_slice(&edit_bytes(&to_gpu_edit(&e)));
    }
    let coord = cfg.world_to_brick_lod(Vec3::ZERO, lod);

    // Replicate the worst-case near-surface brick across many jobs (fills the GPU like a real batch).
    let n_jobs = 4096u32;
    let mut headers = Vec::with_capacity((n_jobs * 48) as usize);
    for _ in 0..n_jobs {
        headers.extend_from_slice(&header_bytes(coord, voxel_size, band, n_edits));
    }

    let real_src = std::fs::read_to_string("assets/shaders/sdf_brick_bake.wgsl").unwrap();
    // Make the curvature gate `abs(f) < gate` never fire → the 6-tap `correct_and_grad` is skipped.
    let nocurv_src = real_src.replace("let gate = 2.5 * h.voxel_size;", "let gate = -1.0;");
    assert_ne!(real_src, nocurv_src, "curvature-gate string must match the shader");

    let run = |src: &str| -> u128 {
        let module = Composer::default()
            .make_naga_module(NagaModuleDescriptor {
                source: src,
                file_path: "sdf_brick_bake.wgsl",
                ..Default::default()
            })
            .expect("compose bake");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bake"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
        });
        let header_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("headers"),
            contents: &headers,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let edit_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("edits"),
            contents: &edits,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out = |n: u32, l: &'static str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(l),
                size: (n * 4 * n_jobs) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let dist_buf = out(DIST_TILE_U32, "dist");
        let mat_buf = out(MAT_TILE_U32, "mat");
        let grad_buf = out(GRAD_TILE_U32, "grad");
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bake"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &layout,
            entries: &[
                bind(0, &header_buf),
                bind(1, &edit_buf),
                bind(2, &dist_buf),
                bind(3, &mat_buf),
                bind(4, &grad_buf),
            ],
        });
        let mut best = u128::MAX;
        for _ in 0..10 {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(n_jobs.min(256), n_jobs.div_ceil(256), 1);
            }
            let t = std::time::Instant::now();
            queue.submit([enc.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).ok();
            best = best.min(t.elapsed().as_micros());
        }
        best
    };

    let real_us = run(&real_src);
    let nocurv_us = run(&nocurv_src);
    println!(
        "bake_perf: {} jobs × {} edits | real={}us nocurv={}us | curvature={}us ({:.2}x slower)",
        n_jobs,
        n_edits,
        real_us,
        nocurv_us,
        real_us.saturating_sub(nocurv_us),
        real_us as f64 / nocurv_us.max(1) as f64,
        );
    }
}
