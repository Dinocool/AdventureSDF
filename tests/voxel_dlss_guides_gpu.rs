//! **Headless GPU assert that the DLSS-RR G-buffer GUIDE textures are populated** (Stage 4c).
//!
//! The DLSS Ray Reconstruction path runs the `raymarch_dlss` compute entry of `voxel_raytrace.wgsl`, which —
//! at each primary hit — writes the demodulation/denoise guides DLSS-RR consumes: the full lit colour, the
//! diffuse albedo, the world-space normal (+roughness in alpha), the reverse-Z hit depth, and the screen-space
//! motion vector. bevy_anti_alias's DLSS-RR node then reads those guides; if they are all-zero, DLSS produces
//! garbage. This rig proves the guides are actually filled WITHOUT a GUI and WITHOUT booting DLSS (which needs
//! the NVIDIA runtime): it builds the static Cornell scene's BLAS/TLAS exactly as the renderer does, dispatches
//! `raymarch_dlss` over a small viewport framed on the box, reads the guide textures back, and asserts that a
//! meaningful fraction of pixels carry non-zero albedo + a unit-length normal + a valid (in-range) reverse-Z
//! depth — i.e. the G-buffer the DLSS node will consume is real.
//!
//! Pure wgpu + `ray_query`; runs under `--no-default-features` (the `dlss` feature is NOT needed to exercise
//! the compute guide-writing — only to wire the DLSS *node*). Skips cleanly without a ray-query Vulkan adapter.

use std::iter;
use std::mem;

use bevy::math::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use adventure::voxel::cornell::{build_cornell, interior_center_world, interior_extent_world};
use adventure::voxel::gpu::pack_brickmap;
use adventure::voxel::palette::BlockRegistry;
use adventure::voxel::raytrace::LightingUniformData;

mod common;

const W: u32 = 128;
const H: u32 = 128;

/// Mirror of the WGSL `CameraUniform` (group 1, binding 0): 160 bytes (the trailing `prev_clip_from_world`
/// drives the non-DLSS ReSTIR temporal reprojection; the DLSS path ignores it but the struct/binding is shared).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    world_from_clip: [[f32; 4]; 4],
    cam_pos: [f32; 3],
    t_max: f32,
    viewport: [u32; 2],
    accum_weight: f32,
    _pad: u32,
    prev_clip_from_world: [[f32; 4]; 4],
}

/// Mirror of the WGSL `DlssCamera` (group 1, binding 10): jittered depth clip + un-jittered prev/cur motion
/// clip, 192 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DlssCamera {
    depth_clip_from_world: [[f32; 4]; 4],
    motion_prev: [[f32; 4]; 4],
    motion_cur: [[f32; 4]; 4],
}

/// Read back a full storage texture into a tightly-packed `Vec<f32>` of `channels` per pixel (handles the GPU
/// row padding). `bytes_per_pixel` is the texel size on the GPU.
fn read_texture_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    channels: usize,
    decode: impl Fn(&[u8]) -> Vec<f32>,
    bytes_per_pixel: usize,
) -> Vec<f32> {
    let unpadded = (W as usize) * bytes_per_pixel;
    let padded = bevy::render::renderer::RenderDevice::align_copy_bytes_per_row(unpadded);
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dlss_guide_read"),
        size: (padded * H as usize) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &read_buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    queue.submit(Some(enc.finish()));
    let slice = read_buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
    let data = slice.get_mapped_range().unwrap();
    let mut out = Vec::with_capacity(W as usize * H as usize * channels);
    for y in 0..H as usize {
        let row = &data[y * padded..y * padded + unpadded];
        for x in 0..W as usize {
            out.extend(decode(&row[x * bytes_per_pixel..(x + 1) * bytes_per_pixel]));
        }
    }
    drop(data);
    read_buf.unmap();
    out
}

#[test]
fn dlss_guides_populated_where_voxels_hit() {
    // `raymarch_dlss` writes 6 storage textures in one stage — wgpu's default limit is 4, so request 6+
    // (the renderer's `wgpu_settings()` raises it the same way under `--features dlss`).
    let Some((device, queue)) = common::headless_ray_query_device_with_storage_textures(6) else {
        eprintln!("no ray-query / 6-storage-texture device — skipping dlss_guides_populated_where_voxels_hit");
        return;
    };

    // --- Build the static Cornell scene (the same packer the renderer's Cornell path uses) ---
    let registry = BlockRegistry::cornell();
    let map = build_cornell(&registry);
    let patch = pack_brickmap(&map, &registry);
    assert!(!patch.is_empty(), "the Cornell scene must pack non-empty");
    let n = patch.brick_count() as u32;

    // --- Scene buffers + BLAS/TLAS (identical to prepare_voxel_rt) ---
    let aabb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("aabbs"),
        contents: bytemuck::cast_slice(&patch.aabbs),
        usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE,
    });
    let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("metas"),
        contents: bytemuck::cast_slice(&patch.metas),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let voxel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxels"),
        contents: bytemuck::cast_slice(&patch.voxels),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let palette_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("palette"),
        contents: bytemuck::cast_slice(&patch.palette),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let size_desc = wgpu::BlasAABBGeometrySizeDescriptor {
        primitive_count: n,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::AABBs { descriptors: vec![size_desc.clone()] },
    );
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("tlas"),
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

    // --- Camera: frame the open −Z front of the box, looking +Z (mirrors the Cornell headless rig) ---
    let [cx, cy, cz] = interior_center_world();
    let extent = interior_extent_world();
    let target = Vec3::new(cx, cy + extent * 0.12, cz);
    let cam_pos = Vec3::new(cx + extent * 0.06, cy, cz - extent * 1.15);
    let world_from_view =
        Mat4::look_at_rh(cam_pos, target, Vec3::Y).inverse();
    // A standard reverse-Z infinite perspective (matches Bevy's projection convention used by the raymarch).
    let aspect = W as f32 / H as f32;
    let fov = 0.6f32;
    let clip_from_view = Mat4::perspective_infinite_reverse_rh(fov, aspect, 0.1);
    let world_from_clip = world_from_view * clip_from_view.inverse();
    let clip_from_world = clip_from_view * world_from_view.inverse();

    // --- Guide + intermediate storage textures (STORAGE_BINDING + COPY_SRC for readback) ---
    let make_tex = |label: &str, format: wgpu::TextureFormat| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    };
    let color = make_tex("color", wgpu::TextureFormat::Rgba16Float);
    let diffuse = make_tex("diffuse_albedo", wgpu::TextureFormat::Rgba8Unorm);
    let specular = make_tex("specular_albedo", wgpu::TextureFormat::Rgba8Unorm);
    let normal_rough = make_tex("normal_roughness", wgpu::TextureFormat::Rgba16Float);
    let depth = make_tex("depth", wgpu::TextureFormat::R32Float);
    // Rgba16Float (.xy used) — `rg16float` storage isn't universally supported (matches the renderer).
    let motion = make_tex("motion", wgpu::TextureFormat::Rgba16Float);
    let v = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());

    // --- Uniforms ---
    let cam = CameraUniform {
        world_from_clip: world_from_clip.to_cols_array_2d(),
        cam_pos: cam_pos.into(),
        t_max: 1.0e4,
        viewport: [W, H],
        accum_weight: 1.0,
        _pad: 0,
        // Un-jittered current clip; the DLSS entries this test drives ignore it (they reproject via
        // `dlss_cam.motion_prev`). Set for binding-size parity with the 160-byte WGSL `CameraUniform`.
        prev_clip_from_world: clip_from_world.to_cols_array_2d(),
    };
    let cam_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera"),
        contents: bytemuck::bytes_of(&cam),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let light_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lighting"),
        contents: bytemuck::bytes_of(&LightingUniformData::cornell()),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let dlss_cam = DlssCamera {
        // No camera motion this frame → prev == cur (motion ≈ 0; the assert below only checks finiteness).
        depth_clip_from_world: clip_from_world.to_cols_array_2d(),
        motion_prev: clip_from_world.to_cols_array_2d(),
        motion_cur: clip_from_world.to_cols_array_2d(),
    };
    let dlss_cam_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dlss_cam"),
        contents: bytemuck::bytes_of(&dlss_cam),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // --- Pipeline (raymarch_dlss entry, auto bind-group layout) ---
    let src = std::fs::read_to_string("assets/shaders/voxel_raytrace.wgsl").expect("read shader");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_raytrace"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("raymarch_dlss"),
        layout: None,
        module: &shader,
        entry_point: Some("raymarch_dlss"),
        compilation_options: Default::default(),
        cache: None,
    });
    let sky_buf = common::sky_uniform_buffer(&device);
    let scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::AccelerationStructure(&tlas) },
            wgpu::BindGroupEntry { binding: 1, resource: meta_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: voxel_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: palette_buf.as_entire_binding() },
        ],
    });
    let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("view_bg"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&v(&color)) },
            wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&v(&diffuse)) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&v(&specular)) },
            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&v(&normal_rough)) },
            wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&v(&depth)) },
            wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&v(&motion)) },
            wgpu::BindGroupEntry { binding: 10, resource: dlss_cam_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: sky_buf.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raymarch_dlss"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, Some(&scene_bg), &[]);
        cpass.set_bind_group(1, Some(&view_bg), &[]);
        cpass.dispatch_workgroups(W.div_ceil(8), H.div_ceil(8), 1);
    }
    queue.submit(Some(enc.finish()));

    // --- Read back the guides ---
    let f16 = |b: &[u8]| half::f16::from_le_bytes([b[0], b[1]]).to_f32();
    let diffuse_px = read_texture_f32(
        &device,
        &queue,
        &diffuse,
        4,
        |b| b.iter().map(|&x| x as f32 / 255.0).collect(),
        4,
    );
    let normal_px = read_texture_f32(
        &device,
        &queue,
        &normal_rough,
        4,
        |b| (0..4).map(|i| f16(&b[i * 2..i * 2 + 2])).collect(),
        8,
    );
    let depth_px = read_texture_f32(
        &device,
        &queue,
        &depth,
        1,
        |b| vec![f32::from_le_bytes([b[0], b[1], b[2], b[3]])],
        4,
    );

    // --- Asserts: where voxels are HIT, the guides must be real ---
    let count = (W * H) as usize;
    let mut hit = 0usize; // pixels with a unit-length world normal (= a surface hit)
    let mut albedo_nonzero = 0usize; // hits whose diffuse albedo is non-black
    let mut depth_valid = 0usize; // hits with a finite reverse-Z depth in (0, 1]
    for i in 0..count {
        let n = Vec3::new(normal_px[i * 4], normal_px[i * 4 + 1], normal_px[i * 4 + 2]);
        let len = n.length();
        // A hit writes a unit face normal; a miss writes (0,0,0). Unit-length ⇒ this pixel hit a voxel.
        if (len - 1.0).abs() < 0.05 {
            hit += 1;
            let a = Vec3::new(diffuse_px[i * 4], diffuse_px[i * 4 + 1], diffuse_px[i * 4 + 2]);
            if a.length() > 0.02 {
                albedo_nonzero += 1;
            }
            let d = depth_px[i];
            if d.is_finite() && d > 0.0 && d <= 1.0 {
                depth_valid += 1;
            }
        }
    }
    let hit_frac = hit as f32 / count as f32;
    eprintln!(
        "dlss guides: {W}x{H} px — hit_frac={hit_frac:.3} ({hit} hits), albedo_nonzero={albedo_nonzero}, \
         depth_valid={depth_valid}"
    );

    // The box fills most of the framed view, so a large fraction of pixels are surface hits.
    assert!(
        hit_frac > 0.30,
        "too few surface hits ({:.1}%) — the camera does not frame the Cornell box (guides untested)",
        100.0 * hit_frac
    );
    // EVERY hit must carry a non-zero diffuse albedo guide (the Cornell walls/floor are all coloured).
    assert_eq!(
        albedo_nonzero, hit,
        "{} of {hit} surface-hit pixels have a BLACK diffuse_albedo guide — DLSS would re-modulate to black",
        hit - albedo_nonzero
    );
    // EVERY hit must carry a valid reverse-Z depth (DLSS-RR Hardware depth mode reads this).
    assert_eq!(
        depth_valid, hit,
        "{} of {hit} surface-hit pixels have an invalid (≤0 or non-finite) depth guide",
        hit - depth_valid
    );
}
