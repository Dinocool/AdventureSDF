//! **Headless GPU assert that the DLSS-RR G-buffer GUIDE textures are populated** (Stage 4c).
//!
//! The DLSS Ray Reconstruction path runs the live two-pass ReSTIR DLSS compute entries
//! (`restir_dlss_p1` → `restir_dlss_p2`) of `voxel_raytrace.wgsl`. Pass 2, at each primary hit, writes the
//! demodulation/denoise guides DLSS-RR consumes: the full lit colour, the diffuse albedo, the world-space
//! normal (+roughness in alpha), the reverse-Z hit depth, and the screen-space motion vector. bevy_anti_alias's
//! DLSS-RR node then reads those guides; if they are all-zero, DLSS produces garbage. This rig proves the guides
//! are actually filled WITHOUT a GUI and WITHOUT booting DLSS (which needs the NVIDIA runtime): it builds the
//! static Cornell scene's BLAS/TLAS exactly as the renderer does, dispatches `restir_dlss_p1` then
//! `restir_dlss_p2` over a small viewport framed on the box, reads the guide textures back, and asserts that a
//! meaningful fraction of pixels carry non-zero albedo + a unit-length normal + a valid (in-range) reverse-Z
//! depth — i.e. the G-buffer the DLSS node will consume is real. (The guide writes are taken purely from the
//! primary `trace`, independent of reservoir/cache content, so the world cache is bound but disabled
//! [`use_world_cache = 0`] and the reservoir buffers start zeroed — no world-cache pass loop is needed.)
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

/// A small world-cache hash table for the test (the live path uses 2^20). The cache is BOUND for the ReSTIR
/// DLSS pipeline layout but never queried (`use_world_cache = 0`), so the size only bounds the dummy buffers.
const TEST_WORLD_CACHE_SIZE: u32 = 1 << 12;

/// Bytes per WGSL `Reservoir` (3×vec4) and `PixelSurface` (2×vec4) — the group(2) ReSTIR buffers.
const RESERVOIR_SIZE: u64 = 48;
const SURFACE_SIZE: u64 = 32;

/// Mirror of the WGSL `RestirParams` (group 2, binding 2): reset + frame + viewport + the ReSTIR knobs. 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RestirParams {
    reset: u32,
    frame_index: u32,
    viewport_x: u32,
    viewport_y: u32,
    spatial_samples: u32,
    confidence_weight_cap: f32,
    spatial_radius: f32,
    di_enabled: u32,
    di_confidence_cap: f32,
    di_initial_samples: u32,
    _pad_gi: u32,
    _pad0: u32, // (was gi_half/gi_half_x/gi_half_y — half-res removed)
    _pad1: u32,
    _pad2: u32,
    gi_dissim_cap_dist: f32,
    _pad3: u32,
}

/// Mirror of the WGSL `WorldCacheUniform` (group 3, binding 0): 64 bytes. `use_world_cache = 0` here — the
/// cache is bound for layout completeness but never queried, so its content is irrelevant.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WorldCacheUniform {
    cell_base_size: f32,
    lod_scale: f32,
    gi_ray_distance: f32,
    cell_lifetime: u32,
    max_temporal_samples: f32,
    frame_index: u32,
    reset: u32,
    use_world_cache: u32,
    gi_multibounce: u32,
    view_x: f32,
    view_y: f32,
    view_z: f32,
    max_active_cells_per_frame: u32,
    light_count: u32,
    nee_enabled: u32,
    nee_samples: u32,
}

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
    // `restir_dlss_p2` writes 6 storage textures in one stage (wgpu's default limit is 4) AND the ReSTIR
    // pipeline layout binds 21 storage buffers (5 scene + 4 reservoir/surface + 12 group(3) cache); the
    // renderer's `wgpu_settings()` raises both the same way (it lifts storage buffers to 48 under the RT path).
    let Some((device, queue)) = common::headless_ray_query_device_with_storage(6, 32) else {
        eprintln!("no ray-query / 6-storage-texture+24-buffer device — skipping dlss_guides_populated_where_voxels_hit");
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
    // Storage plan R2b — the per-brick palettes the bit-packed index stream indirects through.
    let brick_palettes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("palette_brick_palettes"),
        contents: bytemuck::cast_slice(&patch.brick_palettes),
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

    // --- The live two-pass ReSTIR DLSS pipelines (`restir_dlss_p1` → `restir_dlss_p2`). EXPLICIT layouts
    // mirror the engine's `dlss_restir_pl`: group 0 = scene, group 1 = DLSS view (camera/out_tex/guides), group
    // 2 = reservoirs, group 3 = the world cache (bound but disabled). A small hash table keeps the dummy cache
    // buffers light. ---
    let src = adventure::voxel::raytrace::voxel_raytrace_shader_src(TEST_WORLD_CACHE_SIZE);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_raytrace"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });

    // Bind-group-layout-entry helpers (COMPUTE visibility).
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
    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let storage_tex = |binding: u32, format: wgpu::TextureFormat| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    };

    let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
                count: None,
            },
            storage(1, true),
            storage(2, true),
            storage(3, true),
            storage(12, true), // R2b per-brick palettes
            storage(13, true), // A3 instance descriptors
        ],
    });
    // group(1): the DLSS view layout (mirror of the engine's `dlss_view_layout`).
    let dlss_view_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("dlss_view_layout"),
        entries: &[
            uniform(0),                                          // camera
            storage_tex(1, wgpu::TextureFormat::Rgba16Float),   // out_tex (colour)
            uniform(2),                                          // lighting
            storage_tex(5, wgpu::TextureFormat::Rgba8Unorm),    // diffuse_albedo
            storage_tex(6, wgpu::TextureFormat::Rgba8Unorm),    // specular_albedo
            storage_tex(7, wgpu::TextureFormat::Rgba16Float),   // normal_roughness
            storage_tex(8, wgpu::TextureFormat::R32Float),      // depth
            storage_tex(9, wgpu::TextureFormat::Rgba16Float),   // motion
            uniform(10),                                         // dlss_cam
            uniform(11),                                         // sky
        ],
    });
    // group(2): the ReSTIR reservoir buffers + params + surfaces + DI reservoirs (5/6) + STBN texture (7),
    // an exact mirror of the engine's `reservoir_layout` (`voxel_rt_reservoir_layout`).
    let stbn_tex_entry = wgpu::BindGroupLayoutEntry {
        binding: 7,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    };
    let reservoir_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("reservoir_layout"),
        entries: &[
            storage(0, false),
            storage(1, false),
            uniform(2),
            storage(3, false),
            storage(4, false),
            storage(5, false), // DI reservoirs a (GI 4.0)
            storage(6, false), // DI reservoirs b
            stbn_tex_entry,    // 7 = spatiotemporal blue noise texture array
        ],
    });
    // group(4): the screen-probe layout (`voxel_rt_probe_layout`). `restir_dlss_p2` statically reaches
    // `screen_probe_integrate`, so the pipeline layout must declare it even though the probes are disabled
    // (`probe_params.enabled = 0`) and never executed here.
    let probe_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("probe_layout"),
        entries: &[uniform(0), storage(1, false), storage(2, false), storage(3, false)],
    });
    // group(3): the world-cache layout the engine binds to the ReSTIR passes (`world_cache_layout`): the wc
    // uniform + the 10 persistent storage buffers + the NEE light list / alias table (read-only).
    let world_cache_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("world_cache_layout"),
        entries: &[
            uniform(0),
            storage(1, false),
            storage(2, false),
            storage(3, false),
            storage(4, false),
            storage(5, false),
            storage(6, false),
            storage(7, false),
            storage(8, false),
            storage(9, false),
            storage(10, false),
            storage(15, true), // NEE light list (unused — use_world_cache 0)
            storage(16, true), // NEE alias table
        ],
    });
    let restir_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("dlss_restir_pl"),
        bind_group_layouts: &[
            Some(&scene_layout),
            Some(&dlss_view_layout),
            Some(&reservoir_layout),
            Some(&world_cache_layout),
            Some(&probe_layout),
        ],
        immediate_size: 0,
    });
    let mk = |entry: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&restir_pl),
            module: &shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let p1 = mk("restir_dlss_p1");
    let p2 = mk("restir_dlss_p2");

    let sky_buf = common::sky_uniform_buffer(&device);
    let descriptors_buf = common::instance_descriptors_buffer(&device); // A3: one identity descriptor 0

    // group(2) reservoir + surface storage buffers (one per pixel), zero-initialised.
    let px_count = (W * H) as u64;
    let zero_storage = |label: &str, bytes: u64| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };
    let res_a = zero_storage("reservoirs_a", px_count * RESERVOIR_SIZE);
    let res_b = zero_storage("reservoirs_b", px_count * RESERVOIR_SIZE);
    let surf_cur = zero_storage("surfaces_cur", px_count * SURFACE_SIZE);
    let surf_prev = zero_storage("surfaces_prev", px_count * SURFACE_SIZE);
    // DI reservoirs (group 2, bindings 5/6) — 16 bytes/pixel, mirror of the engine's `DI_RESERVOIR_SIZE`.
    let di_a = zero_storage("di_reservoirs_a", px_count * 16);
    let di_b = zero_storage("di_reservoirs_b", px_count * 16);
    // STBN (group 2, binding 7): a 1×1×1 D2Array dummy — the shader's `dims > 1` guard falls back to white
    // noise, so the contents are irrelevant; it only needs to be bindable.
    let stbn_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("stbn_dummy"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let stbn_view = stbn_tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    // group(4) screen probes — disabled (`enabled = 0`), so 1-element storage buffers suffice.
    let probe_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe_params"),
        size: 48, // ScreenProbeParamsData = 12×u32
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    let probe_headers = zero_storage("probe_headers", 32);
    let probe_sh = zero_storage("probe_sh", 9 * 16);
    let probe_sh_hist = zero_storage("probe_sh_history", 9 * 16);
    let restir_params = RestirParams {
        reset: 1, // first frame — no temporal history
        frame_index: 0,
        viewport_x: W,
        viewport_y: H,
        spatial_samples: 0, // pass 2 spatial reuse off — guides don't depend on it
        confidence_weight_cap: 8.0,
        spatial_radius: 16.0,
        di_enabled: 0, // DI off — guides come from the GI/primary trace
        di_confidence_cap: 8.0,
        di_initial_samples: 1,
        _pad_gi: 0,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
        gi_dissim_cap_dist: 0.0, // uncapped (pure Solari relative reject)
        _pad3: 0,
    };
    let restir_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("restir_params"),
        contents: bytemuck::bytes_of(&restir_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // group(3): a DISABLED world cache (`use_world_cache = 0`) — bound for layout completeness, never queried.
    let tsz = TEST_WORLD_CACHE_SIZE as u64;
    let wc_uniform = WorldCacheUniform {
        cell_base_size: 0.15,
        lod_scale: 15.0,
        gi_ray_distance: 50.0,
        cell_lifetime: 10,
        max_temporal_samples: 32.0,
        frame_index: 0,
        reset: 1,
        use_world_cache: 0, // the cache is bound but NEVER queried — guides come from the primary trace
        gi_multibounce: 0,
        view_x: cam_pos.x,
        view_y: cam_pos.y,
        view_z: cam_pos.z,
        max_active_cells_per_frame: 0,
        light_count: 0,
        nee_enabled: 0,
        nee_samples: 1,
    };
    let wc_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wc_uniform"),
        contents: bytemuck::bytes_of(&wc_uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    // The 10 persistent cache storage buffers (zeroed) + 1-element NEE dummies (never indexed, light_count 0).
    let wc_checksums = zero_storage("wc_checksums", tsz * 4);
    let wc_life = zero_storage("wc_life", tsz * 4);
    let wc_radiance = zero_storage("wc_radiance", tsz * 16);
    let wc_geometry = zero_storage("wc_geometry", tsz * 32);
    let wc_lum_deltas = zero_storage("wc_lum_deltas", tsz * 4);
    let wc_new_radiance = zero_storage("wc_new_radiance", tsz * 16);
    let wc_a = zero_storage("wc_a", tsz * 4);
    let wc_b = zero_storage("wc_b", 1024 * 4);
    let wc_active_indices = zero_storage("wc_active_indices", tsz * 4);
    let wc_active_count = zero_storage("wc_active_count", 4);
    let nee_lights = zero_storage("nee_lights", 32); // one VoxelLight (vec4+vec4)
    let nee_alias = zero_storage("nee_alias", 8); // one AliasEntry (f32+u32)

    let scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene_bg"),
        layout: &scene_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::AccelerationStructure(&tlas) },
            wgpu::BindGroupEntry { binding: 1, resource: meta_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: voxel_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: palette_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: brick_palettes_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 13, resource: descriptors_buf.as_entire_binding() },
        ],
    });
    let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("view_bg"),
        layout: &dlss_view_layout,
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
    let reservoir_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reservoir_bg"),
        layout: &reservoir_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: res_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: res_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: restir_params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: surf_cur.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: surf_prev.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: di_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: di_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&stbn_view) },
        ],
    });
    let probe_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe_bg"),
        layout: &probe_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: probe_params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: probe_headers.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: probe_sh.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: probe_sh_hist.as_entire_binding() },
        ],
    });
    let cache_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cache_bg"),
        layout: &world_cache_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wc_uniform_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wc_checksums.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: wc_life.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: wc_radiance.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: wc_geometry.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: wc_lum_deltas.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: wc_new_radiance.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: wc_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: wc_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: wc_active_indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: wc_active_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 15, resource: nee_lights.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 16, resource: nee_alias.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("restir_dlss"),
            timestamp_writes: None,
        });
        cpass.set_bind_group(0, Some(&scene_bg), &[]);
        cpass.set_bind_group(1, Some(&view_bg), &[]);
        cpass.set_bind_group(2, Some(&reservoir_bg), &[]);
        cpass.set_bind_group(3, Some(&cache_bg), &[]);
        cpass.set_bind_group(4, Some(&probe_bg), &[]);
        // Pass 1 fills reservoirs_b (irrelevant to the guides), pass 2 re-traces + writes the DLSS guides.
        cpass.set_pipeline(&p1);
        cpass.dispatch_workgroups(W.div_ceil(8), H.div_ceil(8), 1);
        cpass.set_pipeline(&p2);
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
