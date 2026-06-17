//! Shared headless-GPU bring-up for the integration test rigs. Each rig declares `mod common;`
//! (cargo does NOT compile files under `tests/<dir>/` as their own test binary, so this is a plain
//! shared module). Not every rig uses every helper, hence the crate-level dead-code allow.
#![allow(dead_code)]

use futures_lite::future::block_on;
use wgpu::util::DeviceExt;

/// A3 — a storage buffer holding ONE identity descriptor 0 (the whole test scene = the streamed-world
/// degenerate case: identity transform, meta_base 0, all bases 0). `voxel_raytrace.wgsl`'s hit path now reads
/// the descriptor table at group 0 binding 13, so EVERY rig that builds a group-0 scene bind group must supply
/// it (bind it at `binding: 13`) or pipeline validation fails. With this single identity descriptor the shader
/// is bit-identical-in-effect to the pre-A3 world-space march. Single SSOT for the test descriptor buffer.
pub fn instance_descriptors_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    let descriptors = [adventure::voxel::gpu::GpuInstanceDescriptor::world_identity(0)];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test_instance_descriptors"),
        contents: bytemuck::cast_slice(&descriptors),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

/// A UNIFORM buffer holding the default [`SkyUniformData`] (group 1 binding 11 of `voxel_raytrace.wgsl`).
/// Every entry point that shades or bounces (`trace_one`, `restir_probe`, `raymarch_dlss`, …) now references
/// the `Sky` uniform via `sky_radiance`, so wgpu's auto-derived group(1) layout includes binding 11 — the
/// test's group(1) bind group MUST supply it or pipeline-creation/validation fails. The defaults preserve the
/// previous inline sky look, so behaviour-asserting tests are unaffected. Single SSOT for the test sky buffer.
pub fn sky_uniform_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    let sky = adventure::voxel::raytrace::SkyUniformData::default();
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test_sky"),
        contents: bytemuck::bytes_of(&sky),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

/// **SSOT shader loader for the GPU test rigs.** `voxel_raytrace.wgsl` now contains a `#{WORLD_CACHE_SIZE}`
/// token (the world-cache hash-table size, Phase 2.1) that MUST be substituted before naga parses the source.
/// Every test that loads the shader goes through this (forwarding to the production
/// [`adventure::voxel::raytrace::voxel_raytrace_shader_src`] SSOT) so the token can never reach naga
/// un-substituted. Tests that don't exercise the cache still get a valid (full-size-table) shader.
pub fn voxel_raytrace_shader_src() -> String {
    adventure::voxel::raytrace::voxel_raytrace_shader_src(
        adventure::voxel::raytrace::WORLD_CACHE_SIZE,
    )
}

/// A headless wgpu device requesting `required` features. Returns `None` — the caller then skips —
/// if no adapter is available or the adapter lacks `required` (so a CI box without 16-bit-norm
/// storage skips cleanly instead of failing deep in a texture create). Logs the adapter so CI shows
/// which GPU ran. Pass `wgpu::Features::empty()` when the rig needs no special features.
pub fn headless_device(required: wgpu::Features) -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..Default::default()
    }))
    .ok()?;
    let info = adapter.get_info();
    eprintln!(
        "GPU adapter: name={:?} backend={:?} driver_info={:?} device_type={:?}",
        info.name, info.backend, info.driver_info, info.device_type
    );
    if !adapter.features().contains(required) {
        eprintln!("adapter lacks {required:?} — skipping");
        return None;
    }
    block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("sdf_test_device"),
        required_features: required,
        ..Default::default()
    }))
    .ok()
}

/// A headless wgpu device (no special features) whose compute-workgroup limits are RAISED so a pipeline using
/// `@workgroup_size` larger than wgpu's default 256 can be created — mirroring the renderer's `wgpu_settings()`
/// bump (`max_compute_invocations_per_workgroup` / `max_compute_workgroup_size_x` to 1024). The G-c.1 enumerate
/// pass dispatches one 8³ = 512-thread workgroup per solid WG-cell, so its parity test needs this. Returns
/// `None` (caller skips) if no adapter is present or it can't reach the requested invocation limit.
pub fn headless_device_with_compute_limits(min_invocations: u32) -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..Default::default()
    }))
    .ok()?;
    let info = adapter.get_info();
    eprintln!(
        "GPU adapter: name={:?} backend={:?} device_type={:?}",
        info.name, info.backend, info.device_type
    );
    if adapter.limits().max_compute_invocations_per_workgroup < min_invocations {
        eprintln!(
            "adapter max_compute_invocations_per_workgroup {} < {min_invocations} — skipping",
            adapter.limits().max_compute_invocations_per_workgroup
        );
        return None;
    }
    let mut limits = wgpu::Limits::default();
    limits.max_compute_invocations_per_workgroup =
        limits.max_compute_invocations_per_workgroup.max(min_invocations);
    limits.max_compute_workgroup_size_x = limits.max_compute_workgroup_size_x.max(min_invocations);
    block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("sdf_test_device_compute_limits"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    }))
    .ok()
}

/// A headless wgpu device with `EXPERIMENTAL_RAY_QUERY` enabled (AABB-BLAS `ray_query`), mirroring the
/// device setup in `D:/spike-aabb`: the experimental feature flag, the minimum acceleration-structure
/// limits, and `ExperimentalFeatures::enabled()` (which wgpu-trunk requires at device creation for the
/// ray-query path). Returns `None` — caller skips — if no Vulkan adapter is present or it lacks ray query.
/// **SSOT for a limit-bumped ray-query test device.** Acquires the forced-Vulkan ray-query adapter and
/// requests a device that raises `max_storage_textures_per_shader_stage` to `min_storage_textures` AND
/// `max_storage_buffers_per_shader_stage` to `min_storage_buffers` (both above wgpu's defaults of 4 / 8), plus
/// the 1024-wide compute-workgroup limits the world-cache scan needs. Returns `None` (caller skips) if no
/// ray-query adapter is present or it can't reach the requested limits. The renderer's `wgpu_settings()` makes
/// the same bumps; this mirrors them so a pipeline that compiles in-engine also compiles here.
fn request_ray_query_device(
    min_storage_textures: u32,
    min_storage_buffers: u32,
) -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..Default::default()
    }))
    .ok()?;
    if !adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
        eprintln!("adapter lacks EXPERIMENTAL_RAY_QUERY — skipping");
        return None;
    }
    if adapter.limits().max_storage_textures_per_shader_stage < min_storage_textures {
        eprintln!(
            "adapter max_storage_textures_per_shader_stage {} < {min_storage_textures} — skipping",
            adapter.limits().max_storage_textures_per_shader_stage
        );
        return None;
    }
    if adapter.limits().max_storage_buffers_per_shader_stage < min_storage_buffers {
        eprintln!(
            "adapter max_storage_buffers_per_shader_stage {} < {min_storage_buffers} — skipping",
            adapter.limits().max_storage_buffers_per_shader_stage
        );
        return None;
    }
    let mut limits =
        wgpu::Limits::default().using_minimum_supported_acceleration_structure_values();
    limits.max_storage_textures_per_shader_stage =
        limits.max_storage_textures_per_shader_stage.max(min_storage_textures);
    limits.max_storage_buffers_per_shader_stage =
        limits.max_storage_buffers_per_shader_stage.max(min_storage_buffers);
    // The world-cache decay/compaction passes use `@workgroup_size(1024)` (the prefix-sum scan width). wgpu's
    // default caps invocations-per-workgroup + workgroup_size_x at 256, so raise both to 1024 (desktop RTX
    // supports it; mirrors the renderer's `wgpu_settings()` bump).
    limits.max_compute_invocations_per_workgroup = limits.max_compute_invocations_per_workgroup.max(1024);
    limits.max_compute_workgroup_size_x = limits.max_compute_workgroup_size_x.max(1024);
    block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("voxel_rt_test_device_limits"),
        required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
        required_limits: limits,
        memory_hints: Default::default(),
        trace: wgpu::Trace::Off,
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
    }))
    .ok()
}

/// Like [`headless_ray_query_device`] but raises `max_storage_textures_per_shader_stage` to `min_storage`
/// (wgpu's default is 4) so a G-buffer-style compute that storage-writes more than 4 textures in one stage —
/// e.g. the DLSS-RR `raymarch_dlss` entry (6: colour + diffuse/specular albedo + normal/roughness + depth +
/// motion) — can create its pipeline. Mirrors the renderer's `wgpu_settings()` bump under `--features dlss`.
/// Keeps the default storage-BUFFER limit (8); use [`headless_ray_query_device_with_storage`] if an entry also
/// binds more than 8 storage buffers (e.g. `restir_p1` once it queries the group(3) world cache — Phase 2.2).
pub fn headless_ray_query_device_with_storage_textures(
    min_storage: u32,
) -> Option<(wgpu::Device, wgpu::Queue)> {
    request_ray_query_device(min_storage, 8)
}

/// Like [`headless_ray_query_device`] but raises `max_storage_buffers_per_shader_stage` to `min_storage`
/// (wgpu's default is 8). The Phase-2.1 world-cache compute passes bind 3 scene storage buffers (group 0) +
/// 11 cache storage buffers (group 3) in one pipeline layout = 14 in a single stage, exceeding the default —
/// mirrors the renderer's `wgpu_settings()` bump. Returns `None` (caller skips) if no ray-query adapter is
/// present or it can't reach `min_storage`.
pub fn headless_ray_query_device_with_storage_buffers(
    min_storage: u32,
) -> Option<(wgpu::Device, wgpu::Queue)> {
    request_ray_query_device(4, min_storage)
}

/// A ray-query device that raises BOTH the storage-texture and storage-buffer limits. The Phase-2.2
/// `restir_p1`/`restir_dlss_p1` entries query the group(3) world cache, so their auto-derived layout binds 11
/// storage buffers (3 scene + 4 reservoir/surface + 4 cache) AND, for the DLSS variant, 6 storage textures —
/// both over wgpu's defaults. The screen-space compile gate uses this so it mirrors the in-engine device.
pub fn headless_ray_query_device_with_storage(
    min_storage_textures: u32,
    min_storage_buffers: u32,
) -> Option<(wgpu::Device, wgpu::Queue)> {
    request_ray_query_device(min_storage_textures, min_storage_buffers)
}

pub fn headless_ray_query_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    // Ray query is a Vulkan/DX12 capability; the spike forces VULKAN, so we do too for a stable adapter.
    // Spell out InstanceDescriptor fully (wgpu-trunk `InstanceDescriptor` has no `Default`), forcing the
    // VULKAN backend like `D:/spike-aabb` for a stable ray-query adapter.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..Default::default()
    }))
    .ok()?;
    let info = adapter.get_info();
    eprintln!(
        "ray-query GPU adapter: name={:?} backend={:?} device_type={:?}",
        info.name, info.backend, info.device_type
    );
    if !adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
        eprintln!("adapter lacks EXPERIMENTAL_RAY_QUERY — skipping");
        return None;
    }
    block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("voxel_rt_test_device"),
        required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
        required_limits: wgpu::Limits::default()
            .using_minimum_supported_acceleration_structure_values(),
        memory_hints: Default::default(),
        trace: wgpu::Trace::Off,
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
    }))
    .ok()
}
