//! Shared headless-GPU bring-up for the integration test rigs. Each rig declares `mod common;`
//! (cargo does NOT compile files under `tests/<dir>/` as their own test binary, so this is a plain
//! shared module). Not every rig uses every helper, hence the crate-level dead-code allow.
#![allow(dead_code)]

use futures_lite::future::block_on;
use wgpu::util::DeviceExt;

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

/// A headless wgpu device with `EXPERIMENTAL_RAY_QUERY` enabled (AABB-BLAS `ray_query`), mirroring the
/// device setup in `D:/spike-aabb`: the experimental feature flag, the minimum acceleration-structure
/// limits, and `ExperimentalFeatures::enabled()` (which wgpu-trunk requires at device creation for the
/// ray-query path). Returns `None` — caller skips — if no Vulkan adapter is present or it lacks ray query.
/// Like [`headless_ray_query_device`] but raises `max_storage_textures_per_shader_stage` to `min_storage`
/// (wgpu's default is 4) so a G-buffer-style compute that storage-writes more than 4 textures in one stage —
/// e.g. the DLSS-RR `raymarch_dlss` entry (6: colour + diffuse/specular albedo + normal/roughness + depth +
/// motion) — can create its pipeline. Mirrors the renderer's `wgpu_settings()` bump under `--features dlss`.
pub fn headless_ray_query_device_with_storage_textures(
    min_storage: u32,
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
    if adapter.limits().max_storage_textures_per_shader_stage < min_storage {
        eprintln!(
            "adapter max_storage_textures_per_shader_stage {} < {min_storage} — skipping",
            adapter.limits().max_storage_textures_per_shader_stage
        );
        return None;
    }
    let mut limits =
        wgpu::Limits::default().using_minimum_supported_acceleration_structure_values();
    limits.max_storage_textures_per_shader_stage = min_storage;
    block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("voxel_rt_test_device_storage"),
        required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
        required_limits: limits,
        memory_hints: Default::default(),
        trace: wgpu::Trace::Off,
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
    }))
    .ok()
}

/// Like [`headless_ray_query_device`] but raises `max_storage_buffers_per_shader_stage` to `min_storage`
/// (wgpu's default is 8). The Phase-2.1 world-cache compute passes bind 3 scene storage buffers (group 0) +
/// 11 cache storage buffers (group 3) in one pipeline layout = 14 in a single stage, exceeding the default —
/// mirrors the renderer's `wgpu_settings()` bump. Returns `None` (caller skips) if no ray-query adapter is
/// present or it can't reach `min_storage`.
pub fn headless_ray_query_device_with_storage_buffers(
    min_storage: u32,
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
    if adapter.limits().max_storage_buffers_per_shader_stage < min_storage {
        eprintln!(
            "adapter max_storage_buffers_per_shader_stage {} < {min_storage} — skipping",
            adapter.limits().max_storage_buffers_per_shader_stage
        );
        return None;
    }
    let mut limits =
        wgpu::Limits::default().using_minimum_supported_acceleration_structure_values();
    limits.max_storage_buffers_per_shader_stage = min_storage;
    // The world-cache decay/compaction passes use `@workgroup_size(1024)` (the prefix-sum scan width). wgpu's
    // default caps invocations-per-workgroup + workgroup_size_x at 256, so raise both to 1024 (desktop RTX
    // supports it; mirrors the renderer's `wgpu_settings()` bump).
    limits.max_compute_invocations_per_workgroup = limits.max_compute_invocations_per_workgroup.max(1024);
    limits.max_compute_workgroup_size_x = limits.max_compute_workgroup_size_x.max(1024);
    block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("voxel_rt_test_device_storage_buffers"),
        required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
        required_limits: limits,
        memory_hints: Default::default(),
        trace: wgpu::Trace::Off,
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
    }))
    .ok()
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
