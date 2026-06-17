//! Shared headless-GPU bring-up for the integration test rigs. Each rig declares `mod common;`
//! (cargo does NOT compile files under `tests/<dir>/` as their own test binary, so this is a plain
//! shared module). Not every rig uses every helper, hence the crate-level dead-code allow.
#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{TextureFormat, TextureUsages, WgpuFeatures};
use bevy::render::renderer::RenderDevice;
use bevy::render::settings::{RenderCreation, WgpuSettings};
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;
use futures_lite::future::block_on;
use wgpu::util::DeviceExt;

use adventure::voxel::raytrace::VoxelRtPlugin;

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
/// Every entry point that shades or bounces (`trace_one`, `restir_probe`, `restir_dlss_p2`, …) now references
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

/// A headless wgpu (no special features) compute device that RAISES BOTH the compute-workgroup invocation
/// limits (to `min_invocations`, like [`headless_device_with_compute_limits`]) AND
/// `max_storage_buffers_per_shader_stage` (to `min_storage_buffers`, wgpu's default is 8). The G-c.2a GPU
/// residency-diff pass (`voxel_residency.wgsl` `diff_*` entries) binds ~21 storage buffers in one stage —
/// far over the default — so its parity rig needs this. Returns `None` (caller skips) if no adapter or it can't
/// reach the requested limits.
pub fn headless_compute_device_with_storage(
    min_invocations: u32,
    min_storage_buffers: u32,
) -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..Default::default()
    }))
    .ok()?;
    let info = adapter.get_info();
    eprintln!("GPU adapter: name={:?} backend={:?} device_type={:?}", info.name, info.backend, info.device_type);
    if adapter.limits().max_compute_invocations_per_workgroup < min_invocations {
        eprintln!(
            "adapter max_compute_invocations_per_workgroup {} < {min_invocations} — skipping",
            adapter.limits().max_compute_invocations_per_workgroup
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
    let mut limits = wgpu::Limits::default();
    limits.max_compute_invocations_per_workgroup =
        limits.max_compute_invocations_per_workgroup.max(min_invocations);
    limits.max_compute_workgroup_size_x = limits.max_compute_workgroup_size_x.max(min_invocations);
    limits.max_storage_buffers_per_shader_stage =
        limits.max_storage_buffers_per_shader_stage.max(min_storage_buffers);
    block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("voxel_diff_test_device"),
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
/// e.g. the DLSS-RR `restir_dlss_p2` entry (6: colour + diffuse/specular albedo + normal/roughness + depth +
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

// ===========================================================================================
// Shared headless-render harness (full Bevy `App` + `VoxelRtPlugin` + offscreen readback).
//
// `voxel_render_headless`, `voxel_cornell_headless` and `voxel_temporal_gpu` each boot the SAME
// headless render app — `DefaultPlugins` (no window/winit) + `ray_query` `RenderPlugin` +
// `VoxelRtPlugin`, rendering into an offscreen `Image` they read back every frame. This harness is
// the single SSOT for that boilerplate AND for the **#134 DLSS fix**: every full-App test goes
// through [`HeadlessRender::new`], which inserts the `DlssProjectId` BEFORE `DefaultPlugins`, so a
// default-feature (`dlss`) `cargo test` can never again hit the `DlssInitPlugin` startup panic —
// the failure is structurally impossible because there is no other way to build the test app.
// ===========================================================================================

/// **#134 fix (SSOT).** Insert the DLSS project UUID `bevy/dlss`'s `DlssInitPlugin` REQUIRES before
/// `DefaultPlugins`. On the default feature set (`dlss` on) that plugin (inside `DefaultPlugins`)
/// PANICS at device-init time unless a [`DlssProjectId`](bevy::anti_alias::dlss::DlssProjectId) is
/// already present — exactly the headless-test breakage tracked as issue #134. Mirrors `src/main.rs`.
/// A no-op without `dlss`. Called by [`HeadlessRender::new`] BEFORE `add_plugins(DefaultPlugins…)`,
/// so no full-App rig can forget it.
#[cfg(feature = "dlss")]
fn insert_dlss_project_id(app: &mut App) {
    app.insert_resource(bevy::anti_alias::dlss::DlssProjectId(
        bevy::asset::uuid::uuid!("b4f1d2c8-3a7e-4d92-9f60-7c5e1a8b3d04"),
    ));
}
#[cfg(not(feature = "dlss"))]
fn insert_dlss_project_id(_app: &mut App) {}

/// **#134 fix (SSOT).** Force the NON-DLSS render path for the headless rigs (dlss build only). DLSS-RR
/// resolves into a swapchain-shaped output the offscreen `Readback` does not capture, so with RR attached
/// the readback reads an all-zero (black) frame and every render-correctness assert fails. Setting
/// `DlssSettings.enabled = false` makes `sync_dlss_camera` never attach the `Dlss` component → the
/// temporal-accumulation fallback writes the offscreen target the readback DOES capture (the SAME path the
/// non-dlss build always uses, so the rig tests the same composite either way). A no-op without `dlss`.
/// Applied by [`HeadlessRender::new`] AFTER `add_plugins(VoxelRtPlugin)` so it overrides the plugin default.
#[cfg(feature = "dlss")]
fn disable_dlss_for_headless(app: &mut App) {
    app.insert_resource(adventure::voxel::raytrace::DlssSettings {
        enabled: false,
        mode: bevy::anti_alias::dlss::DlssPerfQualityMode::Quality,
    });
}
#[cfg(not(feature = "dlss"))]
fn disable_dlss_for_headless(_app: &mut App) {}

/// wgpu settings enabling AABB-BLAS `ray_query` — the SAME feature the app's `main.rs` requests.
fn rt_wgpu_settings() -> WgpuSettings {
    WgpuSettings { features: WgpuFeatures::EXPERIMENTAL_RAY_QUERY, ..default() }
}

/// CPU-side latest readback of an offscreen render target (raw `Rgba8UnormSrgb` bytes, row-padded by the
/// GPU copy). Shared by the readback observer (writes) and the test thread (reads).
#[derive(Resource, Clone, Default)]
pub struct LatestFrame(pub Arc<Mutex<Option<Vec<u8>>>>);

/// A headless Bevy render app rendering [`VoxelRtPlugin`] into an offscreen `Image`, read back every frame.
///
/// Construct with [`HeadlessRender::new`] (probes for a ray-query device — returns `None`/skip if absent —
/// and wires `DefaultPlugins` + the #134 DLSS fix + the ray-query `RenderPlugin` + `VoxelRtPlugin`). The
/// caller then customizes the world (insert the scene/streaming/toggle resources), spawns the camera with
/// [`HeadlessRender::spawn_camera`], calls [`HeadlessRender::finalize`] (spawns the readback observer +
/// `finish`/`cleanup`), and pumps frames with [`HeadlessRender::pump_until_lit`] or
/// [`HeadlessRender::collect_distinct_frames`]. Row-padding helpers ([`HeadlessRender::padded_row`],
/// [`HeadlessRender::px`], [`HeadlessRender::region_mean`]) decode the readback.
pub struct HeadlessRender {
    pub app: App,
    pub w: u32,
    pub h: u32,
    image_handle: Handle<Image>,
    pub latest: LatestFrame,
}

impl HeadlessRender {
    /// Boot the headless render app at `w × h`. Returns `None` (caller skips) if no `EXPERIMENTAL_RAY_QUERY`
    /// Vulkan adapter is present. Inserts the #134 `DlssProjectId` before `DefaultPlugins`, disables WinitPlugin
    /// and the primary window, sets the ray-query `RenderPlugin`, adds `VoxelRtPlugin`, forces the
    /// readback-capturable non-DLSS path, and creates the offscreen `COPY_SRC` render target. The scene/camera
    /// are the caller's to add next.
    pub fn new(w: u32, h: u32) -> Option<Self> {
        // Probe for a ray-query-capable adapter first; skip cleanly if absent (CI box without RT).
        headless_ray_query_device()?;

        let mut app = App::new();
        insert_dlss_project_id(&mut app); // #134 — before DefaultPlugins.
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>()
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(Box::new(rt_wgpu_settings())),
                    ..default()
                }),
        );
        app.add_plugins(VoxelRtPlugin);
        disable_dlss_for_headless(&mut app); // #134 — force the readback-capturable non-DLSS path.

        let latest = LatestFrame::default();
        app.insert_resource(latest.clone());

        let image_handle = {
            let mut images = app.world_mut().resource_mut::<Assets<Image>>();
            let mut image = Image::new_target_texture(w, h, TextureFormat::Rgba8UnormSrgb, None);
            image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
            images.add(image)
        };

        Some(Self { app, w, h, image_handle, latest })
    }

    /// Spawn the offscreen-targeted HDR `SdfCamera` looking from `cam_pos` at `target`. `name` labels the
    /// entity for diagnostics. (Streaming follows the `SdfCamera`.)
    pub fn spawn_camera(&mut self, cam_pos: Vec3, target: Vec3, name: &str) {
        self.app.world_mut().spawn((
            Camera3d::default(),
            RenderTarget::Image(self.image_handle.clone().into()),
            bevy::camera::Hdr,
            Msaa::Off,
            Transform::from_translation(cam_pos).looking_at(target, Vec3::Y),
            adventure::sdf_render::SdfCamera,
            Name::new(name.to_owned()),
        ));
    }

    /// Spawn the per-frame readback observer (latches the latest bytes into [`Self::latest`]) and `finish` +
    /// `cleanup` the app (what `App::run` does — unpacks the async `RenderDevice` into the main world, required
    /// before manual `update` loops). Call AFTER the camera + scene are set up.
    pub fn finalize(&mut self) {
        let sink = self.latest.0.clone();
        self.app
            .world_mut()
            .spawn(Readback::texture(self.image_handle.clone()))
            .observe(move |event: On<ReadbackComplete>| {
                *sink.lock().unwrap() = Some(event.data.clone());
            });
        self.app.finish();
        self.app.cleanup();
    }

    /// The GPU copy pads each row up to `COPY_BYTES_PER_ROW_ALIGNMENT`. The real bytes-per-row of the readback.
    pub fn padded_row(&self) -> usize {
        RenderDevice::align_copy_bytes_per_row((self.w * 4) as usize)
    }

    /// Pump frames until the latest read-back frame is meaningfully LIT (centre mean luma over a non-trivial
    /// threshold — the scene actually rendered), capped at `max_frames`, then return the latched bytes. The
    /// readback pipeline is a few frames deep + async, so a FIXED frame count is fragile (its tail can latch a
    /// stale warm-up/black readback). Returns the lit bytes, or the last bytes seen (so a genuine failure still
    /// gives the caller a meaningful diagnostic). `min_luma` is the centre mean-luma threshold (≈10.0 is typical).
    pub fn pump_until_lit(&mut self, max_frames: usize, min_luma: f32) -> Vec<u8> {
        let padded_row = self.padded_row();
        let need = padded_row * self.h as usize;
        let mut last = Vec::new();
        for _ in 0..max_frames {
            self.app.update();
            if let Some(b) = self.latest.0.lock().unwrap().clone()
                && b.len() >= need
            {
                last = b;
                if self.centre_mean_luma(&last) > min_luma {
                    break;
                }
            }
        }
        last
    }

    /// Pump up to `max_frames`, collecting each DISTINCT consecutive read-back snapshot (the readback is async +
    /// a few frames deep, so consecutive `update`s often see the same bytes), stopping once `want` distinct
    /// frames are gathered. Used to measure a time series (e.g. temporal-accumulation convergence).
    pub fn collect_distinct_frames(&mut self, max_frames: usize, want: usize) -> Vec<Vec<u8>> {
        let padded_row = self.padded_row();
        let need = padded_row * self.h as usize;
        let mut frames: Vec<Vec<u8>> = Vec::new();
        let mut last: Option<Vec<u8>> = None;
        for _ in 0..max_frames {
            self.app.update();
            if let Some(b) = self.latest.0.lock().unwrap().clone()
                && b.len() >= need
                && last.as_ref() != Some(&b)
            {
                last = Some(b.clone());
                frames.push(b);
            }
            if frames.len() >= want {
                break;
            }
        }
        frames
    }

    /// Mean luma over the centre half of `bytes` (a non-trivial value once the scene has rendered).
    pub fn centre_mean_luma(&self, bytes: &[u8]) -> f32 {
        let padded_row = self.padded_row();
        let (w, h) = (self.w as usize, self.h as usize);
        let (mut sum, mut n) = (0.0f32, 0.0f32);
        for y in (h / 4)..(h * 3 / 4) {
            for x in (w / 4)..(w * 3 / 4) {
                let p = &bytes[y * padded_row + x * 4..y * padded_row + x * 4 + 4];
                sum += 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32;
                n += 1.0;
            }
        }
        sum / n
    }

    /// One read-back RGB pixel at `(x, y)`.
    pub fn px(&self, bytes: &[u8], x: usize, y: usize) -> (f32, f32, f32) {
        px(bytes, self.padded_row(), x, y)
    }

    /// Average RGB over the rectangular screen region `[x0,x1) × [y0,y1)` (0..255 means).
    pub fn region_mean(&self, bytes: &[u8], x0: usize, x1: usize, y0: usize, y1: usize) -> (f32, f32, f32) {
        region_mean(bytes, self.padded_row(), x0, x1, y0, y1)
    }
}

// ===========================================================================================
// Shared synthetic-scene builders for the CPU/GPU residency + enumerate parity rigs.
// ===========================================================================================

use adventure::voxel::brickmap::{BRICK_EDGE, Brick, BrickMap};
use adventure::voxel::palette::BlockId;
use bevy::math::IVec3;

const BRICK_VOXELS: usize = (BRICK_EDGE * BRICK_EDGE * BRICK_EDGE) as usize;

/// A fully-solid brick of block `id` (every voxel set). `is_full` ⇒ Interior when buried.
pub fn full_brick(id: u16) -> Brick {
    let mut v = Box::new([BlockId(id); BRICK_VOXELS]);
    // (Start from the requested id; AIR base would be identical after the fill, but this is the SSOT form.)
    v.iter_mut().for_each(|c| *c = BlockId(id));
    Brick::from_voxels(v)
}

/// A nearly-solid brick of block `id` with ONE interior air voxel — so it is never `is_full`, hence always
/// classified Surface (exposed) regardless of neighbours. The parity rigs use this for exposed top layers.
pub fn partial_brick(id: u16) -> Brick {
    let mut v = Box::new([BlockId(id); BRICK_VOXELS]);
    v[0] = BlockId::AIR;
    Brick::from_voxels(v)
}

/// **The shared "representative" parity scene** (`voxel_gpu_enumerate_parity` ≡ `voxel_gpu_residency_diff_parity`
/// used byte-identical copies of this). A solid 6×3×6 ground slab straddling the origin into negative coords
/// (top layer partial/exposed, lower layers full/buried ⇒ Interior), a tall pillar threading the surface up
/// through finer→coarser shells (LOD-seam crossings), and an isolated +X cluster exercising a far positive
/// shell. Exercises Surface/Interior classification, negative coords, and shell crossings in one scene.
pub fn slab_pillar_cluster_scene() -> BrickMap {
    let mut map = BrickMap::new();
    for z in -3..3 {
        for x in -3..3 {
            for y in 0..3 {
                let brick = if y == 2 { partial_brick(2) } else { full_brick(1) };
                map.insert(IVec3::new(x, y, z), brick);
            }
        }
    }
    for y in 3..10 {
        map.insert(IVec3::new(0, y, 0), full_brick(3));
    }
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                map.insert(IVec3::new(15 + x, 4 + y, 15 + z), full_brick(4));
            }
        }
    }
    map
}

/// One read-back RGB pixel at `(x, y)` given the row-padding stride.
pub fn px(bytes: &[u8], padded_row: usize, x: usize, y: usize) -> (f32, f32, f32) {
    let row = &bytes[y * padded_row..];
    (row[x * 4] as f32, row[x * 4 + 1] as f32, row[x * 4 + 2] as f32)
}

/// Average RGB over a rectangular screen region `[x0,x1) × [y0,y1)` (returns 0..255 means).
pub fn region_mean(
    bytes: &[u8],
    padded_row: usize,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
) -> (f32, f32, f32) {
    let (mut r, mut g, mut b, mut n) = (0.0, 0.0, 0.0, 0.0);
    for y in y0..y1 {
        for x in x0..x1 {
            let (pr, pg, pb) = px(bytes, padded_row, x, y);
            r += pr;
            g += pg;
            b += pb;
            n += 1.0;
        }
    }
    (r / n, g / n, b / n)
}
