//! Shared headless-GPU bring-up for the integration test rigs. Each rig declares `mod common;`
//! (cargo does NOT compile files under `tests/<dir>/` as their own test binary, so this is a plain
//! shared module). Not every rig uses every helper, hence the crate-level dead-code allow.
#![allow(dead_code)]

use futures_lite::future::block_on;

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
