//! Shared GPU upload helpers for the SDF render world.
//!
//! The single home for turning a `#[repr(C)]` + `bytemuck::Pod` GPU struct (point lights,
//! materials, light-grid cells, plain `u32` index runs) into a storage buffer via one
//! `bytemuck::cast_slice` — replacing the hand-rolled `to_le_bytes` field-by-field packing each
//! `prepare_*_gpu` system used to carry. Pod-cast is correct here because those structs are laid
//! out `#[repr(C)]` to match their std430 WGSL mirrors exactly (no implicit padding); their
//! `ShaderType` derive still drives the bind-group-layout min-size, so the two views agree.

use bevy::render::render_resource::{Buffer, BufferInitDescriptor, BufferUsages};
use bevy::render::renderer::RenderDevice;
use bytemuck::{Pod, Zeroable};

/// Create a read-only `STORAGE | COPY_DST` buffer from a Pod slice. Never zero-sized: an empty
/// slice uploads a single zeroed `T` so the storage binding stays valid (wgpu rejects a 0-byte
/// storage buffer). `label` is the wgpu debug label.
pub(crate) fn storage_buffer_init<T: Pod + Zeroable>(
    device: &RenderDevice,
    label: &str,
    data: &[T],
) -> Buffer {
    let fallback = [T::zeroed()];
    let src: &[T] = if data.is_empty() { &fallback } else { data };
    device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(src),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    })
}
