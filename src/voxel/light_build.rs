//! **Stage 2 — the LIVE GPU emissive light-list builder** (`assets/shaders/voxel_lights.wgsl`).
//!
//! Runs the readback-free GPU NEE light build over the RESIDENT pool each time the pool changes, writing the
//! PERSISTENT `lights`/`alias` buffers the GI world-cache bind group consumes (bindings 15/16) — replacing the CPU
//! bake (`gpu.rs::build_lights_from_entries`). The light COUNT is read back 1-frame-late (a tiny u32 staging ring,
//! the same non-blocking discipline as the residency `change_count` mirror) and stamped into
//! `WorldCacheUniform.light_count`, so the GI shader is UNCHANGED. Proven byte-compatible with the CPU bake by
//! `tests/voxel_gpu_lights_parity.rs`.

use bytemuck::{Pod, Zeroable};

use super::gpu::MAX_VOXEL_LIGHTS;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightConfig {
    brick_count: u32,
    max_lights: u32,
    power_scale: f32,
    _pad: u32,
}

/// The persistent GPU light-build resource (pipelines + buffers + the 1-frame-late count mirror). Held in
/// `VoxelRtResources`; rebound to the live scene pool on an epoch change. `record` dispatches the 3 passes into a
/// caller encoder; `poll_count` reads the previous frame's light count out of band.
pub struct GpuLightBuilder {
    p_gather: wgpu::ComputePipeline,
    p_write: wgpu::ComputePipeline,
    p_alias: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    max_lights: u32,

    cfg_buf: wgpu::Buffer,
    cand_lights: wgpu::Buffer,
    cand_count: wgpu::Buffer,
    sum_power: wgpu::Buffer,
    /// The persistent light list (binding 15 in the world-cache bind group). Cap-sized; written each build.
    pub lights: wgpu::Buffer,
    /// The persistent alias table (binding 16). Cap-sized; written each build.
    pub alias: wgpu::Buffer,
    scaled: wgpu::Buffer,
    small: wgpu::Buffer,
    large: wgpu::Buffer,

    count_staging: Vec<wgpu::Buffer>,
    ring: usize,
    /// The last polled light count (0 until the first poll lands). Stamped into `WorldCacheUniform.light_count`.
    pub count: u32,

    bound: Option<wgpu::BindGroup>,
}

impl GpuLightBuilder {
    pub fn new(device: &wgpu::Device) -> Self {
        let max_lights = MAX_VOXEL_LIGHTS as u32;
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let mk = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: storage, mapped_at_creation: false })
        };
        let cfg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light_cfg"),
            size: core::mem::size_of::<LightConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cand_lights = mk("light_cand", (max_lights as u64) * 32);
        let cand_count = mk("light_cand_count", 4);
        let sum_power = mk("light_sum_power", 4);
        let lights = mk("voxel_rt_nee_lights", (max_lights as u64) * 32);
        let alias = mk("voxel_rt_nee_alias", (max_lights as u64) * 8);
        let scaled = mk("light_scaled", (max_lights as u64) * 4);
        let small = mk("light_small", (max_lights as u64) * 4);
        let large = mk("light_large", (max_lights as u64) * 4);
        let count_staging: Vec<wgpu::Buffer> = (0..2)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(if i == 0 { "light_count_staging0" } else { "light_count_staging1" }),
                    size: 4,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let src = include_str!("../../assets/shaders/voxel_lights.wgsl");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel_lights"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..13u32)
            .map(|b| wgpu::BindGroupLayoutEntry {
                binding: b,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if b == 4 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: b <= 3 }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("light_bgl"), entries: &entries });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("light_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let mk_p = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            p_gather: mk_p("gather_lights"),
            p_write: mk_p("write_lights"),
            p_alias: mk_p("build_alias"),
            bgl,
            max_lights,
            cfg_buf,
            cand_lights,
            cand_count,
            sum_power,
            lights,
            alias,
            scaled,
            small,
            large,
            count_staging,
            ring: 0,
            count: 0,
            bound: None,
        }
    }

    /// (Re)bind to the live scene pool buffers (call on an epoch change). `palette` is the GLOBAL palette buffer.
    pub fn rebind(
        &mut self,
        device: &wgpu::Device,
        metas: &wgpu::Buffer,
        voxel: &wgpu::Buffer,
        brick_palettes: &wgpu::Buffer,
        palette: &wgpu::Buffer,
    ) {
        let bufs: [&wgpu::Buffer; 13] = [
            metas, voxel, brick_palettes, palette, &self.cfg_buf, &self.cand_lights, &self.cand_count,
            &self.sum_power, &self.lights, &self.alias, &self.scaled, &self.small, &self.large,
        ];
        let entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .enumerate()
            .map(|(b, buf)| wgpu::BindGroupEntry { binding: b as u32, resource: buf.as_entire_binding() })
            .collect();
        self.bound = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("light_bg"),
            layout: &self.bgl,
            entries: &entries,
        }));
    }

    pub fn is_bound(&self) -> bool {
        self.bound.is_some()
    }

    pub fn unbind(&mut self) {
        self.bound = None;
    }

    /// Record the 3 light-build passes over `brick_count` resident pool slots into `enc`. Clears the per-build
    /// counters first (via `queue.write_buffer`, ordered before the submit) and copies the final count into this
    /// frame's staging-ring slot. Caller submits `enc`; the build writes the persistent `lights`/`alias` buffers.
    pub fn record(&self, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, brick_count: u32) {
        let Some(bg) = self.bound.as_ref() else { return };
        let cfg = LightConfig { brick_count, max_lights: self.max_lights, power_scale: 1.0e6, _pad: 0 };
        queue.write_buffer(&self.cfg_buf, 0, bytemuck::bytes_of(&cfg));
        queue.write_buffer(&self.cand_count, 0, bytemuck::bytes_of(&0u32));
        queue.write_buffer(&self.sum_power, 0, bytemuck::bytes_of(&0u32));
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("voxel_lights"), timestamp_writes: None });
            pass.set_bind_group(0, bg, &[]);
            pass.set_pipeline(&self.p_gather);
            pass.dispatch_workgroups(brick_count.div_ceil(64).max(1), 1, 1);
            pass.set_pipeline(&self.p_write);
            pass.dispatch_workgroups(self.max_lights.div_ceil(64), 1, 1);
            pass.set_pipeline(&self.p_alias);
            pass.dispatch_workgroups(1, 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.cand_count, 0, &self.count_staging[self.ring], 0, 4);
    }

    /// Read the PREVIOUS frame's light count out of band (1-frame-late, non-blocking-ish — the copy completed a
    /// frame ago). Updates `self.count` (capped to `max_lights`). Call once per frame, BEFORE `record`.
    pub fn poll_count(&mut self, device: &wgpu::Device) {
        let read_slot = 1 - self.ring;
        let staging = &self.count_staging[read_slot];
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r.is_ok());
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        if let Ok(true) = rx.try_recv() {
            if let Ok(d) = staging.slice(..).get_mapped_range() {
                let raw = u32::from_le_bytes([d[0], d[1], d[2], d[3]]);
                self.count = raw.min(self.max_lights);
            }
            staging.unmap();
        }
    }

    /// Advance the staging ring (call AFTER `record`).
    pub fn advance_ring(&mut self) {
        self.ring = 1 - self.ring;
    }
}
