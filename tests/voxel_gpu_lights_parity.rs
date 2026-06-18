//! **Stage 2 parity gate** — the GPU emissive light-list build (`assets/shaders/voxel_lights.wgsl`) must produce
//! the SAME NEE light list + power-weighted alias distribution as the CPU bake (`gpu.rs::build_lights_from_entries`
//! / `finalize_lights` / `build_alias_table`), so it can replace the CPU bake without touching the GI sampler.
//!
//! The GPU gather atomic-appends, so the list ORDER differs from the CPU's brick-then-cell order. We therefore
//! compare the light SET (keyed by quantized world position): exact `pos`/`area`/`radiance`, epsilon `inv_pdf`
//! (the GPU sums power as fixed-point u32 vs the CPU's f64 — unbiased, last-bits differ), and the alias SAMPLING
//! DISTRIBUTION (effective per-light draw probability, order-independent) within epsilon. Skips cleanly with no GPU.

use adventure::voxel::cornell::build_cornell;
use adventure::voxel::gpu::{GpuAliasEntry, GpuVoxelLight, MAX_VOXEL_LIGHTS, pack_brickmap};
use adventure::voxel::palette::BlockRegistry;

#[path = "common/mod.rs"]
mod common;

fn read_u32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, words: usize) -> Vec<u32> {
    let size = (words * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lights_rb"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
    queue.submit(std::iter::once(enc.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = staging.slice(..).get_mapped_range().expect("map");
    let out = bytemuck::cast_slice::<u8, u32>(&data).to_vec();
    drop(data);
    staging.unmap();
    out
}

/// Effective per-light draw probability of a Walker alias table (order-independent): a uniform slot pick (1/n)
/// keeps slot i with `prob[i]`, and every slot j that aliases to i contributes `(1-prob[j])/n`.
fn effective_probs(alias: &[GpuAliasEntry]) -> Vec<f64> {
    let n = alias.len();
    let mut p = vec![0.0f64; n];
    for (j, e) in alias.iter().enumerate() {
        p[j] += e.prob as f64 / n as f64;
        p[e.alias as usize] += (1.0 - e.prob as f64) / n as f64;
    }
    p
}

/// Key a light by its quantized world position (mm precision) so CPU/GPU lists match by identity despite order.
fn pos_key(pos: [f32; 3]) -> (i64, i64, i64) {
    let q = |v: f32| (v as f64 * 1000.0).round() as i64;
    (q(pos[0]), q(pos[1]), q(pos[2]))
}

#[test]
fn gpu_light_list_matches_cpu_bake() {
    let Some((device, queue)) = common::headless_compute_device_with_storage(64, 16) else {
        eprintln!("[skip] no GPU adapter — GPU light parity skipped");
        return;
    };

    // --- CPU oracle: pack the emissive Cornell box; its ceiling panel is the emitter set. ---
    let registry = BlockRegistry::cornell();
    assert!(registry.has_emitters(), "cornell registry must have emitters for this gate");
    let map = build_cornell(&registry);
    let patch = pack_brickmap(&map, &registry);
    let cpu_lights = &patch.lights;
    let cpu_alias = &patch.alias;
    assert!(!cpu_lights.is_empty(), "cornell must bake >0 emissive-voxel lights");
    assert!(cpu_lights.len() <= MAX_VOXEL_LIGHTS, "test scene must be under the cap (no truncation path)");
    eprintln!("[cpu] {} emissive-voxel lights", cpu_lights.len());

    // --- upload the resident pool (metas/voxels/brick_palettes/palette) exactly as the renderer binds it. ---
    use wgpu::util::DeviceExt;
    let mk_in = |label: &str, bytes: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: if bytes.is_empty() { &[0u8; 4] } else { bytes },
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let metas_buf = mk_in("metas", bytemuck::cast_slice(&patch.metas));
    let voxels_buf = mk_in("voxels", bytemuck::cast_slice(&patch.voxels));
    let bpal_buf = mk_in("brick_palettes", bytemuck::cast_slice(&patch.brick_palettes));
    let pal_buf = mk_in("palette", bytemuck::cast_slice(&patch.palette));

    let max_lights = MAX_VOXEL_LIGHTS as u32;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct LightConfig { brick_count: u32, max_lights: u32, power_scale: f32, _pad: u32 }
    let cfg = LightConfig { brick_count: patch.metas.len() as u32, max_lights, power_scale: 1.0e6, _pad: 0 };
    let cfg_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cfg"),
        contents: bytemuck::bytes_of(&cfg),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let mk_rw = |label: &str, size: u64| device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label), size, usage: storage, mapped_at_creation: false,
    });
    let cand_lights = mk_rw("cand_lights", (max_lights as u64) * 32);
    let cand_count = mk_rw("cand_count", 4);
    let sum_power = mk_rw("sum_power", 4);
    let lights_buf = mk_rw("lights", (max_lights as u64) * 32);
    let alias_buf = mk_rw("alias", (max_lights as u64) * 8);
    let scaled = mk_rw("scaled", (max_lights as u64) * 4);
    let small = mk_rw("small", (max_lights as u64) * 4);
    let large = mk_rw("large", (max_lights as u64) * 4);

    // pipelines
    let src = std::fs::read_to_string("assets/shaders/voxel_lights.wgsl").expect("read voxel_lights.wgsl");
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
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let mk_p = |entry: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry), layout: Some(&pl), module: &module, entry_point: Some(entry),
        compilation_options: Default::default(), cache: None,
    });
    let p_gather = mk_p("gather_lights");
    let p_write = mk_p("write_lights");
    let p_alias = mk_p("build_alias");

    let bufs = [
        &metas_buf, &voxels_buf, &bpal_buf, &pal_buf, &cfg_buf, &cand_lights, &cand_count, &sum_power,
        &lights_buf, &alias_buf, &scaled, &small, &large,
    ];
    let bg_entries: Vec<wgpu::BindGroupEntry> = bufs
        .iter()
        .enumerate()
        .map(|(b, buf)| wgpu::BindGroupEntry { binding: b as u32, resource: buf.as_entire_binding() })
        .collect();
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &bg_entries,
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_bind_group(0, &bg, &[]);
        pass.set_pipeline(&p_gather);
        pass.dispatch_workgroups((patch.metas.len() as u32).div_ceil(64).max(1), 1, 1);
        pass.set_pipeline(&p_write);
        pass.dispatch_workgroups(max_lights.div_ceil(64), 1, 1);
        pass.set_pipeline(&p_alias);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit(std::iter::once(enc.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");

    // --- read back + compare ---
    let gpu_count = read_u32(&device, &queue, &cand_count, 1)[0] as usize;
    assert_eq!(gpu_count, cpu_lights.len(), "GPU light count != CPU ({gpu_count} vs {})", cpu_lights.len());

    let lr = read_u32(&device, &queue, &lights_buf, gpu_count * 8);
    let gpu_lights: &[GpuVoxelLight] = bytemuck::cast_slice(&lr[..gpu_count * 8]);
    let ar = read_u32(&device, &queue, &alias_buf, gpu_count * 2);
    let gpu_alias: &[GpuAliasEntry] = bytemuck::cast_slice(&ar[..gpu_count * 2]);

    // Light SET parity (keyed by quantized pos).
    use std::collections::HashMap;
    let cpu_by_pos: HashMap<_, _> = cpu_lights.iter().map(|l| (pos_key(l.pos), l)).collect();
    let gpu_eff = effective_probs(gpu_alias);
    let cpu_eff = effective_probs(cpu_alias);
    let cpu_eff_by_pos: HashMap<_, _> =
        cpu_lights.iter().zip(&cpu_eff).map(|(l, &p)| (pos_key(l.pos), p)).collect();

    let mut max_invpdf_err = 0.0f64;
    let mut max_eff_err = 0.0f64;
    for (gi, gl) in gpu_lights.iter().enumerate() {
        let key = pos_key(gl.pos);
        let cl = cpu_by_pos.get(&key).unwrap_or_else(|| panic!("GPU light at {:?} not in CPU set", gl.pos));
        assert!((gl.area - cl.area).abs() < 1e-6, "area mismatch at {:?}", gl.pos);
        for k in 0..3 {
            assert!((gl.radiance[k] - cl.radiance[k]).abs() < 1e-5, "radiance mismatch at {:?}", gl.pos);
        }
        let rel = (gl.inv_pdf as f64 - cl.inv_pdf as f64).abs() / (cl.inv_pdf as f64).max(1e-6);
        max_invpdf_err = max_invpdf_err.max(rel);
        let ce = cpu_eff_by_pos[&key];
        max_eff_err = max_eff_err.max((gpu_eff[gi] - ce).abs());
    }
    eprintln!(
        "[parity] {gpu_count} lights match; max inv_pdf rel-err {:.2e}, max alias eff-prob err {:.2e}",
        max_invpdf_err, max_eff_err
    );
    assert!(max_invpdf_err < 1e-3, "inv_pdf rel-err too high: {max_invpdf_err:.3e}");
    assert!(max_eff_err < 1e-3, "alias sampling distribution diverges: {max_eff_err:.3e}");
}
