//! **Phase G Stage G-a — the BYTE-IDENTITY GATE for the GPU brick pack** (docs/PHASE_G_GALLERY_PLAN.md §G-a).
//!
//! The make-or-break anchor: the GPU pack (`assets/shaders/voxel_pack.wgsl` + [`ResidentPacker::update_gpu`])
//! MUST produce **byte-identical** pool buffers (`voxel_buf` / `brick_palettes_buf` / `meta_buf`) to the CPU
//! `ResidentPacker::update` + `snapshot_buffers` SSOT — for the SAME allocation decisions. This rig drives both
//! over a battery of resident sets, dispatches the real `voxel_pack` shader on a headless GPU, reads the pool
//! buffers back (TEST-ONLY readback), and asserts:
//!   1. `meta_buf` byte-equal (so a permuted field / wrong lod_and_bits packing fails),
//!   2. each dense brick's INDEX block byte-equal (so a wrong bit-pack fails),
//!   3. each dense brick's PALETTE block byte-equal — INCLUDING ORDER (so a permuted first-seen palette fails,
//!      even though it would decode to the same ids — the hardest risk the serial-palette-build mitigates),
//!   4. and, belt-and-braces, every haloed cell decodes identically via the SSOT `cell_block`.
//!
//! Cases: dense multi-material bricks, a uniform (R1) brick, and bricks with ABSENT neighbours (halo → AIR).
//! Skips cleanly when no GPU adapter is present (plain compute — no special features).

use adventure::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick};
use adventure::voxel::gpu::{GpuBrickMeta, GpuBrickPatch, ResidentBrick, halo_cells};
use adventure::voxel::incremental::{
    GpuClassifyBatch, GpuClassifyOut, GpuPackBatch, ResidentPacker, SnapshotBuffers, index_class_words,
};
use adventure::voxel::palette::{BlockId, BlockRegistry};
use adventure::sdf_render::worldgen::biome::{
    BiomeDef, BiomeId, BiomeLibrary, StrataLayer, TerrainMatId, TerrainSurfaceMaterial,
};
use bevy::math::IVec3;
use wgpu::util::DeviceExt;

#[path = "common/mod.rs"]
mod common;

/// A registry with a handful of materials so dense bricks carry multi-id palettes (k ≥ 2).
fn registry() -> BlockRegistry {
    let mat = |name: &str, c: [f32; 4]| TerrainSurfaceMaterial {
        name: name.into(),
        base_color: c,
        roughness: 0.9,
        blend: 0.0,
        texture: None,
        tiling: 4.0,
        ..Default::default()
    };
    let materials = vec![
        mat("a", [0.1, 0.2, 0.3, 1.0]),
        mat("b", [0.4, 0.5, 0.6, 1.0]),
        mat("c", [0.7, 0.8, 0.9, 1.0]),
        mat("d", [0.2, 0.9, 0.1, 1.0]),
        mat("e", [0.9, 0.1, 0.5, 1.0]),
    ];
    let biomes = BiomeId::ALL
        .iter()
        .map(|_| BiomeDef {
            name: "b".into(),
            surface: TerrainMatId(0),
            surface_rules: vec![],
            strata: vec![StrataLayer { material: TerrainMatId(0), thickness: 1.0 }],
            bedrock: TerrainMatId(1),
        })
        .collect();
    BlockRegistry::from_biome_library(&BiomeLibrary { materials, biomes })
}

/// A brick whose voxels cycle through `n_ids` solid block ids in a position-dependent pattern (plus AIR), so its
/// halo is non-trivial AND its palette has several entries in a content-dependent first-seen order — the case
/// most sensitive to a palette-ORDER bug.
fn multi_brick(seed: i32, n_ids: u16) -> Brick {
    let mut v = Box::new([BlockId::AIR; BRICK_VOXELS]);
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                let idx = (x + y * BRICK_EDGE + z * BRICK_EDGE * BRICK_EDGE) as usize;
                let h = (x * 7 + y * 13 + z * 5 + seed).rem_euclid(n_ids as i32 + 2);
                v[idx] = if h < 2 { BlockId::AIR } else { BlockId((1 + (h - 2) as u16).min(n_ids)) };
            }
        }
    }
    Brick::from_voxels(v)
}

/// The CPU SSOT pool: drive a fresh packer with `update` over `entries` (cold fill), then `snapshot_buffers`.
fn cpu_snapshot(entries: &[ResidentBrick<'_>], reg: &BlockRegistry) -> SnapshotBuffers {
    let mut packer = ResidentPacker::new(4096);
    packer.update(entries, reg.len() as u32);
    packer.snapshot_buffers(reg)
}

/// The GPU pool: drive a SECOND fresh packer with `update_gpu` over the SAME `entries` (so the allocation
/// decisions are identical), returning the batch. The allocation is deterministic + identical to the CPU path
/// (same dirty order, same arena), so the slot/offset layout matches the CPU snapshot byte-for-byte.
fn gpu_batch(entries: &[ResidentBrick<'_>], reg: &BlockRegistry) -> GpuPackBatch {
    let mut packer = ResidentPacker::new(4096);
    packer.update_gpu(entries, reg.len() as u32)
}

/// **Stage G4 — the GPU-CLASSIFY-driven batch.** Drive a fresh packer through `update_gpu_prepare`, dispatch the
/// REAL `classify_brick` shader on the GPU, read back the per-brick `(is_uniform, index_bits, palette_k)`, then
/// `update_gpu_finish` — the production G4 flow (NO CPU `pack_one`). The resulting [`GpuPackBatch`] MUST be the same
/// allocation the CPU-classify [`gpu_batch`] produces (the alloc is a deterministic function of the classification),
/// so feeding it to `run_gpu_pack` yields byte-identical pool buffers vs the CPU snapshot — the proof that moving
/// the classification to the GPU did NOT change the output, only WHO computes the sizes.
fn gpu_batch_g4(device: &wgpu::Device, queue: &wgpu::Queue, entries: &[ResidentBrick<'_>], reg: &BlockRegistry) -> GpuPackBatch {
    let mut packer = ResidentPacker::new(4096);
    let prepared = packer.update_gpu_prepare(entries, reg.len() as u32);
    let classify_out = run_classify(device, queue, &prepared);
    packer.update_gpu_finish(&prepared, &classify_out)
}

/// Dispatch the REAL `classify_brick` entry point over a [`GpuClassifyBatch`] (one workgroup per dirty key, reading
/// the prepared deduped cores + neighbour table) and read back the `classify_out` buffer (4 u32 / brick) as
/// [`GpuClassifyOut`]s. This is the round-trip G4 introduces — its cost is what the measurement weighs.
fn run_classify(device: &wgpu::Device, queue: &wgpu::Queue, batch: &GpuClassifyBatch) -> Vec<GpuClassifyOut> {
    let n = batch.commands.len();
    if n == 0 {
        return Vec::new();
    }
    // The classify pass reads its OWN `ClassifyCommand` (4 u32 / 16 B) at binding 9 (NOT the 60-B PackCommand @0),
    // so the upload is a straight `bytemuck` cast of the `GpuClassifyCommand` slice — no stride confusion.
    let cores_data: &[u32] = if batch.cores.is_empty() { &[0u32] } else { &batch.cores };
    let nbr_data: &[u32] = if batch.neighbour_indices.is_empty() { &[0u32] } else { &batch.neighbour_indices };

    let cmd_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("classify_commands"),
        contents: bytemuck::cast_slice(&batch.commands),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let cores_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("classify_cores"),
        contents: bytemuck::cast_slice(cores_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let nbr_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("classify_neighbours"),
        contents: bytemuck::cast_slice(nbr_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("classify_out"),
        size: (n * 4 * 4) as u64, // 4 u32 / brick
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let src = std::fs::read_to_string("assets/shaders/voxel_pack.wgsl").expect("read voxel_pack.wgsl");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_pack"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    // cores@1, neighbour_indices@2 (read-only), classify_out@8 (read_write), classify_commands@9 (read-only).
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("classify_bgl"),
        entries: &[entry(1, true), entry(2, true), entry(8, false), entry(9, true)],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("classify_pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("classify_pipeline"),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("classify_brick"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("classify_bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 1, resource: cores_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: nbr_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: cmd_buf.as_entire_binding() },
        ],
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("classify_enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("classify_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n as u32, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    let words = readback_u32(device, queue, &out_buf, n * 4);
    words
        .chunks_exact(4)
        .map(|w| GpuClassifyOut { is_uniform: w[0], uniform_block: w[1], palette_k: w[2], index_bits: w[3] })
        .collect()
}

/// Run the GPU pack: zero-init the pool buffers to the CPU snapshot's sizes (the allocations match), apply the
/// batch's CPU META writes (uniform/freed metas), dispatch `pack_brick` over the commands (dense encode) AND
/// `write_aabb` over the aabb commands (Stage G-b — every slot's AABB written GPU-side), and read all four
/// buffers back (meta/voxel/palette + the AABB buffer for the G-b byte-equality gate).
struct GpuPools {
    voxel: Vec<u32>,
    brick_palettes: Vec<u32>,
    meta: Vec<u32>,
    /// Stage G-b — the GPU-written AABB buffer (capacity-length, 8 u32 / 32 B per slot), for byte-equality vs the
    /// CPU `SnapshotBuffers.aabbs`.
    aabb: Vec<u32>,
}

fn run_gpu_pack(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    batch: &GpuPackBatch,
    cpu: &SnapshotBuffers,
) -> GpuPools {
    let meta_words = cpu.metas.len() * 12; // 48 B = 12 u32 per meta
    let voxel_words = cpu.indices.len();
    let palette_words = cpu.brick_palettes.len();
    let aabb_words = cpu.aabbs.len() * 8; // 32 B = 8 u32 per AABB

    // Zero-init the pool buffers. The shader writes the DENSE bricks' index/palette/meta; the CPU writes the
    // uniform/freed metas (we apply those host-side below into the meta backing before upload, so a uniform
    // brick's meta lands byte-identically — its id rides in the meta and the shader never touches it).
    let mut meta_host = vec![0u32; meta_words];
    for w in &batch.cpu_writes {
        let base = w.slot as usize * 12;
        meta_host[base..base + 12].copy_from_slice(bytemuck::cast_slice(std::slice::from_ref(&w.meta)));
    }

    let voxel_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pack_voxel"),
        size: (voxel_words * 4).max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let palette_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pack_palette"),
        size: (palette_words * 4).max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // meta starts from `meta_host` (carries the CPU-written uniform/freed metas already).
    let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pack_meta"),
        contents: bytemuck::cast_slice(&meta_host),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });

    // Stage G-b — the AABB buffer. It starts from the DEGENERATE-filled baseline (the same state the
    // `StreamSnapshot` seeds in production — every slot degenerate), then `write_aabb` overwrites EVERY changed
    // slot's AABB (resident → `brick_aabb`, freed → `degenerate_aabb`). A cold fill writes every resident slot,
    // so the result must equal the CPU `SnapshotBuffers.aabbs` byte-for-byte (incl. the degenerate freed/unused
    // slots, which `write_aabb` either re-writes degenerate or leaves at the degenerate baseline).
    let degenerate = adventure::voxel::incremental::degenerate_aabb();
    let mut aabb_host = vec![0u32; aabb_words];
    for slot in 0..cpu.aabbs.len() {
        let base = slot * 8;
        aabb_host[base..base + 8].copy_from_slice(bytemuck::cast_slice(std::slice::from_ref(&degenerate)));
    }
    let aabb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pack_aabb"),
        contents: bytemuck::cast_slice(&aabb_host),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });
    let aabb_cmd_data: Vec<u8> = if batch.aabb_commands.is_empty() {
        vec![0u8; 32]
    } else {
        bytemuck::cast_slice(&batch.aabb_commands).to_vec()
    };
    let aabb_cmd_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pack_aabb_commands"),
        contents: &aabb_cmd_data,
        usage: wgpu::BufferUsages::STORAGE,
    });

    // The command + cores + neighbour-table scratch SSBOs. Empty-safe (a min-size dummy when there are no dense
    // bricks).
    let cmd_bytes = if batch.commands.is_empty() {
        vec![0u8; 64]
    } else {
        bytemuck::cast_slice(&batch.commands).to_vec()
    };
    let cores_data: &[u32] = if batch.cores.is_empty() { &[0u32] } else { &batch.cores };
    let nbr_data: &[u32] = if batch.neighbour_indices.is_empty() { &[0u32] } else { &batch.neighbour_indices };
    let cmd_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pack_commands"),
        contents: &cmd_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let cores_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pack_cores"),
        contents: bytemuck::cast_slice(cores_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let nbr_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pack_neighbours"),
        contents: bytemuck::cast_slice(nbr_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let src = std::fs::read_to_string("assets/shaders/voxel_pack.wgsl").expect("read voxel_pack.wgsl");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_pack"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });

    let entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("pack_bgl"),
        entries: &[
            entry(0, true),  // commands
            entry(1, true),  // cores
            entry(2, true),  // neighbour_indices
            entry(3, false), // voxel_buf
            entry(4, false), // brick_palettes_buf
            entry(5, false), // meta_buf
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pack_pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pack_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("pack_brick"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pack_bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: cmd_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: cores_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: nbr_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: voxel_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: palette_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: meta_buf.as_entire_binding() },
        ],
    });

    // Stage G-b — the `write_aabb` pipeline + bind group (bindings 6 = aabb_buf read_write, 7 = aabb_commands
    // read-only — the WGSL hard-codes those numbers in the shared `@group(0)`). Same module as `pack_brick`.
    let aabb_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("pack_aabb_bgl"),
        entries: &[entry(6, false), entry(7, true)],
    });
    let aabb_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pack_aabb_pl"),
        bind_group_layouts: &[Some(&aabb_layout)],
        immediate_size: 0,
    });
    let aabb_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pack_aabb_pipeline"),
        layout: Some(&aabb_pl),
        module: &module,
        entry_point: Some("write_aabb"),
        compilation_options: Default::default(),
        cache: None,
    });
    let aabb_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pack_aabb_bg"),
        layout: &aabb_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 6, resource: aabb_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: aabb_cmd_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pack_enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pack_pass"),
            timestamp_writes: None,
        });
        // One workgroup per dense command (the dense encode).
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(batch.commands.len().max(1) as u32, 1, 1);
        // Then the AABB write — one invocation per aabb command (workgroup_size 64). Same pass / same encoder
        // (mirrors `apply_gpu_pack`'s fill-then-build: the AABB is filled here, before any BLAS build reads it).
        if !batch.aabb_commands.is_empty() {
            pass.set_pipeline(&aabb_pipeline);
            pass.set_bind_group(0, &aabb_bg, &[]);
            pass.dispatch_workgroups((batch.aabb_commands.len() as u32).div_ceil(64), 1, 1);
        }
    }
    queue.submit(std::iter::once(encoder.finish()));

    GpuPools {
        voxel: readback_u32(device, queue, &voxel_buf, voxel_words),
        brick_palettes: readback_u32(device, queue, &palette_buf, palette_words),
        meta: readback_u32(device, queue, &meta_buf, meta_words),
        aabb: readback_u32(device, queue, &aabb_buf, aabb_words),
    }
}

/// Copy a storage buffer's first `words` u32 to a staging buffer and map it back (test-only readback).
fn readback_u32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, words: usize) -> Vec<u32> {
    if words == 0 {
        return Vec::new();
    }
    let bytes = (words * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pack_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pack_rb") });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    queue.submit(std::iter::once(encoder.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = staging.slice(..).get_mapped_range().expect("map staging");
    let out: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    out
}

/// Wrap a pool triple as a `GpuBrickPatch` so the SSOT `cell_block` decode reads it.
fn as_patch(meta: &[u32], voxel: &[u32], palette: &[u32]) -> GpuBrickPatch {
    let metas: Vec<GpuBrickMeta> = bytemuck::cast_slice(meta).to_vec();
    GpuBrickPatch {
        aabbs: Vec::new(),
        metas,
        voxels: voxel.to_vec(),
        brick_palettes: palette.to_vec(),
        palette: Vec::new(),
        lights: Vec::new(),
        alias: Vec::new(),
    }
}

/// The full byte-identity assertion for one resident set: dispatch the GPU pack, compare meta/voxel/palette to
/// the CPU snapshot byte-for-byte (per-resident-dense-brick for voxel/palette; whole-buffer for meta), and decode
/// every haloed cell both ways.
fn assert_byte_identical(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    entries: &[ResidentBrick<'_>],
    reg: &BlockRegistry,
    label: &str,
) {
    assert_byte_identical_with(device, queue, entries, reg, label, false);
}

/// The G4 variant: drive the byte-identity gate through the GPU-CLASSIFY path (`update_gpu_prepare` → real
/// `classify_brick` dispatch → readback → `update_gpu_finish`) instead of the CPU-classify `update_gpu`. Proves the
/// pool is byte-identical when the classification comes from the GPU (no CPU `pack_one`) — the G4 correctness proof.
fn assert_byte_identical_g4(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    entries: &[ResidentBrick<'_>],
    reg: &BlockRegistry,
    label: &str,
) {
    assert_byte_identical_with(device, queue, entries, reg, label, true);
}

fn assert_byte_identical_with(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    entries: &[ResidentBrick<'_>],
    reg: &BlockRegistry,
    label: &str,
    g4: bool,
) {
    let cpu = cpu_snapshot(entries, reg);
    let batch = if g4 { gpu_batch_g4(device, queue, entries, reg) } else { gpu_batch(entries, reg) };
    let gpu = run_gpu_pack(device, queue, &batch, &cpu);

    // (1) META byte-equal over the whole capacity buffer (a permuted field / wrong lod_and_bits packing fails).
    let cpu_meta_words: &[u32] = bytemuck::cast_slice(&cpu.metas);
    assert_eq!(gpu.meta.len(), cpu_meta_words.len(), "{label}: meta buffer length differs");
    assert_eq!(gpu.meta, cpu_meta_words, "{label}: meta buffer bytes differ (GPU pack vs CPU snapshot)");

    // (1b) **Stage G-b — AABB byte-equal** over the whole capacity buffer: the GPU `write_aabb` pass must produce
    //      the SAME `aabb_buf` the CPU `SnapshotBuffers.aabbs` holds — `brick_aabb(world_min, lod)` for every
    //      resident (dense + uniform) slot AND `degenerate_aabb()` for every freed/unused slot. If the GPU
    //      epsilon/span math drifted, or a freed slot were left non-degenerate, the BLAS would build over wrong
    //      bounds and the render gate would diverge — this catches it at the byte level first.
    let cpu_aabb_words: &[u32] = bytemuck::cast_slice(&cpu.aabbs);
    assert_eq!(gpu.aabb.len(), cpu_aabb_words.len(), "{label}: aabb buffer length differs");
    assert_eq!(
        gpu.aabb, cpu_aabb_words,
        "{label}: AABB buffer bytes differ (GPU write_aabb vs CPU snapshot) — brick_aabb/degenerate_aabb mirror \
         drifted, or a dense/uniform/freed slot's box is wrong"
    );

    let degenerate = adventure::voxel::incremental::degenerate_aabb();
    let cpu_patch = as_patch(cpu_meta_words, &cpu.indices, &cpu.brick_palettes);
    let gpu_patch = as_patch(&gpu.meta, &gpu.voxel, &gpu.brick_palettes);

    // (2)+(3)+(4) per resident DENSE brick: index block + palette block byte-equal, and a full cell decode.
    let mut dense_seen = 0usize;
    for (slot, m) in cpu.metas.iter().enumerate() {
        if cpu.aabbs[slot] == degenerate || m.is_uniform() {
            continue; // freed slot or uniform brick (no index/palette block)
        }
        dense_seen += 1;
        // (2) INDEX block byte-equal.
        let ioff = m.dense_offset() as usize;
        let ilen = index_class_words(m.index_bits());
        assert_eq!(
            &gpu.voxel[ioff..ioff + ilen],
            &cpu.indices[ioff..ioff + ilen],
            "{label}: slot {slot} (origin {:?}) INDEX block differs (GPU bit-pack vs CPU)",
            m.voxel_origin,
        );
        // (3) PALETTE block byte-equal — INCLUDING ORDER (k entries; first-seen order must match exactly).
        let poff = m.palette_base as usize;
        let k = palette_k(m, &cpu.indices, &cpu.brick_palettes);
        assert_eq!(
            &gpu.brick_palettes[poff..poff + k],
            &cpu.brick_palettes[poff..poff + k],
            "{label}: slot {slot} (origin {:?}) PALETTE block differs — a permuted first-seen palette ORDER \
             (the serial-build mitigation failed?)",
            m.voxel_origin,
        );
        // (4) every haloed cell decodes identically via the SSOT cell_block.
        let gm = &gpu_patch.metas[slot];
        for cell in 0..halo_cells(m.lod()) {
            assert_eq!(
                gpu_patch.cell_block(gm, cell),
                cpu_patch.cell_block(m, cell),
                "{label}: slot {slot} cell {cell} (origin {:?}) decodes differently",
                m.voxel_origin,
            );
        }
    }
    assert!(dense_seen > 0, "{label}: the fixture must contain at least one dense brick to be a real gate");
    eprintln!("[gpu-pack-parity] {label}: OK — {dense_seen} dense bricks, {} commands byte-identical", batch.commands.len());
}

/// Recover a dense brick's palette length `k` by decoding its used indices' max + 1 (the palette is exactly the
/// distinct ids; the highest local index used is k-1). Robust without storing k: scan all haloed cells' local
/// indices. (Used only to bound the palette-block byte compare to the brick's `[base, base+k)`.)
fn palette_k(m: &GpuBrickMeta, indices: &[u32], _palette: &[u32]) -> usize {
    use adventure::voxel::gpu::decode_paletted_cell;
    let bits = m.index_bits();
    let off = m.dense_offset() as usize;
    // Decode local indices directly (a fake 1:1 palette) to find the max local index used.
    let fake: Vec<u16> = (0..=u16::MAX).collect();
    let mut max_local = 0u16;
    for cell in 0..halo_cells(m.lod()) {
        let local = decode_paletted_cell(&fake, bits, &indices[off..], cell);
        max_local = max_local.max(local);
    }
    max_local as usize + 1
}

/// **Stage G-b — the FREED-slot AABB gate (a streamed DROP sequence).** A cold fill (gen 1) then a DROP of part
/// of the set (gen 2) on a PERSISTENT GPU `aabb_buf` (carried across generations, exactly as `apply_gpu_pack`
/// patches the live scene buffer). Asserts the final GPU `aabb_buf` is byte-equal to a fresh CPU snapshot of the
/// gen-2 state — i.e. dropped slots are written `degenerate_aabb()` GPU-side, surviving slots keep their
/// `brick_aabb`. The cold-fill cases above never exercise a freed slot (nothing was resident before); this does.
#[test]
fn gpu_aabb_freed_slots_degenerate_over_drop_sequence() {
    let Some((device, queue)) = common::headless_device(wgpu::Features::empty()) else {
        eprintln!("[skip] no GPU adapter — voxel GPU AABB freed-slot gate skipped");
        return;
    };
    let reg = registry();

    // Gen 1 — a 3×3×3 dense block. Gen 2 — the SAME block minus the top layer (z == 2 dropped → 9 freed slots).
    let mk = |max_z: i32| -> Vec<(IVec3, Brick)> {
        let mut owned: Vec<(IVec3, Brick)> = Vec::new();
        for z in 0..max_z {
            for y in 0..3 {
                for x in 0..3 {
                    owned.push((IVec3::new(x, y, z), multi_brick(x * 11 + y * 7 + z * 3, 5)));
                }
            }
        }
        owned.sort_by_key(|(c, _)| (c.z, c.y, c.x));
        owned
    };
    let gen1_owned = mk(3);
    let gen2_owned = mk(2); // z=2 layer dropped
    let gen1: Vec<ResidentBrick> =
        gen1_owned.iter().map(|(c, b)| ResidentBrick { coord: *c, brick: b, lod: 0 }).collect();
    let gen2: Vec<ResidentBrick> =
        gen2_owned.iter().map(|(c, b)| ResidentBrick { coord: *c, brick: b, lod: 0 }).collect();

    // CPU SSOT: one packer driven gen1 → gen2, then snapshot the final state.
    let mut cpu_packer = ResidentPacker::new(4096);
    cpu_packer.update(&gen1, reg.len() as u32);
    cpu_packer.update(&gen2, reg.len() as u32);
    let cpu = cpu_packer.snapshot_buffers(&reg);

    // GPU: a second packer driven the SAME way; apply each generation's AABB commands to a PERSISTENT aabb_buf
    // seeded degenerate (the StreamSnapshot baseline). The drop's freed slots come back as `degenerate_aabb`.
    let mut gpu_packer = ResidentPacker::new(4096);
    let aabb_words = cpu.aabbs.len() * 8;
    let degenerate = adventure::voxel::incremental::degenerate_aabb();
    let mut aabb_host = vec![0u32; aabb_words];
    for slot in 0..cpu.aabbs.len() {
        let base = slot * 8;
        aabb_host[base..base + 8].copy_from_slice(bytemuck::cast_slice(std::slice::from_ref(&degenerate)));
    }
    let aabb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("seq_aabb"),
        contents: bytemuck::cast_slice(&aabb_host),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });

    for entries in [&gen1, &gen2] {
        let batch = gpu_packer.update_gpu(entries, reg.len() as u32);
        apply_aabb_commands(&device, &queue, &aabb_buf, &batch);
    }

    let gpu_aabb = readback_u32(&device, &queue, &aabb_buf, aabb_words);
    let cpu_aabb_words: &[u32] = bytemuck::cast_slice(&cpu.aabbs);
    assert_eq!(
        gpu_aabb, cpu_aabb_words,
        "GPU aabb_buf after a drop sequence must equal the CPU snapshot — a freed slot's degenerate box or a \
         survivor's brick_aabb diverged"
    );
    // Explicit: the 9 dropped slots are degenerate in the CPU snapshot (sanity that the fixture actually freed
    // slots), and thus the GPU matched them degenerate above.
    let degenerate_count = cpu.aabbs.iter().filter(|&&a| a == degenerate).count();
    assert!(
        degenerate_count >= 9,
        "expected ≥9 freed/unused degenerate slots after dropping the z=2 layer, got {degenerate_count}"
    );
    eprintln!("[gpu-pack-parity] freed-slot drop sequence: OK — {degenerate_count} degenerate slots byte-matched");
}

/// Dispatch ONLY `write_aabb` over a batch's `aabb_commands` into a persistent `aabb_buf` (the G-b AABB pass in
/// isolation — used by the drop-sequence gate to patch the same buffer across generations).
fn apply_aabb_commands(device: &wgpu::Device, queue: &wgpu::Queue, aabb_buf: &wgpu::Buffer, batch: &GpuPackBatch) {
    if batch.aabb_commands.is_empty() {
        return;
    }
    let cmd_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("seq_aabb_cmds"),
        contents: bytemuck::cast_slice(&batch.aabb_commands),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let src = std::fs::read_to_string("assets/shaders/voxel_pack.wgsl").expect("read voxel_pack.wgsl");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_pack"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("seq_aabb_bgl"),
        entries: &[entry(6, false), entry(7, true)],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("seq_aabb_pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("seq_aabb_pipeline"),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("write_aabb"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("seq_aabb_bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 6, resource: aabb_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: cmd_buf.as_entire_binding() },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("seq_aabb_enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("seq_aabb_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((batch.aabb_commands.len() as u32).div_ceil(64), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
}

/// Perf note (the GPU-path CPU cost): how much CPU work `update_gpu` does vs the all-CPU `update`. The GPU path
/// SKIPS `encode_paletted` (the bit-pack) on the CPU entirely, but ADDS the 27-core gather per dense brick. This
/// measures both over a large cold fill so the report is concrete (no GPU needed — it's a CPU-time comparison).
/// `#[ignore]`d (a measurement, not a gate); run with `--ignored --nocapture`.
#[test]
#[ignore = "perf measurement — run with: cargo test --test voxel_gpu_pack_parity gpu_pack_cpu_cost -- --ignored --nocapture"]
fn gpu_pack_cpu_cost() {
    let reg = registry();
    // A 16³ = 4096 dense-brick cold fill (every brick a multi-id brick → a dense command each).
    let edge = 16i32;
    let mut owned: Vec<(IVec3, Brick)> = Vec::new();
    for z in 0..edge {
        for y in 0..edge {
            for x in 0..edge {
                owned.push((IVec3::new(x, y, z), multi_brick(x * 31 + y * 17 + z * 11, 5)));
            }
        }
    }
    owned.sort_by_key(|(c, _)| (c.z, c.y, c.x));
    let entries: Vec<ResidentBrick> =
        owned.iter().map(|(c, b)| ResidentBrick { coord: *c, brick: b, lod: 0 }).collect();

    let mut cpu = ResidentPacker::new(8192);
    let t0 = std::time::Instant::now();
    let delta = cpu.update(&entries, reg.len() as u32);
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut gpu = ResidentPacker::new(8192);
    let t1 = std::time::Instant::now();
    let batch = gpu.update_gpu(&entries, reg.len() as u32);
    let gpu_ms = t1.elapsed().as_secs_f64() * 1e3;

    eprintln!(
        "[gpu-pack-perf] cold fill {} bricks: CPU update {:.2} ms ({} changed slots), \
         GPU update_gpu {:.2} ms ({} dense commands, {} cpu writes, {} core u32 = {} MB uploaded)",
        entries.len(),
        cpu_ms,
        delta.changed.len(),
        gpu_ms,
        batch.commands.len(),
        batch.cpu_writes.len(),
        batch.cores.len(),
        batch.cores.len() * 4 / (1024 * 1024),
    );
}

#[test]
fn gpu_pack_byte_identical_to_cpu_snapshot() {
    let Some((device, queue)) = common::headless_device(wgpu::Features::empty()) else {
        eprintln!("[skip] no GPU adapter — voxel GPU pack parity skipped");
        return;
    };
    let reg = registry();

    // Case A — a dense 3×3×3 block of multi-id bricks: every brick dense, halos non-trivial, several palette
    // entries (the palette-ORDER stress). The interior brick has all 26 neighbours present; the corner/edge/face
    // bricks have ABSENT neighbours (halo → AIR) — both halo branches in one fixture.
    {
        let mut owned: Vec<(IVec3, Brick)> = Vec::new();
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    owned.push((IVec3::new(x, y, z), multi_brick(x * 11 + y * 7 + z * 3, 5)));
                }
            }
        }
        owned.sort_by_key(|(c, _)| (c.z, c.y, c.x));
        let entries: Vec<ResidentBrick> =
            owned.iter().map(|(c, b)| ResidentBrick { coord: *c, brick: b, lod: 0 }).collect();
        assert_byte_identical(&device, &queue, &entries, &reg, "A: dense 3³ multi-material");
    }

    // Case B — a UNIFORM-incl-halo core inside a fully-solid 5×5×5: the inner 3×3×3 of bricks are solid AND
    // surrounded by solid on every side, so the centre 27 collapse to UNIFORM (R1) — no GPU command, a CPU meta
    // write. The OUTER shell bricks see AIR beyond the cube on ≥1 face, so their halo differs → DENSE (a GPU
    // command). This exercises the uniform path (CPU meta) ALONGSIDE dense commands in one batch, and proves a
    // uniform slot's meta lands byte-identically next to GPU-written dense slots.
    {
        let mut owned: Vec<(IVec3, Brick)> = Vec::new();
        for z in 0..5 {
            for y in 0..5 {
                for x in 0..5 {
                    owned.push((IVec3::new(x, y, z), Brick::uniform(BlockId(1))));
                }
            }
        }
        owned.sort_by_key(|(c, _)| (c.z, c.y, c.x));
        let entries: Vec<ResidentBrick> =
            owned.iter().map(|(c, b)| ResidentBrick { coord: *c, brick: b, lod: 0 }).collect();
        // The inner 3³ are uniform-incl-halo (no dense block), the 5³−3³ = 98 shell bricks are dense.
        assert_byte_identical(&device, &queue, &entries, &reg, "B: uniform core + dense shell");
    }

    // Case C — a SINGLE isolated dense brick: all 26 neighbours ABSENT, so the entire halo border is AIR. The
    // pure absent-neighbour halo case (block 0 fill everywhere outside the 8³ core).
    {
        let b = multi_brick(42, 4);
        let entries = [ResidentBrick { coord: IVec3::new(5, 5, 5), brick: &b, lod: 0 }];
        assert_byte_identical(&device, &queue, &entries, &reg, "C: isolated brick (all-AIR halo)");
    }

    // Case D — a non-zero LOD: the same dense block at LOD 3 (world_min / lod_and_bits differ; halo + encode are
    // LOD-invariant, but the meta fields must still pack correctly).
    {
        let mut owned: Vec<(IVec3, Brick)> = Vec::new();
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    owned.push((IVec3::new(x, y, z), multi_brick(x * 5 + y * 9 + z, 3)));
                }
            }
        }
        owned.sort_by_key(|(c, _)| (c.z, c.y, c.x));
        let entries: Vec<ResidentBrick> =
            owned.iter().map(|(c, b)| ResidentBrick { coord: *c, brick: b, lod: 3 }).collect();
        assert_byte_identical(&device, &queue, &entries, &reg, "D: dense 2³ at LOD 3");
    }
}

/// One parity fixture: the owned bricks + their LOD + a label.
type Fixture = (Vec<(IVec3, Brick)>, u32, &'static str);

/// The four parity fixtures — shared by the CPU-classify gate above and the G4 GPU-classify gate below, so both
/// prove byte-identity over the SAME battery (dense, uniform core, isolated, LOD 3).
fn parity_fixtures() -> Vec<Fixture> {
    let dense3 = |seed_lod: i32, n: u16| {
        let mut owned: Vec<(IVec3, Brick)> = Vec::new();
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    owned.push((IVec3::new(x, y, z), multi_brick(x * 11 + y * 7 + z * 3 + seed_lod, n)));
                }
            }
        }
        owned.sort_by_key(|(c, _)| (c.z, c.y, c.x));
        owned
    };
    let mut uniform5: Vec<(IVec3, Brick)> = Vec::new();
    for z in 0..5 {
        for y in 0..5 {
            for x in 0..5 {
                uniform5.push((IVec3::new(x, y, z), Brick::uniform(BlockId(1))));
            }
        }
    }
    uniform5.sort_by_key(|(c, _)| (c.z, c.y, c.x));
    let isolated = vec![(IVec3::new(5, 5, 5), multi_brick(42, 4))];
    let mut lod3: Vec<(IVec3, Brick)> = Vec::new();
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                lod3.push((IVec3::new(x, y, z), multi_brick(x * 5 + y * 9 + z, 3)));
            }
        }
    }
    lod3.sort_by_key(|(c, _)| (c.z, c.y, c.x));
    vec![
        (dense3(0, 5), 0, "A: dense 3³ multi-material"),
        (uniform5, 0, "B: uniform core + dense shell"),
        (isolated, 0, "C: isolated brick (all-AIR halo)"),
        (lod3, 3, "D: dense 2³ at LOD 3"),
    ]
}

/// **Stage G4 — the byte-identity gate through the GPU-CLASSIFY path.** The make-or-break G4 proof: drive each
/// fixture through `update_gpu_prepare` → the REAL `classify_brick` dispatch → readback → `update_gpu_finish` (NO
/// CPU `pack_one`), then run the GPU pack and assert the pool is byte-identical to the CPU snapshot. If the GPU
/// classification (is_uniform / palette_k / index_bits) drifted from the CPU `pack_one`, the alloc would differ and
/// the buffers would diverge — this catches it. Proves G4 changed only WHO computes the sizes, not the output.
#[test]
fn gpu_pack_g4_classify_byte_identical_to_cpu_snapshot() {
    let Some((device, queue)) = common::headless_device(wgpu::Features::empty()) else {
        eprintln!("[skip] no GPU adapter — voxel G4 GPU-classify parity skipped");
        return;
    };
    let reg = registry();
    for (owned, lod, label) in parity_fixtures() {
        let entries: Vec<ResidentBrick> =
            owned.iter().map(|(c, b)| ResidentBrick { coord: *c, brick: b, lod }).collect();
        assert_byte_identical_g4(&device, &queue, &entries, &reg, label);
    }
}

/// **Stage G4 — the perf measurement (the key risk).** Compares, over a 16³ = 4096 dense-brick cold fill, the
/// BASELINE CPU cost (`update`, the all-CPU pack — `pack_one` + `encode_paletted` for every brick), the G-b
/// GPU-pack CPU cost (`update_gpu`, which STILL `pack_one`s for the classification — the break-even), and the G4
/// GPU-classify CPU cost (`update_gpu_prepare` + `update_gpu_finish`, NO `pack_one`) plus the GPU `classify_brick`
/// dispatch + the readback/sync round-trip (the cost G4 introduces).
///
/// The win is `pack_one` GONE from the CPU; the risk is the readback sync. Reports all of them — plus a
/// production-shaped ~512-brick incremental case — so the recommendation is concrete. `#[ignore]`d (a measurement)
/// — run with `--ignored --nocapture`.
#[test]
#[ignore = "perf measurement — run with: cargo test --test voxel_gpu_pack_parity gpu_pack_g4_cpu_cost -- --ignored --nocapture"]
fn gpu_pack_g4_cpu_cost() {
    let Some((device, queue)) = common::headless_device(wgpu::Features::empty()) else {
        eprintln!("[skip] no GPU adapter — G4 perf measurement skipped");
        return;
    };
    let reg = registry();
    let edge = 16i32;
    let mut owned: Vec<(IVec3, Brick)> = Vec::new();
    for z in 0..edge {
        for y in 0..edge {
            for x in 0..edge {
                owned.push((IVec3::new(x, y, z), multi_brick(x * 31 + y * 17 + z * 11, 5)));
            }
        }
    }
    owned.sort_by_key(|(c, _)| (c.z, c.y, c.x));
    let entries: Vec<ResidentBrick> =
        owned.iter().map(|(c, b)| ResidentBrick { coord: *c, brick: b, lod: 0 }).collect();

    // BASELINE — the all-CPU pack.
    let mut cpu = ResidentPacker::new(8192);
    let t = std::time::Instant::now();
    let delta = cpu.update(&entries, reg.len() as u32);
    let cpu_ms = t.elapsed().as_secs_f64() * 1e3;

    // G-b — the CPU-classify GPU-pack producer (STILL runs pack_one for the classification).
    let mut gb = ResidentPacker::new(8192);
    let t = std::time::Instant::now();
    let gb_batch = gb.update_gpu(&entries, reg.len() as u32);
    let gb_ms = t.elapsed().as_secs_f64() * 1e3;

    // G4 — prepare (NO pack_one) + classify dispatch + readback + finish.
    let mut g4 = ResidentPacker::new(8192);
    let t = std::time::Instant::now();
    let prepared = g4.update_gpu_prepare(&entries, reg.len() as u32);
    let prepare_ms = t.elapsed().as_secs_f64() * 1e3;

    // WARM UP — the FIRST classify dispatch pays the one-time shader-module COMPILE + pipeline CREATE (which in
    // production is built ONCE at startup, cached in `VoxelRtPipelines`, NOT per re-pack). Time the WARM round-trip
    // (dispatch + map_async + device.poll(Wait)) so the measurement reflects the live per-re-pack cost, not the
    // cold compile. Average a few iterations to smooth GPU-queue jitter.
    let classify_out = run_classify(&device, &queue, &prepared); // warm-up (compiles the pipeline)
    let iters = 8;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        let _ = run_classify(&device, &queue, &prepared);
    }
    let readback_ms = (t.elapsed().as_secs_f64() * 1e3) / iters as f64;

    let t = std::time::Instant::now();
    let g4_batch = g4.update_gpu_finish(&prepared, &classify_out);
    let finish_ms = t.elapsed().as_secs_f64() * 1e3;
    let g4_cpu_ms = prepare_ms + finish_ms; // the CPU-side cost of the G4 path (no pack_one)

    // ISOLATED PURE-SYNC cost — the marginal GPU→CPU round-trip G4 ADDS over G-b: pipeline + cores/nbr/out buffers
    // built ONCE (production caches them; the cores buffer the PACK pass uploads anyway), then time ONLY
    // dispatch + submit + map_async + device.poll(Wait) (the actual stall). This is the number the stall question
    // hinges on (NOT the buffer-create churn the naive `run_classify` loop above includes).
    let sync_ms = measure_pure_classify_sync(&device, &queue, &prepared, 16);

    eprintln!(
        "[g4-perf] cold fill {} bricks:\n  \
         BASELINE  update (all-CPU)        {:.2} ms ({} changed)\n  \
         G-b       update_gpu (pack_one)   {:.2} ms ({} dense cmds, {} cpu writes)\n  \
         G4        prepare+finish (CPU)    {:.2} ms (prepare {:.2} + finish {:.2}) ({} dense cmds)\n  \
         G4        classify dispatch+read  {:.2} ms (warm GPU→CPU sync round-trip, UPPER bound: re-creates the\n  \
         {:>34}pipeline+scratch each call, which production caches/reuses)\n  \
         => G4 CPU pack-related cost {:.2} ms vs G-b {:.2} ms (pack_one removed); readback {:.2} ms",
        entries.len(),
        cpu_ms,
        delta.changed.len(),
        gb_ms,
        gb_batch.commands.len(),
        gb_batch.cpu_writes.len(),
        g4_cpu_ms,
        prepare_ms,
        finish_ms,
        g4_batch.commands.len(),
        readback_ms,
        "",
        g4_cpu_ms,
        gb_ms,
        readback_ms,
    );
    eprintln!("  G4        PURE classify sync       {sync_ms:.3} ms (pipeline+buffers cached; dispatch+submit+poll only)");
    // Sanity: the two paths agree on the dense command count (same classification).
    assert_eq!(g4_batch.commands.len(), gb_batch.commands.len(), "G4 and G-b must classify the same dense set");

    // PRODUCTION-REPRESENTATIVE incremental re-pack: a camera brick-crossing dirties a SHELL BAND, not a 4096-brick
    // cold fill (which routes through `snapshot_buffers` on the CPU, not the GPU pack). Measure a ~512-brick dirty
    // set so the report reflects the LIVE per-re-pack cost the classify+readback actually pays.
    let small_edge = 8i32; // 512 bricks
    let mut small_owned: Vec<(IVec3, Brick)> = Vec::new();
    for z in 0..small_edge {
        for y in 0..small_edge {
            for x in 0..small_edge {
                small_owned.push((IVec3::new(x, y, z), multi_brick(x * 31 + y * 17 + z * 11, 5)));
            }
        }
    }
    small_owned.sort_by_key(|(c, _)| (c.z, c.y, c.x));
    let small_entries: Vec<ResidentBrick> =
        small_owned.iter().map(|(c, b)| ResidentBrick { coord: *c, brick: b, lod: 0 }).collect();
    let mut g4s = ResidentPacker::new(8192);
    let t = std::time::Instant::now();
    let prepared_s = g4s.update_gpu_prepare(&small_entries, reg.len() as u32);
    let prep_s_ms = t.elapsed().as_secs_f64() * 1e3;
    let sync_s_ms = measure_pure_classify_sync(&device, &queue, &prepared_s, 16);
    eprintln!(
        "[g4-perf] incremental ~{} bricks (production-shaped): prepare {:.2} ms (CPU, no pack_one) + classify \
         sync {:.3} ms (GPU dispatch+readback)",
        small_entries.len(),
        prep_s_ms,
        sync_s_ms,
    );
}

/// Measure the ISOLATED per-re-pack GPU→CPU classify round-trip: build the classify pipeline + the cores/neighbour/
/// out/staging buffers ONCE (as production would — the cores buffer is the SAME one the pack pass uploads), then
/// time `iters` rounds of dispatch + submit + `map_async` + `device.poll(Wait)` + a fresh map. Returns the mean ms
/// — the marginal stall G4 adds (the readback payload is tiny: 4 u32 / brick).
fn measure_pure_classify_sync(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    batch: &GpuClassifyBatch,
    iters: u32,
) -> f64 {
    let n = batch.commands.len();
    if n == 0 {
        return 0.0;
    }
    let cmd_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sync_cmds"),
        contents: bytemuck::cast_slice(&batch.commands),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let cores_data: &[u32] = if batch.cores.is_empty() { &[0u32] } else { &batch.cores };
    let nbr_data: &[u32] = if batch.neighbour_indices.is_empty() { &[0u32] } else { &batch.neighbour_indices };
    let cores_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sync_cores"),
        contents: bytemuck::cast_slice(cores_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let nbr_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sync_nbr"),
        contents: bytemuck::cast_slice(nbr_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sync_out"),
        size: (n * 4 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sync_staging"),
        size: (n * 4 * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let src = std::fs::read_to_string("assets/shaders/voxel_pack.wgsl").expect("read voxel_pack.wgsl");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_pack"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sync_bgl"),
        entries: &[entry(1, true), entry(2, true), entry(8, false), entry(9, true)],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sync_pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sync_pipeline"),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("classify_brick"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sync_bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 1, resource: cores_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: nbr_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: cmd_buf.as_entire_binding() },
        ],
    });
    // Warm up once (prime the pipeline + queue), then time the steady-state round-trip.
    let run = |bytes: u64| {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sync_enc") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sync_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        staging.unmap();
    };
    // Dispatch-only (NO copy/map) timing — submit the classify then poll(Wait). Isolates the GPU classify COMPUTE
    // cost from the readback copy+map, so we can see whether the stall is the sync or the dispatch itself.
    let run_dispatch_only = || {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sync_enc_d") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sync_pass_d"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n as u32, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    };
    let bytes = (n * 4 * 4) as u64;
    run(bytes); // warm-up
    let t = std::time::Instant::now();
    for _ in 0..iters {
        run(bytes);
    }
    let full = (t.elapsed().as_secs_f64() * 1e3) / iters as f64;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        run_dispatch_only();
    }
    let dispatch = (t.elapsed().as_secs_f64() * 1e3) / iters as f64;
    eprintln!("  G4        classify DISPATCH-only    {dispatch:.3} ms (compute+poll, no copy/map — the GPU compute cost)");
    full
}
