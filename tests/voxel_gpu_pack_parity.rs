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
use adventure::voxel::incremental::{GpuPackBatch, ResidentPacker, SnapshotBuffers, index_class_words};
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

/// Run the GPU pack: zero-init the three pool buffers to the CPU snapshot's sizes (the allocations match), apply
/// the batch's CPU writes (uniform/freed metas + every AABB — we only read meta/voxel/palette here), dispatch
/// `voxel_pack` over the commands, and read the three buffers back.
struct GpuPools {
    voxel: Vec<u32>,
    brick_palettes: Vec<u32>,
    meta: Vec<u32>,
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

    // Zero-init the pool buffers. The shader writes the DENSE bricks' index/palette/meta; the CPU writes the
    // uniform/freed metas (we apply those host-side below into the meta backing before upload, so a uniform
    // brick's meta lands byte-identically — its id rides in the meta and the shader never touches it).
    let mut meta_host = vec![0u32; meta_words];
    for w in &batch.cpu_writes {
        if let Some(meta) = w.meta {
            let base = w.slot as usize * 12;
            meta_host[base..base + 12].copy_from_slice(bytemuck::cast_slice(std::slice::from_ref(&meta)));
        }
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

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pack_enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pack_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One workgroup per dense command.
        pass.dispatch_workgroups(batch.commands.len().max(1) as u32, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    GpuPools {
        voxel: readback_u32(device, queue, &voxel_buf, voxel_words),
        brick_palettes: readback_u32(device, queue, &palette_buf, palette_words),
        meta: readback_u32(device, queue, &meta_buf, meta_words),
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
    let cpu = cpu_snapshot(entries, reg);
    let batch = gpu_batch(entries, reg);
    let gpu = run_gpu_pack(device, queue, &batch, &cpu);

    // (1) META byte-equal over the whole capacity buffer (a permuted field / wrong lod_and_bits packing fails).
    let cpu_meta_words: &[u32] = bytemuck::cast_slice(&cpu.metas);
    assert_eq!(gpu.meta.len(), cpu_meta_words.len(), "{label}: meta buffer length differs");
    assert_eq!(gpu.meta, cpu_meta_words, "{label}: meta buffer bytes differ (GPU pack vs CPU snapshot)");

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
