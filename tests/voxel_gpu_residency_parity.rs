//! **Phase G "G-c.0" — the GPU-vs-CPU OCCUPANCY parity gate** (docs/PHASE_G_GC_PLAN.md §6 "G-c.0").
//!
//! The stage's whole point: the GPU-resident sparse brick occupancy
//! ([`adventure::voxel::residency_gpu::SectorOccupancy`] uploaded as the `voxel_residency.wgsl` sector hash)
//! must answer `is_occupied(brick_coord, lod)` IDENTICALLY to the CPU source's verdict over a representative
//! sample of `(coord, lod)` keys — occupied, empty, AND boundary. Mirrors the harness style of
//! `voxel_gpu_pack_parity.rs`: build a small KNOWN scene, upload, dispatch a tiny compute that reads
//! `is_occupied` into a readback buffer, and assert it EQUALS the CPU oracle byte-for-byte (exact, never loose).
//!
//! Two oracles are exercised so the gate proves the structure AND the live source agree:
//!   1. an explicit known `(coord, lod)` set → `SectorOccupancy::from_occupied` (the structure in isolation),
//!   2. a real `StaticVoxSource` over a baked Cornell `BrickMap` → `SectorOccupancy::from_occupied(occupied_keys)`
//!      vs `StaticVoxSource::classify != Air` (the SAME occupied set the CPU residency sees) — so the GPU bit
//!      equals the CPU residency's "this brick exists" decision.
//!
//! Skips cleanly when no GPU adapter is present (plain compute — no special features).

use adventure::voxel::brickmap::{BRICK_EDGE, Brick, BrickMap, MAX_LOD};
use adventure::voxel::palette::BlockId;
use adventure::voxel::residency_gpu::{
    GpuResidencyHeader, GpuSectorEntry, SectorOccupancy, split_sector,
};
use adventure::voxel::source::{BrickClass, BrickSource, StaticVoxSource};
use bevy::math::IVec3;
use wgpu::util::DeviceExt;

#[path = "common/mod.rs"]
mod common;

/// A query key — MUST match the WGSL `QueryKey` (4×u32, 16 B). `lod` rides as a `u32`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QueryKey {
    x: i32,
    y: i32,
    z: i32,
    lod: u32,
}

/// Dispatch `voxel_residency.wgsl`'s `residency_parity` over `keys` against the uploaded occupancy `occ`, and
/// read back the per-key `is_occupied` verdict (1/0). The single GPU round-trip the gate measures GPU-side.
fn gpu_is_occupied(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    occ: &SectorOccupancy,
    keys: &[QueryKey],
) -> Vec<u32> {
    let n = keys.len();
    assert!(n > 0, "the parity sample must be non-empty");

    // The uploaded occupancy = the header uniform (table size) + the sector-hash storage buffer.
    let header = occ.header();
    let header_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("residency_header"),
        contents: bytemuck::bytes_of(&header),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let entries_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("residency_entries"),
        contents: bytemuck::cast_slice(occ.entries()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let keys_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("residency_keys"),
        contents: bytemuck::cast_slice(keys),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("residency_out"),
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let src = std::fs::read_to_string("assets/shaders/voxel_residency.wgsl")
        .expect("read voxel_residency.wgsl");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_residency"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });

    // header@0 (uniform), entries@1 (read storage), keys@2 (read storage), out@3 (read_write storage) — the
    // bindings the WGSL declares at `@group(0)`.
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("residency_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("residency_pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("residency_pipeline"),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("residency_parity"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("residency_bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: header_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: entries_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: keys_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("residency_enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("residency_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    readback_u32(device, queue, &out_buf, n)
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn readback_u32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, words: usize) -> Vec<u32> {
    let bytes = (words * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("residency_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("residency_rb") });
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

/// Sanity: the CPU `GpuSectorEntry`/`GpuResidencyHeader` strides match the WGSL `SectorEntry`/`ResidencyHeader`
/// the shader reads (24 B / 16 B). A drift here would silently mis-stride the GPU read — catch it explicitly.
#[test]
fn gpu_struct_strides_match_wgsl() {
    assert_eq!(std::mem::size_of::<GpuSectorEntry>(), 32, "SectorEntry must be 8×u32 = 32 B");
    assert_eq!(std::mem::size_of::<GpuResidencyHeader>(), 16, "ResidencyHeader must be 4×u32 = 16 B");
    assert_eq!(std::mem::size_of::<QueryKey>(), 16, "QueryKey must be 4×u32 = 16 B");
}

/// **GATE 1 — an explicit KNOWN occupied set.** Build the structure from a scattered `(coord, lod)` set (incl.
/// negative coords + sector-boundary straddles), upload, and assert the GPU `is_occupied` over a dense
/// neighbourhood sample (occupied + empty + boundary) EXACTLY equals set membership.
#[test]
fn gpu_is_occupied_matches_known_set() {
    let Some((device, queue)) = common::headless_device(wgpu::Features::empty()) else {
        eprintln!("[skip] no GPU adapter — voxel GPU residency parity skipped");
        return;
    };

    let occupied: Vec<(IVec3, u32)> = vec![
        (IVec3::new(0, 0, 0), 0),
        (IVec3::new(3, 3, 3), 0),     // last brick of sector (0,0,0)
        (IVec3::new(4, 0, 0), 0),     // first brick of sector (1,0,0)
        (IVec3::new(-1, -1, -1), 0),  // sector (-1,-1,-1), local (3,3,3)
        (IVec3::new(-4, 2, 9), 2),
        (IVec3::new(100, -50, 7), 5),
        (IVec3::new(7, 7, 7), MAX_LOD),
    ];
    let occ = SectorOccupancy::from_occupied(occupied.iter().copied());
    let set: std::collections::HashSet<(IVec3, u32)> = occupied.iter().copied().collect();

    // The parity sample: every occupied key, the SAME coords at OTHER lods (per-LOD namespace), and a dense
    // box around the origin at every LOD (covers occupied / empty / boundary keys).
    let mut keys: Vec<QueryKey> = Vec::new();
    for &(c, l) in &occupied {
        keys.push(QueryKey { x: c.x, y: c.y, z: c.z, lod: l });
        let other = if l == 0 { 1 } else { 0 };
        keys.push(QueryKey { x: c.x, y: c.y, z: c.z, lod: other });
    }
    for lod in 0..=MAX_LOD {
        for z in -6..=6 {
            for y in -6..=6 {
                for x in -6..=6 {
                    keys.push(QueryKey { x, y, z, lod });
                }
            }
        }
    }

    let gpu = gpu_is_occupied(&device, &queue, &occ, &keys);
    assert_eq!(gpu.len(), keys.len());
    for (k, &g) in keys.iter().zip(&gpu) {
        let coord = IVec3::new(k.x, k.y, k.z);
        let cpu = occ.is_occupied(coord, k.lod);
        let want = set.contains(&(coord, k.lod));
        // GPU == CPU == oracle, EXACTLY (the whole point of the stage).
        assert_eq!(cpu, want, "CPU is_occupied disagreed with the known set at {coord:?}@{}", k.lod);
        assert_eq!(
            g == 1,
            want,
            "GPU is_occupied({coord:?}@{}) = {g} but the known set says {want}",
            k.lod
        );
    }
    let occupied_in_sample = gpu.iter().filter(|&&g| g == 1).count();
    eprintln!(
        "[gpu-residency-parity] known set: OK — {} keys, {occupied_in_sample} occupied, table_size {}",
        keys.len(),
        occ.table_size(),
    );
}

/// A small KNOWN [`BrickMap`]: a few solid bricks at scattered LOD0 coords (incl. a fully-solid 2×2×2 block so
/// the coarse pyramid + the `classify` Interior/Surface split are exercised). All-solid bricks so `is_full` is
/// meaningful; the occupancy is the brick's PRESENCE (`classify != Air`), independent of full/partial.
fn known_brickmap() -> BrickMap {
    let mut map = BrickMap::new();
    let solid = |id: u16| {
        let mut v = Box::new([BlockId::AIR; (BRICK_EDGE * BRICK_EDGE * BRICK_EDGE) as usize]);
        for c in v.iter_mut() {
            *c = BlockId(id);
        }
        Brick::from_voxels(v)
    };
    // A fully-solid 3×3×3 block at the origin: its CENTRE brick (1,1,1) has all 6 face-neighbours fully solid,
    // so `classify` returns Interior for it at LOD0 — still OCCUPIED (present), exercising the Interior path of
    // the occupancy. The 26 shell bricks are Surface. Plus a couple of isolated bricks (always Surface) at
    // off-origin + boundary-straddling coords.
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                map.insert(IVec3::new(x, y, z), solid(1));
            }
        }
    }
    map.insert(IVec3::new(5, 6, 7), solid(2));
    map.insert(IVec3::new(-3, 1, 4), solid(3)); // sector (-1,0,1) — negative-coord path
    map.insert(IVec3::new(8, 8, 8), solid(4)); // sector (2,2,2)
    map
}

/// **GATE 2 — a real [`StaticVoxSource`].** Build the occupancy from the source's `occupied_keys` and assert
/// the GPU `is_occupied` EXACTLY equals the SAME predicate the CPU residency uses to decide a brick exists:
/// `StaticVoxSource::classify(coord, lod) != BrickClass::Air`, over a dense sample at every LOD. This is the
/// load-bearing gate — it proves the GPU bit == the CPU residency's occupancy decision.
#[test]
fn gpu_is_occupied_matches_static_source_classify() {
    let Some((device, queue)) = common::headless_device(wgpu::Features::empty()) else {
        eprintln!("[skip] no GPU adapter — voxel GPU residency source parity skipped");
        return;
    };

    let map = known_brickmap();
    let source = StaticVoxSource::new(&map);
    let occ = SectorOccupancy::from_occupied(source.occupied_keys());

    // Dense sample at every LOD over a box that comfortably covers the scene + a margin of empty space (so the
    // sample includes occupied, surface, interior, and far-empty keys). The coarse LODs collapse the scene to
    // a single brick, so the small box still covers their occupied coords.
    let mut keys: Vec<QueryKey> = Vec::new();
    for lod in 0..=MAX_LOD {
        for z in -5..=11 {
            for y in -5..=11 {
                for x in -5..=11 {
                    keys.push(QueryKey { x, y, z, lod });
                }
            }
        }
    }

    let gpu = gpu_is_occupied(&device, &queue, &occ, &keys);
    assert_eq!(gpu.len(), keys.len());

    let mut occupied_seen = 0usize;
    let mut interior_seen = 0usize;
    for (k, &g) in keys.iter().zip(&gpu) {
        let coord = IVec3::new(k.x, k.y, k.z);
        // The CPU residency's "this brick exists" decision — the SSOT the GPU occupancy mirrors.
        let class = source.classify(coord, k.lod);
        let want = class != BrickClass::Air;
        if want {
            occupied_seen += 1;
        }
        if class == BrickClass::Interior {
            interior_seen += 1;
        }
        // The CPU structure must already agree with the source (built from its occupied_keys)...
        assert_eq!(
            occ.is_occupied(coord, k.lod),
            want,
            "CPU SectorOccupancy disagreed with StaticVoxSource::classify at {coord:?}@{} (class {class:?})",
            k.lod,
        );
        // ...and the GPU must agree with the CPU, EXACTLY.
        assert_eq!(
            g == 1,
            want,
            "GPU is_occupied({coord:?}@{}) = {g} but classify says occupied={want} (class {class:?})",
            k.lod,
        );
    }
    assert!(occupied_seen > 0, "the sample must contain occupied bricks to be a real gate");
    assert!(
        interior_seen > 0,
        "the fully-solid 2×2×2 block must yield ≥1 Interior brick (still occupied) to exercise that path"
    );
    eprintln!(
        "[gpu-residency-parity] StaticVoxSource: OK — {} keys, {occupied_seen} occupied ({interior_seen} interior), \
         {} sectors, table_size {}",
        keys.len(),
        occ.occupied_sectors(),
        occ.table_size(),
    );
}

/// Belt-and-braces: the GPU header's `table_size` round-trips (the WGSL probe masks with `table_size - 1`, so a
/// wrong size would corrupt every probe). A trivial occupancy + a single query proves the binding wiring.
#[test]
fn gpu_empty_occupancy_reads_all_unoccupied() {
    let Some((device, queue)) = common::headless_device(wgpu::Features::empty()) else {
        eprintln!("[skip] no GPU adapter — voxel GPU residency empty parity skipped");
        return;
    };
    let occ = SectorOccupancy::from_occupied(std::iter::empty());
    let keys: Vec<QueryKey> = (0..8)
        .map(|i| QueryKey { x: i, y: -i, z: i * 2, lod: (i as u32) % (MAX_LOD + 1) })
        .collect();
    let gpu = gpu_is_occupied(&device, &queue, &occ, &keys);
    assert!(gpu.iter().all(|&g| g == 0), "an empty occupancy must read every key unoccupied on the GPU");
    // Sanity that `split_sector` is the same import the structure uses (exercise the re-export).
    assert_eq!(split_sector(IVec3::new(4, -1, 0)).0, IVec3::new(1, -1, 0));
}
