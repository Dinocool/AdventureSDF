//! **Phase G "G-c.4-paging" — the PAGED-source residency-store parity gate** (`docs/PHASE_G_GC_PLAN.md` §8.4
//! gate 2).
//!
//! Proves the STREAMED `.vxo` prefetcher ([`StreamedResidencyPager`]) builds the GPU OCCUPANCY + the demand-paged
//! CORE store BIT-IDENTICALLY to the EAGER in-RAM structures the proven front end consumes (the
//! `voxel_gpu_residency_converge.rs` path) — over a moving-camera sequence — RESTRICTED to the CPU `ResidentPacker`
//! resident key set + its 26-halo (the COVERAGE INVARIANT). Because the live `GpuResidencyFrontEnd` is
//! store-AGNOSTIC (it reads whatever occupancy/core buffers `rebind_pool` binds, already validated on the eager
//! path), proving the paged buffers EQUAL the eager buffers proves the paged path renders identically.
//!
//! What it asserts after each camera step (a region crossing settle):
//!  * **occupancy parity** — for every CPU-resident key `(coord,lod)` AND its 6/26-neighbours, the paged GPU
//!    occupancy's `is_occupied` / `is_full` (read back + probed) EQUALS the eager `SectorOccupancy`'s (built from
//!    the in-RAM `StaticVoxSource`). No false air (a hole) and no false solid.
//!  * **core coverage + content** — every CPU-resident key + its 26-halo neighbours that EXIST have a resident
//!    core in the paged store, and its `8³` core is byte-identical to the eager `BrickCoreStore`'s. A missing
//!    halo core is the missing-core artifact (a render hole) — this fails it.
//!  * **constant-RAM** — the resident region count + resident core count stay bounded across the whole sequence
//!    (the LRU footprint, not the whole scene).
//!
//! Skips cleanly when no GPU adapter is available.

use adventure::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, BrickMap};
use adventure::voxel::edits::VoxelEdits;
use adventure::voxel::palette::{BlockId, BlockRegistry};
use adventure::voxel::residency_gpu::{BrickCoreStore, ResidencyProducer, SectorOccupancy, brick_key_hash};
use adventure::voxel::residency_pager::StreamedResidencyPager;
use adventure::voxel::source::{BrickSource, StaticVoxSource};
use adventure::voxel::streaming::{ResidencyManager, StreamingConfig};
use adventure::voxel::vxo::writer::{VxoCompression, VxoHeadParams, write_vxo};
use adventure::voxel::vxo::{MergedSource, VxoSource};
use bevy::math::IVec3;
use rustc_hash::FxHashSet;
use std::collections::HashSet;

#[path = "common/mod.rs"]
mod common;

const EMPTY_LOD: u32 = 0xFFFF_FFFF;

// --- GPU readback helpers ----------------------------------------------------------------------------------

fn read_u32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, words: usize) -> Vec<u32> {
    let size = (words * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paged_parity_rb"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
    queue.submit(std::iter::once(enc.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll readback");
    let data = staging.slice(..).get_mapped_range().expect("map readback");
    let out = bytemuck::cast_slice::<u8, u32>(&data).to_vec();
    drop(data);
    staging.unmap();
    out
}

// --- occupancy hash probe (mirror of the WGSL `is_occupied` over a read-back `entries` table) --------------

const SECTOR_EDGE: i32 = 4;

fn sector_split(coord: IVec3) -> (IVec3, IVec3) {
    let s = IVec3::new(coord.x.div_euclid(SECTOR_EDGE), coord.y.div_euclid(SECTOR_EDGE), coord.z.div_euclid(SECTOR_EDGE));
    let l = IVec3::new(coord.x.rem_euclid(SECTOR_EDGE), coord.y.rem_euclid(SECTOR_EDGE), coord.z.rem_euclid(SECTOR_EDGE));
    (s, l)
}

/// FNV-1a + avalanche over `(sector, lod)` — MUST match `residency_gpu::sector_hash` (mirrored here so the test
/// probes the read-back occupancy table exactly like the WGSL would).
fn sector_hash(sector: IVec3, lod: u32) -> u32 {
    let mut h: u32 = 2166136261;
    for w in [sector.x as u32, sector.y as u32, sector.z as u32, lod] {
        h ^= w;
        h = h.wrapping_mul(16777619);
        h ^= h >> 15;
        h = h.wrapping_mul(2654435761);
        h ^= h >> 13;
    }
    h
}

/// Probe a read-back occupancy `entries` table (8 u32/slot: [sx,sy,sz,lod,mlo,mhi,flo,fhi]) for a brick's
/// `(occupied, full)` bits. Mirrors the CPU `SectorOccupancy::is_occupied`/`is_full` SSOT.
fn occ_probe(entries: &[u32], table_size: u32, coord: IVec3, lod: u32) -> (bool, bool) {
    if table_size == 0 {
        return (false, false);
    }
    let (sector, local) = sector_split(coord);
    let bit = (local.x + local.y * SECTOR_EDGE + local.z * SECTOR_EDGE * SECTOR_EDGE) as u32;
    let mask = table_size - 1;
    let mut slot = sector_hash(sector, lod) & mask;
    for _ in 0..table_size {
        let base = slot as usize * 8;
        let e_lod = entries[base + 3];
        if e_lod == EMPTY_LOD {
            return (false, false);
        }
        if e_lod == lod
            && entries[base] == sector.x as u32
            && entries[base + 1] == sector.y as u32
            && entries[base + 2] == sector.z as u32
        {
            let occ = (entries[base + 4] as u64) | ((entries[base + 5] as u64) << 32);
            let full = (entries[base + 6] as u64) | ((entries[base + 7] as u64) << 32);
            return ((occ >> bit) & 1 != 0, (full >> bit) & 1 != 0);
        }
        slot = (slot + 1) & mask;
    }
    (false, false)
}

/// Probe a read-back core `table` (5 u32/slot) — the WGSL `core_lookup` semantics (stop at EMPTY, skip else).
fn core_probe(table: &[u32], table_size: u32, coord: IVec3, lod: u32) -> Option<u32> {
    let mask = table_size - 1;
    let mut slot = brick_key_hash(coord, lod) & mask;
    for _ in 0..table_size {
        let base = slot as usize * 5;
        let e_lod = table[base + 3];
        if e_lod == EMPTY_LOD {
            return None;
        }
        if e_lod == lod
            && table[base] == coord.x as u32
            && table[base + 1] == coord.y as u32
            && table[base + 2] == coord.z as u32
        {
            return Some(table[base + 4]);
        }
        slot = (slot + 1) & mask;
    }
    None
}

// --- the scene (a multi-region shape spanning a few regions + LOD shells) -----------------------------------

fn scene() -> BrickMap {
    let mut map = BrickMap::new();
    let solid = |id: u16| {
        let mut v = Box::new([BlockId::AIR; BRICK_VOXELS]);
        v.iter_mut().for_each(|c| *c = BlockId(id));
        Brick::from_voxels(v)
    };
    let partial = |id: u16| {
        let mut v = Box::new([BlockId(id); BRICK_VOXELS]);
        v[0] = BlockId::AIR; // one air voxel ⇒ never `Interior`
        Brick::from_voxels(v)
    };
    // A filled slab spanning >1 region (region edge K=8 bricks) so paging crosses region boundaries.
    for z in -10..10 {
        for x in -10..10 {
            map.insert(IVec3::new(x, 0, z), solid(1));
            map.insert(IVec3::new(x, 1, z), partial(2));
        }
    }
    // A tall pillar threading the LOD shells.
    for y in 2..14 {
        map.insert(IVec3::new(0, y, 0), solid(3));
    }
    map
}

fn registry() -> BlockRegistry {
    // The Cornell registry (AIR + 4 solids) covers the scene's BlockIds 1..=3. A SINGLE-asset MergedSource places
    // it at block_base 0 (identity), so a merged core's ids equal the raw map ids — `eager_core` is the oracle.
    BlockRegistry::cornell()
}

/// CPU resident key set at `cam` (the clipmap surface) via the SAME `ResidencyManager` cold-fill the converge
/// gate uses, over the in-RAM `StaticVoxSource`.
fn cpu_resident(cam: [f32; 3], cfg: &StreamingConfig, source: &StaticVoxSource, reg: &BlockRegistry) -> HashSet<(IVec3, u32)> {
    let mut mgr = ResidencyManager::new();
    let edits = VoxelEdits::new();
    for _ in 0..64 {
        mgr.update(cam, cfg, source);
        while mgr.pending() > 0 {
            mgr.drain_work_from(cfg, source, reg, &edits);
        }
        let before = mgr.resident_count();
        let dropped = mgr.update(cam, cfg, source);
        while mgr.pending() > 0 {
            mgr.drain_work_from(cfg, source, reg, &edits);
        }
        if dropped == 0 && mgr.resident_count() == before {
            break;
        }
    }
    mgr.resident_entries().into_iter().map(|e| (e.coord, e.lod)).collect()
}

/// The eager core for `(coord,lod)` from the in-RAM source (the oracle core content).
fn eager_core(source: &StaticVoxSource, reg: &BlockRegistry, coord: IVec3, lod: u32) -> [u32; BRICK_VOXELS] {
    let brick = source.brick(coord, lod, reg);
    let mut core = [0u32; BRICK_VOXELS];
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                core[adventure::voxel::brickmap::voxel_index(x, y, z)] = brick.get(x, y, z).0 as u32;
            }
        }
    }
    core
}

const N26: [IVec3; 27] = {
    let mut a = [IVec3::ZERO; 27];
    let mut i = 0;
    let mut dz = -1;
    while dz <= 1 {
        let mut dy = -1;
        while dy <= 1 {
            let mut dx = -1;
            while dx <= 1 {
                a[i] = IVec3::new(dx, dy, dz);
                i += 1;
                dx += 1;
            }
            dy += 1;
        }
        dz += 1;
    }
    a
};

#[test]
fn paged_source_occupancy_and_cores_match_eager_oracle() {
    let Some((device, queue)) = common::headless_device(wgpu::Features::empty()) else {
        eprintln!("paged_source_parity: no GPU adapter — skipping");
        return;
    };

    // 1. Bake a small `.vxo` (WITH a LODS pyramid) from the scene, open it as a single-asset MergedSource.
    let map = scene();
    let reg = registry();
    let dir = std::env::temp_dir().join("vrt_paged_parity");
    std::fs::create_dir_all(&dir).expect("mk tmp dir");
    let path = dir.join("paged_parity.vxo");
    let params = VxoHeadParams { name: "paged_parity".into(), ..Default::default() }; // bake_lods: true
    write_vxo(&path, &map, &reg, &params, VxoCompression::Store).expect("write_vxo");
    let (vxo, vxo_reg) = VxoSource::open(&path).expect("open VxoSource");
    let (merged, _merged_reg) = MergedSource::new(vec![(vxo, vxo_reg, IVec3::ZERO)]);
    let source_arc = std::sync::Arc::new(merged);

    // 2. The EAGER oracle structures (the proven front end's inputs), from the SAME map's in-RAM StaticVoxSource.
    let static_src = StaticVoxSource::new(&map);
    let eager_occ = SectorOccupancy::from_occupied_full(static_src.occupied_keys_full());
    let _eager_cores = BrickCoreStore::from_cores(
        static_src.occupied_keys().map(|(c, l)| (c, l, eager_core(&static_src, &reg, c, l))),
    );

    // 3. The pager over the streamed source.
    let clip_half = 8i32;
    let max_resident = 16384u32;
    let mut pager = StreamedResidencyPager::new(&device, &queue, source_arc, 1, clip_half, max_resident);

    let cfg = StreamingConfig {
        clip_half_bricks: clip_half,
        max_resident_bricks: usize::MAX,
        max_bricks_per_frame: usize::MAX,
    };

    // A camera sequence crossing region + LOD-shell boundaries (cold fill, +X stride, recede up, negative side).
    let span0 = adventure::voxel::brickmap::brick_span(0);
    let cams: [[f32; 3]; 5] = [
        [0.0, 1.0 * span0, 0.0],
        [20.0 * span0, 1.0 * span0, 0.0],
        [40.0 * span0, 6.0 * span0, 10.0 * span0],
        [-30.0 * span0, 2.0 * span0, -30.0 * span0],
        [0.0, 1.0 * span0, 0.0],
    ];

    let mut max_regions = 0usize;
    let mut max_cores = 0usize;
    let mut total_checked = 0usize;

    for (step, &cam) in cams.iter().enumerate() {
        // Drive the prefetcher (page the present regions ∩ clipmap, rebuild occupancy, page cores).
        pager.update(&queue, cam);
        max_regions = max_regions.max(pager.resident_region_count());
        max_cores = max_cores.max(pager.resident_core_count());

        // Read back the paged GPU occupancy + core table.
        let occ = pager.occupancy();
        let entries = read_u32(&device, &queue, &occ.entries, occ.table_size as usize * 8);
        let core_bufs = pager.core_buffers();
        let core_table = read_u32(&device, &queue, &core_bufs.table, core_bufs.table_size as usize * 5);
        let cores = read_u32(&device, &queue, &core_bufs.cores, (max_resident as usize * 2) * BRICK_VOXELS);

        // The CPU resident key set at this camera (the clipmap surface) + the coverage halo set.
        let resident = cpu_resident(cam, &cfg, &static_src, &reg);
        assert!(!resident.is_empty(), "step {step}: CPU resident set must be non-empty");

        // Coverage set = every resident key + its 26-halo neighbours (the cores the GPU halo-fill reads).
        let mut coverage: FxHashSet<(IVec3, u32)> = FxHashSet::default();
        for &(c, l) in &resident {
            for off in N26 {
                coverage.insert((c + off, l));
            }
        }

        for &(coord, lod) in &coverage {
            // (a) occupancy parity — the paged GPU occupancy EQUALS the eager occupancy for every coverage key.
            let (g_occ, g_full) = occ_probe(&entries, occ.table_size, coord, lod);
            let e_occ = eager_occ.is_occupied(coord, lod);
            let e_full = eager_occ.is_full(coord, lod);
            assert_eq!(g_occ, e_occ, "step {step}: occ mismatch @ {coord:?}@{lod}");
            assert_eq!(g_full, e_full, "step {step}: full mismatch @ {coord:?}@{lod}");

            // (b) core coverage + content — an OCCUPIED coverage key MUST have a resident core, byte-identical to
            //     the eager core (a missing core here is the missing-core render hole the invariant forbids).
            if e_occ {
                let idx = core_probe(&core_table, core_bufs.table_size, coord, lod)
                    .unwrap_or_else(|| panic!("step {step}: MISSING CORE (hole) @ {coord:?}@{lod}"));
                let got = &cores[idx as usize * BRICK_VOXELS..idx as usize * BRICK_VOXELS + BRICK_VOXELS];
                let want = eager_core(&static_src, &reg, coord, lod);
                assert_eq!(got, &want[..], "step {step}: core content mismatch @ {coord:?}@{lod}");
                total_checked += 1;
            }
        }
    }

    assert!(total_checked > 0, "the sequence must have checked some resident cores");
    // Constant-RAM: the resident region/core footprint stayed bounded (NOT the whole scene). With clip_half=8 over
    // this scene the resident set is a small surface shell — assert a generous-but-finite bound.
    assert!(max_regions <= 4096, "resident region count {max_regions} unbounded (constant-RAM violated)");
    assert!(max_cores <= max_resident as usize * 2, "resident core count {max_cores} exceeds the cap");
    eprintln!(
        "paged_source_parity: OK — checked {total_checked} resident cores; peak {max_regions} regions / {max_cores} cores"
    );
}
