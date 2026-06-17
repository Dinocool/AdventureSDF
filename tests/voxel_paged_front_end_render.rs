//! **Phase G "G-c.4-paging" — the PAGED front-end DRIVE render gate** (`docs/PHASE_G_GC_PLAN.md` §8.4).
//!
//! The missing gate: drive the PRODUCTION [`GpuResidencyFrontEnd`] over the PRODUCTION
//! [`StreamedResidencyPager`]'s demand-paged occupancy + core stores — EXACTLY as
//! `raytrace.rs::drive_gpu_residency_front_end` does — to convergence, then read back the resident AABB pool and
//! assert it holds the SAME number of LIVE (non-degenerate, non-origin) bricks the EAGER front end produces over
//! the same scene. The paged-source PARITY gate (`voxel_paged_source_parity.rs`) proves the paged STORES are
//! bit-identical to the eager oracle for the CPU-resident COVERAGE set, but it never checks (a) that the front
//! end binds + reads them correctly, nor (b) that `is_occupied` returns FALSE outside the coverage set (a
//! false-positive there over-enumerates → enters the full pool → origin AABBs → blank). This gate closes both.
//!
//! Skips cleanly when no GPU adapter (or its compute/storage limits are too low).

use adventure::voxel::brickmap::{BRICK_VOXELS, Brick, BrickMap, MAX_LOD, brick_span, voxel_index};
use adventure::voxel::brickmap::{BRICK_EDGE};
use adventure::voxel::gpu::{GpuBrickAabb, GpuBrickMeta};
use adventure::voxel::incremental::degenerate_aabb;
use adventure::voxel::palette::{BlockId, BlockRegistry};
use adventure::voxel::residency_front_end::GpuResidencyFrontEnd;
use adventure::voxel::residency_gpu::{
    BrickCoreStore, GpuBrickCoreBuffers, GpuResidencyBuffers, SectorOccupancy,
};
use adventure::voxel::residency_pager::StreamedResidencyPager;
use adventure::voxel::source::{BrickSource, StaticVoxSource};
use adventure::voxel::vxo::writer::{VxoCompression, VxoHeadParams, write_vxo};
use adventure::voxel::vxo::{MergedSource, VxoSource};
use bevy::math::IVec3;
use bytemuck::Zeroable;

#[path = "common/mod.rs"]
mod common;

const META_WORDS: usize = 12;

// --- GPU readback ------------------------------------------------------------------------------------------

fn read_u32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, words: usize) -> Vec<u32> {
    let size = (words * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paged_fe_rb"),
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

// --- a small multi-region scene spanning LOD shells --------------------------------------------------------

fn scene() -> BrickMap {
    let mut map = BrickMap::new();
    let solid = |id: u16| {
        let mut v = Box::new([BlockId::AIR; BRICK_VOXELS]);
        v.iter_mut().for_each(|c| *c = BlockId(id));
        Brick::from_voxels(v)
    };
    let partial = |id: u16| {
        let mut v = Box::new([BlockId(id); BRICK_VOXELS]);
        v[0] = BlockId::AIR; // one air voxel ⇒ never Interior
        Brick::from_voxels(v)
    };
    for z in -10..10 {
        for x in -10..10 {
            map.insert(IVec3::new(x, 0, z), solid(1));
            map.insert(IVec3::new(x, 1, z), partial(2));
        }
    }
    for y in 2..14 {
        map.insert(IVec3::new(0, y, 0), solid(3));
    }
    map
}

fn registry() -> BlockRegistry {
    BlockRegistry::cornell()
}

fn eager_core(source: &StaticVoxSource, reg: &BlockRegistry, coord: IVec3, lod: u32) -> [u32; BRICK_VOXELS] {
    let brick = source.brick(coord, lod, reg);
    let mut core = [0u32; BRICK_VOXELS];
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                core[voxel_index(x, y, z)] = brick.get(x, y, z).0 as u32;
            }
        }
    }
    core
}

// --- a pool the front end writes (meta/voxel/palette/aabb), sized for `max_resident` ------------------------

struct Pool {
    meta: wgpu::Buffer,
    voxel: wgpu::Buffer,
    palette: wgpu::Buffer,
    aabb: wgpu::Buffer,
    max_resident: u32,
}

fn make_pool(device: &wgpu::Device, max_resident: u32) -> Pool {
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let meta = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pool_meta"),
        size: (max_resident as u64) * META_WORDS as u64 * 4,
        usage,
        mapped_at_creation: false,
    });
    let voxel = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pool_voxel"),
        size: ((max_resident as usize * 192).max(512) as u64) * 4,
        usage,
        mapped_at_creation: false,
    });
    let palette = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pool_palette"),
        size: ((max_resident as usize * 16).max(64) as u64) * 4,
        usage,
        mapped_at_creation: false,
    });
    // aabb init to the degenerate sentinel (free slots) — write_aabb overwrites only the packed slots.
    use wgpu::util::DeviceExt;
    let degenerate = degenerate_aabb();
    let mut aabb_host = vec![0u32; max_resident as usize * 8];
    for slot in 0..max_resident as usize {
        aabb_host[slot * 8..slot * 8 + 8]
            .copy_from_slice(bytemuck::cast_slice(std::slice::from_ref(&degenerate)));
    }
    let aabb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pool_aabb"),
        contents: bytemuck::cast_slice(&aabb_host),
        usage,
    });
    Pool { meta, voxel, palette, aabb, max_resident }
}

/// Count LIVE AABBs (min <= max on all axes) AND distinct degenerate vs origin-collapsed counts.
fn aabb_stats(device: &wgpu::Device, queue: &wgpu::Queue, pool: &Pool) -> (usize, usize, usize) {
    let raw = read_u32(device, queue, &pool.aabb, pool.max_resident as usize * 8);
    let aabbs: &[GpuBrickAabb] = bytemuck::cast_slice(&raw);
    let mut live = 0usize;
    let mut origin = 0usize;
    let mut degenerate = 0usize;
    for a in aabbs {
        let is_degen = a.min[0] > a.max[0] || a.min[1] > a.max[1] || a.min[2] > a.max[2];
        if is_degen {
            degenerate += 1;
        } else {
            live += 1;
            if a.min == [0.0; 3] && a.max == [0.0; 3] {
                origin += 1;
            }
        }
    }
    (live, origin, degenerate)
}

/// The set of live bricks as (lod, world_min bits) from the meta pool.
fn live_brick_set(device: &wgpu::Device, queue: &wgpu::Queue, pool: &Pool) -> std::collections::BTreeSet<(u32, [u32; 3])> {
    let raw = read_u32(device, queue, &pool.meta, pool.max_resident as usize * META_WORDS);
    let metas: &[GpuBrickMeta] = bytemuck::cast_slice(&raw);
    let zero = GpuBrickMeta::zeroed();
    metas
        .iter()
        .filter(|m| **m != zero)
        .map(|m| (m.lod(), [m.world_min[0].to_bits(), m.world_min[1].to_bits(), m.world_min[2].to_bits()]))
        .collect()
}

/// Drive a front end over `(occ, core)` to convergence at a fixed `cam` (mirrors the driver's per-frame loop,
/// minus the BLAS build — we only need the GPU-written pool). Returns the converged frame count (panics if it
/// never converges within the bound).
fn drive_to_convergence(
    fe: &mut GpuResidencyFrontEnd,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cam: [f32; 3],
    label: &str,
) -> u32 {
    // The 1-frame-late mirror reads Some(0) spuriously on the first frame (the staging ring starts zeroed), so we
    // cannot use it as the loop's sole convergence test. Drive a generous fixed number of frames — keep-old-until-
    // revealed cold-fill at these clip_half settles in a handful — then confirm the LAST two polled changes are 0.
    let mut history = Vec::new();
    for _ in 0..32 {
        let prev = fe.poll_change_count(device);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("paged_fe_frame") });
        fe.record_frame(queue, &mut enc, cam);
        queue.submit(std::iter::once(enc.finish()));
        fe.advance_ring();
        history.push(prev);
    }
    let tail = &history[history.len().saturating_sub(2)..];
    assert!(
        tail.iter().all(|c| *c == Some(0)),
        "{label}: not converged after 32 frames (last changes {tail:?}); full history {history:?}"
    );
    32
}

#[test]
fn paged_front_end_drive_renders_like_eager() {
    // The residency passes dispatch 512-wide workgroups and bind ~48 storage buffers in one stage.
    let Some((device, queue)) = common::headless_compute_device_with_storage(512, 48) else {
        eprintln!("[skip] no GPU adapter (or compute/storage limits too low) — paged front-end drive skipped");
        return;
    };

    let map = scene();
    let reg = registry();

    // --- the EAGER reference: front end over the in-RAM occupancy + cores (the proven path). ---
    let static_src = StaticVoxSource::new(&map);
    let eager_occ = SectorOccupancy::from_occupied_full(static_src.occupied_keys_full());
    let eager_cores = BrickCoreStore::from_cores(
        static_src.occupied_keys().map(|(c, l)| (c, l, eager_core(&static_src, &reg, c, l))),
    );

    let clip_half = 8i32;
    let max_resident = 16384u32;
    let span0 = brick_span(0);
    let cam = [0.5 * span0, 1.5 * span0, 0.5 * span0];

    let (eager_live, eager_set) = {
        let occ_bufs: GpuResidencyBuffers = eager_occ.upload(&device);
        let core_bufs: GpuBrickCoreBuffers = eager_cores.upload(&device);
        let pool = make_pool(&device, max_resident);
        let mut fe = GpuResidencyFrontEnd::new(&device, clip_half, max_resident);
        fe.rebind_pool(&device, &queue, &occ_bufs, &core_bufs, &pool.meta, &pool.voxel, &pool.palette, &pool.aabb);
        let frames = drive_to_convergence(&mut fe, &device, &queue, cam, "eager");
        let (live, origin, _degen) = aabb_stats(&device, &queue, &pool);
        eprintln!("[eager] converged in {frames} frames — {live} live AABBs ({origin} origin-collapsed)");
        assert!(live > 0, "eager front end produced ZERO live AABBs — the reference itself is broken");
        assert_eq!(origin, 0, "eager front end produced {origin} origin-collapsed AABBs");
        (live, live_brick_set(&device, &queue, &pool))
    };

    // --- the PAGED path: front end over the StreamedResidencyPager (exactly the driver's wiring). ---
    let dir = std::env::temp_dir().join("vrt_paged_fe_render");
    std::fs::create_dir_all(&dir).expect("mk tmp dir");
    let path = dir.join("paged_fe.vxo");
    let params = VxoHeadParams { name: "paged_fe".into(), ..Default::default() };
    write_vxo(&path, &map, &reg, &params, VxoCompression::Store).expect("write_vxo");
    let (vxo, vxo_reg) = VxoSource::open(&path).expect("open VxoSource");
    let (merged, _mreg) = MergedSource::new(vec![(vxo, vxo_reg, IVec3::ZERO)]);
    let source = std::sync::Arc::new(merged);

    let mut pager = StreamedResidencyPager::new(&device, &queue, source, 1, clip_half, max_resident);
    let pool = make_pool(&device, max_resident);
    let mut fe = GpuResidencyFrontEnd::new(&device, clip_half, max_resident);

    // Frame 0 wiring exactly like the driver: update the pager, then (since needs_rebind) rebind the front end.
    let mut history = Vec::new();
    let mut bound = false;
    for _ in 0..32 {
        pager.update(&queue, cam);
        if pager.take_needs_rebind() || !bound {
            let occ = pager.occupancy();
            let occ_owned = GpuResidencyBuffers {
                header: occ.header.clone(),
                entries: occ.entries.clone(),
                table_size: occ.table_size,
            };
            let core_owned = pager.core_buffers();
            fe.rebind_pool(&device, &queue, &occ_owned, &core_owned, &pool.meta, &pool.voxel, &pool.palette, &pool.aabb);
            bound = true;
        }
        let prev = fe.poll_change_count(&device);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("paged_fe_frame") });
        fe.record_frame(&queue, &mut enc, cam);
        queue.submit(std::iter::once(enc.finish()));
        fe.advance_ring();
        history.push(prev);
    }
    let frames = 32u32;
    let tail = &history[history.len().saturating_sub(2)..];
    assert!(tail.iter().all(|c| *c == Some(0)), "paged: not converged after 32 frames (last {tail:?}); history {history:?}");

    let (live, origin, degen) = aabb_stats(&device, &queue, &pool);
    let paged_set = live_brick_set(&device, &queue, &pool);
    let only_eager: Vec<_> = eager_set.difference(&paged_set).take(8).collect();
    let only_paged: Vec<_> = paged_set.difference(&eager_set).take(8).collect();
    eprintln!(
        "[paged] converged in {frames} frames — {live} live AABBs ({origin} origin-collapsed, {degen} degenerate); eager had {eager_live}"
    );
    eprintln!("[diff] only in eager (lod, world_min_bits-as-f32): {:?}", only_eager.iter().map(|(l, w)| (*l, w.map(f32::from_bits))).collect::<Vec<_>>());
    eprintln!("[diff] only in paged: {:?}", only_paged.iter().map(|(l, w)| (*l, w.map(f32::from_bits))).collect::<Vec<_>>());
    let _ = (live, eager_live, frames);
    assert_eq!(origin, 0, "paged front end produced {origin} ORIGIN-COLLAPSED AABBs (the blank-render symptom)");
    // The paged path must produce the SAME resident brick SET as the proven eager path (same scene + camera). The
    // SET is the robust comparator — the raw live-AABB count can flicker by ±1 (one slot's AABB lands degenerate
    // vs live depending on the per-frame drop/repack timing), identically in BOTH paths, so we compare sets.
    assert!(!paged_set.is_empty(), "paged front end produced an EMPTY resident set (blank)");
    assert_eq!(
        paged_set, eager_set,
        "paged resident brick SET diverges from eager — only-eager {only_eager:?}, only-paged {only_paged:?}"
    );

    // Sanity: the converged pool holds a substantial set of live metas (a real packed pool, not blank).
    let metas_raw = read_u32(&device, &queue, &pool.meta, max_resident as usize * META_WORDS);
    let metas: &[GpuBrickMeta] = bytemuck::cast_slice(&metas_raw);
    let zero = GpuBrickMeta::zeroed();
    let live_metas = metas.iter().filter(|m| **m != zero).count();
    assert!(live_metas > 0, "paged pool has zero live metas (blank)");
}

// --- occupancy probe over a read-back paged occupancy entries table (WGSL is_occupied mirror) -------------

const SECTOR_EDGE: i32 = 4;
fn occ_probe(entries: &[u32], table_size: u32, coord: IVec3, lod: u32) -> bool {
    if table_size == 0 {
        return false;
    }
    let sx = coord.x.div_euclid(SECTOR_EDGE);
    let sy = coord.y.div_euclid(SECTOR_EDGE);
    let sz = coord.z.div_euclid(SECTOR_EDGE);
    let lx = coord.x.rem_euclid(SECTOR_EDGE);
    let ly = coord.y.rem_euclid(SECTOR_EDGE);
    let lz = coord.z.rem_euclid(SECTOR_EDGE);
    let bit = (lx + ly * SECTOR_EDGE + lz * SECTOR_EDGE * SECTOR_EDGE) as u32;
    // sector_hash (FNV-1a + avalanche) — must match residency_gpu::sector_hash.
    let mut h: u32 = 2166136261;
    for w in [sx as u32, sy as u32, sz as u32, lod] {
        h ^= w;
        h = h.wrapping_mul(16777619);
        h ^= h >> 15;
        h = h.wrapping_mul(2654435761);
        h ^= h >> 13;
    }
    let mask = table_size - 1;
    let mut slot = h & mask;
    for _ in 0..table_size {
        let base = slot as usize * 8;
        let e_lod = entries[base + 3];
        if e_lod == EMPTY_LOD {
            return false;
        }
        if e_lod == lod && entries[base] == sx as u32 && entries[base + 1] == sy as u32 && entries[base + 2] == sz as u32 {
            let occ = (entries[base + 4] as u64) | ((entries[base + 5] as u64) << 32);
            return (occ >> bit) & 1 != 0;
        }
        slot = (slot + 1) & mask;
    }
    false
}

const EMPTY_LOD: u32 = 0xFFFF_FFFF;

/// **The SCALE reproduction + occupancy false-positive probe.** A LARGE scene + a wide clip_half — the regime the
/// blank-render bug lives in. Asserts the paged occupancy has NO false positives vs the eager oracle over the
/// whole clipmap box (a false positive ⇒ Pass B over-enumerates ⇒ full pool ⇒ origin AABBs ⇒ blank).
#[test]
fn paged_occupancy_no_false_positive_at_scale() {
    let Some((device, queue)) = common::headless_compute_device_with_storage(512, 48) else {
        eprintln!("[skip] no GPU adapter — scale occupancy probe skipped");
        return;
    };

    // A bigger scene: a thick filled slab + several pillars, spanning many regions and LOD shells.
    let mut map = BrickMap::new();
    let solid = |id: u16| {
        let mut v = Box::new([BlockId::AIR; BRICK_VOXELS]);
        v.iter_mut().for_each(|c| *c = BlockId(id));
        Brick::from_voxels(v)
    };
    for z in -32..32 {
        for x in -32..32 {
            for y in 0..4 {
                map.insert(IVec3::new(x, y, z), solid(1));
            }
        }
    }
    for y in 4..40 {
        map.insert(IVec3::new(0, y, 0), solid(2));
        map.insert(IVec3::new(20, y, 20), solid(3));
        map.insert(IVec3::new(-20, y, -20), solid(1));
    }
    let reg = registry();
    let static_src = StaticVoxSource::new(&map);
    let eager_occ = SectorOccupancy::from_occupied_full(static_src.occupied_keys_full());

    let dir = std::env::temp_dir().join("vrt_paged_scale");
    std::fs::create_dir_all(&dir).expect("mk tmp dir");
    let path = dir.join("paged_scale.vxo");
    let params = VxoHeadParams { name: "paged_scale".into(), ..Default::default() };
    write_vxo(&path, &map, &reg, &params, VxoCompression::Store).expect("write_vxo");
    let (vxo, vxo_reg) = VxoSource::open(&path).expect("open");
    let (merged, _m) = MergedSource::new(vec![(vxo, vxo_reg, IVec3::ZERO)]);
    let source = std::sync::Arc::new(merged);

    let clip_half = 24i32;
    let max_resident = 60_000u32; // core_cap=120k → cores buf ~245MB, under the headless 256MB max_buffer_size
    let span0 = brick_span(0);
    let cam = [0.5 * span0, 2.0 * span0, 0.5 * span0];

    let mut pager = StreamedResidencyPager::new(&device, &queue, source, 1, clip_half, max_resident);
    pager.update(&queue, cam);
    eprintln!(
        "[scale] {} regions, {} cores, occ table_size {}",
        pager.resident_region_count(),
        pager.resident_core_count(),
        pager.occupancy().table_size,
    );

    let occ = pager.occupancy();
    let entries = read_u32(&device, &queue, &occ.entries, occ.table_size as usize * 8);

    // Probe is_occupied over the WHOLE clipmap box at every LOD; any (occupied paged && !occupied eager) is a
    // FALSE POSITIVE — the over-enumeration root cause.
    use adventure::voxel::streaming::level_box_pub;
    let mut false_pos = 0usize;
    let mut false_neg = 0usize;
    let mut checked = 0usize;
    let mut first_fp: Option<(IVec3, u32)> = None;
    for lod in 0..=MAX_LOD {
        let (lo, hi) = level_box_pub(cam, lod, clip_half);
        // bound the probe volume so the test stays quick (sample a slab around the surface)
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let c = IVec3::new(x, y, z);
                    let g = occ_probe(&entries, occ.table_size, c, lod);
                    let e = eager_occ.is_occupied(c, lod);
                    checked += 1;
                    if g && !e {
                        false_pos += 1;
                        first_fp.get_or_insert((c, lod));
                    }
                    if !g && e {
                        false_neg += 1;
                    }
                }
            }
        }
    }
    eprintln!("[scale] probed {checked} keys: {false_pos} false-positive, {false_neg} false-negative; first FP {first_fp:?}");
    assert_eq!(false_pos, 0, "paged occupancy has {false_pos} FALSE POSITIVES (first {first_fp:?}) — over-enumeration root cause");
    // (false negatives outside the resident-region coverage are expected — the pager only pages clipmap-covering
    //  regions; eager has the whole scene. We only assert NO false positive, which is what over-enumerates.)
}
