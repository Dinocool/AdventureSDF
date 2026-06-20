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
    // Mirror the PRODUCTION pool reserves (incremental.rs `RESERVE_INDEX_WORDS_PER_BRICK`=256 / index_bits=8's 250w,
    // `RESERVE_PALETTE_WORDS_PER_BRICK`=256 / index_bits=8's max). index_bits=16 bricks degenerate (D3 `fits` guard),
    // so 256 holds every packable brick. A rich index_bits=8 brick needs 250 index + ≤256 palette words — under the
    // OLD 192/16 sizes BOTH pools overflowed (the corruption this gate reproduces). Keep them = the app so faithful.
    let voxel = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pool_voxel"),
        size: ((max_resident as usize * 256).max(256) as u64) * 4,
        usage,
        mapped_at_creation: false,
    });
    let palette = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pool_palette"),
        size: ((max_resident as usize * 256).max(64) as u64) * 4,
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

/// **Phase 4 BUDGET EVICTION.** When the desired surface set EXCEEDS `max_resident`, the front end must keep the
/// NEAREST `max_resident` bricks and EVICT the rest (distance-priority) — bounding VRAM at any view/scene size.
/// Drives the in-RAM eager path (the front-end diff is what evicts) at a budget far below the scene's surface,
/// and asserts: (1) resident NEVER exceeds the budget, (2) eviction actually happened (resident < the full
/// desired set), (3) the resident set is a valid SUBSET of the desired, (4) it converges (no thrash), and
/// (5) it ADAPTS to camera motion (the resident set shifts, still bounded — no leak/thrash).
#[test]
fn paged_front_end_budget_eviction_keeps_nearest() {
    let Some((device, queue)) = common::headless_compute_device_with_storage(512, 48) else {
        eprintln!("[skip] no GPU adapter — budget eviction gate skipped");
        return;
    };
    let map = scene();
    let reg = registry();
    let static_src = StaticVoxSource::new(&map);
    let occ_bufs = SectorOccupancy::from_occupied_full(static_src.occupied_keys_full()).upload(&device);
    let core_bufs = BrickCoreStore::from_cores(
        static_src.occupied_keys().map(|(c, l)| (c, l, eager_core(&static_src, &reg, c, l))),
    )
    .upload(&device);
    let clip_half = 8i32;
    let span0 = brick_span(0);
    let cam = [0.5 * span0, 1.5 * span0, 0.5 * span0];

    // The FULL desired set (huge budget = no eviction) — the reference the eviction must stay a subset of.
    let full = {
        let pool = make_pool(&device, 16384);
        let mut fe = GpuResidencyFrontEnd::new(&device, clip_half, 16384);
        fe.rebind_pool(&device, &queue, &occ_bufs, &core_bufs, &pool.meta, &pool.voxel, &pool.palette, &pool.aabb);
        drive_to_convergence(&mut fe, &device, &queue, cam, "full-budget");
        live_brick_set(&device, &queue, &pool)
    };
    let budget = 64u32;
    assert!(
        full.len() > budget as usize,
        "scene must exceed the budget to exercise eviction (full desired = {}, budget = {budget})",
        full.len()
    );

    // TINY budget — eviction must keep the nearest `budget`.
    let pool = make_pool(&device, budget);
    let mut fe = GpuResidencyFrontEnd::new(&device, clip_half, budget);
    fe.rebind_pool(&device, &queue, &occ_bufs, &core_bufs, &pool.meta, &pool.voxel, &pool.palette, &pool.aabb);
    drive_to_convergence(&mut fe, &device, &queue, cam, "budget-cold");
    let set = live_brick_set(&device, &queue, &pool);
    assert!(set.len() <= budget as usize, "resident {} EXCEEDED the budget {budget}", set.len());
    assert!(set.len() < full.len(), "no eviction happened (resident {} == full {})", set.len(), full.len());
    assert!(set.is_subset(&full), "the evicted resident set must be a SUBSET of the desired set");

    // Camera move → re-converge, still bounded, and the set SHIFTS (eviction adapts; no leak/thrash).
    let cam2 = [4.5 * span0, 1.5 * span0, 4.5 * span0];
    drive_to_convergence(&mut fe, &device, &queue, cam2, "budget-moved");
    let set2 = live_brick_set(&device, &queue, &pool);
    assert!(set2.len() <= budget as usize, "resident {} EXCEEDED the budget {budget} after move", set2.len());
    assert!(set2 != set, "eviction did not adapt to camera motion (resident set unchanged)");
}

/// **MULTI-ASSET paged drive (the gallery reproduction).** Two copies of the scene placed at DIFFERENT +X
/// offsets in one `MergedSource` (exactly the gallery's per-asset offset layout the single-asset gate never
/// exercises). Drive the REAL front end over the REAL pager with the camera AT THE FAR asset, then assert the
/// FAR asset's bricks are LIVE in the pool (non-degenerate). If only the near/first asset packs, this fails —
/// localizing the "gallery renders only Sponza" bug to the front-end/pager pack (NOT the BLAS sweep, which this
/// gate doesn't run).
#[test]
fn paged_front_end_multi_asset_far_asset_packs() {
    let Some((device, queue)) = common::headless_compute_device_with_storage(512, 48) else {
        eprintln!("[skip] no GPU adapter — multi-asset paged drive skipped");
        return;
    };

    let map = scene();
    let reg = registry();

    // Write the scene once, open it TWICE, place at ZERO and at +40 bricks in X (disjoint, like the gallery).
    let dir = std::env::temp_dir().join("vrt_paged_multi");
    std::fs::create_dir_all(&dir).expect("mk tmp dir");
    let path = dir.join("paged_multi.vxo");
    let params = VxoHeadParams { name: "paged_multi".into(), ..Default::default() };
    write_vxo(&path, &map, &reg, &params, VxoCompression::Store).expect("write_vxo");
    let far_off = IVec3::new(40, 0, 0);
    let (a0, r0) = VxoSource::open(&path).expect("open a0");
    let (a1, r1) = VxoSource::open(&path).expect("open a1");
    let (merged, _mreg) = MergedSource::new(vec![(a0, r0, IVec3::ZERO), (a1, r1, far_off)]);
    let source = std::sync::Arc::new(merged);

    let clip_half = 8i32;
    let max_resident = 16384u32;
    let span0 = brick_span(0);
    // Camera at the NEAR asset (origin) — the gallery START condition. The far asset (+40 bricks) is out of LOD0
    // reach (clip_half=8) and must enter via a COARSE shell, exactly like Sibenik/Conference at the gallery start.
    let cam = [0.5 * span0, 1.5 * span0, 0.5 * span0];

    let mut pager = StreamedResidencyPager::new(&device, &queue, source, 1, clip_half, max_resident);
    let pool = make_pool(&device, max_resident);
    let mut fe = GpuResidencyFrontEnd::new(&device, clip_half, max_resident);

    let mut bound = false;
    for _ in 0..48 {
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
        let _prev = fe.poll_change_count(&device);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("paged_multi_frame") });
        fe.record_frame(&queue, &mut enc, cam);
        queue.submit(std::iter::once(enc.finish()));
        fe.advance_ring();
    }

    let (live, origin, _degen) = aabb_stats(&device, &queue, &pool);
    let set = live_brick_set(&device, &queue, &pool);
    // Split the live set by world_min.x into near (asset 0, x≈0) and far (asset 1, x≈40 bricks).
    let far_min_x = 30.0 * span0; // asset 1 starts at brick +30 after the +40 offset shifts [-10,10]→[30,50]
    let near = set.iter().filter(|(_, w)| f32::from_bits(w[0]) < far_min_x).count();
    let far = set.iter().filter(|(_, w)| f32::from_bits(w[0]) >= far_min_x).count();
    eprintln!(
        "[multi] cam@near(origin): {live} live AABBs ({origin} origin) | near-asset bricks={near} far-asset(coarse) bricks={far}",
    );
    assert_eq!(origin, 0, "multi-asset: {origin} origin-collapsed AABBs");
    assert!(near > 0, "NEAR asset packed zero bricks (sanity)");
    assert!(far > 0, "FAR asset packed ZERO live bricks at COARSE LOD from the start camera — the gallery 'only first asset' bug");
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

/// **Headless black-cube reproduction — HALO CORRECTNESS.** The visible "black cubes" during camera motion are
/// wrong gbuffer normals, and a brick's normal is derived from its PACKED HALO (the 1-cell border that mirrors
/// the neighbour brick's boundary voxels). If a surface brick's face toward an OCCUPIED-but-face-culled neighbour
/// (e.g. the interior brick the front end never enters) is packed as spurious AIR, the trace sees a phantom
/// exposed face there → wrong normal → black. This gate drives the REAL front end over a SOLID 3×3×3-brick box
/// (centre (1,1,1) is Interior ⇒ face-culled ⇒ never resident; the face/edge/corner bricks are surface), then
/// decodes EVERY resident brick's 6 halo border planes and asserts a border is AIR **iff** the neighbour brick is
/// empty — never spurious AIR toward an occupied neighbour. With the occupancy-gated `NEIGHBOUR_SOLID` halo fix
/// this is green; without it the +interior-facing planes read AIR and it fails.
#[test]
fn paged_front_end_halo_no_spurious_air() {
    let Some((device, queue)) = common::headless_compute_device_with_storage(512, 48) else {
        eprintln!("[skip] no GPU adapter — halo correctness gate skipped");
        return;
    };

    let reg = registry();
    let mut map = BrickMap::new();
    let solid = || {
        let mut v = Box::new([BlockId::AIR; BRICK_VOXELS]);
        v.iter_mut().for_each(|c| *c = BlockId(1));
        Brick::from_voxels(v)
    };
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                map.insert(IVec3::new(x, y, z), solid());
            }
        }
    }

    let clip_half = 8i32;
    let max_resident = 16384u32;
    let span0 = brick_span(0);
    let cam = [1.5 * span0, 1.5 * span0, 1.5 * span0]; // centred on the box (spans [0,3)·span0)

    let dir = std::env::temp_dir().join("vrt_halo_gate");
    std::fs::create_dir_all(&dir).expect("mk tmp dir");
    let path = dir.join("halo.vxo");
    let params = VxoHeadParams { name: "halo".into(), ..Default::default() };
    write_vxo(&path, &map, &reg, &params, VxoCompression::Store).expect("write_vxo");
    let (vxo, vxo_reg) = VxoSource::open(&path).expect("open VxoSource");
    let (merged, _mreg) = MergedSource::new(vec![(vxo, vxo_reg, IVec3::ZERO)]);
    let source = std::sync::Arc::new(merged);

    let mut pager = StreamedResidencyPager::new(&device, &queue, source, 1, clip_half, max_resident);
    let pool = make_pool(&device, max_resident);
    let mut fe = GpuResidencyFrontEnd::new(&device, clip_half, max_resident);

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
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("halo_frame") });
        fe.record_frame(&queue, &mut enc, cam);
        queue.submit(std::iter::once(enc.finish()));
        fe.advance_ring();
    }

    let metas_raw = read_u32(&device, &queue, &pool.meta, max_resident as usize * META_WORDS);
    let metas: &[GpuBrickMeta] = bytemuck::cast_slice(&metas_raw);
    let voxels = read_u32(&device, &queue, &pool.voxel, (pool.voxel.size() / 4) as usize);
    let palettes = read_u32(&device, &queue, &pool.palette, (pool.palette.size() / 4) as usize);

    // CPU port of WGSL `cell_block` (voxel_lights.wgsl) over the read-back pool. hedge = BRICK_EDGE + 2 = 10.
    let cell_block = |m: &GpuBrickMeta, x: i32, y: i32, z: i32| -> u32 {
        if m.is_uniform() {
            return m.uniform_block().0 as u32;
        }
        let hedge = (BRICK_EDGE as i32) + 2;
        let ci = (x + y * hedge + z * hedge * hedge) as usize;
        let bits = m.index_bits() as u32;
        if bits == 0 {
            return voxels[m.dense_offset() as usize + ci];
        }
        let bit = ci as u32 * bits;
        let word = voxels[m.dense_offset() as usize + (bit / 32) as usize];
        let mask = if bits == 32 { 0xFFFF_FFFF } else { (1u32 << bits) - 1 };
        let local = (word >> (bit % 32)) & mask;
        palettes[m.palette_base as usize + local as usize]
    };

    let occupied = |c: IVec3| c.x >= 0 && c.x < 3 && c.y >= 0 && c.y < 3 && c.z >= 0 && c.z < 3;
    // (face direction, the pinned halo plane index on that face: 0 = low border, 9 = high border).
    let faces: [(IVec3, i32); 6] = [
        (IVec3::X, 9),
        (IVec3::NEG_X, 0),
        (IVec3::Y, 9),
        (IVec3::NEG_Y, 0),
        (IVec3::Z, 9),
        (IVec3::NEG_Z, 0),
    ];

    let zero = GpuBrickMeta::zeroed();
    let mut resident = 0usize;
    let mut spurious_air = 0usize; // border AIR toward an OCCUPIED neighbour — the black-cube bug
    let mut spurious_solid = 0usize; // border SOLID toward an EMPTY neighbour — over-fill (a distinct bug)
    let mut examples: Vec<String> = Vec::new();
    for m in metas.iter().filter(|m| **m != zero) {
        resident += 1;
        // Uniform bricks read their single block for every cell incl. the halo, so they have no AIR border by
        // construction ⇒ never a surface brick. Skip (the interior brick collapses to uniform & isn't resident
        // anyway, but a freed slot could in principle alias one — guard regardless).
        if m.is_uniform() {
            continue;
        }
        let coord = IVec3::new(
            (m.world_min[0] / span0).round() as i32,
            (m.world_min[1] / span0).round() as i32,
            (m.world_min[2] / span0).round() as i32,
        );
        for (d, plane) in faces {
            let want_solid = occupied(coord + d);
            let mut any_air = false;
            let mut any_solid = false;
            for a in 1..=8 {
                for b in 1..=8 {
                    let (x, y, z) = if d.x != 0 {
                        (plane, a, b)
                    } else if d.y != 0 {
                        (a, plane, b)
                    } else {
                        (a, b, plane)
                    };
                    if cell_block(m, x, y, z) == BlockId::AIR.0 as u32 {
                        any_air = true;
                    } else {
                        any_solid = true;
                    }
                }
            }
            if want_solid && any_air {
                spurious_air += 1;
                if examples.len() < 8 {
                    examples.push(format!(
                        "brick {coord:?} face {d:?}: AIR border toward OCCUPIED neighbour {:?} (any_solid={any_solid})",
                        coord + d
                    ));
                }
            }
            if !want_solid && any_solid {
                spurious_solid += 1;
            }
        }
    }
    eprintln!(
        "[halo] {resident} resident bricks; spurious_air(face holes)={spurious_air}, spurious_solid(over-fill)={spurious_solid}"
    );
    for e in &examples {
        eprintln!("  - {e}");
    }
    assert!(resident > 0, "no resident bricks — front end produced an empty pool");
    assert_eq!(
        spurious_air, 0,
        "halo has {spurious_air} face planes packed as AIR toward an OCCUPIED neighbour (the black-cube bug); examples:\n{}",
        examples.join("\n")
    );
}

/// Count STRICTLY-interior (cheb < `clip_half`) spurious-AIR halo faces in the current pool: a resident dense
/// brick whose face toward an OCCUPIED neighbour within the clip window is packed as AIR. Returns
/// `(interior_count, first_example)`. Shared by the motion sweep and its post-sweep SETTLE check (a still camera).
/// LOD-AWARE: every resident dense brick is analysed at ITS OWN lod. The front end pages a CLIPMAP of multiple
/// LODs, so a brick's coord is `world_min / brick_span(lod)` and its same-lod neighbour's occupancy is the lod-L
/// ground truth `occupied_at_lod`. A face is a spurious-AIR HOLE iff the neighbour is occupied (at that lod) AND
/// within the clip yet the ENTIRE border plane decodes to AIR (a definite phantom exposed face → wrong normal →
/// black cube). `all-air` (not any-air) avoids false positives at coarse-LOD partial-occupancy boundaries.
#[allow(clippy::too_many_arguments)]
fn count_interior_spurious(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pool: &Pool,
    max_resident: u32,
    clip_half: i32,
    cam: [f32; 3],
    occupied_at_lod: &dyn Fn(IVec3, u32) -> bool,
) -> (usize, String) {
    let metas_raw = read_u32(device, queue, &pool.meta, max_resident as usize * META_WORDS);
    let metas: &[GpuBrickMeta] = bytemuck::cast_slice(&metas_raw);
    let voxels = read_u32(device, queue, &pool.voxel, (pool.voxel.size() / 4) as usize);
    let palettes = read_u32(device, queue, &pool.palette, (pool.palette.size() / 4) as usize);
    let cell_block = |m: &GpuBrickMeta, x: i32, y: i32, z: i32| -> u32 {
        if m.is_uniform() {
            return m.uniform_block().0 as u32;
        }
        let hedge = (BRICK_EDGE as i32) + 2;
        let ci = (x + y * hedge + z * hedge * hedge) as usize;
        let bits = m.index_bits() as u32;
        if bits == 0 {
            return voxels[m.dense_offset() as usize + ci];
        }
        let bit = ci as u32 * bits;
        let word = voxels[m.dense_offset() as usize + (bit / 32) as usize];
        let mask = if bits == 32 { 0xFFFF_FFFF } else { (1u32 << bits) - 1 };
        let local = (word >> (bit % 32)) & mask;
        palettes[m.palette_base as usize + local as usize]
    };
    let faces: [(IVec3, i32); 6] = [
        (IVec3::X, 9),
        (IVec3::NEG_X, 0),
        (IVec3::Y, 9),
        (IVec3::NEG_Y, 0),
        (IVec3::Z, 9),
        (IVec3::NEG_Z, 0),
    ];
    let zero = GpuBrickMeta::zeroed();
    let mut count = 0usize;
    let mut first = String::new();
    for m in metas.iter().filter(|m| **m != zero) {
        if m.is_uniform() {
            continue; // uniform brick reads its block for every cell incl. halo ⇒ no AIR border by construction
        }
        let lod = m.lod();
        let span = brick_span(lod);
        let coord = IVec3::new(
            (m.world_min[0] / span).round() as i32,
            (m.world_min[1] / span).round() as i32,
            (m.world_min[2] / span).round() as i32,
        );
        let cam_brick = IVec3::new(
            (cam[0] / span).floor() as i32,
            (cam[1] / span).floor() as i32,
            (cam[2] / span).floor() as i32,
        );
        for (d, plane) in faces {
            let nb = coord + d;
            if !occupied_at_lod(nb, lod) {
                continue;
            }
            if (nb - cam_brick).abs().max_element() >= clip_half {
                continue; // edge ring / trailing — not a strictly-interior visible cube
            }
            let all_air = (1..=8).all(|a| {
                (1..=8).all(|b| {
                    let (x, y, z) = if d.x != 0 {
                        (plane, a, b)
                    } else if d.y != 0 {
                        (a, plane, b)
                    } else {
                        (a, b, plane)
                    };
                    cell_block(m, x, y, z) == BlockId::AIR.0 as u32
                })
            });
            if all_air {
                count += 1;
                if first.is_empty() {
                    first = format!("lod{lod} brick {coord:?} face {d:?} ALL-AIR border toward OCCUPIED {nb:?}");
                }
            }
        }
    }
    (count, first)
}

/// **Headless black-cube reproduction — HALO CORRECTNESS UNDER CAMERA MOTION (the transient).** The static gate
/// above only checks the CONVERGED pool. The "black cubes appear while moving" symptom could instead be a
/// TRANSIENT: as the clip window slides, bricks evict at the trailing edge and enter at the leading edge, and a
/// brick re-packed in a frame whose neighbour's occupancy/core is momentarily out of sync could carry a spurious
/// AIR halo for a frame or two → flickering black faces along the motion frontier. This sweeps the camera along a
/// long solid wall (forcing continuous re-paging churn) and re-checks the FULL halo invariant on EVERY frame, not
/// just at the end. If geometry is clean per-frame, the residual black cubes are NOT halo/geometry — they are the
/// lighting/world-cache disocclusion path.
#[test]
fn paged_front_end_halo_no_spurious_air_under_motion() {
    let Some((device, queue)) = common::headless_compute_device_with_storage(512, 48) else {
        eprintln!("[skip] no GPU adapter — moving halo gate skipped");
        return;
    };

    // A long solid WALL: brick coords x∈[0,WX), y,z∈[0,3). Sweeping the camera along +X slides the clip window so
    // bricks continuously evict (trailing) and enter (leading) — the streaming churn the static box never sees.
    const WX: i32 = 24;
    let reg = registry();
    let mut map = BrickMap::new();
    let solid = || {
        let mut v = Box::new([BlockId::AIR; BRICK_VOXELS]);
        v.iter_mut().for_each(|c| *c = BlockId(1));
        Brick::from_voxels(v)
    };
    for x in 0..WX {
        for y in 0..3 {
            for z in 0..3 {
                map.insert(IVec3::new(x, y, z), solid());
            }
        }
    }

    let clip_half = 8i32;
    let max_resident = 16384u32;
    let span0 = brick_span(0);

    let dir = std::env::temp_dir().join("vrt_halo_motion");
    std::fs::create_dir_all(&dir).expect("mk tmp dir");
    let path = dir.join("halo_motion.vxo");
    let params = VxoHeadParams { name: "halo_motion".into(), ..Default::default() };
    write_vxo(&path, &map, &reg, &params, VxoCompression::Store).expect("write_vxo");
    let (vxo, vxo_reg) = VxoSource::open(&path).expect("open VxoSource");
    let (merged, _mreg) = MergedSource::new(vec![(vxo, vxo_reg, IVec3::ZERO)]);
    let source = std::sync::Arc::new(merged);

    let mut pager = StreamedResidencyPager::new(&device, &queue, source, 1, clip_half, max_resident);
    let pool = make_pool(&device, max_resident);
    let mut fe = GpuResidencyFrontEnd::new(&device, clip_half, max_resident);

    // LOD-aware ground truth: a lod-L brick C is occupied iff the lod-0 voxel box it covers,
    // `[C·2^L, (C+1)·2^L)` per axis, intersects the wall (x∈[0,WX), y,z∈[0,3)). The clipmap pages coarse LODs for
    // the far field, so a brick MUST be analysed at its own lod (a coarse brick aliases onto a different lod-0 coord).
    let occupied_at_lod = |c: IVec3, l: u32| -> bool {
        let s = 1i32 << l;
        let hit = |ci: i32, lo: i32, hi: i32| ci * s < hi && (ci + 1) * s > lo;
        hit(c.x, 0, WX) && hit(c.y, 0, 3) && hit(c.z, 0, 3)
    };

    let frames = 48usize;
    let mut bound = false;
    let mut total = 0usize; // total spurious-AIR HOLES across all frames (all LODs)
    let mut worst = (0usize, 0usize);
    let mut first = String::new();
    for f in 0..frames {
        let t = f as f32 / (frames - 1) as f32;
        let cx = (1.5 + t * (WX as f32 - 3.0)) * span0; // sweep the camera centre along the wall
        let cam = [cx, 1.5 * span0, 1.5 * span0];

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
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("halo_motion_frame") });
        fe.record_frame(&queue, &mut enc, cam);
        queue.submit(std::iter::once(enc.finish()));
        fe.advance_ring();

        let (n, ex) = count_interior_spurious(&device, &queue, &pool, max_resident, clip_half, cam, &occupied_at_lod);
        total += n;
        if n > worst.1 {
            worst = (f, n);
        }
        if !ex.is_empty() && first.is_empty() {
            first = format!("frame {f} (cam_x={cx:.3}): {ex}");
        }
    }

    // SETTLE — hold the camera still at the sweep end; a still camera MUST converge the halo to zero holes.
    let last_cam = [(1.5 + (WX as f32 - 3.0)) * span0, 1.5 * span0, 1.5 * span0];
    let mut settled = usize::MAX;
    let mut settled_first = String::new();
    for _ in 0..6 {
        pager.update(&queue, last_cam);
        if pager.take_needs_rebind() {
            let occ = pager.occupancy();
            let occ_owned = GpuResidencyBuffers {
                header: occ.header.clone(),
                entries: occ.entries.clone(),
                table_size: occ.table_size,
            };
            let core_owned = pager.core_buffers();
            fe.rebind_pool(&device, &queue, &occ_owned, &core_owned, &pool.meta, &pool.voxel, &pool.palette, &pool.aabb);
        }
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("halo_settle_frame") });
        fe.record_frame(&queue, &mut enc, last_cam);
        queue.submit(std::iter::once(enc.finish()));
        fe.advance_ring();
        let (n, ex) = count_interior_spurious(&device, &queue, &pool, max_resident, clip_half, last_cam, &occupied_at_lod);
        settled = n;
        settled_first = ex;
    }

    eprintln!(
        "[halo-motion] swept {frames} frames; total spurious-AIR holes (all LODs) = {total} (worst frame {} = {}); settled = {settled}",
        worst.0, worst.1
    );
    if !first.is_empty() {
        eprintln!("  first: {first}");
    }
    assert_eq!(
        settled, 0,
        "halo did NOT converge: {settled} spurious-AIR holes remain after the camera held STILL for 6 frames (a true \
         re-pack/leak convergence bug — a resident brick's face toward an occupied neighbour stays AIR); {settled_first}"
    );
    assert_eq!(
        total, 0,
        "halo developed {total} spurious-AIR holes during camera motion across all LODs (phantom exposed faces → wrong \
         normals → black cubes); {first}"
    );
}

/// **Headless OVER-FILL gate — the stuck flat/black-cube reproduction.** A halo neighbour with NO RESIDENT CORE
/// has no geometry the ray can hit, so its face must be packed AIR — whether the occupancy calls it PARTIAL or
/// FULL. The old `NEIGHBOUR_SOLID` guess (gated on `is_occupied`, then narrowed to `is_full`) packed such a face
/// SOLID, burying a face the ray reaches through the empty neighbour space → an all-solid neighbourhood → zero
/// occupancy gradient → a degenerate normal → a STUCK flat/black cube (seen at coarse-LOD coverage gaps where the
/// guessed neighbour never pages in). Uniform-solid scenes can't expose this (every neighbour really IS present),
/// so this crafts the exact case: a PARTIAL **and** a FULL occupied neighbour, BOTH with their core deliberately
/// absent — both halo faces MUST be AIR. Drives the EAGER front-end path with a hand-built occupancy + an
/// INCOMPLETE core store (the coverage gap).
#[test]
fn paged_front_end_halo_partial_neighbour_no_overfill() {
    let Some((device, queue)) = common::headless_compute_device_with_storage(512, 48) else {
        eprintln!("[skip] no GPU adapter — over-fill gate skipped");
        return;
    };

    let a = IVec3::new(0, 0, 0); // the resident SURFACE brick we inspect (fully solid core)
    let px = IVec3::new(1, 0, 0); // +X neighbour: OCCUPIED, NOT full (partial), core ABSENT ⇒ halo must be AIR
    let pz = IVec3::new(0, 0, 1); // +Z neighbour: OCCUPIED, FULL, core ABSENT ⇒ halo must STILL be AIR (no geometry)

    // Occupancy carries the (occupied, is_full) bits. A full; PX occupied-not-full; PZ full. (is_full no longer
    // licenses a SOLID halo guess — only a resident CORE does.)
    let occ = SectorOccupancy::from_occupied_full(vec![(a, 0, true), (px, 0, false), (pz, 0, true)]);
    // The core store DELIBERATELY omits PX's and PZ's cores (the coverage gap): only A's solid core is resident, so
    // `core_lookup(PX)` / `core_lookup(PZ)` return ABSENT ⇒ both halo faces must be AIR (no geometry the ray can hit).
    let a_core = [1u32; BRICK_VOXELS]; // A fully solid
    let cores = BrickCoreStore::from_cores(vec![(a, 0, a_core)]);

    let clip_half = 8i32;
    let max_resident = 4096u32;
    let span0 = brick_span(0);
    let cam = [0.5 * span0, 1.5 * span0, 0.5 * span0]; // near A

    let occ_bufs: GpuResidencyBuffers = occ.upload(&device);
    let core_bufs: GpuBrickCoreBuffers = cores.upload(&device);
    let pool = make_pool(&device, max_resident);
    let mut fe = GpuResidencyFrontEnd::new(&device, clip_half, max_resident);
    fe.rebind_pool(&device, &queue, &occ_bufs, &core_bufs, &pool.meta, &pool.voxel, &pool.palette, &pool.aabb);
    drive_to_convergence(&mut fe, &device, &queue, cam, "overfill");

    // Decode A's haloed meta.
    let metas_raw = read_u32(&device, &queue, &pool.meta, max_resident as usize * META_WORDS);
    let metas: &[GpuBrickMeta] = bytemuck::cast_slice(&metas_raw);
    let voxels = read_u32(&device, &queue, &pool.voxel, (pool.voxel.size() / 4) as usize);
    let palettes = read_u32(&device, &queue, &pool.palette, (pool.palette.size() / 4) as usize);
    let cell_block = |m: &GpuBrickMeta, x: i32, y: i32, z: i32| -> u32 {
        if m.is_uniform() {
            return m.uniform_block().0 as u32;
        }
        let hedge = (BRICK_EDGE as i32) + 2;
        let ci = (x + y * hedge + z * hedge * hedge) as usize;
        let bits = m.index_bits() as u32;
        if bits == 0 {
            return voxels[m.dense_offset() as usize + ci];
        }
        let bit = ci as u32 * bits;
        let word = voxels[m.dense_offset() as usize + (bit / 32) as usize];
        let mask = if bits == 32 { 0xFFFF_FFFF } else { (1u32 << bits) - 1 };
        let local = (word >> (bit % 32)) & mask;
        palettes[m.palette_base as usize + local as usize]
    };
    let zero = GpuBrickMeta::zeroed();
    let am = metas
        .iter()
        .find(|m| **m != zero && !m.is_uniform() && (m.world_min[0] / span0).round() as i32 == 0
            && (m.world_min[1] / span0).round() as i32 == 0
            && (m.world_min[2] / span0).round() as i32 == 0)
        .expect("brick A must be resident & dense");

    // A halo neighbour with NO RESIDENT CORE has no geometry the ray can hit, so its face MUST be AIR — regardless
    // of whether the occupancy thinks it's partial OR full. Packing it SOLID (the old NEIGHBOUR_SOLID/is_full guess)
    // buried a face the ray reaches anyway ⇒ all-solid neighbourhood ⇒ degenerate normal ⇒ a stuck flat/black cube.
    let px_air = (1..=8).all(|y| (1..=8).all(|z| cell_block(am, 9, y, z) == BlockId::AIR.0 as u32)); // PARTIAL nbr
    let pz_air = (1..=8).all(|x| (1..=8).all(|y| cell_block(am, x, y, 9) == BlockId::AIR.0 as u32)); // FULL nbr
    eprintln!("[overfill] A +X(partial,no-core)_all_air={px_air}  +Z(full,no-core)_all_air={pz_air}");
    assert!(
        px_air,
        "A's +X halo toward a PARTIAL occupied neighbour with NO resident core is SOLID, not AIR — a buried face the \
         ray reaches ⇒ degenerate normal ⇒ stuck cube (the NEIGHBOUR_SOLID over-fill)"
    );
    assert!(
        pz_air,
        "A's +Z halo toward a FULL occupied neighbour with NO resident core is SOLID, not AIR — `is_full` is NOT \
         enough: with no resident geometry the ray reaches this brick, so the face must be AIR (exposed), not solid"
    );
}

/// **Headless LOD-TRANSITION leak gate — the stuck-after-LOD-change black cube.** The user's residual cubes:
/// BLACK, STUCK (persist when still), CREATED BY A LOD CHANGE (not present on fresh load), and visible even up
/// close. That signature is a coarse brick that should have been DROPPED when its region refined (coarse→fine as
/// the camera approached) but was left resident — a stale coarse AABB OVERLAPPING the fine bricks. This drives the
/// REAL paged front end along an APPROACH trajectory over a long wall (forcing per-region lod1→lod0 transitions),
/// settles, then asserts NO two LIVE AABBs at DIFFERENT lods overlap in world space (each world point must be
/// covered at exactly one lod). An overlapping pair after settle == the leak the ray sees as a stuck cube.
#[test]
fn paged_front_end_no_stuck_overlap_after_lod_transition() {
    let Some((device, queue)) = common::headless_compute_device_with_storage(512, 48) else {
        eprintln!("[skip] no GPU adapter — lod-transition leak gate skipped");
        return;
    };

    // A long solid wall down +Z (x,y ∈ [0,4), z ∈ [0,ZN)). The camera approaches along +Z, so each region passes
    // through lod2→lod1→lod0 as it nears — the clipmap LOD transitions this gate exists to stress.
    const ZN: i32 = 64;
    let reg = registry();
    let mut map = BrickMap::new();
    let solid = || {
        let mut v = Box::new([BlockId::AIR; BRICK_VOXELS]);
        v.iter_mut().for_each(|c| *c = BlockId(1));
        Brick::from_voxels(v)
    };
    for z in 0..ZN {
        for y in 0..4 {
            for x in 0..4 {
                map.insert(IVec3::new(x, y, z), solid());
            }
        }
    }

    let clip_half = 8i32;
    let max_resident = 32768u32;
    let span0 = brick_span(0);

    let dir = std::env::temp_dir().join("vrt_lod_leak");
    std::fs::create_dir_all(&dir).expect("mk tmp dir");
    let path = dir.join("lod_leak.vxo");
    let params = VxoHeadParams { name: "lod_leak".into(), ..Default::default() };
    write_vxo(&path, &map, &reg, &params, VxoCompression::Store).expect("write_vxo");
    let (vxo, vxo_reg) = VxoSource::open(&path).expect("open VxoSource");
    let (merged, _mreg) = MergedSource::new(vec![(vxo, vxo_reg, IVec3::ZERO)]);
    let source = std::sync::Arc::new(merged);

    let mut pager = StreamedResidencyPager::new(&device, &queue, source, 1, clip_half, max_resident);
    let pool = make_pool(&device, max_resident);
    let mut fe = GpuResidencyFrontEnd::new(&device, clip_half, max_resident);

    // Approach: sweep the camera from z≈-4 (far end coarse) to z≈ZN-4 (near end fine), then a few SETTLE frames.
    let mut bound = false;
    let approach = 56usize;
    let settle = 8usize;
    for step in 0..(approach + settle) {
        let zf = if step < approach {
            (step as f32 / (approach - 1) as f32) * (ZN as f32 - 1.0)
        } else {
            ZN as f32 - 1.0
        };
        let cam = [2.0 * span0, 2.0 * span0, zf * span0];
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
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("lod_leak_frame") });
        fe.record_frame(&queue, &mut enc, cam);
        queue.submit(std::iter::once(enc.finish()));
        fe.advance_ring();
    }

    // Read back metas (for lod) + the LIVE AABBs (what the BLAS/ray actually traces).
    let metas_raw = read_u32(&device, &queue, &pool.meta, max_resident as usize * META_WORDS);
    let metas: &[GpuBrickMeta] = bytemuck::cast_slice(&metas_raw);
    let aabb_raw = read_u32(&device, &queue, &pool.aabb, max_resident as usize * 8);
    let aabbs: &[GpuBrickAabb] = bytemuck::cast_slice(&aabb_raw);
    let zero = GpuBrickMeta::zeroed();

    // Collect (lod, min, max) for every LIVE (non-degenerate) AABB whose slot has a live meta.
    let mut live: Vec<(u32, [f32; 3], [f32; 3])> = Vec::new();
    for (slot, m) in metas.iter().enumerate() {
        if *m == zero {
            continue;
        }
        let a = &aabbs[slot];
        let degen = a.min[0] > a.max[0] || a.min[1] > a.max[1] || a.min[2] > a.max[2];
        if degen {
            continue;
        }
        live.push((m.lod(), a.min, a.max));
    }
    eprintln!("[lod-leak] {} live bricks after approach+settle", live.len());

    // Assert no two LIVE bricks at DIFFERENT lods overlap in world space (a tiny epsilon avoids face-touch false
    // positives — adjacent bricks share a face). A real overlap (interpenetration) == a stuck un-dropped coarse brick.
    let eps = span0 * 0.01;
    let overlaps = |a: &(u32, [f32; 3], [f32; 3]), b: &(u32, [f32; 3], [f32; 3])| -> bool {
        (0..3).all(|i| a.1[i] + eps < b.2[i] && b.1[i] + eps < a.2[i])
    };
    let mut found: Option<String> = None;
    let mut count = 0usize;
    'outer: for i in 0..live.len() {
        for j in (i + 1)..live.len() {
            if live[i].0 != live[j].0 && overlaps(&live[i], &live[j]) {
                count += 1;
                if found.is_none() {
                    found = Some(format!(
                        "lod{} [{:?}..{:?}] OVERLAPS lod{} [{:?}..{:?}]",
                        live[i].0, live[i].1, live[i].2, live[j].0, live[j].1, live[j].2
                    ));
                }
                if count > 200 {
                    break 'outer;
                }
            }
        }
    }
    eprintln!("[lod-leak] cross-lod overlapping live-AABB pairs after settle: {count}");
    if let Some(ex) = &found {
        eprintln!("  e.g. {ex}");
    }
    assert_eq!(
        count, 0,
        "{count} cross-lod AABB overlaps remain after settling the approach — a coarse brick was NOT dropped when its \
         region refined (the stuck-after-LOD-change black cube). e.g. {found:?}"
    );
}

/// **Headless LOD-transition ACCUMULATION gate.** The user reports the stuck cubes get WORSE with EVERY lod
/// transition — an accumulating leak that a single monotonic approach can't surface. This OSCILLATES the camera
/// back and forth along the wall (many lod1↔lod0 transitions per cycle) and, at the SAME reference camera at the
/// end of each cycle, records (live brick count, cross-lod overlap count). Both MUST stay bounded — if either
/// grows cycle over cycle, a drop/free path is leaking slots/AABBs (the worsening cubes).
#[test]
fn paged_front_end_no_leak_growth_across_lod_oscillation() {
    let Some((device, queue)) = common::headless_compute_device_with_storage(512, 48) else {
        eprintln!("[skip] no GPU adapter — lod oscillation gate skipped");
        return;
    };

    const ZN: i32 = 48;
    let reg = registry();
    let mut map = BrickMap::new();
    let solid = || {
        let mut v = Box::new([BlockId::AIR; BRICK_VOXELS]);
        v.iter_mut().for_each(|c| *c = BlockId(1));
        Brick::from_voxels(v)
    };
    for z in 0..ZN {
        for y in 0..4 {
            for x in 0..4 {
                map.insert(IVec3::new(x, y, z), solid());
            }
        }
    }

    let clip_half = 8i32;
    let max_resident = 32768u32;
    let span0 = brick_span(0);

    let dir = std::env::temp_dir().join("vrt_lod_osc");
    std::fs::create_dir_all(&dir).expect("mk tmp dir");
    let path = dir.join("lod_osc.vxo");
    let params = VxoHeadParams { name: "lod_osc".into(), ..Default::default() };
    write_vxo(&path, &map, &reg, &params, VxoCompression::Store).expect("write_vxo");
    let (vxo, vxo_reg) = VxoSource::open(&path).expect("open VxoSource");
    let (merged, _mreg) = MergedSource::new(vec![(vxo, vxo_reg, IVec3::ZERO)]);
    let source = std::sync::Arc::new(merged);

    let mut pager = StreamedResidencyPager::new(&device, &queue, source, 1, clip_half, max_resident);
    let pool = make_pool(&device, max_resident);
    let mut fe = GpuResidencyFrontEnd::new(&device, clip_half, max_resident);
    let mut bound = false;

    let drive = |pager: &mut StreamedResidencyPager, fe: &mut GpuResidencyFrontEnd, bound: &mut bool, cam: [f32; 3]| {
        pager.update(&queue, cam);
        if pager.take_needs_rebind() || !*bound {
            let occ = pager.occupancy();
            let occ_owned = GpuResidencyBuffers {
                header: occ.header.clone(),
                entries: occ.entries.clone(),
                table_size: occ.table_size,
            };
            let core_owned = pager.core_buffers();
            fe.rebind_pool(&device, &queue, &occ_owned, &core_owned, &pool.meta, &pool.voxel, &pool.palette, &pool.aabb);
            *bound = true;
        }
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("lod_osc_frame") });
        fe.record_frame(&queue, &mut enc, cam);
        queue.submit(std::iter::once(enc.finish()));
        fe.advance_ring();
    };

    // Measure (live count, cross-lod overlap count) at the current pool state.
    let measure = |device: &wgpu::Device, queue: &wgpu::Queue| -> (usize, usize) {
        let metas_raw = read_u32(device, queue, &pool.meta, max_resident as usize * META_WORDS);
        let metas: &[GpuBrickMeta] = bytemuck::cast_slice(&metas_raw);
        let aabb_raw = read_u32(device, queue, &pool.aabb, max_resident as usize * 8);
        let aabbs: &[GpuBrickAabb] = bytemuck::cast_slice(&aabb_raw);
        let zero = GpuBrickMeta::zeroed();
        let mut live: Vec<(u32, [f32; 3], [f32; 3])> = Vec::new();
        for (slot, m) in metas.iter().enumerate() {
            if *m == zero {
                continue;
            }
            let a = &aabbs[slot];
            if a.min[0] > a.max[0] || a.min[1] > a.max[1] || a.min[2] > a.max[2] {
                continue;
            }
            live.push((m.lod(), a.min, a.max));
        }
        let eps = span0 * 0.01;
        let mut ov = 0usize;
        for i in 0..live.len() {
            for j in (i + 1)..live.len() {
                if live[i].0 != live[j].0
                    && (0..3).all(|k| live[i].1[k] + eps < live[j].2[k] && live[j].1[k] + eps < live[i].2[k])
                {
                    ov += 1;
                }
            }
        }
        (live.len(), ov)
    };

    let lo = 4.0f32;
    let hi = ZN as f32 - 4.0;
    let mut results = Vec::new();
    for cycle in 0..4 {
        // Forward then back, ~3-brick steps (fast → genuine transitions).
        let mut z = lo;
        while z < hi {
            drive(&mut pager, &mut fe, &mut bound, [2.0 * span0, 2.0 * span0, z * span0]);
            z += 3.0;
        }
        while z > lo {
            drive(&mut pager, &mut fe, &mut bound, [2.0 * span0, 2.0 * span0, z * span0]);
            z -= 3.0;
        }
        // Settle at the reference camera (z = lo) and measure.
        for _ in 0..6 {
            drive(&mut pager, &mut fe, &mut bound, [2.0 * span0, 2.0 * span0, lo * span0]);
        }
        let (live, ov) = measure(&device, &queue);
        eprintln!("[lod-osc] cycle {cycle}: live={live} overlaps={ov}");
        results.push((live, ov));
    }

    // After cycle 0 (warm-up), the reference state must be STABLE — no growth, no overlaps.
    let base_live = results[1].0;
    for (c, (live, ov)) in results.iter().enumerate().skip(1) {
        assert_eq!(*ov, 0, "cycle {c}: {ov} cross-lod overlaps at the reference camera — a LOD-transition leak");
        assert!(
            *live <= base_live + base_live / 20 + 4,
            "cycle {c}: live brick count GREW to {live} (cycle-1 baseline {base_live}) — an accumulating LOD-transition leak (the cubes that get worse each transition)"
        );
    }
}

/// **Headless GPU-COLD-FILL gate (de-risks removing the one-time CPU snapshot).** With the CPU snapshot gone the
/// GPU front end must cold-fill the pool itself from an EMPTY start. This drives the front end over a SOLID 5³-brick
/// VOLUME (a surface SHELL + a fully-buried interior) and asserts: (a) NON-BLANK — the surface shell IS rendered
/// (live AABBs > 0); (b) NO BURIED BRICK IS RENDERED — every LIVE-AABB brick has AIR in its halo (a visible
/// surface), so no all-solid (degenerate-normal / flat-cube) brick is traced. Interior bricks fail
/// `classify_surface` (never entered) and any `classify_surface`-true-but-all-solid straggler gets the `has_air`
/// degenerate AABB — together: a cold-fill that renders only real surfaces, the property the CPU snapshot violated.
#[test]
fn gpu_cold_fill_renders_surface_only_no_buried() {
    let Some((device, queue)) = common::headless_compute_device_with_storage(512, 48) else {
        eprintln!("[skip] no GPU adapter — cold-fill gate skipped");
        return;
    };

    let reg = registry();
    let mut map = BrickMap::new();
    let solid = || {
        let mut v = Box::new([BlockId::AIR; BRICK_VOXELS]);
        v.iter_mut().for_each(|c| *c = BlockId(1));
        Brick::from_voxels(v)
    };
    for z in 0..5 {
        for y in 0..5 {
            for x in 0..5 {
                map.insert(IVec3::new(x, y, z), solid());
            }
        }
    }

    let clip_half = 8i32;
    let max_resident = 16384u32;
    let span0 = brick_span(0);
    let cam = [2.5 * span0, 2.5 * span0, -6.0 * span0]; // outside the volume, looking at the -Z face

    // EAGER cold-fill (the front-end pack path, empty start) — the SAME path the live front end runs from an empty
    // (no-CPU-snapshot) pool. All occupied cores resident, so a buried interior brick's neighbours are all present.
    let static_src = StaticVoxSource::new(&map);
    let occ = SectorOccupancy::from_occupied_full(static_src.occupied_keys_full());
    let cores = BrickCoreStore::from_cores(
        static_src.occupied_keys().map(|(c, l)| (c, l, eager_core(&static_src, &reg, c, l))),
    );
    let occ_bufs: GpuResidencyBuffers = occ.upload(&device);
    let core_bufs: GpuBrickCoreBuffers = cores.upload(&device);
    let pool = make_pool(&device, max_resident);
    let mut fe = GpuResidencyFrontEnd::new(&device, clip_half, max_resident);
    fe.rebind_pool(&device, &queue, &occ_bufs, &core_bufs, &pool.meta, &pool.voxel, &pool.palette, &pool.aabb);
    drive_to_convergence(&mut fe, &device, &queue, cam, "cold-fill");

    let metas_raw = read_u32(&device, &queue, &pool.meta, max_resident as usize * META_WORDS);
    let metas: &[GpuBrickMeta] = bytemuck::cast_slice(&metas_raw);
    let aabb_raw = read_u32(&device, &queue, &pool.aabb, max_resident as usize * 8);
    let aabbs: &[GpuBrickAabb] = bytemuck::cast_slice(&aabb_raw);
    let voxels = read_u32(&device, &queue, &pool.voxel, (pool.voxel.size() / 4) as usize);
    let palettes = read_u32(&device, &queue, &pool.palette, (pool.palette.size() / 4) as usize);
    let cell_block = |m: &GpuBrickMeta, x: i32, y: i32, z: i32| -> u32 {
        if m.is_uniform() {
            return m.uniform_block().0 as u32;
        }
        let hedge = (BRICK_EDGE as i32) + 2;
        let ci = (x + y * hedge + z * hedge * hedge) as usize;
        let bits = m.index_bits() as u32;
        if bits == 0 {
            return voxels[m.dense_offset() as usize + ci];
        }
        let bit = ci as u32 * bits;
        let word = voxels[m.dense_offset() as usize + (bit / 32) as usize];
        let mask = if bits == 32 { 0xFFFF_FFFF } else { (1u32 << bits) - 1 };
        let local = (word >> (bit % 32)) & mask;
        palettes[m.palette_base as usize + local as usize]
    };
    let zero = GpuBrickMeta::zeroed();
    let hedge = (BRICK_EDGE as i32) + 2;

    let mut live_rendered = 0usize;
    let mut buried_rendered = 0usize; // live-AABB brick with an all-solid halo (a flat/degenerate cube) — must be 0
    let mut first_buried = String::new();
    for (slot, m) in metas.iter().enumerate() {
        if *m == zero {
            continue;
        }
        let a = &aabbs[slot];
        let aabb_live = a.min[0] <= a.max[0] && a.min[1] <= a.max[1] && a.min[2] <= a.max[2];
        if !aabb_live {
            continue; // not rendered (a buried brick correctly given a degenerate AABB, or a free slot)
        }
        live_rendered += 1;
        if m.is_uniform() {
            // a uniform (all-one-solid) brick is buried by definition — it must NOT be rendered
            buried_rendered += 1;
            if first_buried.is_empty() {
                first_buried = format!("slot {slot} UNIFORM rendered");
            }
            continue;
        }
        let any_air = (0..hedge).any(|z| (0..hedge).any(|y| (0..hedge).any(|x| cell_block(m, x, y, z) == 0)));
        if !any_air {
            buried_rendered += 1;
            if first_buried.is_empty() {
                first_buried = format!("slot {slot} ALL-SOLID halo rendered (live AABB)");
            }
        }
    }
    eprintln!("[cold-fill] live-rendered bricks={live_rendered}, buried-rendered (must be 0)={buried_rendered}");
    assert!(live_rendered > 0, "cold-fill produced a BLANK pool (no rendered surface bricks)");
    assert_eq!(
        buried_rendered, 0,
        "{buried_rendered} BURIED (all-solid-halo) bricks are RENDERED after a GPU cold-fill — they trace as \
         flat/degenerate cubes; {first_buried}"
    );
}

/// **Headless RICH-PALETTE pool-capacity gate — the index_bits=8 garbage-content cube.** The user's cubes show
/// (NORMALS view) "garbage jumbled colours" on coarse bricks during fast motion + LOD shifts; the F9 content-vs-
/// SOURCE check pinned the corrupt bricks to `index_bits=8` (large per-brick palettes). Root cause: the per-brick
/// PALETTE slab pool is reserved at the MEAN `RESERVE_PALETTE_WORDS_PER_BRICK` (16 words/slot) but an `index_bits=8`
/// brick needs up to 256 palette words — once enough rich-palette bricks are concurrently resident, the GPU palette
/// BUMP allocator (`alloc_palette_slab`, NO capacity guard) runs the high-water PAST the pool end → OOB / clamped /
/// overlapping palette slabs → bricks decode through a WRONG palette → garbage content. Uniform-solid scenes (every
/// other residency test) have a 1-entry palette so they never approach the bound; only a heterogeneous scene exposes
/// it. This builds a SOLID block whose every brick has ~250 distinct ids (forcing `index_bits=8`, ~250 palette words
/// each), packs WAY more rich bricks than the 16-word/slot pool can hold, and asserts every resident brick's GPU
/// content matches the SOURCE (the exact F9 content-integrity check, headless + deterministic).
#[test]
fn rich_palette_pool_no_content_corruption() {
    let Some((device, queue)) = common::headless_compute_device_with_storage(512, 48) else {
        eprintln!("[skip] no GPU adapter — rich-palette pool gate skipped");
        return;
    };

    // A 250-colour registry (BlockId 1..=250) so the per-brick palettes are genuinely large.
    let colors: Vec<[u8; 4]> = (0..250)
        .map(|i| [(i * 7 % 256) as u8, (i * 13 % 256) as u8, (i * 29 % 256) as u8, 255])
        .collect();
    let reg = BlockRegistry::from_vox_palette(&colors);

    // Every voxel a distinct-ish id so each 8³ brick carries ~250 distinct ids ⇒ index_bits=8 (palette ≤256, but
    // FAR above the 16-word mean reserve). A 3D GRID of ISOLATED solid bricks (spacing 2 ⇒ every brick has all 6
    // faces exposed to AIR ⇒ EVERY brick is a surface brick that ENTERS residency): 8³ = 512 rich bricks all live
    // at lod0, needing 512·250 ≈ 128k palette words — MASSIVELY over the 4096·16 = 65 536-word pool ⇒ guaranteed
    // overflow on the buggy path. (The earlier solid-block scene only exposed one face to the camera → too few.)
    const N: i32 = 8;
    let mut map = BrickMap::new();
    for bz in 0..N {
        for by in 0..N {
            for bx in 0..N {
                let mut v = Box::new([BlockId::AIR; BRICK_VOXELS]);
                for lz in 0..BRICK_EDGE {
                    for ly in 0..BRICK_EDGE {
                        for lx in 0..BRICK_EDGE {
                            let local = lx + ly * BRICK_EDGE + lz * BRICK_EDGE * BRICK_EDGE; // 0..512
                            let id = 1 + (local % 250) as u16; // 1..=250, ~250 distinct per brick
                            v[voxel_index(lx, ly, lz)] = BlockId(id);
                        }
                    }
                }
                map.insert(IVec3::new(bx * 2, by * 2, bz * 2), Brick::from_voxels(v)); // spacing 2 ⇒ isolated
            }
        }
    }

    let clip_half = 8i32;
    let max_resident = 4096u32; // pool palette cap = 4096·16 = 65 536 words ≪ what 512 rich bricks need (≈128k)
    let span0 = brick_span(0);
    let cam = [(N as f32) * span0, (N as f32) * span0, (N as f32) * span0]; // at the grid centre ⇒ all in lod0 window

    let static_src = StaticVoxSource::new(&map);
    let occ = SectorOccupancy::from_occupied_full(static_src.occupied_keys_full());
    let cores = BrickCoreStore::from_cores(
        static_src.occupied_keys().map(|(c, l)| (c, l, eager_core(&static_src, &reg, c, l))),
    );
    let occ_bufs: GpuResidencyBuffers = occ.upload(&device);
    let core_bufs: GpuBrickCoreBuffers = cores.upload(&device);
    let pool = make_pool(&device, max_resident);
    let mut fe = GpuResidencyFrontEnd::new(&device, clip_half, max_resident);
    fe.rebind_pool(&device, &queue, &occ_bufs, &core_bufs, &pool.meta, &pool.voxel, &pool.palette, &pool.aabb);
    drive_to_convergence(&mut fe, &device, &queue, cam, "rich-palette");

    let metas_raw = read_u32(&device, &queue, &pool.meta, max_resident as usize * META_WORDS);
    let metas: &[GpuBrickMeta] = bytemuck::cast_slice(&metas_raw);
    let voxels = read_u32(&device, &queue, &pool.voxel, (pool.voxel.size() / 4) as usize);
    let palettes = read_u32(&device, &queue, &pool.palette, (pool.palette.size() / 4) as usize);
    let cell_block = |m: &GpuBrickMeta, x: i32, y: i32, z: i32| -> u32 {
        if m.is_uniform() {
            return m.uniform_block().0 as u32;
        }
        let hedge = (BRICK_EDGE as i32) + 2;
        let ci = (x + y * hedge + z * hedge * hedge) as usize;
        let bits = m.index_bits() as u32;
        if bits == 0 {
            return voxels[m.dense_offset() as usize + ci];
        }
        let bit = ci as u32 * bits;
        let word = voxels[m.dense_offset() as usize + (bit / 32) as usize];
        let mask = if bits == 32 { 0xFFFF_FFFF } else { (1u32 << bits) - 1 };
        let local = (word >> (bit % 32)) & mask;
        palettes[m.palette_base as usize + local as usize]
    };
    let zero = GpuBrickMeta::zeroed();
    let palette_pool_words = palettes.len();

    // FIRST: no resident dense brick's palette slab may extend PAST the pool — that IS the overflow bug (the GPU
    // bump allocator handed out a `palette_base` beyond the buffer ⇒ OOB/overlapping slabs ⇒ garbage content).
    let mut max_palette_end = 0usize;
    for m in metas.iter() {
        if *m == zero || m.is_uniform() {
            continue;
        }
        // palette slab size = 2^(class+1) ≥ k; bound it by the index_bits class ceiling (≤256 for ≤8, ≤65536 for 16).
        let cap = if m.index_bits() <= 8 { 256 } else { 65536 };
        max_palette_end = max_palette_end.max(m.palette_base as usize + cap.min(1024));
    }
    assert!(
        max_palette_end <= palette_pool_words,
        "a resident brick's palette slab extends to word {max_palette_end} but the palette pool is only \
         {palette_pool_words} words — the palette bump allocator OVERFLOWED the pool (mean 16-word/slot reserve, no \
         capacity guard) ⇒ overlapping/OOB palette slabs ⇒ garbage cubes"
    );

    // For each resident DENSE lod-0 brick, compare its GPU core (haloed cells 1..=8) to the SOURCE core.
    let mut checked = 0usize;
    let mut mismatch_bricks = 0usize;
    let mut max_index_bits = 0u32;
    let mut first = String::new();
    for m in metas.iter() {
        if *m == zero || m.is_uniform() || m.lod() != 0 {
            continue;
        }
        max_index_bits = max_index_bits.max(m.index_bits() as u32);
        let coord = IVec3::new(
            (m.world_min[0] / span0).round() as i32,
            (m.world_min[1] / span0).round() as i32,
            (m.world_min[2] / span0).round() as i32,
        );
        let src = eager_core(&static_src, &reg, coord, 0);
        let mut diff = 0usize;
        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                for x in 0..BRICK_EDGE {
                    let gpu = cell_block(m, x as i32 + 1, y as i32 + 1, z as i32 + 1);
                    if gpu != src[voxel_index(x, y, z)] {
                        diff += 1;
                    }
                }
            }
        }
        checked += 1;
        if diff > 0 {
            mismatch_bricks += 1;
            if first.is_empty() {
                first = format!("brick {coord:?} index_bits={}: {diff}/{BRICK_VOXELS} voxels differ", m.index_bits());
            }
        }
    }
    eprintln!(
        "[rich-palette] checked={checked} lod0 dense bricks (max index_bits={max_index_bits}); content-vs-source \
         mismatched bricks = {mismatch_bricks}"
    );
    assert!(checked > 256, "expected >256 resident rich bricks to overflow the 16-word/slot palette pool, got {checked}");
    assert_eq!(max_index_bits, 8, "scene must produce index_bits=8 bricks to exercise large palettes (got {max_index_bits})");
    assert_eq!(
        mismatch_bricks, 0,
        "{mismatch_bricks} resident bricks decode to WRONG content vs the source — the palette slab pool overflowed \
         (mean 16-word/slot reserve, no capacity guard, index_bits=8 needs ≤256) ⇒ overlapping/OOB palette slabs ⇒ \
         garbage cubes; first: {first}"
    );
}
