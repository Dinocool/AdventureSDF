//! **Phase G "G-c.4-paging" — the demand-paged GPU core-store free-list UNIT gate** (`docs/PHASE_G_GC_PLAN.md`
//! §8.3, §8.4 gate 1).
//!
//! Proves the [`PagedBrickCoreStore`] free-list is correct under insert / evict / re-insert:
//!  * **no leak / no double-free** — the free-core count tracks `cap - resident` exactly across page-in/page-out;
//!  * **evicted keys absent** — a dropped region's keys read absent (`contains == false`) AND probe-miss on the
//!    GPU `core_table` (the WGSL `core_lookup` semantics: stop at EMPTY, skip TOMBSTONE);
//!  * **coverage holds** — a SURVIVING region's keys stay present after a neighbour region is evicted (the
//!    open-addressing deletion never truncates another key's probe chain — the tombstone fix);
//!  * **refcount** — a brick paged by two regions survives the first evictor, freed by the last;
//!  * **GPU == CPU** — the read-back `core_table` slots + `cores` blocks match the CPU mirror exactly.
//!
//! Skips cleanly when no GPU adapter is available.

use adventure::voxel::brickmap::BRICK_VOXELS;
use adventure::voxel::residency_gpu::{PagedBrickCoreStore, brick_key_hash};
use bevy::math::IVec3;

#[path = "common/mod.rs"]
mod common;

const EMPTY_LOD: u32 = 0xFFFF_FFFF;
const TOMBSTONE_LOD: u32 = EMPTY_LOD - 1;

/// Read back a storage buffer to a `Vec<u32>` (blocking — a test helper).
fn read_u32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, words: usize) -> Vec<u32> {
    let size = (words * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paged_core_readback"),
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

/// The WGSL `core_lookup` probe, run on the read-back GPU table: returns the core_index for `(coord,lod)`, or
/// `None` (probe hit EMPTY before a match). MIRRORS the shader EXACTLY (stop at EMPTY_LOD, skip everything else
/// incl. TOMBSTONE) so this is the true GPU-side membership the front end would observe.
fn gpu_lookup(table: &[u32], table_size: u32, coord: IVec3, lod: u32) -> Option<u32> {
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

/// A core whose every voxel = `fill` (so a read-back can confirm the right core landed in the right slot).
fn core_filled(fill: u32) -> [u32; BRICK_VOXELS] {
    [fill; BRICK_VOXELS]
}

#[test]
fn paged_core_store_insert_evict_no_leak_coverage_holds() {
    let Some((device, queue)) = common::headless_device(wgpu::Features::empty()) else {
        eprintln!("paged_core_store: no GPU adapter — skipping");
        return;
    };

    // cap deliberately tight so the free-list / reuse is exercised (no slack to hide a leak).
    let cap = 64u32;
    let mut store = PagedBrickCoreStore::new(&device, &queue, cap);
    let table_size = store.table_size();

    // Region A: 5 distinct bricks @ fill 100.., region B: 4 distinct bricks @ fill 200.. — DISJOINT keys.
    let region_a = (0usize, 0u32, IVec3::new(0, 0, 0));
    let region_b = (0usize, 0u32, IVec3::new(1, 0, 0));
    let bricks_a: Vec<(IVec3, u32, [u32; BRICK_VOXELS])> = (0..5)
        .map(|i| (IVec3::new(i, 0, 0), 0u32, core_filled(100 + i as u32)))
        .collect();
    let bricks_b: Vec<(IVec3, u32, [u32; BRICK_VOXELS])> = (0..4)
        .map(|i| (IVec3::new(8 + i, 0, 0), 0u32, core_filled(200 + i as u32)))
        .collect();

    store.upload_region(&queue, region_a, &bricks_a);
    store.upload_region(&queue, region_b, &bricks_b);
    assert_eq!(store.resident_cores(), 9, "9 distinct bricks paged");
    assert_eq!(store.resident_region_count(), 2);

    // Every key present (CPU + GPU). The GPU core_index resolves to the right fill in `cores`.
    let cores = read_u32(&device, &queue, &store.buffers().cores, cap as usize * BRICK_VOXELS);
    let table = read_u32(&device, &queue, &store.buffers().table, table_size as usize * 5);
    for (coord, lod, core) in bricks_a.iter().chain(bricks_b.iter()) {
        assert!(store.contains(*coord, *lod), "{coord:?}@{lod} should be resident");
        let idx = gpu_lookup(&table, table_size, *coord, *lod).expect("GPU lookup hit");
        assert_eq!(cores[idx as usize * BRICK_VOXELS], core[0], "core content @ {coord:?}");
    }

    // Evict A. A's keys absent (CPU + GPU); B's keys STILL present (coverage / probe-chain intact).
    store.evict_region(&queue, region_a);
    assert_eq!(store.resident_cores(), 4, "only B's 4 bricks remain");
    assert_eq!(store.resident_region_count(), 1);
    let table = read_u32(&device, &queue, &store.buffers().table, table_size as usize * 5);
    for (coord, lod, _) in &bricks_a {
        assert!(!store.contains(*coord, *lod), "{coord:?}@{lod} evicted ⇒ absent (CPU)");
        assert!(gpu_lookup(&table, table_size, *coord, *lod).is_none(), "{coord:?}@{lod} evicted ⇒ GPU miss");
    }
    for (coord, lod, _) in &bricks_b {
        assert!(store.contains(*coord, *lod), "{coord:?}@{lod} survives A's eviction (COVERAGE)");
        assert!(gpu_lookup(&table, table_size, *coord, *lod).is_some(), "{coord:?}@{lod} survives on GPU");
    }
    // The freed slots are tombstoned, not EMPTY (probe-chain preservation) — count tombstones == A's bricks.
    let tombstones = table.chunks_exact(5).filter(|s| s[3] == TOMBSTONE_LOD).count();
    assert_eq!(tombstones, 5, "A's 5 slots tombstoned (not EMPTY)");

    // Re-page A (free-list REUSE — no leak): resident back to 9, no panic (slots reused from the freed pool).
    store.upload_region(&queue, region_a, &bricks_a);
    assert_eq!(store.resident_cores(), 9, "A re-paged via the free-list (no leak)");
    let table = read_u32(&device, &queue, &store.buffers().table, table_size as usize * 5);
    let cores = read_u32(&device, &queue, &store.buffers().cores, cap as usize * BRICK_VOXELS);
    for (coord, lod, core) in bricks_a.iter().chain(bricks_b.iter()) {
        let idx = gpu_lookup(&table, table_size, *coord, *lod).expect("re-paged GPU hit");
        assert_eq!(cores[idx as usize * BRICK_VOXELS], core[0], "re-paged core content @ {coord:?}");
    }

    // Drop everything — every brick freed (no leak: a fresh full upload after this must not panic).
    store.evict_region(&queue, region_a);
    store.evict_region(&queue, region_b);
    assert_eq!(store.resident_cores(), 0, "all freed");
    assert_eq!(store.resident_region_count(), 0);
    // Fill to capacity to prove all `cap` slots are free again (no leaked slots): cap distinct bricks in one region.
    let big: Vec<(IVec3, u32, [u32; BRICK_VOXELS])> =
        (0..cap as i32).map(|i| (IVec3::new(i, 5, 5), 0u32, core_filled(i as u32))).collect();
    store.upload_region(&queue, (0, 0, IVec3::new(9, 9, 9)), &big);
    assert_eq!(store.resident_cores(), cap as usize, "filled to capacity — all slots were free (no leak)");
}

#[test]
fn paged_core_store_refcount_shared_brick() {
    let Some((device, queue)) = common::headless_device(wgpu::Features::empty()) else {
        eprintln!("paged_core_store(refcount): no GPU adapter — skipping");
        return;
    };
    let mut store = PagedBrickCoreStore::new(&device, &queue, 16);

    // The SAME brick paged by two regions (the +1-halo pad can re-page a neighbour region holding the same brick).
    let shared = (IVec3::new(3, 3, 3), 0u32, core_filled(42));
    let only_a = (IVec3::new(0, 0, 0), 0u32, core_filled(1));
    let only_b = (IVec3::new(9, 9, 9), 0u32, core_filled(2));
    let ra = (0usize, 0u32, IVec3::new(0, 0, 0));
    let rb = (0usize, 0u32, IVec3::new(1, 1, 1));

    store.upload_region(&queue, ra, &[only_a, shared]);
    store.upload_region(&queue, rb, &[only_b, shared]);
    // shared is ONE core (deduped), refcount 2; total distinct = 3.
    assert_eq!(store.resident_cores(), 3, "shared brick deduped to one core");

    // Evict A: shared SURVIVES (still referenced by B); only_a gone.
    store.evict_region(&queue, ra);
    assert!(store.contains(shared.0, shared.1), "shared survives A (refcount 2->1)");
    assert!(!store.contains(only_a.0, only_a.1), "A-only brick freed");
    assert_eq!(store.resident_cores(), 2);

    // Evict B: shared now freed (refcount 1->0).
    store.evict_region(&queue, rb);
    assert!(!store.contains(shared.0, shared.1), "shared freed by the last evictor");
    assert_eq!(store.resident_cores(), 0);
}
