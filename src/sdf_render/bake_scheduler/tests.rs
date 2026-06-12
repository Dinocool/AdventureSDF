//! Bake-scheduler tests, moved out of `mod.rs` (now a directory) to keep the production module
//! readable — the in-file test mod was ~1.4k lines, 54% of the file. Same `use super::*` access
//! as when it was inline. Covers the recenter lifecycle, the sync per-frame bake, the pure
//! window/cull/classify units, and the CPU↔GPU parity mirrors.

use super::*;
use crate::sdf_render::edits::{edit_world_aabb, CsgKind, ResolvedEdit};
use bevy::math::bounding::Aabb3d;
use std::collections::HashSet;

fn config() -> SdfGridConfig {
    // Pin the GPU brick-scheduler tests to a FIXED 0.1 m voxel, decoupled from the render default
    // (which we tune freely — currently 0.4 m). These tests validate scheduler residency/spill/flush
    // logic, whose expected brick counts depend on how many bricks a fixed-size test object spans —
    // a property of the voxel scale. Pinning here keeps the whole suite stable across voxel_size tuning
    // (the same reason the deep-interior cull test pins 0.1 explicitly).
    SdfGridConfig { voxel_size: 0.1, ..SdfGridConfig::default() }
}

/// Dirty an ENTIRE chunk in `pending` (all 64 bricks) — the whole-chunk dirty the unit tests used
/// before production moved to brick-level masking. Centralizes the `FULL_CHUNK_MASK` insert so the
/// settle/cull tests keep dirtying known regions wholesale (the brick-mask precision is exercised by
/// the dedicated mask tests + the perf harness).
fn dirty_chunk(sched: &mut BakeScheduler, ck: chunk::ChunkKey) {
    dirty_mask(&mut sched.pending, ck, FULL_CHUNK_MASK);
}

// --- GPU recenter convergence harness -------------------------------------------
//
// Drives the real recenter (step 2 of `schedule_bakes`) + `emit_gpu_bakes` directly on a
// scheduler/atlas pair, so the resident-set convergence invariants are tested against the
// exact production topology code (no ECS App needed). The GPU bake emits synchronously, so
// there's no async lag to model — what's dirtied this frame is baked this frame.

/// Recenter to `cam` and drain the GPU bake emission until idle (the cap may spill over
/// several frames). Returns the resident chunk set — the GPU equivalent of a fresh settle.
fn settle_gpu(sched: &mut BakeScheduler, atlas: &mut SdfAtlas, cfg: &SdfGridConfig, cam: Vec3) -> HashSet<chunk::ChunkKey> {
    recenter_step(sched, atlas, cfg, cam);
    let mut gpu = PendingGpuBakes::default();
    let mut guard = 0;
    loop {
        gpu.jobs.clear();
        gpu.edits.clear();
        atlas.gpu_baked_tiles.clear();
        emit_gpu_bakes(atlas, sched, &mut gpu, cfg, cam, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
        guard += 1;
        assert!(guard < 1000, "settle did not converge");
        if sched.pending.is_empty() && sched.ready.is_empty() {
            break;
        }
    }
    atlas.bricks.keys().map(|k| chunk::chunk_of(*k, cfg).0).collect()
}

/// Apply the live table's per-frame delta to a GPU-buffer mirror EXACTLY as `render.rs` does, by
/// routing through the SAME [`chunk::LiveChunkTables::upload`] accessor that owns the
/// rebuild-vs-delta + headroom policy: a Full re-sizes the buffers once, a Delta writes the dirty
/// directory slots + tile-run regions in place (no row shift, no sentinel tail).
fn apply_table_delta(
    live: &chunk::LiveChunkTables,
    rows: &mut Vec<chunk::ChunkLookup>,
    tiles: &mut Vec<chunk::BrickTile>,
    cap_rows: &mut u32,
    cap_slots: &mut u32,
) {
    match live.upload(*cap_rows, *cap_slots) {
        chunk::ChunkUpload::Full { rows: r, tile_run, cap_rows: cr, cap_slots: cs } => {
            *cap_rows = cr;
            *cap_slots = cs;
            *rows = r;
            *tiles = tile_run;
            tiles.resize(*cap_slots as usize, chunk::BrickTile::default());
        }
        chunk::ChunkUpload::TileGrow { row_updates, tile_run, cap_slots: cs } => {
            *cap_slots = cs;
            for (row, look) in row_updates {
                rows[row as usize] = look; // directory delta (size unchanged)
            }
            *tiles = tile_run; // tile-run rebuild
            tiles.resize(*cap_slots as usize, chunk::BrickTile::default());
        }
        chunk::ChunkUpload::Delta { row_updates, region_updates } => {
            for (row, look) in row_updates {
                rows[row as usize] = look;
            }
            for (slot, region) in region_updates {
                let base = (slot * chunk::TILE_RUN_SLOT) as usize;
                tiles[base..base + chunk::TILE_RUN_SLOT as usize].copy_from_slice(&region);
            }
        }
    }
}

/// No-deferral handoff (replaces the old make-before-break gate): when the camera moves so a fine
/// (LOD-0) chunk leaves its ring, its bricks are evicted IMMEDIATELY — but a COARSER LOD over the
/// same point stays resident, so `resolve_march` falls back to it (a brief LOD pop, never a hole).
/// This is the invariant that lets us drop the whole deferral machinery.
#[test]
fn exited_fine_chunk_evicts_while_coarser_stays_resident() {
    let cfg = SdfGridConfig { lod_count: 5, ring_bricks: 16, recenter_snap_chunks: 1, ..config() };
    let edits = vec![sphere_edit(Vec3::ZERO, 5.0, 0)];
    let mut sched = primed_sched(&edits);
    let mut atlas = SdfAtlas::default();

    let p = Vec3::new(5.0, 0.0, 0.0); // on the sphere surface
    let fine_key = atlas::BrickKey::new(0, cfg.world_to_brick_lod(p, 0));
    let coarser_resident = |a: &SdfAtlas| {
        (1..cfg.lod_count)
            .any(|lod| a.bricks.contains_key(&atlas::BrickKey::new(lod, cfg.world_to_brick_lod(p, lod))))
    };

    // Settle with the camera near the surface: the LOD-0 brick at `p` is resident.
    settle_gpu(&mut sched, &mut atlas, &cfg, Vec3::ZERO);
    assert!(atlas.bricks.contains_key(&fine_key), "fine brick must be resident before the move");

    // Move far enough that `p` leaves the LOD-0 ring but stays deep inside the coarser rings.
    settle_gpu(&mut sched, &mut atlas, &cfg, Vec3::new(40.0, 0.0, 0.0));

    assert!(
        !atlas.bricks.contains_key(&fine_key),
        "exited fine chunk's brick must be evicted immediately (no deferral)"
    );
    assert!(
        coarser_resident(&atlas),
        "a coarser LOD must stay resident so the march falls back to it (a pop, not a hole)"
    );
}

/// THE garbled-LOD-transition guard: drive the REAL recenter + emit lifecycle (fly a camera out
/// across the clipmap LOD bands of a big sphere and back), draining the bake FRAME BY FRAME, and
/// after every frame assert the incrementally-maintained chunk table — BOTH `full_tables()` and
/// the render.rs delta-upload MIRROR — resolves every resident brick to its CORRECT atlas tile.
/// A desync here (brick → wrong/absent tile) is exactly the "garbled geometry during a LOD
/// handoff": the shader reads another brick's texels (wrong shape) or an unbaked tile.
#[test]
fn live_table_resolves_correct_tile_through_recenter_lifecycle() {
    let cfg = SdfGridConfig { lod_count: 5, ring_bricks: 16, recenter_snap_chunks: 1, ..config() };
    let edits = vec![sphere_edit(Vec3::ZERO, 25.0, 0)];
    let mut sched = primed_sched(&edits);
    let mut atlas = SdfAtlas::default();
    let mut gpu = PendingGpuBakes::default();

    let mut rows: Vec<chunk::ChunkLookup> = Vec::new();
    let mut tiles: Vec<chunk::BrickTile> = Vec::new();
    let mut cap_rows = 0u32;
    let mut cap_slots = 0u32;

    // Fly out (0..72) and back — several LOD bands transition each way.
    let mut path: Vec<f32> = (0..=24).map(|i| i as f32 * 3.0).collect();
    let back: Vec<f32> = path.iter().rev().skip(1).copied().collect();
    path.extend(back);

    let mut dbg = crate::sdf_render::BakedBrickDebug::default();
    for (step, &x) in path.iter().enumerate() {
        let cam = Vec3::new(x, 0.0, 0.0);
        recenter_step(&mut sched, &mut atlas, &cfg, cam);
        let mut guard = 0;
        loop {
            gpu.jobs.clear();
            gpu.edits.clear();
            atlas.gpu_baked_tiles.clear();
            emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, cam, &mut dbg, 0.0);

            // Mirror the render world: apply this frame's table delta, then clear the dirty record.
            apply_table_delta(&atlas.live_chunks, &mut rows, &mut tiles, &mut cap_rows, &mut cap_slots);
            let (fr, ft) = atlas.live_chunks.full_tables();
            atlas.live_chunks.clear_dirty();

            // Every resident brick must resolve to its OWN tile, through both table views.
            for key in atlas.bricks.keys() {
                let tile = atlas.tiles.tile(key).expect("resident brick has a tile");
                let want = chunk::tile_atlas_base(tile);
                let (ck, local) = chunk::chunk_of(*key, &cfg);
                assert_eq!(
                    chunk::resolve_via_tables(&fr, &ft, cfg.ring_chunks_per_axis(), ck, local).map(|t| t.atlas_base),
                    Some(want),
                    "step {step}: full_tables resolves brick {key:?} to the wrong/absent tile"
                );
                assert_eq!(
                    chunk::resolve_via_tables(&rows, &tiles, cfg.ring_chunks_per_axis(), ck, local).map(|t| t.atlas_base),
                    Some(want),
                    "step {step}: delta-mirror resolves brick {key:?} to the wrong/absent tile"
                );
            }

            guard += 1;
            assert!(guard < 4000, "drain did not converge at step {step}");
            if sched.pending.is_empty() && sched.ready.is_empty() {
                break;
            }
        }
    }
}

fn box_edit(pos: Vec3, half: f32, mat: u16) -> ResolvedEdit {
    ResolvedEdit::new(
        SdfPrimitive::Box { half_extents: Vec3::splat(half) },
        Transform::from_translation(pos),
        SdfOp { kind: CsgKind::Union, smoothing: 0.0 },
        mat,
    )
}

fn sphere_edit(pos: Vec3, radius: f32, mat: u16) -> ResolvedEdit {
    ResolvedEdit::new(
        SdfPrimitive::Sphere { radius },
        Transform::from_translation(pos),
        SdfOp { kind: CsgKind::Union, smoothing: 0.0 },
        mat,
    )
}

fn subtract_sphere(pos: Vec3, radius: f32) -> ResolvedEdit {
    ResolvedEdit::new(
        SdfPrimitive::Sphere { radius },
        Transform::from_translation(pos),
        SdfOp { kind: CsgKind::Subtract, smoothing: 0.0 },
        0,
    )
}

/// Regression guard for the narrow-band cull on a SUBTRACTED (hollow / bitten) solid: the
/// cull must not drop any brick the TRUE folded surface passes through, and every resident
/// brick's per-brick CULLED candidate set must agree in SIGN with the full edit list at the
/// brick corners (so the GPU bakes the carve, not solid). Covers both an enclosed cavity and
/// an open bite. (Proven the cull is innocent of the interior-hole artefact — that lives in
/// the GPU bake/march, not here.)
#[test]
fn cull_preserves_subtracted_surface_bricks() {
    for (r_in, off) in [(4.0_f32, 0.0_f32), (5.0, 10.0)] {
        let cfg = SdfGridConfig { lod_count: 1, ring_bricks: 8, recenter_snap_chunks: 1, ..config() };
        let r_out = 10.0;
        let edits = vec![
            sphere_edit(Vec3::ZERO, r_out, 0),
            subtract_sphere(Vec3::new(off, 0.0, 0.0), r_in),
        ];
        let mut sched = primed_sched(&edits);
        let mut atlas = SdfAtlas::default();
        for x in -5..=5 {
            for y in -5..=5 {
                for z in -5..=5 {
                    dirty_chunk(&mut sched, chunk::ChunkKey::new(0, IVec3::new(x, y, z)));
                }
            }
        }
        let mut gpu = PendingGpuBakes::default();
        let mut guard = 0;
        loop {
            gpu.clear();
            atlas.gpu_baked_tiles.clear();
            emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
            guard += 1;
            assert!(guard < 1000);
            if sched.pending.is_empty() && sched.ready.is_empty() { break; }
        }

        let all_edits = sched.edits.clone();
        let bw = cfg.brick_world_size(0);
        let mut scratch: Vec<u32> = Vec::new();
        let corner = |bmin: Vec3| {
            let mut cs = [Vec3::ZERO; 8];
            let mut i = 0;
            for cx in [0.0, bw] {
                for cy in [0.0, bw] {
                    for cz in [0.0, bw] {
                        cs[i] = bmin + Vec3::new(cx, cy, cz);
                        i += 1;
                    }
                }
            }
            cs
        };

        // (1) No surface-bearing brick dropped.
        for ck in chunk_window_keys(IVec3::splat(-5), 11, 0) {
            for key in chunk_brick_keys(ck, &cfg) {
                if atlas::SdfAtlas::cull_edit_indices(key, &sched.bvh, &cfg, &mut scratch).is_none() {
                    continue;
                }
                let cs = corner(cfg.brick_min_world(key.coord, 0));
                let (mut neg, mut pos) = (false, false);
                for p in cs {
                    if edits::fold_csg(&all_edits, p, 0.0).dist <= 0.0 { neg = true; } else { pos = true; }
                }
                if neg && pos {
                    assert!(
                        atlas.bricks.contains_key(&key),
                        "r_in={r_in} off={off}: dropped a brick the surface passes through at {:?}",
                        key.coord
                    );
                }
            }
        }

        // (2) Per-brick culled candidate set agrees in sign with the full edit list.
        for key in atlas.bricks.keys() {
            if atlas::SdfAtlas::cull_edit_indices(*key, &sched.bvh, &cfg, &mut scratch).is_none() {
                continue;
            }
            let culled: Vec<edits::ResolvedEdit> =
                scratch.iter().map(|&i| all_edits[i as usize].clone()).collect();
            for p in corner(cfg.brick_min_world(key.coord, 0)) {
                let d_full = edits::fold_csg(&all_edits, p, 0.0).dist;
                let d_cull = edits::fold_csg(&culled, p, 0.0).dist;
                assert_eq!(
                    d_full <= 0.0, d_cull <= 0.0,
                    "r_in={r_in} off={off}: culled-set sign mismatch at brick {:?} (full={d_full:.3} cull={d_cull:.3})",
                    key.coord
                );
            }
        }
    }
}

/// Same correct-tile guard as the sync lifecycle, with CONTINUOUS
/// camera motion (one step per frame) so a background classify task routinely spans camera moves
/// — its snapshot (candidates + window + hashes) goes stale relative to the evictions the
/// recenter applied meanwhile. If the stale-snapshot apply ever re-inserts a brick with a tile
/// that disagrees with the allocator (or leaves the table referencing a freed/reused tile), this
/// catches it as a brick → wrong-tile resolve — the "garbled LOD handoff".
#[test]
fn live_table_correct_through_continuous_motion() {
    let cfg = SdfGridConfig { lod_count: 3, ring_bricks: 16, recenter_snap_chunks: 1, ..config() };
    let edits = vec![sphere_edit(Vec3::ZERO, 12.0, 0)];
    let mut sched = primed_sched(&edits);
    let mut atlas = SdfAtlas::default();
    let mut gpu = PendingGpuBakes::default();
    let mut dbg = crate::sdf_render::BakedBrickDebug::default();

    let mut rows: Vec<chunk::ChunkLookup> = Vec::new();
    let mut tiles: Vec<chunk::BrickTile> = Vec::new();
    let mut cap_rows = 0u32;
    let mut cap_slots = 0u32;

    // Continuous fly out and back, one camera step per frame, then extra frames to drain.
    let mut cams: Vec<f32> = (0..=36).map(|i| i as f32 * 2.0).collect();
    let back: Vec<f32> = cams.iter().rev().skip(1).copied().collect();
    cams.extend(back);
    let end = *cams.last().unwrap();
    for _ in 0..400 {
        cams.push(end); // hold position so the final in-flight task drains (still checking)
    }

    for (frame, &x) in cams.iter().enumerate() {
        let cam = Vec3::new(x, 0.0, 0.0);
        recenter_step(&mut sched, &mut atlas, &cfg, cam);
        gpu.jobs.clear();
        gpu.edits.clear();
        atlas.gpu_baked_tiles.clear();
        emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, cam, &mut dbg, 0.0);

        apply_table_delta(&atlas.live_chunks, &mut rows, &mut tiles, &mut cap_rows, &mut cap_slots);
        let (fr, ft) = atlas.live_chunks.full_tables();
        atlas.live_chunks.clear_dirty();
        for key in atlas.bricks.keys() {
            let tile = atlas.tiles.tile(key).expect("resident brick has a tile");
            let want = chunk::tile_atlas_base(tile);
            let (ck, local) = chunk::chunk_of(*key, &cfg);
            assert_eq!(
                chunk::resolve_via_tables(&fr, &ft, cfg.ring_chunks_per_axis(), ck, local).map(|t| t.atlas_base),
                Some(want),
                "frame {frame} (x={x}): full_tables resolves brick {key:?} to the wrong/absent tile"
            );
            assert_eq!(
                chunk::resolve_via_tables(&rows, &tiles, cfg.ring_chunks_per_axis(), ck, local).map(|t| t.atlas_base),
                Some(want),
                "frame {frame} (x={x}): delta-mirror resolves brick {key:?} to the wrong/absent tile"
            );
        }
    }
}

/// Finest resident LOD with a baked brick at world point `p` — mirrors the shader's
/// `resolve_march` fine→coarse walk (chunk-table presence only). `None` = no LOD covers `p`
/// (the shader would march into empty space there: the visible GAP).
fn served_lod(atlas: &SdfAtlas, cfg: &SdfGridConfig, p: Vec3) -> Option<u32> {
    for lod in 0..cfg.lod_count {
        let coord = cfg.world_to_brick_lod(p, lod);
        if atlas.bricks.contains_key(&atlas::BrickKey::new(lod, coord)) {
            return Some(lod);
        }
    }
    None
}

/// THE coverage-gap guard: fly the camera continuously (one bake per frame) past a surface probe
/// that stays WITHIN clipmap reach the whole path, and assert the probe's coverage NEVER drops
/// once established. A drop to `None` is the "LOD-N → gap → LOD-(N-1)" the renderer shows — the
/// recenter evicting a region's fine chunk before its coarser replacement is resident (worst
/// while the bake spreads over frames and evictions keep landing). Deferred
/// eviction (evict only once the replacement is resident) must keep this hole-free.
#[test]
fn continuous_motion_keeps_probe_covered() {
    let cfg = SdfGridConfig { lod_count: 4, ring_bricks: 16, recenter_snap_chunks: 1, ..config() };
    let radius = 16.0;
    let edits = vec![sphere_edit(Vec3::ZERO, radius, 0)];
    let mut sched = primed_sched(&edits);
    let mut atlas = SdfAtlas::default();
    let mut gpu = PendingGpuBakes::default();
    let mut dbg = crate::sdf_render::BakedBrickDebug::default();

    // Probe on the sphere surface, lateral to the +X flight so it stays in reach across the
    // whole path (coarsest LOD-3 window half-extent ≈ 2·chunk_world(3) ≈ 45 world units; the
    // probe's camera distance peaks at ≈ sqrt(38² + 16²) ≈ 41 < 45 → covered throughout).
    let probe = Vec3::new(0.0, radius, 0.3);
    let mut served: Vec<Option<u32>> = Vec::new();

    let mut cams: Vec<f32> = (0..=76).map(|i| i as f32 * 0.5).collect(); // 0..38 in 0.5 steps
    let back: Vec<f32> = cams.iter().rev().skip(1).copied().collect();
    cams.extend(back);
    let end = *cams.last().unwrap();
    for _ in 0..400 {
        cams.push(end);
    }

    for &x in &cams {
        let cam = Vec3::new(x, 0.0, 0.0);
        recenter_step(&mut sched, &mut atlas, &cfg, cam);
        gpu.jobs.clear();
        gpu.edits.clear();
        atlas.gpu_baked_tiles.clear();
        emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, cam, &mut dbg, 0.0);
        served.push(served_lod(&atlas, &cfg, probe));
    }

    // Once covered, the probe must stay covered for the rest of the (in-reach) path.
    let first = served.iter().position(|s| s.is_some()).expect("probe never covered — test setup");
    for (i, s) in served.iter().enumerate().skip(first) {
        assert!(
            s.is_some(),
            "frame {i}: probe LOST coverage (LOD → gap) — eviction outpaced the bake: {served:?}"
        );
    }
}

/// One `emit_gpu_bakes` frame never emits more than the per-frame budget, and a scene too big to
/// bake in one bounded frame keeps its undone work DEFERRED (un-drained chunks in `pending`,
/// over-budget classified Keeps in `ready`) — never half-baking the atlas.
#[test]
fn gpu_emit_caps_jobs_and_spills_overflow() {
    let cfg = SdfGridConfig { lod_count: 1, ring_bricks: 8, recenter_snap_chunks: 1, ..config() };
    // A big SOLID sphere. The narrow-band cull drops its deep interior, so the work rides on the
    // SHELL — radius 22 gives a surface band of ~30k+ bricks, far more than one bounded frame bakes.
    let edits = vec![sphere_edit(Vec3::ZERO, 22.0, 0)];
    let mut sched = primed_sched(&edits);
    let mut atlas = SdfAtlas::default();
    let mut gpu = PendingGpuBakes::default();

    // Dirty a chunk cube bounding the whole sphere (chunk_world ≈ 2.8, so ±22 ⇒ chunk ±8).
    for x in -9..=9 {
        for y in -9..=9 {
            for z in -9..=9 {
                dirty_chunk(&mut sched, chunk::ChunkKey::new(0, IVec3::new(x, y, z)));
            }
        }
    }

    emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);

    assert!(
        gpu.jobs.len() <= SOFT_BAKE_BUDGET,
        "emitted {} jobs, over the per-frame budget {SOFT_BAKE_BUDGET}",
        gpu.jobs.len(),
    );
    assert!(
        !(sched.pending.is_empty() && sched.ready.is_empty()),
        "a >budget scene can't bake in one bounded frame — undone work must stay deferred (pending/ready)"
    );
    // Every emitted job corresponds to a resident brick (so the shader can read it).
    assert_eq!(
        gpu.jobs.len(),
        atlas.bricks.len(),
        "resident brick count must equal the emitted job count (no half-inserted bricks)"
    );
}

/// CHUNK-ATOMIC budget (make-before-break): a chunk's Keep-set bakes WHOLE or spills WHOLE —
/// never partially. A half-baked chunk would appear in the GPU table with only some of its
/// bricks resident, so the finer LOD shows a sub-chunk mixed-LOD "garbled" patch during a
/// LOD-handoff. Drive `apply_verdicts` with a budget that fits the first chunk but not both, and
/// assert every chunk is all-or-nothing resident (and the over-budget one is spilled whole).
#[test]
fn apply_verdicts_bakes_or_spills_whole_chunks() {
    let cfg = config();
    let edits = vec![sphere_edit(Vec3::ZERO, 5.0, 0)]; // 1-elem snapshot so push_bake_job indexes [0]
    let ck_a = chunk::ChunkKey::new(0, IVec3::new(0, 0, 0));
    let ck_b = chunk::ChunkKey::new(0, IVec3::new(1, 0, 0));
    let a: Vec<_> = chunk_brick_keys(ck_a, &cfg).into_iter().take(3).collect();
    let b: Vec<_> = chunk_brick_keys(ck_b, &cfg).into_iter().take(3).collect();
    let mut candidates = Vec::new();
    let mut verdicts = Vec::new();
    for (h, &k) in a.iter().enumerate() {
        candidates.push((ck_a, k));
        verdicts.push(Verdict::Keep([edits::PALETTE_EMPTY; edits::PALETTE_K], vec![0], h as u64));
    }
    for (h, &k) in b.iter().enumerate() {
        candidates.push((ck_b, k));
        verdicts.push(Verdict::Keep([edits::PALETTE_EMPTY; edits::PALETTE_K], vec![0], 100 + h as u64));
    }

    let mut atlas = SdfAtlas::default();
    let mut gpu = PendingGpuBakes::default();
    let mut deferred: Vec<ReadyChunk> = Vec::new();
    // Budget 4: chunk A's 3 bricks fit (0+3 ≤ 4); A+B's 6 do not (3+3 > 4) → B spills whole.
    apply_verdicts(
        &mut atlas, &mut gpu, &edits, &cfg, &mut crate::sdf_render::BakedBrickDebug::default(),
        0.0, &candidates, verdicts, &mut deferred, 4,
    );

    assert_eq!(gpu.jobs.len(), 3, "exactly chunk A's bricks baked");
    assert_eq!(atlas.bricks.len(), gpu.jobs.len(), "no half-inserted bricks");
    // The over-budget chunk B is carried WHOLE into the ready queue (all 3 of its Keeps), A is not.
    assert!(
        deferred.iter().any(|rc| rc.ck == ck_b && rc.keeps.len() == 3) && deferred.iter().all(|rc| rc.ck != ck_a),
        "the over-budget chunk B carries whole into `ready`, chunk A bakes"
    );
    // The core invariant: each chunk is all-or-nothing resident — never a partial mix.
    for (ck, keys) in [(ck_a, &a), (ck_b, &b)] {
        let resident = keys.iter().filter(|k| atlas.bricks.contains_key(k)).count();
        assert!(
            resident == 0 || resident == keys.len(),
            "chunk {ck:?} partially resident ({resident}/{}) — not chunk-atomic",
            keys.len()
        );
    }
}

/// Drain bounded emits until the carry queue (`ready`) holds work — the mid-bake state where a
/// small edit's cost matters. Returns once `ready` is non-empty (or pending drains, which fails the
/// caller's setup assert).
fn build_carry_backlog(
    sched: &mut BakeScheduler,
    atlas: &mut SdfAtlas,
    cfg: &SdfGridConfig,
    cam: Vec3,
) {
    let mut gpu = PendingGpuBakes::default();
    let mut guard = 0;
    while sched.ready.is_empty() && !sched.pending.is_empty() {
        gpu.jobs.clear();
        gpu.edits.clear();
        atlas.gpu_baked_tiles.clear();
        emit_gpu_bakes(atlas, sched, &mut gpu, cfg, cam, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
        guard += 1;
        assert!(guard < 400, "never built a carry backlog");
    }
}

/// SELECTIVE carry-queue invalidation: a pure MOVE (index-stable) drops ONLY the re-dirtied
/// footprint from `ready`, keeping the far backlog — so a small edit stays cheap even mid-bake.
#[test]
fn ready_selective_flush_keeps_far_backlog() {
    let cfg = SdfGridConfig { lod_count: 1, ring_bricks: 8, recenter_snap_chunks: 1, ..config() };
    let edits = vec![sphere_edit(Vec3::ZERO, 22.0, 0)];
    let mut sched = primed_sched(&edits);
    let mut atlas = SdfAtlas::default();
    for x in -9..=9 { for y in -9..=9 { for z in -9..=9 {
        dirty_chunk(&mut sched, chunk::ChunkKey::new(0, IVec3::new(x, y, z)));
    }}}
    build_carry_backlog(&mut sched, &mut atlas, &cfg, Vec3::ZERO);
    assert!(sched.ready.len() >= 2, "setup: need a carried backlog of ≥2 chunks");
    let before: Vec<_> = sched.ready.iter().map(|rc| rc.ck).collect();

    // A small "move": index-stable edit_gen bump + re-dirty just two carried chunks' footprint.
    let footprint = [before[0], before[1]];
    sched.edit_gen = sched.edit_gen.wrapping_add(1);
    for &ck in &footprint { dirty_chunk(&mut sched, ck); }
    invalidate_ready_on_edit_change(&mut sched, &cfg, false);

    for &ck in &footprint {
        assert!(!sched.ready.iter().any(|rc| rc.ck == ck), "re-dirtied chunk must leave ready");
        assert!(sched.pending.contains_key(&ck), "re-dirtied chunk must be in pending for re-classify");
    }
    assert_eq!(sched.ready.len(), before.len() - footprint.len(), "the far backlog must survive a small move");
    assert_eq!(sched.ready_edit_gen, sched.edit_gen, "ready_edit_gen synced ⇒ no full flush in refresh_ready");
}

/// An add/remove shifts every edit's index position, so the WHOLE carry queue must flush to
/// re-classify against the new snapshot (a kept entry's stale indices would fold the wrong edits).
#[test]
fn ready_full_flush_on_index_shift() {
    let cfg = SdfGridConfig { lod_count: 1, ring_bricks: 8, recenter_snap_chunks: 1, ..config() };
    let edits = vec![sphere_edit(Vec3::ZERO, 22.0, 0)];
    let mut sched = primed_sched(&edits);
    let mut atlas = SdfAtlas::default();
    for x in -9..=9 { for y in -9..=9 { for z in -9..=9 {
        dirty_chunk(&mut sched, chunk::ChunkKey::new(0, IVec3::new(x, y, z)));
    }}}
    build_carry_backlog(&mut sched, &mut atlas, &cfg, Vec3::ZERO);
    let carried: Vec<_> = sched.ready.iter().map(|rc| rc.ck).collect();
    assert!(!carried.is_empty(), "setup: need a carried backlog");

    sched.edit_gen = sched.edit_gen.wrapping_add(1);
    invalidate_ready_on_edit_change(&mut sched, &cfg, true);

    assert!(sched.ready.is_empty(), "an index shift flushes the whole carry queue");
    for ck in carried { assert!(sched.pending.contains_key(&ck), "flushed chunks re-queued to pending for re-classify"); }
}

/// Option-4 core: the BRICK-level footprint (`bricks_in_aabb_windowed`) must be a strict SUBSET of the
/// whole-chunk footprint (every dirtied chunk is one the chunk-level version returns), and its total
/// dirty bricks must be MEANINGFULLY fewer than expanding every straddled chunk to 64 — otherwise the
/// optimization isn't doing anything. A small sphere is sub-chunk at coarse LODs, so the saving is large.
#[test]
fn brick_mask_is_sparse_subset_of_chunk_dirty() {
    let cfg = SdfGridConfig { lod_count: 8, ring_bricks: 128, ..config() };
    for off in [Vec3::ZERO, Vec3::new(1.3, -0.7, 4.1), Vec3::new(-12.0, 3.0, 0.5)] {
        let aabb = edit_world_aabb(&SdfPrimitive::Sphere { radius: 2.0 }, &Transform::from_translation(off), 0.0);
        let r = cfg.ring_chunks_per_axis();
        let mut brick_total = 0usize; // bricks the brick-level footprint dirties
        let mut chunk_total = 0usize; // bricks the whole-chunk footprint would classify (chunks × 64)
        for lod in 0..cfg.lod_count {
            let origin = ring_chunk_origin(&cfg, off, lod);
            let chunk_set: HashSet<_> = chunks_in_aabb_windowed(&cfg, &aabb, lod, origin, r).into_iter().collect();
            chunk_total += chunk_set.len() * chunk::CHUNK_VOLUME as usize;
            for (ck, mask) in bricks_in_aabb_windowed(&cfg, &aabb, lod, origin, r) {
                assert!(mask != 0, "an emitted chunk must carry ≥1 dirty brick");
                assert!(chunk_set.contains(&ck), "brick-level dirty chunk {ck:?} must lie in the whole-chunk footprint");
                brick_total += mask.count_ones() as usize;
            }
        }
        assert!(brick_total > 0, "footprint must dirty something");
        assert!(
            brick_total * 2 <= chunk_total,
            "brick masking must cut candidates ≥2× (got {brick_total} bricks vs {chunk_total} whole-chunk)"
        );
    }
}

/// Surface-pruned move dirtying (`dirty_moving_edit`): dragging a LARGE solid sphere must dirty only
/// its moving surface SHELL, never its interior. Deep-interior bricks stay UNDIRTIED (their clamped
/// SDF is an unchanged constant), and the dirty set is far sparser than the solid-AABB footprint —
/// the O(surface) vs O(volume) win. A trailing brick that the surface just left must still be dirtied
/// (so it can be evicted), which the old-position test guarantees.
#[test]
fn surface_prune_dirties_shell_not_interior() {
    let cfg = SdfGridConfig { lod_count: 3, ring_bricks: 64, recenter_snap_chunks: 1, ..config() };
    let radius = 8.0_f32;
    let op = SdfOp { kind: CsgKind::Union, smoothing: 0.0 };
    let old = ResolvedEdit::new(SdfPrimitive::Sphere { radius }, Transform::from_translation(Vec3::ZERO), op, 0);
    let new = ResolvedEdit::new(SdfPrimitive::Sphere { radius }, Transform::from_translation(Vec3::new(0.3, 0.0, 0.0)), op, 0);

    let mut pending: rustc_hash::FxHashMap<chunk::ChunkKey, u64> = rustc_hash::FxHashMap::default();
    dirty_moving_edit(&mut pending, &old, &new, &cfg, Vec3::ZERO);
    let shell: usize = pending.values().map(|m| m.count_ones() as usize).sum();

    // Compare to the SOLID footprint (every brick in the new sphere's padded AABB).
    let new_aabb = edit_world_aabb(&new.prim, &new.transform, 0.0);
    let solid: usize = (0..cfg.lod_count)
        .map(|lod| {
            let o = ring_chunk_origin(&cfg, Vec3::ZERO, lod);
            chunks_in_aabb_windowed(&cfg, &new_aabb, lod, o, cfg.ring_chunks_per_axis()).len()
                * chunk::CHUNK_VOLUME as usize
        })
        .sum();

    assert!(shell > 0, "the moving surface must dirty something");
    assert!(
        shell * 3 < solid,
        "surface-prune must be far sparser than the solid AABB footprint (shell={shell}, solid={solid})"
    );

    // The deep-interior brick at the sphere center must NOT be dirty (its SDF is saturated-negative,
    // unchanged by the move).
    let center_brick = atlas::BrickKey::new(0, IVec3::ZERO);
    let (center_ck, center_local) = chunk::chunk_of(center_brick, &cfg);
    let center_mask = pending.get(&center_ck).copied().unwrap_or(0);
    assert!(
        center_mask & (1u64 << center_local) == 0,
        "the deep-interior center brick must be pruned, not dirtied"
    );
}

/// Carry-queue correctness under brick masking: when a pure move re-dirties only SOME bricks of a
/// chunk that has a carried (already-classified, already-evicted) `ReadyChunk`, the WHOLE group is
/// invalidated and ALL its carried bricks are re-queued into `pending` — never silently dropped (they
/// were evicted on defer, so a re-queue is the only path that bakes them back). A carried group the
/// move doesn't touch stays in `ready` with its valid palette.
#[test]
fn ready_carried_bricks_requeued_on_overlapping_move() {
    let cfg = config();
    let hit_ck = chunk::ChunkKey::new(0, IVec3::new(2, 0, 0));
    let miss_ck = chunk::ChunkKey::new(0, IVec3::new(9, 0, 0));
    let bricks_hit = chunk_brick_keys(hit_ck, &cfg); // index i == local slot i
    let bricks_miss = chunk_brick_keys(miss_ck, &cfg);
    let keep = |k: atlas::BrickKey| (k, [edits::PALETTE_EMPTY; edits::PALETTE_K], Vec::<u32>::new(), 0u64);

    let mut sched = primed_sched(&[]);
    // Carried group on hit_ck holds local slots {0, 5}; an independent group on miss_ck holds {20}.
    sched.ready = vec![
        ReadyChunk { ck: hit_ck, keeps: vec![keep(bricks_hit[0]), keep(bricks_hit[5])] },
        ReadyChunk { ck: miss_ck, keeps: vec![keep(bricks_miss[20])] },
    ];
    sched.edit_gen = sched.edit_gen.wrapping_add(1); // a pure move (index-stable)
    dirty_mask(&mut sched.pending, hit_ck, 1u64 << 5); // footprint touches only slot 5 of hit_ck

    invalidate_ready_on_edit_change(&mut sched, &cfg, false);

    // hit_ck's group is invalidated → BOTH its carried bricks (0 and 5) are back in pending.
    let m = sched.pending.get(&hit_ck).copied().expect("overlapped chunk stays in pending");
    assert!(m & (1 << 0) != 0, "carried brick 0 must be re-queued, not lost");
    assert!(m & (1 << 5) != 0, "re-dirtied carried brick 5 must be in pending");
    // miss_ck's group is untouched → still carried with its palette.
    assert_eq!(sched.ready.len(), 1, "the non-overlapping carried group survives");
    assert_eq!(sched.ready[0].ck, miss_ck);
}

/// A spilled brick must NOT be inserted into the atlas — it stays non-resident so the
/// shader falls back to the coarser LOD. Conversely, empty bricks are evicted even on a
/// capped frame. Here: resident bricks == emitted jobs, and the spill is purely deferred.
#[test]
fn gpu_emit_spilled_bricks_stay_non_resident() {
    let cfg = SdfGridConfig { lod_count: 1, ring_bricks: 8, recenter_snap_chunks: 1, ..config() };
    // Big SOLID sphere: the cull drops its deep interior, so the >cap overflow rides on the
    // shell + bounding-box exterior alone (~100k+ bricks). (A solid box would cull to ~0.)
    let edits = vec![sphere_edit(Vec3::ZERO, 22.0, 0)];
    let mut sched = primed_sched(&edits);
    let mut atlas = SdfAtlas::default();
    let mut gpu = PendingGpuBakes::default();
    for x in -9..=9 {
        for y in -9..=9 {
            for z in -9..=9 {
                dirty_chunk(&mut sched, chunk::ChunkKey::new(0, IVec3::new(x, y, z)));
            }
        }
    }

    emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);

    // No brick is resident without a corresponding job this frame: a capped (spilled)
    // brick is never inserted, so the atlas never exposes an un-baked (zero) tile.
    assert_eq!(atlas.bricks.len(), gpu.jobs.len());
    assert!(atlas.bricks.len() <= GPU_BAKE_JOB_CAP);

    // Draining the spill over subsequent frames eventually bakes everything (no chunk is
    // dropped). Run until pending empties; the resident set grows monotonically.
    let mut guard = 0;
    let mut last = atlas.bricks.len();
    while !(sched.pending.is_empty() && sched.ready.is_empty()) {
        gpu.jobs.clear();
        gpu.edits.clear();
        atlas.gpu_baked_tiles.clear();
        emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
        assert!(atlas.bricks.len() >= last, "resident set must not shrink while draining spill");
        last = atlas.bricks.len();
        guard += 1;
        assert!(guard < 100, "spill drain did not converge");
    }
    assert!(atlas.bricks.len() > GPU_BAKE_JOB_CAP, "all dirty bricks eventually resident");
}

/// Narrow-band interior cull: a solid object bakes only its surface SHELL, not its deep
/// interior. The march reads the field from OUTSIDE (rays shrink to the surface and stop),
/// so interior bricks are write-only waste — and for a big solid they're the r³ bulk that
/// drives the approach-bake hitch. Assert: (a) the brick at the centre of a large solid is
/// NOT resident, (b) a brick straddling the surface IS, (c) the resident count is far below
/// the solid's full bounding-box brick count (what the old AABB-only cull kept).
#[test]
fn gpu_emit_culls_deep_interior_of_solid() {
    // Pin the 0.1 m voxel this cull test was tuned for (radius/dirty-cube extents + the "< half the AABB
    // cube" ratio), independent of the default voxel_size.
    let cfg = SdfGridConfig { lod_count: 1, ring_bricks: 8, recenter_snap_chunks: 1, voxel_size: 0.1, ..config() };
    let radius = 10.0;
    let edits = vec![sphere_edit(Vec3::ZERO, radius, 0)];
    let mut sched = primed_sched(&edits);
    let mut atlas = SdfAtlas::default();

    // Dirty a chunk cube bounding the sphere (chunk_world = 2.8 ⇒ ±10 ⇒ chunk ±4).
    for x in -5..=5 {
        for y in -5..=5 {
            for z in -5..=5 {
                dirty_chunk(&mut sched, chunk::ChunkKey::new(0, IVec3::new(x, y, z)));
            }
        }
    }
    // Drain fully (cap may spill across frames) so the resident set is the final one.
    let mut gpu = PendingGpuBakes::default();
    let mut guard = 0;
    loop {
        gpu.jobs.clear();
        gpu.edits.clear();
        atlas.gpu_baked_tiles.clear();
        emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
        guard += 1;
        assert!(guard < 1000, "cull-test settle did not converge");
        if sched.pending.is_empty() && sched.ready.is_empty() {
            break;
        }
    }

    let brick_at = |p: Vec3| atlas::BrickKey::new(0, cfg.world_to_brick_lod(p, 0));
    // (a) Dead centre of the solid: surface is `radius` away ≫ brick reach → culled.
    assert!(
        !atlas.bricks.contains_key(&brick_at(Vec3::ZERO)),
        "deep-interior brick at the sphere centre must be culled (write-only waste)"
    );
    // (b) A brick straddling the surface (just inside it) must stay resident.
    assert!(
        atlas.bricks.contains_key(&brick_at(Vec3::new(radius - 0.05, 0.0, 0.0))),
        "surface-shell brick must remain resident (the march reads it)"
    );
    // (c) Resident ≪ what the OLD (AABB-only) cull kept. That cull kept every brick whose
    // box overlapped the sphere's bounding BOX — i.e. the full (2r)³ cube. The narrow-band
    // cull keeps only the shell, so it must be well under half that cube.
    let bw = cfg.brick_world_size(0);
    let bbox_bricks = ((2.0 * radius / bw).ceil() as usize).pow(3);
    assert!(
        atlas.bricks.len() < bbox_bricks / 2,
        "resident {} should be far below the AABB-cull bounding-box count {}",
        atlas.bricks.len(),
        bbox_bricks
    );
}

/// The heightmap is a ONE-SIDED surface (signed vertical distance to the noise height, no box
/// floor/walls), so everything below the surface — the deep interior AND the underside — culls
/// away, leaving only the walkable TOP-surface shell baked. Verifies we're not baking the solid
/// block of dirt (nor a closed box's bottom/walls).
#[ignore = "obsolete since c57fccf: heightmap is GPU-baked only (eval_primitive returns f32::MAX), so \
            CPU classify no longer culls/keeps the heightmap shell — this CPU-side check can't hold"]
#[test]
fn heightmap_culls_solid_interior_keeps_shell() {
    let cfg = SdfGridConfig { lod_count: 1, ring_bricks: 8, recenter_snap_chunks: 1, ..config() };
    // Box [±5 XZ, 0..14 Y] clipped to a noise surface near y≈7 → a ~7 m-thick terrain solid.
    let half_xz = bevy::math::Vec2::new(5.0, 5.0);
    let max_height = 14.0f32;
    let edit = ResolvedEdit::new(
        SdfPrimitive::Heightmap { half_xz, max_height, freq: 0.2, amp: 1.5, seed: 7 },
        Transform::IDENTITY,
        SdfOp { kind: CsgKind::Union, smoothing: 0.0 },
        0,
    );
    let edits = vec![edit];
    let mut sched = primed_sched(&edits);
    let mut atlas = SdfAtlas::default();

    // Dirty the chunk cube bounding the whole box (chunk_world ≈ 2.8 m).
    for x in -2..=2 {
        for y in 0..=5 {
            for z in -2..=2 {
                dirty_chunk(&mut sched, chunk::ChunkKey::new(0, IVec3::new(x, y, z)));
            }
        }
    }
    let mut gpu = PendingGpuBakes::default();
    let mut guard = 0;
    loop {
        gpu.jobs.clear();
        gpu.edits.clear();
        atlas.gpu_baked_tiles.clear();
        emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
        guard += 1;
        assert!(guard < 1000, "heightmap cull settle did not converge");
        if sched.pending.is_empty() && sched.ready.is_empty() {
            break;
        }
    }

    let brick_at = |p: Vec3| atlas::BrickKey::new(0, cfg.world_to_brick_lod(p, 0));
    // (a) Deep interior of the dirt (well below the ~7 m surface, above the bottom, off the
    // walls): nearest surface ≫ a brick's reach → culled (write-only waste).
    assert!(
        !atlas.bricks.contains_key(&brick_at(Vec3::new(0.0, 3.5, 0.0))),
        "deep-interior terrain brick must be culled — we should only bake the shell"
    );
    // (b) A brick at the walkable top surface stays resident (the march reads it).
    assert!(
        atlas.bricks.contains_key(&brick_at(Vec3::new(0.0, 7.0, 0.0))),
        "top-surface shell brick must remain resident"
    );
    // (c) Resident ≪ the full bounding-box brick count (interior + above-surface empty culled).
    let bw = cfg.brick_world_size(0);
    let bbox = ((2.0 * half_xz.x / bw).ceil() as usize)
        * ((max_height / bw).ceil() as usize)
        * ((2.0 * half_xz.y / bw).ceil() as usize);
    eprintln!(
        "heightmap cull: resident={} bricks, full box={} bricks ({}% — shell only)",
        atlas.bricks.len(),
        bbox,
        atlas.bricks.len() * 100 / bbox
    );
    assert!(
        atlas.bricks.len() < bbox / 2,
        "resident {} should be far below the box brick count {} (shell only, no solid fill)",
        atlas.bricks.len(),
        bbox
    );
}

/// Bake-cache skip: re-emitting an already-baked chunk within the SAME edit epoch produces
/// ZERO jobs (the bricks' `baked_epoch` matches → skipped, no re-cull/re-bake), but bumping
/// `edit_epoch` (as an edit change does) lapses every stamp so the next emit re-bakes them.
/// `dirty_edit_footprints` (the SOLE bake-dirty source for an edit change) must enqueue only the
/// chunks an edit's AABB reaches — a SPARSE set — never the whole dense window. Flooding `pending`
/// with the `R³·lod_count` mostly-empty window was the cold-bake stall (the bounded drain grinding
/// through empties for thousands of frames). Guards that the dirty set scales with GEOMETRY, not window.
#[test]
fn dirty_edit_footprints_is_sparse_not_whole_window() {
    // Production-scale window (large) so a small edit is genuinely sparse within it.
    let cfg = SdfGridConfig { lod_count: 8, ring_bricks: 128, ..config() };
    let aabb = edit_world_aabb(
        &SdfPrimitive::Sphere { radius: 2.0 },
        &Transform::from_translation(Vec3::ZERO),
        0.0,
    );
    let mut pending: rustc_hash::FxHashMap<chunk::ChunkKey, u64> = rustc_hash::FxHashMap::default();
    dirty_edit_footprints(&mut pending, &[aabb], &cfg, Vec3::ZERO);
    assert!(!pending.is_empty(), "an edit's footprint must enqueue its chunks");
    let window = (cfg.ring_chunks_per_axis() as usize).pow(3) * cfg.lod_count as usize;
    assert!(
        pending.len() * 100 < window,
        "footprint ({}) must be FAR below the {window}-chunk window — not a window flood",
        pending.len()
    );
}

/// The INCREMENTAL refit fast path (`schedule_bakes` step 1's drag path: update `edits[idx]` in
/// place + `bvh.refit_edit` + dirty the footprint) must bake the SAME resident set as a FULL
/// re-gather + `Bvh::build`. Proves refitting one moved edit's BVH leaf is equivalent to a rebuild.
#[test]
fn incremental_refit_matches_full_rebuild() {
    let cfg = SdfGridConfig { lod_count: 3, ring_bricks: 16, recenter_snap_chunks: 1, ..config() };
    let edits = vec![
        sphere_edit(Vec3::new(-8.0, 0.0, 0.0), 4.0, 0),
        sphere_edit(Vec3::new(0.0, 0.0, 0.0), 4.0, 0), // index 1 — the edit we move
        sphere_edit(Vec3::new(8.0, 0.0, 0.0), 4.0, 0),
    ];
    let cam = Vec3::ZERO;
    let moved = sphere_edit(Vec3::new(0.0, 9.0, 0.0), 4.0, 0);
    let old_aabb = edit_world_aabb(&edits[1].prim, &edits[1].transform, 0.0);
    let new_aabb = edit_world_aabb(&moved.prim, &moved.transform, 0.0);

    // (A) FULL-REBUILD reference: replace edit 1, build a fresh scheduler/BVH, settle.
    let mut full = edits.clone();
    full[1] = moved.clone();
    let mut a_sched = primed_sched(&full);
    let mut a_atlas = SdfAtlas::default();
    settle_gpu(&mut a_sched, &mut a_atlas, &cfg, cam);

    // (B) INCREMENTAL: settle the original, then mirror the step-1 drag path — refit in place.
    let mut b_sched = primed_sched(&edits);
    let mut b_atlas = SdfAtlas::default();
    settle_gpu(&mut b_sched, &mut b_atlas, &cfg, cam);
    Arc::make_mut(&mut b_sched.edits)[1] = moved;
    Arc::make_mut(&mut b_sched.bvh).refit_edit(1, new_aabb);
    let rr = cfg.ring_chunks_per_axis();
    for lod in 0..cfg.lod_count {
        let origin = ring_chunk_origin(&cfg, cam, lod);
        for ck in chunks_in_aabb_windowed(&cfg, &old_aabb, lod, origin, rr) { dirty_chunk(&mut b_sched, ck); }
        for ck in chunks_in_aabb_windowed(&cfg, &new_aabb, lod, origin, rr) { dirty_chunk(&mut b_sched, ck); }
    }
    settle_gpu(&mut b_sched, &mut b_atlas, &cfg, cam);

    let a_bricks: HashSet<_> = a_atlas.bricks.keys().copied().collect();
    let b_bricks: HashSet<_> = b_atlas.bricks.keys().copied().collect();
    assert_eq!(a_bricks, b_bricks, "incremental refit's resident set diverged from a full rebuild");
}

/// This is the core of the multi-frame-bake hitch fix: a spilled chunk re-queued each frame
/// no longer re-processes the bricks it already baked.
#[test]
fn gpu_emit_skips_already_baked_within_epoch() {
    let cfg = SdfGridConfig { lod_count: 1, ring_bricks: 8, recenter_snap_chunks: 1, ..config() };
    let edits = vec![sphere_edit(Vec3::ZERO, 3.0, 0)];
    let mut sched = primed_sched(&edits);
    let mut atlas = SdfAtlas::default();
    let mut gpu = PendingGpuBakes::default();

    let chunks: Vec<_> = (-2..=2)
        .flat_map(|x| (-2..=2).flat_map(move |y| (-2..=2).map(move |z| chunk::ChunkKey::new(0, IVec3::new(x, y, z)))))
        .collect();

    // Frame 1: bake the sphere shell. Some bricks become resident.
    for ck in &chunks { dirty_chunk(&mut sched, *ck); }
    emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
    let baked = atlas.bricks.len();
    assert!(baked > 0, "first emit must bake the shell");

    // Frame 2: same chunks dirtied again, edits UNCHANGED → every brick's content hash
    // matches its resident hash → all skipped, no jobs.
    gpu.clear();
    atlas.gpu_baked_tiles.clear();
    for ck in &chunks { dirty_chunk(&mut sched, *ck); }
    emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
    assert_eq!(gpu.jobs.len(), 0, "re-emit with unchanged edits must skip all baked bricks (content hash)");
    assert_eq!(atlas.bricks.len(), baked, "resident set unchanged on a pure re-emit");

    // Frame 3: the edit MOVED → the bricks it folds now hash differently → they re-bake.
    let moved = vec![sphere_edit(Vec3::new(0.5, 0.0, 0.0), 3.0, 0)];
    sched = primed_sched(&moved);
    gpu.clear();
    atlas.gpu_baked_tiles.clear();
    for ck in &chunks { dirty_chunk(&mut sched, *ck); }
    emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
    assert!(!gpu.jobs.is_empty(), "after the edit moves, its bricks must re-bake (content hash changed)");
}

/// Fold the shared `tower_field_edits` list into a `(ResolvedEdit, world AABB)` pair list — the
/// exact geometry the runtime `TowerSpawner` produces, so the bake-cache test exercises the real
/// stress scene. Roles map to arbitrary distinct material ids.
fn gallery_resolved() -> Vec<(ResolvedEdit, Aabb3d)> {
    use tower_field::TowerRole;
    tower_field::tower_field_edits(&tower_field::TowerFieldParams::default())
        .into_iter()
        .map(|(_order, transform, prim, role)| {
            let mat = match role {
                TowerRole::Ground => 0u16,
                TowerRole::Cube => 1,
                TowerRole::Cap => 2,
            };
            let op = SdfOp { kind: CsgKind::Union, smoothing: 0.0 };
            let aabb = edit_world_aabb(&prim, &transform, op.smoothing);
            (ResolvedEdit::new(prim, transform, op, mat), aabb)
        })
        .collect()
}

/// Drive the production moved-edit dirty path (step 1 of `schedule_bakes`): surface-pruned dirty over
/// the changed edit's old∪new position. Mirrors the real code exactly so the test's dirty set is the
/// one the app would produce.
fn dirty_moved_edit(
    sched: &mut BakeScheduler,
    cfg: &SdfGridConfig,
    cam: Vec3,
    old: &ResolvedEdit,
    new: &ResolvedEdit,
) {
    dirty_moving_edit(&mut sched.pending, old, new, cfg, cam);
}

/// REGRESSION GUARD for the content-hash bake cache (the "moving one object re-bakes the
/// terrain" bug). Uses the REAL gallery — `gallery_demo_edits`: a procedural heightmap ground +
/// six cube-towers (rotated cubes) capped by red spheres — at the production 8-LOD config.
/// Procedure:
///   1. Settle the full multi-LOD resident set at the gallery camera (dominated by heightmap).
///   2. NUDGE one tower's red sphere a few cm and dirty exactly the chunks the production
///      moved-edit path would (old∪new footprint, all LODs).
///   3. Assert the re-bake job count is a small fraction of the resident set — only the moved
///      sphere's own bricks re-bake; every heightmap / neighbour-tower brick its coarse
///      footprint overlaps is content-hash-skipped. Before the per-edit AABB refine in
///      `cull_edit_indices`, the moved sphere leaked into every brick sharing its BVH leaf,
///      flipping their content hash and re-baking the whole overlapping set.
#[test]
fn moving_sphere_near_heightmap_does_not_rebake_heightmap() {
    let cfg = SdfGridConfig { recenter_snap_chunks: 1, ..config() };
    // Gallery camera (orbit default sits ~10 units out, looking at origin).
    let cam = Vec3::new(0.0, 5.0, 10.0);

    let pairs = gallery_resolved();
    let edits0: Vec<ResolvedEdit> = pairs.iter().map(|(e, _)| e.clone()).collect();
    // Move a capping red sphere from a tower NEAR the camera (so it sits in the fine LOD ring
    // and actually re-bakes). Pick the sphere whose XZ is closest to the origin.
    let moved_idx = edits0
        .iter()
        .enumerate()
        .filter(|(_, e)| matches!(e.prim, SdfPrimitive::Sphere { .. }))
        .min_by(|(_, a), (_, b)| {
            let da = a.transform.translation.xz().length_squared();
            let db = b.transform.translation.xz().length_squared();
            da.partial_cmp(&db).unwrap()
        })
        .map(|(i, _)| i)
        .expect("gallery must contain capping spheres");

    let mut sched = primed_sched(&edits0);
    let mut atlas = SdfAtlas::default();

    // 1) Settle the full resident set through the production recenter + emit path.
    settle_gpu(&mut sched, &mut atlas, &cfg, cam);
    let resident = atlas.bricks.len();
    assert!(resident > 500, "gallery heightmap should make the resident set large (got {resident})");

    // 2) Nudge the capping sphere a few cm; everything else UNCHANGED. Rebuild edits/BVH and
    //    dirty exactly the production old∪new footprint chunks.
    let old_edit = edits0[moved_idx].clone();
    let mut new_edits = edits0.clone();
    let moved_tf = {
        let t = new_edits[moved_idx].transform;
        Transform { translation: t.translation + Vec3::new(0.04, 0.0, 0.0), ..t }
    };
    let new_edit = ResolvedEdit::new(new_edits[moved_idx].prim.clone(), moved_tf, new_edits[moved_idx].op, new_edits[moved_idx].material_id);
    new_edits[moved_idx] = new_edit.clone();
    sched.edits = Arc::new(new_edits);
    sched.bvh = Arc::new(build_bvh(&sched.edits));

    dirty_moved_edit(&mut sched, &cfg, cam, &old_edit, &new_edit);
    let mut gpu = PendingGpuBakes::default();
    atlas.gpu_baked_tiles.clear();
    emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, cam, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);

    // 3) Only the moved sphere's own bricks re-bake — a tiny fraction of the resident set. The
    //    scatter gallery is ~14k edits settling ~78k resident bricks; the content-hash cache
    //    keeps the rebake to a few dozen (the moved sphere's own shell). A leak would re-bake
    //    hundreds-to-thousands as the moved edit's coarse footprint dragged in unchanged terrain.
    let rebaked = gpu.jobs.len();
    assert!(
        rebaked > 0,
        "the moved sphere's bricks MUST re-bake (content changed)"
    );
    assert!(
        rebaked < 200,
        "moving one sphere re-baked {rebaked} of {resident} resident bricks — unchanged terrain \
         / neighbour towers are being re-baked too (content-hash cache leak)"
    );
}

/// Empty-space bricks are evicted the same frame even under the job cap (eviction is
/// CPU-only, never spilled) — the fix for the drag trail must survive the cap.
#[test]
fn gpu_emit_evicts_empties_under_cap() {
    let cfg = SdfGridConfig { lod_count: 1, ring_bricks: 8, recenter_snap_chunks: 1, ..config() };
    // Small edit at the origin; most chunks are empty space.
    let edits = vec![box_edit(Vec3::ZERO, 0.5, 0)];
    let mut sched = primed_sched(&edits);
    let mut atlas = SdfAtlas::default();
    let mut gpu = PendingGpuBakes::default();

    // Pre-populate a far brick as if it were resident from a previous position, then dirty
    // its chunk: the edit doesn't reach it, so it must be evicted this frame. Use a real
    // brick key from the chunk's enumeration so it's stride-aligned (chunk_brick_keys must
    // actually visit it).
    let far_chunk = chunk::ChunkKey::new(0, IVec3::new(100, 0, 0));
    let far = chunk_brick_keys(far_chunk, &cfg)[0];
    atlas.insert_gpu_brick(far, [edits::PALETTE_EMPTY; edits::PALETTE_K], 0, &cfg);
    assert!(atlas.bricks.contains_key(&far));
    // The edit doesn't reach this far brick, so it classifies as Empty and is evicted this
    // frame regardless of any content hash — the content-hash skip only applies to a Keep.
    dirty_chunk(&mut sched, far_chunk);
    dirty_chunk(&mut sched, chunk::ChunkKey::new(0, IVec3::ZERO));

    emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);

    assert!(
        !atlas.bricks.contains_key(&far),
        "a now-empty brick must be evicted, not left as a trail"
    );
}

/// The headline correctness guarantee: drive the camera back and forth across geometry
/// (so chunks repeatedly exit and re-enter windows) via the real recenter + GPU emit, then
/// settle at the final camera. The resident set must equal a fresh settle there — no stale
/// leading edge, no missing bricks. Absolute addressing makes a brick that exits and
/// re-enters identical, so the walk must converge to the same set as arriving directly.
#[test]
fn recenter_walk_converges_to_fresh_settle() {
    let cfg = SdfGridConfig { lod_count: 3, ring_bricks: 8, recenter_snap_chunks: 1, ..Default::default() };
    let edits: Vec<ResolvedEdit> = (-6i32..=6).map(|i| box_edit(Vec3::new(i as f32 * 1.2, 0.0, 0.0), 0.4, (i.rem_euclid(3)) as u16)).collect();

    let mut atlas = SdfAtlas::default();
    let mut sched = primed_sched(&edits);

    // Walk a winding path forward and back across several brick/chunk boundaries.
    let path = [0.0f32, 2.0, 4.0, 1.0, -3.0, -1.0, 5.0, 0.0, 3.0, -4.0, 0.0];
    for &x in &path {
        settle_gpu(&mut sched, &mut atlas, &cfg, Vec3::new(x, 0.0, 0.0));
    }
    let final_cam = Vec3::new(*path.last().unwrap(), 0.0, 0.0);
    let walked: HashSet<_> = atlas.bricks.keys().copied().collect();

    // A fresh arrival at the same camera (independent scheduler/atlas).
    let mut fresh_atlas = SdfAtlas::default();
    let mut fresh_sched = primed_sched(&edits);
    settle_gpu(&mut fresh_sched, &mut fresh_atlas, &cfg, final_cam);
    let fresh: HashSet<_> = fresh_atlas.bricks.keys().copied().collect();

    assert_eq!(walked, fresh, "recenter walk diverged from a fresh settle (stale/missing bricks)");
}

/// PERF SIMULATION of the camera-movement hitch in the stress scene. Settles the full tower
/// field at the production 8-LOD config, then walks the camera in small steps (crossing several
/// recenter snap boundaries) and, for each step, counts the recenter's per-frame work:
/// `window_scans` (entered-shell chunk slots visited across all recentering LODs),
/// `geom_queries` (how many ran a BVH AABB query), and `enqueued` (geometry chunks newly
/// dirtied this frame). Printed so we can SEE the spike on the frames where coarse LODs snap.
/// Not an assertion-heavy test, a measurement rig (run with --ignored --nocapture).
#[test]
#[ignore = "perf measurement rig; run explicitly with --ignored --nocapture"]
fn lod_recenter_cost_walk() {
    use tower_field::TowerRole;
    let cfg = SdfGridConfig::default(); // production: 8 LODs, ring 64, snap 2
    let edits: Vec<ResolvedEdit> = tower_field::tower_field_edits(&tower_field::TowerFieldParams::default())
        .into_iter()
        .map(|(_o, t, p, role)| {
            let mat = match role { TowerRole::Ground => 0u16, TowerRole::Cube => 1, TowerRole::Cap => 2 };
            ResolvedEdit::new(p, t, SdfOp { kind: CsgKind::Union, smoothing: 0.0 }, mat)
        })
        .collect();
    eprintln!("LOD-WALK: {} edits, building BVH...", edits.len());

    let mut atlas = SdfAtlas::default();
    let mut sched = primed_sched(&edits);
    // Settle at the start camera (orbit default ~10 units out).
    let r = cfg.ring_chunks_per_axis();
    settle_gpu(&mut sched, &mut atlas, &cfg, Vec3::new(0.0, 5.0, 10.0));
    eprintln!("LOD-WALK: settled, resident bricks = {}", atlas.bricks.len());

    // Walk along +X in 1.5 m steps — at base voxel 0.1 / chunk_world ≈ 3.2 m, snap 2 ⇒ a LOD0
    // recenter every ~6.4 m, coarser LODs every 2^L × that, so steps cross staggered boundaries.
    let mut max_scans = 0usize;
    let mut max_geom = 0usize;
    let mut max_enq = 0usize;
    for step in 1..=40i32 {
        let cam = Vec3::new(step as f32 * 1.5, 5.0, 10.0);
        // Instrumented mirror of step 2 recenter: count window scans + geometry queries + time.
        let mut scans = 0usize;
        let mut geom = 0usize;
        let mut enqueued = 0usize;
        let mut stack: Vec<u32> = Vec::new();
        let t_recenter = std::time::Instant::now();
        for lod in 0..cfg.lod_count {
            let li = lod as usize;
            let new_origin = ring_chunk_origin(&cfg, cam, lod);
            let old_origin = sched.ring_chunk_origin[li];
            if new_origin == old_origin { continue; }
            for_each_entered_chunk(new_origin, old_origin, r, |coord| {
                scans += 1;
                geom += 1;
                let ck = chunk::ChunkKey::new(lod, coord);
                if chunk_has_geometry_with(ck, &sched.bvh, &cfg, &mut stack) {
                    if !sched.pending.contains_key(&ck) {
                        enqueued += 1;
                    }
                    dirty_chunk(&mut sched, ck);
                }
            });
            for_each_exited_chunk(new_origin, old_origin, r, |coord| {
                let ck = chunk::ChunkKey::new(lod, coord);
                sched.pending.remove(&ck);
                for_each_brick_key(ck, &cfg, |bk| { atlas.remove_brick(&bk, &cfg); });
            });
            sched.ring_chunk_origin[li] = new_origin;
        }
        let recenter_us = t_recenter.elapsed().as_micros();
        let mut gpu = PendingGpuBakes::default();
        let t_emit = std::time::Instant::now();
        emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, cam, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
        let emit_us = t_emit.elapsed().as_micros();
        if scans > 0 {
            eprintln!("step {step:2}: scans={scans:6} geom_q={geom:5} enq={enqueued:4} candidates~{:6} baked={:5} | recenter={recenter_us:5}us emit={emit_us:6}us emit_per_job={:.2}us",
                enqueued * 64, gpu.jobs.len(), emit_us as f64 / (gpu.jobs.len().max(1)) as f64);
        }
        max_scans = max_scans.max(scans);
        max_geom = max_geom.max(geom);
        max_enq = max_enq.max(enqueued);
    }
    eprintln!("LOD-WALK MAX: window_scans={max_scans} geom_queries={max_geom} enqueued={max_enq}");
}

/// Flying *away* from a localized scene must still refresh the scene's bricks into their
/// new (coarser) LOD rings — the same resident set as a fresh settle at the destination,
/// and the bake enqueues stay bounded by the scene's shell footprint (NOT the empty volume
/// swept), so flying 4× farther does not enqueue ~4× the chunks (the empty-chunk cull).
#[test]
fn flying_away_still_refreshes_scene_lod() {
    let cfg = SdfGridConfig { lod_count: 4, ring_bricks: 8, recenter_snap_chunks: 1, ..Default::default() };
    let edits: Vec<ResolvedEdit> =
        (-1i32..=1).map(|i| box_edit(Vec3::new(i as f32 * 0.5, 0.0, 0.0), 0.4, 0)).collect();

    // Fly `steps` small steps away from the scene; return (resident chunk set, total
    // geometry-chunk enqueues over the flight, excluding the initial fill).
    let run = |sign: f32, steps: i32| -> (HashSet<chunk::ChunkKey>, usize) {
        let mut atlas = SdfAtlas::default();
        let mut sched = primed_sched(&edits);
        settle_gpu(&mut sched, &mut atlas, &cfg, Vec3::ZERO); // initial fill
        let mut enqueued = 0usize;
        let mut gpu = PendingGpuBakes::default();
        for i in 1..=steps {
            let cam = Vec3::new(sign * i as f32 * 0.4, 0.0, 0.0);
            enqueued += recenter_step(&mut sched, &mut atlas, &cfg, cam);
            // Drain this frame's emission (the GPU bake; spill drains over frames).
            let mut guard = 0;
            loop {
                gpu.jobs.clear();
                gpu.edits.clear();
                atlas.gpu_baked_tiles.clear();
                emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
                guard += 1;
                assert!(guard < 1000, "frame drain did not converge");
                if sched.pending.is_empty() && sched.ready.is_empty() { break; }
            }
        }
        let set = atlas.bricks.keys().map(|k| chunk::chunk_of(*k, &cfg).0).collect();
        (set, enqueued)
    };

    // 1) Symmetry + correctness: flying away either direction leaves the same resident set
    //    as a fresh settle at the destination (no stale fine bricks, nothing missing).
    for (label, sign, steps) in [("forward", 1.0, 16), ("backward", -1.0, 16)] {
        let (flown, _) = run(sign, steps);
        let mut fresh_atlas = SdfAtlas::default();
        let mut fresh_sched = primed_sched(&edits);
        let fresh = settle_gpu(&mut fresh_sched, &mut fresh_atlas, &cfg, Vec3::new(sign * steps as f32 * 0.4, 0.0, 0.0));
        assert_eq!(flown, fresh, "{label}: flew-in resident chunks diverged from a fresh settle");
        assert!(!flown.is_empty(), "{label}: scene vanished after flying away");
    }

    // 2) The cull's core guarantee: enqueues while flying away are bounded by the scene's
    //    shell footprint, NOT the empty volume swept — flying 4× farther must NOT enqueue
    //    ~4× the chunks.
    let (_, near) = run(-1.0, 8);
    let (_, far) = run(-1.0, 32); // 4× the distance
    assert!(
        far <= near * 2,
        "enqueues scaled with flight distance (near={near}, far={far}) — empty chunks not culled, scene will starve"
    );
}

/// `ring_bricks / CHUNK_BRICKS` chunks per axis. With the defaults (12 / 4) that is 3.
#[test]
fn ring_window_is_chunks_per_axis() {
    let cfg = config();
    let r = cfg.ring_chunks_per_axis();
    assert_eq!(r, (cfg.ring_bricks / chunk::CHUNK_BRICKS as u32) as i32);
    assert!(r >= 1, "ring must be at least one chunk wide");
}

/// The ring is centred on the camera: the camera's own chunk sits at the window's
/// middle (`origin + half`), so re-deriving the camera chunk from the world position
/// lands inside the window.
#[test]
fn ring_origin_centres_camera_chunk() {
    let cfg = config();
    let r = cfg.ring_chunks_per_axis();
    let half = r / 2;
    for lod in 0..cfg.lod_count {
        // A few world positions, including off-origin and negative.
        for cam in [
            Vec3::ZERO,
            Vec3::new(37.0, -12.0, 250.0),
            Vec3::new(-400.0, 8.0, -130.0),
        ] {
            let origin = ring_chunk_origin(&cfg, cam, lod);
            // Camera's chunk = origin + half on each axis.
            let cam_chunk = origin + IVec3::splat(half);
            assert!(
                chunk_in_window(cam_chunk, origin, r),
                "camera chunk must be inside its own ring (lod={lod}, cam={cam:?})"
            );
        }
    }
}

/// `chunk_in_window` is a half-open `[origin, origin+r)` box on every axis.
#[test]
fn chunk_in_window_boundaries() {
    let origin = IVec3::new(5, -2, 0);
    let r = 3;
    assert!(chunk_in_window(origin, origin, r), "corner is inside");
    assert!(chunk_in_window(origin + IVec3::splat(r - 1), origin, r), "far corner inside");
    assert!(!chunk_in_window(origin + IVec3::splat(r), origin, r), "one past is outside");
    assert!(!chunk_in_window(origin - IVec3::X, origin, r), "one before is outside");
}

/// `chunk_window_keys` yields exactly `r³` distinct keys, all inside the window and
/// all at the requested LOD.
#[test]
fn chunk_window_keys_cover_the_box() {
    use std::collections::HashSet;
    let origin = IVec3::new(-1, 4, 2);
    let r = 3;
    let lod = 2u32;
    let keys: Vec<_> = chunk_window_keys(origin, r, lod).collect();
    assert_eq!(keys.len(), (r * r * r) as usize, "must enumerate r^3 chunks");
    let set: HashSet<_> = keys.iter().map(|k| k.coord).collect();
    assert_eq!(set.len(), keys.len(), "no duplicate chunk coords");
    for k in &keys {
        assert_eq!(k.lod, lod);
        assert!(chunk_in_window(k.coord, origin, r));
    }
}

/// Each brick key a chunk emits maps back to that exact chunk + a unique local slot
/// 0..CHUNK_VOLUME — the round-trip the GPU resolve relies on.
#[test]
fn chunk_brick_keys_roundtrip_through_chunk_of() {
    use std::collections::HashSet;
    let cfg = config();
    let ck = chunk::ChunkKey::new(1, IVec3::new(-2, 0, 3));
    let bricks = chunk_brick_keys(ck, &cfg);
    assert_eq!(bricks.len(), chunk::CHUNK_VOLUME as usize);
    let mut locals = HashSet::new();
    for bk in &bricks {
        let (back, local) = chunk::chunk_of(*bk, &cfg);
        assert_eq!(back, ck, "brick must belong to the chunk that emitted it");
        assert!(local < chunk::CHUNK_VOLUME, "local slot in range");
        assert!(locals.insert(local), "each brick occupies a distinct local slot");
    }
    assert_eq!(locals.len(), chunk::CHUNK_VOLUME as usize, "all 64 slots covered");
}

/// A small edit AABB at the origin dirties the chunk(s) it overlaps at LOD 0, and the
/// chunk containing the origin is always among them (footprint pad ⇒ never misses).
#[test]
fn chunks_in_aabb_covers_origin_chunk() {
    let cfg = config();
    let aabb = Aabb3d::new(Vec3::ZERO, Vec3::splat(0.3));
    // A window comfortably containing the origin.
    let win = IVec3::splat(-8);
    let r = 16;
    let chunks = chunks_in_aabb_windowed(&cfg, &aabb, 0, win, r);
    assert!(!chunks.is_empty(), "an edit must dirty at least one chunk");
    let origin_chunk = chunk::chunk_of(atlas::BrickKey::new(0, IVec3::ZERO), &cfg).0;
    assert!(
        chunks.contains(&origin_chunk),
        "the origin's chunk must be in the dirtied set"
    );
    // All returned chunks are at the requested LOD.
    assert!(chunks.iter().all(|c| c.lod == 0));
}

/// The windowed clamp is the heightmap-freeze fix: a terrain-scale AABB (huge in XZ) must
/// only ever enumerate chunks INSIDE the window — never the millions of chunks its full
/// extent spans. Asserts the result is bounded by r³ and every chunk is in-window, even
/// though the AABB is vastly larger than the window.
#[test]
fn chunks_in_aabb_windowed_is_bounded_by_window() {
    let cfg = config();
    // A heightmap-like AABB: enormous in XZ, thin in Y.
    let aabb = Aabb3d::new(Vec3::ZERO, Vec3::new(100_000.0, 2.0, 100_000.0));
    let win = IVec3::splat(-8);
    let r = 16;
    let chunks = chunks_in_aabb_windowed(&cfg, &aabb, 0, win, r);
    assert!(
        chunks.len() <= (r * r * r) as usize,
        "windowed dirty set must be bounded by r³ = {}, got {}",
        r * r * r,
        chunks.len()
    );
    assert!(
        chunks.iter().all(|c| chunk_in_window(c.coord, win, r)),
        "every dirtied chunk must lie inside the window"
    );
    assert!(!chunks.is_empty(), "the AABB overlaps the window, so some chunks dirty");
}

/// A one-chunk camera shift exposes only a thin shell: the entered chunks are a face
/// of the window (`r²`), never the whole `r³` volume. This is what keeps incremental
/// recenter cheap (vs re-baking the full ring).
#[test]
fn one_chunk_shift_exposes_only_a_shell() {
    let r = 3;
    let old_origin = IVec3::ZERO;
    let new_origin = IVec3::new(1, 0, 0); // shift +1 chunk on X
    let entered = chunk_window_keys(new_origin, r, 0)
        .filter(|k| !chunk_in_window(k.coord, old_origin, r))
        .count();
    assert_eq!(entered, (r * r) as usize, "a 1-chunk shift enters exactly one r^2 face");
    assert!(entered < (r * r * r) as usize, "shell is far smaller than the volume");
}

/// Hysteresis: with `recenter_snap_chunks > 1`, camera motion that stays within one
/// snap cell must not move the ring origin at all — so no shell is entered/exited and
/// no rebake is triggered. The window only jumps when the snapped camera chunk changes.
#[test]
fn snap_holds_origin_within_a_snap_cell() {
    let snap = 4;
    let cfg = SdfGridConfig { recenter_snap_chunks: snap, ..config() };
    let chunk_world = chunk::chunk_world_size(0, &cfg);
    let lod = 0u32;

    // Origin at the world origin, then nudge the camera across most of a snap cell
    // (snap chunks wide) without crossing the next snap boundary.
    let base = ring_chunk_origin(&cfg, Vec3::ZERO, lod);
    let within = (snap as f32 - 0.5) * chunk_world; // just under one snap cell
    for d in [0.0, 0.25, 0.5, 0.9] {
        let cam = Vec3::new(within * d, 0.0, 0.0);
        assert_eq!(
            ring_chunk_origin(&cfg, cam, lod),
            base,
            "origin moved within a snap cell (cam={cam:?}); hysteresis not holding"
        );
    }

    // Crossing a full snap cell must move the origin by exactly `snap` chunks.
    let past = Vec3::new(snap as f32 * chunk_world + 0.1, 0.0, 0.0);
    let moved = ring_chunk_origin(&cfg, past, lod);
    assert_eq!(moved.x - base.x, snap, "a full snap-cell crossing shifts the origin by snap chunks");
}

// --- Hollow-shell `{native .. native+overlap}` residency -----------------------------------------

/// The chunk coord containing world point `p` at `lod` — the same path the bake/lookup use
/// (`world_to_brick_lod` → `chunk_of`), so the test's notion of "which chunk" matches production.
fn chunk_coord_at(cfg: &SdfGridConfig, p: Vec3, lod: u32) -> IVec3 {
    chunk::chunk_of(atlas::BrickKey::new(lod, cfg.world_to_brick_lod(p, lod)), cfg).0.coord
}

/// THE residency invariant (pure shell geometry). The resident shells at any point form a CONTIGUOUS
/// run starting at `native` (the finest LOD whose outer ring contains it):
/// - **NO GAP** — `native` is ALWAYS resident (never dropped into its own hole). This is the safety
///   property the renderer's fine→coarse `resolve_march` depends on; asserted exactly.
/// - **NO FULL STACK** — the run is short: `{native, native+1}` in the interior, with at most one extra
///   coarse level near a LOD/snap boundary (each LOD snaps its ring centre to its OWN chunk grid, so the
///   per-LOD holes aren't perfectly concentric and a point can sit just outside a coarse hole). Bounded
///   by `native + overlap_depth + 1` — vastly less than the old `lod_count`-deep stack.
#[test]
fn shell_residency_is_contiguous_from_native_no_gap() {
    let cfg = SdfGridConfig { lod_count: 6, ring_bricks: 128, recenter_snap_chunks: 1, ..config() };
    let r = cfg.ring_chunks_per_axis();
    let overlap = cfg.overlap_depth;
    let cams = [Vec3::ZERO, Vec3::new(13.0, 0.0, 0.0), Vec3::new(-7.0, 5.0, 21.0)];
    let dirs = [Vec3::X, Vec3::new(1.0, 1.0, 0.0).normalize(), Vec3::new(1.0, 1.0, 1.0).normalize()];
    let cw0 = chunk::chunk_world_size(0, &cfg);
    for cam in cams {
        for dir in dirs {
            for step in 0..440 {
                let d = step as f32 * 0.5 * cw0;
                let p = cam + dir * d;
                let in_outer = |lod: u32| chunk_in_window(chunk_coord_at(&cfg, p, lod), ring_chunk_origin(&cfg, cam, lod), r);
                let in_shell = |lod: u32| chunk_in_shell(&cfg, lod, chunk_coord_at(&cfg, p, lod), ring_chunk_origin(&cfg, cam, lod));
                let resident: Vec<u32> = (0..cfg.lod_count).filter(|&l| in_shell(l)).collect();
                let Some(native) = (0..cfg.lod_count).find(|&l| in_outer(l)) else {
                    // Beyond every ring → outside the clipmap: nothing resident.
                    assert!(resident.is_empty(), "cam={cam:?} p={p:?} outside all rings but resident at {resident:?}");
                    continue;
                };
                // No gap: native resident.
                assert!(resident.contains(&native), "cam={cam:?} p={p:?}: native LOD {native} NOT resident (GAP) — resident={resident:?}");
                // Contiguous run starting at native (no interior hole, never finer than native).
                let top = *resident.last().unwrap();
                assert_eq!(
                    resident, (native..=top).collect::<Vec<_>>(),
                    "cam={cam:?} p={p:?}: residency {resident:?} is not the contiguous run {native}..={top}"
                );
                // Bounded depth — no full stack. Interior is {native,native+1}; +1 slack at boundaries.
                assert!(
                    top <= native + overlap + 1,
                    "cam={cam:?} p={p:?}: resident up to LOD {top} > native+overlap+1 ({}) — stack too deep ({resident:?})",
                    native + overlap + 1
                );
            }
        }
    }
}

/// The SAME on the REAL baked atlas: after a full settle, no world region is resident at more than
/// `overlap_depth + 2` LODs — the redundant coarse stack a near surface used to carry (up to
/// `lod_count` ≈ 6 levels) is gone, down to ~2 (+1 boundary slack). Fails loudly if the shell cull is
/// ever dropped (a near surface would light up at every LOD again).
#[test]
fn settled_atlas_holds_no_redundant_lod_stack() {
    let cfg = SdfGridConfig { lod_count: 6, ring_bricks: 128, recenter_snap_chunks: 1, ..config() };
    // A big sphere + a camera outside it, so its surface spans many LOD distance bands (near side
    // ~LOD0, far side coarse) — the case where the old full-stack residency was worst.
    let edits = vec![sphere_edit(Vec3::ZERO, 50.0, 0)];
    let mut sched = primed_sched(&edits);
    let mut atlas = SdfAtlas::default();
    settle_gpu(&mut sched, &mut atlas, &cfg, Vec3::new(60.0, 0.0, 0.0));
    assert!(!atlas.bricks.is_empty(), "setup: sphere should bake some bricks");

    let max_lods = cfg.overlap_depth + 2;
    for key in atlas.bricks.keys() {
        let bw = cfg.brick_world_size(key.lod);
        let center = cfg.brick_min_world(key.coord, key.lod) + Vec3::splat(0.5 * bw);
        let n = (0..cfg.lod_count)
            .filter(|&l| atlas.bricks.contains_key(&atlas::BrickKey::new(l, cfg.world_to_brick_lod(center, l))))
            .count() as u32;
        assert!(
            n <= max_lods,
            "region of brick {key:?} is resident at {n} LODs > native+overlap+1 ({max_lods}) — \
             the redundant LOD stack is NOT being culled"
        );
    }
}

/// Approaching the surface makes a region's coarse LOD redundant (a finer LOD now covers it) — and the
/// recenter must EVICT it, not leave the full stack behind. Settle far (region served coarse), then
/// settle near (region served fine) and assert the once-resident coarse brick is gone.
#[test]
fn approaching_evicts_now_redundant_coarse_lod() {
    let cfg = SdfGridConfig { lod_count: 6, ring_bricks: 64, recenter_snap_chunks: 1, ..config() };
    let edits = vec![sphere_edit(Vec3::ZERO, 50.0, 0)];
    let mut sched = primed_sched(&edits);
    let mut atlas = SdfAtlas::default();

    // Surface probe on the +X cap. Far camera ⇒ it's served by a coarse LOD; close camera ⇒ fine.
    let probe = Vec3::new(50.0, 0.0, 0.0);
    settle_gpu(&mut sched, &mut atlas, &cfg, Vec3::new(300.0, 0.0, 0.0));
    let far_lod = served_lod(&atlas, &cfg, probe).expect("probe covered when far");

    settle_gpu(&mut sched, &mut atlas, &cfg, Vec3::new(54.0, 0.0, 0.0));
    let near_lod = served_lod(&atlas, &cfg, probe).expect("probe still covered when near");
    assert!(near_lod < far_lod, "approaching should serve a FINER LOD (far={far_lod} near={near_lod})");

    // The region must now hold at most {native, native+1} — the old far coarse LOD is evicted unless
    // it happens to be within one level of the new native.
    let resident: Vec<u32> = (0..cfg.lod_count)
        .filter(|&l| atlas.bricks.contains_key(&atlas::BrickKey::new(l, cfg.world_to_brick_lod(probe, l))))
        .collect();
    assert!(
        resident.len() as u32 <= cfg.overlap_depth + 1,
        "after approach the probe is resident at {resident:?} ({} LODs) — redundant coarse LOD not evicted",
        resident.len()
    );
    assert!(
        far_lod > near_lod + cfg.overlap_depth,
        "test too weak: far/near LODs ({far_lod}/{near_lod}) within the overlap band — pick a bigger move"
    );
    assert!(
        !resident.contains(&far_lod),
        "the now-redundant far LOD {far_lod} is still resident at the probe — not evicted"
    );
}

// --- Frustum / proximity bake priority -----------------------------------------------------------

/// A frustum whose only restrictive plane is "z ≥ 0" (the other five always pass), so `out_rank`
/// flags a chunk as out-of-view exactly when its bounding sphere is fully behind the camera.
fn forward_z_frustum() -> FrustumPlanes {
    let pass = Vec4::new(0.0, 0.0, 0.0, 1.0e9); // dot(p,1) = 1e9 > 0 ⇒ always inside
    FrustumPlanes([Vec4::new(0.0, 0.0, 1.0, 0.0), pass, pass, pass, pass, pass])
}

/// The frustum FLIPS the in-LOD order: a FARTHER but in-view chunk beats a NEARER off-screen one. But
/// coarse-LOD-first is NEVER overridden by the frustum (the hole-free fallback ordering the
/// chunk-atomic bake relies on). Without a frustum it's pure distance (nearest first).
#[test]
fn priority_is_coarse_then_in_view_then_near() {
    let cfg = config();
    let view = BakeView { pos: Vec3::ZERO, fwd: Vec3::Z, frustum: Some(forward_z_frustum()), margin: 0.0 };

    // Front: far (+Z, in view). Back: near (−Z, off screen). Same LOD.
    let front = chunk::ChunkKey::new(0, IVec3::new(0, 0, 10));
    let back = chunk::ChunkKey::new(0, IVec3::new(0, 0, -4));
    assert!(
        chunk_priority_key(front, &cfg, &view) < chunk_priority_key(back, &cfg, &view),
        "with a frustum, a farther IN-VIEW chunk must outrank a nearer OFF-SCREEN one at the same LOD"
    );
    // Without a frustum, the same pair orders by distance — the nearer (back) wins.
    let nofrustum = BakeView::pos_only(Vec3::ZERO);
    assert!(
        chunk_priority_key(back, &cfg, &nofrustum) < chunk_priority_key(front, &cfg, &nofrustum),
        "without a frustum, ordering is distance-only — the nearer chunk sorts first"
    );

    // Coarse out-of-view vs fine in-view: coarse must STILL sort first (frustum never outranks LOD).
    let coarse_back = chunk::ChunkKey::new(2, IVec3::new(0, 0, -8));
    let fine_front = chunk::ChunkKey::new(0, IVec3::new(0, 0, 8));
    assert!(
        chunk_priority_key(coarse_back, &cfg, &view) < chunk_priority_key(fine_front, &cfg, &view),
        "coarse LOD must outrank a finer LOD regardless of frustum (fallback-coverage ordering)"
    );
}

// --- Conservative empty-space occupancy (no skip-past) -------------------------------------------

/// THE no-skip-past guarantee: the empty-space DDA reads `cons_occ`, so if it is a SUPERSET of the
/// baked `occ`, the DDA can never step over a SAMPLED surface. Settle a multi-LOD sphere and assert
/// every resident (baked) brick's bit is set in its chunk's conservative mask.
#[test]
fn conservative_mask_covers_every_baked_brick() {
    let cfg = SdfGridConfig { lod_count: 6, ring_bricks: 128, recenter_snap_chunks: 1, ..config() };
    let edits = vec![sphere_edit(Vec3::ZERO, 50.0, 0)];
    let mut sched = primed_sched(&edits);
    let mut atlas = SdfAtlas::default();
    settle_gpu(&mut sched, &mut atlas, &cfg, Vec3::new(60.0, 0.0, 0.0));
    assert!(!atlas.bricks.is_empty(), "setup: sphere should bake bricks");
    for key in atlas.bricks.keys() {
        let (ck, local) = chunk::chunk_of(*key, &cfg);
        let cons = atlas.live_chunks.conservative_mask(ck);
        assert!(
            cons & (1u64 << local) != 0,
            "baked brick {key:?} (chunk {ck:?} local {local}) NOT in conservative mask {cons:#018x} — \
             the DDA could skip past a sampled surface"
        );
    }
}

/// A feature THINNER than a coarse voxel: the baked coarse occupancy can read empty (LOD shrinkage
/// erodes it away), but the conservative mask is built from the geometry's AABB via the BVH, so it
/// MUST still catch it — otherwise the empty-space DDA skips straight past the thin feature (the
/// "coarse-empty ≠ fine-empty" bug). Assert a sub-voxel sphere lights ≥1 conservative bit at a coarse LOD.
#[test]
fn conservative_mask_catches_sub_voxel_feature() {
    let cfg = SdfGridConfig { lod_count: 6, ring_bricks: 128, recenter_snap_chunks: 1, ..config() };
    let tiny = sphere_edit(Vec3::new(0.0, 3.3, 0.0), 0.05, 0); // r=0.05 ≪ a LOD-4 voxel (≈1.6)
    let bvh = build_bvh(std::slice::from_ref(&tiny));
    let aabb = edit_world_aabb(&tiny.prim, &tiny.transform, 0.0);
    let lod = 4u32;
    let origin = ring_chunk_origin(&cfg, Vec3::ZERO, lod);
    let r = cfg.ring_chunks_per_axis();
    let edits = std::slice::from_ref(&tiny);
    let mut scratch: Vec<u32> = Vec::new();
    let mut stack: Vec<u32> = Vec::new();
    let any = bricks_in_aabb_windowed(&cfg, &aabb, lod, origin, r)
        .into_iter()
        .any(|(ck, _)| chunk_conservative_mask(ck, edits, &bvh, &cfg, &mut scratch, &mut stack) != 0);
    assert!(
        any,
        "coarse-LOD conservative mask missed a sub-voxel feature — the empty-space DDA would skip past it"
    );
}

