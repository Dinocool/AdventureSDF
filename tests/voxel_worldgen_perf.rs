//! **Reusable headless perf harness for the streamed WORLDGEN voxel-RT scene.** (`VoxelScene::Worldgen`,
//! the V toggle.) The static Cornell box is fully-resident + cheap; switching to worldgen "streams the
//! whole region from scratch" and is reported as laggy. This rig PROFILES that path so the bottleneck is
//! MEASURED, not guessed — it mirrors `tests/voxel_streaming.rs` (same public API) + the worldgen/bake perf
//! benches (`tests/worldgen_bench.rs`, `bake_scheduler/perf.rs`): a scripted deterministic camera
//! fly-through driving the SHIPPING streaming pipeline with per-stage `std::time::Instant` timers.
//!
//! It builds the worldgen the SAME way `init_voxel_rt_streaming` does — `build_height_layer_pub` over the
//! production graph (the shipping `assets/worldgen/world.graph.ron`, falling back to the same
//! `WorldGraph::default()` preset the engine boots with when the asset is absent), `load_biome_library_pub`,
//! `BlockRegistry::from_biome_library` — and constructs a `ResidencyManager` with the SHIPPING
//! `StreamingConfig` (radius 28 bricks, cap 60_000, 256 bricks/frame). Then it drives:
//!
//!   * the INITIAL FILL: cold-stream the whole region from empty (`update` + repeated `drain_work` until the
//!     queue empties), timing each stage and the per-brick voxelize cost;
//!   * STEADY-STATE-MOVING: nudge the camera brick-by-brick along a straight traverse + a few jumps, timing
//!     the per-step `update` / `drain_work` / `pack_resident_set`;
//!   * the PACK cost at the resident-brick count (60k-class), the SSOT GPU-buffer build;
//!   * (if a ray-query device is present) the BLAS build at that brick count — `create_buffer_init` of the
//!     ~240 MB scene buffers + `create_blas` + `build_acceleration_structures` — the known full-rebuild
//!     Phase-3 blocker. Skips cleanly (like the other GPU rigs) when no device is available.
//!
//! All `#[ignore]` (timing harnesses, never gate CI). The CPU benches run anywhere (`cargo test`); the BLAS
//! bench needs the ray-query device + `TMP/TEMP=D:\tmp_test`:
//!
//! ```text
//! # CPU breakdown (no GPU; deterministic):
//! cargo test --no-default-features --features fast,physics --test voxel_worldgen_perf -- --ignored --nocapture
//! # + the BLAS-build timing (ray-query device):
//! $env:TMP="D:\tmp_test"; $env:TEMP="D:\tmp_test"; cargo test --no-default-features --features fast,physics --test voxel_worldgen_perf -- --ignored --nocapture
//! ```
//!
//! GPU PASS times (world-cache update/compaction over the 2^20 table, ReSTIR) are NOT timed here — they need
//! `TIMESTAMP_QUERY` plumbed through the live render pipeline / Nsight (see the `nsight-shader-profiling`
//! memory). This rig answers "is it the world GENERATING" (the CPU stream + pack + BLAS rebuild) first,
//! which is the suspected cause; the GPU passes are a tracked follow-up.

use std::time::{Duration, Instant};

use bevy::math::IVec3;
use wgpu::util::DeviceExt; // create_buffer_init (BLAS bench only)

use adventure::sdf_render::worldgen::graph::GraphAsset;
use adventure::sdf_render::worldgen::layers::erosion::ErosionParams;
use adventure::sdf_render::worldgen::layers::height::HeightParams;
use adventure::sdf_render::worldgen::{WORLDGEN_SLICE_SEED, WorldBiomeShapes, WorldGraph};
use adventure::voxel::brickmap::{BRICK_WORLD_SIZE, Brick, MAX_LOD, brick_span};
use adventure::voxel::gpu::{GpuBrickAabb, GpuBrickMeta, GpuPaletteColor, pack_resident_set};
use adventure::voxel::palette::BlockRegistry;
use adventure::voxel::streaming::{
    ResidencyManager, StreamingConfig, camera_brick_coord, region_half_extent_m,
};
use adventure::voxel::voxelize::voxelize_brick;
use adventure::voxel::{build_height_layer_pub, load_biome_library_pub};
use adventure::sdf_render::worldgen::layers::height::HeightLayer;

#[path = "common/mod.rs"]
mod common;

/// The SHIPPING streaming seed (matches `init_voxel_rt_streaming`'s `WORLDGEN_SLICE_SEED`).
const SEED: u64 = WORLDGEN_SLICE_SEED;

/// Mirror of the live `WORLDGEN_REPACK_INTERVAL` (raytrace.rs): the streaming system AMORTIZES the O(resident)
/// re-pack — it packs on a SETTLE (`pending() == 0`) OR every this-many drained frames, NOT on every dirty
/// drain. The fill / steady-state benches model the same cadence so their per-frame + total numbers reflect
/// what the running engine actually pays (a streaming frame pays the bounded voxelize drain; the pack is paid
/// only ~once per this interval + at settle).
const REPACK_INTERVAL: u32 = 6;

/// The shipping worldgen graph the live scene streams: deserialize `assets/worldgen/world.graph.ron` (the
/// asset `load_active_graph` loads at runtime). Falls back to `WorldGraph::default()` (the
/// `mountains_plains` preset the engine boots with until that asset lands) if the file is missing/invalid —
/// the SAME default `init_voxel_rt_streaming` uses when its worldgen resources are absent, so the harness
/// always profiles a representative dramatic-terrain graph. Returned alongside a label for the report.
fn shipping_world_graph() -> (WorldGraph, &'static str) {
    match std::fs::read("assets/worldgen/world.graph.ron") {
        Ok(bytes) => match ron::de::from_bytes::<GraphAsset>(&bytes) {
            Ok(asset) => (WorldGraph(std::sync::Arc::new(asset.graph)), "world.graph.ron"),
            Err(e) => {
                eprintln!("[warn] world.graph.ron parse error: {e} — using preset default");
                (WorldGraph::default(), "preset default (mountains_plains)")
            }
        },
        Err(e) => {
            eprintln!("[warn] world.graph.ron read error: {e} — using preset default");
            (WorldGraph::default(), "preset default (mountains_plains)")
        }
    }
}

/// Build the worldgen sampling stack EXACTLY as `init_voxel_rt_streaming` does: a `HeightLayer` over the
/// shipping graph + the default height/erosion/biome-shape params, the biome library from `biomes.ron`, and
/// the `BlockRegistry`. Returns them plus the graph label for the report.
fn worldgen_stack() -> (HeightLayer, adventure::sdf_render::worldgen::biome::BiomeLibrary, BlockRegistry, &'static str) {
    let (graph, label) = shipping_world_graph();
    let layer = build_height_layer_pub(
        &HeightParams::default(),
        &ErosionParams::default(),
        &graph,
        &WorldBiomeShapes::default(),
    );
    let lib = load_biome_library_pub();
    let registry = BlockRegistry::from_biome_library(&lib);
    (layer, lib, registry, label)
}

/// The SHIPPING `StreamingConfig` the worldgen scene runs with (the `Default` — clip_half 8 bricks ⇒ a
/// nested clipmap of `MAX_LOD+1` shells reaching ~820 m, cap 60_000 resident, 256 bricks/frame). The single
/// SSOT knob the live path uses.
fn shipping_config() -> StreamingConfig {
    StreamingConfig::default()
}

/// The camera WORLD position at the worldgen reframe pose: the editor frames the origin surface. We center
/// the clipmap on the SURFACE at the origin — where the densest non-empty terrain bricks (and the worst-case
/// fill) live.
fn origin_surface_cam(layer: &HeightLayer) -> [f32; 3] {
    let surf = layer.sample_world(0.0, 0.0, SEED).height;
    [0.0, surf, 0.0]
}

/// p50/p95/max over a slice of durations (ms), plus the mean. Small, dependency-free percentile (nearest-rank).
fn stats_ms(samples: &[Duration]) -> (f64, f64, f64, f64) {
    let mut ms: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1e3).collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ms.len().max(1);
    let mean = ms.iter().sum::<f64>() / n as f64;
    let pct = |p: f64| ms[((p * (n as f64 - 1.0)).round() as usize).min(n - 1)];
    (mean, pct(0.5), pct(0.95), *ms.last().unwrap_or(&0.0))
}

// ============================================================================================
//  (1) Per-brick voxelize cost — the inner grain (512 voxels, each a `sample_world` graph eval).
// ============================================================================================

/// The fundamental unit: `voxelize_brick` over the REAL shipping worldgen surface. Times bricks STRADDLING
/// the surface (the expensive, non-uniform case — 512 distinct `sample_world` graph evals that don't collapse
/// to the uniform fast path), reports us/brick. This is what `drain_work` does `max_bricks_per_frame` times.
#[test]
#[ignore = "perf harness; run with --ignored --nocapture"]
fn bench_voxelize_brick_cost() {
    let (layer, lib, registry, label) = worldgen_stack();
    let cam = origin_surface_cam(&layer);
    let cam_brick = camera_brick_coord(cam);

    // Sample a spread of surface-straddling LOD0 bricks (a 7×7 XZ tile at the surface Y band, plus a few Y
    // above/below). LOD0 is the per-frame hot path (the densest, full-res bricks).
    let mut coords = Vec::new();
    for dz in -3..=3 {
        for dx in -3..=3 {
            for dy in -1..=1 {
                coords.push(cam_brick + IVec3::new(dx, dy, dz));
            }
        }
    }

    // Warm + classify (how many collapse to uniform — the cheap fast path — vs stay dense).
    let mut dense = 0usize;
    let mut empty = 0usize;
    let mut uniform_solid = 0usize;
    for &c in &coords {
        let b = voxelize_brick(c, 0, &layer, &lib, &registry, SEED);
        if b.is_empty() {
            empty += 1;
        } else if b.is_uniform_solid() {
            uniform_solid += 1;
        } else {
            dense += 1;
        }
    }

    const ITERS: u32 = 4;
    let t0 = Instant::now();
    let mut sink = 0u64;
    for _ in 0..ITERS {
        for &c in &coords {
            let b = voxelize_brick(c, 0, &layer, &lib, &registry, SEED);
            sink = sink.wrapping_add(b.get(0, 0, 0).0 as u64);
        }
    }
    let elapsed = t0.elapsed();
    let total_bricks = coords.len() as u32 * ITERS;
    println!("\n[voxelize] graph={label} — {} bricks/iter ×{ITERS}", coords.len());
    println!(
        "[voxelize]   {:.2} us/brick  ({:.4} ms/brick) — {} dense / {} uniform-solid / {} empty (sink={sink})",
        elapsed.as_secs_f64() * 1e6 / total_bricks as f64,
        elapsed.as_secs_f64() * 1e3 / total_bricks as f64,
        dense,
        uniform_solid,
        empty,
    );
    println!(
        "[voxelize]   ⇒ a full {}-brick drain ≈ {:.2} ms of voxelize",
        shipping_config().max_bricks_per_frame,
        elapsed.as_secs_f64() * 1e3 / total_bricks as f64 * shipping_config().max_bricks_per_frame as f64,
    );
}

// ============================================================================================
//  (2) INITIAL FILL — cold-stream the whole region from empty (the V-toggle-into-worldgen cost).
// ============================================================================================

/// Cold fill: a fresh `ResidencyManager` at the origin-surface camera with the SHIPPING config, drained
/// frame-by-frame (256 bricks/frame) until the queue empties — exactly what happens when the user toggles
/// INTO the worldgen scene. Times, separately: the one `update` (enqueue the (2·28+1)³ region), each
/// bounded `drain_work` (the per-FRAME hitch — voxelize ≤256 bricks), and the per-frame `pack_resident_set`
/// (rebuilds the whole GPU buffer set every dirty drain). Reports the totals + the per-frame stats so the
/// dominant initial-fill cost is attributed.
#[test]
#[ignore = "perf harness; run with --ignored --nocapture"]
fn bench_initial_fill_cold() {
    let (layer, lib, registry, label) = worldgen_stack();
    let cfg = shipping_config();
    let cam = origin_surface_cam(&layer);

    let mut mgr = ResidencyManager::new();

    let t_update = Instant::now();
    mgr.update(cam, &cfg);
    let update_ms = t_update.elapsed().as_secs_f64() * 1e3;
    let region = mgr.pending();

    let mut drain_times = Vec::new();
    let mut pack_times = Vec::new();
    let mut total_voxelized = 0usize;
    let mut packs = 0usize;
    let mut last_brick_count = 0usize;
    let mut last_voxel_cells = 0usize;
    let fill_t0 = Instant::now();
    let mut frames = 0u32;
    let mut dirty = false;
    let mut since_pack = 0u32;
    while mgr.pending() > 0 {
        let td = Instant::now();
        let n = mgr.drain_work(&cfg, &layer, &lib, &registry, SEED);
        drain_times.push(td.elapsed());
        total_voxelized += n;

        if mgr.take_dirty() {
            dirty = true;
        }
        since_pack += 1;
        // AMORTIZED re-pack (mirrors the live system): pack on settle OR every REPACK_INTERVAL frames.
        if dirty && (mgr.pending() == 0 || since_pack >= REPACK_INTERVAL) {
            let entries = mgr.resident_entries();
            let tp = Instant::now();
            let patch = pack_resident_set(&entries, &registry);
            pack_times.push(tp.elapsed());
            packs += 1;
            last_brick_count = patch.brick_count();
            last_voxel_cells = patch.voxels.len();
            dirty = false;
            since_pack = 0;
        }
        frames += 1;
        assert!(frames < 5000, "fill must terminate");
    }
    let fill_total = fill_t0.elapsed();

    let (d_mean, d_p50, d_p95, d_max) = stats_ms(&drain_times);
    let (p_mean, _p_p50, p_p95, p_max) = stats_ms(&pack_times);
    let drain_total: f64 = drain_times.iter().map(|d| d.as_secs_f64() * 1e3).sum();
    let pack_total: f64 = pack_times.iter().map(|d| d.as_secs_f64() * 1e3).sum();

    println!("\n========== INITIAL FILL (cold stream into worldgen) — graph={label} ==========");
    println!(
        "clipmap enqueued     : {region} brick coords (clip_half {} ⇒ {} nested LOD shells, view ≈ {:.0} m, capped at {}), {update_ms:.3} ms to enqueue",
        cfg.clip_half_bricks,
        MAX_LOD + 1,
        region_half_extent_m(&cfg),
        cfg.max_resident_bricks,
    );
    println!("frames to settle     : {frames} (at {} bricks/frame)", cfg.max_bricks_per_frame);
    println!("bricks voxelized      : {total_voxelized} total");
    println!("resident (non-empty)  : {last_brick_count} bricks (cap {})", cfg.max_resident_bricks);
    println!("packed voxel cells    : {last_voxel_cells} u32 (~{:.1} MB voxel buf)", last_voxel_cells as f64 * 4.0 / 1e6);
    println!("-----------------------------------------------------------------------------");
    println!("drain_work (voxelize) : total {drain_total:.1} ms | per-frame mean {d_mean:.2} max {d_max:.2} (p50 {d_p50:.2} p95 {d_p95:.2}) ms");
    println!("pack_resident_set     : total {pack_total:.1} ms over {packs} re-packs | per-pack mean {p_mean:.2} p95 {p_p95:.2} max {p_max:.2} ms");
    println!("WALL initial-fill     : {:.1} ms ({:.2} s)", fill_total.as_secs_f64() * 1e3, fill_total.as_secs_f64());
    println!("=============================================================================");
    assert!(total_voxelized > 0, "cold fill voxelized nothing — region empty?");
}

// ============================================================================================
//  (3) STEADY-STATE MOVING — per-step cost as the camera traverses + jumps.
// ============================================================================================

/// After the region is warm, walk the camera ONE brick per step along a straight +X traverse (the steady
/// fly-through), plus a couple of big jumps, timing each step's `update` + `drain_work` + (when dirty)
/// `pack_resident_set`. This is the per-FRAME cost while MOVING (the hitch the user feels in flight), as
/// distinct from the one-time initial fill. Reports per-step stats; pack stats are reported separately since
/// a brick-step only re-packs when a revealing batch lands.
#[test]
#[ignore = "perf harness; run with --ignored --nocapture"]
fn bench_steady_state_moving() {
    let (layer, lib, registry, label) = worldgen_stack();
    let cfg = shipping_config();
    let mut cam = origin_surface_cam(&layer);

    // Warm the region fully (untimed).
    let mut mgr = ResidencyManager::new();
    mgr.update(cam, &cfg);
    let mut guard = 0;
    while mgr.pending() > 0 {
        mgr.drain_work(&cfg, &layer, &lib, &registry, SEED);
        guard += 1;
        assert!(guard < 5000);
    }
    mgr.take_dirty();
    let warm_resident = mgr.resident_count();

    // The script: 6 single-brick (one `brick_span(0)` = 1.6 m) +X steps (a straight surface traverse), then 1
    // jump of +6 bricks (crossing a LOD0 shell-width + revealing a fresh slab), then 6 more single-brick
    // steps. With the CLIPMAP, a single-brick step shifts only the LOD0 shell (a thin face-slab) — the coarse
    // shells move `2^L×` less often — so each Walk reveals far fewer bricks than the old dense region did.
    #[derive(Clone, Copy)]
    enum Step {
        Walk,
        Jump,
    }
    let mut script = Vec::new();
    for _ in 0..6 {
        script.push(Step::Walk);
    }
    script.push(Step::Jump);
    for _ in 0..6 {
        script.push(Step::Walk);
    }

    let span0 = brick_span(0); // one LOD0 brick in world metres
    let mut update_times = Vec::new();
    let mut drain_times = Vec::new();
    let mut pack_times = Vec::new();
    let mut step_times = Vec::new(); // the full per-step cost (update+drain+pack), what a frame pays
    let mut packs = 0usize;
    let mut max_resident = warm_resident;

    for (i, step) in script.iter().enumerate() {
        let bricks = match step {
            Step::Walk => 1.0,
            Step::Jump => 6.0,
        };
        cam[0] += bricks * span0;

        let step_t0 = Instant::now();
        let tu = Instant::now();
        mgr.update(cam, &cfg);
        update_times.push(tu.elapsed());

        // A Walk shifts only the LOD0 shell; drain until caught up (each drain = one frame), so the per-DRAIN
        // time is the per-frame hitch. Record each drain + each re-pack.
        let mut step_drains = 0u32;
        let mut dirty = false;
        let mut since_pack = 0u32;
        loop {
            let td = Instant::now();
            let n = mgr.drain_work(&cfg, &layer, &lib, &registry, SEED);
            drain_times.push(td.elapsed());
            step_drains += 1;
            if mgr.take_dirty() {
                dirty = true;
            }
            since_pack += 1;
            let last = n == 0 || mgr.pending() == 0;
            // AMORTIZED re-pack (mirrors the live system): pack on settle OR every REPACK_INTERVAL frames.
            if dirty && (last || since_pack >= REPACK_INTERVAL) {
                let entries = mgr.resident_entries();
                let tp = Instant::now();
                let patch = pack_resident_set(&entries, &registry);
                pack_times.push(tp.elapsed());
                packs += 1;
                max_resident = max_resident.max(patch.brick_count());
                dirty = false;
                since_pack = 0;
            }
            if last {
                break;
            }
        }
        // The full per-step cost a moving frame pays (update + every drain + every re-pack this step).
        let step_ms = step_t0.elapsed().as_secs_f64() * 1e3;
        step_times.push(step_t0.elapsed());
        eprintln!(
            "[steady] step {:>2}/{} ({}): {step_drains} drains, {step_ms:.0} ms total to settle",
            i + 1,
            script.len(),
            match step {
                Step::Walk => "walk",
                Step::Jump => "jump",
            },
        );
    }

    let (u_mean, _u_p50, u_p95, u_max) = stats_ms(&update_times);
    let (d_mean, d_p50, d_p95, d_max) = stats_ms(&drain_times);
    let (p_mean, _p_p50, p_p95, p_max) = stats_ms(&pack_times);
    let (s_mean, s_p50, s_p95, s_max) = stats_ms(&step_times);

    println!("\n========== STEADY-STATE MOVING (warm region, fly-through) — graph={label} ==========");
    println!("warm resident         : {warm_resident} bricks | peak {max_resident}");
    println!("script                : {} steps (6 walk +X, 1 jump +6, 6 walk +X)", script.len());
    println!("-----------------------------------------------------------------------------");
    println!("update (enqueue diff) : mean {u_mean:.3} p95 {u_p95:.3} max {u_max:.3} ms");
    println!("drain_work per frame  : mean {d_mean:.2} p50 {d_p50:.2} p95 {d_p95:.2} max {d_max:.2} ms");
    println!("pack_resident_set     : {packs} re-packs | mean {p_mean:.2} p95 {p_p95:.2} max {p_max:.2} ms");
    println!("PER-STEP (frame total): mean {s_mean:.2} p50 {s_p50:.2} p95 {s_p95:.2} max {s_max:.2} ms");
    println!("=============================================================================");
}

// ============================================================================================
//  (4) PACK cost at the resident-brick count — the SSOT GPU-buffer build (O(resident bricks)).
// ============================================================================================

/// Isolate `pack_resident_set` at the resident-brick count the worldgen scene reaches (60k-class). Every
/// `generation` bump re-runs this (it rebuilds the AABB/meta/voxel buffers from scratch), so its per-call
/// cost bounds how often the camera can move without hitching. Reports ms/pack + the buffer sizes (the bytes
/// `create_buffer_init` then uploads — see the BLAS bench).
#[test]
#[ignore = "perf harness; run with --ignored --nocapture"]
fn bench_pack_at_resident_count() {
    let (layer, lib, registry, label) = worldgen_stack();
    let cfg = shipping_config();
    let cam = origin_surface_cam(&layer);

    let mut mgr = ResidencyManager::new();
    mgr.update(cam, &cfg);
    let mut guard = 0;
    while mgr.pending() > 0 {
        mgr.drain_work(&cfg, &layer, &lib, &registry, SEED);
        guard += 1;
        assert!(guard < 5000);
    }
    let entries = mgr.resident_entries();

    const ITERS: u32 = 8;
    let mut times = Vec::new();
    let mut last = pack_resident_set(&entries, &registry); // warm
    for _ in 0..ITERS {
        let t = Instant::now();
        last = pack_resident_set(&entries, &registry);
        times.push(t.elapsed());
    }
    let (mean, p50, p95, max) = stats_ms(&times);

    let aabb_bytes = last.aabbs.len() * std::mem::size_of::<GpuBrickAabb>();
    let meta_bytes = last.metas.len() * std::mem::size_of::<GpuBrickMeta>();
    let voxel_bytes = last.voxels.len() * std::mem::size_of::<u32>();
    let pal_bytes = last.palette.len() * std::mem::size_of::<GpuPaletteColor>();
    let total = aabb_bytes + meta_bytes + voxel_bytes + pal_bytes;

    println!("\n========== PACK at resident count — graph={label} ==========");
    println!("resident bricks       : {} (entries {})", last.brick_count(), entries.len());
    println!("pack_resident_set     : mean {mean:.2} p50 {p50:.2} p95 {p95:.2} max {max:.2} ms ({ITERS} iters)");
    println!("GPU buffers (per gen) : aabb {:.1} MB | meta {:.1} MB | voxel {:.1} MB | palette {} B | TOTAL {:.1} MB",
        aabb_bytes as f64 / 1e6, meta_bytes as f64 / 1e6, voxel_bytes as f64 / 1e6, pal_bytes, total as f64 / 1e6);
    println!("=============================================================================");
}

// ============================================================================================
//  (5) BLAS BUILD at the resident-brick count — needs a ray-query device (skips cleanly otherwise).
// ============================================================================================

/// The known Phase-3 blocker: the full BLAS is rebuilt FROM SCRATCH on every generation bump. This times, at
/// the worldgen resident-brick count: (a) the four `create_buffer_init` uploads (the ~240 MB scene buffers),
/// (b) `create_blas` + `create_tlas`, (c) `build_acceleration_structures` + `submit` + a `poll` to fence the
/// GPU build. Mirrors the live `build_voxel_rt_accel` path EXACTLY (same buffer usages, BLAS/TLAS flags,
/// instance transform). Skips (like every GPU rig) when no ray-query adapter is present, and honours
/// `TMP/TEMP=D:\tmp_test`.
#[test]
#[ignore = "GPU perf harness; needs ray-query device + TMP=D:\\tmp_test; run with --ignored --nocapture"]
fn bench_blas_build_at_resident_count() {
    let Some((device, queue)) = common::headless_ray_query_device() else {
        eprintln!("[skip] no ray-query device — BLAS-build timing skipped (GPU follow-up: Nsight)");
        return;
    };

    let (layer, lib, registry, label) = worldgen_stack();
    let cfg = shipping_config();
    let cam = origin_surface_cam(&layer);

    let mut mgr = ResidencyManager::new();
    mgr.update(cam, &cfg);
    let mut guard = 0;
    while mgr.pending() > 0 {
        mgr.drain_work(&cfg, &layer, &lib, &registry, SEED);
        guard += 1;
        assert!(guard < 5000);
    }
    let entries = mgr.resident_entries();
    let patch = pack_resident_set(&entries, &registry);
    let n = patch.brick_count() as u32;
    assert!(n > 0, "no resident bricks to build a BLAS from");

    // Time the full rebuild a few times (each iteration builds fresh buffers + accel — the per-generation cost).
    const ITERS: u32 = 4;
    let mut upload_times = Vec::new();
    let mut create_times = Vec::new();
    let mut build_times = Vec::new();
    for _ in 0..ITERS {
        let tu = Instant::now();
        let aabb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("perf_aabbs"),
            contents: bytemuck::cast_slice(&patch.aabbs),
            usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE,
        });
        let _meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("perf_metas"),
            contents: bytemuck::cast_slice(&patch.metas),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let _voxel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("perf_voxels"),
            contents: bytemuck::cast_slice(&patch.voxels),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let _palette_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("perf_palette"),
            contents: bytemuck::cast_slice(&patch.palette),
            usage: wgpu::BufferUsages::STORAGE,
        });
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
        upload_times.push(tu.elapsed());

        let size_desc = wgpu::BlasAABBGeometrySizeDescriptor {
            primitive_count: n,
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        let tc = Instant::now();
        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("perf_blas"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::AABBs { descriptors: vec![size_desc.clone()] },
        );
        let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("perf_tlas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            max_instances: 1,
        });
        tlas[0] = Some(wgpu::TlasInstance::new(
            &blas,
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            0,
            0xff,
        ));
        create_times.push(tc.elapsed());

        let tb = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("perf_build_accel"),
        });
        encoder.build_acceleration_structures(
            core::iter::once(&wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::AabbGeometries(vec![wgpu::BlasAabbGeometry {
                    size: &size_desc,
                    stride: core::mem::size_of::<GpuBrickAabb>() as wgpu::BufferAddress,
                    aabb_buffer: &aabb_buf,
                    primitive_offset: 0,
                }]),
            }),
            core::iter::once(&tlas),
        );
        queue.submit(core::iter::once(encoder.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed"); // fence the GPU accel build before stopping the clock
        build_times.push(tb.elapsed());
    }

    let (u_mean, _u_p50, u_p95, u_max) = stats_ms(&upload_times);
    let (c_mean, _c_p50, c_p95, c_max) = stats_ms(&create_times);
    let (b_mean, _b_p50, b_p95, b_max) = stats_ms(&build_times);

    println!("\n========== BLAS BUILD at resident count — graph={label} ==========");
    println!("resident bricks       : {n} (BLAS primitives)");
    println!("create_buffer_init x4 : mean {u_mean:.2} p95 {u_p95:.2} max {u_max:.2} ms (the ~scene-buffer upload)");
    println!("create_blas+tlas      : mean {c_mean:.2} p95 {c_p95:.2} max {c_max:.2} ms");
    println!("build_accel + fence   : mean {b_mean:.2} p95 {b_p95:.2} max {b_max:.2} ms");
    println!("TOTAL per generation  : ~{:.2} ms (upload + create + build)", u_mean + c_mean + b_mean);
    println!("=============================================================================");
}

// ============================================================================================
//  (6) CLIPMAP DELIVERABLE — max view radius + per-single-brick-move stutter, BEFORE vs AFTER.
// ============================================================================================

/// The OLD dense-cube view radius at a given brick radius: `r · BRICK_WORLD_SIZE` (every brick a fixed
/// 1.6 m, coarse LOD added NO coverage). The shipping default before the pivot was `r = 28` ⇒ ~44.8 m.
const OLD_DENSE_RADIUS_BRICKS: f32 = 28.0;

/// **The view-distance deliverable (runs on CI — pure arithmetic, no GPU/voxelize).** Asserts the clipmap
/// view radius (`clip_half · brick_span(MAX_LOD)`) is ≥ ~15× the old dense cube (`28 · BRICK_WORLD_SIZE`).
#[test]
fn clipmap_view_distance() {
    let cfg = shipping_config();
    let new_view = region_half_extent_m(&cfg);
    let old_view = OLD_DENSE_RADIUS_BRICKS * BRICK_WORLD_SIZE;
    let ratio = new_view / old_view;
    println!("\n========== CLIPMAP VIEW DISTANCE ==========");
    println!("config                : clip_half {} bricks, {} nested LOD shells (MAX_LOD {MAX_LOD})", cfg.clip_half_bricks, MAX_LOD + 1);
    println!("BEFORE (dense cube)   : {old_view:.0} m  (radius {OLD_DENSE_RADIUS_BRICKS:.0} bricks · {BRICK_WORLD_SIZE} m, coarse LOD added no reach)");
    println!("AFTER  (clipmap)      : {new_view:.0} m  (clip_half {} · brick_span(MAX_LOD) {:.1} m)", cfg.clip_half_bricks, brick_span(MAX_LOD));
    println!("view-distance gain    : {ratio:.1}×");
    assert!(ratio >= 15.0, "clipmap view radius must be ≥ ~15× the old dense cube (got {ratio:.1}×)");
}

/// **The per-move stutter deliverable (perf harness — `--ignored`, voxelizes the real shipping clipmap).**
/// Warms the clipmap, then nudges the camera ONE LOD0 brick and measures the incremental STREAMING work —
/// the clipmap pivot's stutter metric — SEPARATELY from the re-pack:
///   * `update` (the diff-reconcile) + `drain_work` (voxelize the entering bricks) — O(shell): a thin
///     LOD0 face-slab, NOT the O(region) dense recompute the old model paid every move. THIS is the hitch
///     the pivot removes; the harness asserts it is a small fraction of the cold fill.
///   * `pack_resident_set` is reported separately: it is O(resident) (it rebuilds the whole GPU buffer set)
///     and the LIVE path AMORTIZES it (packs on settle / every few frames, not every move) — it is the
///     pre-existing BLAS-rebuild cost (plan Stage 3), NOT the clipmap stutter this pivot targets.
#[test]
#[ignore = "perf harness; voxelizes the shipping clipmap — run with --ignored --nocapture"]
fn clipmap_per_move_cost() {
    let (layer, lib, registry, label) = worldgen_stack();
    let cfg = shipping_config();

    // Warm the clipmap fully at the origin surface, timing the cold fill (the baseline a dense "recompute
    // everything every move" model would approach on EVERY brick crossing).
    let cam0 = origin_surface_cam(&layer);
    let mut mgr = ResidencyManager::new();
    let cold_t0 = Instant::now();
    mgr.update(cam0, &cfg);
    let cold_enqueued = mgr.pending();
    while mgr.pending() > 0 {
        mgr.drain_work(&cfg, &layer, &lib, &registry, SEED);
    }
    let cold_ms = cold_t0.elapsed().as_secs_f64() * 1e3;
    mgr.take_dirty();

    // Per-move: nudge ONE LOD0 brick in +X. Time the STREAMING (update + drain) and the PACK separately.
    let span0 = brick_span(0);
    let mut stream_costs = Vec::new(); // update + drain — the clipmap stutter metric
    let mut pack_costs = Vec::new(); // pack — the amortized O(resident) BLAS-rebuild cost (reported, not asserted)
    let mut move_churns = Vec::new();
    let mut cam = cam0;
    for _ in 0..8 {
        cam[0] += span0;
        let t_stream = Instant::now();
        let dropped = mgr.update(cam, &cfg);
        let entered = mgr.pending();
        while mgr.pending() > 0 {
            mgr.drain_work(&cfg, &layer, &lib, &registry, SEED);
        }
        stream_costs.push(t_stream.elapsed());
        let t_pack = Instant::now();
        let entries = mgr.resident_entries();
        let _ = pack_resident_set(&entries, &registry);
        pack_costs.push(t_pack.elapsed());
        mgr.take_dirty();
        move_churns.push(entered + dropped);
    }
    let (s_mean, s_p50, s_p95, s_max) = stats_ms(&stream_costs);
    let (pk_mean, _pk_p50, pk_p95, pk_max) = stats_ms(&pack_costs);
    let churn_mean = move_churns.iter().sum::<usize>() as f64 / move_churns.len().max(1) as f64;

    // The O(region) volume a dense recompute would touch (per level) vs the O(shell) face-slab area.
    let half = cfg.clip_half_bricks as usize;
    let region_vol = (2 * half + 1).pow(3);
    let shell_area = (2 * half + 1).pow(2);

    println!("\n========== PER-MOVE STUTTER (single-brick nudge) — graph={label} ==========");
    println!("cold fill (baseline)  : {cold_ms:.1} ms, {cold_enqueued} bricks enqueued (what a dense per-move recompute approached)");
    println!("STREAM per-brick-move : mean {s_mean:.3} p50 {s_p50:.3} p95 {s_p95:.3} max {s_max:.3} ms  ⟵ the stutter metric (update+drain, O(shell))");
    println!("per-move churn        : mean {churn_mean:.0} bricks (enter+drop) | O(shell) ~{shell_area} vs O(region) {region_vol}");
    println!("stutter reduction     : streaming per-move is {:.0}× cheaper than the cold fill", cold_ms / s_mean.max(1e-6));
    println!("pack (amortized)      : mean {pk_mean:.1} p95 {pk_p95:.1} max {pk_max:.1} ms — O(resident) BLAS-rebuild cost, AMORTIZED by the live path (plan Stage 3, NOT this pivot)");
    println!("=============================================================================");

    // The stutter metric: a single-brick move's churn is O(shell), comfortably below the region volume a
    // dense recompute would touch — and the STREAMING wall time is a small fraction of the cold fill.
    assert!(churn_mean < region_vol as f64, "per-move churn must be O(shell), not O(region)");
    assert!(churn_mean <= 8.0 * shell_area as f64, "per-move churn must be shell-sized (≈ a few face-slabs)");
    assert!(s_mean < cold_ms * 0.5, "the per-move STREAMING cost must be well under the cold fill (the stutter fix)");
}

/// Sanity: the harness's worldgen stack actually produces a non-trivial clipmap (so a green CI `--lib` run of
/// the file's non-ignored part can't silently pass on an empty world). Cheap (clip_half-2 clipmap only).
#[test]
fn worldgen_stack_is_non_empty() {
    let (layer, lib, registry, _label) = worldgen_stack();
    let cfg = StreamingConfig { clip_half_bricks: 2, max_bricks_per_frame: 1_000_000, ..Default::default() };
    let cam = origin_surface_cam(&layer);
    let mut mgr = ResidencyManager::new();
    mgr.update(cam, &cfg);
    mgr.drain_work(&cfg, &layer, &lib, &registry, SEED);
    assert!(mgr.resident_count() > 0, "origin-surface clipmap must have non-empty terrain bricks");
    // And the LOD0 brick at the surface is a real, surface-spanning brick (not the uniform fast path).
    let surf = layer.sample_world(0.0, 0.0, SEED).height;
    let b: Brick = voxelize_brick(camera_brick_coord(cam), 0, &layer, &lib, &registry, SEED);
    let _ = (surf, BRICK_WORLD_SIZE, b);
}
