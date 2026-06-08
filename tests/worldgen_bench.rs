//! `#[ignore]` timing harnesses for the procedural-worldgen hot paths the plan flagged. These are
//! NOT correctness tests and NEVER gate CI — run them explicitly:
//!
//! ```text
//! cargo test --features editor --test worldgen_bench -- --ignored --nocapture
//! ```
//!
//! No new dependencies: plain `std::time::Instant`, printed under `--nocapture`. They measure the
//! three hot paths: (a) `LayerManager::update` filling the resident window from cold, (b)
//! `build_height_ring` over a full resident store, and (c) per-chunk `HeightLayer::generate`.

use std::time::Instant;

use bevy::math::{DVec2, IVec3};

use adventure::sdf_render::worldgen::coord::{ChunkCoord, LayerId};
use adventure::sdf_render::worldgen::layer::{GenCtx, GenOutput, Layer};
use adventure::sdf_render::worldgen::layers::height::{HEIGHT_CHUNK_CELLS, HeightLayer, HeightParams};
use adventure::sdf_render::worldgen::manager::LayerManager;

/// The slice's resident radius (≈ 3 height chunks), matching `worldgen::WORLDGEN_SLICE_RADIUS`.
const SLICE_RADIUS: f64 = HEIGHT_CHUNK_CELLS as f64 * 3.0;
const SLICE_SEED: u64 = 0xA15E_C0DE_2026;

/// (a) Cold-fill cost: how long `LayerManager::update` takes to stream the entire resident window
/// in from empty. Uses a large per-update budget so the window fills in as few updates as possible,
/// then reports the total wall time + the resident chunk count (the hot path on a focus jump /
/// scene load).
#[test]
#[ignore = "timing harness; run with --ignored --nocapture"]
fn bench_layer_manager_cold_fill() {
    let mut mgr = LayerManager::new_slice(SLICE_SEED, HeightParams::default(), SLICE_RADIUS);
    mgr.budget = 100_000; // no throttle — measure the raw fill cost

    let focus = DVec2::ZERO;
    let t0 = Instant::now();
    let mut updates = 0u32;
    // Pump until settled (one update should suffice at this budget, but loop defensively).
    loop {
        mgr.update(focus);
        updates += 1;
        if mgr.is_settled(focus) || updates > 32 {
            break;
        }
    }
    let elapsed = t0.elapsed();
    let resident = mgr.height_store().len();
    println!(
        "[bench] LayerManager::update cold-fill: {:.3} ms ({} chunks, {} update(s), {:.3} ms/chunk)",
        elapsed.as_secs_f64() * 1e3,
        resident,
        updates,
        elapsed.as_secs_f64() * 1e3 / resident.max(1) as f64,
    );
    assert!(resident > 0, "cold fill produced no chunks");
}

/// (b) `build_height_ring` cost over a FULL resident store: stream the window in first (untimed),
/// then time only the ring assembly (directory + flat node buffer). This rebuilds on every
/// worldgen delta, so it's on the streaming hot path.
#[test]
#[ignore = "timing harness; run with --ignored --nocapture"]
fn bench_build_height_ring() {
    let mut mgr = LayerManager::new_slice(SLICE_SEED, HeightParams::default(), SLICE_RADIUS);
    mgr.budget = 100_000;
    // Fill the window (not timed).
    for _ in 0..32 {
        mgr.update(DVec2::ZERO);
        if mgr.is_settled(DVec2::ZERO) {
            break;
        }
    }
    let resident = mgr.height_store().len();

    // Time the ring build (averaged over a few iterations to smooth out noise).
    const ITERS: u32 = 16;
    let t0 = Instant::now();
    let mut sink = 0usize;
    for _ in 0..ITERS {
        let ring = adventure::sdf_render::worldgen::upload::build_height_ring(mgr.height_store());
        sink = sink.wrapping_add(ring.nodes.len() + ring.directory.len());
    }
    let elapsed = t0.elapsed();
    println!(
        "[bench] build_height_ring over {} resident chunks: {:.3} ms/build (avg of {} iters, sink={})",
        resident,
        elapsed.as_secs_f64() * 1e3 / ITERS as f64,
        ITERS,
        sink,
    );
    assert!(resident > 0, "no resident chunks to build a ring from");
}

/// (c) Per-chunk `HeightLayer::generate` cost: the fBm fill of one chunk's `(res+1)²` height nodes.
/// This is the inner unit `LayerManager::update` calls per newly-required chunk, so it's the
/// fundamental grain of the streaming budget.
#[test]
#[ignore = "timing harness; run with --ignored --nocapture"]
fn bench_height_layer_generate_per_chunk() {
    let layer = HeightLayer::new(LayerId(0), HeightParams::default());
    let size = layer.chunk_size();

    const ITERS: u32 = 64;
    let t0 = Instant::now();
    let mut sink = 0f32;
    for k in 0..ITERS {
        // Vary the coord so the optimizer can't fold the fBm to a constant.
        let coord = ChunkCoord::new(LayerId(0), IVec3::new(k as i32, 0, (k as i32) * 7 - 13));
        let ctx = GenCtx { coord, seed: SLICE_SEED, size };
        let mut out = GenOutput::default();
        layer.generate(&ctx, &mut out);
        let field = out
            .take::<adventure::sdf_render::worldgen::artifact::ScalarField2D>(HeightLayer::OUTPUT)
            .expect("height layer produces its output");
        sink += field.node(0, 0).height;
    }
    let elapsed = t0.elapsed();
    println!(
        "[bench] HeightLayer::generate per chunk: {:.4} ms/chunk (avg of {} iters, sink={})",
        elapsed.as_secs_f64() * 1e3 / ITERS as f64,
        ITERS,
        sink,
    );
}
