//! **D1c throwaway diagnostic — 64 m @ 0.05 m reach/perf scaling.** Measures the STRUCTURAL numbers the
//! `voxel_worldgen_perf` harness reports, but FAST: it isolates the geometry (enumeration ceiling, per-LOD
//! desired tiling, surface-candidate count) and times a SINGLE cold `update` + a SINGLE bounded cold drain,
//! rather than running the harness's repeated O(8M-key) updates (impractically slow at clip_half 160). Pure
//! CPU; no GPU. Run:
//!   cargo run --release --no-default-features --features fast,physics --example d1c_scaling
//!
//! NOT a shipped tool — a one-off de-risk for the D1c benchmark. Mirrors the harness's `worldgen_stack` +
//! `shipping_config` exactly so the numbers are the production path.

use std::time::Instant;

use adventure::sdf_render::worldgen::layers::erosion::ErosionParams;
use adventure::sdf_render::worldgen::layers::height::{HeightLayer, HeightParams};
use adventure::sdf_render::worldgen::{WORLDGEN_SLICE_SEED, WorldBiomeShapes, WorldGraph};
use adventure::sdf_render::worldgen::graph::GraphAsset;
use adventure::voxel::brickmap::{MAX_LOD, brick_span};
use adventure::voxel::gpu::pack_resident_set;
use adventure::voxel::palette::BlockRegistry;
use adventure::voxel::source::{BrickClass, BrickSource, WorldgenSource};
use adventure::voxel::streaming::{
    MAX_CLIP_ENUMERATION, ResidencyManager, StreamingConfig, desired_clipmap, region_half_extent_m,
};
use adventure::voxel::{build_height_layer_pub, load_biome_library_pub};

const SEED: u64 = WORLDGEN_SLICE_SEED;

fn shipping_world_graph() -> (WorldGraph, &'static str) {
    match std::fs::read("assets/worldgen/world.graph.ron") {
        Ok(bytes) => match ron::de::from_bytes::<GraphAsset>(&bytes) {
            Ok(asset) => (WorldGraph(std::sync::Arc::new(asset.graph)), "world.graph.ron"),
            Err(_) => (WorldGraph::default(), "preset default"),
        },
        Err(_) => (WorldGraph::default(), "preset default"),
    }
}

fn main() {
    let (graph, label) = shipping_world_graph();
    let layer: HeightLayer = build_height_layer_pub(
        &HeightParams::default(),
        &ErosionParams::default(),
        &graph,
        &WorldBiomeShapes::default(),
    );
    let lib = load_biome_library_pub();
    let registry = BlockRegistry::from_biome_library(&lib);
    let cfg = StreamingConfig::default();
    let surf = layer.sample_world(0.0, 0.0, SEED).height;
    let cam = [0.0f32, surf, 0.0];

    println!("\n========== D1c SCALING (64 m @ 0.05 m) — graph={label} ==========");
    println!(
        "config                : clip_half {} bricks, max_resident {}, {} LOD shells",
        cfg.clip_half_bricks, cfg.max_resident_bricks, MAX_LOD + 1
    );
    println!(
        "LOD0 reach            : {:.1} m ; total view half-extent : {:.0} m",
        cfg.clip_half_bricks as f32 * brick_span(0),
        region_half_extent_m(&cfg)
    );
    println!("MAX_CLIP_ENUMERATION  : {MAX_CLIP_ENUMERATION}");

    // (A) GEOMETRY: enumerate the desired clipmap once, time it, report per-LOD distribution + whether it hit
    // the enumeration ceiling (⇒ the coarse shells / far LOD0 reach were never enumerated = TRUNCATED).
    let t = Instant::now();
    let desired = desired_clipmap(cam, &cfg);
    let enum_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut per_lod = vec![0usize; (MAX_LOD + 1) as usize];
    for k in desired.keys() {
        per_lod[k.lod as usize] += 1;
    }
    let hit_ceiling = desired.len() > MAX_CLIP_ENUMERATION;
    println!("\n-- (A) desired_clipmap enumeration --");
    println!("enumerated keys       : {} ({enum_ms:.1} ms to enumerate)", desired.len());
    println!("hit ceiling?          : {hit_ceiling}  (true ⇒ coarse shells NOT enumerated = TRUNCATED reach)");
    for (l, n) in per_lod.iter().enumerate() {
        if *n > 0 {
            println!("  LOD{l}                : {n} keys (brick_span {:.1} m)", brick_span(l as u32));
        }
    }

    // (B) CLASSIFY the enumerated keys: surface vs interior vs air — the cold-fill voxelize count = surface.
    let src = WorldgenSource::new(&layer, &lib, SEED);
    let t = Instant::now();
    let (mut surface, mut interior, mut air) = (0usize, 0usize, 0usize);
    for k in desired.keys() {
        match src.classify(k.coord, k.lod) {
            BrickClass::Surface => surface += 1,
            BrickClass::Interior => interior += 1,
            BrickClass::Air => air += 1,
        }
    }
    let classify_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("\n-- (B) classify (surface-following residency) --");
    println!("classify wall         : {classify_ms:.1} ms over {} keys", desired.len());
    println!("surface (voxelized)   : {surface}");
    println!("interior (pruned)     : {interior}");
    println!("air (pruned)          : {air}");

    // (C) SINGLE cold update (enqueue) cost + the capped surface-candidate count.
    let mut mgr = ResidencyManager::new();
    let t = Instant::now();
    mgr.update(cam, &cfg, &src);
    let update_ms = t.elapsed().as_secs_f64() * 1e3;
    let enqueued = mgr.pending();
    println!("\n-- (C) single cold update (enqueue) --");
    println!("update wall           : {update_ms:.0} ms");
    println!("enqueued (capped)     : {enqueued}  (cap {})", cfg.max_resident_bricks);
    println!("capped_total dropped  : {}", mgr.capped_total);

    // (D) Cold DRAIN the whole queue (bounded 256/frame) → steady-state resident count + wall + frames.
    let t = Instant::now();
    let mut frames = 0u32;
    let mut voxelized = 0usize;
    let mut drain_max_ms = 0f64;
    while mgr.pending() > 0 {
        let td = Instant::now();
        voxelized += mgr.drain_work(&cfg, &layer, &lib, &registry, SEED);
        drain_max_ms = drain_max_ms.max(td.elapsed().as_secs_f64() * 1e3);
        frames += 1;
        if frames > 100_000 {
            break;
        }
    }
    let drain_ms = t.elapsed().as_secs_f64() * 1e3;
    let resident = mgr.resident_count();
    println!("\n-- (D) cold drain (256 bricks/frame) --");
    println!("drain wall            : {drain_ms:.0} ms ({:.2} s)", drain_ms / 1e3);
    println!("frames to settle      : {frames}  (per-frame drain max {drain_max_ms:.2} ms)");
    println!("bricks voxelized      : {voxelized}");
    println!("STEADY-STATE RESIDENT : {resident} bricks (cap {})", cfg.max_resident_bricks);
    println!(
        "  ⇒ cap {}",
        if resident >= cfg.max_resident_bricks {
            "HIT — 64 m reach is TRUNCATED by the cap"
        } else {
            "NOT hit — reach fits under the cap"
        }
    );

    // (E) RESIDENT VRAM via the full pack (A4.4-class total: index + palette + meta/aabb).
    let entries = mgr.resident_entries();
    let t = Instant::now();
    let patch = pack_resident_set(&entries, &registry);
    let pack_ms = t.elapsed().as_secs_f64() * 1e3;
    let rep = patch.storage_report();
    println!("\n-- (E) pack_resident_set + RESIDENT VRAM --");
    println!("pack wall             : {pack_ms:.1} ms for {} bricks", patch.brick_count());
    println!(
        "RESIDENT VRAM (after) : {:.1} MB  (before-collapse {:.1} MB, {:.2}× reduction)",
        rep.total_vram_after() as f64 / 1e6,
        rep.total_vram_before() as f64 / 1e6,
        rep.vram_reduction(),
    );
    println!("=============================================================================");
}
