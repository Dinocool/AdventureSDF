//! **D1d throwaway timing — shell-first `update` cost ONLY.** Times the NEW `desired_clipmap_surface` + a
//! single cold `ResidencyManager::update` at the production `StreamingConfig::default()` (clip_half 160),
//! skipping the slow OLD cube baseline + the full pack that `d1c_scaling` runs — so the shell-first number
//! (the thing D1d fixes) is measured in seconds, not minutes. Pure CPU; no GPU. Run:
//!   cargo run --release --no-default-features --features fast,physics --example d1d_shellfirst
//!
//! NOT a shipped tool — a one-off de-risk for the D1d perf re-measure.

use std::time::Instant;

use adventure::sdf_render::worldgen::graph::GraphAsset;
use adventure::sdf_render::worldgen::layers::erosion::ErosionParams;
use adventure::sdf_render::worldgen::layers::height::{HeightLayer, HeightParams};
use adventure::sdf_render::worldgen::{WORLDGEN_SLICE_SEED, WorldBiomeShapes, WorldGraph};
use adventure::voxel::brickmap::{MAX_LOD, brick_span};
use adventure::voxel::palette::BlockRegistry;
use adventure::voxel::source::WorldgenSource;
use adventure::voxel::streaming::{
    MAX_CLIP_ENUMERATION, ResidencyManager, StreamingConfig, desired_clipmap_surface,
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
    let _registry = BlockRegistry::from_biome_library(&lib);
    let cfg = StreamingConfig::default();
    let surf = layer.sample_world(0.0, 0.0, SEED).height;
    let cam = [0.0f32, surf, 0.0];
    let src = WorldgenSource::new(&layer, &lib, SEED);

    println!("\n===== D1d SHELL-FIRST timing — graph={label}, clip_half {} =====", cfg.clip_half_bricks);

    // Shell-first enumeration only.
    let t = Instant::now();
    let surf_desired = desired_clipmap_surface(cam, &cfg, &src);
    let enum_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut per_lod = vec![0usize; (MAX_LOD + 1) as usize];
    for k in surf_desired.keys() {
        per_lod[k.lod as usize] += 1;
    }
    println!("desired_clipmap_surface : {} candidates ({enum_ms:.1} ms)", surf_desired.len());
    println!("hit ceiling?            : {}", surf_desired.len() > MAX_CLIP_ENUMERATION);
    for (l, n) in per_lod.iter().enumerate() {
        if *n > 0 {
            println!("  LOD{l}                  : {n} candidates (span {:.1} m)", brick_span(l as u32));
        }
    }

    // Single cold update (the D1c 38 s hot spot).
    let mut mgr = ResidencyManager::new();
    let t = Instant::now();
    mgr.update(cam, &cfg, &src);
    let update_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("\ncold update wall        : {update_ms:.1} ms  (D1c cube path was ~38_000 ms)");
    println!("enqueued (capped)       : {}  (cap {})", mgr.pending(), cfg.max_resident_bricks);
    println!("capped_total dropped    : {}", mgr.capped_total);
    println!("================================================================");
}
