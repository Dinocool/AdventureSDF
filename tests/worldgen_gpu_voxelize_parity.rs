//! **GPU brick-voxelize parity** — Stage 1b of the GPU-voxel-worldgen pivot (docs/GPU_VOXEL_WORLDGEN_PLAN.md).
//!
//! `tests/worldgen_gpu_parity.rs` proves the GPU `wg_eval_graph` reproduces the CPU height SURFACE (value +
//! gradient) within an f32 tolerance. This rig proves the GPU produces the SAME PER-VOXEL BLOCK IDS the CPU
//! `voxelize_brick` chain does, for the HALOED brick (`halo_edge³` cells) — the full height → climate biome →
//! strata → surface-skin → block-id chain + the halo. This is the foundational de-risk the GPU brick pool
//! (G2) builds on: without per-voxel parity, "GPU-direct voxelization" is unverified.
//!
//! ## The parity CONTRACT (the float-parity subtlety, the thing to be most rigorous about)
//! The CPU evaluates in f64; the GPU in f32 (the ONLY sanctioned divergence — worldgen_gpu.wgsl header). A
//! voxel's block id is a DISCONTINUOUS function of the (continuous) height/depth: it flips Air↔Surface where
//! the surface plane crosses the voxel centre, and flips between strata where a depth band boundary crosses.
//! So an f32-vs-f64 rounding gap of ~ULPs in the height/depth can FLIP the block of a voxel whose centre sits
//! within that gap of a threshold — but ONLY there. The contract is therefore:
//!
//!   * INTERIOR voxels (centre comfortably above/below a threshold) MUST match EXACTLY — any mismatch there
//!     is a real codegen/port bug (wrong op, wrong octave, wrong strata walk), not f32 rounding.
//!   * A mismatch is TOLERATED only when the voxel's centre is within `BAND` metres of a height/depth
//!     threshold (surface plane or a strata boundary) — i.e. the f64 and f32 evaluations legitimately land on
//!     opposite sides of a discontinuity. We compute, per mismatching cell, the f64 distance to the NEAREST
//!     threshold and assert it is `< BAND`; and we cap the TOTAL mismatch fraction. `BAND` is sized to the
//!     measured f32 height error (worldgen_gpu_parity reports ~1e-2..1e-1 m at ±4 km), with margin.
//!
//! This is NOT a blanket "ids may differ" tolerance: every tolerated mismatch must be provably surface- or
//! stratum-boundary-straddling, and the count is bounded. A bulk mismatch (e.g. a swapped biome, a wrong
//! strata order, an off-by-one halo) lands interior cells on the wrong side and fails the EXACT check.
//!
//! Needs a GPU adapter; skips cleanly when none is present (plain f32, no special features).

use adventure::sdf_render::worldgen::biome::{
    BiomeDef, BiomeLibrary, StrataLayer, TerrainMatId, TerrainSurfaceMaterial, surface_biome,
};
use adventure::sdf_render::worldgen::coord::LayerId;
use adventure::sdf_render::worldgen::graph::preset::mountains_plains_graph;
use adventure::sdf_render::worldgen::graph::{Graph, Node, NodeKind};
use adventure::sdf_render::worldgen::graph::node::FbmAxis;
use adventure::sdf_render::worldgen::layers::erosion::ErosionParams;
use adventure::sdf_render::worldgen::layers::height::{HeightLayer, HeightParams};
use adventure::voxel::brickmap::{BRICK_EDGE, brick_span, lod_voxel_size};
use adventure::voxel::gpu::halo_edge;
use adventure::voxel::gpu_voxelize::{WvParams, halo_cell_count, voxelize_shader_src};
use adventure::voxel::palette::{BlockId, BlockRegistry};
use adventure::voxel::voxelize::{ColumnSample, SURFACE_SKIN_DEPTH};
use bevy::math::IVec3;
use bevy::render::render_resource::encase;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[path = "common/mod.rs"]
mod common;

/// A worldgen library whose biomes have DISTINCT surface + strata materials, so the climate classifier
/// genuinely changes the per-voxel block (not a degenerate all-same-column library). Empty `surface_rules`
/// (the G1 scope — the surface skin is the biome's `surface` material, matching `voxelize_brick`'s library).
fn distinct_biome_library() -> BiomeLibrary {
    let mat = |name: &str, c: [f32; 4]| TerrainSurfaceMaterial {
        name: name.into(),
        base_color: c,
        roughness: 0.9,
        ..Default::default()
    };
    // 6 distinguishable materials.
    let materials = (0..6).map(|i| mat(&format!("m{i}"), [i as f32 / 6.0, 0.5, 0.2, 1.0])).collect();
    // Each biome: a distinct surface, a distinct sub-surface band, a shared deep stone, distinct bedrock.
    let col = |surf: u16, sub: u16, rock: u16| BiomeDef {
        name: "b".into(),
        surface: TerrainMatId(surf),
        surface_rules: vec![],
        strata: vec![
            StrataLayer { material: TerrainMatId(surf), thickness: 1.0 },
            StrataLayer { material: TerrainMatId(sub), thickness: 4.0 },
            StrataLayer { material: TerrainMatId(rock), thickness: 20.0 },
        ],
        bedrock: TerrainMatId(5),
    };
    // BiomeId order: Plains, Forest, Desert, Tundra, Snowy.
    let biomes = vec![
        col(0, 1, 4), // Plains
        col(1, 2, 4), // Forest
        col(2, 3, 4), // Desert
        col(3, 1, 4), // Tundra
        col(0, 3, 4), // Snowy
    ];
    BiomeLibrary { materials, biomes }
}

/// A height layer driven by `graph` (the same graph the GPU codegens). Plain (no legacy ridge/erosion); the
/// graph IS the surface — bit-for-bit the path `sample_world` takes for an attached graph.
fn graph_layer(graph: Graph) -> HeightLayer {
    HeightLayer::new(LayerId(0), HeightParams::default(), ErosionParams { enabled: false, ..Default::default() })
        .with_graph(Some(Arc::new(graph)))
}

/// CPU oracle: the HALOED brick's block ids, built through the SAME `ColumnSample`/`block_at` SSOT
/// `voxelize_brick` uses, extended to the `halo_edge³` grid (haloed cell `(hx,hy,hz)` → core-relative voxel
/// `(hx-1,hy-1,hz-1)` at the LOD cell size). This is the authoritative reference the GPU must match.
fn cpu_haloed_brick(
    brick_coord: IVec3,
    lod: u32,
    layer: &HeightLayer,
    lib: &BiomeLibrary,
    reg: &BlockRegistry,
    seed: u64,
) -> Vec<u16> {
    let hedge = halo_edge(lod);
    let span = brick_span(lod) as f64;
    let cell = lod_voxel_size(lod) as f64;
    let world_min = [
        brick_coord.x as f64 * span,
        brick_coord.y as f64 * span,
        brick_coord.z as f64 * span,
    ];
    let mut out = vec![0u16; (hedge * hedge * hedge) as usize];
    for hz in 0..hedge {
        for hx in 0..hedge {
            let wx = world_min[0] + ((hx - 1) as f64 + 0.5) * cell;
            let wz = world_min[2] + ((hz - 1) as f64 + 0.5) * cell;
            let col = ColumnSample::at(wx, wz, layer, seed);
            for hy in 0..hedge {
                let wy = world_min[1] + ((hy - 1) as f64 + 0.5) * cell;
                let idx = (hx + hy * hedge + hz * hedge * hedge) as usize;
                out[idx] = col.block_at(wy, lib, reg).0;
            }
        }
    }
    out
}

/// Dispatch the GPU voxelizer for `(brick_coord, lod)` and read back the `halo_edge³` block ids.
#[allow(clippy::too_many_arguments)]
fn gpu_haloed_brick(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    graph: &Graph,
    brick_coord: IVec3,
    lod: u32,
    lib: &BiomeLibrary,
    reg: &BlockRegistry,
    seed: u64,
) -> Vec<u16> {
    let src = voxelize_shader_src(graph, "assets/shaders");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("worldgen_voxelize"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });

    // The uniform (the flattened library + brick placement + climate knobs).
    let params = WvParams::build(brick_coord, lod, lib, reg, seed);
    let mut ub = encase::UniformBuffer::new(Vec::<u8>::new());
    ub.write(&params).expect("encode WvParams");
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wv_params"),
        contents: ub.as_ref(),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let n = halo_cell_count(lod);
    let out_bytes = (n * std::mem::size_of::<u32>()) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wv_out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wv_staging"),
        size: out_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("wv_bgl"),
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
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wv_pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wv_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("voxelize_main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wv_bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wv_enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wv_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
    queue.submit(std::iter::once(encoder.finish()));

    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = staging.slice(..).get_mapped_range().expect("map staging");
    let words: &[u32] = bytemuck::cast_slice(&data);
    let out: Vec<u16> = words.iter().map(|&w| w as u16).collect();
    drop(data);
    staging.unmap();
    out
}

/// The f64 distance (metres) from a voxel CENTRE at world-Y `wy` in this column to the NEAREST block-id
/// threshold the CPU `block_at` chain crosses: the surface plane (`depth == 0`), the surface-skin boundary
/// (`depth == SURFACE_SKIN_DEPTH`), and every strata-band BOTTOM. A mismatching GPU cell is only tolerated
/// when this distance is small (the f32 height/depth legitimately landed on the other side). `depth = h −
/// wy`, so a 1-metre change in `depth` is a 1-metre change in `wy` — distances are directly in world Y.
fn dist_to_nearest_threshold(
    wx: f64,
    wz: f64,
    wy: f64,
    layer: &HeightLayer,
    lib: &BiomeLibrary,
    seed: u64,
) -> f64 {
    let h = layer.sample_world(wx, wz, seed).height as f64;
    let depth = h - wy;
    // The biome whose strata column applies here (the sub-surface biome the CPU uses below the skin).
    let biome = surface_biome(wx, wz, seed).primary;
    let def = lib.biome(biome);

    let mut best = depth.abs(); // distance to the surface plane (depth == 0)
    best = best.min((depth - SURFACE_SKIN_DEPTH).abs()); // the skin boundary
    let mut cum = 0.0_f64;
    for layer in &def.strata {
        cum += layer.thickness as f64;
        best = best.min((depth - cum).abs());
    }
    best
}

/// Run one brick parity case: voxelize on the CPU + GPU, compare cell-by-cell. Returns
/// `(total_cells, mismatch_cells, worst_interior_threshold_dist)`. INTERIOR (far-from-threshold) mismatches
/// fail HARD here (assertion); the returned worst distance is the band the tolerated mismatches fell within.
#[allow(clippy::too_many_arguments)]
fn parity_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    graph: &Graph,
    brick_coord: IVec3,
    lod: u32,
    lib: &BiomeLibrary,
    reg: &BlockRegistry,
    seed: u64,
) -> (usize, usize, f64) {
    /// Tolerated band (metres) around a block-id threshold within which an f32-vs-f64 flip is allowed. Sized
    /// well above the measured f32 height error (worldgen_gpu_parity: ~1e-1 m worst at ±4 km) with margin,
    /// and below the thinnest strata band (1 m) so a tolerated flip can never reach an interior cell.
    const BAND: f64 = 0.30;

    let layer = graph_layer(graph.clone());
    let cpu = cpu_haloed_brick(brick_coord, lod, &layer, lib, reg, seed);
    let gpu = gpu_haloed_brick(device, queue, graph, brick_coord, lod, lib, reg, seed);
    assert_eq!(cpu.len(), gpu.len(), "{label}: brick length mismatch");

    let hedge = halo_edge(lod);
    let span = brick_span(lod) as f64;
    let cell = lod_voxel_size(lod) as f64;
    let world_min = [
        brick_coord.x as f64 * span,
        brick_coord.y as f64 * span,
        brick_coord.z as f64 * span,
    ];

    let mut mismatch = 0usize;
    let mut worst_band = 0.0_f64;
    let mut interior_fail: Option<String> = None;
    for hz in 0..hedge {
        for hx in 0..hedge {
            let wx = world_min[0] + ((hx - 1) as f64 + 0.5) * cell;
            let wz = world_min[2] + ((hz - 1) as f64 + 0.5) * cell;
            for hy in 0..hedge {
                let idx = (hx + hy * hedge + hz * hedge * hedge) as usize;
                if cpu[idx] == gpu[idx] {
                    continue;
                }
                mismatch += 1;
                let wy = world_min[1] + ((hy - 1) as f64 + 0.5) * cell;
                let d = dist_to_nearest_threshold(wx, wz, wy, &layer, lib, seed);
                worst_band = worst_band.max(d.min(BAND)); // track the worst TOLERATED band
                if d >= BAND && interior_fail.is_none() {
                    interior_fail = Some(format!(
                        "{label}: INTERIOR mismatch at halo ({hx},{hy},{hz}) world ({wx:.3},{wy:.3},{wz:.3}): \
                         CPU block {} vs GPU {} is {d:.4} m from the nearest threshold (>= BAND {BAND}) — a real \
                         port bug, not f32 rounding",
                        cpu[idx], gpu[idx],
                    ));
                }
            }
        }
    }
    if let Some(msg) = interior_fail {
        panic!("{msg}");
    }
    (cpu.len(), mismatch, worst_band)
}

/// THE gate: across a sample of `(coord, lod, graph/seed)`, the GPU haloed brick EQUALS the CPU
/// `voxelize_brick` chain — exactly for interior voxels, and within a `BAND`-thin surface/strata-boundary
/// band for the f32-vs-f64 straddlers (capped at a small fraction). A bulk mismatch (wrong biome / strata /
/// halo) fails the interior EXACT check; honest f32 rounding passes.
#[test]
fn gpu_voxelize_matches_cpu_voxelize_brick() {
    let Some((device, queue)) = common::headless_device(wgpu::Features::empty()) else {
        eprintln!("[skip] no GPU adapter — GPU voxelize parity skipped");
        return;
    };

    let lib = distinct_biome_library();
    let reg = BlockRegistry::from_biome_library(&lib);

    // The shipped mountains+plains graph (Fbm/Curve/Smoothstep/Ridge/Offset/Mix — the full op set) plus a
    // simpler ridged-carrier graph (a different topology / output index) for op coverage.
    let mtn = mountains_plains_graph(700.0);
    let carrier = FbmAxis { octaves: 5, base_freq: 1.0 / 700.0, lacunarity: 2.0, gain: 0.5, amplitude: 30.0, seed_salt: 0x33 };
    let simple = Graph {
        nodes: vec![
            Node::source(NodeKind::Fbm(carrier)),
            Node::unary(NodeKind::Ridge { ridge: 0.6, amp_sum: 30.0 * 1.9375 }, 0),
            Node::unary(NodeKind::Offset(2.0), 1),
        ],
        output: 2,
    };

    // Bricks chosen to STRADDLE the surface (so the brick mixes Air / Surface / Interior — the interesting
    // case; a pure-air or pure-solid brick is a trivial uniform). The brick Y is derived per case from the
    // surface height at the brick's XZ centre (`by = floor(h / span)`), so the brick band contains the
    // surface plane regardless of the graph's elevation — exactly how `voxelize.rs`'s mip test centres a
    // brick on the surface. `(graph, cx, cz, lod, seed)`.
    let proto: &[(&str, &Graph, i32, i32, u32, u64)] = &[
        ("mtn  c(0,?,0)  lod0", &mtn, 0, 0, 0, 1234),
        ("mtn  c(-3,?,5) lod0", &mtn, -3, 5, 0, 1234),
        ("mtn  c(0,?,0)  lod1", &mtn, 0, 0, 1, 1234),
        ("mtn  c(2,?,-1) lod2", &mtn, 2, -1, 2, 1234),
        ("simp c(0,?,0)  lod0", &simple, 0, 0, 0, 42),
        ("simp c(7,?,-7) lod0", &simple, 7, -7, 0, 99),
        ("simp c(0,?,0)  lod1", &simple, 0, 0, 1, 42),
        // FAR-from-origin (≈ +4 km in X, the band the height-parity rig measured ~0.1 m f32 error at): a
        // LOD2 brick (cell 0.2 m) so the larger f32 height error vs the coarse cell genuinely straddles the
        // surface plane → exercises the tolerated surface-band (the contract isn't vacuous). `cx·span(2) ≈
        // 4096 m` at cx = 2560.
        ("mtn  c(2560,?,0) lod2", &mtn, 2560, 0, 2, 1234),
    ];
    let cases: Vec<(&str, &Graph, IVec3, u32, u64)> = proto
        .iter()
        .map(|&(label, graph, cx, cz, lod, seed)| {
            let span = brick_span(lod) as f64;
            // Surface height at the brick's XZ centre → the brick row whose Y band contains it.
            let layer = graph_layer(graph.clone());
            let wxc = (cx as f64 + 0.5) * span;
            let wzc = (cz as f64 + 0.5) * span;
            let h = layer.sample_world(wxc, wzc, seed).height as f64;
            let by = (h / span).floor() as i32;
            (label, graph, IVec3::new(cx, by, cz), lod, seed)
        })
        .collect();

    let mut any_surface_brick = false;
    let mut total_cells = 0usize;
    let mut total_mismatch = 0usize;
    let mut worst_band = 0.0_f64;
    for &(label, graph, coord, lod, seed) in &cases {
        let (cells, mismatch, band) = parity_case(&device, &queue, label, graph, coord, lod, &lib, &reg, seed);
        let pct = 100.0 * mismatch as f64 / cells as f64;
        println!(
            "[voxelize-parity] {label}: {mismatch}/{cells} cells differ ({pct:.3}%), worst tolerated band \
             {band:.4} m"
        );
        total_cells += cells;
        total_mismatch += mismatch;
        worst_band = worst_band.max(band);

        // A brick that mixes air + solid is a real surface brick (the cells we care about exist).
        let cpu = cpu_haloed_brick(coord, lod, &graph_layer(graph.clone()), &lib, &reg, seed);
        let air = cpu.iter().filter(|&&b| b == BlockId::AIR.0).count();
        if air > 0 && air < cpu.len() {
            any_surface_brick = true;
        }
    }

    assert!(any_surface_brick, "no test brick straddled the surface — the parity case set is degenerate");

    // Per-case interior mismatches already failed HARD in `parity_case`. Here cap the AGGREGATE tolerated
    // (surface-band-straddling) mismatch fraction: f32-vs-f64 flips a THIN shell of cells; if more than a
    // few % differ, the surface band is unexpectedly wide ⇒ investigate (a sign the f32 error is larger than
    // the height-parity rig reported, or a near-threshold bias).
    let frac = total_mismatch as f64 / total_cells as f64;
    println!(
        "[voxelize-parity] AGGREGATE: {total_mismatch}/{total_cells} cells differ ({:.3}%), worst tolerated \
         band {worst_band:.4} m — all surface/strata-boundary straddlers (interior cells matched exactly)",
        100.0 * frac
    );
    assert!(
        frac < 0.02,
        "tolerated (surface-band) mismatch fraction {:.3}% exceeds 2% — the f32 band is wider than expected",
        100.0 * frac
    );
    let _ = BRICK_EDGE;
}
