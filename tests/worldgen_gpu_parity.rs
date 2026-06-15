//! **GPU numerical-parity for the Stage-1 worldgen height-field WGSL port** (docs/GPU_VOXEL_WORLDGEN_PLAN.md).
//!
//! `tests/worldgen_codegen.rs` proves the generated `wg_eval_graph` + the `worldgen_gpu.wgsl` library
//! COMPILE (naga validation, no GPU). This rig proves they COMPUTE THE SAME TERRAIN as the CPU node-graph
//! SSOT (`Graph::eval`): it dispatches `wg_eval_graph` over a grid of world coords on a real GPU and asserts
//! the value + analytic world-XZ gradient match the CPU `eval` within an f32 tolerance.
//!
//! This is the correctness gate the GPU brick voxelizer (Stage 1b) and the GPU brick pool (Stage 2) build
//! on: without a numerical parity check, "GPU-direct worldgen" is unverified — a silent codegen or
//! field-op port bug would only ever surface as wrong terrain at runtime.
//!
//! ## f32 vs f64 (the only sanctioned divergence)
//! The CPU evaluates in f64 (bit-portable, shared-seed authoritative); WGSL has only f32. So the FLOAT
//! results differ in the low mantissa bits (worldgen_gpu.wgsl header). The tolerance below is sized to that
//! — tight enough that a real port bug (wrong op, wrong octave count, swapped gradient) fails it, loose
//! enough that honest f32 rounding passes. The INTEGER hash entropy is ported exactly (u32 wrapping), so
//! seed 0 and a u32-range seed both exercise the lattice hash identically on both sides.
//!
//! ## Coordinate range
//! Sampled over a moderate ±4 km region where f32 still resolves world coords to < 1 mm, so the divergence
//! measured here is purely the math port, NOT large-world-coord f32 blowup (at ±131 km f32 has ~16 m
//! resolution — the GPU voxelizer will evaluate chunk-relative to stay in that precise band; tracked
//! separately). Needs a GPU adapter; skips cleanly when none is present (no special features — plain f32).

use adventure::sdf_render::worldgen::WORLDGEN_SLICE_SEED;
use adventure::sdf_render::worldgen::graph::preset::{
    MOUNTAINS_PLAINS_AMPLITUDE, default_terrain_graph, mountains_plains_graph,
};
use adventure::sdf_render::worldgen::graph::node::FbmAxis;
use adventure::sdf_render::worldgen::graph::wgsl_codegen::EVAL_FN_NAME;
use adventure::sdf_render::worldgen::graph::{Graph, GraphAsset, graph_to_wgsl};
use std::path::Path;
use wgpu::util::DeviceExt;

#[path = "common/mod.rs"]
mod common;

/// The `worldgen_gpu.wgsl` library source with its `#define_import_path` line stripped, so it concatenates
/// directly into a self-contained entry shader for wgpu's (non-naga_oil) front-end. Mirrors
/// `tests/worldgen_codegen.rs::worldgen_lib_source`.
fn worldgen_lib_source() -> String {
    let lib = std::fs::read_to_string("assets/shaders/worldgen_gpu.wgsl")
        .expect("read assets/shaders/worldgen_gpu.wgsl");
    lib.lines()
        .filter(|l| !l.trim_start().starts_with("#define_import_path"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build the full self-contained compute shader: the library + the generated `wg_eval_graph` + a `@compute`
/// entry that evaluates the graph at each input coord and stores `(v, dx, dz)`.
fn parity_shader_src(graph: &Graph, seed: u32) -> String {
    let lib = worldgen_lib_source();
    let generated = graph_to_wgsl(graph);
    format!(
        r#"{lib}

{generated}

const WG_SEED: u32 = {seed}u;

struct ParitySample {{
    v: f32,
    dx: f32,
    dz: f32,
    pad: f32,
}}

@group(0) @binding(0) var<storage, read> coords: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> out_samples: array<ParitySample>;

@compute @workgroup_size(64)
fn parity_main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&coords)) {{
        return;
    }}
    let c = coords[i];
    let f = {EVAL_FN_NAME}(c.x, c.y, WG_SEED);
    out_samples[i] = ParitySample(f.v, f.dx, f.dz, 0.0);
}}
"#
    )
}

/// A deterministic grid of world coords over `[-half, half]²` (`n×n` points), flattened as `[x0,z0,x1,z1,…]`
/// for upload as `array<vec2<f32>>`.
fn grid_coords(n: usize, half: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(2 * n * n);
    for iz in 0..n {
        for ix in 0..n {
            // Map [0, n-1] → [-half, half]; offset by a non-lattice fraction so points don't all land on
            // integer noise-lattice corners (where interpolation is trivially exact and hides op bugs).
            let fx = (ix as f32 / (n - 1) as f32) * 2.0 - 1.0;
            let fz = (iz as f32 / (n - 1) as f32) * 2.0 - 1.0;
            out.push(fx * half + 0.3137);
            out.push(fz * half - 0.7191);
        }
    }
    out
}

/// Dispatch `wg_eval_graph` over `coords_flat` (flattened `[x,z,…]`) on the GPU; returns `(v, dx, dz)` per
/// point. Standard compute dispatch + staging-buffer readback.
fn gpu_eval(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    graph: &Graph,
    seed: u32,
    coords_flat: &[f32],
) -> Vec<[f32; 3]> {
    let n = coords_flat.len() / 2;
    let src = parity_shader_src(graph, seed);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("worldgen_parity"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });

    let coords_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("parity_coords"),
        contents: bytemuck::cast_slice(coords_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_bytes = (n * 4 * std::mem::size_of::<f32>()) as u64; // ParitySample = 4×f32 = 16 B
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("parity_out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("parity_staging"),
        size: out_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("parity_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
        label: Some("parity_pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("parity_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("parity_main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("parity_bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: coords_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("parity_enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(n.div_ceil(64) as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
    queue.submit(std::iter::once(encoder.finish()));

    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = staging.slice(..).get_mapped_range().expect("map staging buffer");
    let floats: &[f32] = bytemuck::cast_slice(&data);
    let out: Vec<[f32; 3]> = (0..n).map(|i| [floats[4 * i], floats[4 * i + 1], floats[4 * i + 2]]).collect();
    drop(data);
    staging.unmap();
    out
}

/// Run the parity comparison for one graph + seed, returning the worst value + gradient errors (so the
/// caller can both ASSERT and PRINT them). The CPU reference is the f64 `Graph::eval`; the seed is passed as
/// `u64`/`u32` to CPU/GPU respectively (a u32-range seed is the same integer on both sides).
fn parity_errors(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    graph: &Graph,
    seed: u32,
) -> (f64, f64, f64) {
    const N: usize = 96;
    const HALF: f32 = 4000.0;
    let coords = grid_coords(N, HALF);
    let gpu = gpu_eval(device, queue, graph, seed, &coords);

    let mut max_v_abs = 0.0f64;
    let mut max_v_rel = 0.0f64;
    let mut max_g_abs = 0.0f64;
    for (i, g) in gpu.iter().enumerate() {
        let wx = coords[2 * i] as f64;
        let wz = coords[2 * i + 1] as f64;
        let cpu = graph.eval(wx, wz, seed as u64);
        let v_abs = (g[0] as f64 - cpu.v).abs();
        let v_rel = v_abs / (1.0 + cpu.v.abs());
        let g_abs = ((g[1] as f64 - cpu.dx).abs()).max((g[2] as f64 - cpu.dz).abs());
        max_v_abs = max_v_abs.max(v_abs);
        max_v_rel = max_v_rel.max(v_rel);
        max_g_abs = max_g_abs.max(g_abs);
    }
    (max_v_abs, max_v_rel, max_g_abs)
}

/// The shipping `assets/worldgen/world.graph.ron` (the live scene's terrain) + the two in-code presets all
/// evaluate IDENTICALLY (within f32 tolerance) on the GPU and the CPU — value AND analytic gradient. Seed 0
/// (unambiguous hash fold) is the hard gate; the shipping seed (a real non-zero entropy) is also asserted to
/// catch any seed-fold port bug.
#[test]
fn gpu_graph_eval_matches_cpu() {
    let Some((device, queue)) = common::headless_device(wgpu::Features::empty()) else {
        eprintln!("[skip] no GPU adapter — worldgen GPU parity skipped");
        return;
    };

    // The carrier axis the presets use (matches tests/worldgen_codegen.rs).
    let carrier = FbmAxis { octaves: 6, base_freq: 1.0 / 1536.0, lacunarity: 2.0, gain: 0.5, amplitude: 280.0, seed_salt: 0 };

    let world = {
        let ron = std::fs::read_to_string(Path::new("assets/worldgen/world.graph.ron"))
            .expect("read world.graph.ron");
        let asset: GraphAsset = ron::de::from_str(&ron).expect("parse world.graph.ron");
        asset.graph.validate().expect("world.graph.ron valid");
        asset.graph
    };
    let cases: [(&str, Graph); 3] = [
        ("default_terrain", default_terrain_graph(carrier, 0.5, 280.0 * 1.96875, 0.0)),
        ("mountains_plains", mountains_plains_graph(MOUNTAINS_PLAINS_AMPLITUDE)),
        ("world.graph.ron", world),
    ];

    // f32-vs-f64 tolerance: a faithful port rounds in the low mantissa bits. Terrain values reach ±700 m, so
    // a relative bound catches an op/octave bug (which shifts the value by metres / a large fraction) while
    // passing honest f32 rounding. Gradients (derivatives) carry more f32 error → a looser absolute bound.
    const TOL_V_REL: f64 = 2.0e-3;
    const TOL_V_ABS: f64 = 0.05; // metres — floor so near-zero heights aren't held to an impossible rel bound
    const TOL_G_ABS: f64 = 5.0e-2; // slope units

    for seed in [0u32, (WORLDGEN_SLICE_SEED & 0xFFFF_FFFF) as u32] {
        for (label, graph) in &cases {
            let (v_abs, v_rel, g_abs) = parity_errors(&device, &queue, graph, seed);
            println!(
                "[worldgen-parity] {label:>16} seed={seed:#010x}: max |Δv|={v_abs:.5} m, rel={v_rel:.2e}, max |Δgrad|={g_abs:.2e}"
            );
            assert!(
                v_abs <= TOL_V_ABS || v_rel <= TOL_V_REL,
                "{label} seed {seed:#x}: GPU height diverged from CPU — |Δv|={v_abs:.4} m (rel {v_rel:.2e}) exceeds tol",
            );
            assert!(
                g_abs <= TOL_G_ABS,
                "{label} seed {seed:#x}: GPU gradient diverged from CPU — max |Δgrad|={g_abs:.3e} exceeds {TOL_G_ABS:.1e}",
            );
        }
    }
}
