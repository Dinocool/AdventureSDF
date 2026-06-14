//! Naga validation for the Stage-1a GPU worldgen codegen (docs/GPU_VOXEL_WORLDGEN_PLAN.md).
//!
//! For every shipped terrain graph — the on-disk `assets/worldgen/*.graph.ron` files the engine loads
//! AND the in-code presets — this:
//!   1. parses / builds the [`Graph`],
//!   2. runs [`graph_to_wgsl`] (the `NodeKind → WGSL` codegen),
//!   3. composes `worldgen_gpu.wgsl` (the `worldgen::gpu` library) + the generated `wg_eval_graph` + a
//!      tiny `@compute` entry that calls it and stores the result, and
//!   4. validates the whole module with the FULL-capability `naga_oil` Composer — mirroring
//!      `tests/shader_validation.rs` (and Bevy's runtime composer, which uses `with_capabilities`).
//!
//! This catches a WGSL typo in either the hand-written library OR the codegen without a GPU/launch, and
//! asserts the library + every shipped graph compose. NO GPU dispatch (that is a later sub-stage).

use adventure::sdf_render::worldgen::graph::node::FbmAxis;
use adventure::sdf_render::worldgen::graph::preset::{
    MOUNTAINS_PLAINS_AMPLITUDE, default_terrain_graph, mountains_plains_graph,
};
use adventure::sdf_render::worldgen::graph::wgsl_codegen::EVAL_FN_NAME;
use adventure::sdf_render::worldgen::graph::{Graph, GraphAsset, graph_to_wgsl};
use naga_oil::compose::{Composer, NagaModuleDescriptor};
use std::path::Path;

/// Read the `worldgen_gpu.wgsl` library source with its `#define_import_path` line stripped, so it can be
/// concatenated directly into a self-contained entry shader (the existing engine shaders in
/// `tests/shader_validation.rs` are all self-contained too; naga_oil does NOT support `::*` wildcard
/// imports, and listing every public item by name would be brittle — concatenation keeps the library +
/// codegen validated together as one module).
fn worldgen_lib_source() -> String {
    let lib = std::fs::read_to_string("assets/shaders/worldgen_gpu.wgsl")
        .expect("read assets/shaders/worldgen_gpu.wgsl");
    lib.lines()
        .filter(|l| !l.trim_start().starts_with("#define_import_path"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build the full entry-shader source: the library + the generated `wg_eval_graph` + a tiny `@compute`
/// entry that calls it (and the erosion path) and stores the result so nothing is dead-code-eliminated
/// before validation. Returns the source string.
fn entry_source_for(graph: &Graph) -> String {
    let lib = worldgen_lib_source();
    let generated = graph_to_wgsl(graph);
    // Concatenate the library, the generated graph fn, then a compute entry that exercises BOTH the graph
    // eval AND the erosion path (so wg_erode_with_grad is covered by the same validation).
    format!(
        r#"{lib}

{generated}

@group(0) @binding(0) var<storage, read_write> out_buf: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let wx = f32(gid.x);
    let wz = f32(gid.y);
    let world_seed = gid.z;
    let f = {EVAL_FN_NAME}(wx, wz, world_seed);
    // Exercise the erosion stage on the graph's height + gradient (Hessian seeded zero here; the GPU
    // voxelizer will pass the real base Hessian — this only needs the path to type-check + validate).
    let erosion = WgErosionParams(1u, 55.0f, 5u, 640.0f, 2.0f, 0.5f, 0.6f, 0.6f, 14681360u);
    let eroded = wg_erode_with_grad(f.v, f.dx, f.dz, 0.0f, 0.0f, 0.0f, wx, wz, world_seed, erosion);
    out_buf[gid.x] = eroded.x + eroded.y + eroded.z;
}}
"#
    )
}

/// Compose + validate a graph's generated WGSL through the full-capability composer (mirrors
/// `tests/shader_validation.rs`). Returns `Ok(())` or a formatted error.
fn validate_graph_wgsl(label: &str, graph: &Graph) -> Result<(), String> {
    let mut composer = Composer::default().with_capabilities(naga::valid::Capabilities::all());
    let source = entry_source_for(graph);
    let module = composer
        .make_naga_module(NagaModuleDescriptor {
            source: &source,
            file_path: &format!("generated::{label}"),
            ..Default::default()
        })
        .map_err(|e| format!("compose {label} failed:\n{e}\n--- generated source ---\n{source}"))?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .map_err(|e| format!("WGSL validation error for {label}:\n{e:?}\n--- generated source ---\n{source}"))?;
    Ok(())
}

/// The hand-written `worldgen_gpu.wgsl` library composes + validates on its own (with a trivial entry
/// calling a couple of its helpers), independent of any codegen.
#[test]
fn worldgen_gpu_library_composes() {
    let lib = worldgen_lib_source();
    let source = format!(
        r#"{lib}

@group(0) @binding(0) var<storage, read_write> out_buf: array<f32>;

@compute @workgroup_size(1)
fn main() {{
    let f = wg_world_x(1.0f);
    out_buf[0] = f.v + f.dx + f.dz;
}}
"#
    );
    let mut composer = Composer::default().with_capabilities(naga::valid::Capabilities::all());
    let module = composer
        .make_naga_module(NagaModuleDescriptor { source: &source, file_path: "worldgen_gpu_lib", ..Default::default() })
        .unwrap_or_else(|e| panic!("worldgen_gpu.wgsl library must compose:\n{e}"));
    let mut validator =
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    validator.validate(&module).unwrap_or_else(|e| panic!("worldgen_gpu.wgsl library validation:\n{e:?}"));
}

/// The two in-code presets (`default_terrain_graph`, `mountains_plains_graph`) codegen + validate.
#[test]
fn code_presets_validate() {
    let carrier = FbmAxis { octaves: 6, base_freq: 1.0 / 1536.0, lacunarity: 2.0, gain: 0.5, amplitude: 280.0, seed_salt: 0 };
    let default = default_terrain_graph(carrier, 0.5, 280.0 * 1.96875, 0.0);
    validate_graph_wgsl("default_terrain_graph", &default).unwrap_or_else(|e| panic!("{e}"));

    let mtn = mountains_plains_graph(MOUNTAINS_PLAINS_AMPLITUDE);
    validate_graph_wgsl("mountains_plains_graph", &mtn).unwrap_or_else(|e| panic!("{e}"));
}

/// EVERY shipped `assets/worldgen/*.graph.ron` the engine loads codegens + validates. Includes
/// `world.graph.ron`, which uses `Scale` + `Add` (ops the presets don't), broadening op coverage.
#[test]
fn shipped_graph_ron_files_validate() {
    let dir = Path::new("assets/worldgen");
    let files = ["default.graph.ron", "mountains_plains.graph.ron", "world.graph.ron"];
    for file in files {
        let path = dir.join(file);
        let ron = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let asset: GraphAsset = ron::de::from_str(&ron)
            .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
        asset
            .graph
            .validate()
            .unwrap_or_else(|e| panic!("{} is not a valid graph: {e:?}", path.display()));
        validate_graph_wgsl(file, &asset.graph).unwrap_or_else(|e| panic!("{e}"));
    }
}
