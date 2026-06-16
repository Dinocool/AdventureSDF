//! WGSL shader validation test rig.
//!
//! Validates the SDF shaders at test time — no GPU, no window, no game launch. Catches syntax
//! errors, type mismatches, and invalid constructs before runtime.
//!
//! Since the mesh-bake pivot, the only SDF shader left is the GPU brick-bake compute shader
//! (`sdf_brick_bake.wgsl`); the surface raymarch + its `sdf/*.wgsl` import modules were removed.
//! The bake shader is fully self-contained (no `#import` of local modules), so it validates
//! directly with naga (composed only against the Bevy fullscreen stub, which it doesn't even use).

use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue, ShaderLanguage,
};
use std::collections::HashMap;
use std::path::Path;

/// A `bevy_core_pipeline` import some entry shaders use. naga_oil doesn't know Bevy's built-in
/// modules, so we register a minimal stand-in providing only `FullscreenVertexOutput`.
const FULLSCREEN_STUB: &str = r#"
#define_import_path bevy_core_pipeline::fullscreen_vertex_shader
struct FullscreenVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_position: vec3<f32>,
};
"#;

/// A minimal `bevy_pbr::forward_io` stand-in providing only the `VertexOutput` fields the worldgen
/// preview material shader reads (`uv`). naga_oil doesn't know Bevy's built-in PBR modules.
const FORWARD_IO_STUB: &str = r#"
#define_import_path bevy_pbr::forward_io
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};
"#;

/// Register the Bevy fullscreen stub into a fresh composer.
fn composer_with_stub() -> Composer {
    // Validate with FULL capabilities, exactly as bevy's runtime composer does via
    // `.with_capabilities(device_caps)` (pipeline_cache.rs). The default composer validates with
    // EMPTY capabilities, which rejects the paged atlas's non-uniform `binding_array` index that
    // the device + runtime accept.
    let mut composer = Composer::default().with_capabilities(naga::valid::Capabilities::all());
    for (source, path) in [
        (FULLSCREEN_STUB, "bevy_core_pipeline::fullscreen_vertex_shader"),
        (FORWARD_IO_STUB, "bevy_pbr::forward_io"),
    ] {
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source,
                file_path: path,
                language: ShaderLanguage::Wgsl,
                ..Default::default()
            })
            .unwrap_or_else(|e| panic!("stub {path} must compose: {e}"));
    }
    composer
}

/// Compose an entry shader (importing at most the Bevy stubs) and validate it, with the given
/// `#{…}` shader-def substitutions.
fn validate_entry_with_defs(path: &Path, defs: HashMap<String, ShaderDefValue>) -> Result<(), String> {
    let mut composer = composer_with_stub();
    let source =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let module = composer
        .make_naga_module(NagaModuleDescriptor {
            source: &source,
            file_path: &path.to_string_lossy(),
            shader_defs: defs,
            ..Default::default()
        })
        .map_err(|e| format!("compose {} failed:\n{e}", path.display()))?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .map_err(|e| format!("WGSL validation error in {}:\n{e:?}", path.display()))?;
    Ok(())
}

/// Compose an entry shader (importing at most the Bevy stubs) and validate it.
fn validate_entry(path: &Path) -> Result<(), String> {
    validate_entry_with_defs(path, HashMap::new())
}

#[test]
fn sdf_brick_bake_wgsl_validates() {
    // The brick-bake compute shader is fully self-contained (no sdf::* imports). Validates the
    // ported eval_primitive/fold_csg/material slots + the packed storage-buffer writes.
    let path = Path::new("assets/shaders/sdf_brick_bake.wgsl");
    validate_entry(path).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn voxel_raytrace_wgsl_validates() {
    // The HW-RT voxel raymarch (+ the DLSS-RR `raymarch_dlss` entry point added in Stage 4c + the Phase-2.1
    // world-cache passes). Fully self-contained (no `#import`), but uses `enable wgpu_ray_query` + the
    // `ray_query` types, so it needs the full-capability composer (which the device + runtime also use).
    // The world-cache section is parameterised by the `#{WORLD_CACHE_SIZE}` hash-table-size def (a small
    // power-of-two here for fast validation; the live path uses 2^20). Validating here catches a WGSL typo
    // without a GPU/launch.
    let path = Path::new("assets/shaders/voxel_raytrace.wgsl");
    let defs = HashMap::from([("WORLD_CACHE_SIZE".to_string(), ShaderDefValue::UInt(1024))]);
    validate_entry_with_defs(path, defs).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn voxel_pack_wgsl_validates() {
    // Phase G — the GPU brick PACK compute shader, ALL THREE entry points: `pack_brick` (G-a; one workgroup per
    // dirty dense brick, halo-fill + serial palette build + parallel bit-pack), `write_aabb` (G-b; one invocation
    // per changed slot → the BLAS AABB, resident `brick_aabb` / freed `degenerate_aabb`), AND `classify_brick` (G4;
    // one workgroup per dirty brick, shared halo-fill → per-brick is_uniform / palette_k / index_bits, the cheap
    // classification the CPU reads back so it stops `pack_one`'ing). Fully self-contained (no `#import`, no
    // shader-defs). Validating here catches a WGSL typo / workgroup-array overflow without a GPU.
    let path = Path::new("assets/shaders/voxel_pack.wgsl");
    validate_entry(path).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn voxel_rt_composite_wgsl_validates() {
    // The composite + the DLSS-RR `fs_resolve_dlss` resolve pass (multi-target colour+motion + frag_depth).
    let path = Path::new("assets/shaders/voxel_rt_composite.wgsl");
    validate_entry(path).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn worldgen_preview_wgsl_validates() {
    // The node-preview material shader: orbit raymarch of the baked heightfield. Uses
    // `@group(#{MATERIAL_BIND_GROUP})` + imports `bevy_pbr::forward_io::VertexOutput`, so it needs
    // the forward_io stub + the MATERIAL_BIND_GROUP def (2, matching Bevy's material bind group).
    let path = Path::new("assets/shaders/worldgen_preview.wgsl");
    let defs = HashMap::from([("MATERIAL_BIND_GROUP".to_string(), ShaderDefValue::UInt(2))]);
    validate_entry_with_defs(path, defs).unwrap_or_else(|e| panic!("{e}"));
}
