//! WGSL shader validation test rig.
//!
//! Validates the SDF shaders at test time — no GPU, no window, no game launch. Catches syntax
//! errors, type mismatches, and invalid constructs before runtime.
//!
//! Since the mesh-bake pivot, the only SDF shader left is the GPU brick-bake compute shader
//! (`sdf_brick_bake.wgsl`); the surface raymarch + its `sdf/*.wgsl` import modules were removed.
//! The bake shader is fully self-contained (no `#import` of local modules), so it validates
//! directly with naga (composed only against the Bevy fullscreen stub, which it doesn't even use).
//! The baked-mesh material shader (`mesh_pbr.wgsl`) is an `ExtendedMaterial` shader validated by
//! Bevy's runtime pipeline, not here.

use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderLanguage,
};
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

/// Register the Bevy fullscreen stub into a fresh composer.
fn composer_with_stub() -> Composer {
    // Validate with FULL capabilities, exactly as bevy's runtime composer does via
    // `.with_capabilities(device_caps)` (pipeline_cache.rs). The default composer validates with
    // EMPTY capabilities, which rejects the paged atlas's non-uniform `binding_array` index that
    // the device + runtime accept.
    let mut composer = Composer::default().with_capabilities(naga::valid::Capabilities::all());
    composer
        .add_composable_module(ComposableModuleDescriptor {
            source: FULLSCREEN_STUB,
            file_path: "bevy_core_pipeline::fullscreen_vertex_shader",
            language: ShaderLanguage::Wgsl,
            ..Default::default()
        })
        .expect("fullscreen stub must compose");
    composer
}

/// Compose an entry shader (importing at most the Bevy fullscreen stub) and validate it.
fn validate_entry(path: &Path) -> Result<(), String> {
    let mut composer = composer_with_stub();
    let source =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let module = composer
        .make_naga_module(NagaModuleDescriptor {
            source: &source,
            file_path: &path.to_string_lossy(),
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

#[test]
fn sdf_brick_bake_wgsl_validates() {
    // The brick-bake compute shader is fully self-contained (no sdf::* imports). Validates the
    // ported eval_primitive/fold_csg/material slots + the packed storage-buffer writes.
    let path = Path::new("assets/shaders/sdf_brick_bake.wgsl");
    validate_entry(path).unwrap_or_else(|e| panic!("{e}"));
}
