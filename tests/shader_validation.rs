//! WGSL shader validation test rig.
//!
//! Validates all shader files at test time — no GPU, no window, no game launch.
//! Catches syntax errors, type mismatches, and invalid constructs before runtime.
//!
//! The SDF shader is split into `#import`-composed modules under `shaders/sdf/`, so
//! we resolve them with `naga_oil`'s `Composer` (the same library Bevy's ShaderCache
//! uses at runtime) before validating — composing the whole import graph, exactly
//! as the GPU pipeline would. Standalone files (no `#import` of local modules) are
//! still validated directly with naga.

use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderLanguage,
};
use std::path::Path;

/// A `bevy_core_pipeline` import the SDF entry shader uses. naga_oil doesn't know
/// Bevy's built-in modules, so we register a minimal stand-in providing only the
/// `FullscreenVertexOutput` the entry shader imports.
const FULLSCREEN_STUB: &str = r#"
#define_import_path bevy_core_pipeline::fullscreen_vertex_shader
struct FullscreenVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_position: vec3<f32>,
};
"#;

/// The SDF module files, in dependency order (a module must be added before any
/// module that imports it). The entry shader is composed last via `make_naga_module`.
const SDF_MODULES: [&str; 6] = [
    "assets/shaders/sdf/bindings.wgsl",
    "assets/shaders/sdf/brick.wgsl",
    "assets/shaders/sdf/cubic.wgsl",
    "assets/shaders/sdf/bvh.wgsl",
    "assets/shaders/sdf/material.wgsl",
    "assets/shaders/sdf/pbr.wgsl",
];

const SDF_ENTRY: &str = "assets/shaders/sdf_raymarch.wgsl";

/// Compose the SDF import graph into a single naga module, then validate it.
fn validate_composed_sdf() -> Result<(), String> {
    let mut composer = composer_with_stub();

    // Add each SDF module, dependencies first.
    for path in SDF_MODULES {
        let source = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source: &source,
                file_path: path,
                language: ShaderLanguage::Wgsl,
                ..Default::default()
            })
            .map_err(|e| format!("compose {path} failed: {e}"))?;
    }

    // Compose the entry shader (resolves all #import lines into one naga module).
    let entry_src =
        std::fs::read_to_string(SDF_ENTRY).map_err(|e| format!("read {SDF_ENTRY}: {e}"))?;
    let module = composer
        .make_naga_module(NagaModuleDescriptor {
            source: &entry_src,
            file_path: SDF_ENTRY,
            ..Default::default()
        })
        .map_err(|e| format!("compose {SDF_ENTRY} failed:\n{e}"))?;

    // naga_oil hands back a naga::Module directly; validate it.
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .map_err(|e| format!("WGSL validation error in composed SDF shader:\n{e:?}"))?;
    Ok(())
}

/// Register the Bevy fullscreen stub into a fresh composer.
fn composer_with_stub() -> Composer {
    let mut composer = Composer::default();
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

/// Compose an entry shader that only imports the Bevy fullscreen stub (no local
/// `sdf::*` modules), then validate. Used for self-contained shaders like
/// `sdf_debug.wgsl`.
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
fn sdf_raymarch_wgsl_validates() {
    validate_composed_sdf().unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn standalone_shaders_validate() {
    // Self-contained entry shaders that only import the Bevy fullscreen stub (the
    // `sdf/` modules are validated composed via `sdf_raymarch_wgsl_validates`).
    let entries = ["assets/shaders/sdf_debug.wgsl"];
    let mut failures = Vec::new();
    for path in entries {
        let p = Path::new(path);
        if p.exists()
            && let Err(e) = validate_entry(p)
        {
            failures.push(e);
        }
    }
    if !failures.is_empty() {
        panic!("{}", failures.join("\n\n"));
    }
}
