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

/// The SDF module files, in dependency order — the SAME list the render pipeline composes (single
/// source of truth: `sdf_render::render::SDF_SHADER_MODULES`), with the `assets/` root made explicit
/// for the filesystem reads here. A new `sdf/*.wgsl` module added to the pipeline is validated
/// automatically.
use adventure::sdf_render::render::SDF_SHADER_MODULES;

const SDF_ENTRY: &str = "assets/shaders/sdf_raymarch.wgsl";

/// Compose the SDF import graph into a single naga module, then validate it, with the
/// given shader defs enabled (so `#ifdef` debug branches are actually compiled).
fn validate_composed_sdf_with_defs(defs: &[&str]) -> Result<(), String> {
    validate_composed_entry(SDF_ENTRY, defs)
}

/// As above but for an arbitrary entry shader that imports the `sdf/` modules.
fn validate_composed_entry(entry: &str, defs: &[&str]) -> Result<(), String> {
    use naga_oil::compose::ShaderDefValue;
    use std::collections::HashMap;

    let mut composer = composer_with_stub();

    // Add each SDF module, dependencies first. The pipeline list is asset-server-relative; the
    // filesystem reads here need the explicit `assets/` root.
    for module in SDF_SHADER_MODULES {
        let path = format!("assets/{module}");
        let source = std::fs::read_to_string(&path).map_err(|e| format!("read {path}: {e}"))?;
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source: &source,
                file_path: &path,
                language: ShaderLanguage::Wgsl,
                ..Default::default()
            })
            .map_err(|e| format!("compose {path} failed: {e}"))?;
    }

    let shader_defs: HashMap<String, ShaderDefValue> = defs
        .iter()
        .map(|d| ((*d).to_string(), ShaderDefValue::Bool(true)))
        .collect();

    // Compose the entry shader (resolves all #import lines into one naga module).
    let entry_src =
        std::fs::read_to_string(entry).map_err(|e| format!("read {entry}: {e}"))?;
    let module = composer
        .make_naga_module(NagaModuleDescriptor {
            source: &entry_src,
            file_path: entry,
            shader_defs,
            ..Default::default()
        })
        .map_err(|e| format!("compose {entry} (defs={defs:?}) failed:\n{e}"))?;

    // naga_oil hands back a naga::Module directly; validate it.
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .map_err(|e| format!("WGSL validation error (defs={defs:?}):\n{e:?}"))?;
    Ok(())
}

/// Compose the SDF import graph into a single naga module, then validate it.
fn validate_composed_sdf() -> Result<(), String> {
    validate_composed_sdf_with_defs(&[])
}

/// Register the Bevy fullscreen stub into a fresh composer.
fn composer_with_stub() -> Composer {
    // Validate with FULL capabilities, exactly as bevy's runtime composer does via
    // `.with_capabilities(device_caps)` (pipeline_cache.rs). The default composer validates with
    // EMPTY capabilities, which rejects the paged atlas's non-uniform `binding_array` index
    // (`mat_pages[page]`) that the device + runtime accept. Without this the composer's internal
    // header validation fails with "Function 'load_mat' is invalid".
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


/// Each `#ifdef` debug branch must also compile + validate (they're skipped when no
/// def is set, so the default compose would miss errors inside them).
#[test]
fn sdf_cone_prepass_wgsl_validates() {
    // The cone-prepass compute shader imports the same sdf::* modules; compose + validate
    // the whole graph exactly as the GPU pipeline would.
    validate_composed_entry("assets/shaders/sdf_cone_prepass.wgsl", &[])
        .unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn sdf_brick_bake_wgsl_validates() {
    // The brick-bake compute shader is fully self-contained (no sdf::* imports), so it
    // composes against an empty composer. Validates the ported eval_primitive/fold_csg/
    // material slots + the packed storage-buffer writes.
    let path = Path::new("assets/shaders/sdf_brick_bake.wgsl");
    validate_entry(path).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn sdf_debug_modes_validate() {
    // The primary pass is mostly a pure G-buffer export, but a few debug overlays write their
    // visualization into albedo and return early (the lit pass passes it straight through):
    // SDF_DEBUG_LOD (eff-LOD hue ramp) and SDF_DEBUG_STEP_COUNT (march step-count heatmap). The
    // rest are march-internal toggles gating `#ifdef` branches in the brick/march modules.
    for def in [
        "SDF_DISABLE_CHUNK_CACHE",
        "SDF_DISABLE_LOD",
        "SDF_LINEAR_CHUNK_SEARCH",
        "SDF_DEBUG_LOD",
        "SDF_DEBUG_STEP_COUNT",
        "SDF_SECOND_ORDER_STEP",
    ] {
        validate_composed_sdf_with_defs(&[def]).unwrap_or_else(|e| panic!("{e}"));
    }
}

/// PBR feature toggles gate `#ifdef` branches that the default compose skips — validate each
/// so errors inside them are caught. (Reflections were removed from the primary pass; shadows
/// remain in `pbr.wgsl`, consumed by the composite.)
#[test]
fn sdf_feature_defs_validate() {
    // Only SDF_SHADOWS remains a standalone feature def (reflections were removed); validate it
    // composes on its own so errors inside its `#ifdef` branch are caught.
    validate_composed_sdf_with_defs(&["SDF_SHADOWS"]).unwrap_or_else(|e| panic!("{e}"));
}

/// Compose + validate a STANDALONE entry shader that imports only the named binding-free helper
/// modules (NOT the atlas-bound sdf::bindings graph) plus the fullscreen stub, with the given
/// shader defs enabled (so `#ifdef` debug branches actually compile).
fn validate_standalone_with_defs(entry: &str, modules: &[&str], defs: &[&str]) {
    use naga_oil::compose::ShaderDefValue;
    use std::collections::HashMap;

    let mut composer = composer_with_stub();
    for path in modules {
        let source = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source: &source,
                file_path: path,
                language: ShaderLanguage::Wgsl,
                ..Default::default()
            })
            .unwrap_or_else(|e| panic!("compose {path} failed: {e}"));
    }
    let shader_defs: HashMap<String, ShaderDefValue> = defs
        .iter()
        .map(|d| ((*d).to_string(), ShaderDefValue::Bool(true)))
        .collect();
    let entry_src = std::fs::read_to_string(entry).unwrap_or_else(|e| panic!("read {entry}: {e}"));
    let module = composer
        .make_naga_module(NagaModuleDescriptor {
            source: &entry_src,
            file_path: entry,
            shader_defs,
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("compose {entry} (defs={defs:?}) failed:\n{e}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("WGSL validation error in {entry} (defs={defs:?}):\n{e:?}"));
}

/// As above with no defs.
fn validate_standalone_with_modules(entry: &str, modules: &[&str]) {
    validate_standalone_with_defs(entry, modules, &[]);
}

#[test]
fn sdf_deferred_lit_wgsl_validates() {
    // The deferred lit pass imports `sdf::oct` + `sdf::brdf` (binding-free). Validate the default
    // (lit) build AND each `#ifdef`-gated debug-view branch so errors inside them are caught.
    let modules = ["assets/shaders/sdf/oct.wgsl", "assets/shaders/sdf/brdf.wgsl"];
    let entry = "assets/shaders/sdf_deferred_lit.wgsl";
    validate_standalone_with_modules(entry, &modules);
    for def in [
        "SDF_DEBUG_ALBEDO",
        "SDF_DEBUG_NORMALS",
        "SDF_DEBUG_METALLIC",
        "SDF_DEBUG_ROUGHNESS",
        "SDF_DEBUG_EMISSIVE",
        "SDF_DEBUG_SUN_VIS",
        "SDF_DEBUG_DEPTH",
        "SDF_DEBUG_STEP_COUNT",
    ] {
        validate_standalone_with_defs(entry, &modules, &[def]);
    }
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
