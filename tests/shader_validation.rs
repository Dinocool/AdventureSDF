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

#[test]
fn sdf_deferred_lit_wgsl_validates() {
    // The deferred lit pass now imports the full sdf::* graph (bindings + brick + probe for the
    // DDGI apply, plus oct + brdf). Compose the whole graph + validate the default (lit) build AND
    // each `#ifdef`-gated debug-view branch (incl. the new SDF_DEBUG_GI) so errors inside them are
    // caught exactly as the GPU pipeline would compile them.
    let entry = "assets/shaders/sdf_deferred_lit.wgsl";
    validate_composed_entry(entry, &[]).unwrap_or_else(|e| panic!("{e}"));
    for def in [
        "SDF_DEBUG_ALBEDO",
        "SDF_DEBUG_NORMALS",
        "SDF_DEBUG_METALLIC",
        "SDF_DEBUG_ROUGHNESS",
        "SDF_DEBUG_EMISSIVE",
        "SDF_DEBUG_SUN_VIS",
        "SDF_DEBUG_DEPTH",
        "SDF_DEBUG_STEP_COUNT",
        "SDF_DEBUG_GI",
    ] {
        validate_composed_entry(entry, &[def]).unwrap_or_else(|e| panic!("{e}"));
    }
}

#[test]
fn sdf_probe_trace_wgsl_validates() {
    // The probe-trace compute shader imports the full sdf::* graph (raymarch, material, sky,
    // shadows, probe) plus its own group(3) probe buffers. Compose + validate exactly as the GPU
    // pipeline will — catches every type / binding / call error before any render wiring exists.
    validate_composed_entry("assets/shaders/sdf_probe_trace.wgsl", &[])
        .unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn sdf_gi_resolve_wgsl_validates() {
    // The GI-resolve fragment shader imports the sdf::* probe-addressing graph (bindings, oct, probe)
    // and evaluates sample_gi into a texture. Compose + validate as the pipeline will.
    validate_composed_entry("assets/shaders/sdf_gi_resolve.wgsl", &[])
        .unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn sdf_gi_blur_wgsl_validates() {
    // The edge-aware à-trous GI blur (imports sdf::oct only).
    validate_composed_entry("assets/shaders/sdf_gi_blur.wgsl", &[])
        .unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn sdf_probe_module_validates() {
    // `sdf/probe.wgsl` (DDGI probe addressing) imports `sdf::bindings` + `sdf::brick`. Compose the
    // full sdf graph + a tiny entry that actually CALLS its functions (incl. probe_slot_at, which
    // reaches the chunk directory), so the function bodies are validated (adding a composable module
    // alone doesn't validate bodies that nothing instantiates).
    let mut composer = composer_with_stub();
    for module in SDF_SHADER_MODULES {
        let path = format!("assets/{module}");
        let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source: &src,
                file_path: &path,
                language: ShaderLanguage::Wgsl,
                ..Default::default()
            })
            .unwrap_or_else(|e| panic!("compose {path} failed: {e}"));
    }
    let entry = r#"
#import sdf::probe::{probe_world_pos, subprobe_world_pos, probe_slot_at, decode_chunk_key, brick_coord_in_chunk}
@compute @workgroup_size(1)
fn main() {
    let a = probe_world_pos(vec3<i32>(0, 0, 0), 0u);
    let b = subprobe_world_pos(vec3<i32>(1, 2, 3), 1u, vec3<i32>(0, 0, 0), 2u);
    let id = decode_chunk_key(0u, 0u);
    let bc = brick_coord_in_chunk(id.coord, 5u);
    let s = probe_slot_at(bc, id.lod);
    _ = a.x + b.y + f32(s);
}
"#;
    let module = composer
        .make_naga_module(NagaModuleDescriptor {
            source: entry,
            file_path: "probe_validate.wgsl",
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("compose probe entry failed:\n{e}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("WGSL validation error in sdf::probe:\n{e:?}"));
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
