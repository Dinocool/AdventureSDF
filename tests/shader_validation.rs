//! WGSL shader validation test rig.
//!
//! Uses `naga` (the same WGSL frontend that wgpu uses) to parse and validate
//! all shader files at test time. Catches syntax errors, type mismatches, and
//! invalid constructs *before* runtime — no GPU, no window, no game launch.
//!
//! Limitations:
//! - naga validates WGSL in isolation. Bevy's `#import` directives and
//!   `#define` macros are handled by `naga-oil` at load time, which naga
//!   doesn't know about. We strip `#import` lines and inject stub definitions
//!   for common Bevy types.
//! - Pipeline layout validation (bind group compatibility) is not checked
//!   here — only WGSL syntax and type correctness.

use std::path::Path;

/// Bevy import stubs — minimal definitions that satisfy naga when `#import`
/// directives would normally be resolved by naga-oil.
const BEVY_STUBS: &str = r#"
struct FullscreenVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_position: vec3<f32>,
};
"#;

/// Preprocess a shader file: strip `#import`, `#ifndef`, `#ifdef`, `#endif`
/// lines (naga-oil directives that naga doesn't understand) and inject Bevy stubs.
/// For `#ifndef CONST / const X = val; / #endif` blocks, keep the const definition.
fn preprocess_shader(source: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    let skip_depth = 0;

    for line in source.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("#import") {
            continue;
        }

        // Handle #ifndef / #ifdef — strip the directive but keep the body
        if trimmed.starts_with("#ifndef") || trimmed.starts_with("#ifdef") {
            if skip_depth == 0 {
                // Keep the body lines (don't increment skip_depth for the directive itself)
            }
            continue;
        }

        if trimmed.starts_with("#endif") {
            continue;
        }

        // Strip other naga-oil preprocessor directives
        if trimmed.starts_with("#define") || trimmed.starts_with("#else") {
            continue;
        }

        lines.push(line.to_string());
    }

    format!("{BEVY_STUBS}\n{}\n", lines.join("\n"))
}

/// Validate a single WGSL file using naga.
fn validate_wgsl_file(path: &Path) -> Result<(), String> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;

    let processed = preprocess_shader(&source);

    let result = naga::front::wgsl::parse_str(&processed);
    let module = result.map_err(|e| {
        let mut report = format!("WGSL parse error in {}:\n{e}", path.display());
        report.push_str("\n\n--- Preprocessed source ---\n");
        for (i, line) in processed.lines().enumerate() {
            report.push_str(&format!("{:4}: {}\n", i + 1, line));
        }
        report
    })?;

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .map_err(|e| format!("WGSL validation error in {}:\n{e}", path.display()))?;

    Ok(())
}

/// Discover all .wgsl files under assets/shaders/
fn collect_shaders() -> Vec<std::path::PathBuf> {
    let dir = std::path::Path::new("assets/shaders");
    if !dir.exists() {
        return Vec::new();
    }
    let mut shaders: Vec<_> = walkdir(dir)
        .into_iter()
        .filter(|p| p.extension().is_some_and(|ext| ext == "wgsl"))
        .collect();
    shaders.sort();
    shaders
}

fn walkdir(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut result = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                result.extend(walkdir(&path));
            } else {
                result.push(path);
            }
        }
    }
    result
}

#[test]
fn all_wgsl_shaders_parse_and_validate() {
    let shaders = collect_shaders();
    assert!(
        !shaders.is_empty(),
        "No .wgsl files found in assets/shaders"
    );

    let mut failures = Vec::new();
    for path in &shaders {
        if let Err(e) = validate_wgsl_file(path) {
            failures.push(e);
        }
    }

    if !failures.is_empty() {
        let report = failures.join("\n\n");
        panic!("{report}");
    }
}

#[test]
fn sdf_raymarch_wgsl_validates() {
    validate_wgsl_file(std::path::Path::new("assets/shaders/sdf_raymarch.wgsl"))
        .unwrap_or_else(|e| panic!("{e}"));
}
