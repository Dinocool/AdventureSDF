//! Build script. Its only job: when built with the `dlss` feature, copy the NVIDIA DLSS runtime DLLs
//! from the DLSS SDK to the target profile directory (next to the exe), so `dlss_wgpu` can load them at
//! runtime without the user manually copying them. (`dlss_wgpu`'s own build script already requires the
//! `DLSS_SDK` env var, so if we're here with the feature on, the SDK path is set.)
//!
//! Mirrors the proven solari-gi worktree build.rs. No-op without `--features dlss`.

use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=DLSS_SDK");

    // Cargo sets `CARGO_FEATURE_<NAME>` for each enabled feature.
    if env::var_os("CARGO_FEATURE_DLSS").is_none() {
        return;
    }
    let Some(dlss_sdk) = env::var_os("DLSS_SDK") else {
        return;
    };
    // OUT_DIR = <target>/<profile>/build/<pkg>-<hash>/out → the profile dir (next to the exe) is 4 up.
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by cargo"));
    let Some(profile_dir) = out_dir.ancestors().nth(3) else {
        return;
    };
    // Windows-only dev target (matches the rest of the project). The DLSS Ray Reconstruction DLL is
    // `nvngx_dlssd.dll`; `nvngx_dlss.dll` is the super-resolution DLL (loaded alongside).
    let rel = PathBuf::from(dlss_sdk).join("lib/Windows_x86_64/rel");
    for dll in ["nvngx_dlss.dll", "nvngx_dlssd.dll"] {
        let src = rel.join(dll);
        let dst = profile_dir.join(dll);
        if src.exists() {
            if let Err(e) = fs::copy(&src, &dst) {
                println!("cargo:warning=failed to copy {dll} next to the exe: {e}");
            }
        } else {
            println!("cargo:warning=DLSS DLL not found: {}", src.display());
        }
    }
}
