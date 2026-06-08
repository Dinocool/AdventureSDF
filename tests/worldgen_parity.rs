//! Determinism parity harness for **authoritative** world-gen layers — the CI gate that protects
//! shared-seed multiplayer (WORLD_GEN_PLAN §2.8, §9 phase 1).
//!
//! Every client generates the world independently from the seed and must agree bit-for-bit on
//! gameplay-relevant terrain, across GPU vendors / CPUs / OSes. A silent determinism regression (a
//! "clever" reorder into an FMA, a constant tweak, a transcendental creeping in) would desync the
//! world. So this pins the *exact IEEE bit patterns* of the noise basis + the height layer at a fixed
//! set of hazard coordinates; any drift fails CI loud.
//!
//! These are plain `#[test]`s (no GPU, no `#[ignore]`) so they always run in CI — unlike the GPU
//! rigs that skip without an adapter. Mirrors the constant-pinning discipline of
//! `chunk::wgsl_chunk_constants_match_rust` / `light_grid::wgsl_light_grid_constants_match_rust`, but
//! pins *values* (not just constants).
//!
//! ## Updating the reference vectors
//! If the height layer's output intentionally changes, regenerate the pinned tables:
//! ```text
//! cargo test --features editor --test worldgen_parity -- --ignored print_reference_vectors --nocapture
//! ```
//! Paste the printed literals over the tables below and bump `HEIGHT_GEN_VERSION`. The generator only
//! *prints* (never writes a file), so the change is explicit and review-visible — never auto-healed.

use adventure::sdf_render::worldgen::layers::height::{HEIGHT_GEN_VERSION, HeightLayer, HeightParams};
use adventure::sdf_render::worldgen::{coord::LayerId, noise};

/// Hazard coordinates for the integer hash: origin, ±1 on each axis, asymmetric, large magnitude,
/// and the i32 extremes (the negative/overflow-coord bug class this engine repeatedly hit).
const HASH_POINTS: &[(i32, i32, u32)] = &[
    (0, 0, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, -1, 0),
    (123, -456, 7),
    (-100_000, 250_000, 42),
    (i32::MAX, i32::MIN, 1),
];

/// Hazard coordinates for the height sample: origin, asymmetric pos/neg, far-from-origin (f64
/// precision), a chunk boundary, sub-zero, and distinct seeds.
const HEIGHT_POINTS: &[(f64, f64, u64)] = &[
    (0.0, 0.0, 1),
    (123.5, -456.25, 1),
    (-789.0, 1011.0, 1),
    (1_000_000.5, -2_000_000.25, 1),
    (128.0, 0.0, 1),
    (-0.001, -0.001, 1),
    (12.0, 34.0, 2),
    (12.0, 34.0, 999_999),
];

/// Pinned `hash2` outputs for `HASH_POINTS`, in order. (Filled from the generator.)
const HASH_REFERENCE: &[u32] =
    &[0, 301794027, 387900469, 3507803474, 1805801058, 3561151963, 3689159429];

/// Pinned height-layer outputs (height/∂x/∂z bit patterns) for `HEIGHT_POINTS`, in order, using
/// `HeightParams::default()`. (Filled from the generator.) Changing the default params changes these
/// — bump `HEIGHT_GEN_VERSION`.
const HEIGHT_REFERENCE: &[(u32, u32, u32)] = &[
    (1091346484, 0, 0),
    (1107985926, 3173698594, 3181065001),
    (3249860909, 1049238806, 3172095744),
    (1082975066, 1028616013, 1009231171),
    (1100516248, 1024126484, 0),
    (1091346484, 2954790304, 2943086156),
    (1091368145, 982344722, 1027223944),
    (3251513402, 986105929, 3185347093),
];

fn default_layer() -> HeightLayer {
    HeightLayer::new(LayerId(0), HeightParams::default())
}

/// THE gate: the integer hash basis must reproduce its pinned bit values exactly.
#[test]
fn hash_matches_reference_vectors() {
    assert_eq!(
        HASH_REFERENCE.len(),
        HASH_POINTS.len(),
        "reference table out of sync with HASH_POINTS — regenerate (see module docs)"
    );
    for (&(ix, iz, seed), &expect) in HASH_POINTS.iter().zip(HASH_REFERENCE) {
        let got = noise::hash2(ix, iz, seed);
        assert_eq!(
            got, expect,
            "hash2 drift at ({ix},{iz},{seed}): {got} != pinned {expect} — \
             a determinism regression would desync multiplayer (or bump HEIGHT_GEN_VERSION if intended)"
        );
    }
}

/// THE gate: the authoritative height layer must reproduce its pinned bit values exactly.
#[test]
fn height_layer_matches_reference_vectors() {
    assert_eq!(HEIGHT_REFERENCE.len(), HEIGHT_POINTS.len(), "reference table out of sync — regenerate");
    let layer = default_layer();
    for (&(wx, wz, seed), &(eh, edx, edz)) in HEIGHT_POINTS.iter().zip(HEIGHT_REFERENCE) {
        let n = layer.sample_world(wx, wz, seed);
        assert_eq!(n.height.to_bits(), eh, "height drift at ({wx},{wz},{seed})");
        assert_eq!(n.dh_dx.to_bits(), edx, "∂x drift at ({wx},{wz},{seed})");
        assert_eq!(n.dh_dz.to_bits(), edz, "∂z drift at ({wx},{wz},{seed})");
    }
}

/// The reference set must include the hazard cases (negative coords, origin, large magnitude, ≥2
/// seeds), so a future "fix" can't quietly delete the hard inputs that historically broke this engine.
#[test]
fn reference_vectors_cover_hazard_coords() {
    assert!(HASH_POINTS.iter().any(|&(x, _, _)| x < 0), "need a negative-x hash case");
    assert!(HASH_POINTS.iter().any(|&(_, z, _)| z < 0), "need a negative-z hash case");
    assert!(HASH_POINTS.iter().any(|&(x, z, _)| x == 0 && z == 0), "need the origin");
    assert!(HEIGHT_POINTS.iter().any(|&(x, _, _)| x < 0.0), "need a negative-x height case");
    assert!(HEIGHT_POINTS.iter().any(|&(x, _, _)| x.abs() > 100_000.0), "need a far-from-origin case");
    let seeds: std::collections::HashSet<u64> = HEIGHT_POINTS.iter().map(|&(_, _, s)| s).collect();
    assert!(seeds.len() >= 2, "need ≥2 distinct seeds");
}

/// Determinism without pinned values: recomputing any point yields byte-identical results, and the
/// gen-version is a positive, intentional number.
#[test]
fn recompute_is_bit_identical() {
    let layer = default_layer();
    for &(wx, wz, seed) in HEIGHT_POINTS {
        let a = layer.sample_world(wx, wz, seed);
        let b = layer.sample_world(wx, wz, seed);
        assert_eq!(a.height.to_bits(), b.height.to_bits());
        assert_eq!(a.dh_dx.to_bits(), b.dh_dx.to_bits());
        assert_eq!(a.dh_dz.to_bits(), b.dh_dz.to_bits());
    }
}

/// Compile-time guard: the gen-version must be a real (≥1) version. A 0 would mean "unset" and
/// silently disable the disk-cache keying / reference-vector versioning. Checked at compile time so
/// it costs nothing at runtime (and avoids a constant-value runtime assertion).
const _: () = assert!(HEIGHT_GEN_VERSION >= 1, "HEIGHT_GEN_VERSION must be >= 1");

/// Regenerate the reference tables. `#[ignore]` — run explicitly (see module docs), prints
/// pasteable Rust. Never writes a file, so updating the gate is a deliberate, reviewable edit.
#[test]
#[ignore = "prints reference vectors for paste; run explicitly when intentionally changing output"]
fn print_reference_vectors() {
    println!("\n// --- paste into HASH_REFERENCE ---");
    print!("const HASH_REFERENCE: &[u32] = &[");
    for &(ix, iz, seed) in HASH_POINTS {
        print!("{}, ", noise::hash2(ix, iz, seed));
    }
    println!("];");

    let layer = default_layer();
    println!("// --- paste into HEIGHT_REFERENCE (HEIGHT_GEN_VERSION = {HEIGHT_GEN_VERSION}) ---");
    println!("const HEIGHT_REFERENCE: &[(u32, u32, u32)] = &[");
    for &(wx, wz, seed) in HEIGHT_POINTS {
        let n = layer.sample_world(wx, wz, seed);
        println!("    ({}, {}, {}),", n.height.to_bits(), n.dh_dx.to_bits(), n.dh_dz.to_bits());
    }
    println!("];");
}
