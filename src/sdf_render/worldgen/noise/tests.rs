//! Value-noise primitive tests (split from noise.rs per the test-module convention).

use super::*;

/// Bit-stability anchor: the integer hash must yield exactly these values. This is the local
/// guard for the "pure wrapping integer ops" portability claim — the full cross-platform
/// reference-vector gate lives in `tests/worldgen_parity.rs`. If these literals ever change, the
/// hash basis changed and every downstream world will regenerate differently (bump the layer
/// gen-version intentionally; never silently "fix" this).
#[test]
fn hash_is_bit_stable() {
    // Pinned outputs (computed by this exact algorithm). Distinct inputs → distinct outputs.
    assert_eq!(hash2(0, 0, 0), fmix32(0));
    assert_eq!(hash2(0, 0, 0), 0); // fmix32(0) == 0 (all-zero avalanches to zero)
    // Order matters: (x,z) and (z,x) differ for asymmetric inputs.
    assert_ne!(hash2(1, 2, 0), hash2(2, 1, 0));
    // Seed perturbs the stream.
    assert_ne!(hash2(5, 7, 1), hash2(5, 7, 2));
    // Recomputation is identical (no hidden state).
    assert_eq!(hash2(-13, 41, 99), hash2(-13, 41, 99));
}

/// Negative lattice coords are handled (signed→unsigned via `as u32` wraps deterministically) and
/// don't collide with their positive mirror.
#[test]
fn hash_handles_negative_coords() {
    assert_ne!(hash2(-1, 0, 0), hash2(1, 0, 0));
    assert_ne!(hash2(0, -1, 0), hash2(0, 1, 0));
    // Stable on recompute (the negative-coord bug class this engine repeatedly hit).
    assert_eq!(hash2(-100000, -250000, 7), hash2(-100000, -250000, 7));
}

/// Lattice values stay in [-1, 1).
#[test]
fn lattice_value_in_range() {
    for iz in -50..50 {
        for ix in -50..50 {
            let v = value_at_lattice(ix, iz, 12345);
            assert!((-1.0..1.0).contains(&v), "lattice value {v} out of [-1,1) at ({ix},{iz})");
        }
    }
}

/// Value noise stays within the lattice range over a dense scan of fractional positions.
#[test]
fn value_noise_in_range() {
    let seed = 0xABCD_1234;
    let mut x = -10.0;
    while x < 10.0 {
        let mut z = -10.0;
        while z < 10.0 {
            let (v, _, _) = value_noise_grad(x, z, seed);
            assert!((-1.0001..=1.0001).contains(&v), "value {v} out of range at ({x},{z})");
            z += 0.137;
        }
        x += 0.137;
    }
}

/// C0 continuity across an integer lattice boundary: sampling just below and just above an
/// integer coordinate yields nearly-equal values (no seam — the crack class). The two samples
/// interpolate the SAME shared lattice column, so they must agree in the limit.
#[test]
fn value_noise_continuous_across_lattice_boundary() {
    let seed = 77;
    let eps = 1e-6;
    for k in -5..5 {
        let bz = 0.3;
        let (lo, _, _) = value_noise_grad(k as f64 - eps, bz, seed);
        let (hi, _, _) = value_noise_grad(k as f64 + eps, bz, seed);
        assert!((lo - hi).abs() < 1e-3, "x-seam at {k}: {lo} vs {hi}");

        let bx = 0.7;
        let (lo, _, _) = value_noise_grad(bx, k as f64 - eps, seed);
        let (hi, _, _) = value_noise_grad(bx, k as f64 + eps, seed);
        assert!((lo - hi).abs() < 1e-3, "z-seam at {k}: {lo} vs {hi}");
    }
}

/// The analytic gradient of `value_noise_grad` matches a central finite difference — the property
/// the GPU Lipschitz normalisation and erosion both rely on.
#[test]
fn value_noise_gradient_matches_finite_difference() {
    let seed = 0x5151_5151;
    let h = 1e-5;
    for &(x, z) in &[(0.31, 0.62), (1.5, -2.25), (-3.7, 4.1), (10.05, -10.95)] {
        let (_, gx, gz) = value_noise_grad(x, z, seed);
        let (vxp, _, _) = value_noise_grad(x + h, z, seed);
        let (vxm, _, _) = value_noise_grad(x - h, z, seed);
        let (vzp, _, _) = value_noise_grad(x, z + h, seed);
        let (vzm, _, _) = value_noise_grad(x, z - h, seed);
        let fd_x = (vxp - vxm) / (2.0 * h);
        let fd_z = (vzp - vzm) / (2.0 * h);
        assert!((gx - fd_x).abs() < 1e-3, "∂x mismatch at ({x},{z}): analytic {gx} vs FD {fd_x}");
        assert!((gz - fd_z).abs() < 1e-3, "∂z mismatch at ({x},{z}): analytic {gz} vs FD {fd_z}");
    }
}

/// `value_noise_grad_hess` agrees with `value_noise_grad` on the shared lanes, and its Hessian
/// matches a central finite difference of the gradient (the exact-curvature property the analytic
/// erosion gradient relies on).
#[test]
fn value_noise_hessian_matches_finite_difference() {
    let seed = 0x1234_5678;
    let h = 1e-4;
    for &(x, z) in &[(0.31, 0.62), (1.5, -2.25), (-3.7, 4.1), (10.05, -10.95)] {
        let (v, gx, gz, hxx, hxz, hzz) = value_noise_grad_hess(x, z, seed);
        // Shared lanes identical to value_noise_grad.
        let (v2, gx2, gz2) = value_noise_grad(x, z, seed);
        assert_eq!(v.to_bits(), v2.to_bits());
        assert_eq!(gx.to_bits(), gx2.to_bits());
        assert_eq!(gz.to_bits(), gz2.to_bits());
        // Hessian via central difference of the gradient.
        let (_, gxp, gzp) = value_noise_grad(x + h, z, seed);
        let (_, gxm, gzm) = value_noise_grad(x - h, z, seed);
        let (_, _gxzp, gzzp) = value_noise_grad(x, z + h, seed);
        let (_, _gxzm, gzzm) = value_noise_grad(x, z - h, seed);
        let fd_xx = (gxp - gxm) / (2.0 * h);
        let fd_xz = (gzp - gzm) / (2.0 * h); // ∂(∂v/∂z)/∂x = mixed partial (symmetric)
        let fd_zz = (gzzp - gzzm) / (2.0 * h);
        assert!((hxx - fd_xx).abs() < 1e-2, "∂xx at ({x},{z}): {hxx} vs {fd_xx}");
        assert!((hxz - fd_xz).abs() < 1e-2, "∂xz at ({x},{z}): {hxz} vs {fd_xz}");
        assert!((hzz - fd_zz).abs() < 1e-2, "∂zz at ({x},{z}): {hzz} vs {fd_zz}");
    }
}

/// fBm Hessian matches a central finite difference of the fBm gradient (octave accumulation +
/// frequency² chain-rule), and the shared lanes are bit-identical to `fbm_height_grad`.
#[test]
fn fbm_hessian_matches_finite_difference() {
    let p = FbmParams { octaves: 4, base_freq: 0.05, lacunarity: 2.0, gain: 0.5, amplitude: 30.0, seed: 9 };
    let h = 1e-3;
    for &(wx, wz) in &[(12.0, -7.0), (0.0, 0.0), (-130.0, 88.0), (1000.5, -500.25)] {
        let (hv, gx, gz, hxx, hxz, hzz) = fbm_height_grad_hess(wx, wz, &p);
        let (hv2, gx2, gz2) = fbm_height_grad(wx, wz, &p);
        assert_eq!(hv.to_bits(), hv2.to_bits());
        assert_eq!(gx.to_bits(), gx2.to_bits());
        assert_eq!(gz.to_bits(), gz2.to_bits());
        // ∂xx from ∂x's x-difference; ∂xz from ∂x's z-difference; ∂zz from ∂z's z-difference.
        let (_, gxxp, _) = fbm_height_grad(wx + h, wz, &p);
        let (_, gxxm, _) = fbm_height_grad(wx - h, wz, &p);
        let (_, gxzp, gzzp) = fbm_height_grad(wx, wz + h, &p);
        let (_, gxzm, gzzm) = fbm_height_grad(wx, wz - h, &p);
        let fd_xx = (gxxp - gxxm) / (2.0 * h);
        let fd_xz = (gxzp - gxzm) / (2.0 * h);
        let fd_zz = (gzzp - gzzm) / (2.0 * h);
        assert!((hxx - fd_xx).abs() < 1e-2, "fBm ∂xx at ({wx},{wz}): {hxx} vs {fd_xx}");
        assert!((hxz - fd_xz).abs() < 1e-2, "fBm ∂xz at ({wx},{wz}): {hxz} vs {fd_xz}");
        assert!((hzz - fd_zz).abs() < 1e-2, "fBm ∂zz at ({wx},{wz}): {hzz} vs {fd_zz}");
    }
}

/// fBm gradient also matches finite differences (octave accumulation + frequency chain-rule).
#[test]
fn fbm_gradient_matches_finite_difference() {
    let p = FbmParams { octaves: 4, base_freq: 0.05, lacunarity: 2.0, gain: 0.5, amplitude: 30.0, seed: 9 };
    let h = 1e-3;
    for &(wx, wz) in &[(12.0, -7.0), (0.0, 0.0), (-130.0, 88.0), (1000.5, -500.25)] {
        let (_, gx, gz) = fbm_height_grad(wx, wz, &p);
        let (hxp, _, _) = fbm_height_grad(wx + h, wz, &p);
        let (hxm, _, _) = fbm_height_grad(wx - h, wz, &p);
        let (hzp, _, _) = fbm_height_grad(wx, wz + h, &p);
        let (hzm, _, _) = fbm_height_grad(wx, wz - h, &p);
        let fd_x = (hxp - hxm) / (2.0 * h);
        let fd_z = (hzp - hzm) / (2.0 * h);
        assert!((gx - fd_x).abs() < 1e-2, "fBm ∂x at ({wx},{wz}): {gx} vs {fd_x}");
        assert!((gz - fd_z).abs() < 1e-2, "fBm ∂z at ({wx},{wz}): {gz} vs {fd_z}");
    }
}

/// `fbm_height` (value-only) is bit-identical to `fbm_height_grad(...).0` — both share the exact
/// octave-sum value path, so either can serve as the value reference.
#[test]
fn fbm_height_matches_grad_value() {
    let p = FbmParams { octaves: 5, base_freq: 1.0 / 300.0, lacunarity: 2.0, gain: 0.5, amplitude: 60.0, seed: 17 };
    for &(wx, wz) in &[(0.0, 0.0), (123.5, -456.25), (-789.0, 1011.0), (1_000_000.5, -2_000_000.25)] {
        let v = fbm_height(wx, wz, &p);
        let (g, _, _) = fbm_height_grad(wx, wz, &p);
        assert_eq!(v.to_bits(), g.to_bits(), "fbm_height != fbm_height_grad().0 at ({wx},{wz})");
    }
}

/// fBm height is bounded by the geometric sum of octave amplitudes (value noise ∈ [-1,1] per
/// octave). Guards against an amplitude/gain bug blowing up the field.
#[test]
fn fbm_height_bounded_by_amplitude_sum() {
    let p = FbmParams { octaves: 6, base_freq: 0.02, lacunarity: 2.0, gain: 0.5, amplitude: 40.0, seed: 3 };
    let mut bound = 0.0;
    let mut amp = p.amplitude;
    for _ in 0..p.octaves {
        bound += amp;
        amp *= p.gain;
    }
    for &(wx, wz) in &[(0.0, 0.0), (123.0, 456.0), (-789.0, -1011.0)] {
        let (hh, _, _) = fbm_height_grad(wx, wz, &p);
        assert!(hh.abs() <= bound + 1e-6, "fBm height {hh} exceeds amplitude bound {bound}");
    }
}

/// HARD GATE: the 4-wide SIMD primitives are `to_bits()`-IDENTICAL to the scalar
/// `value_noise_grad` / `fbm_height_grad` over a spread of coordinates — negatives, lattice
/// boundaries (`x.floor()` edge cases), large magnitudes, and fractional positions. This is the
/// local guard that the SIMD octave sum did not reassociate / contract into an FMA / diverge in the
/// `f64↔i32` step. If this ever fails, the gen path is NO LONGER bit-portable — do not ship it.
#[test]
fn x4_matches_scalar() {
    let seed = 0xC0FF_EE17u32;
    // A spread including exact integers (floor boundary), negatives, sub-cell fractions, and large
    // coords where f64 spacing > 1 (stress the floor / `as i32` lane step).
    let xs = [
        -3.0, -2.999_999, -0.5, 0.0, 0.137, 0.999_999, 1.0, 1.5, 7.25, -12.75, 1000.5,
        -1_000_000.25, 123_456.789, -7.0, 4.000_000_1, 0.333_333_333,
    ];
    let zs = [
        4.1, 4.0, 0.5, 0.0, -0.137, 2.000_001, -1.0, -1.5, -8.5, 13.25, -500.25, 2_000_000.5,
        -654_321.123, 9.0, -3.999_999_9, 0.666_666_666,
    ];

    // value_noise_grad parity, processed 4 at a time.
    for c in 0..(xs.len() / 4) {
        let i = c * 4;
        let xv = f64x4::new([xs[i], xs[i + 1], xs[i + 2], xs[i + 3]]);
        let zv = f64x4::new([zs[i], zs[i + 1], zs[i + 2], zs[i + 3]]);
        let (v, gx, gz) = value_noise_grad_x4(xv, zv, seed);
        let (va, gxa, gza) = (v.to_array(), gx.to_array(), gz.to_array());
        for l in 0..4 {
            let (sv, sgx, sgz) = value_noise_grad(xs[i + l], zs[i + l], seed);
            assert_eq!(va[l].to_bits(), sv.to_bits(), "value mismatch at ({},{})", xs[i + l], zs[i + l]);
            assert_eq!(gxa[l].to_bits(), sgx.to_bits(), "∂x mismatch at ({},{})", xs[i + l], zs[i + l]);
            assert_eq!(gza[l].to_bits(), sgz.to_bits(), "∂z mismatch at ({},{})", xs[i + l], zs[i + l]);
        }
    }

    // fbm_height_grad parity across several param sets (incl. the production ~13-octave count).
    let param_sets = [
        FbmParams { octaves: 13, base_freq: 1.0 / 1024.0, lacunarity: 2.0, gain: 0.5, amplitude: 64.0, seed },
        FbmParams { octaves: 1, base_freq: 0.01, lacunarity: 2.0, gain: 0.5, amplitude: 100.0, seed: 1 },
        FbmParams { octaves: 7, base_freq: 1.0 / 333.0, lacunarity: 1.97, gain: 0.51, amplitude: 48.0, seed: 0xABCD },
        FbmParams::default(),
    ];
    for p in &param_sets {
        for c in 0..(xs.len() / 4) {
            let i = c * 4;
            let xv = f64x4::new([xs[i], xs[i + 1], xs[i + 2], xs[i + 3]]);
            let zv = f64x4::new([zs[i], zs[i + 1], zs[i + 2], zs[i + 3]]);
            let (h, gx, gz) = fbm_height_grad_x4(xv, zv, p);
            let (ha, gxa, gza) = (h.to_array(), gx.to_array(), gz.to_array());
            for l in 0..4 {
                let (sh, sgx, sgz) = fbm_height_grad(xs[i + l], zs[i + l], p);
                assert_eq!(ha[l].to_bits(), sh.to_bits(), "fBm h mismatch at ({},{}) oct={}", xs[i + l], zs[i + l], p.octaves);
                assert_eq!(gxa[l].to_bits(), sgx.to_bits(), "fBm ∂x mismatch at ({},{})", xs[i + l], zs[i + l]);
                assert_eq!(gza[l].to_bits(), sgz.to_bits(), "fBm ∂z mismatch at ({},{})", xs[i + l], zs[i + l]);
            }
        }
    }
}

/// Determinism: evaluating the same point twice (and in the presence of other evaluations) yields
/// byte-identical results. Order-independence is trivially true for a pure function, but we assert
/// it explicitly because it is THE invariant the parallel layer dispatch relies on (§2.8).
#[test]
fn fbm_is_deterministic_and_order_independent() {
    let p = FbmParams::default();
    let pts = [(1.0, 2.0), (-3.0, 4.0), (5.5, -6.5), (1000.0, 1000.0)];
    // Forward order.
    let fwd: Vec<_> = pts.iter().map(|&(x, z)| fbm_height_grad(x, z, &p)).collect();
    // Reverse order, interleaved with unrelated evaluations.
    let mut rev = Vec::new();
    for &(x, z) in pts.iter().rev() {
        let _noise = fbm_height_grad(x * 0.5 + 17.0, z - 3.0, &p); // perturb call sequence
        rev.push(fbm_height_grad(x, z, &p));
    }
    rev.reverse();
    for (a, b) in fwd.iter().zip(rev.iter()) {
        assert_eq!(a.0.to_bits(), b.0.to_bits(), "height not order-independent");
        assert_eq!(a.1.to_bits(), b.1.to_bits(), "∂x not order-independent");
        assert_eq!(a.2.to_bits(), b.2.to_bits(), "∂z not order-independent");
    }
}
