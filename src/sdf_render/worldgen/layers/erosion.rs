//! Layer #2: the CPU-authoritative, **bit-portable** analytical EROSION FILTER — a stacked
//! height-filter stage carving branching ridges/gullies into the base height layer.
//!
//! # Why bit-portable (the hard constraint)
//! This is an *authoritative* layer: shared-seed multiplayer requires every client to agree
//! bit-for-bit on the carved surface (WORLD_GEN_PLAN §2.8). So — like [`super::height`] — the entire
//! erosion math is built on the portable basis in [`super::super::noise`]: wrapping integer hashes
//! ([`noise::hash2`]) + IEEE-754 `f64` BASIC ops only (`+ - *`, and the one exact `/2^n` int→float map
//! inside the noise). It calls EXCLUSIVELY [`noise::value_noise_grad`] (C¹, analytic gradient) and
//! [`noise::hash2`]. NO `sin`/`cos`/`powf`/`exp`/transcendentals, NO `mul_add`/FMA, NO `f32`
//! accumulation. `tests/worldgen_parity.rs` pins the carved output and fails CI on any drift.
//!
//! # The filter (portable iq-style ridged erosion)
//! A stack of *ridged*, *slope-damped* value-noise octaves accumulated into a normalized `detail`
//! field, then subtracted (carved) from the base height. Per octave: take `value_noise_grad`, fold to
//! a ridge `r = 1 - |v|`, sharpen it, damp the contribution where the *running* slope is steep (so
//! gullies cut down slopes, not flats/peaks), and accumulate the noise's own gradient so later octaves
//! see the warped slope (the iq "derivative feedback" that makes erosion look directional). A
//! peak/valley fade preserves the base height's sharp extremes. Output = `h_base - strength·fade·detail`.
//!
//! # Gradient
//! [`erode_with_grad`] returns the carved height AND its CLOSED-FORM XZ gradient (one eval/node). The
//! erosion derivative needs the noise *Hessian* — the slope-damp `1/(1+gully·|∇h|²)` differentiates
//! through the running slope's derivative `d(∇h)` — so it consumes the base Hessian from
//! [`super::super::noise::value_noise_grad_hess`] / [`super::super::noise::fbm_height_grad_hess`] and
//! accumulates it through the octaves alongside the gradient. This REPLACED the old 5-tap central
//! difference in [`super::height::HeightLayer::sample_world`] (5× the eval cost + a smoothed FD normal).
//! [`erode_height`] (value-only) is kept as the parity reference its value lane is pinned to.

use bevy::prelude::*;

use super::super::noise::{value_noise_grad, value_noise_grad_hess};

/// Editor-tweakable erosion-layer parameters (reflected resource, mirrors [`super::height::HeightParams`]).
/// A change dirties the layer → full regen (handled by the manager, same path as a height-param edit).
///
/// `enabled: true` by default — the user wants interesting (carved) terrain out of the box. With
/// `enabled = false`, [`erode_height`] is the EXACT identity (returns `h_base` unchanged).
#[derive(Resource, Reflect, Clone, Copy, Debug, PartialEq)]
#[reflect(Resource)]
pub struct ErosionParams {
    /// Master switch. `false` ⇒ [`erode_height`] is the exact identity (no carving, no cost).
    pub enabled: bool,
    /// Carve depth in world metres — how deep the gullies cut into the base height.
    pub strength: f32,
    /// Number of ridged octaves stacked.
    pub octaves: u32,
    /// Coarsest erosion feature size, world metres (octave-0 wavelength). Finest feature ≈
    /// `base_cell_size / lacunarity^(octaves-1)`.
    pub base_cell_size: f32,
    /// Frequency multiplier per octave (≈ 2.0).
    pub lacunarity: f32,
    /// Amplitude multiplier per octave (≈ 0.5).
    pub gain: f32,
    /// Slope coupling: scales both the slope-damping (gullies prefer slopes) and the ridge sharpening.
    /// Higher = tighter, more slope-selective gullies.
    pub gully_weight: f32,
    /// Preserve the base height's sharp extremes: fades erosion out toward the height field's peaks
    /// and valleys (0 = carve everywhere, 1 = carve only the mid-band).
    pub peak_valley_fade: f32,
    /// Per-layer salt mixed with the world seed so erosion has an independent noise stream from height.
    pub seed_salt: u32,
}

impl Default for ErosionParams {
    fn default() -> Self {
        // Tuned for a visible-but-not-overwhelming carved look on the retuned (tall) height layer.
        Self {
            enabled: true,
            strength: 55.0,
            octaves: 5,
            base_cell_size: 640.0,
            lacunarity: 2.0,
            gain: 0.5,
            gully_weight: 0.6,
            peak_valley_fade: 0.6,
            seed_salt: 0x00E2_0510,
        }
    }
}

/// Fold the world seed with the erosion salt into a stable `u32` noise seed — MIRRORS
/// [`super::height::HeightLayer::fbm_params`]'s seed fold (mix both halves of the 64-bit world seed
/// with the layer salt). Pure / deterministic / bit-portable.
#[inline]
pub fn erosion_seed(world_seed: u64, p: &ErosionParams) -> u32 {
    (world_seed as u32) ^ ((world_seed >> 32) as u32) ^ p.seed_salt
}

/// Per-octave salt so octaves are independent noise streams (not frequency-scaled copies of one).
/// Same wrapping-multiply idiom as `fbm_height_grad`'s octave seed.
#[inline]
fn octave_salt(o: u32) -> u32 {
    o.wrapping_mul(0x9E37_79B9)
}

/// Portable polynomial "bump": ~1 across the mid-range of the normalized height and →0 toward the
/// extremes, so [`erode_height`]'s peak/valley fade preserves sharp peaks and valley floors. Basic ops
/// only. `t` is the base height normalized to roughly `[-1, 1]` (clamped); returns `(1 - t²)` clamped to
/// `[0, 1]` — a smooth, transcendental-free hump centred at mid-altitude.
#[inline]
fn smooth_bump(t: f64) -> f64 {
    let tc = t.clamp(-1.0, 1.0);
    let b = 1.0 - tc * tc;
    if b < 0.0 { 0.0 } else { b }
}

/// Derivative of [`smooth_bump`] w.r.t. its argument `t`: `d/dt[1 − clamp(t)²]`. Inside `(−1, 1)` this is
/// `−2t`; outside (where `clamp` pins `t`) it is `0` (the bump is flat). The single non-smooth points
/// `t = ±1` are measure-zero for the analytic gradient (the FD it replaces equally couldn't see them).
/// Basic ops only.
#[inline]
fn smooth_bump_deriv(t: f64) -> f64 {
    if t > -1.0 && t < 1.0 { -2.0 * t } else { 0.0 }
}

/// Carve the base height with the portable ridged-erosion filter. Returns the eroded SCALAR height in
/// world metres (its gradient comes from a central difference in `sample_world` — see module docs).
///
/// - `h_base`        — base surface height (incl. sea level + any ridge fold), metres.
/// - `gx_base/gz_base` — base surface XZ gradient (metres per metre), seeds the slope-damp feedback.
/// - `wx/wz`         — world XZ, metres.
/// - `world_seed`    — the world seed (folded with `p.seed_salt` via [`erosion_seed`]).
///
/// `enabled = false` ⇒ EXACT identity (`erode_height(h, …) == h`, bit-for-bit). All math is `f64`
/// basic ops + [`value_noise_grad`] / [`noise::hash2`] ⇒ bit-portable (the parity harness guards it).
#[inline]
pub fn erode_height(
    h_base: f64,
    gx_base: f64,
    gz_base: f64,
    wx: f64,
    wz: f64,
    world_seed: u64,
    p: &ErosionParams,
) -> f64 {
    if !p.enabled {
        return h_base; // exact identity — no carving, no rounding.
    }
    let seed = erosion_seed(world_seed, p);
    let base_cell = p.base_cell_size as f64;
    // Guard a degenerate cell size (slider could reach 0) without a branch in the hot loop.
    let inv_cell = if base_cell > 1e-6 { 1.0 / base_cell } else { 1.0 };

    let lacunarity = p.lacunarity as f64;
    let gain = p.gain as f64;
    let gully = p.gully_weight as f64;

    let mut freq = inv_cell;
    let mut amp = 1.0;
    // Running slope accumulates the warped gradient (base + each octave's contribution) so later
    // octaves see the carved slope (iq derivative feedback → directional, flow-like gullies).
    let mut gx = gx_base;
    let mut gz = gz_base;
    let mut detail = 0.0;
    let mut norm = 0.0;

    for o in 0..p.octaves {
        let (v, dvx, dvz) = value_noise_grad(wx * freq, wz * freq, seed ^ octave_salt(o));
        // Ridge fold: 1 - |v| ∈ [0, 1], peaks where the noise crosses zero → branching ridge lines.
        let av = if v < 0.0 { -v } else { v };
        let mut r = 1.0 - av;
        // Sharpen (squaring tightens the ridges); gully_weight scales how aggressively.
        r = r * r * (1.0 + gully);
        // Slope-damp: cut more where the running slope is steep (gullies follow slopes), via the
        // rational 1/(1 + gully·|∇|²) — basic ops, no transcendentals.
        let slope2 = gx * gx + gz * gz;
        let damp = 1.0 / (1.0 + gully * slope2);
        detail += amp * r * damp;
        norm += amp;
        // Feed this octave's gradient (scaled by freq·amp, the chain rule of v(wx·freq)) into the
        // running slope so the next octave's damp/ridge sees the warped surface.
        gx += dvx * freq * amp;
        gz += dvz * freq * amp;
        freq *= lacunarity;
        amp *= gain;
    }

    let detail = detail / norm.max(1e-6); // normalize to ~[0, 1] regardless of octave/gain choice.

    // Preserve sharp extremes of the BASE height: fade carving toward peaks/valleys. Normalize the
    // base height by the strength scale so the bump is roughly centred on the surface's mid-band.
    let h_norm = h_base / (p.strength as f64 + 1.0);
    let fade = 1.0 - p.peak_valley_fade as f64 * smooth_bump(h_norm);

    // Carve down: subtract the gully detail. (Ridged detail ≥ 0, so this only lowers — incising
    // valleys without inflating peaks, which reads as erosion rather than added relief.)
    h_base - p.strength as f64 * fade * detail
}

/// Carve the base height AND return the carved surface's ANALYTIC XZ gradient — the closed-form
/// replacement for the central-difference taps the height layer used to take (5× the eval cost + a
/// smoothed FD normal). Returns `(eroded_height, ∂H/∂wx, ∂H/∂wz)`.
///
/// - `h_base`            — base surface height (incl. sea level + ridge fold), metres.
/// - `gx_base/gz_base`   — base surface XZ GRADIENT (∂h_base/∂wx, ∂h_base/∂wz). Seeds both the slope-damp
///   feedback (as before) AND the carved gradient (the `h_base −` term + the fade's chain rule).
/// - `hxx/hxz/hzz`       — base surface HESSIAN (∂²h_base). Seeds the running-slope DERIVATIVE the
///   slope-damp term differentiates through (its whole reason for existing — see the derivation inline).
/// - `wx/wz`, `world_seed`, `p` — as [`erode_height`].
///
/// The VALUE lane is bit-identical to [`erode_height`] (same op order), so this can replace the value
/// path with no parity drift. `enabled = false` ⇒ EXACT identity: `(h_base, gx_base, gz_base)`.
///
/// DERIVATION (matches `erode_height` term by term):
/// `H = h_base − strength·fade·detail`, so `∂H = gx_base − strength·(∂fade·detail + fade·∂detail)`.
/// - `fade = 1 − pvf·bump(h_norm)`, `h_norm = h_base/(strength+1)` ⇒ `∂fade = −pvf·bump'(h_norm)·∂h_norm`,
///   `∂h_norm = gx_base/(strength+1)`.
/// - `detail = (Σ amp·r·damp)/norm`. Per octave `o`:
///   - `r = (1−|v|)²·(1+gully)`, `v = value(wx·freq, ·)` ⇒ `∂r = (1+gully)·2·(1−|v|)·(−sign(v)·∂v)`,
///     `∂v/∂wx = dvx·freq` (chain rule).
///   - `damp = 1/(1+gully·slope²)`, `slope² = gx²+gz²` (the RUNNING slope) ⇒
///     `∂damp = −gully·∂(slope²)·damp²`, `∂(slope²)/∂wx = 2gx·∂gx/∂wx + 2gz·∂gz/∂wx`.
///     The running slope's derivative `∂gx/∂wx = Gxx` accumulates the noise HESSIAN through the octaves
///     (seeded by the base Hessian) exactly as `gx` accumulates the gradient — THIS is why the Hessian
///     is required.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn erode_with_grad(
    h_base: f64,
    gx_base: f64,
    gz_base: f64,
    hxx: f64,
    hxz: f64,
    hzz: f64,
    wx: f64,
    wz: f64,
    world_seed: u64,
    p: &ErosionParams,
) -> (f64, f64, f64) {
    if !p.enabled {
        return (h_base, gx_base, gz_base); // exact identity — no carving, no rounding.
    }
    let seed = erosion_seed(world_seed, p);
    let base_cell = p.base_cell_size as f64;
    let inv_cell = if base_cell > 1e-6 { 1.0 / base_cell } else { 1.0 };

    let lacunarity = p.lacunarity as f64;
    let gain = p.gain as f64;
    let gully = p.gully_weight as f64;

    let mut freq = inv_cell;
    let mut amp = 1.0;
    // Running slope (gradient) AND its derivative (the accumulated Hessian), seeded by the base values.
    let mut gx = gx_base;
    let mut gz = gz_base;
    let mut g_xx = hxx; // ∂gx/∂wx
    let mut g_xz = hxz; // ∂gx/∂wz = ∂gz/∂wx (mixed)
    let mut g_zz = hzz; // ∂gz/∂wz
    let mut detail = 0.0;
    let mut ddetail_dx = 0.0;
    let mut ddetail_dz = 0.0;
    let mut norm = 0.0;

    for o in 0..p.octaves {
        let (v, dvx, dvz, dxx, dxz, dzz) = value_noise_grad_hess(wx * freq, wz * freq, seed ^ octave_salt(o));
        // --- value path (bit-identical to erode_height) ---
        let av = if v < 0.0 { -v } else { v };
        let mut r = 1.0 - av;
        r = r * r * (1.0 + gully);
        let slope2 = gx * gx + gz * gz;
        let damp = 1.0 / (1.0 + gully * slope2);
        detail += amp * r * damp;
        norm += amp;

        // --- gradient path ---
        // ∂v/∂w (world): chain rule from the scaled coord.
        let dv_dx = dvx * freq;
        let dv_dz = dvz * freq;
        // sign(v) (0 at v==0 — measure-zero, FD couldn't see it either).
        let sgn = if v < 0.0 { -1.0 } else { 1.0 };
        // r = (1+gully)·(1-|v|)² ⇒ ∂r = (1+gully)·2·(1-|v|)·(-sgn·∂v).
        let one_minus_av = 1.0 - av;
        let coef = (1.0 + gully) * 2.0 * one_minus_av;
        let dr_dx = coef * (-sgn * dv_dx);
        let dr_dz = coef * (-sgn * dv_dz);
        // ∂(slope²) = 2gx·∂gx + 2gz·∂gz, using the RUNNING slope's derivative (accumulated Hessian).
        let dslope2_dx = 2.0 * gx * g_xx + 2.0 * gz * g_xz;
        let dslope2_dz = 2.0 * gx * g_xz + 2.0 * gz * g_zz;
        // ∂damp = -gully·∂(slope²)·damp².
        let damp2 = damp * damp;
        let ddamp_dx = -gully * dslope2_dx * damp2;
        let ddamp_dz = -gully * dslope2_dz * damp2;
        // ∂(amp·r·damp) = amp·(∂r·damp + r·∂damp).
        ddetail_dx += amp * (dr_dx * damp + r * ddamp_dx);
        ddetail_dz += amp * (dr_dz * damp + r * ddamp_dz);

        // Feed this octave's gradient into the running slope, and its Hessian into the running slope's
        // derivative — MIRRORS the value path's `gx += dvx*freq*amp` (so the next octave's damp/∂damp see
        // the warped surface). ∂(dvx·freq·amp)/∂wx = dxx·freq²·amp, etc.
        let f2 = freq * freq;
        gx += dvx * freq * amp;
        gz += dvz * freq * amp;
        g_xx += dxx * f2 * amp;
        g_xz += dxz * f2 * amp;
        g_zz += dzz * f2 * amp;

        freq *= lacunarity;
        amp *= gain;
    }

    // VALUE-LANE PARITY: use the EXACT same ops as `erode_height` (`detail / norm.max`, `h_base / (s+1)`)
    // so `h` is bit-identical. The gradient lanes apply the SAME `1/norm` and `1/(s+1)` factors (as
    // multiplies — the gradient isn't parity-pinned to the FD it replaces, only ≈, so a reciprocal there
    // is fine).
    let norm_clamped = norm.max(1e-6);
    let detail = detail / norm_clamped;
    let inv_norm = 1.0 / norm_clamped;
    let ddetail_dx = ddetail_dx * inv_norm;
    let ddetail_dz = ddetail_dz * inv_norm;

    let str1 = p.strength as f64 + 1.0;
    let h_norm = h_base / str1;
    let pvf = p.peak_valley_fade as f64;
    let bump = smooth_bump(h_norm);
    let fade = 1.0 - pvf * bump;
    // ∂fade = -pvf·bump'(h_norm)·∂h_norm, ∂h_norm = ∂h_base/(s+1).
    let bump_d = smooth_bump_deriv(h_norm);
    let dfade_dx = -pvf * bump_d * gx_base / str1;
    let dfade_dz = -pvf * bump_d * gz_base / str1;

    let strength = p.strength as f64;
    // H = h_base - strength·fade·detail ⇒ ∂H = ∂h_base - strength·(∂fade·detail + fade·∂detail).
    let h = h_base - strength * fade * detail;
    let dh_dx = gx_base - strength * (dfade_dx * detail + fade * ddetail_dx);
    let dh_dz = gz_base - strength * (dfade_dz * detail + fade * ddetail_dz);
    (h, dh_dx, dh_dz)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two evaluations of the same point are bit-identical (the parallel-dispatch invariant).
    #[test]
    fn erode_is_deterministic() {
        let p = ErosionParams::default();
        for &(wx, wz) in &[(12.0, -7.0), (1000.5, -500.25), (-130.0, 88.0)] {
            let a = erode_height(50.0, 0.3, -0.2, wx, wz, 42, &p);
            let b = erode_height(50.0, 0.3, -0.2, wx, wz, 42, &p);
            assert_eq!(a.to_bits(), b.to_bits(), "erosion not deterministic at ({wx},{wz})");
        }
    }

    /// `enabled = false` is the EXACT identity — bit-for-bit, no carving.
    #[test]
    fn disabled_is_exact_identity() {
        let p = ErosionParams { enabled: false, ..Default::default() };
        for &h in &[0.0, 12.5, -33.0, 250.0, -1.0e-9] {
            let out = erode_height(h, 0.7, -0.4, 123.0, -456.0, 7, &p);
            assert_eq!(out.to_bits(), h.to_bits(), "disabled erosion must be exact identity for h={h}");
        }
    }

    /// `erode_with_grad`'s VALUE lane is BIT-IDENTICAL to `erode_height` (so swapping the height layer onto
    /// the analytic path doesn't drift the carved surface VALUE — only the gradient is new). Same op order.
    #[test]
    fn with_grad_value_matches_erode_height_bitwise() {
        let p = ErosionParams::default();
        for &(h, gx, gz, wx, wz, s) in &[
            (50.0, 0.3, -0.2, 12.0, -7.0, 42u64),
            (40.0, 0.8, 0.5, 321.0, -123.0, 9),
            (200.0, -1.2, 0.9, 1000.5, -500.25, 7),
            (-30.0, 0.1, -0.05, -130.0, 88.0, 1),
        ] {
            let v_only = erode_height(h, gx, gz, wx, wz, s, &p);
            // Seed the Hessian with arbitrary finite values: it must NOT affect the value lane.
            let (v, _, _) = erode_with_grad(h, gx, gz, 0.01, -0.02, 0.03, wx, wz, s, &p);
            assert_eq!(v.to_bits(), v_only.to_bits(), "value lane drifted at ({wx},{wz})");
        }
    }

    /// `enabled = false` ⇒ `erode_with_grad` is the EXACT identity in ALL three lanes.
    #[test]
    fn with_grad_disabled_is_exact_identity() {
        let p = ErosionParams { enabled: false, ..Default::default() };
        for &(h, gx, gz) in &[(0.0, 0.7, -0.4), (250.0, -1.0, 0.5), (-33.0, 0.0, 0.0)] {
            let (v, dx, dz) = erode_with_grad(h, gx, gz, 9.0, -9.0, 9.0, 123.0, -456.0, 7, &p);
            assert_eq!(v.to_bits(), h.to_bits());
            assert_eq!(dx.to_bits(), gx.to_bits());
            assert_eq!(dz.to_bits(), gz.to_bits());
        }
    }

    /// `erode_with_grad` is deterministic (bit-identical on recompute) in all three lanes.
    #[test]
    fn with_grad_is_deterministic() {
        let p = ErosionParams::default();
        for &(wx, wz) in &[(12.0, -7.0), (1000.5, -500.25), (-130.0, 88.0)] {
            let a = erode_with_grad(50.0, 0.3, -0.2, 0.01, 0.0, -0.01, wx, wz, 42, &p);
            let b = erode_with_grad(50.0, 0.3, -0.2, 0.01, 0.0, -0.01, wx, wz, 42, &p);
            assert_eq!(a.0.to_bits(), b.0.to_bits());
            assert_eq!(a.1.to_bits(), b.1.to_bits());
            assert_eq!(a.2.to_bits(), b.2.to_bits());
        }
    }

    /// The ANALYTIC eroded gradient matches a central difference of the eroded height — the correctness
    /// guard for the closed-form differentiation of the erosion formula. The FD must use the SAME base
    /// height/gradient/Hessian field, i.e. a consistent quadratic local model of `h_base`, so the FD of
    /// the eroded value is meaningful. We model `h_base(wx,wz)` as the 2nd-order Taylor expansion about
    /// the eval point from `(h, gx, gz, hxx, hxz, hzz)` and feed that to `erode_with_grad` at the offsets.
    #[test]
    fn analytic_gradient_matches_central_difference() {
        let p = ErosionParams::default();
        // A representative base surface sample (height, gradient, Hessian) — sloped, mid-altitude.
        let (h0, gx0, gz0) = (60.0f64, 0.6f64, -0.4f64);
        let (hxx, hxz, hzz) = (0.002f64, -0.001f64, 0.0015f64);
        for &(wx, wz, seed) in &[(321.0, -123.0, 1u64), (-560.0, 880.0, 1), (1500.5, 700.25, 1)] {
            let (_, dax, daz) = erode_with_grad(h0, gx0, gz0, hxx, hxz, hzz, wx, wz, seed, &p);
            // Locally-consistent base field: h_base(wx+dx, wz+dz) = h0 + g·d + ½ dᵀH d, with gradient
            // g + H·d (so the seed gradient/Hessian handed to erode_with_grad match the FD'd field).
            let e = 0.01f64;
            let base = |dx: f64, dz: f64| -> f64 {
                let h = h0 + gx0 * dx + gz0 * dz + 0.5 * (hxx * dx * dx + 2.0 * hxz * dx * dz + hzz * dz * dz);
                let gx = gx0 + hxx * dx + hxz * dz;
                let gz = gz0 + hxz * dx + hzz * dz;
                erode_with_grad(h, gx, gz, hxx, hxz, hzz, wx + dx, wz + dz, seed, &p).0
            };
            let fd_x = (base(e, 0.0) - base(-e, 0.0)) / (2.0 * e);
            let fd_z = (base(0.0, e) - base(0.0, -e)) / (2.0 * e);
            assert!((dax - fd_x).abs() < 1e-2, "∂x at ({wx},{wz}): analytic {dax} vs FD {fd_x}");
            assert!((daz - fd_z).abs() < 1e-2, "∂z at ({wx},{wz}): analytic {daz} vs FD {fd_z}");
        }
    }

    /// Enabled + sloped erosion actually changes the height (the carve bites).
    #[test]
    fn enabled_changes_height_on_slope() {
        let p = ErosionParams::default();
        let h = 40.0;
        // A sloped, mid-altitude point — where erosion should carve.
        let out = erode_height(h, 0.8, 0.5, 321.0, -123.0, 9, &p);
        assert_ne!(out.to_bits(), h.to_bits(), "enabled erosion must change a sloped mid-altitude height");
        // It carves DOWN (ridged detail ≥ 0 ⇒ subtractive).
        assert!(out <= h + 1e-9, "erosion should only lower the surface, got {out} > {h}");
    }

    /// `smooth_bump` is the expected portable hump: 1 at centre, 0 at the extremes, clamped.
    #[test]
    fn smooth_bump_shape() {
        assert!((smooth_bump(0.0) - 1.0).abs() < 1e-12);
        assert!(smooth_bump(1.0).abs() < 1e-12);
        assert!(smooth_bump(-1.0).abs() < 1e-12);
        assert_eq!(smooth_bump(5.0), 0.0, "clamped beyond the extremes");
        assert_eq!(smooth_bump(-5.0), 0.0);
    }

    /// Bit-portability smoke: the seed fold mixes both halves of the world seed with the salt (so two
    /// world seeds differing only in the high 32 bits give different streams).
    #[test]
    fn seed_fold_mixes_both_halves() {
        let p = ErosionParams::default();
        let lo = erosion_seed(0x0000_0000_DEAD_BEEF, &p);
        let hi = erosion_seed(0xCAFE_0000_DEAD_BEEF, &p);
        assert_ne!(lo, hi, "high 32 bits of the world seed must affect the erosion stream");
        // Determinism of the fold itself.
        assert_eq!(erosion_seed(123, &p), erosion_seed(123, &p));
        // hash2 is the entropy basis erosion's noise stream ultimately derives from (bit-portable).
        assert_eq!(super::super::super::noise::hash2(0, 0, 0), 0);
    }
}
