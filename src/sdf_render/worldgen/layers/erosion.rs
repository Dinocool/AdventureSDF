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
//! `erode_height` returns only the carved scalar height — its analytic derivative would need the noise
//! *Hessian* (which `value_noise_grad` does not expose, since the ridge fold `1-|v|` and the slope-damp
//! both differentiate through `dv`). So the eroded gradient is taken by CENTRAL DIFFERENCE of the full
//! (fBm + ridge + erosion) height in [`super::height::HeightLayer::sample_world`] — see that function.

use bevy::prelude::*;

use super::super::noise::value_noise_grad;

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
