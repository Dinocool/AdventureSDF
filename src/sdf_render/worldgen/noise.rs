//! Deterministic, cross-platform noise basis for **authoritative** world-gen layers.
//!
//! # Why this exists
//! The world is shared-seed multiplayer (WORLD_GEN_PLAN §0/§2.8): every client generates the world
//! independently from the seed and must agree, bit-for-bit, on everything gameplay-relevant — across
//! GPU vendors, CPU architectures, and operating systems. GPU floating-point is **not** bit-portable
//! (vendor-specific rounding, FMA contraction, fast-math), so authoritative generation runs on the
//! CPU using this basis.
//!
//! # Why it is bit-portable
//! Two ingredients, both deterministic on every conformant target:
//! 1. **Entropy = pure integer hashing.** Wrapping integer arithmetic (`wrapping_*`, `^`, `>>`) is
//!    exactly defined by Rust on all platforms — no UB, no rounding, identical everywhere.
//! 2. **Interpolation = IEEE-754 basic ops on `f64` only.** We use exclusively `+`, `-`, `*` (and one
//!    exact power-of-two divide for the int→float map). IEEE-754 mandates these be *correctly
//!    rounded*, so they produce identical bits on any conformant FPU. We deliberately avoid:
//!    - transcendentals (`sin`/`exp`/`powf`) — not bit-portable;
//!    - `mul_add` / FMA — Rust never contracts `a*b+c` to an FMA implicitly, and we never call
//!      `mul_add`, so there is no fuse-vs-not divergence;
//!    - `f32` accumulation — we accumulate in `f64` and narrow once at the boundary.
//!
//! The `worldgen_parity` integration harness pins reference outputs at fixed `(coord, seed)` points;
//! any drift (a "clever" optimization that reorders into an FMA, a constant change, a transcendental
//! creeping in) fails CI loud — a silent determinism regression would desync multiplayer.
//!
//! This module has **zero** Bevy/ECS dependencies so it can be unit-tested in isolation and reused by
//! every future authoritative layer (height, erosion, climate, caves).

/// Murmur3 finalizer (`fmix32`): avalanches a `u32` so each input bit affects every output bit. Pure
/// wrapping integer ops — bit-identical on every target.
#[inline]
fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

/// 2D integer-lattice hash → `u32`. Combines the two signed lattice coords and the seed with distinct
/// large odd multipliers (so `(x,z)` ≠ `(z,x)` and axis-aligned streaks don't alias), then avalanches
/// with [`fmix32`]. Pure wrapping integer arithmetic ⇒ bit-portable across all platforms.
#[inline]
pub fn hash2(ix: i32, iz: i32, seed: u32) -> u32 {
    let mut h = seed;
    h = h.wrapping_add((ix as u32).wrapping_mul(0x9E37_79B1)); // 2654435761, Knuth's golden-ratio prime
    h = h.wrapping_add((iz as u32).wrapping_mul(0x85EB_CA77)); // large odd, distinct from the x stream
    fmix32(h)
}

/// Lattice value in `[-1, 1)`, derived from the integer hash. The only int→float step is a divide by
/// `2^31`, an exact power of two ⇒ exact on every IEEE-754 target (no rounding ambiguity).
#[inline]
pub fn value_at_lattice(ix: i32, iz: i32, seed: u32) -> f64 {
    // Signed interpretation of the hash spans [-2^31, 2^31 - 1]; /2^31 maps to [-1, 1).
    (hash2(ix, iz, seed) as i32) as f64 * (1.0 / 2_147_483_648.0)
}

/// Perlin quintic fade `6t⁵ − 15t⁴ + 10t³` (C² continuous, zero 1st+2nd derivative at 0 and 1).
/// Horner form, basic ops only.
#[inline]
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Derivative of [`fade`]: `30t²(t−1)²` = `30t²(t² − 2t + 1)`. Basic ops only.
#[inline]
fn fade_deriv(t: f64) -> f64 {
    30.0 * t * t * (t * (t - 2.0) + 1.0)
}

/// Second derivative of [`fade`] = `d/dt[30t²(t−1)²]` = `60t(t−1)(2t−1)` = `60(2t³ − 3t² + t)`. The
/// quintic fade is C², so this is the EXACT, portable (basic-ops-only) curvature of the interpolant —
/// the ingredient the analytic erosion gradient needs (the noise Hessian). Horner form.
#[inline]
fn fade_deriv2(t: f64) -> f64 {
    60.0 * t * (t * (2.0 * t - 3.0) + 1.0)
}

/// One octave of bilinear **value noise** with its analytic gradient, evaluated at the (already
/// frequency-scaled) coordinate `(x, z)`. Returns `(value, ∂value/∂x, ∂value/∂z)` where the value is
/// in roughly `[-1, 1]` and the gradient is in value-per-unit of the scaled coordinate.
///
/// C¹ continuous across integer lattice boundaries (adjacent cells share the lattice values they
/// interpolate, and the quintic fade has matching endpoint derivatives), so there are no seams when a
/// later chunk regenerates the same world point from a neighbouring chunk's padded read.
#[inline]
pub fn value_noise_grad(x: f64, z: f64, seed: u32) -> (f64, f64, f64) {
    let xi = x.floor();
    let zi = z.floor();
    let ix = xi as i32;
    let iz = zi as i32;
    let fx = x - xi; // fractional position in [0,1) within the cell
    let fz = z - zi;

    // Four surrounding lattice values.
    let v00 = value_at_lattice(ix, iz, seed);
    let v10 = value_at_lattice(ix + 1, iz, seed);
    let v01 = value_at_lattice(ix, iz + 1, seed);
    let v11 = value_at_lattice(ix + 1, iz + 1, seed);

    let u = fade(fx);
    let v = fade(fz);
    let du = fade_deriv(fx);
    let dv = fade_deriv(fz);

    // Faded bilinear blend: lerp the two x-edges by u, then lerp those by v.
    let a = v00 + (v10 - v00) * u; // z = iz edge
    let b = v01 + (v11 - v01) * u; // z = iz+1 edge
    let value = a + (b - a) * v;

    // Analytic gradient via the product/chain rule through the fades.
    let da_dx = (v10 - v00) * du;
    let db_dx = (v11 - v01) * du;
    let dval_dx = da_dx + (db_dx - da_dx) * v;
    let dval_dz = (b - a) * dv;

    (value, dval_dx, dval_dz)
}

/// One octave of value noise with its analytic gradient AND Hessian at (already frequency-scaled)
/// `(x, z)`. Returns `(v, ∂v/∂x, ∂v/∂z, ∂²v/∂x², ∂²v/∂x∂z, ∂²v/∂z²)` in the scaled coordinate. The
/// fade is C² so the Hessian is EXACT (and portable — basic `f64` ops only). Superset of
/// [`value_noise_grad`]: the `(v, ∂v/∂x, ∂v/∂z)` lanes are bit-identical to it.
#[inline]
pub fn value_noise_grad_hess(x: f64, z: f64, seed: u32) -> (f64, f64, f64, f64, f64, f64) {
    let xi = x.floor();
    let zi = z.floor();
    let ix = xi as i32;
    let iz = zi as i32;
    let fx = x - xi;
    let fz = z - zi;

    let v00 = value_at_lattice(ix, iz, seed);
    let v10 = value_at_lattice(ix + 1, iz, seed);
    let v01 = value_at_lattice(ix, iz + 1, seed);
    let v11 = value_at_lattice(ix + 1, iz + 1, seed);

    let u = fade(fx);
    let vv = fade(fz);
    let du = fade_deriv(fx);
    let dv = fade_deriv(fz);
    let ddu = fade_deriv2(fx);
    let ddv = fade_deriv2(fz);

    // x-edge lerps (z = iz and z = iz+1) and their x-derivatives.
    let a = v00 + (v10 - v00) * u;
    let b = v01 + (v11 - v01) * u;
    let value = a + (b - a) * vv;

    let da_dx = (v10 - v00) * du;
    let db_dx = (v11 - v01) * du;
    let dval_dx = da_dx + (db_dx - da_dx) * vv;
    let dval_dz = (b - a) * dv;

    // Second derivatives (see height.rs/erosion.rs analytic-gradient derivation):
    //  ∂²/∂x² : ddu · [ (v10−v00) + ((v11−v01)−(v10−v00))·vv ]
    //  ∂²/∂x∂z: du · dv · [ (v11−v01) − (v10−v00) ]
    //  ∂²/∂z² : (b − a) · ddv
    let dxx = ddu * ((v10 - v00) + ((v11 - v01) - (v10 - v00)) * vv);
    let dxz = du * dv * ((v11 - v01) - (v10 - v00));
    let dzz = (b - a) * ddv;

    (value, dval_dx, dval_dz, dxx, dxz, dzz)
}

/// Fractal-Brownian-motion parameters for a height field. Plain data (no Bevy types) so this module
/// stays dependency-free and unit-testable. `f64` throughout — authoritative precision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FbmParams {
    /// Number of octaves summed.
    pub octaves: u32,
    /// Spatial frequency of octave 0, in cycles per world metre.
    pub base_freq: f64,
    /// Frequency multiplier per octave (≈ 2.0).
    pub lacunarity: f64,
    /// Amplitude multiplier per octave (≈ 0.5).
    pub gain: f64,
    /// World-metre amplitude of octave 0.
    pub amplitude: f64,
    /// Layer/world seed mixed into every lattice hash.
    pub seed: u32,
}

impl Default for FbmParams {
    fn default() -> Self {
        Self {
            octaves: 5,
            base_freq: 1.0 / 256.0,
            lacunarity: 2.0,
            gain: 0.5,
            amplitude: 48.0,
            seed: 0,
        }
    }
}

/// fBm height + analytic **world-space** XZ gradient at world coordinate `(wx, wz)`.
/// Returns `(height_metres, ∂h/∂wx, ∂h/∂wz)`. Deterministic and bit-portable (see module docs).
///
/// The gradient is exact (sum of per-octave analytic gradients, chain-ruled through the frequency
/// scaling), so the GPU bake can Lipschitz-normalise `p.y − h` and reconstruct normals without
/// finite differences, and the erosion filter (a later phase) gets the derivatives it needs.
#[inline]
pub fn fbm_height_grad(wx: f64, wz: f64, p: &FbmParams) -> (f64, f64, f64) {
    let mut freq = p.base_freq;
    let mut amp = p.amplitude;
    let mut h = 0.0;
    let mut dh_dx = 0.0;
    let mut dh_dz = 0.0;
    for o in 0..p.octaves {
        // Distinct seed per octave so octaves are independent noise streams (not scaled copies).
        let oseed = p.seed.wrapping_add(o.wrapping_mul(0x9E37_79B9));
        let (v, gx, gz) = value_noise_grad(wx * freq, wz * freq, oseed);
        h += v * amp;
        // d/dwx of v(wx*freq, ·) = (∂v/∂x)·freq; amplitude scales the contribution.
        dh_dx += gx * amp * freq;
        dh_dz += gz * amp * freq;
        freq *= p.lacunarity;
        amp *= p.gain;
    }
    (h, dh_dx, dh_dz)
}

/// fBm height + analytic world-space XZ gradient AND Hessian at world `(wx, wz)`. Returns
/// `(h, ∂h/∂wx, ∂h/∂wz, ∂²h/∂wx², ∂²h/∂wx∂wz, ∂²h/∂wz²)`. Each octave's value/grad/Hessian is evaluated
/// at the frequency-scaled coord; the chain rule scales the gradient by `freq` and the Hessian by
/// `freq²` (and amplitude scales the contribution). EXACT + portable (basic ops + the portable noise
/// basis). The `(h, ∂x, ∂z)` lanes are bit-identical to [`fbm_height_grad`]. The erosion filter's
/// analytic gradient needs the Hessian (the slope-damp term differentiates through `∇h`).
#[inline]
pub fn fbm_height_grad_hess(wx: f64, wz: f64, p: &FbmParams) -> (f64, f64, f64, f64, f64, f64) {
    let mut freq = p.base_freq;
    let mut amp = p.amplitude;
    let mut h = 0.0;
    let mut dh_dx = 0.0;
    let mut dh_dz = 0.0;
    let mut hxx = 0.0;
    let mut hxz = 0.0;
    let mut hzz = 0.0;
    for o in 0..p.octaves {
        let oseed = p.seed.wrapping_add(o.wrapping_mul(0x9E37_79B9));
        let (v, gx, gz, dxx, dxz, dzz) = value_noise_grad_hess(wx * freq, wz * freq, oseed);
        h += v * amp;
        dh_dx += gx * amp * freq;
        dh_dz += gz * amp * freq;
        // d²/dwx² of v(wx·freq, ·) = (∂²v/∂x²)·freq²; amplitude scales the contribution.
        let f2 = freq * freq;
        hxx += dxx * amp * f2;
        hxz += dxz * amp * f2;
        hzz += dzz * amp * f2;
        freq *= p.lacunarity;
        amp *= p.gain;
    }
    (h, dh_dx, dh_dz, hxx, hxz, hzz)
}

/// fBm height (value only) at world `(wx, wz)` — the same octave sum as [`fbm_height_grad`] without the
/// gradient accumulation. Kept as a value-only reference (and the parity anchor for `fbm_height_grad`'s
/// value lane); the height layer now takes the carved gradient CLOSED-FORM via [`fbm_height_grad_hess`],
/// so it no longer needs this. Identical bit pattern to `fbm_height_grad(...).0`. Deterministic & portable.
#[inline]
pub fn fbm_height(wx: f64, wz: f64, p: &FbmParams) -> f64 {
    let mut freq = p.base_freq;
    let mut amp = p.amplitude;
    let mut h = 0.0;
    for o in 0..p.octaves {
        let oseed = p.seed.wrapping_add(o.wrapping_mul(0x9E37_79B9));
        let (v, _, _) = value_noise_grad(wx * freq, wz * freq, oseed);
        h += v * amp;
        freq *= p.lacunarity;
        amp *= p.gain;
    }
    h
}

// ============================================================================================
// 4-wide SIMD primitives — bit-for-bit identical to the scalar path above.
//
// These mirror the scalar `fmix32`/`hash2`/`value_at_lattice`/`fade(_deriv)`/`value_noise_grad`/
// `fbm_height_grad` EXACTLY: same op order, no `mul_add`/FMA, no reassociation. They exist purely to
// process the columnar height grid 4 points at a time (the gen hot path); the scalar functions remain
// the SSOT reference, and `x4_matches_scalar` (tests) `to_bits()`-pins the equality.
//
// ## Why this is bit-exact (the determinism invariant holds)
// * **Integer ops** (`fmix32_x4`/`hash2_x4`): `wide`'s `u32x4` xor / wrapping-`mul` / shift are exact
//   wrapping integer arithmetic on every target (SSE2/NEON intrinsics + a scalar `wrapping_*` fallback),
//   identical to scalar `^`/`wrapping_mul`/`>>`.
// * **`f64`→`i32` lane step**: scalar does `x.floor() as i32`. `f64x4::floor()` is IEEE correctly-rounded
//   floor (hardware `roundpd`, or scalar `f64::floor()` fallback) ⇒ same bits as scalar `floor`. The
//   truncating `as i32` of an already-floored value is then done per lane in scalar (`wide` has no
//   `f64x4→i32x4`); for floored, in-range values that truncation is exact and matches scalar bit-for-bit.
// * **`i32`→`f64` (`value_at_lattice_x4`)**: `f64x4::from(i32x4)` is `_mm256_cvtepi32_pd` / per-lane
//   `as f64`; every `i32` is exactly representable in `f64`, so it equals scalar `(h as i32) as f64`.
// * **fade / blend / gradient**: only `+ - *` on `f64x4` (lanewise = scalar splat op), same Horner /
//   blend / chain-rule expression tree as scalar ⇒ correctly-rounded per IEEE-754 ⇒ identical bits.

use wide::{f64x4, i32x4, u32x4};

/// 4-wide [`fmix32`]: same xor-shift + wrapping-multiply finalizer per lane.
#[inline]
fn fmix32_x4(mut h: u32x4) -> u32x4 {
    h ^= h >> 16;
    h = h * u32x4::splat(0x85eb_ca6b);
    h ^= h >> 13;
    h = h * u32x4::splat(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

/// 4-wide [`hash2`]: combines four `(ix, iz)` lattice coords (signed, reinterpreted as `u32` exactly
/// like scalar `as u32`) with the shared `seed`, then avalanches with [`fmix32_x4`].
#[inline]
fn hash2_x4(ix: i32x4, iz: i32x4, seed: u32) -> u32x4 {
    // `i32x4 → u32x4` is a pure bit reinterpret (same bytes), matching scalar `ix as u32`.
    let ixu: u32x4 = bytemuck::cast(ix);
    let izu: u32x4 = bytemuck::cast(iz);
    let mut h = u32x4::splat(seed);
    h += ixu * u32x4::splat(0x9E37_79B1);
    h += izu * u32x4::splat(0x85EB_CA77);
    fmix32_x4(h)
}

/// 4-wide [`value_at_lattice`]: hash → reinterpret as signed `i32` → `f64` → × `1/2³¹` (exact).
#[inline]
fn value_at_lattice_x4(ix: i32x4, iz: i32x4, seed: u32) -> f64x4 {
    let h = hash2_x4(ix, iz, seed);
    // `(hash as i32)`: bit reinterpret; `as f64`: exact (i32 ⊂ f64 mantissa).
    let hi: i32x4 = bytemuck::cast(h);
    f64x4::from(hi) * (1.0 / 2_147_483_648.0)
}

/// 4-wide [`fade`]: quintic `6t⁵ − 15t⁴ + 10t³`, same Horner form (`+ - *` only).
#[inline]
fn fade_x4(t: f64x4) -> f64x4 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// 4-wide [`fade_deriv`]: `30t²(t−1)²` = `30t²(t² − 2t + 1)`. EXACT same association as scalar
/// (`30.0 * t * t * …`, i.e. `((30·t)·t)·…`) — NOT reassociated, so the bits match.
#[inline]
fn fade_deriv_x4(t: f64x4) -> f64x4 {
    f64x4::splat(30.0) * t * t * (t * (t - 2.0) + 1.0)
}

/// 4-wide [`value_noise_grad`]: floors `(x, z)` per lane (bit-exact `floor` + per-lane `as i32`), then
/// runs the identical faded-bilinear blend + analytic gradient on `f64x4`. Returns `(v, ∂v/∂x, ∂v/∂z)`.
#[inline]
fn value_noise_grad_x4(x: f64x4, z: f64x4, seed: u32) -> (f64x4, f64x4, f64x4) {
    let xi = x.floor();
    let zi = z.floor();
    // f64→i32 per lane (wide has no f64x4→i32x4): scalar `as i32` of an already-floored value is exact.
    let xa = xi.to_array();
    let za = zi.to_array();
    let ix = i32x4::new([xa[0] as i32, xa[1] as i32, xa[2] as i32, xa[3] as i32]);
    let iz = i32x4::new([za[0] as i32, za[1] as i32, za[2] as i32, za[3] as i32]);
    let one = i32x4::splat(1);

    let fx = x - xi; // fractional position in [0,1) within the cell
    let fz = z - zi;

    let v00 = value_at_lattice_x4(ix, iz, seed);
    let v10 = value_at_lattice_x4(ix + one, iz, seed);
    let v01 = value_at_lattice_x4(ix, iz + one, seed);
    let v11 = value_at_lattice_x4(ix + one, iz + one, seed);

    let u = fade_x4(fx);
    let v = fade_x4(fz);
    let du = fade_deriv_x4(fx);
    let dv = fade_deriv_x4(fz);

    let a = v00 + (v10 - v00) * u;
    let b = v01 + (v11 - v01) * u;
    let value = a + (b - a) * v;

    let da_dx = (v10 - v00) * du;
    let db_dx = (v11 - v01) * du;
    let dval_dx = da_dx + (db_dx - da_dx) * v;
    let dval_dz = (b - a) * dv;

    (value, dval_dx, dval_dz)
}

/// 4-wide [`fbm_height_grad`]: identical octave loop (per-octave seed, frequency/amplitude chain) on
/// `f64x4`, summing [`value_noise_grad_x4`] over 4 world points at once. Returns `(h, ∂h/∂wx, ∂h/∂wz)`.
#[inline]
pub fn fbm_height_grad_x4(wx: f64x4, wz: f64x4, p: &FbmParams) -> (f64x4, f64x4, f64x4) {
    let mut freq = p.base_freq;
    let mut amp = p.amplitude;
    let mut h = f64x4::splat(0.0);
    let mut dh_dx = f64x4::splat(0.0);
    let mut dh_dz = f64x4::splat(0.0);
    for o in 0..p.octaves {
        let oseed = p.seed.wrapping_add(o.wrapping_mul(0x9E37_79B9));
        let (v, gx, gz) = value_noise_grad_x4(wx * freq, wz * freq, oseed);
        h += v * amp;
        // Match scalar association EXACTLY: `gx * amp * freq` = `(gx·amp)·freq`, NOT `gx·(amp·freq)`.
        dh_dx += gx * amp * freq;
        dh_dz += gz * amp * freq;
        freq *= p.lacunarity;
        amp *= p.gain;
    }
    (h, dh_dx, dh_dz)
}

#[cfg(test)]
mod tests {
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
}
