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
mod tests;
