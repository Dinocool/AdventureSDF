//! `Field` — the value flowing along node-graph edges: a scalar plus its analytic world-XZ gradient,
//! evaluated at one world point. This is a forward-mode **dual number** (`v` + the partials
//! `∂v/∂wx`, `∂v/∂wz`), so a graph built out of these ops auto-differentiates: the final node's
//! `Field` IS `(height, dh_dx, dh_dz)` — exactly what `HeightLayer::sample_world` must return for the
//! terrain normals (see [`super`] / `layers::height`). No finite differences anywhere.
//!
//! **Bit-portable determinism (non-negotiable, like [`super::super::noise`]):** every op is f64 IEEE
//! basic arithmetic in a fixed evaluation order — NO transcendentals, NO `mul_add`/FMA, NO f32
//! accumulation. Shared-seed multiplayer clients must agree bit-for-bit; the `worldgen_parity` harness
//! is the guard. Branch points (`abs`/`min`/`max`/`clamp`/`smoothstep` saturation) are non-smooth on a
//! measure-zero set — handled exactly as the existing code treats `sign(h)` at `h=0`
//! (`layers::height::carved_grad`) and `smooth_bump` clamps (`layers::erosion`).

/// A scalar value at a world point together with its analytic gradient w.r.t. world X/Z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Field {
    /// The value.
    pub v: f64,
    /// ∂v/∂(world x).
    pub dx: f64,
    /// ∂v/∂(world z).
    pub dz: f64,
}

impl Field {
    /// A spatially-constant field (gradient zero).
    #[inline]
    pub const fn constant(v: f64) -> Self {
        Self { v, dx: 0.0, dz: 0.0 }
    }

    /// The world-X coordinate as a field: value `wx`, gradient `(1, 0)`. A graph source.
    #[inline]
    pub const fn world_x(wx: f64) -> Self {
        Self { v: wx, dx: 1.0, dz: 0.0 }
    }

    /// The world-Z coordinate as a field: value `wz`, gradient `(0, 1)`. A graph source.
    #[inline]
    pub const fn world_z(wz: f64) -> Self {
        Self { v: wz, dx: 0.0, dz: 1.0 }
    }

    /// A field from a value + its already-known analytic gradient (e.g. a noise sample).
    #[inline]
    pub const fn new(v: f64, dx: f64, dz: f64) -> Self {
        Self { v, dx, dz }
    }

    #[inline]
    pub fn add(self, b: Self) -> Self {
        Self { v: self.v + b.v, dx: self.dx + b.dx, dz: self.dz + b.dz }
    }

    #[inline]
    pub fn sub(self, b: Self) -> Self {
        Self { v: self.v - b.v, dx: self.dx - b.dx, dz: self.dz - b.dz }
    }

    #[inline]
    pub fn neg(self) -> Self {
        Self { v: -self.v, dx: -self.dx, dz: -self.dz }
    }

    /// Product rule: `(ab)' = a'b + ab'`.
    #[inline]
    pub fn mul(self, b: Self) -> Self {
        Self {
            v: self.v * b.v,
            dx: self.dx * b.v + self.v * b.dx,
            dz: self.dz * b.v + self.v * b.dz,
        }
    }

    /// Multiply by a spatial constant (cheaper than `mul` of a `constant`).
    #[inline]
    pub fn scale(self, k: f64) -> Self {
        Self { v: self.v * k, dx: self.dx * k, dz: self.dz * k }
    }

    /// Add a spatial constant.
    #[inline]
    pub fn offset(self, k: f64) -> Self {
        Self { v: self.v + k, dx: self.dx, dz: self.dz }
    }

    /// `|self|` — gradient flips sign with the value (kink at `v=0`, measure-zero).
    #[inline]
    pub fn abs(self) -> Self {
        if self.v < 0.0 { self.neg() } else { self }
    }

    /// Smaller value, carrying ITS gradient (kink at equality, measure-zero). Ties → `self`.
    #[inline]
    pub fn min(self, b: Self) -> Self {
        if b.v < self.v { b } else { self }
    }

    /// Larger value, carrying ITS gradient (kink at equality, measure-zero). Ties → `self`.
    #[inline]
    pub fn max(self, b: Self) -> Self {
        if b.v > self.v { b } else { self }
    }

    /// Clamp the value to `[lo, hi]` (spatial constants); gradient passes through inside the range,
    /// flat (zero) when saturated. Boundaries are measure-zero kinks.
    #[inline]
    pub fn clamp(self, lo: f64, hi: f64) -> Self {
        if self.v < lo {
            Self::constant(lo)
        } else if self.v > hi {
            Self::constant(hi)
        } else {
            self
        }
    }

    /// Linear interpolation `self + (b - self)·t`, with `t` a field (so `t` may itself vary in space —
    /// the placement/blend op). Full product/sum-rule gradient.
    #[inline]
    pub fn mix(self, b: Self, t: Self) -> Self {
        // d = b - self; result = self + d·t
        let d = b.sub(self);
        self.add(d.mul(t))
    }

    /// Smoothstep of the value across `[edge0, edge1]` (spatial constants): the C¹ Hermite
    /// `s = t²(3 − 2t)`, `t = clamp((v − e0)/(e1 − e0), 0, 1)`. Portable (no transcendentals). The
    /// canonical smooth membership / gate used for biome placement; `s'(t) = 6t(1 − t)`.
    #[inline]
    pub fn smoothstep(self, edge0: f64, edge1: f64) -> Self {
        let inv = 1.0 / (edge1 - edge0);
        // t = (v - e0) * inv, clamped to [0,1]; saturated ends → flat (measure-zero kinks).
        let raw = (self.v - edge0) * inv;
        if raw <= 0.0 {
            return Self::constant(0.0);
        }
        if raw >= 1.0 {
            return Self::constant(1.0);
        }
        let t = raw;
        let s = t * t * (3.0 - 2.0 * t);
        let ds_dt = 6.0 * t * (1.0 - t);
        let ds_dv = ds_dt * inv; // chain rule: ∂t/∂v = inv
        Self { v: s, dx: ds_dv * self.dx, dz: ds_dv * self.dz }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central-difference the world-XZ gradient of `f.v` and assert the analytic `(dx, dz)` matches.
    /// `f` is a closure `(wx, wz) -> Field` built from the ops; this is the autodiff correctness gate
    /// (mirrors the analytic-vs-FD tests in `noise.rs` / `layers::erosion`).
    fn assert_grad_matches_cd(f: impl Fn(f64, f64) -> Field, pts: &[(f64, f64)]) {
        let e = 1e-3;
        for &(wx, wz) in pts {
            let fld = f(wx, wz);
            let cd_x = (f(wx + e, wz).v - f(wx - e, wz).v) / (2.0 * e);
            let cd_z = (f(wx, wz + e).v - f(wx, wz - e).v) / (2.0 * e);
            assert!((fld.dx - cd_x).abs() < 1e-4, "∂x at ({wx},{wz}): analytic {} vs CD {cd_x}", fld.dx);
            assert!((fld.dz - cd_z).abs() < 1e-4, "∂z at ({wx},{wz}): analytic {} vs CD {cd_z}", fld.dz);
        }
    }

    const PTS: &[(f64, f64)] = &[(0.3, -0.7), (1.5, 2.0), (-2.0, 0.4), (5.0, -3.0)];

    #[test]
    fn sources_have_unit_gradients() {
        assert_eq!(Field::world_x(7.0), Field::new(7.0, 1.0, 0.0));
        assert_eq!(Field::world_z(7.0), Field::new(7.0, 0.0, 1.0));
        assert_eq!(Field::constant(3.0), Field::new(3.0, 0.0, 0.0));
    }

    #[test]
    fn add_sub_neg_scale_offset_grad() {
        // f = 2·x·... actually a mix of linear terms in x and z.
        assert_grad_matches_cd(
            |x, z| Field::world_x(x).scale(2.0).add(Field::world_z(z)).offset(5.0).sub(Field::world_x(x).neg()),
            PTS,
        );
    }

    #[test]
    fn mul_product_rule_grad() {
        // f = (x+1)·(z·2) — product of two spatially-varying fields.
        assert_grad_matches_cd(
            |x, z| Field::world_x(x).offset(1.0).mul(Field::world_z(z).scale(2.0)),
            PTS,
        );
    }

    #[test]
    fn mix_grad_with_varying_t() {
        // f = mix(x², z², t) where t = smoothstep of (x) — t varies in space (placement op).
        assert_grad_matches_cd(
            |x, z| {
                let a = Field::world_x(x).mul(Field::world_x(x));
                let b = Field::world_z(z).mul(Field::world_z(z));
                let t = Field::world_x(x).smoothstep(-1.0, 1.0);
                a.mix(b, t)
            },
            PTS,
        );
    }

    #[test]
    fn abs_min_max_clamp_grad() {
        // Avoid the measure-zero kink points in the test set.
        assert_grad_matches_cd(|x, _| Field::world_x(x).offset(0.37).abs(), PTS);
        assert_grad_matches_cd(
            |x, z| Field::world_x(x).min(Field::world_z(z).scale(0.5)),
            PTS,
        );
        assert_grad_matches_cd(
            |x, z| Field::world_x(x).max(Field::world_z(z).scale(0.5)),
            PTS,
        );
        // clamp: interior passes the gradient, saturated is flat.
        let inside = Field::world_x(0.5).clamp(-10.0, 10.0);
        assert_eq!((inside.dx, inside.dz), (1.0, 0.0));
        let sat = Field::world_x(50.0).clamp(-10.0, 10.0);
        assert_eq!((sat.v, sat.dx, sat.dz), (10.0, 0.0, 0.0));
    }

    #[test]
    fn smoothstep_grad_and_saturation() {
        assert_grad_matches_cd(|x, _| Field::world_x(x).scale(0.3).smoothstep(-1.0, 1.0), PTS);
        assert_eq!(Field::world_x(-5.0).smoothstep(-1.0, 1.0), Field::constant(0.0));
        assert_eq!(Field::world_x(5.0).smoothstep(-1.0, 1.0), Field::constant(1.0));
    }

    #[test]
    fn ops_are_deterministic_bitwise() {
        let g = |x: f64, z: f64| {
            Field::world_x(x).offset(1.0).mul(Field::world_z(z).scale(2.0)).mix(
                Field::constant(3.0),
                Field::world_x(x).smoothstep(0.0, 4.0),
            )
        };
        let a = g(1.25, -0.5);
        let b = g(1.25, -0.5);
        assert_eq!(a.v.to_bits(), b.v.to_bits());
        assert_eq!(a.dx.to_bits(), b.dx.to_bits());
        assert_eq!(a.dz.to_bits(), b.dz.to_bits());
    }
}
