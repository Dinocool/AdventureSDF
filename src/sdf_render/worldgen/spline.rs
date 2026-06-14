//! A 1-D **monotone cubic Hermite spline** with an analytic derivative — the `Curve` node's transfer
//! function (e.g. continentalness → base elevation, erosion → relief multiplier). C¹ value AND
//! derivative across knots (a piecewise-linear curve would kink the terrain normals at every knot), so
//! it threads cleanly through the autodiff [`super::graph::Field`].
//!
//! **Bit-portable / deterministic** (like [`super::noise`]): control points are f64, evaluation is f64
//! basic ops in fixed order. The only non-basic op is `sqrt`-free — monotonicity uses the simple
//! per-tangent clamp (`α,β ≤ 3`), a sufficient condition (Fritsch–Carlson). Catmull-Rom tangents,
//! Hermite basis in Horner form (no `mul_add`). Outside the knot domain the value clamps flat
//! (derivative 0) — measure-zero kinks at the ends, same treatment as `clamp`/`abs` in [`super::graph`].

/// Max control points a spline can hold (keeps it `Copy`, alloc-free). 8 is ample for terrain curves.
pub const SPLINE_MAX_POINTS: usize = 8;

/// A monotone cubic Hermite spline over up to [`SPLINE_MAX_POINTS`] control points, sorted ascending in
/// x. `Copy`/alloc-free so it lives in `Copy` params and the compiled graph. Serializes to RON as a
/// flat point list (`points`) — compact + editor-friendly — reconstructed via [`Spline::new`].
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, bevy::reflect::Reflect)]
#[serde(from = "SplineRon", into = "SplineRon")]
pub struct Spline {
    xs: [f64; SPLINE_MAX_POINTS],
    ys: [f64; SPLINE_MAX_POINTS],
    len: usize,
}

/// RON form of [`Spline`]: just the active `(x, y)` points (no fixed-array padding in the file).
#[derive(serde::Serialize, serde::Deserialize)]
struct SplineRon {
    points: Vec<(f64, f64)>,
}

impl From<SplineRon> for Spline {
    fn from(r: SplineRon) -> Self {
        Spline::new(&r.points)
    }
}

impl From<Spline> for SplineRon {
    fn from(s: Spline) -> Self {
        SplineRon { points: (0..s.len).map(|i| (s.xs[i], s.ys[i])).collect() }
    }
}

impl Spline {
    /// Build from `(x, y)` control points (must be ≥1 and sorted strictly ascending in x; extra points
    /// past [`SPLINE_MAX_POINTS`] are ignored). A single point ⇒ a constant.
    pub fn new(points: &[(f64, f64)]) -> Self {
        let len = points.len().min(SPLINE_MAX_POINTS);
        let mut xs = [0.0; SPLINE_MAX_POINTS];
        let mut ys = [0.0; SPLINE_MAX_POINTS];
        for (i, &(x, y)) in points.iter().take(len).enumerate() {
            xs[i] = x;
            ys[i] = y;
        }
        debug_assert!(len >= 1, "spline needs ≥1 control point");
        debug_assert!(xs[..len].windows(2).all(|w| w[1] > w[0]), "spline x must be strictly ascending");
        Self { xs, ys, len }
    }

    /// Largest `|y|` over the control points — the bound the terrain vertical-AABB band needs.
    pub fn max_abs_y(&self) -> f64 {
        self.ys[..self.len].iter().fold(0.0f64, |m, &y| m.max(y.abs()))
    }

    /// The active `(x, y)` control points (read-only) — what the WGSL `Curve` codegen emits as the
    /// per-node spline arrays (see `graph::wgsl_codegen`). Pure accessor; no math/behaviour change.
    pub fn points(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        (0..self.len).map(move |i| (self.xs[i], self.ys[i]))
    }

    /// Evaluate the spline at `x`, returning `(y, dy/dx)`. Flat-clamped outside `[x₀, x_{n-1}]`.
    pub fn eval(&self, x: f64) -> (f64, f64) {
        let n = self.len;
        if n == 1 || x <= self.xs[0] {
            return (self.ys[0], 0.0);
        }
        if x >= self.xs[n - 1] {
            return (self.ys[n - 1], 0.0);
        }
        // Segment i with xs[i] <= x < xs[i+1] (n ≤ 8 → linear scan is fine + branch-predictable).
        let mut i = 0;
        while i + 1 < n && x >= self.xs[i + 1] {
            i += 1;
        }
        let h = self.xs[i + 1] - self.xs[i];
        let mi = self.tangent(i);
        let mi1 = self.tangent(i + 1);
        let t = (x - self.xs[i]) / h;
        // Hermite basis + derivatives (w.r.t. t), Horner form.
        let t2 = t * t;
        let h00 = 2.0 * t2 * t - 3.0 * t2 + 1.0;
        let h10 = t2 * t - 2.0 * t2 + t;
        let h01 = -2.0 * t2 * t + 3.0 * t2;
        let h11 = t2 * t - t2;
        let h00d = 6.0 * t2 - 6.0 * t;
        let h10d = 3.0 * t2 - 4.0 * t + 1.0;
        let h01d = -6.0 * t2 + 6.0 * t;
        let h11d = 3.0 * t2 - 2.0 * t;
        let (y0, y1) = (self.ys[i], self.ys[i + 1]);
        let y = h00 * y0 + h10 * h * mi + h01 * y1 + h11 * h * mi1;
        let dy_dt = h00d * y0 + h10d * h * mi + h01d * y1 + h11d * h * mi1;
        (y, dy_dt / h)
    }

    /// Catmull-Rom tangent at knot `i` (one-sided at the ends), with the simple per-tangent
    /// monotonicity clamp applied via the adjacent secants.
    #[inline]
    fn tangent(&self, i: usize) -> f64 {
        let n = self.len;
        let raw = if i == 0 {
            (self.ys[1] - self.ys[0]) / (self.xs[1] - self.xs[0])
        } else if i == n - 1 {
            (self.ys[n - 1] - self.ys[n - 2]) / (self.xs[n - 1] - self.xs[n - 2])
        } else {
            (self.ys[i + 1] - self.ys[i - 1]) / (self.xs[i + 1] - self.xs[i - 1])
        };
        // Monotonicity clamp: a tangent may not exceed 3× either adjacent secant slope (Fritsch–Carlson
        // sufficient condition, sqrt-free). If an adjacent secant is flat (Δ=0), the tangent must be 0 to
        // avoid a non-monotone overshoot there.
        let mut m = raw;
        if i > 0 {
            let d = (self.ys[i] - self.ys[i - 1]) / (self.xs[i] - self.xs[i - 1]);
            m = clamp_to_secant(m, d);
        }
        if i + 1 < n {
            let d = (self.ys[i + 1] - self.ys[i]) / (self.xs[i + 1] - self.xs[i]);
            m = clamp_to_secant(m, d);
        }
        m
    }
}

/// Clamp tangent `m` so it doesn't exceed `3·d` in `d`'s direction (and is 0 if `d` is flat) — the
/// per-segment monotonicity guard. Same sign as `d` required (else 0).
#[inline]
fn clamp_to_secant(m: f64, d: f64) -> f64 {
    if d == 0.0 {
        return 0.0;
    }
    if m * d < 0.0 {
        return 0.0; // opposite sign to the secant → would overshoot non-monotonically
    }
    let lim = 3.0 * d;
    if d > 0.0 { m.min(lim) } else { m.max(lim) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s() -> Spline {
        Spline::new(&[(-1.0, 0.0), (-0.3, 0.2), (0.2, 1.0), (0.7, 1.6), (1.0, 2.6)])
    }

    #[test]
    fn passes_through_knots() {
        let sp = s();
        for &(x, y) in &[(-1.0, 0.0), (-0.3, 0.2), (0.2, 1.0), (0.7, 1.6), (1.0, 2.6)] {
            let (yy, _) = sp.eval(x);
            assert!((yy - y).abs() < 1e-9, "knot ({x},{y}) → {yy}");
        }
    }

    #[test]
    fn derivative_matches_central_difference() {
        let sp = s();
        let e = 1e-4;
        for k in 0..40 {
            let x = -0.95 + k as f64 * 0.045; // interior, avoids the flat-clamp ends
            let (_, d) = sp.eval(x);
            let cd = (sp.eval(x + e).0 - sp.eval(x - e).0) / (2.0 * e);
            assert!((d - cd).abs() < 1e-3, "d/dx at {x}: analytic {d} vs CD {cd}");
        }
    }

    #[test]
    fn c1_continuous_across_knots() {
        // Derivative is continuous across an interior knot (no kink): left/right limits agree.
        let sp = s();
        for &knot in &[-0.3, 0.2, 0.7] {
            let dl = sp.eval(knot - 1e-5).1;
            let dr = sp.eval(knot + 1e-5).1;
            assert!((dl - dr).abs() < 1e-2, "C1 break at knot {knot}: {dl} vs {dr}");
        }
    }

    #[test]
    fn flat_clamp_outside_domain() {
        let sp = s();
        assert_eq!(sp.eval(-5.0), (0.0, 0.0));
        assert_eq!(sp.eval(5.0), (2.6, 0.0));
    }

    #[test]
    fn monotone_input_stays_monotone() {
        // A monotone-increasing control set must produce a monotone-increasing curve (no overshoot dips).
        let sp = Spline::new(&[(0.0, 0.0), (0.4, 0.05), (0.5, 0.9), (1.0, 1.0)]);
        let mut prev = f64::NEG_INFINITY;
        for k in 0..=100 {
            let x = k as f64 * 0.01;
            let y = sp.eval(x).0;
            assert!(y >= prev - 1e-9, "non-monotone at x={x}: {y} < {prev}");
            prev = y;
        }
    }

    #[test]
    fn deterministic_bitwise() {
        let sp = s();
        assert_eq!(sp.eval(0.137).0.to_bits(), sp.eval(0.137).0.to_bits());
        assert_eq!(sp.eval(0.137).1.to_bits(), sp.eval(0.137).1.to_bits());
    }

    #[test]
    fn max_abs_y_is_largest_magnitude() {
        assert_eq!(Spline::new(&[(-1.0, -120.0), (0.0, 10.0), (1.0, 90.0)]).max_abs_y(), 120.0);
    }
}
