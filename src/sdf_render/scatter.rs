//! Deterministic point scattering over a region — a reusable primitive for placing objects on
//! terrain (towers, foliage, props). Pure and seed-driven: the same `ScatterParams` always yields
//! the same points, so a runtime scene and a headless test can scatter byte-identical layouts.
//!
//! The scatter is a jittered grid (Poisson-ish): one candidate per lattice cell, offset within the
//! cell by a hashed amount, optionally thinned by a hashed keep-probability. This gives an even-but-
//! organic spread at a controllable rough spacing without the cost/State of true Poisson-disk
//! sampling. Heights come from a caller-supplied closure so the scatter is independent of *what*
//! surface it lands on (heightmap today, any SDF surface later).

use bevy::prelude::*;

/// Parameters for a jittered-grid scatter over an XZ region centred on the origin.
#[derive(Clone, Copy, Debug)]
pub struct ScatterParams {
    /// Half-extent of the scatter region per axis (world units). Points land in `[-half, half]²`.
    pub half_extent: f32,
    /// Rough spacing between neighbours (world units) — the lattice cell size. Actual nearest-
    /// neighbour distance varies with jitter but averages near this.
    pub spacing: f32,
    /// Max per-axis jitter as a fraction of `spacing` (0 = perfect grid, 0.5 = fills the cell).
    pub jitter: f32,
    /// Keep probability in `[0, 1]`: each candidate is hash-thinned, so `1.0` keeps the full grid
    /// and `0.6` drops ~40% for a looser, less regular look. `<= 0` keeps none; `>= 1` keeps all.
    pub keep_prob: f32,
    /// Seed — distinct seeds give independent layouts at the same params.
    pub seed: u32,
}

impl Default for ScatterParams {
    fn default() -> Self {
        Self {
            half_extent: 100.0,
            spacing: 10.0,
            jitter: 0.4,
            keep_prob: 1.0,
            seed: 0xA11CE,
        }
    }
}

/// One scattered placement: world XZ plus a stable per-point hash the caller can use to derive
/// further pseudo-random attributes (rotation, scale, variant) without its own RNG.
#[derive(Clone, Copy, Debug)]
pub struct ScatterPoint {
    pub x: f32,
    pub z: f32,
    /// Stable hash of this point's lattice cell — seed for caller-side per-point variation.
    pub hash: u32,
}

/// Generate the scattered XZ points for `params` (no height — see [`scatter_on_surface`] to lift
/// onto a surface). Deterministic and allocation-bounded by the lattice cell count.
pub fn scatter_points(params: &ScatterParams) -> Vec<ScatterPoint> {
    let mut out = Vec::new();
    if params.spacing <= 0.0 || params.half_extent <= 0.0 {
        return out;
    }
    let cells = (params.half_extent / params.spacing).ceil() as i32;
    let jitter = params.jitter.clamp(0.0, 0.5);
    for gz in -cells..=cells {
        for gx in -cells..=cells {
            // Hash the cell once; split into independent streams for jitter X/Z + the keep roll.
            let h = hash_cell(gx, gz, params.seed);
            if params.keep_prob < 1.0 {
                let roll = (split(h, 3) as f32) / (u32::MAX as f32);
                if roll > params.keep_prob {
                    continue;
                }
            }
            let jx = (unit_signed(split(h, 1))) * jitter * params.spacing;
            let jz = (unit_signed(split(h, 2))) * jitter * params.spacing;
            let x = gx as f32 * params.spacing + jx;
            let z = gz as f32 * params.spacing + jz;
            if x.abs() > params.half_extent || z.abs() > params.half_extent {
                continue;
            }
            out.push(ScatterPoint { x, z, hash: h });
        }
    }
    out
}

/// As [`scatter_points`] but lifts each point onto a surface via `surface_y(x, z)`, returning the
/// full world position. The closure decouples the scatter from the surface representation.
pub fn scatter_on_surface(
    params: &ScatterParams,
    mut surface_y: impl FnMut(f32, f32) -> f32,
) -> Vec<(Vec3, u32)> {
    scatter_points(params)
        .into_iter()
        .map(|p| (Vec3::new(p.x, surface_y(p.x, p.z), p.z), p.hash))
        .collect()
}

/// Mix a lattice cell + seed into a well-distributed 32-bit hash.
fn hash_cell(gx: i32, gz: i32, seed: u32) -> u32 {
    let mut h = (gx as u32).wrapping_mul(374_761_393);
    h = h.wrapping_add((gz as u32).wrapping_mul(668_265_263));
    h = h.wrapping_add(seed.wrapping_mul(2_246_822_519));
    h ^= h >> 13;
    h = h.wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    h
}

/// Derive an independent sub-stream `k` from a base hash (so jitter-X, jitter-Z and the keep roll
/// don't correlate).
fn split(h: u32, k: u32) -> u32 {
    let mut x = h ^ k.wrapping_mul(0x9E37_79B9);
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 13;
    x = x.wrapping_mul(0xC2B2_AE35);
    x ^= x >> 16;
    x
}

/// Map a 32-bit hash to a signed unit value in `[-1, 1)`.
fn unit_signed(h: u32) -> f32 {
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_params_same_points() {
        let p = ScatterParams::default();
        let a = scatter_points(&p);
        let b = scatter_points(&p);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!((x.x, x.z, x.hash), (y.x, y.z, y.hash));
        }
    }

    #[test]
    fn points_stay_in_region() {
        let p = ScatterParams { half_extent: 50.0, spacing: 7.0, jitter: 0.5, ..default() };
        for pt in scatter_points(&p) {
            assert!(pt.x.abs() <= p.half_extent + 1e-3);
            assert!(pt.z.abs() <= p.half_extent + 1e-3);
        }
    }

    #[test]
    fn keep_prob_thins_the_set() {
        let full = ScatterParams { half_extent: 80.0, spacing: 5.0, keep_prob: 1.0, ..default() };
        let half = ScatterParams { keep_prob: 0.5, ..full };
        let nf = scatter_points(&full).len();
        let nh = scatter_points(&half).len();
        assert!(nh < nf, "keep_prob 0.5 should drop points ({nh} vs {nf})");
        assert!(nh > nf / 4, "0.5 keep should retain roughly half, not collapse ({nh} of {nf})");
    }

    #[test]
    fn spacing_controls_density() {
        let coarse = ScatterParams { half_extent: 100.0, spacing: 20.0, ..default() };
        let fine = ScatterParams { spacing: 5.0, ..coarse };
        assert!(scatter_points(&fine).len() > scatter_points(&coarse).len());
    }

    #[test]
    fn surface_lift_applies_height() {
        let p = ScatterParams { half_extent: 30.0, spacing: 10.0, jitter: 0.0, ..default() };
        let pts = scatter_on_surface(&p, |x, z| x + z);
        for (world, _) in pts {
            assert!((world.y - (world.x + world.z)).abs() < 1e-3);
        }
    }
}
