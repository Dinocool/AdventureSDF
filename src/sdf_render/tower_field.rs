//! The procedural cube-tower stress-scene field generator. A scene-content builder (not part of the
//! core CSG eval in [`super::edits`]): it scatters a heightmap ground + rotated-cube towers, each
//! capped by a sphere, producing role-tagged edits. Pure + seed-driven so the runtime
//! [`super::stress::TowerSpawner`] and the bake-cache regression test produce byte-identical geometry.

use bevy::prelude::*;

use super::edits::{heightmap_surface_y, SdfOrder, SdfPrimitive};
use super::scatter::{scatter_on_surface, ScatterParams};

/// The role a tower-field edit plays — decouples geometry from material. Callers map each role to
/// a concrete material (the runtime spawner → a `.material.ron` path; the bake test → a `u32` id).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TowerRole {
    /// The wide procedural heightmap ground.
    Ground,
    /// A stacked, rotated cube in a tower.
    Cube,
    /// The sphere capping a tower.
    Cap,
}

/// One tower-field edit: evaluation order, world transform, primitive, and its material role.
pub type TowerEdit = (SdfOrder, Transform, SdfPrimitive, TowerRole);

/// Parameters for the scattered cube-tower field — the SDF stress scene. Procedurally places a
/// heightmap ground and a scatter of rotated-cube towers (each capped by a sphere) resting on it.
/// Pure + seed-driven so the runtime `TowerSpawner` and the bake regression test produce identical
/// geometry. Heavy edit count by design: stresses the BVH cull + per-edit AABB refine.
#[derive(Clone, Copy, Debug)]
pub struct TowerFieldParams {
    /// Heightmap world Y offset (ground dropped below origin so towers sit above terrain).
    pub ground_y: f32,
    pub max_height: f32,
    pub freq: f32,
    pub amp: f32,
    pub seed: u32,
    /// Half-extent of the scatter region (world units) and rough spacing between towers.
    pub half_extent: f32,
    pub spacing: f32,
    pub jitter: f32,
    pub cubes_per_tower: u32,
    pub cube_half: f32,
}

impl Default for TowerFieldParams {
    fn default() -> Self {
        Self {
            ground_y: -30.0,
            max_height: 20.0,
            freq: 0.02,
            amp: 5.0,
            seed: 1337,
            // ~10 m spacing over ±270 m → ~55×55 lattice ≈ 3000 towers.
            half_extent: 270.0,
            spacing: 10.0,
            jitter: 0.4,
            cubes_per_tower: 4,
            cube_half: 0.4,
        }
    }
}

/// Build the scattered cube-tower field deterministically from `params`. Returns role-tagged edits
/// (material-agnostic — see [`TowerRole`]); the first edit is always the [`TowerRole::Ground`]
/// heightmap. Pure and allocation-bounded by the scatter lattice.
pub fn tower_field_edits(params: &TowerFieldParams) -> Vec<TowerEdit> {
    let scatter = ScatterParams {
        half_extent: params.half_extent,
        spacing: params.spacing,
        jitter: params.jitter,
        keep_prob: 1.0,
        seed: params.seed,
    };

    let mut out: Vec<TowerEdit> = Vec::new();
    let mut order = 0u32;
    let mut push = |o: &mut u32, t: Transform, p: SdfPrimitive, role: TowerRole| {
        out.push((SdfOrder(*o), t, p, role));
        *o += 1;
    };

    // Ground: large procedural heightmap, dropped below the origin so towers sit above it.
    push(
        &mut order,
        Transform::from_xyz(0.0, params.ground_y, 0.0),
        SdfPrimitive::Heightmap {
            half_xz: Vec2::new(1000.0, 1000.0),
            max_height: params.max_height,
            freq: params.freq,
            amp: params.amp,
            seed: params.seed,
        },
        TowerRole::Ground,
    );

    let cube_half = params.cube_half;
    let towers = scatter_on_surface(&scatter, |x, z| {
        heightmap_surface_y(x, z, params.max_height, params.freq, params.amp, params.seed, params.ground_y)
    });
    for (base, h) in towers {
        let (tx, base_y, tz) = (base.x, base.y, base.z);
        for c in 0..params.cubes_per_tower {
            // Stack the cubes; first cube rests its base on the terrain (centre at +half).
            let cy = base_y + cube_half + (c as f32) * (2.0 * cube_half);
            // Deterministic FULLY-random orientation from the point hash + cube index (uniform on
            // SO(3) via Shoemake), so each cube tilts/rolls every which way, not just yaw.
            let rot = random_rotation(
                h ^ (c.wrapping_mul(0x9E37)).wrapping_add(1),
                h.rotate_left(7) ^ (c.wrapping_mul(0x85EB)).wrapping_add(3),
                h.rotate_left(13) ^ (c.wrapping_mul(0xC2B2)).wrapping_add(5),
            );
            push(
                &mut order,
                Transform {
                    translation: Vec3::new(tx, cy, tz),
                    rotation: rot,
                    scale: Vec3::ONE,
                },
                SdfPrimitive::Box { half_extents: Vec3::splat(cube_half) },
                TowerRole::Cube,
            );
        }

        // Sphere capping the tower.
        let top_y = base_y + (params.cubes_per_tower as f32) * (2.0 * cube_half) + 0.45;
        push(
            &mut order,
            Transform::from_xyz(tx, top_y, tz),
            SdfPrimitive::Sphere { radius: 0.45 },
            TowerRole::Cap,
        );
    }

    out
}

/// A uniformly-distributed random orientation on SO(3) (Shoemake 1992), built deterministically
/// from three hashes mapped to `[0, 1)`. Gives an unbiased random cube tumble (full tilt + roll),
/// unlike an euler-angle pick which clusters near the poles.
fn random_rotation(h0: u32, h1: u32, h2: u32) -> Quat {
    let u0 = (h0 as f32 / u32::MAX as f32).clamp(0.0, 1.0);
    let u1 = (h1 as f32 / u32::MAX as f32).clamp(0.0, 1.0);
    let u2 = (h2 as f32 / u32::MAX as f32).clamp(0.0, 1.0);
    let tau = std::f32::consts::TAU;
    let (s1, s2) = ((1.0 - u0).sqrt(), u0.sqrt());
    let (t1, t2) = (tau * u1, tau * u2);
    // Quat(x, y, z, w) — already unit length.
    Quat::from_xyzw(s1 * t1.sin(), s1 * t1.cos(), s2 * t2.sin(), s2 * t2.cos())
}
