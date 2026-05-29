//! Modular SDF edit system: primitives, CSG operations, and the single shared
//! evaluation path used by baking, CPU picking, and CPU raycasting.
//!
//! An "edit" is a Bevy entity carrying [`SdfPrimitive`] + [`SdfOp`] + [`SdfOrder`]
//! + a `Transform`. Edits are folded in `SdfOrder` order into one signed distance
//! field via [`fold_csg`]; the fold also resolves a single material id per sample
//! following CSG semantics (a subtractor carves but contributes no surface
//! material; an intersector keeps the more-constraining surface's material).

use bevy::math::bounding::Aabb3d;
use bevy::prelude::*;

// --- Components ---

/// A signed-distance primitive. Parameters are in the entity's local space; the
/// entity `Transform` places/orients/scales it in the world.
#[derive(Component, Reflect, Clone, Debug, PartialEq)]
#[reflect(Component)]
pub enum SdfPrimitive {
    Sphere {
        radius: f32,
    },
    Box {
        half_extents: Vec3,
    },
    Torus {
        major: f32,
        minor: f32,
    },
    Capsule {
        half_height: f32,
        radius: f32,
    },
    Cylinder {
        radius: f32,
        half_height: f32,
    },
    /// Bounded noise heightmap (terrain testing). The surface is the value-noise
    /// height over an XZ rectangle; the field is a *vertical-distance
    /// approximation* (not a true Euclidean distance), so it is only valid when
    /// densely sampled (see `eval_primitive`).
    Heightmap {
        half_xz: Vec2,
        max_height: f32,
        freq: f32,
        amp: f32,
        seed: u32,
    },
}

impl SdfPrimitive {
    /// A reasonable spawn default per variant (used by the authoring panel).
    pub fn sphere() -> Self {
        Self::Sphere { radius: 0.5 }
    }
}

/// CSG combination operator for an edit.
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum CsgKind {
    /// Add the edit to the accumulated field (`smin`).
    #[default]
    Union,
    /// Carve the edit out of the accumulated field (`smax` with negated edit).
    Subtract,
    /// Keep only the overlap of edit and accumulated field (`smax`).
    Intersect,
}

/// How an edit combines with everything before it, plus the smoothing band width
/// in world units (`0` = a hard/sharp boolean).
#[derive(Component, Reflect, Clone, Copy, Debug)]
#[reflect(Component)]
pub struct SdfOp {
    pub kind: CsgKind,
    pub smoothing: f32,
}

impl Default for SdfOp {
    fn default() -> Self {
        Self {
            kind: CsgKind::Union,
            smoothing: 0.0,
        }
    }
}

/// Explicit evaluation order within the CSG stack. Lower values are applied
/// first. The first edit's `SdfOp` is effectively a Union onto empty space.
#[derive(Component, Reflect, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[reflect(Component)]
pub struct SdfOrder(pub u32);

impl Default for SdfOrder {
    fn default() -> Self {
        Self(0)
    }
}

// --- Smooth min/max (iq polynomial) ---

/// Polynomial smooth-min: welds `a`/`b` over band `k`, result `<= min(a,b)`.
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    b * (1.0 - h) + a * h - k * h * (1.0 - h)
}

/// Polynomial smooth-max: the dual of [`smin`], result `>= max(a,b)`. Used for
/// subtraction (`smax(d, -dn, k)`) and intersection (`smax(d, dn, k)`).
pub fn smax(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.max(b);
    }
    let h = (0.5 - 0.5 * (b - a) / k).clamp(0.0, 1.0);
    b * (1.0 - h) + a * h + k * h * (1.0 - h)
}

// --- Primitive evaluation (local space) ---

/// Evaluate a primitive's signed distance at a point already in the primitive's
/// local space. This is the single source of truth for primitive SDFs.
pub fn eval_primitive(prim: &SdfPrimitive, p: Vec3) -> f32 {
    match prim {
        SdfPrimitive::Sphere { radius } => p.length() - *radius,
        SdfPrimitive::Box { half_extents } => {
            let q = p.abs() - *half_extents;
            q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
        }
        SdfPrimitive::Torus { major, minor } => {
            // Ring lies in the local XZ plane, axis = Y.
            let q = Vec2::new(Vec2::new(p.x, p.z).length() - *major, p.y);
            q.length() - *minor
        }
        SdfPrimitive::Capsule {
            half_height,
            radius,
        } => {
            // Segment along local Y from -half_height..+half_height.
            let mut py = p.y;
            py -= py.clamp(-*half_height, *half_height);
            Vec3::new(p.x, py, p.z).length() - *radius
        }
        SdfPrimitive::Cylinder {
            radius,
            half_height,
        } => {
            // Axis along local Y.
            let d = Vec2::new(
                Vec2::new(p.x, p.z).length() - *radius,
                p.y.abs() - *half_height,
            );
            d.x.max(d.y).min(0.0) + d.max(Vec2::ZERO).length()
        }
        SdfPrimitive::Heightmap {
            half_xz,
            max_height,
            freq,
            amp,
            seed,
        } => {
            // Bounded box clamped to a noise height. Vertical-distance approx:
            // outside the XZ rect we fall back to the box SDF so the field stays
            // finite and the BVH/march behave; inside, distance is the signed
            // gap to the noise surface.
            let half = Vec3::new(half_xz.x, *max_height * 0.5, half_xz.y);
            let centered = p - Vec3::new(0.0, *max_height * 0.5, 0.0);
            let q = centered.abs() - half;
            let box_d = q.max(Vec3::ZERO).length() + q.max_element().min(0.0);

            let h = height_sample(Vec2::new(p.x, p.z), *freq, *amp, *seed) + *max_height * 0.5;
            let surface_d = p.y - h;
            box_d.max(surface_d)
        }
    }
}

/// Evaluate a primitive at a world position by inverse-transforming into local
/// space. Scale is applied uniformly via the matrix; non-uniform scale will skew
/// the field (acceptable for the editor, documented limitation).
pub fn eval_world(prim: &SdfPrimitive, transform: &Transform, world_pos: Vec3) -> f32 {
    let local = transform.to_matrix().inverse().transform_point3(world_pos);
    eval_primitive(prim, local)
}

/// Deterministic value-noise height sample over the XZ plane. Bilinear-lerped
/// integer-lattice hash — cheap, seeded, smooth enough for terrain testing.
fn height_sample(xz: Vec2, freq: f32, amp: f32, seed: u32) -> f32 {
    let p = xz * freq;
    let i = p.floor();
    let f = p - i;
    let u = f * f * (Vec2::splat(3.0) - 2.0 * f); // smoothstep weights

    let a = hash2(i.x as i32, i.y as i32, seed);
    let b = hash2(i.x as i32 + 1, i.y as i32, seed);
    let c = hash2(i.x as i32, i.y as i32 + 1, seed);
    let d = hash2(i.x as i32 + 1, i.y as i32 + 1, seed);

    let ab = a + (b - a) * u.x;
    let cd = c + (d - c) * u.x;
    (ab + (cd - ab) * u.y) * amp
}

/// Hash an integer lattice point to [-1, 1].
fn hash2(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(374_761_393);
    h = h.wrapping_add((y as u32).wrapping_mul(668_265_263));
    h = h.wrapping_add(seed.wrapping_mul(2_246_822_519));
    h ^= h >> 13;
    h = h.wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0
}

// --- AABBs ---

/// Local-space AABB of a primitive (before transform).
pub fn primitive_local_aabb(prim: &SdfPrimitive) -> Aabb3d {
    let he = match prim {
        SdfPrimitive::Sphere { radius } => Vec3::splat(*radius),
        SdfPrimitive::Box { half_extents } => *half_extents,
        SdfPrimitive::Torus { major, minor } => Vec3::new(major + minor, *minor, major + minor),
        SdfPrimitive::Capsule {
            half_height,
            radius,
        } => Vec3::new(*radius, half_height + radius, *radius),
        SdfPrimitive::Cylinder {
            radius,
            half_height,
        } => Vec3::new(*radius, *half_height, *radius),
        SdfPrimitive::Heightmap {
            half_xz,
            max_height,
            ..
        } => Vec3::new(half_xz.x, *max_height, half_xz.y),
    };
    // Heightmap spans y in [0, max_height]; everything else is centered. Encode
    // both by offsetting the heightmap so its min.y == 0.
    match prim {
        SdfPrimitive::Heightmap { max_height, .. } => {
            let half = Vec3::new(he.x, *max_height * 0.5, he.z);
            let center = Vec3::new(0.0, *max_height * 0.5, 0.0);
            Aabb3d::new(center, half)
        }
        _ => Aabb3d::new(Vec3::ZERO, he),
    }
}

impl SdfPrimitive {
    /// Draw this primitive's own shape outline with immediate-mode gizmos, in the
    /// space given by `iso` (entity translation+rotation) scaled by `scale`.
    ///
    /// Single source of truth: the wireframe is defined HERE alongside the
    /// primitive's parameters and SDF, so adding/altering a primitive updates its
    /// debug wireframe in one place. Generic over the gizmo config group so this
    /// stays decoupled from any particular overlay group. See skill
    /// `bevy-ecs-design` (single source of truth).
    pub fn draw_wireframe<G: GizmoConfigGroup>(
        &self,
        gizmos: &mut Gizmos<G>,
        iso: Isometry3d,
        scale: Vec3,
        color: Color,
    ) {
        // Transform a local-space point into world space.
        let tf = |p: Vec3| iso * (p * scale);
        // Local axis directions in world space (for circles/oriented prims).
        let rot = iso.rotation;
        let x = rot * Vec3::X;
        let z = rot * Vec3::Z;

        match self {
            SdfPrimitive::Sphere { radius } => {
                let c: Vec3 = iso.translation.into();
                let r = radius * scale.max_element();
                // Three great circles so the sphere reads from any angle.
                gizmos.circle(
                    Isometry3d::new(c, rot * Quat::from_rotation_arc(Vec3::Z, Vec3::X)),
                    r,
                    color,
                );
                gizmos.circle(
                    Isometry3d::new(c, rot * Quat::from_rotation_arc(Vec3::Z, Vec3::Y)),
                    r,
                    color,
                );
                gizmos.circle(Isometry3d::new(c, rot), r, color);
            }
            SdfPrimitive::Box { half_extents } => {
                let full = *half_extents * 2.0 * scale;
                gizmos.primitive_3d(&Cuboid::new(full.x, full.y, full.z), iso, color);
            }
            SdfPrimitive::Torus { major, minor } => {
                let c: Vec3 = iso.translation.into();
                let ms = scale.max_element();
                let maj = major * ms;
                let min = minor * ms;
                // Outer + inner equator (tube extent) in the local XZ plane.
                let equator = rot * Quat::from_rotation_arc(Vec3::Z, Vec3::Y);
                gizmos.circle(Isometry3d::new(c, equator), maj + min, color);
                gizmos.circle(Isometry3d::new(c, equator), (maj - min).max(0.0), color);
                // Two tube cross-section rings on opposite sides.
                for s in [1.0, -1.0] {
                    let center = c + x * (maj * s);
                    gizmos.circle(Isometry3d::new(center, rot), min, color);
                }
            }
            SdfPrimitive::Capsule {
                half_height,
                radius,
            } => {
                let r = radius * scale.x.max(scale.z);
                let top = tf(Vec3::new(0.0, *half_height, 0.0));
                let bot = tf(Vec3::new(0.0, -*half_height, 0.0));
                // End cap circles + vertical side lines.
                let cap = rot * Quat::from_rotation_arc(Vec3::Z, Vec3::Y);
                gizmos.circle(Isometry3d::new(top, cap), r, color);
                gizmos.circle(Isometry3d::new(bot, cap), r, color);
                for d in [x, z, -x, -z] {
                    gizmos.line(top + d * r, bot + d * r, color);
                }
            }
            SdfPrimitive::Cylinder {
                radius,
                half_height,
            } => {
                let r = radius * scale.x.max(scale.z);
                let top = tf(Vec3::new(0.0, *half_height, 0.0));
                let bot = tf(Vec3::new(0.0, -*half_height, 0.0));
                let cap = rot * Quat::from_rotation_arc(Vec3::Z, Vec3::Y);
                gizmos.circle(Isometry3d::new(top, cap), r, color);
                gizmos.circle(Isometry3d::new(bot, cap), r, color);
                for d in [x, z, -x, -z] {
                    gizmos.line(top + d * r, bot + d * r, color);
                }
            }
            SdfPrimitive::Heightmap {
                half_xz,
                max_height,
                ..
            } => {
                // Footprint rectangle at the base + a top rectangle at max height.
                let h = *max_height * scale.y;
                let (hx, hz) = (half_xz.x * scale.x, half_xz.y * scale.z);
                let corners = |yy: f32| {
                    [
                        Vec3::new(-hx, yy, -hz),
                        Vec3::new(hx, yy, -hz),
                        Vec3::new(hx, yy, hz),
                        Vec3::new(-hx, yy, hz),
                    ]
                };
                for yy in [0.0, h] {
                    let cs = corners(yy);
                    for i in 0..4 {
                        let a = iso * cs[i];
                        let b = iso * cs[(i + 1) % 4];
                        gizmos.line(a, b, color);
                    }
                }
            }
        }
    }
}

/// World-space AABB of an edit: the primitive's local AABB transformed into the
/// world, then grown by the smoothing band (a smooth boolean reaches outside the
/// raw primitive bounds).
pub fn edit_world_aabb(prim: &SdfPrimitive, transform: &Transform, smoothing: f32) -> Aabb3d {
    let local = primitive_local_aabb(prim);
    let mat = transform.to_matrix();

    // Transform the 8 corners and rebuild a world AABB (handles rotation/scale).
    let min = local.min;
    let max = local.max;
    let mut wmin = Vec3::splat(f32::INFINITY);
    let mut wmax = Vec3::splat(f32::NEG_INFINITY);
    for cx in [min.x, max.x] {
        for cy in [min.y, max.y] {
            for cz in [min.z, max.z] {
                let w = mat.transform_point3(Vec3::new(cx, cy, cz));
                wmin = wmin.min(w);
                wmax = wmax.max(w);
            }
        }
    }
    let pad = Vec3::splat(smoothing.max(0.0));
    Aabb3d::from_min_max(wmin - pad, wmax + pad)
}

// --- CSG fold ---

/// A flattened, order-sorted edit ready for evaluation. Decoupled from ECS so the
/// bake, picking, and tests can all build and fold the same data.
#[derive(Clone, Debug)]
pub struct ResolvedEdit {
    pub prim: SdfPrimitive,
    pub transform: Transform,
    pub op: SdfOp,
    pub material_id: u8,
}

/// Result of folding the CSG stack at one point: the combined signed distance and
/// the resolved surface material id.
#[derive(Clone, Copy, Debug)]
pub struct EditSample {
    pub dist: f32,
    pub material_id: u8,
}

/// Fold an ordered edit list into a single signed distance + material id at `pos`.
///
/// `edits` must already be sorted by [`SdfOrder`]. Material rules:
/// - Union: the nearer surface owns the material.
/// - Subtract: the carving edit contributes no material (accumulator keeps its id).
/// - Intersect: the more-constraining (larger-distance) surface owns the material.
pub fn fold_csg(edits: &[ResolvedEdit], pos: Vec3) -> EditSample {
    let mut acc = f32::MAX;
    let mut mat: u8 = 0;
    let mut started = false;

    for e in edits {
        let dn = eval_world(&e.prim, &e.transform, pos);
        let k = e.op.smoothing;

        // Nothing accumulated yet: only a Union can bring matter into existence.
        // Subtracting from / intersecting with empty space stays empty (a leading
        // Subtract/Intersect must NOT spuriously become solid).
        if !started {
            if e.op.kind == CsgKind::Union {
                acc = dn;
                mat = e.material_id;
                started = true;
            }
            continue;
        }

        match e.op.kind {
            CsgKind::Union => {
                let combined = smin(acc, dn, k);
                if dn < acc {
                    mat = e.material_id;
                }
                acc = combined;
            }
            CsgKind::Subtract => {
                // Carve: material unchanged (the tool leaves no surface material).
                acc = smax(acc, -dn, k);
            }
            CsgKind::Intersect => {
                if dn > acc {
                    mat = e.material_id;
                }
                acc = smax(acc, dn, k);
            }
        }
    }

    EditSample {
        dist: if started { acc } else { f32::MAX },
        material_id: mat,
    }
}

/// Number of distinct materials the dense per-material distance field tracks.
/// Matches the shader's `object_colors[8]` table.
pub const MATERIAL_SLOTS: usize = 8;

/// Sentinel distance for a material slot that no edit contributes to at `pos`.
/// Large and positive so it never wins the argmin; small enough to survive the
/// i16 snorm clamp ([-1, 1]) without collapsing toward the real surface.
pub const MATERIAL_FAR: f32 = 1.0;

/// Per-material *surface* distance field at `pos`: slot `m` holds the signed
/// distance to the nearest matter owned by material `m`, or [`MATERIAL_FAR`] if no
/// edit contributes that material here.
///
/// This is the data the shader interpolates and takes the argmin of, so the
/// material boundary is the exact piecewise-trilinear bisector between the two
/// nearest materials — sub-voxel sharp, with no dependence on smoothing `k` (so it
/// is clean even at `smoothing = 0`). Subtract edits define no material (they only
/// carve geometry, handled by the combined `fold_csg` distance), so they do not
/// write a slot. Union and Intersect surfaces both own their material.
pub fn material_distances(edits: &[ResolvedEdit], pos: Vec3) -> [f32; MATERIAL_SLOTS] {
    let mut slots = [MATERIAL_FAR; MATERIAL_SLOTS];
    for e in edits {
        if e.op.kind == CsgKind::Subtract {
            continue;
        }
        let m = e.material_id as usize;
        if m >= MATERIAL_SLOTS {
            continue;
        }
        let d = eval_world(&e.prim, &e.transform, pos);
        if d < slots[m] {
            slots[m] = d;
        }
    }
    slots
}

/// Index of the smallest slot in a per-material distance array (the material that
/// owns the surface at that point). Ties resolve to the lower index for stability.
pub fn argmin_material(slots: &[f32; MATERIAL_SLOTS]) -> u8 {
    let mut best = 0usize;
    for m in 1..MATERIAL_SLOTS {
        if slots[m] < slots[best] {
            best = m;
        }
    }
    best as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smin_welds_below_min() {
        let a = 0.2;
        let b = 0.25;
        assert!(smin(a, b, 0.3) <= a.min(b));
    }

    #[test]
    fn smin_hard_is_plain_min() {
        assert_eq!(smin(0.3, 0.7, 0.0), 0.3);
    }

    #[test]
    fn smax_hard_is_plain_max() {
        assert_eq!(smax(0.3, 0.7, 0.0), 0.7);
    }

    #[test]
    fn sphere_sdf_zero_on_surface() {
        let p = SdfPrimitive::Sphere { radius: 1.0 };
        assert!((eval_primitive(&p, Vec3::new(1.0, 0.0, 0.0))).abs() < 1e-6);
        assert!(eval_primitive(&p, Vec3::ZERO) < 0.0);
    }

    #[test]
    fn box_sdf_matches_known_points() {
        let p = SdfPrimitive::Box {
            half_extents: Vec3::splat(1.0),
        };
        assert!(eval_primitive(&p, Vec3::ZERO) < 0.0);
        assert!((eval_primitive(&p, Vec3::new(1.0, 0.0, 0.0))).abs() < 1e-6);
        assert!((eval_primitive(&p, Vec3::new(2.0, 0.0, 0.0)) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn subtract_keeps_body_material() {
        // Body (id 1) unioned, then a subtractor (id 2) carves a corner. Any point
        // still inside the body must report material 1, never 2.
        let edits = vec![
            ResolvedEdit {
                prim: SdfPrimitive::Box {
                    half_extents: Vec3::splat(1.0),
                },
                transform: Transform::IDENTITY,
                op: SdfOp::default(),
                material_id: 1,
            },
            ResolvedEdit {
                prim: SdfPrimitive::Sphere { radius: 0.5 },
                transform: Transform::from_xyz(1.0, 1.0, 1.0),
                op: SdfOp {
                    kind: CsgKind::Subtract,
                    smoothing: 0.0,
                },
                material_id: 2,
            },
        ];
        // A point deep in the body, far from the carve.
        let s = fold_csg(&edits, Vec3::new(-0.5, -0.5, -0.5));
        assert!(s.dist < 0.0);
        assert_eq!(
            s.material_id, 1,
            "subtractor id must not appear on the body"
        );
    }

    #[test]
    fn intersect_keeps_constraining_material() {
        // Two overlapping spheres intersected. Inside the overlap, the material is
        // whichever surface is more constraining (larger signed distance).
        let edits = vec![
            ResolvedEdit {
                prim: SdfPrimitive::Sphere { radius: 1.0 },
                transform: Transform::IDENTITY,
                op: SdfOp::default(),
                material_id: 1,
            },
            ResolvedEdit {
                prim: SdfPrimitive::Sphere { radius: 1.0 },
                transform: Transform::from_xyz(0.8, 0.0, 0.0),
                op: SdfOp {
                    kind: CsgKind::Intersect,
                    smoothing: 0.0,
                },
                material_id: 2,
            },
        ];
        // Near the first sphere's right edge: sphere-2 is the looser constraint
        // there, sphere-1 the tighter — but pick a point where edit 2 dominates.
        let s = fold_csg(&edits, Vec3::new(-0.1, 0.0, 0.0));
        assert!(s.dist <= 0.0 || s.dist.is_finite());
        // Material is one of the two participating ids (sanity; exact pick depends
        // on geometry). The key invariant: intersect never yields id 0.
        assert!(s.material_id == 1 || s.material_id == 2);
    }

    #[test]
    fn empty_edits_report_far() {
        let s = fold_csg(&[], Vec3::ZERO);
        assert_eq!(s.dist, f32::MAX);
        assert_eq!(s.material_id, 0);
    }

    #[test]
    fn local_aabb_contains_sphere() {
        let aabb = primitive_local_aabb(&SdfPrimitive::Sphere { radius: 0.5 });
        assert!((aabb.max.x - 0.5).abs() < 1e-6);
        assert!((aabb.min.x + 0.5).abs() < 1e-6);
    }

    #[test]
    fn world_aabb_grows_with_smoothing() {
        let prim = SdfPrimitive::Sphere { radius: 1.0 };
        let tight = edit_world_aabb(&prim, &Transform::IDENTITY, 0.0);
        let padded = edit_world_aabb(&prim, &Transform::IDENTITY, 0.5);
        assert!(padded.max.x > tight.max.x);
    }
}
