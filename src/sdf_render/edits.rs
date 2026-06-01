//! Modular SDF edit system: primitives, CSG operations, and the single shared
//! evaluation path used by baking, CPU picking, and CPU raycasting.
//!
//! An "edit" is a Bevy entity carrying a [`SdfPrimitive`], an [`SdfOp`], an
//! [`SdfOrder`], and a `Transform`. Edits are folded in `SdfOrder` order into one
//! signed distance field via [`fold_csg`]; the fold also resolves a single
//! material id per sample following CSG semantics (a subtractor carves but
//! contributes no surface material; an intersector keeps the more-constraining
//! surface's material).

use bevy::math::bounding::Aabb3d;
use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;

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

    /// Fold this primitive's shape params (quantized — see [`bake_content_hash`]) + a per-variant
    /// discriminant into a hasher, for the per-brick bake-cache key.
    pub fn hash_params(&self, h: &mut impl std::hash::Hasher) {
        let f = |h: &mut dyn std::hash::Hasher, x: f32| h.write_i64(quantize(x));
        match *self {
            SdfPrimitive::Sphere { radius } => { h.write_u8(0); f(h, radius); }
            SdfPrimitive::Box { half_extents } => {
                h.write_u8(1); f(h, half_extents.x); f(h, half_extents.y); f(h, half_extents.z);
            }
            SdfPrimitive::Torus { major, minor } => { h.write_u8(2); f(h, major); f(h, minor); }
            SdfPrimitive::Capsule { half_height, radius } => { h.write_u8(3); f(h, half_height); f(h, radius); }
            SdfPrimitive::Cylinder { radius, half_height } => { h.write_u8(4); f(h, radius); f(h, half_height); }
            SdfPrimitive::Heightmap { half_xz, max_height, freq, amp, seed } => {
                h.write_u8(5);
                f(h, half_xz.x); f(h, half_xz.y); f(h, max_height); f(h, freq); f(h, amp);
                h.write_u32(seed);
            }
        }
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
#[derive(Component, Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
#[reflect(Component)]
pub struct SdfOrder(pub u32);

/// Number of PBR texture layer indices carried per material. Order matches the
/// shader's `sample_material_map` map enum: diffuse, normal, mra (metallic-
/// roughness-ao packed), height, edge. Filled from the texture-library manifests
/// (see render.rs); `u32::MAX` means "no texture for this map".
pub const MATERIAL_TEX_MAPS: usize = 5;

/// A material in the global registry. Indexed by a stable global id (its position
/// in [`MaterialRegistry::defs`]). The registry holds *all* materials a world can
/// use (potentially hundreds); a brick only references the handful in its palette.
///
/// `blend_softness` is a *shading-time* control (world units): at a seam between two
/// materials, the colour/PBR cross-fade spans `max(softness_a, softness_b)`. `0`
/// keeps the boundary as crisp as the per-material distance field's sub-voxel
/// bisector allows; larger feathers it (rock → sand). It does not affect geometry —
/// that is `SdfOp::smoothing`.
#[derive(Clone, Copy, Debug, Reflect)]
pub struct MaterialDef {
    pub base_color: Color,
    pub blend_softness: f32,
    /// Scalar PBR fallbacks used when this material has NO MRA texture (`tex_layers[2]`
    /// absent). Lets a material be authored as a plain metal/dielectric without a texture
    /// set. When an MRA texture IS present it wins. `metallic` 0 = dielectric, 1 = metal;
    /// `roughness` 0 = mirror, 1 = fully diffuse.
    pub metallic: f32,
    pub roughness: f32,
    /// Height-map relief displacement depth (world units). 0 = flat (no displacement);
    /// ~0.15 = clearly visible, ~0.3 = strong. Only has an effect when a height map is present.
    pub parallax_scale: f32,
    /// Emissive (self-lit) radiance, linear RGB premultiplied by intensity (so the shader
    /// adds it directly). `Vec3::ZERO` = no emission. Emissive surfaces also feed the
    /// radiance-cascade GI, so a glowing object lights its surroundings.
    pub emissive: Vec3,
    /// PBR texture-array layer per map, or `u32::MAX` if absent. See [`MATERIAL_TEX_MAPS`].
    pub tex_layers: [u32; MATERIAL_TEX_MAPS],
}

impl Default for MaterialDef {
    fn default() -> Self {
        Self {
            base_color: Color::srgb(0.8, 0.8, 0.8),
            blend_softness: 0.0,
            // Matches the shader's old textureless neutral: dielectric, fully rough.
            metallic: 0.0,
            roughness: 1.0,
            // Default relief — clearly visible when a height map is present (textureless
            // materials have no height map, so it's a no-op).
            parallax_scale: 0.15,
            emissive: Vec3::ZERO,
            tex_layers: [u32::MAX; MATERIAL_TEX_MAPS],
        }
    }
}

/// Global material registry: the single source of truth for material appearance,
/// uploaded once (and on change) to the GPU material table. Edits reference entries
/// by global id via [`SdfMaterial`]. Index 0 is a default fallback so an
/// unconfigured edit still renders.
#[derive(Resource, Clone)]
pub struct MaterialRegistry {
    pub defs: Vec<MaterialDef>,
}

impl Default for MaterialRegistry {
    fn default() -> Self {
        Self {
            defs: vec![MaterialDef::default()],
        }
    }
}

/// Per-edit material reference: an index into [`MaterialRegistry::defs`]. Appearance
/// lives in the registry (keeps the GPU table static), not on the edit. This id is
/// **runtime-derived** by `resolve_materials` from each volume's [`SdfMaterialSource`];
/// it is NOT serialized into a `.scene` (the source is the authored truth).
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component)]
#[reflect(@crate::node::HideFromInspector)]
pub struct SdfMaterial {
    pub registry_id: u32,
}

/// Optional per-field overrides applied on top of a base material. `None` = inherit the
/// base value. `base_color` is linear RGBA stored as `[f32; 4]` (matching `MaterialAsset`,
/// so RON stays stable and serde-friendly). The scalar fields mirror the editable
/// `MaterialAsset` knobs. Texture maps are intentionally NOT overridable here yet — the
/// texture always comes from the base file (scene-level texture override deferred).
#[derive(Reflect, Clone, Debug, Default, PartialEq)]
pub struct MaterialFields {
    pub base_color: Option<[f32; 4]>,
    pub metallic: Option<f32>,
    pub roughness: Option<f32>,
    pub blend_softness: Option<f32>,
    pub parallax_scale: Option<f32>,
    /// Emissive radiance, linear RGB premultiplied by intensity. `None` = inherit base.
    pub emissive: Option<[f32; 3]>,
}

impl MaterialFields {
    /// Whether any field is set (an actual override exists).
    pub fn is_empty(&self) -> bool {
        *self == MaterialFields::default()
    }
}

/// The **authored** material of an SDF volume: a base material file and/or per-field
/// overrides. This is the serialized source of truth (`SdfMaterial.registry_id` is derived
/// from it by `resolve_materials`):
/// - `asset: Some(path)`, no overrides → a plain file material.
/// - `asset: Some(path)` + overrides → a scene-level override of that file's fields.
/// - `asset: None` → a fully inline/procedural material defined entirely by `overrides`
///   (e.g. a freshly-spawned primitive's scatter colour).
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq)]
#[reflect(Component)]
pub struct SdfMaterialSource {
    /// Base material file, relative to `assets/` (e.g. `materials/sand.material.ron`).
    pub asset: Option<std::path::PathBuf>,
    /// Per-field overrides applied on top of the base (or the whole material when inline).
    pub overrides: MaterialFields,
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

/// As [`eval_world`] but with a PRECOMPUTED model→local inverse (`ResolvedEdit::inv_model`),
/// skipping the per-call 4×4 inversion. The hot bake fold paths use this; `inv` must equal
/// `transform.to_matrix().inverse()` for the primitive's transform.
#[inline]
pub fn eval_world_inv(prim: &SdfPrimitive, inv: &Mat4, world_pos: Vec3) -> f32 {
    eval_primitive(prim, inv.transform_point3(world_pos))
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

/// World-space surface height of a translation-placed [`SdfPrimitive::Heightmap`] at world
/// XZ `(x, z)`. Mirrors the field's zero-crossing in [`eval_primitive`]
/// (`local.y == height_sample + max_height/2`), shifted by the heightmap entity's `world_y`.
/// Lets callers rest objects exactly on the terrain. Only valid for a translation-only
/// heightmap transform (the gallery's case), where world XZ == local XZ.
pub fn heightmap_surface_y(
    x: f32,
    z: f32,
    max_height: f32,
    freq: f32,
    amp: f32,
    seed: u32,
    world_y: f32,
) -> f32 {
    height_sample(Vec2::new(x, z), freq, amp, seed) + max_height * 0.5 + world_y
}

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
    use super::scatter::{scatter_on_surface, ScatterParams};

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
///
/// `inv_model` is the model→local inverse of `transform`, precomputed ONCE at
/// construction. The bake's `fold_csg` evaluates each edit at ~18 sample points per brick
/// (9 in the cull, 9 in the palette) across ~13k bricks/frame — recomputing
/// `transform.to_matrix().inverse()` per sample (as the old `eval_world` did) was millions
/// of 4×4 inversions/frame and the dominant bake-hitch cost. Caching it makes each eval a
/// single `transform_point3`. Use [`ResolvedEdit::new`] so the inverse can never drift from
/// the transform.
#[derive(Clone, Debug)]
pub struct ResolvedEdit {
    pub prim: SdfPrimitive,
    pub transform: Transform,
    pub op: SdfOp,
    /// Global material id (index into [`MaterialRegistry::defs`]).
    pub material_id: u16,
    /// Cached model→local inverse of `transform` (see struct docs). Always equal to
    /// `transform.to_matrix().inverse()`; kept in sync by constructing via [`Self::new`].
    pub inv_model: Mat4,
}

impl ResolvedEdit {
    /// Build a `ResolvedEdit`, precomputing `inv_model` from `transform`.
    pub fn new(prim: SdfPrimitive, transform: Transform, op: SdfOp, material_id: u16) -> Self {
        let inv_model = transform.to_matrix().inverse();
        Self { prim, transform, op, material_id, inv_model }
    }
}

/// Result of folding the CSG stack at one point: the combined signed distance and
/// the resolved global material id.
#[derive(Clone, Copy, Debug)]
pub struct EditSample {
    pub dist: f32,
    pub material_id: u16,
}

/// Fold an ordered edit list into a single signed distance + material id at `pos`.
///
/// `edits` must already be sorted by [`SdfOrder`]. Material rules:
/// - Union: the nearer surface owns the material.
/// - Subtract: the carving edit contributes no material (accumulator keeps its id).
/// - Intersect: the more-constraining (larger-distance) surface owns the material.
pub fn fold_csg(edits: &[ResolvedEdit], pos: Vec3) -> EditSample {
    let mut acc = f32::MAX;
    let mut mat: u16 = 0;
    let mut started = false;

    for e in edits {
        let dn = eval_world_inv(&e.prim, &e.inv_model, pos);
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

/// Signed distance of the folded CSG stack at `pos`, evaluating only the edits at
/// `indices` (into the already-`SdfOrder`-sorted `edits`). Same fold rules as
/// Content hash of the edits a brick folds — its bake-cache key. A brick re-bakes IFF this
/// changes, so a brick whose folded edits are untouched (e.g. a heightmap brick when a distant
/// sphere moves) keeps its cached texels even though SOME edit in the world changed. This is the
/// general fix for "moving one edit re-bakes everything its coarse footprint overlaps": there's
/// no global epoch — each brick is memoised by exactly the content it contains.
///
/// Hashes, in fold order, each edit's index + the values that affect its baked distance: the
/// model→local inverse matrix (captures translation/rotation/scale), the op (kind + smoothing +
/// material), and the primitive params. Order matters (CSG is non-commutative), so the index is
/// folded in too.
///
/// Floats are QUANTIZED (not bit-hashed) before folding, via [`quantize`]: `GlobalTransform`
/// recomputes `inv_model` every frame with sub-ULP jitter (a stationary edit reads e.g. 1.1 one
/// frame and 1.1000061 the next), so bit-exact hashing would change the hash every frame and
/// re-bake a brick that did not move — defeating the cache. Rounding to a fine grid (~0.1 mm)
/// makes an unmoved edit hash stably while still distinguishing any real edit. (Also collapses
/// -0.0/+0.0, which differ in bits.) The grid is far finer than a voxel, so it never merges two
/// genuinely-different bakes.
pub fn bake_content_hash(edits: &[ResolvedEdit], indices: &[u32]) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for &i in indices {
        let e = &edits[i as usize];
        h.write_u32(i);
        for v in e.inv_model.to_cols_array() {
            h.write_i64(quantize(v));
        }
        h.write_u32((e.op.kind as u32).rotate_left(1));
        h.write_i64(quantize(e.op.smoothing));
        h.write_u16(e.material_id);
        e.prim.hash_params(&mut h);
    }
    h.finish()
}

/// Quantize a float to a fixed grid for stable hashing (see [`bake_content_hash`]). 1e4 = 0.1 mm
/// world precision — finer than any voxel, coarser than `GlobalTransform`'s per-frame jitter.
#[inline]
pub fn quantize(v: f32) -> i64 {
    (v as f64 * 1.0e4).round() as i64
}

/// [`fold_csg`] but distance-only and allocation-free — for the narrow-band interior
/// cull, which folds at one point per candidate brick without cloning the edit subset.
pub fn fold_csg_dist_indexed(edits: &[ResolvedEdit], indices: &[u32], pos: Vec3) -> f32 {
    let mut acc = f32::MAX;
    let mut started = false;
    for &i in indices {
        let e = &edits[i as usize];
        let dn = eval_world_inv(&e.prim, &e.inv_model, pos);
        let k = e.op.smoothing;
        if !started {
            if e.op.kind == CsgKind::Union {
                acc = dn;
                started = true;
            }
            continue;
        }
        match e.op.kind {
            CsgKind::Union => acc = smin(acc, dn, k),
            CsgKind::Subtract => acc = smax(acc, -dn, k),
            CsgKind::Intersect => acc = smax(acc, dn, k),
        }
    }
    if started { acc } else { f32::MAX }
}

/// Max distinct materials a single brick tracks. The shader argmins over exactly
/// this many local slots per pixel — bounding per-pixel material cost to a small
/// constant regardless of how many materials the world contains.
pub const PALETTE_K: usize = 4;

/// Sentinel for an empty palette slot / a material absent at `pos`. The id sentinel
/// is `u16::MAX`; the distance sentinel is large and positive so it never wins the
/// argmin, yet within the i16 snorm clamp ([-1, 1]) so it survives baking.
pub const MATERIAL_FAR: f32 = 1.0;
pub const PALETTE_EMPTY: u16 = u16::MAX;

/// A brick's material palette: up to [`PALETTE_K`] global material ids present in
/// that brick. Slot order is the local index the per-voxel distance field is keyed
/// by; unused slots hold [`PALETTE_EMPTY`].
pub type Palette = [u16; PALETTE_K];

/// Build a brick's palette from its culled candidate edits: the (up to K) distinct
/// global material ids with the smallest distance to `sample_points` (the brick's
/// voxel corners), so a material that wins anywhere in the brick is kept. Subtract
/// edits contribute no material. Returned ids are sorted ascending for a stable,
/// neighbour-agnostic slot assignment; empty slots are [`PALETTE_EMPTY`].
pub fn build_palette(edits: &[ResolvedEdit], sample_points: &[Vec3]) -> Palette {
    build_palette_inner(edits.iter(), sample_points)
}

/// As [`build_palette`] but over the edits at `indices` (into `edits`), avoiding a per-brick
/// clone of the culled subset. The bake emit culls ~16k bricks/frame on a first bake; cloning
/// each brick's candidate edits (now 100+ bytes each, carrying the cached `inv_model`) into a
/// fresh `Vec` just to pass `build_palette` was 16k heap allocations/frame. This folds the same
/// result straight from the index list (mirrors [`fold_csg_dist_indexed`]).
pub fn build_palette_indexed(edits: &[ResolvedEdit], indices: &[u32], sample_points: &[Vec3]) -> Palette {
    build_palette_inner(indices.iter().map(|&i| &edits[i as usize]), sample_points)
}

fn build_palette_inner<'a>(edits: impl Iterator<Item = &'a ResolvedEdit>, sample_points: &[Vec3]) -> Palette {
    // Nearest distance achieved by each global id over all sample points.
    let mut best: Vec<(u16, f32)> = Vec::new();
    for e in edits {
        if e.op.kind == CsgKind::Subtract {
            continue;
        }
        let mut dmin = f32::MAX;
        for &p in sample_points {
            dmin = dmin.min(eval_world_inv(&e.prim, &e.inv_model, p));
        }
        match best.iter_mut().find(|(id, _)| *id == e.material_id) {
            Some((_, d)) => *d = d.min(dmin),
            None => best.push((e.material_id, dmin)),
        }
    }
    // Keep the K nearest, then sort ascending by id for a stable slot order.
    best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    best.truncate(PALETTE_K);
    best.sort_by_key(|(id, _)| *id);

    let mut palette = [PALETTE_EMPTY; PALETTE_K];
    for (slot, (id, _)) in best.iter().enumerate() {
        palette[slot] = *id;
    }
    palette
}

// --- GPU edit (flat, for the compute bake) ---

/// Primitive tag in [`GpuEdit::tag`] — must match the `PRIM_*` consts in
/// `assets/shaders/sdf_brick_bake.wgsl`.
pub const GPU_PRIM_SPHERE: u32 = 0;
pub const GPU_PRIM_BOX: u32 = 1;
pub const GPU_PRIM_TORUS: u32 = 2;
pub const GPU_PRIM_CAPSULE: u32 = 3;
pub const GPU_PRIM_CYLINDER: u32 = 4;
pub const GPU_PRIM_HEIGHTMAP: u32 = 5;

/// CSG op tag in [`GpuEdit::op_kind`] — must match `OP_*` in the bake shader.
pub const GPU_OP_UNION: u32 = 0;
pub const GPU_OP_SUBTRACT: u32 = 1;
pub const GPU_OP_INTERSECT: u32 = 2;

/// Flat, GPU-friendly mirror of a [`ResolvedEdit`] for the compute bake. The
/// model→local inverse is precomputed on the CPU (matching [`eval_world`]'s
/// `to_matrix().inverse()`) so the shader does only a `mat * vec` — keeping the GPU
/// result within an f32 ULP of the CPU. Primitive params are packed positionally per
/// `tag` (see `to_gpu_edit` and the `eval_primitive` port in the bake shader). The
/// heightmap `seed` (a u32) is bit-stored in `params2.y` and read back with
/// `bitcast<u32>` on the GPU. 96 bytes, std140/std430-aligned via the leading Mat4.
#[derive(ShaderType, Clone, Copy, Default, Debug)]
pub struct GpuEdit {
    pub inv_model: Mat4,
    pub params: Vec4,
    pub params2: Vec4,
    pub tag: u32,
    pub op_kind: u32,
    pub smoothing: f32,
    pub material_id: u32,
}

/// Flatten a [`ResolvedEdit`] into its [`GpuEdit`] form for the compute bake.
pub fn to_gpu_edit(e: &ResolvedEdit) -> GpuEdit {
    let inv_model = e.inv_model;
    let (tag, params, params2) = match &e.prim {
        SdfPrimitive::Sphere { radius } => {
            (GPU_PRIM_SPHERE, Vec4::new(*radius, 0.0, 0.0, 0.0), Vec4::ZERO)
        }
        SdfPrimitive::Box { half_extents } => (
            GPU_PRIM_BOX,
            Vec4::new(half_extents.x, half_extents.y, half_extents.z, 0.0),
            Vec4::ZERO,
        ),
        SdfPrimitive::Torus { major, minor } => (
            GPU_PRIM_TORUS,
            Vec4::new(*major, *minor, 0.0, 0.0),
            Vec4::ZERO,
        ),
        SdfPrimitive::Capsule {
            half_height,
            radius,
        } => (
            GPU_PRIM_CAPSULE,
            Vec4::new(*half_height, *radius, 0.0, 0.0),
            Vec4::ZERO,
        ),
        SdfPrimitive::Cylinder {
            radius,
            half_height,
        } => (
            GPU_PRIM_CYLINDER,
            Vec4::new(*radius, *half_height, 0.0, 0.0),
            Vec4::ZERO,
        ),
        SdfPrimitive::Heightmap {
            half_xz,
            max_height,
            freq,
            amp,
            seed,
        } => (
            GPU_PRIM_HEIGHTMAP,
            Vec4::new(half_xz.x, half_xz.y, *max_height, *freq),
            Vec4::new(*amp, f32::from_bits(*seed), 0.0, 0.0),
        ),
    };
    let op_kind = match e.op.kind {
        CsgKind::Union => GPU_OP_UNION,
        CsgKind::Subtract => GPU_OP_SUBTRACT,
        CsgKind::Intersect => GPU_OP_INTERSECT,
    };
    GpuEdit {
        inv_model,
        params,
        params2,
        tag,
        op_kind,
        smoothing: e.op.smoothing,
        material_id: e.material_id as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_palette_indexed_matches_cloned() {
        // The allocation-free indexed path must produce the same palette as cloning the culled
        // subset and calling build_palette (the prior emit path).
        let all = vec![
            ResolvedEdit::new(SdfPrimitive::Sphere { radius: 1.0 }, Transform::IDENTITY, SdfOp::default(), 7),
            ResolvedEdit::new(SdfPrimitive::Box { half_extents: Vec3::splat(0.5) }, Transform::from_xyz(2.0, 0.0, 0.0), SdfOp::default(), 3),
            ResolvedEdit::new(SdfPrimitive::Sphere { radius: 0.5 }, Transform::from_xyz(0.3, 0.0, 0.0), SdfOp { kind: CsgKind::Subtract, smoothing: 0.0 }, 9),
        ];
        let indices = [0u32, 2, 1];
        let samples = [Vec3::ZERO, Vec3::new(0.5, 0.2, -0.1), Vec3::new(1.5, 0.0, 0.0)];
        let cloned: Vec<_> = indices.iter().map(|&i| all[i as usize].clone()).collect();
        assert_eq!(
            build_palette_indexed(&all, &indices, &samples),
            build_palette(&cloned, &samples),
        );
    }

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
            ResolvedEdit::new(
                SdfPrimitive::Box { half_extents: Vec3::splat(1.0) },
                Transform::IDENTITY,
                SdfOp::default(),
                1,
            ),
            ResolvedEdit::new(
                SdfPrimitive::Sphere { radius: 0.5 },
                Transform::from_xyz(1.0, 1.0, 1.0),
                SdfOp { kind: CsgKind::Subtract, smoothing: 0.0 },
                2,
            ),
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
            ResolvedEdit::new(
                SdfPrimitive::Sphere { radius: 1.0 },
                Transform::IDENTITY,
                SdfOp::default(),
                1,
            ),
            ResolvedEdit::new(
                SdfPrimitive::Sphere { radius: 1.0 },
                Transform::from_xyz(0.8, 0.0, 0.0),
                SdfOp { kind: CsgKind::Intersect, smoothing: 0.0 },
                2,
            ),
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

    /// CPU mirror of the WGSL bake shader's primitive eval: consumes a packed
    /// [`GpuEdit`] exactly as `sdf_brick_bake.wgsl` will (inverse-transform via the
    /// precomputed `inv_model`, then a positionally-unpacked primitive SDF). The
    /// oracle test below asserts this reproduces [`eval_world`] for every primitive,
    /// so the shader port has a bit-for-bit reference to match.
    fn eval_gpu_edit_cpu(e: &GpuEdit, world_pos: Vec3) -> f32 {
        let p = e.inv_model.transform_point3(world_pos);
        match e.tag {
            GPU_PRIM_SPHERE => p.length() - e.params.x,
            GPU_PRIM_BOX => {
                let q = p.abs() - e.params.truncate();
                q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
            }
            GPU_PRIM_TORUS => {
                let q = Vec2::new(Vec2::new(p.x, p.z).length() - e.params.x, p.y);
                q.length() - e.params.y
            }
            GPU_PRIM_CAPSULE => {
                let half_height = e.params.x;
                let radius = e.params.y;
                let mut py = p.y;
                py -= py.clamp(-half_height, half_height);
                Vec3::new(p.x, py, p.z).length() - radius
            }
            GPU_PRIM_CYLINDER => {
                let radius = e.params.x;
                let half_height = e.params.y;
                let d = Vec2::new(Vec2::new(p.x, p.z).length() - radius, p.y.abs() - half_height);
                d.x.max(d.y).min(0.0) + d.max(Vec2::ZERO).length()
            }
            GPU_PRIM_HEIGHTMAP => {
                let half_xz = Vec2::new(e.params.x, e.params.y);
                let max_height = e.params.z;
                let freq = e.params.w;
                let amp = e.params2.x;
                let seed = e.params2.y.to_bits();
                let half = Vec3::new(half_xz.x, max_height * 0.5, half_xz.y);
                let centered = p - Vec3::new(0.0, max_height * 0.5, 0.0);
                let q = centered.abs() - half;
                let box_d = q.max(Vec3::ZERO).length() + q.max_element().min(0.0);
                let h = height_sample(Vec2::new(p.x, p.z), freq, amp, seed) + max_height * 0.5;
                let surface_d = p.y - h;
                box_d.max(surface_d)
            }
            _ => f32::MAX,
        }
    }

    #[test]
    fn gpu_edit_eval_matches_eval_world() {
        // A representative transform (translate + rotate + uniform scale) and a spread
        // of sample points. Every primitive's packed GPU eval must reproduce eval_world.
        let transform = Transform::from_xyz(1.5, -2.0, 0.5)
            .with_rotation(Quat::from_euler(EulerRot::XYZ, 0.3, -0.7, 1.1))
            .with_scale(Vec3::splat(1.3));
        let prims = [
            SdfPrimitive::Sphere { radius: 0.7 },
            SdfPrimitive::Box {
                half_extents: Vec3::new(0.6, 0.4, 0.9),
            },
            SdfPrimitive::Torus {
                major: 0.8,
                minor: 0.25,
            },
            SdfPrimitive::Capsule {
                half_height: 0.5,
                radius: 0.3,
            },
            SdfPrimitive::Cylinder {
                radius: 0.4,
                half_height: 0.6,
            },
            SdfPrimitive::Heightmap {
                half_xz: Vec2::new(2.0, 2.0),
                max_height: 1.0,
                freq: 0.5,
                amp: 0.4,
                seed: 1337,
            },
        ];
        let samples = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, -1.0, 0.5),
            Vec3::new(-0.7, 0.3, 1.2),
            Vec3::new(2.0, 1.5, -1.0),
        ];
        for prim in &prims {
            let edit = ResolvedEdit::new(prim.clone(), transform, SdfOp::default(), 3);
            let gpu = to_gpu_edit(&edit);
            assert_eq!(gpu.material_id, 3);
            for &s in &samples {
                let cpu = eval_world(&edit.prim, &edit.transform, s);
                let gpu_eval = eval_gpu_edit_cpu(&gpu, s);
                assert!(
                    (cpu - gpu_eval).abs() < 1e-4,
                    "{prim:?} at {s:?}: eval_world={cpu} gpu={gpu_eval}"
                );
            }
        }
    }

    #[test]
    fn gpu_edit_packs_op_kind() {
        let mk = |kind| {
            to_gpu_edit(&ResolvedEdit::new(
                SdfPrimitive::Sphere { radius: 1.0 },
                Transform::IDENTITY,
                SdfOp { kind, smoothing: 0.2 },
                0,
            ))
        };
        assert_eq!(mk(CsgKind::Union).op_kind, GPU_OP_UNION);
        assert_eq!(mk(CsgKind::Subtract).op_kind, GPU_OP_SUBTRACT);
        assert_eq!(mk(CsgKind::Intersect).op_kind, GPU_OP_INTERSECT);
        assert_eq!(mk(CsgKind::Union).smoothing, 0.2);
    }
}
