//! Transform gizmo handles as reusable, single-source geometry.
//!
//! There is no "current mode": the universal manipulator shows every handle at
//! once — translate arrows, rotate rings, per-axis scale handles, and planar
//! scale squares. Clicking any handle performs its action.
//!
//! Each handle's shape is defined ONCE here and consumed by both rendering
//! (`Handle::draw`) and CPU picking (`Handle::sdf`). They cannot drift apart —
//! previously the drawn and picked sizes were duplicated and silently diverged.
//! See skill `bevy-ecs-design` (single source of truth) and memory
//! `gizmo-single-source-geometry`.

use bevy::prelude::*;

use super::SdfOverlayGizmos;

// --- Shared handle dimensions (the single source) ---

const TRANSLATE_LEN: f32 = 1.0;
const SCALE_LEN: f32 = 0.6;
const ROTATE_RADIUS: f32 = 1.2;

const SHAFT_RADIUS: f32 = 0.018;
const ARROW_HEAD_LEN: f32 = 0.18;
const ARROW_HEAD_RADIUS: f32 = 0.06;
/// Per-axis scale plate at the shaft tip: half-size across the face, half-thickness
/// along the axis.
const PLATE_FACE: f32 = 0.06;
const PLATE_THICK: f32 = 0.03;
const RING_MINOR: f32 = 0.03;

/// Planar-scale square: centre offset from origin along each in-plane axis, and
/// its half-size in-plane / half-thickness along the plane normal.
const PLANE_OFFSET: f32 = 0.35;
const PLANE_HALF: f32 = 0.11;
const PLANE_THICK: f32 = 0.02;

/// The three world axes, in index order (0=X, 1=Y, 2=Z).
pub const AXES: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];

fn axis_vec(i: u8) -> Vec3 {
    AXES[i as usize]
}

/// Identifies one handle of the universal manipulator. Axis indices are 0=X,
/// 1=Y, 2=Z; `ScalePlane` stores its two in-plane axes (sorted).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HandleId {
    Translate(u8),
    Rotate(u8),
    ScaleAxis(u8),
    ScalePlane(u8, u8),
}

/// One handle instance, positioned at a gizmo origin. Owns its geometry so
/// drawing and picking share one definition.
#[derive(Clone, Copy)]
pub struct Handle {
    pub id: HandleId,
    pub origin: Vec3,
}

impl Handle {
    /// Every handle of the universal manipulator at `origin`: 3 translate arrows,
    /// 3 rotate rings, 3 per-axis scale handles, 3 planar scale squares.
    pub fn all(origin: Vec3) -> Vec<Handle> {
        let mut handles = Vec::with_capacity(12);
        for a in 0..3u8 {
            handles.push(Handle {
                id: HandleId::Translate(a),
                origin,
            });
            handles.push(Handle {
                id: HandleId::Rotate(a),
                origin,
            });
            handles.push(Handle {
                id: HandleId::ScaleAxis(a),
                origin,
            });
        }
        for (a, b) in [(0u8, 1u8), (1, 2), (0, 2)] {
            handles.push(Handle {
                id: HandleId::ScalePlane(a, b),
                origin,
            });
        }
        handles
    }

    pub fn draw(&self, gizmos: &mut Gizmos<SdfOverlayGizmos>, color: Color) {
        match self.id {
            HandleId::Translate(a) => {
                let axis = axis_vec(a);
                gizmos.arrow(self.origin, self.origin + axis * TRANSLATE_LEN, color);
            }
            HandleId::Rotate(a) => {
                let axis = axis_vec(a);
                let rot = Quat::from_rotation_arc(Vec3::Z, axis);
                gizmos.circle(Isometry3d::new(self.origin, rot), ROTATE_RADIUS, color);
            }
            HandleId::ScaleAxis(a) => {
                let axis = axis_vec(a);
                let tip = self.origin + axis * SCALE_LEN;
                let rot = Quat::from_rotation_arc(Vec3::Z, axis);
                gizmos.line(self.origin, tip, color);
                gizmos.rect(
                    Isometry3d::new(tip, rot),
                    Vec2::splat(PLATE_FACE * 2.0),
                    color,
                );
            }
            HandleId::ScalePlane(a, b) => {
                let (center, rot) = self.plane_isometry(a, b);
                gizmos.rect(
                    Isometry3d::new(center, rot),
                    Vec2::splat(PLANE_HALF * 2.0),
                    color,
                );
            }
        }
    }

    /// Signed distance from `p` to this handle, inflated by `pad` so the thin
    /// visuals stay easy to click. Uses the same constants `draw` does.
    pub fn sdf(&self, p: Vec3, pad: f32) -> f32 {
        let o = self.origin;
        match self.id {
            HandleId::Rotate(a) => sd_torus(p, o, axis_vec(a), ROTATE_RADIUS, RING_MINOR + pad),
            HandleId::Translate(a) => {
                let axis = axis_vec(a);
                let base = o + axis * 0.02;
                let head_base = o + axis * (TRANSLATE_LEN - ARROW_HEAD_LEN);
                let tip = o + axis * TRANSLATE_LEN;
                let shaft = sd_capsule(p, base, head_base, SHAFT_RADIUS + pad);
                let head = sd_cone(p, tip, head_base, ARROW_HEAD_RADIUS + pad);
                shaft.min(head)
            }
            HandleId::ScaleAxis(a) => {
                let axis = axis_vec(a);
                let base = o + axis * 0.02;
                let tip = o + axis * SCALE_LEN;
                let shaft = sd_capsule(p, base, tip, SHAFT_RADIUS + pad);
                let plate = sd_box(
                    p,
                    tip,
                    plate_half_extents(axis, PLATE_FACE + pad, PLATE_THICK + pad),
                );
                shaft.min(plate)
            }
            HandleId::ScalePlane(a, b) => {
                let (center, _) = self.plane_isometry(a, b);
                let normal = AXES[3 - a as usize - b as usize];
                sd_box(
                    p,
                    center,
                    plate_half_extents(normal, PLANE_HALF + pad, PLANE_THICK + pad),
                )
            }
        }
    }

    /// Centre + orientation for the planar-scale square between axes `a`, `b`.
    fn plane_isometry(&self, a: u8, b: u8) -> (Vec3, Quat) {
        let va = axis_vec(a);
        let vb = axis_vec(b);
        let center = self.origin + (va + vb) * PLANE_OFFSET;
        let normal = AXES[3 - a as usize - b as usize];
        (center, Quat::from_rotation_arc(Vec3::Z, normal))
    }

    /// Ray-projection direction used while dragging this handle (the axis the
    /// mouse parameter is measured along).
    pub fn drag_axis(&self) -> Vec3 {
        match self.id {
            HandleId::Translate(a) | HandleId::Rotate(a) | HandleId::ScaleAxis(a) => axis_vec(a),
            HandleId::ScalePlane(a, b) => (axis_vec(a) + axis_vec(b)).normalize(),
        }
    }
}

/// Half-extents for a flat plate whose thin axis is `dir`: `face` on the two
/// other axes, `thick` along `dir`.
fn plate_half_extents(dir: Vec3, face: f32, thick: f32) -> Vec3 {
    if dir.x.abs() > 0.5 {
        Vec3::new(thick, face, face)
    } else if dir.y.abs() > 0.5 {
        Vec3::new(face, thick, face)
    } else {
        Vec3::new(face, face, thick)
    }
}

// --- SDF primitives (shared by every handle) ---

fn sd_capsule(p: Vec3, a: Vec3, b: Vec3, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = (pa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0);
    (pa - ba * h).length() - r
}

fn sd_cone(p: Vec3, tip: Vec3, base: Vec3, base_radius: f32) -> f32 {
    let pa = p - base;
    let ba = tip - base;
    let h = (pa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0);
    let radius_at_h = base_radius * (1.0 - h);
    (pa - ba * h).length() - radius_at_h
}

fn sd_box(p: Vec3, c: Vec3, he: Vec3) -> f32 {
    let q = (p - c).abs() - he;
    q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
}

fn sd_torus(p: Vec3, c: Vec3, axis: Vec3, major: f32, minor: f32) -> f32 {
    let rel = p - c;
    let ax = rel.dot(axis);
    let radial = (rel - axis * ax).length();
    Vec2::new(radial - major, ax).length() - minor
}
