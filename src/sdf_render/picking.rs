use bevy::prelude::*;

use crate::sdf_render::bvh::Bvh;
use crate::sdf_render::edits::{ResolvedEdit, eval_world, fold_csg};
use crate::sdf_render::GatheredEdit;
// `SdfAtlas` + the re-export glob are only needed by the editor-gated `debug_capture_march`.
#[cfg(feature = "editor")]
use crate::sdf_render::atlas::SdfAtlas;
#[cfg(feature = "editor")]
use crate::sdf_render::*;

/// Convert mouse position to a world-space ray
pub fn mouse_to_ray(
    camera: &Camera,
    camera_transform: &Transform,
    window: &Window,
    mouse_pos: Vec2,
) -> Option<Ray> {
    let _viewport_size = camera.physical_viewport_size()?;
    let ndc_x = (2.0 * mouse_pos.x / window.width()) - 1.0;
    let ndc_y = 1.0 - (2.0 * mouse_pos.y / window.height());

    // Match the raymarch shader: reverse-Z near plane is z = 1.0, and the ray
    // must be reconstructed in world space (projection inverse alone gives a
    // view-space point that ignores camera orientation).
    let world_from_view = camera_transform.to_matrix();
    let view_from_clip = camera.clip_from_view().inverse();
    let ndc = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let view_pos = view_from_clip * ndc;
    let view_pos = view_pos.xyz() / view_pos.w;
    let world_pos = world_from_view.transform_point3(view_pos);
    let dir = (world_pos - camera_transform.translation).normalize();

    Some(Ray {
        origin: camera_transform.translation,
        direction: dir,
    })
}

/// Simple ray struct for CPU picking
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

/// An oriented (not axis-aligned) bounding box: a center, the three orthonormal axes of
/// its frame, and the half-extent along each. Used to give node gizmos (lights, empties)
/// a click target that matches the drawn glyph's real orientation + size.
pub struct Obb {
    pub center: Vec3,
    pub axes: [Vec3; 3],
    pub half: Vec3,
}

impl Obb {
    /// Build a world-space OBB from local `(center, half)` bounds carried by the glyph
    /// and the node's world transform (rotation orients the box, scale grows the extents,
    /// translation places the center).
    pub fn from_local(center: Vec3, half: Vec3, xf: &GlobalTransform) -> Self {
        let (scale, rot, translation) = xf.to_scale_rotation_translation();
        Obb {
            center: translation + rot * (center * scale),
            axes: [rot * Vec3::X, rot * Vec3::Y, rot * Vec3::Z],
            half: half * scale,
        }
    }

    /// Ray/OBB intersection via the slab method in the box's local frame. Returns the
    /// entry distance `t` along the (unit-length) ray, or `None` on a miss. A ray
    /// starting inside the box returns `0.0`.
    pub fn ray_hit(&self, ray: &Ray) -> Option<f32> {
        let d = ray.origin - self.center;
        let mut t_min = 0.0_f32;
        let mut t_max = f32::MAX;
        for i in 0..3 {
            let axis = self.axes[i];
            let h = self.half[i];
            let e = axis.dot(d);
            let f = axis.dot(ray.direction);
            if f.abs() > 1e-6 {
                let mut t1 = (-e - h) / f;
                let mut t2 = (-e + h) / f;
                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                }
                t_min = t_min.max(t1);
                t_max = t_max.min(t2);
                if t_min > t_max {
                    return None;
                }
            } else if -e - h > 0.0 || -e + h < 0.0 {
                // Ray parallel to this slab and outside it → miss.
                return None;
            }
        }
        Some(t_min)
    }
}

/// Ray hit against a thin great-circle of `radius` in the plane through `center` with the
/// given `normal` (one of the wireframe-sphere circles). Returns the ray distance `t` to
/// the contact if the ray crosses the plane within `tol` world units of the circle line,
/// else `None`. Picking the drawn line (not the solid disc/sphere) so it doesn't steal
/// clicks from geometry inside the light's range.
pub fn ray_circle(ray: &Ray, center: Vec3, normal: Vec3, radius: f32, tol: f32) -> Option<f32> {
    let denom = ray.direction.dot(normal);
    if denom.abs() < 1e-6 {
        return None; // ray parallel to the circle's plane
    }
    let t = (center - ray.origin).dot(normal) / denom;
    if t < 0.0 {
        return None; // behind the camera
    }
    let hit = ray.origin + ray.direction * t;
    let r = (hit - center).length();
    ((r - radius).abs() <= tol).then_some(t)
}

/// Sphere-trace the CSG scene on the CPU to find the entity owning the nearest
/// hit surface, plus the ray distance `t` to that surface. Edits are culled
/// per-march-step by the BVH ray candidates, then folded via [`fold_csg`] — the
/// same evaluation the bake uses, so picking always matches what is rendered.
/// The returned `t` (world units along the unit-length ray) lets callers depth-sort
/// the SDF hit against other pickables (e.g. light/empty gizmos in front of it).
pub fn pick_entity(bvh: &Bvh, ray: &Ray, edits: &[GatheredEdit]) -> Option<(Entity, f32)> {
    if edits.is_empty() {
        return None;
    }
    let max_dist = 100.0_f32;

    // BVH candidates for this ray (a superset of edits it could touch). For the
    // hit-ownership test we fold only those, then attribute the hit to the nearest
    // individual edit surface at the contact point.
    let mut candidates: Vec<u32> = Vec::new();
    bvh.raycast_candidates(ray.origin, ray.direction, max_dist, &mut candidates);
    if candidates.is_empty() {
        return None;
    }
    let resolved: Vec<ResolvedEdit> = candidates
        .iter()
        .map(|&i| edits[i as usize].edit.clone())
        .collect();

    let mut t = 0.0_f32;
    for _ in 0..256 {
        let pos = ray.origin + ray.direction * t;
        let sample = fold_csg(&resolved, pos);
        if sample.dist < 0.001 {
            // Attribute the hit to the nearest individual edit at this point.
            return nearest_edit(edits, &candidates, pos).map(|e| (e, t));
        }
        t += sample.dist.max(0.001);
        if t > max_dist {
            break;
        }
    }
    None
}

/// Entity whose primitive surface is nearest to `pos` among the candidate edits.
fn nearest_edit(edits: &[GatheredEdit], candidates: &[u32], pos: Vec3) -> Option<Entity> {
    let mut best_d = f32::MAX;
    let mut best = None;
    for &i in candidates {
        let g = &edits[i as usize];
        let d = eval_world(&g.edit.prim, &g.edit.transform, pos).abs();
        if d < best_d {
            best_d = d;
            best = Some(g.entity);
        }
    }
    best
}

/// One recorded step of a CPU raymarch, for the debug ray inspector.
#[derive(Clone, Copy)]
pub struct RayStep {
    pub t: f32,
    pub pos: Vec3,
    pub dist: f32,
    pub brick: IVec3,
    pub in_brick: bool,
}

/// Replay the CSG raymarch on the CPU, recording every step. Mirrors [`pick_entity`]
/// but keeps the per-step trace so the debug inspector can show where the march
/// advanced and which bricks it crossed. No GPU readback. Only the editor's debug ray
/// inspector consumes it, so it's editor-gated (else it's dead in the non-editor build).
#[cfg(feature = "editor")]
pub fn debug_capture_march(
    atlas: &SdfAtlas,
    ray: &Ray,
    edits: &[ResolvedEdit],
    config: &SdfGridConfig,
) -> Vec<RayStep> {
    let mut steps = Vec::new();
    let mut t = 0.0_f32;
    let max_dist = 100.0;

    for _ in 0..256 {
        let pos = ray.origin + ray.direction * t;
        let closest_dist = fold_csg(edits, pos).dist;

        // Informational only (debug ray inspector): does a level-0 brick exist here?
        let brick = config.world_to_brick_lod(pos, 0);
        let in_brick = atlas
            .bricks
            .contains_key(&crate::sdf_render::atlas::BrickKey::new(0, brick));
        steps.push(RayStep {
            t,
            pos,
            dist: closest_dist,
            brick,
            in_brick,
        });

        if closest_dist < 0.001 || t > max_dist {
            break;
        }
        t += closest_dist.max(0.001);
    }

    steps
}
