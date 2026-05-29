use bevy::prelude::*;

use crate::sdf_render::atlas::SdfAtlas;
use crate::sdf_render::bvh::Bvh;
use crate::sdf_render::edits::{ResolvedEdit, eval_world, fold_csg};
use crate::sdf_render::{GatheredEdit, *};

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

/// Sphere-trace the CSG scene on the CPU to find the entity owning the nearest
/// hit surface. Edits are culled per-march-step by the BVH ray candidates, then
/// folded via [`fold_csg`] — the same evaluation the bake uses, so picking always
/// matches what is rendered.
pub fn pick_entity(bvh: &Bvh, ray: &Ray, edits: &[GatheredEdit]) -> Option<Entity> {
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
            return nearest_edit(edits, &candidates, pos);
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
/// advanced and which bricks it crossed. No GPU readback.
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

        let brick = config.world_to_brick(pos);
        let in_brick = atlas.bricks.contains_key(&brick);
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
