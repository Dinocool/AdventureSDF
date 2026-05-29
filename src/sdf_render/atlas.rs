use bevy::math::bounding::Aabb3d;
use bevy::prelude::*;
use std::collections::HashMap;

use crate::sdf_render::bvh::Bvh;
use crate::sdf_render::edits::{MATERIAL_SLOTS, ResolvedEdit, argmin_material, fold_csg, material_distances};

/// Number of voxels stored per brick edge (8 samples spanning 7 cells + apron).
pub const BRICK_EDGE: usize = 8;
/// Total voxel samples in one brick.
pub const BRICK_VOXELS: usize = BRICK_EDGE * BRICK_EDGE * BRICK_EDGE; // 512

/// Signed-distance values for one brick, stored as 16-bit snorm. 16 bits keeps
/// the gradient (and thus shading normals) smooth — 8-bit quantization steps
/// are large enough to produce visible normal noise on flat surfaces.
pub type SdfBrick = [i16; BRICK_VOXELS];
/// Per-voxel, per-material distance field for one brick: `MATERIAL_SLOTS` (8)
/// 16-bit-snorm distances per voxel, laid out voxel-major
/// (`voxel * MATERIAL_SLOTS + material`).
pub type MaterialBrick = [i16; BRICK_VOXELS * MATERIAL_SLOTS];

pub type BrickCoord = IVec3;

/// One brick's baked data.
///
/// `dist` is the CSG-combined signed distance the cubic surface solver marches.
///
/// `mat_dist` is a *dense per-material* distance field: for each voxel, the signed
/// distance to the nearest matter of each of the 8 materials. The shader
/// trilinearly interpolates these and takes the per-pixel argmin, so the material
/// boundary is the exact sub-voxel bisector between the two nearest materials —
/// crisp even at `smoothing = 0`, and correct where three+ materials meet. This
/// replaces the old discrete `object_ids` + canonical-pair blend, which snapped
/// the material boundary to voxel centres (the jagged-seam artifact).
#[derive(Clone)]
pub struct PackedBrick {
    pub dist: SdfBrick,
    pub mat_dist: MaterialBrick,
}

/// CPU-side atlas: brick origin -> baked brick, with dirty tracking.
#[derive(Resource)]
pub struct SdfAtlas {
    pub bricks: HashMap<BrickCoord, PackedBrick>,
    pub dirty: bool,
}

impl Default for SdfAtlas {
    fn default() -> Self {
        Self {
            bricks: HashMap::new(),
            dirty: true,
        }
    }
}

impl SdfAtlas {
    /// Mark all bricks dirty (an edit moved or changed).
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Convert a signed distance to 16-bit signed normalized.
    /// Range: [-1.0, 1.0] maps to [-32767, 32767].
    fn dist_to_snorm(d: f32) -> i16 {
        let clamped = d.clamp(-1.0, 1.0);
        (clamped * 32767.0) as i16
    }

    /// Linear voxel index within a brick from local (x, y, z) corner coords.
    fn voxel_index(x: usize, y: usize, z: usize) -> usize {
        z * BRICK_EDGE * BRICK_EDGE + y * BRICK_EDGE + x
    }

    /// Bake a single brick from its culled candidate edits (from the BVH). Each
    /// voxel stores both the CSG-combined distance (`fold_csg`, for the surface
    /// solver) and the dense per-material distance field (`material_distances`,
    /// for the shader's argmin material boundary).
    fn bake_single_brick(
        brick_origin: BrickCoord,
        config: &super::SdfGridConfig,
        edits: &[ResolvedEdit],
    ) -> PackedBrick {
        let mut dist: SdfBrick = [0; BRICK_VOXELS];
        let mut mat_dist: MaterialBrick = [0; BRICK_VOXELS * MATERIAL_SLOTS];
        let grid_origin = config.world_origin();
        let voxel_size = config.voxel_size;

        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                for x in 0..BRICK_EDGE {
                    let idx = Self::voxel_index(x, y, z);

                    let world_pos = grid_origin
                        + Vec3::new(
                            (brick_origin.x + x as i32) as f32 * voxel_size,
                            (brick_origin.y + y as i32) as f32 * voxel_size,
                            (brick_origin.z + z as i32) as f32 * voxel_size,
                        );

                    dist[idx] = Self::dist_to_snorm(fold_csg(edits, world_pos).dist);

                    let slots = material_distances(edits, world_pos);
                    let base = idx * MATERIAL_SLOTS;
                    for (m, &d) in slots.iter().enumerate() {
                        mat_dist[base + m] = Self::dist_to_snorm(d);
                    }
                }
            }
        }

        PackedBrick { dist, mat_dist }
    }

    /// Re-evaluate every edit and rebuild all bricks that overlap them.
    ///
    /// The BVH culls candidate edits per brick, so a brick only folds the edits
    /// whose influence AABB it overlaps (empty space costs nothing). Invoked
    /// whenever the atlas is marked dirty (e.g. an edit moved or changed).
    pub fn full_bake(
        &mut self,
        edits: &[ResolvedEdit],
        edit_aabbs: &[Aabb3d],
        bvh: &Bvh,
        config: &super::SdfGridConfig,
    ) {
        self.bricks.clear();
        self.dirty = false;

        if edits.is_empty() {
            return;
        }

        let stride = config.cell_stride();

        // Bounding box (in voxel coords) of all bricks that need baking, padded
        // so an edit centred anywhere inside its origin brick stays fully covered.
        let mut min_brick = IVec3::splat(i32::MAX);
        let mut max_brick = IVec3::splat(i32::MIN);

        for aabb in edit_aabbs {
            let lo = config.world_to_brick(Vec3::from(aabb.min));
            let hi = config.world_to_brick(Vec3::from(aabb.max));
            for (lo_v, hi_v, min_v, max_v) in [
                (lo.x, hi.x, &mut min_brick.x, &mut max_brick.x),
                (lo.y, hi.y, &mut min_brick.y, &mut max_brick.y),
                (lo.z, hi.z, &mut min_brick.z, &mut max_brick.z),
            ] {
                *min_v = (*min_v).min(lo_v - stride);
                *max_v = (*max_v).max(hi_v + 2 * stride);
            }
        }

        // Clamp to grid bounds.
        min_brick = min_brick.max(IVec3::ZERO);
        max_brick = max_brick.min(IVec3::splat(config.grid_size as i32));

        let brick_world = voxel_size_brick_extent(config);
        let mut candidates: Vec<u32> = Vec::new();

        let step = stride as usize;
        for z in (min_brick.z..max_brick.z).step_by(step) {
            for y in (min_brick.y..max_brick.y).step_by(step) {
                for x in (min_brick.x..max_brick.x).step_by(step) {
                    let coord = IVec3::new(x, y, z);

                    // Query the BVH for edits overlapping this brick's world AABB.
                    let brick_min = config.world_origin()
                        + Vec3::new(
                            x as f32 * config.voxel_size,
                            y as f32 * config.voxel_size,
                            z as f32 * config.voxel_size,
                        );
                    let brick_aabb =
                        Aabb3d::from_min_max(brick_min, brick_min + Vec3::splat(brick_world));
                    bvh.query_aabb(&brick_aabb, &mut candidates);
                    if candidates.is_empty() {
                        continue;
                    }

                    // Build the culled, order-preserving edit slice for this brick.
                    // `candidates` indexes into `edits`, which is already sorted by
                    // SdfOrder; sort the indices to keep that order.
                    candidates.sort_unstable();
                    let culled: Vec<ResolvedEdit> = candidates
                        .iter()
                        .map(|&i| edits[i as usize].clone())
                        .collect();

                    let brick = Self::bake_single_brick(coord, config, &culled);
                    self.bricks.insert(coord, brick);
                }
            }
        }
    }
}

/// World-space edge length of one brick (cells * voxel_size).
fn voxel_size_brick_extent(config: &super::SdfGridConfig) -> f32 {
    config.cell_stride() as f32 * config.voxel_size
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_render::edits::{CsgKind, SdfOp, SdfPrimitive};

    fn resolved(prim: SdfPrimitive, t: Transform, op: SdfOp, id: u8) -> ResolvedEdit {
        ResolvedEdit {
            prim,
            transform: t,
            op,
            material_id: id,
        }
    }

    /// Helper: bake one brick straddling the given edits at the grid origin.
    fn bake_origin_brick(
        config: &super::super::SdfGridConfig,
        edits: &[ResolvedEdit],
    ) -> PackedBrick {
        let origin = config.world_to_brick(Vec3::ZERO);
        SdfAtlas::bake_single_brick(origin, config, edits)
    }

    /// The winning (nearest) material id for voxel `idx`, by argmin over its 8
    /// per-material distance slots — mirrors what the shader computes per pixel.
    fn voxel_material(brick: &PackedBrick, idx: usize) -> u8 {
        let base = idx * MATERIAL_SLOTS;
        let mut best = 0usize;
        for m in 1..MATERIAL_SLOTS {
            if brick.mat_dist[base + m] < brick.mat_dist[base + best] {
                best = m;
            }
        }
        best as u8
    }

    #[test]
    fn snorm_clamps_correctly() {
        assert_eq!(SdfAtlas::dist_to_snorm(-2.0), -32767);
        assert_eq!(SdfAtlas::dist_to_snorm(-1.0), -32767);
        assert_eq!(SdfAtlas::dist_to_snorm(0.0), 0);
        assert_eq!(SdfAtlas::dist_to_snorm(1.0), 32767);
        assert_eq!(SdfAtlas::dist_to_snorm(2.0), 32767);
    }

    #[test]
    fn atlas_defaults() {
        let atlas = SdfAtlas::default();
        assert!(atlas.bricks.is_empty());
        assert!(atlas.dirty);
    }

    /// Two union shapes far apart must resolve to distinct per-voxel materials via
    /// the dense argmin — voxels near shape 0 win material 0, voxels near shape 1
    /// win material 1. Regression guard for the "orange bleed" bug (a whole brick
    /// adopting one nearest-shape id).
    #[test]
    fn materials_are_per_voxel() {
        let config = super::super::SdfGridConfig::default();
        let edits = vec![
            resolved(
                SdfPrimitive::Sphere { radius: 0.2 },
                Transform::IDENTITY,
                SdfOp::default(),
                0,
            ),
            resolved(
                SdfPrimitive::Sphere { radius: 0.2 },
                Transform::from_xyz(0.6, 0.0, 0.0),
                SdfOp::default(),
                1,
            ),
        ];
        let brick = bake_origin_brick(&config, &edits);

        let saw_zero = (0..BRICK_VOXELS).any(|i| voxel_material(&brick, i) == 0);
        let saw_one = (0..BRICK_VOXELS).any(|i| voxel_material(&brick, i) == 1);
        assert!(
            saw_zero && saw_one,
            "brick should resolve both materials, got zero={saw_zero} one={saw_one}"
        );
    }

    /// The per-material distance field must record each material's own surface: at
    /// a point inside shape 0, slot 0 is negative and well below slot 1 (and vice
    /// versa). This is what lets the shader find the exact sub-voxel bisector.
    #[test]
    fn material_slots_track_their_own_surface() {
        use crate::sdf_render::edits::material_distances;
        let edits = vec![
            resolved(
                SdfPrimitive::Sphere { radius: 0.3 },
                Transform::IDENTITY,
                SdfOp::default(),
                0,
            ),
            resolved(
                SdfPrimitive::Sphere { radius: 0.3 },
                Transform::from_xyz(0.5, 0.0, 0.0),
                SdfOp::default(),
                1,
            ),
        ];
        // Deep inside sphere 0.
        let s = material_distances(&edits, Vec3::ZERO);
        assert!(s[0] < 0.0, "inside shape 0, slot 0 must be negative");
        assert!(s[0] < s[1], "slot 0 must be nearer than slot 1 here");
        // Deep inside sphere 1.
        let s = material_distances(&edits, Vec3::new(0.5, 0.0, 0.0));
        assert!(s[1] < 0.0 && s[1] < s[0]);
    }

    /// A subtractor's material id must never win a surface voxel: Subtract edits
    /// write no material slot, so their id stays at the far sentinel and loses the
    /// argmin everywhere.
    #[test]
    fn subtract_writes_no_material() {
        let config = super::super::SdfGridConfig::default();
        let edits = vec![
            resolved(
                SdfPrimitive::Box {
                    half_extents: Vec3::splat(0.3),
                },
                Transform::IDENTITY,
                SdfOp::default(),
                1,
            ),
            resolved(
                SdfPrimitive::Sphere { radius: 0.2 },
                Transform::from_xyz(0.3, 0.3, 0.3),
                SdfOp {
                    kind: CsgKind::Subtract,
                    smoothing: 0.0,
                },
                2,
            ),
        ];
        let brick = bake_origin_brick(&config, &edits);
        assert!(
            (0..BRICK_VOXELS).all(|i| voxel_material(&brick, i) != 2),
            "subtractor id 2 must never win the material argmin"
        );
    }
}
