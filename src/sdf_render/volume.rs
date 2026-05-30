//! Low-res 3D distance clipmap volume — the empty-space march accelerator (Stage 2).
//!
//! The brick atlas is the near-surface DETAIL layer; empty/sky rays can't sphere-trace
//! against it because empty bricks store no distance. This module adds a dense,
//! always-resident 3D distance texture *clipmap* — nested camera-centred levels (finest
//! near the camera, each coarser level covering 2× the extent) — that empty rays sample
//! with one texel fetch and sphere-trace in BIG steps. When the volume reports "near a
//! surface" the existing brick march takes over for the cubic finish.
//!
//! **The unlock is a per-level distance clamp in VOXEL units** (not the brick atlas's
//! global `SNORM_CLAMP_DIST = 1.0` world unit): a coarse level's big voxels encode large
//! world distances, so far/empty space takes huge steps. Decode = `sample · K · voxel_size`.
//!
//! **Conservative invariant:** every stored voxel is the MINIMUM analytic distance over its
//! cell (`conservative_sample` pattern, mirrored from `atlas.rs`), so the field is a lower
//! bound on the true distance — a sphere-trace step of the stored value can never punch
//! through a surface. Violating this (reporting a distance LARGER than true) makes the
//! march overshoot and leaves holes.
//!
//! Bake runs synchronously (see `recenter_volume`); `bake_level` is `Send` and free of
//! `&mut self` so an async path can slot in later behind `SyncBakeMode`, mirroring
//! `bake_scheduler`.

use bevy::math::bounding::Aabb3d;
use bevy::prelude::*;

use super::bvh::Bvh;
use super::edits::{ResolvedEdit, fold_csg};
use super::{SdfCamera, SdfVolume, VolumeQueryData, gather_sorted_edits};

/// Hard cap on clipmap levels — mirrors `VOLUME_LEVELS` in `render.rs` and bindings.wgsl
/// (the GPU binds exactly this many 3D textures). `VolumeConfig::levels` must be `<=` this.
pub const MAX_VOLUME_LEVELS: usize = 4;

/// Sub-grid resolution per axis for the conservative min sample (matches `atlas::SUBSAMPLES`
/// = the cell corners). Each voxel stores the minimum analytic distance over its cell so
/// the field is a conservative lower bound (see module docs).
pub const VOLUME_SUBSAMPLES: usize = 2;

/// Configuration for the 3D distance clipmap.
#[derive(Resource, Clone, Reflect)]
#[reflect(Resource)]
pub struct VolumeConfig {
    /// Active clipmap level count (`<= MAX_VOLUME_LEVELS`). Finest = level 0.
    pub levels: u32,
    /// Voxels per axis in each level's 3D texture. 128 ⇒ 4 MB/level at R16Snorm.
    pub resolution: u32,
    /// Per-level distance clamp in VOXELS: a voxel stores `±k_voxels · voxel_size` of world
    /// distance. Decode = `sample · k_voxels · voxel_size`. Small (≈4) keeps snorm precision
    /// sub-voxel; the clamp caps far-field step length (fine — it's still conservative).
    pub k_voxels: f32,
    /// Level-0 voxel size (world units). Level `L` uses `base_voxel_size · 2^L`, so level 0
    /// spans `resolution · base_voxel_size` world units and each level covers 2× the prior.
    pub base_voxel_size: f32,
}

impl Default for VolumeConfig {
    fn default() -> Self {
        // 4 levels × 64³ R16Snorm = 4 × 0.5 MB = 2 MB resident. base_voxel_size 0.4 ⇒
        // L0 spans 64·0.4 = 25.6 u, L3 spans 8× = ~205 u. (128³ would be 16 MB; 64³ keeps
        // the first cut cheap — raise resolution once the empty-space win is confirmed.)
        Self {
            levels: MAX_VOLUME_LEVELS as u32,
            resolution: 64,
            k_voxels: 4.0,
            base_voxel_size: 0.4,
        }
    }
}

impl VolumeConfig {
    /// Voxel size (world units) at level `lod`: `base · 2^lod`.
    pub fn level_voxel_size(&self, lod: u32) -> f32 {
        self.base_voxel_size * (1u32 << lod) as f32
    }

    /// World edge length one level spans: `resolution · level_voxel_size`.
    pub fn level_world_size(&self, lod: u32) -> f32 {
        self.resolution as f32 * self.level_voxel_size(lod)
    }

    /// Decode scale for level `lod`: `k_voxels · level_voxel_size`. A snorm sample in
    /// `[-1, 1]` decodes to world distance via `sample · this`.
    pub fn decode_scale(&self, lod: u32) -> f32 {
        self.k_voxels * self.level_voxel_size(lod)
    }
}

/// One resident clipmap level: a dense `resolution³` field of voxel-unit-clamped snorm
/// distances, plus where it sits in the world.
#[derive(Clone)]
pub struct VolumeLevel {
    /// Voxel-lattice min corner on this level's lattice (anchored at world 0). World min
    /// corner = `origin_voxel * voxel_size`. Snapped so the level is camera-centred.
    pub origin_voxel: IVec3,
    /// World units per voxel at this level (`config.level_voxel_size(lod)`).
    pub voxel_size: f32,
    /// Dense `resolution³` R16Snorm distances, z-major (`z*res*res + y*res + x`).
    pub data: Vec<i16>,
    /// True when `data`/`origin` changed since the last GPU upload.
    pub dirty: bool,
}

/// The CPU-side 3D distance clipmap: one `VolumeLevel` per active level + a generation
/// counter the render world watches to decide when to re-upload (mirrors `SdfAtlas`).
#[derive(Resource, Default)]
pub struct VolumeClipmap {
    pub config: VolumeConfig,
    pub levels: Vec<VolumeLevel>,
    /// Bumped whenever any level is re-extracted, so the render world re-uploads.
    pub generation: u64,
}

/// Convert a signed distance to voxel-unit snorm for level `lod`: clamp to `±decode_scale`
/// then map to `[-32767, 32767]`. The inverse of the shader decode `sample · decode_scale`.
pub fn dist_to_snorm_k(d: f32, decode_scale: f32) -> i16 {
    let clamped = (d / decode_scale).clamp(-1.0, 1.0);
    (clamped * 32767.0) as i16
}

/// Camera-centred voxel origin for a level: floor the camera into this level's voxel
/// lattice, then back off half the resolution so the camera sits at the level's centre.
/// Integer-snapped ⇒ stable under sub-voxel camera motion (no popping / re-extract churn).
pub fn snap_origin(camera_pos: Vec3, voxel_size: f32, resolution: u32) -> IVec3 {
    let half = (resolution / 2) as i32;
    let cam_voxel = IVec3::new(
        (camera_pos.x / voxel_size).floor() as i32,
        (camera_pos.y / voxel_size).floor() as i32,
        (camera_pos.z / voxel_size).floor() as i32,
    );
    cam_voxel - IVec3::splat(half)
}

/// World-space centre of voxel `(x,y,z)` in a level whose min corner is `origin_voxel`.
fn voxel_world_pos(origin_voxel: IVec3, x: u32, y: u32, z: u32, voxel_size: f32) -> Vec3 {
    Vec3::new(
        (origin_voxel.x + x as i32) as f32 * voxel_size,
        (origin_voxel.y + y as i32) as f32 * voxel_size,
        (origin_voxel.z + z as i32) as f32 * voxel_size,
    )
}

/// The `±½`-voxel sub-grid offsets (cell corners for `N=2`) used to make a voxel a
/// conservative lower bound — mirrors `atlas::sub_offsets`.
fn sub_offsets() -> [f32; VOLUME_SUBSAMPLES] {
    let mut offs = [0.0; VOLUME_SUBSAMPLES];
    if VOLUME_SUBSAMPLES > 1 {
        for (i, o) in offs.iter_mut().enumerate() {
            *o = i as f32 / (VOLUME_SUBSAMPLES - 1) as f32 - 0.5;
        }
    }
    offs
}

/// Conservative lower bound on the true distance over a voxel's whole cell.
///
/// CRITICAL: the march uses this value directly AS a sphere-trace step, AND `sample_volume`
/// takes the MAX over overlapping levels — so it must NEVER exceed the true distance to the
/// nearest surface anywhere in the cell, or the march overshoots and punches holes. A plain
/// centre (or even min-over-corners) sample is NOT safe: a thin surface passing between the
/// sample points would be missed and the voxel would report "far".
///
/// The CSG field is (at worst) Lipschitz-1 for the box/sphere/etc. primitives here, so the
/// true distance can decrease by at most the distance travelled. From the cell centre the
/// farthest cell point is the half-diagonal `½·√3·voxel_size`, so
/// `centre_distance − ½·√3·voxel_size` is a guaranteed lower bound over the entire cell. We
/// also fold in the corner min (it's cheap and only ever tightens the bound upward toward
/// the true value when the centre estimate is loose), then take the smaller of the two.
fn conservative_sample(edits: &[ResolvedEdit], world_pos: Vec3, voxel_size: f32) -> f32 {
    const HALF_DIAG: f32 = 0.866_025_4; // ½·√3
    let centre = fold_csg(edits, world_pos).dist;
    // Lipschitz lower bound over the cell from the centre sample.
    let lipschitz = centre - HALF_DIAG * voxel_size;
    // Corner min (a sampled lower bound); take the smaller so the result is conservative
    // against both estimates.
    let mut corner_min = centre;
    let offs = sub_offsets();
    for &oz in &offs {
        for &oy in &offs {
            for &ox in &offs {
                let p = world_pos + Vec3::new(ox, oy, oz) * voxel_size;
                corner_min = corner_min.min(fold_csg(edits, p).dist);
            }
        }
    }
    lipschitz.min(corner_min)
}

/// Bake one dense clipmap level from the analytic CSG field. `Send`, no `&mut self`, no
/// shared scratch — so it can run on a task pool later (mirrors `atlas::bake_brick`).
///
/// DENSE: every voxel gets a value (no sparse cull — empty space must report a distance).
/// The BVH is used only to cull which edits to fold (a level far from all geometry folds an
/// empty slice and every voxel reads the clamp ⇒ "far", which is exactly right for sky).
pub fn bake_level(
    origin_voxel: IVec3,
    voxel_size: f32,
    decode_scale: f32,
    resolution: u32,
    edits: &[ResolvedEdit],
    bvh: &Bvh,
) -> Vec<i16> {
    let res = resolution as usize;
    let mut data = vec![0i16; res * res * res];

    // Cull edits to those whose influence overlaps this level's world box, grown by the
    // decode clamp (an edit up to `decode_scale` outside still sets a voxel's value).
    let world_min = Vec3::new(
        origin_voxel.x as f32,
        origin_voxel.y as f32,
        origin_voxel.z as f32,
    ) * voxel_size;
    let world_max = world_min + Vec3::splat(resolution as f32 * voxel_size);
    let pad = Vec3::splat(decode_scale);
    let level_aabb = Aabb3d::from_min_max(world_min - pad, world_max + pad);

    let mut scratch: Vec<u32> = Vec::new();
    bvh.query_aabb(&level_aabb, &mut scratch);
    let culled: Vec<ResolvedEdit> = scratch.iter().map(|&i| edits[i as usize].clone()).collect();

    // Empty region (no edit reaches it) ⇒ everything is "far"; store the positive clamp so
    // the march takes maximum steps through it.
    if culled.is_empty() {
        return vec![32767i16; res * res * res];
    }

    for z in 0..resolution {
        for y in 0..resolution {
            for x in 0..resolution {
                let wpos = voxel_world_pos(origin_voxel, x, y, z, voxel_size);
                let d = conservative_sample(&culled, wpos, voxel_size);
                let idx = (z as usize * res + y as usize) * res + x as usize;
                data[idx] = dist_to_snorm_k(d, decode_scale);
            }
        }
    }
    data
}

/// Per-edit AABB set the volume last baked against, so a moved/added/removed edit forces a
/// re-extract of every level (the volume is global, not per-edit-incremental). Mirrors
/// `bake_scheduler::PrevEditAabbs`'s role for the brick atlas.
#[derive(Resource, Default)]
pub struct VolumePrevEdits {
    count: usize,
    entities: std::collections::HashSet<Entity>,
}

/// Synchronous clipmap recenter + re-extract. Each frame: for every active level, compute
/// its camera-centred snapped origin; if it moved (camera crossed a level cell), or the edit
/// set changed, or the level isn't built yet, re-bake that level densely from the analytic
/// field. Bumps `generation` so the render world re-uploads only on change.
///
/// Re-extracts the WHOLE level on any cross (simplest-correct; a level is at most a few MB).
/// `bake_level` is `Send`, so this can move to `AsyncComputeTaskPool` later — see the plan's
/// Deferred section. Runs every frame in the editor scene (cheap when nothing changed: just
/// the origin compare).
pub fn recenter_volume(
    mut clipmap: ResMut<VolumeClipmap>,
    mut prev: ResMut<VolumePrevEdits>,
    mut bvh: Local<Bvh>,
    mut have_bvh: Local<bool>,
    volumes: Query<VolumeQueryData, With<SdfVolume>>,
    camera: Query<&Transform, (With<SdfCamera>, Without<SdfVolume>)>,
) {
    let camera_pos = camera.iter().next().map(|t| t.translation).unwrap_or(Vec3::ZERO);

    // Detect edit-set changes (add/remove). Any geometry change re-bakes all levels, since a
    // far edit can still set a coarse level's distance.
    let gathered = gather_sorted_edits(&volumes);
    let entities: std::collections::HashSet<Entity> = gathered.iter().map(|g| g.entity).collect();
    let edits_changed =
        !*have_bvh || entities.len() != prev.count || entities.iter().any(|e| !prev.entities.contains(e));

    if edits_changed {
        let resolved: Vec<ResolvedEdit> = gathered.iter().map(|g| g.edit.clone()).collect();
        let aabbs: Vec<Aabb3d> = gathered.iter().map(|g| g.aabb).collect();
        *bvh = Bvh::build(&aabbs);
        *have_bvh = true;
        prev.count = entities.len();
        prev.entities = entities;
        // We need the resolved edits available for baking below; stash via a closure scope.
        recenter_levels(&mut clipmap, &resolved, &bvh, camera_pos, true);
        return;
    }

    // Edits unchanged: only re-extract levels whose camera-centred origin moved. Rebuild the
    // resolved-edit slice once (cheap; the gather already cloned the components).
    let resolved: Vec<ResolvedEdit> = gathered.iter().map(|g| g.edit.clone()).collect();
    recenter_levels(&mut clipmap, &resolved, &bvh, camera_pos, false);
}

/// Re-extract any level whose snapped origin changed (or all of them when `force`). Shared by
/// the edits-changed and camera-moved paths above.
fn recenter_levels(
    clipmap: &mut VolumeClipmap,
    edits: &[ResolvedEdit],
    bvh: &Bvh,
    camera_pos: Vec3,
    force: bool,
) {
    let cfg = clipmap.config.clone();
    let level_count = (cfg.levels as usize).min(MAX_VOLUME_LEVELS);

    // Grow/shrink the level vec to the configured count (first run builds it).
    if clipmap.levels.len() != level_count {
        clipmap.levels.clear();
        clipmap.levels.reserve(level_count);
    }

    let mut any_changed = false;
    for lod in 0..level_count {
        let vs = cfg.level_voxel_size(lod as u32);
        let decode = cfg.decode_scale(lod as u32);
        let new_origin = snap_origin(camera_pos, vs, cfg.resolution);

        let needs = force
            || clipmap.levels.len() <= lod
            || clipmap.levels[lod].origin_voxel != new_origin;
        if !needs {
            continue;
        }

        let data = bake_level(new_origin, vs, decode, cfg.resolution, edits, bvh);
        let level = VolumeLevel {
            origin_voxel: new_origin,
            voxel_size: vs,
            data,
            dirty: true,
        };
        if clipmap.levels.len() <= lod {
            clipmap.levels.push(level);
        } else {
            clipmap.levels[lod] = level;
        }
        any_changed = true;
    }

    if any_changed {
        clipmap.generation = clipmap.generation.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_render::edits::{SdfOp, SdfPrimitive};

    fn sphere(center: Vec3, radius: f32) -> (Vec<ResolvedEdit>, Bvh) {
        let edits = vec![ResolvedEdit {
            prim: SdfPrimitive::Sphere { radius },
            transform: Transform::from_translation(center),
            op: SdfOp::default(),
            material_id: 0,
        }];
        let aabbs: Vec<Aabb3d> = edits
            .iter()
            .map(|e| {
                crate::sdf_render::edits::edit_world_aabb(&e.prim, &e.transform, e.op.smoothing)
            })
            .collect();
        let bvh = Bvh::build(&aabbs);
        (edits, bvh)
    }

    /// Every decoded voxel must be a conservative LOWER BOUND: `decoded <= true centre
    /// distance` over its cell (allowing snorm quantization slack). A voxel reporting a
    /// distance LARGER than true would let the march overshoot and punch a hole.
    #[test]
    fn bake_is_conservative_lower_bound() {
        let cfg = VolumeConfig::default();
        let (edits, bvh) = sphere(Vec3::ZERO, 3.0);

        // Test a coarse level (biggest cells ⇒ where over-estimation would show first).
        for lod in 0..cfg.levels {
            let vs = cfg.level_voxel_size(lod);
            let decode = cfg.decode_scale(lod);
            let origin = snap_origin(Vec3::ZERO, vs, cfg.resolution);
            let data = bake_level(origin, vs, decode, cfg.resolution, &edits, &bvh);

            let res = cfg.resolution;
            // Slack: one snorm step plus the centre-vs-cell-min gap is already <= 0 by
            // construction; allow a snorm quantum for the encode.
            let slack = decode / 32767.0 + 1e-4;
            for z in 0..res {
                for y in 0..res {
                    for x in 0..res {
                        let idx = (z as usize * res as usize + y as usize) * res as usize
                            + x as usize;
                        let decoded = data[idx] as f32 / 32767.0 * decode;
                        let wpos = voxel_world_pos(origin, x, y, z, vs);
                        let truth = fold_csg(&edits, wpos).dist.clamp(-decode, decode);
                        assert!(
                            decoded <= truth + slack,
                            "lod {lod} voxel ({x},{y},{z}): decoded {decoded} must be <= true {truth} (+slack {slack})"
                        );
                    }
                }
            }
        }
    }

    /// The voxel-unit clamp decodes back to the expected world distance: a known distance
    /// inside the clamp range round-trips within one snorm quantum.
    #[test]
    fn decode_roundtrip_within_quantum() {
        let cfg = VolumeConfig::default();
        for lod in 0..cfg.levels {
            let decode = cfg.decode_scale(lod);
            let quantum = decode / 32767.0;
            // A distance at half the clamp range.
            let d = decode * 0.5;
            let encoded = dist_to_snorm_k(d, decode);
            let decoded = encoded as f32 / 32767.0 * decode;
            assert!(
                (decoded - d).abs() <= quantum,
                "lod {lod}: {d} round-tripped to {decoded}, off by more than {quantum}"
            );
        }
        // Beyond the clamp saturates to the positive rail.
        let decode = cfg.decode_scale(0);
        assert_eq!(dist_to_snorm_k(decode * 10.0, decode), 32767);
        assert_eq!(dist_to_snorm_k(-decode * 10.0, decode), -32767);
    }

    /// A sub-voxel camera move must NOT change the snapped origin — otherwise the level
    /// re-extracts every frame (churn) and the surface visibly pops.
    #[test]
    fn origin_snap_is_stable_under_subcell_motion() {
        let cfg = VolumeConfig::default();
        let vs = cfg.level_voxel_size(0);
        // Start at a voxel CENTRE on each axis (n·vs + vs/2) so a small move can't cross a
        // boundary — sitting exactly on a boundary would flip the floor and is not the
        // stability case under test.
        let centre = |n: i32| n as f32 * vs + vs * 0.5;
        let p = Vec3::new(centre(25), centre(-10), centre(17));
        let base = snap_origin(p, vs, cfg.resolution);
        // Move less than HALF a voxel on each axis: still inside the same cell.
        let moved = snap_origin(
            p + Vec3::new(vs * 0.4, -vs * 0.4, vs * 0.3),
            vs,
            cfg.resolution,
        );
        assert_eq!(base, moved, "sub-voxel move must keep the same snapped origin");
        // Crossing a full voxel on +X shifts the origin by exactly one.
        let crossed = snap_origin(p + Vec3::new(vs, 0.0, 0.0), vs, cfg.resolution);
        assert_eq!(crossed, base + IVec3::new(1, 0, 0));
    }
}
