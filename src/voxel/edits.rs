//! **Build / destroy voxel editing — the SSOT edit delta + the CPU pick** (Stage 5).
//!
//! A sparse OVERRIDE layer ([`VoxelEdits`]) sits OVER the base scene voxels (the Cornell box now, worldgen
//! later). It is keyed by WORLD VOXEL coordinate (in voxel units, not metres) and is scene-agnostic: a key
//! mapping to a solid [`BlockId`] PLACES that block; a key mapping to [`BlockId::AIR`] REMOVES whatever the
//! base scene had there (digs a hole). An absent key falls through to the base voxel.
//!
//! This is the ONE source of truth the three edit consumers all read, so they can never disagree:
//!   * the brick VOXELIZER — [`apply_edit_overlay`] resolves `base unless overridden` per voxel, so the
//!     re-baked brick (and therefore the packed GPU bricks the renderer traces) reflects the edit;
//!   * the brick RE-PACK — the dirty-brick set ([`dirty_bricks_for_edit`]) names exactly the brick(s) a single
//!     edit touches, INCLUDING the face/edge/corner neighbours when the voxel lies on a brick boundary (the
//!     1-voxel halo each brick stores must see the edited neighbour voxel, or a seam goes stale);
//!   * the CPU PICK — [`pick_voxel`] DDA-marches the SAME overlaid solidity (base ∪ edits) the GPU traces, so
//!     a click hits exactly the voxel the user sees, and returns its world coord + the entry FACE normal.
//!
//! The pick's per-voxel DDA + entry-face rule mirror the in-shader `dda_brick` in `voxel_raytrace.wgsl`
//! (the entry axis is the axis the DDA crossed to step INTO the first solid cell; the outward normal points
//! back along that axis against the ray) — so CPU pick == GPU render by construction.

use bevy::math::{IVec3, Vec3};
use bevy::prelude::Resource;
use rustc_hash::FxHashMap;

use super::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, BrickMap, VOXEL_SIZE, brick_coord_of_voxel, voxel_index};
use super::palette::BlockId;

/// The sparse edit-delta resource — the SSOT override layer over the base scene voxels. Keyed by WORLD VOXEL
/// coordinate (voxel units). A value of [`BlockId::AIR`] means "removed" (dig a hole through the base); any
/// solid value means "placed". An absent key falls through to the base scene voxel. Scene-agnostic: the same
/// delta works for Cornell now and worldgen later (both keyed by world voxel coord).
#[derive(Resource, Clone, Debug, Default)]
pub struct VoxelEdits {
    /// World-voxel-coord → override block. `AIR` = removed; solid = placed; absent = use the base voxel.
    overrides: FxHashMap<IVec3, BlockId>,
    /// Bumped on every mutation so consumers (the packer) can cheaply detect "edits changed" without diffing.
    generation: u64,
}

impl VoxelEdits {
    /// An empty delta (no overrides) — the base scene shows through everywhere.
    pub fn new() -> Self {
        Self::default()
    }

    /// True iff there are no overrides (the base scene is unmodified).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.overrides.is_empty()
    }

    /// Number of overridden voxels (placed + removed).
    #[inline]
    pub fn len(&self) -> usize {
        self.overrides.len()
    }

    /// A monotonic counter bumped on every mutation. The renderer re-packs + bumps the GPU generation when
    /// this changes, so an edit is visible next frame; an unchanged delta does no work.
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The override at `world_voxel`, or `None` if the base scene should show through there.
    #[inline]
    pub fn get(&self, world_voxel: IVec3) -> Option<BlockId> {
        self.overrides.get(&world_voxel).copied()
    }

    /// Resolve the FINAL block at `world_voxel`: the override if one exists, else `base` (the base scene's
    /// voxel). The single per-voxel resolution rule the voxelizer applies — `base unless overridden`.
    #[inline]
    pub fn resolve(&self, world_voxel: IVec3, base: BlockId) -> BlockId {
        self.overrides.get(&world_voxel).copied().unwrap_or(base)
    }

    /// PLACE a solid block at `world_voxel` (overrides whatever the base had). No-op-equivalent if `block` is
    /// AIR (that is a REMOVE — use [`remove`](Self::remove) for clarity, though this still records it). Bumps
    /// the generation. Returns `true` (the delta always changes — an idempotent re-place still bumps so a
    /// re-pack is cheap and always correct).
    pub fn place(&mut self, world_voxel: IVec3, block: BlockId) {
        self.overrides.insert(world_voxel, block);
        self.generation = self.generation.wrapping_add(1);
    }

    /// REMOVE the voxel at `world_voxel` — records an AIR override so the base scene's voxel (if any) is dug
    /// out. Bumps the generation.
    pub fn remove(&mut self, world_voxel: IVec3) {
        self.overrides.insert(world_voxel, BlockId::AIR);
        self.generation = self.generation.wrapping_add(1);
    }

    /// Drop the override at `world_voxel`, reverting it to the base scene voxel (un-edit). Bumps the
    /// generation iff an override was actually present.
    pub fn clear(&mut self, world_voxel: IVec3) {
        if self.overrides.remove(&world_voxel).is_some() {
            self.generation = self.generation.wrapping_add(1);
        }
    }

    /// Iterate `(world_voxel, override_block)` over every override.
    pub fn iter(&self) -> impl Iterator<Item = (IVec3, BlockId)> + '_ {
        self.overrides.iter().map(|(&c, &b)| (c, b))
    }
}

/// Overlay the edit delta onto an already-voxelized [`Brick`] at brick `coord`: for each of the brick's
/// `BRICK_VOXELS` voxels, the final block is the override (if `edits` has one at that world voxel) else the
/// brick's base voxel. Returns a NEW brick (collapsing to the uniform/empty fast paths via [`Brick::from_voxels`]).
///
/// This is the SSOT overlay the voxelizers use so the packed GPU bricks reflect the edits — the renderer, the
/// re-pack, and the pick all consult the same `base unless overridden` rule. A brick with no overlapping edit
/// returns an identical brick (cheap: the override map is consulted per voxel, but the per-brick caller skips
/// bricks with no edits — see [`dirty_bricks_for_edit`] / the streaming re-bake).
pub fn apply_edit_overlay(coord: IVec3, base: &Brick, edits: &VoxelEdits) -> Brick {
    let origin = coord * BRICK_EDGE;
    let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                let wv = origin + IVec3::new(x, y, z);
                voxels[voxel_index(x, y, z)] = edits.resolve(wv, base.get(x, y, z));
            }
        }
    }
    Brick::from_voxels(voxels)
}

/// Apply the edit delta across an ENTIRE base [`BrickMap`], returning a new map with the overrides folded in.
///
/// Two passes so a place into PREVIOUSLY-EMPTY space (a brick the base map never stored) still appears:
/// (1) every base brick is overlaid (edits may carve holes / repaint it); (2) every PLACED override whose
/// brick is NOT in the base map seeds a fresh brick from edits alone.
///
/// Removed-only overrides in empty space are no-ops (digging air). Empty bricks are dropped by `insert`
/// (sparsity holds). This is the static-scene path (Cornell): a full overlay is cheap (a few hundred bricks).
/// The streaming path re-bakes only the DIRTY bricks instead (see the renderer), but shares the per-brick
/// [`apply_edit_overlay`] rule, so both agree.
pub fn apply_edits_to_map(base: &BrickMap, edits: &VoxelEdits) -> BrickMap {
    let mut out = BrickMap::new();
    // Pass 1: overlay every base brick.
    for (&coord, brick) in base.iter() {
        out.insert(coord, apply_edit_overlay(coord, brick, edits));
    }
    // Pass 2: PLACED overrides in bricks the base never stored — seed those bricks from edits alone.
    if !edits.is_empty() {
        let mut touched: FxHashMap<IVec3, ()> = FxHashMap::default();
        for (wv, block) in edits.iter() {
            if block.is_air() {
                continue; // a remove into empty space digs nothing
            }
            let bc = brick_coord_of_voxel(wv);
            if base.get(bc).is_some() || touched.contains_key(&bc) {
                continue; // already produced in pass 1 / a previous iteration
            }
            touched.insert(bc, ());
            // Build this brick purely from the base (all-air here) overlaid with edits.
            let empty = Brick::uniform(BlockId::AIR);
            out.insert(bc, apply_edit_overlay(bc, &empty, edits));
        }
    }
    out
}

/// The set of BRICK coordinates a single edit at `world_voxel` makes dirty: the brick that OWNS the voxel,
/// plus any neighbour brick whose 1-voxel HALO border reads this voxel (i.e. when the voxel sits on a brick
/// face/edge/corner). Up to `2³ = 8` bricks (a corner voxel touches the owner + 7 diagonal neighbours).
///
/// The halo is the seam fix (see [`super::gpu::halo_edge`]): each brick stores its neighbours' boundary
/// voxels, so editing a boundary voxel must re-bake the neighbour bricks too or their halo goes stale (a
/// re-trace from that side would show the old voxel). Computing this from the voxel's position within its
/// brick is the SSOT for "which bricks does this edit invalidate".
pub fn dirty_bricks_for_edit(world_voxel: IVec3) -> Vec<IVec3> {
    let owner = brick_coord_of_voxel(world_voxel);
    let local = world_voxel - owner * BRICK_EDGE; // in [0, BRICK_EDGE) on each axis
    // For each axis, the voxel touches the owner (offset 0) and, if it sits on the low (0) or high
    // (BRICK_EDGE-1) face, the adjacent brick on that side (offset -1 / +1). Take the cartesian product so a
    // corner voxel yields all 8 incident bricks.
    let axis_offsets = |l: i32| -> [i32; 2] {
        if l == 0 {
            [0, -1]
        } else if l == BRICK_EDGE - 1 {
            [0, 1]
        } else {
            [0, 0] // interior on this axis → only the owner (the dup is dropped below)
        }
    };
    let ox = axis_offsets(local.x);
    let oy = axis_offsets(local.y);
    let oz = axis_offsets(local.z);
    let mut out: Vec<IVec3> = Vec::with_capacity(8);
    for &dz in &oz {
        for &dy in &oy {
            for &dx in &ox {
                let bc = owner + IVec3::new(dx, dy, dz);
                if !out.contains(&bc) {
                    out.push(bc);
                }
            }
        }
    }
    out
}

// --- CPU pick (world-space voxel DDA) ----------------------------------------------------------------

/// The result of a CPU pick ray: the FIRST solid voxel hit, in world voxel coordinates, plus the FACE the ray
/// entered through (the outward unit normal, axis-aligned) and the placed block + world-t. Mirrors the
/// shader's committed hit so a click resolves the same voxel the user sees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelHit {
    /// World VOXEL coordinate (voxel units) of the first solid voxel the ray entered.
    pub voxel: IVec3,
    /// Outward unit FACE normal at the entry face (one of ±X/±Y/±Z) — points back along the ray's crossing
    /// axis. Adding this to `voxel` gives the AIR voxel adjacent to the hit face (the PLACE target).
    pub normal: IVec3,
    /// The hit voxel's block id (always solid).
    pub block: BlockId,
    /// World-metre distance along the ray to the entry face.
    pub t: f32,
}

impl VoxelHit {
    /// The AIR voxel adjacent to the hit FACE (the voxel a PLACE drops a block into) = `voxel + normal`.
    #[inline]
    pub fn place_target(&self) -> IVec3 {
        self.voxel + self.normal
    }
}

/// DDA-march a world ray (`origin` + t·`dir`, metres) through the OVERLAID solidity (base [`BrickMap`] ∪
/// [`VoxelEdits`]) to the FIRST solid voxel within `max_dist` metres, returning its world voxel coord + entry
/// face. `None` if the ray reaches `max_dist` through air.
///
/// This is a per-VOXEL 3D-DDA (Amanatides & Woo) in world space at [`VOXEL_SIZE`] granularity — the same
/// grid the shader's brick DDA walks, but flattened across bricks (the pick doesn't need the per-brick AABB
/// acceleration; it just needs to agree on which voxel is solid). Solidity is `edits.resolve(v, base)`, the
/// SSOT overlay, so the pick sees exactly what the packer baked. The entry FACE is the axis the DDA crossed to
/// step into the solid voxel (back along the ray) — mirroring `dda_brick`'s entry-axis rule — so the returned
/// normal is the visible face.
pub fn pick_voxel(
    base: &BrickMap,
    edits: &VoxelEdits,
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
) -> Option<VoxelHit> {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }
    let inv = Vec3::new(safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z));

    // The voxel the ray starts in (floor of world / VOXEL_SIZE). A start strictly inside a solid voxel would
    // immediately hit with no entry face; the camera is in open space (Cornell interior / above terrain), so
    // we seed the entry axis from the dominant ray direction and refine it on the first DDA step.
    let p = origin / VOXEL_SIZE;
    let mut vox = IVec3::new(p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32);

    let step = IVec3::new(sign_i(dir.x), sign_i(dir.y), sign_i(dir.z));
    // World-metre t to the first cell boundary on each axis, and the t to cross one full voxel per axis.
    let next_boundary = Vec3::new(
        boundary_coord(vox.x, step.x) * VOXEL_SIZE,
        boundary_coord(vox.y, step.y) * VOXEL_SIZE,
        boundary_coord(vox.z, step.z) * VOXEL_SIZE,
    );
    let big = f32::INFINITY;
    let mut t_max = Vec3::new(
        axis_t(next_boundary.x, origin.x, inv.x, dir.x, big),
        axis_t(next_boundary.y, origin.y, inv.y, dir.y, big),
        axis_t(next_boundary.z, origin.z, inv.z, dir.z, big),
    );
    let t_delta = Vec3::new(
        if dir.x != 0.0 { (VOXEL_SIZE * inv.x).abs() } else { big },
        if dir.y != 0.0 { (VOXEL_SIZE * inv.y).abs() } else { big },
        if dir.z != 0.0 { (VOXEL_SIZE * inv.z).abs() } else { big },
    );

    // Entry axis (0=x,1=y,2=z) crossed to enter the CURRENT voxel. Seed with the dominant ray axis (the face
    // the ray most head-on meets); refined to the actual crossed axis on each DDA advance.
    let mut entry_axis = dominant_axis(dir);
    let mut t_cur = 0.0f32;

    // Bound the march by max_dist in voxels (plus slack) so a ray into open space terminates.
    let max_steps = ((max_dist / VOXEL_SIZE).ceil() as i32 * 3).max(1);
    for _ in 0..max_steps {
        if t_cur > max_dist {
            return None;
        }
        let base_block = base.voxel_block(vox);
        let block = edits.resolve(vox, base_block);
        if !block.is_air() {
            let mut normal = IVec3::ZERO;
            // Outward face = back along the axis the ray crossed to enter (against the step direction).
            match entry_axis {
                0 => normal.x = -step.x,
                1 => normal.y = -step.y,
                _ => normal.z = -step.z,
            }
            // A degenerate axis (step 0) can't be an entry face — fall back to the dominant non-zero axis.
            if normal == IVec3::ZERO {
                let a = dominant_axis(dir);
                match a {
                    0 => normal.x = -sign_i(dir.x),
                    1 => normal.y = -sign_i(dir.y),
                    _ => normal.z = -sign_i(dir.z),
                }
            }
            return Some(VoxelHit { voxel: vox, normal, block, t: t_cur.max(0.0) });
        }
        // Advance across the smallest-t axis (Amanatides & Woo).
        if t_max.x < t_max.y && t_max.x < t_max.z {
            t_cur = t_max.x;
            t_max.x += t_delta.x;
            vox.x += step.x;
            entry_axis = 0;
        } else if t_max.y < t_max.z {
            t_cur = t_max.y;
            t_max.y += t_delta.y;
            vox.y += step.y;
            entry_axis = 1;
        } else {
            t_cur = t_max.z;
            t_max.z += t_delta.z;
            vox.z += step.z;
            entry_axis = 2;
        }
    }
    None
}

/// `1/x` with a huge finite fallback for `x == 0` (so the per-axis slab math never produces NaN; a zero
/// direction component yields an effectively-infinite boundary t that never wins the DDA minimum).
#[inline]
fn safe_inv(x: f32) -> f32 {
    if x != 0.0 { 1.0 / x } else { f32::INFINITY }
}

/// Integer sign of a float ray component (−1 / 0 / +1) — the DDA step per axis.
#[inline]
fn sign_i(x: f32) -> i32 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

/// The integer voxel-grid coordinate of the NEXT boundary the ray crosses on an axis: the far side of the
/// current voxel when stepping +1, the near side when stepping −1, the current coord when not moving.
#[inline]
fn boundary_coord(vox: i32, step: i32) -> f32 {
    (vox + step.max(0)) as f32
}

/// World-t to reach `boundary` along an axis: `(boundary - origin) / dir`. `big` (no crossing) when the axis
/// direction is zero.
#[inline]
fn axis_t(boundary: f32, origin: f32, inv: f32, dir: f32, big: f32) -> f32 {
    if dir != 0.0 { (boundary - origin) * inv } else { big }
}

/// The axis (0=x,1=y,2=z) the ray travels most head-on (largest |component|) — the fallback entry face for the
/// very first voxel (no boundary crossed yet) and for degenerate-step cases.
#[inline]
fn dominant_axis(dir: Vec3) -> i32 {
    let a = dir.abs();
    if a.x >= a.y && a.x >= a.z {
        0
    } else if a.y >= a.z {
        1
    } else {
        2
    }
}

#[cfg(test)]
mod tests;
