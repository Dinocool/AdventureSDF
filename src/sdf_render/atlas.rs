use bevy::math::bounding::Aabb3d;
use bevy::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::sdf_render::bvh::Bvh;
use crate::sdf_render::edits::{
    PALETTE_K, Palette, ResolvedEdit, build_palette, fold_csg, material_distances,
};

/// Number of voxels stored per brick edge (8 samples spanning 7 cells + apron).
pub const BRICK_EDGE: usize = 8;
/// Total voxel samples in one brick.
pub const BRICK_VOXELS: usize = BRICK_EDGE * BRICK_EDGE * BRICK_EDGE; // 512

/// Signed-distance values for one brick, stored as 16-bit snorm. 16 bits keeps
/// the gradient (and thus shading normals) smooth — 8-bit quantization steps
/// are large enough to produce visible normal noise on flat surfaces.
pub type SdfBrick = [i16; BRICK_VOXELS];
/// Per-voxel, per-palette-slot distance field for one brick: `PALETTE_K` (4)
/// 16-bit-snorm distances per voxel, laid out voxel-major
/// (`voxel * PALETTE_K + slot`). Slot `k` is keyed to `PackedBrick::palette[k]`.
pub type MaterialBrick = [i16; BRICK_VOXELS * PALETTE_K];

pub type BrickCoord = IVec3;

/// Stable brick→atlas-tile mapping with a free-list, so a re-baked brick keeps its
/// atlas tile slot across frames. Without this the tile was the brick's HashMap
/// iteration index — unstable between bakes, which forced a full re-upload. A stable
/// slot is what lets the GPU upload only the tiles that actually changed.
#[derive(Default)]
pub struct TileAllocator {
    tile_of: HashMap<BrickCoord, u32>,
    /// Tiles freed by removed bricks, reused before growing `next` so the atlas
    /// stays densely packed (bounded height).
    free: Vec<u32>,
    /// High-water mark: one past the largest tile index ever handed out.
    next: u32,
}

impl TileAllocator {
    /// The tile a brick currently occupies, if any.
    pub fn tile(&self, coord: &BrickCoord) -> Option<u32> {
        self.tile_of.get(coord).copied()
    }

    /// One past the largest live tile index — i.e. how many tile rows the atlas
    /// texture must currently span (`high_water().div_ceil(tiles_per_row)`).
    pub fn high_water(&self) -> u32 {
        self.next
    }

    /// Assign (or return the existing) tile for `coord`. Reuses a freed slot first.
    fn alloc(&mut self, coord: BrickCoord) -> u32 {
        if let Some(&t) = self.tile_of.get(&coord) {
            return t;
        }
        let t = self.free.pop().unwrap_or_else(|| {
            let t = self.next;
            self.next += 1;
            t
        });
        self.tile_of.insert(coord, t);
        t
    }

    /// Return `coord`'s tile to the free pool (brick removed). The texels are left
    /// stale; no live lookup references them, and the slot is reused on the next
    /// alloc.
    fn release(&mut self, coord: &BrickCoord) {
        if let Some(t) = self.tile_of.remove(coord) {
            self.free.push(t);
        }
    }

    fn clear(&mut self) {
        self.tile_of.clear();
        self.free.clear();
        self.next = 0;
    }
}

/// One brick's baked data.
///
/// `dist` is the CSG-combined signed distance the cubic surface solver marches.
///
/// `mat_dist` is a per-*palette-slot* distance field: for each voxel, the signed
/// distance to the nearest matter of each of the brick's ≤K palette materials. The
/// shader trilinearly interpolates these K slots and argmins them, so the material
/// boundary is the exact sub-voxel bisector between the two nearest materials —
/// crisp even at `smoothing = 0`. Storing only the brick's local palette (not every
/// material in the world) bounds per-pixel cost and VRAM to K regardless of how many
/// materials the world contains.
///
/// `palette` maps each local slot to a global material id (`PALETTE_EMPTY` =
/// unused). It is uniform across the brick, so slot `k` is the same material at all
/// 8 corners of every cell — keeping the trilinear interpolation valid.
#[derive(Clone)]
pub struct PackedBrick {
    pub dist: SdfBrick,
    pub mat_dist: MaterialBrick,
    pub palette: Palette,
}

/// CPU-side atlas: brick origin -> baked brick, with dirty tracking.
#[derive(Resource)]
pub struct SdfAtlas {
    pub bricks: HashMap<BrickCoord, PackedBrick>,
    /// Force a full rebuild of every brick on the next bake (first bake, or an edit
    /// was added/removed so the whole BVH changed). Cleared after `full_bake`.
    pub rebake_all: bool,
    /// Brick coords needing a targeted rebake (an existing edit moved/changed). The
    /// union of each changed edit's old+new AABB. Drained by `bake_incremental`.
    pub dirty_bricks: HashSet<BrickCoord>,
    /// Monotonic counter bumped whenever the baked brick set changes. The render
    /// world compares it against its own last-seen value to skip re-uploading the
    /// atlas on frames where nothing changed (idle = zero GPU atlas work).
    pub generation: u64,
    /// Stable brick→tile mapping (see [`TileAllocator`]). Drives where each brick's
    /// texels live in the atlas texture and survives across bakes so partial uploads
    /// target the right sub-rect.
    pub tiles: TileAllocator,
    /// Tiles whose texels changed in the most recent bake (re-baked or newly
    /// allocated). The render world uploads only these via `write_texture`. Cleared
    /// at the start of each bake; ignored when `last_bake_was_full` (everything is
    /// re-uploaded then).
    pub changed_tiles: HashSet<u32>,
    /// True if the most recent bake was a `full_bake` (everything re-allocated). The
    /// render world treats this as "re-upload all tiles" and rebuilds the texture.
    pub last_bake_was_full: bool,
}

impl Default for SdfAtlas {
    fn default() -> Self {
        Self {
            bricks: HashMap::new(),
            rebake_all: true,
            dirty_bricks: HashSet::new(),
            generation: 0,
            tiles: TileAllocator::default(),
            changed_tiles: HashSet::new(),
            last_bake_was_full: false,
        }
    }
}

/// Max stored signed distance (world units). `dist_to_snorm` clamps to ±this, so an
/// edit can be the nearest surface — and thus must be folded into a brick — for any
/// voxel within this distance of its tight AABB. The dirty/bake footprint
/// ([`bricks_in_aabb`]) expands by this so a moved edit re-bakes EVERY brick it can
/// affect, not just the ones its tight AABB touches. (Was the source of stale
/// "carved hole" texels: a brick 0.7–1.0 units away folded a moving edit but, sitting
/// outside a 1-brick pad, never got re-dirtied when the edit left.)
pub const SNORM_CLAMP_DIST: f32 = 1.0;

impl SdfAtlas {
    /// Convert a signed distance to 16-bit signed normalized.
    /// Range: [-1.0, 1.0] maps to [-32767, 32767].
    fn dist_to_snorm(d: f32) -> i16 {
        let clamped = d.clamp(-SNORM_CLAMP_DIST, SNORM_CLAMP_DIST);
        (clamped * 32767.0) as i16
    }

    /// Linear voxel index within a brick from local (x, y, z) corner coords.
    fn voxel_index(x: usize, y: usize, z: usize) -> usize {
        z * BRICK_EDGE * BRICK_EDGE + y * BRICK_EDGE + x
    }

    /// World position of voxel `(x,y,z)` within the brick at `brick_origin`.
    fn voxel_world_pos(
        brick_origin: BrickCoord,
        x: usize,
        y: usize,
        z: usize,
        grid_origin: Vec3,
        voxel_size: f32,
    ) -> Vec3 {
        grid_origin
            + Vec3::new(
                (brick_origin.x + x as i32) as f32 * voxel_size,
                (brick_origin.y + y as i32) as f32 * voxel_size,
                (brick_origin.z + z as i32) as f32 * voxel_size,
            )
    }

    /// Bake a single brick from its culled candidate edits (from the BVH). First
    /// builds the brick's material palette (the ≤K global ids present), then per
    /// voxel stores the CSG-combined distance (`fold_csg`, for the surface solver)
    /// and the per-palette-slot distance field (`material_distances`, for the
    /// shader's argmin material boundary).
    fn bake_single_brick(
        brick_origin: BrickCoord,
        config: &super::SdfGridConfig,
        edits: &[ResolvedEdit],
    ) -> PackedBrick {
        let mut dist: SdfBrick = [0; BRICK_VOXELS];
        let mut mat_dist: MaterialBrick = [0; BRICK_VOXELS * PALETTE_K];
        let grid_origin = config.world_origin();
        let voxel_size = config.voxel_size;

        // All voxel world positions, reused for the palette build and the bake.
        let mut positions = [Vec3::ZERO; BRICK_VOXELS];
        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                for x in 0..BRICK_EDGE {
                    positions[Self::voxel_index(x, y, z)] =
                        Self::voxel_world_pos(brick_origin, x, y, z, grid_origin, voxel_size);
                }
            }
        }

        // The palette is the ≤K global ids nearest anywhere in this brick. Slot k
        // of `mat_dist` is keyed to `palette[k]` for every voxel (uniform per brick).
        let palette = build_palette(edits, &positions);

        for (idx, &world_pos) in positions.iter().enumerate() {
            dist[idx] = Self::dist_to_snorm(fold_csg(edits, world_pos).dist);

            let slots = material_distances(edits, &palette, world_pos);
            let base = idx * PALETTE_K;
            for (k, &d) in slots.iter().enumerate() {
                mat_dist[base + k] = Self::dist_to_snorm(d);
            }
        }

        PackedBrick {
            dist,
            mat_dist,
            palette,
        }
    }

    /// Bake one brick at `coord` from the edits the BVH says overlap it, or `None`
    /// if no edit reaches it (empty space — the brick should not exist). The culled
    /// edit slice preserves `SdfOrder` (candidates index into the already-sorted
    /// `edits`). Shared by `full_bake` and `bake_incremental` so both produce
    /// byte-identical bricks for the same inputs.
    fn bake_coord(
        coord: BrickCoord,
        edits: &[ResolvedEdit],
        bvh: &Bvh,
        config: &super::SdfGridConfig,
        scratch: &mut Vec<u32>,
    ) -> Option<PackedBrick> {
        let brick_world = voxel_size_brick_extent(config);
        let brick_min = config.world_origin()
            + Vec3::new(
                coord.x as f32 * config.voxel_size,
                coord.y as f32 * config.voxel_size,
                coord.z as f32 * config.voxel_size,
            );
        let brick_aabb = Aabb3d::from_min_max(brick_min, brick_min + Vec3::splat(brick_world));
        bvh.query_aabb(&brick_aabb, scratch);
        if scratch.is_empty() {
            return None;
        }
        scratch.sort_unstable();
        let culled: Vec<ResolvedEdit> = scratch.iter().map(|&i| edits[i as usize].clone()).collect();
        Some(Self::bake_single_brick(coord, config, &culled))
    }

    /// Re-evaluate every edit and rebuild all bricks that overlap them.
    ///
    /// The BVH culls candidate edits per brick, so a brick only folds the edits
    /// whose influence AABB it overlaps (empty space costs nothing). Used on the
    /// first bake and whenever an edit is added or removed (`rebake_all`).
    pub fn full_bake(
        &mut self,
        edits: &[ResolvedEdit],
        edit_aabbs: &[Aabb3d],
        bvh: &Bvh,
        config: &super::SdfGridConfig,
    ) {
        self.bricks.clear();
        self.tiles.clear();
        self.changed_tiles.clear();
        self.last_bake_was_full = true;
        self.rebake_all = false;
        self.dirty_bricks.clear();
        self.generation = self.generation.wrapping_add(1);

        if edits.is_empty() {
            return;
        }

        let mut scratch: Vec<u32> = Vec::new();
        for aabb in edit_aabbs {
            for coord in bricks_in_aabb(config, aabb) {
                if self.bricks.contains_key(&coord) {
                    continue; // already baked via an earlier edit's overlap
                }
                if let Some(brick) = Self::bake_coord(coord, edits, bvh, config, &mut scratch) {
                    self.tiles.alloc(coord);
                    self.bricks.insert(coord, brick);
                }
            }
        }
    }

    /// Rebuild only the bricks in `dirty`, re-folding all edits that overlap each
    /// (so a moved neighbour is handled correctly). A dirty brick that no edit
    /// reaches any more is removed. `dirty` is the union of each changed edit's
    /// old+new AABB → brick coords, so this is correct as long as a changed edit's
    /// former footprint is included (the caller guarantees it via `prev_aabbs`).
    pub fn bake_incremental(
        &mut self,
        dirty: &HashSet<BrickCoord>,
        edits: &[ResolvedEdit],
        bvh: &Bvh,
        config: &super::SdfGridConfig,
    ) {
        if dirty.is_empty() {
            return;
        }
        self.generation = self.generation.wrapping_add(1);
        self.last_bake_was_full = false;
        self.changed_tiles.clear();

        let mut scratch: Vec<u32> = Vec::new();
        for &coord in dirty {
            match Self::bake_coord(coord, edits, bvh, config, &mut scratch) {
                Some(brick) => {
                    // Stable tile: a re-baked brick keeps its slot, a new one gets a
                    // (possibly freed) slot. Either way its texels changed.
                    let tile = self.tiles.alloc(coord);
                    self.changed_tiles.insert(tile);
                    self.bricks.insert(coord, brick);
                }
                None => {
                    // Vacated: free the slot. No live lookup references it after the
                    // lookup buffer is rebuilt, so its stale texels are harmless.
                    self.tiles.release(&coord);
                    self.bricks.remove(&coord);
                }
            }
        }
    }
}

/// World-space edge length of one brick (cells * voxel_size).
fn voxel_size_brick_extent(config: &super::SdfGridConfig) -> f32 {
    config.cell_stride() as f32 * config.voxel_size
}

/// Brick coords (stride-aligned origins) that an edit with tight world `aabb` can
/// affect. The AABB is first grown by [`SNORM_CLAMP_DIST`] — the edit's true bake
/// footprint, since it can be the nearest surface up to that far — then padded by a
/// brick (so an edit centred anywhere in its origin brick stays covered) and clamped
/// to the grid. Using the SAME footprint here as the bake's per-brick BVH cull is
/// what keeps `full_bake` and `bake_incremental` byte-identical: a moved edit
/// re-dirties every brick that folds it, leaving no stale texels behind.
pub fn bricks_in_aabb(config: &super::SdfGridConfig, aabb: &Aabb3d) -> Vec<BrickCoord> {
    let stride = config.cell_stride();
    let reach = Vec3::splat(SNORM_CLAMP_DIST);
    let lo = config.world_to_brick(Vec3::from(aabb.min) - reach);
    let hi = config.world_to_brick(Vec3::from(aabb.max) + reach);

    let min_brick = (lo - IVec3::splat(stride)).max(IVec3::ZERO);
    let max_brick = (hi + IVec3::splat(2 * stride)).min(IVec3::splat(config.grid_size as i32));

    let step = stride as usize;
    let mut coords = Vec::new();
    for z in (min_brick.z..max_brick.z).step_by(step) {
        for y in (min_brick.y..max_brick.y).step_by(step) {
            for x in (min_brick.x..max_brick.x).step_by(step) {
                coords.push(IVec3::new(x, y, z));
            }
        }
    }
    coords
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_render::edits::{CsgKind, SdfOp, SdfPrimitive};

    fn resolved(prim: SdfPrimitive, t: Transform, op: SdfOp, id: u16) -> ResolvedEdit {
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

    /// The winning (nearest) GLOBAL material id for voxel `idx`: argmin over the K
    /// palette-slot distances, then map the local slot through the brick palette —
    /// mirrors what the shader computes per pixel. `PALETTE_EMPTY` if the winning
    /// slot is unused.
    fn voxel_material(brick: &PackedBrick, idx: usize) -> u16 {
        let base = idx * PALETTE_K;
        let mut best = 0usize;
        for k in 1..PALETTE_K {
            if brick.mat_dist[base + k] < brick.mat_dist[base + best] {
                best = k;
            }
        }
        brick.palette[best]
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
        assert!(atlas.rebake_all, "fresh atlas must force a first full bake");
        assert!(atlas.dirty_bricks.is_empty());
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

    /// The per-palette-slot distance field must record each material's own surface.
    /// With a palette of [mat 0 -> slot 0, mat 1 -> slot 1]: inside shape 0, slot 0
    /// is negative and below slot 1, and vice versa. This is what lets the shader
    /// find the exact sub-voxel bisector.
    #[test]
    fn material_slots_track_their_own_surface() {
        use crate::sdf_render::edits::{build_palette, material_distances};
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
        // Sorted palette => slot 0 = material 0, slot 1 = material 1.
        let palette = build_palette(&edits, &[Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)]);
        assert_eq!(palette[0], 0);
        assert_eq!(palette[1], 1);

        // Deep inside sphere 0.
        let s = material_distances(&edits, &palette, Vec3::ZERO);
        assert!(s[0] < 0.0, "inside shape 0, slot 0 must be negative");
        assert!(s[0] < s[1], "slot 0 must be nearer than slot 1 here");
        // Deep inside sphere 1.
        let s = material_distances(&edits, &palette, Vec3::new(0.5, 0.0, 0.0));
        assert!(s[1] < 0.0 && s[1] < s[0]);
    }

    /// A brick with more than K materials keeps only the K nearest in its palette.
    #[test]
    fn palette_caps_at_k() {
        use crate::sdf_render::edits::{PALETTE_EMPTY, build_palette};
        // K+1 = 5 spheres, each a distinct material, all near the origin.
        let edits: Vec<ResolvedEdit> = (0..(PALETTE_K as u16 + 1))
            .map(|i| {
                resolved(
                    SdfPrimitive::Sphere { radius: 0.2 },
                    Transform::from_xyz(i as f32 * 0.15, 0.0, 0.0),
                    SdfOp::default(),
                    i,
                )
            })
            .collect();
        let palette = build_palette(&edits, &[Vec3::ZERO]);
        let filled = palette.iter().filter(|&&id| id != PALETTE_EMPTY).count();
        assert_eq!(filled, PALETTE_K, "palette must cap at K filled slots");
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

    use crate::sdf_render::edits::edit_world_aabb;

    /// Build the AABBs + BVH for a set of edits (mirrors `bake_dirty_bricks`).
    fn build_bvh(edits: &[ResolvedEdit]) -> (Vec<Aabb3d>, Bvh) {
        let aabbs: Vec<Aabb3d> = edits
            .iter()
            .map(|e| edit_world_aabb(&e.prim, &e.transform, e.op.smoothing))
            .collect();
        let bvh = Bvh::build(&aabbs);
        (aabbs, bvh)
    }

    /// Moving one of two distant edits must rebake only the bricks near its old+new
    /// position; the far edit's bricks stay byte-identical. Regression guard for the
    /// incremental-bake path (it must match a from-scratch full bake locally without
    /// touching unrelated bricks).
    #[test]
    fn incremental_bake_leaves_distant_bricks_untouched() {
        let config = super::super::SdfGridConfig::default();
        // Two spheres far apart on X (well over a brick's world extent).
        let far_pos = Transform::from_xyz(8.0, 0.0, 0.0);
        let edits = vec![
            resolved(
                SdfPrimitive::Sphere { radius: 0.3 },
                Transform::IDENTITY,
                SdfOp::default(),
                0,
            ),
            resolved(
                SdfPrimitive::Sphere { radius: 0.3 },
                far_pos,
                SdfOp::default(),
                1,
            ),
        ];

        let mut atlas = SdfAtlas::default();
        let (aabbs, bvh) = build_bvh(&edits);
        atlas.full_bake(&edits, &aabbs, &bvh, &config);
        let gen0 = atlas.generation;

        // Snapshot a brick owned by the far sphere.
        let far_coord = config.world_to_brick(far_pos.translation);
        let far_before = atlas
            .bricks
            .get(&far_coord)
            .expect("far sphere should occupy a brick")
            .dist;

        // Move only the first sphere a little; dirty = union(old, new) of its AABB.
        let mut moved = edits.clone();
        let old_aabb = aabbs[0];
        moved[0].transform = Transform::from_xyz(0.4, 0.0, 0.0);
        let (new_aabbs, new_bvh) = build_bvh(&moved);

        let mut dirty: HashSet<BrickCoord> = HashSet::new();
        dirty.extend(bricks_in_aabb(&config, &old_aabb));
        dirty.extend(bricks_in_aabb(&config, &new_aabbs[0]));
        assert!(
            !dirty.contains(&far_coord),
            "far sphere's brick must not be in the dirty set"
        );

        atlas.bake_incremental(&dirty, &moved, &new_bvh, &config);

        assert_ne!(atlas.generation, gen0, "incremental bake must bump generation");
        let far_after = atlas
            .bricks
            .get(&far_coord)
            .expect("far sphere brick must still exist")
            .dist;
        assert_eq!(
            far_before, far_after,
            "untouched far brick must be byte-identical after incremental bake"
        );
    }

    /// An incremental rebake of a region must produce the same brick a full bake of
    /// the moved scene would — i.e. incremental is not a lossy shortcut.
    #[test]
    fn incremental_matches_full_bake_locally() {
        let config = super::super::SdfGridConfig::default();
        let edits = vec![resolved(
            SdfPrimitive::Sphere { radius: 0.3 },
            Transform::IDENTITY,
            SdfOp::default(),
            0,
        )];

        let mut atlas = SdfAtlas::default();
        let (aabbs, bvh) = build_bvh(&edits);
        atlas.full_bake(&edits, &aabbs, &bvh, &config);

        // Move it, then update via incremental on the old+new union.
        let mut moved = edits.clone();
        moved[0].transform = Transform::from_xyz(0.5, 0.2, -0.1);
        let (new_aabbs, new_bvh) = build_bvh(&moved);
        let mut dirty: HashSet<BrickCoord> = HashSet::new();
        dirty.extend(bricks_in_aabb(&config, &aabbs[0]));
        dirty.extend(bricks_in_aabb(&config, &new_aabbs[0]));
        atlas.bake_incremental(&dirty, &moved, &new_bvh, &config);

        // Reference: full bake of the moved scene from scratch.
        let mut reference = SdfAtlas::default();
        reference.full_bake(&moved, &new_aabbs, &new_bvh, &config);

        // Every brick the reference has, the incremental atlas must match exactly.
        for (coord, ref_brick) in &reference.bricks {
            let inc = atlas
                .bricks
                .get(coord)
                .unwrap_or_else(|| panic!("incremental atlas missing brick {coord:?}"));
            assert_eq!(inc.dist, ref_brick.dist, "dist mismatch at {coord:?}");
            assert_eq!(inc.palette, ref_brick.palette, "palette mismatch at {coord:?}");
        }
        // And it must not have leftover bricks the reference lacks within the dirty
        // region (vacated bricks removed).
        for coord in &dirty {
            if !reference.bricks.contains_key(coord) {
                assert!(
                    !atlas.bricks.contains_key(coord),
                    "stale brick {coord:?} should have been removed"
                );
            }
        }
    }

    /// Simulate a real drag: many small incremental steps, each dirtying only the
    /// moved edit's old∪new footprint (exactly as `bake_dirty_bricks` does). After
    /// EVERY step the live brick set must equal a from-scratch full bake of that
    /// pose. Regression guard for the "gaps appear past certain thresholds" bug —
    /// i.e. a brick that should exist at the new pose never gets into the dirty set.
    #[test]
    fn incremental_drag_matches_full_bake_every_step() {
        let config = super::super::SdfGridConfig::default();
        let mut edits = vec![resolved(
            SdfPrimitive::Sphere { radius: 0.3 },
            Transform::IDENTITY,
            SdfOp::default(),
            0,
        )];

        let mut atlas = SdfAtlas::default();
        let (aabbs, bvh) = build_bvh(&edits);
        atlas.full_bake(&edits, &aabbs, &bvh, &config);
        // prev footprint, as bake_dirty_bricks tracks via PrevEditAabbs.
        let mut prev_aabb = aabbs[0];

        // Drag across several brick widths in small sub-brick steps (0.07 world units
        // ≈ under one voxel at the default 0.1 voxel size, so we cross boundaries
        // gradually — the regime where gaps appeared).
        for step in 1..=40 {
            let x = step as f32 * 0.07;
            edits[0].transform = Transform::from_xyz(x, 0.0, 0.0);
            let (new_aabbs, new_bvh) = build_bvh(&edits);

            let mut dirty: HashSet<BrickCoord> = HashSet::new();
            dirty.extend(bricks_in_aabb(&config, &prev_aabb));
            dirty.extend(bricks_in_aabb(&config, &new_aabbs[0]));
            atlas.bake_incremental(&dirty, &edits, &new_bvh, &config);
            prev_aabb = new_aabbs[0];

            let mut reference = SdfAtlas::default();
            reference.full_bake(&edits, &new_aabbs, &new_bvh, &config);

            // Same set of live brick coords (no missing, no stale).
            let inc_keys: HashSet<_> = atlas.bricks.keys().copied().collect();
            let ref_keys: HashSet<_> = reference.bricks.keys().copied().collect();
            assert_eq!(
                inc_keys, ref_keys,
                "step {step} (x={x}): live brick set diverged from full bake"
            );
            for (coord, ref_brick) in &reference.bricks {
                assert_eq!(
                    atlas.bricks[coord].dist, ref_brick.dist,
                    "step {step}: dist mismatch at {coord:?}"
                );
            }
        }
    }

    /// Drag a sphere PAST a large static box (a "plane") — the scene in the bug
    /// report. After every step the incremental atlas must match a full bake: a
    /// shared brick (plane ∩ sphere-footprint) must keep the plane surface, never
    /// get carved into an empty hole. If incremental diverges here, the CPU bake is
    /// at fault; if it matches, the desync is in the GPU upload.
    #[test]
    fn incremental_drag_preserves_static_neighbor() {
        let config = super::super::SdfGridConfig::default();
        // id 0 = wide thin "plane" box at the origin; id 1 = the dragged sphere,
        // starting to one side and moving across the top of the plane.
        let plane = resolved(
            SdfPrimitive::Box {
                half_extents: Vec3::new(2.0, 0.1, 1.0),
            },
            Transform::IDENTITY,
            SdfOp::default(),
            0,
        );
        let mut edits = vec![
            plane.clone(),
            resolved(
                SdfPrimitive::Sphere { radius: 0.3 },
                Transform::from_xyz(-1.5, 0.3, 0.0),
                SdfOp::default(),
                1,
            ),
        ];

        let mut atlas = SdfAtlas::default();
        let (aabbs, bvh) = build_bvh(&edits);
        atlas.full_bake(&edits, &aabbs, &bvh, &config);
        let mut prev_sphere_aabb = aabbs[1];

        for step in 1..=50 {
            let x = -1.5 + step as f32 * 0.06;
            edits[1].transform = Transform::from_xyz(x, 0.3, 0.0);
            let (new_aabbs, new_bvh) = build_bvh(&edits);

            // Only the sphere changed → dirty = its old∪new footprint.
            let mut dirty: HashSet<BrickCoord> = HashSet::new();
            dirty.extend(bricks_in_aabb(&config, &prev_sphere_aabb));
            dirty.extend(bricks_in_aabb(&config, &new_aabbs[1]));
            atlas.bake_incremental(&dirty, &edits, &new_bvh, &config);
            prev_sphere_aabb = new_aabbs[1];

            let mut reference = SdfAtlas::default();
            reference.full_bake(&edits, &new_aabbs, &new_bvh, &config);

            let inc_keys: HashSet<_> = atlas.bricks.keys().copied().collect();
            let ref_keys: HashSet<_> = reference.bricks.keys().copied().collect();
            assert_eq!(
                inc_keys, ref_keys,
                "step {step} (x={x}): live brick set diverged from full bake"
            );
            for (coord, ref_brick) in &reference.bricks {
                assert_eq!(
                    atlas.bricks[coord].dist, ref_brick.dist,
                    "step {step} (x={x}): dist mismatch at {coord:?} — static neighbor carved?"
                );
            }
        }
    }
}
