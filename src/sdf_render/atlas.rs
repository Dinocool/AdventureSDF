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

/// A brick's identity in the LOD clipmap: its LOD level plus its stride-aligned origin
/// coord on that level's lattice (anchored at world 0, so coords are signed). Level 0
/// is the base resolution; level `L` has `voxel_size · 2^L`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BrickKey {
    pub lod: u32,
    pub coord: BrickCoord,
}

impl BrickKey {
    pub fn new(lod: u32, coord: BrickCoord) -> Self {
        Self { lod, coord }
    }
}

/// Stable brick→atlas-tile mapping with a free-list, so a re-baked brick keeps its
/// atlas tile slot across frames. Without this the tile was the brick's HashMap
/// iteration index — unstable between bakes, which forced a full re-upload. A stable
/// slot is what lets the GPU upload only the tiles that actually changed.
#[derive(Default)]
pub struct TileAllocator {
    tile_of: HashMap<BrickKey, u32>,
    /// Tiles freed by removed bricks, reused before growing `next` so the atlas
    /// stays densely packed (bounded height).
    free: Vec<u32>,
    /// High-water mark: one past the largest tile index ever handed out.
    next: u32,
}

impl TileAllocator {
    /// The tile a brick currently occupies, if any.
    pub fn tile(&self, key: &BrickKey) -> Option<u32> {
        self.tile_of.get(key).copied()
    }

    /// One past the largest live tile index — i.e. how many tile rows the atlas
    /// texture must currently span (`high_water().div_ceil(tiles_per_row)`).
    pub fn high_water(&self) -> u32 {
        self.next
    }

    /// Assign (or return the existing) tile for `key`. Reuses a freed slot first.
    fn alloc(&mut self, key: BrickKey) -> u32 {
        if let Some(&t) = self.tile_of.get(&key) {
            return t;
        }
        let t = self.free.pop().unwrap_or_else(|| {
            let t = self.next;
            self.next += 1;
            t
        });
        self.tile_of.insert(key, t);
        t
    }

    /// Return `key`'s tile to the free pool (brick removed). The texels are left
    /// stale; no live lookup references them, and the slot is reused on the next
    /// alloc.
    fn release(&mut self, key: &BrickKey) {
        if let Some(t) = self.tile_of.remove(key) {
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

/// CPU-side atlas: brick key (lod + origin) -> baked brick, with dirty tracking.
#[derive(Resource)]
pub struct SdfAtlas {
    pub bricks: HashMap<BrickKey, PackedBrick>,
    /// Force a full rebuild of every brick on the next bake (first bake, or an edit
    /// was added/removed so the whole BVH changed). Cleared after `full_bake`.
    pub rebake_all: bool,
    /// Brick keys needing a targeted rebake (an existing edit moved/changed). The
    /// union of each changed edit's old+new AABB. Drained by `bake_incremental`.
    pub dirty_bricks: HashSet<BrickKey>,
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

    /// World position of voxel `(x,y,z)` within the brick at `brick_origin` (origin
    /// coords on the LOD lattice, anchored at world 0), at voxel size `voxel_size`.
    fn voxel_world_pos(
        brick_origin: BrickCoord,
        x: usize,
        y: usize,
        z: usize,
        voxel_size: f32,
    ) -> Vec3 {
        Vec3::new(
            (brick_origin.x + x as i32) as f32 * voxel_size,
            (brick_origin.y + y as i32) as f32 * voxel_size,
            (brick_origin.z + z as i32) as f32 * voxel_size,
        )
    }

    /// Bake a single brick from its culled candidate edits (from the BVH). First
    /// builds the brick's material palette (the ≤K global ids present), then per voxel
    /// stores the CSG-combined signed distance sampled at the voxel centre (`fold_csg`,
    /// the trilinear field the surface solver marches) and the per-palette-slot distance
    /// field (`material_distances`, for the shader's argmin material boundary). `key`
    /// carries the LOD whose voxel size scales the sample spacing.
    ///
    /// Sampling at the voxel centre (not a min over the cell) keeps the field a true
    /// trilinear SDF — correct shape, no grid-snapped blockiness, and the same surface
    /// position at every LOD (no inter-LOD seam). The trade-off is that a feature thinner
    /// than a voxel can be missed at coarse LOD (its zero-crossing falls between samples);
    /// that sub-voxel detail loss is accepted as the cost of a clean, artifact-free field.
    fn bake_single_brick(
        key: BrickKey,
        config: &super::SdfGridConfig,
        edits: &[ResolvedEdit],
    ) -> PackedBrick {
        let mut dist: SdfBrick = [0; BRICK_VOXELS];
        let mut mat_dist: MaterialBrick = [0; BRICK_VOXELS * PALETTE_K];
        let voxel_size = config.voxel_size_at(key.lod);

        // All voxel world positions, reused for the palette build and the bake.
        let mut positions = [Vec3::ZERO; BRICK_VOXELS];
        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                for x in 0..BRICK_EDGE {
                    positions[Self::voxel_index(x, y, z)] =
                        Self::voxel_world_pos(key.coord, x, y, z, voxel_size);
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
        key: BrickKey,
        edits: &[ResolvedEdit],
        bvh: &Bvh,
        config: &super::SdfGridConfig,
        scratch: &mut Vec<u32>,
    ) -> Option<PackedBrick> {
        let brick_world = config.brick_world_size(key.lod);
        let brick_min = config.brick_min_world(key.coord, key.lod);
        let brick_aabb = Aabb3d::from_min_max(brick_min, brick_min + Vec3::splat(brick_world));
        bvh.query_aabb(&brick_aabb, scratch);
        if scratch.is_empty() {
            return None;
        }
        scratch.sort_unstable();
        let culled: Vec<ResolvedEdit> = scratch.iter().map(|&i| edits[i as usize].clone()).collect();
        Some(Self::bake_single_brick(key, config, &culled))
    }

    /// Public, self-contained bake of one brick — the entry point the async bake tasks
    /// call (no `&mut self`, no shared scratch, so it's `Send` over a snapshot of the
    /// edits, BVH, and config). Returns `None` for empty space (no edit reaches the
    /// brick). Byte-identical to `bake_coord` for the same inputs.
    pub fn bake_brick(
        key: BrickKey,
        edits: &[ResolvedEdit],
        bvh: &Bvh,
        config: &super::SdfGridConfig,
    ) -> Option<PackedBrick> {
        let mut scratch: Vec<u32> = Vec::new();
        Self::bake_coord(key, edits, bvh, config, &mut scratch)
    }

    /// Bump the change counter so the render world re-extracts the atlas next frame.
    pub fn bump_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    /// Insert (or replace) a baked brick at `key`, allocating/keeping its stable atlas
    /// tile and marking that tile changed for the incremental GPU upload. Used by the
    /// async-bake apply path.
    pub fn insert_brick(&mut self, key: BrickKey, brick: PackedBrick) {
        let tile = self.tiles.alloc(key);
        self.changed_tiles.insert(tile);
        self.bricks.insert(key, brick);
    }

    /// Remove the brick at `key` (if present), freeing its tile. Returns whether a brick
    /// was actually removed. The freed tile's texels are harmless once the lookup is
    /// rebuilt (no live entry references them).
    pub fn remove_brick(&mut self, key: &BrickKey) -> bool {
        if self.bricks.remove(key).is_some() {
            self.tiles.release(key);
            true
        } else {
            false
        }
    }

    /// Full clipmap bake: for each LOD ring centred on `camera_pos`, enumerate the
    /// ring's candidate brick coords and bake only the sparse non-empty set (the BVH
    /// cull in `bake_coord` returns `None` for bricks no edit reaches). Coarser rings
    /// reach 2× further per level, so the same `ring_bricks` count nests outward.
    ///
    /// Used on the first bake and whenever an edit is added/removed (`rebake_all`) or
    /// the camera crosses a brick boundary (the ring window shifts).
    pub fn full_bake(
        &mut self,
        edits: &[ResolvedEdit],
        bvh: &Bvh,
        config: &super::SdfGridConfig,
        camera_pos: Vec3,
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
        for key in ring_brick_keys(config, camera_pos) {
            if let Some(brick) = Self::bake_coord(key, edits, bvh, config, &mut scratch) {
                self.tiles.alloc(key);
                self.bricks.insert(key, brick);
            }
        }
    }

    /// Rebuild only the bricks in `dirty`, re-folding all edits that overlap each
    /// (so a moved neighbour is handled correctly). A dirty brick that no edit
    /// reaches any more is removed. `dirty` is the union, over the affected LODs, of
    /// each changed edit's old+new footprint → brick keys, so this is correct as long
    /// as a changed edit's former footprint is included (the caller guarantees it via
    /// `prev_aabbs`).
    pub fn bake_incremental(
        &mut self,
        dirty: &HashSet<BrickKey>,
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
        for &key in dirty {
            match Self::bake_coord(key, edits, bvh, config, &mut scratch) {
                Some(brick) => {
                    // Stable tile: a re-baked brick keeps its slot, a new one gets a
                    // (possibly freed) slot. Either way its texels changed.
                    let tile = self.tiles.alloc(key);
                    self.changed_tiles.insert(tile);
                    self.bricks.insert(key, brick);
                }
                None => {
                    // Vacated: free the slot. No live lookup references it after the
                    // lookup buffer is rebuilt, so its stale texels are harmless.
                    self.tiles.release(&key);
                    self.bricks.remove(&key);
                }
            }
        }
    }
}

/// The stride-aligned brick coords of one LOD ring window whose corner is `origin`:
/// a `ring_bricks³` box on that level's lattice. (LOD-agnostic — coords only; the
/// caller pairs them with a level.)
pub fn ring_window_coords(config: &super::SdfGridConfig, origin: IVec3) -> Vec<BrickCoord> {
    let stride = config.cell_stride();
    let r = config.ring_bricks as i32;
    let mut coords = Vec::with_capacity((r * r * r) as usize);
    for iz in 0..r {
        for iy in 0..r {
            for ix in 0..r {
                coords.push(origin + IVec3::new(ix, iy, iz) * stride);
            }
        }
    }
    coords
}

/// True if `coord` lies inside the `ring_bricks³` window whose corner is `origin` (on
/// the stride lattice). O(1) — used to diff old vs new ring windows on a camera shift.
pub fn coord_in_window(config: &super::SdfGridConfig, coord: IVec3, origin: IVec3) -> bool {
    let stride = config.cell_stride();
    let r = config.ring_bricks as i32;
    let rel = coord - origin;
    rel.x >= 0
        && rel.y >= 0
        && rel.z >= 0
        && rel.x < r * stride
        && rel.y < r * stride
        && rel.z < r * stride
}

/// All candidate brick keys across every LOD ring centred on `camera_pos`. The ring at
/// level `L` is a `ring_bricks³` window of stride-aligned coords on that level's
/// lattice, starting at `config.ring_origin`. These are *candidates*; the per-brick BVH
/// cull decides which actually get baked (sparsity).
pub fn ring_brick_keys(config: &super::SdfGridConfig, camera_pos: Vec3) -> Vec<BrickKey> {
    let mut keys = Vec::new();
    for lod in 0..config.lod_count {
        let origin = config.ring_origin(camera_pos, lod);
        for coord in ring_window_coords(config, origin) {
            keys.push(BrickKey::new(lod, coord));
        }
    }
    keys
}

/// Brick keys (at LOD `lod`) that an edit with tight world `aabb` can affect. The AABB
/// is grown by [`SNORM_CLAMP_DIST`] — the edit's true bake footprint — then padded by a
/// brick (so an edit centred anywhere in its origin brick stays covered). Using the
/// SAME footprint here as the bake's per-brick BVH cull is what keeps the incremental
/// dirty set complete: a moved edit re-dirties every brick that folds it, leaving no
/// stale texels behind. Clamped to the LOD ring so the dirty set never includes bricks
/// outside the resident window.
pub fn bricks_in_aabb_lod(
    config: &super::SdfGridConfig,
    aabb: &Aabb3d,
    lod: u32,
    ring_origin: IVec3,
) -> Vec<BrickKey> {
    let stride = config.cell_stride();
    let r = config.ring_bricks as i32;
    let reach = Vec3::splat(SNORM_CLAMP_DIST);
    let lo = config.world_to_brick_lod(Vec3::from(aabb.min) - reach, lod);
    let hi = config.world_to_brick_lod(Vec3::from(aabb.max) + reach, lod);

    let ring_max = ring_origin + IVec3::splat(r * stride);
    let min_brick = (lo - IVec3::splat(stride)).max(ring_origin);
    let max_brick = (hi + IVec3::splat(2 * stride)).min(ring_max);

    let step = stride as usize;
    let mut keys = Vec::new();
    for z in (min_brick.z..max_brick.z).step_by(step) {
        for y in (min_brick.y..max_brick.y).step_by(step) {
            for x in (min_brick.x..max_brick.x).step_by(step) {
                keys.push(BrickKey::new(lod, IVec3::new(x, y, z)));
            }
        }
    }
    keys
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

    /// Helper: bake one level-0 brick straddling the given edits at the world origin.
    fn bake_origin_brick(
        config: &super::super::SdfGridConfig,
        edits: &[ResolvedEdit],
    ) -> PackedBrick {
        let coord = config.world_to_brick_lod(Vec3::ZERO, 0);
        SdfAtlas::bake_single_brick(BrickKey::new(0, coord), config, edits)
    }

    /// Reproduce the full GPU resolve+sample for a sphere across a grid of world points and
    /// compare the reconstructed brick field against the analytic field. A brick that holds
    /// the wrong region's geometry (the "object baked into two separate bricks" duplication)
    /// shows up as a clamp-range point where the brick sample's sign disagrees with the
    /// analytic sign — printed with its world pos + the brick coord that served it.
    #[test]
    fn baked_field_matches_analytic_across_sphere() {
        let config = super::super::SdfGridConfig::default();
        let lod = 0u32;
        let voxel_size = config.voxel_size_at(lod);
        let stride = config.cell_stride();
        // Reproduce the reported single-cube repro: ONE box far out at all-negative coords
        // (-10.822, -0.339, -5.058), 1.0 half-extents. A single edit means no CSG/cull
        // interaction — so any duplication here is pure coordinate→brick→chunk→tile
        // addressing at a negative, off-origin location.
        let cube_centre = Vec3::new(-10.822, -0.339, -5.058);
        let edits = vec![resolved(
            SdfPrimitive::Box {
                half_extents: Vec3::splat(1.0),
            },
            Transform::from_translation(cube_centre),
            SdfOp::default(),
            0,
        )];
        let (_aabbs, bvh) = build_bvh(&edits);

        // Bake every brick the scene touches, with the LOD rings centred on the cube so it
        // is fully resident (full_bake rings follow the camera position).
        let mut atlas = SdfAtlas::default();
        atlas.full_bake(&edits, &bvh, &config, cube_centre);

        // Build the FULL GPU-side data exactly as the render path does:
        // 1. tile origins from the stable allocator (col_px|row_px<<16), wrapped per row;
        // 2. the chunk lookup + packed tile-run tables (sorted chunks, occupancy popcount);
        // 3. the atlas pixel buffer via the real per-tile blit (pack_tile_texels layout).
        // Then resolve every world point the way the shader does — chunk binary-search →
        // occupancy bit → popcount offset → atlas_base → voxel_pixel — and trilinear-sample
        // the pixel buffer. This exercises the upload-side packing the direct-brick test
        // skips, which is where the multi-edit (cube-dependent) duplication must live.
        const TILES_PER_ROW: u32 = 256;
        let edge = BRICK_EDGE as u32;
        let tile_w = edge * edge; // 64
        let tile_origin = |tile: u32| -> (u32, u32) {
            ((tile % TILES_PER_ROW) * tile_w, (tile / TILES_PER_ROW) * edge)
        };
        let tex_w = TILES_PER_ROW * tile_w;
        let rows = atlas.tiles.high_water().div_ceil(TILES_PER_ROW).max(1);
        let tex_h = rows * edge;

        // Atlas pixel buffer (R16 snorm i16), blitted per the production layout.
        let mut pixels = vec![0i16; (tex_w * tex_h) as usize];
        for (key, brick) in atlas.bricks.iter() {
            let tile = atlas.tiles.tile(key).unwrap();
            let (col_px, row_px) = tile_origin(tile);
            for z in 0..edge {
                for y in 0..edge {
                    for x in 0..edge {
                        let src = (z * edge * edge + y * edge + x) as usize;
                        let u = col_px + y * edge + x; // tile-local: y*EDGE+x across width
                        let v = row_px + z;
                        pixels[(v * tex_w + u) as usize] = brick.dist[src];
                    }
                }
            }
        }

        // Chunk tables with real atlas_base packing.
        let tables = crate::sdf_render::chunk::build_chunk_tables(&atlas, &config, |key| {
            let tile = atlas.tiles.tile(key).unwrap();
            let (col_px, row_px) = tile_origin(tile);
            crate::sdf_render::chunk::BrickTile {
                atlas_base: col_px | (row_px << 16),
                pal01: 0,
                pal23: 0,
            }
        });

        // The shader binary-searches `grid_dims.w` = resident_chunks().len() entries, but
        // the uploaded table is build_chunk_tables().chunks.len(). If those differ the search
        // bound is wrong → reads past/short of the real table → wrong brick. Pin them equal.
        let resident = crate::sdf_render::chunk::resident_chunks(&atlas, &config).len();
        assert_eq!(
            resident,
            tables.chunks.len(),
            "shader chunk-count bound (resident_chunks={resident}) must equal uploaded chunk table len ({})",
            tables.chunks.len()
        );

        // DIRECT TILE-COLLISION CHECK (the reported symptom: two halves share one tile id).
        // Every resolvable brick the scene resolves to must own a UNIQUE atlas_base; if two
        // distinct world bricks map to the same tile, one brick's geometry shows at the
        // other's location. Scan all resident brick coords, resolve each to its atlas_base,
        // and assert no two distinct bricks collide.
        {
            use std::collections::HashMap;
            let mut base_owner: HashMap<u32, BrickKey> = HashMap::new();
            for key in atlas.bricks.keys() {
                // Resolve this brick's atlas_base via the SAME path the shader uses, at the
                // brick's OWN lod (not the fixed scan lod).
                let resolved = {
                    let (ck, li) = crate::sdf_render::chunk::chunk_of(*key, &config);
                    let (kh, kl) = crate::sdf_render::chunk::chunk_gpu_key(ck);
                    tables
                        .chunks
                        .binary_search_by(|c| (c.key_hi, c.key_lo).cmp(&(kh, kl)))
                        .ok()
                        .and_then(|idx| {
                            let chunk = tables.chunks[idx];
                            let occ = (chunk.occ_lo as u64) | ((chunk.occ_hi as u64) << 32);
                            if (occ >> li) & 1 == 0 {
                                return None;
                            }
                            let off = (occ & ((1u64 << li) - 1)).count_ones();
                            Some(tables.tile_run[(chunk.tile_run_base + off) as usize].atlas_base)
                        })
                };
                if let Some(base) = resolved
                    && let Some(other) = base_owner.insert(base, *key)
                {
                    assert!(
                        other == *key,
                        "TILE COLLISION: bricks {key:?} and {other:?} both resolve to atlas_base {base:#x}"
                    );
                }
            }
        }

        // Shader-equivalent resolve: world point → brick coord → chunk key → binary search
        // → occupancy popcount → tile_run[base+off].atlas_base. Returns the packed base.
        let resolve_atlas_base = |p: Vec3| -> Option<u32> {
            let coord = config.world_to_brick_lod(p, lod);
            let (ck, li) = crate::sdf_render::chunk::chunk_of(BrickKey::new(lod, coord), &config);
            let (kh, kl) = crate::sdf_render::chunk::chunk_gpu_key(ck);
            let idx = tables
                .chunks
                .binary_search_by(|c| (c.key_hi, c.key_lo).cmp(&(kh, kl)))
                .ok()?;
            let chunk = tables.chunks[idx];
            let occ = (chunk.occ_lo as u64) | ((chunk.occ_hi as u64) << 32);
            if (occ >> li) & 1 == 0 {
                return None;
            }
            let off = (occ & ((1u64 << li) - 1)).count_ones();
            Some(tables.tile_run[(chunk.tile_run_base + off) as usize].atlas_base)
        };

        // Trilinear-sample the atlas pixel buffer at `p` from a resolved `atlas_base`,
        // mirroring brick.wgsl voxel_pixel + sample_brick_sdf exactly.
        let sample_atlas = |atlas_base: u32, p: Vec3| -> f32 {
            let col_px = (atlas_base & 0xffff) as i32;
            let row_px = (atlas_base >> 16) as i32;
            let e = BRICK_EDGE as i32;
            let load = |lx: i32, ly: i32, lz: i32| -> f32 {
                let cx = lx.clamp(0, e - 1);
                let cy = ly.clamp(0, e - 1);
                let cz = lz.clamp(0, e - 1);
                let u = col_px + cy * e + cx;
                let v = row_px + cz;
                pixels[(v * tex_w as i32 + u) as usize] as f32 / 32767.0 * SNORM_CLAMP_DIST
            };
            let stride_f = stride as f32;
            let voxel_f = p / voxel_size;
            let bo = (voxel_f / stride_f).floor() * stride_f;
            let local_f = voxel_f - bo;
            let i0f = local_f.floor();
            let f = local_f - i0f;
            let i0 = IVec3::new(i0f.x as i32, i0f.y as i32, i0f.z as i32);
            let lp = |a: f32, b: f32, w: f32| a + (b - a) * w;
            let c000 = load(i0.x, i0.y, i0.z);
            let c100 = load(i0.x + 1, i0.y, i0.z);
            let c010 = load(i0.x, i0.y + 1, i0.z);
            let c110 = load(i0.x + 1, i0.y + 1, i0.z);
            let c001 = load(i0.x, i0.y, i0.z + 1);
            let c101 = load(i0.x + 1, i0.y, i0.z + 1);
            let c011 = load(i0.x, i0.y + 1, i0.z + 1);
            let c111 = load(i0.x + 1, i0.y + 1, i0.z + 1);
            let x00 = lp(c000, c100, f.x);
            let x10 = lp(c010, c110, f.x);
            let x01 = lp(c001, c101, f.x);
            let x11 = lp(c011, c111, f.x);
            let y0 = lp(x00, x10, f.y);
            let y1 = lp(x01, x11, f.y);
            lp(y0, y1, f.z)
        };

        let n = 81;
        let span = 2.5_f32; // cover the 1.0-half-extent cube + a margin, centred on it
        let mut mismatches = Vec::new();
        for iz in 0..n {
            for iy in 0..n {
                for ix in 0..n {
                    let p = cube_centre + Vec3::new(
                        (ix as f32 / (n - 1) as f32 - 0.5) * 2.0 * span,
                        (iy as f32 / (n - 1) as f32 - 0.5) * 2.0 * span,
                        (iz as f32 / (n - 1) as f32 - 0.5) * 2.0 * span,
                    );
                    let analytic = fold_csg(&edits, p).dist;
                    if analytic.abs() > SNORM_CLAMP_DIST - voxel_size || analytic.abs() < voxel_size {
                        continue;
                    }
                    match resolve_atlas_base(p) {
                        Some(base) => {
                            let baked = sample_atlas(base, p);
                            if (baked < 0.0) != (analytic < 0.0) {
                                mismatches.push((p, baked, analytic));
                            }
                        }
                        None => {
                            if analytic < 0.0 {
                                mismatches.push((p, f32::NAN, analytic));
                            }
                        }
                    }
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "GPU-resolved baked field sign diverges from analytic at {} point(s) (p, baked, analytic); first few: {:?}",
            mismatches.len(),
            &mismatches[..mismatches.len().min(8)]
        );
    }

    /// Parity check: the GPU computes a brick's chunk key + local slot with WGSL functions
    /// (`abs_chunk_key`, `local_brick_index`) built on `floor_div(a,b) = i32(floor(f32(a)/
    /// f32(b)))` — FLOAT division. The CPU uses Rust integer `div_euclid`/`rem_euclid`
    /// (`chunk_of` + `chunk_gpu_key`). If these disagree for any coord (e.g. large negative
    /// coords where f32 rounding bites, or where f32 can't exactly represent the integer),
    /// the GPU searches a different key/slot than the CPU stored → wrong tile, no CPU test
    /// using `chunk_of` could ever catch it. This ports the WGSL ops verbatim and compares.
    #[test]
    fn wgsl_chunk_math_matches_rust_over_negative_range() {
        let config = super::super::SdfGridConfig::default();
        let s = config.cell_stride();
        let c = crate::sdf_render::chunk::CHUNK_BRICKS;

        // Verbatim port of bindings.wgsl `floor_div`: i32(floor(f32(a)/f32(b))).
        let floor_div = |a: i32, b: i32| -> i32 { (a as f32 / b as f32).floor() as i32 };

        // Verbatim port of `abs_chunk_key` (returns key_hi, key_lo).
        let wgsl_key = |coord: IVec3, lod: u32| -> (u32, u32) {
            let bias = 32768i32;
            let cx = ((floor_div(floor_div(coord.x, s), c) + bias) & 0xffff) as u32;
            let cy = ((floor_div(floor_div(coord.y, s), c) + bias) & 0xffff) as u32;
            let cz = ((floor_div(floor_div(coord.z, s), c) + bias) & 0xffff) as u32;
            ((lod << 16) | cx, (cy << 16) | cz)
        };
        // Verbatim port of `local_brick_index`.
        let wgsl_local = |coord: IVec3| -> u32 {
            let lx = ((floor_div(coord.x, s) % c) + c) % c;
            let ly = ((floor_div(coord.y, s) % c) + c) % c;
            let lz = ((floor_div(coord.z, s) % c) + c) % c;
            (lz * c * c + ly * c + lx) as u32
        };

        // Scan brick coords across a wide negative range covering the failing cube
        // (-10.8,-0.3,-5.1 → brick indices around -108..-50). Use stride-aligned coords.
        let mut mismatches = Vec::new();
        for bz in -180..20 {
            for by in -20..20 {
                for bx in -180..20 {
                    let coord = IVec3::new(bx * s, by * s, bz * s);
                    let key = BrickKey::new(0, coord);
                    let (ck, li) = crate::sdf_render::chunk::chunk_of(key, &config);
                    let (rust_hi, rust_lo) = crate::sdf_render::chunk::chunk_gpu_key(ck);
                    let (g_hi, g_lo) = wgsl_key(coord, 0);
                    let g_li = wgsl_local(coord);
                    if (rust_hi, rust_lo, li) != (g_hi, g_lo, g_li) {
                        mismatches.push((coord, (rust_hi, rust_lo, li), (g_hi, g_lo, g_li)));
                    }
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "WGSL chunk math diverges from Rust at {} coord(s) (coord, rust(hi,lo,li), wgsl(hi,lo,li)); first few: {:?}",
            mismatches.len(),
            &mismatches[..mismatches.len().min(8)]
        );
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

    /// The bake stores the TRUE trilinear field: every baked voxel equals the analytic
    /// `fold_csg` distance sampled at the voxel centre (within snorm quantization + the
    /// ±SNORM_CLAMP_DIST clamp). This is the contract that keeps the field a real SDF —
    /// correct shape, no grid-snapped blockiness, and the same surface position at every
    /// LOD (so no inter-LOD seam). A prior min-over-cell bake violated this (it stored a
    /// lower bound, inflating coarse surfaces); this guards against reintroducing it.
    #[test]
    fn baked_voxel_is_true_centre_distance() {
        let config = super::super::SdfGridConfig::default();
        // A coarse LOD (big voxels) is where any deviation from the true field shows most.
        let lod = 2u32;
        let voxel_size = config.voxel_size_at(lod);
        let edits = vec![resolved(
            SdfPrimitive::Sphere { radius: 0.5 },
            Transform::from_xyz(0.3, 0.1, -0.2),
            SdfOp::default(),
            0,
        )];
        let coord = config.world_to_brick_lod(Vec3::ZERO, lod);
        let brick = SdfAtlas::bake_single_brick(BrickKey::new(lod, coord), &config, &edits);

        let slack = SNORM_CLAMP_DIST / 32767.0;
        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                for x in 0..BRICK_EDGE {
                    let idx = z * BRICK_EDGE * BRICK_EDGE + y * BRICK_EDGE + x;
                    let centre = SdfAtlas::voxel_world_pos(coord, x, y, z, voxel_size);
                    let expected = fold_csg(&edits, centre)
                        .dist
                        .clamp(-SNORM_CLAMP_DIST, SNORM_CLAMP_DIST);
                    let stored = brick.dist[idx] as f32 / 32767.0 * SNORM_CLAMP_DIST;
                    assert!(
                        (stored - expected).abs() <= slack,
                        "voxel ({x},{y},{z}): stored {stored} must equal true centre {expected}"
                    );
                }
            }
        }
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

    /// Build the AABBs + BVH for a set of edits (mirrors `schedule_bakes`).
    fn build_bvh(edits: &[ResolvedEdit]) -> (Vec<Aabb3d>, Bvh) {
        let aabbs: Vec<Aabb3d> = edits
            .iter()
            .map(|e| edit_world_aabb(&e.prim, &e.transform, e.op.smoothing))
            .collect();
        let bvh = Bvh::build(&aabbs);
        (aabbs, bvh)
    }

    /// The incremental dirty set for one changed edit's `aabb`, across every LOD ring
    /// centred on `camera_pos` — mirrors what `schedule_bakes` unions per frame.
    fn dirty_for_aabb(
        config: &super::super::SdfGridConfig,
        aabb: &Aabb3d,
        camera_pos: Vec3,
    ) -> HashSet<BrickKey> {
        let mut dirty = HashSet::new();
        for lod in 0..config.lod_count {
            let origin = config.ring_origin(camera_pos, lod);
            dirty.extend(bricks_in_aabb_lod(config, aabb, lod, origin));
        }
        dirty
    }

    /// Moving one of two distant edits must rebake only the bricks near its old+new
    /// position; the far edit's bricks stay byte-identical. Regression guard for the
    /// incremental-bake path (it must match a from-scratch full bake locally without
    /// touching unrelated bricks).
    #[test]
    fn incremental_bake_leaves_distant_bricks_untouched() {
        // Small ring so two spheres a few units apart both stay resident at LOD 0.
        let config = super::super::SdfGridConfig {
            lod_count: 1,
            ring_bricks: 40,
            ..Default::default()
        };
        let camera = Vec3::ZERO;
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
        atlas.full_bake(&edits, &bvh, &config, camera);
        let gen0 = atlas.generation;

        // Snapshot a brick owned by the far sphere.
        let far_key = BrickKey::new(0, config.world_to_brick_lod(far_pos.translation, 0));
        let far_before = atlas
            .bricks
            .get(&far_key)
            .expect("far sphere should occupy a brick")
            .dist;

        // Move only the first sphere a little; dirty = union(old, new) of its AABB.
        let mut moved = edits.clone();
        let old_aabb = aabbs[0];
        moved[0].transform = Transform::from_xyz(0.4, 0.0, 0.0);
        let (new_aabbs, new_bvh) = build_bvh(&moved);

        let mut dirty = dirty_for_aabb(&config, &old_aabb, camera);
        dirty.extend(dirty_for_aabb(&config, &new_aabbs[0], camera));
        assert!(
            !dirty.contains(&far_key),
            "far sphere's brick must not be in the dirty set"
        );

        atlas.bake_incremental(&dirty, &moved, &new_bvh, &config);

        assert_ne!(atlas.generation, gen0, "incremental bake must bump generation");
        let far_after = atlas
            .bricks
            .get(&far_key)
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
        let camera = Vec3::ZERO;
        let edits = vec![resolved(
            SdfPrimitive::Sphere { radius: 0.3 },
            Transform::IDENTITY,
            SdfOp::default(),
            0,
        )];

        let mut atlas = SdfAtlas::default();
        let (aabbs, bvh) = build_bvh(&edits);
        atlas.full_bake(&edits, &bvh, &config, camera);

        // Move it, then update via incremental on the old+new union.
        let mut moved = edits.clone();
        moved[0].transform = Transform::from_xyz(0.5, 0.2, -0.1);
        let (new_aabbs, new_bvh) = build_bvh(&moved);
        let mut dirty = dirty_for_aabb(&config, &aabbs[0], camera);
        dirty.extend(dirty_for_aabb(&config, &new_aabbs[0], camera));
        atlas.bake_incremental(&dirty, &moved, &new_bvh, &config);

        // Reference: full bake of the moved scene from scratch (same camera/rings).
        let mut reference = SdfAtlas::default();
        reference.full_bake(&moved, &new_bvh, &config, camera);

        // Every brick the reference has, the incremental atlas must match exactly.
        for (key, ref_brick) in &reference.bricks {
            let inc = atlas
                .bricks
                .get(key)
                .unwrap_or_else(|| panic!("incremental atlas missing brick {key:?}"));
            assert_eq!(inc.dist, ref_brick.dist, "dist mismatch at {key:?}");
            assert_eq!(inc.palette, ref_brick.palette, "palette mismatch at {key:?}");
        }
        // And it must not have leftover bricks the reference lacks within the dirty
        // region (vacated bricks removed).
        for key in &dirty {
            if !reference.bricks.contains_key(key) {
                assert!(
                    !atlas.bricks.contains_key(key),
                    "stale brick {key:?} should have been removed"
                );
            }
        }
    }

    /// Simulate a real drag: many small incremental steps, each dirtying only the
    /// moved edit's old∪new footprint (exactly as `schedule_bakes` does). After
    /// EVERY step the live brick set must equal a from-scratch full bake of that
    /// pose. Regression guard for the "gaps appear past certain thresholds" bug —
    /// i.e. a brick that should exist at the new pose never gets into the dirty set.
    #[test]
    fn incremental_drag_matches_full_bake_every_step() {
        let config = super::super::SdfGridConfig::default();
        let camera = Vec3::ZERO;
        let mut edits = vec![resolved(
            SdfPrimitive::Sphere { radius: 0.3 },
            Transform::IDENTITY,
            SdfOp::default(),
            0,
        )];

        let mut atlas = SdfAtlas::default();
        let (aabbs, bvh) = build_bvh(&edits);
        atlas.full_bake(&edits, &bvh, &config, camera);
        // prev footprint, as schedule_bakes tracks via PrevEditAabbs.
        let mut prev_aabb = aabbs[0];

        // Drag across several brick widths in small sub-brick steps (0.07 world units
        // ≈ under one voxel at the default 0.1 voxel size, so we cross boundaries
        // gradually — the regime where gaps appeared).
        for step in 1..=40 {
            let x = step as f32 * 0.07;
            edits[0].transform = Transform::from_xyz(x, 0.0, 0.0);
            let (new_aabbs, new_bvh) = build_bvh(&edits);

            let mut dirty = dirty_for_aabb(&config, &prev_aabb, camera);
            dirty.extend(dirty_for_aabb(&config, &new_aabbs[0], camera));
            atlas.bake_incremental(&dirty, &edits, &new_bvh, &config);
            prev_aabb = new_aabbs[0];

            let mut reference = SdfAtlas::default();
            reference.full_bake(&edits, &new_bvh, &config, camera);

            // Same set of live brick keys (no missing, no stale).
            let inc_keys: HashSet<_> = atlas.bricks.keys().copied().collect();
            let ref_keys: HashSet<_> = reference.bricks.keys().copied().collect();
            assert_eq!(
                inc_keys, ref_keys,
                "step {step} (x={x}): live brick set diverged from full bake"
            );
            for (key, ref_brick) in &reference.bricks {
                assert_eq!(
                    atlas.bricks[key].dist, ref_brick.dist,
                    "step {step}: dist mismatch at {key:?}"
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
        let camera = Vec3::ZERO;
        let (aabbs, bvh) = build_bvh(&edits);
        atlas.full_bake(&edits, &bvh, &config, camera);
        let mut prev_sphere_aabb = aabbs[1];

        for step in 1..=50 {
            let x = -1.5 + step as f32 * 0.06;
            edits[1].transform = Transform::from_xyz(x, 0.3, 0.0);
            let (new_aabbs, new_bvh) = build_bvh(&edits);

            // Only the sphere changed → dirty = its old∪new footprint.
            let mut dirty = dirty_for_aabb(&config, &prev_sphere_aabb, camera);
            dirty.extend(dirty_for_aabb(&config, &new_aabbs[1], camera));
            atlas.bake_incremental(&dirty, &edits, &new_bvh, &config);
            prev_sphere_aabb = new_aabbs[1];

            let mut reference = SdfAtlas::default();
            reference.full_bake(&edits, &new_bvh, &config, camera);

            let inc_keys: HashSet<_> = atlas.bricks.keys().copied().collect();
            let ref_keys: HashSet<_> = reference.bricks.keys().copied().collect();
            assert_eq!(
                inc_keys, ref_keys,
                "step {step} (x={x}): live brick set diverged from full bake"
            );
            for (key, ref_brick) in &reference.bricks {
                assert_eq!(
                    atlas.bricks[key].dist, ref_brick.dist,
                    "step {step} (x={x}): dist mismatch at {key:?} — static neighbor carved?"
                );
            }
        }
    }

    /// A level-1 brick covers exactly 2× the world extent of a level-0 brick (the
    /// clipmap's "twice as coarse / twice the area" property).
    #[test]
    fn lod_doubles_brick_world_size() {
        let config = super::super::SdfGridConfig::default();
        let l0 = config.brick_world_size(0);
        let l1 = config.brick_world_size(1);
        let l2 = config.brick_world_size(2);
        assert!((l1 - 2.0 * l0).abs() < 1e-6, "L1 must be 2× L0");
        assert!((l2 - 4.0 * l0).abs() < 1e-6, "L2 must be 4× L0");
    }

    // (Brick addressing now uses absolute chunk keys — see `super::chunk` tests.)

    /// The sparse cull bakes only bricks an edit actually reaches: a single small
    /// sphere at the origin must occupy only a handful of the ring's candidate bricks,
    /// not the whole `ring_bricks³` window.
    #[test]
    fn ring_bake_is_sparse() {
        let config = super::super::SdfGridConfig {
            lod_count: 1,
            ..Default::default()
        };
        let edits = vec![resolved(
            SdfPrimitive::Sphere { radius: 0.3 },
            Transform::IDENTITY,
            SdfOp::default(),
            0,
        )];
        let mut atlas = SdfAtlas::default();
        let (_aabbs, bvh) = build_bvh(&edits);
        atlas.full_bake(&edits, &bvh, &config, Vec3::ZERO);
        let candidates = (config.ring_bricks * config.ring_bricks * config.ring_bricks) as usize;
        assert!(
            atlas.bricks.len() < candidates,
            "bake must be sparse: {} baked vs {} candidates",
            atlas.bricks.len(),
            candidates
        );
        assert!(!atlas.bricks.is_empty(), "the sphere must bake some bricks");
    }

    /// The async `bake_brick` must produce byte-identical bricks to the synchronous
    /// `full_bake` for the same key (the async path is just a re-host of `bake_coord`).
    #[test]
    fn bake_brick_matches_full_bake() {
        let config = super::super::SdfGridConfig {
            lod_count: 1,
            ..Default::default()
        };
        let edits = vec![resolved(
            SdfPrimitive::Sphere { radius: 0.4 },
            Transform::IDENTITY,
            SdfOp::default(),
            0,
        )];
        let (_aabbs, bvh) = build_bvh(&edits);

        let mut atlas = SdfAtlas::default();
        atlas.full_bake(&edits, &bvh, &config, Vec3::ZERO);

        // Every brick full_bake produced must match a standalone bake_brick of its key.
        for (key, ref_brick) in &atlas.bricks {
            let baked = SdfAtlas::bake_brick(*key, &edits, &bvh, &config)
                .expect("a baked key must rebake non-empty");
            assert_eq!(baked.dist, ref_brick.dist, "dist mismatch at {key:?}");
            assert_eq!(baked.palette, ref_brick.palette, "palette mismatch at {key:?}");
        }
    }

    /// A one-brick camera recenter on a LOD ring exposes only a thin shell — the count
    /// of ENTERED coords must be a face of the window (~R²), never the whole R³ volume.
    /// This is the property that makes incremental recenter cheap (vs the old full bake).
    #[test]
    fn ring_shift_exposes_only_a_shell() {
        let config = super::super::SdfGridConfig::default();
        let stride = config.cell_stride();
        let r = config.ring_bricks as i32;

        // Shift the window by exactly one brick on +X.
        let old_origin = IVec3::ZERO;
        let new_origin = IVec3::new(stride, 0, 0);

        let entered = ring_window_coords(&config, new_origin)
            .into_iter()
            .filter(|c| !coord_in_window(&config, *c, old_origin))
            .count();

        let volume = (r * r * r) as usize;
        let face = (r * r) as usize;
        assert_eq!(
            entered, face,
            "a 1-brick shift must expose exactly one R² face, not the R³ volume ({volume})"
        );
        assert!(entered < volume, "shell must be far smaller than the full window");
    }

    // --- Incremental-recenter convergence -------------------------------------------
    //
    // These model exactly what `schedule_bakes` + `apply_bakes` do to the atlas on a
    // camera move (the ECS systems are thin wrappers over this atlas API), so they pin
    // the core correctness invariant without needing a running App / task pool.

    /// Apply one incremental recenter step to `atlas` for a camera move old→new, baking
    /// entered bricks synchronously and dropping exited ones — the same diff
    /// `schedule_bakes` enqueues and `apply_bakes` applies (eager eviction).
    fn recenter_sync(
        atlas: &mut SdfAtlas,
        config: &super::super::SdfGridConfig,
        edits: &[ResolvedEdit],
        bvh: &Bvh,
        old_cam: Vec3,
        new_cam: Vec3,
    ) {
        for lod in 0..config.lod_count {
            let old_origin = config.ring_origin(old_cam, lod);
            let new_origin = config.ring_origin(new_cam, lod);
            if old_origin == new_origin {
                continue;
            }
            // Entered → bake.
            for coord in ring_window_coords(config, new_origin) {
                if !coord_in_window(config, coord, old_origin) {
                    let key = BrickKey::new(lod, coord);
                    match SdfAtlas::bake_brick(key, edits, bvh, config) {
                        Some(b) => atlas.insert_brick(key, b),
                        None => {
                            atlas.remove_brick(&key);
                        }
                    }
                }
            }
            // Exited → drop.
            for coord in ring_window_coords(config, old_origin) {
                if !coord_in_window(config, coord, new_origin) {
                    atlas.remove_brick(&BrickKey::new(lod, coord));
                }
            }
        }
    }

    /// After an incremental recenter to a new camera position, the resident brick set
    /// must be byte-identical to a from-scratch `full_bake` at that position. This is
    /// the core guarantee that flying the camera never corrupts the atlas.
    #[test]
    fn incremental_recenter_matches_full_bake() {
        let config = super::super::SdfGridConfig {
            lod_count: 3,
            ring_bricks: 6,
            ..Default::default()
        };
        // A terrain-ish row of boxes spread along X so a camera move crosses real
        // surface at several LODs.
        let mut edits = Vec::new();
        for i in -3i32..=3 {
            edits.push(resolved(
                SdfPrimitive::Box {
                    half_extents: Vec3::new(0.4, 0.4, 0.4),
                },
                Transform::from_xyz(i as f32 * 1.5, 0.0, 0.0),
                SdfOp::default(),
                (i.rem_euclid(3)) as u16,
            ));
        }
        let (_aabbs, bvh) = build_bvh(&edits);

        let cam0 = Vec3::ZERO;
        let mut atlas = SdfAtlas::default();
        atlas.full_bake(&edits, &bvh, &config, cam0);

        // Walk the camera across several brick widths in small steps (crosses LOD-0 and
        // LOD-1 boundaries), recentering incrementally each step.
        let mut cam = cam0;
        for step in 1..=12 {
            let next = Vec3::new(step as f32 * 0.35, 0.0, 0.0);
            recenter_sync(&mut atlas, &config, &edits, &bvh, cam, next);
            cam = next;

            let mut reference = SdfAtlas::default();
            reference.full_bake(&edits, &bvh, &config, cam);

            let inc: HashSet<_> = atlas.bricks.keys().copied().collect();
            let refk: HashSet<_> = reference.bricks.keys().copied().collect();
            assert_eq!(
                inc, refk,
                "step {step}: incremental recenter brick set diverged from full bake"
            );
            for (key, rb) in &reference.bricks {
                assert_eq!(
                    atlas.bricks[key].dist, rb.dist,
                    "step {step}: dist mismatch at {key:?}"
                );
            }
        }
    }

    /// Moving the camera far away and back must leave no stale bricks: after returning
    /// to the origin the resident set equals a fresh full_bake there (exited bricks were
    /// truly evicted, not leaked).
    #[test]
    fn recenter_round_trip_leaves_no_stale_bricks() {
        let config = super::super::SdfGridConfig {
            lod_count: 2,
            ring_bricks: 6,
            ..Default::default()
        };
        let edits = vec![resolved(
            SdfPrimitive::Sphere { radius: 0.5 },
            Transform::IDENTITY,
            SdfOp::default(),
            0,
        )];
        let (_aabbs, bvh) = build_bvh(&edits);

        let mut atlas = SdfAtlas::default();
        atlas.full_bake(&edits, &bvh, &config, Vec3::ZERO);

        // Fly far past the sphere (it leaves every ring) then back to the origin.
        recenter_sync(&mut atlas, &config, &edits, &bvh, Vec3::ZERO, Vec3::new(50.0, 0.0, 0.0));
        recenter_sync(
            &mut atlas,
            &config,
            &edits,
            &bvh,
            Vec3::new(50.0, 0.0, 0.0),
            Vec3::ZERO,
        );

        let mut reference = SdfAtlas::default();
        reference.full_bake(&edits, &bvh, &config, Vec3::ZERO);

        let inc: HashSet<_> = atlas.bricks.keys().copied().collect();
        let refk: HashSet<_> = reference.bricks.keys().copied().collect();
        assert_eq!(inc, refk, "round-trip left stale or missing bricks");
    }

    /// While far from all geometry, the atlas must hold zero bricks (the sparse cull +
    /// eviction keep VRAM bounded as the camera roams empty space).
    #[test]
    fn far_from_geometry_evicts_everything() {
        let config = super::super::SdfGridConfig {
            lod_count: 2,
            ring_bricks: 6,
            ..Default::default()
        };
        let edits = vec![resolved(
            SdfPrimitive::Sphere { radius: 0.5 },
            Transform::IDENTITY,
            SdfOp::default(),
            0,
        )];
        let (_aabbs, bvh) = build_bvh(&edits);

        let mut atlas = SdfAtlas::default();
        atlas.full_bake(&edits, &bvh, &config, Vec3::ZERO);
        assert!(!atlas.bricks.is_empty(), "sphere bakes some bricks at origin");

        recenter_sync(
            &mut atlas,
            &config,
            &edits,
            &bvh,
            Vec3::ZERO,
            Vec3::new(200.0, 0.0, 0.0),
        );
        assert!(
            atlas.bricks.is_empty(),
            "no geometry near the camera → all bricks evicted, got {}",
            atlas.bricks.len()
        );
    }
}
