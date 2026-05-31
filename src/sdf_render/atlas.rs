use bevy::math::bounding::Aabb3d;
use bevy::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::sdf_render::bvh::Bvh;
use crate::sdf_render::edits::{PALETTE_K, Palette};

/// Atlas tiles per texture row. The atlas texture is `ATLAS_TILES_PER_ROW × 64` px wide;
/// its height grows in 8-px tile rows as `high_water().div_ceil(ATLAS_TILES_PER_ROW)`.
/// Single source of truth for the layout — render.rs and the GPU-bake realloc mirror in
/// `bake_scheduler.rs` both read it so the CPU and render world agree on tile→pixel.
pub const ATLAS_TILES_PER_ROW: u32 = 256;

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
    /// The atlas `edit_epoch` this brick was baked under. The bake emit skips re-baking a
    /// resident brick whose `baked_epoch` equals the current `edit_epoch` — its GPU texels are
    /// still valid (the edits it folded haven't changed), so a spilled chunk re-queued over
    /// several frames doesn't re-cull / re-bake the bricks it already baked. Set in
    /// [`SdfAtlas::insert_gpu_brick`]; compared in `emit_gpu_bakes`.
    pub baked_epoch: u64,
}

/// CPU-side atlas topology: brick key (lod + origin) -> palette-only placeholder, plus the
/// dirty-tracking the GPU bake + render extract read. The texels live on the GPU.
#[derive(Resource)]
pub struct SdfAtlas {
    pub bricks: HashMap<BrickKey, PackedBrick>,
    /// Force a re-emit of every resident brick on the next schedule (first bake, or an edit
    /// was added/removed so the whole BVH changed).
    pub rebake_all: bool,
    /// Monotonic counter bumped whenever the baked brick set changes. The render
    /// world compares it against its own last-seen value to skip re-uploading the
    /// atlas on frames where nothing changed (idle = zero GPU atlas work).
    pub generation: u64,
    /// Monotonic counter bumped whenever the GPU chunk lookup / tile-run tables would
    /// differ: a brick enters or exits the resident set, OR a resident brick's palette
    /// changes (the tile-run carries each brick's palette + atlas slot). The render world
    /// memos this and rebuilds the O(bricks) chunk tables + re-uploads the lookup buffers only
    /// when it advances.
    pub topology_generation: u64,
    /// Stable brick→tile mapping (see [`TileAllocator`]). Drives where each brick's
    /// texels live in the atlas texture and survives across bakes so the GPU bake node
    /// targets the right sub-rect.
    pub tiles: TileAllocator,
    /// True when the atlas must grow this frame (the GPU bake never shrinks it). The render
    /// world reads this as the grow signal; currently set indirectly via the tile high-water.
    pub last_bake_was_full: bool,
    /// Tiles whose texels the GPU compute bake fills this frame. The render world reads these
    /// so it knows which tiles the bake node will write; the CPU holds only a palette-only
    /// placeholder for them. Cleared each frame at the start of `schedule_bakes`.
    pub gpu_baked_tiles: HashSet<u32>,
    /// Monotonic edit epoch: bumped whenever the edit set / BVH changes (a moved, added, or
    /// removed edit). A brick baked under epoch E folds the edits as they were at E; if the
    /// epoch is still E when its chunk is re-visited (e.g. a spilled chunk re-queued during a
    /// large object's multi-frame bake), the brick's texels are still valid and the bake emit
    /// skips it. Stored per brick as [`PackedBrick::baked_epoch`].
    pub edit_epoch: u64,
}

impl Default for SdfAtlas {
    fn default() -> Self {
        Self {
            bricks: HashMap::new(),
            rebake_all: true,
            generation: 0,
            topology_generation: 0,
            tiles: TileAllocator::default(),
            last_bake_was_full: false,
            gpu_baked_tiles: HashSet::new(),
            edit_epoch: 0,
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

/// Distance-band clamp in VOXELS for the per-LOD distance field. A LOD-`L` brick stores its
/// signed distance clamped to `±DIST_BAND_VOXELS · voxel_size_at(L)`, so a COARSE brick (big
/// voxels) encodes a LARGE world distance and the sphere-trace takes big steps far from the
/// surface — instead of the old fixed ±1.0-world plateau that capped every LOD's step at ~1u
/// (the 100+-step sky cost). The shader decodes by multiplying the snorm sample by the same
/// `band · voxel_size_at(L)` (see `sample_brick_sdf`). A K-sweep (tests/sdf_march_sim.rs)
/// showed step count plateaus at K=4 — larger buys nothing and costs snorm precision.
pub const DIST_BAND_VOXELS: f32 = 4.0;

/// World-units distance band a LOD-`lod` brick's distance field clamps to.
pub fn dist_band_world(config: &super::SdfGridConfig, lod: u32) -> f32 {
    DIST_BAND_VOXELS * config.voxel_size_at(lod)
}

impl SdfAtlas {
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
    /// 9 sample points for a cheap palette build: the brick's 8 corners + centre. The palette
    /// only needs the ≤K material ids *present* in the brick, and a material that owns any
    /// voxel is essentially always nearest at a corner or the centre too — so this matches a
    /// full per-voxel palette for any brick with ≤K materials (the overwhelming common case),
    /// at a fraction of the `eval_world` cost. Used by the GPU bake job emission, where the
    /// per-frame brick count makes a denser palette build the drag bottleneck.
    pub fn brick_palette_samples(key: BrickKey, voxel_size: f32) -> [Vec3; 9] {
        let e = BRICK_EDGE - 1;
        let c = |x: usize, y: usize, z: usize| Self::voxel_world_pos(key.coord, x, y, z, voxel_size);
        [
            c(0, 0, 0), c(e, 0, 0), c(0, e, 0), c(e, e, 0),
            c(0, 0, e), c(e, 0, e), c(0, e, e), c(e, e, e),
            c(e / 2, e / 2, e / 2),
        ]
    }

    /// BVH-cull the edits overlapping brick `key` into `out` (sorted, preserving
    /// `SdfOrder` since candidates index the already-sorted edit list). Returns `None`
    /// for empty space (no edit reaches the brick — the brick should not exist); on
    /// `Some(())`, `out` holds the candidate edit indices. This is the topology decision
    /// the CPU keeps in GPU bake mode (the per-voxel eval moves to the compute shader).
    pub fn cull_edit_indices(
        key: BrickKey,
        bvh: &Bvh,
        config: &super::SdfGridConfig,
        out: &mut Vec<u32>,
    ) -> Option<()> {
        let brick_world = config.brick_world_size(key.lod);
        let brick_min = config.brick_min_world(key.coord, key.lod);
        let brick_aabb = Aabb3d::from_min_max(brick_min, brick_min + Vec3::splat(brick_world));
        bvh.query_aabb(&brick_aabb, out);
        if out.is_empty() {
            return None;
        }
        out.sort_unstable();
        Some(())
    }

    /// Bump the change counter so the render world re-extracts the atlas next frame.
    pub fn bump_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    /// Insert a *palette-only* placeholder for a brick whose texels the compute bake will
    /// write directly into the atlas. Allocates/keeps the stable tile and records it in
    /// `gpu_baked_tiles` and returns that tile so the caller can build the GPU job. Bumps
    /// `generation` (re-extract) always; bumps `topology_generation` on a new key or palette
    /// change — the chunk tables read only the palette and tile, both present here. The
    /// placeholder's `dist`/`mat_dist` are never read (the GPU owns those texels), so they
    /// stay zero-filled.
    pub fn insert_gpu_brick(&mut self, key: BrickKey, palette: Palette) -> u32 {
        let tile = self.tiles.alloc(key);
        self.gpu_baked_tiles.insert(tile);
        self.generation = self.generation.wrapping_add(1);
        let palette_changed = self.bricks.get(&key).is_some_and(|old| old.palette != palette);
        let placeholder = PackedBrick {
            dist: [0; BRICK_VOXELS],
            mat_dist: [0; BRICK_VOXELS * PALETTE_K],
            palette,
            baked_epoch: self.edit_epoch,
        };
        let is_new = self.bricks.insert(key, placeholder).is_none();
        if is_new || palette_changed {
            self.topology_generation = self.topology_generation.wrapping_add(1);
        }
        tile
    }

    /// Remove the brick at `key` (if present), freeing its tile. Returns whether a brick
    /// was actually removed. The freed tile's texels are harmless once the lookup is
    /// rebuilt (no live entry references them).
    pub fn remove_brick(&mut self, key: &BrickKey) -> bool {
        if self.bricks.remove(key).is_some() {
            self.tiles.release(key);
            // Eviction changes both the resident set and the GPU chunk tables. Bump BOTH
            // generations so the render world re-extracts (it gates on `generation`) and
            // rebuilds the lookup tables (gated on `topology_generation`). Missing the
            // `generation` bump here is what froze the GPU on stale bricks when a frame only
            // evicted (e.g. flying away from the scene) without applying any new bake.
            self.generation = self.generation.wrapping_add(1);
            self.topology_generation = self.topology_generation.wrapping_add(1);
            true
        } else {
            false
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
    use crate::sdf_render::edits::{PALETTE_EMPTY, ResolvedEdit, SdfOp, SdfPrimitive, build_palette};

    fn resolved(prim: SdfPrimitive, t: Transform, id: u16) -> ResolvedEdit {
        ResolvedEdit::new(prim, t, SdfOp::default(), id)
    }

    #[test]
    fn atlas_defaults() {
        let atlas = SdfAtlas::default();
        assert!(atlas.bricks.is_empty());
        assert!(atlas.rebake_all, "fresh atlas must force a first full bake");
        assert!(atlas.gpu_baked_tiles.is_empty());
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

    /// A one-brick camera recenter on a LOD ring exposes only a thin shell — the count
    /// of ENTERED coords must be a face of the window (~R²), never the whole R³ volume.
    /// This is the property that makes incremental recenter cheap (vs a full rebake).
    #[test]
    fn ring_shift_exposes_only_a_shell() {
        let config = super::super::SdfGridConfig::default();
        let stride = config.cell_stride();
        let r = config.ring_bricks as i32;

        let old_origin = IVec3::ZERO;
        let new_origin = IVec3::new(stride, 0, 0);

        let entered = ring_window_coords(&config, new_origin)
            .into_iter()
            .filter(|c| !coord_in_window(&config, *c, old_origin))
            .count();

        let volume = (r * r * r) as usize;
        let face = (r * r) as usize;
        assert_eq!(entered, face, "a 1-brick shift must expose exactly one R² face, not the R³ volume ({volume})");
        assert!(entered < volume, "shell must be far smaller than the full window");
    }

    /// A brick with more than K materials keeps only the K nearest in its palette.
    #[test]
    fn palette_caps_at_k() {
        let edits: Vec<ResolvedEdit> = (0..(PALETTE_K as u16 + 1))
            .map(|i| resolved(SdfPrimitive::Sphere { radius: 0.2 }, Transform::from_xyz(i as f32 * 0.15, 0.0, 0.0), i))
            .collect();
        let palette = build_palette(&edits, &[Vec3::ZERO]);
        let filled = palette.iter().filter(|&&id| id != PALETTE_EMPTY).count();
        assert_eq!(filled, PALETTE_K, "palette must cap at K filled slots");
    }

    /// Sorted palette assigns slot order by ascending global id (stable, neighbour-agnostic).
    #[test]
    fn palette_is_sorted_by_id() {
        let edits = vec![
            resolved(SdfPrimitive::Sphere { radius: 0.3 }, Transform::IDENTITY, 5),
            resolved(SdfPrimitive::Sphere { radius: 0.3 }, Transform::from_xyz(0.5, 0.0, 0.0), 2),
        ];
        let palette = build_palette(&edits, &[Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)]);
        assert_eq!(palette[0], 2);
        assert_eq!(palette[1], 5);
    }

    /// `insert_gpu_brick` allocates a stable tile, records it for the GPU bake, and bumps the
    /// generations so the render world re-extracts + rebuilds the chunk tables.
    #[test]
    fn insert_gpu_brick_allocates_and_bumps() {
        let mut atlas = SdfAtlas::default();
        let key = BrickKey::new(0, IVec3::ZERO);
        let gen0 = atlas.generation;
        let topo0 = atlas.topology_generation;
        let tile = atlas.insert_gpu_brick(key, [0; PALETTE_K]);
        assert_eq!(atlas.tiles.tile(&key), Some(tile));
        assert!(atlas.gpu_baked_tiles.contains(&tile));
        assert!(atlas.bricks.contains_key(&key));
        assert_ne!(atlas.generation, gen0, "new brick must bump the upload generation");
        assert_ne!(atlas.topology_generation, topo0, "new brick must bump the topology generation");
    }

    /// Evicting a brick must bump BOTH `generation` and `topology_generation` so the render
    /// world re-extracts (it early-returns on an unchanged `generation`) and rebuilds the
    /// chunk tables — otherwise a frame that only evicts (flying away) leaves the GPU
    /// rendering just-dropped bricks. A no-op remove must NOT bump.
    #[test]
    fn eviction_bumps_generation_for_gpu_extract() {
        let mut atlas = SdfAtlas::default();
        let key = BrickKey::new(0, IVec3::ZERO);
        atlas.insert_gpu_brick(key, [0; PALETTE_K]);

        let gen_before = atlas.generation;
        let topo_before = atlas.topology_generation;
        assert!(atlas.remove_brick(&key), "brick must actually be removed");
        assert_ne!(atlas.generation, gen_before, "eviction must bump the upload generation");
        assert_ne!(atlas.topology_generation, topo_before, "eviction must bump the topology generation");

        let gen_after = atlas.generation;
        assert!(!atlas.remove_brick(&key), "second remove is a no-op");
        assert_eq!(atlas.generation, gen_after, "no-op remove must not bump the generation");
    }
}
