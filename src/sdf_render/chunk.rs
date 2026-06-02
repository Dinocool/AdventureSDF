//! Chunk addressing for the clipmap atlas.
//!
//! A **chunk** groups `CHUNK_BRICKS³ = 64` bricks into the clipmap's addressing,
//! bake-batch, and debug unit. Brick GPU lookup is done per *chunk* (a ~64× smaller
//! table than per-brick) keyed by an **absolute** world-lattice chunk coord that never
//! references the camera/ring origin — so the CPU-built lookup and the GPU shader agree
//! by construction, regardless of where the camera is. This is what fixes the
//! "objects shift / world disappears" bugs: those came from per-brick ids computed
//! relative to a camera-moving ring origin.
//!
//! Within a chunk, bricks are **sparse**: only non-empty bricks get atlas tiles. A
//! 64-bit occupancy mask records which of the 64 local slots are present; the GPU tests
//! one bit and `countOneBits` gives the offset into that chunk's packed tile run.
//!
//! A LOD-`L` chunk holds the same 64 bricks as a LOD-0 chunk but each brick is `2^L`
//! larger, so it covers `2^L`× the world — the nested-shell clipmap structure.
//!
//! This module owns ONLY the coordinate math + table layout (pure, unit-tested). The
//! per-brick texel storage (`atlas::TileAllocator`) and incremental upload are unchanged.

use std::collections::{BTreeSet, HashMap};

use bevy::math::IVec3;

use super::SdfGridConfig;
use super::atlas::{ATLAS_TILES_PER_ROW, BRICK_EDGE, BrickKey, SdfAtlas};
use super::edits::Palette;

/// Bricks per axis in one chunk. 64 = `4³` fits a single u64 occupancy mask.
pub const CHUNK_BRICKS: i32 = 4;
/// Brick slots in one chunk (`CHUNK_BRICKS³`).
pub const CHUNK_VOLUME: u32 = (CHUNK_BRICKS * CHUNK_BRICKS * CHUNK_BRICKS) as u32; // 64
/// Bias added to each signed chunk-axis index so it fits an unsigned 16-bit key field.
/// ±32768 chunks/axis — at LOD0 (chunk ≈ 2.8 m) that's ±90 km, ample for a several-km
/// world; coarser LODs reach exponentially further. Mirrored verbatim in
/// `bindings.wgsl::abs_chunk_key`; the `wgsl_chunk_constants_match_rust` test guards
/// against silent drift (a mismatch reintroduces the camera-shift / blank-world bug).
pub const KEY_BIAS: i32 = 1 << 15;

/// Absolute chunk identity: LOD level + chunk coord on that level's chunk lattice
/// (anchored at world 0, independent of the camera). The GPU key is derived from this.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ChunkKey {
    pub lod: u32,
    /// Chunk coord = brick_index.div_euclid(CHUNK_BRICKS), per axis.
    pub coord: IVec3,
}

impl ChunkKey {
    pub fn new(lod: u32, coord: IVec3) -> Self {
        Self { lod, coord }
    }
}

/// The chunk a brick belongs to, and the brick's local slot (0..63) within it.
pub fn chunk_of(brick: BrickKey, config: &SdfGridConfig) -> (ChunkKey, u32) {
    let s = config.cell_stride();
    // Brick index on the LOD lattice (stride-aligned coord → contiguous index).
    let bi = IVec3::new(
        brick.coord.x.div_euclid(s),
        brick.coord.y.div_euclid(s),
        brick.coord.z.div_euclid(s),
    );
    let cc = IVec3::new(
        bi.x.div_euclid(CHUNK_BRICKS),
        bi.y.div_euclid(CHUNK_BRICKS),
        bi.z.div_euclid(CHUNK_BRICKS),
    );
    let local = IVec3::new(
        bi.x.rem_euclid(CHUNK_BRICKS),
        bi.y.rem_euclid(CHUNK_BRICKS),
        bi.z.rem_euclid(CHUNK_BRICKS),
    );
    let idx = (local.z * CHUNK_BRICKS * CHUNK_BRICKS + local.y * CHUNK_BRICKS + local.x) as u32;
    (ChunkKey::new(brick.lod, cc), idx)
}

/// The absolute 64-bit GPU key for a chunk, packed lexicographically so a sort /
/// binary-search by `(key_hi, key_lo)` orders by lod, then x, y, z. Mirrored exactly by
/// `abs_chunk_key` in `bindings.wgsl`.
pub fn chunk_gpu_key(key: ChunkKey) -> (u32, u32) {
    let cx = ((key.coord.x + KEY_BIAS) as u32) & 0xffff;
    let cy = ((key.coord.y + KEY_BIAS) as u32) & 0xffff;
    let cz = ((key.coord.z + KEY_BIAS) as u32) & 0xffff;
    let key_hi = (key.lod << 16) | cx;
    let key_lo = (cy << 16) | cz;
    (key_hi, key_lo)
}

/// World-space minimum corner of a chunk (its brick-(0,0,0) corner).
pub fn chunk_min_world(key: ChunkKey, config: &SdfGridConfig) -> bevy::math::Vec3 {
    let vs = config.voxel_size_at(key.lod);
    let bricks_per_chunk_world = config.cell_stride() as f32 * vs * CHUNK_BRICKS as f32;
    bevy::math::Vec3::new(
        key.coord.x as f32,
        key.coord.y as f32,
        key.coord.z as f32,
    ) * bricks_per_chunk_world
}

/// World-space edge length of a whole chunk at `lod`.
pub fn chunk_world_size(lod: u32, config: &SdfGridConfig) -> f32 {
    config.cell_stride() as f32 * config.voxel_size_at(lod) * CHUNK_BRICKS as f32
}

/// One entry in the GPU chunk lookup table (sorted by `(key_hi, key_lo)`, binary-
/// searched by the shader). 5×u32 = 20 bytes. `occ_lo|occ_hi` is the 64-bit occupancy
/// mask (bit `i` set ⇒ local brick `i` is resident); `tile_run_base` indexes the packed
/// `tile_run` table where this chunk's `popcount(mask)` brick `atlas_base`s live in
/// ascending local-index order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChunkLookup {
    pub key_hi: u32,
    pub key_lo: u32,
    pub occ_lo: u32,
    pub occ_hi: u32,
    pub tile_run_base: u32,
}

/// One resident brick's GPU record inside a chunk's tile run: its atlas tile origin plus
/// its packed 4-entry material palette (`pal01 = id0|id1<<16`, `pal23 = id2|id3<<16`).
/// 3×u32 = 12 bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BrickTile {
    pub atlas_base: u32,
    pub pal01: u32,
    pub pal23: u32,
}

/// Pixel origin of atlas `tile` (tiles wrap into rows so the texture width stays bounded), packed
/// as `col_px | row_px<<16`. Single source of truth for the `atlas_base` packing, shared by the
/// render-world full rebuild and the incremental [`LiveChunkTables`] so both agree byte-for-byte.
pub fn tile_atlas_base(tile: u32) -> u32 {
    let edge = BRICK_EDGE as u32;
    let tile_width = edge * edge; // 64
    let col_px = (tile % ATLAS_TILES_PER_ROW) * tile_width;
    let row_px = (tile / ATLAS_TILES_PER_ROW) * edge;
    col_px | (row_px << 16)
}

/// Pack a brick's atlas tile + material palette into its GPU [`BrickTile`] record.
pub fn pack_brick_tile(tile: u32, palette: Palette) -> BrickTile {
    BrickTile {
        atlas_base: tile_atlas_base(tile),
        pal01: palette[0] as u32 | ((palette[1] as u32) << 16),
        pal23: palette[2] as u32 | ((palette[3] as u32) << 16),
    }
}

/// The two GPU buffers the shader needs to resolve a brick: the dense per-LOD toroidal DIRECTORY
/// (`chunk_buf` — chunk `c` at `dir_index(c, r)`, empty slots sentinel-tagged) and the packed
/// per-chunk brick runs, plus `r` (ring chunks/axis) for indexing. Built from the resident brick set.
#[derive(Default)]
pub struct ChunkTables {
    pub chunks: Vec<ChunkLookup>,
    /// Per resident brick, grouped by chunk (chunk `c` occupies
    /// `tile_run[c.tile_run_base .. + popcount(c.occ)]`), in ascending local-index order.
    pub tile_run: Vec<BrickTile>,
    /// Ring chunks per axis (`R = ring_bricks / CHUNK_BRICKS`) used to index `chunks` via `dir_index`.
    pub r: i32,
}

/// Group an atlas's resident bricks into the toroidal directory + packed tile-run table by driving
/// the real [`LiveChunkTables`] (one `set_brick` per resident brick) — so the layout matches what
/// the render world uploads byte-for-byte. `tile_of(key)` returns the brick's [`BrickTile`].
pub fn build_chunk_tables(
    atlas: &SdfAtlas,
    config: &SdfGridConfig,
    mut tile_of: impl FnMut(&BrickKey) -> BrickTile,
) -> ChunkTables {
    let mut live = LiveChunkTables::default();
    for key in atlas.bricks.keys() {
        let (ck, local) = chunk_of(*key, config);
        live.set_brick(ck, local, tile_of(key), config);
    }
    let (chunks, tile_run) = live.full_tables();
    ChunkTables {
        chunks,
        tile_run,
        r: config.ring_chunks_per_axis(),
    }
}

// --- Incremental (live) chunk table -------------------------------------------------
//
// `build_chunk_tables` above rebuilds the whole table from the resident brick set every
// topology change — O(bricks log bricks). For a dense scene crossing a coarse LOD snap
// that is ~80k bricks re-grouped every frame of the multi-frame bake drain (the ~20ms
// `extract_sdf` spike). The structures below maintain the SAME logical table incrementally:
// each `insert`/`remove` touches only its own chunk, recording which rows/regions changed
// so the render world can upload just the delta.
//
// The key that makes this cheap: the GPU only needs each chunk's tile-run region to hold
// `popcount(occ)` entries in ascending local order at `tile_run_base` — the regions need
// NOT be contiguous between chunks (verified against `brick.wgsl::brick_in_chunk`). So we
// give every resident chunk a FIXED 64-slot region (`tile_run_base = slot * TILE_RUN_SLOT`),
// and brick churn in one chunk rewrites only that chunk's region, never shifting any other
// chunk's base. Unoccupied slots inside a region are never indexed (popcount skips them).

/// Tile-run entries reserved per resident chunk. Equals [`CHUNK_VOLUME`] (64): the max
/// bricks a chunk can hold, so a chunk's region never overflows regardless of churn.
pub const TILE_RUN_SLOT: u32 = CHUNK_VOLUME;

/// A removed chunk's tail rows are overwritten with this sentinel key so binary search over
/// the fixed physical buffer length still works: `u32::MAX` sorts after every real key, so
/// the search never matches a removed/absent slot. Generalizes the single-entry empty
/// sentinel `prepare_sdf_atlas_gpu` already uploads for a fully-evicted atlas.
pub const SENTINEL_KEY: (u32, u32) = (u32::MAX, u32::MAX);

/// Stable chunk → tile-run-region slot, with a free-list (mirrors [`super::atlas::TileAllocator`]).
/// `tile_run_base = slot * TILE_RUN_SLOT`. Reusing a freed slot before growing `next` keeps the
/// tile-run buffer densely packed in chunk units (bounded by peak resident chunk count).
#[derive(Default)]
pub struct ChunkSlotAllocator {
    slot_of: HashMap<ChunkKey, u32>,
    free: Vec<u32>,
    next: u32,
}

impl ChunkSlotAllocator {
    /// Assign (or return the existing) slot for `ck`. Reuses a freed slot first.
    fn alloc(&mut self, ck: ChunkKey) -> u32 {
        if let Some(&s) = self.slot_of.get(&ck) {
            return s;
        }
        let s = self.free.pop().unwrap_or_else(|| {
            let s = self.next;
            self.next += 1;
            s
        });
        self.slot_of.insert(ck, s);
        s
    }

    /// Return `ck`'s slot to the free pool (chunk emptied). Its stale tile-run entries are
    /// harmless — no live chunk row references them once the slot is unoccupied.
    fn release(&mut self, ck: &ChunkKey) {
        if let Some(s) = self.slot_of.remove(ck) {
            self.free.push(s);
        }
    }

    /// One past the largest slot ever handed out → how many `TILE_RUN_SLOT`-sized regions the
    /// tile-run buffer must span (`high_water() * TILE_RUN_SLOT` entries).
    pub fn high_water(&self) -> u32 {
        self.next
    }
}

/// One resident chunk's live state: its tile-run slot, occupancy mask, and the 64 brick
/// tiles (only `popcount(occ)` are live; the rest are never indexed by the shader).
struct ChunkEntry {
    slot: u32,
    occ: u64,
    tiles: [BrickTile; CHUNK_VOLUME as usize],
}

/// Incrementally-maintained chunk lookup + tile-run table, kept on [`SdfAtlas`]. Mirrors what
/// [`build_chunk_tables`] produces (a dense per-LOD toroidal directory + per-chunk tile runs), but
/// each `insert`/`remove` updates only its own directory slot and records the delta in the dirty
/// fields so the render world uploads just the changed slots/regions instead of recreating both
/// buffers.
#[derive(Default)]
pub struct LiveChunkTables {
    slots: ChunkSlotAllocator,
    chunks: HashMap<ChunkKey, ChunkEntry>,
    /// Reverse of each live chunk's tile-run slot → its key, so a dirty slot resolves to its
    /// region in O(1) (only live chunks; a freed slot is dropped — its region is never indexed).
    slot_to_key: HashMap<u32, ChunkKey>,
    /// The DENSE per-LOD toroidal DIRECTORY = the GPU `chunk_buf`. `R³ × lod_count` fixed slots;
    /// chunk `c` lives at `lod·R³ + flatten(rem_euclid(c, R))`, with empty slots carrying the
    /// [`SENTINEL_KEY`] tag. Lazily sized on the first [`set_brick`](Self::set_brick) (when the
    /// config-derived `R` is known). The GPU indexes it DIRECTLY and compares the key tag — no
    /// binary search, no sort — so insert/remove/evict is an O(1) slot write. See
    /// `brick.wgsl::find_chunk`. `tile_run_base` still points into the SPARSE tile-run buffer.
    dir: Vec<ChunkLookup>,
    /// Ring chunks per axis (`R = ring_bricks / CHUNK_BRICKS`), cached from config at first sizing.
    r: i32,

    /// Directory slots whose `ChunkLookup` changed this frame (GPU `chunk_buf` indices to re-upload).
    /// Private: the per-frame delta is exposed only through [`LiveChunkTables::upload`], so the
    /// full-rebuild-vs-delta policy has a single owner (was leaked + read by render.rs + two tests).
    dirty_rows: BTreeSet<u32>,
    /// Tile-run slots whose 64-entry region changed this frame.
    dirty_slots: BTreeSet<u32>,
}

/// What the render world must write to the GPU lookup buffers this frame, produced by
/// [`LiveChunkTables::upload`] — the SINGLE owner of the full-rebuild-vs-delta decision and the
/// tile-run headroom policy (was hand-copied in `render.rs` extract, the churn-differential test,
/// and the bake-scheduler lifecycle mirror). Carries NATIVE chunk records; the render world maps
/// them onto its GPU mirror, the test mirrors apply them in place.
pub enum ChunkUpload {
    /// Recreate both buffers: the directory outgrew the allocated `chunk_buf`, the tile-run outgrew
    /// its slots, or this is the first upload. `tile_run` is UNPADDED (`slot_high_water *
    /// TILE_RUN_SLOT` entries) — the caller sizes the GPU buffer to `cap_slots` and pads the tail.
    /// `cap_rows`/`cap_slots` are the capacities to (re)allocate to.
    Full {
        rows: Vec<ChunkLookup>,
        tile_run: Vec<BrickTile>,
        cap_rows: u32,
        cap_slots: u32,
    },
    /// In-place writes: the dirty directory slots + their dense tile-run regions. The directory is
    /// fixed-position, so every entry is an index→value write (no shift, no realloc).
    Delta {
        row_updates: Vec<(u32, ChunkLookup)>,
        region_updates: Vec<(u32, [BrickTile; CHUNK_VOLUME as usize])>,
    },
}

/// An empty directory slot: the [`SENTINEL_KEY`] tag (never matches a real chunk), no occupancy.
fn sentinel_lookup() -> ChunkLookup {
    ChunkLookup {
        key_hi: SENTINEL_KEY.0,
        key_lo: SENTINEL_KEY.1,
        occ_lo: 0,
        occ_hi: 0,
        tile_run_base: 0,
    }
}

/// Physical directory slot for chunk `ck` given ring chunks/axis `r`: `lod·R³ + flatten(c mod R)`
/// with `rem_euclid` (handles negative coords). EXACT mirror of `dir_index` in `bindings.wgsl` —
/// the `wgsl_chunk_constants_match_rust` / GPU-rig parity tests guard against drift.
pub fn dir_index(ck: ChunkKey, r: i32) -> usize {
    let mx = ck.coord.x.rem_euclid(r);
    let my = ck.coord.y.rem_euclid(r);
    let mz = ck.coord.z.rem_euclid(r);
    (ck.lod as usize) * (r * r * r) as usize + (mz * r * r + my * r + mx) as usize
}

impl LiveChunkTables {
    /// Mark a resident brick present in its chunk (insert or palette/tile change). `local` is the
    /// brick's 0..63 slot from [`chunk_of`]; `tile` is its packed atlas origin + palette. O(1): one
    /// directory slot write + a tile-run slot. `config` supplies `R`/`lod_count` to lazily size the
    /// directory on first use. The whole 20 B directory record is published ATOMICALLY (tag + occ +
    /// tile_run_base in one write) — a tag-valid slot never points at unbaked/old texels.
    pub fn set_brick(&mut self, ck: ChunkKey, local: u32, tile: BrickTile, config: &SdfGridConfig) {
        if self.dir.is_empty() {
            self.r = config.ring_chunks_per_axis();
            let n = config.directory_len();
            self.dir = vec![sentinel_lookup(); n];
        }
        let idx = dir_index(ck, self.r);
        let tag = chunk_gpu_key(ck);

        // Free-on-overwrite belt: if a DIFFERENT chunk still occupies this physical slot (it left the
        // window without being cleared), reclaim it before publishing the new chunk. With explicit
        // clear-on-exit (the recenter clears exited chunks first) this won't normally fire, but it
        // keeps the tile-run bounded against an ordering edge that leaves a stale slot.
        let cur = self.dir[idx];
        if (cur.key_hi, cur.key_lo) != SENTINEL_KEY && (cur.key_hi, cur.key_lo) != tag {
            let old_slot = cur.tile_run_base / TILE_RUN_SLOT;
            if let Some(old_ck) = self.slot_to_key.remove(&old_slot) {
                self.chunks.remove(&old_ck);
                self.slots.release(&old_ck);
                self.dirty_slots.remove(&old_slot);
            }
        }

        let slot = self.slots.alloc(ck);
        let entry = self.chunks.entry(ck).or_insert_with(|| ChunkEntry {
            slot,
            occ: 0,
            tiles: [BrickTile::default(); CHUNK_VOLUME as usize],
        });
        entry.occ |= 1u64 << local;
        entry.tiles[local as usize] = tile;
        let occ = entry.occ;

        self.slot_to_key.insert(slot, ck);
        self.dir[idx] = ChunkLookup {
            key_hi: tag.0,
            key_lo: tag.1,
            occ_lo: occ as u32,
            occ_hi: (occ >> 32) as u32,
            tile_run_base: slot * TILE_RUN_SLOT,
        };
        self.dirty_rows.insert(idx as u32);
        self.dirty_slots.insert(slot);
    }

    /// Clear a brick from its chunk. If the chunk becomes empty, free its tile-run slot and reset its
    /// directory slot to the sentinel tag (so the GPU's tag compare misses it → coarse fallback).
    /// O(1) — no row shift. (Departed chunks are cleared here on exit, which is the make-before-break
    /// reclaim path that keeps the tile-run bounded; see the migration plan.)
    pub fn clear_brick(&mut self, ck: ChunkKey, local: u32) {
        let Some(entry) = self.chunks.get_mut(&ck) else {
            return;
        };
        entry.occ &= !(1u64 << local);
        entry.tiles[local as usize] = BrickTile::default();
        let slot = entry.slot;
        let occ = entry.occ;
        let idx = dir_index(ck, self.r);
        if occ != 0 {
            // Still resident → just refresh the occupancy in its directory slot + re-upload its region.
            self.dir[idx].occ_lo = occ as u32;
            self.dir[idx].occ_hi = (occ >> 32) as u32;
            self.dirty_rows.insert(idx as u32);
            self.dirty_slots.insert(slot);
            return;
        }
        // Chunk emptied → sentinel its directory slot, free the tile-run slot. The region is now
        // unreferenced (the sentinel tag never resolves), so it needs no upload.
        self.chunks.remove(&ck);
        self.slot_to_key.remove(&slot);
        self.slots.release(&ck);
        self.dir[idx] = sentinel_lookup();
        self.dirty_rows.insert(idx as u32);
        self.dirty_slots.remove(&slot);
    }

    /// Number of directory slots (= GPU `chunk_buf` length = `R³ × lod_count`, 0 until first sized).
    pub fn row_count(&self) -> u32 {
        self.dir.len() as u32
    }

    /// One past the largest tile-run slot → tile-run buffer must span `tile_run_capacity()` entries.
    pub fn tile_run_capacity(&self) -> u32 {
        self.slots.high_water() * TILE_RUN_SLOT
    }

    /// One past the largest tile-run slot ever handed out (each slot owns a `TILE_RUN_SLOT`-sized
    /// region). The full-rebuild path iterates `0..slot_high_water()` to lay out the tile-run buffer.
    pub fn slot_high_water(&self) -> u32 {
        self.slots.high_water()
    }

    /// The `ChunkLookup` at directory slot `idx` (for a delta or full upload) — a direct read of the
    /// dense directory (empty slots return the sentinel tag).
    pub fn lookup_at(&self, idx: u32) -> ChunkLookup {
        debug_assert!(
            (idx as usize) < self.dir.len(),
            "lookup_at({idx}) past directory end (len {})",
            self.dir.len()
        );
        self.dir[idx as usize]
    }

    /// Serialize one chunk's region as the shader reads it: the `popcount(occ)` live bricks packed
    /// DENSELY from index 0 in ascending local order, the rest left default. The shader indexes a
    /// brick at `tile_run_base + popcount(occ & below)` — i.e. by its DENSE rank, not its raw local
    /// slot — so the region must be packed by rank here (it's stored sparse-by-local internally).
    fn dense_region(entry: &ChunkEntry) -> [BrickTile; CHUNK_VOLUME as usize] {
        let mut region = [BrickTile::default(); CHUNK_VOLUME as usize];
        let mut rank = 0usize;
        let mut bits = entry.occ;
        while bits != 0 {
            let local = bits.trailing_zeros() as usize; // next occupied slot, ascending
            region[rank] = entry.tiles[local];
            rank += 1;
            bits &= bits - 1; // clear lowest set bit
        }
        region
    }

    /// The packed tile-run region for tile-run `slot` (dense, ascending-local order — see
    /// [`dense_region`](Self::dense_region)). A freed slot (no live chunk) returns zeros, but the
    /// dirty set never carries freed slots (`clear_brick` drops them), so this is the live path.
    pub fn tile_region(&self, slot: u32) -> [BrickTile; CHUNK_VOLUME as usize] {
        self.slot_to_key
            .get(&slot)
            .map(|ck| Self::dense_region(&self.chunks[ck]))
            .unwrap_or([BrickTile::default(); CHUNK_VOLUME as usize])
    }

    /// Materialize the WHOLE table for a full upload: the entire dense directory (`R³ × lod_count`
    /// slots, empty ones sentinel-tagged) and a `slot_high_water()*TILE_RUN_SLOT`-entry tile-run
    /// buffer with each live slot's DENSELY-packed region at its `slot*TILE_RUN_SLOT` base.
    pub fn full_tables(&self) -> (Vec<ChunkLookup>, Vec<BrickTile>) {
        let mut tile_run =
            vec![BrickTile::default(); (self.slot_high_water() * TILE_RUN_SLOT) as usize];
        for (&slot, ck) in &self.slot_to_key {
            let base = (slot * TILE_RUN_SLOT) as usize;
            tile_run[base..base + TILE_RUN_SLOT as usize]
                .copy_from_slice(&Self::dense_region(&self.chunks[ck]));
        }
        (self.dir.clone(), tile_run)
    }

    /// Decide and materialize this frame's GPU lookup-buffer upload against the caller's CURRENT
    /// buffer capacities (`cap_rows` = current `chunk_buf` length, `cap_slots` = current tile-run
    /// length). Returns [`ChunkUpload::Full`] (with the capacities to grow to) on the first upload
    /// (`cap_rows == 0`) or when the tables outgrew the buffers, else [`ChunkUpload::Delta`] carrying
    /// only the slots/regions marked dirty since the last [`clear_dirty`](Self::clear_dirty). The
    /// full-rebuild predicate and the tile-run headroom (+50%, min one extra slot) live HERE and
    /// nowhere else — every consumer (render extract + the test mirrors) routes through this.
    pub fn upload(&self, cap_rows: u32, cap_slots: u32) -> ChunkUpload {
        let needed_rows = self.row_count();
        let needed_slots = self.tile_run_capacity();
        if cap_rows == 0 || needed_rows > cap_rows || needed_slots > cap_slots {
            // The directory (chunk_buf) is FIXED-size — size it EXACTLY (no headroom; it never
            // grows again). The sparse tile-run can grow, so give it +50% (min one extra slot) to
            // avoid recreating it every frame.
            let new_cap_slots = (needed_slots + needed_slots / 2).max(needed_slots + TILE_RUN_SLOT);
            let (rows, tile_run) = self.full_tables();
            ChunkUpload::Full { rows, tile_run, cap_rows: needed_rows, cap_slots: new_cap_slots }
        } else {
            let row_updates = self.dirty_rows.iter().map(|&r| (r, self.lookup_at(r))).collect();
            let region_updates = self.dirty_slots.iter().map(|&s| (s, self.tile_region(s))).collect();
            ChunkUpload::Delta { row_updates, region_updates }
        }
    }

    /// Clear the per-frame delta record. Called from the main world AFTER the render world has
    /// extracted the delta (see `clear_chunk_table_dirty`), before the next frame accumulates.
    pub fn clear_dirty(&mut self) {
        self.dirty_rows.clear();
        self.dirty_slots.clear();
    }
}

/// The distinct non-empty chunks an atlas currently has resident — for the debug
/// overlay (one wireframe box per chunk).
pub fn resident_chunks(atlas: &SdfAtlas, config: &SdfGridConfig) -> Vec<ChunkKey> {
    use std::collections::HashSet;
    let mut set: HashSet<ChunkKey> = HashSet::new();
    for key in atlas.bricks.keys() {
        set.insert(chunk_of(*key, config).0);
    }
    set.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SdfGridConfig {
        SdfGridConfig::default()
    }

    /// chunk_of maps a brick to a chunk coord + local slot, and local round-trips into
    /// the 0..63 range with the documented packing.
    #[test]
    fn chunk_of_local_index_in_range_and_roundtrips() {
        let cfg = config();
        let s = cfg.cell_stride();
        for bz in 0..CHUNK_BRICKS {
            for by in 0..CHUNK_BRICKS {
                for bx in 0..CHUNK_BRICKS {
                    // Brick at chunk (0,0,0), local (bx,by,bz).
                    let coord = IVec3::new(bx * s, by * s, bz * s);
                    let (ck, local) = chunk_of(BrickKey::new(0, coord), &cfg);
                    assert_eq!(ck.coord, IVec3::ZERO);
                    let expect = (bz * CHUNK_BRICKS * CHUNK_BRICKS + by * CHUNK_BRICKS + bx) as u32;
                    assert_eq!(local, expect);
                    assert!(local < CHUNK_VOLUME);
                }
            }
        }
    }

    /// Negative brick coords land in the chunk below (div_euclid), not chunk 0.
    #[test]
    fn negative_coords_use_euclidean_chunk() {
        let cfg = config();
        let s = cfg.cell_stride();
        // One brick left of the origin → brick index -1 → chunk -1, local CHUNK_BRICKS-1.
        let (ck, local) = chunk_of(BrickKey::new(0, IVec3::new(-s, 0, 0)), &cfg);
        assert_eq!(ck.coord.x, -1);
        assert_eq!(local % CHUNK_BRICKS as u32, (CHUNK_BRICKS - 1) as u32);
    }

    /// The GPU key is order-preserving: sorting by (key_hi,key_lo) orders by lod, x, y, z
    /// — required for the shader's binary search.
    #[test]
    fn gpu_key_is_order_preserving() {
        let mut keys = vec![
            ChunkKey::new(0, IVec3::new(0, 0, 0)),
            ChunkKey::new(0, IVec3::new(0, 0, 1)),
            ChunkKey::new(0, IVec3::new(0, 1, 0)),
            ChunkKey::new(0, IVec3::new(1, 0, 0)),
            ChunkKey::new(0, IVec3::new(-1, 0, 0)),
            ChunkKey::new(1, IVec3::new(-5, -5, -5)),
        ];
        let mut by_packed = keys.clone();
        by_packed.sort_by_key(|k| chunk_gpu_key(*k));
        // Expected lexicographic order on (lod, x, y, z), with x,y,z biased ascending.
        keys.sort_by_key(|k| (k.lod, k.coord.x, k.coord.y, k.coord.z));
        assert_eq!(by_packed, keys);
    }

    /// Distinct (lod,coord) within range never collide on the packed key.
    #[test]
    fn gpu_key_no_collision_in_range() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for lod in 0..4u32 {
            for x in -3..=3 {
                for y in -3..=3 {
                    for z in -3..=3 {
                        let k = chunk_gpu_key(ChunkKey::new(lod, IVec3::new(x, y, z)));
                        assert!(seen.insert(k), "collision at lod={lod} ({x},{y},{z})");
                    }
                }
            }
        }
    }

    /// The chunk-addressing constants are hand-duplicated in `bindings.wgsl`
    /// (`abs_chunk_key` / `local_brick_index`) because WGSL can't import Rust consts.
    /// A silent mismatch there makes the GPU search a different key than the CPU stored
    /// → the camera-shift / blank-world bug class this clipmap rework fixed. This test
    /// parses the shader and pins both constants to the Rust source of truth, so any
    /// future edit to one side without the other fails CI instead of shipping a
    /// hard-to-trace visual corruption.
    #[test]
    fn wgsl_chunk_constants_match_rust() {
        let src = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/shaders/sdf/bindings.wgsl"
        ))
        .expect("read bindings.wgsl");

        // Helper: find `pat` and parse the integer literal that follows it.
        let int_after = |pat: &str| -> i64 {
            let i = src
                .find(pat)
                .unwrap_or_else(|| panic!("bindings.wgsl missing `{pat}`"));
            let tail = &src[i + pat.len()..];
            let digits: String = tail
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            digits
                .parse()
                .unwrap_or_else(|_| panic!("no integer after `{pat}` in bindings.wgsl"))
        };

        // `const CHUNK_BRICKS: i32 = 4;`
        let wgsl_chunk_bricks = int_after("const CHUNK_BRICKS: i32 =");
        assert_eq!(
            wgsl_chunk_bricks, CHUNK_BRICKS as i64,
            "WGSL CHUNK_BRICKS ({wgsl_chunk_bricks}) != Rust chunk::CHUNK_BRICKS ({CHUNK_BRICKS})"
        );

        // `let bias = 32768;` inside abs_chunk_key — must equal Rust KEY_BIAS.
        let wgsl_bias = int_after("let bias =");
        assert_eq!(
            wgsl_bias, KEY_BIAS as i64,
            "WGSL chunk key bias ({wgsl_bias}) != Rust chunk::KEY_BIAS ({KEY_BIAS})"
        );
    }

    // --- Chunk-table build ↔ shader-resolve round-trip ------------------------------

    use super::super::atlas::PackedBrick;

    /// Mirror EXACTLY what `brick.wgsl::find_chunk` + `brick_in_chunk` do on the GPU: index the
    /// dense directory by `dir_index(ck, r)`, accept only if the stored key TAG matches (sentinel /
    /// a different wrapped chunk → miss), then test the occupancy bit and index the tile run at
    /// `tile_run_base + popcount(bits strictly below the slot)`. Keeping this in lockstep with the
    /// shader is the point of the tests below.
    fn shader_resolve(
        tables: &ChunkTables,
        config: &SdfGridConfig,
        brick: BrickKey,
    ) -> Option<BrickTile> {
        let (ck, li) = chunk_of(brick, config); // li = local slot 0..63
        let idx = dir_index(ck, tables.r);
        if idx >= tables.chunks.len() {
            return None;
        }
        let chunk = tables.chunks[idx];
        if (chunk.key_hi, chunk.key_lo) != chunk_gpu_key(ck) {
            return None; // sentinel slot, or a different chunk shares this `mod R` slot
        }
        let occ = (chunk.occ_lo as u64) | ((chunk.occ_hi as u64) << 32);
        if (occ >> li) & 1 == 0 {
            return None; // brick not resident in this chunk
        }
        let below = occ & ((1u64 << li) - 1); // bits strictly below the slot
        let off = below.count_ones();
        Some(tables.tile_run[(chunk.tile_run_base + off) as usize])
    }

    fn dummy_brick() -> PackedBrick {
        use crate::sdf_render::edits::{PALETTE_EMPTY, PALETTE_K};
        PackedBrick {
            palette: [PALETTE_EMPTY; PALETTE_K],
            baked_hash: 0,
        }
    }

    /// End-to-end CPU↔GPU contract: bricks scattered across several chunks and LODs must
    /// each resolve — via the shader's occupancy-mask + popcount-offset unpack — back to
    /// the exact tile `build_chunk_tables` assigned them, and a brick that isn't resident
    /// must miss. A packing bug here silently maps a brick to the wrong tile (the visual
    /// corruption class the chunked rework fixed), so this is the key regression guard.
    #[test]
    fn build_chunk_tables_resolves_each_brick_to_its_tile() {
        let cfg = config();
        let s = cfg.cell_stride();
        let c = CHUNK_BRICKS;

        // Encode each brick's identity into a unique tile so a wrong-tile mapping shows.
        let tile_of = |k: &BrickKey| -> BrickTile {
            let base = (k.lod << 28)
                ^ ((k.coord.x as u32) << 16)
                ^ ((k.coord.y as u32) << 8)
                ^ (k.coord.z as u32);
            BrickTile { atlas_base: base, pal01: base ^ 0x1111, pal23: base ^ 0x2222 }
        };

        // Bricks across: a sparse subset of slots in chunk (0,0,0), a neighbouring chunk,
        // and a negative-coord chunk at lod 1.
        let mut atlas = SdfAtlas::default();
        let mut keys = Vec::new();
        for (lx, ly, lz) in [(0, 0, 0), (1, 0, 0), (3, 2, 1)] {
            keys.push(BrickKey::new(0, IVec3::new(lx * s, ly * s, lz * s)));
        }
        keys.push(BrickKey::new(0, IVec3::new(c * s, 0, 0))); // chunk (+x), local 0
        keys.push(BrickKey::new(1, IVec3::new(-s, -s, -s))); // lod1, chunk (-1,-1,-1)
        for k in &keys {
            atlas.bricks.insert(*k, dummy_brick());
        }

        let tables = build_chunk_tables(&atlas, &cfg, tile_of);

        for k in &keys {
            let got = shader_resolve(&tables, &cfg, *k)
                .unwrap_or_else(|| panic!("brick {k:?} failed to resolve"));
            assert_eq!(got, tile_of(k), "brick {k:?} resolved to the wrong tile");
        }

        // Unoccupied slot in a resident chunk must miss (not alias a neighbour's tile).
        let absent = BrickKey::new(0, IVec3::new(2 * s, 2 * s, 2 * s));
        assert!(
            shader_resolve(&tables, &cfg, absent).is_none(),
            "an unoccupied slot in a resident chunk must not resolve"
        );

        // A brick in a chunk that isn't resident at all must miss.
        let no_chunk = BrickKey::new(0, IVec3::new(50 * c * s, 0, 0));
        assert!(
            shader_resolve(&tables, &cfg, no_chunk).is_none(),
            "a brick in an absent chunk must not resolve"
        );
    }

    // --- Chunk world geometry (debug-viz boxes + LOD-shell convention) --------------

    /// A LOD-`L` chunk covers exactly 2× the world extent of LOD `L-1` — the nested
    /// "twice as coarse / twice the area" shell property the clipmap is built on.
    #[test]
    fn chunk_world_size_doubles_per_lod() {
        let cfg = config();
        for lod in 1..cfg.lod_count {
            let coarse = chunk_world_size(lod, &cfg);
            let fine = chunk_world_size(lod - 1, &cfg);
            assert!(
                (coarse - 2.0 * fine).abs() < 1e-4,
                "lod {lod} chunk ({coarse}) must be 2x lod {} ({fine})",
                lod - 1
            );
        }
        // Anchor the absolute scale: a LOD-0 chunk spans cell_stride·voxel·CHUNK_BRICKS.
        let expect0 = cfg.cell_stride() as f32 * cfg.voxel_size_at(0) * CHUNK_BRICKS as f32;
        assert!((chunk_world_size(0, &cfg) - expect0).abs() < 1e-6);
    }

    /// The world point → chunk mapping is geometrically self-consistent: the chunk a
    /// point resolves to (`chunk_of(world_to_brick_lod(p))`) has a world box that
    /// actually encloses `p` on every axis: `min ≤ p < min + size`. A drift between the
    /// addressing math and the debug-viz geometry would break this.
    #[test]
    fn chunk_box_contains_its_world_point() {
        use bevy::math::Vec3;
        let cfg = config();
        for lod in 0..cfg.lod_count {
            let size = chunk_world_size(lod, &cfg);
            for p in [
                Vec3::ZERO,
                Vec3::new(0.05, 0.05, 0.05),
                Vec3::new(13.7, -4.2, 88.1),
                Vec3::new(-260.0, 30.0, -9.0),
            ] {
                let brick = cfg.world_to_brick_lod(p, lod);
                let (ck, _) = chunk_of(BrickKey::new(lod, brick), &cfg);
                let min = chunk_min_world(ck, &cfg);
                let max = min + Vec3::splat(size);
                assert!(
                    p.x >= min.x && p.x < max.x
                        && p.y >= min.y && p.y < max.y
                        && p.z >= min.z && p.z < max.z,
                    "lod {lod}: point {p:?} not in its chunk box [{min:?}, {max:?})"
                );
            }
        }
    }

    /// Adjacent chunks tile exactly — the next chunk's min corner is one full chunk
    /// further on, with no gap or overlap (so the debug overlay reads as a clean grid).
    #[test]
    fn adjacent_chunks_tile_without_gaps() {
        let cfg = config();
        for lod in 0..cfg.lod_count {
            let size = chunk_world_size(lod, &cfg);
            let base = ChunkKey::new(lod, IVec3::new(2, -1, 0));
            let min = chunk_min_world(base, &cfg);
            for (axis, delta) in [
                (0, IVec3::X),
                (1, IVec3::Y),
                (2, IVec3::Z),
            ] {
                let next = chunk_min_world(ChunkKey::new(lod, base.coord + delta), &cfg);
                let step = next - min;
                // Only the stepped axis advances, by exactly one chunk world size.
                for a in 0..3 {
                    let want = if a == axis { size } else { 0.0 };
                    assert!(
                        (step[a] - want).abs() < 1e-4,
                        "lod {lod} axis {axis}: neighbour offset[{a}]={} want {want}",
                        step[a]
                    );
                }
            }
        }
    }

    /// Clearing a chunk's last brick reverts its directory slot to the sentinel tag — there is no row
    /// shift (the directory is FIXED-size), so the size never changes and an untouched chunk keeps its
    /// slot. Replaces the old sorted-array shrink/dirty-row-pruning test (that whole class is gone).
    #[test]
    fn clear_brick_sentinels_slot_keeps_directory_size() {
        let cfg = config();
        let r = cfg.ring_chunks_per_axis();
        let tile = BrickTile { atlas_base: 7, pal01: 0, pal23: 0 };
        let mut live = LiveChunkTables::default();
        let a = ChunkKey::new(0, IVec3::new(0, 0, 0));
        let b = ChunkKey::new(0, IVec3::new(1, 0, 0));
        live.set_brick(a, 0, tile, &cfg);
        live.set_brick(b, 0, tile, &cfg);
        let n = live.row_count();
        assert!(n > 0, "directory sized on first set");
        live.clear_dirty();

        // Clear b's only brick → b's slot reverts to the sentinel tag; the directory size is unchanged.
        live.clear_brick(b, 0);
        assert_eq!(live.row_count(), n, "directory is fixed-size — clearing never shrinks it");
        let idx_b = dir_index(b, r) as u32;
        assert!(live.dirty_rows.contains(&idx_b), "the cleared slot is marked dirty");
        let eb = live.lookup_at(idx_b);
        assert_eq!((eb.key_hi, eb.key_lo), SENTINEL_KEY, "cleared slot carries the sentinel tag");

        // The untouched chunk `a` still owns its directory slot.
        let ea = live.lookup_at(dir_index(a, r) as u32);
        assert_eq!((ea.key_hi, ea.key_lo), chunk_gpu_key(a), "untouched chunk keeps its tag");
    }

    /// The incremental [`LiveChunkTables`] must produce GPU buffers the SHADER unpacks back to the
    /// exact tile each brick was given — same contract as `build_chunk_tables`, but via the live
    /// path. The shader indexes a chunk's tile run densely (`tile_run_base + popcount(below)`), so a
    /// region serialized sparse-by-local (gaps at empty slots) resolves bricks to the WRONG tile —
    /// the "mangled world" the live rework regressed. Mirrors the ground-truth test's brick set so
    /// the two paths are directly comparable.
    #[test]
    fn live_table_resolves_each_brick_to_its_tile() {
        let cfg = config();
        let s = cfg.cell_stride();
        let c = CHUNK_BRICKS;
        let tile_of = |k: &BrickKey| -> BrickTile {
            let base = (k.lod << 28)
                ^ ((k.coord.x as u32) << 16)
                ^ ((k.coord.y as u32) << 8)
                ^ (k.coord.z as u32);
            BrickTile { atlas_base: base, pal01: base ^ 0x1111, pal23: base ^ 0x2222 }
        };

        let mut keys = Vec::new();
        // (3,2,1) → local 27, sharing chunk (0,0,0) with locals 0 and 16 → a sparse mask whose
        // popcount position differs from the raw local index (the densify bug's trigger).
        for (lx, ly, lz) in [(0, 0, 0), (1, 0, 0), (3, 2, 1)] {
            keys.push(BrickKey::new(0, IVec3::new(lx * s, ly * s, lz * s)));
        }
        keys.push(BrickKey::new(0, IVec3::new(c * s, 0, 0))); // neighbouring chunk
        keys.push(BrickKey::new(1, IVec3::new(-s, -s, -s))); // lod1 negative chunk

        let mut live = LiveChunkTables::default();
        for k in &keys {
            let (ck, local) = chunk_of(*k, &cfg);
            live.set_brick(ck, local, tile_of(k), &cfg);
        }
        let (chunks, tile_run) = live.full_tables();
        let tables = ChunkTables { chunks, tile_run, r: cfg.ring_chunks_per_axis() };

        for k in &keys {
            let got = shader_resolve(&tables, &cfg, *k)
                .unwrap_or_else(|| panic!("live brick {k:?} failed to resolve"));
            assert_eq!(got, tile_of(k), "live brick {k:?} resolved to the wrong tile");
        }
        // An unoccupied slot in a resident chunk must miss, not alias a packed neighbour.
        let absent = BrickKey::new(0, IVec3::new(2 * s, 2 * s, 2 * s));
        assert!(
            shader_resolve(&tables, &cfg, absent).is_none(),
            "an unoccupied slot in a resident chunk must not resolve"
        );
    }

    /// THE end-to-end differential guard: drive a long randomized set/clear churn through the
    /// incremental table + its DELTA-UPLOAD protocol (mirroring `render.rs` exactly — the
    /// fixed-size directory is sized once, then every frame writes only the dirty directory slots +
    /// tile-run regions in place), and after EVERY frame resolve every resident brick through the
    /// shader's direct-index tag-check + popcount unpack. A desync anywhere (a directory slot left
    /// pointing at a departed chunk, a stale occupancy bit, slot reuse, or a missing sentinel on an
    /// add-after-remove in the same cycle) maps a brick to the wrong tile or drops it — the
    /// corruption class that revert proved was in this rework. Deterministic xorshift, no GPU needed.
    #[test]
    fn live_delta_upload_matches_ground_truth_under_churn() {
        use std::collections::HashMap;

        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        let cfg = config();
        let r = cfg.ring_chunks_per_axis();
        let mut live = LiveChunkTables::default();
        let mut truth: HashMap<(ChunkKey, u32), BrickTile> = HashMap::new();

        // GPU mirror = the dense fixed-size directory + sparse tile-run, exactly the two buffers
        // `render.rs` maintains (empty directory slots carry the sentinel tag — no separate tail).
        let mut gpu_rows: Vec<ChunkLookup> = Vec::new();
        let mut gpu_tiles: Vec<BrickTile> = Vec::new();
        let mut cap_rows: u32 = 0;
        let mut cap_slots: u32 = 0;

        // Resolve (ck, local) the way the shader does: direct-index the directory by `dir_index`,
        // tag-check (sentinel / wrapped-different chunk → miss), then occupancy popcount.
        let resolve = |rows: &[ChunkLookup], tiles: &[BrickTile], ck: ChunkKey, local: u32| {
            let idx = dir_index(ck, r);
            if idx >= rows.len() {
                return None;
            }
            let c = rows[idx];
            if (c.key_hi, c.key_lo) != chunk_gpu_key(ck) {
                return None;
            }
            let occ = (c.occ_lo as u64) | ((c.occ_hi as u64) << 32);
            if (occ >> local) & 1 == 0 {
                return None;
            }
            let off = (occ & ((1u64 << local) - 1)).count_ones();
            Some(tiles[(c.tile_run_base + off) as usize])
        };

        // Small coord space (≤128 chunks) → heavy toroidal slot reuse + tile-run free/reuse, the
        // camera-move stress that surfaced the bugs.
        let span = 4u64;
        for frame in 0u32..4000 {
            // A real bake frame applies MANY set/clear ops before one upload (one dirty cycle), so
            // a remove and an add routinely coexist in the same cycle — the case where a slot must
            // end carrying the LATEST occupant's record (or the sentinel), never a stale one. Batch
            // them here; bias toward `set` early so the table fills before churn dominates.
            let batch = 1 + (rng() % 24);
            for _ in 0..batch {
                let r = rng();
                let ck = ChunkKey::new(
                    ((r >> 24) % 2) as u32,
                    IVec3::new(
                        (r % span) as i32,
                        ((r >> 8) % span) as i32,
                        ((r >> 16) % span) as i32,
                    ),
                );
                let local = ((r >> 32) % CHUNK_VOLUME as u64) as u32;
                // ~60% sets so churn reaches a populated steady state (not mostly-empty).
                if (r >> 48) % 5 < 3 {
                    let t = BrickTile {
                        atlas_base: r as u32 ^ frame.wrapping_mul(2_654_435_761),
                        pal01: r as u32 ^ 0xAAAA,
                        pal23: (r >> 16) as u32 ^ 0x5555,
                    };
                    live.set_brick(ck, local, t, &cfg);
                    truth.insert((ck, local), t);
                } else {
                    live.clear_brick(ck, local);
                    truth.remove(&(ck, local));
                }
            }

            // --- apply the upload exactly as render.rs's extract/prepare would, through the SAME
            // `LiveChunkTables::upload` accessor that owns the rebuild-vs-delta + headroom policy. A
            // Full sizes the buffers once; a Delta writes only the dirty slots/regions, in place. ---
            match live.upload(cap_rows, cap_slots) {
                ChunkUpload::Full { rows, tile_run, cap_rows: cr, cap_slots: cs } => {
                    cap_rows = cr;
                    cap_slots = cs;
                    gpu_rows = rows;
                    gpu_tiles = tile_run;
                    gpu_tiles.resize(cap_slots as usize, BrickTile::default());
                }
                ChunkUpload::Delta { row_updates, region_updates } => {
                    for (row, look) in row_updates {
                        gpu_rows[row as usize] = look;
                    }
                    for (slot, region) in region_updates {
                        let base = (slot * TILE_RUN_SLOT) as usize;
                        gpu_tiles[base..base + TILE_RUN_SLOT as usize].copy_from_slice(&region);
                    }
                }
            }
            live.clear_dirty();

            // --- verify the mirror against ground truth ---
            for (&(ck, local), &t) in &truth {
                match resolve(&gpu_rows, &gpu_tiles, ck, local) {
                    Some(got) => assert_eq!(
                        got, t,
                        "frame {frame}: brick {ck:?} local {local} resolved to the wrong tile"
                    ),
                    None => panic!("frame {frame}: resident brick {ck:?} local {local} failed to resolve"),
                }
            }
            // A non-resident slot in a (possibly resident) chunk must miss — never alias.
            let p = rng();
            let probe_ck = ChunkKey::new(0, IVec3::new((p % span) as i32, 0, 0));
            let probe_local = ((p >> 40) % CHUNK_VOLUME as u64) as u32;
            if !truth.contains_key(&(probe_ck, probe_local)) {
                assert!(
                    resolve(&gpu_rows, &gpu_tiles, probe_ck, probe_local).is_none(),
                    "frame {frame}: absent brick {probe_ck:?} local {probe_local} wrongly resolved"
                );
            }
        }
    }
}
