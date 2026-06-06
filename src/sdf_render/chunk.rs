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

use std::collections::BTreeSet;
// FxHashMap for the live chunk-table maps mutated per baked brick (chunk→slot, chunk entries,
// slot→chunk). Integer keys; FxHash over std SipHash cuts the per-brick set_brick/clear_brick cost.
use rustc_hash::{FxHashMap, FxHashSet};

use bevy::math::IVec3;

use super::SdfGridConfig;
use super::atlas::{ATLAS_TILES_PER_ROW, BRICK_EDGE, BrickKey, SdfAtlas};
use super::edits::{PALETTE_EMPTY, Palette};

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
/// searched by the shader). 6×u32 = 24 bytes. `occ_lo|occ_hi` is the 64-bit occupancy
/// mask (bit `i` set ⇒ local brick `i` is resident); `tile_run_base` indexes the packed
/// `tile_run` table where this chunk's `popcount(mask)` brick `atlas_base`s live in
/// ascending local-index order. `probe_base` is the DDGI finest-resident FLAG: `0` = this chunk's
/// occupied bricks own probes (each brick's compact slot is in its [`BrickTile::probe_slot`]);
/// `u32::MAX` = fully covered by a finer LOD (its bricks own NO probes → apply/bounce fall to the finer
/// LOD). A cheap whole-chunk gate; the per-brick slot is exact. See `refresh_probe_bases`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChunkLookup {
    pub key_hi: u32,
    pub key_lo: u32,
    pub occ_lo: u32,
    pub occ_hi: u32,
    pub tile_run_base: u32,
    pub probe_base: u32,
}

/// `mat_atlas_base` sentinel for a SINGLE-material brick: it owns no material atlas tile (the
/// material is `palette[0]` everywhere), so the reader short-circuits before ever sampling it.
/// `u32::MAX` is not a valid `tile_atlas_base` packing (`col_px|row_px<<16`), so it can't collide.
pub const MAT_ATLAS_NONE: u32 = u32::MAX;

/// One resident brick's GPU record inside a chunk's tile run: its DISTANCE atlas tile origin, its
/// MATERIAL atlas tile origin (or [`MAT_ATLAS_NONE`] for a single-material brick — those store no
/// material tile), its packed 4-entry material palette (`pal01 = id0|id1<<16`, `pal23 = id2|id3<<16`),
/// and its DDGI `probe_slot` (the compact finest-resident probe slot, `u32::MAX` = no probe; set by
/// `refresh_probe_bases`). 5×u32 = 20 bytes. (Custom `Default` below for the `probe_slot`/`mat` sentinels.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrickTile {
    pub atlas_base: u32,
    pub mat_atlas_base: u32,
    pub pal01: u32,
    pub pal23: u32,
    pub probe_slot: u32,
}

impl Default for BrickTile {
    fn default() -> Self {
        // `probe_slot` defaults to the no-probe sentinel (not 0, which is a valid slot); `mat_atlas_base`
        // to the single-material sentinel — so an unassigned/empty slot never aliases a real probe or
        // material tile.
        Self {
            atlas_base: 0,
            mat_atlas_base: MAT_ATLAS_NONE,
            pal01: 0,
            pal23: 0,
            probe_slot: u32::MAX,
        }
    }
}

/// Pixel origin of atlas `tile` (tiles wrap into rows so the texture width stays bounded), packed
/// as `col_px | row_px<<16`. Single source of truth for the `atlas_base` packing, shared by the
/// render-world full rebuild and the incremental [`LiveChunkTables`] so both agree byte-for-byte.
/// The distance and material atlases use the SAME packing (same page width/height); a brick's
/// distance and material tiles come from independent allocators, so the two origins differ.
pub fn tile_atlas_base(tile: u32) -> u32 {
    let edge = BRICK_EDGE as u32;
    let tile_width = edge * edge; // 64
    let col_px = (tile % ATLAS_TILES_PER_ROW) * tile_width;
    let row_px = (tile / ATLAS_TILES_PER_ROW) * edge;
    col_px | (row_px << 16)
}

/// Pack a brick's distance tile + material tile + palette into its GPU [`BrickTile`] record.
/// `mat_tile` is `None` for a single-material brick (→ [`MAT_ATLAS_NONE`]).
pub fn pack_brick_tile(tile: u32, mat_tile: Option<u32>, palette: Palette) -> BrickTile {
    BrickTile {
        atlas_base: tile_atlas_base(tile),
        mat_atlas_base: mat_tile.map_or(MAT_ATLAS_NONE, tile_atlas_base),
        pal01: palette[0] as u32 | ((palette[1] as u32) << 16),
        pal23: palette[2] as u32 | ((palette[3] as u32) << 16),
        // No probe yet — `refresh_probe_bases` assigns the finest-resident slot after the bake.
        probe_slot: u32::MAX,
    }
}

/// True when `palette` names more than one material (slot 1 occupied). Palettes fill densely from
/// slot 0 (`edits::build_palette`), so this is exact. Single-material bricks skip material storage.
pub fn palette_is_multi(palette: Palette) -> bool {
    palette[1] != PALETTE_EMPTY
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
    // Assign finest-resident probe slots (the render world does this via `refresh_probe_lod` before
    // extract; the full-rebuild path must do it too so the directory rows carry `probe_base`).
    live.refresh_probe_bases(u32::MAX);
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

/// Stable key → dense slot allocator with a free-list (mirrors [`super::atlas::TileAllocator`]).
/// Reusing a freed slot before growing `next` keeps the slot space densely packed (bounded by the peak
/// live key count). Idempotent `alloc` → a key present across frames keeps the SAME slot. Used for both
/// the chunk → tile-run-region slot ([`ChunkSlotAllocator`]) and the per-brick DDGI probe slot
/// (keyed by `(ChunkKey, local)`).
pub struct SlotAllocator<K: std::hash::Hash + Eq> {
    slot_of: FxHashMap<K, u32>,
    free: Vec<u32>,
    next: u32,
}

impl<K: std::hash::Hash + Eq> Default for SlotAllocator<K> {
    fn default() -> Self {
        Self { slot_of: FxHashMap::default(), free: Vec::new(), next: 0 }
    }
}

impl<K: std::hash::Hash + Eq + Copy> SlotAllocator<K> {
    /// Assign (or return the existing) slot for `k`. Reuses a freed slot first.
    fn alloc(&mut self, k: K) -> u32 {
        if let Some(&s) = self.slot_of.get(&k) {
            return s;
        }
        let s = self.free.pop().unwrap_or_else(|| {
            let s = self.next;
            self.next += 1;
            s
        });
        self.slot_of.insert(k, s);
        s
    }

    /// Return `k`'s slot to the free pool (key gone). Its stale data is harmless — nothing live
    /// references the slot once freed. No-op if `k` isn't allocated (so release is idempotent).
    fn release(&mut self, k: &K) {
        if let Some(s) = self.slot_of.remove(k) {
            self.free.push(s);
        }
    }

    /// One past the largest slot ever handed out → how many slots the backing buffer must span.
    pub fn high_water(&self) -> u32 {
        self.next
    }

    /// Number of slots currently assigned (live keys) — distinct from `high_water` (which includes
    /// freed-but-not-reused slots).
    pub fn live_count(&self) -> usize {
        self.slot_of.len()
    }
}

/// Chunk → tile-run-region slot (`tile_run_base = slot * TILE_RUN_SLOT`).
pub type ChunkSlotAllocator = SlotAllocator<ChunkKey>;

/// One resident chunk's live state: its tile-run slot, occupancy mask, and the 64 brick
/// tiles (only `popcount(occ)` are live; the rest are never indexed by the shader).
struct ChunkEntry {
    slot: u32,
    /// DDGI finest-resident FLAG: `0` = this chunk's occupied bricks own probes (their per-brick slot is
    /// in [`BrickTile::probe_slot`]); `u32::MAX` = covered by a finer LOD (skip; apply/bounce use the
    /// finer LOD). A cheap whole-chunk gate; the actual slot is per-brick. Set by `refresh_probe_bases`;
    /// mirrored into the directory row + resident-row list.
    probe_base: u32,
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
    /// DDGI probe-slot allocator, keyed PER BRICK by `(ChunkKey, local)`, populated ONLY for the
    /// occupied bricks of **finest-resident** chunks (`refresh_probe_bases`). One compact, stable slot
    /// per finest-resident brick — EXACT (no intra-chunk waste, no all-LOD redundancy), so the probe
    /// buffer it sizes (`probe_high_water`) scales with the clipmap window, not the scene's absolute
    /// size. Each brick stores its slot in its [`BrickTile::probe_slot`].
    probe_alloc: SlotAllocator<(ChunkKey, u32)>,
    chunks: FxHashMap<ChunkKey, ChunkEntry>,
    /// Reverse of each live chunk's tile-run slot → its key, so a dirty slot resolves to its
    /// region in O(1) (only live chunks; a freed slot is dropped — its region is never indexed).
    slot_to_key: FxHashMap<u32, ChunkKey>,
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
    /// Chunks whose geometry changed since the last [`drain_wake_keys`](Self::drain_wake_keys) — the
    /// DDGI "wake" set. A change here (or in a neighbour) re-converges only those probes at the active
    /// rate while the rest of a settled scene stays dormant (localized wake, no global FPS cliff on edits).
    wake_keys: FxHashSet<ChunkKey>,
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
    /// Only the TILE-RUN buffer outgrew its capacity; the fixed-size directory did NOT change size.
    /// Rebuild the tile-run buffer (to `cap_slots`, with headroom) but keep the directory and apply
    /// just its dirty `row_updates` in place — so a tile-run grow during a fill no longer drags a
    /// full re-upload of the (tens-of-MB at large rings) directory. `tile_run` is UNPADDED.
    TileGrow {
        row_updates: Vec<(u32, ChunkLookup)>,
        tile_run: Vec<BrickTile>,
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
        probe_base: u32::MAX, // empty slot → never has probes (never read; key tag misses anyway)
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

/// Resolve `(ck, local)` through a GPU lookup-buffer pair EXACTLY as `brick.wgsl::find_chunk` +
/// `brick_in_chunk` do: direct-index the dense directory by [`dir_index`], accept only if the stored
/// key TAG matches (sentinel / a different wrapped chunk → miss), test the occupancy bit, then index
/// the tile run at `tile_run_base + popcount(bits strictly below the slot)`. THE single CPU mirror of
/// the shader's brick resolve — every differential test (the chunk churn, the bake-scheduler
/// lifecycle, the GPU rigs) routes through this instead of hand-copying the unpack, so a change to the
/// resolve contract lands in one place. `#[doc(hidden)] pub` so the `tests/` integration crate can
/// reach it; it is test-support, not real API (and `pub`, so no non-test dead-code warning).
#[doc(hidden)]
pub fn resolve_via_tables(
    rows: &[ChunkLookup],
    tiles: &[BrickTile],
    r: i32,
    ck: ChunkKey,
    local: u32,
) -> Option<BrickTile> {
    let idx = dir_index(ck, r);
    if idx >= rows.len() {
        return None;
    }
    let c = rows[idx];
    if (c.key_hi, c.key_lo) != chunk_gpu_key(ck) {
        return None; // sentinel slot, or a different chunk shares this `mod R` slot
    }
    let occ = (c.occ_lo as u64) | ((c.occ_hi as u64) << 32);
    if (occ >> local) & 1 == 0 {
        return None; // brick not resident in this chunk
    }
    let off = (occ & ((1u64 << local) - 1)).count_ones();
    Some(tiles[(c.tile_run_base + off) as usize])
}

impl LiveChunkTables {
    /// The resident chunk keys (one per chunk holding ≥1 brick) — O(chunks), for debug
    /// stats/overlays. This is the table's own key set, so it's FAR cheaper than re-deriving the set
    /// by scanning every resident BRICK (~261k on the stress scene = a ~7 ms/frame debug-panel hitch).
    pub fn resident_chunk_keys(&self) -> impl Iterator<Item = ChunkKey> + '_ {
        self.chunks.keys().copied()
    }

    /// The COMPACT list of resident chunk directory rows (one [`ChunkLookup`] per non-empty chunk,
    /// any order). The DDGI probe trace dispatches one workgroup per entry — `O(resident chunks)`
    /// (hundreds–thousands) instead of scanning the full `R³·lod_count` toroidal directory (millions
    /// of empty slots) every frame. Each row carries the key (→ decode to lod+coord), occupancy mask,
    /// and `tile_run_base` (→ the probe's storage slot), so the trace needs nothing else to enumerate
    /// every probe.
    pub fn resident_rows(&self) -> Vec<ChunkLookup> {
        self.chunks
            .iter()
            .map(|(ck, e)| {
                let (key_hi, key_lo) = chunk_gpu_key(*ck);
                ChunkLookup {
                    key_hi,
                    key_lo,
                    occ_lo: e.occ as u32,
                    occ_hi: (e.occ >> 32) as u32,
                    tile_run_base: e.slot * TILE_RUN_SLOT,
                    // Finest-resident probe slot (or u32::MAX), assigned by `refresh_probe_bases`.
                    probe_base: e.probe_base,
                }
            })
            .collect()
    }

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
                // Free the evicted chunk's per-brick probe slots before dropping it (idempotent if it
                // owned none — non-finest). The chunk left the window without an explicit clear, so this
                // is the only place its probe slots get reclaimed.
                if let Some(old) = self.chunks.remove(&old_ck) {
                    let mut bits = old.occ;
                    while bits != 0 {
                        let l = bits.trailing_zeros();
                        bits &= bits - 1;
                        self.probe_alloc.release(&(old_ck, l));
                    }
                }
                self.slots.release(&old_ck);
                self.dirty_slots.remove(&old_slot);
            }
        }

        let slot = self.slots.alloc(ck);
        let entry = self.chunks.entry(ck).or_insert_with(|| ChunkEntry {
            slot,
            probe_base: u32::MAX, // assigned by refresh_probe_bases (runs the same frame)
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
            // u32::MAX until `refresh_probe_bases` (runs the same frame, after all set/clear_brick)
            // assigns the finest-resident probe slot. Safe default: u32::MAX = no probes (apply/bounce
            // fall to a coarser LOD) rather than a wrong slot, if refresh somehow didn't run.
            probe_base: u32::MAX,
        };
        self.dirty_rows.insert(idx as u32);
        self.dirty_slots.insert(slot);
        self.wake_keys.insert(ck); // geometry changed here → re-converge this region's probes
    }

    /// Clear a brick from its chunk. If the chunk becomes empty, free its tile-run slot and reset its
    /// directory slot to the sentinel tag (so the GPU's tag compare misses it → coarse fallback).
    /// O(1) — no row shift. (Departed chunks are cleared here on exit, which is the make-before-break
    /// reclaim path that keeps the tile-run bounded; see the migration plan.)
    pub fn clear_brick(&mut self, ck: ChunkKey, local: u32) {
        let Some(entry) = self.chunks.get_mut(&ck) else {
            return;
        };
        self.wake_keys.insert(ck); // geometry changed here → re-converge this region's probes
        entry.occ &= !(1u64 << local);
        entry.tiles[local as usize] = BrickTile::default();
        // Free this brick's DDGI probe slot (the brick is gone; idempotent if it owned none). Disjoint
        // field borrow: `entry` borrows `self.chunks`, `self.probe_alloc` is a separate field.
        self.probe_alloc.release(&(ck, local));
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
        // The just-cleared brick's probe slot was already freed above; an empty chunk owns no others.
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

    /// Recompute the DDGI finest-resident probe assignment. For each resident chunk: if it is the FINEST
    /// resident LOD over its region, set its flag (`probe_base = 0`) and assign each occupied brick a
    /// COMPACT, stable per-brick probe slot (`BrickTile::probe_slot`) from the dedicated per-brick
    /// allocator — idempotent, so a brick that stays finest keeps the SAME slot across frames (its
    /// temporal probe history stays aligned → boil-free). Non-finest chunks flag `u32::MAX` and release
    /// their bricks' slots (their region is served by a finer LOD). Because slots are allocated only over
    /// finest-resident OCCUPIED bricks, the probe buffer they size is EXACT (no intra-chunk waste, no
    /// all-LOD redundancy) and scales with the clipmap window. Call in the MAIN world after all
    /// `set_brick`/`clear_brick` for the frame, whenever the chunk set changed (`topology_generation`);
    /// writes the directory row (→ `chunk_buf`, read by apply + bounce) and the entry's tiles
    /// (→ `resident_rows` / tile-run, read by the trace), marking changed rows/slots dirty for the delta.
    pub fn refresh_probe_bases(&mut self, halve_lod: u32) {
        let keys: Vec<ChunkKey> = self.chunks.keys().copied().collect();
        for ck in keys {
            let finest = self.is_finest(ck);
            let flag = if finest { 0u32 } else { u32::MAX };
            // DENSITY HALVING: at/above `halve_lod` the distant probes are decimated to a checkerboard of
            // bricks (~half), cutting probe count + ray work where the GI is low-frequency. The apply's
            // coverage-weighted trilinear fills the gaps. Below `halve_lod`, every occupied brick keeps one.
            let decimate = finest && ck.lod >= halve_lod;
            let idx = dir_index(ck, self.r);
            if self.dir[idx].probe_base != flag {
                self.dir[idx].probe_base = flag;
                self.dirty_rows.insert(idx as u32);
            }
            // Disjoint field borrows: `e` borrows `self.chunks`, `self.probe_alloc` is a separate field.
            let Some(e) = self.chunks.get_mut(&ck) else { continue };
            e.probe_base = flag;
            let mut bits = e.occ;
            let mut tiles_changed = false;
            while bits != 0 {
                let local = bits.trailing_zeros();
                bits &= bits - 1; // clear lowest set bit
                // Checkerboard on the brick's lattice coord (stable across frames → no probe churn).
                let keep = !decimate || {
                    let lx = (local % CHUNK_BRICKS as u32) as i32;
                    let ly = (local / CHUNK_BRICKS as u32 % CHUNK_BRICKS as u32) as i32;
                    let lz = (local / (CHUNK_BRICKS * CHUNK_BRICKS) as u32) as i32;
                    let b = ck.coord * CHUNK_BRICKS + IVec3::new(lx, ly, lz);
                    (b.x + b.y + b.z) & 1 == 0
                };
                let want = if finest && keep {
                    self.probe_alloc.alloc((ck, local))
                } else {
                    self.probe_alloc.release(&(ck, local));
                    u32::MAX
                };
                if e.tiles[local as usize].probe_slot != want {
                    e.tiles[local as usize].probe_slot = want;
                    tiles_changed = true;
                }
            }
            if tiles_changed {
                // `probe_slot` lives in the tile-run record → re-upload this chunk's region.
                self.dirty_slots.insert(e.slot);
            }
        }
    }

    /// One past the largest DDGI probe slot → the probe irradiance buffer must span `probe_high_water`
    /// per-brick probe slots (each then `× subdiv³ × PROBE_OCT_TEXELS` vec4s). Bounded by the count of
    /// finest-resident OCCUPIED bricks, so it scales with the clipmap WINDOW, not the all-LOD atlas tile
    /// union. The render world sizes the probe buffer from this (replacing `atlas.tiles.high_water()`).
    pub fn probe_high_water(&self) -> u32 {
        self.probe_alloc.high_water()
    }

    /// Count of finest-resident probes (= finest-resident occupied bricks) — for debug stats + the
    /// scaling harness.
    pub fn probe_count(&self) -> usize {
        self.probe_alloc.live_count()
    }

    /// Count of finest-resident chunks (flag `probe_base == 0`) — for debug stats.
    pub fn finest_chunk_count(&self) -> usize {
        self.chunks.values().filter(|e| e.probe_base != u32::MAX).count()
    }

    /// The finest-resident subset of [`resident_rows`](Self::resident_rows) — chunks owning probe blocks
    /// (`probe_base != u32::MAX`). The DDGI trace dispatches over THESE only, so its workgroup count is
    /// bounded by the finest-resident set (the clipmap window), not the all-LOD resident union.
    pub fn finest_rows(&self) -> Vec<ChunkLookup> {
        self.resident_rows().into_iter().filter(|c| c.probe_base != u32::MAX).collect()
    }

    /// `(ChunkKey, tile-run slot)` for every FINEST-resident chunk (those owning probe blocks). The
    /// render-world relevance cull needs each finest chunk's world position (decode the key via
    /// [`chunk_min_world`]) plus the `slot` the dispatch rotation keys on (= `tile_run_base / TILE_RUN_SLOT`).
    pub fn finest_slots_keyed(&self) -> Vec<(ChunkKey, u32)> {
        self.chunks
            .iter()
            .filter(|(_, e)| e.probe_base != u32::MAX)
            .map(|(ck, e)| (*ck, e.slot))
            .collect()
    }

    /// Take + clear the set of chunks whose geometry changed since the last call (the DDGI wake set).
    /// The render world expands these to a 1-chunk neighbourhood and re-converges only those probes at
    /// the active rate (localized wake — no global re-trace cliff on a single edit).
    pub fn drain_wake_keys(&mut self) -> Vec<ChunkKey> {
        self.wake_keys.drain().collect()
    }

    /// The tile-run slot of resident chunk `ck` (its stable per-chunk index), or None if not resident.
    /// Used to map a wake neighbourhood (chunk keys) to the slots the render-world dispatch rotates on.
    pub fn slot_of(&self, ck: ChunkKey) -> Option<u32> {
        self.chunks.get(&ck).map(|e| e.slot)
    }

    /// Whether `ck` is the FINEST resident LOD covering its region — some sub-region of it is NOT
    /// covered by a resident finer (LOD−1) chunk, so it must own probes there (else the recursive
    /// bounce, which returns the first resident probe LOD-0→coarse, would find none → black). LOD 0 is
    /// always finest. A LOD-L chunk spans a 2×2×2 block of LOD-(L−1) chunks (the clipmap's 2× per-LOD
    /// doubling), so it is fully covered iff all 8 finer children are resident.
    fn is_finest(&self, ck: ChunkKey) -> bool {
        if ck.lod == 0 {
            return true; // no finer LOD exists → always finest
        }
        // The 8 LOD-(L−1) children tiling this chunk's footprint (2× per-LOD doubling). If ANY child
        // is not resident, part of this chunk's region has no finer probe → keep this LOD's probes.
        let base = ck.coord * 2;
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let child = ChunkKey::new(ck.lod - 1, base + IVec3::new(dx, dy, dz));
                    if !self.chunks.contains_key(&child) {
                        return true;
                    }
                }
            }
        }
        false // fully covered by finer-resident chunks → no probes here (finer LOD serves the region)
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
        (self.dir.clone(), self.tile_run_dense())
    }

    /// The tile-run half of [`full_tables`] alone (`slot_high_water()*TILE_RUN_SLOT` entries, each
    /// live slot's region at `slot*TILE_RUN_SLOT`). Used by [`ChunkUpload::TileGrow`] so a tile-run
    /// capacity grow rebuilds ONLY the tile-run buffer — not the (unchanged, fixed-size, possibly
    /// tens-of-MB) directory, which keeps deltaing in place.
    pub fn tile_run_dense(&self) -> Vec<BrickTile> {
        let mut tile_run =
            vec![BrickTile::default(); (self.slot_high_water() * TILE_RUN_SLOT) as usize];
        for (&slot, ck) in &self.slot_to_key {
            let base = (slot * TILE_RUN_SLOT) as usize;
            tile_run[base..base + TILE_RUN_SLOT as usize]
                .copy_from_slice(&Self::dense_region(&self.chunks[ck]));
        }
        tile_run
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
        // Tile-run headroom: GROW BY DOUBLING (amortized). A `TileGrow`/`Full` rebuilds the WHOLE tile-run
        // (`tile_run_dense()` — O(all resident bricks), tens of MB on a huge scene = a ~200 ms hitch), so
        // it must be RARE. With doubling, growth costs O(log n) total rebuilds; once a scene's resident set
        // settles, the slack absorbs object-edit churn (a moved object frees one chunk slot + takes another
        // — the free-list keeps high-water stable) so edits stay cheap in-place Deltas, not grows. The
        // wasted VRAM is bounded by the clipmap window (×2 of tens of MB), well under the probe budget.
        let new_cap_slots = needed_slots.saturating_mul(2).max(needed_slots + TILE_RUN_SLOT);
        if cap_rows == 0 || needed_rows > cap_rows {
            // First upload, or the directory itself grew (only on a ring-size change — it's
            // FIXED-size otherwise, sized EXACTLY with no headroom). Rebuild BOTH buffers.
            let (rows, tile_run) = self.full_tables();
            ChunkUpload::Full { rows, tile_run, cap_rows: needed_rows, cap_slots: new_cap_slots }
        } else if needed_slots > cap_slots {
            // ONLY the tile-run outgrew its buffer. The directory is the same size — delta its dirty
            // rows in place and rebuild only the (much smaller) tile-run, NOT the whole directory.
            let row_updates = self.dirty_rows.iter().map(|&r| (r, self.lookup_at(r))).collect();
            ChunkUpload::TileGrow { row_updates, tile_run: self.tile_run_dense(), cap_slots: new_cap_slots }
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
pub fn resident_chunks(atlas: &SdfAtlas, _config: &SdfGridConfig) -> Vec<ChunkKey> {
    // The live chunk table already IS the deduped resident-chunk set (O(chunks)); deriving it by
    // scanning every resident brick (O(bricks)) made the SDF Accel debug panel a ~7 ms/frame hitch.
    atlas.live_chunks.resident_chunk_keys().collect()
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

    /// Resolve a brick through a built [`ChunkTables`] the way the shader does — a thin convenience
    /// over [`resolve_via_tables`] (the single shader mirror) that first maps the brick to its
    /// `(chunk, local)` via [`chunk_of`].
    fn shader_resolve(
        tables: &ChunkTables,
        config: &SdfGridConfig,
        brick: BrickKey,
    ) -> Option<BrickTile> {
        let (ck, li) = chunk_of(brick, config); // li = local slot 0..63
        resolve_via_tables(&tables.chunks, &tables.tile_run, tables.r, ck, li)
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
            BrickTile { atlas_base: base, mat_atlas_base: base ^ 0x3333, pal01: base ^ 0x1111, pal23: base ^ 0x2222, ..Default::default() }
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
            // Compare the tile-resolution fields only; `probe_slot` is assigned by `refresh_probe_bases`
            // (not by `tile_of`), so the full-struct equality would spuriously differ.
            let want = tile_of(k);
            assert_eq!(
                (got.atlas_base, got.pal01, got.pal23),
                (want.atlas_base, want.pal01, want.pal23),
                "brick {k:?} resolved to the wrong tile"
            );
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
        let tile = BrickTile { atlas_base: 7, mat_atlas_base: 9, pal01: 0, pal23: 0, ..Default::default() };
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

    /// finest-resident probe filter (the LOD-aware allocation): a coarse chunk whose 8 finer (LOD−1)
    /// children are ALL resident owns NO probes (`probe_base = u32::MAX`) — the finer LOD serves its
    /// region; a chunk missing any finer child keeps its probes (else the recursive bounce finds none
    /// → black, the dark-hole class); LOD 0 is always finest; evicting a finer child restores the
    /// coarse chunk's probes. This is the cheapest guard on the finest semantics, before any GPU gate.
    #[test]
    fn finest_resident_probe_assignment() {
        let cfg = config();
        let r = cfg.ring_chunks_per_axis();
        let tile = BrickTile { atlas_base: 1, pal01: 0, pal23: 0, ..Default::default() };
        let mut live = LiveChunkTables::default();

        // A LOD-1 chunk and ALL 8 of its LOD-0 children (coords 2·coord + {0,1}³).
        let c1 = ChunkKey::new(1, IVec3::new(0, 0, 0));
        live.set_brick(c1, 0, tile, &cfg);
        let mut children = Vec::new();
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let child = ChunkKey::new(0, IVec3::new(dx, dy, dz));
                    live.set_brick(child, 0, tile, &cfg);
                    children.push(child);
                }
            }
        }
        live.refresh_probe_bases(u32::MAX);
        let probe_base = |live: &LiveChunkTables, ck: ChunkKey| live.lookup_at(dir_index(ck, r) as u32).probe_base;

        // LOD-1 is fully covered by its 8 finer children → owns no probes.
        assert_eq!(probe_base(&live, c1), u32::MAX, "fully-covered coarse chunk must own no probes");
        // Each LOD-0 child is finest (no finer LOD) → owns probes.
        for &child in &children {
            assert_ne!(probe_base(&live, child), u32::MAX, "a LOD-0 chunk is always finest");
        }

        // Evict one finer child → the LOD-1 chunk is no longer fully covered → it regains probes.
        live.clear_brick(children[0], 0);
        live.refresh_probe_bases(u32::MAX);
        assert_ne!(
            probe_base(&live, c1),
            u32::MAX,
            "coarse chunk must regain probes when a finer child leaves (no dark hole)"
        );
    }

    /// RAPID-EDIT PERF: the STABLE probe-slot allocator must keep `refresh_probe_bases`'s per-edit
    /// directory delta proportional to the EDIT footprint — NOT the whole world — so rapid large edits
    /// don't trigger a full directory re-upload (the perf concern). Build a large nested multi-LOD
    /// scene, evict a slab, and assert the resulting dirty-row delta is ≪ the total chunk count and the
    /// scan stays well-bounded. (A naive index-compaction would renumber every chunk → delta == total.)
    #[test]
    fn rapid_edit_keeps_probe_delta_proportional() {
        // ring 128 → R=32 chunks/axis, so coords 0..16 don't wrap the toroidal directory.
        let cfg = SdfGridConfig {
            lod_count: 5,
            ring_bricks: 128,
            recenter_snap_chunks: 1,
            ..Default::default()
        };
        let tile = BrickTile { atlas_base: 1, pal01: 0, pal23: 0, ..Default::default() };
        let mut live = LiveChunkTables::default();
        let n = 16i32; // 16³ LOD-0 + 8³+4³+2³+1³ ancestors ≈ 4681 chunks
        for lod in 0..cfg.lod_count {
            let m = n >> lod;
            for z in 0..m {
                for y in 0..m {
                    for x in 0..m {
                        live.set_brick(ChunkKey::new(lod, IVec3::new(x, y, z)), 0, tile, &cfg);
                    }
                }
            }
        }
        live.refresh_probe_bases(u32::MAX);
        let total = live.chunks.len();
        assert!(total > 4000, "expected a large nested scene (got {total})");

        // Rapid large edit: evict the x=0 LOD-0 slab (n² = 256 chunks). Their LOD-1 parents lose a
        // child → flip to finest → re-allocated. Delta = evicted rows + flipped parents (∝ the edit).
        live.clear_dirty();
        for z in 0..n {
            for y in 0..n {
                live.clear_brick(ChunkKey::new(0, IVec3::new(0, y, z)), 0);
            }
        }
        let t = std::time::Instant::now();
        live.refresh_probe_bases(u32::MAX);
        let elapsed = t.elapsed();
        let delta = live.dirty_rows.len();
        eprintln!(
            "rapid-edit: {total} chunks, evicted {} → refresh {elapsed:?}, delta {delta} rows",
            n * n
        );

        // The delta tracks the EDIT (evicted slab + flipped parents), not the world → small upload.
        assert!(
            delta < total / 4,
            "probe delta ({delta}) must be ≪ total chunks ({total}) — a small edit must not re-upload the world"
        );
        // Bounded scan even at scale (generous ceiling — catastrophic-regression guard, not micro-perf).
        assert!(
            elapsed.as_millis() < 100,
            "refresh after a large edit must stay well-bounded (was {elapsed:?})"
        );
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
            BrickTile { atlas_base: base, mat_atlas_base: base ^ 0x3333, pal01: base ^ 0x1111, pal23: base ^ 0x2222, ..Default::default() }
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

        // Resolve (ck, local) through `resolve_via_tables` — the single shader mirror.
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
                        mat_atlas_base: (r >> 8) as u32 ^ 0x7777,
                        pal01: r as u32 ^ 0xAAAA,
                        pal23: (r >> 16) as u32 ^ 0x5555,
                        ..Default::default()
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
                ChunkUpload::TileGrow { row_updates, tile_run, cap_slots: cs } => {
                    cap_slots = cs;
                    for (row, look) in row_updates {
                        gpu_rows[row as usize] = look; // directory delta (size unchanged)
                    }
                    gpu_tiles = tile_run; // tile-run rebuild
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
                match resolve_via_tables(&gpu_rows, &gpu_tiles, r, ck, local) {
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
                    resolve_via_tables(&gpu_rows, &gpu_tiles, r, probe_ck, probe_local).is_none(),
                    "frame {frame}: absent brick {probe_ck:?} local {probe_local} wrongly resolved"
                );
            }
        }
    }
}
