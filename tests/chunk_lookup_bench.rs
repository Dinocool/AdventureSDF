//! Head-to-head profiling rig for the GPU chunk-lookup data structure.
//!
//! WHY THIS EXISTS: the per-frame chunk-table maintenance spikes to ~448 ms while flying.
//! Root cause is structural — the resident chunk lookup is a SPARSE SORTED ARRAY
//! (`sorted_keys: Vec` + `key_to_row: HashMap`, binary-searched on the GPU). Every chunk
//! insert/remove is O(resident): a `Vec` splice plus a `key_to_row` re-stamp over the tail.
//! When a coarse-LOD recenter exits a shell of thousands of chunks, the cost is
//! O(shell × resident) = the spike.
//!
//! This rig benchmarks candidate structures over a COMMON `ChunkLookup` trait, driven by a
//! production-shaped fly-path workload (R=32 chunks/axis, lod_count=8, recenter snap of 2
//! chunks, sparse ~surface occupancy). It mirrors the real addressing math in `chunk.rs`
//! (`chunk_gpu_key` lexicographic packing) and the real `set_brick`/`clear_brick` splice
//! semantics so the sorted-array baseline reproduces the O(resident) per-mutation cost.
//!
//! It is clean-room and self-contained: minimal faithful versions of the renderer types, no
//! renderer imports, no new dependencies. Run with:
//!
//! ```text
//! CARGO_INCREMENTAL=0 cargo test --test chunk_lookup_bench --release -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------
// Production constants (mirror src/sdf_render/chunk.rs + mod.rs defaults).
// ---------------------------------------------------------------------------------------

/// Bricks per axis in one chunk (chunk.rs CHUNK_BRICKS).
const CHUNK_BRICKS: i32 = 4;
/// Brick slots per chunk (chunk.rs CHUNK_VOLUME = 64).
const CHUNK_VOLUME: u32 = (CHUNK_BRICKS * CHUNK_BRICKS * CHUNK_BRICKS) as u32;
/// Key bias so a signed chunk axis fits the 16-bit packed key field (chunk.rs KEY_BIAS).
const KEY_BIAS: i32 = 1 << 15;
/// Tile-run entries reserved per resident chunk (chunk.rs TILE_RUN_SLOT == CHUNK_VOLUME).
const TILE_RUN_SLOT: u32 = CHUNK_VOLUME;

/// Ring chunks per axis = ring_bricks(128) / CHUNK_BRICKS(4). The production window edge.
const R: i32 = 32;
/// LOD levels (mod.rs DEFAULT_LOD_COUNT).
const LOD_COUNT: u32 = 8;
/// Recenter hysteresis in whole chunks (mod.rs DEFAULT_RECENTER_SNAP_CHUNKS).
const RECENTER_SNAP_CHUNKS: i32 = 2;

// ---------------------------------------------------------------------------------------
// Minimal faithful renderer types.
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct IVec3 {
    x: i32,
    y: i32,
    z: i32,
}
impl IVec3 {
    const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

/// Absolute chunk identity: LOD + chunk coord on that LOD's lattice (chunk.rs ChunkKey).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct ChunkKey {
    lod: u32,
    coord: IVec3,
}
impl ChunkKey {
    fn new(lod: u32, coord: IVec3) -> Self {
        Self { lod, coord }
    }
}

/// One resident brick's GPU record (chunk.rs BrickTile, 12 B). Distinct values per insert so
/// a wrong-tile resolve would be detectable; here it is just the payload moved on mutation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BrickTile {
    atlas_base: u32,
    pal01: u32,
    pal23: u32,
}

// The GPU chunk-lookup row (chunk.rs ChunkLookup) is 20 B — key_hi/key_lo/occ_lo/occ_hi/
// tile_run_base. Here that row's *fields* are modelled directly on each structure: the sorted
// array carries them as a ChunkEntry, the toroidal variants as a DirSlot/DirEntry (where
// key_* becomes a validity tag, not a sort key). The 20-byte GPU size is used in memory_bytes.

/// The absolute 64-bit GPU key, packed lexicographically so a sort/binary-search by
/// `(key_hi, key_lo)` orders by lod, then x, y, z. EXACT mirror of chunk.rs::chunk_gpu_key —
/// the CPU<->GPU contract the GPU rig (tests/sdf_gpu_rig.rs) guards.
fn chunk_gpu_key(key: ChunkKey) -> (u32, u32) {
    let cx = ((key.coord.x + KEY_BIAS) as u32) & 0xffff;
    let cy = ((key.coord.y + KEY_BIAS) as u32) & 0xffff;
    let cz = ((key.coord.z + KEY_BIAS) as u32) & 0xffff;
    let key_hi = (key.lod << 16) | cx;
    let key_lo = (cy << 16) | cz;
    (key_hi, key_lo)
}

// ---------------------------------------------------------------------------------------
// Common trait every candidate implements. A "chunk" here owns up to CHUNK_VOLUME bricks;
// mutation is at brick granularity (set/clear), exactly like LiveChunkTables. `lookup`
// resolves a (chunk, local) to its tile the way the march does each step.
// ---------------------------------------------------------------------------------------

trait ChunkStructure {
    /// Mark a brick present in its chunk (insert chunk on first brick, update otherwise).
    fn set_brick(&mut self, ck: ChunkKey, local: u32, tile: BrickTile);
    /// Clear a brick; drop the chunk when it empties (mirrors clear_brick).
    fn clear_brick(&mut self, ck: ChunkKey, local: u32);
    /// Resolve a brick to its tile exactly as the GPU march does (find chunk -> occ bit ->
    /// popcount-ranked tile-run index). `None` if the brick isn't resident.
    fn lookup(&self, ck: ChunkKey, local: u32) -> Option<BrickTile>;
    /// Resident-set memory footprint in bytes (the structure's own backing storage).
    fn memory_bytes(&self) -> usize;
    /// Per-frame delta bookkeeping reset (mirrors clear_dirty). Default no-op.
    fn end_frame(&mut self) {}
    /// Update the per-LOD window origins (only the toroidal variants need it).
    fn set_window_origins(&mut self, _origins: &[IVec3; LOD_COUNT as usize]) {}
}

/// Densely pack a chunk's `popcount(occ)` live bricks (ascending local order) into a tile-run
/// region — the layout the shader indexes by rank. Shared by every candidate.
fn dense_region(occ: u64, tiles: &[BrickTile; CHUNK_VOLUME as usize]) -> [BrickTile; CHUNK_VOLUME as usize] {
    let mut region = [BrickTile::default(); CHUNK_VOLUME as usize];
    let mut rank = 0usize;
    let mut bits = occ;
    while bits != 0 {
        let local = bits.trailing_zeros() as usize;
        region[rank] = tiles[local];
        rank += 1;
        bits &= bits - 1;
    }
    region
}

/// The popcount-ranked offset of `local` within `occ` (shader's `tile_run_base + popcount(below)`).
#[inline]
fn rank_of(occ: u64, local: u32) -> Option<u32> {
    if (occ >> local) & 1 == 0 {
        return None;
    }
    Some((occ & ((1u64 << local) - 1)).count_ones())
}

// ---------------------------------------------------------------------------------------
// Candidate 1: SORTED-ARRAY baseline. Faithful mirror of chunk.rs LiveChunkTables —
// sorted Vec of keys + key_to_row HashMap, partition_point insert, Vec splice + re-stamp
// on remove. Reproduces the O(resident) per-mutation cost that is the 448 ms spike.
// ---------------------------------------------------------------------------------------

struct ChunkEntry {
    slot: u32,
    occ: u64,
    tiles: [BrickTile; CHUNK_VOLUME as usize],
}

/// Free-list slot allocator (chunk.rs ChunkSlotAllocator) — tile-run regions are slot-addressed
/// so brick churn in one chunk never shifts another's base.
#[derive(Default)]
struct SlotAllocator {
    slot_of: HashMap<ChunkKey, u32>,
    free: Vec<u32>,
    next: u32,
}
impl SlotAllocator {
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
    fn release(&mut self, ck: &ChunkKey) {
        if let Some(s) = self.slot_of.remove(ck) {
            self.free.push(s);
        }
    }
    fn high_water(&self) -> u32 {
        self.next
    }
}

#[derive(Default)]
struct SortedArray {
    slots: SlotAllocator,
    chunks: HashMap<ChunkKey, ChunkEntry>,
    slot_to_key: HashMap<u32, ChunkKey>,
    /// Resident keys in ascending chunk_gpu_key order — the binary-searched GPU row order.
    sorted_keys: Vec<ChunkKey>,
    key_to_row: HashMap<ChunkKey, u32>,
    // Delta bookkeeping (kept so the splice + re-stamp cost is paid exactly as in prod).
    dirty_rows: std::collections::BTreeSet<u32>,
    dirty_slots: std::collections::BTreeSet<u32>,
    sentinel_tail_from: Option<u32>,
    structure_changed: bool,
}

impl ChunkStructure for SortedArray {
    fn set_brick(&mut self, ck: ChunkKey, local: u32, tile: BrickTile) {
        let bit = 1u64 << local;
        if let Some(entry) = self.chunks.get_mut(&ck) {
            entry.occ |= bit;
            entry.tiles[local as usize] = tile;
            let slot = entry.slot;
            let row = self.key_to_row[&ck];
            self.dirty_rows.insert(row);
            self.dirty_slots.insert(slot);
            return;
        }
        // New chunk: allocate slot, splice into the sorted row array, re-stamp shifted rows.
        let slot = self.slots.alloc(ck);
        let mut tiles = [BrickTile::default(); CHUNK_VOLUME as usize];
        tiles[local as usize] = tile;
        self.chunks.insert(ck, ChunkEntry { slot, occ: bit, tiles });
        self.slot_to_key.insert(slot, ck);

        let key = chunk_gpu_key(ck);
        let row = self.sorted_keys.partition_point(|k| chunk_gpu_key(*k) < key) as u32;
        self.sorted_keys.insert(row as usize, ck); // O(resident) Vec splice
        // Every row at/after the insert shifted up → re-stamp + mark dirty (O(tail)).
        for (i, k) in self.sorted_keys.iter().enumerate().skip(row as usize) {
            self.key_to_row.insert(*k, i as u32);
            self.dirty_rows.insert(i as u32);
        }
        self.dirty_slots.insert(slot);
        if self.sentinel_tail_from.is_some() {
            self.sentinel_tail_from = Some(self.sorted_keys.len() as u32);
        }
        self.structure_changed = true;
    }

    fn clear_brick(&mut self, ck: ChunkKey, local: u32) {
        let Some(entry) = self.chunks.get_mut(&ck) else {
            return;
        };
        entry.occ &= !(1u64 << local);
        entry.tiles[local as usize] = BrickTile::default();
        let slot = entry.slot;
        self.dirty_slots.insert(slot);
        if entry.occ != 0 {
            let row = self.key_to_row[&ck];
            self.dirty_rows.insert(row);
            return;
        }
        // Chunk emptied → drop it. Splice out + re-stamp the tail (O(resident)).
        self.dirty_slots.remove(&slot);
        self.chunks.remove(&ck);
        self.slot_to_key.remove(&slot);
        self.slots.release(&ck);
        let row = self.key_to_row.remove(&ck).expect("resident chunk has a row") as usize;
        self.sorted_keys.remove(row); // O(resident) Vec splice
        for (i, k) in self.sorted_keys.iter().enumerate().skip(row) {
            self.key_to_row.insert(*k, i as u32);
            self.dirty_rows.insert(i as u32);
        }
        let new_len = self.sorted_keys.len() as u32;
        self.dirty_rows.remove(&new_len);
        self.sentinel_tail_from = Some(new_len);
        self.structure_changed = true;
    }

    fn lookup(&self, ck: ChunkKey, local: u32) -> Option<BrickTile> {
        // GPU path: binary-search the sorted rows by gpu key, then occ-bit + dense rank.
        let key = chunk_gpu_key(ck);
        let row = self
            .sorted_keys
            .binary_search_by(|k| chunk_gpu_key(*k).cmp(&key))
            .ok()?;
        let entry = &self.chunks[&self.sorted_keys[row]];
        let off = rank_of(entry.occ, local)?;
        // tile_run_base = slot * TILE_RUN_SLOT; resolve into the dense region by rank.
        Some(dense_region(entry.occ, &entry.tiles)[off as usize])
    }

    fn memory_bytes(&self) -> usize {
        // chunk rows (20B GPU) live as ChunkEntry on CPU; tile-run buffer is slot-addressed.
        let n = self.chunks.len();
        let entry_bytes = n * std::mem::size_of::<ChunkEntry>();
        let sorted = self.sorted_keys.capacity() * std::mem::size_of::<ChunkKey>();
        let k2r = self.key_to_row.capacity() * (std::mem::size_of::<ChunkKey>() + 4);
        let s2k = self.slot_to_key.capacity() * (4 + std::mem::size_of::<ChunkKey>());
        // GPU-side buffers the structure forces: chunk_buf (rows × 24B) + tile_run
        // (high_water × TILE_RUN_SLOT × 12B).
        let gpu_chunk_buf = n * 24;
        let gpu_tile_run = self.slots.high_water() as usize * TILE_RUN_SLOT as usize * 12;
        entry_bytes + sorted + k2r + s2k + gpu_chunk_buf + gpu_tile_run
    }

    fn end_frame(&mut self) {
        self.dirty_rows.clear();
        self.dirty_slots.clear();
        self.sentinel_tail_from = None;
        self.structure_changed = false;
    }
}

// ---------------------------------------------------------------------------------------
// Candidate 2: PER-LOD TOROIDAL DIRECTORY. Per LOD a dense R³ array; chunk coord c lives at
// fixed slot (c mod R) (component-wise rem_euclid), flattened + lod*R³. Each slot stores a
// ChunkLookup whose key_* is a VALIDITY TAG (not a sort key). Eviction is free: a leaving
// chunk's slot == an entering chunk's slot (mod R wrap) so the new chunk overwrites it; a
// departed chunk is never read because in_ring_chunk rejects out-of-window coords.
//
// This variant stores tile data INLINE in the directory (occ + 64 tiles per slot) — the
// simplest faithful prototype. Directory VRAM here would be R³·lod·(20B + 64·12B); the
// hybrid below moves the tile run to a sparse free-list to shrink that.
// ---------------------------------------------------------------------------------------

#[inline]
fn rem_euclid(a: i32, b: i32) -> i32 {
    a.rem_euclid(b)
}

#[inline]
fn dir_index(coord: IVec3, lod: u32) -> usize {
    let sx = rem_euclid(coord.x, R) as usize;
    let sy = rem_euclid(coord.y, R) as usize;
    let sz = rem_euclid(coord.z, R) as usize;
    let within = sz * (R * R) as usize + sy * R as usize + sx;
    lod as usize * (R * R * R) as usize + within
}

/// One inline directory slot: validity tag + occupancy + the 64 brick tiles.
#[derive(Clone)]
struct DirSlot {
    /// Validity tag = chunk_gpu_key of the chunk that owns this slot. (0,0) sentinel = empty.
    tag: (u32, u32),
    occ: u64,
    tiles: [BrickTile; CHUNK_VOLUME as usize],
}
impl Default for DirSlot {
    fn default() -> Self {
        Self { tag: (0, 0), occ: 0, tiles: [BrickTile::default(); CHUNK_VOLUME as usize] }
    }
}

struct ToroidalDirectory {
    dir: Vec<DirSlot>,
    origins: [IVec3; LOD_COUNT as usize],
    /// Live resident chunk count (tag set + occ != 0) — for the row/memory report.
    live: usize,
}
impl Default for ToroidalDirectory {
    fn default() -> Self {
        let n = (R * R * R) as usize * LOD_COUNT as usize;
        Self {
            dir: vec![DirSlot::default(); n],
            origins: [IVec3::new(0, 0, 0); LOD_COUNT as usize],
            live: 0,
        }
    }
}

impl ToroidalDirectory {
    /// in_ring_chunk: is `c` inside this LOD's R³ window? (mirrors brick.wgsl::in_ring_chunk).
    fn in_window(&self, c: IVec3, lod: u32) -> bool {
        let o = self.origins[lod as usize];
        let rel = IVec3::new(c.x - o.x, c.y - o.y, c.z - o.z);
        rel.x >= 0 && rel.y >= 0 && rel.z >= 0 && rel.x < R && rel.y < R && rel.z < R
    }
}

impl ChunkStructure for ToroidalDirectory {
    fn set_brick(&mut self, ck: ChunkKey, local: u32, tile: BrickTile) {
        let idx = dir_index(ck.coord, ck.lod);
        let tag = chunk_gpu_key(ck);
        let slot = &mut self.dir[idx];
        let was_live = slot.tag == tag && slot.occ != 0;
        if slot.tag != tag {
            // A different chunk owned this slot (or it was empty) → overwrite = free eviction.
            slot.tiles = [BrickTile::default(); CHUNK_VOLUME as usize];
            slot.occ = 0;
            slot.tag = tag;
        }
        slot.occ |= 1u64 << local;
        slot.tiles[local as usize] = tile;
        if !was_live && slot.occ != 0 {
            self.live += 1;
        }
    }

    fn clear_brick(&mut self, ck: ChunkKey, local: u32) {
        let idx = dir_index(ck.coord, ck.lod);
        let tag = chunk_gpu_key(ck);
        let slot = &mut self.dir[idx];
        if slot.tag != tag {
            return; // slot owned by another chunk; nothing of ours to clear (free eviction)
        }
        let was_live = slot.occ != 0;
        slot.occ &= !(1u64 << local);
        slot.tiles[local as usize] = BrickTile::default();
        if was_live && slot.occ == 0 {
            self.live -= 1;
            slot.tag = (0, 0); // emptied → mark slot free (optional; in_window guards anyway)
        }
    }

    fn lookup(&self, ck: ChunkKey, local: u32) -> Option<BrickTile> {
        // GPU path: in_ring_chunk guard, then O(1) slot read + tag compare.
        if !self.in_window(ck.coord, ck.lod) {
            return None;
        }
        let idx = dir_index(ck.coord, ck.lod);
        let slot = &self.dir[idx];
        if slot.tag != chunk_gpu_key(ck) {
            return None; // stale trailing chunk / empty → coarse fallback
        }
        let off = rank_of(slot.occ, local)?;
        Some(dense_region(slot.occ, &slot.tiles)[off as usize])
    }

    fn memory_bytes(&self) -> usize {
        // The whole dense directory is resident VRAM regardless of occupancy.
        self.dir.len() * std::mem::size_of::<DirSlot>()
    }

    fn set_window_origins(&mut self, origins: &[IVec3; LOD_COUNT as usize]) {
        self.origins = *origins;
    }
}

// ---------------------------------------------------------------------------------------
// Candidate 3: HYBRID — toroidal directory (compact 20B slot) + SPARSE tile-run free-list.
// The recommended structure. The directory holds only the 20B ChunkLookup (tag/occ/base);
// the 64-tile run lives in a slot-allocated buffer (like chunk.rs ChunkSlotAllocator), freed
// when its directory slot is overwritten. Directory VRAM = R³·lod·20B ≈ 5.2 MB; tile-run VRAM
// scales with resident chunks only. Eviction stays free (overwrite); the only extra work vs.
// the inline directory is freeing the departed chunk's tile-run slot on overwrite.
// ---------------------------------------------------------------------------------------

/// Compact directory entry: validity tag, occupancy, tile-run slot index (base = slot*64).
#[derive(Clone, Copy)]
struct DirEntry {
    tag: (u32, u32),
    occ: u64,
    /// Tile-run slot (u32::MAX = none). base = run_slot * TILE_RUN_SLOT.
    run_slot: u32,
}
impl Default for DirEntry {
    fn default() -> Self {
        // run_slot MUST default to u32::MAX (no tile-run), NOT 0 — a derived Default would make
        // every untouched slot claim run-slot 0, corrupting the free-list on first overwrite.
        Self { tag: (0, 0), occ: 0, run_slot: u32::MAX }
    }
}

struct ToroidalHybrid {
    dir: Vec<DirEntry>,
    /// Sparse tile-run buffer, indexed by run_slot * TILE_RUN_SLOT (stored sparse-by-local;
    /// the GPU upload would densify — `lookup` densifies on read, same as the sorted array).
    tile_runs: Vec<[BrickTile; CHUNK_VOLUME as usize]>,
    free_runs: Vec<u32>,
    run_high_water: u32,
    origins: [IVec3; LOD_COUNT as usize],
    live: usize,
    /// Free a departed chunk's tile-run slot when its directory slot is overwritten. TRUE = the fix
    /// (bounded high-water); FALSE models blocker 3 (the leak) for `adversarial_tilerun_leaks_*`.
    free_on_overwrite: bool,
}
impl Default for ToroidalHybrid {
    fn default() -> Self {
        let n = (R * R * R) as usize * LOD_COUNT as usize;
        Self {
            dir: vec![DirEntry::default(); n],
            tile_runs: Vec::new(),
            free_runs: Vec::new(),
            run_high_water: 0,
            origins: [IVec3::new(0, 0, 0); LOD_COUNT as usize],
            live: 0,
            free_on_overwrite: true,
        }
    }
}
impl ToroidalHybrid {
    fn alloc_run(&mut self) -> u32 {
        if let Some(s) = self.free_runs.pop() {
            return s;
        }
        let s = self.run_high_water;
        self.run_high_water += 1;
        if self.tile_runs.len() < self.run_high_water as usize {
            self.tile_runs
                .push([BrickTile::default(); CHUNK_VOLUME as usize]);
        }
        s
    }
    fn in_window(&self, c: IVec3, lod: u32) -> bool {
        let o = self.origins[lod as usize];
        let rel = IVec3::new(c.x - o.x, c.y - o.y, c.z - o.z);
        rel.x >= 0 && rel.y >= 0 && rel.z >= 0 && rel.x < R && rel.y < R && rel.z < R
    }
}

impl ChunkStructure for ToroidalHybrid {
    fn set_brick(&mut self, ck: ChunkKey, local: u32, tile: BrickTile) {
        let idx = dir_index(ck.coord, ck.lod);
        let tag = chunk_gpu_key(ck);
        let entry = self.dir[idx];
        let (mut occ, mut run_slot, was_live) = if entry.tag == tag && entry.run_slot != u32::MAX {
            (entry.occ, entry.run_slot, entry.occ != 0)
        } else {
            // Free eviction: the slot belonged to a departed chunk. Reclaim its tile-run (unless we
            // are modelling blocker 3, the leak, with free_on_overwrite=false).
            if self.free_on_overwrite && entry.run_slot != u32::MAX {
                self.free_runs.push(entry.run_slot);
            }
            (0u64, u32::MAX, false)
        };
        if run_slot == u32::MAX {
            run_slot = self.alloc_run();
            self.tile_runs[run_slot as usize] = [BrickTile::default(); CHUNK_VOLUME as usize];
        }
        occ |= 1u64 << local;
        self.tile_runs[run_slot as usize][local as usize] = tile;
        self.dir[idx] = DirEntry { tag, occ, run_slot };
        if !was_live && occ != 0 {
            self.live += 1;
        }
    }

    fn clear_brick(&mut self, ck: ChunkKey, local: u32) {
        let idx = dir_index(ck.coord, ck.lod);
        let tag = chunk_gpu_key(ck);
        let entry = self.dir[idx];
        if entry.tag != tag || entry.run_slot == u32::MAX {
            return;
        }
        let was_live = entry.occ != 0;
        let occ = entry.occ & !(1u64 << local);
        self.tile_runs[entry.run_slot as usize][local as usize] = BrickTile::default();
        if occ == 0 {
            // Emptied → free the tile-run slot, clear the directory entry.
            self.free_runs.push(entry.run_slot);
            self.dir[idx] = DirEntry { tag: (0, 0), occ: 0, run_slot: u32::MAX };
            if was_live {
                self.live -= 1;
            }
        } else {
            self.dir[idx].occ = occ;
        }
    }

    fn lookup(&self, ck: ChunkKey, local: u32) -> Option<BrickTile> {
        if !self.in_window(ck.coord, ck.lod) {
            return None;
        }
        let idx = dir_index(ck.coord, ck.lod);
        let entry = self.dir[idx];
        if entry.tag != chunk_gpu_key(ck) || entry.run_slot == u32::MAX {
            return None;
        }
        let off = rank_of(entry.occ, local)?;
        let region = dense_region(entry.occ, &self.tile_runs[entry.run_slot as usize]);
        Some(region[off as usize])
    }

    fn memory_bytes(&self) -> usize {
        // Directory (compact 20B-equivalent entries) + sparse tile-run buffer (high water).
        let dir = self.dir.len() * std::mem::size_of::<DirEntry>();
        let gpu_dir = self.dir.len() * 24; // 24B ChunkLookup on the GPU
        let gpu_tile_run = self.run_high_water as usize * TILE_RUN_SLOT as usize * 12;
        dir.max(gpu_dir) + gpu_tile_run
    }

    fn set_window_origins(&mut self, origins: &[IVec3; LOD_COUNT as usize]) {
        self.origins = *origins;
    }
}

// ---------------------------------------------------------------------------------------
// Workload — a production-shaped camera fly-path.
//
// Each "frame" the camera advances; per LOD the window origin snaps to the recenter lattice.
// When an origin moves, the entered/exited chunk shells are the slab difference of the two R³
// windows (mirrors for_each_entered_chunk / for_each_exited_chunk). Coarse LODs snap rarely
// (their world step is 2^L larger), fine LODs snap most frames — the realistic pattern.
//
// Occupancy is SPARSE: only chunks an isosurface passes through are resident. We model the
// surface as a height field (a chunk is resident iff its coord is near `surface_y(x,z)`),
// giving a thin ~2D shell of resident chunks per window (a few % of R³) — like real terrain.
// ---------------------------------------------------------------------------------------

/// Window origin at this LOD for an integer camera chunk position, snapped to the recenter
/// lattice (mirrors ring_chunk_origin: cam_chunk >> lod, snap, minus half-ring).
fn window_origin(cam_chunk_lod0: IVec3, lod: u32) -> IVec3 {
    // Coarser LODs cover 2^lod the world; the camera's chunk coord at that LOD scales down.
    let div = 1i32 << lod;
    let cc = IVec3::new(
        cam_chunk_lod0.x.div_euclid(div),
        cam_chunk_lod0.y.div_euclid(div),
        cam_chunk_lod0.z.div_euclid(div),
    );
    let snap = RECENTER_SNAP_CHUNKS;
    let snapped = IVec3::new(
        cc.x.div_euclid(snap) * snap,
        cc.y.div_euclid(snap) * snap,
        cc.z.div_euclid(snap) * snap,
    );
    let half = R / 2;
    IVec3::new(snapped.x - half, snapped.y - half, snapped.z - half)
}

/// Surface height (in chunk units) of the modelled terrain at chunk (x,z) for a LOD — a
/// smooth ridge so the resident set is a thin shell, sparse within the R³ window.
fn surface_y(x: i32, z: i32) -> i32 {
    // A couple of sinusoids → a rolling terrain band ~3 chunks thick.
    let fx = x as f32 * 0.21;
    let fz = z as f32 * 0.17;
    (fx.sin() * 3.0 + (fx * 0.5 + fz).cos() * 2.5 + (fz * 0.7).sin() * 2.0) as i32
}

/// Is chunk (x,y,z) on the modelled surface shell (resident)? Thin band around surface_y.
fn chunk_resident(coord: IVec3) -> bool {
    let sy = surface_y(coord.x, coord.z);
    (coord.y - sy).abs() <= 1 // ~3-chunk-thick shell
}

/// Slab difference: invoke `f` for coords in the new R³ window not in the old (entered).
/// Faithful mirror of bake_scheduler::for_each_entered_chunk (axis-partitioned, no dedup).
fn for_each_entered(new_o: IVec3, old_o: IVec3, mut f: impl FnMut(IVec3)) {
    let r = R;
    let new_end = IVec3::new(new_o.x + r, new_o.y + r, new_o.z + r);
    let old_end = IVec3::new(old_o.x + r, old_o.y + r, old_o.z + r);
    let ov_min = IVec3::new(new_o.x.max(old_o.x), new_o.y.max(old_o.y), new_o.z.max(old_o.z));
    let ov_max = IVec3::new(new_end.x.min(old_end.x), new_end.y.min(old_end.y), new_end.z.min(old_end.z));
    let x_overlap_empty = ov_min.x >= ov_max.x;
    for x in new_o.x..new_end.x {
        let x_entered = x_overlap_empty || x < ov_min.x || x >= ov_max.x;
        if x_entered {
            for y in new_o.y..new_o.y + r {
                for z in new_o.z..new_o.z + r {
                    f(IVec3::new(x, y, z));
                }
            }
        } else {
            let y_overlap_empty = ov_min.y >= ov_max.y;
            for y in new_o.y..new_o.y + r {
                let y_entered = y_overlap_empty || y < ov_min.y || y >= ov_max.y;
                if y_entered {
                    for z in new_o.z..new_o.z + r {
                        f(IVec3::new(x, y, z));
                    }
                } else {
                    let z_overlap_empty = ov_min.z >= ov_max.z;
                    for z in new_o.z..new_o.z + r {
                        if z_overlap_empty || z < ov_min.z || z >= ov_max.z {
                            f(IVec3::new(x, y, z));
                        }
                    }
                }
            }
        }
    }
}

/// Exited shell (old window minus new) — args swapped, mirrors for_each_exited_chunk.
fn for_each_exited(new_o: IVec3, old_o: IVec3, f: impl FnMut(IVec3)) {
    for_each_entered(old_o, new_o, f);
}

/// Per-frame metrics for one structure.
#[derive(Default)]
struct Metrics {
    per_frame_max_mutate: Duration,
    total_mutate: Duration,
    lookup_total: Duration,
    lookup_ops: u64,
    final_resident_chunks: usize,
    final_memory_bytes: usize,
}

/// Deterministic xorshift so the lookup stream + fly-path are reproducible across structures.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Cheap spatial hash (deterministic, structure-independent) for brick-count / local choice.
fn rng_hash(c: IVec3) -> u32 {
    let mut h = (c.x as u32).wrapping_mul(0x9E37_79B1);
    h ^= (c.y as u32).wrapping_mul(0x85EB_CA77);
    h ^= (c.z as u32).wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h
}

// ---------------------------------------------------------------------------------------
// Benchmark entry points.
// ---------------------------------------------------------------------------------------

const FRAMES: u32 = 600;
const LOOKUPS_PER_FRAME: u32 = 200_000;

fn fmt_ms(d: Duration) -> String {
    format!("{:.3}", d.as_secs_f64() * 1000.0)
}
fn fmt_bytes(b: usize) -> String {
    if b >= 1 << 20 {
        format!("{:.2} MB", b as f64 / (1 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1} KB", b as f64 / (1 << 10) as f64)
    } else {
        format!("{b} B")
    }
}

#[test]
#[ignore = "profiling rig; run explicitly with --ignored --nocapture"]
fn bench_chunk_lookup_structures() {
    println!();
    println!("=== Chunk-lookup structure benchmark ===");
    println!(
        "workload: R={R} chunks/axis, lod_count={LOD_COUNT}, recenter_snap={RECENTER_SNAP_CHUNKS}, \
         {FRAMES} frames, {LOOKUPS_PER_FRAME} lookups/frame"
    );
    println!("camera: diagonal fly-path (~1 LOD0 chunk/frame); sparse surface-shell occupancy");
    println!();

    // We need the resident count out of each run; run each candidate and capture its own count.
    let sorted = run_named("SortedArray (baseline)", SortedArray::default());
    let toroidal = run_named("ToroidalDirectory (inline)", ToroidalDirectory::default());
    let hybrid = run_named("ToroidalHybrid (dir+free-list)", ToroidalHybrid::default());

    println!();
    println!(
        "{:<32} {:>14} {:>14} {:>14} {:>10} {:>12}",
        "structure", "max mutate ms", "total mut ms", "lookup ns/op", "resident", "memory"
    );
    println!("{}", "-".repeat(100));
    for r in [&sorted, &toroidal, &hybrid] {
        let ns_per_op = if r.1.lookup_ops > 0 {
            r.1.lookup_total.as_nanos() as f64 / r.1.lookup_ops as f64
        } else {
            0.0
        };
        println!(
            "{:<32} {:>14} {:>14} {:>14.1} {:>10} {:>12}",
            r.0,
            fmt_ms(r.1.per_frame_max_mutate),
            fmt_ms(r.1.total_mutate),
            ns_per_op,
            r.1.final_resident_chunks,
            fmt_bytes(r.1.final_memory_bytes),
        );
    }
    println!();

    // Sanity: the spike differential is the whole point. Assert the toroidal variants beat the
    // sorted baseline on per-frame max mutate (the 448 ms spike class).
    assert!(
        toroidal.1.per_frame_max_mutate < sorted.1.per_frame_max_mutate,
        "toroidal directory should have a lower per-frame mutate spike than the sorted array"
    );
    assert!(
        hybrid.1.per_frame_max_mutate < sorted.1.per_frame_max_mutate,
        "hybrid should have a lower per-frame mutate spike than the sorted array"
    );
}

/// Run one structure and return its (label, metrics), filling the resident count by re-probing
/// the structure type (each maintains its own `live`/`chunks` count).
fn run_named<S: ChunkStructure + ResidentCount + 'static>(label: &str, s: S) -> (String, Metrics) {
    // Manual run so we can read the structure's own resident count at the end.
    let mut s = s;
    let mut m = Metrics::default();
    let mut origins = [IVec3::new(i32::MIN / 4, i32::MIN / 4, i32::MIN / 4); LOD_COUNT as usize];
    let mut tile_ctr: u32 = 1;
    let mut rng = Rng(0x1234_5678_9abc_def0);

    for frame in 0..FRAMES {
        let cam = IVec3::new(frame as i32, (frame as i32) / 3, (frame as i32) * 2);
        let mut new_origins = origins;
        for lod in 0..LOD_COUNT {
            new_origins[lod as usize] = window_origin(cam, lod);
        }
        s.set_window_origins(&new_origins);

        let t_mut = Instant::now();
        for lod in 0..LOD_COUNT {
            let li = lod as usize;
            let new_o = new_origins[li];
            let old_o = origins[li];
            if new_o == old_o {
                continue;
            }
            let first = old_o.x == i32::MIN / 4;
            for_each_entered(new_o, old_o, |coord| {
                if chunk_resident(coord) {
                    let ck = ChunkKey::new(lod, coord);
                    let nb = 3 + (rng_hash(coord) % 6);
                    for b in 0..nb {
                        let local = rng_hash(IVec3::new(coord.x, coord.y, coord.z + b as i32))
                            % CHUNK_VOLUME;
                        let tile = BrickTile {
                            atlas_base: tile_ctr,
                            pal01: tile_ctr ^ 0xAAAA,
                            pal23: tile_ctr ^ 0x5555,
                        };
                        tile_ctr = tile_ctr.wrapping_add(1);
                        s.set_brick(ck, local, tile);
                    }
                }
            });
            if !first {
                for_each_exited(new_o, old_o, |coord| {
                    if chunk_resident(coord) {
                        let ck = ChunkKey::new(lod, coord);
                        for local in 0..CHUNK_VOLUME {
                            s.clear_brick(ck, local);
                        }
                    }
                });
            }
        }
        let dt = t_mut.elapsed();
        m.total_mutate += dt;
        if dt > m.per_frame_max_mutate {
            m.per_frame_max_mutate = dt;
        }
        s.end_frame();
        origins = new_origins;

        let t_look = Instant::now();
        let mut sink = 0u64;
        for _ in 0..LOOKUPS_PER_FRAME {
            let lod = (rng.next() % LOD_COUNT as u64) as u32;
            let o = new_origins[lod as usize];
            let cx = o.x + (rng.next() % R as u64) as i32;
            let cz = o.z + (rng.next() % R as u64) as i32;
            let sy = surface_y(cx, cz);
            let cy = if rng.next() % 10 < 7 {
                sy + ((rng.next() % 3) as i32 - 1)
            } else {
                o.y + (rng.next() % R as u64) as i32
            };
            let ck = ChunkKey::new(lod, IVec3::new(cx, cy, cz));
            let local = (rng.next() % CHUNK_VOLUME as u64) as u32;
            if let Some(t) = s.lookup(ck, local) {
                sink = sink.wrapping_add(t.atlas_base as u64);
            }
            m.lookup_ops += 1;
        }
        std::hint::black_box(sink);
        m.lookup_total += t_look.elapsed();
    }

    m.final_resident_chunks = s.resident_count();
    m.final_memory_bytes = s.memory_bytes();
    (label.to_string(), m)
}

/// Lets `run_named` read each structure's own resident-chunk count without a downcast.
trait ResidentCount {
    fn resident_count(&self) -> usize;
}
impl ResidentCount for SortedArray {
    fn resident_count(&self) -> usize {
        self.chunks.len()
    }
}
impl ResidentCount for ToroidalDirectory {
    fn resident_count(&self) -> usize {
        self.live
    }
}
impl ResidentCount for ToroidalHybrid {
    fn resident_count(&self) -> usize {
        self.live
    }
}

// ---------------------------------------------------------------------------------------
// Correctness cross-check: every structure must resolve identically (a profiling rig that
// measured a broken structure would be worthless). Runs a short shared workload and asserts
// all three agree on every probe against ground truth.
// ---------------------------------------------------------------------------------------

#[test]
#[ignore = "profiling rig; run explicitly with --ignored --nocapture"]
fn structures_agree_on_lookups() {
    let mut sorted = SortedArray::default();
    let mut toroidal = ToroidalDirectory::default();
    let mut hybrid = ToroidalHybrid::default();

    // Ground truth: only the most-recent set for an IN-WINDOW (chunk, local) is resident. The
    // toroidal variants intentionally drop out-of-window chunks (in_ring_chunk), so truth must
    // mirror that to compare fairly — a departed chunk is "not resident" by design.
    let mut truth: HashMap<(ChunkKey, u32), BrickTile> = HashMap::new();

    let mut origins = [IVec3::new(i32::MIN / 4, i32::MIN / 4, i32::MIN / 4); LOD_COUNT as usize];
    let mut tile_ctr: u32 = 1;

    for frame in 0..120u32 {
        let cam = IVec3::new(frame as i32, (frame as i32) / 3, (frame as i32) * 2);
        let mut new_origins = origins;
        for lod in 0..LOD_COUNT {
            new_origins[lod as usize] = window_origin(cam, lod);
        }
        sorted.set_window_origins(&new_origins);
        toroidal.set_window_origins(&new_origins);
        hybrid.set_window_origins(&new_origins);

        for lod in 0..LOD_COUNT {
            let li = lod as usize;
            let new_o = new_origins[li];
            let old_o = origins[li];
            if new_o == old_o {
                continue;
            }
            let first = old_o.x == i32::MIN / 4;
            for_each_entered(new_o, old_o, |coord| {
                if chunk_resident(coord) {
                    let ck = ChunkKey::new(lod, coord);
                    let nb = 3 + (rng_hash(coord) % 6);
                    for b in 0..nb {
                        let local = rng_hash(IVec3::new(coord.x, coord.y, coord.z + b as i32))
                            % CHUNK_VOLUME;
                        let tile = BrickTile {
                            atlas_base: tile_ctr,
                            pal01: tile_ctr ^ 0xAAAA,
                            pal23: tile_ctr ^ 0x5555,
                        };
                        tile_ctr = tile_ctr.wrapping_add(1);
                        sorted.set_brick(ck, local, tile);
                        toroidal.set_brick(ck, local, tile);
                        hybrid.set_brick(ck, local, tile);
                        truth.insert((ck, local), tile);
                    }
                }
            });
            if !first {
                for_each_exited(new_o, old_o, |coord| {
                    if chunk_resident(coord) {
                        let ck = ChunkKey::new(lod, coord);
                        for local in 0..CHUNK_VOLUME {
                            sorted.clear_brick(ck, local);
                            toroidal.clear_brick(ck, local);
                            hybrid.clear_brick(ck, local);
                            truth.remove(&(ck, local));
                        }
                    }
                });
            }
        }
        sorted.end_frame();
        toroidal.end_frame();
        hybrid.end_frame();
        origins = new_origins;

        // Probe every truth entry that is still in-window (the toroidal contract) + a stream of
        // random in-window probes. All three structures must agree with truth (modulo the
        // toroidal in-window guard, which the sorted array also satisfies for resident chunks).
        for (&(ck, local), &tile) in &truth {
            let in_win = {
                let o = new_origins[ck.lod as usize];
                let rel = IVec3::new(ck.coord.x - o.x, ck.coord.y - o.y, ck.coord.z - o.z);
                rel.x >= 0 && rel.y >= 0 && rel.z >= 0 && rel.x < R && rel.y < R && rel.z < R
            };
            if !in_win {
                continue; // out-of-window: toroidal returns None by design; skip the compare
            }
            let s = sorted.lookup(ck, local);
            let t = toroidal.lookup(ck, local);
            let h = hybrid.lookup(ck, local);
            assert_eq!(s, Some(tile), "frame {frame}: sorted disagrees for {ck:?} {local}");
            assert_eq!(t, Some(tile), "frame {frame}: toroidal disagrees for {ck:?} {local}");
            assert_eq!(h, Some(tile), "frame {frame}: hybrid disagrees for {ck:?} {local}");
        }
    }
    println!("structures_agree_on_lookups: all three resolve identically across the fly-path");
}

// =======================================================================================
// Adversarial tests — prove each of the three design blockers the audit flagged is REAL (a
// naive impl produces a hole / ghost / leak) and that the proposed fix closes it. Fast property
// tests (normal `cargo test`), not the #[ignore] profiling bench above. They also document the
// finding that an EXPLICIT-clear migration (calling clear_brick on exit, as the bench does)
// side-steps blockers 2 and 3 — those two only bite the pure free-eviction (no-clear) mode.
// =======================================================================================

/// Per-LOD window origins for a camera at LOD-0 chunk `cam` (mirrors `run_named`'s setup).
fn origins_for(cam: IVec3) -> [IVec3; LOD_COUNT as usize] {
    let mut o = [IVec3::new(0, 0, 0); LOD_COUNT as usize];
    for lod in 0..LOD_COUNT {
        o[lod as usize] = window_origin(cam, lod);
    }
    o
}

/// Resolve a lookup the way the GPU would: `in_ring_chunk` tests against the LAST-UPLOADED window
/// origins (the camera uniform, which can lag the directory by a frame), then the tag+occ resolve
/// runs against the CURRENT directory contents. Models the CPU(dir)<->GPU(O_lod) upload skew.
fn resolve_with_uploaded(
    h: &ToroidalHybrid,
    uploaded: &[IVec3; LOD_COUNT as usize],
    ck: ChunkKey,
    local: u32,
) -> Option<BrickTile> {
    let o = uploaded[ck.lod as usize];
    let rel = IVec3::new(ck.coord.x - o.x, ck.coord.y - o.y, ck.coord.z - o.z);
    if !(rel.x >= 0 && rel.y >= 0 && rel.z >= 0 && rel.x < R && rel.y < R && rel.z < R) {
        return None; // in_ring_chunk reject against the (possibly stale) uploaded origin
    }
    let idx = dir_index(ck.coord, ck.lod);
    let e = h.dir[idx];
    if e.tag != chunk_gpu_key(ck) || e.run_slot == u32::MAX {
        return None;
    }
    let off = rank_of(e.occ, local)?;
    let region = dense_region(e.occ, &h.tile_runs[e.run_slot as usize]);
    Some(region[off as usize])
}

/// BLOCKER 1 — the directory slot must be published ATOMICALLY at bake-APPLY, never at chunk-ENTER.
/// A departed chunk D and an entering chunk C share one physical slot (c mod R). Publishing C's tag
/// before its tile is baked (enter-time, reusing D's tile-run bytes) makes an in-window tag-match
/// resolve C to D's GEOMETRY — a wrong-geometry class the sorted array cannot produce. Apply-time,
/// whole-record publication yields a clean miss (coarse fallback) until C bakes, then C's own tile.
#[test]
fn adversarial_publish_must_be_at_apply_not_enter() {
    let lod = 0u32;
    let d_coord = IVec3::new(0, 0, 0);
    let c_coord = IVec3::new(R, 0, 0); // R apart → same slot (c mod R == d mod R)
    assert_eq!(dir_index(d_coord, lod), dir_index(c_coord, lod), "test needs a shared slot");
    let d = ChunkKey::new(lod, d_coord);
    let c = ChunkKey::new(lod, c_coord);
    let d_tile = BrickTile { atlas_base: 0xDDDD, pal01: 1, pal23: 2 };
    let c_tile = BrickTile { atlas_base: 0xCCCC, pal01: 3, pal23: 4 };

    // D resident, window covering it.
    let mut h = ToroidalHybrid::default();
    h.set_window_origins(&origins_for(d_coord));
    h.set_brick(d, 0, d_tile);
    assert_eq!(h.lookup(d, 0), Some(d_tile));

    // Camera advances so the window now covers C (D departs).
    h.set_window_origins(&origins_for(c_coord));
    assert_eq!(h.lookup(d, 0), None, "departed D must miss (out of window)");

    // APPLY-TIME publish (what set_brick does): until C's bake applies, the slot still carries D's tag
    // → C cleanly MISSES (coarse fallback = a pop), never wrong geometry. Then it resolves to C's tile.
    assert_eq!(h.lookup(c, 0), None, "C must miss until its bake applies — no wrong geometry");
    h.set_brick(c, 0, c_tile); // apply: whole-record publish
    assert_eq!(h.lookup(c, 0), Some(c_tile), "after apply, C resolves to ITS OWN tile");

    // ENTER-TIME publish (the BUG): publish C's tag before its tile is baked, leaving D's stale
    // tile-run bytes in place. Modelled by writing the tag early on a fresh hybrid.
    let mut bug = ToroidalHybrid::default();
    bug.set_window_origins(&origins_for(d_coord));
    bug.set_brick(d, 0, d_tile);
    bug.set_window_origins(&origins_for(c_coord));
    let idx = dir_index(c_coord, lod);
    bug.dir[idx].tag = chunk_gpu_key(c); // tag published at ENTER; tile NOT yet baked (still D's bytes)
    assert_eq!(
        bug.lookup(c, 0),
        Some(d_tile),
        "enter-time publish resolves C to the DEPARTED chunk's geometry — exactly the wrong-geometry \
         class that apply-time, whole-record publication prevents"
    );
}

/// BLOCKER 2 — in pure free-eviction (no explicit clear), a fly-away into empty space writes no slots
/// and bakes nothing. If the generation isn't bumped on window-advance, the render world (gated on
/// generation) never re-uploads the per-LOD origin O_lod, so `in_ring_chunk` keeps using the STALE
/// origin and a departed chunk still resolves resident = a GHOST. The fix bumps generation (re-uploads
/// O_lod) on every window-advance, independent of whether a bake applied.
#[test]
fn adversarial_flyaway_must_bump_generation() {
    let d = ChunkKey::new(0, IVec3::new(0, 0, 0));
    let d_tile = BrickTile { atlas_base: 0xBEEF, pal01: 7, pal23: 9 };

    let mut h = ToroidalHybrid::default();
    let o_built = origins_for(IVec3::new(0, 0, 0));
    h.set_window_origins(&o_built);
    h.set_brick(d, 0, d_tile);
    // Render world uploaded dir + O_lod here (generation bumped by the bake).
    assert_eq!(resolve_with_uploaded(&h, &o_built, d, 0), Some(d_tile));

    // Fly far away with NO bake and NO clear (pure free-eviction): the window advances past D, whose
    // directory bytes still linger (nothing entered its slot to overwrite them).
    let o_now = origins_for(IVec3::new(R * 6, 0, 0));
    h.set_window_origins(&o_now);

    // NAIVE: generation NOT bumped on window-advance → GPU still holds the OLD O_lod → D ghosts.
    assert_eq!(
        resolve_with_uploaded(&h, &o_built, d, 0),
        Some(d_tile),
        "without a generation bump on fly-away, the stale uploaded O_lod ghosts the departed chunk"
    );

    // FIXED: bump generation on window-advance → O_lod re-uploaded → in_ring_chunk rejects D.
    assert_eq!(
        resolve_with_uploaded(&h, &o_now, d, 0),
        None,
        "bumping generation on window-advance re-uploads O_lod → the departed chunk correctly misses"
    );
}

/// BLOCKER 3 — and a correction to the audit's proposed fix. The audit said "free the departed run on
/// OVERWRITE" bounds the tile-run buffer. This test shows that is INSUFFICIENT: flying in a direction
/// where the surface height shifts means departed chunks' slots are never reused (the new surface lands
/// on different y-slots), so free-on-overwrite never reclaims them and the buffer still leaks. The
/// ROBUST bound is EXPLICIT clear-on-exit — now O(1) with the directory — which frees each departed
/// chunk's run immediately. (This is mode the benchmark already drives, and why it should be the
/// migration: it also moots blockers 2's fly-away ghost, since clear bumps generation.)
#[test]
fn adversarial_tilerun_leak_needs_explicit_clear_not_just_free_on_overwrite() {
    // Fly far in +x over a rolling surface. `clear_on_exit` = mode (a) explicit O(1) clear; otherwise
    // pure free-eviction relying only on overwrite (with free_on_overwrite toggled).
    fn fly(frames: i32, clear_on_exit: bool, free_on_overwrite: bool) -> u32 {
        let mut h = ToroidalHybrid { free_on_overwrite, ..Default::default() };
        let mut origins = [IVec3::new(i32::MIN / 4, 0, 0); LOD_COUNT as usize];
        let mut tile_ctr = 1u32;
        for frame in 0..frames {
            let cam = IVec3::new(frame, 0, 0);
            let mut new_o = origins;
            for lod in 0..LOD_COUNT {
                new_o[lod as usize] = window_origin(cam, lod);
            }
            h.set_window_origins(&new_o);
            for lod in 0..LOD_COUNT {
                let li = lod as usize;
                if new_o[li] == origins[li] {
                    continue;
                }
                let first = origins[li].x == i32::MIN / 4;
                if !first {
                    for_each_entered(new_o[li], origins[li], |coord| {
                        if chunk_resident(coord) {
                            let ck = ChunkKey::new(lod, coord);
                            let local = rng_hash(coord) % CHUNK_VOLUME;
                            h.set_brick(ck, local, BrickTile { atlas_base: tile_ctr, pal01: 0, pal23: 0 });
                            tile_ctr = tile_ctr.wrapping_add(1);
                        }
                    });
                    if clear_on_exit {
                        for_each_exited(new_o[li], origins[li], |coord| {
                            if chunk_resident(coord) {
                                let ck = ChunkKey::new(lod, coord);
                                for local in 0..CHUNK_VOLUME {
                                    h.clear_brick(ck, local);
                                }
                            }
                        });
                    }
                }
                origins[li] = new_o[li];
            }
        }
        h.run_high_water
    }

    // The signature of a LEAK is "high-water grows with fly DISTANCE"; a BOUNDED scheme does not.
    // Measure each mode at a short and a long fly and compare the growth.
    let (short, long) = (R * 4, R * 12);
    let fo_short = fly(short, false, true); // free-on-overwrite only (the audit's fix #3)
    let fo_long = fly(long, false, true);
    let ec_short = fly(short, true, true); // mode (a): explicit O(1) clear-on-exit
    let ec_long = fly(long, true, true);
    println!(
        "tile-run high-water vs fly distance ({short}->{long} frames): \
         free-on-overwrite {fo_short}->{fo_long} (+{}),  explicit-clear {ec_short}->{ec_long} (+{})",
        fo_long - fo_short,
        ec_long - ec_short
    );

    // free-on-overwrite alone LEAKS: high-water keeps climbing with distance (departed chunks whose
    // slot is never reused are never reclaimed).
    assert!(
        fo_long > fo_short + fo_short / 2,
        "free-on-overwrite alone leaks — high-water scales with fly distance: {fo_short}->{fo_long}"
    );
    // explicit clear-on-exit is BOUNDED: high-water is ~flat regardless of distance (each departed
    // chunk's run is freed immediately), and far below the leaking mode at long range.
    assert!(
        ec_long < ec_short + ec_short / 2,
        "explicit clear-on-exit must be bounded (not scale with distance): {ec_short}->{ec_long}"
    );
    assert!(
        ec_long < fo_long,
        "explicit clear-on-exit must stay below the leaking free-on-overwrite mode at long range: \
         {ec_long} vs {fo_long}"
    );
}
