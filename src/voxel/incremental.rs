//! **Incremental, O(changed) re-pack of the resident brick set into FIXED-CAPACITY GPU buffers.**
//!
//! The full [`pack_resident_set`](super::gpu::pack_resident_set) rebuilds the whole AABB / meta / voxel buffer
//! set on EVERY camera move (O(resident) ≈ 137 ms at the shipping `clip_half = 8`), then the render path
//! recreates all GPU buffers + the BLAS/TLAS from scratch. That is the per-move hitch this module removes.
//!
//! # The model — a per-brick SLOT allocator over fixed-capacity buffers
//!
//! Each resident [`BrickKey`] owns ONE slot `= its primitive_index FOR LIFE` (until it drops). The meta + AABB
//! live at a fixed offset (`slot · stride`) in capacity-`max_resident_bricks` buffers; the dense voxel block
//! lives in a SEPARATE voxel ARENA of fixed `halo_cells(0) = 1000`-u32 blocks (the dense stride is
//! LOD-invariant — [`halo_cells`](super::gpu)`(lod)` is `10³` at EVERY LOD — so the arena is a perfect
//! fixed-block free-list with ZERO fragmentation). A UNIFORM (R1) brick consumes no arena block.
//!
//! `primitive_index = slot` and the buffers stay a SINGLE contiguous AABB/meta/voxel set, so the BLAS still has
//! one AABB geometry and the shader's `metas[primitive_index]` is UNCHANGED — the trace is pixel-identical. A
//! dropped slot's AABB is patched to a DEGENERATE (inverted) box so the BLAS never reports a candidate there.
//!
//! # O(changed) — the dirty set
//!
//! An [`update`](ResidentPacker::update) diffs the new resident set against the live one and emits a
//! [`RepackDelta`]: the slots whose meta/AABB/voxel bytes changed, plus the freed slots. CRITICAL: a brick
//! changing ALSO changes the haloed grids of its resident SAME-LOD neighbours (their halo border reads this
//! brick's core — see [`pack_one`](super::gpu::pack_one)), AND a brick can toggle uniform↔dense purely from a
//! neighbour change. So the dirty set is EXPANDED by each entered/dropped/rewritten brick's resident
//! 26-neighbourhood at the same LOD, and those neighbours are re-`pack_one`'d too. This completeness is what
//! makes an incremental patch byte-identical to a from-scratch pack — the
//! [incremental-vs-full A/B test](tests) is the gate.
//!
//! # Byte-identity to the full pack (the SSOT)
//!
//! Both this module and [`pack_resident_set`](super::gpu::pack_resident_set) build each brick through the ONE
//! [`pack_one`](super::gpu::pack_one) per-brick byte producer, so a brick re-packed in isolation here is
//! byte-identical to the same brick in a from-scratch pack. The slot a brick lands in is an implementation
//! detail (a free-list reuses slots in a different order than a from-scratch `(lod,z,y,x)` sort); the render is
//! identical regardless because the shader resolves everything from `metas[slot].world_min` (no dependence on
//! slot ORDER beyond `primitive_index → meta`). The A/B equality test compares the two as a `key → bytes`
//! MAPPING, not raw slot order.

use bevy::math::IVec3;
use rustc_hash::FxHashMap;

use super::brickmap::Brick;
use super::gpu::{
    BrickVoxels, GpuBrickAabb, GpuBrickMeta, GpuPaletteColor, PackedBrick, ResidentBrick, halo_cells,
    pack_one,
};
use super::streaming::BrickKey;

/// The fixed number of `u32`s a DENSE brick occupies in the voxel arena — a haloed `10³` grid. CONSTANT at
/// every LOD ([`halo_cells`]`(0) == halo_cells(lod)`), so the arena is a perfect fixed-block free-list with no
/// fragmentation. A UNIFORM (R1) brick consumes zero arena blocks.
#[inline]
pub fn dense_block_u32() -> usize {
    halo_cells(0)
}

/// A DEGENERATE BLAS AABB (min > max on every axis) for an UNUSED slot: the BLAS build never reports a
/// candidate for it, so a freed slot is invisible to the trace. `primitive_index = slot` is preserved (the
/// buffers stay contiguous + fixed-capacity); only this slot's box is collapsed so the ray query skips it.
#[inline]
pub fn degenerate_aabb() -> GpuBrickAabb {
    GpuBrickAabb { min: [1.0e30, 1.0e30, 1.0e30], max: [-1.0e30, -1.0e30, -1.0e30], _pad: [0.0; 2] }
}

/// The 26 SAME-LOD face/edge/corner neighbours of a brick key (the full one-ring around it). The packer
/// expands every changed key by this set because the halo of each of those neighbours reads the changed
/// brick's core (and a drop flips that halo to AIR / a neighbour change can toggle the brick uniform↔dense) —
/// missing one leaves a stale halo seam or a wrong R1 classification. 26 is the robust-by-construction choice
/// (the dense halo-fill reads diagonal neighbours too, not just the 6 faces).
fn neighbourhood_26(key: BrickKey) -> impl Iterator<Item = BrickKey> {
    let base = key.coord;
    let lod = key.lod;
    (-1..=1).flat_map(move |dz| {
        (-1..=1).flat_map(move |dy| {
            (-1..=1).filter_map(move |dx| {
                if dx == 0 && dy == 0 && dz == 0 {
                    None
                } else {
                    Some(BrickKey { coord: base + IVec3::new(dx, dy, dz), lod })
                }
            })
        })
    })
}

/// A free-list allocator over `[0, capacity)` slot indices. Each resident [`BrickKey`] holds one slot for life
/// (its `primitive_index`); a dropped key frees its slot back. Reuse is bounded; the render never depends on
/// slot ORDER (only `primitive_index → meta`).
#[derive(Clone, Debug)]
struct SlotAllocator {
    capacity: u32,
    /// The next never-yet-allocated slot (bump pointer) until the free list is the only source.
    high_water: u32,
    /// Freed slots available for reuse (LIFO).
    free: Vec<u32>,
}

impl SlotAllocator {
    fn new(capacity: u32) -> Self {
        Self { capacity, high_water: 0, free: Vec::new() }
    }

    /// Claim a free slot, or `None` if at capacity. Prefers the bump pointer first so a FRESH fill lays slots
    /// out in claim order; the free list only kicks in after drops.
    fn claim(&mut self) -> Option<u32> {
        if self.high_water < self.capacity {
            let s = self.high_water;
            self.high_water += 1;
            Some(s)
        } else {
            self.free.pop()
        }
    }

    /// Return a slot to the free list.
    fn release(&mut self, slot: u32) {
        debug_assert!(slot < self.high_water, "releasing a never-claimed slot");
        self.free.push(slot);
    }
}

/// One slot's new GPU bytes after an incremental re-pack: the slot index (= `primitive_index`), its meta, its
/// AABB, and — for a dense brick — the arena byte offset + the voxel block to write there. The GPU uploader
/// patches `metas[slot]`, `aabbs[slot]`, and (dense) the arena slice at `voxel_word_offset` from this.
#[derive(Clone, Debug)]
pub struct ChangedSlot {
    /// The slot whose buffers this patches (= `primitive_index`).
    pub slot: u32,
    /// The new per-brick meta (already carries the correct `voxel_offset` for dense, or the uniform flag/id;
    /// all-zero for a freed slot).
    pub meta: GpuBrickMeta,
    /// The new BLAS AABB ([`degenerate_aabb`] for a freed slot).
    pub aabb: GpuBrickAabb,
    /// `Some(words)` for a DENSE brick whose voxel block must be (re-)written: the `dense_block_u32()`-u32 block
    /// to write at `voxel_word_offset`. `None` for a uniform/freed slot, or a dense brick whose meta changed but
    /// whose voxel bytes did not (no arena re-write needed).
    pub voxels: Option<Vec<u32>>,
    /// The arena WORD offset (`u32` index into the voxel buffer) the dense block goes at (= `meta.dense_offset()`
    /// when `voxels.is_some()`).
    pub voxel_word_offset: u32,
}

/// The set of slots an incremental [`update`](ResidentPacker::update) changed, so the GPU side patches ONLY
/// these via `queue_write_buffer` (O(changed), not O(resident)).
#[derive(Clone, Debug, Default)]
pub struct RepackDelta {
    /// Slots whose `meta`/`aabb` (and, for dense bricks, voxel block) changed — re-upload these.
    pub changed: Vec<ChangedSlot>,
    /// Slots that became UNUSED this update (their AABB is now degenerate + meta zeroed; also present in
    /// `changed`). For the AS-topology-changed signal / bookkeeping.
    pub freed: Vec<u32>,
    /// True iff the resident brick SET (which slots are occupied) changed — the signal the BLAS/TLAS need
    /// rebuilding. A pure meta/voxel edit with no enter/drop leaves AABB occupancy unchanged (the AS can be
    /// refit/kept), but conservatively any entered/dropped brick sets this.
    pub topology_changed: bool,
}

impl RepackDelta {
    /// True iff nothing changed (no slot touched) — the caller can skip the GPU patch entirely.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.changed.is_empty()
    }
}

/// The FIXED-CAPACITY initial buffer contents the render path allocates ONCE per scene epoch (storage plan A1:
/// the O(changed) GPU upload). Built from the packer's current shadow state by [`ResidentPacker::snapshot_buffers`].
/// After this snapshot the render path applies each generation's [`RepackDelta`] via `queue_write_buffer` — it
/// NEVER re-creates these buffers within an epoch (only a scene switch / new epoch re-snapshots).
///
/// **A1-β representation — the streamed arena is RAW, not R2b paletted.** Each dense slot's voxel block is the
/// HALOED `10³` `u32`-per-cell grid written verbatim at `arena_block · dense_block_u32()`, and its meta carries
/// `index_bits == 0` / `palette_base == 0` as the RAW-ARENA marker. The shader's `cell_block` decodes a raw
/// `index_bits == 0` dense brick as `voxel_indices[voxel_offset + cell_index]` directly (no palette indirection);
/// the R2b paletted decode (`index_bits >= 1`) is kept for the static `pack_brickmap`/`pack_resident_set` path.
/// This trades R2b's voxel-VRAM win on the STREAMED path for the O(changed) upload — see the `cell_block` doc and
/// `PHASE_A_GPU_EXECUTION.md` §"The encoding question" (A1-β). R2b is recovered later (A4.4 persistent interner).
#[derive(Clone, Debug)]
pub struct SnapshotBuffers {
    /// Capacity-length AABB buffer; unused slots = [`degenerate_aabb`].
    pub aabbs: Vec<GpuBrickAabb>,
    /// Capacity-length meta buffer; unused slots = [`GpuBrickMeta::zeroed`].
    pub metas: Vec<GpuBrickMeta>,
    /// The `arena_capacity_u32()` raw block pool: each resident dense brick's HALOED `10³` `u32`-per-cell block
    /// written at its arena offset (`arena_block · dense_block_u32()`); zero elsewhere.
    pub voxels: Vec<u32>,
    /// The registry palette (`BlockId(i)` → linear RGBA + emissive). Length == registry length. Fixed per scene
    /// (never re-uploaded on a [`RepackDelta`]).
    pub palette: Vec<GpuPaletteColor>,
    /// The number of resident bricks (== the live BLAS primitive count; the BLAS is still built over `capacity`
    /// primitives with degenerate free slots, but this is reported for diagnostics).
    pub brick_count: u32,
}

/// One resident brick's live state in the packer: its slot + (for a dense brick) its voxel-arena block index.
#[derive(Clone, Copy, Debug)]
struct SlotState {
    slot: u32,
    /// `Some(block)` for a DENSE brick (its arena block index; word offset = `block · dense_block_u32()`).
    /// `None` for a UNIFORM brick (no arena block).
    arena_block: Option<u32>,
}

/// The incremental resident-set packer: owns the slot/arena allocators + the live `key → SlotState` map +
/// shadow copies of the last-uploaded meta/aabb/voxels per slot (so it can emit a [`ChangedSlot`] ONLY when
/// bytes actually differ). The render path holds one alongside the GPU buffers;
/// [`update`](Self::update) returns the [`RepackDelta`] to upload.
///
/// Robust-by-construction: every brick's bytes come from the SSOT [`pack_one`], so an incremental patch equals
/// a from-scratch [`pack_resident_set`](super::gpu::pack_resident_set) for the same `key → brick` mapping (the
/// A/B test). The dirty set is EXPANDED by the 26-neighbourhood so no halo/uniform-classification goes stale.
pub struct ResidentPacker {
    slots: SlotAllocator,
    /// Live resident bricks → their slot + arena block.
    resident: FxHashMap<BrickKey, SlotState>,
    /// The voxel-arena free-list of `dense_block_u32()`-u32 BLOCKS (block index, not word offset). Mirrors the
    /// slot allocator; capacity = `max_resident_bricks` blocks (a dense brick needs one, uniform bricks none —
    /// so the arena never binds before the slot cap).
    arena_high_water: u32,
    arena_free: Vec<u32>,
    arena_capacity: u32,
    /// Shadow of the last-uploaded bytes per slot — the byte-compare source so a re-pack that reproduces the
    /// same bytes (a neighbour that did not actually change) costs no upload.
    last_meta: FxHashMap<u32, GpuBrickMeta>,
    last_aabb: FxHashMap<u32, GpuBrickAabb>,
    last_voxels: FxHashMap<u32, Vec<u32>>,
    /// Keys the caller explicitly marked rewritten (edit / dig re-source) since the last update — re-packed even
    /// though they neither entered nor dropped. Drained each update.
    pending_rewrites: Vec<BrickKey>,
    /// DEFERRED-FREE quarantine (keep-old-until-revealed): slots/arena-blocks dropped THIS update can't be
    /// reused until the NEXT update, so an in-flight frame tracing the old generation never sees a slot's bytes
    /// overwritten by a different brick mid-flight. Released at the top of the next update.
    quarantine_slots: Vec<u32>,
    quarantine_arena: Vec<u32>,
}

impl ResidentPacker {
    /// A fresh packer sized for `max_resident_bricks` slots (and the same number of dense arena blocks — the
    /// worst case where every resident brick is dense). The GPU buffers the render path allocates must match
    /// this capacity.
    pub fn new(max_resident_bricks: u32) -> Self {
        Self {
            slots: SlotAllocator::new(max_resident_bricks),
            resident: FxHashMap::default(),
            arena_high_water: 0,
            arena_free: Vec::new(),
            arena_capacity: max_resident_bricks,
            last_meta: FxHashMap::default(),
            last_aabb: FxHashMap::default(),
            last_voxels: FxHashMap::default(),
            pending_rewrites: Vec::new(),
            quarantine_slots: Vec::new(),
            quarantine_arena: Vec::new(),
        }
    }

    /// The fixed slot CAPACITY (= the meta/AABB buffer length, in bricks).
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.slots.capacity
    }

    /// The voxel ARENA capacity in `u32`s (`arena_capacity · dense_block_u32()`) — the voxel buffer length the
    /// render path must allocate.
    #[inline]
    pub fn arena_capacity_u32(&self) -> usize {
        self.arena_capacity as usize * dense_block_u32()
    }

    /// Number of resident bricks currently slotted.
    #[inline]
    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    /// Build the FULL capacity-sized initial GPU buffers (storage plan A1 — the O(changed) upload). Called ONCE
    /// per scene epoch, right after the packer (re-)created + the first [`update`](Self::update) ran. `aabbs`/
    /// `metas` are capacity-length with [`degenerate_aabb`]/[`GpuBrickMeta::zeroed`] in the unused slots; `voxels`
    /// is the `arena_capacity_u32()` RAW block pool with each resident dense brick's haloed `10³` block copied to
    /// `arena_block · dense_block_u32()`. O(capacity) — paid ONCE per scene switch, never per move. After this the
    /// render path applies each [`RepackDelta`] via `queue_write_buffer` (meta/aabb at `slot · stride`, the raw
    /// block at `voxel_word_offset`), never re-creating these buffers within the epoch.
    ///
    /// The metas/aabbs are filled from the per-slot shadow (`last_meta`/`last_aabb`), which after `update` is
    /// EXACTLY the bytes a `snapshot_buffers`-then-delta sequence converges to (the byte-identity gate). The
    /// `palette` is the registry (fixed per scene); the NEE light list is NOT per-slot and is built separately by
    /// the caller (see `build_lights_from_entries`).
    pub fn snapshot_buffers(&self, registry: &super::palette::BlockRegistry) -> SnapshotBuffers {
        let cap = self.slots.capacity as usize;
        // Capacity-length meta/aabb: each occupied slot's shadow; degenerate/zeroed for the rest. The shadow
        // holds an entry for every slot that has EVER been written (occupied or freed→zeroed), so a slot absent
        // from the shadow is one that was never claimed — fill it degenerate/zeroed too.
        let mut metas = vec![GpuBrickMeta::zeroed(); cap];
        let mut aabbs = vec![degenerate_aabb(); cap];
        for (&slot, meta) in &self.last_meta {
            metas[slot as usize] = *meta;
        }
        for (&slot, aabb) in &self.last_aabb {
            aabbs[slot as usize] = *aabb;
        }
        // Raw voxel arena: each resident DENSE slot's haloed block at its arena offset. `last_voxels` holds the
        // raw cells per slot; `resident` carries the arena_block index. A uniform brick has no arena block / no
        // voxel shadow, so it contributes nothing (its id rides in the meta).
        let mut voxels = vec![0u32; self.arena_capacity_u32()];
        let block_u32 = dense_block_u32();
        for st in self.resident.values() {
            if let Some(block) = st.arena_block
                && let Some(cells) = self.last_voxels.get(&st.slot)
            {
                let off = block as usize * block_u32;
                voxels[off..off + cells.len()].copy_from_slice(cells);
            }
        }
        // Palette = the registry (block id → linear RGBA + emissive), indexed directly. Fixed per scene.
        let palette: Vec<GpuPaletteColor> = (0..registry.len())
            .map(|i| {
                let id = super::palette::BlockId(i as u16);
                let e = registry.emissive(id);
                GpuPaletteColor { rgba: registry.color(id), emissive: [e[0], e[1], e[2], 0.0] }
            })
            .collect();
        SnapshotBuffers { aabbs, metas, voxels, palette, brick_count: self.resident.len() as u32 }
    }

    /// Mark `keys` as REWRITTEN (an edit / dig re-source replaced their voxels in place): they re-pack on the
    /// next [`update`](Self::update) even though they neither entered nor dropped. Mirrors the manager's
    /// `requeue_keys` so the edit/dig path stays incremental (only the affected bricks + their 26-neighbourhood
    /// re-pack). A key that is not resident on the next update is ignored (it may enter then, taking the normal
    /// entered path).
    pub fn mark_rewritten(&mut self, keys: impl IntoIterator<Item = BrickKey>) {
        self.pending_rewrites.extend(keys);
    }

    /// Claim a voxel-arena block, or `None` at capacity.
    fn claim_arena(&mut self) -> Option<u32> {
        if self.arena_high_water < self.arena_capacity {
            let b = self.arena_high_water;
            self.arena_high_water += 1;
            Some(b)
        } else {
            self.arena_free.pop()
        }
    }

    /// Incrementally reconcile the packer toward `entries` (the manager's `resident_entries()`, in the SSOT
    /// `(lod,z,y,x)` order). Returns the [`RepackDelta`] of slots whose GPU bytes changed — O(changed + halo),
    /// never O(resident). The CALLER uploads only `delta.changed` via `queue_write_buffer`.
    pub fn update(&mut self, entries: &[ResidentBrick<'_>]) -> RepackDelta {
        // (1) Deferred-free: last update's quarantined slots/arena blocks are now safe to reuse (the frame that
        // could still be tracing them has been submitted). Release them BEFORE claiming this update's slots.
        for s in self.quarantine_slots.drain(..) {
            self.slots.release(s);
        }
        for b in self.quarantine_arena.drain(..) {
            self.arena_free.push(b);
        }

        // The NEW resident map (key → brick) + the by_key index pack_one reads for halos.
        let new_by_key: FxHashMap<BrickKey, &Brick> =
            entries.iter().map(|e| (BrickKey { coord: e.coord, lod: e.lod }, e.brick)).collect();
        let by_key = super::gpu::build_by_key(entries);

        let mut delta = RepackDelta::default();
        let mut topology_changed = false;
        let mut dirty: FxHashMap<BrickKey, ()> = FxHashMap::default();

        // (2a) DROP keys no longer resident: free their slot/arena (→ quarantine), collapse their slot, and
        // dirty their neighbours (whose halo now reads AIR where this brick was).
        let live_keys: Vec<BrickKey> = self.resident.keys().copied().collect();
        for key in live_keys {
            if new_by_key.contains_key(&key) {
                continue;
            }
            let st = self.resident.remove(&key).expect("key from live set");
            delta.changed.push(ChangedSlot {
                slot: st.slot,
                meta: GpuBrickMeta::zeroed(),
                aabb: degenerate_aabb(),
                voxels: None,
                voxel_word_offset: 0,
            });
            delta.freed.push(st.slot);
            self.quarantine_slots.push(st.slot);
            self.last_meta.insert(st.slot, GpuBrickMeta::zeroed());
            self.last_aabb.insert(st.slot, degenerate_aabb());
            self.last_voxels.remove(&st.slot);
            if let Some(b) = st.arena_block {
                self.quarantine_arena.push(b);
            }
            topology_changed = true;
            for nbr in neighbourhood_26(key) {
                if new_by_key.contains_key(&nbr) {
                    dirty.insert(nbr, ());
                }
            }
        }

        // (2b) ENTER keys not yet resident: claim a slot now (so the expansion sees it), seed them dirty.
        for e in entries {
            let key = BrickKey { coord: e.coord, lod: e.lod };
            if self.resident.contains_key(&key) {
                continue;
            }
            let Some(slot) = self.slots.claim() else {
                // At capacity — drop this brick (the manager already bounds the set; defensive skip).
                continue;
            };
            self.resident.insert(key, SlotState { slot, arena_block: None });
            dirty.insert(key, ());
            topology_changed = true;
        }

        // (2c) Explicitly-rewritten keys (edits / dig re-source) — re-pack even though unchanged-membership.
        for key in std::mem::take(&mut self.pending_rewrites) {
            if new_by_key.contains_key(&key) {
                dirty.insert(key, ());
            }
        }

        // (3) EXPAND by the resident 26-neighbourhood at the same LOD (halo dependency). Snapshot first.
        let seeds: Vec<BrickKey> = dirty.keys().copied().collect();
        for key in seeds {
            for nbr in neighbourhood_26(key) {
                if new_by_key.contains_key(&nbr) {
                    dirty.insert(nbr, ());
                }
            }
        }

        // (4) Re-pack each dirty key against the NEW resident map; emit a ChangedSlot iff its bytes differ from
        // its slot's shadow. Deterministic order so the patch list is reproducible (the perf/A-B tests rely
        // on it).
        let mut dirty_keys: Vec<BrickKey> = dirty.keys().copied().collect();
        dirty_keys.sort_by_key(|k| (k.lod, k.coord.z, k.coord.y, k.coord.x));
        for key in dirty_keys {
            let Some(&brick) = new_by_key.get(&key) else { continue };
            let e = ResidentBrick { coord: key.coord, brick, lod: key.lod };
            let pb = pack_one(&e, &by_key);
            self.emit_changed_slot(key, &pb, &mut delta);
        }

        delta.topology_changed = topology_changed;
        delta
    }

    /// Write `pb`'s bytes into `key`'s slot, allocating/freeing its arena block as the dense/uniform
    /// classification changed, and push a [`ChangedSlot`] iff the bytes actually differ from the slot's shadow.
    fn emit_changed_slot(&mut self, key: BrickKey, pb: &PackedBrick, delta: &mut RepackDelta) {
        let st = *self.resident.get(&key).expect("dirty key is resident");
        match &pb.voxels {
            BrickVoxels::Uniform(_) => {
                // Uniform now: free any arena block it held (→ quarantine, keep-old).
                if let Some(b) = st.arena_block {
                    self.quarantine_arena.push(b);
                }
                let meta = pb.meta_uniform();
                self.resident.insert(key, SlotState { slot: st.slot, arena_block: None });
                let changed =
                    self.last_meta.get(&st.slot) != Some(&meta) || self.last_aabb.get(&st.slot) != Some(&pb.aabb);
                self.last_meta.insert(st.slot, meta);
                self.last_aabb.insert(st.slot, pb.aabb);
                self.last_voxels.remove(&st.slot);
                if changed {
                    delta.changed.push(ChangedSlot {
                        slot: st.slot,
                        meta,
                        aabb: pb.aabb,
                        voxels: None,
                        voxel_word_offset: 0,
                    });
                }
            }
            BrickVoxels::Dense(cells) => {
                // Dense now: ensure it has an arena block (allocate if it just toggled from uniform).
                let block = match st.arena_block {
                    Some(b) => b,
                    None => match self.claim_arena() {
                        Some(b) => b,
                        None => return, // arena full (defensive — slot cap binds first); skip
                    },
                };
                let word_offset = block * dense_block_u32() as u32;
                // FIXED-ARENA shadow meta: the arena stores RAW haloed cells (NOT a paletted index stream — the
                // R2b encode happens at `snapshot_patch` time, see that method), so this dense meta uses the
                // arena `word_offset` as its offset with `index_bits = 0` / `palette_base = 0` as the RAW-ARENA
                // marker. `snapshot_patch` reads only `is_uniform`/`voxel_origin`/`world_min`/`lod()` from the
                // shadow and re-encodes from the raw `last_voxels`, so these two fields are inert here.
                let meta = super::gpu::GpuBrickMeta::dense(pb.voxel_origin, word_offset, pb.world_min, pb.lod, 0, 0);
                self.resident.insert(key, SlotState { slot: st.slot, arena_block: Some(block) });
                let meta_changed =
                    self.last_meta.get(&st.slot) != Some(&meta) || self.last_aabb.get(&st.slot) != Some(&pb.aabb);
                let voxels_changed = self.last_voxels.get(&st.slot) != Some(cells);
                self.last_meta.insert(st.slot, meta);
                self.last_aabb.insert(st.slot, pb.aabb);
                if voxels_changed {
                    self.last_voxels.insert(st.slot, cells.clone());
                }
                if meta_changed || voxels_changed {
                    delta.changed.push(ChangedSlot {
                        slot: st.slot,
                        meta,
                        aabb: pb.aabb,
                        voxels: if voxels_changed { Some(cells.clone()) } else { None },
                        voxel_word_offset: word_offset,
                    });
                }
            }
        }
    }

    /// Assemble a CONTIGUOUS [`GpuBrickPatch`] (resident bricks ONLY, in slot order, with re-based voxel
    /// offsets + palette + NEE lights) from the packer's shadow state — the shape `pack_resident_set` produces,
    /// so the existing render/upload/shader path consumes it unchanged. This is the live re-pack output: it is
    /// assembled by MEMCPY of the cached per-brick bytes (NOT by re-`pack_one`'ing every brick — the
    /// [`update`](Self::update) already re-packed only the O(changed) bricks), so it is far cheaper than a
    /// from-scratch [`pack_resident_set`](super::gpu::pack_resident_set). Byte-identical to a from-scratch pack
    /// for the same resident set (the A/B test proves it), so the render is pixel-identical.
    ///
    /// Slot order here is the packer's free-list order, NOT the from-scratch `(lod,z,y,x)` sort — but the shader
    /// reads everything from `metas[primitive_index].world_min`/`lod` (never the order), so the render is
    /// identical regardless. (The GPU oracle keys on `(chunk, (lod,z,y,x))` content, not raw slot.)
    pub fn snapshot_patch(&self, registry: &super::palette::BlockRegistry) -> super::gpu::GpuBrickPatch {
        use super::gpu::GpuBrickPatch;
        // Resident slots in ascending slot order (a stable, reproducible order).
        let mut slots: Vec<u32> = self.resident.values().map(|s| s.slot).collect();
        slots.sort_unstable();
        let mut patch = GpuBrickPatch {
            aabbs: Vec::with_capacity(slots.len()),
            metas: Vec::with_capacity(slots.len()),
            voxels: Vec::with_capacity(slots.len() * dense_block_u32()),
            brick_palettes: Vec::new(),
            palette: Vec::new(),
            lights: Vec::new(),
            alias: Vec::new(),
        };
        // R3: dedup identical haloed slices in the live re-pack too, the SAME way `pack_resident_set` does.
        let mut interner = super::gpu::VoxelInterner::new();
        for slot in slots {
            let aabb = *self.last_aabb.get(&slot).expect("resident slot has an aabb shadow");
            let meta = *self.last_meta.get(&slot).expect("resident slot has a meta shadow");
            patch.aabbs.push(aabb);
            if meta.is_uniform() {
                patch.metas.push(meta);
            } else {
                // R2b + R3 — re-encode the cached RAW haloed cells into a per-brick palette + bit-packed index
                // stream in the CONTIGUOUS output buffers (the live shadow keeps raw cells; encoding happens here
                // at snapshot time), deduping identical slices so a repeated brick shares one (index, palette)
                // pair. `last_voxels` is RAW cells, so the encode is byte-identical to `pack_resident_set`'s.
                let cells = self.last_voxels.get(&slot).expect("dense slot has a voxel shadow");
                let layout = interner.intern_paletted(&mut patch.voxels, &mut patch.brick_palettes, cells);
                let rebased = super::gpu::GpuBrickMeta::dense(
                    meta.voxel_origin,
                    layout.voxel_offset,
                    meta.world_min,
                    meta.lod(),
                    layout.index_bits,
                    layout.palette_base,
                );
                patch.metas.push(rebased);
            }
        }
        // Keep the index + palette buffers non-empty for upload (mirrors `pack_resident_set`'s
        // `ensure_voxels_nonempty`).
        if patch.voxels.is_empty() {
            patch.voxels.push(0);
        }
        if patch.brick_palettes.is_empty() {
            patch.brick_palettes.push(0);
        }
        // Palette + NEE light list — derived from the assembled buffers via the SHARED gpu tail (one SSOT).
        super::gpu::finalize_patch_palette_and_lights(&mut patch, registry);
        patch
    }
}

#[cfg(test)]
mod tests;
