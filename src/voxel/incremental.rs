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
    BrickVoxels, GpuBrickAabb, GpuBrickMeta, GpuPaletteColor, PackedBrick, ResidentBrick, encode_paletted,
    halo_cells, pack_one,
};
use super::streaming::BrickKey;

/// The fixed number of `u32`s a DENSE brick's RAW haloed `10³` grid occupies (one `u32` id per cell). CONSTANT at
/// every LOD ([`halo_cells`]`(0) == halo_cells(lod)`). The packer's per-slot SHADOW (`last_voxels`) stores cells
/// in this raw form (so the byte-identity gate + the A4.4 re-encode are exact); the GPU arena stores the
/// PALETTED index stream (≤ this size — see [`index_class_words`]).
#[inline]
pub fn dense_block_u32() -> usize {
    halo_cells(0)
}

/// **A4.4 — the 5 paletted-index SIZE CLASSES**, keyed by `index_bits ∈ {1,2,4,8,16}`. A dense brick's bit-packed
/// index stream is `ceil(dense_block_u32() · index_bits / 32)` words; for the `10³` haloed block that is
/// `{32, 63, 125, 250, 500}` words. Each class is a fixed-size free-list (no fragmentation WITHIN a class), so a
/// freed block is always exactly reusable by another brick of the same width.
const INDEX_CLASS_BITS: [u8; 5] = [1, 2, 4, 8, 16];

/// Words a dense brick's bit-packed index stream occupies for `index_bits ∈ {1,2,4,8,16}` (its size-class block
/// size). Mirror of [`encode_paletted`]'s `indices.len()`.
#[inline]
pub fn index_class_words(index_bits: u8) -> usize {
    (dense_block_u32() * index_bits as usize).div_ceil(32)
}

/// The size-class slot index (`0..5`) for an `index_bits ∈ {1,2,4,8,16}`.
#[inline]
fn index_class_of(index_bits: u8) -> usize {
    match index_bits {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        16 => 4,
        _ => unreachable!("index_bits must be a power of 2 in {{1,2,4,8,16}}, got {index_bits}"),
    }
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
/// AABB, and — for a dense brick whose content changed — the PALETTED index block + per-brick palette block to
/// write (A4.4). The GPU uploader patches `metas[slot]`, `aabbs[slot]`, the index arena at `index_word_offset`,
/// and `brick_palettes` at `palette_word_offset` from this.
#[derive(Clone, Debug)]
pub struct ChangedSlot {
    /// The slot whose buffers this patches (= `primitive_index`).
    pub slot: u32,
    /// The new per-brick meta (carries the real `voxel_offset` (index-arena word) + `index_bits` + `palette_base`
    /// for a dense brick, or the uniform flag/id; all-zero for a freed slot).
    pub meta: GpuBrickMeta,
    /// The new BLAS AABB ([`degenerate_aabb`] for a freed slot).
    pub aabb: GpuBrickAabb,
    /// `Some(words)` for a DENSE brick whose voxel content changed: the bit-packed INDEX stream
    /// ([`index_class_words`]`(index_bits)` words) to write at `index_word_offset`. `None` for a uniform/freed
    /// slot, or a dense brick whose meta changed but whose voxel content did not.
    pub index: Option<Vec<u32>>,
    /// The index-arena WORD offset the index block goes at (= `meta.dense_offset()` when `index.is_some()`).
    pub index_word_offset: u32,
    /// `Some(ids)` for a DENSE brick whose voxel content changed: the per-brick palette (its `k` distinct block
    /// ids, one `u32` each) to write into `brick_palettes` at `palette_word_offset`. Paired with `index` (same
    /// `Some`/`None`).
    pub palette: Option<Vec<u32>>,
    /// The `brick_palettes` WORD offset the palette block goes at (= `meta.palette_base` = `slot · palette_stride`
    /// in Checkpoint-1's fixed palette).
    pub palette_word_offset: u32,
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
/// **A4.4 representation — the streamed arena is R2b PALETTED (size-class slabs).** Each dense slot's voxel block
/// is a bit-packed INDEX stream ([`encode_paletted`]) living in its `index_bits` size-class slab in [`Self::indices`]
/// at `meta.voxel_offset`, and a per-brick PALETTE (its `k` distinct ids) in [`Self::brick_palettes`] at
/// `meta.palette_base` (Checkpoint-1: `slot · palette_stride`). The shader's `cell_block` paletted branch
/// (`index_bits >= 1`) decodes `id = brick_palettes[palette_base + (indices[voxel_offset + ..] >> .. & mask)]` —
/// the SAME decode the static `pack_brickmap`/`pack_resident_set` path uses. This recovers R2b's voxel-VRAM win on
/// the STREAMED path while keeping A1's O(changed) upload — see `PHASE_A_GPU_EXECUTION.md` §A4.4 + the scoping note.
#[derive(Clone, Debug)]
pub struct SnapshotBuffers {
    /// Capacity-length AABB buffer; unused slots = [`degenerate_aabb`].
    pub aabbs: Vec<GpuBrickAabb>,
    /// Capacity-length meta buffer; unused slots = [`GpuBrickMeta::zeroed`].
    pub metas: Vec<GpuBrickMeta>,
    /// The index-arena slab pool ([`Self::index_capacity_u32`] words): each resident dense brick's bit-packed
    /// index stream written at its slab offset (`meta.voxel_offset`); zero elsewhere.
    pub indices: Vec<u32>,
    /// The per-brick PALETTE buffer (`capacity · palette_stride` words): each resident dense brick's `k` distinct
    /// block ids written at `slot · palette_stride` (Checkpoint-1 fixed palette); zero elsewhere. Bound at
    /// `group(0)/binding(12)`. A `RepackDelta` patches only changed slots' blocks.
    pub brick_palettes: Vec<u32>,
    /// The registry palette (`BlockId(i)` → linear RGBA + emissive). Length == registry length. Fixed per scene
    /// (never re-uploaded on a [`RepackDelta`]).
    pub palette: Vec<GpuPaletteColor>,
    /// The number of resident bricks (== the live BLAS primitive count; the BLAS is still built over `capacity`
    /// primitives with degenerate free slots, but this is reported for diagnostics).
    pub brick_count: u32,
}

/// A DENSE brick's index-slab allocation (A4.4): where its bit-packed index stream lives in the index arena + its
/// width (which size class to free it back to on a drop/width-change).
#[derive(Clone, Copy, Debug)]
struct DenseSlot {
    /// Word offset of this brick's index stream in the index arena (= `meta.voxel_offset`).
    index_offset: u32,
    /// Index bit width ∈ {1,2,4,8,16} — the size class ([`index_class_of`]).
    index_bits: u8,
}

/// One resident brick's live state in the packer: its slot + (for a dense brick) its index-slab allocation.
#[derive(Clone, Copy, Debug)]
struct SlotState {
    slot: u32,
    /// `Some` for a DENSE brick (its index-slab allocation); `None` for a UNIFORM brick (no index/palette block).
    dense: Option<DenseSlot>,
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
    /// Live resident bricks → their slot + index-slab allocation.
    resident: FxHashMap<BrickKey, SlotState>,
    /// **A4.4 index-arena SIZE-CLASS SLABS.** A bump pointer (`index_high_water`, in words) + a per-class free-list
    /// of freed block word-offsets ([`INDEX_CLASS_BITS`]). A dense brick's index stream is allocated from its
    /// `index_bits` class; a freed block returns to that class's list (exactly reusable). The arena GROWS on
    /// overflow past [`Self::index_gpu_cap`] — `grew` then forces the next upload to a StreamSnapshot.
    index_high_water: u32,
    index_free: [Vec<u32>; INDEX_CLASS_BITS.len()],
    /// The index-arena WORD capacity the LAST [`snapshot_buffers`](Self::snapshot_buffers) sized the GPU buffer to.
    /// A delta-time allocation past this sets [`Self::grew`].
    index_gpu_cap: u32,
    /// Per-slot palette STRIDE in `brick_palettes` (Checkpoint-1 fixed palette = `MAX_PAL`): a dense slot's palette
    /// lives at `slot · palette_stride`. Set from the registry length each [`update`](Self::update) — `k ≤
    /// registry.len()` so a brick's palette never overflows its stride (correct-by-construction). Checkpoint-2
    /// makes the palette a slab too.
    palette_stride: u32,
    /// Set when an index allocation since the last snapshot exceeded [`Self::index_gpu_cap`] — the next upload must
    /// be a StreamSnapshot (re-allocate the larger index buffer), not a Delta. Cleared by `snapshot_buffers`.
    grew: bool,
    /// Shadow of the last-uploaded bytes per slot — the byte-compare source so a re-pack that reproduces the
    /// same bytes (a neighbour that did not actually change) costs no upload. `last_voxels` holds RAW haloed cells
    /// (the A4.4 re-encode source — kept raw so the byte-identity gate is exact).
    last_meta: FxHashMap<u32, GpuBrickMeta>,
    last_aabb: FxHashMap<u32, GpuBrickAabb>,
    last_voxels: FxHashMap<u32, Vec<u32>>,
    /// Keys the caller explicitly marked rewritten (edit / dig re-source) since the last update — re-packed even
    /// though they neither entered nor dropped. Drained each update.
    pending_rewrites: Vec<BrickKey>,
    /// DEFERRED-FREE quarantine (keep-old-until-revealed): slots/index-blocks dropped THIS update can't be
    /// reused until the NEXT update, so an in-flight frame tracing the old generation never sees a slot's bytes
    /// overwritten by a different brick mid-flight. Released at the top of the next update.
    quarantine_slots: Vec<u32>,
    /// Freed index blocks `(word_offset, index_bits)` — returned to their size-class free-list next update.
    quarantine_index: Vec<(u32, u8)>,
}

impl ResidentPacker {
    /// A fresh packer sized for `max_resident_bricks` slots. The index arena (A4.4 size-class slabs) starts empty
    /// and grows as bricks are slotted; the render path sizes the GPU index buffer to [`index_capacity_u32`](Self::index_capacity_u32)
    /// at each snapshot. `palette_stride` starts 0 and is set from the registry length on the first [`update`].
    pub fn new(max_resident_bricks: u32) -> Self {
        Self {
            slots: SlotAllocator::new(max_resident_bricks),
            resident: FxHashMap::default(),
            index_high_water: 0,
            index_free: std::array::from_fn(|_| Vec::new()),
            index_gpu_cap: 0,
            palette_stride: 0,
            grew: false,
            last_meta: FxHashMap::default(),
            last_aabb: FxHashMap::default(),
            last_voxels: FxHashMap::default(),
            pending_rewrites: Vec::new(),
            quarantine_slots: Vec::new(),
            quarantine_index: Vec::new(),
        }
    }

    /// The fixed slot CAPACITY (= the meta/AABB buffer length, in bricks).
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.slots.capacity
    }

    /// The index-arena capacity in `u32`s the render path must allocate for the index buffer — the current
    /// high-water plus headroom (so a few deltas can allocate without forcing an immediate re-snapshot). Set as
    /// [`Self::index_gpu_cap`] by [`snapshot_buffers`](Self::snapshot_buffers).
    #[inline]
    pub fn index_capacity_u32(&self) -> usize {
        // Headroom: 2× the live high-water, with a small absolute floor so an empty/tiny arena still has slack.
        (self.index_high_water as usize * 2).max(self.index_high_water as usize + 8192).max(index_class_words(16))
    }

    /// The per-slot palette buffer capacity in `u32`s (`capacity · palette_stride`) — the `brick_palettes` buffer
    /// length the render path must allocate (Checkpoint-1 fixed palette).
    #[inline]
    pub fn palette_capacity_u32(&self) -> usize {
        self.slots.capacity as usize * self.palette_stride as usize
    }

    /// Number of resident bricks currently slotted.
    #[inline]
    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    /// Build the FULL initial GPU buffers (storage plan A1 — the O(changed) upload). Called ONCE per scene epoch
    /// (and again on a grow), right after the first [`update`](Self::update) ran. `aabbs`/`metas` are
    /// capacity-length with [`degenerate_aabb`]/[`GpuBrickMeta::zeroed`] in unused slots; `indices` is the
    /// [`index_capacity_u32`](Self::index_capacity_u32) slab pool with each resident dense brick's bit-packed
    /// index stream copied to its slab offset (`meta.voxel_offset`); `brick_palettes` is the
    /// [`palette_capacity_u32`](Self::palette_capacity_u32) buffer with each dense brick's palette at
    /// `slot · palette_stride`. O(resident) — paid ONCE per epoch/grow, never per move. After this the render path
    /// applies each [`RepackDelta`] via `queue_write_buffer` (meta/aabb at `slot · stride`, index at
    /// `index_word_offset`, palette at `palette_word_offset`), never re-creating these buffers within the epoch.
    ///
    /// The bytes come from the per-slot shadow (`last_meta`/`last_aabb` + the RE-ENCODED `last_voxels` raw cells),
    /// which after `update` is EXACTLY the state a `snapshot_buffers`-then-delta sequence converges to (the
    /// byte-identity gate). Re-encoding `last_voxels` here with [`encode_paletted`] reproduces the SAME index/
    /// palette bytes `emit_changed_slot` shipped (the shadow stores the offsets in `metas`). The `palette` is the
    /// registry (fixed per scene); the NEE light list is built separately by the caller.
    pub fn snapshot_buffers(&mut self, registry: &super::palette::BlockRegistry) -> SnapshotBuffers {
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
        // A4.4 index slabs + per-brick palettes: each resident DENSE slot's bit-packed index stream at its slab
        // offset + its `k` palette ids at `slot · palette_stride`. `last_voxels` holds the RAW cells; re-encode
        // them (byte-identical to what `emit_changed_slot` shipped). The meta's `voxel_offset`/`palette_base`/
        // `index_bits` are the canonical offsets (written by `emit_changed_slot`), so writing there keeps a fresh
        // snapshot byte-identical to seed+deltas. A uniform brick has no dense slot / voxel shadow — its id rides
        // in the meta.
        let index_cap = self.index_capacity_u32();
        let mut indices = vec![0u32; index_cap];
        let mut brick_palettes = vec![0u32; self.palette_capacity_u32()];
        for st in self.resident.values() {
            let Some(dense) = st.dense else { continue };
            let Some(cells) = self.last_voxels.get(&st.slot) else { continue };
            let pb = encode_paletted(cells);
            debug_assert_eq!(pb.index_bits, dense.index_bits, "shadow index_bits disagrees with re-encode");
            let ioff = dense.index_offset as usize;
            indices[ioff..ioff + pb.indices.len()].copy_from_slice(&pb.indices);
            let poff = st.slot as usize * self.palette_stride as usize;
            for (j, &id) in pb.palette.iter().enumerate() {
                brick_palettes[poff + j] = id as u32;
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
        // Record the committed GPU buffer size for grow detection, and clear the grow flag (this snapshot IS the
        // (re)allocation to `index_cap`). A subsequent delta-time allocation past `index_cap` re-sets `grew`.
        self.index_gpu_cap = index_cap as u32;
        self.grew = false;
        SnapshotBuffers { aabbs, metas, indices, brick_palettes, palette, brick_count: self.resident.len() as u32 }
    }

    /// Peek the GROW signal without clearing it (the render path's snapshot-vs-delta decision). True iff an index
    /// allocation since the last snapshot exceeded the committed GPU buffer capacity — ship a StreamSnapshot.
    #[inline]
    pub fn grew(&self) -> bool {
        self.grew
    }

    /// Mark `keys` as REWRITTEN (an edit / dig re-source replaced their voxels in place): they re-pack on the
    /// next [`update`](Self::update) even though they neither entered nor dropped. Mirrors the manager's
    /// `requeue_keys` so the edit/dig path stays incremental (only the affected bricks + their 26-neighbourhood
    /// re-pack). A key that is not resident on the next update is ignored (it may enter then, taking the normal
    /// entered path).
    pub fn mark_rewritten(&mut self, keys: impl IntoIterator<Item = BrickKey>) {
        self.pending_rewrites.extend(keys);
    }

    /// Claim a WORD offset for a dense brick's index stream of width `index_bits` from its size-class slab: reuse a
    /// freed block of that class (exact size match), else bump the high-water by the class block size. Sets
    /// [`Self::grew`] if the bump pushes past the committed GPU buffer capacity (the next upload re-snapshots into a
    /// larger buffer). Always succeeds — the index arena grows on demand (the slot cap binds first).
    fn claim_index(&mut self, index_bits: u8) -> u32 {
        let class = index_class_of(index_bits);
        if let Some(off) = self.index_free[class].pop() {
            return off;
        }
        let off = self.index_high_water;
        self.index_high_water += index_class_words(index_bits) as u32;
        if self.index_high_water > self.index_gpu_cap {
            self.grew = true;
        }
        off
    }

    /// Incrementally reconcile the packer toward `entries` (the manager's `resident_entries()`, in the SSOT
    /// `(lod,z,y,x)` order). Returns the [`RepackDelta`] of slots whose GPU bytes changed — O(changed + halo),
    /// never O(resident). The CALLER uploads only `delta.changed` via `queue_write_buffer`. `palette_stride` is the
    /// per-slot palette stride (= the packing registry's length, which bounds a brick's `k`) — constant within an
    /// epoch; it sizes `brick_palettes` and the dense metas' `palette_base = slot · palette_stride` (A4.4).
    pub fn update(&mut self, entries: &[ResidentBrick<'_>], palette_stride: u32) -> RepackDelta {
        debug_assert!(
            self.palette_stride == 0 || self.palette_stride == palette_stride,
            "palette_stride must be constant within an epoch ({} → {palette_stride})",
            self.palette_stride,
        );
        self.palette_stride = palette_stride;
        // (1) Deferred-free: last update's quarantined slots/index blocks are now safe to reuse (the frame that
        // could still be tracing them has been submitted). Release them BEFORE claiming this update's slots.
        for s in self.quarantine_slots.drain(..) {
            self.slots.release(s);
        }
        for (off, bits) in std::mem::take(&mut self.quarantine_index) {
            self.index_free[index_class_of(bits)].push(off);
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
                index: None,
                index_word_offset: 0,
                palette: None,
                palette_word_offset: 0,
            });
            delta.freed.push(st.slot);
            self.quarantine_slots.push(st.slot);
            self.last_meta.insert(st.slot, GpuBrickMeta::zeroed());
            self.last_aabb.insert(st.slot, degenerate_aabb());
            self.last_voxels.remove(&st.slot);
            if let Some(d) = st.dense {
                self.quarantine_index.push((d.index_offset, d.index_bits));
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
            self.resident.insert(key, SlotState { slot, dense: None });
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

    /// Write `pb`'s bytes into `key`'s slot, allocating/freeing/re-classing its index slab as the dense/uniform
    /// classification (and the dense brick's `index_bits` size class) changed, and push a [`ChangedSlot`] iff the
    /// bytes actually differ from the slot's shadow. The dense path RE-ENCODES the brick's haloed cells into a
    /// per-brick palette + bit-packed index stream ([`encode_paletted`], A4.4) and writes them into the index slab
    /// and the fixed per-slot palette block — the SAME (palette, indices) bytes `snapshot_buffers` reproduces from
    /// the raw `last_voxels` shadow (the byte-identity gate). `last_voxels` keeps RAW cells so the re-encode is exact.
    fn emit_changed_slot(&mut self, key: BrickKey, pb: &PackedBrick, delta: &mut RepackDelta) {
        let st = *self.resident.get(&key).expect("dirty key is resident");
        match &pb.voxels {
            BrickVoxels::Uniform(_) => {
                // Uniform now: free any index slab it held (→ quarantine, keep-old).
                if let Some(d) = st.dense {
                    self.quarantine_index.push((d.index_offset, d.index_bits));
                }
                let meta = pb.meta_uniform();
                self.resident.insert(key, SlotState { slot: st.slot, dense: None });
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
                        index: None,
                        index_word_offset: 0,
                        palette: None,
                        palette_word_offset: 0,
                    });
                }
            }
            BrickVoxels::Dense(cells) => {
                // Encode the haloed cells → per-brick palette + bit-packed index stream. `index_bits` picks the
                // size class; `k = palette.len() ≤ palette_stride` (the registry bounds it — correct by construction).
                let enc = encode_paletted(cells);
                debug_assert!(
                    enc.palette.len() as u32 <= self.palette_stride,
                    "brick palette k={} exceeds palette_stride={} (registry too large for fixed palette?)",
                    enc.palette.len(),
                    self.palette_stride,
                );
                // Ensure an index slab block of the RIGHT class: reuse the existing block iff its class already
                // matches; else free the old one (→ quarantine) and claim a new one of the new class.
                let index_offset = match st.dense {
                    Some(d) if d.index_bits == enc.index_bits => d.index_offset,
                    Some(d) => {
                        self.quarantine_index.push((d.index_offset, d.index_bits));
                        self.claim_index(enc.index_bits)
                    }
                    None => self.claim_index(enc.index_bits),
                };
                let palette_base = st.slot * self.palette_stride;
                let meta = super::gpu::GpuBrickMeta::dense(
                    pb.voxel_origin,
                    index_offset,
                    pb.world_min,
                    pb.lod,
                    enc.index_bits,
                    palette_base,
                );
                self.resident
                    .insert(key, SlotState { slot: st.slot, dense: Some(DenseSlot { index_offset, index_bits: enc.index_bits }) });
                let meta_changed =
                    self.last_meta.get(&st.slot) != Some(&meta) || self.last_aabb.get(&st.slot) != Some(&pb.aabb);
                let voxels_changed = self.last_voxels.get(&st.slot) != Some(cells);
                self.last_meta.insert(st.slot, meta);
                self.last_aabb.insert(st.slot, pb.aabb);
                if voxels_changed {
                    self.last_voxels.insert(st.slot, cells.clone());
                }
                if meta_changed || voxels_changed {
                    // The index + palette blocks are (re-)written exactly when the brick's content changed (a moved
                    // slab — class change / uniform→dense — implies new content, so `voxels_changed` covers it).
                    let (index, palette) = if voxels_changed {
                        (Some(enc.indices), Some(enc.palette.into_iter().map(|id| id as u32).collect::<Vec<u32>>()))
                    } else {
                        (None, None)
                    };
                    delta.changed.push(ChangedSlot {
                        slot: st.slot,
                        meta,
                        aabb: pb.aabb,
                        index,
                        index_word_offset: index_offset,
                        palette,
                        palette_word_offset: palette_base,
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
