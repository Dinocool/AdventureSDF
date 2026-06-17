//! **Phase G "G-c.0" — the GPU-resident sparse brick OCCUPANCY structure** (`docs/PHASE_G_GC_PLAN.md` §2.2).
//!
//! The FIRST, prerequisite slice of the GPU-driven readback-free streaming pivot: a GPU-resident, per-LOD
//! sparse occupancy structure that answers — cheaply, on the GPU — `is_occupied(brick_coord, lod) -> bool`
//! (the face-cull input the next stage's GPU enumeration will read). It is the dubiousconst282
//! **sector alloc-mask** model (`VoxelNotes.md:289`):
//!
//! * Bricks are grouped into **`SECTOR_EDGE³ = 4³ = 64`-brick SECTORS**; each sector carries ONE 64-bit mask
//!   (stored as `2×u32` — WGSL storage buffers have no `u64`), one bit per brick: set ⇔ that `(coord, lod)`
//!   brick is OCCUPIED (present in the source's brick set at that LOD, i.e. [`OccupancyOracle::is_occupied`]).
//! * **Sparse:** only sectors that contain ≥ 1 occupied brick are stored. The compact sector table is built
//!   ONCE on the CPU and uploaded into a GPU open-addressing HASH (keyed by `(sector_coord, lod)`), so a far,
//!   mostly-empty clipmap costs ~1 bit per occupied brick, not the cube volume.
//! * **Per-LOD:** the `lod` is part of the hash key, so all `MAX_LOD + 1` levels share one table namespace
//!   (a coarse sector and a fine sector of the same coord never collide).
//!
//! From ONE fetch the structure answers BOTH the per-brick face test (`is_occupied`) AND the coarse
//! "is any brick in this sector occupied?" ([`mask != 0`] — the §1 Pass B0 coarse occupancy test).
//!
//! ## Wired to NOTHING (G-c.0 = no behaviour change)
//! This builds + uploads the structure; it is bound to NO existing pipeline / shader. The live CPU
//! residency/pack/render pipeline is untouched. The deliverable is the structure + its GPU-vs-CPU parity gate
//! (`tests/voxel_gpu_residency_parity.rs`). The consumer (GPU enumerate + face-cull, Pass B/B0) lands in
//! G-c.1; the brick-core store (§2.4) in G-c.2. The WGSL `is_occupied` helper lives in
//! `assets/shaders/voxel_residency.wgsl` for those stages (it needs no pipeline yet).
//!
//! ## Source of truth (so the parity test is meaningful)
//! Occupancy is defined as the SAME predicate the CPU residency uses to decide "this brick exists":
//! [`crate::voxel::source::BrickSource::classify`]` != `[`crate::voxel::source::BrickClass::Air`]. A present
//! brick (Surface OR Interior) is occupied; an absent brick is `Air`. The [`OccupancyOracle`] trait captures
//! exactly that, and [`SectorOccupancy::from_oracle`] enumerates the source's candidate set
//! ([`BrickSource::surface_bricks_in`], a SUPERSET of the occupied bricks) and keeps the ones the oracle marks
//! occupied — so the GPU masks are built from the SAME occupied set `classify`/`surface_bricks_in` see.

use bevy::math::IVec3;
use bytemuck::{Pod, Zeroable};

use super::brickmap::MAX_LOD;
use super::source::{BrickClass, BrickSource};

/// Bricks per SECTOR axis: a `4³ = 64`-brick sector ⇒ one 64-bit alloc mask (the dubiousconst282 model). A
/// power of two so the brick→sector split is a cheap `div_euclid` / `rem_euclid` (negative-coord-correct).
pub const SECTOR_EDGE: i32 = 4;

/// Bricks per sector (`SECTOR_EDGE³ = 64`) = bits per sector mask.
pub const SECTOR_BRICKS: usize = (SECTOR_EDGE * SECTOR_EDGE * SECTOR_EDGE) as usize;

/// The hash-table EMPTY-slot sentinel, stored in a slot's `lod` field. `lod` is a real level only in
/// `0..=MAX_LOD` (≤ 7), so `u32::MAX` can never be a live key — an unambiguous empty marker.
pub const EMPTY_LOD: u32 = u32::MAX;

/// The hash-table load factor cap: the table is sized to `next_pow2(occupied_sectors / LOAD_FACTOR)` so
/// open-addressing linear probing stays short (≈ `1/(1-LF)` expected probes). `0.5` ⇒ ≤ ~2 probes average.
const LOAD_FACTOR: f64 = 0.5;

/// The local bit index `[0, SECTOR_BRICKS)` of a brick within its sector, from the brick's in-sector local
/// coord `l ∈ [0, SECTOR_EDGE)³`. +X fastest, then +Y, then +Z (the brickmap voxel-index convention scaled to
/// the sector grid) — the SSOT shared by the CPU build and the WGSL `is_occupied` (they MUST agree bit-for-bit).
#[inline]
pub fn sector_bit_index(local: IVec3) -> u32 {
    debug_assert!(
        (0..SECTOR_EDGE).contains(&local.x)
            && (0..SECTOR_EDGE).contains(&local.y)
            && (0..SECTOR_EDGE).contains(&local.z)
    );
    (local.x + local.y * SECTOR_EDGE + local.z * SECTOR_EDGE * SECTOR_EDGE) as u32
}

/// Split a brick `coord` into `(sector_coord, in_sector_local)` — `sector = coord.div_euclid(SECTOR_EDGE)`,
/// `local = coord.rem_euclid(SECTOR_EDGE)`. Euclidean (NOT `>>` / `&`) so it is correct for NEGATIVE brick
/// coords (the clipmap reaches both signs). SSOT shared by the CPU build + the WGSL helper.
#[inline]
pub fn split_sector(coord: IVec3) -> (IVec3, IVec3) {
    let sector = IVec3::new(
        coord.x.div_euclid(SECTOR_EDGE),
        coord.y.div_euclid(SECTOR_EDGE),
        coord.z.div_euclid(SECTOR_EDGE),
    );
    let local = IVec3::new(
        coord.x.rem_euclid(SECTOR_EDGE),
        coord.y.rem_euclid(SECTOR_EDGE),
        coord.z.rem_euclid(SECTOR_EDGE),
    );
    (sector, local)
}

/// The 32-bit hash of a sector key `(sector_coord, lod)` (the FNV-1a-style mix used to place a sector in the
/// open-addressing table). The SSOT shared by the CPU build + the WGSL helper — they MUST compute the SAME
/// hash so a GPU probe walks the SAME slot sequence the CPU build wrote. Uses `wrapping` u32 arithmetic so it
/// is bit-identical to the WGSL (which is modular-u32 by definition).
#[inline]
pub fn sector_hash(sector: IVec3, lod: u32) -> u32 {
    // A small splittable-mix over the four key words. `as u32` reinterprets the (possibly negative) coord's
    // two's-complement bits — identical to WGSL's `bitcast<u32>(i32)`.
    let mut h: u32 = 2166136261;
    for w in [sector.x as u32, sector.y as u32, sector.z as u32, lod] {
        h ^= w;
        h = h.wrapping_mul(16777619);
        // An extra avalanche step so spatially-adjacent sectors don't cluster into one probe chain.
        h ^= h >> 15;
        h = h.wrapping_mul(2654435761);
        h ^= h >> 13;
    }
    h
}

/// One sector record in the GPU open-addressing hash table. `lod == `[`EMPTY_LOD`] marks a FREE slot. 32 bytes,
/// `bytemuck`-uploadable. The WGSL mirror reads the same 8 `u32` stride.
///
/// `mask_lo`/`mask_hi` are the low/high 32 bits of the 64-bit OCCUPANCY (presence) mask — bit `b`
/// (`= `[`sector_bit_index`]) is set ⇔ the `(sector·SECTOR_EDGE + local, lod)` brick is OCCUPIED (present).
/// `full_lo`/`full_hi` are the same split of the 64-bit FULL mask — bit `b` set ⇔ that brick is present AND
/// fully solid ([`Brick::is_full`](super::brickmap::Brick::is_full)). The face-cull (Pass B / G-c.1) needs BOTH
/// to mirror [`super::source::BrickSource::classify`] EXACTLY: `Interior` (occluded) iff the brick AND all 6
/// face-neighbours are FULL, so a present-but-PARTIAL brick is always `Surface`. Masks split as `2×u32` because
/// WGSL storage buffers have no `u64`. `full ⊆ occupancy` by construction (a brick is full only if present).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuSectorEntry {
    /// The sector coord (`coord.div_euclid(SECTOR_EDGE)`), two's-complement bits (`bitcast` on the GPU).
    pub sector_x: i32,
    pub sector_y: i32,
    pub sector_z: i32,
    /// The LOD level this sector lives at (`0..=MAX_LOD`), or [`EMPTY_LOD`] for a free slot.
    pub lod: u32,
    /// Low 32 bits of the 64-bit OCCUPANCY (presence) mask.
    pub mask_lo: u32,
    /// High 32 bits of the 64-bit OCCUPANCY (presence) mask.
    pub mask_hi: u32,
    /// Low 32 bits of the 64-bit FULL (fully-solid) mask — a subset of the occupancy mask.
    pub full_lo: u32,
    /// High 32 bits of the 64-bit FULL (fully-solid) mask — a subset of the occupancy mask.
    pub full_hi: u32,
}

/// The small uniform/header the WGSL helper needs to address the table: the slot count (a power of two, so the
/// probe wraps with `& (table_size - 1)`). 16 bytes (padded to a `vec4`-friendly stride for a uniform buffer).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuResidencyHeader {
    /// Number of slots in `entries` — ALWAYS a power of two (the WGSL masks the probe with `table_size - 1`).
    pub table_size: u32,
    pub _pad: [u32; 3],
}

/// The trait the occupancy build queries: "is the `(coord, lod)` brick OCCUPIED?" — the SAME predicate the CPU
/// residency uses to decide a brick exists (`classify != Air`). Implemented for any [`BrickSource`] below, so a
/// static `.vxo` / merged source feeds the build directly; the parity test also implements it over a known map
/// so the gate is a pure GPU-vs-CPU oracle comparison.
pub trait OccupancyOracle {
    /// True iff the brick at `(coord, lod)` is occupied (present in the source's brick set at that LOD).
    fn is_occupied(&self, coord: IVec3, lod: u32) -> bool;

    /// True iff the brick at `(coord, lod)` is present AND FULLY SOLID
    /// ([`Brick::is_full`](super::brickmap::Brick::is_full)) — the input the GPU face-cull needs to reproduce
    /// [`BrickSource::classify`]'s `Interior` test. `false` for an absent or partial brick.
    fn is_full(&self, coord: IVec3, lod: u32) -> bool;

    /// The CANDIDATE bricks (a SUPERSET of the occupied set) intersecting the inclusive brick-coord box
    /// `[lo, hi]` at `lod` — the build enumerates these and keeps the [`is_occupied`](Self::is_occupied) ones.
    /// For a [`BrickSource`] this is `surface_bricks_in` (the sparse stored set clipped to the box), so the
    /// build never scans the box VOLUME.
    fn candidates_in(&self, lo: IVec3, hi: IVec3, lod: u32, out: &mut Vec<IVec3>);
}

/// A [`BrickSource`] is an occupancy oracle: a brick is occupied iff `classify != Air`, and its candidate set
/// is `surface_bricks_in` (the source's stored bricks clipped to the box — a superset of the occupied set).
impl<S: BrickSource + ?Sized> OccupancyOracle for S {
    #[inline]
    fn is_occupied(&self, coord: IVec3, lod: u32) -> bool {
        self.classify(coord, lod) != BrickClass::Air
    }

    /// PRESENCE-ONLY conservative full bit: the generic [`BrickSource`] cannot report a brick's `is_full`
    /// without voxelizing it (and the trait carries no registry), so this returns `false` — i.e.
    /// [`from_oracle`](SectorOccupancy::from_oracle) builds a presence-only occupancy with an all-zero FULL
    /// mask (no brick is `Interior`-eligible). That is correct but CONSERVATIVE for the face-cull (it never
    /// culls a buried brick). The exact-`classify` producers — the live [`StaticVoxSource`] build and the
    /// enumerate-parity gate — use [`StaticVoxSource::occupied_keys_full`](super::source::StaticVoxSource::occupied_keys_full)
    /// → [`from_occupied_full`](SectorOccupancy::from_occupied_full), which carries the real per-brick `is_full`.
    #[inline]
    fn is_full(&self, _coord: IVec3, _lod: u32) -> bool {
        false
    }

    #[inline]
    fn candidates_in(&self, lo: IVec3, hi: IVec3, lod: u32, out: &mut Vec<IVec3>) {
        self.surface_bricks_in(lo, hi, lod, out);
    }
}

/// The CPU-built sparse sector occupancy: per-LOD compact sector tables (CPU SSOT), plus the open-addressing
/// hash the GPU probes. Build once from an [`OccupancyOracle`] over the scene's brick-coord bounds, then
/// [`upload`](Self::upload) the two GPU buffers (header + entries). The CPU `is_occupied` here is bit-identical
/// to the WGSL helper (same `split_sector` / `sector_bit_index` / `sector_hash` SSOT), so the parity gate
/// asserts GPU == CPU == oracle exactly.
#[derive(Clone, Debug, Default)]
pub struct SectorOccupancy {
    /// The open-addressing hash slots (`table_size` long, a power of two). Free slots have `lod == `[`EMPTY_LOD`].
    entries: Vec<GpuSectorEntry>,
    /// `entries.len()` as a power of two (cached for the probe mask).
    table_size: u32,
    /// Total occupied bricks (the popcount sum) — a build statistic / sanity bound (not uploaded).
    occupied_bricks: u64,
}

impl SectorOccupancy {
    /// Build the sparse sector tables from an explicit set of occupied `(coord, lod)` bricks, with NO `full`
    /// information — every brick's FULL bit is left 0 (treated as PARTIAL). A presence-only convenience for
    /// callers that only query [`is_occupied`](Self::is_occupied) / [`sector_any_occupied`](Self::sector_any_occupied)
    /// (those ignore the full mask). With an all-zero full mask the face-cull would never classify any brick
    /// `Interior` (every present brick is `Surface`) — so callers that need exact [`classify`](super::source::BrickSource::classify)
    /// parity MUST instead use [`from_occupied_full`](Self::from_occupied_full) with the per-brick `is_full` flag.
    pub fn from_occupied(occupied: impl IntoIterator<Item = (IVec3, u32)>) -> Self {
        Self::from_occupied_full(occupied.into_iter().map(|(c, l)| (c, l, false)))
    }

    /// Build the sparse sector tables from an explicit set of `(coord, lod, is_full)` bricks — the FULL SSOT
    /// both [`from_oracle`](Self::from_oracle) and the parity test feed. `is_full` is the brick's
    /// [`Brick::is_full`](super::brickmap::Brick::is_full): set its bit in the FULL mask too, so the GPU
    /// face-cull can reproduce [`classify`](super::source::BrickSource::classify)'s `Interior` test (fully-solid
    /// brick + fully-solid 6 face-neighbours) EXACTLY. Sectors are accumulated into a temporary
    /// `(sector, lod) -> (occ_mask, full_mask)` map, then laid into a power-of-two open-addressing table sized
    /// for [`LOAD_FACTOR`]. A brick passed `is_full == true` is ALSO marked occupied (full ⊆ occupancy).
    pub fn from_occupied_full(occupied: impl IntoIterator<Item = (IVec3, u32, bool)>) -> Self {
        use rustc_hash::FxHashMap;
        let mut masks: FxHashMap<(IVec3, u32), (u64, u64)> = FxHashMap::default();
        for (coord, lod, is_full) in occupied {
            debug_assert!(lod <= MAX_LOD, "occupancy lod {lod} exceeds MAX_LOD {MAX_LOD}");
            let (sector, local) = split_sector(coord);
            let bit = sector_bit_index(local);
            let e = masks.entry((sector, lod)).or_insert((0, 0));
            e.0 |= 1u64 << bit; // occupancy (presence)
            if is_full {
                e.1 |= 1u64 << bit; // full (fully solid)
            }
        }
        Self::from_sector_masks(masks)
    }

    /// Lay a `(sector, lod) -> (occ_mask, full_mask)` map into the power-of-two open-addressing table (linear
    /// probing on [`sector_hash`]). An EMPTY table (no occupied sectors) is still given ONE empty slot so the
    /// GPU buffer is non-zero-length + every probe immediately misses (every `is_occupied` ⇒ false).
    fn from_sector_masks(masks: rustc_hash::FxHashMap<(IVec3, u32), (u64, u64)>) -> Self {
        let n = masks.len();
        let occupied_bricks: u64 = masks.values().map(|(occ, _)| occ.count_ones() as u64).sum();
        // table_size = next power of two of n / LOAD_FACTOR, at least 1.
        let target = ((n as f64) / LOAD_FACTOR).ceil() as usize;
        let table_size = target.max(1).next_power_of_two();
        let mut entries = vec![
            GpuSectorEntry {
                sector_x: 0,
                sector_y: 0,
                sector_z: 0,
                lod: EMPTY_LOD,
                mask_lo: 0,
                mask_hi: 0,
                full_lo: 0,
                full_hi: 0,
            };
            table_size
        ];
        let mask_bits = (table_size - 1) as u32;
        for ((sector, lod), (occ, full)) in masks {
            let mut slot = (sector_hash(sector, lod) & mask_bits) as usize;
            // Linear probe to the first free slot (the table is < 100% full by construction, so this terminates).
            while entries[slot].lod != EMPTY_LOD {
                slot = (slot + 1) & (mask_bits as usize);
            }
            entries[slot] = GpuSectorEntry {
                sector_x: sector.x,
                sector_y: sector.y,
                sector_z: sector.z,
                lod,
                mask_lo: occ as u32,
                mask_hi: (occ >> 32) as u32,
                full_lo: full as u32,
                full_hi: (full >> 32) as u32,
            };
        }
        Self { entries, table_size: table_size as u32, occupied_bricks }
    }

    /// Build the occupancy from an [`OccupancyOracle`] over the inclusive brick-coord bounds `[lo, hi]` at each
    /// LOD `0..=MAX_LOD`: enumerate the oracle's candidate set per LOD (a superset of the occupied bricks),
    /// keep the occupied ones, and accumulate their sector masks. `bounds` is the scene's per-LOD brick-coord
    /// extent (inclusive); a static `.vxo` / merged source knows it from its `HEAD.bounds`. For G-c.0 the build
    /// runs ONCE at scene-load (or per `.vxo` region) — not per frame.
    pub fn from_oracle<O: OccupancyOracle + ?Sized>(oracle: &O, bounds: impl Fn(u32) -> (IVec3, IVec3)) -> Self {
        use rustc_hash::FxHashMap;
        let mut masks: FxHashMap<(IVec3, u32), (u64, u64)> = FxHashMap::default();
        let mut candidates: Vec<IVec3> = Vec::new();
        for lod in 0..=MAX_LOD {
            let (lo, hi) = bounds(lod);
            if lo.x > hi.x || lo.y > hi.y || lo.z > hi.z {
                continue; // empty box at this LOD
            }
            candidates.clear();
            oracle.candidates_in(lo, hi, lod, &mut candidates);
            for &coord in &candidates {
                if oracle.is_occupied(coord, lod) {
                    let (sector, local) = split_sector(coord);
                    let bit = sector_bit_index(local);
                    let e = masks.entry((sector, lod)).or_insert((0, 0));
                    e.0 |= 1u64 << bit;
                    if oracle.is_full(coord, lod) {
                        e.1 |= 1u64 << bit;
                    }
                }
            }
        }
        Self::from_sector_masks(masks)
    }

    /// Probe the table for `(sector, lod)` and return its `(occupancy, full)` 64-bit masks, or `(0, 0)` if the
    /// sector is absent — the SINGLE fetch every CPU query below derives from (the SSOT mirror of the WGSL
    /// `sector_masks`). Hash the sector, linear-probe to the first matching slot; a free slot before a match ⇒
    /// the sector is absent. Bounded by the table size (the build keeps it < 100% full, so an absent key always
    /// hits a free slot first).
    #[inline]
    fn sector_masks(&self, sector: IVec3, lod: u32) -> (u64, u64) {
        if self.table_size == 0 {
            return (0, 0);
        }
        let mask_bits = self.table_size - 1;
        let mut slot = (sector_hash(sector, lod) & mask_bits) as usize;
        for _ in 0..self.table_size {
            let e = &self.entries[slot];
            if e.lod == EMPTY_LOD {
                return (0, 0); // first free slot ⇒ key absent
            }
            if e.lod == lod && e.sector_x == sector.x && e.sector_y == sector.y && e.sector_z == sector.z {
                let occ = (e.mask_lo as u64) | ((e.mask_hi as u64) << 32);
                let full = (e.full_lo as u64) | ((e.full_hi as u64) << 32);
                return (occ, full);
            }
            slot = (slot + 1) & (mask_bits as usize);
        }
        (0, 0)
    }

    /// The CPU mirror of the WGSL `is_occupied` — the SSOT the parity gate asserts GPU == CPU against. Test the
    /// brick's presence bit in its sector's occupancy mask.
    pub fn is_occupied(&self, coord: IVec3, lod: u32) -> bool {
        let (sector, local) = split_sector(coord);
        let bit = sector_bit_index(local);
        let (occ, _full) = self.sector_masks(sector, lod);
        (occ >> bit) & 1 != 0
    }

    /// The CPU mirror of the WGSL `is_full` — the brick is present AND fully solid. Test the brick's bit in its
    /// sector's FULL mask. The face-cull (Pass B / [`classify_surface`](Self::classify_surface)) input.
    pub fn is_full(&self, coord: IVec3, lod: u32) -> bool {
        let (sector, local) = split_sector(coord);
        let bit = sector_bit_index(local);
        let (_occ, full) = self.sector_masks(sector, lod);
        (full >> bit) & 1 != 0
    }

    /// The CPU mirror of the GPU Pass-B **6-face occlusion cull** — the SSOT the enumerate-parity gate asserts
    /// GPU == CPU == [`StaticVoxSource::classify`](super::source::StaticVoxSource::classify) against. Returns
    /// `true` iff `(coord, lod)` is a SURFACE brick: present, AND NOT fully occluded (NOT [`is_full`](Self::is_full)
    /// itself, OR at least one of its 6 same-LOD face-neighbours is not `is_full`). EXACTLY reproduces
    /// `classify == Surface` for any non-empty static scene (where every LOD maps 1:1 to a built pyramid level):
    /// * absent ⇒ `false` (`classify` ⇒ `Air`),
    /// * present & !full ⇒ `true` (`classify` ⇒ `Surface`, an internal air voxel exposes a face),
    /// * present & full & some face-neighbour !full ⇒ `true` (an exposed face),
    /// * present & full & all 6 face-neighbours full ⇒ `false` (`classify` ⇒ `Interior`, fully buried).
    pub fn classify_surface(&self, coord: IVec3, lod: u32) -> bool {
        if !self.is_occupied(coord, lod) {
            return false; // absent ⇒ Air
        }
        if !self.is_full(coord, lod) {
            return true; // present but partial ⇒ an internal air voxel exposes a face ⇒ Surface
        }
        // Fully solid: Surface iff ANY of the 6 face-neighbours is not fully solid (an exposed face); else Interior.
        const N6: [IVec3; 6] = [
            IVec3::new(1, 0, 0),
            IVec3::new(-1, 0, 0),
            IVec3::new(0, 1, 0),
            IVec3::new(0, -1, 0),
            IVec3::new(0, 0, 1),
            IVec3::new(0, 0, -1),
        ];
        for off in N6 {
            if !self.is_full(coord + off, lod) {
                return true; // a non-full / absent neighbour ⇒ this face is exposed ⇒ Surface
            }
        }
        false // all 6 face-neighbours full ⇒ no exposed face ⇒ Interior
    }

    /// The coarse "is ANY brick in this sector occupied?" — the §1 Pass B0 occupancy test, from the SAME fetch
    /// as `is_occupied` (`occ != 0`). `sector` is the sector coord (`coord.div_euclid(SECTOR_EDGE)`).
    pub fn sector_any_occupied(&self, sector: IVec3, lod: u32) -> bool {
        let (occ, _full) = self.sector_masks(sector, lod);
        occ != 0
    }

    /// The GPU header (the table size the WGSL probe masks with).
    pub fn header(&self) -> GpuResidencyHeader {
        GpuResidencyHeader { table_size: self.table_size, _pad: [0; 3] }
    }

    /// The hash-table slots to upload (the `entries` storage buffer contents).
    pub fn entries(&self) -> &[GpuSectorEntry] {
        &self.entries
    }

    /// Number of OCCUPIED sectors stored (non-empty slots) — a build statistic.
    pub fn occupied_sectors(&self) -> usize {
        self.entries.iter().filter(|e| e.lod != EMPTY_LOD).count()
    }

    /// Total occupied bricks (the popcount sum across all sector masks) — a build statistic / sanity bound.
    pub fn occupied_bricks(&self) -> u64 {
        self.occupied_bricks
    }

    /// The slot count (a power of two).
    pub fn table_size(&self) -> u32 {
        self.table_size
    }

    /// Upload the structure to the GPU as two PERSISTENT storage buffers: the header (table size) and the
    /// entries (the sparse sector hash). Returns the [`GpuResidencyBuffers`] holder (added to
    /// `VoxelRtResources` — bound to NO pipeline in G-c.0). One-time cost at scene-load (or per `.vxo` region).
    pub fn upload(&self, device: &wgpu::Device) -> GpuResidencyBuffers {
        use wgpu::util::DeviceExt;
        let header = self.header();
        let header_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel_residency_header"),
            contents: bytemuck::bytes_of(&header),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // The entries buffer is never empty (`from_sector_masks` guarantees ≥ 1 slot), so the cast is non-empty.
        let entries_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel_residency_entries"),
            contents: bytemuck::cast_slice(&self.entries),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        GpuResidencyBuffers { header: header_buf, entries: entries_buf, table_size: self.table_size }
    }

    /// **Phase G "G-c.4-paging" (§8.2)** — upload into a PRE-SIZED `entries` buffer (`entries_capacity` slots),
    /// for the GROWABLE streamed occupancy rebuilt-whole each region crossing. The `entries` GPU buffer is created
    /// once at `entries_capacity` (a whole-scene sector estimate) and re-written in place via `queue_write_buffer`
    /// on each rebuild — so a per-crossing whole re-upload costs no realloc/rebind (the consumer's bind group stays
    /// valid). Asserts `table_size <= entries_capacity` (the pre-size must cover the densest resident set; the
    /// caller sizes it from the scene's brick counts). Returns the buffers holder; subsequent rebuilds call
    /// [`Self::reupload_into`].
    pub fn upload_presized(&self, device: &wgpu::Device, queue: &wgpu::Queue, entries_capacity: u32) -> GpuResidencyBuffers {
        assert!(
            self.table_size <= entries_capacity,
            "occupancy table_size {} exceeds pre-sized capacity {entries_capacity}",
            self.table_size
        );
        let header_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_residency_header"),
            size: std::mem::size_of::<GpuResidencyHeader>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Allocate the full capacity once; write the live `table_size` slots, leave the tail EMPTY (uninitialised
        // bytes are never probed — the probe masks with `table_size - 1 < entries_capacity`).
        let entries_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_residency_entries"),
            size: (entries_capacity as u64) * std::mem::size_of::<GpuSectorEntry>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut buffers = GpuResidencyBuffers { header: header_buf, entries: entries_buf, table_size: self.table_size };
        // Initial contents via the one write SSOT.
        self.reupload_into(queue, &mut buffers, entries_capacity);
        buffers
    }

    /// **Phase G "G-c.4-paging" (§8.2)** — re-write the header + entries of a pre-sized [`GpuResidencyBuffers`] in
    /// place (the per-crossing whole rebuild). Asserts the new `table_size` fits the buffer's capacity (the caller
    /// pre-sizes generously); writes the `table_size` live slots, then updates the holder's cached `table_size`.
    /// `queue_write_buffer` is a GPU-timeline copy — no host stall, no rebind. The unused tail slots are NOT
    /// rewritten (never probed); a SHRINK leaves stale entries in the tail beyond `table_size` — harmless, the
    /// probe never reaches them (it masks with the NEW `table_size - 1`).
    pub fn reupload_into(&self, queue: &wgpu::Queue, buffers: &mut GpuResidencyBuffers, entries_capacity: u32) {
        assert!(
            self.table_size <= entries_capacity,
            "occupancy rebuild table_size {} exceeds pre-sized capacity {entries_capacity}",
            self.table_size
        );
        queue.write_buffer(&buffers.header, 0, bytemuck::bytes_of(&self.header()));
        queue.write_buffer(&buffers.entries, 0, bytemuck::cast_slice(&self.entries));
        buffers.table_size = self.table_size;
    }
}

/// **Phase G "G-c.2a"** — the 32-bit hash of a BRICK key `(coord, lod)` for the GPU residency-diff hashes
/// (`slot_table` + `present_flag`). The SAME FNV-1a + avalanche family as [`sector_hash`], but over the brick
/// coord directly (the brick coord IS the key — no sector split). The SSOT shared by the CPU reference (the
/// parity test's slot-table / present-flag oracle) and the WGSL `hash_key` — they MUST compute the SAME hash so
/// a GPU probe walks the SAME slot sequence. `wrapping` u32 to be bit-identical to the WGSL modular u32.
#[inline]
pub fn brick_key_hash(coord: IVec3, lod: u32) -> u32 {
    let mut h: u32 = 2166136261;
    for w in [coord.x as u32, coord.y as u32, coord.z as u32, lod] {
        h ^= w;
        h = h.wrapping_mul(16777619);
        h ^= h >> 15;
        h = h.wrapping_mul(2654435761);
        h ^= h >> 13;
    }
    h
}

/// **Phase G "G-c.2a"** — the GPU residency-diff (Pass C) config uniform — MUST match the WGSL `DiffConfig`
/// (4×u32 / 16 B). `slot_table_size`/`present_size` are powers of two (the WGSL probe masks with `size - 1`);
/// `max_resident` is the free-list ring capacity (= the `ResidentPacker` slot capacity, `incremental.rs:580`);
/// `refine_descent_cap` is the keep-old-until-revealed refine bound (`REFINE_DESCENT_CAP`, `streaming.rs:76`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuResidencyDiffConfig {
    /// Slot-table hash slot count (a power of two).
    pub slot_table_size: u32,
    /// Present-flag hash slot count (a power of two).
    pub present_size: u32,
    /// Free-list ring capacity (= max resident bricks).
    pub max_resident: u32,
    /// The keep-old-until-revealed refine descent cap (`REFINE_DESCENT_CAP` = 5).
    pub refine_descent_cap: u32,
}

/// The uploaded GPU occupancy buffers — held PERSISTENTLY in `VoxelRtResources` (G-c.0: bound to no pipeline;
/// the G-c.1 enumerate pass binds them). `header` is a UNIFORM (table size); `entries` is the STORAGE sector
/// hash. `table_size` is cached for the consumer to size its probe mask without reading the buffer back.
pub struct GpuResidencyBuffers {
    /// The [`GpuResidencyHeader`] uniform (table size).
    pub header: wgpu::Buffer,
    /// The [`GpuSectorEntry`] hash slots (storage).
    pub entries: wgpu::Buffer,
    /// The slot count (a power of two) — mirrors the header's `table_size`.
    pub table_size: u32,
}

/// **Phase G "G-c.2b"** — the GPU BRICK-CORE STORE (`docs/PHASE_G_GC_PLAN.md` §2.4): a `(coord,lod) ->
/// deduped-core-index` open-addressing HASH (same FNV-1a family as [`brick_key_hash`], 5-word stride
/// `[x,y,z,lod,core_index]`) PLUS the deduped `8³` cores (512 `u32` each, [`super::brickmap::voxel_index`]
/// order). Pass D's `core_lookup` (in `voxel_residency.wgsl`) probes this to build the per-command 27-neighbour
/// table the GPU halo-fill reads — the GPU analogue of the CPU `update_gpu`'s deduped core pool, but PERSISTENT
/// (built once per scene from the in-RAM static source, NOT per re-pack). The §5 per-region paging from a
/// streamed `.vxo` is G-c.4; here it is built whole from a [`super::source::StaticVoxSource`]'s occupied keys
/// (the same in-RAM source the [`SectorOccupancy`] is built from), so the live in-RAM scenes (Sponza / the
/// `.vox` Gallery) have a complete core store for the GPU-driven pack.
#[derive(Clone, Debug, Default)]
pub struct BrickCoreStore {
    /// The `(coord,lod) -> core_index` open-addressing table (5 `u32`/slot; `lod == `[`EMPTY_LOD`] ⇒ free).
    table: Vec<u32>,
    /// `table.len()/5` as a power of two (the probe mask).
    table_size: u32,
    /// The deduped cores: core `i` is `cores[i·512 .. i·512+512]` (`8³` block ids, voxel-index order).
    cores: Vec<u32>,
}

impl BrickCoreStore {
    /// Build from an explicit `(coord, lod, core)` iterator (each `core` is the brick's `8³` block ids in
    /// [`super::brickmap::voxel_index`] order). Each DISTINCT key gets ONE core slot; the hash is sized to ~0.5
    /// load factor. A key appearing twice keeps its first core (dedup by key).
    pub fn from_cores(items: impl IntoIterator<Item = (IVec3, u32, [u32; super::brickmap::BRICK_VOXELS])>) -> Self {
        use rustc_hash::FxHashMap;
        let collected: Vec<(IVec3, u32, [u32; super::brickmap::BRICK_VOXELS])> = items.into_iter().collect();
        let mut seen: FxHashMap<(IVec3, u32), u32> = FxHashMap::default();
        let mut cores: Vec<u32> = Vec::with_capacity(collected.len() * super::brickmap::BRICK_VOXELS);
        let mut keys: Vec<(IVec3, u32, u32)> = Vec::with_capacity(collected.len());
        for (coord, lod, core) in collected {
            if seen.contains_key(&(coord, lod)) {
                continue;
            }
            let idx = (cores.len() / super::brickmap::BRICK_VOXELS) as u32;
            cores.extend_from_slice(&core);
            seen.insert((coord, lod), idx);
            keys.push((coord, lod, idx));
        }
        let table_size = ((keys.len() * 2).max(2)).next_power_of_two() as u32;
        let mut table = vec![0u32; table_size as usize * 5];
        for s in table.chunks_exact_mut(5) {
            s[3] = EMPTY_LOD;
        }
        let mask = table_size - 1;
        for (coord, lod, idx) in keys {
            let mut s = (brick_key_hash(coord, lod) & mask) as usize;
            while table[s * 5 + 3] != EMPTY_LOD {
                s = (s + 1) & (mask as usize);
            }
            table[s * 5] = coord.x as u32;
            table[s * 5 + 1] = coord.y as u32;
            table[s * 5 + 2] = coord.z as u32;
            table[s * 5 + 3] = lod;
            table[s * 5 + 4] = idx;
        }
        if cores.is_empty() {
            cores.push(0); // a non-empty buffer (no resident bricks)
        }
        Self { table, table_size, cores }
    }

    /// The hash slot count (a power of two) — the WGSL `PackConfigD.core_table_size`.
    pub fn table_size(&self) -> u32 {
        self.table_size
    }

    /// Number of distinct cores stored.
    pub fn core_count(&self) -> usize {
        self.cores.len() / super::brickmap::BRICK_VOXELS
    }

    /// Upload the two PERSISTENT storage buffers (the `core_table` + the `cores`). One-time per scene.
    pub fn upload(&self, device: &wgpu::Device) -> GpuBrickCoreBuffers {
        use wgpu::util::DeviceExt;
        let table = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel_core_table"),
            contents: bytemuck::cast_slice(&self.table),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let cores = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel_cores"),
            contents: bytemuck::cast_slice(&self.cores),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        GpuBrickCoreBuffers { table, cores, table_size: self.table_size }
    }
}

/// The uploaded GPU core-store buffers — held PERSISTENTLY in `VoxelRtResources` (G-c.2b). `table` is the
/// `(coord,lod) -> core-index` hash; `cores` the deduped `8³` cores. `table_size` is cached for the
/// `PackConfigD.core_table_size` uniform.
pub struct GpuBrickCoreBuffers {
    /// The `(coord,lod) -> core_index` hash (5 `u32`/slot).
    pub table: wgpu::Buffer,
    /// The deduped `8³` cores (512 `u32`/core).
    pub cores: wgpu::Buffer,
    /// The hash slot count (a power of two).
    pub table_size: u32,
}

/// `u32` words per `8³` brick core (= [`super::brickmap::BRICK_VOXELS`] = 512).
const CORE_WORDS: usize = super::brickmap::BRICK_VOXELS;

/// The TOMBSTONE marker stored in a deleted slot's `lod` field, distinct from both [`EMPTY_LOD`] (a free slot the
/// WGSL probe STOPS at) and any real lod (`0..=MAX_LOD`). The unchanged WGSL `core_lookup` treats a tombstone as
/// "occupied, no match" — it SKIPS it and CONTINUES the probe (it only stops at `EMPTY_LOD`) — so deleting a key
/// by tombstoning its slot NEVER truncates another key's probe chain. New inserts may reuse a tombstone slot. This
/// is the standard open-addressing deletion fix, with NO WGSL change (the probe semantics are identical).
const TOMBSTONE_LOD: u32 = EMPTY_LOD - 1;

/// **Phase G "G-c.4-paging" (§8.3)** — the DEMAND-PAGED GPU BRICK-CORE STORE: the mutable, incremental successor
/// to the immutable [`BrickCoreStore`], mirroring the `VxoSource` `RegionCache` lifecycle (upload-on-decode,
/// evict-on-drop) so the GPU store ≤ the CPU region LRU budget (constant-RAM). It OWNS the two GPU buffers the
/// unchanged WGSL `core_lookup` reads (the SAME [`GpuBrickCoreBuffers`] layout: a `(coord,lod) -> core_index`
/// open-addressing hash + the `8³` cores), plus a CPU mirror of the hash + a free-slot stack so an
/// [`Self::upload_region`] / [`Self::evict_region`] writes ONLY the touched slots via `queue_write_buffer`
/// (no whole rebuild — a whole Bistro rebuild is ~300 MB/crossing, FORBIDDEN).
///
/// ## Free-list deletion correctness (the open-addressing trap)
/// Deletion uses a [`TOMBSTONE_LOD`] sentinel (not [`EMPTY_LOD`]) so a removed key never breaks another key's
/// linear-probe chain (see [`TOMBSTONE_LOD`]). A core SLOT freed on eviction goes back on [`Self::free_cores`]
/// and may be reused by a later insert; its bytes are NOT cleared (overwritten on reuse). The COVERAGE INVARIANT
/// (§8.3): the prefetcher pages exactly the clipmap-covering present regions PADDED +1 brick, and the GPU
/// enumerate only ENTERS bricks with `level_resident` (inside the clipmap) — so every enterable brick + its
/// 26-halo has its core resident here. Bounded: `core_cap` caps the cores buffer to the resident-region footprint.
pub struct PagedBrickCoreStore {
    /// The GPU `(coord,lod) -> core_index` hash (5 `u32`/slot). `lod == `[`EMPTY_LOD`] free, `== `[`TOMBSTONE_LOD`]
    /// deleted-but-probe-through. A power-of-two `table_size`. Written incrementally per touched slot.
    table_buf: wgpu::Buffer,
    /// The GPU cores buffer (`CORE_WORDS` `u32`/core), capacity `core_cap` cores. Written per inserted slot.
    cores_buf: wgpu::Buffer,
    /// `table_size` (power of two) — the WGSL `core_table_size`.
    table_size: u32,
    /// Max distinct cores (= the resident-region brick footprint). The free-list never exceeds this.
    core_cap: u32,
    /// CPU mirror of the GPU hash table (`table_size * 5` u32), so an insert/evict computes the touched slot index
    /// + probe chain on the CPU, then `queue_write_buffer`s just that slot. Kept bit-identical to the GPU buffer.
    table: Vec<u32>,
    /// `(coord, lod) -> (table_slot, core_index)` — the live keys' table slot + core slot (the CPU index the
    /// incremental insert/evict drive). One entry per resident brick.
    keys: rustc_hash::FxHashMap<(IVec3, u32), (u32, u32)>,
    /// The free CORE slots (a LIFO stack of `core_index`es), refilled on eviction. Empty ⇒ at capacity.
    free_cores: Vec<u32>,
    /// `region key -> the (coord,lod) brick keys it paged in` — so [`Self::evict_region`] removes exactly this
    /// region's keys. A brick present in TWO paged regions (shouldn't happen — regions partition the brick grid,
    /// but a +1 halo pad can re-page a neighbour region holding the SAME brick) is REFERENCE-COUNTED via
    /// [`Self::refcount`] so the last evictor frees it.
    region_keys: rustc_hash::FxHashMap<(usize, u32, IVec3), Vec<(IVec3, u32)>>,
    /// Per-key reference count (how many resident regions paged it) — a brick freed only when its count hits 0.
    refcount: rustc_hash::FxHashMap<(IVec3, u32), u32>,
}

impl PagedBrickCoreStore {
    /// Build an EMPTY paged store + its GPU buffers, sized for `core_cap` resident cores (the resident-region
    /// brick footprint) and a `table_size = next_pow2(2 * core_cap)` hash (≤ 0.5 load factor). All table slots
    /// start [`EMPTY_LOD`]; every core slot is free. `core_cap` is clamped to ≥ 1 (a non-empty buffer).
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, core_cap: u32) -> Self {
        let core_cap = core_cap.max(1);
        let table_size = ((core_cap as usize * 2).max(2)).next_power_of_two() as u32;
        let mut table = vec![0u32; table_size as usize * 5];
        for s in table.chunks_exact_mut(5) {
            s[3] = EMPTY_LOD;
        }
        let table_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_paged_core_table"),
            size: (table_size as u64) * 5 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let cores_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_paged_cores"),
            size: (core_cap as u64) * CORE_WORDS as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&table_buf, 0, bytemuck::cast_slice(&table));
        // free_cores: all slots free, LIFO so the first claim is slot 0 (deterministic for the parity test).
        let free_cores: Vec<u32> = (0..core_cap).rev().collect();
        Self {
            table_buf,
            cores_buf,
            table_size,
            core_cap,
            table,
            keys: rustc_hash::FxHashMap::default(),
            free_cores,
            region_keys: rustc_hash::FxHashMap::default(),
            refcount: rustc_hash::FxHashMap::default(),
        }
    }

    /// The hash slot count (power of two) — the WGSL `core_table_size`.
    #[inline]
    pub fn table_size(&self) -> u32 {
        self.table_size
    }

    /// Number of distinct cores currently resident (live keys).
    #[inline]
    pub fn resident_cores(&self) -> usize {
        self.keys.len()
    }

    /// The core-slot CAPACITY (the bounded-buffer ceiling) — the constant-RAM bound the gates assert against.
    #[inline]
    pub fn core_cap(&self) -> u32 {
        self.core_cap
    }

    /// A [`GpuBrickCoreBuffers`]-shaped VIEW for binding (the front end's `rebind_pool` consumes this). The buffers
    /// are CLONED handles (cheap `Arc`) so the store keeps writing the SAME buffers the bind group references.
    pub fn buffers(&self) -> GpuBrickCoreBuffers {
        GpuBrickCoreBuffers {
            table: self.table_buf.clone(),
            cores: self.cores_buf.clone(),
            table_size: self.table_size,
        }
    }

    /// Probe the CPU table for `key`'s LIVE slot (matching `(x,y,z,lod)`), or `None` if absent. Stops at the first
    /// [`EMPTY_LOD`] (matching the WGSL); SKIPS tombstones (continues). Bounded by `table_size`.
    fn find_slot(&self, coord: IVec3, lod: u32) -> Option<u32> {
        let mask = self.table_size - 1;
        let mut slot = brick_key_hash(coord, lod) & mask;
        for _ in 0..self.table_size {
            let base = slot as usize * 5;
            let e_lod = self.table[base + 3];
            if e_lod == EMPTY_LOD {
                return None;
            }
            if e_lod == lod
                && self.table[base] == coord.x as u32
                && self.table[base + 1] == coord.y as u32
                && self.table[base + 2] == coord.z as u32
            {
                return Some(slot);
            }
            slot = (slot + 1) & mask;
        }
        None
    }

    /// Find the slot to INSERT `key` into: the first EMPTY or TOMBSTONE slot on its probe chain (reusing a
    /// tombstone). Panics if the table is full (the caller sizes `table_size` ≥ 2× `core_cap`, so with ≤ `core_cap`
    /// live keys it never fills — an invariant, not a runtime case).
    fn insert_slot(&self, coord: IVec3, lod: u32) -> u32 {
        let mask = self.table_size - 1;
        let mut slot = brick_key_hash(coord, lod) & mask;
        for _ in 0..self.table_size {
            let e_lod = self.table[slot as usize * 5 + 3];
            if e_lod == EMPTY_LOD || e_lod == TOMBSTONE_LOD {
                return slot;
            }
            slot = (slot + 1) & mask;
        }
        panic!("paged core table full (table_size {} <= live keys) — core_cap mis-sized", self.table_size)
    }

    /// Write one table slot's 5 words to the GPU (the incremental dirty write).
    fn flush_table_slot(&self, queue: &wgpu::Queue, slot: u32) {
        let base = slot as usize * 5;
        queue.write_buffer(&self.table_buf, base as u64 * 4, bytemuck::cast_slice(&self.table[base..base + 5]));
    }

    /// **Insert ONE brick core** `(coord, lod)` (idempotent: a re-insert of a live key just bumps its refcount).
    /// Claims a free core slot, writes the core to `cores[slot]`, inserts the hash entry, and `queue_write_buffer`s
    /// the touched table slot + the core. Returns the `core_index`, or `None` if the store is AT CAPACITY (the
    /// free-list is empty) — a graceful far-detail drop (the caller's cap is the bounded-buffer ceiling, so an
    /// over-full crossing simply leaves the excess bricks core-absent rather than crashing; the GPU `core_lookup`
    /// then returns ABSENT for them, identical to an un-paged halo neighbour). Caller batches via the public paths.
    fn insert_brick(&mut self, queue: &wgpu::Queue, coord: IVec3, lod: u32, core: &[u32; CORE_WORDS]) -> Option<u32> {
        let key = (coord, lod);
        if let Some(&(_, idx)) = self.keys.get(&key) {
            *self.refcount.entry(key).or_insert(0) += 1;
            return Some(idx); // already resident — share it (a +1-halo region re-page)
        }
        let core_index = self.free_cores.pop()?; // None ⇒ at capacity (graceful drop)
        // Write the core.
        queue.write_buffer(
            &self.cores_buf,
            core_index as u64 * CORE_WORDS as u64 * 4,
            bytemuck::cast_slice(core),
        );
        // Insert the hash entry.
        let slot = self.insert_slot(coord, lod);
        let base = slot as usize * 5;
        self.table[base] = coord.x as u32;
        self.table[base + 1] = coord.y as u32;
        self.table[base + 2] = coord.z as u32;
        self.table[base + 3] = lod;
        self.table[base + 4] = core_index;
        self.flush_table_slot(queue, slot);
        self.keys.insert(key, (slot, core_index));
        self.refcount.insert(key, 1);
        Some(core_index)
    }

    /// **Evict ONE brick** `(coord, lod)`: decrement its refcount; on reaching 0, tombstone its table slot (so a
    /// probe chain stays intact), free its core slot, and remove the CPU index. A double-evict of an absent key is
    /// a no-op (defensive). Writes the tombstoned table slot to the GPU.
    fn evict_brick(&mut self, queue: &wgpu::Queue, coord: IVec3, lod: u32) {
        let key = (coord, lod);
        let Some(rc) = self.refcount.get_mut(&key) else {
            return; // not resident
        };
        *rc -= 1;
        if *rc > 0 {
            return; // still referenced by another paged region
        }
        self.refcount.remove(&key);
        let Some((slot, core_index)) = self.keys.remove(&key) else {
            return;
        };
        let base = slot as usize * 5;
        self.table[base + 3] = TOMBSTONE_LOD; // probe-through marker (NOT EMPTY — preserves other chains)
        self.flush_table_slot(queue, slot);
        self.free_cores.push(core_index); // reuse the core slot later (bytes left stale, overwritten on reuse)
    }

    /// **Page IN a region's bricks** (§8.3): insert each `(world_coord, lod, core)` brick, recording the region's
    /// key list so [`Self::evict_region`] can drop exactly them. `region` is keyed `(asset, lod, region_coord)`.
    /// Idempotent for a region already paged (re-records its keys, bumping their refcounts — the caller pages a
    /// region once and evicts once, so this is the defensive path). Bricks are supplied by the caller's
    /// `VxoSource::for_each_region_brick` (world coords + cores).
    pub fn upload_region(
        &mut self,
        queue: &wgpu::Queue,
        region: (usize, u32, IVec3),
        bricks: &[(IVec3, u32, [u32; CORE_WORDS])],
    ) {
        let mut keys = Vec::with_capacity(bricks.len());
        for (coord, lod, core) in bricks {
            if self.insert_brick(queue, *coord, *lod, core).is_some() {
                keys.push((*coord, *lod));
            }
        }
        self.region_keys.insert(region, keys);
    }

    /// **Page OUT a region** (§8.3): evict every brick this region paged in (refcount-decremented; freed at 0).
    /// A region never paged is a no-op. Mirrors `RegionCache` eviction so the GPU store ≤ the CPU LRU budget.
    pub fn evict_region(&mut self, queue: &wgpu::Queue, region: (usize, u32, IVec3)) {
        let Some(keys) = self.region_keys.remove(&region) else {
            return;
        };
        for (coord, lod) in keys {
            self.evict_brick(queue, coord, lod);
        }
    }

    /// The set of region keys currently paged in (for the constant-RAM / coverage assertions in the gates).
    pub fn resident_region_count(&self) -> usize {
        self.region_keys.len()
    }

    /// **Phase G "G-c.4-paging" (§8.3)** — sync the resident core SET to exactly `desired` (a per-brick set diff,
    /// for the SURFACE-shell core paging where the desired cores span bricks across regions, NOT whole regions).
    /// Inserts the cores newly in `desired`, evicts the ones no longer in it. `desired` maps `(coord,lod)` → its
    /// `8³` core. This is the brick-granular alternative to [`Self::upload_region`]/[`Self::evict_region`] (use one
    /// or the other per store — the pager uses this). Bounded by `desired.len()` ≤ `core_cap` (asserted via the
    /// free-list). Each brick is reference-count-1 in this mode (a `(coord,lod)` is unique in the desired set).
    pub fn sync_to(
        &mut self,
        queue: &wgpu::Queue,
        desired: &rustc_hash::FxHashMap<(IVec3, u32), [u32; CORE_WORDS]>,
    ) {
        self.sync_to_keys(queue, &desired.keys().copied().collect(), |c, l| desired.get(&(c, l)).copied());
    }

    /// **Phase G "G-c.4-paging" (§8.3)** — INCREMENTAL set-diff sync to a desired KEY set, decoding a new key's
    /// core LAZILY via `fetch` ONLY when it is NOT already resident. This is the perf-critical path: a crossing
    /// re-derives the desired surface+halo KEY set (cheap — hash probes, no voxel decode) and decodes cores ONLY
    /// for the keys that newly entered (avoiding the per-crossing whole re-decode of every surface core, which
    /// re-introduced the freeze). Evicts resident keys no longer desired. `fetch(coord,lod) -> Option<core>` ⇒
    /// `None` skips (absent brick); a full store ⇒ the insert gracefully drops (the bounded-buffer ceiling).
    pub fn sync_to_keys(
        &mut self,
        queue: &wgpu::Queue,
        desired: &rustc_hash::FxHashSet<(IVec3, u32)>,
        mut fetch: impl FnMut(IVec3, u32) -> Option<[u32; CORE_WORDS]>,
    ) {
        // Evict the resident keys no longer desired.
        let to_evict: Vec<(IVec3, u32)> = self.keys.keys().filter(|k| !desired.contains(k)).copied().collect();
        for (coord, lod) in to_evict {
            self.refcount.insert((coord, lod), 1); // set-diff mode: each key held once
            self.evict_brick(queue, coord, lod);
        }
        // Insert the desired keys not yet resident — decode the core ONLY now (lazy, incremental).
        for &(coord, lod) in desired {
            if !self.keys.contains_key(&(coord, lod))
                && let Some(core) = fetch(coord, lod)
            {
                let _ = self.insert_brick(queue, coord, lod, &core);
            }
        }
    }

    /// **Phase G "G-c.4-paging" (§8.3)** — apply an explicit per-crossing core DELTA (Θ(delta), NOT Θ(all)): EVICT
    /// each key in `evict` (tombstone its slot, free its core), then INSERT each key in `insert` (decode its core
    /// LAZILY via `fetch`; a None ⇒ skip; a full store ⇒ graceful drop at the bounded-buffer ceiling). This is the
    /// perf-critical path the prefetcher uses — it touches only the bricks that crossed the clipmap edge this
    /// frame, never re-scanning the whole resident shell. Each key is reference-count-1 in this mode.
    pub fn apply_delta(
        &mut self,
        queue: &wgpu::Queue,
        insert: &[(IVec3, u32)],
        evict: &[(IVec3, u32)],
        mut fetch: impl FnMut(IVec3, u32) -> Option<[u32; CORE_WORDS]>,
    ) {
        for &(coord, lod) in evict {
            if self.keys.contains_key(&(coord, lod)) {
                self.refcount.insert((coord, lod), 1); // set-diff mode: each key held once
                self.evict_brick(queue, coord, lod);
            }
        }
        for &(coord, lod) in insert {
            if !self.keys.contains_key(&(coord, lod))
                && let Some(core) = fetch(coord, lod)
            {
                let _ = self.insert_brick(queue, coord, lod, &core);
            }
        }
    }

    /// Whether `(coord, lod)` has a resident core (the coverage-invariant probe for the unit test).
    pub fn contains(&self, coord: IVec3, lod: u32) -> bool {
        self.find_slot(coord, lod).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashSet;

    /// A known occupied set → build → assert the CPU `is_occupied` agrees with the set membership over a sample
    /// of occupied, empty, and boundary keys (the CPU side of the GPU-vs-CPU gate; the GPU side is the
    /// integration test). Also checks the coarse `sector_any_occupied` and the popcount statistics.
    #[test]
    fn cpu_is_occupied_matches_the_known_set() {
        // A scattered set across several LODs incl. NEGATIVE coords (the clipmap reaches both signs) and keys
        // that straddle sector boundaries (coord 3→4 crosses a SECTOR_EDGE=4 boundary).
        let occupied: Vec<(IVec3, u32)> = vec![
            (IVec3::new(0, 0, 0), 0),
            (IVec3::new(3, 3, 3), 0),   // last brick of sector (0,0,0)
            (IVec3::new(4, 0, 0), 0),   // first brick of sector (1,0,0)
            (IVec3::new(-1, -1, -1), 0), // sector (-1,-1,-1), local (3,3,3)
            (IVec3::new(-4, 0, 0), 2),
            (IVec3::new(100, -50, 7), 5),
            (IVec3::new(7, 7, 7), 7),   // MAX_LOD
        ];
        let occ = SectorOccupancy::from_occupied(occupied.iter().copied());
        let set: FxHashSet<(IVec3, u32)> = occupied.iter().copied().collect();

        // Every occupied key reads occupied; the SAME coord at a DIFFERENT lod does not (per-LOD namespace).
        for &(c, l) in &occupied {
            assert!(occ.is_occupied(c, l), "occupied {c:?}@{l} read empty");
            let other = if l == 0 { 1 } else { 0 };
            assert_eq!(
                occ.is_occupied(c, other),
                set.contains(&(c, other)),
                "{c:?}@{other} (other lod) disagreed with the set"
            );
        }

        // A sweep of nearby + far keys: GPU/CPU verdict must equal set membership exactly.
        for lod in 0..=MAX_LOD {
            for z in -6..=6 {
                for y in -6..=6 {
                    for x in -6..=6 {
                        let c = IVec3::new(x, y, z);
                        assert_eq!(
                            occ.is_occupied(c, lod),
                            set.contains(&(c, lod)),
                            "{c:?}@{lod} disagreed with the known set"
                        );
                    }
                }
            }
        }

        // Coarse test: a sector is "any-occupied" iff it holds ≥ 1 occupied brick of the set.
        let mut occupied_sectors: FxHashSet<(IVec3, u32)> = FxHashSet::default();
        for &(c, l) in &occupied {
            occupied_sectors.insert((split_sector(c).0, l));
        }
        for lod in 0..=MAX_LOD {
            for z in -3..=30 {
                for y in -15..=3 {
                    for x in -3..=30 {
                        let s = IVec3::new(x, y, z);
                        assert_eq!(
                            occ.sector_any_occupied(s, lod),
                            occupied_sectors.contains(&(s, lod)),
                            "sector {s:?}@{lod} coarse test disagreed"
                        );
                    }
                }
            }
        }

        assert_eq!(occ.occupied_bricks(), occupied.len() as u64);
        assert_eq!(occ.occupied_sectors(), occupied_sectors.len());
        assert!(occ.table_size().is_power_of_two());
    }

    /// An EMPTY occupancy is valid: one empty slot, every probe misses.
    #[test]
    fn empty_occupancy_is_all_unoccupied() {
        let occ = SectorOccupancy::from_occupied(std::iter::empty());
        assert_eq!(occ.table_size(), 1);
        assert_eq!(occ.occupied_sectors(), 0);
        assert!(!occ.is_occupied(IVec3::ZERO, 0));
        assert!(!occ.sector_any_occupied(IVec3::ZERO, 0));
    }

    /// **G-c.1 — the CPU face-cull SSOT (`classify_surface`) reproduces `StaticVoxSource::classify == Surface`
    /// EXACTLY**, including partial (non-full) bricks and the buried-Interior cull. Build a small map with a
    /// fully-solid 3×3×3 block (its centre brick is Interior — full + 6 full neighbours), a PARTIAL surface
    /// brick (one air voxel ⇒ always Surface even when surrounded), and isolated bricks; then assert
    /// `occ.classify_surface == (source.classify == Surface)` over a dense sample at every LOD.
    #[test]
    fn classify_surface_matches_static_source_classify() {
        use crate::voxel::brickmap::{BRICK_EDGE, Brick, BrickMap};
        use crate::voxel::palette::BlockId;
        use crate::voxel::source::{BrickClass, BrickSource, StaticVoxSource};

        let full = |id: u16| {
            let mut v = Box::new([BlockId::AIR; BRICK_VOXELS_LOCAL]);
            for c in v.iter_mut() {
                *c = BlockId(id);
            }
            Brick::from_voxels(v)
        };
        // A brick with exactly one interior air voxel — NOT full, so classify is always Surface.
        let partial = |id: u16| {
            let mut v = Box::new([BlockId(id); BRICK_VOXELS_LOCAL]);
            v[0] = BlockId::AIR;
            Brick::from_voxels(v)
        };
        let mut map = BrickMap::new();
        // Fully-solid 3×3×3 ⇒ centre (1,1,1) is Interior (full + 6 full face-neighbours); the 26 shell bricks
        // are Surface (each has ≥1 non-full / absent face-neighbour).
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    map.insert(IVec3::new(x, y, z), full(1));
                }
            }
        }
        // A PARTIAL brick fully surrounded (a +6 of full neighbours), elsewhere: occupancy-occluded, but
        // classify ⇒ Surface because the brick itself is NOT full — the partial-overrides-occlusion path.
        let p = IVec3::new(9, 9, 9); // within the dense sample box below; isolated from the 3×3×3 block
        map.insert(p, partial(2));
        for off in [
            IVec3::new(1, 0, 0),
            IVec3::new(-1, 0, 0),
            IVec3::new(0, 1, 0),
            IVec3::new(0, -1, 0),
            IVec3::new(0, 0, 1),
            IVec3::new(0, 0, -1),
        ] {
            map.insert(p + off, full(5));
        }
        map.insert(IVec3::new(5, 6, 7), full(3)); // isolated ⇒ Surface
        map.insert(IVec3::new(-3, 1, 4), full(4)); // negative-coord sector ⇒ Surface
        let _ = BRICK_EDGE;

        let source = StaticVoxSource::new(&map);
        let occ = SectorOccupancy::from_occupied_full(source.occupied_keys_full());

        let mut surface_seen = 0usize;
        let mut interior_seen = 0usize;
        for lod in 0..=MAX_LOD {
            for z in -5..=11 {
                for y in -5..=11 {
                    for x in -5..=11 {
                        let c = IVec3::new(x, y, z);
                        let class = source.classify(c, lod);
                        let want = class == BrickClass::Surface;
                        if want {
                            surface_seen += 1;
                        }
                        if class == BrickClass::Interior {
                            interior_seen += 1;
                        }
                        assert_eq!(
                            occ.classify_surface(c, lod),
                            want,
                            "classify_surface({c:?}@{lod}) disagreed with StaticVoxSource::classify ({class:?})"
                        );
                    }
                }
            }
        }
        assert!(surface_seen > 0, "the sample must contain Surface bricks");
        assert!(interior_seen > 0, "the fully-solid block (with the centre swapped) must still yield Interior");
    }

    /// `BRICK_VOXELS` re-exported locally for the test brick builders.
    const BRICK_VOXELS_LOCAL: usize = crate::voxel::brickmap::BRICK_VOXELS;
}
