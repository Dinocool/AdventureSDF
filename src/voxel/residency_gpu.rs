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

/// One sector record in the GPU open-addressing hash table. `lod == `[`EMPTY_LOD`] marks a FREE slot. 24 bytes,
/// `bytemuck`-uploadable. The WGSL mirror reads the same 6 `u32` stride.
///
/// `mask_lo`/`mask_hi` are the low/high 32 bits of the 64-bit alloc mask — bit `b` (`= `[`sector_bit_index`])
/// is set ⇔ the `(sector·SECTOR_EDGE + local, lod)` brick is occupied. Split as `2×u32` because WGSL storage
/// buffers have no `u64`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuSectorEntry {
    /// The sector coord (`coord.div_euclid(SECTOR_EDGE)`), two's-complement bits (`bitcast` on the GPU).
    pub sector_x: i32,
    pub sector_y: i32,
    pub sector_z: i32,
    /// The LOD level this sector lives at (`0..=MAX_LOD`), or [`EMPTY_LOD`] for a free slot.
    pub lod: u32,
    /// Low 32 bits of the 64-bit occupancy mask.
    pub mask_lo: u32,
    /// High 32 bits of the 64-bit occupancy mask.
    pub mask_hi: u32,
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
    /// Build the sparse sector tables from an explicit set of occupied `(coord, lod)` bricks. The lower-level
    /// SSOT both [`from_oracle`](Self::from_oracle) and the parity test feed — so the test can build from a
    /// known map without a [`BrickSource`]. Sectors are accumulated into a temporary `(sector, lod) -> mask`
    /// map, then laid into a power-of-two open-addressing table sized for [`LOAD_FACTOR`].
    pub fn from_occupied(occupied: impl IntoIterator<Item = (IVec3, u32)>) -> Self {
        use rustc_hash::FxHashMap;
        let mut masks: FxHashMap<(IVec3, u32), u64> = FxHashMap::default();
        for (coord, lod) in occupied {
            debug_assert!(lod <= MAX_LOD, "occupancy lod {lod} exceeds MAX_LOD {MAX_LOD}");
            let (sector, local) = split_sector(coord);
            let bit = sector_bit_index(local);
            *masks.entry((sector, lod)).or_insert(0) |= 1u64 << bit;
        }
        Self::from_sector_masks(masks)
    }

    /// Lay a `(sector, lod) -> mask` map into the power-of-two open-addressing table (linear probing on
    /// [`sector_hash`]). An EMPTY table (no occupied sectors) is still given ONE empty slot so the GPU buffer is
    /// non-zero-length + every probe immediately misses (every `is_occupied` ⇒ false).
    fn from_sector_masks(masks: rustc_hash::FxHashMap<(IVec3, u32), u64>) -> Self {
        let n = masks.len();
        let occupied_bricks: u64 = masks.values().map(|m| m.count_ones() as u64).sum();
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
            };
            table_size
        ];
        let mask_bits = (table_size - 1) as u32;
        for ((sector, lod), mask) in masks {
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
                mask_lo: mask as u32,
                mask_hi: (mask >> 32) as u32,
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
        let mut masks: FxHashMap<(IVec3, u32), u64> = FxHashMap::default();
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
                    *masks.entry((sector, lod)).or_insert(0) |= 1u64 << bit;
                }
            }
        }
        Self::from_sector_masks(masks)
    }

    /// The CPU mirror of the WGSL `is_occupied` — the SSOT the parity gate asserts GPU == CPU against. Hash the
    /// brick's sector, linear-probe the table for a matching `(sector, lod)` slot, and test the brick's bit.
    /// A free slot before a match ⇒ the sector is absent ⇒ NOT occupied (the probe terminates at the first free
    /// slot, exactly as the build placed entries). Bounded by the table size (no infinite loop on a full miss
    /// because the table is never 100% full).
    pub fn is_occupied(&self, coord: IVec3, lod: u32) -> bool {
        if self.table_size == 0 {
            return false;
        }
        let (sector, local) = split_sector(coord);
        let bit = sector_bit_index(local);
        let mask_bits = self.table_size - 1;
        let mut slot = (sector_hash(sector, lod) & mask_bits) as usize;
        // Probe at most `table_size` slots — a full pass means the sector is absent (defensive; the build keeps
        // the table < 100% full, so a free slot is always hit first for an absent key).
        for _ in 0..self.table_size {
            let e = &self.entries[slot];
            if e.lod == EMPTY_LOD {
                return false; // first free slot ⇒ key absent
            }
            if e.lod == lod && e.sector_x == sector.x && e.sector_y == sector.y && e.sector_z == sector.z {
                let mask = (e.mask_lo as u64) | ((e.mask_hi as u64) << 32);
                return (mask >> bit) & 1 != 0;
            }
            slot = (slot + 1) & (mask_bits as usize);
        }
        false
    }

    /// The coarse "is ANY brick in this sector occupied?" — the §1 Pass B0 occupancy test, from the SAME fetch
    /// as `is_occupied` (`mask != 0`). `sector` is the sector coord (`coord.div_euclid(SECTOR_EDGE)`).
    pub fn sector_any_occupied(&self, sector: IVec3, lod: u32) -> bool {
        if self.table_size == 0 {
            return false;
        }
        let mask_bits = self.table_size - 1;
        let mut slot = (sector_hash(sector, lod) & mask_bits) as usize;
        for _ in 0..self.table_size {
            let e = &self.entries[slot];
            if e.lod == EMPTY_LOD {
                return false;
            }
            if e.lod == lod && e.sector_x == sector.x && e.sector_y == sector.y && e.sector_z == sector.z {
                return e.mask_lo != 0 || e.mask_hi != 0;
            }
            slot = (slot + 1) & (mask_bits as usize);
        }
        false
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
}
