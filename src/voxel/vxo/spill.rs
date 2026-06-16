//! Constant-RAM `.vxo` bake producer (Constant-RAM Bake plan, **Stages 1 + 2**) — the disk-spill base pass
//! and the windowed coarse downsample, feeding the SAME [`VxoStreamWriter`] sinks (`add_region` /
//! `add_lod_region`) as the in-RAM [`encode_vxo`](super::writer::encode_vxo) path, so the produced `.vxo` is
//! BYTE-IDENTICAL (the parity gate). Peak RAM is constant in the scene SURFACE size:
//!
//!   * **Base spill** ([`RegionSpillPool`]): every solid voxel is appended to a per-region scratch file via an
//!     LRU pool of ≤[`MAX_OPEN_SPILLS`] open `BufWriter`s. RAM = the pool's write buffers (bounded) + an
//!     `FxHashSet<IVec3>` of the region coords seen (O(region count) — sub-linear in surface voxels, ~12 B
//!     each). The whole LOD0 surface `BrickMap` is NEVER resident.
//!   * **Base assemble** ([`assemble_base`]): one region at a time — read its spill, group voxels into bricks
//!     (≤K³ bricks × 512 `BlockId` = bounded), `add_region`, drop, delete the spill. Resident = one region.
//!   * **Windowed coarse** ([`windowed_coarse`]): each level `L ∈ 1..=MAX_LOD` is built from the finer level's
//!     spills. Per COARSE region, load the ≤8 finer regions covering the children footprint
//!     `[2·crc·K, 2·(crc+1)·K)` into a transient `BrickMap`, run the EXACT `source::gather_children` →
//!     `downsample_children` SSOT per coarse brick (bit-identical to `build_coarse_pyramid`), emit via
//!     `add_lod_region`, and re-spill the coarse bricks for level `L+1`. Resident = the ≤8-finer-region window.
//!
//! The cross-region gather (a coarse brick's `2·cc+1` children landing in the adjacent finer region) is the
//! one bit-identity hazard; loading the FULL child footprint (every overlapping finer region, not just the
//! aligned one) is what makes it correct-by-construction, pinned by the byte-identity parity gate + a
//! boundary-straddle case in `tests.rs`.

use std::io::{BufWriter, Read as _, Write as _};
use std::path::{Path, PathBuf};

use bevy::math::IVec3;
use rustc_hash::{FxHashMap, FxHashSet};

use super::writer::{VxoStreamWriter, region_of_brick};
use crate::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, BrickMap, MAX_LOD, voxel_index};
use crate::voxel::palette::BlockId;
use crate::voxel::source::{downsample_children, gather_children};

/// Max simultaneously-open per-region spill `BufWriter`s (the LRU pool bound). 64 keeps the open-handle count
/// and the aggregate write-buffer RAM bounded regardless of how many regions the surface spans. When a 65th
/// region is touched the least-recently-used writer is flushed + closed (and re-opened in APPEND mode if it is
/// written again — the spill is purely additive, so re-open/append is correctness-preserving).
const MAX_OPEN_SPILLS: usize = 64;

/// One spilled solid voxel: its owning brick coord, the brick-local voxel index (`0..BRICK_VOXELS`), and its
/// block id. 16 bytes on disk (`3×i32` + `u16` + `u16`), little-endian. Deterministic routing (one record per
/// solid, by construction) means each `(bc, local)` is written exactly once — so grouping is conflict-free.
const SPILL_RECORD_LEN: usize = 16;

/// Append-only per-region spill writer pool with an LRU eviction bound ([`MAX_OPEN_SPILLS`]). Records are routed
/// to the file for `region_of_brick(bc, k)`. Tracks every region coord ever touched (the single sub-linear
/// residual). Files live under `scratch` named by a `prefix` + region coord, so the base spill and each coarse
/// re-spill use disjoint name spaces.
pub struct RegionSpillPool {
    scratch: PathBuf,
    prefix: String,
    k: i32,
    /// region coord → its open `BufWriter` (only the ≤MAX_OPEN_SPILLS most-recently-used are open).
    open: FxHashMap<IVec3, BufWriter<std::fs::File>>,
    /// LRU order, front = least-recently-used. Small (≤MAX_OPEN_SPILLS).
    lru: std::collections::VecDeque<IVec3>,
    /// Every region coord ever written (the spill files that exist on disk). O(region count).
    regions: FxHashSet<IVec3>,
}

impl RegionSpillPool {
    /// Open a pool routing on region granularity `k` (bricks/region edge), files under `scratch` named
    /// `<prefix>_region_<x>_<y>_<z>.spill`. Existing files for these regions are TRUNCATED on first touch.
    pub fn new(scratch: impl AsRef<Path>, prefix: &str, k: i32) -> Self {
        Self {
            scratch: scratch.as_ref().to_path_buf(),
            prefix: prefix.to_string(),
            k,
            open: FxHashMap::default(),
            lru: std::collections::VecDeque::new(),
            regions: FxHashSet::default(),
        }
    }

    fn spill_path(&self, rc: IVec3) -> PathBuf {
        self.scratch.join(format!("{}_region_{}_{}_{}.spill", self.prefix, rc.x, rc.y, rc.z))
    }

    /// Spill one solid voxel `(brick_coord, local_index, block)` to its owning region's file. Opens/creates the
    /// file on first touch (truncating), re-opens in APPEND mode after an LRU eviction (the spill is additive).
    pub fn push(&mut self, bc: IVec3, local: u16, block: BlockId) -> std::io::Result<()> {
        debug_assert!((local as usize) < BRICK_VOXELS);
        let rc = region_of_brick(bc, self.k);
        self.ensure_open(rc)?;
        // Touch LRU (move to back).
        if let Some(pos) = self.lru.iter().position(|&c| c == rc) {
            self.lru.remove(pos);
        }
        self.lru.push_back(rc);
        let w = self.open.get_mut(&rc).expect("just ensured open");
        let mut rec = [0u8; SPILL_RECORD_LEN];
        rec[0..4].copy_from_slice(&bc.x.to_le_bytes());
        rec[4..8].copy_from_slice(&bc.y.to_le_bytes());
        rec[8..12].copy_from_slice(&bc.z.to_le_bytes());
        rec[12..14].copy_from_slice(&local.to_le_bytes());
        rec[14..16].copy_from_slice(&block.0.to_le_bytes());
        w.write_all(&rec)
    }

    /// Ensure region `rc` has an open writer, evicting the LRU one if the pool is full. A FIRST touch truncates
    /// (`create`); a re-touch after eviction APPENDS (so previously-spilled records are preserved).
    fn ensure_open(&mut self, rc: IVec3) -> std::io::Result<()> {
        if self.open.contains_key(&rc) {
            return Ok(());
        }
        if self.open.len() >= MAX_OPEN_SPILLS
            && let Some(victim) = self.lru.pop_front()
            && let Some(mut w) = self.open.remove(&victim)
        {
            w.flush()?;
        }
        let path = self.spill_path(rc);
        let first_touch = self.regions.insert(rc);
        let file = if first_touch {
            std::fs::File::create(&path)?
        } else {
            std::fs::OpenOptions::new().append(true).open(&path)?
        };
        self.open.insert(rc, BufWriter::new(file));
        Ok(())
    }

    /// Flush + close every open writer (call before reading the spills back). Leaves the files on disk.
    pub fn flush_all(&mut self) -> std::io::Result<()> {
        for (_rc, mut w) in self.open.drain() {
            w.flush()?;
        }
        self.lru.clear();
        Ok(())
    }

    /// Every region coord that has a spill file, sorted `(z,y,x)` — the deterministic feed order for the
    /// streaming writer (matches `encode_vxo`'s BRIK layout).
    pub fn sorted_regions(&self) -> Vec<IVec3> {
        let mut v: Vec<IVec3> = self.regions.iter().copied().collect();
        v.sort_by_key(|c| (c.z, c.y, c.x));
        v
    }

    /// Read one region's spilled voxels back, grouped into dense per-brick arrays keyed by brick coord. The
    /// resident cost is bounded by ONE region (≤K³ bricks × `BRICK_VOXELS` `BlockId`).
    fn read_region_bricks(&self, rc: IVec3) -> std::io::Result<FxHashMap<IVec3, Box<[BlockId; BRICK_VOXELS]>>> {
        let path = self.spill_path(rc);
        let mut bytes = Vec::new();
        std::fs::File::open(&path)?.read_to_end(&mut bytes)?;
        let mut bricks: FxHashMap<IVec3, Box<[BlockId; BRICK_VOXELS]>> = FxHashMap::default();
        let mut i = 0;
        while i + SPILL_RECORD_LEN <= bytes.len() {
            let bx = i32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
            let by = i32::from_le_bytes(bytes[i + 4..i + 8].try_into().unwrap());
            let bz = i32::from_le_bytes(bytes[i + 8..i + 12].try_into().unwrap());
            let local = u16::from_le_bytes(bytes[i + 12..i + 14].try_into().unwrap());
            let block = u16::from_le_bytes(bytes[i + 14..i + 16].try_into().unwrap());
            let arr = bricks
                .entry(IVec3::new(bx, by, bz))
                .or_insert_with(|| Box::new([BlockId::AIR; BRICK_VOXELS]));
            arr[local as usize] = BlockId(block);
            i += SPILL_RECORD_LEN;
        }
        Ok(bricks)
    }

    /// Delete one region's spill file (after it has been consumed). Best-effort.
    fn delete_region(&self, rc: IVec3) {
        let _ = std::fs::remove_file(self.spill_path(rc));
    }

    /// Delete ALL spill files (cleanup on the final level / on error).
    pub fn delete_all(&self) {
        for &rc in &self.regions {
            self.delete_region(rc);
        }
    }
}

/// Spill one solid voxel at WORLD voxel coord `w` with block `block` into `pool` (computes the owning brick +
/// local index). The single bridge a caller uses per solid voxel.
#[inline]
pub fn spill_voxel(pool: &mut RegionSpillPool, w: IVec3, block: BlockId) -> std::io::Result<()> {
    let bc = crate::voxel::brickmap::brick_coord_of_voxel(w);
    let local = w - bc * BRICK_EDGE;
    pool.push(bc, voxel_index(local.x, local.y, local.z) as u16, block)
}

/// **Stage 1 — assemble the base LOD0 regions from the disk spills**, one region at a time, into `writer` via
/// `add_region` (regions fed in `(z,y,x)` order so the BRIK body layout matches `encode_vxo`). Each region's
/// spill is read, grouped into bricks via [`Brick::from_voxels`] (the SSOT — same uniform-collapse + occupancy),
/// added, then its spill is RE-SPILLED into `coarse_l0` (keyed by the COARSE-region of the brick, for the
/// Stage-2 windowed downsample) and DELETED. Resident peak = one region's bricks. Returns the count of base
/// bricks added (for reporting/asserts).
///
/// `base` must have had every solid voxel pushed + [`flush_all`](RegionSpillPool::flush_all) called. `coarse_l0`
/// is a fresh pool that this fills (the LOD0 bricks re-bucketed onto the coarse grid for level 1's window).
pub fn assemble_base(
    base: &RegionSpillPool,
    coarse_l0: &mut RegionSpillPool,
    writer: &mut VxoStreamWriter,
) -> anyhow::Result<u64> {
    let mut total: u64 = 0;
    for rc in base.sorted_regions() {
        let arrs = base.read_region_bricks(rc)?;
        // Build the region's bricks (sorted (z,y,x) within the region — the add_region contract).
        let mut coords: Vec<IVec3> = arrs.keys().copied().collect();
        coords.sort_by_key(|c| (c.z, c.y, c.x));
        let mut bricks_owned: Vec<(IVec3, Brick)> = Vec::with_capacity(coords.len());
        for c in &coords {
            let arr = arrs.get(c).expect("coord present");
            bricks_owned.push((*c, Brick::from_voxels(arr.clone())));
        }
        let bricks: Vec<(IVec3, &Brick)> = bricks_owned.iter().map(|(c, b)| (*c, b)).collect();
        writer.add_region(rc, &bricks)?;
        total += bricks.len() as u64;
        // Re-spill these LOD0 bricks for the Stage-2 windowed coarse (keyed by their COARSE region).
        for (c, brick) in &bricks_owned {
            respill_brick(coarse_l0, *c, brick)?;
        }
        base.delete_region(rc);
    }
    coarse_l0.flush_all()?;
    Ok(total)
}

/// **Stage 2 — windowed constant-RAM coarse downsample.** Build each level `L ∈ 1..=MAX_LOD` from the finer
/// level's spills (`finer`), emitting via `writer.add_lod_region` and re-spilling the coarse bricks for `L+1`.
/// Per COARSE region `crc`, load the ≤8 finer regions covering the child footprint `[2·crc·K, 2·(crc+1)·K)`
/// into one transient `BrickMap`, then per coarse brick `cc ∈ [crc·K, (crc+1)·K)` run
/// `gather_children`→`downsample_children` (the EXACT `source` SSOT — bit-identical to `build_coarse_pyramid`,
/// level chained from the previous). Resident = the ≤8-finer-region window + the emitted coarse region.
///
/// `finer` enters as `coarse_l0` from [`assemble_base`] (LOD0 bricks on the coarse grid). It is consumed level
/// by level: each level reads `finer`, produces `next`, deletes `finer`, then `finer = next`. Stops when a level
/// is empty (no solid coarse bricks) or `MAX_LOD` is reached — matching `build_coarse_pyramid` (solid-if-any keeps
/// every level non-empty until the cap, so the produced pyramid depth == `MAX_LOD` for a non-empty asset).
pub fn windowed_coarse(
    mut finer: RegionSpillPool,
    scratch: &Path,
    k: i32,
    writer: &mut VxoStreamWriter,
) -> anyhow::Result<()> {
    for lod in 1..=MAX_LOD {
        // The coarse grid for THIS level: each coarse brick `cc` aggregates finer bricks `[2cc, 2cc+2)`. The
        // coarse REGION `crc` (granularity k) owns coarse bricks `[crc·k, (crc+1)·k)`, whose children span
        // finer bricks `[2·crc·k, 2·(crc+1)·k)` = finer regions `[2·crc, 2·crc+2)` per axis (≤8 regions).
        let mut next = RegionSpillPool::new(scratch, &format!("coarse_l{lod}"), k);
        let mut any_emitted = false;

        // Every COARSE region we must produce: the coarse region of every finer brick's parent. Derive from the
        // finer spill's region coords — a finer region `fr` contains finer bricks `[fr·k, (fr+1)·k)`, whose
        // parents are coarse bricks `[fr·k/2 .. ((fr+1)·k-1)/2]`, i.e. coarse regions covering that span. The
        // simplest correct enumeration: each finer brick `fb` → coarse brick `fb/2` → coarse region of `fb/2`.
        // We don't have the bricks resident; but a finer region `fr` maps to coarse regions
        // `region_of_brick(fr·k/2, k) ..= region_of_brick(((fr+1)·k-1)/2, k)`. With k even (power of two ≥2),
        // `fr·k` is even so the parent coarse-brick span is `[fr·k/2, (fr·k + k-1)/2] = [fr·k/2, fr·k/2 + k/2 - ...]`.
        // To stay robust-by-construction we instead read each finer region once and route its parents — but that
        // would load finer regions out of the window. Cleanest: enumerate coarse regions directly from finer
        // region coords via the parent map below, then for each load its child window.
        let finer_regions = finer.sorted_regions();
        let mut coarse_regions: FxHashSet<IVec3> = FxHashSet::default();
        for &fr in &finer_regions {
            // Parent coarse-brick range of this finer region's brick span, mapped to coarse regions.
            let fb_lo = fr * k; // first finer brick in the region
            let fb_hi = fr * k + IVec3::splat(k - 1); // last finer brick in the region
            let cc_lo = IVec3::new(fb_lo.x.div_euclid(2), fb_lo.y.div_euclid(2), fb_lo.z.div_euclid(2));
            let cc_hi = IVec3::new(fb_hi.x.div_euclid(2), fb_hi.y.div_euclid(2), fb_hi.z.div_euclid(2));
            for cz in cc_lo.z..=cc_hi.z {
                for cy in cc_lo.y..=cc_hi.y {
                    for cx in cc_lo.x..=cc_hi.x {
                        coarse_regions.insert(region_of_brick(IVec3::new(cx, cy, cz), k));
                    }
                }
            }
        }
        let mut coarse_region_list: Vec<IVec3> = coarse_regions.into_iter().collect();
        coarse_region_list.sort_by_key(|c| (c.z, c.y, c.x));

        for crc in coarse_region_list {
            // Load the full child footprint: finer regions `[2·crc, 2·crc+2)` per axis (≤8 regions) into one
            // transient BrickMap. Loading EVERY overlapping finer region (not just the aligned one) is what makes
            // the cross-region gather bit-identical.
            let mut window = BrickMap::new();
            for dz in 0..2 {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let fr = IVec3::new(crc.x * 2 + dx, crc.y * 2 + dy, crc.z * 2 + dz);
                        load_region_into(&finer, fr, &mut window)?;
                    }
                }
            }
            if window.is_empty() {
                continue;
            }

            // Build this coarse region's bricks: each coarse brick `cc ∈ [crc·k, (crc+1)·k)` via the SSOT. Only
            // emit non-empty coarse bricks (matching `downsample_brickmap`, which inserts solid-if-any bricks).
            let cc_base = crc * k;
            let mut coords: Vec<IVec3> = Vec::new();
            let mut coarse_bricks: FxHashMap<IVec3, Brick> = FxHashMap::default();
            for lz in 0..k {
                for ly in 0..k {
                    for lx in 0..k {
                        let cc = cc_base + IVec3::new(lx, ly, lz);
                        let children = gather_children(cc, |fb| window.get(fb).cloned());
                        if children.iter().all(Option::is_none) {
                            continue;
                        }
                        let brick = downsample_children(&children);
                        if brick.is_empty() {
                            continue; // all children air ⇒ no coarse brick (matches the sparse SSOT)
                        }
                        coords.push(cc);
                        coarse_bricks.insert(cc, brick);
                    }
                }
            }
            if coords.is_empty() {
                continue;
            }
            coords.sort_by_key(|c| (c.z, c.y, c.x));
            let bricks: Vec<(IVec3, &Brick)> =
                coords.iter().map(|&c| (c, coarse_bricks.get(&c).expect("present"))).collect();
            writer.add_lod_region(lod, crc, &bricks)?;
            any_emitted = true;
            // Re-spill the coarse bricks for the NEXT level's window.
            for &c in &coords {
                respill_brick(&mut next, c, coarse_bricks.get(&c).expect("present"))?;
            }
        }

        next.flush_all()?;
        finer.delete_all();
        if !any_emitted {
            // Empty level ⇒ the pyramid stopped (matches `build_coarse_pyramid` stopping at the first empty
            // level). The just-created `next` has no spills; drop it.
            next.delete_all();
            break;
        }
        finer = next;
    }
    finer.delete_all();
    Ok(())
}

/// Load all bricks of finer region `fr` from `pool` into `window` (no-op if the region has no spill file).
fn load_region_into(pool: &RegionSpillPool, fr: IVec3, window: &mut BrickMap) -> anyhow::Result<()> {
    if !pool.regions.contains(&fr) {
        return Ok(());
    }
    let arrs = pool.read_region_bricks(fr)?;
    for (c, arr) in arrs {
        window.insert(c, Brick::from_voxels(arr));
    }
    Ok(())
}

/// Spill every SOLID voxel of `brick` (at brick coord `bc`) into `pool` (keyed by the brick's coarse region).
/// Used both to re-spill LOD0 bricks for level-1's window and to re-spill each coarse level for the next.
fn respill_brick(pool: &mut RegionSpillPool, bc: IVec3, brick: &Brick) -> std::io::Result<()> {
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                let b = brick.get(x, y, z);
                if !b.is_air() {
                    pool.push(bc, voxel_index(x, y, z) as u16, b)?;
                }
            }
        }
    }
    Ok(())
}
