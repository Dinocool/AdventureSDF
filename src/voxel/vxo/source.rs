//! The region-STREAMED `.vxo` loader — **Phase B-ii** (`docs/VXO_FORMAT.md` §B2).
//!
//! [`VxoSource`] memory-maps a whole `.vxo` file ONCE, eagerly parses the small `HEAD`/`MATL`/`BIDX` chunks,
//! and reads each region's compressed `BRIK` span LAZILY — only when a clipmap shell demands it — through a
//! byte-budgeted decoded-region LRU ([`RegionCache`]). So it implements the SAME [`BrickSource`] contract
//! [`super::super::source::StaticVoxSource`] does (`brick`/`classify`), feeding the EXISTING
//! [`super::super::streaming::ResidencyManager`] demand path — ONE residency SSOT for worldgen + static-`.vox`
//! (legacy) + `.vxo`. A 2.6 GB scene NEVER fully expands in RAM: only the demanded regions decode, the LRU
//! caps decoded RAM, and the residency caps the resident set (surface-only after `classify`).
//!
//! ## Why `brick()`/`classify()` stay PURE (the trait contract)
//! The trait requires a pure `Sync` function of `(coord, lod, registry)` so the parallel
//! [`super::super::streaming::ResidencyManager::drain_work_from`] is deterministic. The [`RegionCache`] is a
//! pure MEMOIZATION behind a [`Mutex`]: two threads decoding the same region get the same bytes (the file is
//! immutable), so the cache is observationally transparent — the SAME `(coord, lod)` always yields the SAME
//! `Brick` regardless of cache state or thread. Decoded regions are [`Arc`]'d so an in-flight parallel drain
//! can hold a region across the lock release.
//!
//! ## Coarse LODs — demand-downsampled from streamed LOD0 (§B1.7 option (a))
//! The baked `LODS` pyramid is deferred, so a coarse-LOD brick (`lod > 0`) is SYNTHESIZED on demand by
//! DOWNSAMPLING the streamed LOD0 data: a level-`L` brick is the `2³` downsample of its 8 children at `L-1`,
//! recursing to LOD0, through the SAME [`super::super::source::downsample_children`] reducer SSOT
//! [`super::super::source::StaticVoxSource`] uses — so a coarse `.vxo` brick is BIT-IDENTICAL to the static
//! source's coarse pyramid brick at every LOD (`brick`/`classify` parity, the §B2.8 lod-sweep gate). The
//! synthesis is RECURSIVE + MEMOIZED (a small coarse-brick memo behind the same `Mutex` as the region LRU), so
//! a deep pyramid is built once per demanded key and shared sub-bricks are reused, keeping `brick()` a pure
//! `Sync` function of its inputs. **Accepted v1 cost:** demand-downsampling a far coarse shell transiently
//! streams its LOD0 footprint regions (bounded by the surface-only residency + the region LRU) — the
//! baked-`LODS` optimization (§B1.7 option (b)) stays deferred for Phase C's gallery-RAM measurement.
//!
//! ## Merge composition ([`MergedSource`], §B2.4)
//! Several scenes load "into the world brick map" via a per-asset coordinate OFFSET + a per-asset palette
//! `block_base` remap (so two assets' `BlockId(5)` don't collide). [`MergedSource`] keeps that offset+rebase
//! as the SSOT, dispatching a world `coord` to whichever asset's (offset-applied) `HEAD.bounds` contains it.
//!
//! ## Live scene-switch wiring (deferred to Phase C/D)
//! Both [`VxoSource`] and [`MergedSource`] are [`BrickSource`]s, so a `.vxo` scene drops into the EXISTING
//! `ResidencyManager::drain_work_from(&dyn BrickSource, …)` loop with NO change to the streaming pipeline —
//! [`VxoSource::open`] / [`MergedSource::open_paths`] return the SAME `(source, registry)` shape `load_vox` +
//! `StaticVoxSource::new` / `load_gallery` do. The remaining work is purely the SCENE-SELECTION plumbing in
//! `raytrace.rs` (a `.vxo`-backed `VoxelScene` variant + its source field + lighting/sky preset + a toolbar
//! entry). That is deferred because the corpus (`Sponza`/`Sibenik`/`Conference`/`Bistro`) is still baked to
//! `.vox` — there is no `.vxo` asset to switch to yet. Phase C (the tiled voxelizer) produces the Bistro
//! `.vxo`; the corpus re-bake + the live switch land with it (`docs/VXO_FORMAT.md` "Migration"). Until then
//! the streamed loader ships as the tested public API above + the `open_paths` minimal load path.

use std::sync::{Arc, Mutex};

use bevy::math::IVec3;
use bytemuck::pod_read_unaligned;
use rustc_hash::FxHashMap;

use super::format::*;
use super::reader::{DecodedRegion, decode_region_span, parse_bidx, parse_head, parse_lods, parse_matl};
use crate::voxel::brickmap::{BRICK_EDGE, Brick, MAX_LOD, VOXEL_SIZE};
use crate::voxel::palette::{BlockId, BlockRegistry};
use crate::voxel::source::{BrickClass, BrickSource, downsample_children, gather_children};

/// The default decoded-region LRU byte budget (~256 MB, `VXO_FORMAT.md` §B2.2). Past this the LRU evicts the
/// least-recently-touched region — so the loader holds only the regions the shell currently needs, NEVER the
/// whole expanded scene. The budget bounds `RAM_peak ≈ HEAD/MATL/BIDX + this + the resident-set mirror`.
pub const DEFAULT_DECODED_REGION_BUDGET: usize = 256 * 1024 * 1024;

/// The byte-budgeted decoded-region LRU (`VXO_FORMAT.md` §B2.2). Maps a `(lod, K-brick-grid region coord)` key
/// to its decoded [`DecodedRegion`] (entries + region-local palette/index blobs), tracking a monotonic "last
/// touched" tick per region so eviction drops the least-recently-used once `bytes` exceeds `budget`. Regions
/// are [`Arc`]'d so [`VxoSource::brick`] can clone the handle out and drop the cache lock before decoding the
/// brick (a parallel drain holds the region across the release). This is a pure memoization — it never changes
/// what `brick()` returns, only how fast.
///
/// **The key carries the `lod` (gotcha #2):** base (LOD0) and baked-coarse (`lod > 0`) regions SHARE region
/// coords — region `(0,0,0)` exists at every level — and decode from DIFFERENT byte spans (the base `BRIK` vs.
/// each level's `BRIK_L`). Keying by coord ALONE would collide a coarse region with the base region of the
/// same coord and serve the WRONG bricks (a silent-corruption trap). `(lod, region_coord)` keeps every level's
/// regions in their own cache namespace.
struct RegionCache {
    /// `(lod, region_coord) -> (decoded region, last-touched tick)`.
    map: FxHashMap<(u32, IVec3), (Arc<DecodedRegion>, u64)>,
    /// Sum of every cached region's decoded byte size (the eviction budget is measured against this).
    bytes: usize,
    /// The eviction budget in bytes; `bytes` is kept `<= budget` after each insert (a single oversized region
    /// may transiently exceed it — it is still inserted so `brick` can serve it, then is the first evicted).
    budget: usize,
    /// A monotonic access counter; each touch stamps the region's tick, so the MIN tick is the LRU victim.
    tick: u64,
    /// PURE memoization of SYNTHESIZED COARSE bricks (`lod > 0`), keyed by `(coord, lod)` (§B1.7 option (a)).
    /// A coarse brick is the downsample of its 8 children at `lod-1`, recursing to LOD0 — memoized so a deep
    /// pyramid is built once per demanded key, not re-downsampled per `brick()` call (and the recursion reuses
    /// the same memo for shared sub-bricks). Behind the SAME [`Mutex`] so `brick()` stays a pure `Sync` function
    /// of its inputs (the file is immutable ⇒ a `(coord, lod)` always memoizes the SAME `Brick`). This is NOT
    /// LRU-evicted: a coarse brick is `≤ 4 KB` and the count is bounded by the surface-only resident set, so it
    /// is negligible against the region budget; it shares the source's lifetime.
    coarse: FxHashMap<(IVec3, u32), Arc<Brick>>,
}

impl RegionCache {
    /// An empty cache with the given byte budget.
    fn new(budget: usize) -> Self {
        Self {
            map: FxHashMap::default(),
            bytes: 0,
            budget,
            tick: 0,
            coarse: FxHashMap::default(),
        }
    }

    /// The decoded byte size of a region (its entry table + palette/index blobs) — the LRU budget accounting.
    /// Matches the in-RAM `DecodedRegion` footprint, not the compressed disk span.
    fn region_bytes(region: &DecodedRegion) -> usize {
        region.entries.len() * std::mem::size_of::<VxoBrickEntry>()
            + region.palette_blob.len() * 4
            + region.index_blob.len() * 4
    }

    /// A cached region (bumping its last-touched tick), or `None` on a miss. `key` is `(lod, region_coord)`.
    fn get(&mut self, key: (u32, IVec3)) -> Option<Arc<DecodedRegion>> {
        self.tick += 1;
        let tick = self.tick;
        let (region, last) = self.map.get_mut(&key)?;
        *last = tick;
        Some(Arc::clone(region))
    }

    /// Insert a freshly-decoded region, then EVICT least-recently-used regions until `bytes <= budget`. Returns
    /// the inserted `Arc` handle (so the caller serves this demand without a second lookup). `key` is
    /// `(lod, region_coord)`.
    fn insert(&mut self, key: (u32, IVec3), region: Arc<DecodedRegion>) -> Arc<DecodedRegion> {
        self.tick += 1;
        let sz = Self::region_bytes(&region);
        // Replace any stale entry's bytes (a re-decode of the same region — shouldn't happen with the get-first
        // path, but keep the accounting exact rather than double-count).
        if let Some((old, _)) = self.map.insert(key, (Arc::clone(&region), self.tick)) {
            self.bytes -= Self::region_bytes(&old);
        }
        self.bytes += sz;
        self.evict_to_budget(key);
        region
    }

    /// Evict the least-recently-touched regions until `bytes <= budget`. NEVER evicts `keep` (the region just
    /// inserted + about to be served) so a single demand always succeeds even if its region alone exceeds the
    /// budget. Stops when nothing else is evictable. `keep` is the `(lod, region_coord)` key.
    fn evict_to_budget(&mut self, keep: (u32, IVec3)) {
        while self.bytes > self.budget {
            // Find the LRU victim (lowest tick), excluding `keep`.
            let victim = self
                .map
                .iter()
                .filter(|(c, _)| **c != keep)
                .min_by_key(|(_, (_, last))| *last)
                .map(|(c, _)| *c);
            let Some(victim) = victim else { break }; // only `keep` left ⇒ can't evict further
            if let Some((region, _)) = self.map.remove(&victim) {
                self.bytes -= Self::region_bytes(&region);
            }
        }
    }
}

/// One baked coarse-LOD level's directory VIEW into the mmap'd `LODS` chunk (`VXO_FORMAT.md` §B1.7) — the
/// streamed analogue of [`super::reader::VxoLodsLevel`], holding the level's `BIDX_L` directory + the ABSOLUTE
/// mmap byte range of its `BRIK_L` blob (so a coarse region body is sliced straight off the mmap). `lod ==
/// index + 1` in [`VxoSource::lods`] (the 0-based vec position is `L - 1`).
struct LodLevelView {
    /// The pyramid LOD this level describes (`1..=max_lod`).
    lod: u32,
    /// The level's region directory on the COARSE grid (sorted by `(z,y,x)` — binary-searched on lookup).
    /// A region's `brik_offset` is LEVEL-LOCAL within `BRIK_L` (relative to `brik_l_start`).
    bidx_l: Vec<VxoRegionDirEntry>,
    /// ABSOLUTE byte offset of this level's `BRIK_L` blob within the mmap (`lods_body_start + level.brik_off`).
    /// A region body lives at `brik_l_start + dir.brik_offset`.
    brik_l_start: usize,
    /// Byte length of this level's `BRIK_L` blob (the seek bound for the level's region bodies).
    brik_l_len: usize,
}

/// The eager-parse result of [`VxoSource::parse_eager`] — the small chunks read up-front + the lazily-read
/// `BRIK`/`LODS` byte ranges. A named struct (over a 7-tuple) so the LODS additions don't make the call site
/// positional-soup.
struct EagerParse {
    head: VxoHead,
    bidx: Vec<VxoRegionDirEntry>,
    brik_body_start: usize,
    brik_body_len: usize,
    lods: Option<Vec<LodLevelView>>,
    registry: BlockRegistry,
}

/// A memory-mapped `.vxo` file exposed as a STREAMED [`BrickSource`] (`VXO_FORMAT.md` §B2.1) — the read side
/// feeding the SAME [`super::super::streaming::ResidencyManager`] demand path as the worldgen + static-`.vox`
/// sources. Lazy per-region reads off the mmap; a byte-budgeted decoded-region LRU; an `offset_bricks` merge
/// offset (§B2.4). [`Self::open`] mirrors `load_vox`'s `(map, registry)` return so the scene-load call site
/// swaps with no shape change.
pub struct VxoSource {
    /// The whole file, memory-mapped (durable; region bodies read lazily). Held for the source's lifetime.
    mmap: memmap2::Mmap,
    /// The byte offset of the `BRIK` chunk BODY within the file — a region's compressed span is
    /// `mmap[brik_body_start + dir.brik_offset .. +dir.brik_comp_len)`.
    brik_body_start: usize,
    /// The `BRIK` chunk body length (bounds-check the region spans).
    brik_body_len: usize,
    /// Parsed `HEAD` (voxel_size, bounds, K, anchor, counts).
    head: VxoHead,
    /// The sorted region directory (eager — it IS the spatial index; small even at Bistro scale).
    bidx: Vec<VxoRegionDirEntry>,
    /// The BAKED coarse-LOD pyramid directories (`VXO_FORMAT.md` §B1.7), one [`LodLevelView`] per `L ∈
    /// 1..=max_lod` (`lods[L-1]` is LOD `L`). `None` ⇒ the file has NO `LODS` chunk ⇒ coarse bricks are
    /// demand-DOWNSAMPLED (the forward-compat fallback, §B1.7 option (a)). `Some` ⇒ a coarse read is an O(1)
    /// directory lookup off the mmap (option (b), the Stage-2 freeze fix). `lods.len() == HEAD.max_lod`. Each
    /// [`LodLevelView`] holds its `BRIK_L` blob's ABSOLUTE mmap offset (`brik_l_start`), already rebased from the
    /// LODS-body-local offsets at parse time — so the LODS-body byte offset itself isn't retained past parse (the
    /// rebased per-level start is the SSOT for a coarse region read, with no per-read offset arithmetic).
    lods: Option<Vec<LodLevelView>>,
    /// The decoded-region LRU (§B2.2) behind a `Mutex` (pure memoization; the contract stays pure + `Sync`).
    cache: Mutex<RegionCache>,
    /// The merge OFFSET in LOD0 brick coords (§B2.4): added to incoming world brick coords' inverse —
    /// `local = coord - offset_bricks` — so this asset can be placed anywhere in a merged world. `(0,0,0)` for
    /// a stand-alone load.
    offset_bricks: IVec3,
    /// The per-asset palette `block_base` SHIFT (§B2.4): a solid local `BlockId(b)` decodes as merged
    /// `b + block_base` so two merged assets' ids don't collide. `0` for a stand-alone load (identity).
    block_base: u16,
}

impl VxoSource {
    /// Open + memory-map a `.vxo`, returning the streamed source + its rebuilt [`BlockRegistry`] — the SAME
    /// `(source, registry)` shape `load_vox` returns, so the scene-load call site swaps `load_vox` +
    /// `StaticVoxSource::new` for `VxoSource::open` with no shape change (§B2.1). Parses the small eager chunks
    /// (HEAD/MATL/BIDX) and records the `BRIK` body's byte range for lazy region reads. Asserts the asset's
    /// `voxel_size`/`brick_edge` match the engine (§B2.6); a `vxo-encode`-only / SVDAG file is rejected with a
    /// clear error. The decoded-region LRU starts empty with [`DEFAULT_DECODED_REGION_BUDGET`].
    pub fn open(path: impl AsRef<std::path::Path>) -> anyhow::Result<(Self, BlockRegistry)> {
        Self::open_with_budget(path, DEFAULT_DECODED_REGION_BUDGET)
    }

    /// As [`Self::open`] but with an explicit decoded-region byte budget (the LRU-eviction acceptance test
    /// drives a tiny budget to prove the loader never holds all regions at once, §B2.8 gate 1).
    pub fn open_with_budget(
        path: impl AsRef<std::path::Path>,
        budget: usize,
    ) -> anyhow::Result<(Self, BlockRegistry)> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .map_err(|e| anyhow::anyhow!("vxo: open {}: {e}", path.display()))?;
        // SAFETY: a read-only mmap of a file we hold open for the source's lifetime. The `.vxo` is an immutable
        // baked asset (the engine never writes one it's also reading), so the bytes don't change underneath us
        // — the standard memmap2 read-only contract. The `Mmap` is owned by `Self`, so the mapping outlives
        // every region slice we cast from it.
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| anyhow::anyhow!("vxo: mmap {}: {e}", path.display()))?;
        let parsed = Self::parse_eager(&mmap)?;

        // §B2.6 voxel_size reconciliation: assert-equal (NO silent rescale — the D1 flip + re-bake is one
        // atomic step; a mismatch between flip and re-bake is a BUILD error, not a silently-wrong scene).
        anyhow::ensure!(
            parsed.head.voxel_size == VOXEL_SIZE,
            "vxo: asset '{}' baked at {} m/voxel; engine VOXEL_SIZE is {} m — rebake the asset or flip the \
             engine (no silent rescale)",
            path.display(),
            parsed.head.voxel_size,
            VOXEL_SIZE
        );

        let source = Self {
            mmap,
            brik_body_start: parsed.brik_body_start,
            brik_body_len: parsed.brik_body_len,
            head: parsed.head,
            bidx: parsed.bidx,
            lods: parsed.lods,
            cache: Mutex::new(RegionCache::new(budget)),
            offset_bricks: IVec3::ZERO,
            block_base: 0,
        };
        Ok((source, parsed.registry))
    }

    /// Parse the eager chunks (HEAD/MATL/BIDX + the optional baked `LODS` pyramid) + locate the `BRIK` body byte
    /// range, off the mmapped file image — reusing the EXACT B-i chunk framing + parsers
    /// ([`parse_head`]/[`parse_matl`]/[`parse_bidx`]/[`parse_lods`]), so the streamed loader and the full-file
    /// [`super::reader::VxoFile`] agree on the format (one SSOT). Does NOT copy `BRIK` / the `LODS` `BRIK_L`
    /// blobs (the whole point — they're read lazily per region). Verifies the header + chunk CRCs and skips
    /// unknown chunks (the §B1.0 forward-compat rule).
    fn parse_eager(bytes: &[u8]) -> anyhow::Result<EagerParse> {
        anyhow::ensure!(bytes.len() >= 16, "vxo: file shorter than the 16-byte header");
        let fh: VxoFileHeader = pod_read_unaligned(&bytes[0..16]);
        anyhow::ensure!(fh.magic == VXO_MAGIC, "vxo: bad magic {:?} (expected VXO1)", fh.magic);
        anyhow::ensure!(
            fh.format_version == VXO_FORMAT_VERSION,
            "vxo: format_version {} unsupported (this reader is v{VXO_FORMAT_VERSION})",
            fh.format_version
        );
        anyhow::ensure!(crc32(&bytes[0..8]) == fh.header_crc32, "vxo: header CRC mismatch (file corrupt)");
        anyhow::ensure!(
            fh.flags & VXO_FLAG_SVDAG == 0,
            "vxo: file is SVDAG-encoded (flag bit1) — the B-ii streamed reader handles only plain R2b BRIK (B3)"
        );

        let mut head: Option<(VxoHead, String)> = None;
        let mut registry: Option<BlockRegistry> = None;
        let mut bidx: Option<Vec<VxoRegionDirEntry>> = None;
        let mut brik: Option<(usize, usize)> = None; // (body_start, body_len) — NOT copied
        // The OPTIONAL baked coarse pyramid: the parsed per-level views (each holding the ABSOLUTE mmap offset of
        // its `BRIK_L`, so region bodies are sliced lazily off the mmap, never copied).
        let mut lods: Option<Vec<LodLevelView>> = None;

        let ch_hdr = std::mem::size_of::<VxoChunkHeader>();
        let mut pos = std::mem::size_of::<VxoFileHeader>();
        while pos + ch_hdr <= bytes.len() {
            let ch: VxoChunkHeader = pod_read_unaligned(&bytes[pos..pos + ch_hdr]);
            let body_start = pos + ch_hdr;
            let body_len = ch.body_len as usize;
            anyhow::ensure!(body_start + body_len <= bytes.len(), "vxo: chunk {:?} body overruns file", ch.tag);
            let body = &bytes[body_start..body_start + body_len];
            if ch.body_crc32 != 0 {
                anyhow::ensure!(crc32(body) == ch.body_crc32, "vxo: chunk {:?} body CRC mismatch", ch.tag);
            }
            match ch.tag {
                TAG_HEAD => head = Some(parse_head(body)?),
                TAG_MATL => registry = Some(parse_matl(body)?),
                TAG_BIDX => bidx = Some(parse_bidx(body)?),
                // BRIK: record the body byte RANGE only — region bodies are read lazily off the mmap (§B2.2).
                TAG_BRIK => brik = Some((body_start, body_len)),
                // LODS (OPTIONAL, §B1.7): parse the per-level directories via the SHARED `parse_lods` SSOT, then
                // re-base each level's offsets onto the ABSOLUTE mmap (the parser's offsets are LODS-body-local).
                // Record `body_start` so a coarse region body is sliced straight off the mmap (no copy).
                TAG_LODS => {
                    let parsed = parse_lods(body)?;
                    let views = parsed
                        .levels
                        .into_iter()
                        .map(|lvl| LodLevelView {
                            lod: lvl.lod,
                            bidx_l: lvl.bidx_l,
                            // `brik_l_off` is relative to the LODS BODY START; rebase onto the mmap.
                            brik_l_start: body_start + lvl.brik_l_off,
                            brik_l_len: lvl.brik_l_len,
                        })
                        .collect();
                    lods = Some(views);
                }
                TAG_END => break,
                _ => { /* unknown chunk — skip (forward compat, §B1.0) */ }
            }
            pos = body_start + align16(ch.body_len) as usize;
        }

        let (head, _name) = head.ok_or_else(|| anyhow::anyhow!("vxo: missing REQUIRED HEAD chunk"))?;
        let registry = registry.ok_or_else(|| anyhow::anyhow!("vxo: missing REQUIRED MATL chunk"))?;
        let bidx = bidx.ok_or_else(|| anyhow::anyhow!("vxo: missing REQUIRED BIDX chunk"))?;
        let (brik_start, brik_len) = brik.ok_or_else(|| anyhow::anyhow!("vxo: missing REQUIRED BRIK chunk"))?;

        anyhow::ensure!(
            head.brick_edge == BRICK_EDGE as u32,
            "vxo: brick_edge {} != engine BRICK_EDGE {} — incompatible asset",
            head.brick_edge,
            BRICK_EDGE
        );
        // The region edge K must be a POWER OF TWO and > 0: `brick`/`classify` bucket a coord via
        // `div_euclid(K)`, so a corrupt K=0 would PANIC (div by zero); a non-power-of-two K is not a valid
        // `.vxo` region grid (the encoder always writes a power-of-two K). Mirror the `brick_edge` validation.
        anyhow::ensure!(
            head.region_edge_bricks > 0 && head.region_edge_bricks.is_power_of_two(),
            "vxo: region_edge_bricks {} must be a power of two > 0 (file corrupt)",
            head.region_edge_bricks
        );

        // Cross-check HEAD.max_lod against the LODS pyramid depth (the writer sets both from the same `max_lod`;
        // a mismatch is a corrupt/inconsistent file), mirroring the full-file `VxoFile::parse` check. With no
        // LODS chunk, HEAD.max_lod MUST be 0 (the §B1.7 no-pyramid convention).
        match &lods {
            Some(views) => anyhow::ensure!(
                head.max_lod as usize == views.len(),
                "vxo: HEAD.max_lod {} != LODS level count {} (inconsistent pyramid)",
                head.max_lod,
                views.len()
            ),
            None => anyhow::ensure!(
                head.max_lod == 0,
                "vxo: HEAD.max_lod {} but no LODS chunk (inconsistent — a baked pyramid is missing)",
                head.max_lod
            ),
        }

        Ok(EagerParse { head, bidx, brik_body_start: brik_start, brik_body_len: brik_len, lods, registry })
    }

    /// Place this source at a LOD0-brick `offset` in a merged world, shifting its solid `BlockId`s by
    /// `block_base` so they index the merged registry's slice (§B2.4). Consuming builder used by
    /// [`MergedSource::new`] — the offset+rebase SSOT lives there.
    pub fn placed(mut self, offset: IVec3, block_base: u16) -> Self {
        self.offset_bricks = offset;
        self.block_base = block_base;
        self
    }

    /// The parsed `HEAD` (voxel_size, bounds, K, anchor, counts) — read-only access for the merge bounds check.
    #[inline]
    pub fn head(&self) -> &VxoHead {
        &self.head
    }

    /// The region edge **K** (bricks per region axis), from `HEAD`.
    #[inline]
    fn region_edge(&self) -> i32 {
        self.head.region_edge_bricks as i32
    }

    /// Binary-search `bidx` (any level's directory, sorted by `(z,y,x)`) for `region_coord`; `None` ⇒ the
    /// region is absent (no entry ⇒ all-air at that level, the clipmap bound). Shared by the base LOD0 `BIDX`
    /// and each baked-coarse level's `BIDX_L`.
    fn region_entry_in(bidx: &[VxoRegionDirEntry], region_coord: IVec3) -> Option<&VxoRegionDirEntry> {
        let key = (region_coord.z, region_coord.y, region_coord.x);
        bidx.binary_search_by_key(&key, |e| (e.region_coord[2], e.region_coord[1], e.region_coord[0]))
            .ok()
            .map(|i| &bidx[i])
    }

    /// The deepest BAKED coarse level (`HEAD.max_lod`; `0` ⇒ no `LODS` chunk). A coarse read deeper than this
    /// clamps to this level (gotcha #4 — a tiny asset's pyramid collapsed early; its deepest level IS the answer
    /// for every coarser `L`, mirroring [`StaticVoxSource::level`](super::super::source::StaticVoxSource)).
    #[inline]
    fn max_lod(&self) -> u32 {
        self.head.max_lod
    }

    /// The baked-coarse [`LodLevelView`] that SERVES a request for `lod > 0`, or `None` if there is no `LODS`
    /// chunk. CLAMPS `lod` to `max_lod` (gotcha #4): for `max_lod < lod <= MAX_LOD` the deepest baked level is
    /// the answer (the collapsed-asset case). Panics never — `lod` is pre-clamped to `MAX_LOD` by the caller and
    /// `1 <= max_lod` whenever `lods.is_some()` (a baked pyramid has ≥ 1 level).
    fn coarse_level(&self, lod: u32) -> Option<&LodLevelView> {
        let levels = self.lods.as_ref()?;
        let max = self.max_lod();
        debug_assert!(max as usize == levels.len() && lod >= 1);
        let clamped = lod.min(max); // gotcha #4: clamp to the deepest baked level
        levels.get((clamped - 1) as usize)
    }

    /// The decoded region for `region_coord` at pyramid `lod` — a cache HIT (keyed by `(lod, region_coord)`,
    /// gotcha #2) clones the `Arc`; a MISS resolves the directory entry in `bidx`, reads the region's compressed
    /// span off the mmap at `brik_base + dir.brik_offset` (bounded by `[brik_base, brik_base + span_bound)`),
    /// decodes it via the shared [`decode_region_span`] SSOT (verifying the body's baked `lod`), inserts it into
    /// the `(lod, region)` LRU (evicting past the budget), and serves it. `Ok(None)` iff the region is absent
    /// from `bidx` (all-air). Pure memoization: the same `(lod, region)` always yields the same bytes.
    ///
    /// `bidx`/`brik_base`/`span_bound` select the LEVEL's byte layout: LOD0 ⇒ (`self.bidx`, `brik_body_start`,
    /// `brik_body_len`); coarse `L` ⇒ (`level.bidx_l`, `level.brik_l_start`, `level.brik_l_len`).
    fn decoded_region(
        &self,
        lod: u32,
        region_coord: IVec3,
        bidx: &[VxoRegionDirEntry],
        brik_base: usize,
        span_bound: usize,
    ) -> anyhow::Result<Option<Arc<DecodedRegion>>> {
        // Fast path: a cache hit under the lock (bumps the LRU tick), released before any decode.
        if let Some(region) = self.cache.lock().expect("region cache lock").get((lod, region_coord)) {
            return Ok(Some(region));
        }
        // Miss: resolve the directory span (absent ⇒ all-air).
        let Some(dir) = Self::region_entry_in(bidx, region_coord) else {
            return Ok(None);
        };
        let start = brik_base + dir.brik_offset as usize;
        let end = start + dir.brik_comp_len as usize;
        anyhow::ensure!(
            dir.brik_offset as usize + dir.brik_comp_len as usize <= span_bound && end <= self.mmap.len(),
            "vxo: L{lod} region {region_coord:?} span overruns its BRIK body"
        );
        // Decode OUTSIDE the lock (zstd decode of one region; a parallel drain decodes different regions
        // concurrently), then insert. A benign race where two threads decode the same region just inserts the
        // identical bytes twice — observationally transparent. The body's baked `lod` is verified by
        // `decode_region_span` against `lod` (base/coarse mix-up guard, gotcha #1).
        let region = Arc::new(decode_region_span(&self.mmap[start..end], dir, lod)?);
        let region = self.cache.lock().expect("region cache lock").insert((lod, region_coord), region);
        Ok(Some(region))
    }

    /// The number of regions currently held in the decoded-region LRU + its byte total — for the §B2.8 gate-1
    /// budget/eviction test (assert the loader never holds all regions at once, and `bytes <= budget`).
    pub fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.lock().expect("region cache lock");
        (cache.map.len(), cache.bytes)
    }

    /// The number of SYNTHESIZED coarse bricks currently memoized (`lod > 0`) — the demand-downsample footprint
    /// (every coarse brick built so far, retained for the source's lifetime). A residency-profiling probe for the
    /// coarse-LOD demand-downsample stage: paired with [`Self::cache_stats`] it shows how many transitive LOD0
    /// regions the coarse synthesis touched.
    pub fn coarse_memo_len(&self) -> usize {
        self.cache.lock().expect("region cache lock").coarse.len()
    }

    /// The core brick at LOCAL (offset-applied) brick coord `local`, level `lod`, reading the level's own
    /// directory + `BRIK` layout (`bidx`/`brik_base`/`span_bound`): bucket `local` to the level's Euclidean
    /// region grid → directory binary-search (absent ⇒ `uniform(AIR)`) → region-cache lookup (miss ⇒ lazy mmap
    /// read + decode + `(lod, region)` LRU insert) → in-region brick binary-search (absent ⇒ `uniform(AIR)`) →
    /// decode via [`DecodedRegion::brick_remapped`] (block_base-shifted for the merge). A decode error ⇒
    /// `uniform(AIR)` (the open-time CRC check would already have failed a genuinely corrupt file, so a runtime
    /// miss is the absent case). LOD0 and each baked-coarse level share this one path — only the level table
    /// + byte base differ (one SSOT for the in-region lookup, no per-level drift).
    fn brick_at_level(
        &self,
        local: IVec3,
        lod: u32,
        bidx: &[VxoRegionDirEntry],
        brik_base: usize,
        span_bound: usize,
    ) -> Brick {
        let k = self.region_edge();
        let region = IVec3::new(local.x.div_euclid(k), local.y.div_euclid(k), local.z.div_euclid(k));
        let Ok(Some(decoded)) = self.decoded_region(lod, region, bidx, brik_base, span_bound) else {
            return Brick::uniform(BlockId::AIR);
        };
        match decoded.entry(local) {
            Some(entry) => decoded.brick_remapped(entry, self.block_base),
            None => Brick::uniform(BlockId::AIR), // a coord that buckets here but was never stored ⇒ air
        }
    }

    /// The LOD0 core brick at LOCAL brick coord `local` — the LOD0 specialization of [`Self::brick_at_level`]
    /// (the base `BRIK` table). Also the LOD0 leaf of the demand-downsample fallback recursion ([`Self::coarse_brick`]).
    fn brick_lod0(&self, local: IVec3) -> Brick {
        self.brick_at_level(local, 0, &self.bidx, self.brik_body_start, self.brik_body_len)
    }

    /// The BAKED coarse brick at LOCAL brick coord `local`, level `lod > 0` — reads the `LODS` pyramid directly
    /// (§B1.7 option (b), the Stage-2 freeze fix). Resolves the serving [`LodLevelView`] (clamping past
    /// `max_lod`, gotcha #4), then defers to [`Self::brick_at_level`] over the LEVEL'S `BIDX_L`/`BRIK_L` — an
    /// O(1) directory lookup, NOT the recursive demand-downsample. The CALLER must have checked `lods.is_some()`.
    fn brick_coarse_baked(&self, local: IVec3, lod: u32) -> Brick {
        let level = self.coarse_level(lod).expect("brick_coarse_baked called without a LODS pyramid");
        self.brick_at_level(local, level.lod, &level.bidx_l, level.brik_l_start, level.brik_l_len)
    }

    /// Synthesize the COARSE brick at LOCAL (offset-applied) brick coord `local`, level `lod > 0`, by
    /// DOWNSAMPLING from the streamed LOD0 data — §B1.7 OPTION (a). RECURSIVE + MEMOIZED: a level-`L` brick is
    /// the [`downsample_children`] of its 8 children at `L-1` (each fetched via this same path, recursing to
    /// [`Self::brick_lod0`]); each synthesized `(local, lod)` is memoized in the [`RegionCache::coarse`] memo
    /// (pure — the file is immutable), so a deep pyramid is built ONCE per demanded key and shared sub-bricks
    /// are reused. This mirrors [`StaticVoxSource`](super::super::source::StaticVoxSource)'s iterative
    /// level-by-level pyramid EXACTLY (same octant layout + same `downsample_children` reducer SSOT), so the
    /// result is BIT-IDENTICAL at every LOD. Cost note: demand-downsampling a coarse shell transiently streams
    /// its LOD0 footprint regions (bounded by the surface + the LRU) — the accepted v1 cost (the baked-`LODS`
    /// optimization, §B1.7 option (b), stays deferred).
    fn coarse_brick(&self, local: IVec3, lod: u32) -> Arc<Brick> {
        // Memo hit (pure — `(local, lod)` ⇒ the SAME `Brick` regardless of cache state / thread).
        if let Some(b) = self.cache.lock().expect("region cache lock").coarse.get(&(local, lod)) {
            return Arc::clone(b);
        }
        // Recurse: gather the 8 children at `lod-1` (LOD0 leaf via `brick_lod0`), downsample via the SSOT.
        let children = gather_children(local, |child| {
            let b = if lod - 1 == 0 {
                self.brick_lod0(child)
            } else {
                (*self.coarse_brick(child, lod - 1)).clone()
            };
            // A wholly-air child contributes nothing — pass `None` (the `gather_children`/`downsample_children`
            // absent-octant convention), matching the static source's sparse pyramid (empty bricks unstored).
            (!b.is_empty()).then_some(b)
        });
        let synthesized = Arc::new(downsample_children(&children));
        self.cache
            .lock()
            .expect("region cache lock")
            .coarse
            .insert((local, lod), Arc::clone(&synthesized));
        synthesized
    }
}

impl BrickSource for VxoSource {
    /// The `8³` core brick at clipmap key `(coord, lod)` (§B2.2), BIT-IDENTICAL to
    /// [`StaticVoxSource::brick`](super::super::source::StaticVoxSource) at EVERY LOD.
    ///
    /// * **LOD0:** merge-offset → Euclidean region → `BIDX` binary-search (absent ⇒ `uniform(AIR)`, the clipmap
    ///   bound) → region-cache lookup (miss ⇒ lazy mmap read + decode + LRU insert) → in-region brick
    ///   binary-search (absent ⇒ `uniform(AIR)`) → decode via the B-i [`DecodedRegion::brick_remapped`] SSOT
    ///   (block_base-shifted for the merge), so a stand-alone load is bit-identical to a live brick.
    /// * **Coarse `lod > 0` WITH a baked `LODS` pyramid (§B1.7 OPTION (b), the Stage-2 fix):** an O(1) directory
    ///   lookup into the level's `BIDX_L`/`BRIK_L` via [`Self::brick_coarse_baked`] — bit-identical to the
    ///   demand path because the writer baked the pyramid through the SAME `downsample_brickmap` SSOT. A request
    ///   deeper than `max_lod` clamps to the deepest baked level (gotcha #4).
    /// * **Coarse `lod > 0` WITHOUT `LODS` (forward-compat fallback, OPTION (a)):** SERVED by DOWNSAMPLING the
    ///   streamed LOD0 data — the recursive, memoized [`Self::coarse_brick`] downsample of its 8 children at
    ///   `lod-1`, recursing to LOD0, through the SHARED [`downsample_children`] reducer.
    ///
    /// `StaticVoxSource` builds the same pyramid level-by-level from a non-empty finite map, which never
    /// collapses to empty (solid-if-any keeps ≥ 1 solid voxel forever), so its pyramid spans the full
    /// `MAX_LOD + 1` levels and a request is clamped only past `MAX_LOD` — we MIRROR that with `lod.min(MAX_LOD)`.
    fn brick(&self, coord: IVec3, lod: u32, _registry: &BlockRegistry) -> Brick {
        let local = coord - self.offset_bricks;
        // Clamp past MAX_LOD exactly as `StaticVoxSource::level` does (the pyramid of a non-empty finite map is
        // the full MAX_LOD+1 levels, so `level(lod) == lod.min(MAX_LOD)`).
        let lod = lod.min(MAX_LOD);
        if lod == 0 {
            self.brick_lod0(local)
        } else if self.lods.is_some() {
            // Baked pyramid present ⇒ O(1) LODS read (the freeze fix). `brick_coarse_baked` clamps past max_lod.
            self.brick_coarse_baked(local, lod)
        } else {
            // No LODS ⇒ demand-downsample (forward-compat fallback).
            (*self.coarse_brick(local, lod)).clone()
        }
    }

    /// The SAME conservative enclosed-cull as [`StaticVoxSource::classify`](super::super::source::StaticVoxSource)
    /// (§B2.5) at EVERY LOD, so the surface-only Θ(H²) residency holds for `.vxo` scenes. A brick is `Interior`
    /// (prunable) iff it is fully solid AND all 6 face-neighbours are fully solid; else `Surface`; an absent
    /// region/brick ⇒ `Air`. The coarse `is_full`/cull is bit-identical to the static pyramid (one downsample
    /// SSOT), so classify matches `StaticVoxSource::classify` at coarse LODs too (full parity).
    ///
    /// `is_full` reads the cheap BAKED [`BRICK_FLAG_FULL`] bit from the entry table — at LOD0 from the base
    /// `BRIK` directory, and at coarse LODs WITH a `LODS` pyramid from the level's `BIDX_L` entry table (no
    /// voxel decode either way, the Stage-2 fix). WITHOUT `LODS`, coarse `is_full` falls back to the
    /// demand-synthesized brick's [`Brick::is_full`](crate::voxel::brickmap::Brick::is_full).
    ///
    /// A request past the deepest level (`lod > MAX_LOD`, OR `lod > max_lod` for a baked-but-collapsed asset)
    /// is CLAMPED — its coord grid ≠ the served level grid — so the static source returns `Surface` (never
    /// prune); we mirror that.
    fn classify(&self, coord: IVec3, lod: u32) -> BrickClass {
        // Past the engine's deepest pyramid level ⇒ the static source clamps + returns Surface (the coord grid
        // ≠ the level grid). Mirror it. For a BAKED-but-collapsed asset, a request deeper than `max_lod` is the
        // same clamped case — the LODS branch below would read the deepest level's entries on the WRONG (finer)
        // coord grid, so short-circuit to Surface here to keep parity with StaticVoxSource::classify's clamp.
        if lod > MAX_LOD || (lod > 0 && self.lods.is_some() && lod > self.max_lod()) {
            return BrickClass::Surface;
        }
        let here = coord - self.offset_bricks;
        // The `is_full` of a brick at local coord `c`, level `lod`. The cheap baked path reads the
        // `BRICK_FLAG_FULL` bit straight from the level's entry table (no voxel decode); the no-LODS coarse
        // fallback downsamples. `None` ⇒ the brick is absent (all-air) in this asset.
        let is_full = |c: IVec3| -> Option<bool> {
            let baked = |lod: u32, bidx: &[VxoRegionDirEntry], brik_base: usize, span_bound: usize| {
                let k = self.region_edge();
                let region = IVec3::new(c.x.div_euclid(k), c.y.div_euclid(k), c.z.div_euclid(k));
                let decoded = self.decoded_region(lod, region, bidx, brik_base, span_bound).ok()??;
                let entry = decoded.entry(c)?;
                Some(entry.flags & BRICK_FLAG_FULL != 0)
            };
            if lod == 0 {
                baked(0, &self.bidx, self.brik_body_start, self.brik_body_len)
            } else if let Some(level) = self.coarse_level(lod) {
                // Baked coarse: the `is_full` flag from the level's entry table (cheap, no decode of voxels).
                baked(level.lod, &level.bidx_l, level.brik_l_start, level.brik_l_len)
            } else {
                // No LODS ⇒ demand-downsample, then report the synthesized brick's fullness. A wholly-air brick
                // is the "absent" case (mirrors the static pyramid's unstored empty bricks ⇒ Air).
                let b = self.coarse_brick(c, lod);
                (!b.is_empty()).then(|| b.is_full())
            }
        };

        match is_full(here) {
            None => BrickClass::Air,                    // absent ⇒ all-air outside the loaded asset
            Some(false) => BrickClass::Surface,         // has an internal air voxel ⇒ an exposed surface
            Some(true) => {
                // Fully solid: buried iff all 6 FACE-neighbours are fully solid too (no air-adjacent face).
                const N6: [IVec3; 6] = [
                    IVec3::new(1, 0, 0),
                    IVec3::new(-1, 0, 0),
                    IVec3::new(0, 1, 0),
                    IVec3::new(0, -1, 0),
                    IVec3::new(0, 0, 1),
                    IVec3::new(0, 0, -1),
                ];
                for off in N6 {
                    // A non-full / absent neighbour ⇒ this face is exposed ⇒ keep resident (Surface).
                    if is_full(here + off) != Some(true) {
                        return BrickClass::Surface;
                    }
                }
                BrickClass::Interior
            }
        }
    }

    /// **SHELL-FIRST candidate enumeration (D1d)** for a streamed `.vxo` — at LOD0, yield the world brick
    /// coords of the PRESENT `BIDX` regions that intersect `[lo, hi]` (a region is `K³` bricks). The `.vxo`'s
    /// region directory IS its spatial index, so this is a SUPERSET of every `Surface` brick: `classify`
    /// returns `Surface`/`Interior` only for bricks in a present region (an absent region/brick ⇒ `Air`), so
    /// iterating the present regions' bricks covers every `Surface` one — the buried/absent ones are pruned
    /// by the downstream `classify`. The candidate count is bounded by the present regions overlapping the
    /// shell (`Θ(surface)`), NOT the box volume, and crucially does NOT decode any region (it reads only the
    /// in-RAM `BIDX` + computes coord ranges) — so the residency narrows the cube to the shell BEFORE paying
    /// any region decode.
    ///
    /// **Coarse `lod > 0` WITH a baked `LODS` pyramid (the Stage-2 narrowing):** the SAME region-intersection,
    /// but over the LEVEL's `BIDX_L` instead of the base `BIDX` — each coarse level has its OWN region directory
    /// on its OWN coord grid (a level-`L` region is `K³` level-`L` bricks), so the candidate set narrows to
    /// `Θ(surface)` at coarse LODs too (no decode). A request deeper than `max_lod` is a CLAMPED level (grid ≠
    /// the `lod` grid), so — like [`StaticVoxSource::surface_bricks_in`] — it falls back to the FULL BOX.
    /// **Coarse `lod > 0` WITHOUT `LODS` falls back to the FULL BOX** (the trait default): no coarse region
    /// directory to intersect (the pyramid is demand-synthesized) — a correct superset, bounded by the empty-memo.
    fn surface_bricks_in(&self, lo: IVec3, hi: IVec3, lod: u32, out: &mut Vec<IVec3>) {
        // Pick the region directory for this level: LOD0 ⇒ base BIDX; coarse-with-LODS ⇒ the level's BIDX_L
        // (only when `lod <= max_lod` — a clamped level's grid ≠ the requested grid). Else (no LODS coarse, or
        // a clamped coarse level) there is no directory on this coord grid ⇒ full-box fallback.
        let bidx: Option<&[VxoRegionDirEntry]> = if lod == 0 {
            Some(&self.bidx)
        } else if lod <= self.max_lod() {
            // `coarse_level` clamps, but `lod <= max_lod` here so it returns the EXACT level (grid matches).
            self.coarse_level(lod).map(|lvl| lvl.bidx_l.as_slice())
        } else {
            None
        };
        let Some(bidx) = bidx else {
            // Full box (correct superset; the empty-memo / wholly-outside reject bounds it downstream).
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    for x in lo.x..=hi.x {
                        out.push(IVec3::new(x, y, z));
                    }
                }
            }
            return;
        };
        let k = self.region_edge();
        // World brick coord -> local (offset-applied) coord: `local = coord - offset_bricks`. The box in LOCAL
        // coords; intersect each present region's [r·K, r·K + K) local span with it, then shift back to world.
        let lo_l = lo - self.offset_bricks;
        let hi_l = hi - self.offset_bricks;
        for dir in bidx {
            let rc = IVec3::new(dir.region_coord[0], dir.region_coord[1], dir.region_coord[2]);
            let rlo = rc * k; // inclusive local brick min of this region
            let rhi = rlo + IVec3::splat(k - 1); // inclusive local brick max
            // Intersect the region's local span with the (local) box; skip a region wholly outside.
            let ax = rlo.x.max(lo_l.x);
            let bx = rhi.x.min(hi_l.x);
            let ay = rlo.y.max(lo_l.y);
            let by = rhi.y.min(hi_l.y);
            let az = rlo.z.max(lo_l.z);
            let bz = rhi.z.min(hi_l.z);
            if ax > bx || ay > by || az > bz {
                continue;
            }
            // Yield the overlap (shifted back to WORLD coords). A superset: every stored brick of the region is
            // covered; the downstream classify prunes the absent/buried ones.
            for z in az..=bz {
                for y in ay..=by {
                    for x in ax..=bx {
                        out.push(IVec3::new(x, y, z) + self.offset_bricks);
                    }
                }
            }
        }
    }
}

/// The MERGED-GALLERY source (`VXO_FORMAT.md` §B2.4): several `.vxo` assets loaded into ONE world brick map by
/// a per-asset coordinate OFFSET + a per-asset palette `block_base` remap. Dispatches a world `coord` to
/// whichever asset's (offset-applied) `HEAD.bounds` contains it, taking the non-air result — so each region
/// read still hits exactly ONE asset's mmap. The merged [`BlockRegistry`] is the per-asset concatenation (each
/// asset's solid ids shifted by its base, so two `BlockId(5)` don't collide). This is the offset+rebase SSOT
/// — it composes N independent `.vxo` files with NO re-bake.
pub struct MergedSource {
    /// The placed assets (each already carrying its `offset_bricks` + `block_base` via [`VxoSource::placed`]),
    /// paired with the placed LOD0-brick AABB `[lo, hi)` of its solid extent (for the dispatch bounds test).
    assets: Vec<PlacedAsset>,
}

/// One placed asset in a [`MergedSource`]: the streamed source + its placed brick-coord AABB (the dispatch key).
struct PlacedAsset {
    /// The streamed source, already offset + block-base remapped.
    source: VxoSource,
    /// Inclusive-exclusive placed LOD0 brick-coord bounds `[lo, hi)` — `coord` falls in this asset iff inside.
    lo: IVec3,
    hi: IVec3,
}

impl MergedSource {
    /// Build a merged source + its concatenated [`BlockRegistry`] from `(VxoSource, BlockRegistry, offset)`
    /// triples — the offset is the asset's LOD0-brick placement in the merged world. Each asset's solid blocks
    /// are appended after the running merged registry (its local `BlockId(b)` → merged `block_base - 1 + b`,
    /// the SAME `palette_base` convention as `gallery::merge_scenes`), and the source is `placed` with that
    /// offset + shift. An asset that would push the merged palette past `u16::MAX` BlockIds is logged + SKIPPED
    /// (capping the gallery, never wrapping a `BlockId`), mirroring the gallery merge. Returns
    /// `(MergedSource, merged_registry)` — the `(source, registry)` shape the residency wiring expects.
    pub fn new(assets: Vec<(VxoSource, BlockRegistry, IVec3)>) -> (Self, BlockRegistry) {
        let mut merged_registry = BlockRegistry::air_only();
        let mut placed: Vec<PlacedAsset> = Vec::with_capacity(assets.len());
        for (source, registry, offset) in assets {
            // The merged index this asset's local block 1 lands at (AIR occupies index 0, so a non-empty merged
            // registry has len >= 1). The per-voxel shift is `palette_base - 1` (local b -> base-1+b).
            let palette_base = merged_registry.len() as u32;
            let solid_blocks = registry.len().saturating_sub(1) as u32; // exclude AIR
            if solid_blocks > 0 {
                let highest = palette_base + solid_blocks - 1;
                if highest > u16::MAX as u32 {
                    bevy::log::warn!(
                        "vxo merge: an asset would push the merged palette past u16::MAX BlockIds \
                         (base {palette_base} + {solid_blocks} blocks ⇒ highest {highest}) — skipping it and \
                         any later assets"
                    );
                    break;
                }
                merged_registry.extend_blocks_from(&registry);
            }
            let block_base = (palette_base - 1) as u16; // shift for a solid local id b -> base-1+b
            // The placed brick-coord AABB: HEAD bounds are LOD0 world VOXELS; convert to brick coords + offset.
            let bmin = IVec3::from_array(source.head.bounds_min);
            let bmax = IVec3::from_array(source.head.bounds_max);
            let lo = IVec3::new(bmin.x.div_euclid(BRICK_EDGE), bmin.y.div_euclid(BRICK_EDGE), bmin.z.div_euclid(BRICK_EDGE)) + offset;
            // bounds_max is EXCLUSIVE voxels; the exclusive brick bound is ceil-div, expressed via the last
            // inclusive voxel's brick + 1.
            let hi_incl = IVec3::new((bmax.x - 1).div_euclid(BRICK_EDGE), (bmax.y - 1).div_euclid(BRICK_EDGE), (bmax.z - 1).div_euclid(BRICK_EDGE));
            let hi = hi_incl + offset + IVec3::ONE;
            placed.push(PlacedAsset { source: source.placed(offset, block_base), lo, hi });
        }
        (Self { assets: placed }, merged_registry)
    }

    /// Open + merge several `.vxo` files at `(path, offset)` placements into ONE [`MergedSource`] +
    /// concatenated [`BlockRegistry`] — the disk-facing convenience for the gallery merge path (the
    /// `.vxo` sibling of `gallery::load_gallery`). A file that fails to open is logged + SKIPPED (a
    /// partially-baked gallery still loads the assets that exist, never panics), mirroring the legacy gallery.
    /// Returns `(MergedSource, registry)` — the `(source, registry)` shape the residency wiring expects, so the
    /// gallery call site can swap `load_gallery` + `StaticVoxSource::new` for this with no shape change once the
    /// corpus is re-baked to `.vxo` (the live scene-switch wiring lands in Phase C/D — see the module doc).
    pub fn open_paths(paths: &[(std::path::PathBuf, IVec3)]) -> (Self, BlockRegistry) {
        let mut assets: Vec<(VxoSource, BlockRegistry, IVec3)> = Vec::with_capacity(paths.len());
        for (path, offset) in paths {
            match VxoSource::open(path) {
                Ok((source, registry)) => assets.push((source, registry, *offset)),
                Err(e) => bevy::log::warn!("vxo merge: skipping '{}': {e}", path.display()),
            }
        }
        Self::new(assets)
    }

    /// Aggregate decoded-region-LRU + coarse-memo stats across every placed asset:
    /// `(decoded_regions, decoded_bytes, coarse_bricks)` — the SUM over assets of each asset's
    /// [`VxoSource::cache_stats`] + [`VxoSource::coarse_memo_len`]. A residency-profiling probe (the streamed
    /// gallery's RAM + demand-downsample footprint at a glance); pure read of the per-asset caches.
    pub fn cache_stats(&self) -> (usize, usize, usize) {
        let mut regions = 0usize;
        let mut bytes = 0usize;
        let mut coarse = 0usize;
        for a in &self.assets {
            let (r, b) = a.source.cache_stats();
            regions += r;
            bytes += b;
            coarse += a.source.coarse_memo_len();
        }
        (regions, bytes, coarse)
    }

    /// The asset whose placed brick AABB contains `coord`, or `None` if `coord` is in no asset's extent. The
    /// gallery guarantees disjoint placement (the caller spaces assets apart), so at most one asset matches —
    /// the first containing asset is taken (deterministic by insertion order).
    fn asset_at(&self, coord: IVec3) -> Option<&PlacedAsset> {
        self.assets.iter().find(|a| {
            coord.x >= a.lo.x
                && coord.y >= a.lo.y
                && coord.z >= a.lo.z
                && coord.x < a.hi.x
                && coord.y < a.hi.y
                && coord.z < a.hi.z
        })
    }
}

impl BrickSource for MergedSource {
    /// Dispatch `coord` to the asset whose placed bounds contain it (the merge SSOT), returning that asset's
    /// brick (already offset + block-base remapped). No matching asset ⇒ `uniform(AIR)` (the merged-world
    /// clipmap bound).
    fn brick(&self, coord: IVec3, lod: u32, registry: &BlockRegistry) -> Brick {
        match self.asset_at(coord) {
            Some(asset) => asset.source.brick(coord, lod, registry),
            None => Brick::uniform(BlockId::AIR),
        }
    }

    /// Classify via the owning asset (so the per-asset enclosed-cull is preserved across the merge). No
    /// matching asset ⇒ `Air`. Because assets are placed DISJOINT (with a gap), a brick on one asset's edge has
    /// its outward neighbour absent within that asset ⇒ `Surface` — correct (no false `Interior` across a gap).
    fn classify(&self, coord: IVec3, lod: u32) -> BrickClass {
        match self.asset_at(coord) {
            Some(asset) => asset.source.classify(coord, lod),
            None => BrickClass::Air,
        }
    }

    /// **SHELL-FIRST candidate enumeration (D1d)** across the merged gallery: ask EACH asset whose placed
    /// brick AABB intersects `[lo, hi]` for its surface candidates, clipped to the overlap. Because assets
    /// are placed DISJOINT (with a gap), their candidate sets don't collide; the union is a superset of every
    /// merged-world `Surface` brick (a coord in no asset is `Air` and yields nothing — the merged clipmap
    /// bound). Delegating to each asset's `surface_bricks_in` preserves the per-asset region-directory bound
    /// (`Θ(surface)`, no region decode), so the merge stays shell-first too.
    fn surface_bricks_in(&self, lo: IVec3, hi: IVec3, lod: u32, out: &mut Vec<IVec3>) {
        for asset in &self.assets {
            // The overlap of this asset's placed bounds with the query box (asset bounds are [lo, hi) excl).
            let ax = asset.lo.x.max(lo.x);
            let ay = asset.lo.y.max(lo.y);
            let az = asset.lo.z.max(lo.z);
            let bx = (asset.hi.x - 1).min(hi.x);
            let by = (asset.hi.y - 1).min(hi.y);
            let bz = (asset.hi.z - 1).min(hi.z);
            if ax > bx || ay > by || az > bz {
                continue; // this asset doesn't touch the query box
            }
            asset.source.surface_bricks_in(IVec3::new(ax, ay, az), IVec3::new(bx, by, bz), lod, out);
        }
    }
}

#[cfg(test)]
#[path = "source_tests.rs"]
mod tests;
