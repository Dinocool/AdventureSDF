//! **Phase G "G-c.4-paging" — the STREAMED `.vxo` region PREFETCHER + demand-paged GPU residency stores**
//! (`docs/PHASE_G_GC_PLAN.md` §8).
//!
//! This is the close-out that drives the LIVE GPU residency front end ([`super::residency_front_end`]) over a
//! region-PAGED `.vxo` ([`super::vxo::MergedSource`], the streamed Bistro) instead of an eager whole-scene store.
//! The front end (proven on in-RAM Sponza) face-culls against a GPU OCCUPANCY structure and halo-fills from a GPU
//! CORE STORE; for the streamed path those two structures cannot be built eagerly (whole-Bistro cores are ~GBs —
//! it would break constant-RAM), so this module PAGES them per-region, camera-driven, readback-free:
//!
//! * **The prefetcher ([`StreamedResidencyPager::update`])** is the NEW driver. Camera-driven, no GPU→CPU
//!   readback (re-flora pattern): for each LOD it takes the clipmap [`level_box_pub`] world brick AABB, maps it
//!   into each [`PlacedAsset`]'s LOCAL coord space (`world - offset_at_lod`), and collects the PRESENT regions
//!   intersecting it PADDED +1 brick (the 26-halo core-coverage invariant) via
//!   [`VxoSource::present_regions_in`](super::vxo::source::VxoSource::present_regions_in). On a region-set CHANGE
//!   (a crossing — infrequent) it pages newly-covered regions in + drops uncovered ones.
//! * **Occupancy (§8.2)** rebuilds WHOLE from the resident regions on a crossing (cheap, ~MB; reuses
//!   [`SectorOccupancy::from_occupied_full`]) into a PRE-SIZED, re-uploaded buffer (no realloc → no rebind).
//! * **Cores (§8.3)** are INCREMENTAL: a [`PagedBrickCoreStore`] mirrors the [`MergedSource`] region lifecycle
//!   (upload-on-decode, evict-on-drop) so the GPU core footprint ≤ the resident-region set (constant-RAM). A whole
//!   rebuild (~300 MB/crossing) is forbidden.
//!
//! **COVERAGE INVARIANT (§8.3):** the prefetcher pages exactly the clipmap-covering present regions PADDED +1
//! brick, and the GPU enumerate only ENTERS bricks with `level_resident` (inside the clipmap) — so every enterable
//! brick + its 26-halo has its core resident. The +1-brick prefetch pad + the clipmap-bounded enumerate together
//! guarantee no missing-core hole.

use bevy::math::IVec3;
use rustc_hash::{FxHashMap, FxHashSet};

use super::brickmap::MAX_LOD;
use super::residency_gpu::{
    GpuBrickCoreBuffers, GpuResidencyBuffers, PagedBrickCoreStore, ResidencyProducer, SectorOccupancy,
};
use super::streaming::level_box_pub;
use super::vxo::{DecodedRegion, MergedSource};

/// A resident region key: `(asset index, lod, region_coord on the LOD-lod grid)`.
type RegionKey = (usize, u32, IVec3);

/// The max cores a single GPU `cores` storage BUFFER can hold under wgpu's 2 GiB `max_storage_buffer_binding_size`:
/// `cores · BRICK_VOXELS · 4 bytes ≤ ~2 GiB`. Conservative (1.84 GiB) so the buffer + headroom stays under the
/// limit on every adapter. The surface-shell core footprint is capped to this — the constant-RAM ceiling.
const MAX_CORE_BUFFER_CORES: u32 = 900_000;

/// The streamed-source region prefetcher + the two demand-paged GPU residency stores it drives. Held in
/// `VoxelRtResources` for the live streamed scene; rebuilt on a scene/epoch switch. One [`Self::update`] per frame
/// (camera-driven) pages the clipmap-covering present regions in / out — only doing real work on a region crossing.
pub struct StreamedResidencyPager {
    /// The live streamed `.vxo` source (the merged Bistro / `.vxo` gallery). `Arc` shared with the main world.
    source: std::sync::Arc<MergedSource>,
    /// The scene epoch this pager belongs to (matched against the live params epoch by the caller).
    epoch: u64,
    /// The clip half-extent in bricks (the prefetcher's per-LOD `level_box` reach). Fixed for the epoch.
    clip_half: i32,
    /// 4-S4: the coarse-backdrop LOD threshold (LODs >= this page out to `clip_half · BACKDROP_REACH` so the cheap
    /// coarse backdrop extends beyond the fine clipmap). Read from `ADVENTURE_BACKDROP_LOD` (default off = no extension).
    backdrop_lod: u32,

    /// The currently-paged region set (so a frame's update is a SET DIFF — page in newly-covered, drop uncovered).
    resident: FxHashSet<RegionKey>,
    /// The decoded region handles for every resident region — kept alive HERE (not relying on the source LRU,
    /// which may evict) so the occupancy whole-rebuild can re-enumerate every resident region's bricks each
    /// crossing. Dropped on eviction (mirrors the LRU lifecycle; constant-RAM, bounded by the resident set).
    decoded: FxHashMap<RegionKey, std::sync::Arc<DecodedRegion>>,

    /// The GROWABLE GPU occupancy (§8.2): a PRE-SIZED `entries` buffer re-uploaded whole each crossing.
    occ_buffers: GpuResidencyBuffers,
    /// The pre-sized occupancy `entries` slot capacity (the whole-scene sector estimate; rebuilds must fit it).
    occ_capacity: u32,

    /// The DEMAND-PAGED GPU core store (§8.3): incremental insert-on-page / evict-on-drop.
    core_store: PagedBrickCoreStore,

    /// Whether the GPU stores have been (re)bound since the last structural change — the caller rebinds the front
    /// end's bind group when `occ_buffers`/`core_store` buffers change identity (they don't here — both are
    /// in-place re-uploaded — so a rebind is needed only ONCE after construction).
    needs_rebind: bool,
}

impl StreamedResidencyPager {
    /// Build a fresh pager + its (empty) GPU stores for `source` at `epoch`. `clip_half` is the residency reach;
    /// `max_resident` caps the core store to the resident-set footprint (constant-RAM). The occupancy buffer is
    /// pre-sized from a whole-scene sector estimate so a per-crossing whole re-upload never reallocs/rebinds.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: std::sync::Arc<MergedSource>,
        epoch: u64,
        clip_half: i32,
        max_resident: u32,
    ) -> Self {
        // Occupancy pre-size: the resident occupancy is bounded by the clipmap-surface sectors (a few MB worst
        // case). A SECTOR is 4³ = 64 bricks, so an upper bound on resident sectors is `max_resident` (one sector
        // per resident brick is the pessimistic non-coalesced bound) — pre-size the hash to comfortably hold that
        // at ≤ 0.5 load factor, with a generous floor so a small scene still gets a usable table.
        let est_sectors = max_resident.max(1);
        let occ_capacity = ((est_sectors as usize * 2).max(1 << 14)).next_power_of_two() as u32;
        // Start with an EMPTY occupancy (no resident regions yet); the first `update` rebuilds it.
        let empty_occ = SectorOccupancy::from_occupied_full(std::iter::empty());
        let occ_buffers = empty_occ.upload_presized(device, queue, occ_capacity);

        // Core store: cap to the SURFACE-shell footprint (§8.3). The pager pages cores ONLY for SURFACE bricks +
        // their 26-halo neighbours (NOT deep-interior bricks — the front end never enters them and they are never
        // an entered brick's halo), so the resident core set ≈ the clipmap SURFACE shell, the SAME Θ(H²) footprint
        // the live pool holds. `max_resident` is the front-end resident pool cap (the surface set the pool sizes
        // for); the halo + page-overlap adds a constant factor, but a single `cores` storage BUFFER cannot exceed
        // wgpu's 2 GiB `max_storage_buffer_binding_size` (= `MAX_CORE_BUFFER_CORES` cores), so cap to the min. The
        // surface shell of a dense scene at full clip_half can hit that ceiling — that is the constant-RAM bound
        // (one bounded buffer), and a brick beyond it is simply not core-resident (a graceful far-detail drop, not
        // a crash; the free-list panic would be a mis-size — `MAX_CORE_BUFFER_CORES` keeps us under the limit).
        let core_cap = max_resident.saturating_mul(2).clamp(1, MAX_CORE_BUFFER_CORES);
        // 4-S4: match the front end's backdrop threshold (default MAX_LOD+1 = off) so the pager pages the extended
        // coarse-backdrop regions the front end will enumerate beyond `clip_half`.
        let backdrop_lod = std::env::var("ADVENTURE_BACKDROP_LOD")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(MAX_LOD + 1);

        Self {
            source,
            epoch,
            clip_half,
            backdrop_lod,
            resident: FxHashSet::default(),
            decoded: FxHashMap::default(),
            occ_buffers,
            occ_capacity,
            core_store: PagedBrickCoreStore::new(device, queue, core_cap),
            needs_rebind: true,
        }
    }

    /// Compute the CLIPMAP-COVERING present region set for `cam` (the desired resident set): per LOD, the
    /// `level_box` world brick AABB mapped into each placed asset's local coords, collecting present regions
    /// PADDED +1 brick. Pure read of the in-RAM directories (no decode) — the camera-driven, readback-free driver.
    fn desired_regions(&self, cam: [f32; 3]) -> FxHashSet<RegionKey> {
        let mut desired = FxHashSet::default();
        let mut scratch: Vec<IVec3> = Vec::new();
        for lod in 0..=MAX_LOD {
            // 4-S4: backdrop LODs reach BACKDROP_REACH× farther (page the extended coarse backdrop the front end
            // enumerates beyond clip_half). Matches `residency_front_end::lod_clip_half` + the WGSL.
            let (wlo, whi) = level_box_pub(cam, lod, super::residency_front_end::lod_clip_half(lod, self.clip_half, self.backdrop_lod));
            for (ai, asset) in self.source.placed_assets().iter().enumerate() {
                // Clip the world clipmap box to the asset's placed bounds (on the LOD-lod grid), then to LOCAL.
                let (alo, ahi) = self.source.asset_lod_bounds(ai, lod);
                let lo = IVec3::new(wlo.x.max(alo.x), wlo.y.max(alo.y), wlo.z.max(alo.z));
                let hi = IVec3::new(whi.x.min(ahi.x), whi.y.min(ahi.y), whi.z.min(ahi.z));
                if lo.x > hi.x || lo.y > hi.y || lo.z > hi.z {
                    continue; // this asset doesn't touch the clipmap box at this LOD
                }
                // World→local + region-bucket through the asset's SSOT (no offset math here) so the desired set,
                // the occupancy build, and the core-fetch resolve all agree for any (off-origin) placement.
                scratch.clear();
                asset.source.present_world_regions_in(lod, lo, hi, &mut scratch);
                for &rc in &scratch {
                    desired.insert((ai, lod, rc));
                }
            }
        }
        desired
    }

    /// Rebuild the occupancy hash from all resident regions' bricks + re-upload it in place (§8.2). Returns the CPU
    /// [`SectorOccupancy`] (so `collect_surface_halo_keys` can `classify_surface` without a second build).
    fn rebuild_occupancy(&mut self, queue: &wgpu::Queue) -> SectorOccupancy {
        let mut occupied: Vec<(IVec3, u32, bool)> = Vec::new();
        for (&(ai, lod, _rc), region) in &self.decoded {
            let asset = &self.source.placed_assets()[ai];
            asset.source.for_each_region_brick_occ(lod, region, |world, l, is_full| {
                occupied.push((world, l, is_full));
            });
        }
        let occ = SectorOccupancy::from_occupied_full(occupied);
        occ.reupload_into(queue, &mut self.occ_buffers, self.occ_capacity);
        occ
    }

    /// The FULL surface+halo core KEY set for the CURRENT resident occupancy, paired with each key's owning asset
    /// (for the lazy core decode). Every SURFACE brick (`occ.classify_surface` — the front end's enterable set)
    /// plus each surface brick's 26 PRESENT halo neighbours (the cores the GPU halo-fill reads). Θ(resident bricks).
    /// Because the gallery spaces assets DISJOINT with a gap ≥ 1 brick, a ±1 halo neighbour is always in the SAME
    /// asset, so `key_asset` for a halo key is that surface brick's asset. The SSOT that keeps the core set in
    /// lock-step with the occupancy each crossing (the coverage invariant).
    fn collect_surface_halo_keys(
        &self,
        occ: &SectorOccupancy,
    ) -> (FxHashSet<(IVec3, u32)>, FxHashMap<(IVec3, u32), usize>) {
        let mut desired: FxHashSet<(IVec3, u32)> = FxHashSet::default();
        let mut key_asset: FxHashMap<(IVec3, u32), usize> = FxHashMap::default();
        for (&(ai, lod, _rc), region) in &self.decoded {
            let asset = &self.source.placed_assets()[ai];
            asset.source.for_each_region_brick_occ(lod, region, |world, l, _is_full| {
                if occ.classify_surface(world, l) {
                    desired.insert((world, l));
                    key_asset.insert((world, l), ai);
                    for off in N26 {
                        let n = world + off;
                        if occ.is_occupied(n, l) {
                            desired.insert((n, l));
                            key_asset.entry((n, l)).or_insert(ai);
                        }
                    }
                }
            });
        }
        (desired, key_asset)
    }
}

impl ResidencyProducer for StreamedResidencyPager {
    #[inline]
    fn epoch(&self) -> u64 {
        self.epoch
    }

    /// **The prefetcher (§8.1)** — page the clipmap-covering present regions for `cam`: diff the desired region set
    /// vs the resident one, decode newly-covered regions (drop uncovered). On a crossing, rebuild the OCCUPANCY
    /// whole from the resident regions (§8.2, cheap ~bit/brick) and re-sync the CORE store to the SURFACE shell +
    /// its 26-halo (§8.3, the Θ(H²) footprint — NOT every region brick, which would be ~GBs). Returns `true` iff
    /// the resident region set CHANGED this frame (a crossing) — a diagnostic for the caller's bench logging.
    fn update(&mut self, queue: &wgpu::Queue, cam: [f32; 3]) -> bool {
        let t0 = std::time::Instant::now();
        let desired = self.desired_regions(cam);
        if desired == self.resident {
            return false; // no crossing — the common case (idle / sub-region camera motion)
        }
        let t_desired = t0.elapsed();

        // Decode IN the newly-covered regions (held for the occupancy + core enumeration; the source LRU also
        // caches them). A region absent from the directory is all-air — recorded resident so we don't re-attempt
        // it every frame, but it contributes no decoded handle (no occupancy/cores).
        let to_add: Vec<RegionKey> = desired.difference(&self.resident).copied().collect();
        let to_drop: Vec<RegionKey> = self.resident.difference(&desired).copied().collect();
        let n_add = to_add.len();
        let n_drop = to_drop.len();
        for &key in &to_add {
            let (ai, lod, rc) = key;
            if let Some(region) = self.source.placed_assets()[ai].source.decode_region_pub(lod, rc) {
                self.decoded.insert(key, region);
            }
        }
        for key in &to_drop {
            self.decoded.remove(key);
        }
        self.resident = desired;
        let t_decode = t0.elapsed();

        // §8.2/§8.3 — rebuild the OCCUPANCY whole, then re-derive the FULL surface+halo CORE set and sync the
        // core store to it. Recomputing the full set each crossing (NOT an incremental refcount delta) is the
        // COVERAGE FIX: the resident core set is, by construction, exactly the surface+halo of the CURRENT
        // occupancy, so the front end can never enter a brick whose neighbour core is absent (the stale-halo
        // wrong-normal / "black cube" bug, which the refcount delta let drift out of sync with the rebuilt
        // occupancy). It is Θ(resident bricks) per crossing — the same scan the occupancy rebuild already makes,
        // and the bench-allowed per-crossing transient; `sync_to_keys` decodes ONLY the newly-needed cores.
        let occ = self.rebuild_occupancy(queue);
        let t_occ = t0.elapsed();
        let (desired_cores, key_asset) = self.collect_surface_halo_keys(&occ);
        {
            let Self { core_store, source, decoded, .. } = &mut *self;
            core_store.sync_to_keys(queue, &desired_cores, cam, |world, lod| {
                let ai = *key_asset.get(&(world, lod))?;
                let asset = &source.placed_assets()[ai];
                let rc = asset.source.region_of_world(lod, world);
                let region = decoded.get(&(ai, lod, rc))?;
                asset.source.core_at_world(lod, world, region)
            });
        }
        let total = t0.elapsed();
        if total.as_millis() > 30 {
            bevy::log::debug!(
                "pager crossing: total={:.1}ms desired={:.1}ms decode(+{n_add}/-{n_drop}regs)={:.1}ms occ={:.1}ms cores={:.1}ms | {} regions, {} cores",
                total.as_secs_f64() * 1000.0,
                t_desired.as_secs_f64() * 1000.0,
                (t_decode - t_desired).as_secs_f64() * 1000.0,
                (t_occ - t_decode).as_secs_f64() * 1000.0,
                (total - t_occ).as_secs_f64() * 1000.0,
                self.resident.len(),
                self.core_store.resident_cores(),
            );
        }
        true
    }

    #[inline]
    fn occupancy(&self) -> &GpuResidencyBuffers {
        &self.occ_buffers
    }

    #[inline]
    fn core_buffers(&self) -> GpuBrickCoreBuffers {
        self.core_store.buffers()
    }

    #[inline]
    fn take_needs_rebind(&mut self) -> bool {
        std::mem::take(&mut self.needs_rebind)
    }

    #[inline]
    fn resident_region_count(&self) -> usize {
        self.resident.len()
    }

    #[inline]
    fn resident_core_count(&self) -> usize {
        self.core_store.resident_cores()
    }

    /// Fetch the SOURCE core for `(coord, lod)` — the exact 8³ core (voxel_index order) the GPU pack should have
    /// produced for this brick (F9 dump content-integrity check). `None` if the brick's region isn't decoded.
    fn debug_source_core(&self, coord: IVec3, lod: u32) -> Option<[u32; crate::voxel::brickmap::BRICK_VOXELS]> {
        for (ai, asset) in self.source.placed_assets().iter().enumerate() {
            let rc = asset.source.region_of_world(lod, coord);
            if let Some(region) = self.decoded.get(&(ai, lod, rc)) {
                if let Some(core) = asset.source.core_at_world(lod, coord, region) {
                    return Some(core);
                }
            }
        }
        None
    }

    #[inline]
    fn debug_core_in_store(&self, coord: IVec3, lod: u32) -> bool {
        self.core_store.debug_core_resident(coord, lod)
    }
}

/// The 26 face/edge/corner neighbour offsets (excluding the centre) — the halo neighbourhood whose cores the GPU
/// halo-fill reads for an entered surface brick.
const N26: [IVec3; 26] = {
    let mut a = [IVec3::ZERO; 26];
    let mut i = 0;
    let mut dz = -1;
    while dz <= 1 {
        let mut dy = -1;
        while dy <= 1 {
            let mut dx = -1;
            while dx <= 1 {
                if !(dx == 0 && dy == 0 && dz == 0) {
                    a[i] = IVec3::new(dx, dy, dz);
                    i += 1;
                }
                dx += 1;
            }
            dy += 1;
        }
        dz += 1;
    }
    a
};
