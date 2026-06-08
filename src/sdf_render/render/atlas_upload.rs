//! The chunk-table GPU mirror: extract the resident chunk directory + tile-run from the main-world
//! [`SdfAtlas`] (via [`chunk::LiveChunkTables::upload`]), then full-rebuild or delta-`write_buffer`
//! the two std430 storage buffers, and grow the atlas dist/mat textures when the bake needs more
//! rows. The decision/headroom policy lives on `LiveChunkTables`; this module only serializes +
//! uploads. Writes onto the shared [`SdfGpuAtlas`] (from [`super`]).

use super::super::chunk;
use super::*;

/// One entry in the chunk lookup buffer (20 bytes, std430). `key_*` = the absolute chunk key (see
/// `chunk`), independent of camera so CPU and GPU agree. `occ_*` = 64-bit occupancy mask (bit i ⇒
/// local brick i resident); `tile_run_base` indexes the packed tile-run buffer where this chunk's
/// `popcount(occ)` brick `atlas_base`s live.
///
/// Exists SOLELY as the std430 `ShaderType` for the chunk-lookup storage buffer's binding layout /
/// min-binding-size (see `init_*_pipeline`). The actual data flows as `chunk::ChunkLookup` and is
/// serialized by [`encode_lookup`]; this mirror is kept here, not on `chunk::ChunkLookup`, to
/// preserve `chunk.rs`'s render-free purity. Its fields MUST match `chunk::ChunkLookup` byte-for-byte.
///
/// Only used as the `storage_buffer_read_only::<GpuChunkLookup>` min-binding-size marker in the
/// removed surface bind-group layout; retained for the future cloud-raymarch atlas layout.
#[allow(dead_code)]
#[derive(ShaderType, Clone, Copy, Default)]
pub(super) struct GpuChunkLookup {
    key_hi: u32,
    key_lo: u32,
    occ_lo: u32,
    occ_hi: u32,
    cons_occ_lo: u32,
    cons_occ_hi: u32,
    tile_run_base: u32,
    probe_base: u32,
}

/// (20 bytes: distance tile origin + material tile origin, each `col_px | row_px<<16`, + packed
/// 4-entry palette + DDGI probe slot). Like [`GpuChunkLookup`], the data flows as `chunk::BrickTile`
/// (serialized by [`encode_tile`]); this mirror keeps the GPU derive out of the pure `chunk.rs`. Fields
/// MUST match `chunk::BrickTile`. Like [`GpuChunkLookup`], only a layout marker for the removed
/// surface bind group; retained for the future cloud-raymarch atlas layout.
#[allow(dead_code)]
#[derive(ShaderType, Clone, Copy, Default)]
pub(super) struct GpuBrickTile {
    atlas_base: u32,
    mat_atlas_base: u32,
    pal01: u32,
    pal23: u32,
    probe_slot: u32,
}

#[derive(Resource, Default)]
pub(super) struct ExtractedSdfAtlas {
    /// FULL-rebuild payload (`full_rebuild`): the entire chunk lookup directory
    /// (`chunk_data`, one row per directory slot) + the whole packed tile-run buffer
    /// (`tile_run_data`, capacity-sized, each chunk's 64-entry region at `slot*64`). Used the
    /// first frame, on a capacity grow, and on the empty-atlas sentinel. See `chunk`.
    chunk_data: Vec<chunk::ChunkLookup>,
    tile_run_data: Vec<chunk::BrickTile>,
    /// DELTA payload (`tables_dirty && !full_rebuild`): only the directory slots and tile-run
    /// regions (slot → 64-entry region) that changed this frame. The toroidal directory is fixed-
    /// position, so each is an in-place `write_buffer` — no row shift, no sentinel tail.
    chunk_row_updates: Vec<(u32, chunk::ChunkLookup)>,
    tile_run_updates: Vec<(u32, Vec<chunk::BrickTile>)>,
    /// Directory length (= `R³ × lod_count`); the GPU direct-indexes it (no logical-count bound).
    new_chunk_len: u32,
    /// Buffer capacities (rows / tile-run entries) this frame's table needs. `prepare` grows the
    /// buffers (with headroom) when these exceed the current allocation.
    chunk_cap_needed: u32,
    tile_cap_needed: u32,
    /// Independent rebuild flags for the two buffers (decoupled so a tile-run capacity grow doesn't
    /// drag a full re-upload of the fixed-size, tens-of-MB directory):
    /// - `dir_full` ⇒ recreate the chunk-lookup buffer from `chunk_data`; else `write_buffer` the
    ///   `chunk_row_updates` deltas in place.
    /// - `tile_full` ⇒ recreate the tile-run buffer from `tile_run_data` (to `tile_cap_needed`); else
    ///   `write_buffer` the `tile_run_updates` regions in place.
    ///
    /// Full upload = both true; `TileGrow` = `tile_full` only; Delta = both false.
    dir_full: bool,
    tile_full: bool,
    /// Whether the chunk lookup / tile-run buffers changed at all this frame. False on a
    /// texel-only re-bake — the lookup buffers are reused as-is.
    tables_dirty: bool,
    /// Tile-rows the DISTANCE and MATERIAL atlases must span this frame (= each allocator's
    /// `high_water` rows). `prepare` grows the paged pools ([`AtlasPages::ensure`]) to cover them —
    /// one new page per block, NO copy. The pools only grow (tile origins are stable via the
    /// allocators' free-lists), so these are monotonic until a full eviction resets them. The
    /// material atlas sizes to the MULTI-material brick count only (its own dense allocator).
    dist_rows_needed: u32,
    mat_rows_needed: u32,
    /// Gradient pool rows: equals `dist_rows_needed` when the gradient feature is enabled (it's
    /// dense — one tile per brick, sharing the distance tile index), else 0 (pool stays empty).
    grad_rows_needed: u32,
    dirty: bool,
}

/// Render-world memo of the last atlas generation uploaded, so `extract_sdf_atlas`
/// only flags `dirty` (and `prepare_sdf_atlas_gpu` only re-uploads) when the
/// main-world bake actually changed something. Without this the atlas was rebuilt
/// every frame.
#[derive(Resource, Default)]
pub(super) struct LastAtlasGen(u64);


pub(super) fn extract_sdf_atlas(
    atlas: Extract<Res<SdfAtlas>>,
    // Scene-switch counter (`SdfAtlas::reset` bumped it). When it changes, the previous scene's GPU
    // chunk directory must be FULLY rebuilt — a delta would leave the old scene's rows in `chunk_buf`
    // (`find_chunk` still hits them → ghost geometry). Reset the capacity memo so the next upload is a
    // Full rebuild, not a delta against the stale buffer. Read from the MAIN world (no extract-order race).
    reset_res: Extract<Res<crate::sdf_render::ProbeReset>>,
    mut last_reset: Local<u32>,
    mut last_gen: ResMut<LastAtlasGen>,
    mut chunk_cap: ResMut<super::chunk_tables::ChunkBufCapacity>,
    mut commands: Commands,
) {
    // Nothing changed since the last upload — skip the rebuild entirely so idle
    // frames cost no extract/prepare work. `prepare_sdf_atlas_gpu` keeps last
    // frame's GPU resources because the inserted resource has `dirty = false`.
    if atlas.generation == last_gen.0 {
        commands.insert_resource(ExtractedSdfAtlas::default()); // dirty = false
        return;
    }
    last_gen.0 = atlas.generation;
    // Runs only on a topology change (a move) — instrumented so its CPU `upload()` cost (delta vs the
    // tile-run dense rebuild) is named in the trace rather than an anonymous render-schedule gap.
    let _span = info_span!("extract_sdf_atlas").entered();

    // On a scene switch, force the next upload to be a FULL rebuild (capacity memo → 0). Otherwise the
    // new scene baked over the same frame does a delta, leaving the old scene's chunk rows live on the GPU.
    if reset_res.0 != *last_reset {
        *last_reset = reset_res.0;
        chunk_cap.chunk_rows = 0;
        chunk_cap.tile_slots = 0;
        chunk_cap.probe_slots = 0;
    }

    let live = &atlas.live_chunks;
    // DDGI probe buffer is sized by the FINEST-RESIDENT probe high-water (`finest_chunks · CHUNK_VOLUME`)
    // — one compact probe block per finest-resident chunk. Bounded by the clipmap WINDOW, not the
    // all-LOD atlas tile union (which carried an ~lod_count× redundant copy of every near surface). This
    // is what makes the probe buffer scale with the clipmap instead of the scene's absolute size.
    chunk_cap.probe_slots = live.probe_high_water();
    let num_bricks = atlas.bricks.len() as u32;
    if num_bricks == 0 {
        // Fully evicted (roamed into empty space). Signal a full rebuild with EMPTY chunk data
        // so `prepare_sdf_atlas_gpu` replaces the lookup buffer with a miss-only sentinel. The
        // shader bounds its search by `arrayLength(&chunk_buf)`, so leaving the old buffer bound
        // would search stale entries and render ghost geometry. Reset capacity so the next
        // re-entry triggers a fresh full rebuild rather than a delta against a dropped buffer.
        chunk_cap.chunk_rows = 0;
        chunk_cap.tile_slots = 0;
        chunk_cap.probe_slots = 0;
        commands.insert_resource(ExtractedSdfAtlas {
            tables_dirty: true,
            dir_full: true,
            tile_full: true,
            dirty: true,
            ..Default::default()
        });
        return;
    }

    // Tile origins come from the stable allocators (their high-water marks), NOT brick iteration
    // order — so a re-baked brick keeps its sub-rect across frames. `prepare` grows each paged pool
    // to cover this many tile-rows (one page per block, no copy); the pools only grow. The material
    // atlas tracks its OWN (multi-material-only) allocator, so it stays small when most bricks are
    // single-material.
    let dist_rows_needed = atlas.tiles.high_water().div_ceil(ATLAS_TILES_PER_ROW).max(1);
    let mat_rows_needed = atlas.mat_tiles.high_water().div_ceil(ATLAS_TILES_PER_ROW).max(1);
    // Gradient is dense (shares the distance tile index), so it needs the same rows as distance —
    // but only when enabled; 0 keeps the pool unallocated (the reclamation-style VRAM gate).
    let grad_rows_needed = if atlas.bake_gradient { dist_rows_needed } else { 0 };

    let mut extracted = ExtractedSdfAtlas {
        dist_rows_needed,
        mat_rows_needed,
        grad_rows_needed,
        dirty: true,
        ..Default::default()
    };

    // Full-rebuild-vs-delta + the tile-run headroom policy live on `LiveChunkTables::upload` (the
    // SINGLE source of truth, mirrored by the churn + recenter-lifecycle differential tests). Extract
    // passes its render-world buffer capacities in and maps the returned NATIVE records onto the GPU
    // mirror; the directory is fixed-size so a Full only happens on first upload / a tile-run grow.
    match live.upload(chunk_cap.chunk_rows, chunk_cap.tile_slots) {
        chunk::ChunkUpload::Full { rows, tile_run, cap_rows, cap_slots } => {
            chunk_cap.chunk_rows = cap_rows;
            chunk_cap.tile_slots = cap_slots;
            extracted.chunk_data = rows;
            extracted.tile_run_data = tile_run;
            extracted.new_chunk_len = cap_rows;
            extracted.chunk_cap_needed = cap_rows;
            extracted.tile_cap_needed = cap_slots;
            extracted.dir_full = true;
            extracted.tile_full = true;
            extracted.tables_dirty = true;
        }
        chunk::ChunkUpload::TileGrow { row_updates, tile_run, cap_slots } => {
            // ONLY the tile-run grew — rebuild it, but the fixed-size directory just deltas in place
            // (no wholesale directory re-upload, the former `sdf_tables_full_rebuild` hitch).
            chunk_cap.tile_slots = cap_slots;
            extracted.chunk_row_updates = row_updates;
            extracted.tile_run_data = tile_run;
            extracted.tile_cap_needed = cap_slots;
            extracted.tile_full = true;
            extracted.tables_dirty = true;
            extracted.new_chunk_len = live.row_count();
        }
        chunk::ChunkUpload::Delta { row_updates, region_updates } => {
            // Fixed-position directory → every dirty entry is an in-place index→value write.
            extracted.chunk_row_updates = row_updates;
            extracted.tile_run_updates =
                region_updates.into_iter().map(|(s, reg)| (s, reg.to_vec())).collect();
            extracted.tables_dirty =
                !extracted.chunk_row_updates.is_empty() || !extracted.tile_run_updates.is_empty();
            extracted.new_chunk_len = live.row_count();
        }
    }

    commands.insert_resource(extracted);
}

pub(super) fn prepare_sdf_atlas_gpu(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    extracted: Option<Res<ExtractedSdfAtlas>>,
    // Bumped on a scene switch (`SdfAtlas::reset`). The CPU tables + GPU lookup reset, but the brick
    // TEXEL pages otherwise persist in VRAM — so a reused tile could show the previous scene's texels
    // (esp. if the new scene's bake hash-skips it). Reallocate the pages fresh (zeroed) on a switch.
    reset_res: Option<Res<crate::sdf_render::ProbeReset>>,
    mut last_reset: Local<u32>,
    mut gpu_atlas: ResMut<SdfGpuAtlas>,
) {
    let reset_id = reset_res.map(|r| r.0).unwrap_or(0);
    if reset_id != *last_reset {
        *last_reset = reset_id;
        // Fresh, zeroed atlas pages — the new scene's bake re-grows + writes them; no stale carry-over.
        gpu_atlas.pages = Some(super::atlas_pages::AtlasPages::new(&device));
    }

    let Some(extracted) = extracted else { return };
    if !extracted.dirty {
        return;
    }

    if extracted.tables_dirty {
        // Directory + tile-run are decoupled. Rebuild the directory only on a genuine directory grow
        // (first upload / ring-size change) — it's the fixed-size, tens-of-MB-at-large-rings buffer,
        // so its rebuild is tagged as the heavy upload to watch; otherwise delta its dirty rows in
        // place. A tile-run grow (Full OR TileGrow) rebuilds just the smaller tile-run.
        if extracted.dir_full {
            let _span = info_span!("sdf_directory_rebuild").entered();
            gpu_atlas.tables.rebuild_directory(
                &device,
                &extracted.chunk_data,
                extracted.chunk_cap_needed,
                extracted.new_chunk_len,
            );
        } else {
            gpu_atlas.tables.directory_delta(&queue, &extracted.chunk_row_updates);
        }
        if extracted.tile_full {
            // The tile-run grow rebuilds the WHOLE tile-run (O(all resident bricks)). It was previously
            // UN-instrumented — the ~200 ms hitch on editing a large scene (a new resident chunk grew the
            // tile-run past its slack) showed only as an anonymous gap in the render schedule. Span +
            // log it so the cost is visible; the doubling headroom in `LiveChunkTables::upload` keeps it
            // rare (a settled scene's edit churn stays within slack → cheap in-place deltas instead).
            let _span = info_span!("sdf_tile_run_rebuild").entered();
            debug!(
                "sdf tile-run rebuild: {} slots ({:.1} MB) — a resident-chunk grow outran the buffer slack",
                extracted.tile_cap_needed,
                extracted.tile_cap_needed as f64 * 20.0 / (1 << 20) as f64,
            );
            gpu_atlas.tables.rebuild_tile_run(
                &device,
                &extracted.tile_run_data,
                extracted.tile_cap_needed,
            );
        } else {
            gpu_atlas.tables.tile_run_delta(&queue, &extracted.tile_run_updates);
        }
    }

    // Grow the paged atlas pool to cover this frame's tile-rows. Adds whole PAGES as needed and
    // copies NOTHING — existing pages (and the bricks in them) stay put. This is the fix for the
    // former single-texture realloc, which recreated + full-copied the entire atlas on every
    // row-boundary crossing (old+new alive ≈ 2× the resident bricks → ~6.8 GB VRAM spike, and
    // O(N²) copy over a fill — the `sdf_atlas_realloc` trace hotspot). The per-frame bind group
    // (`atlas_bind_group_1`) re-reads the page list, so a new page is picked up automatically.
    if let Some(pages) = gpu_atlas.pages.as_mut() {
        let _span = info_span!("sdf_atlas_ensure_pages").entered();
        let before = pages.page_count();
        if pages.ensure(
            &device,
            extracted.dist_rows_needed,
            extracted.mat_rows_needed,
            extracted.grad_rows_needed,
        ) {
            debug!("SDF atlas grew {before} -> {} page(s) (no copy)", pages.page_count());
        }
    }
}
