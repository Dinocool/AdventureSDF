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
#[derive(ShaderType, Clone, Copy, Default)]
pub(super) struct GpuChunkLookup {
    key_hi: u32,
    key_lo: u32,
    occ_lo: u32,
    occ_hi: u32,
    tile_run_base: u32,
}

/// std430 `ShaderType` for the tile-run storage buffer's binding layout / min-binding-size only
/// (12 bytes: atlas tile origin `col_px | row_px<<16` + packed 4-entry palette). Like
/// [`GpuChunkLookup`], the data flows as `chunk::BrickTile` (serialized by [`encode_tile`]); this
/// mirror keeps the GPU derive out of the pure `chunk.rs`. Fields MUST match `chunk::BrickTile`.
#[derive(ShaderType, Clone, Copy, Default)]
pub(super) struct GpuBrickTile {
    atlas_base: u32,
    pal01: u32,
    pal23: u32,
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
    /// True ⇒ upload `chunk_data`/`tile_run_data` wholesale; false ⇒ apply the delta updates.
    full_rebuild: bool,
    /// Whether the chunk lookup / tile-run buffers changed at all this frame. False on a
    /// texel-only re-bake — the lookup buffers are reused as-is.
    tables_dirty: bool,
    /// Tile-rows the atlas must span this frame (= `high_water` rows). `prepare` grows the paged
    /// pool ([`AtlasPages::ensure`]) to cover it — one new page per block, NO copy. The pool only
    /// grows (tile origins are stable via the allocator's free-list), so this is monotonic until a
    /// full eviction resets the allocator.
    rows_needed: u32,
    dirty: bool,
}

/// Render-world memo of the last atlas generation uploaded, so `extract_sdf_atlas`
/// only flags `dirty` (and `prepare_sdf_atlas_gpu` only re-uploads) when the
/// main-world bake actually changed something. Without this the atlas was rebuilt
/// every frame.
#[derive(Resource, Default)]
pub(super) struct LastAtlasGen(u64);


/// Render-world record of the allocated chunk-lookup + tile-run buffer capacities (in rows /
/// tile-run entries), so `prepare_sdf_atlas_gpu` knows when an incremental delta needs the
/// buffer grown (recreate larger + full re-upload) versus a plain in-place `write_buffer`.
/// Both buffers are over-sized with headroom on a rebuild so most frames stay in the cheap
/// delta path.
#[derive(Resource, Default)]
pub(super) struct ChunkBufCapacity {
    pub(super) chunk_rows: u32,
    pub(super) tile_slots: u32,
}

pub(super) fn extract_sdf_atlas(
    atlas: Extract<Res<SdfAtlas>>,
    mut last_gen: ResMut<LastAtlasGen>,
    mut chunk_cap: ResMut<ChunkBufCapacity>,
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

    let live = &atlas.live_chunks;
    let num_bricks = atlas.bricks.len() as u32;
    if num_bricks == 0 {
        // Fully evicted (roamed into empty space). Signal a full rebuild with EMPTY chunk data
        // so `prepare_sdf_atlas_gpu` replaces the lookup buffer with a miss-only sentinel. The
        // shader bounds its search by `arrayLength(&chunk_buf)`, so leaving the old buffer bound
        // would search stale entries and render ghost geometry. Reset capacity so the next
        // re-entry triggers a fresh full rebuild rather than a delta against a dropped buffer.
        chunk_cap.chunk_rows = 0;
        chunk_cap.tile_slots = 0;
        commands.insert_resource(ExtractedSdfAtlas {
            tables_dirty: true,
            full_rebuild: true,
            dirty: true,
            ..Default::default()
        });
        return;
    }

    // Tile origins come from the stable allocator (its high-water mark), NOT brick iteration order
    // — so a re-baked brick keeps its sub-rect across frames. `prepare` grows the paged pool to
    // cover this many tile-rows (one page per block, no copy); the pool only grows.
    let required_rows = atlas.tiles.high_water().div_ceil(ATLAS_TILES_PER_ROW).max(1);

    let mut extracted = ExtractedSdfAtlas {
        rows_needed: required_rows,
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
            extracted.full_rebuild = true;
            extracted.tables_dirty = true;
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

// --- chunk-table upload (full rebuild + incremental delta) ---

/// 20-byte std430 encoding of one chunk lookup row.
fn encode_lookup(c: &chunk::ChunkLookup, out: &mut Vec<u8>) {
    out.extend_from_slice(&c.key_hi.to_le_bytes());
    out.extend_from_slice(&c.key_lo.to_le_bytes());
    out.extend_from_slice(&c.occ_lo.to_le_bytes());
    out.extend_from_slice(&c.occ_hi.to_le_bytes());
    out.extend_from_slice(&c.tile_run_base.to_le_bytes());
}

/// 12-byte std430 encoding of one tile-run brick record.
fn encode_tile(b: &chunk::BrickTile, out: &mut Vec<u8>) {
    out.extend_from_slice(&b.atlas_base.to_le_bytes());
    out.extend_from_slice(&b.pal01.to_le_bytes());
    out.extend_from_slice(&b.pal23.to_le_bytes());
}

/// The 20-byte `(u32::MAX, u32::MAX, 0, 0, 0)` chunk-lookup sentinel. Its key tag never matches a
/// real chunk key, so a fixed directory slot that no live chunk occupies resolves to a miss.
fn sentinel_row_bytes() -> [u8; 20] {
    let mut b = [0u8; 20];
    b[0..4].copy_from_slice(&u32::MAX.to_le_bytes());
    b[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    b
}

/// Full (re)allocation + upload of both chunk-table buffers, sized to CAPACITY (with headroom)
/// so later frames can `write_buffer` deltas in place. The chunk-lookup buffer is filled with
/// `new_chunk_len` live rows followed by sentinel rows to capacity; the tile-run buffer is the
/// capacity-sized `tile_run_data` (each live slot's region at `slot*64`, gaps zero). Used on the
/// first upload, a capacity grow, and the empty-atlas case (zero live rows → all sentinel).
fn upload_tables_full(
    device: &RenderDevice,
    gpu_atlas: &mut SdfGpuAtlas,
    extracted: &ExtractedSdfAtlas,
) {
    // Chunk lookup buffer: live rows then sentinel tail to capacity. Capacity is always ≥1 so the
    // storage buffer is never zero-sized (an empty atlas yields a single sentinel — the prior
    // dedicated empty path, now folded in here).
    let cap_rows = extracted.chunk_cap_needed.max(1);
    let live = extracted.new_chunk_len.min(extracted.chunk_data.len() as u32);
    let mut chunk_bytes = Vec::with_capacity(cap_rows as usize * 20);
    for c in extracted.chunk_data.iter().take(live as usize) {
        encode_lookup(c, &mut chunk_bytes);
    }
    let sentinel = sentinel_row_bytes();
    for _ in live..cap_rows {
        chunk_bytes.extend_from_slice(&sentinel);
    }
    gpu_atlas.lookup_buffer = Some(device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_chunk_lookup_buffer"),
        contents: &chunk_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    }));

    // Tile-run buffer: capacity-sized (extract already laid out `tile_run_data` to the slot
    // high-water; pad to `tile_cap_needed` so deltas into freshly-grown slots have room).
    let cap_slots = extracted.tile_cap_needed.max(chunk::TILE_RUN_SLOT) as usize;
    let mut tile_bytes = Vec::with_capacity(cap_slots * 12);
    for b in &extracted.tile_run_data {
        encode_tile(b, &mut tile_bytes);
    }
    tile_bytes.resize(cap_slots * 12, 0);
    gpu_atlas.chunk_tile_buffer = Some(device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_chunk_tile_buffer"),
        contents: &tile_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    }));
}

/// Incremental upload: `write_buffer` only the chunk rows + tile-run regions that changed this
/// frame, plus sentinel-blank the rows a removed chunk vacated. The buffers keep their (capacity)
/// allocation — only the changed byte ranges are touched, so a coarse-LOD snap pages the handful
/// of dirty chunks instead of recreating the whole ~1 MB table.
fn upload_tables_delta(
    queue: &RenderQueue,
    gpu_atlas: &SdfGpuAtlas,
    extracted: &ExtractedSdfAtlas,
) {
    let (Some(lookup), Some(tiles)) = (&gpu_atlas.lookup_buffer, &gpu_atlas.chunk_tile_buffer)
    else {
        return; // no buffers yet (shouldn't happen — first frame is a full rebuild)
    };

    // Changed chunk-lookup rows (20 B each, at row*20). A structural change marks a contiguous
    // suffix `[R..end)` dirty (every row at/after an insert/remove shifts), so coalesce consecutive
    // rows into one `write_buffer` to avoid a long burst of 20-byte writes on a snap frame.
    let mut run_start: Option<u32> = None;
    let mut run_bytes: Vec<u8> = Vec::new();
    let flush = |start: u32, bytes: &[u8]| {
        if !bytes.is_empty() {
            queue.write_buffer(lookup, (start as u64) * 20, bytes);
        }
    };
    for (row, c) in &extracted.chunk_row_updates {
        match run_start {
            Some(s) if *row == s + (run_bytes.len() as u32 / 20) => {}
            _ => {
                if let Some(s) = run_start {
                    flush(s, &run_bytes);
                }
                run_start = Some(*row);
                run_bytes.clear();
            }
        }
        encode_lookup(c, &mut run_bytes);
    }
    if let Some(s) = run_start {
        flush(s, &run_bytes);
    }
    // No sentinel tail: the directory is fixed-size and an emptied chunk's slot was already reset to
    // the sentinel tag in `clear_brick` (it shows up as a normal dirty-row write above).

    // Changed tile-run regions (64 entries × 12 B = 768 B each, at slot*64*12).
    let mut region_bytes = Vec::with_capacity(chunk::TILE_RUN_SLOT as usize * 12);
    for (slot, region) in &extracted.tile_run_updates {
        region_bytes.clear();
        for b in region {
            encode_tile(b, &mut region_bytes);
        }
        let base = (*slot as u64) * chunk::TILE_RUN_SLOT as u64 * 12;
        queue.write_buffer(tiles, base, &region_bytes);
    }
}

pub(super) fn prepare_sdf_atlas_gpu(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    extracted: Option<Res<ExtractedSdfAtlas>>,
    mut gpu_atlas: ResMut<SdfGpuAtlas>,
) {
    let Some(extracted) = extracted else { return };
    if !extracted.dirty {
        return;
    }

    if extracted.tables_dirty {
        if extracted.full_rebuild {
            // Full directory rebuild: recreate + re-upload the whole dense per-LOD chunk buffer
            // (R³·lod_count entries). Tagged so its cost/frequency is visible in a chrome trace —
            // at large ring_bricks this buffer is big and a Full rebuild is a heavy upload.
            let _span = info_span!("sdf_tables_full_rebuild").entered();
            upload_tables_full(&device, &mut gpu_atlas, &extracted);
        } else {
            upload_tables_delta(&queue, &gpu_atlas, &extracted);
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
        if pages.ensure(&device, extracted.rows_needed) {
            debug!("SDF atlas grew {before} -> {} page(s) (no copy)", pages.page_count());
        }
    }
}
