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
    texture_width: u32,
    texture_height: u32,
    /// Grow the atlas texture taller this frame: `prepare` recreates the dist+mat textures at
    /// the new height and `copy_texture_to_texture`s the old content in (the GPU owns the
    /// texels — there is no CPU upload), then the bake node fills the genuinely-new tiles. When
    /// false, `prepare` keeps the existing textures and the bake node patches tiles in place.
    realloc: bool,
    dirty: bool,
}

/// Render-world memo of the last atlas generation uploaded, so `extract_sdf_atlas`
/// only flags `dirty` (and `prepare_sdf_atlas_gpu` only re-uploads) when the
/// main-world bake actually changed something. Without this the atlas was rebuilt
/// every frame.
#[derive(Resource, Default)]
pub(super) struct LastAtlasGen(u64);

/// Render-world record of how many tile rows the persistent atlas texture currently
/// spans. `extract_sdf_atlas` reads it to decide grow-vs-partial-upload; the texture
/// only grows (never shrinks except on a full rebuild), so a tile origin assigned
/// once stays valid until the next full bake.
#[derive(Resource, Default)]
pub(super) struct AtlasCapacity {
    rows: u32,
}

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
    mut capacity: ResMut<AtlasCapacity>,
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

    let edge = BRICK_EDGE as u32;
    let tile_width = edge * edge; // 64
    let texture_width = ATLAS_TILES_PER_ROW * tile_width;

    // Tile origins come from the stable allocator (its high-water mark), NOT brick
    // iteration order — so a re-baked brick keeps its sub-rect across frames.
    let required_rows = atlas.tiles.high_water().div_ceil(ATLAS_TILES_PER_ROW).max(1);
    let texture_height = required_rows * edge;

    // Realloc when the atlas TEXTURE must grow taller (the GPU bake never shrinks it). This is now
    // INDEPENDENT of the chunk table: a brick's `atlas_base` is derived from its stable tile index
    // and is unaffected by the texture's height, so a texture grow never forces a table rebuild.
    let realloc = required_rows > capacity.rows;
    if realloc {
        capacity.rows = required_rows;
    }

    let mut extracted = ExtractedSdfAtlas {
        texture_width,
        texture_height,
        realloc,
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
            upload_tables_full(&device, &mut gpu_atlas, &extracted);
        } else {
            upload_tables_delta(&queue, &gpu_atlas, &extracted);
        }
    }

    if extracted.realloc {
        // Grow the atlas taller. The GPU owns the texels (the CPU has only palette-only
        // placeholders), so create EMPTY textures and copy any prior content into the taller
        // replacement; the bake node fills the genuinely-new tiles this same frame. On the
        // very first bake there's no prior texture — just the empty allocation, no copy. All
        // atlas textures carry COPY_SRC (for this grow copy) + COPY_DST (the bake node's
        // per-tile copy_buffer_to_texture).
        let usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::COPY_SRC;
        let size = Extent3d {
            width: extracted.texture_width,
            height: extracted.texture_height,
            depth_or_array_layers: 1,
        };
        let dist_tex = device.create_texture(&TextureDescriptor {
            label: Some("sdf_dist_atlas"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R16Snorm,
            usage,
            view_formats: &[],
        });
        let mat_tex = device.create_texture(&TextureDescriptor {
            label: Some("sdf_mat_atlas"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Snorm,
            usage,
            view_formats: &[],
        });
        // Copy prior content (full width, old height) into the new taller textures, if any.
        if let (Some(old_dist), Some(old_mat)) = (&gpu_atlas.dist_tex, &gpu_atlas.mat_tex) {
            let old_h = old_dist.height().min(extracted.texture_height);
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("sdf_atlas_grow_copy"),
            });
            let copy_extent = Extent3d {
                width: extracted.texture_width,
                height: old_h,
                depth_or_array_layers: 1,
            };
            for (src, dst) in [(old_dist, &dist_tex), (old_mat, &mat_tex)] {
                encoder.copy_texture_to_texture(
                    TexelCopyTextureInfo {
                        texture: src,
                        mip_level: 0,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    TexelCopyTextureInfo {
                        texture: dst,
                        mip_level: 0,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    copy_extent,
                );
            }
            queue.submit([encoder.finish()]);
        }

        gpu_atlas.dist_view = Some(dist_tex.create_view(&TextureViewDescriptor::default()));
        gpu_atlas.mat_view = Some(mat_tex.create_view(&TextureViewDescriptor::default()));
        gpu_atlas.dist_tex = Some(dist_tex);
        gpu_atlas.mat_tex = Some(mat_tex);
        if gpu_atlas.sampler.is_none() {
            gpu_atlas.sampler = Some(device.create_sampler(&SamplerDescriptor {
                label: Some("sdf_atlas_sampler"),
                mag_filter: FilterMode::Nearest,
                min_filter: FilterMode::Nearest,
                mipmap_filter: FilterMode::Nearest,
                ..default()
            }));
        }
    }
    // Non-grow frames: the existing textures are kept; the bake node patches changed tiles in
    // place via copy_buffer_to_texture. Nothing to upload here.
}
