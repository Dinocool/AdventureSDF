//! The chunk-directory + tile-run GPU storage buffers (bind group 1, bindings 2 & 11) and the
//! application of an upload decided by `chunk::LiveChunkTables::upload` (the SSOT).
//!
//! Both buffers are the same shape — a growable std430 buffer of fixed-stride records, recreated on
//! a capacity grow and `write_buffer`-patched for deltas — so that lifecycle lives in the shared
//! [`PackedBuf`]; only the per-record encoding + the capacity tail-pad differ (`ChunkLookup` rows
//! padded with a sentinel vs `BrickTile` records padded with zeros).
//!
//! The two buffers are decoupled: a tile-run capacity grow ([`ChunkUpload::TileGrow`]) rebuilds ONLY
//! the tile-run and deltas the (fixed-size, tens-of-MB at large rings) directory in place — it no
//! longer drags a full directory re-upload (the former `sdf_tables_full_rebuild` hitch).

use super::*;
use crate::sdf_render::chunk;

/// Render-world memo of the allocated chunk-lookup + tile-run buffer capacities (rows / tile-run
/// entries), so `extract_sdf_atlas` can ask `LiveChunkTables::upload` whether this frame fits the
/// current buffers or needs a (per-buffer) grow. Both grow only (a tile origin assigned once stays
/// valid) except a full eviction, which resets them to 0.
#[derive(Resource, Default)]
pub(super) struct ChunkBufCapacity {
    pub(super) chunk_rows: u32,
    pub(super) tile_slots: u32,
    /// Probe-slot high-water (brick-units) for the finest-resident probe set — sizes the DDGI probe
    /// irradiance buffer (`probe_slots × subdiv³ × PROBE_OCT_TEXELS`), bounded ≪ `tile_slots`.
    pub(super) probe_slots: u32,
}

/// A growable std430 storage buffer of fixed-stride records. Owns the GPU buffer lifecycle shared by
/// the directory and the tile-run: rebuild (recreate sized to capacity, live records then a repeated
/// pad record) and `write_at` (in-place `write_buffer` of one record range).
#[derive(Default)]
struct PackedBuf {
    buf: Option<Buffer>,
    stride: u64,
    label: &'static str,
}

impl PackedBuf {
    /// A 1-record zeroed dummy so the bind group is valid before the first real upload.
    fn new_dummy(device: &RenderDevice, label: &'static str, stride: u64) -> Self {
        let mut me = Self { buf: None, stride, label };
        me.rebuild(device, &[0u8], 1, &[0]);
        me
    }

    /// Recreate the buffer sized to `cap_records`: `live_bytes` (already-encoded records) then the
    /// `pad` record (`stride` bytes) repeated to capacity. Never zero-sized (`cap_records >= 1`).
    fn rebuild(&mut self, device: &RenderDevice, live_bytes: &[u8], cap_records: usize, pad: &[u8]) {
        let total = cap_records.max(1) * self.stride.max(1) as usize;
        let mut bytes = Vec::with_capacity(total);
        bytes.extend_from_slice(live_bytes);
        while bytes.len() < total {
            bytes.extend_from_slice(pad);
        }
        bytes.truncate(total);
        self.buf = Some(device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some(self.label),
            contents: &bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        }));
    }

    /// In-place `write_buffer` of `bytes` (a contiguous run of records) starting at `record_index`.
    fn write_at(&self, queue: &RenderQueue, record_index: u64, bytes: &[u8]) {
        if let Some(buf) = &self.buf {
            queue.write_buffer(buf, record_index * self.stride, bytes);
        }
    }

    fn buffer(&self) -> &Buffer {
        self.buf.as_ref().expect("table buffer initialized in init_sdf_pipeline")
    }
}

/// The two chunk-table buffers (directory lookup + packed tile-runs), with the per-buffer encoding
/// policy on top of the shared [`PackedBuf`] lifecycle.
#[derive(Default)]
pub(super) struct ChunkTableBuffers {
    directory: PackedBuf, // binding 2: dense per-LOD directory, 24-byte ChunkLookup rows
    tiles: PackedBuf,     // binding 11: packed tile-runs, 16-byte BrickTile records
}

impl ChunkTableBuffers {
    pub(super) fn new(device: &RenderDevice) -> Self {
        Self {
            directory: PackedBuf::new_dummy(device, "sdf_chunk_lookup_buffer", 24),
            tiles: PackedBuf::new_dummy(device, "sdf_chunk_tile_buffer", 16),
        }
    }

    pub(super) fn lookup_buffer(&self) -> &Buffer {
        self.directory.buffer()
    }
    pub(super) fn tile_buffer(&self) -> &Buffer {
        self.tiles.buffer()
    }

    /// Rebuild the directory: `live_len` live rows then a sentinel tail to `cap_rows`. Used on the
    /// first upload and a directory grow (ring-size change) — NOT on a tile-run grow.
    pub(super) fn rebuild_directory(
        &mut self,
        device: &RenderDevice,
        rows: &[chunk::ChunkLookup],
        cap_rows: u32,
        live_len: u32,
    ) {
        let live = live_len.min(rows.len() as u32) as usize;
        let mut bytes = Vec::with_capacity(live * 24);
        for c in &rows[..live] {
            encode_lookup(c, &mut bytes);
        }
        self.directory.rebuild(device, &bytes, cap_rows.max(1) as usize, &sentinel_row_bytes());
    }

    /// Rebuild the tile-run buffer: the live regions (already laid out at `slot*TILE_RUN_SLOT`) then
    /// a zero tail to `cap_slots` entries.
    pub(super) fn rebuild_tile_run(
        &mut self,
        device: &RenderDevice,
        tile_run: &[chunk::BrickTile],
        cap_slots: u32,
    ) {
        let mut bytes = Vec::with_capacity(tile_run.len() * 16);
        for b in tile_run {
            encode_tile(b, &mut bytes);
        }
        let cap = cap_slots.max(chunk::TILE_RUN_SLOT) as usize;
        self.tiles.rebuild(device, &bytes, cap, &[0u8; 16]);
    }

    /// In-place directory row deltas. A structural change marks a contiguous suffix dirty, so
    /// coalesce consecutive rows into one `write_buffer` (vs a burst of 24-byte writes on a snap).
    pub(super) fn directory_delta(
        &self,
        queue: &RenderQueue,
        row_updates: &[(u32, chunk::ChunkLookup)],
    ) {
        let mut run_start: Option<u32> = None;
        let mut run_bytes: Vec<u8> = Vec::new();
        for (row, c) in row_updates {
            match run_start {
                Some(s) if *row == s + (run_bytes.len() as u32 / 24) => {}
                _ => {
                    if let Some(s) = run_start {
                        self.directory.write_at(queue, s as u64, &run_bytes);
                    }
                    run_start = Some(*row);
                    run_bytes.clear();
                }
            }
            encode_lookup(c, &mut run_bytes);
        }
        if let Some(s) = run_start {
            self.directory.write_at(queue, s as u64, &run_bytes);
        }
    }

    /// In-place tile-run region deltas (each a slot's dense `TILE_RUN_SLOT` entries at its base).
    pub(super) fn tile_run_delta(
        &self,
        queue: &RenderQueue,
        region_updates: &[(u32, Vec<chunk::BrickTile>)],
    ) {
        let mut bytes = Vec::with_capacity(chunk::TILE_RUN_SLOT as usize * 16);
        for (slot, region) in region_updates {
            bytes.clear();
            for b in region {
                encode_tile(b, &mut bytes);
            }
            self.tiles.write_at(queue, *slot as u64 * chunk::TILE_RUN_SLOT as u64, &bytes);
        }
    }
}

/// 24-byte std430 encoding of one chunk lookup row.
fn encode_lookup(c: &chunk::ChunkLookup, out: &mut Vec<u8>) {
    out.extend_from_slice(&c.key_hi.to_le_bytes());
    out.extend_from_slice(&c.key_lo.to_le_bytes());
    out.extend_from_slice(&c.occ_lo.to_le_bytes());
    out.extend_from_slice(&c.occ_hi.to_le_bytes());
    out.extend_from_slice(&c.tile_run_base.to_le_bytes());
    out.extend_from_slice(&c.probe_base.to_le_bytes());
}

/// 16-byte std430 encoding of one tile-run brick record (atlas origin + palette + DDGI probe slot).
fn encode_tile(b: &chunk::BrickTile, out: &mut Vec<u8>) {
    out.extend_from_slice(&b.atlas_base.to_le_bytes());
    out.extend_from_slice(&b.pal01.to_le_bytes());
    out.extend_from_slice(&b.pal23.to_le_bytes());
    out.extend_from_slice(&b.probe_slot.to_le_bytes());
}

/// The 24-byte chunk-lookup sentinel `(u32::MAX, u32::MAX, 0, 0, 0, u32::MAX)`. Its key tag never
/// matches a real chunk key, so a fixed directory slot that no live chunk occupies resolves to a miss;
/// the trailing `probe_base = u32::MAX` mirrors `sentinel_lookup` (no probes — never read on a miss).
fn sentinel_row_bytes() -> [u8; 24] {
    let mut b = [0u8; 24];
    b[0..4].copy_from_slice(&u32::MAX.to_le_bytes());
    b[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    b[20..24].copy_from_slice(&u32::MAX.to_le_bytes());
    b
}
