//! The offline `.vxo` ENCODER — **Phase B-i** (`docs/VXO_FORMAT.md` "The offline encoder").
//!
//! [`write_vxo`] serializes an in-RAM [`BrickMap`] + its [`BlockRegistry`] to the region-streamed `.vxo`
//! format: region-bucket the bricks by `brick_coord.div_euclid(K)`, per brick `encode_paletted` its **8³
//! CORE** (§B2.7 resolution — NOT the halo: the loader re-halos from the resident set), intern identical
//! cores WITHIN a region ([`VoxelInterner`], R3), and emit the `VxoBrickEntry` table + region-local
//! `palette_blob`/`index_blob` + a [`VxoRegionHeader`]. R1 uniform bricks collapse to the entry's low-16-bit
//! id with the dedicated [`BRICK_FLAG_UNIFORM`] discriminant (no palette/index bytes). Region bodies are
//! STORE'd or per-region zstd'd (§B1.9); `BIDX` is sorted by `(z,y,x)`.
//!
//! **What round-trips (the delivered property):** the disk stores each brick's **8³ CORE** R2b-encoded, and the
//! loader decodes it back to a `Brick`; the packer (`pack_one`) then re-halos + re-encodes from the resident set.
//! So the guarantee is a **bit-identical `Brick` + a byte-identical packed `GpuBrickPatch`** — NOT a raw memcpy
//! of brick bodies into the GPU arena (the on-disk core is re-haloed/re-encoded at pack time, which is the
//! CORRECT `BrickSource` contract). The 8³-core choice keeps regions independently decodable AND the loaded
//! `Brick` bit-identical to a live one (the round-trip gate, §B2.8).

use std::io::Write;

use bevy::math::IVec3;
use bytemuck::bytes_of;
use rustc_hash::FxHashMap;

use super::format::*;
use crate::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, BrickMap, voxel_index};
use crate::voxel::gpu::VoxelInterner;
use crate::voxel::palette::{BlockDef, BlockId, BlockRegistry};

/// The geometry/identity parameters the caller supplies to [`write_vxo`] (the bits not derivable from the
/// [`BrickMap`] alone): the bake's `voxel_size`, the region granularity **K**, the asset pivot, and a name.
/// `bounds`/`brick_count`/`region_count` are computed from the map.
#[derive(Clone, Debug)]
pub struct VxoHeadParams {
    /// Metres per LOD0 voxel the asset was baked at (e.g. `0.05` post the D1 flip; legacy assets were `0.2`).
    pub voxel_size: f32,
    /// **K** — region edge in bricks (power of two; default [`DEFAULT_REGION_EDGE_BRICKS`] = 8).
    pub region_edge_bricks: u32,
    /// The asset PIVOT in LOD0 world-voxel coords (recorded, not baked). `(0,0,0)` for a merge-into-world scene.
    pub anchor_voxel: [i32; 3],
    /// Asset name / tags (debug + path-cache key).
    pub name: String,
}

impl Default for VxoHeadParams {
    fn default() -> Self {
        Self {
            voxel_size: crate::voxel::brickmap::VOXEL_SIZE,
            region_edge_bricks: DEFAULT_REGION_EDGE_BRICKS,
            anchor_voxel: [0, 0, 0],
            name: String::new(),
        }
    }
}

/// Whether a region body is stored uncompressed (`STORE`) or per-region zstd'd (`VXO_FORMAT.md` §B1.9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VxoCompression {
    /// Uncompressed — the region body is `bytemuck`-castable in place; `brik_raw_len == brik_comp_len`.
    Store,
    /// Per-region zstd at the given level (offline ~19 is fine; decode is what matters at runtime).
    Zstd(i32),
}

impl Default for VxoCompression {
    fn default() -> Self {
        VxoCompression::Zstd(19)
    }
}

/// The Euclidean region coord owning a LOD0 brick coord (correct for negatives — mirrors
/// `brick_coord_of_voxel`). `K = region_edge_bricks` (§B1.4).
#[inline]
pub fn region_of_brick(brick_coord: IVec3, k: i32) -> IVec3 {
    IVec3::new(brick_coord.x.div_euclid(k), brick_coord.y.div_euclid(k), brick_coord.z.div_euclid(k))
}

/// Encode `map` + `registry` to a `.vxo` file at `path`. STORE or per-region zstd per `comp`. Pure aside
/// from the final file write — builds the whole byte image in RAM then writes it once (B-i encodes from an
/// in-RAM map; the bounded-RAM region-by-region write is a later concern, `VXO_FORMAT.md` Migration).
pub fn write_vxo(
    path: impl AsRef<std::path::Path>,
    map: &BrickMap,
    registry: &BlockRegistry,
    params: &VxoHeadParams,
    comp: VxoCompression,
) -> anyhow::Result<()> {
    let bytes = encode_vxo(map, registry, params, comp)?;
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(path.as_ref())?;
    f.write_all(&bytes)?;
    f.flush()?;
    Ok(())
}

/// Build the full `.vxo` byte image in memory (the IO-free core of [`write_vxo`], so the round-trip test can
/// encode without touching the filesystem).
pub fn encode_vxo(
    map: &BrickMap,
    registry: &BlockRegistry,
    params: &VxoHeadParams,
    comp: VxoCompression,
) -> anyhow::Result<Vec<u8>> {
    let k = params.region_edge_bricks;
    anyhow::ensure!(k.is_power_of_two() && k > 0, "region_edge_bricks (K={k}) must be a positive power of two");
    let ki = k as i32;

    // 1. Region-bucket the bricks (sorted within each region by (z,y,x) brick coord so a region's entry table
    //    is binary-searchable on decode, and the bake is deterministic).
    let mut regions: FxHashMap<IVec3, Vec<IVec3>> = FxHashMap::default();
    let mut bounds_min = IVec3::splat(i32::MAX);
    let mut bounds_max = IVec3::splat(i32::MIN);
    for (&coord, _) in map.iter() {
        regions.entry(region_of_brick(coord, ki)).or_default().push(coord);
        // The asset's solid extent in LOD0 world VOXELS (brick coord · BRICK_EDGE .. +BRICK_EDGE).
        bounds_min = bounds_min.min(coord * BRICK_EDGE);
        bounds_max = bounds_max.max((coord + IVec3::ONE) * BRICK_EDGE);
    }
    if regions.is_empty() {
        // An empty map ⇒ degenerate bounds (a well-formed but empty asset).
        bounds_min = IVec3::ZERO;
        bounds_max = IVec3::ZERO;
    }

    // 2. Encode each region body; collect the BIDX directory. Iterate regions in (z,y,x) order so BRIK
    //    offsets are deterministic (BIDX is re-sorted below regardless, but a stable BRIK layout eases diffs).
    let mut region_coords: Vec<IVec3> = regions.keys().copied().collect();
    region_coords.sort_by_key(|c| (c.z, c.y, c.x));

    let mut brik_body: Vec<u8> = Vec::new();
    let mut bidx: Vec<VxoRegionDirEntry> = Vec::with_capacity(region_coords.len());
    let mut total_bricks: u64 = 0;
    for rc in &region_coords {
        let mut coords = regions.remove(rc).expect("region present");
        coords.sort_by_key(|c| (c.z, c.y, c.x));
        total_bricks += coords.len() as u64;
        let raw = encode_region(*rc, &coords, map)?;
        anyhow::ensure!(raw.len() as u64 <= u32::MAX as u64, "region {rc:?} body exceeds u32 byte length");
        let raw_len = raw.len() as u32;
        let stored = match comp {
            VxoCompression::Store => raw,
            VxoCompression::Zstd(level) => zstd_compress(&raw, level)?,
        };
        let comp_len = stored.len() as u32;
        let offset = brik_body.len() as u64;
        brik_body.extend_from_slice(&stored);
        let compression = match comp {
            VxoCompression::Store => VXO_REGION_STORE,
            VxoCompression::Zstd(_) => VXO_REGION_ZSTD,
        };
        bidx.push(VxoRegionDirEntry {
            region_coord: [rc.x, rc.y, rc.z],
            brick_count: coords.len() as u32,
            brik_offset: offset,
            brik_comp_len: comp_len,
            brik_raw_len: match comp {
                VxoCompression::Store => comp_len,
                VxoCompression::Zstd(_) => raw_len,
            },
            // EXPLICIT compression discriminant (§B1.5) — the reader branches on THIS, never on length equality.
            compression,
            _pad: [0; 15],
        });
    }
    // BIDX sorted by (z,y,x) — the binary-search key (§B1.5).
    bidx.sort_by_key(|e| (e.region_coord[2], e.region_coord[1], e.region_coord[0]));

    // 3. Assemble the file: header + HEAD + MATL + BIDX + BRIK. The compression mode is signalled PER REGION
    //    by BIDX's EXPLICIT `compression` byte (STORE/zstd), NOT in the file flags and NOT inferred from length
    //    equality — so `flags` is just the little-endian bit (bit1 = SVDAG is a B3 concern, never set by B-i).
    let flags = VXO_FLAG_LITTLE_ENDIAN;
    let mut out: Vec<u8> = Vec::with_capacity(64 + brik_body.len());
    write_file_header(&mut out, flags);
    write_chunk(&mut out, TAG_HEAD, &build_head_body(params, bounds_min, bounds_max, total_bricks, bidx.len() as u32));
    write_chunk(&mut out, TAG_MATL, &build_matl_body(registry));
    write_chunk(&mut out, TAG_BIDX, &build_bidx_body(&bidx));
    write_chunk(&mut out, TAG_BRIK, &brik_body);
    Ok(out)
}

// ============================================================================================
// Bounded-RAM STREAMING writer (Phase C1.7 — the out-of-core `.vxo` assembly)
// ============================================================================================

/// A bounded-RAM, region-at-a-time `.vxo` writer for the OUT-OF-CORE tiled voxelizer (C1.7). Where
/// [`encode_vxo`] builds the WHOLE byte image in RAM (every region body concatenated into one `Vec<u8>` —
/// untenable for a Bistro-scale BRIK chunk of many GB), this streams each region's compressed body straight
/// to a scratch file as it is produced, holding in RAM only the small `BIDX` directory (O(regions) × 48 B)
/// and one region's bricks at a time. At [`finish`](Self::finish) it writes the final file in the spec order
/// (file header, HEAD, MATL, BIDX, BRIK) by writing the header/HEAD/MATL/BIDX — now that the BRIK offsets and
/// counts are known — and then COPYING the scratch BRIK body in fixed-size chunks (so the assembly is
/// bounded-RAM too, never loading the whole BRIK at once).
///
/// USAGE: `new(params, registry, comp)` → `add_region(region_coord, sorted_bricks)` per non-empty region →
/// `finish(out_path)`. Regions may be added in any order; `BIDX` is sorted by `(z,y,x)` at finish, and the
/// caller is expected to feed them in a deterministic order (the tiled voxelizer does — region-id order) so
/// the BRIK body layout is byte-reproducible. The bricks WITHIN a region MUST be pre-sorted by `(z,y,x)`
/// (same contract as [`encode_region_bricks`]).
///
/// The MATL chunk is built eagerly from `registry` (it is small + known up-front). The scratch BRIK file lives
/// next to the output (a sibling `*.brik.tmp`) and is deleted on success; on an error it is left for debugging.
pub struct VxoStreamWriter {
    params: VxoHeadParams,
    comp: VxoCompression,
    /// The MATL chunk body (built once from the registry up-front).
    matl_body: Vec<u8>,
    /// The scratch file accumulating the (compressed) BRIK region bodies in add order.
    brik_scratch_path: std::path::PathBuf,
    brik_scratch: std::io::BufWriter<std::fs::File>,
    /// Running byte offset within the BRIK body (== bytes written to the scratch).
    brik_offset: u64,
    /// The region directory, accumulated in RAM (small: O(regions)).
    bidx: Vec<VxoRegionDirEntry>,
    /// Solid-extent bounds in LOD0 world voxels (accumulated across all added bricks).
    bounds_min: IVec3,
    bounds_max: IVec3,
    /// Total non-empty bricks added.
    total_bricks: u64,
}

impl VxoStreamWriter {
    /// Open a streaming writer. `brik_scratch_path` is the scratch file for the BRIK body (caller-chosen so it
    /// can live under the run's scratch dir); it is created/truncated now and removed by [`finish`](Self::finish)
    /// on success. `registry` is captured into the MATL chunk immediately.
    pub fn new(
        params: VxoHeadParams,
        registry: &BlockRegistry,
        comp: VxoCompression,
        brik_scratch_path: impl AsRef<std::path::Path>,
    ) -> anyhow::Result<Self> {
        let k = params.region_edge_bricks;
        anyhow::ensure!(
            k.is_power_of_two() && k > 0,
            "region_edge_bricks (K={k}) must be a positive power of two"
        );
        let brik_scratch_path = brik_scratch_path.as_ref().to_path_buf();
        if let Some(parent) = brik_scratch_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(&brik_scratch_path)?;
        Ok(Self {
            matl_body: build_matl_body(registry),
            params,
            comp,
            brik_scratch_path,
            brik_scratch: std::io::BufWriter::new(file),
            brik_offset: 0,
            bidx: Vec::new(),
            bounds_min: IVec3::splat(i32::MAX),
            bounds_max: IVec3::splat(i32::MIN),
            total_bricks: 0,
        })
    }

    /// Encode + append ONE region's body. `bricks` is the region's `(brick_coord, brick)` pairs, which MUST be
    /// pre-sorted by `(z,y,x)` (the [`encode_region_bricks`] contract). An empty `bricks` slice is ignored (a
    /// region with no bricks has no directory entry — the sparse-absent convention). Updates the BIDX directory,
    /// the BRIK offset, the brick count, and the solid bounds.
    pub fn add_region(&mut self, region_coord: IVec3, bricks: &[(IVec3, &Brick)]) -> anyhow::Result<()> {
        use std::io::Write as _;
        if bricks.is_empty() {
            return Ok(());
        }
        let raw = encode_region_bricks(region_coord, bricks)?;
        anyhow::ensure!(raw.len() as u64 <= u32::MAX as u64, "region {region_coord:?} body exceeds u32 byte length");
        let raw_len = raw.len() as u32;
        let stored = match self.comp {
            VxoCompression::Store => raw,
            VxoCompression::Zstd(level) => zstd_compress(&raw, level)?,
        };
        let comp_len = stored.len() as u32;
        let offset = self.brik_offset;
        self.brik_scratch.write_all(&stored)?;
        self.brik_offset += stored.len() as u64;
        let compression = match self.comp {
            VxoCompression::Store => VXO_REGION_STORE,
            VxoCompression::Zstd(_) => VXO_REGION_ZSTD,
        };
        self.bidx.push(VxoRegionDirEntry {
            region_coord: [region_coord.x, region_coord.y, region_coord.z],
            brick_count: bricks.len() as u32,
            brik_offset: offset,
            brik_comp_len: comp_len,
            brik_raw_len: match self.comp {
                VxoCompression::Store => comp_len,
                VxoCompression::Zstd(_) => raw_len,
            },
            compression,
            _pad: [0; 15],
        });
        self.total_bricks += bricks.len() as u64;
        for &(coord, _) in bricks {
            self.bounds_min = self.bounds_min.min(coord * BRICK_EDGE);
            self.bounds_max = self.bounds_max.max((coord + IVec3::ONE) * BRICK_EDGE);
        }
        Ok(())
    }

    /// Finalize: flush the scratch BRIK, then write the final `.vxo` at `out_path` in spec order (file header +
    /// HEAD + MATL + BIDX + BRIK). The header/HEAD/MATL/BIDX are written from RAM (all small); the BRIK chunk
    /// header is written, then the scratch body is COPIED in fixed 4 MiB chunks (bounded-RAM) followed by the
    /// 16-byte body-alignment pad. The scratch file is removed on success. Consumes `self`.
    pub fn finish(mut self, out_path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        use std::io::{Read as _, Seek as _, Write as _};

        // Degenerate empty asset ⇒ zeroed bounds (matches `encode_vxo`).
        if self.bidx.is_empty() {
            self.bounds_min = IVec3::ZERO;
            self.bounds_max = IVec3::ZERO;
        }
        // BIDX sorted by (z,y,x) — the binary-search key (§B1.5). The BRIK body layout is the ADD order (the
        // caller feeds regions deterministically), but the directory is sorted for lookup; each entry already
        // carries its own `brik_offset`, so re-sorting the directory does not move the body.
        self.bidx.sort_by_key(|e| (e.region_coord[2], e.region_coord[1], e.region_coord[0]));

        // Flush + close the write handle, then REOPEN the scratch read-only (the write handle from
        // `File::create` is write-only — reading from it errors on Windows).
        self.brik_scratch.flush()?;
        drop(self.brik_scratch.into_inner()?);
        let mut scratch = std::fs::File::open(&self.brik_scratch_path)?;
        scratch.seek(std::io::SeekFrom::Start(0))?;

        let out_path = out_path.as_ref();
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut out = std::io::BufWriter::new(std::fs::File::create(out_path)?);

        // 1. file header + HEAD + MATL + BIDX (all small, built in RAM).
        let mut prefix: Vec<u8> = Vec::new();
        write_file_header(&mut prefix, VXO_FLAG_LITTLE_ENDIAN);
        write_chunk(
            &mut prefix,
            TAG_HEAD,
            &build_head_body(&self.params, self.bounds_min, self.bounds_max, self.total_bricks, self.bidx.len() as u32),
        );
        write_chunk(&mut prefix, TAG_MATL, &self.matl_body);
        write_chunk(&mut prefix, TAG_BIDX, &build_bidx_body(&self.bidx));
        out.write_all(&prefix)?;

        // 2. The BRIK chunk: its 32-byte header (body_len = the scratch length, body CRC over the streamed
        //    body), then the scratch body copied in bounded-RAM chunks, then the 16-byte alignment pad. The
        //    body CRC is computed in the SAME streaming pass (a running CRC32 over the copied chunks) so we
        //    never need the whole BRIK resident.
        let body_len = self.brik_offset;
        // We must write the chunk header (which carries body_crc32) BEFORE the body — but the CRC needs the
        // body. So compute the CRC in a first streaming pass, then a second copy pass. Two sequential reads of
        // a scratch file are cheap vs. holding the multi-GB body in RAM.
        let mut crc_buf = vec![0u8; 4 * 1024 * 1024];
        let mut crc = Crc32Stream::new();
        loop {
            let n = scratch.read(&mut crc_buf)?;
            if n == 0 {
                break;
            }
            crc.update(&crc_buf[..n]);
        }
        let body_crc32 = crc.finalize();
        let brik_header = VxoChunkHeader {
            tag: TAG_BRIK,
            _pad0: 0,
            body_len,
            body_crc32,
            _pad1: [0; 3],
        };
        out.write_all(bytes_of(&brik_header))?;
        // Second pass: copy the body.
        scratch.seek(std::io::SeekFrom::Start(0))?;
        let mut copy_buf = crc_buf; // reuse the 4 MiB buffer
        loop {
            let n = scratch.read(&mut copy_buf)?;
            if n == 0 {
                break;
            }
            out.write_all(&copy_buf[..n])?;
        }
        // The trailing 16-byte body-alignment pad (OUTSIDE body_len, §B1.0).
        let pad = align16(body_len) - body_len;
        out.write_all(&vec![0u8; pad as usize])?;
        out.flush()?;
        drop(out);

        // Success: remove the scratch BRIK file.
        drop(scratch);
        let _ = std::fs::remove_file(&self.brik_scratch_path);
        Ok(())
    }
}

/// Write the 16-byte file header (`VxoFileHeader`) with the magic + version + `flags`, computing the
/// header CRC32 over the first 8 bytes (`VXO_FORMAT.md` §B1.0).
fn write_file_header(out: &mut Vec<u8>, flags: u16) {
    // CRC over the 8 bytes magic + format_version + flags (the spec's "8 bytes above").
    let mut crc_input = [0u8; 8];
    crc_input[0..4].copy_from_slice(&VXO_MAGIC);
    crc_input[4..6].copy_from_slice(&VXO_FORMAT_VERSION.to_le_bytes());
    crc_input[6..8].copy_from_slice(&flags.to_le_bytes());
    let header = VxoFileHeader {
        magic: VXO_MAGIC,
        format_version: VXO_FORMAT_VERSION,
        flags,
        header_crc32: crc32(&crc_input),
        _reserved: 0,
    };
    out.extend_from_slice(bytes_of(&header));
}

/// Frame one chunk: the 32-byte `VxoChunkHeader` (tag + `body_len` + body CRC32, padded to a 16-multiple) then
/// `body`, then zero-pad the body up to a 16-byte multiple (the pad is OUTSIDE `body_len`, §B1.0) so the next
/// chunk starts 16-aligned.
fn write_chunk(out: &mut Vec<u8>, tag: [u8; 4], body: &[u8]) {
    let header = VxoChunkHeader {
        tag,
        _pad0: 0,
        body_len: body.len() as u64,
        body_crc32: crc32(body),
        _pad1: [0; 3],
    };
    out.extend_from_slice(bytes_of(&header));
    out.extend_from_slice(body);
    let pad = align16(body.len() as u64) - body.len() as u64;
    out.extend(std::iter::repeat_n(0u8, pad as usize));
}

/// The `HEAD` body: the `VxoHead` POD prefix then the UTF-8 `name` padded to 4 bytes (§B1.1).
fn build_head_body(
    params: &VxoHeadParams,
    bounds_min: IVec3,
    bounds_max: IVec3,
    brick_count: u64,
    region_count: u32,
) -> Vec<u8> {
    let name = params.name.as_bytes();
    let head = VxoHead {
        head_version: VXO_HEAD_VERSION,
        _pad0: 0,
        voxel_size: params.voxel_size,
        brick_edge: BRICK_EDGE as u32,
        max_lod: 0, // no LODS chunk in B-i
        bounds_min: [bounds_min.x, bounds_min.y, bounds_min.z],
        bounds_max: [bounds_max.x, bounds_max.y, bounds_max.z],
        anchor_voxel: params.anchor_voxel,
        region_edge_bricks: params.region_edge_bricks,
        brick_count,
        region_count,
        _pad1: 0,
        name_len: name.len() as u32,
        _pad2: 0,
    };
    let mut body = Vec::with_capacity(std::mem::size_of::<VxoHead>() + name.len() + 3);
    body.extend_from_slice(bytes_of(&head));
    body.extend_from_slice(name);
    // Pad the name to 4 bytes in-body (§B1.1).
    while !body.len().is_multiple_of(4) {
        body.push(0);
    }
    body
}

/// The `MATL` body: `material_count: u32` + `_pad: u32` then one [`VxoMaterial`] per registered block, index
/// `i` → `BlockId(i)` (§B1.2). Block 0 is AIR. Linear colours straight through from [`BlockDef`].
fn build_matl_body(registry: &BlockRegistry) -> Vec<u8> {
    let count = registry.len() as u32;
    let mut body = Vec::with_capacity(8 + registry.len() * std::mem::size_of::<VxoMaterial>());
    body.extend_from_slice(&count.to_le_bytes());
    body.extend_from_slice(&0u32.to_le_bytes()); // _pad
    for i in 0..registry.len() {
        let def: &BlockDef = registry.block(BlockId(i as u16));
        body.extend_from_slice(bytes_of(&material_from_def(def)));
    }
    body
}

/// Convert a [`BlockDef`] to its on-disk [`VxoMaterial`] (linear colours straight through; the emitter flag
/// is precomputed from any non-zero emissive, §B1.2).
fn material_from_def(def: &BlockDef) -> VxoMaterial {
    let emitter = def.emissive != [0.0, 0.0, 0.0];
    let mut flags = 0u32;
    if def.tintable {
        flags |= MATL_FLAG_TINTABLE;
    }
    if emitter {
        flags |= MATL_FLAG_EMITTER;
    }
    VxoMaterial {
        albedo: def.color,
        emissive: [def.emissive[0], def.emissive[1], def.emissive[2], 1.0],
        roughness: def.roughness,
        metallic: def.metal,
        flags,
        _pad: 0,
    }
}

/// The `BIDX` body: `entry_count: u32` + `_pad: u32` then the sorted `VxoRegionDirEntry` table (§B1.5).
fn build_bidx_body(bidx: &[VxoRegionDirEntry]) -> Vec<u8> {
    let mut body = Vec::with_capacity(8 + std::mem::size_of_val(bidx));
    body.extend_from_slice(&(bidx.len() as u32).to_le_bytes());
    body.extend_from_slice(&0u32.to_le_bytes()); // _pad
    body.extend_from_slice(bytemuck::cast_slice(bidx));
    body
}

/// Encode ONE region's body (DECOMPRESSED) from a `&BrickMap` — looks up each `coord`'s brick and forwards
/// to [`encode_region_bricks`]. The full-RAM [`encode_vxo`] path uses this; the bounded-RAM streaming writer
/// ([`VxoStreamWriter`]) calls [`encode_region_bricks`] directly from disk-loaded bricks (no resident map).
fn encode_region(region_coord: IVec3, coords: &[IVec3], map: &BrickMap) -> anyhow::Result<Vec<u8>> {
    let bricks: Vec<(IVec3, &Brick)> =
        coords.iter().map(|&c| (c, map.get(c).expect("region coord present in map"))).collect();
    encode_region_bricks(region_coord, &bricks)
}

/// Encode ONE region's body (DECOMPRESSED): `VxoRegionHeader` then `[VxoBrickEntry; N]` then `palette_blob`
/// then `index_blob` (§B1.3). Per brick: the **8³ core** is R1-collapsed to a uniform entry, else
/// R2b-encoded + R3-interned WITHIN this region. The `is_full` bit is baked into the entry flags (§B2.5).
///
/// `bricks` is the region's `(brick_coord, brick)` pairs, which the caller MUST pass sorted by `(z,y,x)`
/// (so the entry table is binary-searchable on decode + the bake is deterministic). Taking bricks BY VALUE-REF
/// (not via a `&BrickMap`) lets the streaming out-of-core writer feed bricks straight from disk tiles without
/// a resident map (the bounded-RAM C1 path).
///
/// Returns an error if a dense brick's region-local `index_off`/`palette_off` overflows `u32` (the robust-by-
/// construction backstop A4.1 mandates — region-local offsets always fit `u32` today, but the guard means a
/// future too-large region is a HARD bake error, never a silent wrap).
pub(crate) fn encode_region_bricks(
    region_coord: IVec3,
    bricks: &[(IVec3, &Brick)],
) -> anyhow::Result<Vec<u8>> {
    let mut entries: Vec<VxoBrickEntry> = Vec::with_capacity(bricks.len());
    let mut palette_blob: Vec<u32> = Vec::new();
    let mut index_blob: Vec<u32> = Vec::new();
    // R3 dedup WITHIN this region only (regions stay independently decompressible, §B1.3 note).
    let mut interner = VoxelInterner::new();

    for &(coord, brick) in bricks {
        let mut flags = 0u8;
        // §B2.5: bake the conservative-cull bits without storing voxels — surface unless fully solid; full
        // bricks are prunable when their neighbours are full too (the classify reads this).
        if brick.is_full() {
            flags |= BRICK_FLAG_FULL;
        } else {
            flags |= BRICK_FLAG_SURFACE;
        }

        // R1 — UNIFORM core collapse: a single-block core stores its id in the entry's LOW 16 bits, the uniform
        // discriminant rides in `flags` (A4.1: a dedicated bit, NOT bit-31 of `index_off`), no palette/index
        // bytes. (A uniform-AIR brick is never in the map — `insert` drops empties — so `uniform_block()` is solid.)
        if let Some(block) = brick.uniform_block() {
            entries.push(VxoBrickEntry {
                brick_coord: [coord.x, coord.y, coord.z],
                index_off: block.0 as u32 & 0xFFFF,
                palette_off: 0,
                index_bits: 0,
                palette_len: 0,
                flags: flags | BRICK_FLAG_UNIFORM,
                _pad0: 0,
                _pad1: [0; 2],
            });
            continue;
        }

        // R2b DENSE — the 8³ CORE cells in `voxel_index` order (NOT haloed; §B2.7 resolution). The loader
        // decodes these back into a `Brick` and the packer re-halos from the resident set, so the loaded
        // brick is bit-identical to a live one.
        let cells = core_cells(brick);
        // The §B1.3 prealloc hint: `k` distinct ids, clamped into the entry's `u8` (k ≤ 255 — no shipping brick
        // approaches it; the decode is bounded by `index_bits`, not this).
        let palette_len = palette_len_of(&cells);
        let layout = interner.intern_paletted(&mut index_blob, &mut palette_blob, &cells);
        // A4.1 robust-by-construction backstop: the region-local offsets are derived from the blob lengths, so a
        // blob exceeding u32 would silently wrap on cast. Region-local ⇒ this never fires today, but a HARD error
        // beats silent corruption if a future region grows past 4 Gi words.
        anyhow::ensure!(
            (index_blob.len() as u64) <= u32::MAX as u64 && (palette_blob.len() as u64) <= u32::MAX as u64,
            "region {region_coord:?}: index/palette blob exceeds u32 word offset range (corruption backstop)"
        );
        entries.push(VxoBrickEntry {
            brick_coord: [coord.x, coord.y, coord.z],
            index_off: layout.voxel_offset, // full u32 region-local offset; the uniform discriminant is in `flags`
            palette_off: layout.palette_base,
            index_bits: layout.index_bits,
            palette_len,
            flags,
            _pad0: 0,
            _pad1: [0; 2],
        });
    }

    // Assemble the region blob: header + entries + palette_blob + index_blob.
    let header = VxoRegionHeader {
        region_coord: [region_coord.x, region_coord.y, region_coord.z],
        brick_count: entries.len() as u32,
        palette_u32: palette_blob.len() as u32,
        index_u32: index_blob.len() as u32,
        lod: 0,
        _pad: 0,
    };
    let mut out = Vec::with_capacity(
        std::mem::size_of::<VxoRegionHeader>()
            + entries.len() * std::mem::size_of::<VxoBrickEntry>()
            + (palette_blob.len() + index_blob.len()) * 4,
    );
    out.extend_from_slice(bytes_of(&header));
    out.extend_from_slice(bytemuck::cast_slice(&entries));
    out.extend_from_slice(bytemuck::cast_slice(&palette_blob));
    out.extend_from_slice(bytemuck::cast_slice(&index_blob));
    Ok(out)
}

/// The brick's 8³ CORE cells as `u32` block ids in `voxel_index` order (`+X` fastest, then `+Y`, then `+Z`)
/// — the EXACT order `encode_paletted` / `decode_paletted_cell` use, so the disk R2b triple is byte-identical
/// to the resident one over the core.
fn core_cells(brick: &Brick) -> Vec<u32> {
    let mut cells = Vec::with_capacity(BRICK_VOXELS);
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                debug_assert_eq!(cells.len(), voxel_index(x, y, z));
                cells.push(brick.get(x, y, z).0 as u32);
            }
        }
    }
    cells
}

/// The `k` distinct ids in `cells` as the entry's `palette_len` (clamped into the `u8` field; §B1.3 — a
/// PREALLOC hint, the decode is bounded by `index_bits`, so a `>255` value is impossible here for v1 and
/// would be clamped/asserted).
fn palette_len_of(cells: &[u32]) -> u8 {
    distinct_count(cells).min(255) as u8
}

/// Count the distinct ids in `cells` (the palette `k` — matching `gpu::encode_paletted`'s first-seen palette
/// length, the value the entry's `palette_len` hint records).
fn distinct_count(cells: &[u32]) -> usize {
    let mut seen: rustc_hash::FxHashSet<u16> = rustc_hash::FxHashSet::default();
    for &c in cells {
        seen.insert(c as u16);
    }
    seen.len()
}

/// zstd-compress `raw` at `level` (§B1.9). Wraps the C-backed `zstd::bulk::compress`, which pulls a C toolchain
/// (`zstd-sys`/`cc`) — so it is OFFLINE-ENCODE ONLY, behind the `vxo-encode` feature. The RUNTIME decode path
/// uses pure-Rust `ruzstd` ([`super::reader`]), never C zstd, so the shipped library/runtime build needs no C
/// toolchain. The `voxelize_scene` example enables `vxo-encode`; a default build that asks for `Zstd` without
/// the feature gets a clear error (use `Store`, or enable `vxo-encode`).
#[cfg(feature = "vxo-encode")]
fn zstd_compress(raw: &[u8], level: i32) -> anyhow::Result<Vec<u8>> {
    zstd::bulk::compress(raw, level).map_err(|e| anyhow::anyhow!("zstd compress: {e}"))
}

/// Stub when `vxo-encode` is off: zstd COMPRESSION needs the C `zstd` crate (toolchain). The default build can
/// still `Store` and (via pure-Rust `ruzstd`) DECODE zstd, but cannot PRODUCE a zstd region body.
#[cfg(not(feature = "vxo-encode"))]
fn zstd_compress(_raw: &[u8], _level: i32) -> anyhow::Result<Vec<u8>> {
    anyhow::bail!(
        "vxo: zstd region compression needs the `vxo-encode` feature (the offline encoder's C zstd). \
         Use VxoCompression::Store, or build with --features vxo-encode."
    )
}
