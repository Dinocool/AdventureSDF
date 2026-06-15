//! The offline `.vxo` ENCODER — **Phase B-i** (`docs/VXO_FORMAT.md` "The offline encoder").
//!
//! [`write_vxo`] serializes an in-RAM [`BrickMap`] + its [`BlockRegistry`] to the region-streamed `.vxo`
//! format: region-bucket the bricks by `brick_coord.div_euclid(K)`, per brick `encode_paletted` its **8³
//! CORE** (§B2.7 resolution — NOT the halo: the loader re-halos from the resident set), intern identical
//! cores WITHIN a region ([`VoxelInterner`], R3), and emit the `VxoBrickEntry` table + region-local
//! `palette_blob`/`index_blob` + a [`VxoRegionHeader`]. R1 uniform bricks collapse to the entry's
//! [`BRICK_UNIFORM_FLAG`] id (no palette/index bytes). Region bodies are STORE'd or per-region zstd'd
//! (§B1.9); `BIDX` is sorted by `(z,y,x)`.
//!
//! The per-brick R2b `(palette, index_bits, indices)` triple is byte-identical to `gpu.rs`'s
//! [`encode_paletted`] output — the disk stores the resident encoding verbatim (the memcpy-decode property,
//! §0.1). The 8³-core choice keeps regions independently decodable AND the loaded `Brick` bit-identical to a
//! live one (the round-trip gate, §B2.8).

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
    /// Metres per LOD0 voxel the asset was baked at (e.g. `0.2` today, `0.05` after the D1 flip).
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
        let raw = encode_region(*rc, &coords, map);
        let raw_len = raw.len() as u32;
        let stored = match comp {
            VxoCompression::Store => raw,
            VxoCompression::Zstd(level) => zstd_compress(&raw, level)?,
        };
        let comp_len = stored.len() as u32;
        let offset = brik_body.len() as u64;
        brik_body.extend_from_slice(&stored);
        bidx.push(VxoRegionDirEntry {
            region_coord: [rc.x, rc.y, rc.z],
            brick_count: coords.len() as u32,
            brik_offset: offset,
            brik_comp_len: comp_len,
            // STORE ⇒ raw_len == comp_len (the §B1.5 convention the reader uses to detect STORE).
            brik_raw_len: match comp {
                VxoCompression::Store => comp_len,
                VxoCompression::Zstd(_) => raw_len,
            },
        });
    }
    // BIDX sorted by (z,y,x) — the binary-search key (§B1.5).
    bidx.sort_by_key(|e| (e.region_coord[2], e.region_coord[1], e.region_coord[0]));

    // 3. Assemble the file: header + HEAD + MATL + BIDX + BRIK. The compression mode is signalled PER REGION
    //    in BIDX (`brik_comp_len == brik_raw_len` ⇒ STORE), NOT in the file flags — so `flags` is just the
    //    little-endian bit (bit1 = SVDAG is a B3 concern, never set by B-i).
    let flags = VXO_FLAG_LITTLE_ENDIAN;
    let mut out: Vec<u8> = Vec::with_capacity(64 + brik_body.len());
    write_file_header(&mut out, flags);
    write_chunk(&mut out, TAG_HEAD, &build_head_body(params, bounds_min, bounds_max, total_bricks, bidx.len() as u32));
    write_chunk(&mut out, TAG_MATL, &build_matl_body(registry));
    write_chunk(&mut out, TAG_BIDX, &build_bidx_body(&bidx));
    write_chunk(&mut out, TAG_BRIK, &brik_body);
    Ok(out)
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

/// Frame one chunk: the 16-byte `VxoChunkHeader` (tag + `body_len` + body CRC32) then `body`, then zero-pad
/// the body up to a 16-byte multiple (the pad is OUTSIDE `body_len`, §B1.0) so the next chunk starts aligned.
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

/// Encode ONE region's body (DECOMPRESSED): `VxoRegionHeader` then `[VxoBrickEntry; N]` then `palette_blob`
/// then `index_blob` (§B1.3). Per brick: the **8³ core** is R1-collapsed to a uniform entry, else
/// R2b-encoded + R3-interned WITHIN this region. The `is_full` bit is baked into the entry flags (§B2.5).
fn encode_region(region_coord: IVec3, coords: &[IVec3], map: &BrickMap) -> Vec<u8> {
    let mut entries: Vec<VxoBrickEntry> = Vec::with_capacity(coords.len());
    let mut palette_blob: Vec<u32> = Vec::new();
    let mut index_blob: Vec<u32> = Vec::new();
    // R3 dedup WITHIN this region only (regions stay independently decompressible, §B1.3 note).
    let mut interner = VoxelInterner::new();

    for &coord in coords {
        let brick = map.get(coord).expect("region coord present in map");
        let mut flags = 0u8;
        // §B2.5: bake the conservative-cull bits without storing voxels — surface unless fully solid; full
        // bricks are prunable when their neighbours are full too (the classify reads this).
        if brick.is_full() {
            flags |= BRICK_FLAG_FULL;
        } else {
            flags |= BRICK_FLAG_SURFACE;
        }

        // R1 — UNIFORM core collapse: a single-block core stores its id in the entry, no palette/index bytes.
        // (A uniform-AIR brick is never in the map — `insert` drops empties — so `uniform_block()` is solid.)
        if let Some(block) = brick.uniform_block() {
            entries.push(VxoBrickEntry {
                brick_coord: [coord.x, coord.y, coord.z],
                index_off: BRICK_UNIFORM_FLAG | (block.0 as u32 & 0xFFFF),
                palette_off: 0,
                index_bits: 0,
                palette_len: 0,
                flags,
                _pad0: 0,
                _pad1: [0; 2],
            });
            continue;
        }

        // R2b DENSE — the 8³ CORE cells in `voxel_index` order (NOT haloed; §B2.7 resolution). The loader
        // decodes these back into a `Brick` and the packer re-halos from the resident set, so the loaded
        // brick is bit-identical to a live one.
        let cells = core_cells(brick);
        // Assert k ≤ 255 (the entry's u8 palette_len; §B1.3 — no shipping brick approaches it).
        debug_assert!(distinct_count(&cells) <= 255, "a brick core needs ≤255 distinct ids for v1");
        let layout = interner.intern_paletted(&mut index_blob, &mut palette_blob, &cells);
        let palette_len = palette_len_of(&cells);
        entries.push(VxoBrickEntry {
            brick_coord: [coord.x, coord.y, coord.z],
            index_off: layout.voxel_offset, // bit31 clear (full u32 range, < 2^31) ⇒ not uniform
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
    out
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

/// zstd-compress `raw` at `level` (§B1.9). Wraps `zstd::bulk::compress`; an error is surfaced as
/// `anyhow::Error` (a compression failure is a hard bake error).
fn zstd_compress(raw: &[u8], level: i32) -> anyhow::Result<Vec<u8>> {
    zstd::bulk::compress(raw, level).map_err(|e| anyhow::anyhow!("zstd compress: {e}"))
}
