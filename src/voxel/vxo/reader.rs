//! A full-file `.vxo` READER sufficient for the round-trip — **Phase B-i** (`docs/VXO_FORMAT.md` §B2, the
//! simple-reader subset).
//!
//! [`VxoFile::open`]/[`VxoFile::parse`] read the file header + every chunk (HEAD/MATL/BIDX/BRIK), verify
//! CRCs, and skip unknown chunks (the forward-compat rule, §B1.0). [`VxoFile::decode_region`] decompresses a
//! region body → its `(entries, palette_blob, index_blob)` [`DecodedRegion`]; [`DecodedRegion::brick`]
//! decodes one entry → a [`Brick`] (uniform via the dedicated [`BRICK_FLAG_UNIFORM`] bit; dense via
//! `decode_paletted_cell` over the 8³ core → `Brick::from_voxels`), reusing the EXACT `gpu.rs` decode SSOT so a
//! read-back brick is bit-identical to the live-generated one (the round-trip gate, §B2.8).
//!
//! This is the WHOLE-FILE reader (B-i). The streamed mmap `VxoSource` + LRU + `classify` + `BrickSource`
//! impl are Phase B-ii (out of scope here).

use bevy::math::IVec3;
use bytemuck::{Pod, pod_read_unaligned};

use super::format::*;
use crate::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, voxel_index};
use crate::voxel::gpu::decode_paletted_cell;
use crate::voxel::palette::{BlockId, BlockRegistry};

/// A parsed `.vxo` file held in RAM (B-i full-file reader): the `HEAD`, the rebuilt [`BlockRegistry`], the
/// sorted `BIDX` directory, and the raw `BRIK` body (each region's bytes sliced out lazily by
/// [`Self::decode_region`]). The streamed mmap variant is B-ii.
#[derive(Clone, Debug)]
pub struct VxoFile {
    /// Parsed `HEAD` (`voxel_size`, bounds, K, anchor, counts).
    pub head: VxoHead,
    /// The asset name (UTF-8, from the `HEAD` body suffix).
    pub name: String,
    /// The registry rebuilt from `MATL` (`BlockId(i)` ← entry `i`).
    pub registry: BlockRegistry,
    /// The sorted region directory (`(z,y,x)` order — binary-search key).
    pub bidx: Vec<VxoRegionDirEntry>,
    /// The whole `BRIK` chunk body (the concatenation of all region bodies).
    brik: Vec<u8>,
}

/// A decompressed region (`VXO_FORMAT.md` §B1.3): its brick entry table + region-local palette/index blobs.
/// The `brick(i)` decode reuses the `gpu.rs` SSOT so it is bit-identical to a live brick.
#[derive(Clone, Debug)]
pub struct DecodedRegion {
    /// The region's K-brick-grid coord (verified against the `BIDX` key on decode).
    pub region_coord: IVec3,
    /// One entry per brick in this region (in the encoder's `(z,y,x)` order).
    pub entries: Vec<VxoBrickEntry>,
    /// Region-local palette blob — each dense brick's `k` distinct ids (one `u32` each).
    pub palette_blob: Vec<u32>,
    /// Region-local bit-packed index blob — each dense brick's `encode_paletted(...).indices`.
    pub index_blob: Vec<u32>,
}

impl VxoFile {
    /// Read + parse a `.vxo` from disk.
    pub fn open(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path.as_ref())
            .map_err(|e| anyhow::anyhow!("vxo: read {}: {e}", path.as_ref().display()))?;
        Self::parse(&bytes)
    }

    /// Parse a `.vxo` from an in-RAM byte image (the IO-free core, shared by [`Self::open`] + the round-trip
    /// test). Verifies the header + chunk CRCs, parses HEAD/MATL/BIDX/BRIK, and SKIPS unknown chunks (§B1.0).
    pub fn parse(bytes: &[u8]) -> anyhow::Result<Self> {
        anyhow::ensure!(bytes.len() >= 16, "vxo: file shorter than the 16-byte header");
        let fh: VxoFileHeader = pod_read_unaligned(&bytes[0..16]);
        anyhow::ensure!(fh.magic == VXO_MAGIC, "vxo: bad magic {:?} (expected VXO1)", fh.magic);
        anyhow::ensure!(
            fh.format_version == VXO_FORMAT_VERSION,
            "vxo: format_version {} unsupported (this reader is v{VXO_FORMAT_VERSION})",
            fh.format_version
        );
        // Verify the header CRC over the 8 bytes magic + format_version + flags.
        let hdr_crc = crc32(&bytes[0..8]);
        anyhow::ensure!(hdr_crc == fh.header_crc32, "vxo: header CRC mismatch (file corrupt)");
        anyhow::ensure!(
            fh.flags & VXO_FLAG_SVDAG == 0,
            "vxo: file is SVDAG-encoded (flag bit1) — B-i reader handles only plain R2b BRIK"
        );

        let mut head: Option<(VxoHead, String)> = None;
        let mut registry: Option<BlockRegistry> = None;
        let mut bidx: Option<Vec<VxoRegionDirEntry>> = None;
        let mut brik: Option<Vec<u8>> = None;

        // Loop the chunks: parse a known tag, else skip body_len (rounded up to 16) — the §B1.0 reader rule.
        let ch_hdr = std::mem::size_of::<VxoChunkHeader>();
        let mut pos = std::mem::size_of::<VxoFileHeader>();
        while pos + ch_hdr <= bytes.len() {
            let ch: VxoChunkHeader = pod_read_unaligned(&bytes[pos..pos + ch_hdr]);
            let body_start = pos + ch_hdr;
            let body_len = ch.body_len as usize;
            anyhow::ensure!(body_start + body_len <= bytes.len(), "vxo: chunk {:?} body overruns file", ch.tag);
            let body = &bytes[body_start..body_start + body_len];
            // CRC verify (0 = skip).
            if ch.body_crc32 != 0 {
                anyhow::ensure!(crc32(body) == ch.body_crc32, "vxo: chunk {:?} body CRC mismatch", ch.tag);
            }
            match ch.tag {
                TAG_HEAD => head = Some(parse_head(body)?),
                TAG_MATL => registry = Some(parse_matl(body)?),
                TAG_BIDX => bidx = Some(parse_bidx(body)?),
                TAG_BRIK => brik = Some(body.to_vec()),
                TAG_END => break,
                _ => { /* unknown chunk — skip (forward compat, §B1.0) */ }
            }
            // Advance past the body + its 16-B alignment pad (the pad is OUTSIDE body_len, §B1.0).
            pos = body_start + align16(ch.body_len) as usize;
        }

        let (head, name) = head.ok_or_else(|| anyhow::anyhow!("vxo: missing REQUIRED HEAD chunk"))?;
        let registry = registry.ok_or_else(|| anyhow::anyhow!("vxo: missing REQUIRED MATL chunk"))?;
        let bidx = bidx.ok_or_else(|| anyhow::anyhow!("vxo: missing REQUIRED BIDX chunk"))?;
        let brik = brik.ok_or_else(|| anyhow::anyhow!("vxo: missing REQUIRED BRIK chunk"))?;

        anyhow::ensure!(
            head.brick_edge == BRICK_EDGE as u32,
            "vxo: brick_edge {} != engine BRICK_EDGE {} — incompatible asset",
            head.brick_edge,
            BRICK_EDGE
        );
        Ok(Self { head, name, registry, bidx, brik })
    }

    /// The region edge **K** (bricks per region axis).
    #[inline]
    pub fn region_edge_bricks(&self) -> i32 {
        self.head.region_edge_bricks as i32
    }

    /// Look up a region's `BIDX` entry by its K-brick-grid coord via binary search on the `(z,y,x)` key.
    pub fn region_entry(&self, region_coord: IVec3) -> Option<&VxoRegionDirEntry> {
        let key = (region_coord.z, region_coord.y, region_coord.x);
        self.bidx
            .binary_search_by_key(&key, |e| (e.region_coord[2], e.region_coord[1], e.region_coord[0]))
            .ok()
            .map(|i| &self.bidx[i])
    }

    /// Decompress + parse a region body (sliced from `BRIK` by its `BIDX` entry) into a [`DecodedRegion`]
    /// (§B2.2 step 4). Branches on the EXPLICIT `dir.compression` byte ([`VXO_REGION_STORE`]/[`VXO_REGION_ZSTD`]),
    /// NOT on length equality: STORE borrows the span directly (no decompress); zstd decodes (pure-Rust
    /// `ruzstd`) into a `brik_raw_len` buffer. Either way `parse_region` then COPIES the entry/palette/index
    /// fields out via unaligned reads (the region bytes aren't guaranteed 4/32-aligned) — see `decode_region_span`.
    pub fn decode_region(&self, dir: &VxoRegionDirEntry) -> anyhow::Result<DecodedRegion> {
        let start = dir.brik_offset as usize;
        let end = start + dir.brik_comp_len as usize;
        anyhow::ensure!(end <= self.brik.len(), "vxo: region body overruns BRIK chunk");
        // The compressed region span sliced out of the (in-RAM) BRIK body, decoded via the shared SSOT below
        // (the streamed `VxoSource` feeds the SAME function a slice of its mmap, §B2.2).
        decode_region_span(&self.brik[start..end], dir)
    }
}

/// Decode ONE region's compressed byte SPAN (`comp` = the exact `[brik_offset .. +brik_comp_len)` slice the
/// `BIDX` entry addresses) into a [`DecodedRegion`] (`VXO_FORMAT.md` §B2.2 step 4) — the SHARED decode SSOT
/// for BOTH the full-file [`VxoFile::decode_region`] (in-RAM `BRIK` slice) and the streamed
/// [`super::source::VxoSource`] (an mmap slice). Branches on the EXPLICIT `dir.compression` byte
/// ([`VXO_REGION_STORE`]/[`VXO_REGION_ZSTD`]), NOT on length equality: STORE borrows the slice (no decompress);
/// zstd decodes (pure-Rust `ruzstd`) into a `brik_raw_len` buffer. `parse_region` then COPIES the region's
/// entry table + palette/index blobs out of that slice via UNALIGNED reads (`pod_read_unaligned`/`u32_prefix`
/// into owned `Vec`s) — region bodies are concatenated without inter-region padding so the slice isn't
/// guaranteed POD-aligned; it is NOT a zero-copy in-place cast. Verifies the region header's coord matches the
/// `BIDX` key. Keeping this a free function over a `&[u8]` means the loader never copies the whole `BRIK`.
pub fn decode_region_span(comp: &[u8], dir: &VxoRegionDirEntry) -> anyhow::Result<DecodedRegion> {
    let expected = IVec3::new(dir.region_coord[0], dir.region_coord[1], dir.region_coord[2]);
    let raw: std::borrow::Cow<[u8]> = match dir.compression {
        VXO_REGION_STORE => std::borrow::Cow::Borrowed(comp),
        VXO_REGION_ZSTD => std::borrow::Cow::Owned(zstd_decompress(comp, dir.brik_raw_len as usize)?),
        other => anyhow::bail!("vxo: region has unknown compression code {other} (expected 0=STORE, 1=zstd)"),
    };
    parse_region(&raw, expected)
}

impl DecodedRegion {
    /// Find a brick entry in this region by its absolute LOD0 brick coord (the encoder sorted entries by
    /// `(z,y,x)`, so a binary search resolves it).
    pub fn entry(&self, brick_coord: IVec3) -> Option<&VxoBrickEntry> {
        let key = (brick_coord.z, brick_coord.y, brick_coord.x);
        self.entries
            .binary_search_by_key(&key, |e| (e.brick_coord[2], e.brick_coord[1], e.brick_coord[0]))
            .ok()
            .map(|i| &self.entries[i])
    }

    /// Decode brick `entry` to an in-RAM [`Brick`] (§B2.2 step 6): a uniform entry → `Brick::uniform`; a
    /// dense entry → decode the 8³ core via `decode_paletted_cell` (the `gpu.rs` SSOT) → `Brick::from_voxels`.
    /// Bit-identical to a live brick, since the disk stores the R2b core triple verbatim.
    pub fn brick(&self, entry: &VxoBrickEntry) -> Brick {
        self.brick_remapped(entry, 0)
    }

    /// Decode brick `entry` to a [`Brick`], SHIFTING every SOLID voxel's [`BlockId`] by `block_shift` (AIR
    /// stays AIR) — the merge remap (§B2.4): when several `.vxo` assets concatenate into ONE world brick map,
    /// each asset's local `BlockId(b)` (`b ≥ 1`) must become merged id `b + block_shift` so two assets'
    /// `BlockId(5)` don't collide. `block_shift = 0` is the identity (the verbatim [`Self::brick`] path), so a
    /// single-asset load is bit-identical to a live brick (the round-trip gate). The shift is applied at decode
    /// over the `8³` core (uniform: its single id; dense: each cell), the cheap "+base add" §B2.4 prescribes —
    /// keeping the offset+rebase the [`super::source::MergedSource`] SSOT (this is just the per-cell arithmetic).
    pub fn brick_remapped(&self, entry: &VxoBrickEntry, block_shift: u16) -> Brick {
        if entry.is_uniform() {
            // A stored uniform brick is always SOLID (empty bricks are never written), so the shift always applies.
            return Brick::uniform(BlockId(entry.uniform_block_raw() + block_shift));
        }
        let pb = entry.palette_off as usize;
        let off = entry.dense_index_off() as usize;
        // The remaining palette suffix suffices — the decode only indexes the ≤k entries this brick uses
        // (the index width bounds it), mirroring `gpu.rs`'s `brick_palettes_u16_from`.
        let palette: Vec<u16> = self.palette_blob[pb..].iter().map(|&x| x as u16).collect();
        let index = &self.index_blob[off..];
        let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                for x in 0..BRICK_EDGE {
                    let i = voxel_index(x, y, z);
                    let raw = decode_paletted_cell(&palette, entry.index_bits, index, i);
                    // Shift only SOLID ids into the merged palette range; AIR (0) stays 0 (mirrors
                    // `gallery::merge_brickmap_into`'s per-voxel remap so the merge SSOT is one rule).
                    voxels[i] = if raw == BlockId::AIR.0 { BlockId::AIR } else { BlockId(raw + block_shift) };
                }
            }
        }
        Brick::from_voxels(voxels)
    }
}

/// Parse the `HEAD` body → `(VxoHead, name)` (§B1.1). `pub(crate)` so the streamed [`super::source::VxoSource`]
/// reuses the EXACT chunk parsers (one SSOT for the framing).
pub(crate) fn parse_head(body: &[u8]) -> anyhow::Result<(VxoHead, String)> {
    let n = std::mem::size_of::<VxoHead>();
    anyhow::ensure!(body.len() >= n, "vxo: HEAD body too short");
    let head: VxoHead = pod_read_unaligned(&body[0..n]);
    let name_len = head.name_len as usize;
    anyhow::ensure!(n + name_len <= body.len(), "vxo: HEAD name overruns the body");
    let name = String::from_utf8_lossy(&body[n..n + name_len]).into_owned();
    Ok((head, name))
}

/// Parse the `MATL` body → a rebuilt [`BlockRegistry`] (§B1.2). `material_count: u32` + `_pad` then the
/// `VxoMaterial` table. `pub(crate)` for the streamed [`super::source::VxoSource`] (shared parser).
pub(crate) fn parse_matl(body: &[u8]) -> anyhow::Result<BlockRegistry> {
    anyhow::ensure!(body.len() >= 8, "vxo: MATL body too short");
    let count = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
    let table = &body[8..];
    let stride = std::mem::size_of::<VxoMaterial>();
    anyhow::ensure!(table.len() >= count * stride, "vxo: MATL table truncated ({count} entries)");
    let mats = cast_prefix::<VxoMaterial>(table, count);
    Ok(BlockRegistry::from_vxo_matl(&mats))
}

/// Parse the `BIDX` body → the region directory (§B1.5). `entry_count: u32` + `_pad` then the
/// `VxoRegionDirEntry` table. `pub(crate)` for the streamed [`super::source::VxoSource`] (shared parser).
pub(crate) fn parse_bidx(body: &[u8]) -> anyhow::Result<Vec<VxoRegionDirEntry>> {
    anyhow::ensure!(body.len() >= 8, "vxo: BIDX body too short");
    let count = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
    let table = &body[8..];
    let stride = std::mem::size_of::<VxoRegionDirEntry>();
    anyhow::ensure!(table.len() >= count * stride, "vxo: BIDX table truncated ({count} entries)");
    Ok(cast_prefix::<VxoRegionDirEntry>(table, count))
}

/// Parse a decompressed region body → a [`DecodedRegion`] (§B1.3). Verifies the header's region coord against
/// the `BIDX` key.
fn parse_region(raw: &[u8], expected_coord: IVec3) -> anyhow::Result<DecodedRegion> {
    let hn = std::mem::size_of::<VxoRegionHeader>();
    anyhow::ensure!(raw.len() >= hn, "vxo: region body shorter than its header");
    let header: VxoRegionHeader = pod_read_unaligned(&raw[0..hn]);
    let coord = IVec3::new(header.region_coord[0], header.region_coord[1], header.region_coord[2]);
    anyhow::ensure!(
        coord == expected_coord,
        "vxo: region header coord {coord:?} != BIDX key {expected_coord:?} (corrupt)"
    );
    let n = header.brick_count as usize;
    let estride = std::mem::size_of::<VxoBrickEntry>();
    let entries_end = hn + n * estride;
    let palette_end = entries_end + header.palette_u32 as usize * 4;
    let index_end = palette_end + header.index_u32 as usize * 4;
    anyhow::ensure!(index_end <= raw.len(), "vxo: region body truncated (need {index_end}, have {})", raw.len());

    let entries = cast_prefix::<VxoBrickEntry>(&raw[hn..entries_end], n);
    let palette_blob: Vec<u32> = u32_prefix(&raw[entries_end..palette_end]);
    let index_blob: Vec<u32> = u32_prefix(&raw[palette_end..index_end]);
    Ok(DecodedRegion { region_coord: coord, entries, palette_blob, index_blob })
}

/// Read the first `count` POD `T`s out of `bytes` as an owned `Vec<T>` via UNALIGNED reads. Region bodies are
/// concatenated without inter-region padding (and a zstd-decompressed buffer's start alignment isn't
/// guaranteed), so a borrowed `cast_slice` could trip `bytemuck`'s alignment assert — `pod_read_unaligned`
/// per element is alignment-agnostic and copies into an aligned `Vec`. The slices are small (entries/dir),
/// so the copy is cheap.
fn cast_prefix<T: Pod>(bytes: &[u8], count: usize) -> Vec<T> {
    let sz = std::mem::size_of::<T>();
    (0..count).map(|i| bytemuck::pod_read_unaligned::<T>(&bytes[i * sz..i * sz + sz])).collect()
}

/// Read a byte slice (a `4·k`-length region) as a `Vec<u32>` (little-endian; the slice may not be 4-aligned
/// after a zstd decompress into a fresh `Vec`, so go through `from_le_bytes` rather than `cast_slice`).
fn u32_prefix(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// zstd-decompress `comp` into a `raw_len`-byte buffer (§B1.9) via PURE-RUST `ruzstd` — the runtime decode path
/// pulls NO C toolchain (matching the project's ktx2/`ruzstd` "no C toolchain" discipline; the C `zstd` crate is
/// offline-encode-only, behind `vxo-encode`). `ruzstd` decodes standard zstd frames, so it reads exactly what the
/// encoder's `zstd::bulk::compress` produces. `raw_len` preallocates the output (a size hint, not a hard bound).
fn zstd_decompress(comp: &[u8], raw_len: usize) -> anyhow::Result<Vec<u8>> {
    use std::io::Read;
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(comp)
        .map_err(|e| anyhow::anyhow!("vxo: ruzstd init: {e}"))?;
    let mut out = Vec::with_capacity(raw_len);
    decoder.read_to_end(&mut out).map_err(|e| anyhow::anyhow!("vxo: ruzstd decode: {e}"))?;
    Ok(out)
}
