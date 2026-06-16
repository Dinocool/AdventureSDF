//! The on-disk `.vxo` format types + constants — **Phase B-i** (`docs/VXO_FORMAT.md` §B1.0–§B1.5).
//!
//! Every record here is a fixed-layout `#[repr(C)]` POD (`Pod + Zeroable`) so a chunk body is a flat,
//! self-describing run of records addressable by byte offset and `bytemuck::cast_slice`-able from an mmap
//! (the NanoVDB discipline, `VXO_FORMAT.md` §0.2). The byte layouts are pinned by the spec; the
//! `#[test]` size asserts below are the SSOT guard that a record never silently grows.
//!
//! This is the FORMAT only — the encoder lives in [`super::writer`], the full-file reader in
//! [`super::reader`]. The runtime depends ONLY on these three modules (`dot_vox`/`gltf`/`image` stay
//! offline-import deps).

use bytemuck::{Pod, Zeroable};

/// The file magic — first 4 bytes of every `.vxo` file (`VXO1`). Bump the trailing digit only on a
/// breaking framing change.
pub const VXO_MAGIC: [u8; 4] = *b"VXO1";

/// The file `format_version` (`VXO_FORMAT.md` §B1.0). Starts at `1`; bump only on a breaking framing change.
pub const VXO_FORMAT_VERSION: u16 = 1;

/// `HEAD` chunk per-schema version (`VXO_FORMAT.md` §B1.1) — independent of the file `format_version`.
pub const VXO_HEAD_VERSION: u16 = 2;

/// File-header `flags` bit0: the whole file is little-endian (always 1 — we target LE x86/Vulkan).
pub const VXO_FLAG_LITTLE_ENDIAN: u16 = 1 << 0;
/// File-header `flags` bit1: `BRIK` region bodies are SVDAG-encoded (B3 — NOT produced by B-i).
pub const VXO_FLAG_SVDAG: u16 = 1 << 1;

/// The default region granularity **K** (`VXO_FORMAT.md` §B1.4): a region is `K×K×K` LOD0 bricks. Must be a
/// power of two and align to the residency clipmap; `K = 8` ⇒ a 512-brick region.
pub const DEFAULT_REGION_EDGE_BRICKS: u32 = 8;

// ---- chunk tags (ASCII, 4 bytes) -----------------------------------------------------------------

/// `HEAD` chunk tag — self-describing geometry + identity (REQUIRED, first).
pub const TAG_HEAD: [u8; 4] = *b"HEAD";
/// `MATL` chunk tag — material table keyed by `u16` BlockId (REQUIRED).
pub const TAG_MATL: [u8; 4] = *b"MATL";
/// `BIDX` chunk tag — the sorted region directory (REQUIRED).
pub const TAG_BIDX: [u8; 4] = *b"BIDX";
/// `BRIK` chunk tag — the concatenation of all region-chunk bodies (REQUIRED).
pub const TAG_BRIK: [u8; 4] = *b"BRIK";
/// `LODS` chunk tag — the baked coarse-LOD pyramid (OPTIONAL; `VXO_FORMAT.md` §B1.7).
pub const TAG_LODS: [u8; 4] = *b"LODS";
/// `END ` sentinel chunk tag (optional).
pub const TAG_END: [u8; 4] = *b"END ";

/// `VxoBrickEntry.flags` bit0: the brick is on the asset's air-exposed surface (LITE / light-gather hint).
pub const BRICK_FLAG_SURFACE: u8 = 1 << 0;
/// `VxoBrickEntry.flags` bit1: the brick is FULLY SOLID (every voxel solid). The conservative-enclosed-cull
/// (`classify`) reads this WITHOUT decoding voxels — baked by the encoder from [`crate::voxel::brickmap::Brick::is_full`].
pub const BRICK_FLAG_FULL: u8 = 1 << 1;
/// `VxoBrickEntry.flags` bit2: the brick is UNIFORM (R1 collapse) — its single [`crate::voxel::palette::BlockId`]
/// rides in the LOW 16 bits of `index_off`, and it emits NO palette/index bytes.
///
/// **A4.1 alignment — a DEDICATED discriminant bit, NOT bit-31 of `index_off`.** This mirrors the live VRAM
/// SSOT `gpu.rs`'s `META_FLAG_UNIFORM` (a dedicated `GpuBrickMeta.flags` word): `gpu.rs` RETIRED the old
/// `BRICK_UNIFORM_FLAG = 1<<31` of `voxel_offset` precisely because it silently capped a real dense offset at
/// `< 2^31` (a silent-corruption trap the moment the arena passes `2^31` `u32`s). We make the disk form match:
/// `index_off` uses the FULL `u32` range for a dense brick's region-local index-blob offset, and the uniform
/// marker lives here in `flags`. (Surface = bit0, full = bit1 are taken, so uniform takes bit2.)
pub const BRICK_FLAG_UNIFORM: u8 = 1 << 2;

/// The 16-byte fixed file header (`VXO_FORMAT.md` §B1.0). Bytes: `magic[4] + format_version u16 + flags u16
/// + header_crc32 u32 + _reserved u32`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct VxoFileHeader {
    /// `b"VXO1"`.
    pub magic: [u8; 4],
    /// Format/framing version (`VXO_FORMAT_VERSION`).
    pub format_version: u16,
    /// File-level flags (`VXO_FLAG_*`).
    pub flags: u16,
    /// CRC32 of the 8 bytes above (`magic + format_version + flags`). Loader rejects on mismatch.
    pub header_crc32: u32,
    /// Reserved, 0.
    pub _reserved: u32,
}

/// The chunk-framing header preceding every chunk body (`VXO_FORMAT.md` §B1.0).
///
/// **Deviation from the spec's "16 B" claim** (documented): the spec's own field list — `tag [u8;4]` +
/// `body_len u64` + `body_crc32 u32` + `_pad u32` — sums to 20 logical bytes, and a real `u64` `body_len`
/// forces 8-byte alignment, so the natural `#[repr(C)]` size is 24, NOT 16. The spec's hard requirement is
/// only that **bodies start 16-aligned for mmap + `bytemuck`** (§B1.0/§0.2). We honour THAT by padding the
/// header to a clean **32 bytes** (the next 16-multiple) with all spec fields kept at their real types — so
/// after the 16-byte file header, every chunk header (32, a 16-multiple) + its 16-padded body keeps the
/// running offset 16-aligned. (The alternative, shrinking `body_len` to `u32`, would cap a chunk at 4 GiB —
/// unacceptable for a Bistro-scale BRIK; keeping `u64` + a 32-byte header is the robust choice.)
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct VxoChunkHeader {
    /// ASCII tag (`TAG_*`).
    pub tag: [u8; 4],
    /// Pad so `body_len` is 8-aligned.
    pub _pad0: u32,
    /// Byte length of `body` (NOT incl. this header, NOT incl. the trailing 16-B body alignment pad).
    pub body_len: u64,
    /// CRC32 of `body` (0 = skip verify).
    pub body_crc32: u32,
    /// Pad → 32-byte header (a 16-multiple) so bodies start 16-aligned.
    pub _pad1: [u32; 3],
}

/// `HEAD` — self-describing geometry + identity (`VXO_FORMAT.md` §B1.1). The flat POD prefix; the variable
/// UTF-8 `name` (length `name_len`) follows in-body, padded to 4 bytes. 80 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct VxoHead {
    /// Per-chunk schema version (`VXO_HEAD_VERSION`).
    pub head_version: u16,
    /// Pad, 0.
    pub _pad0: u16,
    /// Metres per LOD0 voxel (e.g. `0.05`). SELF-DESCRIBING (§0.4).
    pub voxel_size: f32,
    /// Voxels per brick edge — MUST equal `brickmap::BRICK_EDGE` (8); loader asserts.
    pub brick_edge: u32,
    /// LODS-chunk pyramid depth (0 if no `LODS`); ≤ `brickmap::MAX_LOD`.
    pub max_lod: u32,
    /// Inclusive LOD0 world-VOXEL min corner of the asset's solid extent.
    pub bounds_min: [i32; 3],
    /// Exclusive LOD0 world-VOXEL max corner.
    pub bounds_max: [i32; 3],
    /// The asset PIVOT in LOD0 world-voxel coords (recorded, not baked). `(0,0,0)` for a merge-into-world scene.
    pub anchor_voxel: [i32; 3],
    /// **K** — the region granularity: a region is `K×K×K` bricks. Power of two; default 8.
    pub region_edge_bricks: u32,
    /// Total non-empty LOD0 bricks (load-budget pre-allocation hint).
    pub brick_count: u64,
    /// Number of `BIDX` entries (non-empty regions).
    pub region_count: u32,
    /// Pad, 0.
    pub _pad1: u32,
    /// UTF-8 name byte length (the `name` bytes follow this struct in-body).
    pub name_len: u32,
    /// Pad → 80-byte struct (16-aligned). 0.
    pub _pad2: u32,
}

/// `MATL` — one material per `u16` BlockId (`VXO_FORMAT.md` §B1.2). 48 bytes, 16-aligned. `albedo`/`emissive`
/// mirror `gpu.rs`'s `GpuPaletteColor` so the resident palette buffer is a near-direct copy. Index `i` →
/// `BlockId(i)`; entry 0 is AIR.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct VxoMaterial {
    /// LINEAR RGBA albedo (disk stores linear, unlike `.vox` sRGB).
    pub albedo: [f32; 4],
    /// LINEAR RGB emissive radiance in `.xyz`; `.w` = emissive_strength multiplier (default 1.0).
    pub emissive: [f32; 4],
    /// Reserved-but-present surface roughness (default 1.0).
    pub roughness: f32,
    /// Reserved-but-present metalness (default 0.0).
    pub metallic: f32,
    /// bit0 = tintable; bit1 = emitter (precomputed = any(emissive>0)); rest reserved 0.
    pub flags: u32,
    /// Pad, 0.
    pub _pad: u32,
}

/// `MATL.flags` bit0: the block is tintable (a future per-instance tint may modulate the base colour).
pub const MATL_FLAG_TINTABLE: u32 = 1 << 0;
/// `MATL.flags` bit1: the block is an emitter (precomputed `any(emissive > 0)`).
pub const MATL_FLAG_EMITTER: u32 = 1 << 1;

/// A [`VxoRegionDirEntry::compression`] value: the region body is STORED uncompressed (`bytemuck`-castable in
/// place; `brik_comp_len == brik_raw_len`).
pub const VXO_REGION_STORE: u8 = 0;
/// A [`VxoRegionDirEntry::compression`] value: the region body is a per-region zstd frame (decode into a
/// `brik_raw_len` buffer).
pub const VXO_REGION_ZSTD: u8 = 1;

/// `BIDX` — one sorted region-directory entry (`VXO_FORMAT.md` §B1.5). 48 bytes. The `BIDX` body is
/// `entry_count: u32` + `_pad: u32` then `entry_count × VxoRegionDirEntry`, sorted by `(z, y, x)` so a coord
/// → entry is an `O(log n)` binary search.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct VxoRegionDirEntry {
    /// The K-brick-grid region coord (the search key).
    pub region_coord: [i32; 3],
    /// Bricks in this region (preallocate the decode).
    pub brick_count: u32,
    /// Byte offset of this region's chunk WITHIN the `BRIK` chunk body.
    pub brik_offset: u64,
    /// COMPRESSED byte length of the region chunk (the seek+read span).
    pub brik_comp_len: u32,
    /// DECOMPRESSED byte length (preallocate the zstd output; for STORE, `brik_raw_len == brik_comp_len`).
    pub brik_raw_len: u32,
    /// EXPLICIT per-region compression: [`VXO_REGION_STORE`] (0) or [`VXO_REGION_ZSTD`] (1). The reader branches
    /// on THIS, never on `brik_comp_len == brik_raw_len` — a zstd body can coincidentally compress to exactly its
    /// raw length, so length-equality is an ambiguous (silently-corrupting) discriminant.
    pub compression: u8,
    /// Pad → 48-byte (16-multiple) stride. 0. (`brik_offset`'s `u64` forces 8-byte align, so adding the
    /// `compression` byte to the formerly-32-byte entry rounds the natural `#[repr(C)]` size up; we pad to the
    /// next clean 16-multiple — 48 — so a `[VxoRegionDirEntry]` body stays internally 16-aligned for `bytemuck`.)
    pub _pad: [u8; 15],
}

/// `BRIK` region-chunk header (`VXO_FORMAT.md` §B1.3). 32 bytes. A decompressed region chunk is:
/// `VxoRegionHeader` then `[VxoBrickEntry; brick_count]` then `palette_blob: [u32; palette_u32]` then
/// `index_blob: [u32; index_u32]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct VxoRegionHeader {
    /// The region's coord on the K-brick grid (redundant w/ BIDX key; verified on decode).
    pub region_coord: [i32; 3],
    /// N bricks in this region (all LOD0).
    pub brick_count: u32,
    /// P — length of `palette_blob` in `u32`s (region-local base = 0).
    pub palette_u32: u32,
    /// I — length of `index_blob` in `u32`s (region-local base = 0).
    pub index_u32: u32,
    /// 0 for the base BRIK; LODS regions carry their level.
    pub lod: u32,
    /// Pad → 32 bytes. 0.
    pub _pad: u32,
}

/// `BRIK` per-brick entry (`VXO_FORMAT.md` §B1.3). 32 bytes — one resident brick, decode-ready. R1 uniform
/// OR R2b dense, distinguished by the dedicated [`BRICK_FLAG_UNIFORM`] bit in `flags` (A4.1-aligned — NOT a
/// stolen `index_off` high bit).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct VxoBrickEntry {
    /// LOD0 brick coord (absolute, world grid).
    pub brick_coord: [i32; 3],
    /// UNIFORM ([`BRICK_FLAG_UNIFORM`] set in `flags`): low 16 bits = the uniform BlockId. Else: REGION-LOCAL
    /// `u32` word offset into `index_blob` — the FULL `u32` range (no reserved high bit; A4.1).
    pub index_off: u32,
    /// REGION-LOCAL `u32` word offset into `palette_blob` (dense only; 0 for uniform).
    pub palette_off: u32,
    /// Index bit width ∈ {1,2,4,8,16} (dense only; 0 for uniform).
    pub index_bits: u8,
    /// `k` distinct ids (dense; ≤ 255 — the encoder asserts; 0 for uniform).
    pub palette_len: u8,
    /// `BRICK_FLAG_*` bits (surface / full).
    pub flags: u8,
    /// Pad, 0.
    pub _pad0: u8,
    /// Pad → 32-byte struct. 0. (The struct's max align is 4, so it needs 8 explicit pad bytes after the
    /// four `u8`s to reach the spec's 32-byte stride — two `u32`s.)
    pub _pad1: [u32; 2],
}

impl VxoBrickEntry {
    /// True iff this entry is a collapsed UNIFORM brick — reads the dedicated [`BRICK_FLAG_UNIFORM`] bit
    /// (A4.1: no longer an `index_off` high bit), exactly as `gpu.rs`'s `GpuBrickMeta::is_uniform` reads its
    /// dedicated `flags` word.
    #[inline]
    pub fn is_uniform(&self) -> bool {
        self.flags & BRICK_FLAG_UNIFORM != 0
    }

    /// The single uniform BlockId raw value (low 16 bits of `index_off`). Meaningful only when
    /// [`Self::is_uniform`].
    #[inline]
    pub fn uniform_block_raw(&self) -> u16 {
        (self.index_off & 0xFFFF) as u16
    }

    /// The region-local index-blob word offset of a DENSE brick (the FULL `index_off` — the uniform discriminant
    /// is in `flags`, so `index_off` never aliases). Meaningless for a uniform brick.
    #[inline]
    pub fn dense_index_off(&self) -> u32 {
        self.index_off
    }
}

/// `LODS` body header (`VXO_FORMAT.md` §B1.7). 16 bytes, 16-aligned. The flat POD prefix of the `LODS`
/// chunk body; the `[VxoLodLevel; level_count]` table follows immediately, then each level's BIDX_L + BRIK_L
/// sub-sections at the offsets the table records (all offsets are byte offsets FROM THE LODS BODY START).
///
/// `LODS` is the BAKED coarse-LOD pyramid: one entry per LOD level `L ∈ 1..=max_lod`, where level-`L` bricks
/// are the LOD0 [`BrickMap`](crate::voxel::brickmap::BrickMap) downsampled `L` times through the
/// `source::downsample_brickmap` SSOT — so a baked coarse brick is bit-identical to a demand-synthesized one.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct VxoLodsHeader {
    /// The deepest baked level (== `level_count`; a tiny asset collapses early so `max_lod < brickmap::MAX_LOD`).
    pub max_lod: u32,
    /// Number of [`VxoLodLevel`] entries in the table (one per `L ∈ 1..=max_lod`, so `level_count == max_lod`).
    pub level_count: u32,
    /// Pad → 16-byte header (16-aligned). 0.
    pub _pad: [u32; 2],
}

/// `LODS` per-level table entry (`VXO_FORMAT.md` §B1.7). 32 bytes, 16-aligned. One per baked LOD level; the
/// table sits at the head of the `LODS` body (right after [`VxoLodsHeader`]). Each level reuses the base
/// [`VxoRegionDirEntry`]/[`VxoRegionHeader`]/[`VxoBrickEntry`] records verbatim — only the framing differs.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct VxoLodLevel {
    /// The pyramid LOD this entry describes (`1..=max_lod`; verified == the 1-based table index on decode).
    pub lod: u32,
    /// Non-empty coarse regions at this level (== the level's `BIDX_L` entry count).
    pub region_count: u32,
    /// Byte offset of this level's `BIDX_L` (`[VxoRegionDirEntry; region_count]`, sorted by `(z,y,x)`),
    /// relative to the LODS BODY START.
    pub bidx_off: u64,
    /// Byte offset of this level's `BRIK_L` (the concatenated level-`L` region bodies) relative to the LODS
    /// BODY START. A region's `VxoRegionDirEntry.brik_offset` is relative to THIS base (region-local within the
    /// level's blob — mirroring how the base BIDX `brik_offset` is relative to the BRIK chunk body start).
    pub brik_off: u64,
    /// Byte length of this level's `BRIK_L` blob (the seek bound for the level's region bodies).
    pub brik_len: u64,
}

/// Round `n` UP to the next multiple of 16 — the in-body chunk pad (`VXO_FORMAT.md` §B1.0: bodies are padded
/// to a 16-B multiple with zeros, the pad OUTSIDE `body_len`).
#[inline]
pub fn align16(n: u64) -> u64 {
    (n + 15) & !15
}

/// Standard CRC-32 (IEEE 802.3, the zlib/PNG polynomial `0xEDB88320`, reflected) over `bytes` — the cheap
/// integrity check on the file header (`VXO_FORMAT.md` §B1.0) and each chunk body. A tiny self-contained
/// table-free-at-rest implementation (a lazily-built 256-entry table) so the format pulls no `crc` crate; it
/// is the SSOT both the writer (computes the stored value) and the reader (verifies) call, so they can never
/// disagree on the polynomial / bit order.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut c = Crc32Stream::new();
    c.update(bytes);
    c.finalize()
}

/// The reflected CRC-32 table, built once on first use (256 entries; cheap, deterministic). Shared by the
/// one-shot [`crc32`] and the streaming [`Crc32Stream`] so they can never disagree on polynomial/bit-order.
fn crc32_table() -> &'static [u32; 256] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        let mut i = 0usize;
        while i < 256 {
            let mut c = i as u32;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
                k += 1;
            }
            t[i] = c;
            i += 1;
        }
        t
    })
}

/// A STREAMING CRC-32 over the same reflected IEEE polynomial as [`crc32`] — for chunk bodies too large to
/// hold in RAM (the out-of-core `.vxo` BRIK body, `writer::VxoStreamWriter`). Feed bytes via [`update`](Self::update)
/// in any chunking; [`finalize`](Self::finalize) yields a value IDENTICAL to `crc32(whole_input)`.
pub struct Crc32Stream {
    state: u32,
}

impl Crc32Stream {
    /// A fresh CRC accumulator (pre-final-xor init state `0xFFFFFFFF`).
    pub fn new() -> Self {
        Self { state: 0xFFFF_FFFF }
    }
    /// Fold `bytes` into the running CRC (chunk size irrelevant — the result is split-independent).
    pub fn update(&mut self, bytes: &[u8]) {
        let table = crc32_table();
        for &b in bytes {
            self.state = table[((self.state ^ b as u32) & 0xFF) as usize] ^ (self.state >> 8);
        }
    }
    /// The final CRC-32 (applies the trailing xor); consumes the accumulator.
    pub fn finalize(self) -> u32 {
        self.state ^ 0xFFFF_FFFF
    }
}

impl Default for Crc32Stream {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pinned byte SIZES (`VXO_FORMAT.md`) — the SSOT guard that a record never silently grows (a size
    /// change is a format break the spec must bless). 16-aligned where the spec says so.
    #[test]
    fn record_sizes_match_spec() {
        assert_eq!(std::mem::size_of::<VxoFileHeader>(), 16, "file header is 16 B (§B1.0)");
        // Deviation: a 32-byte (16-multiple) chunk header, not the spec's stated 16 B — see VxoChunkHeader.
        assert_eq!(std::mem::size_of::<VxoChunkHeader>(), 32, "chunk header padded to 32 B (16-multiple)");
        assert_eq!(std::mem::size_of::<VxoChunkHeader>() % 16, 0, "chunk header is a 16-multiple (bodies aligned)");
        assert_eq!(std::mem::size_of::<VxoHead>(), 80, "HEAD POD prefix (§B1.1)");
        assert_eq!(std::mem::size_of::<VxoMaterial>(), 48, "MATL entry is 48 B (§B1.2)");
        assert_eq!(std::mem::size_of::<VxoRegionDirEntry>(), 48, "BIDX entry is 48 B (§B1.5, +compression byte)");
        assert_eq!(std::mem::size_of::<VxoRegionHeader>(), 32, "region header is 32 B (§B1.3)");
        assert_eq!(std::mem::size_of::<VxoBrickEntry>(), 32, "brick entry is 32 B (§B1.3)");
        assert_eq!(std::mem::size_of::<VxoLodsHeader>(), 16, "LODS header is 16 B (§B1.7)");
        assert_eq!(std::mem::size_of::<VxoLodLevel>(), 32, "LODS level entry is 32 B (§B1.7)");
        // The records carry no 16-aligned member, so their natural `#[repr(C)]` align is ≤ 8 (the reader uses
        // unaligned reads / element copies, so this is fine). What matters is the spec's REAL invariant: each
        // record's SIZE is a clean multiple of 16 so a `[T]` body stays internally aligned and the framing
        // offsets line up. Assert that.
        assert_eq!(std::mem::size_of::<VxoMaterial>() % 16, 0, "MATL stride is a 16-multiple");
        assert_eq!(std::mem::size_of::<VxoRegionDirEntry>() % 16, 0, "BIDX stride is a 16-multiple");
        assert_eq!(std::mem::size_of::<VxoBrickEntry>() % 16, 0, "BRIK entry stride is a 16-multiple");
        assert_eq!(std::mem::size_of::<VxoHead>() % 16, 0, "HEAD prefix is a 16-multiple");
        assert_eq!(std::mem::size_of::<VxoLodsHeader>() % 16, 0, "LODS header is a 16-multiple");
        assert_eq!(std::mem::size_of::<VxoLodLevel>() % 16, 0, "LODS level stride is a 16-multiple");
    }

    /// `align16` rounds up to the next 16-multiple (and is a no-op on an already-aligned value).
    #[test]
    fn align16_rounds_up() {
        assert_eq!(align16(0), 0);
        assert_eq!(align16(1), 16);
        assert_eq!(align16(16), 16);
        assert_eq!(align16(17), 32);
        assert_eq!(align16(31), 32);
    }

    /// CRC-32 matches the known IEEE/zlib check value for the ASCII string `"123456789"` (0xCBF43926) — the
    /// canonical test vector, proving the polynomial + reflection + final-xor are the standard ones.
    #[test]
    fn crc32_known_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0, "empty input ⇒ 0 (init ^ final-xor cancel)");
    }

    /// The uniform discriminant lives in `flags` (A4.1), so a DENSE brick's `index_off` may use the FULL `u32`
    /// range (incl. bit 31) WITHOUT being misread as uniform — the old `1<<31` trap is gone.
    #[test]
    fn uniform_flag_accessors() {
        let uni = VxoBrickEntry {
            brick_coord: [1, 2, 3],
            index_off: 7, // the uniform id rides in the low 16 bits; NO high-bit marker
            palette_off: 0,
            index_bits: 0,
            palette_len: 0,
            flags: BRICK_FLAG_FULL | BRICK_FLAG_UNIFORM,
            _pad0: 0,
            _pad1: [0; 2],
        };
        assert!(uni.is_uniform());
        assert_eq!(uni.uniform_block_raw(), 7);

        // A dense brick whose region-local offset HAS bit 31 set: must NOT be misread as uniform (the A4.1 fix).
        let dense_high = VxoBrickEntry {
            brick_coord: [0, 0, 0],
            index_off: 1 << 31,
            palette_off: 56,
            index_bits: 4,
            palette_len: 3,
            flags: BRICK_FLAG_SURFACE,
            _pad0: 0,
            _pad1: [0; 2],
        };
        assert!(!dense_high.is_uniform(), "bit-31-of-index_off must NOT mark uniform (A4.1)");
        assert_eq!(dense_high.dense_index_off(), 1 << 31);

        let dense = VxoBrickEntry {
            brick_coord: [0, 0, 0],
            index_off: 1234,
            palette_off: 56,
            index_bits: 4,
            palette_len: 3,
            flags: BRICK_FLAG_SURFACE,
            _pad0: 0,
            _pad1: [0; 2],
        };
        assert!(!dense.is_uniform());
        assert_eq!(dense.dense_index_off(), 1234);
    }
}
