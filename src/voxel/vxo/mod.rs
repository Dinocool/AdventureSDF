//! The `.vxo` native voxel asset format — **Phase B-i** (`docs/VXO_FORMAT.md` §B1).
//!
//! `.vxo` is the engine-owned, region-streamed on-disk form of a voxel scene (`.vox` becomes import-only).
//! B-i ships three pieces:
//!
//! * [`format`] — the POD record types + constants (magic `VXO1`, RIFF-style tagged chunks, the HEAD/MATL/
//!   BIDX/BRIK records) with the exact byte layouts the spec pins.
//! * [`writer`] — the offline [`writer::write_vxo`] encoder: region-bucket a [`BrickMap`](super::brickmap),
//!   per brick `encode_paletted` its 8³ core (R1 uniform-collapse / R2b dense / R3 intra-region dedup), emit
//!   the region directory + per-region STORE/zstd bodies.
//! * [`reader`] — the full-file [`reader::VxoFile`] reader: parse the chunks, decode a region → a
//!   [`reader::DecodedRegion`], decode a brick entry → a bit-identical [`Brick`](super::brickmap::Brick).
//! * [`source`] — **Phase B-ii**: the region-STREAMED [`source::VxoSource`] (an mmap'd file + a byte-budgeted
//!   decoded-region LRU implementing [`BrickSource`](super::source::BrickSource)) + the merged-gallery
//!   [`source::MergedSource`], feeding the EXISTING residency demand path.
//!
//! The SVDAG `BRIK` variant (B3) is OUT of scope. The round-trip acceptance gate (`VXO_FORMAT.md` §B2.8 gate
//! 2) lives in `tests`; the B-ii streamed-source gates (round-trip via the streamed source, classify parity,
//! LRU budget/eviction) live in [`source`]'s test module.

pub mod format;
pub mod reader;
pub mod source;
pub mod spill;
pub mod writer;

pub use reader::{DecodedRegion, VxoFile};
pub use source::{MergedSource, VxoSource};
pub use spill::{RegionSpillPool, assemble_base, spill_voxel, windowed_coarse};
pub use writer::{
    VxoCompression, VxoHeadParams, VxoStreamWriter, build_coarse_pyramid, drive_coarse_lods, region_of_brick,
    write_vxo,
};

#[cfg(test)]
mod tests;
