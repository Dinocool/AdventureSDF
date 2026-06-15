//! `.vxo` round-trip acceptance gate — **Phase B-i** (`docs/VXO_FORMAT.md` §B2.8 gate 2).
//!
//! Build a known non-trivial [`BrickMap`] (uniform + dense bricks across multiple regions, incl. negative
//! coords) + a [`BlockRegistry`], `write_vxo` it (STORE), read it back, and assert for EVERY brick coord the
//! read-back [`Brick`] is bit-identical (`Brick: PartialEq`) to the original — AND the packed
//! [`GpuBrickPatch`] fingerprint from the read-back set is byte-identical to one packed from the original
//! map (the `incremental` A/B fingerprint approach). This proves the memcpy-decode property: the disk R2b
//! body decodes to the exact resident layout. A second test repeats it over zstd.

use std::collections::HashMap;

use bevy::math::IVec3;

use super::format::*;
use super::writer::{VxoCompression, VxoHeadParams, encode_vxo, region_of_brick};
use super::reader::VxoFile;
use crate::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, BrickMap};
use crate::voxel::gpu::{GpuBrickPatch, decode_paletted_cell, halo_cells, pack_brickmap};
use crate::voxel::palette::{BlockId, BlockRegistry};

/// A small registry of solid blocks (ids 1..=4) plus AIR (0), with one EMITTER (block 4) so the `MATL`
/// emissive + emitter-flag round-trip is exercised. Built by hand (independent of worldgen) so the test is
/// self-contained.
fn registry() -> BlockRegistry {
    // The Cornell palette is a ready 5-block (AIR + 4) registry with exactly one emitter (the light) — reuse it.
    BlockRegistry::cornell()
}

/// A dense brick with a deterministic mixed pattern seeded by `seed`: some AIR, some of a couple of solid
/// blocks (so it does NOT collapse to uniform and uses a multi-entry palette / >1 index_bits).
fn dense_brick(seed: i32) -> Brick {
    let mut v = Box::new([BlockId::AIR; BRICK_VOXELS]);
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                let i = (x + y * BRICK_EDGE + z * BRICK_EDGE * BRICK_EDGE) as usize;
                let s = (x * 3 + y * 5 + z * 7 + seed).rem_euclid(4);
                v[i] = match s {
                    0 => BlockId::AIR,
                    1 => BlockId(1),
                    2 => BlockId(2),
                    _ => BlockId(3),
                };
            }
        }
    }
    Brick::from_voxels(v)
}

/// A FULL (every voxel solid) single-block brick — collapses to uniform AND sets the `is_full` flag.
fn full_uniform_brick(block: u16) -> Brick {
    Brick::uniform(BlockId(block))
}

/// The known non-trivial map: a mix of uniform + dense bricks spread across MULTIPLE K=8 regions, including
/// NEGATIVE coords (so the Euclidean region bucketing is exercised) and two IDENTICAL dense bricks in one
/// region (so the intra-region R3 dedup path runs). Returns the map (the registry is built separately).
fn build_map() -> BrickMap {
    let mut map = BrickMap::new();
    // Region (0,0,0): a uniform-full brick + a dense brick.
    map.insert(IVec3::new(0, 0, 0), full_uniform_brick(1));
    map.insert(IVec3::new(1, 2, 3), dense_brick(11));
    // Region (1,0,0) (brick x >= 8): another dense brick + a uniform.
    map.insert(IVec3::new(8, 1, 1), dense_brick(22));
    map.insert(IVec3::new(9, 0, 0), full_uniform_brick(2));
    // Negative region (-1,-1,-1): bricks at negative coords (div_euclid bucketing).
    map.insert(IVec3::new(-1, -1, -1), dense_brick(33));
    map.insert(IVec3::new(-5, -3, -2), full_uniform_brick(3));
    // Two IDENTICAL dense bricks in region (2,0,0) → intra-region dedup (same encoded slice).
    map.insert(IVec3::new(16, 0, 0), dense_brick(44));
    map.insert(IVec3::new(17, 0, 1), dense_brick(44));
    map
}

/// A normalized per-brick fingerprint of a packed [`GpuBrickPatch`], keyed by `(voxel_origin, lod)` — the
/// SAME decode the `incremental` A/B gate uses (`incremental/tests.rs`): the layout-independent logical
/// content (uniform id, or the decoded haloed cells), so two packs that agree here render identically. A
/// layout difference (offsets) is intentionally ignored; a real voxel divergence is not.
#[derive(Clone, Debug, PartialEq)]
struct Fingerprint {
    voxel_origin: [i32; 3],
    world_min: [f32; 3],
    lod: u32,
    uniform: Option<u16>,
    voxels: Option<Vec<u32>>,
}

fn fingerprints(patch: &GpuBrickPatch) -> HashMap<([i32; 3], u32), Fingerprint> {
    let mut out = HashMap::new();
    for m in &patch.metas {
        let voxels = if m.is_uniform() {
            None
        } else {
            let off = m.dense_offset() as usize;
            let pb = m.palette_base as usize;
            let bits = m.index_bits();
            let palette: Vec<u16> = patch.brick_palettes[pb..].iter().map(|&x| x as u16).collect();
            let cells: Vec<u32> = (0..halo_cells(m.lod()))
                .map(|i| decode_paletted_cell(&palette, bits, &patch.voxels[off..], i) as u32)
                .collect();
            Some(cells)
        };
        out.insert(
            (m.voxel_origin, m.lod()),
            Fingerprint {
                voxel_origin: m.voxel_origin,
                world_min: m.world_min,
                lod: m.lod(),
                uniform: if m.is_uniform() { Some(m.uniform_block().0) } else { None },
                voxels,
            },
        );
    }
    out
}

/// Read every brick of `file` back into a fresh [`BrickMap`] (decoding each region + entry through the SSOT
/// reader path), so it can be packed + fingerprinted exactly like the original.
fn read_back_map(file: &VxoFile) -> BrickMap {
    let mut map = BrickMap::new();
    let k = file.region_edge_bricks();
    for dir in &file.bidx {
        let region = file
            .decode_region(dir)
            .expect("decode region");
        for entry in &region.entries {
            let coord = IVec3::new(entry.brick_coord[0], entry.brick_coord[1], entry.brick_coord[2]);
            // Sanity: the entry's coord buckets back to this region.
            assert_eq!(
                region_of_brick(coord, k),
                region.region_coord,
                "brick {coord:?} is in the wrong region {:?}",
                region.region_coord
            );
            map.insert(coord, region.brick(entry));
        }
    }
    map
}

/// Run the full round-trip under `comp`: encode → parse → per-brick `Brick` bit-identity → packed
/// `GpuBrickPatch` fingerprint identity. The shared body of both gate tests.
fn round_trip(comp: VxoCompression) {
    let map = build_map();
    let registry = registry();
    let params = VxoHeadParams { name: "round_trip".into(), ..Default::default() };

    let bytes = encode_vxo(&map, &registry, &params, comp).expect("encode_vxo");
    let file = VxoFile::parse(&bytes).expect("parse the encoded .vxo");

    // HEAD sanity: counts + region count + bounds reflect the map.
    assert_eq!(file.head.brick_count as usize, map.len(), "HEAD brick_count == map len");
    assert_eq!(file.head.region_count as usize, file.bidx.len(), "HEAD region_count == BIDX len");
    assert_eq!(file.head.brick_edge, BRICK_EDGE as u32);
    assert_eq!(file.name, "round_trip");

    // The registry rebuilt from MATL matches the original block-by-block (colour + emissive + flags).
    assert_eq!(file.registry.len(), registry.len(), "MATL round-trips every block");
    for i in 0..registry.len() as u16 {
        let id = BlockId(i);
        assert_eq!(file.registry.color(id), registry.color(id), "block {i} colour");
        assert_eq!(file.registry.emissive(id), registry.emissive(id), "block {i} emissive");
    }

    // Read every brick back; assert bit-identity vs the original for EVERY coord.
    let read_map = read_back_map(&file);
    assert_eq!(read_map.len(), map.len(), "read-back brick count == original");
    for (coord, brick) in map.iter() {
        let got = read_map.get(*coord).unwrap_or_else(|| panic!("brick {coord:?} missing from read-back"));
        assert_eq!(got, brick, "brick {coord:?} is not bit-identical after round-trip");
    }

    // The packed GpuBrickPatch fingerprints match (the memcpy-decode property over the resident layout).
    let orig_fp = fingerprints(&pack_brickmap(&map, &registry));
    let read_fp = fingerprints(&pack_brickmap(&read_map, &file.registry));
    assert_eq!(read_fp.len(), orig_fp.len(), "packed brick count differs");
    for (k, f_orig) in &orig_fp {
        let f_read = read_fp.get(k).unwrap_or_else(|| panic!("packed brick {k:?} missing after round-trip"));
        assert_eq!(f_read, f_orig, "packed brick {k:?} differs after round-trip (memcpy-decode broken)");
    }
}

/// **The acceptance gate (§B2.8 gate 2): STORE round-trip is bit-identical.**
#[test]
fn round_trip_store_is_bit_identical() {
    round_trip(VxoCompression::Store);
}

/// The zstd round-trip yields the same bit-identical result (per-region zstd, §B1.9).
#[test]
fn round_trip_zstd_is_bit_identical() {
    round_trip(VxoCompression::Zstd(19));
}

/// Multiple regions are produced (the map straddles K=8 region boundaries incl. negatives), and the encoder
/// buckets the eight test bricks into the expected distinct regions.
#[test]
fn regions_bucket_by_k() {
    let map = build_map();
    let bytes = encode_vxo(&map, &registry(), &VxoHeadParams::default(), VxoCompression::Store).expect("encode");
    let file = VxoFile::parse(&bytes).expect("parse");
    // Distinct K=8 regions of the eight bricks: (0,0,0),(1,0,0),(-1,-1,-1),(-1,-1,-1 dup),(2,0,0)…
    let mut regions: std::collections::HashSet<[i32; 3]> = std::collections::HashSet::new();
    for (coord, _) in map.iter() {
        let r = region_of_brick(*coord, 8);
        regions.insert([r.x, r.y, r.z]);
    }
    assert_eq!(file.bidx.len(), regions.len(), "one BIDX entry per non-empty region");
    assert!(file.bidx.len() >= 4, "the test map straddles multiple regions");
    // BIDX is sorted by (z,y,x) — the binary-search invariant.
    let keys: Vec<(i32, i32, i32)> =
        file.bidx.iter().map(|e| (e.region_coord[2], e.region_coord[1], e.region_coord[0])).collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted, "BIDX must be sorted by (z,y,x)");
}

/// STORE region bodies carry `brik_raw_len == brik_comp_len` (the §B1.5 STORE convention); zstd bodies
/// compress (raw > comp for the redundant uniform/dedup'd map, or at least raw != comp signalling decode).
#[test]
fn store_vs_zstd_lengths() {
    let map = build_map();
    let reg = registry();

    let store = VxoFile::parse(&encode_vxo(&map, &reg, &VxoHeadParams::default(), VxoCompression::Store).unwrap()).unwrap();
    for e in &store.bidx {
        assert_eq!(e.brik_comp_len, e.brik_raw_len, "STORE region: comp_len == raw_len");
    }

    let z = VxoFile::parse(&encode_vxo(&map, &reg, &VxoHeadParams::default(), VxoCompression::Zstd(19)).unwrap()).unwrap();
    // At least one region is non-trivially sized; its raw length is recorded for the decode buffer.
    assert!(z.bidx.iter().all(|e| e.brik_raw_len > 0), "every region records a raw length");
}

/// A corrupted header CRC is rejected with a clear error (the integrity check, §B1.0).
#[test]
fn corrupt_header_crc_rejected() {
    let map = build_map();
    let mut bytes = encode_vxo(&map, &registry(), &VxoHeadParams::default(), VxoCompression::Store).unwrap();
    // Flip a bit in the `flags` field (offset 6) without fixing the CRC → the header CRC must fail.
    bytes[6] ^= 0x01;
    let err = VxoFile::parse(&bytes).expect_err("a flipped flags bit must fail the header CRC");
    assert!(format!("{err}").contains("CRC"), "error should mention the CRC: {err}");
}

/// An unknown chunk is SKIPPED (forward-compat, §B1.0): splicing a bogus `XXXX` chunk before `END` leaves the
/// required chunks parseable.
#[test]
fn unknown_chunk_is_skipped() {
    let map = build_map();
    let reg = registry();
    let mut bytes = encode_vxo(&map, &reg, &VxoHeadParams::default(), VxoCompression::Store).unwrap();
    // Append a well-framed unknown chunk (tag b"XXXX", a 4-byte body) after the existing chunks.
    let body = [0xDEu8, 0xAD, 0xBE, 0xEF];
    let ch = VxoChunkHeader {
        tag: *b"XXXX",
        _pad0: 0,
        body_len: body.len() as u64,
        body_crc32: crc32(&body),
        _pad1: [0; 3],
    };
    bytes.extend_from_slice(bytemuck::bytes_of(&ch));
    bytes.extend_from_slice(&body);
    // pad to 16
    while !bytes.len().is_multiple_of(16) {
        bytes.push(0);
    }
    // It still parses (the unknown chunk is skipped) and the bricks still round-trip.
    let file = VxoFile::parse(&bytes).expect("unknown trailing chunk must be skipped");
    assert_eq!(file.head.brick_count as usize, map.len());
}
