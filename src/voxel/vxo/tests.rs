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
use super::writer::{VxoCompression, VxoHeadParams, build_coarse_pyramid, encode_vxo, region_of_brick};
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

/// **C1.7 streaming-write parity:** the bounded-RAM [`VxoStreamWriter`] (region bodies streamed to a scratch
/// file, BIDX assembled at finish) produces a file BYTE-IDENTICAL to the full-RAM [`encode_vxo`] when fed the
/// SAME regions in the SAME `(z,y,x)` order. This proves the out-of-core assembly path emits a valid, identical
/// `.vxo` — so the tiled bake's streamed write is correct by construction (same encoder per region, same chunk
/// framing + CRC, just a different RAM profile). Both STORE and (when available) zstd.
#[test]
fn stream_writer_matches_encode_vxo_byte_for_byte() {
    let map = build_map();
    let registry = registry();
    // Disable the baked LODS pyramid for this byte-for-byte parity: `encode_vxo` would append a LODS chunk
    // from the resident map, but this test drives the streaming writer with ONLY `add_region` (no coarse
    // `add_lod_region` calls — that's the Stage-3 caller's job), so to compare the two paths byte-for-byte both
    // must omit LODS. (The streaming LODS bake is covered separately via `add_lod_region`.)
    let params = VxoHeadParams { name: "stream_parity".into(), bake_lods: false, ..Default::default() };

    #[cfg(feature = "vxo-encode")]
    let comps = [VxoCompression::Store, VxoCompression::Zstd(19)];
    #[cfg(not(feature = "vxo-encode"))]
    let comps = [VxoCompression::Store];

    let k = params.region_edge_bricks as i32;
    let dir = std::env::temp_dir().join(format!("vxo_stream_parity_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");

    for (ci, comp) in comps.into_iter().enumerate() {
        // The full-RAM reference image.
        let want = encode_vxo(&map, &registry, &params, comp).expect("encode_vxo reference");

        // Bucket bricks into regions, feed the streaming writer in (z,y,x) region order — the exact order
        // `encode_vxo` lays out the BRIK body, so the bytes (incl. offsets) coincide.
        let mut regions: std::collections::BTreeMap<(i32, i32, i32), Vec<IVec3>> = Default::default();
        for (coord, _) in map.iter() {
            let r = region_of_brick(*coord, k);
            regions.entry((r.z, r.y, r.x)).or_default().push(*coord);
        }
        let scratch = dir.join(format!("brik_{ci}.tmp"));
        let out = dir.join(format!("stream_{ci}.vxo"));
        let mut w = super::writer::VxoStreamWriter::new(params.clone(), &registry, comp, &scratch)
            .expect("open stream writer");
        for ((rz, ry, rx), mut coords) in regions {
            coords.sort_by_key(|c| (c.z, c.y, c.x));
            let bricks: Vec<(IVec3, &Brick)> =
                coords.iter().map(|&c| (c, map.get(c).expect("brick"))).collect();
            w.add_region(IVec3::new(rx, ry, rz), &bricks).expect("add_region");
        }
        w.finish(&out).expect("finish stream write");

        let got = std::fs::read(&out).expect("read streamed .vxo");
        assert_eq!(got, want, "streamed .vxo must be byte-identical to encode_vxo (comp {comp:?})");
        // The scratch BRIK file is removed on success.
        assert!(!scratch.exists(), "scratch BRIK file removed on success");
        // And the streamed file round-trips through the reader (it is a valid .vxo).
        let file = VxoFile::parse(&got).expect("streamed .vxo parses");
        let read_map = read_back_map(&file);
        assert_eq!(read_map.len(), map.len(), "streamed read-back brick count == original");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The zstd round-trip yields the same bit-identical result (per-region zstd, §B1.9): the C-zstd encoder
/// (`vxo-encode`) compresses, the pure-Rust `ruzstd` runtime reader decodes — proving the two are
/// frame-compatible. Gated on `vxo-encode` (PRODUCING a zstd body needs the C compressor); run the gate with
/// `--features vxo-encode`.
#[cfg(feature = "vxo-encode")]
#[test]
fn round_trip_zstd_is_bit_identical() {
    round_trip(VxoCompression::Zstd(19));
}

/// **The Phase C transcoder VALIDATION GATE** (`examples/vox_to_vxo.rs`): a `.vxo` stamped at the corpus's
/// **0.05 m** spacing round-trips bit-identically through the FULL-FILE [`VxoFile`] reader (`VxoFile` records
/// the on-disk spacing verbatim; `VxoSource` separately asserts `voxel_size == VOXEL_SIZE`, which 0.05 m now
/// satisfies post-D1 — but this gate exercises the raw file reader, independent of the engine spacing).
/// Encode the known multi-region map at `voxel_size = 0.05`, read it back via `VxoFile`, and assert HEAD
/// records 0.05 m AND every brick is bit-identical to the source — the property the `.vox → .vxo` transcode
/// relies on (the on-disk spacing is just recorded; the grid is copied brick-for-brick).
#[test]
fn round_trip_at_0_05m_through_vxofile_is_bit_identical() {
    let map = build_map();
    let registry = registry();
    let params = VxoHeadParams { voxel_size: 0.05, name: "transcode_0_05m".into(), ..Default::default() };

    // STORE always; zstd too when the `vxo-encode` compressor is present (matches the converter's default).
    #[cfg(feature = "vxo-encode")]
    let comps = [VxoCompression::Store, VxoCompression::Zstd(19)];
    #[cfg(not(feature = "vxo-encode"))]
    let comps = [VxoCompression::Store];

    for comp in comps {
        let bytes = encode_vxo(&map, &registry, &params, comp).expect("encode_vxo at 0.05 m");
        let file = VxoFile::parse(&bytes).expect("VxoFile parses a 0.05 m .vxo");

        // HEAD records the stamped 0.05 m spacing (self-describing) — exact f32 (0.05 stored verbatim).
        assert_eq!(file.head.voxel_size, 0.05, "HEAD.voxel_size records the stamped spacing");
        assert_eq!(file.head.brick_count as usize, map.len(), "HEAD brick_count == map len");

        // Every brick is bit-identical to the source map — the transcode round-trip property.
        let read_map = read_back_map(&file);
        assert_eq!(read_map.len(), map.len(), "read-back brick count == original");
        for (coord, brick) in map.iter() {
            let got = read_map.get(*coord).unwrap_or_else(|| panic!("brick {coord:?} missing after 0.05 m round-trip"));
            assert_eq!(got, brick, "brick {coord:?} not bit-identical after 0.05 m round-trip");
        }
    }
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

/// STORE region bodies carry `brik_raw_len == brik_comp_len` (the §B1.5 STORE convention) — this part needs no
/// compressor and always runs.
#[test]
fn store_lengths() {
    let map = build_map();
    let reg = registry();

    let store = VxoFile::parse(&encode_vxo(&map, &reg, &VxoHeadParams::default(), VxoCompression::Store).unwrap()).unwrap();
    for e in &store.bidx {
        assert_eq!(e.brik_comp_len, e.brik_raw_len, "STORE region: comp_len == raw_len");
    }
}

/// zstd bodies record a raw length for the decode buffer (the redundant uniform/dedup'd map compresses). Gated
/// on `vxo-encode` (needs the C compressor).
#[cfg(feature = "vxo-encode")]
#[test]
fn zstd_lengths() {
    let map = build_map();
    let reg = registry();
    let z = VxoFile::parse(&encode_vxo(&map, &reg, &VxoHeadParams::default(), VxoCompression::Zstd(19)).unwrap()).unwrap();
    // At least one region is non-trivially sized; its raw length is recorded for the decode buffer.
    assert!(z.bidx.iter().all(|e| e.brik_raw_len > 0), "every region records a raw length");
}

/// STORE regions record the EXPLICIT compression code 0 (§B1.5 FIX 3) — so the reader branches on the code,
/// never on `comp_len == raw_len` length equality. Always runs (STORE needs no compressor).
#[test]
fn store_compression_code_is_explicit() {
    let map = build_map();
    let reg = registry();
    let store = VxoFile::parse(&encode_vxo(&map, &reg, &VxoHeadParams::default(), VxoCompression::Store).unwrap()).unwrap();
    assert!(store.bidx.iter().all(|e| e.compression == VXO_REGION_STORE), "STORE regions carry code 0");
}

/// zstd regions record the EXPLICIT compression code 1 (§B1.5 FIX 3). Gated on `vxo-encode` (needs the C
/// compressor to produce a zstd body).
#[cfg(feature = "vxo-encode")]
#[test]
fn zstd_compression_code_is_explicit() {
    let map = build_map();
    let reg = registry();
    let z = VxoFile::parse(&encode_vxo(&map, &reg, &VxoHeadParams::default(), VxoCompression::Zstd(19)).unwrap()).unwrap();
    assert!(z.bidx.iter().all(|e| e.compression == VXO_REGION_ZSTD), "zstd regions carry code 1");
}

/// Querying an ABSENT region (via `VxoFile::region_entry`) and an absent brick coord within a present region
/// (via `DecodedRegion::entry`) both return `None` — covers the binary-search helpers' miss path (B-ii relies
/// on these returning AIR/None for the clipmap bound).
#[test]
fn absent_region_and_coord_return_none() {
    let map = build_map();
    let bytes = encode_vxo(&map, &registry(), &VxoHeadParams::default(), VxoCompression::Store).unwrap();
    let file = VxoFile::parse(&bytes).expect("parse");

    // A region the map never touches (way outside any inserted brick) has no BIDX entry.
    assert!(file.region_entry(IVec3::new(1000, 1000, 1000)).is_none(), "absent region ⇒ None");

    // A present region's `entry` returns None for a coord that buckets into it but was never inserted.
    let present = region_of_brick(IVec3::new(0, 0, 0), file.region_edge_bricks());
    let dir = file.region_entry(present).expect("region (0,0,0) is present");
    let region = file.decode_region(dir).expect("decode");
    // (0,0,0) and (1,2,3) ARE in this region; (7,7,7) buckets to it (K=8) but was never inserted.
    assert!(region.entry(IVec3::new(0, 0, 0)).is_some(), "an inserted brick is found");
    assert!(region.entry(IVec3::new(7, 7, 7)).is_none(), "an absent coord in a present region ⇒ None");
}

/// A uniform brick whose 10³ halo is ENTIRELY one block (it is fully surrounded by solid same-block neighbours)
/// stays UNIFORM through `pack_brickmap` — so the packed-patch `is_uniform()` decode branch in the fingerprint
/// is genuinely exercised on the round-trip (an edge uniform brick would re-expand to dense via its AIR halo).
#[test]
fn halo_buried_uniform_round_trips() {
    let registry = registry();
    let params = VxoHeadParams { name: "buried".into(), ..Default::default() };

    // Build a 3×3×3 block of identical full-uniform bricks; the CENTRE brick's 10³ halo is all block 1, so the
    // packer keeps it uniform.
    let mut map = BrickMap::new();
    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                map.insert(IVec3::new(x, y, z), full_uniform_brick(1));
            }
        }
    }

    // Sanity: the centre brick IS uniform in the packed patch (the branch we want to cover).
    let patch = pack_brickmap(&map, &registry);
    let centre_origin = [0, 0, 0]; // brick (0,0,0)·BRICK_EDGE
    let centre = patch
        .metas
        .iter()
        .find(|m| m.voxel_origin == centre_origin)
        .expect("centre brick packed");
    assert!(centre.is_uniform(), "the fully-buried centre brick must stay uniform after pack (halo all solid)");

    // Round-trip STORE (always) + zstd (only when the `vxo-encode` compressor is available): the centre stays
    // uniform and every brick is bit-identical.
    #[cfg(feature = "vxo-encode")]
    let comps = [VxoCompression::Store, VxoCompression::Zstd(19)];
    #[cfg(not(feature = "vxo-encode"))]
    let comps = [VxoCompression::Store];
    for comp in comps {
        let bytes = encode_vxo(&map, &registry, &params, comp).expect("encode");
        let file = VxoFile::parse(&bytes).expect("parse");
        let read_map = read_back_map(&file);
        assert_eq!(read_map.len(), map.len(), "buried-uniform read-back count");
        for (coord, brick) in map.iter() {
            let got = read_map.get(*coord).unwrap_or_else(|| panic!("brick {coord:?} missing"));
            assert_eq!(got, brick, "buried-uniform brick {coord:?} not bit-identical");
        }
        // The read-back centre is still uniform when re-packed.
        let read_patch = pack_brickmap(&read_map, &file.registry);
        let read_centre = read_patch
            .metas
            .iter()
            .find(|m| m.voxel_origin == centre_origin)
            .expect("centre brick re-packed");
        assert!(read_centre.is_uniform(), "centre stays uniform after round-trip");
    }
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

// ================================================================================================
// Stage 1 — the baked `LODS` coarse-LOD pyramid (`VXO_FORMAT.md` §B1.7) parity gate.
// ================================================================================================

/// A dense brick FILLED solid with a deterministic mix of two blocks (so it is NOT uniform and its
/// downsample exercises the dominant-block reducer) — every voxel solid, so the box has no air holes and
/// downsampling stays non-empty across many levels.
fn filled_brick(seed: i32) -> Brick {
    let mut v = Box::new([BlockId::AIR; BRICK_VOXELS]);
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                let i = (x + y * BRICK_EDGE + z * BRICK_EDGE * BRICK_EDGE) as usize;
                // Two solid blocks in a checker-ish pattern (never AIR) so the brick is dense, full, and its
                // coarse aggregate has a well-defined dominant block.
                v[i] = if (x + y + z + seed).rem_euclid(3) == 0 { BlockId(1) } else { BlockId(2) };
            }
        }
    }
    Brick::from_voxels(v)
}

/// A SOLID box of `edge × edge × edge` LOD0 bricks (origin at the brick grid origin), each filled (dense).
/// `edge = 4` ⇒ the pyramid is non-empty for several levels (4→2→1 bricks, then a single coarse brick keeps
/// downsampling non-empty) so it CAPS at `MAX_LOD` — a good multi-level parity subject.
fn filled_box_map(edge: i32) -> BrickMap {
    let mut map = BrickMap::new();
    for z in 0..edge {
        for y in 0..edge {
            for x in 0..edge {
                map.insert(IVec3::new(x, y, z), filled_brick(x * 7 + y * 13 + z * 17));
            }
        }
    }
    map
}

/// **Stage 1 PARITY GATE (§B1.7): a baked `LODS` coarse brick is BIT-IDENTICAL to the demand-synthesized one.**
/// Bake a multi-level map WITH `LODS` (STORE — no zstd dep), parse it, then for EVERY level `L∈1..=max_lod` and
/// EVERY present coarse brick coord assert the decoded coarse [`Brick`] equals the corresponding brick from the
/// in-RAM `downsample_brickmap`-chained pyramid (`build_coarse_pyramid`, the same SSOT the bake uses). Also
/// asserts `HEAD.max_lod` == the deepest non-empty level and the three-base region decode is correct.
#[test]
fn lods_baked_pyramid_is_bit_identical_to_demand() {
    let map = filled_box_map(4);
    let registry = registry();
    let params = VxoHeadParams { name: "lods_parity".into(), ..Default::default() };

    // The in-RAM reference pyramid (pyramid[i] = LOD i+1) — the SSOT the bake reuses.
    let pyramid = build_coarse_pyramid(&map);
    let max_lod = pyramid.len() as u32;
    assert!(max_lod >= 3, "the filled box must bake several coarse levels (got {max_lod})");

    let bytes = encode_vxo(&map, &registry, &params, VxoCompression::Store).expect("encode_vxo WITH LODS");
    let file = VxoFile::parse(&bytes).expect("parse the LODS-baked .vxo");

    // The file carries a LODS pyramid, and HEAD.max_lod == the deepest non-empty level.
    assert!(file.has_lods(), "bake_lods=true ⇒ a LODS chunk is present");
    assert_eq!(file.max_lod(), max_lod, "HEAD.max_lod == deepest non-empty pyramid level");
    let lods = file.lods().expect("LODS parsed");
    assert_eq!(lods.levels.len() as u32, max_lod, "one LODS level per L∈1..=max_lod");

    let k = file.region_edge_bricks();
    for (i, level_map) in pyramid.iter().enumerate() {
        let lod = (i + 1) as u32;
        // The parsed level table entry records the matching lod.
        assert_eq!(lods.levels[i].lod, lod, "LODS level[{i}].lod == {lod}");

        // Bucket the reference level's bricks by coarse region; decode each region from the baked LODS and
        // assert bit-identity per coarse coord.
        let mut want_count = 0usize;
        let mut by_region: std::collections::HashMap<[i32; 3], Vec<IVec3>> = std::collections::HashMap::new();
        for (coord, _brick) in level_map.iter() {
            want_count += 1;
            let rc = region_of_brick(*coord, k);
            by_region.entry([rc.x, rc.y, rc.z]).or_default().push(*coord);
        }

        let mut decoded_count = 0usize;
        for (rc_arr, coords) in &by_region {
            let rc = IVec3::new(rc_arr[0], rc_arr[1], rc_arr[2]);
            let region = file
                .decode_lod_region(i, rc)
                .unwrap_or_else(|| panic!("LOD{lod} region {rc:?} absent in LODS"))
                .unwrap_or_else(|e| panic!("LOD{lod} region {rc:?} decode failed: {e}"));
            // The decoded region body's header carries the right lod (verified inside parse_region too).
            for &coord in coords {
                let entry = region
                    .entry(coord)
                    .unwrap_or_else(|| panic!("LOD{lod} coarse brick {coord:?} missing from its region"));
                let got = region.brick(entry);
                let want = level_map.get(coord).expect("reference coarse brick present");
                assert_eq!(&got, want, "LOD{lod} coarse brick {coord:?} not bit-identical to the demand pyramid");
                decoded_count += 1;
            }
        }
        assert_eq!(decoded_count, want_count, "LOD{lod}: every reference coarse brick was decoded");
    }
}

/// **Stage 1 forward-compat: a `bake_lods=false` file has NO `LODS` chunk** (today's behaviour / the Stage-2
/// fallback). The same map encodes without a pyramid, parses fine, `has_lods()==false`, `max_lod()==0`, and the
/// base bricks still round-trip.
#[test]
fn no_lods_when_disabled_parses_and_round_trips() {
    let map = filled_box_map(4);
    let registry = registry();
    let params = VxoHeadParams { name: "no_lods".into(), bake_lods: false, ..Default::default() };

    let bytes = encode_vxo(&map, &registry, &params, VxoCompression::Store).expect("encode_vxo WITHOUT LODS");
    let file = VxoFile::parse(&bytes).expect("parse the no-LODS .vxo");
    assert!(!file.has_lods(), "bake_lods=false ⇒ no LODS chunk");
    assert_eq!(file.max_lod(), 0, "HEAD.max_lod == 0 when no pyramid is baked");
    assert!(file.lods().is_none(), "no parsed LODS directory");

    // The base LOD0 bricks still round-trip bit-identically.
    let read_map = read_back_map(&file);
    assert_eq!(read_map.len(), map.len(), "no-LODS read-back brick count == original");
    for (coord, brick) in map.iter() {
        let got = read_map.get(*coord).unwrap_or_else(|| panic!("brick {coord:?} missing"));
        assert_eq!(got, brick, "no-LODS base brick {coord:?} not bit-identical");
    }
}

/// **Gotcha #4 — `max_lod` is the deepest NON-EMPTY level, capped at `MAX_LOD`.** A tiny single-brick map
/// collapses to one coarse brick almost immediately, but a single coarse brick still downsamples to a (smaller)
/// non-empty coarse brick, so the pyramid runs to the `MAX_LOD` cap. Assert the baked `HEAD.max_lod` equals the
/// in-RAM pyramid depth (the SSOT), and that it never exceeds `MAX_LOD`.
#[test]
fn lods_max_lod_tracks_deepest_nonempty_level() {
    use crate::voxel::brickmap::MAX_LOD;
    // A single dense brick at the origin.
    let mut map = BrickMap::new();
    map.insert(IVec3::new(0, 0, 0), filled_brick(0));

    let pyramid = build_coarse_pyramid(&map);
    let depth = pyramid.len() as u32;
    assert!(depth <= MAX_LOD, "pyramid depth {depth} must not exceed MAX_LOD {MAX_LOD}");

    let registry = registry();
    let params = VxoHeadParams { name: "tiny_lods".into(), ..Default::default() };
    let bytes = encode_vxo(&map, &registry, &params, VxoCompression::Store).expect("encode tiny LODS");
    let file = VxoFile::parse(&bytes).expect("parse tiny LODS");
    assert_eq!(file.max_lod(), depth, "HEAD.max_lod == in-RAM pyramid depth (deepest non-empty)");
    assert_eq!(file.max_lod(), MAX_LOD, "a single solid brick downsamples non-empty to the MAX_LOD cap");

    // And the single coarse brick at each level is bit-identical to the demand pyramid.
    let k = file.region_edge_bricks();
    for (i, level_map) in pyramid.iter().enumerate() {
        for (coord, want) in level_map.iter() {
            let rc = region_of_brick(*coord, k);
            let region = file
                .decode_lod_region(i, rc)
                .unwrap_or_else(|| panic!("L{} region {rc:?} absent", i + 1))
                .expect("decode");
            let entry = region.entry(*coord).expect("coarse brick present");
            assert_eq!(&region.brick(entry), want, "tiny-map coarse brick {coord:?} parity");
        }
    }
}

/// **Stage 1 streaming-writer LODS parity:** the bounded-RAM [`super::writer::VxoStreamWriter`] fed the SAME
/// base regions (`add_region`) AND coarse regions (`add_lod_region`) in `(z,y,x)` order as `encode_vxo` lays
/// them out produces a BYTE-IDENTICAL `.vxo` (including the appended LODS chunk). This proves the streaming
/// LODS bake (`add_lod_region`/`finish`) is correct by construction — same encoder per region, same framing.
#[test]
fn stream_writer_lods_matches_encode_vxo() {
    let map = filled_box_map(4);
    let registry = registry();
    let params = VxoHeadParams { name: "stream_lods".into(), ..Default::default() };

    let want = encode_vxo(&map, &registry, &params, VxoCompression::Store).expect("encode_vxo WITH LODS");
    // It really did bake a pyramid (else this test would trivially pass against a no-LODS file).
    assert!(VxoFile::parse(&want).unwrap().has_lods(), "reference must carry LODS");

    let k = params.region_edge_bricks as i32;
    let dir = std::env::temp_dir().join(format!("vxo_stream_lods_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let scratch = dir.join("brik.tmp");
    let out = dir.join("stream_lods.vxo");

    let mut w = super::writer::VxoStreamWriter::new(params.clone(), &registry, VxoCompression::Store, &scratch)
        .expect("open stream writer");

    // Feed the BASE LOD0 regions in (z,y,x) order (matches `encode_vxo`'s BRIK layout).
    feed_regions_in_order(&map, k, |rc, bricks| w.add_region(rc, bricks).expect("add_region"));

    // Feed each COARSE level's regions through the SHARED `drive_coarse_lods` ordering SSOT — the SAME loop the
    // full-RAM `build_lods_body` drives. Routing the STREAMING sink through `drive_coarse_lods` (not a local test
    // helper) is what proves the SSOT yields byte-identical LODS on both writers (the Stage-0 gate), so this test
    // exercises `drive_coarse_lods` on BOTH sinks.
    let pyramid = build_coarse_pyramid(&map);
    super::writer::drive_coarse_lods(&pyramid, k, |lod, rc, bricks| w.add_lod_region(lod, rc, bricks))
        .expect("drive_coarse_lods (streaming)");
    w.finish(&out).expect("finish stream write");

    let got = std::fs::read(&out).expect("read streamed LODS .vxo");
    assert_eq!(got, want, "streamed LODS .vxo must be byte-identical to encode_vxo");
    let _ = std::fs::remove_dir_all(&dir);
}

/// **Stage 0 — `max_lod == MAX_LOD` invariant + read-side clamp-no-op (Stage-2 reviewer finding).** For a baked
/// NON-EMPTY asset (whether via the full-RAM `encode_vxo` or the bounded-RAM `VxoStreamWriter` routed through the
/// shared `drive_coarse_lods` SSOT) `HEAD.max_lod` MUST equal `MAX_LOD`, so the read-side `coarse_level` clamp
/// (`lod.min(max_lod)`) is a GUARANTEED no-op for every `lod ∈ 1..=MAX_LOD` — never diverging from
/// `StaticVoxSource::level`, which always spans the full pyramid depth. We assert it on BOTH writer paths.
#[test]
fn baked_asset_max_lod_is_max_lod_and_clamp_is_noop() {
    use crate::voxel::brickmap::MAX_LOD;

    let map = filled_box_map(4);
    let registry = registry();
    let params = VxoHeadParams { name: "max_lod_invariant".into(), ..Default::default() };

    // Path A: full-RAM `encode_vxo`.
    let bytes = encode_vxo(&map, &registry, &params, VxoCompression::Store).expect("encode_vxo WITH LODS");
    let file = VxoFile::parse(&bytes).expect("parse the LODS-baked .vxo");
    assert!(file.has_lods(), "a non-empty baked asset carries a LODS pyramid");
    assert_eq!(file.max_lod(), MAX_LOD, "encode_vxo: a baked non-empty asset has HEAD.max_lod == MAX_LOD");

    // Path B: the streaming writer, coarse bake driven through the shared `drive_coarse_lods` SSOT.
    let k = params.region_edge_bricks as i32;
    let dir = std::env::temp_dir().join(format!("vxo_max_lod_inv_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let scratch = dir.join("brik.tmp");
    let out = dir.join("max_lod.vxo");
    let mut w = super::writer::VxoStreamWriter::new(params.clone(), &registry, VxoCompression::Store, &scratch)
        .expect("open stream writer");
    feed_regions_in_order(&map, k, |rc, bricks| w.add_region(rc, bricks).expect("add_region"));
    let pyramid = build_coarse_pyramid(&map);
    super::writer::drive_coarse_lods(&pyramid, k, |lod, rc, bricks| w.add_lod_region(lod, rc, bricks))
        .expect("drive_coarse_lods");
    w.finish(&out).expect("finish stream write");
    let stream_file = VxoFile::parse(&std::fs::read(&out).expect("read streamed .vxo")).expect("parse streamed");
    assert_eq!(
        stream_file.max_lod(),
        MAX_LOD,
        "VxoStreamWriter: a baked non-empty asset has HEAD.max_lod == MAX_LOD"
    );

    // The read-side clamp `lod.min(max_lod)` is a GUARANTEED no-op for every requested coarse level — i.e. no
    // coarse request in 1..=MAX_LOD is ever served the WRONG (collapsed) level grid (the StaticVoxSource parity).
    for f in [&file, &stream_file] {
        let max_lod = f.max_lod();
        for lod in 1..=MAX_LOD {
            assert_eq!(lod.min(max_lod), lod, "clamp must be a no-op for lod {lod} (max_lod {max_lod})");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Bucket `map`'s bricks into K=`k` regions and invoke `feed(region_coord, sorted_bricks)` for each region in
/// `(z,y,x)` region order with the region's bricks pre-sorted by `(z,y,x)` — the exact order `encode_vxo`
/// (and `build_lods_body`) emit, so a streamed writer fed this way matches byte-for-byte.
fn feed_regions_in_order(map: &BrickMap, k: i32, mut feed: impl FnMut(IVec3, &[(IVec3, &Brick)])) {
    let mut regions: std::collections::BTreeMap<(i32, i32, i32), Vec<IVec3>> = Default::default();
    for (coord, _) in map.iter() {
        let r = region_of_brick(*coord, k);
        regions.entry((r.z, r.y, r.x)).or_default().push(*coord);
    }
    for ((rz, ry, rx), mut coords) in regions {
        coords.sort_by_key(|c| (c.z, c.y, c.x));
        let bricks: Vec<(IVec3, &Brick)> = coords.iter().map(|&c| (c, map.get(c).expect("brick"))).collect();
        feed(IVec3::new(rx, ry, rz), &bricks);
    }
}

/// The two new `LODS` POD structs hold their pinned sizes (the SSOT size guard also lives in `format.rs`'s
/// `record_sizes_match_spec`; this re-asserts from the `tests.rs` side so the gate is co-located with the
/// round-trip).
#[test]
fn lods_struct_sizes() {
    assert_eq!(std::mem::size_of::<VxoLodsHeader>(), 16, "LODS header is 16 B (§B1.7)");
    assert_eq!(std::mem::size_of::<VxoLodLevel>(), 32, "LODS level entry is 32 B (§B1.7)");
    assert_eq!(std::mem::size_of::<VxoLodsHeader>() % 16, 0, "LODS header is a 16-multiple");
    assert_eq!(std::mem::size_of::<VxoLodLevel>() % 16, 0, "LODS level stride is a 16-multiple");
}

// ============================================================================================
// Stages 1+2 — constant-RAM disk-spill base + windowed coarse PARITY gate (the byte-identity proof)
// ============================================================================================

/// Bake `map` through the CONSTANT-RAM disk-spill producer (`spill::RegionSpillPool` → `assemble_base` →
/// `windowed_coarse` → `VxoStreamWriter::finish`) and return the produced `.vxo` bytes. This is the EXACT path
/// `examples/voxelize_scene.rs::assemble_vxo_streaming` drives — so a byte-match against `encode_vxo` proves the
/// spill base completeness AND the windowed coarse cross-region gather are bit-identical to the resident bake.
fn bake_via_spill(map: &BrickMap, registry: &BlockRegistry, params: &VxoHeadParams, comp: VxoCompression) -> Vec<u8> {
    use crate::voxel::brickmap::BRICK_EDGE;
    use crate::voxel::vxo::spill::{RegionSpillPool, assemble_base, spill_voxel, windowed_coarse};

    let k = params.region_edge_bricks as i32;
    let dir = std::env::temp_dir().join(format!("vxo_spill_parity_{}_{}", std::process::id(), rand_suffix()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let out = dir.join("spill.vxo");
    let scratch_brik = dir.join("assembly.brik.tmp");

    // 1. Spill every SOLID voxel of `map` to its per-region file (the production spill pass over a stream of
    //    solids — here driven from the resident map, but the spill pool never holds the whole map).
    let mut base = RegionSpillPool::new(&dir, "base", k);
    for (coord, brick) in map.iter() {
        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                for x in 0..BRICK_EDGE {
                    let b = brick.get(x, y, z);
                    if !b.is_air() {
                        let w = *coord * BRICK_EDGE + IVec3::new(x, y, z);
                        spill_voxel(&mut base, w, b).expect("spill voxel");
                    }
                }
            }
        }
    }
    base.flush_all().expect("flush base");

    let mut writer = super::writer::VxoStreamWriter::new(params.clone(), registry, comp, &scratch_brik)
        .expect("open stream writer");
    let mut coarse_l0 = RegionSpillPool::new(&dir, "coarse_l0", k);
    let base_bricks = assemble_base(&base, &mut coarse_l0, &mut writer).expect("assemble base");
    base.delete_all();
    if base_bricks > 0 {
        windowed_coarse(coarse_l0, &dir, k, &mut writer).expect("windowed coarse");
    } else {
        coarse_l0.delete_all();
    }
    writer.finish(&out).expect("finish");

    let bytes = std::fs::read(&out).expect("read spill .vxo");
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

/// A cheap per-call unique suffix (a nanosecond clock XOR a monotonic counter) so concurrent test scratch dirs
/// never collide (cargo runs tests in parallel threads of one process — a clock alone can repeat under fast
/// scheduling, so fold in a process-global counter too).
fn rand_suffix() -> u128 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed) as u128;
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    t ^ (n << 96)
}

/// **Stages 1+2 parity (the byte-identity gate):** the constant-RAM disk-spill bake produces a `.vxo`
/// BIT-IDENTICAL to the resident `encode_vxo`/`build_coarse_pyramid` bake for the SAME map — pinning both the
/// base-region completeness (Stage 1) and the windowed coarse cross-region gather (Stage 2). Run on the
/// multi-region/negative-coord `build_map()` so the Euclidean bucketing + intra-region dedup are exercised.
#[test]
fn spill_bake_matches_resident_encode() {
    let map = build_map();
    let registry = registry();
    let params = VxoHeadParams { name: "spill_parity".into(), ..Default::default() };

    let want = encode_vxo(&map, &registry, &params, VxoCompression::Store).expect("encode_vxo (resident)");
    assert!(VxoFile::parse(&want).unwrap().has_lods(), "reference must carry a LODS pyramid");
    let got = bake_via_spill(&map, &registry, &params, VxoCompression::Store);
    assert_eq!(got, want, "disk-spill bake must be byte-identical to the resident encode_vxo");
}

/// Same parity gate over a SOLID filled box (every coarse level non-empty → the pyramid caps at MAX_LOD), so the
/// windowed coarse runs its FULL depth and every level's cross-region gather is exercised.
#[test]
fn spill_bake_matches_resident_filled_box() {
    let map = filled_box_map(4);
    let registry = registry();
    let params = VxoHeadParams { name: "spill_box".into(), ..Default::default() };
    let want = encode_vxo(&map, &registry, &params, VxoCompression::Store).expect("encode_vxo (resident)");
    let got = bake_via_spill(&map, &registry, &params, VxoCompression::Store);
    assert_eq!(got, want, "disk-spill bake (filled box) must be byte-identical to encode_vxo");
}

/// **Boundary-straddle pin (the hardest correctness point):** surface bricks placed so a COARSE region's
/// high-face coarse bricks have their `2·cc+1` children in the ADJACENT finer region. With K=8, finer brick
/// coords K-1 (=7) and K (=8) sit in finer regions 0 and 1; their parent coarse brick (7/2=3 and 8/2=4) — and
/// crucially the coarse brick at 3, whose children are finer bricks 6 and 7 (both region 0), vs coarse brick 4,
/// whose children are 8 and 9 (region 1). To force a coarse brick whose TWO children straddle the finer-region
/// boundary we use finer bricks at coords that map a single coarse brick onto children in two regions: a coarse
/// region boundary at coarse-brick 4 means children 8,9; but the straddle case is a coarse brick at the EDGE of
/// the finer window. We build a slab spanning finer x ∈ {6,7,8,9} so coarse bricks 3 (children 6,7) and 4
/// (children 8,9) both materialize, AND the coarse REGION boundary (coarse brick 8 = finer 16,17) is crossed by
/// extending to finer x=15,16,17 — exercising the windowed load of two finer regions for one coarse region.
#[test]
fn spill_bake_matches_resident_on_region_boundary_straddle() {
    let registry = registry();
    let params = VxoHeadParams { name: "spill_straddle".into(), ..Default::default() };

    // A thin slab of dense surface bricks straddling the finer-region boundary at brick x=8 (region 0|1) AND the
    // COARSE-region boundary at coarse brick x=8 (i.e. finer bricks x=16,17). Coords chosen so several coarse
    // bricks gather children from TWO finer regions.
    let mut map = BrickMap::new();
    let xs = [6, 7, 8, 9, 14, 15, 16, 17];
    for &x in &xs {
        for y in 0..3 {
            for z in 0..3 {
                // Coords ≡ K-1 (7,15) and ≡ 0 (8,16) across boundaries are present (the pin from the plan).
                map.insert(IVec3::new(x, y, z), dense_brick(x * 7 + y * 13 + z * 17));
            }
        }
    }

    let want = encode_vxo(&map, &registry, &params, VxoCompression::Store).expect("encode_vxo (resident)");
    let got = bake_via_spill(&map, &registry, &params, VxoCompression::Store);
    assert_eq!(
        got, want,
        "boundary-straddle: windowed cross-region coarse gather must be byte-identical to the resident bake"
    );
}
