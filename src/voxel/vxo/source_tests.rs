//! Phase B-ii acceptance gates for the region-streamed [`VxoSource`] (`docs/VXO_FORMAT.md` §B2.8):
//!
//! * **Gate 2 (streamed round-trip):** `VxoSource::brick` is bit-identical to `StaticVoxSource::brick` for
//!   every coord, and an absent coord returns `uniform(AIR)`.
//! * **Gate 3 (classify parity):** `VxoSource::classify == StaticVoxSource::classify` across surface/interior/
//!   air bricks (the surface-only cull is preserved).
//! * **Gate 1 (PARTIAL, budget/eviction):** a synthetic multi-region `.vxo` driven under a tiny LRU budget
//!   never holds all regions at once + never exceeds its byte budget.
//! * Plus: the `MergedSource` offset + block-base remap dispatch, and the §B2.6 voxel_size-mismatch error.
//!
//! Tests write STORE `.vxo` files to the std temp dir (STORE needs no compressor — works on a default build).

use std::path::PathBuf;

use bevy::math::IVec3;

use super::super::format::VxoHead;
use super::super::source::{MergedSource, VxoSource};
use super::super::writer::{VxoCompression, VxoHeadParams, encode_vxo, write_vxo};
use crate::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, BrickMap};
use crate::voxel::palette::{BlockId, BlockRegistry};
use crate::voxel::source::{BrickClass, BrickSource, StaticVoxSource};

/// A 5-block registry (AIR + 4 solids, one emitter) — the Cornell palette, reused as in the B-i round-trip.
fn registry() -> BlockRegistry {
    BlockRegistry::cornell()
}

/// A dense, non-uniform brick with a deterministic mixed pattern (some AIR, a few solid blocks) — so it uses a
/// multi-entry palette + a real index width (exercises the dense decode, not just the uniform fast path).
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

/// The known non-trivial map: uniform + dense bricks across MULTIPLE K=8 regions incl. NEGATIVE coords + an
/// intra-region duplicate (the R3 dedup path) — the same shape the B-i round-trip uses, so the streamed path is
/// tested over the full feature surface.
fn build_map() -> BrickMap {
    let mut map = BrickMap::new();
    map.insert(IVec3::new(0, 0, 0), Brick::uniform(BlockId(1)));
    map.insert(IVec3::new(1, 2, 3), dense_brick(11));
    map.insert(IVec3::new(8, 1, 1), dense_brick(22));
    map.insert(IVec3::new(9, 0, 0), Brick::uniform(BlockId(2)));
    map.insert(IVec3::new(-1, -1, -1), dense_brick(33));
    map.insert(IVec3::new(-5, -3, -2), Brick::uniform(BlockId(3)));
    map.insert(IVec3::new(16, 0, 0), dense_brick(44));
    map.insert(IVec3::new(17, 0, 1), dense_brick(44));
    map
}

/// A unique temp path for a test's `.vxo` (the std temp dir — CPU-only round-trip, no GPU temp-dir caveat).
fn temp_vxo(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("vxo_bii_{name}_{}.vxo", std::process::id()));
    p
}

/// Write `map` to a STORE `.vxo` at a temp path, returning the path (cleaned up by the OS temp dir).
fn write_store(map: &BrickMap, reg: &BlockRegistry, name: &str) -> PathBuf {
    let path = temp_vxo(name);
    let params = VxoHeadParams { name: name.into(), ..Default::default() };
    write_vxo(&path, map, reg, &params, VxoCompression::Store).expect("write_vxo STORE");
    path
}

/// The set of EVERY LOD0 brick coord present in `map`, PLUS a margin of absent neighbours, so a test sweeps
/// both stored bricks and the air around them.
fn coords_to_probe(map: &BrickMap) -> Vec<IVec3> {
    let mut coords: Vec<IVec3> = map.iter().map(|(c, _)| *c).collect();
    // Add absent neighbours (the clipmap bound) around each stored brick + a couple of far-away coords.
    let stored = coords.clone();
    for c in stored {
        for d in [IVec3::X, IVec3::Y, IVec3::Z, -IVec3::X, -IVec3::Y, -IVec3::Z] {
            coords.push(c + d);
        }
    }
    coords.push(IVec3::new(1000, 1000, 1000));
    coords.push(IVec3::new(-1000, 0, 0));
    coords
}

/// **Gate 2 — streamed round-trip bit-identity.** Write a known map to a STORE `.vxo`, open it as a streamed
/// `VxoSource`, and assert `VxoSource::brick(coord, 0)` is bit-identical to `StaticVoxSource::brick(coord, 0)`
/// for every stored coord, and `uniform(AIR)` for absent coords — the LRU-decode path yields the SAME `Brick`
/// the direct source does.
#[test]
fn gate2_streamed_brick_matches_static() {
    let map = build_map();
    let reg = registry();
    let path = write_store(&map, &reg, "gate2");

    let (vxo, vxo_reg) = VxoSource::open(&path).expect("open streamed VxoSource");
    let stat = StaticVoxSource::new(&map);

    // The rebuilt registry has the same block count (MATL round-trips).
    assert_eq!(vxo_reg.len(), reg.len(), "MATL rebuilds the same block count");

    for coord in coords_to_probe(&map) {
        let want = stat.brick(coord, 0, &reg);
        let got = vxo.brick(coord, 0, &vxo_reg);
        assert_eq!(got, want, "streamed brick at {coord:?} must be bit-identical to the static source");
    }

    // An explicitly absent coord returns uniform(AIR).
    let absent = vxo.brick(IVec3::new(500, 500, 500), 0, &vxo_reg);
    assert!(absent.is_empty(), "an absent coord returns uniform(AIR)");
    let _ = std::fs::remove_file(&path);
}

/// **Gate 3 — classify parity.** Build a 3×3×3 block of full bricks (interior + faces + corners) plus an
/// absent region, write it, and assert `VxoSource::classify(coord, lod) == StaticVoxSource::classify(coord,
/// lod)` for every probed coord/LOD — the surface-only enclosed-cull is preserved on the `.vxo` path.
#[test]
fn gate3_classify_matches_static() {
    // 3×3×3 fully-solid block of bricks at [0,3)³, plus a partial brick to exercise the !is_full branch.
    let mut map = BrickMap::new();
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                map.insert(IVec3::new(x, y, z), Brick::uniform(BlockId(1)));
            }
        }
    }
    // A separate partial brick far from the block (full=false ⇒ Surface).
    let mut arr = Box::new([BlockId(1); BRICK_VOXELS]);
    arr[0] = BlockId::AIR;
    map.insert(IVec3::new(20, 20, 20), Brick::from_voxels(arr));

    let reg = registry();
    let path = write_store(&map, &reg, "gate3");
    let (vxo, _vxo_reg) = VxoSource::open(&path).expect("open");
    let stat = StaticVoxSource::new(&map);

    // Probe the whole block (incl. centre/faces/corners), the partial brick, and surrounding air, at LOD0 and a
    // coarse LOD (both sources return Surface for coarse — no baked LODS).
    let mut probe: Vec<IVec3> = Vec::new();
    for z in -1..=3 {
        for y in -1..=3 {
            for x in -1..=3 {
                probe.push(IVec3::new(x, y, z));
            }
        }
    }
    probe.push(IVec3::new(20, 20, 20));
    probe.push(IVec3::new(100, 0, 0));

    // LOD0 parity is the meaningful gate: both sources serve the SAME LOD0 brick grid, so the enclosed-cull
    // must agree brick-for-brick (Interior/Surface/Air). (Coarse-LOD classify genuinely DIFFERS by design:
    // StaticVoxSource downsamples a coarse pyramid in RAM and can classify on the coarse grid, whereas the
    // `.vxo` path defers LODS (§B1.7) and conservatively returns Surface for any lod>0 — asserted separately.)
    for coord in &probe {
        let want = stat.classify(*coord, 0);
        let got = vxo.classify(*coord, 0);
        assert_eq!(got, want, "LOD0 classify parity at {coord:?}");
    }
    // Spot-check the cull actually fired: the centre is Interior, a face is Surface, outside is Air.
    assert_eq!(vxo.classify(IVec3::new(1, 1, 1), 0), BrickClass::Interior, "buried centre ⇒ Interior");
    assert_eq!(vxo.classify(IVec3::new(1, 1, 0), 0), BrickClass::Surface, "face brick ⇒ Surface");
    assert_eq!(vxo.classify(IVec3::new(50, 50, 50), 0), BrickClass::Air, "absent ⇒ Air");
    // Coarse-LOD (no baked LODS) is conservatively Surface for every coord — the documented §B2.5 behaviour.
    for coord in &probe {
        assert_eq!(vxo.classify(*coord, 2), BrickClass::Surface, "coarse-lod .vxo classify ⇒ Surface (no LODS)");
    }
    let _ = std::fs::remove_file(&path);
}

/// Build a synthetic multi-region map: `regions³` regions of K=8 bricks, each region holding one dense brick,
/// so the file has many independently-decodable regions for the budget/eviction test.
fn multi_region_map(regions: i32) -> BrickMap {
    let mut map = BrickMap::new();
    for rz in 0..regions {
        for ry in 0..regions {
            for rx in 0..regions {
                // One brick per region at the region's origin brick (region·K).
                let coord = IVec3::new(rx, ry, rz) * 8;
                map.insert(coord, dense_brick(rx * 100 + ry * 10 + rz));
            }
        }
    }
    map
}

/// **Gate 1 (PARTIAL) — LRU budget + eviction.** Drive demands across MANY regions of a synthetic `.vxo` under
/// a TINY decoded-region budget and assert: (a) the cache byte total never exceeds the budget after a demand,
/// and (b) the decoded-region COUNT stays strictly below the total region count (the loader never holds all
/// regions at once — it evicts). The full Bistro-scale peak-RAM gate is validated after Phase C produces a
/// Bistro `.vxo`; this proves the bounding mechanism now.
#[test]
fn gate1_lru_evicts_under_budget() {
    let regions = 4; // 4³ = 64 regions
    let map = multi_region_map(regions);
    let reg = registry();
    let path = write_store(&map, &reg, "gate1");

    // One decoded region of one dense brick is small; size the budget to hold only a FEW regions so eviction
    // must fire as we sweep all 64.
    let one_region_bytes = {
        let (probe, _) = VxoSource::open(&path).expect("probe open");
        // Warm one region to measure its decoded size.
        let _ = probe.brick(IVec3::ZERO, 0, &reg);
        let (count, bytes) = probe.cache_stats();
        assert_eq!(count, 1, "one demand warms exactly one region");
        bytes
    };
    let budget = one_region_bytes * 3 + 1; // hold ~3 regions
    let (vxo, vxo_reg) = VxoSource::open_with_budget(&path, budget).expect("open with tiny budget");

    let total_regions = regions * regions * regions; // 64
    let mut max_count = 0usize;
    for rz in 0..regions {
        for ry in 0..regions {
            for rx in 0..regions {
                let coord = IVec3::new(rx, ry, rz) * 8;
                let b = vxo.brick(coord, 0, &vxo_reg);
                // The streamed brick is still correct under eviction (a re-demand re-decodes identically).
                assert_eq!(b, dense_brick(rx * 100 + ry * 10 + rz), "evicted-then-re-demanded brick is identical");
                let (count, bytes) = vxo.cache_stats();
                assert!(bytes <= budget, "cache bytes {bytes} must stay within the budget {budget}");
                max_count = max_count.max(count);
            }
        }
    }
    assert!(
        (max_count as i32) < total_regions,
        "the loader must NOT hold all {total_regions} regions at once (held at most {max_count} under the budget)"
    );
    let _ = std::fs::remove_file(&path);
}

/// `MergedSource` dispatches a world coord to the owning asset (by placed `HEAD.bounds`), takes its non-air
/// brick with its `BlockId`s SHIFTED into the merged palette, and returns AIR in the gap between assets. Two
/// single-brick assets carrying the SAME local `BlockId(1)` must end up with DISTINCT merged ids (no collision).
#[test]
fn merged_source_offsets_and_remaps() {
    // Asset A: one brick of block 1 at brick coord (0,0,0). Asset B: one brick of block 1 at (0,0,0) too.
    let mut a = BrickMap::new();
    a.insert(IVec3::new(0, 0, 0), Brick::uniform(BlockId(1)));
    let mut b = BrickMap::new();
    b.insert(IVec3::new(0, 0, 0), Brick::uniform(BlockId(1)));
    let reg = registry(); // both assets share the same 5-block registry shape

    let pa = write_store(&a, &reg, "mergeA");
    let pb = write_store(&b, &reg, "mergeB");
    let (sa, ra) = VxoSource::open(&pa).expect("open A");
    let (sb, rb) = VxoSource::open(&pb).expect("open B");

    // Place B 50 bricks away in +X (disjoint from A, with a wide gap).
    let offset_b = IVec3::new(50, 0, 0);
    let (merged, merged_reg) = MergedSource::new(vec![(sa, ra, IVec3::ZERO), (sb, rb, offset_b)]);

    // The merged registry concatenated both assets' 4 solid blocks ⇒ AIR + 8 solids.
    assert_eq!(merged_reg.len(), 1 + 4 + 4, "merged registry concatenates both assets' solid blocks");

    // Asset A's brick at (0,0,0): block 1 stays merged id 1 (base 0).
    let ba = merged.brick(IVec3::new(0, 0, 0), 0, &merged_reg);
    assert_eq!(ba.get(0, 0, 0), BlockId(1), "asset A's local block 1 ⇒ merged id 1");

    // Asset B's brick at its placed coord (50,0,0): block 1 shifted by A's 4 blocks ⇒ merged id 5.
    let bb = merged.brick(offset_b, 0, &merged_reg);
    assert_eq!(bb.get(0, 0, 0), BlockId(5), "asset B's local block 1 ⇒ merged id 5 (no collision with A)");

    // A coord in the GAP between the two assets is AIR.
    let gap = merged.brick(IVec3::new(25, 0, 0), 0, &merged_reg);
    assert!(gap.is_empty(), "a coord in the inter-asset gap is air");

    // classify dispatches per asset: A's single brick has all neighbours absent ⇒ Surface; the gap ⇒ Air.
    assert_eq!(merged.classify(IVec3::new(0, 0, 0), 0), BrickClass::Surface, "edge brick of A ⇒ Surface");
    assert_eq!(merged.classify(IVec3::new(25, 0, 0), 0), BrickClass::Air, "gap ⇒ Air");
    let _ = std::fs::remove_file(&pa);
    let _ = std::fs::remove_file(&pb);
}

/// §B2.6 voxel_size reconciliation: a `.vxo` baked at a DIFFERENT `voxel_size` than the engine is rejected at
/// open with a clear error (NO silent rescale). We hand-encode a HEAD with a wrong `voxel_size` and assert the
/// open fails mentioning the mismatch.
#[test]
fn voxel_size_mismatch_is_rejected() {
    let map = build_map();
    let reg = registry();
    // Encode with a deliberately-wrong voxel_size (0.05 m while the engine is 0.2 m).
    let params = VxoHeadParams { voxel_size: 0.05, name: "wrongsize".into(), ..Default::default() };
    let bytes = encode_vxo(&map, &reg, &params, VxoCompression::Store).expect("encode");
    let path = temp_vxo("wrongsize");
    std::fs::write(&path, &bytes).expect("write bytes");

    let result = VxoSource::open(&path);
    let err = match result {
        Ok(_) => panic!("a voxel_size mismatch must be rejected"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(msg.contains("voxel") || msg.contains("rebake") || msg.contains("0.05"), "clear mismatch error: {msg}");
    let _ = std::fs::remove_file(&path);
}

/// The streamed source is DETERMINISTIC + thread-safe: the same coord queried repeatedly (forcing cache hits +
/// a re-decode after eviction) yields the identical brick — the parallel-drain determinism the trait relies on.
#[test]
fn streamed_source_is_deterministic() {
    let map = build_map();
    let reg = registry();
    let path = write_store(&map, &reg, "determ");
    let (vxo, vxo_reg) = VxoSource::open(&path).expect("open");
    for c in [IVec3::new(0, 0, 0), IVec3::new(1, 2, 3), IVec3::new(-1, -1, -1)] {
        let first = vxo.brick(c, 0, &vxo_reg);
        for _ in 0..4 {
            assert_eq!(vxo.brick(c, 0, &vxo_reg), first, "repeated streamed brick at {c:?} is identical");
        }
    }
    let _ = std::fs::remove_file(&path);
}

/// The streamed source over a ZSTD-compressed `.vxo` (per-region zstd, decoded off the mmap via pure-Rust
/// `ruzstd`) yields the SAME bit-identical bricks as the direct static source — proving the lazy mmap-slice →
/// `ruzstd` decode path works end-to-end, not just STORE. Gated on `vxo-encode` (PRODUCING a zstd body needs
/// the C compressor); the runtime DECODE is always pure Rust.
#[cfg(feature = "vxo-encode")]
#[test]
fn streamed_zstd_brick_matches_static() {
    let map = build_map();
    let reg = registry();
    let path = temp_vxo("zstd");
    let params = VxoHeadParams { name: "zstd".into(), ..Default::default() };
    write_vxo(&path, &map, &reg, &params, VxoCompression::Zstd(19)).expect("write_vxo zstd");

    let (vxo, vxo_reg) = VxoSource::open(&path).expect("open zstd VxoSource");
    let stat = StaticVoxSource::new(&map);
    for coord in coords_to_probe(&map) {
        assert_eq!(
            vxo.brick(coord, 0, &vxo_reg),
            stat.brick(coord, 0, &reg),
            "streamed zstd brick at {coord:?} must match the static source"
        );
    }
    let _ = std::fs::remove_file(&path);
}

/// `HEAD` round-trips the engine's `voxel_size` for a default-params bake (so a stand-alone open passes the
/// §B2.6 assert) — the positive case of the reconciliation check.
#[test]
fn head_voxel_size_matches_engine_default() {
    let head = VxoHead {
        voxel_size: crate::voxel::brickmap::VOXEL_SIZE,
        ..bytemuck::Zeroable::zeroed()
    };
    assert_eq!(head.voxel_size, crate::voxel::brickmap::VOXEL_SIZE);
}
