//! Headless verification that the GALLERY scene — several voxelized `.vox` scenes MERGED side by side into ONE
//! [`BrickMap`] — streams through the EXACT SAME camera-following clipmap residency as Sponza / worldgen. The
//! gallery is the pre-instancing MERGE (true per-object instancing isn't built yet): `gallery::merge_scenes`
//! places each loaded scene in a non-overlapping region and concatenates the palettes, and the result is fed to
//! a [`StaticVoxSource`] + [`ResidencyManager`] — identical to the Sponza path.
//!
//! These tests drive the merge + residency units directly (synthetic per-scene maps, no `.vox` files needed),
//! so they pin:
//!
//!   * the MERGE places two scenes SIDE BY SIDE with NO cross-scene brick overlap, both present at their
//!     offsets, with per-scene palettes preserved (each scene's blocks land in its own merged-id range) and
//!     the brick + solid-voxel counts adding up;
//!   * the merged map STREAMS through the clipmap (entering the region enqueues work; the floors stream in as
//!     resident bricks) and resident bricks SOURCE from the MERGED map (their voxels equal
//!     `StaticVoxSource::brick` over the merged map) — proving the gallery routes through the same residency.

use bevy::math::IVec3;

use adventure::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, BrickMap, brick_coord_of_voxel, voxel_index};
use adventure::voxel::edits::VoxelEdits;
use adventure::voxel::gallery::{LoadedScene, merge_scenes};
use adventure::voxel::gpu::pack_resident_set;
use adventure::voxel::palette::{BlockId, BlockRegistry};
use adventure::voxel::source::{BrickSource, StaticVoxSource};
use adventure::voxel::streaming::{ResidencyManager, StreamingConfig};

/// A registry with `n` solid blocks (the per-scene `.vox`-style palette stand-in). `from_vox_palette` mirrors
/// what `load_vox` builds: AIR + one opaque block per palette colour.
fn registry_with_blocks(n: usize) -> BlockRegistry {
    let colors: Vec<[u8; 4]> = (0..n).map(|i| [(i * 30) as u8 + 20, 100, 150, 255]).collect();
    BlockRegistry::from_vox_palette(&colors)
}

/// A small worldgen-style registry (any registry works — the static source ignores it for block lookups; the
/// PACK uses the active registry's palette length). Used only where a registry is required by the API.
fn merged_pack_registry(reg: &BlockRegistry) -> &BlockRegistry {
    reg
}

/// Build a fine `BrickMap`: a solid floor slab `y∈[0,2)` over a footprint `[−sz, sz)` in X and Z, every voxel
/// `block`. Stands in for one loaded `.vox` scene (floor-anchored + X/Z-centred, like the real loader). The
/// `block` is the scene's LOCAL BlockId (>=1); the merge remaps it into the merged palette range.
fn floor_scene(sz: i32, block: BlockId) -> BrickMap {
    use std::collections::HashMap;
    let mut dense: HashMap<IVec3, Box<[BlockId; BRICK_VOXELS]>> = HashMap::new();
    for z in -sz..sz {
        for y in 0..2 {
            for x in -sz..sz {
                let wv = IVec3::new(x, y, z);
                let bc = brick_coord_of_voxel(wv);
                let local = wv - bc * BRICK_EDGE;
                let arr = dense.entry(bc).or_insert_with(|| Box::new([BlockId::AIR; BRICK_VOXELS]));
                arr[voxel_index(local.x, local.y, local.z)] = block;
            }
        }
    }
    let mut map = BrickMap::new();
    for (c, arr) in dense {
        map.insert(c, Brick::from_voxels(arr));
    }
    map
}

/// Total solid voxel count of a map.
fn solid_voxels(map: &BrickMap) -> usize {
    let mut n = 0;
    for (_bc, brick) in map.iter() {
        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                for x in 0..BRICK_EDGE {
                    if brick.is_solid(x, y, z) {
                        n += 1;
                    }
                }
            }
        }
    }
    n
}

/// All (world-voxel, BlockId) solids of a map.
fn solids(map: &BrickMap) -> Vec<(IVec3, BlockId)> {
    let mut out = Vec::new();
    for (bc, brick) in map.iter() {
        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                for x in 0..BRICK_EDGE {
                    if brick.is_solid(x, y, z) {
                        out.push((*bc * BRICK_EDGE + IVec3::new(x, y, z), brick.get(x, y, z)));
                    }
                }
            }
        }
    }
    out
}

/// The MERGE places two synthetic scenes SIDE BY SIDE with no cross-scene brick overlap: both scenes are
/// present at their (auto-spaced) offsets, each scene's palette is preserved in its own merged-id range, and
/// the brick + solid-voxel counts add up (nothing lost to a collision). The core gallery guarantee.
#[test]
fn merge_two_scenes_side_by_side_palettes_preserved() {
    // Scene A: a 32-wide floor of local block 1; its own 3-colour palette (AIR + 3).
    let a_map = floor_scene(16, BlockId(1));
    let a_reg = registry_with_blocks(3);
    // Scene B: a 32-wide floor of local block 2; its own 4-colour palette (AIR + 4).
    let b_map = floor_scene(16, BlockId(2));
    let b_reg = registry_with_blocks(4);

    let a_bricks = a_map.len();
    let b_bricks = b_map.len();
    let a_solids = solid_voxels(&a_map);
    let b_solids = solid_voxels(&b_map);

    let scenes = vec![
        LoadedScene { map: a_map, registry: a_reg, offset: None, label: "A".into() },
        LoadedScene { map: b_map, registry: b_reg, offset: None, label: "B".into() },
    ];
    let (merged, reg) = merge_scenes(scenes);

    // Palettes concatenated: AIR + 3 (A) + 4 (B) = 8 blocks. A keeps ids 1..=3; B is remapped to 4..=7.
    assert_eq!(reg.len(), 1 + 3 + 4, "merged registry concatenates both palettes (AIR + 3 + 4)");

    // Counts add up — no bricks/voxels lost to a cross-scene collision (the auto-spacing gap guarantees it).
    assert_eq!(merged.len(), a_bricks + b_bricks, "merged brick count = sum (disjoint regions)");
    assert_eq!(solid_voxels(&merged), a_solids + b_solids, "merged solid voxel count = sum");

    // Both scenes present, remapped: A's block 1 stays id 1; B's block 2 lands in the appended B range (>3).
    let merged_ids: std::collections::HashSet<u16> = solids(&merged).iter().map(|(_, b)| b.0).collect();
    assert!(merged_ids.contains(&1), "scene A's block 1 is present at its merged id 1");
    // B's local block 2 → merged id `(palette_base_B − 1) + 2`. palette_base_B = merged.len() before B = 1+3 = 4
    // (AIR + A's 3 blocks), so the shift is 3 and B's local block 2 lands at merged id 5.
    assert!(merged_ids.contains(&(4 - 1 + 2)), "scene B's block 2 is present at its remapped merged id 5");
    // No id is outside the merged palette range.
    assert!(merged_ids.iter().all(|&id| (id as usize) < reg.len()), "every merged voxel id indexes the palette");

    // NO cross-scene brick overlap: split the merged bricks by their +X position. A is centred on the origin
    // (x∈[−2,1] bricks for a 32-voxel footprint); B is auto-spaced strictly past A + a gap (x ≥ some positive
    // column). The two sets of brick X-coords must be disjoint with a clear gap between them.
    let mut a_max_x = i32::MIN;
    let mut b_min_x = i32::MAX;
    for (bc, brick) in merged.iter() {
        // A scene-A brick carries id 1; a scene-B brick carries its remapped id (5). (Each brick is uniform.)
        let id = {
            let mut found = 0u16;
            'outer: for z in 0..BRICK_EDGE {
                for y in 0..BRICK_EDGE {
                    for x in 0..BRICK_EDGE {
                        if brick.is_solid(x, y, z) {
                            found = brick.get(x, y, z).0;
                            break 'outer;
                        }
                    }
                }
            }
            found
        };
        if id == 1 {
            a_max_x = a_max_x.max(bc.x);
        } else {
            b_min_x = b_min_x.min(bc.x);
        }
    }
    assert!(b_min_x > a_max_x + 1, "scene B's bricks start strictly past scene A's +X bound with a gap (no overlap)");
}

/// The MERGED map STREAMS through the clipmap residency and resident bricks SOURCE from the merged map: with
/// the camera over scene A's floor, entering the region enqueues work, the floor streams in as resident bricks,
/// and every resident LOD0 brick equals exactly `StaticVoxSource::brick` over the MERGED map — proving the
/// gallery routes through the identical residency Sponza uses (and the packed set references the merged palette).
#[test]
fn merged_map_streams_through_clipmap_and_sources_from_merge() {
    let a_map = floor_scene(16, BlockId(1));
    let a_reg = registry_with_blocks(3);
    let b_map = floor_scene(16, BlockId(2));
    let b_reg = registry_with_blocks(4);
    let (merged, reg) = merge_scenes(vec![
        LoadedScene { map: a_map, registry: a_reg, offset: None, label: "A".into() },
        LoadedScene { map: b_map, registry: b_reg, offset: None, label: "B".into() },
    ]);

    // Stream the MERGED map exactly as the Sponza path does: a StaticVoxSource over the merged map + a
    // ResidencyManager clipmap.
    let src = StaticVoxSource::new(&merged);
    let edits = VoxelEdits::new();
    let cfg = StreamingConfig { clip_half_bricks: 4, max_resident_bricks: 1_000_000, max_bricks_per_frame: 1_000_000 };
    let mut mgr = ResidencyManager::new();

    // Camera just above scene A's floor at the origin (A is centred on the origin like standalone Sponza).
    let cam = [0.4_f32, 1.0, 0.4];
    mgr.update(cam, &cfg);
    assert!(mgr.pending() > 0, "entering the merged-map clipmap enqueues work");
    while mgr.pending() > 0 {
        mgr.drain_work_from(&cfg, &src, merged_pack_registry(&reg), &edits);
    }
    assert!(mgr.resident_count() > 0, "scene A's floor streams in as resident bricks");

    // Every resident LOD0 brick equals the source brick over the MERGED map at its key (the residency stored
    // exactly what the merged-map source produced — the gallery sources from the merge).
    let mut checked = 0;
    for e in mgr.resident_entries() {
        if e.lod != 0 {
            continue;
        }
        assert_eq!(
            *e.brick,
            src.brick(e.coord, 0, &reg),
            "resident LOD0 brick {:?} must equal StaticVoxSource::brick over the MERGED map",
            e.coord
        );
        checked += 1;
    }
    assert!(checked > 0, "the inner LOD0 cube has resident bricks sourced from the merged map");

    // The packed resident set is non-empty and references the MERGED palette (its length mirrors the merged
    // registry), and contains scene A's block (id 1) — the merged geometry made it through the residency.
    let entries = mgr.resident_entries();
    let patch = pack_resident_set(&entries, &reg);
    assert!(patch.brick_count() > 0, "the merged scene packs a non-empty resident set");
    assert_eq!(patch.palette.len(), reg.len(), "the packed palette mirrors the MERGED registry");
    assert!(patch.voxels.contains(&1), "the packed voxels contain scene A's merged block id 1");
}
