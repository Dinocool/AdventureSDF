//! Unit tests for the edit delta, the per-brick overlay, the dirty-brick set, and the CPU pick DDA.

use super::*;
use crate::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, BrickMap, VOXEL_SIZE, voxel_index};
use crate::voxel::palette::BlockId;

fn solid(n: u16) -> BlockId {
    BlockId(n)
}

/// A base map with a single solid brick at `coord`, every voxel `block`.
fn solid_brick_map(coord: IVec3, block: BlockId) -> BrickMap {
    let mut m = BrickMap::new();
    m.insert(coord, Brick::uniform(block));
    m
}

/// place/remove/clear mutate the delta + bump the generation; resolve applies `base unless overridden`.
#[test]
fn delta_resolve_and_generation() {
    let mut e = VoxelEdits::new();
    assert!(e.is_empty());
    let g0 = e.generation();

    e.place(IVec3::new(1, 2, 3), solid(7));
    assert!(!e.is_empty());
    assert_eq!(e.len(), 1);
    assert!(e.generation() > g0, "place bumps the generation");
    // Override wins; an unrelated voxel falls through to base.
    assert_eq!(e.resolve(IVec3::new(1, 2, 3), solid(99)), solid(7));
    assert_eq!(e.resolve(IVec3::new(4, 4, 4), solid(99)), solid(99));

    let g1 = e.generation();
    e.remove(IVec3::new(5, 5, 5));
    assert!(e.generation() > g1, "remove bumps the generation");
    // A removed voxel resolves to AIR even over a solid base.
    assert_eq!(e.resolve(IVec3::new(5, 5, 5), solid(3)), BlockId::AIR);

    let g2 = e.generation();
    e.clear(IVec3::new(1, 2, 3));
    assert!(e.generation() > g2, "clear of a present override bumps");
    assert_eq!(e.resolve(IVec3::new(1, 2, 3), solid(99)), solid(99), "cleared → base shows through");
    // Clearing an absent override does NOT bump.
    let g3 = e.generation();
    e.clear(IVec3::new(123, 45, 6));
    assert_eq!(e.generation(), g3, "clear of an absent override is a no-op");
}

/// `apply_edit_overlay` carves a removed voxel out of a uniform-solid brick and repaints a placed voxel.
#[test]
fn overlay_carves_and_paints_a_brick() {
    let coord = IVec3::new(0, 0, 0);
    let base = Brick::uniform(solid(1));
    let mut e = VoxelEdits::new();
    // Remove the world voxel at local (2,3,4); place block 9 at local (5,6,7).
    let origin = coord * BRICK_EDGE;
    e.remove(origin + IVec3::new(2, 3, 4));
    e.place(origin + IVec3::new(5, 6, 7), solid(9));

    let out = apply_edit_overlay(coord, &base, &e);
    assert_eq!(out.get(2, 3, 4), BlockId::AIR, "removed voxel is now air");
    assert!(!out.is_solid(2, 3, 4));
    assert_eq!(out.get(5, 6, 7), solid(9), "placed voxel repainted");
    assert_eq!(out.get(0, 0, 0), solid(1), "untouched voxel keeps the base block");
}

/// `apply_edits_to_map` seeds a FRESH brick when a block is placed into previously-empty space (a brick the
/// base map never stored), and carves a base brick where a remove lands.
#[test]
fn map_overlay_seeds_empty_space_and_carves() {
    let base_coord = IVec3::new(0, 0, 0);
    let base = solid_brick_map(base_coord, solid(1));
    let mut e = VoxelEdits::new();

    // Place into a far-away empty brick.
    let far = IVec3::new(100, 50, 7);
    e.place(far, solid(5));
    // Remove a voxel in the existing base brick.
    e.remove(IVec3::new(3, 3, 3));

    let out = apply_edits_to_map(&base, &e);
    // The far brick now exists with exactly that one placed voxel.
    let far_bc = brick_coord_of_voxel(far);
    let far_brick = out.get(far_bc).expect("placed-into-empty-space brick must be created");
    let far_local = far - far_bc * BRICK_EDGE;
    assert_eq!(far_brick.get(far_local.x, far_local.y, far_local.z), solid(5));
    assert!(far_brick.get(0, 0, 0).is_air(), "the rest of the seeded brick is air");
    // The base brick has the hole.
    assert_eq!(out.voxel_block(IVec3::new(3, 3, 3)), BlockId::AIR, "removed voxel dug out of the base brick");
    assert_eq!(out.voxel_block(IVec3::new(0, 0, 0)), solid(1), "neighbouring base voxel intact");
}

/// A remove into empty space is a no-op (digs air, creates no brick).
#[test]
fn remove_into_empty_space_is_noop() {
    let base = BrickMap::new();
    let mut e = VoxelEdits::new();
    e.remove(IVec3::new(9, 9, 9));
    let out = apply_edits_to_map(&base, &e);
    assert!(out.is_empty(), "removing air creates no brick");
}

/// An INTERIOR voxel makes only its owning brick dirty.
#[test]
fn dirty_interior_is_single_brick() {
    // Local (3,4,5) in brick (0,0,0) — interior on every axis.
    let v = IVec3::new(3, 4, 5);
    let dirty = dirty_bricks_for_edit(v);
    assert_eq!(dirty, vec![IVec3::new(0, 0, 0)], "interior voxel → owner only");
}

/// A FACE voxel makes the owner + the one face-neighbour dirty (the neighbour's halo reads it).
#[test]
fn dirty_face_includes_neighbour() {
    // Local x==0 face of brick (0,0,0): world voxel (0, 4, 4). The low-X neighbour is brick (-1,0,0).
    let v = IVec3::new(0, 4, 4);
    let mut dirty = dirty_bricks_for_edit(v);
    dirty.sort_by_key(|c| (c.x, c.y, c.z));
    assert_eq!(dirty, vec![IVec3::new(-1, 0, 0), IVec3::new(0, 0, 0)]);

    // The high-X face (local x == BRICK_EDGE-1) brings in the +X neighbour instead.
    let v2 = IVec3::new(BRICK_EDGE - 1, 4, 4);
    let mut dirty2 = dirty_bricks_for_edit(v2);
    dirty2.sort_by_key(|c| (c.x, c.y, c.z));
    assert_eq!(dirty2, vec![IVec3::new(0, 0, 0), IVec3::new(1, 0, 0)]);
}

/// A CORNER voxel touches all 8 incident bricks (owner + 7 diagonal neighbours).
#[test]
fn dirty_corner_is_eight_bricks() {
    // Local (0,0,0) corner of brick (0,0,0): the low corner touches bricks with each axis offset in {0,-1}.
    let v = IVec3::new(0, 0, 0);
    let dirty = dirty_bricks_for_edit(v);
    assert_eq!(dirty.len(), 8, "a corner voxel touches 8 bricks, got {dirty:?}");
    for dz in [-1, 0] {
        for dy in [-1, 0] {
            for dx in [-1, 0] {
                assert!(dirty.contains(&IVec3::new(dx, dy, dz)), "missing brick ({dx},{dy},{dz})");
            }
        }
    }
}

/// The CPU pick hits the first solid voxel along a +X ray through a wall of voxels, and the entry FACE normal
/// points back toward the camera (−X for a +X ray hitting the wall's −X face).
#[test]
fn pick_hits_first_solid_with_face() {
    // A solid brick at (0,0,0): world voxels [0,8) on each axis are solid block 1.
    let base = solid_brick_map(IVec3::new(0, 0, 0), solid(1));
    let e = VoxelEdits::new();

    // Ray from x = -2 m straight +X at the mid-height/depth of the brick. The brick's −X face is at world
    // x=0 → voxel 0. The first solid voxel is (0, vy, vz); the entry face is −X.
    let mid = (BRICK_EDGE / 2) as f32 * VOXEL_SIZE; // ~0.8 m, inside the brick on Y/Z
    let origin = Vec3::new(-2.0, mid, mid);
    let hit = pick_voxel(&base, &e, origin, Vec3::X, 100.0).expect("ray must hit the wall");
    assert_eq!(hit.voxel.x, 0, "first solid voxel is the −X column at voxel x=0");
    assert_eq!(hit.normal, IVec3::new(-1, 0, 0), "entry face is the −X face (toward the camera)");
    assert_eq!(hit.block, solid(1));
    // The PLACE target is the air voxel just outside that face.
    assert_eq!(hit.place_target(), IVec3::new(-1, hit.voxel.y, hit.voxel.z));
    // t is ~2 m (origin at -2, face at 0).
    assert!((hit.t - 2.0).abs() < 0.05, "hit t ≈ 2 m, got {}", hit.t);
}

/// A pick into open space (no geometry within range) returns None.
#[test]
fn pick_misses_empty_space() {
    let base = BrickMap::new();
    let e = VoxelEdits::new();
    assert!(pick_voxel(&base, &e, Vec3::ZERO, Vec3::X, 50.0).is_none());
}

/// REMOVING a surface voxel makes the pick pass THROUGH it to the voxel behind (the edit is consulted by the
/// pick, not just the base map).
#[test]
fn pick_respects_removed_surface_voxel() {
    let base = solid_brick_map(IVec3::new(0, 0, 0), solid(1));
    let mut e = VoxelEdits::new();
    let mid = (BRICK_EDGE / 2) as f32 * VOXEL_SIZE;
    let vy = (mid / VOXEL_SIZE).floor() as i32;
    let vz = vy;
    // Remove the −X surface column voxel the +X ray would hit first.
    e.remove(IVec3::new(0, vy, vz));

    let origin = Vec3::new(-2.0, mid, mid);
    let hit = pick_voxel(&base, &e, origin, Vec3::X, 100.0).expect("ray still hits the next voxel behind");
    assert_eq!(hit.voxel.x, 1, "the removed voxel 0 is skipped; voxel 1 is the new surface");
    assert_eq!(hit.normal, IVec3::new(-1, 0, 0), "entry face still the −X face of voxel 1");
}

/// PLACING a voxel into empty space makes the pick hit it (overlay is consulted before the base map).
#[test]
fn pick_hits_placed_voxel_in_empty_space() {
    let base = BrickMap::new(); // empty world
    let mut e = VoxelEdits::new();
    // Place a single voxel at world voxel (10, 0, 0).
    e.place(IVec3::new(10, 0, 0), solid(4));

    // Ray from x=-1 m straight +X at y,z inside that voxel's cell ([10,11)·0.2 = [2.0, 2.2) m → y,z ~0.1 m).
    let origin = Vec3::new(-1.0, 0.1, 0.1);
    let hit = pick_voxel(&base, &e, origin, Vec3::X, 100.0).expect("must hit the placed voxel");
    assert_eq!(hit.voxel, IVec3::new(10, 0, 0));
    assert_eq!(hit.block, solid(4));
    assert_eq!(hit.normal, IVec3::new(-1, 0, 0));
}

/// The pick's entry face matches the dominant approach axis: a −Y ray onto a floor's TOP reads the +Y face.
#[test]
fn pick_top_face_from_above() {
    // A solid slab brick at (0,0,0). A ray coming straight DOWN (−Y) from above the brick hits the top
    // (+Y) face of the topmost voxel (local y = BRICK_EDGE-1 → world voxel y = 7).
    let base = solid_brick_map(IVec3::new(0, 0, 0), solid(1));
    let e = VoxelEdits::new();
    let mid = (BRICK_EDGE / 2) as f32 * VOXEL_SIZE;
    let origin = Vec3::new(mid, 5.0, mid); // well above the 1.6 m-tall brick
    let hit = pick_voxel(&base, &e, origin, -Vec3::Y, 100.0).expect("downward ray hits the slab top");
    assert_eq!(hit.voxel.y, BRICK_EDGE - 1, "hits the topmost voxel row");
    assert_eq!(hit.normal, IVec3::new(0, 1, 0), "entry face is the +Y top face");
}

/// `apply_edit_overlay` is consistent with what a full re-voxelize would produce: overlay then read == resolve.
#[test]
fn overlay_matches_resolve_per_voxel() {
    let coord = IVec3::new(2, -1, 3);
    // A dense base brick with a varied pattern.
    let mut base_voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                if (x + y + z) % 2 == 0 {
                    base_voxels[voxel_index(x, y, z)] = solid(2);
                }
            }
        }
    }
    let base = Brick::from_voxels(base_voxels);
    let origin = coord * BRICK_EDGE;
    let mut e = VoxelEdits::new();
    e.place(origin + IVec3::new(0, 0, 0), solid(5));
    e.remove(origin + IVec3::new(2, 2, 2));

    let out = apply_edit_overlay(coord, &base, &e);
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                let wv = origin + IVec3::new(x, y, z);
                assert_eq!(out.get(x, y, z), e.resolve(wv, base.get(x, y, z)), "overlay == resolve at {wv}");
            }
        }
    }
}
