//! Headless physics tests: the greedy-box equivalence contract + collide-and-slide against the REAL
//! Cornell geometry (drop onto the floor, get blocked by a wall). No GPU / no App — pure rapier + brickmap.

use super::*;
use crate::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, BrickMap, VOXEL_SIZE, voxel_index};
use crate::voxel::cornell::{INTERIOR, build_cornell_with_edits};
use crate::voxel::edits::VoxelEdits;
use crate::voxel::palette::{BlockId, BlockRegistry};
use std::collections::HashSet;

fn solid() -> BlockId {
    BlockId(1)
}

/// A box-FREE (x, z) world spawn for the player capsule inside the Cornell interior. The two interior floor
/// boxes (`cornell::in_floor_box`) cover z∈[6, 34); the band z∈[34, INTERIOR) is clear of both, so a capsule
/// footprint centred there can drop/walk without wedging on them. Returns the capsule-CENTRE world (x, z):
/// x at the interior centre (the full +X lane is clear at this z), z at the clear-band centre. Derived from
/// the const layout so it tracks the VOXEL_SIZE flip + any box-layout change.
fn clear_back_lane() -> (f32, f32) {
    let x = INTERIOR as f32 * VOXEL_SIZE * 0.5; // interior centre in X — the +X walk lane is clear here
    // Clear-band centre in Z (between the boxes' far edge at voxel 34 and the back wall at voxel INTERIOR).
    let z_voxel = (34 + INTERIOR) as f32 * 0.5;
    (x, z_voxel * VOXEL_SIZE)
}

/// Build a brick from a per-voxel predicate (`true` = solid block 1, else air).
fn brick_from<F: Fn(i32, i32, i32) -> bool>(f: F) -> Brick {
    let mut v = Box::new([BlockId::AIR; BRICK_VOXELS]);
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                if f(x, y, z) {
                    v[voxel_index(x, y, z)] = solid();
                }
            }
        }
    }
    Brick::from_voxels(v)
}

/// The CONTRACT: the greedy boxes are pairwise disjoint and their union is EXACTLY the brick's solid
/// voxels. Verified by accumulating every covered cell into a set and comparing to the solid set, and by
/// checking the summed box volumes equal the solid count (no double-cover).
fn assert_boxes_partition(brick: &Brick) {
    let boxes = greedy_boxes(brick);
    let mut covered: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut volume = 0usize;
    for b in &boxes {
        for z in b.min.z..=b.max.z {
            for y in b.min.y..=b.max.y {
                for x in b.min.x..=b.max.x {
                    assert!(covered.insert((x, y, z)), "boxes overlap at {:?}", (x, y, z));
                    volume += 1;
                }
            }
        }
    }
    let mut solids: HashSet<(i32, i32, i32)> = HashSet::new();
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                if brick.is_solid(x, y, z) {
                    solids.insert((x, y, z));
                }
            }
        }
    }
    assert_eq!(covered, solids, "boxes must cover exactly the solid voxels");
    assert_eq!(volume, solids.len(), "summed box volume must equal solid count (disjoint cover)");
}

#[test]
fn greedy_uniform_solid_is_one_box() {
    let brick = Brick::uniform(solid());
    let boxes = greedy_boxes(&brick);
    assert_eq!(boxes.len(), 1, "a uniform-solid brick collapses to a single box");
    assert_eq!(boxes[0].min, IVec3::ZERO);
    assert_eq!(boxes[0].max, IVec3::splat(BRICK_EDGE - 1));
    assert_boxes_partition(&brick);
}

#[test]
fn greedy_empty_brick_has_no_boxes() {
    let brick = brick_from(|_, _, _| false);
    // (An all-air brick isn't normally stored, but greedy_boxes must still return nothing.)
    assert!(greedy_boxes(&brick).is_empty());
}

#[test]
fn greedy_partitions_a_floor_slab() {
    // One solid layer at y=0 (a floor): 64 cells, ideally a single 8×1×8 box.
    let brick = brick_from(|_, y, _| y == 0);
    let boxes = greedy_boxes(&brick);
    assert_boxes_partition(&brick);
    assert_eq!(boxes.len(), 1, "a flat slab is one box");
    assert_eq!(boxes[0].max.y, 0);
}

#[test]
fn greedy_partitions_a_hollow_and_checker() {
    // A brick with an interior hole.
    assert_boxes_partition(&brick_from(|x, y, z| !(x == 3 && y == 3 && z == 3)));
    // A 3D checkerboard — the worst case (every box is a single voxel); must still partition exactly.
    let checker = brick_from(|x, y, z| (x + y + z) % 2 == 0);
    assert_boxes_partition(&checker);
    // Checker has no two adjacent solids → one box per solid voxel (256 of 512).
    assert_eq!(greedy_boxes(&checker).len(), 256);
}

#[test]
fn empty_world_lets_the_character_fall() {
    let mut phys = VoxelColliders::default();
    phys.rebuild_from_bricks(&BrickMap::new());
    assert_eq!(phys.box_count, 0, "an empty brickmap builds no colliders");

    let ctrl = walk_controller();
    let (new_feet, grounded) =
        phys.move_character(&ctrl, Vec3::new(0.0, 5.0, 0.0), PLAYER_HALF, Vec3::new(0.0, -1.0, 0.0), 1.0 / 60.0);
    assert!(!grounded, "nothing to stand on");
    assert!(new_feet.y < 5.0, "the character falls through empty space (y {} < 5)", new_feet.y);
}

/// Drop the character into the real Cornell box and integrate gravity: it must come to rest ON the floor
/// (top at world y = 0), not pass through it.
#[test]
fn character_rests_on_the_cornell_floor() {
    let map = build_cornell_with_edits(&BlockRegistry::cornell(), &VoxelEdits::new());
    let mut phys = VoxelColliders::default();
    phys.rebuild_from_bricks(&map);
    assert!(phys.box_count > 0, "Cornell must produce colliders");

    let ctrl = walk_controller();
    let dt = 1.0 / 60.0;
    // Spawn in the CLEAR BACK LANE, away from the two interior floor boxes. At the 0.05 m flip the box interior
    // is only 2.4 m and the player capsule is a fixed 0.5 m wide × 1.8 m tall — large relative to the box — so
    // the old box-CENTRE spawn now overlaps the tall/short floor boxes (`in_floor_box`: tall z∈[20,34), short
    // z∈[6,22)). The band z∈[34, INTERIOR) is clear of both; spawn the capsule's footprint inside it.
    let (sx, sz) = clear_back_lane();
    // Drop from a height that keeps the 1.8 m-tall capsule INSIDE the box (interior ceiling at
    // `INTERIOR·VOXEL_SIZE`): head = feet + 2·PLAYER_HALF.y must clear the ceiling. At 0.05 m the box interior
    // is only 2.4 m tall, so the legacy 3.0 m drop started the player ABOVE the ceiling — derive a drop height
    // that leaves a small gap below the ceiling. (At 0.2 m this resolves to a comfortable mid-box drop.)
    let ceiling = INTERIOR as f32 * VOXEL_SIZE;
    let drop_from = (ceiling - 2.0 * PLAYER_HALF.y - 0.1).max(0.3);
    let mut feet = Vec3::new(sx, drop_from, sz);
    let mut vy = 0.0f32;
    for _ in 0..300 {
        // Ground probe → gravity integrate → move.
        let grounded = phys.move_character(&ctrl, feet, PLAYER_HALF, Vec3::new(0.0, -0.01, 0.0), dt).1;
        if grounded && vy < 0.0 {
            vy = 0.0;
        }
        vy -= GRAVITY * dt;
        let (nf, g2) = phys.move_character(&ctrl, feet, PLAYER_HALF, Vec3::new(0.0, vy * dt, 0.0), dt);
        feet = nf;
        if g2 && vy < 0.0 {
            vy = 0.0;
        }
    }
    assert!(feet.y.abs() < 0.2, "the character should rest on the floor (y≈0), got {}", feet.y);
    assert!(feet.y > -0.25, "the character must NOT sink through the floor, got {}", feet.y);
}

/// Walking into the +X (green) wall must be blocked: the interior ends at `INTERIOR·VOXEL_SIZE` m, so the
/// player's feet can't cross past `interior_x − half_x`.
#[test]
fn character_is_blocked_by_a_wall() {
    let map = build_cornell_with_edits(&BlockRegistry::cornell(), &VoxelEdits::new());
    let mut phys = VoxelColliders::default();
    phys.rebuild_from_bricks(&map);

    let ctrl = walk_controller();
    let dt = 1.0 / 60.0;
    // Spawn in the clear back lane (see `clear_back_lane`) so the box-sized capsule isn't wedged on the
    // interior floor boxes at the 0.05 m scale; start near the −X (red) wall so there is a long +X run to the
    // +X (green) wall. The whole +X lane is clear of boxes at this z.
    let (_, sz) = clear_back_lane();
    let start_x = INTERIOR as f32 * VOXEL_SIZE * 0.25; // a quarter in — room to walk toward +X
    let mut feet = Vec3::new(start_x, 0.0, sz);
    // Push hard toward +X (with a little down-pull to stay grounded) for plenty of steps.
    for _ in 0..400 {
        let desired = Vec3::new(WALK_SPEED * dt, -0.05, 0.0);
        feet = phys.move_character(&ctrl, feet, PLAYER_HALF, desired, dt).0;
    }
    let interior_x = INTERIOR as f32 * VOXEL_SIZE; // 2.4 m at 0.05 m
    assert!(feet.x < interior_x - PLAYER_HALF.x + 0.05, "blocked before the +X wall, got x={}", feet.x);
    assert!(feet.x > start_x, "but the player did move toward +X, got x={}", feet.x);
}
