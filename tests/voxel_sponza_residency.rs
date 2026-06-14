//! Headless verification that a STATIC `.vox` scene (Sponza) now runs the SAME camera-following clipmap
//! residency as worldgen — the unify-the-pipeline refactor. The bespoke one-shot `pack_brickmap` Sponza path
//! is gone; a static [`BrickMap`] is streamed through [`ResidencyManager`] via a [`StaticVoxSource`], with the
//! shared [`VoxelEdits`] overlay applied inside `drain_work_from`. These tests drive the exact units the
//! Sponza routing system uses (minus the Bevy system shell), so they pin:
//!
//!   * the clipmap residency COVERS the bounded static extent and BOUNDS it (bricks outside the loaded map
//!     are all-air — never resident — so the clipmap naturally limits the scene to the building);
//!   * resident bricks SOURCE from the loaded map (their voxels equal `StaticVoxSource::brick`);
//!   * an EDIT applies through the shared overlay and RE-PACKS the affected resident bricks locally (adapt,
//!     not full-clear).
//!
//! A synthetic static map (a bounded solid floor + a column) stands in for the real `assets/models/sponza.vox`
//! so the test is self-contained + fast (the real asset is exercised by `voxel::vox::tests` + the GPU suite).

use bevy::math::IVec3;

use adventure::sdf_render::worldgen::biome::{
    BiomeDef, BiomeId, BiomeLibrary, StrataLayer, TerrainMatId, TerrainSurfaceMaterial,
};
use adventure::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, BrickMap, voxel_index};
use adventure::voxel::edits::VoxelEdits;
use adventure::voxel::gpu::pack_resident_set;
use adventure::voxel::palette::{BlockId, BlockRegistry};
use adventure::voxel::source::{BrickSource, StaticVoxSource};
use adventure::voxel::streaming::{BrickKey, ResidencyManager, StreamingConfig};

/// A small registry with a couple of solid blocks (the static-scene palette stand-in).
fn registry() -> BlockRegistry {
    let mat = |name: &str, c: [f32; 4]| TerrainSurfaceMaterial {
        name: name.into(),
        base_color: c,
        roughness: 0.9,
        blend: 0.0,
        texture: None,
        tiling: 4.0,
        ..Default::default()
    };
    let materials = vec![mat("floor", [0.6, 0.5, 0.4, 1.0]), mat("column", [0.8, 0.8, 0.85, 1.0])];
    let biomes = BiomeId::ALL
        .iter()
        .map(|_| BiomeDef {
            name: "b".into(),
            surface: TerrainMatId(0),
            surface_rules: vec![],
            strata: vec![StrataLayer { material: TerrainMatId(0), thickness: 1.0 }],
            bedrock: TerrainMatId(1),
        })
        .collect();
    BlockRegistry::from_biome_library(&BiomeLibrary { materials, biomes })
}

/// The synthetic static scene's solid block at a WORLD voxel (None = air). A bounded "building": a solid
/// FLOOR slab (y∈[0,2)) over a footprint x∈[-16,16), z∈[-16,16), plus a COLUMN (block 2) at x∈[0,2), z∈[0,2)
/// rising y∈[0,32). Floor at y=0, centred on X/Z — the same anchor the `.vox` loader uses for Sponza.
fn scene_voxel(wv: IVec3) -> Option<BlockId> {
    let in_footprint = (-16..16).contains(&wv.x) && (-16..16).contains(&wv.z);
    if in_footprint && (0..2).contains(&wv.y) {
        return Some(BlockId(1)); // floor slab
    }
    if (0..2).contains(&wv.x) && (0..2).contains(&wv.z) && (0..32).contains(&wv.y) {
        return Some(BlockId(2)); // column
    }
    None
}

/// Build the synthetic static `BrickMap` from [`scene_voxel`] over its bounded extent — the stand-in for the
/// loaded `sponza.vox` (a `BrickMap` of `0.2 m` voxels). Bounds chosen to cover the floor + column with slack.
fn static_scene_map() -> BrickMap {
    use std::collections::HashMap;
    let mut dense: HashMap<IVec3, Box<[BlockId; BRICK_VOXELS]>> = HashMap::new();
    for z in -20..20 {
        for y in -2..34 {
            for x in -20..20 {
                let wv = IVec3::new(x, y, z);
                let Some(b) = scene_voxel(wv) else { continue };
                let bc = IVec3::new(
                    x.div_euclid(BRICK_EDGE),
                    y.div_euclid(BRICK_EDGE),
                    z.div_euclid(BRICK_EDGE),
                );
                let local = wv - bc * BRICK_EDGE;
                let arr = dense.entry(bc).or_insert_with(|| Box::new([BlockId::AIR; BRICK_VOXELS]));
                arr[voxel_index(local.x, local.y, local.z)] = b;
            }
        }
    }
    let mut map = BrickMap::new();
    for (c, arr) in dense {
        map.insert(c, Brick::from_voxels(arr));
    }
    map
}

/// Drain the whole pending queue (no per-frame cap) so the test sees the settled resident set, applying the
/// shared edit overlay. Mirrors the production drain (just unbounded for the test).
fn drain_all(mgr: &mut ResidencyManager, cfg: &StreamingConfig, src: &dyn BrickSource, reg: &BlockRegistry, edits: &VoxelEdits) {
    while mgr.pending() > 0 {
        mgr.drain_work_from(cfg, src, reg, edits);
    }
}

/// The clipmap residency over a STATIC map covers the building AND bounds it: with the camera placed at the
/// scene, the resident set is non-empty (the floor + column stream in) and EVERY resident brick lies within
/// the loaded map's world-voxel extent — bricks beyond the building never become resident (sourced all-air,
/// memoized empty). This is the "static scene now supports clipmaps + is bounded" guarantee.
#[test]
fn static_residency_covers_and_bounds_the_scene() {
    let map = static_scene_map();
    let reg = registry();
    let src = StaticVoxSource::new(&map);
    let edits = VoxelEdits::new();
    // A tight clipmap so the test is fast but still has coarse shells reaching past the building.
    let cfg = StreamingConfig { clip_half_bricks: 4, max_resident_bricks: 1_000_000, max_bricks_per_frame: 1_000_000 };

    let mut mgr = ResidencyManager::new();
    // Camera just above the floor near the column, looking into the scene.
    let cam = [0.4_f32, 1.0, 0.4];
    mgr.update(cam, &cfg);
    assert!(mgr.pending() > 0, "entering the static clipmap enqueues work");
    drain_all(&mut mgr, &cfg, &src, &reg, &edits);
    assert!(mgr.resident_count() > 0, "the floor + column stream in as resident bricks");

    // The loaded map's brick-coord bounds (the building extent). Every resident brick coord must lie within
    // these bounds — a brick beyond the building was sourced all-air and is never resident (the clipmap bound).
    let mut bc_lo = IVec3::splat(i32::MAX);
    let mut bc_hi = IVec3::splat(i32::MIN);
    for (bc, _b) in map.iter() {
        bc_lo = bc_lo.min(*bc);
        bc_hi = bc_hi.max(*bc);
    }
    for e in mgr.resident_entries() {
        // Map the resident (coord, lod) back to the LOD0 brick-coord span it covers and assert it overlaps the
        // building's bounds — a resident brick wholly outside the building would be all-air (impossible here).
        let scale = 1i32 << e.lod;
        let lo0 = e.coord * scale; // the LOD0 brick coords this brick covers: [lo0, lo0+scale)
        let hi0 = lo0 + IVec3::splat(scale) - IVec3::ONE;
        let overlaps = lo0.x <= bc_hi.x && hi0.x >= bc_lo.x
            && lo0.y <= bc_hi.y && hi0.y >= bc_lo.y
            && lo0.z <= bc_hi.z && hi0.z >= bc_lo.z;
        assert!(overlaps, "resident brick {:?}@lod{} must overlap the building extent (the clipmap bounds the scene)", e.coord, e.lod);
    }
}

/// Resident bricks SOURCE from the loaded map: every resident LOD0 brick's voxels equal exactly what
/// `StaticVoxSource::brick` produces from the map (the floor slab / column voxels), and the packed resident
/// set reproduces them. Proves the static scene's geometry flows through the SAME residency + packing the
/// worldgen scene uses.
#[test]
fn resident_bricks_source_from_the_loaded_map() {
    let map = static_scene_map();
    let reg = registry();
    let src = StaticVoxSource::new(&map);
    let edits = VoxelEdits::new();
    let cfg = StreamingConfig { clip_half_bricks: 4, max_resident_bricks: 1_000_000, max_bricks_per_frame: 1_000_000 };

    let mut mgr = ResidencyManager::new();
    let cam = [0.4_f32, 1.0, 0.4];
    mgr.update(cam, &cfg);
    drain_all(&mut mgr, &cfg, &src, &reg, &edits);

    // Every resident LOD0 brick equals the source brick at its key — the residency stored exactly what the
    // source produced (no transformation in between).
    let mut checked = 0;
    for e in mgr.resident_entries() {
        if e.lod != 0 {
            continue;
        }
        let want = src.brick(e.coord, 0, &reg);
        assert_eq!(*e.brick, want, "resident LOD0 brick {:?} must equal StaticVoxSource::brick", e.coord);
        checked += 1;
    }
    assert!(checked > 0, "the inner LOD0 cube has resident bricks sourced from the map");

    // The packed resident set is non-empty and references the static palette (the floor + column blocks).
    let entries = mgr.resident_entries();
    let patch = pack_resident_set(&entries, &reg);
    assert!(patch.brick_count() > 0, "the static scene packs a non-empty resident set");
    assert_eq!(patch.palette.len(), reg.len(), "the packed palette mirrors the static registry");
    // Some packed voxel is the floor block (1) or the column block (2) — the map's geometry made it through.
    assert!(
        patch.voxels.iter().any(|&v| v == 1 || v == 2),
        "the packed voxels contain the static scene's solid blocks"
    );
}

/// An EDIT applies through the SHARED overlay + the resident set RE-PACKS locally: place a block into an air
/// voxel adjacent to the floor, re-queue the affected resident bricks (the production `requeue_keys` path),
/// re-drain WITH the edit delta, and assert the owning resident brick now carries the placed block — and the
/// re-packed set differs from the un-edited one. The residency ADAPTS (the rest of the set is untouched).
#[test]
fn edit_applies_and_repacks_through_the_residency() {
    let map = static_scene_map();
    let reg = registry();
    let src = StaticVoxSource::new(&map);
    let cfg = StreamingConfig { clip_half_bricks: 3, max_resident_bricks: 1_000_000, max_bricks_per_frame: 1_000_000 };

    // Initial residency with NO edits.
    let mut mgr = ResidencyManager::new();
    let cam = [0.4_f32, 1.0, 0.4];
    mgr.update(cam, &cfg);
    drain_all(&mut mgr, &cfg, &src, &reg, &VoxelEdits::new());
    mgr.take_dirty();
    let before = pack_resident_set(&mgr.resident_entries(), &reg);

    // PLACE a block into a known-air voxel just above the floor at world voxel (3, 4, 3) — inside the inner
    // LOD0 cube, above the 2-voxel floor slab so it was air. Use the column block (2) so it's identifiable.
    let target = IVec3::new(3, 4, 3);
    assert!(scene_voxel(target).is_none(), "the place target must start as air");
    let mut edits = VoxelEdits::new();
    edits.place(target, BlockId(2));

    // The production edit path: re-queue the owner + halo-neighbour LOD0 bricks of the edit, then re-drain
    // WITH the delta (the shared overlay folds it in). This is what `affected_resident_keys` + `requeue_keys`
    // + `drain_work_from(.., &edits)` do in the Sponza/worldgen routing.
    let owner = IVec3::new(target.x.div_euclid(BRICK_EDGE), target.y.div_euclid(BRICK_EDGE), target.z.div_euclid(BRICK_EDGE));
    mgr.requeue_keys([BrickKey { coord: owner, lod: 0 }]);
    assert!(mgr.pending() > 0, "the edit re-queues the affected resident brick");
    drain_all(&mut mgr, &cfg, &src, &reg, &edits);

    // The owning resident brick now carries the placed block at the edited local voxel.
    let local = target - owner * BRICK_EDGE;
    let after_brick = mgr
        .resident_entries()
        .into_iter()
        .find(|e| e.coord == owner && e.lod == 0)
        .expect("the edited brick is resident");
    assert_eq!(
        after_brick.brick.get(local.x, local.y, local.z),
        BlockId(2),
        "the placed block is visible in the resident brick after the edit"
    );

    // The re-packed set DIFFERS from the pre-edit one (the edit re-packed) — but only locally (the brick count
    // is unchanged here; the voxel data changed). Compare the packed voxel buffers.
    let after = pack_resident_set(&mgr.resident_entries(), &reg);
    assert_ne!(before.voxels, after.voxels, "the edit changed the packed resident voxels (a local re-pack)");
}

/// The empty-memo BOUNDS a static scene cheaply: after the residency settles, the bricks sourced all-air
/// (outside the building) are memoized, so an idle re-`update` at the same camera enqueues NOTHING (no churn
/// re-sourcing the air around the building every frame) — the same churn fix worldgen relies on.
#[test]
fn outside_air_bricks_are_memoized_no_churn() {
    let map = static_scene_map();
    let reg = registry();
    let src = StaticVoxSource::new(&map);
    let edits = VoxelEdits::new();
    let cfg = StreamingConfig { clip_half_bricks: 4, max_resident_bricks: 1_000_000, max_bricks_per_frame: 1_000_000 };

    let mut mgr = ResidencyManager::new();
    // Place the camera so the clipmap straddles the building EDGE — the footprint ends at x=z=16, so world
    // (15, 1, 15) sits in the near corner with much of the clipmap reaching OUT past the building into air.
    let cam = [15.0_f32, 1.0, 15.0];
    mgr.update(cam, &cfg);
    drain_all(&mut mgr, &cfg, &src, &reg, &edits);
    mgr.take_dirty();

    // A re-update at the SAME camera position must enqueue nothing — the resident bricks are still resident and
    // the air bricks are memoized empty (not re-enqueued). This is the no-churn / bounded-static guarantee.
    mgr.update(cam, &cfg);
    assert_eq!(mgr.pending(), 0, "an idle re-update enqueues nothing (resident + empty-memo cover the clipmap)");
}
