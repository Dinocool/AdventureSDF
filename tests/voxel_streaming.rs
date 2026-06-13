//! Headless (CPU-only) verification of the Stage-3 streaming + LOD bookkeeping, run from the integration
//! test crate (which links the library's PUBLIC API, so it compiles even though some in-crate `#[cfg(test)]`
//! modules of pruned features don't). Mirrors the in-module unit tests, but guaranteed runnable here:
//!
//!   * mip downsample: majority block, thin-feature threshold (near keeps, far erodes), deep LODs;
//!   * LOD selection: distance → ring;
//!   * residency: enters/exits as a simulated camera moves, empty bricks skipped, per-frame cap (carry
//!     queue), keep-old-until-revealed (not dirty until a revealing batch lands).
//!
//! The mixed-LOD GPU oracle lives in `tests/voxel_raytrace_gpu.rs`.

use bevy::math::IVec3;

use adventure::sdf_render::worldgen::biome::{
    BiomeDef, BiomeId, BiomeLibrary, StrataLayer, TerrainMatId, TerrainSurfaceMaterial,
};
use adventure::sdf_render::worldgen::coord::LayerId;
use adventure::sdf_render::worldgen::layers::erosion::ErosionParams;
use adventure::sdf_render::worldgen::layers::height::{HeightLayer, HeightParams};
use adventure::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, BRICK_WORLD_SIZE, Brick, MAX_LOD, voxel_index};
use adventure::voxel::gpu::{ResidentBrick, lod_solid_keep_k, pack_resident_set};
use adventure::voxel::palette::{BlockId, BlockRegistry};
use adventure::voxel::streaming::{
    ResidencyManager, StreamingConfig, brick_lod, camera_brick_coord, desired_residency,
};

const SEED: u64 = 0xA15E_C0DE_2026;

fn solid(n: u16) -> BlockId {
    BlockId(n)
}

fn lindex(x: i32, y: i32, z: i32, edge: i32) -> usize {
    (x + y * edge + z * edge * edge) as usize
}

fn test_layer() -> HeightLayer {
    HeightLayer::new(LayerId(0), HeightParams::default(), ErosionParams::default())
}

fn test_library() -> BiomeLibrary {
    let mat = |name: &str, c: [f32; 4]| TerrainSurfaceMaterial {
        name: name.into(),
        base_color: c,
        roughness: 0.9,
        blend: 0.0,
        texture: None,
        tiling: 4.0,
        ..Default::default()
    };
    let materials = vec![mat("surface", [0.1, 0.5, 0.1, 1.0]), mat("stone", [0.5, 0.5, 0.5, 1.0])];
    let column = |_| BiomeDef {
        name: "b".into(),
        surface: TerrainMatId(0),
        surface_rules: vec![],
        strata: vec![StrataLayer { material: TerrainMatId(0), thickness: 1000.0 }],
        bedrock: TerrainMatId(1),
    };
    let biomes = BiomeId::ALL.iter().map(column).collect();
    BiomeLibrary { materials, biomes }
}

fn registry() -> BlockRegistry {
    BlockRegistry::from_biome_library(&test_library())
}

// --- mip downsample -------------------------------------------------------------------------------

#[test]
fn downsample_majority_and_thin_feature() {
    // Lower-X half block 1, upper-X half block 2 → clean coarse split at LOD1.
    let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                voxels[voxel_index(x, y, z)] = if x < 4 { solid(1) } else { solid(2) };
            }
        }
    }
    let b = Brick::from_voxels(voxels);
    let lod1 = b.downsample(1, 1);
    assert_eq!(lod1.len(), 4 * 4 * 4);
    assert_eq!(lod1[lindex(0, 0, 0, 4)], solid(1));
    assert_eq!(lod1[lindex(3, 3, 3, 4)], solid(2));

    // Thin one-voxel feature: survives k=1 (near), erodes at majority k (far).
    let mut thin = Box::new([BlockId::AIR; BRICK_VOXELS]);
    thin[voxel_index(0, 0, 0)] = solid(5);
    let tb = Brick::from_voxels(thin);
    assert_eq!(tb.downsample(1, 1)[lindex(0, 0, 0, 4)], solid(5), "k=1 keeps a 1/8 sliver");
    assert_eq!(tb.downsample(1, 5)[lindex(0, 0, 0, 4)], BlockId::AIR, "majority k erodes it");
}

#[test]
fn downsample_thin_surface_survives_continuously() {
    let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
    for z in 0..BRICK_EDGE {
        for x in 0..BRICK_EDGE {
            voxels[voxel_index(x, 3, z)] = solid(6); // one-voxel-thick slab at y=3
        }
    }
    let b = Brick::from_voxels(voxels);
    let lod1 = b.downsample(1, 1);
    for cz in 0..4 {
        for cx in 0..4 {
            assert_eq!(lod1[lindex(cx, 1, cz, 4)], solid(6), "slab survives, no holes");
            assert_eq!(lod1[lindex(cx, 0, cz, 4)], BlockId::AIR, "it thinned, didn't spread");
        }
    }
}

#[test]
fn downsample_deep_lods_uniform_solid() {
    let b = Brick::uniform(solid(2));
    assert!(b.downsample(2, 1).iter().all(|&v| v == solid(2)));
    assert_eq!(b.downsample(3, 1), vec![solid(2)]);
}

// --- LOD selection --------------------------------------------------------------------------------

#[test]
fn lod_rings_by_distance() {
    let cfg = StreamingConfig { lod_ring_bricks: [6, 12, 18], ..Default::default() };
    let cam = IVec3::ZERO;
    assert_eq!(brick_lod(cam, cam, &cfg), 0);
    assert_eq!(brick_lod(IVec3::new(5, 0, 0), cam, &cfg), 0);
    assert_eq!(brick_lod(IVec3::new(6, 0, 0), cam, &cfg), 1);
    assert_eq!(brick_lod(IVec3::new(0, 0, 12), cam, &cfg), 2);
    assert_eq!(brick_lod(IVec3::new(18, 0, 0), cam, &cfg), 3);
    assert_eq!(brick_lod(IVec3::new(9999, 0, 0), cam, &cfg), MAX_LOD);
    assert_eq!(brick_lod(IVec3::new(6, 6, 0), cam, &cfg), 1, "Chebyshev: diagonal uses max axis");
}

#[test]
fn desired_region_cube_and_cap() {
    let cfg = StreamingConfig {
        residency_radius_bricks: 3,
        lod_ring_bricks: [2, 3, 4],
        max_resident_bricks: 100_000,
        ..Default::default()
    };
    let cam = IVec3::new(10, -5, 7);
    let d = desired_residency(cam, &cfg);
    assert_eq!(d.len(), (2 * 3 + 1usize).pow(3));
    assert_eq!(d[&cam], 0);
    assert!(!d.contains_key(&(cam + IVec3::new(4, 0, 0))));

    let capped = StreamingConfig { residency_radius_bricks: 4, max_resident_bricks: 10, ..cfg };
    let dc = desired_residency(IVec3::ZERO, &capped);
    assert_eq!(dc.len(), 10);
    assert!(dc.contains_key(&IVec3::ZERO), "the nearest (camera) brick is always kept");
}

#[test]
fn camera_brick_coord_maps_world_to_brick() {
    // World origin → brick 0; one brick over in +X (BRICK_WORLD_SIZE m) → brick (1,0,0).
    assert_eq!(camera_brick_coord([0.1, 0.1, 0.1]), IVec3::ZERO);
    assert_eq!(camera_brick_coord([BRICK_WORLD_SIZE + 0.1, 0.0, 0.0]), IVec3::new(1, 0, 0));
    assert_eq!(camera_brick_coord([-0.1, 0.0, 0.0]), IVec3::new(-1, 0, 0));
}

// --- residency ------------------------------------------------------------------------------------

fn cheby(a: IVec3, b: IVec3) -> i32 {
    (a.x - b.x).abs().max((a.y - b.y).abs()).max((a.z - b.z).abs())
}

#[test]
fn residency_enters_and_exits_as_camera_moves() {
    let layer = test_layer();
    let lib = test_library();
    let reg = registry();
    let surf = layer.sample_world(0.0, 0.0, SEED).height;
    let surf_brick_y = (surf / BRICK_WORLD_SIZE).floor() as i32;
    let cfg = StreamingConfig {
        residency_radius_bricks: 2,
        lod_ring_bricks: [1, 2, 3],
        max_resident_bricks: 10_000,
        max_bricks_per_frame: 1000,
    };

    let mut mgr = ResidencyManager::new();
    let cam0 = IVec3::new(0, surf_brick_y, 0);
    mgr.update(cam0, &cfg);
    assert!(mgr.pending() > 0, "entering a fresh region enqueues work");
    assert!(!mgr.is_dirty(), "keep-old: nothing voxelized yet → not dirty");

    let n0 = mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
    assert_eq!(n0, (2 * 2 + 1i32).pow(3) as usize, "drained the whole 5³ region");
    assert!(mgr.take_dirty(), "voxelizing real terrain reveals geometry → dirty");
    let r0 = mgr.resident_count();
    assert!(r0 > 0 && r0 <= 125, "some non-empty bricks resident, ≤ region size");

    // Move +5 bricks in X: region fully shifts. New bricks enter, far ones drop.
    let cam1 = cam0 + IVec3::new(5, 0, 0);
    let dropped = mgr.update(cam1, &cfg);
    assert!(dropped > 0, "moving away drops bricks left behind");
    mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
    for e in mgr.resident_entries() {
        assert!(cheby(e.coord, cam1) <= cfg.residency_radius_bricks, "resident bricks stay in-region");
    }
}

#[test]
fn empty_sky_bricks_skipped() {
    let layer = test_layer();
    let lib = test_library();
    let reg = registry();
    // A region far ABOVE any terrain → every brick is all-air → none become resident.
    let cfg = StreamingConfig {
        residency_radius_bricks: 1,
        lod_ring_bricks: [1, 2, 3],
        max_resident_bricks: 10_000,
        max_bricks_per_frame: 1000,
    };
    let mut mgr = ResidencyManager::new();
    let sky = IVec3::new(0, 4000, 0); // ~+6.4 km up
    mgr.update(sky, &cfg);
    mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
    assert_eq!(mgr.resident_count(), 0, "all-air sky bricks are skipped (sparsity)");
    assert!(!mgr.is_dirty(), "an all-empty batch does not reveal geometry → not dirty (keep-old)");
}

#[test]
fn carry_queue_caps_per_frame_work() {
    let layer = test_layer();
    let lib = test_library();
    let reg = registry();
    let surf = layer.sample_world(0.0, 0.0, SEED).height;
    let surf_brick_y = (surf / BRICK_WORLD_SIZE).floor() as i32;
    let cfg = StreamingConfig {
        residency_radius_bricks: 4,
        lod_ring_bricks: [2, 3, 4],
        max_resident_bricks: 10_000,
        max_bricks_per_frame: 50,
    };
    let mut mgr = ResidencyManager::new();
    mgr.update(IVec3::new(0, surf_brick_y, 0), &cfg);
    let total = (2 * 4 + 1i32).pow(3) as usize; // 729
    assert_eq!(mgr.pending(), total);

    let mut drains = 0;
    while mgr.pending() > 0 {
        let n = mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
        assert!(n <= 50, "never exceeds the per-frame cap");
        drains += 1;
        assert!(drains <= 20);
    }
    assert_eq!(drains, total.div_ceil(50), "carries the rest across frames");
}

#[test]
fn lod_change_requeues_and_repacks_coarser() {
    let layer = test_layer();
    let lib = test_library();
    let reg = registry();
    let surf = layer.sample_world(0.0, 0.0, SEED).height;
    let surf_brick_y = (surf / BRICK_WORLD_SIZE).floor() as i32;
    let cfg = StreamingConfig {
        residency_radius_bricks: 6,
        lod_ring_bricks: [2, 4, 6],
        max_resident_bricks: 10_000,
        max_bricks_per_frame: 10_000,
    };
    let mut mgr = ResidencyManager::new();
    let cam0 = IVec3::new(0, surf_brick_y, 0);
    mgr.update(cam0, &cfg);
    mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
    mgr.take_dirty();

    let probe = IVec3::new(0, surf_brick_y, 1); // distance 1 → LOD0
    let resident_probe = mgr.resident_entries().into_iter().find(|e| e.coord == probe);
    if let Some(e0) = resident_probe {
        assert_eq!(e0.lod, 0);
        let cam1 = probe + IVec3::new(3, 0, 0); // probe now at distance 3 → ring threshold 2 → LOD1
        mgr.update(cam1, &cfg);
        assert!(mgr.pending() > 0, "the LOD-changed brick is re-queued");
        mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
        let lod_after = mgr.resident_entries().into_iter().find(|e| e.coord == probe).map(|e| e.lod);
        assert_eq!(lod_after, Some(1), "stored at the coarser ring's LOD now");
    }
}

// --- packing SSOT ---------------------------------------------------------------------------------

#[test]
fn pack_resident_set_encodes_lod_and_skips_eroded() {
    let reg = registry();
    let solidb = Brick::uniform(solid(1));
    let mut thinv = Box::new([BlockId::AIR; BRICK_VOXELS]);
    thinv[voxel_index(0, 0, 0)] = solid(1);
    let thin = Brick::from_voxels(thinv);

    let entries = vec![
        ResidentBrick { coord: IVec3::new(0, 0, 0), brick: &solidb, lod: 1 },
        ResidentBrick { coord: IVec3::new(1, 0, 0), brick: &thin, lod: 3 }, // erodes to air at LOD3
    ];
    let patch = pack_resident_set(&entries, &reg);
    assert_eq!(patch.brick_count(), 1, "the eroded brick is dropped (sparsity at coarse LOD)");
    assert_eq!(patch.metas[0].lod, 1);
    // A LOD1 brick stores a HALOED grid: core 4³ + a 1-cell border each side → 6³ (the brick-seam fix).
    assert_eq!(patch.voxels.len(), adventure::voxel::gpu::halo_cells(1), "a LOD1 brick stores a haloed 6³ grid");
    assert_eq!(adventure::voxel::gpu::halo_cells(1), 6 * 6 * 6);
    assert_eq!(lod_solid_keep_k(1), 1);
    assert_eq!(lod_solid_keep_k(3), (1u32 << 9).div_ceil(2));
}
