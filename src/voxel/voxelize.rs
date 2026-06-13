//! Voxelize the procedural worldgen surface into [`Brick`]s.
//!
//! For each voxel in a brick we sample the REAL worldgen surface — the same [`HeightLayer::sample_world`]
//! the renderer's terrain uses — to decide solid vs air, and the same climate→biome→strata material chain
//! ([`temperature`]/[`humidity`]/[`classify`]/[`strata_material`]) to pick the block. This is a pure,
//! deterministic function of `(brick_coord, seed, layer, library, registry)`: identical inputs always
//! yield an identical brick (the determinism the tests pin).

use bevy::math::IVec3;

use crate::sdf_render::worldgen::artifact::HeightNode;
use crate::sdf_render::worldgen::biome::{
    BiomeLibrary, TerrainMatId, classify, humidity, resolve_surface, strata_material, surface_biome,
    temperature,
};
use crate::sdf_render::worldgen::layers::height::HeightLayer;

use super::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, VOXEL_SIZE, voxel_index};
use super::palette::{BlockId, BlockRegistry};

/// The world-space metre position of the CENTRE of the voxel at world voxel coordinate `world_voxel`.
/// Voxel `v` spans `[v·VOXEL_SIZE, (v+1)·VOXEL_SIZE)`; we sample the surface at its centre so the
/// solid/air boundary lands cleanly at the half-voxel rather than biasing to a face.
#[inline]
pub fn voxel_center_world(world_voxel: IVec3) -> [f64; 3] {
    [
        (world_voxel.x as f64 + 0.5) * VOXEL_SIZE as f64,
        (world_voxel.y as f64 + 0.5) * VOXEL_SIZE as f64,
        (world_voxel.z as f64 + 0.5) * VOXEL_SIZE as f64,
    ]
}

/// The block at a single WORLD voxel coordinate, sampling the worldgen surface. Returns [`BlockId::AIR`]
/// for any voxel ABOVE the surface (`depth < 0`); otherwise it resolves the climate biome at the column
/// and the strata material at `depth = surface_height − voxel_y`, mapped through the [`BlockRegistry`] to
/// a [`BlockId`]. The SSOT for one voxel — both [`voxelize_brick`] and the per-column tests call it, so
/// they can never diverge.
#[inline]
pub fn voxel_block_at(
    world_voxel: IVec3,
    layer: &HeightLayer,
    lib: &BiomeLibrary,
    registry: &BlockRegistry,
    seed: u64,
) -> BlockId {
    let [wx, wy, wz] = voxel_center_world(world_voxel);
    let node = layer.sample_world(wx, wz, seed);
    let h = node.height as f64;
    let depth = h - wy;
    if depth < 0.0 {
        return BlockId::AIR; // above the surface → empty
    }
    let mat = surface_material_at(wx, wz, h, depth, &node, lib, seed);
    registry.block_for_material(mat)
}

/// The undug RENDER-SURFACE skin thickness (metres) the [`resolve_surface`] rules paint: voxels within this
/// depth of the surface take the surface-rule material (snow caps, cliff rock, flower / EMISSIVE lava +
/// crystal patches); deeper voxels fall to the volumetric [`strata_material`] column (dug walls). One voxel
/// edge ([`VOXEL_SIZE`]) — the exposed shell — so the glow sits on the surface, not buried under dirt.
const SURFACE_SKIN_DEPTH: f64 = VOXEL_SIZE as f64;

/// The terrain material occupying world voxel `(wx,wz)` at `depth` metres below the surface — the worldgen
/// SSOT for the voxel block. WITHIN the surface skin (`depth < SURFACE_SKIN_DEPTH`) it uses the SAME
/// [`resolve_surface`] the SDF/mesh renderer uses (so the voxel surface respects the biome `surface_rules`:
/// altitude caps, cliffs, patches, and the new EMISSIVE lava/crystal placement), taking its DOMINANT
/// material (`mat_a`); below the skin it walks the volumetric [`strata_material`] column. A base-only column
/// (no surface rule firing) yields `mat_a == def.surface`, i.e. bit-identical to the old `strata_material`
/// surface — so non-emissive terrain output is unchanged.
#[inline]
fn surface_material_at(
    wx: f64,
    wz: f64,
    h: f64,
    depth: f64,
    node: &HeightNode,
    lib: &BiomeLibrary,
    seed: u64,
) -> TerrainMatId {
    if depth < SURFACE_SKIN_DEPTH {
        // Surface skin: choose the dominant surface-rule material (matches the rendered terrain surface).
        // Surface-normal cos from the stored gradient: n = (-dh_dx, 1, -dh_dz) normalized ⇒ n_y = 1/|n|.
        let (dx, dz) = (node.dh_dx as f64, node.dh_dz as f64);
        let n_y = 1.0 / (dx * dx + dz * dz + 1.0).sqrt();
        let biome = surface_biome(wx, wz, seed);
        let blend = resolve_surface(wx, wz, h, n_y, biome, seed, lib);
        TerrainMatId(blend.mat_a)
    } else {
        // Sub-surface: the volumetric strata column (dug walls / depth ordering). Climate is height-indep.
        let biome = classify(temperature(wx, wz, seed), humidity(wx, wz, seed)).primary;
        strata_material(biome, depth, lib)
    }
}

/// Voxelize one `8³` brick at integer brick coordinate `brick_coord`. Samples [`voxel_block_at`] for each
/// of its `BRICK_VOXELS` voxels and builds a [`Brick`] (collapsing to the uniform fast path when every
/// voxel is identical — buried solids or pure air). Pure + deterministic in `(brick_coord, seed, layer,
/// lib, registry)`.
pub fn voxelize_brick(
    brick_coord: IVec3,
    layer: &HeightLayer,
    lib: &BiomeLibrary,
    registry: &BlockRegistry,
    seed: u64,
) -> Brick {
    let origin = brick_coord * BRICK_EDGE; // world voxel coordinate of the brick's (0,0,0) corner
    let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                let world_voxel = origin + IVec3::new(x, y, z);
                voxels[voxel_index(x, y, z)] = voxel_block_at(world_voxel, layer, lib, registry, seed);
            }
        }
    }
    Brick::from_voxels(voxels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_render::worldgen::biome::{
        BiomeDef, BiomeId, BiomeLibrary, StrataLayer, TerrainMatId, TerrainSurfaceMaterial,
    };
    use crate::sdf_render::worldgen::coord::LayerId;
    use crate::sdf_render::worldgen::layers::erosion::ErosionParams;
    use crate::sdf_render::worldgen::layers::height::{HeightLayer, HeightParams};

    const SEED: u64 = 0xA15E_C0DE_2026;

    /// A worldgen library whose materials are distinguishable per strata depth (surface / sub / stone /
    /// bedrock all differ) so a column's depth-ordering is observable as distinct blocks.
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
        // 0 surface, 1 sub-surface, 2 stone, 3 bedrock — distinct colours/ids.
        let materials = vec![
            mat("surface", [0.1, 0.5, 0.1, 1.0]),
            mat("sub", [0.3, 0.2, 0.1, 1.0]),
            mat("stone", [0.5, 0.5, 0.5, 1.0]),
            mat("bedrock", [0.0, 0.0, 0.0, 1.0]),
        ];
        // Every biome shares the same simple column: 1 m surface, 4 m sub, 20 m stone, then bedrock.
        let column = |_| BiomeDef {
            name: "b".into(),
            surface: TerrainMatId(0),
            surface_rules: vec![],
            strata: vec![
                StrataLayer { material: TerrainMatId(0), thickness: 1.0 },
                StrataLayer { material: TerrainMatId(1), thickness: 4.0 },
                StrataLayer { material: TerrainMatId(2), thickness: 20.0 },
            ],
            bedrock: TerrainMatId(3),
        };
        let biomes = BiomeId::ALL.iter().map(column).collect();
        BiomeLibrary { materials, biomes }
    }

    /// A plain-fBm height layer (deterministic surface) for the voxelizer tests.
    fn test_layer() -> HeightLayer {
        HeightLayer::new(LayerId(0), HeightParams::default(), ErosionParams::default())
    }

    fn registry() -> (BiomeLibrary, BlockRegistry) {
        let lib = test_library();
        let reg = BlockRegistry::from_biome_library(&lib);
        (lib, reg)
    }

    /// Same (coord, seed) → bit-identical brick. The core determinism guarantee.
    #[test]
    fn voxelize_is_deterministic() {
        let (lib, reg) = registry();
        let layer = test_layer();
        let coord = IVec3::new(2, -1, 3);
        let a = voxelize_brick(coord, &layer, &lib, &reg, SEED);
        let b = voxelize_brick(coord, &layer, &lib, &reg, SEED);
        assert_eq!(a, b, "voxelizing the same brick twice must be identical");
    }

    /// A column's solid/air boundary matches the worldgen surface height: a voxel just BELOW the sampled
    /// surface is solid, the voxel just ABOVE it is air. Checks several columns across one brick.
    #[test]
    fn column_boundary_matches_surface_height() {
        let (lib, reg) = registry();
        let layer = test_layer();
        for &(cx, cz) in &[(0, 0), (5, -3), (-7, 12)] {
            // World voxel column at brick-aligned XZ; pick a wide Y span around the surface.
            let wx = (cx as f64 + 0.5) * VOXEL_SIZE as f64;
            let wz = (cz as f64 + 0.5) * VOXEL_SIZE as f64;
            let h = layer.sample_world(wx, wz, SEED).height as f64;
            // The voxel whose centre is just below the surface, and the one just above.
            let below_y = ((h / VOXEL_SIZE as f64) - 1.0).floor() as i32;
            let above_y = ((h / VOXEL_SIZE as f64) + 1.0).ceil() as i32;
            let below = voxel_block_at(IVec3::new(cx, below_y, cz), &layer, &lib, &reg, SEED);
            let above = voxel_block_at(IVec3::new(cx, above_y, cz), &layer, &lib, &reg, SEED);
            assert!(!below.is_air(), "voxel below surface (y={below_y}, h={h:.2}) must be solid");
            assert!(above.is_air(), "voxel above surface (y={above_y}, h={h:.2}) must be air");
        }
    }

    /// Deeper voxels get deeper strata materials: walking DOWN a column, the block id sequence follows the
    /// surface → sub → stone → bedrock strata order (distinct blocks at increasing depth).
    #[test]
    fn deeper_voxels_get_deeper_strata() {
        let (lib, reg) = registry();
        let layer = test_layer();
        let (cx, cz) = (3, 4);
        let wx = (cx as f64 + 0.5) * VOXEL_SIZE as f64;
        let wz = (cz as f64 + 0.5) * VOXEL_SIZE as f64;
        let h = layer.sample_world(wx, wz, SEED).height as f64;
        let surf_voxel = (h / VOXEL_SIZE as f64).floor() as i32;

        // Sample the surface block, a block ~2 m down (sub-surface), and ~10 m down (stone).
        let depth_voxels = |m: f64| surf_voxel - (m / VOXEL_SIZE as f64).round() as i32;
        let at_surface = voxel_block_at(IVec3::new(cx, depth_voxels(0.0), cz), &layer, &lib, &reg, SEED);
        let at_sub = voxel_block_at(IVec3::new(cx, depth_voxels(2.5), cz), &layer, &lib, &reg, SEED);
        let at_stone = voxel_block_at(IVec3::new(cx, depth_voxels(10.0), cz), &layer, &lib, &reg, SEED);

        // Blocks 1/2/3 mirror TerrainMatId 0/1/2 = surface/sub/stone (registry shifts by +1 for AIR).
        assert_eq!(at_surface, reg.block_for_material(TerrainMatId(0)), "top is surface material");
        assert_eq!(at_sub, reg.block_for_material(TerrainMatId(1)), "~2.5 m down is sub-surface material");
        assert_eq!(at_stone, reg.block_for_material(TerrainMatId(2)), "~10 m down is stone");
        // And they are genuinely distinct (the depth ordering is observable).
        assert_ne!(at_surface, at_sub);
        assert_ne!(at_sub, at_stone);
    }

    /// A brick far ABOVE any terrain is entirely air (empty); a brick far BELOW is fully solid (uniform).
    #[test]
    fn sky_brick_empty_underground_uniform() {
        let (lib, reg) = registry();
        let layer = test_layer();
        // Very high brick (Y ≈ +6000 m) — guaranteed above the surface everywhere → empty.
        let sky = voxelize_brick(IVec3::new(0, 4000, 0), &layer, &lib, &reg, SEED);
        assert!(sky.is_empty(), "a brick far above terrain must be empty");
        // Very deep brick (Y ≈ -6000 m) — guaranteed below → all solid, uniform fast path.
        let deep = voxelize_brick(IVec3::new(0, -4000, 0), &layer, &lib, &reg, SEED);
        assert!(!deep.is_empty(), "a deep brick must be solid");
    }
}
