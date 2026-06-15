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

use super::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, VOXEL_SIZE, brick_span, lod_voxel_size, voxel_index};
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

/// The undug RENDER-SURFACE skin thickness (metres) the [`resolve_surface`] rules paint: voxels within this
/// depth of the surface take the surface-rule material (snow caps, cliff rock, flower / EMISSIVE lava +
/// crystal patches); deeper voxels fall to the volumetric [`strata_material`] column (dug walls). One voxel
/// edge ([`VOXEL_SIZE`]) — the exposed shell — so the glow sits on the surface, not buried under dirt.
const SURFACE_SKIN_DEPTH: f64 = VOXEL_SIZE as f64;

/// All the COLUMN-CONSTANT worldgen evaluation for one `(wx, wz)` ground column — the expensive 2D work
/// (the `sample_world` fBm+erosion+biome graph eval, the surface gradient, and the climate biome lookups).
/// Height + climate are height-INDEPENDENT, so this is computed ONCE per column and reused for every voxel in
/// it (8 voxels per column in a brick) — that is the SSOT both [`voxel_block_at`] (per-voxel) and
/// [`voxelize_brick`] (per-column, the hot path) build through, so they can never diverge. Hoisting it out of
/// the per-voxel loop removes the 8× redundant `sample_world` the per-voxel form did.
pub struct ColumnSample {
    node: HeightNode,
    /// Surface-normal cos: n = (−dh_dx, 1, −dh_dz) normalized ⇒ n_y = 1/|n| (for the slope surface-rules).
    n_y: f64,
    /// Biome SAMPLE (climate blend) for the SURFACE-skin rules (`resolve_surface`).
    surf_biome: crate::sdf_render::worldgen::biome::BiomeSample,
    /// Primary biome for the volumetric sub-surface strata column.
    sub_biome: crate::sdf_render::worldgen::biome::BiomeId,
    wx: f64,
    wz: f64,
    seed: u64,
}

impl ColumnSample {
    /// Evaluate the column at world XZ `(wx, wz)` — ONE `sample_world` + the two climate-biome lookups.
    #[inline]
    pub fn at(wx: f64, wz: f64, layer: &HeightLayer, seed: u64) -> Self {
        let node = layer.sample_world(wx, wz, seed);
        let (dx, dz) = (node.dh_dx as f64, node.dh_dz as f64);
        let n_y = 1.0 / (dx * dx + dz * dz + 1.0).sqrt();
        let surf_biome = surface_biome(wx, wz, seed);
        let sub_biome = classify(temperature(wx, wz, seed), humidity(wx, wz, seed)).primary;
        Self { node, n_y, surf_biome, sub_biome, wx, wz, seed }
    }

    /// Surface height (metres) of this column.
    #[inline]
    pub fn height(&self) -> f64 {
        self.node.height as f64
    }

    /// The block at world-Y centre `wy` in this column. AIR above the surface; within `SURFACE_SKIN_DEPTH`
    /// the dominant surface-rule material (`resolve_surface`, matching the rendered terrain surface — altitude
    /// caps, cliffs, patches, EMISSIVE lava/crystal); below the skin the volumetric `strata_material` column.
    /// A base-only column yields `mat_a == def.surface`, bit-identical to the old per-voxel `strata_material`.
    #[inline]
    pub fn block_at(&self, wy: f64, lib: &BiomeLibrary, registry: &BlockRegistry) -> BlockId {
        let h = self.height();
        let depth = h - wy;
        if depth < 0.0 {
            return BlockId::AIR; // above the surface → empty
        }
        let mat = if depth < SURFACE_SKIN_DEPTH {
            let blend = resolve_surface(self.wx, self.wz, h, self.n_y, self.surf_biome, self.seed, lib);
            TerrainMatId(blend.mat_a)
        } else {
            strata_material(self.sub_biome, depth, lib)
        };
        registry.block_for_material(mat)
    }
}

/// The block at a single WORLD voxel coordinate — the per-voxel SSOT (used by the per-column tests). Builds a
/// one-shot [`ColumnSample`] for the voxel's column and reads its block. [`voxelize_brick`] shares the SAME
/// `ColumnSample`/`block_at` path but amortizes the column eval across the column's 8 voxels.
#[inline]
pub fn voxel_block_at(
    world_voxel: IVec3,
    layer: &HeightLayer,
    lib: &BiomeLibrary,
    registry: &BlockRegistry,
    seed: u64,
) -> BlockId {
    let [wx, wy, wz] = voxel_center_world(world_voxel);
    ColumnSample::at(wx, wz, layer, seed).block_at(wy, lib, registry)
}

/// Voxelize one `8³` brick at clipmap key `(brick_coord, lod)`. The brick spans world
/// `[brick_coord · brick_span(lod), +brick_span(lod))` per axis and is sampled at its OWN (coarse) cell
/// size [`lod_voxel_size`]`(lod) = VOXEL_SIZE · 2^lod`: voxel `v`'s world centre is
/// `world_min + (v + 0.5) · lod_voxel_size(lod)`. A coarse brick therefore samples the worldgen surface at
/// coarse spacing — a TRUE in-place mip (not a downsample of a finer brick), giving more world coverage per
/// brick at the same `8³` resolution. At LOD0 (`lod_voxel_size == VOXEL_SIZE`) this is the original full-res
/// brick, bit-identical to the old per-voxel path. Pure + deterministic in `(brick_coord, lod, seed, layer,
/// lib, registry)`.
///
/// Builds a [`Brick`] (collapsing to the uniform fast path when every voxel is identical — buried solids or
/// pure air). Keeps the per-COLUMN `ColumnSample` optimization (one `sample_world` graph eval per `(x,z)`,
/// shared across the 8 voxels in the column).
pub fn voxelize_brick(
    brick_coord: IVec3,
    lod: u32,
    layer: &HeightLayer,
    lib: &BiomeLibrary,
    registry: &BlockRegistry,
    seed: u64,
) -> Brick {
    // The brick's world-min corner + the per-LOD cell size (the clipmap SSOT). At LOD0 `cell == VOXEL_SIZE`
    // and `world_min == origin · VOXEL_SIZE`, so this collapses to the original full-res sampling exactly.
    let span = brick_span(lod) as f64;
    let cell = lod_voxel_size(lod) as f64;
    let world_min = [
        brick_coord.x as f64 * span,
        brick_coord.y as f64 * span,
        brick_coord.z as f64 * span,
    ];
    let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
    // Loop COLUMNS (x,z) outermost: evaluate the column-constant worldgen ONCE per (x,z) at the LOD's coarse
    // spacing, then read all 8 voxels in the column from it (height/climate are height-independent). The SAME
    // ColumnSample / block_at SSOT `voxel_block_at` uses, so a LOD0 brick is bit-identical to the per-voxel
    // path — but with 1 `sample_world` per column (8× fewer of the expensive graph evals — the dominant cost).
    for z in 0..BRICK_EDGE {
        for x in 0..BRICK_EDGE {
            let wx = world_min[0] + (x as f64 + 0.5) * cell;
            let wz = world_min[2] + (z as f64 + 0.5) * cell;
            let col = ColumnSample::at(wx, wz, layer, seed);
            for y in 0..BRICK_EDGE {
                let wy = world_min[1] + (y as f64 + 0.5) * cell;
                voxels[voxel_index(x, y, z)] = col.block_at(wy, lib, registry);
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
        let a = voxelize_brick(coord, 0, &layer, &lib, &reg, SEED);
        let b = voxelize_brick(coord, 0, &layer, &lib, &reg, SEED);
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
        // The TOPMOST SOLID voxel: `voxel_block_at` samples the voxel CENTRE `(y+0.5)·VOXEL_SIZE`, so the
        // highest solid voxel is the largest y with `(y+0.5)·v < h` ⇒ `y = floor(h/v - 0.5)`. (At the old
        // 0.2 m a bare `floor(h/v)` happened to land solid; at 0.05 m it overshoots into the air voxel above —
        // derive the topmost-solid index from the centre-sampling rule instead.)
        let surf_voxel = (h / VOXEL_SIZE as f64 - 0.5).floor() as i32;

        // Sample the surface block, a block ~2.5 m down (sub-surface), and ~10 m down (stone).
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
        let sky = voxelize_brick(IVec3::new(0, 4000, 0), 0, &layer, &lib, &reg, SEED);
        assert!(sky.is_empty(), "a brick far above terrain must be empty");
        // Very deep brick (Y ≈ -6000 m) — guaranteed below → all solid, uniform fast path.
        let deep = voxelize_brick(IVec3::new(0, -4000, 0), 0, &layer, &lib, &reg, SEED);
        assert!(!deep.is_empty(), "a deep brick must be solid");
    }

    /// A coarse-LOD brick is a TRUE in-place mip: it samples the surface at `lod_voxel_size(lod)` spacing and
    /// spans `brick_span(lod)` world. A LOD-`L` brick at coord 0 covers exactly the world a `2^L`-wide block
    /// of LOD0 bricks would — its surface voxels match LOD0 surface voxels sampled at the SAME coarse cell
    /// centres (a mip, not a finer brick re-aggregated). We check coverage + that the coarse brick's surface
    /// column boundary agrees with a direct coarse-spacing surface sample.
    #[test]
    fn coarse_lod_brick_is_in_place_mip() {
        let (lib, reg) = registry();
        let layer = test_layer();
        // A LOD2 brick at coord (0, ?, 0): cell = 0.05·4 = 0.2 m, span = 0.4·4 = 1.6 m. Centre it on the
        // surface Y band so it straddles the surface (non-trivial brick).
        let lod = 2u32;
        let cell = lod_voxel_size(lod) as f64;
        let span = brick_span(lod) as f64;
        let surf = layer.sample_world(span * 0.5, span * 0.5, SEED).height as f64;
        let by = (surf / span).floor() as i32;
        let coord = IVec3::new(0, by, 0);
        let b = voxelize_brick(coord, lod, &layer, &lib, &reg, SEED);
        assert!(!b.is_empty(), "a surface-straddling coarse brick must have solid voxels");

        // Each local voxel `v` must read the block at its coarse cell-centre world position — the in-place
        // mip rule. Cross-check a column against a direct ColumnSample at the SAME coarse cell centre.
        let world_min = [coord.x as f64 * span, coord.y as f64 * span, coord.z as f64 * span];
        for &(x, z) in &[(0i32, 0i32), (3, 5), (7, 7)] {
            let wx = world_min[0] + (x as f64 + 0.5) * cell;
            let wz = world_min[2] + (z as f64 + 0.5) * cell;
            let col = ColumnSample::at(wx, wz, &layer, SEED);
            for y in 0..BRICK_EDGE {
                let wy = world_min[1] + (y as f64 + 0.5) * cell;
                assert_eq!(
                    b.get(x, y, z),
                    col.block_at(wy, &lib, &reg),
                    "coarse voxel must equal a direct surface sample at its coarse cell centre"
                );
            }
        }
    }
}
