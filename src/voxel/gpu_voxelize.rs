//! GPU brick voxelizer — the host side of `assets/shaders/worldgen_voxelize.wgsl` (Stage 1b of the
//! GPU-voxel-worldgen pivot, docs/GPU_VOXEL_WORLDGEN_PLAN.md). It assembles the compute shader (the
//! `worldgen::gpu` height library + the `NodeKind → WGSL` codegen'd `wg_eval_graph` + the voxelize
//! library/entry) and flattens the worldgen [`BiomeLibrary`] + [`BlockRegistry`] + brick `(coord, lod)`
//! into the [`WvParams`] uniform the shader reads.
//!
//! ## SSOT discipline
//! NOTHING here re-implements worldgen logic: the height surface is the SAME node-graph the CPU
//! `eval_into` walks (one codegen, [`graph_to_wgsl`]); the climate seeds + knobs come from the biome.rs
//! SSOT ([`climate_gpu_params`]); the strata table is the biome columns flattened verbatim; the
//! material→block map is [`BlockRegistry::block_for_material`]. The shader mirrors the CPU
//! `voxelize_brick` chain op-for-op (see the WGSL header). The result is the GPU equivalent of
//! [`super::voxelize::voxelize_brick`], for the HALOED brick, proved bit-for-bit (within a pinned
//! surface-band tolerance) by `tests/worldgen_gpu_voxelize_parity.rs`.
//!
//! The LIVE render path does NOT use this yet — G1 is correctness-only (the GPU pool / enumeration /
//! BLAS are G2+). This module is the de-risk: prove the GPU voxelizes a brick like the CPU.

use bevy::math::IVec3;
use bevy::render::render_resource::ShaderType;

use crate::sdf_render::worldgen::biome::{
    BiomeId, BiomeLibrary, ClimateGpuParams, GPU_MAX_MATERIALS, TerrainMatId, climate_gpu_params,
};
use crate::sdf_render::worldgen::graph::{Graph, graph_to_wgsl};

use super::brickmap::{brick_span, lod_voxel_size};
use super::gpu::halo_edge;
use super::palette::BlockRegistry;
use super::voxelize::SURFACE_SKIN_DEPTH;

/// Per-biome strata cap in the GPU table. Matches `WV_STRATA_MAX` in `worldgen_voxelize.wgsl`; ≥ the CPU
/// `GPU_STRATA_MAX_LAYERS` (6) so the demo biomes' columns fit. Packed as 2 vec4 lanes (8 slots).
pub const WV_STRATA_MAX: usize = 8;

/// One biome's flattened strata column — the GPU mirror of [`BiomeId`]'s [`BiomeDef`] strata walk
/// ([`crate::sdf_render::worldgen::biome::strata_material`]). `layer_bottom[i]` is the cumulative BOTTOM
/// depth (metres) of band `i`; `layer_mat[i]` its [`TerrainMatId`]; `surface_mat` is depth ≤ 0 (and the
/// surface skin); below the last band → `bedrock_mat`. Packed as `vec4` lanes for std140 (the `[scalar;N]`
/// array-stride gotcha — keep every field a `Vec*`/`UVec*`).
#[derive(ShaderType, Clone, Copy)]
pub struct WvBiomeColumn {
    /// 8 cumulative bottom depths (metres), packed `lane0.xyzw + lane1.xyzw`.
    pub layer_bottom: [bevy::math::Vec4; 2],
    /// 8 `TerrainMatId`s, packed `lane0.xyzw + lane1.xyzw`.
    pub layer_mat: [bevy::math::UVec4; 2],
    pub surface_mat: u32,
    pub bedrock_mat: u32,
    pub layer_count: u32,
    pub _pad: u32,
}

impl Default for WvBiomeColumn {
    fn default() -> Self {
        Self {
            layer_bottom: [bevy::math::Vec4::ZERO; 2],
            layer_mat: [bevy::math::UVec4::ZERO; 2],
            surface_mat: 0,
            bedrock_mat: 0,
            layer_count: 0,
            _pad: 0,
        }
    }
}

/// The full uniform the GPU voxelizer reads — the flattened worldgen library + the brick placement +
/// climate knobs. The GPU mirror of everything `voxelize_brick`'s `ColumnSample::block_at` chain needs
/// that is NOT the height graph. Field order/types match `WvParams` in `worldgen_voxelize.wgsl` exactly.
#[derive(ShaderType, Clone)]
pub struct WvParams {
    /// One column per [`BiomeId`] (id order). Length = `BiomeId::ALL.len()` (= 5).
    pub columns: [WvBiomeColumn; BiomeId::ALL.len()],
    /// `mat_to_block[i].x` = [`crate::voxel::palette::BlockId`] for `TerrainMatId(i)` (the +1-offset
    /// worldgen registry map baked in). A `UVec4` per id keeps the std140 16-byte stride.
    pub mat_to_block: [bevy::math::UVec4; GPU_MAX_MATERIALS],
    /// `xyz` = the brick's world-min corner (metres); `w` unused.
    pub world_min: bevy::math::Vec4,
    /// `lod_voxel_size(lod)` — the per-LOD coarse cell edge (metres).
    pub cell_size: f32,
    /// `halo_edge(lod)` = `BRICK_EDGE + 2`.
    pub halo_edge: u32,
    /// `SURFACE_SKIN_DEPTH` (metres).
    pub surface_skin_depth: f32,
    /// The u32-collapsed world seed (the graph + climate fold base).
    pub world_seed: u32,
    /// Climate temperature stream seed (`climate_gpu_params().temp_seed`).
    pub temp_seed: u32,
    /// Climate humidity stream seed.
    pub humid_seed: u32,
    pub climate_octaves: u32,
    /// Number of real materials in `mat_to_block` (the GPU clamps `mat >= mat_count` → AIR).
    pub mat_count: u32,
    pub climate_base_freq: f32,
    pub climate_lacunarity: f32,
    pub climate_gain: f32,
    /// `climate_norm_bound()` — the `[0,1]` normalize divisor.
    pub climate_norm_bound: f32,
}

/// Collapse a `u64` world seed to the `u32` the height graph's fBm folds with — the SAME collapse the CPU
/// `HeightLayer::fbm_params` / `FbmAxis::params` does (`lo ^ hi`). The `wg_eval_graph` then re-folds it
/// with each axis salt, matching the CPU bit-for-bit.
#[inline]
pub fn world_seed_u32(world_seed: u64) -> u32 {
    (world_seed as u32) ^ ((world_seed >> 32) as u32)
}

impl WvParams {
    /// Build the uniform for brick `(brick_coord, lod)` from the worldgen library + registry + world seed.
    /// The brick placement mirrors [`super::voxelize::voxelize_brick`] (`world_min = coord · brick_span`,
    /// `cell = lod_voxel_size`). The strata columns + material map flatten the library through the SAME
    /// query API the CPU voxelizer uses, so the GPU can't diverge. Robust to an empty / under-populated
    /// library (missing biome columns stay zeroed → AIR).
    pub fn build(
        brick_coord: IVec3,
        lod: u32,
        lib: &BiomeLibrary,
        registry: &BlockRegistry,
        world_seed: u64,
    ) -> Self {
        let span = brick_span(lod);
        let cell = lod_voxel_size(lod);
        let world_min = bevy::math::Vec4::new(
            brick_coord.x as f32 * span,
            brick_coord.y as f32 * span,
            brick_coord.z as f32 * span,
            0.0,
        );

        // Strata columns: one per biome, in id order, flattened exactly like the CPU depth walk reads them.
        let mut columns = [WvBiomeColumn::default(); BiomeId::ALL.len()];
        if lib.biomes.len() == BiomeId::ALL.len() {
            for (slot, &id) in BiomeId::ALL.iter().enumerate() {
                let def = lib.biome(id);
                let mut col = WvBiomeColumn {
                    surface_mat: def.surface.0 as u32,
                    bedrock_mat: def.bedrock.0 as u32,
                    ..Default::default()
                };
                let mut cum = 0.0_f32;
                let mut n = 0usize;
                for layer in &def.strata {
                    if n >= WV_STRATA_MAX {
                        break;
                    }
                    cum += layer.thickness;
                    col.layer_bottom[n / 4][n % 4] = cum;
                    col.layer_mat[n / 4][n % 4] = layer.material.0 as u32;
                    n += 1;
                }
                col.layer_count = n as u32;
                columns[slot] = col;
            }
        }

        // Material → block id map (the +1-offset worldgen registry bridge, via the SSOT query).
        let mat_count = lib.materials.len().min(GPU_MAX_MATERIALS);
        let mut mat_to_block = [bevy::math::UVec4::ZERO; GPU_MAX_MATERIALS];
        for (i, slot) in mat_to_block.iter_mut().enumerate().take(mat_count) {
            slot.x = registry.block_for_material(TerrainMatId(i as u16)).0 as u32;
        }

        let ClimateGpuParams {
            temp_seed,
            humid_seed,
            octaves,
            base_freq,
            lacunarity,
            gain,
            norm_bound,
        } = climate_gpu_params(world_seed);

        Self {
            columns,
            mat_to_block,
            world_min,
            cell_size: cell,
            halo_edge: halo_edge(lod) as u32,
            surface_skin_depth: SURFACE_SKIN_DEPTH as f32,
            world_seed: world_seed_u32(world_seed),
            temp_seed,
            humid_seed,
            climate_octaves: octaves,
            mat_count: mat_count as u32,
            climate_base_freq: base_freq as f32,
            climate_lacunarity: lacunarity as f32,
            climate_gain: gain as f32,
            climate_norm_bound: norm_bound as f32,
        }
    }
}

/// Read the `worldgen_gpu.wgsl` + `worldgen_voxelize.wgsl` library sources with their
/// `#define_import_path` lines stripped, so they concatenate directly into a self-contained module for
/// wgpu's (non-naga_oil) front-end. Mirrors the height-parity rig's `worldgen_lib_source`.
fn lib_source(path: &str) -> String {
    let lib = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    lib.lines()
        .filter(|l| !l.trim_start().starts_with("#define_import_path"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The number of cells in a haloed brick at `lod` (`halo_edge³`) — the dispatch size + the output buffer
/// length the entry writes. SSOT shared by the host dispatch and the readback.
#[inline]
pub fn halo_cell_count(lod: u32) -> usize {
    let h = halo_edge(lod) as usize;
    h * h * h
}

/// Assemble the full self-contained voxelize compute shader for `graph`: the height library
/// (`worldgen_gpu.wgsl`) + the codegen'd `wg_eval_graph` + the voxelize library/entry
/// (`worldgen_voxelize.wgsl`). The shader's `voxelize_main` entry writes `halo_edge³` block ids into the
/// `wv_out` storage buffer for the brick described by the bound [`WvParams`].
///
/// `assets_dir` is the path to the `assets/shaders` directory (so tests can pass a workspace-relative
/// path); the engine wiring (G2+) will load these through the asset server instead.
pub fn voxelize_shader_src(graph: &Graph, assets_dir: &str) -> String {
    let height_lib = lib_source(&format!("{assets_dir}/worldgen_gpu.wgsl"));
    let voxelize_lib = lib_source(&format!("{assets_dir}/worldgen_voxelize.wgsl"));
    let generated = graph_to_wgsl(graph);
    format!("{height_lib}\n\n{generated}\n\n{voxelize_lib}\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::brickmap::BRICK_EDGE;
    use crate::sdf_render::worldgen::biome::{BiomeDef, StrataLayer, TerrainSurfaceMaterial};
    use crate::sdf_render::worldgen::graph::preset::mountains_plains_graph;

    fn lib_with_distinct_biomes() -> BiomeLibrary {
        let mat = |name: &str, c: [f32; 4]| TerrainSurfaceMaterial {
            name: name.into(),
            base_color: c,
            roughness: 0.9,
            ..Default::default()
        };
        let materials = (0..6).map(|i| mat(&format!("m{i}"), [i as f32 / 6.0; 4])).collect();
        let col = |surface: u16, sub: u16| BiomeDef {
            name: "b".into(),
            surface: TerrainMatId(surface),
            surface_rules: vec![],
            strata: vec![
                StrataLayer { material: TerrainMatId(surface), thickness: 1.0 },
                StrataLayer { material: TerrainMatId(sub), thickness: 4.0 },
            ],
            bedrock: TerrainMatId(5),
        };
        // Distinct surface/sub per biome so the climate classifier visibly changes the block.
        let biomes = vec![col(0, 1), col(1, 2), col(2, 3), col(3, 4), col(0, 2)];
        BiomeLibrary { materials, biomes }
    }

    /// The flattened strata columns reproduce the CPU `strata_material` depth walk (cumulative bottoms +
    /// the +1 material→block map), so the GPU table is a faithful mirror of the library.
    #[test]
    fn wvparams_flattens_library_columns() {
        let lib = lib_with_distinct_biomes();
        let reg = BlockRegistry::from_biome_library(&lib);
        let p = WvParams::build(IVec3::new(1, 2, 3), 0, &lib, &reg, 0xABCD);
        // Biome 0 (Plains): surface m0, strata [m0@1, m1@5), bedrock m5.
        let c0 = &p.columns[0];
        assert_eq!(c0.surface_mat, 0);
        assert_eq!(c0.layer_count, 2);
        assert_eq!(c0.layer_bottom[0][0], 1.0);
        assert_eq!(c0.layer_bottom[0][1], 5.0);
        assert_eq!(c0.layer_mat[0][0], 0);
        assert_eq!(c0.layer_mat[0][1], 1);
        assert_eq!(c0.bedrock_mat, 5);
        // mat_to_block is the +1 worldgen bridge: TerrainMatId(i) → BlockId(i+1).
        assert_eq!(p.mat_count, 6);
        for i in 0..6u32 {
            assert_eq!(p.mat_to_block[i as usize].x, i + 1);
        }
        // Brick placement mirrors voxelize_brick (LOD0: world_min = coord · brick_span(0)).
        assert_eq!(p.world_min.x, 1.0 * brick_span(0));
        assert_eq!(p.halo_edge, (BRICK_EDGE + 2) as u32);
    }

    /// The WGSL `worldgen_voxelize.wgsl` classify thresholds + biome count MUST equal the biome.rs SSOT —
    /// the partition lines are baked as consts in the shader (not editor knobs), so a drift would silently
    /// mis-classify biomes on the GPU. Pin them here (mirrors the `wgsl_*_constants_match_rust` discipline).
    #[test]
    fn wgsl_classify_constants_match_rust() {
        use crate::sdf_render::worldgen::biome::{H_COLD_WET, H_MID_WET, T_COLD, T_WARM};
        let wgsl = std::fs::read_to_string("assets/shaders/worldgen_voxelize.wgsl")
            .expect("read worldgen_voxelize.wgsl");
        let has = |needle: &str| assert!(wgsl.contains(needle), "worldgen_voxelize.wgsl must contain `{needle}`");
        has(&format!("const WV_T_COLD: f32 = {T_COLD};"));
        has(&format!("const WV_T_WARM: f32 = {T_WARM};"));
        has(&format!("const WV_H_MID_WET: f32 = {H_MID_WET};"));
        has(&format!("const WV_H_COLD_WET: f32 = {H_COLD_WET};"));
        has(&format!("const WV_BIOME_COUNT: u32 = {}u;", BiomeId::ALL.len()));
        has(&format!("const WV_STRATA_MAX: u32 = {WV_STRATA_MAX}u;"));
    }

    /// The assembled shader contains the height library, the codegen'd graph fn, AND the voxelize entry —
    /// so a single module dispatch covers the whole chain.
    #[test]
    fn shader_src_assembles_all_three_parts() {
        let g = mountains_plains_graph(700.0);
        let src = voxelize_shader_src(&g, "assets/shaders");
        assert!(src.contains("fn wg_eval_graph("), "codegen'd graph fn present");
        assert!(src.contains("fn wg_fbm_height_grad("), "height library present");
        assert!(src.contains("fn voxelize_main("), "voxelize entry present");
        assert!(src.contains("fn wv_block_at("), "voxelize library present");
        // No leftover import-path lines (stripped for the self-contained module).
        assert!(!src.contains("#define_import_path"), "import-path lines stripped");
    }
}
