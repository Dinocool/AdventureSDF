// GPU brick voxelizer — the WGSL port of the CPU `voxelize_brick` chain (Stage 1b of the GPU-voxel-
// worldgen pivot, docs/GPU_VOXEL_WORLDGEN_PLAN.md). It turns a `(brick_coord, lod)` into the SAME per-voxel
// block ids `src/voxel/voxelize.rs::voxelize_brick` produces, for the HALOED brick (`halo_edge³` cells).
//
// It mirrors, function-for-function, the CPU SSOT chain a column walks:
//   - graph height eval                 → `wg_eval_graph` (codegen'd, imports worldgen::gpu)
//   - biome.rs temperature/humidity     → wv_climate / wv_temperature / wv_humidity
//   - biome.rs classify (primary)       → wv_primary_biome   (the Whittaker T×H partition)
//   - biome.rs strata_material          → wv_strata_material (per-biome depth walk, a uniform table)
//   - voxelize.rs ColumnSample::block_at→ wv_block_at        (Air / surface-skin / strata decision)
//   - palette.rs block_for_material     → wv_block_for_material (+1 offset for worldgen registries)
//
// ## Scope of the G1 parity contract (see worldgen_gpu_voxelize_parity.rs + the plan)
// The surface SKIN material is the biome's `surface` material (the CPU `resolve_surface` with EMPTY
// `surface_rules`, which `voxelize_brick`'s test library uses). The AboveY/Slope/Patch surface RULE
// stacks + the biome-border surface cross-fade are RENDER attributes deferred to when the GPU pool is
// wired (G2+); this stage proves the height→voxel→biome→strata→block-id chain + the halo, the hard
// numeric parity. Biomes DO differ in strata, so the climate classifier genuinely drives the output.
//
// ## f64 vs f32 (the only sanctioned divergence — same as worldgen_gpu.wgsl)
// The CPU evaluates in f64; WGSL has only f32. So a voxel whose surface/strata-boundary threshold the f32
// height/depth straddles within the f32 vs f64 rounding gap can flip Air↔Surface or pick the adjacent
// stratum — a bounded, surface-adjacent mismatch the parity test counts + caps (NOT a blanket tolerance).
// The INTEGER hash entropy is exact (u32 wrapping); only the float interpolation of it diverges.
//
// ## Knobs stay knobs
// The biome strata table, the surface-skin depth, the climate seeds/params, and the brick (coord, lod,
// origin, cell size) all arrive as a uniform — never WGSL consts. The CPU library is the authoring SSOT.

#define_import_path worldgen::voxelize

// =====================================================================================================
// Uniforms — the knobs + the flattened worldgen library (data-driven, mirrors biome.rs StrataTableStd).
// =====================================================================================================

const WV_BIOME_COUNT: u32 = 5u;        // biome.rs BIOME_COUNT (BiomeId::ALL.len())
const WV_STRATA_MAX: u32 = 8u;         // per-biome strata cap in the GPU table (≥ GPU_STRATA_MAX_LAYERS)
const WV_AIR_BLOCK: u32 = 0u;          // palette.rs BlockId::AIR

// One biome's flattened strata column for the depth walk — mirror of biome.rs `strata_material`:
//   `surface_mat` is depth ≤ 0 (and the surface skin); each `[bottom, mat]` band covers `[top, bottom)`
//   of depth (top = the previous band's bottom); below the last band → `bedrock_mat`. `layer_count` real
//   bands. Stored as f32 bottoms + u32 mat ids (TerrainMatId), packed 16-byte aligned for std140.
struct WvBiomeColumn {
    // layer_bottom[i] = cumulative thickness at the BOTTOM of band i (metres). Packed 8 floats = 2 vec4.
    layer_bottom: array<vec4<f32>, 2>,
    // layer_mat[i] = TerrainMatId of band i. Packed 8 u32 = 2 vec4<u32>.
    layer_mat: array<vec4<u32>, 2>,
    surface_mat: u32,
    bedrock_mat: u32,
    layer_count: u32,
    _pad: u32,
}

// The whole worldgen library + brick + climate knobs the voxelizer needs (one uniform).
struct WvParams {
    columns: array<WvBiomeColumn, 5>,  // WV_BIOME_COUNT, one per BiomeId (id order)
    // mat_to_block[i].x = BlockId for TerrainMatId(i). A vec4<u32> per id keeps std140 16-byte stride
    // (the [u32;N]-array gotcha). Length GPU_MAX_MATERIALS; index by TerrainMatId. (+1 offset for the
    // worldgen registry is already baked in here so the GPU need not assume it.)
    mat_to_block: array<vec4<u32>, 32>,
    // Brick placement (the clipmap SSOT, computed CPU-side from brick_span/lod_voxel_size).
    world_min: vec4<f32>,              // xyz = brick world-min corner; w unused
    cell_size: f32,                    // lod_voxel_size(lod) — the per-LOD coarse cell edge
    halo_edge: u32,                    // halo_edge(lod) = BRICK_EDGE + 2 = 10
    surface_skin_depth: f32,           // voxelize.rs SURFACE_SKIN_DEPTH (= VOXEL_SIZE)
    world_seed: u32,                   // the u32-collapsed world seed (graph + climate fold base)
    // Climate field knobs (biome.rs climate_fbm): the two climate streams + the normalize bound.
    temp_seed: u32,                    // climate_seed(seed, TEMPERATURE_SALT) — exact u32, CPU-folded
    humid_seed: u32,                   // climate_seed(seed, HUMIDITY_SALT)
    climate_octaves: u32,
    mat_count: u32,
    climate_base_freq: f32,            // 1 / CLIMATE_WAVELENGTH_M
    climate_lacunarity: f32,
    climate_gain: f32,
    climate_norm_bound: f32,           // Σ gain^k (the normalize divisor) — biome.rs climate_norm_bound
}

@group(0) @binding(0) var<uniform> wv: WvParams;
@group(0) @binding(1) var<storage, read_write> wv_out: array<u32>;  // halo_edge³ block ids

// =====================================================================================================
// Climate — mirror of biome.rs temperature/humidity/classify (the Whittaker partition).
// =====================================================================================================

// biome.rs classify thresholds (the partition lines). Knobs are fixed in the CPU code (not editor knobs),
// so they stay consts here too — the SSOT is biome.rs; a const-mismatch test pins them.
const WV_T_COLD: f32 = 0.33;
const WV_T_WARM: f32 = 0.66;
const WV_H_MID_WET: f32 = 0.55;
const WV_H_COLD_WET: f32 = 0.5;

// biome.rs `normalize_climate` — raw fBm sum → [0,1] via the geometric-bound affine map, clamped.
fn wv_normalize_climate(raw: f32) -> f32 {
    let unit = raw / wv.climate_norm_bound;
    let mapped = unit * 0.5 + 0.5;
    return clamp(mapped, 0.0, 1.0);
}

// biome.rs `temperature`/`humidity` — fBm height (value lane only) at the climate stream + normalize.
// Reuses the worldgen::gpu fBm (`wg_fbm_height_grad`); its `.x` value lane is bit-identical to the CPU
// `fbm_height` the climate fields use (same octave sum).
fn wv_climate(wx: f32, wz: f32, seed: u32) -> f32 {
    var p: WgFbmParams;
    p.octaves = wv.climate_octaves;
    p.base_freq = wv.climate_base_freq;
    p.lacunarity = wv.climate_lacunarity;
    p.gain = wv.climate_gain;
    p.amplitude = 1.0;             // climate_fbm amplitude
    p.seed = seed;                 // already the folded climate stream seed (temp/humid)
    let raw = wg_fbm_height_grad(wx, wz, p).x;
    return wv_normalize_climate(raw);
}

// biome.rs `primary_biome` — the total T×H partition → BiomeId (0..4). Matches the CPU branch order.
//   BiomeId: Plains=0, Forest=1, Desert=2, Tundra=3, Snowy=4.
fn wv_primary_biome(t: f32, h: f32) -> u32 {
    if (t >= WV_T_WARM) {
        return 2u; // Desert
    }
    if (t >= WV_T_COLD) {
        if (h >= WV_H_MID_WET) {
            return 1u; // Forest
        }
        return 0u; // Plains
    }
    if (h >= WV_H_COLD_WET) {
        return 4u; // Snowy
    }
    return 3u; // Tundra
}

// The classified primary biome at world (wx, wz) — surface_biome(...).primary (the sub-surface biome the
// strata column uses, AND the surface-skin biome since the test library has empty surface_rules).
fn wv_biome_at(wx: f32, wz: f32) -> u32 {
    let t = wv_climate(wx, wz, wv.temp_seed);
    let h = wv_climate(wx, wz, wv.humid_seed);
    return wv_primary_biome(clamp(t, 0.0, 1.0), clamp(h, 0.0, 1.0));
}

// =====================================================================================================
// Strata + block-id resolve — mirror of biome.rs strata_material + palette.rs block_for_material.
// =====================================================================================================

// biome.rs `strata_material` — depth (m below the surface) → TerrainMatId for `biome`. `depth ≤ 0` →
// surface; walk the bands (top→down), the first band whose BOTTOM exceeds depth wins; past the last → bedrock.
fn wv_strata_material(biome: u32, depth: f32) -> u32 {
    let c = wv.columns[biome];
    if (depth <= 0.0) {
        return c.surface_mat;
    }
    for (var i = 0u; i < c.layer_count; i = i + 1u) {
        let bottom = c.layer_bottom[i / 4u][i % 4u];
        if (depth < bottom) {
            return c.layer_mat[i / 4u][i % 4u];
        }
    }
    return c.bedrock_mat;
}

// palette.rs `block_for_material` — TerrainMatId → BlockId via the uniform map (AIR for an unknown id).
fn wv_block_for_material(mat: u32) -> u32 {
    if (mat >= wv.mat_count) {
        return WV_AIR_BLOCK;
    }
    return wv.mat_to_block[mat].x;
}

// voxelize.rs `ColumnSample::block_at` — the per-voxel decision at world-Y centre `wy` in this column
// (surface height `h`, classified `biome`). AIR above the surface; within the surface skin the biome's
// surface material; deeper the volumetric strata column. Then map the material → block id.
fn wv_block_at(h: f32, wy: f32, biome: u32) -> u32 {
    let depth = h - wy;
    if (depth < 0.0) {
        return WV_AIR_BLOCK; // above the surface → empty
    }
    var mat: u32;
    if (depth < wv.surface_skin_depth) {
        mat = wv.columns[biome].surface_mat; // resolve_surface with empty rules ⇒ the biome surface
    } else {
        mat = wv_strata_material(biome, depth);
    }
    return wv_block_for_material(mat);
}

// =====================================================================================================
// Brick voxelize entry — one invocation per HALOED cell. Mirrors voxelize.rs `voxelize_brick`, extended
// to the haloed grid: haloed cell (hx,hy,hz) ∈ [0, halo_edge) maps to CORE-relative voxel (hx-1,hy-1,hz-1),
// i.e. world voxel `brick_coord·8 + (h-1)` — so the 1-cell border is the neighbour's boundary column, AIR
// where the neighbour's column is above the surface (the natural worldgen halo — defined everywhere).
// =====================================================================================================

fn wv_cell_index(x: u32, y: u32, z: u32, hedge: u32) -> u32 {
    return x + y * hedge + z * hedge * hedge;  // mirror of gpu.rs halo_index (+X fastest)
}

@compute @workgroup_size(64)
fn voxelize_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hedge = wv.halo_edge;
    let total = hedge * hedge * hedge;
    let i = gid.x;
    if (i >= total) {
        return;
    }
    // De-linearize the flat index into haloed (hx,hy,hz) (the SAME +X-fastest order as wv_cell_index).
    let hx = i % hedge;
    let hy = (i / hedge) % hedge;
    let hz = i / (hedge * hedge);

    // Haloed cell → core-relative voxel (−1 border), then to its world cell-CENTRE (voxelize.rs rule:
    // world_min + (v + 0.5)·cell, with v = h − 1 since the core starts one cell in from the halo origin).
    let vx = f32(hx) - 1.0;
    let vy = f32(hy) - 1.0;
    let vz = f32(hz) - 1.0;
    let cell = wv.cell_size;
    let wx = wv.world_min.x + (vx + 0.5) * cell;
    let wy = wv.world_min.y + (vy + 0.5) * cell;
    let wz = wv.world_min.z + (vz + 0.5) * cell;

    // Column-constant worldgen: the graph surface height + the classified biome (height/climate are
    // height-independent — one eval per column on the CPU; the GPU recomputes per voxel, same result).
    let surf = wg_eval_graph(wx, wz, wv.world_seed); // WgField: .v = height
    let biome = wv_biome_at(wx, wz);
    wv_out[i] = wv_block_at(surf.v, wy, biome);
}
