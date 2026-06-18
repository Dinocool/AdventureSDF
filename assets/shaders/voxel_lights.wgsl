// **Stage 2 — GPU emissive-voxel NEE light-list build** (`docs/UNIFIED_GPU_RESIDENCY_PLAN.md` directive 4).
//
// Replaces the CPU NEE bake (`src/voxel/gpu.rs::build_lights_from_entries` / `finalize_lights` /
// `build_alias_table`): scans the RESIDENT GPU brick pool, finds every air-exposed emissive voxel, and builds the
// byte-compatible `VoxelLight` list + power-weighted Walker alias table the GI shader already consumes
// (`voxel_raytrace.wgsl::wc_sample_light_nee`) — readback-free. The GI sampling code is UNCHANGED; only the
// PRODUCER moves from CPU to GPU.
//
// Pipeline (one dispatch each, in order, after the residency front end packs the pool):
//   gather_lights   — 1 invocation per resident brick; loop its 8³ core cells, atomic-append air-exposed emitters
//                     to `cand_lights` (+ accumulate `sum_power` as fixed-point).
//   write_lights    — 1 invocation per candidate; compute `inv_pdf` from the final `sum_power`, write `lights`.
//   build_alias     — SINGLE invocation; serial Walker two-stack over `cand` weights into `alias` (the proven
//                     "serialize the order-sensitive step, parallelize nothing tiny" pattern — n ≤ cap).
//   The CPU reads back only the live `cand_count` (the light count) — one tiny u32, like the change_count mirror.
//
// Mirrors the CPU SSOT cell-by-cell: `cell_block` decode, `light_luma`, the 6-face air-exposed test, the voxel
// centre + face-area, and the `inv_pdf = sum_power/luma` (fallback `area·count`) — see the cited CPU fns.

// ---- shared constants (mirror src/voxel/brickmap.rs + gpu.rs) ----
const VOXEL_SIZE: f32 = 0.05;
const BRICK_EDGE: i32 = 8;
const MAX_LOD: u32 = 7u;
const META_FLAG_UNIFORM: u32 = 1u;
const AIR: u32 = 0u;

// ---- pool format (EXACT mirror of voxel_raytrace.wgsl / GpuBrickMeta) ----
struct BrickMeta {
    voxel_origin: vec3<i32>,
    voxel_offset: u32,
    world_min: vec3<f32>,
    lod_and_bits: u32,
    palette_base: u32,
    flags: u32,
    _pad1: u32,
    _pad2: u32,
};
struct Palette { rgba: vec4<f32>, emissive: vec4<f32> };

// ---- output light + alias (EXACT mirror of GpuVoxelLight / GpuAliasEntry) ----
struct VoxelLight {
    pos: vec3<f32>,
    area: f32,
    radiance: vec3<f32>,
    inv_pdf: f32,
};
struct AliasEntry { prob: f32, alias_idx: u32 };

// One gathered candidate (pre-finalize). `weight = luma(radiance)·area` (the alias power weight).
struct CandLight {
    pos: vec3<f32>,
    area: f32,
    radiance: vec3<f32>,
    weight: f32,
};

struct LightConfig {
    brick_count: u32,   // number of resident pool slots to scan (live bricks; free slots are skipped)
    max_lights: u32,    // MAX_VOXEL_LIGHTS cap
    power_scale: f32,    // fixed-point scale for the atomic sum_power accumulator (weight*scale -> u32 add)
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> metas: array<BrickMeta>;
@group(0) @binding(1) var<storage, read> voxel_indices: array<u32>;
@group(0) @binding(2) var<storage, read> brick_palettes: array<u32>;
@group(0) @binding(3) var<storage, read> palette: array<Palette>;
@group(0) @binding(4) var<uniform> cfg: LightConfig;

@group(0) @binding(5) var<storage, read_write> cand_lights: array<CandLight>;
@group(0) @binding(6) var<storage, read_write> cand_count: atomic<u32>;
@group(0) @binding(7) var<storage, read_write> sum_power_fx: atomic<u32>; // fixed-point Σ weight
@group(0) @binding(8) var<storage, read_write> lights: array<VoxelLight>;
@group(0) @binding(9) var<storage, read_write> alias_out: array<AliasEntry>;
// alias-build scratch (storage, not workgroup — n can be up to the cap, exceeds shared-mem budget):
@group(0) @binding(10) var<storage, read_write> scaled: array<f32>;
@group(0) @binding(11) var<storage, read_write> small_stack: array<u32>;
@group(0) @binding(12) var<storage, read_write> large_stack: array<u32>;

// ---- decode helpers (EXACT mirror of voxel_raytrace.wgsl) ----
fn meta_lod(m: BrickMeta) -> u32 { return m.lod_and_bits & 0x7u; }
fn meta_index_bits(m: BrickMeta) -> u32 { return (m.lod_and_bits >> 3u) & 0x1Fu; }
fn meta_is_uniform(m: BrickMeta) -> bool { return (m.flags & META_FLAG_UNIFORM) != 0u; }
fn meta_uniform_block(m: BrickMeta) -> u32 { return m.voxel_offset & 0xFFFFu; }
fn lod_cell_size(lod: u32) -> f32 { return VOXEL_SIZE * f32(1u << min(lod, MAX_LOD)); }

// Linear cell index in a haloed grid of edge `hedge` (+X fastest), mirror of cell_index.
fn cell_index(x: i32, y: i32, z: i32, hedge: i32) -> u32 {
    return u32(x + y * hedge + z * hedge * hedge);
}

// Block id at haloed cell (x,y,z), brick `m`. Mirror of voxel_raytrace.wgsl::cell_block.
fn cell_block(m: BrickMeta, x: i32, y: i32, z: i32, hedge: i32) -> u32 {
    if (meta_is_uniform(m)) {
        return meta_uniform_block(m);
    }
    let bits = meta_index_bits(m);
    if (bits == 0u) {
        return voxel_indices[m.voxel_offset + cell_index(x, y, z, hedge)];
    }
    let bit = cell_index(x, y, z, hedge) * bits;
    let word = voxel_indices[m.voxel_offset + bit / 32u];
    let mask = select((1u << bits) - 1u, 0xFFFFFFFFu, bits == 32u);
    let local = (word >> (bit % 32u)) & mask;
    return brick_palettes[m.palette_base + local];
}

fn light_luma(c: vec3<f32>) -> f32 { return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z; }

// True iff the pool slot is a FREE/degenerate slot (no live brick): a freed slot has a zeroed meta. We treat a
// slot with no LOD bits AND zero world_min AND zero flags/offset as empty — but the robust test is: the residency
// front end leaves a freed slot's meta all-zero. A zeroed meta is `uniform == false, bits == 0, offset 0` ⇒ a RAW
// brick reading voxel_indices[0..512]; to avoid mis-scanning free slots, the caller passes `brick_count` = live
// count and we additionally skip an all-zero meta.
fn meta_is_empty(m: BrickMeta) -> bool {
    return m.lod_and_bits == 0u && m.voxel_offset == 0u && m.flags == 0u
        && m.world_min.x == 0.0 && m.world_min.y == 0.0 && m.world_min.z == 0.0;
}

// ============================================================================================================
//  Pass 1 — gather air-exposed emissive voxels (1 invocation per resident brick; loop its 8³ core cells).
// ============================================================================================================
@compute @workgroup_size(64)
fn gather_lights(@builtin(global_invocation_id) gid: vec3<u32>) {
    let brick = gid.x;
    if (brick >= cfg.brick_count) {
        return;
    }
    let m = metas[brick];
    // UNIFORM bricks expose no air faces within their own haloed grid (every cell the same block) ⇒ no lights,
    // matching the CPU gather (which skips uniform bricks). Also skip empty/free slots.
    if (meta_is_uniform(m) || meta_is_empty(m)) {
        return;
    }
    let lod = meta_lod(m);
    let cell = lod_cell_size(lod);
    let area = cell * cell;
    let hedge = BRICK_EDGE + 2; // haloed grid edge (10)
    // Core cells are haloed indices [1, BRICK_EDGE]; the halo ring [0] / [BRICK_EDGE+1] is the neighbour border.
    for (var cz = 1; cz <= BRICK_EDGE; cz = cz + 1) {
        for (var cy = 1; cy <= BRICK_EDGE; cy = cy + 1) {
            for (var cx = 1; cx <= BRICK_EDGE; cx = cx + 1) {
                let id = cell_block(m, cx, cy, cz, hedge);
                if (id == AIR) {
                    continue;
                }
                let e = palette[id].emissive.xyz;
                let luma = light_luma(e);
                if (luma <= 0.0) {
                    continue; // not an emitter
                }
                // Air-exposed iff any of the 6 face neighbours (haloed grid) is AIR.
                let exposed =
                    cell_block(m, cx + 1, cy, cz, hedge) == AIR
                    || cell_block(m, cx - 1, cy, cz, hedge) == AIR
                    || cell_block(m, cx, cy + 1, cz, hedge) == AIR
                    || cell_block(m, cx, cy - 1, cz, hedge) == AIR
                    || cell_block(m, cx, cy, cz + 1, hedge) == AIR
                    || cell_block(m, cx, cy, cz - 1, hedge) == AIR;
                if (!exposed) {
                    continue;
                }
                // World centre of this core voxel (core cell c∈[1,8] ⇒ local voxel c-1).
                let lx = f32(cx - 1);
                let ly = f32(cy - 1);
                let lz = f32(cz - 1);
                let centre = m.world_min + vec3<f32>((lx + 0.5) * cell, (ly + 0.5) * cell, (lz + 0.5) * cell);
                let weight = luma * area;
                let slot = atomicAdd(&cand_count, 1u);
                if (slot < cfg.max_lights) {
                    cand_lights[slot] = CandLight(centre, area, e, weight);
                    atomicAdd(&sum_power_fx, u32(weight * cfg.power_scale));
                }
            }
        }
    }
}

// ============================================================================================================
//  Pass 2 — write the final VoxelLight list (1 invocation per kept candidate). inv_pdf from the final sum_power.
// ============================================================================================================
@compute @workgroup_size(64)
fn write_lights(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let count = min(atomicLoad(&cand_count), cfg.max_lights);
    if (i >= count) {
        return;
    }
    let c = cand_lights[i];
    let sum_power = f32(atomicLoad(&sum_power_fx)) / cfg.power_scale;
    let luma = light_luma(c.radiance);
    let usable = sum_power > 0.0;
    var inv_pdf: f32;
    if (usable && c.weight > 0.0) {
        inv_pdf = sum_power / luma;
    } else {
        inv_pdf = c.area * f32(count);
    }
    lights[i] = VoxelLight(c.pos, c.area, c.radiance, inv_pdf);
}

// ============================================================================================================
//  Pass 3 — build Walker's power-weighted alias table (SINGLE invocation, serial). Exact port of
//  src/voxel/gpu.rs::build_alias_table over the candidate weights. n ≤ cap, so a serial build is microseconds.
// ============================================================================================================
@compute @workgroup_size(1)
fn build_alias(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    let n = min(atomicLoad(&cand_count), cfg.max_lights);
    if (n == 0u) {
        return;
    }
    let sum_power = f32(atomicLoad(&sum_power_fx)) / cfg.power_scale;
    let uniform = !(sum_power > 0.0);
    // scaled[i] = p_i · n ; default prob 1 / self-alias.
    for (var i = 0u; i < n; i = i + 1u) {
        if (uniform) {
            scaled[i] = 1.0;
        } else {
            scaled[i] = (cand_lights[i].weight / sum_power) * f32(n);
        }
        alias_out[i] = AliasEntry(1.0, i);
    }
    var small_n = 0u;
    var large_n = 0u;
    for (var i = 0u; i < n; i = i + 1u) {
        if (scaled[i] < 1.0) {
            small_stack[small_n] = i;
            small_n = small_n + 1u;
        } else {
            large_stack[large_n] = i;
            large_n = large_n + 1u;
        }
    }
    loop {
        if (small_n == 0u || large_n == 0u) {
            break;
        }
        small_n = small_n - 1u;
        let s = small_stack[small_n];
        large_n = large_n - 1u;
        let l = large_stack[large_n];
        alias_out[s] = AliasEntry(scaled[s], l);
        scaled[l] = (scaled[l] + scaled[s]) - 1.0;
        if (scaled[l] < 1.0) {
            small_stack[small_n] = l;
            small_n = small_n + 1u;
        } else {
            large_stack[large_n] = l;
            large_n = large_n + 1u;
        }
    }
    // Leftover buckets (FP round-off) keep prob 1 / self-alias (already initialised).
    for (var k = 0u; k < large_n; k = k + 1u) {
        let i = large_stack[k];
        alias_out[i] = AliasEntry(1.0, i);
    }
    for (var k = 0u; k < small_n; k = k + 1u) {
        let i = small_stack[k];
        alias_out[i] = AliasEntry(1.0, i);
    }
}
