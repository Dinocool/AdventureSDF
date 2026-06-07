#define_import_path sdf::probe

// DDGI probe addressing — WGSL mirror of `src/sdf_render/probe.rs`. A probe is anchored at the
// center of an occupied SDF brick; its identity is the absolute `(lod, brick_coord)` of that brick,
// so it is world-anchored and boil-free (history aligns across frames). A probe's storage slot is its
// chunk's finest-resident probe-block base + the brick's stable local index (`probe_base + local`) —
// COMPACT over the finest-resident set, so the buffer scales with the clipmap window. Both the trace
// (iterating finest-resident chunks) and the apply (resolving a world position via `chunk_buf`) derive
// the identical slot with no separate lookup table.
//
// Constants are hand-mirrored from Rust (WGSL can't import Rust consts); the
// `wgsl_probe_constants_match_rust` test pins them to the source of truth.

#import sdf::bindings::{voxel_size_at, cell_stride, chunk_buf, chunk_tile_buf, local_brick_index, CHUNK_BRICKS}
#import sdf::brick::{find_chunk}

// Per-probe octahedral irradiance map resolution (OCT_RES × OCT_RES texels, stored flat in the
// irradiance buffer at `slot * PROBE_OCT_TEXELS`). The apply samples it by the surface normal so GI is
// directional, not flat. No border (bilinear clamps at the tile edge) — a small seam at grazing dirs.
const PROBE_OCT_RES: u32 = 8u;
const PROBE_OCT_TEXELS: u32 = 64u; // PROBE_OCT_RES²

// Octahedral irradiance tile edge in texels, INCLUDING the 1px wrap border (interior = TILE-2).
const PROBE_IRR_TILE: u32 = 8u;
// Octahedral depth/visibility (Chebyshev moments) tile edge, including border.
const PROBE_DEPTH_TILE: u32 = 16u;
const PROBE_IRR_INTERIOR: u32 = 6u;
const PROBE_DEPTH_INTERIOR: u32 = 14u;
// Chunk-key axis bias — mirrors `chunk::KEY_BIAS` / `abs_chunk_key`'s `bias` (guarded by the
// chunk-constants test for `abs_chunk_key`; `wgsl_probe_constants_match_rust` pins this copy too).
const PROBE_KEY_BIAS: i32 = 32768;

// World-space center of a brick's probe (subdiv == 1). Mirrors `probe::probe_world_pos`.
fn probe_world_pos(brick_coord: vec3<i32>, lod: u32) -> vec3<f32> {
    let vs = voxel_size_at(lod);
    let bw = f32(cell_stride()) * vs;
    let p_min = vec3<f32>(brick_coord) * vs;
    return p_min + vec3<f32>(0.5 * bw);
}

// World-space center of sub-probe `sub` (each component in 0..subdiv) of a brick subdivided
// subdiv³. Mirrors `probe::subprobe_world_pos`. subdiv == 1 collapses to `probe_world_pos`.
fn subprobe_world_pos(brick_coord: vec3<i32>, lod: u32, sub: vec3<i32>, subdiv: u32) -> vec3<f32> {
    let vs = voxel_size_at(lod);
    let bw = f32(cell_stride()) * vs;
    let cell = bw / f32(subdiv);
    let p_min = vec3<f32>(brick_coord) * vs;
    return p_min + (vec3<f32>(sub) + vec3<f32>(0.5)) * cell;
}

// Decode a `ChunkLookup` 64-bit key into (lod, chunk_coord). Inverse of `abs_chunk_key`.
struct ChunkId { lod: u32, coord: vec3<i32> };
fn decode_chunk_key(key_hi: u32, key_lo: u32) -> ChunkId {
    var id: ChunkId;
    id.lod = key_hi >> 16u;
    id.coord = vec3<i32>(
        i32(key_hi & 0xffffu) - PROBE_KEY_BIAS,
        i32(key_lo >> 16u) - PROBE_KEY_BIAS,
        i32(key_lo & 0xffffu) - PROBE_KEY_BIAS,
    );
    return id;
}

// Stride-aligned brick coord of local slot `local` (0..63) within chunk `chunk_coord`. Inverse of
// `chunk_of`'s local packing (`idx = lz*16 + ly*4 + lx`) + brick-index → stride coord.
fn brick_coord_in_chunk(chunk_coord: vec3<i32>, local: u32) -> vec3<i32> {
    let c = CHUNK_BRICKS;
    let lx = i32(local) % c;
    let ly = (i32(local) / c) % c;
    let lz = i32(local) / (c * c);
    let brick_index = chunk_coord * c + vec3<i32>(lx, ly, lz);
    return brick_index * cell_stride();
}

// Popcount of occupancy-mask bits strictly below `local` (the dense tile-run rank). Mirrors
// `brick_in_chunk`'s offset computation.
fn occ_rank_below(occ_lo: u32, occ_hi: u32, local: u32) -> u32 {
    var below_lo: u32;
    var below_hi: u32;
    if (local < 32u) {
        below_lo = occ_lo & ((1u << local) - 1u);
        below_hi = 0u;
    } else {
        below_lo = occ_lo;
        below_hi = occ_hi & ((1u << (local - 32u)) - 1u);
    }
    return countOneBits(below_lo) + countOneBits(below_hi);
}

// Is local brick `local` resident in this chunk's occupancy mask?
fn occ_bit_set(occ_lo: u32, occ_hi: u32, local: u32) -> bool {
    if (local < 32u) {
        return ((occ_lo >> local) & 1u) != 0u;
    }
    return ((occ_hi >> (local - 32u)) & 1u) != 0u;
}

// The probe storage slot of the brick at (coord, lod), or -1 if absent. Reads the brick's COMPACT
// finest-resident probe slot straight from its tile-run record (`BrickTile.probe_slot`) — a stable
// per-brick slot (free-list, idempotent), compact over the finest-resident set, so the irradiance
// buffer scales with the clipmap window. The trace writes the same `BrickTile.probe_slot`, so apply
// and trace address the identical (world-anchored) probe.
fn probe_slot_at(coord: vec3<i32>, lod: u32) -> i32 {
    let ci = find_chunk(coord, lod);
    if (ci < 0) {
        return -1;
    }
    let chunk = chunk_buf[u32(ci)];
    // Not finest-resident here (fully covered by a finer LOD) → no probe; the LOD-loop callers
    // (apply `sample_gi`, recursive bounce `sample_probe_gi`) fall through to the finer LOD's probe.
    if (chunk.probe_base == 0xffffffffu) {
        return -1;
    }
    let li = local_brick_index(coord);
    if (!occ_bit_set(chunk.occ_lo, chunk.occ_hi, li)) {
        return -1; // empty brick slot → no probe
    }
    // Index the chunk's tile run by the brick's dense rank (popcount of mask bits below `li`) — the same
    // record the raymarch samples — and read its compact probe slot. `u32::MAX` = no probe assigned.
    let off = occ_rank_below(chunk.occ_lo, chunk.occ_hi, li);
    let tile = chunk_tile_buf[chunk.tile_run_base + off];
    if (tile.probe_slot == 0xffffffffu) {
        return -1;
    }
    return i32(tile.probe_slot);
}
