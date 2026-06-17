// **Phase G "G-c.0" — the GPU sparse brick OCCUPANCY helper** (docs/PHASE_G_GC_PLAN.md §2.2).
//
// The GPU mirror of `src/voxel/residency_gpu.rs`'s `SectorOccupancy::is_occupied`. The next stage's GPU
// enumerate/face-cull (Pass B/B0, G-c.1) reads this; G-c.0 wires it to NO pipeline — the parity test
// (`tests/voxel_gpu_residency_parity.rs`) dispatches a tiny compute over it to prove GPU == CPU == oracle.
//
// ## The structure (the dubiousconst282 sector alloc-mask)
// Bricks are grouped into 4³ = 64-brick SECTORS; each occupied sector carries a 64-bit alloc mask (one bit per
// brick: set ⇔ that `(coord, lod)` brick is occupied). Only occupied sectors are stored, in an open-addressing
// HASH keyed by `(sector_coord, lod)`. From ONE fetch this answers BOTH the per-brick `is_occupied` AND the
// coarse "is any brick in this sector occupied?" (`mask != 0`).
//
// ## The SSOT it MUST match bit-for-bit (src/voxel/residency_gpu.rs)
// - `SECTOR_EDGE = 4`, `sector_bit_index` (+X fastest, +Y, +Z at edge 4),
// - `split_sector` (Euclidean div/rem — correct for NEGATIVE brick coords),
// - `sector_hash` (the FNV-1a + avalanche mix over the 4 key words, modular u32),
// - the linear-probe walk (stop at the first free slot, `lod == EMPTY_LOD = 0xffffffffu`).
// If ANY drifts, the parity gate fails — that test is this stage's whole point.

const SECTOR_EDGE: i32 = 4;
const EMPTY_LOD: u32 = 0xffffffffu;
// MUST match `src/voxel/brickmap.rs` MAX_LOD (the clipmap's coarsest level). 8 levels: 0..=7.
const MAX_LOD: u32 = 7u;

// The per-sector record is 8 u32 words (matching `GpuSectorEntry`, 32 B). We read the entries buffer as a FLAT
// `array<u32>` and index it with an explicit 8-word stride, so there is ZERO struct-layout/stride ambiguity
// across naga back-ends (a struct-array's element stride is back-end-rounded; a flat u32 array is not). The 8
// words per slot, in order: [sector_x, sector_y, sector_z, lod, mask_lo, mask_hi, full_lo, full_hi].
const WORDS_PER_ENTRY: u32 = 8u;

// The header — MUST match `GpuResidencyHeader`. `table_size` is a power of two (probe mask = table_size - 1).
struct ResidencyHeader {
    table_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Euclidean floor-division of `a` by a POSITIVE `b` — correct for NEGATIVE `a` (matches Rust `i32::div_euclid`
// when `b > 0`, which holds here since `b == SECTOR_EDGE == 4`). Computed WITHOUT WGSL `%`/signed-`/` on negative
// operands: those are inconsistent across naga back-ends (some return a EUCLIDEAN, non-dividend-signed remainder,
// so `a/b` and `a%b` don't satisfy `a == b*q + r`, which silently corrupted `sector_of` for negative coords). We
// only ever divide NON-NEGATIVE operands here: `a >= 0` uses `a / b` directly; `a < 0` uses the positive-operand
// identity `floor(a/b) = -ceil((-a)/b) = -(((-a) + b - 1) / b)`. SSOT-equivalent to the CPU `i32::div_euclid`.
fn div_euclid_pos(a: i32, b: i32) -> i32 {
    if (a >= 0) {
        return a / b;
    }
    return -(((-a) + b - 1) / b);
}

// Euclidean remainder of `a` mod `b > 0` — always in `[0, b)` (matches Rust `i32::rem_euclid`). Derived from the
// floor-division above (`r = a - b * floor(a/b)`), so it inherits the same back-end-robust negative-coord
// handling and never calls WGSL `%` on a negative operand.
fn rem_euclid_pos(a: i32, b: i32) -> i32 {
    return a - b * div_euclid_pos(a, b);
}

// The sector coord owning brick `coord` (`coord.div_euclid(SECTOR_EDGE)`) — SSOT mirror of `split_sector`.
fn sector_of(coord: vec3<i32>) -> vec3<i32> {
    return vec3<i32>(
        div_euclid_pos(coord.x, SECTOR_EDGE),
        div_euclid_pos(coord.y, SECTOR_EDGE),
        div_euclid_pos(coord.z, SECTOR_EDGE),
    );
}

// The in-sector local coord of brick `coord` (`coord.rem_euclid(SECTOR_EDGE)`) — SSOT mirror of `split_sector`.
fn local_in_sector(coord: vec3<i32>) -> vec3<i32> {
    return vec3<i32>(
        rem_euclid_pos(coord.x, SECTOR_EDGE),
        rem_euclid_pos(coord.y, SECTOR_EDGE),
        rem_euclid_pos(coord.z, SECTOR_EDGE),
    );
}

// The local bit index `[0, 64)` of a brick within its sector — SSOT mirror of `sector_bit_index`.
fn bit_index(local: vec3<i32>) -> u32 {
    return u32(local.x + local.y * SECTOR_EDGE + local.z * SECTOR_EDGE * SECTOR_EDGE);
}

// The 32-bit sector-key hash — SSOT mirror of `sector_hash` (modular u32, FNV-1a + avalanche). `bitcast` the
// (possibly negative) coords to u32, exactly as the Rust `coord as u32` reinterprets the two's-complement bits.
fn hash_sector(sector: vec3<i32>, lod: u32) -> u32 {
    var h: u32 = 2166136261u;
    var words = array<u32, 4>(
        bitcast<u32>(sector.x),
        bitcast<u32>(sector.y),
        bitcast<u32>(sector.z),
        lod,
    );
    for (var i = 0u; i < 4u; i = i + 1u) {
        h = h ^ words[i];
        h = h * 16777619u;
        h = h ^ (h >> 15u);
        h = h * 2654435761u;
        h = h ^ (h >> 13u);
    }
    return h;
}

// The core query: is the `(coord, lod)` brick occupied? Reads `header.table_size` + the `entries` hash. A free
// slot before a match ⇒ the sector is absent ⇒ false. `entries`/`header` are provided by the consuming pipeline
// (G-c.1 binds them); this helper is binding-agnostic so the parity test + the live pass share it.
//
// Usage (the consumer declares the bindings, then includes this logic):
//   @group(G) @binding(B0) var<uniform> residency_header: ResidencyHeader;
//   @group(G) @binding(B1) var<storage, read> residency_entries: array<SectorEntry>;
// then `residency_is_occupied(&residency_header, &residency_entries, coord, lod)`.
//
// WGSL can't take pointers to storage/uniform across module boundaries cleanly in all back-ends, so the parity
// test (and G-c.1) declare the two bindings with the names below and call `is_occupied` directly.

@group(0) @binding(0) var<uniform> residency_header: ResidencyHeader;
@group(0) @binding(1) var<storage, read> residency_entries: array<u32>;

// The probed sector masks: `occ` = the 64-bit OCCUPANCY (presence) mask, `full` = the 64-bit FULL (fully-solid)
// mask, each split as `vec2<u32>(lo, hi)`. The SSOT mirror of `SectorOccupancy::sector_masks`.
struct SectorMasks {
    occ: vec2<u32>,
    full: vec2<u32>,
}

// Probe the table for `(sector, lod)`; return its `(occupancy, full)` masks, or all-zero if the sector is
// absent. The SINGLE fetch every query derives from. Reads the FLAT u32 entries buffer at an explicit 8-word
// stride (no struct-layout ambiguity).
fn sector_masks(sector: vec3<i32>, lod: u32) -> SectorMasks {
    let table_size = residency_header.table_size;
    if (table_size == 0u) {
        return SectorMasks(vec2<u32>(0u, 0u), vec2<u32>(0u, 0u));
    }
    let mask_bits = table_size - 1u;
    var slot = hash_sector(sector, lod) & mask_bits;
    // Probe at most `table_size` slots; a free slot ⇒ absent (the build keeps the table < 100% full).
    for (var i = 0u; i < table_size; i = i + 1u) {
        let base = slot * WORDS_PER_ENTRY;
        let e_lod = residency_entries[base + 3u];
        if (e_lod == EMPTY_LOD) {
            return SectorMasks(vec2<u32>(0u, 0u), vec2<u32>(0u, 0u)); // first free slot ⇒ key absent
        }
        let e_sx = bitcast<i32>(residency_entries[base + 0u]);
        let e_sy = bitcast<i32>(residency_entries[base + 1u]);
        let e_sz = bitcast<i32>(residency_entries[base + 2u]);
        if (e_lod == lod && e_sx == sector.x && e_sy == sector.y && e_sz == sector.z) {
            return SectorMasks(
                vec2<u32>(residency_entries[base + 4u], residency_entries[base + 5u]),
                vec2<u32>(residency_entries[base + 6u], residency_entries[base + 7u]),
            );
        }
        slot = (slot + 1u) & mask_bits;
    }
    return SectorMasks(vec2<u32>(0u, 0u), vec2<u32>(0u, 0u));
}

// Test bit `bit` (`[0,64)`) of a 64-bit mask split as `vec2<u32>(lo, hi)`.
fn mask_bit_set(mask: vec2<u32>, bit: u32) -> bool {
    if (bit < 32u) {
        return ((mask.x >> bit) & 1u) != 0u;
    }
    return ((mask.y >> (bit - 32u)) & 1u) != 0u;
}

// Is the `(coord, lod)` brick occupied (present)?
fn is_occupied(coord: vec3<i32>, lod: u32) -> bool {
    let m = sector_masks(sector_of(coord), lod);
    return mask_bit_set(m.occ, bit_index(local_in_sector(coord)));
}

// Is the `(coord, lod)` brick present AND FULLY SOLID? (The face-cull input for the Interior test.)
fn is_full(coord: vec3<i32>, lod: u32) -> bool {
    let m = sector_masks(sector_of(coord), lod);
    return mask_bit_set(m.full, bit_index(local_in_sector(coord)));
}

// The coarse "is ANY brick in this sector occupied?" — the §1 Pass B0 test, from the SAME `sector_masks` fetch.
fn sector_any_occupied(sector: vec3<i32>, lod: u32) -> bool {
    let m = sector_masks(sector, lod);
    return (m.occ.x != 0u) || (m.occ.y != 0u);
}

// The Pass-B **6-face occlusion cull** — the SSOT mirror of `SectorOccupancy::classify_surface` /
// `StaticVoxSource::classify == Surface`. A SURFACE brick is present, AND NOT fully occluded (NOT `is_full`
// itself, OR ≥1 of its 6 same-LOD face-neighbours is not `is_full`). Identical to re-flora `is_occluded`
// (`make_surface_sparse.comp:116`) generalized to our brick granularity's full/partial distinction.
fn classify_surface(coord: vec3<i32>, lod: u32) -> bool {
    if (!is_occupied(coord, lod)) {
        return false; // absent ⇒ Air
    }
    if (!is_full(coord, lod)) {
        return true; // present but partial ⇒ an exposed internal face ⇒ Surface
    }
    // Fully solid: Surface iff ANY of the 6 face-neighbours is not fully solid (an exposed face); else Interior.
    if (!is_full(coord + vec3<i32>(1, 0, 0), lod)) { return true; }
    if (!is_full(coord + vec3<i32>(-1, 0, 0), lod)) { return true; }
    if (!is_full(coord + vec3<i32>(0, 1, 0), lod)) { return true; }
    if (!is_full(coord + vec3<i32>(0, -1, 0), lod)) { return true; }
    if (!is_full(coord + vec3<i32>(0, 0, 1), lod)) { return true; }
    if (!is_full(coord + vec3<i32>(0, 0, -1), lod)) { return true; }
    return false; // all 6 face-neighbours full ⇒ no exposed face ⇒ Interior
}

// --- Parity-test entry point (G-c.0 ONLY; G-c.1 replaces this with the real enumerate pass) ---
// Reads a list of query keys and writes each key's `is_occupied` verdict (1/0) to an output buffer, so the
// headless parity test can compare the GPU verdict to the CPU `SectorOccupancy::is_occupied` byte-for-byte.

struct QueryKey {
    x: i32,
    y: i32,
    z: i32,
    lod: u32,
}

@group(0) @binding(2) var<storage, read> query_keys: array<QueryKey>;
@group(0) @binding(3) var<storage, read_write> query_out: array<u32>;

@compute @workgroup_size(64)
fn residency_parity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&query_keys)) {
        return;
    }
    let k = query_keys[i];
    let occ = is_occupied(vec3<i32>(k.x, k.y, k.z), k.lod);
    query_out[i] = select(0u, 1u, occ);
}

// =====================================================================================================
//  G-c.1 — GPU clipmap enumeration + 6-face surface cull (Pass B0 + Pass B)
//  The clipmap math (`level_box`/`level_hole`/`level_resident`/`snap_even_odd`/`shell_subboxes`) is the SSOT
//  ported VERBATIM from `src/voxel/streaming.rs` (~159-349). It MUST stay bit-for-bit with the CPU: the
//  enumerate-parity gate (`tests/voxel_gpu_enumerate_parity.rs`) asserts the GPU surface set EQUALS the CPU
//  `desired_clipmap` ∩ `classify == Surface` exactly, incl. negative coords + brick-boundary crossings.
//
//  Integer-op discipline (the G-c.0 hazard): naga signed `/`/`%` are inconsistent for NEGATIVE operands across
//  back-ends, and the clipmap reaches negative coords. So we NEVER use signed `/`/`%` here:
//   * the even/odd SNAP is `& ~1` (floor to even) / `| 1` (ceil to odd) — bitwise, exact on two's complement;
//   * `level_hole`'s `/2` divides values that are GUARANTEED EVEN (the finer box is `[even, odd]`-snapped, and
//     `fhi - 1` is even), so it is the arithmetic shift `>> 1u` (sign-preserving, exact for even values) —
//     bit-identical to the CPU `i32 /` there (truncation == euclid == shift when the operand is even);
//   * `camera_brick_coord_lod` (`floor(cam/span)`) is NOT computed here — the CPU passes the per-LOD brick
//     coord in the uniform, so no float floor crosses the CPU/GPU boundary.
// =====================================================================================================

// `ResidencyParams` — the per-tick clipmap uniform. `cam_brick_coord[L]` is the CPU `camera_brick_coord_lod`
// (the brick the camera falls in on grid L); `clip_half_bricks` is `StreamingConfig::clip_half_bricks`. The
// per-LOD WG-cell tiling (Pass B0/B dispatch decode) is precomputed CPU-side: `cell_lo` = the LOD's `level_box`
// lo floored to the 8-brick WG-cell grid; `cell_dims` = the count of 8³ WG-cells per axis covering the box;
// `cell_offset` = the prefix-sum start of this LOD's cells in the flat dispatch (so a flat cell index decodes
// to (lod, local cell) without a per-thread loop). `total_cells` = Σ cells (the Pass B0 dispatch size).
struct LevelParams {
    cam_brick_coord: vec3<i32>,
    _pad_a: i32,
    cell_lo: vec3<i32>,    // WG-cell grid origin in BRICK coords (multiple of 8, = level_box lo floored to 8)
    cell_offset: u32,      // prefix-sum start of this LOD's WG-cells in the flat list
    cell_dims: vec3<u32>,  // number of 8³ WG-cells per axis covering this LOD's level_box
    cell_count: u32,       // cell_dims.x * .y * .z (this LOD's WG-cell count)
}

struct ResidencyParams {
    levels: array<LevelParams, 8>, // index = lod, 0..=MAX_LOD
    clip_half_bricks: i32,
    total_cells: u32,              // Σ_lod cell_count — the Pass B0 dispatch invocation count
    _pad0: u32,
    _pad1: u32,
}

// --- clipmap math (SSOT port of streaming.rs) ---

// Floor `lo` to even (`& ~1`) and ceil `hi` to odd (`| 1`) — `snap_even_odd` (streaming.rs:159). Bitwise, so
// correct for two's-complement negatives. Returns `vec2<i32>(lo_snapped, hi_snapped)`.
fn snap_even(v: i32) -> i32 { return v & ~1; }
fn snap_odd(v: i32) -> i32 { return v | 1; }

// Level `lod`'s INCLUSIVE resident AABB on grid `lod` (`level_box`, streaming.rs:168): a cube of half-extent
// `half` around the camera's brick on that grid, snapped per axis to the 2×-coarser grid.
fn level_box_lo(lod: u32, half: i32) -> vec3<i32> {
    let c = params.levels[lod].cam_brick_coord;
    return vec3<i32>(snap_even(c.x - half), snap_even(c.y - half), snap_even(c.z - half));
}
fn level_box_hi(lod: u32, half: i32) -> vec3<i32> {
    let c = params.levels[lod].cam_brick_coord;
    return vec3<i32>(snap_odd(c.x + half), snap_odd(c.y + half), snap_odd(c.z + half));
}

// The INCLUSIVE hole AABB (on grid `lod`) that level `lod` cedes to the finer level `lod-1` (`level_hole`,
// streaming.rs:185): the finer level's box downsampled by 2. `flo` is even and `fhi` odd, so `/2` is the exact
// sign-preserving shift `>> 1u` (NOT signed `/`). `out_lo`/`out_hi` valid only when `has_hole` (lod > 0).
struct Hole { has: bool, lo: vec3<i32>, hi: vec3<i32> }
fn level_hole(lod: u32, half: i32) -> Hole {
    if (lod == 0u) {
        return Hole(false, vec3<i32>(0), vec3<i32>(0));
    }
    let flo = level_box_lo(lod - 1u, half); // even per axis
    let fhi = level_box_hi(lod - 1u, half); // odd per axis
    // flo/2 (flo even) and (fhi-1)/2 (fhi-1 even) via arithmetic shift — exact for even, sign-preserving.
    let hlo = vec3<i32>(flo.x >> 1u, flo.y >> 1u, flo.z >> 1u);
    let hhi = vec3<i32>((fhi.x - 1) >> 1u, (fhi.y - 1) >> 1u, (fhi.z - 1) >> 1u);
    return Hole(true, hlo, hhi);
}

fn in_box(c: vec3<i32>, lo: vec3<i32>, hi: vec3<i32>) -> bool {
    return c.x >= lo.x && c.x <= hi.x && c.y >= lo.y && c.y <= hi.y && c.z >= lo.z && c.z <= hi.z;
}

// True iff brick `coord` on grid `lod` is RESIDENT at that level (`level_resident`, streaming.rs:211): inside
// the level box and NOT in the hole ceded to the finer level.
fn level_resident(coord: vec3<i32>, lod: u32, half: i32) -> bool {
    if (!in_box(coord, level_box_lo(lod, half), level_box_hi(lod, half))) {
        return false;
    }
    let hole = level_hole(lod, half);
    if (hole.has && in_box(coord, hole.lo, hole.hi)) {
        return false;
    }
    return true;
}

@group(0) @binding(4) var<uniform> params: ResidencyParams;

// Pass B0 output: the SOLID, in-shell 8³ WG-cells (each `cell_index` is a flat index into the per-LOD cell
// grids, decoded by `decode_cell`). `shell_dispatch[0..3]` is the (x,1,1) indirect-dispatch for Pass B.
@group(0) @binding(5) var<storage, read_write> shell_wg_indices: array<u32>;
@group(0) @binding(6) var<storage, read_write> shell_count: atomic<u32>;
@group(0) @binding(7) var<storage, read_write> shell_dispatch: array<atomic<u32>>; // [wg_x, 1, 1]

// A WG-cell key decoded from a flat dispatch index: which LOD + which 8³-brick cell within that LOD's grid.
struct CellKey { lod: u32, cell: vec3<i32>, valid: bool }

// Decode a flat WG-cell index `idx` (`< total_cells`) to (lod, cell brick-coord). Finds the LOD whose
// [cell_offset, cell_offset+cell_count) range contains `idx`, then unpacks the local cell within that LOD's
// `cell_dims` grid (X fastest, then Y, then Z), scaled to the cell's BRICK-coord origin (`cell_lo + 8*cell`).
fn decode_cell(idx: u32) -> CellKey {
    for (var lod = 0u; lod <= MAX_LOD; lod = lod + 1u) {
        let lp = params.levels[lod];
        if (lp.cell_count != 0u && idx >= lp.cell_offset && idx < lp.cell_offset + lp.cell_count) {
            let local = idx - lp.cell_offset;
            let cx = local % lp.cell_dims.x;
            let cy = (local / lp.cell_dims.x) % lp.cell_dims.y;
            let cz = local / (lp.cell_dims.x * lp.cell_dims.y);
            let cell = lp.cell_lo + vec3<i32>(i32(cx), i32(cy), i32(cz)) * 8;
            return CellKey(lod, cell, true);
        }
    }
    return CellKey(0u, vec3<i32>(0), false);
}

// **Pass B0 — `prepare_shell_dispatch`.** One invocation per candidate 8³ WG-cell (across all LODs). A cell is
// kept iff it (a) intersects the shell `level_box \ level_hole` (not wholly in the hole / not wholly outside
// the box) AND (b) the coarse occupancy says ≥1 of its bricks is occupied (`sector_any_occupied` over the
// cell's sectors — the cell is 8³ bricks = 8 sectors of 4³). Kept cells are atomic-appended to
// `shell_wg_indices` and `atomicMax` the Pass-B indirect dispatch. Mirrors re-flora
// `prepare_sparse_surface_dispatch.comp`. **Bounds the work by the OCCUPIED surface, not the H³ cube.**
@compute @workgroup_size(64)
fn prepare_shell_dispatch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_cells) {
        return;
    }
    let key = decode_cell(idx);
    if (!key.valid) {
        return;
    }
    let half = params.clip_half_bricks;
    let lod = key.lod;
    let box_lo = level_box_lo(lod, half);
    let box_hi = level_box_hi(lod, half);
    let cell_hi = key.cell + vec3<i32>(7, 7, 7); // inclusive 8³ extent

    // (a) shell intersection: the cell must overlap the box, and must NOT be entirely inside the hole.
    let ov_lo = max(key.cell, box_lo);
    let ov_hi = min(cell_hi, box_hi);
    if (ov_lo.x > ov_hi.x || ov_lo.y > ov_hi.y || ov_lo.z > ov_hi.z) {
        return; // wholly outside the box
    }
    let hole = level_hole(lod, half);
    if (hole.has) {
        // entirely inside the hole (every brick ceded to the finer level) ⇒ skip — but only if the WHOLE
        // box-overlap region is in the hole. (A partially-holed cell still has shell bricks; keep it.)
        if (in_box(ov_lo, hole.lo, hole.hi) && in_box(ov_hi, hole.lo, hole.hi)) {
            return;
        }
    }

    // (b) coarse occupancy: ANY occupied brick in the cell? The cell spans 8³ bricks = a 2×2×2 block of 4³
    // sectors; test the 8 sectors (sector coord = brick.div_euclid(4); the cell origin is a multiple of 8, so
    // its sectors are `cell/4 + {0,1}` per axis — non-negative remainder, exact).
    let s0 = vec3<i32>(key.cell.x >> 2u, key.cell.y >> 2u, key.cell.z >> 2u); // cell/4 (cell multiple of 8 ⇒ even, shift exact)
    var any = false;
    for (var dz = 0; dz < 2; dz = dz + 1) {
        for (var dy = 0; dy < 2; dy = dy + 1) {
            for (var dx = 0; dx < 2; dx = dx + 1) {
                if (sector_any_occupied(s0 + vec3<i32>(dx, dy, dz), lod)) {
                    any = true;
                }
            }
        }
    }
    if (!any) {
        return;
    }

    let slot = atomicAdd(&shell_count, 1u);
    shell_wg_indices[slot] = idx;
    atomicMax(&shell_dispatch[0], slot + 1u); // wg_x ≥ #solid cells (one workgroup per solid cell)
}

// **Pass B — `enumerate_shells`** (`record_indirect` over Pass B0's `shell_dispatch`). One WORKGROUP per solid
// WG-cell; `workgroup_size = 512` (8³) so one invocation per brick in the cell. For each brick: apply the
// `level_resident` predicate AND the 6-face occlusion cull (`classify_surface`). Surface bricks → `atomicOr` a
// present-flag (dedup) + `atomicAdd` the active count + write the `(coord, lod)` key to `candidate_list`.
// Mirrors re-flora `make_surface_sparse.comp:181-230`.
@group(0) @binding(8) var<storage, read_write> candidate_count: atomic<u32>;
@group(0) @binding(9) var<storage, read_write> candidate_list: array<vec4<i32>>; // (x, y, z, lod)

@compute @workgroup_size(512)
fn enumerate_shells(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let wg = wid.x;
    if (wg >= atomicLoad(&shell_count)) {
        return;
    }
    let key = decode_cell(shell_wg_indices[wg]);
    if (!key.valid) {
        return;
    }
    // The brick within the 8³ cell this invocation owns (X fastest, then Y, then Z).
    let lx = i32(lidx % 8u);
    let ly = i32((lidx / 8u) % 8u);
    let lz = i32(lidx / 64u);
    let coord = key.cell + vec3<i32>(lx, ly, lz);
    let lod = key.lod;
    let half = params.clip_half_bricks;

    if (!level_resident(coord, lod, half)) {
        return;
    }
    if (!classify_surface(coord, lod)) {
        return;
    }
    let slot = atomicAdd(&candidate_count, 1u);
    candidate_list[slot] = vec4<i32>(coord.x, coord.y, coord.z, i32(lod));
}
