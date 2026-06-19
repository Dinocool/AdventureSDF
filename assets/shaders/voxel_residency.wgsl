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
// The persistent `slot_table` DELETE marker — a dropped key's hole. DISTINCT from `EMPTY_LOD`: `slot_lookup` stops
// only at `EMPTY_LOD` and probes THROUGH a tombstone, so deleting a key never breaks another key's linear-probe
// chain. Using `EMPTY_LOD` for a drop (the original bug) punched a hole that made `slot_lookup` report a still-
// resident key as ABSENT → it re-entered into a 2nd slot, orphaning the 1st (stale meta+AABB = a stuck black cube).
// Mirror of `residency_gpu.rs::TOMBSTONE_LOD` (= EMPTY_LOD - 1). Only the persistent slot_table needs it; the
// per-frame present_flag/dirty_flag hashes are cleared every frame so they never accumulate tombstones.
const TOMBSTONE_LOD: u32 = 0xfffffffeu;
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
    hist_scale: f32,              // ENTER-CAP: candidate distance → histogram bucket = floor(dist * hist_scale)
    _pad1: u32,
    cam_world: vec3<f32>,         // ENTER-CAP: the camera world position (for the nearest-priority distance rank)
    _pad2: u32,
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
@compute @workgroup_size(256)
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
    // 1D dispatch dim = #solid cells (one workgroup per cell). VALID as-is when #cells <= 65535;
    // `finalize_shell_dispatch_2d` (live path) then UPGRADES this to a 2D [x, y, 1] grid so the indirect
    // enumerate is size-agnostic past the 65535 workgroup-per-dimension cap. (`enumerate_shells` recovers
    // wg = wid.x + wid.y·65535, which equals wid.x when the dim is 1D — so a harness that skips the finalize
    // pass still enumerates correctly at <= 65535 cells.)
    atomicMax(&shell_dispatch[0], slot + 1u);
}

// **Pass B — `enumerate_shells`** (`record_indirect` over Pass B0's `shell_dispatch`). One WORKGROUP per solid
// WG-cell; `workgroup_size = 512` (8³) so one invocation per brick in the cell. For each brick the pass emits
// up to TWO lists, both derived from the SAME clipmap tiling so Pass C's diff is exact-by-construction:
//   * `candidate_list` — the brick is `level_resident` AND passes the 6-face occlusion cull
//     (`classify_surface`). This is the RESIDENT-TARGET set (= CPU `desired_clipmap_surface` ∩ `classify ==
//     Surface` = the live `ResidencyManager` resident set). Mirrors re-flora `make_surface_sparse.comp:181-230`.
//   * `desired_list` — the brick is `level_resident` AND `is_occupied` (present, BEFORE the face cull). This is
//     the DESIRED-MEMBERSHIP superset = CPU `desired_clipmap_surface` (the `surface_bricks_in` candidates clipped
//     to `level_box \ level_hole`, i.e. the occupied bricks in the shell — incl. buried Interior). Pass C2's
//     `safe_to_drop` (keep-old-until-revealed) tests membership against THIS set, exactly as the CPU `update`
//     passes its `desired` map (the superset, not the surface set) to `ResidencyManager::safe_to_drop`.
// A surface brick is necessarily occupied, so `candidate_list ⊆ desired_list`. Pass C0 builds a present-flag hash
// from `desired_list`; Pass C1 enters from `candidate_list`; Pass C2 drops using the present-flag + slot_table.
@group(0) @binding(8) var<storage, read_write> candidate_count: atomic<u32>;
@group(0) @binding(9) var<storage, read_write> candidate_list: array<vec4<i32>>; // (x, y, z, lod)
@group(0) @binding(10) var<storage, read_write> desired_count: atomic<u32>;
@group(0) @binding(11) var<storage, read_write> desired_list: array<vec4<i32>>; // (x, y, z, lod) — occupied-in-shell

@compute @workgroup_size(512)
fn enumerate_shells(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    // 2D grid (size-agnostic): wg = wid.x + wid.y·65535 (mirror of pack_brick). When shell_count <= 65535 the
    // dispatch is [n,1,1] ⇒ wid.y == 0 ⇒ wg == wid.x; above it the stride is 65535 and the count guard below
    // skips the partial last row.
    let wg = wid.x + wid.y * 65535u;
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
    // DESIRED-MEMBERSHIP superset: occupied-in-shell (the CPU `desired_clipmap_surface` set). Emitted FIRST so the
    // present-flag (Pass C0) covers every brick `safe_to_drop` may test, incl. the buried Interior bricks the
    // face cull below drops from the resident-target set.
    if (!is_occupied(coord, lod)) {
        return;
    }
    let d_slot = atomicAdd(&desired_count, 1u);
    desired_list[d_slot] = vec4<i32>(coord.x, coord.y, coord.z, i32(lod));
    // RESIDENT-TARGET set: also passes the 6-face occlusion cull ⇒ a surface brick.
    if (!classify_surface(coord, lod)) {
        return;
    }
    let slot = atomicAdd(&candidate_count, 1u);
    candidate_list[slot] = vec4<i32>(coord.x, coord.y, coord.z, i32(lod));
}

// =====================================================================================================
//  G-c.2a — GPU RESIDENCY DIFF (Pass C): candidate surface set → enter/drop decisions + a GPU resident slot
//  table, the GPU port of the CPU `ResidencyManager` drop/enqueue decision + `ResidentPacker`'s slot/free-list
//  allocator (`src/voxel/incremental.rs`). SCOPE = Pass C ONLY (the pack-command build / GPU slab allocator is
//  the NEXT stage G-c.2b). The pack still comes from the CPU path; this runs only in the parity test.
//
//  Structures (all GPU-resident, persistent across frames except the per-frame lists):
//   * `slot_table` — open-addressing hash `(coord,lod) -> slot`, the GPU port of `ResidentPacker::resident`
//     (incremental.rs:731). Same hash family as `SectorOccupancy` (FNV-1a + avalanche). Free slot ⇒ `lod ==
//     EMPTY_LOD`. 5-word stride: [x, y, z, lod, slot].
//   * `free_list` — a ring of `max_resident_bricks` free slot ids with an atomic head/tail (the GPU port of
//     `SlotAllocator`, incremental.rs:580). Claim = atomic pop at `head`; release pushes to the QUARANTINE
//     (`quarantine_*`) which is drained back into `free_list` at the TOP of the NEXT frame's Pass A
//     (incremental.rs:755 — so an in-flight frame never sees a reused slot).
//   * `present_flag` — open-addressing hash of `desired_list` membership (the CPU `desired` superset), built by
//     Pass C0. `safe_to_drop` (Pass C2) tests "is this key still desired?" against it.
//   * `enter_list` / `drop_list` — per-frame atomic-append lists of the entered / dropped `(coord,lod)` keys;
//     `change_count = enter_count + drop_count` is the idempotency signal.
// =====================================================================================================

// MUST match the Rust `GpuResidencyDiffConfig`. `slot_table_size`/`present_size` are powers of two (probe mask =
// size - 1); `max_resident` is the free-list ring capacity (= `ResidentPacker` slot capacity).
struct DiffConfig {
    slot_table_size: u32,
    present_size: u32,
    max_resident: u32,
    refine_descent_cap: u32, // = REFINE_DESCENT_CAP (streaming.rs:76) = 5
}
@group(0) @binding(12) var<uniform> diff_cfg: DiffConfig;

// The slot table: 5 u32 words per slot ([x, y, z, lod, slot_id]); `lod == EMPTY_LOD` ⇒ free. Read+written by
// Pass C1 (insert on enter) and Pass C2 (clear on drop); probed by `safe_to_drop`'s residency tests. Declared
// `atomic<u32>` so Pass C1's insert can CAS the `lod` word; the coord/slot words are written with `atomicStore`
// AFTER the CAS claims the slot (single writer), and read with `atomicLoad` (no atomic SEMANTICS needed there —
// WGSL just requires atomic-typed memory be accessed via atomic builtins).
const SLOT_WORDS: u32 = 5u;
@group(0) @binding(13) var<storage, read_write> slot_table: array<atomic<u32>>;

// The free-list ring: `free_ring[i]` is a slot id; `free_head`/`free_tail` are monotonic atomic indices (masked
// by `max_resident` on access — `max_resident` is NOT required power-of-two for the ring, we wrap with rem since
// claims/releases are bounded by capacity and head ≤ tail always holds within a frame).
@group(0) @binding(14) var<storage, read_write> free_ring: array<u32>;
@group(0) @binding(15) var<storage, read_write> free_ctrl: array<atomic<u32>>; // [head, tail]

// The QUARANTINE ring: slots released THIS frame, drained back into the free-list next frame's Pass A.
@group(0) @binding(16) var<storage, read_write> quarantine_ring: array<u32>;
@group(0) @binding(17) var<storage, read_write> quarantine_ctrl: array<atomic<u32>>; // [head, tail]

// The present-flag hash (desired-set membership), 4 u32 words per slot ([x, y, z, lod]); `lod == EMPTY_LOD` ⇒
// free. Built by Pass C0 from `desired_list` (CAS the lod word to claim a slot). Probed (read via `atomicLoad`)
// by Pass C2's `safe_to_drop`. `atomic<u32>` for the Pass C0 CAS claim.
const PRESENT_WORDS: u32 = 4u;
@group(0) @binding(18) var<storage, read_write> present_flag: array<atomic<u32>>;

// Per-frame enter/drop lists + counts (atomic-append). `change_count` = enter + drop (idempotency signal).
@group(0) @binding(19) var<storage, read_write> enter_count: atomic<u32>;
@group(0) @binding(20) var<storage, read_write> enter_list: array<vec4<i32>>;
@group(0) @binding(21) var<storage, read_write> drop_count: atomic<u32>;
@group(0) @binding(22) var<storage, read_write> drop_list: array<vec4<i32>>;

// **ENTER-CAP (G-c.4 BUG-2 fix) — the NEAREST-priority admission cap (mirror of streaming.rs:783-806).** When
// the surface candidate set exceeds the free pool room, the CPU keeps the NEAREST `room` candidates by world
// distance and drops the farthest, so a static camera converges to a STABLE nearest set. The GPU mirror is a
// per-frame DISTANCE HISTOGRAM + a prefix-sum cut radius: a candidate enters iff its distance bucket is strictly
// below the cut bucket `enter_cap[0]` (the largest bucket whose cumulative candidate count ≤ `room`). Same
// candidates ⇒ same histogram ⇒ same cut ⇒ same admitted set ⇒ change→0 (converges) AND ≤ room (never overfills
// the pool). `enter_hist[b]` = candidate count in bucket b; `enter_cap = [cut_bucket, room]`.
const HIST_BUCKETS: u32 = 4096u;
@group(0) @binding(50) var<storage, read_write> enter_hist: array<atomic<u32>>;
@group(0) @binding(51) var<storage, read_write> enter_cap: array<u32>; // [cut_bucket, room]

// The world-distance of candidate `(coord,lod)`'s CENTRE to the camera (SSOT mirror of `brick_world_dist`).
fn cand_world_dist(coord: vec3<i32>, lod: u32) -> f32 {
    let span = brick_span_d(lod);
    let c = (vec3<f32>(coord) + vec3<f32>(0.5)) * span - params.cam_world;
    return sqrt(dot(c, c));
}
// The histogram bucket of a candidate distance (clamped to the last bucket).
fn dist_bucket(dist: f32) -> u32 {
    return min(u32(max(dist, 0.0) * params.hist_scale), HIST_BUCKETS - 1u);
}

// The 32-bit hash of a brick key `(coord, lod)` — the SAME FNV-1a + avalanche family as `hash_sector`, over the
// four key words (the brick coord IS the key here, no sector split). SSOT mirror of the Rust `brick_key_hash`.
fn hash_key(coord: vec3<i32>, lod: u32) -> u32 {
    var h: u32 = 2166136261u;
    var words = array<u32, 4>(bitcast<u32>(coord.x), bitcast<u32>(coord.y), bitcast<u32>(coord.z), lod);
    for (var i = 0u; i < 4u; i = i + 1u) {
        h = h ^ words[i];
        h = h * 16777619u;
        h = h ^ (h >> 15u);
        h = h * 2654435761u;
        h = h ^ (h >> 13u);
    }
    return h;
}

// --- present_flag (desired-set membership) ---

// Is `(coord, lod)` in the desired set (present-flag)? Linear-probe; a free slot before a match ⇒ absent.
fn present_contains(coord: vec3<i32>, lod: u32) -> bool {
    let size = diff_cfg.present_size;
    if (size == 0u) {
        return false;
    }
    let mask = size - 1u;
    var slot = hash_key(coord, lod) & mask;
    for (var i = 0u; i < size; i = i + 1u) {
        let base = slot * PRESENT_WORDS;
        let e_lod = atomicLoad(&present_flag[base + 3u]);
        if (e_lod == EMPTY_LOD) {
            return false;
        }
        if (e_lod == lod
            && bitcast<i32>(atomicLoad(&present_flag[base + 0u])) == coord.x
            && bitcast<i32>(atomicLoad(&present_flag[base + 1u])) == coord.y
            && bitcast<i32>(atomicLoad(&present_flag[base + 2u])) == coord.z) {
            return true;
        }
        slot = (slot + 1u) & mask;
    }
    return false;
}

// **Pass C0 — build the present-flag** from `desired_list`. One invocation per desired key: atomic linear-probe
// insert into `present_flag` (CAS the `lod` word from EMPTY_LOD to claim a slot). The desired set is DEDUPED by
// Pass B already (one emit per brick), so there are no duplicate keys to race; the CAS only guards two DIFFERENT
// keys probing the same start slot.
@compute @workgroup_size(256)
fn build_present_flag(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= atomicLoad(&desired_count)) {
        return;
    }
    let k = desired_list[i];
    let coord = vec3<i32>(k.x, k.y, k.z);
    let lod = u32(k.w);
    let size = diff_cfg.present_size;
    let mask = size - 1u;
    var slot = hash_key(coord, lod) & mask;
    for (var p = 0u; p < size; p = p + 1u) {
        let base = slot * PRESENT_WORDS;
        // CAS the lod word EMPTY_LOD -> lod. Success ⇒ we own a fresh slot: write the coord. Failure ⇒ another
        // key already owns this slot, probe on (the desired set is deduped, so no two invocations share a key).
        let prev = atomicCompareExchangeWeak(&present_flag[base + 3u], EMPTY_LOD, lod);
        if (prev.exchanged) {
            atomicStore(&present_flag[base + 0u], bitcast<u32>(coord.x));
            atomicStore(&present_flag[base + 1u], bitcast<u32>(coord.y));
            atomicStore(&present_flag[base + 2u], bitcast<u32>(coord.z));
            return;
        }
        slot = (slot + 1u) & mask;
    }
}

// --- slot_table (resident key -> slot) ---

// Probe the slot table for `(coord, lod)`; return the slot id, or SLOT_ABSENT. Read-only (used by `safe_to_drop`
// + Pass C1's absence test). Linear-probe; a free slot ⇒ absent.
const SLOT_ABSENT: u32 = 0xFFFFFFFFu;
fn slot_lookup(coord: vec3<i32>, lod: u32) -> u32 {
    let size = diff_cfg.slot_table_size;
    if (size == 0u) {
        return SLOT_ABSENT;
    }
    let mask = size - 1u;
    var slot = hash_key(coord, lod) & mask;
    for (var i = 0u; i < size; i = i + 1u) {
        let base = slot * SLOT_WORDS;
        let e_lod = atomicLoad(&slot_table[base + 3u]);
        if (e_lod == EMPTY_LOD) {
            return SLOT_ABSENT;
        }
        if (e_lod == lod
            && bitcast<i32>(atomicLoad(&slot_table[base + 0u])) == coord.x
            && bitcast<i32>(atomicLoad(&slot_table[base + 1u])) == coord.y
            && bitcast<i32>(atomicLoad(&slot_table[base + 2u])) == coord.z) {
            return atomicLoad(&slot_table[base + 4u]);
        }
        slot = (slot + 1u) & mask;
    }
    return SLOT_ABSENT;
}

fn is_resident(coord: vec3<i32>, lod: u32) -> bool {
    return slot_lookup(coord, lod) != SLOT_ABSENT;
}

// --- keep-old-until-revealed: the GPU port of `safe_to_drop` (streaming.rs:587-600) ---

// Euclidean halving for the parent walk: `coord.div_euclid(2)` per axis (streaming.rs:619). Uses the
// back-end-robust positive-operand division (the naga signed-div hazard) — SSOT of `div_euclid_pos`.
fn half_coord(c: vec3<i32>) -> vec3<i32> {
    return vec3<i32>(div_euclid_pos(c.x, 2), div_euclid_pos(c.y, 2), div_euclid_pos(c.z, 2));
}

// `region_replacement_resident` (streaming.rs:636) — is the region of `(coord, lod)` already covered by RESIDENT
// bricks of the DESIRED set at this LOD or finer? Pruned at each desired brick; descends into the 8 children
// only where this LOD is not itself desired. Bounded by `depth` (REFINE_DESCENT_CAP). NON-recursive (WGSL has no
// recursion): an explicit stack of (coord, lod, depth) frames, max `8^cap` leaves but pruned at each desired hit.
const REFINE_STACK_CAP: u32 = 512u; // generous bound for the pruned descent (8^cap is the unpruned worst case)
fn region_replacement_resident(coord0: vec3<i32>, lod0: u32, depth0: u32) -> bool {
    var stack_coord: array<vec3<i32>, REFINE_STACK_CAP>;
    var stack_lod: array<u32, REFINE_STACK_CAP>;
    var stack_depth: array<u32, REFINE_STACK_CAP>;
    var sp = 0u;
    stack_coord[0] = coord0;
    stack_lod[0] = lod0;
    stack_depth[0] = depth0;
    sp = 1u;
    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let coord = stack_coord[sp];
        let lod = stack_lod[sp];
        let depth = stack_depth[sp];
        if (present_contains(coord, lod)) {
            // Desired here (don't descend further). But only a SURFACE brick is ever ENTERED — the candidate face
            // cull drops buried INTERIOR bricks from the resident-target set, so an interior brick is desired yet
            // never resident AND never rendered. Requiring it resident (the original bug) made `safe_to_drop` FALSE
            // forever for any region with buried interior ⇒ a refined-away coarse brick was NEVER dropped → a stuck
            // coarse AABB overlapping the fine bricks (the LOD-transition black cube). A surface descendant must be
            // resident (it carries the visible surface); an interior one is "covered" by being buried — skip it.
            if (classify_surface(coord, lod) && !is_resident(coord, lod)) {
                return false;
            }
            continue;
        }
        // Not desired at this LOD: a leaf (lod 0 / depth exhausted) needs no coverage; else descend the 8 kids.
        if (lod == 0u || depth == 0u) {
            continue;
        }
        let base = coord * 2;
        // Push the 8 children. Guard the stack bound (the pruned descent stays well under REFINE_STACK_CAP at
        // the shipping refine cap; the guard makes an overflow safe rather than UB).
        for (var dz = 0; dz < 2; dz = dz + 1) {
            for (var dy = 0; dy < 2; dy = dy + 1) {
                for (var dx = 0; dx < 2; dx = dx + 1) {
                    if (sp < REFINE_STACK_CAP) {
                        stack_coord[sp] = base + vec3<i32>(dx, dy, dz);
                        stack_lod[sp] = lod - 1u;
                        stack_depth[sp] = depth - 1u;
                        sp = sp + 1u;
                    }
                }
            }
        }
    }
    return true;
}

// `safe_to_drop` (streaming.rs:616) for a resident brick `(coord, lod)` that LEFT the desired set: coarsened ⇒
// walk parents to the first DESIRED ancestor, droppable iff that ancestor is RESIDENT; else refined/left ⇒
// `region_replacement_resident` over the children.
fn safe_to_drop(coord: vec3<i32>, lod: u32) -> bool {
    // Coarsened (possibly multi-level): first desired ancestor covers the region ⇒ droppable once resident.
    var anc_coord = half_coord(coord);
    var anc_lod = lod + 1u;
    loop {
        if (anc_lod > MAX_LOD) { break; }
        if (present_contains(anc_coord, anc_lod)) {
            return is_resident(anc_coord, anc_lod);
        }
        anc_coord = half_coord(anc_coord);
        anc_lod = anc_lod + 1u;
    }
    // Otherwise refined to a finer LOD (or left the clipmap): every desired descendant must be resident.
    return region_replacement_resident(coord, lod, diff_cfg.refine_descent_cap);
}

// --- Pass A — release the previous frame's quarantine into the free-list, then clear the DIFF per-frame counts.
// Run as ONE invocation (single-threaded) so the head/tail arithmetic is race-free; the quarantine is small
// (≤ one frame's drops). Mirrors `ResidentPacker::update`'s top-of-frame quarantine drain (incremental.rs:755).
//
// SCOPE: this pass touches ONLY the DIFF-scoped state (quarantine, free-list, enter/drop counts) — the SAME set
// the G-c.2a diff gate binds (bindings ≤ 23). The pack-tail dispatch SEEDING + the enumerate/pack counts live in
// the separate `seed_frame` pass below, so a diff-ONLY driver (the diff parity gate) need not bind the pack
// buffers. The full G-c.3 frame runs `seed_frame` + `diff_release_quarantine` + `clear_per_frame_hashes` at the
// top of every frame.
@compute @workgroup_size(1)
fn diff_release_quarantine() {
    let q_head = atomicLoad(&quarantine_ctrl[0]);
    let q_tail = atomicLoad(&quarantine_ctrl[1]);
    var fi = atomicLoad(&free_ctrl[1]); // free tail
    let cap = diff_cfg.max_resident;
    for (var i = q_head; i < q_tail; i = i + 1u) {
        let slot = quarantine_ring[i % cap];
        free_ring[fi % cap] = slot;
        fi = fi + 1u;
    }
    atomicStore(&free_ctrl[1], fi);
    // Reset the quarantine ring for THIS frame's releases.
    atomicStore(&quarantine_ctrl[0], 0u);
    atomicStore(&quarantine_ctrl[1], 0u);
    // Clear the per-frame enter/drop counts.
    atomicStore(&enter_count, 0u);
    atomicStore(&drop_count, 0u);
}

// --- Pass A0 (`seed_frame`) — clear the enumerate/pack per-frame COUNTS + SEED the GPU-written indirect dispatch
//     buffers to `(0, 1, 1)` (the G-c.3 self-gating zero). ---
// Run as ONE invocation at the TOP of a full G-c.3 frame (before B0/B/C/D), so a persistent-buffer multi-frame
// drive needs NO host re-zero of any count or dispatch between frames.
//
// **G-c.3 self-gating (docs/PHASE_G_GC_PLAN.md §3.1):** every GPU-written dispatch-indirect buffer is seeded to
// `(0,1,1)` here; Passes B0/B/D then `atomicMax` each dispatch's X up to the work they actually find. So on a
// CONVERGED frame (0 enter + 0 drop ⇒ 0 dirty keys ⇒ 0 pack/aabb/classify commands) the dispatch buffers STAY
// `(0,1,1)`, and the `record_indirect` pack tail (classify_brick / pack_brick / write_aabb) launches ZERO
// workgroups at ~0 GPU cost — NO CPU branch, NO readback. Seeded on the GPU timeline (re-flora warns NEVER on the
// host, surface/mod.rs:624-629), so the whole front end is one readback-free dependency chain, idempotent for a
// static camera: same `ResidencyParams` ⇒ same candidate set ⇒ 0 enter + 0 drop ⇒ change_count == 0 ⇒ idle tail.
@compute @workgroup_size(1)
fn seed_frame() {
    atomicStore(&shell_count, 0u);
    atomicStore(&candidate_count, 0u);
    atomicStore(&desired_count, 0u);
    atomicStore(&dirty_count, 0u);
    atomicStore(&pack_count, 0u);
    atomicStore(&aabb_count, 0u);
    atomicStore(&shell_dispatch[0], 0u);
    atomicStore(&shell_dispatch[1], 1u);
    atomicStore(&shell_dispatch[2], 1u);
    atomicStore(&classify_dispatch[0], 0u);
    atomicStore(&classify_dispatch[1], 1u);
    atomicStore(&classify_dispatch[2], 1u);
    atomicStore(&pack_dispatch[0], 0u);
    atomicStore(&pack_dispatch[1], 1u);
    atomicStore(&pack_dispatch[2], 1u);
    atomicStore(&aabb_dispatch[0], 0u);
    atomicStore(&aabb_dispatch[1], 1u);
    atomicStore(&aabb_dispatch[2], 1u);
}

// --- Pass A2 — parallel clear of the per-frame HASHES (present_flag + dirty_flag) + the change_count signal. ---
// `present_flag` and `dirty_flag` are open-addressing hashes rebuilt every frame, so their `lod` words must be
// reset to EMPTY_LOD at the top of the frame. They are too large for the single-threaded Pass A, so this is a
// SEPARATE parallel pass (one invocation per MAX(present, dirty) slot). It ALSO zeroes `change_count` (the G-c.4
// mirror signal) on invocation 0 so a stale value never leaks into the out-of-band read. GPU-timeline, readback-
// free — the persistent-buffer drive calls this each frame instead of re-uploading zeroed hashes from the host.
@compute @workgroup_size(256)
fn clear_per_frame_hashes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i == 0u) {
        atomicStore(&change_count, 0u);
    }
    if (i < diff_cfg.present_size) {
        atomicStore(&present_flag[i * PRESENT_WORDS + 3u], EMPTY_LOD);
    }
    // dirty_flag is sized = slot_table_size (>= resident set); its stride is 4 words ([x,y,z,lod]).
    if (i < diff_cfg.slot_table_size) {
        atomicStore(&dirty_flag[i * 4u + 3u], EMPTY_LOD);
    }
    // Clear the enter-cap distance histogram (rebuilt each frame from this frame's candidates).
    if (i < HIST_BUCKETS) {
        atomicStore(&enter_hist[i], 0u);
    }
}

// --- Pass C-tail — publish the change_count signal (G-c.4's non-blocking, 1-frame-late CPU mirror reads this). ---
// `change_count = enter_count + drop_count` is the idempotency signal. A dedicated single-invocation pass writes
// it into the `change_count` buffer AFTER Pass C1/C2 so a `COPY_SRC` staging copy + `map_async` (built but NOT
// wired to gate the AS build here — that is G-c.4) can read it out-of-band. Static camera ⇒ change_count == 0.
@compute @workgroup_size(1)
fn write_change_count() {
    atomicStore(&change_count, atomicLoad(&enter_count) + atomicLoad(&drop_count));
}

// **Pass C-cap.A — build the candidate distance HISTOGRAM** (BUG-2 nearest-priority cap). One invocation per
// candidate; bins the NON-resident candidates (only those can enter, mirror of the CPU `to_classify` filter that
// excludes the resident set) by world distance. The histogram drives the cut radius (Pass C-cap.B).
@compute @workgroup_size(256)
fn enter_cap_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= atomicLoad(&candidate_count)) {
        return;
    }
    let k = candidate_list[i];
    let coord = vec3<i32>(k.x, k.y, k.z);
    let lod = u32(k.w);
    if (is_resident(coord, lod)) {
        return; // resident candidates never enter and never count against `room` (CPU excludes them)
    }
    atomicAdd(&enter_hist[dist_bucket(cand_world_dist(coord, lod))], 1u);
}

// **Pass C-cap.B — compute the CUT bucket from the histogram + the live pool room** (single invocation). `room` =
// the free-list slots available THIS frame (`tail - head`, = CPU `budget - already_resident`). The cut bucket is
// the LARGEST `b` whose cumulative candidate count `Σ_{<b} hist` ≤ `room` — admit buckets strictly below it. This
// keeps ≤ `room` nearest candidates (never overfills the pool) and is DETERMINISTIC (same hist ⇒ same cut), so a
// static camera converges. Mirror of `surface_candidates.select_nth_unstable_by(room, dist)` (streaming.rs:798).
@compute @workgroup_size(1)
fn enter_cap_compute() {
    let head = atomicLoad(&free_ctrl[0]);
    let tail = atomicLoad(&free_ctrl[1]);
    let room = select(0u, tail - head, tail > head);
    var acc = 0u;
    var cut = HIST_BUCKETS; // default: no cut (all candidates fit)
    for (var b = 0u; b < HIST_BUCKETS; b = b + 1u) {
        let next = acc + atomicLoad(&enter_hist[b]);
        if (next > room) {
            cut = b; // bucket b would overflow `room` ⇒ admit only buckets strictly below b
            break;
        }
        acc = next;
    }
    enter_cap[0] = cut;
    enter_cap[1] = room;
}

// **Pass C1 — enter scan.** One invocation per candidate (the surface resident-target set). If `slot_table[key]`
// is absent AND its distance bucket is within the cut (BUG-2 nearest cap), claim a free slot (atomic pop the
// free-list ring), insert the key→slot into the slot table, and atomic-append to `enter_list` (+ `enter_count`).
// Mirrors design §1 Pass C1 + `ResidencyManager::update`'s "enqueue desired-but-not-resident" +
// `SlotAllocator::claim` (incremental.rs:595) + the nearest-`room` cap (streaming.rs:783-806).
@compute @workgroup_size(256)
fn diff_enter_scan(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= atomicLoad(&candidate_count)) {
        return;
    }
    let k = candidate_list[i];
    let coord = vec3<i32>(k.x, k.y, k.z);
    let lod = u32(k.w);
    if (is_resident(coord, lod)) {
        return; // already resident — no change
    }
    // NEAREST-priority cap: skip candidates at/after the cut bucket (the farthest, beyond `room`) — keeps the
    // nearest `room`, deterministic + stable ⇒ converges. (cut == HIST_BUCKETS ⇒ everything fits, no cap.)
    if (dist_bucket(cand_world_dist(coord, lod)) >= enter_cap[0]) {
        return;
    }
    // Claim a free slot: atomically advance the free-list head ring index, read the slot id at that index.
    let cap = diff_cfg.max_resident;
    let head = atomicAdd(&free_ctrl[0], 1u);
    let tail = atomicLoad(&free_ctrl[1]);
    if (head >= tail) {
        // Out of slots (would exceed `max_resident`) — undo the claim and skip (the CPU cap drops farthest; here
        // the test sizes the ring to fit the whole resident set, so this never triggers).
        atomicSub(&free_ctrl[0], 1u);
        return;
    }
    let slot = free_ring[head % cap];
    // Insert (coord,lod)->slot into the slot table (atomic linear-probe claim of an EMPTY slot).
    let size = diff_cfg.slot_table_size;
    let mask = size - 1u;
    var t = hash_key(coord, lod) & mask;
    for (var p = 0u; p < size; p = p + 1u) {
        let base = t * SLOT_WORDS;
        // Claim the first EMPTY **or TOMBSTONE** slot on the probe chain (the key is verified absent above, so this
        // never duplicates — and reusing tombstones keeps them from accumulating unbounded under churn, mirror of the
        // reference `PagedBrickCoreStore::insert_slot`). Try EMPTY first; if the slot is a tombstone, claim that.
        let prev = atomicCompareExchangeWeak(&slot_table[base + 3u], EMPTY_LOD, lod);
        var claimed = prev.exchanged;
        if (!claimed && prev.old_value == TOMBSTONE_LOD) {
            claimed = atomicCompareExchangeWeak(&slot_table[base + 3u], TOMBSTONE_LOD, lod).exchanged;
        }
        if (claimed) {
            atomicStore(&slot_table[base + 0u], bitcast<u32>(coord.x));
            atomicStore(&slot_table[base + 1u], bitcast<u32>(coord.y));
            atomicStore(&slot_table[base + 2u], bitcast<u32>(coord.z));
            atomicStore(&slot_table[base + 4u], slot);
            let e = atomicAdd(&enter_count, 1u);
            enter_list[e] = vec4<i32>(coord.x, coord.y, coord.z, i32(lod));
            return;
        }
        t = (t + 1u) & mask;
    }
}

// Per-slot DROP-DECISION flag (1 = this slot will drop), written by Pass C2a (mark) and consumed by Pass C2b
// (apply). Splitting decide-vs-mutate makes `safe_to_drop` see the CONSISTENT pre-drop slot table, exactly as
// the CPU `update` evaluates `safe_to_drop` over the full pre-drop resident set BEFORE removing any
// (streaming.rs:703-713). Without the split a concurrent C2 invocation could clear a table entry another
// invocation's `safe_to_drop` still needs to read as resident → divergence.
@group(0) @binding(23) var<storage, read_write> drop_decision: array<u32>;

// **Pass C2a — drop MARK.** One invocation per slot-table slot. Decides (without mutating the table) whether the
// slot's key drops: occupied AND not present (not desired) AND `safe_to_drop` (keep-old-until-revealed). Writes
// the verdict to `drop_decision[slot_idx]`. All invocations read the SAME pre-drop slot table + present-flag, so
// the residency tests inside `safe_to_drop` are consistent (the CPU evaluates them over the pre-drop set too).
@compute @workgroup_size(256)
fn diff_drop_mark(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot_idx = gid.x;
    if (slot_idx >= diff_cfg.slot_table_size) {
        return;
    }
    drop_decision[slot_idx] = 0u;
    let base = slot_idx * SLOT_WORDS;
    let e_lod = atomicLoad(&slot_table[base + 3u]);
    if (e_lod == EMPTY_LOD || e_lod == TOMBSTONE_LOD) {
        return; // free slot (EMPTY) or a deleted-key hole (TOMBSTONE) — neither holds a live key to drop
    }
    let coord = vec3<i32>(
        bitcast<i32>(atomicLoad(&slot_table[base + 0u])),
        bitcast<i32>(atomicLoad(&slot_table[base + 1u])),
        bitcast<i32>(atomicLoad(&slot_table[base + 2u])),
    );
    let lod = e_lod;
    // Keep ONLY a brick that is still a desired SURFACE candidate (the ENTERED set = present ∧ `classify_surface`).
    // A brick that became BURIED (all 6 face-neighbours occupied ⇒ `classify_surface` false) is INTERIOR — never a
    // visible surface — so drop it even though it's still "present" in the occupied-in-shell desired superset.
    // Leaving buried bricks resident let them accumulate as all-solid-halo bricks that render as flat/degenerate
    // cubes when the ray reaches them through the shell (the stuck-cube-at-LOD-transition bug). A clip-boundary
    // brick still has an AIR neighbour beyond the +1 occupancy pad ⇒ stays `classify_surface` ⇒ kept, so this never
    // holes the loaded-region boundary. (`safe_to_drop` still gates the removal — keep-old-until-revealed.)
    if (present_contains(coord, lod) && classify_surface(coord, lod)) {
        return; // still a desired surface candidate — keep
    }
    if (!safe_to_drop(coord, lod)) {
        return; // keep-old-until-revealed: its replacement is not resident yet
    }
    drop_decision[slot_idx] = 1u;
}

// **Pass C2b — drop APPLY.** One invocation per slot. For each slot Pass C2a marked, release its slot id to the
// QUARANTINE (freed next frame's Pass A, incremental.rs:755), clear the table entry, and atomic-append to
// `drop_list` (+ `drop_count`). Mirrors `ResidencyManager::update`'s drop loop + `SlotAllocator::release`.
@compute @workgroup_size(256)
fn diff_drop_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot_idx = gid.x;
    if (slot_idx >= diff_cfg.slot_table_size) {
        return;
    }
    if (drop_decision[slot_idx] == 0u) {
        return;
    }
    let base = slot_idx * SLOT_WORDS;
    let coord = vec3<i32>(
        bitcast<i32>(atomicLoad(&slot_table[base + 0u])),
        bitcast<i32>(atomicLoad(&slot_table[base + 1u])),
        bitcast<i32>(atomicLoad(&slot_table[base + 2u])),
    );
    let lod = atomicLoad(&slot_table[base + 3u]);
    let cap = diff_cfg.max_resident;
    let q = atomicAdd(&quarantine_ctrl[1], 1u);
    quarantine_ring[q % cap] = atomicLoad(&slot_table[base + 4u]);
    atomicStore(&slot_table[base + 3u], TOMBSTONE_LOD); // TOMBSTONE (not EMPTY) — preserve other keys' probe chains
    let d = atomicAdd(&drop_count, 1u);
    drop_list[d] = vec4<i32>(coord.x, coord.y, coord.z, i32(lod));
}

// =====================================================================================================
//  G-c.2b — PASS D: the GPU PACK-COMMAND BUILD + the GPU SLAB ALLOCATOR (docs/PHASE_G_GC_PLAN.md §1 Pass D,
//  §2.3, §2.4). From `enter_list`/`drop_list` + `slot_table` (Pass C output), GPU-build the SAME
//  `PackCommand`/`AabbCommand`/`ClassifyCommand`/uniform-meta buffers the LANDED `voxel_pack.wgsl`
//  (`pack_brick`/`write_aabb`/`classify_brick`) consumes — so the GPU-built commands replace the CPU
//  `ResidentPacker::update_gpu` driver. The SSOT this MIRRORS bit-for-bit is `src/voxel/incremental.rs`
//  (`update_gpu`/`emit_pack_command`/`build_neighbour_table`/`neighbourhood_26` + the `SlabArena` index/palette
//  size-class allocators). Where the CPU is SERIAL (free-list LIFO + slab bump order), the GPU is PARALLEL, so
//  the SLOT and SLAB OFFSETS differ in order — but each resident key's CONTENT (decoded via the SSOT
//  `cell_block`) and the rendered RESULT are IDENTICAL. The gate is per-KEY content parity + ray-HIT parity, NOT
//  per-slot byte identity (see `tests/voxel_gpu_residency_pack_parity.rs`).
//
//  ## The dependency chain (why Pass D is SPLIT around the classify pass)
//  The CPU does, per dirty brick, classify → slab-alloc → emit. On the GPU the classify is `classify_brick`
//  (LANDED, in `voxel_pack.wgsl`), which needs the per-command 27-neighbour TABLE. The slab alloc + the final
//  PackCommand (which carries the slab offsets) need the classify result. So Pass D is THREE GPU sub-passes
//  around the landed classify:
//    * Pass D1 (`pack_build_dirty`)      — from enter/drop lists, build the DEDUPED dirty key set (the
//      entered/dropped keys' resident SAME-LOD 26-neighbourhood ∪ the entered keys themselves — the halo
//      dependency, mirror of `neighbourhood_26` + the §3 expansion), each tagged with its slot (from
//      `slot_table`). Append to `dirty_list`/`dirty_count` + `atomicMax` the classify dispatch.
//    * Pass D2 (`pack_build_neighbours`) — per dirty key, build its 27-entry neighbour table into
//      `neighbour_indices` (core-pool index per neighbour via `core_table`, or NEIGHBOUR_ABSENT) AND emit one
//      `classify_command` (its `neighbour_base`). Mirror of `build_neighbour_table`.
//      [classify_brick runs here — LANDED `voxel_pack.wgsl`, `record_indirect` over the classify dispatch.]
//    * Pass D3 (`pack_build_commands`)   — per dirty key, read `classify_out`: a DENSE brick atomically
//      allocates an index slab (its `index_bits` size class) + a palette slab (the power-of-2 ladder) from the
//      GPU slab allocators, emits a `PackCommand` (the GPU slab offsets) + a resident `AabbCommand` +
//      `atomicMax` the pack/aabb dispatches; a UNIFORM brick GPU-writes its 48-B meta straight + a resident
//      `AabbCommand`. Mirror of `emit_pack_command`. Drops are handled in Pass D0 (degenerate meta + AABB).
//    * Pass D0 (`pack_build_drops`)      — per `drop_list` key, GPU-write its slot's ZEROED meta + a freed
//      (degenerate) `AabbCommand` + `atomicMax` the aabb dispatch. Mirror of `update_gpu`'s drop loop.
//
//  ## The GPU SLAB ALLOCATOR (§2.3) — bump + per-class free-list, fixed-cap pre-sized
//  `index_slab_ctrl`/`palette_slab_ctrl` are per-size-class bump high-waters + free-list rings, replacing the
//  CPU `SlabArena`. An alloc takes the smallest class ≥ the request, popping the class free-list first (LIFO,
//  mirror of `SlabArena::alloc`) else bumping the class high-water; a release (on a re-class / drop) pushes to
//  the class free-list. The 5 INDEX classes are `index_class_words({1,2,4,8,16})` = {32,63,125,250,500}; the 16
//  PALETTE classes are the power-of-2 ladder {2,4,…,65536}. Pre-sized to `max_resident` × the RESERVE_* means so
//  a normal load never overflows (the §2.3 fixed-cap pool).

// =====================================================================================================
//  Pack-command structs — MIRROR `voxel_pack.wgsl` / `src/voxel/incremental.rs` FIELD-FOR-FIELD (so the GPU
//  emits the IDENTICAL records the landed pack/aabb/classify passes consume).
// =====================================================================================================

const NEIGHBOUR_ABSENT: u32 = 0xFFFFFFFFu;
// A neighbour that is OCCUPIED (per the occupancy structure) but has no resident CORE (a face-culled interior
// brick, or a surface brick not yet entered this frame during streaming). The halo must treat it as SOLID — NOT
// air — so a brick's face toward it reads correctly BURIED (the normal/exposed-face test). Treating it as air was
// the motion-only "black cube" (wrong-normal speck): a streaming brick whose neighbour hadn't entered yet baked a
// spurious exposed face. `fill_halo` fills these border cells with a solid marker instead of reading a core.
const NEIGHBOUR_SOLID: u32 = 0xFFFFFFFEu;
const NEIGHBOUR_COUNT: u32 = 27u;     // the 27-entry neighbour table per command (brick + its 26 neighbours)
const BRICK_EDGE_D: i32 = 8;
const BRICK_VOXELS_D: u32 = 512u;     // BRICK_EDGE³ — one core's u32 count
const META_WORDS: u32 = 12u;          // 48-B GpuBrickMeta = 12 u32
const META_FLAG_UNIFORM: u32 = 1u;

// 60-B PackCommand (15 u32) — mirror of `voxel_pack.wgsl::PackCommand` / `GpuPackCommand`.
struct PackCommandD {
    origin_x: i32,
    origin_y: i32,
    origin_z: i32,
    slot: u32,
    world_min_x: f32,
    world_min_y: f32,
    world_min_z: f32,
    index_word_offset: u32,
    lod: u32,
    index_bits: u32,
    palette_word_offset: u32,
    neighbour_base: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// 32-B AabbCommand (8 u32) — mirror of `voxel_pack.wgsl::AabbCommand` / `GpuAabbCommand`.
struct AabbCommandD {
    slot: u32,
    lod: u32,
    flag: u32, // 1 = resident, 0 = freed
    _pad0: u32,
    world_min_x: f32,
    world_min_y: f32,
    world_min_z: f32,
    _pad1: u32,
}

// 16-B ClassifyCommand (4 u32) — mirror of `voxel_pack.wgsl::ClassifyCommand` / `GpuClassifyCommand`.
struct ClassifyCommandD {
    neighbour_base: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// =====================================================================================================
//  Pass D bindings (24+). The CORE STORE (§2.4): `core_table` is a `(coord,lod) -> core-pool index` hash (same
//  FNV-1a family as `slot_table`, 5-word stride [x,y,z,lod,core_index]), `cores` the deduped 8³ cores. Built per
//  region CPU-side (the test builds it from the scene's occupied keys). The DENSE-brick output buffers
//  (`pack_commands`/`aabb_commands`/`classify_commands`/`neighbour_indices`/`meta_buf`) are the SAME the landed
//  pack passes read. `pack_dispatch`/`aabb_dispatch`/`classify_dispatch` are `atomicMax`-built (x,1,1) indirect
//  dispatches so E/F/G `record_indirect`.
// =====================================================================================================

struct PackConfigD {
    core_table_size: u32,   // power of two (probe mask = size - 1)
    max_resident: u32,
    index_stride: u32,      // WORDS per slot in the index pool (= RESERVE_INDEX_WORDS_PER_BRICK; fixed per-slot slab)
    palette_stride: u32,    // WORDS per slot in the palette pool (= RESERVE_PALETTE_WORDS_PER_BRICK)
}
@group(0) @binding(24) var<uniform> pack_cfg: PackConfigD;
@group(0) @binding(25) var<storage, read> core_table: array<u32>;        // 5 u32/slot: [x,y,z,lod,core_index]
@group(0) @binding(26) var<storage, read> cores: array<u32>;             // deduped 8³ cores (512 u32 each)

// The DEDUPED, slot-tagged DIRTY key list (Pass D1 output) — each entry (x,y,z,lod) + its slot in a parallel
// `dirty_slot` array. Pass D2 builds its neighbour table; Pass D3 emits its command after the classify.
@group(0) @binding(27) var<storage, read_write> dirty_count: atomic<u32>;
@group(0) @binding(28) var<storage, read_write> dirty_list: array<vec4<i32>>;
@group(0) @binding(29) var<storage, read_write> dirty_slot: array<u32>;
// The dirty-dedup hash (4 u32/slot [x,y,z,lod]; lod==EMPTY_LOD ⇒ free), CAS-claimed in Pass D1.
@group(0) @binding(30) var<storage, read_write> dirty_flag: array<atomic<u32>>;

// Pack/aabb/classify command output buffers (the LANDED pack passes' inputs) + their atomic counts + indirect
// dispatches. `neighbour_indices` is the per-command 27-entry table (concatenated).
@group(0) @binding(31) var<storage, read_write> pack_count: atomic<u32>;
@group(0) @binding(32) var<storage, read_write> pack_commands: array<PackCommandD>;
@group(0) @binding(33) var<storage, read_write> aabb_count: atomic<u32>;
@group(0) @binding(34) var<storage, read_write> aabb_commands: array<AabbCommandD>;
@group(0) @binding(35) var<storage, read_write> classify_commands: array<ClassifyCommandD>;
@group(0) @binding(36) var<storage, read_write> neighbour_indices: array<u32>;
@group(0) @binding(37) var<storage, read_write> meta_buf: array<u32>;    // 12 u32/slot (48 B GpuBrickMeta)
@group(0) @binding(38) var<storage, read_write> pack_dispatch: array<atomic<u32>>;    // [x,1,1]
@group(0) @binding(39) var<storage, read_write> aabb_dispatch: array<atomic<u32>>;    // [x,1,1]
@group(0) @binding(40) var<storage, read_write> classify_dispatch: array<atomic<u32>>;// [x,1,1]

// FIXED PER-SLOT SLABS (§2.3) — the index + palette pools are reserved worst-case-per-slot (incremental.rs
// RESERVE_INDEX_WORDS_PER_BRICK / RESERVE_PALETTE_WORDS_PER_BRICK, passed as `pack_cfg.index_stride` /
// `palette_stride`), so slot `s` OWNS the region `[s·stride, (s+1)·stride)`. Pass D3 writes a brick's slab at
// `pool_base + slot·stride` directly — NO allocator, NO free-list, NO per-slot state. (This replaced a shared
// bump+free-list allocator whose `free` published a ring slot via `atomicAdd(tail)` BEFORE writing its offset, so
// a concurrent `alloc` pop could read the slot mid-write ⇒ two live bricks alias one slab ⇒ garbage content. With
// both pools worst-case-per-slot the allocator gave ZERO VRAM benefit, so fixed offsets are strictly better.)
@group(0) @binding(45) var<storage, read> index_pool_base: array<u32>;   // [0] = the index pool's word base
@group(0) @binding(46) var<storage, read> palette_pool_base: array<u32>; // [0] = the palette pool's word base

// --- core_table lookup (key -> deduped core-pool index, or NEIGHBOUR_ABSENT) ---
fn core_lookup(coord: vec3<i32>, lod: u32) -> u32 {
    let size = pack_cfg.core_table_size;
    if (size == 0u) {
        return NEIGHBOUR_ABSENT;
    }
    let mask = size - 1u;
    var slot = hash_key(coord, lod) & mask;
    for (var i = 0u; i < size; i = i + 1u) {
        let base = slot * 5u;
        let e_lod = core_table[base + 3u];
        if (e_lod == EMPTY_LOD) {
            return NEIGHBOUR_ABSENT;
        }
        if (e_lod == lod
            && bitcast<i32>(core_table[base + 0u]) == coord.x
            && bitcast<i32>(core_table[base + 1u]) == coord.y
            && bitcast<i32>(core_table[base + 2u]) == coord.z) {
            return core_table[base + 4u];
        }
        slot = (slot + 1u) & mask;
    }
    return NEIGHBOUR_ABSENT;
}

// --- per-brick GEOMETRY (mirror of `BrickGeom::of` / brickmap `brick_span`) — pure function of the key. ---
fn brick_span_d(lod: u32) -> f32 {
    return f32(BRICK_EDGE_D) * 0.05 * f32(1u << min(lod, MAX_LOD)); // 0.05 = VOXEL_SIZE
}

// Write the 48-B UNIFORM meta for `slot` (mirror of `GpuBrickMeta::uniform`): id in low 16b of voxel_offset,
// META_FLAG_UNIFORM in flags. `lod_and_bits = (lod & 7) | (0 << 3)` (index_bits 0 for uniform).
fn write_uniform_meta(slot: u32, coord: vec3<i32>, lod: u32, block: u32) {
    let base = slot * META_WORDS;
    let span = brick_span_d(lod);
    meta_buf[base + 0u] = bitcast<u32>(coord.x * BRICK_EDGE_D);
    meta_buf[base + 1u] = bitcast<u32>(coord.y * BRICK_EDGE_D);
    meta_buf[base + 2u] = bitcast<u32>(coord.z * BRICK_EDGE_D);
    meta_buf[base + 3u] = block & 0xFFFFu;                       // voxel_offset = uniform id
    meta_buf[base + 4u] = bitcast<u32>(f32(coord.x) * span);     // world_min
    meta_buf[base + 5u] = bitcast<u32>(f32(coord.y) * span);
    meta_buf[base + 6u] = bitcast<u32>(f32(coord.z) * span);
    meta_buf[base + 7u] = lod & 0x7u;                            // lod_and_bits (index_bits = 0)
    meta_buf[base + 8u] = 0u;                                    // palette_base
    meta_buf[base + 9u] = META_FLAG_UNIFORM;                     // flags
    meta_buf[base + 10u] = 0u;
    meta_buf[base + 11u] = 0u;
}

// Write the 48-B ZEROED meta for a freed `slot` (mirror of `GpuBrickMeta::zeroed`).
fn write_zeroed_meta(slot: u32) {
    let base = slot * META_WORDS;
    for (var w = 0u; w < META_WORDS; w = w + 1u) {
        meta_buf[base + w] = 0u;
    }
}

// **Pass D0 — DROPS.** One invocation per `drop_list` key: GPU-write its slot's ZEROED meta + a FREED
// (degenerate) AABB command. The slot was released to the quarantine by Pass C2b; its slot id is no longer in
// the slot_table, so we recover it from `dirty_slot`? No — drops carry no dirty entry. The drop's slot was the
// table slot Pass C2b cleared; we re-derive nothing — instead Pass C2b ALSO records the freed slot here. To keep
// Pass C unchanged we instead pass the freed slot via `drop_list.w`? The CPU path knows the slot. SIMPLER: the
// drop's meta is zeroed by re-using the quarantine ring (the slots Pass C2b pushed). One invocation per
// quarantine entry pushed THIS frame.
@compute @workgroup_size(256)
fn pack_build_drops(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= atomicLoad(&drop_count)) {
        return;
    }
    // The drop's slot is the quarantine entry pushed by Pass C2b in the SAME order (drop i -> quarantine i,
    // both atomic-appended in `diff_drop_apply`). The quarantine head is 0 at this point (Pass A reset it; this
    // frame's pushes start at 0), so quarantine_ring[i] is drop i's slot.
    let cap = diff_cfg.max_resident;
    let slot = quarantine_ring[i % cap];
    // With FIXED per-slot slabs there is nothing to free on a drop (the slot's [slot·stride] region stays its own;
    // a future re-enter overwrites it in place) — just zero the meta so the slot reads as freed.
    write_zeroed_meta(slot);
    let a = atomicAdd(&aabb_count, 1u);
    aabb_commands[a] = AabbCommandD(slot, 0u, 0u, 0u, 0.0, 0.0, 0.0, 0u);
    atomicMax(&aabb_dispatch[0], (a + 1u + 63u) / 64u); // workgroup_size 64 for write_aabb
}

// **Pass D1 — build the DEDUPED dirty key set.** One invocation per (enter ∪ drop) key. The dirty set =
// {entered keys} ∪ {resident SAME-LOD 26-neighbours of each entered/dropped key} (the halo dependency). A
// resident key is dirty iff it is in the slot_table (so we can carry its slot). Dedup via the `dirty_flag` CAS
// hash; the winner appends to `dirty_list`/`dirty_slot` + atomicMax the classify dispatch. (The seed itself: an
// ENTERED key is resident — add it; a DROPPED key is NOT resident — only its resident neighbours matter.)
//
// SCOPE (G-c.2b): this is the FIRST-ORDER expansion (entered keys + their/dropped keys' 26-ring). It is EXACT
// for the COLD-FILL gate (`tests/voxel_gpu_residency_pack_parity.rs` drives one B/C/D+pack round over a known
// scene → every key is ENTERED → dirty = the whole resident set, the neighbours adding nothing new), which is
// the per-KEY content + ray-HIT parity proof. The CPU `update_gpu` (step 3) ALSO expands a dropped-key
// NEIGHBOUR by its OWN neighbours (a second-order ring); that only matters across MULTI-round dynamic moves
// (drop a brick → its neighbour's halo flips → that neighbour's OWN neighbours' halos are unaffected, so the
// second-order ring is a conservative re-pack, never a CORRECTNESS gap for a single round) — wiring the full
// multi-round closure + its idempotency is the NEXT stage G-c.3 (convergence), per the design.
fn try_mark_dirty(coord: vec3<i32>, lod: u32) {
    let slot = slot_lookup(coord, lod);
    if (slot == SLOT_ABSENT) {
        return; // only RESIDENT keys are re-packed (a dropped key has no slot / no command)
    }
    // CAS-claim the dirty-dedup hash so each resident key is emitted ONCE even when reached from many seeds.
    let size = diff_cfg.slot_table_size; // dirty_flag is sized = slot_table_size (>= resident set)
    let mask = size - 1u;
    var t = hash_key(coord, lod) & mask;
    for (var p = 0u; p < size; p = p + 1u) {
        let base = t * 4u;
        let e_lod = atomicLoad(&dirty_flag[base + 3u]);
        if (e_lod == lod
            && bitcast<i32>(atomicLoad(&dirty_flag[base + 0u])) == coord.x
            && bitcast<i32>(atomicLoad(&dirty_flag[base + 1u])) == coord.y
            && bitcast<i32>(atomicLoad(&dirty_flag[base + 2u])) == coord.z) {
            return; // already claimed by this exact key
        }
        let prev = atomicCompareExchangeWeak(&dirty_flag[base + 3u], EMPTY_LOD, lod);
        if (prev.exchanged) {
            atomicStore(&dirty_flag[base + 0u], bitcast<u32>(coord.x));
            atomicStore(&dirty_flag[base + 1u], bitcast<u32>(coord.y));
            atomicStore(&dirty_flag[base + 2u], bitcast<u32>(coord.z));
            let d = atomicAdd(&dirty_count, 1u);
            dirty_list[d] = vec4<i32>(coord.x, coord.y, coord.z, i32(lod));
            dirty_slot[d] = slot;
            atomicMax(&classify_dispatch[0], d + 1u); // 1 workgroup per dirty key (classify_brick)
            return;
        }
        // CAS failed: a DIFFERENT key owns this slot — probe on.
        t = (t + 1u) & mask;
    }
}

@compute @workgroup_size(256)
fn pack_build_dirty(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_enter = atomicLoad(&enter_count);
    let n_drop = atomicLoad(&drop_count);
    let i = gid.x;
    if (i >= n_enter + n_drop) {
        return;
    }
    var coord: vec3<i32>;
    var lod: u32;
    var is_enter: bool;
    if (i < n_enter) {
        let k = enter_list[i];
        coord = vec3<i32>(k.x, k.y, k.z);
        lod = u32(k.w);
        is_enter = true;
    } else {
        let k = drop_list[i - n_enter];
        coord = vec3<i32>(k.x, k.y, k.z);
        lod = u32(k.w);
        is_enter = false;
    }
    // The entered key itself is resident ⇒ dirty.
    if (is_enter) {
        try_mark_dirty(coord, lod);
    }
    // Its (and a dropped key's) resident SAME-LOD 26-neighbours' halos read this brick's core ⇒ dirty them.
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }
                try_mark_dirty(coord + vec3<i32>(dx, dy, dz), lod);
            }
        }
    }
}

// **Pass D2 — build the 27-neighbour table + the classify command** per dirty key. `neighbour_base = i*27`
// (the table is laid out command-major, mirror of `build_neighbour_table`). Slot 13 is the brick itself.
@compute @workgroup_size(256)
fn pack_build_neighbours(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= atomicLoad(&dirty_count)) {
        return;
    }
    let k = dirty_list[i];
    let coord = vec3<i32>(k.x, k.y, k.z);
    let lod = u32(k.w);
    let base = i * NEIGHBOUR_COUNT;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nslot = u32((dz + 1) * 9 + (dy + 1) * 3 + (dx + 1));
                let nbr = coord + vec3<i32>(dx, dy, dz);
                // The halo reflects ACTUAL RESIDENT geometry only: the neighbour's real core if it's paged (the core
                // store holds both entered bricks AND the +1-halo-ring cores the pager pages for occupied neighbours),
                // else AIR. A neighbour with NO resident core has no geometry the ray can hit, so its face is EXPOSED
                // from the ray's point of view — the ray reaches THIS brick through the empty neighbour space. Packing
                // that face SOLID (the old `NEIGHBOUR_SOLID`/`is_full` guess) buried a face the ray then hit anyway,
                // giving an all-solid neighbourhood ⇒ zero occupancy gradient ⇒ a degenerate normal (a flat/black
                // cube), and it STUCK at coarse-LOD coverage gaps where the guessed neighbour never pages in. AIR is
                // correct: a real exposed face (valid normal), and if the neighbour's core later pages in this brick
                // re-packs to its true boundary. (If the neighbour IS resident with geometry, the ray hits IT first,
                // so this brick's face is never primary-hit — AIR there is invisible.)
                neighbour_indices[base + nslot] = core_lookup(nbr, lod);
            }
        }
    }
    classify_commands[i] = ClassifyCommandD(base, 0u, 0u, 0u);
}

// **Pass D3 — emit the per-dirty-key COMMAND from the classify result.** Reads `classify_out[i]` (LANDED
// `classify_brick` output, 4 u32: is_uniform, uniform_block, palette_k, index_bits). DENSE ⇒ alloc the GPU index
// + palette slabs, emit a PackCommand (the slab offsets) + a resident AabbCommand. UNIFORM ⇒ GPU-write the meta
// straight + a resident AabbCommand. Mirror of `emit_pack_command`. `classify_out` is bound at the SAME binding
// as `voxel_pack.wgsl::classify_out` (binding 47 here) so the pass reads it without a readback.
@group(0) @binding(47) var<storage, read> classify_out: array<u32>;

// **G-c.3 — the change_count signal buffer** (`docs/PHASE_G_GC_PLAN.md` §3.1). A single-`u32` storage buffer the
// `write_change_count` pass publishes `enter_count + drop_count` into, and `clear_per_frame_hashes` zeroes at the
// top of the frame. G-c.4's non-blocking 1-frame-late CPU mirror will `map_async` a `COPY_SRC` staging copy of it
// to decide whether to RECORD the AS build (no indirect AS on the fork). NOT wired to gate the AS build here.
@group(0) @binding(48) var<storage, read_write> change_count: atomic<u32>;

@compute @workgroup_size(256)
fn pack_build_commands(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= atomicLoad(&dirty_count)) {
        return;
    }
    let k = dirty_list[i];
    let coord = vec3<i32>(k.x, k.y, k.z);
    let lod = u32(k.w);
    let slot = dirty_slot[i];
    let base = i * 4u;
    // word 0 packs is_uniform (bit 0) + has_air (bit 1, from classify_brick). A brick with NO air anywhere in its
    // haloed grid is BURIED (uniform-incl-halo, or dense fully surrounded by solid) — it has no visible surface, so
    // it gets a DEGENERATE (freed) AABB and is never traced: rendering it gave an all-solid-neighbourhood hit ⇒
    // degenerate normal ⇒ a flat/black cube when the ray reached it through the shell. It stays RESIDENT (its core
    // still feeds neighbours' halos via the pager); only its BLAS primitive is suppressed.
    let is_uniform = classify_out[base + 0u] & 1u;
    let has_air = (classify_out[base + 0u] >> 1u) & 1u;
    let palette_k = classify_out[base + 2u];
    let index_bits = classify_out[base + 3u];
    // FIXED PER-SLOT SLABS (replaces the shared bump+free-list allocator, which RACED — an alloc could pop a
    // free-list slot mid-write ⇒ two LIVE bricks share one slab offset ⇒ one reads the other's content ⇒ garbage
    // cubes). Each slot OWNS a unique [slot·stride, (slot+1)·stride) region in BOTH pools, reserved worst-case-
    // per-slot (incremental.rs RESERVE_INDEX_WORDS_PER_BRICK / RESERVE_PALETTE_WORDS_PER_BRICK = the strides). So
    // two bricks can NEVER alias a slab — the race is gone by construction. The ONE case a fixed slot can't hold
    // is an `index_bits=16` brick (>256-entry palette > the 256-word palette stride): treat it as BURIED (degenerate
    // AABB, no pool write) rather than overflow into the next slot — real `index_bits ≤ 8` scenes never hit this.
    let palette_fits = index_bits != 16u;
    let buried = (is_uniform != 0u) || (has_air == 0u) || (!palette_fits);
    let span = brick_span_d(lod);
    let world_min = vec3<f32>(f32(coord.x) * span, f32(coord.y) * span, f32(coord.z) * span);

    // AABB command: resident (flag 1) for a brick with a visible surface that FITS; freed/degenerate (flag 0) for a
    // BURIED brick (no air ⇒ never a primary-hit surface) or a non-fitting one. write_aabb_slot writes degenerate for 0.
    let a = atomicAdd(&aabb_count, 1u);
    let aabb_flag = select(1u, 0u, buried);
    aabb_commands[a] = AabbCommandD(slot, lod, aabb_flag, 0u, world_min.x, world_min.y, world_min.z, 0u);
    atomicMax(&aabb_dispatch[0], (a + 1u + 63u) / 64u);

    if (is_uniform != 0u || !palette_fits) {
        // UNIFORM (or a non-fitting index_bits=16 brick degraded to empty) — GPU-write the meta straight; the slot
        // holds no dense pool data (fixed per-slot ⇒ no slab to free, nothing to track).
        let ublock = select(0u, classify_out[base + 1u], is_uniform != 0u); // !fits ⇒ uniform-AIR (empty, degenerate)
        write_uniform_meta(slot, coord, lod, ublock);
        return;
    }
    // DENSE — fixed per-slot offsets. `pack_brick` writes the index stream at `index_off` and the palette at
    // `palette_off`; the brick reuses the SAME region every re-pack (no churn, no high-water, no aliasing).
    let index_off = index_pool_base[0] + slot * pack_cfg.index_stride;
    let palette_off = palette_pool_base[0] + slot * pack_cfg.palette_stride;
    let p = atomicAdd(&pack_count, 1u);
    pack_commands[p] = PackCommandD(
        coord.x * BRICK_EDGE_D, coord.y * BRICK_EDGE_D, coord.z * BRICK_EDGE_D, slot,
        world_min.x, world_min.y, world_min.z,
        index_off, lod, index_bits, palette_off,
        i * NEIGHBOUR_COUNT, 0u, 0u, 0u,
    );
    atomicMax(&pack_dispatch[0], p + 1u); // 1 workgroup per dense pack command (pack_brick)
}

// Convert a per-item indirect dispatch (built as `[count, 1, 1]` by the atomicMax above / `try_mark_dirty`) into a
// 2D `[x, y, 1]` grid: `x = min(count, 65535)`, `y = ceil(count / 65535)`. A per-brick pass (1 workgroup/brick)
// otherwise can't exceed the 65535 workgroups-per-dimension limit (Bistro needs ~610k). The kernel recovers
// `cmd_idx = wg.x + wg.y*65535`. Run (1 invocation) AFTER the count is final, BEFORE the indirect dispatch.
@compute @workgroup_size(1)
fn finalize_shell_dispatch_2d() {
    let n = atomicLoad(&shell_count);
    atomicStore(&shell_dispatch[0u], select(n, 65535u, n > 65535u));
    atomicStore(&shell_dispatch[1u], (n + 65534u) / 65535u);
    atomicStore(&shell_dispatch[2u], 1u);
}
@compute @workgroup_size(1)
fn finalize_pack_dispatch_2d() {
    let n = atomicLoad(&pack_dispatch[0u]);
    atomicStore(&pack_dispatch[0u], select(n, 65535u, n > 65535u));
    atomicStore(&pack_dispatch[1u], (n + 65534u) / 65535u);
    atomicStore(&pack_dispatch[2u], 1u);
}
@compute @workgroup_size(1)
fn finalize_classify_dispatch_2d() {
    let n = atomicLoad(&classify_dispatch[0u]);
    atomicStore(&classify_dispatch[0u], select(n, 65535u, n > 65535u));
    atomicStore(&classify_dispatch[1u], (n + 65534u) / 65535u);
    atomicStore(&classify_dispatch[2u], 1u);
}
