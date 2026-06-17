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

// The per-sector record is 6 u32 words (matching `GpuSectorEntry`, 24 B). We read the entries buffer as a FLAT
// `array<u32>` and index it with an explicit 6-word stride, so there is ZERO struct-layout/stride ambiguity
// across naga back-ends (a struct-array's element stride is back-end-rounded; a flat u32 array is not). The 6
// words per slot, in order: [sector_x, sector_y, sector_z, lod, mask_lo, mask_hi].
const WORDS_PER_ENTRY: u32 = 6u;

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

// Probe the table for `(sector, lod)`; return its 64-bit mask split as `vec2<u32>(lo, hi)`, or `(0,0)` if the
// sector is absent. The SINGLE fetch both `is_occupied` and the coarse test derive from. Reads the FLAT u32
// entries buffer at an explicit 6-word stride (no struct-layout ambiguity).
fn sector_mask(sector: vec3<i32>, lod: u32) -> vec2<u32> {
    let table_size = residency_header.table_size;
    if (table_size == 0u) {
        return vec2<u32>(0u, 0u);
    }
    let mask_bits = table_size - 1u;
    var slot = hash_sector(sector, lod) & mask_bits;
    // Probe at most `table_size` slots; a free slot ⇒ absent (the build keeps the table < 100% full).
    for (var i = 0u; i < table_size; i = i + 1u) {
        let base = slot * WORDS_PER_ENTRY;
        let e_lod = residency_entries[base + 3u];
        if (e_lod == EMPTY_LOD) {
            return vec2<u32>(0u, 0u); // first free slot ⇒ key absent
        }
        let e_sx = bitcast<i32>(residency_entries[base + 0u]);
        let e_sy = bitcast<i32>(residency_entries[base + 1u]);
        let e_sz = bitcast<i32>(residency_entries[base + 2u]);
        if (e_lod == lod && e_sx == sector.x && e_sy == sector.y && e_sz == sector.z) {
            return vec2<u32>(residency_entries[base + 4u], residency_entries[base + 5u]);
        }
        slot = (slot + 1u) & mask_bits;
    }
    return vec2<u32>(0u, 0u);
}

// Is the `(coord, lod)` brick occupied? (The face-cull input for G-c.1.)
fn is_occupied(coord: vec3<i32>, lod: u32) -> bool {
    let sector = sector_of(coord);
    let local = local_in_sector(coord);
    let bit = bit_index(local);
    let mask = sector_mask(sector, lod);
    if (bit < 32u) {
        return ((mask.x >> bit) & 1u) != 0u;
    }
    return ((mask.y >> (bit - 32u)) & 1u) != 0u;
}

// The coarse "is ANY brick in this sector occupied?" — the §1 Pass B0 test, from the SAME `sector_mask` fetch.
fn sector_any_occupied(sector: vec3<i32>, lod: u32) -> bool {
    let mask = sector_mask(sector, lod);
    return (mask.x != 0u) || (mask.y != 0u);
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
