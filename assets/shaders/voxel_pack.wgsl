// **Phase G Stage G-a — the GPU brick PACK** (docs/PHASE_G_GALLERY_PLAN.md §"Stage G-a").
//
// Moves the PURE per-brick pack — `pack_one`'s halo-fill + `encode_paletted` + the buffer-write of the
// bit-packed index stream / per-brick palette / `GpuBrickMeta` — off the CPU and onto the GPU. The CPU keeps
// ALLOCATION (the slot/arena claim, the dirty-set + 26-neighbourhood expansion); this shader only encodes the
// bytes the CPU `ResidentPacker` would have written, BYTE-IDENTICALLY.
//
// ## The byte SSOT this shader mirrors EXACTLY (src/voxel/gpu.rs)
// - `halo_index(x,y,z)` — the haloed-grid cell layout (+X fastest, then +Y, then +Z, at edge 10).
// - `pack_one` — the dense halo-fill: core from the brick, the 1-cell border from the SAME-LOD neighbour
//   (AIR where absent), in `halo_index` order.
// - `neighbour_border_cell` — which neighbour owns an out-of-core cell (`div_euclid`/`rem_euclid` on edge 8).
// - `encode_paletted` — first-seen palette + `pow2_index_bits` bit-packing into `u32` words.
// - `GpuBrickMeta::dense` (48 B) — the meta layout written into `meta_buf` at `slot·12` u32s.
// If ANY of these drifts the headless byte-equality gate (`tests/voxel_gpu_pack_parity.rs`) fails — that test
// is the make-or-break anchor.
//
// ## The palette-ORDER risk + its mitigation (the hardest part)
// `encode_paletted` appends palette ids in CELL-ITERATION order (first-seen). A naive parallel encode would
// permute the palette → different bytes (decodes to the same ids, but FAILS the byte gate). So the
// palette-build step here is SERIAL within the workgroup: invocation 0 walks all 1000 haloed cells in exact
// `halo_index` order, building the palette + the per-cell local-index map in workgroup shared memory. Only
// AFTER that (a workgroup barrier) do all invocations bit-pack the local indices in parallel. Order-identical
// to the CPU by construction.
//
// ## Layout
// One WORKGROUP per dirty DENSE brick (a uniform / freed brick needs no GPU encode — the CPU emits its meta
// straight, identical to the Delta arm). Each command names its slot + alloc offsets and points at the 27
// `8³` neighbour cores (the brick + its 26 neighbours) the halo reads, with a presence bit per neighbour.

// **Phase G Stage G-b — the GPU AABB write** (docs/PHASE_G_GALLERY_PLAN.md §"Stage G-b").
//
// G-a wrote each slot's BLAS AABB on the CPU (a per-slot `queue_write_buffer` into `aabb_buf`, lifted from the
// `Delta` arm — the `vox_blas_delta` cost). G-b moves that write to the GPU so the AABB fill can run in the SAME
// submission as the BLAS build (fill-then-build, readback-free), eliminating the per-slot CPU upload entirely.
// A SECOND, lightweight entry point `write_aabb` (one INVOCATION per changed slot, NOT one workgroup) consumes an
// `aabb_commands` array covering EVERY changed slot — dense, uniform, AND freed — and writes `aabb_buf[slot]`:
//   - a RESIDENT slot (dense or uniform) → `brick_aabb(world_min, lod)` (the epsilon-grown box),
//   - a FREED slot → `degenerate_aabb()` (min > max, a BLAS non-candidate).
// The per-brick `pack_brick` workgroups (dense encode) do NOT touch the AABB; the dedicated AABB pass owns every
// slot's box, so the CPU `aabb` upload is gone. The `brick_aabb` / `degenerate_aabb` / `brick_aabb_epsilon` math
// below MIRRORS src/voxel/gpu.rs (`brick_aabb`/`brick_aabb_epsilon`/`BRICK_AABB_REL_EPS`) + src/voxel/
// incremental.rs (`degenerate_aabb`) EXACTLY — the G-b byte-equality gate (`tests/voxel_gpu_pack_parity.rs`)
// asserts `aabb_buf` byte-equal to the CPU `SnapshotBuffers.aabbs`, freed slots included.

// The brick edge (mirror of BRICK_EDGE in src/voxel/brickmap.rs). 8³ = 512 voxels per core.
const BRICK_EDGE: i32 = 8;
const CORE_CELLS: u32 = 512u;       // BRICK_EDGE³
// The haloed edge (= BRICK_EDGE + 2) and cell count (mirror of halo_edge/halo_cells in src/voxel/gpu.rs).
const HALO_EDGE: i32 = 10;
const HALO_CELLS: u32 = 1000u;      // HALO_EDGE³
// The 27 neighbour slots: index = (dz+1)*9 + (dy+1)*3 + (dx+1), dx,dy,dz ∈ {-1,0,1}. Slot 13 is the centre.
const NEIGHBOUR_COUNT: u32 = 27u;
// Mirror of META_FLAG_UNIFORM in src/voxel/gpu.rs (unused here — every command is dense — kept for clarity).
const META_FLAG_UNIFORM: u32 = 1u;

// One per-brick PACK command — a FLAT 15-u32 (60 B) record (NO `vec3`, whose 16-byte WGSL alignment would
// silently insert padding and misalign the `array<PackCommand>` stride against the tightly-packed Rust
// `#[repr(C)]`). Every field is a scalar `u32`/`i32`, so the WGSL stride is exactly 60 B = the Rust struct.
// Mirrors `GpuPackCommand` in src/voxel/incremental.rs FIELD-FOR-FIELD (the byte producer of this struct).
struct PackCommand {
    origin_x: i32,                  // brick world-voxel origin x (= coord.x · BRICK_EDGE)
    origin_y: i32,
    origin_z: i32,
    slot: u32,                      // the slot (= primitive_index); meta lands at meta_buf[slot·12]
    world_min_x: f32,               // brick world-min corner
    world_min_y: f32,
    world_min_z: f32,
    index_word_offset: u32,         // start u32 of this brick's index stream in voxel_buf
    lod: u32,                       // brick LOD (bits 0-2 of the packed lod_and_bits)
    index_bits: u32,                // the R2b bit width ∈ {1,2,4,8,16} (the CPU pre-computed it)
    palette_word_offset: u32,       // start u32 of this brick's palette in brick_palettes_buf
    neighbour_base: u32,            // base into `neighbour_indices` for this command's 27-entry table
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// NEIGHBOUR_ABSENT — a `neighbour_indices` entry meaning the neighbour is not resident (halo → AIR). Mirror of
// `NEIGHBOUR_ABSENT` in src/voxel/incremental.rs.
const NEIGHBOUR_ABSENT: u32 = 0xFFFFFFFFu;

// **Stage G-b — one per-CHANGED-slot AABB command** (mirrors `GpuAabbCommand` in src/voxel/incremental.rs
// FIELD-FOR-FIELD). A FLAT 8-u32 (32 B) record — every field a scalar (no `vec3`, whose 16-byte WGSL alignment
// would pad the array stride against the tightly-packed Rust `#[repr(C)]`). `flag = 1` → resident (write the
// epsilon-grown `brick_aabb(world_min, lod)`); `flag = 0` → freed (write `degenerate_aabb()`).
struct AabbCommand {
    slot: u32,                      // the slot (= primitive_index); the AABB lands at aabb_buf[slot·8] (32 B)
    lod: u32,                       // brick LOD (only read when resident; selects the per-LOD span + epsilon)
    flag: u32,                      // 1 = resident (real box), 0 = freed (degenerate box)
    _pad0: u32,
    world_min_x: f32,               // brick world-min corner (only read when resident)
    world_min_y: f32,
    world_min_z: f32,
    _pad1: u32,
}

// --- Stage G-b AABB math — EXACT mirror of src/voxel/gpu.rs (`brick_span`/`brick_aabb_epsilon`/`brick_aabb`) +
//     the WGSL constants in voxel_raytrace.wgsl. MUST agree with both or the seam fix / byte gate breaks. ---
const VOXEL_SIZE: f32 = 0.05;
const BRICK_WORLD_SIZE: f32 = f32(BRICK_EDGE) * VOXEL_SIZE; // = 0.4 m (LOD0 brick span)
const BRICK_AABB_REL_EPS: f32 = 1.25e-4;                   // mirror of gpu.rs::BRICK_AABB_REL_EPS
const MAX_LOD_PACK: u32 = 7u;                              // = brickmap.rs MAX_LOD / voxel_raytrace.wgsl MAX_LOD

// The world-metre SPAN of a brick at LOD `lod`: BRICK_WORLD_SIZE · 2^lod (mirror of gpu.rs/brickmap.rs
// `brick_span`, which clamps `lod.min(MAX_LOD)`). The clamp MUST match (MAX_LOD = 7) or a `lod > 7` brick's box
// would diverge from the CPU `brick_aabb` and fail the byte-equality gate.
fn brick_span(lod: u32) -> f32 {
    return BRICK_WORLD_SIZE * f32(1u << min(lod, MAX_LOD_PACK));
}
// The per-side BLAS-AABB grow (the seam-overlap fudge), in world metres (mirror of gpu.rs::brick_aabb_epsilon).
fn brick_aabb_epsilon(lod: u32) -> f32 {
    return brick_span(lod) * BRICK_AABB_REL_EPS;
}

@group(0) @binding(0) var<storage, read> commands: array<PackCommand>;
// The DEDUPED core pool: each distinct resident brick's `8³` core ONCE (512 u32, voxel_index order). Core `i`'s
// voxel is `cores[i·512 + voxel_index]`.
@group(0) @binding(1) var<storage, read> cores: array<u32>;
// The per-command 27-entry NEIGHBOUR TABLE (`command.neighbour_base + nslot`): a CORE-POOL index into `cores`,
// or NEIGHBOUR_ABSENT. Slot 13 is the command's own brick. This is the dedup indirection that avoids uploading
// each brick once per command that neighbours it.
@group(0) @binding(2) var<storage, read> neighbour_indices: array<u32>;
// The EXISTING pool buffers (bound read_write for this pass): the bit-packed index stream, the per-brick
// palettes, and the 48-B meta directory (as a flat u32 array — meta `slot` lands at `meta_buf[slot·12]`).
@group(0) @binding(3) var<storage, read_write> voxel_buf: array<u32>;
@group(0) @binding(4) var<storage, read_write> brick_palettes_buf: array<u32>;
@group(0) @binding(5) var<storage, read_write> meta_buf: array<u32>;
// Stage G-b — the AABB pass's bindings (its OWN pipeline/bind-group; `pack_brick` ignores these). `aabb_buf` is
// the BLAS-input AABB buffer (8 u32 / 32 B per slot: min[3] @ words 0-2, max[3] @ words 3-5, _pad @ words 6-7 —
// the `#[repr(C)] GpuBrickAabb` layout); `aabb_commands` is the per-changed-slot command list `write_aabb`
// consumes (one invocation each).
@group(0) @binding(6) var<storage, read_write> aabb_buf: array<u32>;
@group(0) @binding(7) var<storage, read> aabb_commands: array<AabbCommand>;
// **Dirty-chunk bitmask (the per-frame AS-rebuild driver).** One BIT per BLAS chunk (a slot-band of `CHUNK_SLOTS`
// slots — `chunk = slot / CHUNK_SLOTS`). `write_aabb` atomically sets the bit for every CHANGED slot's chunk, so
// the CPU reads back this tiny mask (1-frame-late, like `change_count`) and rebuilds ONLY the chunks that actually
// changed this frame — not a blind sweep of the whole (mostly-empty) pool. `CHUNK_SLOTS` MUST match
// `raytrace.rs::CHUNK_SLOTS`. Bound only to the `write_aabb` pipeline (its own bind group).
const CHUNK_SLOTS: u32 = 512u;
@group(0) @binding(10) var<storage, read_write> dirty_chunk: array<atomic<u32>>;
// **Stage G4 — the classify pass output.** One [`ClassifyOut`] (4 u32 / 16 B) per `commands` entry, written by
// `classify_brick` (its OWN pipeline/bind-group — `pack_brick`/`write_aabb` ignore it). The CPU reads this back to
// drive the EXISTING `SlabArena` allocation WITHOUT the CPU `pack_one` (that is the G4 win). Mirrors `GpuClassifyOut`
// in src/voxel/incremental.rs FIELD-FOR-FIELD (4 u32). Bound at its own group(0)/binding(8).
@group(0) @binding(8) var<storage, read_write> classify_out: array<u32>;
// **Stage G4 — the classify command list.** One [`ClassifyCommand`] (4 u32 / 16 B) per dirty brick — its OWN binding
// (NOT the `PackCommand` @0, whose 15-u32/60-B stride differs), so `classify_brick` reads a clean `neighbour_base`
// without the pack-command stride. Mirrors `GpuClassifyCommand` in src/voxel/incremental.rs FIELD-FOR-FIELD.
struct ClassifyCommand {
    neighbour_base: u32,            // base into `neighbour_indices` for this command's 27-entry table
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(9) var<storage, read> classify_commands: array<ClassifyCommand>;

// Workgroup shared state for ONE brick's pack.
// `halo` — the 1000 haloed cells (block ids), in halo_index order (filled in parallel from the cores).
var<workgroup> halo: array<u32, 1000>;
// `local` — per-cell local palette index (filled SERIALLY by invocation 0, first-seen order).
var<workgroup> local_idx: array<u32, 1000>;
// `palette` — the brick's first-seen distinct ids; `palette_len` how many (k). Built serially by invocation 0.
var<workgroup> palette: array<u32, 1000>;
var<workgroup> palette_len: u32;

const WG_SIZE: u32 = 256u;

// Linear cell index in the haloed grid (mirror of `halo_index` / WGSL `cell_index`): +X fastest, then +Y, +Z.
fn halo_index(x: i32, y: i32, z: i32) -> i32 {
    return x + y * HALO_EDGE + z * HALO_EDGE * HALO_EDGE;
}

// Linear voxel index in an 8³ core (mirror of `voxel_index` in src/voxel/brickmap.rs): +X fastest.
fn voxel_index(x: i32, y: i32, z: i32) -> u32 {
    return u32(x + y * BRICK_EDGE + z * BRICK_EDGE * BRICK_EDGE);
}

// Euclidean div/rem on edge 8 for the tiny range cc ∈ [-1, 8] (the halo border) — mirror of the CPU's
// `cc.div_euclid(8)` / `cc.rem_euclid(8)` exactly over that range. cc=-1 → (div -1, rem 7); cc∈[0,7] → (0, cc);
// cc=8 → (1, 0). Branch-explicit so it can never diverge from the CPU integer semantics.
fn nbr_off(cc: i32) -> i32 {
    if (cc < 0) { return -1; }
    if (cc >= BRICK_EDGE) { return 1; }
    return 0;
}
fn nbr_local(cc: i32) -> i32 {
    if (cc < 0) { return BRICK_EDGE - 1; } // -1 → 7
    if (cc >= BRICK_EDGE) { return 0; }    // 8 → 0
    return cc;
}

// The smallest power-of-2 bit width in {1,2,4,8,16} that indexes k ids — mirror of `pow2_index_bits`.
// (Only used as an assert mirror; the CPU passes the already-resolved `index_bits` in the command.)
fn pow2_index_bits(k: u32) -> u32 {
    if (k <= 2u) { return 1u; }
    if (k <= 4u) { return 2u; }
    if (k <= 16u) { return 4u; }
    if (k <= 256u) { return 8u; }
    return 16u;
}

// **The SHARED halo-fill** — reproduce `pack_one`'s dense fill into the workgroup `halo` array: walk the haloed
// grid in halo_index order; a core cell (local coord in [0,8)) reads the centre core; a border cell reads the
// owning same-LOD neighbour core, or AIR (block 0) when that neighbour is absent. Parallel over the 1000 cells
// (pure per-cell function). The SINGLE halo SSOT both `pack_brick` and `classify_brick` call — so the G4 classify
// can NEVER drift from the pack (they fill IDENTICAL cells, by construction). Caller barriers AFTER this.
fn fill_halo(neighbour_base: u32, li: u32) {
    for (var cell = li; cell < HALO_CELLS; cell = cell + WG_SIZE) {
        // Decode the haloed local coord (hx,hy,hz) from the linear `cell` (inverse of halo_index).
        let hz = i32(cell) / (HALO_EDGE * HALO_EDGE);
        let rem = i32(cell) % (HALO_EDGE * HALO_EDGE);
        let hy = rem / HALO_EDGE;
        let hx = rem % HALO_EDGE;
        // Core-local coord (the halo border ring is the -1 / 8 shell). `pack_one` uses cc = h - 1.
        let cx = hx - 1;
        let cy = hy - 1;
        let cz = hz - 1;
        let in_core = (cx >= 0 && cx < BRICK_EDGE) && (cy >= 0 && cy < BRICK_EDGE) && (cz >= 0 && cz < BRICK_EDGE);
        var block: u32 = 0u; // AIR default (BlockId::AIR == 0)
        if (in_core) {
            // The centre core is neighbour slot 13 = (0+1)*9 + (0+1)*3 + (0+1).
            let core = neighbour_indices[neighbour_base + 13u];
            block = cores[core * CORE_CELLS + voxel_index(cx, cy, cz)];
        } else {
            // Resolve the owning same-LOD neighbour (mirror of `neighbour_border_cell`).
            let dx = nbr_off(cx);
            let dy = nbr_off(cy);
            let dz = nbr_off(cz);
            let nslot = u32((dz + 1) * 9 + (dy + 1) * 3 + (dx + 1));
            let core = neighbour_indices[neighbour_base + nslot];
            if (core != NEIGHBOUR_ABSENT) {
                let lx = nbr_local(cx);
                let ly = nbr_local(cy);
                let lz = nbr_local(cz);
                block = cores[core * CORE_CELLS + voxel_index(lx, ly, lz)];
            }
            // else: neighbour absent (or a shell boundary) → AIR, the conservative pre-halo behaviour.
        }
        halo[cell] = block;
    }
}

@compute @workgroup_size(256)
fn pack_brick(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let cmd_idx = wg.x;
    if (cmd_idx >= arrayLength(&commands)) {
        return;
    }
    let cmd = commands[cmd_idx];
    let li = lid.x;

    // (1) HALO FILL — the shared SSOT fill (see `fill_halo`). A pack command is ALWAYS dense (the CPU emits a
    //     pack command only for a dense brick whose voxels changed), so this always proceeds to encode.
    fill_halo(cmd.neighbour_base, li);
    workgroupBarrier();

    // (2) SERIAL PALETTE BUILD — invocation 0 walks all 1000 cells in halo_index order, building the first-seen
    //     palette + the per-cell local-index map. Order-identical to `encode_paletted` by construction (this is
    //     the palette-order mitigation: a parallel build would permute first-seen order → wrong bytes).
    if (li == 0u) {
        var k: u32 = 0u;
        for (var cell = 0u; cell < HALO_CELLS; cell = cell + 1u) {
            let id = halo[cell];
            // Linear scan for `id` in the palette built so far (k is tiny — a strata band is 2–8 ids).
            var found: i32 = -1;
            for (var p = 0u; p < k; p = p + 1u) {
                if (palette[p] == id) {
                    found = i32(p);
                    break;
                }
            }
            if (found < 0) {
                palette[k] = id;
                local_idx[cell] = k;
                k = k + 1u;
            } else {
                local_idx[cell] = u32(found);
            }
        }
        palette_len = k;
    }
    workgroupBarrier();

    // (3) BIT-PACK the local indices into voxel_buf, in parallel. index_bits ∈ {1,2,4,8,16} all divide 32, so a
    //     cell's index never straddles a word boundary — but two cells CAN share a word, so we accumulate per
    //     32-bit WORD (one invocation per word) to avoid a read-modify-write race. Mirror of `encode_paletted`'s
    //     pack loop: `indices[bit/32] |= (idx & mask) << (bit % 32)`.
    let bits = cmd.index_bits;
    let cells_per_word = 32u / bits;
    let total_words = (HALO_CELLS * bits + 31u) / 32u; // = ceil(1000·bits / 32) — same as encode_paletted's `words`
    let mask = select((1u << bits) - 1u, 0xFFFFFFFFu, bits == 32u);
    for (var w = li; w < total_words; w = w + WG_SIZE) {
        var word: u32 = 0u;
        let first_cell = w * cells_per_word;
        for (var j = 0u; j < cells_per_word; j = j + 1u) {
            let cell = first_cell + j;
            if (cell < HALO_CELLS) {
                let idx = local_idx[cell] & mask;
                word = word | (idx << (j * bits));
            }
        }
        voxel_buf[cmd.index_word_offset + w] = word;
    }

    // (4) PALETTE WRITE — the k distinct ids (one u32 each), first-seen order, into brick_palettes_buf.
    let k = palette_len;
    for (var p = li; p < k; p = p + WG_SIZE) {
        brick_palettes_buf[cmd.palette_word_offset + p] = palette[p];
    }

    // (5) META WRITE — the 48-B GpuBrickMeta::dense, as 12 u32 at meta_buf[slot·12]. Field order MUST match
    //     `#[repr(C)] GpuBrickMeta`: voxel_origin[3], voxel_offset, world_min[3], lod_and_bits, palette_base,
    //     flags, _pad[2]. `lod_and_bits = (lod & 7) | ((index_bits & 0x1F) << 3)` (mirror of `pack_lod_bits`).
    if (li == 0u) {
        let base = cmd.slot * 12u;
        meta_buf[base + 0u] = bitcast<u32>(cmd.origin_x);
        meta_buf[base + 1u] = bitcast<u32>(cmd.origin_y);
        meta_buf[base + 2u] = bitcast<u32>(cmd.origin_z);
        meta_buf[base + 3u] = cmd.index_word_offset;                       // voxel_offset
        meta_buf[base + 4u] = bitcast<u32>(cmd.world_min_x);
        meta_buf[base + 5u] = bitcast<u32>(cmd.world_min_y);
        meta_buf[base + 6u] = bitcast<u32>(cmd.world_min_z);
        meta_buf[base + 7u] = (cmd.lod & 0x7u) | ((cmd.index_bits & 0x1Fu) << 3u); // lod_and_bits
        meta_buf[base + 8u] = cmd.palette_word_offset;                     // palette_base
        meta_buf[base + 9u] = 0u;                                          // flags (0 = dense)
        meta_buf[base + 10u] = 0u;                                         // _pad[0]
        meta_buf[base + 11u] = 0u;                                         // _pad[1]
    }
}

// **Stage G-b — the GPU AABB write.** One INVOCATION per changed slot (NOT one workgroup): write `aabb_buf[slot]`
// (8 u32 / 32 B). A RESIDENT slot (`flag == 1`) gets the epsilon-grown `brick_aabb(world_min, lod)` (the SAME box
// `src/voxel/gpu.rs::brick_aabb` builds — `[world_min - eps, world_min + span + eps]`); a FREED slot (`flag == 0`)
// gets `degenerate_aabb()` (min = +1e30, max = -1e30 — a BLAS non-candidate; mirror of incremental.rs). This
// dedicated pass owns EVERY changed slot's box, so the per-slot CPU `queue_write_buffer(aabb)` upload is gone.
// The shared per-slot AABB write (no dirty-chunk side effect — used by BOTH entry points). Factored so the
// `write_aabb` entry (the CPU-pack `apply_gpu_pack` path, which does its own dirty-chunk rebuild) does NOT
// reference the dirty-chunk binding, while `write_aabb_dirty` (the LIVE front end) adds the mask write.
fn write_aabb_slot(i: u32) {
    if (i >= arrayLength(&aabb_commands)) {
        return;
    }
    let c = aabb_commands[i];
    let base = c.slot * 8u; // 32 B = 8 u32 per slot
    if (c.flag == 0u) {
        // degenerate_aabb(): min > max on every axis (mirror of src/voxel/incremental.rs::degenerate_aabb).
        let big = bitcast<u32>(1.0e30);
        let neg = bitcast<u32>(-1.0e30);
        aabb_buf[base + 0u] = big; // min.x
        aabb_buf[base + 1u] = big; // min.y
        aabb_buf[base + 2u] = big; // min.z
        aabb_buf[base + 3u] = neg; // max.x
        aabb_buf[base + 4u] = neg; // max.y
        aabb_buf[base + 5u] = neg; // max.z
    } else {
        let eps = brick_aabb_epsilon(c.lod);
        let span = brick_span(c.lod);
        aabb_buf[base + 0u] = bitcast<u32>(c.world_min_x - eps);        // min.x
        aabb_buf[base + 1u] = bitcast<u32>(c.world_min_y - eps);        // min.y
        aabb_buf[base + 2u] = bitcast<u32>(c.world_min_z - eps);        // min.z
        aabb_buf[base + 3u] = bitcast<u32>(c.world_min_x + span + eps); // max.x
        aabb_buf[base + 4u] = bitcast<u32>(c.world_min_y + span + eps); // max.y
        aabb_buf[base + 5u] = bitcast<u32>(c.world_min_z + span + eps); // max.z
    }
    aabb_buf[base + 6u] = 0u; // _pad[0]
    aabb_buf[base + 7u] = 0u; // _pad[1]
}

// CPU-pack path (`apply_gpu_pack`): plain AABB write — it rebuilds its dirty chunks from the CPU-side batch, so
// it needs no dirty-chunk mask (and its pipeline layout has only bindings 6+7).
@compute @workgroup_size(64)
fn write_aabb(@builtin(global_invocation_id) gid: vec3<u32>) {
    write_aabb_slot(gid.x);
}

// LIVE GPU front end (`GpuResidencyFrontEnd`): same AABB write PLUS mark this slot's BLAS chunk dirty, so the CPU
// rebuilds exactly the changed chunks this frame (resident OR freed — a drop must rebuild its chunk too, else the
// stale AABB lingers / aliases a reused slot ⇒ black square). Its pipeline layout adds binding 10.
@compute @workgroup_size(64)
fn write_aabb_dirty(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&aabb_commands)) {
        let chunk = aabb_commands[i].slot / CHUNK_SLOTS;
        atomicOr(&dirty_chunk[chunk / 32u], 1u << (chunk % 32u));
    }
    write_aabb_slot(i);
}

// **Stage G4 — the GPU CLASSIFY pass.** One WORKGROUP per `commands` entry (the SAME per-dirty-brick command list
// `pack_brick` consumes, but emitted for EVERY dirty key — uniform OR dense — so the GPU classifies them all). It
// does the SAME shared `fill_halo` as `pack_brick`, then computes the per-brick `(is_uniform, uniform_block,
// palette_k, index_bits)` — the CHEAP classification the CPU `SlabArena` allocation needs — WITHOUT bit-packing.
// The CPU reads `classify_out` back and feeds the sizes into the existing `SlabArena` alloc, so the CPU no longer
// runs `pack_one` (the G4 win). Because the sizes are a DETERMINISTIC function of the haloed brick, the GPU
// computes EXACTLY the sizes a CPU `pack_one`+`encode_paletted` would → the pool stays BYTE-IDENTICAL (the parity
// gate is unchanged). `classify_out[cmd_idx·4]` is the 4-u32 `GpuClassifyOut` (mirror of src/voxel/incremental.rs):
//   word 0 = is_uniform (1 = uniform-incl-halo, 0 = dense),
//   word 1 = uniform_block (the single solid id when uniform; 0 when dense),
//   word 2 = palette_k (distinct-id count of the haloed cells; the palette size class — 0 when uniform),
//   word 3 = index_bits ∈ {1,2,4,8,16} (= pow2_index_bits(palette_k); 0 when uniform).
//
// ## Classification math — EXACT mirror of the CPU SSOT (src/voxel/gpu.rs)
// - `uniform_incl_halo_block`: the brick is uniform-incl-halo IFF every one of the 1000 haloed cells equals a
//   SINGLE NON-AIR id. (The CPU requires the CORE to be one solid block AND all halo border cells to equal it;
//   since the core is the 512 interior cells and the border the remaining 488, "all 1000 equal a non-AIR id" is
//   exactly that — a SINGLE robust-by-construction test over the same haloed cells, no separate core/border scan.)
// - `distinct_count` / `encode_paletted` palette size: the count of DISTINCT ids over the 1000 cells (first-seen
//   ORDER is irrelevant to the COUNT — the alloc only needs the size class, not the order; the order is reproduced
//   later by `pack_brick`'s serial palette build, which the byte gate pins).
// - `pow2_index_bits`: the smallest width in {1,2,4,8,16} that indexes `k` ids (the WGSL `pow2_index_bits` above).
//
// Serial within the workgroup (invocation 0) — the same first-seen scan `pack_brick`'s palette build does, reused
// here only to COUNT distinct ids (we discard the per-cell map). `k` is tiny (a strata band is 2–8 ids), so the
// linear scan is cheap; one workgroup per brick keeps it parallel ACROSS bricks.
@compute @workgroup_size(256)
fn classify_brick(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let cmd_idx = wg.x;
    if (cmd_idx >= arrayLength(&classify_commands)) {
        return;
    }
    let cmd = classify_commands[cmd_idx];
    let li = lid.x;

    // (1) HALO FILL — the SAME shared SSOT fill `pack_brick` uses (so classify can never disagree with pack).
    fill_halo(cmd.neighbour_base, li);
    workgroupBarrier();

    // (2) SERIAL CLASSIFY — invocation 0 walks all 1000 cells once: (a) the uniform-incl-halo test (all cells one
    //     non-AIR id), and (b) the first-seen distinct-id COUNT (palette_k). Mirrors the CPU SSOT exactly.
    if (li == 0u) {
        let first = halo[0];
        var all_equal = true;
        // First-seen palette, reused only to COUNT k (order discarded — the alloc needs the size class only).
        var k: u32 = 0u;
        for (var cell = 0u; cell < HALO_CELLS; cell = cell + 1u) {
            let id = halo[cell];
            if (id != first) {
                all_equal = false;
            }
            var found = false;
            for (var p = 0u; p < k; p = p + 1u) {
                if (palette[p] == id) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                palette[k] = id;
                k = k + 1u;
            }
        }
        // Uniform-incl-halo IFF all 1000 cells equal a SINGLE NON-AIR id (BlockId::AIR == 0). Else DENSE.
        let is_uniform = all_equal && (first != 0u);
        let base = cmd_idx * 4u;
        if (is_uniform) {
            classify_out[base + 0u] = 1u;     // is_uniform
            classify_out[base + 1u] = first;  // uniform_block (the single solid id)
            classify_out[base + 2u] = 0u;     // palette_k (unused for uniform)
            classify_out[base + 3u] = 0u;     // index_bits (unused for uniform)
        } else {
            classify_out[base + 0u] = 0u;             // is_uniform = dense
            classify_out[base + 1u] = 0u;             // uniform_block (unused for dense)
            classify_out[base + 2u] = k;              // palette_k (distinct-id count)
            classify_out[base + 3u] = pow2_index_bits(k); // index_bits ∈ {1,2,4,8,16}
        }
    }
}
