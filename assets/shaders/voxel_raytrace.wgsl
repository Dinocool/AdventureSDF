// Hardware-ray-traced voxel raymarch (Stage 2).
//
// A per-pixel primary ray is cast into a TLAS of per-brick procedural AABBs (one AABB per resident
// brick of the brickmap patch). For each AABB candidate the brick's `8³` voxels are DDA-marched in
// world space (read from a GPU storage buffer) to the first SOLID voxel; the brick-local hit distance is
// committed via `rayQueryGenerateIntersection`, so the TLAS resolves the NEAREST brick hit across all
// candidates automatically. The committed hit's first-solid voxel block id → palette colour is written.
//
// This is the "intersection shader = in-shader DDA" pattern (the Teardown-successor approach). The GPU
// buffer layout (AABBs, brick directory, voxels, palette) is the SSOT in `src/voxel/gpu.rs`; this shader
// and the headless test BOTH consume it, so they cannot drift.
//
// Two entry points share one tracer:
//   * `trace_one`  — the headless correctness test: traces ONE ray from a uniform, writes a `Hit`.
//   * `raymarch`   — the render path: one ray per pixel, writes the palette colour to a storage texture.

enable wgpu_ray_query;

// --- SSOT layout (mirrors src/voxel/gpu.rs) ---------------------------------------------------------

// Brick geometry constants (mirror src/voxel/brickmap.rs). A brick is ALWAYS 8³ voxels; only its world span
// scales with LOD (the clipmap). LOD0 = 8³ voxels of 0.05 m → a 0.4 m brick.
const BRICK_EDGE: i32 = 8;
const VOXEL_SIZE: f32 = 0.05;
// World-metre span of a LOD0 brick (= BRICK_EDGE · VOXEL_SIZE = 0.4 m). A LOD-L brick spans
// brick_span(L) = BRICK_WORLD_SIZE · 2^L — see `brick_span` below (the clipmap SSOT mirror of brickmap.rs).
const BRICK_WORLD_SIZE: f32 = f32(BRICK_EDGE) * VOXEL_SIZE;
// A4.2 — the per-side AABB grow (the SEAM fix) is now RELATIVE-PER-LOD: a fixed FRACTION of the brick's
// per-LOD world span, via `brick_aabb_epsilon(lod)` below. MUST mirror `BRICK_AABB_REL_EPS` / `brick_aabb_epsilon`
// in src/voxel/gpu.rs. The shader recomputes the per-brick ray/AABB slab from the GROWN bounds (matching the
// BLAS geometry), so a ray grazing a shared face/edge enters the brick instead of falling in the FP gap between
// abutting AABBs. The DDA still reconstructs cells from the TRUE `world_min` and clamps into the real grid, so
// the halo never adds phantom voxels. The grow scales with the (2^lod×-wider) coarse span, so it bridges the FP
// slab-gap identically at every LOD / voxel size; at LOD0/0.05 m it equals the historical `VOXEL_SIZE·1e-3`.
const BRICK_AABB_REL_EPS: f32 = 1.25e-4;

// Max LOD level (mirrors src/voxel/brickmap.rs MAX_LOD = 7). A brick is 8³ at every LOD; the coarsest level
// spans 2^7 = 128× the LOD0 world extent — the clipmap's outer reach (~8.2 km half-extent at clip_half 160).
const MAX_LOD: u32 = 7u;

// One brick's metadata (parallel to the AABB / BLAS primitive array). 48 bytes — matches `GpuBrickMeta`
// (storage plan R2b grew it from 32 → 48 B to carry the paletted-index decode params). The grid is ALWAYS 8³
// (clipmap); the per-cell world size + the brick SPAN are DERIVED from the LOD, so a coarse brick is DDA-marched
// over the SAME 8³ grid covering 2^lod× more world.
struct BrickMeta {
    voxel_origin: vec3<i32>,  // brick's world-VOXEL origin (= brick_coord · BRICK_EDGE)
    voxel_offset: u32,        // DENSE: start word of this brick's bit-packed INDEX STREAM in `voxel_indices`
                              //   (A4.1: full u32 range — no reserved high bit). UNIFORM (flags bit 0 set): no
                              //   stream — the single block id is the low 16 bits (R1).
    world_min: vec3<f32>,     // brick's world-metre min corner (= coord · brick_span(lod))
    lod_and_bits: u32,        // bits 0-2: brick LOD level (0 = finest). bits 3-7: R2b index bit width ∈ {1,2,4,8,16}.
    palette_base: u32,        // R2b: start word of this brick's palette in `brick_palettes` (dense only)
    flags: u32,               // A4.1: per-brick flag bits. bit 0 (META_FLAG_UNIFORM) marks a UNIFORM brick.
    // Pad to a 48-byte stride (matches Rust `GpuBrickMeta`). SCALAR u32s (align 4) — a `vec3<u32>` here would be
    // align-16 and land at offset 48, blowing the struct to 64 B and silently breaking the field layout.
    _pad1: u32,
    _pad2: u32,
};

// STORAGE PLAN R1 / A4.1 — the `flags` bit marking a UNIFORM brick (its FULL haloed 10³ grid is ONE solid
// block, so it carries NO index stream). MUST mirror `META_FLAG_UNIFORM` in src/voxel/gpu.rs. When set, the DDA
// reads the block id straight from the meta (low 16 bits of `voxel_offset`) — NO buffer fetch per step (strictly
// fewer memory ops); a fully-buried uniform brick is a couple of geometric DDA steps with zero loads. A DENSE
// brick has the bit clear and its `voxel_offset` uses the FULL u32 range (A4.1 retired the bit-31 invariant).
const META_FLAG_UNIFORM: u32 = 1u;

// The brick LOD level (bits 0-2 of `lod_and_bits`). Mirror of `GpuBrickMeta::lod`.
fn meta_lod(m: BrickMeta) -> u32 {
    return m.lod_and_bits & 0x7u;
}

// The R2b paletted INDEX BIT WIDTH ∈ {1,2,4,8,16} (bits 3-7 of `lod_and_bits`). Mirror of
// `GpuBrickMeta::index_bits`. Meaningless for a uniform brick.
fn meta_index_bits(m: BrickMeta) -> u32 {
    return (m.lod_and_bits >> 3u) & 0x1Fu;
}

// True iff brick meta `m` is a collapsed UNIFORM brick (no index stream; block id in the low 16 bits). A4.1:
// reads the dedicated `flags` word — no longer a `voxel_offset` high bit.
fn meta_is_uniform(m: BrickMeta) -> bool {
    return (m.flags & META_FLAG_UNIFORM) != 0u;
}

// The single block id of a UNIFORM brick (low 16 bits of `voxel_offset`). Only valid when `meta_is_uniform`.
fn meta_uniform_block(m: BrickMeta) -> u32 {
    return m.voxel_offset & 0xFFFFu;
}

// The block id at HALOED-grid cell `(x,y,z)` of brick `m` — the SSOT read used by the DDA (storage plan R2b).
// A UNIFORM brick returns its single id with NO memory access (every cell is that block by construction). A
// DENSE brick decodes its bit-packed local palette index then indirects through the per-brick palette:
//   bit  = cell_index · index_bits;   word = voxel_indices[voxel_offset + bit/32]
//   local = (word >> (bit % 32)) & mask;   id = brick_palettes[palette_base + local]
// index_bits ∈ {1,2,4,8,16} all divide 32 ⇒ a cell NEVER straddles a word ⇒ a SINGLE fetch + shift + mask
// (no 2-word path). EXACT mirror of the CPU oracle `decode_paletted_cell` in src/voxel/gpu.rs. Callers pass
// only IN-BOUNDS cells; out-of-bounds is handled at the call site (treated as air for the normal scan).
fn cell_block(m: BrickMeta, x: i32, y: i32, z: i32, hedge: i32) -> u32 {
    if (meta_is_uniform(m)) {
        return meta_uniform_block(m);
    }
    let bits = meta_index_bits(m);
    // STORAGE PLAN A1-β — RAW-ARENA dense brick (the STREAMED path's fixed-block voxel arena). `index_bits == 0`
    // (with bit 31 clear ⇒ not uniform) is the RAW marker: `voxel_indices[voxel_offset + cell_index]` IS the
    // block id directly — one `u32` per cell, NO bit-pack + NO per-brick-palette indirection. The fixed-cap
    // incremental upload (`queue_write_buffer` of only the CHANGED slots) needs a stable per-slot block stride,
    // so the streamed arena stores raw haloed cells; the R2b paletted decode below serves the STATIC
    // (`pack_brickmap`/`pack_resident_set`) path where `index_bits >= 1`. Mirror of the Rust
    // `GpuBrickPatch::cell_block` raw branch. See `SnapshotBuffers` / PHASE_A_GPU_EXECUTION.md §A1-β.
    if (bits == 0u) {
        return voxel_indices[m.voxel_offset + cell_index(x, y, z, hedge)];
    }
    let bit = cell_index(x, y, z, hedge) * bits;
    let word = voxel_indices[m.voxel_offset + bit / 32u];
    let mask = select((1u << bits) - 1u, 0xFFFFFFFFu, bits == 32u);
    let local = (word >> (bit % 32u)) & mask;
    return brick_palettes[m.palette_base + local];
}

// The CORE grid edge (cells per axis) of a brick at LOD `lod`: a CONSTANT BRICK_EDGE (8) at every LOD — the
// clipmap scales the world span, not the resolution. SSOT mirror of `lod_edge` in brickmap.rs.
fn lod_edge(lod: u32) -> i32 {
    return BRICK_EDGE;
}

// The STORED grid edge: the core 8³ grid PLUS a 1-cell HALO border on every side (so 10). The packer fills
// the border with the adjacent SAME-LOD NEIGHBOUR brick's boundary voxels (AIR where absent / a different
// LOD — a clipmap shell boundary), so the DDA always crosses a real air→solid cell boundary AT the true
// surface — even when that surface lies exactly on a brick face. This is the seam fix: it gives the
// first-solid hit the correct entry-face normal (and an always-present boundary cell) from EVERY direction.
// Mirrors `halo_edge` in gpu.rs.
fn halo_edge(lod: u32) -> i32 {
    return lod_edge(lod) + 2;
}

// The world-metre size of one cell of a brick at LOD `lod`: VOXEL_SIZE · 2^lod. Mirror of `lod_voxel_size`.
fn lod_cell_size(lod: u32) -> f32 {
    return VOXEL_SIZE * f32(1u << min(lod, MAX_LOD));
}

// The world-metre SPAN of a brick at LOD `lod`: BRICK_WORLD_SIZE · 2^lod (= BRICK_EDGE · lod_cell_size). The
// clipmap SSOT — mirrors `brick_span` in brickmap.rs / gpu.rs. The brick's true world AABB is
// [world_min, world_min + brick_span(lod)); a coarse brick covers 2^lod× more world.
fn brick_span(lod: u32) -> f32 {
    return BRICK_WORLD_SIZE * f32(1u << min(lod, MAX_LOD));
}

// A4.2 — the per-side BLAS-AABB grow (seam-overlap fudge) for a brick at LOD `lod`, in world metres: a fixed
// FRACTION (BRICK_AABB_REL_EPS) of the brick's per-LOD world span. SSOT mirror of `brick_aabb_epsilon` in
// src/voxel/gpu.rs — the in-shader slab grow MUST equal the CPU AABB grow or the seam fix breaks.
fn brick_aabb_epsilon(lod: u32) -> f32 {
    return brick_span(lod) * BRICK_AABB_REL_EPS;
}

// One palette entry (mirrors `GpuPaletteColor` in src/voxel/gpu.rs): linear-RGBA albedo + linear-RGB
// emissive radiance (`.xyz`; `.w` pad). 32 bytes.
struct Palette { rgba: vec4<f32>, emissive: vec4<f32> };

// A3 — the per-instance DESCRIPTOR (multi-instance TLAS + object-local DDA). EXACT mirror of
// `GpuInstanceDescriptor` in src/voxel/gpu.rs (80 bytes). Indexed by the candidate's `instance_custom_data`
// (the TLAS instance's `custom_index`). `object_from_world`/`world_from_object_rot` are 3×4 affines stored
// row-major as twelve f32 (loaded into a WGSL `mat3x4` = 4 columns × vec3 by `desc_*` below). Descriptor 0 (the
// streamed world) is the IDENTITY degenerate case: both transforms identity, all bases 0, inv_scale 1 ⇒ the hit
// path reduces to the pre-A3 world-space march. See PHASE_A_GPU_EXECUTION.md §A3.
struct InstanceDescriptor {
    // object_from_world (world→object), row-major 3×4 as 12 scalars.
    ofw0: vec4<f32>, ofw1: vec4<f32>, ofw2: vec4<f32>,
    // world_from_object_rot (object→world), row-major 3×4 as 12 scalars.
    wfo0: vec4<f32>, wfo1: vec4<f32>, wfo2: vec4<f32>,
    meta_base: u32,
    voxel_base: u32,
    palette_base: u32,
    inv_scale: f32,
    edit_base: u32,
    mask: u32,
    pad0: u32,
    pad1: u32,
};

// Apply a row-major 3×4 affine (rows r0/r1/r2 = vec4 [a b c | t]) to a POINT: `M · [p, 1]`. For the identity
// transform (descriptor 0) this returns `p` unchanged.
fn affine_point(r0: vec4<f32>, r1: vec4<f32>, r2: vec4<f32>, p: vec3<f32>) -> vec3<f32> {
    let h = vec4<f32>(p, 1.0);
    return vec3<f32>(dot(r0, h), dot(r1, h), dot(r2, h));
}

// Apply the ROTATION (upper 3×3, the .xyz of each row) of a row-major 3×4 affine to a DIRECTION/normal (no
// translation). For the identity transform (descriptor 0) this returns `v` unchanged.
fn affine_dir(r0: vec4<f32>, r1: vec4<f32>, r2: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(r0.xyz, v), dot(r1.xyz, v), dot(r2.xyz, v));
}

@group(0) @binding(0) var acc: acceleration_structure;
// A3 — the per-instance descriptor table (group 0, binding 13). Indexed by `instance_custom_data`. For the
// streamed world it holds ONE identity descriptor 0 (Stage 1) / one per CHUNK (Stage 3), all identity-transform.
@group(0) @binding(13) var<storage, read> descriptors: array<InstanceDescriptor>;
@group(0) @binding(1) var<storage, read> metas: array<BrickMeta>;
// Storage plan R2b — the bit-packed INDEX STREAM (was a raw `u32`-per-cell id buffer). A dense brick's indices
// begin at `metas[].voxel_offset`, are `metas[].index_bits()`-wide, and reference the per-brick palette below.
@group(0) @binding(2) var<storage, read> voxel_indices: array<u32>;
@group(0) @binding(3) var<storage, read> palette: array<Palette>;
// Storage plan R2b — the per-brick PALETTES (concatenated). A dense brick's palette (its `k` distinct block ids,
// one `u32` each) begins at `metas[].palette_base`; `cell_block` indirects `brick_palettes[palette_base + local]`.
@group(0) @binding(12) var<storage, read> brick_palettes: array<u32>;

// --- shared tracer ----------------------------------------------------------------------------------

// Result of tracing one ray: which brick/voxel was hit, its block id + colour, the world hit-t, and the
// outward face normal of the hit voxel (axis the DDA crossed to enter the cell, sign opposing the ray).
struct TraceResult {
    hit: u32,          // 1 if a solid voxel was hit, else 0
    block_id: u32,     // first-solid voxel's BlockId (0 = none)
    prim: u32,         // committed brick primitive_index (== brick directory index)
    t: f32,            // committed world-space hit distance
    color: vec4<f32>,  // palette colour of `block_id`
    normal: vec3<f32>, // outward unit face normal at the hit (0,0,0 on a miss)
    emissive: vec3<f32>, // palette emissive radiance of `block_id` (0 on a miss / non-emitter)
};

// Local cell linear index at grid `edge` (+X fastest, then +Y, then +Z) — MUST match the coarse-edge
// `voxel_index` convention used by `Brick::downsample` / `pack_resident_set` in Rust. At LOD0, `edge` ==
// BRICK_EDGE and this is the original brickmap.rs `voxel_index`.
fn cell_index(x: i32, y: i32, z: i32, edge: i32) -> u32 {
    return u32(x + y * edge + z * edge * edge);
}

// The result of DDA-marching one brick, packed into a `vec4<f32>` (NOT a struct: a struct-returning call used
// inside an `if`/loop branch trips a naga-trunk SSA-scoping bug). Decode with the `dh_*` accessors:
//   .x = found (1.0 if a solid cell was hit, else 0.0)
//   .y = world-t where the ray ENTERED the first solid cell (valid only when found)
//   .z = the first-solid cell's block id, as f32 (exact for ids ≤ 2^24)
//   .w = the entry-face axis index (0=x,1=y,2=z) — the axis the DDA actually CROSSED to step into the cell
// Carrying the crossed axis (not re-deriving the normal from the hit point) is the seam fix: the committed
// candidate (t), the recovered colour (id), and the shading normal all come from ONE walk, so they can never
// disagree at a brick boundary (the old "closest cell-plane" heuristic flipped the normal sideways at voxel
// corners → the thin dark seam line the user saw at oblique angles).
fn dh_found(d: vec4<f32>) -> bool { return d.x > 0.5; }
fn dh_t(d: vec4<f32>) -> f32 { return d.y; }
fn dh_block(d: vec4<f32>) -> u32 { return u32(d.z); }
// The outward face normal, decoded from the packed occupancy-gradient code (`dda_brick` packs
// (gx+1)+(gy+1)*3+(gz+1)*9 in [0,26] into `.w`). Camera-INDEPENDENT: a pure function of the hit cell's
// 6-neighbour occupancy, so it never flips with the view angle. `rd` is unused (kept for call-site stability).
fn dh_normal(d: vec4<f32>, rd: vec3<f32>) -> vec3<f32> {
    let code = i32(d.w);
    let g = vec3<f32>(
        f32((code % 3) - 1),
        f32(((code / 3) % 3) - 1),
        f32((code / 9) - 1),
    );
    return select(vec3<f32>(0.0), normalize(g), d.x > 0.5 && dot(g, g) > 0.0);
}

// DDA-march brick `prim`'s voxels along the world ray (`ro` + t·`rd`, t in [t_enter, t_exit]) to the first
// SOLID voxel, returning the packed (found, t, id, axis). The entry axis is the axis the DDA crossed to ENTER
// the solid cell: for the first cell (no step taken yet) it is the AABB-entry face (the largest-near-slab
// axis); for a later cell it is the axis of the last advance. The SSOT for both the committed intersection
// distance (the candidate uses the t) and the recovered shading data (colour + normal).
fn dda_brick(prim: u32, ro: vec3<f32>, rd: vec3<f32>, t_enter: f32, t_exit: f32) -> vec4<f32> {
    let m = metas[prim];
    let core = lod_edge(meta_lod(m));        // CORE grid cells per axis at this brick's LOD
    let hedge = core + 2;                    // STORED grid edge (core + 1-cell halo border each side)
    let csize = lod_cell_size(meta_lod(m));  // world-metre size of one cell at this brick's LOD
    // The haloed grid's world origin is ONE cell BEFORE the brick min (halo index 0 sits at world_min−csize),
    // so a ray entering the brick first traverses the halo cell that holds the NEIGHBOUR's voxel (AIR where
    // the neighbour is absent). This restores a real air→solid cell crossing AT the true surface even when the
    // surface lies on a brick face — giving the first-solid hit the correct entry-face normal from any angle.
    let gmin = m.world_min - vec3<f32>(csize);
    // Enter the grid a hair past the AABB boundary to land inside the first cell.
    let t0 = max(t_enter, 0.0);
    let p_enter = ro + rd * (t0 + 1e-4);
    // Position in haloed-grid CELL units.
    let local = (p_enter - gmin) / csize;
    var vox = vec3<i32>(floor(local));
    // Clamp the entry cell into [0, hedge) — grazing rays can land a hair outside due to FP.
    vox = clamp(vox, vec3<i32>(0), vec3<i32>(hedge - 1));

    // Standard 3D-DDA setup in world space. Axis-aligned rays have zero direction components; for those the
    // axis never crosses a boundary, so its step is 0 and its t_max/t_delta are +inf (never the minimum), so
    // the DDA never advances or terminates on it. This keeps degenerate rays robust.
    let step = vec3<i32>(sign(rd));
    let inv = 1.0 / rd; // ±inf where rd==0
    // World coordinate of the next cell boundary in each axis (relative to the haloed-grid origin).
    let next_boundary = gmin + (vec3<f32>(vox) + max(vec3<f32>(step), vec3<f32>(0.0))) * csize;
    let big = vec3<f32>(3.4e38);
    let nonzero = abs(rd) > vec3<f32>(1e-12);
    var t_max = select(big, (next_boundary - ro) * inv, nonzero);   // world-t to cross each axis boundary
    let t_delta = select(big, abs(vec3<f32>(csize) * inv), nonzero); // world-t to cross one cell per axis

    // Scalar accumulators with a SINGLE return at the end (naga dislikes returning a mutated `var` struct from
    // multiple in-loop exits). `found` flips true on the first solid CORE cell; `hit_vox` records its cell.
    // `last_axis` is the axis whose boundary the ray CROSSED to enter the current cell — seeded with the
    // AABB-entry face (largest near-slab t) and updated on every DDA advance. It is the face the ray actually
    // struck, which is what gives a cube CRISP PER-FACE normals (each face reads its own normal).
    var found = false;
    var hit_t = -1.0;
    var hit_id = 0u;
    var hit_vox = vec3<i32>(0);
    let tn = min((gmin - ro) * inv, (gmin + vec3<f32>(csize * f32(hedge)) - ro) * inv);
    var last_axis: i32 = 0;
    if (tn.y >= tn.x && tn.y >= tn.z) { last_axis = 1; }
    else if (tn.z >= tn.x && tn.z >= tn.y) { last_axis = 2; }

    // Walk at most the full diagonal of the haloed grid (3·hedge cells is a safe bound).
    var t_cur = t0;
    let lim = 3 * (BRICK_EDGE + 2);
    for (var i = 0; i < lim; i = i + 1) {
        let oob = vox.x < 0 || vox.x >= hedge || vox.y < 0 || vox.y >= hedge || vox.z < 0 || vox.z >= hedge;
        if (oob || found) {
            break;
        }
        // UNIFORM brick (R1): `cell_block` returns the single id with NO buffer fetch — fewer memory ops per
        // step. DENSE brick (R2b): one `voxel_indices[]` fetch + shift/mask + one `brick_palettes[]` indirection.
        let id = cell_block(m, vox.x, vox.y, vox.z, hedge);
        // A solid cell is a HIT only when it is a CORE cell (halo index in [1, core]); a solid HALO cell is the
        // neighbour's voxel — the neighbour brick owns it, so we don't commit it (we only marched it so the
        // surface normal can see across the brick boundary). Continue marching through it.
        let is_core = vox.x >= 1 && vox.x <= core && vox.y >= 1 && vox.y <= core && vox.z >= 1 && vox.z <= core;
        if (id != 0u && is_core) {
            found = true;
            hit_t = t_cur;
            hit_id = id;
            hit_vox = vox;
        } else {
            // Advance to the next cell across the smallest t_max axis; record which axis we crossed.
            if (t_max.x < t_max.y && t_max.x < t_max.z) {
                t_cur = t_max.x; t_max.x = t_max.x + t_delta.x; vox.x = vox.x + step.x; last_axis = 0;
            } else if (t_max.y < t_max.z) {
                t_cur = t_max.y; t_max.y = t_max.y + t_delta.y; vox.y = vox.y + step.y; last_axis = 1;
            } else {
                t_cur = t_max.z; t_max.z = t_max.z + t_delta.z; vox.z = vox.z + step.z; last_axis = 2;
            }
            if (t_cur > t_exit) {
                break; // left the brick before hitting anything solid
            }
        }
    }

    // Outward FACE NORMAL = the face the ray ENTERED the hit cell through (the crossed axis), so each face of a
    // cube gets its OWN crisp normal and it is camera-INDEPENDENT per face (no whole-flat-face "normal swap").
    // BUT a grazing ray skimming a FLAT surface enters the surface voxel through a BURIED side face (the
    // surface continues there), which would read a sideways normal = the dark brick-seam line. So: use the
    // crossed face ONLY when it is EXPOSED (its incoming-side neighbour is air); otherwise fall back to the
    // (single, for a flat surface) exposed face, taken in a fixed axis order so it stays camera-independent.
    // The halo carries the neighbours, so a core cell's face-neighbours are all in-bounds. Packed as a
    // single-axis unit `grad` and decoded+normalized in `dh_normal`.
    var grad = vec3<i32>(0);
    var cnb = hit_vox; cnb[last_axis] = cnb[last_axis] - step[last_axis]; // neighbour the ray came from
    let crossed_air = cnb[last_axis] < 0 || cnb[last_axis] >= hedge
        || cell_block(m, cnb.x, cnb.y, cnb.z, hedge) == 0u;
    if (crossed_air) {
        grad[last_axis] = -step[last_axis]; // crisp crossed-axis face (outward = back along the ray)
    } else {
        // Grazing into a flat/buried surface → take the first EXPOSED face (a flat surface has exactly one).
        for (var a = 0; a < 3; a = a + 1) {
            if (grad.x == 0 && grad.y == 0 && grad.z == 0) {
                var pn = hit_vox; pn[a] = pn[a] + 1;
                var mn = hit_vox; mn[a] = mn[a] - 1;
                let p_air = pn[a] >= hedge || cell_block(m, pn.x, pn.y, pn.z, hedge) == 0u;
                let m_air = mn[a] < 0 || cell_block(m, mn.x, mn.y, mn.z, hedge) == 0u;
                if (p_air) { grad[a] = 1; } else if (m_air) { grad[a] = -1; }
            }
        }
    }
    let code = (grad.x + 1) + (grad.y + 1) * 3 + (grad.z + 1) * 9;

    return vec4<f32>(select(0.0, 1.0, found), hit_t, f32(hit_id), f32(code));
}

// Re-DDA the COMMITTED brick to recover the first-solid voxel's block id AND its entry-face axis together (the
// ray query only carries primitive_index + t across the commit, not the voxel id/normal, so we re-walk the
// winning brick). The re-walk reproduces the candidate-pass `dda_brick` EXACTLY — the SAME grown-AABB slab
// (`±brick_aabb_epsilon(lod)`) and the same march — so the recovered block id + normal are guaranteed to be those
// of the cell the candidate committed (no boundary drift, no sideways-normal seam). `dda_brick` clamps the
// entry cell into `[0, edge)`, so the grown halo never adds a phantom cell.
fn brick_hit_at(prim: u32, ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
    let m = metas[prim];
    let span = brick_span(meta_lod(m)); // clipmap: a LOD-L brick spans 2^L× more world (NOT LOD-invariant)
    let eps = brick_aabb_epsilon(meta_lod(m)); // A4.2: relative-per-LOD grow (mirror of the CPU AABB)
    let bmin = m.world_min - vec3<f32>(eps);
    let bmax = m.world_min + vec3<f32>(span + eps);
    let inv = 1.0 / rd;
    let ta = (bmin - ro) * inv;
    let tb = (bmax - ro) * inv;
    let t_enter = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z));
    let t_exit = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
    return dda_brick(prim, ro, rd, t_enter, t_exit);
}

// Trace one world ray through the brick TLAS, DDA-marching each AABB candidate. Returns the nearest solid
// voxel hit (block id + palette colour) or `hit = 0`.
fn trace(ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32) -> TraceResult {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc, RayDesc(0u, 0xFFu, t_min, t_max, ro, rd));
    // Track the nearest VOXEL hit OURSELVES across the candidate bricks, rather than trusting the ray
    // query's committed intersection. On the wgpu-trunk fork, `rayQueryGetCommittedIntersection().t` for a
    // procedural-AABB hit is the AABB-ENTRY distance, not the per-voxel `t` we pass to
    // `rayQueryGenerateIntersection` — so a brick the ray enters early but only hits solid DEEP inside would
    // win at a too-near depth, painting its face THROUGH nearer geometry (the "show-through" bug; the
    // logic-equivalent CPU oracle in tests/voxel_show_through.rs has zero such hits). We still call
    // `generateIntersection` so the ray query culls far candidates, but the committed t/prim are NOT used.
    var best_t: f32 = t_max * 2.0;
    var best_prim: u32 = 0xffffffffu;
    var best_inst: u32 = 0u; // A3: the winning candidate's descriptor index (for the object-space re-walk)
    loop {
        if (!rayQueryProceed(&rq)) { break; }
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind == RAY_QUERY_INTERSECTION_AABB) {
            // A3 — resolve the candidate's INSTANCE DESCRIPTOR (the TLAS instance's custom_index). For the
            // streamed world this is descriptor 0/per-chunk: identity transform, meta_base = the chunk's slot
            // base. Transform the WORLD ray into OBJECT space (identity ⇒ ro_l==ro, rd_l==rd for the world) and
            // resolve the GLOBAL brick index `prim = meta_base + primitive_index` (== primitive_index for the
            // world). The DDA + slab then run in object space; the committed t is converted back to world.
            let d = descriptors[c.instance_custom_data];
            let ro_l = affine_point(d.ofw0, d.ofw1, d.ofw2, ro);
            let rd_l = affine_dir(d.ofw0, d.ofw1, d.ofw2, rd);
            let prim = d.meta_base + c.primitive_index;
            // The candidate carries the brick's primitive_index but NOT its t-range, so re-derive the
            // ray/AABB entry & exit from the brick bounds (in OBJECT space), then DDA between them.
            let m = metas[prim];
            // Slab against the GROWN brick bounds (same overlap as the BLAS AABB — the seam fix). Using the
            // grown bounds keeps a face-grazing axis-parallel ray off the exact tangent plane (where the true
            // bounds give a 0·inf = NaN slab t), so it reliably enters the brick. `brick_span(meta_lod(m))` is the
            // clipmap span — a coarse brick covers 2^lod× more world (the AABB is NOT LOD-invariant).
            let span = brick_span(meta_lod(m));
            let eps = brick_aabb_epsilon(meta_lod(m)); // A4.2: relative-per-LOD grow
            let bmin = m.world_min - vec3<f32>(eps);
            let bmax = m.world_min + vec3<f32>(span + eps);
            let inv = 1.0 / rd_l;
            let ta = (bmin - ro_l) * inv;
            let tb = (bmax - ro_l) * inv;
            let t_enter = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z));
            let t_exit  = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
            // The slab t-range is in OBJECT space; the t_min/t_exit gate is against the WORLD t (× inv_scale).
            if (t_enter <= t_exit && t_exit * d.inv_scale >= t_min) {
                let bh = dda_brick(prim, ro_l, rd_l, t_enter, t_exit);
                if (dh_found(bh)) {
                    let ht = dh_t(bh) * d.inv_scale; // local-t → WORLD-t for the cross-instance nearest compare
                    if (ht < best_t) {
                        best_t = ht;
                        best_prim = prim;
                        best_inst = c.instance_custom_data;
                    }
                    rayQueryGenerateIntersection(&rq, ht);
                }
            }
        }
    }
    var r: TraceResult;
    let has_hit = best_prim != 0xffffffffu;
    // Re-walk the winning brick ONCE to recover the first-solid cell's id AND its entry-face normal from the
    // SAME DDA — so the colour and the shading normal are always the committed cell's (no boundary drift /
    // sideways-normal seam). Called with a safe index on a miss; gated by `has_hit`. A3: re-derive the winning
    // descriptor + transform the ray into its object space; the recovered local normal is rotated back to world.
    let prim = select(0u, best_prim, has_hit);
    let dwin = descriptors[best_inst];
    let ro_w = affine_point(dwin.ofw0, dwin.ofw1, dwin.ofw2, ro);
    let rd_w = affine_dir(dwin.ofw0, dwin.ofw1, dwin.ofw2, rd);
    let bh = brick_hit_at(prim, ro_w, rd_w);
    let id = select(0u, dh_block(bh), has_hit);
    if (has_hit) {
        r.hit = select(0u, 1u, id != 0u);
        r.block_id = id;
        r.prim = best_prim;
        r.t = best_t;
        r.color = palette[id].rgba;
        r.emissive = palette[id].emissive.xyz;
        // Rotate the object-local entry-face normal back to WORLD via the descriptor's object→world ROTATION.
        // For the streamed world (identity) `affine_dir` returns the input components UNCHANGED (dot with the
        // identity rows), so descriptor 0 is bit-identical to the pre-A3 `dh_normal(bh, rd)`. NO outer
        // `normalize` (it would inject a sqrt/divide that could perturb the byte-identical world render); a
        // PURE-ROTATION `world_from_object_rot` preserves the already-unit `dh_normal`, and a future scaled prop
        // would carry a normalized rotation so the result stays unit.
        r.normal = affine_dir(dwin.wfo0, dwin.wfo1, dwin.wfo2, dh_normal(bh, rd_w));
    } else {
        r.hit = 0u;
        r.block_id = 0u;
        r.prim = 0xffffffffu;
        r.t = -1.0;
        r.color = vec4<f32>(0.0);
        r.emissive = vec3<f32>(0.0);
        r.normal = vec3<f32>(0.0);
    }
    return r;
}

// --- occlusion (shadow / AO) tracing ----------------------------------------------------------------

// Trace an OCCLUSION ray: does any solid voxel lie along (`ro` + t·`rd`) within [t_min, t_max]? Returns
// `true` if occluded (a solid voxel was committed before t_max), `false` if the ray reaches t_max in air.
// Reuses the SAME DDA-in-AABB intersection as the primary tracer (so shadows/AO see exactly the geometry
// the camera sees). This is the visibility query for direct sun shadows and ambient occlusion — it only
// needs a boolean, so it stops at the first committed hit (the TLAS already returns the nearest).
fn trace_occluded(ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32) -> bool {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc, RayDesc(0u, 0xFFu, t_min, t_max, ro, rd));
    loop {
        if (!rayQueryProceed(&rq)) { break; }
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind == RAY_QUERY_INTERSECTION_AABB) {
            // A3 — resolve the candidate's descriptor + transform the ray into object space (identity for the
            // streamed world ⇒ unchanged) + the GLOBAL brick index. Same as `trace` but boolean-only.
            let d = descriptors[c.instance_custom_data];
            let ro_l = affine_point(d.ofw0, d.ofw1, d.ofw2, ro);
            let rd_l = affine_dir(d.ofw0, d.ofw1, d.ofw2, rd);
            let prim = d.meta_base + c.primitive_index;
            let m = metas[prim];
            // Grown-bounds slab (matches the BLAS AABB overlap — the seam fix; see `trace`). `brick_span(meta_lod(m))`
            // is the clipmap span (a coarse brick covers 2^lod× more world).
            let span = brick_span(meta_lod(m));
            let eps = brick_aabb_epsilon(meta_lod(m)); // A4.2: relative-per-LOD grow
            let bmin = m.world_min - vec3<f32>(eps);
            let bmax = m.world_min + vec3<f32>(span + eps);
            let inv = 1.0 / rd_l;
            let ta = (bmin - ro_l) * inv;
            let tb = (bmax - ro_l) * inv;
            let t_enter = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z));
            let t_exit  = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
            if (t_enter <= t_exit && t_exit * d.inv_scale >= t_min) {
                let bh = dda_brick(prim, ro_l, rd_l, t_enter, t_exit);
                let world_t = dh_t(bh) * d.inv_scale; // local-t → WORLD-t for the t_max occlusion gate
                if (dh_found(bh) && world_t <= t_max) {
                    rayQueryGenerateIntersection(&rq, world_t);
                }
            }
        }
    }
    let committed = rayQueryGetCommittedIntersection(&rq);
    return committed.kind != RAY_QUERY_INTERSECTION_NONE;
}

// --- test entry point: one ray from a uniform → a Hit buffer ----------------------------------------

struct RayUniform {
    origin: vec3<f32>,
    t_min: f32,
    dir: vec3<f32>,
    t_max: f32,
};
struct Hit {
    hit: u32,
    block_id: u32,
    prim: u32,
    t: f32,
    color: vec4<f32>,
    normal: vec3<f32>,  // outward face normal at the hit (lighting oracle)
    shadowed: u32,      // 1 if the sun shadow ray from the hit is occluded, else 0
    direct: vec3<f32>,  // direct-lit colour at the hit (albedo × (ambient·ao + sun·N·shadow)) — GI oracle
    _p0: u32,
    indirect: vec3<f32>,// single-bounce indirect IRRADIANCE × albedo at the hit (GI colour-bleed oracle)
    _p1: u32,
    emissive_out: vec3<f32>, // this block's own emissive glow (palette emissive × emissive_strength)
    _p2: u32,
};
@group(0) @binding(4) var<uniform> ray: RayUniform;
@group(0) @binding(5) var<storage, read_write> out_hit: Hit;

@compute @workgroup_size(1)
fn trace_one() {
    let r = trace(ray.origin, ray.dir, ray.t_min, ray.t_max);
    out_hit.hit = r.hit;
    out_hit.block_id = r.block_id;
    out_hit.prim = r.prim;
    out_hit.t = r.t;
    out_hit.color = r.color;
    out_hit.normal = r.normal;
    // Trace the sun shadow ray from the hit (normal-offset) toward the sun, exactly as `shade` does, so the
    // lighting oracle can assert which ground points are in shadow. On a miss, report unshadowed.
    var shadowed = 0u;
    var direct = vec3<f32>(0.0);
    var indirect = vec3<f32>(0.0);
    var emissive_out = vec3<f32>(0.0);
    if (r.hit != 0u) {
        let p = ray.origin + ray.dir * r.t;
        let origin = p + r.normal * light.shadow_bias;
        let to_sun = -light.sun_direction;
        if (trace_occluded(origin, to_sun, 0.0, 1.0e4)) {
            shadowed = 1u;
        }
        // Direct term (shadowed sun only — NO flat ambient, matching the live `shade`), the indirect
        // single-bounce GI (× albedo), and the surface's own emissive glow — separated so the GI oracle can
        // assert each independently. A fixed seed keeps the headless GI estimate reproducible.
        let ndotl = max(dot(r.normal, to_sun), 0.0);
        var shadow = 1.0;
        if (shadowed == 1u) { shadow = 0.0; }
        direct = r.color.rgb * (light.sun_color * (light.sun_intensity * ndotl * shadow));
        indirect = gather_gi(r.normal, p, 12345u) * r.color.rgb;
        emissive_out = r.emissive * light.emissive_strength;
    }
    out_hit.shadowed = shadowed;
    out_hit.direct = direct;
    out_hit.indirect = indirect;
    out_hit.emissive_out = emissive_out;
}

// --- batched test entry point: an ARRAY of rays → an array of Hits (one dispatch) -------------------
// A throughput hook for the seam ORACLE: trace many rays in a single dispatch (the per-ray `trace_one`
// round-trips the GPU once per ray, far too slow to sweep millions of grazing rays). Only the fields the
// seam test asserts on are written (hit / block_id / colour / normal / t / prim); the lighting fields are
// left zero. Uses its own bindings (6 = ray array in, 7 = hit array out) so it coexists with `trace_one`.
@group(0) @binding(6) var<storage, read> rays_in: array<RayUniform>;
@group(0) @binding(7) var<storage, read_write> hits_out: array<Hit>;

@compute @workgroup_size(64)
fn trace_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&rays_in)) { return; }
    let rin = rays_in[i];
    let r = trace(rin.origin, rin.dir, rin.t_min, rin.t_max);
    var h: Hit;
    h.hit = r.hit;
    h.block_id = r.block_id;
    h.prim = r.prim;
    h.t = r.t;
    h.color = r.color;
    h.normal = r.normal;
    h.shadowed = 0u;
    h.direct = vec3<f32>(0.0);
    h._p0 = 0u;
    h.indirect = vec3<f32>(0.0);
    h._p1 = 0u;
    h.emissive_out = vec3<f32>(0.0);
    h._p2 = 0u;
    hits_out[i] = h;
}

// --- render entry point: one ray per pixel → a storage texture --------------------------------------

// Camera + viewport for primary-ray generation. `world_from_clip` unprojects NDC corners; `cam_pos` is the
// ray origin. 80 bytes.
struct CameraUniform {
    world_from_clip: mat4x4<f32>,
    cam_pos: vec3<f32>,
    t_max: f32,
    viewport: vec2<u32>,
    // x: TEMPORAL-ACCUMULATION blend weight in (0,1] — the fraction of the NEW frame mixed into history this
    //    frame (running mean: 1/sample_count). `accum_weight == 1.0` means "no history" (camera moved / first
    //    frame → reset). The renderer drives it: 1.0 on a camera move, then 1/n while the camera holds still,
    //    so a static Cornell view converges to a clean mean over n frames.
    // y: reserved pad.
    accum_weight: f32,
    _pad: u32,
    // Previous-frame UN-jittered clip_from_world for the non-DLSS ReSTIR temporal reprojection
    // (`reproject_pixel`). The non-DLSS path is not jittered, so the current clip IS un-jittered; the renderer
    // stores it each frame and feeds last frame's here. First frame: prev == cur (self-tap). The DLSS path
    // fills this for layout parity but reprojects via `dlss_cam.motion_prev` instead.
    prev_clip_from_world: mat4x4<f32>,
};
@group(1) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(1) var out_tex: texture_storage_2d<rgba16float, write>;
// Temporal-accumulation HISTORY (the previous frame's accumulated result). Sampled (nearest) so the blend
// reuses the same pixel. Bound to a 1×1 dummy on the reset frame; `accum_weight == 1.0` ignores it anyway.
@group(1) @binding(3) var history_tex: texture_2d<f32>;
@group(1) @binding(4) var history_sampler: sampler;

// Lighting + GI knobs (SSOT mirror of `LightingUniformData` in src/voxel/raytrace.rs). All runtime
// uniforms (knobs-as-uniforms), never WGSL consts. 80 bytes:
//   sun_direction (12) + sun_intensity (4)
//   sun_color     (12) + shadow_bias    (4)   — bias = normal-offset epsilon for the shadow/AO ray origin
//   ambient_color (12) + ao_radius      (4)   — ao_radius = AO ray length in world metres
//   ao_samples    (4)  + _pad0 (4) + gi_intensity (4) + gi_bounce_dist (4)
//   emissive_strength (4) + frame_index (4) + debug_view (4) + _pad (4)
struct LightingUniform {
    sun_direction: vec3<f32>,  // normalized direction the sunlight travels (points away from the sun)
    sun_intensity: f32,        // scalar multiplier on sun_color
    sun_color: vec3<f32>,      // linear RGB of the sun
    shadow_bias: f32,          // world-metre normal offset for shadow/AO ray origins (avoids self-hit)
    ambient_color: vec3<f32>,  // linear RGB sky/ambient fill
    ao_radius: f32,            // AO ray length in world metres
    ao_samples: u32,           // number of AO rays in the hemisphere (0 disables AO → ao = 1)
    _pad0: u32,                // was gi_rays (removed — ReSTIR's correct initial count is always 1); keeps the 80 B layout
    gi_intensity: f32,         // scalar multiplier on accumulated indirect irradiance
    gi_bounce_dist: f32,       // max world-metre length of a diffuse bounce ray (miss past it = sky)
    emissive_strength: f32,    // scalar multiplier on every block's palette emissive
    frame_index: u32,          // per-frame counter to decorrelate the bounce-direction hash
    debug_view: u32,           // 0 = lit; 1 = normals; 2 = depth; 3 = albedo; 4 = AO; 5 = GI-only; 6 = face-toward-camera
    _pad: f32,                 // was gi_firefly_clamp (firefly clamping discarded in 2.2 — best-practice); keeps the struct exactly 80 B
};
@group(1) @binding(2) var<uniform> light: LightingUniform;

// Procedural-sky SSOT (mirror of `SkyUniformData` in src/voxel/raytrace.rs, group 1 binding 11). A SEPARATE
// UBO — `LightingUniform` is full at 80 bytes, so the sky/environment knobs live here. All runtime uniforms
// (knobs-as-uniforms), never WGSL consts. 64 bytes (std140-safe, four 16-byte rows):
//   horizon_color (12) + intensity        (4)
//   zenith_color  (12) + gi_sky_intensity (4)
//   ground_color  (12) + sun_size         (4)
//   sun_tint      (12) + _pad             (4)
struct Sky {
    horizon_color: vec3<f32>,   // linear RGB at the horizon (dir.y == 0)
    intensity: f32,             // scalar multiplier on ALL sky radiance (gradient + sun disk)
    zenith_color: vec3<f32>,    // linear RGB straight up (dir.y == +1)
    gi_sky_intensity: f32,      // how strongly a bounce that escapes to sky lights GI (× sky_radiance)
    ground_color: vec3<f32>,    // linear RGB straight down (dir.y == -1) — the lower hemisphere fill
    sun_size: f32,              // angular HALF-size of the soft sun disk, in radians (0 disables the disk)
    sun_tint: vec3<f32>,        // linear RGB tint on the sun disk (multiplied by light.sun_color)
    _pad: f32,
};
@group(1) @binding(11) var<uniform> sky: Sky;

// THE single source of sky/environment radiance for a ray travelling in direction `dir`. A directional
// gradient (ground → horizon → zenith keyed off dir.y) plus a soft sun disk toward the sun, scaled by
// `sky.intensity`. Used by EVERY primary miss, by `bounce_sky` (GI bounce miss), and by the ReSTIR
// bounce-miss sky sample — so the look is defined ONCE (no inline-duplicated gradients).
fn sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    // Upper hemisphere: horizon→zenith by the up component; lower hemisphere: horizon→ground. `dir.y` in
    // [-1,1]; remap so 0 = horizon, +1 = zenith, -1 = ground.
    let up = clamp(dir.y, -1.0, 1.0);
    var grad: vec3<f32>;
    if (up >= 0.0) {
        grad = mix(sky.horizon_color, sky.zenith_color, up);
    } else {
        grad = mix(sky.horizon_color, sky.ground_color, -up);
    }
    // Soft sun disk toward the sun (the direction TOWARD the sun is the opposite of where the light travels).
    // `cos_to_sun` is the cosine of the angle between the view ray and the sun; smoothstep gives a soft edge
    // over `sun_size` radians. Tinted by `sun_tint × light.sun_color × sun_intensity`.
    var sun = vec3<f32>(0.0);
    if (sky.sun_size > 0.0) {
        let to_sun = -light.sun_direction;
        let cos_to_sun = dot(normalize(dir), to_sun);
        let cos_edge = cos(sky.sun_size);
        let disk = smoothstep(cos_edge, mix(cos_edge, 1.0, 0.5), cos_to_sun);
        sun = disk * sky.sun_tint * light.sun_color * light.sun_intensity;
    }
    return (grad + sun) * sky.intensity;
}

// The sky radiance a GI BOUNCE sees on a miss — the gradient ONLY, sun DISK EXCLUDED (the directional sun's GI
// is delivered through the cache's `wc_sun_direct`, so a GI bounce toward the sun must not double-count the
// disk). The PRIMARY camera miss still uses the full `sky_radiance`; only GI bounces use this.
fn sky_radiance_no_sun(dir: vec3<f32>) -> vec3<f32> {
    let up = clamp(dir.y, -1.0, 1.0);
    if (up >= 0.0) {
        return mix(sky.horizon_color, sky.zenith_color, up) * sky.intensity;
    }
    return mix(sky.horizon_color, sky.ground_color, -up) * sky.intensity;
}

// Build an orthonormal basis (tangent, bitangent) around unit normal `n` (Frisvad / Duff branchless).
fn onb(n: vec3<f32>) -> mat2x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return mat2x3<f32>(t, bt);
}

// Ambient-occlusion fraction in [0,1]: trace `light.ao_samples` short rays into the hemisphere around `n`
// from `p` (already normal-offset). Fixed cosine-weighted-ish directions from a small deterministic set so
// the result is stable (no per-frame noise this increment). Returns the UNOCCLUDED fraction (1 = fully
// open). `ao_samples == 0` short-circuits to 1.
fn ambient_occlusion(p: vec3<f32>, n: vec3<f32>) -> f32 {
    let samples = min(light.ao_samples, 8u);
    if (samples == 0u || light.ao_radius <= 0.0) {
        return 1.0;
    }
    let basis = onb(n);
    let tang = basis[0];
    let bitang = basis[1];
    // Eight fixed hemisphere directions: one straight up the normal + a tilted ring. Cosine-ish — tilted
    // ~45° off the normal so they sample the visible hemisphere without grazing the surface.
    var open = 0.0;
    let ring = 0.7071; // sin/cos 45°
    for (var i = 0u; i < samples; i = i + 1u) {
        var dir: vec3<f32>;
        if (i == 0u) {
            dir = n;
        } else {
            let ang = 6.2831853 * f32(i - 1u) / f32(max(samples - 1u, 1u));
            let tan_dir = tang * cos(ang) + bitang * sin(ang);
            dir = normalize(n * ring + tan_dir * ring);
        }
        if (!trace_occluded(p, dir, 0.0, light.ao_radius)) {
            open = open + 1.0;
        }
    }
    return open / f32(samples);
}

// Direct lighting at a surface hit WITHOUT ambient occlusion: Lambert sun term gated by a traced hard
// shadow, plus the flat ambient/sky fill. `albedo` is the voxel palette colour, `n` the face normal, `p`
// the world hit point. This is the reusable "what light does this surface reflect" term — the primary hit
// applies it (with AO) AND each GI bounce hit re-evaluates it so a bounce sees the lit-vs-shadowed surface.
// Output is LINEAR HDR.
fn direct_lighting(albedo: vec3<f32>, n: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    // Offset the secondary-ray origin off the surface along the normal to avoid self-intersection.
    let origin = p + n * light.shadow_bias;
    // Direction TOWARD the sun is the opposite of the direction the light travels.
    let to_sun = -light.sun_direction;
    let ndotl = max(dot(n, to_sun), 0.0);
    // Hard sun shadow: occlusion ray toward the sun. Skip the trace where the face points away (ndotl==0).
    var shadow = 1.0;
    if (ndotl > 0.0) {
        if (trace_occluded(origin, to_sun, 0.0, 1.0e4)) {
            shadow = 0.0;
        }
    }
    let direct = light.sun_color * (light.sun_intensity * ndotl * shadow);
    // No flat ambient_color: the fill light at this (bounce-)hit comes from the sun + the surface's own
    // emissive + further bounces, matching Solari. Adding a flat ambient here double-counted the indirect/sky
    // term AND polluted the world cache (this feeds world_cache_update + gather_gi bounce hits → compounded
    // across bounces). Occluded points correctly go dark.
    return albedo * direct;
}

// The sky radiance a MISSED diffuse bounce returns (a bounce that escapes to open sky), scaled by
// `gi_sky_intensity`. ONE source of sky radiance (`sky_radiance`) shared with the primary-miss sky, so a
// bounce into the sky brings back exactly the directional sky it would see — open-world GI now gets sky fill
// instead of the old flat `ambient_color`. `gi_sky_intensity` is the knob for how strongly the sky lights GI.
fn bounce_sky(dir: vec3<f32>) -> vec3<f32> {
    return sky_radiance(dir) * sky.gi_sky_intensity;
}

// A 2D → 1D hash (PCG-ish) for cheap per-pixel+frame jitter of the bounce directions. Deterministic given
// the pixel and frame, so the headless oracle is reproducible; it varies per frame in the render path so
// the noise pattern animates (and a future temporal accumulator can average it out).
fn hash_u32(seed: u32) -> u32 {
    var x = seed;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn rand01(seed: u32) -> f32 {
    return f32(hash_u32(seed)) * (1.0 / 4294967296.0);
}

// Van der Corput radical inverse in base 2 (bit-reversal) — the y coordinate of a Hammersley point set.
// Paired with i/N for x, it gives a LOW-DISCREPANCY 2D set: far more uniform hemisphere coverage than
// white noise for the same sample count, so the per-pixel GI estimate has much lower variance (less boil).
fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10; // / 2^32
}

// Single-bounce diffuse GLOBAL ILLUMINATION at a surface hit. Cosine-sample a fixed (high-spp) set of directions in the
// hemisphere about the face normal `n` (Frisvad ONB + concentric cosine mapping), trace each as a bounce
// ray on the SAME TLAS, and gather incoming radiance:
//   * bounce HIT  → that surface's direct lighting (albedo × (ambient + sun·N'·shadow)) PLUS its emissive
//                   (palette emissive × emissive_strength). So a glowing block lights its neighbours.
//   * bounce MISS → the sky/ambient radiance.
// The cosine pdf cancels the Lambert cosine, so the estimator is the plain MEAN of the gathered radiance
// (already the irradiance/π × π for a Lambertian). Returns the indirect IRRADIANCE colour (NOT yet ×albedo
// of the receiving surface — the caller multiplies by the receiver albedo). `seed_base` decorrelates the
// per-ray hash. Structured so a ReSTIR reservoir can later replace the plain mean without touching callers.
fn gather_gi(n: vec3<f32>, p: vec3<f32>, seed_base: u32) -> vec3<f32> {
    // High-spp forward cosine-mean: this is the LEGACY non-ReSTIR forward GI estimator used by the debug
    // GI-only view and the headless probe ORACLE (the low-variance reference the ReSTIR estimator is asserted
    // to converge to). The live ReSTIR path (`restir_p1_core`) does NOT use this. The sample count is a local
    // const (the `gi_rays` uniform was removed — ReSTIR's correct initial count is always 1); a high fixed
    // count keeps the oracle reference low-variance.
    let rays = 32u;
    let origin = p + n * light.shadow_bias;
    let basis = onb(n);
    let tang = basis[0];
    let bitang = basis[1];
    // Per-pixel + per-frame Cranley–Patterson rotation: shift the whole Hammersley set by a random toroidal
    // offset. Within a pixel the N samples stay LOW-DISCREPANCY (well-stratified hemisphere coverage); across
    // pixels and frames the offset decorrelates, so the residual noise ANIMATES uniformly — exactly the
    // blue-noise-like input DLSS-RR (and the temporal accumulator) reproject + average best. Far less boil
    // than independent white-noise directions, at the same ray count.
    let rot = vec2<f32>(rand01(seed_base * 2u + 1u), rand01(seed_base * 2u + 2u));
    var acc_rad = vec3<f32>(0.0);
    for (var i = 0u; i < rays; i = i + 1u) {
        // Hammersley point i of `rays`, rotated. (x → azimuth, y → cosine radius.)
        let u1 = fract(f32(i) / f32(rays) + rot.x);
        let u2 = fract(radical_inverse_vdc(i) + rot.y);
        // Cosine-weighted hemisphere sample (Malley / concentric-ish): r = sqrt(u1), phi = 2π·u2; z = sqrt(1-u1).
        let r = sqrt(u1);
        let phi = 6.2831853 * u2;
        let x = r * cos(phi);
        let y = r * sin(phi);
        let z = sqrt(max(0.0, 1.0 - u1));
        let dir = normalize(tang * x + bitang * y + n * z);
        // Trace the bounce ray (bounded to gi_bounce_dist) and gather incoming radiance.
        let h = trace(origin, dir, 0.0, light.gi_bounce_dist);
        var contrib: vec3<f32>;
        if (h.hit != 0u) {
            let hp = origin + dir * h.t;
            let surf = direct_lighting(h.color.rgb, h.normal, hp);
            let emit = h.emissive * light.emissive_strength;
            contrib = surf + emit;
        } else {
            contrib = bounce_sky(dir);
        }
        // No firefly clamp: a biased radiance cap is discarded in Phase 2.2 (best practice). Bright bounce
        // samples are handled correctly by ReSTIR resampling + the world cache's temporal averaging + DLSS-RR,
        // so `gather_gi` accumulates the unbiased radiance directly (matching Solari `sample_gi`).
        acc_rad = acc_rad + contrib;
    }
    // Cosine-pdf importance sampling ⇒ the irradiance estimate is the mean of the gathered radiance.
    return (acc_rad / f32(rays)) * light.gi_intensity;
}

// Distinct, high-contrast colour per LOD ring for the LOD debug view (`debug_view == 7`). Cycles a small
// palette so adjacent rings always contrast: LOD 0 (finest, native) = green, rising green→yellow→orange→red
// →magenta→blue→cyan→grey for progressively coarser rings. The instrument for validating clipmap/LOD-ring
// placement + cross-LOD continuity (GPU-worldgen plan Stages 3–4).
fn lod_color(lod: u32) -> vec3<f32> {
    var pal = array<vec3<f32>, 8>(
        vec3<f32>(0.15, 0.85, 0.25),  // 0 finest — green
        vec3<f32>(0.95, 0.90, 0.20),  // 1 — yellow
        vec3<f32>(0.95, 0.55, 0.15),  // 2 — orange
        vec3<f32>(0.90, 0.20, 0.20),  // 3 — red
        vec3<f32>(0.85, 0.30, 0.85),  // 4 — magenta
        vec3<f32>(0.25, 0.55, 0.95),  // 5 — blue
        vec3<f32>(0.20, 0.85, 0.85),  // 6 — cyan
        vec3<f32>(0.80, 0.80, 0.80),  // 7+ — grey
    );
    return pal[min(lod, 7u)];
}

// SSOT for the debug-view overlay colour (`debug_view` 1..7), shared by `restir_p2` and `restir_dlss_p2` so
// the two entries can NEVER disagree on a debug mode. `gi` is the caller's own GI-only estimate (the reservoir
// estimate `restir_p2_core` for the ReSTIR entries) — used only for `debug_view == 5`. Returns black on a miss.
fn debug_overlay_color(r: TraceResult, ro: vec3<f32>, rd: vec3<f32>, gi: vec3<f32>) -> vec3<f32> {
    if (r.hit == 0u) { return vec3<f32>(0.0); }
    let p = ro + rd * r.t;
    let origin = p + r.normal * light.shadow_bias;
    if (light.debug_view == 1u) {
        return r.normal * 0.5 + 0.5;                              // world-space face normals
    } else if (light.debug_view == 2u) {
        return vec3<f32>(clamp(r.t / 20.0, 0.0, 1.0));            // depth (0..20 m → black..white)
    } else if (light.debug_view == 3u) {
        return r.color.rgb;                                      // raw palette albedo
    } else if (light.debug_view == 4u) {
        return vec3<f32>(ambient_occlusion(origin, r.normal));   // AO only
    } else if (light.debug_view == 5u || light.debug_view == 8u) {
        return gi;                                               // 5 = indirect (GI) only; 8 = DI only — the caller chooses what it puts in `gi`
    } else if (light.debug_view == 6u) {
        // Face orientation: GREEN = front face (normal opposes the ray); RED = BACK face (normal along the
        // ray — i.e. we hit the inside/back of a voxel = the show-through bug).
        return select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), dot(r.normal, rd) > 0.0);
    } else if (light.debug_view == 7u) {
        return lod_color(meta_lod(metas[r.prim]));               // LOD ring of the hit brick
    }
    return r.color.rgb;
}

// --- DLSS Ray Reconstruction guide bindings ---------------------------------------------------------
// (Stage 4c.) When built with `--features dlss`, the live ReSTIR DLSS pass (`restir_dlss_p2`) writes the
// per-pixel inputs DLSS-RR consumes:
//   * out_tex (the HDR view colour → DLSS `color`) — the FULL noisy LIT colour (albedo × lighting + glow).
//     DLSS-RR DEMODULATES internally using the albedo guides below, denoises the lighting, then re-modulates;
//     so we pass the full radiance, NOT a pre-divided signal. This matches the validated Solari contract
//     (its RestIR shaders write `radiance × brdf` to the view target and supply albedo guides alongside).
//   * diffuse_albedo   (rgba8)   — the voxel palette albedo (DLSS's demodulation guide)
//   * specular_albedo  (rgba8)   — a tiny dielectric F0 floor for these matte diffuse voxels (~non-specular)
//   * normal_roughness (rgba16f) — world-space face normal (xyz) + perceptual roughness (w ≈ 1.0, matte)
//   * out_dlss_depth   (r32f)    — the trace hit's reverse-Z clip depth (matches Bevy's depth prepass)
//   * out_dlss_motion  (rg16f)   — screen-space motion: this pixel's hit reprojected into the PREVIOUS frame
// There is NO temporal accumulation here — DLSS-RR IS the denoiser. The guides are written by the ReSTIR DLSS
// pass; the resolve render pass (`resolve_dlss` in voxel_rt_composite.wgsl) copies depth+motion into the
// RENDER-ATTACHMENT-only prepass textures (which compute can't storage-write) and the colour into the view
// target.
@group(1) @binding(5) var out_diffuse_albedo: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(6) var out_specular_albedo: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(7) var out_normal_roughness: texture_storage_2d<rgba16float, write>;
@group(1) @binding(8) var out_dlss_depth: texture_storage_2d<r32float, write>;
// Motion is an intermediate STORAGE texture; `rg16float` storage isn't universally supported, so use a
// widely-supported `rgba16float` (only .xy carry the screen-space motion; .zw are 0). The resolve pass writes
// the final `Rg16Float` PREPASS motion texture via a render attachment (no storage requirement there).
@group(1) @binding(9) var out_dlss_motion: texture_storage_2d<rgba16float, write>;

// DLSS camera matrices. The render is UN-jittered (this voxel renderer disables jitter — see
// `voxel_rt_dlss_pass`), so all three are the same un-jittered clip_from_world: `depth_clip_from_world` for the
// depth-guide write, `motion_prev`/`motion_cur` for the motion vector (geometry motion only → exactly zero for a
// static camera). `camera.world_from_clip` is likewise un-jittered, so the primary ray IS the GI/DI receiver.
struct DlssCamera {
    depth_clip_from_world: mat4x4<f32>,
    motion_prev: mat4x4<f32>,
    motion_cur: mat4x4<f32>,
};
@group(1) @binding(10) var<uniform> dlss_cam: DlssCamera;


// ====================================================================================================
// ReSTIR GI — reservoir-based spatiotemporal resampling of the single-bounce diffuse GI (Ouyang 2021 /
// Wyman 2023 course notes). Ported from `bevy_solari::restir_gi` and adapted to OUR tracer + palette.
//
// The plain `gather_gi` mean boils because it re-randomises the bounce directions every frame and never
// REUSES samples. ReSTIR keeps a per-shading-point RESERVOIR holding one selected sample (the bounce hit
// point + the outgoing radiance there), resampled by RIS and REUSED across frames (temporal) and
// neighbours (spatial) with balance-heuristic MIS + a Jacobian for the solid-angle reparametrisation.
// Effective sample count grows into the hundreds for ~1 trace/pixel → the boil collapses.
//
// KEY ADAPTATION vs Solari: Solari EXCLUDES emissive sample points (its separate ReSTIR DI handles
// emitters). Our only light IS the emissive panel, so we INCLUDE emissive sample points and define the
// sample's outgoing radiance as `direct_lighting(sp) + emissive(sp)` — exactly the `contrib` term
// `gather_gi` accumulates. Resampling then concentrates samples toward the bright panel (NEE-by-
// resampling) with no separate DI pass. The estimator converges to the SAME irradiance `gather_gi`
// estimates, so the headless harness can assert ReSTIR ≈ a high-spp `gather_gi` reference.
//
// This block defines the reusable core (struct + helpers + initial-reservoir generation + merge) used by
// both the R0 headless probe test below and the live screen-space passes (R1+). The screen-space
// G-buffer/motion entries are added in R1; here `restir_probe` exercises the estimator math in isolation.

const RESTIR_PI: f32 = 3.14159265358979;
const RESTIR_CONFIDENCE_CAP: f32 = 8.0; // temporal history cap (frames) — bounds lag/ghosting

// A ReSTIR GI reservoir (48 bytes = 3×vec4; field order MUST match `GpuReservoir` in src/voxel/restir.rs
// and bevy_solari's `Reservoir`).
struct Reservoir {
    sample_point_world_position: vec3<f32>,
    weight_sum: f32,
    radiance: vec3<f32>,            // OUTGOING radiance L_o at the sample point toward the shading point
    confidence_weight: f32,         // ~ effective sample count M (capped)
    sample_point_world_normal: vec3<f32>,
    unbiased_contribution_weight: f32, // RIS contribution weight W (1/pdf · normalisation)
}

fn empty_reservoir() -> Reservoir {
    return Reservoir(vec3<f32>(0.0), 0.0, vec3<f32>(0.0), 0.0, vec3<f32>(0.0), 0.0);
}

// Rec.709 luminance — the scalar target function ReSTIR resamples by.
fn restir_luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Balance-heuristic MIS weight for a pair (Veach), Solari's NaN-safe form: a/(a+b) rewritten as
// 1/(1+b/a). The naive `a/(a+b)` gives `inf/inf → NaN` when a target function overflows, and that NaN is
// STORED in the reservoir and reused forever (a permanent dead pixel). This form returns 1 for a=inf and 0
// for b=inf instead. 0 when a == 0.
fn balance_heuristic(a: f32, b: f32) -> f32 {
    if (a == 0.0) { return 0.0; }
    return max(0.0, 1.0 / (1.0 + b / a));
}

fn restir_isinf(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u; }
fn restir_isnan(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u; }
// Replace any NaN/Inf component of a radiance triple with 0 — a robustness chokepoint so a single bad value
// (a div-by-near-zero, a coincident-point normalize, a degenerate cache cell) can NEVER be STORED into a
// reservoir / the world cache (where it would poison spatial+temporal neighbours forever) or reach out_tex.
fn sanitize3(v: vec3<f32>) -> vec3<f32> {
    let bad = restir_isnan(v.x) || restir_isnan(v.y) || restir_isnan(v.z) ||
              restir_isinf(v.x) || restir_isinf(v.y) || restir_isinf(v.z);
    return select(v, vec3<f32>(0.0), bad);
}

// A mutating PCG RNG (ReSTIR needs a stream: candidate dir + stochastic reservoir selection + neighbours).
fn rand_next(rng: ptr<function, u32>) -> f32 {
    *rng = *rng * 747796405u + 2891336453u;
    let s = *rng;
    let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return f32((word >> 22u) ^ word) * (1.0 / 4294967296.0);
}

// Uniform hemisphere sample about unit normal `n` (matches Solari's bounce sampling; pdf = 1/2π).
fn sample_uniform_hemisphere(n: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let z = rand_next(rng);                       // cos(theta) uniform in [0,1]
    let r = sqrt(max(0.0, 1.0 - z * z));
    let phi = 6.2831853 * rand_next(rng);
    let basis = onb(n);
    return normalize(basis[0] * (r * cos(phi)) + basis[1] * (r * sin(phi)) + n * z);
}
fn uniform_hemisphere_inverse_pdf() -> f32 { return 6.2831853; } // 2π

// Build an INITIAL reservoir from ONE uniform-hemisphere bounce in direction `dir` (pdf = 1/2π, so the
// unbiased contribution weight is 2π). The sample's outgoing radiance L_o = direct lighting at the hit +
// its emissive (emissive INCLUDED — the adaptation).
//
// MISS → a valid DISTANT SKY sample (open-world GI): the bounce escapes to open sky, so we record a far
// sample point along `dir` carrying `sky_radiance(dir) · gi_sky_intensity`. This keeps the uniform-hemisphere
// estimator UNBIASED (a single sky sample → E[2π · sky · cosθ/π] = sky integrated over the hemisphere), and
// putting the sample at `gi_bounce_dist` (far) makes the spatial/temporal-reuse Jacobian (cosθ'/dist²) ≈ 1 for
// nearby receivers, so sky reuse across pixels/frames is stable. A closed box rarely misses (the sky term is
// negligible there) so the energy probe test is unchanged; open scenes now get sky fill instead of black.
// Shared by the white-noise `generate_initial_reservoir` (the headless probe test) and the live
// low-discrepancy path (`restir_p1_core`) — the ONLY difference between them is how `dir` is chosen.
fn reservoir_from_bounce(world_position: vec3<f32>, world_normal: vec3<f32>, dir: vec3<f32>) -> Reservoir {
    var reservoir = empty_reservoir();
    let origin = world_position + world_normal * light.shadow_bias;
    let r = trace(origin, dir, 0.0, light.gi_bounce_dist);
    if (r.hit == 0u) {
        // Distant sky sample: a far virtual surface facing back along the ray, radiating the procedural sky.
        reservoir.sample_point_world_position = origin + dir * light.gi_bounce_dist;
        reservoir.sample_point_world_normal = -dir;
        reservoir.confidence_weight = 1.0;
        reservoir.radiance = sky_radiance(dir) * sky.gi_sky_intensity;
    } else {
        let hp = origin + dir * r.t;
        reservoir.sample_point_world_position = hp;
        reservoir.sample_point_world_normal = r.normal;
        reservoir.confidence_weight = 1.0;
        reservoir.radiance = direct_lighting(r.color.rgb, r.normal, hp) + r.emissive * light.emissive_strength;
    }
    // No firefly clamp (discarded in Phase 2.2, best practice): a biased radiance cap is gone. ReSTIR's
    // resampling, the world-cache temporal averaging, and DLSS-RR handle bright outliers correctly, so the
    // reservoir stores the unbiased radiance (matching Solari, whose initial sample is unclamped too).
    reservoir.unbiased_contribution_weight = uniform_hemisphere_inverse_pdf();
    return reservoir;
}

// Cache-fed INITIAL reservoir (Phase 2.2 go-live). Identical to `reservoir_from_bounce` EXCEPT the bounce-HIT
// radiance ADDS one reflected indirect bounce read from the world-space radiance cache (`query_world_cache`):
// the cache holds PRE-ACCUMULATED, multi-frame-averaged INCOMING indirect radiance (cosine-pre-divided), which
// collapses the per-frame variance that a fresh re-trace of the indirect term would boil with. The LD bounce
// DIRECTION still stratifies WHICH cell we sample; the cache supplies the reflected indirect leaving it.
//
//   * bounce HIT  → L_o(hp) = direct_lighting(hp) + emissive(hp) + albedo(hp)·query_world_cache(hp, …). The
//                   first two terms are byte-identical to the fresh path; the third is the reflected indirect.
//                   `query_world_cache` LAZY-INSERTS: an empty/just-claimed cell stores the hit geometry, marks
//                   itself alive, and returns 0 (it fills over the next ~1-2 frames via the update/blend passes
//                   — Solari's query-driven fill), so cache-off degrades cleanly to the fresh single bounce and
//                   cache-on adds the reflected indirect on top. Going live is ALSO what populates the cache.
//   * bounce MISS → `sky_radiance(dir) · gi_sky_intensity` (Phase 1A sky SSOT), UNCHANGED — the sky is not
//                   cached (it has no surface to anchor a cell), so a distant sky sample is recorded directly.
//
// `rng` is a mutating PCG stream the query uses for its stochastic cell-LOD rounding + tangent-plane jitter.
fn reservoir_from_bounce_cached(world_position: vec3<f32>, world_normal: vec3<f32>, dir: vec3<f32>, drop_emissive: bool, rng: ptr<function, u32>) -> Reservoir {
    var reservoir = empty_reservoir();
    let origin = world_position + world_normal * light.shadow_bias;
    let r = trace(origin, dir, 0.0, light.gi_bounce_dist);
    if (r.hit == 0u) {
        // Distant sky sample — GRADIENT only (sun disk excluded; the sun's GI is delivered through the cache via
        // `wc_sun_direct`, so a bounce toward the sun must not also see the disk). The sky has no cache cell.
        reservoir.sample_point_world_position = origin + dir * light.gi_bounce_dist;
        reservoir.sample_point_world_normal = -dir;
        reservoir.confidence_weight = 1.0;
        reservoir.radiance = sky_radiance_no_sun(dir) * sky.gi_sky_intensity;
    } else {
        let hp = origin + dir * r.t;
        reservoir.sample_point_world_position = hp;
        reservoir.sample_point_world_normal = r.normal;
        reservoir.confidence_weight = 1.0;
        // OUTGOING radiance of the bounce surface toward the shading point, read ENTIRELY from the world cache
        // (Solari `restir_gi`): L_o(hp) = emissive(hp) + albedo(hp)·cache(hp), where cache(hp) holds hp's TOTAL
        // incoming radiance — its own DIRECT (sun via `wc_sun_direct` + emitters via NEE) PLUS indirect — all
        // multi-frame-averaged in `world_cache_update`. We NO LONGER recompute `direct_lighting(hp)` FRESH here.
        //
        // THIS IS THE BOIL FIX (GI 5.0): the fresh per-frame `direct_lighting(hp)` carried the HARD sun-shadow at
        // the bounce hit — a high-dynamic-range term that made the ReSTIR candidate's target function swing ~10×
        // between sun-lit and shadowed bounces, so RIS kept SWITCHING the held sample to whatever bright bounce it
        // found (a switch the confidence cap can't damp — it's driven by the target RATIO). That switching is the
        // low-frequency "boil" DLSS-RR exposes (the heavy non-RR accumulator hid it). Reading the bounce radiance
        // from the SMOOTH, temporally-averaged cache makes the candidate target ~constant across bounce
        // directions → the reservoir stops switching → it converges. Pure Solari alignment. Cosine-pre-divided
        // cache ⇒ albedo only (no /π). The cache lazy-inserts on first sight (returns 0, fills over ~1-2 frames),
        // so a freshly-disoccluded pixel can be briefly dark until it fills (acceptable; static views are clean).
        let emit_term = select(r.emissive * light.emissive_strength, vec3<f32>(0.0), drop_emissive);
        reservoir.radiance = emit_term
            + r.color.rgb * query_world_cache(hp, r.normal, camera.cam_pos, r.t, wc.cell_lifetime, rng);
    }
    reservoir.unbiased_contribution_weight = uniform_hemisphere_inverse_pdf();
    return reservoir;
}

// White-noise initial reservoir (one random uniform-hemisphere bounce). Used by the headless `restir_probe`
// estimator test, which asserts convergence to the high-spp `gather_gi` oracle (unbiased in expectation —
// white noise keeps that test simple). The live path uses the low-discrepancy variant below.
fn generate_initial_reservoir(world_position: vec3<f32>, world_normal: vec3<f32>, rng: ptr<function, u32>) -> Reservoir {
    return reservoir_from_bounce(world_position, world_normal, sample_uniform_hemisphere(world_normal, rng));
}

// Low-discrepancy UNIFORM-hemisphere direction: Hammersley point (i/N, van-der-Corput(i)) Cranley–Patterson-
// rotated by `rot`, mapped so z = cosθ is uniform — IDENTICAL parametrisation to `sample_uniform_hemisphere`,
// so the 1/pdf = 2π convention (and the resolve math) is unchanged; only the noise structure improves. Within
// a pixel the N RIS candidates stay well-stratified → a STEADIER fraction of them catch the bright emitter
// frame-to-frame → far less of the per-frame COUNT variance that is the boil source (the cap-8 temporal
// reservoir cannot average that away). The per-pixel/frame `rot` animates the residual noise uniformly
// (blue-noise-like → exactly what DLSS-RR + the temporal accumulator reproject + average best). This is the
// same technique the legacy `gather_gi` proved cuts boil at equal ray count; the ReSTIR path had regressed
// to white noise, which is why boil persisted even at DLAA (native res, where reprojection error is nil).
fn ld_uniform_hemisphere(n: vec3<f32>, i: u32, count: u32, rot: vec2<f32>) -> vec3<f32> {
    let u1 = fract(f32(i) / f32(count) + rot.x);
    let u2 = fract(radical_inverse_vdc(i) + rot.y);
    let z = u1;                                   // cos(theta) uniform in [0,1] → uniform hemisphere
    let r = sqrt(max(0.0, 1.0 - z * z));
    let phi = 6.2831853 * u2;
    let basis = onb(n);
    return normalize(basis[0] * (r * cos(phi)) + basis[1] * (r * sin(phi)) + n * z);
}

// Jacobian of the solid-angle reparametrisation when a sample taken at `original_world_position` is reused
// at `new_world_position` (Ouyang 2021 eq. / Solari). 0 on degenerate (inf/nan).
fn restir_jacobian(new_world_position: vec3<f32>, original_world_position: vec3<f32>, sample_point_world_position: vec3<f32>, sample_point_world_normal: vec3<f32>) -> f32 {
    let rr = new_world_position - sample_point_world_position;
    let qq = original_world_position - sample_point_world_position;
    let rl = length(rr);
    let ql = length(qq);
    let phi_r = saturate(dot(rr / rl, sample_point_world_normal));
    let phi_q = saturate(dot(qq / ql, sample_point_world_normal));
    let j = (phi_r * ql * ql) / (phi_q * rl * rl);
    return select(j, 0.0, restir_isinf(j) || restir_isnan(j));
}

struct ReservoirMergeResult {
    merged_reservoir: Reservoir,
    selected_sample_radiance: vec3<f32>,
    wi: vec3<f32>,
}

// --- Defensive pairwise-MIS helpers (RTXDI / Wyman 2023 "A Gentle Introduction to ReSTIR", Algo 7) ----------
// The GI target function p̂ of a sample carrying outgoing radiance `radiance` at `sample_pos`, evaluated at a
// receiver (`recv_pos`, `recv_n`): luminance(L_o)·cosθ (brdf = 1, receiver albedo applied by the caller). The
// Jacobian for a cross-receiver shift is applied by the CALLER (multiplied onto this scalar).
fn restir_target_at(radiance: vec3<f32>, sample_pos: vec3<f32>, recv_pos: vec3<f32>, recv_n: vec3<f32>) -> f32 {
    let wi = normalize(sample_pos - recv_pos);
    return restir_luminance(radiance) * saturate(dot(wi, recv_n));
}

// Generalized pairwise balance-heuristic weight (RTXDI `RTXDI_PairwiseMisWeight`): m0·w0 / (m0·w0 + m1·w1),
// 0-guarded. `w0`/`w1` are the two techniques' target functions, `m0`/`m1` their confidence weights.
fn pairwise_mis(w0: f32, w1: f32, m0: f32, m1: f32) -> f32 {
    let denom = m0 * w0 + m1 * w1;
    return select(0.0, max(0.0, m0 * w0) / denom, denom > 0.0);
}

// Confidence attenuation for a shifted neighbour (RTXDI `RTXDI_MFactor`): down-weights a neighbour's M
// contribution when its target at the OTHER receiver is much smaller than at its own (a poorly-shifted /
// dissimilar sample), preventing confidence inflation. 1 when the native target is 0.
fn mis_m_factor(q0: f32, q1: f32) -> f32 {
    return select(clamp(pow(min(q1 / q0, 1.0), 8.0), 0.0, 1.0), 1.0, q0 <= 0.0);
}

// Pairwise RIS merge of a canonical reservoir with another (temporal or spatial), with balance-heuristic
// MIS over the two target functions and the Jacobian for the shifted sample. Ported verbatim from
// bevy_solari::restir_gi::merge_reservoirs (our `restir_luminance`/`restir_jacobian`). The diffuse BRDF is
// `albedo / π`; for the irradiance estimate the receiver albedo is applied by the caller.
fn merge_reservoirs(
    canonical_reservoir: Reservoir,
    canonical_world_position: vec3<f32>,
    canonical_world_normal: vec3<f32>,
    canonical_diffuse_brdf: vec3<f32>,
    other_reservoir: Reservoir,
    other_world_position: vec3<f32>,
    other_world_normal: vec3<f32>,
    other_diffuse_brdf: vec3<f32>,
    rng: ptr<function, u32>,
) -> ReservoirMergeResult {
    let canonical_sample_wi = normalize(canonical_reservoir.sample_point_world_position - canonical_world_position);
    let other_sample_wi = normalize(other_reservoir.sample_point_world_position - canonical_world_position);

    let canonical_target_function_canonical_sample = restir_luminance(
        canonical_reservoir.radiance * saturate(dot(canonical_sample_wi, canonical_world_normal)) * canonical_diffuse_brdf);
    let canonical_target_function_other_sample = restir_luminance(
        other_reservoir.radiance * saturate(dot(other_sample_wi, canonical_world_normal)) * canonical_diffuse_brdf);
    let other_target_function_canonical_sample = restir_luminance(
        canonical_reservoir.radiance * saturate(dot(normalize(canonical_reservoir.sample_point_world_position - other_world_position), other_world_normal)) * other_diffuse_brdf);
    let other_target_function_other_sample = restir_luminance(
        other_reservoir.radiance * saturate(dot(normalize(other_reservoir.sample_point_world_position - other_world_position), other_world_normal)) * other_diffuse_brdf);

    let canonical_target_function_other_sample_jacobian = restir_jacobian(
        canonical_world_position, other_world_position, other_reservoir.sample_point_world_position, other_reservoir.sample_point_world_normal);
    let other_target_function_canonical_sample_jacobian = restir_jacobian(
        other_world_position, canonical_world_position, canonical_reservoir.sample_point_world_position, canonical_reservoir.sample_point_world_normal);

    // Huge jacobians explode the variance — skip the merge (keep the canonical).
    if (canonical_target_function_other_sample_jacobian > 1.2 || other_target_function_canonical_sample_jacobian > 1.2) {
        return ReservoirMergeResult(canonical_reservoir, canonical_reservoir.radiance, canonical_sample_wi);
    }

    let canonical_sample_mis_weight = balance_heuristic(
        canonical_reservoir.confidence_weight * canonical_target_function_canonical_sample,
        other_reservoir.confidence_weight * other_target_function_canonical_sample * other_target_function_canonical_sample_jacobian);
    let canonical_sample_resampling_weight = canonical_sample_mis_weight * canonical_target_function_canonical_sample * canonical_reservoir.unbiased_contribution_weight;

    let other_sample_mis_weight = balance_heuristic(
        other_reservoir.confidence_weight * other_target_function_other_sample,
        canonical_reservoir.confidence_weight * canonical_target_function_other_sample * canonical_target_function_other_sample_jacobian);
    let other_sample_resampling_weight = other_sample_mis_weight * canonical_target_function_other_sample * other_reservoir.unbiased_contribution_weight * canonical_target_function_other_sample_jacobian;

    var combined = empty_reservoir();
    combined.confidence_weight = canonical_reservoir.confidence_weight + other_reservoir.confidence_weight;
    combined.weight_sum = canonical_sample_resampling_weight + other_sample_resampling_weight;

    if (rand_next(rng) < other_sample_resampling_weight / max(combined.weight_sum, 1e-12)) {
        combined.sample_point_world_position = other_reservoir.sample_point_world_position;
        combined.sample_point_world_normal = other_reservoir.sample_point_world_normal;
        combined.radiance = other_reservoir.radiance;
        let inv_tf = select(0.0, 1.0 / canonical_target_function_other_sample, canonical_target_function_other_sample > 0.0);
        combined.unbiased_contribution_weight = combined.weight_sum * inv_tf;
        return ReservoirMergeResult(combined, other_reservoir.radiance, other_sample_wi);
    } else {
        combined.sample_point_world_position = canonical_reservoir.sample_point_world_position;
        combined.sample_point_world_normal = canonical_reservoir.sample_point_world_normal;
        combined.radiance = canonical_reservoir.radiance;
        let inv_tf = select(0.0, 1.0 / canonical_target_function_canonical_sample, canonical_target_function_canonical_sample > 0.0);
        combined.unbiased_contribution_weight = combined.weight_sum * inv_tf;
        return ReservoirMergeResult(combined, canonical_reservoir.radiance, canonical_sample_wi);
    }
}

// Resolve a reservoir to the indirect IRRADIANCE at the shading point (the quantity `gather_gi` returns,
// BEFORE the receiver albedo). With uniform-hemisphere sampling (1/pdf = 2π) and a cosine-weighted
// integrand g = (1/π)·L_o·cos, the RIS estimate of I = (1/π)∫L_o cosθ dω is
// `radiance · W · cos / π` — the same I that `gather_gi`'s cosine-mean estimates. ×gi_intensity to match.
fn restir_resolve_irradiance(res: Reservoir, recv_pos: vec3<f32>, recv_normal: vec3<f32>) -> vec3<f32> {
    if (res.confidence_weight <= 0.0) { return vec3<f32>(0.0); }
    let wi = normalize(res.sample_point_world_position - recv_pos);
    let cos = saturate(dot(wi, recv_normal));
    // Backstop: guard radiance (only `unbiased_contribution_weight` is checked at the merge sites) so a NaN/Inf
    // radiance × a finite weight can't reach out_tex or DLSS-RR (which would smear the firefly across the frame).
    return sanitize3(res.radiance * res.unbiased_contribution_weight * cos * (1.0 / RESTIR_PI) * light.gi_intensity);
}

// --- R0 headless probe test entry -------------------------------------------------------------------
// Exercises the ReSTIR estimator math WITHOUT the screen-space G-buffer plumbing (added in R1). For each
// probe (a shading point: world position + normal), each dispatch generates one initial reservoir and
// merges it into the probe's PERSISTENT reservoir (temporal accumulation; same-surface merge so the
// Jacobian = 1). Over N dispatches the resolved irradiance must converge (variance → 0) to the high-spp
// `gather_gi` reference, and concentrate samples toward the emissive panel. `reset` clears the reservoir.

struct ProbePoint { world_position: vec3<f32>, _p0: u32, world_normal: vec3<f32>, _p1: u32 };
struct ProbeOut {
    irradiance: vec3<f32>,        // resolved ReSTIR indirect irradiance this dispatch
    confidence: f32,
    reference: vec3<f32>,         // high-spp gather_gi irradiance (the unbiased oracle)
    ucw: f32,
};
struct RestirProbeParams { frame_index: u32, reset: u32, n_probes: u32, _p: u32 };

@group(0) @binding(8) var<storage, read> probes_in: array<ProbePoint>;
@group(0) @binding(9) var<storage, read_write> probe_reservoirs: array<Reservoir>;
@group(0) @binding(10) var<storage, read_write> probe_out: array<ProbeOut>;
@group(0) @binding(11) var<uniform> rp_params: RestirProbeParams;

@compute @workgroup_size(64)
fn restir_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= rp_params.n_probes) { return; }
    let probe = probes_in[i];
    let pos = probe.world_position;
    let n = probe.world_normal;

    var rng = (i * 9781u + rp_params.frame_index * 26699u) | 1u;

    var canonical = generate_initial_reservoir(pos, n, &rng);

    // Temporal reuse: merge the probe's previous-dispatch reservoir (same surface → Jacobian 1), unless reset.
    if (rp_params.reset == 0u) {
        var temporal = probe_reservoirs[i];
        temporal.confidence_weight = min(temporal.confidence_weight, RESTIR_CONFIDENCE_CAP);
        let brdf = vec3<f32>(1.0); // receiver albedo factored out of the irradiance estimate
        let merged = merge_reservoirs(canonical, pos, n, brdf, temporal, pos, n, brdf, &rng);
        canonical = merged.merged_reservoir;
    }

    probe_reservoirs[i] = canonical;

    var out: ProbeOut;
    out.irradiance = restir_resolve_irradiance(canonical, pos, n);
    out.confidence = canonical.confidence_weight;
    out.ucw = canonical.unbiased_contribution_weight;
    // Reference: the established cosine-mean GI estimator at this probe (fixed high-spp inside `gather_gi`).
    out.reference = gather_gi(n, pos, (i * 2654435761u + rp_params.frame_index * 40503u) | 1u);
    // Per-FRAME slot so the harness reads the whole convergence history back in one map.
    probe_out[rp_params.frame_index * rp_params.n_probes + i] = out;
}

// ====================================================================================================
// R1 — LIVE screen-space ReSTIR GI (single-pass, same-pixel temporal reuse).
//
// Each pixel keeps a per-pixel reservoir. This pass: traces the primary ray, generates ONE initial GI
// candidate, merges the PREVIOUS frame's SAME-pixel reservoir (temporal reuse), writes the merged reservoir
// to the current buffer, and resolves it to the indirect irradiance — replacing `gather_gi`'s per-frame mean.
// For a STILL camera the previous-frame pixel is the same world surface, so same-pixel reuse is exact and the
// estimate converges (the boil collapses). On camera motion the renderer raises `reset` (like the temporal
// accumulator), so reservoirs are dropped that frame — no reprojection yet (motion-vector reprojection +
// spatial reuse are R2). Two reservoir storage buffers ping-pong (cur written, prev read) — no G-buffer
// textures, so no storage-texture-limit pressure.
// Runtime ReSTIR knobs (group-2 uniform; editor-driven, knobs-as-uniforms). `spatial_samples` neighbours are
// merged per pixel from last frame's reservoirs (smooths dark/shadow regions where the temporal permute alone
// is too slow); `spatial_radius` is the disk radius in pixels; `confidence_weight_cap` bounds temporal/spatial
// history (lag vs stability). The final row holds the GI 4.0 screen-space ReSTIR DI knobs (emissive-voxel
// direct light): `di_enabled` gates the DI pass; `di_confidence_cap` is the DI temporal history cap (Solari DI
// uses 20, higher than GI's because DI samples are cheaper/more stable); `di_initial_samples` = RIS candidates
// drawn per pixel per frame (Solari 8). 48 bytes (3×vec4).
struct RestirParams {
    reset: u32,
    frame_index: u32,
    viewport_x: u32,
    viewport_y: u32,
    spatial_samples: u32,
    confidence_weight_cap: f32,
    spatial_radius: f32,
    di_enabled: u32,
    di_confidence_cap: f32,
    di_initial_samples: u32,
    // (vestigial) was the M>1 initial-RIS candidate count; restir_p1 is now straight-line M=1 (camera jitter,
    // not low M, was the boil cause — the M-loop's live state pinned p1 occupancy). Kept as padding for layout.
    _pad_gi: u32,
    // Half-resolution GI: `gi_half` = 1 → restir_p1 + the GI spatial pass run at (gi_half_x, gi_half_y) =
    // render_res/2; the full-res shade reservoir-resolve-gathers the half-res finals. The GI cores index
    // reservoirs at the half-res dims when this is set; DI/shade stay full-res (viewport_*).
    gi_half: u32,
    gi_half_x: u32,
    gi_half_y: u32,
    // Caps the view-distance used in `surfaces_dissimilar`'s RELATIVE tangent-plane reject (0 = uncapped / pure
    // Solari). Beyond this distance the threshold stops loosening, becoming an ABSOLUTE tangent cap
    // (≈ 0.003·cap_dist) — which rejects far-side-of-a-thin-wall spatial-reuse neighbours that the relative test
    // lets through at distance (GI leaking through thin walls). Tunable: lower = catches thinner walls farther
    // out but risks over-rejecting genuine slope reuse; raise toward off if it adds boil on terrain.
    gi_dissim_cap_dist: f32,
    _pad3: u32,
};
// Per-pixel RECEIVER surface (world pos + face normal) — needed so a temporal/neighbour reservoir can be
// merged with the correct Jacobian + rejected when it lands on a dissimilar surface (port of Solari's
// gbuffer-resolve, but we store pos/normal directly instead of repacking depth).
struct PixelSurface { world_position: vec3<f32>, valid: f32, world_normal: vec3<f32>, _pad: f32 };
// Two-pass split (Solari `initial_and_temporal` → `spatial_and_shade`): FIXED-ROLE reservoir buffers, NOT
// ping-ponged. `reservoirs_a` = the FINAL/history pool (read by pass 1's temporal tap = last frame's final;
// written by pass 2 = this frame's final). `reservoirs_b` = the intermediate POST-TEMPORAL pool (written by
// pass 1; read by pass 2's same-frame spatial reuse). Both passes run in ONE compute dispatch sequence; the
// intra-pass storage barrier orders pass-1-writes-b before pass-2-reads-b. Surfaces still ping-pong (cur/prev).
@group(2) @binding(0) var<storage, read_write> reservoirs_a: array<Reservoir>;
@group(2) @binding(1) var<storage, read_write> reservoirs_b: array<Reservoir>;
@group(2) @binding(2) var<uniform> restir_params: RestirParams;
@group(2) @binding(3) var<storage, read_write> surfaces_cur: array<PixelSurface>;
@group(2) @binding(4) var<storage, read_write> surfaces_prev: array<PixelSurface>;

// ====================================================================================================
// GI 4.0 — SCREEN-SPACE ReSTIR DI (direct light from the emissive-voxel list). Port of bevy_solari's
// `restir_di.wgsl` adapted to our engine: our LIGHT LIST is the per-voxel `voxel_lights` + power-weighted
// `voxel_light_alias` (group 3) — NOT Solari's triangle `light_sources`. The dominant EMITTER-scene boil is
// the emitter being found only when a random GI bounce HITS it (`emissive(hit)`, hit-or-miss → high variance,
// docs/GI_BOIL_PLAN.md §6). DI replaces that with a RESAMPLED, temporally+spatially reused, visibility-checked
// direct estimate at the PRIMARY surface — low variance by construction (Bitterli 2020 / RTXDI). When DI is
// on, the GI bounce DROPS its raw `emissive(hit)` term (now DI's job) so there is no double-count; the cache's
// emitter NEE still carries INDIRECT (emitter→wall→eye) bounces, which DI does not.
//
// A DI reservoir stores the SELECTED light as (light_index, seed) — the seed regenerates the face-jitter point
// deterministically (Solari's light_id+seed scheme), so reuse across pixels/frames needs only 16 B/pixel. Two
// fixed-role buffers ping-pong like the GI reservoirs: `di_reservoirs_a` = final/history, `di_reservoirs_b` =
// post-temporal (pass-1 → pass-2). No light-tile presampling yet (a memory-coherence optimization for LARGE
// light counts; our scenes have few emitters — the variance win is the reservoir, not the tiles; tiles are a
// documented scalability follow-up).
struct DiReservoir {
    light_index: u32,                  // index into voxel_lights; DI_NULL = invalid/empty
    seed: u32,                         // RNG seed → regenerates the face-jitter sample point (deterministic)
    confidence_weight: f32,            // ~ effective sample count M (capped at di_confidence_cap)
    unbiased_contribution_weight: f32, // RIS contribution weight W
};
const DI_NULL: u32 = 0xFFFFFFFFu;
@group(2) @binding(5) var<storage, read_write> di_reservoirs_a: array<DiReservoir>;
@group(2) @binding(6) var<storage, read_write> di_reservoirs_b: array<DiReservoir>;
// Spatiotemporal blue noise (Bevy `stbn.ktx2`): `.xy` is a blue-noise-in-space-AND-time uniform value in
// [0,1)², indexed by frame → array layer. A 1×1 dummy is bound until it's resident (the `dims > 1` guard in
// `stbn_rotation` then uses white noise). Sampled by `restir_p1_core` for the GI sample-direction rotation.
@group(2) @binding(7) var stbn_tex: texture_2d_array<f32>;

fn di_empty() -> DiReservoir { return DiReservoir(DI_NULL, 0u, 0.0, 0.0); }
fn di_valid(r: DiReservoir) -> bool { return r.light_index != DI_NULL; }

// Resolve a (light_index, seed) to a concrete sample POINT on the emitter's face + its radiance + area
// inverse-pdf. RECEIVER-INDEPENDENT (a fixed jitter basis, NOT the receiver-facing one `wc_sample_light_nee`
// uses) so the SAME (light_index, seed) yields the SAME point at every pixel that reuses it — required for
// unbiased spatial/temporal reuse. `radiance` already folds the `emissive_strength` knob. Isotropic voxel
// emitter (cos_light = 1), matching the NEE model so the alias `inv_pdf` is self-consistent.
struct DiLightSample { pos: vec3<f32>, radiance: vec3<f32>, inv_pdf: f32 };
fn di_resolve(light_index: u32, seed: u32) -> DiLightSample {
    let lgt = voxel_lights[light_index];
    var rng = seed;
    let half = sqrt(lgt.area) * 0.5;
    let basis = onb(vec3<f32>(0.5773502691, 0.5773502691, 0.5773502691)); // fixed axis ⇒ receiver-independent
    let j = (vec2<f32>(rand_next(&rng), rand_next(&rng)) * 2.0 - 1.0) * half;
    let pos = lgt.pos + basis[0] * j.x + basis[1] * j.y;
    return DiLightSample(pos, lgt.radiance * light.emissive_strength, lgt.inv_pdf);
}

// The unshadowed contribution of a resolved light sample at a receiver (pos, normal): the reflected-radiance
// colour (brdf factored to 1 — receiver albedo applied by the caller, matching the GI/sun convention) and the
// scalar RIS target function. ENGINE CONVENTION: irradiance E = radiance·cos_recv·cos_light/dist² with NO 1/π
// (the whole direct path — sun in `shade_restir_p2`, emitter glow — omits 1/π; matching it keeps DI consistent
// with the direct sun). `inv_pdf` (area-measure) converts the single sample to an unbiased E estimate.
struct DiContribution { radiance: vec3<f32>, tf: f32, wi: vec3<f32>, dist: f32 };
fn di_contribution(smp: DiLightSample, recv_pos: vec3<f32>, recv_normal: vec3<f32>) -> DiContribution {
    let to_light = smp.pos - recv_pos;
    let dist2 = max(dot(to_light, to_light), 1.0e-8);
    let dist = sqrt(dist2);
    let wi = to_light / dist;
    let cos_recv = max(dot(recv_normal, wi), 0.0);
    // Single-sample unbiased estimate of the reflected radiance from this light (brdf = 1, cos_light = 1):
    let contrib = smp.radiance * (cos_recv / dist2) * smp.inv_pdf;
    return DiContribution(contrib, restir_luminance(contrib), wi, dist);
}

// Trace a shadow ray from the receiver to the light sample point; 1.0 if visible, 0.0 if occluded. Stops ONE
// emitter-cell short (the light point sits ~half a cell inside the solid emissive voxel, like the cache NEE) so
// the emitter's own body doesn't self-shadow every sample.
fn di_visibility(recv_pos: vec3<f32>, recv_normal: vec3<f32>, light_pos: vec3<f32>, light_index: u32) -> f32 {
    let origin = recv_pos + recv_normal * light.shadow_bias;
    let to_light = light_pos - origin;
    let dist = length(to_light);
    if (dist <= 1.0e-6) { return 1.0; }
    let cell = sqrt(voxel_lights[light_index].area);
    let t_max = dist - cell;
    if (t_max <= light.shadow_bias) { return 1.0; } // adjacent emitter — no room for an occluder
    if (trace_occluded(origin, to_light / dist, light.shadow_bias, t_max)) { return 0.0; }
    return 1.0;
}

// Draw ONE power-weighted light via the alias table, returning (light_index, seed-for-its-jitter). Advances rng.
fn di_pick_light(rng: ptr<function, u32>) -> vec2<u32> {
    let slot = min(u32(rand_next(rng) * f32(wc.light_count)), wc.light_count - 1u);
    let entry = voxel_light_alias[slot];
    var li = slot;
    if (rand_next(rng) >= entry.prob) { li = entry.alias_idx; }
    let seed = *rng; // a deterministic seed for THIS candidate's face jitter (distinct per candidate)
    rand_next(rng);  // advance so the next candidate's seed differs
    return vec2<u32>(li, seed);
}

// INITIAL + RIS: draw `di_initial_samples` power-weighted candidates, resample one by its target function
// (Solari `generate_initial_reservoir`). No per-candidate visibility (that would be N shadow rays); the
// selected sample's visibility is folded into the resolve in pass 2 (we store the UNSHADOWED reservoir so
// neighbours reuse an unbiased estimate, then shade with per-receiver visibility — same contract as our GI).
fn di_generate_initial(recv_pos: vec3<f32>, recv_normal: vec3<f32>, rng: ptr<function, u32>) -> DiReservoir {
    var res = di_empty();
    if (wc.light_count == 0u) { return res; }
    let m = max(restir_params.di_initial_samples, 1u);
    let mis_weight = 1.0 / f32(m);
    var weight_sum = 0.0;
    var sel_target = 0.0;
    for (var i = 0u; i < m; i = i + 1u) {
        let pick = di_pick_light(rng);
        let smp = di_resolve(pick.x, pick.y);
        let c = di_contribution(smp, recv_pos, recv_normal);
        // The candidate is already a 1/pdf-weighted single-sample E estimate, so its target IS its luminance;
        // mis_weight (1/m) makes the M candidates a balanced mixture (Solari).
        let resampling_weight = mis_weight * c.tf;
        weight_sum += resampling_weight;
        if (rand_next(rng) < resampling_weight / max(weight_sum, 1e-12)) {
            sel_target = c.tf;
            res.light_index = pick.x;
            res.seed = pick.y;
        }
    }
    if (sel_target > 0.0) {
        res.unbiased_contribution_weight = weight_sum / sel_target;
        // CRUCIAL (Solari `generate_initial_reservoir`): fold the SELECTED sample's visibility into the ucw, so
        // the stored reservoir is VISIBILITY-AWARE. Without this, ReSTIR keeps an occluded-but-bright light in
        // the reservoir and the per-frame shade-time visibility ray flickers 0↔1 as the selected light changes
        // → the dominant DI boil (measured CoV ~1.25 before this). With it, an occluded light's ucw collapses to
        // 0 so it is dropped from temporal/spatial reuse. The pass-2 shade still traces a per-receiver visibility
        // for the final (a reused neighbour's light may be occluded HERE).
        let sel_smp = di_resolve(res.light_index, res.seed);
        res.unbiased_contribution_weight *= di_visibility(recv_pos, recv_normal, sel_smp.pos, res.light_index);
    }
    res.confidence_weight = 1.0;
    return res;
}

struct DiMergeResult { reservoir: DiReservoir, radiance: vec3<f32>, wi: vec3<f32>, dist: f32, light_pos: vec3<f32> };

// Pairwise balance-heuristic MIS merge of two DI reservoirs at the canonical receiver (mirrors our GI
// `merge_reservoirs` / Solari DI merge). Each light sample is resolved + evaluated at both the canonical and
// the other receiver for the MIS partition; the target function is the contribution luminance.
fn di_merge(
    canonical: DiReservoir, canon_pos: vec3<f32>, canon_n: vec3<f32>,
    other: DiReservoir, other_pos: vec3<f32>, other_n: vec3<f32>,
    rng: ptr<function, u32>,
) -> DiMergeResult {
    // Resolve each reservoir's sample once.
    var canon_smp: DiLightSample;
    var other_smp: DiLightSample;
    if (di_valid(canonical)) { canon_smp = di_resolve(canonical.light_index, canonical.seed); }
    if (di_valid(other))     { other_smp = di_resolve(other.light_index, other.seed); }

    // Contributions at the canonical receiver (for resampling) and the other receiver (for MIS).
    var c_at_c: DiContribution; var o_at_c: DiContribution; var c_at_o: DiContribution; var o_at_o: DiContribution;
    if (di_valid(canonical)) { c_at_c = di_contribution(canon_smp, canon_pos, canon_n); c_at_o = di_contribution(canon_smp, other_pos, other_n); }
    if (di_valid(other))     { o_at_c = di_contribution(other_smp, canon_pos, canon_n); o_at_o = di_contribution(other_smp, other_pos, other_n); }

    let canon_mis = balance_heuristic(canonical.confidence_weight * c_at_c.tf, other.confidence_weight * c_at_o.tf);
    let canon_w = canon_mis * c_at_c.tf * canonical.unbiased_contribution_weight;
    let other_mis = balance_heuristic(other.confidence_weight * o_at_o.tf, canonical.confidence_weight * o_at_c.tf);
    let other_w = other_mis * o_at_c.tf * other.unbiased_contribution_weight;

    var combined = di_empty();
    combined.confidence_weight = canonical.confidence_weight + other.confidence_weight;
    let wsum = canon_w + other_w;

    if (di_valid(other) && rand_next(rng) < other_w / max(wsum, 1e-12)) {
        combined.light_index = other.light_index;
        combined.seed = other.seed;
        let inv_tf = select(0.0, 1.0 / o_at_c.tf, o_at_c.tf > 0.0);
        combined.unbiased_contribution_weight = wsum * inv_tf;
        return DiMergeResult(combined, o_at_c.radiance, o_at_c.wi, o_at_c.dist, other_smp.pos);
    } else {
        combined.light_index = canonical.light_index;
        combined.seed = canonical.seed;
        let inv_tf = select(0.0, 1.0 / c_at_c.tf, c_at_c.tf > 0.0);
        combined.unbiased_contribution_weight = wsum * inv_tf;
        return DiMergeResult(combined, c_at_c.radiance, c_at_c.wi, c_at_c.dist, canon_smp.pos);
    }
}

// PASS 1 (DI): initial RIS + temporal reuse → di_reservoirs_b. Reads the PERMUTED reprojected previous-frame
// final (di_reservoirs_a) for this surface, capped at `di_confidence_cap`. Shares `surfaces_*` + the reproject
// /permute/dissimilar machinery with the GI pass. `seed` is offset so the DI rng stream decorrelates from GI.
fn di_p1_core(n: vec3<f32>, p: vec3<f32>, pix: vec2<u32>, temporal_base: vec2<i32>, seed: u32) {
    let vp = vec2<u32>(restir_params.viewport_x, restir_params.viewport_y);
    let idx = pix.y * vp.x + pix.x;
    if (restir_params.di_enabled == 0u || wc.light_count == 0u) { di_reservoirs_b[idx] = di_empty(); return; }
    var rng = seed ^ 0x9E3779B9u;
    var res = di_generate_initial(p, n, &rng);

    if (restir_params.reset == 0u) {
        // SAME-PIXEL (reprojected, NOT permuted) temporal tap. The GI pass permutes its temporal tap to keep
        // EXPLORING bounce directions (continuous radiance); DI is the OPPOSITE — its sample is a DISCRETE light,
        // and we WANT each pixel to CONVERGE onto and HOLD one good visible light. Permuting reads a different
        // neighbour's (different) light every frame, so the held light keeps getting swapped → the contribution
        // jitters frame-to-frame (measured: permuted DI CoV ~0.30; same-pixel collapses it). Spatial reuse still
        // shares lights across pixels. Off-screen reprojection falls back to the current pixel.
        var tb = temporal_base;
        if (tb.x < 0 || tb.y < 0 || tb.x >= i32(vp.x) || tb.y >= i32(vp.y)) { tb = vec2<i32>(pix); }
        let tidx = u32(tb.y) * vp.x + u32(tb.x);
        let surf = surfaces_prev[tidx];
        if (surf.valid > 0.5 && !surfaces_dissimilar(p, n, surf.world_position, surf.world_normal)) {
            var temporal = di_reservoirs_a[tidx];
            temporal.confidence_weight = min(temporal.confidence_weight, restir_params.di_confidence_cap);
            res = di_merge(res, p, n, temporal, surf.world_position, surf.world_normal, &rng).reservoir;
        }
    }
    if (restir_isnan(res.unbiased_contribution_weight) || restir_isinf(res.unbiased_contribution_weight)) {
        res.unbiased_contribution_weight = 0.0;
    }
    di_reservoirs_b[idx] = res;
}

// PASS 2 (DI): one same-frame spatial neighbour merge (from di_reservoirs_b) → store final to di_reservoirs_a,
// then resolve the selected light at THIS receiver, trace ONE visibility ray, and return the reflected DIRECT
// emitter radiance (× receiver albedo by the caller). Store-before-visibility (the unbiased contract: the
// stored reservoir stays an unshadowed estimate neighbours can reuse; visibility is a per-receiver shading
// correction). Returns 0 when DI is off / no lights / miss.
fn di_p2_core(n: vec3<f32>, p: vec3<f32>, pix: vec2<u32>, seed: u32) -> vec3<f32> {
    if (restir_params.di_enabled == 0u || wc.light_count == 0u) { return vec3<f32>(0.0); }
    let vp = vec2<u32>(restir_params.viewport_x, restir_params.viewport_y);
    let idx = pix.y * vp.x + pix.x;
    var rng = (seed ^ 0x9E3779B9u) ^ 0xD2511F53u;
    var res = di_reservoirs_b[idx];

    // DI is TEMPORAL-ONLY (no spatial neighbour merge). Spatial reuse of a CONTINUOUS GI radiance averages
    // smoothly, but merging two pixels' DISCRETE light choices picks ONE ~50/50 and OSCILLATES the held light
    // frame-to-frame → boil (measured: with spatial, DI CoV ~0.26; temporal-only collapses it). The temporal
    // reservoir alone accumulates enough effective samples for a stable per-pixel light. (`spatial_samples`
    // still drives the GI pass.) A discrete-light-aware spatial scheme (light-tile presampling) is the upgrade.
    // No spatial DI merge here (deliberately — see above). The temporal reservoir is the whole estimate.
    if (restir_isnan(res.unbiased_contribution_weight) || restir_isinf(res.unbiased_contribution_weight)) {
        res.unbiased_contribution_weight = 0.0;
    }
    di_reservoirs_a[idx] = res; // FINAL → next frame's pass-1 temporal tap

    if (!di_valid(res) || res.unbiased_contribution_weight <= 0.0) { return vec3<f32>(0.0); }
    let smp = di_resolve(res.light_index, res.seed);
    let c = di_contribution(smp, p, n);
    let vis = di_visibility(p, n, smp.pos, res.light_index);
    // E estimate × visibility. `c.radiance` already folds cos_recv/dist²·inv_pdf·emissive_strength (no /π).
    return sanitize3(c.radiance * (res.unbiased_contribution_weight * vis) * light.gi_intensity);
}

// Frame-dependent in-4×4-block pixel shuffle (Solari `permute_pixel`). Decorrelates the temporal tap so a
// pixel doesn't re-consult ITS OWN previous reservoir every frame (which freezes it onto an early sample →
// the grain that fades in). Over frames it cycles through the local neighbourhood, folding light spatial
// reuse into the temporal step.
fn permute_pixel(pixel_id: vec2<u32>, frame_index: u32, vp: vec2<u32>) -> vec2<u32> {
    let r = frame_index;
    let offset = vec2<u32>(r & 3u, (r >> 2u) & 3u);
    var shifted = pixel_id + offset;
    shifted = shifted ^ vec2<u32>(3u);
    shifted = shifted - offset;
    return min(shifted, vp - vec2<u32>(1u));
}

// Reject reuse across a surface discontinuity (Solari `pixel_dissimilar`, world-space form): tangent-plane
// distance > ~1% of the camera distance, or normals more than ~60° apart. Keeps neighbour reuse on the same
// wall/face (smooths grain) but never leaks GI across depth/normal edges.
fn surfaces_dissimilar(p: vec3<f32>, n: vec3<f32>, op: vec3<f32>, on: vec3<f32>) -> bool {
    let tangent_plane_distance = abs(dot(n, op - p));
    // Cap the view-distance (if `gi_dissim_cap_dist > 0`) so the relative tangent threshold becomes ABSOLUTE
    // beyond that range — closes the thin-wall reuse leak at distance (see the field doc). 0 = uncapped Solari.
    let cap = select(1.0e30, restir_params.gi_dissim_cap_dist, restir_params.gi_dissim_cap_dist > 0.0);
    let view_dist = clamp(length(p - camera.cam_pos), 1.0e-3, cap);
    // Solari thresholds (gbuffer_utils.wgsl:45 parity): reject if the neighbour is >0.3% of view-distance out
    // of the tangent plane, or its normal is >90° away (`dot < 0`).
    //
    // THIN-WALL CAVEAT (post-D1a 0.05 m flip — corrected, was stale): the thinnest production wall is now a
    // 0.05 m voxel (Cornell WALL = 2 voxels = 0.1 m), NOT ">=0.4 m". The RELATIVE 0.003·view_dist tangent-plane
    // threshold equals the 0.05 m voxel at view_dist ≈ 16.7 m, so BEYOND ~16.7 m (well within the new 64 m LOD0
    // reach) a far-side-of-a-thin-wall neighbour is NOT rejected by this test → ReSTIR spatial reuse can leak GI
    // across a thin wall at distance. The world-cache thin-wall leak is handled at its source by the first-bounce
    // cell-size clamp in `query_world_cache`; this ReSTIR-reuse leak is a SEPARATE, currently-unguarded path.
    // The `gi_dissim_cap_dist` clamp above turns the relative threshold into an absolute tangent cap beyond that
    // range (≈0.003·cap), closing this leak. It's a TUNABLE (editor slider) defaulted conservatively, because an
    // absolute cap can over-reject genuine co-planar/slope reuse (increasing boil) — so the value wants eyeballing
    // on terrain (the voxel-rt-gi-noise 2.2.1 lessons); 0 restores the pure-Solari uncapped behaviour.
    return tangent_plane_distance / view_dist > 0.003 || dot(n, on) < 0.0;
}

// Uniform sample in a disk of `radius` pixels (concentric area-uniform), for spatial-neighbour selection.
fn sample_disk(radius: f32, rng: ptr<function, u32>) -> vec2<f32> {
    let r = radius * sqrt(rand_next(rng));
    let a = 6.2831853 * rand_next(rng);
    return vec2<f32>(r * cos(a), r * sin(a));
}

// Spatiotemporal-blue-noise value for the GI sample-direction Cranley–Patterson rotation. Returns the STBN
// `.xy` (uniform in [0,1)², blue in space AND time), indexed by pixel (wrapped to the mask) and frame (→ array
// layer). Mirrors `bevy_pbr::ssr`'s access. Falls back to white noise when the dummy 1×1 texture is bound (the
// real `stbn.ktx2` not yet resident) so the result is always well-defined.
fn stbn_rotation(pix: vec2<u32>, frame: u32, seed: u32) -> vec2<f32> {
    let dims = textureDimensions(stbn_tex);
    if (all(dims > vec2<u32>(1u))) {
        let layers = textureNumLayers(stbn_tex);
        return textureLoad(stbn_tex, pix % dims, i32(frame % layers), 0).xy;
    }
    return vec2<f32>(rand01(seed * 2u + 1u), rand01(seed * 2u + 2u));
}

// The GI dispatch viewport — half-res (gi_half_x/y) when half-res GI is on, else the full render viewport. The
// GI reservoir/surface buffers are indexed at THIS resolution (so the GI cores work at either res unchanged).
fn gi_vp() -> vec2<u32> {
    if (restir_params.gi_half != 0u) {
        return vec2<u32>(restir_params.gi_half_x, restir_params.gi_half_y);
    }
    return vec2<u32>(restir_params.viewport_x, restir_params.viewport_y);
}

// In-pixel sample offset for the half-res primary ray. Measured: a ROTATING 2×2 offset (one quadrant per frame)
// recovers spatial detail over ~4 frames BUT injects temporal variance (+~20% blotch on the boil meter); since
// the boil is the priority here and DLSS-RR recovers detail downstream, we sample the CENTRE (stable). The
// rotating variant `vec2(select(0.25,0.75,(frame&1)!=0), select(0.25,0.75,(frame&2)!=0))` is kept as a future
// knob for detail-over-boil scenes.
fn half_res_jitter(frame: u32) -> vec2<f32> {
    return vec2<f32>(0.5);
}

// PASS 1 (Solari `initial_and_temporal`): generate the initial RIS candidate, merge LAST frame's final
// reservoir for this surface (reprojected+permuted tap into `reservoirs_a`), and write the POST-TEMPORAL
// reservoir to `reservoirs_b` + this pixel's receiver surface to `surfaces_cur`. NO spatial reuse, NO shading
// here — pass 2 does same-frame spatial + the visibility-corrected resolve. `temporal_base` is the previous
// frame pixel this surface reprojects to (== `pix` for a still camera / the non-DLSS path); reprojection lets
// accumulation CONTINUE through motion (disocclusions are caught by the dissimilarity reject).
fn restir_p1_core(n: vec3<f32>, p: vec3<f32>, pix: vec2<u32>, temporal_base: vec2<i32>, seed: u32) {
    let vp = gi_vp(); // half-res when gi_half (the GI reservoirs/surfaces are indexed at this resolution)
    let idx = pix.y * vp.x + pix.x;
    surfaces_cur[idx] = PixelSurface(p, 1.0, n, 0.0); // this pixel's receiver surface (for neighbours/next frame)
    var rng = seed;
    let brdf = vec3<f32>(1.0); // receiver albedo factored out (applied by the caller)

    // ONE initial RIS candidate — a SINGLE first bounce (canonical ReSTIR / Solari `sample_gi`). The effective
    // sample count is built by the temporal + spatial reservoir REUSE below, NOT by a per-pixel RIS loop: that
    // is what ReSTIR is. Tracing exactly one bounce here (vs the old `gi_rays`-deep inlined trace/shade tree)
    // collapses the register pressure that bound this occupancy-limited pass — same converged GI, far higher
    // occupancy. The candidate counts as ONE frame (confidence 1) so the temporal reuse stays strong. `rng`
    // drives the merges' stochastic selection.
    //
    // SPATIOTEMPORAL BLUE NOISE rotation (Bevy `stbn.ktx2`). The Cranley–Patterson rotation of the GI sample
    // direction is drawn from an STBN mask — blue in space AND time. Blue-in-space decorrelates neighbours (so
    // spatial reuse integrates cleanly); blue-in-time makes each pixel's frame-to-frame sequence low-discrepancy
    // with NO coherent drift, so temporal reuse + DLSS-RR average it to a stable estimate instead of the
    // wandering mean white noise produced (the boil). (An earlier additive-R2 shortcut shifted EVERY pixel by the
    // same per-frame constant → the whole field drifted coherently and read as boil; a real STBN mask doesn't.)
    // Falls back to white noise when the texture isn't resident yet (`stbn_rotation`'s `dims > 1` guard).
    let rot = stbn_rotation(pix, restir_params.frame_index, seed);
    // A/B gate (2.2): when `wc.use_world_cache` is on (default), the bounce-HIT radiance is read from the
    // world-space radiance cache (pre-accumulated → low variance, multi-bounce in 2.3) instead of a fresh
    // single trace. The query LAZY-INSERTS, so this live path is ALSO what populates the cache (Solari's
    // query-driven fill). When off, the FRESH `reservoir_from_bounce` path runs — identical to pre-2.2
    // behaviour (minus the now-removed firefly clamp), and no query marks any cell alive, so the cache stays
    // idle (update/blend no-op) exactly like Phase 2.1. The LD direction stratifies the sampling either way.
    // GI initial reservoir: ONE fresh bounce (M=1 canonical ReSTIR). An M>1 same-receiver streaming-RIS loop
    // used to combine M candidates here, but it's removed: this pass is occupancy/register-pressure bound and
    // the loop held a reservoir + the merge state live ACROSS the heavy bounce trace, pinning restir_p1's
    // occupancy (the dominant GI kernel) — and the memory established CAMERA JITTER (now disabled), not a low
    // sample count, was the boil source M was masking. Straight-line M=1 measured −22% time / +6pt occupancy on
    // restir_p1 (Sponza atrium, Nsight). The temporal + spatial reservoir reuse builds the effective sample count.
    let use_cache = wc.use_world_cache != 0u;
    let drop_emissive = restir_params.di_enabled != 0u && wc.light_count != 0u;
    // M=1 straight-line GI candidate: trace ONE bounce and build `res` DIRECTLY from it, so no reservoir/merge
    // state is held live ACROSS the heavy bounce trace — that overlapping live state pinned restir_p1's
    // occupancy (the dominant, lowest-occupancy GI kernel). The old M>1 streaming-RIS loop is removed: the
    // memory established CAMERA JITTER (now disabled), not a low sample count, was the boil source, so M=1 is the
    // shipping default and `gi_initial_samples` no longer drives a per-frame candidate count.
    let dir = ld_uniform_hemisphere(n, 0u, 1u, rot);
    var res: Reservoir;
    if (use_cache) {
        res = reservoir_from_bounce_cached(p, n, dir, drop_emissive, &rng);
    } else {
        res = reservoir_from_bounce(p, n, dir);
    }
    // RIS source weight for one uniform-hemisphere sample: ucw = 1/pdf = 2π (set by the bounce fn); target =
    // luminance(L_o·cosθ) (brdf = 1; receiver albedo + 1/π applied by the resolve); weight_sum = target·ucw;
    // confidence_weight = 1 (one fresh frame). The temporal reuse below leans on this fresh estimate.
    let cwi = normalize(res.sample_point_world_position - p);
    res.confidence_weight = 1.0;
    res.weight_sum = restir_luminance(res.radiance * saturate(dot(cwi, n))) * res.unbiased_contribution_weight;

    // TEMPORAL reuse. CRUCIAL: read a PERMUTED previous-frame neighbour, NOT this pixel's own previous
    // reservoir — same-pixel feedback freezes each pixel onto an early sample (grain that fades in); the
    // permute decorrelates it (folds in genuinely new info each frame → variance falls, not grows). Reject
    // dissimilar surfaces (no GI leak across edges) and merge with the NEIGHBOUR's surface so the Jacobian is
    // correct. Read last frame's FINAL reservoir from `reservoirs_a`. Skipped on reset (camera move / re-pack).
    if (restir_params.reset == 0u) {
        // Reproject to where this surface was last frame, then permute (decorrelate). Off-screen → fall back
        // to the current pixel (best effort); a dissimilar surface (disocclusion) is rejected below.
        var tb = temporal_base;
        if (tb.x < 0 || tb.y < 0 || tb.x >= i32(vp.x) || tb.y >= i32(vp.y)) {
            tb = vec2<i32>(pix);
        }
        let tpix = permute_pixel(vec2<u32>(tb), restir_params.frame_index, vp);
        // Try the PERMUTED tap; if it lands on a dissimilar/invalid surface, fall back to the un-permuted
        // reprojected pixel (Solari's point-reprojection fallback). This matters at DLSS upscaling, where the
        // permute lands off-surface more often per render-res pixel — without it the history drops and boils.
        var tidx = tpix.y * vp.x + tpix.x;
        var surf = surfaces_prev[tidx];
        if (surf.valid <= 0.5 || surfaces_dissimilar(p, n, surf.world_position, surf.world_normal)) {
            tidx = u32(tb.y) * vp.x + u32(tb.x);
            surf = surfaces_prev[tidx];
        }
        if (surf.valid > 0.5 && !surfaces_dissimilar(p, n, surf.world_position, surf.world_normal)) {
            var temporal = reservoirs_a[tidx];
            temporal.confidence_weight = min(temporal.confidence_weight, restir_params.confidence_weight_cap);
            let merged =
                merge_reservoirs(res, p, n, brdf, temporal, surf.world_position, surf.world_normal, brdf, &rng);
            res = merged.merged_reservoir;
        }
    }

    // Robustness: never persist a non-finite contribution weight — a stored NaN/Inf is reused forever (a
    // permanent dead pixel). (A1's balance-heuristic form should prevent the source; this is belt-and-braces.)
    if (restir_isnan(res.unbiased_contribution_weight) || restir_isinf(res.unbiased_contribution_weight)) {
        res.unbiased_contribution_weight = 0.0;
    }
    reservoirs_b[idx] = res; // POST-TEMPORAL → the same-frame buffer pass 2's spatial reuse reads
}

// PASS 2 (Solari `spatial_and_shade`): start from this pixel's SAME-FRAME post-temporal reservoir
// (`reservoirs_b[idx]`), merge exactly ONE valid same-frame spatial neighbour (also from `reservoirs_b`),
// store the unbiased FINAL reservoir to `reservoirs_a` (history for next frame's pass 1), then shade a
// throwaway visibility-corrected copy and resolve the indirect irradiance (× albedo by the caller). Reading
// the SAME-FRAME post-temporal pool — rather than last frame's finals — decorrelates spatial reuse and
// converges shadows faster (no recursive last-frame feedback). Returns 0 for GI-off / misses.
fn restir_p2_core(n: vec3<f32>, p: vec3<f32>, pix: vec2<u32>, seed: u32) -> vec3<f32> {
    let vp = gi_vp(); // half-res when gi_half — the half-res GI spatial pass stores reservoirs_a at this res
    let idx = pix.y * vp.x + pix.x;
    var res = reservoirs_b[idx]; // this pixel's post-temporal reservoir (written by pass 1, this frame)

    // SPATIAL reuse via DEFENSIVE PAIRWISE MIS (RTXDI / Wyman 2023 course, Algo 7). The OLD path merged exactly
    // ONE neighbour because iterated pairwise *balance* merges double-count the canonical and amplify variance as
    // neighbours are added (the "more spatial → more boil" bug). Pairwise MIS instead forms ONE valid MIS
    // partition over {canonical + K neighbours} (Σm = 1 across the whole set), so combining K neighbours REDUCES
    // variance. Each neighbour is an ALREADY-TRACED reservoir, so this adds ~K effective samples per frame for
    // FREE — the cheap analogue of raising the initial sample count M. Two passes over the SAME disk-tap sequence
    // (re-seeded `rng`): pass A counts valid neighbours `n` (needed for the `M_i·n` balance scaling), pass B
    // streams them. A separate `rng_sel` drives reservoir selection so it can't desync the tap sequence.
    let canonical = res;
    let p_cc = restir_target_at(canonical.radiance, canonical.sample_point_world_position, p, n); // p̂_c(X_c)
    let m_c = canonical.confidence_weight;
    let spatial_seed = seed ^ 0x9E3779B9u;
    var rng_tap = spatial_seed;
    var valid_n = 0u;
    for (var s = 0u; s < restir_params.spatial_samples; s = s + 1u) {
        let off = sample_disk(restir_params.spatial_radius, &rng_tap);
        let npix = vec2<i32>(pix) + vec2<i32>(i32(round(off.x)), i32(round(off.y)));
        if (npix.x < 0 || npix.y < 0 || npix.x >= i32(vp.x) || npix.y >= i32(vp.y)) { continue; }
        let nidx = u32(npix.y) * vp.x + u32(npix.x);
        if (nidx == idx) { continue; }
        let nsurf = surfaces_cur[nidx];
        if (nsurf.valid > 0.5 && !surfaces_dissimilar(p, n, nsurf.world_position, nsurf.world_normal)) {
            valid_n = valid_n + 1u;
        }
    }
    if (valid_n > 0u) {
        let nf = f32(valid_n);
        var weight_sum = 0.0;
        var sel_target = 0.0;
        var m_total = 0.0;
        var canonical_weight = 0.0;
        var sel_pos = canonical.sample_point_world_position;
        var sel_nrm = canonical.sample_point_world_normal;
        var sel_rad = canonical.radiance;
        rng_tap = spatial_seed;          // re-seed → identical tap sequence
        var rng_sel = seed ^ 0x68E31DA4u; // independent selection stream
        for (var s = 0u; s < restir_params.spatial_samples; s = s + 1u) {
            let off = sample_disk(restir_params.spatial_radius, &rng_tap);
            let npix = vec2<i32>(pix) + vec2<i32>(i32(round(off.x)), i32(round(off.y)));
            if (npix.x < 0 || npix.y < 0 || npix.x >= i32(vp.x) || npix.y >= i32(vp.y)) { continue; }
            let nidx = u32(npix.y) * vp.x + u32(npix.x);
            if (nidx == idx) { continue; }
            let nsurf = surfaces_cur[nidx];
            if (!(nsurf.valid > 0.5) || surfaces_dissimilar(p, n, nsurf.world_position, nsurf.world_normal)) { continue; }
            let nb = reservoirs_b[nidx];
            let m_i = min(nb.confidence_weight, restir_params.confidence_weight_cap);
            let nb_pos = nsurf.world_position;
            let nb_n = nsurf.world_normal;
            // Four target evaluations; the Jacobian multiplies the two CROSS-receiver shifts only.
            let p_nn = restir_target_at(nb.radiance, nb.sample_point_world_position, nb_pos, nb_n);          // neighbour@neighbour
            let j_n = restir_jacobian(p, nb_pos, nb.sample_point_world_position, nb.sample_point_world_normal);
            let p_nc = restir_target_at(nb.radiance, nb.sample_point_world_position, p, n) * j_n;            // neighbour@canonical
            let j_c = restir_jacobian(nb_pos, p, canonical.sample_point_world_position, canonical.sample_point_world_normal);
            let p_cn = restir_target_at(canonical.radiance, canonical.sample_point_world_position, nb_pos, nb_n) * j_c; // canonical@neighbour
            let w0 = pairwise_mis(p_nn, p_nc, m_i * nf, m_c); // neighbour's MIS weight
            let w1 = pairwise_mis(p_cn, p_cc, m_i * nf, m_c); // canonical's per-pair share complement
            // Stream the neighbour (ris = m_i_weight · W_i · p̂_c(X_i)).
            let ris = w0 * nb.unbiased_contribution_weight * p_nc;
            weight_sum = weight_sum + ris;
            m_total = m_total + m_i * min(mis_m_factor(p_nn, p_nc), mis_m_factor(p_cn, p_cc));
            if (rand_next(&rng_sel) * weight_sum < ris) {
                sel_pos = nb.sample_point_world_position;
                sel_nrm = nb.sample_point_world_normal;
                sel_rad = nb.radiance;
                sel_target = p_nc;
            }
            canonical_weight = canonical_weight + (1.0 - w1);
        }
        // Stream the canonical LAST (its MIS weight = the accumulated per-pair shares).
        let ris_c = canonical_weight * canonical.unbiased_contribution_weight * p_cc;
        weight_sum = weight_sum + ris_c;
        m_total = m_total + m_c;
        if (rand_next(&rng_sel) * weight_sum < ris_c) {
            sel_pos = canonical.sample_point_world_position;
            sel_nrm = canonical.sample_point_world_normal;
            sel_rad = canonical.radiance;
            sel_target = p_cc;
        }
        res.sample_point_world_position = sel_pos;
        res.sample_point_world_normal = sel_nrm;
        res.radiance = sel_rad;
        res.confidence_weight = m_total;
        res.weight_sum = weight_sum;
        // Unbiased ucw: weight_sum / (p̂_selected · n)  — the `1/n` removes the `M_i·n` spread in the balance.
        // Floor the divisor: a grazing-angle selected sample gives a tiny-but-positive denormal `sel_target` that
        // passes `> 0` yet makes a huge finite ucw spike (a firefly the NaN/Inf guards don't catch). `1e-12` caps it.
        res.unbiased_contribution_weight = select(0.0, weight_sum / max(sel_target * nf, 1e-12), sel_target > 0.0);
    }

    // Store the UNBIASED reservoir (true ucw) BEFORE the visibility test — Solari's unbiased path. The stored
    // reservoir must remain an unbiased estimate of incident radiance, because NEIGHBOURS resample it next
    // frame; visibility is a per-RECEIVER shading correction only. Baking THIS pixel's occlusion into the
    // stored reservoir and then reusing it at other pixels is exactly what makes bright (e.g. red-wall)
    // samples diffuse across the buffer over frames (the leak). So: store first, then shade with a throwaway
    // visibility-corrected copy.
    if (restir_isnan(res.unbiased_contribution_weight) || restir_isinf(res.unbiased_contribution_weight)) {
        res.unbiased_contribution_weight = 0.0;
    }
    reservoirs_a[idx] = res; // FINAL → the history pool next frame's pass-1 temporal tap reads
    var shaded = res;
    if (shaded.confidence_weight > 0.0) {
        let origin = p + n * light.shadow_bias;
        let to_sample = shaded.sample_point_world_position - origin;
        let dist = length(to_sample);
        // Pull t_max back by a RELATIVE epsilon (sub-voxel at these scales) so the ray doesn't self-hit the
        // sample point's own surface. A FIXED `dist - shadow_bias` pull-back (one voxel near the wall) could
        // drop a near-floor occluder into the trimmed tail and DISARM the occlusion backstop, letting a sample
        // on the far side of a thin floor shade an interior face. Relative trim keeps the backstop armed.
        if (dist > 0.0 && trace_occluded(origin, to_sample / dist, 0.0, dist * (1.0 - 1.0e-3))) {
            shaded.unbiased_contribution_weight = 0.0;
        }
    }
    return restir_resolve_irradiance(shaded, p, n);
}

// ===== Screen-space radiance probes (group 4) — Lumen-style downsampled diffuse GI ============================
// Independent samples in a downsampled domain: each probe traces its OWN octahedral ray set (full sphere, fixed
// world frame) and projects to order-2 spherical harmonics. The SH projection IS the variance-reduction
// mechanism (a low-pass that discards the angular noise the boil meter measures), NOT an optimisation. Per pixel
// we bilaterally blend the 2×2 neighbour probes' SH and do ONE SH·cosine-lobe dot product. The probe ray is one
// bounce -> world radiance cache (the level-2 cache) so it's cheap + multi-bounce. docs/SCREEN_PROBE_PLAN.md.
//
// NOT to be confused with the test-only `restir_probe`/`ProbePoint` estimator (group-0 8..11) — different thing.
struct ScreenProbeParams {
    grid_x: u32,        // probe grid dims = ceil(viewport / probe_size)
    grid_y: u32,
    probe_size: u32,    // pixels per probe cell
    oct_res: u32,       // octahedral resolution; dirs traced per probe = oct_res²
    viewport_x: u32,
    viewport_y: u32,
    reset: u32,
    frame_index: u32,
    enabled: u32,       // gi_mode: 1 = probes drive diffuse GI, 0 = ReSTIR
    temporal: u32,      // 1 = blend prev-frame SH history (P3)
    _pad0: u32,
    _pad1: u32,
};
@group(4) @binding(0) var<uniform> probe_params: ScreenProbeParams;
struct ScreenProbeHeader { world_pos: vec3<f32>, valid: f32, world_normal: vec3<f32>, view_z: f32 };
@group(4) @binding(1) var<storage, read_write> probe_headers: array<ScreenProbeHeader>;
@group(4) @binding(2) var<storage, read_write> probe_sh: array<vec4<f32>>;          // 9 coeffs/probe (.xyz=RGB)
@group(4) @binding(3) var<storage, read_write> probe_sh_history: array<vec4<f32>>;  // prev frame (temporal reuse)

// Order-2 real SH basis (9 coeffs) for a unit direction.
fn sh9(d: vec3<f32>) -> array<f32, 9> {
    return array<f32, 9>(
        0.282095,
        0.488603 * d.y,
        0.488603 * d.z,
        0.488603 * d.x,
        1.092548 * d.x * d.y,
        1.092548 * d.y * d.z,
        0.315392 * (3.0 * d.z * d.z - 1.0),
        1.092548 * d.x * d.z,
        0.546274 * (d.x * d.x - d.y * d.y),
    );
}

// Diffuse irradiance E(n) from 9 incoming-radiance SH coeffs via the cosine-lobe convolution (Ramamoorthi:
// A0=π, A1..3=2π/3, A4..8=π/4). E = ∫ L(d)·max(cosθ,0) dω. `c` = the 9 blended coeffs.
fn sh_irradiance_local(c: array<vec4<f32>, 9>, n: vec3<f32>) -> vec3<f32> {
    let y = sh9(n);
    var e = 3.14159265 * c[0].xyz * y[0];
    e += 2.0943951 * (c[1].xyz * y[1] + c[2].xyz * y[2] + c[3].xyz * y[3]);
    e += 0.785398163 * (c[4].xyz * y[4] + c[5].xyz * y[5] + c[6].xyz * y[6]
                      + c[7].xyz * y[7] + c[8].xyz * y[8]);
    return e;
}

// Per-pixel probe integration (called from shade): bilinear 2×2 probe gather, bilateral depth/normal reject,
// SH·cosine-lobe. Returns indirect irradiance/π × gi_intensity — the SAME contract as `restir_p2_core` (the
// caller multiplies receiver albedo). Falls back to 0 when no nearby probe is valid (edge — P4 adaptive fills).
fn screen_probe_integrate(n: vec3<f32>, p: vec3<f32>, view_z: f32, pix: vec2<u32>) -> vec3<f32> {
    let ps = f32(probe_params.probe_size);
    // Probe cell c is centred at pixel c*ps + ps/2 → this pixel's continuous probe-grid coord:
    let pc = (vec2<f32>(pix) - vec2<f32>(ps * 0.5)) / ps;
    let base = vec2<i32>(floor(pc));
    let fr = pc - floor(pc);
    var acc = array<vec4<f32>, 9>(
        vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
        vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    var wsum = 0.0;
    for (var dj = 0; dj < 2; dj = dj + 1) {
        for (var di = 0; di < 2; di = di + 1) {
            let cell = base + vec2<i32>(di, dj);
            if (cell.x < 0 || cell.y < 0 || cell.x >= i32(probe_params.grid_x) || cell.y >= i32(probe_params.grid_y)) {
                continue;
            }
            let pidx = u32(cell.y) * probe_params.grid_x + u32(cell.x);
            let h = probe_headers[pidx];
            if (h.valid < 0.5) { continue; }
            let bw = select(1.0 - fr.x, fr.x, di == 1) * select(1.0 - fr.y, fr.y, dj == 1);
            let nw = max(0.0, dot(h.world_normal, n));
            // depth/plane reject: distance of p from the probe's tangent plane, relative to view depth.
            let plane = abs(dot(h.world_pos - p, h.world_normal)) / max(abs(view_z), 0.1);
            let dw = exp(-plane * 8.0);
            let w = bw * nw * nw * dw;
            if (w <= 0.0) { continue; }
            for (var c = 0u; c < 9u; c = c + 1u) {
                acc[c] = acc[c] + probe_sh[pidx * 9u + c] * w;
            }
            wsum = wsum + w;
        }
    }
    // Edge fallback (M3): the bilinear 2×2 may have NO valid probe (silhouette / disocclusion). Widen to a 5×5
    // search for the nearest normal-matching valid probe and use its SH directly, so edges aren't black (until
    // P4 adaptive probes place dedicated edge probes).
    if (wsum <= 0.0) {
        var best = 1.0e30;
        var found = false;
        for (var dj = -2; dj <= 2; dj = dj + 1) {
            for (var di = -2; di <= 2; di = di + 1) {
                let cell = base + vec2<i32>(di, dj);
                if (cell.x < 0 || cell.y < 0 || cell.x >= i32(probe_params.grid_x) || cell.y >= i32(probe_params.grid_y)) {
                    continue;
                }
                let pidx = u32(cell.y) * probe_params.grid_x + u32(cell.x);
                let h = probe_headers[pidx];
                if (h.valid < 0.5 || dot(h.world_normal, n) < 0.5) { continue; }
                let d = length(h.world_pos - p);
                if (d < best) {
                    best = d;
                    found = true;
                    for (var c = 0u; c < 9u; c = c + 1u) { acc[c] = probe_sh[pidx * 9u + c]; }
                }
            }
        }
        if (!found) { return vec3<f32>(0.0); }
        return (sh_irradiance_local(acc, n) / 3.14159265) * light.gi_intensity;
    }
    let inv = 1.0 / wsum;
    for (var c = 0u; c < 9u; c = c + 1u) { acc[c] = acc[c] * inv; }
    return (sh_irradiance_local(acc, n) / 3.14159265) * light.gi_intensity;
}

// PASS (screen probes): one thread per probe cell. Place the probe on the surface at the cell's centre pixel,
// trace oct_res² full-sphere directions (upper hemisphere contributes), and project incoming radiance to order-2
// SH. Each direction = ONE bounce reading the world cache (`reservoir_from_bounce_cached`) — independent samples.
@compute @workgroup_size(8, 8, 1)
fn screen_probe_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= probe_params.grid_x || gid.y >= probe_params.grid_y) { return; }
    let pidx = gid.y * probe_params.grid_x + gid.x;
    let b = pidx * 9u;
    let ps = probe_params.probe_size;
    let cx = gid.x * ps + ps / 2u;
    let cy = gid.y * ps + ps / 2u;
    var hdr: ScreenProbeHeader;
    hdr.valid = 0.0;
    if (cx < probe_params.viewport_x && cy < probe_params.viewport_y) {
        let ray = restir_primary_ray(vec3<u32>(cx, cy, 0u));
        let r = trace(ray[0], ray[1], 0.0, camera.t_max);
        if (r.hit != 0u) {
            let p = ray[0] + ray[1] * r.t;
            let n = r.normal;
            hdr.world_pos = p;
            hdr.world_normal = n;
            hdr.view_z = r.t;
            hdr.valid = 1.0;
            let ores = probe_params.oct_res;
            let nd = ores * ores;
            let inv_n = 1.0 / f32(nd);
            let inv_dw = 12.566370614 * inv_n; // 4π / N — EXACT, because the Fibonacci sphere is equal-area
            let drop_emissive = restir_params.di_enabled != 0u && wc.light_count != 0u;
            var rng = (pidx * 9781u + probe_params.frame_index * 26699u) | 1u;
            var c0 = vec3<f32>(0.0); var c1 = vec3<f32>(0.0); var c2 = vec3<f32>(0.0);
            var c3 = vec3<f32>(0.0); var c4 = vec3<f32>(0.0); var c5 = vec3<f32>(0.0);
            var c6 = vec3<f32>(0.0); var c7 = vec3<f32>(0.0); var c8 = vec3<f32>(0.0);
            for (var i = 0u; i < nd; i = i + 1u) {
                // Spherical Fibonacci direction (equal-area by construction → the 4π/N weight is exact, no
                // octahedral area distortion). Fixed world frame (shared across probes/frames for P3 filtering).
                let z = 1.0 - (2.0 * f32(i) + 1.0) * inv_n;
                let rr = sqrt(max(0.0, 1.0 - z * z));
                let phi = f32(i) * 2.39996323; // golden angle
                let dir = vec3<f32>(rr * cos(phi), rr * sin(phi), z);
                if (dot(dir, n) <= 0.0) { continue; }
                let L = reservoir_from_bounce_cached(p, n, dir, drop_emissive, &rng).radiance;
                let y = sh9(dir);
                c0 += L * (y[0] * inv_dw); c1 += L * (y[1] * inv_dw); c2 += L * (y[2] * inv_dw);
                c3 += L * (y[3] * inv_dw); c4 += L * (y[4] * inv_dw); c5 += L * (y[5] * inv_dw);
                c6 += L * (y[6] * inv_dw); c7 += L * (y[7] * inv_dw); c8 += L * (y[8] * inv_dw);
            }
            // Light TEMPORAL accumulation: blend the prev-frame SH (probe_sh_history at this cell) into the raw
            // SH when the surface matches (a position/normal validity reject — NOT reprojected, so on camera
            // motion the cell's world point changes, the reject fires, and the probe falls back to fresh: no
            // smear/ghosting, just noisier during motion which DLSS-RR cleans). Validity meta (view_z, normal) is
            // packed into the unused SH `.w` lanes of the history — no extra buffer. Reset → fresh.
            var a = 1.0; // history weight of the NEW sample (1 = no temporal)
            if (probe_params.temporal != 0u && probe_params.reset == 0u) {
                let h0 = probe_sh_history[b + 0u];
                let prev_vz = h0.w;
                let prev_n = vec3<f32>(probe_sh_history[b + 1u].w, probe_sh_history[b + 2u].w, probe_sh_history[b + 3u].w);
                let z_ok = prev_vz > 0.0 && abs(prev_vz - r.t) < 0.05 * r.t;
                if (z_ok && dot(prev_n, n) > 0.9) {
                    a = 0.1; // ~10-frame light history; DLSS-RR does the rest of the temporal lift (anti-ghost)
                    c0 = mix(probe_sh_history[b + 0u].xyz, c0, a);
                    c1 = mix(probe_sh_history[b + 1u].xyz, c1, a);
                    c2 = mix(probe_sh_history[b + 2u].xyz, c2, a);
                    c3 = mix(probe_sh_history[b + 3u].xyz, c3, a);
                    c4 = mix(probe_sh_history[b + 4u].xyz, c4, a);
                    c5 = mix(probe_sh_history[b + 5u].xyz, c5, a);
                    c6 = mix(probe_sh_history[b + 6u].xyz, c6, a);
                    c7 = mix(probe_sh_history[b + 7u].xyz, c7, a);
                    c8 = mix(probe_sh_history[b + 8u].xyz, c8, a);
                }
            }
            // Write the FINAL SH to probe_sh (integration reads it) + history (next frame), with validity meta
            // (view_z + normal) in the .w lanes of coeffs 0..3.
            probe_sh[b + 0u] = vec4<f32>(c0, 0.0); probe_sh[b + 1u] = vec4<f32>(c1, 0.0);
            probe_sh[b + 2u] = vec4<f32>(c2, 0.0); probe_sh[b + 3u] = vec4<f32>(c3, 0.0);
            probe_sh[b + 4u] = vec4<f32>(c4, 0.0); probe_sh[b + 5u] = vec4<f32>(c5, 0.0);
            probe_sh[b + 6u] = vec4<f32>(c6, 0.0); probe_sh[b + 7u] = vec4<f32>(c7, 0.0);
            probe_sh[b + 8u] = vec4<f32>(c8, 0.0);
            probe_sh_history[b + 0u] = vec4<f32>(c0, r.t);
            probe_sh_history[b + 1u] = vec4<f32>(c1, n.x);
            probe_sh_history[b + 2u] = vec4<f32>(c2, n.y);
            probe_sh_history[b + 3u] = vec4<f32>(c3, n.z);
            probe_sh_history[b + 4u] = vec4<f32>(c4, 0.0); probe_sh_history[b + 5u] = vec4<f32>(c5, 0.0);
            probe_sh_history[b + 6u] = vec4<f32>(c6, 0.0); probe_sh_history[b + 7u] = vec4<f32>(c7, 0.0);
            probe_sh_history[b + 8u] = vec4<f32>(c8, 0.0);
        }
    }
    probe_headers[pidx] = hdr;
    if (hdr.valid < 0.5) {
        for (var c = 0u; c < 9u; c = c + 1u) {
            probe_sh[b + c] = vec4<f32>(0.0);
            probe_sh_history[b + c] = vec4<f32>(0.0);
        }
    }
}

// HALF-RES GI resolve (SOTA, not bilateral-on-color): for a FULL-res pixel, bilinearly gather the 2×2 half-res
// final GI reservoirs (`reservoirs_a`) and RE-RESOLVE each against THIS pixel's own normal/position — the real
// sample direction's cosθ — weighted by depth/normal similarity. Reconstructs from samples → stays sharp (no
// SH/colour smoothing). Half-res surfaces (`surfaces_cur`, written by the half-res pass-1) supply the geometry.
fn restir_gi_gather(n: vec3<f32>, p: vec3<f32>, pix: vec2<u32>) -> vec3<f32> {
    let hvp = vec2<u32>(restir_params.gi_half_x, restir_params.gi_half_y);
    let hc = (vec2<f32>(pix) - 0.5) * 0.5; // full-res pixel → continuous half-res grid coord
    let base = vec2<i32>(floor(hc));
    let fr = hc - floor(hc);
    let recv_z = max(length(camera.cam_pos - p), 0.1);
    let origin = p + n * light.shadow_bias;
    var acc = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var dj = 0; dj < 2; dj = dj + 1) {
        for (var di = 0; di < 2; di = di + 1) {
            let cell = base + vec2<i32>(di, dj);
            if (cell.x < 0 || cell.y < 0 || cell.x >= i32(hvp.x) || cell.y >= i32(hvp.y)) { continue; }
            let hidx = u32(cell.y) * hvp.x + u32(cell.x);
            let surf = surfaces_cur[hidx];
            if (surf.valid < 0.5) { continue; }
            let bw = select(1.0 - fr.x, fr.x, di == 1) * select(1.0 - fr.y, fr.y, dj == 1);
            let nw = max(0.0, dot(surf.world_normal, n));
            let plane = abs(dot(surf.world_position - p, n)) / recv_z;
            let w = bw * nw * nw * exp(-plane * 16.0);
            if (w <= 0.0) { continue; }
            // Per-full-res-pixel visibility per reservoir (the full-res path shadow-tests its resolve too): keeps
            // contact shadows sharp + stops GI leaking through occluders the half-res sample didn't see. Occlusion
            // rays are cheap vs the GI bounces we saved (3/4 at half-res).
            let res = reservoirs_a[hidx];
            let to_s = res.sample_point_world_position - origin;
            let ds = length(to_s);
            var vis = 1.0;
            if (ds > 0.0 && trace_occluded(origin, to_s / ds, 0.0, ds * (1.0 - 1.0e-3))) { vis = 0.0; }
            acc += restir_resolve_irradiance(res, p, n) * (w * vis); // reservoir-aware: resolve per full-res n
            wsum += w;
        }
    }
    if (wsum <= 0.0) {
        // Fallback: nearest same-orientation valid half-res reservoir in a 4×4, re-resolved here (no black edges).
        var best = 1.0e30;
        var out = vec3<f32>(0.0);
        for (var dj = -1; dj <= 2; dj = dj + 1) {
            for (var di = -1; di <= 2; di = di + 1) {
                let cell = base + vec2<i32>(di, dj);
                if (cell.x < 0 || cell.y < 0 || cell.x >= i32(hvp.x) || cell.y >= i32(hvp.y)) { continue; }
                let hidx = u32(cell.y) * hvp.x + u32(cell.x);
                let surf = surfaces_cur[hidx];
                if (surf.valid < 0.5 || dot(surf.world_normal, n) < 0.5) { continue; }
                let d = length(surf.world_position - p);
                if (d < best) { best = d; out = restir_resolve_irradiance(reservoirs_a[hidx], p, n); }
            }
        }
        return out;
    }
    return acc / wsum;
}

// Like `shade`, but the indirect term comes from pass 2's reservoir resolve (`restir_p2_core`) instead of
// `gather_gi`. Direct sun + AO + emissive glow are unchanged. Called from the pass-2 entries only (pass 1 has
// already filled the reservoir + surface for this pixel this frame).
// Shade a primary hit: sun direct + shadow + diffuse-indirect (ReSTIR GI / probes) + ReSTIR DI + emissive glow,
// all at the hit `(n, p)`, modulated by `albedo`. The primary ray is un-jittered on both paths, so the receiver
// is a stable per-pixel point (no jitter wander → no boil; DLSS-RR runs as a denoiser).
fn shade_restir_p2(albedo: vec3<f32>, n: vec3<f32>, p: vec3<f32>, emissive: vec3<f32>, pix: vec2<u32>, seed: u32) -> vec3<f32> {
    let origin = p + n * light.shadow_bias;
    let to_sun = -light.sun_direction;
    let ndotl = max(dot(n, to_sun), 0.0);
    var shadow = 1.0;
    if (ndotl > 0.0) {
        if (trace_occluded(origin, to_sun, 0.0, 1.0e4)) {
            shadow = 0.0;
        }
    }
    let direct = light.sun_color * (light.sun_intensity * ndotl * shadow);
    // Diffuse indirect: screen-space radiance probes (gi_mode) or the per-pixel ReSTIR reservoir resolve. Both
    // return irradiance/π with receiver albedo factored out, so the × albedo is shared.
    var indirect: vec3<f32>;
    if (probe_params.enabled != 0u) {
        indirect = screen_probe_integrate(n, p, length(camera.cam_pos - p), pix) * albedo;
    } else if (restir_params.gi_half != 0u) {
        indirect = restir_gi_gather(n, p, pix) * albedo; // half-res reservoirs, re-resolved at full-res
    } else {
        indirect = restir_p2_core(n, p, pix, seed) * albedo;
    }
    // GI 4.0: DIRECT emissive-voxel light via screen-space ReSTIR DI (low variance — resampled + reused +
    // visibility-checked), replacing the high-variance `emissive(hit)`-via-random-bounce. `di_p2_core` returns
    // the reflected E (brdf factored out), so apply the receiver albedo here like the sun direct. 0 when DI off.
    // DI is skipped under half-res GI (v1: GI-only at half-res; DI stays full-res in a later promotion).
    var direct_emitter = vec3<f32>(0.0);
    if (restir_params.gi_half == 0u) {
        direct_emitter = di_p2_core(n, p, pix, seed);
    }
    let glow = emissive * light.emissive_strength;
    // No flat ambient (see `shade`): the ReSTIR GI provides the fill; occluded → dark, matching Solari.
    return albedo * (direct + direct_emitter) + indirect + glow;
}

// Screen-space reprojection: the previous-frame pixel that the world point `p` projected to, using the
// UN-jittered previous clip. For a still camera this is the current pixel. (Caller supplies the prev clip.)
fn reproject_pixel(p: vec3<f32>, prev_clip_from_world: mat4x4<f32>, vp: vec2<u32>) -> vec2<i32> {
    let prev_clip = prev_clip_from_world * vec4<f32>(p, 1.0);
    let prev_ndc = prev_clip.xy / prev_clip.w;
    let prev_uv = prev_ndc * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    // ROUND to the nearest previous-frame pixel centre (Solari `load_temporal_reservoir`). `floor` biases the
    // tap up to a full pixel toward the origin — and at DLSS upscaling modes one render-res pixel = several
    // output pixels, so a `floor` bias visibly de-stabilises the temporal reuse (clean at DLAA, boils at
    // Quality/Performance). `prev_uv*vp - 0.5` is the pixel-centre coordinate; round it to the pixel index.
    return vec2<i32>(round(prev_uv * vec2<f32>(vp) - vec2<f32>(0.5)));
}

// Primary ray for a pixel in a viewport `vp` with an in-pixel sample `offset` (0.5 = centre). Half-res GI passes
// pass `vp = gi_vp()` and a ROTATING `offset` (different sub-pixel of the 2×2 each frame) so temporal reuse
// recovers TRUE full-res detail over ~4 frames (kajiya), not interpolation.
fn restir_primary_ray_vp(gid: vec3<u32>, vp: vec2<u32>, offset: vec2<f32>) -> array<vec3<f32>, 2> {
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + offset) / vec2<f32>(vp);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let near = camera.world_from_clip * vec4<f32>(ndc, 1.0, 1.0);
    let world_near = near.xyz / near.w;
    return array<vec3<f32>, 2>(camera.cam_pos, normalize(world_near - camera.cam_pos));
}

// Shared primary-ray setup for the full-res ReSTIR entries (pixel centre, full viewport). Un-jittered on both
// paths (the DLSS path disables jitter), so the primary hit is a stable per-pixel point — it IS the GI/DI receiver.
fn restir_primary_ray(gid: vec3<u32>) -> array<vec3<f32>, 2> {
    return restir_primary_ray_vp(gid, camera.viewport, vec2<f32>(0.5));
}

// ===== PASS 1 entries: trace the primary ray, fill `reservoirs_b` (post-temporal) + `surfaces_cur`. No
// shading, no out_tex, no guides — pass 2 re-traces the primary ray and does the spatial reuse + shading. =====

// Non-DLSS pass 1 — reproject the temporal tap via the UN-jittered previous clip (`camera.prev_clip_from_world`)
// so reservoir accumulation continues under camera motion instead of resetting (disocclusions on fast motion
// caught by the `surfaces_dissimilar` reject in `restir_p1_core`). Same contract as `restir_dlss_p1`, just
// without DLSS jitter. The reservoir `reset` flag now fires only on first-frame / resolution change.
@compute @workgroup_size(8, 8, 1)
fn restir_p1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vp = gi_vp(); // half-res when gi_half (dispatched at gi_vp; the GI buffers are indexed at this res)
    if (gid.x >= vp.x || gid.y >= vp.y) { return; }
    let idx = gid.y * vp.x + gid.x;
    reservoirs_b[idx] = empty_reservoir(); // default for misses / debug; overwritten for lit hits
    di_reservoirs_b[idx] = di_empty();     // DI default for misses
    surfaces_cur[idx] = PixelSurface(vec3<f32>(0.0), 0.0, vec3<f32>(0.0), 0.0); // invalid until a lit hit
    // Half-res: ROTATE the in-pixel sample across the 2×2 per frame so temporal reuse recovers full-res detail.
    let off = select(vec2<f32>(0.5), half_res_jitter(light.frame_index), restir_params.gi_half != 0u);
    let ray = restir_primary_ray_vp(gid, vp, off);
    let r = trace(ray[0], ray[1], 0.0, camera.t_max);
    if (r.hit != 0u) {
        let p = ray[0] + ray[1] * r.t;
        let seed = (gid.x * 1973u + gid.y * 9277u + light.frame_index * 26699u) | 1u;
        let temporal_base = reproject_pixel(p, camera.prev_clip_from_world, vp);
        // Skip the per-pixel ReSTIR GI (the M-bounce candidate gen) when screen probes drive the diffuse — shade
        // reads the probe SH instead, so the reservoir work is pure waste. DI still runs (probes are diffuse-only).
        if (probe_params.enabled == 0u) {
            restir_p1_core(r.normal, p, gid.xy, temporal_base, seed);
        }
        // DI runs full-res; it is SKIPPED in the half-res GI pass (v1 — promoted to a full-res DI pass later).
        if (restir_params.gi_half == 0u) {
            di_p1_core(r.normal, p, gid.xy, temporal_base, seed);
        }
    }
}

// PASS 1.5 (half-res GI only): same-frame spatial reuse at HALF res → store the final GI reservoir to
// `reservoirs_a`. Runs ONLY in half-res mode (full-res does the spatial inline in pass 2). The full-res shade
// then reservoir-resolve-gathers these. Re-traces the (jittered) half-res primary to recover this pixel's n/p.
@compute @workgroup_size(8, 8, 1)
fn restir_gi_spatial(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Runs for half-res GI (writes half-res reservoirs_a) AND the full-res spatial-average filter (writes full-res
    // reservoirs_a so the shade can average the POST-SPATIAL finals). `gi_vp()` = half or full accordingly.
    if (restir_params.gi_half == 0u) { return; }
    let vp = gi_vp();
    if (gid.x >= vp.x || gid.y >= vp.y) { return; }
    let idx = gid.y * vp.x + gid.x;
    reservoirs_a[idx] = empty_reservoir();
    if (probe_params.enabled != 0u) { return; }
    let ray = restir_primary_ray_vp(gid, vp, half_res_jitter(light.frame_index));
    let r = trace(ray[0], ray[1], 0.0, camera.t_max);
    if (r.hit != 0u) {
        let p = ray[0] + ray[1] * r.t;
        let seed = (gid.x * 1973u + gid.y * 9277u + light.frame_index * 26699u) | 1u;
        _ = restir_p2_core(r.normal, p, gid.xy, seed); // stores reservoirs_a[idx] (final); resolve ignored
    }
}

// DLSS pass 1 — reproject the temporal tap via the UN-jittered previous clip so accumulation continues under
// camera motion (disocclusions caught by the dissimilarity reject).
@compute @workgroup_size(8, 8, 1)
fn restir_dlss_p1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vp = gi_vp(); // half-res when gi_half
    if (gid.x >= vp.x || gid.y >= vp.y) { return; }
    let idx = gid.y * vp.x + gid.x;
    reservoirs_b[idx] = empty_reservoir();
    di_reservoirs_b[idx] = di_empty();
    surfaces_cur[idx] = PixelSurface(vec3<f32>(0.0), 0.0, vec3<f32>(0.0), 0.0);
    // Un-jittered primary ray (the render disables jitter) → stable per-pixel receiver. Half-res still rotates the
    // in-pixel sample across frames to recover detail (half-res is a non-RR knob).
    let off = select(vec2<f32>(0.5), half_res_jitter(light.frame_index), restir_params.gi_half != 0u);
    let ray = restir_primary_ray_vp(gid, vp, off);
    let r = trace(ray[0], ray[1], 0.0, camera.t_max);
    if (r.hit != 0u) {
        let p = ray[0] + ray[1] * r.t;
        let seed = (gid.x * 1973u + gid.y * 9277u + light.frame_index * 26699u) | 1u;
        let temporal_base = reproject_pixel(p, dlss_cam.motion_prev, vp);
        if (probe_params.enabled == 0u) {
            restir_p1_core(r.normal, p, gid.xy, temporal_base, seed);
        }
        if (restir_params.gi_half == 0u) {
            di_p1_core(r.normal, p, gid.xy, temporal_base, seed);
        }
    }
}

// ===== PASS 2 entries: re-trace the primary ray, do same-frame spatial reuse + shading from `reservoirs_b`,
// store the final reservoir to `reservoirs_a`, write out_tex (+ history blend non-DLSS / + guides DLSS). The
// re-trace (vs threading the pass-1 surface through a wider buffer) keeps `out_tex` write-only and the surface
// buffer 32 B — one extra primary trace per pixel, negligible on the target GPU. =====

// Non-DLSS pass 2: shade + the on-top history accumulation that further smooths the (already low-variance)
// ReSTIR output. Carries the debug-view selector (debug_view==5 GI-only = pass-2 reservoir resolve).
@compute @workgroup_size(8, 8, 1)
fn restir_p2(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= camera.viewport.x || gid.y >= camera.viewport.y) { return; }
    let idx = gid.y * camera.viewport.x + gid.x;
    // Under half-res GI OR the full-res spatial-average filter, `reservoirs_a` holds the POST-SPATIAL finals
    // (written by `restir_gi_spatial`); the full-res shade only READS them (gather/average), so must NOT clear
    // them here (the idx range overlaps). Only clear when restir_p2 itself owns the per-pixel final write.
    if (restir_params.gi_half == 0u) {
        reservoirs_a[idx] = empty_reservoir(); // default for misses / debug; overwritten for lit hits
        di_reservoirs_a[idx] = di_empty();
    }
    let ray = restir_primary_ray(gid);
    let ro = ray[0];
    let rd = ray[1];
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / vec2<f32>(camera.viewport);
    let r = trace(ro, rd, 0.0, camera.t_max);

    if (light.debug_view != 0u) {
        let dpx = vec2<i32>(i32(gid.x), i32(gid.y));
        var gi = vec3<f32>(0.0);
        if (r.hit != 0u && light.debug_view == 5u) {
            let p = ro + rd * r.t;
            let seed = (gid.x * 1973u + gid.y * 9277u + light.frame_index * 26699u) | 1u;
            // GI-only debug = the LIVE diffuse path: probes (gi_mode) / half-res gather / full-res reservoir.
            if (probe_params.enabled != 0u) {
                gi = screen_probe_integrate(r.normal, p, length(camera.cam_pos - p), gid.xy);
            } else if (restir_params.gi_half != 0u) {
                gi = restir_gi_gather(r.normal, p, gid.xy);
            } else {
                gi = restir_p2_core(r.normal, p, gid.xy, seed);
            }
        } else if (r.hit != 0u && light.debug_view == 8u) {
            let p = ro + rd * r.t;
            let seed = (gid.x * 1973u + gid.y * 9277u + light.frame_index * 26699u) | 1u;
            gi = di_p2_core(r.normal, p, gid.xy, seed); // DI-only debug = the direct-emitter reservoir estimate
        }
        textureStore(out_tex, dpx, vec4<f32>(debug_overlay_color(r, ro, rd, gi), 1.0));
        return;
    }

    var color: vec4<f32>;
    if (r.hit != 0u) {
        let p = ro + rd * r.t;
        let seed = (gid.x * 1973u + gid.y * 9277u + light.frame_index * 26699u) | 1u;
        let lit = shade_restir_p2(r.color.rgb, r.normal, p, r.emissive, gid.xy, seed);
        color = vec4<f32>(lit, 1.0);
    } else {
        color = vec4<f32>(sky_radiance(rd), 1.0);
    }

    let prev = textureSampleLevel(history_tex, history_sampler, uv, 0.0).rgb;
    let w = clamp(camera.accum_weight, 0.0, 1.0);
    let accumulated = mix(prev, color.rgb, w);
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(accumulated, color.a));
}

// DLSS pass 2: shade + write the DLSS-RR guides. RR is fed a LOW-VARIANCE indirect term (the reservoir
// integrated many frames + same-frame spatial), so it only has to clean a near-converged signal.
@compute @workgroup_size(8, 8, 1)
fn restir_dlss_p2(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= camera.viewport.x || gid.y >= camera.viewport.y) { return; }
    let idx = gid.y * camera.viewport.x + gid.x;
    // Under half-res GI, `reservoirs_a` holds the HALF-res finals (from `restir_gi_spatial`); the full-res shade
    // only READS them via the gather, so must NOT clear them (the full-res idx range overlaps the half-res).
    if (restir_params.gi_half == 0u) {
        reservoirs_a[idx] = empty_reservoir();
        di_reservoirs_a[idx] = di_empty();
    }
    let px = vec2<i32>(i32(gid.x), i32(gid.y));
    let ray = restir_primary_ray(gid); // un-jittered (jitter disabled) — gbuffer/depth/albedo + lighting receiver
    let ro = ray[0];
    let rd = ray[1];
    let r = trace(ro, rd, 0.0, camera.t_max);
    if (r.hit != 0u) {
        let p = ro + rd * r.t;
        let seed = (gid.x * 1973u + gid.y * 9277u + light.frame_index * 26699u) | 1u;
        // Un-jittered primary hit = a stable per-pixel point → all lighting (direct + GI + DI) is jitter-free.
        let lit = shade_restir_p2(r.color.rgb, r.normal, p, r.emissive, gid.xy, seed);
        textureStore(out_tex, px, vec4<f32>(lit, 1.0));
        textureStore(out_diffuse_albedo, px, vec4<f32>(r.color.rgb, 1.0));
        // Specular albedo = 0: this renderer shades DIFFUSE-only (sun direct + ReSTIR GI + DI + emissive, no
        // specular lobe — see `shade_restir_p2`). The old constant 0.04 told DLSS-RR a dielectric specular lobe
        // existed and made it demodulate the noisy color by (diffuse + 0.04) instead of diffuse alone — a large
        // error on DARK surfaces (albedo 0.04 ⇒ divide by 0.08 ⇒ half the lighting attributed), so RR denoised
        // an inconsistent signal. 0 makes the guide match the actual diffuse-only color: color/albedo = lighting.
        textureStore(out_specular_albedo, px, vec4<f32>(vec3<f32>(0.0), 1.0));
        textureStore(out_normal_roughness, px, vec4<f32>(r.normal, 1.0)); // roughness 1.0 (.w): fully diffuse
        // Depth + motion are both UN-jittered (this renderer disables camera jitter): motion `(cur−prev)·(0.5,−0.5)`
        // ⇒ exactly zero for a static camera; depth is the reverse-Z `z/w` of the same un-jittered hit.
        let depth_clip = dlss_cam.depth_clip_from_world * vec4<f32>(p, 1.0);
        textureStore(out_dlss_depth, px, vec4<f32>(depth_clip.z / depth_clip.w, 0.0, 0.0, 0.0));
        let prev_clip = dlss_cam.motion_prev * vec4<f32>(p, 1.0);
        let cur_clip = dlss_cam.motion_cur * vec4<f32>(p, 1.0);
        let prev_ndc = prev_clip.xy / prev_clip.w;
        let cur_ndc = cur_clip.xy / cur_clip.w;
        let motion = (cur_ndc - prev_ndc) * vec2<f32>(0.5, -0.5);
        textureStore(out_dlss_motion, px, vec4<f32>(motion, 0.0, 0.0));
    } else {
        textureStore(out_tex, px, vec4<f32>(sky_radiance(rd), 1.0));
        textureStore(out_diffuse_albedo, px, vec4<f32>(1.0, 1.0, 1.0, 1.0));
        textureStore(out_specular_albedo, px, vec4<f32>(vec3<f32>(0.0), 1.0));
        textureStore(out_normal_roughness, px, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(out_dlss_depth, px, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(out_dlss_motion, px, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }

    // Debug overlay (ReSTIR DLSS path): override the colour AFTER the guides; albedo = debug colour so
    // DLSS-RR passes it through ~unchanged, depth/normal/motion stay real for stable reprojection. Shared
    // `debug_overlay_color` SSOT; GI-only uses the reservoir estimate `restir_p2_core` (matches `restir_p2`).
    // This is the fix for "debug views stopped working" — the default DLSS path ignored `debug_view`.
    if (light.debug_view != 0u) {
        var dbg: vec3<f32>;
        // debug_view 9 = DLSS MOTION-VECTOR magnitude in PIXELS (RR-diagnostic): the per-pixel guide motion
        // scaled by the viewport → 1.0 (white) per pixel of screen motion. On a STATIC camera this MUST be
        // BLACK everywhere; any non-black means our motion guide is non-zero when it shouldn't be, which makes
        // DLSS-RR reject history (the "boils even when static under RR" symptom). Red = X motion, green = Y.
        if (light.debug_view == 9u && r.hit != 0u) {
            let p = ro + rd * r.t;
            let prev_clip = dlss_cam.motion_prev * vec4<f32>(p, 1.0);
            let cur_clip = dlss_cam.motion_cur * vec4<f32>(p, 1.0);
            let mv = (cur_clip.xy / cur_clip.w - prev_clip.xy / prev_clip.w) * vec2<f32>(0.5, -0.5);
            let mpix = abs(mv) * vec2<f32>(camera.viewport);
            dbg = vec3<f32>(mpix.x, mpix.y, 0.0);
        } else {
            var gi = vec3<f32>(0.0);
            if (r.hit != 0u && light.debug_view == 5u) {
                let p = ro + rd * r.t;
                let seed = (gid.x * 1973u + gid.y * 9277u + light.frame_index * 26699u) | 1u;
                if (probe_params.enabled != 0u) {
                    gi = screen_probe_integrate(r.normal, p, length(camera.cam_pos - p), gid.xy);
                } else if (restir_params.gi_half != 0u) {
                    gi = restir_gi_gather(r.normal, p, gid.xy);
                } else {
                    gi = restir_p2_core(r.normal, p, gid.xy, seed);
                }
            } else if (r.hit != 0u && light.debug_view == 8u) {
                let p = ro + rd * r.t;
                let seed = (gid.x * 1973u + gid.y * 9277u + light.frame_index * 26699u) | 1u;
                gi = di_p2_core(r.normal, p, gid.xy, seed);
            }
            dbg = debug_overlay_color(r, ro, rd, gi);
        }
        textureStore(out_tex, px, vec4<f32>(dbg, 1.0));
        textureStore(out_diffuse_albedo, px, vec4<f32>(dbg, 1.0));
        textureStore(out_specular_albedo, px, vec4<f32>(vec3<f32>(0.0), 1.0));
    }
}

// ====================================================================================================
// WORLD-SPACE RADIANCE CACHE (Phase 2.1) — ported from `bevy_solari::world_cache_*` and adapted to our
// tracer (no light list). The cache stores PRE-ACCUMULATED outgoing radiance per (quantized world position +
// quantized normal) in a GPU hash grid, refreshed by a per-frame six-pass compute loop. In 2.1 the cache RUNS
// and converges but is NOT read by the live image (the reservoir/shading path is untouched) — so there is
// ZERO visual change. Stage 2.2 wires `query_world_cache` into the initial reservoir.
//
// The full Solari loop is structurally required for a race-free, full-coverage, query-populated cache:
//   decay -> compact_single_block -> compact_blocks -> compact_write_active -> update -> blend.
// The active-cell compaction means each ACTIVE cell is owned by exactly ONE update thread, so the
// `new_radiance` write is race-free WITHOUT float atomics. Cells are populated by LAZY-INSERT on query.
//
// ADAPTATION (no light list): Solari's update does sample_di (NEE over a light list) + sample_gi (a GI bounce
// that queries the cache). WE HAVE NO LIGHT LIST, so we SKIP sample_di/presample_light_tiles entirely. The
// update pass, per active cell, traces ONE cosine-hemisphere bounce from the cell's stored (pos,normal); the
// sample radiance = `direct_lighting(hit) + emissive(hit)` on a hit, or `sky_radiance(dir)·gi_sky_intensity`
// (the 1A sky SSOT) on a miss. This is SINGLE-BOUNCE in 2.1 — no cache-query-at-hit term yet (that is 2.3).
//
// The cache is WORLD-SPACE / resolution-independent: its buffers are PERSISTENT (allocated once,
// zero-initialised so all cells start empty), never realloc'd on resize, and NEVER globally cleared on a
// terrain edit ([[feedback-gi-adapt-not-reset]]) — stale cells decay (life→0) and re-fill locally.

// Number of entries in the hash table (MUST be a power of two, >= 2^10). Substituted by the Rust pass /
// headless test so the table can be shrunk for a fast deterministic test (the live path uses 2^20).
const WORLD_CACHE_SIZE: u32 = #{WORLD_CACHE_SIZE}u;
// Marker value for an empty cell (a checksum of 0). A real checksum is forced >= 1 (see `wc_checksum`).
const WORLD_CACHE_EMPTY_CELL: u32 = 0u;
// Max linear-probe steps after a hash collision (Solari).
const WORLD_CACHE_MAX_SEARCH_STEPS: u32 = 3u;

// Geometry stored per cell: the world position + face normal of the surface that first claimed the cell.
// 16-byte-aligned rows (vec3 + pad) so the std430 layout matches the Rust `[f32; 8]` row stride (Solari's
// `WorldCacheGeometryData`).
struct WorldCacheGeometryData {
    world_position: vec3<f32>,
    padding_a: u32,
    world_normal: vec3<f32>,
    padding_b: u32,
};

// **SSOT for the world-cache KNOBS** (group 3 binding 0) — every tunable is a runtime UNIFORM
// (knobs-as-uniforms mandate), never a WGSL const. Mirrors Solari's `WORLD_CACHE_*` constants. 64 bytes.
struct WorldCacheUniform {
    cell_base_size: f32,    // size of a cache cell at the lowest LOD, in metres (Solari 0.15)
    lod_scale: f32,         // how fast the cell LOD grows with camera distance (Solari 15.0)
    gi_ray_distance: f32,   // max length of an update-pass GI bounce ray, in metres (Solari 50.0)
    cell_lifetime: u32,     // frames a cell survives un-queried before decay clears it (Solari 10)
    max_temporal_samples: f32, // temporal-blend sample-count cap (Solari 32.0)
    frame_index: u32,       // per-frame counter (decorrelates the update RNG)
    reset: u32,             // 1 = first-allocation clear: blend overwrites instead of accumulating
    use_world_cache: u32,   // 2.2 A/B gate: 1 = the initial reservoir reads the cache (default), 0 = fresh bounce
    gi_multibounce: u32,    // 2.3 A/B gate: 1 = the update pass FEEDS-FORWARD the cache at the bounce hit (default), 0 = single-bounce
    // The camera world position, stamped by the render pass. The update pass's multi-bounce cache query needs
    // it for the distance-adaptive cell LOD (`wc_get_cell_size`) — the cache view layout binds light+sky only,
    // NOT the `camera` uniform, so the view position rides in here instead. Three scalars (not a vec3) keep the
    // struct a clean 64-byte / four-16-byte-row std140 layout with no vec3 alignment padding.
    view_x: f32,
    view_y: f32,
    view_z: f32,
    // STOCHASTIC per-frame active-cell soft cap (Solari's value 40000). 0 = UNLIMITED (every active cell
    // updated+blended each frame). When > 0, the dispatch covers the FULL active count and each cell's
    // update+blend thread keeps itself with Bernoulli probability `cap / active_count` (`wc_skip_this_frame`),
    // so ~cap cells are refreshed each frame — a RANDOM subset the temporal blend integrates to the same
    // converged radiance. No starvation (equal per-frame survival probability); never corrupts the cache.
    max_active_cells_per_frame: u32,
    // Phase 2.5 NEE: number of emissive-voxel lights in `voxel_lights` (0 ⇒ NEE skipped — no emitters; the
    // light buffers are bound 1-long dummies, never indexed). Stamped by the render pass from the packed list.
    light_count: u32,
    // Phase 2.5 NEE A/B gate (knobs-as-uniforms): 1 = the update pass adds DIRECT light sampling (NEE) with MIS
    // (the live default), 0 = bounce-only (the pre-2.5 behaviour — emitters found only by the cosine bounce).
    nee_enabled: u32,
    // Phase 2.5 NEE: shadow-ray light samples per cell update (≥1). More samples cut the direct-light variance
    // further at a linear shadow-ray cost; 1 is the Solari-class default (the temporal blend averages frames).
    nee_samples: u32,
};

// The camera world position the update pass feeds to `query_world_cache` for its LOD (reconstructed from the
// three scalars above). Matches the live `reservoir_from_bounce_cached` consumer, which passes `camera.cam_pos`.
fn wc_view_position() -> vec3<f32> {
    return vec3<f32>(wc.view_x, wc.view_y, wc.view_z);
}

// STOCHASTIC per-frame active-cell soft cap (Solari `world_cache_update.wgsl` gate, ports lines 37/52/71). The
// indirect dispatch covers the FULL active count; each active cell's update+blend thread independently keeps
// itself with Bernoulli probability `cap / active_count`, so on AVERAGE `cap` cells are refreshed each frame —
// a RANDOM subset, not a fixed window. The temporal blend (`max_temporal_samples`) integrates the random subset
// over frames to the SAME converged radiance as the unlimited pass (no per-cell starvation: every cell has the
// same survival probability every frame). `max_active_cells_per_frame == 0` (default) ⇒ unlimited (the gate is
// skipped — every active cell processed, the pre-cap behaviour). When `cap >= active_count` the ratio ≥ 1 so
// nothing is dropped (a clean no-op). Returns `true` when this cell should be SKIPPED this frame.
//
// Both `world_cache_update` and `world_cache_blend` call this with the SAME per-cell seed (cell_index + frame),
// so they make the IDENTICAL keep/skip decision — blend folds exactly the samples update refreshed (a skipped
// cell's `new_radiance` slot is stale, so blending it would re-fold an old sample; skipping both keeps them in
// lock-step, the skipped cell's `world_cache_radiance` untouched — never corrupted).
fn wc_skip_this_frame(cell_index: u32, active_cell_count: u32) -> bool {
    if (wc.max_active_cells_per_frame == 0u || active_cell_count == 0u) {
        return false;
    }
    var rng = (cell_index * 9781u + wc.frame_index * 26699u) | 1u;
    return rand_next(&rng) >= f32(wc.max_active_cells_per_frame) / f32(active_cell_count);
}

@group(3) @binding(0) var<uniform> wc: WorldCacheUniform;
// 0 = empty; a non-zero IQ checksum marks an occupied cell. ATOMIC: lazy-insert claims a slot via
// `atomicCompareExchangeWeak`, so concurrent queries to colliding keys are race-free.
@group(3) @binding(1) var<storage, read_write> world_cache_checksums: array<atomic<u32>, #{WORLD_CACHE_SIZE}u>;
// Frames-to-live. ATOMIC so concurrent queries (and 2.3's cache-querying update) can `atomicStore`/`atomicMax`
// it without a race; the decay pass owns each cell singly so it reads/writes plainly via atomic load/store.
@group(3) @binding(2) var<storage, read_write> world_cache_life: array<atomic<u32>, #{WORLD_CACHE_SIZE}u>;
// Accumulated outgoing radiance (rgb) + temporal sample_count (.a).
@group(3) @binding(3) var<storage, read_write> world_cache_radiance: array<vec4<f32>, #{WORLD_CACHE_SIZE}u>;
@group(3) @binding(4) var<storage, read_write> world_cache_geometry: array<WorldCacheGeometryData, #{WORLD_CACHE_SIZE}u>;
// |luminance(new) - luminance(old)| EWMA — drives the adaptive blend responsiveness.
@group(3) @binding(5) var<storage, read_write> world_cache_luminance_deltas: array<f32, #{WORLD_CACHE_SIZE}u>;
// The update pass's per-active-cell fresh radiance, blended into `world_cache_radiance` by the blend pass.
@group(3) @binding(6) var<storage, read_write> world_cache_active_cells_new_radiance: array<vec3<f32>, #{WORLD_CACHE_SIZE}u>;
// Prefix-sum scratch: `a` is the per-cell exclusive prefix-sum within its 1024-block; `b` is the per-block
// running offset. Together they give each active cell its compacted index.
@group(3) @binding(7) var<storage, read_write> world_cache_a: array<u32, #{WORLD_CACHE_SIZE}u>;
@group(3) @binding(8) var<storage, read_write> world_cache_b: array<u32, 1024u>;
// The compacted list of active (life != 0) cell indices, one per update/blend thread.
@group(3) @binding(9) var<storage, read_write> world_cache_active_cell_indices: array<u32, #{WORLD_CACHE_SIZE}u>;
// Scalar count of active cells (the update/blend bound).
@group(3) @binding(10) var<storage, read_write> world_cache_active_cells_count: u32;
// Indirect dispatch args (ceil(active / 64), 1, 1) for the update + blend passes. In a SEPARATE bind group
// (group 2), written ONLY by `compact_write_active`, because wgpu forbids a buffer being both bound
// read-write storage AND used as an indirect-dispatch source within one compute-pass usage scope — so it must
// be UNBOUND (the update/blend pipeline layout omits group 2) when consumed as the indirect arg.
@group(2) @binding(0) var<storage, read_write> world_cache_active_cells_dispatch: vec3<u32>;

// --- Phase 2.5 NEE: emissive-voxel LIGHT LIST + power-weighted alias table (group 3) -----------------
// One VoxelLight per air-exposed emissive voxel of the resident set (mirror of `GpuVoxelLight` in gpu.rs):
// `pos` = voxel centre, `area` = one face area at the voxel's LOD, `radiance` = palette emissive (BEFORE the
// runtime `emissive_strength` knob, applied here), `weight` = luminance·area (the alias-table power). Built
// CPU-side during pack; sampled DIRECTLY (next-event estimation) by the world-cache update pass so emitters
// are found without relying on a random bounce — the principled variance/firefly fix. `wc.light_count == 0`
// (no emitters) ⇒ NEE is skipped cleanly. The buffers are bound 1-long dummies when empty (never zero-length).
struct VoxelLight { pos: vec3<f32>, area: f32, radiance: vec3<f32>, inv_pdf: f32 };
// One alias-table entry (Walker's method, mirror of `GpuAliasEntry`): with prob `prob` keep this slot's light
// `i`, else fall through to light `alias`. Picks a light proportional to its power in O(1).
struct AliasEntry { prob: f32, alias_idx: u32 };
@group(3) @binding(15) var<storage, read> voxel_lights: array<VoxelLight>;
@group(3) @binding(16) var<storage, read> voxel_light_alias: array<AliasEntry>;

// --- hash + quantization (ported verbatim from Solari world_cache_query.wgsl) -----------------------

fn wc_pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn wc_iqint_hash(input: u32) -> u32 {
    let n = (input << 13u) ^ input;
    return n * (n * n * 15731u + 789221u) + 1376312589u;
}

fn wc_wrap_key(key: u32) -> u32 {
    return key & (WORLD_CACHE_SIZE - 1u);
}

// Distance-adaptive cell size: `cell_base_size · 2^lod`, lod growing with camera distance. The fractional LOD
// is stochastically rounded (cubed-fract probability) so the transition between LODs dithers instead of
// banding (Solari `get_cell_size`).
fn wc_get_cell_size(world_position: vec3<f32>, view_position: vec3<f32>, rng: ptr<function, u32>) -> f32 {
    let camera_distance = distance(view_position, world_position) / wc.lod_scale;
    let lod_f = log2(1.0 + camera_distance);
    let lod_fract = fract(lod_f);
    let lod = floor(lod_f) + select(0.0, 1.0, rand_next(rng) < lod_fract * lod_fract * lod_fract);
    return wc.cell_base_size * exp2(lod);
}

fn wc_quantize_position(world_position: vec3<f32>, quantization_factor: f32) -> vec3<f32> {
    // CAMERA-RELATIVE quantization: `floor(world_position/qf + eps)` directly loses f32 precision when the world
    // coordinate is large (a streamed scene placed far from the origin) — the `world/qf` dividend exceeds the f32
    // integer-exact range and adjacent fine cells collapse / the eps bias is lost, leaking GI across the hash.
    // Computing the cell index RELATIVE to the camera's cell keeps the dividend small (≈ view radius / qf), so
    // precision holds near the camera (where the fine cells live) regardless of absolute world position. The key
    // is algebraically identical (`cam_cell + floor(local) == floor(world/qf)`), so cells stay world-stable as
    // the camera moves. (Full precision at >~1e5 m still needs CPU-side f64 world-rebasing; this is the GPU-side
    // mitigation that covers the realistic streamed-scene range.)
    // Use `wc.view_*` (the cache uniform's camera position) NOT `camera.cam_pos`: the world-cache compute passes
    // bind the cache uniform (group 3) but NOT the camera uniform (group 1), and they call this too.
    let cam = vec3<f32>(wc.view_x, wc.view_y, wc.view_z);
    let cam_cell = floor(cam / quantization_factor);
    return cam_cell + floor((world_position - cam_cell * quantization_factor) / quantization_factor + 0.0001);
}

fn wc_quantize_normal(world_normal: vec3<f32>) -> vec3<f32> {
    return floor(world_normal * 2.0 + 0.0001);
}

fn wc_compute_key(world_position: vec3<u32>, world_normal: vec3<u32>) -> u32 {
    var key = wc_pcg_hash(world_position.x);
    key = wc_pcg_hash(key + world_position.y);
    key = wc_pcg_hash(key + world_position.z);
    key = wc_pcg_hash(key + world_normal.x);
    key = wc_pcg_hash(key + world_normal.y);
    key = wc_pcg_hash(key + world_normal.z);
    return wc_wrap_key(key);
}

fn wc_compute_checksum(world_position: vec3<u32>, world_normal: vec3<u32>) -> u32 {
    var key = wc_iqint_hash(world_position.x);
    key = wc_iqint_hash(key + world_position.y);
    key = wc_iqint_hash(key + world_position.z);
    key = wc_iqint_hash(key + world_normal.x);
    key = wc_iqint_hash(key + world_normal.y);
    key = wc_iqint_hash(key + world_normal.z);
    return max(key, 1u); // 0 is reserved for WORLD_CACHE_EMPTY_CELL
}

// Query the cache for the outgoing radiance at (`world_position`, `world_normal`), as seen from `view_position`
// (drives the cell LOD). Distance-adaptive cell size, tangent-plane jitter (blurs the grid), PCG key + IQ
// checksum, <=3-step linear probe. On a MATCH: mark the cell alive (life = `cell_lifetime`) and return its
// radiance.rgb. On an EMPTY slot: LAZY-INSERT — claim it via `atomicCompareExchangeWeak` on the checksum,
// store the query's geometry, mark it alive, and return 0 (it fills over the next frames). Ported from Solari
// `query_world_cache`. `ray_t` is reserved for the first-bounce light-leak guard (2.2+); unused in 2.1.
fn query_world_cache(world_position_in: vec3<f32>, world_normal: vec3<f32>, view_position: vec3<f32>, ray_t: f32, cell_lifetime: u32, rng: ptr<function, u32>) -> vec3<f32> {
    var world_position = world_position_in;
    var cell_size = wc_get_cell_size(world_position, view_position, rng);

    // FIRST-BOUNCE LIGHT-LEAK PREVENTION (Solari world_cache_query.wgsl:47-52, on-by-default node.rs:564).
    // A bounce shorter than the distance-LOD cell straddles thin geometry (e.g. a cube face → adjacent floor) —
    // the over-sized cell + tangent jitter then quantize the query onto the cell on the FAR side of the wall,
    // reading exterior radiance into an interior face (the reported Cornell leak). Clamping back to the small
    // base cell (`wc.cell_base_size`, default `0.09375·BRICK_WORLD_SIZE` ≈ 0.0375 m at the 0.05 m flip — sized to
    // fit INSIDE the 0.1 m Cornell wall) makes the straddle impossible and shrinks the subsequent jitter
    // amplitude (±0.5·cell_size). Brick-relative / flip-proof. Robust-by-construction.
    if (ray_t < cell_size) {
        cell_size = wc.cell_base_size;
    }

    // Jitter the query point in the tangent plane (blurs the cache so it is not so grid-like).
    let TBN = onb(world_normal);
    let offset = (vec2<f32>(rand_next(rng), rand_next(rng)) * 2.0 - 1.0) * cell_size * 0.5;
    world_position += offset.x * TBN[0] + offset.y * TBN[1];
    cell_size = wc_get_cell_size(world_position, view_position, rng);

    let world_position_quantized = bitcast<vec3<u32>>(wc_quantize_position(world_position, cell_size));
    let world_normal_quantized = bitcast<vec3<u32>>(wc_quantize_normal(world_normal));
    var key = wc_compute_key(world_position_quantized, world_normal_quantized);
    let checksum = wc_compute_checksum(world_position_quantized, world_normal_quantized);

    var result = vec3<f32>(0.0);
    var done = false;
    for (var i = 0u; i < WORLD_CACHE_MAX_SEARCH_STEPS; i = i + 1u) {
        if (done) { continue; }
        let existing_checksum = atomicCompareExchangeWeak(&world_cache_checksums[key], WORLD_CACHE_EMPTY_CELL, checksum).old_value;

        if (existing_checksum == checksum || existing_checksum == WORLD_CACHE_EMPTY_CELL) {
            // Cell exists or was just claimed — (re)set its lifetime so it stays active.
            atomicMax(&world_cache_life[key], cell_lifetime);
        }

        if (existing_checksum == checksum) {
            result = world_cache_radiance[key].rgb; // existing entry — return its accumulated radiance
            done = true;
        } else if (existing_checksum == WORLD_CACHE_EMPTY_CELL) {
            // We claimed an empty cell — store the query's geometry; radiance fills over the next frames.
            world_cache_geometry[key].world_position = world_position;
            world_cache_geometry[key].world_normal = world_normal;
            done = true;
        } else {
            key = key + 1u; // collision — linear probe to the next slot (wrap handled by the table size)
            if (key >= WORLD_CACHE_SIZE) { key = 0u; }
        }
    }
    return result;
}

// --- the six compute passes (one compute pass, dispatched in order: consecutive dispatches get WebGPU's
//     implicit storage barrier, so each pass sees the previous pass's writes) ---------------------------

var<workgroup> wc_w1: array<u32, 1024u>;
var<workgroup> wc_w2: array<u32, 1024u>;

// PASS 1 — DECAY. Every cell: life--; if it hit 0, mark the cell empty + clear its radiance/luminance so a
// future query can re-claim the slot. The world-space ADAPT-NOT-RESET mechanism: stale cells age out locally;
// there is no global clear (the only clear is the first-allocation `reset`, handled by the host zero-init).
@compute @workgroup_size(1024, 1, 1)
fn world_cache_decay(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    var life = atomicLoad(&world_cache_life[i]);
    if (life > 0u) {
        life = life - 1u;
        atomicStore(&world_cache_life[i], life);
        if (life == 0u) {
            atomicStore(&world_cache_checksums[i], WORLD_CACHE_EMPTY_CELL);
            world_cache_radiance[i] = vec4<f32>(0.0);
            world_cache_luminance_deltas[i] = 0.0;
        }
    }
}

// PASS 2 — COMPACT (single block): a 1024-wide exclusive prefix-sum of `life != 0` within each 1024-block,
// written to `world_cache_a`. (Hillis–Steele scan, ported verbatim from Solari compact_world_cache_single_block.)
@compute @workgroup_size(1024, 1, 1)
fn world_cache_compact_single_block(
    @builtin(global_invocation_id) cell_id: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    if (t == 0u) { wc_w1[0u] = 0u; } else { wc_w1[t] = u32(atomicLoad(&world_cache_life[cell_id.x - 1u]) != 0u); } workgroupBarrier();
    if (t < 1u) { wc_w2[t] = wc_w1[t]; } else { wc_w2[t] = wc_w1[t] + wc_w1[t - 1u]; } workgroupBarrier();
    if (t < 2u) { wc_w1[t] = wc_w2[t]; } else { wc_w1[t] = wc_w2[t] + wc_w2[t - 2u]; } workgroupBarrier();
    if (t < 4u) { wc_w2[t] = wc_w1[t]; } else { wc_w2[t] = wc_w1[t] + wc_w1[t - 4u]; } workgroupBarrier();
    if (t < 8u) { wc_w1[t] = wc_w2[t]; } else { wc_w1[t] = wc_w2[t] + wc_w2[t - 8u]; } workgroupBarrier();
    if (t < 16u) { wc_w2[t] = wc_w1[t]; } else { wc_w2[t] = wc_w1[t] + wc_w1[t - 16u]; } workgroupBarrier();
    if (t < 32u) { wc_w1[t] = wc_w2[t]; } else { wc_w1[t] = wc_w2[t] + wc_w2[t - 32u]; } workgroupBarrier();
    if (t < 64u) { wc_w2[t] = wc_w1[t]; } else { wc_w2[t] = wc_w1[t] + wc_w1[t - 64u]; } workgroupBarrier();
    if (t < 128u) { wc_w1[t] = wc_w2[t]; } else { wc_w1[t] = wc_w2[t] + wc_w2[t - 128u]; } workgroupBarrier();
    if (t < 256u) { wc_w2[t] = wc_w1[t]; } else { wc_w2[t] = wc_w1[t] + wc_w1[t - 256u]; } workgroupBarrier();
    if (t < 512u) { world_cache_a[cell_id.x] = wc_w2[t]; } else { world_cache_a[cell_id.x] = wc_w2[t] + wc_w2[t - 512u]; }
}

// PASS 3 — COMPACT (blocks): exclusive prefix-sum across the per-block totals (the last `a` entry of each
// 1024-block) → `world_cache_b`, the per-block running offset. ONE workgroup of 1024 (covers up to 1024
// blocks = 2^20 cells). Ported from Solari compact_world_cache_blocks.
@compute @workgroup_size(1024, 1, 1)
fn world_cache_compact_blocks(@builtin(local_invocation_index) t: u32) {
    // Seed each block's total (the last `a` entry of the PREVIOUS block). Blocks beyond the table's block
    // count contribute 0, so the scan is correct for any WORLD_CACHE_SIZE in [2^10, 2^20] (the live path is
    // 2^20 = 1024 blocks; a smaller test table has fewer, and the high lanes seed 0).
    let num_blocks = WORLD_CACHE_SIZE / 1024u;
    if (t == 0u || t > num_blocks) { wc_w1[t] = 0u; } else { wc_w1[t] = world_cache_a[t * 1024u - 1u]; } workgroupBarrier();
    if (t < 1u) { wc_w2[t] = wc_w1[t]; } else { wc_w2[t] = wc_w1[t] + wc_w1[t - 1u]; } workgroupBarrier();
    if (t < 2u) { wc_w1[t] = wc_w2[t]; } else { wc_w1[t] = wc_w2[t] + wc_w2[t - 2u]; } workgroupBarrier();
    if (t < 4u) { wc_w2[t] = wc_w1[t]; } else { wc_w2[t] = wc_w1[t] + wc_w1[t - 4u]; } workgroupBarrier();
    if (t < 8u) { wc_w1[t] = wc_w2[t]; } else { wc_w1[t] = wc_w2[t] + wc_w2[t - 8u]; } workgroupBarrier();
    if (t < 16u) { wc_w2[t] = wc_w1[t]; } else { wc_w2[t] = wc_w1[t] + wc_w1[t - 16u]; } workgroupBarrier();
    if (t < 32u) { wc_w1[t] = wc_w2[t]; } else { wc_w1[t] = wc_w2[t] + wc_w2[t - 32u]; } workgroupBarrier();
    if (t < 64u) { wc_w2[t] = wc_w1[t]; } else { wc_w2[t] = wc_w1[t] + wc_w1[t - 64u]; } workgroupBarrier();
    if (t < 128u) { wc_w1[t] = wc_w2[t]; } else { wc_w1[t] = wc_w2[t] + wc_w2[t - 128u]; } workgroupBarrier();
    if (t < 256u) { wc_w2[t] = wc_w1[t]; } else { wc_w2[t] = wc_w1[t] + wc_w1[t - 256u]; } workgroupBarrier();
    if (t < 512u) { world_cache_b[t] = wc_w2[t]; } else { world_cache_b[t] = wc_w2[t] + wc_w2[t - 512u]; }
}

// PASS 4 — COMPACT (write active): each active cell's compacted index = its in-block prefix + its block offset;
// scatter the cell index into `world_cache_active_cell_indices`. The last thread writes the active-cell count +
// the indirect dispatch args (ceil(count / 64)). Ported from Solari compact_world_cache_write_active_cells.
@compute @workgroup_size(1024, 1, 1)
fn world_cache_compact_write_active(
    @builtin(global_invocation_id) cell_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) thread_index: u32,
) {
    let compacted_index = world_cache_a[cell_id.x] + world_cache_b[workgroup_id.x];
    let cell_active = atomicLoad(&world_cache_life[cell_id.x]) != 0u;

    if (cell_active) {
        world_cache_active_cell_indices[compacted_index] = cell_id.x;
    }

    if (thread_index == 1023u && workgroup_id.x == (WORLD_CACHE_SIZE / 1024u) - 1u) {
        let active_cell_count = compacted_index + u32(cell_active);
        world_cache_active_cells_count = active_cell_count;
        // Dispatch the FULL active count (ceil(count/64) workgroups), uncapped — Solari `node.rs:307-330`. The
        // STOCHASTIC soft cap (`wc_skip_this_frame`) gates INSIDE the update/blend kernels per cell, so the
        // dispatch must cover every active cell for the Bernoulli gate to draw from the whole set.
        world_cache_active_cells_dispatch = vec3<u32>((active_cell_count + 63u) / 64u, 1u, 1u);
    }
}

// --- Phase 2.5 NEE: direct emissive-voxel light sampling with MIS ------------------------------------
//
// The world-cache update finds emitters ONLY by the random cosine bounce — high variance (a cell whose
// hemisphere a small bright emitter subtends catches it in few samples, so the per-cell radiance "boils"; this
// is the variance the discarded firefly clamp used to band-aid). NEE samples the emissive-voxel LIGHT LIST
// (`voxel_lights`) DIRECTLY: pick a light by power (alias table), trace ONE shadow ray, and add its unoccluded
// area-light contribution. The two estimators of the SAME emitter contribution (the cosine bounce that may hit
// an emitter, and this direct light sample) are reconciled by the BALANCE-HEURISTIC MIS WEIGHT so emitters are
// never double-counted (Veach; mirrors Solari's `sample_di` + the path tracer's `power_heuristic`).
//
// CONVENTION: the cache is COSINE-PRE-DIVIDED — a cosine-sampled bounce contributes the gathered radiance
// directly (the `cos θ/π` pdf cancels the `cos θ/π` Lambert kernel), so the cell stores `(1/π)∫ L cos θ dω`.
// The NEE term is the SAME quantity estimated by area sampling, so it carries the matching `1/π` and a `cos θ`
// at the receiver: `(1/π) · L · cosθ_surf · cosθ_light/dist² · V · inverse_pdf_area`. Returned RAW (no MIS) so
// the update pass can apply the MIS weight + `emissive_strength` knob once, consistently with the bounce term.

const WC_INV_PI: f32 = 0.31830988618; // 1/π — the cosine-pre-divide factor

// Solid-angle pdf of the cosine-hemisphere bounce in direction `dir` about normal `n` (= cosθ/π). Used as the
// "other technique" pdf in the MIS weight for both estimators.
fn wc_bounce_pdf(n: vec3<f32>, dir: vec3<f32>) -> f32 {
    return max(dot(n, dir), 0.0) * WC_INV_PI;
}

// One NEE light sample's contribution to the cache quantity, ALREADY cosine-pre-divided and MIS-weighted, plus
// the geometry the caller needs for nothing more (the MIS is applied inside). Picks a light from the alias
// table by power, forms the area-light estimator with the receiver cosine, traces a shadow ray for visibility,
// and multiplies by `balance_heuristic(p_light, p_bounce)` so the cosine bounce (which also catches emitters)
// is not double-counted. Returns 0 when there are no lights, the light is back-facing/behind the surface, or
// the shadow ray is occluded. `world_position`/`n` are the cell's stored geometry; `rng` is the cell's stream.
fn wc_sample_light_nee(world_position: vec3<f32>, n: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let count = wc.light_count;
    if (count == 0u) {
        return vec3<f32>(0.0);
    }
    // Power-weighted alias draw: a uniform slot pick, then keep-or-fall-through by `prob`.
    let slot = min(u32(rand_next(rng) * f32(count)), count - 1u);
    let entry = voxel_light_alias[slot];
    var li = slot;
    if (rand_next(rng) >= entry.prob) {
        li = entry.alias_idx;
    }
    let lgt = voxel_lights[li];

    // Sample a point on the light's voxel FACE: jitter within ±half a cell in the surface tangent plane around
    // the voxel centre (the face-area measure `lgt.area` matches this — a square of side sqrt(area)). Keeps the
    // estimator unbiased (uniform over the area) and softens contact shadows vs a bare point sample.
    let half = sqrt(lgt.area) * 0.5;
    let jb = onb(normalize(lgt.pos - world_position)); // a basis facing the receiver (any face-tangent is fine)
    let j = (vec2<f32>(rand_next(rng), rand_next(rng)) * 2.0 - 1.0) * half;
    let y = lgt.pos + jb[0] * j.x + jb[1] * j.y;

    let to_light = y - world_position;
    let dist2 = dot(to_light, to_light);
    if (dist2 <= 1e-8) {
        return vec3<f32>(0.0);
    }
    let dist = sqrt(dist2);
    let wi = to_light / dist;
    let cos_surf = dot(n, wi);
    if (cos_surf <= 0.0) {
        return vec3<f32>(0.0); // light is below the receiver's hemisphere
    }
    // FACE-ORIENTATION MODEL: the light list does NOT store which of the emissive voxel's faces is exposed (a
    // voxel can emit from up to six faces), so we model the emitter as facing the receiver head-on — `cos_light
    // = 1`. This is the standard ISOTROPIC-voxel-emitter approximation: it keeps the estimator unbiased for the
    // chosen `inverse_pdf` (the SAME `cos_light = 1` is used in `p_light` below, so the two are self-consistent)
    // and avoids an oriented-face fetch. `cos_light` cancels out of `geom · inv_pdf` here and reappears in
    // `p_light`, so the cancellation is exact — no hidden bias.
    let cos_light = 1.0;
    let geom = cos_surf * cos_light / dist2;

    // Shadow ray: is the light point visible? Offset off the RECEIVER along its normal to avoid self-hit, and —
    // crucially — stop the ray ONE VOXEL CELL SHORT of the light point. The light reference (`lgt.pos`) is the
    // emitter voxel CENTRE, which sits ~half a cell INSIDE the solid emissive voxel, so a ray reaching it would
    // be occluded by the emitter's OWN surface (the voxel is solid) — making EVERY NEE sample spuriously
    // shadowed (a silent total energy loss). Pulling `t_max` back by `sqrt(area)` (one voxel edge ≥ the
    // centre-to-face distance) stops the ray in the AIR just before the emitter, so only TRUE occluders between
    // the receiver and the emitter shadow it. `cell = sqrt(area)` is the emitter voxel's edge length.
    let cell = sqrt(lgt.area);
    let t_max = dist - cell;
    let origin = world_position + n * light.shadow_bias;
    if (t_max <= light.shadow_bias) {
        // The light is within one voxel of the receiver (e.g. an adjacent emissive voxel) — treat as unoccluded
        // (no room for an occluder), so contact light from a directly-adjacent emitter is not lost.
    } else if (trace_occluded(origin, wi, light.shadow_bias, t_max)) {
        return vec3<f32>(0.0);
    }

    // Area-light estimator of the cosine-pre-divided cache quantity (apply the `emissive_strength` knob — the
    // per-block emissive SSOT — exactly like the bounce term does).
    let radiance = lgt.radiance * light.emissive_strength;
    let estimator = radiance * (WC_INV_PI * geom * lgt.inv_pdf);

    // MIS (balance heuristic): weight NEE by p_light / (p_light + p_bounce), both in SOLID-ANGLE measure, so the
    // bounce estimator (weighted symmetrically where it hits an emitter) and NEE together are unbiased + low
    // variance with no double-count. p_light (sa) = pdf_area · dist² / cos_light; pdf_area = 1/inv_pdf.
    let p_light = (1.0 / lgt.inv_pdf) * dist2 / max(cos_light, 1e-4);
    let p_bounce = wc_bounce_pdf(n, wi);
    let w_nee = balance_heuristic(p_light, p_bounce);
    return estimator * w_nee;
}

// The MIS weight to apply to the cosine-BOUNCE estimator when the bounce HITS an emitter (so it is not
// double-counted against NEE). The bounce sampled `dir` with pdf `cosθ/π`; the light it struck would have been
// drawn by NEE with solid-angle pdf `p_light` (computed from the hit emitter's area + distance + the alias
// inverse_pdf). Weight = p_bounce / (p_bounce + p_light) (balance heuristic). When NEE is OFF (`nee_enabled==0`)
// or there are no lights, returns 1 (the bounce carries the FULL emitter term, the pre-2.5 behaviour).
//
// We do NOT know WHICH light-list entry the bounce struck, so we reconstruct the NEE solid-angle pdf the hit
// emitter WOULD have had from its own geometry: a voxel face of area `VOXEL_SIZE²` at distance `hit_t`, picked
// with probability `≈ 1/light_count` (the equal-power surrogate — exact for equal emitters, the common case;
// for unequal emitters the alias pick differs but MIS stays unbiased for any partition, only the variance
// shifts). With the SAME isotropic `cos_light = 1` model NEE uses, `p_light(sa) = pick_prob · hit_t² / area`,
// and `p_bounce(sa) = cosθ/π` at the ACTUAL bounce direction (`wc_bounce_pdf(n, dir)`) — the SAME representation
// NEE evaluates for a shared direction, so the two `balance_heuristic` weights (args swapped) form a valid
// partition summing to ≤ 1 (no double-count). NOTE: the original used the `1/π` PEAK here, which made the two
// weights disagree on `p_bounce` off normal incidence → an energy bias up to ~50% at grazing angles (caught in
// GI 2.5 review). Matching the real cosθ/π restores the partition. The `1/light_count` pick-prob + LOD0
// `VOXEL_SIZE²` area remain a surrogate for the hit light's true alias pdf — EXACT for equal-power LOD0 emitters
// (Cornell's single emitter; Sponza has none), approximate for unequal-power/mixed-LOD scenes (a documented
// residual that needs a per-hit light-id lookup to close; both GI 2.5 reviewers flagged it as unavoidable here).
fn wc_bounce_emitter_mis(n: vec3<f32>, hit_t: f32, dir: vec3<f32>) -> f32 {
    if (wc.nee_enabled == 0u || wc.light_count == 0u) {
        return 1.0; // NEE off → the bounce is the only emitter estimator (no MIS split)
    }
    // The bounce-hit emitter as a NEE light: a VOXEL_SIZE² face, pick prob ≈ 1/light_count, isotropic (cos_light
    // = 1) — matching `wc_sample_light_nee`'s model so the two weights partition consistently.
    let area = VOXEL_SIZE * VOXEL_SIZE;
    let pick_prob = 1.0 / f32(wc.light_count);
    let p_light = pick_prob * (hit_t * hit_t) / area;
    let p_bounce = wc_bounce_pdf(n, dir); // cosθ/π at the actual dir — matches NEE's p_bounce (valid partition)
    return balance_heuristic(p_bounce, p_light);
}

// (GI 5.0) The cell's OWN direct SUN — a delta next-event sample folded into the cache so the GI consumer reads
// the bounce-hit's sun from the (multi-frame-averaged) cache instead of recomputing the hard sun-shadow FRESH
// per frame (Solari `sample_di`; the boil fix). One shadow ray from the cell toward the sun; 0 where the face
// points away or the sun is occluded. CONVENTION: returns the SAME quantity `direct_lighting` uses
// (sun_color·intensity·cosθ·shadow, NO 1/π — the engine's whole direct path omits it), so `albedo·cache`
// reproduces the old `direct_lighting` term and the brightness is unchanged. No MIS partner: the sun is a delta
// light and GI bounces use `sky_radiance_no_sun`, so the cosine bounce never also samples it (no double-count).
fn wc_sun_direct(world_position: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let to_sun = -light.sun_direction;
    let ndotl = max(dot(n, to_sun), 0.0);
    if (ndotl <= 0.0) {
        return vec3<f32>(0.0);
    }
    let origin = world_position + n * light.shadow_bias;
    if (trace_occluded(origin, to_sun, 0.0, 1.0e4)) {
        return vec3<f32>(0.0);
    }
    return light.sun_color * (light.sun_intensity * ndotl);
}

// PASS 5 — UPDATE (indirect, one thread per ACTIVE cell). ADAPTATION (Phase 2.5: NEE light list): trace ONE
// cosine-weighted hemisphere bounce from the cell's stored (pos,normal); the sample radiance = direct lighting
// at the hit + the hit's emissive glow (MIS-weighted), or the procedural sky (the 1A SSOT) on a miss; PLUS a
// DIRECT next-event light sample (`wc_sample_light_nee`) of the emissive-voxel list. Because the compaction
// gives each active cell exactly ONE owning thread, the `new_radiance` write is race-free WITHOUT float atomics.
//
// MULTI-BOUNCE (Phase 2.3, gated by `wc.gi_multibounce`, default ON; mirrors Solari `world_cache_update.wgsl`
// `sample_gi`:44-62): at the bounce HIT we ALSO add the reflected indirect read FROM the cache —
//   new_radiance += albedo(hit) · query_world_cache(hit_pos, hit_normal, …)
// so each cell's outgoing radiance gathers its neighbours' CACHED outgoing radiance → cells query cells →
// every frame the cache carries one MORE light bounce than the last (feed-forward, NOT in-frame recursion: the
// query reads LAST frame's blended `world_cache_radiance` and this thread writes THIS frame's `new_radiance`).
// Convention: our cache is COSINE-PRE-DIVIDED (the cosine-weighted gather bakes the 1/π in), so the consumer
// multiplies by albedo ONLY — NO further /π — byte-identical to the 2.2 `reservoir_from_bounce_cached` term
// (which also uses `r.color.rgb * query_world_cache(…)`). The cache lazy-inserts the hit cell on a miss
// (returns 0, fills over the next frames). BOUNDED by construction — see the energy note below.
@compute @workgroup_size(64, 1, 1)
fn world_cache_update(@builtin(global_invocation_id) active_cell_id: vec3<u32>) {
    // The dispatch is ceil(count/64) workgroups, so re-bound here: the last workgroup can over-launch threads
    // past the active count — those keep nothing (no slot) and must early-out.
    if (active_cell_id.x >= world_cache_active_cells_count) { return; }
    let cell_index = world_cache_active_cell_indices[active_cell_id.x];
    // STOCHASTIC soft cap (Solari gate): keep this cell with probability `cap / active_count`. Skipped cells
    // keep their last radiance+life untouched (never corrupted); the temporal blend integrates the random
    // subset to the same converged radiance. `world_cache_blend` makes the IDENTICAL decision (same cell+frame
    // seed) so it folds exactly the cells refreshed here. Cap 0 (default) ⇒ never skips (full pass).
    if (wc_skip_this_frame(cell_index, world_cache_active_cells_count)) { return; }
    let geo = world_cache_geometry[cell_index];
    var rng = (cell_index * 9781u + wc.frame_index * 26699u) | 1u;

    // Cosine-weighted hemisphere bounce (the cosine pdf cancels the Lambert cosine, so the per-sample estimate
    // is the gathered radiance directly — same convention as `gather_gi` / `reservoir_from_bounce`).
    let n = geo.world_normal;
    let basis = onb(n);
    let u1 = rand_next(&rng);
    let u2 = rand_next(&rng);
    let r = sqrt(u1);
    let phi = 6.2831853 * u2;
    let dir = normalize(basis[0] * (r * cos(phi)) + basis[1] * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));

    let origin = geo.world_position + n * light.shadow_bias;
    let hit = trace(origin, dir, 0.0, wc.gi_ray_distance);
    // INDIRECT (Solari `sample_gi`): the radiance arriving at THIS cell from one cosine bounce = the bounce-hit's
    // OUTGOING radiance, read from the cache (`emissive(hit)·mis + albedo·cache(hit)`), NOT a fresh
    // `direct_lighting(hit)`. The hit's own direct (sun via its `wc_sun_direct`, emitters via its NEE) lives in
    // cache(hit), so the hard sun-shadow is averaged in the cache instead of re-rolled per frame (the boil fix —
    // GI 5.0; see `reservoir_from_bounce_cached`).
    var radiance: vec3<f32>;
    if (hit.hit != 0u) {
        let hp = origin + dir * hit.t;
        let emit_mis = wc_bounce_emitter_mis(n, hit.t, dir);
        radiance = hit.emissive * (light.emissive_strength * emit_mis);
        // Feed-forward multi-bounce: the reflected light the cache holds for the hit (includes the hit's own
        // direct via its sample_di). Reads LAST frame's blended radiance (no in-frame recursion). With it OFF the
        // cell still gets its OWN direct (sample_di below) + the hit's emitter glow; the diffuse-sun first bounce
        // flows via cache(hit) when ON. Albedo only (cosine-pre-divided cache convention).
        if (wc.gi_multibounce != 0u) {
            let cell_life = atomicLoad(&world_cache_life[cell_index]);
            radiance += hit.color.rgb * query_world_cache(hp, hit.normal, wc_view_position(), hit.t, cell_life, &rng);
        }
    } else {
        // GI-bounce miss → sky GRADIENT only (sun disk excluded — the sun is handled by `wc_sun_direct`).
        radiance = sky_radiance_no_sun(dir) * sky.gi_sky_intensity;
    }

    // DIRECT at the cell (Solari `sample_di`), folded into the cache so the GI consumer reads it smoothed: the
    // SUN as a delta next-event sample (`wc_sun_direct`) + the emissive-voxel NEE below. The sun term is THE one
    // that, recomputed fresh at the GI bounce hit, produced the sun/sky-lit boil; storing it in the multi-frame-
    // averaged cache is the fix.
    radiance += wc_sun_direct(geo.world_position, n);

    // Phase 2.5 NEE: add the DIRECT emissive-voxel light sample(s), MIS-balanced against the bounce above.
    if (wc.nee_enabled != 0u && wc.light_count != 0u) {
        let ns = max(wc.nee_samples, 1u);
        var nee = vec3<f32>(0.0);
        for (var s = 0u; s < ns; s = s + 1u) {
            nee += wc_sample_light_nee(geo.world_position, n, &rng);
        }
        radiance += nee / f32(ns);
    }

    // Finite-guard: a NEE near-singularity or degenerate bounce must never write a NaN/Inf cell — a poisoned
    // cell is read by every querier and (since an actively-queried cell never decays) would persist forever.
    world_cache_active_cells_new_radiance[active_cell_id.x] = sanitize3(radiance);
}

// PASS 6 — BLEND (indirect, one thread per ACTIVE cell). Solari's adaptive temporal blend: an exponential
// running mean with a sample-count cap, made MORE responsive when the luminance is changing fast (so a newly
// lit/shadowed cell adapts quickly but a stable cell stays smooth). LOCAL adaptation — never a global clear;
// the only reset is the first-allocation `wc.reset` (host zero-init), which overwrites instead of blending.
// Ported from Solari blend_new_samples.
@compute @workgroup_size(64, 1, 1)
fn world_cache_blend(@builtin(global_invocation_id) active_cell_id: vec3<u32>) {
    // Same full-count bound as `world_cache_update`: early-out threads the last workgroup over-launched.
    if (active_cell_id.x >= world_cache_active_cells_count) { return; }
    let cell_index = world_cache_active_cell_indices[active_cell_id.x];
    // SAME stochastic soft cap as `world_cache_update` — identical cell+frame seed ⇒ identical keep/skip
    // decision, so blend folds EXACTLY the cells update refreshed. A skipped cell's `new_radiance` slot is
    // stale; skipping the blend too leaves its `world_cache_radiance` untouched (no re-fold, no corruption).
    if (wc_skip_this_frame(cell_index, world_cache_active_cells_count)) { return; }

    let old_radiance = world_cache_radiance[cell_index];
    let new_radiance = world_cache_active_cells_new_radiance[active_cell_id.x];
    let luminance_delta = world_cache_luminance_deltas[cell_index];

    let sample_count = min(old_radiance.a + 1.0, wc.max_temporal_samples);
    let alpha = abs(luminance_delta) / max(restir_luminance(old_radiance.rgb), 0.001);
    let max_sample_count = mix(wc.max_temporal_samples, 1.0, pow(saturate(alpha), 1.0 / 8.0));
    var blend_amount = 1.0 / min(sample_count, max_sample_count);
    if (wc.reset != 0u) {
        blend_amount = 1.0;
    }

    let blended_radiance = sanitize3(mix(old_radiance.rgb, new_radiance, blend_amount));
    let new_delta = mix(luminance_delta, restir_luminance(blended_radiance) - restir_luminance(old_radiance.rgb), 1.0 / 8.0);
    let blended_luminance_delta = select(new_delta, 0.0, wc.reset != 0u);

    world_cache_radiance[cell_index] = vec4<f32>(blended_radiance, sample_count);
    world_cache_luminance_deltas[cell_index] = blended_luminance_delta;
}

// --- headless TEST entry: seed cells via the cache hash + read a chosen cell back -------------------
// The live path does NOT call `query_world_cache` in 2.1, so without a seeder no cell would ever become
// active. This entry lets the headless harness INSERT a known set of (pos,normal) query points each frame
// (driving the lazy-insert + the alive-mark), then read back the resolved cell index / checksum / radiance so
// the test can assert the cache converges to the analytic single-bounce irradiance. It runs FIRST each frame
// (before decay), exactly where the live reservoir query will sit in 2.2.
struct WcQueryPoint { world_position: vec3<f32>, _p0: u32, world_normal: vec3<f32>, _p1: u32 };
struct WcQueryOut { radiance: vec3<f32>, cell_index: u32, checksum: u32, life: u32, _p0: u32, _p1: u32 };
struct WcQueryParams { view_position: vec3<f32>, n_points: u32, frame_index: u32, ray_t: f32, _p1: u32, _p2: u32 };

@group(3) @binding(12) var<storage, read> wc_query_points: array<WcQueryPoint>;
@group(3) @binding(13) var<storage, read_write> wc_query_out: array<WcQueryOut>;
@group(3) @binding(14) var<uniform> wc_query_params: WcQueryParams;

@compute @workgroup_size(64, 1, 1)
fn world_cache_query_seed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= wc_query_params.n_points) { return; }
    let q = wc_query_points[i];
    // Mark the cell alive with the full lifetime and recover its radiance. The recomputed key/checksum below
    // mirror `query_world_cache` (no jitter here, so the harness reads back a DETERMINISTIC slot — jitter would
    // scatter the same query across neighbouring cells frame-to-frame, which is good for coverage but bad for a
    // single-cell read-back). This keeps the test's cell stable while still exercising the real insert+probe.
    let cell_size = wc.cell_base_size; // LOD 0 (the test view is close), no jitter — stable read-back slot
    let qpos = bitcast<vec3<u32>>(wc_quantize_position(q.world_position, cell_size));
    let qnrm = bitcast<vec3<u32>>(wc_quantize_normal(q.world_normal));
    var key = wc_compute_key(qpos, qnrm);
    let checksum = wc_compute_checksum(qpos, qnrm);

    var found_key = key;
    var rad = vec3<f32>(0.0);
    var done = false;
    for (var s = 0u; s < WORLD_CACHE_MAX_SEARCH_STEPS; s = s + 1u) {
        if (done) { continue; }
        let existing = atomicCompareExchangeWeak(&world_cache_checksums[key], WORLD_CACHE_EMPTY_CELL, checksum).old_value;
        if (existing == checksum || existing == WORLD_CACHE_EMPTY_CELL) {
            atomicMax(&world_cache_life[key], wc.cell_lifetime);
            if (existing == WORLD_CACHE_EMPTY_CELL) {
                world_cache_geometry[key].world_position = q.world_position;
                world_cache_geometry[key].world_normal = q.world_normal;
            } else {
                rad = world_cache_radiance[key].rgb;
            }
            found_key = key;
            done = true;
        } else {
            found_key = key;
            key = key + 1u;
            if (key >= WORLD_CACHE_SIZE) { key = 0u; }
        }
    }

    var o: WcQueryOut;
    o.radiance = rad;
    o.cell_index = found_key;
    o.checksum = atomicLoad(&world_cache_checksums[found_key]);
    o.life = atomicLoad(&world_cache_life[found_key]);
    o._p0 = 0u;
    o._p1 = 0u;
    wc_query_out[i] = o;
}

// --- headless TEST entry: drive the ACTUAL initial-reservoir builders through the resolve --------------
// The convergence test (above) proves the cache FILLS to the analytic incoming radiance, and the restir_probe
// test proves the resolve constant. NEITHER exercises `reservoir_from_bounce_cached` (the live cache-fed
// initial reservoir) end-to-end — the only other coverage was a compile gate. This entry runs BOTH builders
// for one shading point whose fixed bounce direction hits the (already-cache-filled) floor, then resolves each
// to indirect irradiance, and reports the raw reservoir radiances + the deterministic cache value so the
// harness can PIN the energy relation that the 2.2 bug violated:
//     cache_on.radiance  ==  cache_off.radiance + albedo(hp) · cache(hp)
// i.e. the cache adds exactly ONE reflected indirect bounce (albedo·cache), on top of the fresh path's
// direct+emissive — NOT the raw cache (the bug) and NOT replacing direct+emissive (the prior reviewer's
// mistake). Both builders trace the SAME `dir`, so they share `hp`, `r.color`, `r.emissive`; the ONLY
// difference is the `+ albedo·cache` term, which is exactly what we assert. The `camera` uniform supplies the
// cache-LOD view position (group 1 binding 0).
struct EnergyProbeParams {
    shading_position: vec3<f32>, _p0: u32,
    shading_normal: vec3<f32>,   _p1: u32,
    bounce_dir: vec3<f32>,       _p2: u32,
};
struct EnergyProbeOut {
    cache_off_radiance: vec3<f32>, _p0: u32,   // reservoir_from_bounce(...).radiance  (fresh: direct+emissive)
    cache_on_radiance: vec3<f32>,  _p1: u32,   // reservoir_from_bounce_cached(...).radiance (adds albedo·cache)
    cache_off_irradiance: vec3<f32>, _p2: u32, // resolved indirect irradiance (cache OFF)
    cache_on_irradiance: vec3<f32>,  _p3: u32, // resolved indirect irradiance (cache ON)
    hit_albedo: vec3<f32>,         _p4: u32,   // albedo of the bounce-hit surface (the floor)
    cache_value: vec3<f32>,        _p5: u32,   // deterministic (no-jitter, LOD0) cache read at the hit cell
    hit: u32, _p6: u32, _p7: u32, _p8: u32,    // 1 = the bounce hit a surface (the relation is meaningful)
};

@group(0) @binding(8) var<uniform> energy_params: EnergyProbeParams;
@group(0) @binding(9) var<storage, read_write> energy_out: EnergyProbeOut;

// Deterministic (no-jitter, LOD0) cache read — mirrors `world_cache_query_seed`'s stable read-back so the
// harness sees the exact incoming radiance the floor cell holds, decoupled from `query_world_cache`'s jitter.
fn energy_read_cache_deterministic(world_position: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    let cell_size = wc.cell_base_size; // LOD 0 (test view is close), no jitter — stable read-back slot
    let qpos = bitcast<vec3<u32>>(wc_quantize_position(world_position, cell_size));
    let qnrm = bitcast<vec3<u32>>(wc_quantize_normal(world_normal));
    var key = wc_compute_key(qpos, qnrm);
    let checksum = wc_compute_checksum(qpos, qnrm);
    for (var s = 0u; s < WORLD_CACHE_MAX_SEARCH_STEPS; s = s + 1u) {
        let existing = atomicLoad(&world_cache_checksums[key]);
        if (existing == checksum) { return world_cache_radiance[key].rgb; }
        if (existing == WORLD_CACHE_EMPTY_CELL) { return vec3<f32>(0.0); }
        key = key + 1u;
        if (key >= WORLD_CACHE_SIZE) { key = 0u; }
    }
    return vec3<f32>(0.0);
}

@compute @workgroup_size(1, 1, 1)
fn world_cache_energy_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let p = energy_params.shading_position;
    let n = normalize(energy_params.shading_normal);
    let dir = normalize(energy_params.bounce_dir);
    var rng = (wc.frame_index * 26699u) | 1u;

    // The two REAL builders, same shading point + bounce direction (so they differ only by the cache term).
    let off = reservoir_from_bounce(p, n, dir);
    let on = reservoir_from_bounce_cached(p, n, dir, false, &rng); // energy gate: DI off → keep emissive

    // Re-trace to recover the bounce-hit geometry the relation references (albedo + the cache cell).
    let origin = p + n * light.shadow_bias;
    let r = trace(origin, dir, 0.0, light.gi_bounce_dist);
    let hp = origin + dir * r.t;

    energy_out.cache_off_radiance = off.radiance;
    energy_out.cache_on_radiance = on.radiance;
    energy_out.cache_off_irradiance = restir_resolve_irradiance(off, p, n);
    energy_out.cache_on_irradiance = restir_resolve_irradiance(on, p, n);
    energy_out.hit_albedo = r.color.rgb;
    energy_out.cache_value = energy_read_cache_deterministic(hp, r.normal);
    energy_out.hit = r.hit;
}

// --- headless TEST entry: thin-wall LIGHT-LEAK probe (Phase 2.2.1 regression gate) ----------------------
// Drives the REAL `query_world_cache` (so the first-bounce light-leak-prevention clamp is exercised exactly as
// in the live path) for each query point, using a caller-chosen SHORT `ray_t` (`wc_query_params.ray_t`) — the
// distance of the bounce that produced this query. A cube-face → adjacent-floor bounce is short (~0.3-0.8 m).
// Many RNG samples are averaged so the tangent-plane jitter (which, with an over-sized cell, stochastically
// crosses a thin wall into an exterior cell) is fully represented — the leak is "infrequent" precisely because
// only some jitter offsets cross. Without the clamp the averaged read is contaminated by the bright exterior
// cell; WITH the clamp the cell shrinks to `cell_base_size` (fits inside the wall) so the query NEVER reaches
// the exterior cell and the read stays ≈ the interior cell's (dark) radiance. Writes the averaged radiance to
// `wc_query_out[i].radiance` (cell_index/checksum/life unused here). `view_position` (params) sets the LOD so
// the harness can put the un-clamped cell size above the wall thickness.
const WC_LEAK_PROBE_SAMPLES: u32 = 256u;
@compute @workgroup_size(64, 1, 1)
fn world_cache_leak_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= wc_query_params.n_points) { return; }
    let q = wc_query_points[i];
    var rng = (wc_query_params.frame_index * 26699u + i * 747796405u) | 1u;
    var acc = vec3<f32>(0.0);
    for (var s = 0u; s < WC_LEAK_PROBE_SAMPLES; s = s + 1u) {
        acc += query_world_cache(
            q.world_position, q.world_normal, wc_query_params.view_position,
            wc_query_params.ray_t, wc.cell_lifetime, &rng);
    }
    var o: WcQueryOut;
    o.radiance = acc / f32(WC_LEAK_PROBE_SAMPLES);
    o.cell_index = 0u;
    o.checksum = 0u;
    o.life = 0u;
    o._p0 = 0u;
    o._p1 = 0u;
    wc_query_out[i] = o;
}
