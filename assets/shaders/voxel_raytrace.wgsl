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

// Brick geometry constants (mirror src/voxel/brickmap.rs). 8³ voxels of 0.2 m → 1.6 m brick.
const BRICK_EDGE: i32 = 8;
const VOXEL_SIZE: f32 = 0.2;
// World-metre extent of a brick (= BRICK_EDGE · VOXEL_SIZE = 1.6 m).
const BRICK_WORLD_SIZE: f32 = f32(BRICK_EDGE) * VOXEL_SIZE;
// The per-side AABB grow used to overlap abutting bricks — the SEAM fix. MUST mirror `BRICK_AABB_EPSILON`
// in src/voxel/gpu.rs. The shader recomputes the per-brick ray/AABB slab from the GROWN bounds (matching
// the BLAS geometry), so a ray grazing a shared face/edge enters the brick instead of falling in the FP gap
// between abutting AABBs. The DDA still reconstructs cells from the TRUE `world_min` and clamps into the
// real grid, so the halo never adds phantom voxels.
const BRICK_AABB_EPSILON: f32 = VOXEL_SIZE * 1.0e-3;

// Max LOD level (mirrors src/voxel/brickmap.rs MAX_LOD). LOD0 = 8³, LOD3 = 1³.
const MAX_LOD: u32 = 3u;

// One brick's metadata (parallel to the AABB / BLAS primitive array). 32 bytes — matches `GpuBrickMeta`.
// Variable per-brick LOD: the grid EDGE and per-cell world size are DERIVED from `lod`, not constant, so a
// coarse brick is DDA-marched over its coarse grid (`BRICK_EDGE >> lod` cells of `VOXEL_SIZE << lod` m).
struct BrickMeta {
    voxel_origin: vec3<i32>,  // brick's world-VOXEL origin (= brick_coord · BRICK_EDGE)
    voxel_offset: u32,        // start of this brick's lod_edge(lod)³ block ids in `voxels`
    world_min: vec3<f32>,     // brick's world-metre min corner
    lod: u32,                 // brick LOD level (0 = full 8³)
};

// The CORE grid edge (cells per axis) of a brick at LOD `lod`: BRICK_EDGE >> lod. SSOT mirror of `lod_edge`.
fn lod_edge(lod: u32) -> i32 {
    return BRICK_EDGE >> min(lod, MAX_LOD);
}

// The STORED grid edge: the core grid PLUS a 1-cell HALO border on every side (so `core + 2`). The packer
// fills the border with the adjacent NEIGHBOUR brick's boundary voxels (AIR where the neighbour is absent),
// so the DDA always crosses a real air→solid cell boundary AT the true surface — even when that surface lies
// exactly on a brick face. This is the seam fix: it gives the first-solid hit the correct entry-face normal
// (and an always-present boundary cell) from EVERY direction, instead of guessing the face at a brick corner
// (which produced the sideways normal → thin dark line at oblique angles). Mirrors `halo_edge` in gpu.rs.
fn halo_edge(lod: u32) -> i32 {
    return lod_edge(lod) + 2;
}

// The world-metre size of one cell of a brick at LOD `lod`: VOXEL_SIZE << lod. Mirror of `lod_voxel_size`.
fn lod_cell_size(lod: u32) -> f32 {
    return VOXEL_SIZE * f32(1u << min(lod, MAX_LOD));
}

// One palette entry (mirrors `GpuPaletteColor` in src/voxel/gpu.rs): linear-RGBA albedo + linear-RGB
// emissive radiance (`.xyz`; `.w` pad). 32 bytes.
struct Palette { rgba: vec4<f32>, emissive: vec4<f32> };

@group(0) @binding(0) var acc: acceleration_structure;
@group(0) @binding(1) var<storage, read> metas: array<BrickMeta>;
@group(0) @binding(2) var<storage, read> voxels: array<u32>;   // one BlockId per u32 (zero-extended u16)
@group(0) @binding(3) var<storage, read> palette: array<Palette>;

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
    let core = lod_edge(m.lod);        // CORE grid cells per axis at this brick's LOD
    let hedge = core + 2;              // STORED grid edge (core + 1-cell halo border each side)
    let csize = lod_cell_size(m.lod);  // world-metre size of one cell at this brick's LOD
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
        let id = voxels[m.voxel_offset + cell_index(vox.x, vox.y, vox.z, hedge)];
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
        || voxels[m.voxel_offset + cell_index(cnb.x, cnb.y, cnb.z, hedge)] == 0u;
    if (crossed_air) {
        grad[last_axis] = -step[last_axis]; // crisp crossed-axis face (outward = back along the ray)
    } else {
        // Grazing into a flat/buried surface → take the first EXPOSED face (a flat surface has exactly one).
        for (var a = 0; a < 3; a = a + 1) {
            if (grad.x == 0 && grad.y == 0 && grad.z == 0) {
                var pn = hit_vox; pn[a] = pn[a] + 1;
                var mn = hit_vox; mn[a] = mn[a] - 1;
                let p_air = pn[a] >= hedge || voxels[m.voxel_offset + cell_index(pn.x, pn.y, pn.z, hedge)] == 0u;
                let m_air = mn[a] < 0 || voxels[m.voxel_offset + cell_index(mn.x, mn.y, mn.z, hedge)] == 0u;
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
// (`±BRICK_AABB_EPSILON`) and the same march — so the recovered block id + normal are guaranteed to be those
// of the cell the candidate committed (no boundary drift, no sideways-normal seam). `dda_brick` clamps the
// entry cell into `[0, edge)`, so the grown halo never adds a phantom cell.
fn brick_hit_at(prim: u32, ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
    let m = metas[prim];
    let bmin = m.world_min - vec3<f32>(BRICK_AABB_EPSILON);
    let bmax = m.world_min + vec3<f32>(BRICK_WORLD_SIZE + BRICK_AABB_EPSILON);
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
    loop {
        if (!rayQueryProceed(&rq)) { break; }
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind == RAY_QUERY_INTERSECTION_AABB) {
            // The candidate carries the brick's primitive_index but NOT its t-range, so re-derive the
            // ray/AABB entry & exit from the brick bounds, then DDA between them.
            let m = metas[c.primitive_index];
            // Slab against the GROWN brick bounds (same overlap as the BLAS AABB — the seam fix). Using the
            // grown bounds keeps a face-grazing axis-parallel ray off the exact tangent plane (where the true
            // bounds give a 0·inf = NaN slab t), so it reliably enters the brick.
            let bmin = m.world_min - vec3<f32>(BRICK_AABB_EPSILON);
            let bmax = m.world_min + vec3<f32>(BRICK_WORLD_SIZE + BRICK_AABB_EPSILON);
            let inv = 1.0 / rd;
            let ta = (bmin - ro) * inv;
            let tb = (bmax - ro) * inv;
            let t_enter = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z));
            let t_exit  = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
            if (t_enter <= t_exit && t_exit >= t_min) {
                let bh = dda_brick(c.primitive_index, ro, rd, t_enter, t_exit);
                if (dh_found(bh)) {
                    let ht = dh_t(bh);
                    if (ht < best_t) {
                        best_t = ht;
                        best_prim = c.primitive_index;
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
    // sideways-normal seam). Called with a safe index on a miss; gated by `has_hit`.
    let prim = select(0u, best_prim, has_hit);
    let bh = brick_hit_at(prim, ro, rd);
    let id = select(0u, dh_block(bh), has_hit);
    if (has_hit) {
        r.hit = select(0u, 1u, id != 0u);
        r.block_id = id;
        r.prim = best_prim;
        r.t = best_t;
        r.color = palette[id].rgba;
        r.emissive = palette[id].emissive.xyz;
        r.normal = dh_normal(bh, rd);
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
            let m = metas[c.primitive_index];
            // Grown-bounds slab (matches the BLAS AABB overlap — the seam fix; see `trace`).
            let bmin = m.world_min - vec3<f32>(BRICK_AABB_EPSILON);
            let bmax = m.world_min + vec3<f32>(BRICK_WORLD_SIZE + BRICK_AABB_EPSILON);
            let inv = 1.0 / rd;
            let ta = (bmin - ro) * inv;
            let tb = (bmax - ro) * inv;
            let t_enter = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z));
            let t_exit  = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
            if (t_enter <= t_exit && t_exit >= t_min) {
                let bh = dda_brick(c.primitive_index, ro, rd, t_enter, t_exit);
                if (dh_found(bh) && dh_t(bh) <= t_max) {
                    rayQueryGenerateIntersection(&rq, dh_t(bh));
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
        // Direct term (AO-modulated ambient + shadowed sun), the indirect single-bounce GI (× albedo), and
        // the surface's own emissive glow — separated so the GI oracle can assert each independently. A fixed
        // seed keeps the headless GI estimate reproducible.
        let ndotl = max(dot(r.normal, to_sun), 0.0);
        var shadow = 1.0;
        if (shadowed == 1u) { shadow = 0.0; }
        let ao = ambient_occlusion(origin, r.normal);
        direct = r.color.rgb * (light.ambient_color * ao + light.sun_color * (light.sun_intensity * ndotl * shadow));
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
//   ao_samples    (4)  + gi_rays (4) + gi_intensity (4) + gi_bounce_dist (4)
//   emissive_strength (4) + frame_index (4) + debug_view (4) + gi_firefly_clamp (4)
struct LightingUniform {
    sun_direction: vec3<f32>,  // normalized direction the sunlight travels (points away from the sun)
    sun_intensity: f32,        // scalar multiplier on sun_color
    sun_color: vec3<f32>,      // linear RGB of the sun
    shadow_bias: f32,          // world-metre normal offset for shadow/AO ray origins (avoids self-hit)
    ambient_color: vec3<f32>,  // linear RGB sky/ambient fill
    ao_radius: f32,            // AO ray length in world metres
    ao_samples: u32,           // number of AO rays in the hemisphere (0 disables AO → ao = 1)
    gi_rays: u32,              // cosine-sampled diffuse bounce rays per pixel (0 disables GI)
    gi_intensity: f32,         // scalar multiplier on accumulated indirect irradiance
    gi_bounce_dist: f32,       // max world-metre length of a diffuse bounce ray (miss past it = sky)
    emissive_strength: f32,    // scalar multiplier on every block's palette emissive
    frame_index: u32,          // per-frame counter to decorrelate the bounce-direction hash
    debug_view: u32,           // 0 = lit; 1 = normals; 2 = depth; 3 = albedo; 4 = AO; 5 = GI-only; 6 = face-toward-camera
    gi_firefly_clamp: f32,     // max per-bounce-sample GI radiance (0 = unclamped); tames emissive fireflies/boil
};
@group(1) @binding(2) var<uniform> light: LightingUniform;

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
    return albedo * (light.ambient_color + direct);
}

// The sky/ambient radiance a MISSED ray returns (a diffuse bounce that escapes to open sky). Keep it
// consistent with the primary-miss sky and the ambient fill so GI energy is plausible — a bounce into the
// sky brings back roughly the ambient term (the same hemispheric fill the direct ambient uses).
fn bounce_sky() -> vec3<f32> {
    return light.ambient_color;
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

// Single-bounce diffuse GLOBAL ILLUMINATION at a surface hit. Cosine-sample `gi_rays` directions in the
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
    let rays = min(light.gi_rays, 32u);
    if (rays == 0u || light.gi_intensity <= 0.0) {
        return vec3<f32>(0.0);
    }
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
            contrib = bounce_sky();
        }
        // Firefly clamp: cap a single bounce sample's radiance so one very bright hit (e.g. a grazing ray
        // catching the emissive panel) can't dominate the mean and pop/boil under temporal denoise. Scales
        // the whole sample so hue is preserved. `gi_firefly_clamp == 0` disables it (unbiased; tests rely on
        // this). A small bias for a large variance cut — standard path-tracer practice.
        if (light.gi_firefly_clamp > 0.0) {
            let m = max(contrib.r, max(contrib.g, contrib.b));
            if (m > light.gi_firefly_clamp) {
                contrib = contrib * (light.gi_firefly_clamp / m);
            }
        }
        acc_rad = acc_rad + contrib;
    }
    // Cosine-pdf importance sampling ⇒ the irradiance estimate is the mean of the gathered radiance.
    return (acc_rad / f32(rays)) * light.gi_intensity;
}

// Compose the FINAL surface colour at the primary hit: direct lighting (with traced AO on the ambient
// fill) + single-bounce indirect GI (× receiver albedo) + the surface's OWN emissive glow. `albedo` is the
// palette colour, `n` the face normal, `p` the world hit point, `emissive` the palette emissive radiance.
// Output is LINEAR HDR — Bevy tonemaps downstream.
fn shade(albedo: vec3<f32>, n: vec3<f32>, p: vec3<f32>, emissive: vec3<f32>, seed: u32) -> vec3<f32> {
    let origin = p + n * light.shadow_bias;
    let to_sun = -light.sun_direction;
    let ndotl = max(dot(n, to_sun), 0.0);
    var shadow = 1.0;
    if (ndotl > 0.0) {
        if (trace_occluded(origin, to_sun, 0.0, 1.0e4)) {
            shadow = 0.0;
        }
    }
    let ao = ambient_occlusion(origin, n);
    let ambient = light.ambient_color * ao;
    let direct = light.sun_color * (light.sun_intensity * ndotl * shadow);
    // Indirect single-bounce GI: gathered irradiance × this surface's albedo (Lambertian reflection).
    let indirect = gather_gi(n, p, seed) * albedo;
    // The surface's own emissive glow (so an emitter block visibly lights up, not just its neighbours).
    let glow = emissive * light.emissive_strength;
    return albedo * (ambient + direct) + indirect + glow;
}

@compute @workgroup_size(8, 8, 1)
fn raymarch(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= camera.viewport.x || gid.y >= camera.viewport.y) { return; }
    // Pixel centre → NDC. Bevy/wgpu clip space: x∈[-1,1] right, y∈[-1,1] UP (flip the texel row).
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / vec2<f32>(camera.viewport);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    // Unproject the NEAR plane to get a finite world point on the ray. Bevy uses an INFINITE-far reverse-Z
    // perspective (near at z=1, far at z=0 → infinity), so unprojecting z=0 yields w=0 → a NaN point and
    // every ray misses. The near plane (z=1) always unprojects to a finite point; the ray direction from the
    // camera through it is identical to the true primary-ray direction.
    let near = camera.world_from_clip * vec4<f32>(ndc, 1.0, 1.0);
    let world_near = near.xyz / near.w;
    let ro = camera.cam_pos;
    let rd = normalize(world_near - ro);

    let r = trace(ro, rd, 0.0, camera.t_max);

    // --- Debug overlays (RAW output, no temporal accumulation, so they stay crisp under motion) ----------
    if (light.debug_view != 0u) {
        let dpx = vec2<i32>(i32(gid.x), i32(gid.y));
        var dbg = vec3<f32>(0.0);
        if (r.hit != 0u) {
            let p = ro + rd * r.t;
            let origin = p + r.normal * light.shadow_bias;
            if (light.debug_view == 1u) {
                dbg = r.normal * 0.5 + 0.5;                                  // world-space face normals
            } else if (light.debug_view == 2u) {
                dbg = vec3<f32>(clamp(r.t / 20.0, 0.0, 1.0));                // depth (0..20 m → black..white)
            } else if (light.debug_view == 3u) {
                dbg = r.color.rgb;                                          // raw palette albedo
            } else if (light.debug_view == 4u) {
                dbg = vec3<f32>(ambient_occlusion(origin, r.normal));       // AO only
            } else if (light.debug_view == 5u) {
                let seed = (gid.x * 1973u + gid.y * 9277u + light.frame_index * 26699u) | 1u;
                dbg = gather_gi(r.normal, origin, seed);                    // indirect (GI) only
            } else if (light.debug_view == 6u) {
                // Face orientation: GREEN = front face (normal opposes the ray, correct); RED = BACK face
                // (normal points ALONG the ray — i.e. we hit the inside/back of a voxel = the show-through bug).
                dbg = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), dot(r.normal, rd) > 0.0);
            } else {
                dbg = r.color.rgb;
            }
        }
        textureStore(out_tex, dpx, vec4<f32>(dbg, 1.0));
        return;
    }

    var color: vec4<f32>;
    if (r.hit != 0u) {
        // Hit: physically-plausible DIRECT lighting (Lambert sun + traced hard shadow + traced AO over the
        // palette albedo), fully opaque (replaces the view). Linear HDR — Bevy tonemaps downstream.
        let p = ro + rd * r.t;
        // Per-pixel + per-frame seed for the GI bounce-direction hash (decorrelates noise spatially and
        // animates it across frames so a future temporal accumulator can average it out).
        let seed = (gid.x * 1973u + gid.y * 9277u + light.frame_index * 26699u) | 1u;
        let lit = shade(r.color.rgb, r.normal, p, r.emissive, seed);
        color = vec4<f32>(lit, 1.0);
    } else {
        // Miss: a simple vertical sky gradient (horizon → zenith) keyed off the ray's up component, fully
        // opaque. This makes the HW-RT view a complete renderer (no cube crutch to show through) AND lets the
        // headless oracle distinguish "rays ran but missed" (sky) from "the composite never ran" (clear
        // colour). Linear-space colours — tonemapping converts them to display.
        let up = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
        let horizon = vec3<f32>(0.55, 0.65, 0.78);
        let zenith = vec3<f32>(0.12, 0.22, 0.45);
        color = vec4<f32>(mix(horizon, zenith, up), 1.0);
    }

    // --- Temporal accumulation (denoise the per-frame GI noise) ---------------------------------------
    // Blend this frame's shaded colour into the running history mean. `accum_weight` is 1/sample_count: the
    // renderer holds it at 1.0 on the frame the camera moves (full reset — show the fresh frame), then ramps
    // it down (1/2, 1/3, …) while the camera is still, so the displayed value converges to the average of all
    // frames since the last move. Because the GI bounce directions are decorrelated by `frame_index`, that
    // average is a Monte-Carlo estimate whose variance falls ~1/n → the sparkle vanishes over a few dozen
    // frames. RGB only (alpha is the hit mask, kept from the current frame). The history is the PREVIOUS
    // accumulated output, copied back after this pass by the render system.
    let prev = textureSampleLevel(history_tex, history_sampler, uv, 0.0).rgb;
    let w = clamp(camera.accum_weight, 0.0, 1.0);
    let accumulated = mix(prev, color.rgb, w);
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(accumulated, color.a));
}

// --- DLSS Ray Reconstruction entry point ------------------------------------------------------------
// (Stage 4c.) When built with `--features dlss`, the renderer runs THIS entry instead of `raymarch`. It
// writes the per-pixel inputs DLSS-RR consumes:
//   * out_tex (the HDR view colour → DLSS `color`) — the FULL noisy LIT colour (albedo × lighting + glow).
//     DLSS-RR DEMODULATES internally using the albedo guides below, denoises the lighting, then re-modulates;
//     so we pass the full radiance, NOT a pre-divided signal. This matches the validated Solari contract
//     (its RestIR shaders write `radiance × brdf` to the view target and supply albedo guides alongside).
//   * diffuse_albedo   (rgba8)   — the voxel palette albedo (DLSS's demodulation guide)
//   * specular_albedo  (rgba8)   — a tiny dielectric F0 floor for these matte diffuse voxels (~non-specular)
//   * normal_roughness (rgba16f) — world-space face normal (xyz) + perceptual roughness (w ≈ 1.0, matte)
//   * out_dlss_depth   (r32f)    — the raymarch hit's reverse-Z clip depth (matches Bevy's depth prepass)
//   * out_dlss_motion  (rg16f)   — screen-space motion: this pixel's hit reprojected into the PREVIOUS frame
// There is NO temporal accumulation here — DLSS-RR IS the denoiser. The guides are written by THIS compute;
// the resolve render pass (`resolve_dlss` in voxel_rt_composite.wgsl) copies depth+motion into the
// RENDER-ATTACHMENT-only prepass textures (which compute can't storage-write) and the colour into the view
// target. `shade` (the non-dlss composer) is reused verbatim for the colour, so the lit look is identical.
@group(1) @binding(5) var out_diffuse_albedo: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(6) var out_specular_albedo: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(7) var out_normal_roughness: texture_storage_2d<rgba16float, write>;
@group(1) @binding(8) var out_dlss_depth: texture_storage_2d<r32float, write>;
// Motion is an intermediate STORAGE texture; `rg16float` storage isn't universally supported, so use a
// widely-supported `rgba16float` (only .xy carry the screen-space motion; .zw are 0). The resolve pass writes
// the final `Rg16Float` PREPASS motion texture via a render attachment (no storage requirement there).
@group(1) @binding(9) var out_dlss_motion: texture_storage_2d<rgba16float, write>;

// DLSS camera matrices. `depth_clip_from_world` is JITTERED (matches Bevy's jittered reverse-Z depth prepass
// — used only for the depth write). `motion_cur`/`motion_prev` are UN-JITTERED clip_from_world for the
// PREVIOUS and CURRENT frame: the motion vector must encode GEOMETRY/camera motion only, because DLSS is
// given the sub-pixel jitter offset separately (via the TemporalJitter component) and resolves it itself.
// Differencing jittered matrices would double-count the jitter → a per-frame sub-pixel "shake" (the bug).
struct DlssCamera {
    depth_clip_from_world: mat4x4<f32>,
    motion_prev: mat4x4<f32>,
    motion_cur: mat4x4<f32>,
};
@group(1) @binding(10) var<uniform> dlss_cam: DlssCamera;

@compute @workgroup_size(8, 8, 1)
fn raymarch_dlss(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= camera.viewport.x || gid.y >= camera.viewport.y) { return; }
    let px = vec2<i32>(i32(gid.x), i32(gid.y));
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / vec2<f32>(camera.viewport);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let near = camera.world_from_clip * vec4<f32>(ndc, 1.0, 1.0);
    let world_near = near.xyz / near.w;
    let ro = camera.cam_pos;
    let rd = normalize(world_near - ro);

    let r = trace(ro, rd, 0.0, camera.t_max);
    if (r.hit != 0u) {
        let p = ro + rd * r.t;
        let seed = (gid.x * 1973u + gid.y * 9277u + light.frame_index * 26699u) | 1u;
        // FULL noisy lit colour (same `shade` as the non-dlss path) → DLSS demodulates with the albedo guide.
        let lit = shade(r.color.rgb, r.normal, p, r.emissive, seed);
        textureStore(out_tex, px, vec4<f32>(lit, 1.0));
        textureStore(out_diffuse_albedo, px, vec4<f32>(r.color.rgb, 1.0));
        // Matte diffuse voxels: a tiny dielectric specular floor (F0 ≈ 0.04), near-black so DLSS treats
        // them as non-specular. Keeps the specular guide valid (all-black confuses some DLSS paths).
        textureStore(out_specular_albedo, px, vec4<f32>(vec3<f32>(0.04), 1.0));
        textureStore(out_normal_roughness, px, vec4<f32>(r.normal, 1.0));

        // True reverse-Z clip depth of the hit (JITTERED, matching Bevy's jittered reverse-Z depth prepass).
        let depth_clip = dlss_cam.depth_clip_from_world * vec4<f32>(p, 1.0);
        textureStore(out_dlss_depth, px, vec4<f32>(depth_clip.z / depth_clip.w, 0.0, 0.0, 0.0));

        // Screen-space motion = where this hit point WAS vs IS, from the UN-JITTERED matrices (geometry motion
        // only; DLSS adds the jitter offset itself). `(cur − prev)·(0.5,−0.5)` matches Bevy's prepass; the DLSS
        // node's motion_vector_scale = −render_res converts the UV delta to pixels. ~0 for a static frame.
        let prev_clip = dlss_cam.motion_prev * vec4<f32>(p, 1.0);
        let cur_clip = dlss_cam.motion_cur * vec4<f32>(p, 1.0);
        let prev_ndc = prev_clip.xy / prev_clip.w;
        let cur_ndc = cur_clip.xy / cur_clip.w;
        let motion = (cur_ndc - prev_ndc) * vec2<f32>(0.5, -0.5);
        textureStore(out_dlss_motion, px, vec4<f32>(motion, 0.0, 0.0));
    } else {
        // Miss: sky into the colour, far depth (0 in reverse-Z), no motion, no albedo (so DLSS doesn't
        // re-modulate sky with a stale albedo), default normal.
        let up = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
        let horizon = vec3<f32>(0.55, 0.65, 0.78);
        let zenith = vec3<f32>(0.12, 0.22, 0.45);
        textureStore(out_tex, px, vec4<f32>(mix(horizon, zenith, up), 1.0));
        textureStore(out_diffuse_albedo, px, vec4<f32>(1.0, 1.0, 1.0, 1.0));
        textureStore(out_specular_albedo, px, vec4<f32>(vec3<f32>(0.0), 1.0));
        textureStore(out_normal_roughness, px, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(out_dlss_depth, px, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(out_dlss_motion, px, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }
}

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

// Generate the per-shading-point INITIAL reservoir: one uniform-hemisphere bounce ray → its sample point +
// outgoing radiance. Misses contribute nothing (the closed Cornell box rarely lets a bounce escape; the sky
// term is negligible there, matching the assumption the energy test relies on).
fn generate_initial_reservoir(world_position: vec3<f32>, world_normal: vec3<f32>, rng: ptr<function, u32>) -> Reservoir {
    var reservoir = empty_reservoir();
    let dir = sample_uniform_hemisphere(world_normal, rng);
    let origin = world_position + world_normal * light.shadow_bias;
    let r = trace(origin, dir, 0.0, light.gi_bounce_dist);
    if (r.hit == 0u) {
        return reservoir;
    }
    let hp = origin + dir * r.t;
    reservoir.sample_point_world_position = hp;
    reservoir.sample_point_world_normal = r.normal;
    reservoir.confidence_weight = 1.0;
    // L_o at the sample point = direct lighting there + its emissive (emissive INCLUDED — the adaptation).
    reservoir.radiance = direct_lighting(r.color.rgb, r.normal, hp) + r.emissive * light.emissive_strength;
    // Firefly clamp (hue-preserving), same as `gather_gi`. ReSTIR STORES + reuses samples, so an unclamped
    // bright outlier (a grazing emissive hit) persists + propagates across the buffer — worse than in the
    // plain mean. `gi_firefly_clamp == 0` disables it (the unbiased probe test sets 0). Small bias for stability.
    if (light.gi_firefly_clamp > 0.0) {
        let mx = max(reservoir.radiance.r, max(reservoir.radiance.g, reservoir.radiance.b));
        if (mx > light.gi_firefly_clamp) {
            reservoir.radiance = reservoir.radiance * (light.gi_firefly_clamp / mx);
        }
    }
    reservoir.unbiased_contribution_weight = uniform_hemisphere_inverse_pdf();
    return reservoir;
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
    return res.radiance * res.unbiased_contribution_weight * cos * (1.0 / RESTIR_PI) * light.gi_intensity;
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
@group(0) @binding(11) var<uniform> probe_params: RestirProbeParams;

@compute @workgroup_size(64)
fn restir_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= probe_params.n_probes) { return; }
    let probe = probes_in[i];
    let pos = probe.world_position;
    let n = probe.world_normal;

    var rng = (i * 9781u + probe_params.frame_index * 26699u) | 1u;

    var canonical = generate_initial_reservoir(pos, n, &rng);

    // Temporal reuse: merge the probe's previous-dispatch reservoir (same surface → Jacobian 1), unless reset.
    if (probe_params.reset == 0u) {
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
    // Reference: the established cosine-mean GI estimator at this probe (high gi_rays driven by the harness).
    out.reference = gather_gi(n, pos, (i * 2654435761u + probe_params.frame_index * 40503u) | 1u);
    // Per-FRAME slot so the harness reads the whole convergence history back in one map.
    probe_out[probe_params.frame_index * probe_params.n_probes + i] = out;
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
// history (lag vs stability). 32 bytes (2×vec4).
struct RestirParams {
    reset: u32,
    frame_index: u32,
    viewport_x: u32,
    viewport_y: u32,
    spatial_samples: u32,
    confidence_weight_cap: f32,
    spatial_radius: f32,
    _pad: u32,
};
// Per-pixel RECEIVER surface (world pos + face normal) — needed so a temporal/neighbour reservoir can be
// merged with the correct Jacobian + rejected when it lands on a dissimilar surface (port of Solari's
// gbuffer-resolve, but we store pos/normal directly instead of repacking depth).
struct PixelSurface { world_position: vec3<f32>, valid: f32, world_normal: vec3<f32>, _pad: f32 };
@group(2) @binding(0) var<storage, read_write> reservoirs_cur: array<Reservoir>;
@group(2) @binding(1) var<storage, read_write> reservoirs_prev: array<Reservoir>;
@group(2) @binding(2) var<uniform> restir_params: RestirParams;
@group(2) @binding(3) var<storage, read_write> surfaces_cur: array<PixelSurface>;
@group(2) @binding(4) var<storage, read_write> surfaces_prev: array<PixelSurface>;

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
    let view_dist = max(length(p - camera.cam_pos), 1.0e-3);
    // Solari thresholds: reject if the neighbour is >0.3% of view-distance out of the tangent plane, or its
    // normal is >90° away. (Looser values leak GI across co-planar patches with different occlusion.)
    return tangent_plane_distance / view_dist > 0.003 || dot(n, on) < 0.0;
}

// Uniform sample in a disk of `radius` pixels (concentric area-uniform), for spatial-neighbour selection.
fn sample_disk(radius: f32, rng: ptr<function, u32>) -> vec2<f32> {
    let r = radius * sqrt(rand_next(rng));
    let a = 6.2831853 * rand_next(rng);
    return vec2<f32>(r * cos(a), r * sin(a));
}

// Reservoir-based indirect irradiance at pixel `idx` for surface (n, p): generate an initial candidate, merge
// the previous frame's reservoir for this pixel (unless reset), store the merged reservoir, and resolve it.
// Writes `reservoirs_cur[idx]` for the caller's pixel. Returns the indirect irradiance (× albedo by caller).
// `temporal_base` is the PREVIOUS-frame pixel this surface reprojects to (motion-vector reprojection done by
// the caller; == `pix` for a still camera or the non-DLSS path). Reprojection lets temporal accumulation
// CONTINUE through camera motion instead of resetting — disocclusions are caught by the dissimilarity reject.
fn restir_gi(n: vec3<f32>, p: vec3<f32>, pix: vec2<u32>, temporal_base: vec2<i32>, seed: u32) -> vec3<f32> {
    let vp = vec2<u32>(restir_params.viewport_x, restir_params.viewport_y);
    let idx = pix.y * vp.x + pix.x;
    surfaces_cur[idx] = PixelSurface(p, 1.0, n, 0.0); // this pixel's receiver surface (for neighbours/next frame)
    if (light.gi_rays == 0u || light.gi_intensity <= 0.0) {
        reservoirs_cur[idx] = empty_reservoir();
        return vec3<f32>(0.0);
    }
    var rng = seed;
    let brdf = vec3<f32>(1.0); // receiver albedo factored out (applied by the caller)

    // INITIAL RIS over `gi_rays` candidates (same per-pixel sample budget as the legacy `gather_gi`): cuts the
    // per-pixel variance ~1/M so each frame is smooth. Same-surface merges (Jacobian = 1). Counts as ONE frame
    // (confidence 1) so the temporal reuse below stays strong.
    let m = min(light.gi_rays, 32u);
    var res = generate_initial_reservoir(p, n, &rng);
    for (var i = 1u; i < m; i = i + 1u) {
        let cand = generate_initial_reservoir(p, n, &rng);
        let merged = merge_reservoirs(res, p, n, brdf, cand, p, n, brdf, &rng);
        res = merged.merged_reservoir;
    }
    res.confidence_weight = 1.0;

    // TEMPORAL + light-spatial reuse. CRUCIAL: read a PERMUTED previous-frame neighbour, NOT this pixel's own
    // previous reservoir — same-pixel feedback freezes each pixel onto an early sample (grain that fades in);
    // the permute decorrelates it (folds in genuinely new info each frame → variance falls, not grows). Reject
    // dissimilar surfaces (no GI leak across edges) and merge with the NEIGHBOUR's surface so the Jacobian is
    // correct. Skipped on reset (camera move / re-pack).
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
            var temporal = reservoirs_prev[tidx];
            temporal.confidence_weight = min(temporal.confidence_weight, restir_params.confidence_weight_cap);
            let merged =
                merge_reservoirs(res, p, n, brdf, temporal, surf.world_position, surf.world_normal, brdf, &rng);
            res = merged.merged_reservoir;
        }

        // SPATIAL reuse: merge exactly ONE valid neighbour per frame (Solari `load_spatial_reservoir` / RTXDI /
        // Wyman-2023). `spatial_samples` is the SEARCH BUDGET — how many disk taps to try to find a
        // geometrically-valid neighbour — NOT an accumulation count. Merging many neighbours via iterated
        // pairwise merges is biased (the balance-heuristic MIS partition Σm_i=1 only holds for two reservoirs)
        // and inflates the combined confidence unboundedly, which AMPLIFIES variance with more samples (the
        // "more spatial → more boil" bug). The effective sample count is built by TEMPORAL accumulation; one
        // clean 2-reservoir spatial merge per frame keeps confidence bounded and variance falling.
        for (var s = 0u; s < restir_params.spatial_samples; s = s + 1u) {
            let off = sample_disk(restir_params.spatial_radius, &rng);
            let npix = vec2<i32>(pix) + vec2<i32>(i32(round(off.x)), i32(round(off.y)));
            if (npix.x < 0 || npix.y < 0 || npix.x >= i32(vp.x) || npix.y >= i32(vp.y)) {
                continue;
            }
            let nidx = u32(npix.y) * vp.x + u32(npix.x);
            let nsurf = surfaces_prev[nidx];
            if (nsurf.valid > 0.5 && !surfaces_dissimilar(p, n, nsurf.world_position, nsurf.world_normal)) {
                var nres = reservoirs_prev[nidx];
                nres.confidence_weight = min(nres.confidence_weight, restir_params.confidence_weight_cap);
                let merged =
                    merge_reservoirs(res, p, n, brdf, nres, nsurf.world_position, nsurf.world_normal, brdf, &rng);
                res = merged.merged_reservoir;
                break; // one neighbour only — temporal accumulation provides the rest
            }
        }
    }

    // Store the UNBIASED reservoir (true ucw) BEFORE the visibility test — Solari's unbiased path. The stored
    // reservoir must remain an unbiased estimate of incident radiance, because NEIGHBOURS resample it next
    // frame; visibility is a per-RECEIVER shading correction only. Baking THIS pixel's occlusion into the
    // stored reservoir and then reusing it at other pixels is exactly what makes bright (e.g. red-wall)
    // samples diffuse across the buffer over frames (the leak). So: store first, then shade with a throwaway
    // visibility-corrected copy.
    // Robustness: never persist a non-finite contribution weight — a stored NaN/Inf is reused forever (a
    // permanent dead pixel). (A1's balance-heuristic form should prevent the source; this is belt-and-braces.)
    if (restir_isnan(res.unbiased_contribution_weight) || restir_isinf(res.unbiased_contribution_weight)) {
        res.unbiased_contribution_weight = 0.0;
    }
    reservoirs_cur[idx] = res;
    var shaded = res;
    if (shaded.confidence_weight > 0.0) {
        let origin = p + n * light.shadow_bias;
        let to_sample = shaded.sample_point_world_position - origin;
        let dist = length(to_sample);
        // Pull t_max back by one shadow_bias so the ray doesn't self-hit the sample point's own surface.
        if (dist > light.shadow_bias
            && trace_occluded(origin, to_sample / dist, 0.0, dist - light.shadow_bias)) {
            shaded.unbiased_contribution_weight = 0.0;
        }
    }
    return restir_resolve_irradiance(shaded, p, n);
}

// Like `shade`, but the indirect term comes from the per-pixel reservoir (ReSTIR) instead of `gather_gi`.
fn shade_restir(albedo: vec3<f32>, n: vec3<f32>, p: vec3<f32>, emissive: vec3<f32>, pix: vec2<u32>, temporal_base: vec2<i32>, seed: u32) -> vec3<f32> {
    let origin = p + n * light.shadow_bias;
    let to_sun = -light.sun_direction;
    let ndotl = max(dot(n, to_sun), 0.0);
    var shadow = 1.0;
    if (ndotl > 0.0) {
        if (trace_occluded(origin, to_sun, 0.0, 1.0e4)) {
            shadow = 0.0;
        }
    }
    let ao = ambient_occlusion(origin, n);
    let ambient = light.ambient_color * ao;
    let direct = light.sun_color * (light.sun_intensity * ndotl * shadow);
    let indirect = restir_gi(n, p, pix, temporal_base, seed) * albedo;
    let glow = emissive * light.emissive_strength;
    return albedo * (ambient + direct) + indirect + glow;
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

// The ReSTIR variant of `raymarch` (non-DLSS path). Identical primary-ray + debug + temporal-accumulation
// structure, but the lit colour uses `shade_restir` (reservoir GI). The on-top history accumulation further
// smooths the (already low-variance) ReSTIR output.
@compute @workgroup_size(8, 8, 1)
fn raymarch_restir(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= camera.viewport.x || gid.y >= camera.viewport.y) { return; }
    let idx = gid.y * camera.viewport.x + gid.x;
    reservoirs_cur[idx] = empty_reservoir(); // default for misses / debug; overwritten for lit hits
    surfaces_cur[idx] = PixelSurface(vec3<f32>(0.0), 0.0, vec3<f32>(0.0), 0.0); // invalid until a lit hit

    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / vec2<f32>(camera.viewport);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let near = camera.world_from_clip * vec4<f32>(ndc, 1.0, 1.0);
    let world_near = near.xyz / near.w;
    let ro = camera.cam_pos;
    let rd = normalize(world_near - ro);

    let r = trace(ro, rd, 0.0, camera.t_max);

    if (light.debug_view != 0u) {
        let dpx = vec2<i32>(i32(gid.x), i32(gid.y));
        var dbg = vec3<f32>(0.0);
        if (r.hit != 0u) {
            let p = ro + rd * r.t;
            let origin = p + r.normal * light.shadow_bias;
            if (light.debug_view == 1u) {
                dbg = r.normal * 0.5 + 0.5;
            } else if (light.debug_view == 2u) {
                dbg = vec3<f32>(clamp(r.t / 20.0, 0.0, 1.0));
            } else if (light.debug_view == 3u) {
                dbg = r.color.rgb;
            } else if (light.debug_view == 4u) {
                dbg = vec3<f32>(ambient_occlusion(origin, r.normal));
            } else if (light.debug_view == 5u) {
                let seed = (gid.x * 1973u + gid.y * 9277u + light.frame_index * 26699u) | 1u;
                dbg = restir_gi(r.normal, p, gid.xy, vec2<i32>(gid.xy), seed); // GI-only debug = reservoir estimate
            } else if (light.debug_view == 6u) {
                dbg = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), dot(r.normal, rd) > 0.0);
            } else {
                dbg = r.color.rgb;
            }
        }
        textureStore(out_tex, dpx, vec4<f32>(dbg, 1.0));
        return;
    }

    var color: vec4<f32>;
    if (r.hit != 0u) {
        let p = ro + rd * r.t;
        let seed = (gid.x * 1973u + gid.y * 9277u + light.frame_index * 26699u) | 1u;
        // Non-DLSS path has no previous clip → no reprojection (it resets on move); base = current pixel.
        let lit = shade_restir(r.color.rgb, r.normal, p, r.emissive, gid.xy, vec2<i32>(gid.xy), seed);
        color = vec4<f32>(lit, 1.0);
    } else {
        let up = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
        let horizon = vec3<f32>(0.55, 0.65, 0.78);
        let zenith = vec3<f32>(0.12, 0.22, 0.45);
        color = vec4<f32>(mix(horizon, zenith, up), 1.0);
    }

    let prev = textureSampleLevel(history_tex, history_sampler, uv, 0.0).rgb;
    let w = clamp(camera.accum_weight, 0.0, 1.0);
    let accumulated = mix(prev, color.rgb, w);
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(accumulated, color.a));
}

// The ReSTIR variant of `raymarch_dlss`. Same guide-writing contract; the lit colour uses `shade_restir`, so
// DLSS-RR is fed a LOW-VARIANCE indirect term (the reservoir already integrated many frames) — RR then only
// has to clean a near-converged signal, which is what fixes the residual boiling under DLSS.
@compute @workgroup_size(8, 8, 1)
fn raymarch_dlss_restir(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= camera.viewport.x || gid.y >= camera.viewport.y) { return; }
    let idx = gid.y * camera.viewport.x + gid.x;
    reservoirs_cur[idx] = empty_reservoir();
    surfaces_cur[idx] = PixelSurface(vec3<f32>(0.0), 0.0, vec3<f32>(0.0), 0.0); // invalid until a lit hit
    let px = vec2<i32>(i32(gid.x), i32(gid.y));
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / vec2<f32>(camera.viewport);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let near = camera.world_from_clip * vec4<f32>(ndc, 1.0, 1.0);
    let world_near = near.xyz / near.w;
    let ro = camera.cam_pos;
    let rd = normalize(world_near - ro);

    let r = trace(ro, rd, 0.0, camera.t_max);
    if (r.hit != 0u) {
        let p = ro + rd * r.t;
        let seed = (gid.x * 1973u + gid.y * 9277u + light.frame_index * 26699u) | 1u;
        // Reproject the temporal tap via the UN-jittered previous clip so accumulation continues under motion.
        let temporal_base = reproject_pixel(p, dlss_cam.motion_prev, camera.viewport);
        let lit = shade_restir(r.color.rgb, r.normal, p, r.emissive, gid.xy, temporal_base, seed);
        textureStore(out_tex, px, vec4<f32>(lit, 1.0));
        textureStore(out_diffuse_albedo, px, vec4<f32>(r.color.rgb, 1.0));
        textureStore(out_specular_albedo, px, vec4<f32>(vec3<f32>(0.04), 1.0));
        textureStore(out_normal_roughness, px, vec4<f32>(r.normal, 1.0));
        // Depth JITTERED (matches the jittered depth prepass); motion UN-JITTERED (geometry only — DLSS adds
        // the jitter), `(cur − prev)·(0.5,−0.5)`. Jittered motion would double-count jitter ⇒ a sub-pixel shake.
        let depth_clip = dlss_cam.depth_clip_from_world * vec4<f32>(p, 1.0);
        textureStore(out_dlss_depth, px, vec4<f32>(depth_clip.z / depth_clip.w, 0.0, 0.0, 0.0));
        let prev_clip = dlss_cam.motion_prev * vec4<f32>(p, 1.0);
        let cur_clip = dlss_cam.motion_cur * vec4<f32>(p, 1.0);
        let prev_ndc = prev_clip.xy / prev_clip.w;
        let cur_ndc = cur_clip.xy / cur_clip.w;
        let motion = (cur_ndc - prev_ndc) * vec2<f32>(0.5, -0.5);
        textureStore(out_dlss_motion, px, vec4<f32>(motion, 0.0, 0.0));
    } else {
        let up = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
        let horizon = vec3<f32>(0.55, 0.65, 0.78);
        let zenith = vec3<f32>(0.12, 0.22, 0.45);
        textureStore(out_tex, px, vec4<f32>(mix(horizon, zenith, up), 1.0));
        textureStore(out_diffuse_albedo, px, vec4<f32>(1.0, 1.0, 1.0, 1.0));
        textureStore(out_specular_albedo, px, vec4<f32>(vec3<f32>(0.0), 1.0));
        textureStore(out_normal_roughness, px, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(out_dlss_depth, px, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(out_dlss_motion, px, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }
}
