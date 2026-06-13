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
//   emissive_strength (4) + frame_index (4) + _pad (8)
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
    _pad1: u32,
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
    var acc_rad = vec3<f32>(0.0);
    for (var i = 0u; i < rays; i = i + 1u) {
        // Two decorrelated uniforms per ray from the pixel/frame seed.
        let s = seed_base + i * 9781u;
        let u1 = rand01(s * 2u + 1u);
        let u2 = rand01(s * 2u + 2u);
        // Cosine-weighted hemisphere sample (Malley / concentric-ish): r = sqrt(u1), phi = 2π·u2; z = sqrt(1-u1).
        let r = sqrt(u1);
        let phi = 6.2831853 * u2;
        let x = r * cos(phi);
        let y = r * sin(phi);
        let z = sqrt(max(0.0, 1.0 - u1));
        let dir = normalize(tang * x + bitang * y + n * z);
        // Trace the bounce ray (bounded to gi_bounce_dist) and gather incoming radiance.
        let h = trace(origin, dir, 0.0, light.gi_bounce_dist);
        if (h.hit != 0u) {
            let hp = origin + dir * h.t;
            let surf = direct_lighting(h.color.rgb, h.normal, hp);
            let emit = h.emissive * light.emissive_strength;
            acc_rad = acc_rad + surf + emit;
        } else {
            acc_rad = acc_rad + bounce_sky();
        }
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

// Previous-frame view-projection (clip_from_world), uploaded each frame, for motion-vector reprojection.
struct DlssCamera {
    prev_clip_from_world: mat4x4<f32>,
    clip_from_world: mat4x4<f32>,
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

        // True reverse-Z clip depth of the hit (matches Bevy's infinite-far reverse-Z prepass): project the
        // world hit point with clip_from_world and take z/w. A hit always has finite positive depth.
        let clip = dlss_cam.clip_from_world * vec4<f32>(p, 1.0);
        let depth = clip.z / clip.w;
        textureStore(out_dlss_depth, px, vec4<f32>(depth, 0.0, 0.0, 0.0));

        // Screen-space motion: where THIS hit point was in the previous frame's screen, minus where it is now.
        // DLSS-RR expects motion in pixels (LowResolutionMotionVectors + motion_vector_scale = -render_res in
        // node.rs scales our NDC delta by -render_res → pixels). We output the NDC-space delta (cur−prev) here;
        // the node's motion_vector_scale converts it. For a static scene + still camera this is ~0.
        let prev_clip = dlss_cam.prev_clip_from_world * vec4<f32>(p, 1.0);
        let prev_ndc = prev_clip.xy / prev_clip.w;
        let cur_ndc = clip.xy / clip.w;
        // NDC delta in [-1,1] space → UV-space delta (×0.5). DLSS node multiplies by -render_res.
        let motion = (prev_ndc - cur_ndc) * vec2<f32>(0.5, -0.5);
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
