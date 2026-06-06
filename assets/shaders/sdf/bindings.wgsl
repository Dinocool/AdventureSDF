#define_import_path sdf::bindings

// Shared bindings, struct layouts, and constants for the SDF raymarch shader.
// Single-sourced here so every module references the same `@group@binding` slots
// (naga_oil requires each slot be declared exactly once across the import graph).

// --- Structs ---

struct SdfCameraUniform {
    inv_view_proj: mat4x4<f32>,
    clip_from_world: mat4x4<f32>,
    // LAST frame's clip_from_world, for SSR: project a reflected world point into the previous
    // frame's screen to sample its already-shaded colour (sdf_raymarch SSR path).
    prev_clip_from_world: mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen_params: vec4<f32>,  // xy = screen_size; z = overlap_depth (u32); w = shadow LOD floor (u32)
    grid_origin: vec4<f32>,
    grid_dims: vec4<f32>,
    debug_params: vec4<f32>,   // x = max_steps, y = max_dist, z = sdf_eps, w = recenter_snap_chunks
    march_params: vec4<f32>,   // x = pixel_cone (world radius/unit-dist/pixel), y = reserved (was cubic_band), z = over_relax, w = lod_blend_band
    lod_params: vec4<f32>,     // x = lod_count, y = ring_bricks, z = base voxel_size, w = cell_stride
    sun_dir: vec4<f32>,        // xyz = direction toward the key light; w = shadow light cap (u32)
    sun_color: vec4<f32>,      // rgb = physical sun radiance (illuminance, lux); w = camera exposure scalar (exp2(-ev100)/1.2)
};

// One material row, indexed by global material id. Mirrors `GpuSdfMaterial`
// (render.rs): base colour + seam softness + per-map texture-array layer indices
// (0xffffffff = no texture for that map) + scalar metallic/roughness fallbacks +
// emissive. 80 bytes.
struct SdfMaterial {
    base_color: vec4<f32>,
    blend_softness: f32,   // world-units colour-feather width at a seam
    tex_diffuse: u32,
    tex_normal: u32,
    tex_mra: u32,
    tex_height: u32,
    tex_edge: u32,
    // Used when tex_mra is absent (0xffffffff): lets a material be a plain metal/dielectric
    // without an MRA texture. Range 0..1 each.
    metallic: f32,
    roughness: f32,
    // Parallax-occlusion relief depth (UV units) for this material's height map. 0 = flat.
    parallax_scale: f32,
    // Three SEPARATE u32 pads — NOT vec3<u32>, which has 16-byte alignment in WGSL — aligning
    // `emissive` to its 16-byte boundary (offset 64). Names avoid trailing digits (naga_oil
    // writeback rejects `pad0` etc).
    pad_a: u32,
    pad_b: u32,
    pad_c: u32,
    // Emissive radiance, linear RGB in xyz (intensity premultiplied CPU-side); w spare.
    emissive: vec4<f32>,
};

// Per-brick lookup. `key_hi`/`key_lo` are the absolute 64-bit brick key (lod + biased
// world-lattice brick index; see SdfGridConfig::abs_brick_key), independent of camera
// position so the CPU table and shader agree. `pal01`/`pal23` pack the brick's 4-entry
// material palette: pal01 = id0 | id1<<16, pal23 = id2 | id3<<16. Slot k of the
// per-voxel distance atlas is keyed to palette entry k; PALETTE_EMPTY (0xffff) unused.

// One entry in the per-LOD chunk directory (indexed by `dir_index`, tag-compared). `key_*` is the
// camera-independent chunk key (see chunk.rs). TWO occupancy masks: `occ_*` = BAKED (a tile is
// resident — for SAMPLING, honours shells); `cons_occ_*` = CONSERVATIVE (the geometry BVH overlaps
// the brick — for the empty-space DDA, full ring, coarse-empty⇒fine-empty by construction).
// `tile_run_base` indexes `chunk_tile_buf` where this chunk's popcount(occ) baked bricks live.
// Field order MUST match chunk::ChunkLookup + encode_lookup (chunk_tables.rs).
struct ChunkLookup {
    key_hi: u32,
    key_lo: u32,
    occ_lo: u32,
    occ_hi: u32,
    cons_occ_lo: u32,
    cons_occ_hi: u32,
    tile_run_base: u32,
    probe_base: u32, // DDGI probe-slot base (u32::MAX = no probes here → fall back to a coarser LOD)
};

// One resident brick's record in the packed chunk tile run: distance tile origin, material tile origin
// (or MAT_ATLAS_NONE for a single-material brick), palette, + DDGI probe slot. MUST match chunk::BrickTile
// field order (encode_tile): atlas_base, mat_atlas_base, pal01, pal23, probe_slot (20 bytes std430).
struct BrickTile {
    atlas_base: u32,      // distance tile origin: col_px | row_px<<16
    mat_atlas_base: u32,  // material tile origin, or MAT_ATLAS_NONE (single-material brick)
    pal_lo: u32,          // palette ids 0,1 (id0 | id1<<16)
    pal_hi: u32,          // palette ids 2,3 (id2 | id3<<16)
    probe_slot: u32,      // DDGI compact finest-resident probe slot (0xffffffff = none)
};

// NOTE: DDGI probe slots are no longer derived from the atlas tile index. They are now compact over the
// FINEST-resident chunk set — `chunk.probe_base + local_brick_index(coord)` (see `sdf::probe`) — so the
// probe buffer scales with the clipmap window, not the all-LOD atlas tile union. `grid_dims.w` (tiles-
// per-row) is consequently unused by the probe path (kept in the uniform for layout stability).

// --- Bindings ---
//
// Empty-space skipping is driven by the conservative SDF field itself (see the bake in
// atlas.rs and the march in sdf_raymarch.wgsl), so there is NO GPU BVH binding — the BVH
// lives CPU-side only, as the bake cull.

@group(0) @binding(0) var<uniform> camera: SdfCameraUniform;
// Distance + material atlases as PAGED pools (bindless): each is a runtime-sized array of
// fixed-height page textures, so growth allocates ONE new page (no realloc + full-copy of the
// whole atlas, which spiked VRAM ~2× and cost O(N²) during a fill). A brick's global tile row
// splits into (page = row / ATLAS_PAGE_HEIGHT_PX, local row) — see `brick::voxel_loc`. The page
// index is data-dependent (non-uniform across fragments), so the pipeline needs the sampled-array
// non-uniform-indexing feature. Sampled via `textureLoad` (manual trilinear), so no sampler.
// SIZED binding_array (not the unsized `binding_array<T>` — naga_oil's composable-module name
// mangler routes through the GLSL backend, which can't write an unsized binding array and panics;
// Bevy's own bindless bindings are likewise sized). `ATLAS_MAX_PAGES` (= 64) is the COMPILE-TIME
// max page count; fewer real pages are bound and the rest are dummy 1×1 fills. MUST match the Rust
// `atlas_upload::ATLAS_MAX_PAGES`.
@group(1) @binding(0) var atlas_pages: binding_array<texture_2d<f32>, 64>;   // R16Snorm distance pages
@group(1) @binding(1) var atlas_sampler: sampler;
@group(1) @binding(2) var<storage, read> chunk_buf: array<ChunkLookup>;  // sorted, binary-searched
@group(1) @binding(3) var mat_pages: binding_array<texture_2d<f32>, 64>;     // Rgba16Snorm: 4 palette-slot dist pages
@group(1) @binding(4) var<storage, read> materials: array<SdfMaterial>;  // material table, by global id
// PBR texture arrays + their filtering sampler. Each is a texture_2d_array indexed
// by a material's tex layer; sampled triplanar in `material`/`pbr`.
@group(1) @binding(5) var pbr_sampler: sampler;
@group(1) @binding(6) var tex_diffuse: texture_2d_array<f32>;
@group(1) @binding(7) var tex_normal: texture_2d_array<f32>;
@group(1) @binding(8) var tex_mra: texture_2d_array<f32>;
@group(1) @binding(9) var tex_height: texture_2d_array<f32>;
@group(1) @binding(10) var tex_edge: texture_2d_array<f32>;
@group(1) @binding(11) var<storage, read> chunk_tile_buf: array<BrickTile>;  // packed per-chunk brick runs
// Per-voxel gradient (outward unit normal) pages, Rgba8Snorm. PAGED like the distance atlas and
// indexed by the SAME tile origin (dense — one tile per brick). Only populated when the gradient
// feature is enabled (SDF_GRAD_NORMALS); a dummy 1×1 fills every slot otherwise.
// SIZED binding_array (see the atlas_pages note above). MUST match `atlas_upload::ATLAS_MAX_PAGES`.
@group(1) @binding(12) var grad_pages: binding_array<texture_2d<f32>, 64>;

// --- Shared constants ---

const PALETTE_EMPTY: u32 = 0xffffu;
// `mat_atlas_base` sentinel for a single-material brick (owns no material tile). Mirrors
// `chunk::MAT_ATLAS_NONE`. The reader short-circuits on `palette[1]==EMPTY` before sampling, so
// this is never used as a real origin.
const MAT_ATLAS_NONE: u32 = 0xffffffffu;
const TEXTURE_WORLD_SCALE: f32 = 0.5;  // world units per texture tile = 2.0
const PI: f32 = 3.14159265359;
// Per-LOD distance-field clamp band in VOXELS. The geometry distance atlas stores
// `d / (DIST_BAND_VOXELS · voxel_size_at(lod))` as snorm; `sample_brick_sdf` multiplies back.
// MUST match atlas::DIST_BAND_VOXELS on the CPU. Coarse LODs get a large world band → big
// sphere-trace steps far from the surface (the per-LOD voxel-unit clamp).
const DIST_BAND_VOXELS: f32 = 4.0;

// Height (in PIXELS) of one atlas page texture in the paged pool. A brick tile is `edge` (8) px
// tall, so this MUST be a multiple of 8 (a tile never straddles two pages). MUST match the Rust
// `atlas_upload::ATLAS_PAGE_HEIGHT_PX` exactly (the bake's tile→page routing and this sampler agree
// on it). 2048 px = 256 tile-rows = 256·ATLAS_TILES_PER_ROW bricks per page.
const ATLAS_PAGE_HEIGHT_PX: i32 = 2048;

// --- Uniform accessors ---

fn max_steps() -> u32 { return u32(camera.debug_params.x); }
fn max_dist() -> f32 { return camera.debug_params.y; }
fn sdf_eps() -> f32 { return camera.debug_params.z; }
// Pixel cone half-width per unit ray distance (world radius a pixel covers at t=1).
// The march terminates when the conservative field is below `pixel_cone * t` — i.e. the
// surface is within a pixel — so far geometry resolves at coarse LOD instead of marching
// down to LOD 0 (the vast-distance efficiency win).
fn pixel_cone() -> f32 { return camera.march_params.x; }
// march_params.y = soft-shadow penumbra hardness `k` (the IQ `min(k*d/t)` factor in
// sdf::shadows). Lower = softer/wider penumbra, which blurs coarse-LOD brick faceting and
// softens the penumbra→umbra edge. Editor "Shadow Softness" slider. 0 = unset (shader falls
// back to the default in `surface_shadow`).
fn shadow_softness() -> f32 { return camera.march_params.y; }
// How many point lights (brightest-first, of those reaching a surface) cast an SDF shadow per
// pixel — editor "Shadow lights" slider, packed in the unused `sun_dir.w`. 0 = no point shadows.
fn shadow_light_cap() -> u32 { return u32(camera.sun_dir.w); }
// Sphere-trace over-relaxation factor (Keinert 2014): the march steps `over_relax * d`
// instead of `d`, with a safe fallback when consecutive unbounding spheres separate.
// 1.0 = plain sphere tracing; (1,2) accelerates convergence on grazing rays.
fn over_relax() -> f32 { return camera.march_params.z; }
// LOD cross-fade band width, as a fraction of each ring's half-extent. In the outer
// `lod_blend_band` shell the march fades the serving LOD toward its coarser neighbour so
// the surface morphs smoothly across the ring boundary. 0 = hard LOD seams (disabled).
fn lod_blend_band() -> f32 { return camera.march_params.w; }
// `recenter_snap_chunks` (hysteresis snap of the resident ring origin, in whole chunks).
// The LOD cross-fade keys off the chunk-SNAPPED ring centre, so the shader recomputes it
// from camera_pos + this (mirrors bake_scheduler::ring_chunk_origin). >= 1.
fn recenter_snap() -> i32 { return max(i32(camera.debug_params.w), 1); }
// How many COARSER LODs each region keeps resident beyond its native LOD ({native..native+overlap}).
// Drives the inner-hole size in `in_ring_chunk` — MUST mirror CPU `SdfGridConfig::overlap_depth` /
// `inner_hole_half_chunks` exactly, or the empty-space DDA skip and the resident set diverge (gaps).
fn overlap_depth() -> u32 { return u32(camera.screen_params.z); }
// Minimum LOD the shadow march samples in-brick (editor "Shadow detail" slider, in screen_params.w).
// 0 = finest (sharpest, slowest); higher = coarser/blobbier shadows but far fewer march steps.
fn shadow_lod_bias() -> u32 { return u32(camera.screen_params.w); }

// --- LOD clipmap / chunk accessors ---

// Bricks per axis in a chunk. Must match chunk::CHUNK_BRICKS on the CPU.
const CHUNK_BRICKS: i32 = 4;

fn lod_count() -> u32 { return u32(camera.lod_params.x); }
// Ring window size in BRICKS per axis (mirrors SdfGridConfig::ring_bricks). The chunk-DDA
// empty-space skip uses ring_bricks/CHUNK_BRICKS chunks per axis.
fn ring_bricks() -> i32 { return i32(camera.lod_params.y); }
// Brick spatial stride in voxels (cell_stride; same at every LOD — only the world
// size of a voxel changes). Mirrors SdfGridConfig::cell_stride.
fn cell_stride() -> i32 { return i32(camera.lod_params.w); }
// Voxel size (world units) at LOD `lod`: base · 2^lod.
fn voxel_size_at(lod: u32) -> f32 { return camera.lod_params.z * exp2(f32(lod)); }
// World edge length of one brick at LOD `lod`.
fn brick_world_at(lod: u32) -> f32 { return f32(cell_stride()) * voxel_size_at(lod); }

// World-space CENTRE of the resident clipmap ring window at LOD `lod` for a camera at
// `cam`. Mirrors `bake_scheduler::ring_chunk_origin` (the async/default bake path): the
// window is centred on the camera's chunk SNAPPED to the `recenter_snap` lattice
// (hysteresis), so the cross-fade must key off this snapped centre, not raw `cam`, or the
// boundary is off by up to a snap cell and the LOD pop persists. The window spans whole
// chunks, so its centre lands on the chunk-index `cam_chunk_snapped` ⇒ that index times the
// chunk world size. `floor_div` (forward-declared below) floors toward -inf so the lattice
// is continuous across the world origin, matching the CPU's `div_euclid`.
fn ring_center_lod(cam: vec3<f32>, lod: u32) -> vec3<f32> {
    let s = cell_stride();
    let vs = voxel_size_at(lod);
    let snap = recenter_snap();
    let vox = vec3<i32>(floor(cam / vs));                                  // voxel coord on this LOD
    let brick = vec3<i32>(floor_div(vox.x, s), floor_div(vox.y, s), floor_div(vox.z, s));
    let chunk = vec3<i32>(
        floor_div(brick.x, CHUNK_BRICKS),
        floor_div(brick.y, CHUNK_BRICKS),
        floor_div(brick.z, CHUNK_BRICKS),
    );
    let snapped = vec3<i32>(
        floor_div(chunk.x, snap) * snap,
        floor_div(chunk.y, snap) * snap,
        floor_div(chunk.z, snap) * snap,
    );
    let chunk_world = f32(CHUNK_BRICKS) * brick_world_at(lod);
    return vec3<f32>(snapped) * chunk_world;
}

// Forward exit distance of a ray through the resident clipmap's outer AABB — the bound that
// makes a MISS ray terminate in O(1) instead of crawling brick-by-brick through the void.
// Shared by the primary/shadow march (`sdf::march`) AND the cone prepass so both stop a sky ray
// at the volume edge.
//
// The coarsest ring is the LARGEST clipmap shell and is centred on the camera, so it contains
// every finer ring and therefore ALL resident geometry (geometry exists only inside resident
// chunks, which are all inside some ring). The box is convex and the march origin (the camera,
// or a surface point, both inside the volume) sits inside it, so the ray crosses the far face
// exactly once and can never re-enter — the instant `t` passes this distance the ray is sky.
//
// Standard slab test, but we only need the FAR intersection (`t_far`): per axis the two slab
// planes give `t1`,`t2`; the box exit is the nearest of the three per-axis maxima. `1.0/dir` is
// ±inf on an axis-aligned ray, which the min/max collapses to the finite axes (correct slab
// behaviour). The ring centre is keyed off `camera.camera_pos` (NOT `origin`) because the ring
// is camera-centred — a shadow ray starting at a surface point shares the same box.
fn clipmap_exit_t(origin: vec3<f32>, dir: vec3<f32>) -> f32 {
    let cl = lod_count() - 1u;
    let center = ring_center_lod(camera.camera_pos.xyz, cl);
    let half = 0.5 * f32(ring_bricks()) * brick_world_at(cl);   // per-axis half-extent
    let inv = 1.0 / dir;
    let t_hi = (center + vec3<f32>(half) - origin) * inv;
    let t_lo = (center - vec3<f32>(half) - origin) * inv;
    let t_far = max(t_hi, t_lo);                                // per-axis exit plane
    return min(min(t_far.x, t_far.y), t_far.z);
}

// Floored division of `a` by `b` (b > 0), rounding toward negative infinity.
//
// Avoids BOTH broken ops observed on this hardware (verified in tests/sdf_gpu_rig.rs):
//   1. Signed `%` on a runtime negative returns the UNSIGNED result (`-109 % 7` -> 0
//      instead of -4), so it can't be used to build a remainder.
//   2. Float `/` has a 1-ULP error, so `i32(floor(f32(a)/f32(b)))` mis-floors exact
//      multiples: `-49/7` computes as -7.0000001, `floor` -> -8 instead of -7.
// Integer truncating `/` IS correct on this hardware, so: take the truncated quotient,
// reconstruct the remainder by multiply/subtract (no `%`), and step the quotient down by
// one when the remainder is negative (b > 0), converting truncation to floor.
fn floor_div(a: i32, b: i32) -> i32 {
    let q = a / b;              // truncated toward zero — verified correct on GPU
    let r = a - q * b;          // remainder without the `%` operator
    return select(q, q - 1, r < 0);  // floor: step down when remainder is negative (b > 0)
}

// Euclidean remainder of `a` by `b` (b > 0): always in [0, b). Built from `floor_div`
// (multiply/subtract), so it never touches the signed `%` operator.
fn euclid_mod(a: i32, b: i32) -> i32 {
    return a - floor_div(a, b) * b;
}

// Absolute 64-bit CHUNK key for the chunk containing brick `coord` at `lod` — mirrors
// chunk::chunk_gpu_key + chunk_of. Independent of camera so the CPU table and this agree.
// vec2(key_hi, key_lo): key_hi=(lod<<16)|cx, key_lo=(cy<<16)|cz, each chunk index biased
// by 2^15 into a 16-bit field.
fn abs_chunk_key(coord: vec3<i32>, lod: u32) -> vec2<u32> {
    let s = cell_stride();
    let bias = 32768;
    // brick index → chunk index, Euclidean so negatives map continuously.
    let cx = u32((floor_div(floor_div(coord.x, s), CHUNK_BRICKS) + bias) & 0xffff);
    let cy = u32((floor_div(floor_div(coord.y, s), CHUNK_BRICKS) + bias) & 0xffff);
    let cz = u32((floor_div(floor_div(coord.z, s), CHUNK_BRICKS) + bias) & 0xffff);
    return vec2<u32>((lod << 16u) | cx, (cy << 16u) | cz);
}

// Local brick slot (0..63) of brick `coord` within its chunk — mirrors chunk_of.
fn local_brick_index(coord: vec3<i32>) -> u32 {
    let s = cell_stride();
    let c = CHUNK_BRICKS;
    let lx = euclid_mod(floor_div(coord.x, s), c);
    let ly = euclid_mod(floor_div(coord.y, s), c);
    let lz = euclid_mod(floor_div(coord.z, s), c);
    return u32(lz * c * c + ly * c + lx);
}

// Ring chunks per axis: R = ring_bricks / CHUNK_BRICKS. Mirrors `LiveChunkTables::r`.
fn ring_chunks() -> i32 {
    return ring_bricks() / CHUNK_BRICKS;
}

// The chunk coord (on the LOD's chunk lattice) containing brick `coord` — the chunk-coord step of
// chunk::chunk_of (brick index → chunk index, Euclidean so negatives map continuously).
fn chunk_coord_of(coord: vec3<i32>) -> vec3<i32> {
    let s = cell_stride();
    return vec3<i32>(
        floor_div(floor_div(coord.x, s), CHUNK_BRICKS),
        floor_div(floor_div(coord.y, s), CHUNK_BRICKS),
        floor_div(floor_div(coord.z, s), CHUNK_BRICKS),
    );
}

// Physical directory slot of the chunk containing brick `coord` at `lod`, into the dense per-LOD
// toroidal directory `chunk_buf`: lod*R³ + flatten(euclid_mod(chunk_coord, R)). EXACT mirror of
// chunk::dir_index — the GPU-rig parity test guards against drift. `euclid_mod` (never raw `%`) so
// negative coords + non-power-of-two R both index correctly.
fn dir_index(coord: vec3<i32>, lod: u32) -> u32 {
    let r = ring_chunks();
    let cc = chunk_coord_of(coord);
    let mx = euclid_mod(cc.x, r);
    let my = euclid_mod(cc.y, r);
    let mz = euclid_mod(cc.z, r);
    return lod * u32(r * r * r) + u32(mz * r * r + my * r + mx);
}
