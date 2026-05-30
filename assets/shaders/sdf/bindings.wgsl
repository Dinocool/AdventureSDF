#define_import_path sdf::bindings

// Shared bindings, struct layouts, and constants for the SDF raymarch shader.
// Single-sourced here so every module references the same `@group@binding` slots
// (naga_oil requires each slot be declared exactly once across the import graph).

// --- Structs ---

struct SdfCameraUniform {
    inv_view_proj: mat4x4<f32>,
    clip_from_world: mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen_params: vec4<f32>,
    grid_origin: vec4<f32>,
    grid_dims: vec4<f32>,
    debug_params: vec4<f32>,   // x = max_steps, y = max_dist, z = sdf_eps, w = recenter_snap_chunks
    march_params: vec4<f32>,   // x = pixel_cone (world radius/unit-dist/pixel), y = cubic_band, z = over_relax, w unused
    lod_params: vec4<f32>,     // x = lod_count, y = ring_bricks, z = base voxel_size, w = cell_stride
};

// One material row, indexed by global material id. Mirrors `GpuSdfMaterial`
// (render.rs): base colour + seam softness + per-map texture-array layer indices
// (0xffffffff = no texture for that map). 48 bytes.
struct SdfMaterial {
    base_color: vec4<f32>,
    blend_softness: f32,   // world-units colour-feather width at a seam
    tex_diffuse: u32,
    tex_normal: u32,
    tex_mra: u32,
    tex_height: u32,
    tex_edge: u32,
    pad: vec2<u32>,
};

// Per-brick lookup. `key_hi`/`key_lo` are the absolute 64-bit brick key (lod + biased
// world-lattice brick index; see SdfGridConfig::abs_brick_key), independent of camera
// position so the CPU table and shader agree. `pal01`/`pal23` pack the brick's 4-entry
// material palette: pal01 = id0 | id1<<16, pal23 = id2 | id3<<16. Slot k of the
// per-voxel distance atlas is keyed to palette entry k; PALETTE_EMPTY (0xffff) unused.

// One entry in the sorted chunk lookup table (binary-searched by absolute key). `key_*`
// is the camera-independent chunk key (see chunk.rs); `occ_*` is the 64-bit occupancy
// mask (bit i ⇒ local brick i resident); `tile_run_base` indexes `chunk_tile_buf` where
// this chunk's popcount(occ) resident bricks live in ascending local order.
struct ChunkLookup {
    key_hi: u32,
    key_lo: u32,
    occ_lo: u32,
    occ_hi: u32,
    tile_run_base: u32,
};

// One resident brick's record in the packed chunk tile run: atlas tile origin + palette.
struct BrickTile {
    atlas_base: u32,  // col_px | row_px<<16
    pal_lo: u32,      // palette ids 0,1 (id0 | id1<<16)
    pal_hi: u32,      // palette ids 2,3 (id2 | id3<<16)
};

// --- Bindings ---
//
// Empty-space skipping is driven by the conservative SDF field itself (see the bake in
// atlas.rs and the march in sdf_raymarch.wgsl), so there is NO GPU BVH binding — the BVH
// lives CPU-side only, as the bake cull.

@group(0) @binding(0) var<uniform> camera: SdfCameraUniform;
@group(1) @binding(0) var atlas_tex: texture_2d<f32>;       // R16Snorm distance field
@group(1) @binding(1) var atlas_sampler: sampler;
@group(1) @binding(2) var<storage, read> chunk_buf: array<ChunkLookup>;  // sorted, binary-searched
@group(1) @binding(3) var mat_tex: texture_2d<f32>;         // Rgba16Snorm: 4 palette-slot distances
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

// --- Shared constants ---

const PALETTE_EMPTY: u32 = 0xffffu;
const TEXTURE_WORLD_SCALE: f32 = 0.5;  // world units per texture tile = 2.0
const PI: f32 = 3.14159265359;

// --- Uniform accessors ---

fn max_steps() -> u32 { return u32(camera.debug_params.x); }
fn max_dist() -> f32 { return camera.debug_params.y; }
fn sdf_eps() -> f32 { return camera.debug_params.z; }
// Pixel cone half-width per unit ray distance (world radius a pixel covers at t=1).
// The march terminates when the conservative field is below `pixel_cone * t` — i.e. the
// surface is within a pixel — so far geometry resolves at coarse LOD instead of marching
// down to LOD 0 (the vast-distance efficiency win).
fn pixel_cone() -> f32 { return camera.march_params.x; }
// Distance band (world units) within which a LOD-0 sample switches to the exact analytic
// cubic for a crisp near silhouette. Outside it (or at coarse LOD) the march sphere-traces
// the conservative field.
fn cubic_band() -> f32 { return camera.march_params.y; }
// Sphere-trace over-relaxation factor (Keinert 2014): the march steps `over_relax * d`
// instead of `d`, with a safe fallback when consecutive unbounding spheres separate.
// 1.0 = plain sphere tracing; (1,2) accelerates convergence on grazing rays.
fn over_relax() -> f32 { return camera.march_params.z; }

// --- LOD clipmap / chunk accessors ---

// Bricks per axis in a chunk. Must match chunk::CHUNK_BRICKS on the CPU.
const CHUNK_BRICKS: i32 = 4;

fn lod_count() -> u32 { return u32(camera.lod_params.x); }
// Ring window size in BRICKS per axis (mirrors SdfGridConfig::ring_bricks). The chunk-DDA
// empty-space skip uses ring_bricks/CHUNK_BRICKS chunks per axis.
fn ring_bricks() -> i32 { return i32(camera.lod_params.y); }
// Coarse chunk lattice the per-LOD ring origin snaps to (hysteresis). Mirrors
// SdfGridConfig::recenter_snap_chunks; the chunk-DDA in-ring test must use the SAME snap as
// bake_scheduler::ring_chunk_origin or it will mis-classify chunks near the ring edge.
fn recenter_snap_chunks() -> i32 { return i32(camera.debug_params.w); }
// Brick spatial stride in voxels (cell_stride; same at every LOD — only the world
// size of a voxel changes). Mirrors SdfGridConfig::cell_stride.
fn cell_stride() -> i32 { return i32(camera.lod_params.w); }
// Voxel size (world units) at LOD `lod`: base · 2^lod.
fn voxel_size_at(lod: u32) -> f32 { return camera.lod_params.z * exp2(f32(lod)); }
// World edge length of one brick at LOD `lod`.
fn brick_world_at(lod: u32) -> f32 { return f32(cell_stride()) * voxel_size_at(lod); }

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
