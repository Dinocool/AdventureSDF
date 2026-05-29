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
    debug_params: vec4<f32>,   // x = max_steps, y = max_dist, z = sdf_eps, w = bvh_node_count
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

// Per-brick lookup. `pal01`/`pal23` pack the brick's 4-entry material palette:
// pal01 = id0 | id1<<16, pal23 = id2 | id3<<16. Slot k of the per-voxel distance
// atlas is keyed to palette entry k; PALETTE_EMPTY (0xffff) marks an unused slot.
struct BrickLookup {
    brick_id: u32,
    atlas_base: u32,  // tile origin in the 2D-tiled atlas: col_px | row_px<<16
    pal_lo: u32,      // palette ids 0,1 packed (id0 | id1<<16)
    pal_hi: u32,      // palette ids 2,3 packed (id2 | id3<<16)
};

// BVH node: 32 bytes (two vec3<f32> + u32 rows). `count_or_right`'s high bit
// (0x80000000) marks an internal node — then the field is the right-child index
// and `left_or_first` is the left child. Clear high bit => leaf, where the field
// is the edit count and `left_or_first` the first edit index. The shader only
// needs the AABBs for empty-space skipping, so it ignores edit indices entirely.
struct BvhNode {
    aabb_min: vec3<f32>,
    left_or_first: u32,
    aabb_max: vec3<f32>,
    count_or_right: u32,
};

// --- Bindings ---

@group(0) @binding(0) var<uniform> camera: SdfCameraUniform;
@group(1) @binding(0) var atlas_tex: texture_2d<f32>;       // R16Snorm distance field
@group(1) @binding(1) var atlas_sampler: sampler;
@group(1) @binding(2) var<storage, read> lookup_buf: array<BrickLookup>;
@group(1) @binding(3) var mat_tex: texture_2d<f32>;         // Rgba16Snorm: 4 palette-slot distances
@group(1) @binding(4) var<storage, read> bvh_buf: array<BvhNode>;  // edit-AABB BVH (empty-space skip)
@group(1) @binding(5) var<storage, read> materials: array<SdfMaterial>;  // material table, by global id
// PBR texture arrays + their filtering sampler. Each is a texture_2d_array indexed
// by a material's tex layer; sampled triplanar in `material`/`pbr`.
@group(1) @binding(6) var pbr_sampler: sampler;
@group(1) @binding(7) var tex_diffuse: texture_2d_array<f32>;
@group(1) @binding(8) var tex_normal: texture_2d_array<f32>;
@group(1) @binding(9) var tex_mra: texture_2d_array<f32>;
@group(1) @binding(10) var tex_height: texture_2d_array<f32>;
@group(1) @binding(11) var tex_edge: texture_2d_array<f32>;

// --- Shared constants ---

const PALETTE_EMPTY: u32 = 0xffffu;
const BVH_INTERNAL_FLAG: u32 = 0x80000000u;
const TEXTURE_WORLD_SCALE: f32 = 0.5;  // world units per texture tile = 2.0
const PI: f32 = 3.14159265359;

// --- Uniform accessors ---

fn num_bvh_nodes() -> u32 { return u32(camera.debug_params.w); }
fn max_steps() -> u32 { return u32(camera.debug_params.x); }
fn max_dist() -> f32 { return camera.debug_params.y; }
fn sdf_eps() -> f32 { return camera.debug_params.z; }
