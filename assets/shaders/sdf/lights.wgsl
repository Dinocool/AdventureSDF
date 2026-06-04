#define_import_path sdf::lights

#import sdf::brdf::frostbite_brdf

// Scene point lights for the SDF G-buffer pass, with a sparse world-wide light grid.
//
// Group 3 (FRAGMENT|COMPUTE so the future DDGI probe-trace compute pass can bind the same data):
//   binding 0 — the `PointLightGpu` array (mirror of `GpuPointLight`, render/mod.rs).
//   binding 1 — the SPARSE light-grid directory: occupied cells, each `{key, base, count}`,
//               SORTED ascending by 64-bit key so it can be binary-searched.
//   binding 2 — the flat per-cell light-index runs.
//
// `lights_in_cell(world_pos)` binary-searches the directory for the cell containing `world_pos`
// and returns the (base, count) of the lights binned there, so a pixel/probe only iterates the
// handful of lights near it. World-wide (no camera window) — depends only on `world_pos` + the
// group-3 buffers, so it's reusable verbatim from a compute pass. The CPU build (`light_grid.rs`)
// and this lookup must agree on the cell math + key packing; `wgsl_light_grid_constants_match_rust`
// pins the constants.

// Mirrors `GpuPointLight` (render/mod.rs): two vec4s, 32 bytes, 16-byte aligned (no padding).
struct PointLightGpu {
    pos_range: vec4<f32>,      // xyz = world pos, w = range (falloff cutoff)
    color_radius: vec4<f32>,   // rgb = physical radiance (candela-scaled linear), w = source radius
};

// Mirrors `GpuLightCell` (light_grid.rs): a 64-bit cell key + (base, count) into `light_indices`.
struct LightCell {
    key_hi: u32,
    key_lo: u32,
    base: u32,
    count: u32,
};

@group(3) @binding(0) var<storage, read> point_lights: array<PointLightGpu>;
@group(3) @binding(1) var<storage, read> light_cells: array<LightCell>;
@group(3) @binding(2) var<storage, read> light_indices: array<u32>;

// MUST match light_grid::LIGHT_CELL_SIZE (parity test). A mismatch makes the GPU look up a
// different cell than the CPU binned into → lights flicker/vanish.
const LIGHT_CELL_SIZE: f32 = 8.0;

// (The shadow light cap — how many lights cast SDF shadows per pixel — is now a live uniform,
// `sdf::bindings::shadow_light_cap()`, driven by the editor "Shadow lights" slider.)

// Smooth windowed inverse-square falloff (Frostbite / Lagarde 2014, also Bevy's point-light
// attenuation): physically `1/d²`, multiplied by a `(1 - (d/range)⁴)²` window so the contribution
// reaches exactly zero at `range`. `d2` is the squared distance; `radius` is the light's source
// size — the `1/d²` term is clamped at the sphere SURFACE (`max(d2, radius²)`) so a surface at/inside
// the light volume gets a bounded radiance instead of the point-light singularity blowing up.
fn point_attenuation(d2: f32, range: f32, radius: f32) -> f32 {
    let factor = d2 / max(range * range, 1e-6);   // (d/range)²
    let window = clamp(1.0 - factor * factor, 0.0, 1.0);
    let inv_sq = 1.0 / max(d2, max(radius * radius, 1e-4));
    return window * window * inv_sq;
}

// Direct contribution of ONE light through the Frostbite BRDF, shadowed. Shared by EVERY light —
// the directional sun and each point light shade identically through this. `irradiance` is the
// light's radiance reaching the surface: a point light passes `radiance × point_attenuation` (1/d²
// falloff); a directional light passes its illuminance with no falloff. `vis` is the shadow term
// (1 = lit, 0 = occluded). `frostbite_brdf` already includes the N·L cosine.
fn direct_light(
    view: vec3<f32>,
    n: vec3<f32>,
    l: vec3<f32>,
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
    f0: vec3<f32>,
    irradiance: vec3<f32>,
    vis: f32,
) -> vec3<f32> {
    return frostbite_brdf(view, n, l, albedo, roughness, metallic, f0) * irradiance * vis;
}

// Pack a world cell coord into the order-preserving 64-bit light key — byte-matches
// light_grid::light_gpu_key (biased 16-bit axis fields; sorting (key_hi,key_lo) orders by x,y,z).
fn light_cell_key(cell: vec3<i32>) -> vec2<u32> {
    let bias = 32768;                             // LIGHT_KEY_BIAS — pinned by the parity test
    let cx = u32((cell.x + bias) & 0xffff);
    let cy = u32((cell.y + bias) & 0xffff);
    let cz = u32((cell.z + bias) & 0xffff);
    return vec2<u32>(cx, (cy << 16u) | cz);       // key_hi = cx, key_lo = cy<<16 | cz
}

// (base, count) of the lights in the world cell containing `world_pos`. Binary-searches the
// key-sorted directory (lower bound); miss (incl. the empty-scene sentinel) → (0, 0). Same float
// floor as the CPU (`light_grid::cell_of`).
fn lights_in_cell(world_pos: vec3<f32>) -> vec2<u32> {
    let cell = vec3<i32>(floor(world_pos / LIGHT_CELL_SIZE));
    let want = light_cell_key(cell);
    let n = arrayLength(&light_cells);
    var lo = 0u;
    var hi = n;
    while (lo < hi) {
        let mid = lo + (hi - lo) / 2u;
        let e = light_cells[mid];
        // Compare (key_hi, key_lo) as one 64-bit magnitude.
        if (e.key_hi < want.x || (e.key_hi == want.x && e.key_lo < want.y)) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo < n) {
        let e = light_cells[lo];
        if (e.key_hi == want.x && e.key_lo == want.y) {
            return vec2<u32>(e.base, e.count);
        }
    }
    return vec2<u32>(0u, 0u);
}
