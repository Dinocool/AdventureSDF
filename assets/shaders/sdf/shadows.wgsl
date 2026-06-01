#define_import_path sdf::shadows

// SDF soft shadows. Native to the raymarcher: march a secondary ray from the surface
// toward the light through the same conservative field (`sample_sdf_world`), tracking
// the IQ penumbra estimate (the closest the ray passes to an occluder, scaled by how
// far along it that miss happened). No shadow maps. Returns 1 = fully lit, 0 = occluded.

#import sdf::bindings::{voxel_size_at}
#import sdf::brick::{sample_sdf_world_cached, calc_normal, ChunkCache, new_chunk_cache}

// The baked distance field is snorm-clamped to ±SNORM_CLAMP_DIST world units (atlas.rs).
// A sample at the ceiling means "nearest surface is ≥ this far" — saturated, carrying no
// occluder info — so the shadow ray treats it as clear. MUST track atlas::SNORM_CLAMP_DIST.
const SHADOW_FIELD_CEIL: f32 = 1.0;

// Inigo Quilez soft shadow: along the ray to the light, `res = min(res, k*d/t)`. A miss
// that passes close to an occluder (small d) early in the march (small t) softens most.
// Early-out on a real hit (d < eps → hard occlusion → 0).
//
// CRITICAL for this voxel field: the stored distance saturates at SHADOW_FIELD_CEIL, so
// once the ray climbs ~1 unit off the surface `d` stops growing and the naive `k*d/t`
// would decay with t — dimming every lit surface ("whole scene darker"). Outside baked
// bricks the field also returns a huge sentinel. Both cases mean "clear from here", so we
// STOP and return the result so far the moment `d` saturates. Soft penumbrae still form
// within the ~1-unit band around real occluders (the contact-shadow case).
//
// `mint` starts the march off the originating surface so its own near-field can't
// self-shadow. `k` is penumbra hardness. `max_t` bounds the ray.
fn soft_shadow(origin: vec3<f32>, light_dir: vec3<f32>, mint: f32, max_t: f32, k: f32) -> f32 {
    var res = 1.0;
    var t = mint;
    // One per-ray chunk-search memo (like the primary march): a shadow ray stays in the same
    // chunk for many steps, so each LOD's probe is O(1) until it crosses a chunk boundary —
    // turning the previously UNCACHED per-step binary search into a cache hit.
    var cache = new_chunk_cache();
    for (var i = 0u; i < 64u; i = i + 1u) {
        if (t >= max_t) { break; }
        let d = sample_sdf_world_cached(origin + light_dir * t, &cache);
        if (d < 1e-3) {
            return 0.0;  // hit an occluder → fully shadowed
        }
        if (d >= SHADOW_FIELD_CEIL) {
            break;  // field saturated / empty space → clear beyond here
        }
        res = min(res, k * d / t);
        t += clamp(d, 1e-3, max_t);
    }
    return clamp(res, 0.0, 1.0);
}

// Shadow factor at a hit toward the sun. A small normal offset moves the ray to the lit
// side (kills self-acne); `mint` along the ray is kept SMALL so the march still samples
// the near-field where another object makes contact — a large mint skips the contact
// occluder and leaves the touch point unshadowed. The normal offset (not mint) is what
// prevents self-intersection, so mint can be sub-voxel.
fn surface_shadow(hit_pos: vec3<f32>, geo_n: vec3<f32>, light_dir: vec3<f32>, lod: u32, max_t: f32) -> f32 {
    let vs = voxel_size_at(lod);
    let origin = hit_pos + geo_n * vs;
    return soft_shadow(origin, light_dir, vs * 0.5, max_t, 8.0);
}
