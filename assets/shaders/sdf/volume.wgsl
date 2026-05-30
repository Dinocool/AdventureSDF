#define_import_path sdf::volume

// 3D distance-clipmap sampling (Stage 2 empty-space accelerator). The dense volume levels
// (bound at group 1, slots 12..15) give empty/sky rays a conservative distance to a surface
// so they can sphere-trace in BIG steps instead of crawling one brick face at a time.
//
// `sample_volume(p)` returns a conservative lower bound on the world distance to the nearest
// surface at `p`, using the FINEST clipmap level whose box contains `p` (finest = densest,
// so the tightest bound near the camera; coarser levels reach further). Returns a large
// value when `p` is outside every level (true sky) so the march takes a maximum step.
//
// Decode: each level stores distance clamped to ±(K·voxel_size) as snorm in [-1,1]; world
// distance = sample · decode[L].x  (decode[L].x = K·voxel_size, fed from VolumeConfig).

#import sdf::bindings::{volume, volume_l0, volume_l1, volume_l2, volume_l3, volume_sampler}

// Texture-space UV (0..1 per axis) of world point `p` in level `L`. A level spans
// `resolution · voxel_size` world units from its origin corner; `volume.levels[L].xyz` is
// that corner, `.w` is the level's voxel_size.
fn volume_uv(p: vec3<f32>, level: u32) -> vec3<f32> {
    let origin = volume.levels[level].xyz;
    let vs = volume.levels[level].w;
    let res = volume.volume_dims.y;          // voxels per axis
    let extent = res * vs;
    return (p - origin) / extent;
}

// True if `uv` lies inside the level's box (with a small inset so a point exactly on the
// far face still samples a valid texel rather than the ClampToEdge rail).
fn volume_uv_inside(uv: vec3<f32>) -> bool {
    let lo = all(uv >= vec3<f32>(0.0));
    let hi = all(uv <= vec3<f32>(1.0));
    return lo && hi;
}

// Sample one level's R16Snorm distance at `uv` and decode to world distance.
fn volume_sample_level(level: u32, uv: vec3<f32>) -> f32 {
    var s: f32;
    // texture_3d bindings can't be indexed dynamically in WGSL, so branch on the level.
    if (level == 0u) {
        s = textureSampleLevel(volume_l0, volume_sampler, uv, 0.0).r;
    } else if (level == 1u) {
        s = textureSampleLevel(volume_l1, volume_sampler, uv, 0.0).r;
    } else if (level == 2u) {
        s = textureSampleLevel(volume_l2, volume_sampler, uv, 0.0).r;
    } else {
        s = textureSampleLevel(volume_l3, volume_sampler, uv, 0.0).r;
    }
    return s * volume.decode[level].x;
}

// Conservative world distance to the nearest surface at `p`, taken as the MAXIMUM over all
// clipmap levels containing `p`. Every level stores a conservative lower bound (min over the
// cell, clamped), so the true distance is >= each level's sample and therefore >= their max
// — the max is still a valid lower bound, and it's the LARGEST safe step. This is the whole
// point: in open space the coarse levels' big voxel-unit clamps dominate (huge jumps); near
// a surface even the coarse level's min-over-cell reports small, so the step stays safe.
// Returns a large sentinel when `p` is outside every level (open sky → maximum step).
fn sample_volume(p: vec3<f32>) -> f32 {
    let count = u32(volume.volume_dims.x);   // active level count (0 ⇒ volume absent)
    var best = -1.0;
    var any = false;
    for (var level = 0u; level < count; level = level + 1u) {
        let uv = volume_uv(p, level);
        if (volume_uv_inside(uv)) {
            let d = volume_sample_level(level, uv);
            if (!any || d > best) { best = d; }
            any = true;
        }
    }
    if (!any) {
        return 1e9;   // outside every level: open sky
    }
    return best;
}

// The level whose sample is largest at `p` (the one that drives the step), or `count` if
// `p` is outside every level. Used by the SDF_DEBUG_VOLUME overlay to tint by serving level.
fn volume_serving_level(p: vec3<f32>) -> u32 {
    let count = u32(volume.volume_dims.x);
    var best = -1.0;
    var best_level = count;
    for (var level = 0u; level < count; level = level + 1u) {
        let uv = volume_uv(p, level);
        if (volume_uv_inside(uv)) {
            let d = volume_sample_level(level, uv);
            if (best_level == count || d > best) {
                best = d;
                best_level = level;
            }
        }
    }
    return best_level;
}
