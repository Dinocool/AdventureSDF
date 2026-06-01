// Radiance-cascade trace + merge pass (three-rc port; Cody Bennett, github.com/CodyJasonBennett/
// three-rc). One fullscreen pass per cascade level, run coarse→fine (N-1 .. 0) ping-ponging two
// screen-sized targets. Each output pixel is one (probe, ray-direction) radiance interval:
//   - decode (probe, direction) from the pixel position
//   - reconstruct the probe's world position from the G-buffer (camera distance in albedo.a)
//   - TRACE that interval through the SDF — `raymarch` plays three-rc's `traceScene` role; on a
//     hit it returns the hit surface's emissive radiance (the GI light source), on a miss the sky
//   - if no hit and not the top cascade, MERGE the 4 nearest cascade-(N+1) probes × 4 directions
//     (2D direction bilinear × depth-aware 3D probe bilinear, the Sannikov upscaler)
//
// The SDF trace replaces three-rc's BVH `traceScene` — and because it marches the real volume it
// retains off-screen occluders/emitters (strictly better than a screen-space trace).

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import sdf::bindings::{camera, max_steps, sdf_eps}
#import sdf::march::{raymarch, MarchQuality}
#import sdf::material::material_at
#import sdf::oct::oct_decode
#import sdf::sky::sky_gradient

// group 0 = camera, group 1 = atlas (both imported above via sdf::bindings / sdf::march).
// group 2 = cascade-pass resources.
@group(2) @binding(0) var gbuf_albedo: texture_2d<f32>;   // a = camera distance (probe placement)
@group(2) @binding(1) var gbuf_normal: texture_2d<f32>;   // rg = octEncode(normal) (self-hit bias)
@group(2) @binding(2) var cascade_prev: texture_2d<f32>;  // cascade N+1 radiance (for the merge)
@group(2) @binding(3) var cascade_sampler: sampler;       // non-filtering (textureLoad only)
// The ONLY per-pass-varying value. `x` = this pass's cascade level (the rest derive from the
// camera uniform + the consts below). A vec4 for uniform-buffer alignment.
@group(2) @binding(4) var<uniform> cascade_level_param: vec4<u32>;

// Cascade parameters. `PROBE_SPACING` = cascade-0 probe tile edge in pixels — the GI's spatial
// resolution (one probe per PROBE_SPACING² pixels). Smaller = finer GI = larger cascade textures
// (width = rays_per_dim · res / PROBE_SPACING per axis). Cascade 0's DIRECTION count is separate:
// rays_per_dim = 1<<(level+2) = 4 at level 0, independent of PROBE_SPACING.
const PROBE_SPACING: i32 = 2;
// The cascade-0 interval base length, in WORLD units. three-rc hardcodes 0.005 for its
// sub-millimetre triangle scene; we scale it to our voxel size so the near cascades actually
// span geometry. With base = 0.5·voxel (0.05 at voxel 0.1) the nested intervals are contiguous
// [0,0.2],[0.2,0.8],[0.8,3.2],[3.2,12.8],… — cascade 0 ≈ 2 voxels, plenty of far reach by ~5.
fn base_interval_length() -> f32 {
    return camera.lod_params.z * 0.5;   // lod_params.z = base voxel_size
}
// albedo.a >= this ⇒ the probe pixel saw sky (matches SKY_DIST in sdf_raymarch.wgsl).
const SKY_DIST: f32 = 1e8;

fn cascade_level() -> i32 { return i32(cascade_level_param.x); }

// Number of cascades = ceil(log4(max(w, h))). Derived from the render size so no uniform is
// needed; identical across passes.
fn num_cascades() -> i32 {
    let res = camera.screen_params.xy;
    let m = max(res.x, res.y);
    return i32(ceil(log(max(m, 2.0)) / log(4.0)));
}

// World position of a probe whose centre is at `uv`, given the camera distance `dist` stored in
// the G-buffer there. Same reconstruction the composite uses (no depth-texture sampling).
fn world_from_uv(uv: vec2<f32>, dist: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    let wn = camera.inv_view_proj * ndc;
    let rd = normalize(wn.xyz / wn.w - camera.camera_pos.xyz);
    return camera.camera_pos.xyz + rd * dist;
}

// Camera distance stored in the G-buffer at `uv` (albedo.a). Clamped so a sky probe lands at a
// finite far point rather than 1e8 (its trace still misses → sky, harmlessly).
fn probe_distance(uv: vec2<f32>) -> f32 {
    let px = vec2<i32>(uv * camera.screen_params.xy);
    let d = textureLoad(gbuf_albedo, px, 0).a;
    return min(d, 1e4);
}

// Interval [near, far] for cascade N (nested, each 4× longer, contiguous: [0,L·4],[L·4,L·16],…).
fn interval_scale(level: i32) -> f32 {
    if (level == 0) { return 0.0; }
    return pow(4.0, f32(level));
}
fn interval_range(level: i32) -> vec2<f32> {
    return base_interval_length() * vec2<f32>(interval_scale(level), interval_scale(level + 1));
}

// --- bilinear helpers (three-rc verbatim) ---

fn bilinear_weights(ratio: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(
        (1.0 - ratio.x) * (1.0 - ratio.y),
        ratio.x * (1.0 - ratio.y),
        (1.0 - ratio.x) * ratio.y,
        ratio.x * ratio.y,
    );
}

fn bilinear_offset(i: i32) -> vec2<i32> {
    var offs = array<vec2<i32>, 4>(
        vec2<i32>(0, 0), vec2<i32>(1, 0), vec2<i32>(0, 1), vec2<i32>(1, 1),
    );
    return offs[i];
}

// Project `point` onto the segment [a,b], returning the clamped parametric position. Used by the
// 3D-aware bilinear ratio solve.
fn project_line_perp(a: vec3<f32>, b: vec3<f32>, point: vec3<f32>) -> f32 {
    let line = b - a;
    let len2 = dot(line, line);
    return clamp(dot(point - a, line) / max(len2, 1e-12), 0.0, 1.0);
}

// Iteratively refine the bilinear ratios so the interpolation tracks the 4 upper probes' actual
// 3D positions (not their grid coords) — keeps the merge continuous across a depth discontinuity.
fn bilinear3d_ratio(src: array<vec3<f32>, 4>, dst: vec3<f32>, init: vec2<f32>) -> vec2<f32> {
    var ratio = init;
    for (var i = 0; i < 4; i = i + 1) {
        let my1 = mix(src[0], src[2], ratio.y);
        let my2 = mix(src[1], src[3], ratio.y);
        ratio.x = project_line_perp(my1, my2, dst);
        let mx1 = mix(src[0], src[1], ratio.x);
        let mx2 = mix(src[2], src[3], ratio.x);
        ratio.y = project_line_perp(mx1, mx2, dst);
    }
    return ratio;
}

@fragment
fn main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let level = cascade_level();
    let res = vec2<i32>(camera.screen_params.xy);
    let pixel = vec2<i32>(in.position.xy);

    // Probe + direction layout for cascade N.
    let tile_size = PROBE_SPACING * (1 << u32(level));
    let rays_per_dim = 1 << u32(level + 2);
    let probe_grid = (res + vec2<i32>(tile_size - 1)) / vec2<i32>(tile_size);

    let probe2d = vec2<i32>(pixel.x % probe_grid.x, pixel.y % probe_grid.y);
    let dir2d = vec2<i32>(pixel.x / probe_grid.x, pixel.y / probe_grid.y);

    // Probe world position from the G-buffer distance at the probe centre + its surface normal.
    let probe_uv = (vec2<f32>(probe2d) + 0.5) / vec2<f32>(probe_grid);
    let probe_px = vec2<i32>(probe_uv * camera.screen_params.xy);
    let probe_dist = probe_distance(probe_uv);
    let world_pos = world_from_uv(probe_uv, probe_dist);
    let probe_n = oct_decode(textureLoad(gbuf_normal, probe_px, 0).rg);

    // Ray direction for this bin (octahedral over the full sphere).
    let ray_uv = (vec2<f32>(dir2d) + 0.5) / f32(rays_per_dim);
    let ray_dir = oct_decode(ray_uv);

    // Trace the radiance interval. `raymarch` = three-rc's `traceScene`: start at the interval
    // near edge, cap the march at the far edge. UNLIKE three-rc's BVH, the SDF march would
    // self-hit the probe's own surface at t≈0 — so offset the origin off the surface along the
    // normal before tracing.
    let iv = interval_range(level);
    // CONSTANT off-surface bias, the SAME for every ray from this probe (a direction-dependent
    // grazing lift warps the origin per-ray and breaks the floor — see git history). Sized at
    // 2× the self-hit threshold (sdf_eps 0.001) → 0.002 world = 0.02 voxel: hugs the surface as
    // tightly as possible while still clearing self-hit, so a SUNK emitter with the thinnest
    // sliver protruding is caught by a floor probe's near-horizontal rays. The tight cone
    // (cone_k 1) + no LOD inflation keep this from false-hitting the flat surface it skims.
    // Lower risks a ray grazing its own surface; higher flies over short caps.
    let origin = world_pos + probe_n * (sdf_eps() * 2.0);
    // TIGHT cone (cone_k 1.0, not the 4.0 a cheap primary-style march would use). A fat cone
    // accepts a "hit" whenever the ray passes within `pixel_cone·cone_k·t` of ANY surface — so a
    // long receiver→emitter ray skimming over the ground gets FALSELY terminated on the ground
    // partway there. GI wants to hit only ACTUAL surfaces, so we trade steps for a tight cone.
    //
    // NO forced LOD floor (lod_floor 0). A coarse LOD floor was a cheap-march optimization, but
    // coarse LODs deliberately INFLATE geometry (the iso-offset + trilinear over-estimation push
    // surfaces outward to offset coarse-voxel thinning) — so a floored ray hits the FATTENED
    // shell of a nearby object instead of passing it, and whether it clips depends on how close
    // the emitter sits to other geometry (position-dependent false occlusion). With floor 0 the
    // march still coarsens far geometry by distance (the cross-fade), but near objects keep their
    // true LOD-0 size, so rays don't false-hit inflated neighbours.
    let q = MarchQuality(1.0, max_steps(), iv.y, 0u);
    var rm = raymarch(origin, ray_dir, iv.x, q);

    // COPLANAR SELF-HIT REJECTION. A grazing ray skims just above the COPLANAR surface the probe
    // sits on (e.g. a floor probe's near-horizontal rays); the SDF's acceptance band catches that
    // surface as a false hit and returns its (usually black) emissive → black speckles that
    // shimmer as the grazing angle shifts with the camera. A real hit on an object SITTING on the
    // surface protrudes ABOVE the probe's plane, so it survives this test; only the launch-plane
    // skim is rejected. `height` = the hit's signed distance out of the probe's tangent plane.
    if (rm.hit) {
        let height = dot(rm.hit_pos - world_pos, probe_n);
        if (height < camera.lod_params.z * 0.5) {   // within ½ voxel of the launch plane → self-hit
            rm.hit = false;
        }
    }

    var radiance = vec3<f32>(0.0);

    if (rm.hit) {
        // Hit: the surface's emissive radiance is the bounce light (three-rc returns emissive).
        radiance = material_at(rm.object_id).emissive.rgb;
    } else if (level == num_cascades() - 1) {
        // Top cascade miss: the sky DOME is the environment radiance — `sky_gradient` (NO sun
        // disk). The sharp sun is added analytically + shadowed in the composite; including the
        // disk here too would double-count it (and a delta-like sun is badly under-sampled by the
        // cascade's 16 directions anyway). So the cascade carries soft sky ambient + emissive
        // bounce only; the sun is the composite's job.
        radiance = sky_gradient(ray_dir);
    } else {
        // Merge cascade N+1 → N: 2D direction bilinear × depth-aware 3D probe bilinear.
        let upper_tile = PROBE_SPACING * (1 << u32(level + 1));
        let upper_grid = (res + vec2<i32>(upper_tile - 1)) / vec2<i32>(upper_tile);

        // Direction bilinear (each lower dir maps to a 2×2 block of upper dirs).
        let dir_coord = ray_uv * f32(rays_per_dim);
        let dir_ratio = fract(dir_coord);
        let dir_weights = bilinear_weights(dir_ratio);
        let dir_base = vec2<i32>(floor(dir_coord));

        // Probe bilinear, depth-aware. Reconstruct the 4 upper probes' world positions.
        let upper_probe_coord = probe_uv * vec2<f32>(upper_grid) - 0.5;
        let upper_base = vec2<i32>(floor(upper_probe_coord));
        let upper_frac = fract(upper_probe_coord);

        var probe_coords = array<vec2<i32>, 4>();
        var probe_world = array<vec3<f32>, 4>();
        for (var p = 0; p < 4; p = p + 1) {
            let c = clamp(upper_base + bilinear_offset(p), vec2<i32>(0), upper_grid - 1);
            probe_coords[p] = c;
            let uv = (vec2<f32>(c) + 0.5) / vec2<f32>(upper_grid);
            probe_world[p] = world_from_uv(uv, probe_distance(uv));
        }
        let ratio3d = bilinear3d_ratio(probe_world, world_pos, upper_frac);
        let w3d = bilinear_weights(ratio3d);

        for (var p = 0; p < 4; p = p + 1) {
            for (var d = 0; d < 4; d = d + 1) {
                let upper_ray = dir_base * 2 + bilinear_offset(d);
                let texel = upper_ray * upper_grid + probe_coords[p];
                let interval = textureLoad(cascade_prev, texel, 0).rgb;
                radiance += interval * dir_weights[d] * w3d[p];
            }
        }
    }

    return vec4<f32>(radiance, 1.0);
}
