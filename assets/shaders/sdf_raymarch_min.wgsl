// MINIMAL SDF raymarch — rebuilding up from the known-good core after the division-free
// brick-origin fix (world_to_brick_lod / sample_brick_sdf now snap via `coord - euclid_mod`,
// no integer `/`). STAGE 1: LOD 0 only, plain sphere trace, direct find_brick_lookup,
// trilinear sample_brick_sdf, flat lambert. No cache, no resolve_march, no cubic, no LOD
// walk, no over-relax, no cone, no PBR. Add features back one at a time once this is clean.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

#import sdf::bindings::{camera, max_steps, max_dist, sdf_eps, voxel_size_at}
#import sdf::brick::{sample_sdf_world, scene_sdf}
#import sdf::material::{material_at}

// Finest voxel size, for the empty-space crawl step + normal probe offset.
fn vsize() -> f32 { return voxel_size_at(0u); }

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

struct Hit {
    hit: bool,
    pos: vec3<f32>,
};

// Scene distance: LOD-walking lookup (find_brick_at coarse->fine) + trilinear sample at
// the serving LOD. `sample_sdf_world` re-derives the brick per call so it reads correctly
// across brick AND LOD seams. Returns 1e10 in empty (unbaked) space.
fn scene_dist(p: vec3<f32>) -> f32 {
    return sample_sdf_world(p);
}

// Surface normal via tetrahedron finite differences on the trilinear field. Offset ≈ one
// voxel so snorm quantization doesn't dominate the gradient.
fn calc_normal0(p: vec3<f32>) -> vec3<f32> {
    let h = vsize();
    let k = vec2<f32>(1.0, -1.0);
    let n = k.xyy * scene_dist(p + k.xyy * h)
          + k.yyx * scene_dist(p + k.yyx * h)
          + k.yxy * scene_dist(p + k.yxy * h)
          + k.xxx * scene_dist(p + k.xxx * h);
    if (dot(n, n) > 1e-12) {
        return normalize(n);
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

fn march(origin: vec3<f32>, dir: vec3<f32>) -> Hit {
    let MAX_STEPS = max_steps();
    let MAX_DIST = max_dist();
    let SDF_EPS = sdf_eps();
    let vs = vsize();

    // Plain sphere trace: step by the true distance. No forced minimum step — a min-step
    // floor makes the march land on whichever sample first dips below eps rather than on
    // the real zero-crossing, so the recovered surface jumps by up to that step where the
    // (C0-but-not-C1) trilinear field's gradient kinks at a brick seam — reads as a
    // doubled/shifted silhouette at brick boundaries (e.g. the seam at world 0).
    var t = 0.0;
    var prev_t = 0.0;
    for (var i = 0u; i < MAX_STEPS; i = i + 1u) {
        if (t > MAX_DIST) { break; }
        let p = origin + dir * t;
        let d = scene_dist(p);
        if (d < SDF_EPS) {
            // Bisect between the last outside sample and here to pin the hit on the
            // surface independent of step size (kills seam-aligned terracing).
            var lo = prev_t;
            var hi = t;
            for (var b = 0u; b < 6u; b = b + 1u) {
                let mid = 0.5 * (lo + hi);
                if (scene_dist(origin + dir * mid) < SDF_EPS) { hi = mid; }
                else { lo = mid; }
            }
            return Hit(true, origin + dir * hi);
        }
        prev_t = t;
        // Empty space has no brick (scene_dist returns 1e10) and nothing to sphere-trace
        // against — crawl forward by a brick-ish step instead of jumping to infinity.
        // Inside a brick, step by the true distance (real sphere trace).
        let step = select(max(d, vs * 0.01), vs, d > 1e9);
        t += step;
    }
    return Hit(false, vec3<f32>(0.0));
}

@fragment
fn main(in: FullscreenVertexOutput) -> FragmentOutput {
    let uv = in.uv;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    let world_near = camera.inv_view_proj * ndc;
    let world_pos = world_near.xyz / world_near.w;
    let ray_dir = normalize(world_pos - camera.camera_pos.xyz);
    let ray_origin = camera.camera_pos.xyz;

    let bg = vec3<f32>(0.05, 0.05, 0.12);

    let h = march(ray_origin, ray_dir);
    if (!h.hit) {
        return FragmentOutput(vec4<f32>(bg, 1.0), 0.0);
    }

    let clip = camera.clip_from_world * vec4<f32>(h.pos, 1.0);
    let ndc_depth = clip.z / clip.w;

    // Resolve the material at the hit (palette argmin) -> base colour, then lambert.
    let surf = scene_sdf(h.pos);
    let albedo = material_at(surf.object_id).base_color.rgb;
    let n = calc_normal0(h.pos);
    let light_dir = normalize(vec3<f32>(0.4, 0.8, 0.3));
    let diff = max(dot(n, light_dir), 0.0);
    let col = albedo * (0.2 + 0.8 * diff);
    return FragmentOutput(vec4<f32>(col, 1.0), ndc_depth);
}
