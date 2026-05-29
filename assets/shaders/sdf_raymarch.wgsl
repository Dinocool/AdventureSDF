// SDF Raymarching Shader — entry point (Bevy 0.18).
//
// Composed from the `sdf::*` modules under shaders/sdf/ via naga_oil #import:
//   bindings — bind-group layout, structs, shared consts/accessors
//   brick    — grid/lookup/palette, trilinear distance + material sampling, normal
//   cubic    — analytic per-voxel cubic intersection
//   bvh      — empty-space-skip traversal
//   material — material table + triplanar texture sampling
//   pbr      — Cook-Torrance shading + material-seam cross-fade
// This file owns only the raymarch loop and the fragment `main` (+ debug modes).

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

#import sdf::bindings::{camera, max_steps, max_dist, sdf_eps, brick_stride}
#import sdf::brick::{
    world_to_brick,
    compute_brick_id,
    find_brick_lookup,
    scene_sdf,
    load_material_distances,
    pick_material,
    calc_normal,
}
#import sdf::cubic::{build_cell_cubic, solve_cell_cubic, dist_to_cell_exit}
#import sdf::bvh::bvh_ray_advance
#import sdf::pbr::shade_material

// --- Raymarching ---

struct RaymarchResult {
    hit: bool,
    dist: f32,
    object_id: u32,
    steps: u32,
    hit_pos: vec3<f32>,
    // Why the march ended: 0 = hit, 1 = escaped (t > MAX_DIST, i.e. skipped past
    // everything), 2 = ran out of steps. Lets the ray-fate debug view distinguish a
    // genuine empty-space miss from a BVH over-skip that jumps over real geometry.
    fate: u32,
};

fn raymarch(origin: vec3<f32>, dir: vec3<f32>) -> RaymarchResult {
    var t = 0.0;
    var steps = 0u;
    var result = RaymarchResult(false, 0.0, 0u, 0u, vec3<f32>(0.0), 2u);

    let MAX_STEPS = max_steps();
    let MAX_DIST = max_dist();
    let SDF_EPS = sdf_eps();

    let voxel_size = camera.grid_origin.w;
    let grid_orig = camera.grid_origin.xyz;
    let edge = i32(camera.grid_dims.z);

    // Ray direction in voxels-per-world-unit. The per-cell cubic uses a local
    // entry point for its origin (computed inside the loop) so coefficients stay
    // well-conditioned; only the direction is precomputed here.
    let ray_d_voxel = dir / voxel_size;

    for (var i = 0u; i < MAX_STEPS; i = i + 1u) {
        steps = i + 1u;
        let p = origin + dir * t;

        if (t > MAX_DIST) {
            result.steps = steps;
            result.fate = 1u; // escaped: marched past MAX_DIST without a hit
            return result;
        }

        let scene = scene_sdf(p);

        if (scene.in_brick) {
            // Inside a baked brick: solve the cubic for the single voxel cell
            // containing `p`. This yields the exact ray/trilinear-surface
            // intersection rather than a sphere-traced approximation.
            let loc = find_brick_lookup(compute_brick_id(world_to_brick(p)));

            // Global voxel-space position and the integer cell (lower corner)
            // containing it. The cell's local frame is [0,1]^3 over that voxel.
            let gv = (p - grid_orig) / voxel_size;
            let cell_g = floor(gv);
            let o_local = gv - cell_g;   // entry point in [0,1]^3 (small, stable)

            // Brick-local cell index; clamp so cell+1 stays within stored samples
            // (0..edge-1, the last being the shared apron plane).
            let brick_origin_v = vec3<f32>(world_to_brick(p));
            let cell_local = clamp(
                vec3<i32>(cell_g - brick_origin_v),
                vec3<i32>(0),
                vec3<i32>(edge - 2),
            );

            let cubic = build_cell_cubic(loc.atlas_base, cell_local, o_local, ray_d_voxel);

            let advance = dist_to_cell_exit(p, dir);

            // Solve in the cell-local parameter [0, advance] (distance from `p`),
            // then offset by the global `t` to recover the true ray distance.
            let cell_hit = solve_cell_cubic(cubic, 0.0, advance);
            if (cell_hit.hit) {
                let t_hit = t + cell_hit.t;
                let hit_p = origin + dir * t_hit;
                result.hit = true;
                result.dist = t_hit;
                result.object_id =
                    pick_material(load_material_distances(loc.atlas_base, hit_p), loc.palette).id;
                result.steps = steps;
                result.hit_pos = hit_p;
                result.fate = 0u; // hit
                return result;
            }

            t += advance + voxel_size * 0.001;
        } else {
            // Empty space: jump straight to the next occupied edit-AABB using the
            // BVH (big skips across truly empty space), instead of stepping one
            // brick at a time. Falls back to the brick DDA when the BVH is empty.
            t += bvh_ray_advance(p, dir) + voxel_size * 0.01;
        }
    }

    result.steps = MAX_STEPS;
    return result;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

// --- Fragment shader ---

@fragment
fn main(in: FullscreenVertexOutput) -> FragmentOutput {
    let uv = in.uv;
    // Bevy/wgpu clip space is z in [0,1] with reverse-Z (near plane = 1.0).
    // Reconstruct the ray via the near-plane point — always finite, unlike the
    // far plane which sits at infinity for Bevy's infinite reverse-Z projection.
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    let world_near = camera.inv_view_proj * ndc;
    let world_pos = world_near.xyz / world_near.w;
    let ray_dir = normalize(world_pos - camera.camera_pos.xyz);
    let ray_origin = camera.camera_pos.xyz;

    // Background gradient
    let bg_color = mix(
        vec3<f32>(0.05, 0.05, 0.12),
        vec3<f32>(0.1, 0.1, 0.18),
        uv.y,
    );

    let rm = raymarch(ray_origin, ray_dir);

    #ifdef SDF_DEBUG_RAY_FATE
    // Paint EVERY pixel by how its ray ended — placed BEFORE the miss early-return so
    // missed rays are visible (not painted as background): green = hit, red = escaped
    // past MAX_DIST (skipped over everything — the BVH-over-skip signature), blue =
    // exhausted MAX_STEPS. If the visual gap is RED, the marcher is wrongly skipping
    // geometry; if it's GREEN yet still a gap in the real render, shading is at fault.
    {
        var fate_col = vec3<f32>(0.0, 1.0, 0.0);   // hit
        if (rm.fate == 1u) { fate_col = vec3<f32>(1.0, 0.0, 0.0); }   // escaped
        if (rm.fate == 2u) { fate_col = vec3<f32>(0.0, 0.0, 1.0); }   // out of steps
        var fd = 1.0;
        if (rm.hit) {
            let c = camera.clip_from_world * vec4<f32>(rm.hit_pos, 1.0);
            fd = c.z / c.w;
        }
        return FragmentOutput(vec4<f32>(fate_col, 1.0), fd);
    }
    #endif

    if (!rm.hit) {
        return FragmentOutput(vec4<f32>(bg_color, 1.0), 0.0);
    }

    let hit_pos = rm.hit_pos;

    // True reverse-Z projection depth so the SDF surface shares the depth buffer
    // with normal geometry (wireframe, gizmos): project the world hit through the
    // forward view-proj and divide. Bevy clip space is z in [0,1], near = 1.
    let clip = camera.clip_from_world * vec4<f32>(hit_pos, 1.0);
    let ndc_depth = clip.z / clip.w;

    let normal = calc_normal(hit_pos);
    // Texture LOD from hit distance: farther hits cover more texels per pixel, so
    // bias up the mip to avoid shimmer. (No screen-space derivatives in a fullscreen
    // raymarch, so we derive LOD analytically.) Tuned constant; clamped to a sane
    // range. With single-mip textures this is currently a no-op but keeps the call
    // shape ready for the mip follow-up.
    let lod = clamp(log2(max(rm.dist, 1.0)) - 1.0, 0.0, 8.0);
    // Full PBR: triplanar textures + Cook-Torrance + material-seam cross-fade,
    // returned tonemapped/gamma-corrected. `normal` is the geometric SDF normal;
    // shade_material perturbs it per-material with the normal map.
    let shaded = shade_material(scene_sdf(hit_pos), hit_pos, normal, lod);

    // --- Debug output modes (toggled via shader_defs) ---

    #ifdef SDF_DEBUG_STEP_COUNT
    // Step count heatmap: blue (few) -> red (many)
    let t = f32(rm.steps) / f32(max_steps());
    let heatmap = vec3<f32>(t, 0.3 * (1.0 - t), 1.0 - t);
    if (rm.hit) {
        return FragmentOutput(vec4<f32>(heatmap, 1.0), ndc_depth);
    }
    return FragmentOutput(vec4<f32>(bg_color * 0.3, 1.0), 1.0);
    #endif

    #ifdef SDF_DEBUG_BVH_STEPS
    // Like the step heatmap, but colours *every* pixel (hit and miss) by march
    // cost so the empty-space traversal — which the BVH accelerates — is visible.
    // Compare against SDF_DEBUG_STEP_COUNT: with the BVH, background rays should
    // resolve in far fewer steps (deep blue) than brick-by-brick DDA.
    let bt = f32(rm.steps) / f32(max_steps());
    let bvh_heat = vec3<f32>(bt, 0.3 * (1.0 - bt), 1.0 - bt);
    let depth_out = select(1.0, ndc_depth, rm.hit);
    return FragmentOutput(vec4<f32>(bvh_heat, 1.0), depth_out);
    #endif

    #ifdef SDF_DEBUG_NORMALS
    if (rm.hit) {
        let debug_normal = normal * 0.5 + 0.5;
        return FragmentOutput(vec4<f32>(debug_normal, 1.0), ndc_depth);
    }
    return FragmentOutput(vec4<f32>(bg_color * 0.3, 1.0), 1.0);
    #endif

    #ifdef SDF_DEBUG_OBJECT_ID
    if (rm.hit) {
        // Generate distinct colors from object ID
        let hue = f32(rm.object_id) * 0.618033988749895;
        let h = fract(hue) * 6.0;
        let x = 1.0 - abs(h - 2.0) + 1.0;
        let sector = vec3<f32>(
            1.0 - abs(h - 3.0),
            1.0 - abs(h - 2.0),
            1.0 - abs(h - 1.0),
        );
        return FragmentOutput(vec4<f32>(sector, 1.0), ndc_depth);
    }
    return FragmentOutput(vec4<f32>(bg_color * 0.3, 1.0), 1.0);
    #endif

    #ifdef SDF_DEBUG_BRICK_BOUNDS
    if (rm.hit) {
        // Color the surface by the brick that contains the hit, and draw grid
        // lines on brick-cell boundaries to expose the sparse brick layout.
        let brick = world_to_brick(hit_pos);
        let brick_id = compute_brick_id(brick);
        let hue = f32(brick_id) * 0.618033988749895;
        let h = fract(hue) * 6.0;
        let tint = vec3<f32>(
            clamp(1.0 - abs(h - 3.0), 0.0, 1.0),
            clamp(1.0 - abs(h - 2.0), 0.0, 1.0),
            clamp(1.0 - abs(h - 1.0), 0.0, 1.0),
        );

        // Distance (in voxels) from the nearest brick-cell boundary plane.
        let voxel_size = camera.grid_origin.w;
        let s = f32(brick_stride());
        let rel = (hit_pos - camera.grid_origin.xyz) / voxel_size;
        let cell = rel / s;
        let frac3 = abs(fract(cell) - 0.5);
        let edge_dist = (0.5 - max(max(frac3.x, frac3.y), frac3.z)) * s;
        let line = select(1.0, 0.0, edge_dist < 0.15);

        let col = mix(vec3<f32>(0.05), tint, line * 0.85 + 0.15);
        return FragmentOutput(vec4<f32>(col, 1.0), ndc_depth);
    }
    return FragmentOutput(vec4<f32>(bg_color * 0.3, 1.0), 1.0);
    #endif

    return FragmentOutput(vec4<f32>(shaded, 1.0), ndc_depth);
}
