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

#import sdf::bindings::{camera, max_steps, max_dist, sdf_eps, pixel_cone, cubic_band, cell_stride, voxel_size_at}
#import sdf::brick::{
    world_to_brick_lod,
    scene_sdf,
    load_material_distances,
    pick_material,
    calc_normal,
    step_voxel_at,
    dist_to_brick_exit,
}
#import sdf::cubic::{build_cell_cubic, solve_cell_cubic, dist_to_cell_exit}
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
    lod: u32,  // LOD level that served the hit (for the SDF_DEBUG_LOD overlay)
};

// Single unified raymarch. One loop, one resolve per step (`scene_sdf` → finest resident
// LOD + conservative distance + tile/palette), branching three ways:
//
//   1. Empty here (no resident brick at `p`): advance to the next brick face at the
//      finest resident LOD (conservative DDA — never skips over a baked surface).
//   2. LOD 0 and near the surface (`d < cubic_band`): solve the exact analytic cubic in
//      this cell for a crisp silhouette; on a miss step to the cell exit.
//   3. Otherwise (coarse LOD, or far): sphere-trace the conservative field, and accept a
//      hit once the surface is within the pixel cone (`d < max(eps, cone·t)`), so distant
//      geometry resolves at coarse LOD instead of marching all the way down to LOD 0.
//
// The conservative bake (atlas.rs) guarantees the stored field is a lower bound, so every
// sphere-trace step and DDA skip is safe — there is no GPU BVH in this path.
fn raymarch(origin: vec3<f32>, dir: vec3<f32>) -> RaymarchResult {
    var t = 0.0;
    var steps = 0u;
    var result = RaymarchResult(false, 0.0, 0u, 0u, vec3<f32>(0.0), 2u, 0u);

    let MAX_STEPS = max_steps();
    let MAX_DIST = max_dist();
    let SDF_EPS = sdf_eps();
    let CONE = pixel_cone();
    let CUBIC_BAND = cubic_band();

    let edge = i32(camera.grid_dims.z);
    let s = cell_stride();

    for (var i = 0u; i < MAX_STEPS; i = i + 1u) {
        steps = i + 1u;
        let p = origin + dir * t;

        if (t > MAX_DIST) {
            result.steps = steps;
            result.fate = 1u; // escaped: marched past MAX_DIST without a hit
            return result;
        }

        let scene = scene_sdf(p);

        // --- 1. Empty space: conservative DDA to the next brick face -----------------
        if (!scene.in_brick) {
            t += dist_to_brick_exit(p, dir) + step_voxel_at(p) * 0.01;
            continue;
        }

        let lod = scene.lod;
        let voxel_size = voxel_size_at(lod);
        let d = scene.dist;                          // conservative field (lower bound)
        let cone = CONE * t;                         // pixel-cone half-width here

        // --- 2. LOD 0 near the surface: exact analytic cubic -------------------------
        if (lod == 0u && d < CUBIC_BAND) {
            let ray_d_voxel = dir / voxel_size;       // voxels per world unit at LOD 0
            let gv = p / voxel_size;                  // LOD voxel space (world-0 anchored)
            let cell_g = floor(gv);
            let o_local = gv - cell_g;                // entry in [0,1]^3 (small, stable)
            let brick_origin_v = floor(gv / f32(s)) * f32(s);
            let cell_local = clamp(
                vec3<i32>(cell_g - brick_origin_v),
                vec3<i32>(0),
                vec3<i32>(edge - 2),
            );
            let cubic = build_cell_cubic(scene.atlas_base, cell_local, o_local, ray_d_voxel);
            let advance = dist_to_cell_exit(p, dir, lod);
            let cell_hit = solve_cell_cubic(cubic, 0.0, advance);
            if (cell_hit.hit) {
                let t_hit = t + cell_hit.t;
                let hit_p = origin + dir * t_hit;
                result.hit = true;
                result.dist = t_hit;
                result.object_id =
                    pick_material(load_material_distances(scene.atlas_base, hit_p, lod), scene.palette).id;
                result.steps = steps;
                result.hit_pos = hit_p;
                result.fate = 0u;
                result.lod = lod;
                return result;
            }
            t += advance + voxel_size * 0.001;
            continue;
        }

        // --- 3. Coarse LOD / far: sphere-trace the conservative field ----------------
        if (d < max(SDF_EPS, cone)) {
            let hit_p = p;
            result.hit = true;
            result.dist = t;
            result.object_id =
                pick_material(load_material_distances(scene.atlas_base, hit_p, lod), scene.palette).id;
            result.steps = steps;
            result.hit_pos = hit_p;
            result.fate = 0u;
            result.lod = lod;
            return result;
        }
        // Step by the conservative distance (safe lower bound), with a floor so we never
        // stall, and never past the brick exit so we re-resolve LOD as the ray moves.
        let brick_exit = dist_to_brick_exit(p, dir);
        t += clamp(d, voxel_size * 0.01, brick_exit + voxel_size * 0.01);
    }

    result.steps = MAX_STEPS;
    return result;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

// Reverse-Z projected depth for a debug pixel: the hit's true depth so the overlay
// shares the depth buffer with other geometry, or far (1.0) on a miss.
fn debug_depth(rm: RaymarchResult) -> f32 {
    if (rm.hit) {
        let c = camera.clip_from_world * vec4<f32>(rm.hit_pos, 1.0);
        return c.z / c.w;
    }
    return 1.0;
}

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

    // --- Cost / fate debug modes -------------------------------------------------
    // These are placed BEFORE the miss early-return so they paint EVERY pixel — hit
    // AND miss. Missed rays (escaped past MAX_DIST or out of steps) are usually the
    // most expensive, so a cost heatmap that drops them to background hides exactly
    // the rays you want to measure. Depth is the hit's projected depth when there is a
    // hit, else far (1.0).

    #ifdef SDF_DEBUG_RAY_FATE
    // Colour by how the ray ended: green = hit, red = escaped past MAX_DIST (skipped
    // over everything), blue = exhausted MAX_STEPS. If a visual gap is RED the marcher
    // is wrongly skipping geometry; if GREEN yet still a gap in the real render, shading
    // is at fault.
    {
        var fate_col = vec3<f32>(0.0, 1.0, 0.0);   // hit
        if (rm.fate == 1u) { fate_col = vec3<f32>(1.0, 0.0, 0.0); }   // escaped
        if (rm.fate == 2u) { fate_col = vec3<f32>(0.0, 0.0, 1.0); }   // out of steps
        return FragmentOutput(vec4<f32>(fate_col, 1.0), debug_depth(rm));
    }
    #endif

    #ifdef SDF_DEBUG_STEP_COUNT
    // March-cost heatmap over every pixel: blue (few steps) → red (many). Misses are
    // included so escaped / out-of-steps rays show their true cost, not background.
    {
        let c = f32(rm.steps) / f32(max_steps());
        let heatmap = vec3<f32>(c, 0.3 * (1.0 - c), 1.0 - c);
        return FragmentOutput(vec4<f32>(heatmap, 1.0), debug_depth(rm));
    }
    #endif

    #ifdef SDF_DEBUG_BVH_STEPS
    // Same march-cost heatmap, kept as a distinct toggle for empty-space-traversal
    // analysis. Also covers every pixel (hit and miss).
    {
        let c = f32(rm.steps) / f32(max_steps());
        let heat = vec3<f32>(c, 0.3 * (1.0 - c), 1.0 - c);
        return FragmentOutput(vec4<f32>(heat, 1.0), debug_depth(rm));
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

    // --- Debug output modes (hit-only; cost/fate modes return earlier) ---

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
        // Color the surface by the brick that contains the hit (at the serving LOD),
        // and draw grid lines on brick-cell boundaries to expose the brick layout.
        let lod = rm.lod;
        let voxel_size = voxel_size_at(lod);
        let brick = world_to_brick_lod(hit_pos, lod);
        let brick_hash = u32(brick.x * 73856093 ^ brick.y * 19349663 ^ brick.z * 83492791);
        let hue = f32(brick_hash & 0xffu) * 0.618033988749895;
        let h = fract(hue) * 6.0;
        let tint = vec3<f32>(
            clamp(1.0 - abs(h - 3.0), 0.0, 1.0),
            clamp(1.0 - abs(h - 2.0), 0.0, 1.0),
            clamp(1.0 - abs(h - 1.0), 0.0, 1.0),
        );

        // Distance (in voxels) from the nearest brick-cell boundary plane.
        let s = f32(cell_stride());
        let rel = hit_pos / voxel_size;
        let cell = rel / s;
        let frac3 = abs(fract(cell) - 0.5);
        let edge_dist = (0.5 - max(max(frac3.x, frac3.y), frac3.z)) * s;
        let line = select(1.0, 0.0, edge_dist < 0.15);

        let col = mix(vec3<f32>(0.05), tint, line * 0.85 + 0.15);
        return FragmentOutput(vec4<f32>(col, 1.0), ndc_depth);
    }
    return FragmentOutput(vec4<f32>(bg_color * 0.3, 1.0), 1.0);
    #endif

    #ifdef SDF_DEBUG_LOD
    // Tint the hit by the LOD that served it, so the clipmap rings are directly
    // visible. Discrete 4-colour cycle by `lod % 4`: white, green, blue, red.
    if (rm.hit) {
        var col = vec3<f32>(1.0, 1.0, 0.0);        // 4+: yellow
        if (rm.lod == 0u) { col = vec3<f32>(1.0, 1.0, 1.0); }   // 0: white
        else if (rm.lod == 1u) { col = vec3<f32>(0.0, 1.0, 0.0); }   // 1: green
        else if (rm.lod == 2u) { col = vec3<f32>(0.0, 0.4, 1.0); }   // 2: blue
        else if (rm.lod == 3u) { col = vec3<f32>(1.0, 0.0, 0.0); }   // 3: red
        let shaded_lod = mix(shaded, col, 0.65);
        return FragmentOutput(vec4<f32>(shaded_lod, 1.0), ndc_depth);
    }
    return FragmentOutput(vec4<f32>(bg_color * 0.3, 1.0), 1.0);
    #endif

    return FragmentOutput(vec4<f32>(shaded, 1.0), ndc_depth);
}
