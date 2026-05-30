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

#import sdf::bindings::{camera, max_steps, max_dist, sdf_eps, pixel_cone, cubic_band, over_relax, lod_blend_band, recenter_snap, surface_bias, lod_count, brick_world_at, CHUNK_BRICKS, cell_stride, voxel_size_at, abs_chunk_key, local_brick_index, chunk_buf, ChunkLookup}
#import sdf::brick::{
    world_to_brick_lod,
    scene_sdf,
    sample_brick_sdf,
    load_material_distances,
    pick_material,
    calc_normal,
    BrickLocation,
    brick_in_chunk,
    ChunkCache,
    new_chunk_cache,
    find_chunk_cached,
    resolve_march,
    dist_to_brick_exit_lod,
    dist_to_chunk_exit_lod,
    in_ring_chunk,
}
#import sdf::cubic::{build_cell_cubic, solve_cell_cubic, dist_to_cell_exit}
#import sdf::pbr::{resolve_surface, shade_surface, shade_material_env, sun_dir, PbrInputs}
#import sdf::sky::sky_color

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
    atlas_base: u32,  // packed tile origin of the serving brick (for SDF_DEBUG_TILE_ID)
    // Cross-fade weight at the hit (0 = pure serving LOD, 1 = fully the coarser neighbour).
    // Surfaced for the SDF_DEBUG_LOD overlay so the per-pixel blend band is visible.
    blend_w: f32,
};

// Single unified raymarch. One cached resolve per step (`resolve_march` → finest resident
// LOD + trilinear distance + tile/palette, memoising the chunk search across steps via a
// per-ray `ChunkCache`), branching three ways:
//
//   1. Empty here (no resident brick at `p`): advance to the next brick face at the finest
//      resident-window LOD via brick-geometry DDA (`dist_to_brick_exit_lod` — a pure lattice
//      step, so it never skips over a baked brick). The LOD comes from the SAME resolve, so
//      empty steps cost no second chunk search.
//   2. LOD 0 and near the surface (`d < cubic_band`): solve the exact analytic cubic in this
//      cell for a crisp silhouette; on a miss step to the cell exit.
//   3. Otherwise (coarse LOD, or far): sphere-trace the trilinear field with Keinert
//      over-relaxation (step `over_relax · d`, fall back when unbounding spheres separate),
//      accepting a hit once the surface is within the pixel cone (`d < max(eps, cone·t)`).
//
// The stored field is the true trilinear SDF sampled at voxel centres (atlas.rs); empty-space
// DDA steps by brick geometry (always safe) and the in-brick sphere-trace is bounded by the
// brick exit, so the march is robust. There is no GPU BVH in this path.
fn raymarch(origin: vec3<f32>, dir: vec3<f32>) -> RaymarchResult {
    var t = 0.0;
    var steps = 0u;
    var result = RaymarchResult(false, 0.0, 0u, 0u, vec3<f32>(0.0), 2u, 0u, 0u, 0.0);

    let MAX_STEPS = max_steps();
    let MAX_DIST = max_dist();
    let SDF_EPS = sdf_eps();
    let CONE = pixel_cone();
    let CUBIC_BAND = cubic_band();
    let OMEGA = over_relax();

    let edge = i32(camera.grid_dims.z);
    let s = cell_stride();

    // Per-ray chunk-search memo (NanoVDB/Tree64 accessor): a marching ray stays in the same
    // chunk for many steps, so each LOD's probe is O(1) until it crosses a chunk boundary.
    var cache = new_chunk_cache();
    // Previous unbounding-sphere radius + the step actually taken, for over-relaxation
    // fallback (Keinert 2014): if the new sphere doesn't reach back to the previous one,
    // the relaxed step overshot — undo it and re-take the safe `d` step.
    var prev_d = 0.0;
    var prev_step = 0.0;

    for (var i = 0u; i < MAX_STEPS; i = i + 1u) {
        steps = i + 1u;
        let p = origin + dir * t;

        if (t > MAX_DIST) {
            result.steps = steps;
            result.fate = 1u; // escaped: marched past MAX_DIST without a hit
            return result;
        }

        let scene = resolve_march(p, &cache);

        // --- 1. Empty space: hierarchical chunk-DDA skip -----------------------------
        //
        // Skip whole CHUNK boxes, not one brick at a time. A chunk absent from the table
        // AND inside its LOD's resident ring is (treated as) empty — the bake cull never
        // enqueues a chunk that has geometry — so we step to the far face of the LARGEST
        // such box around `p`. Walk coarse→fine so the biggest provably-empty box wins.
        // A present chunk (may hold occupied bricks) or an out-of-ring chunk (unbaked;
        // a coarser LOD's ring covers it) is never chunk-skipped — we fall through to the
        // brick-exit floor, which never crosses a baked brick. (Accepted caveat: a chunk
        // still queued for bake reads as empty and may be skipped for a few frames under
        // fast camera motion — a transient hole that fills in.)
        if (!scene.in_brick) {
            let wl = scene.window_lod;
            var adv = dist_to_brick_exit_lod(p, dir, wl) + voxel_size_at(wl) * 0.01;
            let levels = lod_count();
            for (var L = levels; L > 0u; ) {
                L = L - 1u;                              // coarsest first = biggest box
                let coord = world_to_brick_lod(p, L);
                let key = abs_chunk_key(coord, L);
                let ci = find_chunk_cached(L, key.x, key.y, &cache);
                if (ci < 0 && in_ring_chunk(coord, L)) {
                    adv = max(adv, dist_to_chunk_exit_lod(p, dir, L) + voxel_size_at(L) * 0.01);
                    break;
                }
            }
            t += adv;
            prev_d = 0.0;
            prev_step = 0.0;
            continue;
        }

        let lod = scene.lod;
        let voxel_size = voxel_size_at(lod);
        let d = scene.dist;                          // trilinear SDF at p
        let cone = CONE * t;                         // pixel-cone half-width here

        // --- LOD cross-fade: morph L → L+1, gliding PER-PIXEL with the camera ----------
        // The serving LOD L's resident ring box is chunk-snapped (it only re-centres in
        // discrete jumps), but the cross-fade does NOT have to follow that snapped edge: a
        // coarser ring nests around the finer one, so wherever L is resident L+1 is too. So
        // we place the fade at a CAMERA-RELATIVE radius that is guaranteed to sit inside L's
        // resident window for any camera offset, and measure from the RAW camera — so the
        // transition slides smoothly with the camera while residency snaps invisibly under it.
        //
        // The camera sits at most `snap` chunks from the snapped window centre, so a
        // Chebyshev ball of radius `R_safe = half_l - snap*chunk_world` around the camera is
        // fully inside the window (resident at L, and at L+1). Complete the fade by R_safe
        // (`end_frac` of the half-extent); start it `band` earlier. `brick_world` cancels, so
        // end_frac is the clean constant `1 - 2*snap*CHUNK_BRICKS/ring_bricks`.
        let band = lod_blend_band();
        var d_eff = d;
        var blending = false;
        var blend_w = 0.0;        // exposed for the LOD debug overlay
        let ring_bricks = camera.lod_params.y;
        let end_frac = 1.0 - 2.0 * f32(recenter_snap() * CHUNK_BRICKS) / max(ring_bricks, 1.0);
        if (band > 0.0 && end_frac > 0.0 && lod + 1u < lod_count()) {
            let half_l = 0.5 * ring_bricks * brick_world_at(lod);   // ring half-extent
            let cheb_cam = max(max(abs(p.x - camera.camera_pos.x), abs(p.y - camera.camera_pos.y)), abs(p.z - camera.camera_pos.z));
            let frac_cam = cheb_cam / max(half_l, 1e-6);
            let w = smoothstep(end_frac - band, end_frac, frac_cam);
            if (w > 0.0) {
                // Probe the coarser neighbour through the per-ray chunk cache (the fine→
                // coarse resolve already searched + cached L+1's chunk, so this is ~free).
                let coord1 = world_to_brick_lod(p, lod + 1u);
                let key1 = abs_chunk_key(coord1, lod + 1u);
                let ci1 = find_chunk_cached(lod + 1u, key1.x, key1.y, &cache);
                if (ci1 >= 0) {
                    let loc1 = brick_in_chunk(chunk_buf[u32(ci1)], coord1);
                    if (loc1.found) {
                        let d_l1 = sample_brick_sdf(loc1.atlas_base, p, lod + 1u);
                        d_eff = mix(d, d_l1, w);
                        blending = true;
                        blend_w = w;
                    }
                }
            }
        }

        // --- 2. LOD 0 near the surface: exact analytic cubic -------------------------
        // (SDF_DISABLE_CUBIC skips this branch → pure sphere-trace everywhere, for bisect.)
        // Gated off inside the cross-fade shell (`blending`) so those near-surface hits fall
        // through to the sphere-trace and blend, instead of snapping via the unblended cubic.
#ifdef SDF_DISABLE_CUBIC
        if (false) {
#else
        if (lod == 0u && d < CUBIC_BAND && !blending) {
#endif
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
                result.atlas_base = scene.atlas_base;
                return result;
            }
            t += advance + voxel_size * 0.001;
            prev_d = 0.0;
            prev_step = 0.0;
            continue;
        }

        // --- Coarse-LOD iso-offset: re-inflate the trilinear shrink ------------------
        // Trilinear interpolation over-estimates distance on a convex surface, pulling the
        // zero-isosurface inward by ≈(h²/8)·κ — so coarse LODs render objects too thin. Take
        // the surface where the field equals `eff_eps` (> 0) instead of 0, pushing it back
        // out by ≈eff_eps (|∇field| ≈ 1). QUADRATIC in voxel size to match the h² bias law
        // (one α works across LODs); ZERO at LOD 0 so the analytic cubic's crisp near surface
        // is untouched. Lerp by `blend_w` so it stays continuous into LOD+1's (2×) offset
        // across the cross-fade. `d_iso` is the distance to this inflated surface.
        let eps_l = select(
            0.0,
            surface_bias() * voxel_size * voxel_size / camera.lod_params.z,
            lod > 0u,
        );
        let eff_eps = mix(eps_l, 2.0 * eps_l, blend_w);
        let d_iso = d_eff - eff_eps;

        // --- 3. Coarse LOD / far: sphere-trace the (inflated) trilinear field ---------
        //
        // Traces `d_iso` (the cross-faded field shifted out by the iso-offset; = `d` when
        // both are off). Over-relaxation validation FIRST (Keinert 2014): the previous
        // relaxed step `prev_step` was safe only if the new unbounding sphere of radius
        // `d_iso` still reaches back over it (`d_iso + prev_d >= prev_step`). If not, the
        // relaxed step jumped PAST the surface — `p` is now inside/beyond it, so we must NOT
        // accept this point as a hit (doing so lands the hit at a view-dependent spot →
        // swimming normals/textures on camera rotation). Back up and resume plain tracing.
        if (prev_step > 0.0 && d_iso + prev_d < prev_step) {
            t += prev_d - prev_step;                 // undo the overshoot (negative)
            prev_d = 0.0;
            prev_step = 0.0;
            continue;
        }

        // Accept only a point reached by a validated step (never an overshoot).
        if (d_iso < max(SDF_EPS, cone)) {
            let hit_p = p;
            result.hit = true;
            result.dist = t;
            result.object_id =
                pick_material(load_material_distances(scene.atlas_base, hit_p, lod), scene.palette).id;
            result.steps = steps;
            result.hit_pos = hit_p;
            result.fate = 0u;
            result.lod = lod;
            result.atlas_base = scene.atlas_base;
            result.blend_w = blend_w;     // cross-fade amount toward LOD+1 (for the debug overlay)
            return result;
        }

        // Step `omega * d_iso`, floored so we never stall (a negative `d_iso` just inside the
        // inflated surface becomes a safe tiny forward step), and capped at the brick exit so
        // we re-resolve LOD as the ray crosses bricks. omega = 1 is plain sphere tracing.
        // Force omega = 1 whenever the cross-fade OR the iso-offset is active: both make the
        // effective field mildly non-eikonal, so over-relaxation could overshoot — and these
        // are far/thin shells where over-relaxation buys almost nothing anyway.
        let brick_exit = dist_to_brick_exit_lod(p, dir, lod);
        let local_omega = select(OMEGA, 1.0, blending || eff_eps > 0.0);
        let step = clamp(local_omega * d_iso, voxel_size * 0.01, brick_exit + voxel_size * 0.01);
        t += step;
        prev_d = d_iso;
        prev_step = step;
    }

    result.steps = MAX_STEPS;
    return result;
}

// --- Reflections (Stage 4) ---------------------------------------------------------
//
// Mirror radiance along `refl_dir` from a primary hit: march a SECONDARY ray through the
// same SDF. On a hit, shade that surface with the analytic sky as ITS environment (one
// bounce only — the reflected surface doesn't itself spawn another reflection ray, which
// would be unbounded recursion). On a miss, return the sky. The result is the env_radiance
// the primary surface's specular term consumes, so a smooth metal mirrors real geometry.
//
// `origin` is the primary hit nudged off its surface (caller offsets by the geometric
// normal) so the reflection ray doesn't immediately re-hit the surface it left.
fn trace_reflection(origin: vec3<f32>, refl_dir: vec3<f32>) -> vec3<f32> {
    let rm = raymarch(origin, refl_dir);
    if (!rm.hit) {
        return sky_color(refl_dir, sun_dir());
    }
    let hp = rm.hit_pos;
    let n = calc_normal(hp);
    let lod = clamp(log2(max(rm.dist, 1.0)) - 1.0, 0.0, 8.0);
    // One bounce: the reflected surface uses the sky as its own environment.
    let sky_env = sky_color(reflect(refl_dir, n), sun_dir());
    return shade_material_env(scene_sdf(hp), hp, n, lod, sky_env);
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

// Discrete per-LOD tint for the SDF_DEBUG_LOD overlay: white, green, blue, red, then
// yellow for 4+. Factored out so the overlay can `mix` LOD L's and LOD L+1's colours by the
// cross-fade weight, painting the per-pixel blend band as a gradient.
fn lod_debug_color(lod: u32) -> vec3<f32> {
    if (lod == 0u) { return vec3<f32>(1.0, 1.0, 1.0); }
    if (lod == 1u) { return vec3<f32>(0.0, 1.0, 0.0); }
    if (lod == 2u) { return vec3<f32>(0.0, 0.4, 1.0); }
    if (lod == 3u) { return vec3<f32>(1.0, 0.0, 0.0); }
    return vec3<f32>(1.0, 1.0, 0.0);
}

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

    // Background = the same analytic sky the IBL/reflections sample, so a ray miss and
    // a metal's reflection of the horizon agree. Tonemapped to match shaded surfaces.
    let sky = sky_color(ray_dir, sun_dir());
    let bg_color = pow(sky / (sky + vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));

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

    // Texture LOD from hit distance: farther hits cover more texels per pixel, so
    // bias up the mip to avoid shimmer. (No screen-space derivatives in a fullscreen
    // raymarch, so we derive LOD analytically.) Tuned constant; clamped to a sane range.
    let lod = clamp(log2(max(rm.dist, 1.0)) - 1.0, 0.0, 8.0);

    // Height-map relief is baked into the SDF field (see sdf_render::height) — the hit position
    // and its gradient normal already reflect the carved surface, so shading needs no extra work.
    let hit_pos = rm.hit_pos;
    let geo_normal = calc_normal(rm.hit_pos);

    // True reverse-Z projection depth so the SDF surface shares the depth buffer with normal
    // geometry (wireframe, gizmos): project the (displaced) world hit through the forward
    // view-proj and divide. Bevy clip space is z in [0,1], near = 1.
    let clip = camera.clip_from_world * vec4<f32>(hit_pos, 1.0);
    let ndc_depth = clip.z / clip.w;

    let normal = geo_normal;

    // Resolve the cross-faded PBR inputs at the (displaced) surface, pick the environment
    // radiance, then shade once. `p.normal` is the normal-mapped shading normal; `normal` is
    // the geometric one (used for the reflection-ray offset so it leaves along the surface).
    let scene = scene_sdf(hit_pos);
    let p: PbrInputs = resolve_surface(scene, hit_pos, normal, lod);

    // Environment radiance for the specular/IBL term: the analytic sky by default. With
    // SDF_REFLECTIONS, smooth/metallic surfaces trace a real reflection ray for true
    // geometry-to-geometry mirroring; rough/diffuse surfaces keep the cheap sky (the
    // gate keeps the common case at one march).
    var env_radiance = sky_color(reflect(-normalize(camera.camera_pos.xyz - hit_pos), p.normal), sun_dir());
#ifdef SDF_REFLECTIONS
    if (p.metallic > 0.1 || p.roughness < 0.3) {
        let view = normalize(camera.camera_pos.xyz - hit_pos);
        let refl_dir = reflect(-view, p.normal);
        let bias = voxel_size_at(rm.lod) * 2.0;
        env_radiance = trace_reflection(hit_pos + normal * bias, refl_dir);
    }
#endif

    // Shade to LINEAR radiance, then tonemap (Reinhard) + approximate gamma once here.
    let lit = shade_surface(p, hit_pos, normal, rm.lod, env_radiance);
    let shaded = pow(lit / (lit + vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));

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
        // Per-brick colour cycle, keyed on the INTEGER brick index the lookup uses
        // (`world_to_brick_lod` → stride-aligned origin → / stride). Adjacent bricks step
        // through a hue sequence, so the grid reads as a smoothly cycling patchwork. A
        // duplicated brick shows the SAME colour repeating where the hue should have
        // advanced; a missing brick shows background. Distinct colours in a duplicated
        // region mean the lookup is returning genuinely different bricks there.
        let lod = rm.lod;
        let s = cell_stride();
        let origin = world_to_brick_lod(hit_pos, lod);   // stride-aligned, signed
        // Exact-multiple-of-stride / stride → the integer brick index (works for negatives).
        let bi = origin / s;

        // Sequential hue: weight the axes by small coprime steps so neighbours in any
        // direction land on clearly different hues that cycle through the full wheel.
        let seq = f32(bi.x) * 0.13 + f32(bi.y) * 0.27 + f32(bi.z) * 0.41;
        let h = fract(seq) * 6.0;
        let col = clamp(
            vec3<f32>(abs(h - 3.0) - 1.0, 2.0 - abs(h - 2.0), 2.0 - abs(h - 4.0)),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );

        // Light shading so the surface shape still reads under the colours.
        let shade = clamp(dot(normal, normalize(vec3<f32>(0.4, 0.8, 0.3))) * 0.4 + 0.6, 0.2, 1.0);
        return FragmentOutput(vec4<f32>(col * shade, 1.0), ndc_depth);
    }
    return FragmentOutput(vec4<f32>(bg_color * 0.3, 1.0), 1.0);
    #endif

    #ifdef SDF_DEBUG_TILE_ID
    if (rm.hit) {
        // Colour by the resolved ATLAS TILE (atlas_base) — the actual texels the hit
        // sampled. Distinguishes two failure modes for the "half renders twice" bug:
        //   • duplicated halves SAME colour  → tile collision (two bricks → one tile)
        //   • duplicated halves DIFFERENT    → distinct tiles holding duplicated bake data
        // Integer hash of atlas_base (Wang-style mix) so ADJACENT tiles get unrelated hues
        // — a linear-in-base hue makes neighbouring tiles look identical and mislead the
        // "same tile?" read (col_px differs by only 64 between adjacent tiles).
        var hsh = rm.atlas_base * 0x9e3779b9u + 0x85ebca6bu;
        hsh = hsh ^ (hsh >> 16u);
        hsh = hsh * 0x7feb352du;
        hsh = hsh ^ (hsh >> 15u);
        let h = f32(hsh & 0xffffu) / 65535.0 * 6.0;
        let col = clamp(
            vec3<f32>(abs(h - 3.0) - 1.0, 2.0 - abs(h - 2.0), 2.0 - abs(h - 4.0)),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        let shade = clamp(dot(normal, normalize(vec3<f32>(0.4, 0.8, 0.3))) * 0.4 + 0.6, 0.2, 1.0);
        return FragmentOutput(vec4<f32>(col * shade, 1.0), ndc_depth);
    }
    return FragmentOutput(vec4<f32>(bg_color * 0.3, 1.0), 1.0);
    #endif

    #ifdef SDF_DEBUG_CHUNK_ID
    if (rm.hit) {
        // Colour by the resolved CHUNK key at the hit (same key the lookup binary-searches).
        // Paired with Tile ID (which showed the two duplicated halves share ONE tile): if
        // the halves here are the SAME colour → both bricks are in the same chunk (a local
        // index / popcount collapse); DIFFERENT colour → two chunks alias to one tile
        // (cross-chunk tile-run packing overlap).
        let coord = world_to_brick_lod(hit_pos, rm.lod);
        let key = abs_chunk_key(coord, rm.lod);
        let li = local_brick_index(coord);
        // Integer hash of the full chunk key (Wang-style mix) so ADJACENT chunks get
        // unrelated hues — a linear-in-index hue makes neighbouring chunks (e.g. -1 vs 0
        // where a sphere straddles the origin) look identical and mislead the diagnosis.
        var hsh = key.x * 0x9e3779b9u + key.y;
        hsh = hsh ^ (hsh >> 16u);
        hsh = hsh * 0x7feb352du;
        hsh = hsh ^ (hsh >> 15u);
        let h = f32(hsh & 0xffffu) / 65535.0 * 6.0;
        var col = clamp(
            vec3<f32>(abs(h - 3.0) - 1.0, 2.0 - abs(h - 2.0), 2.0 - abs(h - 4.0)),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        // Brightness by local slot so two bricks in the SAME chunk are the same hue but
        // distinguishable shades (and identical shade ⇒ literally the same local slot).
        col = col * (0.4 + 0.6 * f32(li) / 63.0);
        return FragmentOutput(vec4<f32>(col, 1.0), ndc_depth);
    }
    return FragmentOutput(vec4<f32>(bg_color * 0.3, 1.0), 1.0);
    #endif

    #ifdef SDF_DEBUG_LOD
    // Tint the hit by the LOD that served it, so the clipmap rings are directly
    // visible. Discrete 4-colour cycle by `lod % 4`: white, green, blue, red. Where the
    // per-pixel cross-fade is active the tint is `mix`-blended L→L+1 by `blend_w`, so the
    // transition band reads as a colour gradient between two ring colours (not a hard
    // line) — directly visualising the blended region.
    if (rm.hit) {
        let col_l = lod_debug_color(rm.lod);
        let col_l1 = lod_debug_color(rm.lod + 1u);
        let col = mix(col_l, col_l1, rm.blend_w);
        let shaded_lod = mix(shaded, col, 0.65);
        return FragmentOutput(vec4<f32>(shaded_lod, 1.0), ndc_depth);
    }
    return FragmentOutput(vec4<f32>(bg_color * 0.3, 1.0), 1.0);
    #endif

    return FragmentOutput(vec4<f32>(shaded, 1.0), ndc_depth);
}
