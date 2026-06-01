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

#import sdf::bindings::{camera, max_steps, max_dist, sdf_eps, pixel_cone, over_relax, lod_blend_band, recenter_snap, surface_bias, lod_count, brick_world_at, CHUNK_BRICKS, cell_stride, voxel_size_at, abs_chunk_key, local_brick_index, chunk_buf, ChunkLookup, TEXTURE_WORLD_SCALE}
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
    sample_level_at_or_coarser,
    dist_to_brick_exit_lod,
    dist_to_chunk_exit_lod,
    in_ring_chunk,
}
#import sdf::pbr::{resolve_surface, shade_surface, shade_material_env, sun_dir, PbrInputs, fresnel_schlick_roughness}
#import sdf::sky::sky_color

// Cone-prepass seed texture: per-8×8-tile start distance (R32Float), written by
// sdf_cone_prepass.wgsl. The march starts each pixel at its tile's seed-t instead of 0,
// amortising the empty-corridor march across the tile. The seed is a guaranteed lower
// bound on every pixel's hit distance (the cone stops before any surface enters the tile),
// so starting from it never skips geometry. Group 2 — groups 0/1 are camera + atlas.
@group(2) @binding(0) var cone_seed: texture_2d<f32>;
const CONE_TILE: i32 = 8;

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
    // CONTINUOUS effective LOD actually RENDERED at the hit (distance-driven morph), as a
    // float — e.g. 2.4 = level 2 morphing 40% toward level 3. This is what the surface is
    // drawn from; `lod` above is only the finest-OCCUPIED level resolve_march found (patchy).
    // The SDF_DEBUG_LOD overlay colours by THIS so it reflects what we actually draw.
    eff_lod: f32,
};

// Per-ray quality profile. The PRIMARY ray uses full quality (cone_scale 1, the uniform
// step/dist caps). A SECONDARY ray (a reflection) gets a degraded profile scaled by its hit
// roughness: a wider cone accepts hits in fewer steps and capped steps/dist bound the march —
// invisible once the result is blurred into the IBL specular term.
struct MarchQuality {
    cone_k: f32,     // multiplies pixel_cone → larger = accept sooner = fewer steps
    steps_cap: u32,  // hard iteration cap for this ray
    dist_cap: f32,   // distance cap for this ray
    lod_floor: u32,  // minimum LOD served: a rough reflection floors to a COARSE level so the
                     // reflected geometry AND material read blurry (true glossy softening) and
                     // the bigger coarse-brick steps make the march cheaper still. 0 = no floor.
};

// Single unified raymarch. One cached resolve per step (`resolve_march` → finest resident
// LOD + trilinear distance + tile/palette, memoising the chunk search across steps via a
// per-ray `ChunkCache`), branching three ways:
//
//   1. Empty here (no resident brick at `p`): advance to the next brick face at the finest
//      resident-window LOD via brick-geometry DDA (`dist_to_brick_exit_lod` — a pure lattice
//      step, so it never skips over a baked brick). The LOD comes from the SAME resolve, so
//      empty steps cost no second chunk search.
//   2. Otherwise (in a brick): sphere-trace the trilinear field with Keinert over-relaxation
//      (step `over_relax · d`, fall back when unbounding spheres separate), accepting a hit
//      once the surface is within the pixel cone (`d < max(eps, cone·t)`).
//
// The stored field is the true trilinear SDF sampled at voxel centres (atlas.rs); empty-space
// DDA steps by brick geometry (always safe) and the in-brick sphere-trace is bounded by the
// brick exit, so the march is robust. There is no GPU BVH in this path.
fn raymarch(origin: vec3<f32>, dir: vec3<f32>, start_t: f32, q: MarchQuality) -> RaymarchResult {
    var t = start_t;
    var steps = 0u;
    var result = RaymarchResult(false, 0.0, 0u, 0u, vec3<f32>(0.0), 2u, 0u, 0u, 0.0, 0.0);

    let MAX_STEPS = q.steps_cap;
    let MAX_DIST = q.dist_cap;
    let SDF_EPS = sdf_eps();
    let CONE = pixel_cone() * q.cone_k;
    let OMEGA = over_relax();

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

        var scene = resolve_march(p, &cache);

        // Quality LOD floor (secondary rays). If this ray must render no finer than
        // `q.lod_floor` and `resolve_march` served a finer brick, re-resolve at the floor
        // (degrading coarser-only). Rewriting `scene` here makes the WHOLE downstream path —
        // distance, served LOD, tile/palette for the reflected material, cross-fade — use the
        // coarse level, so a rough reflection reads blurry in both geometry and texture, and
        // the larger coarse-brick steps make the march cheaper. Primary rays pass 0 = no-op.
        if (q.lod_floor > 0u && scene.in_brick && scene.lod < q.lod_floor) {
            let coarse = sample_level_at_or_coarser(p, q.lod_floor, &cache);
            if (coarse.in_brick) {
                scene.dist = coarse.dist;
                scene.lod = coarse.lod;
                scene.atlas_base = coarse.atlas_base;
                scene.palette = coarse.palette;
            }
        }

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

        // --- LOD cross-fade: DISTANCE-driven continuous-LOD morph ----------------------
        // We render the field at a CONTINUOUS LOD `lodc` set purely by the camera distance,
        // NOT by which LOD `resolve_march` happened to find occupied. That decoupling is the
        // fix for the hard blend seam: `resolve_march` serves the FINEST OCCUPIED LOD, so a
        // finer occupancy ISLAND (grazing surface → sparse per-brick cull, lattice-phase
        // dependent) used to show through inside a coarser region; the old weight keyed off
        // that served LOD's ring half-extent, which DOUBLES at a served-LOD flip → the blend
        // weight jumped → a hard step. Driving the morph from distance makes both the weight
        // AND the two levels being mixed continuous in screen space, so the island simply
        // renders at the LOD its distance calls for (no patch).
        //
        // `lodc = log2(cheb_cam / ref)` with `ref = end_frac · half_l_base` (the LOD-0 ring's
        // fade-complete radius), so `lodc ≈ L` at LOD L's fade-complete distance. We sample
        // the two BRACKETING absolute levels `k = floor(lodc)` and `k+1` (degrading to coarser
        // only — never finer — via `sample_level_at_or_coarser`) and morph between them over
        // the top `blend_lod` fraction of each level. CONTINUITY across a level boundary
        // k→k+1: from below `t_in_level→1 ⇒ w→1 ⇒ d_eff→mix(lvl_k, lvl_{k+1}, 1)=lvl_{k+1}`;
        // from above `level_lo=k+1, t_in_level→0 ⇒ w→0 ⇒ d_eff=lvl_{k+1}` — they match. Near
        // the camera `lodc→0`, `w→0`, so `d_eff=level 0` and the analytic cubic still owns the
        // crisp near silhouette. `half_l_base`/`cheb_cam` measured from the RAW camera so the
        // transition glides while chunk-snapped residency moves invisibly under it.
        let band = lod_blend_band();
        var d_eff = d;
        var blending = false;
        var blend_w = 0.0;        // morph weight toward the coarser level (LOD debug overlay)
        var eff_lod = f32(lod);   // continuous LOD actually rendered (for the iso-offset)
        let ring_bricks = camera.lod_params.y;
        let end_frac = 1.0 - 2.0 * f32(recenter_snap() * CHUNK_BRICKS) / max(ring_bricks, 1.0);
        if (band > 0.0 && end_frac > 0.0) {
            let half_l_base = 0.5 * ring_bricks * brick_world_at(0u);   // LOD-0 ring half-extent
            let cheb_cam = max(max(abs(p.x - camera.camera_pos.x), abs(p.y - camera.camera_pos.y)), abs(p.z - camera.camera_pos.z));
            // Reference distance where lodc crosses an integer. LOD L's USABLE ring edge is
            // d_L = end_frac·half_l_base·2^L; the morph L→L+1 must be COMPLETE there (LOD L
            // stops being resident beyond it), i.e. lodc(d_L) = L+1. Solving log2(d_L/ref)=L+1
            // gives ref = 0.5·end_frac·half_l_base. (Dropping the 0.5 — the earlier bug —
            // shifts lodc down by one whole LOD: renders a level too fine and the morph reads
            // inverted/late.)
            let ref_dist = 0.5 * end_frac * half_l_base;
            let lodc = clamp(log2(max(cheb_cam / max(ref_dist, 1e-6), 1e-6)), 0.0, f32(lod_count() - 1u));
            let level_lo = floor(lodc);
            let t_in_level = lodc - level_lo;          // 0..1 position within the bracketing level
            // Fade band as a fraction of ONE LOD level (log-space image of the world-space
            // `band` fraction of a ring). Clamp to [0,1]; >0 keeps a sharp per-level core.
            let blend_lod = clamp(log2(end_frac / max(end_frac - band, 1e-6)), 0.0, 1.0);
            let w = smoothstep(1.0 - blend_lod, 1.0, t_in_level);   // 0 in the level core, →1 at its top
            let k = u32(level_lo);
            let s0 = sample_level_at_or_coarser(p, k, &cache);
            if (s0.in_brick) {
                if (w > 0.0 && k + 1u < lod_count()) {
                    let s1 = sample_level_at_or_coarser(p, k + 1u, &cache);
                    if (s1.in_brick) {
                        d_eff = mix(s0.dist, s1.dist, w);
                        blending = true;
                        blend_w = w;
                        eff_lod = f32(s0.lod) + w * f32(s1.lod - s0.lod);
                    } else {
                        d_eff = s0.dist;
                        eff_lod = f32(s0.lod);
                    }
                } else {
                    d_eff = s0.dist;
                    eff_lod = f32(s0.lod);
                }
            }
        }

        // --- Coarse-LOD iso-offset: re-inflate the trilinear shrink ------------------
        // Trilinear interpolation over-estimates distance on a convex surface, pulling the
        // zero-isosurface inward by ≈(h²/8)·κ — so coarse LODs render objects too thin. Take
        // the surface where the field equals `eff_eps` (> 0) instead of 0, pushing it back
        // out by ≈eff_eps (|∇field| ≈ 1). QUADRATIC in voxel size to match the h² bias law
        // (one α works across LODs); ZERO at LOD 0 so the analytic cubic's crisp near surface
        // is untouched. Lerp by `blend_w` so it stays continuous into LOD+1's (2×) offset
        // across the cross-fade. `d_iso` is the distance to this inflated surface.
        // Keyed on the CONTINUOUS rendered LOD `eff_lod` (= served `lod` when not morphing),
        // so the inflation grows smoothly with the distance-driven morph instead of stepping
        // at a served-LOD flip. `vs_eff = base · 2^eff_lod` is the voxel size at that LOD;
        // QUADRATIC in it (the h² bias law). Zero at LOD 0 (cubic owns the near surface).
        let vs_eff = camera.lod_params.z * exp2(eff_lod);
        let eff_eps = select(
            0.0,
            surface_bias() * vs_eff * vs_eff / camera.lod_params.z,
            eff_lod > 0.0,
        );
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
            result.eff_lod = eff_lod;
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
//
// `roughness` (the reflecting surface's) scales a CHEAP quality profile: the result is
// blurred into the IBL specular term by roughness anyway, so a rough reflection can use a
// fat cone (accept hits in far fewer steps), a low step cap, and a short distance — only a
// near-mirror keeps a tight, longer ray. This is what keeps the secondary march affordable.
// `out_steps` returns the reflection march's step count (for the SDF_DEBUG_REFLECT_STEPS
// overlay); pass a throwaway when the cost isn't needed.
fn trace_reflection(origin: vec3<f32>, refl_dir: vec3<f32>, roughness: f32, start_t: f32, out_steps: ptr<function, u32>) -> vec3<f32> {
    // Roughness-driven profile, but with a RAISED FLOOR so even a perfect mirror (r→0) can't
    // approach full primary cost — a measured low-roughness reflection was ~+10ms, doubling the
    // SDF draw. The floor: min cone ×4 (accept hits sooner), steps capped at ~1/3 the primary
    // (not 1/2), min dist 0.4× (reflections matter most nearby), and lod_floor ≥1 ALWAYS (a
    // mirror reads one LOD coarse — barely visible, but the coarse bricks take bigger steps).
    // Rough end (r→1): cone ×16, ~24 steps, 0.2× dist, lod_floor 4 — a cheap blurry probe.
    let r = clamp(roughness, 0.0, 1.0);
    let q = MarchQuality(
        mix(4.0, 16.0, r),
        u32(mix(f32(max_steps()) * 0.34, 24.0, r)),
        max_dist() * mix(0.4, 0.2, r),
        u32(clamp(1.0 + r * 4.0, 1.0, 4.0)),
    );
    // Reflection rays have no cone-prepass tile seed; the caller passes a cheap `start_t` to
    // skip the near-field empty gap above the surface (the origin is already nudged off it).
    let rm = raymarch(origin, refl_dir, start_t, q);
    *out_steps = rm.steps;
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

    // Seed the march from the cone prepass: the per-tile start distance for this pixel's
    // 8×8 tile (a guaranteed lower bound on its hit distance, so no geometry is skipped).
    let tile = vec2<i32>(uv * camera.screen_params.xy) / CONE_TILE;
    let start_t = textureLoad(cone_seed, tile, 0).r;

    // Primary ray: full quality (cone ×1, the uniform step/dist caps, no LOD floor).
    let rm = raymarch(ray_origin, ray_dir, start_t, MarchQuality(1.0, max_steps(), max_dist(), 0u));

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

    // Height-map relief is baked into the SDF field (see sdf_render::height) — the hit position
    // and its gradient normal already reflect the carved surface, so shading needs no extra work.
    let hit_pos = rm.hit_pos;
    let geo_normal = calc_normal(rm.hit_pos);

    // Analytic texture LOD (no screen-space derivatives in a fullscreen raymarch). Pick the
    // mip whose texel covers ~1 pixel: the pixel's world footprint at this hit is
    // `pixel_cone · dist`, stretched by 1/|cosθ| at grazing angles (a glancing pixel covers
    // a much longer streak of surface). Divide by the texture's world-per-texel
    // (TEXTURE_WORLD_SCALE) to get texels/pixel, then log2 → mip. This replaces the old
    // `log2(dist)-1` constant, which ignored both the pixel cone and the grazing stretch and
    // so under-mipped grazing terrain → cache thrash + shimmer.
    let cos_graze = max(abs(dot(ray_dir, geo_normal)), 0.15);  // floor caps the stretch (~6.7×)
    let footprint_world = pixel_cone() * max(rm.dist, 1.0) / cos_graze;
    let texels_per_pixel = footprint_world / TEXTURE_WORLD_SCALE;
    let lod = clamp(log2(max(texels_per_pixel, 1.0)), 0.0, 8.0);

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
    var refl_steps = 0u;  // reflection-march cost for the SDF_DEBUG_REFLECT_STEPS overlay
#ifdef SDF_REFLECTIONS
    // FRESNEL IMPORTANCE GATE. A traced reflection only matters where it's actually VISIBLE —
    // and visibility is `env * fresnel_schlick_roughness(ndv, f0, roughness)` (the exact weight
    // ambient_ibl applies). For a dielectric (terrain, f0≈0.04) head-on, that weight is ~0.04,
    // so the reflection contributes a few percent and is dominated by diffuse — yet a pure
    // roughness gate still traces a full march there. Computing the Fresnel weight FIRST and
    // skipping when it's below a small threshold kills the camera-facing terrain pixels (the
    // bulk of a heightmap) and keeps the grazing horizon-ward slopes where F→1 and the
    // reflection reads. Angle- AND roughness-aware, physically the term we'd multiply anyway.
    let view = normalize(camera.camera_pos.xyz - hit_pos);
    let ndv = max(dot(p.normal, view), 0.0);
    let f0 = mix(vec3<f32>(0.04), p.albedo, p.metallic);
    let fres = fresnel_schlick_roughness(ndv, f0, p.roughness);
    let refl_weight = max(max(fres.r, fres.g), fres.b) * (1.0 - p.roughness);
    if (refl_weight > 0.04) {
        let refl_dir = reflect(-view, p.normal);
        let bias = voxel_size_at(rm.lod) * 2.0;
        // CHEAP START: the reflection ray has no cone-prepass seed, so from t=0 it crawls the
        // near-field empty space above the surface. Start it a few coarse voxels out (past the
        // self-bias shell) so it skips the guaranteed-empty gap it just left — the march's own
        // chunk-DDA handles the rest. Conservative (small) so no real near reflection is missed.
        let start = voxel_size_at(rm.lod) * 4.0;
        env_radiance = trace_reflection(hit_pos + normal * bias, refl_dir, p.roughness, start, &refl_steps);
    }
#endif

    // Reflection-march cost heatmap: black where no reflection ray was traced (gated out),
    // else blue (few steps) → red (many), normalised to the primary step cap. Shows exactly
    // which pixels pay for a secondary march and how deep it goes.
    #ifdef SDF_DEBUG_REFLECT_STEPS
    {
        let c = f32(refl_steps) / f32(max_steps());
        let heat = vec3<f32>(c, 0.3 * (1.0 - c), 1.0 - c) * step(0.5, f32(refl_steps));
        return FragmentOutput(vec4<f32>(heat, 1.0), ndc_depth);
    }
    #endif

    // Raw reflected radiance: `env_radiance` exactly as trace_reflection returned it (the sky
    // on a gated-out / missed ray), BEFORE shade_surface's ambient_ibl folds it in. The
    // shading path does `mix(env_radiance, irradiance, roughness)` (pbr.wgsl), so at high
    // roughness a perfectly-good reflection becomes invisible there — this view bypasses that
    // mix so you can see the traced result itself and confirm the march/roughness profile.
    #ifdef SDF_DEBUG_REFLECT_RAW
    {
        let raw = pow(env_radiance / (env_radiance + vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
        return FragmentOutput(vec4<f32>(raw, 1.0), ndc_depth);
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
        // Colour by the CONTINUOUS effective LOD actually rendered (distance-driven morph),
        // not the patchy finest-occupied `rm.lod` — so the overlay shows what we DRAW. A
        // smooth spatial gradient ⇒ continuous morph; scattered patches ⇒ a real seam.
        let lf = floor(rm.eff_lod);
        let col = mix(lod_debug_color(u32(lf)), lod_debug_color(u32(lf) + 1u), rm.eff_lod - lf);
        let shaded_lod = mix(shaded, col, 0.65);
        return FragmentOutput(vec4<f32>(shaded_lod, 1.0), ndc_depth);
    }
    return FragmentOutput(vec4<f32>(bg_color * 0.3, 1.0), 1.0);
    #endif

    return FragmentOutput(vec4<f32>(shaded, 1.0), ndc_depth);
}
