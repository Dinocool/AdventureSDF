#define_import_path sdf::march

// The unified SDF raymarch, extracted so BOTH entry shaders can share it: the primary
// G-buffer pass (`sdf_raymarch.wgsl`) and the radiance-cascade trace pass
// (`sdf_rc_cascade.wgsl`, where `raymarch` plays the role of three-rc's `traceScene`). The
// loop, LOD cross-fade, iso-offset, and over-relaxation logic are unchanged from when this
// lived in the entry shader — only the module boundary is new.

#import sdf::bindings::{camera, sdf_eps, pixel_cone, over_relax, lod_blend_band, recenter_snap, lod_count, brick_world_at, CHUNK_BRICKS, voxel_size_at, abs_chunk_key}
#import sdf::brick::{
    world_to_brick_lod,
    load_material_distances,
    pick_material,
    new_chunk_cache,
    find_chunk_cached,
    resolve_march,
    sample_level_at_or_coarser,
    dist_to_brick_exit_lod,
    dist_to_chunk_exit_lod,
    in_ring_chunk,
}

struct RaymarchResult {
    hit: bool,
    dist: f32,
    object_id: u32,
    steps: u32,
    hit_pos: vec3<f32>,
    // Why the march ended: 0 = hit, 1 = escaped (t > MAX_DIST, i.e. skipped past
    // everything), 2 = ran out of steps. Lets a ray-fate debug view distinguish a
    // genuine empty-space miss from a BVH over-skip that jumps over real geometry.
    fate: u32,
    lod: u32,  // LOD level that served the hit
    atlas_base: u32,  // packed tile origin of the serving brick
    // Cross-fade weight at the hit (0 = pure serving LOD, 1 = fully the coarser neighbour).
    blend_w: f32,
    // CONTINUOUS effective LOD actually RENDERED at the hit (distance-driven morph), as a
    // float — e.g. 2.4 = level 2 morphing 40% toward level 3.
    eff_lod: f32,
};

// Per-ray quality profile. The PRIMARY ray uses full quality (cone_k 1, the uniform
// step/dist caps). A SECONDARY ray (a cascade interval) gets a degraded profile: a wider cone
// accepts hits in fewer steps and capped steps/dist bound the march.
struct MarchQuality {
    cone_k: f32,     // multiplies pixel_cone → larger = accept sooner = fewer steps
    steps_cap: u32,  // hard iteration cap for this ray
    dist_cap: f32,   // distance cap for this ray
    lod_floor: u32,  // minimum LOD served (0 = no floor); a coarse floor reads blurry + cheaper
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
        // (degrading coarser-only). Primary rays pass 0 = no-op.
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
        // Render the field at a CONTINUOUS LOD `lodc` set purely by camera distance, NOT by
        // which LOD `resolve_march` found occupied — so a finer occupancy island doesn't show
        // through a coarser region. Sample the two bracketing levels and morph between them.
        let band = lod_blend_band();
        var d_eff = d;
        var blending = false;
        var blend_w = 0.0;        // morph weight toward the coarser level
        var eff_lod = f32(lod);   // continuous LOD actually rendered (for debug/material LOD)
        let ring_bricks = camera.lod_params.y;
        let end_frac = 1.0 - 2.0 * f32(recenter_snap() * CHUNK_BRICKS) / max(ring_bricks, 1.0);
        if (band > 0.0 && end_frac > 0.0) {
            let half_l_base = 0.5 * ring_bricks * brick_world_at(0u);   // LOD-0 ring half-extent
            let cheb_cam = max(max(abs(p.x - camera.camera_pos.x), abs(p.y - camera.camera_pos.y)), abs(p.z - camera.camera_pos.z));
            // Reference distance where lodc crosses an integer. Solving so the morph L→L+1 is
            // complete at LOD L's usable ring edge gives ref = 0.5·end_frac·half_l_base.
            let ref_dist = 0.5 * end_frac * half_l_base;
            let lodc = clamp(log2(max(cheb_cam / max(ref_dist, 1e-6), 1e-6)), 0.0, f32(lod_count() - 1u));
            let level_lo = floor(lodc);
            let t_in_level = lodc - level_lo;          // 0..1 position within the bracketing level
            let blend_lod = clamp(log2(end_frac / max(end_frac - band, 1e-6)), 0.0, 1.0);
            let w = smoothstep(1.0 - blend_lod, 1.0, t_in_level);   // 0 in level core, →1 at its top
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

        // --- Sphere-trace the trilinear field ----------------------------------------
        // Over-relaxation validation FIRST (Keinert 2014): the previous relaxed step was safe
        // only if the new unbounding sphere of radius `d_eff` still reaches back over it. If
        // not, the relaxed step jumped PAST the surface — back up and resume plain tracing.
        if (prev_step > 0.0 && d_eff + prev_d < prev_step) {
            t += prev_d - prev_step;                 // undo the overshoot (negative)
            prev_d = 0.0;
            prev_step = 0.0;
            continue;
        }

        // Accept only a point reached by a validated step (never an overshoot).
        if (d_eff < max(SDF_EPS, cone)) {
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
            result.blend_w = blend_w;
            result.eff_lod = eff_lod;
            return result;
        }

        // Step `omega * d_eff`, floored so we never stall and capped at the brick exit so we
        // re-resolve LOD as the ray crosses bricks. Force omega = 1 while the cross-fade is active
        // (the blended field is then mildly non-eikonal).
        let brick_exit = dist_to_brick_exit_lod(p, dir, lod);
        let local_omega = select(OMEGA, 1.0, blending);
        let step = clamp(local_omega * d_eff, voxel_size * 0.01, brick_exit + voxel_size * 0.01);
        t += step;
        prev_d = d_eff;
        prev_step = step;
    }

    result.steps = MAX_STEPS;
    return result;
}
