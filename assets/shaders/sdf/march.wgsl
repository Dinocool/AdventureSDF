#define_import_path sdf::march

// The unified SDF raymarch, extracted so BOTH entry shaders can share it: the primary
// G-buffer pass (`sdf_raymarch.wgsl`) and the radiance-cascade trace pass
// (`sdf_rc_cascade.wgsl`, where `raymarch` plays the role of three-rc's `traceScene`). The
// loop, LOD cross-fade, iso-offset, and over-relaxation logic are unchanged from when this
// lived in the entry shader — only the module boundary is new.

#import sdf::bindings::{camera, sdf_eps, pixel_cone, over_relax, lod_blend_band, recenter_snap, lod_count, brick_world_at, CHUNK_BRICKS, voxel_size_at, clipmap_exit_t, chunk_buf}
#import sdf::brick::{
    world_to_brick_lod,
    resolve_material,
    new_chunk_cache,
    find_chunk_cached,
    resolve_march,
    sample_level_at_or_coarser,
    dist_to_brick_exit_lod,
    dist_to_chunk_exit_lod,
    dist_over_empty_bricks,
    in_ring_chunk,
    ChunkCache,
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

// How close (in coarse-LOD voxels, plus the pixel cone) to the surface the LOD cross-fade
// engages. The morph only shifts the rendered ISO-SURFACE position, so its 2-3 extra field
// samples per step only matter near the surface; far from it we step on the single cheap
// `resolve_march` distance. The margin must exceed the worst-case adjacent-LOD distance
// discrepancy (~a coarse voxel) so the morph is always active before a hit can be accepted.
// Raise it if LOD-seam artifacts appear; lower it for a slightly cheaper march.
const NEAR_SURFACE_VOXELS: f32 = 3.0;

// Ray-differential LOD floor: sample the SDF no finer than this many pixel-footprints per voxel.
// The clipmap rings are tuned to ~1 voxel/pixel, so 1.0 would be a no-op; >1 samples COARSER than a
// pixel on far/grazing rays, merging the distant dense object field into fewer, bigger blobs so a
// horizon ray skims far fewer voxels (the red-band fix). Sub-pixel ⇒ ~quality-neutral; coarser ⇒
// never overshoots (safe). Higher = cheaper but blockier far horizon. Tune with the camera FOV/res.
const LOD_PIXEL_BIAS: f32 = 2.0;


// Result of the LOD cross-fade morph at a point `p`. `d_eff` is the DISTANCE the surface is
// rendered from (the camera-distance-driven continuous-LOD blend of two bracketing levels);
// `blending` is true when the morph actually engaged (the two-level mix ran, so the field is
// mildly non-eikonal and the caller should disable over-relaxation); `blend_w` / `eff_lod`
// are the morph weight and the continuous rendered LOD (for debug/material LOD).
struct LodMorph {
    d_eff: f32,
    blending: bool,
    blend_w: f32,
    eff_lod: f32,
};

// DISTANCE-driven continuous-LOD morph, factored out so BOTH the primary raymarch and the
// shadow march (`sdf::shadows::soft_shadow`) sample the SAME LOD-blended field. Before this
// was extracted, the primary march rendered the occluder through this morph while the shadow
// march sampled the RAW finest-occupied field (`resolve_march`) — so a smoothly-rendered
// surface cast a blockier (coarser-LOD-faceted) shadow. Sharing this helper makes the shadow
// "see" exactly the occluder the camera sees.
//
// Inputs: `p` the sample point; `d_raw` the field value already resolved at `p` (the finest-
// occupied `resolve_march`/`sample_sdf_world` distance) used both as the morph's fallback and
// for the near-surface gate; `lod` the serving LOD of that resolve; `cone` the pixel-cone
// half-width at `p` (= CONE·t — pass 0 for an ungated, always-on morph); `cache` the per-ray
// chunk memo. The LOD selection keys off CAMERA distance (the clipmap is camera-centred), so it
// is correct for a shadow ray's point too. Returns the raw `d_raw` unchanged (blending=false)
// when the band is off, the ray is outside the usable ring, or the point is too far from the
// surface for the morph to matter (the perf gate) — behaviourally identical to the pre-refactor
// inline block.
fn lod_crossfade(
    p: vec3<f32>,
    d_raw: f32,
    lod: u32,
    cone: f32,
    cache: ptr<function, ChunkCache>,
) -> LodMorph {
    let SDF_EPS = sdf_eps();
    let band = lod_blend_band();
    var out = LodMorph(d_raw, false, 0.0, f32(lod));

    let ring_bricks = camera.lod_params.y;
    let end_frac = 1.0 - 2.0 * f32(recenter_snap() * CHUNK_BRICKS) / max(ring_bricks, 1.0);
    // PERF gate: the morph only moves the rendered iso-surface, so its 2-3 extra field samples
    // per step only matter near the surface. Far from it the caller steps on the cheap `d_raw`,
    // exactly as the band==0 path does; the morph re-engages within `morph_reach` of the
    // surface, before any hit-accept test.
    let coarse_vox = voxel_size_at(min(lod + 1u, lod_count() - 1u));
    let morph_reach = max(SDF_EPS, cone) + coarse_vox * NEAR_SURFACE_VOXELS;
    if (!(band > 0.0 && end_frac > 0.0 && d_raw < morph_reach)) {
        return out;
    }

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
    let s0 = sample_level_at_or_coarser(p, k, cache);
    if (s0.in_brick) {
        if (w > 0.0 && k + 1u < lod_count()) {
            let s1 = sample_level_at_or_coarser(p, k + 1u, cache);
            if (s1.in_brick) {
                out.d_eff = mix(s0.dist, s1.dist, w);
                out.blending = true;
                out.blend_w = w;
                out.eff_lod = f32(s0.lod) + w * f32(s1.lod - s0.lod);
            } else {
                out.d_eff = s0.dist;
                out.eff_lod = f32(s0.lod);
            }
        } else {
            out.d_eff = s0.dist;
            out.eff_lod = f32(s0.lod);
        }
    }
    return out;
}

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
    // Bound the march at whichever comes first: the requested far cap, or the exit of the resident
    // clipmap box. The latter is what stops a MISS ray (sky, or a grazing crest that clears the
    // hill) from crawling the void brick-by-brick once it has left all geometry — the dominant
    // wasted-step cost. `t > MAX_DIST` below then returns it as escaped (fate = 1 = sky).
    let MAX_DIST = min(q.dist_cap, clipmap_exit_t(origin, dir));
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

        // Quality LOD floor. (a) Secondary rays clamp to `q.lod_floor`. (b) RAY-DIFFERENTIAL floor:
        // never sample FINER than the pixel footprint — the LOD whose voxel ≈ `cone·t`. With the wide
        // clipmap rings the finest ring reaches far, so a distant GRAZING ray would otherwise skim the
        // horizon at the FINE-voxel scale (tiny steps → step-budget blowout — the red horizon band).
        // Flooring the sampled LOD to ~1 voxel/pixel makes far/grazing rays take coarse-voxel-sized
        // steps; the far geometry is sub-pixel anyway, so it's quality-neutral, and coarser sampling
        // never overshoots (safe). Near geometry (cone·t < base voxel ⇒ cone_lod 0) is untouched/sharp.
        let cone_t = CONE * t;
        let cone_lod = u32(clamp(ceil(log2(max(cone_t * LOD_PIXEL_BIAS / camera.lod_params.z, 1.0))), 0.0, f32(lod_count() - 1u)));
        let floor_lod = max(q.lod_floor, cone_lod);
        if (floor_lod > 0u && scene.in_brick && scene.lod < floor_lod) {
            let coarse = sample_level_at_or_coarser(p, floor_lod, &cache);
            if (coarse.in_brick) {
                scene.dist = coarse.dist;
                scene.lod = coarse.lod;
                scene.atlas_base = coarse.atlas_base;
                scene.mat_atlas_base = coarse.mat_atlas_base;
                scene.palette = coarse.palette;
            }
        }

        // --- 1. Empty space: hierarchical skip, coarsest→fine (biggest box wins) ------
        //
        // Two cases, both keyed off the chunk directory only (no field samples):
        //  • a coarse ABSENT in-ring chunk  → jump its whole CHUNK box (provably empty: the bake cull
        //    never enqueues a chunk that has geometry);
        //  • a coarse RESIDENT chunk whose brick at `p` is empty — air ABOVE the terrain that shares
        //    the chunk holding the surface below — → occupancy-DDA across its empty-brick run
        //    (`dist_over_empty_bricks`). This is the horizon-crawl fix: instead of one brick per march
        //    step it skips the whole empty run in one shot. Coarse bricks ⇒ big skips, and a brick
        //    empty at a coarse LOD is empty at every finer LOD, so the skip can't pass a surface.
        // Walking coarsest→fine takes the biggest applicable box; the finest-resident-brick fallback
        // only fires if `p` is outside every ring (next iteration's MAX_DIST then ends the ray).
        if (!scene.in_brick) {
            let levels = lod_count();
            var adv = dist_to_brick_exit_lod(p, dir, scene.window_lod) + voxel_size_at(scene.window_lod) * 0.01;
            for (var L = levels; L > 0u; ) {
                L = L - 1u;                              // coarsest first
                let coord = world_to_brick_lod(p, L);
                let ci = find_chunk_cached(coord, L, &cache);
                if (ci >= 0) {
                    adv = dist_over_empty_bricks(chunk_buf[u32(ci)], p, dir, L) + voxel_size_at(L) * 0.01;
                    break;
                } else if (in_ring_chunk(coord, L)) {
                    adv = dist_to_chunk_exit_lod(p, dir, L) + voxel_size_at(L) * 0.01;
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
        let cone = cone_t;                           // pixel-cone half-width here (= CONE·t)

        // --- LOD cross-fade: DISTANCE-driven continuous-LOD morph ----------------------
        // Render the field at a CONTINUOUS LOD `lodc` set purely by camera distance, NOT by
        // which LOD `resolve_march` found occupied — so a finer occupancy island doesn't show
        // through a coarser region. Sample the two bracketing levels and morph between them.
        // Shared with the shadow march (`soft_shadow`) so the shadow sees the SAME morphed field.
        let morph = lod_crossfade(p, d, lod, cone, &cache);
        let d_eff = morph.d_eff;
        let blending = morph.blending;
        let blend_w = morph.blend_w;       // morph weight toward the coarser level
        let eff_lod = morph.eff_lod;       // continuous LOD actually rendered (for debug/material LOD)

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
            // Linear-crossing hit refinement (Inigo Quilez, "Terrain Marching"). A conservative step
            // lands NEAR the surface, not ON it — accepting `p` verbatim quantises the hit to the step,
            // so ridges read soft/blocky and the shading normal is sampled off-surface. Use the last
            // two samples' slope to project along the ray to where the field reaches 0 and place the
            // hit there: with `d_eff` and the previous `prev_d` over `prev_step`, the field drops by
            // `denom = prev_d - d_eff` across the step, so the zero crossing is `dt = d_eff·prev_step/
            // denom` from here (>0 = surface ahead of a cone-accept, <0 = back from a slight overshoot).
            // Clamped to ±one step so a near-tangent slope can't fling the hit; falls back to `t` when
            // there's no usable prior sample (prev_step == 0) or the slope is ~flat.
            var t_hit = t;
            if (prev_step > 0.0) {
                let denom = prev_d - d_eff;
                if (denom > 1e-5) {
                    t_hit = clamp(t + d_eff * prev_step / denom, t - prev_step, t + prev_step);
                }
            }
            let hit_p = origin + dir * t_hit;
            result.hit = true;
            result.dist = t_hit;
            result.object_id =
                resolve_material(scene.mat_atlas_base, hit_p, lod, scene.palette).id;
            result.steps = steps;
            result.hit_pos = hit_p;
            result.fate = 0u;
            result.lod = lod;
            result.atlas_base = scene.atlas_base;
            result.blend_w = blend_w;
            result.eff_lod = eff_lod;
            return result;
        }

        // Step `omega · d_eff` (Keinert over-relaxation), floored so we never stall and capped at the
        // brick exit so the next iteration re-resolves LOD across bricks. An active cross-fade steps
        // plain (omega = 1) — the blended field is non-eikonal. (The empty-space horizon crawl is
        // handled by the occupancy-aware chunk/brick DDA above; in-brick grazing acceleration —
        // segment tracing with a local directional-Lipschitz bound — is a separate planned step.)
        let brick_exit = dist_to_brick_exit_lod(p, dir, lod);
        let local_omega = select(OMEGA, 1.0, blending);
        let step = clamp(local_omega * d_eff, voxel_size * 0.01, brick_exit + voxel_size * 0.01);
        t += step;
        // Carry the slope memo for the Keinert undo + the linear-crossing hit refinement.
        prev_d = d_eff;
        prev_step = step;
    }

    result.steps = MAX_STEPS;
    return result;
}
