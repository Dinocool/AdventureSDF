#define_import_path sdf::shadows

// SDF soft shadows. A secondary ray marched from the surface toward the sun through the SAME
// sparse field as the primary march — INCLUDING its empty-space skipping — so a distant occluder
// (a tower) casts a real shadow across the gap onto the ground, not just a contact shadow where
// objects touch. Tracks the Inigo Quilez penumbra estimate (closest approach to an occluder,
// scaled by how far along the ray it happened) for soft edges. No shadow maps. Returns
// 1 = fully lit, 0 = occluded.
//
// KNOWN ARTIFACTS (to be solved via a dedicated harness): a hard penumbra→umbra transition and
// brick-faceted silhouettes on distant occluders, both from the discrete/clamped voxel field and
// the empty-space DDA skip. Neither a near-surface step-floor change nor a coarse-LOD shadow floor
// fixed them; an optimal approach is still TBD.

#import sdf::bindings::{voxel_size_at, lod_count, DIST_BAND_VOXELS, shadow_softness}
#import sdf::brick::{
    resolve_march,
    world_to_brick_lod,
    find_chunk_cached,
    dist_to_brick_exit_lod,
    dist_to_chunk_exit_lod,
    in_ring_chunk,
    new_chunk_cache,
}
// For the hard-shadow path (softness 0): a binary occlusion test via the primary raymarch.
#import sdf::march::{raymarch, MarchQuality}

// A sample this small means the ray entered an occluder → hard shadow.
const SHADOW_HIT_EPS: f32 = 1e-3;
// Voxels to trim off the band edge before a sample is trusted by the penumbra. The field
// saturates at `DIST_BAND_VOXELS · vs`; a trilinear cell spans up to ~√3 voxels, so a corner can
// hit the clamp even when the centre `d` is up to ~2 voxels inside the band. Excluding that
// margin keeps the boxy clamp shell entirely out of `k*d/t` (the fix that removed the boxy
// sun-visibility artifact that was obvious near LOD 0).
const PENUMBRA_BAND_MARGIN_VOXELS: f32 = 2.0;
// Iteration cap. Empty space is skipped a whole chunk at a time, so this covers long shadows;
// the cost is the in-brick steps near an occluder, which terminate fast (a hit returns 0).
const SHADOW_MAX_STEPS: u32 = 96u;

// Inigo Quilez soft shadow over the sparse field. `mint` starts the march off the originating
// surface; `k` is penumbra hardness; `max_t` bounds the ray.
fn soft_shadow(origin: vec3<f32>, light_dir: vec3<f32>, mint: f32, max_t: f32, k: f32) -> f32 {
    var res = 1.0;
    var t = mint;
    // Per-ray chunk-search memo (like the primary march): the ray stays in one chunk for many
    // steps, so each LOD probe is O(1) until it crosses a chunk boundary.
    var cache = new_chunk_cache();

    for (var i = 0u; i < SHADOW_MAX_STEPS; i = i + 1u) {
        if (t >= max_t) { break; }
        let p = origin + light_dir * t;
        let scene = resolve_march(p, &cache);

        // --- Empty space: hierarchical chunk-DDA skip (the key difference from the old march) ---
        // No resident brick here. Step to the far face of the LARGEST provably-empty box around
        // `p` (a chunk absent from the table AND inside its LOD's resident ring is empty), walking
        // coarse→fine so the biggest box wins. This is what lets the ray jump the gap between an
        // object and the ground instead of bailing at the first saturated sample.
        if (!scene.in_brick) {
            let wl = scene.window_lod;
            var adv = dist_to_brick_exit_lod(p, light_dir, wl) + voxel_size_at(wl) * 0.01;
            for (var L = lod_count(); L > 0u; ) {
                L = L - 1u;
                let coord = world_to_brick_lod(p, L);
                if (find_chunk_cached(coord, L, &cache) < 0 && in_ring_chunk(coord, L)) {
                    adv = max(adv, dist_to_chunk_exit_lod(p, light_dir, L) + voxel_size_at(L) * 0.01);
                    break;
                }
            }
            t += adv;
            continue;
        }

        // --- In a brick: cone-trace the penumbra ---
        // Sample the RAW finest-resident-LOD distance (`resolve_march`) — NO LOD cross-fade blend.
        // The cross-fade pulls in coarser neighbour levels, which still smeared the penumbra; the
        // cone term alone provides the softening.
        let d = scene.dist;
        if (d < SHADOW_HIT_EPS) {
            return 0.0; // entered an occluder → fully shadowed
        }
        let vs = voxel_size_at(scene.lod);
        // The baked distance saturates at the PER-LOD band (`DIST_BAND_VOXELS · vs`, atlas.rs); the
        // outer voxels of that band are the boxy snorm-clamp shell. Only feed samples comfortably
        // inside the band to the IQ penumbra so the clamp shell never paints into `k*d/t`. (Invisible
        // at coarse LOD, where band ≥ the old hardcoded 1.0 ceiling, but obvious near LOD 0.)
        let valid_band = (DIST_BAND_VOXELS - PENUMBRA_BAND_MARGIN_VOXELS) * vs;
        if (d < valid_band) {
            // Fade the penumbra contribution to NOTHING as `d` approaches the band edge. The clamp
            // band limits how far the field can SEE an occluder, so a sample near the edge is "maybe
            // much farther" and must not darken the result. Without this fade, a low `k` makes the
            // first in-band sample `k*d/t` < 1 right at the edge → a hard dark onset (1 → dark in one
            // step), which is the artifact the Shadow Softness slider exposed at low values. Now the
            // onset is smooth at every `k`; confident darkening only well inside the band.
            let conf = smoothstep(valid_band, valid_band * 0.5, d);
            res = min(res, mix(1.0, k * d / t, conf));
        }
        // Sphere-trace by the unbounding sphere `d`, floored so we never stall and capped at the
        // brick exit so the next `resolve_march` re-picks the LOD across bricks. A saturated `d` is
        // still a valid lower bound on the true distance, so the step stays conservative.
        let brick_exit = dist_to_brick_exit_lod(p, light_dir, scene.lod);
        t += clamp(d, vs * 0.5, brick_exit + vs * 0.01);
    }

    return clamp(res, 0.0, 1.0);
}

// Shadow factor at a hit toward the sun. A small normal offset moves the ray to the lit side
// (kills self-acne); `mint` is kept sub-voxel so a near-field contact occluder still registers —
// the normal offset (not mint) is what prevents self-intersection.
//
// Driven by the "Shadow Softness" slider `k` (= `shadow_softness()`, march_params.y):
//   * k == 0  → a HARD shadow: a binary occlusion test through the primary `raymarch` (exactly
//               like the reflection pass), artifact-free because it only reacts to a real surface
//               hit, never to a near-miss off the surface.
//   * k  > 0  → a cone-traced soft shadow: the sun subtends a cone of half-angle θ, `k = 1/θ`,
//               and the penumbra `min(k·d/t) = min(d/(θ·t))` is the clear fraction of that cone.
//               HIGHER k = sharper/tighter (and less near-miss darkening); lower = softer/wider.
fn surface_shadow(hit_pos: vec3<f32>, geo_n: vec3<f32>, light_dir: vec3<f32>, lod: u32, max_t: f32) -> f32 {
    let vs = voxel_size_at(lod);
    let origin = hit_pos + geo_n * vs;
    let k = shadow_softness();
    if (k <= 0.0) {
        let q = MarchQuality(1.0, SHADOW_MAX_STEPS, max_t, 0u);
        return select(1.0, 0.0, raymarch(origin, light_dir, vs * 0.5, q).hit);
    }
    return soft_shadow(origin, light_dir, vs * 0.5, max_t, k);
}
