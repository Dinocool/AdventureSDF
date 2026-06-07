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

#import sdf::bindings::{voxel_size_at, lod_count, DIST_BAND_VOXELS, shadow_softness, clipmap_exit_t}
#import sdf::brick::{
    resolve_march,
    dist_to_brick_exit_lod,
    empty_space_advance,
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

        // --- Empty space: the SHARED hierarchical skip (brick::empty_space_advance) ---
        // Previously the shadow march had its own skip that lacked the resident-chunk occupancy-DDA, so
        // grazing sun-shadow rays crawled brick-by-brick (the "horizon crawl"). It now uses the same
        // accelerator as the primary march — one definition, applies everywhere.
        if (!scene.in_brick) {
            t += empty_space_advance(p, light_dir, scene.window_lod, &cache);
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
    // Cap the ray at the resident clipmap boundary: past it there's no resident geometry to occlude,
    // so a sun-lit ray would otherwise crawl brick-by-brick past the volume (the empty-space skip
    // only jumps chunks INSIDE a ring) and burn its whole 96-step / 256-unit budget for nothing.
    // Exact, not approximate — nothing beyond the volume can cast a shadow.
    let reach = min(max_t, clipmap_exit_t(origin, light_dir));
    if (k <= 0.0) {
        // Hard shadow: a binary occlusion test through the primary march. lod_floor 0 (the shared
        // empty-space accelerator makes the coarse shadow-LOD floor unnecessary for perf).
        let q = MarchQuality(1.0, SHADOW_MAX_STEPS, reach, 0u);
        return select(1.0, 0.0, raymarch(origin, light_dir, vs * 0.5, q).hit);
    }
    return soft_shadow(origin, light_dir, vs * 0.5, reach, k);
}

// Soft shadow toward a SPHERE light of `radius` centred at distance `dist` along `light_dir` — for
// point lights (`PointLight.radius` is the source size). The size drives two things, both physical:
//   * A surface INSIDE the light volume (`dist <= radius`) is fully lit — so a light hovering close
//     above its host geometry doesn't self-shadow against it.
//   * The sphere subtends a cone of half-angle ≈ radius/dist, so the penumbra hardness is
//     `k = dist / radius` (bigger / closer light ⇒ softer edge). `radius = 0` ⇒ a sharp point.
// The ray marches the FULL distance to the light (`max_t = dist`); the loop breaks at `t >= max_t`
// before ever sampling the light's own position, so there's no false self-hit. (Earlier bounds were
// wrong: `dist - radius` clipped occluders sitting near the light, and `dist - vs` collapsed the ray
// to nothing at coarse LOD — where `vs` is large — killing shadows at overview distance.)
fn sphere_light_shadow(hit_pos: vec3<f32>, geo_n: vec3<f32>, light_dir: vec3<f32>, lod: u32, dist: f32, radius: f32) -> f32 {
    if (dist <= radius) { return 1.0; }            // surface within the light volume → unshadowed
    let vs = voxel_size_at(lod);
    let origin = hit_pos + geo_n * vs;
    let k = dist / max(radius, vs);                // sphere angular size → penumbra hardness
    return soft_shadow(origin, light_dir, vs * 0.5, dist, k);
}
