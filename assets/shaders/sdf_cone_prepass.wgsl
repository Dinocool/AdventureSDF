// SDF cone-prepass compute shader (Bevy 0.18).
//
// One invocation per TILE×TILE screen tile. Marches a single conservative *cone* down the
// tile-centre ray and writes a per-tile START DISTANCE (`seed_t`) to an R32Float storage
// texture. The full-resolution raymarch then starts each pixel at its tile's `seed_t`
// instead of 0, so the 64 pixels of a tile amortise ONE empty-corridor march instead of
// each crawling it independently.
//
// QUALITY GUARANTEE (no silhouette fattening, no missed geometry): the cone half-width at
// distance t is `tile_cone · t`, chosen to enclose the whole tile footprint (every pixel
// ray in the tile lies within `tile_cone · t` of the centre ray). The march advances only
// while the centre-ray field `d` exceeds that cone radius — i.e. while NO surface can lie
// within the tile. The instant `d - cone <= 0` a surface might enter the tile, so we stop:
// `seed_t` is a distance the per-pixel march can start from with the surface still strictly
// ahead of EVERY pixel in the tile. Empty (unbaked) space is skipped by the same chunk-DDA
// the main march uses (provably empty regardless of the cone). Result is purely a lower
// bound on each pixel's hit distance — the per-pixel march is unchanged past the seed.

#import sdf::bindings::{
    camera,
    max_steps,
    max_dist,
    pixel_cone,
    voxel_size_at,
    lod_count,
}
#import sdf::brick::{
    world_to_brick_lod,
    resolve_march,
    dist_to_brick_exit_lod,
    dist_to_chunk_exit_lod,
    in_ring_chunk,
    find_chunk_cached,
    new_chunk_cache,
}

// Output: one texel per screen tile, holding the tile's seed start-distance. R32Float so
// the main pass reads it with a plain textureLoad. group 2 — groups 0/1 are owned by
// sdf::bindings (camera + atlas), so the prepass output lives in its own group.
@group(2) @binding(0) var seed_out: texture_storage_2d<r32float, write>;

// Screen tile edge in pixels. MUST match the divisor the fragment pass uses to index this
// texture (sdf_raymarch.wgsl). 8 → 64 pixels amortise one cone march.
const TILE: u32 = 8u;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile = gid.xy;
    let screen = camera.screen_params.xy;
    // Only tiles that cover the screen need a seed (dispatch is rounded up to whole
    // workgroups, and the storage texture is sized for the max resolution).
    let tiles = vec2<u32>(ceil(screen / f32(TILE)));
    if (tile.x >= tiles.x || tile.y >= tiles.y) {
        return;
    }

    // Tile-centre ray, reconstructed exactly like the fragment pass (near-plane point of the
    // reverse-Z projection, always finite).
    let px = (vec2<f32>(tile) + 0.5) * f32(TILE);
    let uv = px / screen;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    let world_near = camera.inv_view_proj * ndc;
    let world_pos = world_near.xyz / world_near.w;
    let origin = camera.camera_pos.xyz;
    let dir = normalize(world_pos - origin);

    let MAX_STEPS = max_steps();
    let MAX_DIST = max_dist();
    // Cone half-width per unit ray distance covering the WHOLE tile: a pixel covers
    // `pixel_cone·t`; the tile spans TILE pixels, so `pixel_cone·TILE·t` conservatively
    // encloses every pixel ray in the tile (corner included). Larger cone = stop sooner =
    // can only UNDER-estimate the safe skip, never over-skip.
    let TILE_CONE = pixel_cone() * f32(TILE);

    var t = 0.0;
    var cache = new_chunk_cache();

    for (var i = 0u; i < MAX_STEPS; i = i + 1u) {
        if (t > MAX_DIST) {
            t = MAX_DIST;   // clear sky: per-pixel march will escape in ~1 step
            break;
        }
        let p = origin + dir * t;
        let scene = resolve_march(p, &cache);

        // --- Empty space: hierarchical chunk-DDA skip (mirrors sdf_raymarch.wgsl) -------
        // Provably-empty regardless of the cone, so step across the largest in-ring absent
        // chunk box around p (coarsest first). Everything skipped stays clear, so `t` after
        // the jump is still a valid seed.
        if (!scene.in_brick) {
            let wl = scene.window_lod;
            var adv = dist_to_brick_exit_lod(p, dir, wl) + voxel_size_at(wl) * 0.01;
            let levels = lod_count();
            for (var L = levels; L > 0u; ) {
                L = L - 1u;
                let coord = world_to_brick_lod(p, L);
                let ci = find_chunk_cached(coord, L, &cache);
                if (ci < 0 && in_ring_chunk(coord, L)) {
                    adv = max(adv, dist_to_chunk_exit_lod(p, dir, L) + voxel_size_at(L) * 0.01);
                    break;
                }
            }
            t += adv;
            continue;
        }

        // --- Occupied region: advance only while the cone stays clear of the surface ----
        let cone = TILE_CONE * t;
        if (scene.dist - cone <= 0.0) {
            break;   // a surface may now lie within the tile — stop; `t` is the seed
        }
        // Largest step that keeps the cone clear (sphere-trace on the cone-shrunk field),
        // floored so we never stall and capped at the brick exit so coarse LOD re-resolves.
        let voxel = voxel_size_at(scene.lod);
        let brick_exit = dist_to_brick_exit_lod(p, dir, scene.lod);
        t += clamp(scene.dist - cone, voxel * 0.01, brick_exit + voxel * 0.01);
    }

    // Back off one fine voxel so floating-point error in the cone bound can never seed the
    // per-pixel march at or past the true surface. Negligible cost; guarantees correctness.
    let seed_t = max(t - voxel_size_at(0u), 0.0);
    textureStore(seed_out, vec2<i32>(tile), vec4<f32>(seed_t, 0.0, 0.0, 0.0));
}
