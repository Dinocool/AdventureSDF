// SDF G-buffer Shader — primary entry point (Bevy 0.18).
//
// Composed from the `sdf::*` modules under shaders/sdf/ via naga_oil #import. The unified
// raymarch lives in the shared `sdf::march` module; this file owns the cone-seeded primary ray
// + the fragment `main` that exports the deferred G-buffer.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

#import sdf::bindings::{camera, max_steps, max_dist, pixel_cone, voxel_size_at, TEXTURE_WORLD_SCALE}
#import sdf::march::{raymarch, MarchQuality, RaymarchResult}
#import sdf::brick::{scene_sdf, calc_normal}
#import sdf::pbr::{resolve_surface, sun_dir, PbrInputs}
#import sdf::material::material_at
#import sdf::oct::oct_encode
#import sdf::sky::sky_color
#import sdf::shadows::surface_shadow

// Cone-prepass seed texture: per-8×8-tile start distance (R32Float), written by
// sdf_cone_prepass.wgsl. The march starts each pixel at its tile's seed-t instead of 0,
// amortising the empty-corridor march across the tile. The seed is a guaranteed lower
// bound on every pixel's hit distance (the cone stops before any surface enters the tile),
// so starting from it never skips geometry. Group 2 — groups 0/1 are camera + atlas.
@group(2) @binding(0) var cone_seed: texture_2d<f32>;
const CONE_TILE: i32 = 8;

// Deferred G-buffer. The primary SDF march no longer shades — it exports surface attributes into
// three MRT targets that the deferred lit pass (and, later, the world-space GI probe pass)
// consume. Dedicated emissive channel (location 2) keeps emitted radiance independent of albedo,
// so a surface can be both lit and glowing.
struct FragmentOutput {
    // rgb = albedo (linear); a = camera distance to the hit (>= SKY_DIST sentinel = sky/miss).
    // The distance doubles as a coverage/validity bit AND lets the lit pass reconstruct the
    // world position (cam + ray_dir * dist) without sampling the depth texture.
    @location(0) albedo: vec4<f32>,
    // rg = octEncode(world shading normal); b = metallic; a = roughness.
    @location(1) normal_mat: vec4<f32>,
    // rgb = emissive radiance (premultiplied, linear); a spare. Zero for non-emissive surfaces.
    @location(2) emissive: vec4<f32>,
    // True reverse-Z projection depth so the SDF surface still shares the hardware depth buffer
    // with other geometry (wireframe, gizmos, transparent pass).
    @builtin(frag_depth) depth: f32,
};

// Camera-distance sentinel written into albedo.a for a sky/miss pixel. Large enough that the
// lit pass's `a >= SKY_DIST` test never trips on a real surface.
const SKY_DIST: f32 = 1e8;

// Debug ramp: map the CONTINUOUS rendered LOD to a hue sweep (red = LOD 0 → blue/violet at the
// coarsest), so a LOD cross-fade reads as a smooth gradient between two hues — the LOD-blend view.
fn lod_ramp(eff_lod: f32) -> vec3<f32> {
    let h = clamp(eff_lod * 0.16, 0.0, 0.83); // ~one hue step per LOD, no wrap-around
    let r = clamp(abs(h * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    let g = clamp(2.0 - abs(h * 6.0 - 2.0), 0.0, 1.0);
    let b = clamp(2.0 - abs(h * 6.0 - 4.0), 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

// --- Fragment shader (G-buffer export) ---

@fragment
fn main(in: FullscreenVertexOutput) -> FragmentOutput {
    let uv = in.uv;
    // Bevy/wgpu clip space is z in [0,1] with reverse-Z (near plane = 1.0). Reconstruct the
    // ray via the near-plane point — always finite, unlike the far plane which sits at infinity
    // for Bevy's infinite reverse-Z projection.
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    let world_near = camera.inv_view_proj * ndc;
    let world_pos = world_near.xyz / world_near.w;
    let ray_dir = normalize(world_pos - camera.camera_pos.xyz);
    let ray_origin = camera.camera_pos.xyz;

    // Seed the march from the cone prepass: the per-tile start distance for this pixel's 8×8
    // tile (a guaranteed lower bound on its hit distance, so no geometry is skipped).
    let tile = vec2<i32>(uv * camera.screen_params.xy) / CONE_TILE;
    let start_t = textureLoad(cone_seed, tile, 0).r;

    // Primary ray: full quality (cone ×1, the uniform step/dist caps, no LOD floor).
    let rm = raymarch(ray_origin, ray_dir, start_t, MarchQuality(1.0, max_steps(), max_dist(), 0u));

#ifdef SDF_DEBUG_STEP_COUNT
    // Step-count heatmap: blue (few) → red (at the budget). Colours EVERY pixel — hit, sky-miss,
    // AND step-capped — by march cost, BEFORE the miss/sky branch, so a grazing crest that
    // exhausts `max_steps` glows red instead of vanishing into the sky (which is exactly the case
    // we want to see). Written as albedo (a < SKY_DIST on a hit so it shares depth; = SKY_DIST on a
    // miss so the lit pass treats it as a passthrough); the lit pass returns albedo straight through.
    let heat = clamp(f32(rm.steps) / f32(max_steps()), 0.0, 1.0);
    let heat_rgb = vec3<f32>(heat, 0.3 * (1.0 - heat), 1.0 - heat);
    var heat_depth = 0.0; // reverse-Z far for a miss
    if (rm.hit) {
        let hc = camera.clip_from_world * vec4<f32>(rm.hit_pos, 1.0);
        heat_depth = hc.z / hc.w;
    }
    return FragmentOutput(
        vec4<f32>(heat_rgb, select(SKY_DIST, rm.dist, rm.hit)),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        heat_depth,
    );
#endif

    if (!rm.hit) {
        // Sky/miss: store the analytic sky as "albedo" (the lit pass passes it straight
        // through), distance = sentinel, no normal, no emission, depth = far (reverse-Z 0).
        let sky = sky_color(ray_dir, sun_dir());
        return FragmentOutput(
            vec4<f32>(sky, SKY_DIST),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            0.0,
        );
    }

    // Height-map relief is baked into the SDF field (see sdf_render::height) — the hit position
    // and its gradient normal already reflect the carved surface.
    let hit_pos = rm.hit_pos;
    let geo_normal = calc_normal(rm.hit_pos);

    // Analytic texture LOD (no screen-space derivatives in a fullscreen raymarch). Pick the mip
    // whose texel covers ~1 pixel: footprint = pixel_cone · dist, stretched by 1/|cosθ| at
    // grazing angles, divided by the texture's world-per-texel, then log2 → mip.
    let cos_graze = max(abs(dot(ray_dir, geo_normal)), 0.15);  // floor caps the stretch (~6.7×)
    let footprint_world = pixel_cone() * max(rm.dist, 1.0) / cos_graze;
    let texels_per_pixel = footprint_world / TEXTURE_WORLD_SCALE;
    let lod = clamp(log2(max(texels_per_pixel, 1.0)), 0.0, 8.0);

    // True reverse-Z projection depth so the SDF surface shares the depth buffer with normal
    // geometry. Bevy clip space is z in [0,1], near = 1.
    let clip = camera.clip_from_world * vec4<f32>(hit_pos, 1.0);
    let ndc_depth = clip.z / clip.w;

#ifdef SDF_DEBUG_LOD
    // LOD-blend debug: write the eff-LOD ramp as albedo (the lit pass returns it straight through),
    // depth kept so it occludes correctly. The blend band shows as a gradient between two LOD hues.
    return FragmentOutput(
        vec4<f32>(lod_ramp(rm.eff_lod), rm.dist),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        ndc_depth,
    );
#endif

    // Resolve the cross-faded PBR inputs at the surface. `p.normal` is the normal-mapped
    // shading normal — the one the lit pass + GI want.
    let scene = scene_sdf(hit_pos);
    let p: PbrInputs = resolve_surface(scene, hit_pos, geo_normal, lod);

    // Emissive radiance from the hit material (premultiplied by intensity CPU-side). This is the
    // self-lit term the lit pass adds directly; the GI probe tracer will read the SAME table.
    let emissive = material_at(scene.object_id).emissive.rgb;

    // Sun visibility, marched HERE (the G-buffer pass has the atlas + march; the lit pass is
    // deliberately atlas-free). Stored in emissive.a so the lit pass can shadow the analytic sun
    // without re-marching. 1 = lit, 0 = shadowed. Off (1.0) when SDF_SHADOWS is disabled.
    var sun_vis = 1.0;
#ifdef SDF_SHADOWS
    sun_vis = surface_shadow(hit_pos, geo_normal, sun_dir(), rm.lod, 256.0);
#endif

    return FragmentOutput(
        vec4<f32>(p.albedo, rm.dist),
        vec4<f32>(oct_encode(p.normal), p.metallic, p.roughness),
        vec4<f32>(emissive, sun_vis),
        ndc_depth,
    );
}
