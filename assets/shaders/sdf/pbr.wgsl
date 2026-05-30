#define_import_path sdf::pbr

// Cook-Torrance PBR: GGX distribution, Smith geometry, Schlick fresnel, plus the
// per-material PBR-input gather (triplanar diffuse/normal/MRA + edge wear) and the
// material-seam cross-fade. `resolve_surface` returns the cross-faded PBR inputs;
// `shade_surface` shades them to LINEAR radiance (the entry shader tonemaps once), taking
// the environment radiance (sky or a traced reflection) for the specular/IBL term.

#import sdf::bindings::{camera, PI}
#import sdf::brick::SceneSdfResult
#import sdf::material::{material_at, sample_material_map, triplanar_normal}
#import sdf::shadows::surface_shadow
#import sdf::sky::sky_ambient

fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, rough: f32) -> f32 {
    let a = rough * rough;
    let a2 = a * a;
    let ndh = max(dot(n, h), 0.0);
    let d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-6);
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, rough: f32) -> f32 {
    // Schlick-GGX with the direct-lighting k = (r+1)²/8.
    let k = (rough + 1.0) * (rough + 1.0) / 8.0;
    let ndv = max(dot(n, v), 0.0);
    let ndl = max(dot(n, l), 0.0);
    let gv = ndv / (ndv * (1.0 - k) + k);
    let gl = ndl / (ndl * (1.0 - k) + k);
    return gv * gl;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Roughness-aware Fresnel (Sébastien Lagarde) for environment/IBL terms: rough surfaces
// keep a higher grazing reflectance than the sharp Schlick gives, so ambient specular on
// a rough metal doesn't darken at glancing angles.
fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let inv_rough = vec3<f32>(1.0 - roughness);
    return f0 + (max(inv_rough, f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Single source for the scene's key light. Hardcoded for now — the deferred lighting
// pass rewrites ONLY these two helpers to read a light uniform, leaving the shadow /
// reflection / ambient code (which all route through them) untouched.
fn sun_dir() -> vec3<f32> {
    return normalize(vec3<f32>(0.5, 1.0, 0.3));
}
fn sun_color() -> vec3<f32> {
    return vec3<f32>(3.0);
}

// Evaluate Cook-Torrance for one material's resolved PBR inputs.
fn shade_pbr(
    albedo: vec3<f32>,
    n: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    light_color: vec3<f32>,
    shadow: f32,
) -> vec3<f32> {
    let h = normalize(view_dir + light_dir);
    let ndl = max(dot(n, light_dir), 0.0);

    // Dielectric base reflectance 0.04; metals take their albedo as F0.
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    let ndf = distribution_ggx(n, h, roughness);
    let g = geometry_smith(n, view_dir, light_dir, roughness);
    let f = fresnel_schlick(max(dot(h, view_dir), 0.0), f0);

    let ndv = max(dot(n, view_dir), 0.0);
    let specular = (ndf * g * f) / max(4.0 * ndv * ndl, 1e-4);

    // Energy conservation: diffuse only from the non-reflected, non-metal fraction.
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * albedo / PI;

    // Direct lighting only — ambient/environment is added once in `shade_material` via
    // `ambient_ibl` (it's view/normal dependent, not per-light).
    return (diffuse + specular) * light_color * ndl * shadow;
}

// Environment ambient (image-based-lighting approximation): a diffuse hemisphere-
// irradiance term plus a specular reflection along the view-reflected normal. The mirror
// radiance comes IN as `env_radiance` — the caller passes the analytic sky (Stage 3) or a
// secondary SDF-traced reflection of nearby geometry (Stage 4). This is what makes metals
// read as metal — a pure-metal surface has no diffuse and would be near-black otherwise.
fn ambient_ibl(
    albedo: vec3<f32>,
    n: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32,
    view_dir: vec3<f32>,
    sun: vec3<f32>,
    env_radiance: vec3<f32>,
) -> vec3<f32> {
    let ndv = max(dot(n, view_dir), 0.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let f = fresnel_schlick_roughness(ndv, f0, roughness);

    // Diffuse: hemisphere irradiance, only the non-metal / non-reflected fraction.
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let irradiance = sky_ambient(n, sun);
    let diffuse = kd * albedo * irradiance;

    // Specular: the mirror radiance along the reflected view ray. Rougher surfaces blur
    // toward the diffuse irradiance (a cheap stand-in for a prefiltered mip chain).
    let env = mix(env_radiance, irradiance, roughness);
    let specular = env * f;

    return (diffuse + specular) * ao;
}

// Per-material resolved PBR inputs at a point (post-triplanar). Cross-faded across
// a seam in `shade_material`.
struct PbrInputs {
    albedo: vec3<f32>,
    normal: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32,
}

fn gather_pbr(id: u32, wpos: vec3<f32>, geo_n: vec3<f32>, lod: f32) -> PbrInputs {
    let mat = material_at(id);
    // Note: relief DISPLACEMENT (the height map) happens at the hit in the entry shader
    // (`relief_displace`), which moves `wpos` itself before this is called — so the maps
    // here already sample the carved surface position. No UV-shift parallax here.
    let albedo = sample_material_map(id, 0u, wpos, geo_n, lod).rgb * mat.base_color.rgb;
    let edge = sample_material_map(id, 4u, wpos, geo_n, lod).r;
    let nrm = triplanar_normal(id, wpos, geo_n, lod);

    // Metallic / roughness / AO: from the MRA texture when present, else the material's
    // scalar fallbacks (AO = 1). This lets a textureless material be a plain metal or
    // dielectric — e.g. a deep-red metallic exemplar with no map set.
    var metal: f32;
    var rough: f32;
    var ao: f32;
    if (mat.tex_mra == 0xffffffffu) {
        metal = mat.metallic;
        rough = mat.roughness;
        ao = 1.0;
    } else {
        let mra = sample_material_map(id, 2u, wpos, geo_n, lod).rgb; // r=metal g=rough b=ao
        metal = mra.r;
        rough = mra.g;
        ao = mra.b;
    }

    // Edge-wear: convex edges (bright in the edge map) read as worn — lighter and
    // rougher, a cheap stand-in for exposed/scuffed material until it's art-driven.
    let wear = smoothstep(0.6, 1.0, edge);
    let albedo_worn = mix(albedo, albedo * 1.3 + vec3<f32>(0.05), wear * 0.5);
    let rough_worn = clamp(rough + wear * 0.3, 0.04, 1.0);

    return PbrInputs(albedo_worn, nrm, metal, rough_worn, ao);
}

// Resolve the cross-faded PBR inputs at a hit: the two nearest materials' fully-resolved
// inputs lerped across the seam. Split out from shading so the entry shader can read the
// surface (its reflected normal, metallic/roughness) to decide whether to trace a
// reflection ray, then shade once with the resulting environment radiance.
//
// The seam lives where the two nearest materials are equidistant (gap == 0). The cross-
// fade half-width is the larger of fwidth(gap) (≥1px, anti-aliased) and the pair's
// blend_softness (world units; the artist control). Safe to call fwidth: `main` is uniform
// control flow. Fully sampling both materials is gated to the seam band to save taps.
fn resolve_surface(res: SceneSdfResult, wpos: vec3<f32>, geo_n: vec3<f32>, lod: f32) -> PbrInputs {
    let mat_a = material_at(res.object_id);
    let mat_b = material_at(res.object_id_b);
    let soft = max(mat_a.blend_softness, mat_b.blend_softness);
    let band = max(max(fwidth(res.gap), soft), 1e-5);
    let w = clamp(0.5 + 0.5 * res.gap / band, 0.5, 1.0);  // 1 = pure A, 0.5 = seam

    var p = gather_pbr(res.object_id, wpos, geo_n, lod);
    // Only sample the second material near the seam (w < 1), where it contributes.
    if (w < 0.999 && res.object_id_b != res.object_id) {
        let pb = gather_pbr(res.object_id_b, wpos, geo_n, lod);
        p.albedo = mix(pb.albedo, p.albedo, w);
        p.normal = normalize(mix(pb.normal, p.normal, w));
        p.metallic = mix(pb.metallic, p.metallic, w);
        p.roughness = mix(pb.roughness, p.roughness, w);
        p.ao = mix(pb.ao, p.ao, w);
    }
    return p;
}

// Shade a resolved surface to LINEAR radiance (the caller tonemaps once): Cook-Torrance
// direct lighting + environment ambient. `env_radiance` is the mirror reflection colour —
// the analytic sky (Stage 3) or a secondary SDF-traced reflection (Stage 4). `geo_n` is
// the geometric normal (not the normal-mapped `p.normal`) so the shadow bias tracks the
// real surface.
fn shade_surface(
    p: PbrInputs,
    wpos: vec3<f32>,
    geo_n: vec3<f32>,
    res_lod: u32,
    env_radiance: vec3<f32>,
) -> vec3<f32> {
    let view_dir = normalize(camera.camera_pos.xyz - wpos);
    let light_dir = sun_dir();
    let light_color = sun_color();

    var shadow = 1.0;
#ifdef SDF_SHADOWS
    shadow = surface_shadow(wpos, geo_n, light_dir, res_lod, 256.0);
#endif

    let direct = shade_pbr(
        p.albedo, p.normal, p.metallic, p.roughness, p.ao,
        view_dir, light_dir, light_color, shadow,
    );
    let ambient = ambient_ibl(
        p.albedo, p.normal, p.metallic, p.roughness, p.ao, view_dir, light_dir, env_radiance,
    );
    return direct + ambient;
}

// Convenience: resolve + shade with the analytic sky as the environment. Used for the
// reflection ray's own hit (one bounce — no further reflection) and any caller that
// doesn't trace reflections. Returns LINEAR radiance.
fn shade_material_env(
    res: SceneSdfResult,
    wpos: vec3<f32>,
    geo_n: vec3<f32>,
    lod: f32,
    env_radiance: vec3<f32>,
) -> vec3<f32> {
    let p = resolve_surface(res, wpos, geo_n, lod);
    return shade_surface(p, wpos, geo_n, res.lod, env_radiance);
}
