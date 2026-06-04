#define_import_path sdf::sky

// Analytic environment: a vertical sky gradient (ground → horizon → zenith) plus a soft
// sun disk along the light direction. Single-sourced here and used three ways so they all
// agree: the ray-miss background, the diffuse ambient (hemisphere irradiance), and the
// specular ambient / reflection environment. The sun is passed in (not imported) so this
// module stays below `pbr` in the import graph.

// The bare vertical gradient (no sun) in direction `dir`: ground below, horizon at the
// equator, zenith overhead. Shared by the radiance and irradiance variants.
fn sky_gradient(dir: vec3<f32>) -> vec3<f32> {
    let up = clamp(dir.y, -1.0, 1.0);
    let zenith = vec3<f32>(0.10, 0.16, 0.38);
    let horizon = vec3<f32>(0.42, 0.52, 0.68);
    let ground = vec3<f32>(0.05, 0.05, 0.08);
    if (up >= 0.0) {
        return mix(horizon, zenith, pow(up, 0.5));
    }
    return mix(horizon, ground, pow(-up, 0.5));
}

// Approximate physical luminance scale (nits) for the analytic sky. The renderer is fully
// physical (sun in lux, point lights in candela), and the lit pass applies the camera exposure to
// the sky too — so the artistic gradient below must be lifted into physical luminance or it would
// crush to black after exposure. Tuned against `SDF_EXPOSURE_EV100` (~11.5) so a daytime sky reads
// naturally; raise/lower alongside ev100.
const SKY_LUMINANCE: f32 = 4000.0;

// Full environment radiance in direction `dir` (sun direction `sun`): the gradient plus a
// crisp sun disk and a soft halo. This is what a mirror ray sees — used as the background
// and as the specular reflection colour. Returned in physical luminance (× SKY_LUMINANCE).
fn sky_color(dir: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    // Sun disk (crisp) + glow (soft halo) where the ray aligns with the sun.
    let sd = max(dot(dir, sun), 0.0);
    let disk = smoothstep(0.9985, 0.9996, sd);
    let glow = pow(sd, 64.0) * 0.4;
    let artistic = sky_gradient(dir) + vec3<f32>(1.0, 0.95, 0.85) * (disk * 8.0 + glow);
    return artistic * SKY_LUMINANCE;
}

// Diffuse hemisphere irradiance for a surface normal `n`: the sky gradient WITHOUT the
// crisp sun disk (a Lambertian surface integrates the disk into a broad term, so adding
// the disk here would over-brighten surfaces facing the sun). A mild sun-facing tint
// stands in for the integrated sun contribution.
fn sky_ambient(n: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    let sun_tint = max(dot(n, sun), 0.0) * 0.2;
    return sky_gradient(n) + vec3<f32>(1.0, 0.95, 0.85) * sun_tint;
}
