#define_import_path sdf::brdf

// Frostbite BRDF (Lagarde 2015, "Moving Frostbite to Physically Based Rendering"), ported
// verbatim from three-rc's composite. Binding-free — shared by the GI gather pass (BRDF per
// cascade direction) and the combine pass (analytic sun).

const PI: f32 = 3.14159265358979;

// Schlick 1994 / Karis 2013.
fn f_schlick(u: f32, f0: vec3<f32>, f90: f32) -> vec3<f32> {
    let w = exp2(u * (-5.55473 * u - 6.98316));
    return f0 + (vec3<f32>(f90) - f0) * w;
}

// Walter 2007.
fn d_ggx(noh: f32, a: f32) -> f32 {
    let a2 = a * a;
    let f = (noh * a2 - noh) * noh + 1.0;
    return a2 / (PI * f * f);
}

// Heitz 2014 (height-correlated Smith).
fn v_smith_ggx_correlated(nov: f32, nol: f32, a: f32) -> f32 {
    let a2 = a * a;
    let ggxl = nov * sqrt((-nol * a2 + nol) * nol + a2);
    let ggxv = nol * sqrt((-nov * a2 + nov) * nov + a2);
    return 0.5 / max(ggxv + ggxl, 1e-6);
}

// Burley 2012 diffuse with Frostbite energy renormalisation.
fn fd_burley(nov: f32, nol: f32, loh: f32, roughness: f32) -> f32 {
    let energy_bias = mix(0.0, 0.5, roughness);
    let energy_factor = mix(1.0, 1.0 / 1.51, roughness);
    let f90 = energy_bias + 2.0 * loh * loh * roughness;
    let light_scatter = f_schlick(nol, vec3<f32>(1.0), f90).x;
    let view_scatter = f_schlick(nov, vec3<f32>(1.0), f90).x;
    return light_scatter * view_scatter * energy_factor * (1.0 / PI);
}

// Full Frostbite BRDF (diffuse + specular) × NoL, for one light direction `l`.
fn frostbite_brdf(
    v: vec3<f32>, n: vec3<f32>, l: vec3<f32>,
    albedo: vec3<f32>, roughness: f32, metallic: f32, f0: vec3<f32>,
) -> vec3<f32> {
    let nol = max(dot(n, l), 0.0);
    // Below the horizon the BRDF is exactly zero — return early BEFORE the half-vector. For a
    // light direction opposite the view (l ≈ -v, common among the cascade's full-sphere bins),
    // `normalize(v + l)` is normalize(~0) = NaN; `NaN * nol` stays NaN even at nol = 0, so without
    // this guard back-facing bins inject NaN → black speckles.
    if (nol <= 0.0) {
        return vec3<f32>(0.0);
    }
    let h = normalize(v + l);
    let nov = max(dot(n, v), 1e-3);
    let noh = max(dot(n, h), 0.0);
    let loh = max(dot(l, h), 0.0);
    let alpha = roughness * roughness;

    let d = d_ggx(noh, alpha);
    let vis = v_smith_ggx_correlated(nov, nol, alpha);
    let f = f_schlick(loh, f0, 1.0);
    let fr = d * vis * f;                              // specular
    let fd = albedo * fd_burley(nov, nol, loh, roughness);  // diffuse
    return ((1.0 - metallic) * fd + fr) * nol;
}
