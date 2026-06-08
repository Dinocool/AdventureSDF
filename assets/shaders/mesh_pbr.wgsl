// Triplanar PBR fragment for baked Transvoxel meshes (no UVs). An ExtendedMaterial<StandardMaterial, _>:
// sample diffuse + normal by projecting the world position on the 3 axis planes (blended by the surface
// normal), write them into the StandardMaterial PbrInput, then run Bevy's PBR lighting. Ported from the
// retired raymarch `sdf::material` shader. FULL triplanar (3 taps, no branches) so auto-mip derivatives stay
// valid in mesh fragments.

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    forward_io::{VertexOutput, FragmentOutput},
}

struct MeshExtParams {
    world_scale: f32,
    has_diffuse: u32,
    has_normal: u32,
    _pad: u32,
};

// `#{MATERIAL_BIND_GROUP}` is the material bind-group index (NOT always 2 — varies by pipeline); hardcoding
// `@group(2)` makes the binding land in the wrong group → "binding 100 missing from pipeline layout".
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> ext: MeshExtParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var diffuse_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var diffuse_s: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var normal_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var normal_s: sampler;

// Triplanar blend weights: emphasise the dominant axis (so a +X-facing surface samples mostly the YZ plane).
fn triplanar_weights(n: vec3<f32>) -> vec3<f32> {
    var w = pow(abs(n), vec3<f32>(4.0));
    return w / max(w.x + w.y + w.z, 1e-5);
}

// Accumulate all 3 axis projections (no branch → screen-space derivatives valid → auto mipmapping).
fn sample_triplanar(t: texture_2d<f32>, s: sampler, wpos: vec3<f32>, w: vec3<f32>, scale: f32) -> vec4<f32> {
    return textureSample(t, s, wpos.zy * scale) * w.x
        + textureSample(t, s, wpos.xz * scale) * w.y
        + textureSample(t, s, wpos.xy * scale) * w.z;
}

// Triplanar normal mapping (Ben Golus "whiteout" blend): perturb the world axes by each plane's tangent xy,
// flipped by the geometric normal's sign so concavity matches. No per-plane TBN needed.
fn triplanar_normal(wpos: vec3<f32>, n: vec3<f32>, w: vec3<f32>, scale: f32) -> vec3<f32> {
    let sn = sign(n);
    let tnx = textureSample(normal_tex, normal_s, wpos.zy * scale).xyz * 2.0 - 1.0;
    let tny = textureSample(normal_tex, normal_s, wpos.xz * scale).xyz * 2.0 - 1.0;
    let tnz = textureSample(normal_tex, normal_s, wpos.xy * scale).xyz * 2.0 - 1.0;
    let nx = vec3<f32>(n.x + tnx.z * sn.x, n.y + tnx.y, n.z + tnx.x);
    let ny = vec3<f32>(n.x + tny.x, n.y + tny.z * sn.y, n.z + tny.y);
    let nz = vec3<f32>(n.x + tnz.x, n.y + tnz.y, n.z + tnz.z * sn.z);
    return normalize(nx * w.x + ny * w.y + nz * w.z);
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    let wpos = in.world_position.xyz;
    let n = normalize(in.world_normal);
    let w = triplanar_weights(n);

    if (ext.has_diffuse != 0u) {
        let d = sample_triplanar(diffuse_tex, diffuse_s, wpos, w, ext.world_scale);
        pbr_input.material.base_color = vec4<f32>(
            pbr_input.material.base_color.rgb * d.rgb,
            pbr_input.material.base_color.a,
        );
    }
    if (ext.has_normal != 0u) {
        pbr_input.N = triplanar_normal(wpos, n, w, ext.world_scale);
    }

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
