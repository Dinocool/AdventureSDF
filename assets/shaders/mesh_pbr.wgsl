// Per-vertex multi-material triplanar PBR for baked Transvoxel meshes (no UVs). An
// ExtendedMaterial<StandardMaterial, _>: each vertex carries its top-2 material ids in UV_0 and a blend weight
// in the COLOUR alpha; sample BOTH materials' diffuse/normal/MRA from shared texture ARRAYS triplanar, cross-
// fade, write into the StandardMaterial PbrInput, then run Bevy PBR lighting. Ported from the retired raymarch
// sdf::material/pbr shaders. FULL triplanar (3 taps, no branch) keeps auto-mip derivatives valid.

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    forward_io::{VertexOutput, FragmentOutput},
}

struct MeshMat {
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    layer: u32,
    has_diffuse: u32,
    has_normal: u32,
    has_mra: u32,
    metallic: f32,
    roughness: f32,
    texture_scale: f32,
    blend_softness: f32,
};

struct BlendParams {
    debug_lod: u32,
    debug_normals: u32,
    _pad: vec2<u32>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> params: BlendParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var diffuse_arr: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var arr_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var normal_arr: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var mra_arr: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(105) var<storage, read> table: array<MeshMat>;

fn material_at(id: u32) -> MeshMat {
    let n = arrayLength(&table);
    if (n == 0u) {
        var m: MeshMat;
        m.base_color = vec4<f32>(1.0);
        m.roughness = 1.0;
        return m;
    }
    return table[min(id, n - 1u)];
}

// Triplanar blend weights: emphasise the dominant axis (a +X-facing surface samples mostly the YZ plane).
fn tri_weights(n: vec3<f32>) -> vec3<f32> {
    var w = pow(abs(n), vec3<f32>(4.0));
    return w / max(w.x + w.y + w.z, 1e-5);
}

// Full triplanar sample of one array layer (3 taps, no branch → derivatives valid → auto mipmapping).
fn tri_arr(t: texture_2d_array<f32>, wpos: vec3<f32>, w: vec3<f32>, layer: u32, scale: f32) -> vec4<f32> {
    let li = i32(layer);
    return textureSample(t, arr_sampler, wpos.zy * scale, li) * w.x
        + textureSample(t, arr_sampler, wpos.xz * scale, li) * w.y
        + textureSample(t, arr_sampler, wpos.xy * scale, li) * w.z;
}

// Whiteout-blend triplanar normal (Ben Golus): perturb world axes by each plane's tangent xy, flipped by the
// geometric normal's sign. No mesh tangents needed.
fn tri_normal(wpos: vec3<f32>, n: vec3<f32>, w: vec3<f32>, layer: u32, scale: f32) -> vec3<f32> {
    let sn = sign(n);
    let tx = tri_tap(normal_arr, wpos.zy * scale, layer);
    let ty = tri_tap(normal_arr, wpos.xz * scale, layer);
    let tz = tri_tap(normal_arr, wpos.xy * scale, layer);
    let nx = vec3<f32>(n.x + tx.z * sn.x, n.y + tx.y, n.z + tx.x);
    let ny = vec3<f32>(n.x + ty.x, n.y + ty.z * sn.y, n.z + ty.y);
    let nz = vec3<f32>(n.x + tz.x, n.y + tz.y, n.z + tz.z * sn.z);
    return normalize(nx * w.x + ny * w.y + nz * w.z);
}

fn tri_tap(t: texture_2d_array<f32>, uv: vec2<f32>, layer: u32) -> vec3<f32> {
    return textureSample(t, arr_sampler, uv, i32(layer)).xyz * 2.0 - 1.0;
}

struct Surf {
    albedo: vec3<f32>,
    normal: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32,
    emissive: vec3<f32>,
};

// Gather one material's surface at the world point (triplanar textures where present, else the scalars).
fn gather(id: u32, wpos: vec3<f32>, n: vec3<f32>, w: vec3<f32>) -> Surf {
    let m = material_at(id);
    var s: Surf;
    s.albedo = m.base_color.rgb;
    if (m.has_diffuse != 0u) {
        s.albedo *= tri_arr(diffuse_arr, wpos, w, m.layer, m.texture_scale).rgb;
    }
    s.metallic = m.metallic;
    s.roughness = m.roughness;
    s.ao = 1.0;
    if (m.has_mra != 0u) {
        let mra = tri_arr(mra_arr, wpos, w, m.layer, m.texture_scale);
        s.metallic = mra.r;
        s.roughness = mra.g;
        s.ao = mra.b;
    }
    if (m.has_normal != 0u) {
        s.normal = tri_normal(wpos, n, w, m.layer, m.texture_scale);
    } else {
        s.normal = n;
    }
    s.emissive = m.emissive.rgb;
    return s;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // Per-vertex: UV_0 = (matA, matB) ids (rounded — interpolation is constant within a single-material
    // triangle), COLOUR.a = blend weight (1 = pure A). Debug "Colour by LOD" = unlit vertex-colour tint.
    if (params.debug_lod != 0u) {
        var out: FragmentOutput;
        out.color = vec4<f32>(in.color.rgb, 1.0);
        return out;
    }
    // "View normals" debug: the mesh world-normal as RGB (unlit), for inspecting the baked geometry normals.
    if (params.debug_normals != 0u) {
        var out: FragmentOutput;
        out.color = vec4<f32>(normalize(in.world_normal) * 0.5 + 0.5, 1.0);
        return out;
    }

    let wpos = in.world_position.xyz;
    let n = normalize(in.world_normal);
    let w = tri_weights(n);
    let mat_a = u32(round(in.uv.x));
    let mat_b = u32(round(in.uv.y));

    // Material-seam cross-fade (ported from the retired raymarch `resolve_surface`): COLOUR.a carries the
    // per-vertex SIGNED WORLD-DISTANCE to the seam against this triangle's fixed pair (baked: the raw gap
    // `d(mat_b)-d(mat_a)` divided by |∇gap|, so it's a true world distance — a geometry quantity, so
    // `blend_softness` stays a LIVE control with no re-bake). The seam is at distance 0; > 0 is A's side,
    // < 0 is B's. `blend_softness` is then a real world half-width (in the same units as the scene).
    //
    // `blend_softness` is DIRECTIONAL: a material's softness is how far IT spreads into the OTHER's region. So
    // on B's side (gap < 0) the band is A's softness (A bleeding into B); on A's side (gap > 0) it's B's. If
    // one side's softness is 0 that material doesn't bleed in (hard cut, modulo a 1px `fwidth` AA floor); if
    // both are set their magnitudes set the split. `fwidth` is safe — `fragment` is uniform control flow.
    let gap = in.color.a;
    let soft = select(material_at(mat_b).blend_softness, material_at(mat_a).blend_softness, gap < 0.0);
    let band = max(max(fwidth(gap), soft), 1e-5);
    let weight = clamp(0.5 + 0.5 * gap / band, 0.0, 1.0); // 1 = pure A, 0 = pure B, 0.5 = seam

    var s = gather(mat_a, wpos, n, w);
    if (weight < 0.999 && mat_b != mat_a) {
        let sb = gather(mat_b, wpos, n, w);
        s.albedo = mix(sb.albedo, s.albedo, weight);
        s.normal = normalize(mix(sb.normal, s.normal, weight));
        s.metallic = mix(sb.metallic, s.metallic, weight);
        s.roughness = mix(sb.roughness, s.roughness, weight);
        s.ao = mix(sb.ao, s.ao, weight);
        s.emissive = mix(sb.emissive, s.emissive, weight);
    }

    pbr_input.material.base_color = vec4<f32>(s.albedo, 1.0);
    pbr_input.material.metallic = s.metallic;
    pbr_input.material.perceptual_roughness = max(s.roughness, 0.045);
    pbr_input.material.emissive = vec4<f32>(s.emissive, 1.0);
    pbr_input.N = s.normal;
    pbr_input.diffuse_occlusion = vec3<f32>(s.ao);

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
