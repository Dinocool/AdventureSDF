#define_import_path sdf::material

// Material-table access + triplanar texture sampling. SDF surfaces have no UVs, so
// each PBR map is projected three times (one per world axis plane) and blended by
// the normal's weights. `sample_material_map` is the SINGLE point of texture access
// — a future virtual-texturing swap only changes this function.

#import sdf::bindings::{
    SdfMaterial,
    materials,
    PALETTE_EMPTY,
    TEXTURE_WORLD_SCALE,
    pbr_sampler,
    tex_diffuse,
    tex_normal,
    tex_mra,
    tex_height,
    tex_edge,
}

// Safe material-table lookup: PALETTE_EMPTY (an unused palette slot) maps to global
// id 0 (the registry's default fallback) so we never index out of bounds.
fn material_at(id: u32) -> SdfMaterial {
    if (id == PALETTE_EMPTY || id >= arrayLength(&materials)) {
        return materials[0];
    }
    return materials[id];
}

// Triplanar blend weights from a world normal: emphasise the dominant axis, so a
// surface facing +X samples mostly the YZ plane. `pow` sharpens the transition.
fn triplanar_weights(n: vec3<f32>) -> vec3<f32> {
    var w = pow(abs(n), vec3<f32>(4.0));
    return w / max(w.x + w.y + w.z, 1e-5);
}

// BIPLANAR weights: drop the weakest of the three projection planes and renormalise the
// other two. A surface only ever sees two planes meaningfully (the third's weight is the
// smallest), so zeroing it costs one texture tap per map fewer with no visible change —
// the dropped plane contributed <~6% even at a 45° corner. The zeroed component lets the
// samplers below SKIP that axis (a branch on `w.* > 0`), turning 3 taps into 2.
fn biplanar_weights(n: vec3<f32>) -> vec3<f32> {
    var w = pow(abs(n), vec3<f32>(4.0));
    let mn = min(w.x, min(w.y, w.z));
    // Zero exactly one (the smallest) plane. Ties are fine — dropping either is symmetric.
    if (w.x <= mn) { w.x = 0.0; } else if (w.y <= mn) { w.y = 0.0; } else { w.z = 0.0; }
    return w / max(w.x + w.y + w.z, 1e-5);
}

// Biplanar sample of one texture array: accumulate the two non-zero-weight planes only.
// `textureSampleLevel` (explicit LOD) carries no derivative requirement, so the per-axis
// branch is legal even inside the seam-gated material path. `w` comes from
// `biplanar_weights`, so exactly one component is 0 and that tap is skipped.
fn sample_biplanar(
    t: texture_2d_array<f32>, s: sampler,
    wpos: vec3<f32>, w: vec3<f32>, li: i32, lod: f32,
) -> vec4<f32> {
    var acc = vec4<f32>(0.0);
    if (w.x > 0.0) { acc += textureSampleLevel(t, s, wpos.zy * TEXTURE_WORLD_SCALE, li, lod) * w.x; }
    if (w.y > 0.0) { acc += textureSampleLevel(t, s, wpos.xz * TEXTURE_WORLD_SCALE, li, lod) * w.y; }
    if (w.z > 0.0) { acc += textureSampleLevel(t, s, wpos.xy * TEXTURE_WORLD_SCALE, li, lod) * w.z; }
    return acc;
}

// Height-map relief is applied at BAKE TIME (folded into the SDF field; see
// sdf_render::height), not in the shader — so there are no relief helpers here. The shader
// just marches the already-displaced field.

// Sample one PBR map for material `id` via triplanar projection at `lod`. The
// `map` selector picks the array; an absent layer (tex == 0xffffffff) returns a
// neutral default so unconfigured materials still shade. The map enum mirrors
// `MapArray` (render side): 0 diffuse, 1 normal, 2 mra, 3 height, 4 edge.
fn sample_material_map(id: u32, map: u32, wpos: vec3<f32>, n: vec3<f32>, lod: f32) -> vec4<f32> {
    let mat = material_at(id);
    var layer: u32 = 0xffffffffu;
    switch (map) {
        case 0u: { layer = mat.tex_diffuse; }
        case 1u: { layer = mat.tex_normal; }
        case 2u: { layer = mat.tex_mra; }
        case 3u: { layer = mat.tex_height; }
        default: { layer = mat.tex_edge; }
    }
    if (layer == 0xffffffffu) {
        // Sensible neutral per map: white diffuse/edge, flat normal, mid MRA.
        if (map == 1u) { return vec4<f32>(0.5, 0.5, 1.0, 1.0); }   // flat normal
        if (map == 2u) { return vec4<f32>(0.0, 1.0, 1.0, 1.0); }   // metal 0, rough 1, ao 1
        return vec4<f32>(1.0);
    }

    let w = biplanar_weights(n);
    let li = i32(layer);

    switch (map) {
        case 0u: { return sample_biplanar(tex_diffuse, pbr_sampler, wpos, w, li, lod); }
        case 1u: { return sample_biplanar(tex_normal, pbr_sampler, wpos, w, li, lod); }
        case 2u: { return sample_biplanar(tex_mra, pbr_sampler, wpos, w, li, lod); }
        case 3u: { return sample_biplanar(tex_height, pbr_sampler, wpos, w, li, lod); }
        default: { return sample_biplanar(tex_edge, pbr_sampler, wpos, w, li, lod); }
    }
}

// --- Triplanar normal mapping (whiteout blend) ---
//
// Reorient each plane's tangent-space normal into world space and blend by the
// triplanar weights. Uses the "whiteout" trick (Ben Golus): perturb the relevant
// world axes by the tangent xy, keep the geometric normal as the base, so the
// result follows the surface without a per-plane TBN.
fn triplanar_normal(id: u32, wpos: vec3<f32>, n: vec3<f32>, lod: f32) -> vec3<f32> {
    let mat = material_at(id);
    if (mat.tex_normal == 0xffffffffu) {
        return n;
    }
    let scale = TEXTURE_WORLD_SCALE;
    let li = i32(mat.tex_normal);
    let w = biplanar_weights(n);
    let sn = sign(n);

    // Whiteout blend (Ben Golus): add each plane's tangent xy onto the world normal's
    // other two axes, flipping by the geometric normal's sign so concavity matches. Only
    // the two non-zero-weight planes are sampled (biplanar) — the dropped axis's tap and
    // its perturbation are skipped together.
    var acc = vec3<f32>(0.0);
    if (w.x > 0.0) {
        let tnx = textureSampleLevel(tex_normal, pbr_sampler, wpos.zy * scale, li, lod).xyz * 2.0 - 1.0;
        acc += vec3<f32>(n.x + tnx.z * sn.x, n.y + tnx.y, n.z + tnx.x) * w.x;
    }
    if (w.y > 0.0) {
        let tny = textureSampleLevel(tex_normal, pbr_sampler, wpos.xz * scale, li, lod).xyz * 2.0 - 1.0;
        acc += vec3<f32>(n.x + tny.x, n.y + tny.z * sn.y, n.z + tny.y) * w.y;
    }
    if (w.z > 0.0) {
        let tnz = textureSampleLevel(tex_normal, pbr_sampler, wpos.xy * scale, li, lod).xyz * 2.0 - 1.0;
        acc += vec3<f32>(n.x + tnz.x, n.y + tnz.y, n.z + tnz.z * sn.z) * w.z;
    }
    return normalize(acc);
}
