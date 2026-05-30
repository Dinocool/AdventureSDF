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

    let uv_x = wpos.zy * TEXTURE_WORLD_SCALE;  // YZ plane (normal ~ ±X)
    let uv_y = wpos.xz * TEXTURE_WORLD_SCALE;  // XZ plane (normal ~ ±Y)
    let uv_z = wpos.xy * TEXTURE_WORLD_SCALE;  // XY plane (normal ~ ±Z)
    let w = triplanar_weights(n);
    let li = i32(layer);

    var sx: vec4<f32>;
    var sy: vec4<f32>;
    var sz: vec4<f32>;
    switch (map) {
        case 0u: {
            sx = textureSampleLevel(tex_diffuse, pbr_sampler, uv_x, li, lod);
            sy = textureSampleLevel(tex_diffuse, pbr_sampler, uv_y, li, lod);
            sz = textureSampleLevel(tex_diffuse, pbr_sampler, uv_z, li, lod);
        }
        case 1u: {
            sx = textureSampleLevel(tex_normal, pbr_sampler, uv_x, li, lod);
            sy = textureSampleLevel(tex_normal, pbr_sampler, uv_y, li, lod);
            sz = textureSampleLevel(tex_normal, pbr_sampler, uv_z, li, lod);
        }
        case 2u: {
            sx = textureSampleLevel(tex_mra, pbr_sampler, uv_x, li, lod);
            sy = textureSampleLevel(tex_mra, pbr_sampler, uv_y, li, lod);
            sz = textureSampleLevel(tex_mra, pbr_sampler, uv_z, li, lod);
        }
        case 3u: {
            sx = textureSampleLevel(tex_height, pbr_sampler, uv_x, li, lod);
            sy = textureSampleLevel(tex_height, pbr_sampler, uv_y, li, lod);
            sz = textureSampleLevel(tex_height, pbr_sampler, uv_z, li, lod);
        }
        default: {
            sx = textureSampleLevel(tex_edge, pbr_sampler, uv_x, li, lod);
            sy = textureSampleLevel(tex_edge, pbr_sampler, uv_y, li, lod);
            sz = textureSampleLevel(tex_edge, pbr_sampler, uv_z, li, lod);
        }
    }
    return sx * w.x + sy * w.y + sz * w.z;
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

    // Tangent-space normals from each plane ([0,1] → [-1,1]).
    let tnx = textureSampleLevel(tex_normal, pbr_sampler, wpos.zy * scale, li, lod).xyz * 2.0 - 1.0;
    let tny = textureSampleLevel(tex_normal, pbr_sampler, wpos.xz * scale, li, lod).xyz * 2.0 - 1.0;
    let tnz = textureSampleLevel(tex_normal, pbr_sampler, wpos.xy * scale, li, lod).xyz * 2.0 - 1.0;

    // Whiteout blend: add the tangent xy onto the world normal's other two axes,
    // flipping with the sign of the geometric normal so concavity matches.
    let sn = sign(n);
    let nx = vec3<f32>(n.x + tnx.z * sn.x, n.y + tnx.y, n.z + tnx.x);
    let ny = vec3<f32>(n.x + tny.x, n.y + tny.z * sn.y, n.z + tny.y);
    let nz = vec3<f32>(n.x + tnz.x, n.y + tnz.y, n.z + tnz.z * sn.z);

    let w = triplanar_weights(n);
    return normalize(nx * w.x + ny * w.y + nz * w.z);
}
