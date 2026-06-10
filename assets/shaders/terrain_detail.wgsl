// DETAIL-NORMAL terrain fragment shader (Zylann-style detail rendering). A dedicated
// ExtendedMaterial<StandardMaterial, TerrainDetailExt> for TERRAIN-ONLY coarse-LOD chunks: a per-chunk
// Rg16Float normal-map texture stores the FINE (mip-0-scale) band-limited surface slope (dh/dx, dh/dz) at
// far higher resolution than the coarse mesh's vertices. The fragment reconstructs the hi-fi surface normal
// from the texel slope and feeds it to Bevy PBR, so a low-poly distant chunk SHADES as if it had the fine
// relief its averaged geometry lacks. Terrain is a HEIGHTFIELD (`y - h(x,z)`) so a top-down PLANAR
// projection (world XZ → UV) is exact — no triplanar/atlas/axis-selection needed.

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    forward_io::{VertexOutput, FragmentOutput},
}

struct TerrainDetailParams {
    // World-XZ minimum corner of the chunk's footprint (the detail map covers [chunk_min, chunk_min + size]).
    chunk_min: vec2<f32>,
    // World-XZ edge length of the chunk's (square) footprint, in metres.
    chunk_size: f32,
    // Detail-normal blend strength in [0, 1]: 0 = pure geometry normal, 1 = pure baked hi-fi detail normal.
    strength: f32,
    // .x = 1 for "View normals" debug (unlit, the applied world-normal as RGB), else 0 (lit PBR); .yzw pad.
    flags: vec4<u32>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> params: TerrainDetailParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var detail_normal: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var detail_sampler: sampler;

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // Top-down planar UV over the chunk's world-XZ footprint, clamped (sampler is clamp-to-edge too, so a
    // fragment at the very chunk border still reads the edge texel rather than wrapping).
    let uv = clamp((in.world_position.xz - params.chunk_min) / params.chunk_size, vec2<f32>(0.0), vec2<f32>(1.0));
    // The Rg16Float texel stores the ABSOLUTE fine surface slope (dh/dx, dh/dz). Reconstruct the hi-fi
    // surface normal of the heightfield `y - h(x,z)`: N = normalize(-dh/dx, 1, -dh/dz).
    let slope = textureSample(detail_normal, detail_sampler, uv).rg;
    let n_detail = normalize(vec3<f32>(-slope.x, 1.0, -slope.y));
    // Blend from the coarse geometry normal toward the hi-fi detail normal by `strength` (a live uniform).
    let n_geo = normalize(in.world_normal);
    let n = normalize(mix(n_geo, n_detail, params.strength));

    // "View normals" debug: the APPLIED world-normal as RGB (unlit) — mirrors mesh_pbr's debug branch so the
    // baked detail-normal can be inspected directly.
    if (params.flags.x != 0u) {
        var out: FragmentOutput;
        out.color = vec4<f32>(n * 0.5 + 0.5, 1.0);
        return out;
    }

    pbr_input.N = n;

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
