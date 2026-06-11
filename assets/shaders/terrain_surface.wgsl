// TERRAIN-SURFACE fragment shader (Stages 2+3 of the terrain-materials feature; see
// docs/TERRAIN_MATERIALS_PLAN.md). A dedicated ExtendedMaterial<StandardMaterial, TerrainSurfaceExt> for
// TERRAIN-ONLY chunks. Per fragment it shades the surface by the biome's VOLUMETRIC strata column:
//
//   uv    = (world.xz - chunk_min) / chunk_size          // planar — terrain is a heightfield
//   depth = surface_height(uv) - world.y                 // pristine surface, NOT the carved geometry
//   biome = biome(uv)                                    // Stage-1 Whittaker classification (baked)
//   color = strata_column(biome, depth)                  // grass -> dirt -> stone -> bedrock by depth
//   if depth ~ 0 (top, undug): surface treatment (snow high+cold / rock steep / sand near sea level)
//
// + the baked hi-fi detail normal (coarse chunks) + Bevy PBR. `depth` uses the PRISTINE surface height, so
// dug pit walls read their true stratum automatically. The strata GPU table is the SHARED flatten the
// editor biome/slice preview also uploads (biome::StrataTableStd) — one SSOT, no WGSL port of the
// Whittaker/strata logic.

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    forward_io::{VertexOutput, FragmentOutput},
}

// Must match GPU_STRATA_MAX_LAYERS in src/sdf_render/worldgen/biome.rs.
const STRATA_MAX_LAYERS: u32 = 6u;
// Must match BiomeId::ALL.len() (= biome::BIOME_COUNT).
const BIOME_COUNT: u32 = 5u;
// Must match GPU_MAX_MATERIALS in src/sdf_render/worldgen/biome.rs.
const MAX_MATERIALS: u32 = 32u;

struct TerrainSurfaceParams {
    // World-XZ minimum corner of the chunk's footprint (all maps cover [chunk_min, chunk_min + size]).
    chunk_min: vec2<f32>,
    // World-XZ edge length of the chunk's (square) footprint, in metres.
    chunk_size: f32,
    // Detail-normal blend strength in [0, 1]: 0 = pure geometry normal, 1 = pure baked hi-fi detail normal.
    strength: f32,
    // .x = view-normals debug; .y = force legacy (no strata) look; .z = detail-normal present (else geometry
    // normal); .w pad.
    flags: vec4<u32>,
    // .x rock slope-start (cos), .y rock slope-full (cos), .z snow height-start (y), .w snow height-full (y).
    surf_a: vec4<f32>,
    // .x sand half-band below sea, .y sea level (y), .z treatment master strength, .w boundary blend (m).
    surf_b: vec4<f32>,
};

// One biome's flattened strata column (mirror of biome::GpuStrataColumnStd, std140-padded).
struct StrataColumn {
    surface_color: vec4<f32>,
    layer_color: array<vec4<f32>, STRATA_MAX_LAYERS>,
    // The STRATA_MAX_LAYERS cumulative layer bottoms packed into 2 vec4 lanes (lane0.xyzw + lane1.xy).
    layer_bottom: array<vec4<f32>, 2>,
    bedrock_color: vec4<f32>,
    layer_count: u32,
    _pad: vec3<u32>,
};

struct StrataTable {
    columns: array<StrataColumn, BIOME_COUNT>,
};

// One palette material (mirror of biome::GpuMaterialStd): colour + props (.x = roughness).
struct MaterialEntry {
    color: vec4<f32>,
    props: vec4<f32>,
};

// The flat material palette (mirror of biome::MaterialPaletteStd) the baked surface-material ids index.
struct MaterialPalette {
    materials: array<MaterialEntry, MAX_MATERIALS>,
    count: u32,
    _pad: vec3<u32>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> params: TerrainSurfaceParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var detail_normal: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var detail_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var surface_height: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var biome_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(105) var<uniform> strata: StrataTable;
@group(#{MATERIAL_BIND_GROUP}) @binding(106) var surface_mat_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(107) var<uniform> palette: MaterialPalette;
// Shared PBR texture arrays — one layer per material (layer == TerrainMatId). Triplanar-sampled by the baked
// mat ids; a material with props.y (has_tex) == 0 uses its flat palette colour instead.
@group(#{MATERIAL_BIND_GROUP}) @binding(108) var diffuse_arr: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(109) var normal_arr: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(110) var mra_arr: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(111) var tex_sampler: sampler;

// Manual bilinear fetch of the R32Float surface-height map (unfilterable → textureLoad). `uv` in [0,1].
fn sample_height(uv: vec2<f32>) -> f32 {
    let dims = vec2<f32>(textureDimensions(surface_height));
    let p = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * (dims - 1.0);
    let i0 = floor(p);
    let f = p - i0;
    let x0 = i32(i0.x);
    let y0 = i32(i0.y);
    let x1 = min(x0 + 1, i32(dims.x) - 1);
    let y1 = min(y0 + 1, i32(dims.y) - 1);
    let a = textureLoad(surface_height, vec2<i32>(x0, y0), 0).r;
    let b = textureLoad(surface_height, vec2<i32>(x1, y0), 0).r;
    let c = textureLoad(surface_height, vec2<i32>(x0, y1), 0).r;
    let d = textureLoad(surface_height, vec2<i32>(x1, y1), 0).r;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Nearest-fetch the low-res biome map: (primary id, secondary id, blend). NO bilinear — ids must not
// interpolate; the cross-fade is done analytically by the stored blend weight.
fn sample_biome(uv: vec2<f32>) -> vec3<f32> {
    let dims = vec2<f32>(textureDimensions(biome_tex));
    let px = vec2<i32>(clamp(uv * dims, vec2<f32>(0.0), dims - 1.0));
    return textureLoad(biome_tex, px, 0).xyz;
}

// The cumulative bottom-depth of layer `i` (i < layer_count), unpacking the 2-vec4 packed array.
fn strata_bottom(col: StrataColumn, i: u32) -> f32 {
    let v = col.layer_bottom[i / 4u];
    let lane = i % 4u;
    if (lane == 0u) { return v.x; }
    if (lane == 1u) { return v.y; }
    if (lane == 2u) { return v.z; }
    return v.w;
}

// Walk ONE biome's strata column for `depth` (m below the original surface) → its base colour, cross-fading
// across each layer boundary over `boundary` metres so the bands blend smoothly. Mirror of
// biome::strata_material + preview_color (the SSOT), with a boundary blend the CPU walk doesn't need.
fn strata_color_for(biome: u32, depth: f32, boundary: f32) -> vec3<f32> {
    let b = min(biome, BIOME_COUNT - 1u);
    let col = strata.columns[b];
    if (depth <= 0.0) { return col.surface_color.rgb; }
    let n = min(col.layer_count, STRATA_MAX_LAYERS);
    var top = 0.0;
    var prev = col.surface_color.rgb; // the band just above the current one (surface above layer 0)
    for (var i = 0u; i < n; i = i + 1u) {
        let bottom = strata_bottom(col, i);
        let here = col.layer_color[i].rgb;
        if (depth < bottom) {
            // Inside layer i: blend from the previous band across the TOP boundary over `boundary` metres.
            let t = clamp((depth - top) / max(boundary, 1e-4), 0.0, 1.0);
            return mix(prev, here, t);
        }
        top = bottom;
        prev = here;
    }
    // Below the last layer → bedrock, blended across its top boundary too.
    let t = clamp((depth - top) / max(boundary, 1e-4), 0.0, 1.0);
    return mix(prev, col.bedrock_color.rgb, t);
}

// The volumetric strata colour at this fragment: look up BOTH the primary and secondary biome columns and
// cross-fade by the baked blend weight (so biome boundaries are seamless). `depth` m below the surface.
fn volumetric_color(bio: vec3<f32>, depth: f32, boundary: f32) -> vec3<f32> {
    let prim = u32(bio.x + 0.5);
    let sec = u32(bio.y + 0.5);
    let blend = clamp(bio.z, 0.0, 1.0);
    let cp = strata_color_for(prim, depth, boundary);
    let cs = strata_color_for(sec, depth, boundary);
    // blend → 1 at a border mixes halfway toward the neighbour (primary still dominates). Matches the
    // preview's biome_surface_color intent.
    return mix(cp, cs, blend * 0.5);
}

// One palette material's colour (rgb) + roughness (.a), clamped into the palette.
// Triplanar projection weights from the world normal (sharpened so one axis dominates on near-flat ground).
fn tri_weights(n: vec3<f32>) -> vec3<f32> {
    let w = pow(abs(n), vec3<f32>(4.0));
    return w / max(w.x + w.y + w.z, 1e-5);
}

// Triplanar sample of one array `layer` at world position `wp`, `tiling` world-metres per texture tile,
// projection weights `tw`. textureSample is called UNCONDITIONALLY (uniform control flow — required for the
// implicit-derivative LOD); callers `select` textured-vs-flat afterward rather than branching the sample.
fn tri_sample(t: texture_2d_array<f32>, layer: i32, wp: vec3<f32>, tiling: f32, tw: vec3<f32>) -> vec4<f32> {
    let inv = 1.0 / max(tiling, 0.01);
    let x = textureSample(t, tex_sampler, wp.zy * inv, layer);
    let y = textureSample(t, tex_sampler, wp.xz * inv, layer);
    let z = textureSample(t, tex_sampler, wp.xy * inv, layer);
    return x * tw.x + y * tw.y + z * tw.z;
}

// One material id → its surface (rgb albedo + roughness in `.a`). Textured materials (palette `props.y`)
// triplanar-sample the array layer (== the material id, clamped to the live layer count) and tint by the
// palette colour, modulated by baked AO; untextured materials are the flat palette colour. ALWAYS samples
// (uniform control flow), then `select`s — so an untextured material costs the samples but reads the flat.
fn resolve_mat(id: u32, wp: vec3<f32>, tw: vec3<f32>) -> vec4<f32> {
    let m = palette.materials[min(id, MAX_MATERIALS - 1u)];
    let layer = min(i32(id), max(i32(textureNumLayers(diffuse_arr)) - 1, 0));
    let tiling = max(m.props.z, 0.5);
    // Raw texture albedo (no palette tint — the texture provides the colour; `m.color` is the flat
    // fallback / preview colour for UNTEXTURED materials only), modulated by baked AO.
    let alb = tri_sample(diffuse_arr, layer, wp, tiling, tw).rgb;
    let mra = tri_sample(mra_arr, layer, wp, tiling, tw); // x=metal, y=rough, z=ao
    let textured = vec4<f32>(alb * mra.z, mra.y);
    let flat = vec4<f32>(m.color.rgb, m.props.x);
    return select(flat, textured, m.props.y > 0.5);
}

// The undug SURFACE (rgb albedo + roughness) at this fragment: the worldgen-baked `(mat_a, mat_b, weight)`
// from the nearest surface-material texel, each resolved (textured or flat) and blended by `weight`. The
// triplanar sampling means cliffs + dug walls don't stretch. The baked surface ALREADY encodes biome base +
// snow/rock caps + cliffs + patches — the shader only turns ids into textured PBR.
fn surface_textured(uv: vec2<f32>, wp: vec3<f32>, tw: vec3<f32>) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(surface_mat_tex));
    let px = vec2<i32>(clamp(uv * dims, vec2<f32>(0.0), dims - 1.0));
    let s = textureLoad(surface_mat_tex, px, 0);
    let a = resolve_mat(u32(s.x + 0.5), wp, tw);
    let b = resolve_mat(u32(s.y + 0.5), wp, tw);
    return mix(a, b, clamp(s.z, 0.0, 1.0));
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // Top-down planar UV over the chunk's world-XZ footprint (clamped — the samplers clamp too).
    let uv = clamp((in.world_position.xz - params.chunk_min) / params.chunk_size, vec2<f32>(0.0), vec2<f32>(1.0));

    // ---- Depth below the PRISTINE surface (drives the surface-vs-dug-strata split) ----
    let surf_h = sample_height(uv);
    // SURFACE SKIN (dead-zone): the baked `surf_h` (bilinear of the coarse clipmap) and the mesh geometry
    // (triangulated Transvoxel surface) differ by a sub-voxel residual that, with a thin top stratum, crosses
    // the grass→dirt boundary across the UNDUG surface → speckled dirt/stone. Treat depth within a fraction of
    // the chunk's cell scale as the SURFACE (depth ≤ 0); the strata begin below it. Scales with LOD via
    // chunk_size (residual ∝ cell size). The EXACT fix is a per-vertex pristine-surface-Y attribute (depth
    // interpolates to 0 on the undug face) — lands with D3 of digging.
    let depth = surf_h - in.world_position.y - params.chunk_size * 0.15;
    let boundary = params.surf_b.w;
    let top_band = max(boundary * 2.0, 1.0);
    let depth_w = smoothstep(0.0, top_band, max(depth, 0.0)); // 0 = surface, 1 = below the surface band
    let surf_w = 1.0 - depth_w;                               // 1 = undug surface, 0 = a dug wall

    // ---- Detail normal (coarse chunks) — only on the UNDUG surface ----
    let n_geo = normalize(in.world_normal);
    var n = n_geo;
    if (params.flags.z != 0u) {
        // The Rg16Float texel stores the absolute fine surface slope (dh/dx, dh/dz); reconstruct the hi-fi
        // heightfield normal N = normalize(-dh/dx, 1, -dh/dz) and blend toward it by `strength`, FADED OUT on a
        // dug wall (`surf_w`): the detail normal is the heightfield-up relief and would wrongly tilt a vertical
        // cavity wall toward "up" — a carved chunk keeps its true CSG geometry normal on the walls.
        let slope = textureSample(detail_normal, detail_sampler, uv).rg;
        let n_detail = normalize(vec3<f32>(-slope.x, 1.0, -slope.y));
        n = normalize(mix(n_geo, n_detail, params.strength * surf_w));
    }

    // "View normals" debug: the APPLIED world-normal as RGB (unlit).
    if (params.flags.x != 0u) {
        var out: FragmentOutput;
        out.color = vec4<f32>(n * 0.5 + 0.5, 1.0);
        return out;
    }

    // ---- Volumetric biome strata ----
    let bio = sample_biome(uv);
    // UNDUG surface = the worldgen-baked SURFACE MATERIAL (palette colour + roughness, bilinear so material
    // boundaries are smooth); below the surface band = the id-based depth strata (the DUG cavity walls — grass→
    // dirt→stone→bedrock). ALL the "which material is here" logic is resolved at BAKE time — the shader renders.
    // The undug surface = triplanar-textured baked materials; the dug walls = the flat strata colour (Stage 5
    // can texture those too). `n_geo` drives the triplanar projection so cliffs/walls sample the side planes.
    let surf = surface_textured(uv, in.world_position.xyz, tri_weights(n_geo));
    var albedo = mix(surf.rgb, volumetric_color(bio, depth, boundary), depth_w);

    pbr_input.material.base_color = vec4<f32>(albedo, 1.0);
    pbr_input.material.perceptual_roughness = mix(surf.a, 0.9, depth_w);
    pbr_input.N = n;

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
