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

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> params: TerrainSurfaceParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var detail_normal: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var detail_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var surface_height: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var biome_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(105) var<uniform> strata: StrataTable;

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

// Surface treatment for the TOP (undug) layer: override the strata surface colour by slope/height/biome.
// `n` = the geometric/world surface normal (its .y = cos of the slope from vertical); `y` = world height;
// `bio` = the biome sample (so snow only caps COLD biomes). Returns the treated colour. Tunable via
// params.surf_a / surf_b (editor sliders); the master strength surf_b.z fades the whole treatment.
fn surface_treatment(base: vec3<f32>, n: vec3<f32>, y: f32, bio: vec3<f32>) -> vec3<f32> {
    let master = clamp(params.surf_b.z, 0.0, 1.0);
    if (master <= 0.0) { return base; }
    var col = base;

    // SAND near sea level (any biome): a band ±surf_b.x around the sea level surf_b.y.
    let sea = params.surf_b.y;
    let sand_band = max(params.surf_b.x, 1e-3);
    let sand = vec3<f32>(0.62, 0.50, 0.27);
    let sand_w = 1.0 - smoothstep(0.0, sand_band, abs(y - sea));
    col = mix(col, sand, sand_w * 0.85);

    // SNOW on high + COLD ground. Cold = a low-temperature biome: Tundra (3) or Snowy (4). Blend the cold
    // factor across the biome boundary (primary↔secondary by the baked blend) so the snow edge is SMOOTH,
    // not a hard per-texel flip on the primary id (that read as a blocky patch).
    let prim = u32(bio.x + 0.5);
    let sec = u32(bio.y + 0.5);
    let cold_p = select(0.0, 1.0, prim == 3u || prim == 4u);
    let cold_s = select(0.0, 1.0, sec == 3u || sec == 4u);
    // `blend * 0.5` — the SAME weighting volumetric_color uses (primary frac = 1 - blend*0.5). At the border
    // both sides read 0.5 ⇒ CONTINUOUS; the raw `blend` jumped 0↔1 as primary/secondary swapped → a hard edge.
    let cold = mix(cold_p, cold_s, clamp(bio.z, 0.0, 1.0) * 0.5);
    let snow = vec3<f32>(0.85, 0.88, 0.95);
    let snow_w = smoothstep(params.surf_a.z, params.surf_a.w, y) * cold;
    col = mix(col, snow, clamp(snow_w, 0.0, 1.0));

    // ROCK on STEEP slopes (any biome): n.y is the cos of the slope; rock from surf_a.x (start) to
    // surf_a.y (full). Smaller cos = steeper, so the ramp goes start → full as n.y DECREASES.
    let rock = vec3<f32>(0.13, 0.13, 0.14);
    let rock_w = 1.0 - smoothstep(params.surf_a.y, params.surf_a.x, n.y);
    col = mix(col, rock, clamp(rock_w, 0.0, 1.0));

    return mix(base, col, master);
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // Top-down planar UV over the chunk's world-XZ footprint (clamped — the samplers clamp too).
    let uv = clamp((in.world_position.xz - params.chunk_min) / params.chunk_size, vec2<f32>(0.0), vec2<f32>(1.0));

    // ---- Detail normal (coarse chunks) ----
    let n_geo = normalize(in.world_normal);
    var n = n_geo;
    if (params.flags.z != 0u) {
        // The Rg16Float texel stores the absolute fine surface slope (dh/dx, dh/dz); reconstruct the hi-fi
        // heightfield normal N = normalize(-dh/dx, 1, -dh/dz) and blend toward it by `strength`.
        let slope = textureSample(detail_normal, detail_sampler, uv).rg;
        let n_detail = normalize(vec3<f32>(-slope.x, 1.0, -slope.y));
        n = normalize(mix(n_geo, n_detail, params.strength));
    }

    // "View normals" debug: the APPLIED world-normal as RGB (unlit).
    if (params.flags.x != 0u) {
        var out: FragmentOutput;
        out.color = vec4<f32>(n * 0.5 + 0.5, 1.0);
        return out;
    }

    // ---- Volumetric biome strata ----
    let surf_h = sample_height(uv);
    // SURFACE SKIN (dead-zone): the baked `surf_h` (bilinear of the coarse clipmap) and the mesh geometry
    // (triangulated Transvoxel surface) differ by a sub-voxel residual that, with a thin top stratum, crosses
    // the grass→dirt boundary across the UNDUG surface → speckled dirt/stone. Treat depth within a fraction of
    // the chunk's cell scale as the SURFACE (depth ≤ 0); the strata begin below it. Scales with LOD via
    // chunk_size (residual ∝ cell size). The EXACT fix is a per-vertex pristine-surface-Y attribute (depth
    // interpolates to 0 on the undug face) — that lands with Stage-4 digging, which needs it regardless.
    let depth = surf_h - in.world_position.y - params.chunk_size * 0.15;
    let bio = sample_biome(uv);
    let boundary = params.surf_b.w;
    var albedo = volumetric_color(bio, depth, boundary);
    // Surface treatment only on the TOP (undug) band — fade it out as we go below the surface so exposed
    // strata on a dug wall are NOT snowed/sanded over. Tied to the first stratum thickness (~ boundary*2).
    let top_band = max(boundary * 2.0, 1.0);
    let top_w = 1.0 - smoothstep(0.0, top_band, max(depth, 0.0));
    let treated = surface_treatment(albedo, n_geo, in.world_position.y, bio);
    albedo = mix(albedo, treated, top_w);

    pbr_input.material.base_color = vec4<f32>(albedo, 1.0);
    pbr_input.N = n;

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
