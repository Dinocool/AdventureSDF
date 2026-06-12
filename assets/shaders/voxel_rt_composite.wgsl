// Composite the HW-RT voxel raymarch output over the camera view target.
//
// A fullscreen triangle samples the raymarch storage texture (rgba16float; alpha=1 where a voxel was hit,
// alpha=0 on a miss) and alpha-blends it over the existing view contents (the Stage-1 cubes / clear),
// so toggling the HW-RT view replaces the cube image where rays hit and leaves the rest untouched.

@group(0) @binding(0) var raymarch_tex: texture_2d<f32>;
@group(0) @binding(1) var raymarch_sampler: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Standard fullscreen triangle (covers NDC with a single oversized tri).
@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    out.uv = uv;
    out.clip = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    // Flip Y so uv (0,0) maps to the top-left texel (matches the compute shader's row indexing).
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(raymarch_tex, raymarch_sampler, in.uv);
    // Premultiplied-style: emit the colour with its hit alpha; ALPHA_BLENDING in the pipeline blends it
    // over the view. Misses (alpha 0) leave the view target unchanged.
    return vec4<f32>(c.rgb, c.a);
}

// --- DLSS-RR resolve pass (Stage 4c, `--features dlss`) ---------------------------------------------
// A fullscreen pass that lands the raymarch's per-pixel guides into the textures DLSS-RR reads but which a
// compute shader CANNOT storage-write (they're RENDER_ATTACHMENT-only): the engine DEPTH prepass texture
// (written via `@builtin(frag_depth)`) and the MOTION-VECTOR prepass texture (colour attachment). It also
// REPLACES the HDR view target with the full lit colour (attachment 0, no blend — DLSS reads it as `color`).
//
// Inputs are the storage textures the `raymarch_dlss` compute filled (sampled by the SAME fullscreen UV):
@group(0) @binding(2) var dlss_color_tex: texture_2d<f32>;
@group(0) @binding(3) var dlss_depth_tex: texture_2d<f32>;
@group(0) @binding(4) var dlss_motion_tex: texture_2d<f32>;

struct ResolveOut {
    @location(0) color: vec4<f32>,   // → HDR view target (DLSS `color`)
    @location(1) motion: vec4<f32>,  // → motion-vector prepass texture (Rg16Float; only .xy used)
    @builtin(frag_depth) depth: f32, // → depth prepass texture (reverse-Z)
};

// Use the INTEGER fragment position (= dest pixel) to `textureLoad` the matching source texel 1:1. The
// raymarch compute wrote the top-left `render_res` subrect; the render pass viewport is clamped to the same
// subrect, so `position.xy` indexes both identically — no UV scaling mismatch between full-size textures and
// the partial render resolution.
@fragment
fn fs_resolve_dlss(in: VsOut) -> ResolveOut {
    var out: ResolveOut;
    let px = vec2<i32>(i32(in.clip.x), i32(in.clip.y));
    let c = textureLoad(dlss_color_tex, px, 0);
    let d = textureLoad(dlss_depth_tex, px, 0).r;
    let mv = textureLoad(dlss_motion_tex, px, 0).xy;
    out.color = vec4<f32>(c.rgb, 1.0);
    out.motion = vec4<f32>(mv, 0.0, 0.0);
    out.depth = d;
    return out;
}
