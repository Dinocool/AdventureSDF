// Worldgen node-preview GPU raymarch — STAGE 1 placeholder.
// Renders a UV gradient to prove the custom-material → offscreen → egui pipeline works end to end.
// Stage 3 replaces the body with a heightfield raymarch (sampling a CPU-baked height+normal texture).

#import bevy_pbr::forward_io::VertexOutput

struct PreviewParams {
    // Stage 1: only `tint` is used (a constant blue channel). Stage 3 adds camera/zoom/levels.
    tint: vec4<f32>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: PreviewParams;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // UV gradient: red across, green up, blue from the uniform — a recognisable checkpoint pattern.
    return vec4<f32>(in.uv.x, in.uv.y, params.tint.b, 1.0);
}
