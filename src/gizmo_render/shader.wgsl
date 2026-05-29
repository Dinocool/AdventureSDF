// Flat 2D overlay: vertices arrive in NDC (x,y ∈ [-1,1]); we map to clip space and
// pass the per-vertex color straight through. The y is negated to match the
// CPU-side screen→NDC convention (ported from transform-gizmo's gizmo.wgsl).

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position.x, -in.position.y, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
