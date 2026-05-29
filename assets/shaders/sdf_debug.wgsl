#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@fragment
fn main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let color = vec3<f32>(0.0, uv.x, uv.y);
    return vec4<f32>(color, 1.0);
}
