//! Scene-camera gizmo: a wireframe frustum pointing along the node's forward (local -Z),
//! marking a [`SceneCamera`](crate::node::SceneCamera) node. A depth-tested locator glyph.

use bevy::prelude::*;

use super::{NodeGizmoCtx, NodeGizmoPainter};

const COLOR: Color = Color::srgb(0.6, 0.8, 1.0);

/// Draw the frustum from the node's `GlobalTransform` (forward = local -Z).
pub fn draw(ctx: &NodeGizmoCtx, painter: &mut NodeGizmoPainter) {
    let gizmos = &mut *painter.scene;
    let o = ctx.origin;
    let s = ctx.scale;
    let rot = ctx.rotation;
    let fwd = (rot * Vec3::NEG_Z).normalize_or_zero();
    let right = (rot * Vec3::X).normalize_or_zero();
    let up = (rot * Vec3::Y).normalize_or_zero();

    // Apex at the origin; a rectangular far plane `len` ahead, half-size (hw, hh).
    let len = s * 1.2;
    let hw = s * 0.5;
    let hh = s * 0.4;
    let center = o + fwd * len;
    let corners = [
        center + right * hw + up * hh, // top-right
        center - right * hw + up * hh, // top-left
        center - right * hw - up * hh, // bottom-left
        center + right * hw - up * hh, // bottom-right
    ];

    // Edges from the apex to each far corner.
    for c in corners {
        gizmos.line(o, c, COLOR);
    }
    // Far rectangle.
    for i in 0..4 {
        gizmos.line(corners[i], corners[(i + 1) % 4], COLOR);
    }
    // Up indicator: a small triangle above the top edge of the far plane.
    let top_mid = center + up * hh;
    let tip = top_mid + up * (hh * 0.6);
    gizmos.line(corners[0], tip, COLOR);
    gizmos.line(corners[1], tip, COLOR);
}

/// Local-space oriented pick bounds: the frustum spans the apex (origin) to the far plane
/// along local -Z (out to ~1.2), with the far rect's half-extents in the right/up plane.
pub fn pick_bounds(scale: f32) -> (Vec3, Vec3) {
    (
        Vec3::new(0.0, 0.0, -0.6 * scale),
        Vec3::new(0.5, 0.5, 0.6) * scale,
    )
}
