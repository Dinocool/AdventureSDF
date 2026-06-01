//! Directional-light gizmo: a sun disc + radiating spokes + parallel rays along the
//! node's forward (local -Z), showing the light's travel direction.

use bevy::prelude::*;

use super::{NodeGizmoCtx, NodeGizmoPainter};

const COLOR: Color = Color::srgb(1.0, 0.85, 0.3);

/// Draw the sun glyph from the node's `GlobalTransform` (orientation drives the rays).
/// A depth-tested locator glyph (no manipulator handles), so it draws into `painter.scene`.
pub fn draw(ctx: &NodeGizmoCtx, painter: &mut NodeGizmoPainter) {
    let gizmos = &mut *painter.scene;
    let origin = ctx.origin;
    let scale = ctx.scale;
    let rot = ctx.rotation;
    // Forward (the light's travel direction) is local -Z.
    let dir = (rot * Vec3::NEG_Z).normalize_or_zero();
    let right = (rot * Vec3::X).normalize_or_zero();
    let up = (rot * Vec3::Y).normalize_or_zero();

    // Sun disc: a small ring facing the light direction.
    gizmos
        .circle(Isometry3d::new(origin, rot), scale * 0.4, COLOR)
        .resolution(24);
    // Radiating spokes from the disc (classic sun glyph).
    for k in 0..8 {
        let a = k as f32 * std::f32::consts::TAU / 8.0;
        let d = right * a.cos() + up * a.sin();
        gizmos.line(origin + d * scale * 0.4, origin + d * scale * 0.62, COLOR);
    }
    // Parallel rays offset around the disc, all pointing along `dir`, with an arrowhead so
    // the travel direction is unambiguous.
    let len = scale * 1.6;
    for (ox, oy) in [(0.0, 0.0), (0.55, 0.0), (-0.55, 0.0), (0.0, 0.55), (0.0, -0.55)] {
        let base = origin + (right * ox + up * oy) * scale;
        gizmos.arrow(base, base + dir * len, COLOR);
    }
}

/// Local-space oriented pick bounds: disc (radius .4) + spokes (to .62) in the right/up
/// plane; rays run from the origin along local -Z out to length 1.6.
pub fn pick_bounds(scale: f32) -> (Vec3, Vec3) {
    (
        Vec3::new(0.0, 0.0, -0.8 * scale),
        Vec3::new(0.62, 0.62, 0.8) * scale,
    )
}
