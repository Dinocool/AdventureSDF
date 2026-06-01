//! Empty-node gizmo: three short colored axis lines (X red / Y green / Z blue) at the
//! origin — a generic locator marker for otherwise-invisible nodes.

use bevy::prelude::*;

use super::{NodeGizmoCtx, NodeGizmoPainter};

/// Draw the three oriented axis lines from the node's `GlobalTransform`.
/// A depth-tested locator glyph, so it draws into `painter.scene`.
pub fn draw(ctx: &NodeGizmoCtx, painter: &mut NodeGizmoPainter) {
    let gizmos = &mut *painter.scene;
    let o = ctx.origin;
    let rot = ctx.rotation;
    let s = ctx.scale;
    gizmos.line(o, o + rot * Vec3::X * s, Color::srgb(0.9, 0.2, 0.2));
    gizmos.line(o, o + rot * Vec3::Y * s, Color::srgb(0.3, 0.9, 0.2));
    gizmos.line(o, o + rot * Vec3::Z * s, Color::srgb(0.2, 0.4, 0.95));
}

/// Local-space pick bounds: lines run from the origin out to +scale on each axis.
pub fn pick_bounds(scale: f32) -> (Vec3, Vec3) {
    (Vec3::splat(0.5 * scale), Vec3::splat(0.5 * scale))
}
