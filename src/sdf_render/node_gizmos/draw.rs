//! Shared drawing primitives for node gizmos, so each kind doesn't re-derive billboard /
//! ring / screen-constant-handle math. Generic over the gizmo config group, so a kind can
//! draw locator glyphs into the depth-respecting
//! [`SdfNodeGizmos`](crate::sdf_render::SdfNodeGizmos) group and always-on-top manipulator
//! elements (rings, handles) into [`SdfOverlayGizmos`](crate::sdf_render::SdfOverlayGizmos).

use bevy::gizmos::config::GizmoConfigGroup;
use bevy::prelude::*;

/// World units per pixel at `origin`'s depth from the camera. Multiply by a pixel size to
/// get a screen-constant world size (a handle that stays the same on-screen regardless of
/// distance). `proj_y = cot(fov/2)` from `camera.clip_from_view().y_axis.y`.
pub fn world_per_pixel(origin: Vec3, cam: &Transform, proj_y: f32, window_h: f32) -> f32 {
    let dist = (origin - cam.translation).length().max(0.001);
    (2.0 * dist / proj_y) / window_h
}

/// Rotation that orients a +Z-facing shape (disc/ring) to face the camera along `view_fwd`.
pub fn billboard(view_fwd: Vec3) -> Quat {
    Quat::from_rotation_arc(Vec3::Z, view_fwd)
}

/// Draw a camera-facing ring of `radius` at `origin` into any gizmo config group.
pub fn face_circle<G: GizmoConfigGroup>(
    gizmos: &mut Gizmos<G>,
    origin: Vec3,
    view_fwd: Vec3,
    radius: f32,
    color: Color,
    resolution: u32,
) {
    gizmos
        .circle(Isometry3d::new(origin, billboard(view_fwd)), radius, color)
        .resolution(resolution);
}

/// World-axis normals of the two great circles that make up the wireframe sphere (a
/// vertical meridian in the YZ plane + a horizontal equator in the XZ plane). Picking
/// uses the same pair so clicking either drawn circle selects the light.
pub const SPHERE_CIRCLE_NORMALS: [Vec3; 2] = [Vec3::X, Vec3::Y];

/// Draw a wireframe sphere of `radius` at `origin` as two perpendicular great circles
/// (Godot-style omni-light look), into any gizmo config group.
pub fn wire_sphere<G: GizmoConfigGroup>(
    gizmos: &mut Gizmos<G>,
    origin: Vec3,
    radius: f32,
    color: Color,
    resolution: u32,
) {
    for n in SPHERE_CIRCLE_NORMALS {
        let rot = Quat::from_rotation_arc(Vec3::Z, n);
        gizmos
            .circle(Isometry3d::new(origin, rot), radius, color)
            .resolution(resolution);
    }
}

/// Draw a square outline (4 lines) centered at `center`, spanning `half` world units along
/// the `right`/`up` axes. Used for draggable handles on a gizmo.
pub fn square_handle<G: GizmoConfigGroup>(
    gizmos: &mut Gizmos<G>,
    center: Vec3,
    right: Vec3,
    up: Vec3,
    half: f32,
    color: Color,
) {
    let (r, u) = (right * half, up * half);
    let corners = [
        center - r - u,
        center + r - u,
        center + r + u,
        center - r + u,
    ];
    for i in 0..4 {
        gizmos.line(corners[i], corners[(i + 1) % 4], color);
    }
}
