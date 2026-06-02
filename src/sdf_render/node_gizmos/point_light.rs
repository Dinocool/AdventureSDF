//! Point-light gizmo: a small central bulb + a camera-facing ring of radius
//! `PointLight.range`, with a square handle on the ring's edge that the user drags to
//! change the radius. The radius lives on the entity's Bevy [`PointLight`] (`range`), the
//! single source of truth shared with standard PBR lighting and any SDF reader.

use bevy::prelude::*;

use crate::sdf_render::gizmo::GizmoState;
use crate::sdf_render::picking::{Ray, mouse_to_ray};
use crate::sdf_render::{SdfCamera, SdfNodeGizmos, SdfSelection};

use super::draw::{face_circle, square_handle, wire_sphere, world_per_pixel};
use super::{NodeGizmoCtx, NodeGizmoPainter};

/// Falloff-cutoff (`range`) ring + handle colour.
const RANGE_COLOR: Color = Color::srgb(1.0, 0.9, 0.6);
/// Physical-size (`radius`) ring + handle colour (cooler, to read distinct from range).
const RADIUS_COLOR: Color = Color::srgb(0.5, 0.85, 1.0);
/// Fallbacks when a point-light gizmo has no `PointLight` (defensive; a spawned light
/// always has one). `(range, radius)`.
const DEFAULT_RANGE: f32 = 5.0;
const DEFAULT_RADIUS: f32 = 0.5;
/// Square handle half-size, in pixels (screen-constant, like the transform handles).
const HANDLE_PX: f32 = 6.0;
/// Lightbulb glyph size, in pixels (screen-constant so it never vanishes at large range).
const BULB_PX: f32 = 14.0;
/// Click tolerance for grabbing a handle, in pixels.
const HANDLE_FOCUS_PX: f32 = 12.0;
/// Minimum value a drag can set (so a ring never collapses to a point).
const MIN_VALUE: f32 = 0.05;

/// Which of a point light's two scalar handles is being manipulated.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Knob {
    /// Falloff cutoff (`PointLight.range`) — outer ring, handle on the camera-right axis.
    Range,
    /// Physical light size (`PointLight.radius`) — inner ring, handle on the camera-up axis.
    Radius,
}

/// Camera-facing geometry of the point-light gizmo, shared by [`draw`] and the drag system
/// so they agree on exactly where each ring + handle sits.
struct Geom {
    view_right: Vec3,
    view_up: Vec3,
    view_fwd: Vec3,
    /// Handle half-extent in world units (screen-constant).
    handle_half: f32,
    /// World position of the range handle (`origin + view_right * range`).
    range_handle: Vec3,
    /// World position of the radius handle (`origin + view_up * radius`).
    radius_handle: Vec3,
}

/// Resolve the gizmo geometry for a point light at `origin` with `range` + `radius`.
fn geom(
    origin: Vec3,
    range: f32,
    radius: f32,
    cam: &Transform,
    proj_y: f32,
    window_h: f32,
) -> Geom {
    let view_right = cam.right().as_vec3();
    let view_up = cam.up().as_vec3();
    Geom {
        view_right,
        view_up,
        view_fwd: cam.forward().as_vec3(),
        handle_half: world_per_pixel(origin, cam, proj_y, window_h) * HANDLE_PX,
        range_handle: origin + view_right * range,
        radius_handle: origin + view_up * radius,
    }
}

/// Draw the bulb + the range & radius wireframe spheres (Godot-style omni-light look).
/// The bulb + spheres always draw depth-tested in-scene, so they're occluded by geometry
/// in front of them (they read as part of the scene). When the light is SELECTED, the
/// draggable handles are added always-on-top so they stay grabbable. Needs the camera (the
/// bulb + handles are camera-facing), so it no-ops without one.
pub fn draw(ctx: &NodeGizmoCtx, painter: &mut NodeGizmoPainter) {
    let Some((camera, cam_xf)) = ctx.camera else {
        return;
    };
    let (range, radius) = ctx.light.unwrap_or((DEFAULT_RANGE, DEFAULT_RADIUS));
    let proj_y = camera.clip_from_view().y_axis.y;
    let g = geom(ctx.origin, range, radius, cam_xf, proj_y, ctx.window_h);

    // Lightbulb glyph at the center, screen-constant so it stays readable at any distance.
    let s = world_per_pixel(ctx.origin, cam_xf, proj_y, ctx.window_h) * BULB_PX;
    draw_bulb(painter.scene, ctx.origin, g.view_right, g.view_up, g.view_fwd, s, RANGE_COLOR);

    // Range = outer falloff sphere; radius = inner physical-size sphere. Each is two
    // perpendicular great circles (the Godot omni-light wireframe). Always depth-tested so
    // they occlude behind geometry.
    wire_sphere(painter.scene, ctx.origin, range, RANGE_COLOR, 48);
    wire_sphere(painter.scene, ctx.origin, radius, RADIUS_COLOR, 32);

    // When selected, add the draggable handles always-on-top so they stay grabbable
    // regardless of what's in front (range on the right axis, radius on the up axis).
    if ctx.selected {
        square_handle(painter.overlay, g.range_handle, g.view_right, g.view_up, g.handle_half, RANGE_COLOR);
        square_handle(painter.overlay, g.radius_handle, g.view_right, g.view_up, g.handle_half, RADIUS_COLOR);
    }
}

/// Draw a small lightbulb glyph in the camera plane: a round glass envelope above a
/// short screw base (two stems + a foot), centered at `origin` and sized by `s` (world
/// units; pass a screen-constant value for a constant on-screen size).
fn draw_bulb(
    gizmos: &mut Gizmos<SdfNodeGizmos>,
    origin: Vec3,
    right: Vec3,
    up: Vec3,
    fwd: Vec3,
    s: f32,
    color: Color,
) {
    // Glass envelope: a circle centered slightly above the origin.
    let glass_c = origin + up * (s * 0.35);
    face_circle(gizmos, glass_c, fwd, s * 0.7, color, 20);
    // Screw base: two short vertical stems dropping from the glass, joined by a foot.
    let half_w = s * 0.35;
    let top = origin - up * (s * 0.25);
    let bot = origin - up * (s * 0.7);
    let l_top = top - right * half_w;
    let r_top = top + right * half_w;
    let l_bot = bot - right * half_w;
    let r_bot = bot + right * half_w;
    gizmos.line(l_top, l_bot, color);
    gizmos.line(r_top, r_bot, color);
    gizmos.line(l_bot, r_bot, color);
}

/// Local-space pick bounds: only the central bulb is OBB-pickable (for selecting the
/// light). The ring + handle are camera-facing, handled by the drag system below.
pub fn pick_bounds(scale: f32) -> (Vec3, Vec3) {
    (Vec3::ZERO, Vec3::splat(0.2 * scale))
}

/// Tracks an in-progress handle drag: which entity + which knob. `None` = idle.
#[derive(Resource, Default)]
pub struct PointLightDrag(Option<(Entity, Knob)>);

/// Closest-point parameter of the mouse ray on the infinite axis line `origin + t*axis`.
fn project_onto_axis(ray: &Ray, origin: Vec3, axis: Vec3) -> f32 {
    let w = ray.origin - origin;
    let a = ray.direction.dot(ray.direction);
    let b = ray.direction.dot(axis);
    let c = axis.dot(axis);
    let d = ray.direction.dot(w);
    let e = axis.dot(w);
    let denom = a * c - b * b;
    if denom.abs() < 1e-8 {
        return 0.0;
    }
    (a * e - b * d) / denom
}

/// Drag the selected point light's `range` or `radius` via its ring handles. Runs in
/// `Last` BEFORE `gizmo::gizmo_update` and claims the click (`GizmoState.claimed_click`) so
/// the transform gizmo + `sdf_picking` skip it. A press near a handle begins a drag of that
/// knob; while held, the cursor projects onto the handle's axis and writes the field. A
/// drag begins ONLY on a press that lands on a handle, so a click elsewhere never grabs one.
#[allow(clippy::too_many_arguments)]
pub fn point_light_radius_drag(
    mouse: Res<ButtonInput<MouseButton>>,
    mut drag: ResMut<PointLightDrag>,
    mut gizmo_state: ResMut<GizmoState>,
    selection: Res<SdfSelection>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &Transform), With<SdfCamera>>,
    globals: Query<&GlobalTransform>,
    mut lights: Query<&mut PointLight, With<crate::node::Node3D>>,
    gizmos: Query<&crate::node::EditorGizmo>,
) {
    // This system is first in the `Last` interaction chain, so it owns the per-frame reset
    // of `claimed_click`: clear last frame's claim (which may have been the TRANSFORM
    // gizmo's) up front, then re-assert below only if WE are dragging. `gizmo_update` (next
    // in the chain) yields when it sees this flag set, so the two never fight over a press.
    gizmo_state.claimed_click = false;

    let Some(entity) = selection.entity else {
        return;
    };
    // Only point-light nodes have these handles.
    if !matches!(
        gizmos.get(entity),
        Ok(crate::node::EditorGizmo::PointLight { .. })
    ) {
        return;
    }
    let (Ok(window), Ok((camera, cam_xf))) = (windows.single(), cameras.single()) else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Some(ray) = mouse_to_ray(camera, cam_xf, window, cursor) else {
        return;
    };
    let Ok(origin) = globals.get(entity).map(|g| g.translation()) else {
        return;
    };
    let (range, radius) = lights
        .get(entity)
        .map(|l| (l.range, l.radius))
        .unwrap_or((DEFAULT_RANGE, DEFAULT_RADIUS));
    let proj_y = camera.clip_from_view().y_axis.y;
    let g = geom(origin, range, radius, cam_xf, proj_y, window.height());

    // Begin a drag ONLY on the press frame, and only if it lands on a handle. Pick the
    // nearer of the two if both are within tolerance.
    if mouse.just_pressed(MouseButton::Left) && drag.0.is_none() {
        let cam_global = GlobalTransform::from(*cam_xf);
        let near = |world: Vec3| {
            camera
                .world_to_viewport(&cam_global, world)
                .ok()
                .map(|s| s.distance(cursor))
                .filter(|d| *d <= HANDLE_FOCUS_PX)
        };
        let range_d = near(g.range_handle).map(|d| (d, Knob::Range));
        let radius_d = near(g.radius_handle).map(|d| (d, Knob::Radius));
        drag.0 = [range_d, radius_d]
            .into_iter()
            .flatten()
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, knob)| (entity, knob));
    }

    // While dragging THIS entity, write the grabbed knob from the cursor's projection onto
    // its handle axis. Claim the click so nothing else reacts.
    if let Some((dragged, knob)) = drag.0
        && dragged == entity
    {
        let axis = match knob {
            Knob::Range => g.view_right,
            Knob::Radius => g.view_up,
        };
        let t = project_onto_axis(&ray, origin, axis).max(MIN_VALUE);
        if let Ok(mut light) = lights.get_mut(entity) {
            match knob {
                Knob::Range => light.range = t,
                Knob::Radius => light.radius = t,
            }
        }
        gizmo_state.claimed_click = true;
    }
}

/// End any active point-light handle drag the moment the button is released. Runs UNGATED
/// (not behind `ViewportInputAllowed`) so a release while the pointer is over a dock panel
/// still clears the drag — otherwise a stuck drag resumes on the next click anywhere.
/// Mirrors `gizmo::clear_gizmo_drag_on_release`.
pub fn clear_point_light_drag_on_release(
    mouse: Res<ButtonInput<MouseButton>>,
    mut drag: ResMut<PointLightDrag>,
) {
    if drag.0.is_some() && !mouse.pressed(MouseButton::Left) {
        drag.0 = None;
    }
}

/// Register the drag resource + systems (the grab runs before the transform gizmo in
/// `Last`; the release-clear runs ungated so a release off-viewport still ends the drag).
pub fn register(app: &mut App) {
    use crate::scene_manager::AppScene;
    use crate::sdf_render::{ViewportInputAllowed, gizmo};
    use bevy::prelude::*;

    app.init_resource::<PointLightDrag>()
        .add_systems(
            Last,
            point_light_radius_drag
                .before(gizmo::gizmo_update)
                .run_if(in_state(AppScene::SdfEditor))
                .run_if(|allowed: Res<ViewportInputAllowed>| allowed.0),
        )
        .add_systems(
            Last,
            clear_point_light_drag_on_release.run_if(in_state(AppScene::SdfEditor)),
        );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bulb_bounds_scale_with_size() {
        let (center, half) = pick_bounds(2.0);
        assert_eq!(center, Vec3::ZERO);
        assert_eq!(half, Vec3::splat(0.4));
    }
}
