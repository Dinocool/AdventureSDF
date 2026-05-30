//! In-tree transform gizmo for the SDF editor — a faithful port of
//! `transform-gizmo`'s handle set, rendered as a filled 2D overlay (see
//! [`crate::gizmo_render`]).
//!
//! Handle set (matches the plugin):
//! - Translate: 3 axis arrows + 3 plane squares (XY/YZ/XZ) + a view-plane disc.
//! - Rotate: 3 axis rings + a camera-facing view ring (outer).
//! - Scale: 3 axis handles (blunt fat cap) + 3 plane squares + a view-plane (uniform).
//!
//! Geometry is positioned/sized from the target's **translation only** (plus
//! rotation in Local orientation) so object scale never distorts the manipulator.
//! Sizes are screen-constant via a world-per-pixel factor. Single source of truth:
//! [`Handle`] both tessellates (draw) and ray-picks one definition.

use bevy::prelude::*;

use crate::gizmo_render::{GizmoDraw, GizmoMesh, ShapeBuilder};

use super::bake_scheduler::SyncBakeRequest;
use super::picking::{Ray, mouse_to_ray};
use super::{SdfCamera, SdfSelection, SdfVolume};

// --- Pixel constants (matched to transform-gizmo defaults) ---
/// Axis length / outer extent, in pixels (the plugin's `gizmo_size`).
const GIZMO_SIZE_PX: f32 = 90.0;
/// Stroke width, in pixels (the plugin's `stroke_width`).
const STROKE_PX: f32 = 4.0;
/// Click tolerance, in pixels.
const FOCUS_PX: f32 = 10.0;
/// Inner-circle radius fraction (arrows start here; view-plane disc radius).
const INNER_FRAC: f32 = 0.2;
/// Plane handle: centre offset fraction (`0.5 * gizmo_size`) and size fraction.
const PLANE_OFFSET_FRAC: f32 = 0.5;
const PLANE_SIZE_FRAC: f32 = 0.1; // + 2*stroke (added in world units below)
/// View rotation ring radius fraction (just beyond the axis rings).
const VIEW_RING_FRAC: f32 = 1.1;
/// Axes fade out when their screen projection is within this dot-to-view band.
const FADE_START: f32 = 0.95;
const FADE_END: f32 = 0.99;

/// World axes in index order (0=X, 1=Y, 2=Z).
const AXES: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];

/// The three in-plane axis pairs for plane handles (the third axis is the normal).
const PLANE_PAIRS: [(u8, u8); 3] = [(0, 1), (1, 2), (0, 2)];

/// Which transform modes the manipulator currently shows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GizmoModes {
    pub translate: bool,
    pub rotate: bool,
    pub scale: bool,
}

impl GizmoModes {
    pub const TRANSLATE: Self = Self {
        translate: true,
        rotate: false,
        scale: false,
    };
    pub const ROTATE: Self = Self {
        translate: false,
        rotate: true,
        scale: false,
    };
    pub const SCALE: Self = Self {
        translate: false,
        rotate: false,
        scale: true,
    };

    pub fn all() -> Self {
        Self {
            translate: true,
            rotate: true,
            scale: true,
        }
    }
}

/// Gizmo orientation: world axes, or the target's local axes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Orientation {
    #[default]
    World,
    Local,
}

/// Identifies one handle. Axis indices are 0=X, 1=Y, 2=Z; plane handles store their
/// two in-plane axes; `*View` handles are camera-facing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HandleId {
    TranslateAxis(u8),
    TranslatePlane(u8, u8),
    TranslateView,
    RotateAxis(u8),
    RotateView,
    ScaleAxis(u8),
    ScalePlane(u8, u8),
    ScaleView,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Translate,
    Rotate,
    Scale,
}

impl HandleId {
    fn kind(self) -> Kind {
        match self {
            HandleId::TranslateAxis(_) | HandleId::TranslatePlane(..) | HandleId::TranslateView => {
                Kind::Translate
            }
            HandleId::RotateAxis(_) | HandleId::RotateView => Kind::Rotate,
            HandleId::ScaleAxis(_) | HandleId::ScalePlane(..) | HandleId::ScaleView => Kind::Scale,
        }
    }

    fn shown(self, m: GizmoModes) -> bool {
        match self.kind() {
            Kind::Translate => m.translate,
            Kind::Rotate => m.rotate,
            Kind::Scale => m.scale,
        }
    }
}

/// In-progress drag of one handle.
pub struct DragState {
    pub id: HandleId,
    /// World-space axis the drag is measured along (or rotation axis / plane normal).
    pub axis: Vec3,
    /// Secondary in-plane axis (plane handles), else `Vec3::ZERO`.
    pub axis2: Vec3,
    /// Parameter along `axis` (or angle) at drag start.
    pub start_proj: f32,
    pub start_point: Vec3,
    pub start_xf: Transform,
}

/// Editor gizmo state. A core resource; the editor panel + keybinds mutate the
/// mode/snap fields.
#[derive(Resource)]
pub struct GizmoState {
    pub modes: GizmoModes,
    pub orientation: Orientation,
    /// Effective snap for this frame (what `apply_drag` reads): `snap_sticky` OR Ctrl
    /// held. Recomputed each frame by the keybind system.
    pub snap: bool,
    /// Sticky snap toggle, driven by the toolbar magnet button. Persists across frames
    /// (unlike the momentary Ctrl-hold).
    pub snap_sticky: bool,
    pub snap_move: f32,
    pub snap_angle: f32,
    pub snap_scale: f32,
    pub hovered: Option<HandleId>,
    pub drag: Option<DragState>,
    /// Set when the gizmo consumed this frame's left-press, so `sdf_picking` skips
    /// selecting a volume underneath.
    pub claimed_click: bool,
}

impl Default for GizmoState {
    fn default() -> Self {
        Self {
            modes: GizmoModes::all(),
            orientation: Orientation::World,
            snap: false,
            snap_sticky: false,
            snap_move: 0.25,
            snap_angle: std::f32::consts::FRAC_PI_8,
            snap_scale: 0.1,
            hovered: None,
            drag: None,
            claimed_click: false,
        }
    }
}

/// A resolved handle: world-space geometry derived from the gizmo origin, basis,
/// and a screen-constant `unit` (world units per pixel). Camera-facing handles use
/// `view_right`/`view_up`.
struct Handle {
    id: HandleId,
    origin: Vec3,
    basis: [Vec3; 3],
    /// World units per pixel at the gizmo's depth (size = unit * pixels).
    unit: f32,
    view_right: Vec3,
    view_up: Vec3,
    view_fwd: Vec3,
    /// Whether translate+scale are both shown (axis layout must not overlap).
    both_ts: bool,
    /// Whether rotate rings are shown (translate arrows then sit outside the rings).
    show_rotate: bool,
}

impl Handle {
    fn axis(&self, i: u8) -> Vec3 {
        self.basis[i as usize]
    }

    /// Inner gap where axis handles begin.
    fn inner(&self) -> f32 {
        self.unit * GIZMO_SIZE_PX * INNER_FRAC
    }
    fn size(&self) -> f32 {
        self.unit * GIZMO_SIZE_PX
    }
    fn stroke_w(&self) -> f32 {
        self.unit * STROKE_PX
    }

    /// Axis arrow/scale shaft `[start, end]` (world distance from origin along axis).
    /// Translate arrows sit OUTSIDE the rotation rings (radius `gs`) when rotate is
    /// shown; scale handles stay inside, so the three never overlap.
    fn axis_span(&self, kind: Kind) -> (f32, f32) {
        let gs = self.size();
        match kind {
            // Translate: beyond the ring when rotate is shown, else full from-center.
            Kind::Translate => {
                if self.show_rotate {
                    (gs * 1.18, gs * 1.55)
                } else {
                    (self.stroke_w() * 0.5 + self.inner(), gs)
                }
            }
            // Scale: inner handle (shorter when translate also shows, so its blunt
            // cap stays clear of the translate arrow / ring).
            _ => {
                let end = if self.both_ts { gs * 0.6 } else { gs };
                (self.stroke_w() * 0.5 + self.inner(), end)
            }
        }
    }

    fn plane_center(&self, a: u8, b: u8) -> Vec3 {
        self.origin + (self.axis(a) + self.axis(b)) * (self.size() * PLANE_OFFSET_FRAC)
    }
    fn plane_half(&self) -> f32 {
        self.size() * PLANE_SIZE_FRAC * 0.5 + self.stroke_w()
    }

    /// Tessellate this handle into filled overlay geometry.
    fn tessellate(&self, sb: &ShapeBuilder, color: Color) -> GizmoMesh {
        let sw = STROKE_PX;
        match self.id {
            HandleId::TranslateAxis(a) | HandleId::ScaleAxis(a) => {
                let axis = self.axis(a);
                if self.axis_fade(axis) <= 0.0 {
                    return GizmoMesh::default();
                }
                let kind = self.id.kind();
                let (s, e) = self.axis_span(kind);
                let start = self.origin + axis * s;
                let end = self.origin + axis * e;
                // Thin shaft + cap. Scale → blunt fat segment; translate → tall cone.
                let cap_w = 2.4 * sw;
                // Translate cone is longer than the scale cap so the arrowhead reads
                // as a tall point rather than a stout nub.
                let tip_len = self.unit * cap_w * if kind == Kind::Scale { 1.0 } else { 2.2 };
                let tip_start = end - axis * tip_len;
                let mut m = sb.line(start, tip_start, sw, color);
                if kind == Kind::Scale {
                    m += &sb.line(tip_start, end, cap_w, color);
                } else {
                    m += &sb.arrow(tip_start, end, cap_w, color);
                }
                m
            }
            HandleId::TranslatePlane(a, b) | HandleId::ScalePlane(a, b) => {
                let center = self.plane_center(a, b);
                let h = self.plane_half();
                let (ua, ub) = (self.axis(a) * h, self.axis(b) * h);
                sb.quad(
                    [
                        center - ua - ub,
                        center + ua - ub,
                        center + ua + ub,
                        center - ua + ub,
                    ],
                    color,
                )
            }
            HandleId::TranslateView | HandleId::ScaleView => {
                // Screen-facing square at the centre.
                let h = self.inner();
                let (r, u) = (self.view_right * h, self.view_up * h);
                let o = self.origin;
                sb.quad([o - r - u, o + r - u, o + r + u, o - r + u], color)
            }
            HandleId::RotateAxis(a) => sb.ring(self.origin, self.axis(a), self.size(), sw, color),
            HandleId::RotateView => sb.ring(
                self.origin,
                self.view_fwd,
                self.size() * VIEW_RING_FRAC,
                sw,
                color,
            ),
        }
    }

    /// Filled pie sector showing the swept angle during a rotation drag, from
    /// `start_angle` (the drag-start angle, same `plane_basis` convention as
    /// `drag_param`) sweeping `sweep` radians. A faint fill on top of the ring.
    fn tessellate_sector(
        &self,
        sb: &ShapeBuilder,
        start_angle: f32,
        sweep: f32,
        color: Color,
    ) -> GizmoMesh {
        let (normal, radius) = match self.id {
            HandleId::RotateAxis(a) => (self.axis(a), self.size()),
            HandleId::RotateView => (self.view_fwd, self.size() * VIEW_RING_FRAC),
            _ => return GizmoMesh::default(),
        };
        sb.sector(
            self.origin,
            normal,
            radius,
            start_angle,
            sweep,
            color.with_alpha(0.25),
        )
    }

    /// Axis visibility (1 = head-on, 0 = edge-on/parallel to view), so axes nearly
    /// parallel to the camera fade out and don't steal clicks.
    fn axis_fade(&self, axis: Vec3) -> f32 {
        let dot = self.view_fwd.dot(axis).abs();
        (1.0 - (dot - FADE_START) / (FADE_END - FADE_START)).clamp(0.0, 1.0)
    }

    /// Analytic pick: ray parameter `t` of the hit, or `None`. `tol` is world-space.
    fn pick(&self, ray: &Ray, tol: f32) -> Option<f32> {
        match self.id {
            HandleId::TranslateAxis(a) | HandleId::ScaleAxis(a) => {
                let axis = self.axis(a);
                if self.axis_fade(axis) <= 0.0 {
                    return None;
                }
                let (s, e) = self.axis_span(self.id.kind());
                let (rt, seg_t, dist) =
                    segment_to_ray(self.origin + axis * s, self.origin + axis * e, ray);
                (dist <= tol && (0.0..=1.0).contains(&seg_t)).then_some(rt)
            }
            HandleId::TranslatePlane(a, b) | HandleId::ScalePlane(a, b) => {
                let normal = self.axis(3 - a - b);
                let center = self.plane_center(a, b);
                let hit = ray_plane(ray, center, normal)?;
                let d = hit.point - center;
                let h = self.plane_half();
                (d.dot(self.axis(a)).abs() <= h && d.dot(self.axis(b)).abs() <= h).then_some(hit.t)
            }
            HandleId::TranslateView | HandleId::ScaleView => {
                let hit = ray_plane(ray, self.origin, self.view_fwd)?;
                let d = hit.point - self.origin;
                let h = self.inner();
                (d.dot(self.view_right).abs() <= h && d.dot(self.view_up).abs() <= h)
                    .then_some(hit.t)
            }
            HandleId::RotateAxis(a) => {
                let hit = ray_plane(ray, self.origin, self.axis(a))?;
                let r = (hit.point - self.origin).length();
                ((r - self.size()).abs() <= tol).then_some(hit.t)
            }
            HandleId::RotateView => {
                let hit = ray_plane(ray, self.origin, self.view_fwd)?;
                let r = (hit.point - self.origin).length();
                ((r - self.size() * VIEW_RING_FRAC).abs() <= tol).then_some(hit.t)
            }
        }
    }

    /// Primary drag axis (axis handles / rotation axis / plane normal).
    fn drag_axis(&self) -> Vec3 {
        match self.id {
            HandleId::TranslateAxis(a) | HandleId::ScaleAxis(a) => self.axis(a),
            HandleId::RotateAxis(a) => self.axis(a),
            HandleId::RotateView => self.view_fwd,
            HandleId::TranslatePlane(a, b) | HandleId::ScalePlane(a, b) => self.axis(3 - a - b),
            HandleId::TranslateView | HandleId::ScaleView => self.view_fwd,
        }
    }

    /// Secondary in-plane axis (plane/view handles); else ZERO.
    fn drag_axis2(&self) -> Vec3 {
        match self.id {
            HandleId::TranslatePlane(a, _) | HandleId::ScalePlane(a, _) => self.axis(a),
            HandleId::TranslateView | HandleId::ScaleView => self.view_right,
            _ => Vec3::ZERO,
        }
    }
}

/// Build the visible handles for the current state, sized to the camera distance.
fn build_handles(
    state: &GizmoState,
    origin: Vec3,
    target_rot: Quat,
    cam: &Transform,
    proj_y: f32,
    window_h: f32,
) -> Vec<Handle> {
    let dist = (origin - cam.translation).length().max(0.001);
    // world-per-pixel at the gizmo's depth: 2*dist*tan(fov/2)/height; proj_y=cot(fov/2).
    let unit = (2.0 * dist / proj_y) / window_h;

    let basis = match state.orientation {
        Orientation::World => AXES,
        Orientation::Local => [
            target_rot * Vec3::X,
            target_rot * Vec3::Y,
            target_rot * Vec3::Z,
        ],
    };
    let fwd = cam.forward().as_vec3();
    let view_right = cam.right().as_vec3();
    let view_up = cam.up().as_vec3();
    let both_ts = state.modes.translate && state.modes.scale;
    let show_rotate = state.modes.rotate;

    let mk = |id: HandleId| Handle {
        id,
        origin,
        basis,
        unit,
        view_right,
        view_up,
        view_fwd: fwd,
        both_ts,
        show_rotate,
    };

    let mut handles = Vec::with_capacity(20);
    let mut push = |id: HandleId| {
        if id.shown(state.modes) {
            handles.push(mk(id));
        }
    };

    for a in 0..3u8 {
        push(HandleId::TranslateAxis(a));
        push(HandleId::RotateAxis(a));
        push(HandleId::ScaleAxis(a));
    }
    for (a, b) in PLANE_PAIRS {
        push(HandleId::TranslatePlane(a, b));
        push(HandleId::ScalePlane(a, b));
    }
    push(HandleId::TranslateView);
    push(HandleId::ScaleView);
    push(HandleId::RotateView);
    handles
}

/// Per-frame gizmo interaction: hover, begin/continue/end drag, claim the click.
/// Runs in `Last` before `sdf_picking`.
#[allow(clippy::too_many_arguments)]
pub fn gizmo_update(
    mouse: Res<ButtonInput<MouseButton>>,
    mut state: ResMut<GizmoState>,
    selection: Res<SdfSelection>,
    mut sync_bake: ResMut<SyncBakeRequest>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &Transform), With<SdfCamera>>,
    mut volumes: Query<&mut Transform, (With<SdfVolume>, Without<SdfCamera>)>,
) {
    state.claimed_click = false;

    if !mouse.pressed(MouseButton::Left) {
        state.drag = None;
    }

    let Some(entity) = selection.entity else {
        state.hovered = None;
        return;
    };
    let (Ok(window), Ok((camera, cam_xf))) = (windows.single(), cameras.single()) else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Some(ray) = mouse_to_ray(camera, cam_xf, window, cursor) else {
        return;
    };
    let Ok(target_xf) = volumes.get(entity).copied() else {
        return;
    };

    let proj_y = camera.clip_from_view().y_axis.y;
    let handles = build_handles(
        &state,
        target_xf.translation,
        target_xf.rotation,
        cam_xf,
        proj_y,
        window.height(),
    );

    // Continue an active drag.
    if let Some(drag) = state.drag.take() {
        if let Ok(mut t) = volumes.get_mut(entity) {
            // Mutating Transform fires `Changed<Transform>`, which `schedule_bakes`
            // uses to rebake just the affected chunks — no explicit dirty flag needed.
            apply_drag(&drag, &ray, &mut t, &state);
            // Bake the touched chunks this frame so the volume tracks the cursor live;
            // the async path would otherwise lose every frame's result to the epoch
            // race and not show until release.
            sync_bake.0 = true;
        }
        state.hovered = Some(drag.id);
        state.drag = Some(drag);
        return;
    }

    let dist = (target_xf.translation - cam_xf.translation)
        .length()
        .max(0.001);
    let tol = (2.0 * dist / proj_y) / window.height() * FOCUS_PX;

    // Hover: nearest handle the ray hits.
    let mut best: Option<(f32, usize)> = None;
    for (i, h) in handles.iter().enumerate() {
        if let Some(t) = h.pick(&ray, tol)
            && best.is_none_or(|(bt, _)| t < bt)
        {
            best = Some((t, i));
        }
    }
    state.hovered = best.map(|(_, i)| handles[i].id);

    // Begin drag on press over a handle → claim the click.
    if mouse.just_pressed(MouseButton::Left)
        && let Some((_, i)) = best
    {
        let h = &handles[i];
        let axis = h.drag_axis();
        let axis2 = h.drag_axis2();
        let start_point = drag_start_point(h, &ray, axis, axis2);
        state.drag = Some(DragState {
            id: h.id,
            axis,
            axis2,
            start_proj: drag_param(h.id, &ray, target_xf.translation, axis),
            start_point,
            start_xf: target_xf,
        });
        state.claimed_click = true;
    }
}

/// Tessellate the visible handles into [`GizmoDraw`] each frame.
pub fn draw_gizmo(
    mut draw: ResMut<GizmoDraw>,
    state: Res<GizmoState>,
    selection: Res<SdfSelection>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &Transform), With<SdfCamera>>,
    volumes: Query<&Transform, With<SdfVolume>>,
) {
    draw.0.clear();

    let Some(entity) = selection.entity else {
        return;
    };
    let (Ok(window), Ok((camera, cam_xf))) = (windows.single(), cameras.single()) else {
        return;
    };
    let Ok(target_xf) = volumes.get(entity) else {
        return;
    };
    let active = state.drag.as_ref().map(|d| d.id).or(state.hovered);

    let proj_y = camera.clip_from_view().y_axis.y;
    let handles = build_handles(
        &state,
        target_xf.translation,
        target_xf.rotation,
        cam_xf,
        proj_y,
        window.height(),
    );

    let view_proj = camera.clip_from_view() * cam_xf.to_matrix().inverse();
    let sb = ShapeBuilder::new(
        view_proj,
        Vec2::new(window.width(), window.height()),
        window.scale_factor(),
    );

    let mut mesh = GizmoMesh::default();
    // While dragging, show ONLY the active handle (the plugin's focus behaviour); a
    // rotation drag also draws a filled sector of the swept angle.
    if let Some(drag) = &state.drag {
        if let Some(h) = handles.iter().find(|h| h.id == drag.id) {
            let color = handle_color(h.id, active);
            mesh += &h.tessellate(&sb, color);
            if let HandleId::RotateAxis(_) | HandleId::RotateView = h.id {
                let cur = drag_param(
                    drag.id,
                    &cursor_ray(camera, cam_xf, window),
                    drag.start_xf.translation,
                    drag.axis,
                );
                mesh += &h.tessellate_sector(&sb, drag.start_proj, cur - drag.start_proj, color);
            }
        }
    } else {
        for h in &handles {
            mesh += &h.tessellate(&sb, handle_color(h.id, active));
        }
    }
    draw.0 = mesh;
}

/// Build the mouse ray for the draw step (cursor may be absent → degenerate ray
/// that yields no sector).
fn cursor_ray(camera: &Camera, cam_xf: &Transform, window: &Window) -> Ray {
    window
        .cursor_position()
        .and_then(|c| mouse_to_ray(camera, cam_xf, window, c))
        .unwrap_or(Ray {
            origin: Vec3::ZERO,
            direction: Vec3::Z,
        })
}

/// World point under the cursor at drag start (for plane/view drags); for axis
/// drags this is the closest point on the axis.
fn drag_start_point(h: &Handle, ray: &Ray, axis: Vec3, axis2: Vec3) -> Vec3 {
    match h.id {
        HandleId::TranslatePlane(..)
        | HandleId::ScalePlane(..)
        | HandleId::TranslateView
        | HandleId::ScaleView => ray_plane(ray, h.origin, axis)
            .map(|p| p.point)
            .unwrap_or(h.origin),
        _ => {
            let _ = axis2;
            let t = project_onto_axis(ray, h.origin, axis);
            h.origin + axis * t
        }
    }
}

/// Apply a drag to the target transform, with optional snapping.
fn apply_drag(drag: &DragState, ray: &Ray, transform: &mut Transform, state: &GizmoState) {
    let start = &drag.start_xf;
    match drag.id {
        HandleId::TranslateAxis(_) => {
            let cur = project_onto_axis(ray, start.translation, drag.axis);
            let mut d = cur - drag.start_proj;
            if state.snap && state.snap_move > 0.0 {
                d = (d / state.snap_move).round() * state.snap_move;
            }
            transform.translation = start.translation + drag.axis * d;
        }
        HandleId::TranslatePlane(..) | HandleId::TranslateView => {
            if let Some(hit) = ray_plane(ray, drag.start_point, drag.axis) {
                let mut delta = hit.point - drag.start_point;
                if state.snap && state.snap_move > 0.0 {
                    // Snap each in-plane component.
                    let (u, v) = (drag.axis2, drag.axis.cross(drag.axis2).normalize());
                    let du = (delta.dot(u) / state.snap_move).round() * state.snap_move;
                    let dv = (delta.dot(v) / state.snap_move).round() * state.snap_move;
                    delta = u * du + v * dv;
                }
                transform.translation = start.translation + delta;
            }
        }
        HandleId::RotateAxis(a) => {
            let cur = drag_param(drag.id, ray, start.translation, drag.axis);
            let mut ang = cur - drag.start_proj;
            if state.snap && state.snap_angle > 0.0 {
                ang = (ang / state.snap_angle).round() * state.snap_angle;
            }
            transform.rotation = Quat::from_axis_angle(AXES[a as usize], ang) * start.rotation;
        }
        HandleId::RotateView => {
            let cur = drag_param(drag.id, ray, start.translation, drag.axis);
            let mut ang = cur - drag.start_proj;
            if state.snap && state.snap_angle > 0.0 {
                ang = (ang / state.snap_angle).round() * state.snap_angle;
            }
            transform.rotation = Quat::from_axis_angle(drag.axis, ang) * start.rotation;
        }
        HandleId::ScaleAxis(a) => {
            let cur = project_onto_axis(ray, start.translation, drag.axis);
            let mut delta = cur - drag.start_proj;
            if state.snap && state.snap_scale > 0.0 {
                delta = (delta / state.snap_scale).round() * state.snap_scale;
            }
            let factor = (1.0 + delta).max(0.01);
            let mut s = start.scale;
            s[a as usize] = start.scale[a as usize] * factor;
            transform.scale = s;
        }
        HandleId::ScalePlane(a, b) => {
            let cur = project_onto_axis(ray, start.translation, drag.axis2.normalize_or_zero());
            let mut delta = cur - drag.start_proj;
            if state.snap && state.snap_scale > 0.0 {
                delta = (delta / state.snap_scale).round() * state.snap_scale;
            }
            let factor = (1.0 + delta).max(0.01);
            let mut s = start.scale;
            s[a as usize] = start.scale[a as usize] * factor;
            s[b as usize] = start.scale[b as usize] * factor;
            transform.scale = s;
        }
        HandleId::ScaleView => {
            // Uniform scale by cursor distance from the gizmo centre.
            if let Some(hit) = ray_plane(ray, start.translation, drag.axis) {
                let mut delta = (hit.point - drag.start_point).length()
                    * (hit.point - start.translation)
                        .dot(drag.start_point - start.translation)
                        .signum();
                if state.snap && state.snap_scale > 0.0 {
                    delta = (delta / state.snap_scale).round() * state.snap_scale;
                }
                let factor = (1.0 + delta).max(0.01);
                transform.scale = start.scale * factor;
            }
        }
    }
}

/// Drag start parameter: angle (rotate) or axis distance (translate/scale).
fn drag_param(id: HandleId, ray: &Ray, origin: Vec3, axis: Vec3) -> f32 {
    match id {
        HandleId::RotateAxis(_) | HandleId::RotateView => match ray_plane(ray, origin, axis) {
            Some(hit) => {
                let v = hit.point - origin;
                let (u, w) = plane_basis(axis);
                v.dot(w).atan2(v.dot(u))
            }
            None => 0.0,
        },
        _ => project_onto_axis(ray, origin, axis),
    }
}

/// Blender-ish handle colours: X red, Y green, Z blue; planes blend their axes;
/// view handles white/grey; active/hover → near-white.
fn handle_color(id: HandleId, active: Option<HandleId>) -> Color {
    let axis_rgb = |a: u8| match a {
        0 => Srgba::rgb(1.0, 0.33, 0.40),
        1 => Srgba::rgb(0.55, 0.92, 0.20),
        _ => Srgba::rgb(0.17, 0.46, 1.0),
    };
    let base = match id {
        HandleId::TranslateAxis(a) | HandleId::RotateAxis(a) | HandleId::ScaleAxis(a) => {
            axis_rgb(a)
        }
        HandleId::TranslatePlane(a, b) | HandleId::ScalePlane(a, b) => {
            axis_rgb(a).mix(&axis_rgb(b), 0.5)
        }
        HandleId::TranslateView | HandleId::ScaleView | HandleId::RotateView => {
            Srgba::rgb(0.85, 0.85, 0.85)
        }
    };
    if active == Some(id) {
        base.mix(&Srgba::rgb(1.0, 1.0, 1.0), 0.6).into()
    } else {
        base.into()
    }
}

// --- geometry helpers ---

struct PlaneHit {
    point: Vec3,
    t: f32,
}

fn ray_plane(ray: &Ray, p: Vec3, n: Vec3) -> Option<PlaneHit> {
    let denom = ray.direction.dot(n);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (p - ray.origin).dot(n) / denom;
    (t >= 0.0).then(|| PlaneHit {
        point: ray.origin + ray.direction * t,
        t,
    })
}

fn plane_basis(n: Vec3) -> (Vec3, Vec3) {
    let a = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = a.cross(n).normalize();
    (u, n.cross(u).normalize())
}

/// Closest approach between the mouse ray and a finite segment `[a, b]`. Returns
/// `(ray_t, seg_t, distance)` with `seg_t ∈ [0,1]`.
fn segment_to_ray(a: Vec3, b: Vec3, ray: &Ray) -> (f32, f32, f32) {
    let d1 = ray.direction;
    let d2 = b - a;
    let r = ray.origin - a;
    let a11 = d1.dot(d1);
    let a12 = d1.dot(d2);
    let a22 = d2.dot(d2);
    let b1 = d1.dot(r);
    let b2 = d2.dot(r);
    let denom = a11 * a22 - a12 * a12;
    let (mut s, mut t);
    if denom.abs() < 1e-6 {
        s = 0.0;
        t = 0.0;
    } else {
        s = (a12 * b2 - a22 * b1) / denom;
        t = (a11 * b2 - a12 * b1) / denom;
    }
    s = s.max(0.0);
    t = t.clamp(0.0, 1.0);
    let p_ray = ray.origin + d1 * s;
    let p_seg = a + d2 * t;
    (s, t, (p_ray - p_seg).length())
}

/// Closest-point parameter of the mouse ray on an infinite axis line.
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
