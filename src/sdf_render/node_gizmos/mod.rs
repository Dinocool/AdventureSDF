//! Per-node-type editor gizmos. Each node kind (directional light, point light, empty-node
//! axes) owns its own module with its drawing, pick bounds, and any interaction system, so
//! adding a kind is one new file + one match arm here — not edits scattered across the
//! codebase. The data marker [`EditorGizmo`](crate::node::EditorGizmo) stays in `node`; this
//! module holds the logic (it needs `SdfNodeGizmos`, `SdfCamera`, and the picker, all here).
//!
//! Shared drawing primitives live in [`draw`]; kinds with a drag interaction (point light)
//! ship their own system, since interaction needs component access a uniform trait can't
//! express.

use bevy::prelude::*;

use crate::node::EditorGizmo;
use crate::sdf_render::{SdfCamera, SdfNodeGizmos, SdfOverlayGizmos};

pub mod axes;
pub mod camera;
pub mod directional_light;
pub mod draw;
pub mod point_light;

/// Everything a kind's `draw` needs, resolved once per node by the dispatcher.
pub struct NodeGizmoCtx<'a> {
    pub origin: Vec3,
    pub rotation: Quat,
    /// The variant's base `scale`.
    pub scale: f32,
    /// The entity's `PointLight` falloff cutoff + physical light size, if it has a
    /// `PointLight` (point lights only). `(range, radius)`.
    pub light: Option<(f32, f32)>,
    /// Whether this node is the current editor selection. Drives whether a manipulator
    /// (e.g. the point-light rings) draws always-on-top with handles, or depth-tested
    /// in-scene without them.
    pub selected: bool,
    /// Camera (camera, transform); `None` if the scene has no `SdfCamera` this frame.
    pub camera: Option<(&'a Camera, &'a Transform)>,
    pub window_h: f32,
}

/// The two gizmo groups a kind can draw into: `scene` is depth-respecting (locator glyphs
/// occlude behind geometry); `overlay` is always-on-top (manipulator rings/handles stay
/// grabbable regardless of what's in front, like the transform handles). Each `Gizmos`
/// borrow gets its own lifetimes (they are distinct system params).
pub struct NodeGizmoPainter<'a, 's1, 'w1, 's2, 'w2> {
    pub scene: &'a mut Gizmos<'s1, 'w1, SdfNodeGizmos>,
    pub overlay: &'a mut Gizmos<'s2, 'w2, SdfOverlayGizmos>,
}

/// Local-space oriented pick bounds `(center, half_extents)` of a gizmo's glyph, in the
/// node's own frame. `sdf_picking` turns this into a world OBB for selecting the node.
pub fn pick_bounds(gizmo: &EditorGizmo) -> (Vec3, Vec3) {
    match *gizmo {
        EditorGizmo::DirectionalLight { scale } => directional_light::pick_bounds(scale),
        EditorGizmo::PointLight { scale } => point_light::pick_bounds(scale),
        EditorGizmo::Camera { scale } => camera::pick_bounds(scale),
        EditorGizmo::Axes { scale } => axes::pick_bounds(scale),
    }
}

/// Dispatch a single node's draw to its kind.
fn draw_one(gizmo: &EditorGizmo, ctx: &NodeGizmoCtx, painter: &mut NodeGizmoPainter) {
    match gizmo {
        EditorGizmo::DirectionalLight { .. } => directional_light::draw(ctx, painter),
        EditorGizmo::PointLight { .. } => point_light::draw(ctx, painter),
        EditorGizmo::Camera { .. } => camera::draw(ctx, painter),
        EditorGizmo::Axes { .. } => axes::draw(ctx, painter),
    }
}

/// Draw every node's editor gizmo. Locator glyphs go into the depth-respecting
/// `SdfNodeGizmos` group; always-on-top manipulator elements into `SdfOverlayGizmos`.
fn draw_node_gizmos(
    mut scene: Gizmos<SdfNodeGizmos>,
    mut overlay: Gizmos<SdfOverlayGizmos>,
    nodes: Query<(Entity, &GlobalTransform, &EditorGizmo, Option<&PointLight>)>,
    cameras: Query<(&Camera, &Transform), With<SdfCamera>>,
    windows: Query<&Window>,
    selection: Res<crate::sdf_render::SdfSelection>,
) {
    let camera = cameras.single().ok();
    let window_h = windows.single().map(|w| w.height()).unwrap_or(1080.0);
    let mut painter = NodeGizmoPainter {
        scene: &mut scene,
        overlay: &mut overlay,
    };
    for (entity, xf, gizmo, point_light) in &nodes {
        let scale = match *gizmo {
            EditorGizmo::DirectionalLight { scale }
            | EditorGizmo::PointLight { scale }
            | EditorGizmo::Camera { scale }
            | EditorGizmo::Axes { scale } => scale,
        };
        let ctx = NodeGizmoCtx {
            origin: xf.translation(),
            rotation: xf.rotation(),
            scale,
            light: point_light.map(|l| (l.range, l.radius)),
            selected: selection.entity == Some(entity),
            camera,
            window_h,
        };
        draw_one(gizmo, &ctx, &mut painter);
    }
}

/// Register the node-gizmo draw system + each kind's own interaction systems.
pub fn register(app: &mut App) {
    use crate::scene_manager::AppScene;
    app.add_systems(
        Update,
        draw_node_gizmos.run_if(in_state(AppScene::SdfEditor)),
    );
    point_light::register(app);
}
