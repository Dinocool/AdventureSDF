//! Godot-style universal node system. Every scene object is a [`SceneNode`] (the
//! base "Node": named + selectable, no spatial data). Spatial objects are
//! [`Node3D`], which adds a `Transform` on top of `SceneNode`. SDF volumes, lights,
//! and cameras *extend* `Node3D` via Bevy required components (`#[require(...)]`).
//!
//! Parent/child structure reuses Bevy's hierarchy: `ChildOf` (source of truth) and
//! the auto-managed `Children`, which also gives transform propagation for free.
//! The editor's Scene panel walks this hierarchy to draw the node tree.

use bevy::prelude::*;

/// Reflect custom-attribute marker: a component tagged `#[reflect(@HideFromInspector)]`
/// is skipped by the editor's generic component inspector. Lives in core (not the
/// feature-gated editor) so any component — including these node markers — can be tagged
/// without depending on editor code.
#[derive(Reflect)]
pub struct HideFromInspector;

/// The base "Node" (Godot's `Node`). A named, selectable scene object with **no**
/// spatial data. Named `SceneNode` because `Node` is taken by `bevy_ui`.
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct SceneNode;

/// A spatial node (Godot's `Node3D`): a [`SceneNode`] that also has a `Transform`
/// (and thus `GlobalTransform`, which `Transform` itself requires). Concrete object
/// types (SDF volumes, lights, cameras) require this.
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
#[require(SceneNode, Transform)]
pub struct Node3D;

/// Editor-only visual gizmo for a [`Node3D`]. Pure data: it names *what* to draw at
/// the node's transform; the actual immediate-mode drawing lives wherever a gizmo
/// group is available (see `sdf_render::draw_node_editor_gizmos`). Invisible objects
/// (lights, cameras, empty nodes) carry one so they're locatable and orientable in
/// the viewport. Never affects the runtime render — strictly an editor aid.
#[derive(Component, Reflect, Clone, Copy, Debug, PartialEq)]
#[reflect(Component)]
#[reflect(@HideFromInspector)]
pub enum EditorGizmo {
    /// A sun glyph + parallel rays along the node's forward (-Z), showing the light's
    /// travel direction. Sized in world units by `scale`.
    DirectionalLight { scale: f32 },
    /// A point-light glyph: a small central bulb plus a camera-facing ring of radius
    /// `PointLight.range` with a square handle on its edge for dragging the radius. The
    /// radius itself lives on the entity's Bevy `PointLight` (`range`); this variant
    /// carries only the bulb's base size.
    PointLight { scale: f32 },
    /// A wireframe frustum glyph pointing along the node's forward (-Z), for a scene
    /// [`SceneCamera`] node. Sized in world units by `scale`.
    Camera { scale: f32 },
    /// Three short colored axis lines (X red / Y green / Z blue) at the origin — a
    /// generic "empty"/locator marker for otherwise-invisible nodes.
    Axes { scale: f32 },
}

impl Default for EditorGizmo {
    fn default() -> Self {
        EditorGizmo::Axes { scale: 0.5 }
    }
}

/// A scene camera node: authored projection data (not an active render camera). Serialized
/// into a `.scene` and shown in the hierarchy with an [`EditorGizmo::Camera`] frustum. The
/// editor's viewport camera can "look through" it (snap the orbit pose to this node). A
/// future runtime could promote this to a real `Camera3d`.
#[derive(Component, Reflect, Clone, Copy, Debug, PartialEq)]
#[reflect(Component)]
#[require(Node3D)]
pub struct SceneCamera {
    /// Vertical field of view, radians.
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for SceneCamera {
    fn default() -> Self {
        Self {
            fov_y_radians: std::f32::consts::FRAC_PI_4, // 45°
            near: 0.1,
            far: 1000.0,
        }
    }
}

pub struct NodePlugin;

impl Plugin for NodePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<SceneNode>()
            .register_type::<Node3D>()
            .register_type::<EditorGizmo>()
            .register_type::<SceneCamera>()
            .register_type::<HideFromInspector>()
            // So the inspector can show a node's parent link; `Children` is managed
            // by Bevy and reflected by the engine.
            .register_type::<ChildOf>();
    }
}
