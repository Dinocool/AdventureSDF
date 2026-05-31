//! Godot-style universal node system. Every scene object is a [`SceneNode`] (the
//! base "Node": named + selectable, no spatial data). Spatial objects are
//! [`Node3D`], which adds a `Transform` on top of `SceneNode`. SDF volumes, lights,
//! and cameras *extend* `Node3D` via Bevy required components (`#[require(...)]`).
//!
//! Parent/child structure reuses Bevy's hierarchy: `ChildOf` (source of truth) and
//! the auto-managed `Children`, which also gives transform propagation for free.
//! The editor's Scene panel walks this hierarchy to draw the node tree.

use bevy::prelude::*;

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
pub enum EditorGizmo {
    /// A sun glyph + parallel rays along the node's forward (-Z), showing the light's
    /// travel direction. Sized in world units by `scale`.
    DirectionalLight { scale: f32 },
    /// Three short colored axis lines (X red / Y green / Z blue) at the origin — a
    /// generic "empty"/locator marker for otherwise-invisible nodes.
    Axes { scale: f32 },
}

impl Default for EditorGizmo {
    fn default() -> Self {
        EditorGizmo::Axes { scale: 0.5 }
    }
}

pub struct NodePlugin;

impl Plugin for NodePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<SceneNode>()
            .register_type::<Node3D>()
            .register_type::<EditorGizmo>()
            // So the inspector can show a node's parent link; `Children` is managed
            // by Bevy and reflected by the engine.
            .register_type::<ChildOf>();
    }
}
