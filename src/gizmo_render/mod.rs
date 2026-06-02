//! Reusable filled, antialiased **2D-overlay** gizmo renderer.
//!
//! Ported from `transform-gizmo-bevy`'s render layer: gizmo geometry is tessellated
//! (via egui's `epaint`) into filled triangles in screen space, converted to NDC,
//! and drawn as a flat top-most overlay through a tiny shader in the `Transparent3d`
//! phase. This gives the crisp solid look Bevy's line-only immediate-mode `Gizmos`
//! can't.
//!
//! Generic + immediate-mode: any system fills the [`GizmoDraw`] resource each frame
//! (via [`shapes::ShapeBuilder`]); the overlay draws it on the camera tagged
//! [`GizmoCamera`]. The SDF editor's gizmo is just one producer.

use std::ops::AddAssign;

use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};

pub mod render;
pub mod shapes;

pub use shapes::ShapeBuilder;

/// A batch of filled triangles in **normalized device coordinates** (x,y ∈ [-1,1]),
/// with per-vertex linear-RGBA colors and triangle indices. Built by
/// [`ShapeBuilder`] and accumulated into [`GizmoDraw`].
#[derive(Clone, Default)]
pub struct GizmoMesh {
    pub vertices: Vec<[f32; 2]>,
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
}

impl GizmoMesh {
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.colors.clear();
        self.indices.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Append another mesh, offsetting its indices by the current vertex count.
    pub fn append(&mut self, other: &GizmoMesh) {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.colors.extend_from_slice(&other.colors);
        self.indices.extend(other.indices.iter().map(|i| i + base));
    }
}

impl AddAssign<&GizmoMesh> for GizmoMesh {
    fn add_assign(&mut self, rhs: &GizmoMesh) {
        self.append(rhs);
    }
}

/// Per-frame submit target. A producer clears it then appends shapes; the render
/// layer extracts and draws it. Replacing the contents each frame = immediate mode.
#[derive(Resource, Default, Clone, ExtractResource)]
pub struct GizmoDraw(pub GizmoMesh);

/// Marks the camera the gizmo overlay draws onto. Extracted to the render world so
/// the queue step can find the overlay view.
#[derive(Component, Default, Clone, Copy, Reflect, ExtractComponent)]
#[reflect(Component)]
pub struct GizmoCamera;

/// Registers the overlay renderer. Core (not feature-gated).
pub struct GizmoRenderPlugin;

impl Plugin for GizmoRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GizmoDraw>()
            .register_type::<GizmoCamera>()
            .add_plugins((
                ExtractResourcePlugin::<GizmoDraw>::default(),
                ExtractComponentPlugin::<GizmoCamera>::default(),
            ));

        render::build_render(app);
    }

    fn finish(&self, app: &mut App) {
        render::finish_render(app);
    }
}
