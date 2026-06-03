pub mod assets;
pub mod camera;
pub mod combat;
pub mod gizmo_render;
pub mod instrument;
pub mod inventory;
pub mod networking;
pub mod node;
pub mod player;
pub mod scene_manager;
pub mod sdf_render;
pub mod soul_scene;
pub mod ui;
pub mod world;

#[cfg(feature = "editor")]
pub mod editor;

// Shared headless-app + spawn helpers for BOTH the in-crate unit tests and the `tests/` integration
// crate (which can't see `#[cfg(test)]` items). Compiled always — it's a small set of `pub` helpers,
// not pruned/warned, and the binary never calls them so they cost nothing at runtime.
pub mod test_utils;
