pub mod assets;
pub mod camera;
pub mod dev_flycam;
// Legacy WoW gameplay modules — removed in the voxel-RT rebuild (kept on disk, off compilation).
// pub mod combat;
pub mod gizmo_render;
pub mod instrument;
// pub mod inventory;
// pub mod networking;
pub mod node;
pub mod player;
pub mod scene_manager;
pub mod sdf_render;
pub mod soul_scene;
pub mod voxel;
// pub mod ui;
// pub mod world;

#[cfg(feature = "editor")]
pub mod editor;

// Dev-only FPS benchmark harness (Bistro-interior perf gate). Editor build only; entirely env-gated at runtime
// (see `bench::install_bistro_bench`, called from `main.rs` under ADVENTURE_BENCH_BISTRO).
#[cfg(feature = "editor")]
pub mod bench;

// Shared headless-app + spawn helpers for BOTH the in-crate unit tests and the `tests/` integration
// crate (which can't see `#[cfg(test)]` items). Compiled always — it's a small set of `pub` helpers,
// not pruned/warned, and the binary never calls them so they cost nothing at runtime.
pub mod test_utils;
