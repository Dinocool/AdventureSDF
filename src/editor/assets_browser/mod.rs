//! Assets browser dock tab (Godot FileSystem-style): navigate the project's
//! `assets/` tree and show each direct child of the current folder as a tile with a
//! thumbnail. Folders descend on click; a ↑ button and a clickable breadcrumb move
//! back up. Thumbnails are **modular** — each resource kind implements a
//! [`ThumbnailProvider`]; anything without one falls back to a type icon.
//!
//! Rendering can't happen inside an immediate-mode panel, so the heavy work (image
//! loading, offscreen material-sphere rendering) lives in Bevy systems and the panel
//! only reads the cached results. See [`thumbnail`].

use std::path::{Path, PathBuf};

use bevy::prelude::*;
use bevy_egui::egui;

pub mod thumbnail;

pub use thumbnail::{
    ImageThumbnailProvider, MaterialThumbnailProvider, PbrTextureThumbnailProvider,
    PendingSceneThumbnail, SceneThumbnailProvider, ThumbnailRenderPlugin,
};

/// Root the browser walks, relative to the working dir — matches how Bevy's
/// `AssetServer` resolves `assets/`.
pub const ASSETS_ROOT: &str = "assets";

/// egui drag-and-drop payload: a material file (working-dir `.material.ron` path) dragged
/// from the assets tray. Drop targets (viewport, inspector Material section) consume this
/// to set an entity's material. A distinct newtype so it never collides with the
/// hierarchy's `Entity` reparent payload.
#[derive(Clone)]
pub struct MaterialDrag(pub PathBuf);

/// Edge length (px) of a thumbnail tile's image area.
const TILE_PX: f32 = 72.0;
/// Total tile width (image + padding), governs how many fit per row.
const TILE_W: f32 = 84.0;

/// What a tile should display for an asset path.
pub enum Thumbnail {
    /// A ready egui texture (image preview or rendered material sphere).
    Texture(egui::TextureId),
    /// Fallback: an icon glyph drawn large.
    Icon(&'static str),
    /// Load/render still in flight — show a placeholder this frame.
    Pending,
}

/// Implemented per resource kind. `matches` claims a path (usually by extension);
/// `thumbnail` returns what to draw, kicking off any load/render as an idempotent
/// side effect. Adding a previewable type = one impl + registering it; the panel
/// never changes.
pub trait ThumbnailProvider: Send + Sync + 'static {
    /// Does this provider handle `path`?
    fn matches(&self, path: &Path) -> bool;
    /// Thumbnail for `path`, starting/continuing any async work as needed.
    fn thumbnail(&self, world: &mut World, path: &Path) -> Thumbnail;
}

/// Ordered list of thumbnail providers. The first whose `matches` is true renders a
/// path; otherwise the panel draws a generic icon by extension.
#[derive(Resource, Default)]
pub struct ThumbnailRegistry {
    providers: Vec<Box<dyn ThumbnailProvider>>,
}

impl ThumbnailRegistry {
    pub fn register(&mut self, provider: impl ThumbnailProvider) {
        self.providers.push(Box::new(provider));
    }
}

/// Current folder of the assets browser, relative to [`ASSETS_ROOT`] (empty = root).
/// Also the sync target the Project Files tree writes when a folder is clicked.
#[derive(Resource, Default)]
pub struct AssetsBrowserState {
    pub current: PathBuf,
}

impl AssetsBrowserState {
    /// Absolute (working-dir-relative) path of the current folder.
    fn abs(&self) -> PathBuf {
        Path::new(ASSETS_ROOT).join(&self.current)
    }
}

/// A direct child of the current folder.
struct Entry {
    path: PathBuf,
    name: String,
    is_dir: bool,
}

/// Render the assets browser panel.
pub fn assets_browser_ui(world: &mut World, ui: &mut egui::Ui) {
    let current = world.resource::<AssetsBrowserState>().current.clone();

    // --- Toolbar: up button + breadcrumb -----------------------------------------
    let mut nav_to: Option<PathBuf> = None;
    let mut select_asset: Option<PathBuf> = None;
    let mut open_scene: Option<PathBuf> = None;
    ui.horizontal(|ui| {
        let at_root = current.as_os_str().is_empty();
        if ui
            .add_enabled(!at_root, egui::Button::new("\u{2191}"))
            .on_hover_text("Up one folder")
            .clicked()
        {
            nav_to = Some(parent_of(&current));
        }
        ui.separator();
        // Breadcrumb: "assets" then each path segment, each clickable.
        if ui.selectable_label(at_root, "assets").clicked() {
            nav_to = Some(PathBuf::new());
        }
        let mut acc = PathBuf::new();
        for comp in current.components() {
            acc.push(comp);
            let seg = comp.as_os_str().to_string_lossy().into_owned();
            ui.label("/");
            let is_last = acc == current;
            if ui.selectable_label(is_last, seg).clicked() {
                nav_to = Some(acc.clone());
            }
        }
    });
    ui.separator();

    // --- Collect the current folder's children ------------------------------------
    let dir = world.resource::<AssetsBrowserState>().abs();
    let entries = read_entries(&dir);

    egui::ScrollArea::vertical().show(ui, |ui| {
        if entries.is_empty() {
            ui.weak("(empty folder)");
        }
        ui.horizontal_wrapped(|ui| {
            for entry in &entries {
                let resp = tile_ui(world, ui, entry);
                if entry.is_dir {
                    // Folders descend on click or double-click.
                    if resp.clicked() || resp.double_clicked() {
                        nav_to = rel_to_root(&entry.path);
                    }
                } else if is_scene_file(&entry.path) && resp.double_clicked() {
                    // Double-click a scene → open it (handled by the multi-scene tab manager).
                    open_scene = Some(entry.path.clone());
                } else if resp.clicked() {
                    // File click → select it in the unified inspector.
                    select_asset = Some(entry.path.clone());
                }
                // Material files are draggable onto scene entities / the inspector Material
                // section to set their material (see `MaterialDrag`). Mirror the hierarchy's
                // floating-preview drag pattern.
                if !entry.is_dir
                    && crate::editor::fs_util::is_material_ron(&entry.path)
                    && resp.dragged()
                {
                    resp.dnd_set_drag_payload(MaterialDrag(entry.path.clone()));
                    // Drag preview: a small floating copy of the material's thumbnail image
                    // (falls back to a glyph if its render isn't ready), tracking the cursor.
                    if let Some(pointer) = ui.ctx().pointer_interact_pos() {
                        const SIZE: f32 = 48.0;
                        let layer_id = egui::LayerId::new(
                            egui::Order::Tooltip,
                            resp.id.with("mat_drag_preview"),
                        );
                        let icon_rect =
                            egui::Rect::from_center_size(pointer + egui::vec2(SIZE * 0.5 + 8.0, 0.0), egui::vec2(SIZE, SIZE));
                        let painter = ui.ctx().layer_painter(layer_id);
                        match thumbnail_for_path(world, &entry.path) {
                            Thumbnail::Texture(tex) => {
                                painter.image(
                                    tex,
                                    icon_rect,
                                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                    egui::Color32::WHITE,
                                );
                            }
                            _ => {
                                painter.text(
                                    icon_rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    "◆",
                                    egui::FontId::proportional(SIZE * 0.7),
                                    egui::Color32::WHITE,
                                );
                            }
                        }
                    }
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                }
            }
        });
    });

    if let Some(target) = nav_to {
        world.resource_mut::<AssetsBrowserState>().current = target;
    }
    if let Some(path) = select_asset {
        world
            .resource_mut::<crate::editor::selection::EditorSelection>()
            .select_asset(path);
    }
    if let Some(path) = open_scene {
        // Routed through EditorRequests::open, drained by the scene-tab manager next frame
        // (opens a new tab, or focuses the scene if it's already open).
        world.resource_mut::<crate::editor::menu_bar::EditorRequests>().open = Some(path);
    }
}

/// What a shared tile should render: a thumbnail for an asset path (via the registry),
/// or a fixed icon glyph (folders, "(none)", etc).
pub enum TileThumb {
    /// Render the thumbnail the [`ThumbnailRegistry`] produces for this path.
    Path(PathBuf),
    /// Render this glyph directly (no registry lookup).
    Icon(&'static str),
}

/// Draw one fixed-size tile (thumbnail/icon + truncated label) and return its click
/// response. The shared tile renderer used by both the Assets tray and the resource
/// picker, so they look identical.
pub fn draw_tile(
    world: &mut World,
    ui: &mut egui::Ui,
    thumb: &TileThumb,
    label: &str,
    selected: bool,
) -> egui::Response {
    ui.allocate_ui_with_layout(
        egui::vec2(TILE_W, TILE_PX + 28.0),
        egui::Layout::top_down(egui::Align::Center),
        |ui| {
            let (rect, resp) = ui
                .allocate_exact_size(egui::vec2(TILE_PX, TILE_PX), egui::Sense::click_and_drag());
            let painter = ui.painter_at(rect);
            match thumb {
                TileThumb::Icon(glyph) => draw_glyph(&painter, rect, glyph),
                TileThumb::Path(path) => match thumbnail_for_path(world, path) {
                    Thumbnail::Texture(tex) => {
                        painter.image(
                            tex,
                            rect,
                            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                            egui::Color32::WHITE,
                        );
                    }
                    Thumbnail::Icon(glyph) => draw_glyph(&painter, rect, glyph),
                    Thumbnail::Pending => {
                        painter.text(
                            rect.center(),
                            egui::Align2::CENTER_CENTER,
                            "…",
                            egui::FontId::proportional(20.0),
                            ui.visuals().weak_text_color(),
                        );
                    }
                },
            }
            // Selection / hover frame.
            if selected {
                ui.painter().rect_stroke(
                    rect,
                    3.0,
                    ui.visuals().selection.stroke,
                    egui::StrokeKind::Inside,
                );
            } else if resp.hovered() {
                ui.painter().rect_stroke(
                    rect,
                    3.0,
                    ui.visuals().widgets.hovered.fg_stroke,
                    egui::StrokeKind::Inside,
                );
            }
            ui.add(egui::Label::new(elide(label, 12)).truncate())
                .on_hover_text(label);
            resp
        },
    )
    .inner
}

/// Resolve a path to its [`Thumbnail`] via the registry, else a generic icon by
/// extension. Self-contained: takes the registry out of the world for the dispatch and
/// restores it, so callers (tray + picker) don't manage that themselves.
pub fn thumbnail_for_path(world: &mut World, path: &Path) -> Thumbnail {
    let result = crate::editor::fs_util::with_registry(
        world,
        |world, registry: &ThumbnailRegistry| {
            for provider in &registry.providers {
                if provider.matches(path) {
                    return Some(provider.thumbnail(world, path));
                }
            }
            None
        },
    );
    result.unwrap_or_else(|| Thumbnail::Icon(icon_for(path)))
}

/// Draw one asset-tray tile (folder icon or file thumbnail) and return its response.
fn tile_ui(world: &mut World, ui: &mut egui::Ui, entry: &Entry) -> egui::Response {
    let thumb = if entry.is_dir {
        TileThumb::Icon("\u{1F4C1}")
    } else {
        TileThumb::Path(entry.path.clone())
    };
    draw_tile(world, ui, &thumb, &entry.name, false)
}

/// Generic icon glyph for a file with no provider, chosen by extension.
fn icon_for(path: &Path) -> &'static str {
    match ext_lower(path).as_deref() {
        Some("ron") => "\u{2699}",   // gear — data/resource
        Some("scene") => "\u{1F3AC}", // clapperboard — scene
        Some("wgsl") => "\u{1F4DC}", // scroll — shader
        _ => "\u{1F4C4}",            // generic file
    }
}

fn draw_glyph(painter: &egui::Painter, rect: egui::Rect, glyph: &str) {
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        glyph,
        egui::FontId::proportional(34.0),
        egui::Color32::from_gray(200),
    );
}

/// Direct children of `dir`: directories first, then files, each alphabetically. Reads via
/// `read_sorted_cached` (5s TTL) so the visible folder isn't `read_dir`'d every frame
/// inside the egui pass (perf roadmap E1).
fn read_entries(dir: &Path) -> Vec<Entry> {
    let (dirs, files) = crate::editor::fs_util::read_sorted_cached(dir);
    let mk = |path: PathBuf, is_dir: bool| Entry {
        name: crate::editor::fs_util::file_name_str(&path),
        path,
        is_dir,
    };
    dirs.into_iter()
        .map(|d| mk(d, true))
        .chain(files.into_iter().map(|f| mk(f, false)))
        .collect()
}

/// Lowercased final extension of `path`, if any.
fn ext_lower(path: &Path) -> Option<String> {
    path.extension().map(|e| e.to_string_lossy().to_lowercase())
}

/// Whether `path` is a soul `.scene` file (case-insensitive).
fn is_scene_file(path: &Path) -> bool {
    ext_lower(path).as_deref() == Some("scene")
}

/// Parent of a root-relative folder path (empty path = already root).
fn parent_of(current: &Path) -> PathBuf {
    current.parent().map(Path::to_path_buf).unwrap_or_default()
}

/// Convert a working-dir path under `assets/` back to a root-relative path.
fn rel_to_root(path: &Path) -> Option<PathBuf> {
    crate::editor::fs_util::relative_to_assets(path)
}

/// Truncate `s` to `max` chars with an ellipsis (egui also truncates visually; this
/// keeps the label compact).
fn elide(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{head}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parent_navigation_pops_one_segment() {
        assert_eq!(parent_of(Path::new("textures/cobble")), PathBuf::from("textures"));
        assert_eq!(parent_of(Path::new("textures")), PathBuf::new());
        assert_eq!(parent_of(Path::new("")), PathBuf::new());
    }

    #[test]
    fn rel_to_root_strips_assets_prefix() {
        assert_eq!(
            rel_to_root(Path::new("assets/textures/x")),
            Some(PathBuf::from("textures/x"))
        );
        assert_eq!(rel_to_root(Path::new("elsewhere/x")), None);
    }

    #[test]
    fn icon_for_known_extensions() {
        assert_eq!(icon_for(Path::new("a.ron")), "\u{2699}");
        assert_eq!(icon_for(Path::new("a.scene")), "\u{1F3AC}");
        assert_eq!(icon_for(Path::new("a.unknown")), "\u{1F4C4}");
    }

    #[test]
    fn elide_long_names() {
        assert_eq!(elide("short", 12), "short");
        assert_eq!(elide("a_very_long_filename", 6), "a_ver…");
    }
}
