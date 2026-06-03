//! In-editor scene file-browser modals (File→Open / File→Save As). Both navigate the
//! project `assets/` tree; Open picks an existing `.scene` to load, Save As picks a
//! directory + filename to write. Neither performs I/O directly — they set
//! [`EditorRequests`] flags (`open` / `save_as`) that the `soul_scene` systems drain.

use std::path::{Path, PathBuf};

use bevy::prelude::*;
use bevy_egui::egui;

use super::fs_util::{file_name_str, read_sorted};
use super::menu_bar::EditorRequests;

/// Where the editor keeps its scenes; the browser opens here.
pub const SCENES_ROOT: &str = "assets/scenes";
/// The browser won't navigate above this — scenes and their dependencies live under it.
pub const BROWSE_ROOT: &str = "assets";

/// State for the modal "Open Scene" file browser. `open` toggles visibility; `dir` is the
/// directory currently being browsed (always within [`BROWSE_ROOT`]).
#[derive(Resource)]
pub struct OpenSceneDialog {
    pub open: bool,
    pub dir: PathBuf,
}

impl Default for OpenSceneDialog {
    fn default() -> Self {
        Self {
            open: false,
            dir: PathBuf::from(SCENES_ROOT),
        }
    }
}

impl OpenSceneDialog {
    /// Show the browser, starting from `start` if it's a readable directory, else the
    /// scenes root. Used by the File→Open menu item.
    pub fn show_at(&mut self, start: &Path) {
        self.dir = if start.is_dir() {
            start.to_path_buf()
        } else {
            PathBuf::from(SCENES_ROOT)
        };
        self.open = true;
    }
}

/// True for files this editor's scene loader can open (the soul `*.scene` format).
fn is_scene_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("scene"))
}

/// Whether the browser may navigate above `dir` — only while still strictly inside the
/// browse root, so the user can't escape `assets/`.
fn can_go_up(dir: &Path) -> bool {
    dir != Path::new(BROWSE_ROOT) && dir.starts_with(BROWSE_ROOT) && dir.parent().is_some()
}

/// Resolve `dir` + a user-typed `name` into a destination path, forcing a `.scene`
/// extension so saves always land in the loadable format.
fn scene_dest(dir: &Path, name: &str) -> PathBuf {
    let mut p = dir.join(name.trim());
    if !p.extension().is_some_and(|e| e.eq_ignore_ascii_case("scene")) {
        p.set_extension("scene");
    }
    p
}

/// The shared body of both scene modals: the up/breadcrumb row, the subdirectory buttons, and the
/// scrollable `.scene` list. Sets `navigate_to` on an up/dir click; calls `on_scene_click(name, path)`
/// when a scene row is clicked. `selected_name` highlights the matching row (Save's overwrite target;
/// `None` for Open). `scroll_max_height` caps the list so a footer fits (Save). The two
/// `*_dialog_ui` fns supply only the window chrome + their distinct footer.
#[allow(clippy::too_many_arguments)]
fn file_browser_body(
    ui: &mut egui::Ui,
    dir: &Path,
    dirs: &[PathBuf],
    scenes: &[PathBuf],
    navigate_to: &mut Option<PathBuf>,
    selected_name: Option<&str>,
    scroll_max_height: Option<f32>,
    mut on_scene_click: impl FnMut(&str, &Path),
) {
    let up_enabled = can_go_up(dir);
    ui.horizontal(|ui| {
        if ui
            .add_enabled(up_enabled, egui::Button::new("\u{2B11} Up"))
            .clicked()
            && let Some(parent) = dir.parent()
        {
            *navigate_to = Some(parent.to_path_buf());
        }
        ui.weak(dir.display().to_string());
    });
    ui.separator();

    let mut scroll = egui::ScrollArea::vertical();
    if let Some(h) = scroll_max_height {
        scroll = scroll.max_height(h);
    }
    scroll.show(ui, |ui| {
        for d in dirs {
            if ui
                .button(format!("\u{1F4C1} {}", file_name_str(d)))
                .clicked()
            {
                *navigate_to = Some(d.clone());
            }
        }
        if scenes.is_empty() && dirs.is_empty() {
            ui.weak("(empty)");
        }
        for s in scenes {
            let name = file_name_str(s);
            let selected = selected_name == Some(name.as_str());
            if ui
                .selectable_label(selected, format!("\u{1F4C4} {name}"))
                .clicked()
            {
                on_scene_click(&name, s);
            }
        }
    });
}

/// Render the modal scene browser when open, draining a pick into [`EditorRequests::open`].
/// No-op while the dialog is closed. Called from the dock each frame (after the menu bar).
pub fn open_scene_dialog_ui(world: &mut World, ctx: &egui::Context) {
    if !world.resource::<OpenSceneDialog>().open {
        return;
    }

    let dir = world.resource::<OpenSceneDialog>().dir.clone();
    let (dirs, files) = read_sorted(&dir);
    let scenes: Vec<PathBuf> = files.into_iter().filter(|p| is_scene_file(p)).collect();

    let mut keep_open = true;
    let mut navigate_to: Option<PathBuf> = None;
    let mut pick: Option<PathBuf> = None;

    egui::Window::new("Open Scene")
        .id(egui::Id::new("open_scene_dialog"))
        .collapsible(false)
        .resizable(true)
        .default_size([460.0, 420.0])
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .open(&mut keep_open)
        .show(ctx, |ui| {
            // Open: a scene-row click picks it (the shared body draws the listing).
            file_browser_body(ui, &dir, &dirs, &scenes, &mut navigate_to, None, None, |_name, path| {
                pick = Some(path.to_path_buf());
            });
        });

    // Apply navigation / selection / close after the UI closure releases its borrows.
    {
        let mut dialog = world.resource_mut::<OpenSceneDialog>();
        if let Some(next) = navigate_to {
            dialog.dir = next;
        }
        if pick.is_some() || !keep_open {
            dialog.open = false;
        }
    }

    if let Some(path) = pick {
        world.resource_mut::<EditorRequests>().open = Some(path);
    }
}

/// State for the modal "Save Scene As" browser. `open` toggles visibility, `dir` is the
/// directory being browsed, and `filename` is the editable destination name.
#[derive(Resource, Default)]
pub struct SaveSceneDialog {
    pub open: bool,
    pub dir: PathBuf,
    pub filename: String,
}

impl SaveSceneDialog {
    /// Show the dialog pre-filled from the `current` scene path: browse its directory with
    /// its filename as the default. Falls back to the scenes root for an unknown location.
    pub fn show_for(&mut self, current: &Path) {
        let parent = current.parent().filter(|p| p.is_dir());
        self.dir = parent.map_or_else(|| PathBuf::from(SCENES_ROOT), Path::to_path_buf);
        self.filename = file_name_str(current);
        if self.filename.is_empty() {
            self.filename = "untitled.scene".to_string();
        }
        self.open = true;
    }
}

/// Render the modal "Save As" browser when open, draining a confirmed destination into
/// [`EditorRequests::save_as`] (which the `soul_scene` save system writes + adopts as the
/// new current path). No-op while closed. Called from the dock each frame.
pub fn save_scene_dialog_ui(world: &mut World, ctx: &egui::Context) {
    if !world.resource::<SaveSceneDialog>().open {
        return;
    }

    let dir = world.resource::<SaveSceneDialog>().dir.clone();
    let mut filename = world.resource::<SaveSceneDialog>().filename.clone();
    let (dirs, files) = read_sorted(&dir);
    let scenes: Vec<PathBuf> = files.into_iter().filter(|p| is_scene_file(p)).collect();

    let mut keep_open = true;
    let mut navigate_to: Option<PathBuf> = None;
    let mut confirm = false;
    let mut cancel = false;

    egui::Window::new("Save Scene As")
        .id(egui::Id::new("save_scene_dialog"))
        .collapsible(false)
        .resizable(true)
        .default_size([460.0, 420.0])
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .open(&mut keep_open)
        .show(ctx, |ui| {
            // Save: a scene-row click fills the name field (to overwrite it); the current name is
            // highlighted. `sel` snapshots `filename` so the highlight read doesn't fight the
            // on-click write (which the shared body's callback performs).
            let sel = filename.clone();
            file_browser_body(
                ui,
                &dir,
                &dirs,
                &scenes,
                &mut navigate_to,
                Some(&sel),
                Some(280.0),
                |name, _path| filename = name.to_string(),
            );

            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Name:");
                ui.add(
                    egui::TextEdit::singleline(&mut filename)
                        .hint_text("scene name")
                        .desired_width(f32::INFINITY),
                );
            });

            let dest = scene_dest(&dir, &filename);
            ui.weak(format!("\u{2192} {}", dest.display()));
            ui.separator();
            ui.horizontal(|ui| {
                let valid = !filename.trim().is_empty();
                if ui
                    .add_enabled(valid, egui::Button::new("Save"))
                    .clicked()
                {
                    confirm = true;
                }
                if ui.button("Cancel").clicked() {
                    cancel = true;
                }
            });
        });

    let dest = scene_dest(&dir, &filename);
    let confirmed = confirm && !filename.trim().is_empty();

    {
        let mut dialog = world.resource_mut::<SaveSceneDialog>();
        if let Some(next) = navigate_to {
            dialog.dir = next;
        }
        dialog.filename = filename;
        if confirmed || cancel || !keep_open {
            dialog.open = false;
        }
    }

    if confirmed {
        world.resource_mut::<EditorRequests>().save_as = Some(dest);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_file_predicate_is_case_insensitive() {
        assert!(is_scene_file(Path::new("assets/scenes/gallery.scene")));
        assert!(is_scene_file(Path::new("X.SCENE")));
        assert!(!is_scene_file(Path::new("assets/scenes/world.scn.ron")));
        assert!(!is_scene_file(Path::new("a.ron")));
        assert!(!is_scene_file(Path::new("noext")));
    }

    #[test]
    fn scene_dest_forces_scene_extension() {
        let dir = Path::new("assets/scenes");
        assert_eq!(scene_dest(dir, "level1"), PathBuf::from("assets/scenes/level1.scene"));
        assert_eq!(scene_dest(dir, "level1.scene"), PathBuf::from("assets/scenes/level1.scene"));
        assert_eq!(scene_dest(dir, "LEVEL.SCENE"), PathBuf::from("assets/scenes/LEVEL.SCENE"));
        // Whitespace is trimmed; a foreign extension is replaced, not appended.
        assert_eq!(scene_dest(dir, "  boss  "), PathBuf::from("assets/scenes/boss.scene"));
        assert_eq!(scene_dest(dir, "old.bak"), PathBuf::from("assets/scenes/old.scene"));
    }

    #[test]
    fn can_go_up_is_clamped_to_browse_root() {
        assert!(!can_go_up(Path::new(BROWSE_ROOT)));
        assert!(can_go_up(Path::new("assets/scenes")));
        assert!(can_go_up(Path::new("assets/scenes/sub")));
        // Outside the browse root: refuse to climb.
        assert!(!can_go_up(Path::new("elsewhere/dir")));
    }

    #[test]
    fn save_show_for_prefills_dir_and_name() {
        let mut d = SaveSceneDialog::default();
        // Unknown directory → scenes root, but the filename still carries over.
        d.show_for(Path::new("no/such/dir/mylevel.scene"));
        assert!(d.open);
        assert_eq!(d.dir, PathBuf::from(SCENES_ROOT));
        assert_eq!(d.filename, "mylevel.scene");
    }

    #[test]
    fn show_at_falls_back_to_scenes_root_for_non_dir() {
        let mut d = OpenSceneDialog {
            open: false,
            ..Default::default()
        };
        d.show_at(Path::new("definitely/not/a/real/dir"));
        assert!(d.open);
        assert_eq!(d.dir, PathBuf::from(SCENES_ROOT));
    }
}
