//! Reusable resource picker: a button showing the current selection's thumbnail that
//! opens a **searchable grid of thumbnail tiles** (same rendering as the Assets tray,
//! via [`draw_tile`]/[`thumbnail_for_path`]). Used to pick materials and texture maps;
//! resources are opaque string keys paired with a [`TileThumb`] the thumbnail registry
//! knows how to render.

use bevy::prelude::*;
use bevy_egui::egui;

use crate::editor::assets_browser::{TileThumb, draw_tile};

/// One selectable resource in a picker.
pub struct PickerEntry {
    /// Opaque id returned when chosen (e.g. a material file path, or `"slug/dir"`).
    pub key: String,
    /// Display + filter text.
    pub label: String,
    /// What the tile renders.
    pub thumb: TileThumb,
}

/// The outcome of a pick this frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PickResult {
    /// A resource key was chosen.
    Key(String),
    /// The explicit "(none)" tile was chosen (only offered when `allow_none`).
    None,
}

/// Render a picker button (current selection's thumbnail + label) that opens a
/// searchable grid popup. Returns `Some(..)` only on the frame a choice is made. Open +
/// filter state live in egui memory keyed by `id`, so independent pickers coexist.
///
/// `entries` is built lazily — only invoked while the popup is open — so callers can
/// scan the filesystem without paying for it every frame.
pub fn resource_picker(
    world: &mut World,
    ui: &mut egui::Ui,
    id: egui::Id,
    current: Option<&PickerEntry>,
    allow_none: bool,
    entries: impl FnOnce() -> Vec<PickerEntry>,
) -> Option<PickResult> {
    let open_id = id.with("open");
    let filter_id = id.with("filter");
    let mut open: bool = ui.memory(|m| m.data.get_temp(open_id).unwrap_or(false));

    // --- Button: just the current selection's name (no thumbnail) ----------------
    // The thumbnail grid lives in the popup; inline we want a compact, instantly-stable
    // label (a tile here would flash "…" while its preview renders).
    let btn_label = match current {
        Some(e) => e.label.clone(),
        None => "(none)".to_string(),
    };
    if ui
        .button(format!("{btn_label}  \u{25BE}"))
        .clicked()
    {
        open = !open;
    }

    let mut result = None;
    if open {
        let mut keep_open = true;
        let mut filter: String = ui.memory(|m| m.data.get_temp(filter_id).unwrap_or_default());

        egui::Window::new("Pick resource")
            .id(id.with("window"))
            .collapsible(false)
            .resizable(true)
            .default_size([360.0, 420.0])
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .open(&mut keep_open)
            .show(ui.ctx(), |ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut filter)
                        .hint_text("Search…")
                        .desired_width(f32::INFINITY),
                );
                ui.separator();

                let needle = filter.trim().to_lowercase();
                let all = entries();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        if allow_none
                            && draw_tile(world, ui, &TileThumb::Icon("\u{2014}"), "(none)", false)
                                .clicked()
                        {
                            result = Some(PickResult::None);
                        }
                        for e in &all {
                            if !needle.is_empty()
                                && !e.label.to_lowercase().contains(&needle)
                                && !e.key.to_lowercase().contains(&needle)
                            {
                                continue;
                            }
                            let is_sel = current.is_some_and(|c| c.key == e.key);
                            if draw_tile(world, ui, &e.thumb, &e.label, is_sel).clicked() {
                                result = Some(PickResult::Key(e.key.clone()));
                            }
                        }
                    });
                });
            });

        // A choice (or closing the window) dismisses the popup.
        if result.is_some() || !keep_open {
            open = false;
        }
        ui.memory_mut(|m| m.data.insert_temp(filter_id, filter));
    }

    ui.memory_mut(|m| m.data.insert_temp(open_id, open));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(key: &str, label: &str) -> PickerEntry {
        PickerEntry {
            key: key.to_string(),
            label: label.to_string(),
            thumb: TileThumb::Icon("?"),
        }
    }

    /// The filter predicate the popup uses (label OR key contains the needle).
    fn matches(e: &PickerEntry, needle: &str) -> bool {
        let needle = needle.to_lowercase();
        needle.is_empty()
            || e.label.to_lowercase().contains(&needle)
            || e.key.to_lowercase().contains(&needle)
    }

    #[test]
    fn filter_matches_label_and_key() {
        let e = entry("textures/cobble/1", "Cobble Stone 1");
        assert!(matches(&e, ""));
        assert!(matches(&e, "cobble")); // label + key
        assert!(matches(&e, "stone")); // label only
        assert!(matches(&e, "/1")); // key only
        assert!(!matches(&e, "sand"));
    }
}
