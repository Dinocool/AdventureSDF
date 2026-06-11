//! The **Biome Graph** dock panel (exclusive system): toolbar (Apply/Save/Load/Reset/Auto-arrange),
//! path field, validity hint, biome breadcrumb, the `egui-snarl` graph itself (driven by [`Viewer`]), and
//! the popped-out preview windows.

use std::sync::Arc;

use bevy::prelude::*;
use bevy_egui::egui;
use egui_phosphor::regular as icon;
use egui_snarl::ui::{SnarlStyle, SnarlWidget};

use crate::assets::Asset as _;
use crate::editor::worldgen_gpu_preview::{GpuPreviewRequest, GpuPreviewRequests, GpuPreviewTextures};
use crate::sdf_render::worldgen::WorldGraph;
use crate::sdf_render::worldgen::graph::GraphAsset;

use super::convert::{
    breadcrumb_names, copy_selection, current_snarl_mut, paste_clipboard, valid_depth, world_biome_snarl,
};
use super::persist::{apply_view, load_editor_doc, load_session, save_editor_doc};
use super::preview::{
    PoppedPreview, PreviewView, WorldgenPreviewPanels, apply_scroll_zoom, nav_hash, popped_preview_window,
};
use super::viewer::Viewer;
use super::{ViewerSignals, WorldGraphEditor, auto_arrange, snarl_to_graph};

pub(super) fn graph_panel(world: &mut World, ui: &mut egui::Ui) {
    // Seed the editor once by LOADING the graph from disk (the saved .worldgraph.ron / .graph.ron, falling
    // back to the built-in default), and drive the live terrain from it.
    world.resource_scope::<WorldGraphEditor, ()>(|world, mut editor| {
        if !editor.seeded {
            // Seed by LOADING state, then restore the view (per-node settings, nav, panel target, pop-outs)
            // and drive the world. PREFER the auto-persisted session (resumes the last session exactly,
            // even without an explicit Save) and fall back to the on-disk asset document.
            let (snarl, view) = load_session().unwrap_or_else(|| load_editor_doc(&editor.path));
            editor.snarl = snarl;
            editor.seeded = true;
            editor.needs_arrange = true;
            if let Ok(g) = snarl_to_graph(&editor.snarl) {
                world.resource_mut::<WorldGraph>().0 = Arc::new(g);
            }
            world.resource_scope::<WorldgenPreviewPanels, ()>(|_w, mut panels| {
                apply_view(view, &mut editor, &mut panels);
            });
        }

        ui.horizontal(|ui| {
            // APPLY — rebuild the engine graph + push it live into the world (roll_worldgen re-meshes).
            if ui.button("Apply").on_hover_text("Rebuild + drive the live world terrain from this graph").clicked() {
                match snarl_to_graph(&editor.snarl) {
                    Ok(g) => {
                        world.resource_mut::<WorldGraph>().0 = Arc::new(g);
                        editor.status = "applied to world".into();
                    }
                    Err(e) => editor.status = format!("invalid: {e}"),
                }
            }
            // SAVE — write BOTH the compiled flat engine graph (.graph.ron, the world hot-reloads it) AND
            // the versioned editor document with biomes + view-state (.worldgraph.ron, so the hierarchy
            // AND where-the-user-was survive reload).
            if ui.button("Save").on_hover_text("Write the flat .graph.ron (world reloads it) + the .worldgraph.ron document").clicked() {
                editor.status = match snarl_to_graph(&editor.snarl) {
                    Ok(g) => {
                        let flat = (GraphAsset { graph: g }).save(std::path::Path::new(&editor.path));
                        let doc = world
                            .resource_scope::<WorldgenPreviewPanels, _>(|_w, panels| save_editor_doc(&editor, &panels));
                        match (flat, doc) {
                            (Ok(()), Ok(())) => format!("saved {} (+document)", editor.path),
                            (Err(e), _) => format!("save failed: {e}"),
                            (_, Err(e)) => format!("flat saved; document failed: {e}"),
                        }
                    }
                    Err(e) => format!("invalid: {e}"),
                };
            }
            // LOAD — re-read the persisted document (snarl + view-state) and restore the editor + panel,
            // degrading to the flat graph / default exactly like the startup seed.
            if ui.button("Load").clicked() {
                let (snarl, view) = load_editor_doc(&editor.path);
                editor.snarl = snarl;
                editor.nav.clear();
                editor.clear_node_caches();
                editor.needs_arrange = true;
                world.resource_scope::<WorldgenPreviewPanels, ()>(|_w, mut panels| {
                    apply_view(view, &mut editor, &mut panels);
                });
                editor.status = format!("loaded {}", editor.path);
            }
            if ui.button("Reset").on_hover_text("Restore the default multi-biome world graph").clicked() {
                editor.snarl = world_biome_snarl();
                editor.nav.clear();
                editor.clear_node_caches();
                editor.needs_arrange = true;
                editor.status = "reset to biome world".into();
            }
            if ui.button("Auto-arrange").on_hover_text("Lay nodes out left→right by dependency depth").clicked() {
                // Arrange the CURRENTLY shown level (inside a biome, not the top graph).
                let WorldGraphEditor { snarl, nav, caches, .. } = &mut *editor;
                let vd = valid_depth(snarl, nav);
                nav.truncate(vd);
                auto_arrange(current_snarl_mut(snarl, nav), &caches.body_size);
                editor.status = "arranged".into();
            }
        });
        ui.horizontal(|ui| {
            ui.label("Path:");
            // Borrow path mutably without conflicting with the snarl borrow below.
            let path = &mut editor.path;
            ui.add(egui::TextEdit::singleline(path).desired_width(360.0));
        });
        // Live validity hint + last status.
        ui.horizontal(|ui| {
            match snarl_to_graph(&editor.snarl) {
                Ok(g) => ui.colored_label(egui::Color32::from_rgb(140, 200, 140), format!("{} nodes ✓", g.nodes.len())),
                Err(e) => ui.colored_label(egui::Color32::from_rgb(220, 120, 120), e),
            };
            if !editor.status.is_empty() {
                ui.label(format!("· {}", editor.status));
            }
        });

        // Drop any stale tail of the nav path (e.g. a biome was deleted), then a breadcrumb to walk out.
        let valid = valid_depth(&editor.snarl, &editor.nav);
        if valid != editor.nav.len() {
            editor.nav.truncate(valid);
            editor.clear_node_caches();
        }
        let mut nav_to: Option<usize> = None;
        let crumbs = breadcrumb_names(&editor.snarl, &editor.nav);
        ui.horizontal(|ui| {
            if ui.selectable_label(editor.nav.is_empty(), format!("{} World", icon::GLOBE)).clicked() {
                nav_to = Some(0);
            }
            for (i, name) in crumbs.iter().enumerate() {
                ui.label(icon::CARET_RIGHT);
                if ui.selectable_label(i + 1 == editor.nav.len(), format!("{} {name}", icon::PLANT)).clicked() {
                    nav_to = Some(i + 1);
                }
            }
        });
        if let Some(d) = nav_to.filter(|&d| d != editor.nav.len()) {
            editor.nav.truncate(d);
            editor.clear_node_caches();
            editor.needs_arrange = true;
        }
        ui.separator();

        // Intercept scroll-zoom for the inline preview hovered last frame — egui-snarl applies its own
        // graph zoom BEFORE drawing nodes, so consume the scroll here (before the show) and route it to
        // the preview instead.
        if let Some(node) = editor.hovered_preview.take() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let h = &mut editor.caches.views.entry(node).or_default().zoom_half_m;
                apply_scroll_zoom(ui, scroll, h);
            }
        }

        // GPU preview plumbing: read last frame's textures + gather this frame's requests (shared by the
        // inline 3D previews below + the pop-out windows).
        let gpu_tex = world.get_resource::<GpuPreviewTextures>().map(|t| t.0.clone()).unwrap_or_default();
        let mut gpu_reqs: Vec<GpuPreviewRequest> = Vec::new();
        let level_salt = nav_hash(&editor.nav);

        // Show the snarl at the current nav depth. Disjoint borrows: `snarl`+`nav` resolve the level;
        // the rest are the per-node preview caches the Viewer drives.
        editor.signals = ViewerSignals::default();
        {
            let WorldGraphEditor { snarl, nav, caches, signals, hovered_preview, .. } = &mut *editor;
            let current = current_snarl_mut(snarl, nav);
            let mut viewer = Viewer {
                caches,
                signals,
                gpu_tex: &gpu_tex,
                gpu_reqs: &mut gpu_reqs,
                hovered_preview,
                level_salt,
            };
            // Keep nodes readable on load: egui-snarl's initial view auto-fits the graph clamped to
            // [min_scale, max_scale], so the floor doubles as the default zoom — 0.75 keeps a freshly-loaded
            // graph legible (the compact auto-arrange usually fits above this). Allow zooming in to 3×.
            // egui-snarl's header-triangle node-collapse stays ON (the user wants whole-node collapse); the
            // header eye is a SEPARATE toggle for just the preview. min_scale is the floor the default
            // auto-fit clamps to, so it sets the default on-screen node size — 0.86 (~15% up from 0.75) keeps
            // a freshly-loaded graph comfortably readable. Allow zooming in to 3×.
            let style = SnarlStyle { min_scale: Some(0.86), max_scale: Some(3.0), ..SnarlStyle::new() };
            SnarlWidget::new()
                .id(egui::Id::new(("worldgen-biome-graph", level_salt)))
                .style(style)
                .show(current, &mut viewer, ui);
        }

        // Node clipboard + keyboard: select (built-in shift/cmd-click) → delete / copy / cut / paste, on
        // the CURRENT nav level's snarl. Resolved AFTER the show (so we don't hold `current` across it)
        // and keyed by the per-level snarl id (matches `get_selected_nodes`'s selection store).
        {
            let snarl_id = egui::Id::new(("worldgen-biome-graph", level_salt));
            let selected = egui_snarl::ui::get_selected_nodes(snarl_id, ui.ctx());
            let (delete, copy, cut, paste) = ui.input(|i| {
                let cmd = i.modifiers.command;
                (
                    i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace),
                    cmd && i.key_pressed(egui::Key::C),
                    cmd && i.key_pressed(egui::Key::X),
                    cmd && i.key_pressed(egui::Key::V),
                )
            });
            if delete || copy || cut || paste {
                let WorldGraphEditor { snarl, nav, clipboard, .. } = &mut *editor;
                let current = current_snarl_mut(snarl, nav);
                if copy || cut {
                    *clipboard = copy_selection(current, &selected);
                }
                if (delete || cut) && !selected.is_empty() {
                    for id in &selected {
                        if current.get_node(*id).is_some() {
                            current.remove_node(*id);
                        }
                    }
                }
                let pasted = paste && !clipboard.is_empty();
                if pasted {
                    paste_clipboard(current, clipboard, egui::vec2(24.0, 24.0));
                }
                // (drop `current`/`clipboard` borrows here before touching another `editor` field)
                editor.needs_arrange |= pasted;
            }
        }

        // A node's preview was collapsed/expanded this frame → re-pack so the layout tightens.
        editor.needs_arrange |= editor.signals.needs_arrange;
        // After a seed/load (or a collapse-toggle), auto-arrange once the nodes have been measured this
        // frame (so the layout uses real sizes). Applies on the next frame.
        if std::mem::take(&mut editor.needs_arrange) {
            editor.rearrange();
        }
        // Re-centre menu-added nodes on the cursor (egui-snarl positions a new node at the menu's top-left;
        // it grows down-right from there → it would land offset from the right-click). Done once the node's
        // real GRAPH-space size is measured (next frame), so it's exact at ANY zoom. Unmeasured ⇒ stays queued.
        {
            let ed = &mut *editor;
            ed.pending_center.append(&mut ed.signals.added);
        }
        if !editor.pending_center.is_empty() {
            let WorldGraphEditor { snarl, nav, caches, pending_center, .. } = &mut *editor;
            let current = current_snarl_mut(snarl, nav);
            pending_center.retain(|&id| match (caches.body_size.get(&id), current.get_node_info_mut(id)) {
                (Some(size), Some(info)) => {
                    info.pos -= 0.5 * *size; // centre the node on the recorded position
                    false
                }
                (None, Some(_)) => true, // not measured yet → keep for next frame
                (_, None) => false,      // node gone → drop
            });
        }
        // Descend into a biome the user opened this frame.
        if let Some(id) = editor.signals.enter.take() {
            editor.nav.push(id);
            editor.clear_node_caches();
            editor.needs_arrange = true;
        }
        // Spawn a NEW dockable preview instance + tab (unbounded — every click opens another), snapshotting
        // the node's nav + view state. `open` queues the tab; `open_preview_panel` creates + focuses it
        // OUTSIDE the dock render (the dock state isn't in the World here).
        if let Some(node) = editor.signals.to_panel.take() {
            let nav = editor.nav.clone();
            let v = editor.caches.views.get(&node).copied().unwrap_or_default();
            if let Some(mut panels) = world.get_resource_mut::<WorldgenPreviewPanels>() {
                let view = PreviewView {
                    half: v.zoom_half_m,
                    cx: v.pan.0,
                    cz: v.pan.1,
                    yaw: v.cam.0,
                    pitch: v.cam.1,
                };
                panels.open((nav, node), view, v.surface, v.modes);
            }
        }
        // Pop a node's preview out into a movable window (snapshotting its current view state + nav path).
        if let Some(node) = editor.signals.pop_request.take() {
            let v = editor.caches.views.get(&node).copied().unwrap_or_default();
            let nav = editor.nav.clone();
            let id = editor.next_pop_id;
            editor.next_pop_id += 1;
            editor.popped.push(PoppedPreview {
                id,
                nav,
                node,
                half: v.zoom_half_m,
                cx: 0.0,
                cz: 0.0,
                size: v.disp_px.max(260.0),
                is3d: v.surface,
                cam: v.cam,
                modes: v.modes,
                open: true,
            });
        }
        // Render the popped-out preview windows (float above everything; drag anywhere incl. top panel).
        // 3D pop-outs render on the GPU via the same shared request/texture buffers as the inline previews.
        {
            let WorldGraphEditor { snarl, popped, .. } = &mut *editor;
            for p in popped.iter_mut() {
                popped_preview_window(ui, p, snarl, &gpu_tex, &mut gpu_reqs);
            }
            popped.retain(|p| p.open);
        }
        if !gpu_reqs.is_empty()
            && let Some(mut reqs) = world.get_resource_mut::<GpuPreviewRequests>()
        {
            reqs.0.append(&mut gpu_reqs);
        }
    });
}
