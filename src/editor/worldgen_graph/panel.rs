//! The **Biome Graph** dock panel (exclusive system): toolbar (Apply/Save/Load/Reset/Auto-arrange),
//! path field, validity hint, biome breadcrumb, the `egui-snarl` graph itself (driven by [`Viewer`]), and
//! the popped-out preview windows.

use std::sync::Arc;

use bevy::prelude::*;
use bevy_egui::egui;
use egui_phosphor::regular as icon;
use egui_snarl::Snarl;
use egui_snarl::ui::{SnarlStyle, SnarlWidget};

use crate::assets::Asset as _;
use crate::editor::worldgen_gpu_preview::{GpuPreviewRequest, GpuPreviewRequests, GpuPreviewTextures};
use crate::sdf_render::worldgen::WorldGraph;
use crate::sdf_render::worldgen::graph::GraphAsset;

use super::convert::{
    breadcrumb_names, current_snarl_mut, graph_to_snarl, load_editor_snarl, valid_depth, world_biome_snarl,
    worldgraph_path,
};
use super::preview::{
    CAM_DEFAULT, DEFAULT_PREVIEW_PX, PREVIEW_HALF_M, PoppedPreview, WorldgenPreviewPanel, apply_scroll_zoom,
    nav_hash, popped_preview_window,
};
use super::viewer::Viewer;
use super::{EdNode, ViewerSignals, WorldGraphEditor, auto_arrange, snarl_to_graph};

pub(super) fn graph_panel(world: &mut World, ui: &mut egui::Ui) {
    // Seed the editor once by LOADING the graph from disk (the saved .worldgraph.ron / .graph.ron, falling
    // back to the built-in default), and drive the live terrain from it.
    world.resource_scope::<WorldGraphEditor, ()>(|world, mut editor| {
        if !editor.seeded {
            editor.snarl = load_editor_snarl(&editor.path);
            editor.seeded = true;
            if let Ok(g) = snarl_to_graph(&editor.snarl) {
                world.resource_mut::<WorldGraph>().0 = Arc::new(g);
            }
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
            // the hierarchical editor snarl with biomes (.worldgraph.ron, so the hierarchy survives reload).
            if ui.button("Save").on_hover_text("Write the flat .graph.ron (world reloads it) + the .worldgraph.ron hierarchy").clicked() {
                editor.status = match snarl_to_graph(&editor.snarl) {
                    Ok(g) => {
                        let flat = (GraphAsset { graph: g }).save(std::path::Path::new(&editor.path));
                        let wg = worldgraph_path(&editor.path);
                        let hier = ron::ser::to_string_pretty(&editor.snarl, ron::ser::PrettyConfig::default())
                            .map_err(|e| e.to_string())
                            .and_then(|s| std::fs::write(&wg, s).map_err(|e| e.to_string()));
                        match (flat, hier) {
                            (Ok(()), Ok(())) => format!("saved {} (+hierarchy)", editor.path),
                            (Err(e), _) => format!("save failed: {e}"),
                            (_, Err(e)) => format!("flat saved; hierarchy failed: {e}"),
                        }
                    }
                    Err(e) => format!("invalid: {e}"),
                };
            }
            // LOAD — prefer the hierarchical .worldgraph.ron (restores biomes); else the flat .graph.ron.
            if ui.button("Load").clicked() {
                let wg = worldgraph_path(&editor.path);
                editor.status = match std::fs::read_to_string(&wg) {
                    Ok(s) => match ron::de::from_str::<Snarl<EdNode>>(&s) {
                        Ok(snarl) => {
                            editor.snarl = snarl;
                            editor.nav.clear();
                            editor.clear_node_caches();
                            editor.needs_arrange = true;
                            format!("loaded {wg}")
                        }
                        Err(e) => format!("hierarchy parse failed: {e}"),
                    },
                    Err(_) => match std::fs::read_to_string(&editor.path) {
                        Ok(s) => match ron::de::from_str::<GraphAsset>(&s) {
                            Ok(asset) => {
                                editor.snarl = graph_to_snarl(&asset.graph);
                                editor.nav.clear();
                                editor.clear_node_caches();
                                editor.needs_arrange = true;
                                format!("loaded {} (flat)", editor.path)
                            }
                            Err(e) => format!("parse failed: {e}"),
                        },
                        Err(e) => format!("read failed: {e}"),
                    },
                };
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
        }
        ui.separator();

        // Intercept scroll-zoom for the inline preview hovered last frame — egui-snarl applies its own
        // graph zoom BEFORE drawing nodes, so consume the scroll here (before the show) and route it to
        // the preview instead.
        if let Some(node) = editor.hovered_preview.take() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let h = editor.caches.zoom_half_m.entry(node).or_insert(PREVIEW_HALF_M);
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
            let style = SnarlStyle { min_scale: Some(0.75), max_scale: Some(3.0), ..SnarlStyle::new() };
            SnarlWidget::new()
                .id(egui::Id::new("worldgen-biome-graph"))
                .style(style)
                .show(current, &mut viewer, ui);
        }
        // After a seed/load, auto-arrange once the nodes have been measured this frame (so the layout uses
        // real sizes). Applies on the next frame.
        if std::mem::take(&mut editor.needs_arrange) {
            editor.rearrange();
        }
        // Descend into a biome the user opened this frame.
        if let Some(id) = editor.signals.enter.take() {
            editor.nav.push(id);
            editor.clear_node_caches();
        }
        // Retarget the dockable preview panel (snapshotting the node's nav + view state).
        if let Some(node) = editor.signals.to_panel.take() {
            let nav = editor.nav.clone();
            let half = editor.caches.zoom_half_m.get(&node).copied().unwrap_or(PREVIEW_HALF_M);
            let cam = editor.caches.cam.get(&node).copied().unwrap_or(CAM_DEFAULT);
            let pan = editor.caches.pan.get(&node).copied().unwrap_or((0.0, 0.0));
            let is3d = editor.caches.surface.contains(&node);
            if let Some(mut panel) = world.get_resource_mut::<WorldgenPreviewPanel>() {
                panel.target = Some((nav, node));
                panel.half = half;
                panel.cam = cam;
                panel.pan = pan;
                panel.is3d = is3d;
                // Ensure/focus the dock tab — but only OUTSIDE the dock render (the dock state isn't in
                // the World here). `open_preview_panel` handles it next frame.
                panel.pending_open = true;
            }
        }
        // Pop a node's preview out into a movable window (snapshotting its current view state + nav path).
        if let Some(node) = editor.signals.pop_request.take() {
            let half = editor.caches.zoom_half_m.get(&node).copied().unwrap_or(PREVIEW_HALF_M);
            let is3d = editor.caches.surface.contains(&node);
            let cam = editor.caches.cam.get(&node).copied().unwrap_or(CAM_DEFAULT);
            let size = editor.caches.disp_px.get(&node).copied().unwrap_or(DEFAULT_PREVIEW_PX).max(260.0);
            let nav = editor.nav.clone();
            let id = editor.next_pop_id;
            editor.next_pop_id += 1;
            editor.popped.push(PoppedPreview {
                id,
                nav,
                node,
                half,
                cx: 0.0,
                cz: 0.0,
                size,
                is3d,
                cam,
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
