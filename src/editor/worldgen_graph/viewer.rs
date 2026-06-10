//! The `egui-snarl` [`SnarlViewer`] for the biome graph: per-node titles/pins/bodies, the (default-on,
//! collapsible, resizable, zoomable) inline 2D/3D preview, wiring, and the add-node context menu.

use bevy_egui::egui;
use egui_phosphor::regular as icon;
use egui_snarl::ui::{PinInfo, SnarlPin, SnarlViewer};
use egui_snarl::{InPin, NodeId, OutPin, Snarl};

use crate::editor::worldgen_gpu_preview::GpuPreviewRequest;

use super::convert::new_biome_subgraph;
use super::node::{input_label, node_catalog, node_kind_name, node_params_ui};
use super::preview::{
    CAM_DEFAULT, DEFAULT_PREVIEW_PX, PREVIEW_HALF_M, gpu_inline_key, handle_preview_gestures, preview_image,
};
use super::{CLIMATE_INPUTS, EdNode, NodeCaches, ViewerSignals, climate_name, graph_rooted_at};

/// The Snarl UI viewer. Borrows the editor's per-node preview caches for the frame so each node can
/// draw a (default-on, collapsible, resizable, zoomable) 2D heatmap of its sub-graph (see
/// [`Viewer::show_body`]).
pub(super) struct Viewer<'a> {
    /// The per-node UI caches (collapsed/zoom/surface/cam/body_size/disp_px/pan).
    pub(super) caches: &'a mut NodeCaches,
    /// One-shot Viewer→panel signals (Open / pop-out / → panel), raised here, drained by the panel.
    pub(super) signals: &'a mut ViewerSignals,
    /// Last frame's GPU preview textures (key → egui id) read by 3D inline previews.
    pub(super) gpu_tex: &'a std::collections::HashMap<u64, egui::TextureId>,
    /// This frame's GPU preview requests, pushed by 3D inline previews; drained by the panel.
    pub(super) gpu_reqs: &'a mut Vec<GpuPreviewRequest>,
    /// Set to the node whose preview image the pointer is over (for next-frame scroll interception).
    pub(super) hovered_preview: &'a mut Option<NodeId>,
    /// Hash of the current nav path — combined with the node id into a stable GPU pool key per preview.
    pub(super) level_salt: u64,
}

impl SnarlViewer<EdNode> for Viewer<'_> {
    fn title(&mut self, node: &EdNode) -> String {
        match node {
            EdNode::Output => "Output".into(),
            EdNode::Op(k) => node_kind_name(k).into(),
            EdNode::Biome { name, .. } => format!("{} {name}", icon::PLANT),
            EdNode::Input(k) => format!("{} {}", icon::ARROW_ELBOW_DOWN_RIGHT, climate_name(*k)),
        }
    }

    fn inputs(&mut self, node: &EdNode) -> usize {
        match node {
            EdNode::Output => 1,
            EdNode::Op(k) => k.arity(),
            EdNode::Biome { .. } => CLIMATE_INPUTS.len(),
            EdNode::Input(_) => 0,
        }
    }

    fn outputs(&mut self, node: &EdNode) -> usize {
        match node {
            EdNode::Output => 0,
            EdNode::Op(_) | EdNode::Biome { .. } | EdNode::Input(_) => 1,
        }
    }

    // Op + Biome nodes get a body (preview / biome controls); Input + Output don't.
    fn has_body(&mut self, node: &EdNode) -> bool {
        matches!(node, EdNode::Op(_) | EdNode::Biome { .. })
    }

    fn show_body(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<EdNode>,
    ) {
        // Node params / biome header, stacked vertically at the top of the body.
        match &mut snarl[node] {
            EdNode::Op(kind) => node_params_ui(ui, kind),
            EdNode::Biome { name, .. } => {
                ui.add(egui::TextEdit::singleline(name).desired_width(120.0).hint_text("biome name"));
            }
            _ => {}
        }
        if matches!(snarl.get_node(node), Some(EdNode::Biome { .. }))
            && ui.button(format!("{} Open", icon::CARET_RIGHT)).on_hover_text("Edit this biome's sub-graph").clicked()
        {
            self.signals.enter = Some(node);
        }
        // Divider between the node params (above) and the preview section (below).
        ui.separator();

        // Collapsed: just an expand toggle.
        if self.caches.collapsed.contains(&node) {
            if ui
                .small_button(format!("{} Preview", icon::CARET_RIGHT))
                .on_hover_text("Show this node's 2D/3D preview")
                .clicked()
            {
                self.caches.collapsed.remove(&node);
            }
            self.caches.body_size.insert(node, ui.min_rect().size());
            return;
        }

        // Open: the preview IMAGE on the LEFT, its controls in a column on the RIGHT (no overlap).
        let is3d = self.caches.surface.contains(&node);
        let size = self.caches.disp_px.get(&node).copied().unwrap_or(DEFAULT_PREVIEW_PX);
        // Render at the displayed size in physical pixels (no cap) so the preview is always crisp.
        let ppp = ui.ctx().pixels_per_point();
        let res = ((size * ppp).round() as usize).max(32);
        let half = *self.caches.zoom_half_m.get(&node).unwrap_or(&PREVIEW_HALF_M);
        let (yaw, pitch) = *self.caches.cam.get(&node).unwrap_or(&CAM_DEFAULT);
        let (cx, cz) = *self.caches.pan.get(&node).unwrap_or(&(0.0, 0.0));

        match graph_rooted_at(snarl, node) {
            Ok(g) => {
                // Both 2D and 3D render on the GPU pool (one shader, one `height_colour` SSOT). Push a
                // request and draw last frame's pool texture.
                let gkey = gpu_inline_key(self.level_salt, node);
                self.gpu_reqs.push(GpuPreviewRequest {
                    key: gkey,
                    graph: g,
                    half,
                    center: (cx, cz),
                    is3d,
                    yaw,
                    pitch,
                    res_w: res as u32,
                    res_h: res as u32,
                });
                let tex = self.gpu_tex.get(&gkey).copied();
                ui.horizontal_top(|ui| {
                    // LEFT — the preview image (a flat placeholder for the ~1 frame before the GPU texture
                    // warms up) with on-image gestures: scroll = zoom, drag = orbit (3D) / pan (2D),
                    // right-drag = pan (3D). The scroll is consumed so the graph doesn't also zoom.
                    let img_resp = preview_image(ui, tex, egui::vec2(size, size));
                    {
                        let h = self.caches.zoom_half_m.entry(node).or_insert(PREVIEW_HALF_M);
                        let cam = self.caches.cam.entry(node).or_insert(CAM_DEFAULT);
                        let pan = self.caches.pan.entry(node).or_insert((0.0, 0.0));
                        handle_preview_gestures(ui, &img_resp, is3d, size, h, &mut pan.0, &mut pan.1, &mut cam.0, &mut cam.1);
                    }
                    // Record hover so the panel can intercept this preview's scroll-zoom next frame
                    // (before egui-snarl applies its own graph zoom).
                    if img_resp.hovered() {
                        *self.hovered_preview = Some(node);
                    }
                    // RIGHT — controls column (collapse, pop-out, zoom, 2D/3D, size).
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            if ui.small_button(icon::CARET_DOWN).on_hover_text("Collapse preview").clicked() {
                                self.caches.collapsed.insert(node);
                            }
                            if ui.small_button(icon::ARROWS_OUT).on_hover_text("Pop out into a movable window").clicked() {
                                self.signals.pop_request = Some(node);
                            }
                            if ui.small_button(icon::PICTURE_IN_PICTURE).on_hover_text("Show in the dockable preview panel (by the viewport)").clicked() {
                                self.signals.to_panel = Some(node);
                            }
                        });
                        let h = self.caches.zoom_half_m.entry(node).or_insert(PREVIEW_HALF_M);
                        let mut km = *h * 2.0 / 1000.0;
                        if ui
                            .add(egui::DragValue::new(&mut km).speed(0.25).range(0.05..=512.0).suffix(" km"))
                            .on_hover_text("Zoom: width of the sampled world window")
                            .changed()
                        {
                            *h = (km * 1000.0 / 2.0).max(1.0);
                        }
                        if ui
                            .selectable_label(is3d, "3D")
                            .on_hover_text("3D SDF-raymarched surface (drag the image to orbit)")
                            .clicked()
                        {
                            if is3d {
                                self.caches.surface.remove(&node);
                            } else {
                                self.caches.surface.insert(node);
                            }
                        }
                        let sz = self.caches.disp_px.entry(node).or_insert(DEFAULT_PREVIEW_PX);
                        ui.add(egui::DragValue::new(sz).speed(2.0).range(64.0..=1024.0).suffix(" px"))
                            .on_hover_text("Preview size");
                    });
                });
            }
            Err(e) => {
                ui.colored_label(egui::Color32::from_rgb(200, 150, 120), format!("connect inputs ({e})"));
            }
        }
        self.caches.body_size.insert(node, ui.min_rect().size());
    }

    fn show_input(&mut self, pin: &InPin, ui: &mut egui::Ui, snarl: &mut Snarl<EdNode>) -> impl SnarlPin + 'static {
        ui.label(input_label(&snarl[pin.id.node], pin.id.input));
        PinInfo::circle().with_fill(egui::Color32::from_rgb(120, 160, 220))
    }

    fn show_output(&mut self, _pin: &OutPin, _ui: &mut egui::Ui, _snarl: &mut Snarl<EdNode>) -> impl SnarlPin + 'static {
        // Single output pin — self-evident, and an "out" label here overlaps the pin (bad right margin).
        // Params live in the body (stacked vertically) to keep nodes narrow.
        PinInfo::circle().with_fill(egui::Color32::from_rgb(160, 210, 140))
    }

    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<EdNode>) {
        // An input takes a single wire: replace any existing connection on the target pin.
        snarl.drop_inputs(to.id);
        snarl.connect(from.id, to.id);
    }

    fn has_graph_menu(&mut self, _pos: egui::Pos2, _snarl: &mut Snarl<EdNode>) -> bool {
        true
    }

    fn show_graph_menu(&mut self, pos: egui::Pos2, ui: &mut egui::Ui, snarl: &mut Snarl<EdNode>) {
        ui.label("Add node");
        for kind in node_catalog() {
            if ui.button(node_kind_name(&kind)).clicked() {
                snarl.insert_node(pos, EdNode::Op(kind));
                ui.close();
            }
        }
        ui.separator();
        if ui.button(format!("{} Biome", icon::PLANT)).on_hover_text("A nested biome sub-graph (climate in, height out)").clicked() {
            snarl.insert_node(pos, EdNode::Biome { name: "biome".into(), graph: Box::new(new_biome_subgraph()) });
            ui.close();
        }
        ui.menu_button("Climate input", |ui| {
            for (k, name) in CLIMATE_INPUTS.iter().enumerate() {
                if ui.button(*name).on_hover_text("A climate value piped in from the parent biome").clicked() {
                    snarl.insert_node(pos, EdNode::Input(k));
                    ui.close();
                }
            }
        });
    }
}
