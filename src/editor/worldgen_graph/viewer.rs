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
    PreviewView, gpu_inline_key, handle_preview_gestures, paint_scale_label, preview_image,
    preview_resize_grip,
};
use super::{CLIMATE_INPUTS, EdNode, NodeCaches, ViewerSignals, climate_name, graph_rooted_at};

/// The Snarl UI viewer. Borrows the editor's per-node preview caches for the frame so each node can
/// draw a (default-on, collapsible, resizable, zoomable) 2D heatmap of its sub-graph (see
/// [`Viewer::show_body`]).
pub(super) struct Viewer<'a> {
    /// The per-node UI caches (the persisted [`NodeView`] settings + the transient body_size).
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

    /// Header: the node title + (for nodes that own a preview — Op/Biome) an **eye checkbox** that
    /// shows/hides the inline preview. Keeping the toggle here means a preview-OFF node collapses to JUST
    /// its params — no body divider, no empty space (unlike an in-body collapse button).
    fn show_header(&mut self, node: NodeId, _inputs: &[InPin], _outputs: &[OutPin], ui: &mut egui::Ui, snarl: &mut Snarl<EdNode>) {
        ui.horizontal(|ui| {
            ui.label(self.title(&snarl[node]));
            if matches!(snarl.get_node(node), Some(EdNode::Op(_) | EdNode::Biome { .. })) {
                let nv = self.caches.views.entry(node).or_default();
                let mut shown = !nv.collapsed;
                if ui.checkbox(&mut shown, icon::EYE).on_hover_text("Show this node's 2D/3D preview").changed() {
                    nv.collapsed = !shown;
                    self.signals.needs_arrange = true;
                }
            }
        });
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

        // Preview hidden (the header eye is unchecked) ⇒ the body is just the params above — no divider,
        // no preview, no wasted space. The node stays compact.
        if self.caches.views.entry(node).or_default().collapsed {
            self.caches.body_size.insert(node, ui.min_rect().size());
            return;
        }
        // Divider between the node params (above) and the preview section (below).
        ui.separator();

        // Open: the preview IMAGE on the LEFT, its controls in a column on the RIGHT (no overlap).
        let view = *self.caches.views.entry(node).or_default();
        let is3d = view.surface;
        let size = view.disp_px;
        // Render at the displayed size in physical pixels (no cap) so the preview is always crisp.
        let ppp = ui.ctx().pixels_per_point();
        let res = ((size * ppp).round() as usize).max(32);
        let half = view.zoom_half_m;
        let (yaw, pitch) = view.cam;
        let (cx, cz) = view.pan;

        match graph_rooted_at(snarl, node) {
            Ok(g) => {
                // Both 2D and 3D render on the GPU pool (one shader, one `height_colour` SSOT). Push a
                // request and draw last frame's pool texture.
                let gkey = gpu_inline_key(self.level_salt, node);
                let view = PreviewView { half, cx, cz, yaw, pitch };
                self.gpu_reqs.push(view.to_request(gkey, g, is3d, res as u32));
                let tex = self.gpu_tex.get(&gkey).copied();
                ui.horizontal_top(|ui| {
                    // LEFT — the preview image (a flat placeholder for the ~1 frame before the GPU texture
                    // warms up) with on-image gestures: scroll = zoom, drag = orbit (3D) / pan (2D),
                    // right-drag = pan (3D). The scroll is consumed so the graph doesn't also zoom.
                    let img_resp = preview_image(ui, tex, egui::vec2(size, size));
                    // Scale-label overlay: the visible world width, in a corner of the preview.
                    paint_scale_label(ui, img_resp.rect, half);
                    {
                        // Read the node's view into one PreviewView, apply gestures, write back.
                        let mut v = PreviewView { half, cx, cz, yaw, pitch };
                        handle_preview_gestures(&img_resp, is3d, size, &mut v);
                        let nv = self.caches.views.entry(node).or_default();
                        nv.zoom_half_m = v.half;
                        nv.cam = (v.yaw, v.pitch);
                        nv.pan = (v.cx, v.cz);
                    }
                    // Record hover so the panel can intercept this preview's scroll-zoom next frame
                    // (before egui-snarl applies its own graph zoom).
                    if img_resp.hovered() {
                        *self.hovered_preview = Some(node);
                    }
                    // Drag-resize grip at the image's BOTTOM-RIGHT corner — added AFTER (and over) the
                    // image as its own widget rect, so its drag is separate from the image's orbit/pan
                    // gesture. Dragging it grows/shrinks `disp_px` (the preview side), clamped 64..=1024.
                    if let Some(delta) = preview_resize_grip(ui, img_resp.rect) {
                        let sz = &mut self.caches.views.entry(node).or_default().disp_px;
                        *sz = (*sz + delta).clamp(64.0, 1024.0);
                    }
                    // RIGHT — controls column (pop-out, → panel, zoom, 2D/3D). Show/hide is the header eye.
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            if ui.small_button(icon::ARROWS_OUT).on_hover_text("Pop out into a movable window").clicked() {
                                self.signals.pop_request = Some(node);
                            }
                            if ui.small_button(icon::PICTURE_IN_PICTURE).on_hover_text("Show in the dockable preview panel (by the viewport)").clicked() {
                                self.signals.to_panel = Some(node);
                            }
                        });
                        let h = &mut self.caches.views.entry(node).or_default().zoom_half_m;
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
                            self.caches.views.entry(node).or_default().surface = !is3d;
                        }
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

    fn has_node_menu(&mut self, _node: &EdNode) -> bool {
        true
    }

    fn show_node_menu(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<EdNode>,
    ) {
        // Delete — `remove_node` drops the node AND its dangling wires (in + out). Deleting Output/Input
        // just makes the graph invalid, which the existing validity hint surfaces — kept uniform/simple.
        if ui.button(format!("{} Delete", icon::TRASH)).clicked() {
            snarl.remove_node(node);
            ui.close();
            return;
        }
        // Duplicate — drop a clone of this node's EdNode just down-right of it (no wires).
        if ui.button(format!("{} Duplicate", icon::COPY)).clicked() {
            if let Some(kind) = snarl.get_node(node).cloned() {
                let pos = snarl.get_node_info(node).map(|n| n.pos).unwrap_or(egui::Pos2::ZERO);
                snarl.insert_node(pos + egui::vec2(30.0, 30.0), kind);
            }
            ui.close();
        }
    }

    fn has_graph_menu(&mut self, _pos: egui::Pos2, _snarl: &mut Snarl<EdNode>) -> bool {
        true
    }

    fn show_graph_menu(&mut self, pos: egui::Pos2, ui: &mut egui::Ui, snarl: &mut Snarl<EdNode>) {
        // `pos` is the menu's top-left (egui-snarl passes `from_global * cursor`, not the click), and a node
        // grows down-right from its position — so the panel re-centres each new node (`signals.added` →
        // `pending_center`) on `pos` once its real size is measured, landing it at the right-click.
        ui.label("Add node");
        for kind in node_catalog() {
            if ui.button(node_kind_name(&kind)).clicked() {
                self.signals.added.push(snarl.insert_node(pos, EdNode::Op(kind)));
                ui.close();
            }
        }
        ui.separator();
        if ui.button(format!("{} Biome", icon::PLANT)).on_hover_text("A nested biome sub-graph (climate in, height out)").clicked() {
            let id = snarl.insert_node(pos, EdNode::Biome { name: "biome".into(), graph: Box::new(new_biome_subgraph()) });
            self.signals.added.push(id);
            ui.close();
        }
        ui.menu_button("Climate input", |ui| {
            for (k, name) in CLIMATE_INPUTS.iter().enumerate() {
                if ui.button(*name).on_hover_text("A climate value piped in from the parent biome").clicked() {
                    self.signals.added.push(snarl.insert_node(pos, EdNode::Input(k)));
                    ui.close();
                }
            }
        });
    }
}
