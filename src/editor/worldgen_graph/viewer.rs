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
        let title = self.title(&snarl[node]);
        let has_preview = matches!(snarl.get_node(node), Some(EdNode::Op(_) | EdNode::Biome { .. }));
        // Right-align the eye to the node's right edge. This is safe ONLY because the body is now strictly
        // CONTENT-sized, so `body_size.x` is stable — span the header to it and the eye sits at the right.
        // (When the body had a full-width widget, `body_size.x` tracked `available_width` and this fed back
        // into a rightward runaway — see `tests::node_width_does_not_run_away`, which guards it.) The
        // `right_to_left` is kept INSIDE a content-height horizontal so it can't grow the header vertically.
        let want_w = self.caches.body_size.get(&node).map_or(0.0, |s| s.x);
        ui.horizontal(|ui| {
            if has_preview {
                ui.set_min_width(want_w);
            }
            ui.label(title);
            if has_preview {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let nv = self.caches.views.entry(node).or_default();
                    let mut shown = !nv.collapsed;
                    if ui.checkbox(&mut shown, icon::EYE).on_hover_text("Show this node's 2D/3D preview").changed() {
                        // Show/hide resizes the node IN PLACE — no re-arrange (don't shift other nodes).
                        nv.collapsed = !shown;
                    }
                });
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
        // egui-snarl lays the body out LEFT-TO-RIGHT, so WRAP everything in a vertical stack — otherwise the
        // preview image sits BESIDE the params and the node ~doubles in size (and won't shrink with it).
        ui.vertical(|ui| {
            // Node params / biome header at the top.
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

            // Preview hidden (header eye unchecked) ⇒ body is just the params above — compact.
            if self.caches.views.entry(node).or_default().collapsed {
                return;
            }
            // A fixed GAP, not `ui.separator()`: in this vertical layout a separator is a full-WIDTH line that
            // spans `available_width`, and egui-snarl feeds the node's own (growing) width back as that
            // available width — so a full-width widget makes the node expand rightward every frame. Keeping the
            // body strictly content-sized (gap, fixed/looped widgets only) is what stops the runaway.
            ui.add_space(6.0);

            let view = *self.caches.views.entry(node).or_default();
            let is3d = view.surface;
            let size = view.disp_px;
            // Render at the displayed size in physical pixels (no cap) so the preview is always crisp.
            let res = ((size * ui.ctx().pixels_per_point()).round() as usize).max(32);
            let half = view.zoom_half_m;
            let (yaw, pitch) = view.cam;
            let (cx, cz) = view.pan;

            match graph_rooted_at(snarl, node) {
                Ok(g) => {
                    // Both 2D and 3D render on the GPU pool (one shader, one `height_colour` SSOT).
                    let gkey = gpu_inline_key(self.level_salt, node);
                    self.gpu_reqs.push(PreviewView { half, cx, cz, yaw, pitch }.to_request(gkey, g, is3d, res as u32));
                    let tex = self.gpu_tex.get(&gkey).copied();
                    // Controls ABOVE the image: a compact icon row (no km field — zoom is scroll on the image).
                    ui.horizontal(|ui| {
                        if ui.small_button(icon::ARROWS_OUT).on_hover_text("Pop out into a movable window").clicked() {
                            self.signals.pop_request = Some(node);
                        }
                        if ui.small_button(icon::PICTURE_IN_PICTURE).on_hover_text("Show in a dockable preview panel").clicked() {
                            self.signals.to_panel = Some(node);
                        }
                        if ui.selectable_label(is3d, "3D").on_hover_text("3D surface (drag the image to orbit)").clicked() {
                            self.caches.views.entry(node).or_default().surface = !is3d;
                        }
                    });
                    // The preview image (placeholder for the ~1 frame before the GPU texture warms up) with
                    // on-image gestures: scroll = zoom, drag = orbit (3D) / pan (2D), right-drag = pan (3D).
                    let img_resp = preview_image(ui, tex, egui::vec2(size, size));
                    paint_scale_label(ui, img_resp.rect, half);
                    {
                        let mut v = PreviewView { half, cx, cz, yaw, pitch };
                        handle_preview_gestures(&img_resp, is3d, size, &mut v);
                        let nv = self.caches.views.entry(node).or_default();
                        nv.zoom_half_m = v.half;
                        nv.cam = (v.yaw, v.pitch);
                        nv.pan = (v.cx, v.cz);
                    }
                    if img_resp.hovered() {
                        *self.hovered_preview = Some(node);
                    }
                    // Drag-resize grip at the image's bottom-right corner (own widget rect, separate from the
                    // orbit/pan gesture); grows/shrinks `disp_px`, clamped 64..=1024.
                    if let Some(delta) = preview_resize_grip(ui, img_resp.rect) {
                        let sz = &mut self.caches.views.entry(node).or_default().disp_px;
                        *sz = (*sz + delta).clamp(64.0, 1024.0);
                    }
                }
                Err(e) => {
                    ui.colored_label(egui::Color32::from_rgb(200, 150, 120), format!("connect inputs ({e})"));
                }
            }
        });
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_render::worldgen::graph::node::{FbmAxis, NodeKind};
    use egui_snarl::Snarl;
    use egui_snarl::ui::{SnarlStyle, SnarlWidget};

    /// Drive the REAL `SnarlWidget` (with our `Viewer`) over `frames` headless egui frames and return the
    /// rendered NODE-FRAME width on each frame. This exercises egui-snarl's actual node sizing + its
    /// content↔available-width feedback across frames (the thing that made nodes "expand to the right") —
    /// not a hand-rolled simulation. Scale is pinned to 1.0 so a graph-space growth shows in the measured
    /// rect instead of being hidden by auto-fit zoom.
    fn node_frame_widths(snarl: &mut Snarl<EdNode>, node: NodeId, frames: usize) -> Vec<f32> {
        let ctx = egui::Context::default();
        let mut caches = NodeCaches::default();
        let mut signals = ViewerSignals::default();
        let gpu_tex = std::collections::HashMap::new();
        let mut gpu_reqs = Vec::new();
        let mut hovered = None;
        let snarl_id = egui::Id::new("harness-snarl");
        // egui-snarl interacts each node's frame at this id (snarl id ⊕ node ⊕ "frame").
        let node_frame_id = snarl_id.with(("snarl-node", node)).with("frame");
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1200.0, 900.0))),
            ..Default::default()
        };
        let style = SnarlStyle { min_scale: Some(1.0), max_scale: Some(1.0), ..SnarlStyle::new() };
        let mut widths = Vec::with_capacity(frames);
        for _ in 0..frames {
            let _ = ctx.run(input.clone(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let mut viewer = Viewer {
                        caches: &mut caches,
                        signals: &mut signals,
                        gpu_tex: &gpu_tex,
                        gpu_reqs: &mut gpu_reqs,
                        hovered_preview: &mut hovered,
                        level_salt: 0,
                    };
                    SnarlWidget::new().id(snarl_id).style(style).show(snarl, &mut viewer, ui);
                });
            });
            if let Some(r) = ctx.read_response(node_frame_id) {
                widths.push(r.rect.width());
            }
            gpu_reqs.clear();
        }
        widths
    }

    /// Regression: a node must NOT grow every frame. egui-snarl feeds a node's measured width back as the
    /// body's available width, so any body widget that spans `available_width` (e.g. a horizontal
    /// `ui.separator()` in our vertical body) makes the node expand rightward without bound. Assert the
    /// rendered node width CONVERGES (stable across the last frames). With the `ui.separator()` this fails;
    /// with the fixed `ui.add_space` it passes.
    #[test]
    fn node_width_does_not_run_away() {
        let mut snarl: Snarl<EdNode> = Snarl::new();
        // An Fbm source node (arity 0) — `graph_rooted_at` compiles it directly, so the preview path runs.
        let node = snarl.insert_node(
            egui::pos2(0.0, 0.0),
            EdNode::Op(NodeKind::Fbm(FbmAxis { octaves: 3, base_freq: 1.0 / 512.0, lacunarity: 2.0, gain: 0.5, amplitude: 100.0, seed_salt: 1 })),
        );
        let widths = node_frame_widths(&mut snarl, node, 24);
        assert!(widths.len() >= 4, "node frame never rendered: {widths:?}");
        let n = widths.len();
        // A runaway grows by a fixed margin each frame; require the last three frames to be stable (≤1px).
        let stable = (widths[n - 1] - widths[n - 2]).abs() < 1.0 && (widths[n - 2] - widths[n - 3]).abs() < 1.0;
        assert!(stable, "node width runs away (grows each frame): {widths:?}");
    }
}
