//! Node-preview plumbing: the per-node view caches' helpers (GPU pool keys, on-image scroll/drag
//! gestures, the image widget), the dockable preview panel + its deferred dock-focus system, and the
//! popped-out floating preview windows. Both 2D and 3D previews render on the shared GPU pool.

use bevy::prelude::*;
use bevy_egui::egui;
use egui_snarl::{NodeId, Snarl};

use crate::sdf_render::worldgen::graph::node::Graph;

use crate::editor::worldgen_gpu_preview::{GpuPreviewRequest, GpuPreviewRequests, GpuPreviewTextures};
use super::{EdNode, WorldGraphEditor, graph_rooted_at, resolve_snarl};

/// Default on-screen size (points) of a node preview; adjustable per node via the size control.
pub(super) const DEFAULT_PREVIEW_PX: f32 = 120.0;
/// Default half-extent (metres) of the world window a preview samples, centred on the origin.
pub(super) const PREVIEW_HALF_M: f64 = 2048.0;
/// Default 3D orbit camera (yaw, pitch) in radians.
pub(super) const CAM_DEFAULT: (f32, f32) = (0.7, 0.6);

/// GPU pool key for the dockable preview instance `id`. Bit 62 is set so the keys can never collide with
/// the inline high-bit keys (bit 63) or the small pop-out ids (≥ 1000) — see [`gpu_inline_key`].
pub(super) fn preview_gpu_key(id: u64) -> u64 {
    (1u64 << 62) | id
}

/// The dynamic, id-keyed SET of independent dockable Node Preview panels (one per dock tab). "→ panel"
/// allocates a fresh id + tab every click ([`WorldgenPreviewPanels::open`]), so the count is unbounded;
/// closing a tab removes its entry (see the dock `on_close`). Each instance renders into its own GPU key
/// ([`preview_gpu_key`]) + dock tab `EditorTab::WorldgenPreview(id)`.
#[derive(Resource, Default)]
pub(super) struct WorldgenPreviewPanels {
    /// Live preview instances keyed by their unique id.
    pub(super) map: std::collections::HashMap<u64, WorldgenPreviewPanel>,
    /// Monotonic id source (never reused, so a closed tab's id can't alias a new one).
    pub(super) next_id: u64,
    /// Ids whose dock tab still needs creating — drained by [`open_preview_panel`] outside the dock
    /// render (the dock state is taken OUT of the World while the dock renders, so a panel callback can't
    /// touch it).
    pub(super) to_open: Vec<u64>,
}

impl WorldgenPreviewPanels {
    /// Spawn a NEW preview instance targeting `target` with `view`/`is3d`, queue its dock tab for
    /// creation, and return its fresh id. Each "→ panel" click (or restored `PanelView`) calls this.
    pub(super) fn open(&mut self, target: (Vec<NodeId>, NodeId), view: PreviewView, is3d: bool) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let mut panel = WorldgenPreviewPanel { target: Some(target), is3d, ..Default::default() };
        panel.set_view(view);
        self.map.insert(id, panel);
        self.to_open.push(id);
        id
    }
}

/// The dockable, viewport-located preview panel's state: which node it shows + its own view. One per
/// entry in [`WorldgenPreviewPanels::map`].
pub(super) struct WorldgenPreviewPanel {
    pub(super) target: Option<(Vec<NodeId>, NodeId)>,
    pub(super) half: f64,
    pub(super) cam: (f32, f32),
    pub(super) pan: (f64, f64),
    pub(super) is3d: bool,
}

impl Default for WorldgenPreviewPanel {
    fn default() -> Self {
        Self {
            target: None,
            half: PREVIEW_HALF_M,
            cam: CAM_DEFAULT,
            pan: (0.0, 0.0),
            is3d: true,
        }
    }
}

impl WorldgenPreviewPanel {
    /// The panel's view params (half/pan/cam) as a [`PreviewView`].
    pub(super) fn view(&self) -> PreviewView {
        PreviewView { half: self.half, cx: self.pan.0, cz: self.pan.1, yaw: self.cam.0, pitch: self.cam.1 }
    }
    /// Write a [`PreviewView`] back into the panel's half/pan/cam fields.
    pub(super) fn set_view(&mut self, v: PreviewView) {
        self.half = v.half;
        self.pan = (v.cx, v.cz);
        self.cam = (v.yaw, v.pitch);
    }
}

/// A preview's view parameters: the sampled world window (half-extent + XZ centre) and the 3D orbit
/// camera (yaw/pitch). The single carrier for on-image gestures ([`handle_preview_gestures`]) and for
/// building a [`GpuPreviewRequest`] ([`PreviewView::to_request`]).
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(super) struct PreviewView {
    pub(super) half: f64,
    pub(super) cx: f64,
    pub(super) cz: f64,
    pub(super) yaw: f32,
    pub(super) pitch: f32,
}

impl PreviewView {
    /// Build a GPU preview request for this view (the pool renders it next frame into key `key`).
    pub(super) fn to_request(self, key: u64, graph: Graph, is3d: bool, res: u32) -> GpuPreviewRequest {
        GpuPreviewRequest {
            key,
            graph,
            half: self.half,
            center: (self.cx, self.cz),
            is3d,
            yaw: self.yaw,
            pitch: self.pitch,
            res_w: res,
            res_h: res,
        }
    }
}

/// A node preview detached into its own floating window — carries its own nav path, view state, and
/// texture so it stays live across navigation independently of the in-graph preview.
pub(super) struct PoppedPreview {
    /// Stable id (the GPU pool slot key for this window — unchanged across rotate/zoom/nav).
    pub(super) id: u64,
    pub(super) nav: Vec<NodeId>,
    pub(super) node: NodeId,
    pub(super) half: f64,
    /// World-XZ pan centre (offset X/Y).
    pub(super) cx: f64,
    pub(super) cz: f64,
    pub(super) size: f32,
    pub(super) is3d: bool,
    pub(super) cam: (f32, f32),
    pub(super) open: bool,
}

impl PoppedPreview {
    /// This window's view params (half/cx/cz/cam) as a [`PreviewView`].
    pub(super) fn view(&self) -> PreviewView {
        PreviewView { half: self.half, cx: self.cx, cz: self.cz, yaw: self.cam.0, pitch: self.cam.1 }
    }
    /// Write a [`PreviewView`] back into this window's half/cx/cz/cam fields.
    fn set_view(&mut self, v: PreviewView) {
        self.half = v.half;
        self.cx = v.cx;
        self.cz = v.cz;
        self.cam = (v.yaw, v.pitch);
    }
}

/// Stable GPU pool key for an inline preview = nav-level salt ⊕ node id, with the top bit set so it can
/// never collide with the small pop-out window ids.
pub(super) fn gpu_inline_key(level_salt: u64, node: NodeId) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    level_salt.hash(&mut h);
    node.hash(&mut h);
    h.finish() | (1u64 << 63)
}

/// Hash of a nav path (the per-level salt for inline preview keys).
pub(super) fn nav_hash(nav: &[NodeId]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    nav.hash(&mut h);
    h.finish()
}

/// Apply (and CONSUME) scroll-zoom over a hovered preview image: zooms `half`, zeroes the ctx scroll so
/// the surrounding window/panel doesn't also scroll. (Inline snarl previews intercept scroll BEFORE the
/// snarl reads it — see `graph_panel` — because egui-snarl applies its own zoom before drawing nodes.)
pub(super) fn scroll_zoom_consume(ui: &egui::Ui, resp: &egui::Response, half: &mut f64) {
    if !resp.hovered() {
        return;
    }
    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
    apply_scroll_zoom(ui, scroll, half);
}

/// The shared scroll→zoom core (single source of the zoom curve + scroll-consume): given the vertical
/// scroll delta, zoom `half` and zero the ctx scroll so nothing else scrolls. Used by the response-based
/// [`scroll_zoom_consume`] and by `graph_panel`'s pre-show interception (which gates on the node hovered
/// last frame rather than a live `Response`).
pub(super) fn apply_scroll_zoom(ui: &egui::Ui, scroll: f32, half: &mut f64) {
    if scroll != 0.0 {
        ui.ctx().input_mut(|i| {
            i.smooth_scroll_delta = egui::Vec2::ZERO;
            i.raw_scroll_delta = egui::Vec2::ZERO;
        });
        *half = (*half * (1.0 - scroll as f64 * 0.0015)).clamp(20.0, 1_000_000.0);
    }
}

/// On-image drag gestures: left-drag = orbit (3D) / pan (2D), right-drag = pan (3D). `size` is the
/// on-screen image side (px). (Scroll-zoom is handled separately — see [`scroll_zoom_consume`].)
pub(super) fn handle_preview_gestures(resp: &egui::Response, is3d: bool, size: f32, view: &mut PreviewView) {
    let wpp = (2.0 * view.half) / size.max(1.0) as f64; // world units per display pixel
    if is3d {
        if resp.dragged_by(egui::PointerButton::Primary) {
            let d = resp.drag_delta();
            view.yaw += d.x * 0.01;
            view.pitch = (view.pitch - d.y * 0.01).clamp(0.05, 1.5);
        }
        if resp.dragged_by(egui::PointerButton::Secondary) {
            let d = resp.drag_delta();
            view.cx -= d.x as f64 * wpp;
            view.cz -= d.y as f64 * wpp;
        }
    } else if resp.dragged_by(egui::PointerButton::Primary) {
        let d = resp.drag_delta();
        view.cx -= d.x as f64 * wpp;
        view.cz -= d.y as f64 * wpp;
    }
}

/// Format the visible world width (`2 * half_m`, metres) as a scale-label string: `">= 1000 m"` reads
/// as kilometres (`"{:.1} km"`), below that as metres (`"{:.0} m"`). Pure — the testable core of
/// [`paint_scale_label`].
pub(super) fn scale_label_text(half_m: f64) -> String {
    let width_m = 2.0 * half_m;
    if width_m >= 1000.0 {
        format!("{:.1} km", width_m / 1000.0)
    } else {
        format!("{width_m:.0} m")
    }
}

/// Paint a small **scale label** — the visible world width (`2 * half_m`) — in the bottom-left corner
/// of `rect`, with a semi-opaque rounded backing for legibility. A reusable preview OVERLAY: one
/// helper, called at every preview draw site (inline node body, the dockable panel, pop-out windows),
/// so a future overlay (grid, compass, crosshair) plugs in the exact same way. Pure painting — no
/// state, no input.
pub(super) fn paint_scale_label(ui: &egui::Ui, rect: egui::Rect, half_m: f64) {
    let text = scale_label_text(half_m);
    let painter = ui.painter();
    // Scale the label with the preview: small inline previews get ~13px, large panel/pop-out up to 22px.
    let fs = (rect.width() * 0.055).clamp(13.0, 22.0);
    let font = egui::FontId::proportional(fs);
    // Lay the text out so the backing rect hugs it exactly (bottom-left corner, small inset).
    let galley = painter.layout_no_wrap(text, font, egui::Color32::from_gray(235));
    let pad = egui::vec2(4.0, 2.0);
    let pos = egui::pos2(rect.left() + 4.0, rect.bottom() - 4.0 - galley.size().y - pad.y * 2.0);
    let bg = egui::Rect::from_min_size(pos, galley.size() + pad * 2.0);
    painter.rect_filled(bg, 3.0, egui::Color32::from_black_alpha(150));
    painter.galley(pos + pad, galley, egui::Color32::from_gray(235));
}

/// A small **drag-resize grip** at the bottom-right corner of a preview `rect`: a ~14px square,
/// `Sense::drag()`, painted as two short diagonal lines. Returns the resize delta (px, the larger of the
/// drag's x/y components) while dragged, else `None` — the caller adds it to the preview's display size
/// (clamped). A reusable preview overlay (one helper, like [`paint_scale_label`]) so every preview can
/// gain a corner-resize handle the same way. The grip is its own widget rect, so its drag never triggers
/// the underlying image's orbit/pan gesture.
pub(super) fn preview_resize_grip(ui: &mut egui::Ui, rect: egui::Rect) -> Option<f32> {
    const GRIP: f32 = 14.0;
    let grip_rect = egui::Rect::from_min_max(rect.max - egui::vec2(GRIP, GRIP), rect.max);
    let resp = ui.interact(grip_rect, ui.id().with(("preview-resize-grip", rect.left_top().x as i32, rect.left_top().y as i32)), egui::Sense::drag());
    // Paint two short diagonal lines (the conventional resize grip), brightening on hover/drag.
    let bright = resp.hovered() || resp.dragged();
    let col = if bright { egui::Color32::from_gray(220) } else { egui::Color32::from_gray(150) };
    let stroke = egui::Stroke::new(1.5, col);
    let p = ui.painter();
    let br = grip_rect.right_bottom();
    for off in [4.0_f32, 9.0] {
        p.line_segment([egui::pos2(br.x - off, br.y - 2.0), egui::pos2(br.x - 2.0, br.y - off)], stroke);
    }
    if resp.hovered() || resp.dragged() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeNwSe);
    }
    if resp.dragged() {
        let d = resp.drag_delta();
        // Use the larger-magnitude axis so dragging in either direction feels natural.
        Some(if d.x.abs() >= d.y.abs() { d.x } else { d.y })
    } else {
        None
    }
}

/// Draw a preview image at `size`, or a flat "baking…" placeholder for the ~1 frame before the GPU pool
/// texture is ready. Returns the (click-and-drag-sensing) response so on-image gestures work either way.
pub(super) fn preview_image(ui: &mut egui::Ui, tex: Option<egui::TextureId>, size: egui::Vec2) -> egui::Response {
    match tex {
        Some(t) => ui.add(egui::Image::new(egui::load::SizedTexture::new(t, size)).sense(egui::Sense::click_and_drag())),
        None => {
            let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
            ui.painter().rect_filled(rect, 4.0, egui::Color32::from_gray(20));
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "baking…",
                egui::FontId::proportional(12.0),
                egui::Color32::from_gray(90),
            );
            resp
        }
    }
}

/// The dockable, viewport-located **Node Preview** panel for instance `id` — shows whichever node was
/// sent to it via "→ panel", large, with its own 2D/3D + zoom/pan/orbit (both rendered on the shared GPU
/// pool, into the instance's own key [`preview_gpu_key`]). A closed/missing instance degrades to a hint.
pub(super) fn preview_panel_impl(world: &mut World, ui: &mut egui::Ui, id: u64) {
    let gpu_key = preview_gpu_key(id);
    let Some(target) = world.resource::<WorldgenPreviewPanels>().map.get(&id).and_then(|p| p.target.clone()) else {
        ui.label("No preview targeted. In the Biome Graph, click a node preview's ▢ button to show it here.");
        return;
    };
    let (nav, node) = target;
    // Compile the targeted node's sub-graph from the editor snarl.
    let g = world.resource_scope::<WorldGraphEditor, Option<Graph>>(|_w, ed| {
        resolve_snarl(&ed.snarl, &nav).and_then(|s| graph_rooted_at(s, node).ok())
    });
    let Some(g) = g else {
        ui.label("the targeted node no longer exists");
        return;
    };

    world.resource_scope::<WorldgenPreviewPanels, ()>(|world, mut panels| {
        let Some(panel) = panels.map.get_mut(&id) else { return };
        ui.horizontal(|ui| {
            if ui.selectable_label(panel.is3d, "3D").on_hover_text("GPU 3D surface").clicked() {
                panel.is3d = !panel.is3d;
            }
            let mut km = panel.half * 2.0 / 1000.0;
            if ui.add(egui::DragValue::new(&mut km).speed(0.5).range(0.05..=512.0).suffix(" km")).changed() {
                panel.half = (km * 1000.0 / 2.0).max(1.0);
            }
            ui.add(egui::DragValue::new(&mut panel.pan.0).speed(10.0).prefix("X ").suffix(" m"));
            ui.add(egui::DragValue::new(&mut panel.pan.1).speed(10.0).prefix("Y ").suffix(" m"));
            ui.label("· drag orbit · right-drag pan · scroll zoom");
        });
        let ppp = ui.ctx().pixels_per_point();
        // Square preview sized to fit the panel (drag the dock edge to resize), centred in the leftover space.
        let avail = ui.available_size();
        let side = avail.x.min(avail.y).max(64.0);
        let res = ((side * ppp).round() as usize).max(32);
        let view = panel.view();
        world.resource_mut::<GpuPreviewRequests>().0.push(view.to_request(gpu_key, g, panel.is3d, res as u32));
        let tex = world.resource::<GpuPreviewTextures>().0.get(&gpu_key).copied();
        ui.vertical_centered(|ui| {
            let resp = preview_image(ui, tex, egui::vec2(side, side));
            paint_scale_label(ui, resp.rect, panel.half);
            scroll_zoom_consume(ui, &resp, &mut panel.half);
            let mut view = panel.view();
            handle_preview_gestures(&resp, panel.is3d, side, &mut view);
            panel.set_view(view);
        });
    });
}

/// Outside the dock render (when `EditorDockState` is back in the World), CREATE + focus a dock tab for
/// each id queued in [`WorldgenPreviewPanels::to_open`] (drained here). The dock state is taken OUT of
/// the World during its own render, so tab creation can't happen from a panel callback — this deferred
/// system is the safe point. Mirrors how scenes add center tabs.
pub(super) fn open_preview_panel(world: &mut World) {
    // Drain the queue. If the dock state isn't present yet (pre-`init_dock_state`), keep the ids queued
    // so they open once it exists.
    if !world.contains_resource::<crate::editor::dock::EditorDockState>() {
        return;
    }
    let to_open: Vec<u64> = std::mem::take(&mut world.resource_mut::<WorldgenPreviewPanels>().to_open);
    if to_open.is_empty() {
        return;
    }
    let mut dock = world.resource_mut::<crate::editor::dock::EditorDockState>();
    for id in to_open {
        let tab = crate::editor::dock::EditorTab::WorldgenPreview(id);
        crate::editor::scene_tabs::add_center_tab(&mut dock, tab.clone());
        if let Some((n, t)) = dock.state.find_main_surface_tab(&tab) {
            dock.state.set_active_tab((egui_dock::SurfaceIndex::main(), n, t));
        }
    }
}

/// Draw one popped-out preview as a floating, resizable `egui::Window`. Both 2D and 3D render on the
/// shared GPU pool (push a request, draw last frame's texture). `gpu_tex` is last frame's pool output.
pub(super) fn popped_preview_window(
    ui: &egui::Ui,
    p: &mut PoppedPreview,
    root: &Snarl<EdNode>,
    gpu_tex: &std::collections::HashMap<u64, egui::TextureId>,
    gpu_reqs: &mut Vec<GpuPreviewRequest>,
) {
    let mut open = p.open;
    egui::Window::new(format!("Preview {}", p.id))
        .id(egui::Id::new(("wg-pop", p.id)))
        .open(&mut open)
        .resizable(true)
        .default_size([p.size + 80.0, p.size + 60.0])
        .show(ui.ctx(), |ui| {
            let g = match resolve_snarl(root, &p.nav).map(|s| graph_rooted_at(s, p.node)) {
                Some(Ok(g)) => g,
                _ => {
                    ui.colored_label(egui::Color32::from_rgb(200, 150, 120), "node no longer exists");
                    return;
                }
            };
            ui.horizontal(|ui| {
                if ui.selectable_label(p.is3d, "3D").on_hover_text("GPU 3D surface (drag to orbit)").clicked() {
                    p.is3d = !p.is3d;
                }
                let mut km = p.half * 2.0 / 1000.0;
                if ui.add(egui::DragValue::new(&mut km).speed(0.25).range(0.05..=512.0).suffix(" km")).changed() {
                    p.half = (km * 1000.0 / 2.0).max(1.0);
                }
            });
            if p.is3d {
                ui.horizontal(|ui| {
                    ui.label("offset");
                    ui.add(egui::DragValue::new(&mut p.cx).speed(10.0).prefix("X ").suffix(" m"));
                    ui.add(egui::DragValue::new(&mut p.cz).speed(10.0).prefix("Y ").suffix(" m"));
                    if ui.button("center").clicked() {
                        p.cx = 0.0;
                        p.cz = 0.0;
                    }
                });
            }

            let ppp = ui.ctx().pixels_per_point();
            // Square preview sized to fit the window (drag its edge to resize), centred in the leftover space.
            let avail = ui.available_size();
            let side = avail.x.min(avail.y).max(64.0);
            let res = ((side * ppp).round() as usize).max(32);
            // GPU path (2D + 3D): request a render for next frame; draw last frame's pool texture.
            gpu_reqs.push(p.view().to_request(p.id, g, p.is3d, res as u32));
            let tex = gpu_tex.get(&p.id).copied();
            ui.vertical_centered(|ui| {
                let resp = preview_image(ui, tex, egui::vec2(side, side));
                paint_scale_label(ui, resp.rect, p.half);
                scroll_zoom_consume(ui, &resp, &mut p.half);
                let mut view = p.view();
                handle_preview_gestures(&resp, p.is3d, side, &mut view);
                p.set_view(view);
            });
        });
    p.open = open;
}
