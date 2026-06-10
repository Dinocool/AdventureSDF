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
/// Fixed GPU pool key for the dockable preview panel (distinct from inline high-bit keys + pop-out ids).
pub(super) const PANEL_GPU_KEY: u64 = 7;

/// The dockable, viewport-located preview panel's state: which node it shows + its own view.
#[derive(Resource)]
pub(super) struct WorldgenPreviewPanel {
    pub(super) target: Option<(Vec<NodeId>, NodeId)>,
    pub(super) half: f64,
    pub(super) cam: (f32, f32),
    pub(super) pan: (f64, f64),
    pub(super) is3d: bool,
    /// Set by "→ panel"; a system outside the dock render ensures + focuses the tab (the dock state is
    /// taken OUT of the World while the dock renders, so it can't be touched from a panel callback).
    pub(super) pending_open: bool,
}

impl Default for WorldgenPreviewPanel {
    fn default() -> Self {
        Self {
            target: None,
            half: PREVIEW_HALF_M,
            cam: CAM_DEFAULT,
            pan: (0.0, 0.0),
            is3d: true,
            pending_open: false,
        }
    }
}

impl WorldgenPreviewPanel {
    /// The panel's view params (half/pan/cam) as a [`PreviewView`].
    fn view(&self) -> PreviewView {
        PreviewView { half: self.half, cx: self.pan.0, cz: self.pan.1, yaw: self.cam.0, pitch: self.cam.1 }
    }
    /// Write a [`PreviewView`] back into the panel's half/pan/cam fields.
    fn set_view(&mut self, v: PreviewView) {
        self.half = v.half;
        self.pan = (v.cx, v.cz);
        self.cam = (v.yaw, v.pitch);
    }
}

/// A preview's view parameters: the sampled world window (half-extent + XZ centre) and the 3D orbit
/// camera (yaw/pitch). The single carrier for on-image gestures ([`handle_preview_gestures`]) and for
/// building a [`GpuPreviewRequest`] ([`PreviewView::to_request`]).
#[derive(Clone, Copy)]
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
    fn view(&self) -> PreviewView {
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

/// The dockable, viewport-located **Node Preview** panel — shows whichever node was sent via "→ panel",
/// large, with its own 2D/3D + zoom/pan/orbit (both rendered on the shared GPU pool).
pub(super) fn preview_panel(world: &mut World, ui: &mut egui::Ui) {
    let Some((nav, node)) = world.resource::<WorldgenPreviewPanel>().target.clone() else {
        ui.label("No preview targeted. In the Biome Graph, click a node preview's ▢ button to show it here.");
        return;
    };
    // Compile the targeted node's sub-graph from the editor snarl.
    let g = world.resource_scope::<WorldGraphEditor, Option<Graph>>(|_w, ed| {
        resolve_snarl(&ed.snarl, &nav).and_then(|s| graph_rooted_at(s, node).ok())
    });
    let Some(g) = g else {
        ui.label("the targeted node no longer exists");
        return;
    };

    world.resource_scope::<WorldgenPreviewPanel, ()>(|world, mut panel| {
        let panel = &mut *panel; // reborrow once so disjoint field borrows don't alias through Mut's deref
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
        // Fill the panel non-square (drag the dock edge to resize); render res tracks the on-screen size.
        // Square preview sized to fit the panel (drag the dock edge to resize), centred in the leftover space.
        let avail = ui.available_size();
        let side = avail.x.min(avail.y).max(64.0);
        let res = ((side * ppp).round() as usize).max(32);
        let view = panel.view();
        world.resource_mut::<GpuPreviewRequests>().0.push(view.to_request(PANEL_GPU_KEY, g, panel.is3d, res as u32));
        let tex = world.resource::<GpuPreviewTextures>().0.get(&PANEL_GPU_KEY).copied();
        ui.vertical_centered(|ui| {
            let resp = preview_image(ui, tex, egui::vec2(side, side));
            scroll_zoom_consume(ui, &resp, &mut panel.half);
            let mut view = panel.view();
            handle_preview_gestures(&resp, panel.is3d, side, &mut view);
            panel.set_view(view);
        });
    });
}

/// Outside the dock render (when `EditorDockState` is back in the World), ensure + focus the dockable
/// Node Preview tab if "→ panel" was requested this/last frame.
pub(super) fn open_preview_panel(world: &mut World) {
    if !world.resource::<WorldgenPreviewPanel>().pending_open {
        return;
    }
    world.resource_mut::<WorldgenPreviewPanel>().pending_open = false;
    if !world.contains_resource::<crate::editor::dock::EditorDockState>() {
        return;
    }
    let tab = crate::editor::dock::EditorTab::Registered("worldgen/node-preview".into());
    crate::editor::layout::set_panel_present(world, tab.clone(), crate::editor::panels::DockSide::Center, true);
    if let Some(mut dock) = world.get_resource_mut::<crate::editor::dock::EditorDockState>()
        && let Some((n, t)) = dock.state.find_main_surface_tab(&tab)
    {
        dock.state.set_active_tab((egui_dock::SurfaceIndex::main(), n, t));
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
                scroll_zoom_consume(ui, &resp, &mut p.half);
                let mut view = p.view();
                handle_preview_gestures(&resp, p.is3d, side, &mut view);
                p.set_view(view);
            });
        });
    p.open = open;
}
