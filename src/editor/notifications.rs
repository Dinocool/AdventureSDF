//! Transient toast notifications. A small stack of messages in the top-right corner that
//! fade out after a few seconds; newest on top, capped to [`MAX_TOASTS`] (oldest dropped),
//! each with an `✕` to dismiss early. Push from anywhere with `&mut World`:
//!
//! ```ignore
//! world.resource_mut::<Notifications>().success("Scene saved");
//! ```

use bevy::prelude::*;
use bevy_egui::egui;

/// Most toasts shown at once; pushing past this drops the oldest.
const MAX_TOASTS: usize = 5;
/// Default seconds a toast is visible (including its fade-out tail).
const DEFAULT_DURATION: f32 = 4.0;
/// Fade-out tail length (seconds) and fade-in ramp (seconds).
const FADE_OUT: f32 = 0.6;
const FADE_IN: f32 = 0.15;

/// Accent/severity of a toast.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NotifyKind {
    Info,
    Success,
    Warning,
    Error,
}

impl NotifyKind {
    /// Leading icon glyph.
    fn icon(self) -> &'static str {
        match self {
            NotifyKind::Info => "\u{2139}",    // ℹ
            NotifyKind::Success => "\u{2714}", // ✔
            NotifyKind::Warning => "\u{26A0}", // ⚠
            NotifyKind::Error => "\u{2716}",   // ✖
        }
    }

    /// Accent colour for the icon + side bar.
    fn accent(self) -> egui::Color32 {
        match self {
            NotifyKind::Info => egui::Color32::from_rgb(90, 160, 255),
            NotifyKind::Success => egui::Color32::from_rgb(90, 200, 120),
            NotifyKind::Warning => egui::Color32::from_rgb(240, 190, 90),
            NotifyKind::Error => egui::Color32::from_rgb(240, 110, 100),
        }
    }
}

struct Toast {
    id: u64,
    kind: NotifyKind,
    message: String,
    /// Seconds this toast has been alive (advanced each editor frame).
    age: f32,
    duration: f32,
}

/// The active toast stack. Insert with the helper methods; [`notifications_ui`] renders +
/// ages them.
#[derive(Resource, Default)]
pub struct Notifications {
    toasts: Vec<Toast>,
    next_id: u64,
}

impl Notifications {
    /// Push a toast of `kind`. Caps the stack at [`MAX_TOASTS`], dropping the oldest.
    pub fn push(&mut self, kind: NotifyKind, message: impl Into<String>) {
        let id = self.next_id;
        self.next_id += 1;
        self.toasts.push(Toast {
            id,
            kind,
            message: message.into(),
            age: 0.0,
            duration: DEFAULT_DURATION,
        });
        if self.toasts.len() > MAX_TOASTS {
            self.toasts.remove(0);
        }
    }

    pub fn info(&mut self, message: impl Into<String>) {
        self.push(NotifyKind::Info, message);
    }
    pub fn success(&mut self, message: impl Into<String>) {
        self.push(NotifyKind::Success, message);
    }
    pub fn warning(&mut self, message: impl Into<String>) {
        self.push(NotifyKind::Warning, message);
    }
    pub fn error(&mut self, message: impl Into<String>) {
        self.push(NotifyKind::Error, message);
    }
}

/// Opacity for a toast given its age: quick ramp in, hold, then fade out over the tail.
fn fade_alpha(age: f32, duration: f32) -> f32 {
    let fade_in = (age / FADE_IN).clamp(0.0, 1.0);
    let fade_out = ((duration - age) / FADE_OUT).clamp(0.0, 1.0);
    fade_in.min(fade_out)
}

/// Age, expire, and render the toast stack. Call once per editor frame with the primary egui
/// context. No-op when there are no toasts.
pub fn notifications_ui(world: &mut World, ctx: &egui::Context) {
    let dt = world.resource::<Time>().delta_secs();
    {
        let mut notes = world.resource_mut::<Notifications>();
        for toast in &mut notes.toasts {
            toast.age += dt;
        }
        notes.toasts.retain(|t| t.age < t.duration);
    }

    if world.resource::<Notifications>().toasts.is_empty() {
        return;
    }
    // Keep animating (fade) even if the app would otherwise idle this frame.
    ctx.request_repaint();

    // Snapshot what to draw so the egui closure doesn't borrow the resource. Oldest first so
    // the newest toast sits nearest the bottom-right corner; the stack grows upward.
    let drawn: Vec<(u64, NotifyKind, String, f32)> = world
        .resource::<Notifications>()
        .toasts
        .iter()
        .map(|t| (t.id, t.kind, t.message.clone(), fade_alpha(t.age, t.duration)))
        .collect();

    let mut dismiss: Option<u64> = None;
    egui::Area::new(egui::Id::new("editor_toasts"))
        // Bottom-right, lifted above the status bar.
        .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-10.0, -36.0))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            ui.set_max_width(320.0);
            for (id, kind, message, alpha) in &drawn {
                ui.scope(|ui| {
                    ui.set_opacity(*alpha);
                    egui::Frame::new()
                        .fill(ui.visuals().panel_fill)
                        .stroke(egui::Stroke::new(1.0, kind.accent()))
                        .corner_radius(6.0)
                        .inner_margin(egui::Margin::symmetric(10, 8))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.colored_label(kind.accent(), kind.icon());
                                ui.add(egui::Label::new(message).wrap());
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui.small_button("\u{2716}").clicked() {
                                            dismiss = Some(*id);
                                        }
                                    },
                                );
                            });
                        });
                });
                ui.add_space(6.0);
            }
        });

    if let Some(id) = dismiss {
        world
            .resource_mut::<Notifications>()
            .toasts
            .retain(|t| t.id != id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_caps_at_max_and_drops_oldest() {
        let mut n = Notifications::default();
        for i in 0..(MAX_TOASTS + 3) {
            n.success(format!("msg {i}"));
        }
        assert_eq!(n.toasts.len(), MAX_TOASTS);
        // Oldest (msg 0..=2) dropped; the newest is the last pushed.
        assert_eq!(n.toasts.last().unwrap().message, format!("msg {}", MAX_TOASTS + 2));
        assert_eq!(n.toasts.first().unwrap().message, "msg 3");
    }

    #[test]
    fn fade_ramps_in_and_out() {
        // Mid-life: fully opaque.
        assert_eq!(fade_alpha(2.0, 4.0), 1.0);
        // Just spawned: ramping in.
        assert!(fade_alpha(0.0, 4.0) < 1.0);
        // Near the end: ramping out.
        assert!(fade_alpha(3.9, 4.0) < 1.0);
        assert!(fade_alpha(4.0, 4.0) <= 0.0);
    }
}
