//! Runtime-toggleable Chrome tracing.
//!
//! `bevy/trace_chrome` installs an always-on chrome layer that captures the WHOLE session.
//! We don't want that — a profiler you can't turn off floods a huge JSON file every run.
//! Instead we keep `bevy/trace` (so Bevy still emits its per-system / render-graph spans)
//! and install our OWN chrome layer here, via [`LogPlugin::custom_layer`], gated behind an
//! atomic flag. While the flag is off the layer receives nothing; the editor's Performance
//! panel flips it to start/stop capture live (see [`capture_ui`]).
//!
//! The toggle works because we gate with [`dynamic_filter_fn`] (not `filter_fn`): a dynamic
//! filter reports `Interest::sometimes()`, so `tracing` re-asks per span/event instead of
//! caching the answer at the callsite — letting the flag take effect mid-run.

use std::sync::atomic::{AtomicBool, Ordering};

use bevy::log::BoxedLayer;
use bevy::log::tracing_subscriber::{Layer, filter::dynamic_filter_fn};
use bevy::prelude::*;
use bevy_egui::egui;

/// Whether the chrome layer currently records. Read on the logging thread for every span /
/// event (via the dynamic filter, so it's re-checked live) and written by the editor toggle.
/// Off by default — capture is opt-in.
static CAPTURING: AtomicBool = AtomicBool::new(false);

/// The trace file this run writes to, surfaced in the editor so the user knows where to
/// look. Inserted once when the layer is built (start of the run).
#[derive(Resource)]
pub struct ChromeTraceFile(pub String);

/// `tracing-chrome`'s flush guard. Held as a non-send resource so it lives for the whole
/// App and its `Drop` (which flushes + closes the JSON array) runs at shutdown. The guard
/// isn't `Sync`, hence non-send rather than a plain `Resource`.
struct TraceFlushGuard(#[allow(dead_code)] tracing_chrome::FlushGuard);

/// Is chrome capture currently recording?
pub fn is_capturing() -> bool {
    CAPTURING.load(Ordering::Relaxed)
}

/// Start (`true`) or pause (`false`) chrome capture. Takes effect on the next span/event.
pub fn set_capturing(on: bool) {
    CAPTURING.store(on, Ordering::Relaxed);
}

/// [`LogPlugin::custom_layer`] hook (wired in `main`). Builds the gated chrome layer and
/// stashes its flush guard + file path on the `App`. Runs once, very early, during
/// `DefaultPlugins` build. Returns `None` if the trace file can't be created — capture is
/// then silently unavailable and the rest of logging is unaffected.
pub fn custom_layer(app: &mut App) -> Option<BoxedLayer> {
    // A known filename (rather than tracing-chrome's internal default) so the editor can
    // show it. Matches `main::prune_old_traces`' `trace-*.json` glob for retention.
    let micros = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros())
        .unwrap_or(0);
    let path = format!("./trace-{micros}.json");

    let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
        .file(&path)
        .include_args(true)
        .build();

    app.insert_non_send_resource(TraceFlushGuard(guard));
    app.insert_resource(ChromeTraceFile(path));

    // Gate the chrome layer on `CAPTURING`. `dynamic_filter_fn` (not `filter_fn`) is the
    // crux: it re-evaluates per span/event, so toggling the flag at runtime works. While the
    // flag is false the chrome layer is fed nothing, so an idle capture costs ~a branch.
    Some(
        chrome_layer
            .with_filter(dynamic_filter_fn(|_meta, _cx| is_capturing()))
            .boxed(),
    )
}

/// F6 → start/stop chrome capture (sibling to the F7 RenderDoc capture and the F11 Nsight
/// GPU-Trace trigger). Global, not gated to a scene, so you can capture any frame. Logs the
/// new state + where the trace is written.
pub fn toggle_on_f6(keyboard: Res<ButtonInput<KeyCode>>, file: Option<Res<ChromeTraceFile>>) {
    if !keyboard.just_pressed(KeyCode::F6) {
        return;
    }
    let on = !is_capturing();
    set_capturing(on);
    match (on, file) {
        (true, Some(f)) => info!("Chrome trace capture started → {}", f.0),
        (true, None) => info!("Chrome trace capture started"),
        (false, _) => info!("Chrome trace capture paused"),
    }
}

/// The capture toggle shown in the editor's Performance panel: a checkbox plus the target
/// file and a reminder of how to read it. The checkbox drives the global flag directly.
pub fn capture_ui(world: &mut World, ui: &mut egui::Ui) {
    let mut on = is_capturing();
    if ui
        .checkbox(&mut on, "Chrome trace capture")
        .on_hover_text(
            "Record Bevy system + render-graph spans to a Chrome trace. Off by default; \
             enable only while profiling. Shortcut: F6.",
        )
        .changed()
    {
        set_capturing(on);
        info!(
            "Chrome trace capture {}",
            if on { "started" } else { "paused" }
        );
    }

    if on {
        ui.colored_label(egui::Color32::from_rgb(120, 220, 120), "● recording");
    } else {
        ui.weak("idle — enable to record this run's spans");
    }

    if let Some(file) = world.get_resource::<ChromeTraceFile>() {
        ui.weak(format!("→ {}", file.0));
    }
    ui.weak("Open the .json in Perfetto or chrome://tracing after exit.");
}
