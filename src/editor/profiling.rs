use bevy::diagnostic::DiagnosticsStore;
use bevy::prelude::*;
use bevy_egui::egui;

/// How many per-frame samples the perf graph retains. Sized for high refresh rates so
/// the graph condenses rather than scrolling too fast — ~1000 frames is ~6 s at 165 fps
/// (and ~16 s at 60 fps).
const HISTORY: usize = 1000;

#[derive(Resource, Default)]
pub struct ShaderProfilingData {
    pub fps_smoothed: f64,
    pub frame_time_ms: f64,
    pub frame_count: u64,
    /// Rolling per-frame history (oldest first), capped at [`HISTORY`].
    pub fps_history: std::collections::VecDeque<f32>,
    pub frame_time_history: std::collections::VecDeque<f32>,
}

pub struct ProfilingPlugin;

impl Plugin for ProfilingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShaderProfilingData>()
            .add_systems(Update, collect_profiling_data);
    }
}

fn collect_profiling_data(
    mut data: ResMut<ShaderProfilingData>,
    diagnostics: Res<DiagnosticsStore>,
) {
    data.frame_count += 1;

    if let Some(fps_diagnostic) =
        diagnostics.get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FPS)
        && let Some(smoothed) = fps_diagnostic.smoothed()
    {
        data.fps_smoothed = smoothed;
    }

    if let Some(ft_diagnostic) =
        diagnostics.get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FRAME_TIME)
        && let Some(smoothed) = ft_diagnostic.smoothed()
    {
        data.frame_time_ms = smoothed;
    }

    // Push this frame's samples, trimming to the history window.
    let fps = data.fps_smoothed as f32;
    let ft = data.frame_time_ms as f32;
    data.fps_history.push_back(fps);
    data.frame_time_history.push_back(ft);
    while data.fps_history.len() > HISTORY {
        data.fps_history.pop_front();
    }
    while data.frame_time_history.len() > HISTORY {
        data.frame_time_history.pop_front();
    }
}

/// The dedicated Performance dock panel: current readout + a shared graph plotting both
/// the FPS history and the frame-time history (two series, independently auto-scaled, in
/// one graph area), followed by the SDF GPU-atlas stats. The whole panel scrolls.
pub fn performance_panel(world: &mut World, ui: &mut egui::Ui) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            perf_graph_ui(world, ui);

            // Chrome-trace capture toggle (off by default). Records Bevy system /
            // render-graph spans to a JSON file only while enabled.
            ui.separator();
            ui.heading("Profiling capture");
            crate::editor::chrome_trace::capture_ui(world, ui);

            // GPU / SDF atlas stats, rolled in from the (now-removed) SDF Atlas tab.
            ui.separator();
            ui.heading("GPU");
            let stats = world.resource::<crate::sdf_render::debug::SdfAtlasStats>();
            crate::sdf_render::debug::atlas_stats_ui(stats, ui);
        });
}

/// The FPS + frame-time readout and graph.
fn perf_graph_ui(world: &mut World, ui: &mut egui::Ui) {
    let data = world.resource::<ShaderProfilingData>();
    let fps: Vec<f32> = data.fps_history.iter().copied().collect();
    let frame_time: Vec<f32> = data.frame_time_history.iter().copied().collect();
    let (cur_fps, cur_ft) = (data.fps_smoothed, data.frame_time_ms);

    let fps_color = egui::Color32::from_rgb(120, 220, 120);
    let ft_color = egui::Color32::from_rgb(240, 180, 90);

    // Stable axis ceilings (each series fills the graph against its own ceiling):
    //  - FPS: snapped to the next common refresh-rate tier above the peak, so the axis
    //    doesn't rescale every frame and a 165 fps line sits near the top, not pinned.
    //  - Frame time: the max over the whole (long) history, rounded up to a nice ms — a
    //    transient spike sets a steady ceiling rather than making the curve jump around.
    let peak_fps = fps.iter().copied().fold(0.0_f32, f32::max);
    let fps_ceiling = fps_axis_ceiling(peak_fps);
    let peak_ft = frame_time.iter().copied().fold(0.0_f32, f32::max);
    let ft_ceiling = ft_axis_ceiling(peak_ft);

    ui.horizontal(|ui| {
        ui.colored_label(fps_color, format!("{cur_fps:.1} FPS"));
        ui.separator();
        ui.colored_label(ft_color, format!("{cur_ft:.2} ms"));
    });

    // Shared graph area: both series share the rect, each scaled to its own ceiling so
    // they're readable despite different units.
    let (rect, _resp) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 140.0),
        egui::Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, egui::Color32::from_gray(20));

    draw_series(&painter, rect, &fps, fps_ceiling, fps_color);
    draw_series(&painter, rect, &frame_time, ft_ceiling, ft_color);

    // Legend, showing each series' current axis ceiling.
    ui.horizontal(|ui| {
        ui.colored_label(fps_color, format!("■ FPS (0–{fps_ceiling:.0})"));
        ui.colored_label(ft_color, format!("■ Frame time (0–{ft_ceiling:.1} ms)"));
    });
}

/// Snap a peak FPS up to the next common refresh-rate tier, so the FPS axis is stable and
/// a typical max line sits near (not at) the top.
fn fps_axis_ceiling(peak: f32) -> f32 {
    const TIERS: [f32; 7] = [60.0, 90.0, 120.0, 144.0, 165.0, 240.0, 360.0];
    for t in TIERS {
        if peak <= t {
            return t;
        }
    }
    // Above the highest tier: round up to the next 60.
    (peak / 60.0).ceil() * 60.0
}

/// Round a peak frame time up to a "nice" ms ceiling (1/2/5 × 10ⁿ), giving the
/// frame-time axis a steady headroom that a brief spike won't keep nudging.
fn ft_axis_ceiling(peak: f32) -> f32 {
    let peak = peak.max(1.0);
    let pow = 10f32.powf(peak.log10().floor());
    for step in [1.0, 2.0, 5.0, 10.0] {
        let c = step * pow;
        if peak <= c {
            return c;
        }
    }
    pow * 10.0
}

/// Plot one series as a polyline filling `rect` horizontally, scaled to `ceiling`.
/// Empty series draw nothing.
fn draw_series(
    painter: &egui::Painter,
    rect: egui::Rect,
    series: &[f32],
    ceiling: f32,
    color: egui::Color32,
) {
    if series.len() < 2 {
        return;
    }
    let ceiling = ceiling.max(1e-3);
    let n = series.len();
    let pts: Vec<egui::Pos2> = series
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let x = rect.left() + rect.width() * (i as f32 / (n - 1) as f32);
            let y = rect.bottom() - rect.height() * (v / ceiling).clamp(0.0, 1.0);
            egui::pos2(x, y)
        })
        .collect();
    painter.add(egui::Shape::line(pts, egui::Stroke::new(1.5, color)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fps_ceiling_snaps_to_refresh_tiers() {
        assert_eq!(fps_axis_ceiling(58.0), 60.0);
        assert_eq!(fps_axis_ceiling(165.0), 165.0);
        assert_eq!(fps_axis_ceiling(150.0), 165.0);
        assert_eq!(fps_axis_ceiling(200.0), 240.0);
        assert_eq!(fps_axis_ceiling(400.0), 420.0); // above tiers → next 60
    }

    #[test]
    fn ft_ceiling_rounds_to_nice_ms() {
        assert_eq!(ft_axis_ceiling(8.0), 10.0);
        assert_eq!(ft_axis_ceiling(12.0), 20.0);
        assert_eq!(ft_axis_ceiling(33.0), 50.0);
        assert_eq!(ft_axis_ceiling(0.5), 1.0); // clamped floor
    }
}
