use std::collections::{BTreeMap, VecDeque};

use bevy::diagnostic::DiagnosticsStore;
use bevy::prelude::*;
use bevy_egui::egui;

/// Most contributors drawn as their own band in the stacked graph; the remainder are folded
/// into a single "other" band so the legend stays readable.
const TOP_N: usize = 10;

/// Committed columns the stacked graph draws (one per time-bucket). Bounds the mesh vertex
/// count (two verts per column per band) regardless of panel width.
const GRAPH_COLS: usize = 240;

/// Wall-clock span the graph shows. With [`GRAPH_COLS`] columns this fixes the bucket length
/// (≈ 33 ms / ~30 Hz) and therefore the scroll speed — and crucially it's tied to real time,
/// not the render frame rate, so the graph scrolls at the same pace at 60 fps or 165 fps.
const GRAPH_WINDOW_SECS: f32 = 8.0;

/// Seconds of wall-clock per committed column (one time-bucket). Derived so window = cols ×
/// interval. The graph advances one column per this many seconds, sliding fractionally in
/// between so it glides rather than steps.
const COMMIT_INTERVAL_SECS: f32 = GRAPH_WINDOW_SECS / GRAPH_COLS as f32;

#[derive(Resource, Default)]
pub struct ShaderProfilingData {
    pub fps_smoothed: f64,
    pub frame_time_ms: f64,
    pub frame_count: u64,
}

/// Host + process RAM, in bytes, sampled from `sysinfo` (see [`sample_memory`]). Zero until
/// the first sample lands. We query sysinfo ourselves rather than via Bevy's
/// `SystemInformationDiagnosticsPlugin` because that only exposes host memory as a percentage
/// (no byte total), which can't drive a byte-level breakdown.
#[derive(Resource, Default)]
pub struct MemoryStats {
    pub process_rss: u64,
    pub system_used: u64,
    pub system_total: u64,
}

/// How often to re-sample host/process RAM. Sampling walks the OS, so we throttle it well
/// below the frame rate — memory moves slowly and this keeps the cost off the frame budget.
const MEMORY_SAMPLE_SECS: f32 = 1.0;

/// Refresh [`MemoryStats`] from `sysinfo` at most every [`MEMORY_SAMPLE_SECS`]. Keeps the
/// `System` handle in a `Local` so allocations/OS handles persist across samples. Only the
/// current process is refreshed (cheap) plus the global memory figures.
fn sample_memory(
    mut stats: ResMut<MemoryStats>,
    mut sys: Local<Option<sysinfo::System>>,
    mut since: Local<f32>,
    time: Res<Time<Real>>,
) {
    *since += time.delta_secs();
    if sys.is_some() && *since < MEMORY_SAMPLE_SECS {
        return;
    }
    *since = 0.0;
    let system = sys.get_or_insert_with(sysinfo::System::new);
    system.refresh_memory();
    stats.system_total = system.total_memory();
    stats.system_used = system.used_memory();
    if let Ok(pid) = sysinfo::get_current_pid() {
        system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        stats.process_rss = system.process(pid).map_or(0, |p| p.memory());
    }
}

/// Rolling breakdown of where compute goes: GPU render passes (timestamp queries) and tagged
/// CPU sections (via [`crate::instrument`]), in milliseconds. Per-frame samples are averaged
/// into fixed-duration time buckets ("columns") so the graph's scroll speed is decoupled from
/// the render frame rate; each column's history is kept the same length (zero-padded) so a
/// given index is the same bucket across all labels — the alignment that lets them stack.
#[derive(Resource, Default)]
pub struct FrameCostHistory {
    /// label → committed per-bucket ms (oldest first), capped at [`GRAPH_COLS`]. Labels are
    /// display-ready (e.g. `gpu: gbuffer`, `cpu: bake schedule`).
    series: BTreeMap<String, VecDeque<f32>>,
    /// Sum of this (in-progress) bucket's per-frame samples, and how many frames went in. On
    /// commit, each label's mean (`sum / frames`) becomes the next column.
    pending: BTreeMap<String, f32>,
    pending_frames: u32,
    /// Wall-clock accumulated toward the next commit; `phase` is its fraction of an interval,
    /// used to slide the drawn columns sub-column for smooth scrolling.
    accum_secs: f32,
    phase: f32,
}

impl FrameCostHistory {
    /// Committed columns currently retained (every series has this length).
    fn len(&self) -> usize {
        self.series.values().next().map_or(0, VecDeque::len)
    }

    /// Fold this frame's `samples` (label → ms) into the pending bucket and advance wall-clock
    /// by `dt`. Commits one or more columns once enough time has elapsed. `dt` decoupling is
    /// what keeps the scroll speed framerate-independent.
    fn accumulate(&mut self, samples: &BTreeMap<String, f32>, dt: f32) {
        for (label, &v) in samples {
            *self.pending.entry(label.clone()).or_default() += v;
        }
        self.pending_frames += 1;
        self.accum_secs += dt.max(0.0);
        // Commit whole buckets; cap iterations so a long stall (huge dt) can't spin forever.
        let mut guard = 0;
        while self.accum_secs >= COMMIT_INTERVAL_SECS && guard < GRAPH_COLS {
            self.commit();
            self.accum_secs -= COMMIT_INTERVAL_SECS;
            guard += 1;
        }
        if guard == GRAPH_COLS {
            self.accum_secs = 0.0; // drop the backlog after a big stall
        }
        self.phase = (self.accum_secs / COMMIT_INTERVAL_SECS).clamp(0.0, 1.0);
    }

    /// Push one column: each label's bucket mean. New labels are back-filled with zeros so all
    /// series stay aligned; series that are all-zero across the window are pruned.
    fn commit(&mut self) {
        let frames = self.pending_frames.max(1) as f32;
        let len = self.len();
        for label in self.pending.keys() {
            self.series.entry(label.clone()).or_insert_with(|| {
                let mut q = VecDeque::with_capacity(len + 1);
                q.extend(std::iter::repeat_n(0.0, len));
                q
            });
        }
        for (label, q) in self.series.iter_mut() {
            q.push_back(self.pending.get(label).copied().unwrap_or(0.0) / frames);
            while q.len() > GRAPH_COLS {
                q.pop_front();
            }
        }
        self.pending.clear();
        self.pending_frames = 0;
        self.series.retain(|_, q| q.iter().any(|&v| v > 1e-4));
    }
}

pub struct ProfilingPlugin;

impl Plugin for ProfilingPlugin {
    fn build(&self, app: &mut App) {
        // Turn on CPU span collection for the tagged core systems (no-op until now).
        crate::instrument::set_enabled(true);
        app.init_resource::<ShaderProfilingData>()
            .init_resource::<FrameCostHistory>()
            .init_resource::<MemoryStats>()
            .add_systems(Update, (collect_profiling_data, sample_memory));
    }
}

fn collect_profiling_data(
    mut data: ResMut<ShaderProfilingData>,
    mut costs: ResMut<FrameCostHistory>,
    diagnostics: Res<DiagnosticsStore>,
    time: Res<Time<Real>>,
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

    // This frame's per-contributor costs: GPU passes from the render-diagnostics timestamp
    // queries + CPU sections drained from the instrument sink.
    let mut samples: BTreeMap<String, f32> = BTreeMap::new();
    for d in diagnostics.iter() {
        if let Some(name) = d
            .path()
            .as_str()
            .strip_prefix("render/")
            .and_then(|r| r.strip_suffix("/elapsed_gpu"))
            && name.starts_with("sdf_")
            && let Some(v) = d.value()
        {
            let short = name.trim_start_matches("sdf_").trim_end_matches("_pass");
            samples.insert(format!("gpu: {short}"), v as f32);
        }
    }
    for (tag, ms) in crate::instrument::drain() {
        samples.insert(format!("cpu: {tag}"), ms);
    }
    costs.accumulate(&samples, time.delta_secs());
}

/// The dedicated Performance dock panel: a per-frame compute breakdown (stacked GPU + CPU
/// contributors) followed by the capture toggle and SDF atlas stats. The whole panel scrolls.
pub fn performance_panel(world: &mut World, ui: &mut egui::Ui) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            frame_cost_ui(world, ui);

            // Chrome-trace capture toggle (off by default). Records Bevy system /
            // render-graph spans to a JSON file only while enabled.
            ui.separator();
            ui.heading("Profiling capture");
            crate::editor::chrome_trace::capture_ui(world, ui);

            // System (process / host) RAM.
            ui.separator();
            ui.heading("System memory");
            system_memory_ui(world, ui);
        });
}

/// One slice of a memory breakdown: a label, its byte size, and its bar/swatch color.
struct MemSeg {
    label: String,
    bytes: u64,
    color: egui::Color32,
}

/// Human-readable byte size (binary units).
fn fmt_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

/// Draw a horizontal stacked bar filling the width, one segment per `(fraction, color)` (the
/// fractions should sum to ≤ 1; any remainder is left as background).
fn draw_hbar(ui: &mut egui::Ui, segments: &[(f32, egui::Color32)]) {
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 16.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, egui::Color32::from_gray(28));
    let mut x = rect.left();
    for &(frac, color) in segments {
        let w = rect.width() * frac.clamp(0.0, 1.0);
        if w <= 0.0 {
            continue;
        }
        let seg = egui::Rect::from_min_size(egui::pos2(x, rect.top()), egui::vec2(w, rect.height()));
        painter.rect_filled(seg, 0.0, color);
        x += w;
    }
}

/// Stacked memory bar + a swatch/label/size/share table for `segments`, biggest-first.
fn mem_breakdown_ui(ui: &mut egui::Ui, id: &str, mut segments: Vec<MemSeg>) {
    segments.sort_by_key(|s| std::cmp::Reverse(s.bytes));
    let total: u64 = segments.iter().map(|s| s.bytes).sum();
    let denom = total.max(1) as f32;

    draw_hbar(
        ui,
        &segments
            .iter()
            .map(|s| (s.bytes as f32 / denom, s.color))
            .collect::<Vec<_>>(),
    );
    ui.add_space(2.0);
    ui.strong(format!("Total: {}", fmt_bytes(total)));

    egui::Grid::new(("mem_breakdown", id))
        .num_columns(3)
        .striped(true)
        .show(ui, |ui| {
            for s in &segments {
                let share = 100.0 * s.bytes as f32 / denom;
                let (r, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter().rect_filled(r, 1.0, s.color);
                ui.label(&s.label);
                ui.label(fmt_bytes(s.bytes));
                ui.weak(format!("{share:.0}%"));
                ui.end_row();
            }
        });
}

/// System memory: a byte-level breakdown of host RAM into this process, other processes, and
/// free — from [`MemoryStats`] (sampled via `sysinfo`).
fn system_memory_ui(world: &mut World, ui: &mut egui::Ui) {
    let m = world.resource::<MemoryStats>();
    if m.system_total == 0 {
        ui.weak("sampling…");
        return;
    }
    let other = m.system_used.saturating_sub(m.process_rss);
    let free = m.system_total.saturating_sub(m.system_used);
    let segments = vec![
        MemSeg { label: "This process".into(), bytes: m.process_rss, color: band_color(0) },
        MemSeg { label: "Other processes".into(), bytes: other, color: band_color(5) },
        MemSeg { label: "Free".into(), bytes: free, color: egui::Color32::from_gray(70) },
    ];
    mem_breakdown_ui(ui, "sys", segments);
}

/// A ranked contributor: its display label, color in the stack, and per-frame ms history.
struct Band {
    label: String,
    color: egui::Color32,
    series: Vec<f32>,
}

/// FPS / frame-time readout + the stacked per-frame compute graph + a breakdown table.
fn frame_cost_ui(world: &mut World, ui: &mut egui::Ui) {
    let (fps, ft) = {
        let d = world.resource::<ShaderProfilingData>();
        (d.fps_smoothed, d.frame_time_ms)
    };

    ui.horizontal(|ui| {
        ui.colored_label(egui::Color32::from_rgb(120, 220, 120), format!("{fps:.1} FPS"));
        ui.separator();
        ui.colored_label(egui::Color32::from_rgb(240, 180, 90), format!("{ft:.2} ms"));
    });

    // Two independent stacks: GPU render passes and tagged CPU sections. They measure
    // different things (GPU-busy time vs CPU wall time, the latter running async to the
    // former), so stacking them together would be misleading — each gets its own graph,
    // ranking, and ceiling.
    let (gpu, cpu, phase) = {
        let hist = world.resource::<FrameCostHistory>();
        (ranked_bands(hist, "gpu: "), ranked_bands(hist, "cpu: "), hist.phase)
    };
    if gpu.is_empty() && cpu.is_empty() {
        ui.add_space(4.0);
        ui.weak(
            "No per-frame costs yet — GPU timings need an editor build (RenderDiagnosticsPlugin \
             + TIMESTAMP_QUERY) and a few frames to warm up; CPU tags appear when their systems run.",
        );
        return;
    }

    ui.add_space(4.0);
    ui.strong("GPU passes");
    stack_section(ui, "gpu", &gpu, phase, "needs TIMESTAMP_QUERY — warming up");
    ui.add_space(8.0);
    ui.strong("CPU sections");
    stack_section(ui, "cpu", &cpu, phase, "idle — tags appear when their systems run");
}

/// Render one labelled stack: the stacked-area graph, a current total, and a color-keyed
/// breakdown table. `id` disambiguates egui widget ids between the two sections; `empty_hint`
/// shows when this stack has no contributors yet.
fn stack_section(ui: &mut egui::Ui, id: &str, bands: &[Band], phase: f32, empty_hint: &str) {
    if bands.is_empty() {
        ui.weak(empty_hint);
        return;
    }

    // Stack ceiling: the worst-case total over the committed window (everything draw_stack
    // shows), rounded to a steady ms so a brief spike sets the headroom rather than making the
    // curve jump every frame.
    let n = bands[0].series.len();
    let peak_total = (0..n)
        .map(|i| bands.iter().map(|b| b.series[i]).sum::<f32>())
        .fold(0.0_f32, f32::max);
    let ceiling = ft_axis_ceiling(peak_total);

    let (rect, _resp) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 130.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, egui::Color32::from_gray(20));
    draw_stack(&painter, rect, bands, ceiling, phase);

    let cur_total: f32 = bands.iter().map(|b| *b.series.last().unwrap_or(&0.0)).sum();
    ui.horizontal(|ui| {
        ui.strong(format!("Σ {cur_total:.2} ms"));
        ui.weak(format!("(0–{ceiling:.1} ms)"));
    });

    // Breakdown table: each band's current ms + share, biggest first (matches stack order).
    egui::Grid::new(("frame_cost_breakdown", id))
        .num_columns(3)
        .striped(true)
        .show(ui, |ui| {
            for b in bands {
                let cur = *b.series.last().unwrap_or(&0.0);
                let share = if cur_total > 1e-4 { 100.0 * cur / cur_total } else { 0.0 };
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter().rect_filled(rect, 1.0, b.color);
                ui.label(&b.label);
                ui.label(format!("{cur:.3} ms"));
                ui.weak(format!("{share:.0}%"));
                ui.end_row();
            }
        });
}

/// Build the stacked bands for the contributors whose label starts with `prefix` (e.g.
/// `"gpu: "` / `"cpu: "`): rank each by its mean over the window, keep the top [`TOP_N`] as
/// their own bands, and fold the rest into one "other" band. The prefix is stripped from the
/// displayed label (the section heading already says GPU/CPU). Returned largest-mean first
/// (drawn bottom-up so the dominant cost sits at the base).
fn ranked_bands(hist: &FrameCostHistory, prefix: &str) -> Vec<Band> {
    let n = hist.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = |s: &VecDeque<f32>| s.iter().sum::<f32>() / s.len().max(1) as f32;
    let mut ranked: Vec<(&str, &VecDeque<f32>)> = hist
        .series
        .iter()
        .filter_map(|(k, q)| k.strip_prefix(prefix).map(|short| (short, q)))
        .collect();
    if ranked.is_empty() {
        return Vec::new();
    }
    ranked.sort_by(|a, b| mean(b.1).total_cmp(&mean(a.1)));

    let mut bands: Vec<Band> = Vec::new();
    for (i, (label, q)) in ranked.iter().take(TOP_N).enumerate() {
        bands.push(Band {
            label: (*label).to_string(),
            color: band_color(i),
            series: q.iter().copied().collect(),
        });
    }
    // Fold everything past TOP_N into a single "other" band (element-wise sum).
    if ranked.len() > TOP_N {
        let mut other = vec![0.0_f32; n];
        for (_, q) in &ranked[TOP_N..] {
            for (o, v) in other.iter_mut().zip(q.iter()) {
                *o += *v;
            }
        }
        bands.push(Band {
            label: format!("other (+{})", ranked.len() - TOP_N),
            color: egui::Color32::from_gray(110),
            series: other,
        });
    }
    bands
}

/// Distinct band colors, cycled by rank. Twelve hand-picked hues stay legible against the
/// dark graph background and the striped breakdown table.
fn band_color(i: usize) -> egui::Color32 {
    const PALETTE: [egui::Color32; 12] = [
        egui::Color32::from_rgb(90, 170, 255),  // blue
        egui::Color32::from_rgb(255, 140, 90),  // orange
        egui::Color32::from_rgb(120, 220, 120), // green
        egui::Color32::from_rgb(220, 110, 200), // magenta
        egui::Color32::from_rgb(240, 210, 90),  // yellow
        egui::Color32::from_rgb(110, 215, 215), // cyan
        egui::Color32::from_rgb(200, 130, 250), // violet
        egui::Color32::from_rgb(250, 120, 120), // red
        egui::Color32::from_rgb(150, 200, 100), // lime
        egui::Color32::from_rgb(180, 160, 240), // periwinkle
        egui::Color32::from_rgb(240, 160, 200), // pink
        egui::Color32::from_rgb(160, 190, 210), // slate
    ];
    PALETTE[i % PALETTE.len()]
}

/// Draw `bands` as a stacked area chart filling `rect`, scaled so `ceiling` ms reaches the
/// top. Bands are laid down bottom-up in order; each is filled as a SINGLE egui mesh (a
/// triangle strip between its lower and upper edges) rather than per-column quads — a single
/// mesh has no internal anti-aliased seams, so the fill is clean instead of a grid of lines.
///
/// Columns are committed time-buckets (newest at the right). `phase` ∈ [0,1) is how far the
/// in-progress bucket has filled; the whole curve is slid left by `phase` of one column so it
/// glides between commits instead of jumping a full column each time one lands.
fn draw_stack(painter: &egui::Painter, rect: egui::Rect, bands: &[Band], ceiling: f32, phase: f32) {
    use egui::epaint::{Mesh, Vertex, WHITE_UV};

    let cols = bands.first().map_or(0, |b| b.series.len());
    if cols < 2 {
        return;
    }
    let ceiling = ceiling.max(1e-3);
    // Fixed column spacing (GRAPH_COLS slots across the rect) so the slide rate is constant
    // regardless of how many columns have filled yet. Newest column sits `phase` of a slot in
    // from the right edge; older columns step left from there, oldest scrolling off the left.
    let slot = rect.width() / (GRAPH_COLS - 1) as f32;
    let x_of = |c: usize| {
        let from_right = (cols - 1 - c) as f32 + phase;
        rect.right() - slot * from_right
    };
    let ypos = |v: f32| rect.bottom() - rect.height() * (v / ceiling).clamp(0.0, 1.0);

    // Running lower edge of the current band, per column — accumulates as bands stack up.
    let mut lower = vec![0.0_f32; cols];
    for b in bands {
        let mut mesh = Mesh::default();
        // Two vertices per column: the band's lower then upper edge. Triangles tie each
        // column pair into a quad (lower_c, upper_c, upper_c+1, lower_c+1).
        for (c, &l) in lower.iter().enumerate() {
            let u = l + b.series[c];
            let x = x_of(c);
            mesh.vertices.push(Vertex { pos: egui::pos2(x, ypos(l)), uv: WHITE_UV, color: b.color });
            mesh.vertices.push(Vertex { pos: egui::pos2(x, ypos(u)), uv: WHITE_UV, color: b.color });
        }
        for c in 0..cols - 1 {
            let i = c as u32 * 2; // lower_c=i, upper_c=i+1, lower_c+1=i+2, upper_c+1=i+3
            mesh.indices.extend_from_slice(&[i, i + 1, i + 3, i, i + 3, i + 2]);
        }
        painter.add(egui::Shape::mesh(mesh));
        for (l, &v) in lower.iter_mut().zip(b.series.iter()) {
            *l += v;
        }
    }
}

/// Round a peak frame time up to a "nice" ms ceiling (1/2/5 × 10ⁿ), giving the graph a steady
/// headroom that a brief spike won't keep nudging.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ft_ceiling_rounds_to_nice_ms() {
        assert_eq!(ft_axis_ceiling(8.0), 10.0);
        assert_eq!(ft_axis_ceiling(12.0), 20.0);
        assert_eq!(ft_axis_ceiling(33.0), 50.0);
        assert_eq!(ft_axis_ceiling(0.5), 1.0); // clamped floor
    }

    /// Accumulate `samples` and advance a full interval so exactly one column commits — the
    /// bucket holds one frame, so the committed value equals the sample.
    fn commit_one(h: &mut FrameCostHistory, samples: &BTreeMap<String, f32>) {
        h.accumulate(samples, COMMIT_INTERVAL_SECS);
    }

    #[test]
    fn history_stays_aligned_and_padded() {
        let mut h = FrameCostHistory::default();
        // Bucket 0: only A present.
        commit_one(&mut h, &BTreeMap::from([("gpu: a".to_string(), 1.0)]));
        // Bucket 1: A and B — B must back-fill so both deques align.
        commit_one(
            &mut h,
            &BTreeMap::from([("gpu: a".to_string(), 2.0), ("cpu: b".to_string(), 3.0)]),
        );
        assert_eq!(h.len(), 2);
        let a: Vec<f32> = h.series["gpu: a"].iter().copied().collect();
        let b: Vec<f32> = h.series["cpu: b"].iter().copied().collect();
        // One frame per bucket → committed value == sample; B padded with a leading zero for
        // the bucket before it appeared.
        assert_eq!(a, vec![1.0, 2.0]);
        assert_eq!(b, vec![0.0, 3.0]);
    }

    #[test]
    fn bucket_averages_frames_and_is_framerate_independent() {
        let mut h = FrameCostHistory::default();
        // Three sub-interval frames (each 0.4 of an interval) of values 3,6,9: the first two
        // (0.8 interval) don't commit; the third crosses one interval → commit their mean (6),
        // regardless of how many frames composed the bucket.
        let dt = COMMIT_INTERVAL_SECS * 0.4;
        h.accumulate(&BTreeMap::from([("cpu: x".to_string(), 3.0)]), dt);
        h.accumulate(&BTreeMap::from([("cpu: x".to_string(), 6.0)]), dt);
        assert_eq!(h.len(), 0); // 0.8 interval elapsed — nothing committed yet
        h.accumulate(&BTreeMap::from([("cpu: x".to_string(), 9.0)]), dt);
        assert_eq!(h.len(), 1);
        assert!((h.series["cpu: x"][0] - 6.0).abs() < 1e-3);
    }

    #[test]
    fn stopped_series_zeroes_then_prunes() {
        let mut h = FrameCostHistory::default();
        commit_one(&mut h, &BTreeMap::from([("cpu: blip".to_string(), 5.0)]));
        assert!(h.series.contains_key("cpu: blip"));
        // Absent in the next bucket: its column is 0, but the earlier 5 keeps it in the window.
        commit_one(&mut h, &BTreeMap::from([("gpu: a".to_string(), 1.0)]));
        assert!(h.series.contains_key("cpu: blip"));
        assert_eq!(*h.series["cpu: blip"].back().unwrap(), 0.0);
    }

    #[test]
    fn ranked_bands_filter_by_prefix_and_fold_top_n() {
        let mut h = FrameCostHistory::default();
        let mut frame = BTreeMap::new();
        // 12 CPU contributors (descending cost) plus one GPU pass.
        for k in 0..12 {
            frame.insert(format!("cpu: s{k:02}"), (12 - k) as f32);
        }
        frame.insert("gpu: gbuffer".to_string(), 4.0);
        commit_one(&mut h, &frame);

        let cpu = ranked_bands(&h, "cpu: ");
        // TOP_N individual CPU bands + one "other"; GPU pass excluded.
        assert_eq!(cpu.len(), TOP_N + 1);
        assert!(cpu.last().unwrap().label.starts_with("other"));
        // Prefix stripped, largest cost first.
        assert_eq!(cpu[0].label, "s00");

        let gpu = ranked_bands(&h, "gpu: ");
        assert_eq!(gpu.len(), 1);
        assert_eq!(gpu[0].label, "gbuffer");
    }
}
