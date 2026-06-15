//! Lightweight CPU span instrumentation, compiled into every build but **off by default**.
//!
//! Core systems (in non-editor modules like `sdf_render`) can tag a notable section with
//! [`span`] without depending on the editor or threading a profiling resource through their
//! signatures. The editor's profiling panel turns collection on via [`set_enabled`] and
//! [`drain`]s the accumulated per-tag times once per frame to feed its stacked frame-cost
//! graph.
//!
//! Cost when disabled (the shipping default): a single relaxed atomic load per [`span`] call
//! — no clock read, no allocation, no lock. When enabled, each span does one [`std::time::Instant`]
//! read at creation and a short mutex-guarded map update on drop (a handful per frame).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Whether spans actually measure. Flipped on by the editor profiler; stays false otherwise.
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Per-tag accumulated CPU time (milliseconds) for the current frame. `&'static str` keys so
/// the set of tags is fixed and cheap to hash. Lazily created on first use.
static SINK: OnceLock<Mutex<HashMap<&'static str, f32>>> = OnceLock::new();

fn sink() -> &'static Mutex<HashMap<&'static str, f32>> {
    SINK.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Enable or disable span collection process-wide. The editor profiler calls this with `true`
/// on startup; when `false`, [`span`] is a no-op beyond an atomic load.
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
    if !on {
        // Drop any residue so a later re-enable starts clean.
        if let Some(m) = SINK.get() {
            m.lock().unwrap().clear();
        }
        if let Some(m) = GPU_SINK.get() {
            m.lock().unwrap().clear();
        }
    }
}

/// Whether collection is currently enabled.
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Add `dur` to `tag`'s accumulated time for this frame. No-op while disabled.
fn record(tag: &'static str, dur: Duration) {
    if !enabled() {
        return;
    }
    *sink().lock().unwrap().entry(tag).or_default() += dur.as_secs_f32() * 1000.0;
}

/// Take and clear all accumulated per-tag milliseconds for the frame. Returns empty while
/// disabled. Called once per frame by the profiler.
pub fn drain() -> HashMap<&'static str, f32> {
    if !enabled() {
        return HashMap::new();
    }
    std::mem::take(&mut *sink().lock().unwrap())
}

/// RAII timer: records the elapsed time to `tag` when dropped. Hold it for the scope you want
/// to measure (`let _span = instrument::span("foo");`). Carries no clock read while disabled.
#[must_use = "the span measures until it is dropped; binding it to `_` ends it immediately"]
pub struct Span {
    tag: &'static str,
    start: Option<Instant>,
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            record(self.tag, start.elapsed());
        }
    }
}

/// Begin timing a section tagged `tag` (a stable `&'static str`). The returned [`Span`] records
/// the elapsed wall time to `tag` when it drops. Cheap no-op while collection is disabled.
pub fn span(tag: &'static str) -> Span {
    Span {
        tag,
        start: enabled().then(Instant::now),
    }
}

/// Per-tag accumulated GPU time (milliseconds) for the current frame, fed from the render world's
/// timestamp-query read-back (1-frame latency). Kept SEPARATE from the CPU [`SINK`] because GPU-busy time
/// and CPU wall time are different axes (the profiler stacks them in their own graphs); the panel drains
/// this with a `gpu:` prefix. `String` keys (not `&'static str`) so render-world callers can pass owned
/// pass labels without leaking. Lazily created on first use.
static GPU_SINK: OnceLock<Mutex<HashMap<String, f32>>> = OnceLock::new();

fn gpu_sink() -> &'static Mutex<HashMap<String, f32>> {
    GPU_SINK.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Set `tag`'s GPU milliseconds for this frame (the render-world timestamp read-back computes the whole
/// per-pass delta, so this STORES rather than accumulates — repeated passes in a frame would overwrite,
/// which matches the single-dispatch-per-pass reality). No-op while collection is disabled. Mirrors the CPU
/// [`record`] path so the editor profiler turns both on with one [`set_enabled`].
pub fn record_gpu(tag: &str, ms: f32) {
    if !enabled() {
        return;
    }
    let mut sink = gpu_sink().lock().unwrap();
    sink.insert(tag.to_owned(), ms);
}

/// Take and clear all accumulated per-tag GPU milliseconds for the frame. Returns empty while disabled.
/// Called once per frame by the profiler (alongside [`drain`]).
pub fn drain_gpu() -> HashMap<String, f32> {
    if !enabled() {
        return HashMap::new();
    }
    std::mem::take(&mut *gpu_sink().lock().unwrap())
}

/// Cache of leaked tag strings so a dynamically-built tag (e.g. a panel title) can be used
/// with [`span`], which needs a `&'static str`. One small leak per unique string, ever.
static INTERNED: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();

/// Intern `s` into a process-lifetime `&'static str` so it can tag a [`span`]. Intended for a
/// small, bounded set of tags (panel titles, etc.) — each unique string leaks exactly once.
pub fn intern(s: &str) -> &'static str {
    let map = INTERNED.get_or_init(|| Mutex::new(HashMap::new()));
    let mut m = map.lock().unwrap();
    if let Some(&found) = m.get(s) {
        return found;
    }
    let leaked: &'static str = Box::leak(s.to_owned().into_boxed_str());
    m.insert(s.to_owned(), leaked);
    leaked
}

#[cfg(test)]
mod tests {
    use super::*;

    /// These tests drive PROCESS-GLOBAL state (the `enabled` flag + the single accumulation sink), so running
    /// them concurrently (the default `cargo test` thread pool) lets one's `set_enabled`/`drain` clobber the
    /// other's — the `section_a` entry the enabled test asserts can be drained or its flag flipped mid-flight by
    /// the disabled test. Serialize them on this guard so each runs in isolation (robust-by-construction, not a
    /// timing fudge). A poisoned lock (a prior panic) is recovered so one failure doesn't cascade.
    static TEST_GUARD: Mutex<()> = Mutex::new(());

    #[test]
    fn disabled_spans_record_nothing() {
        let _g = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        set_enabled(false);
        let _ = drain(); // clear any leftover from a prior test that shared the global sink
        {
            let _s = span("disabled_tag");
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(drain().is_empty());
    }

    #[test]
    fn enabled_spans_accumulate_then_drain_clears() {
        let _g = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        set_enabled(true);
        let _ = drain(); // start from a clean sink (the disabled test may have left it enabled-off)
        {
            let _s = span("section_a");
            std::thread::sleep(Duration::from_millis(2));
        }
        {
            let _s = span("section_a");
            std::thread::sleep(Duration::from_millis(1));
        }
        let first = drain();
        assert!(first.get("section_a").copied().unwrap_or(0.0) > 0.0);
        // Two spans of the same tag accumulate into one entry.
        assert_eq!(first.len(), 1);
        // Drain cleared the sink.
        assert!(drain().is_empty());
        set_enabled(false);
    }
}
