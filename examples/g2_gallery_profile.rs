//! **G2 step 1 — GALLERY residency CPU profile (find the freeze).** Drives the SHIPPED streamed `.vxo`
//! gallery corpus (`vxo_gallery_placements(GALLERY_SCENES)` → [`MergedSource::open_paths`]) through the
//! PRODUCTION residency at [`StreamingConfig::default()`] (clip_half 160, 0.05 m, max_resident 400k) and breaks
//! the COLD switch-to-Gallery + a few camera brick-crossings down per stage, so we know which stage to move to
//! GPU first (measure-don't-guess). Pure CPU; no GPU device — the BLAS/pack-to-GPU stage is noted separately.
//! Run:
//!   cargo run --release --no-default-features --features fast,physics --example g2_gallery_profile
//!
//! NOT a shipped tool — a one-off de-risk for the Phase-G GPU-residency pivot. Mirrors the EXACT call sequence
//! `raytrace::stream_voxel_rt_residency` runs on the Gallery switch (placements → `MergedSource` → fresh
//! `ResidencyManager` → `update` → `drain_work_from` → `pack_resident_set`), instrumented per stage.
//!
//! The stages reported (the prompt's (a)–(g)):
//!   (a) `desired_clipmap_surface` enumeration (MergedSource `surface_bricks_in` over the 8 LOD shells)
//!   (b) classify pass (per-candidate `BrickClass` 6-neighbour predicate; MergedSource is a SUPERSET source)
//!   (c) the `select_nth` cap (folded into `update`; measured by the residual update time)
//!   (d) the cold snapshot / pack (`pack_resident_set` — O(resident) buffer build)
//!   (e) the drain: `.vxo` region decode (LRU miss → decompress + parse) — measured via cache-stats deltas
//!   (f) the coarse-LOD demand-downsample (recursive `VxoSource::brick(coord, lod>0)`) — the suspected hog
//!   (g) BLAS build — needs a GPU device; noted as out-of-scope here (see the report)

use std::cell::Cell;
use std::time::Instant;

use bevy::math::IVec3;

use adventure::voxel::brickmap::{MAX_LOD, brick_span};
use adventure::voxel::edits::VoxelEdits;
use adventure::voxel::gallery::{GALLERY_SCENES, vxo_gallery_placements};
use adventure::voxel::gpu::pack_resident_set;
use adventure::voxel::palette::BlockRegistry;
use adventure::voxel::source::{BrickClass, BrickSource};
use adventure::voxel::streaming::{
    ResidencyManager, StreamingConfig, camera_brick_coord, desired_clipmap_surface,
};
use adventure::voxel::vxo::MergedSource;
use adventure::voxel::brickmap::Brick;

/// An instrumenting [`BrickSource`] decorator — counts + times every trait call so the profile can split
/// enumeration (a) / classify (b) / drain-source ((e)+(f)) and the LOD0-vs-coarse `brick` mix. Thread-local-free:
/// the residency drives it from the compute pool IN PARALLEL, so the counters are `Cell`s behind a `Sync` shim —
/// WRONG for exact parallel counts. To keep the counts EXACT we force the residency single-threaded by setting
/// the compute pool to 1 thread (see `main`), so the decorator is only ever touched serially. The counts are
/// therefore the true per-call totals; the TIMES are wall (single-threaded), the honest serial cost.
struct Counting<'a> {
    inner: &'a MergedSource,
    brick_lod0: Cell<u64>,
    brick_coarse: Cell<u64>,
    classify_calls: Cell<u64>,
    surface_calls: Cell<u64>,
}

// SAFETY: we run the residency single-threaded (compute pool forced to 1 thread in `main`), so the `Cell`s are
// only ever accessed from one thread. `BrickSource: Sync` is required by the trait; this shim upholds it under
// the single-thread harness invariant only (it is a throwaway diagnostic, never shipped).
unsafe impl Sync for Counting<'_> {}

impl<'a> Counting<'a> {
    fn new(inner: &'a MergedSource) -> Self {
        Self {
            inner,
            brick_lod0: Cell::new(0),
            brick_coarse: Cell::new(0),
            classify_calls: Cell::new(0),
            surface_calls: Cell::new(0),
        }
    }
    fn reset(&self) {
        self.brick_lod0.set(0);
        self.brick_coarse.set(0);
        self.classify_calls.set(0);
        self.surface_calls.set(0);
    }
}

impl BrickSource for Counting<'_> {
    fn brick(&self, coord: IVec3, lod: u32, registry: &BlockRegistry) -> Brick {
        if lod == 0 {
            self.brick_lod0.set(self.brick_lod0.get() + 1);
        } else {
            self.brick_coarse.set(self.brick_coarse.get() + 1);
        }
        self.inner.brick(coord, lod, registry)
    }
    fn classify(&self, coord: IVec3, lod: u32) -> BrickClass {
        self.classify_calls.set(self.classify_calls.get() + 1);
        self.inner.classify(coord, lod)
    }
    fn surface_bricks_in(&self, lo: IVec3, hi: IVec3, lod: u32, out: &mut Vec<IVec3>) {
        self.surface_calls.set(self.surface_calls.get() + 1);
        self.inner.surface_bricks_in(lo, hi, lod, out);
    }
    fn surface_bricks_are_exact(&self) -> bool {
        self.inner.surface_bricks_are_exact()
    }
}

fn main() {
    // Force the Bevy compute pool to a SINGLE thread so the instrumented `Counting` counters are exact (the
    // residency parallelizes the classify + drain over the pool). The reported TIMES are then the honest serial
    // wall — the freeze is a blocking call on the MAIN schedule, so the serial cost is exactly what stalls.
    bevy::tasks::ComputeTaskPool::get_or_init(|| {
        bevy::tasks::TaskPoolBuilder::new().num_threads(1).build()
    });

    let placements = vxo_gallery_placements(GALLERY_SCENES);
    if placements.is_empty() {
        eprintln!("no gallery `.vxo` baked in this checkout — nothing to profile (bake the corpus first)");
        return;
    }
    println!("\n========== G2 GALLERY RESIDENCY PROFILE (streamed `.vxo`, 0.05 m) ==========");
    let t = Instant::now();
    let (merged, registry): (MergedSource, BlockRegistry) = MergedSource::open_paths(&placements);
    let open_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("placements            : {} `.vxo` asset(s)", placements.len());
    for (p, off) in &placements {
        println!("  {}  @ +X brick {}", p.display(), off.x);
    }
    println!("MergedSource::open    : {open_ms:.1} ms (mmap + eager HEAD/MATL/BIDX parse; NO region decode)");
    println!("merged registry       : {} block ids", registry.len());

    let cfg = StreamingConfig::default();
    println!(
        "config                : clip_half {} bricks, max_resident {}, {}/frame, {} LOD shells",
        cfg.clip_half_bricks,
        cfg.max_resident_bricks,
        cfg.max_bricks_per_frame,
        MAX_LOD + 1
    );
    println!(
        "LOD0 reach            : {:.1} m ; brick_span(0) = {:.3} m",
        cfg.clip_half_bricks as f32 * brick_span(0),
        brick_span(0)
    );

    // COLD camera: near the merged scenes. The gallery anchors the first asset (Sponza) at brick column 0,
    // floor-anchored at the origin; the row marches along +X. Put the eye a couple of metres up inside the −X
    // end so a populated shell streams in — the SAME cold framing the ignored gallery-residency gate uses.
    let cam = [1.0f32, 2.0, 1.0];
    println!("cold camera (world)   : {cam:?}  (near Sponza's −X end, eye ~2 m up)");
    let cam_brick = camera_brick_coord(cam);
    println!("cold camera brick     : {cam_brick:?}");

    let counting = Counting::new(&merged);

    // ===================== (a) desired_clipmap_surface enumeration =====================
    // Reset counters; time the shell-first surface enumeration in isolation (it calls `surface_bricks_in`).
    counting.reset();
    let t = Instant::now();
    let desired = desired_clipmap_surface(cam, &cfg, &counting);
    let enum_ms = t.elapsed().as_secs_f64() * 1e3;
    let surface_calls = counting.surface_calls.get();
    let mut cand_per_lod = vec![0usize; (MAX_LOD + 1) as usize];
    for k in desired.keys() {
        cand_per_lod[k.lod as usize] += 1;
    }
    println!("\n-- (a) desired_clipmap_surface enumeration --");
    println!("ENUM WALL             : {enum_ms:.1} ms");
    println!("candidate keys        : {}", desired.len());
    println!("surface_bricks_in()   : {surface_calls} calls (shell sub-boxes across the 8 LOD shells)");
    let (r0, b0, c0) = merged.cache_stats();
    println!("region decodes so far : {r0} regions, {:.1} MB, {c0} coarse memo (enum must NOT decode)", b0 as f64 / 1e6);
    for (l, n) in cand_per_lod.iter().enumerate() {
        if *n > 0 {
            println!("  LOD{l}                : {n} candidates (span {:.2} m)", brick_span(l as u32));
        }
    }

    let edits = VoxelEdits::new();
    // NOTE: the (b)+(f) per-LOD coarse-classify SAMPLE runs LAST (after the (e) drain + (d) pack) — sampling the
    // coarse downsample populates a MULTI-MILLION-entry coarse memo that puts the process under RAM pressure, so
    // running it before the pack would contaminate the pack's wall. It is deferred to the end (nothing measured
    // after it). See the `coarse_classify_sample` block below the pack.

    // ===================== (e) LOD0-only drain — region decode + pack =====================
    // The LOD0 candidates alone are a tractable, REAL resident set (the fine shell — what the camera is inside).
    // Drain ONLY the LOD0 candidates so we measure (e) the `.vxo` region decode + (d) the pack WITHOUT the coarse
    // recursion that OOMs. This is the part of the cold load that WOULD complete; the coarse shells are the part
    // that doesn't (the freeze). Build a manager whose desired set is the LOD0 candidates only.
    let mut mgr = ResidencyManager::new();
    // Enqueue the LOD0 candidates directly via `requeue_keys` (bypasses classify — we already have them, and
    // they're the fine shell the camera occupies). This mirrors what `update` WOULD enqueue at LOD0.
    let lod0_keys: Vec<adventure::voxel::streaming::BrickKey> = desired
        .keys()
        .filter(|k| k.lod == 0)
        .copied()
        .collect();
    let lod0_cand = lod0_keys.len();
    mgr.requeue_keys(lod0_keys);
    counting.reset();
    let (r_pre_drain, b_pre_drain, _c_pre_drain) = merged.cache_stats();
    let t = Instant::now();
    let mut frames = 0u32;
    let mut voxelized = 0usize;
    let mut drain_max_ms = 0f64;
    let mut drain_max_frame = 0u32;
    while mgr.pending() > 0 {
        let td = Instant::now();
        voxelized += mgr.drain_work_from(&cfg, &counting, &registry, &edits);
        let fms = td.elapsed().as_secs_f64() * 1e3;
        if fms > drain_max_ms {
            drain_max_ms = fms;
            drain_max_frame = frames;
        }
        frames += 1;
        if frames > 2_000_000 {
            break;
        }
    }
    let drain_ms = t.elapsed().as_secs_f64() * 1e3;
    let (r_post_drain, b_post_drain, _c_post_drain) = merged.cache_stats();
    let resident = mgr.resident_count();
    println!("\n-- (e) LOD0-only cold drain ({}/frame): region decode (the tractable part of the cold load) --", cfg.max_bricks_per_frame);
    println!("LOD0 candidates       : {lod0_cand}");
    println!("DRAIN WALL            : {drain_ms:.0} ms ({:.2} s)", drain_ms / 1e3);
    println!("frames to settle      : {frames}");
    println!("LONGEST DRAIN FRAME   : {drain_max_ms:.1} ms  (frame {drain_max_frame})  ⇐ the per-frame hitch unit");
    println!("bricks sourced        : {voxelized}  (non-empty resident: {resident})");
    println!("brick(lod0) calls     : {}  (e: each = region binary-search + LRU; miss ⇒ zstd decode + parse)", counting.brick_lod0.get());
    println!(
        "region decodes (drain): +{} regions ({:.1} → {:.1} MB) — the LOD0 footprint of Sponza's −X end",
        r_post_drain.saturating_sub(r_pre_drain),
        b_pre_drain as f64 / 1e6,
        b_post_drain as f64 / 1e6
    );
    let lod_counts = mgr.resident_lod_counts();
    let lod0_resident = lod_counts.first().copied().unwrap_or(0);
    println!("\n-- RESIDENT SET (LOD0 fine shell only — coarse shells would add ~1.85 M more candidates) --");
    println!("LOD0 RESIDENT         : {lod0_resident} bricks (of {lod0_cand} candidates; rest were air/outside)");
    let (r_final, b_final, c_final) = merged.cache_stats();
    println!("decoded-region LRU    : {r_final} regions, {:.1} MB resident; coarse memo {c_final} bricks", b_final as f64 / 1e6);

    // ===================== (d) the cold snapshot / pack =====================
    // The O(resident) buffer build on the fresh epoch (the production path uses the incremental packer's
    // `snapshot_buffers`, but `pack_resident_set` is the same O(resident) cost SSOT the d1c rig reports — and is
    // what the `None`-packer fallback ships; the incremental snapshot is bit-identical, asserted in tests).
    let entries = mgr.resident_entries();
    let t = Instant::now();
    let patch = pack_resident_set(&entries, &registry);
    let pack_ms = t.elapsed().as_secs_f64() * 1e3;
    let rep = patch.storage_report();
    println!("\n-- (d) pack_resident_set (cold snapshot / O(resident) buffer build) --");
    println!("PACK WALL             : {pack_ms:.1} ms for {} bricks", patch.brick_count());
    println!(
        "RESIDENT VRAM         : {:.1} MB (before-collapse {:.1} MB, {:.2}× reduction)",
        rep.total_vram_after() as f64 / 1e6,
        rep.total_vram_before() as f64 / 1e6,
        rep.vram_reduction()
    );

    // ===================== (b)+(f) per-LOD classify cost — BOUNDED SAMPLE (deferred; OOMs if run whole) =====
    // The PRODUCTION cold `update` OOMs (a 26 GB allocation): the `.vxo` `surface_bricks_in` falls back to the
    // FULL BOX at coarse LODs (no coarse region directory), so each coarse shell enumerates ~264 k full-box
    // candidates, and `update` then `classify`s every one — and a coarse `classify` calls `coarse_brick`, which
    // RECURSIVELY demand-downsamples to LOD0. A single LOD7 brick spans 2^7 = 128 LOD0 bricks PER AXIS = 128³ ≈
    // 2.1 M LOD0 bricks, all synthesized/decoded. So instead of the OOMing full `update`, MEASURE the per-candidate
    // `classify` cost on a BOUNDED sample per LOD; per-brick cost × the candidate count = the cold-update cost the
    // freeze pays. Run LAST — the sample bloats the coarse memo to millions of entries (RAM pressure).
    println!("\n-- (b)+(f) per-LOD classify cost — BOUNDED SAMPLE (the full update OOMs at 26 GB) --");
    println!("(the `.vxo` coarse `surface_bricks_in` = FULL BOX, and a coarse `classify` recursively downsamples to LOD0)");
    const SAMPLE: usize = 32;
    let mut by_lod: Vec<Vec<IVec3>> = vec![Vec::new(); (MAX_LOD + 1) as usize];
    for k in desired.keys() {
        by_lod[k.lod as usize].push(k.coord);
    }
    let mut projected_classify_ms = 0f64;
    for (lod, coords) in by_lod.iter().enumerate() {
        if coords.is_empty() {
            continue;
        }
        let n = coords.len();
        let stride = (n / SAMPLE).max(1);
        let sample: Vec<IVec3> = coords.iter().step_by(stride).take(SAMPLE).copied().collect();
        let (rs0, _, cs0) = merged.cache_stats();
        let t = Instant::now();
        for &c in &sample {
            std::hint::black_box(merged.classify(c, lod as u32));
        }
        let sample_ms = t.elapsed().as_secs_f64() * 1e3;
        let (rs1, bs1, cs1) = merged.cache_stats();
        let per_brick_us = sample_ms * 1e3 / sample.len() as f64;
        let proj_ms = per_brick_us * n as f64 / 1e3;
        projected_classify_ms += proj_ms;
        println!(
            "  LOD{lod} (span {:5.2} m): {n:>7} cands; {} sampled in {sample_ms:8.2} ms = {per_brick_us:9.1} µs/brick \
             ⇒ PROJECTED {proj_ms:10.0} ms; sample +{} regions, +{} coarse memo (total {:.1} MB)",
            brick_span(lod as u32),
            sample.len(),
            rs1 - rs0,
            cs1 - cs0,
            bs1 as f64 / 1e6,
        );
    }
    println!("PROJECTED full-update classify : ~{projected_classify_ms:.0} ms ({:.0} s) over {} candidates", projected_classify_ms / 1e3, desired.len());
    println!("  ⇒ this is the cold `update`'s classify cost — the OOM is the SAME recursion blowing RAM, not just time");

    // ===================== (g) BLAS build =====================
    println!("\n-- (g) BLAS build --");
    println!("BLAS build            : NEEDS A GPU DEVICE (acceleration-structure build) — out of scope for this");
    println!("                        CPU rig. It is O(resident) prims; see the report for the separate note.");

    // ===================== steady-state per-crossing — ENUMERATION ONLY (the full update OOMs) =====================
    // The full per-crossing `update` OOMs for the same coarse-recursion reason, so we measure the part that is
    // safe + dominant: the O(shell) re-enumeration cost of one LOD0 brick step (`desired_clipmap_surface`). The
    // classify/coarse cost on a crossing is bounded by the SHELL slab that entered (a thin face of the LOD0 box
    // + the coarse shells that shifted), but it pays the SAME per-coarse-brick recursion measured above on
    // whatever coarse bricks entered — so a coarse shell crossing is ALSO heavy, not just the cold load.
    println!("\n-- STEADY-STATE per-crossing — enumeration only (full update OOMs, same coarse recursion) --");
    let span0 = brick_span(0);
    let mut cam_s = cam;
    for step in 1..=4u32 {
        cam_s[0] += span0; // cross exactly one LOD0 brick boundary
        counting.reset();
        let t = Instant::now();
        let d = desired_clipmap_surface(cam_s, &cfg, &counting);
        let enum_ms = t.elapsed().as_secs_f64() * 1e3;
        let lod0 = d.keys().filter(|k| k.lod == 0).count();
        println!("  step {step}: re-enumerate {enum_ms:6.1} ms ({} candidates, {lod0} at LOD0)", d.len());
    }

    // ===================== the verdict =====================
    println!("\n-- LONGEST SINGLE SYNCHRONOUS STAGE (the freeze) --");
    let stages: [(&str, f64); 4] = [
        ("(a) cold enumeration", enum_ms),
        ("(b)+(f) PROJECTED full-update classify (coarse downsample recursion)", projected_classify_ms),
        ("(d) cold pack (LOD0 set)", pack_ms),
        ("(e) LOD0-only drain (sum of frames)", drain_ms),
    ];
    let worst = stages
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("non-empty");
    println!("worst stage           : {} = {:.0} ms ({:.1} s)", worst.0, worst.1, worst.1 / 1e3);
    println!("note                  : the cold `update` does NOT even complete — it OOMs at 26 GB inside the coarse");
    println!("                        `classify`→`coarse_brick` recursion. The FREEZE is that single synchronous");
    println!("                        `update` call on the main schedule: it runs for the projected classify time");
    println!("                        AND blows RAM. The drain/pack only matter AFTER an update that can't finish.");
    println!("=============================================================================");
}
