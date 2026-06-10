//! Worldgen terrain mesh-bake **performance / benchmark rig** at the full LOD-8 clipmap.
//!
//! Mirrors `sdf_render::bake_scheduler::perf` in SHAPE (an `#[ignore]` test driving the REAL production
//! bake code with NO ECS App and NO GPU — the per-chunk `mesh_chunk` Transvoxel bake is the main-thread
//! cost we care about), but for the MESH bake of the streamed worldgen terrain. It MEASURES — it does NOT
//! change the bake algorithm or "optimize" anything. It answers three questions:
//!
//!   1. **Cold full-LOD-8 fill** — how big/expensive is a cold fill of the whole `lod_count`-tier clipmap
//!      around a fixed camera? Total main-thread CPU time, worst single chunk, mean/p50/p99, total
//!      verts+tris, and frames-to-fill at `MAX_NEW_TASKS_PER_FRAME` (= ceil(surface_chunks / 256)).
//!   2. **Narrow-band shell cull** — does `chunk_has_surface` actually drop the non-surface volume? Reported
//!      as the SHELL-CULL RATIO (surface / total resident) per LOD and overall: low = the bake really only
//!      touches the thin surface shell, not the interior/exterior volume.
//!   3. **Bounded streaming** — walk the camera in a fixed direction one LOD-0 chunk-width per step; diff the
//!      surface-chunk set each step to get new-entering / leaving counts; bake only the new ones. ASSERTS the
//!      max new-chunks-per-step is a small fraction of the total resident set (world-anchored per-chunk
//!      staging streams a leading edge, NOT a whole-band re-bake).
//!
//! The residency + cull are REPLICATED here (not reused) because `mesh_resident_chunks` is a Bevy system —
//! it cannot run without an App/Query. The replication calls the SAME private helpers production uses
//! (`effective_lod_count` / `lod0_half_chunks` / `lod_centre` / `shell_cube` clip / `chunks_in_aabb` /
//! `mesh_chunk_in_shell` / `terrain_chunk_covered` / `cull_into` + the `edit_resolvable_at` retain /
//! `chunk_has_surface`), so the topology it measures is the production topology. See [`surface_chunks`] —
//! its body is line-for-line the residency loop (lines ~869-921) + the bake-loop cull (line ~1124) of
//! `mesh_resident_chunks`; keep them in lockstep.
//!
//! Run it (it is `#[ignore]`, like every perf rig here):
//! ```sh
//! cargo test -p adventure --release sdf_render::mesh_bake::perf -- --ignored --nocapture
//! ```
//! It prints a `MESH-BAKE-PERF` report and writes `.soul/mesh_bake_perf.json` for before/after diffing.

use super::*;
use crate::sdf_render::edits::{CsgKind, ResolvedEdit, SdfOp, SdfPrimitive};
use crate::sdf_render::worldgen::graph::{Graph, GraphAsset};
use crate::sdf_render::worldgen::layers::erosion::ErosionParams;
use crate::sdf_render::worldgen::layers::height::{
    HEIGHT_CHUNK_CELLS, HEIGHT_FIELD_RES, HeightParams,
};
use crate::sdf_render::worldgen::manager::LayerManager;
use crate::sdf_render::worldgen::upload::{
    HeightClipmap, build_height_clipmap, set_cpu_height_clipmap,
};
use crate::sdf_render::worldgen::{
    WORLDGEN_SLICE_SEED, WORLDGEN_TERRAIN_HALF_XZ, height_clipmap_tiers, terrain_band,
};
// NB: `coarsest_lod_outer_reach`, `BLEND_REACH`, `MAX_NEW_TASKS_PER_FRAME`, `MeshBakeConfig`, and every
// private residency helper (`lod0_half_chunks`, `lod_centre`, `chunk_aabb`, `chunks_in_aabb`,
// `mesh_chunk_in_shell`, `terrain_chunk_covered`, `chunk_finer_faces`, `cull_into`, `edit_resolvable_at`,
// `chunk_has_surface`, `effective_lod_count`, `mesh_chunk`, `BrickKey`) come in via `use super::*`.
use bevy::math::DVec2;
use bevy::math::bounding::Aabb3d;
use std::collections::HashSet;
use std::sync::Arc;

/// Build the single world-anchored Terrain `ResolvedEdit` EXACTLY as `worldgen::spawn_terrain_volume` does:
/// the `SdfPrimitive::Terrain` at IDENTITY, material 0, plain Union, no smoothing. So the topology baked
/// here is the real worldgen terrain's.
fn terrain_edit() -> ResolvedEdit {
    let (min_y, max_y) = terrain_band(&HeightParams::default(), &ErosionParams::default());
    ResolvedEdit::new(
        SdfPrimitive::Terrain {
            half_xz: Vec2::splat(WORLDGEN_TERRAIN_HALF_XZ),
            min_height: min_y,
            max_height: max_y,
        },
        Transform::IDENTITY,
        SdfOp { kind: CsgKind::Union, smoothing: 0.0 },
        0,
    )
}

/// Load the REAL authored terrain graph from `assets/worldgen/world.graph.ron` so the rig measures the
/// production terrain (the one the editor authors + ships), NOT the legacy `HeightParams` fBm path. The
/// graph drives EVERY clipmap tier (the cross-tier-agreement invariant) via `LayerManager::set_graph`.
///
/// `None` (file missing / unparseable) ⇒ the caller falls back to the legacy default path with a noted
/// `eprintln!`, so the rig still runs (e.g. on a checkout without the authored asset). Mirrors
/// `roll_worldgen`'s `set_graph(world_graph.0.clone())` drive.
fn load_world_graph() -> Option<Arc<Graph>> {
    const PATH: &str = "assets/worldgen/world.graph.ron";
    match std::fs::read_to_string(PATH) {
        Ok(src) => match ron::de::from_str::<GraphAsset>(&src) {
            Ok(asset) => Some(Arc::new(asset.graph)),
            Err(e) => {
                eprintln!(
                    "MESH-BAKE-PERF: could not parse {PATH} ({e}); falling back to the LEGACY HeightParams terrain"
                );
                None
            }
        },
        Err(e) => {
            eprintln!(
                "MESH-BAKE-PERF: could not read {PATH} ({e}); falling back to the LEGACY HeightParams terrain"
            );
            None
        }
    }
}

/// Cold-generation timing of the settled window (Task 2 baseline / optimization target). Captured by
/// `build_and_publish_clipmap` when a `&mut GenStats` is passed (the cold focus-0 build only — the
/// streaming steps don't time gen). All fields are the cold fill of the WHOLE settled clipmap window.
#[derive(Default, Clone)]
struct GenStats {
    /// Wall-clock ms of the whole `update(focus)` settle loop (cold generation of every required chunk).
    settle_ms: f64,
    /// Total height chunks generated to settle the window (= final resident count, cold from empty).
    chunks: usize,
    /// Per-tier generated chunk counts (index = tier / `LayerId`).
    chunks_by_tier: Vec<usize>,
    /// `update(focus)` calls it took to settle (budget-bounded, so ≥ ceil(chunks / budget)).
    updates: u32,
}

impl GenStats {
    /// Graph evals per chunk = field nodes `(HEIGHT_FIELD_RES + 1)²` (tier-independent — every tier
    /// samples `HEIGHT_FIELD_RES` cells per axis). This is the true per-chunk graph-eval count.
    fn samples_per_chunk() -> usize {
        let n = (HEIGHT_FIELD_RES + 1) as usize;
        n * n
    }
    fn total_samples(&self) -> usize {
        self.chunks * Self::samples_per_chunk()
    }
    fn us_per_chunk(&self) -> f64 {
        if self.chunks == 0 { 0.0 } else { self.settle_ms * 1000.0 / self.chunks as f64 }
    }
    fn chunks_per_sec(&self) -> f64 {
        if self.settle_ms <= 0.0 { 0.0 } else { self.chunks as f64 / (self.settle_ms / 1000.0) }
    }
}

/// Build + publish the full tiered height clipmap settled around `focus` (camera XZ), mirroring
/// `worldgen::roll_worldgen`: derive the tier count from the mesh-bake coarsest-LOD reach, apply the
/// authored `graph` to every tier (`set_graph`), drive `LayerManager::update(focus)` with a high budget
/// until `is_settled`, build the clipmap, and publish it via `set_cpu_height_clipmap` (the Terrain eval +
/// coverage gate read this process-global — the strict sampler panics if it's missing). Returns the built
/// clipmap (also left published). When `gen_stats` is `Some`, the cold `update` settle loop is timed into
/// it (Task 2 — used only for the cold focus-0 build, not the streaming steps).
fn build_and_publish_clipmap(
    cfg: &SdfGridConfig,
    mesh_cfg: &MeshBakeConfig,
    focus: DVec2,
    graph: Option<&Arc<Graph>>,
    gen_stats: Option<&mut GenStats>,
) -> Arc<HeightClipmap> {
    let reach = coarsest_lod_outer_reach(cfg, mesh_cfg) as f64;
    let tiers = height_clipmap_tiers(reach);
    let mut manager =
        LayerManager::new_clipmap(WORLDGEN_SLICE_SEED, HeightParams::default(), ErosionParams::default(), tiers);
    // Drive generation from the REAL authored graph (every tier) — exactly as `roll_worldgen` does — so
    // the clipmap (and thus everything the rig measures) is the production terrain. Done BEFORE the settle
    // loop so the cold gen we time is the graph path. `None` ⇒ legacy `HeightParams` (already on the layers).
    if let Some(g) = graph {
        manager.set_graph(Some(g.clone()));
    }
    manager.budget = 1_000_000; // fill the whole window per update (we want a settled cold clipmap)

    // Settle the rolling window at `focus` (a handful of high-budget updates; guard against non-convergence).
    // Time the WHOLE settle loop when capturing gen stats: this is the cold generation of every required
    // height chunk (each = `HeightLayer::generate` evaluating the graph over the field grid).
    let t_gen = std::time::Instant::now();
    let mut guard = 0u32;
    while !manager.is_settled(focus) {
        manager.update(focus);
        guard += 1;
        assert!(guard < 1000, "clipmap did not settle around focus {focus:?}");
    }
    if let Some(stats) = gen_stats {
        stats.settle_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
        stats.updates = guard;
        let store = manager.height_store();
        stats.chunks = store.len();
        let mut by_tier = vec![0usize; manager.tier_count() as usize];
        for c in store.resident_coords() {
            let t = c.layer.0 as usize;
            if t < by_tier.len() {
                by_tier[t] += 1;
            }
        }
        stats.chunks_by_tier = by_tier;
    }

    let tier_cells: Vec<u32> = (0..manager.tier_count()).map(|t| HEIGHT_CHUNK_CELLS << t).collect();
    let clipmap = Arc::new(build_height_clipmap(manager.height_store(), &tier_cells));
    // Publish the tier-0 hi-fi DETAIL-NORMAL sampler too (mirrors `roll_worldgen`), so the rig's bake fires
    // the per-chunk detail-normal map exactly as production does — its N² gradient cost shows in the timings.
    crate::sdf_render::worldgen::upload::set_cpu_terrain_hifi(Some(Arc::new(manager.make_terrain_hifi())));
    set_cpu_height_clipmap(Some(clipmap.clone()));
    clipmap
}

/// Per-LOD residency / surface counts for one camera position.
#[derive(Default, Clone)]
struct ResidencyCounts {
    /// Resident (shell ∩ coverage-gated) chunk count per LOD.
    resident_by_lod: Vec<usize>,
    /// Surface (resident ∩ `chunk_has_surface`) chunk count per LOD — the chunks that actually bake.
    surface_by_lod: Vec<usize>,
}

impl ResidencyCounts {
    fn resident_total(&self) -> usize {
        self.resident_by_lod.iter().sum()
    }
    fn surface_total(&self) -> usize {
        self.surface_by_lod.iter().sum()
    }
    fn culled_total(&self) -> usize {
        self.resident_total() - self.surface_total()
    }
}

/// FAITHFUL replica of the production residency + narrow-band cull (mirrors `mesh_resident_chunks`
/// lines ~869-921 for residency and line ~1124 for the per-chunk surface cull). Given the live edits,
/// the two configs, and a camera, returns: the SURFACE chunk set (the chunks that WOULD actually call
/// `mesh_chunk` — shell-resident ∩ coverage-gated ∩ `chunk_has_surface`), and the per-LOD resident /
/// surface counts (so the resident-but-culled empty count = resident − surface is reported).
///
/// KEEP IN LOCKSTEP with `mesh_resident_chunks`: every formula here (the `edit_aabbs` BLEND_REACH pad,
/// the `edit_extent`, the `shell_cube` clip, `chunks_in_aabb`, `mesh_chunk_in_shell`, the coverage gate,
/// the `chunk_sampled` cull + `edit_resolvable_at` retain, `chunk_has_surface`) is copied from there.
fn surface_chunks(
    edits: &[ResolvedEdit],
    config: &SdfGridConfig,
    mesh_cfg: &MeshBakeConfig,
    cam: Vec3,
    clipmap: &HeightClipmap,
) -> (HashSet<BrickKey>, ResidencyCounts) {
    let k = mesh_cfg.chunk_bricks.clamp(1, 8);
    let cam = Some(cam);

    // --- Per-edit AABB (padded by BLEND_REACH, like production) + sub-voxel-cull extent. ---
    let n_edits = edits.len();
    let mut edit_aabbs: Vec<Aabb3d> = Vec::with_capacity(n_edits);
    let mut edit_extent: Vec<f32> = Vec::with_capacity(n_edits);
    for e in edits {
        let aabb = edits::edit_world_aabb(&e.prim, &e.transform, e.op.smoothing);
        edit_extent.push((Vec3::from(aabb.max) - Vec3::from(aabb.min)).max_element());
        let pad = bevy::math::Vec3A::splat(BLEND_REACH);
        edit_aabbs.push(Aabb3d { min: aabb.min - pad, max: aabb.max + pad });
    }

    // --- Coverage-gate inputs: the world-XZ AABBs of every Terrain edit (mirror of production). ---
    let terrain_xz_aabbs: Vec<(Vec2, Vec2)> = edits
        .iter()
        .zip(&edit_aabbs)
        .filter(|(e, _)| matches!(e.prim, SdfPrimitive::Terrain { .. }))
        .map(|(_, a)| {
            // Use the UNPADDED edit AABB XZ exactly as production builds `terrain_xz_aabbs` from `gathered`.
            (Vec2::new(a.min.x, a.min.z), Vec2::new(a.max.x, a.max.z))
        })
        .collect();

    let half0 = lod0_half_chunks(config, mesh_cfg, k);
    let lod_count = effective_lod_count(config, mesh_cfg, cam.is_some());
    let cw0 = k as f32 * config.brick_world_size(0);

    // Padded sampled AABB of a chunk (cell span + 1-voxel apron) — production's `chunk_sampled` closure.
    let chunk_sampled = |key: BrickKey| -> Aabb3d {
        let b = chunk_aabb(key, config, k);
        let apron = Vec3::splat(config.voxel_size_at(key.lod));
        Aabb3d::from_min_max(Vec3::from(b.min) - apron, Vec3::from(b.max) + apron)
    };

    // --- RESIDENCY: per LOD, the shell-resident ∩ coverage-gated chunks (mirror of the production loop). ---
    let mut resident: HashSet<BrickKey> = HashSet::new();
    let mut cand: HashSet<BrickKey> = HashSet::new();
    for lod in 0..lod_count {
        cand.clear();
        let shell_cube = cam.map(|c| {
            let centre = lod_centre(config, k, c, lod).as_vec3() * cw0;
            let half = (half0 << lod) as f32 * cw0;
            (centre - Vec3::splat(half), centre + Vec3::splat(half))
        });
        for (ei, a) in edit_aabbs.iter().enumerate() {
            if !edit_resolvable_at(edit_extent[ei], config, lod) {
                continue;
            }
            let clipped = match shell_cube {
                Some((smin, smax)) => {
                    let mn = Vec3::from(a.min).max(smin);
                    let mx = Vec3::from(a.max).min(smax);
                    if mn.cmpgt(mx).any() {
                        continue;
                    }
                    Aabb3d::from_min_max(mn, mx)
                }
                None => *a,
            };
            chunks_in_aabb(&clipped, config, k, lod, &mut cand);
        }
        for &key in &cand {
            if !mesh_chunk_in_shell(key, config, k, cam, half0) {
                continue;
            }
            if !terrain_xz_aabbs.is_empty()
                && !terrain_chunk_covered(key, config, k, &terrain_xz_aabbs, Some(clipmap))
            {
                continue;
            }
            resident.insert(key);
        }
    }

    // --- Per-chunk narrow-band SURFACE cull (mirror of the bake-loop, line ~1124). A chunk WOULD bake iff
    // its blend-padded cull set (after the sub-voxel retain) has a surface crossing. ---
    let mut counts = ResidencyCounts {
        resident_by_lod: vec![0; lod_count as usize],
        surface_by_lod: vec![0; lod_count as usize],
    };
    let mut surface: HashSet<BrickKey> = HashSet::with_capacity(resident.len());
    let mut idx: Vec<u32> = Vec::new();
    for &key in &resident {
        let l = key.lod as usize;
        counts.resident_by_lod[l] += 1;
        let vs_l = config.voxel_size_at(key.lod);
        cull_into(&edit_aabbs, &chunk_sampled(key), &mut idx);
        idx.retain(|&i| {
            let a = edit_aabbs[i as usize];
            edit_resolvable_at((Vec3::from(a.max) - Vec3::from(a.min)).max_element(), config, key.lod)
        });
        if chunk_has_surface(edits, &idx, config, k, key, vs_l) {
            counts.surface_by_lod[l] += 1;
            surface.insert(key);
        }
    }
    (surface, counts)
}

/// Cull set (sub-voxel-retained) for ONE chunk against the live edits — the exact `indices` slice
/// production passes to `mesh_chunk` for `key`. Used by the timing loop so the measured bake folds the
/// same edit subset the real bake would.
fn chunk_cull_indices(
    edits: &[ResolvedEdit],
    config: &SdfGridConfig,
    k: u32,
    key: BrickKey,
) -> Vec<u32> {
    let mut edit_aabbs: Vec<Aabb3d> = Vec::with_capacity(edits.len());
    for e in edits {
        let aabb = edits::edit_world_aabb(&e.prim, &e.transform, e.op.smoothing);
        let pad = bevy::math::Vec3A::splat(BLEND_REACH);
        edit_aabbs.push(Aabb3d { min: aabb.min - pad, max: aabb.max + pad });
    }
    let b = chunk_aabb(key, config, k);
    let apron = Vec3::splat(config.voxel_size_at(key.lod));
    let sampled = Aabb3d::from_min_max(Vec3::from(b.min) - apron, Vec3::from(b.max) + apron);
    let mut idx: Vec<u32> = Vec::new();
    cull_into(&edit_aabbs, &sampled, &mut idx);
    idx.retain(|&i| {
        let a = edit_aabbs[i as usize];
        edit_resolvable_at((Vec3::from(a.max) - Vec3::from(a.min)).max_element(), config, key.lod)
    });
    idx
}

/// Bake one chunk with the REAL `mesh_chunk` (no apron; transition flags from the live shell) and return
/// `(micros, verts, tris)`. Mirrors the production bake-task body (line ~1136).
fn time_one_chunk(
    edits: &[ResolvedEdit],
    config: &SdfGridConfig,
    mesh_cfg: &MeshBakeConfig,
    cam: Vec3,
    key: BrickKey,
) -> (u128, usize, usize) {
    let k = mesh_cfg.chunk_bricks.clamp(1, 8);
    let cs = config.cell_stride() as u32;
    let half0 = lod0_half_chunks(config, mesh_cfg, k);
    let idx = chunk_cull_indices(edits, config, k, key);
    let grid_origin = config.brick_min_world(key.coord, key.lod);
    let flags = chunk_finer_faces(key, config, k, Some(cam), half0);
    let vs_l = config.voxel_size_at(key.lod);
    let t = std::time::Instant::now();
    // Pass the published clipmap as the per-bake snapshot (mirrors production `round.clipmap`) so the
    // DETAIL-NORMAL bake's `bake_terrain_hifi()` sees the matching tier-0 hi-fi source and bakes the per-chunk
    // map — its N² gradient cost is then included in the timing. The terrain perf scene is terrain-only, so
    // `terrain_only = true`; `detail_normal_res` from the live config (default 128) drives the bake (gated to
    // coarse LODs inside `mesh_chunk`).
    let terrain = crate::sdf_render::worldgen::upload::cpu_height_clipmap();
    let out = mesh_chunk(
        edits, &idx, grid_origin, vs_l, k * cs, flags, key.lod, false, terrain, true,
        mesh_cfg.detail_normal_res,
    );
    let us = t.elapsed().as_micros();
    let (verts, tris) = out.map_or((0, 0), |d| (d.positions.len(), d.indices.len() / 3));
    (us, verts, tris)
}

/// `p`-percentile (0..=100) of `xs` by value (nearest-rank). `xs` is sorted in place. Empty → 0.
fn pct(xs: &mut [u128], p: u32) -> u128 {
    if xs.is_empty() {
        return 0;
    }
    xs.sort_unstable();
    let n = xs.len();
    let rank = ((p as usize * n).div_ceil(100)).clamp(1, n) - 1;
    xs[rank]
}

fn pct_usize(xs: &mut [usize], p: u32) -> usize {
    if xs.is_empty() {
        return 0;
    }
    xs.sort_unstable();
    let n = xs.len();
    let rank = ((p as usize * n).div_ceil(100)).clamp(1, n) - 1;
    xs[rank]
}

#[test]
#[ignore = "perf measurement rig; run explicitly with --release --ignored --nocapture"]
fn mesh_bake_perf_terrain() {
    let config = SdfGridConfig::default();
    let mesh_cfg = MeshBakeConfig::default(); // production: lod_count 9 (LOD 0..=8), lod0_radius 16, K=4.
    let k = mesh_cfg.chunk_bricks.clamp(1, 8);
    let lod_count = effective_lod_count(&config, &mesh_cfg, true);
    let half0 = lod0_half_chunks(&config, &mesh_cfg, k);
    let cw0 = k as f32 * config.brick_world_size(0);
    let reach = coarsest_lod_outer_reach(&config, &mesh_cfg);

    let edits = vec![terrain_edit()];

    // A sensible eye height above the terrain band (the worldgen camera frames the origin). Fixed for the
    // whole cold-fill scenario.
    let cam0 = Vec3::new(0.0, 40.0, 0.0);
    let focus0 = DVec2::new(cam0.x as f64, cam0.z as f64);

    // Load the REAL authored terrain graph (falls back to legacy with a note if missing/unparseable).
    let graph = load_world_graph();
    let terrain_src = if graph.is_some() { "AUTHORED graph (assets/worldgen/world.graph.ron)" } else { "LEGACY HeightParams" };

    // Build + publish the full tiered clipmap settled around the camera (required before any bake — the
    // strict Terrain sampler panics on a coverage miss). Time the cold generation of the settled window.
    let mut gen_st = GenStats::default();
    let clipmap = build_and_publish_clipmap(&config, &mesh_cfg, focus0, graph.as_ref(), Some(&mut gen_st));

    eprintln!(
        "MESH-BAKE-PERF: terrain = {} edit(s) [{terrain_src}] | mesh-bake clipmap: LODs 0..={} (K={k}, lod0_radius={}, half0={half0} chunks, cw0={cw0:.2} m, coarsest reach=±{reach:.0} m) | height clipmap: {} tiers | MAX_NEW_TASKS_PER_FRAME={MAX_NEW_TASKS_PER_FRAME}",
        edits.len(),
        lod_count - 1,
        mesh_cfg.lod0_radius,
        clipmap.len(),
    );

    // ================= [gen] — cold height-chunk GENERATION of the settled window =================
    // The COLD generation cost (the optimization target): how long to generate every height chunk of the
    // settled clipmap window from empty. Each chunk = `HeightLayer::generate` evaluating the terrain graph
    // over `(HEIGHT_FIELD_RES+1)²` field nodes. This is the SERIAL baseline before parallelization.
    eprintln!(
        "MESH-BAKE-PERF [gen]: COLD generate {} height chunks in {:.2}ms ({} update(s), budget=1M) | {} samples ({}² nodes/chunk × {} chunks) | {:.1} us/chunk | {:.0} chunks/sec",
        gen_st.chunks,
        gen_st.settle_ms,
        gen_st.updates,
        gen_st.total_samples(),
        HEIGHT_FIELD_RES + 1,
        gen_st.chunks,
        gen_st.us_per_chunk(),
        gen_st.chunks_per_sec(),
    );
    eprintln!("MESH-BAKE-PERF [gen] chunks by tier:");
    for (t, &n) in gen_st.chunks_by_tier.iter().enumerate() {
        if n == 0 {
            continue;
        }
        eprintln!("    tier {t}: {n} chunks");
    }

    // ================= SCENARIO A — cold full-LOD-8 fill =================
    let (surface0, counts0) = surface_chunks(&edits, &config, &mesh_cfg, cam0, &clipmap);
    let resident_total = counts0.resident_total();
    let surface_total = counts0.surface_total();
    let culled_total = counts0.culled_total();

    eprintln!("MESH-BAKE-PERF [shell-cull] per LOD (surface / resident = ratio):");
    for l in 0..lod_count as usize {
        let res = counts0.resident_by_lod[l];
        let surf = counts0.surface_by_lod[l];
        if res == 0 {
            continue;
        }
        let ratio = surf as f64 / res as f64;
        eprintln!("    LOD {l}: {surf} / {res}  = {:.3}  ({} culled empty)", ratio, res - surf);
    }
    let overall_ratio = if resident_total == 0 { 0.0 } else { surface_total as f64 / resident_total as f64 };
    eprintln!(
        "MESH-BAKE-PERF [shell-cull] OVERALL: surface={surface_total} / resident={resident_total} = {overall_ratio:.3}  ({culled_total} resident-but-culled empty chunks)"
    );

    // Bake every surface chunk with the REAL `mesh_chunk`; collect timings + geometry.
    let mut chunk_us: Vec<u128> = Vec::with_capacity(surface_total);
    let mut us_by_lod = vec![0u128; lod_count as usize];
    let mut total_verts = 0usize;
    let mut total_tris = 0usize;
    let cold_start = std::time::Instant::now();
    for &key in &surface0 {
        let (us, verts, tris) = time_one_chunk(&edits, &config, &mesh_cfg, cam0, key);
        chunk_us.push(us);
        us_by_lod[key.lod as usize] += us;
        total_verts += verts;
        total_tris += tris;
    }
    let cold_total_ms = cold_start.elapsed().as_secs_f64() * 1000.0;
    let worst_us = chunk_us.iter().copied().max().unwrap_or(0);
    let mean_us = if chunk_us.is_empty() { 0 } else { chunk_us.iter().sum::<u128>() / chunk_us.len() as u128 };
    let p50_us = pct(&mut chunk_us.clone(), 50);
    let p99_us = pct(&mut chunk_us.clone(), 99);
    let frames_to_fill = surface_total.div_ceil(MAX_NEW_TASKS_PER_FRAME);

    eprintln!(
        "MESH-BAKE-PERF [cold-fill]: surface_chunks={surface_total} | mesh_chunk CPU total={cold_total_ms:.2}ms (single-threaded sum; the pool runs these concurrently) | per-chunk us: mean={mean_us} p50={p50_us} p99={p99_us} WORST={worst_us} | geometry: verts={total_verts} tris={total_tris} | frames-to-fill @256/frame = {frames_to_fill}"
    );
    eprintln!("MESH-BAKE-PERF [cold-fill] mesh_chunk CPU time by LOD (ms):");
    for (l, &us) in us_by_lod.iter().enumerate() {
        let surf = counts0.surface_by_lod[l];
        if surf == 0 {
            continue;
        }
        eprintln!(
            "    LOD {l}: {:.2}ms over {surf} surface chunks ({:.1} us/chunk)",
            us as f64 / 1000.0,
            us as f64 / surf.max(1) as f64,
        );
    }

    // ================= SCENARIO B — streaming walk (+X) =================
    // Walk the camera +X one LOD-0-chunk-width (`cw0`) per step for `steps` steps. Each step: rebuild +
    // publish the clipmap settled at the new focus (the ring rolls), recompute the surface-chunk set, diff
    // vs the previous step → new-entering / leaving, and BAKE only the new ones (timing them). The
    // bounded-streaming verdict asserts max(new/step) is a small fraction of the resident set.
    let steps = 64usize;
    let mut prev_surface = surface0.clone();
    let mut new_per_step: Vec<usize> = Vec::with_capacity(steps);
    let mut left_per_step: Vec<usize> = Vec::with_capacity(steps);
    let mut step_bake_us: Vec<u128> = Vec::with_capacity(steps);
    let mut step_resident: Vec<usize> = Vec::with_capacity(steps);

    for s in 1..=steps {
        let cam = cam0 + Vec3::new(s as f32 * cw0, 0.0, 0.0);
        let focus = DVec2::new(cam.x as f64, cam.z as f64);
        let clip = build_and_publish_clipmap(&config, &mesh_cfg, focus, graph.as_ref(), None);
        let (surf, counts) = surface_chunks(&edits, &config, &mesh_cfg, cam, &clip);

        let entering: Vec<BrickKey> = surf.difference(&prev_surface).copied().collect();
        let leaving = prev_surface.difference(&surf).count();
        new_per_step.push(entering.len());
        left_per_step.push(leaving);
        step_resident.push(counts.resident_total());

        // Bake ONLY the newly-entering surface chunks (the leading-edge work a real streaming frame pays).
        let t = std::time::Instant::now();
        for &key in &entering {
            let _ = time_one_chunk(&edits, &config, &mesh_cfg, cam, key);
        }
        step_bake_us.push(t.elapsed().as_micros());

        prev_surface = surf;
    }

    let new_mean = if new_per_step.is_empty() { 0 } else { new_per_step.iter().sum::<usize>() / new_per_step.len() };
    let new_max = new_per_step.iter().copied().max().unwrap_or(0);
    let new_p50 = pct_usize(&mut new_per_step.clone(), 50);
    let new_p99 = pct_usize(&mut new_per_step.clone(), 99);
    let bake_mean = if step_bake_us.is_empty() { 0 } else { step_bake_us.iter().sum::<u128>() / step_bake_us.len() as u128 };
    let bake_max = step_bake_us.iter().copied().max().unwrap_or(0);
    let bake_p99 = pct(&mut step_bake_us.clone(), 99);
    let steady_resident = *step_resident.last().unwrap_or(&resident_total);

    // BOUNDED-STREAMING VERDICT: max new-chunks-per-step must be a SMALL FRACTION of the resident set —
    // proving the world-anchored per-chunk staging streams a leading edge, not a whole-band re-bake.
    let bound = steady_resident / 4;
    let bounded = new_max < bound.max(1);

    eprintln!(
        "MESH-BAKE-PERF [streaming]: {steps} steps of {cw0:.2} m (+X) | new surface chunks/step: mean={new_mean} p50={new_p50} p99={new_p99} MAX={new_max} | leaving/step mean={} | per-step new-chunk bake us: mean={bake_mean} p99={bake_p99} MAX={bake_max} | resident set ≈ {steady_resident}",
        if left_per_step.is_empty() { 0 } else { left_per_step.iter().sum::<usize>() / left_per_step.len() },
    );
    eprintln!(
        "MESH-BAKE-PERF [streaming] BOUNDED VERDICT: max_new_per_step={new_max} {} bound={bound} (resident/4)  ⇒  {}",
        if bounded { "<" } else { ">=" },
        if bounded { "BOUNDED (leading-edge streaming OK)" } else { "UNBOUNDED — whole-band re-bake REGRESSION" },
    );

    // Structured JSON for before/after diffing.
    write_json(
        &config,
        &mesh_cfg,
        lod_count,
        &gen_st,
        &counts0,
        cold_total_ms,
        worst_us,
        mean_us,
        p50_us,
        p99_us,
        total_verts,
        total_tris,
        frames_to_fill,
        new_mean,
        new_p50,
        new_p99,
        new_max,
        bake_max,
        steady_resident,
        bounded,
    );

    // Clear the published clipmap so the global doesn't leak into other tests in the same process.
    set_cpu_height_clipmap(None);
    crate::sdf_render::worldgen::upload::set_cpu_terrain_hifi(None);

    assert!(
        bounded,
        "streaming is UNBOUNDED: max new surface chunks/step = {new_max} ≥ resident/4 = {bound} — a camera \
         walk is re-baking a whole band, not a leading edge. This is a real regression in the world-anchored \
         per-chunk staging; investigate before landing."
    );
}

/// Append a JSON line with the structured metrics to `.soul/mesh_bake_perf.json` (best-effort — a write
/// failure just warns). Each run overwrites, so the file always holds the latest baseline.
#[allow(clippy::too_many_arguments)]
fn write_json(
    config: &SdfGridConfig,
    mesh_cfg: &MeshBakeConfig,
    lod_count: u32,
    gen_st: &GenStats,
    counts: &ResidencyCounts,
    cold_total_ms: f64,
    worst_us: u128,
    mean_us: u128,
    p50_us: u128,
    p99_us: u128,
    verts: usize,
    tris: usize,
    frames_to_fill: usize,
    new_mean: usize,
    new_p50: usize,
    new_p99: usize,
    new_max: usize,
    stream_bake_max_us: u128,
    steady_resident: usize,
    bounded: bool,
) {
    let per_lod: Vec<String> = (0..lod_count as usize)
        .map(|l| {
            let res = counts.resident_by_lod[l];
            let surf = counts.surface_by_lod[l];
            let ratio = if res == 0 { 0.0 } else { surf as f64 / res as f64 };
            format!("{{\"lod\":{l},\"resident\":{res},\"surface\":{surf},\"cull_ratio\":{ratio:.4}}}")
        })
        .collect();
    let resident_total = counts.resident_total();
    let surface_total = counts.surface_total();
    let overall_ratio = if resident_total == 0 { 0.0 } else { surface_total as f64 / resident_total as f64 };
    let k = mesh_cfg.chunk_bricks.clamp(1, 8);
    let gen_by_tier: Vec<String> = gen_st.chunks_by_tier.iter().map(|n| n.to_string()).collect();
    let body = format!(
        "{{\"config\":{{\"lod_count\":{lod_count},\"chunk_bricks\":{k},\"lod0_radius\":{},\"voxel_size\":{}}},\
         \"gen\":{{\"settle_ms\":{:.3},\"chunks\":{},\"updates\":{},\"samples\":{},\"samples_per_chunk\":{},\
         \"us_per_chunk\":{:.2},\"chunks_per_sec\":{:.1},\"chunks_by_tier\":[{}]}},\
         \"cold_fill\":{{\"resident_total\":{resident_total},\"surface_total\":{surface_total},\"culled_total\":{},\
         \"overall_cull_ratio\":{overall_ratio:.4},\"per_lod\":[{}],\
         \"mesh_chunk_cpu_total_ms\":{cold_total_ms:.3},\"per_chunk_us\":{{\"mean\":{mean_us},\"p50\":{p50_us},\"p99\":{p99_us},\"worst\":{worst_us}}},\
         \"verts\":{verts},\"tris\":{tris},\"frames_to_fill\":{frames_to_fill}}},\
         \"streaming\":{{\"steps\":64,\"new_per_step\":{{\"mean\":{new_mean},\"p50\":{new_p50},\"p99\":{new_p99},\"max\":{new_max}}},\
         \"per_step_bake_max_us\":{stream_bake_max_us},\"steady_resident\":{steady_resident},\"bounded\":{bounded}}}}}\n",
        mesh_cfg.lod0_radius,
        config.voxel_size,
        gen_st.settle_ms,
        gen_st.chunks,
        gen_st.updates,
        gen_st.total_samples(),
        GenStats::samples_per_chunk(),
        gen_st.us_per_chunk(),
        gen_st.chunks_per_sec(),
        gen_by_tier.join(","),
        resident_total - surface_total,
        per_lod.join(","),
    );
    if let Err(e) = std::fs::create_dir_all(".soul").and_then(|()| std::fs::write(".soul/mesh_bake_perf.json", &body)) {
        eprintln!("MESH-BAKE-PERF: could not write .soul/mesh_bake_perf.json: {e}");
    } else {
        eprintln!("MESH-BAKE-PERF: wrote .soul/mesh_bake_perf.json");
    }
}

/// REGRESSION (the FAR-camera "Apply" panic): the SYNCHRONOUS narrow-band cull `chunk_has_surface` — run
/// inside `mesh_resident_chunks`' REQUEST loop — must sample the ROUND's FROZEN clipmap snapshot (the SAME
/// one whose coverage gate admitted the chunk), NOT the live process-global `cpu_height_clipmap()`.
///
/// THE BUG: a bake round freezes its residency + `round.clipmap` together (mutually consistent — the gate
/// admitted the residency against that very clipmap). But `chunk_has_surface` installed NO `BAKE_TERRAIN`
/// snapshot, so `terrain_sdf` fell through to the GLOBAL. When `roll_worldgen` rebuilt/rolled that global
/// to DIFFERENT (sparser) coverage after the round froze — clicking "Apply" republishes the graph →
/// `set_graph` evicts residency → the rebuilt clipmap transiently has its COARSE tiers empty/un-restreamed,
/// or a camera roll/`lod_count` rebuild drops a tier — a far, coarse chunk the gate had admitted (against
/// the full frozen clipmap) tripped the strict `sample_clipmap_lod` panic in the cull (against the now-
/// uncovered live global). That is the reported panic at `upload.rs:490` inside `mesh_resident_chunks`.
///
/// THE FIX installs the round's frozen snapshot for the whole REQUEST loop, so gate + sync cull + async
/// bake share ONE clipmap SSOT and a gate-admitted chunk can NEVER sample uncovered ground.
///
/// This test forges that exact desync WITHOUT an App: a fully-settled clipmap (`frozen`) covers a far,
/// coarse chunk via a coarse tier; the live global is that clipmap with its covering coarse tiers DROPPED
/// (the post-"Apply" transient where coarse tiers haven't re-streamed). With the frozen snapshot installed,
/// the cull resolves cleanly; with only the truncated global (the OLD behaviour), the same cull panics in
/// the strict sampler — proving the discrepancy is real and that the frozen-snapshot SSOT closes it.
#[test]
fn sync_cull_uses_frozen_clipmap_not_rolled_global() {
    use crate::sdf_render::worldgen::upload::{
        clipmap_covers_aabb, cpu_terrain_offset, set_bake_terrain,
    };

    let config = SdfGridConfig::default();
    let mesh_cfg = MeshBakeConfig::default();
    let k = mesh_cfg.chunk_bricks.clamp(1, 8);
    let edits = vec![terrain_edit()];
    let idx = vec![0u32]; // the single Terrain edit overlaps every chunk

    // The round's FROZEN snapshot: a fully-settled clipmap around the origin (all tiers present). Legacy
    // path (graph=None) — this regression test is about clipmap coverage desync, not the terrain source.
    let frozen = build_and_publish_clipmap(&config, &mesh_cfg, DVec2::ZERO, None, None);

    // A far, coarse chunk like the production panic (`world_xz≈(-17203,-2867)`, a coarse tier). Build its
    // gate footprint (chunk AABB ± the gate margin) and find the FINEST clipmap tier that covers it.
    let far_xz = Vec2::new(-17203.0, -2867.0);
    // Choose a coarse LOD whose chunk world-size is comparable to the panic's voxel_size·subdivisions.
    let lod = (effective_lod_count(&config, &mesh_cfg, true) - 1).min(7);
    let stride = k as i32 * config.cell_stride();
    let cw = k as f32 * config.brick_world_size(lod);
    let jx = (far_xz.x / cw).floor() as i32;
    let jz = (far_xz.y / cw).floor() as i32;
    let victim = BrickKey::new(lod, IVec3::new(jx, 0, jz) * stride);
    let vs = config.voxel_size_at(lod);
    let b = chunk_aabb(victim, &config, k);
    let m = Vec2::splat(vs + 2.0 * HEIGHT_CHUNK_CELLS as f32); // mirror of `terrain_chunk_covered`
    let cmin = Vec2::new(b.min.x, b.min.z) - m;
    let cmax = Vec2::new(b.max.x, b.max.z) + m;

    // The full frozen clipmap must cover the far footprint (some coarse tier reaches it) — that's WHY the
    // gate admitted the chunk into the round.
    assert!(
        clipmap_covers_aabb(&frozen, cmin, cmax),
        "the fully-settled frozen clipmap must cover the far chunk (the gate admitted it on this basis)"
    );
    // Find the finest covering tier, then forge a "rolled global" = the frozen clipmap with EVERY tier from
    // that one up DROPPED (the post-Apply transient: coarse tiers evicted / not yet re-streamed). The far
    // footprint is then UNCOVERED by the global but still covered by the full frozen snapshot.
    let cover_tier = frozen
        .iter()
        .position(|r| {
            crate::sdf_render::worldgen::upload::ring_covers_aabb(r, cmin, cmax)
        })
        .expect("a covering tier exists (asserted above)");
    let rolled_global: Arc<HeightClipmap> = Arc::new(frozen[..cover_tier].to_vec());
    assert!(
        !clipmap_covers_aabb(&rolled_global, cmin, cmax),
        "the rolled global (covering coarse tiers dropped) must NOT cover the far chunk — the desync"
    );

    // Publish the rolled (uncovered-far) global as the live process-global (what Apply left behind).
    set_cpu_height_clipmap(Some(rolled_global.clone()));

    // WITH the frozen snapshot installed (the FIX): the sync cull samples `frozen` → resolves, no panic.
    {
        let _g = set_bake_terrain(Some(frozen.clone()), cpu_terrain_offset());
        let _ = chunk_has_surface(&edits, &idx, &config, k, victim, vs); // must not panic
    }

    // WITHOUT the snapshot (the OLD behaviour): the sync cull reads the rolled global → strict panic. Catch
    // it to prove the discrepancy is real (and to leave no installed snapshot behind for sibling tests).
    let bug = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = chunk_has_surface(&edits, &idx, &config, k, victim, vs);
    }));
    assert!(
        bug.is_err(),
        "expected the rolled-global cull to panic on the uncovered far chunk (the bug the frozen snapshot fixes)"
    );

    set_cpu_height_clipmap(None); // clean up the global for sibling tests
    crate::sdf_render::worldgen::upload::set_cpu_terrain_hifi(None);
}
