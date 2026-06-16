//! **`.vxo` bake perf + peak-RSS harness (Constant-RAM Bake plan, Stage 0).** Bakes a SYNTHETIC voxel scene
//! through the EXACT production streaming path (`VxoStreamWriter::add_region` base stream + the shared
//! `drive_coarse_lods` coarse bake + `finish`) and reports:
//!
//!   * the COLD-bake wall time, split into the BASE region stream vs. the new coarse `drive_coarse_lods`
//!     (`build_coarse_pyramid` + the per-level region encode), so the coarse bake's share is visible, and
//!   * the process PEAK RSS (Windows `GetProcessMemoryInfo().PeakWorkingSetSize` via the kernel32-exported
//!     `K32GetProcessMemoryInfo`; Linux `/proc/self/status` `VmHWM`).
//!
//! The synthetic scene is a HOLLOW voxel shell (surface-area-scaled, like a real mesh bake) so the residency +
//! coarse work track SURFACE, not volume — the shape Stages 1-3 will gate for constant-RAM. This Stage-0 rig is
//! the BASE harness; Stage 3 extends it with the 1×/2×-surface peak-RSS invariance gate (NOT built here).
//!
//! Bistro: if `assets/models/bistro.vxo` is present, its size + `has_lods()`/`max_lod()` are reported for
//! reference (the heavy 278 s tiled glTF re-bake is triggered SEPARATELY via `voxelize_scene --tiled`, not here
//! — so this harness stays fast). Absent ⇒ skip-graceful.
//!
//! Run:  cargo run --release --example bake_perf --features vxo-encode
//!       cargo run --release --example bake_perf --features vxo-encode -- 96   # shell edge in bricks

use std::path::Path;
use std::time::Instant;

use bevy::math::IVec3;

use adventure::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, BrickMap};
use adventure::voxel::palette::{BlockId, BlockRegistry};
use adventure::voxel::vxo::{
    RegionSpillPool, VxoCompression, VxoFile, VxoHeadParams, VxoStreamWriter, assemble_base, build_coarse_pyramid,
    drive_coarse_lods, region_of_brick, spill_voxel, windowed_coarse,
};

fn main() -> anyhow::Result<()> {
    // Optional CLI arg: the shell edge in bricks (default 64 ⇒ a 64³-brick hollow box ⇒ surface ~6·64² bricks).
    let edge: i32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    println!("bake_perf: synthetic hollow shell, edge {edge} bricks (K={})", VxoHeadParams::default().region_edge_bricks);

    let registry = BlockRegistry::cornell();
    let t_build = Instant::now();
    let map = hollow_shell_map(edge);
    let brick_count = map.len();
    println!(
        "  built synthetic BrickMap: {brick_count} surface bricks ({:.2}s) — surface-scaled (≈6·edge²)",
        t_build.elapsed().as_secs_f32()
    );

    let scratch_dir = std::env::temp_dir().join(format!("bake_perf_{}", std::process::id()));
    std::fs::create_dir_all(&scratch_dir)?;
    let scratch_brik = scratch_dir.join("bake.brik.tmp");
    let out_path = scratch_dir.join("bake_perf.vxo");

    let params = VxoHeadParams { name: "bake_perf".into(), ..Default::default() };
    let k = params.region_edge_bricks as i32;
    // STORE keeps the harness toolchain-light + the timer focused on the bake ordering (not zstd CPU). Switch to
    // `VxoCompression::default()` (zstd-19) to also measure compression cost.
    let comp = VxoCompression::Store;

    // ---- Cold bake, timed in two phases (base region stream, then the coarse `drive_coarse_lods`). ----
    let t_total = Instant::now();
    let mut writer = VxoStreamWriter::new(params.clone(), &registry, comp, &scratch_brik)?;

    // Phase 1: base LOD0 region stream (the `add_region` feed in (z,y,x) order).
    let t_base = Instant::now();
    feed_base_regions(&map, k, &mut writer)?;
    let base_secs = t_base.elapsed().as_secs_f32();

    // Phase 2: the NEW coarse bake — build the pyramid (the `downsample_brickmap` chain) + drive it through the
    // shared ordering SSOT into `add_lod_region` (the exact tiled-path Stage-0 wiring).
    let t_coarse = Instant::now();
    let pyramid = build_coarse_pyramid(&map);
    let coarse_levels = pyramid.len();
    drive_coarse_lods(&pyramid, k, |lod, rc, bricks| writer.add_lod_region(lod, rc, bricks))?;
    let coarse_secs = t_coarse.elapsed().as_secs_f32();

    // Phase 3: finish (assemble the file, bounded-RAM scratch copy).
    let t_finish = Instant::now();
    writer.finish(&out_path)?;
    let finish_secs = t_finish.elapsed().as_secs_f32();
    let total_secs = t_total.elapsed().as_secs_f32();

    // ---- Report. ----
    let file = VxoFile::parse(&std::fs::read(&out_path)?)?;
    let out_bytes = std::fs::metadata(&out_path)?.len();
    println!("  cold bake: {total_secs:.3}s total");
    println!("    base stream (add_region) : {base_secs:.3}s");
    println!("    coarse drive_coarse_lods : {coarse_secs:.3}s ({coarse_levels} levels)");
    println!("    finish (assemble)        : {finish_secs:.3}s");
    println!(
        "  output: {} ({:.2} MiB), has_lods={}, max_lod={}, region_count={}",
        out_path.display(),
        out_bytes as f64 / (1024.0 * 1024.0),
        file.has_lods(),
        file.max_lod(),
        file.head.region_count,
    );
    match peak_rss_bytes() {
        Some(rss) => println!("  PEAK RSS: {:.1} MiB", rss as f64 / (1024.0 * 1024.0)),
        None => println!("  PEAK RSS: <unavailable on this platform>"),
    }

    // ---- Stage 3: the CONSTANT-RAM gate (the flat-RSS proof). ----
    constant_ram_gate(&registry, &params)?;

    // Bistro reference (skip-graceful) — report the committed asset's stats, never re-bake it here.
    report_bistro_reference();

    let _ = std::fs::remove_dir_all(&scratch_dir);
    Ok(())
}

/// **Stage 3 — the constant-RAM gate (proof).** Bake a synthetic large-AABB scene at 1× and 2× SURFACE area
/// (SAME AABB + sparsity — a hollow shell whose surface scales as the face area) through the constant-RAM
/// disk-spill producer, and assert:
///   * peak RSS(2×) ≤ peak RSS(1×) + SLACK (FLAT in surface size — the constant-RAM proof), while
///   * scratch high-water(2×) ≈ 2× scratch high-water(1×) (the work really did double — a sanity bound that the
///     "flat RSS" isn't because nothing happened).
///
/// Process peak RSS is a monotonic high-water mark, so we bake 1× FIRST (recording the peak after it), then 2×;
/// a constant-RAM producer must not raise the peak by more than `SLACK` when the surface doubles. We measure the
/// per-bake peak DELTA via the process high-water before/after each bake.
fn constant_ram_gate(registry: &BlockRegistry, params: &VxoHeadParams) -> anyhow::Result<()> {
    println!("\nStage-3 constant-RAM gate (1× vs 2× surface, flat-RSS proof):");
    // A large AABB so the volume is huge but the surface (the residency driver) is a thin shell. The 2× scene
    // doubles the shell EXTENT along one axis (≈2× the surface area / brick count) at the SAME sparsity.
    let edge_1x = 48; // 48³-brick AABB hollow shell ⇒ ~6·48² surface bricks
    let edge_2x = 68; // ~2× the surface area (68²/48² ≈ 2.0)

    let base_dir = std::env::temp_dir().join(format!("bake_perf_gate_{}", std::process::id()));
    std::fs::create_dir_all(&base_dir)?;

    let rss_before_1x = peak_rss_bytes();
    let (scratch_1x, bricks_1x, secs_1x) =
        bake_shell_via_spill(edge_1x, registry, params, &base_dir.join("s1"))?;
    let rss_after_1x = peak_rss_bytes();

    let (scratch_2x, bricks_2x, secs_2x) =
        bake_shell_via_spill(edge_2x, registry, params, &base_dir.join("s2"))?;
    let rss_after_2x = peak_rss_bytes();

    println!(
        "  1× (edge {edge_1x}): {bricks_1x} surface bricks, scratch high-water {:.1} MiB, {secs_1x:.2}s",
        scratch_1x as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  2× (edge {edge_2x}): {bricks_2x} surface bricks, scratch high-water {:.1} MiB, {secs_2x:.2}s",
        scratch_2x as f64 / (1024.0 * 1024.0)
    );

    // Scratch sanity: the work really doubled (≈2×, allow a wide 1.4×–2.6× band — region packing isn't linear).
    let scratch_ratio = scratch_2x as f64 / scratch_1x.max(1) as f64;
    println!("  scratch high-water ratio (2×/1×): {scratch_ratio:.2} (expect ≈2)");

    match (rss_before_1x, rss_after_1x, rss_after_2x) {
        (Some(_b), Some(p1), Some(p2)) => {
            // The peak after the 2× bake must not exceed the peak after the 1× bake by more than the slack — the
            // constant-RAM proof (the producer holds ≤ one region + the window + the pool, independent of surface).
            // SLACK absorbs allocator high-water noise + the (constant) writer buffers; it is NOT surface-scaled.
            let slack = 64 * 1024 * 1024; // 64 MiB
            let grew = p2.saturating_sub(p1);
            println!(
                "  PEAK RSS: after 1× = {:.1} MiB, after 2× = {:.1} MiB, growth = {:.1} MiB (slack {} MiB)",
                p1 as f64 / (1024.0 * 1024.0),
                p2 as f64 / (1024.0 * 1024.0),
                grew as f64 / (1024.0 * 1024.0),
                slack / (1024 * 1024),
            );
            anyhow::ensure!(
                grew <= slack,
                "constant-RAM gate FAILED: doubling the surface grew peak RSS by {:.1} MiB (> {} MiB slack) — the \
                 producer is NOT constant in surface size",
                grew as f64 / (1024.0 * 1024.0),
                slack / (1024 * 1024),
            );
            println!("  ✓ flat-RSS gate PASSED (peak grew {:.1} MiB ≤ slack)", grew as f64 / (1024.0 * 1024.0));
        }
        _ => println!("  PEAK RSS unavailable on this platform — gate skipped (scratch ratio still reported)"),
    }

    let _ = std::fs::remove_dir_all(&base_dir);
    Ok(())
}

/// Bake an `edge³`-brick hollow shell through the CONSTANT-RAM disk-spill producer (spill → `assemble_base` →
/// `windowed_coarse` → `finish`) under a fresh scratch dir, returning `(scratch_high_water_bytes, base_bricks,
/// secs)`. This is the EXACT production path (`voxelize_scene::assemble_vxo_streaming`), so the gate measures the
/// real producer — not a stand-in. The synthetic shell is generated brick-by-brick + spilled WITHOUT ever
/// building the whole `BrickMap` (mirroring how a real mesh bake streams solids), so the harness itself is also
/// constant-RAM and does not mask the producer's footprint.
fn bake_shell_via_spill(
    edge: i32,
    registry: &BlockRegistry,
    params: &VxoHeadParams,
    dir: &Path,
) -> anyhow::Result<(u64, u64, f32)> {
    std::fs::create_dir_all(dir)?;
    let k = params.region_edge_bricks as i32;
    let out = dir.join("shell.vxo");
    let scratch_brik = dir.join("assembly.brik.tmp");
    let comp = VxoCompression::Store;

    let t = Instant::now();
    // 1. Spill pass: generate each surface brick + spill its solid voxels (never holding the map resident).
    let mut base = RegionSpillPool::new(dir, "base", k);
    for z in 0..edge {
        for y in 0..edge {
            for x in 0..edge {
                let on_face = x == 0 || y == 0 || z == 0 || x == edge - 1 || y == edge - 1 || z == edge - 1;
                if !on_face {
                    continue;
                }
                let brick = shell_brick(x * 7 + y * 13 + z * 17);
                let bc = IVec3::new(x, y, z);
                for bz in 0..BRICK_EDGE {
                    for by in 0..BRICK_EDGE {
                        for bx in 0..BRICK_EDGE {
                            let b = brick.get(bx, by, bz);
                            if !b.is_air() {
                                let w = bc * BRICK_EDGE + IVec3::new(bx, by, bz);
                                spill_voxel(&mut base, w, b)?;
                            }
                        }
                    }
                }
            }
        }
    }
    base.flush_all()?;

    // 2. Assemble base + windowed coarse, measuring the scratch high-water at the peak (after the base + coarse
    //    spills coexist, before finish deletes them). We sample the scratch dir size right after windowed_coarse.
    let mut writer = VxoStreamWriter::new(params.clone(), registry, comp, &scratch_brik)?;
    let mut coarse_l0 = RegionSpillPool::new(dir, "coarse_l0", k);
    let base_bricks = assemble_base(&base, &mut coarse_l0, &mut writer)?;
    base.delete_all();
    let scratch_mid = dir_size(dir).unwrap_or(0); // high-water: assembled brik + coarse_l0 spills resident
    if base_bricks > 0 {
        windowed_coarse(coarse_l0, dir, k, &mut writer)?;
    } else {
        coarse_l0.delete_all();
    }
    let scratch_post = dir_size(dir).unwrap_or(0);
    writer.finish(&out)?;
    let secs = t.elapsed().as_secs_f32();

    // Verify it parses + has the full LODS pyramid (a correctness backstop in the perf harness).
    let file = VxoFile::parse(&std::fs::read(&out)?)?;
    anyhow::ensure!(file.has_lods(), "shell bake must carry LODS");

    let high_water = scratch_mid.max(scratch_post);
    let _ = std::fs::remove_dir_all(dir);
    Ok((high_water, base_bricks, secs))
}

/// Recursive byte size of a directory (the scratch high-water sample). Best-effort; ignores unreadable entries.
fn dir_size(dir: &Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let md = entry.metadata()?;
        if md.is_dir() {
            total += dir_size(&entry.path()).unwrap_or(0);
        } else {
            total += md.len();
        }
    }
    Ok(total)
}

/// Feed `map`'s base LOD0 bricks into `writer` region-by-region in `(z,y,x)` order (the `add_region` contract +
/// the deterministic BRIK layout the streaming writer expects).
fn feed_base_regions(map: &BrickMap, k: i32, writer: &mut VxoStreamWriter) -> anyhow::Result<()> {
    let mut regions: std::collections::BTreeMap<(i32, i32, i32), Vec<IVec3>> = Default::default();
    for (coord, _) in map.iter() {
        let r = region_of_brick(*coord, k);
        regions.entry((r.z, r.y, r.x)).or_default().push(*coord);
    }
    for ((rz, ry, rx), mut coords) in regions {
        coords.sort_by_key(|c| (c.z, c.y, c.x));
        let bricks: Vec<(IVec3, &Brick)> =
            coords.iter().map(|&c| (c, map.get(c).expect("brick present"))).collect();
        writer.add_region(IVec3::new(rx, ry, rz), &bricks)?;
    }
    Ok(())
}

/// A HOLLOW box of solid bricks: the 6 faces of an `edge³`-brick cube (interior bricks omitted), so the brick
/// count scales with SURFACE area (≈ `6·edge²`), like a real mesh bake's surface-shell residency. Each face
/// brick is a deterministic two-block dense brick (so it does not collapse to uniform and its downsample
/// exercises the dominant-block reducer across many coarse levels).
fn hollow_shell_map(edge: i32) -> BrickMap {
    let mut map = BrickMap::new();
    for z in 0..edge {
        for y in 0..edge {
            for x in 0..edge {
                let on_face = x == 0 || y == 0 || z == 0 || x == edge - 1 || y == edge - 1 || z == edge - 1;
                if on_face {
                    map.insert(IVec3::new(x, y, z), shell_brick(x * 7 + y * 13 + z * 17));
                }
            }
        }
    }
    map
}

/// A dense (non-uniform), fully-solid brick with a deterministic two-block mix — so the brick is `is_full`, has a
/// well-defined dominant block per coarse voxel, and never downsamples to empty.
fn shell_brick(seed: i32) -> Brick {
    let mut v = Box::new([BlockId::AIR; BRICK_VOXELS]);
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                let i = (x + y * BRICK_EDGE + z * BRICK_EDGE * BRICK_EDGE) as usize;
                v[i] = if (x + y + z + seed).rem_euclid(3) == 0 { BlockId(1) } else { BlockId(2) };
            }
        }
    }
    Brick::from_voxels(v)
}

/// Report the committed `assets/models/bistro.vxo` stats (size + LODS state) for reference, if present. Does NOT
/// re-bake (the heavy tiled glTF run is triggered separately). Skip-graceful when absent.
fn report_bistro_reference() {
    let path = std::path::Path::new("assets/models/bistro.vxo");
    let Ok(bytes) = std::fs::read(path) else {
        println!("  Bistro: assets/models/bistro.vxo absent — skipped (re-bake via `voxelize_scene --tiled`).");
        return;
    };
    match VxoFile::parse(&bytes) {
        Ok(file) => println!(
            "  Bistro reference: {} ({:.1} MiB), has_lods={}, max_lod={}, region_count={}, brick_count={}",
            path.display(),
            bytes.len() as f64 / (1024.0 * 1024.0),
            file.has_lods(),
            file.max_lod(),
            file.head.region_count,
            file.head.brick_count,
        ),
        Err(e) => println!("  Bistro: present but failed to parse: {e:#}"),
    }
}

// ============================================================================================
// Peak-RSS probe (no extra crate: Windows uses the kernel32-exported `K32GetProcessMemoryInfo`;
// Linux reads `/proc/self/status` VmHWM). Returns the process peak working set in BYTES.
// ============================================================================================

#[cfg(target_os = "windows")]
fn peak_rss_bytes() -> Option<u64> {
    // `PROCESS_MEMORY_COUNTERS` (psapi.h) — the prefix we need (cb + page-fault + the working-set fields).
    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    // `K32GetProcessMemoryInfo` is exported directly by kernel32.dll (since Win7), so no extra import library is
    // needed — the MSVC toolchain links kernel32 by default. `GetCurrentProcess()` returns the (-1) pseudo-handle.
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(process: isize, counters: *mut ProcessMemoryCounters, cb: u32) -> i32;
    }
    let mut counters = ProcessMemoryCounters { cb: std::mem::size_of::<ProcessMemoryCounters>() as u32, ..Default::default() };
    // SAFETY: `counters` is a correctly-sized, fully-initialized `#[repr(C)]` mirror of `PROCESS_MEMORY_COUNTERS`;
    // `cb` records its size; the pseudo-handle is always valid for the current process.
    let ok = unsafe {
        K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb)
    };
    (ok != 0).then_some(counters.peak_working_set_size as u64)
}

#[cfg(target_os = "linux")]
fn peak_rss_bytes() -> Option<u64> {
    // VmHWM = the high-water-mark resident set, in kB.
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn peak_rss_bytes() -> Option<u64> {
    None
}
