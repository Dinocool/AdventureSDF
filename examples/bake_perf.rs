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

use std::time::Instant;

use bevy::math::IVec3;

use adventure::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, BrickMap};
use adventure::voxel::palette::{BlockId, BlockRegistry};
use adventure::voxel::vxo::{
    VxoCompression, VxoFile, VxoHeadParams, VxoStreamWriter, build_coarse_pyramid, drive_coarse_lods,
    region_of_brick,
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

    // Bistro reference (skip-graceful) — report the committed asset's stats, never re-bake it here.
    report_bistro_reference();

    let _ = std::fs::remove_dir_all(&scratch_dir);
    Ok(())
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
