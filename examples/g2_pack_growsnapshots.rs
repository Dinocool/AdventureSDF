//! **TIER-1 GROW-SNAPSHOT BENCHMARK — `ResidentPacker` arena pre-size, BEFORE vs AFTER.**
//!
//! The gallery LIVE freeze is the runtime GPU-buffer pack: a trace showed `vox_pack_update` (incremental pack)
//! 4.16 s/load and `vox_pack_snapshot` 317 ms (n=2), the worst single frame 482 ms = a slab-arena GROW forcing a
//! full O(capacity) re-snapshot MID-LOAD. When the A4.4 `SlabArena` overflows its committed GPU buffer during a
//! load it grows + re-allocates, forcing a `StreamSnapshot` (the ~200 ms `vox_pack_snapshot`). Pre-sizing the
//! arena to the resident cap up front means it never grows during a normal load → those snapshot spikes vanish.
//!
//! This rig drives the PRODUCTION `ResidentPacker::update` → snapshot/delta loop the EXACT way
//! `raytrace::stream_voxel_rt_residency` does (the `vox_repack` block): a streamed load arrives in
//! `max_bricks_per_frame` batches; each batch is `update`d, and the upload is a `StreamSnapshot` on the first
//! pack OR a `grew()` (the GROW we are eliminating), else a `Delta`. We count the grow-snapshots and time each
//! pack stage for an UN-pre-sized packer (`new_unreserved`, the BEFORE) vs the pre-sized one (`new`, the AFTER).
//!
//! Pure CPU — no GPU device. The CPU pack + arena path is exactly the cost the freeze pays on the main schedule;
//! the GPU buffer (re)allocation a grow triggers is the *consequence* of the grow-snapshot this rig counts.
//! Run:  cargo run --release --example g2_pack_growsnapshots
//!
//! NOT a shipped tool — a de-risk/regression rig for the Tier-1 pre-size fix.

use std::time::Instant;

use bevy::math::IVec3;

use adventure::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick};
use adventure::voxel::gpu::ResidentBrick;
use adventure::voxel::incremental::ResidentPacker;
use adventure::voxel::palette::{BlockId, BlockRegistry};

/// A patterned brick with a handful of distinct block ids so it packs DENSE (not uniform) with a representative
/// per-brick palette — exercises the index + palette slab arenas the way real terrain/architectural bricks do.
fn patterned_brick(seed: i32, n_ids: u16) -> Brick {
    let mut v = Box::new([BlockId::AIR; BRICK_VOXELS]);
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                let h = (x.wrapping_mul(73) + y.wrapping_mul(19) + z.wrapping_mul(7) + seed).rem_euclid(5);
                let idx = (x + y * BRICK_EDGE + z * BRICK_EDGE * BRICK_EDGE) as usize;
                // ~3/5 cells solid (a surface-ish density); spread across a few ids for a non-trivial palette.
                v[idx] = if h < 3 { BlockId(1 + ((x + y + z + seed).rem_euclid(n_ids as i32)) as u16) } else { BlockId::AIR };
            }
        }
    }
    Brick::from_voxels(v)
}

/// A streamed-load corpus: an `edge³` cube of patterned LOD0 bricks (a stand-in for a surface shell). Each is
/// dense with a few palette ids. Returns the owned `(coord, brick)` pairs in a deterministic order.
fn build_corpus(edge: i32, n_ids: u16) -> Vec<(IVec3, Brick)> {
    let mut out = Vec::with_capacity((edge * edge * edge) as usize);
    for z in 0..edge {
        for y in 0..edge {
            for x in 0..edge {
                out.push((IVec3::new(x, y, z), patterned_brick(x * 31 + y * 17 + z * 11, n_ids)));
            }
        }
    }
    out
}

struct Stats {
    grow_snapshots: u32,
    epoch_snapshots: u32,
    deltas: u32,
    total_update_ms: f64,
    total_snapshot_ms: f64,
    worst_frame_ms: f64,
    worst_frame_kind: &'static str,
    last_index_words: usize,
    last_palette_words: usize,
}

/// Drive the corpus through `packer` in `batch`-sized increments, mirroring the production
/// `update`→{StreamSnapshot | grow-snapshot | Delta} loop. Returns the per-stage stats incl. the grow-snapshot
/// count (the headline).
fn drive(packer: &mut ResidentPacker, corpus: &[(IVec3, Brick)], reg: &BlockRegistry, batch: usize) -> Stats {
    let mut st = Stats {
        grow_snapshots: 0,
        epoch_snapshots: 0,
        deltas: 0,
        total_update_ms: 0.0,
        total_snapshot_ms: 0.0,
        worst_frame_ms: 0.0,
        worst_frame_kind: "none",
        last_index_words: 0,
        last_palette_words: 0,
    };
    let mut resident: Vec<(IVec3, Brick)> = Vec::with_capacity(corpus.len());
    let mut epoch_snapshotted = false;
    let mut next = 0usize;
    while next < corpus.len() {
        let end = (next + batch).min(corpus.len());
        resident.extend_from_slice(&corpus[next..end]);
        next = end;

        let entries: Vec<ResidentBrick<'_>> =
            resident.iter().map(|(c, b)| ResidentBrick { coord: *c, brick: b, lod: 0 }).collect();

        let t = Instant::now();
        let delta = packer.update(&entries, reg.len() as u32);
        let update_ms = t.elapsed().as_secs_f64() * 1e3;
        st.total_update_ms += update_ms;

        // The EXACT snapshot-vs-delta decision `stream_voxel_rt_residency` makes (`!epoch_snapshotted || grew()`).
        let mut frame_ms = update_ms;
        if !epoch_snapshotted || packer.grew() {
            let grew = packer.grew();
            let ts = Instant::now();
            let snap = packer.snapshot_buffers(reg);
            let snap_ms = ts.elapsed().as_secs_f64() * 1e3;
            st.total_snapshot_ms += snap_ms;
            frame_ms += snap_ms;
            st.last_index_words = snap.indices.len();
            st.last_palette_words = snap.brick_palettes.len();
            if grew && epoch_snapshotted {
                st.grow_snapshots += 1; // the spike we are eliminating
            } else {
                st.epoch_snapshots += 1; // the once-per-epoch seed (unavoidable)
            }
            epoch_snapshotted = true;
        } else if !delta.is_empty() {
            st.deltas += 1;
        }
        if frame_ms > st.worst_frame_ms {
            st.worst_frame_ms = frame_ms;
            st.worst_frame_kind =
                if !epoch_snapshotted { "epoch-snapshot" } else if packer.grew() { "GROW-snapshot" } else { "delta" };
        }
    }
    st
}

fn report(label: &str, st: &Stats, resident: usize) {
    println!("\n-- {label} --");
    println!("  resident bricks       : {resident}");
    println!("  GROW-snapshots        : {}  ⇐ the mid-load hitch (target: 0)", st.grow_snapshots);
    println!("  epoch snapshots       : {}  (the unavoidable once-per-epoch seed)", st.epoch_snapshots);
    println!("  deltas                : {}", st.deltas);
    println!("  Σ vox_pack_update     : {:.1} ms", st.total_update_ms);
    println!("  Σ vox_pack_snapshot   : {:.1} ms", st.total_snapshot_ms);
    println!("  worst single frame    : {:.1} ms ({})", st.worst_frame_ms, st.worst_frame_kind);
    println!(
        "  final arena words     : {} index ({:.1} MB), {} palette ({:.1} MB)",
        st.last_index_words,
        st.last_index_words as f64 * 4.0 / 1e6,
        st.last_palette_words,
        st.last_palette_words as f64 * 4.0 / 1e6,
    );
}

fn main() {
    // A representative streamed load. The cap is the production default (400k); the corpus is a dense cube that
    // streams in over many batches (so the un-pre-sized arena overflows + re-snapshots repeatedly, the BEFORE).
    let cap = 400_000u32;
    let batch = 256usize; // StreamingConfig::max_bricks_per_frame default
    let edge = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(40i32); // 40³ = 64k dense bricks
    let n_ids = 6u16;

    println!("========== TIER-1 GROW-SNAPSHOT BENCHMARK (ResidentPacker arena pre-size) ==========");
    println!("cap {cap} · batch {batch}/frame · corpus {edge}³ = {} dense bricks · {n_ids} palette ids", edge * edge * edge);

    let reg = BlockRegistry::cornell();
    let corpus = build_corpus(edge, n_ids);
    let resident = corpus.len();

    // BEFORE — un-pre-sized: the arena grows from empty, re-snapshotting on every overflow during the load.
    let mut before_packer = ResidentPacker::new_unreserved(cap);
    let before = drive(&mut before_packer, &corpus, &reg, batch);
    report("BEFORE (un-pre-sized — grows from empty)", &before, resident);

    // AFTER — pre-sized: the arena is reserved to the cap up front, so the load fits the first snapshot.
    let mut after_packer = ResidentPacker::new(cap);
    let after = drive(&mut after_packer, &corpus, &reg, batch);
    report("AFTER (pre-sized to the resident cap)", &after, resident);

    println!("\n========== VERDICT ==========");
    println!("GROW-snapshots  BEFORE {} → AFTER {}", before.grow_snapshots, after.grow_snapshots);
    println!(
        "Σ snapshot ms   BEFORE {:.1} → AFTER {:.1}  (Δ {:.1} ms saved)",
        before.total_snapshot_ms,
        after.total_snapshot_ms,
        before.total_snapshot_ms - after.total_snapshot_ms,
    );
    println!("worst frame ms  BEFORE {:.1} → AFTER {:.1}", before.worst_frame_ms, after.worst_frame_ms);
    if after.grow_snapshots == 0 {
        println!("PASS: zero grow-snapshots after pre-sizing — the mid-load re-snapshot spikes are eliminated.");
    } else {
        println!("NOTE: {} grow-snapshots remain after pre-sizing (corpus denser than the representative reserve).", after.grow_snapshots);
    }
    println!("=====================================================================================");
}
