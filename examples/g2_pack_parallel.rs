//! **TIER-2 PARALLEL-PACK BENCHMARK — `ResidentPacker::update` Phase-1 SERIAL vs PARALLEL.**
//!
//! After the Tier-1 grow-snapshot fix, the residual gallery cold-load freeze isolated to `vox_pack_update`: the
//! CPU `ResidentPacker` incremental pack of the streaming shell, ~282 ms/call × ~12 = 3.4 s/load. The expensive
//! part is the per-dirty-brick `pack_one` halo-fill + `encode_paletted`, which are PURE functions of the shared
//! immutable `by_key` map. Tier-2 splits `update` into PHASE 1 (the pure pack, now `par_iter` across cores) and
//! PHASE 2 (the serial, order-dependent arena alloc/free + `delta.changed` fold — UNCHANGED). The emitted
//! `RepackDelta` is byte-identical; this rig measures the speed-up.
//!
//! It drives the PRODUCTION `update`→snapshot/delta loop the EXACT way `g2_pack_growsnapshots` does (a streamed
//! cold load arriving in `max_bricks_per_frame` batches), once via `update_serial` (Phase 1 forced serial =
//! BEFORE) and once via `update` (Phase 1 parallel = AFTER), and reports per-call + total `vox_pack_update` ms.
//! It also ASSERTS the two paths emit identical deltas (changed-slot count + bytes) so the speed-up is free.
//!
//! Pure CPU — no GPU device. Run:  cargo run --release --example g2_pack_parallel [edge] [batch]

use std::time::Instant;

use bevy::math::IVec3;

use adventure::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick};
use adventure::voxel::gpu::ResidentBrick;
use adventure::voxel::incremental::{RepackDelta, ResidentPacker};
use adventure::voxel::palette::{BlockId, BlockRegistry};

/// A patterned brick with a handful of distinct block ids so it packs DENSE (not uniform) with a representative
/// per-brick palette — exercises the `pack_one` halo-fill + `encode_paletted` the way real bricks do.
fn patterned_brick(seed: i32, n_ids: u16) -> Brick {
    let mut v = Box::new([BlockId::AIR; BRICK_VOXELS]);
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                let h = (x.wrapping_mul(73) + y.wrapping_mul(19) + z.wrapping_mul(7) + seed).rem_euclid(5);
                let idx = (x + y * BRICK_EDGE + z * BRICK_EDGE * BRICK_EDGE) as usize;
                v[idx] = if h < 3 { BlockId(1 + ((x + y + z + seed).rem_euclid(n_ids as i32)) as u16) } else { BlockId::AIR };
            }
        }
    }
    Brick::from_voxels(v)
}

/// A streamed-load corpus: an `edge³` cube of patterned LOD0 bricks (a stand-in for a surface shell).
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

/// A reproducible signature of a delta's emitted bytes — the byte-identity oracle the serial/parallel paths must
/// match (the patch must be IDENTICAL, not merely same-length). Slot + meta/aabb raw bytes + index/palette blocks.
fn delta_signature(d: &RepackDelta) -> Vec<u8> {
    let mut sig = Vec::new();
    sig.extend_from_slice(&(d.changed.len() as u32).to_le_bytes());
    for c in &d.changed {
        sig.extend_from_slice(&c.slot.to_le_bytes());
        sig.extend_from_slice(bytemuck::bytes_of(&c.meta));
        sig.extend_from_slice(bytemuck::bytes_of(&c.aabb));
        sig.extend_from_slice(&c.index_word_offset.to_le_bytes());
        sig.extend_from_slice(&c.palette_word_offset.to_le_bytes());
        if let Some(ix) = &c.index {
            for &w in ix {
                sig.extend_from_slice(&w.to_le_bytes());
            }
        }
        if let Some(pal) = &c.palette {
            for &w in pal {
                sig.extend_from_slice(&w.to_le_bytes());
            }
        }
    }
    let mut freed = d.freed.clone();
    freed.sort_unstable();
    for f in freed {
        sig.extend_from_slice(&f.to_le_bytes());
    }
    sig
}

struct Stats {
    calls: u32,
    total_update_ms: f64,
    worst_call_ms: f64,
    per_call_ms: Vec<f64>,
    sigs: Vec<Vec<u8>>,
}

/// Drive the corpus through `packer` in `batch` increments. `parallel` picks `update` (true) or `update_serial`
/// (false). Records the per-call `update` ms + a byte signature of every delta (for the A/B identity assert).
fn drive(corpus: &[(IVec3, Brick)], reg: &BlockRegistry, batch: usize, cap: u32, parallel: bool) -> Stats {
    let mut packer = ResidentPacker::new(cap);
    let mut st = Stats { calls: 0, total_update_ms: 0.0, worst_call_ms: 0.0, per_call_ms: Vec::new(), sigs: Vec::new() };
    let mut resident: Vec<(IVec3, Brick)> = Vec::with_capacity(corpus.len());
    let mut next = 0usize;
    while next < corpus.len() {
        let end = (next + batch).min(corpus.len());
        resident.extend_from_slice(&corpus[next..end]);
        next = end;

        let entries: Vec<ResidentBrick<'_>> =
            resident.iter().map(|(c, b)| ResidentBrick { coord: *c, brick: b, lod: 0 }).collect();

        let t = Instant::now();
        let delta = if parallel {
            packer.update(&entries, reg.len() as u32)
        } else {
            packer.update_serial(&entries, reg.len() as u32)
        };
        let ms = t.elapsed().as_secs_f64() * 1e3;
        st.total_update_ms += ms;
        st.worst_call_ms = st.worst_call_ms.max(ms);
        st.per_call_ms.push(ms);
        st.sigs.push(delta_signature(&delta));
        st.calls += 1;
    }
    st
}

fn report(label: &str, st: &Stats) {
    let mean = if st.calls > 0 { st.total_update_ms / st.calls as f64 } else { 0.0 };
    println!("\n-- {label} --");
    println!("  update calls          : {}", st.calls);
    println!("  Σ vox_pack_update     : {:.1} ms", st.total_update_ms);
    println!("  mean per call         : {:.1} ms", mean);
    println!("  worst single call     : {:.1} ms", st.worst_call_ms);
}

fn main() {
    let cap = 400_000u32;
    let batch = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(256usize);
    let edge = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(40i32); // 40³ = 64k dense bricks
    let n_ids = 6u16;
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    println!("========== TIER-2 PARALLEL-PACK BENCHMARK (ResidentPacker::update Phase-1) ==========");
    println!("cap {cap} · batch {batch}/frame · corpus {edge}³ = {} dense bricks · {n_ids} ids · {cores} cores", edge * edge * edge);

    let reg = BlockRegistry::cornell();
    let corpus = build_corpus(edge, n_ids);

    // BEFORE — Phase 1 forced serial (the pre-Tier-2 cost).
    let before = drive(&corpus, &reg, batch, cap, false);
    report("BEFORE (Phase 1 serial)", &before);

    // AFTER — Phase 1 parallel across cores.
    let after = drive(&corpus, &reg, batch, cap, true);
    report("AFTER (Phase 1 parallel)", &after);

    // BYTE-IDENTITY GATE — every delta the two paths emitted must be byte-identical (the parallel path may NOT
    // change the emitted bytes or slot assignment — Phase 2's alloc/emit stayed serial for exactly this reason).
    assert_eq!(before.sigs.len(), after.sigs.len(), "serial/parallel produced a different number of updates");
    let mismatches = before.sigs.iter().zip(&after.sigs).filter(|(a, b)| a != b).count();
    println!("\n========== BYTE-IDENTITY ==========");
    if mismatches == 0 {
        println!("PASS: all {} per-batch deltas byte-identical (serial == parallel).", before.sigs.len());
    } else {
        println!("FAIL: {mismatches}/{} deltas differ between serial and parallel!", before.sigs.len());
    }
    assert_eq!(mismatches, 0, "parallel pack changed the emitted bytes — byte-identity violated");

    println!("\n========== VERDICT ==========");
    let s_mean = before.total_update_ms / before.calls.max(1) as f64;
    let p_mean = after.total_update_ms / after.calls.max(1) as f64;
    println!("Σ vox_pack_update  SERIAL {:.1} ms → PARALLEL {:.1} ms", before.total_update_ms, after.total_update_ms);
    println!("mean per call      SERIAL {:.1} ms → PARALLEL {:.1} ms", s_mean, p_mean);
    println!("worst call         SERIAL {:.1} ms → PARALLEL {:.1} ms", before.worst_call_ms, after.worst_call_ms);
    println!("speed-up           {:.2}× total · {:.2}× mean-per-call  ({cores} cores)", before.total_update_ms / after.total_update_ms.max(1e-9), s_mean / p_mean.max(1e-9));
    println!("=====================================================================================");
}
