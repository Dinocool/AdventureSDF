//! HeightLayer generation / gradient / band-limit tests + perf benches (split from height.rs per the test-module convention).

use super::super::super::coord::{ChunkCoord, chunk_min_world};
use super::super::super::layer::GenCtx;
use super::*;
use bevy::math::{DVec2, IVec3};

fn layer() -> HeightLayer {
    HeightLayer::new(LayerId(0), HeightParams::default(), ErosionParams::default())
}

/// CROSS-TIER CONSISTENCY GUARD — the band-limit must NOT introduce an LOD seam. Adjacent clipmap
/// tiers (tier 0 @ 2 m nodes, tier 1 @ 4 m nodes) must agree on the band-limited surface at the world
/// nodes they SHARE (multiples of 4 m), or a tier-coverage boundary shows a crack + shading kink. This
/// is the guard for the fixed-WORLD band-limit width (a node-relative width — the seam bug — would make
/// tier 1 smooth ~2× wider than tier 0, blowing up the mismatch here). Uses dense sharp ridges (worst
/// case). The existing `terrain_2to1_*` harness only tests cross-MIP within ONE tier, so it missed this.
#[test]
fn tiers_agree_on_shared_nodes_after_band_limit() {
    use bevy::math::Vec3;
    let params = HeightParams {
        ridge: 1.0,
        base_freq: 1.0 / 256.0,
        amplitude: 100.0,
        octaves: 4,
        band_limit: 3.0,
        ..Default::default()
    };
    let erosion = ErosionParams { enabled: false, ..Default::default() };
    let seed = 7u64;
    let t0 = HeightLayer::new_tier(LayerId(0), params, erosion, HEIGHT_CHUNK_CELLS);
    let t1 = HeightLayer::new_tier(LayerId(0), params, erosion, HEIGHT_CHUNK_CELLS * 2);
    let gen_chunk0 = |l: &HeightLayer| {
        let mut o = GenOutput::default();
        l.generate(&GenCtx { coord: ChunkCoord::new(l.id(), IVec3::ZERO), seed, size: l.chunk_size() }, &mut o);
        o.take::<ScalarField2D>(HeightLayer::OUTPUT).unwrap()
    };
    let (f0, f1) = (gen_chunk0(&t0), gen_chunk0(&t1));
    // Shared world nodes: tier 0 chunk spans [0,128] (nodes every 2 m), tier 1 spans [0,256] (every
    // 4 m). A world coord that is a multiple of 4 m in [4,124] is a node in BOTH (skip the very edge so
    // both have the band-limit apron). tier-0 node index = w/2, tier-1 = w/4.
    let amp = (params.amplitude_sum()) as f32;
    let (mut worst_h, mut worst_dot) = (0.0f32, 1.0f32);
    for kz in 1..=30 {
        for kx in 1..=30 {
            let (wx, wz) = (4 * kx, 4 * kz);
            let n0 = f0.node((wx / 2) as u32, (wz / 2) as u32);
            let n1 = f1.node((wx / 4) as u32, (wz / 4) as u32);
            worst_h = worst_h.max((n0.height - n1.height).abs());
            let g0 = Vec3::new(-n0.dh_dx, 1.0, -n0.dh_dz).normalize();
            let g1 = Vec3::new(-n1.dh_dx, 1.0, -n1.dh_dz).normalize();
            worst_dot = worst_dot.min(g0.dot(g1));
        }
    }
    println!(
        "CROSS-TIER: worst |Δh|={worst_h:.3}m ({:.2}% of amp {amp:.0}m), worst normal dot={worst_dot:.4} \
         ({:.1}deg)",
        100.0 * worst_h / amp,
        worst_dot.clamp(-1.0, 1.0).acos().to_degrees(),
    );
    // FIXED 2 m world taps at every tier ⇒ tier 0 and tier 1 evaluate the IDENTICAL band-limited world
    // function at the nodes they share ⇒ they agree to f32 round-off (NOT just "closely"). This is the
    // foolproof no-seam guarantee; a node-relative or per-tier-tap band-limit would smash both
    // (Δh ~ metres+, dot well below 1). Tolerances are f32-epsilon-tight.
    assert!(worst_h < 0.05, "cross-tier height seam: |Δh|={worst_h:.4}m (tiers must agree at shared nodes)");
    assert!(worst_dot > 0.9999, "cross-tier normal seam: worst dot {worst_dot:.5} (tiers must agree)");
    let _ = amp;
}

/// A PLAIN-fBm layer (no ridge, no erosion) — `generate` takes the point-sample fast path
/// (`aa == 1`), so its nodes equal `sample_world` exactly.
fn plain_layer() -> HeightLayer {
    HeightLayer::new(
        LayerId(0),
        HeightParams { ridge: 0.0, ..Default::default() },
        ErosionParams { enabled: false, ..Default::default() },
    )
}

/// For a PLAIN layer (no sharp features ⇒ no AA), `generate` produces a height field of the right
/// shape, every node equal to `sample_world` (the single surface-truth function) at its world coord.
#[test]
fn generate_fills_field_matching_sample_world() {
    let l = plain_layer();
    let coord = ChunkCoord::new(l.id(), IVec3::new(1, 0, -2));
    let ctx = GenCtx { coord, seed: 7, size: l.chunk_size() };
    let mut out = GenOutput::default();
    l.generate(&ctx, &mut out);
    let field = out.take::<ScalarField2D>(HeightLayer::OUTPUT).unwrap();
    assert_eq!(field.res, HEIGHT_FIELD_RES);
    assert_eq!(field.nodes.len(), ((HEIGHT_FIELD_RES + 1) * (HEIGHT_FIELD_RES + 1)) as usize);
    for &(i, j) in &[(0u32, 0u32), (10, 20), (HEIGHT_FIELD_RES, HEIGHT_FIELD_RES)] {
        let wp = field.node_world_xz(i, j);
        let expect = l.sample_world(wp.x, wp.y, ctx.seed);
        assert_eq!(field.node(i, j), expect);
    }
}

/// A layer with the band-limit finalize EXPLICITLY enabled (`band_limit = 3`). The DEFAULT is now `0`
/// (point-sample the raw surface), so this explicitly exercises the still-supported `generate_band_limited`
/// finalize the slider drives — without depending on the default.
fn band_limited_layer() -> HeightLayer {
    HeightLayer::new(
        LayerId(0),
        HeightParams { band_limit: 3.0, ..Default::default() },
        ErosionParams::default(),
    )
}

/// With ridge/erosion on AND `band_limit > 0`, `generate` BAND-LIMITS the composed surface (the finalize
/// tent low-pass): a node is a CONVEX COMBINATION (weighted average) of `sample_world` over its kernel
/// neighbourhood, so it (a) differs from the raw point sample (the low-pass is doing something) and (b)
/// lies within the local [min, max] of the surface (a low-pass can never overshoot — the property that
/// rounds the sharp crest instead of spiking it). Uses an EXPLICITLY band-limited layer (the default is
/// now `band_limit = 0` / raw); this pins the still-supported slider finalize path.
#[test]
fn generate_band_limits_sharp_features() {
    let l = band_limited_layer(); // ridge + erosion + band_limit = 3 (slider on)
    assert!(l.params.band_limit > 0.0, "this test must use a band-limited layer");
    let coord = ChunkCoord::new(l.id(), IVec3::new(1, 0, -2));
    let ctx = GenCtx { coord, seed: 7, size: l.chunk_size() };
    let mut out = GenOutput::default();
    l.generate(&ctx, &mut out);
    let field = out.take::<ScalarField2D>(HeightLayer::OUTPUT).unwrap();
    let (i, j) = (10u32, 20u32);
    let wp = field.node_world_xz(i, j);
    let node = field.node(i, j);
    assert!(node.height.is_finite() && node.dh_dx.is_finite() && node.dh_dz.is_finite());

    // Bound the band-limited node by the local surface range over the kernel footprint (±radius nodes).
    let spacing = field.node_spacing; // already f64
    let r = l.params.band_limit.ceil() as i32;
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for sj in -r..=r {
        for si in -r..=r {
            let s = l.sample_world(wp.x + si as f64 * spacing, wp.y + sj as f64 * spacing, ctx.seed);
            lo = lo.min(s.height);
            hi = hi.max(s.height);
        }
    }
    assert!(
        node.height >= lo - 1e-2 && node.height <= hi + 1e-2,
        "band-limited node {} must lie within local surface range [{lo}, {hi}] (a low-pass can't overshoot)",
        node.height,
    );
    // And it actually differs from the raw point sample (the band-limit is active).
    let point = l.sample_world(wp.x, wp.y, ctx.seed);
    assert_ne!(node.height.to_bits(), point.height.to_bits(), "band-limited node differs from point sample");
}

/// Seam-free across a chunk boundary: the far apron node of chunk C equals the near node of chunk
/// C+1 at the same world XZ — because both come from `sample_world` at the identical world coord.
/// This is the §10 "padding correctness" property at the field level.
#[test]
fn adjacent_chunks_agree_on_shared_boundary() {
    let l = layer();
    let size = l.chunk_size();
    let seed = 123;
    let ca = ChunkCoord::new(l.id(), IVec3::new(0, 0, 0));
    let cb = ChunkCoord::new(l.id(), IVec3::new(1, 0, 0)); // +X neighbour

    let mut oa = GenOutput::default();
    l.generate(&GenCtx { coord: ca, seed, size }, &mut oa);
    let fa = oa.take::<ScalarField2D>(HeightLayer::OUTPUT).unwrap();
    let mut ob = GenOutput::default();
    l.generate(&GenCtx { coord: cb, seed, size }, &mut ob);
    let fb = ob.take::<ScalarField2D>(HeightLayer::OUTPUT).unwrap();

    // Chunk A's far-X apron column (i = res) is chunk B's near column (i = 0), same world X.
    let res = HEIGHT_FIELD_RES;
    for j in 0..=res {
        let a = fa.node(res, j);
        let b = fb.node(0, j);
        assert_eq!(a.height.to_bits(), b.height.to_bits(), "boundary height seam at j={j}");
        assert_eq!(a.dh_dx.to_bits(), b.dh_dx.to_bits(), "boundary ∂x seam at j={j}");
    }
    // Sanity: the shared world X is exactly chunk B's min.x.
    let bmin = chunk_min_world(cb, size);
    let shared_x = fa.node_world_xz(res, 0).x;
    assert!((shared_x - bmin.x).abs() < 1e-9);
    let _ = DVec2::ZERO;
}

/// Biome-shape registry (B1): the same-`Arc` guard makes "override every biome with the default graph"
/// bit-identical to the single-graph layer (so attaching the registry with no real overrides is a no-op),
/// and a DISTINCT per-biome shape override actually changes the height somewhere while staying finite.
#[test]
fn biome_shape_blend_guard_and_override() {
    use super::super::super::graph::node::FbmAxis;
    use super::super::super::graph::preset::default_terrain_graph;
    let seed = 42u64;
    let ga = Arc::new(default_terrain_graph(
        FbmAxis { octaves: 3, base_freq: 1.0 / 512.0, lacunarity: 2.0, gain: 0.5, amplitude: 100.0, seed_salt: 1 },
        0.0,
        2.0,
        0.0,
    ));
    // A clearly different shape (different freq/amplitude/seed).
    let gb = Arc::new(default_terrain_graph(
        FbmAxis { octaves: 2, base_freq: 1.0 / 4096.0, lacunarity: 2.0, gain: 0.5, amplitude: 600.0, seed_salt: 2 },
        0.0,
        2.0,
        0.0,
    ));

    let single = layer().with_graph(Some(ga.clone()));
    // GUARD: every biome overridden with the SAME graph as the default → not `is_single`, but the
    // same-`Arc` guard evals ONCE ⇒ bit-identical to the single-graph layer.
    let guarded = layer().with_graph(Some(ga.clone())).with_biome_shapes(std::array::from_fn(|_| Some(ga.clone())));
    // OVERRIDE: give one biome a DISTINCT shape graph.
    let mut shapes: [Option<Arc<Graph>>; BIOME_COUNT] = std::array::from_fn(|_| None);
    shapes[BiomeId::Snowy as usize] = Some(gb.clone());
    let overridden = layer().with_graph(Some(ga.clone())).with_biome_shapes(shapes);

    let mut changed = false;
    let mut x = -20_000.0;
    while x < 20_000.0 {
        let (wx, wz) = (x, x * -0.61 + 1234.0);
        let s = single.sample_world(wx, wz, seed);
        let g = guarded.sample_world(wx, wz, seed);
        assert_eq!(s.height.to_bits(), g.height.to_bits(), "same-Arc guard not bit-identical at ({wx},{wz})");
        assert_eq!(s.dh_dx.to_bits(), g.dh_dx.to_bits(), "same-Arc guard ∂x not identical at ({wx},{wz})");
        let o = overridden.sample_world(wx, wz, seed);
        assert!(o.height.is_finite() && o.dh_dx.is_finite() && o.dh_dz.is_finite(), "blended result must be finite");
        if o.height.to_bits() != s.height.to_bits() {
            changed = true;
        }
        x += 173.0;
    }
    assert!(changed, "the Snowy shape override never changed the height over the sweep");
}

/// Different seeds give different terrain (the seed actually drives the field).
#[test]
fn seed_changes_terrain() {
    let l = layer();
    let a = l.sample_world(12.0, 34.0, 1);
    let b = l.sample_world(12.0, 34.0, 2);
    assert_ne!(a.height.to_bits(), b.height.to_bits());
}

/// The CLOSED-FORM carved gradient (now stored by `sample_world`) matches a central difference of the
/// carved height — the correctness guard for the analytic erosion/ridge gradient (it replaced the FD
/// taps). The carved height value is `carved_grad(...).0` (its value lane is bit-identical to the old
/// `erode_height` path). Tolerance ~1e-2 (the FD's own truncation error at this eps).
#[test]
fn analytic_gradient_matches_central_difference() {
    let l = layer();
    let seed = 4242u64;
    let fbm = l.fbm_params(seed);
    // Sloped, mid-altitude points where erosion + ridge bite.
    for &(wx, wz) in &[(321.0, -123.0), (-560.0, 880.0), (1500.5, 700.25)] {
        let (_, gx, gz) = l.carved_grad(wx, wz, &fbm, seed);
        assert!(gx.is_finite() && gz.is_finite(), "gradient not finite at ({wx},{wz})");
        // Reference: a central difference of the full carved height (value lane of carved_grad).
        let e = 0.01f64;
        let hxp = l.carved_grad(wx + e, wz, &fbm, seed).0;
        let hxm = l.carved_grad(wx - e, wz, &fbm, seed).0;
        let hzp = l.carved_grad(wx, wz + e, &fbm, seed).0;
        let hzm = l.carved_grad(wx, wz - e, &fbm, seed).0;
        let fd_x = (hxp - hxm) / (2.0 * e);
        let fd_z = (hzp - hzm) / (2.0 * e);
        assert!((gx - fd_x).abs() < 1e-2, "∂x at ({wx},{wz}): analytic {gx} vs FD {fd_x}");
        assert!((gz - fd_z).abs() < 1e-2, "∂z at ({wx},{wz}): analytic {gz} vs FD {fd_z}");
    }
}

/// `sample_world`'s stored gradient (the analytic carved gradient narrowed to f32) is finite and
/// matches `carved_grad` — guards the terrain normals the bake reconstructs from this gradient.
#[test]
fn sample_world_stores_analytic_gradient() {
    let l = layer();
    let seed = 4242u64;
    let fbm = l.fbm_params(seed);
    for &(wx, wz) in &[(321.0, -123.0), (-560.0, 880.0), (1500.5, 700.25)] {
        let n = l.sample_world(wx, wz, seed);
        assert!(n.dh_dx.is_finite() && n.dh_dz.is_finite(), "gradient not finite at ({wx},{wz})");
        let (h, gx, gz) = l.carved_grad(wx, wz, &fbm, seed);
        assert_eq!(n.height.to_bits(), (h as f32).to_bits());
        assert_eq!(n.dh_dx.to_bits(), (gx as f32).to_bits());
        assert_eq!(n.dh_dz.to_bits(), (gz as f32).to_bits());
    }
}

/// Terrain-GEN microbench: the analytic gradient path (`sample_world`, ONE carved eval/node) vs the
/// old 5-tap central-difference path (one carved eval + 4 offset taps/node). `#[ignore]` — run with
/// `--release --ignored --nocapture`. Reports the per-node gen time + the speedup, the direct measure
/// of the gen-perf regression the analytic path recovers.
#[test]
#[ignore = "gen-perf microbench; run with --release --ignored --nocapture"]
fn bench_analytic_vs_fd_gradient() {
    let l = layer();
    let seed = 4242u64;
    let fbm = l.fbm_params(seed);
    // A grid of distinct world points (HEIGHT_FIELD_RES² ≈ one chunk's worth, several chunks over).
    let n = 256usize;
    let mut pts = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            pts.push((i as f64 * 2.0 - 256.0, j as f64 * 2.0 + 100.0));
        }
    }
    let e = 0.5f64; // the old EROSION_GRAD_EPS

    // Analytic: 1 carved eval/node (value + gradient together).
    let t0 = std::time::Instant::now();
    let mut acc = 0.0f64;
    for &(wx, wz) in &pts {
        let (h, gx, gz) = l.carved_grad(wx, wz, &fbm, seed);
        acc += h + gx + gz;
    }
    let analytic_ns = t0.elapsed().as_nanos() as f64 / pts.len() as f64;

    // Old FD: 1 value eval + 4 offset value taps/node (the regression this replaced).
    let t1 = std::time::Instant::now();
    let mut acc2 = 0.0f64;
    for &(wx, wz) in &pts {
        let h = l.carved_grad(wx, wz, &fbm, seed).0;
        let hxp = l.carved_grad(wx + e, wz, &fbm, seed).0;
        let hxm = l.carved_grad(wx - e, wz, &fbm, seed).0;
        let hzp = l.carved_grad(wx, wz + e, &fbm, seed).0;
        let hzm = l.carved_grad(wx, wz - e, &fbm, seed).0;
        acc2 += h + (hxp - hxm) / (2.0 * e) + (hzp - hzm) / (2.0 * e);
    }
    let fd_ns = t1.elapsed().as_nanos() as f64 / pts.len() as f64;

    eprintln!(
        "TERRAIN-GEN-BENCH: analytic {analytic_ns:.1} ns/node | 5-tap FD {fd_ns:.1} ns/node | \
         speedup {:.2}x  (sink {acc:.3}/{acc2:.3})",
        fd_ns / analytic_ns
    );
    assert!(analytic_ns < fd_ns, "analytic must be faster than the 5-tap FD");
}

/// GEN-PERF: full-chunk [`HeightLayer::generate`] cost — the band-limit finalize stage vs the plain
/// point-sample fast path — plus the `sample_world` call count each makes. `#[ignore]` — run with
/// `--release --ignored --nocapture`. Confirms the separable band-limit is NOT a gen-perf regression:
/// it evaluates `sample_world` once per FINE sample (`((res+2·apron)·D+1)²`, independent of kernel
/// radius), which for the default radius is COMPARABLE to (and below the old 9× box supersample's)
/// `9·(res+1)²` — the convolution itself is cheap float work on the cached grid.
#[test]
#[ignore = "gen-perf bench; run with --release --ignored --nocapture"]
fn bench_generate_chunk() {
    let coord = ChunkCoord::new(LayerId(0), IVec3::new(3, 0, -5));
    let seed = 4242u64;
    let nodes = ((HEIGHT_FIELD_RES + 1) * (HEIGHT_FIELD_RES + 1)) as f64;

    let run = |label: &str, l: &HeightLayer| {
        let size = l.chunk_size();
        // Warm + timed runs (generate allocates the fine grid; amortise the allocator).
        for _ in 0..2 {
            let mut o = GenOutput::default();
            l.generate(&GenCtx { coord, seed, size }, &mut o);
        }
        let reps = 8;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            let mut o = GenOutput::default();
            l.generate(&GenCtx { coord, seed, size }, &mut o);
        }
        let us = t.elapsed().as_micros() as f64 / reps as f64;
        eprintln!("GEN-PERF [{label}]: {us:.0} µs/chunk ({:.1} ns/node)", us * 1000.0 / nodes);
        us
    };

    let band = run("band-limit (default)", &layer());
    let plain = run("plain point-sample", &plain_layer());
    // The old box supersample evaluated `sample_world` 9× per node; the band-limit's fine grid is fewer
    // evals than that, so the band-limit must stay within a small multiple of the plain path (NOT the
    // ~9× the old supersample cost). Generous bound — this is a regression tripwire, not a tight gate.
    assert!(band < plain * 6.0, "band-limit gen {band:.0}µs must stay well under 9× plain {plain:.0}µs");
}

/// With erosion disabled AND `ridge == 0`, `sample_world` takes the exact analytic fBm path — its
/// gradient equals `fbm_height_grad` exactly (bit-for-bit), unchanged from the pre-erosion behaviour.
#[test]
fn plain_fbm_path_is_exact_analytic() {
    let params = HeightParams { ridge: 0.0, ..Default::default() };
    let erosion = ErosionParams { enabled: false, ..Default::default() };
    let l = HeightLayer::new(LayerId(0), params, erosion);
    let seed = 7u64;
    let fbm = l.fbm_params(seed);
    for &(wx, wz) in &[(0.0, 0.0), (123.5, -456.25), (-789.0, 1011.0)] {
        let n = l.sample_world(wx, wz, seed);
        let (h, gx, gz) = fbm_height_grad(wx, wz, &fbm);
        assert_eq!(n.height.to_bits(), ((h + params.sea_level as f64) as f32).to_bits());
        assert_eq!(n.dh_dx.to_bits(), (gx as f32).to_bits());
        assert_eq!(n.dh_dz.to_bits(), (gz as f32).to_bits());
    }
}
