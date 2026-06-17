//! Worldgen upload / ring-build tests (split from upload.rs per the test-module convention).

use super::super::coord::{ChunkCoord, LayerId};
use super::super::layers::erosion::ErosionParams;
use super::super::layers::height::{HeightLayer, HeightParams};
use super::*;
use std::sync::Arc;

fn store_with(coords: &[(i32, i32)], seed: u64) -> ArtifactStore<ScalarField2D> {
    let layer = HeightLayer::new(LayerId(0), HeightParams::default(), ErosionParams::default());
    let size = ChunkSize::new(HEIGHT_CHUNK_CELLS);
    let mut store = ArtifactStore::new();
    for &(x, z) in coords {
        let coord = ChunkCoord::new(LayerId(0), IVec3::new(x, 0, z));
        let mut field = ScalarField2D::zeroed(coord, size, HEIGHT_FIELD_RES);
        for j in 0..=HEIGHT_FIELD_RES {
            for i in 0..=HEIGHT_FIELD_RES {
                let wp = field.node_world_xz(i, j);
                field.set(i, j, layer.sample_world(wp.x, wp.y, seed));
            }
        }
        store.insert(coord, Arc::new(field));
    }
    store
}

/// Build chunks for a SPECIFIC tier into a store: `LayerId(tier)`, chunk edge `cells`, sampled from
/// the same world-anchored fBm (so cross-tier values agree). Lets one store hold several tiers.
fn insert_tier(store: &mut ArtifactStore<ScalarField2D>, tier: u32, cells: u32, coords: &[(i32, i32)], seed: u64) {
    let layer = HeightLayer::new_tier(LayerId(tier), HeightParams::default(), ErosionParams::default(), cells);
    let size = ChunkSize::new(cells);
    for &(x, z) in coords {
        let coord = ChunkCoord::new(LayerId(tier), IVec3::new(x, 0, z));
        let mut field = ScalarField2D::zeroed(coord, size, HEIGHT_FIELD_RES);
        for j in 0..=HEIGHT_FIELD_RES {
            for i in 0..=HEIGHT_FIELD_RES {
                let wp = field.node_world_xz(i, j);
                field.set(i, j, layer.sample_world(wp.x, wp.y, seed));
            }
        }
        store.insert(coord, Arc::new(field));
    }
}

#[test]
fn cell_struct_is_16_bytes() {
    assert_eq!(std::mem::size_of::<GpuHeightCell>(), 16);
}

/// The ring resolves a world point to the SAME height the chunk's own `ScalarField2D::sample`
/// gives — the CPU↔GPU surface-parity contract (the `sample_ring` ↔ shader mirror is what makes
/// picking match the render).
#[test]
fn ring_sample_matches_field_sample() {
    let seed = 77;
    let store = store_with(&[(0, 0), (1, 0), (-1, -1), (3, 2)], seed);
    let ring = build_height_ring(&store);
    // Probe interior points of several resident chunks.
    let s = HEIGHT_CHUNK_CELLS as f64;
    for &(cx, cz) in &[(0, 0), (1, 0), (-1, -1), (3, 2)] {
        let field = store.get(ChunkCoord::new(LayerId(0), IVec3::new(cx, 0, cz))).unwrap();
        for &(u, v) in &[(0.1, 0.2), (0.5, 0.5), (0.83, 0.27)] {
            let wp = DVec2::new((cx as f64 + u) * s, (cz as f64 + v) * s);
            let ring_h = sample_ring(&ring, wp).expect("resident chunk resolves");
            let field_h = field.sample(wp);
            assert!((ring_h.height - field_h.height).abs() < 1e-3,
                "chunk ({cx},{cz}) at ({u},{v}): ring {} vs field {}", ring_h.height, field_h.height);
            assert!((ring_h.dh_dx - field_h.dh_dx).abs() < 1e-3);
        }
    }
}

/// A world point in a non-resident chunk misses (flat fallback), never aliasing a neighbour.
#[test]
fn absent_chunk_misses() {
    let store = store_with(&[(0, 0)], 1);
    let ring = build_height_ring(&store);
    let s = HEIGHT_CHUNK_CELLS as f64;
    // Chunk (0,0) resident; chunk (2,2) is not.
    assert!(sample_ring(&ring, DVec2::new(0.5 * s, 0.5 * s)).is_some());
    assert!(sample_ring(&ring, DVec2::new(2.5 * s, 2.5 * s)).is_none());
}

/// The CPU-ring global round-trips a published ring and clears back to `None` — the seam the
/// `Terrain` `eval_primitive` branch reads for picking/render parity.
#[test]
fn cpu_height_ring_global_roundtrips() {
    let store = store_with(&[(0, 0)], 5);
    let ring = Arc::new(build_height_ring(&store));
    set_cpu_height_ring(Some(ring.clone()));
    let got = cpu_height_ring().expect("ring published");
    // Same underlying allocation (Arc shared), and it samples the resident chunk.
    assert!(Arc::ptr_eq(&got, &ring));
    let s = HEIGHT_CHUNK_CELLS as f64;
    assert!(sample_ring(&got, DVec2::new(0.5 * s, 0.5 * s)).is_some());
    set_cpu_height_ring(None);
    assert!(cpu_height_ring().is_none());
}

/// The mip layout constants are internally consistent: offsets are the prefix sums of the
/// per-axis node counts squared, and the total matches `NODES_PER_CHUNK_MIPPED`.
#[test]
fn mip_layout_constants_consistent() {
    assert_eq!(MAX_HEIGHT_MIP, 6);
    let mut acc = 0u32;
    for m in 0..=MAX_HEIGHT_MIP as usize {
        assert_eq!(MIP_NODES_PER_AXIS[m], (HEIGHT_FIELD_RES >> m) + 1, "npa[{m}]");
        assert_eq!(MIP_NODE_OFFSET[m], acc, "offset[{m}]");
        acc += MIP_NODES_PER_AXIS[m] * MIP_NODES_PER_AXIS[m];
    }
    assert_eq!(acc, NODES_PER_CHUNK_MIPPED);
    assert_eq!(NODES_PER_CHUNK_MIPPED, 5722);
}

/// The ring now allocates `NODES_PER_CHUNK_MIPPED` nodes per slot (the whole mip pyramid).
#[test]
fn ring_node_buffer_is_mipped_size() {
    let store = store_with(&[(0, 0)], 1);
    let ring = build_height_ring(&store);
    assert_eq!(
        ring.nodes.len(),
        HEIGHT_RING_SLOTS as usize * NODES_PER_CHUNK_MIPPED as usize
    );
}

/// A CONSTANT height field stays constant at every mip (the tent filter preserves DC), and a
/// linear RAMP is a fixed point of the position-preserving 1-2-1 tent — so a planar field
/// downsamples to itself EXACTLY (the band-limiting property the coarse bake relies on).
#[test]
fn mip_downsample_constant_and_planar_exact() {
    let size = ChunkSize::new(HEIGHT_CHUNK_CELLS);
    let coord = ChunkCoord::new(LayerId(0), IVec3::new(0, 0, 0));

    // Constant field.
    let mut konst = ScalarField2D::zeroed(coord, size, HEIGHT_FIELD_RES);
    for j in 0..=HEIGHT_FIELD_RES {
        for i in 0..=HEIGHT_FIELD_RES {
            konst.set(i, j, HeightNode { height: 3.5, dh_dx: 0.0, dh_dz: 0.0 });
        }
    }
    let mut out = vec![[0.0f32; 4]; NODES_PER_CHUNK_MIPPED as usize];
    build_chunk_mips(&konst.nodes, &mut out);
    for m in 0..=MAX_HEIGHT_MIP as usize {
        let off = MIP_NODE_OFFSET[m] as usize;
        let n = (MIP_NODES_PER_AXIS[m] * MIP_NODES_PER_AXIS[m]) as usize;
        for node in &out[off..off + n] {
            assert!((node[0] - 3.5).abs() < 1e-5, "const mip {m} = {}", node[0]);
        }
    }

    // Planar ramp h = a·x + b·z + c; node-aligned coarse samples must equal the plane exactly.
    let (a, b, c) = (0.3f64, -0.7f64, 12.0f64);
    let mut plane = ScalarField2D::zeroed(coord, size, HEIGHT_FIELD_RES);
    for j in 0..=HEIGHT_FIELD_RES {
        for i in 0..=HEIGHT_FIELD_RES {
            let wp = plane.node_world_xz(i, j);
            plane.set(i, j, HeightNode {
                height: (a * wp.x + b * wp.y + c) as f32,
                dh_dx: a as f32,
                dh_dz: b as f32,
            });
        }
    }
    let store = {
        let mut s = ArtifactStore::new();
        s.insert(coord, Arc::new(plane));
        s
    };
    let ring = build_height_ring(&store);
    let base = ring.directory[ring_slot(IVec2::new(0, 0))].node_base as usize;
    for m in 0..=MAX_HEIGHT_MIP {
        let napa = MIP_NODES_PER_AXIS[m as usize];
        let spacing = ring.node_spacing as f64 * (1u32 << m) as f64;
        let off = base + MIP_NODE_OFFSET[m as usize] as usize;
        for jj in 0..napa {
            for ii in 0..napa {
                let wx = ii as f64 * spacing;
                let wz = jj as f64 * spacing;
                let expect = (a * wx + b * wz + c) as f32;
                let got = ring.nodes[off + (jj * napa + ii) as usize];
                assert!((got[0] - expect).abs() < 1e-2,
                    "planar mip {m} node ({ii},{jj}): {} vs {expect}", got[0]);
                assert!((got[1] - a as f32).abs() < 1e-4 && (got[2] - b as f32).abs() < 1e-4);
            }
        }
    }
    // sample_ring_mip on the planar ring reproduces the plane at off-node points too.
    let s = HEIGHT_CHUNK_CELLS as f64;
    for &(u, v) in &[(0.21, 0.62), (0.5, 0.5)] {
        let wp = DVec2::new(u * s, v * s);
        for m in 0..=MAX_HEIGHT_MIP {
            let n = sample_ring_mip(&ring, wp, m).expect("resident");
            let expect = (a * wp.x + b * wp.y + c) as f32;
            assert!((n.height - expect).abs() < 1e-2, "sample_ring_mip {m}: {} vs {expect}", n.height);
        }
    }
    // Mip 0 of sample_ring_mip equals sample_ring exactly (same data, same path).
    let wp = DVec2::new(0.33 * s, 0.77 * s);
    let a0 = sample_ring(&ring, wp).unwrap();
    let b0 = sample_ring_mip(&ring, wp, 0).unwrap();
    assert_eq!(a0.height.to_bits(), b0.height.to_bits());
}

/// The voxel→mip select rounds UP to the finest mip whose node spacing ≥ the voxel: the `0.0`
/// sentinel and any voxel ≤ the base spacing give mip 0; each spacing-doubling steps up one mip;
/// and it clamps to `MAX_HEIGHT_MIP`. Base node spacing here is `128/64 = 2 m`.
#[test]
fn mip_select_rounds_up_to_voxel() {
    let base = HEIGHT_CHUNK_CELLS as f32 / HEIGHT_FIELD_RES as f32; // 2 m
    assert_eq!(select_height_mip(base, 0.0), 0, "sentinel ⇒ finest");
    assert_eq!(select_height_mip(base, base), 0, "voxel == base ⇒ mip 0");
    assert_eq!(select_height_mip(base, base * 0.5), 0, "voxel finer than base ⇒ mip 0");
    // spacing(m) = base·2^m: 2,4,8,16,... A voxel just above spacing(m) needs mip m+1.
    assert_eq!(select_height_mip(base, base * 2.0), 1, "exactly one doubling ⇒ mip 1");
    assert_eq!(select_height_mip(base, base * 2.0 + 0.01), 2, "just over ⇒ rounds up to mip 2");
    assert_eq!(select_height_mip(base, base * 4.0), 2);
    // Beyond the pyramid clamps to the coarsest mip.
    assert_eq!(select_height_mip(base, base * 100_000.0), MAX_HEIGHT_MIP);
}

/// `try_sample_ring_lod` with `voxel_size == 0.0` is identical to `sample_ring` (mip 0), and a
/// coarse voxel routes through the matching coarse mip (`sample_ring_mip`) — the band-limited LOD
/// path the Terrain eval uses (the non-strict, NON-RENDERING variant).
#[test]
fn sample_ring_lod_selects_mip() {
    let store = store_with(&[(0, 0)], 11);
    let ring = build_height_ring(&store);
    let base = ring.node_spacing;
    let s = HEIGHT_CHUNK_CELLS as f64;
    let wp = DVec2::new(0.4 * s, 0.6 * s);
    // 0.0 sentinel ⇒ mip 0 ⇒ exactly sample_ring.
    let lod0 = try_sample_ring_lod(&ring, wp, 0.0).unwrap();
    let mip0 = sample_ring(&ring, wp).unwrap();
    assert_eq!(lod0.height.to_bits(), mip0.height.to_bits());
    // A voxel 4× the base spacing selects mip 2 — matches sample_ring_mip(.., 2).
    let lod = try_sample_ring_lod(&ring, wp, base * 4.0).unwrap();
    let mip = sample_ring_mip(&ring, wp, 2).unwrap();
    assert_eq!(lod.height.to_bits(), mip.height.to_bits());
}

/// `ring_covers_aabb` is true for an AABB wholly inside a built ring's resident region and false
/// for one straddling into an unloaded chunk — the predicate the residency coverage gate uses to
/// forbid meshing ground the artifact hasn't loaded.
#[test]
fn ring_covers_aabb_inside_and_outside() {
    // Resident chunks (0,0),(1,0),(0,1),(1,1) — a 2×2 loaded block.
    let store = store_with(&[(0, 0), (1, 0), (0, 1), (1, 1)], 3);
    let ring = build_height_ring(&store);
    let s = HEIGHT_CHUNK_CELLS as f32;
    // Fully inside the loaded block.
    assert!(ring_covers_aabb(
        &ring,
        bevy::math::Vec2::new(0.25 * s, 0.25 * s),
        bevy::math::Vec2::new(1.75 * s, 1.75 * s),
    ));
    // Straddles into chunk (2,0), which is NOT resident.
    assert!(!ring_covers_aabb(
        &ring,
        bevy::math::Vec2::new(1.5 * s, 0.5 * s),
        bevy::math::Vec2::new(2.5 * s, 0.5 * s),
    ));
    // Wholly outside the loaded region.
    assert!(!ring_covers_aabb(
        &ring,
        bevy::math::Vec2::new(5.0 * s, 5.0 * s),
        bevy::math::Vec2::new(5.5 * s, 5.5 * s),
    ));
}

/// `ring_resident_bounds` reports the min/max chunk-XZ over the loaded slots (decoded from the
/// directory key-tags), or `None` for an empty ring.
#[test]
fn ring_resident_bounds_spans_loaded_chunks() {
    let store = store_with(&[(-2, 1), (3, -4), (0, 0)], 7);
    let ring = build_height_ring(&store);
    assert_eq!(ring_resident_bounds(&ring), Some((IVec2::new(-2, -4), IVec2::new(3, 1))));
    let empty = build_height_ring(&ArtifactStore::new());
    assert_eq!(ring_resident_bounds(&empty), None);
}

/// The STRICT `sample_ring_lod` PANICS on a miss — a rendered bake sampling outside loaded
/// coverage is a coverage-gate bug, never a silent fallback.
#[test]
#[should_panic(expected = "outside loaded coverage")]
fn strict_sample_ring_lod_panics_on_miss() {
    let store = store_with(&[(0, 0)], 2);
    let ring = build_height_ring(&store);
    let s = HEIGHT_CHUNK_CELLS as f64;
    // Chunk (5,5) is not resident → strict sampler must panic.
    let _ = sample_ring_lod(&ring, DVec2::new(5.5 * s, 5.5 * s), 0.0);
}

/// Negative-coord chunks resolve correctly (the rem_euclid slot + key-tag path).
#[test]
fn negative_chunk_resolves() {
    let store = store_with(&[(-3, -5)], 9);
    let ring = build_height_ring(&store);
    let s = HEIGHT_CHUNK_CELLS as f64;
    let wp = DVec2::new((-3.0 + 0.5) * s, (-5.0 + 0.5) * s);
    assert!(sample_ring(&ring, wp).is_some(), "negative-coord chunk must resolve");
}

// --- Tiered clipmap tests ---

/// Build a 2-tier clipmap: tier 0 (fine, edge `HEIGHT_CHUNK_CELLS`) resident only near the origin,
/// tier 1 (coarse, edge `2·HEIGHT_CHUNK_CELLS`) resident over a wider region. A NEAR point is covered
/// by both → finest (tier 0) serves it; a FAR point is covered only by tier 1 → tier 1 serves it.
fn two_tier_clipmap(seed: u64) -> HeightClipmap {
    let c0 = HEIGHT_CHUNK_CELLS;
    let c1 = HEIGHT_CHUNK_CELLS * 2;
    let mut store = ArtifactStore::new();
    // Tier 0: a 2×2 fine block around the origin (covers chunks {0,1}²).
    insert_tier(&mut store, 0, c0, &[(0, 0), (1, 0), (0, 1), (1, 1)], seed);
    // Tier 1: a 3×3 coarse block (covers chunks {0,1,2}² → world out to 6·HEIGHT_CHUNK_CELLS).
    insert_tier(&mut store, 1, c1, &[(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)], seed);
    build_height_clipmap(&store, &[c0, c1])
}

/// `build_height_ring_for_tier` builds a coarse tier with the right chunk size + node spacing, and
/// it ignores chunks belonging to other tiers in the shared store.
#[test]
fn build_tier_ring_uses_tier_chunk_size_and_filters_layer() {
    let c0 = HEIGHT_CHUNK_CELLS;
    let c1 = HEIGHT_CHUNK_CELLS * 2;
    let mut store = ArtifactStore::new();
    insert_tier(&mut store, 0, c0, &[(0, 0)], 1);
    insert_tier(&mut store, 1, c1, &[(0, 0)], 1);
    let ring1 = build_height_ring_for_tier(&store, LayerId(1), c1);
    assert_eq!(ring1.chunk_world_size, c1 as f32);
    assert_eq!(ring1.node_spacing, c1 as f32 / HEIGHT_FIELD_RES as f32);
    // Only tier-1's chunk (0,0) is resident in this ring; tier-0's chunk didn't leak in.
    assert_eq!(ring_resident_bounds(&ring1), Some((IVec2::ZERO, IVec2::ZERO)));
}

/// The clipmap sampler picks the FINEST covering tier: a near point in tier 0 is served by tier 0;
/// a far point covered only by tier 1 is served by tier 1. We distinguish which tier served by the
/// node spacing the sample interpolated over (tier 0 = 2 m, tier 1 = 4 m) — sampling at a point
/// off-node in tier 0 but on a tier-1 node should match tier 1 exactly only when tier 1 serves it.
#[test]
fn clipmap_samples_finest_covering_tier() {
    let clip = two_tier_clipmap(123);
    let s0 = HEIGHT_CHUNK_CELLS as f64;
    // NEAR point inside tier-0's loaded block (chunk (0,0)) → tier 0 serves it. Matches tier 0's ring.
    let near = DVec2::new(0.5 * s0, 0.5 * s0);
    let got_near = try_sample_clipmap_lod(&clip, near, 0.0).expect("near covered");
    let tier0 = sample_ring(&clip[0], near).expect("tier0 covers near");
    assert_eq!(got_near.height.to_bits(), tier0.height.to_bits(), "near point served by finest tier 0");
    // FAR point beyond tier-0's block (chunk (2,2) in fine units) but inside tier-1 → tier 1 serves.
    let far = DVec2::new(2.5 * s0, 2.5 * s0);
    assert!(sample_ring(&clip[0], far).is_none(), "tier 0 does NOT cover the far point");
    let got_far = try_sample_clipmap_lod(&clip, far, 0.0).expect("far covered by coarse tier");
    let tier1 = sample_ring(&clip[1], far).expect("tier1 covers far");
    assert_eq!(got_far.height.to_bits(), tier1.height.to_bits(), "far point served by coarse tier 1");
}

/// Build a 4-tier concentric clipmap mirroring production residency: every tier resident as a
/// `radius`-disc of chunks around the origin focus, `radius_t = HEIGHT_CHUNK_CELLS·3.75·2^t` (the
/// `new_clipmap` window). So a point's covering set is the contiguous suffix the optimized sampler
/// relies on: near = all tiers, far = only the coarse ones.
fn concentric_clipmap(tiers: u32, seed: u64) -> HeightClipmap {
    let mut store = ArtifactStore::new();
    let mut cells_per_tier = Vec::new();
    for t in 0..tiers {
        let cells = HEIGHT_CHUNK_CELLS << t;
        cells_per_tier.push(cells);
        // Chunks within this tier's window radius (in this tier's chunk units), centred on origin.
        let radius_m = HEIGHT_CHUNK_CELLS as f64 * 3.75 * (1u32 << t) as f64;
        let cw = cells as f64;
        let r_chunks = (radius_m / cw).ceil() as i32;
        let mut coords = Vec::new();
        for cz in -r_chunks..=r_chunks {
            for cx in -r_chunks..=r_chunks {
                coords.push((cx, cz));
            }
        }
        insert_tier(&mut store, t, cells, &coords, seed);
    }
    build_height_clipmap(&store, &cells_per_tier)
}

/// Plain finest→coarsest reference walk (the pre-optimization sampler): the first tier that covers
/// `world_xz` serves it. The optimized `try_sample_clipmap_lod` MUST match this bit-for-bit.
fn ref_sample_clipmap_lod(clipmap: &HeightClipmap, world_xz: DVec2, voxel_size: f32) -> Option<HeightNode> {
    for ring in clipmap.iter() {
        if let Some(node) = try_sample_ring_lod(ring, world_xz, voxel_size) {
            return Some(node);
        }
    }
    None
}

/// THE INVARIANT GUARD: the hint-seeded `try_sample_clipmap_lod` returns BIT-IDENTICAL
/// `(height, dh_dx, dh_dz)` to the plain finest-covering walk for every `(world_xz, voxel_size)` across
/// a grid spanning multiple tiers — including near (all tiers cover), far (only coarse cover), the exact
/// tier-boundary radii, and beyond-coverage misses (both must be `None`). The marching order varies (the
/// thread-local hint must not corrupt a later query), so we sweep forward, backward, and a jumpy order.
#[test]
fn terrain_optimized_sampler_matches_plain_finest_covering_walk() {
    let clip = concentric_clipmap(4, 4242);
    let c0 = HEIGHT_CHUNK_CELLS as f64;

    // A grid of probe points from the origin out past the finest tier's reach into coarse-only land,
    // and a few beyond every tier (misses). Off-node fractions exercise the bilinear+mip blend.
    let mut probes: Vec<DVec2> = Vec::new();
    let mut t = -40.0;
    while t <= 40.0 {
        for &frac in &[0.0, 0.13, 0.5, 0.87] {
            probes.push(DVec2::new((t + frac) * c0, (t * 0.5 - frac) * c0));
        }
        t += 0.37;
    }
    // Voxel sizes spanning several mips (incl. the 0.0 sentinel and coarse voxels).
    let base = clip[0].node_spacing;
    let voxels = [0.0, base * 0.5, base, base * 2.0, base * 3.3, base * 8.0, base * 64.0];

    // Sweep forward, backward, and interleaved — the hint persists across calls, so all orders must agree.
    let mut order: Vec<usize> = (0..probes.len()).collect();
    let forward = order.clone();
    let mut backward = order.clone();
    backward.reverse();
    // Jumpy: even indices ascending then odd indices descending.
    order.sort_by_key(|&i| (i % 2, if i % 2 == 0 { i } else { probes.len() - i }));
    for sweep in [&forward, &backward, &order] {
        for &pi in sweep.iter() {
            let wp = probes[pi];
            for &vs in &voxels {
                let got = try_sample_clipmap_lod(&clip, wp, vs);
                let want = ref_sample_clipmap_lod(&clip, wp, vs);
                match (got, want) {
                    (Some(g), Some(w)) => {
                        assert_eq!(g.height.to_bits(), w.height.to_bits(), "height @ {wp:?} vs={vs}");
                        assert_eq!(g.dh_dx.to_bits(), w.dh_dx.to_bits(), "dh_dx @ {wp:?} vs={vs}");
                        assert_eq!(g.dh_dz.to_bits(), w.dh_dz.to_bits(), "dh_dz @ {wp:?} vs={vs}");
                    }
                    (None, None) => {}
                    (g, w) => panic!("coverage mismatch @ {wp:?} vs={vs}: optimized={:?} plain={:?}", g.is_some(), w.is_some()),
                }
            }
        }
    }
}

/// REGRESSION (the streaming crash): during streaming a coarser tier can be only PARTIALLY resident
/// (still filling) while a FINER tier is fully resident and covers — so a point's covering set is
/// NON-CONTIGUOUS (covered at tier `c`, NOT at `c+1`). `finest_covering_tier` MUST still return the
/// covered finer tier regardless of query order; a tier-select that assumed a contiguous suffix and
/// seeded from a prior high-tier query skipped the finer covering tier → returned `None` →
/// `sample_clipmap_lod`'s strict panic (the cull's full `any-tier` gate had already admitted the chunk).
#[test]
fn sampler_handles_non_contiguous_streaming_coverage() {
    let mut store = ArtifactStore::new();
    let cells = [HEIGHT_CHUNK_CELLS, HEIGHT_CHUNK_CELLS << 1, HEIGHT_CHUNK_CELLS << 2]; // 128 / 256 / 512
    insert_tier(&mut store, 0, cells[0], &[(0, 0), (0, 1), (1, 0), (1, 1)], 7); // tier 0: near origin only
    insert_tier(&mut store, 1, cells[1], &[(0, 0), (0, 1), (1, 0), (1, 1), (-1, -1), (-1, 0)], 7); // tier 1
    insert_tier(&mut store, 2, cells[2], &[(-1, -1)], 7); // tier 2 PARTIAL — a far chunk only, not at wp
    let clip = build_height_clipmap(&store, &cells);

    // wp: tier-0 chunk (2,0) ✗, tier-1 chunk (1,0) ✓, tier-2 chunk (0,0) ✗ ⇒ covered set = {1} (non-contiguous).
    let wp = DVec2::new(300.0, 100.0);
    assert!(!ring_covers(&clip[0], wp), "tier 0 doesn't reach wp");
    assert!(ring_covers(&clip[1], wp), "tier 1 covers wp");
    assert!(!ring_covers(&clip[2], wp), "tier 2 partial — doesn't cover wp");
    assert_eq!(finest_covering_tier(&clip, wp), Some(1), "must find the covered FINER tier 1, not None");

    // `far`: covered ONLY by tier 2 (the coarse chunk (-1,-1)). Query it FIRST, then wp — a hint-based
    // optimizer would carry tier 2 into the wp query and wrongly skip tier 1. The sampler must match the
    // plain finest-covering walk for EVERY query regardless of order.
    let far = DVec2::new(-300.0, -300.0);
    assert_eq!(finest_covering_tier(&clip, far), Some(2), "far covered only by tier 2");
    for &p in &[far, wp, far, wp] {
        for &vs in &[0.0f32, clip[0].node_spacing, clip[0].node_spacing * 4.0] {
            assert_eq!(
                try_sample_clipmap_lod(&clip, p, vs).map(|n| n.height.to_bits()),
                ref_sample_clipmap_lod(&clip, p, vs).map(|n| n.height.to_bits()),
                "non-contiguous sampler must match the plain walk @ {p:?} vs={vs}"
            );
        }
    }
}

/// In the STEADY STATE (every tier fully resident) the covering set of a point is a contiguous suffix
/// `[c, T-1]` — concentric windows, so once a tier covers, all coarser ones do. NOTE: the sampler does
/// NOT rely on this (it handles the non-contiguous streaming case above); this just documents the
/// steady-state geometry. A finer covering tier under an uncovered coarser one only arises mid-stream.
#[test]
fn clipmap_coverage_is_a_contiguous_suffix() {
    let clip = concentric_clipmap(4, 99);
    let c0 = HEIGHT_CHUNK_CELLS as f64;
    let mut t = -40.0;
    while t <= 40.0 {
        for &frac in &[0.0, 0.5, 0.91] {
            let wp = DVec2::new((t + frac) * c0, (t * 0.6 - frac) * c0);
            // Index of the first covering tier (plain walk), then assert ALL coarser tiers also cover.
            let first = clip.iter().position(|r| ring_covers(r, wp));
            if let Some(c) = first {
                for (ti, r) in clip.iter().enumerate().skip(c) {
                    assert!(ring_covers(r, wp), "tier {ti} must cover once tier {c} does, @ {wp:?}");
                }
            }
        }
        t += 0.41;
    }
}

/// `clipmap_covers_aabb` is true for a far footprint once its COARSE tier is resident (even though
/// the fine tier doesn't reach), and false when NO tier covers it.
#[test]
fn clipmap_covers_far_via_coarse_tier() {
    let clip = two_tier_clipmap(7);
    let s0 = HEIGHT_CHUNK_CELLS as f32;
    // A far footprint the FINE tier can't reach but the COARSE tier (3×3 of 2·cell chunks) covers.
    let far_min = bevy::math::Vec2::new(2.1 * s0, 2.1 * s0);
    let far_max = bevy::math::Vec2::new(3.9 * s0, 3.9 * s0); // within coarse chunks {1,2}² (world [2·c0, 6·c0])
    assert!(!ring_covers_aabb(&clip[0], far_min, far_max), "fine tier does not reach the far footprint");
    assert!(clipmap_covers_aabb(&clip, far_min, far_max), "coarse tier admits the far footprint");
    // Wholly outside every tier → not covered.
    let out_min = bevy::math::Vec2::new(50.0 * s0, 50.0 * s0);
    let out_max = bevy::math::Vec2::new(50.5 * s0, 50.5 * s0);
    assert!(!clipmap_covers_aabb(&clip, out_min, out_max), "no tier covers a far-far footprint");
    // Empty clipmap covers nothing.
    let empty: HeightClipmap = Vec::new();
    assert!(!clipmap_covers_aabb(&empty, far_min, far_max));
}

/// The STRICT clipmap sampler PANICS when no tier covers — a rendered miss is a coverage-gate bug.
#[test]
#[should_panic(expected = "outside loaded clipmap coverage")]
fn strict_clipmap_sampler_panics_on_miss() {
    let clip = two_tier_clipmap(2);
    let s0 = HEIGHT_CHUNK_CELLS as f64;
    // Far outside every tier's loaded region.
    let _ = sample_clipmap_lod(&clip, DVec2::new(100.0 * s0, 100.0 * s0), 1.0);
}

/// `continuous_height_mip` is the fractional sibling of `select_height_mip`: the `0.0`/NaN sentinels and
/// any voxel ≤ base give 0.0; it is monotone non-decreasing in voxel size; it returns the exact
/// `log2(ratio)` (so a spacing-doubling is mip 1.0, a √2 voxel is mip 0.5); and it clamps to
/// `MAX_HEIGHT_MIP`. At exact doublings it agrees with the integer `select_height_mip`.
#[test]
fn continuous_height_mip_monotone_and_clamped() {
    let base = HEIGHT_CHUNK_CELLS as f32 / HEIGHT_FIELD_RES as f32; // 2 m
    assert_eq!(continuous_height_mip(base, 0.0), 0.0, "sentinel ⇒ 0");
    assert_eq!(continuous_height_mip(base, f32::NAN), 0.0, "NaN ⇒ 0");
    assert_eq!(continuous_height_mip(base, base), 0.0, "voxel == base ⇒ 0");
    assert_eq!(continuous_height_mip(base, base * 0.5), 0.0, "voxel finer than base ⇒ 0");
    assert!((continuous_height_mip(base, base * 2.0) - 1.0).abs() < 1e-5, "one doubling ⇒ 1.0");
    assert!((continuous_height_mip(base, base * 4.0) - 2.0).abs() < 1e-5, "two doublings ⇒ 2.0");
    // √2 voxel ⇒ exactly halfway between mip 0 and mip 1.
    assert!((continuous_height_mip(base, base * 2.0f32.sqrt()) - 0.5).abs() < 1e-5, "√2 ⇒ 0.5");
    // Monotone non-decreasing.
    let mut prev = -1.0;
    for k in 0..200 {
        let v = base * (1.0 + k as f32 * 0.5);
        let m = continuous_height_mip(base, v);
        assert!(m >= prev - 1e-6, "monotone: {m} < {prev}");
        prev = m;
    }
    // Clamps to MAX_HEIGHT_MIP and agrees with the integer select at exact doublings.
    assert_eq!(continuous_height_mip(base, base * 100_000.0), MAX_HEIGHT_MIP as f32);
    for m in 0..=MAX_HEIGHT_MIP {
        let v = base * (1u32 << m) as f32;
        assert_eq!(continuous_height_mip(base, v) as u32, select_height_mip(base, v), "doubling {m}");
    }
}

/// `sample_ring_mip_frac` with `frac == 0` is BIT-identical to the integer `sample_ring_mip` (the fast
/// path), and `frac == 0.5` is the exact LERP midpoint of the two bracketing mips — for height AND both
/// gradient lanes. This is the trilinear-mip blend the geomorph ramp drives.
#[test]
fn sample_ring_mip_frac_blends_bracketing_mips() {
    let store = store_with(&[(0, 0)], 21);
    let ring = build_height_ring(&store);
    let s = HEIGHT_CHUNK_CELLS as f64;
    let wp = DVec2::new(0.37 * s, 0.61 * s);
    // frac == 0 ⇒ identical to the integer mip (fast path), for several mips.
    for m in 0..=MAX_HEIGHT_MIP {
        let frac = sample_ring_mip_frac(&ring, wp, m as f32).unwrap();
        let intg = sample_ring_mip(&ring, wp, m).unwrap();
        assert_eq!(frac.height.to_bits(), intg.height.to_bits(), "frac==0 mip {m} height");
        assert_eq!(frac.dh_dx.to_bits(), intg.dh_dx.to_bits(), "frac==0 mip {m} dh_dx");
        assert_eq!(frac.dh_dz.to_bits(), intg.dh_dz.to_bits(), "frac==0 mip {m} dh_dz");
    }
    // frac == 0.5 between mip 1 and mip 2 ⇒ the LERP midpoint of the two integer samples.
    let m1 = sample_ring_mip(&ring, wp, 1).unwrap();
    let m2 = sample_ring_mip(&ring, wp, 2).unwrap();
    let mid = sample_ring_mip_frac(&ring, wp, 1.5).unwrap();
    let want = |a: f32, b: f32| 0.5 * (a + b);
    assert!((mid.height - want(m1.height, m2.height)).abs() < 1e-5, "midpoint height");
    assert!((mid.dh_dx - want(m1.dh_dx, m2.dh_dx)).abs() < 1e-5, "midpoint dh_dx");
    assert!((mid.dh_dz - want(m1.dh_dz, m2.dh_dz)).abs() < 1e-5, "midpoint dh_dz");
}

/// The CPU clipmap global round-trips a published clipmap and clears back to `None`.
#[test]
fn cpu_height_clipmap_global_roundtrips() {
    let clip = Arc::new(two_tier_clipmap(5));
    set_cpu_height_clipmap(Some(clip.clone()));
    let got = cpu_height_clipmap().expect("clipmap published");
    assert!(Arc::ptr_eq(&got, &clip));
    set_cpu_height_clipmap(None);
    assert!(cpu_height_clipmap().is_none());
}
