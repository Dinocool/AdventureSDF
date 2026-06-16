//! Incremental re-pack correctness — the COMPLETENESS GATE.
//!
//! The headline test ([`incremental_matches_full_pack_over_camera_sequence`]) drives the [`ResidentPacker`]
//! through a sequence of resident-set changes (enter / drop / edit) and asserts its snapshot is, per resident
//! brick, BYTE-IDENTICAL to a from-scratch [`pack_resident_set`]. Because both paths produce each brick through
//! the SSOT [`pack_one`], any divergence means the incremental dirty set was INCOMPLETE (a stale halo or a wrong
//! uniform↔dense classification) — exactly the trap the 26-neighbourhood expansion exists to prevent. So this
//! equality IS the dirty-set-completeness proof.

use super::*;
use crate::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick};
use crate::voxel::gpu::{GpuBrickMeta, ResidentBrick, decode_paletted_cell, halo_cells, pack_resident_set};
use crate::voxel::palette::{BlockId, BlockRegistry};
use crate::sdf_render::worldgen::biome::{
    BiomeDef, BiomeId, BiomeLibrary, StrataLayer, TerrainMatId, TerrainSurfaceMaterial,
};
use bevy::math::IVec3;

fn registry() -> BlockRegistry {
    let mat = |name: &str, c: [f32; 4]| TerrainSurfaceMaterial {
        name: name.into(),
        base_color: c,
        roughness: 0.9,
        blend: 0.0,
        texture: None,
        tiling: 4.0,
        ..Default::default()
    };
    let materials = vec![mat("a", [0.1, 0.2, 0.3, 1.0]), mat("b", [0.4, 0.5, 0.6, 1.0])];
    let biomes = BiomeId::ALL
        .iter()
        .map(|_| BiomeDef {
            name: "b".into(),
            surface: TerrainMatId(0),
            surface_rules: vec![],
            strata: vec![StrataLayer { material: TerrainMatId(0), thickness: 1.0 }],
            bedrock: TerrainMatId(1),
        })
        .collect();
    BlockRegistry::from_biome_library(&BiomeLibrary { materials, biomes })
}

/// A brick with a checker-ish pattern so adjacent bricks have non-trivial halos (some core cells solid, some
/// air), exercising the halo border + uniform↔dense classification under neighbour changes.
fn patterned_brick(seed: i32) -> Brick {
    let mut v = Box::new([BlockId::AIR; BRICK_VOXELS]);
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                let s = (x + y + z + seed).rem_euclid(2);
                let idx = (x + y * BRICK_EDGE + z * BRICK_EDGE * BRICK_EDGE) as usize;
                v[idx] = if s == 0 { BlockId(1) } else { BlockId::AIR };
            }
        }
    }
    Brick::from_voxels(v)
}

/// A normalized per-brick fingerprint independent of slot/arena LAYOUT: the meta with its layout-dependent
/// `voxel_offset` field masked to just the uniform-flag bit (so a uniform brick's id survives but a dense
/// brick's running/arena offset is ignored), plus the dense voxel block (by value) when present. Two packs that
/// agree on this for every resident brick produce an identical render (the shader reads metas[].world_min/lod +
/// the voxel block at the offset — never the offset value itself, only what it points at).
#[derive(Clone, Debug, PartialEq)]
struct Fingerprint {
    voxel_origin: [i32; 3],
    world_min: [f32; 3],
    lod: u32,
    /// Uniform: `Some(block id)`; dense: `None` (the bytes live in `voxels`).
    uniform: Option<u16>,
    /// Dense voxel block (haloed 10³) by value; `None` for uniform.
    voxels: Option<Vec<u32>>,
}

fn meta_uniform(m: &GpuBrickMeta) -> Option<u16> {
    if m.is_uniform() { Some(m.uniform_block().0) } else { None }
}

/// Fingerprint EVERY brick of a packed [`GpuBrickPatch`], keyed by `(voxel_origin, lod)` (a unique per-brick key
/// independent of slot order). The ONE fingerprint function — both the from-scratch `pack_resident_set` and the
/// live `snapshot_patch` produce a `GpuBrickPatch` in the SAME (R2b paletted) representation, so the A/B gate
/// compares them through this single decode path (no dual raw-vs-paletted maintenance). R2b — each dense brick's
/// bit-packed index stream is DECODED back to the raw haloed cells (the logical voxel content the gate compares),
/// so a layout change can't mask a real divergence.
fn fingerprints(patch: &crate::voxel::gpu::GpuBrickPatch) -> std::collections::HashMap<([i32; 3], u32), Fingerprint> {
    let mut out = std::collections::HashMap::new();
    for m in &patch.metas {
        let voxels = if m.is_uniform() {
            None
        } else {
            let off = m.dense_offset() as usize;
            let pb = m.palette_base as usize;
            let bits = m.index_bits();
            // The remaining palette buffer suffices: decode only ever indexes the ≤k entries this brick uses.
            let palette: Vec<u16> = patch.brick_palettes[pb..].iter().map(|&x| x as u16).collect();
            let cells: Vec<u32> = (0..halo_cells(m.lod()))
                .map(|i| decode_paletted_cell(&palette, bits, &patch.voxels[off..], i) as u32)
                .collect();
            Some(cells)
        };
        out.insert(
            (m.voxel_origin, m.lod()),
            Fingerprint {
                voxel_origin: m.voxel_origin,
                world_min: m.world_min,
                lod: m.lod(),
                uniform: meta_uniform(m),
                voxels,
            },
        );
    }
    out
}

/// Fingerprint a from-scratch `pack_resident_set` of `entries`.
fn fingerprints_full(entries: &[ResidentBrick<'_>], reg: &BlockRegistry) -> std::collections::HashMap<([i32; 3], u32), Fingerprint> {
    fingerprints(&pack_resident_set(entries, reg))
}

/// Fingerprint the packer's LIVE re-pack output (`snapshot_patch`) — the EXACT `GpuBrickPatch` the render path
/// uploads. Routing the A/B gate through the same upload path the engine ships means the test can never validate
/// a representation that diverges from production.
fn fingerprints_incremental(packer: &ResidentPacker, reg: &BlockRegistry) -> std::collections::HashMap<([i32; 3], u32), Fingerprint> {
    fingerprints(&packer.snapshot_patch(reg))
}

fn assert_ab_equal(packer: &ResidentPacker, entries: &[ResidentBrick<'_>], reg: &BlockRegistry, label: &str) {
    let full = fingerprints_full(entries, reg);
    let inc = fingerprints_incremental(packer, reg);
    assert_eq!(inc.len(), full.len(), "{label}: brick COUNT differs (incremental {} vs full {})", inc.len(), full.len());
    for (k, f_full) in &full {
        let f_inc = inc.get(k).unwrap_or_else(|| panic!("{label}: brick {k:?} missing from incremental snapshot"));
        assert_eq!(f_inc, f_full, "{label}: brick {k:?} bytes differ (incremental dirty set incomplete?)");
    }
}

/// THE COMPLETENESS GATE: enter, drop, and edit bricks; after each step the packer's snapshot must equal a
/// from-scratch pack of the SAME resident set, byte-for-byte per brick.
#[test]
fn incremental_matches_full_pack_over_camera_sequence() {
    let reg = registry();
    let mut packer = ResidentPacker::new(4096);

    // Build a 4×4×4 block of patterned bricks (so halos are non-trivial) plus a solid 3×3×3 core (so some
    // bricks collapse uniform-incl-halo and some don't, exercising the uniform↔dense toggle on neighbour
    // change).
    let make_set = |omit: Option<IVec3>, solid_core: bool, edit: Option<(IVec3, Brick)>| -> Vec<(IVec3, Brick, u32)> {
        let mut v = Vec::new();
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    let c = IVec3::new(x, y, z);
                    if Some(c) == omit {
                        continue;
                    }
                    if let Some((ec, ref eb)) = edit
                        && ec == c
                    {
                        v.push((c, eb.clone(), 0));
                        continue;
                    }
                    let brick = if solid_core && (1..=2).contains(&x) && (1..=2).contains(&y) && (1..=2).contains(&z) {
                        Brick::uniform(BlockId(1))
                    } else {
                        patterned_brick(x + y + z)
                    };
                    v.push((c, brick, 0));
                }
            }
        }
        v
    };
    fn to_entries(owned: &[(IVec3, Brick, u32)]) -> Vec<ResidentBrick<'_>> {
        owned.iter().map(|(c, b, l)| ResidentBrick { coord: *c, brick: b, lod: *l }).collect()
    }
    let sorted = |mut owned: Vec<(IVec3, Brick, u32)>| {
        owned.sort_by_key(|(c, _, l)| (*l, c.z, c.y, c.x));
        owned
    };

    // Step 1: cold fill (everything enters).
    let s1 = sorted(make_set(None, true, None));
    let e1 = to_entries(&s1);
    let d1 = packer.update(&e1, reg.len() as u32);
    assert!(d1.topology_changed);
    assert_ab_equal(&packer, &e1, &reg, "step1 cold fill");

    // Step 2: drop one interior brick (its 26 neighbours' halos must update; some uniform cores re-expand
    // dense because their halo now reads AIR).
    let s2 = sorted(make_set(Some(IVec3::new(2, 2, 2)), true, None));
    let e2 = to_entries(&s2);
    let d2 = packer.update(&e2, reg.len() as u32);
    assert!(d2.topology_changed, "a drop is a topology change");
    assert!(!d2.changed.is_empty());
    assert_ab_equal(&packer, &e2, &reg, "step2 drop interior");

    // Step 3: re-add it (enter) — back to the full set.
    let s3 = sorted(make_set(None, true, None));
    let e3 = to_entries(&s3);
    packer.update(&e3, reg.len() as u32);
    assert_ab_equal(&packer, &e3, &reg, "step3 re-add");

    // Step 4: EDIT a boundary brick in place (rewrite) + mark it — its neighbours' halos must update.
    let edited = patterned_brick(99); // different pattern
    let s4 = sorted(make_set(None, true, Some((IVec3::new(0, 0, 0), edited))));
    let e4 = to_entries(&s4);
    packer.mark_rewritten([BrickKey { coord: IVec3::new(0, 0, 0), lod: 0 }]);
    let d4 = packer.update(&e4, reg.len() as u32);
    assert!(!d4.changed.is_empty(), "an edit must patch at least the edited brick");
    assert_ab_equal(&packer, &e4, &reg, "step4 edit boundary");

    // Step 5: drop a whole face slab (a camera-move-class churn) — exercises many drops + neighbour re-packs.
    let mut s5_owned = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 1..4 {
                // drop x==0 slab
                let c = IVec3::new(x, y, z);
                let brick = if (1..=2).contains(&x) && (1..=2).contains(&y) && (1..=2).contains(&z) {
                    Brick::uniform(BlockId(1))
                } else {
                    patterned_brick(x + y + z)
                };
                s5_owned.push((c, brick, 0u32));
            }
        }
    }
    let s5 = sorted(s5_owned);
    let e5 = to_entries(&s5);
    packer.update(&e5, reg.len() as u32);
    assert_ab_equal(&packer, &e5, &reg, "step5 drop face slab");
}

/// `snapshot_patch` (the live re-pack output, assembled by memcpy of cached bytes) is byte-identical — as a
/// `key → (meta, voxels)` mapping AND in palette + the NEE light list — to a from-scratch `pack_resident_set`.
/// This is the proof the LIVE render path (which uploads `snapshot_patch`) is pixel-identical to the old
/// full-pack path.
#[test]
fn snapshot_patch_matches_full_pack() {
    let reg = registry();
    let mut packer = ResidentPacker::new(4096);
    // A block with some emissive voxels to exercise the light list (block 1 isn't emissive in this registry, so
    // the light list is empty either way — but the palette + the empty-light invariant must still match).
    let mut owned = Vec::new();
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                let brick = if (x, y, z) == (1, 1, 1) {
                    Brick::uniform(BlockId(1))
                } else {
                    patterned_brick(x + 2 * y + 3 * z)
                };
                owned.push((IVec3::new(x, y, z), brick, 0u32));
            }
        }
    }
    owned.sort_by_key(|(c, _, l)| (*l, c.z, c.y, c.x));
    let entries: Vec<ResidentBrick> = owned.iter().map(|(c, b, l)| ResidentBrick { coord: *c, brick: b, lod: *l }).collect();
    packer.update(&entries, reg.len() as u32);

    let full = pack_resident_set(&entries, &reg);
    let snap = packer.snapshot_patch(&reg);
    assert_eq!(snap.brick_count(), full.brick_count(), "brick count matches");
    assert_eq!(snap.palette, full.palette, "palette identical");
    assert_eq!(snap.lights.len(), full.lights.len(), "light count identical");

    // Per-brick bytes match as a key→fingerprint mapping (slot/order differs, content does not). Reuses the ONE
    // shared `fingerprints` decode so this test and the completeness gate validate the same representation.
    let ff = fingerprints(&full);
    let fs = fingerprints(&snap);
    assert_eq!(ff.len(), fs.len());
    for (k, vfull) in &ff {
        assert_eq!(fs.get(k), Some(vfull), "brick {k:?} bytes differ between snapshot_patch and full pack");
    }
}

/// O(changed): a single-brick edit (rewrite) patches only the edited brick + its resident 26-neighbourhood,
/// NOT the whole resident set — the perf claim, asserted on the changed-slot count.
#[test]
fn edit_patches_only_local_neighbourhood() {
    let reg = registry();
    let mut packer = ResidentPacker::new(4096);
    // A 5×5×5 patterned block.
    let mut owned = Vec::new();
    for z in 0..5 {
        for y in 0..5 {
            for x in 0..5 {
                owned.push((IVec3::new(x, y, z), patterned_brick(x * 7 + y * 3 + z), 0u32));
            }
        }
    }
    owned.sort_by_key(|(c, _, l)| (*l, c.z, c.y, c.x));
    let entries: Vec<ResidentBrick> = owned.iter().map(|(c, b, l)| ResidentBrick { coord: *c, brick: b, lod: *l }).collect();
    packer.update(&entries, reg.len() as u32); // cold fill
    let resident = packer.resident_count();
    assert_eq!(resident, 125);

    // Edit the centre brick (2,2,2): rewrite it with a new pattern, mark it.
    let edited = patterned_brick(123);
    let mut owned2 = owned.clone();
    for (c, b, _) in owned2.iter_mut() {
        if *c == IVec3::new(2, 2, 2) {
            *b = edited.clone();
        }
    }
    let entries2: Vec<ResidentBrick> = owned2.iter().map(|(c, b, l)| ResidentBrick { coord: *c, brick: b, lod: *l }).collect();
    packer.mark_rewritten([BrickKey { coord: IVec3::new(2, 2, 2), lod: 0 }]);
    let d = packer.update(&entries2, reg.len() as u32);
    // At most the centre brick + its 26 neighbours can change — well under the 125 resident bricks (O(changed),
    // not O(resident)). The actual count is ≤ 27.
    assert!(d.changed.len() <= 27, "edit touched {} slots — must be O(neighbourhood) not O(resident)", d.changed.len());
    assert!(!d.changed.is_empty());
}

/// A CPU mirror of the render-world's fixed-cap GPU buffers (A4.4): the meta/aabb arrays + the paletted index
/// slab arena + the per-brick palette buffer. A `StreamSnapshot` overwrites all four; a `Delta` `queue_write`s
/// ONLY the changed slots' meta/aabb/index/palette blocks — the exact `queue_write_buffer` the render path runs.
/// After a sequence of deltas this must reconstruct the same logical content a from-scratch `snapshot_buffers()`
/// produces (the GPU-side analogue of the contiguous A/B gate).
struct SimBuffers {
    metas: Vec<GpuBrickMeta>,
    aabbs: Vec<crate::voxel::gpu::GpuBrickAabb>,
    indices: Vec<u32>,
    brick_palettes: Vec<u32>,
}

impl SimBuffers {
    /// Initialise from a full `snapshot_buffers()` (the epoch-start StreamSnapshot the render path uploads once).
    fn from_snapshot(s: &SnapshotBuffers) -> Self {
        Self {
            metas: s.metas.clone(),
            aabbs: s.aabbs.clone(),
            indices: s.indices.clone(),
            brick_palettes: s.brick_palettes.clone(),
        }
    }

    /// Apply a `RepackDelta` exactly as the render world's `apply_delta` does: write each changed slot's meta + aabb
    /// (at `slot`) and, for a dense slot whose content changed, the index block (at `index_word_offset`) + the
    /// palette block (at `palette_word_offset`).
    fn apply(&mut self, delta: &RepackDelta) {
        for cs in &delta.changed {
            self.metas[cs.slot as usize] = cs.meta;
            self.aabbs[cs.slot as usize] = cs.aabb;
            if let Some(idx) = &cs.index {
                let off = cs.index_word_offset as usize;
                self.indices[off..off + idx.len()].copy_from_slice(idx);
            }
            if let Some(pal) = &cs.palette {
                let off = cs.palette_word_offset as usize;
                self.brick_palettes[off..off + pal.len()].copy_from_slice(pal);
            }
        }
    }

    /// Wrap the mirror as a [`GpuBrickPatch`] so the SSOT `cell_block` decode reads it (the shader mirror). The
    /// registry palette / lights are irrelevant to `cell_block`, so they are left empty.
    fn as_patch(&self) -> crate::voxel::gpu::GpuBrickPatch {
        crate::voxel::gpu::GpuBrickPatch {
            aabbs: self.aabbs.clone(),
            metas: self.metas.clone(),
            voxels: self.indices.clone(),
            brick_palettes: self.brick_palettes.clone(),
            palette: Vec::new(),
            lights: Vec::new(),
            alias: Vec::new(),
        }
    }
}

/// **THE A1 BYTE-IDENTITY GATE.** Drive the packer through a camera-class sequence (cold fill → drop → re-add →
/// edit → face-slab drop), applying each `RepackDelta` to a CPU mirror of the fixed-cap GPU buffers the SAME way
/// the render world's `queue_write_buffer` does, and after EACH step assert the delta-mirrored buffers
/// byte-equal a from-scratch `snapshot_buffers()` at that generation. This proves the streamed O(changed) upload
/// reconstructs EXACTLY the buffer state a full re-snapshot would — so a half-applied delta (a stale slot, a
/// missed neighbour) can't slip through as a divergent GPU buffer.
#[test]
fn delta_upload_matches_snapshot_buffers_over_sequence() {
    let reg = registry();
    let mut packer = ResidentPacker::new(4096);

    let make = |omit: Option<IVec3>, edit: Option<(IVec3, Brick)>| -> Vec<(IVec3, Brick, u32)> {
        let mut v = Vec::new();
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    let c = IVec3::new(x, y, z);
                    if Some(c) == omit {
                        continue;
                    }
                    if let Some((ec, ref eb)) = edit
                        && ec == c
                    {
                        v.push((c, eb.clone(), 0));
                        continue;
                    }
                    let brick = if (1..=2).contains(&x) && (1..=2).contains(&y) && (1..=2).contains(&z) {
                        Brick::uniform(BlockId(1))
                    } else {
                        patterned_brick(x + y + z)
                    };
                    v.push((c, brick, 0));
                }
            }
        }
        v.sort_by_key(|(c, _, l)| (*l, c.z, c.y, c.x));
        v
    };
    fn to_entries(owned: &[(IVec3, Brick, u32)]) -> Vec<ResidentBrick<'_>> {
        owned.iter().map(|(c, b, l)| ResidentBrick { coord: *c, brick: b, lod: *l }).collect()
    }
    fn check(
        packer: &mut ResidentPacker,
        owned: &[(IVec3, Brick, u32)],
        sim: &mut SimBuffers,
        reg: &BlockRegistry,
        label: &str,
    ) {
        let delta = packer.update(&to_entries(owned), reg.len() as u32);
        if packer.grew() {
            // An index-arena GROW: the render ships a StreamSnapshot (re-allocating the larger buffer), NOT this
            // delta — so re-seed the mirror from the fresh snapshot (which also clears `grew`). The snapshot build
            // itself is validated against the shadow by the other gates; the small test sequence rarely grows.
            *sim = SimBuffers::from_snapshot(&packer.snapshot_buffers(reg));
            return;
        }
        sim.apply(&delta);
        let fresh = packer.snapshot_buffers(reg);
        // Metas + AABBs are byte-identical: a freed slot's delta writes a zeroed meta + degenerate AABB, exactly
        // what `snapshot_buffers` puts there. So the directory is reconstructed byte-for-byte by the delta path.
        assert_eq!(sim.metas, fresh.metas, "{label}: delta-mirrored metas != fresh snapshot");
        assert_eq!(sim.aabbs, fresh.aabbs, "{label}: delta-mirrored aabbs != fresh snapshot");
        // The index slab + per-brick palette: only RESIDENT DENSE slots' blocks are referenced (a freed/reused
        // slot keeps stale-but-unread bytes — its meta is zeroed/degenerate or its new palette has fewer entries;
        // the shader only reads `[palette_base, palette_base + k)` so stale tail bytes never render). So compare
        // the fully-written INDEX block byte-for-byte (proves the slab offset/content landed) AND decode every
        // haloed cell via the SSOT `cell_block` from both buffers (proves palette correctness without needing `k`).
        let degenerate = degenerate_aabb();
        let sim_patch = sim.as_patch();
        let fresh_patch = fresh_as_patch(&fresh);
        for (slot, m) in fresh.metas.iter().enumerate() {
            if fresh.aabbs[slot] == degenerate || m.is_uniform() {
                continue; // a freed slot (degenerate) or a uniform brick (no index/palette block)
            }
            let off = m.dense_offset() as usize;
            let len = index_class_words(m.index_bits());
            assert_eq!(
                &sim.indices[off..off + len],
                &fresh.indices[off..off + len],
                "{label}: delta-mirrored index block at {off} (slot {slot}, origin {:?}) != fresh snapshot",
                m.voxel_origin,
            );
            let sm = &sim.metas[slot];
            for cell in 0..halo_cells(m.lod()) {
                assert_eq!(
                    sim_patch.cell_block(sm, cell),
                    fresh_patch.cell_block(m, cell),
                    "{label}: cell {cell} of slot {slot} (origin {:?}) decodes differently (delta vs fresh)",
                    m.voxel_origin,
                );
            }
        }
    }

    /// Wrap a `SnapshotBuffers` as a `GpuBrickPatch` for `cell_block` decode (the fresh-snapshot side).
    fn fresh_as_patch(s: &SnapshotBuffers) -> crate::voxel::gpu::GpuBrickPatch {
        crate::voxel::gpu::GpuBrickPatch {
            aabbs: s.aabbs.clone(),
            metas: s.metas.clone(),
            voxels: s.indices.clone(),
            brick_palettes: s.brick_palettes.clone(),
            palette: Vec::new(),
            lights: Vec::new(),
            alias: Vec::new(),
        }
    }

    // Step 1: cold fill → StreamSnapshot (the epoch-start upload). Seed the mirror from it.
    let s1 = make(None, None);
    let d1 = packer.update(&to_entries(&s1), reg.len() as u32);
    assert!(d1.topology_changed);
    let snap = packer.snapshot_buffers(&reg);
    let mut sim = SimBuffers::from_snapshot(&snap);
    assert_eq!(sim.metas, snap.metas, "step1 snapshot mirror metas");
    assert_eq!(sim.indices, snap.indices, "step1 snapshot mirror indices");
    assert_eq!(sim.brick_palettes, snap.brick_palettes, "step1 snapshot mirror palettes");

    // Step 2: drop an interior brick (its 26 neighbours' halos update; its slot collapses to degenerate/zeroed).
    check(&mut packer, &make(Some(IVec3::new(2, 2, 2)), None), &mut sim, &reg, "step2 drop interior");
    // Step 3: re-add it (enter).
    check(&mut packer, &make(None, None), &mut sim, &reg, "step3 re-add");
    // Step 4: edit a boundary brick in place (mark rewritten so it re-packs).
    let edited = patterned_brick(99);
    packer.mark_rewritten([BrickKey { coord: IVec3::new(0, 0, 0), lod: 0 }]);
    check(&mut packer, &make(None, Some((IVec3::new(0, 0, 0), edited))), &mut sim, &reg, "step4 edit boundary");
    // Step 5: drop a whole face slab (x==0) — many drops + neighbour re-packs.
    let mut slab = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 1..4 {
                let c = IVec3::new(x, y, z);
                let brick = if (1..=2).contains(&x) && (1..=2).contains(&y) && (1..=2).contains(&z) {
                    Brick::uniform(BlockId(1))
                } else {
                    patterned_brick(x + y + z)
                };
                slab.push((c, brick, 0u32));
            }
        }
    }
    slab.sort_by_key(|(c, _, l)| (*l, c.z, c.y, c.x));
    check(&mut packer, &slab, &mut sim, &reg, "step5 drop face slab");
}

/// **The A4.4 streamed-arena LOGICAL-CONTENT gate.** Both the streamed `snapshot_buffers` (paletted size-class
/// slabs + fixed per-slot palette) and the from-scratch `pack_resident_set` (R2b contiguous palettes) store a
/// dense brick as a bit-packed index stream + per-brick palette. Decoding the streamed arena via the SSOT
/// `cell_block` (the same paletted branch the shader runs) must yield the EXACT same logical block ids per haloed
/// cell as decoding the R2b patch — proving the streamed paletted representation renders identically to the static
/// path through the one shared decode. Keyed by `(voxel_origin, lod)`, slot-order independent.
#[test]
fn streamed_snapshot_decodes_same_logical_cells_as_r2b() {
    use crate::voxel::gpu::{GpuBrickPatch, halo_cells};
    let reg = registry();
    let mut packer = ResidentPacker::new(4096);
    let mut owned = Vec::new();
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                let brick = if (x, y, z) == (1, 1, 1) {
                    Brick::uniform(BlockId(1))
                } else {
                    patterned_brick(x + 2 * y + 3 * z)
                };
                owned.push((IVec3::new(x, y, z), brick, 0u32));
            }
        }
    }
    owned.sort_by_key(|(c, _, l)| (*l, c.z, c.y, c.x));
    let entries: Vec<ResidentBrick> =
        owned.iter().map(|(c, b, l)| ResidentBrick { coord: *c, brick: b, lod: *l }).collect();
    packer.update(&entries, reg.len() as u32);

    // The from-scratch R2b patch (the static path).
    let r2b = pack_resident_set(&entries, &reg);
    // The streamed paletted snapshot wrapped as a `GpuBrickPatch` so the SSOT `cell_block` paletted branch decodes
    // it (`index_bits >= 1` ⇒ `brick_palettes[palette_base + (indices[off..] >> .. & mask)]`). Only the resident
    // metas matter; degenerate slots are uniform-zeroed (block 0, never compared).
    let snap = packer.snapshot_buffers(&reg);
    let streamed = GpuBrickPatch {
        aabbs: snap.aabbs.clone(),
        metas: snap.metas.clone(),
        voxels: snap.indices.clone(),
        brick_palettes: snap.brick_palettes.clone(),
        palette: snap.palette.clone(),
        lights: Vec::new(),
        alias: Vec::new(),
    };

    // Index the streamed metas by (voxel_origin, lod) so we compare the same brick regardless of slot order. Only
    // RESIDENT slots count — an unused capacity slot is `zeroed()` (degenerate AABB) and would otherwise collide
    // on the `([0,0,0], 0)` key. The non-degenerate AABB is the resident-slot signal.
    let degenerate = degenerate_aabb();
    let streamed_by_key: std::collections::HashMap<([i32; 3], u32), GpuBrickMeta> = streamed
        .metas
        .iter()
        .zip(streamed.aabbs.iter())
        .filter(|(_, a)| **a != degenerate)
        .map(|(m, _)| ((m.voxel_origin, m.lod()), *m))
        .collect();

    // For every R2b brick, decode all haloed cells both ways and assert they agree.
    for m_r2b in &r2b.metas {
        let key = (m_r2b.voxel_origin, m_r2b.lod());
        let m_s = streamed_by_key.get(&key).unwrap_or_else(|| panic!("brick {key:?} missing from streamed arena"));
        // A uniform brick must stay uniform with the same id; a dense brick must be paletted (index_bits >= 1).
        assert_eq!(m_s.is_uniform(), m_r2b.is_uniform(), "brick {key:?} uniform-ness differs");
        if m_r2b.is_uniform() {
            assert_eq!(m_s.uniform_block(), m_r2b.uniform_block(), "brick {key:?} uniform id differs");
            continue;
        }
        assert!(m_s.index_bits() >= 1, "streamed dense brick must carry a paletted index_bits >= 1");
        for cell in 0..halo_cells(m_r2b.lod()) {
            let a = streamed.cell_block(m_s, cell);
            let b = r2b.cell_block(m_r2b, cell);
            assert_eq!(a, b, "brick {key:?} cell {cell} decodes differently (streamed {a:?} vs r2b {b:?})");
        }
    }
}

/// Slots are reused after a drop (the free list), and the deferred-free quarantine means a slot freed this
/// update is NOT reclaimed until the next update (keep-old-until-revealed at the slot level).
#[test]
fn dropped_slot_is_quarantined_then_reused() {
    let reg = registry();
    let mut packer = ResidentPacker::new(8);
    let b = patterned_brick(0);
    // Fill 4 bricks (slots 0..4).
    let e0: Vec<ResidentBrick> = (0..4)
        .map(|x| ResidentBrick { coord: IVec3::new(x, 0, 0), brick: &b, lod: 0 })
        .collect();
    packer.update(&e0, reg.len() as u32);
    assert_eq!(packer.resident_count(), 4);

    // Drop brick x=1. Its slot is freed → quarantine (NOT yet reusable).
    let e1: Vec<ResidentBrick> = [0, 2, 3]
        .iter()
        .map(|&x| ResidentBrick { coord: IVec3::new(x, 0, 0), brick: &b, lod: 0 })
        .collect();
    let d1 = packer.update(&e1, reg.len() as u32);
    assert_eq!(packer.resident_count(), 3);
    assert_eq!(d1.freed.len(), 1, "one brick dropped");

    // Add a NEW brick this same... next update: the quarantined slot is released at the TOP of update, so the
    // new brick claims the bump pointer (slot 4) on THIS update (quarantined slot 1 only freed now), then a
    // FURTHER add could reuse slot 1. Verify capacity isn't exceeded and the set is consistent.
    let e2: Vec<ResidentBrick> = [0, 2, 3, 5]
        .iter()
        .map(|&x| ResidentBrick { coord: IVec3::new(x, 0, 0), brick: &b, lod: 0 })
        .collect();
    packer.update(&e2, reg.len() as u32);
    assert_eq!(packer.resident_count(), 4);
}

/// **TIER-1 pre-size gate.** Streaming a load in `max_bricks_per_frame`-sized batches the way the production
/// `stream_voxel_rt_residency` does: the UN-pre-sized packer ([`ResidentPacker::new_unreserved`]) overflows its
/// arena repeatedly mid-load and `grew()`s (forcing the ~200 ms re-snapshots), while the PRE-SIZED packer
/// ([`ResidentPacker::new`], the production constructor) reserves to the cap up front and `grew()`s ZERO times
/// after its first snapshot — for the SAME load. This is the regression gate for the grow-snapshot fix.
#[test]
fn presized_arena_eliminates_mid_load_grows() {
    let reg = registry();
    // A small cap so the test is fast but the reserve still covers the corpus (cap ≥ corpus so the cap path is
    // exercised, not the slot-capacity drop path).
    let cap = 8192u32;
    let batch = 256usize;
    // A dense cube of patterned bricks (each non-uniform, so it allocates index + palette slabs).
    let corpus: Vec<(IVec3, Brick)> = {
        let edge = 16i32; // 16³ = 4096 dense bricks
        let mut v = Vec::new();
        for z in 0..edge {
            for y in 0..edge {
                for x in 0..edge {
                    v.push((IVec3::new(x, y, z), patterned_brick(x * 31 + y * 17 + z * 11)));
                }
            }
        }
        v
    };

    // Drive a packer through the corpus in batches, mirroring the production update→{snapshot|grow|delta} loop.
    // Returns the number of GROW-snapshots (a `grew()` AFTER the epoch's first snapshot).
    fn drive(packer: &mut ResidentPacker, corpus: &[(IVec3, Brick)], reg: &BlockRegistry, batch: usize) -> u32 {
        let mut resident: Vec<(IVec3, Brick)> = Vec::new();
        let mut epoch_snapshotted = false;
        let mut grows = 0u32;
        let mut next = 0usize;
        while next < corpus.len() {
            let end = (next + batch).min(corpus.len());
            resident.extend_from_slice(&corpus[next..end]);
            next = end;
            let entries: Vec<ResidentBrick> =
                resident.iter().map(|(c, b)| ResidentBrick { coord: *c, brick: b, lod: 0 }).collect();
            packer.update(&entries, reg.len() as u32);
            if !epoch_snapshotted || packer.grew() {
                if epoch_snapshotted && packer.grew() {
                    grows += 1; // a mid-load grow-snapshot
                }
                packer.snapshot_buffers(reg);
                epoch_snapshotted = true;
            }
        }
        grows
    }

    let before = drive(&mut ResidentPacker::new_unreserved(cap), &corpus, &reg, batch);
    let after = drive(&mut ResidentPacker::new(cap), &corpus, &reg, batch);
    assert!(before > 0, "the un-pre-sized arena must grow mid-load (else the test corpus is too small to be a gate)");
    assert_eq!(after, 0, "the pre-sized arena must NOT grow mid-load for a normal load (got {after} grows)");
}
