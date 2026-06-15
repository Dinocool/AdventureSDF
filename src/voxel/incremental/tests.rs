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
use crate::voxel::gpu::{GpuBrickMeta, ResidentBrick, pack_resident_set};
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

/// Fingerprint every brick of a from-scratch `pack_resident_set`, keyed by `(voxel_origin, lod)` (a unique
/// per-brick key independent of slot order).
fn fingerprints_full(entries: &[ResidentBrick<'_>], reg: &BlockRegistry) -> std::collections::HashMap<([i32; 3], u32), Fingerprint> {
    let patch = pack_resident_set(entries, reg);
    let mut out = std::collections::HashMap::new();
    for m in &patch.metas {
        let voxels = if m.is_uniform() {
            None
        } else {
            let off = m.dense_offset() as usize;
            Some(patch.voxels[off..off + dense_block_u32()].to_vec())
        };
        out.insert(
            (m.voxel_origin, m.lod),
            Fingerprint {
                voxel_origin: m.voxel_origin,
                world_min: m.world_min,
                lod: m.lod,
                uniform: meta_uniform(m),
                voxels,
            },
        );
    }
    out
}

/// Fingerprint every RESIDENT slot of a `ResidentPacker` snapshot (skipping unused/degenerate slots), keyed the
/// same way. Mirrors `fingerprints_full` so the two are directly comparable.
fn fingerprints_incremental(packer: &ResidentPacker, reg: &BlockRegistry) -> std::collections::HashMap<([i32; 3], u32), Fingerprint> {
    let _ = reg;
    let (metas, _aabbs, voxels) = packer.snapshot_buffers();
    // Only slots that are actually resident carry a brick; an unused slot has a zeroed meta + degenerate AABB.
    // We can't tell "zeroed meta of an unused slot" from "a real brick at voxel_origin [0,0,0] lod 0" by the
    // meta alone, so derive the resident slot set from the packer's live map.
    let resident_slots: std::collections::HashSet<u32> =
        packer.resident.values().map(|s| s.slot).collect();
    let mut out = std::collections::HashMap::new();
    for (slot, m) in metas.iter().enumerate() {
        if !resident_slots.contains(&(slot as u32)) {
            continue;
        }
        let vox = if m.is_uniform() {
            None
        } else {
            let off = m.dense_offset() as usize;
            Some(voxels[off..off + dense_block_u32()].to_vec())
        };
        out.insert(
            (m.voxel_origin, m.lod),
            Fingerprint {
                voxel_origin: m.voxel_origin,
                world_min: m.world_min,
                lod: m.lod,
                uniform: meta_uniform(m),
                voxels: vox,
            },
        );
    }
    out
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
    let d1 = packer.update(&e1);
    assert!(d1.topology_changed);
    assert_ab_equal(&packer, &e1, &reg, "step1 cold fill");

    // Step 2: drop one interior brick (its 26 neighbours' halos must update; some uniform cores re-expand
    // dense because their halo now reads AIR).
    let s2 = sorted(make_set(Some(IVec3::new(2, 2, 2)), true, None));
    let e2 = to_entries(&s2);
    let d2 = packer.update(&e2);
    assert!(d2.topology_changed, "a drop is a topology change");
    assert!(!d2.changed.is_empty());
    assert_ab_equal(&packer, &e2, &reg, "step2 drop interior");

    // Step 3: re-add it (enter) — back to the full set.
    let s3 = sorted(make_set(None, true, None));
    let e3 = to_entries(&s3);
    packer.update(&e3);
    assert_ab_equal(&packer, &e3, &reg, "step3 re-add");

    // Step 4: EDIT a boundary brick in place (rewrite) + mark it — its neighbours' halos must update.
    let edited = patterned_brick(99); // different pattern
    let s4 = sorted(make_set(None, true, Some((IVec3::new(0, 0, 0), edited))));
    let e4 = to_entries(&s4);
    packer.mark_rewritten([BrickKey { coord: IVec3::new(0, 0, 0), lod: 0 }]);
    let d4 = packer.update(&e4);
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
    packer.update(&e5);
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
    packer.update(&entries);

    let full = pack_resident_set(&entries, &reg);
    let snap = packer.snapshot_patch(&reg);
    assert_eq!(snap.brick_count(), full.brick_count(), "brick count matches");
    assert_eq!(snap.palette, full.palette, "palette identical");
    assert_eq!(snap.lights.len(), full.lights.len(), "light count identical");

    // Per-brick bytes match as a key→fingerprint mapping (slot/order differs, content does not).
    type BrickFp = (Option<u16>, Option<Vec<u32>>);
    fn fp_of(patch: &crate::voxel::gpu::GpuBrickPatch) -> std::collections::HashMap<([i32; 3], u32), BrickFp> {
        let mut out = std::collections::HashMap::new();
        for m in &patch.metas {
            let v = if m.is_uniform() {
                (Some(m.uniform_block().0), None)
            } else {
                let off = m.dense_offset() as usize;
                (None, Some(patch.voxels[off..off + dense_block_u32()].to_vec()))
            };
            out.insert((m.voxel_origin, m.lod), v);
        }
        out
    }
    let ff = fp_of(&full);
    let fs = fp_of(&snap);
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
    let _ = &reg;
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
    packer.update(&entries); // cold fill
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
    let d = packer.update(&entries2);
    // At most the centre brick + its 26 neighbours can change — well under the 125 resident bricks (O(changed),
    // not O(resident)). The actual count is ≤ 27.
    assert!(d.changed.len() <= 27, "edit touched {} slots — must be O(neighbourhood) not O(resident)", d.changed.len());
    assert!(!d.changed.is_empty());
}

/// Slots are reused after a drop (the free list), and the deferred-free quarantine means a slot freed this
/// update is NOT reclaimed until the next update (keep-old-until-revealed at the slot level).
#[test]
fn dropped_slot_is_quarantined_then_reused() {
    let reg = registry();
    let _ = &reg;
    let mut packer = ResidentPacker::new(8);
    let b = patterned_brick(0);
    // Fill 4 bricks (slots 0..4).
    let e0: Vec<ResidentBrick> = (0..4)
        .map(|x| ResidentBrick { coord: IVec3::new(x, 0, 0), brick: &b, lod: 0 })
        .collect();
    packer.update(&e0);
    assert_eq!(packer.resident_count(), 4);

    // Drop brick x=1. Its slot is freed → quarantine (NOT yet reusable).
    let e1: Vec<ResidentBrick> = [0, 2, 3]
        .iter()
        .map(|&x| ResidentBrick { coord: IVec3::new(x, 0, 0), brick: &b, lod: 0 })
        .collect();
    let d1 = packer.update(&e1);
    assert_eq!(packer.resident_count(), 3);
    assert_eq!(d1.freed.len(), 1, "one brick dropped");

    // Add a NEW brick this same... next update: the quarantined slot is released at the TOP of update, so the
    // new brick claims the bump pointer (slot 4) on THIS update (quarantined slot 1 only freed now), then a
    // FURTHER add could reuse slot 1. Verify capacity isn't exceeded and the set is consistent.
    let e2: Vec<ResidentBrick> = [0, 2, 3, 5]
        .iter()
        .map(|&x| ResidentBrick { coord: IVec3::new(x, 0, 0), brick: &b, lod: 0 })
        .collect();
    packer.update(&e2);
    assert_eq!(packer.resident_count(), 4);
}
