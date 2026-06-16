//! **The pre-instancing GALLERY merge — several voxelized `.vox` scenes placed SIDE BY SIDE in one world.**
//!
//! The user wants "several voxelized demo scenes side by side" (Sponza + classics like Sibenik / Bistro) for
//! a GI / LOD comparison row. True per-object instancing (a TLAS with one BLAS per scene — see
//! `docs/VOXEL_INSTANCING_PLAN.md`) is NOT built yet, so this module does the SIMPLE thing that needs zero
//! new GPU plumbing: it LOADS each `.vox` into a [`BrickMap`] + [`BlockRegistry`] (via [`super::vox::load_vox`]),
//! SHIFTS each map by a per-scene brick offset so the scenes occupy NON-OVERLAPPING regions side by side, and
//! MERGES them into ONE [`BrickMap`] + ONE [`BlockRegistry`]. The merged map then streams through the EXACT
//! SAME [`super::source::StaticVoxSource`] + [`super::streaming::ResidencyManager`] clipmap path Sponza uses —
//! [`super::raytrace`] builds the source ONCE on the switch and reuses it every frame (never per-frame).
//!
//! ## How the merge stays correct
//! * **Non-overlapping placement.** Each scene is offset (in LOD0 BRICK coords) so its world-voxel AABB sits
//!   entirely beside its neighbour's, separated by a [`GALLERY_GAP_BRICKS`] gap — so NO two scenes can ever
//!   share a brick (the merge would otherwise lose voxels when one scene's brick overwrote another's). Offsets
//!   are either AUTO-SPACED along +X (each scene placed past the previous scene's bounds + the gap) or given
//!   EXPLICITLY per row entry.
//! * **Palette preservation.** Each scene keeps its OWN colours: the merged registry CONCATENATES the
//!   per-scene palettes, and every scene's local [`BlockId`] `b` (`b >= 1`) is remapped to merged id
//!   `palette_base - 1 + b`, where `palette_base` is the merged registry's length just before the scene's
//!   palette is appended (so the scene's local block 1 lands at the merged index it's appended to). [`BlockId`]
//!   is a `u16`, so the total block count must stay `<= u16::MAX`; a scene that would overflow is logged +
//!   SKIPPED (never panics).
//! * **Determinism.** The merge is a PURE function of the input list (each scene's load is deterministic, the
//!   offsets are computed deterministically from the scene bounds, the shift + remap are arithmetic), so the
//!   streamed result is reproducible run-to-run.
//!
//! A row entry whose `.vox` file is ABSENT is logged with `warn!` and SKIPPED — exactly like the missing-Sponza
//! path — so a partially-baked gallery still loads the scenes that exist (never a panic, never a hard fail).

use bevy::math::IVec3;
use bevy::log::warn;

use super::brickmap::{BRICK_EDGE, Brick, BrickMap, BRICK_VOXELS, voxel_index};
use super::palette::{BlockId, BlockRegistry};
use super::vox::load_vox;

/// Inter-scene GAP, in LOD0 bricks, inserted between consecutive auto-spaced gallery scenes. A few bricks of
/// empty space guarantees NO cross-scene brick overlap (the merge writes each scene into disjoint bricks) AND
/// keeps each scene's coarse-LOD mip footprint separate (a coarse brick aggregates `2^L` fine bricks, so a
/// 1-brick gap could let two scenes share a coarse cell — this gap is wide enough that even coarse shells stay
/// disjoint at the LODs the clipmap actually streams a side-by-side row at). Also a visual aisle between scenes.
pub const GALLERY_GAP_BRICKS: i32 = 8;

/// One scene in the gallery row: the baked `.vox` asset to load, and WHERE to place it. The placement is a
/// LOD0-brick offset added to every brick of the loaded (already floor-anchored, X/Z-centred) map.
///
/// `offset: None` ⇒ AUTO-SPACE: the loader places this scene immediately past the previous scene's +X bound
/// plus [`GALLERY_GAP_BRICKS`], on the same floor (Y) and Z centreline — the common "row of scenes" case.
/// `offset: Some(o)` ⇒ place it at the EXPLICIT brick offset `o` (for a custom 2D layout). Either way the
/// resulting regions must not overlap; auto-spacing guarantees it, an explicit offset is the author's job (the
/// merge does not move an explicitly-placed scene, but it still never lets two bricks collide — see the merge).
#[derive(Clone, Copy, Debug)]
pub struct GalleryEntry {
    /// Path to the baked `.vox` (relative to the crate root, like [`super::raytrace::SPONZA_VOX_PATH`]).
    /// Used by the LEGACY full-RAM merge ([`load_gallery`]) — kept as the import side + the fallback when no
    /// `.vxo` is present.
    pub vox_path: &'static str,
    /// Path to the STREAMED `.vxo` sibling (the region-streamed, bounded-RAM form of the same asset, baked at
    /// 0.05 m). The LIVE gallery prefers this — opened via [`super::vxo::VxoSource`] + merged into one
    /// [`super::vxo::MergedSource`] (see [`vxo_gallery_placements`]). Absent on disk ⇒ skipped with a warn,
    /// exactly like an absent `.vox` (a fresh checkout with no `.vxo` falls back to the `.vox` path).
    pub vxo_path: &'static str,
    /// Explicit LOD0-brick placement offset, or `None` to auto-space along +X past the previous scene.
    pub offset: Option<IVec3>,
    /// A short human label (for logging / a future per-scene UI marker). Not load-bearing for the merge.
    pub label: &'static str,
}

/// The DATA-DRIVEN gallery scene list — the SSOT row of side-by-side scenes. Starts with ONLY Sponza (the one
/// asset baked today) and is trivially extensible: add a [`GalleryEntry`] row per classic scene as it's baked
/// (Sibenik, San Miguel, Bistro — note Bistro ships as FBX and needs external conversion to glTF/OBJ first,
/// see `examples/voxelize_scene.rs`). Auto-spacing (`offset: None`) lays each new scene in +X order with a
/// [`GALLERY_GAP_BRICKS`] aisle, so adding a row is a one-liner — no offset math by hand. Absent assets are
/// skipped with a `warn!` at load (never a panic), so an un-baked row simply doesn't appear in the merge.
pub const GALLERY_SCENES: &[GalleryEntry] = &[
    GalleryEntry {
        vox_path: super::raytrace::SPONZA_VOX_PATH,
        vxo_path: "assets/models/sponza.vxo",
        offset: None,
        label: "Sponza",
    },
    GalleryEntry {
        vox_path: "assets/models/sibenik.vox",
        vxo_path: "assets/models/sibenik.vxo",
        offset: None,
        label: "Sibenik",
    },
    GalleryEntry {
        vox_path: "assets/models/conference.vox",
        vxo_path: "assets/models/conference.vxo",
        offset: None,
        label: "Conference",
    },
    GalleryEntry {
        // Bistro-Exterior @0.05 m, baked via the C1 tiled out-of-core voxelizer (the only path that fits its
        // ~13 B-cell AABB under the RAM budget — see `docs/TILED_VOXELIZER_PLAN.md` §C1 + `docs/TESTING.md`).
        // The legacy full-RAM `.vox` is too large to load; the gallery prefers the region-streamed `.vxo`.
        vox_path: "assets/models/bistro.vox",
        vxo_path: "assets/models/bistro.vxo",
        offset: None,
        label: "Bistro",
    },
    // Roadmap (uncomment as each is baked — auto-spaced +X with a gap, no offset math needed):
    // GalleryEntry { vox_path: "…/san_miguel.vox", vxo_path: "…/san_miguel.vxo", offset: None, label: "San Miguel" },
];

/// Load every entry in `scenes`, SHIFT each into a non-overlapping side-by-side region, and MERGE into ONE
/// [`BrickMap`] + ONE [`BlockRegistry`]. The single entry point used by [`super::raytrace`] on the switch into
/// [`super::VoxelScene::Gallery`] (loaded + cached ONCE, exactly like the Sponza `.vox`).
///
/// Behaviour, in order:
/// * Each entry is loaded via [`load_vox`]. A load FAILURE (missing file, parse error) is logged with `warn!`
///   and the entry is SKIPPED — a partially-baked gallery loads the scenes that exist (never panics).
/// * AUTO-SPACING: an entry with `offset: None` is placed so its world-voxel AABB begins just past the running
///   +X cursor (the previous scene's +X bound + [`GALLERY_GAP_BRICKS`] bricks), aligned on the floor and Z
///   centreline. An explicit `offset` overrides the cursor for that entry but still advances it (so a later
///   auto-spaced scene clears the explicitly-placed one too).
/// * MERGE: each loaded brick is re-keyed by `+offset` into the merged map ([`merge_brickmap_into`]); each
///   scene's local [`BlockId`] `b` is remapped to `palette_base + b` so it indexes the concatenated registry
///   ([`merge_registry`]). [`BlockId`] is a `u16`, so if the running palette base + a scene's palette would
///   exceed `u16::MAX`, that scene is logged + SKIPPED (capping `N` scenes by total colour count, never
///   wrapping a `BlockId`).
///
/// Returns the merged `(BrickMap, BlockRegistry)`. An EMPTY input list (or all entries skipped) yields an empty
/// map + an AIR-only registry — robust, like the empty-`.vox` path. Pure + deterministic (load + bounds +
/// arithmetic only).
pub fn load_gallery(scenes: &[GalleryEntry]) -> (BrickMap, BlockRegistry) {
    // Load each entry (a missing/invalid asset is skipped with a warn — like missing Sponza), pairing it with
    // its placement offset + label, then merge the loaded maps side by side.
    let mut loaded: Vec<LoadedScene> = Vec::with_capacity(scenes.len());
    for entry in scenes {
        match load_vox(entry.vox_path) {
            Ok((map, registry)) => loaded.push(LoadedScene {
                map,
                registry,
                offset: entry.offset,
                label: entry.label.to_string(),
            }),
            Err(e) => warn!(
                "gallery: skipping '{}' ({}): {e} — bake it via `cargo run --example voxelize_scene`",
                entry.label, entry.vox_path
            ),
        }
    }
    merge_scenes(loaded)
}

/// Compute the STREAMED `.vxo` gallery placement list `&[(vxo_path, +X brick offset)]` — the input to
/// [`super::vxo::MergedSource::open_paths`], which loads each `.vxo` region-streamed (bounded-RAM) instead of
/// the legacy full-RAM `.vox` merge ([`load_gallery`]). The OFFSETS auto-space the assets along +X exactly like
/// [`merge_scenes`] auto-spaces the `.vox` maps: each asset is placed so its −X brick bound lands at the running
/// cursor, and the cursor advances past its +X bound plus [`GALLERY_GAP_BRICKS`]. The per-asset brick WIDTH is
/// read from each `.vxo`'s `HEAD.bounds` (no region decode — only the eager header), so the spacing is a pure
/// function of the baked bounds, matching the `.vox` auto-spacer's "place past the previous scene + a gap".
///
/// A `.vxo` ABSENT on disk (or that fails to open) is SKIPPED with a `warn!` and does NOT advance the cursor —
/// mirroring [`load_gallery`]'s absent-`.vox` skip — so a partially-baked gallery still streams the assets that
/// exist (never a panic). An EMPTY return (no `.vxo` present) signals the caller to fall back to the legacy
/// `.vox` path. Pure + deterministic (open + bounds arithmetic only).
pub fn vxo_gallery_placements(scenes: &[GalleryEntry]) -> Vec<(std::path::PathBuf, IVec3)> {
    use super::vxo::VxoSource;

    let mut placements: Vec<(std::path::PathBuf, IVec3)> = Vec::with_capacity(scenes.len());
    // The +X auto-spacing cursor in LOD0 brick coords — the next free brick column past everything placed so
    // far. Starts at 0 (the first asset anchors at the origin), identical to `merge_scenes`.
    let mut x_cursor_bricks = 0i32;

    for entry in scenes {
        // Open the `.vxo` only to read its eager HEAD (bounds) — a missing/invalid asset is skipped with a warn
        // (no panic), exactly like a missing `.vox` in `load_gallery`. We re-open it inside `MergedSource` so the
        // streamed source owns its own mmap; this open is short-lived (header parse only, no region decode).
        let head = match VxoSource::open(entry.vxo_path) {
            Ok((source, _registry)) => *source.head(),
            Err(e) => {
                warn!(
                    "gallery: skipping streamed '{}' ({}): {e} — bake it via \
                     `cargo run --example voxelize_scene` (or the `.vox` fallback path will be used)",
                    entry.label, entry.vxo_path
                );
                continue;
            }
        };

        // The asset's LOCAL (un-offset) inclusive LOD0 brick-coord X span, derived from `HEAD.bounds` the SAME
        // way `MergedSource::new` does: bounds are LOD0 world VOXELS; convert to bricks via Euclidean floor, and
        // the exclusive max maps to the last inclusive voxel's brick.
        let bmin = IVec3::from_array(head.bounds_min);
        let bmax = IVec3::from_array(head.bounds_max);
        let lo_x = bmin.x.div_euclid(BRICK_EDGE);
        let hi_x = (bmax.x - 1).div_euclid(BRICK_EDGE);

        // Place this asset's −X bound at the cursor (auto-space), or honour an explicit offset. Either way we
        // advance the cursor past the PLACED +X bound + the gap so the next asset clears it. Y/Z are left where
        // the bake anchored them (offset 0) — the `.vxo` is floor/centre-anchored just like the `.vox` loader.
        let offset = match entry.offset {
            Some(o) => o,
            None => IVec3::new(x_cursor_bricks - lo_x, 0, 0),
        };
        let placed_hi_x = hi_x + offset.x;
        x_cursor_bricks = placed_hi_x + 1 + GALLERY_GAP_BRICKS;

        placements.push((std::path::PathBuf::from(entry.vxo_path), offset));
    }
    placements
}

/// BENCH HARNESS placement list (dev-only, `ADVENTURE_BENCH_BISTRO=1`): JUST `bistro.vxo` placed at offset
/// (0,0,0) so Bistro sits at the world origin (the `.vxo` is already floor-anchored y=0 + X/Z-centred, so a
/// zero offset leaves it centred at origin). A single-entry [`MergedSource::open_paths`] input — the FPS
/// benchmark target. Returns the Bistro row from [`GALLERY_SCENES`] with an explicit zero offset; empty (with a
/// warn) if Bistro isn't in the table (so the caller falls back exactly like an absent asset, never panics).
pub fn bistro_bench_placements() -> Vec<(std::path::PathBuf, IVec3)> {
    match GALLERY_SCENES.iter().find(|e| e.label == "Bistro") {
        Some(entry) => vec![(std::path::PathBuf::from(entry.vxo_path), IVec3::ZERO)],
        None => {
            warn!("bench: no 'Bistro' entry in GALLERY_SCENES — ADVENTURE_BENCH_BISTRO has nothing to load");
            Vec::new()
        }
    }
}

/// One already-loaded gallery scene ready to merge: its loaded [`BrickMap`] + [`BlockRegistry`], its placement
/// offset (`None` = auto-space), and a label for logging. The unit [`merge_scenes`] operates on — decoupled
/// from disk so the merge is testable with synthetic maps (no `.vox` files needed).
pub struct LoadedScene {
    /// The loaded fine [`BrickMap`] (already floor-anchored + X/Z-centred by [`load_vox`], or synthetic).
    pub map: BrickMap,
    /// The scene's own palette (preserved per-scene in the merge via a `palette_base` offset).
    pub registry: BlockRegistry,
    /// Explicit LOD0-brick placement offset, or `None` to auto-space along +X past the previous scene.
    pub offset: Option<IVec3>,
    /// A short label for logging (which scene was placed / skipped).
    pub label: String,
}

/// MERGE already-loaded scenes side by side into ONE [`BrickMap`] + ONE [`BlockRegistry`] — the pure core of
/// the gallery (shared by [`load_gallery`] and the integration tests, which feed it synthetic maps). For each
/// scene: place it (auto-spaced along +X past the running cursor + [`GALLERY_GAP_BRICKS`], or at its explicit
/// offset), CONCATENATE its palette (so it keeps its own colours, remapped by a `palette_base`), and re-key its
/// bricks into the merged map at the offset with BlockIds shifted by the base. A scene that would push the
/// merged palette past `u16::MAX` BlockIds is logged + SKIPPED (capping the gallery; never wraps a `BlockId`).
/// An empty scene (no solid bricks) is skipped. Pure + deterministic (bounds + arithmetic only).
pub fn merge_scenes(scenes: impl IntoIterator<Item = LoadedScene>) -> (BrickMap, BlockRegistry) {
    let mut merged_map = BrickMap::new();
    // The merged registry, built by concatenating each scene's palette. Starts AIR-only.
    let mut merged_registry = BlockRegistry::air_only();
    // The +X auto-spacing cursor, in LOD0 brick coords: the next free brick column past everything placed so
    // far. Starts at 0 (the first scene anchors at the origin).
    let mut x_cursor_bricks = 0i32;

    for scene in scenes {
        // The scene's loaded brick-coord AABB (already floor-anchored + X/Z-centred by the loader). An empty
        // scene (no solid bricks) has no bounds — skip it (nothing to merge, and it shouldn't move the cursor
        // misleadingly; auto-spacing measures real extents).
        let Some((bc_lo, bc_hi)) = brickmap_brick_bounds(&scene.map) else {
            warn!("gallery: '{}' loaded EMPTY (no solid bricks) — skipping", scene.label);
            continue;
        };

        // The placement offset (LOD0 brick coords). Auto-spaced: shift the scene's −X bound to `x_cursor`,
        // leaving Y/Z where the loader anchored them (floor at y=0, centred on Z). Explicit: use it verbatim.
        let offset = match scene.offset {
            Some(o) => o,
            None => IVec3::new(x_cursor_bricks - bc_lo.x, 0, 0),
        };

        // PALETTE merge: this scene's SOLID blocks are APPENDED after the merged registry's current blocks. The
        // first appended block (the scene's local block 1) lands at merged index `palette_base = merged.len()`
        // (AIR occupies index 0, so a non-empty merged registry has `len >= 1`). So the per-voxel BlockId SHIFT
        // is `palette_base - 1`: local block `b` (`b >= 1`) → merged id `(palette_base - 1) + b` = the index it
        // was appended to. (`merged_registry` is never empty here — it's at least AIR-only, `len == 1`, giving
        // shift 0 for the first scene, so its block 1 stays merged id 1.)
        let palette_base = merged_registry.len() as u32; // merged index the scene's local block 1 lands at
        let scene_solid_blocks = scene.registry.len().saturating_sub(1) as u32; // exclude AIR
        // BlockId is u16: the LAST appended block lands at merged index `palette_base + scene_solid_blocks - 1`
        // (== the shift `palette_base - 1` plus the scene's highest local id `scene_solid_blocks`). If that
        // exceeds u16::MAX, this scene (and any after it) can't be represented — cap + log + skip rather than
        // wrap a BlockId into a wrong colour.
        let highest_merged_id = palette_base + scene_solid_blocks - 1;
        if highest_merged_id > u16::MAX as u32 {
            warn!(
                "gallery: '{}' would push the merged palette past u16::MAX BlockIds \
                 (base {palette_base} + {scene_solid_blocks} blocks ⇒ highest id {highest_merged_id}) — capping \
                 the gallery here, skipping it and any later scenes",
                scene.label
            );
            break;
        }

        // Concatenate this scene's palette into the merged registry (its block i+1 → merged index palette_base+i).
        merge_registry(&mut merged_registry, &scene.registry);
        // Re-key + remap every brick of the scene into the merged map at `offset`, with its BlockIds shifted by
        // `palette_base - 1` so a local block `b` lands at the merged index it was appended to.
        merge_brickmap_into(&mut merged_map, &scene.map, offset, (palette_base - 1) as u16);

        // Advance the +X auto-spacing cursor past this scene's placed +X bound plus the inter-scene gap, so the
        // NEXT auto-spaced scene clears this one (and any explicitly-placed one) with a guaranteed gap.
        let placed_hi_x = bc_hi.x + offset.x;
        x_cursor_bricks = placed_hi_x + 1 + GALLERY_GAP_BRICKS;
    }

    (merged_map, merged_registry)
}

/// The inclusive LOD0 brick-coord AABB `(lo, hi)` of a [`BrickMap`] (every stored brick is non-empty, so its
/// coord bounds the solids). `None` iff the map is empty. The SSOT the auto-spacer + the side-by-side merge
/// use to measure a scene's extent.
fn brickmap_brick_bounds(map: &BrickMap) -> Option<(IVec3, IVec3)> {
    let mut lo = IVec3::splat(i32::MAX);
    let mut hi = IVec3::splat(i32::MIN);
    let mut any = false;
    for (bc, _brick) in map.iter() {
        lo = lo.min(*bc);
        hi = hi.max(*bc);
        any = true;
    }
    any.then_some((lo, hi))
}

/// Concatenate `scene`'s solid blocks onto `merged` so the scene keeps its OWN colours: the scene's block
/// `i+1` (its palette entry `i`) becomes merged block `merged.len() + i` (== `palette_base + i`, the same base
/// the caller computed BEFORE this push). AIR (block 0) is shared and never copied. After this, a scene voxel
/// carrying local [`BlockId`] `b` (`b >= 1`) reads the right colour at merged id `palette_base + b` (note
/// `palette_base = merged.len()` pre-push, so local `1` → first appended block). Preserves every per-scene
/// material attribute (colour, roughness, emissive, …) via [`BlockRegistry::extend_blocks_from`].
fn merge_registry(merged: &mut BlockRegistry, scene: &BlockRegistry) {
    merged.extend_blocks_from(scene);
}

/// Re-key + remap every brick of `scene` into `merged` at the LOD0-brick `offset`, shifting each solid voxel's
/// [`BlockId`] by `palette_base` so it indexes the concatenated merged palette. Because the gallery guarantees
/// each scene occupies a DISJOINT brick region (auto-spacing + the gap), a scene's bricks never collide with
/// another's in `merged` — so this is a straight insert per brick (no per-voxel blending). AIR voxels stay AIR
/// (the +base shift applies only to solids — AIR's merged id is still 0). Uniform-solid bricks are remapped
/// cheaply (one block id); dense bricks are remapped per voxel. Empty bricks are never stored (the map drops
/// them), matching the loader.
fn merge_brickmap_into(merged: &mut BrickMap, scene: &BrickMap, offset: IVec3, palette_base: u16) {
    for (bc, brick) in scene.iter() {
        let dst = *bc + offset;
        // Rebuild the brick with each solid voxel's BlockId shifted by palette_base. AIR is left as AIR (id 0)
        // so the brick's occupancy/empty invariants are unchanged — only solid ids move into the merged range.
        let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                for x in 0..BRICK_EDGE {
                    let b = brick.get(x, y, z);
                    if !b.is_air() {
                        voxels[voxel_index(x, y, z)] = BlockId(b.0 + palette_base);
                    }
                }
            }
        }
        merged.insert(dst, Brick::from_voxels(voxels));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::brickmap::brick_coord_of_voxel;
    use rustc_hash::FxHashMap;

    /// Build a fine `BrickMap` of a single solid block over a world-voxel AABB `[lo, hi)`.
    fn solid_box(lo: IVec3, hi: IVec3, block: BlockId) -> BrickMap {
        let mut dense: FxHashMap<IVec3, Box<[BlockId; BRICK_VOXELS]>> = FxHashMap::default();
        for z in lo.z..hi.z {
            for y in lo.y..hi.y {
                for x in lo.x..hi.x {
                    let wv = IVec3::new(x, y, z);
                    let bc = brick_coord_of_voxel(wv);
                    let local = wv - bc * BRICK_EDGE;
                    let arr = dense.entry(bc).or_insert_with(|| Box::new([BlockId::AIR; BRICK_VOXELS]));
                    arr[voxel_index(local.x, local.y, local.z)] = block;
                }
            }
        }
        let mut map = BrickMap::new();
        for (c, arr) in dense {
            map.insert(c, Brick::from_voxels(arr));
        }
        map
    }

    /// All solid (world-voxel, BlockId) cells of a map (for exact merge assertions).
    fn solids(map: &BrickMap) -> Vec<(IVec3, BlockId)> {
        let mut out = Vec::new();
        for (bc, brick) in map.iter() {
            for z in 0..BRICK_EDGE {
                for y in 0..BRICK_EDGE {
                    for x in 0..BRICK_EDGE {
                        if brick.is_solid(x, y, z) {
                            let wv = *bc * BRICK_EDGE + IVec3::new(x, y, z);
                            out.push((wv, brick.get(x, y, z)));
                        }
                    }
                }
            }
        }
        out
    }

    /// `merge_brickmap_into` places a scene at its brick offset and shifts its BlockIds by the palette base:
    /// the merged map carries the scene's voxels at `world + offset·8` with id `local + base`, and AIR stays AIR.
    #[test]
    fn merge_shifts_coords_and_remaps_blocks() {
        let scene = solid_box(IVec3::ZERO, IVec3::new(8, 8, 8), BlockId(3));
        let mut merged = BrickMap::new();
        let offset = IVec3::new(5, 0, 0); // 5 bricks in +X
        merge_brickmap_into(&mut merged, &scene, offset, 10);

        // Every solid voxel moved by offset·8 and its id bumped by 10.
        for (wv, b) in solids(&scene) {
            let moved = wv + offset * BRICK_EDGE;
            assert_eq!(merged.voxel_block(moved), BlockId(b.0 + 10), "voxel {wv:?} → {moved:?} id+10");
        }
        // The original (unshifted) location is air in the merged map (the scene was moved, not copied).
        assert!(!merged.voxel_is_solid(IVec3::new(0, 0, 0)), "the scene's original origin is empty after the shift");
        assert_eq!(merged.len(), scene.len(), "brick count is preserved (one disjoint region)");
    }

    /// Two synthetic scenes merge SIDE BY SIDE with no cross-scene brick overlap, BOTH present at their offsets,
    /// per-scene palettes preserved (each scene's blocks land in its own merged-id range), and the voxel counts
    /// add up. The core merge guarantee (mirrors the integration `vox::merge` test the task asks for, at the
    /// map level).
    #[test]
    fn two_scenes_merge_side_by_side_no_overlap() {
        // Scene A: a 16³-voxel block of id 1 at the origin (2×2×2 bricks).
        let a = solid_box(IVec3::ZERO, IVec3::splat(16), BlockId(1));
        // Scene B: a 16³-voxel block of id 1 at the origin too — but it will be placed past A with a gap.
        let b = solid_box(IVec3::ZERO, IVec3::splat(16), BlockId(1));

        let (a_lo, a_hi) = brickmap_brick_bounds(&a).unwrap();
        assert_eq!(a_lo, IVec3::ZERO);
        assert_eq!(a_hi, IVec3::splat(1)); // 2×2×2 bricks ⇒ coords 0..=1

        let mut merged = BrickMap::new();
        // A at the origin, BlockId shift 0 (ids stay 1..). Then B placed past A's +X bound + a gap, with its
        // ids shifted by `a_blocks` (this exercises `merge_brickmap_into`'s raw shift directly — `merge_scenes`
        // computes the shift as `palette_base - 1`; here we pass the shift explicitly).
        merge_brickmap_into(&mut merged, &a, IVec3::ZERO, 0);
        let a_blocks = 4u16; // pretend scene A contributed 4 solid blocks ⇒ B's ids shift by 4
        let b_offset_x = a_hi.x + 1 + GALLERY_GAP_BRICKS; // first free column past A + the gap
        merge_brickmap_into(&mut merged, &b, IVec3::new(b_offset_x, 0, 0), a_blocks);

        // Both scenes present: A's voxel at (0,0,0) is id 1; B's first voxel sits at brick column b_offset_x
        // with its id remapped to 1 + a_blocks.
        assert_eq!(merged.voxel_block(IVec3::new(0, 0, 0)), BlockId(1), "scene A present at its origin");
        let b_origin = IVec3::new(b_offset_x * BRICK_EDGE, 0, 0);
        assert_eq!(merged.voxel_block(b_origin), BlockId(1 + a_blocks), "scene B present at its offset, id remapped");

        // No cross-scene brick overlap: A occupies brick x∈[0,1]; B occupies brick x∈[b_offset_x, b_offset_x+1].
        // With GALLERY_GAP_BRICKS ≥ 1 those ranges are disjoint with a gap between them.
        assert!(b_offset_x > a_hi.x + 1, "B starts strictly past A's +X bound with a gap");
        // Brick count adds up: 8 (A: 2³) + 8 (B: 2³) = 16, none lost to a collision.
        assert_eq!(merged.len(), a.len() + b.len(), "no bricks lost — disjoint regions sum");
        assert_eq!(merged.len(), 16);
    }

    /// `load_gallery` is robust to a MISSING asset: a row pointing at a non-existent `.vox` is skipped with a
    /// warn (no panic), and the remaining scenes still merge. With ALL rows missing the result is empty +
    /// AIR-only (never a hard fail).
    #[test]
    fn load_gallery_skips_missing_assets() {
        let scenes = [
            GalleryEntry { vox_path: "assets/models/__does_not_exist_a.vox", vxo_path: "assets/models/__does_not_exist_a.vxo", offset: None, label: "MissingA" },
            GalleryEntry { vox_path: "assets/models/__does_not_exist_b.vox", vxo_path: "assets/models/__does_not_exist_b.vxo", offset: None, label: "MissingB" },
        ];
        let (map, reg) = load_gallery(&scenes);
        assert!(map.is_empty(), "all rows missing ⇒ empty merged map (no panic)");
        assert_eq!(reg.len(), 1, "only AIR in the merged registry when nothing loaded");
    }

    /// `vxo_gallery_placements` SKIPS absent `.vxo` assets (no panic) and returns an EMPTY list when none exist —
    /// the signal the live path uses to fall back to the legacy `.vox` merge. (With the corpus baked in this
    /// checkout the shipped table yields 3 placements; absent everywhere it's empty — both are non-panicking.)
    #[test]
    fn vxo_placements_skip_absent_and_signal_fallback() {
        let missing = [
            GalleryEntry { vox_path: "x.vox", vxo_path: "assets/models/__nope_a.vxo", offset: None, label: "A" },
            GalleryEntry { vox_path: "y.vox", vxo_path: "assets/models/__nope_b.vxo", offset: None, label: "B" },
        ];
        assert!(
            vxo_gallery_placements(&missing).is_empty(),
            "all `.vxo` absent ⇒ empty placement list (caller falls back to the `.vox` path) — no panic"
        );

        // The shipped table is auto-spaced: every placement's +X offset is non-decreasing (assets march along +X
        // past the previous one). This holds whether or not the assets are baked (absent ones are simply skipped).
        let placed = vxo_gallery_placements(GALLERY_SCENES);
        let mut last_x = i32::MIN;
        for (_path, off) in &placed {
            assert!(off.x >= last_x, "auto-spaced placements march monotonically along +X");
            last_x = off.x;
        }
    }

    /// The shipped `GALLERY_SCENES` table is well-formed and loads through `load_gallery` without panicking
    /// regardless of whether the assets are baked in this checkout: if `sponza.vox` exists the merge is
    /// non-empty with a populated palette; if not, it's the empty/AIR-only result (skipped with a warn). Either
    /// way the data-driven path is exercised end-to-end.
    #[test]
    fn shipped_gallery_table_loads_or_skips() {
        let (map, reg) = load_gallery(GALLERY_SCENES);
        let sponza_present = std::path::Path::new(super::super::raytrace::SPONZA_VOX_PATH).exists();
        if sponza_present {
            assert!(!map.is_empty(), "with sponza.vox baked, the gallery merges a non-empty map");
            assert!(reg.len() > 1, "the merged registry carries Sponza's palette");
        } else {
            assert!(map.is_empty(), "with no assets baked, the gallery is empty (skipped with a warn)");
            assert_eq!(reg.len(), 1, "AIR-only registry when nothing loaded");
        }
    }
}
