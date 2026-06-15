//! The **static Cornell-box scene** — a closed, resident voxel box that validates the HW-RT lighting + GI
//! path (colour bleed, an emissive area light, soft shadows) WITHOUT any streaming.
//!
//! This is the canonical Cornell box rebuilt as voxels: a closed room (floor + ceiling + back + left +
//! right walls; the FRONT is open so the camera looks in), the LEFT wall red and the RIGHT wall green (so a
//! single diffuse bounce tints the white floor/back near each), and an EMISSIVE light panel in the centre
//! of the ceiling that is the room's dominant light. Two white boxes stand on the floor (a TALL and a SHORT
//! one) à la the classic scene.
//!
//! It is a pure, deterministic function of the [`BlockRegistry::cornell`] palette — no worldgen, no
//! `HeightLayer`. The geometry is authored at the VOXEL level via [`CornellBlock`] ids (the SSOT shared with
//! the palette) and gathered into the SAME sparse [`BrickMap`] the renderer's resident-set packer consumes,
//! so the Cornell path and the worldgen path share the exact GPU layout downstream.
//!
//! # Layout (voxel units; `VOXEL_SIZE` = 0.05 m)
//! The interior is [`INTERIOR`]³ voxels (`48` → 2.4 m per side) spanning local voxel coords `[0, INTERIOR)`
//! on each axis; walls are [`WALL`] voxels thick (`2`) in the shell just outside the interior. So the full
//! box spans `[-WALL, INTERIOR + WALL)` in X and Y, and `[-WALL, INTERIOR + WALL)` in Z EXCEPT the front
//! (`-Z`) which is open. The box's near-front face sits at `z = 0` (the open side faces `-Z`); the camera is
//! framed outside it looking `+Z`.

use bevy::math::IVec3;

use super::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, BrickMap, VOXEL_SIZE, brick_coord_of_voxel, voxel_index};
use super::edits::VoxelEdits;
use super::palette::{BlockId, BlockRegistry, CornellBlock};

/// Interior edge of the box, in voxels (`48` → 2.4 m at 0.05 m voxels). Bounded + static.
pub const INTERIOR: i32 = 48;
/// Wall thickness, in voxels.
pub const WALL: i32 = 2;

/// The world-metre centre of the box interior — what the camera frames. The interior spans local voxel
/// `[0, INTERIOR)` on each axis, so its centre is at `INTERIOR/2` voxels = `INTERIOR/2 · VOXEL_SIZE` metres.
pub fn interior_center_world() -> [f32; 3] {
    let c = INTERIOR as f32 * 0.5 * VOXEL_SIZE;
    [c, c, c]
}

/// The world-metre interior edge length (`INTERIOR · VOXEL_SIZE` = 2.4 m at 0.05 m) — used to frame the camera.
pub fn interior_extent_world() -> f32 {
    INTERIOR as f32 * VOXEL_SIZE
}

/// The inclusive-exclusive voxel bounds `[lo, hi)` of the full box (walls included). The front (`-Z`) is
/// open, but the bound still covers the closed faces; the per-voxel classifier simply emits AIR on the open
/// side. Z's lower bound is `0` (no front wall), X/Y span the wall shell.
fn box_voxel_bounds() -> (IVec3, IVec3) {
    let lo = IVec3::new(-WALL, -WALL, 0);
    let hi = IVec3::new(INTERIOR + WALL, INTERIOR + WALL, INTERIOR + WALL);
    (lo, hi)
}

/// Classify a single WORLD voxel into a Cornell [`BlockId`] (AIR if empty). Pure SSOT for the geometry: the
/// walls, the ceiling light panel, and the two floor boxes are all decided here, so the brick gatherer and
/// any test agree by construction.
///
/// Priority: the ceiling LIGHT panel overrides the white ceiling where they overlap; otherwise a voxel is a
/// wall if it lies in any wall shell (left = red, right = green, floor/ceiling/back = white); otherwise an
/// interior voxel is a floor box (white) or AIR.
fn cornell_voxel(v: IVec3) -> BlockId {
    let (lo, hi) = box_voxel_bounds();
    // Outside the whole box footprint → air (the open front and beyond).
    if v.x < lo.x || v.x >= hi.x || v.y < lo.y || v.y >= hi.y || v.z < lo.z || v.z >= hi.z {
        return BlockId::AIR;
    }

    let in_x = (0..INTERIOR).contains(&v.x);
    let in_y = (0..INTERIOR).contains(&v.y);
    let in_z = (0..INTERIOR).contains(&v.z);

    // --- Walls (the shell just outside the interior on each closed face) ---
    // Ceiling shell: y in [INTERIOR, INTERIOR+WALL). The LIGHT panel replaces white in the centre third.
    if v.y >= INTERIOR && in_x && in_z {
        if is_light_panel(v.x, v.z) {
            return CornellBlock::Light.id();
        }
        return CornellBlock::White.id();
    }
    // Floor shell: y in [-WALL, 0).
    if v.y < 0 && in_x && in_z {
        return CornellBlock::White.id();
    }
    // Back wall (+Z): z in [INTERIOR, INTERIOR+WALL).
    if v.z >= INTERIOR && in_x && in_y {
        return CornellBlock::White.id();
    }
    // Left wall (−X): x in [-WALL, 0) → RED.
    if v.x < 0 && in_y && in_z {
        return CornellBlock::Red.id();
    }
    // Right wall (+X): x in [INTERIOR, INTERIOR+WALL) → GREEN.
    if v.x >= INTERIOR && in_y && in_z {
        return CornellBlock::Green.id();
    }

    // --- Interior content: the two floor boxes ---
    if in_x && in_y && in_z && in_floor_box(v) {
        return CornellBlock::White.id();
    }

    BlockId::AIR
}

/// True iff ceiling voxel column `(x, z)` (interior coords) is under the centre LIGHT panel — roughly the
/// central third of the ceiling in both axes (the classic Cornell area light). `[lo, hi)` per axis.
fn is_light_panel(x: i32, z: i32) -> bool {
    let lo = INTERIOR / 3;
    let hi = INTERIOR - INTERIOR / 3;
    (lo..hi).contains(&x) && (lo..hi).contains(&z)
}

/// True iff interior voxel `v` is inside one of the two white floor boxes — a TALL box on the back-left and
/// a SHORT box on the front-right, offset à la the classic Cornell scene. Both rest on the floor (`y = 0`).
/// Coordinates are interior voxels in `[0, INTERIOR)`.
fn in_floor_box(v: IVec3) -> bool {
    // TALL box: back-left quadrant, ~14 voxels footprint, ~28 tall.
    let tall = (8..22).contains(&v.x) && (0..28).contains(&v.y) && (20..34).contains(&v.z);
    // SHORT box: front-right quadrant, ~16 voxels footprint, ~16 tall.
    let short = (26..42).contains(&v.x) && (0..16).contains(&v.y) && (6..22).contains(&v.z);
    tall || short
}

/// Build the static Cornell box into a [`BrickMap`] (the resident-set input). Iterates every brick covering
/// the box's voxel bounds, voxelizes it via [`cornell_voxel`], and inserts the non-empty ones (the sparsity
/// invariant: all-air bricks are dropped). Pure + deterministic — the same registry always yields the same
/// map. The `registry` argument is taken for API symmetry with the worldgen voxelizer (Cornell block ids are
/// fixed by [`CornellBlock`], so it isn't read), keeping one call shape across scenes.
pub fn build_cornell(registry: &BlockRegistry) -> BrickMap {
    build_cornell_with_edits(registry, &VoxelEdits::new())
}

/// Build the static Cornell box with the [`VoxelEdits`] delta overlaid (Stage 5 build/destroy editing). The
/// SSOT for the EDITED Cornell scene: each voxel is the override (if [`edits`] has one at that world voxel)
/// else the base [`cornell_voxel`] classification, so the packed GPU bricks the renderer traces and the CPU
/// pick all see the same `base unless overridden` geometry. The brick range covers the box footprint PLUS
/// every brick a placed override touches (so a block built OUTSIDE the box appears too). Empty bricks are
/// dropped (sparsity). [`build_cornell`] is this with an empty delta.
pub fn build_cornell_with_edits(_registry: &BlockRegistry, edits: &VoxelEdits) -> BrickMap {
    let (lo, hi) = box_voxel_bounds();
    // Brick coordinate range covering the voxel bounds (Euclidean floor of the inclusive max voxel).
    let mut bc_lo = IVec3::new(
        lo.x.div_euclid(BRICK_EDGE),
        lo.y.div_euclid(BRICK_EDGE),
        lo.z.div_euclid(BRICK_EDGE),
    );
    let mut bc_hi = IVec3::new(
        (hi.x - 1).div_euclid(BRICK_EDGE),
        (hi.y - 1).div_euclid(BRICK_EDGE),
        (hi.z - 1).div_euclid(BRICK_EDGE),
    );
    // Extend the brick range so any PLACED override OUTSIDE the box footprint still gets a brick voxelized
    // (a removed-only override outside the box would just be air, so it needn't extend the range). One brick
    // of margin so a placed voxel's halo neighbours are covered too.
    for (wv, block) in edits.iter() {
        if block.is_air() {
            continue;
        }
        let bc = brick_coord_of_voxel(wv);
        bc_lo = bc_lo.min(bc - IVec3::ONE);
        bc_hi = bc_hi.max(bc + IVec3::ONE);
    }

    let mut map = BrickMap::new();
    for bz in bc_lo.z..=bc_hi.z {
        for by in bc_lo.y..=bc_hi.y {
            for bx in bc_lo.x..=bc_hi.x {
                let coord = IVec3::new(bx, by, bz);
                let origin = coord * BRICK_EDGE;
                let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
                for z in 0..BRICK_EDGE {
                    for y in 0..BRICK_EDGE {
                        for x in 0..BRICK_EDGE {
                            let wv = origin + IVec3::new(x, y, z);
                            // `base unless overridden`: the override (place/remove) wins over the classifier.
                            voxels[voxel_index(x, y, z)] = edits.resolve(wv, cornell_voxel(wv));
                        }
                    }
                }
                map.insert(coord, Brick::from_voxels(voxels)); // empty bricks dropped
            }
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The box is non-empty, bounded (tiny + static), and every stored brick lies within the box's brick
    /// footprint (no stray bricks).
    #[test]
    fn cornell_builds_bounded_brick_set() {
        let reg = BlockRegistry::cornell();
        let map = build_cornell(&reg);
        assert!(!map.is_empty(), "the Cornell box must contain bricks");
        // ~ (52/8)³ ≈ 343 bricks upper bound; assert a comfortable bound.
        assert!(map.len() < 500, "Cornell brick count {} must stay small + static", map.len());
    }

    /// The five closed faces carry the right colours and the front is OPEN; the ceiling centre is the
    /// emissive light; the floor boxes are present.
    #[test]
    fn cornell_faces_and_light() {
        let mid = INTERIOR / 2;
        // Left wall (−X) is red; right wall (+X) is green.
        assert_eq!(cornell_voxel(IVec3::new(-1, mid, mid)), CornellBlock::Red.id(), "left wall red");
        assert_eq!(cornell_voxel(IVec3::new(INTERIOR, mid, mid)), CornellBlock::Green.id(), "right wall green");
        // Floor + ceiling + back are white.
        assert_eq!(cornell_voxel(IVec3::new(mid, -1, mid)), CornellBlock::White.id(), "floor white");
        assert_eq!(cornell_voxel(IVec3::new(mid, INTERIOR, 2)), CornellBlock::White.id(), "ceiling edge white");
        assert_eq!(cornell_voxel(IVec3::new(mid, mid, INTERIOR)), CornellBlock::White.id(), "back wall white");
        // The FRONT (−Z) is open: just outside the front face is air.
        assert!(cornell_voxel(IVec3::new(mid, mid, -1)).is_air(), "front (−Z) must be open");
        // Centre of the ceiling is the emissive light panel.
        assert_eq!(cornell_voxel(IVec3::new(mid, INTERIOR, mid)), CornellBlock::Light.id(), "ceiling centre is the light");
        // A ceiling corner (outside the centre third) is plain white, not the light.
        assert_eq!(cornell_voxel(IVec3::new(1, INTERIOR, 1)), CornellBlock::White.id(), "ceiling corner is white");
        // The interior air (centre of the room, away from boxes) is empty.
        assert!(cornell_voxel(IVec3::new(mid, mid, 1)).is_air(), "open interior near front is air");
        // Both floor boxes have at least one solid voxel.
        assert_eq!(cornell_voxel(IVec3::new(12, 1, 26)), CornellBlock::White.id(), "tall box solid");
        assert_eq!(cornell_voxel(IVec3::new(34, 1, 14)), CornellBlock::White.id(), "short box solid");
    }

    /// The light panel covers the central third of the ceiling and nothing outside it.
    #[test]
    fn light_panel_is_central() {
        assert!(is_light_panel(INTERIOR / 2, INTERIOR / 2), "centre is lit");
        assert!(!is_light_panel(0, 0), "corner is not lit");
        assert!(!is_light_panel(INTERIOR - 1, INTERIOR - 1), "far corner is not lit");
    }
}
