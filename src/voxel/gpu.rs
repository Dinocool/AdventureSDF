//! The **single source of truth** for how a resident [`BrickMap`] patch is laid out in GPU storage for
//! the hardware-ray-traced voxel path. The CPU uploader (the render-world prepare system), the WGSL
//! raymarch shader, and the headless ray_query correctness test all consume THIS module's packing so they
//! can never drift: change the layout here and every consumer changes with it.
//!
//! # Layout
//!
//! A patch is uploaded as three parallel GPU storage buffers plus a palette buffer:
//!
//! - **AABB buffer** (`Vec<GpuBrickAabb>`): one procedural AABB per resident brick, in world metres. This
//!   is the BLAS geometry — `primitive_index` in the ray query indexes it. AABBs are the brick's world
//!   bounds at its LOD (`brick_coord · brick_span(lod) .. +brick_span(lod)`); a coarse brick covers more
//!   world (the clipmap span scales `2^lod`), so the AABB is NOT LOD-invariant.
//! - **Brick directory** (`Vec<GpuBrickMeta>`): parallel to the AABB buffer (same index = same brick).
//!   Each entry carries the brick's world-voxel origin and the offset (in `u32`s) into the voxel buffer
//!   where its [`halo_cells`] block ids start. The shader, given `primitive_index`, reads this to locate
//!   the brick's voxels and place them in world space.
//! - **Voxel buffer** (`Vec<u32>`): every resident brick's HALOED grid block ids — a `(lod_edge+2)³` block
//!   ([`halo_cells`]) with a 1-cell border on every side holding the adjacent NEIGHBOUR brick's boundary
//!   voxels (AIR where the neighbour is absent), one [`BlockId`] per `u32` (zero-extended `u16`), in
//!   [`halo_index`] order. Densely concatenated; a brick's slice begins at its directory `voxel_offset`.
//!   The halo is the robust brick-SEAM fix: it lets the in-shader DDA always cross a real air→solid cell
//!   boundary AT the true surface (even when the surface lies on a brick face), so the first-solid hit gets
//!   the correct entry-face normal from EVERY angle — killing the thin dark seam lines at oblique views.
//!   Cost: LOD0 stores `10³ = 1000` u32 vs the bare `8³ = 512` (~1.95×); a few MB at Cornell/patch scale.
//! - **Palette buffer** (`Vec<GpuPaletteColor>`): `BlockId(i)` → linear RGBA, indexed directly by block id.
//!
//! Every offset/stride below is derived from the [`brickmap`](super::brickmap) constants, so the brick
//! geometry constants live in exactly one place.

use bevy::math::IVec3;
use bytemuck::{Pod, Zeroable};

use super::brickmap::{BRICK_EDGE, BRICK_WORLD_SIZE, Brick, BrickMap, VOXEL_SIZE, brick_span, lod_edge};
use super::palette::{BlockId, BlockRegistry};

/// The STORED per-axis grid edge of a brick at LOD `lod`: the core grid ([`lod_edge`], a constant
/// [`BRICK_EDGE`] at every LOD) PLUS a 1-cell HALO border on every side (`core + 2` = 10). The packer fills
/// that border with the adjacent SAME-LOD NEIGHBOUR brick's
/// boundary voxels (AIR where the neighbour is absent), so the in-shader DDA always crosses a real air→solid
/// cell boundary AT the true surface — even when the surface lies exactly on a brick face. This is the
/// robust brick-seam fix (see the WGSL `halo_edge`): it gives the first-solid hit the correct entry-face
/// normal and an always-present boundary cell from EVERY direction, killing the thin dark seam lines at
/// oblique angles. SSOT shared by both packers and the shader.
#[inline]
pub fn halo_edge(lod: u32) -> i32 {
    lod_edge(lod) + 2
}

/// Number of `u32` block ids a haloed brick at LOD `lod` stores (`halo_edge³`).
#[inline]
pub fn halo_cells(lod: u32) -> usize {
    let h = halo_edge(lod) as usize;
    h * h * h
}

/// Linear index of a HALOED-grid local cell `(x,y,z)` in `[0, halo_edge(lod))` — +X fastest, then +Y, then
/// +Z (the same convention as [`voxel_index`], at the haloed edge). Halo index 0 / `halo_edge-1` are the
/// border ring; core cells are `[1, lod_edge]`. SSOT mirror of the shader's `cell_index(x,y,z,hedge)`.
#[inline]
pub fn halo_index(x: i32, y: i32, z: i32, lod: u32) -> usize {
    let h = halo_edge(lod);
    (x + y * h + z * h * h) as usize
}

/// How far each brick's BLAS AABB is GROWN, on every side, beyond its true world bounds, in world metres.
///
/// **The seam fix.** Bricks abut exactly (`brick.max == neighbour.min`), so the shared face/edge/corner is a
/// half-open boundary the procedural-AABB BLAS does not treat watertightly: a ray travelling *along* a shared
/// plane — or grazing a shared edge between four bricks — can be reported as intersecting NEITHER AABB,
/// producing the black "brick seam" lines. Growing every AABB by this epsilon makes neighbours OVERLAP, so
/// every shared boundary is strictly interior to at least one AABB and is always a BLAS candidate.
///
/// This ONLY enlarges the BLAS candidate volume — it does NOT move any voxel. The in-shader DDA reconstructs
/// cells from the brick's TRUE `world_min` and clamps the entry cell into `[0, edge)`, so a ray that enters
/// only the epsilon halo (and never the real brick) finds no solid cell there and the true-bounds slab test
/// in the shader rejects it. Chosen at `1e-3` of a voxel (~0.2 µm): far below sub-voxel precision yet well
/// above the FP tangency that causes the miss. SSOT: both packers call [`brick_aabb`] so the overlap is
/// defined exactly once.
pub const BRICK_AABB_EPSILON: f32 = VOXEL_SIZE * 1.0e-3;

/// The (epsilon-grown) BLAS AABB for a LOD-`lod` brick whose TRUE world-min corner is `world_min`. The true
/// extent is [`brick_span`]`(lod)` per axis (the clipmap span scales `2^lod`, so a coarse brick covers more
/// world); this grows it by [`BRICK_AABB_EPSILON`] on every side so abutting bricks overlap (the seam fix —
/// see that constant). The single place the BLAS AABB bounds are formed, shared by both packers so the
/// overlap rule (and the per-LOD span) never drifts from the WGSL `brick_span`.
#[inline]
pub fn brick_aabb(world_min: [f32; 3], lod: u32) -> GpuBrickAabb {
    let e = BRICK_AABB_EPSILON;
    let span = brick_span(lod);
    GpuBrickAabb {
        min: [world_min[0] - e, world_min[1] - e, world_min[2] - e],
        max: [world_min[0] + span + e, world_min[1] + span + e, world_min[2] + span + e],
        _pad: [0.0; 2],
    }
}

/// A procedural AABB for one brick, in world metres. Field layout is bit-identical to the proven
/// `GpuAabb` in `D:/spike-aabb` (`min[3], max[3]` + two `f32` pad → 32 bytes, the AABB stride the BLAS
/// build expects). `bytemuck`-uploadable.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuBrickAabb {
    /// World-metre minimum corner.
    pub min: [f32; 3],
    /// World-metre maximum corner.
    pub max: [f32; 3],
    /// Pad to 32 bytes (the AABB stride wgpu's BLAS AABB build reads).
    pub _pad: [f32; 2],
}

/// Per-brick metadata, parallel to the AABB buffer (index `i` describes the brick whose AABB is
/// `aabbs[i]` and whose `primitive_index` in the ray query is `i`). 32 bytes, `bytemuck`-uploadable.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuBrickMeta {
    /// The brick's world-VOXEL origin (its local `(0,0,0)` corner in world voxel coordinates) =
    /// `brick_coord · BRICK_EDGE`. The shader maps a world position to a local voxel via this.
    pub voxel_origin: [i32; 3],
    /// Offset (in `u32` elements) into the voxel buffer where this brick's voxel block ids begin. A brick
    /// stores [`halo_cells`]`(lod)` = `10³` ids (the `8³` core + 1-cell halo) at EVERY LOD (the grid is a
    /// constant `8³`; only the world span scales), so this stride is LOD-independent.
    pub voxel_offset: u32,
    /// The brick's world-metre minimum corner (= `aabbs[i].min`), duplicated here so the shader's DDA has
    /// the brick origin without a second buffer fetch. `world_min = coord · brick_span(lod)`.
    pub world_min: [f32; 3],
    /// The brick's LOD level. The grid is ALWAYS `8³` ([`lod_edge`]); the shader derives the per-cell world
    /// size ([`brick_span`]`(lod) / 8 = VOXEL_SIZE · 2^lod`) + the brick span (`brick_span(lod)`) from this,
    /// so a coarse brick is DDA-marched over the SAME `8³` grid covering `2^lod×` more world. Part of the
    /// SSOT — uploader, shader, and tests agree on it.
    pub lod: u32,
}

/// One palette entry: linear-RGBA albedo + linear-RGB emissive radiance. Indexed by `BlockId(i)`
/// directly. 32 bytes (`rgba` 16 + `emissive` 16; `emissive.w` is unused pad). Emissive is the per-block
/// glow the GI bounce treats as a light source — a non-zero `emissive` makes that block an emitter.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuPaletteColor {
    /// Linear RGBA albedo (block 0 = AIR is transparent black).
    pub rgba: [f32; 4],
    /// Linear-RGB emissive radiance in `.xyz` (scaled by `emissive_strength` in the shader); `.w` pad.
    pub emissive: [f32; 4],
}

/// The packed, GPU-ready representation of a resident [`BrickMap`] patch: the three parallel per-brick
/// buffers plus the palette. Built once by [`pack_brickmap`]; uploaded verbatim to storage buffers. The
/// ORDER of `aabbs`/`metas` defines each brick's `primitive_index` (= its position here) — the BLAS is
/// built from `aabbs`, so the indices line up by construction.
#[derive(Clone, Debug, Default)]
pub struct GpuBrickPatch {
    /// One AABB per brick (the BLAS geometry). `aabbs[i].primitive_index == i`.
    pub aabbs: Vec<GpuBrickAabb>,
    /// Per-brick metadata, parallel to `aabbs`.
    pub metas: Vec<GpuBrickMeta>,
    /// Concatenated per-voxel block ids (one `u32` each). `metas[i].voxel_offset` is brick `i`'s start.
    pub voxels: Vec<u32>,
    /// `BlockId(i)` → linear RGBA. Length == registry length.
    pub palette: Vec<GpuPaletteColor>,
}

impl GpuBrickPatch {
    /// Number of resident bricks (== BLAS primitive count == `aabbs.len()`).
    #[inline]
    pub fn brick_count(&self) -> usize {
        self.aabbs.len()
    }

    /// True iff no bricks are resident (an empty patch — nothing to trace).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.aabbs.is_empty()
    }
}

/// Pack a resident [`BrickMap`] + its [`BlockRegistry`] palette into GPU-ready buffers (the SSOT layout).
///
/// Iterates the map's stored bricks in a DETERMINISTIC order (sorted by brick coordinate) so the packing
/// — and therefore every brick's `primitive_index` — is reproducible run-to-run and matches what the
/// headless test asserts against. For each brick it appends its AABB, its metadata (origin + voxel
/// offset), and its `BRICK_VOXELS` block ids (one `u32` each, in [`voxel_index`] order). Empty bricks
/// never appear in the map, so every packed brick has at least one solid voxel.
pub fn pack_brickmap(map: &BrickMap, registry: &BlockRegistry) -> GpuBrickPatch {
    // Deterministic brick order: sort by (z, y, x) of the brick coordinate. The map is a hash map, so we
    // must impose an order or `primitive_index ↔ brick` would vary run-to-run (breaking the test oracle).
    let mut coords: Vec<_> = map.iter().map(|(c, _)| *c).collect();
    coords.sort_by_key(|c| (c.z, c.y, c.x));

    let mut patch = GpuBrickPatch {
        aabbs: Vec::with_capacity(coords.len()),
        metas: Vec::with_capacity(coords.len()),
        voxels: Vec::with_capacity(coords.len() * halo_cells(0)),
        palette: Vec::with_capacity(registry.len()),
    };

    let h = halo_edge(0); // LOD0 haloed edge (= BRICK_EDGE + 2)
    for coord in coords {
        let world_min = [
            coord.x as f32 * BRICK_WORLD_SIZE,
            coord.y as f32 * BRICK_WORLD_SIZE,
            coord.z as f32 * BRICK_WORLD_SIZE,
        ];
        // BLAS AABB is the brick's world extent GROWN by the seam epsilon (so abutting bricks overlap — see
        // `brick_aabb`). `pack_brickmap` is the static all-LOD0 path (Cornell), so the span is `brick_span(0)
        // == BRICK_WORLD_SIZE`. `world_min` stored in the meta stays the TRUE corner the DDA reconstructs from.
        patch.aabbs.push(brick_aabb(world_min, 0));

        let voxel_offset = patch.voxels.len() as u32;
        let voxel_origin = [coord.x * BRICK_EDGE, coord.y * BRICK_EDGE, coord.z * BRICK_EDGE];
        // LOD0 (full res) for the static patch packer — every brick keeps its 8³ core grid (+ halo).
        patch.metas.push(GpuBrickMeta { voxel_origin, voxel_offset, world_min, lod: 0 });

        // Append the brick's voxels in HALOED-grid order (+X fastest, then +Y, then +Z): the haloed grid is
        // `(BRICK_EDGE+2)³`, with halo index 0/`h-1` the border ring and core cells at `[1, BRICK_EDGE]`. The
        // border holds the NEIGHBOUR brick's adjacent voxel (read from the map via `voxel_block`; AIR where the
        // neighbour is absent), so the DDA sees a real air→solid crossing at the true surface. The brick's
        // world-voxel origin is `voxel_origin`; haloed local `(hx,hy,hz)` ↦ world voxel `origin + (h*-1)`.
        let origin = IVec3::new(voxel_origin[0], voxel_origin[1], voxel_origin[2]);
        for hz in 0..h {
            for hy in 0..h {
                for hx in 0..h {
                    debug_assert_eq!(patch.voxels.len() - voxel_offset as usize, halo_index(hx, hy, hz, 0));
                    let wv = origin + IVec3::new(hx - 1, hy - 1, hz - 1);
                    patch.voxels.push(map.voxel_block(wv).0 as u32);
                }
            }
        }
    }

    push_palette(&mut patch, registry);
    patch
}

/// One resident brick ready to pack: its `(coord, lod)` clipmap key + the voxelized brick. The streaming
/// layer ([`super::streaming`]) voxelizes each `(coord, lod)` DIRECTLY at its LOD spacing (a true in-place
/// mip — NOT a downsample of a finer brick), so the `8³` voxels are ALREADY at the right resolution; the
/// packer stores them verbatim. Produced in a DETERMINISTIC order; the packer preserves it so
/// `primitive_index ↔ brick` is stable (the test oracle relies on it).
pub struct ResidentBrick<'a> {
    /// Integer brick coordinate, on the LOD-`lod` grid (`world_min = coord · brick_span(lod)`).
    pub coord: IVec3,
    /// The brick's `8³` voxels, voxelized at LOD `lod` (already at the coarse spacing — packed as-is).
    pub brick: &'a Brick,
    /// The clipmap LOD level of this brick. Different LODs are different coord grids.
    pub lod: u32,
}

/// Pack a camera-following RESIDENT brick set (clipmap-keyed by `(coord, lod)`) into the SSOT GPU layout —
/// the streaming successor to [`pack_brickmap`]. Each entry's brick is ALREADY the `8³` grid at its LOD
/// (the voxelizer samples each `(coord, lod)` directly at its `lod_voxel_size(lod)` spacing — a true mip),
/// so the packer stores the `8³` core verbatim (no downsampling) plus the 1-cell halo (the seam fix). The
/// AABB is the brick's per-LOD world extent ([`brick_span`]`(lod)`, so a coarse brick covers `2^lod×` more
/// world). The empty bricks never reach here (the streaming layer drops all-air ones), so every entry is
/// emitted. The entry ORDER defines each brick's `primitive_index`, so the caller must pass a deterministic
/// order. The halo border reads the SAME-LOD neighbour at `(coord ± 1, lod)` from a shared map (one lookup,
/// no per-brick re-voxelize); an absent / different-LOD neighbour (a clipmap SHELL boundary) contributes
/// AIR — the conservative seam behaviour, which the AABB-overlap + nearest-hit DDA then resolve across the
/// LOD step (see the module / streaming docs on cross-LOD seams).
pub fn pack_resident_set(entries: &[ResidentBrick<'_>], registry: &BlockRegistry) -> GpuBrickPatch {
    use std::collections::HashMap;

    let h = halo_edge(0); // constant haloed edge (= BRICK_EDGE + 2 = 10) at every LOD
    let mut patch = GpuBrickPatch {
        aabbs: Vec::with_capacity(entries.len()),
        metas: Vec::with_capacity(entries.len()),
        voxels: Vec::with_capacity(entries.len() * halo_cells(0)),
        palette: Vec::with_capacity(registry.len()),
    };

    // Index every resident brick by its `(coord, lod)` clipmap key, so a brick's HALO border can read its
    // SAME-LOD neighbour's adjacent face voxel (the seam fix) with one map lookup. Keyed by `(coord, lod)`
    // because coords now OVERLAP across LOD grids — the same integer coord at two LODs is two different world
    // bricks, so the lod must be part of the key. A border whose neighbour is absent or at a DIFFERENT lod (a
    // shell boundary) falls back to AIR (the conservative pre-halo behaviour — no cross-LOD halo).
    let by_key: HashMap<(IVec3, u32), &Brick> =
        entries.iter().map(|e| ((e.coord, e.lod), e.brick)).collect();

    for e in entries {
        let lod = e.lod;
        let coord = e.coord;
        let span = brick_span(lod);
        let world_min = [coord.x as f32 * span, coord.y as f32 * span, coord.z as f32 * span];
        // BLAS AABB grown by the seam epsilon (overlapping neighbours); the meta keeps the TRUE `world_min`.
        patch.aabbs.push(brick_aabb(world_min, lod));

        let voxel_offset = patch.voxels.len() as u32;
        let voxel_origin = [coord.x * BRICK_EDGE, coord.y * BRICK_EDGE, coord.z * BRICK_EDGE];
        patch.metas.push(GpuBrickMeta { voxel_origin, voxel_offset, world_min, lod });

        for hz in 0..h {
            for hy in 0..h {
                for hx in 0..h {
                    debug_assert_eq!(patch.voxels.len() - voxel_offset as usize, halo_index(hx, hy, hz, lod));
                    // Core cells are halo index [1, BRICK_EDGE]; halo index 0 / h-1 is the 1-cell border ring.
                    let cx = hx - 1;
                    let cy = hy - 1;
                    let cz = hz - 1;
                    let in_core =
                        (0..BRICK_EDGE).contains(&cx) && (0..BRICK_EDGE).contains(&cy) && (0..BRICK_EDGE).contains(&cz);
                    let block = if in_core {
                        e.brick.get(cx, cy, cz)
                    } else {
                        // A border cell: resolve the SAME-LOD neighbour brick + the wrapped voxel inside it.
                        neighbour_border_cell(&by_key, coord, lod, IVec3::new(cx, cy, cz))
                    };
                    patch.voxels.push(block.0 as u32);
                }
            }
        }
    }

    push_palette(&mut patch, registry);
    patch
}

/// Resolve one HALO BORDER cell at local voxel coordinate `cc` (outside `[0, BRICK_EDGE)` on ≥1 axis) for the
/// brick at `(coord, lod)`: find the SAME-LOD neighbour brick that owns the wrapped voxel and return that
/// voxel. Returns AIR when the owning neighbour is absent or at a DIFFERENT LOD (a clipmap shell boundary) —
/// so a border with no same-LOD neighbour is air, the conservative pre-halo behaviour the AABB-overlap +
/// nearest-hit DDA then resolve across the LOD step (no cross-LOD halo by design).
fn neighbour_border_cell(
    by_key: &std::collections::HashMap<(IVec3, u32), &Brick>,
    coord: IVec3,
    lod: u32,
    cc: IVec3,
) -> BlockId {
    // The neighbour brick coordinate = `coord` shifted by which face(s) `cc` overflows; the wrapped voxel
    // inside the neighbour is `cc mod BRICK_EDGE` (Euclidean, so −1 ↦ BRICK_EDGE−1). Same-LOD by construction.
    let nbr = coord
        + IVec3::new(cc.x.div_euclid(BRICK_EDGE), cc.y.div_euclid(BRICK_EDGE), cc.z.div_euclid(BRICK_EDGE));
    let Some(brick) = by_key.get(&(nbr, lod)) else {
        return BlockId::AIR;
    };
    let lx = cc.x.rem_euclid(BRICK_EDGE);
    let ly = cc.y.rem_euclid(BRICK_EDGE);
    let lz = cc.z.rem_euclid(BRICK_EDGE);
    brick.get(lx, ly, lz)
}

/// Append the palette buffer: `BlockId(i)` → linear RGBA, indexed directly (block 0 = AIR). Shared by both
/// packers so the palette chain has one SSOT.
fn push_palette(patch: &mut GpuBrickPatch, registry: &BlockRegistry) {
    for i in 0..registry.len() {
        let id = BlockId(i as u16);
        let e = registry.emissive(id);
        patch.palette.push(GpuPaletteColor {
            rgba: registry.color(id),
            emissive: [e[0], e[1], e[2], 0.0],
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_render::worldgen::biome::{
        BiomeDef, BiomeId, BiomeLibrary, StrataLayer, TerrainMatId, TerrainSurfaceMaterial,
    };
    use bevy::math::IVec3;

    use super::super::brickmap::{BRICK_VOXELS, Brick, voxel_index};

    /// A tiny registry + a small hand-built brick map for the packing tests.
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

    /// A brick with a single solid voxel at local `(x,y,z)` of block `id`, the rest air. Returns the brick
    /// and the HALOED-grid index of that voxel (core cell `(x,y,z)` ↦ halo index `(x+1,y+1,z+1)`) for
    /// cross-checking the packed buffer.
    fn one_voxel_brick(x: i32, y: i32, z: i32, id: BlockId) -> (Brick, usize) {
        let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
        voxels[voxel_index(x, y, z)] = id;
        (Brick::from_voxels(voxels), halo_index(x + 1, y + 1, z + 1, 0))
    }

    /// Packing produces parallel AABB/meta arrays of length == brick count, a voxel buffer of
    /// `brick_count · halo_cells(0)` u32s (each brick is a haloed `10³` grid), and a palette of
    /// `registry.len()`. The per-brick voxel slice starts at the recorded offset and reproduces the brick's
    /// block ids in haloed-grid order.
    #[test]
    fn pack_layout_is_consistent() {
        let reg = registry();
        let mut map = BrickMap::new();
        let (b0, i0) = one_voxel_brick(1, 2, 3, BlockId(1));
        let (b1, i1) = one_voxel_brick(4, 5, 6, BlockId(2));
        map.insert(IVec3::new(0, 0, 0), b0);
        map.insert(IVec3::new(1, 0, 0), b1);

        let patch = pack_brickmap(&map, &reg);
        assert_eq!(patch.brick_count(), 2);
        assert_eq!(patch.aabbs.len(), patch.metas.len());
        assert_eq!(patch.voxels.len(), 2 * halo_cells(0));
        assert_eq!(patch.palette.len(), reg.len());

        // Deterministic order: sorted by (z,y,x) → brick (0,0,0) then (1,0,0).
        assert_eq!(patch.metas[0].voxel_origin, [0, 0, 0]);
        assert_eq!(patch.metas[1].voxel_origin, [BRICK_EDGE, 0, 0]);
        assert_eq!(patch.metas[0].voxel_offset, 0);
        assert_eq!(patch.metas[1].voxel_offset, halo_cells(0) as u32);

        // The solid voxel of each brick lands at its haloed index within its slice, with the right id.
        assert_eq!(patch.voxels[patch.metas[0].voxel_offset as usize + i0], 1);
        assert_eq!(patch.voxels[patch.metas[1].voxel_offset as usize + i1], 2);

        // AABB bounds match the LOD0 brick world extent GROWN by the seam epsilon (overlapping neighbours).
        assert_eq!(patch.aabbs[0], brick_aabb([0.0, 0.0, 0.0], 0));
        assert_eq!(patch.aabbs[1], brick_aabb([BRICK_WORLD_SIZE, 0.0, 0.0], 0));
        // The grow makes neighbours OVERLAP: brick 1's min.x is below brick 0's max.x (no gap → no seam).
        assert!(patch.aabbs[1].min[0] < patch.aabbs[0].max[0], "abutting bricks' AABBs must overlap");
    }

    /// The palette buffer mirrors the registry colour for every block id (the SSOT colour chain).
    #[test]
    fn palette_mirrors_registry() {
        let reg = registry();
        let map = BrickMap::new();
        let patch = pack_brickmap(&map, &reg);
        for i in 0..reg.len() {
            let id = BlockId(i as u16);
            assert_eq!(patch.palette[i].rgba, reg.color(id));
            let e = reg.emissive(id);
            assert_eq!(patch.palette[i].emissive, [e[0], e[1], e[2], 0.0]);
        }
    }

    /// A brick fully solid with block `id` (uniform — used for LOD packing tests).
    fn solid_brick(id: BlockId) -> Brick {
        Brick::uniform(id)
    }

    /// `pack_resident_set` at LOD0 reproduces `pack_brickmap`'s layout for the same bricks (same order,
    /// same offsets, lod==0), and the per-brick voxel slice is the full HALOED `10³` grid. The two solid
    /// uniform bricks are adjacent, so each fills the other's shared-face halo with solid (every haloed cell
    /// on that face is the neighbour's voxel) — but the two bricks here are NOT adjacent in every direction,
    /// so the far halo faces are air; we only assert the CORE cells are solid.
    #[test]
    fn resident_lod0_matches_full_res() {
        let reg = registry();
        let b0 = solid_brick(BlockId(1));
        let b1 = solid_brick(BlockId(2));
        let entries = vec![
            ResidentBrick { coord: IVec3::new(0, 0, 0), brick: &b0, lod: 0 },
            ResidentBrick { coord: IVec3::new(1, 0, 0), brick: &b1, lod: 0 },
        ];
        let patch = pack_resident_set(&entries, &reg);
        assert_eq!(patch.brick_count(), 2);
        assert_eq!(patch.metas[0].lod, 0);
        assert_eq!(patch.metas[0].voxel_offset, 0);
        assert_eq!(patch.metas[1].voxel_offset, halo_cells(0) as u32);
        assert_eq!(patch.voxels.len(), 2 * halo_cells(0));
        // Brick 0 is uniform block 1 — every CORE cell reads 1 (halo cells may be 0 where no neighbour).
        for z in 1..=BRICK_EDGE {
            for y in 1..=BRICK_EDGE {
                for x in 1..=BRICK_EDGE {
                    assert_eq!(patch.voxels[halo_index(x, y, z, 0)], 1);
                }
            }
        }
    }

    /// A coarse brick is the SAME haloed `10³` grid (the clipmap keeps resolution constant); what changes is
    /// its world span + per-cell size. The meta records the LOD, and the AABB is the per-LOD span
    /// (`brick_span(lod)` — a coarse brick covers `2^lod×` more world), grown by the seam epsilon.
    #[test]
    fn resident_coarse_brick_spans_more_world() {
        let reg = registry();
        let b = solid_brick(BlockId(1));
        let lod = 2u32;
        let entries = vec![ResidentBrick { coord: IVec3::new(2, -1, 3), brick: &b, lod }];
        let patch = pack_resident_set(&entries, &reg);
        assert_eq!(patch.brick_count(), 1);
        assert_eq!(patch.metas[0].lod, lod);
        assert_eq!(patch.voxels.len(), halo_cells(lod), "every LOD stores a haloed 10³ grid");
        assert_eq!(halo_cells(lod), 10 * 10 * 10);
        // Core cells (halo index [1, BRICK_EDGE]) are solid; this lone brick has no neighbours → air border.
        for z in 1..=BRICK_EDGE {
            for y in 1..=BRICK_EDGE {
                for x in 1..=BRICK_EDGE {
                    assert_eq!(patch.voxels[halo_index(x, y, z, lod)], 1, "core cell solid");
                }
            }
        }
        // world_min = coord · brick_span(lod) (the clipmap span, 2^lod× the LOD0 span).
        let span = brick_span(lod);
        let wmin = [2.0 * span, -span, 3.0 * span];
        assert_eq!(patch.metas[0].world_min, wmin);
        assert_eq!(patch.aabbs[0], brick_aabb(wmin, lod));
        // The AABB extent is the per-LOD span (grown by the seam epsilon): a LOD2 brick is 4× wider than LOD0.
        let extent = patch.aabbs[0].max[0] - patch.aabbs[0].min[0];
        assert!((extent - (span + 2.0 * BRICK_AABB_EPSILON)).abs() < 1e-3, "AABB spans brick_span(lod)");
        assert!((span - 4.0 * BRICK_WORLD_SIZE).abs() < 1e-4, "LOD2 span is 4× the LOD0 span");
    }

    /// The clipmap voxelizes each `(coord, lod)` directly (a true in-place mip), so the packer stores the
    /// brick's `8³` core VERBATIM — no downsampling/erosion. A brick with a single solid voxel is packed with
    /// that voxel at every LOD (the streaming layer, not the packer, drops all-AIR bricks).
    #[test]
    fn resident_packs_core_verbatim_no_erosion() {
        let reg = registry();
        let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
        voxels[voxel_index(0, 0, 0)] = BlockId(1); // one solid voxel
        let thin = Brick::from_voxels(voxels);
        let entries = vec![ResidentBrick { coord: IVec3::new(0, 0, 0), brick: &thin, lod: 5 }];
        let patch = pack_resident_set(&entries, &reg);
        // Not eroded — the brick is packed as-is (one solid core voxel) at LOD5.
        assert_eq!(patch.brick_count(), 1);
        assert_eq!(patch.metas[0].lod, 5);
        assert_eq!(patch.voxels[halo_index(1, 1, 1, 5)], 1, "the lone solid voxel survives verbatim");
    }

    /// Packing is deterministic: same map → byte-identical buffers (the property the test oracle relies
    /// on — `primitive_index ↔ brick` must be stable).
    #[test]
    fn packing_is_deterministic() {
        let reg = registry();
        let mut map = BrickMap::new();
        for i in 0..5 {
            let (b, _) = one_voxel_brick(i, i, i, BlockId(1));
            map.insert(IVec3::new(i, -i, i * 2), b);
        }
        let a = pack_brickmap(&map, &reg);
        let b = pack_brickmap(&map, &reg);
        assert_eq!(a.aabbs, b.aabbs);
        assert_eq!(a.metas, b.metas);
        assert_eq!(a.voxels, b.voxels);
    }
}
