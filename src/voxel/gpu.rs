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
//!   bounds (`brick_coord · BRICK_WORLD_SIZE .. +BRICK_WORLD_SIZE`).
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

use super::brickmap::{BRICK_EDGE, BRICK_WORLD_SIZE, Brick, BrickMap, VOXEL_SIZE, lod_edge};
use super::palette::{BlockId, BlockRegistry};

/// The STORED per-axis grid edge of a brick at LOD `lod`: the core grid ([`lod_edge`]) PLUS a 1-cell HALO
/// border on every side (`core + 2`). The packer fills that border with the adjacent NEIGHBOUR brick's
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

/// The (epsilon-grown) BLAS AABB for a brick whose TRUE world-min corner is `world_min`. The true extent is
/// `BRICK_WORLD_SIZE` per axis; this grows it by [`BRICK_AABB_EPSILON`] on every side so abutting bricks
/// overlap (the seam fix — see that constant). The single place the BLAS AABB bounds are formed, shared by
/// both packers so the overlap rule never drifts.
#[inline]
pub fn brick_aabb(world_min: [f32; 3]) -> GpuBrickAabb {
    let e = BRICK_AABB_EPSILON;
    GpuBrickAabb {
        min: [world_min[0] - e, world_min[1] - e, world_min[2] - e],
        max: [world_min[0] + BRICK_WORLD_SIZE + e, world_min[1] + BRICK_WORLD_SIZE + e, world_min[2] + BRICK_WORLD_SIZE + e],
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
    /// Offset (in `u32` elements) into the voxel buffer where this brick's voxel block ids begin. The brick
    /// stores `lod_edge(lod)³` ids (the LOD-downsampled grid), NOT always `BRICK_VOXELS`.
    pub voxel_offset: u32,
    /// The brick's world-metre minimum corner (= `aabbs[i].min`), duplicated here so the shader's DDA has
    /// the brick origin without a second buffer fetch.
    pub world_min: [f32; 3],
    /// The brick's LOD level (0 = full `8³`, 1 = `4³`, …). The shader derives the grid EDGE
    /// (`BRICK_EDGE >> lod`) and the per-cell world size (`VOXEL_SIZE << lod`) from this, so a coarse brick
    /// is DDA-marched over its coarse grid. Part of the SSOT — uploader, shader, and tests agree on it.
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
        // `brick_aabb`). `world_min` stored in the meta stays the TRUE corner the DDA reconstructs cells from.
        patch.aabbs.push(brick_aabb(world_min));

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

/// One resident brick ready to pack: its brick coordinate, the brick voxels, and the LOD it should be
/// stored at. The streaming layer ([`super::streaming`]) produces these in a DETERMINISTIC order; the
/// packer preserves that order so `primitive_index ↔ brick` is stable (the test oracle relies on it).
pub struct ResidentBrick<'a> {
    /// Integer brick coordinate.
    pub coord: IVec3,
    /// The brick's full-resolution `8³` voxels (downsampled here per its `lod`).
    pub brick: &'a Brick,
    /// The LOD level to store this brick at (0 = full `8³`).
    pub lod: u32,
}

/// The k-of-N "keep solid" threshold used when downsampling a brick at LOD `lod` (the thin-feature rule —
/// see [`Brick::downsample`](super::brickmap::Brick::downsample)). The SSOT rule shared by the packer and
/// the tests: the NEAREST coarse ring (`lod == 1`) is CONSERVATIVE (`k = 1`, "keep solid if ANY child is
/// solid") so a thin one-voxel surface survives its first downsample without holes; deeper LODs use a
/// majority threshold (half the `2^lod`-cubed children, rounded up) where the brick is far enough that
/// erosion is sub-pixel and majority occupancy reads cleaner.
#[inline]
pub fn lod_solid_keep_k(lod: u32) -> u32 {
    match lod {
        0 | 1 => 1,
        l => {
            let children = 1u32 << (3 * l); // (2^l)³
            children.div_ceil(2)
        }
    }
}

/// Pack a camera-following RESIDENT brick set (with per-brick LOD) into the SSOT GPU layout — the
/// streaming successor to [`pack_brickmap`]. Each entry's brick is downsampled to its `lod` grid
/// (`lod_edge(lod)³` cells) via [`lod_solid_keep_k`]'s thin-feature rule, and only NON-EMPTY downsampled
/// bricks are emitted (a coarse brick whose every cell eroded to air is skipped — the sparsity invariant
/// holds at every LOD). The AABB is the brick's full world extent regardless of LOD (only the marched grid
/// resolution differs). The entry ORDER defines each brick's `primitive_index`, so the caller must pass a
/// deterministic order.
pub fn pack_resident_set(entries: &[ResidentBrick<'_>], registry: &BlockRegistry) -> GpuBrickPatch {
    use std::collections::HashMap;

    let mut patch = GpuBrickPatch {
        aabbs: Vec::with_capacity(entries.len()),
        metas: Vec::with_capacity(entries.len()),
        voxels: Vec::with_capacity(entries.len() * halo_cells(0)),
        palette: Vec::with_capacity(registry.len()),
    };

    // Pre-downsample EVERY resident brick EXACTLY ONCE (at its own LOD), keyed by coord, storing (lod, grid).
    // Both a brick's own core cells AND its neighbours' HALO border cells (the seam fix — see `halo_edge`) read
    // from this shared map. Previously each brick re-downsampled all 6 of its neighbours via a per-brick
    // `neighbour_cache`, so every brick was downsampled ~7× (once as self + once per neighbour that borders it),
    // with a fresh HashMap + thousands of redundant Vec allocations per brick — the pack hot spot (~700 ms at
    // ~19k bricks). One downsample per brick + one shared map collapses that to O(resident) once.
    // Same-LOD neighbour contributes its adjacent face cell; an absent / different-LOD neighbour contributes
    // AIR (the pre-halo behaviour — no regression).
    let grids: HashMap<IVec3, (u32, Vec<BlockId>)> =
        entries.iter().map(|e| (e.coord, (e.lod, e.brick.downsample(e.lod, lod_solid_keep_k(e.lod))))).collect();

    for e in entries {
        let lod = e.lod;
        let cedge = lod_edge(lod);
        let grid = &grids[&e.coord].1;
        debug_assert_eq!(grid.len(), (cedge * cedge * cedge) as usize);
        // Skip a brick that downsampled to all-air (sparsity at coarse LOD): no AABB, no DDA work.
        if grid.iter().all(|b| b.is_air()) {
            continue;
        }
        let coord = e.coord;
        let world_min = [
            coord.x as f32 * BRICK_WORLD_SIZE,
            coord.y as f32 * BRICK_WORLD_SIZE,
            coord.z as f32 * BRICK_WORLD_SIZE,
        ];
        // BLAS AABB grown by the seam epsilon (overlapping neighbours); the meta keeps the TRUE `world_min`.
        patch.aabbs.push(brick_aabb(world_min));

        let voxel_offset = patch.voxels.len() as u32;
        let voxel_origin = [coord.x * BRICK_EDGE, coord.y * BRICK_EDGE, coord.z * BRICK_EDGE];
        patch.metas.push(GpuBrickMeta { voxel_origin, voxel_offset, world_min, lod });

        let h = halo_edge(lod);
        for hz in 0..h {
            for hy in 0..h {
                for hx in 0..h {
                    debug_assert_eq!(patch.voxels.len() - voxel_offset as usize, halo_index(hx, hy, hz, lod));
                    // Coarse-cell coordinate this haloed cell maps to (core cells are halo index [1, cedge]).
                    let cx = hx - 1;
                    let cy = hy - 1;
                    let cz = hz - 1;
                    let in_core = (0..cedge).contains(&cx) && (0..cedge).contains(&cy) && (0..cedge).contains(&cz);
                    let block = if in_core {
                        grid[(cx + cy * cedge + cz * cedge * cedge) as usize]
                    } else {
                        // A border cell: resolve the owning neighbour brick + the wrapped coarse cell inside it.
                        neighbour_border_cell(&grids, coord, lod, cedge, IVec3::new(cx, cy, cz))
                    };
                    patch.voxels.push(block.0 as u32);
                }
            }
        }
    }

    push_palette(&mut patch, registry);
    patch
}

/// Resolve one HALO BORDER cell at coarse coordinate `cc` (which lies outside `[0, cedge)` on ≥1 axis) for the
/// brick at `coord`/`lod`: find the neighbour brick that owns the world coarse-cell, downsample it at the same
/// LOD (cached), and return its cell there. Returns AIR when the owning neighbour is absent or stored at a
/// different LOD (so a border with no same-LOD neighbour is air — the conservative pre-halo behaviour).
fn neighbour_border_cell(
    grids: &std::collections::HashMap<IVec3, (u32, Vec<BlockId>)>,
    coord: IVec3,
    lod: u32,
    cedge: i32,
    cc: IVec3,
) -> BlockId {
    // The neighbour brick coordinate = `coord` shifted by which face(s) `cc` overflows; the wrapped coarse
    // cell inside the neighbour is `cc mod cedge` (Euclidean, so −1 ↦ cedge−1).
    let nbr = coord
        + IVec3::new(
            cc.x.div_euclid(cedge),
            cc.y.div_euclid(cedge),
            cc.z.div_euclid(cedge),
        );
    let Some((nbr_lod, grid)) = grids.get(&nbr) else {
        return BlockId::AIR;
    };
    if *nbr_lod != lod {
        return BlockId::AIR; // different-LOD neighbour: cell sizes differ, fall back to air (no regression)
    }
    // Read the neighbour's already-downsampled grid (computed once in the shared map — no re-downsample).
    let lx = cc.x.rem_euclid(cedge);
    let ly = cc.y.rem_euclid(cedge);
    let lz = cc.z.rem_euclid(cedge);
    grid[(lx + ly * cedge + lz * cedge * cedge) as usize]
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

        // AABB bounds match the brick world extent GROWN by the seam epsilon (overlapping neighbours).
        assert_eq!(patch.aabbs[0], brick_aabb([0.0, 0.0, 0.0]));
        assert_eq!(patch.aabbs[1], brick_aabb([BRICK_WORLD_SIZE, 0.0, 0.0]));
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

    /// A coarse brick stores fewer voxels: a uniform brick at LOD1 stores a HALOED `6³ = 216` block ids
    /// (core `4³` + the 1-cell border), the meta records lod==1, and the AABB is still the full brick world
    /// extent (LOD changes resolution, not bounds).
    #[test]
    fn resident_coarse_brick_is_smaller() {
        let reg = registry();
        let b = solid_brick(BlockId(1));
        let entries = vec![ResidentBrick { coord: IVec3::new(2, -1, 3), brick: &b, lod: 1 }];
        let patch = pack_resident_set(&entries, &reg);
        assert_eq!(patch.brick_count(), 1);
        assert_eq!(patch.metas[0].lod, 1);
        assert_eq!(patch.voxels.len(), halo_cells(1), "LOD1 stores a haloed 6³ grid");
        assert_eq!(halo_cells(1), 6 * 6 * 6);
        // Core cells are solid; this lone brick has no neighbours, so the border ring is all air.
        for z in 1..=4 {
            for y in 1..=4 {
                for x in 1..=4 {
                    assert_eq!(patch.voxels[halo_index(x, y, z, 1)], 1, "core cell solid");
                }
            }
        }
        let wmin = [2.0 * BRICK_WORLD_SIZE, -BRICK_WORLD_SIZE, 3.0 * BRICK_WORLD_SIZE];
        // AABB is the full world extent (LOD changes resolution, not bounds), grown by the seam epsilon.
        assert_eq!(patch.aabbs[0], brick_aabb(wmin));
    }

    /// A brick that downsamples to all-air at its LOD is SKIPPED (sparsity at coarse LOD): a single solid
    /// voxel at LOD3 (k = majority of 512) erodes away, so the brick contributes no AABB/meta/voxels.
    #[test]
    fn resident_eroded_brick_is_skipped() {
        let reg = registry();
        let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
        voxels[voxel_index(0, 0, 0)] = BlockId(1); // one solid voxel
        let thin = Brick::from_voxels(voxels);
        let solid = solid_brick(BlockId(2));
        let entries = vec![
            ResidentBrick { coord: IVec3::new(0, 0, 0), brick: &thin, lod: 3 },
            ResidentBrick { coord: IVec3::new(1, 0, 0), brick: &solid, lod: 3 },
        ];
        let patch = pack_resident_set(&entries, &reg);
        // The thin brick eroded to air at LOD3 and was dropped; only the solid brick remains.
        assert_eq!(patch.brick_count(), 1);
        assert_eq!(patch.metas[0].voxel_origin, [BRICK_EDGE, 0, 0]);
    }

    /// The keep-k rule: LOD0/1 are conservative (k=1), deeper LODs are a majority of their child count.
    #[test]
    fn keep_k_rule() {
        assert_eq!(lod_solid_keep_k(0), 1);
        assert_eq!(lod_solid_keep_k(1), 1);
        assert_eq!(lod_solid_keep_k(2), (1u32 << 6).div_ceil(2)); // 64 children → 32
        assert_eq!(lod_solid_keep_k(3), (1u32 << 9).div_ceil(2)); // 512 children → 256
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
