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

/// One EMISSIVE-VOXEL LIGHT (Phase 2.5 NEE). Built CPU-side from the resident set: every emissive voxel
/// (palette `emissive·strength > 0`) on the air-exposed surface becomes one light, so the world-cache update
/// pass can sample emitters DIRECTLY (next-event estimation) instead of only finding them by random bounce —
/// the principled variance / firefly fix. 32 bytes (`pos` 16 + `radiance` 16), `bytemuck`-uploadable.
///
/// The light is treated as a small AREA light = the emissive voxel's world cell: `pos` is the voxel CENTRE and
/// `area` is one voxel FACE area (`cell²`) at the voxel's LOD (the solid-angle measure the shader uses for the
/// area-light pdf). The shader applies the runtime `emissive_strength` knob to `radiance` so it stays the
/// per-block palette emissive SSOT (never pre-baked here — knobs-as-uniforms).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuVoxelLight {
    /// World-metre position of the emissive voxel CENTRE (the area light's reference point).
    pub pos: [f32; 3],
    /// One voxel FACE area at the voxel's LOD, in m² (`cell_size²`). The area measure for the NEE pdf
    /// (`inverse_pdf = area · light_count`), so a coarse (larger) emissive voxel is a proportionally
    /// stronger light.
    pub area: f32,
    /// Linear-RGB palette emissive radiance of the voxel's block (BEFORE the runtime `emissive_strength`
    /// knob, which the shader multiplies in — the per-block emissive SSOT, not pre-scaled).
    pub radiance: [f32; 3],
    /// The NEE area-measure INVERSE PDF for THIS light when it is drawn by the power-weighted alias table:
    /// `inverse_pdf = sum_power / luminance(radiance)` (the `area` cancels because the alias pick probability is
    /// `luminance·area / sum_power` and the per-area sample pdf divides by `area`). Pre-baked at build time so
    /// the shader needs no global `sum_power` (a degenerate luminance-0 light falls back to `area · light_count`,
    /// the uniform-pick inverse_pdf, so its estimator is still unbiased). This is exactly the `inverse_pdf` the
    /// NEE contribution `radiance · G · inverse_pdf` uses.
    pub inv_pdf: f32,
}

/// One POWER-WEIGHTED ALIAS-TABLE entry (Walker's alias method) for O(1) importance sampling of the
/// emissive-voxel light list (Phase 2.5 NEE). Parallel to [`GpuVoxelLight`] — entry `i` is drawn when a
/// uniform pick lands on slot `i`: with probability `prob` keep light `i`, else take light `alias`. Built so a
/// light is selected proportional to its power (`luminance·area`), concentrating shadow rays on the brightest
/// emitters. 8 bytes, `bytemuck`-uploadable.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuAliasEntry {
    /// Probability in `[0,1]` of KEEPING this slot's own light `i` (else fall through to `alias`).
    pub prob: f32,
    /// The light index to fall through to when the `prob` test fails.
    pub alias: u32,
}

/// Max emissive-voxel lights packed into the NEE light list. A worldgen lava field could expose far more
/// emissive surface voxels than is useful to shadow-sample (a few thousand picked by the alias table already
/// importance-samples the dominant emitters); cap the list so the buffer + alias build stay bounded, logging
/// once when the cap truncates. Power-sorted truncation keeps the brightest lights (see [`build_light_list`]).
pub const MAX_VOXEL_LIGHTS: usize = 4096;

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
    /// The EMISSIVE-VOXEL LIGHT LIST (Phase 2.5 NEE): one [`GpuVoxelLight`] per air-exposed emissive voxel in
    /// the resident set, capped at [`MAX_VOXEL_LIGHTS`] (power-sorted, brightest kept). EMPTY when the scene has
    /// no emitters (NEE is then skipped cleanly). Built by [`build_light_list`] in the SAME pack that produces
    /// the voxel buffers, so the light list and the geometry can never drift.
    pub lights: Vec<GpuVoxelLight>,
    /// The power-weighted alias table parallel to `lights` (same length): O(1) importance sampling of the light
    /// list by power (`luminance·area`). Empty iff `lights` is empty.
    pub alias: Vec<GpuAliasEntry>,
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
        lights: Vec::new(),
        alias: Vec::new(),
    };
    // The air-exposed emissive voxels found across all bricks — finalized into the NEE light list at the end.
    let mut found: Vec<EmissiveVoxel> = Vec::new();

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
        // Gather this brick's air-exposed emissive voxels into the NEE light list (Phase 2.5). Done AFTER the
        // brick's voxels (incl. halo) are written, so face-exposure reads the just-packed haloed grid.
        gather_lights_into(&mut found, &patch, registry, voxel_offset as usize, world_min, 0);
    }

    push_palette(&mut patch, registry);
    finalize_lights(&mut patch, found);
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
        lights: Vec::new(),
        alias: Vec::new(),
    };
    // The air-exposed emissive voxels across all resident bricks — finalized into the NEE light list at the end.
    let mut found: Vec<EmissiveVoxel> = Vec::new();

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
        // Gather this brick's air-exposed emissive voxels into the NEE light list (after its haloed grid is
        // written, so cross-brick face-exposure uses the just-packed neighbour halo). Per-LOD `world_min`/`lod`
        // place a coarse brick's larger cells correctly (its emissive voxel is a proportionally larger light).
        gather_lights_into(&mut found, &patch, registry, voxel_offset as usize, world_min, lod);
    }

    push_palette(&mut patch, registry);
    finalize_lights(&mut patch, found);
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

/// Rec.709 luminance of a linear-RGB triple — the scalar POWER measure the alias table importance-samples by
/// (mirrors the shader's `restir_luminance`). One SSOT for the light-power weight, shared by the builder + the
/// per-light `weight`.
#[inline]
fn light_luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Build Walker's POWER-WEIGHTED ALIAS TABLE over `weights` (one per light): the O(1) sampler that picks a
/// light proportional to its power. Returns one [`GpuAliasEntry`] per input weight. Robust to all-zero weights
/// (degenerate emitters with luminance 0 but `area > 0`): falls back to a uniform table so sampling never
/// divides by zero. Empty input → empty table. The classic two-stack construction (small/large) so the build
/// is O(n) and the table is exact (each entry's expected draw probability == `weight / sum`).
fn build_alias_table(weights: &[f32]) -> Vec<GpuAliasEntry> {
    let n = weights.len();
    if n == 0 {
        return Vec::new();
    }
    let sum: f64 = weights.iter().map(|&w| w as f64).sum();
    // All-zero (or non-finite) total → uniform table (prob 1, alias self): every light equally likely, no NaN.
    let uniform = !(sum.is_finite() && sum > 0.0);
    // Scaled probabilities: `p_i · n` so the average is 1.0 (alias-method convention).
    let mut scaled: Vec<f64> = weights
        .iter()
        .map(|&w| if uniform { 1.0 } else { (w as f64) / sum * n as f64 })
        .collect();
    let mut prob = vec![1.0f32; n];
    let mut alias = vec![0u32; n];
    for (i, a) in alias.iter_mut().enumerate() {
        *a = i as u32; // default: keep self (correct for the exactly-balanced case)
    }
    let mut small: Vec<usize> = Vec::new();
    let mut large: Vec<usize> = Vec::new();
    for (i, &p) in scaled.iter().enumerate() {
        if p < 1.0 { small.push(i) } else { large.push(i) }
    }
    while let (Some(s), Some(l)) = (small.pop(), large.pop()) {
        prob[s] = scaled[s] as f32;
        alias[s] = l as u32;
        // Move the leftover mass from the large bucket back onto the working set.
        scaled[l] = (scaled[l] + scaled[s]) - 1.0;
        if scaled[l] < 1.0 { small.push(l) } else { large.push(l) }
    }
    // Any buckets left (FP round-off) keep prob 1 / self-alias — they are exactly balanced.
    for i in large.into_iter().chain(small) {
        prob[i] = 1.0;
        alias[i] = i as u32;
    }
    prob.into_iter().zip(alias).map(|(prob, alias)| GpuAliasEntry { prob, alias }).collect()
}

/// One emissive voxel found during light-list gathering: its world centre, the world cell size at its LOD (for
/// the area measure), and its palette emissive radiance. Internal to [`gather_lights_into`].
struct EmissiveVoxel {
    centre: [f32; 3],
    cell: f32,
    emissive: [f32; 3],
}

/// Finalize the gathered emissive voxels into the patch's NEE light list + alias table (Phase 2.5). Each
/// emissive voxel becomes one [`GpuVoxelLight`] (centre, face area = `cell²`, palette emissive). When the count
/// exceeds [`MAX_VOXEL_LIGHTS`] the list is POWER-SORTED (descending) and TRUNCATED to the cap so the brightest
/// emitters survive (a logged event), then the alias table is built over the kept lights. Empty input → empty
/// list (NEE skips cleanly). The SSOT both packers call so the light list is derived exactly once.
fn finalize_lights(patch: &mut GpuBrickPatch, mut found: Vec<EmissiveVoxel>) {
    if found.is_empty() {
        return; // no emitters — NEE is skipped (light_count == 0)
    }
    if found.len() > MAX_VOXEL_LIGHTS {
        // Keep the brightest (power = luminance · face-area) — they dominate the direct-light estimate.
        let power = |v: &EmissiveVoxel| light_luma(v.emissive) * (v.cell * v.cell);
        found.sort_by(|a, b| {
            power(b).partial_cmp(&power(a)).unwrap_or(std::cmp::Ordering::Equal)
        });
        let dropped = found.len() - MAX_VOXEL_LIGHTS;
        found.truncate(MAX_VOXEL_LIGHTS);
        bevy::log::warn!(
            "voxel NEE: {} emissive-voxel lights exceeded the cap {MAX_VOXEL_LIGHTS} — kept the {} brightest, dropped {dropped}",
            MAX_VOXEL_LIGHTS + dropped,
            MAX_VOXEL_LIGHTS
        );
    }
    let light_count = found.len() as f32;
    // Power weights (`luminance · face-area`) drive the alias table; the total is the `sum_power` that turns a
    // power-weighted pick into an unbiased area-measure inverse_pdf per light.
    let weights: Vec<f32> = found.iter().map(|v| light_luma(v.emissive) * (v.cell * v.cell)).collect();
    let sum_power: f64 = weights.iter().map(|&w| w as f64).sum();
    let usable = sum_power.is_finite() && sum_power > 0.0;
    patch.lights = found
        .iter()
        .zip(&weights)
        .map(|(v, &w)| {
            let area = v.cell * v.cell;
            // Area-measure inverse_pdf for the alias (power-weighted) pick: `sum_power / luminance`. The `area`
            // cancels (pick prob `= luma·area / sum_power`, per-area pdf divides by `area`). A luminance-0 light
            // (weight 0) can't be drawn by a non-degenerate alias table, but the fallback uniform table CAN draw
            // it, so give it the uniform-pick inverse_pdf `area · light_count` to stay unbiased there.
            let inv_pdf = if usable && w > 0.0 {
                (sum_power / (light_luma(v.emissive) as f64)) as f32
            } else {
                area * light_count
            };
            GpuVoxelLight { pos: v.centre, area, radiance: v.emissive, inv_pdf }
        })
        .collect();
    patch.alias = build_alias_table(&weights);
}

/// Gather every AIR-EXPOSED emissive voxel of one packed brick into `found` (Phase 2.5 NEE). A voxel is an
/// emitter iff its block's palette emissive luminance is `> 0`; it is "exposed" iff at least one of its six
/// face neighbours (read from the SAME haloed grid the packer just wrote, so the brick-boundary neighbour is
/// included) is AIR — an interior emissive voxel radiates into solid and can't light anything, so it is
/// skipped (this both bounds the list and matches what the bounce actually sees). `voxel_offset` is the brick's
/// start in `patch.voxels`; `lod`/`world_min` place the cell in world space. Reads ONLY the just-packed buffers
/// + the registry emissive, so the light list can never drift from the geometry.
fn gather_lights_into(
    found: &mut Vec<EmissiveVoxel>,
    patch: &GpuBrickPatch,
    registry: &BlockRegistry,
    voxel_offset: usize,
    world_min: [f32; 3],
    lod: u32,
) {
    let cell = lod_voxel_size_pub(lod);
    let core = lod_edge(lod);
    // Iterate the CORE cells (halo index [1, core]); the brick OWNS these (the halo ring is the neighbour's).
    for cz in 1..=core {
        for cy in 1..=core {
            for cx in 1..=core {
                let id = BlockId(patch.voxels[voxel_offset + halo_index(cx, cy, cz, lod)] as u16);
                if id.is_air() {
                    continue;
                }
                let e = registry.emissive(id);
                if light_luma(e) <= 0.0 {
                    continue; // not an emitter
                }
                // Air-exposed iff any of the 6 face neighbours (in the haloed grid) is AIR.
                let face = |dx: i32, dy: i32, dz: i32| -> bool {
                    let nid = patch.voxels[voxel_offset + halo_index(cx + dx, cy + dy, cz + dz, lod)];
                    nid == BlockId::AIR.0 as u32
                };
                let exposed = face(1, 0, 0)
                    || face(-1, 0, 0)
                    || face(0, 1, 0)
                    || face(0, -1, 0)
                    || face(0, 0, 1)
                    || face(0, 0, -1);
                if !exposed {
                    continue;
                }
                // World centre of this core voxel. Core cell `c∈[1,core]` is brick-local voxel `c-1`, whose
                // world min is `world_min + (c-1)·cell`; the centre adds half a cell.
                let lx = (cx - 1) as f32;
                let ly = (cy - 1) as f32;
                let lz = (cz - 1) as f32;
                found.push(EmissiveVoxel {
                    centre: [
                        world_min[0] + (lx + 0.5) * cell,
                        world_min[1] + (ly + 0.5) * cell,
                        world_min[2] + (lz + 0.5) * cell,
                    ],
                    cell,
                    emissive: e,
                });
            }
        }
    }
}

/// The per-LOD world voxel-cell size (`VOXEL_SIZE · 2^lod`) — a thin re-export of [`super::brickmap::
/// lod_voxel_size`] so the light gather stays inside this module's imports. SSOT-correct (the same function the
/// DDA + AABB use).
#[inline]
fn lod_voxel_size_pub(lod: u32) -> f32 {
    super::brickmap::lod_voxel_size(lod)
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

    // --- Phase 2.5 NEE light-list builder tests --------------------------------------------------------

    /// A registry whose block `1` is a bright emitter, block `2` a non-emitter (for the light-list tests).
    fn emissive_registry() -> BlockRegistry {
        let mut reg = registry();
        reg.set_emissive(BlockId(1), [4.0, 4.0, 4.0]);
        reg
    }

    /// A scene with NO emitters packs an EMPTY light list + alias table (NEE then skips cleanly).
    #[test]
    fn no_emitters_empty_light_list() {
        let reg = registry(); // nothing emissive
        let b = solid_brick(BlockId(1));
        let entries = vec![ResidentBrick { coord: IVec3::new(0, 0, 0), brick: &b, lod: 0 }];
        let patch = pack_resident_set(&entries, &reg);
        assert!(patch.lights.is_empty(), "no emissive blocks ⇒ no lights");
        assert!(patch.alias.is_empty(), "no lights ⇒ no alias table");
    }

    /// A single ISOLATED emissive brick: every one of its 8³ surface voxels is air-exposed, so each becomes a
    /// light carrying the block's palette emissive; the centres lie inside the brick's world AABB and the alias
    /// table is parallel (one entry per light). An isolated solid brick's INTERIOR voxels (6³ of the 8³) are
    /// NOT air-exposed, so the light count is the SHELL only (8³ − 6³ = 296), proving interior voxels are culled.
    #[test]
    fn emissive_brick_gathers_exposed_shell_lights() {
        let reg = emissive_registry();
        let b = solid_brick(BlockId(1));
        let entries = vec![ResidentBrick { coord: IVec3::new(0, 0, 0), brick: &b, lod: 0 }];
        let patch = pack_resident_set(&entries, &reg);
        // Only the air-exposed SHELL voxels (not the solid interior) are lights: 8³ − 6³ = 512 − 216 = 296.
        assert_eq!(patch.lights.len(), 296, "an isolated solid emitter exposes only its surface shell as lights");
        assert_eq!(patch.alias.len(), patch.lights.len(), "the alias table is parallel to the light list");
        let span = BRICK_WORLD_SIZE;
        for l in &patch.lights {
            // Palette emissive carried verbatim (the runtime emissive_strength is applied in the shader).
            assert_eq!(l.radiance, [4.0, 4.0, 4.0], "light radiance = the block's palette emissive");
            // One voxel FACE area at LOD0 (`VOXEL_SIZE²`).
            assert!((l.area - VOXEL_SIZE * VOXEL_SIZE).abs() < 1e-6, "light area = one voxel face");
            // Centre lies within the brick's world AABB [0, span]³ (with a half-voxel margin).
            for k in 0..3 {
                assert!(l.pos[k] > -1e-3 && l.pos[k] < span + 1e-3, "light centre {:?} inside the brick AABB", l.pos);
            }
            assert!(l.inv_pdf > 0.0 && l.inv_pdf.is_finite(), "the per-light inverse_pdf must be positive + finite");
        }
    }

    /// The power-weighted alias table is a VALID distribution: every `prob ∈ [0,1]`, every `alias` index is in
    /// range, and for EQUAL-power lights the table reduces to ~uniform (each entry keeps itself with prob ~1).
    #[test]
    fn alias_table_is_a_valid_distribution() {
        let reg = emissive_registry();
        let b = solid_brick(BlockId(1));
        let entries = vec![ResidentBrick { coord: IVec3::new(0, 0, 0), brick: &b, lod: 0 }];
        let patch = pack_resident_set(&entries, &reg);
        let n = patch.alias.len() as u32;
        assert!(n > 0);
        for (i, e) in patch.alias.iter().enumerate() {
            assert!((0.0..=1.0).contains(&e.prob), "alias[{i}].prob {} must be in [0,1]", e.prob);
            assert!(e.alias < n, "alias[{i}].alias {} out of range (n={n})", e.alias);
            // Equal-power lights ⇒ each entry's own probability ≈ 1 (no fall-through needed).
            assert!(e.prob > 0.99, "equal-power lights ⇒ alias entry keeps itself (prob {} ≈ 1)", e.prob);
        }
    }

    /// Two emitters of DIFFERENT power: the brighter one's alias entries concentrate probability toward it.
    /// (A direct check that the table is POWER-weighted, not uniform.) We sample the alias table many times and
    /// assert the bright light is picked far more often than the dim one.
    #[test]
    fn alias_table_is_power_weighted() {
        // Two single-voxel emitter bricks, far apart, with very different emissive — one 10× the other.
        let mut reg = registry();
        reg.set_emissive(BlockId(1), [10.0, 10.0, 10.0]);
        reg.set_emissive(BlockId(2), [1.0, 1.0, 1.0]);
        let mut bright_v = Box::new([BlockId::AIR; BRICK_VOXELS]);
        bright_v[voxel_index(0, 0, 0)] = BlockId(1);
        let bright = Brick::from_voxels(bright_v);
        let mut dim_v = Box::new([BlockId::AIR; BRICK_VOXELS]);
        dim_v[voxel_index(0, 0, 0)] = BlockId(2);
        let dim = Brick::from_voxels(dim_v);
        let entries = vec![
            ResidentBrick { coord: IVec3::new(0, 0, 0), brick: &bright, lod: 0 },
            ResidentBrick { coord: IVec3::new(10, 0, 0), brick: &dim, lod: 0 },
        ];
        let patch = pack_resident_set(&entries, &reg);
        assert_eq!(patch.lights.len(), 2, "two single-voxel emitters ⇒ two lights");
        // Find which light is the bright one (radiance 10).
        let bright_idx = patch.lights.iter().position(|l| l.radiance[0] > 5.0).unwrap();
        // Walk the alias table deterministically: count, over all slots, the expected pick mass landing on each
        // light. Slot `i` contributes `prob` to light `i` and `1-prob` to light `alias`.
        let mut mass = [0.0f64; 2];
        for (i, e) in patch.alias.iter().enumerate() {
            mass[i] += e.prob as f64;
            mass[e.alias as usize] += (1.0 - e.prob) as f64;
        }
        // The bright light (10× the power) should carry ~10/11 of the total pick mass.
        let frac_bright = mass[bright_idx] / (mass[0] + mass[1]);
        assert!(
            frac_bright > 0.8,
            "the power-weighted alias table must pick the 10×-brighter light far more often (got {frac_bright:.3} \
             of the mass, expected ≈ 0.91)"
        );
    }

    /// Exceeding the light cap truncates to the BRIGHTEST lights (power-sorted) and keeps the alias table
    /// parallel — the list never grows past [`MAX_VOXEL_LIGHTS`].
    #[test]
    fn light_list_caps_and_keeps_brightest() {
        // Build a registry with two emitter blocks: a DIM one (most voxels) + a BRIGHT one (few voxels). Pack
        // enough emissive voxels to exceed the cap, then assert the kept lights are the bright ones.
        let mut reg = registry();
        reg.set_emissive(BlockId(1), [0.5, 0.5, 0.5]); // dim — the bulk
        reg.set_emissive(BlockId(2), [50.0, 50.0, 50.0]); // bright — must survive the cap
        // Many solid DIM bricks (each contributes 296 shell lights) to blow past MAX_VOXEL_LIGHTS, plus a few
        // BRIGHT bricks. 16 dim bricks ⇒ 16·296 = 4736 > 4096 cap.
        let dim = solid_brick(BlockId(1));
        let bright = solid_brick(BlockId(2));
        let mut entries: Vec<ResidentBrick> = Vec::new();
        for i in 0..16i32 {
            entries.push(ResidentBrick { coord: IVec3::new(i * 2, 0, 0), brick: &dim, lod: 0 });
        }
        entries.push(ResidentBrick { coord: IVec3::new(0, 4, 0), brick: &bright, lod: 0 });
        let patch = pack_resident_set(&entries, &reg);
        assert_eq!(patch.lights.len(), MAX_VOXEL_LIGHTS, "the light list is capped at MAX_VOXEL_LIGHTS");
        assert_eq!(patch.alias.len(), MAX_VOXEL_LIGHTS, "the alias table stays parallel after the cap");
        // ALL of the bright block's exposed lights (296) must survive (they are the brightest).
        let bright_count = patch.lights.iter().filter(|l| l.radiance[0] > 25.0).count();
        assert_eq!(bright_count, 296, "the cap must keep EVERY bright light (power-sorted truncation)");
    }

    /// A coarse-LOD emissive brick produces lights whose area scales with the LOD cell size (a coarse emissive
    /// voxel is a proportionally larger area light).
    #[test]
    fn coarse_lod_light_area_scales() {
        let reg = emissive_registry();
        let b = solid_brick(BlockId(1));
        let lod = 2u32;
        let entries = vec![ResidentBrick { coord: IVec3::new(0, 0, 0), brick: &b, lod }];
        let patch = pack_resident_set(&entries, &reg);
        assert!(!patch.lights.is_empty());
        let cell = VOXEL_SIZE * (1u32 << lod) as f32;
        for l in &patch.lights {
            assert!((l.area - cell * cell).abs() < 1e-4, "coarse light area = (VOXEL_SIZE·2^lod)²");
        }
    }
}
