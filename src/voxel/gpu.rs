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

/// The `voxel_offset` high bit (bit 31) marking a UNIFORM brick (storage plan R1): when set, the brick's
/// FULL haloed `10³` grid is ONE block, so it carries NO per-voxel array — its single block id is packed into
/// the LOW 16 bits of `voxel_offset` (see [`GpuBrickMeta::uniform`] / [`GpuBrickMeta::is_uniform`]). A
/// non-uniform brick stores a real byte offset into the voxel buffer here, which is always `< 2^31` (≤ ~60k
/// bricks × `halo_cells` = ~60 M `u32`s), so bit 31 is free to repurpose without growing the meta or breaking
/// the std140/`bytemuck` layout. The WGSL `BRICK_UNIFORM_FLAG` mirror MUST match this exactly.
pub const BRICK_UNIFORM_FLAG: u32 = 1u32 << 31;

/// Per-brick metadata, parallel to the AABB buffer (index `i` describes the brick whose AABB is
/// `aabbs[i]` and whose `primitive_index` in the ray query is `i`). 32 bytes, `bytemuck`-uploadable.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuBrickMeta {
    /// The brick's world-VOXEL origin (its local `(0,0,0)` corner in world voxel coordinates) =
    /// `brick_coord · BRICK_EDGE`. The shader maps a world position to a local voxel via this.
    pub voxel_origin: [i32; 3],
    /// Offset (in `u32` elements) into the voxel buffer where this brick's voxel block ids begin — UNLESS the
    /// [`BRICK_UNIFORM_FLAG`] high bit is set, in which case this is a UNIFORM brick (storage plan R1): no
    /// voxel-array entries are emitted and the LOW 16 bits hold the single [`BlockId`] of the whole brick.
    /// A DENSE brick stores [`halo_cells`]`(lod)` = `10³` ids (the `8³` core + 1-cell halo) at EVERY LOD (the
    /// grid is a constant `8³`; only the world span scales), so its stride is LOD-independent and bit 31 is
    /// always 0 (real offsets are `< 2^31`). Read via [`Self::is_uniform`] / [`Self::uniform_block`] /
    /// [`Self::dense_offset`] — never compare the raw field.
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

impl GpuBrickMeta {
    /// Build a DENSE-brick meta whose voxels begin at `voxel_offset` in the voxel buffer (bit 31 clear). The
    /// non-uniform path the engine has always emitted — byte-identical to before storage plan R1.
    #[inline]
    pub fn dense(voxel_origin: [i32; 3], voxel_offset: u32, world_min: [f32; 3], lod: u32) -> Self {
        debug_assert!(voxel_offset & BRICK_UNIFORM_FLAG == 0, "voxel offset must leave bit 31 free for the uniform flag");
        Self { voxel_origin, voxel_offset, world_min, lod }
    }

    /// Build a UNIFORM-brick meta (storage plan R1): the whole haloed `10³` grid is `block`, so NO voxel-array
    /// entries are emitted — the block id is packed into the low 16 bits of `voxel_offset` with the
    /// [`BRICK_UNIFORM_FLAG`] high bit set. The shader's DDA reads the id straight from the meta (no
    /// `voxels[]` fetch). `block` must be a SOLID block (a uniform-air brick is empty and never resident).
    #[inline]
    pub fn uniform(voxel_origin: [i32; 3], block: BlockId, world_min: [f32; 3], lod: u32) -> Self {
        Self { voxel_origin, voxel_offset: BRICK_UNIFORM_FLAG | block.0 as u32, world_min, lod }
    }

    /// An ALL-ZERO meta for an UNUSED slot in the incremental fixed-capacity buffers (storage plan: the
    /// per-brick slot allocator never trace-references a freed slot — its AABB is collapsed to
    /// [`super::incremental::degenerate_aabb`] so the BLAS skips it — but the meta still needs defined bytes for
    /// the buffer. `voxel_offset == 0` clears the uniform flag, so even if it were ever read it points at the
    /// arena's first word, not garbage). `Zeroable`-equivalent, written by name for clarity.
    #[inline]
    pub fn zeroed() -> Self {
        Self { voxel_origin: [0; 3], voxel_offset: 0, world_min: [0.0; 3], lod: 0 }
    }

    /// True iff this meta is a collapsed UNIFORM brick (no voxel array; block id in the low bits).
    #[inline]
    pub fn is_uniform(&self) -> bool {
        self.voxel_offset & BRICK_UNIFORM_FLAG != 0
    }

    /// The single [`BlockId`] of a UNIFORM brick (low 16 bits of `voxel_offset`). Meaningless for a dense
    /// brick — gate on [`Self::is_uniform`] first.
    #[inline]
    pub fn uniform_block(&self) -> BlockId {
        BlockId((self.voxel_offset & 0xFFFF) as u16)
    }

    /// The voxel-buffer start offset of a DENSE brick (bit 31 masked off — it is always 0 for a dense brick,
    /// so this equals the raw field, but masking keeps the accessor total). Meaningless for a uniform brick.
    #[inline]
    pub fn dense_offset(&self) -> u32 {
        self.voxel_offset & !BRICK_UNIFORM_FLAG
    }
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

    /// The number of UNIFORM-collapsed bricks in this patch (storage plan R1: a fully-buried uniform-incl-halo
    /// brick whose voxel array was dropped — its block id lives in the meta). The win predictor: a high fraction
    /// means most resident bricks cost ~0 voxel-buffer bytes.
    pub fn uniform_brick_count(&self) -> usize {
        self.metas.iter().filter(|m| m.is_uniform()).count()
    }

    /// A device-free STORAGE-BYTES report for this packed patch (storage plan R1 measurement — the
    /// benchmark-every-delivery number). Compares the AFTER layout (uniform bricks carry no voxel array) against
    /// the pre-R1 BEFORE layout (EVERY brick expanded to a haloed `10³` `u32` array, the content-blind cost the
    /// engine paid before this change). All numbers are CPU-computable from the patch — no GPU device needed.
    pub fn storage_report(&self) -> StorageReport {
        let bricks = self.brick_count();
        let uniform = self.uniform_brick_count();
        let meta_aabb_bytes =
            bricks * (std::mem::size_of::<GpuBrickMeta>() + std::mem::size_of::<GpuBrickAabb>());
        let palette_bytes = self.palette.len() * std::mem::size_of::<GpuPaletteColor>();
        let light_bytes = self.lights.len() * std::mem::size_of::<GpuVoxelLight>()
            + self.alias.len() * std::mem::size_of::<GpuAliasEntry>();
        // AFTER (R1): the actual emitted voxel buffer (uniform bricks contributed nothing).
        let voxel_bytes_after = self.voxels.len() * std::mem::size_of::<u32>();
        // BEFORE (pre-R1): every brick expanded to a haloed 10³ array regardless of content.
        let voxel_bytes_before = bricks * halo_cells(0) * std::mem::size_of::<u32>();
        StorageReport {
            bricks,
            uniform_bricks: uniform,
            meta_aabb_bytes,
            palette_bytes,
            light_bytes,
            voxel_bytes_before,
            voxel_bytes_after,
        }
    }
}

/// A device-free storage-bytes breakdown of a packed [`GpuBrickPatch`] (storage plan R1 measurement). Built by
/// [`GpuBrickPatch::storage_report`]. The headline number is `total_vram_after` vs `total_vram_before` — the
/// resident VRAM the uniform-brick collapse claws back. `meta_aabb_bytes`/`palette_bytes`/`light_bytes` are
/// unchanged by R1 (only the voxel buffer shrinks), so they appear in both totals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StorageReport {
    /// Resident brick count (== BLAS primitives).
    pub bricks: usize,
    /// Of those, the UNIFORM-collapsed bricks (no voxel array — R1's win predictor).
    pub uniform_bricks: usize,
    /// Per-brick meta + AABB bytes (`bricks · (32 + 32)`). Unchanged by R1.
    pub meta_aabb_bytes: usize,
    /// Palette buffer bytes. Unchanged by R1.
    pub palette_bytes: usize,
    /// NEE light-list + alias-table bytes. Unchanged by R1.
    pub light_bytes: usize,
    /// Voxel buffer bytes BEFORE R1 (every brick a haloed `10³` `u32` array — content-blind).
    pub voxel_bytes_before: usize,
    /// Voxel buffer bytes AFTER R1 (uniform bricks emit nothing — only dense surface bricks remain).
    pub voxel_bytes_after: usize,
}

impl StorageReport {
    /// The fraction of resident bricks that collapsed to uniform (0..=1) — R1's win predictor.
    pub fn uniform_fraction(&self) -> f64 {
        if self.bricks == 0 { 0.0 } else { self.uniform_bricks as f64 / self.bricks as f64 }
    }
    /// Mean bytes/brick of the voxel buffer AFTER R1 (the headline density number).
    pub fn voxel_bytes_per_brick_after(&self) -> f64 {
        if self.bricks == 0 { 0.0 } else { self.voxel_bytes_after as f64 / self.bricks as f64 }
    }
    /// Total resident VRAM estimate BEFORE R1 (all GPU buffers, content-blind voxel layout).
    pub fn total_vram_before(&self) -> usize {
        self.meta_aabb_bytes + self.palette_bytes + self.light_bytes + self.voxel_bytes_before
    }
    /// Total resident VRAM estimate AFTER R1 (the single number each phase must reduce).
    pub fn total_vram_after(&self) -> usize {
        self.meta_aabb_bytes + self.palette_bytes + self.light_bytes + self.voxel_bytes_after
    }
    /// The VRAM-reduction factor (`before / after`) — `1.0` when nothing collapsed.
    pub fn vram_reduction(&self) -> f64 {
        let after = self.total_vram_after();
        if after == 0 { 1.0 } else { self.total_vram_before() as f64 / after as f64 }
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

        let voxel_origin = [coord.x * BRICK_EDGE, coord.y * BRICK_EDGE, coord.z * BRICK_EDGE];
        let origin = IVec3::new(voxel_origin[0], voxel_origin[1], voxel_origin[2]);

        // STORAGE PLAN R1 — UNIFORM-INCLUDING-HALO collapse (same rule as `pack_resident_set`, expressed over
        // the `BrickMap`'s world-voxel addressing). A brick whose core is one solid block AND whose entire
        // 1-cell halo (read via `map.voxel_block`, AIR where a neighbour is absent) is that same block is fully
        // buried: drop its voxel array + halo and flag it uniform in the meta. A core-uniform brick on a SEAM
        // (an exposed face whose halo reads AIR) keeps its dense halo for the correct boundary-face normal.
        if let Some(block) = map_uniform_incl_halo_block(map, coord, origin) {
            patch.metas.push(GpuBrickMeta::uniform(voxel_origin, block, world_min, 0));
            continue; // no voxel emit, no light gather (fully buried ⇒ no air-exposed emissive face)
        }

        let voxel_offset = patch.voxels.len() as u32;
        // LOD0 (full res) for the static patch packer — every brick keeps its 8³ core grid (+ halo).
        patch.metas.push(GpuBrickMeta::dense(voxel_origin, voxel_offset, world_min, 0));

        // Append the brick's voxels in HALOED-grid order (+X fastest, then +Y, then +Z): the haloed grid is
        // `(BRICK_EDGE+2)³`, with halo index 0/`h-1` the border ring and core cells at `[1, BRICK_EDGE]`. The
        // border holds the NEIGHBOUR brick's adjacent voxel (read from the map via `voxel_block`; AIR where the
        // neighbour is absent), so the DDA sees a real air→solid crossing at the true surface. The brick's
        // world-voxel origin is `voxel_origin`; haloed local `(hx,hy,hz)` ↦ world voxel `origin + (h*-1)`.
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

    ensure_voxels_nonempty(&mut patch);
    push_palette(&mut patch, registry);
    finalize_lights(&mut patch, found);
    patch
}

/// Keep the voxel buffer NON-EMPTY for upload (storage plan R1 edge case). With uniform-brick collapse a patch
/// whose every brick is uniform-incl-halo emits ZERO voxel-array entries; `wgpu::create_buffer_init` rejects a
/// zero-sized storage buffer, so push a single unreferenced sentinel `0u32`. No `GpuBrickMeta` points at it
/// (every uniform brick reads its id from the meta; every dense brick has its own slice at index 0+), so this
/// is invisible to the trace and keeps every consumer's `cast_slice(&patch.voxels)` valid with no per-site
/// guard. Only fires for the all-uniform degenerate case — a normal scene has dense surface bricks.
#[inline]
fn ensure_voxels_nonempty(patch: &mut GpuBrickPatch) {
    if patch.voxels.is_empty() {
        patch.voxels.push(0);
    }
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

/// The voxel content one resident brick contributes to the GPU buffers, produced by the [`pack_one`] SSOT and
/// consumed BOTH by the full [`pack_resident_set`] (which concatenates these in deterministic order) and by the
/// INCREMENTAL re-packer ([`super::incremental`], which writes each into a fixed slot via `queue_write_buffer`).
/// Factoring the per-brick byte production here is what guarantees the incremental patch is byte-identical to a
/// from-scratch pack for the same `(key → brick)` mapping — the two paths can never drift.
#[derive(Clone, Debug, PartialEq)]
pub enum BrickVoxels {
    /// A UNIFORM-incl-halo brick (storage plan R1): no voxel-array entries; the single block id rides in the
    /// meta (`GpuBrickMeta::uniform`). Costs ZERO voxel-buffer bytes.
    Uniform(BlockId),
    /// A DENSE brick: its full haloed `10³` (= [`halo_cells`]`(lod)`) grid of block ids, in [`halo_index`]
    /// order. The incremental packer writes this into the brick's voxel-arena slot; the full packer appends it.
    Dense(Vec<u32>),
}

/// The fully-resolved GPU contribution of ONE resident brick: its BLAS AABB, its world-voxel origin, its
/// world-min corner, its LOD, and its voxel content ([`BrickVoxels`]). The `voxel_offset` in the final
/// [`GpuBrickMeta`] is filled in by the CALLER (it owns the layout: a running offset for the full packer, a
/// fixed arena slot for the incremental one), so this struct is layout-independent — pure function of the
/// brick + its same-LOD neighbours. This is the SSOT both packers build a brick from.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedBrick {
    /// The (epsilon-grown) BLAS AABB for this brick — pure function of `(coord, lod)`.
    pub aabb: GpuBrickAabb,
    /// The brick's world-VOXEL origin (`coord · BRICK_EDGE`).
    pub voxel_origin: [i32; 3],
    /// The brick's TRUE world-min corner (`coord · brick_span(lod)`).
    pub world_min: [f32; 3],
    /// The brick's clipmap LOD.
    pub lod: u32,
    /// Uniform (no voxel bytes) or dense (a haloed `10³` grid).
    pub voxels: BrickVoxels,
}

impl PackedBrick {
    /// The [`GpuBrickMeta`] for this brick once its dense voxels live at `dense_offset` in the voxel buffer
    /// (ignored for a uniform brick, which packs its id into the meta). The SSOT meta builder both packers call
    /// so the uniform-flag / offset rules are defined exactly once.
    #[inline]
    pub fn meta(&self, dense_offset: u32) -> GpuBrickMeta {
        match &self.voxels {
            BrickVoxels::Uniform(block) => {
                GpuBrickMeta::uniform(self.voxel_origin, *block, self.world_min, self.lod)
            }
            BrickVoxels::Dense(_) => {
                GpuBrickMeta::dense(self.voxel_origin, dense_offset, self.world_min, self.lod)
            }
        }
    }
}

/// Produce ONE resident brick's GPU contribution (AABB + origin + voxel content) from the brick `e` and the
/// resident map `by_key` (its same-LOD neighbours, for the halo + the R1 uniform-incl-halo decision). The
/// SSOT per-brick byte producer: [`pack_resident_set`] concatenates these in deterministic order and the
/// incremental re-packer ([`super::incremental`]) writes each into a fixed slot — so a brick re-packed in
/// isolation is byte-identical to the same brick in a from-scratch full pack (the incremental-vs-full A/B
/// equality test is the completeness gate). Does NOT gather lights (the caller owns the light list); a uniform
/// brick exposes no emissive faces, and a dense brick's faces are gathered from the concatenated buffer.
pub fn pack_one(
    e: &ResidentBrick<'_>,
    by_key: &std::collections::HashMap<(IVec3, u32), &Brick>,
) -> PackedBrick {
    let lod = e.lod;
    let coord = e.coord;
    let span = brick_span(lod);
    let world_min = [coord.x as f32 * span, coord.y as f32 * span, coord.z as f32 * span];
    let voxel_origin = [coord.x * BRICK_EDGE, coord.y * BRICK_EDGE, coord.z * BRICK_EDGE];
    let aabb = brick_aabb(world_min, lod);

    // STORAGE PLAN R1 — UNIFORM-INCLUDING-HALO collapse (the exact rule pack_resident_set's loop used): a brick
    // whose full haloed 10³ grid is one solid block carries no voxel array.
    if let Some(block) = uniform_incl_halo_block(e, by_key) {
        return PackedBrick { aabb, voxel_origin, world_min, lod, voxels: BrickVoxels::Uniform(block) };
    }

    // DENSE: the full haloed 10³ grid in halo_index order (core from the brick, border from the same-LOD
    // neighbour, AIR where absent) — byte-identical to the inline fill pack_resident_set previously did.
    let h = halo_edge(0); // constant haloed edge (= BRICK_EDGE + 2 = 10) at every LOD
    let mut voxels = Vec::with_capacity(halo_cells(0));
    for hz in 0..h {
        for hy in 0..h {
            for hx in 0..h {
                debug_assert_eq!(voxels.len(), halo_index(hx, hy, hz, lod));
                let cx = hx - 1;
                let cy = hy - 1;
                let cz = hz - 1;
                let in_core =
                    (0..BRICK_EDGE).contains(&cx) && (0..BRICK_EDGE).contains(&cy) && (0..BRICK_EDGE).contains(&cz);
                let block = if in_core {
                    e.brick.get(cx, cy, cz)
                } else {
                    neighbour_border_cell(by_key, coord, lod, IVec3::new(cx, cy, cz))
                };
                voxels.push(block.0 as u32);
            }
        }
    }
    PackedBrick { aabb, voxel_origin, world_min, lod, voxels: BrickVoxels::Dense(voxels) }
}

/// Build the `(coord, lod) → &Brick` index over a resident-brick slice — the same map [`pack_resident_set`]
/// builds internally, exposed so the incremental re-packer ([`super::incremental`]) can pass it to [`pack_one`]
/// (one SSOT for the key shape). A brick's halo reads its SAME-LOD neighbours through this map.
pub fn build_by_key<'a>(entries: &[ResidentBrick<'a>]) -> std::collections::HashMap<(IVec3, u32), &'a Brick> {
    entries.iter().map(|e| ((e.coord, e.lod), e.brick)).collect()
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
/// **Storage plan R3 — brick-level dedup.** Interns identical HALOED voxel slices so duplicate dense bricks
/// (repeated columns/arches, identical strata bands, an interior brick that escaped R1's fully-buried test)
/// share ONE slice in the voxel buffer — two such bricks' `metas[].voxel_offset` point at the SAME offset.
///
/// Tier A (GPU-DDA-traceable): **shader-invisible** — the DDA addresses voxels purely through `voxel_offset`,
/// so pointing two metas at one slice is undetectable on the trace (the AABB/meta stay per-brick; only the
/// voxel payload is shared). Identical INCLUDING the halo (the seam-fix border), so two shared bricks render
/// byte-identical geometry AND boundary-face normals. A cut into a shared brick re-packs it with different
/// content → a fresh slice — copy-on-write falls out of keying on content. Used by BOTH [`pack_resident_set`]
/// (the SSOT pack) and [`ResidentPacker::snapshot_patch`](super::incremental) (the live streamed re-pack) so
/// their voxel buffers dedup identically and never drift.
///
/// Cost: one hash of the slice per brick — the same O(cells) order as the `extend_from_slice` copy it replaces
/// on a first sighting, and it SAVES that copy on a hit. FxHash over the 4 KB slice is cheap; the distinct-slice
/// set is small (the dedup ratio), so memory is bounded.
#[derive(Default)]
pub struct VoxelInterner {
    /// Haloed slice content → the `voxel_offset` it was first written at.
    seen: rustc_hash::FxHashMap<Box<[u32]>, u32>,
}

impl VoxelInterner {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `cells` to `buffer` and return its `voxel_offset` — UNLESS an identical slice was already
    /// interned, in which case return the shared offset and append nothing. The returned offset always points
    /// at a slice equal to `cells` (so a caller's subsequent read at that offset is valid).
    pub fn intern(&mut self, buffer: &mut Vec<u32>, cells: &[u32]) -> u32 {
        if let Some(&off) = self.seen.get(cells) {
            return off;
        }
        let off = buffer.len() as u32;
        debug_assert!(off & BRICK_UNIFORM_FLAG == 0, "voxel offset must leave bit 31 free for the uniform flag");
        buffer.extend_from_slice(cells);
        self.seen.insert(cells.into(), off);
        off
    }
}

/// **Storage plan R2 — a brick's haloed cells encoded as a tiny palette + bit-packed indices.** A dense brick
/// touches only `k` distinct block ids (a strata band, a couple of surface materials); storing a
/// `ceil(log2 k)`-bit INDEX per cell + a `k`-entry palette is far smaller than a `u32` id per cell.
///
/// `index_bits` is restricted to a POWER OF 2 in `{1,2,4,8,16}` (the smallest that fits `k`). Because each of
/// those divides 32, a cell's index NEVER straddles a `u32` word boundary — so the GPU `dda_brick` decode is a
/// single fetch + shift + mask (R2b), with no 2-word straddle path. The (small) cost is rounding `k=3` up to
/// 2-bit etc.; the simplicity + the no-straddle guarantee are worth it. (R1's uniform brick is the degenerate
/// `k=1` case and is handled separately — a dense brick here always has `k >= 2`.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PalettedBrick {
    /// The distinct block ids, indexed by the packed indices (first-seen order). Length `k` (`>= 1`).
    pub palette: Vec<u16>,
    /// Bits per index ∈ `{1,2,4,8,16}` — the smallest power of 2 with `2^index_bits >= k`.
    pub index_bits: u8,
    /// The bit-packed index stream: `ceil(cells.len() * index_bits / 32)` `u32` words.
    pub indices: Vec<u32>,
}

/// The smallest power-of-2 bit width in `{1,2,4,8,16}` that can index `k` distinct ids (`2^bits >= k`).
fn pow2_index_bits(k: usize) -> u8 {
    // ceil(log2 k) for k >= 2; k <= 1 still needs 1 bit (a single 0 index).
    let needed = if k <= 1 { 1 } else { usize::BITS - (k - 1).leading_zeros() };
    match needed {
        0 | 1 => 1,
        2 => 2,
        3 | 4 => 4,
        5..=8 => 8,
        _ => 16,
    }
}

/// Encode a brick's haloed `cells` (one block id per cell, as the packer produces them — `u16` zero-extended
/// into `u32`) into a [`PalettedBrick`]. The inverse is [`decode_paletted_cell`]; the round-trip is exact.
pub fn encode_paletted(cells: &[u32]) -> PalettedBrick {
    let mut palette: Vec<u16> = Vec::new();
    let mut id_to_idx: rustc_hash::FxHashMap<u16, u32> = rustc_hash::FxHashMap::default();
    let mut locals: Vec<u32> = Vec::with_capacity(cells.len());
    for &c in cells {
        let id = c as u16;
        let idx = *id_to_idx.entry(id).or_insert_with(|| {
            let i = palette.len() as u32;
            palette.push(id);
            i
        });
        locals.push(idx);
    }
    let index_bits = pow2_index_bits(palette.len());
    let bits = index_bits as usize;
    let words = (cells.len() * bits).div_ceil(32);
    let mut indices = vec![0u32; words];
    let mask = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
    for (i, &idx) in locals.iter().enumerate() {
        let bit = i * bits;
        // index_bits ∈ {1,2,4,8,16} all divide 32 ⇒ a cell never straddles a word ⇒ one word, one shift.
        indices[bit / 32] |= (idx & mask) << (bit % 32);
    }
    PalettedBrick { palette, index_bits, indices }
}

/// Decode cell `i`'s block id from a paletted brick — the exact inverse of [`encode_paletted`]. The GPU
/// `dda_brick` does this same single-word fetch + shift + mask (R2b); this is the CPU oracle for it.
pub fn decode_paletted_cell(palette: &[u16], index_bits: u8, indices: &[u32], i: usize) -> u16 {
    let bits = index_bits as usize;
    let bit = i * bits;
    let mask = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
    let idx = (indices[bit / 32] >> (bit % 32)) & mask;
    palette[idx as usize]
}

pub fn pack_resident_set(entries: &[ResidentBrick<'_>], registry: &BlockRegistry) -> GpuBrickPatch {
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
    let by_key = build_by_key(entries);
    // R3: dedup identical haloed slices so duplicate dense bricks share one voxel slice (shader-invisible).
    let mut interner = VoxelInterner::new();

    for e in entries {
        // Produce this brick's GPU contribution through the SSOT `pack_one` (the SAME per-brick byte producer
        // the incremental re-packer uses, so a from-scratch pack and an incremental patch can never drift).
        let pb = pack_one(e, &by_key);
        patch.aabbs.push(pb.aabb);
        match &pb.voxels {
            BrickVoxels::Uniform(_) => {
                // R1: no voxel-array emit, and no light gather — a uniform-incl-halo brick is fully buried (every
                // face neighbour is the same solid block), so it exposes no emissive faces.
                patch.metas.push(pb.meta(0));
            }
            BrickVoxels::Dense(cells) => {
                // R3: intern the haloed slice — an identical brick shares ONE slice. `intern` appends only on a
                // first sighting and returns the shared offset on a hit, so the gather below always reads a
                // (present) slice equal to `cells` at `voxel_offset` (its own world_min → its own lights).
                let voxel_offset = interner.intern(&mut patch.voxels, cells);
                patch.metas.push(pb.meta(voxel_offset));
                gather_lights_into(&mut found, &patch, registry, voxel_offset as usize, pb.world_min, pb.lod);
            }
        }
    }

    ensure_voxels_nonempty(&mut patch);
    push_palette(&mut patch, registry);
    finalize_lights(&mut patch, found);
    patch
}

/// Storage plan R1: is the resident brick `e` UNIFORM INCLUDING ITS HALO — i.e. its CORE is a single solid
/// block AND every one of its `10³ − 8³` halo border cells (the SAME-LOD neighbour voxels the packer would
/// write) is that same block? Returns the block id when so (safe to drop both the voxel array and the halo),
/// else `None` (keep the dense array). Computed from the resident map (`by_key`) the same way the dense
/// halo-fill does (`neighbour_border_cell`), so the collapse decision can never disagree with what would have
/// been packed. Only the FULLY-buried case qualifies: a core-uniform brick with even one differing halo cell
/// (a real neighbour with a hole, or a shell boundary contributing AIR) is a SURFACE brick that keeps its
/// halo for the correct boundary-face normal.
fn uniform_incl_halo_block(
    e: &ResidentBrick<'_>,
    by_key: &std::collections::HashMap<(IVec3, u32), &Brick>,
) -> Option<BlockId> {
    // The core must be a single SOLID block (the cheap CPU fast path already collapses these). A uniform-AIR
    // brick is empty and never reaches the packer, so `uniform_block()` here is always solid when `Some`.
    let block = e.brick.uniform_block()?;
    if block.is_air() {
        return None;
    }
    // Every halo border cell (one ring beyond each face/edge/corner) must equal `block`. The border spans
    // local voxel coords in `[-1, BRICK_EDGE]` with at least one axis at `-1` or `BRICK_EDGE`; check exactly
    // those cells via the same neighbour resolution the dense halo-fill uses.
    for cz in -1..=BRICK_EDGE {
        for cy in -1..=BRICK_EDGE {
            for cx in -1..=BRICK_EDGE {
                let in_core =
                    (0..BRICK_EDGE).contains(&cx) && (0..BRICK_EDGE).contains(&cy) && (0..BRICK_EDGE).contains(&cz);
                if in_core {
                    continue; // core is `block` by construction (uniform)
                }
                if neighbour_border_cell(by_key, e.coord, e.lod, IVec3::new(cx, cy, cz)) != block {
                    return None; // a halo cell differs ⇒ this is a surface brick, keep it dense
                }
            }
        }
    }
    Some(block)
}

/// Storage plan R1 for the static [`BrickMap`] packer ([`pack_brickmap`]): is the LOD0 brick at `coord`
/// (world-voxel origin `origin`) UNIFORM INCLUDING ITS HALO? Mirrors [`uniform_incl_halo_block`] but resolves
/// the halo through the map's world-voxel addressing (`voxel_block`, AIR for an absent neighbour) — exactly
/// what `pack_brickmap`'s dense halo-fill reads — so the collapse decision matches the bytes that would have
/// been packed. Returns the single solid block when fully buried, else `None` (keep the dense halo).
fn map_uniform_incl_halo_block(map: &BrickMap, coord: IVec3, origin: IVec3) -> Option<BlockId> {
    let block = map.get(coord)?.uniform_block()?;
    if block.is_air() {
        return None;
    }
    for hz in -1..=BRICK_EDGE {
        for hy in -1..=BRICK_EDGE {
            for hx in -1..=BRICK_EDGE {
                let in_core =
                    (0..BRICK_EDGE).contains(&hx) && (0..BRICK_EDGE).contains(&hy) && (0..BRICK_EDGE).contains(&hz);
                if in_core {
                    continue; // core is `block` by construction (uniform)
                }
                if map.voxel_block(origin + IVec3::new(hx, hy, hz)) != block {
                    return None; // a halo cell differs ⇒ surface brick, keep it dense
                }
            }
        }
    }
    Some(block)
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

/// Finalize a [`GpuBrickPatch`]'s palette + NEE light list from its already-assembled `metas`/`voxels` — the
/// SHARED tail both the from-scratch [`pack_resident_set`] and the incremental
/// [`ResidentPacker::snapshot_patch`](super::incremental::ResidentPacker::snapshot_patch) run so the palette +
/// light list are derived EXACTLY ONCE (no drift). Iterates the dense bricks in the patch (uniform bricks
/// expose no air faces, so they contribute no lights — same as the per-brick path) gathering their air-exposed
/// emissive voxels, then builds the alias table. Assumes `aabbs`/`metas`/`voxels` are populated; clears + fills
/// `palette`/`lights`/`alias`.
pub fn finalize_patch_palette_and_lights(patch: &mut GpuBrickPatch, registry: &BlockRegistry) {
    patch.palette.clear();
    patch.lights.clear();
    patch.alias.clear();
    let mut found: Vec<EmissiveVoxel> = Vec::new();
    // SKIP the O(resident) per-brick gather entirely when the palette has NO emitters (the common worldgen
    // case) — it would find nothing, so this keeps a non-emissive scene's re-pack O(changed). When emitters
    // exist, snapshot the dense metas first (gather_lights_into borrows `patch` immutably, so collect the dense
    // brick params up front to avoid an aliasing borrow). A uniform brick exposes no faces → skipped.
    if registry.has_emitters() {
        let dense: Vec<(usize, [f32; 3], u32)> = patch
            .metas
            .iter()
            .filter(|m| !m.is_uniform())
            .map(|m| (m.dense_offset() as usize, m.world_min, m.lod))
            .collect();
        for (offset, world_min, lod) in dense {
            gather_lights_into(&mut found, patch, registry, offset, world_min, lod);
        }
    }
    push_palette(patch, registry);
    finalize_lights(patch, found);
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

    // --- STORAGE PLAN R1: uniform-including-halo collapse ----------------------------------------------

    /// The `GpuBrickMeta` flag encoding round-trips and stays 32 bytes (std140/encase-safe): a uniform meta is
    /// flagged + carries its block id in the low bits; a dense meta is byte-identical to before R1 (bit 31
    /// clear, offset readable). The Rust struct size is pinned so the WGSL mirror can't silently drift.
    #[test]
    fn meta_uniform_flag_roundtrips_without_growing() {
        assert_eq!(std::mem::size_of::<GpuBrickMeta>(), 32, "meta must stay 32 bytes (WGSL byte-match)");
        let u = GpuBrickMeta::uniform([8, 0, -8], BlockId(1234), [1.6, 0.0, -1.6], 3);
        assert!(u.is_uniform());
        assert_eq!(u.uniform_block(), BlockId(1234));
        assert_eq!(u.voxel_origin, [8, 0, -8]);
        assert_eq!(u.lod, 3);

        let d = GpuBrickMeta::dense([0, 0, 0], 5000, [0.0; 3], 0);
        assert!(!d.is_uniform(), "a dense brick must NOT be flagged uniform");
        assert_eq!(d.dense_offset(), 5000, "dense offset reads back unchanged (bit 31 clear)");
    }

    /// A fully-buried uniform brick (its 6/26 neighbours all the SAME solid block) collapses: the meta is
    /// flagged uniform with that block id and NO voxel-array entries are emitted for it; its surrounding
    /// SURFACE bricks (an exposed face whose halo reads AIR) stay DENSE. This is the deep-interior win.
    #[test]
    fn uniform_incl_halo_brick_collapses_no_array() {
        let reg = registry();
        let block = BlockId(1);
        let b = solid_brick(block);
        // A 3×3×3 block of identical solid bricks. The CENTER (1,1,1) is fully surrounded by same-block
        // neighbours ⇒ uniform-incl-halo ⇒ collapses. The 26 shell bricks have ≥1 air-halo face ⇒ dense.
        let mut entries = Vec::new();
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    entries.push(ResidentBrick { coord: IVec3::new(x, y, z), brick: &b, lod: 0 });
                }
            }
        }
        let patch = pack_resident_set(&entries, &reg);
        assert_eq!(patch.brick_count(), 27);
        assert_eq!(patch.uniform_brick_count(), 1, "only the fully-buried center brick collapses");

        // The center brick's meta is uniform with the right block id; the surface bricks are dense.
        let center = patch.metas.iter().find(|m| m.voxel_origin == [BRICK_EDGE, BRICK_EDGE, BRICK_EDGE]).unwrap();
        assert!(center.is_uniform());
        assert_eq!(center.uniform_block(), block);

        // Exactly the 26 dense surface bricks emit a haloed 10³ array each; the uniform brick emits nothing.
        assert_eq!(patch.voxels.len(), 26 * halo_cells(0), "only the 26 dense bricks carry voxel arrays");
    }

    /// An ISOLATED uniform brick (no neighbours ⇒ AIR halo) is NOT collapsed — its boundary faces are exposed
    /// and need the dense halo for the correct entry-face normal. R1 collapses ONLY the fully-buried case.
    #[test]
    fn isolated_uniform_brick_stays_dense() {
        let reg = registry();
        let b = solid_brick(BlockId(1));
        let entries = vec![ResidentBrick { coord: IVec3::new(0, 0, 0), brick: &b, lod: 0 }];
        let patch = pack_resident_set(&entries, &reg);
        assert_eq!(patch.uniform_brick_count(), 0, "an air-haloed (surface) uniform brick must stay dense");
        assert!(!patch.metas[0].is_uniform());
        assert_eq!(patch.voxels.len(), halo_cells(0), "a dense brick keeps its full haloed array");
    }

    /// R3 (brick dedup): four IDENTICAL isolated dense bricks share ONE voxel slice — their metas point at the
    /// same `voxel_offset` and the voxel buffer holds a single haloed slice, not four. (Isolated ⇒ every halo
    /// border is AIR ⇒ the four slices are byte-identical; they stay dense because that AIR halo ≠ the solid
    /// core.) The win is shader-invisible: the trace addresses voxels purely through `voxel_offset`.
    #[test]
    fn r3_dedups_identical_dense_bricks() {
        let reg = registry();
        let b = solid_brick(BlockId(1));
        // Four bricks far enough apart that none is another's neighbour (so every halo border resolves to AIR).
        let coords = [IVec3::new(0, 0, 0), IVec3::new(50, 0, 0), IVec3::new(0, 50, 0), IVec3::new(0, 0, 50)];
        let entries: Vec<ResidentBrick> =
            coords.iter().map(|&c| ResidentBrick { coord: c, brick: &b, lod: 0 }).collect();
        let patch = pack_resident_set(&entries, &reg);
        assert_eq!(patch.brick_count(), 4);
        assert_eq!(patch.uniform_brick_count(), 0, "isolated solid bricks have AIR halos ⇒ dense, not R1-collapsed");
        let offsets: std::collections::HashSet<u32> = patch.metas.iter().map(|m| m.dense_offset()).collect();
        assert_eq!(offsets.len(), 1, "four identical bricks dedup to one slice");
        assert_eq!(patch.voxels.len(), halo_cells(0), "the deduped voxel buffer holds exactly one haloed slice");
    }

    /// R2 (palette + bit-pack): encoding a brick's haloed cells then decoding each cell round-trips EXACTLY,
    /// across a range of distinct-id counts `k` spanning every bit width (1/2/4/8/16); the chosen width is the
    /// smallest power of 2 that fits `k`; and the packed stream is far smaller than the `u32`-per-cell baseline
    /// (the storage win). `decode_paletted_cell` IS the CPU oracle the GPU `dda_brick` decode (R2b) must match.
    #[test]
    fn r2_paletted_brick_roundtrips_at_every_width() {
        for (k, want_bits) in [(2usize, 1u8), (3, 2), (4, 2), (5, 4), (16, 4), (17, 8), (256, 8), (300, 16)] {
            // `halo_cells(0)` cells cycling through `k` distinct ids (1..=k), so all k appear in the palette.
            let cells: Vec<u32> = (0..halo_cells(0)).map(|i| ((i % k) + 1) as u32).collect();
            let pb = encode_paletted(&cells);
            assert_eq!(pb.index_bits, want_bits, "k={k} ⇒ {want_bits}-bit");
            assert_eq!(pb.palette.len(), k, "k={k} distinct ids in the palette");
            for (i, &c) in cells.iter().enumerate() {
                assert_eq!(
                    decode_paletted_cell(&pb.palette, pb.index_bits, &pb.indices, i) as u32,
                    c,
                    "cell {i} (k={k}) must decode to its original id"
                );
            }
            let baseline = cells.len() * 4; // u32-per-cell
            let packed = pb.indices.len() * 4 + pb.palette.len() * 2;
            assert!(packed < baseline, "k={k}: packed {packed} B must beat the {baseline} B u32 baseline");
        }
    }

    /// A core-uniform brick whose HALO DIFFERS (a same-block neighbour with a hole on the shared face) is NOT
    /// collapsed — the differing halo cell means a boundary face is exposed, so the dense halo is required.
    #[test]
    fn core_uniform_but_halo_differs_stays_dense() {
        let reg = registry();
        let block = BlockId(1);
        let solid = solid_brick(block);
        // The neighbour at (+X) is solid EXCEPT one voxel on its shared (−X) face is air, so the center brick's
        // +X halo ring is not all `block` ⇒ the center must stay dense.
        let mut holed = Box::new([block; BRICK_VOXELS]);
        holed[voxel_index(0, 3, 3)] = BlockId::AIR; // a hole on the neighbour's −X face (shared with center +X)
        let holed = Brick::from_voxels(holed);
        let entries = vec![
            ResidentBrick { coord: IVec3::new(0, 0, 0), brick: &solid, lod: 0 },
            ResidentBrick { coord: IVec3::new(1, 0, 0), brick: &holed, lod: 0 },
        ];
        let patch = pack_resident_set(&entries, &reg);
        let center = patch.metas.iter().find(|m| m.voxel_origin == [0, 0, 0]).unwrap();
        assert!(!center.is_uniform(), "a core-uniform brick whose halo differs must stay dense");
    }

    /// `pack_brickmap` (the static path) collapses a fully-buried uniform brick too, identically to
    /// `pack_resident_set`. A 3×3×3 solid `BrickMap` ⇒ 1 collapsed center + 26 dense shell bricks.
    #[test]
    fn pack_brickmap_collapses_buried_uniform() {
        let reg = registry();
        let block = BlockId(1);
        let mut map = BrickMap::new();
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    map.insert(IVec3::new(x, y, z), Brick::uniform(block));
                }
            }
        }
        let patch = pack_brickmap(&map, &reg);
        assert_eq!(patch.brick_count(), 27);
        assert_eq!(patch.uniform_brick_count(), 1, "the buried center brick collapses");
        assert_eq!(patch.voxels.len(), 26 * halo_cells(0));
    }

    /// The storage report quantifies the R1 win: for the 3×3×3 solid block the 1 collapsed brick's 4 KB of
    /// voxel array vanishes, so the AFTER voxel buffer is 26/27 of the BEFORE content-blind layout and the
    /// reported VRAM reduction is `> 1`.
    #[test]
    fn storage_report_quantifies_uniform_win() {
        let reg = registry();
        let b = solid_brick(BlockId(1));
        let mut entries = Vec::new();
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    entries.push(ResidentBrick { coord: IVec3::new(x, y, z), brick: &b, lod: 0 });
                }
            }
        }
        let patch = pack_resident_set(&entries, &reg);
        let rep = patch.storage_report();
        assert_eq!(rep.bricks, 27);
        assert_eq!(rep.uniform_bricks, 1);
        assert_eq!(rep.voxel_bytes_before, 27 * halo_cells(0) * 4);
        assert_eq!(rep.voxel_bytes_after, 26 * halo_cells(0) * 4, "the collapsed brick emits no voxel bytes");
        assert!(rep.total_vram_after() < rep.total_vram_before(), "R1 reduces resident VRAM");
        assert!(rep.vram_reduction() > 1.0, "the reduction factor is > 1");
        assert!((rep.uniform_fraction() - 1.0 / 27.0).abs() < 1e-9);
    }

    /// An all-uniform-collapse patch keeps the voxel buffer NON-EMPTY (the `wgpu` zero-sized-buffer guard): a
    /// patch whose only brick collapsed would emit zero voxel entries, but the sentinel keeps `cast_slice`
    /// valid. Asserts the guard directly on a hand-built empty-voxel patch.
    #[test]
    fn all_uniform_patch_keeps_voxels_nonempty() {
        let mut patch = GpuBrickPatch {
            aabbs: vec![brick_aabb([0.0; 3], 0)],
            metas: vec![GpuBrickMeta::uniform([0, 0, 0], BlockId(1), [0.0; 3], 0)],
            voxels: Vec::new(),
            palette: Vec::new(),
            lights: Vec::new(),
            alias: Vec::new(),
        };
        ensure_voxels_nonempty(&mut patch);
        assert_eq!(patch.voxels.len(), 1, "an all-uniform patch gets a single unreferenced sentinel voxel");
        assert_eq!(patch.voxels[0], 0);
    }
}
