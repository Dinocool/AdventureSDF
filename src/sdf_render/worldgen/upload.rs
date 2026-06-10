//! Producer→consumer bridge: assemble the resident height-field chunks into the GPU's toroidal
//! height **ring** (a world-anchored 2D clipmap the bake samples), and define its layout.
//!
//! The ring mirrors `chunk.rs`'s discipline exactly: a dense toroidal directory of fixed slots
//! (`RING²`), each slot tagged with the absolute chunk key it holds (sentinel when empty) and a base
//! index into a flat node buffer. A world XZ resolves by `floor` to a chunk coord, `rem_euclid` to a
//! ring slot, a key-tag compare (miss ⇒ flat fallback, no hole), then bilinear over that chunk's
//! `(res+1)²` nodes. CPU-built here and parity-tested via [`sample_ring`]; the WGSL sampler mirrors it
//! (pinned by a constants-match test), so CPU picking and GPU rendering agree on the surface.

use std::sync::{Arc, RwLock};

use bevy::math::{DVec2, IVec2, IVec3};
use bytemuck::{Pod, Zeroable};

use super::artifact::{HeightNode, ScalarField2D};
use super::coord::{ChunkSize, chunk_coord_from_gpu_key, chunk_gpu_key};
use super::layers::height::{HEIGHT_CHUNK_CELLS, HEIGHT_FIELD_RES, HeightLayer};
use super::store::ArtifactStore;

/// Toroidal ring width in chunks per axis. Covers `RING × HEIGHT_CHUNK_CELLS` metres; the manager's
/// generation radius must satisfy `2·radius < RING·chunk_size` so no two resident chunks alias one
/// ring slot (the slot-collision invariant the directory's key-tag compare also guards).
pub const HEIGHT_RING_CHUNKS: i32 = 8;
/// Nodes per axis in a chunk's field at mip 0 (`res + 1`, including the apron).
pub const HEIGHT_NODES_PER_AXIS: u32 = HEIGHT_FIELD_RES + 1;
/// Nodes per chunk at mip 0 (`(res+1)²`) — the mip-0 sub-block of a chunk's slot.
pub const HEIGHT_NODES_PER_CHUNK: u32 = HEIGHT_NODES_PER_AXIS * HEIGHT_NODES_PER_AXIS;
/// Total ring slots.
pub const HEIGHT_RING_SLOTS: u32 = (HEIGHT_RING_CHUNKS * HEIGHT_RING_CHUNKS) as u32;

/// Number of MIP levels in the per-chunk height pyramid. Mip `m` has `res>>m` cells per axis →
/// `(res>>m)+1` nodes, node spacing `base · 2^m`. `MAX_HEIGHT_MIP = log2(HEIGHT_FIELD_RES) = 6`
/// (64 → 32 → 16 → 8 → 4 → 2 → 1 cells), so mips have 65² 33² 17² 9² 5² 3² 2² nodes. The coarse-LOD
/// bake samples the mip whose node spacing ≈ its voxel size — a properly band-limited surface a big
/// voxel CAN resolve (no aliased zero-crossing → no black holes at the far extents). MUST mirror the
/// WGSL `MAX_HEIGHT_MIP` (pinned by `wgsl_terrain_constants_match_rust`).
pub const MAX_HEIGHT_MIP: u32 = 6;

/// Nodes per axis at each mip level: `(HEIGHT_FIELD_RES >> m) + 1` for `m ∈ 0..=MAX_HEIGHT_MIP`.
/// `(65, 33, 17, 9, 5, 3, 2)`. Mirrors WGSL `MIP_NODES_PER_AXIS`.
pub const MIP_NODES_PER_AXIS: [u32; 7] = [65, 33, 17, 9, 5, 3, 2];

/// Prefix sum of `MIP_NODES_PER_AXIS[m]²` — the per-mip base offset inside a chunk's node block.
/// `(0, 4225, 5314, 5603, 5684, 5709, 5718)`. Mirrors WGSL `MIP_NODE_OFFSET`.
pub const MIP_NODE_OFFSET: [u32; 7] = [0, 4225, 5314, 5603, 5684, 5709, 5718];

/// Total nodes per chunk across the whole mip pyramid (`Σ (res>>m + 1)²`) = the fixed node-buffer
/// slot size with mips. `4225+1089+289+81+25+9+4 = 5722`. Equal to `MIP_NODE_OFFSET[6] + 2²`.
pub const NODES_PER_CHUNK_MIPPED: u32 = 5722;
/// Sentinel key for an empty/absent ring slot (never equals a real chunk key, so the tag compare
/// misses → flat fallback). Mirrors `chunk::SENTINEL_KEY`.
pub const HEIGHT_SENTINEL_KEY: (u32, u32) = (u32::MAX, u32::MAX);

/// One ring directory record (std430-clean, 16 B): the chunk key tag this slot holds + the base
/// index of its nodes in the flat node buffer. `Pod` for direct byte upload.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuHeightCell {
    pub key_hi: u32,
    pub key_lo: u32,
    pub node_base: u32,
    pub _pad: u32,
}

impl GpuHeightCell {
    fn sentinel() -> Self {
        Self { key_hi: HEIGHT_SENTINEL_KEY.0, key_lo: HEIGHT_SENTINEL_KEY.1, node_base: 0, _pad: 0 }
    }
}

/// One GPU height node: `(height, ∂h/∂x, ∂h/∂z, 0)` world metres. `Rgba32Float`-shaped for direct
/// upload; the trailing lane is reserved (erosion/material weight later).
pub type GpuHeightNode = [f32; 4];

/// CPU-assembled height ring, ready to upload. Built from the manager's resident store. `Clone` so
/// the GPU-upload payload and the CPU picking snapshot (`CPU_HEIGHT_RING`) can share one build
/// instead of running the fBm twice.
#[derive(Clone)]
pub struct HeightRingCpu {
    /// `RING²` directory records, slot = `ring_slot(chunk_xz)`.
    pub directory: Vec<GpuHeightCell>,
    /// `RING² × NODES_PER_CHUNK_MIPPED` nodes; chunk at `slot` occupies
    /// `[slot·NODES_PER_CHUNK_MIPPED, (slot+1)·NODES_PER_CHUNK_MIPPED)`. Within a chunk's block, mip
    /// `m` starts at `MIP_NODE_OFFSET[m]` and holds `MIP_NODES_PER_AXIS[m]²` nodes (row-major, +X
    /// fastest). Mip 0 is the full-detail field; finer mips are box-filtered downsamples the coarse
    /// bake samples (the GPU picks the mip whose spacing ≈ its voxel size).
    pub nodes: Vec<GpuHeightNode>,
    /// World-metre edge of one chunk (= `HEIGHT_CHUNK_CELLS`).
    pub chunk_world_size: f32,
    /// World metres between nodes.
    pub node_spacing: f32,
    /// Cells per axis (`res`); nodes per axis = `res + 1`.
    pub res: u32,
}

/// Physical ring slot for chunk XZ index, `rem_euclid` over the ring (handles negative coords). EXACT
/// mirror of the WGSL `height_ring_slot`.
#[inline]
pub fn ring_slot(chunk_xz: IVec2) -> usize {
    let r = HEIGHT_RING_CHUNKS;
    let mx = chunk_xz.x.rem_euclid(r);
    let mz = chunk_xz.y.rem_euclid(r);
    (mz * r + mx) as usize
}

/// Assemble the resident TIER-0 height fields into a fresh ring (full rebuild — invoked only when the
/// store has a delta, i.e. terrain streamed or regenerated, not every frame). Delta-uploading only
/// changed slots is a later optimization; the ring is small (~few MB).
///
/// Tier 0 only: filters the store to `LayerId(0)`, the finest tier (chunk edge `HEIGHT_CHUNK_CELLS`).
/// The multi-tier clipmap is assembled by [`build_height_clipmap`], which calls
/// [`build_height_ring_for_tier`] once per tier.
pub fn build_height_ring(store: &ArtifactStore<ScalarField2D>) -> HeightRingCpu {
    build_height_ring_for_tier(store, super::coord::LayerId(0), HEIGHT_CHUNK_CELLS)
}

/// Assemble the resident height fields of ONE clipmap tier into a fresh ring. `layer` selects the
/// tier's chunks in the shared store; `chunk_cells` is that tier's chunk edge in base cells
/// (`HEIGHT_CHUNK_CELLS·2^t`). The tier's `chunk_world_size`/`node_spacing` are derived from
/// `chunk_cells` (every tier keeps `HEIGHT_FIELD_RES` nodes per chunk, so node spacing scales with the
/// tier). The per-chunk mip pyramid build is identical to tier 0.
pub fn build_height_ring_for_tier(
    store: &ArtifactStore<ScalarField2D>,
    layer: super::coord::LayerId,
    chunk_cells: u32,
) -> HeightRingCpu {
    let npc_mip = NODES_PER_CHUNK_MIPPED as usize;
    let mut directory = vec![GpuHeightCell::sentinel(); HEIGHT_RING_SLOTS as usize];
    let mut nodes = vec![[0.0f32; 4]; HEIGHT_RING_SLOTS as usize * npc_mip];

    let chunk_size = ChunkSize::new(chunk_cells);
    let node_spacing = chunk_size.world_size() as f32 / HEIGHT_FIELD_RES as f32;

    for c in store.resident_coords() {
        if c.layer != layer {
            continue; // a different tier's chunk — skip (this ring is one tier only)
        }
        let Some(field) = store.get(c) else { continue };
        let slot = ring_slot(IVec2::new(c.xyz.x, c.xyz.z));
        let base = slot * npc_mip;
        let (key_hi, key_lo) = chunk_gpu_key(c.xyz);
        directory[slot] = GpuHeightCell { key_hi, key_lo, node_base: base as u32, _pad: 0 };
        debug_assert_eq!(
            field.nodes.len(),
            HEIGHT_NODES_PER_CHUNK as usize,
            "field resolution must match the ring's mip-0 node count"
        );
        debug_assert!(
            (field.node_spacing as f32 - node_spacing).abs() < 1e-3,
            "tier chunk spacing mismatch: field {} vs derived {node_spacing}",
            field.node_spacing
        );
        build_chunk_mips(&field.nodes, &mut nodes[base..base + npc_mip]);
    }

    HeightRingCpu {
        directory,
        nodes,
        chunk_world_size: chunk_size.world_size() as f32,
        node_spacing,
        res: HEIGHT_FIELD_RES,
    }
}

/// Build a chunk's full MIP pyramid into its node-buffer block. Writes mip 0 (the full-detail field,
/// `(h, dh/dx, dh/dz, 0)`) at `MIP_NODE_OFFSET[0]`, then derives each finer mip from the previous one
/// by a **separable 1-2-1 tent downsample that PRESERVES node positions** (coarse node `i` sits at the
/// same world XZ as fine node `2i`):
///
/// `coarse[i] = 0.25·fine[2i-1] + 0.5·fine[2i] + 0.25·fine[2i+1]`,
///
/// with the off-grid taps clamped at the `0` and last-node boundaries so the corner nodes (`0` and
/// `res_m`) stay aligned to the chunk corners (seam-free across chunks — the corner value is unchanged,
/// matching the neighbour's corner). Height AND gradient are downsampled identically. A linear ramp is
/// a fixed point of this filter, so a planar field downsamples to itself exactly (the property the
/// coarse-LOD bake needs: a low-passed surface a big voxel can still resolve).
fn build_chunk_mips(mip0: &[HeightNode], out: &mut [GpuHeightNode]) {
    // Mip 0: copy the full-detail field as-is.
    debug_assert_eq!(mip0.len(), HEIGHT_NODES_PER_CHUNK as usize);
    for (i, n) in mip0.iter().enumerate() {
        out[MIP_NODE_OFFSET[0] as usize + i] = [n.height, n.dh_dx, n.dh_dz, 0.0];
    }
    // Each finer mip is a separable tent downsample of the previous, in-place into `out`.
    for m in 1..=MAX_HEIGHT_MIP as usize {
        let fine_npa = MIP_NODES_PER_AXIS[m - 1] as usize;
        let coarse_npa = MIP_NODES_PER_AXIS[m] as usize;
        let fine_off = MIP_NODE_OFFSET[m - 1] as usize;
        let coarse_off = MIP_NODE_OFFSET[m] as usize;

        // Pass 1: downsample columns (X axis) → an intermediate of (coarse_npa × fine_npa) nodes.
        let mut tmp = vec![[0.0f32; 4]; coarse_npa * fine_npa];
        for jf in 0..fine_npa {
            for ic in 0..coarse_npa {
                let fi = 2 * ic; // fine X index aligned to this coarse node
                tmp[jf * coarse_npa + ic] =
                    tent_x(&out[fine_off..], fine_npa, fi, jf);
            }
        }
        // Pass 2: downsample rows (Z axis) of the intermediate → the coarse mip.
        for jc in 0..coarse_npa {
            for ic in 0..coarse_npa {
                let fj = 2 * jc; // fine Z index aligned to this coarse node
                out[coarse_off + jc * coarse_npa + ic] = tent_z(&tmp, coarse_npa, fine_npa, ic, fj);
            }
        }
    }
}

/// 1-2-1 tent over the X axis at fine node `(fi, j)` with REFLECTING boundaries: the off-grid tap at
/// `fi±1` mirrors across the edge node when it would fall outside `[0, npa-1]`. Reflection (not
/// clamping) is what makes a linear ramp a FIXED POINT at the corners too — `v[-1] = 2v[0] - v[1]`,
/// so `0.25·v[-1] + 0.5·v[0] + 0.25·v[1] = v[0]` — keeping node 0 / node `res_m` aligned to (and
/// equal to) the chunk-corner value, hence seam-free across chunks. Reads a `(npa × *)` row-major grid.
#[inline]
fn tent_x(grid: &[GpuHeightNode], npa: usize, fi: usize, j: usize) -> GpuHeightNode {
    let c = grid[j * npa + fi];
    let l = if fi == 0 { reflect(c, grid[j * npa + 1]) } else { grid[j * npa + fi - 1] };
    let r = if fi + 1 >= npa { reflect(c, grid[j * npa + fi - 1]) } else { grid[j * npa + fi + 1] };
    weighted3(l, c, r)
}

/// 1-2-1 tent over the Z axis at fine row `fj`, column `i`, REFLECTING boundaries (see [`tent_x`]).
/// Reads the column-downsampled intermediate (`coarse_npa` wide, `fine_npa` tall, row-major).
#[inline]
fn tent_z(tmp: &[GpuHeightNode], coarse_npa: usize, fine_npa: usize, i: usize, fj: usize) -> GpuHeightNode {
    let c = tmp[fj * coarse_npa + i];
    let u = if fj == 0 { reflect(c, tmp[coarse_npa + i]) } else { tmp[(fj - 1) * coarse_npa + i] };
    let d = if fj + 1 >= fine_npa { reflect(c, tmp[(fj - 1) * coarse_npa + i]) } else { tmp[(fj + 1) * coarse_npa + i] };
    weighted3(u, c, d)
}

/// Reflected ghost node `2·edge − inner` per lane (linear extrapolation across the boundary edge).
#[inline]
fn reflect(edge: GpuHeightNode, inner: GpuHeightNode) -> GpuHeightNode {
    let mut o = [0.0f32; 4];
    for k in 0..4 {
        o[k] = 2.0 * edge[k] - inner[k];
    }
    o
}

/// `0.25·a + 0.5·b + 0.25·c` per lane (the normalized 1-2-1 tent weights).
#[inline]
fn weighted3(a: GpuHeightNode, b: GpuHeightNode, c: GpuHeightNode) -> GpuHeightNode {
    let mut o = [0.0f32; 4];
    for k in 0..4 {
        o[k] = 0.25 * a[k] + 0.5 * b[k] + 0.25 * c[k];
    }
    o
}

/// CPU mirror of the WGSL ring sampler: resolve world `world_xz` to its chunk + ring slot, key-tag
/// check, bilinear over that chunk's nodes. `None` on a miss (slot empty or wrapped to a different
/// chunk) → the GPU treats this as a flat fallback. THE function the GPU sampler must match
/// bit-for-relevant; parity-tested below.
pub fn sample_ring(ring: &HeightRingCpu, world_xz: DVec2) -> Option<HeightNode> {
    let s = ring.chunk_world_size as f64;
    let cx = (world_xz.x / s).floor() as i32;
    let cz = (world_xz.y / s).floor() as i32;
    let slot = ring_slot(IVec2::new(cx, cz));
    let rec = ring.directory[slot];
    if (rec.key_hi, rec.key_lo) != chunk_gpu_key(IVec3::new(cx, 0, cz)) {
        return None; // empty slot, or a different (wrapped) chunk occupies it
    }
    // Local node coordinate within the chunk.
    let chunk_min_x = cx as f64 * s;
    let chunk_min_z = cz as f64 * s;
    let lx = (world_xz.x - chunk_min_x) / ring.node_spacing as f64;
    let lz = (world_xz.y - chunk_min_z) / ring.node_spacing as f64;
    let last = (ring.res - 1) as f64;
    let fi = lx.floor().clamp(0.0, last);
    let fj = lz.floor().clamp(0.0, last);
    let i = fi as u32;
    let j = fj as u32;
    let tx = (lx - fi).clamp(0.0, 1.0) as f32;
    let tz = (lz - fj).clamp(0.0, 1.0) as f32;

    // Mip 0 (full detail): MIP_NODE_OFFSET[0] == 0, so this indexes the chunk's mip-0 sub-block.
    let npa = HEIGHT_NODES_PER_AXIS;
    let at = |ii: u32, jj: u32| -> GpuHeightNode { ring.nodes[rec.node_base as usize + (jj * npa + ii) as usize] };
    let n00 = at(i, j);
    let n10 = at(i + 1, j);
    let n01 = at(i, j + 1);
    let n11 = at(i + 1, j + 1);
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let bilerp = |a: GpuHeightNode, b: GpuHeightNode, c: GpuHeightNode, d: GpuHeightNode, k: usize| {
        lerp(lerp(a[k], b[k], tx), lerp(c[k], d[k], tx), tz)
    };
    Some(HeightNode {
        height: bilerp(n00, n10, n01, n11, 0),
        dh_dx: bilerp(n00, n10, n01, n11, 1),
        dh_dz: bilerp(n00, n10, n01, n11, 2),
    })
}

/// CPU mirror of the SHADER's mip-aware sampler (`sample_terrain_height` with mip `m`): like
/// [`sample_ring`] but reads mip level `mip` of the resolved chunk (spacing `base · 2^mip`, nodes
/// `MIP_NODES_PER_AXIS[mip]²` at `MIP_NODE_OFFSET[mip]`). `mip = 0` is identical to [`sample_ring`].
/// Used by the mip unit tests and as the parity reference for the coarse-LOD bake. `None` on a miss.
pub fn sample_ring_mip(ring: &HeightRingCpu, world_xz: DVec2, mip: u32) -> Option<HeightNode> {
    let mip = mip.min(MAX_HEIGHT_MIP) as usize;
    let s = ring.chunk_world_size as f64;
    let cx = (world_xz.x / s).floor() as i32;
    let cz = (world_xz.y / s).floor() as i32;
    let slot = ring_slot(IVec2::new(cx, cz));
    let rec = ring.directory[slot];
    if (rec.key_hi, rec.key_lo) != chunk_gpu_key(IVec3::new(cx, 0, cz)) {
        return None;
    }
    let npa = MIP_NODES_PER_AXIS[mip];
    let res_m = npa - 1; // cells per axis at this mip
    let spacing = ring.node_spacing as f64 * (1u32 << mip) as f64;
    let chunk_min_x = cx as f64 * s;
    let chunk_min_z = cz as f64 * s;
    let lx = (world_xz.x - chunk_min_x) / spacing;
    let lz = (world_xz.y - chunk_min_z) / spacing;
    let last = (res_m - 1) as f64;
    let fi = lx.floor().clamp(0.0, last);
    let fj = lz.floor().clamp(0.0, last);
    let i = fi as u32;
    let j = fj as u32;
    let tx = (lx - fi).clamp(0.0, 1.0) as f32;
    let tz = (lz - fj).clamp(0.0, 1.0) as f32;

    let mip_base = rec.node_base as usize + MIP_NODE_OFFSET[mip] as usize;
    let at = |ii: u32, jj: u32| -> GpuHeightNode { ring.nodes[mip_base + (jj * npa + ii) as usize] };
    let n00 = at(i, j);
    let n10 = at(i + 1, j);
    let n01 = at(i, j + 1);
    let n11 = at(i + 1, j + 1);
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let bilerp = |a: GpuHeightNode, b: GpuHeightNode, c: GpuHeightNode, d: GpuHeightNode, k: usize| {
        lerp(lerp(a[k], b[k], tx), lerp(c[k], d[k], tx), tz)
    };
    Some(HeightNode {
        height: bilerp(n00, n10, n01, n11, 0),
        dh_dx: bilerp(n00, n10, n01, n11, 1),
        dh_dz: bilerp(n00, n10, n01, n11, 2),
    })
}

/// Select the band-limited mip whose node spacing best matches a bake `voxel_size`, then sample the
/// ring at `world_xz` through that mip — the CPU mirror of the (deleted) GPU bake's `voxel → mip`
/// anti-alias rule. Picking the mip whose spacing ≥ `voxel_size` (rounding UP, never finer than the
/// voxel) guarantees the sampled surface is already low-passed below the voxel's Nyquist, so a coarse
/// LOD brick can't alias a sub-voxel zero-crossing into a black hole at the far extents.
///
/// Rule: the FINEST mip `m` with `node_spacing · 2^m ≥ voxel_size`, clamped to `[0, MAX_HEIGHT_MIP]`.
/// `voxel_size == 0.0` is the documented sentinel for "finest / no band-limit" ⇒ mip 0 (identical to
/// [`sample_ring`]) — used by non-LOD callers (picking, classification, tests). `None` on a ring miss.
///
/// OPTION-RETURNING. This is the NON-STRICT sampler for genuinely-optional NON-RENDERING queries
/// (picking, classification, tests) where unloaded ground LEGITIMATELY means "no surface here" and the
/// caller wants to handle the miss itself. The RENDERED bake path must use the strict
/// [`sample_ring_lod`], which PANICS on a miss (a rendered miss is a coverage bug the residency gate
/// should have prevented). Don't reach for this one from a rendering bake.
pub fn try_sample_ring_lod(ring: &HeightRingCpu, world_xz: DVec2, voxel_size: f32) -> Option<HeightNode> {
    let mip = continuous_height_mip(ring.node_spacing, voxel_size);
    sample_ring_mip_frac(ring, world_xz, mip)
}

/// STRICT mip-aware ring sampler for the RENDERED bake path. Like [`try_sample_ring_lod`] but PANICS
/// on a miss instead of returning `None`: a rendered terrain bake samples only chunks the residency
/// coverage gate (`mesh_bake::mesh_resident_chunks`) already proved are fully backed by loaded height,
/// so a miss here is a COVERAGE BUG (the gate let an uncovered chunk become resident), not an expected
/// "no surface" — papering it over with a fallback would re-introduce the corrupt-slab artifact this
/// gate exists to kill. The panic reports the accessed `world_xz`, the `voxel_size`, the selected mip,
/// the ring's `chunk_world_size`/`node_spacing`, and the ring's resident bounds so the offending chunk
/// is identifiable.
pub fn sample_ring_lod(ring: &HeightRingCpu, world_xz: DVec2, voxel_size: f32) -> HeightNode {
    if let Some(node) = try_sample_ring_lod(ring, world_xz, voxel_size) {
        return node;
    }
    let mip = select_height_mip(ring.node_spacing, voxel_size);
    let bounds = ring_resident_bounds(ring);
    let resident = ring.directory.iter().filter(|c| c.key_hi != HEIGHT_SENTINEL_KEY.0).count();
    panic!(
        "terrain sampled outside loaded coverage — the residency coverage gate should have prevented \
         this. world_xz={world_xz:?}, voxel_size={voxel_size}, selected mip={mip}, \
         chunk_world_size={}, node_spacing={}, resident_bounds={bounds:?}, resident_slots={resident}",
        ring.chunk_world_size, ring.node_spacing,
    );
}

/// True iff EVERY ring-chunk overlapping the world-XZ rectangle `[min_xz, max_xz]` is resident (its
/// directory slot's key-tag matches the chunk it should hold). The residency coverage gate uses this
/// to forbid a terrain chunk from becoming resident until its full XZ footprint is backed by loaded
/// height — so the strict [`sample_ring_lod`] can never miss inside a rendered bake. A `false` here
/// means at least one overlapped chunk hasn't streamed in yet (or a different wrapped chunk aliases
/// its slot). Allocation-free.
pub fn ring_covers_aabb(ring: &HeightRingCpu, min_xz: bevy::math::Vec2, max_xz: bevy::math::Vec2) -> bool {
    let s = ring.chunk_world_size as f64;
    let cx0 = (min_xz.x as f64 / s).floor() as i32;
    let cx1 = (max_xz.x as f64 / s).floor() as i32;
    let cz0 = (min_xz.y as f64 / s).floor() as i32;
    let cz1 = (max_xz.y as f64 / s).floor() as i32;
    for cz in cz0..=cz1 {
        for cx in cx0..=cx1 {
            let rec = ring.directory[ring_slot(IVec2::new(cx, cz))];
            if (rec.key_hi, rec.key_lo) != chunk_gpu_key(IVec3::new(cx, 0, cz)) {
                return false;
            }
        }
    }
    true
}

/// True iff this ring COVERS the single world point `world_xz` — i.e. the directory slot `world_xz`
/// resolves to holds the chunk it should (key-tag match). The coverage-ONLY predicate (no node
/// sample): exactly the gate [`sample_ring`]/[`try_sample_ring_lod`] apply before sampling, factored
/// out so the clipmap's tier search can probe coverage WITHOUT paying for the bilinear+mip sample on
/// every tier it rejects. `try_sample_ring_lod(ring, world_xz, vs).is_some() == ring_covers(ring,
/// world_xz)` for any `vs` (the mip select never changes which chunk/slot a point lands in, only how
/// it's interpolated). Allocation-free, a single slot read + key compare.
#[inline]
pub fn ring_covers(ring: &HeightRingCpu, world_xz: DVec2) -> bool {
    let s = ring.chunk_world_size as f64;
    let cx = (world_xz.x / s).floor() as i32;
    let cz = (world_xz.y / s).floor() as i32;
    let rec = ring.directory[ring_slot(IVec2::new(cx, cz))];
    (rec.key_hi, rec.key_lo) == chunk_gpu_key(IVec3::new(cx, 0, cz))
}

/// The min/max chunk-XZ index over the ring directory's NON-sentinel slots (decoded back from each
/// resident cell's key-tag via [`chunk_coord_from_gpu_key`]), or `None` if the ring is empty. Cheap,
/// allocation-free — used only by the strict sampler's panic diagnostics to report the loaded region.
pub fn ring_resident_bounds(ring: &HeightRingCpu) -> Option<(IVec2, IVec2)> {
    let mut bounds: Option<(IVec2, IVec2)> = None;
    for cell in &ring.directory {
        if cell.key_hi == HEIGHT_SENTINEL_KEY.0 && cell.key_lo == HEIGHT_SENTINEL_KEY.1 {
            continue;
        }
        let c = chunk_coord_from_gpu_key(cell.key_hi, cell.key_lo);
        let xz = IVec2::new(c.x, c.z);
        bounds = Some(match bounds {
            None => (xz, xz),
            Some((mn, mx)) => (mn.min(xz), mx.max(xz)),
        });
    }
    bounds
}

// =====================================================================================================
// TIERED HEIGHT CLIPMAP — `T` nested rings (finest tier 0 → coarsest tier T-1), thin wrappers over the
// per-ring functions above.
//
// WHY TIERS ARE SEAMLESS: every tier's ring is built from chunks evaluated against the SAME continuous,
// world-anchored fBm — only the grid spacing differs (tier `t` samples on a `HEIGHT_CHUNK_CELLS·2^t`
// chunk grid). The fBm is already band-limited (gentle params, ~64 m finest feature), so a coarse tier
// doesn't alias, and since all tiers represent the SAME surface their height values AGREE wherever they
// overlap. So picking the finest covering tier per voxel (fine near, coarse far) introduces NO seam and
// NO cross-LOD crack: the value is the same surface either way, just band-limited to the voxel's Nyquist.
// =====================================================================================================

/// A built tiered clipmap: `clipmap[t]` is tier `t`'s ring (tier 0 = finest, chunk edge
/// `HEIGHT_CHUNK_CELLS`; tier `t` = `HEIGHT_CHUNK_CELLS·2^t`). Coarser tiers cover larger footprints.
pub type HeightClipmap = Vec<HeightRingCpu>;

/// Build the full tiered clipmap from the shared store: one ring per tier. `tier_cells[t]` is tier
/// `t`'s chunk edge in base cells; tier `t`'s chunks live under `LayerId(t)` in the store. The result
/// is finest→coarsest (`tier_cells` must be ascending: `HEIGHT_CHUNK_CELLS·2^t`).
pub fn build_height_clipmap(store: &ArtifactStore<ScalarField2D>, tier_cells: &[u32]) -> HeightClipmap {
    tier_cells
        .iter()
        .enumerate()
        .map(|(t, &cells)| build_height_ring_for_tier(store, super::coord::LayerId(t as u32), cells))
        .collect()
}

/// STRICT clipmap sampler for the RENDERED bake path. Walk tiers FINEST→coarsest; sample the FIRST tier
/// that COVERS `world_xz` (its directory slot key-tag matches) at the band-limited mip for `voxel_size`.
/// The finest covering tier = fine near the focus, coarse far — automatically, with no seam (all tiers
/// are the same fBm surface). PANICS if NO tier covers (a rendered miss is a coverage-gate bug, never a
/// silent fallback — same contract as [`sample_ring_lod`]), reporting per-tier coverage diagnostics.
pub fn sample_clipmap_lod(clipmap: &HeightClipmap, world_xz: DVec2, voxel_size: f32) -> HeightNode {
    if let Some(node) = try_sample_clipmap_lod(clipmap, world_xz, voxel_size) {
        return node;
    }
    // Build a per-tier diagnostic line: covered? / node_spacing / resident bounds.
    let mut diag = String::new();
    for (t, ring) in clipmap.iter().enumerate() {
        let covered = try_sample_ring_lod(ring, world_xz, voxel_size).is_some();
        let bounds = ring_resident_bounds(ring);
        diag.push_str(&format!(
            "\n  tier {t}: covered={covered}, chunk_world_size={}, node_spacing={}, resident_bounds={bounds:?}",
            ring.chunk_world_size, ring.node_spacing,
        ));
    }
    panic!(
        "terrain sampled outside loaded clipmap coverage — the residency coverage gate should have \
         prevented this. world_xz={world_xz:?}, voxel_size={voxel_size}, tiers={}{diag}",
        clipmap.len(),
    );
}

/// The index of the FINEST clipmap tier that COVERS `world_xz` (finest→coarsest, first hit), or `None`
/// if no tier covers. PLAIN walk — checks every tier from finest up to the first covering one.
///
/// This is deliberately CONTIGUITY-FREE. A prior optimization assumed "the covering tiers are a
/// contiguous suffix `[c, T-1]`" and hint-seeded the search — but that's FALSE DURING STREAMING: a
/// coarser tier can be only PARTIALLY resident (still filling in) while a finer tier is fully resident
/// and covers, so a point can be covered by tier `c` but NOT by `c+1`. A hint that skips tiers then
/// misses the true finest covering and trips `sample_clipmap_lod`'s strict panic (the cull's gate uses a
/// full `any-tier` check, so the gate and a hint-skipping sampler disagree → crash). The plain
/// finest-first scan always lands on the smallest covering index regardless of contiguity, matching the
/// gate. (A correct distance-bounded fast path can be reintroduced later — it needs the rolling focus to
/// lower-bound the search by which tier's ring can even REACH the point.)
#[inline]
fn finest_covering_tier(clipmap: &HeightClipmap, world_xz: DVec2) -> Option<usize> {
    clipmap.iter().position(|ring| ring_covers(ring, world_xz))
}

/// OPTION-RETURNING clipmap sampler for NON-RENDERING queries (picking/classification/tests) AND the hot
/// mesh-bake path. Returns the sample from the FINEST tier that covers `world_xz` (its within-tier mip
/// select uses that tier's own `node_spacing`), or `None` if NO tier covers (legitimately "no surface
/// loaded here" for a non-rendering query). The NON-STRICT sibling of [`sample_clipmap_lod`].
///
/// The finest covering tier is found by [`finest_covering_tier`] (a plain, contiguity-free finest→coarsest
/// scan). Finest-covering is selected PER SAMPLE, which the cross-LOD seam fix + the geomorph depend on.
pub fn try_sample_clipmap_lod(clipmap: &HeightClipmap, world_xz: DVec2, voxel_size: f32) -> Option<HeightNode> {
    let t = finest_covering_tier(clipmap, world_xz)?;
    // The covering check above proves this tier resolves; sample it at the band-limited mip for `voxel_size`.
    try_sample_ring_lod(&clipmap[t], world_xz, voxel_size)
}

/// True iff SOME tier fully covers the world-XZ footprint `[min_xz, max_xz]` — `(0..T).any(t ⇒
/// ring_covers_aabb(tier t, …))`. Coarser tiers cover larger footprints, so a far chunk is admitted once
/// its coarse tier is resident (the distance then fills in). Consistent with [`sample_clipmap_lod`]: if
/// the coarsest covering tier covers the whole footprint, every point inside has a finest-covering tier,
/// so the strict per-voxel sampler can't miss inside a chunk this gate admitted.
pub fn clipmap_covers_aabb(clipmap: &HeightClipmap, min_xz: bevy::math::Vec2, max_xz: bevy::math::Vec2) -> bool {
    clipmap.iter().any(|ring| ring_covers_aabb(ring, min_xz, max_xz))
}

/// The CONTINUOUS (fractional) mip for a bake `voxel_size`: `clamp(log2(voxel/base), 0, MAX_HEIGHT_MIP)`,
/// NOT rounded. The whole-number part picks the bracketing integer mips; the fraction blends between them
/// (see [`sample_ring_mip_frac`]). `voxel_size ≤ base` (incl. the `0.0`/NaN sentinels) ⇒ `0.0` (full
/// detail, single tap). This is the GEOMORPH lever: a voxel whose effective size ramps from `vs` (coarse
/// interior) to `vs·0.5` (transition face) ramps its sampled mip continuously, so the coarse surface
/// morphs into the finer mip across the transition band instead of stepping at the integer mip boundary.
/// Monotone non-decreasing in `voxel_size`; equals `select_height_mip` at exact spacing doublings (the
/// `ceil` and the `floor`-of-an-integer agree there) but interpolates between.
#[inline]
pub fn continuous_height_mip(base_spacing: f32, voxel_size: f32) -> f32 {
    if voxel_size.is_nan() || voxel_size <= base_spacing {
        return 0.0; // sentinel 0.0, NaN, or a voxel finer than the base node spacing → full detail
    }
    let ratio = (voxel_size / base_spacing) as f64;
    (ratio.log2() as f32).clamp(0.0, MAX_HEIGHT_MIP as f32)
}

/// FRACTIONAL-mip ring sampler: like [`sample_ring_mip`] but `mip` is continuous — it samples the two
/// bracketing integer mips `⌊mip⌋` and `⌈mip⌉` and LERPs BOTH `height` and `dh_dx`/`dh_dz` by
/// `frac = mip − ⌊mip⌋`. `frac == 0` (an integer mip, incl. the common `0.0` interior case) takes the
/// FAST PATH — one [`sample_ring_mip`] tap, no second sample — so the extra cost is bounded to the
/// transition band where the geomorph ramp puts a non-integer mip. `None` on a ring miss (either tap a
/// miss ⇒ miss, but both resolve the same chunk so they agree). This trilinear blend also smooths the
/// LOD-shell mip pop for free (a coarse voxel crossing a spacing-doubling no longer jumps a whole mip).
pub fn sample_ring_mip_frac(ring: &HeightRingCpu, world_xz: DVec2, mip: f32) -> Option<HeightNode> {
    let mip = mip.clamp(0.0, MAX_HEIGHT_MIP as f32);
    let lo = mip.floor();
    let frac = mip - lo;
    let lo_u = lo as u32;
    let lo_node = sample_ring_mip(ring, world_xz, lo_u)?;
    if frac == 0.0 {
        return Some(lo_node); // integer mip → single tap (fast path)
    }
    let hi_node = sample_ring_mip(ring, world_xz, lo_u + 1)?;
    let lerp = |a: f32, b: f32| a + (b - a) * frac;
    Some(HeightNode {
        height: lerp(lo_node.height, hi_node.height),
        dh_dx: lerp(lo_node.dh_dx, hi_node.dh_dx),
        dh_dz: lerp(lo_node.dh_dz, hi_node.dh_dz),
    })
}

/// The finest mip level whose node spacing (`base · 2^m`) is still ≥ `voxel_size` — the "round the mip
/// UP to the voxel" anti-alias select (see [`sample_ring_lod`]). `voxel_size <= base` (incl. the `0.0`
/// sentinel) ⇒ mip 0; coarser voxels step up one mip per spacing doubling, clamped to `MAX_HEIGHT_MIP`.
/// The INTEGER select, kept for non-blended callers/tests; the blended LOD path uses
/// [`continuous_height_mip`] + [`sample_ring_mip_frac`] instead.
#[inline]
pub fn select_height_mip(base_spacing: f32, voxel_size: f32) -> u32 {
    if voxel_size.is_nan() || voxel_size <= base_spacing {
        return 0; // sentinel 0.0, NaN, or a voxel finer than the base node spacing → full detail
    }
    // Smallest m with base·2^m ≥ voxel ⇒ m = ceil(log2(voxel / base)).
    let ratio = (voxel_size / base_spacing) as f64;
    (ratio.log2().ceil() as i64).clamp(0, MAX_HEIGHT_MIP as i64) as u32
}

/// Process-global snapshot of the most-recently-built height ring, shared with the CPU
/// `edits::eval_primitive` `Terrain` branch so picking/classification samples the SAME surface the
/// GPU bake renders (CPU↔GPU parity). The `WorldGenPlugin` swaps a fresh `Arc` in on every ring
/// rebuild via [`set_cpu_height_ring`]; `eval_primitive` reads it via [`cpu_height_ring`]. `None`
/// until the first ring is built (Terrain then falls back to the flat mid-band plane).
///
/// A global (rather than a resource threaded through `eval_primitive`) because `eval_primitive` is
/// the shared pure SDF path called from baking, picking, and tests — none of which carry a Bevy
/// `World`/resource handle — and the ring is a single world-spanning artifact, so a process-global
/// is the minimal seam that keeps that signature untouched. The `Arc` keeps reads lock-free-cheap
/// (clone the handle, drop the guard, then sample).
static CPU_HEIGHT_RING: RwLock<Option<Arc<HeightRingCpu>>> = RwLock::new(None);

/// Publish a freshly-built ring as the CPU picking snapshot (see [`CPU_HEIGHT_RING`]). Called by the
/// `WorldGenPlugin` after each `build_height_ring`. Replaces the prior snapshot wholesale.
pub fn set_cpu_height_ring(ring: Option<Arc<HeightRingCpu>>) {
    *CPU_HEIGHT_RING.write().expect("CPU_HEIGHT_RING poisoned") = ring;
}

/// Current CPU height-ring snapshot (a cheap `Arc` clone), or `None` if no ring has been built yet.
/// The `Terrain` `eval_primitive` branch samples this via [`sample_ring`] so CPU picking matches the
/// GPU render; on `None` (worldgen disabled / not yet built) the caller uses the flat fallback.
pub fn cpu_height_ring() -> Option<Arc<HeightRingCpu>> {
    CPU_HEIGHT_RING.read().expect("CPU_HEIGHT_RING poisoned").clone()
}

/// Process-global snapshot of the most-recently-built tiered height CLIPMAP — the multi-tier sibling of
/// [`CPU_HEIGHT_RING`]. THIS is what the `edits::eval_primitive` `Terrain` branch and the mesh-bake
/// coverage gate read now (fine-near/coarse-far terrain out to the full mesh-bake reach). `CPU_HEIGHT_RING`
/// is kept in lockstep, pointed at tier 0, for the gated single-ring GPU bake + the per-ring parity tests.
/// `None` until the first clipmap is built. `Arc<Vec<…>>` so reads are a cheap handle clone.
static CPU_HEIGHT_CLIPMAP: RwLock<Option<Arc<HeightClipmap>>> = RwLock::new(None);

/// Publish a freshly-built clipmap as the CPU snapshot (see [`CPU_HEIGHT_CLIPMAP`]). Replaces the prior
/// snapshot wholesale. Called by the `WorldGenPlugin` after each `build_height_clipmap`.
pub fn set_cpu_height_clipmap(clipmap: Option<Arc<HeightClipmap>>) {
    *CPU_HEIGHT_CLIPMAP.write().expect("CPU_HEIGHT_CLIPMAP poisoned") = clipmap;
}

/// Current CPU clipmap snapshot (a cheap `Arc` clone), or `None` if none built yet. The `Terrain`
/// `eval_primitive` branch + the mesh-bake coverage gate read this.
pub fn cpu_height_clipmap() -> Option<Arc<HeightClipmap>> {
    CPU_HEIGHT_CLIPMAP.read().expect("CPU_HEIGHT_CLIPMAP poisoned").clone()
}

/// The pure, FULL-FIDELITY terrain surface source for DETAIL-NORMAL baking: the tier-0 [`HeightLayer`]
/// (its `sample_world` is tier-independent, so tier 0 evaluates the same world surface every tier samples)
/// plus the `world_seed`. The detail-normal texture bake samples its RAW `sample_world` slope at texel
/// centres to write the mip-0-scale fine surface slope onto a coarse-LOD chunk's normal map. Published
/// alongside the clipmap by `roll_worldgen` ([`set_cpu_terrain_hifi`]) and captured per-bake in
/// [`BAKE_TERRAIN`]. It is a DERIVED RENDER attribute — NOT keyed by `HEIGHT_GEN_VERSION` (the height
/// itself is unchanged).
pub struct TerrainHifi {
    /// Tier-0 layer whose `sample_world` is the pure surface function (incl. the active biome graph when one
    /// is attached — the same graph every clipmap tier samples).
    pub layer: HeightLayer,
    /// The world seed the surface was generated with (folded into the noise / graph stream).
    pub world_seed: u64,
}

impl TerrainHifi {
    /// The full-fidelity surface SLOPE `(dh/dx, dh/dz)` at world `(wx, wz)` — the RAW analytic
    /// [`HeightLayer::sample_world`] gradient (ONE eval/texel, NO band-limit convolution). The detail-normal
    /// bake stores these two lanes per texel; the shader reconstructs `N = normalize(-dh/dx, 1, -dh/dz)`. The
    /// texel density (and future mips) handle anti-aliasing — the terrain's finest feature is coarse enough
    /// that the new texel resolution samples the raw slope well. Pure / deterministic / bit-portable.
    #[inline]
    pub fn slope(&self, wx: f64, wz: f64) -> (f32, f32) {
        let n = self.layer.sample_world(wx, wz, self.world_seed);
        (n.dh_dx, n.dh_dz)
    }
}

/// Process-global snapshot of the tier-0 terrain hi-fi sampler — the sibling of [`CPU_HEIGHT_CLIPMAP`]
/// for DETAIL-NORMAL baking. `roll_worldgen` republishes it (via [`set_cpu_terrain_hifi`]) whenever it
/// rebuilds the clipmap, so the hi-fi normal source stays in lockstep with the meshed height. `None`
/// until the first publish (the bake then bakes no detail map and uses the cheap stored gradient).
static CPU_TERRAIN_HIFI: RwLock<Option<Arc<TerrainHifi>>> = RwLock::new(None);

/// Publish the tier-0 terrain hi-fi sampler (see [`CPU_TERRAIN_HIFI`]). Replaces the prior snapshot.
pub fn set_cpu_terrain_hifi(hifi: Option<Arc<TerrainHifi>>) {
    *CPU_TERRAIN_HIFI.write().expect("CPU_TERRAIN_HIFI poisoned") = hifi;
}

/// Current tier-0 terrain hi-fi snapshot (a cheap `Arc` clone), or `None` if none published yet.
pub fn cpu_terrain_hifi() -> Option<Arc<TerrainHifi>> {
    CPU_TERRAIN_HIFI.read().expect("CPU_TERRAIN_HIFI poisoned").clone()
}

thread_local! {
    /// Per-bake-thread Terrain snapshot — the clipmap `Arc`, world-XZ offset, and the hi-fi normal sampler
    /// captured ONCE at the top of `mesh_chunk` (via [`set_bake_terrain`]). [`terrain_sdf`] /
    /// [`terrain_normal`] read this with a thread-local `RefCell` borrow (no atomics, no cross-core sharing)
    /// instead of the process-global `RwLock` + `Arc::clone` on EVERY field sample — the bake samples the
    /// field hundreds of thousands of times per chunk, and across the async pool that per-sample
    /// lock/refcount was cache-line-contended (the dominant mesh-bake cost). It also makes a chunk's whole
    /// bake sample ONE stable clipmap + hi-fi source (no mid-bake ring roll). `None` ⇒ no bake snapshot
    /// installed (picking/classification/tests) → fall back to the process-global.
    static BAKE_TERRAIN: std::cell::RefCell<Option<BakeTerrainSnapshot>> =
        const { std::cell::RefCell::new(None) };
}

/// The per-bake Terrain snapshot held in [`BAKE_TERRAIN`]: the frozen clipmap + its world-XZ offset, plus
/// the optional hi-fi normal sampler (the SAME terrain the clipmap was built from, kept in lockstep). The
/// detail-normal texture bake reads `hifi` + `offset` via [`bake_terrain_hifi`] to sample texel slopes.
struct BakeTerrainSnapshot {
    clipmap: Arc<HeightClipmap>,
    offset: bevy::math::Vec2,
    hifi: Option<Arc<TerrainHifi>>,
}

/// RAII guard installing a per-bake Terrain snapshot on THIS thread (see [`BAKE_TERRAIN`]); clears it on
/// drop. Capture once at the top of `mesh_chunk`: `let _g = set_bake_terrain(cpu_height_clipmap(), …);`.
/// A `None` clipmap installs nothing (the rendering path then panics via the global fallback — a bug, as
/// the coverage gate only admits covered chunks).
#[must_use = "the snapshot is cleared when the guard drops; bind it for the bake's duration"]
pub struct BakeTerrainGuard(());

impl Drop for BakeTerrainGuard {
    fn drop(&mut self) {
        BAKE_TERRAIN.with(|tl| *tl.borrow_mut() = None);
    }
}

/// Install a per-bake-thread Terrain snapshot (see [`BAKE_TERRAIN`]) for the lifetime of the returned guard.
/// The hi-fi normal sampler is captured from the process-global ([`cpu_terrain_hifi`]) so it stays in
/// lockstep with the clipmap the same `mesh_chunk` installs (one frozen terrain SSOT for the whole bake).
pub fn set_bake_terrain(clipmap: Option<Arc<HeightClipmap>>, offset: bevy::math::Vec2) -> BakeTerrainGuard {
    let hifi = cpu_terrain_hifi();
    BAKE_TERRAIN.with(|tl| {
        *tl.borrow_mut() = clipmap.map(|c| BakeTerrainSnapshot { clipmap: c, offset, hifi });
    });
    BakeTerrainGuard(())
}

/// The per-bake hi-fi terrain sampler + world-XZ offset, for the DETAIL-NORMAL texture bake. Returns the
/// frozen [`TerrainHifi`] snapshot installed by [`set_bake_terrain`] (in lockstep with the clipmap the
/// chunk meshes against) and the chunk's world-XZ offset (`world_xz = local.xz + offset`), or `None` when
/// no bake snapshot or no hi-fi source is installed (then no detail map is baked). Read once per chunk bake
/// — NOT in the hot per-sample march path.
pub fn bake_terrain_hifi() -> Option<(Arc<TerrainHifi>, bevy::math::Vec2)> {
    BAKE_TERRAIN.with(|tl| {
        tl.borrow().as_ref().and_then(|snap| snap.hifi.clone().map(|h| (h, snap.offset)))
    })
}

/// The Terrain primitive's signed field at local point `p`, sampling the rolling height clipmap — the
/// single SSOT for the `edits::eval_primitive` `Terrain` branch. Reads the per-bake thread-local snapshot
/// ([`BAKE_TERRAIN`]) when one is installed (the hot mesh-bake path: no per-sample global lock), else the
/// process-global ([`cpu_height_clipmap`]) for non-rendering callers. `world_xz = p.xz + offset`.
///
/// STRICT vs TRY, gated by `voxel_size` (Step-1 contract): `voxel_size > 0.0` (a RENDERING bake — the
/// mesh bake always passes the chunk's real voxel size) ⇒ strict [`sample_clipmap_lod`], which PANICS on
/// a miss (a rendered miss is a coverage-gate bug, never a silent flat fallback). `voxel_size == 0.0` (the
/// NON-RENDERING sentinel: picking/classification/tests) ⇒ [`try_sample_clipmap_lod`]; a miss reads as
/// EMPTY SPACE (large POSITIVE distance), not a mid-band plane.
pub fn terrain_sdf(p: bevy::math::Vec3, voxel_size: f32, max_height: f32) -> f32 {
    BAKE_TERRAIN.with(|tl| {
        if let Some(snap) = tl.borrow().as_ref() {
            // A per-bake snapshot is installed ⇒ this is the MESH BAKE marching the field → RAW `p.y − h`
            // (stable Transvoxel crossing solve; the Lipschitz form goes near-zero on a sharp ridge → spiky
            // sliver triangles). `normalize = false`.
            let world_xz = DVec2::new((p.x + snap.offset.x) as f64, (p.z + snap.offset.y) as f64);
            return terrain_height_to_sdf(&snap.clipmap, p.y, world_xz, voxel_size, max_height, false);
        }
        // No per-bake snapshot installed (the narrow-band CULL / picking / classification / tests) → read
        // the process-global and use the LIPSCHITZ-NORMALISED true distance (`normalize = true`) so the
        // cull's `|dist| ≤ circumradius` doesn't false-drop steep chunks → no holes.
        let offset = cpu_terrain_offset();
        let world_xz = DVec2::new((p.x + offset.x) as f64, (p.z + offset.y) as f64);
        match cpu_height_clipmap() {
            Some(clipmap) => terrain_height_to_sdf(&clipmap, p.y, world_xz, voxel_size, max_height, true),
            None if voxel_size > 0.0 => panic!(
                "terrain sampled outside loaded coverage — a rendering bake (voxel_size={voxel_size}) ran \
                 before any height clipmap was built; the coverage gate should have prevented this. \
                 world_xz={world_xz:?}"
            ),
            None => max_height - p.y + 1.0e4, // non-rendering miss ⇒ empty space (no flat fallback)
        }
    })
}

/// The Terrain surface NORMAL at local `p`, from the clipmap's STORED analytic gradient (`dh/dx, dh/dz`):
/// `normalize(-dh/dx, 1, -dh/dz)`. This is the SMOOTH (C0) gradient of the height field — unlike a central
/// difference of the bilinear height, whose gradient JUMPS at every node-cell boundary (the bilinear field
/// is C0 but not C1), giving faceting that worsens at coarse LODs. It also matches across an LOD boundary
/// (both sides sample the same mip via the transition rule) and costs ONE clipmap sample vs the 6-tap
/// central difference. Reads the per-bake snapshot ([`BAKE_TERRAIN`]) else the global. `None` on a miss
/// (the mesh builder then falls back to the CSG central-difference gradient). `voxel_size` picks the mip,
/// same as [`terrain_sdf`], so the normal's band-limit matches the height's.
pub fn terrain_normal(p: bevy::math::Vec3, voxel_size: f32) -> Option<bevy::math::Vec3> {
    let node = BAKE_TERRAIN.with(|tl| {
        if let Some(snap) = tl.borrow().as_ref() {
            let world_xz = DVec2::new((p.x + snap.offset.x) as f64, (p.z + snap.offset.y) as f64);
            try_sample_clipmap_lod(&snap.clipmap, world_xz, voxel_size)
        } else {
            let offset = cpu_terrain_offset();
            let world_xz = DVec2::new((p.x + offset.x) as f64, (p.z + offset.y) as f64);
            cpu_height_clipmap().and_then(|cm| try_sample_clipmap_lod(&cm, world_xz, voxel_size))
        }
    })?;
    Some(bevy::math::Vec3::new(-node.dh_dx, 1.0, -node.dh_dz).normalize_or_zero())
}

/// Strict/try clipmap sample → signed Terrain field; a non-rendering miss is empty space.
///
/// `normalize` selects between two forms of the SAME zero-crossing (the surface is identical either way):
/// - `false` (the MESH BAKE path): the RAW vertical gap `p.y − h`. The bake MARCHES this — and raw is the
///   STABLE field for the Transvoxel edge-crossing solve. On an extremely sharp ridge `|∇h|` spikes, so the
///   normalised form below goes near-zero over a band around the crest, making the crossing solve
///   `t = fₐ/(fₐ−f_b)` ill-conditioned → vertices land erratically → degenerate SPIKY/sliver triangles. Raw
///   `p.y − h` (smooth at the band-limited node scale) marches the sharp ridge cleanly, preserving it.
/// - `true` (the CULL / picking path): LIPSCHITZ-NORMALISED `(p.y − h) / √(1+|∇h|²)`. Raw `p.y − h`
///   over-estimates the true distance on a steep slope (Lipschitz `√(1+|∇h|²)≫1`), which makes the
///   narrow-band cull (`mesh_bake::chunk_has_surface`) false-drop steep chunks → HOLES. The normalised
///   form is a first-order TRUE distance (Lipschitz ≤ 1) so the cull's `|dist| ≤ circumradius` is accurate.
///
/// The bake and the cull are different phases (the bake installs a per-bake snapshot, the cull/picking
/// don't — see `terrain_sdf`), so each gets the form it needs without affecting the other.
#[inline]
fn terrain_height_to_sdf(
    clipmap: &HeightClipmap,
    p_y: f32,
    world_xz: DVec2,
    voxel_size: f32,
    max_height: f32,
    normalize: bool,
) -> f32 {
    let to_sdf = |node: HeightNode| -> f32 {
        let raw = p_y - node.height;
        if normalize {
            let lip = (1.0 + node.dh_dx * node.dh_dx + node.dh_dz * node.dh_dz).sqrt();
            raw / lip
        } else {
            raw
        }
    };
    if voxel_size > 0.0 {
        to_sdf(sample_clipmap_lod(clipmap, world_xz, voxel_size)) // strict: panics on a miss
    } else {
        match try_sample_clipmap_lod(clipmap, world_xz, voxel_size) {
            Some(node) => to_sdf(node),
            None => max_height - p_y + 1.0e4,
        }
    }
}

/// Process-global world-XZ offset of the streaming `Terrain` volume's transform — the sibling of
/// [`CPU_HEIGHT_RING`] that lets the CPU `eval_primitive` `Terrain` branch convert its LOCAL sample
/// point back to WORLD XZ before sampling the (world-anchored) ring.
///
/// Why this exists: the terrain volume now FOLLOWS the camera (its `Transform.translation` is snapped
/// to a chunk grid and slides as the camera explores), so the volume's local space is no longer the
/// world (it was, when the volume sat at IDENTITY). `eval_primitive` runs in the edit's local space
/// (post-`inv_model`), but the height ring is keyed by WORLD XZ — so the Terrain branch must add this
/// offset (`world_xz = local.xz + offset`) to land on the correct ring slot. The follow system keeps
/// it in sync with the volume's translation via [`set_cpu_terrain_offset`].
///
/// ASSUMPTION: the Terrain volume is TRANSLATION-ONLY (no rotation/scale) and `translation.y == 0`,
/// which the follow system guarantees. Under that assumption `local.xz + offset == world.xz` exactly
/// (and local.y == world.y), so the CPU lookup matches the GPU bake (which uses the raw world XZ).
static CPU_TERRAIN_OFFSET: RwLock<bevy::math::Vec2> = RwLock::new(bevy::math::Vec2::ZERO);

/// Set the CPU Terrain world-XZ offset (see [`CPU_TERRAIN_OFFSET`]). Called by the worldgen follow
/// system whenever the terrain volume's translation changes.
pub fn set_cpu_terrain_offset(offset: bevy::math::Vec2) {
    *CPU_TERRAIN_OFFSET.write().expect("CPU_TERRAIN_OFFSET poisoned") = offset;
}

/// Current CPU Terrain world-XZ offset (see [`CPU_TERRAIN_OFFSET`]).
pub fn cpu_terrain_offset() -> bevy::math::Vec2 {
    *CPU_TERRAIN_OFFSET.read().expect("CPU_TERRAIN_OFFSET poisoned")
}

#[cfg(test)]
mod tests {
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
}
