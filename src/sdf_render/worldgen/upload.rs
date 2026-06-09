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
use super::coord::{ChunkSize, chunk_gpu_key};
use super::layers::height::{HEIGHT_CHUNK_CELLS, HEIGHT_FIELD_RES};
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

/// Assemble the resident height fields into a fresh ring (full rebuild — invoked only when the store
/// has a delta, i.e. terrain streamed or regenerated, not every frame). Delta-uploading only changed
/// slots is a later optimization; the ring is small (~few MB).
pub fn build_height_ring(store: &ArtifactStore<ScalarField2D>) -> HeightRingCpu {
    let npc_mip = NODES_PER_CHUNK_MIPPED as usize;
    let mut directory = vec![GpuHeightCell::sentinel(); HEIGHT_RING_SLOTS as usize];
    let mut nodes = vec![[0.0f32; 4]; HEIGHT_RING_SLOTS as usize * npc_mip];

    let chunk_size = ChunkSize::new(HEIGHT_CHUNK_CELLS);
    let mut node_spacing = chunk_size.world_size() as f32 / HEIGHT_FIELD_RES as f32;

    for c in store.resident_coords() {
        let Some(field) = store.get(c) else { continue };
        let slot = ring_slot(IVec2::new(c.xyz.x, c.xyz.z));
        let base = slot * npc_mip;
        let (key_hi, key_lo) = chunk_gpu_key(c.xyz);
        directory[slot] = GpuHeightCell { key_hi, key_lo, node_base: base as u32, _pad: 0 };
        node_spacing = field.node_spacing as f32; // all chunks share the tier spacing
        debug_assert_eq!(
            field.nodes.len(),
            HEIGHT_NODES_PER_CHUNK as usize,
            "field resolution must match the ring's mip-0 node count"
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
pub fn sample_ring_lod(ring: &HeightRingCpu, world_xz: DVec2, voxel_size: f32) -> Option<HeightNode> {
    let mip = select_height_mip(ring.node_spacing, voxel_size);
    sample_ring_mip(ring, world_xz, mip)
}

/// The finest mip level whose node spacing (`base · 2^m`) is still ≥ `voxel_size` — the "round the mip
/// UP to the voxel" anti-alias select (see [`sample_ring_lod`]). `voxel_size <= base` (incl. the `0.0`
/// sentinel) ⇒ mip 0; coarser voxels step up one mip per spacing doubling, clamped to `MAX_HEIGHT_MIP`.
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
    use super::super::layers::height::{HeightLayer, HeightParams};
    use super::*;
    use std::sync::Arc;

    fn store_with(coords: &[(i32, i32)], seed: u64) -> ArtifactStore<ScalarField2D> {
        let layer = HeightLayer::new(LayerId(0), HeightParams::default());
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

    /// `sample_ring_lod` with `voxel_size == 0.0` is identical to `sample_ring` (mip 0), and a coarse
    /// voxel routes through the matching coarse mip (`sample_ring_mip`) — the band-limited LOD path the
    /// Terrain eval uses.
    #[test]
    fn sample_ring_lod_selects_mip() {
        let store = store_with(&[(0, 0)], 11);
        let ring = build_height_ring(&store);
        let base = ring.node_spacing;
        let s = HEIGHT_CHUNK_CELLS as f64;
        let wp = DVec2::new(0.4 * s, 0.6 * s);
        // 0.0 sentinel ⇒ mip 0 ⇒ exactly sample_ring.
        let lod0 = sample_ring_lod(&ring, wp, 0.0).unwrap();
        let mip0 = sample_ring(&ring, wp).unwrap();
        assert_eq!(lod0.height.to_bits(), mip0.height.to_bits());
        // A voxel 4× the base spacing selects mip 2 — matches sample_ring_mip(.., 2).
        let lod = sample_ring_lod(&ring, wp, base * 4.0).unwrap();
        let mip = sample_ring_mip(&ring, wp, 2).unwrap();
        assert_eq!(lod.height.to_bits(), mip.height.to_bits());
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
}
