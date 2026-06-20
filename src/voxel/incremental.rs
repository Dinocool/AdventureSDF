//! **Incremental, O(changed) re-pack of the resident brick set into FIXED-CAPACITY GPU buffers.**
//!
//! The full [`pack_resident_set`](super::gpu::pack_resident_set) rebuilds the whole AABB / meta / voxel buffer
//! set on EVERY camera move (O(resident); the per-move hitch grows with the shipping `clip_half`), then the render path
//! recreates all GPU buffers + the BLAS/TLAS from scratch. That is the per-move hitch this module removes.
//!
//! # The model — a per-brick SLOT allocator over fixed-capacity buffers
//!
//! Each resident [`BrickKey`] owns ONE slot `= its primitive_index FOR LIFE` (until it drops). The meta + AABB
//! live at a fixed offset (`slot · stride`) in capacity-`max_resident_bricks` buffers; the dense voxel block
//! lives in a SEPARATE voxel ARENA of fixed `halo_cells(0) = 1000`-u32 blocks (the dense stride is
//! LOD-invariant — [`halo_cells`](super::gpu)`(lod)` is `10³` at EVERY LOD — so the arena is a perfect
//! fixed-block free-list with ZERO fragmentation). A UNIFORM (R1) brick consumes no arena block.
//!
//! `primitive_index = slot` and the buffers stay a SINGLE contiguous AABB/meta/voxel set, so the BLAS still has
//! one AABB geometry and the shader's `metas[primitive_index]` is UNCHANGED — the trace is pixel-identical. A
//! dropped slot's AABB is patched to a DEGENERATE (inverted) box so the BLAS never reports a candidate there.
//!
//! # O(changed) — the dirty set
//!
//! An [`update`](ResidentPacker::update) diffs the new resident set against the live one and emits a
//! [`RepackDelta`]: the slots whose meta/AABB/voxel bytes changed, plus the freed slots. CRITICAL: a brick
//! changing ALSO changes the haloed grids of its resident SAME-LOD neighbours (their halo border reads this
//! brick's core — see [`pack_one`](super::gpu::pack_one)), AND a brick can toggle uniform↔dense purely from a
//! neighbour change. So the dirty set is EXPANDED by each entered/dropped/rewritten brick's resident
//! 26-neighbourhood at the same LOD, and those neighbours are re-`pack_one`'d too. This completeness is what
//! makes an incremental patch byte-identical to a from-scratch pack — the
//! [incremental-vs-full A/B test](tests) is the gate.
//!
//! # Byte-identity to the full pack (the SSOT)
//!
//! Both this module and [`pack_resident_set`](super::gpu::pack_resident_set) build each brick through the ONE
//! [`pack_one`](super::gpu::pack_one) per-brick byte producer, so a brick re-packed in isolation here is
//! byte-identical to the same brick in a from-scratch pack. The slot a brick lands in is an implementation
//! detail (a free-list reuses slots in a different order than a from-scratch `(lod,z,y,x)` sort); the render is
//! identical regardless because the shader resolves everything from `metas[slot].world_min` (no dependence on
//! slot ORDER beyond `primitive_index → meta`). The A/B equality test compares the two as a `key → bytes`
//! MAPPING, not raw slot order.

use bevy::math::IVec3;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick};
use super::gpu::{
    BrickVoxels, GpuBrickAabb, GpuBrickMeta, GpuPaletteColor, PackedBrick, PalettedBrick, ResidentBrick,
    build_by_key, encode_paletted, halo_cells, pack_one, pow2_index_bits,
};
use super::streaming::BrickKey;

/// **Stage G-a — the per-dirty-DENSE-brick GPU PACK command.** Emitted by [`ResidentPacker::update_gpu`]
/// instead of the packed bytes: it names the slot + the alloc offsets the CPU claimed and points at the brick's
/// 27 same-LOD cores (the brick + its 26 neighbours) the GPU halo-fill reads. `assets/shaders/voxel_pack.wgsl`
/// (`pack_brick`) consumes one per workgroup and writes the bit-packed index stream / palette / meta itself —
/// byte-identical to what [`emit_changed_slot`](ResidentPacker::emit_changed_slot) would have written on the CPU.
/// `#[repr(C)]` + `bytemuck`-uploadable; field order/size MUST match the WGSL `PackCommand` (15 u32 / 60 B).
/// A FLAT 15-u32 (60 B) record — NO `[f32;3]`/vec3 fields, because the WGSL `vec3` 16-byte alignment would
/// silently pad the `array<PackCommand>` stride and misalign it against this tightly-packed `#[repr(C)]`. Every
/// field is a scalar so the Rust struct and the WGSL `PackCommand` agree on a 60-B stride field-for-field.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPackCommand {
    /// Brick world-voxel origin (`coord · BRICK_EDGE`) — written into the meta verbatim.
    pub origin_x: i32,
    pub origin_y: i32,
    pub origin_z: i32,
    /// The slot (= `primitive_index`); the meta lands at `meta_buf[slot·12]` (48 B).
    pub slot: u32,
    /// Brick world-min corner — written into the meta verbatim.
    pub world_min_x: f32,
    pub world_min_y: f32,
    pub world_min_z: f32,
    /// Start `u32` of this brick's index stream in `voxel_buf` (= `meta.voxel_offset`).
    pub index_word_offset: u32,
    /// Brick LOD (bits 0-2 of the packed `lod_and_bits`).
    pub lod: u32,
    /// The R2b index bit width ∈ `{1,2,4,8,16}` (the CPU resolved it from the palette size up front).
    pub index_bits: u32,
    /// Start `u32` of this brick's palette in `brick_palettes_buf` (= `meta.palette_base`).
    pub palette_word_offset: u32,
    /// Base index into the per-command 27-entry NEIGHBOUR TABLE (`neighbour_indices`): this command's neighbour
    /// slot `n`'s entry is `neighbour_indices[neighbour_base + n]`, a CORE-POOL index (in `BRICK_VOXELS`-cell
    /// units) into `cores`, or [`NEIGHBOUR_ABSENT`] when that neighbour is not resident (the halo reads AIR).
    /// Each resident core lives ONCE in the pool (deduped across all 27-neighbourhoods), so the upload is
    /// O(resident cores)·512 — NOT O(commands·27)·512 (the naive per-command duplication).
    pub neighbour_base: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// A [`GpuPackCommand`] neighbour-table entry meaning "this neighbour is absent" (its halo cells read AIR). Any
/// value `≥` the core-pool length works; `u32::MAX` is the unambiguous sentinel the WGSL tests against.
pub const NEIGHBOUR_ABSENT: u32 = 0xFFFF_FFFF;

/// **Stage G-b — the per-CHANGED-slot GPU AABB command.** Emitted by [`ResidentPacker::update_gpu`] for EVERY
/// slot whose AABB changed this generation — dense, uniform, AND freed — so the GPU AABB pass
/// (`assets/shaders/voxel_pack.wgsl::write_aabb`, one invocation each) writes `aabb_buf[slot]` itself, byte-
/// identically to [`brick_aabb`](super::gpu::brick_aabb) / [`degenerate_aabb`]. This moves the AABB write off the
/// CPU (G-a's per-slot `queue_write_buffer` — the `vox_blas_delta` cost) so the fill can run in the SAME
/// submission as the BLAS build (fill-then-build). `flag = 1` → resident (write the epsilon-grown box from
/// `world_min`/`lod`); `flag = 0` → freed (write the degenerate box). `#[repr(C)]` + `bytemuck`-uploadable; field
/// order/size MUST match the WGSL `AabbCommand` (8 u32 / 32 B). A FLAT scalar record (NO `[f32;3]`/vec3) so the
/// Rust struct and the WGSL `AabbCommand` agree on a 32-B stride field-for-field (same vec3-padding hazard as
/// [`GpuPackCommand`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAabbCommand {
    /// The slot (= `primitive_index`); the AABB lands at `aabb_buf[slot · 8]` (32 B).
    pub slot: u32,
    /// Brick LOD (only read when `flag == 1`) — selects the per-LOD span + epsilon.
    pub lod: u32,
    /// `1` = resident (write the real epsilon-grown box); `0` = freed (write the degenerate box).
    pub flag: u32,
    pub _pad0: u32,
    /// Brick world-min corner (only read when `flag == 1`).
    pub world_min_x: f32,
    pub world_min_y: f32,
    pub world_min_z: f32,
    pub _pad1: u32,
}

/// **Stage G4 — the per-dirty-brick GPU CLASSIFY command.** Emitted by [`ResidentPacker::update_gpu_prepare`] for
/// EVERY dirty key (uniform OR dense — the CPU does not yet know which; that is what the GPU decides). It points at
/// the brick's 27 same-LOD cores (the brick + its 26 neighbours) the GPU halo-fill reads — the SAME deduped core
/// pool + neighbour table the later `pack_brick` consumes. The GPU (`voxel_pack.wgsl::classify_brick`, one workgroup
/// each) halo-fills then writes a [`GpuClassifyOut`] the CPU reads back to drive the `SlabArena` allocation WITHOUT
/// the CPU `pack_one` (the G4 win). A FLAT 4-u32 (16 B) record — `neighbour_base` + padding (scalar-only, same
/// vec3-padding hazard as [`GpuPackCommand`]). `#[repr(C)]` + `bytemuck`; field order/size MUST match the WGSL
/// `ClassifyCommand` (which reads `commands[cmd_idx].neighbour_base` — see below).
///
/// NOTE the WGSL `classify_brick` shares the `PackCommand` binding (`@group(0) @binding(0)`), so the classify pass
/// uploads a `GpuPackCommand`-shaped buffer too — but only `neighbour_base` is meaningful at classify time (the
/// offsets/bits are unknown until AFTER the readback). [`GpuClassifyCommand`] is the CPU-side carrier of just the
/// `neighbour_base` per dirty key (uploaded as a `GpuPackCommand` with the not-yet-known fields zeroed).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuClassifyCommand {
    /// Base index into the per-command 27-entry NEIGHBOUR TABLE ([`GpuPackBatch::neighbour_indices`]); identical
    /// meaning to [`GpuPackCommand::neighbour_base`]. The classify halo-fill reads `neighbour_indices[base + n]`.
    pub neighbour_base: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// **Stage G4 — the per-dirty-brick GPU CLASSIFY output.** One per [`GpuClassifyCommand`], written by
/// `voxel_pack.wgsl::classify_brick` and read back by [`ResidentPacker::update_gpu_finish`]. Carries the CHEAP
/// per-brick classification the [`SlabArena`] allocation needs — the dense/uniform decision + the uniform id + the
/// palette size class — computed on the GPU from the haloed brick so the CPU stops `pack_one`'ing. A FLAT 4-u32
/// (16 B) record; field order/size MUST match the WGSL `classify_out` words (see `classify_brick`). Because these
/// are a DETERMINISTIC function of the haloed brick, they EQUAL what the CPU `pack_one`+`encode_paletted` would
/// compute → the alloc is byte-identical → the pool is byte-identical (the parity gate is unchanged).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuClassifyOut {
    /// `1` = uniform-incl-halo (the brick + its halo are one solid block); `0` = dense.
    pub is_uniform: u32,
    /// The single solid block id (only when `is_uniform == 1`; `0` when dense).
    pub uniform_block: u32,
    /// The distinct-id count of the haloed cells (the palette size class; `0` when uniform).
    pub palette_k: u32,
    /// `pow2_index_bits(palette_k) ∈ {1,2,4,8,16}` (the index size class; `0` when uniform).
    pub index_bits: u32,
}

/// `GpuAabbCommand::flag` for a RESIDENT slot (write the real epsilon-grown `brick_aabb`).
const AABB_FLAG_RESIDENT: u32 = 1;
/// `GpuAabbCommand::flag` for a FREED slot (write `degenerate_aabb`).
const AABB_FLAG_FREED: u32 = 0;

impl GpuAabbCommand {
    /// A RESIDENT slot's AABB command (the GPU writes `brick_aabb(world_min, lod)`).
    #[inline]
    fn resident(slot: u32, world_min: [f32; 3], lod: u32) -> Self {
        Self {
            slot,
            lod,
            flag: AABB_FLAG_RESIDENT,
            _pad0: 0,
            world_min_x: world_min[0],
            world_min_y: world_min[1],
            world_min_z: world_min[2],
            _pad1: 0,
        }
    }

    /// A FREED slot's AABB command (the GPU writes `degenerate_aabb()`).
    #[inline]
    fn freed(slot: u32) -> Self {
        Self {
            slot,
            lod: 0,
            flag: AABB_FLAG_FREED,
            _pad0: 0,
            world_min_x: 0.0,
            world_min_y: 0.0,
            world_min_z: 0.0,
            _pad1: 0,
        }
    }
}

/// **Stage G-a — the GPU-pack output of one [`ResidentPacker::update_gpu`].** The CPU did the allocation; this
/// carries everything the render world needs to (a) `queue_write_buffer` the slots that need NO GPU encode
/// (uniform + freed bricks: meta + AABB, exactly the [`RepackDelta`] path) and the dense bricks' AABBs (the
/// AABB write stays CPU for G-a), and (b) dispatch `voxel_pack` over `commands` (reading `cores`) to encode the
/// dense bricks' index/palette/meta. The two together reconstruct the SAME buffer state the all-CPU
/// `update`+`apply_delta` would — pinned by `tests/voxel_gpu_pack_parity.rs`.
#[derive(Clone, Debug, Default)]
pub struct GpuPackBatch {
    /// Per dirty DENSE brick: the pack command the shader consumes (one workgroup each).
    pub commands: Vec<GpuPackCommand>,
    /// The DEDUPED core pool: each DISTINCT resident brick referenced by any command (as the centre OR a
    /// neighbour) contributes its `8³` core ONCE, as 512 `u32` in [`super::brickmap::voxel_index`] order. Core
    /// `i`'s voxel is `cores[i·512 + voxel_index(x,y,z)]`. Uploaded to the scratch cores SSBO. O(resident cores)
    /// — not O(commands·27) (each brick is a neighbour of up to 26 others; deduping kills that 27× blow-up).
    pub cores: Vec<u32>,
    /// The per-command 27-entry NEIGHBOUR TABLE (concatenated, `command·27 + nslot`): each entry is a core-pool
    /// index (into `cores`, in 512-cell units) or [`NEIGHBOUR_ABSENT`]. Slot 13 (`neighbour_base + 13`) is the
    /// command's own brick. Uploaded to a scratch SSBO the shader indexes via `command.neighbour_base`.
    pub neighbour_indices: Vec<u32>,
    /// Slots whose META the GPU does NOT write — uniform bricks (id rides in the meta) + freed slots (zeroed).
    /// Each is `queue_write_buffer`d into `meta_buf` at `slot · 48` exactly as the [`RepackDelta`] meta path. A
    /// DENSE slot is NOT here (the shader writes its meta). **Stage G-b: the AABB is NO LONGER carried here** —
    /// EVERY changed slot's AABB (dense/uniform/freed) is written GPU-side from [`Self::aabb_commands`], so the
    /// per-slot CPU AABB upload is gone.
    pub cpu_writes: Vec<GpuCpuWrite>,
    /// **Stage G-b** — per CHANGED slot (dense/uniform/freed) a [`GpuAabbCommand`] the GPU AABB pass
    /// (`voxel_pack.wgsl::write_aabb`) consumes to write `aabb_buf[slot]` itself (resident → `brick_aabb`,
    /// freed → `degenerate_aabb`). Replaces G-a's per-slot CPU AABB `queue_write_buffer`; runs in the SAME
    /// submission as the BLAS build (fill-then-build). Byte-equal to the CPU `SnapshotBuffers.aabbs` (the gate).
    pub aabb_commands: Vec<GpuAabbCommand>,
    /// True iff the resident brick SET changed (the BLAS/TLAS rebuild signal) — same meaning as
    /// [`RepackDelta::topology_changed`].
    pub topology_changed: bool,
}

impl GpuPackBatch {
    /// True iff nothing changed (no command, no CPU write, no AABB command).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty() && self.cpu_writes.is_empty() && self.aabb_commands.is_empty()
    }
}

/// **Stage G4 — the CLASSIFY-pass batch** ([`ResidentPacker::update_gpu_prepare`]). Carries everything the GPU
/// classify needs (the deduped core pool + the per-dirty-key 27-neighbour table + one [`GpuClassifyCommand`] per
/// dirty key), so the render world can dispatch `classify_brick` and read back the per-brick `(is_uniform,
/// index_bits, palette_k)`. The render world then calls [`ResidentPacker::update_gpu_finish`] with the readback to
/// run the `SlabArena` allocation (NO CPU `pack_one`) and produce the final [`GpuPackBatch`].
///
/// The `cores` + `neighbour_indices` here are the SAME the final pack consumes: a dense pack command reuses the
/// classify command's `neighbour_base` (the GPU pack halo-fills from the identical neighbour table), so the pool is
/// built ONCE in prepare, not twice. `command_count == commands.len()` is the number of `GpuClassifyOut` the
/// readback returns (one per dirty key, in `dirty_keys` order).
#[derive(Clone, Debug, Default)]
pub struct GpuClassifyBatch {
    /// One per dirty key (uniform OR dense), in the deterministic `dirty_keys` order. The GPU classify dispatches
    /// one workgroup per entry; the readback returns one [`GpuClassifyOut`] per entry in this order.
    pub commands: Vec<GpuClassifyCommand>,
    /// The DEDUPED core pool (same meaning as [`GpuPackBatch::cores`]) — each distinct resident brick's `8³` core
    /// ONCE. Built for the classify halo-fill; REUSED by the final pack (same pool).
    pub cores: Vec<u32>,
    /// The per-command 27-entry NEIGHBOUR TABLE (same meaning as [`GpuPackBatch::neighbour_indices`]). Built for
    /// classify; REUSED by the final pack (a dense pack command points at the same `neighbour_base`).
    pub neighbour_indices: Vec<u32>,
}

impl GpuClassifyBatch {
    /// True iff there is nothing to classify (no dirty key this generation).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

/// **Stage G4 — the deferred state [`update_gpu_prepare`] stashes for [`update_gpu_finish`].** The prepare phase did
/// the dirty-set/expansion + the drop/enter bookkeeping + the deduped core pool, but DEFERRED the per-dirty-key
/// allocation to after the GPU classify readback. This holds the ordered dirty keys + the already-emitted freed-slot
/// CPU/AABB writes + the topology flag so `finish` can resume the allocation exactly where prepare left off. Held on
/// the packer (one prepare must be followed by one finish before the next prepare).
#[derive(Clone, Debug, Default)]
struct PendingClassify {
    /// The dirty keys in the SAME order as the classify commands (`commands[i]` classifies `dirty_keys[i]`).
    dirty_keys: Vec<BrickKey>,
    /// The FREED-slot CPU meta writes prepare already emitted (drops — they need no classification).
    freed_cpu_writes: Vec<GpuCpuWrite>,
    /// The FREED-slot AABB commands prepare already emitted (drops → degenerate).
    freed_aabb_commands: Vec<GpuAabbCommand>,
    /// The per-command `neighbour_base` (= `i·27`), so `finish` reuses the classify neighbour table for the pack
    /// command of a dense brick WITHOUT rebuilding the pool.
    neighbour_bases: Vec<u32>,
    /// The topology-changed flag accumulated during prepare (drops + enters).
    topology_changed: bool,
    /// True while a prepare is awaiting its finish (debug guard against an out-of-order call).
    active: bool,
}

/// One CPU-side META write the GPU-pack batch carries (Stage G-b: META only — the AABB moved to
/// [`GpuAabbCommand`]). Emitted for a UNIFORM or FREED slot (the shader does not touch its meta): the 48-B meta to
/// `queue_write_buffer` at `slot · 48`. A DENSE slot is never here (the shader writes its meta).
#[derive(Clone, Copy, Debug)]
pub struct GpuCpuWrite {
    /// The slot (= `primitive_index`) this patches.
    pub slot: u32,
    /// The 48-B meta to write at `slot · 48`.
    pub meta: GpuBrickMeta,
}

/// The 27-neighbour slot index of a `(dx,dy,dz)` offset, each ∈ `{-1,0,1}` — `(dz+1)·9 + (dy+1)·3 + (dx+1)`.
/// Slot 13 is the centre. SSOT shared by the CPU command builder + the WGSL `pack_brick` (which recomputes it).
#[inline]
fn neighbour_slot(dx: i32, dy: i32, dz: i32) -> u32 {
    ((dz + 1) * 9 + (dy + 1) * 3 + (dx + 1)) as u32
}

/// The CHEAP per-brick GEOMETRY (a pure function of `key` — NO halo, NO neighbour reads): the AABB, the world-voxel
/// origin, the world-min corner. Recomputed in the Stage-G4 GPU-classify path (where the CPU has no [`PackedBrick`]
/// because it never ran `pack_one`), identically to the fields [`pack_one`] fills. SSOT for both producers.
struct BrickGeom {
    aabb: GpuBrickAabb,
    voxel_origin: [i32; 3],
    world_min: [f32; 3],
}

impl BrickGeom {
    fn of(key: BrickKey) -> Self {
        let coord = key.coord;
        let span = super::brickmap::brick_span(key.lod);
        let world_min = [coord.x as f32 * span, coord.y as f32 * span, coord.z as f32 * span];
        let voxel_origin = [coord.x * BRICK_EDGE, coord.y * BRICK_EDGE, coord.z * BRICK_EDGE];
        Self { aabb: super::gpu::brick_aabb(world_min, key.lod), voxel_origin, world_min }
    }
}

/// The CHEAP per-brick CLASSIFICATION the [`SlabArena`] allocation needs — the dense/uniform decision + (dense) the
/// palette size class. Produced EITHER by the CPU [`pack_one`] (the [`update_gpu`](ResidentPacker::update_gpu) path)
/// OR by the GPU classify pass + readback ([`update_gpu_finish`](ResidentPacker::update_gpu_finish), the Stage-G4
/// win — no CPU `pack_one`). The Phase-2 emit ([`emit_pack_command`](ResidentPacker::emit_pack_command)) is driven by
/// this, so the two producers converge to byte-identical allocations (the parity gate proves it).
#[derive(Clone, Copy, Debug)]
enum Classification {
    /// A uniform-incl-halo brick: its single solid block id (rides in the meta — no index/palette block).
    Uniform(super::palette::BlockId),
    /// A dense brick: its distinct-id count `k` (the palette size class; `index_bits = pow2_index_bits(k)`).
    Dense { palette_k: u32 },
}

impl Classification {
    /// The GPU classify readback ([`GpuClassifyOut`]) → a [`Classification`]. The byte-identity proof: the GPU
    /// computed `(is_uniform, uniform_block, palette_k)` as a deterministic function of the SAME haloed brick
    /// `pack_one` builds, so this equals the CPU `pack_one`'s classification.
    fn from_gpu(out: &GpuClassifyOut) -> Self {
        if out.is_uniform != 0 {
            Classification::Uniform(super::palette::BlockId(out.uniform_block as u16))
        } else {
            Classification::Dense { palette_k: out.palette_k }
        }
    }

    /// The CPU [`pack_one`] result → a [`Classification`] (the [`update_gpu`](ResidentPacker::update_gpu) path).
    fn from_packed(pb: &PackedBrick) -> Self {
        match &pb.voxels {
            BrickVoxels::Uniform(block) => Classification::Uniform(*block),
            BrickVoxels::Dense(cells) => Classification::Dense { palette_k: distinct_count(cells) },
        }
    }
}

/// Extract a brick's `8³` core as 512 `u32` block ids, in [`super::brickmap::voxel_index`] order (+X fastest).
/// The raw core data the GPU halo-fill reads (the CPU never packs the bytes in the GPU path). Mirror of the
/// inner read `pack_one` does (`brick.get(cx,cy,cz)`), so the GPU produces the same haloed cells.
/// The number of DISTINCT block ids in a brick's haloed `cells` — the palette size `k` that picks the index +
/// palette size class. Equals `encode_paletted(cells).palette.len()` (first-seen ORDER doesn't change the SET),
/// computed here without building the bit-packed stream (the GPU does that). The size classes (`pow2_index_bits`
/// + `palette_classes`) depend only on `k`, so this is the only palette fact the CPU allocator needs.
fn distinct_count(cells: &[u32]) -> u32 {
    let mut seen: rustc_hash::FxHashSet<u32> = rustc_hash::FxHashSet::default();
    for &c in cells {
        seen.insert(c & 0xFFFF); // ids are u16 zero-extended (mirror `encode_paletted`'s `c as u16`)
    }
    seen.len() as u32
}

fn extract_core(brick: &Brick) -> [u32; BRICK_VOXELS] {
    let mut core = [0u32; BRICK_VOXELS];
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                core[(x + y * BRICK_EDGE + z * BRICK_EDGE * BRICK_EDGE) as usize] = brick.get(x, y, z).0 as u32;
            }
        }
    }
    core
}

/// The fixed number of `u32`s a DENSE brick's RAW haloed `10³` grid occupies (one `u32` id per cell). CONSTANT at
/// every LOD ([`halo_cells`]`(0) == halo_cells(lod)`). The packer's per-slot SHADOW (`last_voxels`) stores cells
/// in this raw form (so the byte-identity gate + the A4.4 re-encode are exact); the GPU arena stores the
/// PALETTED index stream (≤ this size — see [`index_class_words`]).
#[inline]
pub fn dense_block_u32() -> usize {
    halo_cells(0)
}

/// **A4.4 — the 5 paletted-index SIZE CLASSES**, keyed by `index_bits ∈ {1,2,4,8,16}`. A dense brick's bit-packed
/// index stream is `ceil(dense_block_u32() · index_bits / 32)` words; for the `10³` haloed block that is
/// `{32, 63, 125, 250, 500}` words. Each class is a fixed-size free-list (no fragmentation WITHIN a class), so a
/// freed block is always exactly reusable by another brick of the same width.
const INDEX_CLASS_BITS: [u8; 5] = [1, 2, 4, 8, 16];

/// Words a dense brick's bit-packed index stream occupies for `index_bits ∈ {1,2,4,8,16}` (its size-class block
/// size). Mirror of [`encode_paletted`]'s `indices.len()`.
#[inline]
pub fn index_class_words(index_bits: u8) -> usize {
    (dense_block_u32() * index_bits as usize).div_ceil(32)
}

/// The power-of-2 PALETTE size-class ladder (A4.4 Checkpoint-2): a dense brick's per-brick palette of `k` distinct
/// ids is stored in the smallest `2^j ≥ k` class. Covers any `u16`-id registry (`k ≤ 65536`); unused large
/// classes are never bump-allocated so they cost nothing. Smallest is 2 (a dense brick always has `k ≥ 2`).
fn palette_classes() -> Vec<u32> {
    (1..=16).map(|j| 1u32 << j).collect() // {2, 4, 8, …, 65536}
}

/// A generic SIZE-CLASS SLAB allocator over a single GPU buffer (A4.4): a bump high-water + a per-class free-list
/// of freed block word-offsets. An allocation of `words` takes the smallest class `≥ words`; a free returns the
/// block to that class (exactly reusable — no fragmentation WITHIN a class). The arena GROWS on overflow past the
/// committed GPU capacity ([`Self::commit`]); `grew` then forces the next upload to a StreamSnapshot (re-allocate
/// the larger buffer). ONE SSOT for both the index-stream arena and the per-brick palette arena.
#[derive(Clone, Debug)]
struct SlabArena {
    /// Block sizes (in `u32` words) per class, ASCENDING. An alloc rounds up to the smallest class that fits.
    classes: Vec<u32>,
    /// Per-class free-list of freed block word-offsets.
    free: Vec<Vec<u32>>,
    /// Bump pointer (words) for never-yet-allocated blocks.
    high_water: u32,
    /// The word capacity the last [`commit`](Self::commit) sized the GPU buffer to (grow detection).
    gpu_cap: u32,
    /// A PRE-SIZED capacity floor (words) — the first [`commit`](Self::commit) sizes the GPU buffer to at least
    /// this, so streaming a full resident shell fits WITHOUT a mid-load grow ([`reserve`](Self::reserve)). 0 = no
    /// reserve (legacy behaviour: the buffer grows from the live high-water). A genuine overflow PAST the reserve
    /// still grows safely (the grow path is unchanged) — the reserve only ensures a NORMAL load never triggers it.
    reserved_floor: u32,
    /// Set when an alloc since the last commit exceeded `gpu_cap` — the next upload must re-snapshot.
    grew: bool,
}

impl SlabArena {
    fn new(classes: Vec<u32>) -> Self {
        let n = classes.len();
        Self { classes, free: vec![Vec::new(); n], high_water: 0, gpu_cap: 0, reserved_floor: 0, grew: false }
    }

    /// PRE-SIZE the arena to an AGGREGATE capacity of `blocks · mean_words_per_block` words: raise the
    /// [`reserved_floor`](Self::reserved_floor) so the FIRST [`commit`](Self::commit) sizes the GPU buffer large
    /// enough for a full normal load, eliminating the mid-load GROW that would otherwise force a full O(capacity)
    /// re-snapshot. `mean_words_per_block` is an AGGREGATE mean across the resident set (NOT a single block's class
    /// — the floor is a total-pool capacity, so the mean is what bounds it, not the per-brick worst case). A later
    /// overflow beyond this floor still grows — the reserve is a no-grow guarantee for the COMMON load only.
    fn reserve(&mut self, blocks: u32, mean_words_per_block: u32) {
        self.reserved_floor = self.reserved_floor.max(blocks.saturating_mul(mean_words_per_block));
    }

    /// The class index whose block size is the smallest `≥ words`.
    #[inline]
    fn class_of(&self, words: u32) -> usize {
        self.classes
            .iter()
            .position(|&c| c >= words)
            .unwrap_or_else(|| panic!("block of {words} words exceeds the largest slab class {:?}", self.classes))
    }

    /// Allocate a block large enough for `words`; returns its word offset. Reuses a freed block of the class, else
    /// bumps the high-water by the class size (setting `grew` if it passes the committed GPU capacity).
    fn alloc(&mut self, words: u32) -> u32 {
        let cls = self.class_of(words);
        if let Some(off) = self.free[cls].pop() {
            return off;
        }
        let off = self.high_water;
        self.high_water += self.classes[cls];
        if self.high_water > self.gpu_cap {
            self.grew = true;
        }
        off
    }

    /// Return a block (allocated for `words`) to its class free-list.
    #[inline]
    fn free_block(&mut self, off: u32, words: u32) {
        let cls = self.class_of(words);
        self.free[cls].push(off);
    }

    /// The buffer capacity (words) to allocate at the next snapshot: the live high-water + headroom (so a few
    /// deltas can allocate before forcing a re-snapshot), with a small floor so an empty arena is still non-empty,
    /// AND clamped up to the [`reserved_floor`](Self::reserved_floor) — the pre-sized capacity for a full normal
    /// load — so the first snapshot is large enough to absorb the whole streamed shell without a mid-load grow.
    #[inline]
    fn capacity_u32(&self) -> usize {
        let hw = self.high_water as usize;
        (hw * 2)
            .max(hw + 8192)
            .max(self.classes.last().copied().unwrap_or(1) as usize)
            .max(self.reserved_floor as usize)
    }

    /// Commit a (re)allocation to [`capacity_u32`](Self::capacity_u32): record it as `gpu_cap` + clear `grew`.
    /// Returns the committed capacity (words) the GPU buffer is sized to.
    fn commit(&mut self) -> usize {
        let cap = self.capacity_u32();
        self.gpu_cap = cap as u32;
        self.grew = false;
        cap
    }
}

/// Map an `index_bits ∈ {1,2,4,8,16}` to its index-stream slab block size in words (the alloc request for the
/// index [`SlabArena`]). The arena rounds it to its exact class.
#[inline]
fn index_slab_words(index_bits: u8) -> u32 {
    index_class_words(index_bits) as u32
}

/// **Pre-size estimate — the MEAN index-stream words a RESIDENT brick costs** (the index-arena pre-grow per
/// brick; [`ResidentPacker::new`] → [`SlabArena::reserve`]). The reserve is an AGGREGATE pool capacity, so the
/// figure that bounds it is the MEAN over the whole resident set, not a single brick's class. MEASURED on the
/// A4.4 worldgen slice: the index arena settled at **6.4 MB / 10 k resident bricks ≈ 160 words/brick** (the mean
/// over the dense/uniform mix — uniform bricks cost 0 index words, dense bricks 32–500 by their `index_bits`).
/// We reserve at the **largest index class (500 words, `index_bits = 16`), rounded to 512** so the pool is sized
/// correct-by-construction for ANY brick mix: every dense brick costs ≤ 500 ≤ 512 index words, so a converged set
/// of `max_resident` bricks NEVER overflows `max_resident · 512` — true for the CPU grow path AND, critically, for
/// the FIXED (no-grow) GPU residency FRONT END pool (`residency_front_end.rs`), which binds this buffer once and
/// writes the whole GPU-decided set into it. The index stream is `ceil(1000·index_bits/32)` words (1000 = the 10³
/// haloed grid); the MAX SUPPORTED width is `index_bits=8` ⇒ **250 words** — `index_bits=16` (500 words) is
/// DEGENERATED to empty by the pack (`voxel_residency.wgsl` D3 `fits` guard: a brick whose index/palette overflows
/// its slab gets a degenerate AABB, never spills). So **256 bounds every packable brick by construction** (the
/// guard makes it robust at any stride — an over-large brick is dropped, never corrupts). Halved from the earlier
/// 512 (which was sized for `index_bits=16` BEFORE the pack degenerated it — stale). Cost: the index GPU buffer is
/// committed at `max_resident_bricks · 256 · 4 B` regardless of the live count; a low-entropy scene leaves it
/// partly unused. (8 GB budget: this halves the index pool, ~1.84 GB → ~0.92 GB at max_resident=900k.)
const RESERVE_INDEX_WORDS_PER_BRICK: u32 = 256;

/// **Pre-size bound — the per-brick PALETTE words pool reserve.** This is NOT a soft mean estimate: on the GPU
/// residency path the palette pool is a HARD-CAPPED bump arena (`voxel_residency.wgsl::alloc_palette_slab`) sized
/// EXACTLY `max_resident_bricks · this · 4 B`, and the GPU bump has no readback to grow it. The old MEAN value (16
/// words ≈ the measured low-entropy worldgen mean) was a latent OOB: an `index_bits=8` brick (a high-registry
/// `.vox`/voxelized scene — Sponza/Sibenik/etc.) carries up to a 256-word palette, so once enough rich-palette
/// bricks are concurrently resident the bump high-water ran PAST the pool → overlapping/OOB palette slabs → bricks
/// decode through a WRONG palette → garbage-content cubes (the user's coarse-LOD "jumbled colours", repro'd by
/// `tests/voxel_paged_front_end_render.rs::rich_palette_pool_no_content_corruption`). We now reserve the
/// `index_bits ≤ 8` WORST case (256 words/brick), MIRRORING the index pool's worst-case `RESERVE_INDEX_WORDS_PER_BRICK`
/// (512 = the `index_bits=16` block) — so a scene of all-`index_bits=8` bricks can NEVER overflow by construction. A
/// pathological `index_bits=16` brick (>256 distinct ids, up to a 1024-word palette) is the ONLY remaining overflow
/// path; `alloc_palette_slab`'s capacity GUARD degrades it to uniform rather than corrupting (a hard backstop, not a
/// floor — it is unreachable for the ≤8 scenes this reserve covers).
const RESERVE_PALETTE_WORDS_PER_BRICK: u32 = 256;

/// A DEGENERATE BLAS AABB (min > max on every axis) for an UNUSED slot: the BLAS build never reports a
/// candidate for it, so a freed slot is invisible to the trace. `primitive_index = slot` is preserved (the
/// buffers stay contiguous + fixed-capacity); only this slot's box is collapsed so the ray query skips it.
#[inline]
pub fn degenerate_aabb() -> GpuBrickAabb {
    GpuBrickAabb { min: [1.0e30, 1.0e30, 1.0e30], max: [-1.0e30, -1.0e30, -1.0e30], _pad: [0.0; 2] }
}

/// The 26 SAME-LOD face/edge/corner neighbours of a brick key (the full one-ring around it). The packer
/// expands every changed key by this set because the halo of each of those neighbours reads the changed
/// brick's core (and a drop flips that halo to AIR / a neighbour change can toggle the brick uniform↔dense) —
/// missing one leaves a stale halo seam or a wrong R1 classification. 26 is the robust-by-construction choice
/// (the dense halo-fill reads diagonal neighbours too, not just the 6 faces).
fn neighbourhood_26(key: BrickKey) -> impl Iterator<Item = BrickKey> {
    let base = key.coord;
    let lod = key.lod;
    (-1..=1).flat_map(move |dz| {
        (-1..=1).flat_map(move |dy| {
            (-1..=1).filter_map(move |dx| {
                if dx == 0 && dy == 0 && dz == 0 {
                    None
                } else {
                    Some(BrickKey { coord: base + IVec3::new(dx, dy, dz), lod })
                }
            })
        })
    })
}

/// A free-list allocator over `[0, capacity)` slot indices. Each resident [`BrickKey`] holds one slot for life
/// (its `primitive_index`); a dropped key frees its slot back. Reuse is bounded; the render never depends on
/// slot ORDER (only `primitive_index → meta`).
#[derive(Clone, Debug)]
struct SlotAllocator {
    capacity: u32,
    /// The next never-yet-allocated slot (bump pointer) until the free list is the only source.
    high_water: u32,
    /// Freed slots available for reuse (LIFO).
    free: Vec<u32>,
}

impl SlotAllocator {
    fn new(capacity: u32) -> Self {
        Self { capacity, high_water: 0, free: Vec::new() }
    }

    /// Claim a free slot, or `None` if at capacity. Prefers the bump pointer first so a FRESH fill lays slots
    /// out in claim order; the free list only kicks in after drops.
    fn claim(&mut self) -> Option<u32> {
        if self.high_water < self.capacity {
            let s = self.high_water;
            self.high_water += 1;
            Some(s)
        } else {
            self.free.pop()
        }
    }

    /// Return a slot to the free list.
    fn release(&mut self, slot: u32) {
        debug_assert!(slot < self.high_water, "releasing a never-claimed slot");
        self.free.push(slot);
    }
}

/// One slot's new GPU bytes after an incremental re-pack: the slot index (= `primitive_index`), its meta, its
/// AABB, and — for a dense brick whose content changed — the PALETTED index block + per-brick palette block to
/// write (A4.4). The GPU uploader patches `metas[slot]`, `aabbs[slot]`, the index arena at `index_word_offset`,
/// and `brick_palettes` at `palette_word_offset` from this.
#[derive(Clone, Debug)]
pub struct ChangedSlot {
    /// The slot whose buffers this patches (= `primitive_index`).
    pub slot: u32,
    /// The new per-brick meta (carries the real `voxel_offset` (index-arena word) + `index_bits` + `palette_base`
    /// for a dense brick, or the uniform flag/id; all-zero for a freed slot).
    pub meta: GpuBrickMeta,
    /// The new BLAS AABB ([`degenerate_aabb`] for a freed slot).
    pub aabb: GpuBrickAabb,
    /// `Some(words)` for a DENSE brick whose voxel content changed: the bit-packed INDEX stream
    /// ([`index_class_words`]`(index_bits)` words) to write at `index_word_offset`. `None` for a uniform/freed
    /// slot, or a dense brick whose meta changed but whose voxel content did not.
    pub index: Option<Vec<u32>>,
    /// The index-arena WORD offset the index block goes at (= `meta.dense_offset()` when `index.is_some()`).
    pub index_word_offset: u32,
    /// `Some(ids)` for a DENSE brick whose voxel content changed: the per-brick palette (its `k` distinct block
    /// ids, one `u32` each) to write into `brick_palettes` at `palette_word_offset`. Paired with `index` (same
    /// `Some`/`None`).
    pub palette: Option<Vec<u32>>,
    /// The `brick_palettes` WORD offset the palette block goes at (= `meta.palette_base` = the brick's palette
    /// slab offset, A4.4 Checkpoint-2).
    pub palette_word_offset: u32,
}

/// The set of slots an incremental [`update`](ResidentPacker::update) changed, so the GPU side patches ONLY
/// these via `queue_write_buffer` (O(changed), not O(resident)).
#[derive(Clone, Debug, Default)]
pub struct RepackDelta {
    /// Slots whose `meta`/`aabb` (and, for dense bricks, voxel block) changed — re-upload these.
    pub changed: Vec<ChangedSlot>,
    /// Slots that became UNUSED this update (their AABB is now degenerate + meta zeroed; also present in
    /// `changed`). For the AS-topology-changed signal / bookkeeping.
    pub freed: Vec<u32>,
    /// True iff the resident brick SET (which slots are occupied) changed — the signal the BLAS/TLAS need
    /// rebuilding. A pure meta/voxel edit with no enter/drop leaves AABB occupancy unchanged (the AS can be
    /// refit/kept), but conservatively any entered/dropped brick sets this.
    pub topology_changed: bool,
}

impl RepackDelta {
    /// True iff nothing changed (no slot touched) — the caller can skip the GPU patch entirely.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.changed.is_empty()
    }
}

/// The FIXED-CAPACITY initial buffer contents the render path allocates ONCE per scene epoch (storage plan A1:
/// the O(changed) GPU upload). Built from the packer's current shadow state by [`ResidentPacker::snapshot_buffers`].
/// After this snapshot the render path applies each generation's [`RepackDelta`] via `queue_write_buffer` — it
/// NEVER re-creates these buffers within an epoch (only a scene switch / new epoch re-snapshots).
///
/// **A4.4 representation — the streamed arena is R2b PALETTED (size-class slabs).** Each dense slot's voxel block
/// is a bit-packed INDEX stream ([`encode_paletted`]) living in its `index_bits` size-class slab in [`Self::indices`]
/// at `meta.voxel_offset`, and a per-brick PALETTE (its `k` distinct ids) in its palette size-class slab in
/// [`Self::brick_palettes`] at `meta.palette_base`. The shader's `cell_block` paletted branch
/// (`index_bits >= 1`) decodes `id = brick_palettes[palette_base + (indices[voxel_offset + ..] >> .. & mask)]` —
/// the SAME decode the static `pack_brickmap`/`pack_resident_set` path uses. This recovers R2b's voxel-VRAM win on
/// the STREAMED path while keeping A1's O(changed) upload — see `PHASE_A_GPU_EXECUTION.md` §A4.4 + the scoping note.
#[derive(Clone, Debug)]
pub struct SnapshotBuffers {
    /// Capacity-length AABB buffer; unused slots = [`degenerate_aabb`].
    pub aabbs: Vec<GpuBrickAabb>,
    /// Capacity-length meta buffer; unused slots = [`GpuBrickMeta::zeroed`].
    pub metas: Vec<GpuBrickMeta>,
    /// The index-arena slab pool ([`Self::index_capacity_u32`] words): each resident dense brick's bit-packed
    /// index stream written at its slab offset (`meta.voxel_offset`); zero elsewhere.
    pub indices: Vec<u32>,
    /// The per-brick PALETTE slab pool (A4.4 Checkpoint-2 — sized to the palette [`SlabArena`]'s committed
    /// capacity): each resident dense brick's `k` distinct block ids written at its palette slab offset
    /// (`meta.palette_base`); zero elsewhere. Bound at `group(0)/binding(12)`. A `RepackDelta` patches only changed
    /// slots' blocks.
    pub brick_palettes: Vec<u32>,
    /// The registry palette (`BlockId(i)` → linear RGBA + emissive). Length == registry length. Fixed per scene
    /// (never re-uploaded on a [`RepackDelta`]).
    pub palette: Vec<GpuPaletteColor>,
    /// The number of resident bricks (== the live BLAS primitive count; the BLAS is still built over `capacity`
    /// primitives with degenerate free slots, but this is reported for diagnostics).
    pub brick_count: u32,
}

/// A DENSE brick's slab allocations (A4.4): where its bit-packed index stream lives in the index arena (+ its
/// width) AND where its per-brick palette lives in the palette arena (+ its `k`) — the offsets/sizes needed to
/// free both blocks back to their classes on a drop / class change.
#[derive(Clone, Copy, Debug)]
struct DenseSlot {
    /// Word offset of this brick's index stream in the index arena (= `meta.voxel_offset`).
    index_offset: u32,
    /// Index bit width ∈ {1,2,4,8,16} — the index size class.
    index_bits: u8,
    /// Word offset of this brick's palette in the palette arena (= `meta.palette_base`).
    palette_offset: u32,
    /// Palette length `k` (distinct ids) — the palette size class ([`palette_classes`]).
    palette_k: u32,
}

/// One resident brick's live state in the packer: its slot + (for a dense brick) its index-slab allocation.
#[derive(Clone, Copy, Debug)]
struct SlotState {
    slot: u32,
    /// `Some` for a DENSE brick (its index-slab allocation); `None` for a UNIFORM brick (no index/palette block).
    dense: Option<DenseSlot>,
}

/// The incremental resident-set packer: owns the slot/arena allocators + the live `key → SlotState` map +
/// shadow copies of the last-uploaded meta/aabb/voxels per slot (so it can emit a [`ChangedSlot`] ONLY when
/// bytes actually differ). The render path holds one alongside the GPU buffers;
/// [`update`](Self::update) returns the [`RepackDelta`] to upload.
///
/// Robust-by-construction: every brick's bytes come from the SSOT [`pack_one`], so an incremental patch equals
/// a from-scratch [`pack_resident_set`](super::gpu::pack_resident_set) for the same `key → brick` mapping (the
/// A/B test). The dirty set is EXPANDED by the 26-neighbourhood so no halo/uniform-classification goes stale.
pub struct ResidentPacker {
    slots: SlotAllocator,
    /// Live resident bricks → their slot + index-slab allocation.
    resident: FxHashMap<BrickKey, SlotState>,
    /// **A4.4 index-stream SIZE-CLASS SLAB arena** ([`INDEX_CLASS_BITS`] → `{32,63,125,250,500}`-word classes). A
    /// dense brick's bit-packed index stream is allocated from its `index_bits` class; a freed block returns to
    /// that class. Grows on overflow (forces a re-snapshot).
    index_arena: SlabArena,
    /// **A4.4 Checkpoint-2 per-brick PALETTE SIZE-CLASS SLAB arena** ([`palette_classes`] power-of-2 ladder). A
    /// dense brick's `k`-id palette is allocated from the smallest `2^j ≥ k` class — variable, so a 2-id brick
    /// costs 2 words (not the registry length). Replaces Checkpoint-1's fixed `slot · registry.len()` palette.
    palette_arena: SlabArena,
    /// The packing registry length (= the largest possible `k`) — set each [`update`](Self::update) for the
    /// `k ≤ palette_stride` debug invariant. The palette ladder covers any `u16` registry regardless.
    palette_stride: u32,
    /// Shadow of the last-uploaded bytes per slot — the byte-compare source so a re-pack that reproduces the
    /// same bytes (a neighbour that did not actually change) costs no upload. `last_voxels` holds RAW haloed cells
    /// (the A4.4 re-encode source — kept raw so the byte-identity gate is exact).
    last_meta: FxHashMap<u32, GpuBrickMeta>,
    last_aabb: FxHashMap<u32, GpuBrickAabb>,
    last_voxels: FxHashMap<u32, Vec<u32>>,
    /// Keys the caller explicitly marked rewritten (edit / dig re-source) since the last update — re-packed even
    /// though they neither entered nor dropped. Drained each update.
    pending_rewrites: Vec<BrickKey>,
    /// DEFERRED-FREE quarantine (keep-old-until-revealed): slots/index/palette blocks dropped THIS update can't be
    /// reused until the NEXT update, so an in-flight frame tracing the old generation never sees a slot's bytes
    /// overwritten by a different brick mid-flight. Released at the top of the next update.
    quarantine_slots: Vec<u32>,
    /// Freed index blocks `(word_offset, index_bits)` — returned to their index size-class free-list next update.
    quarantine_index: Vec<(u32, u8)>,
    /// Freed palette blocks `(word_offset, k)` — returned to their palette size-class free-list next update.
    quarantine_palette: Vec<(u32, u32)>,
    /// **Stage G4** — the deferred state a [`update_gpu_prepare`](Self::update_gpu_prepare) stashes for the matching
    /// [`update_gpu_finish`](Self::update_gpu_finish) (the GPU-classify split of [`update_gpu`](Self::update_gpu)).
    /// `active == false` outside a prepare/finish pair.
    pending_classify: PendingClassify,
}

impl ResidentPacker {
    /// A fresh packer sized for `max_resident_bricks` slots. The index arena (A4.4 size-class slabs) starts empty
    /// and grows as bricks are slotted; the render path sizes the GPU index buffer to [`index_capacity_u32`](Self::index_capacity_u32)
    /// at each snapshot. `palette_stride` starts 0 and is set from the registry length on the first [`update`].
    pub fn new(max_resident_bricks: u32) -> Self {
        // PRE-SIZE both arenas to the resident CAP so streaming a full shell fits the FIRST snapshot — no mid-load
        // GROW re-snapshot (the ~200 ms `vox_pack_snapshot` spikes). This is the Tier-1 grow-snapshot fix.
        Self::with_reserve(max_resident_bricks, true)
    }

    /// `new` WITHOUT the arena pre-size (the legacy grow-from-empty behaviour). Used by the grow-snapshot
    /// benchmark to measure BEFORE (un-pre-sized) vs AFTER (pre-sized) — production always uses [`new`](Self::new).
    #[doc(hidden)]
    pub fn new_unreserved(max_resident_bricks: u32) -> Self {
        Self::with_reserve(max_resident_bricks, false)
    }

    fn with_reserve(max_resident_bricks: u32, reserve: bool) -> Self {
        let index_classes: Vec<u32> = INDEX_CLASS_BITS.iter().map(|&b| index_class_words(b) as u32).collect();
        let mut index_arena = SlabArena::new(index_classes);
        let mut palette_arena = SlabArena::new(palette_classes());
        if reserve {
            // Derived from `max_resident_bricks` × the MEASURED MEAN per-brick words (see
            // `RESERVE_INDEX_WORDS_PER_BRICK`/`RESERVE_PALETTE_WORDS_PER_BRICK`); a load denser than the mean still
            // grows safely (the grow path is intact), so this kills the COMMON-load grow-snapshots without removing
            // the safety net.
            index_arena.reserve(max_resident_bricks, RESERVE_INDEX_WORDS_PER_BRICK);
            palette_arena.reserve(max_resident_bricks, RESERVE_PALETTE_WORDS_PER_BRICK);
        }
        Self {
            slots: SlotAllocator::new(max_resident_bricks),
            resident: FxHashMap::default(),
            index_arena,
            palette_arena,
            palette_stride: 0,
            last_meta: FxHashMap::default(),
            last_aabb: FxHashMap::default(),
            last_voxels: FxHashMap::default(),
            pending_rewrites: Vec::new(),
            quarantine_slots: Vec::new(),
            quarantine_index: Vec::new(),
            quarantine_palette: Vec::new(),
            pending_classify: PendingClassify::default(),
        }
    }

    /// The fixed slot CAPACITY (= the meta/AABB buffer length, in bricks).
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.slots.capacity
    }

    /// Number of resident bricks currently slotted.
    #[inline]
    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    /// Build the FULL initial GPU buffers (storage plan A1 — the O(changed) upload). Called ONCE per scene epoch
    /// (and again on a grow), right after the first [`update`](Self::update) ran. `aabbs`/`metas` are
    /// capacity-length with [`degenerate_aabb`]/[`GpuBrickMeta::zeroed`] in unused slots; `indices` is the index
    /// slab pool with each resident dense brick's bit-packed index stream at its slab offset (`meta.voxel_offset`);
    /// `brick_palettes` is the palette slab pool with each dense brick's `k`-id palette at its slab offset
    /// (`meta.palette_base`). Both pools are sized to their [`SlabArena`]'s committed capacity. O(resident) — paid
    /// ONCE per epoch/grow, never per move. After this the render path applies each [`RepackDelta`] via
    /// `queue_write_buffer` (meta/aabb at `slot · stride`, index at `index_word_offset`, palette at
    /// `palette_word_offset`), never re-creating these buffers within the epoch.
    ///
    /// The bytes come from the per-slot shadow (`last_meta`/`last_aabb` + the RE-ENCODED `last_voxels` raw cells),
    /// which after `update` is EXACTLY the state a `snapshot_buffers`-then-delta sequence converges to (the
    /// byte-identity gate). Re-encoding `last_voxels` here with [`encode_paletted`] reproduces the SAME index/
    /// palette bytes `emit_changed_slot` shipped (the shadow stores the offsets in `metas`). The `palette` is the
    /// registry (fixed per scene); the NEE light list is built separately by the caller.
    pub fn snapshot_buffers(&mut self, registry: &super::palette::BlockRegistry) -> SnapshotBuffers {
        let cap = self.slots.capacity as usize;
        // Capacity-length meta/aabb: each occupied slot's shadow; degenerate/zeroed for the rest. The shadow
        // holds an entry for every slot that has EVER been written (occupied or freed→zeroed), so a slot absent
        // from the shadow is one that was never claimed — fill it degenerate/zeroed too.
        let mut metas = vec![GpuBrickMeta::zeroed(); cap];
        let mut aabbs = vec![degenerate_aabb(); cap];
        for (&slot, meta) in &self.last_meta {
            metas[slot as usize] = *meta;
        }
        for (&slot, aabb) in &self.last_aabb {
            aabbs[slot as usize] = *aabb;
        }
        // A4.4 index + palette slabs: each resident DENSE slot's bit-packed index stream at its index slab offset
        // + its `k` palette ids at its palette slab offset. `last_voxels` holds the RAW cells; re-encode them
        // (byte-identical to what `emit_changed_slot` shipped). The meta's `voxel_offset`/`palette_base`/
        // `index_bits` are the canonical offsets (written by `emit_changed_slot`), so writing there keeps a fresh
        // snapshot byte-identical to seed+deltas. A uniform brick has no dense slot / voxel shadow — its id rides
        // in the meta. `commit` records each pool's GPU capacity + clears its grow flag (this snapshot IS the
        // (re)allocation); a later delta-time alloc past it re-sets the flag.
        let index_cap = self.index_arena.commit();
        let palette_cap = self.palette_arena.commit();
        let mut indices = vec![0u32; index_cap];
        let mut brick_palettes = vec![0u32; palette_cap];
        for st in self.resident.values() {
            let Some(dense) = st.dense else { continue };
            let Some(cells) = self.last_voxels.get(&st.slot) else { continue };
            let pb = encode_paletted(cells);
            debug_assert_eq!(pb.index_bits, dense.index_bits, "shadow index_bits disagrees with re-encode");
            debug_assert_eq!(pb.palette.len() as u32, dense.palette_k, "shadow palette_k disagrees with re-encode");
            let ioff = dense.index_offset as usize;
            indices[ioff..ioff + pb.indices.len()].copy_from_slice(&pb.indices);
            let poff = dense.palette_offset as usize;
            for (j, &id) in pb.palette.iter().enumerate() {
                brick_palettes[poff + j] = id as u32;
            }
        }
        // Palette = the registry (block id → linear RGBA + emissive), indexed directly. Fixed per scene.
        let palette: Vec<GpuPaletteColor> = (0..registry.len())
            .map(|i| {
                let id = super::palette::BlockId(i as u16);
                let e = registry.emissive(id);
                GpuPaletteColor { rgba: registry.color(id), emissive: [e[0], e[1], e[2], 0.0] }
            })
            .collect();
        SnapshotBuffers { aabbs, metas, indices, brick_palettes, palette, brick_count: self.resident.len() as u32 }
    }

    /// Peek the GROW signal without clearing it (the render path's snapshot-vs-delta decision). True iff an index OR
    /// palette allocation since the last snapshot exceeded its committed GPU buffer capacity — ship a StreamSnapshot.
    #[inline]
    pub fn grew(&self) -> bool {
        self.index_arena.grew || self.palette_arena.grew
    }

    /// **Stage G-wire — repopulate the raw-cell shadow (`last_voxels`) for EVERY resident dense slot from `entries`.**
    /// The GPU-classify path ([`update_gpu_finish`](Self::update_gpu_finish)) DROPS `last_voxels` for the bricks it
    /// packs (the GPU owns those bytes; the CPU never built the cells). [`snapshot_buffers`](Self::snapshot_buffers)
    /// re-encodes `last_voxels` to fill the dense pool, so it would emit ZERO voxel/palette for any GPU-packed slot.
    /// In the live flow that is harmless — the ONLY snapshot is the FIRST pack of an epoch, which runs the CPU
    /// [`update_gpu`](Self::update_gpu) path (cells kept), BEFORE any GPU `finish` drops them. But a later GROW past
    /// the pre-sized reserve ([`grew`](Self::grew)) would force a SECOND snapshot AFTER GPU finishes ran. This method
    /// restores the shadow for that rare case: it re-`pack_one`s every resident dense brick (the SSOT halo-fill — the
    /// SAME bytes the GPU pack wrote) so the subsequent `snapshot_buffers` is byte-correct again. O(resident) — paid
    /// ONLY on a grow (never on the steady-state delta ticks), so it does not cost the G4 win. `entries` must be the
    /// CURRENT resident set (the one the just-run `update_gpu_finish` reconciled toward), so every resident key is
    /// present in the rebuilt `by_key`.
    pub fn repopulate_last_voxels(&mut self, entries: &[ResidentBrick<'_>]) {
        let new_by_key: FxHashMap<BrickKey, &Brick> =
            entries.iter().map(|e| (BrickKey { coord: e.coord, lod: e.lod }, e.brick)).collect();
        let by_key = build_by_key(entries);
        // Re-pack every resident DENSE slot whose raw-cell shadow is missing (a GPU-packed slot). A uniform slot has
        // no `last_voxels` entry by design (its id rides in the meta); a dense slot the CPU path packed still has it.
        let dense_keys: Vec<(BrickKey, u32)> = self
            .resident
            .iter()
            .filter_map(|(&k, st)| st.dense.map(|_| (k, st.slot)))
            .collect();
        for (key, slot) in dense_keys {
            if self.last_voxels.contains_key(&slot) {
                continue; // already valid (a CPU-path slot) — nothing to restore
            }
            let Some(&brick) = new_by_key.get(&key) else { continue };
            let e = ResidentBrick { coord: key.coord, brick, lod: key.lod };
            let pb = pack_one(&e, &by_key);
            if let BrickVoxels::Dense(cells) = &pb.voxels {
                self.last_voxels.insert(slot, cells.clone());
            }
        }
    }

    /// Mark `keys` as REWRITTEN (an edit / dig re-source replaced their voxels in place): they re-pack on the
    /// next [`update`](Self::update) even though they neither entered nor dropped. Mirrors the manager's
    /// `requeue_keys` so the edit/dig path stays incremental (only the affected bricks + their 26-neighbourhood
    /// re-pack). A key that is not resident on the next update is ignored (it may enter then, taking the normal
    /// entered path).
    pub fn mark_rewritten(&mut self, keys: impl IntoIterator<Item = BrickKey>) {
        self.pending_rewrites.extend(keys);
    }

    /// Incrementally reconcile the packer toward `entries` (the manager's `resident_entries()`, in the SSOT
    /// `(lod,z,y,x)` order). Returns the [`RepackDelta`] of slots whose GPU bytes changed — O(changed + halo),
    /// never O(resident). The CALLER uploads only `delta.changed` via `queue_write_buffer`. `palette_stride` is the
    /// packing registry's length (which bounds a brick's `k`) — used for the `k ≤ palette_stride` debug invariant.
    pub fn update(&mut self, entries: &[ResidentBrick<'_>], palette_stride: u32) -> RepackDelta {
        self.update_inner(entries, palette_stride, true)
    }

    /// [`update`](Self::update) with the Phase-1 re-pack forced SERIAL (no rayon) — production always uses the
    /// PARALLEL `update`. Exposed ONLY for the parallel-vs-serial pack benchmark (`examples/g2_pack_parallel.rs`)
    /// to measure the speed-up; the two paths emit byte-identical deltas (Phase 1 is a pure map; the
    /// order-dependent Phase 2 fold is serial + identical in both).
    #[doc(hidden)]
    pub fn update_serial(&mut self, entries: &[ResidentBrick<'_>], palette_stride: u32) -> RepackDelta {
        self.update_inner(entries, palette_stride, false)
    }

    fn update_inner(&mut self, entries: &[ResidentBrick<'_>], palette_stride: u32, parallel: bool) -> RepackDelta {
        debug_assert!(
            self.palette_stride == 0 || self.palette_stride == palette_stride,
            "palette_stride must be constant within an epoch ({} → {palette_stride})",
            self.palette_stride,
        );
        self.palette_stride = palette_stride;
        // (1) Deferred-free: last update's quarantined slots/index blocks are now safe to reuse (the frame that
        // could still be tracing them has been submitted). Release them BEFORE claiming this update's slots.
        for s in self.quarantine_slots.drain(..) {
            self.slots.release(s);
        }
        for (off, bits) in std::mem::take(&mut self.quarantine_index) {
            self.index_arena.free_block(off, index_slab_words(bits));
        }
        for (off, k) in std::mem::take(&mut self.quarantine_palette) {
            self.palette_arena.free_block(off, k);
        }

        // The NEW resident map (key → brick) + the by_key index pack_one reads for halos.
        let new_by_key: FxHashMap<BrickKey, &Brick> =
            entries.iter().map(|e| (BrickKey { coord: e.coord, lod: e.lod }, e.brick)).collect();
        let by_key = super::gpu::build_by_key(entries);

        let mut delta = RepackDelta::default();
        let mut topology_changed = false;
        let mut dirty: FxHashMap<BrickKey, ()> = FxHashMap::default();

        // (2a) DROP keys no longer resident: free their slot/arena (→ quarantine), collapse their slot, and
        // dirty their neighbours (whose halo now reads AIR where this brick was).
        let live_keys: Vec<BrickKey> = self.resident.keys().copied().collect();
        for key in live_keys {
            if new_by_key.contains_key(&key) {
                continue;
            }
            let st = self.resident.remove(&key).expect("key from live set");
            delta.changed.push(ChangedSlot {
                slot: st.slot,
                meta: GpuBrickMeta::zeroed(),
                aabb: degenerate_aabb(),
                index: None,
                index_word_offset: 0,
                palette: None,
                palette_word_offset: 0,
            });
            delta.freed.push(st.slot);
            self.quarantine_slots.push(st.slot);
            self.last_meta.insert(st.slot, GpuBrickMeta::zeroed());
            self.last_aabb.insert(st.slot, degenerate_aabb());
            self.last_voxels.remove(&st.slot);
            if let Some(d) = st.dense {
                self.quarantine_index.push((d.index_offset, d.index_bits));
                self.quarantine_palette.push((d.palette_offset, d.palette_k));
            }
            topology_changed = true;
            for nbr in neighbourhood_26(key) {
                if new_by_key.contains_key(&nbr) {
                    dirty.insert(nbr, ());
                }
            }
        }

        // (2b) ENTER keys not yet resident: claim a slot now (so the expansion sees it), seed them dirty.
        for e in entries {
            let key = BrickKey { coord: e.coord, lod: e.lod };
            if self.resident.contains_key(&key) {
                continue;
            }
            let Some(slot) = self.slots.claim() else {
                // At capacity — drop this brick (the manager already bounds the set; defensive skip).
                continue;
            };
            self.resident.insert(key, SlotState { slot, dense: None });
            dirty.insert(key, ());
            topology_changed = true;
        }

        // (2c) Explicitly-rewritten keys (edits / dig re-source) — re-pack even though unchanged-membership.
        for key in std::mem::take(&mut self.pending_rewrites) {
            if new_by_key.contains_key(&key) {
                dirty.insert(key, ());
            }
        }

        // (3) EXPAND by the resident 26-neighbourhood at the same LOD (halo dependency). Snapshot first.
        let seeds: Vec<BrickKey> = dirty.keys().copied().collect();
        for key in seeds {
            for nbr in neighbourhood_26(key) {
                if new_by_key.contains_key(&nbr) {
                    dirty.insert(nbr, ());
                }
            }
        }

        // (4) Re-pack each dirty key against the NEW resident map; emit a ChangedSlot iff its bytes differ from
        // its slot's shadow. Deterministic order so the patch list is reproducible (the perf/A-B tests rely
        // on it).
        let mut dirty_keys: Vec<BrickKey> = dirty.keys().copied().collect();
        dirty_keys.sort_by_key(|k| (k.lod, k.coord.z, k.coord.y, k.coord.x));

        // PHASE 1 (PARALLEL) — the expensive, embarrassingly-parallel part: for each dirty key, build its
        // `PackedBrick` via the SSOT `pack_one` (the halo-fill) AND, for a dense brick, its R2b paletted
        // encoding (`encode_paletted`). Both are PURE functions of `(key, brick, by_key)` — they read only the
        // shared immutable maps and own no packer state — so rayon over `by_key` is sound and produces the SAME
        // bytes as the serial path. The results are gathered in the EXACT `dirty_keys` order (par_iter over an
        // ordered slice + `.collect()` preserves order; the serial fallback maps in the same order), so Phase 2
        // below consumes them in the identical deterministic sequence — byte-identity + free-list/patch-order
        // determinism hold regardless of `parallel`.
        let pack = |key: BrickKey| -> Option<(BrickKey, PackedBrick, Option<PalettedBrick>)> {
            let &brick = new_by_key.get(&key)?;
            let e = ResidentBrick { coord: key.coord, brick, lod: key.lod };
            let pb = pack_one(&e, &by_key);
            let enc = match &pb.voxels {
                BrickVoxels::Dense(cells) => Some(encode_paletted(cells)),
                BrickVoxels::Uniform(_) => None,
            };
            Some((key, pb, enc))
        };
        let packed: Vec<(BrickKey, PackedBrick, Option<PalettedBrick>)> = if parallel {
            dirty_keys.par_iter().filter_map(|&key| pack(key)).collect()
        } else {
            dirty_keys.iter().filter_map(|&key| pack(key)).collect()
        };

        // PHASE 2 (SERIAL, UNCHANGED ORDER) — fold the pre-packed results into the packer's mutable state
        // (arena alloc/free, the resident/shadow maps, `delta.changed` pushes) IN THE SAME SORTED KEY ORDER. All
        // order-dependent mutation (free-list LIFO, deterministic patch order) stays serial here.
        for (key, pb, enc) in &packed {
            self.emit_changed_slot(*key, pb, enc.as_ref(), &mut delta);
        }

        delta.topology_changed = topology_changed;
        delta
    }

    /// Write `pb`'s bytes into `key`'s slot, allocating/freeing/re-classing its index slab as the dense/uniform
    /// classification (and the dense brick's `index_bits` size class) changed, and push a [`ChangedSlot`] iff the
    /// bytes actually differ from the slot's shadow. The dense path uses the PRE-COMPUTED `enc` (the brick's
    /// haloed cells already encoded into a per-brick palette + bit-packed index stream by [`encode_paletted`],
    /// A4.4, in the PARALLEL Phase 1 of [`update`](Self::update)) and writes them into their index + palette
    /// size-class slabs — the SAME (palette, indices) bytes `snapshot_buffers` reproduces from the raw
    /// `last_voxels` shadow (the byte-identity gate). `last_voxels` keeps RAW cells so a fresh re-encode is exact.
    /// `enc` MUST be `Some` for a dense `pb` and `None` for a uniform one (Phase 1 pairs them by `pb.voxels`).
    fn emit_changed_slot(
        &mut self,
        key: BrickKey,
        pb: &PackedBrick,
        enc: Option<&PalettedBrick>,
        delta: &mut RepackDelta,
    ) {
        let st = *self.resident.get(&key).expect("dirty key is resident");
        match &pb.voxels {
            BrickVoxels::Uniform(_) => {
                // Uniform now: free any index + palette slabs it held (→ quarantine, keep-old).
                if let Some(d) = st.dense {
                    self.quarantine_index.push((d.index_offset, d.index_bits));
                    self.quarantine_palette.push((d.palette_offset, d.palette_k));
                }
                let meta = pb.meta_uniform();
                self.resident.insert(key, SlotState { slot: st.slot, dense: None });
                let changed =
                    self.last_meta.get(&st.slot) != Some(&meta) || self.last_aabb.get(&st.slot) != Some(&pb.aabb);
                self.last_meta.insert(st.slot, meta);
                self.last_aabb.insert(st.slot, pb.aabb);
                self.last_voxels.remove(&st.slot);
                if changed {
                    delta.changed.push(ChangedSlot {
                        slot: st.slot,
                        meta,
                        aabb: pb.aabb,
                        index: None,
                        index_word_offset: 0,
                        palette: None,
                        palette_word_offset: 0,
                    });
                }
            }
            BrickVoxels::Dense(cells) => {
                // The haloed cells → per-brick palette + bit-packed index stream were encoded in Phase 1 (the
                // parallel pass). `index_bits` picks the index class; `k = palette.len()` picks the palette class
                // (≤ palette_stride = registry length).
                let enc = enc.expect("dense brick must carry its pre-computed paletted encoding");
                let k = enc.palette.len() as u32;
                debug_assert!(
                    k <= self.palette_stride,
                    "brick palette k={k} exceeds registry length palette_stride={} (impossible — ids come from it)",
                    self.palette_stride,
                );
                // Ensure an index slab block of the RIGHT class: reuse the existing block iff its width class already
                // matches; else free the old one (→ quarantine) and claim a new one of the new class.
                let index_offset = match st.dense {
                    Some(d) if d.index_bits == enc.index_bits => d.index_offset,
                    Some(d) => {
                        self.quarantine_index.push((d.index_offset, d.index_bits));
                        self.index_arena.alloc(index_slab_words(enc.index_bits))
                    }
                    None => self.index_arena.alloc(index_slab_words(enc.index_bits)),
                };
                // Same for the palette slab: reuse iff the same SIZE CLASS (the old block still fits `k`); else
                // free + re-claim. (Re-using within a class avoids churn when `k` jiggles inside a power-of-2 band.)
                let palette_offset = match st.dense {
                    Some(d) if self.palette_arena.class_of(d.palette_k) == self.palette_arena.class_of(k) => {
                        d.palette_offset
                    }
                    Some(d) => {
                        self.quarantine_palette.push((d.palette_offset, d.palette_k));
                        self.palette_arena.alloc(k)
                    }
                    None => self.palette_arena.alloc(k),
                };
                let meta = super::gpu::GpuBrickMeta::dense(
                    pb.voxel_origin,
                    index_offset,
                    pb.world_min,
                    pb.lod,
                    enc.index_bits,
                    palette_offset,
                );
                self.resident.insert(
                    key,
                    SlotState {
                        slot: st.slot,
                        dense: Some(DenseSlot { index_offset, index_bits: enc.index_bits, palette_offset, palette_k: k }),
                    },
                );
                let meta_changed =
                    self.last_meta.get(&st.slot) != Some(&meta) || self.last_aabb.get(&st.slot) != Some(&pb.aabb);
                let voxels_changed = self.last_voxels.get(&st.slot) != Some(cells);
                self.last_meta.insert(st.slot, meta);
                self.last_aabb.insert(st.slot, pb.aabb);
                if voxels_changed {
                    self.last_voxels.insert(st.slot, cells.clone());
                }
                if meta_changed || voxels_changed {
                    // The index + palette blocks are (re-)written exactly when the brick's content changed (a moved
                    // slab — class change / uniform→dense — implies new content, so `voxels_changed` covers it).
                    let (index, palette) = if voxels_changed {
                        (
                            Some(enc.indices.clone()),
                            Some(enc.palette.iter().map(|&id| id as u32).collect::<Vec<u32>>()),
                        )
                    } else {
                        (None, None)
                    };
                    delta.changed.push(ChangedSlot {
                        slot: st.slot,
                        meta,
                        aabb: pb.aabb,
                        index,
                        index_word_offset: index_offset,
                        palette,
                        palette_word_offset: palette_offset,
                    });
                }
            }
        }
    }

    /// **Stage G-a — the GPU-pack producer.** Runs the IDENTICAL dirty-set + allocation as [`update`](Self::update)
    /// (drop/enter/rewrite → 26-neighbourhood expansion → per-dirty re-classify, the slot/arena claim+free with
    /// quarantine, the shadow byte-compare so an unchanged brick costs nothing) — but instead of packing the bytes
    /// on the CPU it emits a [`GpuPackBatch`]: per dirty DENSE brick a [`GpuPackCommand`] + its 27 same-LOD cores
    /// (for the GPU halo-fill), and per uniform/freed slot (plus every dense AABB) a [`GpuCpuWrite`]. The GPU
    /// shader (`assets/shaders/voxel_pack.wgsl`) then writes the index/palette/meta byte-identically to what
    /// [`emit_changed_slot`](Self::emit_changed_slot) would have. The shadow (`last_meta`/`last_aabb`/`last_voxels`)
    /// is kept EXACTLY consistent with the CPU path, so a later [`snapshot_buffers`](Self::snapshot_buffers) (the
    /// grow / fresh-epoch path) is byte-identical regardless of which path ran. The A/B flag (`gpu_pack`) selects
    /// this vs [`update`](Self::update); OFF by default — only the parity gate + an explicit toggle exercise it.
    pub fn update_gpu(&mut self, entries: &[ResidentBrick<'_>], palette_stride: u32) -> GpuPackBatch {
        debug_assert!(
            self.palette_stride == 0 || self.palette_stride == palette_stride,
            "palette_stride must be constant within an epoch ({} → {palette_stride})",
            self.palette_stride,
        );
        self.palette_stride = palette_stride;
        // (1) Deferred-free: release last update's quarantined slots/blocks BEFORE claiming this update's — the
        //     SAME order as `update_inner` so the free-list LIFO (and thus the allocated offsets) match.
        for s in self.quarantine_slots.drain(..) {
            self.slots.release(s);
        }
        for (off, bits) in std::mem::take(&mut self.quarantine_index) {
            self.index_arena.free_block(off, index_slab_words(bits));
        }
        for (off, k) in std::mem::take(&mut self.quarantine_palette) {
            self.palette_arena.free_block(off, k);
        }

        let new_by_key: FxHashMap<BrickKey, &Brick> =
            entries.iter().map(|e| (BrickKey { coord: e.coord, lod: e.lod }, e.brick)).collect();
        let by_key = build_by_key(entries);

        let mut batch = GpuPackBatch::default();
        let mut topology_changed = false;
        let mut dirty: FxHashMap<BrickKey, ()> = FxHashMap::default();

        // (2a) DROP keys no longer resident — collapse to degenerate/zeroed (a CPU write), dirty neighbours.
        let live_keys: Vec<BrickKey> = self.resident.keys().copied().collect();
        for key in live_keys {
            if new_by_key.contains_key(&key) {
                continue;
            }
            let st = self.resident.remove(&key).expect("key from live set");
            // The meta is CPU-written (zeroed); the AABB is GPU-written (degenerate) via an aabb command.
            batch.cpu_writes.push(GpuCpuWrite { slot: st.slot, meta: GpuBrickMeta::zeroed() });
            batch.aabb_commands.push(GpuAabbCommand::freed(st.slot));
            self.quarantine_slots.push(st.slot);
            self.last_meta.insert(st.slot, GpuBrickMeta::zeroed());
            self.last_aabb.insert(st.slot, degenerate_aabb());
            self.last_voxels.remove(&st.slot);
            if let Some(d) = st.dense {
                self.quarantine_index.push((d.index_offset, d.index_bits));
                self.quarantine_palette.push((d.palette_offset, d.palette_k));
            }
            topology_changed = true;
            for nbr in neighbourhood_26(key) {
                if new_by_key.contains_key(&nbr) {
                    dirty.insert(nbr, ());
                }
            }
        }

        // (2b) ENTER keys not yet resident — claim a slot (so the expansion sees it), seed dirty.
        for e in entries {
            let key = BrickKey { coord: e.coord, lod: e.lod };
            if self.resident.contains_key(&key) {
                continue;
            }
            let Some(slot) = self.slots.claim() else {
                continue;
            };
            self.resident.insert(key, SlotState { slot, dense: None });
            dirty.insert(key, ());
            topology_changed = true;
        }

        // (2c) Explicitly-rewritten keys.
        for key in std::mem::take(&mut self.pending_rewrites) {
            if new_by_key.contains_key(&key) {
                dirty.insert(key, ());
            }
        }

        // (3) EXPAND by the resident 26-neighbourhood (halo dependency).
        let seeds: Vec<BrickKey> = dirty.keys().copied().collect();
        for key in seeds {
            for nbr in neighbourhood_26(key) {
                if new_by_key.contains_key(&nbr) {
                    dirty.insert(nbr, ());
                }
            }
        }

        // (4) Re-pack each dirty key against the NEW map, in the SAME deterministic order as `update_inner` so the
        //     free-list/alloc offsets (and thus the emitted commands) are reproducible.
        let mut dirty_keys: Vec<BrickKey> = dirty.keys().copied().collect();
        dirty_keys.sort_by_key(|k| (k.lod, k.coord.z, k.coord.y, k.coord.x));

        // PHASE 1 (PARALLEL) — pure per-key `pack_one` (the SSOT halo-fill + R1 uniform-incl-halo decision). We
        // DON'T `encode_paletted` here (the GPU does the encode); we only need the dense/uniform classification +
        // the palette size `k` to pick the index/palette size class. `k` is computed from the haloed cells'
        // distinct-id count (the SAME count `encode_paletted` produces — first-seen order doesn't affect the SET).
        let pack = |key: BrickKey| -> Option<(BrickKey, PackedBrick, Option<u32>)> {
            let &brick = new_by_key.get(&key)?;
            let e = ResidentBrick { coord: key.coord, brick, lod: key.lod };
            let pb = pack_one(&e, &by_key);
            let k = match &pb.voxels {
                BrickVoxels::Dense(cells) => Some(distinct_count(cells)),
                BrickVoxels::Uniform(_) => None,
            };
            Some((key, pb, k))
        };
        let packed: Vec<(BrickKey, PackedBrick, Option<u32>)> =
            dirty_keys.par_iter().filter_map(|&key| pack(key)).collect();

        // PHASE 2 (SERIAL) — fold into the mutable allocator state IN THE SAME ORDER, emitting GPU commands + CPU
        // writes. Order-dependent mutation (free-list LIFO, command order) stays serial here.
        let _ = &by_key; // Phase 1's `pack_one` read it for halos; Phase 2 resolves cores via `new_by_key`.
        // The DEDUPED core pool index: a resident brick's core lands in `batch.cores` ONCE, the first time any
        // command references it (as centre or neighbour). Shared across all commands so each brick is uploaded
        // once, not once per command that neighbours it.
        let mut core_index: FxHashMap<BrickKey, u32> = FxHashMap::default();
        for (key, pb, _k) in &packed {
            let class = Classification::from_packed(pb);
            let geom = BrickGeom { aabb: pb.aabb, voxel_origin: pb.voxel_origin, world_min: pb.world_min };
            let cells = match &pb.voxels {
                BrickVoxels::Dense(c) => Some(c),
                BrickVoxels::Uniform(_) => None,
            };
            self.emit_pack_command(*key, class, &geom, cells, None, &new_by_key, &mut core_index, &mut batch);
        }

        batch.topology_changed = topology_changed;
        batch
    }

    /// **Stage G4 — the GPU-classify PREPARE phase** (the split of [`update_gpu`](Self::update_gpu) that removes the
    /// CPU `pack_one`). Runs the IDENTICAL dirty-set + drop/enter/rewrite + 26-neighbourhood expansion as
    /// [`update_gpu`](Self::update_gpu) — but does **NOT** run `pack_one`. Instead it builds, for EVERY dirty key (it
    /// does not yet know dense vs uniform — that is what the GPU decides), the deduped core pool + the 27-neighbour
    /// table + one [`GpuClassifyCommand`], and stashes the ordered dirty keys + the freed-slot writes in
    /// [`PendingClassify`]. The render world dispatches `classify_brick`, reads back the per-brick
    /// [`GpuClassifyOut`], and calls [`update_gpu_finish`](Self::update_gpu_finish) to run the `SlabArena` allocation
    /// (the cheap part) and emit the final [`GpuPackBatch`]. The classification is a deterministic function of the
    /// haloed brick, so the alloc — and thus the pool — is byte-identical to the CPU path (the parity gate).
    ///
    /// MUST be paired with exactly one [`update_gpu_finish`](Self::update_gpu_finish) before the next prepare (the
    /// pending state + the mid-flight allocator quarantine assume one in-flight prepare). The deferred-free at the
    /// top mirrors [`update`](Self::update)/[`update_gpu`](Self::update_gpu) so the free-list LIFO (and the offsets)
    /// match. `palette_stride` is the registry length (the `k ≤ palette_stride` invariant).
    pub fn update_gpu_prepare(&mut self, entries: &[ResidentBrick<'_>], palette_stride: u32) -> GpuClassifyBatch {
        debug_assert!(!self.pending_classify.active, "update_gpu_prepare called twice without an update_gpu_finish");
        debug_assert!(
            self.palette_stride == 0 || self.palette_stride == palette_stride,
            "palette_stride must be constant within an epoch ({} → {palette_stride})",
            self.palette_stride,
        );
        self.palette_stride = palette_stride;
        // (1) Deferred-free — release last update's quarantined slots/blocks BEFORE claiming this update's.
        for s in self.quarantine_slots.drain(..) {
            self.slots.release(s);
        }
        for (off, bits) in std::mem::take(&mut self.quarantine_index) {
            self.index_arena.free_block(off, index_slab_words(bits));
        }
        for (off, k) in std::mem::take(&mut self.quarantine_palette) {
            self.palette_arena.free_block(off, k);
        }

        let new_by_key: FxHashMap<BrickKey, &Brick> =
            entries.iter().map(|e| (BrickKey { coord: e.coord, lod: e.lod }, e.brick)).collect();

        let mut pending = PendingClassify::default();
        let mut dirty: FxHashMap<BrickKey, ()> = FxHashMap::default();

        // (2a) DROP keys no longer resident — collapse to degenerate/zeroed (a CPU write), dirty neighbours. No
        //      classification needed (a freed slot has no brick); emitted straight into `pending.freed_*`.
        let live_keys: Vec<BrickKey> = self.resident.keys().copied().collect();
        for key in live_keys {
            if new_by_key.contains_key(&key) {
                continue;
            }
            let st = self.resident.remove(&key).expect("key from live set");
            pending.freed_cpu_writes.push(GpuCpuWrite { slot: st.slot, meta: GpuBrickMeta::zeroed() });
            pending.freed_aabb_commands.push(GpuAabbCommand::freed(st.slot));
            self.quarantine_slots.push(st.slot);
            self.last_meta.insert(st.slot, GpuBrickMeta::zeroed());
            self.last_aabb.insert(st.slot, degenerate_aabb());
            self.last_voxels.remove(&st.slot);
            if let Some(d) = st.dense {
                self.quarantine_index.push((d.index_offset, d.index_bits));
                self.quarantine_palette.push((d.palette_offset, d.palette_k));
            }
            pending.topology_changed = true;
            for nbr in neighbourhood_26(key) {
                if new_by_key.contains_key(&nbr) {
                    dirty.insert(nbr, ());
                }
            }
        }

        // (2b) ENTER keys not yet resident — claim a slot (so the expansion sees it), seed dirty.
        for e in entries {
            let key = BrickKey { coord: e.coord, lod: e.lod };
            if self.resident.contains_key(&key) {
                continue;
            }
            let Some(slot) = self.slots.claim() else {
                continue;
            };
            self.resident.insert(key, SlotState { slot, dense: None });
            dirty.insert(key, ());
            pending.topology_changed = true;
        }

        // (2c) Explicitly-rewritten keys.
        for key in std::mem::take(&mut self.pending_rewrites) {
            if new_by_key.contains_key(&key) {
                dirty.insert(key, ());
            }
        }

        // (3) EXPAND by the resident 26-neighbourhood (halo dependency).
        let seeds: Vec<BrickKey> = dirty.keys().copied().collect();
        for key in seeds {
            for nbr in neighbourhood_26(key) {
                if new_by_key.contains_key(&nbr) {
                    dirty.insert(nbr, ());
                }
            }
        }

        // (4) The deterministic dirty order (same as `update_inner`/`update_gpu`) — `commands[i]` classifies
        //     `dirty_keys[i]`, so the readback order matches.
        let mut dirty_keys: Vec<BrickKey> = dirty.keys().copied().collect();
        dirty_keys.sort_by_key(|k| (k.lod, k.coord.z, k.coord.y, k.coord.x));

        // (5) Build the deduped core pool + the 27-neighbour table + a classify command FOR EVERY dirty key (the
        //     GPU needs the halo for the classification; the table is REUSED by the final pack — see `finish`).
        let mut classify = GpuClassifyBatch::default();
        let mut core_index: FxHashMap<BrickKey, u32> = FxHashMap::default();
        // `build_neighbour_table` appends to a `GpuPackBatch`'s cores/neighbour_indices; classify uses the same
        // shape, so build into a scratch `GpuPackBatch` then move the pools across (no duplication of the SSOT).
        let mut scratch = GpuPackBatch::default();
        for &key in &dirty_keys {
            let base = scratch.neighbour_indices.len() as u32;
            Self::build_neighbour_table(key, base, &new_by_key, &mut core_index, &mut scratch);
            classify.commands.push(GpuClassifyCommand { neighbour_base: base, _pad0: 0, _pad1: 0, _pad2: 0 });
            pending.neighbour_bases.push(base);
        }
        classify.cores = scratch.cores;
        classify.neighbour_indices = scratch.neighbour_indices;

        pending.dirty_keys = dirty_keys;
        pending.active = true;
        self.pending_classify = pending;
        classify
    }

    /// **Stage G4 — the GPU-classify FINISH phase.** Consumes the GPU `classify_brick` readback (`classify_out`, one
    /// [`GpuClassifyOut`] per [`GpuClassifyBatch::commands`] entry, in `dirty_keys` order) + the
    /// [`GpuClassifyBatch`] from the matching [`update_gpu_prepare`](Self::update_gpu_prepare) (whose core pool +
    /// neighbour table the final pack REUSES). Runs the EXACT Phase-2 serial allocation [`update_gpu`](Self::update_gpu)
    /// would — but the dense/uniform decision + the palette size class come from the GPU classification, NOT a CPU
    /// `pack_one` (the G4 win: no `pack_one` on the dirty bricks). Emits the byte-identical [`GpuPackBatch`] the
    /// render world dispatches `pack_brick` + `write_aabb` over.
    pub fn update_gpu_finish(&mut self, prepared: &GpuClassifyBatch, classify_out: &[GpuClassifyOut]) -> GpuPackBatch {
        debug_assert!(self.pending_classify.active, "update_gpu_finish called without a matching update_gpu_prepare");
        debug_assert_eq!(
            classify_out.len(),
            self.pending_classify.dirty_keys.len(),
            "classify readback length must equal the prepared dirty-key count",
        );
        let pending = std::mem::take(&mut self.pending_classify);

        let mut batch = GpuPackBatch {
            // The final pack REUSES the prepared core pool + neighbour table (the classify pass + the pack pass index
            // the IDENTICAL table; a dense pack command points at the same `neighbour_base`).
            cores: prepared.cores.clone(),
            neighbour_indices: prepared.neighbour_indices.clone(),
            cpu_writes: pending.freed_cpu_writes,
            aabb_commands: pending.freed_aabb_commands,
            topology_changed: pending.topology_changed,
            ..Default::default()
        };

        // `new_by_key`/`core_index` are unused on the GPU path (the table is prebuilt) — pass empty placeholders.
        let new_by_key: FxHashMap<BrickKey, &Brick> = FxHashMap::default();
        let mut core_index: FxHashMap<BrickKey, u32> = FxHashMap::default();
        for (i, &key) in pending.dirty_keys.iter().enumerate() {
            let class = Classification::from_gpu(&classify_out[i]);
            let geom = BrickGeom::of(key);
            let base = pending.neighbour_bases[i];
            self.emit_pack_command(key, class, &geom, None, Some(base), &new_by_key, &mut core_index, &mut batch);
        }
        batch
    }

    /// Intern `key`'s brick into the deduped core pool, returning its core-pool index (in `BRICK_VOXELS` units).
    /// Uploads the `8³` core ONCE on first sight. `NEIGHBOUR_ABSENT` is returned by the CALLER for an absent key
    /// (this is only called for resident keys).
    fn intern_core(
        cores: &mut Vec<u32>,
        core_index: &mut FxHashMap<BrickKey, u32>,
        key: BrickKey,
        brick: &Brick,
    ) -> u32 {
        if let Some(&idx) = core_index.get(&key) {
            return idx;
        }
        let idx = (cores.len() / BRICK_VOXELS) as u32;
        cores.extend_from_slice(&extract_core(brick));
        core_index.insert(key, idx);
        idx
    }

    /// Allocate `key`'s slot's index/palette slabs (as the dense/uniform classification + the index/palette size
    /// class changed) and emit either a [`GpuPackCommand`] (dense) or a [`GpuCpuWrite`] (uniform) into `batch`,
    /// updating the shadow EXACTLY as [`emit_changed_slot`](Self::emit_changed_slot) does — so BOTH the
    /// CPU-classified [`update_gpu`](Self::update_gpu) AND the GPU-classified [`update_gpu_finish`](Self::update_gpu_finish)
    /// paths converge to byte-identical shadow/buffer state.
    ///
    /// Driven by the [`Classification`] (dense/uniform + the palette size class) — produced by the CPU `pack_one`
    /// OR the GPU classify readback — plus the cheap [`BrickGeom`] (a pure function of `key`). `cells` is `Some` ONLY
    /// on the CPU path (the haloed cells, for the `last_voxels` shadow + the `voxels_changed` byte-compare); on the
    /// GPU path it is `None` (the CPU never built them) and the dense brick is conservatively treated as changed
    /// (`force_voxels_changed`) — a redundant re-pack is still byte-identical, never wrong. `prebuilt_base` is the
    /// GPU path's already-built 27-neighbour table base (built in `prepare`); `None` ⇒ build it lazily here (CPU).
    #[allow(clippy::too_many_arguments)]
    fn emit_pack_command(
        &mut self,
        key: BrickKey,
        class: Classification,
        geom: &BrickGeom,
        cells: Option<&Vec<u32>>,
        prebuilt_base: Option<u32>,
        new_by_key: &FxHashMap<BrickKey, &Brick>,
        core_index: &mut FxHashMap<BrickKey, u32>,
        batch: &mut GpuPackBatch,
    ) {
        let lod = key.lod;
        let st = *self.resident.get(&key).expect("dirty key is resident");
        match class {
            Classification::Uniform(block) => {
                if let Some(d) = st.dense {
                    self.quarantine_index.push((d.index_offset, d.index_bits));
                    self.quarantine_palette.push((d.palette_offset, d.palette_k));
                }
                let meta = GpuBrickMeta::uniform(geom.voxel_origin, block, geom.world_min, lod);
                self.resident.insert(key, SlotState { slot: st.slot, dense: None });
                let changed =
                    self.last_meta.get(&st.slot) != Some(&meta) || self.last_aabb.get(&st.slot) != Some(&geom.aabb);
                self.last_meta.insert(st.slot, meta);
                self.last_aabb.insert(st.slot, geom.aabb);
                self.last_voxels.remove(&st.slot);
                if changed {
                    // Meta is CPU-written (uniform id rides in it); the AABB is GPU-written via an aabb command.
                    batch.cpu_writes.push(GpuCpuWrite { slot: st.slot, meta });
                    batch.aabb_commands.push(GpuAabbCommand::resident(st.slot, geom.world_min, lod));
                }
            }
            Classification::Dense { palette_k: k } => {
                let index_bits = pow2_index_bits(k as usize);
                debug_assert!(
                    k <= self.palette_stride,
                    "brick palette k={k} exceeds registry length palette_stride={}",
                    self.palette_stride,
                );
                let index_offset = match st.dense {
                    Some(d) if d.index_bits == index_bits => d.index_offset,
                    Some(d) => {
                        self.quarantine_index.push((d.index_offset, d.index_bits));
                        self.index_arena.alloc(index_slab_words(index_bits))
                    }
                    None => self.index_arena.alloc(index_slab_words(index_bits)),
                };
                let palette_offset = match st.dense {
                    Some(d) if self.palette_arena.class_of(d.palette_k) == self.palette_arena.class_of(k) => {
                        d.palette_offset
                    }
                    Some(d) => {
                        self.quarantine_palette.push((d.palette_offset, d.palette_k));
                        self.palette_arena.alloc(k)
                    }
                    None => self.palette_arena.alloc(k),
                };
                let meta = GpuBrickMeta::dense(
                    geom.voxel_origin,
                    index_offset,
                    geom.world_min,
                    lod,
                    index_bits,
                    palette_offset,
                );
                self.resident.insert(
                    key,
                    SlotState {
                        slot: st.slot,
                        dense: Some(DenseSlot { index_offset, index_bits, palette_offset, palette_k: k }),
                    },
                );
                let meta_changed =
                    self.last_meta.get(&st.slot) != Some(&meta) || self.last_aabb.get(&st.slot) != Some(&geom.aabb);
                // CPU path: byte-compare the cells against the shadow. GPU path (`cells == None`): the CPU never
                // built the cells, so conservatively re-pack (the GPU produces byte-identical bytes regardless).
                let voxels_changed = match cells {
                    Some(c) => self.last_voxels.get(&st.slot) != Some(c),
                    None => true,
                };
                self.last_meta.insert(st.slot, meta);
                self.last_aabb.insert(st.slot, geom.aabb);
                match cells {
                    // CPU path: keep the raw-cell shadow current (the `snapshot_buffers` re-encode source).
                    Some(c) if voxels_changed => {
                        self.last_voxels.insert(st.slot, c.clone());
                    }
                    // GPU path: the CPU has no cells. Drop any stale shadow so a later `snapshot_buffers` does NOT
                    // re-encode a wrong (old-CPU-path) block for this slot — the GPU pack owns the bytes now (the
                    // GPU-path snapshot fills dense blocks GPU-side, see `snapshot_buffers_gpu`).
                    None => {
                        self.last_voxels.remove(&st.slot);
                    }
                    _ => {}
                }
                if meta_changed || voxels_changed {
                    // Stage G-b: the meta + index + palette are GPU-written (the `pack_brick` command); the AABB
                    // too (a GPU `write_aabb` command). We emit the AABB command whenever the meta changed (the
                    // AABB is a pure function of `world_min`/`lod`, which live in the meta); the pack command ONLY
                    // when the voxel CONTENT changed (a meta-only move with identical content needs no re-encode).
                    batch.aabb_commands.push(GpuAabbCommand::resident(st.slot, geom.world_min, lod));
                    if voxels_changed {
                        // The 27-entry NEIGHBOUR TABLE (the brick + its 26 same-LOD neighbours) for the GPU
                        // halo-fill. On the GPU path it is ALREADY built (in `prepare`) — reuse `prebuilt_base`.
                        // On the CPU path build it lazily, interning each resident core into the DEDUPED pool.
                        let neighbour_base = match prebuilt_base {
                            Some(base) => base,
                            None => {
                                let base = batch.neighbour_indices.len() as u32;
                                Self::build_neighbour_table(key, base, new_by_key, core_index, batch);
                                base
                            }
                        };
                        batch.commands.push(GpuPackCommand {
                            origin_x: geom.voxel_origin[0],
                            origin_y: geom.voxel_origin[1],
                            origin_z: geom.voxel_origin[2],
                            slot: st.slot,
                            world_min_x: geom.world_min[0],
                            world_min_y: geom.world_min[1],
                            world_min_z: geom.world_min[2],
                            index_word_offset: index_offset,
                            lod,
                            index_bits: index_bits as u32,
                            palette_word_offset: palette_offset,
                            neighbour_base,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        });
                    }
                }
            }
        }
    }

    /// Build the 27-entry NEIGHBOUR TABLE for `key` at `base` (= `batch.neighbour_indices.len()` on entry), interning
    /// each resident core into the DEDUPED pool (uploaded once); an absent neighbour → [`NEIGHBOUR_ABSENT`] (the halo
    /// reads AIR — mirror of `neighbour_border_cell`). The SSOT both the lazy CPU emit + the Stage-G4 `prepare` use,
    /// so the classify pass and the pack pass index the IDENTICAL table.
    fn build_neighbour_table(
        key: BrickKey,
        base: u32,
        new_by_key: &FxHashMap<BrickKey, &Brick>,
        core_index: &mut FxHashMap<BrickKey, u32>,
        batch: &mut GpuPackBatch,
    ) {
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nslot = neighbour_slot(dx, dy, dz);
                    debug_assert_eq!(batch.neighbour_indices.len(), base as usize + nslot as usize);
                    let nkey = BrickKey { coord: key.coord + IVec3::new(dx, dy, dz), lod: key.lod };
                    let entry = match new_by_key.get(&nkey) {
                        Some(&brick) => Self::intern_core(&mut batch.cores, core_index, nkey, brick),
                        None => NEIGHBOUR_ABSENT,
                    };
                    batch.neighbour_indices.push(entry);
                }
            }
        }
    }

    /// Assemble a CONTIGUOUS [`GpuBrickPatch`] (resident bricks ONLY, in slot order, with re-based voxel
    /// offsets + palette + NEE lights) from the packer's shadow state — the shape `pack_resident_set` produces,
    /// so the existing render/upload/shader path consumes it unchanged. This is the live re-pack output: it is
    /// assembled by MEMCPY of the cached per-brick bytes (NOT by re-`pack_one`'ing every brick — the
    /// [`update`](Self::update) already re-packed only the O(changed) bricks), so it is far cheaper than a
    /// from-scratch [`pack_resident_set`](super::gpu::pack_resident_set). Byte-identical to a from-scratch pack
    /// for the same resident set (the A/B test proves it), so the render is pixel-identical.
    ///
    /// Slot order here is the packer's free-list order, NOT the from-scratch `(lod,z,y,x)` sort — but the shader
    /// reads everything from `metas[primitive_index].world_min`/`lod` (never the order), so the render is
    /// identical regardless. (The GPU oracle keys on `(chunk, (lod,z,y,x))` content, not raw slot.)
    pub fn snapshot_patch(&self, registry: &super::palette::BlockRegistry) -> super::gpu::GpuBrickPatch {
        use super::gpu::GpuBrickPatch;
        // Resident slots in ascending slot order (a stable, reproducible order).
        let mut slots: Vec<u32> = self.resident.values().map(|s| s.slot).collect();
        slots.sort_unstable();
        let mut patch = GpuBrickPatch {
            aabbs: Vec::with_capacity(slots.len()),
            metas: Vec::with_capacity(slots.len()),
            voxels: Vec::with_capacity(slots.len() * dense_block_u32()),
            brick_palettes: Vec::new(),
            palette: Vec::new(),
            lights: Vec::new(),
            alias: Vec::new(),
        };
        // R3: dedup identical haloed slices in the live re-pack too, the SAME way `pack_resident_set` does.
        let mut interner = super::gpu::VoxelInterner::new();
        for slot in slots {
            let aabb = *self.last_aabb.get(&slot).expect("resident slot has an aabb shadow");
            let meta = *self.last_meta.get(&slot).expect("resident slot has a meta shadow");
            patch.aabbs.push(aabb);
            if meta.is_uniform() {
                patch.metas.push(meta);
            } else {
                // R2b + R3 — re-encode the cached RAW haloed cells into a per-brick palette + bit-packed index
                // stream in the CONTIGUOUS output buffers (the live shadow keeps raw cells; encoding happens here
                // at snapshot time), deduping identical slices so a repeated brick shares one (index, palette)
                // pair. `last_voxels` is RAW cells, so the encode is byte-identical to `pack_resident_set`'s.
                let cells = self.last_voxels.get(&slot).expect("dense slot has a voxel shadow");
                let layout = interner.intern_paletted(&mut patch.voxels, &mut patch.brick_palettes, cells);
                let rebased = super::gpu::GpuBrickMeta::dense(
                    meta.voxel_origin,
                    layout.voxel_offset,
                    meta.world_min,
                    meta.lod(),
                    layout.index_bits,
                    layout.palette_base,
                );
                patch.metas.push(rebased);
            }
        }
        // Keep the index + palette buffers non-empty for upload (mirrors `pack_resident_set`'s
        // `ensure_voxels_nonempty`).
        if patch.voxels.is_empty() {
            patch.voxels.push(0);
        }
        if patch.brick_palettes.is_empty() {
            patch.brick_palettes.push(0);
        }
        // Palette + NEE light list — derived from the assembled buffers via the SHARED gpu tail (one SSOT).
        super::gpu::finalize_patch_palette_and_lights(&mut patch, registry);
        patch
    }
}

#[cfg(test)]
mod tests;
