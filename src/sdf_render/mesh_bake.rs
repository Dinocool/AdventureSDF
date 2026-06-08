//! SDF→mesh bake (see `docs/MESH_BAKE_PLAN.md`): a residency-driven, **async**, content-hash-driven
//! Surface Nets bake. The bake/render UNIT is a configurable **chunk** of `K×K×K` finest bricks
//! (`MeshBakeConfig::chunk_bricks`, runtime-tunable). `K = 1` is one mesh per finest brick; larger `K`
//! aggregates more bricks into one contiguous mesh — fewer draw calls/entities, coherent atomic swaps,
//! and contiguous geometry for later decimation/LOD.
//!
//! **Generational coherent rounds (the update model).** To make a whole multi-chunk edit appear
//! UNIFORMLY (not chunk-by-chunk) while staying as real-time as possible, the bake advances in rounds:
//!  1. SNAPSHOT — when idle and something is stale, freeze the current edit list as the round's target
//!     and record each resident chunk's target content hash.
//!  2. BAKE — async-mesh every stale chunk against that FROZEN snapshot (one pending bake per chunk; a
//!     completed bake is STAGED, not shown). The in-flight target is never superseded mid-round, so no
//!     work is evicted before it's displayed.
//!  3. COMMIT — the instant every chunk of the round is staged (or already current), swap them all in
//!     ONE frame (and reap departed chunks the same frame). The whole edit pops together.
//!  4. Immediately snapshot the next position (same frame) and repeat. During a drag the mesh advances
//!     in coherent snapshots that trail the live position by ~one bake-round; on release the final
//!     position is just the last round. Latency is bounded by bake time → tune via `K` (smaller =
//!     faster rounds = more real-time; larger = fewer draws).
//!
//! Staleness is a CONTENT HASH (`edits::bake_content_hash` of the edits overlapping a chunk — the same
//! key the GPU bake scheduler uses, quantized so `GlobalTransform` jitter doesn't churn it). Residency
//! and staleness derive from the SAME overlap test, so they can't diverge — stale/ghost geometry is
//! structurally impossible (a key-stamped `ChunkMesh` reaper is the closed loop on residency departure).
//!
//! Same-LOD seams are crack-free for free: adjacent chunks share their boundary sample plane (apron).
//!
//! VIEWING: use the **Mesh Bake** editor panel ([`mesh_bake_panel`]) to toggle the SDF render off and
//! reveal these meshes (+ wireframe / chunk-size slider / rebake / counts).

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::Hasher;
use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::math::bounding::Aabb3d;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};
use fast_surface_nets::{surface_nets, SurfaceNetsBuffer};
use ndshape::RuntimeShape;

use crate::sdf_render::atlas::BrickKey;
use crate::sdf_render::{
    edits, gather_sorted_edits, SdfCamera, SdfGridConfig, SdfVolume, VolumeQueryData,
};

/// Max NEW meshing tasks spawned per frame (the pool runs them concurrently; this bounds the spawn
/// burst when a large region enters at once).
const MAX_NEW_TASKS_PER_FRAME: usize = 256;

/// Hash-mix multiplier for folding the "Rebake all" epoch into a chunk's content hash.
const EPOCH_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// The 6 chunk faces: `(bit, axis, is_high_face, the two in-face tangent axes)`. Bit order matches
/// `chunk_face_flags` (−X,+X,−Y,+Y,−Z,+Z). Apron-aware boundary cell: −face = cell `1`, +face = `edge-2`
/// (cell 0 / edge-1 are the apron). Shared by `append_skirts`, boundary-vertex extraction, and the seam pass.
const FACES: [(u8, usize, bool, [usize; 2]); 6] = [
    (0, 0, false, [1, 2]),
    (1, 0, true, [1, 2]),
    (2, 1, false, [0, 2]),
    (3, 1, true, [0, 2]),
    (4, 2, false, [0, 1]),
    (5, 2, true, [0, 1]),
];

/// A meshed surface vertex on a chunk FACE, cached for the cross-LOD GEOMORPH. Holds the reprojected WORLD
/// position and analytic normal (a conforming fine vertex adopts the coarse vertex's pos AND normal, for
/// continuous shading), and `idx`, the vertex's index in the chunk's mesh arrays (so the fine side can be
/// mutated in place). Ordering along the boundary curve (the `BoundaryLoop`) is what the geomorph merge uses;
/// the coarse side only reads `pos`/`normal`.
#[derive(Clone, Copy)]
struct BoundaryVert {
    pos: Vec3,
    normal: Vec3,
    idx: u32,
}

/// One ordered boundary component on a chunk face: its vertices in curve order (following the mesh's actual
/// OPEN boundary edges), and whether it CLOSES (a cycle) vs is an open chain (ends at the face's edges). The
/// geomorph's monotone merge needs this: a loop's first/last verts wrap, a chain's are fixed endpoints.
#[derive(Clone)]
struct BoundaryLoop {
    verts: Vec<BoundaryVert>,
    is_loop: bool,
}

/// Raw mesh data produced off-thread by a meshing task (turned into a `Mesh` asset on the main thread).
struct ChunkMeshData {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
    /// Dominant material id (at the surface centroid) — selects the chunk's `StandardMaterial` PBR params.
    material: u16,
    /// Ordered boundary loops on each of the 6 faces (indexed by face bit), for the cross-LOD seam pass.
    boundary: [Vec<BoundaryLoop>; 6],
}

/// A completed bake for a chunk's round target, held until the coherent COMMIT (`None` = empty chunk).
struct StagedBake {
    data: Option<ChunkMeshData>,
}

/// Resolved appearance of a material id, snapshotted from `MaterialRegistry` for the bake: linear base
/// colour + emissive radiance + PBR scalars. Indexed by `EditSample::material_id`. Base goes on the
/// vertex COLOUR (per-vertex); metallic/roughness/emissive go on the chunk's `StandardMaterial`.
#[derive(Clone, Copy)]
struct MatAppearance {
    base: [f32; 3],
    emissive: [f32; 3],
    metallic: f32,
    roughness: f32,
}

/// Fallback when a material id isn't in the registry snapshot (neutral dielectric grey, no emission).
const DEFAULT_APPEARANCE: MatAppearance =
    MatAppearance { base: [0.6, 0.6, 0.6], emissive: [0.0; 3], metallic: 0.0, roughness: 1.0 };

/// The frozen snapshot a bake round is meshing against. `edits = Some` ⇒ a round is in progress; all of
/// that round's bakes use THESE edits/AABBs, so they are mutually coherent regardless of how the live
/// edits move while the round bakes. Cleared on COMMIT.
#[derive(Default)]
struct BakeRound {
    edits: Option<Arc<Vec<edits::ResolvedEdit>>>,
    aabbs: Vec<Aabb3d>,
    /// Frozen camera world position for this round (`None` = no camera, single-LOD fallback). Frozen with
    /// the edits so the round's per-face skirt flags are self-consistent even if the camera moves mid-round.
    cam: Option<Vec3>,
    /// Frozen LOD-0 cube half-extent in LOD-0 chunks (even, so shells tile cleanly).
    half0: i32,
}

/// Per-system scalar `Local` state, bundled (Bevy systems cap at 16 params).
#[derive(Default)]
struct MeshBakeScalars {
    /// "Rebake all" / appearance / debug epoch, mixed into every content hash.
    epoch: u64,
    /// Last frame's chunk size K — detects a live K change.
    prev_k: u32,
    /// Last frame's material-appearance hash — detects a colour/PBR edit.
    prev_mat_hash: u64,
    /// Held clipmap centre while "Freeze LOD" is on (captured on the rising edge; cleared on release).
    frozen_cam: Option<Vec3>,
}

/// Marks a baked chunk mesh entity AND stamps it with its chunk key (a `BrickKey` whose coord is the
/// chunk's min-brick coord), so departed/orphaned meshes can be reaped by a query (residency = the
/// single source of truth) regardless of `ChunkStates` bookkeeping. This is what makes ghost meshes
/// impossible: the entity carries its own identity.
#[derive(Component)]
struct ChunkMesh(BrickKey);

/// Per-chunk bake state.
#[derive(Default)]
struct ChunkState {
    /// Currently displayed mesh (None = meshed-empty, or not meshed yet).
    entity: Option<Entity>,
    /// Content hash of the inputs the DISPLAYED mesh was baked from.
    displayed_hash: u64,
    /// Content hash this chunk is baking toward THIS round — frozen at the round's SNAPSHOT, so the
    /// in-flight bake is never superseded by a newer position before it's displayed. Equals
    /// `displayed_hash` when the chunk is idle / up to date.
    target_hash: u64,
    /// The single in-flight meshing task (baking `target_hash`), if any.
    task: Option<Task<Option<ChunkMeshData>>>,
    /// Completed bake of `target_hash`, awaiting the round COMMIT.
    staged: Option<StagedBake>,
    /// Per-face boundary loops of the DISPLAYED mesh, copied at COMMIT — read by the geomorph so a finer
    /// neighbour can snap (conform) its boundary onto THIS chunk's when it borders this coarser LOD.
    boundary: [Vec<BoundaryLoop>; 6],
    /// Dominant material id of the DISPLAYED mesh — selects this chunk's lit `StandardMaterial`.
    material: u16,
}

/// Per-resident-chunk bake state.
#[derive(Resource, Default)]
struct ChunkStates(HashMap<BrickKey, ChunkState>);

/// Runtime-tunable mesh-bake config. `chunk_bricks` (K) sets the bake/render unit to `K×K×K` finest
/// bricks; the editor panel exposes it as a slider (1..=8). NOTE: this is the mesh-bake aggregation
/// unit, NOT `chunk::CHUNK_BRICKS` (the GPU-atlas residency chunk — a different concept).
#[derive(Resource)]
struct MeshBakeConfig {
    chunk_bricks: u32,
    /// World half-extent of the LOD-0 (finest) cube around the camera. Geometry within this radius meshes
    /// at LOD 0; each coarser LOD doubles the radius (2:1 clipmap). Larger = more fine geometry (slower).
    lod0_radius: f32,
    /// How many LOD levels the mesh bake uses (clamped to `SdfGridConfig::lod_count`). 1 = single-LOD.
    lod_count: u32,
    /// Skirt length in LOD-L voxels (the curtain that hides cross-LOD cracks). 0 = no skirts.
    skirt_cells: f32,
    /// Debug: tint each chunk mesh by its LOD level (+ skirts a contrasting colour), rendered unlit.
    debug_lod_colour: bool,
    /// Cross-LOD SEAM strips (stitch fine↔coarse boundaries crack-free). When on, skirts are suppressed
    /// (the strip replaces them); when off, falls back to skirts. The structurally-correct crack fix.
    seams_enabled: bool,
    /// Debug: FREEZE the clipmap centre at the camera's current position so the LOD structure stops
    /// following the camera — fly through and inspect a fixed LOD boundary / its seams up close.
    freeze_lod: bool,
}

impl Default for MeshBakeConfig {
    fn default() -> Self {
        // K=4 → 64 bricks/chunk. lod0_radius 10 keeps the finest LOD out to a comfortable distance (push
        // it down to shrink the LOD-0 cube); lod_count 9 spans LOD 0..=8 (the lod_test showcase scene);
        // seams on (transition strips) — the real crack fix; skirts are the fallback when off.
        Self {
            chunk_bricks: 4,
            lod0_radius: 10.0,
            lod_count: 9,
            skirt_cells: 3.0,
            debug_lod_colour: false,
            seams_enabled: true,
            freeze_lod: false,
        }
    }
}

/// Distinct unlit debug tint per LOD level for the "Colour by LOD" view (LODs 0..=8).
const LOD_DEBUG_PALETTE: [[f32; 3]; 9] = [
    [0.85, 0.20, 0.20], // LOD0 red
    [0.95, 0.55, 0.15], // LOD1 orange
    [0.90, 0.85, 0.20], // LOD2 yellow
    [0.30, 0.80, 0.25], // LOD3 green
    [0.20, 0.80, 0.80], // LOD4 cyan
    [0.25, 0.45, 0.95], // LOD5 blue
    [0.55, 0.30, 0.90], // LOD6 violet
    [0.90, 0.35, 0.85], // LOD7 magenta
    [0.75, 0.75, 0.80], // LOD8 grey
];
/// Skirt debug tint (bright white) so the crack-filling curtains stand out in the "Colour by LOD" view.
const SKIRT_DEBUG_COLOUR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

/// Set by the editor panel's "Rebake all" button to force a full re-mesh.
#[derive(Resource, Default)]
struct MeshBakeRebuild(bool);

/// Live diagnostics for the editor panel.
#[derive(Resource, Default)]
struct MeshBakeStats {
    /// Number of SDF volumes (edits) gathered this frame.
    edits: usize,
    /// Resident chunks the edits currently occupy.
    resident: usize,
    /// Chunk-mesh entities despawned by the most recent COMMIT.
    reaped: usize,
    /// Resident chunk count per LOD level (index = lod), for the panel readout.
    resident_by_lod: [usize; 8],
    /// Set by the panel's "Capture diagnostics" button; consumed by the system, which fills `dump`.
    capture: bool,
    /// Copy-paste-able diagnostic dump — filled when `capture` is requested.
    dump: String,
}

/// Mesh-bake plugin. Added in `main.rs`. The bake itself is editor- AND scene-INDEPENDENT (it runs
/// every frame and bakes SDF world edits in gameplay too); only the optional debug panel is editor-only.
pub struct MeshBakePlugin;

impl Plugin for MeshBakePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChunkStates>()
            .init_resource::<MeshBakeConfig>()
            .init_resource::<MeshBakeRebuild>()
            .init_resource::<MeshBakeStats>()
            // Editor- AND scene-INDEPENDENT: runs every frame so SDF world edits are baked during
            // gameplay too. It self-determines which chunks to mesh from the SDF edits (no dependency
            // on the editor-scene-gated GPU SDF atlas) and no-ops when no SDF volumes exist — which
            // also clears the meshes when an SDF scene is left.
            .add_systems(Update, mesh_resident_chunks);
        // Editor-only: a dedicated bottom dock panel for the mesh-bake controls (a debug overlay; the
        // bake above does not depend on it).
        #[cfg(feature = "editor")]
        crate::editor::panels::register_panel(
            app,
            "sdf/mesh_bake",
            "Mesh Bake",
            crate::editor::panels::DockSide::Bottom,
            15,
            mesh_bake_panel,
        );
    }
}

/// World-space AABB overlap (inclusive).
fn aabb_overlap(a: &Aabb3d, b: &Aabb3d) -> bool {
    a.min.x <= b.max.x
        && a.max.x >= b.min.x
        && a.min.y <= b.max.y
        && a.max.y >= b.min.y
        && a.min.z <= b.max.z
        && a.max.z >= b.min.z
}

/// World-space AABB of a chunk (`K×K×K` bricks at `key.lod`).
fn chunk_aabb(key: BrickKey, config: &SdfGridConfig, k: u32) -> Aabb3d {
    let min = config.brick_min_world(key.coord, key.lod);
    let cw = k as f32 * config.brick_world_size(key.lod);
    Aabb3d::from_min_max(min, min + Vec3::splat(cw))
}

/// LOD-0-chunk index RANGE `[lo, hi)` (per axis) occupied by a LOD-`key.lod` chunk. A LOD-L chunk spans
/// `2^L` LOD-0 chunks per axis (its world size is `2^L×` a LOD-0 chunk), so all per-LOD shells can be
/// compared on the common LOD-0 chunk lattice. `key.coord` is a multiple of `K·cell_stride` in LOD-L
/// voxel units, so the LOD-L chunk index is `coord / (K·cell_stride)`.
fn chunk_lod0_range(key: BrickKey, config: &SdfGridConfig, k: u32) -> (IVec3, IVec3) {
    let stride = k as i32 * config.cell_stride();
    let j = key.coord / stride; // LOD-L chunk index (integer; coord is a stride multiple)
    let span = 1i32 << key.lod; // LOD-0 chunks per LOD-L chunk, per axis
    (j * span, j * span + IVec3::splat(span))
}

/// Is the LOD-0-chunk range `[lo,hi)` fully inside the cube of half-extent `half` (LOD-0 chunks) centred
/// on `cam0` (Chebyshev / axis-aligned cube)?
fn range_in_cube(lo: IVec3, hi: IVec3, cam0: IVec3, half: i32) -> bool {
    (0..3).all(|a| lo[a] >= cam0[a] - half && hi[a] <= cam0[a] + half)
}

/// LOD-`lod` cube CENTRE in LOD-0 chunk units. Snapped (round-to-nearest, so the camera stays centred) to
/// a `2^(lod+1)`-LOD-0-chunk lattice — the next-coarser chunk grid — so the cube's boundary aligns to
/// LOD-(lod+1) chunks and the LOD step tiles cleanly. Per-LOD round-to-nearest = frequent recentring (the
/// camera never leaves the fine cube) with hysteresis (only jumps on a lattice crossing).
fn lod_centre(config: &SdfGridConfig, k: u32, cam: Vec3, lod: u32) -> IVec3 {
    let cw0 = k as f32 * config.brick_world_size(0);
    let align = 1i32 << (lod + 1); // 2 LOD-`lod` chunks, in LOD-0 chunk units
    let snap = |c: f32| ((c / cw0 / align as f32).round() as i32) * align;
    IVec3::new(snap(cam.x), snap(cam.y), snap(cam.z))
}

/// LOD-0 cube half-extent in LOD-0 chunks — rounded to an EVEN number so the finer cube (half this) stays
/// chunk-aligned at the next LOD too (clean partition; see `mesh_chunk_in_shell`).
fn lod0_half_chunks(config: &SdfGridConfig, mesh_cfg: &MeshBakeConfig, k: u32) -> i32 {
    let cw0 = k as f32 * config.brick_world_size(0);
    let h = (mesh_cfg.lod0_radius / cw0).round().max(2.0) as i32;
    (h + 1) & !1 // next even, ≥ 2
}

/// Effective LOD count: `mesh_cfg.lod_count` clamped to the debug palette (the mesh path's LODs are
/// independent of the GPU atlas `lod_count` — `voxel_size_at(lod)` is just `·2^lod`), or 1 with no camera.
fn effective_lod_count(_config: &SdfGridConfig, mesh_cfg: &MeshBakeConfig, has_cam: bool) -> u32 {
    if !has_cam {
        1
    } else {
        mesh_cfg.lod_count.clamp(1, LOD_DEBUG_PALETTE.len() as u32)
    }
}

/// Minimum size (in voxels of the chunk's own LOD) an edit's LARGEST dimension must span to be meshable
/// there. Below this an object is only a cell or two across, so Surface Nets degenerates into a glitchy
/// sliver ("goes inverse"). Such an edit is dropped from that LOD's residency AND its fold, so it cleanly
/// VANISHES (it's sub-pixel at that distance anyway) instead of flickering.
const SUBVOXEL_MIN_VOXELS: f32 = 2.5;

/// Is an edit resolvable at `lod`? `max_extent` is the edit's largest world dimension. Keyed on the
/// LARGEST (not smallest) extent so a thin SHEET — big in two dims, meshed fine as a ~1-voxel slab — is
/// never culled; only objects small in ALL dimensions (true sub-voxel blobs) are. SSOT for the sub-voxel
/// cull: used identically by residency enumeration and the per-chunk fold so the two never disagree.
fn edit_resolvable_at(max_extent: f32, config: &SdfGridConfig, lod: u32) -> bool {
    max_extent >= SUBVOXEL_MIN_VOXELS * config.voxel_size_at(lod)
}

/// Is a LOD-`key.lod` chunk resident in its 2:1 clipmap shell? Resident ⟺ inside `cube(L)` (centred on the
/// camera, snapped per-LOD) AND (L==0 OR NOT fully inside the finer `cube(L-1)` hole). Each cube boundary
/// is aligned to the coarser side's chunk grid (even `half0` + the per-LOD `lod_centre` snap), so adjacent
/// LODs tile without overlap. No camera ⇒ LOD-0 everywhere (scene/camera-independent fallback).
fn mesh_chunk_in_shell(
    key: BrickKey,
    config: &SdfGridConfig,
    k: u32,
    cam: Option<Vec3>,
    half0: i32,
) -> bool {
    let Some(cam) = cam else {
        return key.lod == 0;
    };
    let (lo, hi) = chunk_lod0_range(key, config, k);
    let outer = half0 * (1i32 << key.lod); // cube(L) half in LOD-0 chunks
    if !range_in_cube(lo, hi, lod_centre(config, k, cam, key.lod), outer) {
        return false;
    }
    if key.lod == 0 {
        return true;
    }
    let hole = half0 * (1i32 << (key.lod - 1)); // cube(L-1) — covered by the finer LOD
    !range_in_cube(lo, hi, lod_centre(config, k, cam, key.lod - 1), hole)
}

/// Per-face "borders a COARSER LOD" flags (bit 0..5 = −X,+X,−Y,+Y,−Z,+Z) for a resident chunk — the faces
/// that need a skirt. A face borders coarser ⟺ the adjacent LOD-L chunk across it is NOT inside `cube(L)`
/// (so that region is served by LOD L+1). Folded into the content hash so a chunk re-bakes (with the right
/// skirts) exactly when the camera moves a shell line.
fn chunk_face_flags(
    key: BrickKey,
    config: &SdfGridConfig,
    k: u32,
    cam: Option<Vec3>,
    half0: i32,
) -> u8 {
    let Some(cam) = cam else {
        return 0;
    };
    let centre = lod_centre(config, k, cam, key.lod);
    let step = k as i32 * config.cell_stride(); // LOD-L voxel stride to the adjacent chunk
    let outer = half0 * (1i32 << key.lod);
    let dirs = [IVec3::NEG_X, IVec3::X, IVec3::NEG_Y, IVec3::Y, IVec3::NEG_Z, IVec3::Z];
    let mut flags = 0u8;
    for (bit, d) in dirs.iter().enumerate() {
        let nbr = BrickKey::new(key.lod, key.coord + *d * step);
        let (lo, hi) = chunk_lod0_range(nbr, config, k);
        if !range_in_cube(lo, hi, centre, outer) {
            flags |= 1 << bit;
        }
    }
    flags
}

/// Bitmask of a chunk's LOW faces (bits 0,2,4 = −X,−Y,−Z) that border a FINER LOD (the neighbour region is
/// inside the finer `cube(lod-1)`). The crate meshes the low apron (cell 0), so on these faces the mesh
/// over-reaches ~1 voxel INTO the finer region — an intrusion into the seam band. The bake INSETS them (skips
/// cell 0 → boundary at the nominal plane) to give a hard boundary + clean gap for the seam. HIGH faces end
/// at the real boundary (no over-reach), so they are never trimmed.
fn chunk_finer_low_faces(key: BrickKey, config: &SdfGridConfig, k: u32, cam: Option<Vec3>, half0: i32) -> u8 {
    let Some(cam) = cam else {
        return 0;
    };
    if key.lod == 0 {
        return 0; // nothing finer than LOD 0
    }
    let centre = lod_centre(config, k, cam, key.lod - 1); // finer cube centre
    let hole = half0 * (1i32 << (key.lod - 1)); // finer cube half-extent (LOD-0 chunks)
    let step = k as i32 * config.cell_stride();
    let mut mask = 0u8;
    for &(bit, d) in &[(0u8, IVec3::NEG_X), (2u8, IVec3::NEG_Y), (4u8, IVec3::NEG_Z)] {
        let nbr = BrickKey::new(key.lod, key.coord + d * step);
        let (lo, hi) = chunk_lod0_range(nbr, config, k);
        if range_in_cube(lo, hi, centre, hole) {
            mask |= 1 << bit;
        }
    }
    mask
}

/// The edits (into `aabbs`) overlapping `sampled` — the set folded for this chunk. Same test drives
/// residency AND the content hash, so they can't diverge.
fn cull_into(aabbs: &[Aabb3d], sampled: &Aabb3d, out: &mut Vec<u32>) {
    out.clear();
    for (i, a) in aabbs.iter().enumerate() {
        if aabb_overlap(a, sampled) {
            out.push(i as u32);
        }
    }
}

/// Enumerate the chunks overlapping `aabb` (padded by one chunk so surface chunks at the boundary are
/// caught) into `out`. Chunk coords are multiples of `K·cell_stride` in voxel units, so a chunk edge
/// spans `K·brick_world_size` in world space and sits at `idx · K·brick_world_size`. The key is a
/// `BrickKey` whose coord is the chunk's min-brick voxel coord.
fn chunks_in_aabb(aabb: &Aabb3d, config: &SdfGridConfig, k: u32, lod: u32, out: &mut HashSet<BrickKey>) {
    let cw = k as f32 * config.brick_world_size(lod); // chunk world size at LOD
    let stride = k as i32 * config.cell_stride(); // chunk voxel stride (LOD-L voxel units)
    let min = Vec3::from(aabb.min) - Vec3::splat(cw);
    let max = Vec3::from(aabb.max) + Vec3::splat(cw);
    let lo = (min / cw).floor();
    let hi = (max / cw).floor();
    // Guard against a pathologically large edit AABB exploding the enumeration (mostly defused now that
    // the shell clip bounds each LOD's window). Kept as a backstop.
    let count = (hi.x - lo.x + 1.0) as i64 * (hi.y - lo.y + 1.0) as i64 * (hi.z - lo.z + 1.0) as i64;
    if count > 200_000 {
        return;
    }
    for ix in lo.x as i32..=hi.x as i32 {
        for iy in lo.y as i32..=hi.y as i32 {
            for iz in lo.z as i32..=hi.z as i32 {
                out.insert(BrickKey::new(lod, IVec3::new(ix, iy, iz) * stride));
            }
        }
    }
}

/// Sample the (pseudo-)SDF for one chunk: an `edge³` grid (linear `x + y·edge + z·edge²`) at world points
/// `grid_origin + (x,y,z)·vs`. Point-sampling the EXACT analytic CSG field is the correct coarse
/// representation for an analytic SDF (there's no fine grid to mip-reduce, so no averaging distortion). The
/// 1-voxel apron makes neighbouring chunks share identical boundary samples, so Surface Nets welds them
/// crack-free by construction. Coarse-LOD shrinkage is fixed by RE-PROJECTING the meshed vertices onto the
/// true surface (`reproject_to_surface`), NOT by sharpening this field — an unsharp/Laplacian filter rings
/// and punches holes (researched 2026-06-08; see [[render-pivot-mesh-baking]]).
fn sample_field(
    edits: &[edits::ResolvedEdit],
    indices: &[u32],
    grid_origin: Vec3,
    vs: f32,
    edge: u32,
) -> Vec<f32> {
    let band = 4.0 * vs;
    let mut sdf = vec![0.0f32; (edge * edge * edge) as usize];
    let mut i = 0usize;
    for z in 0..edge {
        for y in 0..edge {
            for x in 0..edge {
                let p = grid_origin + Vec3::new(x as f32, y as f32, z as f32) * vs;
                // Sub-voxel iso-shift so no sample lands exactly on dist == 0 (Surface Nets treats 0 as
                // "outside", dropping a cell — a pinhole at grid-aligned features).
                sdf[i] = (edits::fold_csg_dist_indexed(edits, indices, p) - 1e-3).clamp(-band, band);
                i += 1;
            }
        }
    }
    sdf
}

/// Central-difference gradient of the CSG field at `p` (the outward surface direction). `eps` should be a
/// small fraction of a voxel.
fn field_gradient(edits: &[edits::ResolvedEdit], indices: &[u32], p: Vec3, eps: f32) -> Vec3 {
    let d = |o: Vec3| edits::fold_csg_dist_indexed(edits, indices, p + o);
    Vec3::new(
        d(Vec3::X * eps) - d(Vec3::X * -eps),
        d(Vec3::Y * eps) - d(Vec3::Y * -eps),
        d(Vec3::Z * eps) - d(Vec3::Z * -eps),
    )
}

/// Push a meshed vertex onto the true analytic iso-surface (`fold_csg_dist == 0`) with a few DAMPED Newton
/// steps `p −= t·f(p)·∇̂f(p)`, returning the projected world point and its unit (analytic) normal.
///
/// WHY: Naive Surface Nets places each vertex at the centroid of its edge crossings, which sits INSIDE a
/// convex surface by ~h²·curvature — the coarse-LOD shrinkage. Re-projecting onto the exact field removes
/// that bias at its SOURCE (the geometry), with no field sharpening (which rings → holes), and the gradient
/// is the exact surface normal — sharper than the discrete one. `smin` blends are pseudo-SDF (‖∇f‖ < 1), so
/// the step is damped to avoid overshoot.
///
/// CRACK-FREE WELDING: this is a PURE function of world position + the global field, so two chunks that
/// share a boundary vertex (same Surface-Nets position via the apron) re-project it identically → same-LOD
/// welds are preserved. The cumulative displacement is clamped to ~one voxel so a vertex can never jump to a
/// neighbouring feature (fold-over) near the medial axis. (`reproject_lands_on_surface` locks the contract.)
fn reproject_to_surface(
    edits: &[edits::ResolvedEdit],
    indices: &[u32],
    start: Vec3,
    vs: f32,
) -> (Vec3, Vec3) {
    let eps = vs * 0.01;
    let mut p = start;
    let mut grad = Vec3::Y; // overwritten on the first iteration (used as the returned normal)
    for _ in 0..4 {
        let d = edits::fold_csg_dist_indexed(edits, indices, p);
        grad = field_gradient(edits, indices, p, eps);
        let dir = grad.normalize_or_zero();
        if dir == Vec3::ZERO {
            break;
        }
        p += dir * (-0.8 * d);
        // Never move a vertex more than ~one voxel from where Surface Nets put it.
        let disp = p - start;
        if disp.length() > vs {
            p = start + disp.normalize() * vs;
        }
        if d.abs() < eps {
            break;
        }
    }
    (p, grad.normalize_or_zero())
}

/// Sample + Surface-Nets one chunk (runs off-thread on the task pool). Returns `None` for an empty chunk
/// (no surface crossing). `indices` are the edits (into the CSG-sorted list) that overlap this chunk —
/// exactly the set the chunk's content hash was taken over, so geometry and hash always agree. `edge` is
/// the padded grid edge in samples (`K·cell_stride + 2`).
#[allow(clippy::too_many_arguments)]
fn mesh_chunk(
    edits: &[edits::ResolvedEdit],
    indices: &[u32],
    appearances: &[MatAppearance],
    grid_origin: Vec3,
    vs: f32,
    edge: u32,
    // Bits 0..5 (−X,+X,−Y,+Y,−Z,+Z): faces that border a COARSER LOD → emit a skirt.
    face_flags: u8,
    // Skirt curtain length (world units); 0 = none.
    skirt_len: f32,
    // This chunk's LOD level (for the debug colour-by-LOD view).
    lod: u32,
    // Bits 0,2,4 (−X,−Y,−Z): LOW faces bordering a FINER LOD → INSET (skip the over-reaching apron cell) so
    // the boundary sits at the nominal plane, leaving a clean gap for the seam instead of intruding into it.
    trim_low: u8,
    // Debug: vertex COLOUR = per-LOD tint (+ skirts a contrasting tint) instead of material base colour.
    debug: bool,
) -> Option<ChunkMeshData> {
    let sdf = sample_field(edits, indices, grid_origin, vs, edge);
    let shape = RuntimeShape::<u32, 3>::new([edge, edge, edge]);
    let mut buffer = SurfaceNetsBuffer::default();
    // Inset the over-reaching low apron on finer-bordering faces (mesh from cell 1, not 0, on that axis).
    let mut smin = [0u32; 3];
    for (bit, axis, is_high, _t) in FACES {
        if !is_high && trim_low & (1 << bit) != 0 {
            smin[axis] = 1;
        }
    }
    // TODO(perf): pool the sample buffer + SurfaceNetsBuffer per `edge` to avoid per-task allocation.
    surface_nets(&sdf, &shape, smin, [edge - 1, edge - 1, edge - 1], &mut buffer);
    if buffer.positions.is_empty() {
        return None;
    }
    // Re-project each Surface-Nets vertex onto the exact iso-surface (removes coarse-LOD shrinkage at its
    // source; yields exact analytic normals). SN positions are in cell units → world = grid_origin + cell·vs;
    // meshes store chunk-LOCAL positions (the entity Transform is grid_origin), so subtract it back off.
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(buffer.positions.len());
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(buffer.positions.len());
    for p in &buffer.positions {
        let world = grid_origin + Vec3::new(p[0], p[1], p[2]) * vs;
        let (proj, n) = reproject_to_surface(edits, indices, world, vs);
        let local = proj - grid_origin;
        positions.push([local.x, local.y, local.z]);
        normals.push([n.x, n.y, n.z]);
    }
    // Per-vertex COLOUR: debug = a per-LOD tint; normal = the resolved material's LINEAR base colour (real
    // PBR lighting shades it; the chunk's StandardMaterial carries the dominant material's PBR scalars).
    let lod_tint = LOD_DEBUG_PALETTE[(lod as usize).min(LOD_DEBUG_PALETTE.len() - 1)];
    let mut colors: Vec<[f32; 4]> = buffer
        .positions
        .iter()
        .map(|p| {
            if debug {
                return [lod_tint[0], lod_tint[1], lod_tint[2], 1.0];
            }
            let world = grid_origin + Vec3::new(p[0], p[1], p[2]) * vs;
            let mid = edits::fold_csg(edits, world).material_id as usize;
            let a = appearances.get(mid).copied().unwrap_or(DEFAULT_APPEARANCE);
            [a.base[0], a.base[1], a.base[2], 1.0]
        })
        .collect();
    let mut indices = buffer.indices.clone();
    // SKIRTS: a curtain hanging from each coarser-neighbour boundary edge into the solid, hiding the
    // fine↔coarse crack. Appends to the mesh buffers (see `append_skirts`).
    if skirt_len > 0.0 && face_flags != 0 {
        append_skirts(
            &buffer, face_flags, edge, vs, skirt_len, debug, &mut positions, &mut normals, &mut colors,
            &mut indices,
        );
    }
    // Dominant material (at the surface centroid) → the chunk's StandardMaterial PBR params. Off-the-shelf
    // StandardMaterial is per-mesh, so metallic/roughness/emissive use the dominant material.
    let mut centroid = Vec3::ZERO;
    for p in &buffer.positions {
        centroid += Vec3::new(p[0], p[1], p[2]);
    }
    centroid = grid_origin + (centroid / buffer.positions.len().max(1) as f32) * vs;
    let material = edits::fold_csg(edits, centroid).material_id;
    let boundary = extract_boundary(&buffer, &positions, &normals, grid_origin, edge, trim_low);
    Some(ChunkMeshData { positions, normals, colors, indices, material, boundary })
}

/// Bucket the surface vertices lying on each chunk FACE (apron-aware boundary cell) into per-face lists,
/// `positions` are chunk-LOCAL, so `grid_origin` is added back for cached WORLD positions. Iterates the
/// original Surface-Nets mesh (`buffer.indices`, before skirts): an edge in exactly ONE triangle is a mesh
/// boundary edge; those whose endpoints both sit on a face's boundary cell, linked into ordered loops, are
/// that face's geomorph input (the fine side conforms a loop onto the matching coarse loop in curve order).
fn extract_boundary(
    buffer: &SurfaceNetsBuffer,
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    grid_origin: Vec3,
    edge: u32,
    trim_low: u8,
) -> [Vec<BoundaryLoop>; 6] {
    // Open edges of the surface mesh (appear in exactly one triangle).
    let mut ecount: HashMap<(u32, u32), u32> = HashMap::new();
    for t in buffer.indices.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            *ecount.entry(if a < b { (a, b) } else { (b, a) }).or_insert(0) += 1;
        }
    }
    let open: Vec<(u32, u32)> = ecount.iter().filter(|(_, c)| **c == 1).map(|(e, _)| *e).collect();

    let mut out: [Vec<BoundaryLoop>; 6] = std::array::from_fn(|_| Vec::new());
    for (bit, axis, is_high, _tan) in FACES {
        // The crate meshes cells [smin, edge-1): a HIGH boundary sits at cell edge-2; a LOW boundary at cell 0
        // normally, but at cell 1 when the face was INSET (`trim_low`) to skip the over-reaching apron.
        let bcell = if is_high {
            edge - 2
        } else if trim_low & (1 << bit) != 0 {
            1
        } else {
            0
        };
        let on_face = |i: u32| buffer.surface_points[i as usize][axis] == bcell;
        // Adjacency among on-face boundary verts via the open edges that lie on this face.
        let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
        for &(a, b) in &open {
            if on_face(a) && on_face(b) {
                adj.entry(a).or_default().push(b);
                adj.entry(b).or_default().push(a);
            }
        }
        let nodes: Vec<u32> = adj.keys().copied().collect();
        let mut visited: HashSet<u32> = HashSet::new();
        let bv = |i: u32| BoundaryVert {
            pos: grid_origin + Vec3::from(positions[i as usize]),
            normal: Vec3::from(normals[i as usize]),
            idx: i,
        };
        let to_loop =
            |comp: Vec<u32>, is_loop: bool| BoundaryLoop { verts: comp.into_iter().map(bv).collect(), is_loop };
        // Chains first (start at a degree-1 endpoint); any remaining cycles are closed loops.
        for &s in &nodes {
            if !visited.contains(&s) && adj[&s].len() == 1 {
                out[bit as usize].push(to_loop(walk_open_edges(&adj, &mut visited, s), false));
            }
        }
        for &s in &nodes {
            if !visited.contains(&s) {
                out[bit as usize].push(to_loop(walk_open_edges(&adj, &mut visited, s), true));
            }
        }
    }
    out
}

// ─────────────────────────── Cross-LOD boundary geomorph ───────────────────────────
// A fine chunk's face that borders a coarser LOD leaves a crack: the fine boundary curve (dense) and the
// coarse boundary curve (sparse) are meshed independently and don't coincide. We REMOVE the mismatch — the
// coarse chunk OWNS the shared boundary, the fine chunk CONFORMS to it (snaps its boundary verts onto the
// coarse ones) — so the two meshes share vertices and weld directly, with no seam strip and no T-junctions.
// Conforming is transitive (fine→medium→coarse), so it also closes 3-LOD corners. (See memory.)

/// Outward unit direction of face `bit` (FACES order: −X,+X,−Y,+Y,−Z,+Z).
fn face_dir(bit: u8) -> IVec3 {
    match bit {
        0 => IVec3::NEG_X,
        1 => IVec3::X,
        2 => IVec3::NEG_Y,
        3 => IVec3::Y,
        4 => IVec3::NEG_Z,
        _ => IVec3::Z,
    }
}

/// The opposite face bit (−X↔+X, −Y↔+Y, −Z↔+Z): faces are axis-paired (0/1, 2/3, 4/5).
#[inline]
fn opposite_face(bit: u8) -> u8 {
    bit ^ 1
}

/// Key of the LOD-(L+1) chunk on the far side of a fine chunk's `dir` face. `step = K·cell_stride`. The
/// same-LOD neighbour min is `coord + dir·step` (LOD-L voxel units); halve to LOD-(L+1) units, then snap to
/// the coarse chunk lattice.
fn coarse_neighbour_key(fine: BrickKey, dir: IVec3, step: i32) -> BrickKey {
    let n = fine.coord + dir * step;
    let half = IVec3::new(n.x.div_euclid(2), n.y.div_euclid(2), n.z.div_euclid(2));
    let snap = |c: i32| c.div_euclid(step) * step;
    BrickKey::new(fine.lod + 1, IVec3::new(snap(half.x), snap(half.y), snap(half.z)))
}

/// Walk one connected component of the open-edge adjacency graph from `start`, in curve order.
fn walk_open_edges(adj: &HashMap<u32, Vec<u32>>, visited: &mut HashSet<u32>, start: u32) -> Vec<u32> {
    let mut comp = Vec::new();
    let (mut prev, mut cur) = (u32::MAX, start);
    loop {
        comp.push(cur);
        visited.insert(cur);
        match adj[&cur].iter().copied().find(|&x| x != prev && !visited.contains(&x)) {
            Some(nx) => {
                prev = cur;
                cur = nx;
            }
            None => break,
        }
    }
    comp
}

/// GEOMORPH the fine chunk's boundary onto its coarser neighbours — the "remove the mismatch" cross-LOD fix.
/// The coarse chunk OWNS the shared boundary; this fine chunk CONFORMS: every fine boundary vertex on a
/// coarser-bordering face is snapped onto the nearest coarse boundary vertex (taking its position AND its
/// normal), so the fine boundary collapses to a refinement of the coarse one and the two meshes share
/// vertices → they weld watertight, with no separate seam strip and no T-junctions. Triangles that collapse
/// to a degenerate sliver (two corners landing on one coarse vertex) are dropped. Conforming is TRANSITIVE —
/// a fine→medium→coarse chain closes 3-LOD corners by construction, which the per-face strip never could.
///
/// `neighbours` pairs each coarser-bordering fine face `bit` with THAT neighbour's opposite-face boundary
/// loops (same tangent frame, so fine cells map to coarse cells by `div_euclid(2)`). A vertex shared by two
/// faces (a chunk-edge corner) is snapped ONCE, to the globally nearest candidate across its faces. `origin`
/// is the fine chunk's grid origin (the same one `extract_boundary` used), to map a coarse world target back
/// to chunk-local mesh space.
fn conform_boundary(data: &mut ChunkMeshData, origin: Vec3, neighbours: &[(u8, &[BoundaryLoop])]) {
    if neighbours.is_empty() {
        return;
    }
    // Chosen coarse target (dist², world pos, world normal) per fine vertex INDEX — global min across the faces
    // a vertex lies on (so a chunk-edge corner resolves consistently).
    let mut snap: HashMap<u32, (f32, Vec3, Vec3)> = HashMap::new();
    for &(bit, coarse) in neighbours {
        if coarse.is_empty() {
            continue;
        }
        for fl in &data.boundary[bit as usize] {
            if fl.verts.is_empty() {
                continue;
            }
            // Match this fine boundary component to the coarse component tracing the same curve (nearest
            // centroid), then assign fine→coarse by a monotone, surjective arc-length merge.
            let fc = loop_centroid(&fl.verts);
            let Some(cl) = coarse
                .iter()
                .filter(|c| !c.verts.is_empty())
                .min_by(|a, b| {
                    loop_centroid(&a.verts)
                        .distance_squared(fc)
                        .total_cmp(&loop_centroid(&b.verts).distance_squared(fc))
                })
            else {
                continue;
            };
            assign_monotone(fl, cl, &mut snap);
        }
    }
    if snap.is_empty() {
        return;
    }
    // Move each snapped fine vertex onto its coarse target (world → chunk-local), adopting the coarse normal
    // so shading is continuous across the weld.
    for (&idx, &(_, p, nrm)) in &snap {
        let local = p - origin;
        data.positions[idx as usize] = [local.x, local.y, local.z];
        data.normals[idx as usize] = [nrm.x, nrm.y, nrm.z];
    }
    // Drop triangles that became degenerate (two corners snapped onto the same coarse vertex). Positions were
    // set EXACTLY equal above, so a collapsed boundary edge is caught by exact `==`.
    let old = std::mem::take(&mut data.indices);
    let same = |a: u32, b: u32| data.positions[a as usize] == data.positions[b as usize];
    let mut kept = Vec::with_capacity(old.len());
    for t in old.chunks_exact(3) {
        if !same(t[0], t[1]) && !same(t[1], t[2]) && !same(t[2], t[0]) {
            kept.extend_from_slice(t);
        }
    }
    data.indices = kept;
}

/// Mean position of a boundary component's vertices (to match a fine component to its coarse counterpart).
fn loop_centroid(verts: &[BoundaryVert]) -> Vec3 {
    let s: Vec3 = verts.iter().map(|v| v.pos).sum();
    s / verts.len().max(1) as f32
}

/// Normalised cumulative arc-length (chord) parameter in `[0,1)` for each vertex along a polyline; a LOOP
/// includes the closing edge in the total so the parameterisation is uniform around the cycle.
fn arclen_params(pos: &[Vec3], is_loop: bool) -> Vec<f32> {
    let n = pos.len();
    let mut cum = vec![0.0f32; n];
    for i in 1..n {
        cum[i] = cum[i - 1] + pos[i].distance(pos[i - 1]);
    }
    let total = if is_loop && n > 1 { cum[n - 1] + pos[0].distance(pos[n - 1]) } else { cum[n - 1] };
    let total = if total > 1e-12 { total } else { 1.0 };
    cum.iter().map(|c| c / total).collect()
}

/// Monotone, SURJECTIVE merge of `tf.len()` fine params onto `tc.len()` coarse params (both sorted in `[0,1)`):
/// returns, per fine index, the coarse index it maps to. The mapping never decreases (so fine boundary edges
/// reproduce coarse edges → no skip/cross), and `must_advance` forces the coarse pointer to reach the last
/// vertex by the last fine vertex (so every coarse vertex is hit → no gap), provided fine is at least as dense.
fn merge_assign(tf: &[f32], tc: &[f32]) -> Vec<usize> {
    let (n, m) = (tf.len(), tc.len());
    let mut out = vec![0usize; n];
    let mut j = 0usize;
    for (i, out_i) in out.iter_mut().enumerate() {
        *out_i = j;
        if j + 1 < m {
            let must = (n - 1 - i) <= (m - 1 - j); // reserve enough fine verts to cover the remaining coarse
            let want = tf[i] >= 0.5 * (tc[j] + tc[j + 1]); // past the midpoint toward the next coarse vert
            if must || want {
                j += 1;
            }
        }
    }
    out
}

/// Assign every vertex of fine boundary loop `fl` onto a vertex of the matching coarse loop `cl` by a monotone
/// arc-length merge, writing the nearest-wins target into `snap`. Tries both orientations (and, for a loop,
/// every rotational offset) and keeps the alignment with the least total snap distance — so the fine boundary
/// collapses onto the coarse one in curve order, welding watertight with no edge skips.
fn assign_monotone(fl: &BoundaryLoop, cl: &BoundaryLoop, snap: &mut HashMap<u32, (f32, Vec3, Vec3)>) {
    let (f, c) = (&fl.verts, &cl.verts);
    let (n, m) = (f.len(), c.len());
    if n == 0 || m == 0 {
        return;
    }
    let mut consider = |idx: u32, fp: Vec3, cp: Vec3, cn: Vec3| {
        let d = fp.distance_squared(cp);
        let e = snap.entry(idx).or_insert((f32::INFINITY, Vec3::ZERO, Vec3::ZERO));
        if d < e.0 {
            *e = (d, cp, cn);
        }
    };
    if m == 1 {
        for fv in f {
            consider(fv.idx, fv.pos, c[0].pos, c[0].normal);
        }
        return;
    }
    let fpos: Vec<Vec3> = f.iter().map(|v| v.pos).collect();
    let tf = arclen_params(&fpos, fl.is_loop);
    // Candidate coarse orderings: a loop can start at any vertex and run either way; a chain only forward/reversed.
    let mut cands: Vec<Vec<usize>> = Vec::new();
    if fl.is_loop {
        for r in 0..m {
            cands.push((0..m).map(|k| (r + k) % m).collect());
            cands.push((0..m).map(|k| (r + m - k) % m).collect());
        }
    } else {
        cands.push((0..m).collect());
        cands.push((0..m).rev().collect());
    }
    let mut best: Option<(f32, Vec<usize>, Vec<usize>)> = None; // (cost, coarse ordering, fine→ordering-pos)
    for cseq in cands {
        let cpos: Vec<Vec3> = cseq.iter().map(|&j| c[j].pos).collect();
        let tc = arclen_params(&cpos, cl.is_loop);
        let assign = merge_assign(&tf, &tc);
        let cost: f32 = (0..n).map(|i| fpos[i].distance_squared(cpos[assign[i]])).sum();
        if best.as_ref().is_none_or(|(bc, ..)| cost < *bc) {
            best = Some((cost, cseq, assign));
        }
    }
    let (_, cseq, assign) = best.unwrap();
    for i in 0..n {
        let cv = &c[cseq[assign[i]]];
        consider(f[i].idx, f[i].pos, cv.pos, cv.normal);
    }
}

/// Append skirt curtains for the coarser-neighbour faces (`face_flags`). Variant A′: a boundary vertex's
/// in-face tangent neighbours are found via the crate's `surface_strides`/`stride_to_index` map (the SDF
/// array linearises as `x + y·edge + z·edge²`, so the tangent strides are `[1, edge, edge²]`), and each
/// boundary edge is extruded by `skirt_len` along `−normal` (into the solid — hidden from outside, plugs
/// the seam). Boundary cells are apron-aware: the chunk's real cells are `1..=edge-2` (cell 0 is the low
/// apron), so the −face boundary is cell `1` and the +face boundary is cell `edge-2`. (Skirt length /
/// boundary-cell are visually tunable via the panel + the Colour-by-LOD debug view.)
#[allow(clippy::too_many_arguments)]
fn append_skirts(
    buffer: &SurfaceNetsBuffer,
    face_flags: u8,
    edge: u32,
    _vs: f32,
    skirt_len: f32,
    debug: bool,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let lin = [1u32, edge, edge * edge]; // SDF/array linear strides (match the fill order + RuntimeShape)
    for (bit, axis, is_high, tan) in FACES {
        if face_flags & (1 << bit) == 0 {
            continue;
        }
        let bcell = if is_high { edge - 2 } else { 1 };
        for i in 0..buffer.positions.len() {
            if buffer.surface_points[i][axis] != bcell {
                continue;
            }
            for &t in &tan {
                // The boundary vertex in the next cell along this tangent (one direction only → each edge once).
                let ns = buffer.surface_strides[i] + lin[t];
                let Some(nidx) = buffer.stride_to_index.get(ns as usize).copied() else { continue };
                if nidx == u32::MAX {
                    continue;
                }
                let ni = nidx as usize;
                if buffer.surface_points[ni][axis] != bcell {
                    continue; // neighbour is not on this boundary face → no boundary edge here
                }
                // Extrude both endpoints into the solid (−normalize(normal) · skirt_len) → a curtain quad.
                // Read positions/colours BEFORE pushing (Copy values) to avoid aliasing the Vec.
                let extrude = |p: [f32; 3], n: [f32; 3]| -> [f32; 3] {
                    let n = Vec3::from(n).normalize_or_zero();
                    [p[0] - n.x * skirt_len, p[1] - n.y * skirt_len, p[2] - n.z * skirt_len]
                };
                let (v0, v1) = (i as u32, ni as u32);
                // Use the reprojected ANALYTIC normals (`normals[i]`), not the crate's discrete ones.
                let (n0, n1) = (normals[i], normals[ni]);
                let (e0, c0) = (extrude(positions[i], n0), colors[i]);
                let (e1, c1) = (extrude(positions[ni], n1), colors[ni]);
                let s0 = positions.len() as u32;
                positions.push(e0);
                normals.push(n0);
                colors.push(if debug { SKIRT_DEBUG_COLOUR } else { c0 });
                let s1 = positions.len() as u32;
                positions.push(e1);
                normals.push(n1);
                colors.push(if debug { SKIRT_DEBUG_COLOUR } else { c1 });
                // Curtain quad (boundary edge v0-v1 → extruded edge s0-s1). The chunk material is
                // double-sided, so winding doesn't matter for visibility.
                indices.extend_from_slice(&[v0, v1, s1, v0, s1, s0]);
            }
        }
    }
}

/// Cheap narrow-band test: could the chunk's sampled region contain a surface crossing? Mirrors the GPU
/// scheduler's `narrow_band_keep`. For a LARGE solid most resident chunks are fully INTERIOR (they
/// overlap the edit AABB but the surface is nowhere near) — baking them is a wasted `edge³` sample +
/// Surface Nets that returns empty. Folding ONCE at the chunk centre and comparing `|dist|` to the
/// chunk's circumradius (+ apron + a smoothing margin) drops them for ~one SDF eval instead of a full
/// bake, turning the bake from O(volume) into O(surface-area). CONSERVATIVE: `reach` is an over-estimate
/// and a smoothed chunk force-keeps on a corner sign change, so it can only ever drop a chunk with no
/// crossing — it can never punch a hole.
fn chunk_has_surface(
    edits: &[edits::ResolvedEdit],
    indices: &[u32],
    config: &SdfGridConfig,
    k: u32,
    key: BrickKey,
    vs: f32,
) -> bool {
    if indices.is_empty() {
        return false;
    }
    let cw = k as f32 * config.brick_world_size(key.lod);
    let min = config.brick_min_world(key.coord, key.lod);
    let center = min + Vec3::splat(0.5 * cw);
    let smooth_sum: f32 = indices.iter().map(|&i| edits[i as usize].op.smoothing.max(0.0)).sum();
    // Force-keep on a sign change across the chunk corners — covers a smoothed surface the centre test
    // could miss when smoothing inflates the gradient. The common hard-CSG path (smooth_sum == 0) skips
    // this and pays a single eval below.
    if smooth_sum > 0.0 {
        let mut neg = false;
        let mut pos = false;
        for dx in [0.0, cw] {
            for dy in [0.0, cw] {
                for dz in [0.0, cw] {
                    let d = edits::fold_csg_dist_indexed(edits, indices, min + Vec3::new(dx, dy, dz));
                    if d <= 0.0 {
                        neg = true;
                    } else {
                        pos = true;
                    }
                    if neg && pos {
                        return true;
                    }
                }
            }
        }
    }
    // circumradius (½·√3·side) + apron/iso-shift slack + smoothing inflation margin.
    let reach = cw * 0.866_025_4 + 2.0 * vs + 0.5 * smooth_sum;
    edits::fold_csg_dist_indexed(edits, indices, center).abs() <= reach
}

/// Content-hash-driven, async, generational-coherent Surface Nets bake (see the module doc). The unit is
/// a configurable `K×K×K`-brick chunk; whole edits commit uniformly via frozen bake rounds.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn mesh_resident_chunks(
    mut commands: Commands,
    volumes: Query<VolumeQueryData, With<SdfVolume>>,
    config: Res<SdfGridConfig>,
    mesh_cfg: Res<MeshBakeConfig>,
    mat_reg: Res<edits::MaterialRegistry>,
    // Drives the clipmap LOD (finer near the camera). No `SdfCamera` ⇒ single-LOD fallback (mesh
    // everything at LOD 0 — the original scene/camera-independent behaviour for gameplay scenes).
    cameras: Query<&GlobalTransform, (With<SdfCamera>, Without<SdfVolume>)>,
    chunk_meshes: Query<(Entity, &ChunkMesh)>,
    mut states: ResMut<ChunkStates>,
    mut rebuild: ResMut<MeshBakeRebuild>,
    mut stats: ResMut<MeshBakeStats>,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    // Lit `StandardMaterial` per material id (base WHITE — per-vertex base comes from the vertex COLOUR —
    // plus the material's metallic/roughness/emissive). Cleared + rebuilt when material appearances change.
    mut mat_cache: Local<HashMap<u16, Handle<StandardMaterial>>>,
    // Bundled scalar Locals: rebake epoch, prev K, prev material-appearance hash.
    mut scal: Local<MeshBakeScalars>,
    // The in-progress bake round's frozen edit + clipmap snapshot.
    mut round: Local<BakeRound>,
) {
    let k = mesh_cfg.chunk_bricks.clamp(1, 8);

    // Resolve the CSG edits (SdfOrder-sorted) + each volume's world AABB (the AABB already includes the
    // smoothing margin).
    let gathered = gather_sorted_edits(&volumes);
    if gathered.is_empty() {
        // Scene unloaded — drop everything (tasks cancel on drop).
        if !states.0.is_empty() {
            for (e, _) in &chunk_meshes {
                commands.entity(e).despawn();
            }
            states.0.clear();
        }
        round.edits = None;
        round.aabbs.clear();
        scal.prev_k = k;
        return;
    }

    // K changed live (slider): the key set is at a different stride now, so every old-stride chunk mesh
    // is stale. Despawn all + clear state + abort any round for a clean swap.
    if scal.prev_k != 0 && scal.prev_k != k {
        for (e, _) in &chunk_meshes {
            commands.entity(e).despawn();
        }
        states.0.clear();
        round.edits = None;
        round.aabbs.clear();
    }
    scal.prev_k = k;

    let cs = config.cell_stride() as u32;
    let edge = k * cs + 2; // padded grid edge in samples (1-voxel apron each side)

    let n_edits = gathered.len();
    let mut edit_aabbs: Vec<Aabb3d> = Vec::with_capacity(n_edits);
    let mut edit_vec: Vec<edits::ResolvedEdit> = Vec::with_capacity(n_edits);
    for g in &gathered {
        edit_aabbs.push(g.aabb);
        edit_vec.push(g.edit.clone());
    }
    let edits_arc = Arc::new(edit_vec);
    // Each edit's largest world dimension — the sub-voxel-cull SSOT (`edit_resolvable_at`); indexed like
    // `edit_aabbs`. (Includes the smoothing margin, which only makes the cull MORE conservative.)
    let edit_extent: Vec<f32> =
        edit_aabbs.iter().map(|a| (Vec3::from(a.max) - Vec3::from(a.min)).max_element()).collect();

    // Material appearance snapshot (linear base + emissive) for the off-thread bake, indexed by material
    // id. Cheap (a handful of materials); cloned into each bake task.
    let appearances: Arc<Vec<MatAppearance>> = Arc::new(
        mat_reg
            .defs
            .iter()
            .map(|d| {
                let l = d.base_color.to_linear();
                MatAppearance {
                    base: [l.red, l.green, l.blue],
                    emissive: d.emissive.to_array(),
                    metallic: d.metallic,
                    roughness: d.roughness,
                }
            })
            .collect(),
    );
    // Vertex colours + the chunk material read material APPEARANCE, but the per-chunk content hash keys on
    // material *id* — so a material colour/PBR edit wouldn't otherwise re-bake. Hash the appearance set
    // (quantized; authored values don't jitter) and re-bake + rebuild the StandardMaterials when it changes.
    let mat_hash = {
        let mut h = DefaultHasher::new();
        for a in appearances.iter() {
            for v in a.base.iter().chain(a.emissive.iter()).chain([&a.metallic, &a.roughness]) {
                h.write_i64((*v as f64 * 1.0e4) as i64);
            }
        }
        h.finish()
    };
    let mat_changed = scal.prev_mat_hash != 0 && scal.prev_mat_hash != mat_hash;
    // "Rebake all" (button) OR a material-appearance change bumps a global epoch mixed into every chunk
    // hash → every hash changes once → full re-bake.
    if std::mem::replace(&mut rebuild.0, false) || mat_changed {
        scal.epoch = scal.epoch.wrapping_add(1);
    }
    if mat_changed {
        mat_cache.clear(); // rebuild StandardMaterials with the new params
    }
    scal.prev_mat_hash = mat_hash;
    // Fold the debug-colour flag into the epoch so toggling "Colour by LOD" re-bakes (vertex colours change).
    let epoch_mix = scal.epoch.wrapping_mul(EPOCH_MIX)
        ^ if mesh_cfg.debug_lod_colour { 0xDEB0_C010_0000_0000 } else { 0 };

    // CLIPMAP: camera position + LOD count (camera-driven; no camera ⇒ LOD-0 everywhere).
    let live_cam = cameras.iter().next().map(|t| t.translation());
    // Debug "Freeze LOD": hold the clipmap centre at the position captured when freeze turned on, so the LOD
    // structure stays put while the camera flies through it. Capture on the rising edge; clear on release.
    let cam = if mesh_cfg.freeze_lod {
        if scal.frozen_cam.is_none() {
            scal.frozen_cam = live_cam;
        }
        scal.frozen_cam
    } else {
        scal.frozen_cam = None;
        live_cam
    };
    let half0 = lod0_half_chunks(&config, &mesh_cfg, k);
    let lod_count = effective_lod_count(&config, &mesh_cfg, cam.is_some());

    // The padded sampled AABB of a chunk (cell span + 1-voxel apron at the chunk's own LOD).
    let chunk_sampled = |key: BrickKey| -> Aabb3d {
        let b = chunk_aabb(key, &config, k);
        let apron = Vec3::splat(config.voxel_size_at(key.lod));
        Aabb3d::from_min_max(Vec3::from(b.min) - apron, Vec3::from(b.max) + apron)
    };

    // RESIDENCY: per LOD, the chunks within reach of the CURRENT edits AND inside that LOD's 2:1 clipmap
    // shell (disjoint — each region meshed at exactly one LOD). NO dependency on the GPU SDF atlas.
    let cw0 = k as f32 * config.brick_world_size(0);
    let mut resident: HashSet<BrickKey> = HashSet::new();
    {
        let mut cand: HashSet<BrickKey> = HashSet::new();
        for lod in 0..lod_count {
            cand.clear();
            // Outer cube world bounds of this LOD's shell — clip the enumeration to it so a HUGE far
            // object doesn't enumerate (and trip the 200k guard on) millions of fine-LOD chunks it can
            // never be resident at. The precise hollow-shell test stays `mesh_chunk_in_shell`.
            let shell_cube = cam.map(|c| {
                let centre = lod_centre(&config, k, c, lod).as_vec3() * cw0;
                let half = (half0 << lod) as f32 * cw0;
                (centre - Vec3::splat(half), centre + Vec3::splat(half))
            });
            for (ei, a) in edit_aabbs.iter().enumerate() {
                // Sub-voxel cull: an object too small to mesh at this LOD never becomes resident here, so
                // it vanishes cleanly rather than degenerating (see `edit_resolvable_at`).
                if !edit_resolvable_at(edit_extent[ei], &config, lod) {
                    continue;
                }
                let clipped = match shell_cube {
                    Some((smin, smax)) => {
                        let mn = Vec3::from(a.min).max(smin);
                        let mx = Vec3::from(a.max).min(smax);
                        if mn.cmpgt(mx).any() {
                            continue; // edit doesn't reach this LOD's shell
                        }
                        Aabb3d::from_min_max(mn, mx)
                    }
                    None => *a,
                };
                chunks_in_aabb(&clipped, &config, k, lod, &mut cand);
            }
            for &key in &cand {
                if mesh_chunk_in_shell(key, &config, k, cam, half0) {
                    resident.insert(key);
                }
            }
        }
    }

    // Current content hash for every resident chunk (over the LIVE edits + lod + per-face coarser-neighbour
    // flags) — drives "is the displayed mesh out of date" (a NEW round needed). The lod+flags mix makes a
    // chunk re-bake (with the right skirts) exactly when the camera moves a shell line.
    let mut current_hashes: HashMap<BrickKey, u64> = HashMap::with_capacity(resident.len());
    let mut by_lod = [0usize; 8];
    {
        let mut idx: Vec<u32> = Vec::new();
        // Pass 1 — each chunk's OWN content hash (edits ∩ chunk + lod + face flags + trim).
        let mut own: HashMap<BrickKey, (u64, u8)> = HashMap::with_capacity(resident.len());
        for &key in &resident {
            by_lod[(key.lod as usize).min(7)] += 1;
            cull_into(&edit_aabbs, &chunk_sampled(key), &mut idx);
            // Drop edits that are sub-voxel at this chunk's LOD so a tiny object can't contaminate a chunk
            // resident for a larger one (the residency cull already keeps lone sub-voxel objects out). Same
            // predicate as the bake fold below → hash and geometry always agree.
            idx.retain(|&i| edit_resolvable_at(edit_extent[i as usize], &config, key.lod));
            let base = if idx.is_empty() { 0 } else { edits::bake_content_hash(&edits_arc, &idx) };
            let flags = chunk_face_flags(key, &config, k, cam, half0);
            let trim = chunk_finer_low_faces(key, &config, k, cam, half0);
            let lf = (key.lod as u64).wrapping_mul(0xA24B_AED4_963E_E407)
                ^ (flags as u64).wrapping_mul(EPOCH_MIX)
                ^ (trim as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            own.insert(key, ((base ^ epoch_mix).wrapping_add(lf), flags));
        }
        // Pass 2 — fold each coarser-bordering neighbour's OWN hash into this chunk's hash. The geomorph snaps
        // this chunk's boundary onto that neighbour, so a change in the neighbour's geometry must re-stale
        // (hence re-bake + re-conform) this chunk: robust-by-construction invalidation, no extra scheduling.
        // (Only the neighbour's OWN hash is folded — its boundary toward THIS chunk is the neighbour's own
        // meshed face, unaffected by the neighbour conforming its OTHER, coarser faces.)
        let k_step = k as i32 * config.cell_stride();
        for (&key, &(own_h, flags)) in &own {
            let mut h = own_h;
            for (bit, _axis, _is_high, _tan) in FACES {
                if flags & (1 << bit) == 0 {
                    continue;
                }
                let ckey = coarse_neighbour_key(key, face_dir(bit), k_step);
                if let Some(&(ch, _)) = own.get(&ckey) {
                    h ^= ch.rotate_left(17).wrapping_add(0x9E37_79B9_7F4A_7C15);
                }
            }
            current_hashes.insert(key, h);
        }
    }
    stats.edits = n_edits;
    stats.resident = resident.len();
    stats.resident_by_lod = by_lod;

    // 1. RECEIVE: poll in-flight bakes; on completion STAGE the result (held until the round COMMIT).
    for (_key, st) in states.0.iter_mut() {
        let Some(task) = st.task.as_mut() else {
            continue;
        };
        let Some(result) = block_on(poll_once(&mut *task)) else {
            continue;
        };
        st.task = None;
        st.staged = Some(StagedBake { data: result });
    }

    // Departed chunks won't be part of any commit — free pending work; their displayed entity is HELD
    // (still on screen) until the next COMMIT reaps it, so old geometry clears as the new appears.
    for (key, st) in states.0.iter_mut() {
        if !resident.contains(key) {
            st.staged = None;
            st.task = None;
        }
    }

    // 2. COMMIT the round when every resident chunk is settled — no chunk still baking, and each is
    // either already displaying its target or holding a staged bake of it. Swap them ALL in one frame
    // (and reap departed meshes the same frame) so the whole edit pops together.
    let round_done = resident.iter().all(|key| match states.0.get(key) {
        Some(st) => st.task.is_none() && (st.displayed_hash == st.target_hash || st.staged.is_some()),
        None => true, // not yet tracked → nothing to wait on (it joins the next round)
    });
    let has_staged = states.0.values().any(|s| s.staged.is_some());
    let has_departed = chunk_meshes.iter().any(|(_, cm)| !resident.contains(&cm.0));
    stats.reaped = 0;
    if round_done && (has_staged || has_departed) {
        let mut reaped = 0usize;
        // PHASE A — cache every committing chunk's boundary + material and TAKE its staged mesh into
        // `pending`, retiring its old entity. All boundary caches must be fresh before any fine chunk
        // conforms (it reads its coarse neighbours' caches), so meshes are NOT built until phase B.
        let mut pending: Vec<(BrickKey, ChunkMeshData)> = Vec::new();
        for (key, st) in states.0.iter_mut() {
            let Some(sb) = st.staged.take() else {
                continue;
            };
            if let Some(old) = st.entity.take() {
                commands.entity(old).despawn();
            }
            st.displayed_hash = st.target_hash;
            match sb.data {
                // Empty chunk: no mesh, and clear any stale boundary cache.
                None => st.boundary = std::array::from_fn(|_| Vec::new()),
                Some(data) => {
                    // Cache a COPY of the boundary verts + dominant material; the staged data keeps its own
                    // boundary so the geomorph can read this chunk's fine boundary in phase B.
                    st.boundary = data.boundary.clone();
                    st.material = data.material;
                    pending.push((*key, data));
                }
            }
        }

        // PHASE B-1 — GEOMORPH: each fine chunk bordering a coarser LOD conforms its boundary onto those
        // neighbours' cached boundaries (now all fresh), so the meshes weld with no seam strip. Skipped when
        // seams are disabled — skirts (already baked into the chunk meshes) are the fallback then.
        if mesh_cfg.seams_enabled {
            let k_step = k as i32 * config.cell_stride();
            for (fkey, data) in pending.iter_mut() {
                let flags = chunk_face_flags(*fkey, &config, k, round.cam, round.half0);
                if flags == 0 {
                    continue;
                }
                let mut neighbours: Vec<(u8, &[BoundaryLoop])> = Vec::new();
                for (bit, _axis, _is_high, _tan) in FACES {
                    if flags & (1 << bit) == 0 {
                        continue; // face doesn't border a coarser LOD
                    }
                    let ckey = coarse_neighbour_key(*fkey, face_dir(bit), k_step);
                    if let Some(cst) = states.0.get(&ckey) {
                        neighbours.push((bit, cst.boundary[opposite_face(bit) as usize].as_slice()));
                    }
                }
                let vs_l = config.voxel_size_at(fkey.lod);
                let origin = config.brick_min_world(fkey.coord, fkey.lod) - Vec3::splat(vs_l);
                conform_boundary(data, origin, &neighbours);
            }
        }

        // PHASE B-2 — build + spawn each committed (now conformed) chunk mesh. Debug "Colour by LOD": one
        // shared UNLIT white material (the LOD/skirt tint lives in the vertex COLOUR). Normal: a lit
        // StandardMaterial per dominant material id (cached) — base WHITE so the per-vertex base COLOUR rules.
        for (key, data) in pending {
            let mat = if mesh_cfg.debug_lod_colour {
                mat_cache
                    .entry(u16::MAX)
                    .or_insert_with(|| {
                        materials.add(StandardMaterial {
                            base_color: Color::WHITE,
                            unlit: true,
                            double_sided: true,
                            cull_mode: None,
                            ..default()
                        })
                    })
                    .clone()
            } else {
                mat_cache
                    .entry(data.material)
                    .or_insert_with(|| {
                        let a = appearances
                            .get(data.material as usize)
                            .copied()
                            .unwrap_or(DEFAULT_APPEARANCE);
                        materials.add(StandardMaterial {
                            base_color: Color::WHITE,
                            metallic: a.metallic,
                            perceptual_roughness: a.roughness.max(0.045),
                            emissive: LinearRgba::rgb(a.emissive[0], a.emissive[1], a.emissive[2]),
                            double_sided: true,
                            cull_mode: None,
                            ..default()
                        })
                    })
                    .clone()
            };
            let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, data.positions)
                .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals)
                .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, data.colors)
                .with_inserted_indices(Indices::U32(data.indices));
            // Apron offset: SN sample 0 is one voxel BEFORE the chunk min — MUST stay exactly
            // `brick_min_world(coord,lod) - vs(lod)`, or the chunk shifts a voxel and every weld cracks.
            let vs_l = config.voxel_size_at(key.lod);
            let origin = config.brick_min_world(key.coord, key.lod) - Vec3::splat(vs_l);
            let entity = commands
                .spawn((
                    Mesh3d(mesh_assets.add(mesh)),
                    MeshMaterial3d(mat),
                    Transform::from_translation(origin),
                    ChunkMesh(key),
                    Name::new("SDF Chunk Mesh"),
                ))
                .id();
            if let Some(st) = states.0.get_mut(&key) {
                st.entity = Some(entity);
            }
        }

        // Reap departed meshes (query-based — catches every non-resident `ChunkMesh` regardless of state).
        for (e, cm) in &chunk_meshes {
            if !resident.contains(&cm.0) {
                commands.entity(e).despawn();
                reaped += 1;
            }
        }
        states.0.retain(|key, _| resident.contains(key));
        stats.reaped = reaped;

        // The round is finished — allow a new snapshot (below) to start the next one THIS frame.
        round.edits = None;
        round.aabbs.clear();
    }

    // 3. SNAPSHOT: if no round is in progress and some chunk is stale vs the live edits, freeze a new
    // round — capture the current edit list + AABBs and each resident chunk's current hash as its target.
    // Frozen until the next commit, so a continuously-moving object advances one coherent snapshot at a
    // time (real-time trailing) instead of chasing and evicting every intermediate position.
    if round.edits.is_none() {
        let stale = resident
            .iter()
            .any(|key| states.0.get(key).is_none_or(|st| st.displayed_hash != current_hashes[key]));
        if stale {
            round.edits = Some(edits_arc.clone());
            round.aabbs = edit_aabbs.clone();
            round.cam = cam; // freeze the camera so the round's skirt flags are self-consistent
            round.half0 = half0;
            for &key in &resident {
                states.0.entry(key).or_default().target_hash = current_hashes[&key];
            }
        }
    }

    // Diagnostic dump (panel "Capture diagnostics"). At rest: round=idle, staged=in-flight=stale=held=0.
    if stats.capture {
        stats.capture = false;
        let round_active = round.edits.is_some();
        let staged_n = states.0.values().filter(|s| s.staged.is_some()).count();
        let inflight_n = states.0.values().filter(|s| s.task.is_some()).count();
        let stale_n = resident
            .iter()
            .filter(|k| states.0.get(*k).is_none_or(|s| s.displayed_hash != current_hashes[*k]))
            .count();
        let held_n = chunk_meshes.iter().filter(|(_, cm)| !resident.contains(&cm.0)).count();
        let displayed_n = states.0.values().filter(|s| s.entity.is_some()).count();
        let mut s = String::new();
        s.push_str("=== Mesh Bake Diagnostics ===\n");
        s.push_str(&format!(
            "volumes(edits)={n_edits}  chunk_bricks(K)={k}  resident_chunks={}\n",
            resident.len()
        ));
        s.push_str(&format!(
            "round_active={round_active}  displayed={displayed_n}  staged={staged_n}  in-flight={inflight_n}  stale={stale_n}  held={held_n}\n"
        ));
        s.push_str("(at rest: round_active=false, staged=in-flight=stale=held=0)\n");
        s.push_str("-- volumes (entity : world AABB) --\n");
        for g in &gathered {
            let a = g.aabb;
            s.push_str(&format!(
                "  {:?}  min[{:.2},{:.2},{:.2}] max[{:.2},{:.2},{:.2}]\n",
                g.entity, a.min.x, a.min.y, a.min.z, a.max.x, a.max.y, a.max.z
            ));
        }
        stats.dump = s;
    }

    // 4. REQUEST: bake every stale chunk toward its FROZEN round target, against the round's frozen edit
    // snapshot (so all of a round's bakes are coherent). One pending bake per chunk; never supersede an
    // in-flight or staged bake — it is always displayed (committed) before the next round is snapshotted.
    if let Some(round_edits) = round.edits.clone() {
        let pool = AsyncComputeTaskPool::get();
        let mut budget = MAX_NEW_TASKS_PER_FRAME;
        let mut idx: Vec<u32> = Vec::new();
        let debug = mesh_cfg.debug_lod_colour;
        // Seams and skirts are mutually exclusive: when the seam pass is on it fills the cracks, so suppress
        // skirts (skirt_len 0); otherwise skirts are the fallback.
        let skirt_cells = if mesh_cfg.seams_enabled { 0.0 } else { mesh_cfg.skirt_cells };
        for &key in &resident {
            let st = states.0.entry(key).or_default();
            if st.task.is_some() || st.staged.is_some() {
                continue; // already baking / baked this round
            }
            if st.displayed_hash == st.target_hash {
                continue; // already showing the round target
            }
            if budget == 0 {
                continue; // re-detected next frame; the round target stays frozen
            }
            let vs_l = config.voxel_size_at(key.lod);
            cull_into(&round.aabbs, &chunk_sampled(key), &mut idx);
            // Sub-voxel cull (same predicate as the hash fold): exclude edits too small to mesh at this LOD
            // from the field so they can't bake a degenerate sliver into a chunk resident for a larger edit.
            idx.retain(|&i| {
                let a = round.aabbs[i as usize];
                edit_resolvable_at((Vec3::from(a.max) - Vec3::from(a.min)).max_element(), &config, key.lod)
            });
            // NARROW-BAND CULL: skip chunks with no surface crossing (interior/exterior of a solid) for a
            // single SDF eval instead of a full edge³ bake — the big win for large objects. Commit them
            // empty (no task, no budget) so the round still settles.
            if !chunk_has_surface(&round_edits, &idx, &config, k, key, vs_l) {
                st.staged = Some(StagedBake { data: None });
                continue;
            }
            let grid_origin = config.brick_min_world(key.coord, key.lod) - Vec3::splat(vs_l);
            // Skirt faces + apron inset from the FROZEN shell (so all of a round's chunks agree on the boundary).
            let flags = chunk_face_flags(key, &config, k, round.cam, round.half0);
            let trim_low = chunk_finer_low_faces(key, &config, k, round.cam, round.half0);
            let skirt_len = skirt_cells * vs_l;
            let lod = key.lod;
            let edits = round_edits.clone();
            let indices = idx.clone();
            let apps = appearances.clone();
            st.task = Some(pool.spawn(async move {
                mesh_chunk(
                    &edits, &indices, &apps, grid_origin, vs_l, edge, flags, skirt_len, lod, trim_low, debug,
                )
            }));
            budget -= 1;
        }
    }
}

/// Dedicated "Mesh Bake" bottom dock panel (editor builds): the controls for viewing/inspecting the
/// Surface Nets bake.
#[cfg(feature = "editor")]
fn mesh_bake_panel(world: &mut World, ui: &mut bevy_egui::egui::Ui) {
    use bevy::pbr::wireframe::WireframeConfig;
    use crate::sdf_render::SdfRenderEnabled;

    ui.label("Surface Nets chunk bake (async). Uncheck the SDF render to view the meshes.");
    ui.separator();

    // Toggle the SDF raymarch render off so the baked meshes are visible (its combine pass otherwise
    // paints over them).
    let mut sdf_on = world.resource::<SdfRenderEnabled>().0;
    if ui.checkbox(&mut sdf_on, "SDF raymarch render").changed() {
        world.resource_mut::<SdfRenderEnabled>().0 = sdf_on;
    }

    // Wireframe overlay (black, so it reads over the light normal-coloured fill).
    let mut wire = world.resource::<WireframeConfig>().global;
    if ui.checkbox(&mut wire, "Wireframe").changed() {
        let mut cfg = world.resource_mut::<WireframeConfig>();
        cfg.global = wire;
        cfg.default_color = Color::BLACK;
    }

    // Chunk size (K): the bake/render unit is K×K×K bricks. Smaller K = faster rounds (more real-time);
    // larger K = fewer draw calls but heavier per-chunk re-bakes (grid ≈ (K·7+2)³). Changing it live
    // re-bakes the whole scene at the new granularity.
    let mut k = world.resource::<MeshBakeConfig>().chunk_bricks;
    if ui
        .add(bevy_egui::egui::Slider::new(&mut k, 1..=8).text("Chunk bricks (K)"))
        .on_hover_text("Bake unit = K³ bricks. Smaller K = faster/more real-time rounds; bigger K = fewer draws, heavier re-bakes.")
        .changed()
    {
        world.resource_mut::<MeshBakeConfig>().chunk_bricks = k;
    }

    // Clipmap LOD: geometry within "LOD-0 radius" of the camera meshes at LOD 0; each coarser LOD doubles
    // the radius (2:1 rings). A SMALL radius pushes the tiny test scene into coarser LODs as you fly the
    // camera. "Skirt cells" = the curtain length that hides the cross-LOD cracks. "Colour by LOD" tints
    // each chunk by its LOD (+ skirts white), unlit, so the rings + crack-filling are visible.
    let mut radius = world.resource::<MeshBakeConfig>().lod0_radius;
    if ui
        .add(bevy_egui::egui::Slider::new(&mut radius, 1.0..=64.0).text("LOD-0 radius"))
        .on_hover_text("World radius of the finest (LOD-0) cube around the camera; coarser rings are 2× each.")
        .changed()
    {
        world.resource_mut::<MeshBakeConfig>().lod0_radius = radius;
    }
    let mut lods = world.resource::<MeshBakeConfig>().lod_count;
    if ui.add(bevy_egui::egui::Slider::new(&mut lods, 1..=9).text("LOD levels")).changed() {
        world.resource_mut::<MeshBakeConfig>().lod_count = lods;
    }
    let mut skirt = world.resource::<MeshBakeConfig>().skirt_cells;
    if ui
        .add(bevy_egui::egui::Slider::new(&mut skirt, 0.0..=6.0).text("Skirt cells"))
        .on_hover_text("Cross-LOD crack-filling curtain length, in voxels. Too short leaks; too long shows a lip.")
        .changed()
    {
        world.resource_mut::<MeshBakeConfig>().skirt_cells = skirt;
    }
    let mut seams = world.resource::<MeshBakeConfig>().seams_enabled;
    if ui
        .checkbox(&mut seams, "Cross-LOD seams")
        .on_hover_text(
            "Stitch fine↔coarse boundaries with a transition strip (crack-free). Off → skirts fallback. \
             Toggle bumps a rebake.",
        )
        .changed()
    {
        world.resource_mut::<MeshBakeConfig>().seams_enabled = seams;
        // Re-mesh so chunks pick up / drop their skirts (seams replace them); the seam pass also rebuilds.
        world.resource_mut::<MeshBakeRebuild>().0 = true;
    }
    let mut dbg = world.resource::<MeshBakeConfig>().debug_lod_colour;
    if ui.checkbox(&mut dbg, "Colour by LOD (debug)").changed() {
        world.resource_mut::<MeshBakeConfig>().debug_lod_colour = dbg;
    }
    let mut freeze = world.resource::<MeshBakeConfig>().freeze_lod;
    if ui
        .checkbox(&mut freeze, "Freeze LOD (debug)")
        .on_hover_text("Hold the clipmap centre at the camera's current spot so the LOD stops following — \
                        fly through to inspect a fixed LOD boundary + its seams up close.")
        .changed()
    {
        world.resource_mut::<MeshBakeConfig>().freeze_lod = freeze;
    }

    // Stats. `staged`/`meshing` are transiently non-zero while a round bakes; they drop to 0 once an edit
    // settles (the round has committed).
    let states = world.resource::<ChunkStates>();
    let meshes = states.0.values().filter(|s| s.entity.is_some()).count();
    let in_flight = states.0.values().filter(|s| s.task.is_some()).count();
    let staged = states.0.values().filter(|s| s.staged.is_some()).count();
    ui.label(format!("Chunk meshes: {meshes}  ·  meshing: {in_flight}  ·  staged: {staged}"));

    // System view. `entities` may briefly exceed `resident` during an edit — departed meshes are HELD
    // until the round commit reaps them (so old + new swap together); at rest they match.
    let stats = world.resource::<MeshBakeStats>();
    let (edits, resident, reaped) = (stats.edits, stats.resident, stats.reaped);
    let entities = world.query_filtered::<(), With<ChunkMesh>>().iter(world).count();
    ui.label(format!(
        "edits: {edits}  ·  resident: {resident}  ·  entities: {entities}  ·  reaped/commit: {reaped}"
    ));
    let by_lod = world.resource::<MeshBakeStats>().resident_by_lod;
    let lod_counts: Vec<String> = by_lod
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 0)
        .map(|(l, c)| format!("L{l}:{c}"))
        .collect();
    if !lod_counts.is_empty() {
        ui.label(format!("resident by LOD: {}", lod_counts.join("  ")));
    }

    ui.horizontal(|ui| {
        if ui.button("Rebake all").clicked() {
            world.resource_mut::<MeshBakeRebuild>().0 = true;
        }
        // Fill the copy-paste diagnostic dump on the next bake-system run (this frame / next).
        if ui.button("Capture diagnostics").clicked() {
            world.resource_mut::<MeshBakeStats>().capture = true;
        }
        let dump = world.resource::<MeshBakeStats>().dump.clone();
        if ui.add_enabled(!dump.is_empty(), bevy_egui::egui::Button::new("Copy")).clicked() {
            ui.ctx().copy_text(dump);
        }
    });

    // Selectable diagnostic dump — click Capture, then Copy (or select).
    let dump = world.resource::<MeshBakeStats>().dump.clone();
    if !dump.is_empty() {
        bevy_egui::egui::ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
            let mut text = dump;
            ui.add(
                bevy_egui::egui::TextEdit::multiline(&mut text)
                    .font(bevy_egui::egui::TextStyle::Monospace)
                    .desired_width(f32::INFINITY)
                    .interactive(true),
            );
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfgs() -> (SdfGridConfig, MeshBakeConfig) {
        (SdfGridConfig::default(), MeshBakeConfig::default())
    }

    /// A LOD-`lod` chunk at LOD-`lod` chunk index `(j,0,0)`.
    fn chunk(cfg: &SdfGridConfig, k: u32, lod: u32, j: i32) -> BrickKey {
        let stride = k as i32 * cfg.cell_stride();
        BrickKey::new(lod, IVec3::new(j, 0, 0) * stride)
    }

    #[test]
    fn half0_is_even_and_at_least_two() {
        let (cfg, mc) = cfgs();
        for k in 1..=8 {
            let h = lod0_half_chunks(&cfg, &mc, k);
            assert!(h >= 2 && h % 2 == 0, "half0 must be even ≥2; got {h} for K={k}");
        }
    }

    #[test]
    fn centre_is_stable_under_sub_snap_drift() {
        let (cfg, _) = cfgs();
        let k = 4;
        let c0 = lod_centre(&cfg, k, Vec3::ZERO, 0);
        let c1 = lod_centre(&cfg, k, Vec3::new(0.5, -0.3, 0.2), 0);
        assert_eq!(c0, c1, "LOD centre churned on sub-snap camera drift (hysteresis broken)");
    }

    #[test]
    fn shells_partition_and_nest() {
        let (cfg, mc) = cfgs();
        let k = 4;
        let cam = Some(Vec3::ZERO); // centre at chunk (0,0,0) for every LOD
        let half0 = lod0_half_chunks(&cfg, &mc, k); // 4 with defaults (radius 10, chunk_world0 2.8)
        let r = |key: BrickKey| mesh_chunk_in_shell(key, &cfg, k, cam, half0);

        // LOD 0 fills cube(0) = [-half0, half0] chunks; chunk index `half0` is just outside.
        assert!(r(chunk(&cfg, k, 0, 0)), "centre LOD-0 chunk resident");
        assert!(r(chunk(&cfg, k, 0, half0 - 1)), "inner-rim LOD-0 chunk resident");
        assert!(!r(chunk(&cfg, k, 0, half0)), "LOD-0 chunk past cube(0) not resident at LOD 0");

        // LOD 1's shell covers cube(1)\cube(0); the LOD-1 chunk covering index `half0` is resident,
        // the LOD-1 chunk fully inside cube(0) (the hole) is covered by LOD 0 → NOT resident at LOD 1.
        // LOD-1 chunk index j1 occupies LOD-0 range [2*j1, 2*j1+2); half0=2 so j1=1 covers [2,4).
        assert!(r(chunk(&cfg, k, 1, half0 / 2)), "LOD-1 shell chunk resident");
        assert!(!r(chunk(&cfg, k, 1, 0)), "LOD-1 chunk in the hole is covered by LOD 0 (not resident)");
    }

    #[test]
    fn face_flags_mark_outer_rim_coarser() {
        let (cfg, mc) = cfgs();
        let k = 4;
        let cam = Some(Vec3::ZERO);
        let half0 = lod0_half_chunks(&cfg, &mc, k);
        // Centre chunk: all neighbours inside cube(0) → no coarser faces.
        assert_eq!(chunk_face_flags(chunk(&cfg, k, 0, 0), &cfg, k, cam, half0), 0);
        // Chunk on the +X rim (index half0-1): its +X neighbour (index half0) is outside cube(0) → bit1.
        let f = chunk_face_flags(chunk(&cfg, k, 0, half0 - 1), &cfg, k, cam, half0);
        assert_eq!(f & (1 << 1), 1 << 1, "+X face should border a coarser LOD");
        assert_eq!(f & (1 << 0), 0, "−X face should not (neighbour still inside cube(0))");
    }

    fn sphere_edit(centre: Vec3, radius: f32) -> edits::ResolvedEdit {
        edits::ResolvedEdit::new(
            crate::sdf_render::SdfPrimitive::Sphere { radius },
            Transform::from_translation(centre),
            crate::sdf_render::SdfOp { kind: crate::sdf_render::CsgKind::Union, smoothing: 0.0 },
            0,
        )
    }

    /// World-space triangle triples of a baked chunk (positions are chunk-local → add `origin`).
    fn chunk_tris(data: &ChunkMeshData, origin: Vec3) -> Vec<(Vec3, Vec3, Vec3)> {
        data.indices
            .chunks_exact(3)
            .map(|t| {
                let v = |i: u32| origin + Vec3::from(data.positions[i as usize]);
                (v(t[0]), v(t[1]), v(t[2]))
            })
            .collect()
    }

    /// Count mesh edges NOT shared by exactly 2 triangles, after welding vertices by quantized WORLD
    /// position (0.1 mm). 0 ⇒ closed 2-manifold = watertight. Position-welding lets it span SEPARATE chunk
    /// meshes (fine + coarse), so it is the geomorph's correctness gate: after the fine conforms, the two
    /// must weld with no open edge (gap) and no edge in >2 triangles (overlap).
    fn open_edge_count(tris: &[(Vec3, Vec3, Vec3)]) -> usize {
        let q = |p: Vec3| {
            [
                (p.x as f64 * 1e4).round() as i64,
                (p.y as f64 * 1e4).round() as i64,
                (p.z as f64 * 1e4).round() as i64,
            ]
        };
        let mut edges: HashMap<([i64; 3], [i64; 3]), u32> = HashMap::new();
        for (a, b, c) in tris {
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let (mut ka, mut kb) = (q(*u), q(*v));
                if ka > kb {
                    std::mem::swap(&mut ka, &mut kb);
                }
                *edges.entry((ka, kb)).or_insert(0) += 1;
            }
        }
        edges.values().filter(|&&n| n != 2).count()
    }

    #[test]
    fn single_chunk_closed_surface_is_watertight() {
        // Validates the watertightness checker + that the boundary-cache path doesn't disturb meshing: a
        // sphere fully inside one chunk (touching no face) must mesh as a closed 2-manifold.
        let edits = [sphere_edit(Vec3::ZERO, 1.0)];
        let (vs, edge) = (0.1f32, 30u32); // chunk real span ≈ 2.8 > sphere Ø 2.0 → clears all faces
        let origin = Vec3::splat(-1.5);
        let data = mesh_chunk(&edits, &[0], &[], origin, vs, edge, 0, 0.0, 0, 0, false)
            .expect("sphere meshes");
        assert_eq!(open_edge_count(&chunk_tris(&data, origin)), 0, "closed sphere must be watertight");
    }

    /// Count fine boundary verts on `face` that did NOT land exactly on a coarse boundary vertex after a
    /// conform (within `tol` world units). 0 ⇒ the fine boundary is a subset of the coarse boundary
    /// (no T-junctions). Coarse verts are taken from `coarse.boundary[opp]` (already world-space).
    fn unwelded_boundary_verts(
        fine: &ChunkMeshData,
        origin: Vec3,
        face: usize,
        coarse: &ChunkMeshData,
        opp: usize,
        tol: f32,
    ) -> usize {
        let cverts: Vec<Vec3> = coarse.boundary[opp].iter().flat_map(|l| l.verts.iter().map(|v| v.pos)).collect();
        let mut miss = 0;
        for l in &fine.boundary[face] {
            for v in &l.verts {
                let w = origin + Vec3::from(fine.positions[v.idx as usize]);
                if !cverts.iter().any(|c| c.distance(w) <= tol) {
                    miss += 1;
                }
            }
        }
        miss
    }

    #[test]
    fn geomorph_harsh_undersampling_stays_bounded() {
        // EXTREME 2:1+ undersampling (vs_c = 1.0 vs vs_f = 0.5, sphere R = 1.2 → only ~4 coarse boundary verts
        // for ~20 fine). The monotone arc-length merge can't fully weld when the coarse boundary is THIS sparse
        // (a coarse vertex can have no fine vertex closest to it), but it must stay BOUNDED — strictly better
        // than the pre-merge independent snap (which left 8 open edges here) — not explode. Realistic 2:1 still
        // welds to 0 (see `geomorph_welds_2to1_sphere_watertight`). This guards the merge against regressing.
        let edits = [sphere_edit(Vec3::ZERO, 1.2)];
        let idx = [0u32];
        let (vsf, vsc, edge) = (0.5f32, 1.0f32, 30u32);
        let of = Vec3::new(-vsf, -7.0 - vsf, -7.0 - vsf);
        let oc = Vec3::new(-28.0 - vsc, -14.0 - vsc, -14.0 - vsc);
        let mut fine = mesh_chunk(&edits, &idx, &[], of, vsf, edge, 0, 0.0, 0, 0, false).expect("fine");
        let coarse = mesh_chunk(&edits, &idx, &[], oc, vsc, edge, 0, 0.0, 1, 0, false).expect("coarse");
        let bare = {
            let mut t = chunk_tris(&fine, of);
            t.extend(chunk_tris(&coarse, oc));
            open_edge_count(&t)
        };
        conform_boundary(&mut fine, of, &[(0, &coarse.boundary[1])]);
        let mut all = chunk_tris(&fine, of);
        all.extend(chunk_tris(&coarse, oc));
        let open = open_edge_count(&all);
        assert!(open < bare, "merge must reduce open edges ({open} vs bare {bare})");
        assert!(open <= 6, "even at extreme undersampling the weld must stay bounded, got {open} open edges");
    }

    #[test]
    fn geomorph_welds_2to1_sphere_watertight() {
        // A sphere straddling a forced fine|coarse 2:1 boundary at x = 0: the FINE chunk (vs 0.1) meshes the
        // +X side (its −X face borders coarse), the COARSE chunk (vs 0.2) the −X side, each ending in a
        // boundary circle one fine-voxel apart. The geomorph snaps the FINE boundary onto the coarse one; the
        // two then weld with no seam strip. Bare = cracked; after conform fine+coarse is a closed 2-manifold.
        let edits = [sphere_edit(Vec3::ZERO, 1.0)];
        let idx = [0u32];
        let (vsf, vsc, edge) = (0.1f32, 0.2f32, 30u32);
        let of = Vec3::new(-vsf, -1.4 - vsf, -1.4 - vsf);
        let oc = Vec3::new(-5.6 - vsc, -2.8 - vsc, -2.8 - vsc);
        let mut fine = mesh_chunk(&edits, &idx, &[], of, vsf, edge, 0, 0.0, 0, 0, false).expect("fine meshes");
        let coarse =
            mesh_chunk(&edits, &idx, &[], oc, vsc, edge, 0, 0.0, 1, 0, false).expect("coarse meshes");

        // Bare chunks ARE cracked (two open boundary circles) — proves the geomorph is what closes it.
        let mut bare = chunk_tris(&fine, of);
        bare.extend(chunk_tris(&coarse, oc));
        assert!(open_edge_count(&bare) > 0, "bare 2-LOD sphere must be cracked before conforming");

        // Fine −X (bit 0) conforms onto coarse +X (bit 1).
        conform_boundary(&mut fine, of, &[(0, &coarse.boundary[1])]);
        assert_eq!(
            unwelded_boundary_verts(&fine, of, 0, &coarse, 1, 1e-4),
            0,
            "every conformed fine boundary vert must land on a coarse vert (no T-junction)"
        );
        let mut all = chunk_tris(&fine, of);
        all.extend(chunk_tris(&coarse, oc));
        assert_eq!(open_edge_count(&all), 0, "fine conformed to coarse must be watertight");
    }

    #[test]
    fn geomorph_terrain_welds_no_tjunction() {
        // A heightmap crossing a fine|coarse 2:1 boundary → CHAIN boundaries (the lod_test terrain regime).
        // After the fine conforms, EVERY fine boundary vertex must coincide with a coarse boundary vertex
        // (so the fine boundary is a refinement of the coarse one → no T-junctions / no fine-side gap), and
        // no degenerate (zero-area) triangle may survive the conform.
        let edit = edits::ResolvedEdit::new(
            crate::sdf_render::SdfPrimitive::Heightmap {
                half_xz: Vec2::new(400.0, 400.0),
                max_height: 30.0,
                freq: 0.05,
                amp: 8.0,
                seed: 7,
            },
            Transform::IDENTITY,
            crate::sdf_render::SdfOp { kind: crate::sdf_render::CsgKind::Union, smoothing: 0.0 },
            0,
        );
        let edits = [edit];
        let idx = [0u32];
        let (vsf, vsc, edge) = (1.0f32, 2.0f32, 30u32);
        let of = Vec3::new(-vsf, -vsf, -vsf);
        let oc = Vec3::new(-56.0 - vsc, -vsc, -vsc);
        let mut fine = mesh_chunk(&edits, &idx, &[], of, vsf, edge, 0, 0.0, 0, 0, false).expect("fine meshes");
        let coarse = mesh_chunk(&edits, &idx, &[], oc, vsc, edge, 0, 0.0, 1, 0, false).expect("coarse meshes");

        conform_boundary(&mut fine, of, &[(0, &coarse.boundary[1])]);

        let miss = unwelded_boundary_verts(&fine, of, 0, &coarse, 1, 1e-4);
        assert_eq!(miss, 0, "{miss} fine boundary verts did not weld onto a coarse vert → T-junction/gap");

        // No surviving degenerate triangles (two corners on the same position).
        let degen = fine
            .indices
            .chunks_exact(3)
            .filter(|t| {
                let p = |i: u32| fine.positions[i as usize];
                p(t[0]) == p(t[1]) || p(t[1]) == p(t[2]) || p(t[2]) == p(t[0])
            })
            .count();
        assert_eq!(degen, 0, "{degen} degenerate triangles survived the conform");
    }

    #[test]
    fn geomorph_corner_snaps_consistently() {
        // A fine chunk bordering a coarser LOD on TWO faces (−X and −Z) — the chunk-edge corner. Snapping
        // against the UNION of both neighbours must move each fine boundary vert onto a coarse vert on every
        // face it lies on, so the corner has no leftover T-junction. A sphere centred at the −X/−Z corner.
        let edits = [sphere_edit(Vec3::ZERO, 1.0)];
        let idx = [0u32];
        let (vsf, vsc, edge) = (0.1f32, 0.2f32, 30u32);
        // Fine occupies +X,+Z (its −X and −Z faces border coarse). Two coarse neighbours: one across −X, one
        // across −Z. All anchored so grids are 2:1-nested.
        let of = Vec3::new(-vsf, -1.4 - vsf, -vsf);
        let ox = Vec3::new(-5.6 - vsc, -2.8 - vsc, -2.8 - vsc); // coarse across −X
        let oz = Vec3::new(-2.8 - vsc, -2.8 - vsc, -5.6 - vsc); // coarse across −Z
        let mut fine = mesh_chunk(&edits, &idx, &[], of, vsf, edge, 0, 0.0, 0, 0, false).expect("fine meshes");
        let cx = mesh_chunk(&edits, &idx, &[], ox, vsc, edge, 0, 0.0, 1, 0, false).expect("coarse X meshes");
        let cz = mesh_chunk(&edits, &idx, &[], oz, vsc, edge, 0, 0.0, 1, 0, false).expect("coarse Z meshes");

        // Conform against BOTH neighbours at once (the union — corner verts resolve to one global nearest).
        conform_boundary(&mut fine, of, &[(0, &cx.boundary[1]), (4, &cz.boundary[5])]);

        // Every fine boundary vert on either coarser-bordering face must land on a coarse vert of the UNION
        // (cx ∪ cz). A corner vertex correctly welds to whichever neighbour is nearer — not necessarily both —
        // so the check is against the union, not per-face.
        let union: Vec<Vec3> = cx.boundary[1]
            .iter()
            .chain(cz.boundary[5].iter())
            .flat_map(|l| l.verts.iter().map(|v| v.pos))
            .collect();
        let mut miss = 0;
        for face in [0usize, 4] {
            for l in &fine.boundary[face] {
                for v in &l.verts {
                    let w = of + Vec3::from(fine.positions[v.idx as usize]);
                    if !union.iter().any(|c| c.distance(w) <= 1e-4) {
                        miss += 1;
                    }
                }
            }
        }
        assert_eq!(miss, 0, "{miss} fine corner boundary verts did not weld onto any coarse vert");
    }

    #[test]
    fn merge_assign_is_monotone_and_surjective() {
        // THE watertightness guarantee, isolated: mapping a denser fine boundary onto a sparser coarse one
        // must be MONOTONE (so fine boundary edges never skip/cross coarse edges → the live terrain gaps) and
        // SURJECTIVE (every coarse vertex hit → no coarse-side gap), with the ends pinned. Holds for any curve.
        let tc: Vec<f32> = (0..5).map(|j| j as f32 / 5.0).collect();
        let tf: Vec<f32> = (0..23).map(|i| i as f32 / 23.0).collect();
        let a = merge_assign(&tf, &tc);
        for w in a.windows(2) {
            assert!(w[1] >= w[0], "assignment must be non-decreasing (monotone)");
        }
        let hit: std::collections::HashSet<_> = a.iter().copied().collect();
        assert_eq!(hit.len(), tc.len(), "every coarse vertex must be covered (surjective)");
        assert_eq!(a[0], 0, "first fine maps to first coarse");
        assert_eq!(*a.last().unwrap(), tc.len() - 1, "last fine maps to last coarse");
    }

    #[test]
    fn conform_snaps_to_nearest_and_drops_degenerate() {
        // Pure-fn check: two fine boundary verts that map into one coarse cell both snap onto that coarse
        // vertex (exact pos + normal), and the fine triangle spanning them collapses to a degenerate that is
        // dropped. A synthetic fine chunk: two boundary verts on the −X face (bit 0) sharing a triangle.
        let cpos = Vec3::new(0.0, 0.5, 0.5);
        let cnrm = Vec3::new(-1.0, 0.0, 0.0);
        let coarse = vec![BoundaryLoop {
            verts: vec![BoundaryVert { pos: cpos, normal: cnrm, idx: 0 }],
            is_loop: false,
        }];
        // Fine: 3 verts (idx 0,1 boundary near cpos; idx 2 interior), one triangle (0,1,2).
        let origin = Vec3::ZERO;
        let mut data = ChunkMeshData {
            positions: vec![[0.0, 0.4, 0.5], [0.0, 0.6, 0.5], [0.5, 0.5, 0.5]],
            normals: vec![[-0.9, 0.1, 0.0], [-0.9, -0.1, 0.0], [1.0, 0.0, 0.0]],
            colors: vec![[1.0; 4]; 3],
            indices: vec![0, 1, 2],
            material: 0,
            boundary: std::array::from_fn(|_| Vec::new()),
        };
        // Both fine boundary verts map to the single coarse vert (m==1 branch).
        data.boundary[0] = vec![BoundaryLoop {
            verts: vec![
                BoundaryVert { pos: Vec3::new(0.0, 0.4, 0.5), normal: Vec3::ZERO, idx: 0 },
                BoundaryVert { pos: Vec3::new(0.0, 0.6, 0.5), normal: Vec3::ZERO, idx: 1 },
            ],
            is_loop: false,
        }];
        conform_boundary(&mut data, origin, &[(0, &coarse)]);

        assert_eq!(data.positions[0], [cpos.x, cpos.y, cpos.z], "vert 0 snapped to coarse pos");
        assert_eq!(data.positions[1], [cpos.x, cpos.y, cpos.z], "vert 1 snapped to coarse pos");
        assert_eq!(data.normals[0], [cnrm.x, cnrm.y, cnrm.z], "vert 0 adopted coarse normal");
        assert!(data.indices.is_empty(), "the collapsed triangle (two corners on one coarse vert) is dropped");
    }

    #[test]
    fn reproject_lands_on_surface() {
        // A vertex sitting INSIDE the true surface (mimicking Surface Nets' h²·curvature shrinkage) must
        // re-project back ONTO the iso-surface (|f| ≈ 0) — this is the shrinkage fix. The analytic normal
        // is radially outward on a sphere.
        let edits = [sphere_edit(Vec3::ZERO, 1.5)];
        let idx = [0u32];
        let start = Vec3::new(1.5 - 0.06, 0.0, 0.0); // ~0.06 inside the +X pole
        let (p, n) = reproject_to_surface(&edits, &idx, start, 0.2);
        let d = edits::fold_csg_dist_indexed(&edits, &idx, p);
        assert!(d.abs() < 1e-3, "vertex not on surface after reprojection: f={d}");
        assert!(n.dot(Vec3::X) > 0.99, "normal not radially outward: {n:?}");
    }

    #[test]
    fn reproject_welds_across_index_supersets() {
        // WELDING contract: re-projection is a pure function of world position + the RELEVANT field. A chunk
        // that folds an extra distant edit must land its shared boundary vertex at the same place as a
        // neighbour that doesn't — else cross-chunk seams. A far second sphere must not perturb a projection
        // on the first sphere's surface.
        let edits = [sphere_edit(Vec3::ZERO, 1.5), sphere_edit(Vec3::new(100.0, 0.0, 0.0), 1.5)];
        let start = Vec3::new(1.45, 0.0, 0.0);
        let (p_all, _) = reproject_to_surface(&edits, &[0, 1], start, 0.2);
        let (p_one, _) = reproject_to_surface(&edits, &[0], start, 0.2);
        assert!(
            (p_all - p_one).length() < 1e-5,
            "distant edit perturbed the projection ({p_all:?} vs {p_one:?}) → cross-chunk seam"
        );
    }

    #[test]
    fn subvoxel_cull_drops_small_blobs_keeps_sheets() {
        let cfg = SdfGridConfig::default();
        let vs0 = cfg.voxel_size_at(0); // 0.1 with defaults
        // A small blob ~3 voxels across at LOD 0: resolvable fine, but sub-voxel by a few coarser LODs.
        let blob = 3.0 * vs0;
        assert!(edit_resolvable_at(blob, &cfg, 0), "3-voxel blob meshes at LOD 0");
        assert!(!edit_resolvable_at(blob, &cfg, 2), "same blob is sub-voxel at LOD 2 → culled");
        // A thin SHEET: tiny thickness but a huge footprint → max-extent keyed, so never culled.
        let sheet = 1000.0 * vs0;
        for lod in 0..9 {
            assert!(edit_resolvable_at(sheet, &cfg, lod), "thin sheet kept at every LOD (lod={lod})");
        }
    }

    #[test]
    fn no_camera_is_lod0_everywhere_no_skirts() {
        let (cfg, mc) = cfgs();
        let k = 4;
        let half0 = lod0_half_chunks(&cfg, &mc, k);
        assert!(mesh_chunk_in_shell(chunk(&cfg, k, 0, 9), &cfg, k, None, half0), "LOD 0 everywhere");
        assert!(!mesh_chunk_in_shell(chunk(&cfg, k, 1, 9), &cfg, k, None, half0), "no LOD>0 w/o camera");
        assert_eq!(chunk_face_flags(chunk(&cfg, k, 0, 3), &cfg, k, None, half0), 0, "no skirts w/o camera");
    }
}
