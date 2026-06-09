//! SDF→mesh bake (see `docs/MESH_BAKE_PLAN.md`): a residency-driven, **async**, content-hash-driven
//! Transvoxel bake. The bake/render UNIT is a configurable **chunk** of `K×K×K` finest bricks
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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::math::bounding::Aabb3d;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};
use transvoxel::prelude::*;
use transvoxel::structs::grid_point::GridPoint;
use transvoxel::structs::vertex_index::VertexIndex;
use transvoxel::traits::mesh_builder::MeshBuilder;

use crate::sdf_render::atlas::BrickKey;
use crate::sdf_render::{
    edits, gather_sorted_edits, SdfCamera, SdfGridConfig, SdfVolume, VolumeQueryData,
};

/// Max NEW meshing tasks spawned per frame (the pool runs them concurrently; this bounds the spawn
/// burst when a large region enters at once).
const MAX_NEW_TASKS_PER_FRAME: usize = 256;

/// Hash-mix multiplier for folding the "Rebake all" epoch into a chunk's content hash.
const EPOCH_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// Max world-units a material's blend can reach beyond its surface (= the `blend_softness` slider max). The
/// chunk edit-cull AABBs are padded by this so a chunk a neighbour's blend touches lists that edit in its
/// content hash → moving the edit re-bakes the chunk (no stale blended remnant). Keep ≥ the slider max.
const BLEND_REACH: f32 = 1.0;

/// Raw mesh data produced off-thread by a meshing task (chunk-LOCAL positions; one mesh per chunk). Every
/// chunk uses the SINGLE shared per-vertex-blend `MeshMaterial`: `uvs` carry each vertex's top-2 material ids
/// `(matA, matB)` and `colors.a` the blend weight (`colors.rgb` = the per-LOD debug tint when colour-by-LOD).
struct ChunkMeshData {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    /// `(matA as f32, matB as f32)` per vertex (read in the shader as the two materials to cross-fade).
    uvs: Vec<[f32; 2]>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

/// A completed bake for a chunk's round target, held until the coherent COMMIT (`None` = empty chunk).
struct StagedBake {
    data: Option<ChunkMeshData>,
}

/// The frozen snapshot a bake round is meshing against. `edits = Some` ⇒ a round is in progress; all of
/// that round's bakes use THESE edits/AABBs, so they are mutually coherent regardless of how the live
/// edits move while the round bakes. Cleared on COMMIT.
#[derive(Default)]
struct BakeRound {
    edits: Option<Arc<Vec<edits::ResolvedEdit>>>,
    aabbs: Vec<Aabb3d>,
    /// Frozen camera world position for this round (`None` = no camera, single-LOD fallback). Frozen with
    /// the edits so the round's per-face transition flags are self-consistent even if the camera moves mid-round.
    cam: Option<Vec3>,
    /// Frozen LOD-0 cube half-extent in LOD-0 chunks (even, so shells tile cleanly).
    half0: i32,
    /// Frozen RESIDENT chunk set for this round. The round bakes, commits, and reaps against THIS set — never
    /// the live (current-camera) set — so the displayed geometry only ever swaps as a complete coherent round.
    /// That makes a LOD swap atomic: the old meshes are held until every chunk of the new residency is baked,
    /// then old-out/new-in happen in the same commit (no 1-frame hole). The live set seeds the NEXT snapshot.
    resident: HashSet<BrickKey>,
}

/// Per-system scalar `Local` state, bundled (Bevy systems cap at 16 params).
#[derive(Default)]
struct MeshBakeScalars {
    /// "Rebake all" / debug epoch, mixed into every content hash.
    epoch: u64,
    /// Last frame's chunk size K — detects a live K change.
    prev_k: u32,
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
    /// Currently displayed mesh entities — ONE per material sub-mesh (empty = meshed-empty / not yet meshed).
    entities: Vec<Entity>,
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
}

/// Per-resident-chunk bake state.
#[derive(Resource, Default)]
struct ChunkStates(HashMap<BrickKey, ChunkState>);

/// Runtime-tunable mesh-bake config. `chunk_bricks` (K) sets the bake/render unit to `K×K×K` finest
/// bricks; the editor panel exposes it as a slider (1..=8). NOTE: this is the mesh-bake aggregation
/// unit, NOT `chunk::CHUNK_BRICKS` (the GPU-atlas residency chunk — a different concept).
#[derive(Resource)]
pub(crate) struct MeshBakeConfig {
    chunk_bricks: u32,
    /// World half-extent of the LOD-0 (finest) cube around the camera. Geometry within this radius meshes
    /// at LOD 0; each coarser LOD doubles the radius (2:1 clipmap). Larger = more fine geometry (slower).
    lod0_radius: f32,
    /// How many LOD levels the mesh bake uses (clamped to `SdfGridConfig::lod_count`). 1 = single-LOD.
    lod_count: u32,
    /// Debug: tint each chunk mesh by its LOD level, rendered unlit ("Colour by LOD").
    pub(crate) debug_lod_colour: bool,
    /// Debug: render the mesh world-normal as RGB (`n*0.5+0.5`), unlit ("View normals") — for inspecting
    /// the baked geometry normals directly.
    pub(crate) debug_normals: bool,
    /// Debug: FREEZE the clipmap centre at the camera's current position so the LOD structure stops
    /// following the camera — fly through and inspect a fixed LOD boundary up close.
    freeze_lod: bool,
}

impl Default for MeshBakeConfig {
    fn default() -> Self {
        // K=4 → 64 bricks/chunk. lod0_radius 16 keeps the finest LOD out to a comfortable distance (push
        // it down to shrink the LOD-0 cube); lod_count 9 spans LOD 0..=8 (the lod_test showcase scene).
        // Cross-LOD seams are crack-free BY CONSTRUCTION (Transvoxel transition cells) — no toggle needed.
        Self {
            chunk_bricks: 4,
            lod0_radius: 16.0,
            lod_count: 9,
            debug_lod_colour: false,
            debug_normals: false,
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

/// Set by the editor panel's "Rebake all" button to force a full re-mesh. Also pulsed by
/// `worldgen::roll_worldgen` when the height ring regenerates without the Terrain volume moving (a
/// param edit / streaming delta in fixed mode): the Terrain content hash is unchanged by a ring swap,
/// so the mesh-bake needs an explicit nudge to re-mesh the affected chunks.
#[derive(Resource, Default)]
pub(crate) struct MeshBakeRebuild(pub bool);

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
        app.add_plugins(super::mesh_material::MeshMaterialPlugin)
            .init_resource::<ChunkStates>()
            .init_resource::<MeshBakeConfig>()
            .init_resource::<MeshBakeRebuild>()
            .init_resource::<MeshBakeStats>()
            // Editor- AND scene-INDEPENDENT: runs every frame so SDF world edits are baked during
            // gameplay too. It self-determines which chunks to mesh from the SDF edits (no dependency
            // on the editor-scene-gated GPU SDF atlas) and no-ops when no SDF volumes exist — which
            // also clears the meshes when an SDF scene is left.
            .add_systems(
                Update,
                mesh_resident_chunks.after(super::mesh_material::rebuild_mesh_material),
            );
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

/// COVERAGE GATE predicate: is this chunk allowed to mesh against the worldgen `Terrain`? True iff
/// EITHER the chunk's world-XZ footprint doesn't touch any `Terrain` edit (so it samples no terrain —
/// nothing to gate), OR the loaded height `ring` fully covers that footprint EXPANDED by an apron
/// margin. The margin = the chunk's voxel size + `2·HEIGHT_CHUNK_CELLS` slack so the gate stays ahead
/// of bilinear/gradient taps that reach a node past the chunk edge AND the async lag between a camera
/// roll and the next ring rebuild. A `None` ring (nothing loaded yet) ⇒ never covered ⇒ not resident.
/// This is what makes the strict `eval_primitive` Terrain sampler safe: a resident terrain chunk is
/// always backed by loaded height, so the strict sampler can panic on a miss with no false positives.
fn terrain_chunk_covered(
    key: BrickKey,
    config: &SdfGridConfig,
    k: u32,
    terrain_xz_aabbs: &[(Vec2, Vec2)],
    ring: Option<&crate::sdf_render::worldgen::upload::HeightRingCpu>,
) -> bool {
    let b = chunk_aabb(key, config, k);
    let cmin = Vec2::new(b.min.x, b.min.z);
    let cmax = Vec2::new(b.max.x, b.max.z);
    // Does this chunk sample any Terrain edit? (XZ-overlap test.)
    let touches_terrain = terrain_xz_aabbs.iter().any(|(tmin, tmax)| {
        cmax.x >= tmin.x && cmin.x <= tmax.x && cmax.y >= tmin.y && cmin.y <= tmax.y
    });
    if !touches_terrain {
        return true; // no terrain sampled here → gate doesn't apply.
    }
    let Some(ring) = ring else {
        return false; // touches terrain but nothing loaded → not generatable yet.
    };
    let margin = config.voxel_size_at(key.lod)
        + 2.0 * crate::sdf_render::worldgen::layers::height::HEIGHT_CHUNK_CELLS as f32;
    let m = Vec2::splat(margin);
    crate::sdf_render::worldgen::upload::ring_covers_aabb(ring, cmin - m, cmax + m)
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
/// there. Below this an object is only a cell or two across, so Transvoxel degenerates into a glitchy
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

/// Per-face "borders a FINER LOD" flags (bit 0..5 = −X,+X,−Y,+Y,−Z,+Z) for a resident chunk — the
/// TRANSVOXEL TRANSITION faces. Transvoxel puts the transition cell on the LOW-resolution (this, coarser)
/// block, on the face toward the HIGHER-resolution neighbour, so it matches the finer mesh by construction.
/// A face borders finer ⟺ the adjacent LOD-L chunk across it lies inside the finer `cube(L-1)` (that region
/// is served by LOD L-1). LOD 0 (the finest) has none. Folded into the content hash so a chunk re-bakes with
/// the right transition faces exactly when the camera moves a shell line.
fn chunk_finer_faces(key: BrickKey, config: &SdfGridConfig, k: u32, cam: Option<Vec3>, half0: i32) -> u8 {
    let Some(cam) = cam else {
        return 0;
    };
    if key.lod == 0 {
        return 0; // nothing finer than LOD 0
    }
    let centre = lod_centre(config, k, cam, key.lod - 1); // the finer cube's centre
    let hole = half0 * (1i32 << (key.lod - 1)); // the finer cube's half-extent (LOD-0 chunks)
    let step = k as i32 * config.cell_stride(); // LOD-L voxel stride to the adjacent chunk
    let dirs = [IVec3::NEG_X, IVec3::X, IVec3::NEG_Y, IVec3::Y, IVec3::NEG_Z, IVec3::Z];
    let mut flags = 0u8;
    for (bit, d) in dirs.iter().enumerate() {
        let nbr = BrickKey::new(key.lod, key.coord + *d * step);
        let (lo, hi) = chunk_lod0_range(nbr, config, k);
        if range_in_cube(lo, hi, centre, hole) {
            flags |= 1 << bit;
        }
    }
    flags
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

/// Central-difference gradient of the CSG field at `p` (the outward surface direction). `eps` should be a
/// small fraction of a voxel.
fn field_gradient(edits: &[edits::ResolvedEdit], indices: &[u32], p: Vec3, eps: f32, vs: f32) -> Vec3 {
    let d = |o: Vec3| edits::fold_csg_dist_indexed(edits, indices, p + o, vs);
    Vec3::new(
        d(Vec3::X * eps) - d(Vec3::X * -eps),
        d(Vec3::Y * eps) - d(Vec3::Y * -eps),
        d(Vec3::Z * eps) - d(Vec3::Z * -eps),
    )
}

/// Mesh one chunk with the TRANSVOXEL algorithm (runs off-thread on the task pool). Returns `None` for an
/// empty chunk (no surface). `indices` are the edits overlapping this chunk (the set its content hash was
/// taken over). `subdivisions` is the chunk's cell count per axis (`K·cell_stride`); `grid_origin` is the
/// chunk's world MIN corner — NO apron (Transvoxel samples the field on the block boundary, not beyond).
/// `flags` (faces bordering a FINER LOD) become Transvoxel TRANSITION sides — placed on the coarse side of
/// each 2:1 boundary — so neighbouring LODs weld crack-free BY CONSTRUCTION (no seam pass needed).
#[allow(clippy::too_many_arguments)]
fn mesh_chunk(
    edits: &[edits::ResolvedEdit],
    indices: &[u32],
    grid_origin: Vec3,
    vs: f32,
    subdivisions: u32,
    flags: u8,
    lod: u32,
    debug: bool,
) -> Option<ChunkMeshData> {
    // Transvoxel treats density > threshold as INSIDE; our CSG distance is NEGATIVE inside → negate it. The
    // tiny iso-shift keeps no sample landing EXACTLY on 0 (density > 0 is strict, so a 0 sample reads
    // "outside" — a pinhole at grid-aligned features like a sphere pole on a grid corner).
    let field = |x: f32, y: f32, z: f32| 1e-3 - edits::fold_csg_dist_indexed(edits, indices, Vec3::new(x, y, z), vs);
    let block = Block::new(
        [grid_origin.x, grid_origin.y, grid_origin.z],
        subdivisions as f32 * vs,
        subdivisions as usize,
    );
    // Faces bordering a coarser LOD → transition (high-res) sides. Bit order matches `TransitionSide`.
    const SIDES: [TransitionSide; 6] = [
        TransitionSide::LowX,
        TransitionSide::HighX,
        TransitionSide::LowY,
        TransitionSide::HighY,
        TransitionSide::LowZ,
        TransitionSide::HighZ,
    ];
    let mut sides = TransitionSide::none();
    for (i, &s) in SIDES.iter().enumerate() {
        if flags & (1 << i) != 0 {
            sides |= s;
        }
    }
    let builder = ChunkMeshBuilder::new(edits, indices, grid_origin, vs, lod, debug);
    // MUST be CacheNothing: `CacheCentralBlockOnly` caches the central block at THIS chunk's (coarse)
    // resolution, which then serves the transition cell's FINE-resolution face samples too — collapsing the
    // transition so the cross-LOD weld fails. The analytic CSG field is cheap to re-evaluate, so just query it.
    let builder = extract_from_field(&field, FieldCaching::CacheNothing, block, sides, 0.0, builder);
    builder.finish()
}

/// `MeshBuilder` that turns Transvoxel's per-edge vertices into our `ChunkMeshData`: chunk-LOCAL positions,
/// EXACT analytic normals (from the CSG gradient, not Marching-Cubes' estimate), and the per-vertex
/// multi-material blend data. It first accumulates the indexed Transvoxel output (one entry per unique edge
/// vertex: position, analytic normal, NEAREST material id), then `finish` UN-INDEXES it one triangle at a
/// time so each triangle's three vertices carry the SAME `(mat_a, mat_b)` pair — material ids can't be
/// interpolated (rounding an interpolated id passes through phantom intermediate materials → colour bands),
/// so they must be constant across a triangle. The pair is `(matA = the surface material, matB = the nearby
/// competing material)` — taken from the corners' nearest AND runner-up so a triangle sitting ENTIRELY on one
/// surface still blends toward a nearby second surface (a WIDE feather, not just the one-triangle seam strip).
/// Each corner's blend coordinate is a SIGNED gap `d(matB) − d(matA)` against that fixed pair (sign consistent
/// across the triangle); the shader feathers it by `blend_softness`. Where matB is irrelevant (deep interior),
/// the gap is huge ⇒ weight pins to pure A ⇒ matB is never sampled (no spurious blend, no phantom). In debug
/// ("Colour by LOD") the per-LOD tint is written into `colors.rgb` and the shader renders it unlit.
struct ChunkMeshBuilder<'a> {
    edits: &'a [edits::ResolvedEdit],
    /// The chunk's candidate edits. Culled with AABBs PADDED by `BLEND_REACH` (see `mesh_resident_chunks`),
    /// so it lists not just the edits whose surface enters the chunk but every edit whose MATERIAL blend
    /// could reach it. That makes the material pair/gap consistent across chunk borders AND folds those
    /// edits into the content hash (so moving a blend-contributing edit re-bakes the chunk — no remnant).
    indices: &'a [u32],
    origin: Vec3,
    eps: f32,
    /// This chunk's voxel size (world metres/voxel) — the LOD context forwarded to every CSG eval so
    /// the Terrain sample picks the band-limited height mip for this LOD (see `edits::eval_primitive`).
    vs: f32,
    lod: u32,
    debug: bool,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    /// Per unique vertex: `(nearest, runner-up)` CSG material ids (the top-2 argmin). The triangle pair folds
    /// from the three corners' values; `runner-up == nearest` when only one material is present at the vertex.
    vmat: Vec<(u16, u16)>,
    tris: Vec<u32>,
}

impl<'a> ChunkMeshBuilder<'a> {
    fn new(
        edits: &'a [edits::ResolvedEdit],
        indices: &'a [u32],
        origin: Vec3,
        vs: f32,
        lod: u32,
        debug: bool,
    ) -> Self {
        Self {
            edits,
            indices,
            origin,
            eps: vs * 0.01,
            vs,
            lod,
            debug,
            positions: Vec::new(),
            normals: Vec::new(),
            vmat: Vec::new(),
            tris: Vec::new(),
        }
    }

    fn finish(self) -> Option<ChunkMeshData> {
        if self.positions.is_empty() || self.tris.is_empty() {
            return None;
        }
        let cap = self.tris.len();
        let mut positions = Vec::with_capacity(cap);
        let mut normals = Vec::with_capacity(cap);
        let mut uvs = Vec::with_capacity(cap);
        let mut colors = Vec::with_capacity(cap);
        let mut indices = Vec::with_capacity(cap);
        let tint = LOD_DEBUG_PALETTE[(self.lod as usize).min(LOD_DEBUG_PALETTE.len() - 1)];

        for t in self.tris.chunks_exact(3) {
            let v = [t[0] as usize, t[1] as usize, t[2] as usize];
            // The triangle's two materials: the majority NEAREST (the surface) and the majority RUNNER-UP (the
            // nearby competitor), then ORDERED BY ID into (mat_a, mat_b). Ordering is critical: the surface
            // material flips from one to the other ACROSS a seam, so if `mat_a` tracked "which is the surface"
            // it would swap A↔B at the seam and `weight` (= fraction of A) would jump. Sorting by id makes the
            // pair identical on both sides, so the signed gap drives a CONTINUOUS pure-A → 0.5 → pure-B fade.
            // m0 == m1 ⇒ single material (no blend).
            let near = [self.vmat[v[0]].0, self.vmat[v[1]].0, self.vmat[v[2]].0];
            let runner = [self.vmat[v[0]].1, self.vmat[v[1]].1, self.vmat[v[2]].1];
            let m0 = majority(near);
            let m1 = majority(runner);
            let (mat_a, mat_b) = (m0.min(m1), m0.max(m1));
            for &vi in &v {
                let n = positions.len() as u32;
                positions.push(self.positions[vi]);
                normals.push(self.normals[vi]);
                uvs.push([mat_a as f32, mat_b as f32]);
                let col = if self.debug {
                    [tint[0], tint[1], tint[2], 1.0]
                } else {
                    // SIGNED WORLD-DISTANCE to the material seam against the triangle's fixed pair: >0 ⇒
                    // nearer mat_a, <0 ⇒ nearer mat_b. The raw gap `d(matB)−d(matA)` is a distance DIFFERENCE
                    // whose magnitude compresses where the two surfaces are near-parallel (so a fixed world
                    // `blend_softness` band could never reach pure colours). Dividing by |∇gap| linearises it
                    // to the actual world distance to the gap==0 isosurface (the seam): the blend then
                    // localises to where the surfaces truly cross at an angle, and reaches pure A / pure B
                    // away from it, with `blend_softness` a real world half-width. Single-material triangle
                    // (mat_a == mat_b) ⇒ 0 (the shader's pair-equal guard then keeps it pure A).
                    let seam = if mat_a == mat_b {
                        0.0
                    } else {
                        let world = Vec3::from(self.positions[vi]) + self.origin;
                        let gap = |w: Vec3| {
                            edits::material_dist(self.edits, self.indices, w, mat_b, self.vs)
                                - edits::material_dist(self.edits, self.indices, w, mat_a, self.vs)
                        };
                        let g = gap(world);
                        let e = self.eps;
                        let grad = Vec3::new(
                            gap(world + Vec3::X * e) - gap(world - Vec3::X * e),
                            gap(world + Vec3::Y * e) - gap(world - Vec3::Y * e),
                            gap(world + Vec3::Z * e) - gap(world - Vec3::Z * e),
                        ) / (2.0 * e);
                        g / grad.length().max(1e-3)
                    };
                    [1.0, 1.0, 1.0, seam]
                };
                colors.push(col);
                indices.push(n);
            }
        }
        Some(ChunkMeshData { positions, normals, uvs, colors, indices })
    }
}

/// Majority of three ids — the value present ≥2×, else (all distinct) the min id. Deterministic in the id
/// SET (order-independent), so two triangles sharing an edge fold the same pair from their shared corners.
fn majority(x: [u16; 3]) -> u16 {
    if x[0] == x[1] || x[0] == x[2] {
        x[0]
    } else if x[1] == x[2] {
        x[1]
    } else {
        x[0].min(x[1]).min(x[2])
    }
}

impl MeshBuilder<f32, f32> for ChunkMeshBuilder<'_> {
    fn add_vertex_between(
        &mut self,
        a: GridPoint<f32, f32>,
        b: GridPoint<f32, f32>,
        t: f32,
    ) -> VertexIndex {
        let p = a.position.interpolate_toward(&b.position, t);
        let world = Vec3::new(p.x, p.y, p.z);
        let local = world - self.origin;
        self.positions.push([local.x, local.y, local.z]);
        // Exact outward normal = ∇(CSG distance) (points toward increasing distance = outside the solid).
        let n = field_gradient(self.edits, self.indices, world, self.eps, self.vs).normalize_or_zero();
        self.normals.push([n.x, n.y, n.z]);
        // (nearest, runner-up) materials at this vertex over the blend-padded chunk set; `finish` folds the
        // per-triangle pair from the three corners' values.
        let (near, runner, _) = edits::fold_csg_top2(self.edits, self.indices, world, self.vs);
        self.vmat.push((near, runner));
        VertexIndex(self.positions.len() - 1)
    }

    fn add_triangle(&mut self, v1: VertexIndex, v2: VertexIndex, v3: VertexIndex) {
        // Material is double-sided (cull_mode None) and normals are analytic, so winding is irrelevant.
        self.tris.extend_from_slice(&[v1.0 as u32, v2.0 as u32, v3.0 as u32]);
    }
}

/// Cheap narrow-band test: could the chunk's sampled region contain a surface crossing? Mirrors the GPU
/// scheduler's `narrow_band_keep`. For a LARGE solid most resident chunks are fully INTERIOR (they
/// overlap the edit AABB but the surface is nowhere near) — baking them is a wasted `edge³` sample +
/// Transvoxel that returns empty. Folding ONCE at the chunk centre and comparing `|dist|` to the
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
                    let d = edits::fold_csg_dist_indexed(edits, indices, min + Vec3::new(dx, dy, dz), vs);
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
    edits::fold_csg_dist_indexed(edits, indices, center, vs).abs() <= reach
}

/// Content-hash-driven, async, generational-coherent Transvoxel bake (see the module doc). The unit is
/// a configurable `K×K×K`-brick chunk; whole edits commit uniformly via frozen bake rounds.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn mesh_resident_chunks(
    mut commands: Commands,
    volumes: Query<VolumeQueryData, With<SdfVolume>>,
    config: Res<SdfGridConfig>,
    mesh_cfg: Res<MeshBakeConfig>,
    // Drives the clipmap LOD (finer near the camera). No `SdfCamera` ⇒ single-LOD fallback (mesh
    // everything at LOD 0 — the original scene/camera-independent behaviour for gameplay scenes).
    cameras: Query<&GlobalTransform, (With<SdfCamera>, Without<SdfVolume>)>,
    chunk_meshes: Query<(Entity, &ChunkMesh)>,
    mut states: ResMut<ChunkStates>,
    mut rebuild: ResMut<MeshBakeRebuild>,
    mut stats: ResMut<MeshBakeStats>,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    // The single shared triplanar `MeshMaterial` handle (built by `mesh_material::rebuild_mesh_material`);
    // EVERY chunk mesh uses it — the per-vertex ids + blend weight select/cross-fade materials in-shader.
    mesh_mats: Res<super::mesh_material::MeshMaterials>,
    // Bundled scalar Locals: rebake epoch, prev K.
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

    let cs = config.cell_stride() as u32; // cells per brick (chunk subdivisions = k·cs)

    let n_edits = gathered.len();
    let mut edit_aabbs: Vec<Aabb3d> = Vec::with_capacity(n_edits);
    let mut edit_vec: Vec<edits::ResolvedEdit> = Vec::with_capacity(n_edits);
    // Sub-voxel-cull SSOT (`edit_resolvable_at`): each edit's GEOMETRY extent (unpadded — resolvability is a
    // property of the surface size, not the blend reach). Indexed like `edit_aabbs`.
    let mut edit_extent: Vec<f32> = Vec::with_capacity(n_edits);
    for g in &gathered {
        edit_extent.push((Vec3::from(g.aabb.max) - Vec3::from(g.aabb.min)).max_element());
        // PAD the cull/hash AABB by the max material-blend reach: a material's cross-fade bleeds up to
        // `blend_softness` (≤ `BLEND_REACH`) world units beyond its surface onto a NEIGHBOUR, so a chunk
        // within that range must list this edit in its `idx` — otherwise its content hash omits the edit and
        // MOVING the edit leaves a stale blended remnant on the neighbour (it never re-bakes). A fixed pad
        // (not per-material) keeps the baked seam-distance blend-softness-INDEPENDENT, so softness stays live.
        let pad = bevy::math::Vec3A::splat(BLEND_REACH);
        edit_aabbs.push(Aabb3d { min: g.aabb.min - pad, max: g.aabb.max + pad });
        edit_vec.push(g.edit.clone());
    }
    let edits_arc = Arc::new(edit_vec);

    // COVERAGE GATE inputs: the world-XZ AABBs of every `Terrain` edit, and a snapshot of the loaded
    // height ring. A chunk that samples a `Terrain` primitive must NOT become resident until its full XZ
    // footprint is backed by LOADED height — otherwise an oversized far-LOD chunk would sample OUTSIDE
    // the ±radius ring and (now strictly) panic, instead of silently rendering a corrupt flat slab. The
    // ring is the world-anchored toroidal clipmap the worldgen plugin rolls (`worldgen::upload`).
    let terrain_xz_aabbs: Vec<(Vec2, Vec2)> = gathered
        .iter()
        .filter(|g| matches!(g.edit.prim, edits::SdfPrimitive::Terrain { .. }))
        .map(|g| {
            (
                Vec2::new(g.aabb.min.x, g.aabb.min.z),
                Vec2::new(g.aabb.max.x, g.aabb.max.z),
            )
        })
        .collect();
    let height_ring = crate::sdf_render::worldgen::upload::cpu_height_ring();

    // The baked mesh is appearance-INDEPENDENT: vertices carry only geometry + top-2 material *ids* + a blend
    // weight, never colours/PBR scalars. A material colour/PBR edit therefore needs no re-bake — the shared
    // `MeshMaterials` table + texture arrays rebuild themselves on `MaterialRegistry` change (see
    // `mesh_material.rs`). So only "Rebake all" (button) bumps the global epoch → full re-bake.
    if std::mem::replace(&mut rebuild.0, false) {
        scal.epoch = scal.epoch.wrapping_add(1);
    }
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
                if !mesh_chunk_in_shell(key, &config, k, cam, half0) {
                    continue;
                }
                // COVERAGE GATE: if this chunk's world-XZ footprint touches any `Terrain` edit, it must
                // be fully backed by loaded height before it can mesh — never mesh ground the artifact
                // hasn't loaded (no silent flat-plane fallback). A km-wide far-LOD chunk against the
                // ±radius ring fails this → it stops at the coverage edge (the corrupt far slab is gone);
                // as the ring rolls, newly-covered chunks enter resident per-chunk, evicted ones leave.
                if !terrain_xz_aabbs.is_empty() && !terrain_chunk_covered(
                    key, &config, k, &terrain_xz_aabbs, height_ring.as_deref(),
                ) {
                    continue;
                }
                resident.insert(key);
            }
        }
    }

    // Current content hash for every resident chunk (over the LIVE edits + lod + per-face transition flags) —
    // drives "is the displayed mesh out of date" (a NEW round needed). The lod+flags mix makes a chunk re-bake
    // (with the right Transvoxel transition sides) exactly when the camera moves a shell line. Transvoxel needs
    // only the per-face LOD RELATIONSHIP (already in `flags`), NOT the neighbour's geometry — so no cross-chunk
    // hash folding is required (the transition cell samples the field itself and welds by construction).
    let mut current_hashes: HashMap<BrickKey, u64> = HashMap::with_capacity(resident.len());
    let mut by_lod = [0usize; 8];
    {
        let mut idx: Vec<u32> = Vec::new();
        for &key in &resident {
            by_lod[(key.lod as usize).min(7)] += 1;
            cull_into(&edit_aabbs, &chunk_sampled(key), &mut idx);
            // Drop edits that are sub-voxel at this chunk's LOD so a tiny object can't contaminate a chunk
            // resident for a larger one (the residency cull already keeps lone sub-voxel objects out). Same
            // predicate as the bake fold below → hash and geometry always agree.
            idx.retain(|&i| edit_resolvable_at(edit_extent[i as usize], &config, key.lod));
            let base = if idx.is_empty() { 0 } else { edits::bake_content_hash(&edits_arc, &idx) };
            let flags = chunk_finer_faces(key, &config, k, cam, half0);
            let lf = (key.lod as u64).wrapping_mul(0xA24B_AED4_963E_E407)
                ^ (flags as u64).wrapping_mul(EPOCH_MIX);
            current_hashes.insert(key, (base ^ epoch_mix).wrapping_add(lf));
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

    // Free pending work for chunks OUTSIDE the round's FROZEN residency (a chunk that left the round's set —
    // e.g. the live camera moved on). Their displayed entity is HELD until the round COMMIT reaps it, so old
    // geometry only clears as the new round appears.
    for (key, st) in states.0.iter_mut() {
        if !round.resident.contains(key) {
            st.staged = None;
            st.task = None;
        }
    }

    // 2. COMMIT the round when every chunk of its FROZEN residency is settled — none still baking, and each
    // either already displays its target or holds a staged bake of it. Swap them ALL in one frame (and reap
    // every mesh outside the frozen set the same frame) so a whole edit — or a whole LOD shift — pops together
    // with no 1-frame hole. We commit against `round.resident`, NOT the live set, so the round only ever
    // displays a coherent residency it actually finished baking.
    let round_done = round.resident.iter().all(|key| match states.0.get(key) {
        Some(st) => st.task.is_none() && (st.displayed_hash == st.target_hash || st.staged.is_some()),
        None => true, // not tracked yet → nothing to wait on (a frozen-set chunk always has a state)
    });
    let has_staged = states.0.values().any(|s| s.staged.is_some());
    let has_departed = chunk_meshes.iter().any(|(_, cm)| !round.resident.contains(&cm.0));
    stats.reaped = 0;
    if round.edits.is_some() && round_done && (has_staged || has_departed) {
        let mut reaped = 0usize;
        // Build + spawn each committing chunk's staged mesh in one pass (Transvoxel welds neighbouring LODs by
        // construction — no cross-chunk seam step). ONE mesh + ONE entity per chunk, all sharing the single
        // triplanar `MeshMaterial` handle: per-vertex top-2 material ids (UV_0) + blend weight (COLOR.a) drive
        // the in-shader material select / cross-fade, so co-located materials no longer fight over a dominant.
        for (key, st) in states.0.iter_mut() {
            let Some(sb) = st.staged.take() else {
                continue;
            };
            for old in st.entities.drain(..) {
                commands.entity(old).despawn();
            }
            st.displayed_hash = st.target_hash;
            let Some(data) = sb.data else {
                continue; // empty chunk: no mesh
            };
            // Transvoxel positions are chunk-LOCAL relative to the chunk's world MIN corner (NO apron), so the
            // entity Transform is exactly `brick_min_world(coord, lod)`.
            let origin = config.brick_min_world(key.coord, key.lod);
            let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, data.positions)
                .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals)
                .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, data.uvs)
                .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, data.colors)
                .with_inserted_indices(Indices::U32(data.indices));
            let e = commands
                .spawn((
                    Mesh3d(mesh_assets.add(mesh)),
                    MeshMaterial3d(mesh_mats.handle.clone()),
                    Transform::from_translation(origin),
                    ChunkMesh(*key),
                    Name::new("SDF Chunk Mesh"),
                ))
                .id();
            st.entities.push(e);
        }

        // Reap every mesh OUTSIDE the frozen round set (query-based, so it also catches orphans). The new set
        // was fully spawned above, so this is the atomic old-out half of the swap — there is no hole because
        // the new geometry is already on screen this same frame. A re-baked resident chunk's OLD entity was
        // already despawned in the spawn loop (its key stays in the set), so it is not double-despawned here.
        for (e, cm) in &chunk_meshes {
            if !round.resident.contains(&cm.0) {
                commands.entity(e).despawn();
                reaped += 1;
            }
        }
        states.0.retain(|key, _| round.resident.contains(key));
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
            round.cam = cam; // freeze the camera so the round's transition flags are self-consistent
            round.half0 = half0;
            round.resident = resident.clone(); // FREEZE the residency — the round bakes/commits/reaps this set
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
        let displayed_n = states.0.values().filter(|s| !s.entities.is_empty()).count();
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
        // Bake the round's FROZEN residency (not the live set), so the bake, commit, and reap all agree.
        for &key in &round.resident {
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
            // Transvoxel block = the chunk's exact world extent (NO apron); its origin is the chunk MIN corner.
            let grid_origin = config.brick_min_world(key.coord, key.lod);
            // Transvoxel transition faces — those bordering a FINER LOD — from the FROZEN shell, so all of a
            // round's chunks agree on the boundary. Folded into the content hash → re-bakes on a shell move.
            let flags = chunk_finer_faces(key, &config, k, round.cam, round.half0);
            let lod = key.lod;
            let edits = round_edits.clone();
            let indices = idx.clone();
            st.task = Some(pool.spawn(async move {
                mesh_chunk(&edits, &indices, grid_origin, vs_l, k * cs, flags, lod, debug)
            }));
            budget -= 1;
        }
    }
}

/// Dedicated "Mesh Bake" bottom dock panel (editor builds): the controls for viewing/inspecting the
/// Transvoxel bake.
#[cfg(feature = "editor")]
fn mesh_bake_panel(world: &mut World, ui: &mut bevy_egui::egui::Ui) {
    use bevy::pbr::wireframe::WireframeConfig;
    use crate::sdf_render::SdfRenderEnabled;

    ui.label("Transvoxel chunk bake (async). Baked meshes are the renderer.");
    ui.separator();

    // The SDF raymarch render is gone (meshes render the scene now). This flag now only gates the GPU
    // SDF-volume brick bake, kept off by default as a future volumetric-cloud foundation — enable it
    // only when working on that (it has no on-screen output on its own and costs bake time).
    let mut bake_on = world.resource::<SdfRenderEnabled>().0;
    if ui
        .checkbox(&mut bake_on, "GPU SDF volume bake (clouds; off)")
        .on_hover_text("Runs the GPU brick bake into the SDF atlas — scaffolding for a future cloud raymarcher. No visible output yet.")
        .changed()
    {
        world.resource_mut::<SdfRenderEnabled>().0 = bake_on;
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
    let mut dbg = world.resource::<MeshBakeConfig>().debug_lod_colour;
    if ui.checkbox(&mut dbg, "Colour by LOD (debug)").changed() {
        world.resource_mut::<MeshBakeConfig>().debug_lod_colour = dbg;
    }
    let mut dbg_n = world.resource::<MeshBakeConfig>().debug_normals;
    if ui
        .checkbox(&mut dbg_n, "View normals (debug)")
        .on_hover_text("Render the mesh world-normal as RGB (unlit) to inspect the baked geometry normals.")
        .changed()
    {
        world.resource_mut::<MeshBakeConfig>().debug_normals = dbg_n;
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
    let meshes = states.0.values().map(|s| s.entities.len()).sum::<usize>();
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

    /// The COVERAGE GATE excludes a terrain chunk whose XZ footprint reaches OUTSIDE the loaded height
    /// ring, and admits one fully inside it. Build a small resident ring (a 4×4 chunk block at the
    /// origin), then check a fine LOD-0 chunk well inside is covered while a HUGE far chunk (and any
    /// chunk against a `None` ring) is not — exactly the gate that kills the corrupt oversized far slab.
    #[test]
    fn coverage_gate_excludes_uncovered_terrain_chunk() {
        use crate::sdf_render::worldgen::artifact::ScalarField2D;
        use crate::sdf_render::worldgen::coord::{ChunkCoord, ChunkSize, LayerId};
        use crate::sdf_render::worldgen::layers::height::{
            HEIGHT_CHUNK_CELLS, HEIGHT_FIELD_RES, HeightLayer, HeightParams,
        };
        use crate::sdf_render::worldgen::store::ArtifactStore;
        use crate::sdf_render::worldgen::upload::build_height_ring;
        use std::sync::Arc;

        // Build a resident ring covering height chunks (-3..5, -3..5) around the origin — a generous
        // loaded block so a chunk near the origin clears the gate's `2·HEIGHT_CHUNK_CELLS` apron margin.
        let layer = HeightLayer::new(LayerId(0), HeightParams::default());
        let size = ChunkSize::new(HEIGHT_CHUNK_CELLS);
        let mut store = ArtifactStore::new();
        for cz in -3..5 {
            for cx in -3..5 {
                let coord = ChunkCoord::new(LayerId(0), IVec3::new(cx, 0, cz));
                let mut field = ScalarField2D::zeroed(coord, size, HEIGHT_FIELD_RES);
                for j in 0..=HEIGHT_FIELD_RES {
                    for i in 0..=HEIGHT_FIELD_RES {
                        let wp = field.node_world_xz(i, j);
                        field.set(i, j, layer.sample_world(wp.x, wp.y, 1));
                    }
                }
                store.insert(coord, Arc::new(field));
            }
        }
        let ring = Arc::new(build_height_ring(&store));

        let (cfg, _mc) = cfgs();
        let k = 4u32;
        // One global terrain edit whose XZ footprint spans everything (effectively infinite, as in prod).
        let big = 131072.0f32;
        let terrain = vec![(Vec2::splat(-big), Vec2::splat(big))];

        // A fine LOD-0 chunk at the origin → deep inside the loaded block → covered → gate passes.
        let inside_coord = chunk(&cfg, k, 0, 0);
        assert!(
            terrain_chunk_covered(inside_coord, &cfg, k, &terrain, Some(&ring)),
            "a fine chunk inside the loaded block must pass the coverage gate"
        );

        // A HUGE far chunk: a coarse LOD that spans kilometres reaches far outside the ±loaded ring.
        let far = chunk(&cfg, k, 7, 64);
        assert!(
            !terrain_chunk_covered(far, &cfg, k, &terrain, Some(&ring)),
            "an oversized far chunk must be excluded (outside loaded coverage)"
        );

        // No ring loaded yet ⇒ any terrain-touching chunk is excluded.
        assert!(
            !terrain_chunk_covered(inside_coord, &cfg, k, &terrain, None),
            "with no ring loaded, a terrain chunk must not be resident"
        );

        // A chunk that touches NO terrain edit is unaffected by the gate (passes regardless of ring).
        let no_terrain: Vec<(Vec2, Vec2)> = Vec::new();
        assert!(terrain_chunk_covered(far, &cfg, k, &no_terrain, None));
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
    fn finer_faces_mark_inner_rim_transitions() {
        let (cfg, mc) = cfgs();
        let k = 4;
        let cam = Some(Vec3::ZERO);
        let half0 = lod0_half_chunks(&cfg, &mc, k);
        // LOD 0 (finest) never has transition faces.
        assert_eq!(chunk_finer_faces(chunk(&cfg, k, 0, 0), &cfg, k, cam, half0), 0, "LOD 0 has no finer faces");
        // A LOD-1 chunk on the INNER rim of its shell: index half0/2 occupies LOD-0 [half0, half0+2); its −X
        // neighbour occupies [half0-2, half0), fully inside the finer LOD-0 cube [−half0, half0] → its −X face
        // is a transition (it borders the finer LOD).
        let f = chunk_finer_faces(chunk(&cfg, k, 1, half0 / 2), &cfg, k, cam, half0);
        assert_eq!(f & (1 << 0), 1 << 0, "−X face should border the finer LOD-0 cube (a transition face)");
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
        let mut tris = Vec::new();
        for t in data.indices.chunks_exact(3) {
            let v = |i: u32| origin + Vec3::from(data.positions[i as usize]);
            tris.push((v(t[0]), v(t[1]), v(t[2])));
        }
        tris
    }

    /// Count mesh edges NOT shared by exactly 2 triangles, after welding vertices by quantized WORLD
    /// position (0.1 mm). 0 ⇒ closed 2-manifold = watertight. Position-welding lets it span SEPARATE chunk
    /// meshes (fine + coarse), so it is the cross-LOD correctness gate: the Transvoxel transition face must
    /// weld the two with no open edge (gap) and no edge in >2 triangles (overlap).
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
        // A sphere fully inside one chunk (touching no face) must mesh as a closed 2-manifold.
        let edits = [sphere_edit(Vec3::ZERO, 1.0)];
        let (vs, sub) = (0.1f32, 28u32); // block span = 28·0.1 = 2.8 > sphere Ø 2.0 → clears all faces
        let origin = Vec3::splat(-1.4);
        let data = mesh_chunk(&edits, &[0], origin, vs, sub, 0, 0, false).expect("sphere meshes");
        assert_eq!(open_edge_count(&chunk_tris(&data, origin)), 0, "closed sphere must be watertight");
    }

    #[test]
    fn transvoxel_2to1_boundary_is_watertight() {
        // THE crack-free guarantee, by construction: the COARSE block whose +X face is a TRANSITION (toward
        // the higher-res neighbour) welds to its abutting FINE block with NO post-hoc stitching. Transvoxel
        // puts the transition cell on the LOW-res block facing the high-res one. A sphere straddles the forced
        // 2:1 boundary at x = 0: the fine block (vs 0.1) meshes x∈[0,2.8] REGULAR; the coarse block (vs 0.2)
        // meshes x∈[−5.6,0] with +X (bit 1 = HighX) transition. Origins sit on the world-0 coarse lattice
        // (Y/Z origins are integer multiples of vsc), so the transition face samples coincide with the fine
        // face. Fine + coarse must be a closed 2-manifold (no gap, no overlap) at the shared plane.
        // Sphere offset off the boundary so x=0 cuts it TRANSVERSALLY (the equator-on-the-plane case is
        // tangent — degenerate for a transition cell).
        let edits = [sphere_edit(Vec3::new(0.4, 0.0, 0.0), 1.0)];
        let idx = [0u32];
        let (vsf, vsc, sub) = (0.1f32, 0.2f32, 28u32);
        let of = Vec3::new(0.0, -1.4, -1.4); // fine x∈[0,2.8]; −X face at x=0 (regular, high-res)
        let oc = Vec3::new(-5.6, -2.8, -2.8); // coarse x∈[−5.6,0]; +X face at x=0 is the transition side
        let fine = mesh_chunk(&edits, &idx, of, vsf, sub, 0, 0, false).expect("fine meshes");
        let coarse = mesh_chunk(&edits, &idx, oc, vsc, sub, 1 << 1, 1, false).expect("coarse meshes");
        let mut all = chunk_tris(&fine, of);
        all.extend(chunk_tris(&coarse, oc));
        assert_eq!(
            open_edge_count(&all),
            0,
            "coarse (with transition face) + fine must weld watertight by construction"
        );
    }

    /// Bake a sphere ∪ cube (both centred at origin, so the solid is star-shaped about origin) in one chunk.
    /// `mat_s`/`mat_c` are the sphere/cube material ids. Returns the baked mesh + its world origin.
    fn merged_sphere_cube(mat_s: u16, mat_c: u16, smoothing: f32) -> (ChunkMeshData, Vec3) {
        use crate::sdf_render::{CsgKind, SdfOp, SdfPrimitive};
        let edits = [
            edits::ResolvedEdit::new(
                SdfPrimitive::Sphere { radius: 1.0 },
                Transform::from_translation(Vec3::ZERO),
                SdfOp { kind: CsgKind::Union, smoothing: 0.0 },
                mat_s,
            ),
            edits::ResolvedEdit::new(
                SdfPrimitive::Box { half_extents: Vec3::splat(0.8) },
                Transform::from_translation(Vec3::ZERO),
                SdfOp { kind: CsgKind::Union, smoothing },
                mat_c,
            ),
        ];
        let (vs, sub) = (0.1f32, 32u32); // span 3.2 > shape Ø (cube corner ≈ 1.39) → closed in one chunk
        let origin = Vec3::splat(-1.6);
        let data = mesh_chunk(&edits, &[0, 1], origin, vs, sub, 0, 0, false).expect("merged shape meshes");
        (data, origin)
    }

    #[test]
    fn merged_sphere_cube_normals_point_outward() {
        // The solid is star-shaped about origin, so EVERY outward surface normal `n` must satisfy
        // dot(n, pos) > 0. A normal pointing inward (the dark-triangle bug) shows up as dot < 0.
        let (data, origin) = merged_sphere_cube(1, 2, 0.0);
        let (mut worst, mut inward, mut degenerate, mut oblique) = (1.0f32, 0, 0, 0);
        for i in 0..data.positions.len() {
            let pos = Vec3::from(data.positions[i]) + origin;
            let n = Vec3::from(data.normals[i]);
            if n.length() < 0.5 {
                degenerate += 1;
                continue;
            }
            let d = n.normalize().dot(pos.normalize_or_zero());
            worst = worst.min(d);
            if d < -0.05 {
                inward += 1;
            } else if d < 0.5 {
                oblique += 1; // >60° off outward — would shade noticeably dark vs neighbours
            }
        }
        println!(
            "verts={} degenerate={degenerate} inward={inward} oblique={oblique} worst_dot={worst:.3}",
            data.positions.len()
        );
        assert_eq!(degenerate, 0, "no degenerate (black) normals");
        assert_eq!(inward, 0, "no inward-pointing normals (dark triangles); worst dot {worst:.3}");
    }

    #[test]
    fn coarse_transition_normals_point_outward() {
        // The dark patches were on the coarse-LOD / transition side. Bake the COARSE block of a 2:1 boundary
        // (with a +X TRANSITION face) — its transition cells are the suspect geometry. A sphere centred at
        // (0.4,0,0) crosses the block's +X face (x=0); every normal must point outward from the sphere centre.
        let edits = [sphere_edit(Vec3::new(0.4, 0.0, 0.0), 1.0)];
        let oc = Vec3::new(-5.6, -2.8, -2.8);
        let coarse = mesh_chunk(&edits, &[0], oc, 0.2, 28, 1 << 1, 1, false).expect("coarse+transition meshes");
        let center = Vec3::new(0.4, 0.0, 0.0);
        let (mut worst, mut inward, mut degenerate) = (1.0f32, 0, 0);
        for i in 0..coarse.positions.len() {
            let pos = Vec3::from(coarse.positions[i]) + oc;
            let n = Vec3::from(coarse.normals[i]);
            if n.length() < 0.5 {
                degenerate += 1;
                continue;
            }
            let d = n.normalize().dot((pos - center).normalize_or_zero());
            worst = worst.min(d);
            if d < -0.05 {
                inward += 1;
            }
        }
        println!(
            "coarse+transition verts={} degenerate={degenerate} inward={inward} worst_dot={worst:.3}",
            coarse.positions.len()
        );
        assert_eq!(degenerate, 0, "no degenerate (black) normals on the transition mesh");
        assert_eq!(inward, 0, "transition-cell normals must point outward; worst dot {worst:.3}");
    }

    #[test]
    fn blend_reaches_pure_colours_and_blends() {
        // The baked COLOUR.a is the signed WORLD-DISTANCE to the seam, so a world-unit `blend_softness` band
        // must yield the FULL range: pure A (weight→1) and pure B (weight→0) away from the seam, plus a
        // genuine transition between. (The earlier raw-gap version compressed to a muddy ~50% everywhere on
        // these unit-scale objects — the "won't blend from full red to white" bug.)
        let (data, _) = merged_sphere_cube(2, 5, 0.3);
        // Mirror the shader's directional ramp with both softness = 0.25 (denom 0.5).
        let (da, db) = (0.25f32, 0.25f32);
        let (mut pure_a, mut pure_b, mut mid) = (0u32, 0u32, 0u32);
        for c in &data.colors {
            let w = ((c[3] + da) / (da + db)).clamp(0.0, 1.0);
            if w > 0.9 {
                pure_a += 1;
            } else if w < 0.1 {
                pure_b += 1;
            } else if (0.3..=0.7).contains(&w) {
                mid += 1;
            }
        }
        assert!(pure_a > 0, "blend must reach pure A (weight→1) away from the seam");
        assert!(pure_b > 0, "blend must reach pure B (weight→0) away from the seam");
        assert!(mid > 0, "blend must have a genuine transition region (not a hard cut)");
    }

    #[test]
    fn merged_sphere_cube_blend_has_no_phantom_materials() {
        // Sphere (id 1) ∪ cube (id 7). Every emitted vertex must reference ONLY {1, 7} (no phantom
        // intermediate id), the pair must be id-ordered + CONSTANT within each triangle (so smooth UV
        // interpolation can't sweep through other ids), and the signed gap must change sign across the
        // merge (a real blend region exists).
        let (data, _) = merged_sphere_cube(1, 7, 0.3);
        for uv in &data.uvs {
            let (a, b) = (uv[0] as u16, uv[1] as u16);
            assert!(a == 1 || a == 7, "matA {a} is a phantom material");
            assert!(b == 1 || b == 7, "matB {b} is a phantom material");
            assert!(a <= b, "pair must be id-ordered, got ({a},{b})");
        }
        for t in data.indices.chunks_exact(3) {
            let uvs: Vec<_> = t.iter().map(|&i| data.uvs[i as usize]).collect();
            assert!(uvs[0] == uvs[1] && uvs[1] == uvs[2], "material pair must be constant within a triangle");
        }
        let (mut pos_gap, mut neg_gap) = (false, false);
        for c in &data.colors {
            if c[3] > 0.05 {
                pos_gap = true;
            } else if c[3] < -0.05 {
                neg_gap = true;
            }
        }
        assert!(pos_gap && neg_gap, "signed gap must straddle 0 → a real A↔B blend region exists");
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
    fn no_camera_is_lod0_everywhere_no_transition_faces() {
        let (cfg, mc) = cfgs();
        let k = 4;
        let half0 = lod0_half_chunks(&cfg, &mc, k);
        assert!(mesh_chunk_in_shell(chunk(&cfg, k, 0, 9), &cfg, k, None, half0), "LOD 0 everywhere");
        assert!(!mesh_chunk_in_shell(chunk(&cfg, k, 1, 9), &cfg, k, None, half0), "no LOD>0 w/o camera");
        assert_eq!(
            chunk_finer_faces(chunk(&cfg, k, 0, 3), &cfg, k, None, half0),
            0,
            "no transition faces w/o camera"
        );
    }
}
