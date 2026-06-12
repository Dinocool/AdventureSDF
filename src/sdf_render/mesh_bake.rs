//! SDF→mesh bake (see `docs/MESH_BAKE_PLAN.md`): a residency-driven, **async**, content-hash-driven
//! Transvoxel bake. The bake/render UNIT is a configurable **chunk** of `K×K×K` finest bricks
//! (`MeshBakeConfig::chunk_bricks`, runtime-tunable). `K = 1` is one mesh per finest brick; larger `K`
//! aggregates more bricks into one contiguous mesh — fewer draw calls/entities, coherent atomic swaps,
//! and contiguous geometry for later decimation/LOD.
//!
//! **Per-chunk, hash-driven lifecycle (the update model).** Each resident chunk is an INDEPENDENT state
//! machine ([`ChunkState`]) keyed on its LIVE desired content hash (`current_hashes[key]` = edits ⊕ lod ⊕
//! transition flags ⊕ epoch). There is NO frozen round; every chunk always advances toward the live target:
//!  1. REQUEST — a chunk needs a bake iff neither its `displayed` nor its `hidden` geometry matches the
//!     desired hash and no in-flight `task` already targets it. An in-flight task whose target is SUPERSEDED
//!     (the camera/edits moved on) is dropped + re-issued toward the live hash. Spawn order is `bake_priority`
//!     (near rings first, then in-view), budget-capped per frame.
//!  2. COMMIT — when a bake completes it is displayed (`commit_baked`): an object/empty chunk swaps in
//!     immediately; a terrain chunk is spawned `Visibility::Hidden` and held in `hidden` until its per-chunk
//!     material textures are GPU-live (keep-old-until-revealed — the OLD `displayed` stays on screen).
//!  3. REVEAL — `reveal_ready_chunks` flips a settled hidden mesh `Visible`, despawns the old `displayed`,
//!     and promotes it. The swap is lockstep with the material, so no green flash / no hole.
//!  4. EVICT — a displayed mesh is reaped only when a REVEALED resident chunk tiles its region
//!     (`render_commit_reap`), or when the chunk is `gone` (off the live clipmap). A chunk's own key ∈
//!     resident ⇒ it is never reaped — only superseded / out-of-residency meshes are.
//!
//! Because nothing is frozen, a chunk can never stall waiting for a round that never re-forms — the camera
//! can move continuously and new chunks always bake + appear. (The tradeoff vs the old atomic-band swap: an
//! LOD-shell snap re-bakes + reveals the affected band per-chunk; keep-old-until-revealed keeps cross-LOD
//! cracks brief/rare but not provably zero. See the `TODO(watertight)` on `mesh_resident_chunks`.)
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
use bevy::camera::primitives::Frustum;
use bevy::math::bounding::Aabb3d;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};
use transvoxel::prelude::*;
use transvoxel::structs::grid_point::GridPoint;
use transvoxel::structs::vertex_index::VertexIndex;
use transvoxel::traits::mesh_builder::MeshBuilder;

use crate::sdf_render::atlas::BrickKey;
use crate::sdf_render::worldgen::upload::HeightClipmap;
use crate::sdf_render::{
    edits, gather_sorted_edits, SdfCamera, SdfGridConfig, SdfVolume, VolumeQueryData,
};

/// Max NEW meshing tasks spawned per frame (the pool runs them concurrently; this bounds the spawn
/// burst when a large region enters at once).
const MAX_NEW_TASKS_PER_FRAME: usize = 256;

/// Hard ceiling on the mesh-bake LOD count (`lod_count` slider max). LODs `0..=MAX_MESH_LODS-1`. The
/// worldgen height clipmap derives its tier count from the live `lod_count`, so this also bounds the
/// sample-area window. Stats arrays + the debug tint cover this whole range so nothing clips at runtime.
/// (Kept ≤ 32 so the `[usize; MAX_MESH_LODS]` stat arrays still derive `Default`.)
pub(crate) const MAX_MESH_LODS: u32 = 32;

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
    /// For a TERRAIN-ONLY chunk: the baked per-chunk surface maps (volumetric biome strata + pristine
    /// surface height + the coarse-gated detail normal, over the chunk's world-XZ footprint). `Some` ⇒
    /// commit spawns the chunk with a dedicated `TerrainMaterial` (per-fragment biome strata + PBR);
    /// `None` ⇒ the chunk keeps the shared mesh material (mixed/object/CSG-cave chunks — no biome strata v1).
    terrain_surface: Option<super::terrain_material::TerrainSurfaceBake>,
}


/// The biome library + terrain texture arrays, bundled into one `SystemParam` so `mesh_resident_chunks`
/// (already at Bevy's param-arity limit) can read both from a single slot.
#[derive(bevy::ecs::system::SystemParam)]
struct TerrainMatRes<'w> {
    lib: Res<'w, super::worldgen::biome::BiomeLibrary>,
    tex: Res<'w, super::terrain_textures::TerrainTextureArrays>,
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

/// A freshly-spawned terrain chunk mesh that fades in lockstep with its per-chunk `TerrainMaterial` textures.
/// `frames_left` counts down a TWO-PHASE window: it's spawned `Visibility::Hidden`; after `REVEAL_SHOW_AFTER`
/// frames (its per-chunk `Image`s have extracted+prepared to the render world, so it can draw textured — not
/// flat-green) it's made `Visible`; then at `0` the OLD geometry it replaces is finally dropped. The gap
/// between "shown" and "old dropped" is a brief OVERLAP that guarantees the new mesh is actually on screen
/// before the old one leaves — so a GPU-upload that lags the counter can never open a hole.
#[derive(Component)]
struct PendingReveal {
    frames_left: u8,
}

/// Total frames from spawn until the OLD geometry a terrain chunk replaces is dropped (`frames_left` start).
/// Covers the per-chunk `Image` extract+prepare latency PLUS an overlap margin so the new mesh is provably
/// rendering before the old is reaped (keep-old-until-revealed). Larger = safer under heavy bursts, slightly
/// more streaming latency + overlap. 0 ⇒ no delay.
const REVEAL_SETTLE_FRAMES: u8 = 5;

/// Frames after spawn at which the (hidden) new mesh is made `Visible` — long enough for its per-chunk GPU
/// textures to be live (draws textured, not flat-green), short enough to leave an overlap before the old mesh
/// is dropped at `0`. So `[REVEAL_SHOW_AFTER, REVEAL_SETTLE_FRAMES]` is the new-and-old overlap window.
const REVEAL_SHOW_AFTER: u8 = 2;

/// Is a terrain chunk's replacement now SAFE TO SWAP — its settle window fully elapsed, so the new mesh is on
/// screen and the OLD geometry can be dropped (it counts as a "cover" for the keep-old-until-revealed reap)?
/// Pure → the reveal system and the reap share ONE testable predicate. (If the fixed window ever proves
/// insufficient, swap this for a true render-world `RenderAssets<GpuImage>` readiness check — callers
/// don't change.)
fn chunk_renderable(frames_left: u8) -> bool {
    frames_left == 0
}

/// Per-chunk bake state — a pure, hash-driven lifecycle keyed on the LIVE desired content hash (no frozen
/// round). A chunk always bakes toward the current `current_hashes[key]`; an in-flight bake whose target is
/// superseded is dropped and re-issued. This is what makes the post-fill stall impossible BY CONSTRUCTION.
#[derive(Default)]
struct ChunkState {
    /// Currently VISIBLE (revealed) mesh entities — ONE per material sub-mesh (empty = meshed-empty / nothing
    /// shown yet). These keep rendering until their replacement in `hidden` is REVEALED.
    displayed: Vec<Entity>,
    /// Content hash the `displayed` meshes were baked from (`0` = nothing shown).
    displayed_hash: u64,
    /// Freshly-baked replacement entities, spawned `Visibility::Hidden`, awaiting reveal (their per-chunk
    /// material textures GPU-live). On reveal: the old `displayed` are despawned and these are promoted to
    /// `displayed`. This is what swaps a re-baked/streamed chunk's OLD→NEW mesh in lockstep with its material.
    hidden: Vec<Entity>,
    /// Content hash the `hidden` meshes were baked from.
    hidden_hash: u64,
    /// Is `displayed` this chunk's CURRENT, on-screen result (material GPU-live)? An empty/no-surface bake sets
    /// this `true` with an empty `displayed`. A chunk counts as a "cover" for keep-old-until-revealed (drives
    /// `is_revealed` for the reap) ONLY when true — a not-yet-revealed replacement can't evict the geometry it
    /// overlaps. Set by `commit_baked` (object/empty) or `reveal_ready_chunks` (terrain).
    revealed: bool,
    /// The single in-flight meshing task, if any.
    task: Option<Task<Option<ChunkMeshData>>>,
    /// Content hash the in-flight `task` targets. A bake whose `task_hash != desired` is superseded → dropped.
    task_hash: u64,
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
    /// DETAIL-NORMAL bake: `N×N` texel resolution of the per-chunk surface-slope normal map baked onto
    /// coarse terrain-only chunks (`detail_normal.wgsl`). Higher = finer baked relief but more `N²` gradient
    /// samples per chunk + a larger per-chunk `Image`. Changing it re-bakes (the texel data changes).
    pub(crate) detail_normal_res: u32,
    /// DETAIL-NORMAL strength in `[0, 1]`: how far the per-pixel baked hi-fi normal pulls the coarse geometry
    /// normal (0 = no detail, 1 = full hi-fi detail). A LIVE shader uniform — no re-bake on change.
    pub(crate) detail_normal_strength: f32,
    /// BIOME map resolution (`N×N`) baked per terrain-only chunk (Stage 2). Biome is LOW-FREQUENCY
    /// (km-scale climate fields), so a small map suffices; it indexes the strata table per fragment.
    /// Changing it re-bakes (the texel data changes).
    pub(crate) biome_res: u32,
    /// BIOME-BORDER blend HALF-WIDTH in WORLD metres: the baked biome `blend` ramps a biome→neighbour
    /// surface-colour cross-fade over this distance (uniform width regardless of the local climate
    /// gradient — see [`biome::surface_biome_world`]). Larger = softer, wider biome transitions.
    /// Changing it re-bakes (the texel blend changes).
    pub(crate) biome_blend_m: f32,
    /// SURFACE-TREATMENT master strength `[0,1]` for the top (undug) layer (snow/sand/rock overrides): 0 =
    /// pure strata surface colour, 1 = full treatment. A LIVE shader uniform — no re-bake on change.
    pub(crate) surface_treatment: f32,
    /// Enable the SEPARATE coarse physics-collider clipmap layer (`physics_resident_chunks`): a self-contained
    /// set of collider-only chunks around the player/camera, baked + streamed INDEPENDENTLY of the render mesh
    /// so the colliders persist while the render mesh re-bakes underneath (no fall-through during streaming).
    pub(crate) physics: bool,
    /// FINEST physics LOD — the collider layer's "+N" coarseness vs render LOD 0 (default 2 ⇒ +2 LOD, ~16×
    /// fewer collider triangles). A character controller doesn't need render fidelity, so the colliders are a
    /// couple of LODs coarser than the near render mesh.
    pub(crate) physics_base_lod: u32,
    /// How many physics LODs the collider clipmap spans from `physics_base_lod` (default 2 ⇒ a small 2-ring
    /// clipmap: finer near the player, one coarser ring out). Each ring doubles the world radius (2:1).
    pub(crate) physics_lod_count: u32,
    /// Physics clipmap radius in `physics_base_lod` chunks (half-extent of the finest physics cube around the
    /// focus). Small — colliders only need to reach a bit past the player; the set stays tiny so it rarely
    /// re-streams. Higher = colliders reach further (more chunks to keep baked).
    pub(crate) physics_half_chunks: i32,
    /// DEBUG: render each baked physics-collider chunk as a green wireframe (the COARSE collider mesh + the
    /// physics clipmap's reach). Toggling rebuilds the small physics set (the wireframe mesh is attached at
    /// bake time).
    pub(crate) physics_wireframe: bool,
}

/// The clipmap's finest node spacing (the tier-0 height grid). A terrain-only chunk whose voxel size is at
/// or below this already carries the full mip-0 relief in its geometry, so it gets NO baked detail map (the
/// LOD gate); only COARSER chunks (`voxel_size > DETAIL_NORMAL_FINEST_SPACING`) do. SSOT for the gate +
/// the one-time gate log. Mirrors `HEIGHT_BAND_LIMIT_TAP` (= `HEIGHT_CHUNK_CELLS / HEIGHT_FIELD_RES` = 2 m).
const DETAIL_NORMAL_FINEST_SPACING: f32 = 2.0;

impl Default for MeshBakeConfig {
    fn default() -> Self {
        // K=4 → 64 bricks/chunk. lod0_radius 32 keeps the finest LOD out to a comfortable distance (push
        // it down to shrink the LOD-0 cube); lod_count 12 spans LOD 0..=11 (far worldgen horizon — the
        // height-clipmap window auto-grows to match). Cross-LOD seams are crack-free BY CONSTRUCTION
        // (Transvoxel transition cells) — no toggle needed.
        Self {
            chunk_bricks: 4,
            lod0_radius: 32.0,
            lod_count: 12,
            debug_lod_colour: false,
            debug_normals: false,
            freeze_lod: false,
            // 256×256 per-chunk detail map: the slope source is the RAW `sample_world` analytic gradient
            // (ONE eval/texel, no 2 m convolution). At 256 a far LOD-8 chunk (~2.9 km footprint) resolves
            // ~11 m/texel; cost is N² (~4× the map portion of the bake vs 128, 256 KB Rg16Float per chunk).
            // Tune via the editor slider (down for cheaper, up to 512 for finer) when iterating on the look.
            detail_normal_res: 256,
            detail_normal_strength: 1.0,
            // 128×128 biome + surface-material map per chunk. The surface-material pair (mat_a, mat_b) is
            // NEAREST-sampled per texel (ids can't interpolate + the textures are per-fragment), so the
            // material BOUNDARIES step at the texel grid — 128 keeps the steps fine (~the detail scale) while
            // the textures mask the rest. Higher = smoother boundaries but more resolve_surface calls/chunk.
            biome_res: 128,
            // 150 m biome-border cross-fade: the baked blend is WORLD-normalised (gradient-divided), so
            // every biome border fades over ~150 m regardless of how steep the local climate gradient is —
            // no more hard lines where the climate happens to change quickly. Tune via the editor slider.
            biome_blend_m: 150.0,
            surface_treatment: 1.0,
            // Separate coarse physics clipmap: colliders at +3 LOD (physics_base_lod 3), a 3-ring clipmap of
            // radius 8 base-chunks around the player/camera. Coarse ⇒ rarely re-streams, persists across render
            // re-bakes. Bump physics_half_chunks/lod_count to collide further out.
            physics: true,
            physics_base_lod: 3,
            physics_lod_count: 3,
            physics_half_chunks: 8,
            physics_wireframe: false,
        }
    }
}

/// Hand-picked distinct unlit debug tints for the first LODs of the "Colour by LOD" view; LODs beyond
/// this are tinted procedurally (golden-ratio hue) by [`lod_debug_tint`], so the view stays distinct for
/// any `lod_count` up to [`MAX_MESH_LODS`].
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

/// Unlit debug tint for a LOD level ("Colour by LOD"). Uses the hand-picked [`LOD_DEBUG_PALETTE`] for the
/// first levels, then a golden-ratio hue for any higher LOD (well-separated colours for any count) so the
/// debug view matches the dynamic `lod_count` instead of clamping every coarse LOD to one colour.
fn lod_debug_tint(lod: u32) -> [f32; 3] {
    if (lod as usize) < LOD_DEBUG_PALETTE.len() {
        return LOD_DEBUG_PALETTE[lod as usize];
    }
    // HSV → RGB at a golden-ratio-spaced hue, fixed saturation/value.
    let h = (lod as f32 * 0.618_034).fract() * 6.0;
    let c = 0.7_f32;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = 0.9 - c;
    [r + m, g + m, b + m]
}

/// Set by the editor panel's "Rebake all" button to force a full re-mesh. Also pulsed by
/// `worldgen::roll_worldgen` when the height ring regenerates without the Terrain volume moving (a
/// param edit / streaming delta in fixed mode): the Terrain content hash is unchanged by a ring swap,
/// so the mesh-bake needs an explicit nudge to re-mesh the affected chunks.
#[derive(Resource, Default)]
pub(crate) struct MeshBakeRebuild(pub bool);

/// Live diagnostics for the editor panel.
#[derive(Resource, Default)]
pub(crate) struct MeshBakeStats {
    /// Number of SDF volumes (edits) gathered this frame.
    edits: usize,
    /// Resident chunks the edits currently occupy.
    resident: usize,
    /// Resident chunks not yet displaying their current target (in-flight, staged, or not-yet-started) —
    /// the honest "mesh bake still working" signal for the editor status bar. 0 ⇒ everything is baked.
    pub(crate) pending: usize,
    /// Chunk-mesh entities despawned by the most recent COMMIT.
    reaped: usize,
    /// Resident chunk count per LOD level (index = lod), for the panel readout.
    resident_by_lod: [usize; MAX_MESH_LODS as usize],
    /// Set by the panel's "Capture diagnostics" button; consumed by the system, which fills `dump`.
    capture: bool,
    /// Copy-paste-able diagnostic dump — filled when `capture` is requested.
    dump: String,
    // --- SEPARATE PHYSICS-COLLIDER CLIPMAP (`physics_resident_chunks`) diagnostics ---
    /// Resident collider chunks this frame (`0` ⇒ physics off / not in player mode).
    pub(crate) physics_resident: usize,
    /// Collider entities currently live (baked + spawned).
    pub(crate) physics_displayed: usize,
    /// Collider bakes in flight on the dedicated physics task pool (the honest "physics still baking" signal).
    pub(crate) physics_baking: usize,
}

/// Mesh-bake plugin. Added in `main.rs`. The bake itself is editor- AND scene-INDEPENDENT (it runs
/// every frame and bakes SDF world edits in gameplay too); only the optional debug panel is editor-only.
pub struct MeshBakePlugin;

impl Plugin for MeshBakePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(super::mesh_material::MeshMaterialPlugin)
            .add_plugins(super::terrain_material::TerrainMaterialPlugin)
            .init_resource::<ChunkStates>()
            .init_resource::<PhysicsChunkStates>()
            .init_resource::<PhysicsBakePool>()
            .init_resource::<PhysicsWireMat>()
            .init_resource::<MeshBakeConfig>()
            .init_resource::<MeshBakeRebuild>()
            .init_resource::<MeshBakeStats>()
            // Editor- AND scene-INDEPENDENT: runs every frame so SDF world edits are baked during
            // gameplay too. It self-determines which chunks to mesh from the SDF edits (no dependency
            // on the editor-scene-gated GPU SDF atlas) and no-ops when no SDF volumes exist — which
            // also clears the meshes when an SDF scene is left.
            .add_systems(Update, sync_terrain_detail_params)
            .add_systems(
                Update,
                mesh_resident_chunks.after(super::mesh_material::rebuild_mesh_material),
            )
            // Reveal terrain chunks once their per-chunk material is GPU-live (keep-old-until-revealed swap).
            .add_systems(Update, reveal_ready_chunks.after(mesh_resident_chunks))
            // The separate coarse physics-collider clipmap, streamed independently of the render mesh.
            .add_systems(Update, physics_resident_chunks);
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

/// World centre of a chunk. (Test-only since the reap switched to all-8-octant coverage sampling.)
#[cfg(test)]
fn chunk_centre_world(key: BrickKey, config: &SdfGridConfig, k: u32) -> Vec3 {
    let b = chunk_aabb(key, config, k);
    (Vec3::from(b.min) + Vec3::from(b.max)) * 0.5
}

/// The LOD-`lod` chunk whose region CONTAINS world point `p` (the chunk that tiles `p` at that LOD). Chunk
/// index `j = floor(p / chunk_world_size)`, key coord `= j · (K·cell_stride)` — the inverse of
/// [`chunk_aabb`]/[`chunk_lod0_range`]. Used to ask "does a committed chunk tile this region?" for
/// keep-old-until-covered.
fn chunk_key_at(p: Vec3, lod: u32, config: &SdfGridConfig, k: u32) -> BrickKey {
    let cw = k as f32 * config.brick_world_size(lod);
    let stride = k as i32 * config.cell_stride();
    let j = (p / cw).floor().as_ivec3();
    BrickKey::new(lod, j * stride)
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
    clipmap: Option<&crate::sdf_render::worldgen::upload::HeightClipmap>,
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
    let Some(clipmap) = clipmap else {
        return false; // touches terrain but nothing loaded → not generatable yet.
    };
    let margin = config.voxel_size_at(key.lod)
        + 2.0 * crate::sdf_render::worldgen::layers::height::HEIGHT_CHUNK_CELLS as f32;
    let m = Vec2::splat(margin);
    // SOME tier must fully cover the footprint. A km-wide far-LOD chunk is admitted once its COARSE
    // tier is resident (coarser tiers cover larger footprints) → the distance fills in to the full
    // mesh-bake reach. The clipmap sampler picks the finest covering tier per voxel, so a chunk this
    // gate admits can never miss the strict `eval_primitive` Terrain sampler.
    crate::sdf_render::worldgen::upload::clipmap_covers_aabb(clipmap, cmin - m, cmax + m)
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

/// The mesh-bake clipmap's COARSEST-LOD outer reach (world metres from the focus): the half-extent of
/// the LOD-`(lod_count-1)` shell cube, mirroring the `shell_cube` formula in the residency loop —
/// `half = (half0 << lod) · cw0`, where `cw0 = k · brick_world_size(0)`, `half0 = lod0_half_chunks`,
/// `lod = lod_count - 1`. SSOT for the worldgen height-clipmap tier count: the coarsest height tier must
/// cover at least this reach so terrain extends to the full mesh-bake extent. Pure function of the two
/// configs (uses the SAME `k` clamp the bake uses), so it auto-tracks any default `lod_count` change.
pub(crate) fn coarsest_lod_outer_reach(config: &SdfGridConfig, mesh_cfg: &MeshBakeConfig) -> f32 {
    let k = mesh_cfg.chunk_bricks.clamp(1, 8);
    let cw0 = k as f32 * config.brick_world_size(0);
    let half0 = lod0_half_chunks(config, mesh_cfg, k);
    let lod = effective_lod_count(config, mesh_cfg, true).saturating_sub(1); // coarsest LOD index
    (half0 << lod) as f32 * cw0
}

/// Is chunk `key`'s CENTRE inside the outer-LOD clipmap cube centred on `cam` (half-extent
/// `(half0 << (lod_count-1)) · cw0`)? The SSOT for "is this region within a clipmap's coverage". Two uses,
/// which together make keep-old-until-covered hold BY CONSTRUCTION:
/// - against the LIVE camera → `gone` (outside ⇒ the chunk left the world; safe to drop its mesh + work);
/// - against the COMMITTED round's `cam`/`half0` → "does the just-committed clipmap cover this region?" A
///   displayed mesh is reaped at a commit ONLY when the committed clipmap covers it (its replacement is now
///   displayed) OR it's `gone` — so an old mesh is never despawned before a covering chunk exists.
///
/// `None` camera ⇒ single-LOD fallback ⇒ always "covered" (no clipmap to leave).
fn chunk_in_outer_cube(
    key: BrickKey,
    cam: Option<Vec3>,
    half0: i32,
    lod_count: u32,
    config: &SdfGridConfig,
    k: u32,
) -> bool {
    let Some(c) = cam else {
        return true;
    };
    let cw0 = k as f32 * config.brick_world_size(0);
    let outer = lod_count.saturating_sub(1);
    let centre = lod_centre(config, k, c, outer).as_vec3() * cw0;
    let half = ((half0 << outer) as f32) * cw0;
    let b = chunk_aabb(key, config, k);
    let cc = (Vec3::from(b.min) + Vec3::from(b.max)) * 0.5;
    (cc - centre).abs().cmple(Vec3::splat(half)).all()
}

/// KEEP-OLD-UNTIL-COVERED commit-reap decision — the single SSOT used by BOTH the render reap loop and its
/// unit tests, so the invariant holds by construction. At a round COMMIT, a displayed chunk mesh is despawned
/// (`true`) iff ALL of:
/// - it is NOT re-committed by this round (`!in_committed_round` — a re-baked chunk's own commit swaps it);
/// - it is NOT `gone` (still inside the LIVE clipmap; a gone chunk is despawned by the per-frame self-evict
///   pass, never here — so we don't double-despawn);
/// - this chunk's WHOLE region is TILED by REVEALED resident chunks (see [`region_covered_by_revealed`]), which
///   handles an ARBITRARY LOD gap between the old mesh and its replacements — essential when fast movement puts
///   SEVERAL LODs between where you were (the old chunk) and where you are (its many fine replacements).
///
/// The region test (not a coarse outer-cube test) is also what makes this correct under fast flight: when the
/// height-coverage gate drops a region (the height clipmap lagging behind a fast camera), that region is
/// ABSENT from `round_resident`, so the old mesh there is KEPT until the height loads and a committed chunk
/// finally tiles it. This is the structural guarantee that a displayed mesh is never despawned before a
/// covering, on-screen replacement exists.
#[allow(clippy::too_many_arguments)] // clipmap geometry context; threading it is clearer than a wrapper struct
fn render_commit_reap(
    key: BrickKey,
    in_committed_round: bool,
    round_resident: &HashSet<BrickKey>,
    is_revealed: &impl Fn(BrickKey) -> bool,
    live_cam: Option<Vec3>,
    live_half0: i32,
    lod_count: u32,
    config: &SdfGridConfig,
    k: u32,
) -> bool {
    if in_committed_round {
        return false;
    }
    if !chunk_in_outer_cube(key, live_cam, live_half0, lod_count, config, k) {
        return false; // gone → despawned by the self-evict pass, not here
    }
    region_covered_by_revealed(key, round_resident, is_revealed, config, k, lod_count)
}

/// Is `key`'s WHOLE world region tiled by REVEALED resident chunks (its replacement is fully on screen)? The
/// SSOT keep-old coverage test, correct for an ARBITRARY LOD gap. The resident chunk owning the region's CENTRE
/// either COVERS the whole region (coarser-or-equal in the nested clipmap lattice → one `is_revealed` check) or
/// is FINER (the region is split among finer chunks → RECURSE into the 8 child sub-chunks and require all
/// covered). A point with NO resident owner anywhere (a coverage-gate gap, or outside the clipmap) makes it
/// NOT covered → the old mesh is kept. The recursion descends only where coverage is actually finer (it returns
/// immediately at a coarse owner), so it's cheap for the common adjacent (2:1) swap and bounded by the LOD gap
/// for fast multi-LOD jumps. Replaces an 8-octant SAMPLE that silently under-counted coverage across >1 LOD and
/// punched holes when the camera outran the bakes by several rings.
fn region_covered_by_revealed(
    key: BrickKey,
    resident: &HashSet<BrickKey>,
    is_revealed: &impl Fn(BrickKey) -> bool,
    config: &SdfGridConfig,
    k: u32,
    lod_count: u32,
) -> bool {
    let b = chunk_aabb(key, config, k);
    let min = Vec3::from(b.min);
    let centre = (min + Vec3::from(b.max)) * 0.5;
    // The resident chunk owning the centre — the finest resident LOD present there.
    let owner = (0..lod_count).find_map(|lod| {
        let o = chunk_key_at(centre, lod, config, k);
        resident.contains(&o).then_some(o)
    });
    match owner {
        None => false,                                  // no resident chunk here → gap → not covered
        Some(o) if o.lod >= key.lod => is_revealed(o),  // a coarser-or-equal owner covers the whole region
        Some(_) => {
            // Finer coverage → split into the 8 child sub-chunks (key.lod − 1) and require ALL covered.
            if key.lod == 0 {
                return is_revealed(chunk_key_at(centre, 0, config, k)); // (unreachable: lod-0 owner is ≥ lod 0)
            }
            let child_lod = key.lod - 1;
            let cw_child = k as f32 * config.brick_world_size(child_lod);
            [0.0_f32, 1.0].iter().all(|&ox| {
                [0.0_f32, 1.0].iter().all(|&oy| {
                    [0.0_f32, 1.0].iter().all(|&oz| {
                        let p = min + Vec3::new(ox, oy, oz) * cw_child + Vec3::splat(cw_child * 0.5);
                        let child = chunk_key_at(p, child_lod, config, k);
                        region_covered_by_revealed(child, resident, is_revealed, config, k, lod_count)
                    })
                })
            })
        }
    }
}

/// Does a resident chunk need a NEW bake issued this frame? The SSOT for the REQUEST decision (shared by the
/// production loop and the streaming simulation test). True iff neither the chunk's DISPLAYED nor its HIDDEN
/// geometry matches the `desired` hash AND nothing is already baking.
///
/// CRITICAL — `has_task` short-circuits to `false`: we NEVER cancel an in-flight bake whose target was
/// superseded. Under continuous movement the desired hash (its transition flags) changes most frames; a
/// cancel-and-restart would drop the bake EVERY frame so it would never finish and the chunk would only appear
/// once the camera stopped ("takes ages to show up"). Instead the in-flight bake completes + commits its
/// slightly-stale result (the chunk appears promptly), then a follow-up bake re-issues toward the live target.
fn chunk_needs_bake(displayed_hash: u64, hidden_hash: u64, has_task: bool, desired: u64) -> bool {
    !has_task && displayed_hash != desired && hidden_hash != desired
}

/// LOD-0 cube half-extent in LOD-0 chunks — rounded to an EVEN number so the finer cube (half this) stays
/// chunk-aligned at the next LOD too (clean partition; see `mesh_chunk_in_shell`).
fn lod0_half_chunks(config: &SdfGridConfig, mesh_cfg: &MeshBakeConfig, k: u32) -> i32 {
    let cw0 = k as f32 * config.brick_world_size(0);
    let h = (mesh_cfg.lod0_radius / cw0).round().max(2.0) as i32;
    (h + 1) & !1 // next even, ≥ 2
}

/// Effective LOD count: the live `mesh_cfg.lod_count` clamped to `[1, MAX_MESH_LODS]` (the mesh path's
/// LODs are independent of the GPU atlas `lod_count` — `voxel_size_at(lod)` is just `·2^lod`), or 1 with
/// no camera. This is the SSOT the worldgen height-clipmap tier count tracks, so the loaded sample-area
/// window always matches the configured LOD reach.
fn effective_lod_count(_config: &SdfGridConfig, mesh_cfg: &MeshBakeConfig, has_cam: bool) -> u32 {
    if !has_cam {
        1
    } else {
        mesh_cfg.lod_count.clamp(1, MAX_MESH_LODS)
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

/// The voxel size to use for a Terrain HEIGHT sample at `p` so a coarse chunk's surface MORPHS smoothly from
/// its own (coarse) height mip in the interior to its FINER neighbour's mip at a TRANSITION face — GEOMORPH,
/// the structural cure for the cross-LOD "mip-step" kink. A face borders a finer LOD when its `flags` bit is
/// set (bit order = the `SIDES`/`TransitionSide` order LowX,HighX,LowY,HighY,LowZ,HighZ).
///
/// Instead of a hard switch (interior `vs` / on-face `vs·0.5`), the effective voxel size RAMPS over a band one
/// coarse voxel deep (`band = vs`): let `d` be the sample's MINIMUM inward distance from any set transition
/// face, `w = smoothstep(clamp(d/band, 0, 1))` (a portable cubic `3t²−2t³`, no transcendentals), and
/// `vs_eff = vs·0.5 + (vs − vs·0.5)·w`. At a face (`d=0 ⇒ w=0`) this is `vs·0.5` EXACTLY — so the coarse
/// transition vertices still bit-match the finer neighbour (the continuous-mip sampler picks the SAME mip on
/// both sides) and the cross-LOD weld stays watertight. At/beyond the band (`d≥band ⇒ w=1`) it is the coarse
/// `vs` (interior). In between, the fractional voxel size feeds `continuous_height_mip`, so the sampled height
/// mip slides continuously and the coarse surface morphs into the fine surface across the band instead of
/// stepping. The SAME function feeds the density field AND the builder normals, so geometry + shading morph
/// together. Only the Terrain eval reads `vs` (for its band-limited mip select), so this is a no-op for object
/// chunks. `flags == 0` ⇒ `vs` (interior of a uniform-LOD region — common fast path).
fn transition_sample_vs(p: Vec3, cmin: Vec3, cmax: Vec3, vs: f32, flags: u8) -> f32 {
    if flags == 0 {
        return vs; // no transition faces (interior of a uniform-LOD region) — common fast path
    }
    // Inward distance from each SET transition face plane; take the MIN (nearest face governs the ramp).
    let mut d = f32::INFINITY;
    if flags & 0b00_0001 != 0 {
        d = d.min(p.x - cmin.x); // LowX
    }
    if flags & 0b00_0010 != 0 {
        d = d.min(cmax.x - p.x); // HighX
    }
    if flags & 0b00_0100 != 0 {
        d = d.min(p.y - cmin.y); // LowY
    }
    if flags & 0b00_1000 != 0 {
        d = d.min(cmax.y - p.y); // HighY
    }
    if flags & 0b01_0000 != 0 {
        d = d.min(p.z - cmin.z); // LowZ
    }
    if flags & 0b10_0000 != 0 {
        d = d.min(cmax.z - p.z); // HighZ
    }
    let band = vs; // one coarse voxel deep
    let t = (d / band).clamp(0.0, 1.0);
    let w = t * t * (3.0 - 2.0 * t); // smoothstep — C1 at both ends (zero slope ⇒ no kink re-introduced)
    let fine = vs * 0.5;
    fine + (vs - fine) * w
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
    terrain: Option<Arc<HeightClipmap>>,
    // Take the surface NORMAL from the clipmap's smooth stored gradient (no central-difference faceting at
    // coarse LODs / cross-LOD borders). TRUE only for PURE (undug) terrain; a carved chunk uses CSG normals
    // (the clipmap normal is wrong on the dug cavity walls), and mixed/object chunks always do.
    terrain_normals: bool,
    // Route this chunk through the terrain-surface material (volumetric strata): TRUE for terrain, including
    // DUG terrain (so the cavity walls show the strata). A superset of `terrain_normals`.
    surface_material: bool,
    // DETAIL-NORMAL bake resolution (`N`): a COARSE terrain-only chunk additionally bakes an `N×N`
    // surface-slope map (gated below). 0 disables the detail bake (height/biome still bake at `detail_res`/
    // a floor; see `bake_terrain_surface`).
    detail_res: u32,
    // BIOME map resolution (`N`): the per-chunk low-res biome (primary/secondary/blend) map.
    biome_res: u32,
    // BIOME-border blend half-width in WORLD metres (the baked colour cross-fade width).
    biome_blend_m: f32,
) -> Option<ChunkMeshData> {
    // Install THIS ROUND'S frozen Terrain clipmap snapshot ONCE on the bake thread (held for the whole
    // bake), so every field sample reads it via a thread-local borrow instead of a process-global RwLock +
    // Arc-clone (the per-sample lock/atomic, contended across the async pool, was the dominant bake cost).
    // Crucially the snapshot is the LIVE clipmap whose coverage gate ADMITTED this chunk this frame, so a
    // clipmap that changes mid-bake (camera roll / `lod_count` slider rebuild) can't make this bake
    // sample uncovered ground and trip the strict sampler. `None` (object-only scene) ⇒ falls back to the
    // global (the eval then panics only on a genuine rendering miss, which the gate still prevents).
    let _bake_terrain = crate::sdf_render::worldgen::upload::set_bake_terrain(
        terrain,
        crate::sdf_render::worldgen::upload::cpu_terrain_offset(),
    );
    // Transvoxel treats density > threshold as INSIDE; our CSG distance is NEGATIVE inside → negate it. The
    // tiny iso-shift keeps no sample landing EXACTLY on 0 (density > 0 is strict, so a 0 sample reads
    // "outside" — a pinhole at grid-aligned features like a sphere pole on a grid corner). Samples ON a
    // TRANSITION face (bordering a finer LOD) use the FINER neighbour's voxel size for the Terrain height
    // mip (`transition_sample_vs`), so the coarse transition vertices match the fine neighbour bit-for-bit
    // → watertight cross-LOD (no tiny height seam).
    let cmin = grid_origin;
    let cmax = grid_origin + Vec3::splat(subdivisions as f32 * vs);
    let field = |x: f32, y: f32, z: f32| {
        let p = Vec3::new(x, y, z);
        let vs_eff = transition_sample_vs(p, cmin, cmax, vs, flags);
        1e-3 - edits::fold_csg_dist_indexed(edits, indices, p, vs_eff)
    };
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
    let builder =
        ChunkMeshBuilder::new(edits, indices, grid_origin, vs, lod, debug, cmin, cmax, flags, terrain_normals);
    // MUST be CacheNothing: `CacheCentralBlockOnly` caches the central block at THIS chunk's (coarse)
    // resolution, which then serves the transition cell's FINE-resolution face samples too — collapsing the
    // transition so the cross-LOD weld fails. The analytic CSG field is cheap to re-evaluate, so just query it.
    let builder = extract_from_field(&field, FieldCaching::CacheNothing, block, sides, 0.0, builder);
    let mut data = builder.finish()?;
    // TERRAIN-SURFACE bake (terrain-only chunks): over the chunk's world-XZ footprint, sample the PRISTINE
    // surface height (depth reference) + the fine surface slope (detail normal, coarse-gated) + the biome
    // (low-res Whittaker classification). The per-bake hi-fi snapshot is the SAME terrain the clipmap was
    // built from. Attached to the mesh data; the commit turns it into the chunk's `TerrainMaterial`.
    data.terrain_surface = bake_terrain_surface(
        grid_origin, subdivisions as f32 * vs, vs, surface_material, detail_res, biome_res, biome_blend_m,
    );
    Some(data)
}

/// One-shot latch so the detail-normal LOD-gate log line prints only on the first gated chunk (never silent).
static DETAIL_GATE_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Floor resolution for the surface-height grid when the detail-normal bake is disabled (`detail_res < 2`)
/// but a terrain-only chunk still needs a height (depth-reference) map. Height is smooth, so a small grid
/// suffices for the volumetric strata depth lookup even without detail normals.
const TERRAIN_HEIGHT_FALLBACK_RES: u32 = 32;

/// Bake a TERRAIN-ONLY chunk's per-chunk surface maps (Stages 2+3). Over the chunk's world-XZ footprint
/// (`[chunk_min, chunk_min + chunk_world]`), at TEXEL CENTRES (`(i + 0.5)·step`, matching the shader's
/// lookup):
/// - **surface height** `h(x,z)` (`R32Float`, `detail_grid²`): the PRISTINE `sample_world` height — the
///   depth reference (`depth = surf_h − world.y`). Baked on EVERY terrain-only chunk.
/// - **detail normal** `(dh/dx, dh/dz)` (`Rg16Float`, same grid): the fine band-limited slope. GATED to
///   COARSE chunks — fine chunks (`vs ≤ DETAIL_NORMAL_FINEST_SPACING`) already carry mip-0 relief in their
///   geometry, so their detail map is ZERO-FILLED (all-zero slope ⇒ the shader uses the geometry normal),
///   bounding the `N²` gradient cost to coarse chunks while every terrain chunk still gets strata/height.
///   (Both come from ONE `sample_world` eval/texel via [`TerrainHifi::surface`], so the height is free.)
/// - **biome** `(primary, secondary, blend)` (`Rgba16Float`, `biome_res²`): the Stage-1 Whittaker
///   classifier per texel (low-res — biome is km-scale).
///
/// Returns `None` (the chunk keeps the shared mesh material) when it is not terrain-only or no hi-fi terrain
/// source is installed. Uses the per-bake [`bake_terrain_hifi`] snapshot (no global lock) + its `world_seed`
/// for the biome classification (a RENDER attribute — NOT keyed by `HEIGHT_GEN_VERSION`).
fn bake_terrain_surface(
    grid_origin: Vec3,
    chunk_world: f32,
    vs: f32,
    surface_material: bool,
    detail_res: u32,
    biome_res: u32,
    biome_blend_m: f32,
) -> Option<super::terrain_material::TerrainSurfaceBake> {
    use super::terrain_material::TerrainSurfaceBake;
    if !surface_material {
        return None;
    }
    let (hifi, offset) = crate::sdf_render::worldgen::upload::bake_terrain_hifi()?;
    // The depth-reference height comes from the SAME clipmap the mesh geometry is built from — NOT the finer
    // `sample_world` — so `depth = surf_h − mesh.y ≈ 0` on undug terrain (else the sub-voxel detail the coarse
    // mesh dropped makes `depth` cross the thin surface stratum → speckled dirt/stone). The detail-normal
    // SLOPE still uses the fine hi-fi gradient below.
    let (clip, _) = crate::sdf_render::worldgen::upload::bake_terrain_clipmap()?;
    // The compiled biome library snapshot — the bake resolves the SURFACE MATERIAL per texel from it
    // (biome base + altitude caps + cliffs + patches). `None` until `biomes.ron` compiles (a lib change
    // triggers a rebake), in which case the surface-material map zero-fills (palette is empty too).
    let lib = crate::sdf_render::worldgen::upload::cpu_biome_library();

    // DETAIL-NORMAL LOD GATE: only coarse chunks bake real slopes; fine chunks zero-fill (geometry normal).
    // Logged ONCE so the cap is visible, never silent.
    let detail_enabled = detail_res >= 2 && vs > DETAIL_NORMAL_FINEST_SPACING;
    if !detail_enabled && vs <= DETAIL_NORMAL_FINEST_SPACING
        && !DETAIL_GATE_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed)
    {
        bevy::log::info!(
            "worldgen terrain-surface: per-chunk DETAIL-NORMAL maps GATED to coarse terrain LODs \
             (voxel_size > finest node spacing {DETAIL_NORMAL_FINEST_SPACING} m); fine chunks already carry \
             full mip-0 relief in their geometry. Biome strata + surface height STILL bake on every terrain \
             chunk (the strata render everywhere)."
        );
    }

    // The detail/height grid: `detail_res` when the detail bake is on, else a small height-only floor.
    let n = if detail_res >= 2 { detail_res } else { TERRAIN_HEIGHT_FALLBACK_RES };
    let chunk_min = Vec2::new(grid_origin.x, grid_origin.z);
    let step = (chunk_world / n as f32) as f64;
    let (ox, oz) = ((chunk_min.x + offset.x) as f64, (chunk_min.y + offset.y) as f64);

    let mut detail_texels = Vec::with_capacity((n * n * 4) as usize);
    let mut height_texels = Vec::with_capacity((n * n * 4) as usize);
    for j in 0..n {
        let wz = oz + (j as f64 + 0.5) * step;
        for i in 0..n {
            let wx = ox + (i as f64 + 0.5) * step;
            // Slope (detail normal) = the FINE hi-fi gradient; height (depth reference) = the CLIPMAP height
            // the mesh is built from (mesh-matching ⇒ depth ≈ 0 on undug terrain — fixes the strata mottle).
            let (h_fine, dhdx, dhdz) = hifi.surface(wx, wz);
            let h = crate::sdf_render::worldgen::upload::try_sample_clipmap_lod(
                &clip,
                bevy::math::DVec2::new(wx, wz),
                vs,
            )
            .map_or(h_fine, |node| node.height);
            height_texels.extend_from_slice(&h.to_le_bytes());
            if detail_enabled {
                detail_texels.extend_from_slice(&TerrainSurfaceBake::pack_slope(dhdx, dhdz));
            } else {
                detail_texels.extend_from_slice(&TerrainSurfaceBake::pack_slope(0.0, 0.0));
            }
        }
    }

    // BIOME map: the Whittaker classifier (CPU SSOT) per texel over the SAME footprint, low-res. Uses the
    // hi-fi snapshot's world seed so the in-world biome placement matches the preview's.
    let bn = biome_res.max(2);
    let bstep = (chunk_world / bn as f32) as f64;
    let mut biome_texels = Vec::with_capacity((bn * bn * 8) as usize);
    let mut surface_mat_texels = Vec::with_capacity((bn * bn * 8) as usize);
    for j in 0..bn {
        let wz = oz + (j as f64 + 0.5) * bstep;
        for i in 0..bn {
            let wx = ox + (i as f64 + 0.5) * bstep;
            let s = crate::sdf_render::worldgen::biome::surface_biome_world(
                wx,
                wz,
                hifi.world_seed,
                biome_blend_m as f64,
            );
            let temp = crate::sdf_render::worldgen::biome::temperature(wx, wz, hifi.world_seed) as f32;
            biome_texels.extend_from_slice(&TerrainSurfaceBake::pack_biome(
                s.primary as u8,
                s.secondary as u8,
                s.blend,
                temp,
            ));
            // SURFACE MATERIAL (undug top): the worldgen resolves it from the library — biome base + altitude
            // caps + cliffs + patches — using this texel's surface altitude + slope. The shader just renders
            // the baked (mat_a, mat_b, weight). No library yet ⇒ zero-fill (the palette is empty too).
            let sm = match lib.as_deref() {
                Some(lib) => {
                    let (h, dhdx, dhdz) = hifi.surface(wx, wz);
                    // cos of the surface slope = N.y for N = normalize(-dh/dx, 1, -dh/dz).
                    let n_y = 1.0 / (1.0 + (dhdx * dhdx + dhdz * dhdz) as f64).sqrt();
                    crate::sdf_render::worldgen::biome::resolve_surface(wx, wz, h as f64, n_y, s, hifi.world_seed, lib)
                }
                None => crate::sdf_render::worldgen::biome::SurfaceBlend { mat_a: 0, mat_b: 0, weight: 0.0 },
            };
            surface_mat_texels.extend_from_slice(&TerrainSurfaceBake::pack_surface(sm.mat_a, sm.mat_b, sm.weight));
        }
    }

    Some(TerrainSurfaceBake {
        detail_present: detail_enabled,
        detail_res: n,
        biome_res: bn,
        chunk_min,
        chunk_size: chunk_world,
        detail_texels,
        height_texels,
        biome_texels,
        surface_mat_texels,
    })
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
    /// Chunk world min/max corner + transition-face flags — so a boundary vertex ON a face bordering a finer
    /// LOD samples its normal + material at the FINER neighbour's voxel size (`transition_sample_vs`), matching
    /// the density closure so position, normal AND material all agree across the cross-LOD seam.
    cmin: Vec3,
    cmax: Vec3,
    flags: u8,
    /// Pure (undug) terrain ⇒ analytic stored-gradient normals (smooth, no central-diff faceting). A carved
    /// or mixed chunk is `false` (CSG normals — correct on dug cavity walls / object surfaces).
    terrain_normals: bool,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    /// Per unique vertex: `(nearest, runner-up)` CSG material ids (the top-2 argmin). The triangle pair folds
    /// from the three corners' values; `runner-up == nearest` when only one material is present at the vertex.
    vmat: Vec<(u16, u16)>,
    tris: Vec<u32>,
}

impl<'a> ChunkMeshBuilder<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        edits: &'a [edits::ResolvedEdit],
        indices: &'a [u32],
        origin: Vec3,
        vs: f32,
        lod: u32,
        debug: bool,
        cmin: Vec3,
        cmax: Vec3,
        flags: u8,
        terrain_normals: bool,
    ) -> Self {
        Self {
            edits,
            indices,
            origin,
            eps: vs * 0.01,
            vs,
            lod,
            debug,
            cmin,
            cmax,
            flags,
            terrain_normals,
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
        let tint = lod_debug_tint(self.lod);

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
        Some(ChunkMeshData { positions, normals, uvs, colors, indices, terrain_surface: None })
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
        // A boundary vertex ON a transition face samples its height at the FINER neighbour's voxel size (same
        // rule as the density closure), so its normal + material match the fine neighbour bit-for-bit — no
        // shading/material seam across the cross-LOD weld. Interior vertices use the chunk's own `vs`.
        let vs_eff = transition_sample_vs(world, self.cmin, self.cmax, self.vs, self.flags);
        // Outward normal. For a terrain-only chunk, take it from the clipmap's SMOOTH stored gradient (no
        // central-difference faceting at coarse LODs / cross-LOD borders) — falling back to the CSG gradient
        // on a clipmap miss. Mixed/object chunks use the exact ∇(CSG distance) (toward increasing distance).
        let csg_normal =
            || field_gradient(self.edits, self.indices, world, vs_eff * 0.01, vs_eff).normalize_or_zero();
        let n = if self.terrain_normals {
            crate::sdf_render::worldgen::upload::terrain_normal(world, vs_eff).unwrap_or_else(csg_normal)
        } else {
            csg_normal()
        };
        self.normals.push([n.x, n.y, n.z]);
        // (nearest, runner-up) materials at this vertex over the blend-padded chunk set; `finish` folds the
        // per-triangle pair from the three corners' values.
        let (near, runner, _) = edits::fold_csg_top2(self.edits, self.indices, world, vs_eff);
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
    // Force-keep on a sign change across the chunk corners — the ROBUST test: if any pair of corners
    // straddles the surface the chunk certainly crosses it, regardless of how badly the field's distance
    // is estimated. Covers BOTH a smoothed surface (smoothing inflates the gradient) AND a steep TERRAIN
    // surface (the eroded `p.y−h` field, even Lipschitz-normalised, can over/under-estimate enough that
    // the single centre test below would false-drop a steep chunk → holes). 8 cheap evals; can only ever
    // KEEP, never drop — so it can't punch a hole. The centre test below is the cheap early-out for the
    // interior of large solids (all corners same sign).
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
    // circumradius (½·√3·side) + apron/iso-shift slack + smoothing inflation margin.
    let reach = cw * 0.866_025_4 + 2.0 * vs + 0.5 * smooth_sum;
    edits::fold_csg_dist_indexed(edits, indices, center, vs).abs() <= reach
}

/// Push the LIVE shader-uniform config (`detail_normal_strength`, `debug_normals`, `surface_treatment`) AND
/// the hot-reloadable strata table into EVERY per-chunk `TerrainMaterial` whenever the mesh-bake config OR
/// the biome library changes — so the strength/treatment sliders + "View normals" debug + a `biomes.ron`
/// edit are LIVE (no re-bake). Materials are baked with a snapshot of these at spawn; this keeps them in
/// sync. Runs only on a change; for the scalar uniforms it touches only materials whose value differs
/// (cheap, no churn at rest). When the library changes, the strata table is re-flattened ONCE (the shared
/// SSOT flatten) and pushed to all terrain materials.
fn sync_terrain_detail_params(
    cfg: Res<MeshBakeConfig>,
    biome_lib: Res<super::worldgen::biome::BiomeLibrary>,
    mut mats: ResMut<Assets<super::terrain_material::TerrainMaterial>>,
    mut rebuild: ResMut<MeshBakeRebuild>,
) {
    let cfg_changed = cfg.is_changed();
    let lib_changed = biome_lib.is_changed();
    if !cfg_changed && !lib_changed {
        return;
    }
    let (strength, debug, treatment) =
        (cfg.detail_normal_strength, cfg.debug_normals as u32, cfg.surface_treatment);
    // Re-flatten the strata table + material palette only when the library changed (the shared SSOT flattens).
    let (strata, palette) = if lib_changed {
        (
            Some(super::worldgen::biome::StrataTableStd::from_library(&biome_lib)),
            Some(super::worldgen::biome::MaterialPaletteStd::from_library(&biome_lib)),
        )
    } else {
        (None, None)
    };
    if lib_changed {
        // The bake resolves SURFACE-MATERIAL ids from the library, so a `biomes.ron` change must re-bake (the
        // baked `surface_mat` ids can't be live-patched like the colour tables). Publish the snapshot the
        // off-thread bake reads, then request a rebuild. (The strata/palette tables ARE patched live below so
        // dug-wall colours update without waiting for the rebake.)
        crate::sdf_render::worldgen::upload::set_cpu_biome_library(Some(std::sync::Arc::new(biome_lib.clone())));
        rebuild.0 = true;
    }
    // Touch a material if any live scalar differs OR the library changed (the tables must be re-pushed).
    let ids: Vec<_> = mats
        .iter()
        .filter(|(_, m)| {
            lib_changed
                || m.extension.params.strength != strength
                || m.extension.params.flags.x != debug
                || m.extension.params.surf_b.z != treatment
        })
        .map(|(id, _)| id)
        .collect();
    for id in ids {
        if let Some(m) = mats.get_mut(id) {
            m.extension.params.strength = strength;
            m.extension.params.flags.x = debug;
            m.extension.params.surf_b.z = treatment;
            if let Some(table) = strata {
                m.extension.strata = table;
            }
            if let Some(p) = palette {
                m.extension.palette = p;
            }
        }
    }
}

// =====================================================================================================
// SEPARATE PHYSICS-COLLIDER CLIPMAP LAYER
// =====================================================================================================
//
// Colliders live in their OWN coarse clipmap (`physics_resident_chunks`), decoupled from the render-mesh
// bake. A collider-only entity (`PhysicsChunk`) is a static Rapier `trimesh` baked from the SAME CSG+terrain
// field as the render mesh, but at a COARSE LOD (`physics_base_lod`, default +2) and over a SMALL clipmap
// around the player (else camera). Because the set is small + coarse it changes slowly → it rarely
// re-streams, so colliders PERSIST while the render mesh churns under LOD streaming (no fall-through). No
// cross-LOD weld is needed — a capsule controller tolerates the tiny gap/overlap at a coarse chunk border.

/// A baked physics-collider chunk in the separate coarse physics clipmap (collider-only — no render mesh /
/// material unless `physics_wireframe` attaches a debug mesh). Stamped with its chunk key. Distinct from the
/// render `ChunkMesh`.
#[derive(Component)]
struct PhysicsChunk(BrickKey);

/// A coarse physics bake's output: chunk-LOCAL collider positions + the flat triangle index list (`None` =
/// empty chunk, no surface crossing).
type PhysicsBakeResult = Option<(Vec<[f32; 3]>, Vec<u32>)>;

/// Per-physics-chunk bake state. Far simpler than the render `ChunkState` — no rounds, no atomic swap, no
/// transition flags (colliders don't weld across LODs). One in-flight bake + one displayed collider entity.
#[derive(Default)]
struct PhysicsChunkState {
    /// The displayed collider entity (`None` = empty chunk / not yet baked).
    entity: Option<Entity>,
    /// In-flight coarse bake (returns the collider geometry: chunk-LOCAL positions + flat index list).
    task: Option<Task<PhysicsBakeResult>>,
    /// Content hash the displayed collider was baked from; `target_hash` is what the in-flight task targets.
    displayed_hash: u64,
    target_hash: u64,
}

#[derive(Resource, Default)]
struct PhysicsChunkStates(HashMap<BrickKey, PhysicsChunkState>);

/// DEDICATED task pool for collider bakes — separate from the global `AsyncComputeTaskPool` the RENDER mesh
/// floods (up to `MAX_NEW_TASKS_PER_FRAME` chunks/frame). On the shared pool the few physics bakes queue
/// BEHIND the entire render backlog (hundreds of chunks) and the player's colliders lag badly; their own pool
/// runs them on independent threads so they're never blocked by render — collider bakes get effective top
/// priority. A couple of threads is plenty for the small + coarse physics set.
#[derive(Resource)]
struct PhysicsBakePool(bevy::tasks::TaskPool);

impl FromWorld for PhysicsBakePool {
    fn from_world(_world: &mut World) -> Self {
        let threads = (std::thread::available_parallelism().map_or(4, |n| n.get()) / 4).max(1);
        Self(
            bevy::tasks::TaskPoolBuilder::new()
                .num_threads(threads)
                .thread_name("physics-bake".to_string())
                .build(),
        )
    }
}

/// Shared unlit material for the `physics_wireframe` debug mesh (green). Created once.
#[derive(Resource)]
struct PhysicsWireMat(Handle<StandardMaterial>);

impl FromWorld for PhysicsWireMat {
    fn from_world(world: &mut World) -> Self {
        let mut mats = world.resource_mut::<Assets<StandardMaterial>>();
        Self(mats.add(StandardMaterial {
            base_color: Color::srgb(0.1, 1.0, 0.35),
            unlit: true,
            ..default()
        }))
    }
}

/// Per-frame budget for new physics-collider bakes (the set is small + coarse, so a low cap still fills
/// quickly while keeping the spawn cost off the frame's critical path).
/// Per-frame collider-bake spawn cap. Higher than before now that physics has its OWN [`PhysicsBakePool`] (it
/// no longer competes with the render backlog), so colliders fill in fast under the moving player.
const PHYSICS_MAX_NEW_TASKS_PER_FRAME: u32 = 24;

/// Is a chunk resident in the SEPARATE physics clipmap? Mirrors [`mesh_chunk_in_shell`] but with `base` as
/// the FINEST level (the base cube fills SOLID — no hole), so the physics layer is a self-contained coarse
/// clipmap of LODs `[base, base+count)`, half-extent `base_half` LOD-0 chunks at the finest level, doubling
/// per coarser ring (2:1).
fn physics_chunk_in_shell(
    key: BrickKey,
    config: &SdfGridConfig,
    k: u32,
    center: Vec3,
    base: u32,
    count: u32,
    base_half: i32,
) -> bool {
    if key.lod < base || key.lod >= base + count {
        return false;
    }
    let lvl = key.lod - base;
    let (lo, hi) = chunk_lod0_range(key, config, k);
    let outer = base_half << lvl; // cube(L) half in LOD-0 chunks
    if !range_in_cube(lo, hi, lod_centre(config, k, center, key.lod), outer) {
        return false;
    }
    if key.lod == base {
        return true; // finest physics LOD fills its cube (no finer ring to hollow it out)
    }
    let hole = base_half << (lvl - 1); // covered by the finer physics ring
    !range_in_cube(lo, hi, lod_centre(config, k, center, key.lod - 1), hole)
}

/// The separate coarse physics-collider clipmap: residency → async coarse trimesh bake → collider-only
/// entities, streamed on its own (slow) lifecycle independent of the render mesh. Tears everything down when
/// `physics` is off, no edits exist, or there is no focus (player/camera).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn physics_resident_chunks(
    mut commands: Commands,
    volumes: Query<VolumeQueryData, With<SdfVolume>>,
    config: Res<SdfGridConfig>,
    mesh_cfg: Res<MeshBakeConfig>,
    mode: Res<super::editor_camera::SdfCameraMode>,
    players: Query<&GlobalTransform, (With<super::player::WorldgenPlayer>, Without<SdfVolume>)>,
    phys_chunks: Query<(Entity, &PhysicsChunk)>,
    mut states: ResMut<PhysicsChunkStates>,
    mut meshes: ResMut<Assets<Mesh>>,
    wire_mat: Res<PhysicsWireMat>,
    bake_pool: Res<PhysicsBakePool>,
    mut stats: ResMut<MeshBakeStats>,
) {
    // Default the physics diagnostics to "off"; the live counts are written at the end when we have a focus.
    stats.physics_resident = 0;
    stats.physics_displayed = 0;
    stats.physics_baking = 0;
    let teardown = |commands: &mut Commands, states: &mut PhysicsChunkStates, q: &Query<(Entity, &PhysicsChunk)>| {
        if !states.0.is_empty() || q.iter().next().is_some() {
            for (e, _) in q.iter() {
                commands.entity(e).despawn();
            }
            states.0.clear();
        }
    };

    let gathered = gather_sorted_edits(&volumes);
    if !mesh_cfg.physics || gathered.is_empty() {
        teardown(&mut commands, &mut states, &phys_chunks);
        return;
    }

    // FOCUS = the PLAYER. Colliders are baked ONLY in player mode (when the capsule exists) — never under the
    // free fly camera, which has nothing to collide with: baking coarse colliders there is pure wasted work
    // (and, with the wireframe debug on, shows as stray green meshes leading the textured terrain). In fly mode
    // the layer tears down; switching to player re-bakes it (coarse + small, so it fills near-instantly).
    let center = mode
        .player
        .then(|| players.iter().next().map(|t| t.translation()))
        .flatten();
    let Some(center) = center else {
        teardown(&mut commands, &mut states, &phys_chunks);
        return;
    };

    // A `physics_wireframe` toggle is folded into the content hash below, so it RE-BAKES each collider (the new
    // entity attaches/omits the debug mesh) and the keep-old-until-covered evict holds the old collider until
    // the re-baked one commits — the wireframe toggles in place with NO collider teardown (no fall-through).

    let k = mesh_cfg.chunk_bricks.clamp(1, 8);
    let cs = config.cell_stride() as u32;
    let base = mesh_cfg.physics_base_lod.min(MAX_MESH_LODS - 1);
    let count = mesh_cfg.physics_lod_count.clamp(1, MAX_MESH_LODS - base);
    let base_half = mesh_cfg.physics_half_chunks.max(1) << base; // finest-level half-extent, in LOD-0 chunks

    // Edit AABBs (padded by the material blend reach, as the render path) + terrain coverage-gate inputs.
    let n_edits = gathered.len();
    let mut edit_aabbs: Vec<Aabb3d> = Vec::with_capacity(n_edits);
    let mut edit_extent: Vec<f32> = Vec::with_capacity(n_edits);
    let mut edit_vec: Vec<edits::ResolvedEdit> = Vec::with_capacity(n_edits);
    for g in &gathered {
        edit_extent.push((Vec3::from(g.aabb.max) - Vec3::from(g.aabb.min)).max_element());
        let pad = bevy::math::Vec3A::splat(BLEND_REACH);
        edit_aabbs.push(Aabb3d { min: g.aabb.min - pad, max: g.aabb.max + pad });
        edit_vec.push(g.edit.clone());
    }
    let edits_arc = Arc::new(edit_vec);
    let terrain_xz_aabbs: Vec<(Vec2, Vec2)> = gathered
        .iter()
        .filter(|g| matches!(g.edit.prim, edits::SdfPrimitive::Terrain { .. }))
        .map(|g| (Vec2::new(g.aabb.min.x, g.aabb.min.z), Vec2::new(g.aabb.max.x, g.aabb.max.z)))
        .collect();
    let height_clipmap = crate::sdf_render::worldgen::upload::cpu_height_clipmap();

    // RESIDENCY: per physics LOD, the chunks inside that ring's cube (clipped to the edit AABBs so we only
    // bake where there's geometry) that pass the shell test + the terrain coverage gate.
    let cw0 = k as f32 * config.brick_world_size(0);
    let mut resident: HashSet<BrickKey> = HashSet::new();
    let mut cand: HashSet<BrickKey> = HashSet::new();
    for lvl in 0..count {
        let lod = base + lvl;
        cand.clear();
        let centre = lod_centre(&config, k, center, lod).as_vec3() * cw0;
        let half = ((base_half << lvl) as f32) * cw0;
        let (smin, smax) = (centre - Vec3::splat(half), centre + Vec3::splat(half));
        for a in &edit_aabbs {
            let mn = Vec3::from(a.min).max(smin);
            let mx = Vec3::from(a.max).min(smax);
            if mn.cmpgt(mx).any() {
                continue;
            }
            chunks_in_aabb(&Aabb3d::from_min_max(mn, mx), &config, k, lod, &mut cand);
        }
        for &key in &cand {
            if !physics_chunk_in_shell(key, &config, k, center, base, count, base_half) {
                continue;
            }
            if !terrain_xz_aabbs.is_empty()
                && !terrain_chunk_covered(key, &config, k, &terrain_xz_aabbs, height_clipmap.as_deref())
            {
                continue;
            }
            resident.insert(key);
        }
    }

    // The padded sampled AABB of a chunk (cell span + 1-voxel apron) — same as the render path.
    let chunk_sampled = |key: BrickKey| -> Aabb3d {
        let b = chunk_aabb(key, &config, k);
        let apron = Vec3::splat(config.voxel_size_at(key.lod));
        Aabb3d::from_min_max(Vec3::from(b.min) - apron, Vec3::from(b.max) + apron)
    };

    // Per-chunk content hash over the edits that touch it (sub-voxel-culled the same way as the fold). No
    // LOD-transition flags (physics has none) — so a chunk re-bakes only when an edit it samples changes.
    let mut current: HashMap<BrickKey, u64> = HashMap::with_capacity(resident.len());
    {
        // Fold the wireframe flag into the hash so toggling it re-bakes every collider (the new entity
        // attaches/omits the debug mesh) → an in-place toggle via the normal keep-old swap, no teardown.
        let wire_mix = if mesh_cfg.physics_wireframe { 0xC0DE_F00D_BA5E_DEAD_u64 } else { 0 };
        let mut idx: Vec<u32> = Vec::new();
        for &key in &resident {
            cull_into(&edit_aabbs, &chunk_sampled(key), &mut idx);
            idx.retain(|&i| edit_resolvable_at(edit_extent[i as usize], &config, key.lod));
            let h = if idx.is_empty() { 0 } else { edits::bake_content_hash(&edits_arc, &idx) };
            current.insert(key, ((key.lod as u64).wrapping_mul(0xA24B_AED4_963E_E407) ^ h) ^ wire_mix);
        }
    }

    // EVICT (keep-old-until-covered — a collider is NEVER despawned before its replacement exists, so the
    // player can't fall through during streaming / a physics-LOD transition). A collider whose key left the
    // residency is kept until EITHER (a) a currently-DISPLAYED resident collider covers its centre (the
    // replacement ground is in place), OR (b) it lies outside the live physics clipmap's outer cube (the
    // focus flew away — truly gone). `displayed_resident` = resident chunks that already have a baked entity,
    // collected first so the eviction predicate borrows no state. The query is a frame-start snapshot and
    // RECEIVE below only spawns NEW (deferred) entities, so nothing is double-despawned.
    let phys_outer_lod = base + count - 1;
    let phys_outer_half = (base_half << (count - 1)) as f32 * cw0;
    let phys_outer_centre = lod_centre(&config, k, center, phys_outer_lod).as_vec3() * cw0;
    let chunk_centre = |key: BrickKey| -> Vec3 {
        let b = chunk_aabb(key, &config, k);
        (Vec3::from(b.min) + Vec3::from(b.max)) * 0.5
    };
    let phys_gone = |key: BrickKey| -> bool {
        !(chunk_centre(key) - phys_outer_centre).abs().cmple(Vec3::splat(phys_outer_half)).all()
    };
    let displayed_resident: Vec<Aabb3d> = resident
        .iter()
        .filter(|r| states.0.get(r).is_some_and(|s| s.entity.is_some()))
        .map(|r| chunk_aabb(*r, &config, k))
        .collect();
    let covered = |key: BrickKey| -> bool {
        let cc = chunk_centre(key);
        displayed_resident
            .iter()
            .any(|a| Vec3::from(a.min).cmple(cc).all() && cc.cmple(Vec3::from(a.max)).all())
    };
    for (e, pc) in &phys_chunks {
        if !resident.contains(&pc.0) && (covered(pc.0) || phys_gone(pc.0)) {
            commands.entity(e).despawn();
        }
    }
    states.0.retain(|key, st| {
        resident.contains(key)
            || (st.entity.is_some() && !covered(*key) && !phys_gone(*key))
    });

    // RECEIVE: poll bakes; on completion replace the chunk's collider entity (despawn old → spawn new).
    let wireframe = mesh_cfg.physics_wireframe;
    for (key, st) in states.0.iter_mut() {
        let Some(task) = st.task.as_mut() else {
            continue;
        };
        let Some(result) = block_on(poll_once(&mut *task)) else {
            continue;
        };
        st.task = None;
        st.displayed_hash = st.target_hash;
        if let Some(old) = st.entity.take() {
            commands.entity(old).despawn();
        }
        let Some((pos, ind)) = result else {
            continue; // empty chunk → no collider
        };
        let Some(collider) = chunk_trimesh_collider(&pos, &ind) else {
            continue;
        };
        let origin = config.brick_min_world(key.coord, key.lod);
        let mut ent = commands.spawn((
            bevy_rapier3d::prelude::RigidBody::Fixed,
            collider,
            Transform::from_translation(origin),
            PhysicsChunk(*key),
            Name::new("Physics Collider Chunk"),
        ));
        if wireframe {
            // Debug: render the coarse collider mesh as a green wireframe (chunk-LOCAL, like the collider).
            let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos)
                .with_inserted_indices(Indices::U32(ind));
            // `compute_flat_normals` panics on INDEXED geometry; the coarse collider mesh is indexed and this is
            // only a debug wireframe, so smooth normals (index-compatible) are fine.
            mesh.compute_smooth_normals();
            ent.insert((
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(wire_mat.0.clone()),
                bevy::pbr::wireframe::Wireframe,
                bevy::pbr::wireframe::WireframeColor { color: Color::srgb(0.1, 1.0, 0.35) },
            ));
        }
        st.entity = Some(ent.id());
    }

    // EVICT: a chunk no longer resident is dropped immediately (the set is small + coarse, and the focus is
    // always well inside it, so eviction only ever clips the far edge — no hole under the player). Dropping
    // the state cancels any in-flight bake.
    states.0.retain(|key, st| {
        if resident.contains(key) {
            true
        } else {
            if let Some(e) = st.entity.take() {
                commands.entity(e).despawn();
            }
            false
        }
    });

    // REQUEST: bake stale resident chunks (nearest-to-focus first), budget-limited. Spawn on the DEDICATED
    // physics pool so the collider bakes are never queued behind the render mesh's backlog (top priority for
    // the player's ground). Install the live clipmap snapshot on this thread for the SYNCHRONOUS surface cull.
    let pool = &bake_pool.0;
    let mut budget = PHYSICS_MAX_NEW_TASKS_PER_FRAME;
    let _phys_terrain = crate::sdf_render::worldgen::upload::set_bake_terrain(
        height_clipmap.clone(),
        crate::sdf_render::worldgen::upload::cpu_terrain_offset(),
    );
    let mut pending: Vec<BrickKey> = resident
        .iter()
        .copied()
        .filter(|key| match states.0.get(key) {
            Some(st) => st.task.is_none() && st.displayed_hash != current[key],
            None => true,
        })
        .collect();
    // Finest physics LOD first (the collider directly under the player), then nearest within a LOD by distance
    // to the chunk's NEAREST point (AABB) — an explicit tuple key (was a fragile bit-or of lod and dist-bits).
    pending.sort_unstable_by_key(|&key| {
        let b = chunk_aabb(key, &config, k);
        let d2 = center.distance_squared(center.clamp(Vec3::from(b.min), Vec3::from(b.max)));
        (key.lod, d2.to_bits())
    });
    let mut idx: Vec<u32> = Vec::new();
    for key in pending {
        if budget == 0 {
            break;
        }
        let st = states.0.entry(key).or_default();
        let target = current[&key];
        st.target_hash = target;
        let vs_l = config.voxel_size_at(key.lod);
        cull_into(&edit_aabbs, &chunk_sampled(key), &mut idx);
        idx.retain(|&i| edit_resolvable_at(edit_extent[i as usize], &config, key.lod));
        if !chunk_has_surface(&edits_arc, &idx, &config, k, key, vs_l) {
            // Empty chunk → no collider; settle it (drop any old entity) without spending budget.
            st.displayed_hash = target;
            if let Some(e) = st.entity.take() {
                commands.entity(e).despawn();
            }
            continue;
        }
        let grid_origin = config.brick_min_world(key.coord, key.lod);
        let edits = edits_arc.clone();
        let indices = idx.clone();
        let terrain = height_clipmap.clone();
        let lod = key.lod;
        let sub = k * cs;
        st.task = Some(pool.spawn(async move {
            // Plain coarse bake (no transition faces, no materials/normals/detail) — take just the geometry.
            let d = mesh_chunk(
                &edits, &indices, grid_origin, vs_l, sub, 0, lod, false, terrain, false, false, 0, 0, 0.0,
            )?;
            Some((d.positions, d.indices))
        }));
        budget -= 1;
    }

    // Diagnostics: resident collider chunks, live collider entities, and in-flight collider bakes (was not
    // recorded anywhere). `physics_baking == 0` ⇒ all colliders are on the ground.
    stats.physics_resident = resident.len();
    stats.physics_displayed = states.0.values().filter(|s| s.entity.is_some()).count();
    stats.physics_baking = states.0.values().filter(|s| s.task.is_some()).count();
}

/// The main-thread asset stores + config the commit needs to spawn a chunk mesh AND (for a coarse
/// terrain-only chunk) its per-chunk DETAIL-NORMAL `Image` + `TerrainMaterial`. Bundled so `spawn_chunk_mesh`
/// stays under the arg cap and both commit sites pass the same set.
struct SpawnAssets<'a> {
    mesh_assets: &'a mut Assets<Mesh>,
    images: &'a mut Assets<Image>,
    terrain_mats: &'a mut Assets<super::terrain_material::TerrainMaterial>,
    mesh_mats: &'a super::mesh_material::MeshMaterials,
    /// Live detail-normal strength (shader uniform) + debug-normals flag, from `MeshBakeConfig`.
    detail_strength: f32,
    debug_normals: bool,
    /// The shared per-biome strata table (flattened from the live `BiomeLibrary`), baked into each
    /// terrain-only chunk's `TerrainMaterial`. Hot-reload of `biomes.ron` re-syncs it (see
    /// [`sync_terrain_detail_params`]).
    strata: super::worldgen::biome::StrataTableStd,
    /// The shared material palette (colour + roughness, flattened from the live `BiomeLibrary`) the baked
    /// `surface_mat` ids index. Re-synced live on a `biomes.ron` edit, same as `strata`.
    palette: super::worldgen::biome::MaterialPaletteStd,
    /// The shared terrain PBR texture arrays (diffuse, normal, MRA) — the current handles to bake into a new
    /// chunk's material. `sync_terrain_texture_arrays` keeps already-spawned chunks current.
    tex_arrays: (Handle<Image>, Handle<Image>, Handle<Image>),
}

/// Build a static Rapier `trimesh` collider from a chunk's baked geometry (chunk-LOCAL positions + the flat
/// `u32` index list grouped into triangles). `None` for a degenerate chunk (< 1 triangle) so the caller
/// simply skips the collider. The collider matches the rendered surface (incl. dug/CSG geometry).
fn chunk_trimesh_collider(positions: &[[f32; 3]], indices: &[u32]) -> Option<bevy_rapier3d::prelude::Collider> {
    if positions.len() < 3 || indices.len() < 3 {
        return None;
    }
    let verts: Vec<Vec3> = positions.iter().map(|p| Vec3::from_array(*p)).collect();
    let tris: Vec<[u32; 3]> = indices.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    bevy_rapier3d::prelude::Collider::trimesh(verts, tris).ok()
}

/// Spawn one chunk-mesh entity from baked data — the single SSOT [`commit_baked`] uses for every committed
/// chunk. Transvoxel positions are chunk-LOCAL relative to the chunk's world MIN corner (no
/// apron), so the entity `Transform` is exactly `brick_min_world`; one entity per chunk.
///
/// A TERRAIN-ONLY chunk that baked a surface payload (`data.terrain_surface = Some`) is spawned with a
/// DEDICATED per-chunk `TerrainMaterial` (volumetric biome strata + per-fragment depth + PBR). The per-chunk
/// `Image`s + material handles are parked on the entity via [`TerrainDetailAssets`] + `MeshMaterial3d` so
/// they're FREED when the entity despawns (the same ref-counted lifecycle as the mesh). Every OTHER chunk
/// (mixed/object/CSG-cave) keeps the single shared triplanar `MeshMaterial`.
///
/// Returns `(entity, hidden)`. A TERRAIN chunk (per-chunk `TerrainMaterial` textures) is spawned
/// `Visibility::Hidden` + [`PendingReveal`] so it stays invisible until its per-chunk `Image`s are GPU-live
/// (`hidden = true` — the caller stages it in `pending_entities` and the OLD mesh is held until reveal). An
/// OBJECT/mixed chunk uses the shared material (no per-chunk textures, renderable immediately) → spawned
/// visible (`hidden = false`).
fn spawn_chunk_mesh(
    commands: &mut Commands,
    assets: &mut SpawnAssets,
    config: &SdfGridConfig,
    key: BrickKey,
    data: ChunkMeshData,
) -> (Entity, bool) {
    use super::terrain_material::{self, TerrainDetailAssets};
    let origin = config.brick_min_world(key.coord, key.lod);
    let surface = data.terrain_surface;
    let is_terrain = surface.is_some();
    // NOTE: physics colliders are NOT attached here anymore — they live in the SEPARATE coarse physics clipmap
    // layer (`physics_resident_chunks`), decoupled from the render-mesh lifecycle so they persist across
    // render re-bakes (no fall-through during LOD streaming).
    let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, data.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, data.uvs)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, data.colors)
        .with_inserted_indices(Indices::U32(data.indices));
    let mut ent = commands.spawn((
        Mesh3d(assets.mesh_assets.add(mesh)),
        Transform::from_translation(origin),
        ChunkMesh(key),
        Name::new("SDF Chunk Mesh"),
    ));
    if let Some(bake) = surface {
        // Terrain-only chunk: dedicated per-chunk terrain-surface material (volumetric biome strata + PBR).
        // Strong handles to the 3 per-chunk images AND the material live on the entity (in `MeshMaterial3d` +
        // `TerrainDetailAssets`) → all are freed when this entity despawns on evict/rebuild (no leak).
        let detail_normal = assets.images.add(terrain_material::make_detail_image(&bake));
        let surface_height = assets.images.add(terrain_material::make_height_image(&bake));
        let biome = assets.images.add(terrain_material::make_biome_image(&bake));
        let surface_mat = assets.images.add(terrain_material::make_surface_mat_image(&bake));
        let mat = assets.terrain_mats.add(terrain_material::make_terrain_material(
            detail_normal.clone(),
            surface_height.clone(),
            biome.clone(),
            surface_mat.clone(),
            &bake,
            assets.detail_strength,
            assets.debug_normals,
            assets.strata,
            assets.palette,
            assets.tex_arrays.clone(),
        ));
        ent.insert((
            MeshMaterial3d(mat.clone()),
            TerrainDetailAssets { material: mat, detail_normal, surface_height, biome, surface_mat },
            // LOCKSTEP: hidden until its per-chunk textures are GPU-live (see `reveal_ready_chunks`).
            Visibility::Hidden,
            PendingReveal { frames_left: REVEAL_SETTLE_FRAMES },
        ));
    } else {
        ent.insert(MeshMaterial3d(assets.mesh_mats.handle.clone()));
    }
    (ent.id(), is_terrain)
}

/// Commit a chunk's freshly-baked result (target hash `task_hash`), keep-old-until-REVEALED. Three cases:
/// - EMPTY (`data` None, no-surface / meshed-empty): despawn old `displayed` + stale `hidden`, clear both,
///   `displayed_hash = hidden_hash = task_hash`, `revealed = true` (the chunk's current result is "nothing").
/// - TERRAIN (`spawn_chunk_mesh` returns `hidden = true`: per-chunk material textures not yet GPU-live):
///   despawn any stale `hidden`, push the new entity to `hidden`, `hidden_hash = task_hash`. KEEP `displayed`
///   visible — [`reveal_ready_chunks`] swaps it in when the textures settle. Do NOT set `revealed`.
/// - OBJECT/mixed (`hidden = false`: shared material, renderable now): despawn old `displayed` + stale
///   `hidden`, push the new entity to `displayed`, `displayed_hash = task_hash`, `revealed = true`.
fn commit_baked(
    commands: &mut Commands,
    assets: &mut SpawnAssets,
    config: &SdfGridConfig,
    key: BrickKey,
    data: Option<ChunkMeshData>,
    task_hash: u64,
    st: &mut ChunkState,
) {
    let Some(data) = data else {
        // Meshed-empty / no surface: nothing to display; drop old displayed + any staged hidden.
        for e in st.displayed.drain(..).chain(st.hidden.drain(..)) {
            commands.entity(e).despawn();
        }
        st.displayed_hash = task_hash;
        st.hidden_hash = task_hash;
        st.revealed = true;
        return;
    };
    let (e, hidden) = spawn_chunk_mesh(commands, assets, config, key, data);
    if hidden {
        // Terrain: keep `displayed` visible; stage the new mesh hidden (reveal swaps it when GPU-live).
        for stale in st.hidden.drain(..) {
            commands.entity(stale).despawn(); // a superseded not-yet-revealed replacement
        }
        st.hidden.push(e);
        st.hidden_hash = task_hash;
    } else {
        // Object/mixed (shared material, renderable now): swap immediately.
        for old in st.displayed.drain(..).chain(st.hidden.drain(..)) {
            commands.entity(old).despawn();
        }
        st.displayed.push(e);
        st.displayed_hash = task_hash;
        st.revealed = true;
    }
}

/// Fade terrain chunks in lockstep with their material, in TWO phases over the [`PendingReveal`] window.
/// PHASE 1 (after `REVEAL_SHOW_AFTER` frames, per-chunk textures GPU-live): make the new mesh `Visible` while
/// the OLD geometry it replaces stays on screen — a brief overlap, never a hole. PHASE 2 (at `frames_left == 0`,
/// once the new mesh has had frames to actually render): SWAP — despawn the OLD `displayed`, promote the
/// now-revealed `hidden` to live, mark the chunk `revealed` (so the keep-old-until-revealed reap may now drop
/// anything IT covers). The phase-1→phase-2 gap guarantees the new mesh is rendering before the old leaves, so a
/// GPU-upload lagging the counter can't open a hole. Runs after `mesh_resident_chunks`. Only terrain chunks
/// carry `PendingReveal`; object/mixed chunks are visible + revealed at spawn.
fn reveal_ready_chunks(
    mut commands: Commands,
    mut pending: Query<(Entity, &ChunkMesh, &mut Visibility, &mut PendingReveal)>,
    mut states: ResMut<ChunkStates>,
) {
    for (e, cm, mut vis, mut pr) in &mut pending {
        if pr.frames_left > 0 {
            pr.frames_left -= 1;
        }
        // Phase 1 — show the new mesh once its textures are live, BUT keep the old one (overlap).
        if pr.frames_left <= REVEAL_SETTLE_FRAMES - REVEAL_SHOW_AFTER {
            *vis = Visibility::Visible;
        }
        // Phase 2 — only once the full window has elapsed (the new mesh has had frames to actually render) do
        // we drop the old geometry.
        if !chunk_renderable(pr.frames_left) {
            continue;
        }
        commands.entity(e).remove::<PendingReveal>();
        if let Some(st) = states.0.get_mut(&cm.0) {
            // Swap in lockstep: despawn the OLD displayed mesh, promote this revealed entity to live.
            for old in st.displayed.drain(..) {
                if old != e {
                    commands.entity(old).despawn();
                }
            }
            st.hidden.retain(|&p| p != e);
            for stale in st.hidden.drain(..) {
                commands.entity(stale).despawn(); // a superseded replacement that slipped through
            }
            st.displayed.push(e);
            st.displayed_hash = st.hidden_hash;
            st.revealed = true;
        }
    }
}

/// True iff the chunk's world AABB is ENTIRELY outside the frustum (behind some plane) — for bake
/// PRIORITY only (in-view bakes before off-screen), never correctness. `planes[i] = (normal, d)` with
/// inside = `normal·p + d ≥ 0` (Bevy's `HalfSpace`). Tests the AABB's farthest-positive corner per plane.
fn aabb_outside_frustum(planes: &[Vec4; 6], min: Vec3, max: Vec3) -> bool {
    planes.iter().any(|p| {
        let n = p.truncate();
        let far = Vec3::new(
            if n.x >= 0.0 { max.x } else { min.x },
            if n.y >= 0.0 { max.y } else { min.y },
            if n.z >= 0.0 { max.z } else { min.z },
        );
        n.dot(far) + p.w < 0.0 // fully behind this plane ⇒ outside the frustum
    })
}

/// The finest LOD levels that ALWAYS bake first, in every direction, BEFORE the frustum split — so there
/// is always nearby baked terrain all around the camera (incl. behind it), not just in view.
const ALWAYS_NEAR_LOD_MAX: u32 = 1;

/// Bake-scheduling priority key for a stale chunk (LOWER = baked first). Order:
/// (1) the always-near rings (LOD ≤ [`ALWAYS_NEAR_LOD_MAX`]) bake first OMNIDIRECTIONALLY — nearby terrain
///     exists in every direction regardless of view;
/// (2) then IN-VIEW before off-screen — the entire visible set before any off-screen chunk (frustum);
/// (3) within a bucket, LOD ascending — finest/nearest ring first, building outward from the camera;
/// (4) nearest first within a LOD (distance²).
/// A `None` frustum degrades to near-then-LOD-then-distance. Packed
/// `near_rank<<37 | frustum_rank<<36 | lod<<32 | dist²bits` (dist² ≥ 0 ⇒ its f32 bits sort monotonically).
fn bake_priority(key: BrickKey, config: &SdfGridConfig, k: u32, cam: Vec3, frustum: Option<&[Vec4; 6]>) -> u64 {
    let b = chunk_aabb(key, config, k);
    let min = Vec3::from(b.min);
    let max = Vec3::from(b.max);
    // Distance to the chunk's NEAREST point (AABB), not its centre — so a large coarse chunk whose near face
    // is under the camera sorts as "near", not "far" (centre-distance over-penalised big LOD chunks). 0 inside.
    let d2 = cam.distance_squared(cam.clamp(min, max));
    let near = key.lod <= ALWAYS_NEAR_LOD_MAX;
    let near_rank: u64 = if near { 0 } else { 1 };
    // The near rings are view-independent (frustum_rank forced 0) so they never split by view; only the
    // coarser rings are frustum-ordered (in-view first).
    let frustum_rank: u64 = if near {
        0
    } else {
        match frustum {
            Some(planes) if aabb_outside_frustum(planes, min, max) => 1,
            _ => 0,
        }
    };
    (near_rank << 37) | (frustum_rank << 36) | ((key.lod as u64) << 32) | (d2.to_bits() as u64)
}

/// Content-hash-driven, async Transvoxel bake (see the module doc). The unit is a configurable
/// `K×K×K`-brick chunk. Each chunk is an INDEPENDENT, hash-driven state machine: it always bakes toward
/// its LIVE desired content hash (`current_hashes[key]`), and an in-flight bake whose target is superseded
/// is dropped + re-issued. There is NO frozen bake round — so a chunk can never get stuck waiting for a
/// round that never re-forms (the post-fill streaming stall is impossible by construction).
///
/// TODO(watertight): optional atomic band-reveal on shell snap. Without a frozen-round atomic band swap,
/// when the LOD shell snaps the affected band re-bakes + reveals per-chunk; keep-old-until-revealed +
/// near-simultaneous reveal makes cross-LOD cracks brief/rare but not provably zero. Acceptable for v1.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn mesh_resident_chunks(
    mut commands: Commands,
    volumes: Query<VolumeQueryData, With<SdfVolume>>,
    config: Res<SdfGridConfig>,
    mesh_cfg: Res<MeshBakeConfig>,
    // Drives the clipmap LOD (finer near the camera). No `SdfCamera` ⇒ single-LOD fallback (mesh
    // everything at LOD 0 — the original scene/camera-independent behaviour for gameplay scenes). The
    // optional `Frustum` drives BAKE PRIORITY (in-view first); absent ⇒ LOD-then-distance ordering.
    cameras: Query<(&GlobalTransform, Option<&Frustum>), (With<SdfCamera>, Without<SdfVolume>)>,
    chunk_meshes: Query<(Entity, &ChunkMesh)>,
    mut states: ResMut<ChunkStates>,
    mut rebuild: ResMut<MeshBakeRebuild>,
    mut stats: ResMut<MeshBakeStats>,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    // Per-chunk DETAIL-NORMAL assets for coarse terrain-only chunks: each gets its own `Rg16Float` `Image`
    // + `TerrainMaterial`, freed when the chunk entity despawns (handles parked on the entity).
    mut images: ResMut<Assets<Image>>,
    mut terrain_mats: ResMut<Assets<super::terrain_material::TerrainMaterial>>,
    // The single shared triplanar `MeshMaterial` handle (built by `mesh_material::rebuild_mesh_material`);
    // EVERY chunk mesh uses it — the per-vertex ids + blend weight select/cross-fade materials in-shader.
    mesh_mats: Res<super::mesh_material::MeshMaterials>,
    // The live biome library → flattened into the shared strata GPU table baked into each terrain-only
    // chunk's `TerrainMaterial` (Stage 3). Hot-reload re-syncs existing materials (`sync_terrain_detail_params`).
    // The live biome library + shared terrain texture arrays, bundled into one SystemParam (the system is at
    // Bevy's param-arity limit, so they share a slot). `.lib` flattens into the strata/palette GPU tables;
    // `.tex` supplies the texture-array handles baked into each new terrain chunk's material (Stage 5).
    terrain_mat: TerrainMatRes,
    // Bundled scalar Locals: rebake epoch, prev K.
    mut scal: Local<MeshBakeScalars>,
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
        scal.prev_k = k;
        return;
    }

    // K changed live (slider): the key set is at a different stride now, so every old-stride chunk mesh
    // is stale. Despawn all + clear state for a clean swap.
    if scal.prev_k != 0 && scal.prev_k != k {
        for (e, _) in &chunk_meshes {
            commands.entity(e).despawn();
        }
        states.0.clear();
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
    // Per-edit "is this the Terrain primitive" (indexed like `edits_arc`) — a chunk whose candidate edits
    // are ALL terrain is "terrain-only": independent streamed surface with no atomic-edit grouping, so it
    // commits the instant it bakes (no round barrier — see the immediate-commit pass below).
    let is_terrain_edit: Vec<bool> = gathered
        .iter()
        .map(|g| matches!(g.edit.prim, edits::SdfPrimitive::Terrain { .. }))
        .collect();
    // Per-edit "is this a SUBTRACT (carving) edit" — a subtractor only removes geometry + carries no material,
    // so a chunk of Terrain + only-Subtract edits is still a terrain SURFACE (just dug): it keeps the
    // terrain-surface material (strata on the dug walls) but takes CSG normals (the clipmap normal is wrong on
    // a cavity wall). See the `terrain_surface`/`carved` split below.
    let is_subtract_edit: Vec<bool> =
        gathered.iter().map(|g| g.edit.op.kind == edits::CsgKind::Subtract).collect();
    let height_clipmap = crate::sdf_render::worldgen::upload::cpu_height_clipmap();

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

    // CLIPMAP: camera position + LOD count (camera-driven; no camera ⇒ LOD-0 everywhere). Capture the
    // frustum's 6 inward half-spaces (normal, d) for bake priority (in-view first) — a one-frame-stale
    // copy is fine for ordering, and it lets the bake hold no ECS borrow into the REQUEST loop.
    let cam_view = cameras.iter().next();
    let live_cam = cam_view.map(|(t, _)| t.translation());
    let cam_frustum: Option<[Vec4; 6]> =
        cam_view.and_then(|(_, f)| f.map(|fr| fr.half_spaces.map(|hs| hs.normal_d())));
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
                    key, &config, k, &terrain_xz_aabbs, height_clipmap.as_deref(),
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
    // Chunks whose candidate edits are ALL the Terrain primitive — they commit per-chunk immediately
    // (no atomic-edit round barrier). Recomputed every frame over the live residency.
    // `terrain_surface`: chunks eligible for the terrain-surface material (volumetric strata). Every candidate
    // edit is Terrain OR a Subtract carve, AND at least one is Terrain — i.e. terrain, possibly DUG, but with
    // no additive/object material placed. `carved`: the subset that has a subtractor, so it takes CSG normals
    // (the clipmap normal is wrong on the dug cavity walls). A pure-terrain chunk is `terrain_surface` and NOT
    // `carved` → smooth clipmap normals.
    let mut terrain_surface: HashSet<BrickKey> = HashSet::new();
    let mut carved: HashSet<BrickKey> = HashSet::new();
    let mut by_lod = [0usize; MAX_MESH_LODS as usize];
    {
        let mut idx: Vec<u32> = Vec::new();
        for &key in &resident {
            by_lod[(key.lod as usize).min(MAX_MESH_LODS as usize - 1)] += 1;
            cull_into(&edit_aabbs, &chunk_sampled(key), &mut idx);
            // Drop edits that are sub-voxel at this chunk's LOD so a tiny object can't contaminate a chunk
            // resident for a larger one (the residency cull already keeps lone sub-voxel objects out). Same
            // predicate as the bake fold below → hash and geometry always agree.
            idx.retain(|&i| edit_resolvable_at(edit_extent[i as usize], &config, key.lod));
            let all_terrain_or_carve =
                idx.iter().all(|&i| is_terrain_edit[i as usize] || is_subtract_edit[i as usize]);
            let has_terrain = idx.iter().any(|&i| is_terrain_edit[i as usize]);
            if !idx.is_empty() && all_terrain_or_carve && has_terrain {
                terrain_surface.insert(key);
                if idx.iter().any(|&i| is_subtract_edit[i as usize]) {
                    carved.insert(key);
                }
            }
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

    // SELF-EVICTION test: a chunk is "gone" when its centre lies outside the LIVE clipmap's outer-LOD cube —
    // the camera has flown past it, so the world no longer contains it. Its bake work is wasted and its mesh
    // can be dropped (it's off the clipmap entirely). This is the ONLY per-frame mesh despawn; every OTHER
    // displayed mesh is held until a COMMITTED chunk covers its region (keep-old-until-covered), so a near
    // chunk is never evicted before its replacement is ready. `chunk_in_outer_cube` is the shared SSOT.
    let gone = |key: &BrickKey| -> bool { !chunk_in_outer_cube(*key, cam, half0, lod_count, &config, k) };

    // Bundled main-thread asset stores + live detail-normal config for the commits below (the per-chunk
    // mesh / detail-normal Image / TerrainMaterial allocations).
    let mut spawn_assets = SpawnAssets {
        mesh_assets: &mut mesh_assets,
        images: &mut images,
        terrain_mats: &mut terrain_mats,
        mesh_mats: &mesh_mats,
        detail_strength: mesh_cfg.detail_normal_strength,
        debug_normals: mesh_cfg.debug_normals,
        strata: super::worldgen::biome::StrataTableStd::from_library(&terrain_mat.lib),
        palette: super::worldgen::biome::MaterialPaletteStd::from_library(&terrain_mat.lib),
        tex_arrays: (
            terrain_mat.tex.diffuse.clone(),
            terrain_mat.tex.normal.clone(),
            terrain_mat.tex.mra.clone(),
        ),
    };

    // 1. GONE EVICT: a chunk the camera has flown entirely past ("gone", off the live clipmap) is no longer
    // valid — despawn its meshes (displayed AND hidden both carry `ChunkMesh`) and drop its state (which
    // cancels its in-flight `Task` by drop). This is the ONLY unconditional per-frame mesh despawn; every
    // OTHER displayed mesh is held until a REVEALED replacement covers its region (keep-old-until-revealed).
    for (e, cm) in &chunk_meshes {
        if gone(&cm.0) {
            commands.entity(e).despawn();
        }
    }
    states.0.retain(|key, _| !gone(key));

    // 2. RECEIVE + COMMIT: poll each in-flight bake; on completion display its result (keep-old-until-revealed).
    // The task targeted `task_hash` (the desired hash when it was issued); `commit_baked` records that as the
    // hash the new geometry was baked from.
    for (key, st) in states.0.iter_mut() {
        let Some(task) = st.task.as_mut() else {
            continue;
        };
        let Some(result) = block_on(poll_once(&mut *task)) else {
            continue;
        };
        st.task = None;
        let task_hash = st.task_hash;
        commit_baked(&mut commands, &mut spawn_assets, &config, *key, result, task_hash, st);
    }

    // 3. EVICT (keep-old-until-revealed): despawn a displayed mesh once a REVEALED resident chunk TILES its
    // region — its replacement is on screen, so dropping the old opens no hole and shows no green flash.
    // `revealed_keys` snapshots the revealed set so the reap predicate borrows no live `states` (the despawn
    // bookkeeping below mutates it). A displayed chunk's OWN key ∈ resident ⇒ `render_commit_reap` returns
    // false ⇒ it is never reaped; only superseded / out-of-residency meshes covered by a revealed replacement
    // are. `gone` meshes were already handled above.
    let revealed_keys: HashSet<BrickKey> =
        states.0.iter().filter(|(_, st)| st.revealed).map(|(key, _)| *key).collect();
    let is_revealed = |key: BrickKey| revealed_keys.contains(&key);
    let mut to_reap: Vec<(Entity, BrickKey)> = Vec::new();
    for (e, cm) in &chunk_meshes {
        if render_commit_reap(cm.0, resident.contains(&cm.0), &resident, &is_revealed, cam, half0, lod_count, &config, k)
        {
            to_reap.push((e, cm.0));
        }
    }
    stats.reaped = to_reap.len();
    for (e, key) in &to_reap {
        commands.entity(*e).despawn();
        if let Some(st) = states.0.get_mut(key) {
            st.displayed.retain(|x| x != e);
            st.hidden.retain(|x| x != e);
        }
    }
    // Drop states with nothing displayed, nothing hidden, no in-flight task, and no longer resident.
    states.0.retain(|key, st| {
        resident.contains(key) || !st.displayed.is_empty() || !st.hidden.is_empty() || st.task.is_some()
    });

    // Diagnostic dump (panel "Capture diagnostics"). At rest: in-flight=stale=held=0.
    if stats.capture {
        stats.capture = false;
        let inflight_n = states.0.values().filter(|s| s.task.is_some()).count();
        let stale_n = resident
            .iter()
            .filter(|k| states.0.get(*k).is_none_or(|s| s.displayed_hash != current_hashes[*k]))
            .count();
        let held_n = chunk_meshes.iter().filter(|(_, cm)| !resident.contains(&cm.0)).count();
        let displayed_n = states.0.values().filter(|s| !s.displayed.is_empty()).count();
        let mut s = String::new();
        s.push_str("=== Mesh Bake Diagnostics ===\n");
        s.push_str(&format!(
            "volumes(edits)={n_edits}  chunk_bricks(K)={k}  resident_chunks={}\n",
            resident.len()
        ));
        s.push_str(&format!(
            "displayed={displayed_n}  in-flight={inflight_n}  stale={stale_n}  held={held_n}\n"
        ));
        s.push_str("(at rest: in-flight=stale=held=0)\n");
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

    // 4. REQUEST: bake every resident chunk that needs work toward its LIVE desired hash. A chunk needs a bake
    // iff neither its displayed nor its hidden geometry matches `desired`, and it has no task already targeting
    // `desired`. A task whose `task_hash != desired` is SUPERSEDED (the camera/edits moved on) → drop it so it
    // re-issues toward the live target. Spawn in PRIORITY order (`bake_priority`); the per-frame budget caps
    // task spawns. Because there is no frozen round, every chunk always advances toward the live desired hash —
    // the post-fill stall cannot happen.
    {
        let pool = AsyncComputeTaskPool::get();
        let mut budget = MAX_NEW_TASKS_PER_FRAME;
        let mut idx: Vec<u32> = Vec::new();
        let debug = mesh_cfg.debug_lod_colour;
        // Terrain-surface bake resolutions forwarded to each terrain-only chunk's task (detail_res 0 ⇒
        // detail-normal disabled; height/biome still bake).
        let detail_res = mesh_cfg.detail_normal_res;
        let biome_res = mesh_cfg.biome_res;
        let biome_blend_m = mesh_cfg.biome_blend_m;
        // Install the LIVE clipmap on THIS (system) thread for the whole REQUEST loop, so the SYNCHRONOUS
        // narrow-band cull below (`chunk_has_surface` → `terrain_sdf`) samples the SAME clipmap the async
        // `mesh_chunk` bakes against. Residency is live-gated this frame (the coverage gate ran against this
        // exact `height_clipmap`), so a chunk only reaches here if the live clipmap covers it — no strict-
        // sampler panic. Without this the cull would fall through to the process-global store.
        let _bake_terrain = crate::sdf_render::worldgen::upload::set_bake_terrain(
            height_clipmap.clone(),
            crate::sdf_render::worldgen::upload::cpu_terrain_offset(),
        );
        let prio_cam = cam.unwrap_or(Vec3::ZERO);
        // Resident chunks needing a NEW bake: neither displayed nor hidden matches `desired`, AND nothing is
        // already baking. We do NOT cancel an in-flight bake whose target was superseded: under continuous
        // movement the desired hash (its transition flags) changes most frames, so cancel-and-restart would
        // drop the bake EVERY frame → it would NEVER finish → the chunk only ever appears once you stop ("takes
        // ages to show up"). Instead we let the in-flight bake COMPLETE and commit its (slightly stale) result —
        // the chunk appears promptly — then a follow-up bake re-issues toward the live target and refines it.
        let mut pending: Vec<BrickKey> = resident
            .iter()
            .copied()
            .filter(|key| {
                let desired = current_hashes[key];
                match states.0.get(key) {
                    Some(st) => {
                        chunk_needs_bake(st.displayed_hash, st.hidden_hash, st.task.is_some(), desired)
                    }
                    None => true,
                }
            })
            .collect();
        // Only `budget` chunks are baked this frame, but `pending` can be thousands (a cold fill). Partition
        // the budget-highest-priority chunks to the front in O(n) (`select_nth`), drop the rest, then sort just
        // that prefix for the spawn order — instead of an O(n log n) sort of the whole list every frame.
        let prio = |key: &BrickKey| bake_priority(*key, &config, k, prio_cam, cam_frustum.as_ref());
        if pending.len() > budget {
            pending.select_nth_unstable_by_key(budget, prio);
            pending.truncate(budget);
        }
        pending.sort_unstable_by_key(prio);
        for key in pending {
            if budget == 0 {
                break; // remaining (lower-priority) chunks re-detected next frame
            }
            let desired = current_hashes[&key];
            let vs_l = config.voxel_size_at(key.lod);
            cull_into(&edit_aabbs, &chunk_sampled(key), &mut idx);
            // Sub-voxel cull (same predicate as the hash fold): exclude edits too small to mesh at this LOD
            // from the field so they can't bake a degenerate sliver into a chunk resident for a larger edit.
            idx.retain(|&i| {
                let a = edit_aabbs[i as usize];
                edit_resolvable_at((Vec3::from(a.max) - Vec3::from(a.min)).max_element(), &config, key.lod)
            });
            // NARROW-BAND CULL: skip chunks with no surface crossing (interior/exterior of a solid) for a
            // single SDF eval instead of a full edge³ bake — the big win for large objects. Commit them
            // empty INLINE (no task, no budget spent): the chunk's current result is "nothing here".
            if !chunk_has_surface(&edits_arc, &idx, &config, k, key, vs_l) {
                let st = states.0.entry(key).or_default();
                commit_baked(&mut commands, &mut spawn_assets, &config, key, None, desired, st);
                continue;
            }
            // Transvoxel block = the chunk's exact world extent (NO apron); its origin is the chunk MIN corner.
            let grid_origin = config.brick_min_world(key.coord, key.lod);
            // Transvoxel transition faces — those bordering a FINER LOD — from the LIVE shell. Folded into the
            // content hash (`desired`) → a chunk re-bakes with the right transition sides when the shell moves.
            let flags = chunk_finer_faces(key, &config, k, cam, half0);
            let lod = key.lod;
            let edits = edits_arc.clone();
            let indices = idx.clone();
            // The LIVE clipmap — the same snapshot the coverage gate admitted this chunk against this frame.
            let terrain = height_clipmap.clone();
            // Surface-material chunks (terrain, incl. dug) render the strata; only PURE (uncarved) terrain
            // takes the smooth clipmap normals — a carved chunk uses CSG normals for its cavity walls.
            let surface_material = terrain_surface.contains(&key);
            let terrain_normals = surface_material && !carved.contains(&key);
            let st = states.0.entry(key).or_default();
            st.task_hash = desired;
            st.task = Some(pool.spawn(async move {
                mesh_chunk(
                    &edits, &indices, grid_origin, vs_l, k * cs, flags, lod, debug, terrain, terrain_normals,
                    surface_material, detail_res, biome_res, biome_blend_m,
                )
            }));
            budget -= 1;
        }
    }

    // "Still baking" signal for the editor status bar: resident chunks whose DISPLAYED geometry doesn't yet
    // match the desired hash — in-flight, hidden-awaiting-reveal, or not-yet-started (budget-limited / just
    // entered residency). 0 ⇒ everything resident is on screen at its current target.
    stats.pending = resident
        .iter()
        .filter(|k| match states.0.get(k) {
            Some(st) => st.displayed_hash != current_hashes[k],
            None => true,
        })
        .count();
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
    if ui
        .add(bevy_egui::egui::Slider::new(&mut lods, 1..=MAX_MESH_LODS).text("LOD levels"))
        .on_hover_text("Mesh-bake LOD ring count. The worldgen height-clipmap window grows to match, so \
                        terrain extends to the configured LOD reach (coarser = much farther, much more to bake).")
        .changed()
    {
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
    // DETAIL-NORMAL bake (Zylann-style): a per-chunk normal-map texture baked on COARSE terrain-only chunks
    // from the fine band-limited surface gradient, so far/low-poly terrain SHADES with sub-triangle relief.
    // "Detail normal res" = the N×N texel resolution (changing it RE-BAKES the maps); "Detail normal
    // strength" = how far the per-pixel hi-fi normal pulls the coarse geometry normal (a LIVE shader uniform,
    // no re-bake). Gated to coarse LODs (near chunks already have full geometric detail).
    let mut dres = world.resource::<MeshBakeConfig>().detail_normal_res;
    if ui
        .add(bevy_egui::egui::Slider::new(&mut dres, 0..=512).text("Detail normal res"))
        .on_hover_text("N×N per-chunk detail-normal map resolution baked on coarse terrain chunks (0 = off). \
                        Higher = finer baked relief but more N² gradient samples + bigger per-chunk textures. \
                        Changing it re-bakes the terrain.")
        .changed()
    {
        world.resource_mut::<MeshBakeConfig>().detail_normal_res = dres;
        // The baked texel data changes with resolution → force a re-bake of every chunk.
        world.resource_mut::<MeshBakeRebuild>().0 = true;
    }
    let mut dstr = world.resource::<MeshBakeConfig>().detail_normal_strength;
    if ui
        .add(bevy_egui::egui::Slider::new(&mut dstr, 0.0..=1.0).text("Detail normal strength"))
        .on_hover_text("How far the baked per-pixel hi-fi normal pulls the coarse geometry normal \
                        (0 = none, 1 = full detail). Live shader uniform — no re-bake.")
        .changed()
    {
        world.resource_mut::<MeshBakeConfig>().detail_normal_strength = dstr;
    }
    // BIOME STRATA (Stages 2+3): the volumetric biome strata + surface materials render on every terrain-only
    // chunk. "Biome map res" = the per-chunk N×N biome + surface-material map resolution (RE-BAKES; biome is
    // low-frequency so a small map suffices). The surface material (biome base + snow/rock caps + cliffs +
    // patches) is authored in `biomes.ron` surface_rules and baked — no live treatment slider.
    let mut bres = world.resource::<MeshBakeConfig>().biome_res;
    if ui
        .add(bevy_egui::egui::Slider::new(&mut bres, 2..=256).text("Biome map res"))
        .on_hover_text("N×N per-chunk biome (primary/secondary/blend) map resolution. Biome is km-scale, so \
                        small is plenty. Changing it re-bakes the terrain.")
        .changed()
    {
        world.resource_mut::<MeshBakeConfig>().biome_res = bres;
        world.resource_mut::<MeshBakeRebuild>().0 = true;
    }
    let mut bblend = world.resource::<MeshBakeConfig>().biome_blend_m;
    if ui
        .add(bevy_egui::egui::Slider::new(&mut bblend, 0.0..=600.0).text("Biome blend width (m)"))
        .on_hover_text("WORLD-space half-width of the biome→neighbour surface-colour cross-fade. The baked \
                        blend is gradient-normalised, so borders fade over this many metres EVERYWHERE \
                        regardless of how fast the climate changes locally (no hard lines). Changing it re-bakes.")
        .changed()
    {
        world.resource_mut::<MeshBakeConfig>().biome_blend_m = bblend;
        world.resource_mut::<MeshBakeRebuild>().0 = true;
    }
    // (The old "Surface treatment" slider is gone — snow caps / cliff rock are now authored per-biome
    // SURFACE RULES in `biomes.ron`, resolved + baked by the worldgen, not a live shader override.)

    // PHYSICS — the SEPARATE coarse collider clipmap (`physics_resident_chunks`): a small set of collider-only
    // chunks around the player/camera at +N LOD, streamed independently of the render mesh so colliders persist
    // through render re-bakes. These knobs are LIVE (the physics system re-streams itself; no render re-bake).
    let mut phys = world.resource::<MeshBakeConfig>().physics;
    if ui
        .checkbox(&mut phys, "Physics colliders")
        .on_hover_text("Separate coarse collider clipmap around the player/camera, decoupled from the render \
                        mesh so colliders persist while it streams (no fall-through).")
        .changed()
    {
        world.resource_mut::<MeshBakeConfig>().physics = phys;
    }
    let mut pbase = world.resource::<MeshBakeConfig>().physics_base_lod;
    if ui
        .add(bevy_egui::egui::Slider::new(&mut pbase, 0..=6).text("Physics base LOD (+N)"))
        .on_hover_text("Finest collider LOD = render LOD 0 + this. Higher = coarser colliders (far fewer \
                        triangles); a character controller doesn't need render fidelity. Default 2 (+2 LOD).")
        .changed()
    {
        world.resource_mut::<MeshBakeConfig>().physics_base_lod = pbase;
    }
    let mut pcount = world.resource::<MeshBakeConfig>().physics_lod_count;
    if ui
        .add(bevy_egui::egui::Slider::new(&mut pcount, 1..=4).text("Physics LOD rings"))
        .on_hover_text("How many collider LODs the physics clipmap spans (each ring doubles the radius).")
        .changed()
    {
        world.resource_mut::<MeshBakeConfig>().physics_lod_count = pcount;
    }
    let mut phalf = world.resource::<MeshBakeConfig>().physics_half_chunks;
    if ui
        .add(bevy_egui::egui::Slider::new(&mut phalf, 1..=8).text("Physics radius (chunks)"))
        .on_hover_text("Half-extent of the finest collider cube around the focus, in physics-base chunks. \
                        Small keeps the set tiny so it rarely re-streams. Higher = colliders reach further.")
        .changed()
    {
        world.resource_mut::<MeshBakeConfig>().physics_half_chunks = phalf;
    }
    let mut pwire = world.resource::<MeshBakeConfig>().physics_wireframe;
    if ui
        .checkbox(&mut pwire, "Physics wireframe (debug)")
        .on_hover_text("Render each baked collider chunk as a green wireframe (the COARSE physics mesh + the \
                        clipmap's reach). Toggling rebuilds the small physics set.")
        .changed()
    {
        world.resource_mut::<MeshBakeConfig>().physics_wireframe = pwire;
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

    // Stats. `meshing`/`hidden` are transiently non-zero while chunks bake + stream in; they drop to 0 once
    // the resident set is fully baked + revealed.
    let states = world.resource::<ChunkStates>();
    let meshes = states.0.values().map(|s| s.displayed.len()).sum::<usize>();
    let in_flight = states.0.values().filter(|s| s.task.is_some()).count();
    let hidden = states.0.values().filter(|s| !s.hidden.is_empty()).count();
    ui.label(format!("Chunk meshes: {meshes}  ·  meshing: {in_flight}  ·  hidden: {hidden}"));

    // System view. `entities` may briefly exceed `resident` during an edit — departed meshes are HELD
    // until a revealed replacement covers them (so old + new swap together); at rest they match.
    let stats = world.resource::<MeshBakeStats>();
    let (edits, resident, reaped) = (stats.edits, stats.resident, stats.reaped);
    let entities = world.query_filtered::<(), With<ChunkMesh>>().iter(world).count();
    ui.label(format!(
        "edits: {edits}  ·  resident: {resident}  ·  entities: {entities}  ·  reaped/commit: {reaped}"
    ));
    // PHYSICS collider clipmap (separate, dedicated-pool bake). `baking` > 0 ⇒ colliders still filling in.
    let stats = world.resource::<MeshBakeStats>();
    ui.label(format!(
        "physics: resident {}  ·  colliders {}  ·  baking {}",
        stats.physics_resident, stats.physics_displayed, stats.physics_baking
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

// Performance / benchmark rig for the full LOD-8 terrain mesh-bake. Declared `#[path]`-inline so it gets
// `super::*` (full private access to the residency helpers it faithfully replicates). Run command in its
// module doc. It MEASURES only — drives the real `mesh_chunk` + the production residency/cull formulas.
#[cfg(test)]
#[path = "mesh_bake_perf.rs"]
mod perf;

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

    /// `chunk_key_at` is the exact inverse of a chunk's region: the chunk that tiles a chunk's own centre, at
    /// that chunk's LOD, is the chunk itself. Guards the point→chunk mapping the coverage test relies on.
    #[test]
    fn chunk_key_at_inverts_chunk_region() {
        let (cfg, _) = cfgs();
        let k = 4;
        for lod in 0..5 {
            for j in [-7, -1, 0, 3, 12, 40] {
                let key = chunk(&cfg, k, lod, j);
                let c = chunk_centre_world(key, &cfg, k);
                assert_eq!(chunk_key_at(c, lod, &cfg, k), key, "lod {lod} j {j}");
            }
        }
    }

    /// KEEP-OLD-UNTIL-COVERED, the structural invariant: a displayed chunk is reaped ONLY when a committed
    /// chunk actually tiles its region; an uncovered region (coverage-gate-dropped under fast flight, or the
    /// round froze behind) is KEPT so no hole opens; a re-committed or fully-departed chunk is left to its own
    /// path. Tests the SSOT `render_commit_reap` the production reap loop calls.
    #[test]
    fn render_commit_reap_keeps_uncovered_until_replacement_exists() {
        let (cfg, _) = cfgs();
        let (k, lod_count, half0) = (4u32, 12u32, 4i32);
        let live_cam = Some(Vec3::ZERO);
        let d = chunk(&cfg, k, 0, 0); // a displayed near chunk, inside the live clipmap
        assert!(chunk_in_outer_cube(d, live_cam, half0, lod_count, &cfg, k));
        let all_revealed = |_: BrickKey| true; // baseline: every covering chunk is renderable

        // (1) re-committed this round → never reaped here (its own commit swaps it).
        let just_d: HashSet<BrickKey> = [d].into_iter().collect();
        assert!(!render_commit_reap(d, true, &just_d, &all_revealed, live_cam, half0, lod_count, &cfg, k));

        // (2) region ABSENT from the committed set (gate-dropped / frozen behind) → KEEP (no hole).
        let empty = HashSet::new();
        assert!(
            !render_commit_reap(d, false, &empty, &all_revealed, live_cam, half0, lod_count, &cfg, k),
            "uncovered region must be kept"
        );

        // (3) a committed+REVEALED chunk that TILES d's region (the coarser LOD-1 owner) → reap (swap).
        let owner = chunk_key_at(chunk_centre_world(d, &cfg, k), 1, &cfg, k);
        let covered: HashSet<BrickKey> = [owner].into_iter().collect();
        assert!(render_commit_reap(d, false, &covered, &all_revealed, live_cam, half0, lod_count, &cfg, k));

        // (3b) KEEP-OLD-UNTIL-REVEALED: the covering owner exists but is NOT yet revealed (its textured
        // material isn't GPU-live) → the old geometry must be KEPT (no green flash, no hole).
        let none_revealed = |_: BrickKey| false;
        assert!(
            !render_commit_reap(d, false, &covered, &none_revealed, live_cam, half0, lod_count, &cfg, k),
            "a not-yet-revealed replacement must not evict the old mesh"
        );

        // (4) a chunk far outside the live clipmap is GONE → not reaped here (the self-evict pass owns it).
        let faraway = chunk(&cfg, k, 0, 1_000_000);
        assert!(!chunk_in_outer_cube(faraway, live_cam, half0, lod_count, &cfg, k));
        assert!(!render_commit_reap(faraway, false, &empty, &all_revealed, live_cam, half0, lod_count, &cfg, k));
    }

    /// MULTI-LOD keep-old (the fast-movement bug): when the camera outruns the bakes by several rings, an old
    /// COARSE chunk's region is tiled by MANY finer chunks SEVERAL LODs down — it must be kept until EVERY one
    /// of them is revealed. A partial cover (one missing) must NOT reap it. Guards the recursive coverage test
    /// against the old 8-octant SAMPLE, which under-counted across a >1 LOD gap and punched holes.
    #[test]
    fn render_commit_reap_keeps_coarse_across_multi_lod_gap() {
        let (cfg, _) = cfgs();
        let (k, lod_count, half0) = (4u32, 12u32, 4i32);
        let live_cam = Some(Vec3::ZERO);
        let all_revealed = |_: BrickKey| true;
        let coarse = chunk(&cfg, k, 3, 0); // a displayed LOD-3 chunk — replaced by LOD-1 chunks (a 2-LOD gap)

        // Enumerate EVERY LOD-1 chunk tiling the LOD-3 chunk's region (4×4×4 = 64 of them).
        let b = chunk_aabb(coarse, &cfg, k);
        let dmin = Vec3::from(b.min);
        let cw1 = k as f32 * cfg.brick_world_size(1);
        let span = (k as f32 * cfg.brick_world_size(3) / cw1).round() as i32; // 4 per axis
        let mut fine: HashSet<BrickKey> = HashSet::new();
        for i in 0..span {
            for j in 0..span {
                for l in 0..span {
                    let p = dmin + Vec3::new(i as f32, j as f32, l as f32) * cw1 + Vec3::splat(cw1 * 0.5);
                    fine.insert(chunk_key_at(p, 1, &cfg, k));
                }
            }
        }
        assert_eq!(fine.len(), (span * span * span) as usize, "expected a full 4×4×4 fine tiling");

        // FULL cover (all 64 finer chunks resident + revealed) → reap.
        assert!(render_commit_reap(coarse, false, &fine, &all_revealed, live_cam, half0, lod_count, &cfg, k));

        // PARTIAL cover (drop ONE of the 64) → KEEP — the old coarse mesh must not leave a hole over the gap.
        let mut partial = fine.clone();
        let drop_key = *partial.iter().next().unwrap();
        partial.remove(&drop_key);
        assert!(
            !render_commit_reap(coarse, false, &partial, &all_revealed, live_cam, half0, lod_count, &cfg, k),
            "a coarse mesh must be kept until ALL its multi-LOD-finer replacements are on screen"
        );
    }

    /// COARSE→FINE keep-old: a coarse chunk replaced by its 2×2×2 finer sub-chunks must be KEPT until ALL of
    /// them are revealed — a PARTIAL cover (only some sub-chunks present) must NOT reap it (else the centre
    /// test would punch holes over the missing sub-chunks). Guards the all-8-octant coverage rule.
    #[test]
    fn render_commit_reap_keeps_coarse_until_all_finer_subchunks_revealed() {
        let (cfg, _) = cfgs();
        let (k, lod_count, half0) = (4u32, 12u32, 4i32);
        let live_cam = Some(Vec3::ZERO);
        let all_revealed = |_: BrickKey| true;
        let coarse = chunk(&cfg, k, 2, 0); // a displayed LOD-2 chunk over the origin

        // The full set of finer (LOD-1) chunks tiling the coarse chunk's region — sample its 8 octant centres.
        let b = chunk_aabb(coarse, &cfg, k);
        let (mn, w) = (Vec3::from(b.min), Vec3::from(b.max) - Vec3::from(b.min));
        let mut all_fine: HashSet<BrickKey> = HashSet::new();
        for fx in [0.25_f32, 0.75] {
            for fy in [0.25_f32, 0.75] {
                for fz in [0.25_f32, 0.75] {
                    all_fine.insert(chunk_key_at(mn + w * Vec3::new(fx, fy, fz), 1, &cfg, k));
                }
            }
        }
        // FULL cover (all finer sub-chunks resident + revealed) → reap.
        assert!(render_commit_reap(coarse, false, &all_fine, &all_revealed, live_cam, half0, lod_count, &cfg, k));

        // PARTIAL cover (drop one sub-chunk) → KEEP (the missing octant isn't on screen).
        let mut partial = all_fine.clone();
        let dropped = *partial.iter().next().unwrap();
        partial.remove(&dropped);
        assert!(
            !render_commit_reap(coarse, false, &partial, &all_revealed, live_cam, half0, lod_count, &cfg, k),
            "a coarse mesh must be kept until ALL its finer replacements are on screen"
        );
    }

    /// `chunk_renderable` is the reveal/keep-old SSOT: a terrain chunk is renderable (safe to reveal + count
    /// as a cover) only once its settle window has elapsed to `0`.
    #[test]
    fn chunk_renderable_only_after_settle() {
        assert!(!chunk_renderable(REVEAL_SETTLE_FRAMES));
        assert!(!chunk_renderable(1));
        assert!(chunk_renderable(0));
    }

    /// An in-flight bake is NEVER cancelled because its target was superseded (the SSOT rule): a chunk that
    /// already has a task in flight never asks for a new one, even when its desired hash has changed. This is
    /// what stops the "restart every frame → never finish → takes ages to show up" failure.
    #[test]
    fn chunk_needs_bake_never_restarts_an_inflight_bake() {
        // Nothing displayed/hidden matches the desired, AND a bake is in flight → do NOT issue another.
        assert!(!chunk_needs_bake(0, 0, true, 999));
        // No task in flight + nothing matches desired → DO bake.
        assert!(chunk_needs_bake(0, 0, false, 999));
        // Already displaying (or hidden) the desired result → no bake.
        assert!(!chunk_needs_bake(999, 0, false, 999));
        assert!(!chunk_needs_bake(0, 999, false, 999));
    }

    // ============================ STREAMING LIFECYCLE SIMULATION HARNESS ============================
    // A deterministic, headless model of the per-chunk streaming lifecycle (residency → mock async bake →
    // reveal → keep-old-until-revealed reap), driving the REAL production helpers (`mesh_chunk_in_shell`
    // residency, `chunk_finer_faces` transition flags, `render_commit_reap`, `chunk_needs_bake`) with a
    // fixed-latency mock bake. It drives a camera path and asserts the invariant the runtime kept breaking,
    // with NO async/ECS: a chunk that stays resident long enough is ALWAYS on screen — even under CONTINUOUS
    // movement, where every frame changes its desired transition-flag hash (the "drop the superseded in-flight
    // bake" regression restarts bakes every frame so they never finish → "takes ages to show up").

    #[derive(Default, Clone)]
    struct SimChunk {
        displayed_hash: u64, // mesh ON SCREEN (0 = nothing shown)
        hidden_hash: u64,    // baked, awaiting reveal (0 = none)
        reveal_left: u8,
        task_hash: u64, // bake in flight toward this (0 = none)
        task_left: u8,
    }

    struct Sim {
        cfg: SdfGridConfig,
        k: u32,
        half0: i32,
        lod_count: u32,
        bake_latency: u8,
        reveal_frames: u8,
        chunks: HashMap<BrickKey, SimChunk>,
        age: HashMap<BrickKey, u32>, // consecutive frames a chunk has been resident
    }

    impl Sim {
        fn resident(&self, cam: Vec3) -> HashSet<BrickKey> {
            let cw0 = self.k as f32 * self.cfg.brick_world_size(0);
            let (mut out, mut cand) = (HashSet::new(), HashSet::new());
            for lod in 0..self.lod_count {
                cand.clear();
                let centre = lod_centre(&self.cfg, self.k, cam, lod).as_vec3() * cw0;
                let half = ((self.half0 << lod) as f32) * cw0;
                let aabb = Aabb3d::from_min_max(centre - Vec3::splat(half), centre + Vec3::splat(half));
                chunks_in_aabb(&aabb, &self.cfg, self.k, lod, &mut cand);
                for &key in &cand {
                    if mesh_chunk_in_shell(key, &self.cfg, self.k, Some(cam), self.half0) {
                        out.insert(key);
                    }
                }
            }
            out
        }
        // A non-zero content hash that CHANGES with lod + the live transition flags (so it churns under motion).
        fn desired(&self, key: BrickKey, cam: Vec3) -> u64 {
            let f = chunk_finer_faces(key, &self.cfg, self.k, Some(cam), self.half0);
            ((key.lod as u64 + 1) << 8) ^ (f as u64 + 1)
        }
        fn shown(&self, key: BrickKey) -> bool {
            self.chunks.get(&key).is_some_and(|c| c.displayed_hash != 0)
        }
        fn frame(&mut self, cam: Vec3) {
            let resident = self.resident(cam);
            // RECEIVE: advance + complete in-flight bakes → hidden (terrain reveal path).
            for c in self.chunks.values_mut() {
                if c.task_left > 0 {
                    c.task_left -= 1;
                    if c.task_left == 0 {
                        c.hidden_hash = c.task_hash;
                        c.task_hash = 0;
                        c.reveal_left = self.reveal_frames.max(1);
                    }
                }
            }
            // REVEAL: swap hidden → displayed once settled.
            for c in self.chunks.values_mut() {
                if c.hidden_hash != 0 && c.reveal_left > 0 {
                    c.reveal_left -= 1;
                    if c.reveal_left == 0 {
                        c.displayed_hash = c.hidden_hash;
                        c.hidden_hash = 0;
                    }
                }
            }
            // REAP (keep-old-until-revealed): drop a shown mesh only once its region is fully covered by shown
            // chunks. `shown` (displayed != 0) is the sim's `is_revealed`.
            let shown: HashSet<BrickKey> =
                self.chunks.iter().filter(|(_, c)| c.displayed_hash != 0).map(|(k, _)| *k).collect();
            let is_rev = |key: BrickKey| shown.contains(&key);
            let reap: Vec<BrickKey> = self
                .chunks
                .iter()
                .filter(|(key, c)| {
                    c.displayed_hash != 0
                        && render_commit_reap(
                            **key, resident.contains(key), &resident, &is_rev, Some(cam), self.half0,
                            self.lod_count, &self.cfg, self.k,
                        )
                })
                .map(|(k, _)| *k)
                .collect();
            for key in reap {
                self.chunks.get_mut(&key).unwrap().displayed_hash = 0;
            }
            self.chunks
                .retain(|key, c| resident.contains(key) || c.displayed_hash != 0 || c.hidden_hash != 0 || c.task_left > 0);
            // REQUEST: issue bakes (no budget cap in the sim; NO cancel of in-flight — the fix).
            for &key in &resident {
                let d = self.desired(key, cam);
                let c = self.chunks.entry(key).or_default();
                if chunk_needs_bake(c.displayed_hash, c.hidden_hash, c.task_left > 0, d) {
                    c.task_hash = d;
                    c.task_left = self.bake_latency.max(1);
                }
            }
            // Track consecutive residency age.
            self.age.retain(|key, _| resident.contains(key));
            for &key in &resident {
                *self.age.entry(key).or_default() += 1;
            }
        }
    }

    /// HARNESS: cold-fill then fly continuously. A chunk that has been resident for at least the bake+reveal
    /// budget is ALWAYS on screen — the regression guard for "takes ages to show up" (a cancel-on-supersede
    /// would leave long-resident chunks perpetually un-baked under motion).
    #[test]
    fn streaming_chunks_appear_even_under_continuous_movement() {
        let cfg = SdfGridConfig { voxel_size: 0.4, ..SdfGridConfig::default() };
        let mc = MeshBakeConfig { lod0_radius: 16.0, ..MeshBakeConfig::default() };
        let k = mc.chunk_bricks.clamp(1, 8);
        let (lod_count, bake_latency, reveal_frames) = (3u32, 4u8, REVEAL_SETTLE_FRAMES);
        let half0 = lod0_half_chunks(&cfg, &mc, k);
        let mut sim = Sim {
            cfg: cfg.clone(), k, half0, lod_count, bake_latency, reveal_frames,
            chunks: HashMap::new(), age: HashMap::new(),
        };
        let appear_bound = bake_latency as u32 + reveal_frames as u32 + 2;

        // (1) COLD FILL at the origin — after the budget elapses, everything resident is on screen.
        let cam0 = Vec3::new(0.0, 40.0, 0.0);
        for _ in 0..(appear_bound + 4) {
            sim.frame(cam0);
        }
        for &key in &sim.resident(cam0) {
            assert!(sim.shown(key), "cold fill: a resident chunk is not on screen");
        }

        // (2) CONTINUOUS MOVEMENT — sub-chunk steps so the transition-flag hash churns most frames. Every frame,
        // every chunk continuously resident for >= appear_bound frames MUST be on screen.
        let step = Vec3::new(cfg.voxel_size * 0.5, 0.0, 0.0);
        for f in 1..150u32 {
            sim.frame(cam0 + step * f as f32);
            for (&key, &age) in &sim.age {
                assert!(
                    age < appear_bound || sim.shown(key),
                    "chunk resident {age} frames still off screen at frame {f} (lod {})",
                    key.lod
                );
            }
        }
    }

    /// The SEPARATE physics clipmap is a self-contained coarse shell: it admits ONLY its `[base, base+count)`
    /// LODs, the finest level fills SOLID at the focus (no hole, unlike the render clipmap's LOD 0), and a
    /// chunk far outside the small cube is rejected. Guards the physics residency math against drift.
    #[test]
    fn physics_clipmap_is_a_self_contained_coarse_shell() {
        let (cfg, _mc) = cfgs();
        let k = 4u32;
        let (base, count, base_half) = (2u32, 2u32, 2i32 << 2); // matches the defaults (half=2 base-chunks)
        let center = Vec3::ZERO;

        // LODs outside the physics band are never resident — even a chunk at the focus.
        assert!(!physics_chunk_in_shell(chunk(&cfg, k, 0, 0), &cfg, k, center, base, count, base_half));
        assert!(!physics_chunk_in_shell(chunk(&cfg, k, base + count, 0), &cfg, k, center, base, count, base_half));

        // The finest physics LOD fills SOLID at the focus (no inner hole) — the chunk over the origin is in.
        assert!(physics_chunk_in_shell(chunk(&cfg, k, base, 0), &cfg, k, center, base, count, base_half));

        // A chunk far beyond the small cube is rejected at every physics LOD.
        for lod in base..base + count {
            assert!(!physics_chunk_in_shell(chunk(&cfg, k, lod, 9999), &cfg, k, center, base, count, base_half));
        }
    }

    /// The COVERAGE GATE excludes a terrain chunk whose XZ footprint reaches OUTSIDE the loaded height
    /// ring, and admits one fully inside it. Build a small resident ring (a 4×4 chunk block at the
    /// origin), then check a fine LOD-0 chunk well inside is covered while a HUGE far chunk (and any
    /// chunk against a `None` ring) is not — exactly the gate that kills the corrupt oversized far slab.
    #[test]
    fn coverage_gate_excludes_uncovered_terrain_chunk() {
        use crate::sdf_render::worldgen::artifact::ScalarField2D;
        use crate::sdf_render::worldgen::coord::{ChunkCoord, ChunkSize, LayerId};
        use crate::sdf_render::worldgen::layers::erosion::ErosionParams;
        use crate::sdf_render::worldgen::layers::height::{
            HEIGHT_CHUNK_CELLS, HEIGHT_FIELD_RES, HeightLayer, HeightParams,
        };
        use crate::sdf_render::worldgen::store::ArtifactStore;
        use crate::sdf_render::worldgen::upload::build_height_ring;
        use std::sync::Arc;

        // Build a resident ring covering height chunks (-3..5, -3..5) around the origin — a generous
        // loaded block so a chunk near the origin clears the gate's `2·HEIGHT_CHUNK_CELLS` apron margin.
        let layer = HeightLayer::new(LayerId(0), HeightParams::default(), ErosionParams::default());
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
        // Wrap the single ring as a 1-tier clipmap (the gate samples a clipmap now).
        let clipmap = vec![build_height_ring(&store)];

        let (cfg, _mc) = cfgs();
        let k = 4u32;
        // One global terrain edit whose XZ footprint spans everything (effectively infinite, as in prod).
        let big = 131072.0f32;
        let terrain = vec![(Vec2::splat(-big), Vec2::splat(big))];

        // A fine LOD-0 chunk at the origin → deep inside the loaded block → covered → gate passes.
        let inside_coord = chunk(&cfg, k, 0, 0);
        assert!(
            terrain_chunk_covered(inside_coord, &cfg, k, &terrain, Some(&clipmap)),
            "a fine chunk inside the loaded block must pass the coverage gate"
        );

        // A HUGE far chunk: a coarse LOD that spans kilometres reaches far outside the ±loaded ring.
        let far = chunk(&cfg, k, 7, 64);
        assert!(
            !terrain_chunk_covered(far, &cfg, k, &terrain, Some(&clipmap)),
            "an oversized far chunk must be excluded (outside loaded coverage)"
        );

        // No clipmap loaded yet ⇒ any terrain-touching chunk is excluded.
        assert!(
            !terrain_chunk_covered(inside_coord, &cfg, k, &terrain, None),
            "with no clipmap loaded, a terrain chunk must not be resident"
        );

        // A chunk that touches NO terrain edit is unaffected by the gate (passes regardless of clipmap).
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
        let data = mesh_chunk(&edits, &[0], origin, vs, sub, 0, 0, false, None, false, false, 0, 0, 0.0).expect("sphere meshes");
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
        let fine = mesh_chunk(&edits, &idx, of, vsf, sub, 0, 0, false, None, false, false, 0, 0, 0.0).expect("fine meshes");
        let coarse = mesh_chunk(&edits, &idx, oc, vsc, sub, 1 << 1, 1, false, None, false, false, 0, 0, 0.0).expect("coarse meshes");
        let mut all = chunk_tris(&fine, of);
        all.extend(chunk_tris(&coarse, oc));
        assert_eq!(
            open_edge_count(&all),
            0,
            "coarse (with transition face) + fine must weld watertight by construction"
        );
    }

    // =================================================================================================
    // TERRAIN CROSS-LOD REGRESSION HARNESS (Step 1) — the structural guard against LOD seams on the REAL
    // eroded terrain (sphere/cube watertight tests never exercised the height-field path). Bakes a fine
    // chunk (LOD L-1) and an abutting coarse chunk (LOD L) across a forced 2:1 boundary on the actual
    // eroded `HeightLayer::sample_world` surface, then asserts (a) geometric watertightness and (b)
    // normal continuity across the shared boundary — the latter is what catches the visible shading KINK.
    // =================================================================================================

    /// Build + publish a single-tier (tier 0) eroded-terrain height clipmap covering height chunks
    /// `(cx, cz)` over `xrange × zrange`, with `set_cpu_terrain_offset(ZERO)`. Returns the published
    /// `Arc<HeightClipmap>` (also installed in the process-global so the bake/coverage gate read it).
    /// Mirrors the `terrain_eval_*` publish pattern in `edits.rs`.
    fn publish_eroded_terrain_clipmap(
        xrange: std::ops::RangeInclusive<i32>,
        zrange: std::ops::RangeInclusive<i32>,
        seed: u64,
    ) -> Arc<HeightClipmap> {
        use crate::sdf_render::worldgen::artifact::ScalarField2D;
        use crate::sdf_render::worldgen::coord::{ChunkCoord, ChunkSize, LayerId};
        use crate::sdf_render::worldgen::layers::erosion::ErosionParams;
        use crate::sdf_render::worldgen::layers::height::{
            HEIGHT_CHUNK_CELLS, HEIGHT_FIELD_RES, HeightLayer, HeightParams,
        };
        use crate::sdf_render::worldgen::store::ArtifactStore;
        use crate::sdf_render::worldgen::upload::{
            build_height_clipmap, set_cpu_height_clipmap, set_cpu_terrain_offset,
        };

        // The REAL eroded terrain layer (default params ⇒ ridge fold + erosion carve ON).
        let layer = HeightLayer::new(LayerId(0), HeightParams::default(), ErosionParams::default());
        let size = ChunkSize::new(HEIGHT_CHUNK_CELLS);
        let mut store = ArtifactStore::new();
        for cz in zrange.clone() {
            for cx in xrange.clone() {
                let coord = ChunkCoord::new(LayerId(0), IVec3::new(cx, 0, cz));
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
        // Single tier 0 (chunk edge = HEIGHT_CHUNK_CELLS), as Step 1 specifies.
        let clip = Arc::new(build_height_clipmap(&store, &[HEIGHT_CHUNK_CELLS]));
        set_cpu_height_clipmap(Some(clip.clone()));
        set_cpu_terrain_offset(Vec2::ZERO);
        clip
    }

    /// The world-anchored Terrain edit spanning the whole test region (IDENTITY transform, material 0,
    /// plain Union) — its vertical band brackets the eroded surface so both chunks' Y windows cross it.
    fn terrain_edit_for_band(min_h: f32, max_h: f32) -> edits::ResolvedEdit {
        edits::ResolvedEdit::new(
            crate::sdf_render::SdfPrimitive::Terrain {
                half_xz: Vec2::splat(1.0e5),
                min_height: min_h,
                max_height: max_h,
            },
            Transform::IDENTITY,
            crate::sdf_render::SdfOp { kind: crate::sdf_render::CsgKind::Union, smoothing: 0.0 },
            0,
        )
    }

    /// Count open mesh edges (not shared by exactly 2 triangles) whose midpoint is INTERIOR to the
    /// combined bounding box `[bmin, bmax]` — i.e. NOT on any of its 6 outer faces. For an OPEN surface
    /// patch (terrain), the outer perimeter is legitimately open (the surface exits the box there); a real
    /// crack at the 2:1 seam shows up as an INTERIOR open edge (on the x=0 plane between the two chunks).
    /// So this isolates the cross-LOD weld correctness from the patch's outer boundary. Welds vertices by
    /// quantised world position (0.1 mm), exactly like [`open_edge_count`].
    fn interior_open_edge_count(tris: &[(Vec3, Vec3, Vec3)], bmin: Vec3, bmax: Vec3) -> usize {
        let q = |p: Vec3| {
            [
                (p.x as f64 * 1e4).round() as i64,
                (p.y as f64 * 1e4).round() as i64,
                (p.z as f64 * 1e4).round() as i64,
            ]
        };
        // Key edges by their quantised endpoints (welds across the two chunk meshes); count incidences.
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
        // Recover a world point from a quantised key (inverse of `q`).
        let unq = |k: [i64; 3]| Vec3::new(k[0] as f32 * 1e-4, k[1] as f32 * 1e-4, k[2] as f32 * 1e-4);
        // A point lies "on an outer face" if it is within tol of any of the box's 6 outer planes. An edge
        // that TOUCHES the outer perimeter (either endpoint on an outer face) sits on the boundary of the
        // open patch and is legitimately open — only edges with BOTH endpoints strictly interior count as a
        // real cross-LOD crack. (Tol is generous to absorb the surface exiting near a corner.)
        let tol = 1.0e-2;
        let on_outer = |p: Vec3| {
            (p.x - bmin.x).abs() <= tol
                || (p.x - bmax.x).abs() <= tol
                || (p.y - bmin.y).abs() <= tol
                || (p.y - bmax.y).abs() <= tol
                || (p.z - bmin.z).abs() <= tol
                || (p.z - bmax.z).abs() <= tol
        };
        edges
            .iter()
            .filter(|(k, n)| **n != 2 && !on_outer(unq(k.0)) && !on_outer(unq(k.1)))
            .count()
    }

    /// Pair boundary vertices of two meshes by quantised WORLD position (0.1 mm, same key as
    /// `open_edge_count`) and return the WORST (smallest) normal dot over the shared vertices, plus the
    /// count of shared vertices. A small dot ⇒ a shading KINK at the LOD seam (the visible artifact). The
    /// worst dot over ALL shared positions bounds the seam; `None` if the meshes share no boundary vertex.
    fn worst_boundary_normal_dot(
        a: &ChunkMeshData,
        a_origin: Vec3,
        b: &ChunkMeshData,
        b_origin: Vec3,
    ) -> Option<(f32, usize)> {
        let q = |p: Vec3| {
            [
                (p.x as f64 * 1e4).round() as i64,
                (p.y as f64 * 1e4).round() as i64,
                (p.z as f64 * 1e4).round() as i64,
            ]
        };
        // World position → normalized normal for mesh A (first occurrence wins; co-located verts of one
        // mesh carry the same analytic normal, so any is representative).
        let mut a_norm: HashMap<[i64; 3], Vec3> = HashMap::new();
        for (p, n) in a.positions.iter().zip(&a.normals) {
            let world = Vec3::from(*p) + a_origin;
            a_norm.entry(q(world)).or_insert_with(|| Vec3::from(*n).normalize_or_zero());
        }
        let mut worst = 1.0f32;
        let mut shared = 0usize;
        for (p, n) in b.positions.iter().zip(&b.normals) {
            let world = Vec3::from(*p) + b_origin;
            if let Some(na) = a_norm.get(&q(world)) {
                let nb = Vec3::from(*n).normalize_or_zero();
                if na.length() < 0.5 || nb.length() < 0.5 {
                    continue; // skip degenerate normals (not a shading-continuity signal)
                }
                worst = worst.min(na.dot(nb));
                shared += 1;
            }
        }
        if shared == 0 { None } else { Some((worst, shared)) }
    }

    #[test]
    fn terrain_2to1_boundary_is_watertight_and_normal_continuous() {
        use crate::sdf_render::worldgen::layers::erosion::ErosionParams;
        use crate::sdf_render::worldgen::layers::height::{HeightLayer, HeightParams};
        use crate::sdf_render::worldgen::coord::LayerId;

        let seed = 7u64;
        // Resident height block around the origin (chunks (-2..2)² of HEIGHT_CHUNK_CELLS=128 m each) → the
        // ±64 m chunk footprints below are deep inside loaded coverage (the strict sampler can't miss).
        let clip = publish_eroded_terrain_clipmap(-2..=2, -2..=2, seed);

        // 2:1 boundary on the world-0 coarse lattice, plane x = 0. The FINE chunk (+X side) and the COARSE
        // chunk (−X side) span the SAME world size in EVERY axis (so the coarse +X transition face is FULLY
        // tiled by the fine −X face — no dangling open boundary, only the genuine seam to test). The coarse
        // chunk uses `sub` cells of `vsc`; the fine uses `2·sub` cells of `vsf = vsc/2`. The coarse +X face
        // (bit 1 = HighX) is the transition (toward the finer neighbour). All origins sit on the coarse
        // lattice so the transition-face samples coincide with the fine face (watertight by construction).
        let (vsf, vsc, sub) = (1.0f32, 2.0f32, 32u32);
        let span = sub as f32 * vsc; // 64 m — the common world edge of BOTH chunks
        let sub_f = sub * 2; // fine cells per axis (same world span at half the voxel)

        // Bracket the local eroded surface in Y so it crosses BOTH chunks. Sample the surface at the shared
        // face's centre and centre the (tall enough) Y window on it.
        let layer = HeightLayer::new(LayerId(0), HeightParams::default(), ErosionParams::default());
        let h_mid = layer.sample_world(0.0, (span as f64) * 0.5, seed).height;
        // Snap the chunk Y min to the coarse lattice so transition faces stay watertight, and make the Y
        // window tall enough to contain the surface's variation across the footprint.
        let y_min = ((h_mid - span * 0.5) / vsc).floor() * vsc;
        let of = Vec3::new(0.0, y_min, 0.0); // fine: x∈[0, 64], y∈[y_min, y_min+64], z∈[0, 64]
        let oc = Vec3::new(-span, y_min, 0.0); // coarse: x∈[-64, 0], same y/z span; +X (bit 1) = transition

        // The vertical band only matters for the non-rendering miss fallback / band; size it generously.
        let edit = terrain_edit_for_band(h_mid - 4.0 * span, h_mid + 4.0 * span);
        let edits_v = [edit];
        let idx = [0u32];

        // Bake: fine = LOD 0, regular; coarse = LOD 1 with +X (HighX = bit 1) transition. `terrain_only =
        // true` ⇒ analytic stored-gradient normals (the smooth normal the LOD seam is judged on). Pass the
        // published clipmap as `mesh_chunk`'s `terrain` param (installed as the per-bake thread-local).
        let fine = mesh_chunk(&edits_v, &idx, of, vsf, sub_f, 0, 0, false, Some(clip.clone()), true, true, 0, 0, 0.0)
            .expect("fine terrain chunk meshes");
        let coarse = mesh_chunk(&edits_v, &idx, oc, vsc, sub, 1 << 1, 1, false, Some(clip.clone()), true, true, 0, 0, 0.0)
            .expect("coarse terrain chunk meshes");

        // (a) GEOMETRIC: fine ∪ coarse must weld watertight across the shared x=0 seam — no gap / overlap.
        // The terrain is an OPEN patch, so the combined box's OUTER perimeter is legitimately open (the
        // surface exits there); only an INTERIOR open edge (on the x=0 seam) is a real cross-LOD crack.
        let mut all = chunk_tris(&fine, of);
        all.extend(chunk_tris(&coarse, oc));
        let bmin = Vec3::new(oc.x, y_min, 0.0); // combined box min: x=-64, y=y_min, z=0
        let bmax = Vec3::new(of.x + span, y_min + span, span); // x=+64, y=y_min+64, z=64
        let open = interior_open_edge_count(&all, bmin, bmax);
        assert_eq!(open, 0, "eroded terrain fine ∪ coarse must weld watertight across the 2:1 boundary");

        // (b) NORMAL CONTINUITY across the boundary: shared boundary vertices' baked normals must agree.
        // This catches the visible LOD seam (a shading kink, not a gap). The achieved tolerance below is
        // what the analytic gradient (Step 2) + the transition-cell mip widening (Step 3) reach on the real
        // eroded surface; it BOUNDS the seam and fails CI on regression.
        let (worst, shared) = worst_boundary_normal_dot(&fine, of, &coarse, oc)
            .expect("fine and coarse must share boundary vertices on the x=0 face");
        assert!(shared >= 4, "expected several shared boundary verts on the seam, got {shared}");
        println!("terrain 2:1 LOD seam: worst boundary normal dot {worst:.5} over {shared} shared verts");
        // ACHIEVED tolerance on the eroded terrain across the 2:1 LOD boundary. The SHARED-FACE normals
        // come from the clipmap's analytic STORED gradient (terrain_only ⇒ `terrain_normal`), and the
        // Transvoxel transition rule (`transition_sample_vs`) makes the coarse transition FACE sample the
        // SAME (finer) height mip as the abutting fine face — so both sides read the identical stored
        // gradient and the shared-boundary normals match to ~1.0 (measured worst dot 1.00000). With the
        // analytic gradient (Step 2) this is exact-by-construction at the seam; the gate is pinned just
        // below 1.0 to absorb mip-downsample float noise and catch any regression that re-introduces a
        // shared-face kink (e.g. a divergent mip select, or reverting terrain normals to a per-chunk FD).
        const TERRAIN_LOD_NORMAL_DOT_MIN: f32 = 0.999;
        assert!(
            worst >= TERRAIN_LOD_NORMAL_DOT_MIN,
            "LOD-boundary normal kink: worst shared-vertex normal dot {worst:.4} < tolerance \
             {TERRAIN_LOD_NORMAL_DOT_MIN} over {shared} shared verts (a visible shading seam regressed)"
        );

        // Clean up the process-global so other tests aren't perturbed.
        crate::sdf_render::worldgen::upload::set_cpu_height_clipmap(None);
    }

    #[test]
    fn terrain_geomorph_band_has_no_mip_step_kink() {
        // GEOMORPH SMOOTHNESS GUARD. Bake the COARSE chunk of the same 2:1 boundary and walk its baked
        // surface INWARD from the +X transition face across the transition band (one coarse voxel deep). The
        // hard-switch `transition_sample_vs` produced an abrupt mip STEP exactly one voxel in from the face
        // (the surface jumped from the fine mip to the coarse mip in a single cell) — the faint LOD-ring kink.
        // The smoothstep ramp morphs the effective voxel size continuously across the band, so consecutive
        // surface normals along the walk must stay nearly parallel: no abrupt step. We bin surface vertices by
        // their inward X distance from the face into thin slabs, average each slab's normal, and assert every
        // consecutive slab-to-slab normal dot stays above a tolerance — pinning the kink gone.
        use crate::sdf_render::worldgen::layers::erosion::ErosionParams;
        use crate::sdf_render::worldgen::layers::height::{HeightLayer, HeightParams};
        use crate::sdf_render::worldgen::coord::LayerId;

        let seed = 7u64;
        let clip = publish_eroded_terrain_clipmap(-2..=2, -2..=2, seed);
        let (vsc, sub) = (2.0f32, 32u32);
        let span = sub as f32 * vsc; // 64 m

        let layer = HeightLayer::new(LayerId(0), HeightParams::default(), ErosionParams::default());
        let h_mid = layer.sample_world(0.0, (span as f64) * 0.5, seed).height;
        let y_min = ((h_mid - span * 0.5) / vsc).floor() * vsc;
        let oc = Vec3::new(-span, y_min, 0.0); // coarse: x∈[-64,0]; +X (bit 1 = HighX) = transition face at x=0
        let edit = terrain_edit_for_band(h_mid - 4.0 * span, h_mid + 4.0 * span);
        let edits_v = [edit];
        let idx = [0u32];

        // Bake the coarse chunk WITH the +X transition (matching the watertight harness). terrain_only ⇒
        // analytic stored-gradient normals (the smooth normal the surface morph is judged on).
        let coarse = mesh_chunk(&edits_v, &idx, oc, vsc, sub, 1 << 1, 1, false, Some(clip.clone()), true, true, 0, 0, 0.0)
            .expect("coarse terrain chunk meshes");

        // Bin coarse-surface vertices by inward distance from the +X face (d = cmax.x − world.x = −world.x,
        // since cmax.x = 0) into half-voxel slabs across the band [0, 2·vsc] (band itself = vsc; sample a bit
        // past it so the step that USED to appear at d≈vsc is inside the walk). Average each slab's normal.
        let cmax_x = oc.x + span; // 0.0
        let slab = vsc * 0.5; // 1 m slabs
        let n_slabs = 4usize; // covers d ∈ [0, 2·vsc] = [0, 4 m]
        let mut acc = vec![Vec3::ZERO; n_slabs];
        let mut cnt = vec![0u32; n_slabs];
        for i in 0..coarse.positions.len() {
            let world = Vec3::from(coarse.positions[i]) + oc;
            let d = cmax_x - world.x; // inward distance from the +X transition face
            if d < 0.0 {
                continue;
            }
            let b = (d / slab) as usize;
            if b >= n_slabs {
                continue;
            }
            let n = Vec3::from(coarse.normals[i]);
            if n.length() < 0.5 {
                continue;
            }
            acc[b] += n.normalize();
            cnt[b] += 1;
        }
        // Consecutive non-empty slabs' average normals must stay nearly parallel (no abrupt step). The hard
        // switch made a step at the d≈vsc slab boundary; the ramp keeps every consecutive dot high. Tolerance
        // is what the smoothstep geomorph achieves on this real eroded surface (measured worst ≈ 0.999+); it
        // BOUNDS the kink and fails CI if a future change re-introduces the mip step.
        const GEOMORPH_STEP_DOT_MIN: f32 = 0.995;
        let avg: Vec<Option<Vec3>> = acc
            .iter()
            .zip(&cnt)
            .map(|(a, &c)| if c > 0 { Some((*a / c as f32).normalize_or_zero()) } else { None })
            .collect();
        let mut worst = 1.0f32;
        let mut steps = 0u32;
        let mut prev: Option<Vec3> = None;
        for slab_n in avg.iter().flatten() {
            if let Some(p) = prev
                && p.length() > 0.5
                && slab_n.length() > 0.5
            {
                worst = worst.min(p.dot(*slab_n));
                steps += 1;
            }
            prev = Some(*slab_n);
        }
        println!("terrain geomorph band: worst consecutive-slab normal dot {worst:.5} over {steps} steps");
        assert!(steps >= 2, "expected several populated slabs across the band, got {steps}");
        assert!(
            worst >= GEOMORPH_STEP_DOT_MIN,
            "geomorph mip-step kink: worst consecutive-slab normal dot {worst:.4} < {GEOMORPH_STEP_DOT_MIN} \
             over {steps} steps (the LOD-ring kink regressed — `transition_sample_vs` ramp broke)"
        );

        crate::sdf_render::worldgen::upload::set_cpu_height_clipmap(None);
    }

    /// Build + publish a tier-0 terrain clipmap whose nodes come from the layer's `generate` — the
    /// PRODUCTION path, including the band-limit finalize stage — NOT raw `sample_world` point samples
    /// (which is what `publish_eroded_terrain_clipmap` does). The triangle-quality harness below uses this
    /// so it measures the surface the renderer ACTUALLY meshes (the regression harnesses point-sample, so
    /// they never exercised the finalize filter — the gap this measures).
    fn publish_generated_terrain_clipmap(
        xrange: std::ops::RangeInclusive<i32>,
        zrange: std::ops::RangeInclusive<i32>,
        seed: u64,
        params: crate::sdf_render::worldgen::layers::height::HeightParams,
        erosion: crate::sdf_render::worldgen::layers::erosion::ErosionParams,
    ) -> Arc<HeightClipmap> {
        use crate::sdf_render::worldgen::artifact::ScalarField2D;
        use crate::sdf_render::worldgen::coord::{ChunkCoord, ChunkSize, LayerId};
        use crate::sdf_render::worldgen::layer::{GenCtx, GenOutput, Layer};
        use crate::sdf_render::worldgen::layers::height::{HEIGHT_CHUNK_CELLS, HeightLayer};
        use crate::sdf_render::worldgen::store::ArtifactStore;
        use crate::sdf_render::worldgen::upload::{
            build_height_clipmap, set_cpu_height_clipmap, set_cpu_terrain_offset,
        };
        let layer = HeightLayer::new(LayerId(0), params, erosion);
        let size = ChunkSize::new(HEIGHT_CHUNK_CELLS);
        let mut store = ArtifactStore::new();
        for cz in zrange.clone() {
            for cx in xrange.clone() {
                let coord = ChunkCoord::new(LayerId(0), IVec3::new(cx, 0, cz));
                let ctx = GenCtx { coord, seed, size };
                let mut out = GenOutput::default();
                layer.generate(&ctx, &mut out);
                let field = out.take::<ScalarField2D>(HeightLayer::OUTPUT).unwrap();
                store.insert(coord, field);
            }
        }
        let clip = Arc::new(build_height_clipmap(&store, &[HEIGHT_CHUNK_CELLS]));
        set_cpu_height_clipmap(Some(clip.clone()));
        set_cpu_terrain_offset(Vec2::ZERO);
        clip
    }

    /// TERRAIN-SURFACE bake correctness (Stages 2+3): (1) a COARSE chunk's central detail texel stores the
    /// slope `(dh/dx, dh/dz)` = `TerrainHifi::surface` (= raw `sample_world` grad) at that world XZ, packed
    /// through f16; (2) the central height texel stores the PRISTINE `sample_world` height; (3) the biome map
    /// is filled at `biome_res²`; (4) the DETAIL-NORMAL LOD gate ZERO-FILLS a FINE chunk's slope (but it
    /// STILL bakes — height/biome render everywhere), while a COARSE chunk has real slope; (5) a non-terrain
    /// chunk → `None`. Installs a matching per-bake snapshot (clipmap + hi-fi).
    #[test]
    fn terrain_surface_bake_samples_hifi_and_gates_detail_normal() {
        use crate::sdf_render::terrain_material::TerrainSurfaceBake;
        use crate::sdf_render::worldgen::coord::LayerId;
        use crate::sdf_render::worldgen::layers::erosion::ErosionParams;
        use crate::sdf_render::worldgen::layers::height::{HeightLayer, HeightParams};
        use crate::sdf_render::worldgen::upload::{
            TerrainHifi, set_bake_terrain, set_cpu_height_clipmap, set_cpu_terrain_hifi, set_cpu_terrain_offset,
        };
        use half::f16;

        let seed = 7u64;
        // Publish a clipmap over a few tier-0 chunks + the MATCHING tier-0 hi-fi sampler (same layer + seed).
        let clip =
            publish_generated_terrain_clipmap(-1..=2, -1..=2, seed, HeightParams::default(), ErosionParams::default());
        let layer = HeightLayer::new(LayerId(0), HeightParams::default(), ErosionParams::default());
        let hifi = Arc::new(TerrainHifi { layer, world_seed: seed });
        set_cpu_terrain_hifi(Some(hifi.clone()));
        set_cpu_terrain_offset(Vec2::ZERO);
        // Install the per-bake snapshot (the bake reads its hi-fi via the thread-local). Held for this scope.
        let _g = set_bake_terrain(Some(clip), Vec2::ZERO);

        // COARSE chunk (vs = 8 m > finest 2 m) at a non-trivial world origin → real detail + height + biome.
        let (res, bres) = (16u32, 4u32);
        let (vs, sub) = (8.0f32, 8u32);
        let chunk_world = sub as f32 * vs; // 64 m footprint
        let origin = Vec3::new(128.0, 0.0, 192.0);
        let bake = bake_terrain_surface(origin, chunk_world, vs, true, res, bres, 150.0)
            .expect("coarse terrain chunk must bake a terrain-surface payload");
        assert_eq!(bake.detail_res, res);
        assert_eq!(bake.biome_res, bres);
        assert_eq!(bake.chunk_size, chunk_world);
        assert_eq!(bake.chunk_min, Vec2::new(origin.x, origin.z));
        assert_eq!(bake.detail_texels.len(), (res * res * 4) as usize);
        assert_eq!(bake.height_texels.len(), (res * res * 4) as usize);
        assert_eq!(bake.biome_texels.len(), (bres * bres * 8) as usize);

        // The texel at (i, j) stores the surface at world ((i+0.5)·step, (j+0.5)·step) + origin.xz. Check a
        // central texel reproduces `hifi.surface` (height + slope) at that exact world XZ, through the pack.
        let step = (chunk_world / res as f32) as f64;
        let (i, j) = (res / 2, res / 2);
        let wx = origin.x as f64 + (i as f64 + 0.5) * step;
        let wz = origin.z as f64 + (j as f64 + 0.5) * step;
        let (sh, sx, sz) = hifi.surface(wx, wz);
        let off = ((j * res + i) * 4) as usize;
        let r = f16::from_bits(u16::from_le_bytes([bake.detail_texels[off], bake.detail_texels[off + 1]])).to_f32();
        let g = f16::from_bits(u16::from_le_bytes([bake.detail_texels[off + 2], bake.detail_texels[off + 3]])).to_f32();
        assert_eq!(r, f16::from_f32(sx).to_f32(), "texel dh/dx must match hifi.surface slope at the texel centre");
        assert_eq!(g, f16::from_f32(sz).to_f32(), "texel dh/dz must match hifi.surface slope at the texel centre");
        // SSOT: the slope packing matches `pack_slope` bit-for-bit.
        assert_eq!(&bake.detail_texels[off..off + 4], &TerrainSurfaceBake::pack_slope(sx, sz));
        // The height texel stores the depth-reference surface height (R32Float LE). That is the CLIPMAP
        // height the mesh is built from (the mottle fix — `depth = surf_h − mesh.y ≈ 0` on undug ground), NOT
        // the finer `sample_world` height, so compare against the SAME clipmap sample the bake reads.
        let hr = f32::from_le_bytes([
            bake.height_texels[off],
            bake.height_texels[off + 1],
            bake.height_texels[off + 2],
            bake.height_texels[off + 3],
        ]);
        let (clip_snap, _) = crate::sdf_render::worldgen::upload::bake_terrain_clipmap()
            .expect("the per-bake clipmap snapshot is installed");
        let expected_h = crate::sdf_render::worldgen::upload::try_sample_clipmap_lod(
            &clip_snap,
            bevy::math::DVec2::new(wx, wz),
            vs,
        )
        .map_or(sh, |node| node.height);
        assert_eq!(hr, expected_h, "height texel must store the clipmap-sampled depth-reference height");

        // DETAIL-NORMAL GATE: a FINE chunk (vs = 2 m = finest node spacing) STILL bakes (height/biome render
        // everywhere) but its detail-normal slope is ZERO-FILLED → geometry normal in the shader.
        let fine = bake_terrain_surface(origin, 2.0 * sub as f32, 2.0, true, res, bres, 150.0)
            .expect("fine terrain chunk still bakes height + biome (strata render everywhere)");
        assert!(
            fine.detail_texels.iter().all(|&b| b == 0),
            "fine LOD (vs ≤ 2 m) must zero-fill the detail-normal slope (geometry normal)"
        );
        // GATE: a non-terrain chunk gets no surface payload regardless of LOD.
        assert!(
            bake_terrain_surface(origin, chunk_world, vs, false, res, bres, 150.0).is_none(),
            "mixed/object chunk → no terrain-surface payload"
        );

        drop(_g);
        set_cpu_height_clipmap(None);
        set_cpu_terrain_hifi(None);
    }

    /// Smallest interior angle (degrees) of a triangle — 0 for a sliver/degenerate. Law of cosines on the
    /// three edge lengths.
    fn tri_min_angle_deg(a: Vec3, b: Vec3, c: Vec3) -> f32 {
        let ab = (b - a).length();
        let bc = (c - b).length();
        let ca = (a - c).length();
        let ang = |opp: f32, x: f32, y: f32| -> f32 {
            if x < 1e-9 || y < 1e-9 {
                return 0.0;
            }
            (((x * x + y * y - opp * opp) / (2.0 * x * y)).clamp(-1.0, 1.0)).acos().to_degrees()
        };
        ang(bc, ab, ca).min(ang(ca, ab, bc)).min(ang(ab, bc, ca))
    }

    /// TRIANGLE-QUALITY ROOT-CAUSE HARNESS. Bakes a grid of LOD-0 terrain chunks off the PRODUCTION
    /// (`generate`-filtered) surface and reports the min-interior-angle distribution + degenerate-triangle
    /// fraction, correlated with the local surface slope `|∇h|` at each sliver. This answers the open
    /// question driving the finalize-stage design: are the degenerate "spiky" ridges caused by SHARP
    /// CREASES (high-frequency, fixed by band-limiting) or by STEEP FLANKS (low-frequency steepness, which
    /// band-limiting can't fix)? It prints, for the ridge+erosion default AND a plain-fBm control, the
    /// sliver count and the mean `|∇h|` of slivers vs the mesh overall. `#[ignore]` — measurement; run with
    /// `--release --ignored --nocapture`.
    #[test]
    #[ignore = "triangle-quality measurement; run with --release --ignored --nocapture"]
    fn terrain_triangle_quality_report() {
        use crate::sdf_render::worldgen::layers::erosion::ErosionParams;
        use crate::sdf_render::worldgen::layers::height::HeightParams;
        use crate::sdf_render::worldgen::upload::{sample_clipmap_lod, set_cpu_height_clipmap};
        use bevy::math::DVec2;

        let seed = 7u64;
        // Bake a grid of LOD-0 chunks. Span a WIDE region (≥ the fBm base wavelength) so it contains real
        // ridge CRESTS (the ridge fold peaks where the base fBm crosses 0), not just one gentle hillside.
        let (vs, sub) = (1.0f32, 32u32);
        let span = sub as f32 * vs; // 32 m per chunk
        // 26·32 = 832 m of baked terrain (> the 1536 m base wavelength's half-period, so it spans a ridge
        // crest). Stays inside the published 8×8 height-chunk ring (−1..=6 = world [−128, 896) with margin
        // on BOTH sides for the bake's ~1 m apron) — the ring is toroidal with `HEIGHT_RING_CHUNKS=8`, so
        // chunk 7 would alias slot −1; the block must be ≤ 8 chunks per axis.
        let chunks = 26i32;

        // Curvature (Laplacian) of the height field at world (x,z) via central differences — the CREST
        // detector: |∇²h| is large at a sharp ridge crest, ~0 on a smooth slope. Uses the same band-limited
        // clipmap the bake reads, so it measures the crest the MESHER actually sees.
        let curvature = |clip: &HeightClipmap, x: f32, z: f32| -> f32 {
            use crate::sdf_render::worldgen::upload::sample_clipmap_lod;
            let e = 2.0f32;
            let h = |dx: f32, dz: f32| {
                sample_clipmap_lod(clip, DVec2::new((x + dx) as f64, (z + dz) as f64), vs).height
            };
            ((h(e, 0.0) + h(-e, 0.0) + h(0.0, e) + h(0.0, -e) - 4.0 * h(0.0, 0.0)) / (e * e)).abs()
        };

        let report = |label: &str, params: HeightParams, erosion: ErosionParams| {
            let clip = publish_generated_terrain_clipmap(-1..=6, -1..=6, seed, params, erosion);
            let edit = terrain_edit_for_band(-2000.0, 2000.0);
            let edits_v = [edit];
            let idx = [0u32];

            let mut angles: Vec<f32> = Vec::new();
            // Per-triangle (min_angle°, normal_spread°, |∇²h|): min_angle = sliver test; normal_spread =
            // max pairwise angle between the triangle's 3 vertex normals (the SHADING-discontinuity metric —
            // a serrated crest has wildly disagreeing vertex normals); |∇²h| = crest-vs-slope classifier.
            let mut tri_data: Vec<(f32, f32, f32)> = Vec::new();
            // SPIKE metric: max per-vertex |vertex.y − true_surface_h(vertex.xz)| — distinguishes flat
            // slivers (on the surface, ~0 dev) from actual displaced/spiked vertices (large dev).
            let mut max_dev = 0.0f32;
            for cz in 0..chunks {
                for cx in 0..chunks {
                    let ox = cx as f32 * span;
                    let oz = cz as f32 * span;
                    // Centre the Y window on the local surface so it crosses the chunk.
                    let h_mid = sample_clipmap_lod(
                        &clip,
                        DVec2::new((ox + span * 0.5) as f64, (oz + span * 0.5) as f64),
                        vs,
                    )
                    .height;
                    let y_min = ((h_mid - span * 0.5) / vs).floor() * vs;
                    let origin = Vec3::new(ox, y_min, oz);
                    let Some(data) = mesh_chunk(&edits_v, &idx, origin, vs, sub, 0, 0, false, Some(clip.clone()), true, true, 0, 0, 0.0)
                    else {
                        continue;
                    };
                    for &p in &data.positions {
                        let w = Vec3::from(p) + origin;
                        let h = sample_clipmap_lod(&clip, DVec2::new(w.x as f64, w.z as f64), vs).height;
                        max_dev = max_dev.max((w.y - h).abs());
                    }
                    for t in data.indices.chunks_exact(3) {
                        let vi = [t[0] as usize, t[1] as usize, t[2] as usize];
                        let pos = |k: usize| Vec3::from(data.positions[vi[k]]) + origin;
                        let (a, b, c) = (pos(0), pos(1), pos(2));
                        let ang = tri_min_angle_deg(a, b, c);
                        angles.push(ang);
                        // Max pairwise angle between the 3 vertex normals (degrees) = shading discontinuity.
                        let nrm = |k: usize| Vec3::from(data.normals[vi[k]]).normalize_or_zero();
                        let (n0, n1, n2) = (nrm(0), nrm(1), nrm(2));
                        let ang2 = |x: Vec3, y: Vec3| x.dot(y).clamp(-1.0, 1.0).acos().to_degrees();
                        let nspread = ang2(n0, n1).max(ang2(n1, n2)).max(ang2(n2, n0));
                        let cen = (a + b + c) / 3.0;
                        tri_data.push((ang, nspread, curvature(&clip, cen.x, cen.z)));
                    }
                }
            }
            set_cpu_height_clipmap(None);

            let n = angles.len().max(1);
            let lt5 = angles.iter().filter(|&&a| a < 5.0).count();
            // CREST vs SLOPE split: a triangle is "crest" if its centroid curvature is in the top quartile.
            // Report BOTH sliver% (geometry) AND mean normal-spread° (shading) in each bin. The user's
            // complaint is shading: normal-spread should be MUCH higher on crests, and the band-limit must
            // bring it down.
            let mut curvs: Vec<f32> = tri_data.iter().map(|&(_, _, c)| c).collect();
            curvs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q75 = curvs.get(curvs.len() * 3 / 4).copied().unwrap_or(0.0);
            let (mut crest_n, mut crest_sliver, mut crest_nsp) = (0usize, 0usize, 0.0f64);
            let (mut slope_n, mut slope_sliver, mut slope_nsp) = (0usize, 0usize, 0.0f64);
            let mut worst_nsp = 0.0f32;
            for &(ang, nsp, c) in &tri_data {
                worst_nsp = worst_nsp.max(nsp);
                if c >= q75 {
                    crest_n += 1;
                    crest_nsp += nsp as f64;
                    if ang < 5.0 {
                        crest_sliver += 1;
                    }
                } else {
                    slope_n += 1;
                    slope_nsp += nsp as f64;
                    if ang < 5.0 {
                        slope_sliver += 1;
                    }
                }
            }
            let pct = |a: usize, b: usize| 100.0 * a as f32 / b.max(1) as f32;
            let mean = |s: f64, c: usize| s / c.max(1) as f64;
            let crest_mean = mean(crest_nsp, crest_n);
            println!(
                "TRI-QUALITY [{label}]: tris={n} slivers<5°={:.2}% | \
                 CREST(top-25%-curv): sliver%={:.2} normal_spread={crest_mean:.1}° | \
                 SLOPE: sliver%={:.2} normal_spread={:.1}° | worst_normal_spread={worst_nsp:.1}° \
                 (q75|∇²h|={q75:.3}) | max_surface_dev={max_dev:.3}m",
                pct(lt5, n),
                pct(crest_sliver, crest_n),
                pct(slope_sliver, slope_n),
                mean(slope_nsp, slope_n),
            );
            (crest_mean, worst_nsp) // (mean crest normal spread°, worst triangle normal spread°)
        };

        // Default (ridge + erosion ON) vs plain fBm control (no ridge, no erosion).
        report("ridge+erosion", HeightParams::default(), ErosionParams::default());
        report(
            "plain-fbm",
            HeightParams { ridge: 0.0, ..Default::default() },
            ErosionParams { enabled: false, ..Default::default() },
        );
        // DENSE SHARP RIDGES: full ridge fold at a short wavelength (≈256 m) GUARANTEES several sharp crests
        // inside the 832 m window — the crest path the default-seed region happened to miss. Compare the
        // RAW crest (band_limit=0) against the band-limited finalize stage: the crest `normal_spread°` (the
        // serrated-shading metric) must drop sharply, proving the fix.
        let dense = |bl: f32| HeightParams {
            ridge: 1.0,
            base_freq: 1.0 / 256.0,
            amplitude: 100.0,
            octaves: 4,
            band_limit: bl,
            ..Default::default()
        };
        let off = ErosionParams { enabled: false, ..Default::default() };
        // RADIUS SWEEP — does crest serration keep dropping with the band-limit radius (→ a tuning/default
        // issue, crank the slider) or plateau (→ structural: bilinear-grid interp or inherent apex)?
        let (raw_crest, raw_worst) = report("dense-ridge bl=0", dense(0.0), off);
        report("dense-ridge bl=2", dense(2.0), off);
        report("dense-ridge bl=4", dense(4.0), off);
        let (bl_crest, bl_worst) = report("dense-ridge bl=8", dense(8.0), off);
        // REGRESSION GUARD: the band-limit finalize stage must measurably reduce crest serration (both the
        // mean crest normal-spread and the worst single-triangle spread). If a future change breaks the
        // band-limit (or reverts it), these fail.
        assert!(
            bl_crest < raw_crest * 0.95,
            "band-limit must reduce mean crest normal-spread: bl={bl_crest:.2}° vs raw={raw_crest:.2}°"
        );
        assert!(
            bl_worst < raw_worst,
            "band-limit must reduce worst crest normal-spread: bl={bl_worst:.1}° vs raw={raw_worst:.1}°"
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
        let data = mesh_chunk(&edits, &[0, 1], origin, vs, sub, 0, 0, false, None, false, false, 0, 0, 0.0).expect("merged shape meshes");
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
        let coarse = mesh_chunk(&edits, &[0], oc, 0.2, 28, 1 << 1, 1, false, None, false, false, 0, 0, 0.0).expect("coarse+transition meshes");
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
