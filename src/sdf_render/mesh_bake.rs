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
    edits, gather_sorted_edits, SdfGridConfig, SdfVolume, VolumeQueryData,
};

/// Max NEW meshing tasks spawned per frame (the pool runs them concurrently; this bounds the spawn
/// burst when a large region enters at once).
const MAX_NEW_TASKS_PER_FRAME: usize = 256;

/// Hash-mix multiplier for folding the "Rebake all" epoch into a chunk's content hash.
const EPOCH_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// Raw mesh data produced off-thread by a meshing task (turned into a `Mesh` asset on the main thread).
struct ChunkMeshData {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
    /// Dominant material id (at the surface centroid) — selects the chunk's `StandardMaterial` PBR params.
    material: u16,
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
}

impl Default for MeshBakeConfig {
    fn default() -> Self {
        // 4 → 64 bricks/chunk; matches the GPU-atlas chunk size. Tunable 1..=8 via the panel; smaller K
        // = faster rounds (more real-time), larger K = fewer draws but heavier per-chunk re-bakes.
        Self { chunk_bricks: 4 }
    }
}

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

/// World-space AABB of a chunk (`K×K×K` finest bricks, LOD 0).
fn chunk_aabb(key: BrickKey, config: &SdfGridConfig, k: u32) -> Aabb3d {
    let min = config.brick_min_world(key.coord, 0);
    let cw = k as f32 * config.brick_world_size(0);
    Aabb3d::from_min_max(min, min + Vec3::splat(cw))
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
fn chunks_in_aabb(aabb: &Aabb3d, config: &SdfGridConfig, k: u32, out: &mut HashSet<BrickKey>) {
    let cw = k as f32 * config.brick_world_size(0); // chunk world size
    let stride = k as i32 * config.cell_stride(); // chunk voxel stride
    let min = Vec3::from(aabb.min) - Vec3::splat(cw);
    let max = Vec3::from(aabb.max) + Vec3::splat(cw);
    let lo = (min / cw).floor();
    let hi = (max / cw).floor();
    // Guard against a pathologically large edit AABB exploding the enumeration; such cases need LOD /
    // camera-radius culling (Phase 3), not naive finest meshing. (In chunk units it trips far later.)
    let count = (hi.x - lo.x + 1.0) as i64 * (hi.y - lo.y + 1.0) as i64 * (hi.z - lo.z + 1.0) as i64;
    if count > 200_000 {
        return;
    }
    for ix in lo.x as i32..=hi.x as i32 {
        for iy in lo.y as i32..=hi.y as i32 {
            for iz in lo.z as i32..=hi.z as i32 {
                out.insert(BrickKey::new(0, IVec3::new(ix, iy, iz) * stride));
            }
        }
    }
}

/// Sample + Surface-Nets one chunk (runs off-thread on the task pool). Returns `None` for an empty chunk
/// (no surface crossing). `indices` are the edits (into the CSG-sorted list) that overlap this chunk —
/// exactly the set the chunk's content hash was taken over, so geometry and hash always agree. `edge` is
/// the padded grid edge in samples (`K·cell_stride + 2`).
fn mesh_chunk(
    edits: &[edits::ResolvedEdit],
    indices: &[u32],
    appearances: &[MatAppearance],
    grid_origin: Vec3,
    vs: f32,
    edge: u32,
) -> Option<ChunkMeshData> {
    let band = 4.0 * vs;
    let mut sdf = vec![0.0f32; (edge * edge * edge) as usize];
    // Fill in the shape's linear order (i = x + y·edge + z·edge²) with x innermost, incrementing `i` —
    // avoids a per-voxel `RuntimeShape::delinearize` (runtime strides can't strength-reduce the div/mod).
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
    let shape = RuntimeShape::<u32, 3>::new([edge, edge, edge]);
    let mut buffer = SurfaceNetsBuffer::default();
    // TODO(perf): pool the sample buffer + SurfaceNetsBuffer per `edge` to avoid per-task allocation.
    surface_nets(&sdf, &shape, [0, 0, 0], [edge - 1, edge - 1, edge - 1], &mut buffer);
    if buffer.positions.is_empty() {
        return None;
    }
    let positions: Vec<[f32; 3]> =
        buffer.positions.iter().map(|p| [p[0] * vs, p[1] * vs, p[2] * vs]).collect();
    // Per-vertex base COLOUR = the resolved material's LINEAR base colour. Real PBR lighting shades it
    // (the chunk's StandardMaterial carries the dominant material's metallic/roughness/emissive); the
    // per-vertex base still varies for the rare mixed-material chunk.
    let colors: Vec<[f32; 4]> = buffer
        .positions
        .iter()
        .map(|p| {
            let world = grid_origin + Vec3::new(p[0], p[1], p[2]) * vs;
            let mid = edits::fold_csg(edits, world).material_id as usize;
            let a = appearances.get(mid).copied().unwrap_or(DEFAULT_APPEARANCE);
            [a.base[0], a.base[1], a.base[2], 1.0]
        })
        .collect();
    // Dominant material (at the surface centroid) → the chunk's StandardMaterial PBR params. Off-the-shelf
    // StandardMaterial is per-mesh, so metallic/roughness/emissive use the dominant material.
    let mut centroid = Vec3::ZERO;
    for p in &buffer.positions {
        centroid += Vec3::new(p[0], p[1], p[2]);
    }
    centroid = grid_origin + (centroid / buffer.positions.len().max(1) as f32) * vs;
    let material = edits::fold_csg(edits, centroid).material_id;
    Some(ChunkMeshData { positions, normals: buffer.normals, colors, indices: buffer.indices, material })
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
    let cw = k as f32 * config.brick_world_size(0);
    let min = config.brick_min_world(key.coord, 0);
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
    chunk_meshes: Query<(Entity, &ChunkMesh)>,
    mut states: ResMut<ChunkStates>,
    mut rebuild: ResMut<MeshBakeRebuild>,
    mut stats: ResMut<MeshBakeStats>,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    // Lit `StandardMaterial` per material id (base WHITE — per-vertex base comes from the vertex COLOUR —
    // plus the material's metallic/roughness/emissive). Cleared + rebuilt when material appearances change.
    mut mat_cache: Local<HashMap<u16, Handle<StandardMaterial>>>,
    // "Rebake all" epoch: bumped on the panel button, mixed into every hash to force one full re-bake.
    mut epoch: Local<u64>,
    // Last frame's chunk size, to detect a live K change (which changes the whole key set / stride).
    mut prev_k: Local<u32>,
    // Last frame's material-appearance hash, to detect a material COLOUR edit (which the per-chunk
    // content hash — keyed on material *id* — wouldn't otherwise catch).
    mut prev_mat_hash: Local<u64>,
    // The in-progress bake round's frozen edit snapshot.
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
        *prev_k = k;
        return;
    }

    // K changed live (slider): the key set is at a different stride now, so every old-stride chunk mesh
    // is stale. Despawn all + clear state + abort any round for a clean swap.
    if *prev_k != 0 && *prev_k != k {
        for (e, _) in &chunk_meshes {
            commands.entity(e).despawn();
        }
        states.0.clear();
        round.edits = None;
        round.aabbs.clear();
    }
    *prev_k = k;

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
    let mat_changed = *prev_mat_hash != 0 && *prev_mat_hash != mat_hash;
    // "Rebake all" (button) OR a material-appearance change bumps a global epoch mixed into every chunk
    // hash → every hash changes once → full re-bake.
    if std::mem::replace(&mut rebuild.0, false) || mat_changed {
        *epoch = epoch.wrapping_add(1);
    }
    if mat_changed {
        mat_cache.clear(); // rebuild StandardMaterials with the new params
    }
    *prev_mat_hash = mat_hash;
    let epoch_mix = epoch.wrapping_mul(EPOCH_MIX);

    let vs = config.voxel_size_at(0); // finest (LOD-0) voxel size
    // A chunk samples its cell span + a 1-voxel apron each side; the apron is one sample regardless of K.
    let apron = Vec3::splat(vs);

    // RESIDENCY (candidate chunks): the chunks within reach of the CURRENT edits — straight from the edit
    // AABBs, NO dependency on the editor-scene-gated GPU SDF atlas, so the bake runs in any scene.
    let mut resident: HashSet<BrickKey> = HashSet::new();
    for a in &edit_aabbs {
        chunks_in_aabb(a, &config, k, &mut resident);
    }

    // The padded sampled AABB of a chunk (cell span + 1-voxel apron).
    let chunk_sampled = |key: BrickKey| -> Aabb3d {
        let b = chunk_aabb(key, &config, k);
        Aabb3d::from_min_max(Vec3::from(b.min) - apron, Vec3::from(b.max) + apron)
    };

    // Current content hash for every resident chunk (over the LIVE edits) — drives "is the displayed mesh
    // out of date" (i.e. is a NEW round needed). The in-round target hashes are a frozen subset of these.
    let mut current_hashes: HashMap<BrickKey, u64> = HashMap::with_capacity(resident.len());
    {
        let mut idx: Vec<u32> = Vec::new();
        for &key in &resident {
            cull_into(&edit_aabbs, &chunk_sampled(key), &mut idx);
            let base = if idx.is_empty() { 0 } else { edits::bake_content_hash(&edits_arc, &idx) };
            current_hashes.insert(key, base ^ epoch_mix);
        }
    }
    stats.edits = n_edits;
    stats.resident = resident.len();

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
        for (key, st) in states.0.iter_mut() {
            let Some(sb) = st.staged.take() else {
                continue;
            };
            if let Some(old) = st.entity.take() {
                commands.entity(old).despawn();
            }
            st.entity = sb.data.map(|data| {
                // Lit StandardMaterial for this chunk's dominant material (cached by id). base WHITE so
                // the per-vertex base COLOUR rules; metallic/roughness/emissive from the registry.
                let mat = mat_cache
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
                            ..default()
                        })
                    })
                    .clone();
                let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
                    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, data.positions)
                    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals)
                    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, data.colors)
                    .with_inserted_indices(Indices::U32(data.indices));
                // Apron offset: SN sample 0 is one voxel BEFORE the chunk min — MUST stay exactly
                // `brick_min_world(coord,0) - vs`, or the chunk shifts a voxel and every seam cracks.
                let origin = config.brick_min_world(key.coord, 0) - Vec3::splat(vs);
                commands
                    .spawn((
                        Mesh3d(mesh_assets.add(mesh)),
                        MeshMaterial3d(mat),
                        Transform::from_translation(origin),
                        ChunkMesh(*key),
                        Name::new("SDF Chunk Mesh"),
                    ))
                    .id()
            });
            st.displayed_hash = st.target_hash;
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
            cull_into(&round.aabbs, &chunk_sampled(key), &mut idx);
            // NARROW-BAND CULL: skip chunks with no surface crossing (interior/exterior of a solid) for a
            // single SDF eval instead of a full edge³ bake — the big win for large objects. Commit them
            // empty (no task, no budget) so the round still settles.
            if !chunk_has_surface(&round_edits, &idx, &config, k, key, vs) {
                st.staged = Some(StagedBake { data: None });
                continue;
            }
            let grid_origin = config.brick_min_world(key.coord, 0) - Vec3::splat(vs);
            let edits = round_edits.clone();
            let indices = idx.clone();
            let apps = appearances.clone();
            st.task = Some(
                pool.spawn(async move { mesh_chunk(&edits, &indices, &apps, grid_origin, vs, edge) }),
            );
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
