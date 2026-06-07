//! **Phase 1** of the SDF→mesh bake (see `docs/MESH_BAKE_PLAN.md`): a residency-driven, **async**,
//! **incremental** per-brick Surface Nets bake.
//!
//! For every **finest-resident** brick (`SdfAtlas`, LOD 0) we sample the CPU SDF (`edits::fold_csg`,
//! no GPU readback) over the brick's 9³ padded voxel grid on the [`AsyncComputeTaskPool`], run
//! [`fast_surface_nets`] off the main thread, then (back on the main thread) build a stock Bevy
//! `Mesh3d`. As the clipmap window moves, newly-resident bricks are queued and departed ones removed.
//!
//! **Update protocol (per brick):** request → wait → receive → request again — exactly ONE pending
//! update per brick at a time. Staleness is a CONTENT HASH: each brick hashes the edits overlapping it
//! (`edits::bake_content_hash`, the same key the GPU bake scheduler uses); a brick re-bakes iff its
//! current hash differs from the displayed mesh's. The displayed mesh is KEPT until the new task
//! completes and is then atomically swapped — we never cancel a task or despawn early, so a moving
//! object doesn't flicker. Because residency and staleness derive from the SAME overlap test, a moved
//! edit re-bakes every brick it enters OR leaves automatically — no separate dirty region to drift out
//! of sync, so stale/ghost geometry is structurally impossible (residency departure is still swept by a
//! key-stamped reaper as the closed loop on the other axis).
//!
//! Same-LOD seams are crack-free for free: adjacent bricks share their boundary sample plane (apron).
//!
//! VIEWING: use the **Mesh Bake** editor panel ([`mesh_bake_panel`]) to toggle the SDF render off and
//! reveal these meshes (+ wireframe / rebake / counts).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::math::bounding::Aabb3d;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};
use fast_surface_nets::{surface_nets, SurfaceNetsBuffer};
use ndshape::{ConstShape, ConstShape3u32};

use crate::sdf_render::atlas::BrickKey;
use crate::sdf_render::{
    edits, gather_sorted_edits, SdfGridConfig, SdfVolume, VolumeQueryData,
};

/// Padded brick grid edge: a finest brick spans `cell_stride` (= `BRICK_EDGE - 1` = 7) cells; we add
/// one apron sample each side so neighbours share a boundary plane → `7 + 2 = 9` samples per edge.
const PAD: u32 = 9;
type BrickShape = ConstShape3u32<9, 9, 9>;

/// Max NEW meshing tasks spawned per frame (the pool runs them concurrently; this bounds the spawn
/// burst when a large clipmap shell enters at once).
const MAX_NEW_TASKS_PER_FRAME: usize = 256;

/// Raw mesh data produced off-thread by a meshing task (turned into a `Mesh` asset on the main thread).
struct BrickMeshData {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

/// Marks a baked brick mesh entity AND stamps it with its brick key, so departed/orphaned meshes can
/// be reaped by a query (residency = the single source of truth) regardless of `BrickStates`
/// bookkeeping. This is what makes ghost meshes impossible: the entity carries its own identity.
#[derive(Component)]
struct BrickMesh(BrickKey);

/// Per-brick bake state. ONE pending update per brick: a brick requests a (re)mesh only when its
/// CURRENT content hash differs from the displayed mesh's and `task.is_none()`; the displayed `entity` is
/// kept until that `task` completes and is then atomically replaced — never cancelled / never despawned
/// early. Staleness is a pure hash comparison (no "dirty" bookkeeping to drift), which is what makes a
/// stale remnant structurally impossible.
#[derive(Default)]
struct BrickState {
    /// Currently displayed mesh (None = meshed-empty, or not meshed yet).
    entity: Option<Entity>,
    /// The single in-flight meshing task (the one pending update), if any.
    task: Option<Task<Option<BrickMeshData>>>,
    /// Content hash of the inputs the DISPLAYED mesh was baked from (`edits::bake_content_hash` of the
    /// edits overlapping this brick, ⊕ epoch). The brick is up to date iff this equals the brick's current
    /// content hash; otherwise it re-bakes. 0 = nothing displayed yet / displayed-empty at epoch 0.
    displayed_hash: u64,
    /// Content hash the in-flight `task` is baking — becomes `displayed_hash` when it lands.
    task_hash: u64,
}

/// Per-finest-resident-brick bake state.
#[derive(Resource, Default)]
struct BrickStates(HashMap<BrickKey, BrickState>);

/// Set by the editor panel's "Rebake all" button to force a full re-mesh.
#[derive(Resource, Default)]
struct MeshBakeRebuild(bool);

/// Live diagnostics for the editor panel (helps tell "ghost the reaper missed" from "residency wrong").
#[derive(Resource, Default)]
struct MeshBakeStats {
    /// Number of SDF volumes (edits) gathered this frame — if this climbs while dragging, the editor
    /// is spawning extra volumes (the gather sees more than the authored set).
    edits: usize,
    /// Finest bricks the edits currently occupy (the resident set the reaper keeps).
    resident: usize,
    /// Brick-mesh entities despawned by the REAP pass this frame.
    reaped: usize,
    /// Set by the panel's "Capture diagnostics" button; consumed by the system, which fills `dump`.
    capture: bool,
    /// Copy-paste-able diagnostic dump (volumes + ghost meshes that survived the reaper) — filled when
    /// `capture` is requested. The panel shows it in a selectable box + a Copy button.
    dump: String,
}

/// Mesh-bake plugin. Added in `main.rs`. The bake itself is editor- AND scene-INDEPENDENT (it runs
/// every frame and bakes SDF world edits in gameplay too); only the optional debug panel is editor-only.
pub struct MeshBakePlugin;

impl Plugin for MeshBakePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BrickStates>()
            .init_resource::<MeshBakeRebuild>()
            .init_resource::<MeshBakeStats>()
            // Editor- AND scene-INDEPENDENT: runs every frame so SDF world edits are baked during
            // gameplay too. It self-determines which bricks to mesh from the SDF edits (no dependency
            // on the editor-scene-gated GPU SDF atlas) and no-ops when no SDF volumes exist — which
            // also clears the meshes when an SDF scene is left.
            .add_systems(Update, mesh_resident_bricks);
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

/// World-space AABB of a finest (LOD-0) brick.
fn brick_aabb(key: BrickKey, config: &SdfGridConfig) -> Aabb3d {
    let min = config.brick_min_world(key.coord, 0);
    let bw = config.brick_world_size(0);
    Aabb3d::from_min_max(min, min + Vec3::splat(bw))
}

/// Enumerate the finest (LOD-0) bricks overlapping `aabb` (padded by one brick so surface bricks at
/// the boundary are caught) into `out`. This is how the bake locates geometry WITHOUT the GPU atlas:
/// brick coords are multiples of `cell_stride` in voxel units, so a brick edge spans `brick_world_size`
/// in world space and sits at `idx * brick_world_size`.
fn bricks_in_aabb(aabb: &Aabb3d, config: &SdfGridConfig, out: &mut HashSet<BrickKey>) {
    let bw = config.brick_world_size(0);
    let cs = config.cell_stride();
    let min = Vec3::from(aabb.min) - Vec3::splat(bw);
    let max = Vec3::from(aabb.max) + Vec3::splat(bw);
    let lo = (min / bw).floor();
    let hi = (max / bw).floor();
    // Guard against a pathologically large edit AABB (e.g. a big heightmap) exploding the enumeration;
    // such cases need LOD / camera-radius culling (Phase 3), not naive finest meshing.
    let count = (hi.x - lo.x + 1.0) as i64 * (hi.y - lo.y + 1.0) as i64 * (hi.z - lo.z + 1.0) as i64;
    if count > 200_000 {
        return;
    }
    for ix in lo.x as i32..=hi.x as i32 {
        for iy in lo.y as i32..=hi.y as i32 {
            for iz in lo.z as i32..=hi.z as i32 {
                out.insert(BrickKey::new(0, IVec3::new(ix, iy, iz) * cs));
            }
        }
    }
}

/// Sample + Surface-Nets one brick (runs off-thread on the task pool). Returns `None` for an empty
/// brick (no surface crossing). `indices` are the edits (into the CSG-sorted list) that overlap this
/// brick — exactly the set the brick's content hash was taken over, so geometry and hash always agree.
fn mesh_brick(
    edits: &[edits::ResolvedEdit],
    indices: &[u32],
    grid_origin: Vec3,
    vs: f32,
) -> Option<BrickMeshData> {
    let band = 4.0 * vs;
    let mut sdf = vec![0.0f32; BrickShape::SIZE as usize];
    for i in 0..BrickShape::SIZE {
        let [x, y, z] = BrickShape::delinearize(i);
        let p = grid_origin + Vec3::new(x as f32, y as f32, z as f32) * vs;
        // Sub-voxel iso-shift so no sample lands exactly on dist == 0 (Surface Nets treats 0 as
        // "outside", dropping a cell — a pinhole at grid-aligned features).
        sdf[i as usize] = (edits::fold_csg_dist_indexed(edits, indices, p) - 1e-3).clamp(-band, band);
    }
    let mut buffer = SurfaceNetsBuffer::default();
    surface_nets(&sdf, &BrickShape {}, [0, 0, 0], [PAD - 1, PAD - 1, PAD - 1], &mut buffer);
    if buffer.positions.is_empty() {
        return None;
    }
    let positions = buffer.positions.iter().map(|p| [p[0] * vs, p[1] * vs, p[2] * vs]).collect();
    let colors = buffer
        .normals
        .iter()
        .map(|n| [n[0] * 0.5 + 0.5, n[1] * 0.5 + 0.5, n[2] * 0.5 + 0.5, 1.0])
        .collect();
    Some(BrickMeshData { positions, normals: buffer.normals, colors, indices: buffer.indices })
}

/// Content-hash-driven, async, per-resident-brick Surface Nets bake — ROBUST BY CONSTRUCTION.
///
/// A brick's displayed mesh is a pure function of the edits overlapping it. Each frame we recompute the
/// brick's content hash (`edits::bake_content_hash` of its overlapping edits — the SAME primitive the GPU
/// bake scheduler uses, quantized so `GlobalTransform`'s sub-ULP jitter doesn't churn it) and re-bake iff
/// it differs from the displayed mesh's hash. There is NO separate "dirty region" to keep in sync with
/// residency: residency (which bricks exist) and staleness (when to re-bake) both derive from ONE overlap
/// test, so they cannot disagree. This structurally eliminates the ghost/remnant bug class — previously a
/// brick baked via residency's 1-brick pad was never re-dirtied because the raw swept AABB never reached it.
///
/// Update protocol per brick: request → wait → receive → request — exactly ONE pending bake per brick.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn mesh_resident_bricks(
    mut commands: Commands,
    volumes: Query<VolumeQueryData, With<SdfVolume>>,
    config: Res<SdfGridConfig>,
    brick_meshes: Query<(Entity, &BrickMesh)>,
    mut states: ResMut<BrickStates>,
    mut rebuild: ResMut<MeshBakeRebuild>,
    mut stats: ResMut<MeshBakeStats>,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut material: Local<Option<Handle<StandardMaterial>>>,
    // "Rebake all" epoch: bumped on the panel button, mixed into every hash to force one full re-bake.
    mut epoch: Local<u64>,
) {
    // Resolve the CSG edits (SdfOrder-sorted) + each volume's world AABB (the AABB already includes the
    // smoothing margin). The scene has a handful; per-brick BVH culling of the fold is a later scaling step.
    let gathered = gather_sorted_edits(&volumes);
    if gathered.is_empty() {
        // Scene unloaded — drop everything (tasks cancel on drop).
        if !states.0.is_empty() {
            for (e, _) in &brick_meshes {
                commands.entity(e).despawn();
            }
            states.0.clear();
        }
        return;
    }
    let n_edits = gathered.len();
    let mut edit_aabbs: Vec<Aabb3d> = Vec::with_capacity(n_edits);
    let mut edit_vec: Vec<edits::ResolvedEdit> = Vec::with_capacity(n_edits);
    for g in &gathered {
        edit_aabbs.push(g.aabb);
        edit_vec.push(g.edit.clone());
    }
    let edits_arc = Arc::new(edit_vec);

    // "Rebake all" mixes a bumped epoch into every brick hash → every hash changes once → full re-bake.
    if std::mem::replace(&mut rebuild.0, false) {
        *epoch = epoch.wrapping_add(1);
    }
    let epoch = *epoch;

    let vs = config.voxel_size_at(0); // finest (LOD-0) voxel size
    debug_assert_eq!(config.cell_stride() as u32 + 2, PAD, "BrickShape must be cell_stride + 2");
    // A brick samples its cell span + a 1-voxel apron each side; an edit whose AABB reaches that region
    // can put a surface in the brick. Conservative — an edge-touching edit that contributes nothing only
    // costs a harmless identical re-bake if it later moves.
    let apron = Vec3::splat(vs);

    // RESIDENCY (candidate bricks): the finest bricks within reach of the edits — straight from the edit
    // AABBs, NO dependency on the editor-scene-gated GPU SDF atlas, so the bake runs in any scene / during
    // gameplay. The reaper is keyed off this set; the per-brick content hash drives re-bake.
    let mut resident: HashSet<BrickKey> = HashSet::new();
    for a in &edit_aabbs {
        bricks_in_aabb(a, &config, &mut resident);
    }

    // THE SSOT: a brick's bake-cache key = hash of exactly the edits overlapping its sampled region, in
    // CSG order, quantized (jitter-stable). Empty set → 0 (no geometry). `epoch` lets "Rebake all"
    // invalidate everything. This one function decides BOTH "does this brick have geometry" and "is the
    // displayed mesh current" — so the two can never diverge (the root of every prior remnant bug).
    let brick_content_hash = |key: BrickKey, idx: &mut Vec<u32>| -> u64 {
        let b = brick_aabb(key, &config);
        let sampled = Aabb3d::from_min_max(Vec3::from(b.min) - apron, Vec3::from(b.max) + apron);
        idx.clear();
        for (i, a) in edit_aabbs.iter().enumerate() {
            if aabb_overlap(a, &sampled) {
                idx.push(i as u32);
            }
        }
        let base = if idx.is_empty() { 0 } else { edits::bake_content_hash(&edits_arc, idx.as_slice()) };
        base ^ epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15)
    };

    // On-demand diagnostic dump (panel "Capture diagnostics"): volumes, any ghost meshes (key not
    // resident — the reaper keeps this 0), any non-brick world meshes, and any STALE displayed brick
    // (displayed_hash != current content hash). With the content-hash design a persistent stale entry is
    // impossible at rest — it would re-bake next frame — so STALE should read 0 once settled.
    if stats.capture {
        stats.capture = false;
        let mut idx: Vec<u32> = Vec::new();
        let mut s = String::new();
        s.push_str("=== Mesh Bake Diagnostics ===\n");
        s.push_str(&format!("volumes(edits)={n_edits}  resident_bricks={}\n", resident.len()));
        s.push_str("-- volumes (entity : world AABB) --\n");
        for g in &gathered {
            let a = g.aabb;
            s.push_str(&format!(
                "  {:?}  min[{:.2},{:.2},{:.2}] max[{:.2},{:.2},{:.2}]\n",
                g.entity, a.min.x, a.min.y, a.min.z, a.max.x, a.max.y, a.max.z
            ));
        }
        let mut live = 0usize;
        let mut ghost_n = 0usize;
        for (_e, bm) in &brick_meshes {
            live += 1;
            if !resident.contains(&bm.0) {
                ghost_n += 1;
            }
        }
        let mut stale_n = 0usize;
        let mut stale_s = String::new();
        for (_e, bm) in &brick_meshes {
            let h = brick_content_hash(bm.0, &mut idx);
            if let Some(st) = states.0.get(&bm.0)
                && st.displayed_hash != h
            {
                stale_n += 1;
                let w = config.brick_min_world(bm.0.coord, 0);
                stale_s.push_str(&format!(
                    "  world[{:.2},{:.2},{:.2}] displayed={} current={} task={}\n",
                    w.x, w.y, w.z, st.displayed_hash, h, st.task.is_some()
                ));
            }
        }
        s.push_str(&format!("brick meshes={live}  ghosts(should be 0)={ghost_n}\n"));
        s.push_str(&format!("STALE displayed bricks (displayed_hash != current)={stale_n}\n"));
        s.push_str(if stale_s.is_empty() { "  (none)\n" } else { &stale_s });
        stats.dump = s;
    }

    let material_handle = material
        .get_or_insert_with(|| {
            // Unlit, normal-as-colour: visible without scene lights; edges read as a colour gradient.
            materials.add(StandardMaterial { base_color: Color::WHITE, unlit: true, ..default() })
        })
        .clone();

    // 0. REAP (the ghost-proof guarantee): despawn ANY brick mesh whose key is no longer in the
    // geometry footprint — keyed off the entity's own `BrickMesh(key)`, so a moved/edited object's old
    // meshes are cleared regardless of `BrickStates` bookkeeping, command-ordering, or task races.
    // Residency is the single source of truth; this is the only departed-despawn site.
    let mut reaped = 0usize;
    for (e, bm) in &brick_meshes {
        if !resident.contains(&bm.0) {
            commands.entity(e).despawn();
            reaped += 1;
        }
    }
    // Diagnostics: total live mesh entities (pre-despawn snapshot), edits gathered, resident bricks,
    // and how many ghosts the reaper is clearing this frame.
    stats.edits = n_edits;
    stats.resident = resident.len();
    stats.reaped = reaped;

    // 1. RECEIVE: poll in-flight tasks; on completion, swap the displayed mesh and adopt the hash the task
    // baked. (Departed bricks: the REAP pass owns the despawn — just forget the entity, never double-despawn.)
    for (key, st) in states.0.iter_mut() {
        let Some(task) = st.task.as_mut() else {
            continue;
        };
        let Some(result) = block_on(poll_once(&mut *task)) else {
            continue;
        };
        st.task = None;
        if !resident.contains(key) {
            st.entity = None;
            continue;
        }
        if let Some(old) = st.entity.take() {
            commands.entity(old).despawn();
        }
        if let Some(data) = result {
            let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, data.positions)
                .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals)
                .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, data.colors)
                .with_inserted_indices(Indices::U32(data.indices));
            let origin = config.brick_min_world(key.coord, 0) - Vec3::splat(vs);
            let entity = commands
                .spawn((
                    Mesh3d(mesh_assets.add(mesh)),
                    MeshMaterial3d(material_handle.clone()),
                    Transform::from_translation(origin),
                    BrickMesh(*key),
                    Name::new("SDF Brick Mesh"),
                ))
                .id();
            st.entity = Some(entity);
        }
        st.displayed_hash = st.task_hash; // the displayed mesh now reflects the task's inputs
    }

    // 2. Drop the bake state for departed bricks (mesh entity already despawned by REAP; dropping the
    // state cancels any in-flight task).
    states.0.retain(|key, _| resident.contains(key));

    // 3. REQUEST: for each resident brick whose CURRENT content hash differs from the displayed mesh's and
    // that has no in-flight bake, queue one. No "dirty" bookkeeping — staleness is recomputed from the field
    // each frame, so a brick that can't queue now (budget) is simply re-detected next frame (no state leaks).
    let pool = AsyncComputeTaskPool::get();
    let mut budget = MAX_NEW_TASKS_PER_FRAME;
    let mut idx: Vec<u32> = Vec::new();
    for &key in &resident {
        let st = states.0.entry(key).or_default();
        if st.task.is_some() {
            continue;
        }
        let hash = brick_content_hash(key, &mut idx);
        if st.displayed_hash == hash {
            continue; // up to date
        }
        if budget == 0 {
            continue; // re-detected next frame; no dirty flag to leak
        }
        let grid_origin = config.brick_min_world(key.coord, 0) - Vec3::splat(vs);
        let edits = edits_arc.clone();
        let indices = idx.clone();
        st.task = Some(pool.spawn(async move { mesh_brick(&edits, &indices, grid_origin, vs) }));
        st.task_hash = hash;
        budget -= 1;
    }
}

/// Dedicated "Mesh Bake" bottom dock panel (editor builds): the controls for viewing/inspecting the
/// Surface Nets bake — replaces the old F1/F2 hotkeys.
#[cfg(feature = "editor")]
fn mesh_bake_panel(world: &mut World, ui: &mut bevy_egui::egui::Ui) {
    use bevy::pbr::wireframe::WireframeConfig;
    use crate::sdf_render::SdfRenderEnabled;

    ui.label("Surface Nets brick bake (Phase 1, async). Uncheck the SDF render to view the meshes.");
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

    // Stats.
    let states = world.resource::<BrickStates>();
    let meshes = states.0.values().filter(|s| s.entity.is_some()).count();
    let in_flight = states.0.values().filter(|s| s.task.is_some()).count();
    ui.label(format!("Brick meshes: {meshes}  ·  meshing: {in_flight}"));

    // Diagnostics: the system's own view vs the actual world. If `entities` exceeds `resident`, ghost
    // meshes are surviving the reaper (closed-loop residency is failing); if `edits` climbs above the
    // authored volume count while dragging, the gather is seeing extra volumes (the real ghost source).
    let stats = world.resource::<MeshBakeStats>();
    let (edits, resident, reaped) = (stats.edits, stats.resident, stats.reaped);
    let entities = world.query_filtered::<(), With<BrickMesh>>().iter(world).count();
    ui.label(format!(
        "edits: {edits}  ·  resident: {resident}  ·  entities: {entities}  ·  reaped/frame: {reaped}"
    ));
    if entities > resident {
        ui.colored_label(
            bevy_egui::egui::Color32::from_rgb(230, 120, 40),
            format!("⚠ {} ghost mesh(es) (entities > resident)", entities - resident),
        );
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

    // Selectable diagnostic dump (volumes + any ghost meshes) — click Capture, then Copy (or select).
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
