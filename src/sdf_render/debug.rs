//! SDF-specific debug tooling. The whole module is gated behind `editor`.
//!
//! Everything here registers itself into the generic debug toolkit (panels +
//! shader-mode registry) at plugin build. The toolkit framework knows nothing
//! about the SDF pipeline — this module is purely a consumer, which is the
//! pattern a future BVH/AABB visualizer would follow too.

use bevy::prelude::*;
use bevy_egui::egui;

use crate::editor::panels::{DockSide, register_panel};
use crate::editor::registry::{
    DebugModeKind, ShaderDebugMode, ShaderDebugRegistry, ShaderDebugState, debug_modes_ui,
};
use crate::scene_manager::{AppScene, SceneEntity};

use super::atlas::{BRICK_EDGE, BRICK_VOXELS, SdfAtlas};
use super::bvh::Bvh;
use super::{
    CsgKind, RayStepCapture, SdfCamera, SdfGridConfig, SdfMaterial, SdfOp, SdfOrbitCamera,
    DdgiParams, SdfOrder, SdfOverlayGizmos, SdfPrimitive, SdfRaymarchParams, SdfVolume,
    WireframeBoundsVisible, picking,
};

// --- Resources ---

#[derive(Resource, Reflect, Default)]
#[reflect(Resource)]
pub struct SdfAtlasStats {
    pub total_bricks: u32,
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub dirty: bool,
    // Memory breakdown, in bytes.
    pub dist_bytes: u64,
    /// Material atlas (the single `mat_pages` texture: per-voxel palette-slot distances).
    pub object_bytes: u64,
    /// Gradient atlas (per-voxel baked normal; 0 until the Phase-3 gradient channel lands).
    pub blend_bytes: u64,
    pub lookup_bytes: u64,
    pub total_bytes: u64,
    // DDGI probes: one block of `subdiv³` octahedral probes per FINEST-resident brick (the compact
    // clipmap-bounded set). `probe_bytes` is the irradiance buffer; `probe_redundancy` = all-LOD bricks /
    // finest probes (how much the finest-resident collapse saves vs the old all-LOD sizing).
    pub probe_count: u32,
    pub probe_bytes: u64,
    pub probe_redundancy: f32,
}

/// Controls the BVH wireframe overlay (drawn via [`draw_bvh`]).
#[derive(Resource, Reflect)]
#[reflect(Resource)]
pub struct BvhDebugState {
    pub visible: bool,
    /// Only draw nodes up to this tree depth (root = 0).
    pub max_depth: u32,
    /// Draw only leaf nodes (the boxes that actually hold edits).
    pub leaves_only: bool,
}

impl Default for BvhDebugState {
    fn default() -> Self {
        Self {
            visible: false,
            max_depth: 16,
            leaves_only: false,
        }
    }
}

/// Controls the non-empty-chunk wireframe overlay (drawn via [`draw_chunks`]). One box
/// per resident chunk, coloured by LOD — the chunk-grid analogue of the BVH visualizer.
#[derive(Resource, Reflect, Default)]
#[reflect(Resource)]
pub struct ChunkDebugState {
    pub visible: bool,
}

// --- Plugin ---

pub struct SdfDebugPlugin;

impl Plugin for SdfDebugPlugin {
    fn build(&self, app: &mut App) {
        register_shader_modes(app);

        // Custom Inspector editor for SdfMaterialSource (base-file picker + inline overrides).
        // The other SDF components reflect cleanly, so the generic inspector handles those.
        // SdfMaterial is the derived runtime id — hidden from the inspector below.
        crate::editor::inspector::register_component_editor::<crate::sdf_render::SdfMaterialSource>(
            app,
            sdf_material_editor,
        );

        app.init_resource::<SdfAtlasStats>()
            .init_resource::<BvhDebugState>()
            .init_resource::<ChunkDebugState>()
            .register_type::<SdfAtlasStats>()
            .register_type::<BvhDebugState>()
            .register_type::<ChunkDebugState>()
            .add_systems(
                Update,
                (update_atlas_stats, sync_gradient_bake_flag).run_if(in_state(AppScene::SdfEditor)),
            );

        // Gizmo-drawing systems need GizmoPlugin's Assets<GizmoAsset>; absent under
        // MinimalPlugins test harnesses. Register only when present (see the same
        // guard in SdfScenePlugin).
        if app.world().get_resource::<Assets<GizmoAsset>>().is_some() {
            app.add_systems(
                Update,
                (draw_bounds, draw_bvh, draw_chunks, draw_baked_bricks, live_ray_capture)
                    .run_if(in_state(AppScene::SdfEditor)),
            );
        }

        // Left dock: BVH + chunk visualizers. (Atlas stats moved into the Performance tab;
        // Bottom dock: combined render tuning (overlay + raymarch), ray inspector, and the
        // BVH + chunk acceleration-structure visualizers (one combined tab).
        register_panel(
            app,
            "sdf/render",
            "SDF Render",
            DockSide::Bottom,
            20,
            render_panel,
        );
        register_panel(
            app,
            "sdf/accel",
            "SDF Accel",
            DockSide::Bottom,
            30,
            accel_panel,
        );
        register_panel(
            app,
            "sdf/ray_inspector",
            "SDF Ray Inspector",
            DockSide::Bottom,
            40,
            ray_inspector_panel,
        );
    }
}

fn register_shader_modes(app: &mut App) {
    // Init in case SdfScenePlugin builds before EditorPlugin (main.rs order).
    app.init_resource::<ShaderDebugRegistry>();
    app.init_resource::<ShaderDebugState>();

    let overlay = |id: &str, label: &str, define: &str, desc: &str| ShaderDebugMode {
        id: id.into(),
        label: label.into(),
        shader_define: define.into(),
        kind: DebugModeKind::Exclusive {
            group: "sdf_overlay".into(),
        },
        description: desc.into(),
    };

    // Deferred G-buffer visualizers — `#ifdef`-gated early returns in the deferred lit pass
    // (sdf_deferred_lit.wgsl), which holds every G-buffer channel. Exclusive group: at most one
    // active. The lit pipeline rebuilds on def change so these compile in/out.
    let mut registry = app.world_mut().resource_mut::<ShaderDebugRegistry>();
    registry.register(overlay(
        "sdf/albedo",
        "Albedo",
        "SDF_DEBUG_ALBEDO",
        "G-buffer albedo (base colour)",
    ));
    registry.register(overlay(
        "sdf/normals",
        "Normals",
        "SDF_DEBUG_NORMALS",
        "G-buffer world normal as RGB",
    ));
    registry.register(overlay(
        "sdf/metallic",
        "Metallic",
        "SDF_DEBUG_METALLIC",
        "G-buffer metallic (greyscale)",
    ));
    registry.register(overlay(
        "sdf/roughness",
        "Roughness",
        "SDF_DEBUG_ROUGHNESS",
        "G-buffer roughness (greyscale)",
    ));
    registry.register(overlay(
        "sdf/emissive",
        "Emissive",
        "SDF_DEBUG_EMISSIVE",
        "G-buffer emissive radiance",
    ));
    registry.register(overlay(
        "sdf/sun_vis",
        "Sun vis",
        "SDF_DEBUG_SUN_VIS",
        "Marched sun visibility (white = lit, black = shadowed)",
    ));
    registry.register(overlay(
        "sdf/depth",
        "Depth",
        "SDF_DEBUG_DEPTH",
        "Camera distance (scaled greyscale)",
    ));
    registry.register(overlay(
        "sdf/lod",
        "LOD blend",
        "SDF_DEBUG_LOD",
        "Continuous rendered LOD as a hue ramp (red = LOD 0 → blue); the cross-fade band reads as a gradient between two LOD hues",
    ));
    registry.register(overlay(
        "sdf/step_count",
        "Steps",
        "SDF_DEBUG_STEP_COUNT",
        "Raymarch step-count heatmap (blue = few → red = at the budget); step-capped pixels (e.g. grazing hill crests) glow red",
    ));
    registry.register(overlay(
        "sdf/gi",
        "GI",
        "SDF_DEBUG_GI",
        "DDGI indirect irradiance term only (albedo × probe GI × intensity), no direct/emissive",
    ));
    registry.register(overlay(
        "sdf/probe_lod",
        "Probe LOD",
        "SDF_DEBUG_PROBE_LOD",
        "Finest-resident DDGI probe LOD as a hue ramp (LOD0 red → coarse blue) — the clipmap annuli of the probe allocation; black = no probe (coverage hole)",
    ));
    registry.register(overlay(
        "sdf/probe_coverage",
        "Probe coverage",
        "SDF_DEBUG_PROBE_COVERAGE",
        "DDGI probe coverage: green = a finest-resident probe covers the pixel, magenta = uncovered (GI hole)",
    ));

    // Independent toggle (not part of the overlay group): bypass the per-ray chunk
    // lookup cache, forcing a fresh binary search every probe. If enabling this fixes a
    // visual artifact, the cache is the cause. Diagnostic — leave OFF normally.
    registry.register(ShaderDebugMode {
        id: "sdf/no_chunk_cache".into(),
        label: "No chunk cache".into(),
        shader_define: "SDF_DISABLE_CHUNK_CACHE".into(),
        kind: DebugModeKind::Toggle,
        description: "Bypass the per-ray chunk lookup cache (always binary-search)".into(),
    });

    // Independent toggle: force LOD 0 only (no clipmap shells). If enabling this fixes a
    // visual artifact, the bug is LOD/shell related. Diagnostic — leave OFF normally.
    registry.register(ShaderDebugMode {
        id: "sdf/disable_lod".into(),
        label: "LOD 0 only".into(),
        shader_define: "SDF_DISABLE_LOD".into(),
        kind: DebugModeKind::Toggle,
        description: "Force LOD 0 only (disable clipmap shells)".into(),
    });

    // Independent toggle: linear chunk-table scan instead of the binary search. If enabling
    // this fixes a visual artifact, the cause is the binary search / table sortedness / the
    // grid_dims.w count bound.
    registry.register(ShaderDebugMode {
        id: "sdf/linear_chunk_search".into(),
        label: "Linear chunk search".into(),
        shader_define: "SDF_LINEAR_CHUNK_SEARCH".into(),
        kind: DebugModeKind::Toggle,
        description: "Brute-force linear chunk lookup (bypass binary search)".into(),
    });

    // PBR feature toggles (independent checkboxes, not exclusive overlays). These
    // gate real shading features behind shader-defs so their cost is opt-in/measurable.
    // (Sun shadows are marched in the G-buffer pass and consumed by the combine pass.)
    registry.register(ShaderDebugMode {
        id: "sdf/shadows".into(),
        label: "Shadows".into(),
        shader_define: "SDF_SHADOWS".into(),
        kind: DebugModeKind::Toggle,
        description: "SDF soft shadows (sun-visibility ray marched into the G-buffer)".into(),
    });
    registry.register(ShaderDebugMode {
        id: "sdf/edge_wear".into(),
        label: "Edge wear".into(),
        shader_define: "SDF_EDGE_WEAR".into(),
        kind: DebugModeKind::Toggle,
        description: "Convex-edge wear from the edge map (2 extra texture taps per hit pixel)"
            .into(),
    });
    registry.register(ShaderDebugMode {
        id: "sdf/grad_normals".into(),
        label: "Gradient normals".into(),
        shader_define: "SDF_GRAD_NORMALS".into(),
        kind: DebugModeKind::Toggle,
        description: "Shade normals from the baked per-voxel gradient atlas (1 fetch vs the 5-tap \
            finite difference — sharper + cheaper). Enabling bakes the gradient atlas (extra VRAM), \
            so toggling triggers a one-time re-bake."
            .into(),
    });
    // Note: height-map relief is baked into the SDF field (see sdf_render::height) — no shader
    // toggle. Strength is the per-material "Relief depth" (Inspect panel).

    // Default sun shadows ON so the lit render shows them without hunting for the checkbox. The
    // state resource is separate from the registry; seed it after the `registry` borrow above
    // ends (NLL drops it at last use).
    {
        let mut state = app.world_mut().resource_mut::<ShaderDebugState>();
        state.set("sdf/shadows", true);
        // Gradient normals ON by default: the baked-gradient normal (1 fetch) is sharper + cheaper
        // than the 5-tap finite difference. `sync_gradient_bake_flag` turns this into
        // `bake_gradient = true`, so the gradient atlas bakes (the standing VRAM is accepted).
        state.set("sdf/grad_normals", true);
    }
}

/// Drive the per-voxel gradient bake from the `SDF_GRAD_NORMALS` toggle: when it flips, set
/// `SdfAtlas::bake_gradient` and force a full re-bake so every resident brick (re)fills — or stops
/// filling — the gradient atlas. Editor-only; in a non-editor build the flag stays false and the
/// gradient atlas costs nothing.
fn sync_gradient_bake_flag(state: Res<ShaderDebugState>, mut atlas: ResMut<SdfAtlas>) {
    let want = state.is_active("sdf/grad_normals");
    if atlas.bake_gradient != want {
        atlas.bake_gradient = want;
        atlas.rebake_all = true;
    }
}

// --- Atlas stats ---

// Per-voxel GPU byte widths of each brick atlas channel — the SSOT for the memory panel.
// MUST mirror the texture formats created in `render::atlas_pages`. Distance is the single
// R16Snorm atlas (2B); material is the single Rgba16Snorm atlas (8B = 4 palette-slot distances);
// gradient is the single Rgba8Snorm atlas (4B = xyz normal). There is no second material texture —
// the earlier `mat_lo`+`mat_hi` split double-counted a `mat_hi` that never existed.
const DIST_BYTES_PER_VOXEL: u64 = 2;
const MAT_BYTES_PER_VOXEL: u64 = 8;
const GRAD_BYTES_PER_VOXEL: u64 = 4;
const LOOKUP_BYTES_PER_BRICK: u64 = 16;

/// Pure GPU-memory breakdown as `(distance, material, gradient, lookup, total)` bytes. Distance +
/// lookup scale with ALL resident bricks; material scales with the MULTI-material count (`mat_bricks`,
/// single-material bricks store no material tile — the reclamation win); gradient scales with
/// `grad_bricks` (= all bricks when the gradient feature is on, else 0 — it's gated). Extracted so
/// the layout invariant is unit-testable without an `App`/`SdfAtlas`.
fn atlas_byte_breakdown(bricks: u64, mat_bricks: u64, grad_bricks: u64) -> (u64, u64, u64, u64, u64) {
    let voxels = bricks * BRICK_VOXELS as u64;
    let dist = voxels * DIST_BYTES_PER_VOXEL;
    let mat = mat_bricks * BRICK_VOXELS as u64 * MAT_BYTES_PER_VOXEL;
    let grad = grad_bricks * BRICK_VOXELS as u64 * GRAD_BYTES_PER_VOXEL;
    let lookup = bricks * LOOKUP_BYTES_PER_BRICK;
    (dist, mat, grad, lookup, dist + mat + grad + lookup)
}

fn update_atlas_stats(
    mut stats: ResMut<SdfAtlasStats>,
    atlas: Res<SdfAtlas>,
    ddgi: Res<super::DdgiParams>,
) {
    let total = atlas.bricks.len() as u64;
    let mat_bricks = atlas.mat_tiles.len() as u64;
    // Gradient is dense (one tile per brick) but only baked when the feature is on.
    let grad_bricks = if atlas.bake_gradient { total } else { 0 };

    let (dist, mat, grad, lookup, total_bytes) = atlas_byte_breakdown(total, mat_bricks, grad_bricks);
    stats.dist_bytes = dist;
    stats.object_bytes = mat;
    stats.blend_bytes = grad;
    stats.lookup_bytes = lookup;
    stats.total_bytes = total_bytes;

    stats.total_bricks = total as u32;
    // DDGI: subdiv³ octahedral probes per FINEST-resident brick (the compact, clipmap-bounded set);
    // the irradiance buffer is PROBE_OCT_TEXELS vec4<f32> (16 B) per probe. Sized by the per-brick
    // finest high-water — NOT all resident bricks (which the old scheme paid for at every LOD).
    let subdiv = ddgi.subdiv.clamp(1, 4) as u64;
    let finest = atlas.live_chunks.probe_high_water() as u64;
    let probes = finest * subdiv * subdiv * subdiv;
    stats.probe_count = probes as u32;
    stats.probe_bytes = probes * super::probe::PROBE_OCT_TEXELS as u64 * 16;
    // Redundancy eliminated: all-LOD resident bricks vs the finest-resident probe set.
    stats.probe_redundancy = total as f32 / finest.max(1) as f32;
    // 2D-tiled dims (matches the render atlas + preview): tiles wrap at 256/row.
    let tiles_per_row: u32 = 256;
    let num_rows = (total as u32).div_ceil(tiles_per_row).max(1);
    stats.atlas_width = tiles_per_row * (BRICK_EDGE * BRICK_EDGE) as u32;
    stats.atlas_height = num_rows * BRICK_EDGE as u32;
    stats.dirty = atlas.rebake_all || !atlas.gpu_baked_tiles.is_empty();
}

/// Distinct color per object id (golden-ratio hue spacing). id 0 = dark gray.
fn object_color(id: u8) -> [u8; 3] {
    if id == 0 {
        return [40, 40, 40];
    }
    let hue = (id as f32 * 0.618_034).fract() * 6.0;
    let r = (1.0 - (hue - 3.0).abs()).clamp(0.0, 1.0);
    let g = (1.0 - (hue - 2.0).abs()).clamp(0.0, 1.0);
    let b = (1.0 - (hue - 1.0).abs()).clamp(0.0, 1.0);
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

// --- BVH visualizer ---

/// Draw the BVH node AABBs as wireframe boxes, colored by tree depth, when the
/// `BvhDebugState` toggle is on. Walks the flat node array iteratively (BFS so we
/// know each node's depth) and respects the depth / leaves-only filters.
fn draw_bvh(mut gizmos: Gizmos<SdfOverlayGizmos>, state: Res<BvhDebugState>, bvh: Res<Bvh>) {
    if !state.visible || bvh.nodes.is_empty() {
        return;
    }
    const INTERNAL_FLAG: u32 = 0x8000_0000;

    // (node index, depth) queue.
    let mut queue: Vec<(u32, u32)> = vec![(0, 0)];
    while let Some((ni, depth)) = queue.pop() {
        let node = bvh.nodes[ni as usize];
        let internal = node.count_or_right & INTERNAL_FLAG != 0;

        if depth <= state.max_depth && (!state.leaves_only || !internal) {
            let min = Vec3::from(node.aabb_min);
            let max = Vec3::from(node.aabb_max);
            let center = (min + max) * 0.5;
            let size = max - min;
            let color = depth_color(depth);
            gizmos.primitive_3d(
                &Cuboid::new(size.x, size.y, size.z),
                Isometry3d::from_translation(center),
                color,
            );
        }

        if internal && depth < state.max_depth {
            queue.push((node.left_or_first, depth + 1));
            queue.push((node.count_or_right & !INTERNAL_FLAG, depth + 1));
        }
    }
}

/// Distinct wireframe colour per BVH depth (golden-ratio hue spacing).
fn depth_color(depth: u32) -> Color {
    let hue = (depth as f32 * 0.618_034).fract() * 6.0;
    let r = (1.0 - (hue - 3.0).abs()).clamp(0.0, 1.0);
    let g = (1.0 - (hue - 2.0).abs()).clamp(0.0, 1.0);
    let b = (1.0 - (hue - 1.0).abs()).clamp(0.0, 1.0);
    Color::srgb(r.max(0.2), g.max(0.2), b.max(0.2))
}

// --- Chunk visualizer ---

/// Per-LOD chunk wireframe colour, matching the `SDF_DEBUG_LOD` shader ramp:
/// 0 white, 1 green, 2 blue, 3 red, 4+ yellow.
fn lod_color(lod: u32) -> Color {
    match lod {
        0 => Color::srgb(1.0, 1.0, 1.0),
        1 => Color::srgb(0.0, 1.0, 0.0),
        2 => Color::srgb(0.0, 0.4, 1.0),
        3 => Color::srgb(1.0, 0.0, 0.0),
        _ => Color::srgb(1.0, 1.0, 0.0),
    }
}

/// Draw a wireframe box around every resident (non-empty) chunk, coloured by LOD, when
/// the `ChunkDebugState` toggle is on. The chunk-grid analogue of [`draw_bvh`]: shows
/// exactly the sparse set of chunks the clipmap has baked + their nested LOD shells.
fn draw_chunks(
    mut gizmos: Gizmos<SdfOverlayGizmos>,
    state: Res<ChunkDebugState>,
    atlas: Res<SdfAtlas>,
    config: Res<SdfGridConfig>,
) {
    if !state.visible {
        return;
    }
    for ck in super::chunk::resident_chunks(&atlas, &config) {
        let min = super::chunk::chunk_min_world(ck, &config);
        let size = super::chunk::chunk_world_size(ck.lod, &config);
        let center = min + Vec3::splat(size * 0.5);
        gizmos.primitive_3d(
            &Cuboid::new(size, size, size),
            Isometry3d::from_translation(center),
            lod_color(ck.lod),
        );
    }
}

/// Diagnostic: a bright wire cube over every brick the bake EMITTED this frame (from
/// `BakedBrickDebug`, filled in `emit_gpu_bakes` when enabled). Lets you SEE exactly which
/// bricks an edit move dirties — e.g. confirm dragging a small object far from the heightmap
/// doesn't re-bake terrain bricks. Toggled in the SDF Chunks panel.
fn draw_baked_bricks(
    mut gizmos: Gizmos<SdfOverlayGizmos>,
    dbg: Res<super::BakedBrickDebug>,
    time: Res<Time>,
) {
    if !dbg.enabled {
        return;
    }
    let now = time.elapsed_secs();
    for &(center, size, baked_at) in &dbg.bricks {
        // Fade alpha from 1 (just baked) to 0 at the fade window's end. Newest = brightest.
        let age = (now - baked_at).max(0.0);
        let alpha = (1.0 - age / super::BAKED_BRICK_FADE_SECS).clamp(0.0, 1.0);
        if alpha <= 0.0 {
            continue;
        }
        gizmos.primitive_3d(
            &Cuboid::new(size, size, size),
            Isometry3d::from_translation(center),
            Color::srgba(1.0, 0.2, 0.8, alpha),
        );
    }
}

/// Panel: resident-chunk count + the overlay toggle (mirrors `bvh_panel`).
/// Combined acceleration-structure panel: BVH stats + visualizer toggles, then the
/// resident-chunk stats + visualizer toggle. One bottom-dock tab for "what the bake's
/// spatial structures look like".
fn accel_panel(world: &mut World, ui: &mut egui::Ui) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.heading("BVH");
            bvh_ui(world, ui);
            ui.separator();
            ui.heading("Chunks");
            chunk_ui(world, ui);
        });
}

fn chunk_ui(world: &mut World, ui: &mut egui::Ui) {
    let (count, by_lod) = {
        let atlas = world.resource::<SdfAtlas>();
        let config = world.resource::<SdfGridConfig>();
        let chunks = super::chunk::resident_chunks(atlas, config);
        let mut by_lod = [0u32; 16];
        for ck in &chunks {
            if (ck.lod as usize) < by_lod.len() {
                by_lod[ck.lod as usize] += 1;
            }
        }
        (chunks.len(), by_lod)
    };
    ui.label(format!("Resident chunks: {count}"));
    for (lod, n) in by_lod.iter().enumerate() {
        if *n > 0 {
            ui.label(format!("  LOD {lod}: {n}"));
        }
    }
    ui.separator();
    {
        let mut state = world.resource_mut::<ChunkDebugState>();
        ui.checkbox(&mut state.visible, "Show chunk boxes");
    }
    {
        let baked_count = world.resource::<super::BakedBrickDebug>().bricks.len();
        let mut dbg = world.resource_mut::<super::BakedBrickDebug>();
        ui.checkbox(&mut dbg.enabled, "Show baked bricks (this frame)");
        if dbg.enabled {
            ui.label(format!("  baked this frame: {baked_count}"));
        }
    }
}

// --- Wireframe bounds ---

/// Draw each SDF primitive's own shape wireframe (sphere circles, box edges,
/// torus rings, etc.) with immediate-mode gizmos (always-on-top via the
/// `SdfOverlayGizmos` config group) when the toggle is on. The per-primitive
/// geometry lives on `SdfPrimitive::draw_wireframe` — single source of truth, so
/// debug.rs doesn't re-encode any shape. No mesh lifecycle.
fn draw_bounds(
    mut gizmos: Gizmos<SdfOverlayGizmos>,
    visible: Res<WireframeBoundsVisible>,
    registry: Res<super::edits::MaterialRegistry>,
    volumes: Query<(&Transform, &SdfPrimitive, &SdfMaterial), With<SdfVolume>>,
) {
    if !visible.0 {
        return;
    }
    for (transform, prim, material) in &volumes {
        let iso = Isometry3d::new(transform.translation, transform.rotation);
        let color = registry
            .defs
            .get(material.registry_id as usize)
            .map(|d| d.base_color)
            .unwrap_or(Color::WHITE);
        prim.draw_wireframe(&mut gizmos, iso, transform.scale, color);
    }
}

// --- Panels ---

/// Combined render-tuning panel: debug-overlay selection (dropdowns + diagnostics)
/// plus the raymarch quality sliders. One tab for "how the SDF is drawn".
fn render_panel(world: &mut World, ui: &mut egui::Ui) {
    debug_modes_ui(world, ui);

    ui.separator();
    ui.label("Raymarch");
    let mut params = world.resource_mut::<SdfRaymarchParams>();
    ui.add(egui::Slider::new(&mut params.max_steps, 16..=512).text("Steps"));
    ui.add(
        egui::Slider::new(&mut params.max_dist, 10.0..=1_000_000.0)
            .logarithmic(true)
            .text("Max Dist"),
    );
    ui.add(egui::Slider::new(&mut params.sdf_eps, 0.0001..=0.1).text("Epsilon"));
    ui.add(egui::Slider::new(&mut params.lod_blend_band, 0.0..=0.5).text("LOD Blend Band"));
    ui.add(
        egui::Slider::new(&mut params.shadow_softness, 0.0..=256.0)
            .text("Shadow Softness")
            .custom_formatter(|v, _| {
                if v <= 0.0 {
                    "0 (hard)".to_string()
                } else {
                    format!("{v:.0} (higher = sharper)")
                }
            }),
    );
    // How many point lights cast SDF shadows per pixel (brightest-first of those reaching the
    // surface); the rest add unshadowed. Higher = more shadowed lights but costlier; 0 = none.
    ui.add(
        egui::Slider::new(&mut params.shadow_light_cap, 0..=32).text("Shadow lights"),
    );
    ui.separator();
    ui.label("DDGI (Global Illumination — always on)");
    // Live probe stats (finest-resident, clipmap-bounded): probe count, irradiance-buffer size, and the
    // redundancy the finest collapse eliminated vs the old all-LOD sizing.
    {
        let stats = world.resource::<SdfAtlasStats>();
        ui.label(format!(
            "Probes: {} finest · {:.1} MiB · {:.1}× redundancy removed",
            stats.probe_count,
            stats.probe_bytes as f64 / (1u64 << 20) as f64,
            stats.probe_redundancy,
        ));
        let rel = world.resource::<super::ProbeRelevanceSet>();
        if rel.total > 0 {
            ui.label(format!(
                "Relevance cull: {} / {} finest chunks off-screen ({:.0}% throttled)",
                rel.culled,
                rel.total,
                100.0 * rel.culled as f32 / rel.total as f32,
            ));
        }
    }
    let mut ddgi = world.resource_mut::<DdgiParams>();
    ui.add(
        egui::Slider::new(&mut ddgi.subdiv, 1..=4)
            .text("Probe subdiv (LOD0 density)")
            .custom_formatter(|v, _| format!("{v:.0} ({:.0}³/brick)", v)),
    );
    ui.add(egui::Slider::new(&mut ddgi.ray_count, 8..=256).text("Rays / probe"));
    ui.add(
        egui::Slider::new(&mut ddgi.update_stride, 1..=16)
            .text("Update stride (1/N probes per frame)"),
    );
    ui.add(
        egui::Slider::new(&mut ddgi.max_probe_chunks_per_frame, 0..=4096)
            .text("Max probe-chunks / frame (0 = ∞, nearest-first)"),
    );
    ui.checkbox(&mut ddgi.classify_enabled, "Classify (settled probes go dormant)");
    ui.add_enabled(
        ddgi.classify_enabled,
        egui::Slider::new(&mut ddgi.dormant_stride, 4..=128).text("Dormant stride (converged re-trace)"),
    );
    // Distant-probe cost controls (cut the dominant trace cost in the far field).
    ui.add(egui::Slider::new(&mut ddgi.probe_halve_lod, 1..=10).text("Halve density ≥ LOD"));
    ui.add(egui::Slider::new(&mut ddgi.ray_falloff_lod, 1..=10).text("Distant rays ≥ LOD"));
    ui.add(egui::Slider::new(&mut ddgi.distant_ray_count, 8..=256).text("Distant ray count"));
    ui.add(egui::Slider::new(&mut ddgi.gi_march_steps, 4..=48).text("GI march steps / ray"));
    // View-relevance cull: throttle finest probes that are off-screen (the moving-camera saving).
    ui.checkbox(&mut ddgi.relevance_cull, "Relevance cull (throttle off-screen probes)");
    ui.add_enabled_ui(ddgi.relevance_cull, |ui| {
        ui.add(
            egui::Slider::new(&mut ddgi.cull_off_stride, 8..=256)
                .text("Off-screen stride (1/N re-trace)"),
        );
        ui.add(
            egui::Slider::new(&mut ddgi.cull_near_radius, 0.0..=64.0)
                .text("Always-relevant near radius (m)"),
        );
        ui.add(
            egui::Slider::new(&mut ddgi.cull_cone_dot, -1.0..=0.5)
                .text("View-cone cull (−1 off … 0 rear)"),
        );
    });
    ui.add(
        egui::Slider::new(&mut ddgi.gi_range, 4.0..=200.0)
            .logarithmic(true)
            .text("GI ray range (world units)"),
    );
    ui.add(
        egui::Slider::new(&mut ddgi.hysteresis, 0.0..=0.99)
            .text("Accumulation (N_max = 1/(1−h))"),
    );
    ui.add(egui::Slider::new(&mut ddgi.intensity, 0.0..=8.0).text("Intensity"));
    ui.add(egui::Slider::new(&mut ddgi.gi_sky_intensity, 0.0..=2.0).text("Sky GI intensity"));
    ui.checkbox(&mut ddgi.gi_bounce_shadows, "Bounce shadows (sun + points)");
    ui.add(egui::Slider::new(&mut ddgi.normal_bias, 0.0..=2.0).text("Normal bias (×cell)"));
    ui.add(egui::Slider::new(&mut ddgi.view_bias, 0.0..=2.0).text("View bias (×cell)"));
    ui.add(
        egui::Slider::new(&mut ddgi.gi_blur_depth_sigma, 0.01..=1.0)
            .logarithmic(true)
            .text("GI blur depth tol"),
    );
    ui.add(
        egui::Slider::new(&mut ddgi.gi_blur_normal_power, 1.0..=64.0)
            .text("GI blur normal stop"),
    );
    // Probe buffer ceiling (MiB). The probe count is clamped to this (capped further by the device
    // binding limit); over-budget probes go inactive. Sized in whole MiB for a readable slider.
    let mut budget_mib = ddgi.probe_budget_bytes / (1 << 20);
    if ui
        .add(egui::Slider::new(&mut budget_mib, 64..=2048).text("Probe budget (MiB)"))
        .changed()
    {
        ddgi.probe_budget_bytes = budget_mib.max(1) * (1 << 20);
    }
}

fn ray_inspector_panel(world: &mut World, ui: &mut egui::Ui) {
    ui.label("Hold C over the viewport to capture the ray under the cursor.");

    let capture = world.resource::<RayStepCapture>();
    ui.label(format!("Steps: {}", capture.steps.len()));
    if capture.steps.is_empty() {
        return;
    }
    let steps = capture.steps.clone();

    // Surface hit (last step): the world point the CPU march converged on. Compare
    // this against where the GPU renders the surface to spot atlas-upload drift.
    if let Some(last) = steps.last() {
        let hit = if last.dist < 0.01 { "HIT" } else { "miss" };
        ui.label(format!(
            "{hit} @ ({:.3}, {:.3}, {:.3})  d={:.3}",
            last.pos.x, last.pos.y, last.pos.z, last.dist
        ));
    }
    egui::ScrollArea::vertical()
        .max_height(140.0)
        .show(ui, |ui| {
            for (i, s) in steps.iter().enumerate() {
                ui.label(format!(
                    "{i:>3}  t={:.3}  d={:.3}  pos=({:.2},{:.2},{:.2})  brick=({},{},{}) {}",
                    s.t,
                    s.dist,
                    s.pos.x,
                    s.pos.y,
                    s.pos.z,
                    s.brick.x,
                    s.brick.y,
                    s.brick.z,
                    if s.in_brick { "in" } else { "--" }
                ));
            }
        });
}

/// While C is held, cast the ray under the live cursor and store/visualize the
/// trace. A per-frame system (not a panel button) so the cast follows the cursor
/// in the viewport rather than landing on the button. Also draws the path as
/// overlay gizmos: green segments inside a baked brick, gray in empty space.
#[allow(clippy::too_many_arguments)] // Bevy system params; splitting is artificial.
fn live_ray_capture(
    keyboard: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &Transform), With<SdfCamera>>,
    volumes: Query<super::VolumeQueryData, With<SdfVolume>>,
    atlas: Res<SdfAtlas>,
    config: Res<SdfGridConfig>,
    mut capture: ResMut<RayStepCapture>,
    mut gizmos: Gizmos<SdfOverlayGizmos>,
) {
    if !keyboard.pressed(KeyCode::KeyC) {
        return;
    }
    let (Ok(window), Ok((camera, cam_transform))) = (windows.single(), cameras.single()) else {
        return;
    };
    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };
    let Some(ray) = picking::mouse_to_ray(camera, cam_transform, window, cursor_pos) else {
        return;
    };

    // Same CSG edit set the bake/pick use, folded by debug_capture_march.
    let resolved: Vec<super::edits::ResolvedEdit> = super::gather_sorted_edits(&volumes)
        .into_iter()
        .map(|g| g.edit)
        .collect();
    let steps = picking::debug_capture_march(&atlas, &ray, &resolved, &config);

    // Visualize the marched path: each step segment colored by brick occupancy.
    for pair in steps.windows(2) {
        let color = if pair[0].in_brick {
            Srgba::rgb(0.3, 0.9, 0.4)
        } else {
            Srgba::rgb(0.5, 0.5, 0.5)
        };
        gizmos.line(pair[0].pos, pair[1].pos, color);
    }
    if let Some(last) = steps.last()
        && last.dist < 0.01
    {
        gizmos.sphere(
            Isometry3d::from_translation(last.pos),
            0.04,
            Srgba::rgb(1.0, 0.9, 0.2),
        );
    }

    capture.steps = steps;
}

// --- BVH panel ---

fn bvh_ui(world: &mut World, ui: &mut egui::Ui) {
    // Node/leaf/depth stats from the flat node array.
    const INTERNAL_FLAG: u32 = 0x8000_0000;
    let (node_count, leaf_count, max_depth, edits) = {
        let bvh = world.resource::<Bvh>();
        let nodes = bvh.nodes.len();
        let leaves = bvh
            .nodes
            .iter()
            .filter(|n| n.count_or_right & INTERNAL_FLAG == 0)
            .count();
        // Depth via the same BFS walk used for drawing.
        let mut depth = 0u32;
        if !bvh.nodes.is_empty() {
            let mut queue = vec![(0u32, 0u32)];
            while let Some((ni, d)) = queue.pop() {
                depth = depth.max(d);
                let node = bvh.nodes[ni as usize];
                if node.count_or_right & INTERNAL_FLAG != 0 {
                    queue.push((node.left_or_first, d + 1));
                    queue.push((node.count_or_right & !INTERNAL_FLAG, d + 1));
                }
            }
        }
        (nodes, leaves, depth, bvh.edit_indices.len())
    };

    ui.label(format!("Nodes: {node_count}"));
    ui.label(format!("Leaves: {leaf_count}"));
    ui.label(format!("Tree depth: {max_depth}"));
    ui.label(format!("Leaf edit refs: {edits}"));
    ui.separator();

    let mut state = world.resource_mut::<BvhDebugState>();
    ui.checkbox(&mut state.visible, "Show BVH boxes");
    ui.checkbox(&mut state.leaves_only, "Leaves only");
    ui.add(egui::Slider::new(&mut state.max_depth, 0..=24).text("Max depth"));
}

// --- Spawn helper ---

/// Spawn an SDF volume of a specific primitive shape (Union, fresh material, scattered
/// near the orbit target). Shared by the Scene panel's `+` button and the Create Node
/// dialog. Returns the new entity; the caller may reparent it.
pub fn spawn_sdf_primitive(world: &mut World, prim: SdfPrimitive) -> Entity {
    let target = world.resource::<SdfOrbitCamera>().target;
    let pos = target + random_spawn_offset();
    let next_order = next_sdf_order(world);
    let label = sdf_primitive_label(&prim);

    // Fresh primitives get an inline (file-less) procedural material: a distinct scatter
    // colour. `resolve_materials` derives the GPU `SdfMaterial` id from this source.
    let color = spawn_color(next_order as usize).to_linear();
    let source = crate::sdf_render::SdfMaterialSource {
        asset: None,
        overrides: crate::sdf_render::MaterialFields {
            base_color: Some([color.red, color.green, color.blue, color.alpha]),
            ..Default::default()
        },
    };

    world
        .spawn((
            Name::new(format!("{label} {next_order}")),
            Transform::from_translation(pos),
            prim,
            SdfOp {
                kind: CsgKind::Union,
                smoothing: 0.4,
            },
            SdfOrder(next_order),
            source,
            SdfVolume,
            SceneEntity,
        ))
        .id()
}

/// The next free `SdfOrder` value (max existing + 1, else 0).
pub fn next_sdf_order(world: &mut World) -> u32 {
    world
        .query_filtered::<&SdfOrder, With<SdfVolume>>()
        .iter(world)
        .map(|o| o.0)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0)
}

/// Spawn a directional light node near the orbit target, with its editor gizmo so it
/// is locatable/orientable in the viewport. Returns the new entity.
pub fn spawn_directional_light(world: &mut World) -> Entity {
    let pos = world.resource::<SdfOrbitCamera>().target + Vec3::Y * 3.0;
    world
        .spawn((
            Name::new("Directional Light"),
            DirectionalLight {
                illuminance: 10000.0,
                shadows_enabled: false,
                ..default()
            },
            Transform::from_translation(pos).with_rotation(Quat::from_rotation_x(-0.5)),
            crate::node::Node3D,
            crate::node::EditorGizmo::DirectionalLight { scale: 1.0 },
            SceneEntity,
        ))
        .id()
}

/// Spawn a point-light node near the orbit target, with its editor gizmo (a camera-facing
/// ring + draggable radius handle). Uses a real Bevy `PointLight` so it lights both SDF and
/// regular meshes; `range` is the source of truth for the gizmo's ring radius. Returns the
/// new entity.
pub fn spawn_point_light(world: &mut World) -> Entity {
    let pos = world.resource::<SdfOrbitCamera>().target + Vec3::Y * 2.0;
    world
        .spawn((
            Name::new("Point Light"),
            PointLight {
                // Warm white, bright enough to clearly light nearby geometry without
                // blowing out. `range` = falloff cutoff (outer gizmo ring); `radius` =
                // physical light size for soft shadows (inner gizmo ring) — both editable
                // via their handles and the inspector.
                color: Color::srgb(1.0, 0.95, 0.85),
                intensity: 1_000_000.0,
                range: 5.0,
                radius: 0.5,
                shadows_enabled: false,
                ..default()
            },
            Transform::from_translation(pos),
            crate::node::Node3D,
            crate::node::EditorGizmo::PointLight { scale: 1.0 },
            SceneEntity,
        ))
        .id()
}

/// Spawn a scene camera node near the orbit target, looking at it. Authored data
/// (`SceneCamera`) + a frustum gizmo — NOT an active render camera (no `Camera3d`/
/// `SdfCamera`), so it stays off the render path. The editor can "look through" it.
pub fn spawn_camera(world: &mut World) -> Entity {
    let target = world.resource::<SdfOrbitCamera>().target;
    let pos = target + Vec3::new(2.0, 1.5, 2.0);
    world
        .spawn((
            Name::new("Camera"),
            Transform::from_translation(pos).looking_at(target, Vec3::Y),
            crate::node::Node3D,
            crate::node::SceneCamera::default(),
            crate::node::EditorGizmo::Camera { scale: 1.0 },
            SceneEntity,
        ))
        .id()
}

/// Spawn an empty `Node3D` (a transform-only grouping/locator node) at the orbit
/// target, with an axes gizmo so it is visible. Returns the new entity.
pub fn spawn_empty_node(world: &mut World) -> Entity {
    let pos = world.resource::<SdfOrbitCamera>().target;
    world
        .spawn((
            Name::new("Node3D"),
            Transform::from_translation(pos),
            crate::node::Node3D,
            crate::node::EditorGizmo::Axes { scale: 0.5 },
            SceneEntity,
        ))
        .id()
}

/// Human-readable shape name for a primitive (used in default node names).
fn sdf_primitive_label(prim: &SdfPrimitive) -> &'static str {
    match prim {
        SdfPrimitive::Sphere { .. } => "Sphere",
        SdfPrimitive::Box { .. } => "Box",
        SdfPrimitive::Torus { .. } => "Torus",
        SdfPrimitive::Capsule { .. } => "Capsule",
        SdfPrimitive::Cylinder { .. } => "Cylinder",
        SdfPrimitive::Heightmap { .. } => "Heightmap",
    }
}

/// A distinct spawn colour per material slot (golden-ratio hue), matching the
/// debug object-id palette so spawned edits read as different materials.
fn spawn_color(index: usize) -> Color {
    let [r, g, b] = object_color(index.min(7) as u8 + 1);
    Color::srgb(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
}

/// A small random offset (world units) for scattering newly-spawned edits around
/// the orbit target. Seeded from the wall clock so each click lands somewhere new;
/// avoids pulling in a full RNG dependency for a debug-only jitter.
fn random_spawn_offset() -> Vec3 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    // Three decorrelated hashes -> [-1, 1] per axis, scaled to a modest radius.
    let comp = |salt: u32| -> f32 {
        let mut h = nanos
            .wrapping_mul(2_654_435_761)
            .wrapping_add(salt.wrapping_mul(40_503));
        h ^= h >> 15;
        h = h.wrapping_mul(2_246_822_519);
        h ^= h >> 13;
        (h as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    Vec3::new(comp(1), comp(2), comp(3)) * 1.5
}

// --- Inspector: SdfMaterialSource editor ---

/// Set `entity`'s material to the `.material.ron` at `working_path` (a working-dir path like
/// `assets/materials/sand.material.ron`). Converts to the assets-relative form expected by
/// [`SdfMaterialSource::asset`] and writes it if the entity accepts a material (has the
/// component). No-op otherwise. Shared by the inspector drop, the viewport drop, and the
/// picker — mutating the source fires `Changed` → `resolve_materials` re-derives the GPU id.
/// Returns whether the material was set.
pub fn set_entity_material(world: &mut World, entity: Entity, working_path: &std::path::Path) -> bool {
    let rel = crate::editor::fs_util::relative_to_assets(working_path)
        .unwrap_or_else(|| working_path.to_path_buf());
    if let Some(mut source) = world.get_mut::<crate::sdf_render::SdfMaterialSource>(entity) {
        source.asset = Some(rel);
        true
    } else {
        false
    }
}

/// Custom Inspector editor for [`SdfMaterialSource`]: pick the base material file this
/// volume uses (or none → inline), and edit the appropriate appearance. The authored
/// source drives the runtime `SdfMaterial` id via `resolve_materials`. Registered via
/// `register_component_editor::<SdfMaterialSource>`.
///
/// - File material (`asset: Some`): the full asset editor (edits the on-disk `.material.ron`,
///   shared by every volume using it).
/// - Inline material (`asset: None`): edit the per-field overrides directly (base colour +
///   scalar PBR), stored on the volume and serialized into the scene.
pub fn sdf_material_editor(world: &mut World, entity: Entity, ui: &mut egui::Ui) {
    use crate::editor::material_editor::material_picker_entries;
    use crate::editor::resource_picker::{PickResult, PickerEntry, resource_picker};
    use crate::sdf_render::SdfMaterialSource;

    let Some(source) = world.get::<SdfMaterialSource>(entity).cloned() else {
        return;
    };

    // Current base file (working-dir path) → picker entry, for the selection highlight.
    let current_entry = source.asset.as_ref().map(|rel| {
        let working = std::path::Path::new(crate::editor::assets_browser::ASSETS_ROOT).join(rel);
        PickerEntry {
            key: working.to_string_lossy().into_owned(),
            label: rel
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.trim_end_matches(".material.ron").to_string())
                .unwrap_or_default(),
            thumb: crate::editor::assets_browser::TileThumb::Path(working),
        }
    });

    // The ENTIRE material section is a drop target: drag a material tile from the assets tray
    // anywhere over this section to set this volume's material. Remember where the section
    // starts so we can interact over its full rect after the body is laid out.
    let section_top = ui.min_rect().bottom();

    ui.label("Material");
    let picked = resource_picker(
        world,
        ui,
        ui.make_persistent_id(("sdf_mat_picker", entity)),
        current_entry.as_ref(),
        true, // allow clearing → inline material
        material_picker_entries,
    );

    // On pick: set the source's base file (as an assets-relative path), or clear to inline.
    // Mutating `SdfMaterialSource` fires `Changed` → `resolve_materials` re-derives the id.
    match picked {
        Some(PickResult::Key(path)) => {
            let rel = std::path::Path::new(&path)
                .strip_prefix(crate::editor::assets_browser::ASSETS_ROOT)
                .map(std::path::Path::to_path_buf)
                .unwrap_or_else(|_| std::path::PathBuf::from(&path));
            if let Some(mut s) = world.get_mut::<SdfMaterialSource>(entity) {
                s.asset = Some(rel);
            }
        }
        Some(PickResult::None) => {
            if let Some(mut s) = world.get_mut::<SdfMaterialSource>(entity) {
                s.asset = None;
            }
        }
        None => {}
    }

    // Re-read after a possible pick.
    let Some(source) = world.get::<SdfMaterialSource>(entity).cloned() else {
        return;
    };

    if let Some(rel) = source.asset {
        // File-backed: edit the on-disk asset (the authored truth, shared by all users).
        // Resolve it to a handle via the editor's path→handle helper.
        if let Some(handle) = crate::editor::material_editor::handle_for_path(
            world,
            &std::path::Path::new(crate::editor::assets_browser::ASSETS_ROOT).join(&rel),
        ) {
            crate::editor::material_editor::material_editor_ui(world, &handle, ui);
        } else {
            ui.weak("Could not resolve material file.");
        }
    } else {
        // Inline material: edit the per-field overrides (stored on the volume, serialized).
        sdf_inline_material_ui(world, entity, ui);
    }

    // Section-wide material drop target: the rect from `section_top` down to the current
    // layout bottom, spanning the editor's width. Drop a material tile anywhere in here to
    // set this volume's material; highlight while a material drag hovers.
    use crate::editor::assets_browser::MaterialDrag;
    let section = egui::Rect::from_min_max(
        egui::pos2(ui.min_rect().left(), section_top),
        egui::pos2(ui.min_rect().right(), ui.min_rect().bottom()),
    );
    let drop = ui.interact(
        section,
        ui.make_persistent_id(("sdf_mat_drop", entity)),
        egui::Sense::hover(),
    );
    if egui::DragAndDrop::payload::<MaterialDrag>(ui.ctx()).is_some() && drop.contains_pointer() {
        ui.painter().rect_stroke(
            section.expand(2.0),
            3.0,
            ui.visuals().selection.stroke,
            egui::StrokeKind::Outside,
        );
    }
    if let Some(drag) = drop.dnd_release_payload::<MaterialDrag>() {
        set_entity_material(world, entity, &drag.0);
    }
}

/// Edit an inline `SdfMaterialSource`'s overrides (base colour + scalar PBR). Each change
/// fires `Changed<SdfMaterialSource>` → `resolve_materials` rebuilds the registry row.
#[cfg(feature = "editor")]
fn sdf_inline_material_ui(world: &mut World, entity: Entity, ui: &mut egui::Ui) {
    use crate::sdf_render::SdfMaterialSource;

    let Some(mut source) = world.get_mut::<SdfMaterialSource>(entity) else {
        return;
    };
    let o = &mut source.overrides;

    // Base colour (defaults to mid-grey when unset).
    let mut rgb = o.base_color.map(|c| [c[0], c[1], c[2]]).unwrap_or([0.8, 0.8, 0.8]);
    ui.horizontal(|ui| {
        ui.label("Base color");
        if ui.color_edit_button_rgb(&mut rgb).changed() {
            let a = o.base_color.map(|c| c[3]).unwrap_or(1.0);
            o.base_color = Some([rgb[0], rgb[1], rgb[2], a]);
        }
    });

    // Scalar overrides: edit a working value, write back as `Some(..)` on change.
    let scalar = |ui: &mut egui::Ui, label: &str, cur: &mut Option<f32>, default: f32, max: f32| {
        let mut v = cur.unwrap_or(default);
        if ui.add(egui::Slider::new(&mut v, 0.0..=max).text(label)).changed() {
            *cur = Some(v);
        }
    };
    scalar(ui, "Blend softness", &mut o.blend_softness, 0.0, 1.0);
    scalar(ui, "Metallic", &mut o.metallic, 0.0, 1.0);
    scalar(ui, "Roughness", &mut o.roughness, 1.0, 1.0);
    scalar(ui, "Relief depth", &mut o.parallax_scale, 0.0, 0.4);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The memory panel's per-channel breakdown must sum to the reported total. Distance is ONE
    /// R16Snorm atlas over all bricks; material is ONE Rgba16Snorm atlas (8 B/voxel) sized to the
    /// MULTI-material count only (the reclamation win); gradient is ONE Rgba8Snorm atlas (4 B/voxel)
    /// over all bricks but only when enabled (`grad_bricks`). Locks the decoupled+gated layout.
    #[test]
    fn atlas_byte_breakdown_is_consistent_and_decoupled() {
        // (total bricks, multi-material bricks, gradient bricks)
        for (bricks, mat_bricks, grad_bricks) in
            [(0u64, 0u64, 0u64), (1, 0, 0), (1000, 0, 0), (1000, 250, 1000), (7, 7, 0)]
        {
            let (dist, mat, grad, lookup, total) =
                atlas_byte_breakdown(bricks, mat_bricks, grad_bricks);
            assert_eq!(dist + mat + grad + lookup, total, "breakdown must sum to total");

            assert_eq!(dist, bricks * BRICK_VOXELS as u64 * 2, "distance: one R16Snorm tile per brick");
            assert_eq!(
                mat,
                mat_bricks * BRICK_VOXELS as u64 * 8,
                "material: one Rgba16Snorm tile per MULTI-material brick only"
            );
            assert_eq!(
                grad,
                grad_bricks * BRICK_VOXELS as u64 * 4,
                "gradient: one Rgba8Snorm tile per brick, only when enabled"
            );
            assert_eq!(lookup, bricks * 16);
        }
        // Reclamation invariant: no multi-material bricks ⇒ zero material VRAM. Gating invariant:
        // gradient off (grad_bricks=0) ⇒ zero gradient VRAM.
        let (_, mat_none, grad_off, _, _) = atlas_byte_breakdown(1000, 0, 0);
        assert_eq!(mat_none, 0, "single-material-only scene allocates no material atlas");
        assert_eq!(grad_off, 0, "gradient-disabled scene allocates no gradient atlas");
    }
}
