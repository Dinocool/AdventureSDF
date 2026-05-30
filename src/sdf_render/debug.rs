//! SDF-specific debug tooling. The whole module is gated behind `editor`.
//!
//! Everything here registers itself into the generic debug toolkit (panels +
//! shader-mode registry) at plugin build. The toolkit framework knows nothing
//! about the SDF pipeline — this module is purely a consumer, which is the
//! pattern a future BVH/AABB visualizer would follow too.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy_egui::{EguiTextureHandle, EguiUserTextures, egui};

use crate::editor::panels::{DockSide, register_panel};
use crate::editor::registry::{
    DebugModeKind, ShaderDebugMode, ShaderDebugRegistry, ShaderDebugState, debug_modes_ui,
};
use crate::scene_manager::{AppScene, SceneEntity};

use super::atlas::{BRICK_EDGE, BRICK_VOXELS, SdfAtlas};
use super::bvh::Bvh;
use super::edits::PALETTE_K;
use super::{
    CsgKind, RayStepCapture, SdfCamera, SdfGridConfig, SdfMaterial, SdfOp, SdfOrbitCamera,
    SdfOrder, SdfOverlayGizmos, SdfPrimitive, SdfRaymarchParams, SdfSelection, SdfVolume,
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
    pub object_bytes: u64,
    pub blend_bytes: u64,
    pub lookup_bytes: u64,
    pub total_bytes: u64,
}

/// Egui-displayable views of the CPU atlas, rebuilt when the atlas changes.
#[derive(Resource)]
pub struct SdfAtlasTextures {
    pub dist_id: Option<egui::TextureId>,
    pub object_id: Option<egui::TextureId>,
    pub dist_handle: Handle<Image>,
    pub object_handle: Handle<Image>,
    pub width: u32,
    pub height: u32,
    /// Display height (px) for each atlas row in the panel; user-adjustable zoom.
    pub view_height: f32,
}

impl Default for SdfAtlasTextures {
    fn default() -> Self {
        Self {
            dist_id: None,
            object_id: None,
            dist_handle: Handle::default(),
            object_handle: Handle::default(),
            width: 0,
            height: 0,
            view_height: 64.0,
        }
    }
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

/// Authoring panel state: the primitive/op/material to spawn next.
#[derive(Resource)]
pub struct SpawnState {
    pub kind: SpawnKind,
    pub op: CsgKind,
    pub smoothing: f32,
    /// Registry id for the next spawn, or `u32::MAX` = create a fresh material.
    pub material: u32,
}

impl Default for SpawnState {
    fn default() -> Self {
        Self {
            kind: SpawnKind::Sphere,
            op: CsgKind::Union,
            smoothing: 0.0,
            material: u32::MAX,
        }
    }
}

/// Which primitive the spawn panel will create (decoupled from the parameterised
/// [`SdfPrimitive`] so the combo box has a simple discriminant).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpawnKind {
    Sphere,
    Box,
    Torus,
    Capsule,
    Cylinder,
    Heightmap,
}

impl SpawnKind {
    const ALL: [SpawnKind; 6] = [
        SpawnKind::Sphere,
        SpawnKind::Box,
        SpawnKind::Torus,
        SpawnKind::Capsule,
        SpawnKind::Cylinder,
        SpawnKind::Heightmap,
    ];

    fn label(self) -> &'static str {
        match self {
            SpawnKind::Sphere => "Sphere",
            SpawnKind::Box => "Box",
            SpawnKind::Torus => "Torus",
            SpawnKind::Capsule => "Capsule",
            SpawnKind::Cylinder => "Cylinder",
            SpawnKind::Heightmap => "Heightmap",
        }
    }

    fn default_primitive(self) -> SdfPrimitive {
        match self {
            SpawnKind::Sphere => SdfPrimitive::Sphere { radius: 0.5 },
            SpawnKind::Box => SdfPrimitive::Box {
                half_extents: Vec3::splat(0.5),
            },
            SpawnKind::Torus => SdfPrimitive::Torus {
                major: 0.5,
                minor: 0.2,
            },
            SpawnKind::Capsule => SdfPrimitive::Capsule {
                half_height: 0.4,
                radius: 0.25,
            },
            SpawnKind::Cylinder => SdfPrimitive::Cylinder {
                radius: 0.4,
                half_height: 0.5,
            },
            SpawnKind::Heightmap => SdfPrimitive::Heightmap {
                half_xz: Vec2::splat(1.5),
                max_height: 0.6,
                freq: 1.5,
                amp: 0.4,
                seed: 1,
            },
        }
    }
}

// --- Plugin ---

pub struct SdfDebugPlugin;

impl Plugin for SdfDebugPlugin {
    fn build(&self, app: &mut App) {
        register_shader_modes(app);

        app.init_resource::<SdfAtlasStats>()
            .init_resource::<SdfAtlasTextures>()
            .init_resource::<BvhDebugState>()
            .init_resource::<ChunkDebugState>()
            .init_resource::<SpawnState>()
            .register_type::<SdfAtlasStats>()
            .register_type::<BvhDebugState>()
            .register_type::<ChunkDebugState>()
            .add_systems(
                Update,
                update_atlas_stats.run_if(in_state(AppScene::SdfEditor)),
            )
            // Needs EguiUserTextures (provided by EditorPlugin). Guarded so
            // SdfScenePlugin can run standalone (e.g. in tests) without egui.
            .add_systems(
                Update,
                update_atlas_textures
                    .run_if(in_state(AppScene::SdfEditor))
                    .run_if(resource_exists::<EguiUserTextures>),
            );

        // Gizmo-drawing systems need GizmoPlugin's Assets<GizmoAsset>; absent under
        // MinimalPlugins test harnesses. Register only when present (see the same
        // guard in SdfScenePlugin).
        if app.world().get_resource::<Assets<GizmoAsset>>().is_some() {
            app.add_systems(
                Update,
                (draw_bounds, draw_bvh, draw_chunks, live_ray_capture)
                    .run_if(in_state(AppScene::SdfEditor)),
            );
        }

        // Left dock: atlas info + textures, gizmo/camera state, wireframe toggle.
        register_panel(
            app,
            "sdf/atlas",
            "SDF Atlas",
            DockSide::Left,
            0,
            atlas_panel,
        );
        register_panel(
            app,
            "sdf/spawn",
            "SDF Spawn",
            DockSide::Left,
            5,
            spawn_panel,
        );
        register_panel(
            app,
            "sdf/inspect",
            "SDF Inspect",
            DockSide::Left,
            6,
            inspect_panel,
        );
        register_panel(app, "sdf/bvh", "SDF BVH", DockSide::Left, 25, bvh_panel);
        register_panel(
            app,
            "sdf/chunks",
            "SDF Chunks",
            DockSide::Left,
            26,
            chunk_panel,
        );
        register_panel(
            app,
            "sdf/gizmo",
            "SDF Gizmo",
            DockSide::Left,
            10,
            gizmo_panel,
        );
        register_panel(
            app,
            "sdf/wireframe",
            "SDF Wireframe",
            DockSide::Left,
            20,
            wireframe_panel,
        );

        // Bottom dock: overlay modes, raymarch tuning, ray inspector.
        register_panel(
            app,
            "sdf/modes",
            "SDF Modes",
            DockSide::Bottom,
            20,
            modes_panel,
        );
        register_panel(
            app,
            "sdf/raymarch",
            "SDF Raymarch",
            DockSide::Bottom,
            30,
            raymarch_panel,
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

    let mut registry = app.world_mut().resource_mut::<ShaderDebugRegistry>();
    registry.register(overlay(
        "sdf/step_count",
        "Steps",
        "SDF_DEBUG_STEP_COUNT",
        "Step heatmap (blue -> red)",
    ));
    registry.register(overlay(
        "sdf/normals",
        "Normals",
        "SDF_DEBUG_NORMALS",
        "Surface normals as RGB",
    ));
    registry.register(overlay(
        "sdf/object_id",
        "Obj ID",
        "SDF_DEBUG_OBJECT_ID",
        "Object ID as distinct colors",
    ));
    registry.register(overlay(
        "sdf/brick_bounds",
        "Bricks",
        "SDF_DEBUG_BRICK_BOUNDS",
        "Color by brick id + cell grid lines",
    ));
    registry.register(overlay(
        "sdf/bvh_steps",
        "BVH cost",
        "SDF_DEBUG_BVH_STEPS",
        "Raymarch step heatmap with BVH skipping (compare vs Steps)",
    ));
    registry.register(overlay(
        "sdf/ray_fate",
        "Ray fate",
        "SDF_DEBUG_RAY_FATE",
        "Every pixel: green=hit, red=escaped (over-skip), blue=out of steps",
    ));
    registry.register(overlay(
        "sdf/lod",
        "LOD",
        "SDF_DEBUG_LOD",
        "Tint hit by clipmap LOD: 0 white, 1 green, 2 blue, 3 red, 4+ yellow",
    ));
    registry.register(overlay(
        "sdf/tile_id",
        "Tile ID",
        "SDF_DEBUG_TILE_ID",
        "Color by resolved atlas tile: same color on duplicated halves = tile collision",
    ));
    registry.register(overlay(
        "sdf/chunk_id",
        "Chunk ID",
        "SDF_DEBUG_CHUNK_ID",
        "Color by resolved chunk key (shade = local slot): same=one chunk, diff=cross-chunk",
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

    // Independent toggle: skip the LOD-0 analytic cubic solver → pure sphere-trace
    // everywhere. If enabling this fixes a visual artifact, the cubic is the cause.
    registry.register(ShaderDebugMode {
        id: "sdf/disable_cubic".into(),
        label: "No cubic".into(),
        shader_define: "SDF_DISABLE_CUBIC".into(),
        kind: DebugModeKind::Toggle,
        description: "Skip the LOD-0 cubic solver (pure sphere-trace)".into(),
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
    registry.register(ShaderDebugMode {
        id: "sdf/shadows".into(),
        label: "Shadows".into(),
        shader_define: "SDF_SHADOWS".into(),
        kind: DebugModeKind::Toggle,
        description: "SDF soft shadows (secondary ray toward the sun)".into(),
    });
    registry.register(ShaderDebugMode {
        id: "sdf/reflections".into(),
        label: "Reflections".into(),
        shader_define: "SDF_REFLECTIONS".into(),
        kind: DebugModeKind::Toggle,
        description: "SDF-traced reflections on metallic/smooth surfaces (secondary ray)"
            .into(),
    });
    registry.register(ShaderDebugMode {
        id: "sdf/parallax".into(),
        label: "Parallax".into(),
        shader_define: "SDF_PARALLAX".into(),
        kind: DebugModeKind::Toggle,
        description: "Inward relief from the height map (carve within the envelope, no silhouette change)".into(),
    });
    registry.register(ShaderDebugMode {
        id: "sdf/displace".into(),
        label: "Displace".into(),
        shader_define: "SDF_DISPLACE".into(),
        kind: DebugModeKind::Toggle,
        description: "TRUE height displacement: peaks bulge past the envelope (real silhouette). Overrides Parallax; costs a detail march.".into(),
    });

    // Default the PBR feature toggles ON so the enhanced shading shows without hunting
    // for the checkbox. The state resource is separate from the registry; seed it after
    // the `registry` borrow above ends (NLL drops it at last use).
    {
        let mut state = app.world_mut().resource_mut::<ShaderDebugState>();
        state.set("sdf/shadows", true);
        state.set("sdf/reflections", true);
        state.set("sdf/parallax", true);
    }
}

// --- Atlas stats ---

fn update_atlas_stats(mut stats: ResMut<SdfAtlasStats>, atlas: Res<SdfAtlas>) {
    let total = atlas.bricks.len() as u64;
    let voxels = total * BRICK_VOXELS as u64;

    // GPU footprint per voxel: distance R16Snorm (2B) + two dense material-distance
    // atlases, mat_lo and mat_hi, each Rgba16Snorm (8B = 4 materials × 2B). Plus the
    // per-brick lookup entry (16B, std430). Matches the formats uploaded in render.rs.
    // (`object_bytes`/`blend_bytes` fields are reused for mat_lo/mat_hi.)
    stats.dist_bytes = voxels * 2;
    stats.object_bytes = voxels * 8; // mat_lo (materials 0..3)
    stats.blend_bytes = voxels * 8; // mat_hi (materials 4..7)
    stats.lookup_bytes = total * 16;
    stats.total_bytes =
        stats.dist_bytes + stats.object_bytes + stats.blend_bytes + stats.lookup_bytes;

    stats.total_bricks = total as u32;
    // 2D-tiled dims (matches the render atlas + preview): tiles wrap at 256/row.
    let tiles_per_row: u32 = 256;
    let num_rows = (total as u32).div_ceil(tiles_per_row).max(1);
    stats.atlas_width = tiles_per_row * (BRICK_EDGE * BRICK_EDGE) as u32;
    stats.atlas_height = num_rows * BRICK_EDGE as u32;
    stats.dirty = atlas.rebake_all || !atlas.dirty_bricks.is_empty();
}

/// Human-readable byte size (B / KB / MB).
fn fmt_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

// --- Atlas texture viewer ---

/// Rebuild the egui-displayable atlas images whenever the CPU atlas changes.
/// Uses the same tile layout as `render::extract_sdf_atlas` so the on-screen
/// image matches what the GPU samples.
fn update_atlas_textures(
    atlas: Res<SdfAtlas>,
    mut images: ResMut<Assets<Image>>,
    mut egui_textures: ResMut<EguiUserTextures>,
    mut tex: ResMut<SdfAtlasTextures>,
) {
    if !atlas.is_changed() {
        return;
    }

    let num_bricks = atlas.bricks.len() as u32;
    if num_bricks == 0 {
        return;
    }

    let edge = BRICK_EDGE as u32;
    let tile_width = edge * edge; // 64
    // 2D-tile (wrap into rows) so the egui preview image never exceeds the GPU's
    // max texture dimension — mirrors the render atlas packing.
    let tiles_per_row: u32 = 256;
    let num_rows = num_bricks.div_ceil(tiles_per_row);
    let width = tiles_per_row * tile_width;
    let height = num_rows * edge;
    let pixels = (width * height) as usize;

    let mut dist_rgba = vec![0u8; pixels * 4];
    let mut object_rgba = vec![0u8; pixels * 4];

    for (i, packed) in atlas.bricks.values().enumerate() {
        let tile = i as u32;
        let col_px = (tile % tiles_per_row) * tile_width;
        let row_px = (tile / tiles_per_row) * edge;
        for z in 0..edge {
            for y in 0..edge {
                for x in 0..edge {
                    let src = (z * edge * edge + y * edge + x) as usize;
                    let dst_u = col_px + y * edge + x;
                    let dst_v = row_px + z;
                    let dst = (dst_v * width + dst_u) as usize;

                    // Distance: snorm [-1,1] -> grayscale, with the zero-crossing
                    // at mid gray so surfaces read as a clear edge.
                    let d = packed.dist[src] as f32 / 32767.0;
                    let g = ((d * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                    dist_rgba[dst * 4] = g;
                    dist_rgba[dst * 4 + 1] = g;
                    dist_rgba[dst * 4 + 2] = g;
                    dist_rgba[dst * 4 + 3] = 255;

                    // Material = argmin over the K palette-slot distances (what the
                    // shader resolves per pixel), mapped through the brick palette to
                    // a global id -> distinct palette color.
                    let base = src * PALETTE_K;
                    let mut best = 0usize;
                    for k in 1..PALETTE_K {
                        if packed.mat_dist[base + k] < packed.mat_dist[base + best] {
                            best = k;
                        }
                    }
                    let global_id = packed.palette[best];
                    let [r, gg, b] = object_color((global_id & 0xff) as u8);
                    object_rgba[dst * 4] = r;
                    object_rgba[dst * 4 + 1] = gg;
                    object_rgba[dst * 4 + 2] = b;
                    object_rgba[dst * 4 + 3] = 255;
                }
            }
        }
    }

    let size = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    // Nearest sampling so the upscaled atlas shows crisp per-texel detail in the
    // panel instead of a blurry smear.
    let sampler = bevy::image::ImageSampler::nearest();
    let mut dist_img = Image::new(
        size,
        TextureDimension::D2,
        dist_rgba,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::all(),
    );
    dist_img.sampler = sampler.clone();
    let mut object_img = Image::new(
        size,
        TextureDimension::D2,
        object_rgba,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::all(),
    );
    object_img.sampler = sampler;

    tex.dist_handle = images.add(dist_img);
    tex.object_handle = images.add(object_img);
    tex.dist_id = Some(egui_textures.add_image(EguiTextureHandle::Strong(tex.dist_handle.clone())));
    tex.object_id =
        Some(egui_textures.add_image(EguiTextureHandle::Strong(tex.object_handle.clone())));
    tex.width = width;
    tex.height = height;
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

/// Panel: resident-chunk count + the overlay toggle (mirrors `bvh_panel`).
fn chunk_panel(world: &mut World, ui: &mut egui::Ui) {
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
    ui.separator();
    {
        // Diagnostic: bypass the async/incremental bake + partial upload, rebuilding the
        // whole atlas synchronously each change. If a visual bug disappears here, it lives
        // in the async path.
        let mut sync = world.resource_mut::<super::SyncBakeMode>();
        ui.checkbox(&mut sync.0, "Sync bake (no async)");
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

fn atlas_panel(world: &mut World, ui: &mut egui::Ui) {
    let stats = world.resource::<SdfAtlasStats>();
    ui.label(format!("Bricks: {}", stats.total_bricks));
    ui.label(format!(
        "Texels: {}x{}",
        stats.atlas_width, stats.atlas_height
    ));
    let (color, text) = if stats.dirty {
        (egui::Color32::YELLOW, "DIRTY")
    } else {
        (egui::Color32::GREEN, "CLEAN")
    };
    ui.colored_label(color, text);

    ui.separator();
    ui.strong(format!("GPU memory: {}", fmt_bytes(stats.total_bytes)));
    egui::Grid::new("sdf_atlas_mem")
        .num_columns(2)
        .show(ui, |ui| {
            ui.label("Distance (R16)");
            ui.label(fmt_bytes(stats.dist_bytes));
            ui.end_row();
            ui.label("Mat lo (Rgba16)");
            ui.label(fmt_bytes(stats.object_bytes));
            ui.end_row();
            ui.label("Mat hi (Rgba16)");
            ui.label(fmt_bytes(stats.blend_bytes));
            ui.end_row();
            ui.label("Lookup buffer");
            ui.label(fmt_bytes(stats.lookup_bytes));
            ui.end_row();
        });

    // Atlas images. Native layout is extreme aspect (num_bricks*64 wide x 8 tall),
    // so display each row scaled up to a readable height inside a horizontal
    // scroll area and let the user zoom the row height.
    let (dist_id, object_id, tex_w, tex_h) = {
        let tex = world.resource::<SdfAtlasTextures>();
        (tex.dist_id, tex.object_id, tex.width, tex.height)
    };
    if dist_id.is_none() && object_id.is_none() {
        return;
    }

    ui.separator();
    let row_h = world.resource::<SdfAtlasTextures>().view_height;
    let mut h = row_h;
    if ui
        .add(egui::Slider::new(&mut h, 24.0..=256.0).text("Atlas zoom (px tall)"))
        .changed()
    {
        world.resource_mut::<SdfAtlasTextures>().view_height = h;
    }

    let aspect = if tex_h > 0 {
        tex_w as f32 / tex_h as f32
    } else {
        1.0
    };
    let img_w = h * aspect;

    egui::ScrollArea::horizontal()
        .id_salt("sdf_atlas_imgs")
        .show(ui, |ui| {
            if let Some(id) = dist_id {
                ui.label("Distance atlas");
                ui.image(egui::load::SizedTexture::new(id, egui::vec2(img_w, h)));
            }
            if let Some(id) = object_id {
                ui.label("Object-id atlas");
                ui.image(egui::load::SizedTexture::new(id, egui::vec2(img_w, h)));
            }
        });
}

fn gizmo_panel(world: &mut World, ui: &mut egui::Ui) {
    let selection = world.resource::<SdfSelection>();
    match selection.entity {
        Some(e) => ui.label(format!("Selected: Entity {:?}", e.index())),
        None => ui.label("Selected: None"),
    };

    let orbit = world.resource::<SdfOrbitCamera>();
    ui.separator();
    ui.label(format!("Cam dist: {:.2}", orbit.distance));
    ui.label(format!("Yaw: {:.2}  Pitch: {:.2}", orbit.yaw, orbit.pitch));
    ui.label(format!(
        "Target: {:.2}, {:.2}, {:.2}",
        orbit.target.x, orbit.target.y, orbit.target.z
    ));
}

fn wireframe_panel(world: &mut World, ui: &mut egui::Ui) {
    let mut visible = world.resource::<WireframeBoundsVisible>().0;
    if ui
        .checkbox(&mut visible, "Show bounds wireframes")
        .changed()
    {
        world.resource_mut::<WireframeBoundsVisible>().0 = visible;
    }
}

fn modes_panel(world: &mut World, ui: &mut egui::Ui) {
    debug_modes_ui(world, ui);
}

fn raymarch_panel(world: &mut World, ui: &mut egui::Ui) {
    let mut params = world.resource_mut::<SdfRaymarchParams>();
    ui.add(egui::Slider::new(&mut params.max_steps, 16..=512).text("Steps"));
    ui.add(egui::Slider::new(&mut params.max_dist, 10.0..=500.0).text("Max Dist"));
    ui.add(egui::Slider::new(&mut params.sdf_eps, 0.0001..=0.1).text("Epsilon"));
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

fn bvh_panel(world: &mut World, ui: &mut egui::Ui) {
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

// --- Spawn panel ---

fn spawn_panel(world: &mut World, ui: &mut egui::Ui) {
    // Snapshot panel state for the combo boxes.
    let (mut kind, mut op, mut smoothing) = {
        let s = world.resource::<SpawnState>();
        (s.kind, s.op, s.smoothing)
    };

    egui::ComboBox::from_label("Primitive")
        .selected_text(kind.label())
        .show_ui(ui, |ui| {
            for k in SpawnKind::ALL {
                ui.selectable_value(&mut kind, k, k.label());
            }
        });

    egui::ComboBox::from_label("Operation")
        .selected_text(format!("{op:?}"))
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut op, CsgKind::Union, "Union");
            ui.selectable_value(&mut op, CsgKind::Subtract, "Subtract");
            ui.selectable_value(&mut op, CsgKind::Intersect, "Intersect");
        });

    ui.add(egui::Slider::new(&mut smoothing, 0.0..=1.0).text("Smoothing"));

    {
        let mut s = world.resource_mut::<SpawnState>();
        s.kind = kind;
        s.op = op;
        s.smoothing = smoothing;
    }

    // Material picker: choose an existing registry material, or spawn a fresh one.
    // (A brick still only shows its K=4 nearest materials, but the world/registry is
    // unbounded — no global cap.)
    let mut spawn_mat = world.resource::<SpawnState>().material;
    let mat_names: Vec<(u32, String)> = {
        let reg = world.resource::<super::edits::MaterialRegistry>();
        reg.defs
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let c = d.base_color.to_srgba();
                (
                    i as u32,
                    format!("#{i} ({:.2},{:.2},{:.2})", c.red, c.green, c.blue),
                )
            })
            .collect()
    };
    let sel_label = mat_names
        .iter()
        .find(|(i, _)| *i == spawn_mat)
        .map(|(_, n)| n.clone())
        .unwrap_or_else(|| "New material".into());
    egui::ComboBox::from_label("Material")
        .selected_text(sel_label)
        .show_ui(ui, |ui| {
            // u32::MAX sentinel = "create a fresh registry material on spawn".
            ui.selectable_value(&mut spawn_mat, u32::MAX, "New material");
            for (id, name) in &mat_names {
                ui.selectable_value(&mut spawn_mat, *id, name);
            }
        });
    world.resource_mut::<SpawnState>().material = spawn_mat;

    ui.horizontal(|ui| {
        if ui.button("Spawn").clicked() {
            // Scatter around the orbit target so successive spawns don't stack on
            // top of each other. A small random offset keeps them in view.
            let target = world.resource::<SdfOrbitCamera>().target;
            let pos = target + random_spawn_offset();
            let next_order = world
                .query_filtered::<&SdfOrder, With<SdfVolume>>()
                .iter(world)
                .map(|o| o.0)
                .max()
                .map(|m| m + 1)
                .unwrap_or(0);

            // Resolve the material id: either the picked existing one, or a new
            // registry entry with a distinct palette colour.
            let registry_id = if spawn_mat == u32::MAX {
                let mut reg = world.resource_mut::<super::edits::MaterialRegistry>();
                let id = reg.defs.len() as u32;
                reg.defs.push(super::edits::MaterialDef {
                    base_color: spawn_color(id as usize),
                    blend_softness: 0.0,
                    ..Default::default()
                });
                id
            } else {
                spawn_mat
            };

            world.spawn((
                Transform::from_translation(pos),
                kind.default_primitive(),
                SdfOp {
                    kind: op,
                    smoothing,
                },
                SdfOrder(next_order),
                SdfMaterial { registry_id },
                SdfVolume,
                SceneEntity,
            ));
            // Spawn changes the edit set → `schedule_bakes` detects the new
            // entity and re-dirties the affected chunks.
        }

        if ui.button("Delete selected").clicked() {
            let selected = world.resource::<SdfSelection>().entity;
            if let Some(e) = selected {
                world.despawn(e);
                world.resource_mut::<SdfSelection>().entity = None;
                // Despawn changes the edit set → full rebuild on the next bake.
            }
        }
    });
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

// --- Inspect panel ---

fn inspect_panel(world: &mut World, ui: &mut egui::Ui) {
    let Some(entity) = world.resource::<SdfSelection>().entity else {
        ui.label("No edit selected. Click an edit in the viewport.");
        return;
    };

    // Pull the current components out, edit copies in the UI, write back if changed.
    let Ok((mut prim, mut op, mut order, mut material)) = world
        .query::<(&SdfPrimitive, &SdfOp, &SdfOrder, &SdfMaterial)>()
        .get(world, entity)
        .map(|(p, o, ord, m)| (p.clone(), *o, *ord, *m))
    else {
        ui.label("Selected entity is not an SDF edit.");
        return;
    };

    let mut changed = false;
    changed |= primitive_params_ui(ui, &mut prim);

    ui.separator();
    let mut op_kind = op.kind;
    egui::ComboBox::from_label("Operation")
        .selected_text(format!("{op_kind:?}"))
        .show_ui(ui, |ui| {
            for (k, name) in [
                (CsgKind::Union, "Union"),
                (CsgKind::Subtract, "Subtract"),
                (CsgKind::Intersect, "Intersect"),
            ] {
                if ui.selectable_value(&mut op_kind, k, name).changed() {
                    changed = true;
                }
            }
        });
    if op_kind != op.kind {
        op.kind = op_kind;
        changed = true;
    }
    if ui
        .add(egui::Slider::new(&mut op.smoothing, 0.0..=1.0).text("Smoothing"))
        .changed()
    {
        changed = true;
    }

    let mut order_v = order.0;
    if ui
        .add(
            egui::DragValue::new(&mut order_v)
                .range(0..=64)
                .prefix("Order "),
        )
        .changed()
    {
        order.0 = order_v;
        changed = true;
    }

    ui.separator();

    // Material assignment: pick which registry material this edit uses.
    let mat_names: Vec<(u32, String)> = {
        let reg = world.resource::<super::edits::MaterialRegistry>();
        reg.defs
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let c = d.base_color.to_srgba();
                (
                    i as u32,
                    format!("#{i} ({:.2},{:.2},{:.2})", c.red, c.green, c.blue),
                )
            })
            .collect()
    };
    let cur_label = mat_names
        .iter()
        .find(|(i, _)| *i == material.registry_id)
        .map(|(_, n)| n.clone())
        .unwrap_or_else(|| "?".into());
    egui::ComboBox::from_label("Material")
        .selected_text(cur_label)
        .show_ui(ui, |ui| {
            for (id, name) in &mat_names {
                if ui
                    .selectable_value(&mut material.registry_id, *id, name)
                    .changed()
                {
                    changed = true;
                }
            }
        });

    // Edit the *referenced registry material's* appearance (shared by every edit
    // that uses it). Color + seam blend softness.
    let mut reg_changed = false;
    {
        let mut reg = world.resource_mut::<super::edits::MaterialRegistry>();
        if let Some(def) = reg.defs.get_mut(material.registry_id as usize) {
            let lin = def.base_color.to_linear();
            let mut rgb = [lin.red, lin.green, lin.blue];
            if ui.color_edit_button_rgb(&mut rgb).changed() {
                def.base_color = Color::linear_rgb(rgb[0], rgb[1], rgb[2]);
                reg_changed = true;
            }
            // Per-material colour-feather width at seams (world units). Does not
            // affect geometry — see SdfOp::smoothing for that.
            if ui
                .add(egui::Slider::new(&mut def.blend_softness, 0.0..=1.0).text("Blend softness"))
                .changed()
            {
                reg_changed = true;
            }
            // Scalar metallic/roughness — these drive shading only when the material has
            // NO MRA texture (the textureless exemplars). For a textured material the MRA
            // map wins and these sliders have no visible effect.
            if ui
                .add(egui::Slider::new(&mut def.metallic, 0.0..=1.0).text("Metallic"))
                .changed()
            {
                reg_changed = true;
            }
            if ui
                .add(egui::Slider::new(&mut def.roughness, 0.0..=1.0).text("Roughness"))
                .changed()
            {
                reg_changed = true;
            }
            // Parallax relief depth (UV units). Only visible with a height map + SDF_PARALLAX.
            if ui
                .add(egui::Slider::new(&mut def.parallax_scale, 0.0..=0.4).text("Parallax"))
                .changed()
            {
                reg_changed = true;
            }
        }
    }

    if changed && let Ok(mut e) = world.get_entity_mut(entity) {
        if let Some(mut p) = e.get_mut::<SdfPrimitive>() {
            *p = prim;
        }
        if let Some(mut o) = e.get_mut::<SdfOp>() {
            *o = op;
        }
        if let Some(mut ord) = e.get_mut::<SdfOrder>() {
            *ord = order;
        }
        if let Some(mut m) = e.get_mut::<SdfMaterial>() {
            *m = material;
        }
    }
    // Geometry/material-id edits go through `get_mut` above, which triggers the
    // `Changed<…>` filters `schedule_bakes` watches → targeted rebake. A
    // registry colour/softness change is shading-only (the GPU material table
    // re-uploads on registry change), so it needs no rebake.
    let _ = (changed, reg_changed);
}

/// Per-primitive parameter sliders. Returns true if any value changed.
fn primitive_params_ui(ui: &mut egui::Ui, prim: &mut SdfPrimitive) -> bool {
    let mut changed = false;
    match prim {
        SdfPrimitive::Sphere { radius } => {
            changed |= ui
                .add(egui::Slider::new(radius, 0.05..=3.0).text("Radius"))
                .changed();
        }
        SdfPrimitive::Box { half_extents } => {
            changed |= ui
                .add(egui::Slider::new(&mut half_extents.x, 0.05..=3.0).text("Half X"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut half_extents.y, 0.05..=3.0).text("Half Y"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut half_extents.z, 0.05..=3.0).text("Half Z"))
                .changed();
        }
        SdfPrimitive::Torus { major, minor } => {
            changed |= ui
                .add(egui::Slider::new(major, 0.1..=3.0).text("Major"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(minor, 0.02..=1.0).text("Minor"))
                .changed();
        }
        SdfPrimitive::Capsule {
            half_height,
            radius,
        } => {
            changed |= ui
                .add(egui::Slider::new(half_height, 0.05..=3.0).text("Half height"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(radius, 0.05..=2.0).text("Radius"))
                .changed();
        }
        SdfPrimitive::Cylinder {
            radius,
            half_height,
        } => {
            changed |= ui
                .add(egui::Slider::new(radius, 0.05..=2.0).text("Radius"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(half_height, 0.05..=3.0).text("Half height"))
                .changed();
        }
        SdfPrimitive::Heightmap {
            half_xz,
            max_height,
            freq,
            amp,
            seed,
        } => {
            changed |= ui
                .add(egui::Slider::new(&mut half_xz.x, 0.5..=5.0).text("Half X"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut half_xz.y, 0.5..=5.0).text("Half Z"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(max_height, 0.1..=3.0).text("Max height"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(freq, 0.1..=5.0).text("Freq"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(amp, 0.0..=2.0).text("Amp"))
                .changed();
            let mut seed_v = *seed;
            if ui
                .add(egui::DragValue::new(&mut seed_v).prefix("Seed "))
                .changed()
            {
                *seed = seed_v;
                changed = true;
            }
        }
    }
    changed
}
