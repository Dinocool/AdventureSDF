//! GPU node-preview raymarch (stages 1-3) — see `docs/GPU_PREVIEW_RAYMARCH_PLAN.md` (Option B).
//!
//! The CPU bakes the worldgen graph's height + analytic normal into an `Rgba32Float` texture (the single
//! `Graph::eval` source of truth — no noise re-implemented on the GPU), and a custom-material fullscreen
//! quad raymarches that texture in WGSL with an orbit camera passed as a uniform. So **rotating is
//! pure-GPU** (the heightfield only re-bakes when the graph or zoom window changes) and high-res stays
//! cheap. Rendered offscreen into an egui-displayed image (a debug "GPU Preview" panel drives the live
//! `WorldGraph`). Stage 4 will wire this per-node; for now it previews the whole world graph.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::{RenderTarget, ScalingMode};
use bevy::image::{Image, ImageSampler};
use bevy::pbr::{Material, MaterialPlugin, MeshMaterial3d};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::shader::ShaderRef;
use bevy_egui::{EguiTextureHandle, EguiUserTextures, egui};

use crate::sdf_render::worldgen::WorldGraph;
use crate::sdf_render::worldgen::graph::node::Graph;

/// Render layer the preview rig lives on (isolated — material preview uses 17, thumbnails 16).
const PREVIEW_LAYER: usize = 18;
/// Offscreen output image edge length (px).
const PREVIEW_SIZE: u32 = 384;
/// Baked heightfield texture resolution (per side).
const HEIGHTFIELD_RES: usize = 256;
/// Seed used for the preview bake (matches the editor's CPU previews + the default world seed).
const PREVIEW_SEED: u64 = 7;
const SEA_LEVEL: f32 = 0.0;
const SNOW_LEVEL: f32 = 1000.0;
const WATER_DEPTH: f32 = 400.0;

/// Camera + framing uniform (scalars packed into the `w` lanes to keep std140 alignment trivial).
#[derive(ShaderType, Clone, Copy, Default)]
struct PreviewParams {
    eye: Vec4,    // xyz eye, w = image-plane tan
    fwd: Vec4,    // xyz forward, w = world half-extent
    right: Vec4,  // xyz right, w = height min
    up: Vec4,     // xyz up, w = height max
    levels: Vec4, // sea, snow, water-depth, heightfield res
}

/// Custom fullscreen-quad material: the camera uniform + the CPU-baked height/normal texture.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
struct HeightPreviewMaterial {
    #[uniform(0)]
    params: PreviewParams,
    // Rgba32Float, fetched via `textureLoad` (unfilterable → manual bilinear in the shader).
    #[texture(1, sample_type = "float", filterable = false)]
    height: Handle<Image>,
}

impl Material for HeightPreviewMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/worldgen_preview.wgsl".into()
    }
}

/// The offscreen rig handles.
#[derive(Resource)]
struct GpuPreviewRig {
    tex_id: egui::TextureId,
    material: Handle<HeightPreviewMaterial>,
    height: Handle<Image>,
}

/// Panel-driven view state (orbit + zoom) + the last bake fingerprint.
#[derive(Resource)]
struct GpuPreviewView {
    yaw: f32,
    pitch: f32,
    half: f64,
    ymin: f32,
    ymax: f32,
    baked_key: u64,
}

impl Default for GpuPreviewView {
    fn default() -> Self {
        Self { yaw: 0.7, pitch: 0.6, half: 2048.0, ymin: 0.0, ymax: 1.0, baked_key: 0 }
    }
}

/// Offscreen render-attachment image (the raymarch output egui samples).
fn make_output_image(images: &mut Assets<Image>) -> Handle<Image> {
    let size = Extent3d { width: PREVIEW_SIZE, height: PREVIEW_SIZE, depth_or_array_layers: 1 };
    let mut image =
        Image::new_fill(size, TextureDimension::D2, &[0, 0, 0, 0], TextureFormat::Rgba8UnormSrgb, RenderAssetUsages::all());
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    image.sampler = ImageSampler::linear();
    images.add(image)
}

/// The baked height+normal texture (Rgba32Float: R = height m, GBA = normal), filled by [`bake_height`].
fn make_height_image(images: &mut Assets<Image>) -> Handle<Image> {
    let n = HEIGHTFIELD_RES as u32;
    let data = vec![0u8; (n * n) as usize * 16]; // 4 × f32
    let mut image = Image::new(
        Extent3d { width: n, height: n, depth_or_array_layers: 1 },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba32Float,
        RenderAssetUsages::all(),
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    image.sampler = ImageSampler::nearest();
    images.add(image)
}

/// Evaluate the graph over the ±`half` window into `image` (height + analytic normal); returns the height
/// range (for camera framing). The single `Graph::eval` — no GPU noise.
fn bake_height(image: &mut Image, g: &Graph, half: f64) -> (f32, f32) {
    let n = HEIGHTFIELD_RES;
    let mut data = vec![0f32; n * n * 4];
    let (mut ymin, mut ymax) = (f32::INFINITY, f32::NEG_INFINITY);
    for j in 0..n {
        for i in 0..n {
            let wx = -half + (i as f64 + 0.5) / n as f64 * 2.0 * half;
            let wz = -half + (j as f64 + 0.5) / n as f64 * 2.0 * half;
            let f = g.eval(wx, wz, PREVIEW_SEED);
            let h = f.v as f32;
            let nrm = Vec3::new(-f.dx as f32, 1.0, -f.dz as f32).normalize();
            let k = (j * n + i) * 4;
            data[k] = h;
            data[k + 1] = nrm.x;
            data[k + 2] = nrm.y;
            data[k + 3] = nrm.z;
            if h.is_finite() {
                ymin = ymin.min(h);
                ymax = ymax.max(h);
            }
        }
    }
    if !ymin.is_finite() {
        ymin = 0.0;
        ymax = 1.0;
    }
    image.data = Some(bytemuck::cast_slice(&data).to_vec());
    (ymin, ymax)
}

/// Fingerprint the bake inputs (graph + zoom) so the heightfield is only re-baked on change.
fn bake_key(g: &Graph, half: f64) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ron::to_string(g).unwrap_or_default().hash(&mut h);
    half.to_bits().hash(&mut h);
    h.finish()
}

/// Orbit camera + framing → the shader uniform.
fn build_params(view: &GpuPreviewView) -> PreviewParams {
    let half = view.half as f32;
    let (ymin, ymax) = (view.ymin, view.ymax);
    let span = (ymax - ymin).max(1.0);
    let centre = Vec3::new(0.0, (ymin + ymax) * 0.5, 0.0);
    let dist = half * 2.4 + span;
    let (sp, cp) = view.pitch.sin_cos();
    let (sy, cyaw) = view.yaw.sin_cos();
    let eye = centre + Vec3::new(cp * cyaw, sp, cp * sy) * dist;
    let fwd = (centre - eye).normalize();
    let right = fwd.cross(Vec3::Y).normalize_or_zero();
    let up = right.cross(fwd);
    let tan = (0.6f32 * 0.5).tan() * 2.0;
    PreviewParams {
        eye: eye.extend(tan),
        fwd: fwd.extend(half),
        right: right.extend(ymin),
        up: up.extend(ymax),
        levels: Vec4::new(SEA_LEVEL, SNOW_LEVEL, WATER_DEPTH, HEIGHTFIELD_RES as f32),
    }
}

/// Spawn the rig once: a fullscreen quad with the custom material + an ortho camera into an egui image.
fn setup_gpu_preview(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<HeightPreviewMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut egui_textures: ResMut<EguiUserTextures>,
    existing: Option<Res<GpuPreviewRig>>,
) {
    if existing.is_some() {
        return;
    }
    let layer = RenderLayers::layer(PREVIEW_LAYER);
    let height = make_height_image(&mut images);
    let quad = meshes.add(Rectangle::new(2.0, 2.0));
    let material = materials.add(HeightPreviewMaterial { params: PreviewParams::default(), height: height.clone() });
    commands.spawn((Mesh3d(quad), MeshMaterial3d(material.clone()), Transform::IDENTITY, layer.clone()));

    let output = make_output_image(&mut images);
    commands.spawn((
        Camera3d::default(),
        Camera { order: -6, clear_color: ClearColorConfig::Custom(Color::BLACK), ..default() },
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::Fixed { width: 2.0, height: 2.0 },
            ..OrthographicProjection::default_3d()
        }),
        RenderTarget::Image(output.clone().into()),
        Transform::from_xyz(0.0, 0.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
        layer,
    ));

    let tex_id = egui_textures.add_image(EguiTextureHandle::Strong(output));
    commands.insert_resource(GpuPreviewRig { tex_id, material, height });
}

/// Re-bake the heightfield when the graph/zoom changes; update the camera uniform every frame (cheap, so
/// rotation is pure-GPU).
fn drive_gpu_preview(
    rig: Option<Res<GpuPreviewRig>>,
    world_graph: Option<Res<WorldGraph>>,
    mut view: ResMut<GpuPreviewView>,
    mut materials: ResMut<Assets<HeightPreviewMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let (Some(rig), Some(world_graph)) = (rig, world_graph) else {
        return;
    };
    let g: &Graph = &world_graph.0;
    let key = bake_key(g, view.half);
    if key != view.baked_key
        && let Some(img) = images.get_mut(&rig.height)
    {
        let (ymin, ymax) = bake_height(img, g, view.half);
        view.ymin = ymin;
        view.ymax = ymax;
        view.baked_key = key;
    }
    if let Some(mat) = materials.get_mut(&rig.material) {
        mat.params = build_params(&view);
    }
}

/// Debug panel: orbit (drag) + zoom + the GPU-raymarched image.
fn gpu_preview_panel(world: &mut World, ui: &mut egui::Ui) {
    let Some(tex) = world.get_resource::<GpuPreviewRig>().map(|r| r.tex_id) else {
        ui.label("GPU preview initialising…");
        return;
    };
    world.resource_scope::<GpuPreviewView, ()>(|_w, mut view| {
        ui.horizontal(|ui| {
            let mut km = view.half * 2.0 / 1000.0;
            if ui.add(egui::DragValue::new(&mut km).speed(0.5).range(0.1..=512.0).suffix(" km")).changed() {
                view.half = (km * 1000.0 / 2.0).max(1.0);
            }
            ui.label("· drag to orbit");
        });
        let resp = ui.add(
            egui::Image::new(egui::load::SizedTexture::new(tex, egui::vec2(384.0, 384.0)))
                .sense(egui::Sense::drag()),
        );
        if resp.dragged() {
            let d = resp.drag_delta();
            view.yaw += d.x * 0.01;
            view.pitch = (view.pitch - d.y * 0.01).clamp(0.05, 1.5);
        }
    });
}

/// Plugin: the custom material pipeline + the offscreen rig + the debug panel.
pub struct WorldgenGpuPreviewPlugin;

impl Plugin for WorldgenGpuPreviewPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<HeightPreviewMaterial>::default());
        app.init_resource::<GpuPreviewView>();
        app.add_systems(
            Update,
            (setup_gpu_preview, drive_gpu_preview).chain().run_if(resource_exists::<EguiUserTextures>),
        );
        super::panels::register_panel(
            app,
            "worldgen/gpu-preview",
            "GPU Preview",
            super::panels::DockSide::Right,
            31,
            gpu_preview_panel,
        );
    }
}
