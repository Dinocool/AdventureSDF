//! GPU node-preview raymarch — **Stage 1: plumbing**.
//!
//! Proves the custom-material → offscreen `Image` → egui texture pipeline end to end before the real
//! raymarch is built on top (see `docs/GPU_PREVIEW_RAYMARCH_PLAN.md`). A fullscreen quad with a custom
//! [`HeightPreviewMaterial`] is rendered by an orthographic camera (on an isolated render layer) into an
//! offscreen image that a debug "GPU Preview" panel displays. The Stage-1 shader just draws a UV gradient
//! — if that shows up, the pipeline works and Stage 2/3 (heightfield bake + WGSL march) slot in.

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

/// Render layer the preview rig lives on (isolated — material preview uses 17, thumbnails 16).
const PREVIEW_LAYER: usize = 18;
/// Offscreen image edge length.
const PREVIEW_SIZE: u32 = 320;

/// Per-preview uniform. Stage 1 uses only `tint.b` (a constant blue); Stage 3 adds camera/zoom/levels.
#[derive(ShaderType, Clone, Copy, Default)]
struct PreviewParams {
    tint: Vec4,
}

/// Custom fullscreen-quad material whose fragment shader will (Stage 3) raymarch a baked heightfield.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
struct HeightPreviewMaterial {
    #[uniform(0)]
    params: PreviewParams,
}

impl Material for HeightPreviewMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/worldgen_preview.wgsl".into()
    }
}

/// The offscreen rig's egui texture id, set once the rig is built.
#[derive(Resource)]
struct GpuPreviewRig {
    tex_id: egui::TextureId,
}

/// Build the offscreen image (continuous render target, sampled by egui).
fn make_image(images: &mut Assets<Image>) -> Handle<Image> {
    let size = Extent3d { width: PREVIEW_SIZE, height: PREVIEW_SIZE, depth_or_array_layers: 1 };
    let mut image =
        Image::new_fill(size, TextureDimension::D2, &[0, 0, 0, 0], TextureFormat::Rgba8UnormSrgb, RenderAssetUsages::all());
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    image.sampler = ImageSampler::linear();
    images.add(image)
}

/// Spawn the rig once: a fullscreen quad with the custom material + an ortho camera on [`PREVIEW_LAYER`]
/// rendering into an egui-registered image.
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
    let quad = meshes.add(Rectangle::new(2.0, 2.0));
    let material = materials.add(HeightPreviewMaterial { params: PreviewParams { tint: Vec4::new(0.0, 0.0, 0.6, 1.0) } });
    commands.spawn((Mesh3d(quad), MeshMaterial3d(material), Transform::IDENTITY, layer.clone()));

    let image = make_image(&mut images);
    commands.spawn((
        Camera3d::default(),
        Camera { order: -6, clear_color: ClearColorConfig::Custom(Color::BLACK), ..default() },
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::Fixed { width: 2.0, height: 2.0 },
            ..OrthographicProjection::default_3d()
        }),
        RenderTarget::Image(image.clone().into()),
        Transform::from_xyz(0.0, 0.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
        layer,
    ));

    let tex_id = egui_textures.add_image(EguiTextureHandle::Strong(image));
    commands.insert_resource(GpuPreviewRig { tex_id });
}

/// Debug panel: show the offscreen render so the Stage-1 plumbing can be eyeballed.
fn gpu_preview_panel(world: &mut World, ui: &mut egui::Ui) {
    ui.label("GPU preview (stage 1): custom-material → offscreen → egui.");
    ui.label("Expect a UV gradient — red →, green ↑, constant blue. If you see it, the pipeline works.");
    ui.separator();
    match world.get_resource::<GpuPreviewRig>() {
        Some(rig) => {
            ui.image(egui::load::SizedTexture::new(rig.tex_id, egui::vec2(320.0, 320.0)));
        }
        None => {
            ui.label("initialising…");
        }
    }
}

/// Plugin: the custom material pipeline + the offscreen rig + the debug panel.
pub struct WorldgenGpuPreviewPlugin;

impl Plugin for WorldgenGpuPreviewPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<HeightPreviewMaterial>::default());
        app.add_systems(Update, setup_gpu_preview.run_if(resource_exists::<EguiUserTextures>));
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
