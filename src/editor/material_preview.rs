//! Live, interactive material preview. Unlike the assets-browser thumbnail (a frozen
//! one-shot readback capture), this is a persistent offscreen rig that renders
//! **continuously** to an egui-displayed image: a mesh with the material applied, lit,
//! framed by a camera the user can orbit by click-dragging the preview. The preview
//! shape switches between Sphere / Cube / Torus.
//!
//! The inspector points [`MaterialPreviewState::material`] at the `StandardMaterial`
//! it builds from the edited `MaterialAsset`; because the rig renders live, edits show
//! up the next frame with no recapture.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::RenderTarget;
use bevy::image::{Image, ImageSampler};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy_egui::{EguiTextureHandle, EguiUserTextures, egui};

/// Render layer the preview rig lives on (isolated from the main view and the thumbnail
/// rig on layer 16).
const PREVIEW_LAYER: usize = 17;
/// Offscreen image edge length.
const PREVIEW_SIZE: u32 = 320;

/// Which mesh the preview displays.
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum PreviewShape {
    #[default]
    Sphere,
    Cube,
    Torus,
}

/// Interactive preview state, driven by the material editor UI.
#[derive(Resource)]
pub struct MaterialPreviewState {
    /// Material applied to the preview mesh (set by the editor each frame it's shown).
    pub material: Option<Handle<StandardMaterial>>,
    pub shape: PreviewShape,
    /// Orbit parameters (radians / world units).
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    /// egui texture id for the offscreen image (set once the rig is built).
    pub tex_id: Option<egui::TextureId>,
}

impl Default for MaterialPreviewState {
    fn default() -> Self {
        Self {
            material: None,
            shape: PreviewShape::Sphere,
            yaw: 0.6,
            pitch: 0.4,
            distance: 3.2,
            tex_id: None,
        }
    }
}

/// Handles for the three preview meshes + the offscreen image, built once.
#[derive(Resource)]
struct PreviewRig {
    sphere: Handle<Mesh>,
    cube: Handle<Mesh>,
    torus: Handle<Mesh>,
    mesh_entity: Entity,
    camera: Entity,
}

/// Marker for the preview's mesh entity (material + mesh swapped per state).
#[derive(Component)]
struct PreviewMesh;

/// Marker for the preview camera (transform recomputed from orbit each frame).
#[derive(Component)]
struct PreviewCamera;

/// Build the offscreen image (continuous render target, sampled by egui).
fn make_image(images: &mut Assets<Image>) -> Handle<Image> {
    let size = Extent3d {
        width: PREVIEW_SIZE,
        height: PREVIEW_SIZE,
        depth_or_array_layers: 1,
    };
    let mut image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::all(),
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    image.sampler = ImageSampler::linear();
    images.add(image)
}

/// Spawn the preview rig once: mesh entity + light + camera on [`PREVIEW_LAYER`],
/// rendering continuously into an egui-registered image.
fn setup_preview_rig(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut egui_textures: ResMut<EguiUserTextures>,
    mut state: ResMut<MaterialPreviewState>,
    existing: Option<Res<PreviewRig>>,
) {
    if existing.is_some() {
        return;
    }
    let layer = RenderLayers::layer(PREVIEW_LAYER);
    let sphere = meshes.add(Sphere::new(1.0).mesh().uv(48, 24));
    let cube = meshes.add(Cuboid::from_length(1.6));
    let torus = meshes.add(Torus::new(0.5, 1.0));

    let mesh_entity = commands
        .spawn((
            Mesh3d(sphere.clone()),
            MeshMaterial3d(std_materials.add(StandardMaterial::default())),
            Transform::IDENTITY,
            layer.clone(),
            PreviewMesh,
        ))
        .id();

    commands.spawn((
        DirectionalLight {
            illuminance: 9000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(2.0, 3.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
        layer.clone(),
    ));

    let image = make_image(&mut images);
    let camera = commands
        .spawn((
            Camera3d::default(),
            Camera {
                order: -5,
                clear_color: ClearColorConfig::Custom(Color::srgb(0.12, 0.12, 0.14)),
                ..default()
            },
            RenderTarget::Image(image.clone().into()),
            Transform::from_xyz(0.0, 0.0, state.distance).looking_at(Vec3::ZERO, Vec3::Y),
            layer,
            PreviewCamera,
        ))
        .id();

    state.tex_id = Some(egui_textures.add_image(EguiTextureHandle::Strong(image)));
    commands.insert_resource(PreviewRig {
        sphere,
        cube,
        torus,
        mesh_entity,
        camera,
    });
}

/// Apply the live state each frame: material, shape mesh, and orbit camera transform.
fn drive_preview(
    rig: Option<Res<PreviewRig>>,
    state: Res<MaterialPreviewState>,
    mut meshes: Query<(&mut Mesh3d, &mut MeshMaterial3d<StandardMaterial>), With<PreviewMesh>>,
    mut cameras: Query<&mut Transform, With<PreviewCamera>>,
) {
    let Some(rig) = rig else {
        return;
    };

    if let Ok((mut mesh, mut mat)) = meshes.get_mut(rig.mesh_entity) {
        let want = match state.shape {
            PreviewShape::Sphere => &rig.sphere,
            PreviewShape::Cube => &rig.cube,
            PreviewShape::Torus => &rig.torus,
        };
        if mesh.0.id() != want.id() {
            mesh.0 = want.clone();
        }
        if let Some(material) = &state.material
            && mat.0.id() != material.id()
        {
            mat.0 = material.clone();
        }
    }

    // Orbit camera around the origin from yaw/pitch/distance.
    if let Ok(mut transform) = cameras.get_mut(rig.camera) {
        let (sy, cy) = state.yaw.sin_cos();
        let (sp, cp) = state.pitch.sin_cos();
        let pos = Vec3::new(state.distance * cy * cp, state.distance * sp, state.distance * sy * cp);
        *transform = Transform::from_translation(pos).looking_at(Vec3::ZERO, Vec3::Y);
    }
}

/// Plugin: register state + the preview rig systems. Gated on `EguiUserTextures` like
/// the other editor render rigs.
pub struct MaterialPreviewPlugin;

impl Plugin for MaterialPreviewPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MaterialPreviewState>().add_systems(
            Update,
            (setup_preview_rig, drive_preview)
                .chain()
                .run_if(resource_exists::<EguiUserTextures>),
        );
    }
}
