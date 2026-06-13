//! The editor's 3D **viewport**: the HDR voxel [`SdfCamera`] renders into an offscreen [`Image`], which
//! the dock's "Viewport" tab displays as an egui texture.
//!
//! Why render-to-texture (not show-through): egui can't render onto an HDR camera on the wgpu-trunk fork,
//! so the editor UI lives on a separate **non-HDR `Camera2d` overlay** (see `dock::spawn_editor_egui_camera`)
//! that clears the window and paints the dock. The ray-traced scene therefore can't simply show *through*
//! a transparent panel — instead the SdfCamera renders into an image and the Viewport tab draws that image,
//! resized to the tab so the scene fills the viewport at the right aspect ratio.

use bevy::asset::RenderAssetUsages;
use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy_egui::{EguiTextureHandle, EguiUserTextures, egui};

use crate::sdf_render::SdfCamera;

/// The offscreen image the [`SdfCamera`] renders into, plus its egui texture id and current/desired size.
#[derive(Resource)]
pub struct EditorViewport {
    pub image: Handle<Image>,
    pub texture_id: egui::TextureId,
    /// The image's current pixel size.
    pub size: UVec2,
    /// The size the Viewport tab last requested (px); [`resize_editor_viewport`] grows/shrinks the image
    /// toward it so the 3D isn't aspect-distorted by egui's scaling.
    pub desired_size: UVec2,
}

/// Initial viewport image size (resized to the real tab rect on the first frame it's shown).
const INITIAL: UVec2 = UVec2::new(1280, 720);

pub struct ViewportPlugin;

impl Plugin for ViewportPlugin {
    fn build(&self, app: &mut App) {
        // PostStartup: after `spawn_editor_camera` (Startup) has created the SdfCamera so we can retarget it.
        app.add_systems(PostStartup, setup_editor_viewport)
            .add_systems(Update, resize_editor_viewport);
    }
}

/// A blank render-target image: `RENDER_ATTACHMENT` (a camera draws into it) + `TEXTURE_BINDING` (egui
/// samples it). sRGB 8-bit — the HDR camera tonemaps to this LDR target, which egui then displays.
fn make_render_image(size: UVec2) -> Image {
    let mut image = Image::new_fill(
        Extent3d { width: size.x.max(1), height: size.y.max(1), depth_or_array_layers: 1 },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    image
}

/// Create the viewport image, register it with egui, and retarget the SdfCamera to render INTO it (so the
/// window is left for the egui overlay camera to own).
fn setup_editor_viewport(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut user_textures: ResMut<EguiUserTextures>,
    cam: Query<Entity, With<SdfCamera>>,
) {
    let handle = images.add(make_render_image(INITIAL));
    let texture_id = user_textures.add_image(EguiTextureHandle::Strong(handle.clone()));
    // In Bevy 0.19 the render target is a `RenderTarget` COMPONENT on the camera entity (not a
    // `Camera.target` field) — insert it to redirect the SdfCamera from the window into our image.
    if let Ok(cam_entity) = cam.single() {
        commands
            .entity(cam_entity)
            .insert(RenderTarget::Image(handle.clone().into()));
    }
    commands.insert_resource(EditorViewport {
        image: handle,
        texture_id,
        size: INITIAL,
        desired_size: INITIAL,
    });
}

/// Resize the offscreen image to match the Viewport tab's last-requested size (set by the tab's `ui`).
fn resize_editor_viewport(mut vp: ResMut<EditorViewport>, mut images: ResMut<Assets<Image>>) {
    let want = vp.desired_size;
    if want.x < 16 || want.y < 16 || want == vp.size {
        return;
    }
    if let Some(mut image) = images.get_mut(&vp.image) {
        image.resize(Extent3d { width: want.x, height: want.y, depth_or_array_layers: 1 });
        vp.size = want;
    }
}
