//! Scene thumbnail backend: when a `.scene` is saved, capture a screenshot of the primary
//! window, crop it to the 3D viewport region, downscale, and cache it to disk. The assets
//! browser then shows that image for the scene file. Capture is requested by
//! `editor::scene_tabs` on save (via [`PendingSceneThumbnail`]); the provider only loads the
//! cached PNG, so the panel stays immediate-mode-cheap.

use std::path::{Path, PathBuf};

use bevy::asset::RenderAssetUsages;
use bevy::image::{Image, ImageSampler};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::window::PrimaryWindow;
use bevy_egui::{EguiTextureHandle, EguiUserTextures, egui};

use crate::editor::dock::EditorDockState;

use super::super::{Thumbnail, ThumbnailProvider};

/// Pixel size of a cached scene thumbnail (square).
const THUMB_SIZE: u32 = 96;

/// Request to capture a thumbnail for a just-saved scene (working-dir-relative path). Set by
/// `editor::scene_tabs` after a successful save; drained by [`trigger_scene_capture`].
#[derive(Resource, Default)]
pub struct PendingSceneThumbnail(pub Option<PathBuf>);

/// Single-flight guard so only one window screenshot is in flight at a time.
#[derive(Resource, Default)]
struct SceneCaptureInFlight(bool);

/// On the screenshot entity: the scene to write for, and the physical-pixel region of the
/// window to crop to (the 3D viewport).
#[derive(Component)]
struct SceneThumbCapture {
    scene_path: PathBuf,
    rect: URect,
}

/// Cached scene thumbnails, keyed by scene path. `mtime` is the scene file's modification
/// time the cached image was loaded for, so a re-save reloads (keeping the old image shown
/// until the new one is ready, to avoid a flicker to the fallback icon).
#[derive(Resource, Default)]
pub struct SceneThumbnailCache {
    entries: std::collections::HashMap<PathBuf, SceneSlot>,
}

struct SceneSlot {
    tex_id: egui::TextureId,
    _handle: Handle<Image>,
    mtime: u64,
}

/// Provider for `.scene` files: shows the saved screenshot thumbnail, or a clapperboard
/// icon until one has been captured.
pub struct SceneThumbnailProvider;

impl ThumbnailProvider for SceneThumbnailProvider {
    fn matches(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("scene"))
    }

    fn thumbnail(&self, world: &mut World, path: &Path) -> Thumbnail {
        let key = path.to_path_buf();
        let mtime = file_mtime(path);

        // Up-to-date cached texture → use it directly.
        if let Some(slot) = world.resource::<SceneThumbnailCache>().entries.get(&key)
            && slot.mtime == mtime
        {
            return Thumbnail::Texture(slot.tex_id);
        }

        // Stale or missing: try to (re)load the cached PNG for the current mtime.
        let loaded = load_thumb_image(
            &scene_thumb_cache_path(path, mtime),
            &mut world.resource_mut::<Assets<Image>>(),
        );
        match loaded {
            Some(handle) => {
                let tex_id = world
                    .resource_mut::<EguiUserTextures>()
                    .add_image(EguiTextureHandle::Strong(handle.clone()));
                world.resource_mut::<SceneThumbnailCache>().entries.insert(
                    key,
                    SceneSlot {
                        tex_id,
                        _handle: handle,
                        mtime,
                    },
                );
                Thumbnail::Texture(tex_id)
            }
            // Not captured yet: keep showing the previous image if we have one, else the icon.
            None => match world.resource::<SceneThumbnailCache>().entries.get(&key) {
                Some(slot) => Thumbnail::Texture(slot.tex_id),
                None => Thumbnail::Icon("\u{1F3AC}"),
            },
        }
    }
}

/// Scene file's modification time as whole seconds since the epoch (0 if unavailable).
fn file_mtime(path: &Path) -> u64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Disk-cache path for a scene's thumbnail: a PNG in a temp dir keyed by a hash of the scene
/// path + its mtime, so a re-saved scene gets a fresh cache entry. The path string is
/// separator-normalized so the writer (which sees `doc.path`, forward slashes) and the
/// browser (which sees `read_dir` output, backslashes on Windows) agree on the same key.
fn scene_thumb_cache_path(scene_path: &Path, mtime: u64) -> PathBuf {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    scene_path.to_string_lossy().replace('\\', "/").hash(&mut h);
    mtime.hash(&mut h);
    std::env::temp_dir()
        .join("adventure_scene_thumbs")
        .join(format!("{:016x}.png", h.finish()))
}

/// Decode a cached thumbnail PNG into a Bevy `Image`, or `None` if it isn't there yet.
fn load_thumb_image(thumb_path: &Path, images: &mut Assets<Image>) -> Option<Handle<Image>> {
    let decoded = image::open(thumb_path).ok()?.to_rgba8();
    let (w, h) = decoded.dimensions();
    let mut img = Image::new(
        Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        decoded.into_raw(),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::all(),
    );
    img.sampler = ImageSampler::linear();
    Some(images.add(img))
}

/// Drain a pending scene-thumbnail request: snapshot the current viewport region (in physical
/// pixels) and spawn a primary-window screenshot whose observer writes the cropped thumbnail.
fn trigger_scene_capture(
    mut req: ResMut<PendingSceneThumbnail>,
    mut in_flight: ResMut<SceneCaptureInFlight>,
    dock: Option<Res<EditorDockState>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut commands: Commands,
) {
    if in_flight.0 {
        return;
    }
    let Some(scene_path) = req.0.clone() else {
        return;
    };
    let Some(dock) = dock else { return };
    let Ok(window) = windows.single() else { return };

    let r = dock.viewport_rect;
    if !(r.width() > 0.0 && r.height() > 0.0 && r.width().is_finite() && r.height().is_finite()) {
        // No usable viewport rect yet — keep the request and retry next frame.
        return;
    }

    // egui points → physical pixels, clamped to the window.
    let scale = window.scale_factor();
    let pw = window.physical_width();
    let ph = window.physical_height();
    let to_px = |v: f32, max: u32| (v * scale).round().clamp(0.0, max as f32) as u32;
    let rect = URect {
        min: UVec2::new(to_px(r.min.x, pw), to_px(r.min.y, ph)),
        max: UVec2::new(to_px(r.max.x, pw), to_px(r.max.y, ph)),
    };

    req.0 = None;
    in_flight.0 = true;
    debug!(
        "scene thumbnail: capturing screenshot for {} (crop {}x{} of {}x{})",
        scene_path.display(),
        rect.width(),
        rect.height(),
        pw,
        ph,
    );
    commands
        .spawn((Screenshot::primary_window(), SceneThumbCapture { scene_path, rect }))
        .observe(on_scene_screenshot);
}

/// Observer: the window screenshot is ready. Crop it to the viewport region, downscale, and
/// write the scene's thumbnail PNG; then despawn the screenshot entity and free the lock.
fn on_scene_screenshot(
    event: On<ScreenshotCaptured>,
    caps: Query<&SceneThumbCapture>,
    mut in_flight: ResMut<SceneCaptureInFlight>,
    mut commands: Commands,
) {
    let entity = event.event_target();
    in_flight.0 = false;
    if let Ok(cap) = caps.get(entity) {
        match write_scene_thumb(&event.image, cap) {
            Ok(out) => debug!("scene thumbnail: wrote {}", out.display()),
            Err(e) => warn!("scene thumbnail capture failed: {e}"),
        }
    }
    commands.entity(entity).despawn();
}

/// Crop the captured window image to `cap.rect`, resize to a square thumbnail, and save it to
/// the scene's disk-cache path.
fn write_scene_thumb(image: &Image, cap: &SceneThumbCapture) -> Result<PathBuf, String> {
    let dynamic = image
        .clone()
        .try_into_dynamic()
        .map_err(|e| format!("unsupported screenshot format: {e}"))?;
    // Drop the alpha channel (HDR brightness lives there) before cropping/resizing.
    let rgb = image::DynamicImage::ImageRgb8(dynamic.to_rgb8());

    let (iw, ih) = (rgb.width(), rgb.height());
    let x = cap.rect.min.x.min(iw);
    let y = cap.rect.min.y.min(ih);
    let w = cap.rect.max.x.min(iw).saturating_sub(x);
    let h = cap.rect.max.y.min(ih).saturating_sub(y);
    if w == 0 || h == 0 {
        return Err(format!(
            "viewport crop empty (rect {:?} vs image {iw}x{ih})",
            cap.rect
        ));
    }

    let thumb = rgb
        .crop_imm(x, y, w, h)
        .resize_to_fill(THUMB_SIZE, THUMB_SIZE, image::imageops::FilterType::Triangle);

    let mtime = file_mtime(&cap.scene_path);
    let out = scene_thumb_cache_path(&cap.scene_path, mtime);
    if let Some(parent) = out.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    thumb.save(&out).map_err(|e| e.to_string())?;
    Ok(out)
}

/// Register the scene-thumbnail cache + capture systems.
pub(super) fn register(app: &mut App) {
    app.init_resource::<SceneThumbnailCache>()
        .init_resource::<PendingSceneThumbnail>()
        .init_resource::<SceneCaptureInFlight>()
        .add_systems(Update, trigger_scene_capture);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_matches_scene_files() {
        let p = SceneThumbnailProvider;
        assert!(p.matches(Path::new("assets/scenes/gallery.scene")));
        assert!(p.matches(Path::new("X.SCENE")));
        assert!(!p.matches(Path::new("a.material.ron")));
        assert!(!p.matches(Path::new("a.png")));
    }

    #[test]
    fn cache_path_varies_with_mtime() {
        let a = scene_thumb_cache_path(Path::new("scenes/x.scene"), 100);
        let b = scene_thumb_cache_path(Path::new("scenes/x.scene"), 200);
        let a2 = scene_thumb_cache_path(Path::new("scenes/x.scene"), 100);
        assert_ne!(a, b);
        assert_eq!(a, a2);
        assert_eq!(a.extension().and_then(|e| e.to_str()), Some("png"));
    }
}
