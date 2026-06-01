//! Material-sphere thumbnail backend: captures a lit PBR sphere with each `*.material.ron`
//! applied. A shared sphere + light live on a dedicated [`RenderLayers`]; each material
//! gets a one-shot capture camera (fixed render target, never swapped) with a [`Readback`]
//! on its image. The capture is frozen exactly when the GPU returns a non-blank readback
//! (the authoritative "render done" signal) — no frame-count guessing. One capture is in
//! flight at a time. Rendered thumbnails are cached to disk so previews survive runs.

use std::path::{Path, PathBuf};

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::RenderTarget;
use bevy::image::{Image, ImageSampler};
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy_egui::{EguiTextureHandle, EguiUserTextures, egui};

use crate::assets::MaterialAsset;

use super::super::{Thumbnail, ThumbnailProvider};

/// Pixel size of a rendered thumbnail texture.
const THUMB_SIZE: u32 = 96;
/// Render layer the material-sphere rig lives on, isolated from the main view.
const THUMB_LAYER: usize = 16;

/// Texture usage for a material-thumbnail image. Both freshly-made and disk-cached
/// images use this: a material edit reuses the *same* image as the capture camera's
/// render target, so it must always carry RENDER_ATTACHMENT (+ COPY_SRC for readback)
/// even when first loaded from a cached PNG.
const THUMB_TARGET_USAGE: TextureUsages = TextureUsages::TEXTURE_BINDING
    .union(TextureUsages::COPY_DST)
    .union(TextureUsages::COPY_SRC)
    .union(TextureUsages::RENDER_ATTACHMENT);

/// One material's thumbnail capture slot.
struct MaterialSlot {
    /// Loaded source asset (drives `StandardMaterial` construction + re-capture).
    source: Handle<MaterialAsset>,
    /// Offscreen render target the sphere is captured into.
    image: Handle<Image>,
    /// Standard material applied to the sphere for this thumbnail.
    material: Option<Handle<StandardMaterial>>,
    /// Texture handles the material depends on; capture waits for these so a textured
    /// sphere is never frozen blank.
    deps: Vec<Handle<Image>>,
    /// This slot's dedicated capture camera, despawned once the GPU readback confirms
    /// the image is rendered.
    camera: Option<Entity>,
    /// egui id once registered.
    tex_id: Option<egui::TextureId>,
    /// True once the GPU has confirmed the sphere is rendered into `image` (via
    /// [`ReadbackComplete`]).
    captured: bool,
}

/// Cache of material-sphere thumbnails, keyed by `.material.ron` path. Filled by
/// [`render_material_thumbnails`] (starts one capture at a time) and completed by the
/// [`on_readback_complete`] observer.
#[derive(Resource, Default)]
pub struct MaterialThumbnailCache {
    entries: std::collections::HashMap<PathBuf, MaterialSlot>,
    /// Path of the capture currently in flight (single-flight). `None` = idle.
    in_flight: Option<PathBuf>,
}

/// Marker for the shared sphere whose material is set per capture.
#[derive(Component)]
struct ThumbnailSphere;

/// The shared rig (one sphere + one light), spawned once. Cameras are per-slot.
#[derive(Resource)]
struct ThumbnailRig;

/// On a capture camera: the `.material.ron` path it is rendering, so the
/// [`ReadbackComplete`] observer can find the slot to finalize.
#[derive(Component)]
struct CapturingSlot(PathBuf);

/// Disk-cache path for a material's rendered thumbnail: a PNG in a temp dir keyed by a
/// hash of the material path. Lets previews survive across runs (and avoids re-rendering
/// every material every launch). The material file's mtime is folded into the hash so an
/// edited material gets a fresh cache entry.
fn thumb_cache_path(material_path: &Path) -> PathBuf {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    material_path.to_string_lossy().hash(&mut h);
    if let Ok(meta) = std::fs::metadata(material_path)
        && let Ok(mtime) = meta.modified()
        && let Ok(dur) = mtime.duration_since(std::time::UNIX_EPOCH)
    {
        dur.as_secs().hash(&mut h);
    }
    std::env::temp_dir()
        .join("adventure_mat_thumbs")
        .join(format!("{:016x}.png", h.finish()))
}

/// Load a material's disk-cached thumbnail into a Bevy `Image`, if present + valid.
/// Returns `None` when there is no cache hit (so the caller renders fresh).
fn load_cached_thumb(material_path: &Path, images: &mut Assets<Image>) -> Option<Handle<Image>> {
    let cache = thumb_cache_path(material_path);
    let decoded = image::open(&cache).ok()?.to_rgba8();
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
    img.texture_descriptor.usage = THUMB_TARGET_USAGE;
    img.sampler = ImageSampler::linear();
    Some(images.add(img))
}

/// Write a freshly-captured thumbnail to its disk cache. `padded` is the raw GPU
/// readback for a `THUMB_SIZE²` RGBA8 target — its rows are padded to wgpu's 256-byte
/// copy alignment, so strip that padding to a tight RGBA buffer before encoding PNG.
fn write_thumb_cache(material_path: &Path, padded: &[u8]) {
    let tight_row = (THUMB_SIZE * 4) as usize;
    // Row stride rounded up to the next multiple of 256.
    let padded_row = tight_row.div_ceil(256) * 256;
    let expected = padded_row * THUMB_SIZE as usize;
    if padded.len() < expected {
        warn!(
            "material thumb cache: readback too small ({} < {expected}); skipping",
            padded.len()
        );
        return;
    }
    let mut rgba = Vec::with_capacity(tight_row * THUMB_SIZE as usize);
    for row in 0..THUMB_SIZE as usize {
        let start = row * padded_row;
        rgba.extend_from_slice(&padded[start..start + tight_row]);
    }

    let out = thumb_cache_path(material_path);
    if let Some(parent) = out.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = image::save_buffer(
        &out,
        &rgba,
        THUMB_SIZE,
        THUMB_SIZE,
        image::ExtendedColorType::Rgba8,
    ) {
        warn!("material thumb cache write failed ({}): {e}", out.display());
    }
}

/// Provider for material resources.
pub struct MaterialThumbnailProvider;

impl ThumbnailProvider for MaterialThumbnailProvider {
    fn matches(&self, path: &Path) -> bool {
        crate::editor::fs_util::is_material_ron(path)
    }

    fn thumbnail(&self, world: &mut World, path: &Path) -> Thumbnail {
        let Some(asset_path) = crate::editor::fs_util::relative_to_assets(path) else {
            return Thumbnail::Icon("\u{1F535}");
        };
        let key = path.to_path_buf();

        if !world.resource::<MaterialThumbnailCache>().entries.contains_key(&key) {
            // Fast path: a disk-cached PNG from a previous render (this run or a prior
            // one). Decode it into an Image + register with egui — skip rendering.
            if let Some(image) = load_cached_thumb(&key, &mut world.resource_mut::<Assets<Image>>())
            {
                let tex_id = world
                    .resource_mut::<EguiUserTextures>()
                    .add_image(EguiTextureHandle::Strong(image.clone()));
                let source = world.resource::<AssetServer>().load::<MaterialAsset>(asset_path);
                world.resource_mut::<MaterialThumbnailCache>().entries.insert(
                    key.clone(),
                    MaterialSlot {
                        source,
                        image,
                        material: None,
                        deps: Vec::new(),
                        camera: None,
                        tex_id: Some(tex_id),
                        captured: true,
                    },
                );
            } else {
                // Create the target image + load the source asset; the capture system fills it.
                let image = make_target_image(&mut world.resource_mut::<Assets<Image>>());
                let source = world.resource::<AssetServer>().load::<MaterialAsset>(asset_path);
                world.resource_mut::<MaterialThumbnailCache>().entries.insert(
                    key.clone(),
                    MaterialSlot {
                        source,
                        image,
                        material: None,
                        deps: Vec::new(),
                        camera: None,
                        tex_id: None,
                        captured: false,
                    },
                );
            }
        }

        let slot = &world.resource::<MaterialThumbnailCache>().entries[&key];
        if let Some(id) = slot.tex_id
            && slot.captured
        {
            return Thumbnail::Texture(id);
        }
        Thumbnail::Pending
    }
}

/// Build a render-target image sized for a thumbnail.
fn make_target_image(images: &mut Assets<Image>) -> Handle<Image> {
    let size = Extent3d {
        width: THUMB_SIZE,
        height: THUMB_SIZE,
        depth_or_array_layers: 1,
    };
    let mut image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::all(),
    );
    image.texture_descriptor.usage = THUMB_TARGET_USAGE;
    image.sampler = ImageSampler::linear();
    images.add(image)
}

/// Spawn the shared sphere + light on [`THUMB_LAYER`] once. The sphere's material is
/// set per capture; cameras are per-slot (fixed targets — never swapped — to dodge
/// Bevy's render-target-swap bug, #18366).
fn setup_thumbnail_rig(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    existing: Option<Res<ThumbnailRig>>,
) {
    if existing.is_some() {
        return;
    }
    let layer = RenderLayers::layer(THUMB_LAYER);

    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(1.0).mesh().uv(32, 18))),
        MeshMaterial3d(std_materials.add(StandardMaterial::default())),
        Transform::from_xyz(0.0, 0.0, 0.0),
        layer.clone(),
        ThumbnailSphere,
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(2.0, 3.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
        layer,
    ));
    commands.insert_resource(ThumbnailRig);
}

/// Drive material thumbnail captures: one material at a time, gated on its
/// `StandardMaterial` + texture deps being loaded. Spawns a capture camera with a
/// [`Readback`] on its target image; the [`on_readback_complete`] observer finalizes
/// the slot when the GPU signals the render is done — no frame-count guessing.
#[allow(clippy::too_many_arguments)]
fn render_material_thumbnails(
    mut commands: Commands,
    rig: Option<Res<ThumbnailRig>>,
    mut cache: ResMut<MaterialThumbnailCache>,
    asset_server: Res<AssetServer>,
    material_assets: Res<Assets<MaterialAsset>>,
    pbr_textures: Res<Assets<crate::assets::PbrTextureAsset>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut egui_textures: ResMut<EguiUserTextures>,
    mut sphere: Query<&mut MeshMaterial3d<StandardMaterial>, With<ThumbnailSphere>>,
) {
    if rig.is_none() {
        return;
    }

    // 1. Build StandardMaterials (+ record their texture deps) once each slot's source
    //    asset has loaded, and register any completed capture with egui.
    for slot in cache.entries.values_mut() {
        if slot.material.is_none()
            && let Some(asset) = material_assets.get(&slot.source)
        {
            let (mat, deps) = standard_from_material(asset, &asset_server, &pbr_textures);
            slot.material = Some(std_materials.add(mat));
            slot.deps = deps;
        }
        if slot.captured && slot.tex_id.is_none() {
            slot.tex_id =
                Some(egui_textures.add_image(EguiTextureHandle::Strong(slot.image.clone())));
        }
    }

    // 2. One capture in flight at a time (single-flight). The observer clears
    //    `in_flight` when its readback completes.
    if cache.in_flight.is_some() {
        return;
    }

    // 3. Pick the next slot whose material AND all texture deps are loaded.
    let next = cache.entries.iter().find_map(|(k, s)| {
        let ready = s.material.is_some()
            && !s.captured
            && s.camera.is_none()
            && s.deps
                .iter()
                .all(|h| asset_server.is_loaded_with_dependencies(h));
        ready.then(|| (k.clone(), s.material.clone().unwrap(), s.image.clone()))
    });

    let Some((key, material, image)) = next else {
        return;
    };

    // Apply the material to the shared sphere.
    if let Ok(mut sphere_mat) = sphere.single_mut() {
        sphere_mat.0 = material;
    }

    // Spawn the capture camera with a Readback on its target image. When the GPU
    // finishes, `ReadbackComplete` fires on this entity → `on_readback_complete`.
    let cam = commands
        .spawn((
            Camera3d::default(),
            Camera {
                order: -10,
                clear_color: ClearColorConfig::Custom(Color::linear_rgba(0.0, 0.0, 0.0, 0.0)),
                ..default()
            },
            RenderTarget::Image(image.clone().into()),
            Transform::from_xyz(0.0, 0.0, 3.2).looking_at(Vec3::ZERO, Vec3::Y),
            RenderLayers::layer(THUMB_LAYER),
            Readback::texture(image),
            CapturingSlot(key.clone()),
        ))
        .observe(on_readback_complete)
        .id();

    if let Some(slot) = cache.entries.get_mut(&key) {
        slot.camera = Some(cam);
    }
    cache.in_flight = Some(key);
}

/// Observer: the GPU finished rendering this capture camera's target. Mark the slot
/// captured and despawn the camera (its `RenderTarget` image stays alive via the slot's
/// handle), then free the single-flight lock so the next material can capture.
fn on_readback_complete(
    event: On<ReadbackComplete>,
    mut commands: Commands,
    cameras: Query<&CapturingSlot>,
    mut cache: ResMut<MaterialThumbnailCache>,
) {
    let entity = event.event_target();
    let Ok(CapturingSlot(path)) = cameras.get(entity) else {
        return;
    };

    // `Readback` reads the target EVERY frame it's attached, and the camera's first
    // draw can lag a frame behind extraction — so the first readback may carry the
    // CLEARED (all-transparent) target, not the rendered sphere. The event derefs to
    // the actual pixel bytes, so gate on them: only accept a frame that actually drew
    // something (any non-zero pixel). Otherwise ignore this readback and wait for the
    // next — the `Readback` component stays attached and fires again next frame.
    let drew_something = event.data.iter().any(|&b| b != 0);
    if !drew_something {
        return;
    }

    let path = path.clone();
    // Persist the rendered pixels to the disk cache so this preview is reused next time
    // (this run and future runs) without re-rendering.
    write_thumb_cache(&path, &event.data);
    if let Some(slot) = cache.entries.get_mut(&path) {
        slot.captured = true;
        slot.camera = None;
    }
    if cache.in_flight.as_deref() == Some(path.as_path()) {
        cache.in_flight = None;
    }
    // Despawn the capture camera (removing its `Readback`); the slot's image handle
    // keeps the now-frozen texture.
    commands.entity(entity).despawn();
}

/// Resolve a material's effective PBR-texture bundle (its `.pbrtex.ron` merged with its
/// per-role overrides), loading the bundle via the asset server. Shared so the thumbnail
/// and the live preview build the same texture set. Returns the default (overrides-only)
/// set while the bundle is still loading.
pub(crate) fn effective_pbr_texture(
    asset: &MaterialAsset,
    server: &AssetServer,
    bundles: &Assets<crate::assets::PbrTextureAsset>,
) -> crate::assets::PbrTextureAsset {
    let bundle = asset.texture.as_ref().and_then(|p| {
        let h = server.load::<crate::assets::PbrTextureAsset>(p.clone());
        bundles.get(&h).cloned()
    });
    bundle.unwrap_or_default().merge(&asset.overrides)
}

/// Build a `StandardMaterial` approximating a `MaterialAsset` for the thumbnail (base
/// color tint, scalar metallic/roughness, diffuse + normal maps if any) and return the
/// texture handles it depends on, so capture can wait for them to load. Texture files
/// come from the material's effective PBR-texture bundle.
pub(crate) fn standard_from_material(
    asset: &MaterialAsset,
    server: &AssetServer,
    bundles: &Assets<crate::assets::PbrTextureAsset>,
) -> (StandardMaterial, Vec<Handle<Image>>) {
    let tex = effective_pbr_texture(asset, server, bundles);
    let mut deps = Vec::new();
    let mut load = |role: &Option<std::path::PathBuf>| -> Option<Handle<Image>> {
        role.as_ref().map(|p| {
            let h = server.load::<Image>(p.clone());
            deps.push(h.clone());
            h
        })
    };
    let base_color_texture = load(&tex.diffuse);
    let normal_map_texture = load(&tex.normal);
    let mat = StandardMaterial {
        base_color: asset.color(),
        base_color_texture,
        normal_map_texture,
        metallic: asset.metallic,
        perceptual_roughness: asset.roughness.max(0.05),
        ..default()
    };
    (mat, deps)
}

/// Invalidate a material thumbnail when its source asset is edited so it re-captures.
/// Despawns any stale capture camera and resets the slot so the picker re-runs it.
fn invalidate_on_material_change(
    mut events: MessageReader<AssetEvent<MaterialAsset>>,
    mut commands: Commands,
    mut cache: ResMut<MaterialThumbnailCache>,
) {
    let modified: Vec<_> = events
        .read()
        .filter_map(|ev| match ev {
            AssetEvent::Modified { id } => Some(*id),
            _ => None,
        })
        .collect();
    if modified.is_empty() {
        return;
    }

    let mut cleared_paths = Vec::new();
    for (path, slot) in cache.entries.iter_mut() {
        if modified.contains(&slot.source.id()) {
            slot.captured = false;
            slot.material = None; // rebuild from the new asset values
            slot.deps.clear();
            if let Some(cam) = slot.camera.take() {
                commands.entity(cam).despawn();
            }
            cleared_paths.push(path.clone());
        }
    }
    // If the in-flight capture was invalidated, free the single-flight lock.
    if cache
        .in_flight
        .as_ref()
        .is_some_and(|p| cleared_paths.contains(p))
    {
        cache.in_flight = None;
    }
}

/// Register the material-thumbnail cache + render systems. Systems are gated on
/// `EguiUserTextures` (present only with the editor's egui).
pub(super) fn register(app: &mut App) {
    app.init_resource::<MaterialThumbnailCache>().add_systems(
        Update,
        (
            setup_thumbnail_rig,
            render_material_thumbnails,
            invalidate_on_material_change,
        )
            .chain()
            .run_if(resource_exists::<EguiUserTextures>),
    );
}
