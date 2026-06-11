//! GPU node-preview raymarch (see `docs/GPU_PREVIEW_RAYMARCH_PLAN.md`, Option B).
//!
//! The CPU bakes a graph's height + analytic normal into an `Rgba32Float` texture (the single
//! `Graph::eval` — no GPU noise / no SSOT drift) and a custom-material fullscreen quad raymarches it in
//! WGSL with an orbit camera passed as a uniform, so **rotating is pure-GPU** (rebake only on graph/zoom
//! change) and high-res stays cheap.
//!
//! **Pool + request decoupling** (Stage 4): a fixed pool of [`POOL_SIZE`] pre-allocated render targets
//! (each its own camera + offscreen image on its own render layer) is shared by all preview consumers.
//! egui panels can't touch the `World` mid-render, so they instead PUSH [`GpuPreviewRequest`]s into the
//! [`GpuPreviewRequests`] resource and READ the resulting [`GpuPreviewTextures`] map (1-frame lag);
//! [`process_gpu_previews`] assigns requests to slots (LRU-ish), re-bakes only on change, and toggles
//! each slot's camera active. Overflow past `POOL_SIZE` simply gets no GPU texture for the unassigned
//! previews this frame — they draw the "baking…" placeholder until a slot frees up (and the overflow is
//! `warn_once!`d). There is no CPU-preview fallback.

use std::collections::{HashMap, HashSet};

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

use crate::sdf_render::worldgen::biome::{
    self, BiomeId, BiomeLibrary, GPU_STRATA_MAX_LAYERS, GpuStrataColumn,
};
use crate::sdf_render::worldgen::graph::node::Graph;

/// Number of pre-allocated GPU preview targets (cap on simultaneous GPU-backed previews).
const POOL_SIZE: usize = 12;
/// First render layer the pool uses (each target gets its own to isolate its quad from the others +
/// the main scene; the editor's other rigs use 16/17).
const POOL_LAYER_BASE: usize = 20;
/// Offscreen output image edge length (px).
const PREVIEW_SIZE: u32 = 384;
/// Baked heightfield texture resolution (per side).
const HEIGHTFIELD_RES: usize = 256;
/// Seed used for the preview bake (matches the editor's CPU previews + the default world seed).
const PREVIEW_SEED: u64 = 7;
const SEA_LEVEL: f32 = 0.0;
const SNOW_LEVEL: f32 = 1000.0;
const WATER_DEPTH: f32 = 400.0;

/// Camera + framing uniform (scalars packed into the `w` lanes for trivial std140 alignment).
#[derive(ShaderType, Clone, Copy, Default)]
struct PreviewParams {
    eye: Vec4,    // xyz eye, w = image-plane tan
    fwd: Vec4,    // xyz forward, w = world half-extent Z
    right: Vec4,  // xyz right, w = height min
    up: Vec4,     // xyz up, w = height max
    levels: Vec4, // sea, snow, water-depth-ramp, water-level
    flags: Vec4,  // x = mode (0 = 3D orbit, 1 = 2D top-down), y = halfX, z = halfZ
    modes: Vec4,  // x = biome-map on, y = slice on, z = water on, w = unused
    slice: Vec4,  // x = axis (0=X,1=Z,2=Y), y = world-plane coord, (z,w) unused
}

/// GPU mirror of [`GpuStrataColumn`] (one biome's flattened strata column) laid out for std140: the 6
/// cumulative layer bottoms are packed into 2 `Vec4` lanes so the array stays 16-byte aligned. Built from
/// the CPU SSOT [`BiomeLibrary::gpu_strata_table`] — the same table the Stage-3 in-world shader will use.
#[derive(ShaderType, Clone, Copy, Default)]
struct GpuStrataColumnStd {
    surface_color: Vec4,
    layer_color: [Vec4; GPU_STRATA_MAX_LAYERS],
    layer_bottom: [Vec4; 2], // 6 floats packed: lane0.xyzw + lane1.xy
    bedrock_color: Vec4,
    layer_count: u32,
    _pad: [u32; 3],
}

impl From<&GpuStrataColumn> for GpuStrataColumnStd {
    fn from(c: &GpuStrataColumn) -> Self {
        let mut layer_color = [Vec4::ZERO; GPU_STRATA_MAX_LAYERS];
        for (i, col) in c.layer_color.iter().enumerate() {
            layer_color[i] = Vec4::from_array(*col);
        }
        // Pack 6 bottoms into 2 vec4 lanes (xyzw, then xy…).
        let mut bottom = [Vec4::ZERO; 2];
        for (i, &b) in c.layer_bottom.iter().enumerate() {
            bottom[i / 4][i % 4] = b;
        }
        Self {
            surface_color: Vec4::from_array(c.surface_color),
            layer_color,
            layer_bottom: bottom,
            bedrock_color: Vec4::from_array(c.bedrock_color),
            layer_count: c.layer_count,
            _pad: [0; 3],
        }
    }
}

/// The full per-biome strata table uniform (one column per [`BiomeId`], id order) — the GPU side of
/// [`BiomeLibrary::gpu_strata_table`]. A fixed-size array sized to the demo biome count.
#[derive(ShaderType, Clone, Copy)]
struct StrataTableStd {
    columns: [GpuStrataColumnStd; BIOME_COUNT],
}

/// Demo biome count (matches `BiomeId::ALL.len()` and the shader's `BIOME_COUNT`).
const BIOME_COUNT: usize = BiomeId::ALL.len();

impl Default for StrataTableStd {
    fn default() -> Self {
        Self { columns: [GpuStrataColumnStd::default(); BIOME_COUNT] }
    }
}

impl StrataTableStd {
    /// Flatten a [`BiomeLibrary`] into the GPU table (via the CPU SSOT), clamped/padded to [`BIOME_COUNT`].
    fn from_library(lib: &BiomeLibrary) -> Self {
        let table = lib.gpu_strata_table();
        let mut columns = [GpuStrataColumnStd::default(); BIOME_COUNT];
        for (i, c) in table.iter().take(BIOME_COUNT).enumerate() {
            columns[i] = GpuStrataColumnStd::from(c);
        }
        Self { columns }
    }
}

/// Custom fullscreen-quad material: the camera uniform + the CPU-baked height/normal + biome textures and
/// the flattened strata table.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
struct HeightPreviewMaterial {
    #[uniform(0)]
    params: PreviewParams,
    // Rgba32Float, fetched via `textureLoad` (unfilterable → manual bilinear in the shader).
    #[texture(1, sample_type = "float", filterable = false)]
    height: Handle<Image>,
    // Rgba32Float biome map: R = primary biome id, G = secondary id, B = blend (CPU-classified per texel).
    #[texture(2, sample_type = "float", filterable = false)]
    biome: Handle<Image>,
    #[uniform(3)]
    strata: StrataTableStd,
}

impl Material for HeightPreviewMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/worldgen_preview.wgsl".into()
    }
}

/// A request to GPU-render a graph preview, pushed by a panel during its egui render.
pub struct GpuPreviewRequest {
    /// Stable per-preview key (the caller's `preview_key`) — identifies the slot + the returned texture.
    pub key: u64,
    /// The compiled sub-graph to preview (owned snapshot).
    pub graph: Graph,
    /// World half-extent (m) of the sampled window (the VERTICAL/Z half; X widens by the display aspect).
    pub half: f64,
    /// World-XZ centre the window is panned to (offset X/Y).
    pub center: (f64, f64),
    /// `true` = 3D orbit raymarch; `false` = 2D top-down orthographic field map.
    pub is3d: bool,
    /// Orbit camera (3D only).
    pub yaw: f32,
    pub pitch: f32,
    /// Biome-map / slice / water preview toggles + their parameters.
    pub modes: PreviewModes,
    /// Desired output resolution (px) — the pool resizes the slot's image to this (quantised) so the render
    /// tracks the on-screen size AND fills a non-square window (the camera fov widens by `res_w/res_h`).
    pub res_w: u32,
    pub res_h: u32,
}

/// The biome-map / strata-slice / water preview toggles + their parameters, carried per preview through
/// the request and persisted per node/panel/pop-out. The single carrier for the three optional preview
/// overlays (TERRAIN_MATERIALS_PLAN "Biome preview" / "SLICE" / "WATER plane").
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PreviewModes {
    /// Colour the preview by the SURFACE biome's `preview_color` (climate→biome→surface) instead of the
    /// height ramp. 2D = a climate/biome map; 3D = the surface shaded by biome.
    pub biome_map: bool,
    /// Slice/cutaway: hide the near half past the clip plane + shade the exposed strata cross-section.
    pub slice_on: bool,
    /// Slice clip-plane axis: 0 = X, 1 = Z, 2 = Y.
    pub slice_axis: u8,
    /// Slice plane position along its axis, normalized `[0,1]` over the sampled window (mapped to a world
    /// coordinate at build time).
    pub slice_pos: f32,
    /// Semi-transparent water plane at the water level, compositing over terrain below it.
    pub water_on: bool,
    /// Water level (world metres) — defaults to the preview sea level ([`SEA_LEVEL`]); an optional slider.
    pub water_level: f32,
}

impl Default for PreviewModes {
    fn default() -> Self {
        // Sensible defaults: every overlay OFF (height preview), slice across X at the centre.
        Self {
            biome_map: false,
            slice_on: false,
            slice_axis: 0,
            slice_pos: 0.5,
            water_on: false,
            water_level: SEA_LEVEL,
        }
    }
}

/// Inbox: preview consumers push requests here each frame; [`process_gpu_previews`] drains it.
#[derive(Resource, Default)]
pub struct GpuPreviewRequests(pub Vec<GpuPreviewRequest>);

/// Outbox: `key → egui texture id` for the GPU-rendered previews (read by consumers; 1-frame lag).
#[derive(Resource, Default)]
pub struct GpuPreviewTextures(pub HashMap<u64, egui::TextureId>);

/// One pooled render target.
struct GpuTarget {
    camera: Entity,
    material: Handle<HeightPreviewMaterial>,
    height: Handle<Image>,
    /// The CPU-classified biome map (R = primary id, G = secondary id, B = blend), baked alongside height.
    biome: Handle<Image>,
    /// The offscreen output image (resized to track the requested on-screen size).
    output: Handle<Image>,
    out_w: u32,
    out_h: u32,
    tex_id: egui::TextureId,
    /// Preview key currently assigned to this slot (0 = free).
    key: u64,
    /// Graph+zoom fingerprint last baked into `height`/`biome`.
    baked_key: u64,
    ymin: f32,
    ymax: f32,
}

/// Quantise a desired output resolution to bound how often the slot image is recreated.
fn quantize_res(res: u32) -> u32 {
    (res.clamp(128, 2048) / 128).max(1) * 128
}

#[derive(Resource, Default)]
struct GpuPreviewPool {
    targets: Vec<GpuTarget>,
}

/// Offscreen render-attachment image (the raymarch output egui samples) at `w`×`h`.
fn make_output_image(images: &mut Assets<Image>, w: u32, h: u32) -> Handle<Image> {
    let size = Extent3d { width: w, height: h, depth_or_array_layers: 1 };
    // Pure render target sampled by egui — no CPU read-back, so it only needs to live in the render world.
    let mut image =
        Image::new_fill(size, TextureDimension::D2, &[0, 0, 0, 0], TextureFormat::Rgba8UnormSrgb, RenderAssetUsages::RENDER_WORLD);
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

/// The CPU-classified biome map (Rgba32Float: R = primary biome id, G = secondary id, B = blend), filled
/// by [`bake_biome`] over the same window as the heightfield. Same resolution as the heightfield so the
/// shader can index both by the same planar UV.
fn make_biome_image(images: &mut Assets<Image>) -> Handle<Image> {
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

/// Run the Stage-1 Whittaker classifier ([`biome::surface_biome`]) per texel over the same ±`half` window
/// as the heightfield, writing (primary id, secondary id, blend) into `image`. This keeps ALL biome
/// classification on the CPU SSOT — the shader never re-implements the climate/Whittaker logic. Uses the
/// preview seed so the map matches the world bake's biome placement.
fn bake_biome(image: &mut Image, half: f64, center: (f64, f64)) {
    let n = HEIGHTFIELD_RES;
    let mut data = vec![0f32; n * n * 4];
    for j in 0..n {
        for i in 0..n {
            let wx = center.0 - half + (i as f64 + 0.5) / n as f64 * 2.0 * half;
            let wz = center.1 - half + (j as f64 + 0.5) / n as f64 * 2.0 * half;
            let s = biome::surface_biome(wx, wz, PREVIEW_SEED);
            let k = (j * n + i) * 4;
            data[k] = s.primary as u8 as f32;
            data[k + 1] = s.secondary as u8 as f32;
            data[k + 2] = s.blend;
            data[k + 3] = 0.0;
        }
    }
    image.data = Some(bytemuck::cast_slice(&data).to_vec());
}

/// Evaluate the graph over the square ±`half` window into `image` (height + analytic world normal); returns
/// the height range (for camera framing). The single `Graph::eval` — no GPU noise. The window is square and
/// axis-aligned (world x/z); the shader orbits it on the GPU, so rotating costs no re-bake.
fn bake_height(image: &mut Image, g: &Graph, half: f64, center: (f64, f64)) -> (f32, f32) {
    let n = HEIGHTFIELD_RES;
    let mut data = vec![0f32; n * n * 4];
    let (mut ymin, mut ymax) = (f32::INFINITY, f32::NEG_INFINITY);
    for j in 0..n {
        for i in 0..n {
            let wx = center.0 - half + (i as f64 + 0.5) / n as f64 * 2.0 * half;
            let wz = center.1 - half + (j as f64 + 0.5) / n as f64 * 2.0 * half;
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

/// Fingerprint the bake inputs (graph + window) so the heightfield is only re-baked on change. Orbit
/// (yaw/pitch) is NOT included — rotation is a pure GPU-shader operation on the baked square.
fn bake_key(g: &Graph, half: f64, center: (f64, f64)) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ron::to_string(g).unwrap_or_default().hash(&mut h);
    half.to_bits().hash(&mut h);
    center.0.to_bits().hash(&mut h);
    center.1.to_bits().hash(&mut h);
    h.finish()
}

/// Orbit camera + framing → the shader uniform. The window is **square** (±`half`); the camera distance is
/// fit so the terrain AABB just fills the (square) frustum at this orbit angle — a tight margin, every
/// corner in frame, no fixed slack.
fn build_params(
    yaw: f32,
    pitch: f32,
    half: f32,
    ymin: f32,
    ymax: f32,
    is3d: bool,
    modes: &PreviewModes,
) -> PreviewParams {
    let span = (ymax - ymin).max(1.0);
    let pad = span * 0.02 + 1.0; // minimal vertical breathing room (was 0.08)
    let centre = Vec3::new(0.0, (ymin + ymax) * 0.5, 0.0);
    let (sp, cp) = pitch.sin_cos();
    let (sy, cyaw) = yaw.sin_cos();
    let fwd = -Vec3::new(cp * cyaw, sp, cp * sy); // points from eye toward centre
    let right = fwd.cross(Vec3::Y).normalize_or_zero();
    let up = right.cross(fwd);
    let tan = (0.6f32 * 0.5).tan() * 2.0; // half-fov tan (square → same horizontal + vertical)
    // Smallest forward distance d so every AABB corner projects within [-1,1] in both axes.
    let (bmin, bmax) = (Vec3::new(-half, ymin - pad, -half), Vec3::new(half, ymax + pad, half));
    let mut d = 1.0f32;
    for cx in [bmin.x, bmax.x] {
        for cy in [bmin.y, bmax.y] {
            for cz in [bmin.z, bmax.z] {
                let rel = Vec3::new(cx, cy, cz) - centre;
                let f = rel.dot(fwd);
                d = d.max(rel.dot(right).abs() / tan - f).max(rel.dot(up).abs() / tan - f);
            }
        }
    }
    let eye = centre - fwd * (d * 1.015); // tight margin so corners aren't pixel-clipped (was 1.06)

    // Map the normalized slice position to a world coordinate along its axis: X/Z over [-half, half], Y
    // over the baked height span [ymin, ymax].
    let p = modes.slice_pos.clamp(0.0, 1.0);
    let slice_coord = match modes.slice_axis {
        2 => ymin + p * (ymax - ymin),
        _ => -half + p * 2.0 * half,
    };

    PreviewParams {
        eye: eye.extend(tan),
        fwd: fwd.extend(half),
        right: right.extend(ymin),
        up: up.extend(ymax),
        levels: Vec4::new(SEA_LEVEL, SNOW_LEVEL, WATER_DEPTH, modes.water_level),
        flags: Vec4::new(if is3d { 0.0 } else { 1.0 }, half, half, 0.0),
        modes: Vec4::new(
            if modes.biome_map { 1.0 } else { 0.0 },
            if modes.slice_on { 1.0 } else { 0.0 },
            if modes.water_on { 1.0 } else { 0.0 },
            0.0,
        ),
        slice: Vec4::new(modes.slice_axis as f32, slice_coord, 0.0, 0.0),
    }
}

/// Pre-spawn the pool once: [`POOL_SIZE`] quads + cameras (initially inactive) on consecutive layers.
fn setup_gpu_pool(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<HeightPreviewMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut egui_textures: ResMut<EguiUserTextures>,
    existing: Option<Res<GpuPreviewPool>>,
) {
    if existing.is_some() {
        return;
    }
    let mut targets = Vec::with_capacity(POOL_SIZE);
    for slot in 0..POOL_SIZE {
        let layer = RenderLayers::layer(POOL_LAYER_BASE + slot);
        let height = make_height_image(&mut images);
        let biome = make_biome_image(&mut images);
        let output = make_output_image(&mut images, PREVIEW_SIZE, PREVIEW_SIZE);
        let material = materials.add(HeightPreviewMaterial {
            params: PreviewParams::default(),
            height: height.clone(),
            biome: biome.clone(),
            strata: StrataTableStd::default(),
        });
        let quad = meshes.add(Rectangle::new(2.0, 2.0));
        commands.spawn((Mesh3d(quad), MeshMaterial3d(material.clone()), Transform::IDENTITY, layer.clone()));
        let camera = commands
            .spawn((
                Camera3d::default(),
                Camera {
                    order: -30 - slot as isize,
                    is_active: false,
                    clear_color: ClearColorConfig::Custom(Color::BLACK),
                    ..default()
                },
                Projection::Orthographic(OrthographicProjection {
                    scaling_mode: ScalingMode::Fixed { width: 2.0, height: 2.0 },
                    ..OrthographicProjection::default_3d()
                }),
                RenderTarget::Image(output.clone().into()),
                Transform::from_xyz(0.0, 0.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
                layer,
            ))
            .id();
        let tex_id = egui_textures.add_image(EguiTextureHandle::Strong(output.clone()));
        targets.push(GpuTarget {
            camera,
            material,
            height,
            biome,
            output,
            out_w: PREVIEW_SIZE,
            out_h: PREVIEW_SIZE,
            tex_id,
            key: 0,
            baked_key: 0,
            ymin: 0.0,
            ymax: 1.0,
        });
    }
    commands.insert_resource(GpuPreviewPool { targets });
}

/// Drain this frame's requests: assign to slots, re-bake on change, drive the camera uniforms, publish
/// the textures, and deactivate unused slots.
fn process_gpu_previews(world: &mut World) {
    if !world.contains_resource::<GpuPreviewPool>() {
        return;
    }
    let requests = std::mem::take(&mut world.resource_mut::<GpuPreviewRequests>().0);
    // Flatten the live biome library into the GPU strata table once per frame (cheap — 5 columns) via the
    // CPU SSOT, so the slice cut-face + biome map use the same authored colours the Stage-3 shader will.
    let strata_table = world
        .get_resource::<BiomeLibrary>()
        .filter(|lib| lib.biomes.len() == BIOME_COUNT)
        .map(StrataTableStd::from_library)
        .unwrap_or_default();
    world.resource_scope::<GpuPreviewPool, ()>(|world, mut pool| {
        let req_keys: HashSet<u64> = requests.iter().map(|r| r.key).collect();
        // request index → slot index. Reuse a slot already holding the key, else take a free one.
        let mut assign: Vec<Option<usize>> = vec![None; requests.len()];
        for (ri, r) in requests.iter().enumerate() {
            if r.key != 0
                && let Some(si) = pool.targets.iter().position(|t| t.key == r.key)
            {
                assign[ri] = Some(si);
            }
        }
        let mut free: Vec<usize> =
            (0..pool.targets.len()).filter(|&si| !req_keys.contains(&pool.targets[si].key)).collect();
        for slot in assign.iter_mut() {
            if slot.is_none()
                && let Some(si) = free.pop()
            {
                *slot = Some(si);
            }
        }
        // Overflow: more distinct previews requested this frame than the pool can render. The unassigned
        // ones get no texture (they show the "baking…" placeholder). Warn once so it's not a silent stall.
        let overflow = assign.iter().filter(|s| s.is_none()).count();
        if overflow > 0 {
            bevy::log::warn_once!(
                "worldgen GPU preview pool exhausted: {overflow} of {} requested previews have no slot \
                 (POOL_SIZE = {POOL_SIZE}); they will show the baking placeholder until a slot frees",
                requests.len()
            );
        }

        let mut out: HashMap<u64, egui::TextureId> = HashMap::new();
        let active: HashSet<usize> = assign.iter().filter_map(|s| *s).collect();
        for (ri, slot) in assign.iter().enumerate() {
            let Some(si) = *slot else { continue };
            let r = &requests[ri];
            // Resize the slot's output image to the requested on-screen size (quantised, possibly
            // non-square to fill the window) — recreate the image + re-point the camera + re-register the
            // egui texture.
            let (want_w, want_h) = (quantize_res(r.res_w), quantize_res(r.res_h));
            if pool.targets[si].out_w != want_w || pool.targets[si].out_h != want_h {
                let new_out = make_output_image(&mut world.resource_mut::<Assets<Image>>(), want_w, want_h);
                let old = pool.targets[si].output.clone();
                {
                    let mut et = world.resource_mut::<EguiUserTextures>();
                    et.remove_image(old.id());
                    pool.targets[si].tex_id = et.add_image(EguiTextureHandle::Strong(new_out.clone()));
                }
                let cam_e = pool.targets[si].camera;
                world.entity_mut(cam_e).insert(RenderTarget::Image(new_out.clone().into()));
                pool.targets[si].output = new_out;
                pool.targets[si].out_w = want_w;
                pool.targets[si].out_h = want_h;
            }
            // Square world window (the previews are drawn square). The heightfield is baked once per
            // graph/zoom/pan; orbiting (yaw/pitch) is a pure GPU operation, so no re-bake on rotate.
            let bk = bake_key(&r.graph, r.half, r.center);
            if pool.targets[si].key != r.key || pool.targets[si].baked_key != bk {
                let h = pool.targets[si].height.clone();
                let bio = pool.targets[si].biome.clone();
                let mut images = world.resource_mut::<Assets<Image>>();
                if let Some(img) = images.get_mut(&h) {
                    let (ymin, ymax) = bake_height(img, &r.graph, r.half, r.center);
                    pool.targets[si].ymin = ymin;
                    pool.targets[si].ymax = ymax;
                }
                // Re-classify the biome map over the same window (CPU Whittaker SSOT, no GPU port).
                if let Some(img) = images.get_mut(&bio) {
                    bake_biome(img, r.half, r.center);
                }
                pool.targets[si].key = r.key;
                pool.targets[si].baked_key = bk;
            }
            let params = build_params(
                r.yaw,
                r.pitch,
                r.half as f32,
                pool.targets[si].ymin,
                pool.targets[si].ymax,
                r.is3d,
                &r.modes,
            );
            let mat_h = pool.targets[si].material.clone();
            if let Some(mat) = world.resource_mut::<Assets<HeightPreviewMaterial>>().get_mut(&mat_h) {
                mat.params = params;
                mat.strata = strata_table;
            }
            out.insert(r.key, pool.targets[si].tex_id);
        }
        // Toggle camera activity (inactive slots keep their last image but cost nothing).
        for (si, t) in pool.targets.iter_mut().enumerate() {
            let want = active.contains(&si);
            if !want {
                t.key = 0;
            }
            let cam_e = t.camera;
            if let Some(mut cam) = world.get_mut::<Camera>(cam_e)
                && cam.is_active != want
            {
                cam.is_active = want;
            }
        }
        world.resource_mut::<GpuPreviewTextures>().0 = out;
    });
}

/// Plugin: the custom material pipeline + the pool + the request processor. Consumers (the node-graph
/// panel's inline previews + pop-out windows) push/read the request/texture resources.
pub struct WorldgenGpuPreviewPlugin;

impl Plugin for WorldgenGpuPreviewPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<HeightPreviewMaterial>::default());
        app.init_resource::<GpuPreviewRequests>();
        app.init_resource::<GpuPreviewTextures>();
        app.add_systems(
            Update,
            (setup_gpu_pool, process_gpu_previews).chain().run_if(resource_exists::<EguiUserTextures>),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader's `HF_TEX_RES` const MUST match the Rust `HEIGHTFIELD_RES` (the bake fills a
    /// `HEIGHTFIELD_RES`² texture; the shader's manual bilinear fetch indexes by `HF_TEX_RES`). A mismatch
    /// silently corrupts every preview's sampling — catch it at build time instead.
    #[test]
    fn shader_hf_tex_res_matches_rust() {
        let src = include_str!("../../assets/shaders/worldgen_preview.wgsl");
        // Find `const HF_TEX_RES: f32 = <N>.0;` and parse the integer part.
        let line = src
            .lines()
            .find(|l| l.contains("HF_TEX_RES") && l.contains("const"))
            .expect("worldgen_preview.wgsl declares `const HF_TEX_RES`");
        let rhs = line.split('=').nth(1).expect("HF_TEX_RES has an `= value`").trim();
        let digits: String = rhs.chars().take_while(|c| c.is_ascii_digit()).collect();
        let shader_res: usize = digits.parse().expect("HF_TEX_RES value is numeric");
        assert_eq!(
            shader_res, HEIGHTFIELD_RES,
            "HF_TEX_RES in worldgen_preview.wgsl ({shader_res}) != HEIGHTFIELD_RES in \
             worldgen_gpu_preview.rs ({HEIGHTFIELD_RES})"
        );
    }

    /// Parse `const <name>: <ty> = <N>u?;` from the shader and return the integer literal.
    fn shader_uint_const(src: &str, name: &str) -> usize {
        let line = src
            .lines()
            .find(|l| l.contains(name) && l.contains("const"))
            .unwrap_or_else(|| panic!("worldgen_preview.wgsl declares `const {name}`"));
        let rhs = line.split('=').nth(1).expect("const has an `= value`").trim();
        let digits: String = rhs.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse().unwrap_or_else(|_| panic!("`{name}` value is numeric, got `{rhs}`"))
    }

    /// The shader's strata-table dimensions MUST match the Rust/biome SSOT — the `strata` uniform is laid
    /// out as `[StrataColumn; BIOME_COUNT]` with `[Vec4; STRATA_MAX_LAYERS]` colours; a mismatch silently
    /// corrupts the strata slice / biome map. Catch it at build time (like `HF_TEX_RES`).
    #[test]
    fn shader_strata_dims_match_rust() {
        let src = include_str!("../../assets/shaders/worldgen_preview.wgsl");
        assert_eq!(
            shader_uint_const(src, "STRATA_MAX_LAYERS"),
            GPU_STRATA_MAX_LAYERS,
            "STRATA_MAX_LAYERS in the shader != GPU_STRATA_MAX_LAYERS in biome.rs"
        );
        assert_eq!(
            shader_uint_const(src, "BIOME_COUNT"),
            BIOME_COUNT,
            "BIOME_COUNT in the shader != BiomeId::ALL.len()"
        );
    }

    /// The packed `layer_bottom: [Vec4; 2]` must hold all `GPU_STRATA_MAX_LAYERS` layer bottoms.
    #[test]
    fn packed_layer_bottom_fits_all_layers() {
        assert!(GPU_STRATA_MAX_LAYERS <= 8, "layer_bottom packs into 2 vec4 (8 floats)");
    }

    /// The GPU strata column mirrors the CPU `GpuStrataColumn`: a depth probe of `GpuStrataColumnStd`
    /// (the packed/unpacked layout the shader reads) reproduces the source column's colours.
    #[test]
    fn gpu_strata_column_std_mirrors_cpu() {
        let cpu = GpuStrataColumn {
            surface_color: [0.1, 0.2, 0.3, 1.0],
            layer_color: {
                let mut a = [[0.0; 4]; GPU_STRATA_MAX_LAYERS];
                a[0] = [0.4, 0.0, 0.0, 1.0];
                a[1] = [0.0, 0.5, 0.0, 1.0];
                a[2] = [0.0, 0.0, 0.6, 1.0];
                a
            },
            layer_bottom: {
                let mut a = [0.0; GPU_STRATA_MAX_LAYERS];
                a[0] = 1.0;
                a[1] = 5.0;
                a[2] = 1005.0;
                a
            },
            bedrock_color: [0.01, 0.01, 0.02, 1.0],
            layer_count: 3,
            _pad: [0; 3],
        };
        let std = GpuStrataColumnStd::from(&cpu);
        // Surface / bedrock colours.
        assert_eq!(std.surface_color, Vec4::from_array(cpu.surface_color));
        assert_eq!(std.bedrock_color, Vec4::from_array(cpu.bedrock_color));
        assert_eq!(std.layer_count, cpu.layer_count);
        // Packed bottoms unpack to the source order.
        for i in 0..3 {
            assert_eq!(std.layer_bottom[i / 4][i % 4], cpu.layer_bottom[i], "bottom {i}");
            assert_eq!(std.layer_color[i], Vec4::from_array(cpu.layer_color[i]), "colour {i}");
        }
    }
}
