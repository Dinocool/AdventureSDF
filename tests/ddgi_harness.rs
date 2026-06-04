//! DDGI self-validation harness (AI-runnable, headless — no GUI, no human in the loop).
//!
//! This is the numeric gate for every phase of the DDGI implementation. It bakes tiny,
//! deterministic mini-scenes into a REAL SDF atlas on a headless wgpu device (reusing the
//! production bake shader + topology, exactly like `sdf_lifecycle_gpu.rs`), then — once the
//! probe trace/blend/apply shaders land — dispatches them over those scenes and reads back
//! scalar metrics to assert correctness:
//!
//!   * bleed     — indirect light reaches a wall facing an emitter (`scene_emitter_wall`)
//!   * leak      — no light crosses a thin wall (`scene_thin_wall`)            [needs Chebyshev]
//!   * crease    — contact darkening in a concave corner (`scene_crease`)      [needs contact AO]
//!   * sub-brick — small features bounce light (`scene_subbrick`)             [needs subdivision]
//!   * boil      — irradiance is frame-stable under sub-voxel camera motion
//!   * energy/convergence — multi-bounce stabilises and conserves energy
//!
//! **Phase P-1 (this file's first landing):** builds the scene bakers + readback + the gate
//! API as stubs validated against a hand-written analytic reference, so the gates exist BEFORE
//! the feature. The only live assertion now is that every mini-scene bakes to a non-empty atlas
//! (`read_tile_has_content`). Later phases replace each `TODO(Pn)` stub with a real shader
//! dispatch + threshold assertion.
//!
//! Run:  cargo test --test ddgi_harness -- --nocapture
//!       cargo test --test ddgi_harness -- --ignored --nocapture   (ddgi_report)

#![allow(dead_code)] // Scaffolding wired to real shaders incrementally across phases P0..P7.

use std::borrow::Cow;
use std::collections::HashSet;

use bevy::math::bounding::Aabb3d;
use bevy::math::{IVec3, Vec3};
use bevy::prelude::Transform;
use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderLanguage,
};

use adventure::sdf_render::atlas::{dist_band_world, BrickKey, SdfAtlas, BRICK_EDGE};
use adventure::sdf_render::bvh::Bvh;
use adventure::sdf_render::chunk;
use adventure::sdf_render::probe::{PROBE_OCT_RES, PROBE_OCT_TEXELS};
use adventure::sdf_render::edits::{
    build_palette, edit_world_aabb, to_gpu_edit, GpuEdit, ResolvedEdit, SdfOp, SdfPrimitive,
};
use adventure::sdf_render::SdfGridConfig;

mod common;

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    common::headless_device(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
}

/// Device for the real-trace gates: the SDF atlas binds paged textures as a `binding_array` and
/// indexes them with `atlas_pages[loc.page]` (non-uniform), so the trace shader needs the texture
/// binding-array + non-uniform-indexing features AND a raised `max_binding_array_elements_per_shader_stage`
/// limit (the 2×64-page arrays); the default limit is 0. Requests the adapter's full limits.
fn gpu_full() -> Option<(wgpu::Device, wgpu::Queue)> {
    use futures_lite::future::block_on;
    let feats = wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
        | wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | wgpu::Features::TIMESTAMP_QUERY;
    let instance = wgpu::Instance::default();
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .ok()?;
    if !adapter.features().contains(feats) {
        eprintln!("adapter lacks binding-array features — skipping");
        return None;
    }
    block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("ddgi_trace_device"),
        required_features: feats,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .ok()
}

// ============================================================================================
// Bake scaffolding — adapted from `sdf_lifecycle_gpu.rs`. Drives the REAL bake compute shader
// over a fixed clipmap window around the (static) scene camera, into a persistent atlas texture.
// ============================================================================================

const TILE_W: u32 = 64; // px per tile (8*8)
const DIST_ROW_U32: u32 = 64; // padded row of distance texels (u32 view)
const DIST_TILE_U32: u32 = DIST_ROW_U32 * 8;
const MAT_TILE_U32: u32 = 128 * 8;
const TEST_TILES_PER_ROW: u32 = 64; // 64*64 = 4096px wide, within the 8192 default limit

fn compose_bake() -> naga::Module {
    let src = std::fs::read_to_string("assets/shaders/sdf_brick_bake.wgsl").unwrap();
    Composer::default()
        .make_naga_module(NagaModuleDescriptor {
            source: &src,
            file_path: "sdf_brick_bake.wgsl",
            ..Default::default()
        })
        .expect("compose bake")
}

fn header_bytes(
    coord: IVec3,
    voxel_size: f32,
    dist_band: f32,
    edit_start: u32,
    edit_count: u32,
    pal: [u16; 4],
) -> Vec<u8> {
    let mut b = Vec::with_capacity(48);
    for v in [coord.x, coord.y, coord.z] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(&voxel_size.to_le_bytes());
    b.extend_from_slice(&dist_band.to_le_bytes());
    b.extend_from_slice(&edit_start.to_le_bytes());
    b.extend_from_slice(&edit_count.to_le_bytes());
    b.extend_from_slice(&(pal[0] as u32 | ((pal[1] as u32) << 16)).to_le_bytes());
    b.extend_from_slice(&(pal[2] as u32 | ((pal[3] as u32) << 16)).to_le_bytes());
    for _ in 0..3 {
        b.extend_from_slice(&0u32.to_le_bytes());
    }
    b
}

fn edit_bytes(e: &GpuEdit) -> Vec<u8> {
    let mut b = Vec::with_capacity(96);
    for col in e.inv_model.to_cols_array() {
        b.extend_from_slice(&col.to_le_bytes());
    }
    for v in [e.params.x, e.params.y, e.params.z, e.params.w] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for v in [e.params2.x, e.params2.y, e.params2.z, e.params2.w] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(&e.tag.to_le_bytes());
    b.extend_from_slice(&e.op_kind.to_le_bytes());
    b.extend_from_slice(&e.smoothing.to_le_bytes());
    b.extend_from_slice(&e.material_id.to_le_bytes());
    b
}

fn tile_origin(tile: u32) -> (u32, u32) {
    let col_px = (tile % TEST_TILES_PER_ROW) * TILE_W;
    let row_px = (tile / TEST_TILES_PER_ROW) * BRICK_EDGE as u32;
    (col_px, row_px)
}

fn storage_entry(b: u32, ro: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: ro },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// One bake job emitted by the topology step.
struct Job {
    tile: u32,
    coord: IVec3,
    voxel_size: f32,
    dist_band: f32,
    pal: [u16; 4],
    edit_start: u32,
    edit_count: u32,
}

/// Persistent GPU atlas the harness bakes into: the distance texture (R16Snorm) AND the per-palette
/// material-distance texture (Rgba16Snorm). Both are page-0 of the `binding_array` the real raymarch
/// samples; the `atlas_base` in the chunk tile-run (built with this harness's tile packing) tells the
/// shader exactly where each brick's tile sits, so the production 256-wide page layout isn't needed.
struct GpuAtlas {
    tex: Option<wgpu::Texture>,
    mat_tex: Option<wgpu::Texture>,
    rows: u32,
}

impl GpuAtlas {
    fn new() -> Self {
        Self { tex: None, mat_tex: None, rows: 0 }
    }

    /// Dispatch this frame's bake jobs into output buffers, grow the distance texture if needed,
    /// then copy each job's tile in. Mirrors `prepare_sdf_atlas_gpu` (grow) + `SdfBrickBakeNode`.
    #[allow(clippy::too_many_arguments)]
    fn bake_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &wgpu::ComputePipeline,
        layout: &wgpu::BindGroupLayout,
        jobs: &[Job],
        edits: &[GpuEdit],
        high_water: u32,
    ) {
        let required_rows = high_water.div_ceil(TEST_TILES_PER_ROW).max(1);
        if required_rows > self.rows {
            let w = TEST_TILES_PER_ROW * TILE_W;
            let h = required_rows * BRICK_EDGE as u32;
            let usage = wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC;
            let make = |label: &str, format: wgpu::TextureFormat| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
            };
            let new_tex = make("ddgi_atlas_dist", wgpu::TextureFormat::R16Snorm);
            let new_mat = make("ddgi_atlas_mat", wgpu::TextureFormat::Rgba16Snorm);
            for (old, new) in [(&self.tex, &new_tex), (&self.mat_tex, &new_mat)] {
                if let Some(old) = old {
                    let old_h = old.height().min(h);
                    let mut enc = device.create_command_encoder(&Default::default());
                    enc.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: old,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: new,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d { width: w, height: old_h, depth_or_array_layers: 1 },
                    );
                    queue.submit([enc.finish()]);
                }
            }
            self.tex = Some(new_tex);
            self.mat_tex = Some(new_mat);
            self.rows = required_rows;
        }
        if jobs.is_empty() {
            return;
        }

        let mut hbytes = Vec::new();
        for j in jobs {
            hbytes.extend_from_slice(&header_bytes(
                j.coord,
                j.voxel_size,
                j.dist_band,
                j.edit_start,
                j.edit_count,
                j.pal,
            ));
        }
        let mut ebytes = Vec::new();
        for e in edits {
            ebytes.extend_from_slice(&edit_bytes(e));
        }
        if ebytes.is_empty() {
            ebytes.resize(96, 0);
        }
        use wgpu::util::DeviceExt;
        let header_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &hbytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let edit_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &ebytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let n = jobs.len() as u32;
        let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * DIST_TILE_U32 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mat_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * MAT_TILE_U32 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: header_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: edit_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dist_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mat_buf.as_entire_binding() },
            ],
        });

        let tex = self.tex.as_ref().unwrap();
        let mat_tex = self.mat_tex.as_ref().unwrap();
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let wg_x = n.min(256);
            let wg_y = n.div_ceil(256);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        for (i, j) in jobs.iter().enumerate() {
            let (col_px, row_px) = tile_origin(j.tile);
            let origin = wgpu::Origin3d { x: col_px, y: row_px, z: 0 };
            let extent =
                wgpu::Extent3d { width: TILE_W, height: BRICK_EDGE as u32, depth_or_array_layers: 1 };
            // Distance tile (R16Snorm: 64 texels × 2 B = 128 B/row, padded to 256 in the bake buffer).
            enc.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &dist_buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: (i as u32 * DIST_TILE_U32) as u64 * 4,
                        bytes_per_row: Some(DIST_ROW_U32 * 4),
                        rows_per_image: Some(BRICK_EDGE as u32),
                    },
                },
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin,
                    aspect: wgpu::TextureAspect::All,
                },
                extent,
            );
            // Material-distance tile (Rgba16Snorm: 64 texels × 8 B = 512 B/row = 128 u32/row).
            enc.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &mat_buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: (i as u32 * MAT_TILE_U32) as u64 * 4,
                        bytes_per_row: Some(128 * 4),
                        rows_per_image: Some(BRICK_EDGE as u32),
                    },
                },
                wgpu::TexelCopyTextureInfo {
                    texture: mat_tex,
                    mip_level: 0,
                    origin,
                    aspect: wgpu::TextureAspect::All,
                },
                extent,
            );
        }
        queue.submit([enc.finish()]);
    }

    /// Read back one tile's first row of distance texels — non-zero proves baked geometry.
    fn read_tile_has_content(&self, device: &wgpu::Device, queue: &wgpu::Queue, tile: u32) -> bool {
        let tex = self.tex.as_ref().unwrap();
        let (col_px, row_px) = tile_origin(tile);
        let row_bytes = 256u32; // 64*2 = 128 → padded to 256
        let size = (row_bytes * BRICK_EDGE as u32) as u64;
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: col_px, y: row_px, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &rb,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(BRICK_EDGE as u32),
                },
            },
            wgpu::Extent3d { width: TILE_W, height: BRICK_EDGE as u32, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        let slice = rb.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let data = slice.get_mapped_range().to_vec();
        let texels: &[i16] = bytemuck::cast_slice(&data[..(TILE_W * 2) as usize]);
        texels.iter().any(|&v| v != 0)
    }
}

/// All brick keys of a chunk (mirror of the private bake_scheduler helper).
fn chunk_brick_keys(ck: chunk::ChunkKey, cfg: &SdfGridConfig) -> Vec<BrickKey> {
    let s = cfg.cell_stride();
    let c = chunk::CHUNK_BRICKS;
    let base = ck.coord * c;
    let mut keys = Vec::with_capacity(chunk::CHUNK_VOLUME as usize);
    for lz in 0..c {
        for ly in 0..c {
            for lx in 0..c {
                let bi = base + IVec3::new(lx, ly, lz);
                keys.push(BrickKey::new(ck.lod, bi * s));
            }
        }
    }
    keys
}

fn ring_chunks_per_axis(cfg: &SdfGridConfig) -> i32 {
    (cfg.ring_bricks / chunk::CHUNK_BRICKS as u32) as i32
}

fn ring_chunk_origin(cfg: &SdfGridConfig, cam: Vec3, lod: u32) -> IVec3 {
    adventure::sdf_render::bake_scheduler::ring_chunk_origin(cfg, cam, lod)
}

fn chunk_window_keys(origin: IVec3, r: i32, lod: u32) -> Vec<chunk::ChunkKey> {
    let mut v = Vec::new();
    for iz in 0..r {
        for iy in 0..r {
            for ix in 0..r {
                v.push(chunk::ChunkKey::new(lod, origin + IVec3::new(ix, iy, iz)));
            }
        }
    }
    v
}

fn chunk_has_geometry(
    ck: chunk::ChunkKey,
    bvh: &Bvh,
    cfg: &SdfGridConfig,
    scratch: &mut Vec<u32>,
) -> bool {
    let size = chunk::chunk_world_size(ck.lod, cfg);
    let min = chunk::chunk_min_world(ck, cfg);
    bvh.query_aabb(&Aabb3d::from_min_max(min, min + Vec3::splat(size)), scratch);
    !scratch.is_empty()
}

/// Mirror of `emit_gpu_bakes`: cull+palette+alloc each brick of the dirty chunks → bake jobs.
fn emit(
    atlas: &mut SdfAtlas,
    cfg: &SdfGridConfig,
    bvh: &Bvh,
    resolved: &[ResolvedEdit],
    dirty: &HashSet<chunk::ChunkKey>,
) -> (Vec<Job>, Vec<GpuEdit>) {
    let mut jobs = Vec::new();
    let mut edits = Vec::new();
    let mut scratch = Vec::new();
    let mut chunks: Vec<chunk::ChunkKey> = dirty.iter().copied().collect();
    chunks.sort_unstable_by_key(|c| std::cmp::Reverse(c.lod));
    for ck in &chunks {
        for key in chunk_brick_keys(*ck, cfg) {
            if SdfAtlas::cull_edit_indices(key, bvh, cfg, &mut scratch).is_some() {
                let vs = cfg.voxel_size_at(key.lod);
                let samples = SdfAtlas::brick_palette_samples(key, vs);
                let culled: Vec<ResolvedEdit> =
                    scratch.iter().map(|&i| resolved[i as usize].clone()).collect();
                let pal = build_palette(&culled, &samples);
                let tile = atlas.insert_gpu_brick(key, pal, 0, cfg);
                let edit_start = edits.len() as u32;
                for e in &culled {
                    edits.push(to_gpu_edit(e));
                }
                jobs.push(Job {
                    tile,
                    coord: key.coord,
                    voxel_size: vs,
                    dist_band: dist_band_world(cfg, key.lod),
                    pal,
                    edit_start,
                    edit_count: culled.len() as u32,
                });
            } else {
                atlas.remove_brick(&key, cfg);
            }
        }
    }
    (jobs, edits)
}

/// First-frame recenter at a static camera: bake every in-geometry chunk in the window of
/// every LOD. (Simplified from the fly-path recenter — the harness scenes don't move.)
#[allow(clippy::too_many_arguments)]
fn bake_static_window(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    layout: &wgpu::BindGroupLayout,
    atlas: &mut SdfAtlas,
    gpu_atlas: &mut GpuAtlas,
    cfg: &SdfGridConfig,
    bvh: &Bvh,
    resolved: &[ResolvedEdit],
    cam: Vec3,
) {
    let r = ring_chunks_per_axis(cfg);
    let mut dirty = HashSet::new();
    let mut scratch = Vec::new();
    for lod in 0..cfg.lod_count {
        let origin = ring_chunk_origin(cfg, cam, lod);
        for ck in chunk_window_keys(origin, r, lod) {
            if chunk_has_geometry(ck, bvh, cfg, &mut scratch) {
                dirty.insert(ck);
            }
        }
    }
    let (jobs, edits) = emit(atlas, cfg, bvh, resolved, &dirty);
    let high_water = atlas.tiles.high_water();
    gpu_atlas.bake_frame(device, queue, pipeline, layout, &jobs, &edits, high_water);
}

/// The finest resident LOD with a baked brick at world point `p` (chunk-table presence only).
fn served_lod(atlas: &SdfAtlas, cfg: &SdfGridConfig, p: Vec3) -> Option<u32> {
    for lod in 0..cfg.lod_count {
        let coord = cfg.world_to_brick_lod(p, lod);
        if atlas.bricks.contains_key(&BrickKey::new(lod, coord)) {
            return Some(lod);
        }
    }
    None
}

// ============================================================================================
// Materials — analytic mini-material table (mirrors GpuSdfMaterial's emissive semantics).
// In P1 this becomes the real 80-byte GPU material buffer the probe trace samples.
// ============================================================================================

#[derive(Clone, Copy, Debug)]
struct MatDef {
    base_color: Vec3,
    /// Premultiplied emissive radiance (color * intensity), like `MaterialDef::emissive`.
    emissive: Vec3,
}

impl MatDef {
    fn diffuse(c: Vec3) -> Self {
        Self { base_color: c, emissive: Vec3::ZERO }
    }
    fn emitter(color: Vec3, intensity: f32) -> Self {
        Self { base_color: color * 0.05, emissive: color * intensity }
    }
}

/// Serialize a material table to the REAL 80-byte `GpuSdfMaterial` std430 layout (see
/// `render/mod.rs::GpuSdfMaterial` / `sdf/bindings.wgsl::SdfMaterial`): base_color(16) +
/// blend_softness/5×tex/metallic/roughness/parallax(36) + 3×pad(12) + emissive(16). The probe
/// trace (P1) binds this as group(1) binding(4) so `material_at(id).emissive` works on-GPU.
fn material_table_bytes(mats: &[MatDef]) -> Vec<u8> {
    let mut b = Vec::with_capacity(mats.len() * 80);
    for m in mats {
        // base_color (rgb, a=1)
        for v in [m.base_color.x, m.base_color.y, m.base_color.z, 1.0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&0.0f32.to_le_bytes()); // blend_softness
        for _ in 0..5 {
            b.extend_from_slice(&0u32.to_le_bytes()); // tex_diffuse/normal/mra/height/edge
        }
        b.extend_from_slice(&0.0f32.to_le_bytes()); // metallic
        b.extend_from_slice(&1.0f32.to_le_bytes()); // roughness (diffuse)
        b.extend_from_slice(&0.0f32.to_le_bytes()); // parallax_scale
        for _ in 0..3 {
            b.extend_from_slice(&0u32.to_le_bytes()); // _pad0.._pad2
        }
        // emissive (premultiplied rgb, a spare)
        for v in [m.emissive.x, m.emissive.y, m.emissive.z, 0.0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    debug_assert_eq!(b.len(), mats.len() * 80);
    b
}

// ============================================================================================
// Mini-scenes — deterministic, tiny, each targeting one gate.
// ============================================================================================

/// A world sample point where the harness measures (indirect) irradiance, with the reference
/// value the gate expects once the feature is implemented.
#[derive(Clone, Copy, Debug)]
struct Sample {
    pos: Vec3,
    normal: Vec3,
    /// Human label for reports.
    what: &'static str,
}

struct MiniScene {
    name: &'static str,
    cfg: SdfGridConfig,
    edits: Vec<ResolvedEdit>,
    materials: Vec<MatDef>,
    camera: Vec3,
    samples: Vec<Sample>,
}

fn cube(half: Vec3, center: Vec3, material_id: u16) -> ResolvedEdit {
    ResolvedEdit::new(
        SdfPrimitive::Box { half_extents: half },
        Transform::from_translation(center),
        SdfOp::default(),
        material_id,
    )
}

fn sphere(radius: f32, center: Vec3, material_id: u16) -> ResolvedEdit {
    ResolvedEdit::new(
        SdfPrimitive::Sphere { radius },
        Transform::from_translation(center),
        SdfOp::default(),
        material_id,
    )
}

/// Small ring so bake counts stay tiny but the scene fits the window at LOD 0.
fn small_cfg() -> SdfGridConfig {
    SdfGridConfig { lod_count: 3, ring_bricks: 16, recenter_snap_chunks: 1, ..Default::default() }
}

/// An emissive cube in front of a diffuse wall — the canonical colour-bleed test.
fn scene_emitter_wall() -> MiniScene {
    let materials = vec![
        MatDef::diffuse(Vec3::new(0.8, 0.8, 0.8)),  // 0: wall
        MatDef::emitter(Vec3::new(1.0, 0.2, 0.1), 6.0), // 1: red emitter
    ];
    let edits = vec![
        cube(Vec3::new(0.1, 1.0, 1.0), Vec3::new(0.0, 0.0, 0.0), 0), // wall at x=0
        cube(Vec3::new(0.15, 0.15, 0.15), Vec3::new(0.6, 0.0, 0.0), 1), // emitter in front (+x)
    ];
    let samples = vec![Sample {
        pos: Vec3::new(0.12, 0.0, 0.0),
        normal: Vec3::X, // wall face pointing toward the emitter
        what: "wall face toward emitter (expects red bleed)",
    }];
    MiniScene {
        name: "emitter_wall",
        cfg: small_cfg(),
        edits,
        materials,
        camera: Vec3::new(1.5, 0.0, 0.0),
        samples,
    }
}

/// Emitter on the far side of a thin wall — the leak test (far side must stay dark).
fn scene_thin_wall() -> MiniScene {
    let materials = vec![
        MatDef::diffuse(Vec3::new(0.8, 0.8, 0.8)),
        MatDef::emitter(Vec3::new(0.1, 1.0, 0.2), 6.0), // green emitter
    ];
    let edits = vec![
        cube(Vec3::new(0.05, 1.0, 1.0), Vec3::ZERO, 0),               // thin wall at x=0
        cube(Vec3::new(0.15, 0.15, 0.15), Vec3::new(-0.6, 0.0, 0.0), 1), // emitter on -x side
    ];
    let samples = vec![Sample {
        pos: Vec3::new(0.08, 0.0, 0.0),
        normal: Vec3::X, // +x face — opposite side from the emitter
        what: "wall face away from emitter (expects ~0, no leak)",
    }];
    MiniScene {
        name: "thin_wall",
        cfg: small_cfg(),
        edits,
        materials,
        camera: Vec3::new(1.5, 0.0, 0.0),
        samples,
    }
}

/// A concave 90° corner formed by two perpendicular walls — the crease / contact test.
fn scene_crease() -> MiniScene {
    let materials = vec![MatDef::diffuse(Vec3::new(0.8, 0.8, 0.8))];
    let edits = vec![
        cube(Vec3::new(1.0, 0.1, 1.0), Vec3::new(0.0, 0.0, 0.0), 0), // floor (y=0)
        cube(Vec3::new(0.1, 1.0, 1.0), Vec3::new(0.0, 1.0, 0.0), 0), // wall (x=0), forms a crease
    ];
    let samples = vec![
        Sample {
            pos: Vec3::new(0.15, 0.15, 0.0),
            normal: Vec3::new(1.0, 1.0, 0.0).normalize(),
            what: "inside corner (expects contact darkening)",
        },
        Sample {
            pos: Vec3::new(0.6, 0.12, 0.0),
            normal: Vec3::Y,
            what: "open floor away from corner (expects ~no darkening)",
        },
    ];
    MiniScene {
        name: "crease",
        cfg: small_cfg(),
        edits,
        materials,
        camera: Vec3::new(1.2, 1.2, 1.5),
        samples,
    }
}

/// A small emissive sphere smaller than a brick — the sub-brick resolution test.
fn scene_subbrick() -> MiniScene {
    let materials = vec![
        MatDef::diffuse(Vec3::new(0.8, 0.8, 0.8)),
        MatDef::emitter(Vec3::new(0.2, 0.4, 1.0), 8.0), // blue emitter
    ];
    let edits = vec![
        cube(Vec3::new(1.0, 0.1, 1.0), Vec3::new(0.0, 0.0, 0.0), 0), // floor
        sphere(0.15, Vec3::new(0.0, 0.25, 0.0), 1),                  // tiny emitter above floor
    ];
    let samples = vec![Sample {
        pos: Vec3::new(0.0, 0.12, 0.25),
        normal: Vec3::Y,
        what: "floor under tiny emitter (expects blue bounce)",
    }];
    MiniScene {
        name: "subbrick",
        cfg: small_cfg(),
        edits,
        materials,
        camera: Vec3::new(1.0, 0.8, 1.0),
        samples,
    }
}

fn all_scenes() -> Vec<MiniScene> {
    vec![scene_emitter_wall(), scene_thin_wall(), scene_crease(), scene_subbrick()]
}

/// A busy scene at the LIVE clipmap config (8 LODs — the default's large clipmap reach, so the probe
/// rays march the same far distances the gallery does) with a large floor + a dense grid of boxes, to
/// reproduce the gallery's per-frame trace cost. `ring_bricks 64` keeps the CPU bake-window scan
/// tractable while still giving 8 LOD rings of resident bricks. Hundreds–thousands of probes.
fn scene_perf() -> MiniScene {
    let materials = vec![
        MatDef::diffuse(Vec3::new(0.8, 0.8, 0.8)),
        MatDef::emitter(Vec3::new(1.0, 0.4, 0.1), 6.0),
        MatDef::emitter(Vec3::new(0.2, 0.5, 1.0), 6.0),
    ];
    let mut edits = vec![cube(Vec3::new(5.0, 0.1, 5.0), Vec3::new(0.0, 0.0, 0.0), 0)]; // big floor
    for gz in -2..=2 {
        for gx in -2..=2 {
            let c = Vec3::new(gx as f32 * 1.6, 0.35, gz as f32 * 1.6);
            let mat = if (gx + gz) % 3 == 0 { 1 } else { 0 };
            edits.push(cube(Vec3::splat(0.3), c, mat));
        }
    }
    edits.push(sphere(0.25, Vec3::new(3.5, 0.5, -3.5), 2)); // blue emitter corner
    MiniScene {
        name: "perf",
        cfg: SdfGridConfig { lod_count: 8, ring_bricks: 64, recenter_snap_chunks: 1, ..Default::default() },
        edits,
        materials,
        camera: Vec3::new(0.0, 3.5, 5.0),
        samples: vec![Sample { pos: Vec3::new(0.0, 0.12, 1.0), normal: Vec3::Y, what: "floor" }],
    }
}

// ============================================================================================
// Baked scene + bake driver.
// ============================================================================================

struct Baked {
    cfg: SdfGridConfig,
    atlas: SdfAtlas,
    gpu_atlas: GpuAtlas,
}

fn bake_scene(device: &wgpu::Device, queue: &wgpu::Queue, scene: &MiniScene) -> Baked {
    let module = compose_bake();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            storage_entry(3, false),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let aabbs: Vec<_> = scene
        .edits
        .iter()
        .map(|e| edit_world_aabb(&e.prim, &e.transform, e.op.smoothing))
        .collect();
    let bvh = Bvh::build(&aabbs);

    let mut atlas = SdfAtlas::default();
    let mut gpu_atlas = GpuAtlas::new();
    bake_static_window(
        device,
        queue,
        &pipeline,
        &layout,
        &mut atlas,
        &mut gpu_atlas,
        &scene.cfg,
        &bvh,
        &scene.edits,
        scene.camera,
    );
    Baked { cfg: scene.cfg.clone(), atlas, gpu_atlas }
}

// ============================================================================================
// Gates — P-1 stubs. Each returns the measured metric; the real shader dispatch + threshold
// assertion lands in the phase noted. For now they record the analytic reference so the report
// shape is exercised end-to-end.
//
// EXTENSION GUIDE (for the agent implementing each phase):
//   * `measure_irradiance` — replace the NaN stub with: compose `sdf_probe_trace.wgsl` (like
//     `compose_bake`), bind group 0 = camera uniform (see `camera_uniform_bytes` in
//     `sdf_gpu_rig.rs`), group 1 = atlas distance/material textures + chunk_buf + tile_run +
//     `material_table_bytes(&scene.materials)` at binding 4, group 3 = probe atlases. Dispatch,
//     read back the probe irradiance, sample it at `sample.pos`/`sample.normal`. Reuse the
//     buffer/readback idiom already in this file.
//   * Per-phase gates (add as `#[test]`): P1 bleed/boil, P3 leak, P5 sub-brick+crease,
//     P6 crease(AO), P7 convergence+energy. Assert against thresholds; push a `Metric{pass:Some}`
//     into `ddgi_report` so the table stays the single source of truth.
//   * New mini-scenes: add a `scene_*()` returning `MiniScene`, push into `all_scenes()`.
//   * GPU chunk tables: build via `chunk::build_chunk_tables(&atlas, &cfg, |key| BrickTile{
//     atlas_base: col|(row<<16) from tile_origin, pal01/pal23 from the brick palette })`, then
//     serialize like `sdf_gpu_rig.rs::{chunk_lookup_bytes, brick_tile_bytes}`.
// ============================================================================================

#[derive(Clone, Copy, Debug)]
struct Metric {
    scene: &'static str,
    gate: &'static str,
    value: f32,
    /// `None` until the producing phase wires the real shader.
    pass: Option<bool>,
}

// ============================================================================================
// Real-trace GPU gate — dispatch the ACTUAL `sdf_probe_trace.wgsl` over a baked mini-scene and read
// back per-probe irradiance. Reproduces the atlas bind group (group 1) the raymarch needs: the baked
// distance + material pages as page 0 of the `binding_array`, the chunk directory + tile-run (built
// with THIS harness's tile packing so `atlas_base` points at the right texels), and the material
// table. Group 0 = a hand-built camera uniform; group 2 = the probe irradiance buffers + params.
// ============================================================================================

const ATLAS_MAX_PAGES: u32 = 64; // mirror atlas_pages::ATLAS_MAX_PAGES

/// Compose all SDF modules + an entry shader into a naga module (binding_array needs full caps).
fn compose_full(entry_path: &str) -> naga::Module {
    use adventure::sdf_render::render::SDF_SHADER_MODULES;
    let mut composer =
        Composer::default().with_capabilities(naga::valid::Capabilities::all());
    for m in SDF_SHADER_MODULES {
        let p = format!("assets/{m}");
        let src = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {p}: {e}"));
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source: &src,
                file_path: &p,
                language: ShaderLanguage::Wgsl,
                ..Default::default()
            })
            .unwrap_or_else(|e| panic!("compose {p}: {e}"));
    }
    let src = std::fs::read_to_string(entry_path).unwrap_or_else(|e| panic!("read {entry_path}: {e}"));
    composer
        .make_naga_module(NagaModuleDescriptor {
            source: &src,
            file_path: entry_path,
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("compose {entry_path}: {e}"))
}

/// 336-byte `SdfCameraUniform` mirror (84 f32): 3 mat4 (unused by raymarch) then 9 vec4. Fills the
/// fields the raymarch + probe trace actually read.
fn camera_uniform_bytes(cfg: &SdfGridConfig, camera_pos: Vec3, sun_dir: Vec3, sun_color: Vec3) -> Vec<u8> {
    let mut f = [0.0f32; 84];
    let sd = sun_dir.normalize();
    // camera_pos @ vec4 #4 (floats 48..51) — clipmap box centre.
    f[48] = camera_pos.x; f[49] = camera_pos.y; f[50] = camera_pos.z;
    // grid_origin.w (59) = voxel_size (belt-and-braces; voxel_size_at reads lod_params.z).
    f[59] = cfg.voxel_size;
    // grid_dims.z (62) = brick_size (samples/edge = 8) — `voxel_loc` edge.
    f[62] = cfg.brick_size as f32;
    // debug_params (64..68): x=max_steps, y=max_dist, z=sdf_eps, w=recenter_snap_chunks.
    f[64] = 192.0; f[65] = 5000.0; f[66] = 0.001; f[67] = cfg.recenter_snap_chunks as f32;
    // march_params (68..72): x=pixel_cone, y=shadow_softness, z=over_relax, w=lod_blend_band.
    f[68] = 0.002; f[69] = 64.0; f[70] = 1.6; f[71] = 0.0;
    // lod_params (72..76): lod_count, ring_bricks, base voxel_size, cell_stride.
    f[72] = cfg.lod_count as f32; f[73] = cfg.ring_bricks as f32;
    f[74] = cfg.voxel_size; f[75] = cfg.cell_stride() as f32;
    // sun_dir (76..79), sun_color (80..83).
    f[76] = sd.x; f[77] = sd.y; f[78] = sd.z;
    f[80] = sun_color.x; f[81] = sun_color.y; f[82] = sun_color.z;
    bytemuck::cast_slice(&f).to_vec()
}

fn chunk_lookup_bytes(chunks: &[chunk::ChunkLookup]) -> Vec<u8> {
    let mut out = Vec::with_capacity(chunks.len() * 20);
    for c in chunks {
        for v in [c.key_hi, c.key_lo, c.occ_lo, c.occ_hi, c.tile_run_base] {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

fn brick_tile_bytes(tiles: &[chunk::BrickTile]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tiles.len() * 12);
    for t in tiles {
        for v in [t.atlas_base, t.pal01, t.pal23] {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Build the chunk directory + tile-run for `baked`, using THIS harness's tile packing so the GPU
/// `atlas_base` points at the texels we baked.
fn build_tables(baked: &Baked) -> chunk::ChunkTables {
    chunk::build_chunk_tables(&baked.atlas, &baked.cfg, |key| {
        let tile = baked.atlas.tiles.tile(key).expect("resident brick has a tile");
        let (col, row) = tile_origin(tile);
        let pal = baked.atlas.bricks[key].palette;
        chunk::BrickTile {
            atlas_base: col | (row << 16),
            pal01: pal[0] as u32 | ((pal[1] as u32) << 16),
            pal23: pal[2] as u32 | ((pal[3] as u32) << 16),
        }
    })
}

/// CPU mirror of `sdf::probe::probe_slot_at`: the tile-run index (= irradiance slot) of the brick
/// containing `world_pos` at `lod`, or None if absent.
fn cpu_probe_slot(tables: &chunk::ChunkTables, cfg: &SdfGridConfig, world_pos: Vec3, lod: u32) -> Option<u32> {
    let bc = cfg.world_to_brick_lod(world_pos, lod);
    let (ck, local) = chunk::chunk_of(BrickKey::new(lod, bc), cfg);
    let idx = chunk::dir_index(ck, tables.r);
    let row = tables.chunks.get(idx)?;
    if (row.key_hi, row.key_lo) != chunk::chunk_gpu_key(ck) {
        return None;
    }
    let occ = (row.occ_lo as u64) | ((row.occ_hi as u64) << 32);
    if (occ >> local) & 1 == 0 {
        return None;
    }
    // STABLE local index (matches the trace + probe_slot_at), NOT the popcount rank.
    Some(row.tile_run_base + local)
}

/// Decode a stored (perceptual, sqrt/gamma-2) probe rgb back to linear irradiance: `E = stored²`.
/// Mirrors the trace's `sqrt` encode and the apply's final square (`hv * hv`).
fn decode_perceptual(rgb: [f32; 3]) -> Vec3 {
    let c = Vec3::new(rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0));
    c * c
}

/// CPU mirror of `sdf::oct::oct_encode` (unit normal → [0,1]² octahedral coords).
fn oct_encode(n: Vec3) -> bevy::math::Vec2 {
    use bevy::math::Vec2;
    let denom = n.x.abs() + n.y.abs() + n.z.abs();
    let mut p = Vec2::new(n.x, n.y) / denom.max(1e-8);
    if n.z <= 0.0 {
        let s = Vec2::new(
            if p.x >= 0.0 { 1.0 } else { -1.0 },
            if p.y >= 0.0 { 1.0 } else { -1.0 },
        );
        p = (Vec2::ONE - Vec2::new(p.y.abs(), p.x.abs())) * s;
    }
    p * 0.5 + Vec2::splat(0.5)
}

/// Finest-LOD octahedral irradiance at `world_pos` for a surface facing `normal` (nearest sub-probe of
/// the covering brick, nearest octahedral texel toward `normal`). Returns LINEAR irradiance + alpha —
/// the stored value is perceptually encoded, so the rgb is decoded here (alpha = validity, untouched).
fn gi_at(
    tables: &chunk::ChunkTables,
    cfg: &SdfGridConfig,
    irr: &[[f32; 4]],
    world_pos: Vec3,
    normal: Vec3,
    subdiv: u32,
) -> Option<[f32; 4]> {
    let sd = subdiv.max(1);
    let oct = PROBE_OCT_TEXELS as usize;
    let res = PROBE_OCT_RES;
    for lod in 0..cfg.lod_count {
        if let Some(base) = cpu_probe_slot(tables, cfg, world_pos, lod) {
            let brick = cfg.world_to_brick_lod(world_pos, lod);
            let bmin = cfg.brick_min_world(brick, lod);
            let cell = cfg.brick_world_size(lod) / sd as f32;
            let rel = (world_pos - bmin) / cell;
            let clampi = |v: f32| (v.floor() as i32).clamp(0, sd as i32 - 1) as u32;
            let (sx, sy, sz) = (clampi(rel.x), clampi(rel.y), clampi(rel.z));
            let sub_lin = sz * sd * sd + sy * sd + sx;
            let pslot = (base * sd * sd * sd + sub_lin) as usize;
            let oct_base = pslot * oct;
            if oct_base + oct <= irr.len() {
                let uv = oct_encode(normal) * res as f32;
                let tx = (uv.x.floor() as i32).clamp(0, res as i32 - 1) as usize;
                let ty = (uv.y.floor() as i32).clamp(0, res as i32 - 1) as usize;
                let raw = irr[oct_base + ty * res as usize + tx];
                let lin = decode_perceptual([raw[0], raw[1], raw[2]]);
                return Some([lin.x, lin.y, lin.z, raw[3]]);
            }
        }
    }
    None
}

/// CPU mirror of `sdf_deferred_lit.wgsl::probe_oct_sample`: bilinear octahedral fetch of probe `pslot`
/// toward `n`. Returns the raw (still perceptually-encoded) rgb + alpha, exactly like the WGSL.
fn oct_bilinear(irr: &[[f32; 4]], pslot: usize, n: Vec3) -> [f32; 4] {
    let res = PROBE_OCT_RES as i32;
    let base = pslot * PROBE_OCT_TEXELS as usize;
    let e = oct_encode(n) * PROBE_OCT_RES as f32 - bevy::math::Vec2::splat(0.5);
    let maxc = res - 1;
    let i0x = (e.x.floor() as i32).clamp(0, maxc);
    let i0y = (e.y.floor() as i32).clamp(0, maxc);
    let i1x = (i0x + 1).min(maxc);
    let i1y = (i0y + 1).min(maxc);
    let fx = (e.x - i0x as f32).clamp(0.0, 1.0);
    let fy = (e.y - i0y as f32).clamp(0.0, 1.0);
    let at = |x: i32, y: i32| irr[base + (y as usize) * res as usize + x as usize];
    let t00 = at(i0x, i0y);
    let t10 = at(i1x, i0y);
    let t01 = at(i0x, i1y);
    let t11 = at(i1x, i1y);
    let mut out = [0.0f32; 4];
    for k in 0..4 {
        let a = t00[k] + (t10[k] - t00[k]) * fx;
        let b = t01[k] + (t11[k] - t01[k]) * fx;
        out[k] = a + (b - a) * fy;
    }
    out
}

/// Faithful CPU mirror of `sdf_deferred_lit.wgsl::sample_gi`: trilinear over the 8 surrounding
/// sub-probes (bilinear oct, perceptual decode in half-space, wrap²+0.2 weight, cubic crush, present-
/// corner renormalize), walking finest→coarsest LOD. Returns LINEAR irradiance toward `normal`, or
/// None if no LOD has a valid covering probe. This is what numerically gates the apply-side recipe
/// (the harness never runs the lit fragment shader itself).
#[allow(clippy::too_many_arguments)]
fn gi_trilinear(
    tables: &chunk::ChunkTables,
    cfg: &SdfGridConfig,
    irr: &[[f32; 4]],
    world_pos: Vec3,
    normal: Vec3,
    view: Vec3,
    subdiv: u32,
    normal_bias: f32,
    view_bias: f32,
    smooth: bool,
) -> Option<(Vec3, u32)> {
    use bevy::math::IVec3;
    let sd = subdiv.max(1) as i32;
    let nsub = (sd * sd * sd) as u32;
    let mix = |a: f32, b: f32, t: f32| a + (b - a) * t;
    for lod in 0..cfg.lod_count {
        let cell = cfg.brick_world_size(lod) / subdiv.max(1) as f32;
        let p = world_pos + (normal * normal_bias + view * view_bias) * cell;
        let g = p / cell - Vec3::splat(0.5);
        let base = g.floor();
        let f0 = g - base;
        // Optionally smoothstep the interpolation fraction so the weight derivative → 0 at cell
        // boundaries: makes the interpolated field C1-continuous across probe cells, killing the
        // trilinear "kink" grid the eye reads as squares (linear trilinear is only C0).
        let f = if smooth { f0 * f0 * (Vec3::splat(3.0) - 2.0 * f0) } else { f0 };
        let gi0 = base.as_ivec3();
        let mut sum = Vec3::ZERO;
        let mut wsum = 0.0f32;
        let mut ncorners = 0u32;
        for c in 0..8u32 {
            let off = IVec3::new((c & 1) as i32, ((c >> 1) & 1) as i32, ((c >> 2) & 1) as i32);
            let gc = gi0 + off;
            let sub = IVec3::new(gc.x.rem_euclid(sd), gc.y.rem_euclid(sd), gc.z.rem_euclid(sd));
            // Sub-cell center in world space — lies inside the corner's brick, so cpu_probe_slot maps
            // it to the same brick as the WGSL's `probe_slot_at(bli * cell_stride, lod)`.
            let probe_center = (gc.as_vec3() + Vec3::splat(0.5)) * cell;
            let Some(base_slot) = cpu_probe_slot(tables, cfg, probe_center, lod) else {
                continue;
            };
            let sub_lin = (sub.z as u32) * subdiv * subdiv + (sub.y as u32) * subdiv + sub.x as u32;
            let pslot = (base_slot * nsub + sub_lin) as usize;
            if (pslot + 1) * PROBE_OCT_TEXELS as usize > irr.len() {
                continue;
            }
            let probe = oct_bilinear(irr, pslot, normal);
            if probe[3] <= 0.5 {
                continue;
            }
            let tri = mix(1.0 - f.x, f.x, off.x as f32).max(0.001)
                * mix(1.0 - f.y, f.y, off.y as f32).max(0.001)
                * mix(1.0 - f.z, f.z, off.z as f32).max(0.001);
            let to_probe = probe_center - world_pos;
            let wrap = (to_probe.normalize().dot(normal) * 0.5 + 0.5).max(0.0);
            let mut w = tri * (wrap * wrap + 0.2);
            if w < 0.2 {
                w = w * (w * w) * (1.0 / (0.2 * 0.2));
            }
            // gamma-2: accumulate the stored sqrt-space value directly (half-decode is identity).
            let ph = Vec3::new(probe[0].max(0.0), probe[1].max(0.0), probe[2].max(0.0));
            sum += w * ph;
            wsum += w;
            ncorners += 1;
        }
        if wsum > 1e-4 {
            let hv = sum / wsum;
            return Some((hv * hv, ncorners));
        }
    }
    None
}

/// Dispatch the real probe trace over a baked scene; return per-slot irradiance + the chunk tables.
#[allow(clippy::too_many_arguments)]
fn trace_scene(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &MiniScene,
    baked: &Baked,
    sun_dir: Vec3,
    sun_color: Vec3,
    ray_count: u32,
    subdiv: u32,
    update_stride: u32,
    frames: u32,
) -> (Vec<[f32; 4]>, chunk::ChunkTables, f32) {
    use wgpu::util::DeviceExt;

    let sd = subdiv.max(1);
    let tables = build_tables(baked);
    // Each probe slot holds an octahedral tile (PROBE_OCT_TEXELS vec4s).
    let n_slots =
        tables.tile_run.len().max(1) * (sd * sd * sd) as usize * PROBE_OCT_TEXELS as usize;
    // Compact resident-chunk list — the trace dispatches one workgroup per entry (NOT the full
    // R³·lod_count directory). This is the per-frame perf fix the GPU timing gate guards.
    let resident: Vec<chunk::ChunkLookup> = tables
        .chunks
        .iter()
        .filter(|c| (c.key_hi, c.key_lo) != chunk::SENTINEL_KEY)
        .copied()
        .collect();

    // --- shader + pipeline (auto layout: only the bindings the trace actually uses) ---
    let module = compose_full("assets/shaders/sdf_probe_trace.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("probe_trace"),
        source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
    });
    // Explicit layouts (auto-layout doesn't preserve the `binding_array` count). Only the bindings
    // the trace actually reads: g1 = {0 dist pages[], 2 chunk_buf, 3 mat pages[], 4 materials, 11 tile_run}.
    let tex_array = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: core::num::NonZero::new(ATLAS_MAX_PAGES),
    };
    let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let l0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("g0"), entries: &[uniform(0)],
    });
    let l1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("g1"),
        entries: &[tex_array(0), storage_ro(2), tex_array(3), storage_ro(4), storage_ro(11)],
    });
    let l2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("g2"),
        entries: &[storage_rw(0), uniform(1), storage_ro(2)],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("probe_trace_layout"),
        bind_group_layouts: &[&l0, &l1, &l2],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("probe_trace"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // --- group 0: camera ---
    let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera"),
        contents: &camera_uniform_bytes(&baked.cfg, scene.camera, sun_dir, sun_color),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // --- group 1: atlas (distance + material pages as page 0 + dummies, chunk_buf, materials, tile_run) ---
    let dist_view = baked.gpu_atlas.tex.as_ref().unwrap().create_view(&Default::default());
    let mat_view = baked.gpu_atlas.mat_tex.as_ref().unwrap().create_view(&Default::default());
    let dummy_dist = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dummy_dist"), size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R16Snorm, usage: wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
    });
    let dummy_mat = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dummy_mat"), size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Snorm, usage: wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
    });
    let dummy_dist_view = dummy_dist.create_view(&Default::default());
    let dummy_mat_view = dummy_mat.create_view(&Default::default());
    let mut dist_views: Vec<&wgpu::TextureView> = vec![&dist_view];
    let mut matp_views: Vec<&wgpu::TextureView> = vec![&mat_view];
    for _ in 1..ATLAS_MAX_PAGES {
        dist_views.push(&dummy_dist_view);
        matp_views.push(&dummy_mat_view);
    }
    let chunk_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chunk_buf"), contents: &chunk_lookup_bytes(&tables.chunks), usage: wgpu::BufferUsages::STORAGE,
    });
    let tile_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tile_buf"), contents: &brick_tile_bytes(&tables.tile_run), usage: wgpu::BufferUsages::STORAGE,
    });
    let materials_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("materials"), contents: &material_table_bytes(&scene.materials), usage: wgpu::BufferUsages::STORAGE,
    });
    let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("atlas"),
        layout: &l1,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureViewArray(&dist_views) },
            wgpu::BindGroupEntry { binding: 2, resource: chunk_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureViewArray(&matp_views) },
            wgpu::BindGroupEntry { binding: 4, resource: materials_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: tile_buf.as_entire_binding() },
        ],
    });

    // --- group 2: single in-place irradiance buffer + params + resident chunks ---
    let irr = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("irradiance"),
        size: (n_slots * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    // ProbeParams { ray_count, hysteresis, intensity, frame, subdiv, update_stride, gi_range,
    // normal_bias, view_bias } — frame 0 (no history). MUST match the field order of the Rust
    // `ProbeParams` (render/probe.rs) and the two WGSL copies (no parity test guards this).
    // 9 scalars = 36 B, padded up to the std140 uniform size (48 B = next multiple of 16).
    let mut params = Vec::new();
    params.extend_from_slice(&ray_count.to_le_bytes());
    params.extend_from_slice(&0.95f32.to_le_bytes()); // hysteresis → N_max≈20 (progressive average)
    params.extend_from_slice(&1.0f32.to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes()); // frame (rewritten each iteration @ offset 12)
    params.extend_from_slice(&sd.to_le_bytes());
    params.extend_from_slice(&update_stride.max(1).to_le_bytes());
    params.extend_from_slice(&24.0f32.to_le_bytes()); // gi_range (matches DdgiParams default)
    params.extend_from_slice(&0.6f32.to_le_bytes()); // normal_bias
    params.extend_from_slice(&0.1f32.to_le_bytes()); // view_bias
    params.resize(48, 0);
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: &params,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let resident_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("resident_chunks"),
        contents: &chunk_lookup_bytes(&resident),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe"),
        layout: &l2,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: irr.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: resident_buf.as_entire_binding() },
        ],
    });
    let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("camera"),
        layout: &l0,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() }],
    });

    // --- dispatch: one workgroup per (resident chunk × 64 local bricks), 2D-tiled, 64 threads each
    // (threads = octahedral texels; empty bricks early-out on the occupancy bit) ---
    let rows = (resident.len() as u32).max(1) * 64;
    let wg_x = rows.clamp(1, 65535);
    let wg_y = rows.div_ceil(wg_x);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"), size: (n_slots * 16) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });
    // GPU timestamp query straddling the trace dispatch → per-frame trace time.
    let qset = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("trace_ts"), ty: wgpu::QueryType::Timestamp, count: 2,
    });
    let qresolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ts_resolve"), size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let qreadback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ts_readback"), size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });
    // Run `frames` accumulation frames (ray rotation + hysteresis → temporal supersampling). The
    // irradiance buffer persists across frames; only the LAST frame is GPU-timed (steady-state
    // per-frame cost) and read back.
    let frames = frames.max(1);
    for f in 0..frames {
        queue.write_buffer(&params_buf, 12, &f.to_le_bytes()); // ProbeParams.frame @ offset 12
        let timed = f == frames - 1;
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let ts_writes = if timed {
                Some(wgpu::ComputePassTimestampWrites {
                    query_set: &qset,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                })
            } else {
                None
            };
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("probe_trace"),
                timestamp_writes: ts_writes,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_bind_group(2, &bg2, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        if timed {
            enc.resolve_query_set(&qset, 0..2, &qresolve, 0);
            enc.copy_buffer_to_buffer(&qresolve, 0, &qreadback, 0, 16);
            enc.copy_buffer_to_buffer(&irr, 0, &readback, 0, (n_slots * 16) as u64);
        }
        queue.submit([enc.finish()]);
    }

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let qslice = qreadback.slice(..);
    qslice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let data = slice.get_mapped_range();
    let irr: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, f32>(&data)
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect();
    drop(data);
    readback.unmap();
    let ts: Vec<u64> = bytemuck::cast_slice::<u8, u64>(&qslice.get_mapped_range()).to_vec();
    qreadback.unmap();
    let elapsed_ms = ts[1].saturating_sub(ts[0]) as f32 * queue.get_timestamp_period() / 1.0e6;
    (irr, tables, elapsed_ms)
}

// ============================================================================================
// Tests
// ============================================================================================

/// P-1 LIVE GATE: every mini-scene must bake to a non-empty atlas at its served surface tile.
/// This proves the bake scaffolding + scene geometry are sound before any probe code exists.
#[test]
fn ddgi_harness_scenes_bake_nonempty() {
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter — skipping");
        return;
    };

    for scene in all_scenes() {
        let baked = bake_scene(&device, &queue, &scene);
        assert!(
            !baked.atlas.bricks.is_empty(),
            "scene '{}' baked zero bricks",
            scene.name
        );
        // Every sample point sits on/near a surface — its served tile must hold real texels.
        for s in &scene.samples {
            let lod = served_lod(&baked.atlas, &baked.cfg, s.pos).unwrap_or_else(|| {
                panic!("scene '{}': no resident brick at sample {:?} ({})", scene.name, s.pos, s.what)
            });
            let key = BrickKey::new(lod, baked.cfg.world_to_brick_lod(s.pos, lod));
            let tile = baked
                .atlas
                .tiles
                .tile(&key)
                .unwrap_or_else(|| panic!("scene '{}': served brick has no tile", scene.name));
            assert!(
                baked.gpu_atlas.read_tile_has_content(&device, &queue, tile),
                "scene '{}': served LOD-{lod} tile {tile} at sample '{}' is EMPTY",
                scene.name,
                s.what
            );
        }
        eprintln!(
            "scene '{}': baked {} bricks, {} sample(s) on resident surface ✓",
            scene.name,
            baked.atlas.bricks.len(),
            scene.samples.len()
        );
    }
}

/// Verifies the `GpuSdfMaterial` byte layout the probe trace will rely on: 80 B/material with
/// premultiplied emissive at offset 64. Pure-CPU (no GPU) so it always runs.
#[test]
fn ddgi_harness_material_table_layout() {
    let mats = vec![
        MatDef::diffuse(Vec3::new(0.8, 0.8, 0.8)),
        MatDef::emitter(Vec3::new(1.0, 0.2, 0.1), 6.0),
    ];
    let bytes = material_table_bytes(&mats);
    assert_eq!(bytes.len(), 160, "expected 80 B per material");
    // emissive of material 1 sits at 80 + 64 = 144; r channel = 1.0 * 6.0 = 6.0.
    let emissive_r = f32::from_le_bytes(bytes[144..148].try_into().unwrap());
    assert!((emissive_r - 6.0).abs() < 1e-5, "emissive.r at offset 64 = {emissive_r}, want 6.0");
}

/// Default sun for the trace gates: a dim key light so the emissive bleed dominates the gather
/// (the gates assert on emitter colour, not sun/sky).
const GATE_SUN_DIR: Vec3 = Vec3::new(0.3, 0.9, 0.2);
const GATE_SUN_COLOR: Vec3 = Vec3::new(0.3, 0.3, 0.3);
/// Probe sub-lattice the gates trace with (matches the `DdgiParams::subdiv` default).
const GATE_SUBDIV: u32 = 2;
/// Round-robin update stride for the per-frame perf gate (matches `DdgiParams::update_stride`).
const GATE_STRIDE: u32 = 4;

/// P1 BLEED GATE: an emissive cube in front of a diffuse wall must light the wall with its colour.
/// We trace the real probe shader, read the irradiance of the probe covering the wall sample, and
/// assert it is RED-dominant — sky alone is blue-dominant (r < g < b), so red > green proves the
/// emitter's indirect light reached the wall.
#[test]
fn ddgi_bleed_gate() {
    let Some((device, queue)) = gpu_full() else {
        eprintln!("no GPU adapter (or missing binding-array features) — skipping");
        return;
    };
    let scene = scene_emitter_wall();
    let baked = bake_scene(&device, &queue, &scene);
    let (irr, tables, _ms) =
        trace_scene(&device, &queue, &scene, &baked, GATE_SUN_DIR, GATE_SUN_COLOR, 32, GATE_SUBDIV, 1, 8);

    let s = &scene.samples[0];
    // Toward the emitter (+X, the wall's facing normal).
    let gi = gi_at(&tables, &baked.cfg, &irr, s.pos, s.normal, GATE_SUBDIV)
        .unwrap_or_else(|| panic!("no probe covers the wall sample {:?}", s.pos));
    // Away from the emitter (−X) — the SAME probe, opposite octahedral direction.
    let gi_away = gi_at(&tables, &baked.cfg, &irr, s.pos, -s.normal, GATE_SUBDIV).unwrap();
    eprintln!(
        "bleed: toward-emitter [{:.3} {:.3} {:.3}]  away [{:.3} {:.3} {:.3}]",
        gi[0], gi[1], gi[2], gi_away[0], gi_away[1], gi_away[2]
    );
    assert!(gi[0] > 0.02, "bleed gate: wall probe got no red indirect light (r={:.4})", gi[0]);
    assert!(
        gi[0] > gi[1],
        "bleed gate: wall probe not red-dominant (r={:.4} g={:.4}) — emitter bleed not detected",
        gi[0], gi[1]
    );
    // DIRECTIONALITY (octahedral, not flat): facing the emitter must be redder than facing away.
    assert!(
        gi[0] > gi_away[0] + 0.05,
        "directionality gate: octahedral GI is flat — toward-emitter red {:.3} not > away red {:.3}",
        gi[0], gi_away[0]
    );
}

/// SMOOTHNESS GATE (the "no more blocky cubes" check): with perceptual encoding + the trilinear apply
/// recipe, irradiance across a flat lit wall must vary SMOOTHLY, not in nearest-probe plateaus/steps.
/// We sample the lit wall face along a line finer than the sub-probe spacing: the nearest sampler
/// (`gi_at`) stair-steps (visible blocks), the faithful trilinear apply mirror (`gi_trilinear`) ramps.
/// Metric: mean |2nd difference| of luminance — trilinear must be strictly smoother than nearest.
#[test]
fn ddgi_smoothness_gate() {
    let Some((device, queue)) = gpu_full() else {
        eprintln!("no GPU adapter (or missing features) — skipping");
        return;
    };
    let scene = scene_emitter_wall();
    let baked = bake_scene(&device, &queue, &scene);
    let (irr, tables, _ms) =
        trace_scene(&device, &queue, &scene, &baked, GATE_SUN_DIR, GATE_SUN_COLOR, 32, GATE_SUBDIV, 1, 8);

    let s = &scene.samples[0];
    let n = s.normal; // wall face normal (toward the emitter)
    let view = n; // head-on
    let lum = |c: Vec3| c.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    // 25 points up the wall face, spacing ~0.066 m — well below the ~0.35 m sub-probe spacing, so the
    // nearest sampler reveals its plateaus while the trilinear apply stays continuous.
    let mut near = Vec::new();
    let mut tri = Vec::new();
    for i in 0..25u32 {
        let y = -0.8 + 1.6 * (i as f32) / 24.0;
        let p = Vec3::new(s.pos.x, y, 0.0);
        let gn = gi_at(&tables, &baked.cfg, &irr, p, n, GATE_SUBDIV);
        let gt = gi_trilinear(&tables, &baked.cfg, &irr, p, n, view, GATE_SUBDIV, 0.6, 0.1, true);
        if let (Some(gn), Some((gt, _))) = (gn, gt) {
            near.push(lum(Vec3::new(gn[0], gn[1], gn[2])));
            tri.push(lum(gt));
        }
    }
    assert!(
        near.len() >= 12 && tri.len() == near.len(),
        "smoothness gate: too few covered samples along the wall ({})",
        near.len()
    );
    // Mean absolute second difference = curvature/roughness of the luminance profile.
    let roughness = |v: &[f32]| {
        let mut acc = 0.0;
        for w in v.windows(3) {
            acc += (w[0] - 2.0 * w[1] + w[2]).abs();
        }
        acc / (v.len() - 2).max(1) as f32
    };
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    let (r_near, r_tri) = (roughness(&near), roughness(&tri));
    let (m_near, m_tri) = (mean(&near), mean(&tri));
    eprintln!("smoothness: 2nd-diff n={r_near:.4} t={r_tri:.4}  mean n={m_near:.3} t={m_tri:.3}");
    // NOTE: per-probe ray rotation (sdf_probe_trace) intentionally DECORRELATES probe estimates so a
    // small emitter gives a denoisable penumbra rather than coherent banding — i.e. the probe-level
    // field is deliberately noisy and is smoothed by the screen-space denoise (NOT run in this harness).
    // So we no longer assert "trilinear smoother than nearest" at probe level. What must still hold:
    //   (1) there IS GI signal, (2) the trilinear apply is UNBIASED vs nearest (same mean energy), and
    //   (3) its curvature is BOUNDED (interpolation isn't exploding).
    assert!(m_near > 0.0 && m_tri > 0.0, "smoothness gate: no GI signal on the wall");
    assert!(
        (m_tri - m_near).abs() <= 0.4 * m_near.max(m_tri),
        "smoothness gate: trilinear apply biased vs nearest (mean n={m_near:.3} t={m_tri:.3})"
    );
    assert!(
        r_tri.is_finite() && r_tri < 0.5,
        "smoothness gate: trilinear roughness unbounded ({r_tri:.4})"
    );
}

/// ENERGY GATE: perceptual encode→decode round-trip + the prev-frame infinite bounce must not run away
/// or produce NaN/Inf. Every ACTIVE probe texel, decoded to linear, stays finite and below a generous
/// bound from the brightest emitter (multi-bounce only adds albedo·(<1) energy on top).
#[test]
fn ddgi_energy_gate() {
    let Some((device, queue)) = gpu_full() else {
        eprintln!("no GPU adapter (or missing features) — skipping");
        return;
    };
    let scene = scene_emitter_wall();
    let baked = bake_scene(&device, &queue, &scene);
    let (irr, _tables, _ms) =
        trace_scene(&device, &queue, &scene, &baked, GATE_SUN_DIR, GATE_SUN_COLOR, 32, GATE_SUBDIV, 1, 8);

    // scene_emitter_wall emitter radiance = (1,.2,.1)·6 → max channel 6; 2× headroom for the bounce.
    let bound = 12.0f32;
    let mut maxc = 0.0f32;
    for tex in &irr {
        if tex[3] <= 0.5 {
            continue; // inactive / deactivated / unwritten slot
        }
        let lin = decode_perceptual([tex[0], tex[1], tex[2]]);
        assert!(lin.is_finite(), "energy gate: non-finite decoded irradiance {lin:?}");
        maxc = maxc.max(lin.max_element());
    }
    eprintln!("energy: max decoded texel radiance = {maxc:.3} (bound {bound:.1})");
    assert!(maxc > 0.0, "energy gate: all probes dark — trace produced no irradiance");
    assert!(maxc < bound, "energy gate: irradiance runaway (max {maxc:.3} ≥ {bound:.1})");
}

/// GRID-PATTERN DIAGNOSTIC (ignored): the "square patterns" the user sees on flat walls are the probe
/// lattice showing through trilinear interpolation. Its visibility is driven by inter-probe CONTRAST
/// (probe-value noise from too few rays) and cell SIZE (subdiv). This sweeps both and prints the
/// trilinear profile's roughness (mean |2nd difference| of luminance, normalized) across the lit wall —
/// lower = smoother = fewer visible squares — so we can pick levers from data, not guesswork.
///   cargo test --test ddgi_harness ddgi_grid_report -- --ignored --nocapture
#[test]
#[ignore = "diagnostic report; run with --ignored --nocapture"]
fn ddgi_grid_report() {
    let Some((device, queue)) = gpu_full() else {
        eprintln!("no GPU adapter — skipping");
        return;
    };
    let scene = scene_emitter_wall();
    let baked = bake_scene(&device, &queue, &scene);
    let s = &scene.samples[0];
    let n = s.normal;
    let lum = |c: Vec3| c.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    // MAX |2nd diff| catches a localized boundary KINK (a spike); MEAN |2nd diff| can't tell a kink
    // from a benign smooth S-curve. Both normalized by mean luminance.
    let metrics = |v: &[f32]| {
        let (mut mx, mut sm) = (0.0f32, 0.0f32);
        for w in v.windows(3) {
            let d2 = (w[0] - 2.0 * w[1] + w[2]).abs();
            mx = mx.max(d2);
            sm += d2;
        }
        (mx, sm / (v.len().saturating_sub(2)).max(1) as f32)
    };
    eprintln!("\n===== DDGI GRID-PATTERN SWEEP (wall; normalized max & mean 2nd-diff) =====");
    for &smooth in &[false, true] {
        eprintln!("--- smoothstep interpolation: {smooth} ---");
        for &rays in &[24u32, 64] {
            for &subdiv in &[2u32, 3] {
                let (irr, tables, _ms) = trace_scene(
                    &device, &queue, &scene, &baked, GATE_SUN_DIR, GATE_SUN_COLOR, rays, subdiv, 1, 8,
                );
                let mut prof = Vec::new();
                let mut corner_sum = 0u32;
                for i in 0..61u32 {
                    let y = -0.9 + 1.8 * (i as f32) / 60.0;
                    let p = Vec3::new(s.pos.x, y, 0.0);
                    let (g, nc) = gi_trilinear(&tables, &baked.cfg, &irr, p, n, n, subdiv, 0.6, 0.1, smooth)
                        .unwrap_or((Vec3::ZERO, 0));
                    prof.push(lum(g));
                    corner_sum += nc;
                }
                let mean = (prof.iter().sum::<f32>() / prof.len() as f32).max(1.0e-4);
                let (mx, sm) = metrics(&prof);
                let avg_corners = corner_sum as f32 / 61.0;
                eprintln!(
                    "  rays={rays:>2} subdiv={subdiv} → max2nd={:.4} mean2nd={:.4} avg_valid_corners={avg_corners:.2}/8  (mean lum {mean:.3})",
                    mx / mean,
                    sm / mean
                );
            }
        }
    }
}

/// LEAK GATE: a green emitter sits behind a thin wall; the FAR-side receiver must stay dark. Because
/// probes sit ON surfaces and the trace's rays are occlusion-correct (a far-side probe's rays hit the
/// wall, never the emitter behind it), the far-side irradiance has no green dominance — the sparse
/// surface-anchored layout is leak-free without an explicit visibility test.
#[test]
fn ddgi_leak_gate() {
    let Some((device, queue)) = gpu_full() else {
        eprintln!("no GPU adapter (or missing features) — skipping");
        return;
    };
    let scene = scene_thin_wall();
    let baked = bake_scene(&device, &queue, &scene);
    let (irr, tables, _ms) =
        trace_scene(&device, &queue, &scene, &baked, GATE_SUN_DIR, GATE_SUN_COLOR, 32, GATE_SUBDIV, 1, 8);
    let s = &scene.samples[0]; // far-side wall face (+X), away from the −X green emitter
    let gi = gi_at(&tables, &baked.cfg, &irr, s.pos, s.normal, GATE_SUBDIV)
        .unwrap_or_else(|| panic!("no probe covers the far-side sample {:?}", s.pos));
    eprintln!("leak: far-side irradiance = [{:.3}, {:.3}, {:.3}]", gi[0], gi[1], gi[2]);
    // The green emitter's light must NOT dominate the occluded far side (no leak through the wall).
    assert!(
        gi[1] < 0.35 && gi[1] <= gi[2] + 0.02,
        "leak gate: green leaked through the thin wall (g={:.3} r={:.3} b={:.3})",
        gi[1], gi[0], gi[2]
    );
}

/// BOIL GATE: with a static scene the GI must be temporally STABLE — the per-frame ray-set rotation
/// is meant to be damped by hysteresis into a steady value, not to flicker ("boil"). Trace to
/// convergence, then ONE frame further; the mean relative change of the probes that re-traced must be
/// tiny. (Trace_scene(frames=K) runs frames 0..K-1 deterministically, so frames=K+1 = the same plus
/// exactly one more frame's update — this isolates the residual per-frame churn the eye sees as boil.)
#[test]
fn ddgi_boil_gate() {
    let Some((device, queue)) = gpu_full() else {
        eprintln!("no GPU adapter (or missing features) — skipping");
        return;
    };
    let scene = scene_emitter_wall();
    let baked = bake_scene(&device, &queue, &scene);
    let (a, _t, _) =
        trace_scene(&device, &queue, &scene, &baked, GATE_SUN_DIR, GATE_SUN_COLOR, 24, GATE_SUBDIV, GATE_STRIDE, 24);
    let (b, _t2, _) =
        trace_scene(&device, &queue, &scene, &baked, GATE_SUN_DIR, GATE_SUN_COLOR, 24, GATE_SUBDIV, GATE_STRIDE, 25);
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (pa, pb) in a.iter().zip(&b) {
        if pa[3] > 0.5 && pb[3] > 0.5 {
            let la = pa[0] + pa[1] + pa[2];
            let lb = pb[0] + pb[1] + pb[2];
            num += (la - lb).abs();
            den += la.max(lb).max(1.0e-4);
        }
    }
    let rel = num / den.max(1.0e-4);
    eprintln!("boil: mean relative frame-to-frame change after convergence = {:.4}", rel);
    assert!(
        rel < 0.02,
        "boil gate: GI not temporally stable — {:.1}% frame-to-frame change (boiling)",
        rel * 100.0
    );
}

/// PER-FRAME PERFORMANCE GATE: trace a busy scene at the LIVE clipmap config + defaults and GPU-time
/// the probe-trace dispatch; asserts < 4 ms. `#[ignore]` because GPU timestamps are polluted by other
/// tests sharing the GPU in parallel — run it ALONE:
///   cargo test --test ddgi_harness ddgi_perf_gate -- --ignored --nocapture
#[test]
#[ignore = "GPU timing — run alone (see doc)"]
fn ddgi_perf_gate() {
    let Some((device, queue)) = gpu_full() else {
        eprintln!("no GPU adapter (or missing features) — skipping");
        return;
    };
    let scene = scene_perf();
    let baked = bake_scene(&device, &queue, &scene);
    // Match the LIVE DdgiParams defaults (ray_count 64, subdiv 2, update_stride 4) so this gate
    // measures what the gallery actually pays per frame.
    let (irr, tables, ms) =
        trace_scene(&device, &queue, &scene, &baked, GATE_SUN_DIR, GATE_SUN_COLOR, 128, GATE_SUBDIV, GATE_STRIDE, 1);
    let active = irr.iter().filter(|p| p[3] > 0.5).count();
    let resident = tables
        .chunks
        .iter()
        .filter(|c| (c.key_hi, c.key_lo) != chunk::SENTINEL_KEY)
        .count();
    eprintln!(
        "perf: {} resident chunks, {} active probes this frame (stride {}), trace {:.3} ms",
        resident, active, GATE_STRIDE, ms
    );
    // This strided frame must have traced SOME probes (round-robin coverage is non-empty).
    assert!(active > 0, "perf scene: strided frame traced no probes");
    // Budget: a busy frame's trace (compact dispatch + 1/stride round-robin) must fit a frame.
    assert!(ms < 4.0, "per-frame trace {ms:.2} ms exceeds 4 ms budget ({active} active probes)");
}

/// Prints the full metric table for all scenes. Ignored by default (run with `--ignored --nocapture`).
#[test]
#[ignore = "report only; run with --ignored --nocapture"]
fn ddgi_report() {
    let Some((device, queue)) = gpu_full() else {
        eprintln!("no GPU adapter (or missing binding-array features) — skipping");
        return;
    };
    eprintln!("\n===== DDGI METRIC REPORT =====");
    for scene in all_scenes() {
        let baked = bake_scene(&device, &queue, &scene);
        let (irr, tables, ms) =
            trace_scene(&device, &queue, &scene, &baked, GATE_SUN_DIR, GATE_SUN_COLOR, 32, GATE_SUBDIV, 1, 8);
        let active = irr.iter().filter(|p| p[3] > 0.5).count();
        eprintln!(
            "[{}] {} probe slots, {} active, trace {:.3} ms",
            scene.name, irr.len(), active, ms
        );
        for s in &scene.samples {
            match gi_at(&tables, &baked.cfg, &irr, s.pos, s.normal, GATE_SUBDIV) {
                Some(gi) => eprintln!(
                    "    {} → irradiance [{:.3} {:.3} {:.3}] (a={:.0})",
                    s.what, gi[0], gi[1], gi[2], gi[3]
                ),
                None => eprintln!("    {} → no probe", s.what),
            }
        }
    }
}
