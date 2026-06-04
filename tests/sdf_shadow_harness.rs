//! Headless GPU shadow-evaluation harness for `sdf::shadows::soft_shadow`.
//!
//! This is an ITERATION HARNESS for the soft-shadow algorithm, not a parity test. It bakes a
//! known two-object scene (a ground slab + a floating sphere) into the renderer's REAL
//! brick-atlas format — the actual R16Snorm / clamped / trilinear field the live march reads,
//! produced by the REAL GPU bake shader (`sdf_brick_bake.wgsl`) and the REAL chunk directory
//! (`chunk.rs::build_chunk_tables`) — then runs the REAL `soft_shadow` over a ground grid and
//! evaluates the result against a CPU ANALYTIC reference (sun-disk vs sphere occlusion).
//!
//! It prints (under `--nocapture`):
//!   * an ASCII dump of the GPU sun-visibility grid (so boxy bands / hard penumbra edges show),
//!   * RMSE + max-abs-error vs the analytic reference,
//!   * a boxiness/anisotropy metric (iso-radius ring variance) and the max visibility gradient.
//!
//! It is INTENTIONALLY loose on asserts (just: shadow exists under the sphere, far corners lit,
//! harness ran) — the metrics are diagnostic. Tighten them as the shadow algorithm improves.
//!
//! Field-path provenance (verified by reading the shaders): `soft_shadow` reads only
//! `@group(1)` `atlas_tex`(0, R16Snorm) + `chunk_buf`(2) + `chunk_tile_buf`(11), and
//! `@group(0)` `camera`(0). No sampler (it uses `textureLoad`), no material textures. The bake +
//! chunk-table construction mirror `tests/sdf_bake_gpu.rs` and `tests/sdf_gpu_rig.rs` verbatim.

use std::borrow::Cow;

use bevy::math::{IVec3, Vec3};
use naga_oil::compose::{
    Composer, ComposableModuleDescriptor, NagaModuleDescriptor, ShaderLanguage,
};

use adventure::sdf_render::atlas::{
    dist_band_world, ring_window_coords, BrickKey, SdfAtlas, BRICK_EDGE,
};
use adventure::sdf_render::bvh::Bvh;
use adventure::sdf_render::chunk::{
    build_chunk_tables, pack_brick_tile, tile_atlas_base, BrickTile, ChunkLookup,
};
use adventure::sdf_render::edits::{
    build_palette_indexed, edit_world_aabb, to_gpu_edit, GpuEdit, ResolvedEdit, SdfOp,
    SdfPrimitive,
};
use adventure::sdf_render::SdfGridConfig;

mod common;

// R16Snorm (the brick atlas format) needs TEXTURE_FORMAT_16BIT_NORM.
fn device_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
    common::headless_device(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
}

// --- bake buffer layout constants (mirror sdf_brick_bake.wgsl / sdf_bake_gpu.rs) -------------
const DIST_ROW_U32: u32 = 64;
const DIST_TILE_U32: u32 = DIST_ROW_U32 * 8; // 512

// --- Camera uniform (336 bytes, 84 f32). Mirrors bindings.wgsl::SdfCameraUniform. -----------
// 3× mat4 (48 floats) then 9× vec4 (camera_pos, screen_params, grid_origin, grid_dims,
// debug_params, march_params, lod_params, sun_dir, sun_color).
// `soft_shadow` reads: voxel_size_at / cell_stride / lod_count (lod_params), brick_size
// (grid_dims.z — REQUIRED; `brick_stride` reads it and the brick edge collapses to 0 without
// it), recenter_snap (debug_params.w, via in_ring_chunk), ring_bricks (lod_params.y), and
// camera_pos (in_ring_chunk's ring centre).
fn camera_uniform_bytes(config: &SdfGridConfig, camera_pos: Vec3) -> Vec<u8> {
    let mut f = [0.0f32; 84]; // 336 bytes
    f[48] = camera_pos.x; // camera_pos.xyz (4th field, after the 3 matrices)
    f[49] = camera_pos.y;
    f[50] = camera_pos.z;
    // grid_dims is the 7th field (offset 48 + 4*3 = 60). grid_dims.z = brick_size.
    f[62] = config.brick_size as f32;
    f[67] = config.recenter_snap_chunks as f32; // debug_params.w
    f[69] = 8.0; // march_params.y = shadow_softness k (low value to stress the band-edge onset)
    f[72] = config.lod_count as f32; // lod_params.x
    f[73] = config.ring_bricks as f32; // lod_params.y
    f[74] = config.voxel_size; // lod_params.z
    f[75] = config.cell_stride() as f32; // lod_params.w
    bytemuck::cast_slice(&f).to_vec()
}

// --- GPU job header (48 bytes, 12 u32) — mirror bake_scheduler::GpuJobHeader upload order ----
fn header_bytes(coord: IVec3, voxel_size: f32, dist_band: f32, edit_count: u32) -> Vec<u8> {
    let mut b = Vec::with_capacity(48);
    b.extend_from_slice(&coord.x.to_le_bytes());
    b.extend_from_slice(&coord.y.to_le_bytes());
    b.extend_from_slice(&coord.z.to_le_bytes());
    b.extend_from_slice(&voxel_size.to_le_bytes());
    b.extend_from_slice(&dist_band.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes()); // edit_start (edits packed per-job from 0)
    b.extend_from_slice(&edit_count.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes()); // pal01 (unused for the distance field)
    b.extend_from_slice(&0u32.to_le_bytes()); // pal23
    b.extend_from_slice(&0u32.to_le_bytes()); // pad
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
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

// 20-byte ChunkLookup serialization (reuse of sdf_gpu_rig.rs::chunk_lookup_bytes).
fn chunk_lookup_bytes(chunks: &[ChunkLookup]) -> Vec<u8> {
    let mut out = Vec::with_capacity(chunks.len() * 20);
    for c in chunks {
        out.extend_from_slice(&c.key_hi.to_le_bytes());
        out.extend_from_slice(&c.key_lo.to_le_bytes());
        out.extend_from_slice(&c.occ_lo.to_le_bytes());
        out.extend_from_slice(&c.occ_hi.to_le_bytes());
        out.extend_from_slice(&c.tile_run_base.to_le_bytes());
    }
    out
}

// 12-byte BrickTile serialization (reuse of sdf_gpu_rig.rs::brick_tile_bytes).
fn brick_tile_bytes(tiles: &[BrickTile]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tiles.len() * 12);
    for t in tiles {
        out.extend_from_slice(&t.atlas_base.to_le_bytes());
        out.extend_from_slice(&t.pal01.to_le_bytes());
        out.extend_from_slice(&t.pal23.to_le_bytes());
    }
    out
}

// --- shader composition ----------------------------------------------------------------------
const SDF_MODULES: [&str; 4] = [
    "assets/shaders/sdf/bindings.wgsl",
    "assets/shaders/sdf/brick.wgsl",
    // march before shadows: soft_shadow now imports sdf::march::lod_crossfade.
    "assets/shaders/sdf/march.wgsl",
    "assets/shaders/sdf/shadows.wgsl",
];

fn compose_entry(entry_src: &str, file: &str) -> naga::Module {
    let mut composer = Composer::default();
    for path in SDF_MODULES {
        let source = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source: &source,
                file_path: path,
                language: ShaderLanguage::Wgsl,
                ..Default::default()
            })
            .unwrap_or_else(|e| panic!("compose {path}: {e}"));
    }
    composer
        .make_naga_module(NagaModuleDescriptor {
            source: entry_src,
            file_path: file,
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("compose {file}: {e}"))
}

// The shadow entry. Reads camera(g0,b0), the ground points(g0,b1), writes vis(g0,b2). The field
// is read from atlas_tex(g1,b0) + chunk_buf(g1,b2) + chunk_tile_buf(g1,b11) — auto layout pulls
// in exactly the bindings soft_shadow touches.
const SHADOW_WGSL: &str = r#"
#import sdf::shadows::surface_shadow

struct Pt { x: f32, y: f32, z: f32, pad: f32 };
@group(0) @binding(1) var<storage, read> points: array<Pt>;
@group(0) @binding(2) var<storage, read_write> vis: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let p = vec3<f32>(points[i].x, points[i].y, points[i].z);
    let sun = normalize(vec3<f32>(0.3, 1.0, 0.2));
    // Call the REAL renderer entry: it lifts the origin off the surface by geo_n*voxel (so the
    // ground doesn't self-shadow), uses mint = vs*0.5 and k = 8. geo_n = (0,1,0) (ground top).
    vis[i] = surface_shadow(p, vec3<f32>(0.0, 1.0, 0.0), sun, 0u, 20.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PtIn {
    x: f32,
    y: f32,
    z: f32,
    pad: f32,
}

// One baked brick job: its origin coord + the GPU edits it folds + its atlas tile origin.
struct BakeJob {
    coord: IVec3,
    edits: Vec<GpuEdit>,
    tile: u32,
}

// IGNORED: this harness predates the paged/bindless atlas migration. `bindings.wgsl` now declares
// the distance atlas as `atlas_pages: binding_array<texture_2d, 64>` (page-addressed in
// `load_voxel`), but this rig still builds a SINGLE `atlas_tex` view + a `headless_device` without
// the texture-binding-array / non-uniform-indexing features — so `load_voxel` fails device shader
// validation before any shadow is compared. Reviving it needs the group-1 setup rewritten to the
// 64-element paged binding array (+ those device features), independent of the point-light work.
// Run explicitly once updated: `cargo test --test sdf_shadow_harness -- --ignored`.
#[ignore = "stale vs the paged/bindless atlas; needs the group-1 binding-array rewrite"]
#[test]
fn sdf_soft_shadow_vs_analytic_reference() {
    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter (or no TEXTURE_FORMAT_16BIT_NORM) — skipping");
        return;
    };
    use wgpu::util::DeviceExt;

    // --- 1. config + scene ------------------------------------------------------------------
    // Small ring so the resident brick set (= atlas tiles) stays well under 127 (atlas width =
    // tiles*64 must stay < 8192 since tile_atlas_base lays the first 256 tiles in row 0).
    let config = SdfGridConfig {
        lod_count: 1,
        ring_bricks: 16,
        // Coarse voxels: the sphere bakes blocky, so its shadow shows the brick-faceted
        // silhouette artifact the live renderer has at coarse LOD / distant occluders — the case
        // the fine LOD-0 default could NOT reproduce (boxiness ~0 there).
        voxel_size: 0.2,
        ..Default::default()
    };
    let lod = 0u32;
    let voxel_size = config.voxel_size_at(lod);
    let band = dist_band_world(&config, lod);

    // Ground slab: top face at y=0 (centre y=-0.3, half-height 0.3). Kept compact (half 1.2 in
    // x/z, thin in y) so the resident brick count fits one atlas row (tiles*64 < 8192 ⇒ ≤127).
    let ground = ResolvedEdit::new(
        SdfPrimitive::Box {
            half_extents: Vec3::new(1.2, 0.3, 1.2),
        },
        bevy::prelude::Transform::from_xyz(0.0, -0.3, 0.0),
        SdfOp::default(),
        0,
    );
    // Floating sphere occluder centred at (0, 1.5, 0), radius 0.5 — casts a shadow onto y=0.
    let sphere_center = Vec3::new(0.0, 1.5, 0.0);
    let sphere_radius = 0.5f32;
    let sphere = ResolvedEdit::new(
        SdfPrimitive::Sphere {
            radius: sphere_radius,
        },
        bevy::prelude::Transform::from_translation(sphere_center),
        SdfOp::default(),
        1,
    );
    let edits = [ground, sphere];

    let aabbs: Vec<_> = edits
        .iter()
        .map(|e| edit_world_aabb(&e.prim, &e.transform, e.op.smoothing))
        .collect();
    let bvh = Bvh::build(&aabbs);

    // --- 2. topology + per-brick bake jobs --------------------------------------------------
    let mut atlas = SdfAtlas::default();
    let mut jobs: Vec<BakeJob> = Vec::new();
    let mut idx: Vec<u32> = Vec::new();
    for coord in ring_window_coords(&config, config.ring_origin(Vec3::ZERO, lod)) {
        let key = BrickKey::new(lod, coord);
        if SdfAtlas::cull_edit_indices(key, &bvh, &config, &mut idx).is_none() {
            continue; // empty space — no brick
        }
        let samples = SdfAtlas::brick_palette_samples(key, voxel_size);
        let palette = build_palette_indexed(&edits, &idx, &samples);
        let tile = atlas.insert_gpu_brick(key, palette, 0, &config);
        let job_edits: Vec<GpuEdit> = idx.iter().map(|&i| to_gpu_edit(&edits[i as usize])).collect();
        jobs.push(BakeJob {
            coord,
            edits: job_edits,
            tile,
        });
    }
    assert!(!jobs.is_empty(), "scene baked no bricks");
    let n_tiles = atlas.tiles.high_water();
    eprintln!("resident bricks (atlas tiles) = {n_tiles}");
    // Atlas is one row of tiles (tile_atlas_base wraps at 256). Keep < 8192 px wide.
    let atlas_w = n_tiles * (BRICK_EDGE as u32 * BRICK_EDGE as u32); // tiles * 64
    assert!(
        atlas_w <= 8192,
        "atlas width {atlas_w} exceeds 8192 — shrink the scene"
    );
    let atlas_h = BRICK_EDGE as u32; // 8 (all tiles in row 0)

    // --- 3. the real atlas texture ----------------------------------------------------------
    let atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow_atlas_r16snorm"),
        size: wgpu::Extent3d {
            width: atlas_w,
            height: atlas_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R16Snorm,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    // --- 4. bake each brick on the GPU and copy its tile into the atlas ---------------------
    let bake_module = {
        let src = std::fs::read_to_string("assets/shaders/sdf_brick_bake.wgsl")
            .expect("read sdf_brick_bake.wgsl");
        Composer::default()
            .make_naga_module(NagaModuleDescriptor {
                source: &src,
                file_path: "sdf_brick_bake.wgsl",
                ..Default::default()
            })
            .expect("compose bake shader")
    };
    let bake_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bake"),
        source: wgpu::ShaderSource::Naga(Cow::Owned(bake_module)),
    });
    let bake_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bake_bgl"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            storage_entry(3, false),
        ],
    });
    let bake_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bake_pl"),
        bind_group_layouts: &[&bake_bgl],
        push_constant_ranges: &[],
    });
    let bake_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bake"),
        layout: Some(&bake_pl),
        module: &bake_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let edge = BRICK_EDGE as u32; // 8
    let tile_w = edge * edge; // 64
    let mat_tile_u32 = 128 * 8u32; // bake writes a material tile too; we discard it

    for job in &jobs {
        // edits packed for THIS job from index 0; header edit_start=0, edit_count=N.
        let mut edit_blob = Vec::new();
        for e in &job.edits {
            edit_blob.extend_from_slice(&edit_bytes(e));
        }
        let header_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("headers"),
            contents: &header_bytes(job.coord, voxel_size, band, job.edits.len() as u32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let edit_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("edits"),
            contents: &edit_blob,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dist_out"),
            size: (DIST_TILE_U32 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mat_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mat_out"),
            size: (mat_tile_u32 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bake_bg"),
            layout: &bake_bgl,
            entries: &[
                bind(0, &header_buf),
                bind(1, &edit_buf),
                bind(2, &dist_buf),
                bind(3, &mat_buf),
            ],
        });

        // CRITICAL: the copy origin comes from tile_atlas_base(tile) so it matches what the GPU
        // voxel_pixel decodes for this brick — never hand-rolled.
        let base = tile_atlas_base(job.tile);
        let col_px = base & 0xffff;
        let row_px = base >> 16;

        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&bake_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        enc.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &dist_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(DIST_ROW_U32 * 4), // 256
                    rows_per_image: Some(edge),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &atlas_tex,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: col_px,
                    y: row_px,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: tile_w,
                height: edge,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
    }

    // --- 5. chunk directory -----------------------------------------------------------------
    let tables = build_chunk_tables(&atlas, &config, |key| {
        pack_brick_tile(atlas.tiles.tile(key).unwrap(), atlas.bricks[key].palette)
    });
    let chunk_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chunk_buf"),
        contents: &chunk_lookup_bytes(&tables.chunks),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let tile_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chunk_tile_buf"),
        contents: &brick_tile_bytes(&tables.tile_run),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // --- 6. ground grid ---------------------------------------------------------------------
    const GRID: usize = 48;
    let lo = -2.0f32;
    let hi = 2.0f32;
    let mut points: Vec<PtIn> = Vec::with_capacity(GRID * GRID);
    let mut world_xy: Vec<(f32, f32)> = Vec::with_capacity(GRID * GRID);
    for iz in 0..GRID {
        for ix in 0..GRID {
            let x = lo + (hi - lo) * (ix as f32 / (GRID - 1) as f32);
            let z = lo + (hi - lo) * (iz as f32 / (GRID - 1) as f32);
            points.push(PtIn {
                x,
                y: 0.0,
                z,
                pad: 0.0,
            });
            world_xy.push((x, z));
        }
    }

    // --- 7. run soft_shadow -----------------------------------------------------------------
    let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera"),
        contents: &camera_uniform_bytes(&config, Vec3::ZERO),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let points_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("points"),
        contents: bytemuck::cast_slice(&points),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let vis_size = (points.len() * 4) as u64;
    let vis_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vis"),
        size: vis_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vis_readback"),
        size: vis_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let module = compose_entry(SHADOW_WGSL, "shadow_entry.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shadow_entry"),
        source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("shadow_pipeline"),
        layout: None, // auto layout — pulls exactly the bindings soft_shadow uses
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let atlas_view = atlas_tex.create_view(&Default::default());
    let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg0"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: points_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: vis_buf.as_entire_binding(),
            },
        ],
    });
    let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg1"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&atlas_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: chunk_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: tile_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg0, &[]);
        pass.set_bind_group(1, &bg1, &[]);
        pass.dispatch_workgroups(points.len() as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&vis_buf, 0, &readback, 0, vis_size);
    queue.submit([enc.finish()]);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let vis: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    readback.unmap();

    // --- 8. analytic reference (CPU) --------------------------------------------------------
    let sun = Vec3::new(0.3, 1.0, 0.2).normalize();
    let sun_angular_radius = 0.5f32.to_radians(); // ~the real sun (~0.27° radius); tunable.
    let analytic: Vec<f32> = world_xy
        .iter()
        .map(|&(x, z)| {
            analytic_sun_visibility(
                Vec3::new(x, 0.0, z),
                sun,
                sphere_center,
                sphere_radius,
                sun_angular_radius,
            )
        })
        .collect();

    // --- 9. ASCII dump ----------------------------------------------------------------------
    println!("\n=== GPU soft_shadow sun-visibility ({GRID}x{GRID}, x/z in [{lo},{hi}]) ===");
    println!("(sun dir = {sun:?}; sphere C={sphere_center:?} r={sphere_radius})");
    print_grid(&vis, GRID);
    println!("\n=== analytic reference (sun-disk vs sphere occlusion) ===");
    print_grid(&analytic, GRID);

    // --- 10. metrics ------------------------------------------------------------------------
    let (rmse, max_err) = rmse_max(&vis, &analytic);
    let boxiness = ring_variance_boxiness(&vis, &world_xy, shadow_center(sphere_center, sun));
    let max_grad = max_gradient(&vis, GRID, (hi - lo) / (GRID - 1) as f32);
    println!("\n=== metrics (diagnostic) ===");
    println!("RMSE(gpu vs analytic)      = {rmse:.4}");
    println!("max abs error              = {max_err:.4}");
    println!("boxiness (ring vis variance, lower=rounder) = {boxiness:.5}");
    println!("max |grad vis| (hard-edge indicator)        = {max_grad:.4}");

    // --- 11. loose structural asserts -------------------------------------------------------
    // The shadow blob sits near the projected sphere centre; sample its darkest few cells.
    let sc = shadow_center(sphere_center, sun);
    let mut min_vis_near = 1.0f32;
    for (&v, &(x, z)) in vis.iter().zip(&world_xy) {
        let d = ((x - sc.x).powi(2) + (z - sc.z).powi(2)).sqrt();
        if d < 0.3 {
            min_vis_near = min_vis_near.min(v);
        }
    }
    assert!(
        min_vis_near < 0.9,
        "expected a shadow (vis<0.9) under the sphere; darkest near-centre vis={min_vis_near}"
    );

    // Far corners (away from the shadow) must be lit.
    let corners = [0usize, GRID - 1, GRID * (GRID - 1), GRID * GRID - 1];
    for &c in &corners {
        assert!(
            vis[c] > 0.9,
            "far corner {c} should be lit (vis>0.9) but vis={}",
            vis[c]
        );
    }

    println!("\nharness OK: shadow present (min near-centre vis {min_vis_near:.3}), corners lit.");
}

// --- ASCII grid dump -------------------------------------------------------------------------
fn print_grid(v: &[f32], grid: usize) {
    const RAMP: &[u8] = b" .:-=+*#%@"; // 0 (lit) -> 9 (occluded)
    for iz in 0..grid {
        let mut line = String::with_capacity(grid);
        for ix in 0..grid {
            let val = v[iz * grid + ix].clamp(0.0, 1.0);
            // vis=1 (lit) -> ' ', vis=0 (shadow) -> '@'
            let occ = 1.0 - val;
            let idx = ((occ * (RAMP.len() - 1) as f32).round() as usize).min(RAMP.len() - 1);
            line.push(RAMP[idx] as char);
        }
        println!("{line}");
    }
}

// --- analytic sun-disk vs sphere occlusion ---------------------------------------------------
//
// Model the sun as a disk of angular radius `sun_r` centred on direction `sun` (a cone of
// half-angle `sun_r`). The sphere (centre `c`, radius `r`) subtends, as seen from `origin`
// along the sun direction, a cone whose axis points at `c` and whose half-angle is
// `asin(r / |c - origin|)` (the sphere's angular radius). Visibility = fraction of the sun disk
// NOT covered by the sphere disk, approximated by the standard two-disk overlap reduced to the
// 1-D angular-separation smoothstep used for penumbrae:
//
//   sep    = angle between the sun axis and the (origin -> c) axis
//   occ_r  = sphere angular radius
//   The sphere fully covers the sun when sep + sun_r <= occ_r (umbra);
//   it misses entirely when sep >= occ_r + sun_r (full light);
//   between, the covered fraction ramps smoothly (penumbra). A smoothstep over
//   sep ∈ [occ_r - sun_r, occ_r + sun_r] is the soft transition a real area light produces.
//
// Only occluders IN FRONT of the point along the sun direction count (the sphere must be on the
// sunward side), matching a shadow ray.
fn analytic_sun_visibility(
    origin: Vec3,
    sun: Vec3,
    c: Vec3,
    r: f32,
    sun_r: f32,
) -> f32 {
    let to_c = c - origin;
    let dist = to_c.length();
    if dist <= r {
        return 0.0; // inside the sphere — fully shadowed
    }
    let dir_c = to_c / dist;
    // Sphere must be on the sunward side: its axis within 90° of the sun direction.
    let cos_sep = sun.dot(dir_c).clamp(-1.0, 1.0);
    if cos_sep <= 0.0 {
        return 1.0; // sphere is behind the point relative to the sun
    }
    let sep = cos_sep.acos(); // angular separation of sun axis vs sphere axis
    let occ_r = (r / dist).clamp(-1.0, 1.0).asin(); // sphere angular radius

    let lo = occ_r - sun_r; // fully covered for sep <= lo
    let hi = occ_r + sun_r; // fully clear for sep >= hi
    if sep <= lo {
        return 0.0;
    }
    if sep >= hi {
        return 1.0;
    }
    // smoothstep covered-fraction → visibility = 1 - covered.
    let t = (sep - lo) / (hi - lo); // 0 at umbra edge, 1 at light edge
    let covered = 1.0 - t * t * (3.0 - 2.0 * t); // smoothstep(0,1,t) reversed → covered fraction
    1.0 - covered
}

/// Where the sphere's shadow lands on the y=0 plane: project the sphere centre along the sun
/// direction down to y=0. (Used to centre the boxiness rings + the structural shadow probe.)
fn shadow_center(c: Vec3, sun: Vec3) -> Vec3 {
    if sun.y.abs() < 1e-4 {
        return Vec3::new(c.x, 0.0, c.z);
    }
    let t = c.y / sun.y; // c - sun*t hits y=0
    c - sun * t
}

// --- metrics ---------------------------------------------------------------------------------
fn rmse_max(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut se = 0.0f32;
    let mut max = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let e = (x - y).abs();
        se += e * e;
        max = max.max(e);
    }
    ((se / a.len() as f32).sqrt(), max)
}

/// Boxiness via iso-radius ring variance: sample `vis` in annular rings around the shadow
/// centre; a smooth ROUND shadow is near-constant around each ring (low variance), a BOXY /
/// faceted one varies a lot azimuthally. Returns the mean per-ring variance over the penumbra
/// band of radii. Higher = boxier.
fn ring_variance_boxiness(vis: &[f32], world_xy: &[(f32, f32)], center: Vec3) -> f32 {
    // Bucket points by radius into rings of width 0.1, over radii [0.1, 1.2) (the shadow's
    // penumbra band). Skip the innermost (umbra, ~flat) and far (lit, ~flat) rings implicitly
    // since their variance is ~0 and just dilutes — but include them; they contribute little.
    let ring_w = 0.1f32;
    let n_rings = 12usize;
    let mut sums = vec![0.0f64; n_rings];
    let mut sqs = vec![0.0f64; n_rings];
    let mut cnt = vec![0u32; n_rings];
    for (&v, &(x, z)) in vis.iter().zip(world_xy) {
        let rad = ((x - center.x).powi(2) + (z - center.z).powi(2)).sqrt();
        let ring = (rad / ring_w) as usize;
        if ring < n_rings {
            sums[ring] += v as f64;
            sqs[ring] += (v * v) as f64;
            cnt[ring] += 1;
        }
    }
    let mut total = 0.0f64;
    let mut used = 0u32;
    for i in 0..n_rings {
        if cnt[i] >= 6 {
            let n = cnt[i] as f64;
            let mean = sums[i] / n;
            let var = (sqs[i] / n - mean * mean).max(0.0);
            total += var;
            used += 1;
        }
    }
    if used == 0 {
        0.0
    } else {
        (total / used as f64) as f32
    }
}

/// Max gradient magnitude of `vis` across the grid (central differences) — a hard penumbra→umbra
/// edge shows as a large spike. `cell` is the world spacing between adjacent grid points.
fn max_gradient(vis: &[f32], grid: usize, cell: f32) -> f32 {
    let at = |x: usize, z: usize| vis[z * grid + x];
    let mut max = 0.0f32;
    for z in 1..grid - 1 {
        for x in 1..grid - 1 {
            let gx = (at(x + 1, z) - at(x - 1, z)) / (2.0 * cell);
            let gz = (at(x, z + 1) - at(x, z - 1)) / (2.0 * cell);
            max = max.max((gx * gx + gz * gz).sqrt());
        }
    }
    max
}

// --- bake bind-group helpers (mirror sdf_bake_gpu.rs) ----------------------------------------
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}
