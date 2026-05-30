//! Real-GPU compute harness for the SDF brick/coord logic.
//!
//! Unlike `shader_validation.rs` (which only *composes + validates*), this runs the
//! ACTUAL composed WGSL (`sdf/bindings.wgsl` + `sdf/brick.wgsl`) on a headless wgpu
//! device and reads back result buffers. So `world_to_brick_lod` / `brick_snap` /
//! `euclid_mod` are exercised exactly as the GPU runs them — catching GPU-specific
//! float/int divergence the CPU can't reproduce.
//!
//! The test under investigation: for a cube at NEGATIVE world coords, does the GPU's
//! `world_to_brick_lod(center)` produce the same brick coord the CPU bake stored as a
//! key? If the GPU coord differs (negative-coord floor/int divergence), the flat
//! lookup misses and nothing renders — the reported "only renders for positive
//! translations" bug.

use std::borrow::Cow;

use bevy::math::{IVec3, Vec3};
use futures_lite::future::block_on;
use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderLanguage,
};

use adventure::sdf_render::atlas::SdfAtlas;
use adventure::sdf_render::bvh::Bvh;
use adventure::sdf_render::edits::{edit_world_aabb, ResolvedEdit, SdfOp, SdfPrimitive};
use adventure::sdf_render::SdfGridConfig;

// --- GPU device ----------------------------------------------------------------

fn device_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .ok()?;
    let info = adapter.get_info();
    eprintln!(
        "GPU adapter: name={:?} backend={:?} driver={:?} driver_info={:?} device_type={:?}",
        info.name, info.backend, info.driver, info.driver_info, info.device_type
    );
    // R16Snorm (the brick atlas + volume distance format) needs TEXTURE_FORMAT_16BIT_NORM;
    // request it if the adapter supports it so the rig can create those textures.
    let wanted = wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
    let features = adapter.features() & wanted;
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("sdf_gpu_rig"),
        required_features: features,
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

// --- SdfCameraUniform mirror (240 bytes) ---------------------------------------
// Layout MUST match bindings.wgsl::SdfCameraUniform: 2× mat4x4 then 7× vec4.
// We only need lod_params.z = voxel_size and lod_params.w = cell_stride filled.
fn camera_uniform_bytes(config: &SdfGridConfig) -> Vec<u8> {
    let mut f = [0.0f32; 60]; // 240 bytes
    // lod_params is the 9th field: 2 mats (32 floats) + 6 vec4 (24 floats) = 56.
    // lod_params = [lod_count, ring_bricks, base_voxel_size, cell_stride].
    f[56] = config.lod_count as f32;
    f[57] = config.ring_bricks as f32;
    f[58] = config.voxel_size; // lod_params.z
    f[59] = config.cell_stride() as f32; // lod_params.w
    bytemuck::cast_slice(&f).to_vec()
}

// One ChunkLookup entry (20 bytes, 5× u32) — repurposed as a flat brick entry:
// key_hi=coord.x, key_lo=coord.y, occ_lo=coord.z (i32 bitcast), occ_hi=atlas_base.
fn flat_brick_bytes(coords: &[IVec3]) -> Vec<u8> {
    let mut out = Vec::with_capacity(coords.len() * 20);
    for c in coords {
        out.extend_from_slice(&(c.x as u32).to_le_bytes());
        out.extend_from_slice(&(c.y as u32).to_le_bytes());
        out.extend_from_slice(&(c.z as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // atlas_base (unused here)
        out.extend_from_slice(&0u32.to_le_bytes()); // tile_run_base
    }
    out
}

/// The compute entry — composes the REAL sdf modules and calls the REAL
/// `world_to_brick_lod`, then scans the flat brick list the same way the min shader
/// does. Writes back the GPU-computed coord + whether it matched a stored brick.
const PROBE_WGSL: &str = r#"
#import sdf::bindings::{camera, chunk_buf, euclid_mod, cell_stride, voxel_size_at}
#import sdf::brick::{world_to_brick_lod, brick_snap}

struct Probe { x: f32, y: f32, z: f32, pad: f32 };
// cx,cy,cz = GPU world_to_brick_lod coord. dbg_* = step-by-step on the X axis:
//   dbg_vox = floor(x/vs), dbg_mod = euclid_mod(dbg_vox, stride), dbg_snap = brick_snap(...)
struct Res { cx: i32, cy: i32, cz: i32, found: i32, dbg_vox: i32, dbg_mod: i32, dbg_snap: i32, pad: i32 };

@group(0) @binding(1) var<storage, read> probes: array<Probe>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let p = vec3<f32>(probes[i].x, probes[i].y, probes[i].z);
    let coord = world_to_brick_lod(p, 0u);
    var found = 0i;
    let n = arrayLength(&chunk_buf);
    for (var k = 0u; k < n; k = k + 1u) {
        let e = chunk_buf[k];
        if (bitcast<i32>(e.key_hi) == coord.x
            && bitcast<i32>(e.key_lo) == coord.y
            && bitcast<i32>(e.occ_lo) == coord.z) {
            found = 1;
            break;
        }
    }
    let s = cell_stride();
    let vs = voxel_size_at(0u);
    let vox = i32(floor(p.x / vs));
    let raw_rem = vox % s;          // dbg_mod: raw WGSL `%`
    let snap = brick_snap(vox, s);
    results[i] = Res(coord.x, coord.y, coord.z, found, vox, raw_rem, snap, s);
}
"#;

// Probes the REAL abs_chunk_key + local_brick_index (from bindings.wgsl) for given brick
// coords. `camera` comes from the bindings import (binding 0); the chunk math reads only
// cell_stride() from it. Coords in @ binding(1), key/local results out @ binding(2).
const CHUNK_KEY_PROBE_WGSL: &str = r#"
#import sdf::bindings::{camera, abs_chunk_key, local_brick_index, floor_div, euclid_mod, cell_stride}

struct CoordIn { x: i32, y: i32, z: i32, lod: u32 };
// key_hi/lo = abs_chunk_key; local_idx = local_brick_index; the rest decompose the y axis:
//   fd_y = floor_div(coord.y, stride); em_y = euclid_mod(fd_y, 4)  (the `ly` term)
struct KeyOut { key_hi: u32, key_lo: u32, local_idx: u32, fd_y: i32, em_y: i32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(1) var<storage, read> coords: array<CoordIn>;
@group(0) @binding(2) var<storage, read_write> keys: array<KeyOut>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let c = vec3<i32>(coords[i].x, coords[i].y, coords[i].z);
    let k = abs_chunk_key(c, coords[i].lod);
    let li = local_brick_index(c);
    let s = cell_stride();
    // decompose the Z axis (the divergent one): fd_z = floor_div(z,7), em_z = euclid_mod(fd_z,4).
    let fd_z = floor_div(c.z, s);
    let em_z = euclid_mod(fd_z, 4);
    keys[i] = KeyOut(k.x, k.y, li, fd_z, em_z, 0u, 0u, 0u);
}
"#;

// Probes the FULL find_brick_lookup (chunk key -> find_chunk -> brick_in_chunk occupancy
// + popcount -> tile_run[base+off].atlas_base) on the GPU. Uses real chunk_buf (g1 b2) +
// chunk_tile_buf (g1 b11). Returns the resolved atlas_base + found flag per brick coord,
// to compare against the CPU shader_resolve. Isolates brick_in_chunk (the only chunk-path
// piece not yet GPU-verified).
const FULL_LOOKUP_PROBE_WGSL: &str = r#"
#import sdf::bindings::{camera, chunk_buf, chunk_tile_buf, local_brick_index, abs_chunk_key}
#import sdf::brick::{find_brick_lookup, find_chunk}

struct CoordIn { x: i32, y: i32, z: i32, lod: u32 };
struct LookupOut { atlas_base: u32, found: u32, local_idx: u32, ci: u32 };
@group(0) @binding(1) var<storage, read> coords: array<CoordIn>;
@group(0) @binding(2) var<storage, read_write> outs: array<LookupOut>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let c = vec3<i32>(coords[i].x, coords[i].y, coords[i].z);
    let loc = find_brick_lookup(c, coords[i].lod);
    let li = local_brick_index(c);
    let key = abs_chunk_key(c, coords[i].lod);
    let ci = find_chunk(key.x, key.y);
    outs[i] = LookupOut(loc.atlas_base, select(0u, 1u, loc.found), li, bitcast<u32>(ci));
}
"#;

const SDF_MODULES: [&str; 2] = [
    "assets/shaders/sdf/bindings.wgsl",
    "assets/shaders/sdf/brick.wgsl",
];

fn compose_probe() -> naga::Module {
    compose_entry(PROBE_WGSL, "probe.wgsl")
}

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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ProbeIn {
    x: f32,
    y: f32,
    z: f32,
    pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ResOut {
    cx: i32,
    cy: i32,
    cz: i32,
    found: i32,
    dbg_vox: i32,
    dbg_mod: i32,
    dbg_snap: i32,
    pad: i32,
}

/// Run the probe shader: for each world point, return GPU coord + found flag.
fn run_probe(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &SdfGridConfig,
    brick_coords: &[IVec3],
    points: &[Vec3],
) -> Vec<ResOut> {
    use wgpu::util::DeviceExt;

    let module = compose_probe();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("probe"),
        source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
    });

    let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera"),
        contents: &camera_uniform_bytes(config),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bricks_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chunk_buf(flat bricks)"),
        contents: &flat_brick_bytes(brick_coords),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let probe_in: Vec<ProbeIn> = points
        .iter()
        .map(|p| ProbeIn { x: p.x, y: p.y, z: p.z, pad: 0.0 })
        .collect();
    let probes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("probes"),
        contents: bytemuck::cast_slice(&probe_in),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let result_size = (points.len() * std::mem::size_of::<ResOut>()) as u64;
    let results_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("results"),
        size: result_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: result_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("probe_pipeline"),
        layout: None, // auto layout — only the bindings the entry actually uses
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Group 0: camera(0), probes(1), results(2). Group 1: chunk_buf(2).
    let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg0"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: probes_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: results_buf.as_entire_binding() },
        ],
    });
    let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg1"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[wgpu::BindGroupEntry { binding: 2, resource: bricks_buf.as_entire_binding() }],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg0, &[]);
        pass.set_bind_group(1, &bg1, &[]);
        pass.dispatch_workgroups(points.len() as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&results_buf, 0, &readback, 0, result_size);
    queue.submit([enc.finish()]);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let data = slice.get_mapped_range();
    let out: Vec<ResOut> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback.unmap();
    out
}

fn single_cube(center: Vec3) -> (Vec<ResolvedEdit>, Bvh) {
    let edits = vec![ResolvedEdit {
        prim: SdfPrimitive::Box { half_extents: Vec3::splat(1.0) },
        transform: bevy::prelude::Transform::from_translation(center),
        op: SdfOp::default(),
        material_id: 0,
    }];
    let aabbs: Vec<_> = edits
        .iter()
        .map(|e| edit_world_aabb(&e.prim, &e.transform, e.op.smoothing))
        .collect();
    let bvh = Bvh::build(&aabbs);
    (edits, bvh)
}

/// THE decisive test: bake a cube at a negative position, take the REAL brick keys the
/// CPU stored, and ask the GPU `world_to_brick_lod` whether the cube center resolves to
/// one of them. The GPU coord is read back and compared against the CPU coord.
#[test]
fn gpu_world_to_brick_matches_cpu_for_negative_cube() {
    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter — skipping");
        return;
    };

    let config = SdfGridConfig { lod_count: 1, ring_bricks: 64, ..Default::default() };

    // The reported negative repro, plus positive + origin controls.
    let centers = [
        Vec3::new(-10.822, -0.339, -5.058),
        Vec3::new(10.822, 0.339, 5.058),
        Vec3::ZERO,
        Vec3::new(-3.0, -3.0, -3.0),
    ];

    for center in centers {
        let (edits, bvh) = single_cube(center);
        let mut atlas = SdfAtlas::default();
        atlas.full_bake(&edits, &bvh, &config, center);
        let brick_coords: Vec<IVec3> = atlas.bricks.keys().map(|k| k.coord).collect();
        assert!(!brick_coords.is_empty(), "cube at {center:?} baked no bricks");

        let cpu_coord = config.world_to_brick_lod(center, 0);
        let cpu_has = brick_coords.contains(&cpu_coord);

        let res = run_probe(&device, &queue, &config, &brick_coords, &[center]);
        let r = res[0];
        let gpu_coord = IVec3::new(r.cx, r.cy, r.cz);

        let cpu_vox = (center.x / config.voxel_size).floor() as i32;
        let s = config.cell_stride();
        let cpu_raw_rem = cpu_vox % s; // Rust truncated `%`
        let cpu_snap = cpu_vox - cpu_vox.rem_euclid(s);
        println!(
            "center={center:?}\n  CPU coord={cpu_coord:?} (in baked set: {cpu_has})\n  GPU coord={gpu_coord:?} found={}\n  X: CPU vox={cpu_vox} raw_rem(%)={cpu_raw_rem} snap={cpu_snap}  |  GPU vox={} raw_rem(%)={} snap={} stride={}",
            r.found, r.dbg_vox, r.dbg_mod, r.dbg_snap, r.pad
        );

        if gpu_coord != cpu_coord {
            println!("  >>> DIVERGENCE at {center:?}: gpu {gpu_coord:?} != cpu {cpu_coord:?}");
        }
    }
}

// =====================================================================================
// OP SWEEP across i32, u32, AND f32 native ops. For each input value the shader reports
// every op and we diff against Rust. Catalogues exactly which scalar ops this GPU gets
// wrong (the i32 `%`/`/` divergence was found here; this widens it to float + unsigned).
// Float results are compared as raw bits so we catch sign-of-zero / NaN / rounding too.
// =====================================================================================

const OP_SWEEP_WGSL: &str = r#"
struct In { a: i32, b: i32, fa: f32, fb: f32 };
struct Out {
    // signed integer
    i_rem: i32,    // a % b
    i_div: i32,    // a / b
    // unsigned integer (reinterpret a,b as u32)
    u_rem: u32,    // u32(a) % u32(b)   (only meaningful when both >=0)
    u_div: u32,    // u32(a) / u32(b)
    // float
    f_div: f32,    // fa / fb
    f_floor: f32,  // floor(fa / fb)
    f_fract: f32,  // fract(fa)
    f_rem: f32,    // fa % fb           (WGSL float modulo)
    f_trunc_i: i32,// i32(fa)           (truncating cast of negative float)
    f_floor_i: i32,// i32(floor(fa))
    // signedness-isolation: same `% 7` on the SAME value reached three ways.
    rem_lit: i32,     // (-109) % 7  — compile-time literal
    rem_computed: i32,// (0 - 109) % 7 — value computed on-GPU at runtime
    rem_neg_a: i32,   // (-a_pos) % 7 where a_pos = abs(a), built on GPU
    pad1: i32,
};
@group(0) @binding(0) var<storage, read> ins: array<In>;
@group(0) @binding(1) var<storage, read_write> outs: array<Out>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let a = ins[i].a;
    let b = ins[i].b;
    let fa = ins[i].fa;
    let fb = ins[i].fb;
    let a_neg = 0 - abs(a);          // a forced negative, computed on-GPU
    outs[i] = Out(
        a % b,
        a / b,
        u32(max(a,0)) % u32(max(b,1)),
        u32(max(a,0)) / u32(max(b,1)),
        fa / fb,
        floor(fa / fb),
        fract(fa),
        fa % fb,
        i32(fa),
        i32(floor(fa)),
        (-109) % 7,                  // rem_lit: compile-time literal
        (0 - 109) % 7,               // rem_computed: runtime-computed negative
        a_neg % 7,                   // rem_neg_a
        0,
    );
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OpIn { a: i32, b: i32, fa: f32, fb: f32 }

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct OpOut {
    i_rem: i32,
    i_div: i32,
    u_rem: u32,
    u_div: u32,
    f_div: f32,
    f_floor: f32,
    f_fract: f32,
    f_rem: f32,
    f_trunc_i: i32,
    f_floor_i: i32,
    rem_lit: i32,
    rem_computed: i32,
    rem_neg_a: i32,
    p1: i32,
}

#[test]
fn gpu_scalar_ops_vs_rust() {
    use wgpu::util::DeviceExt;
    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter — skipping");
        return;
    };

    let module = {
        let mut composer = Composer::default();
        composer
            .make_naga_module(NagaModuleDescriptor {
                source: OP_SWEEP_WGSL,
                file_path: "op_sweep.wgsl",
                ..Default::default()
            })
            .unwrap_or_else(|e| panic!("compose op_sweep: {e}"))
    };
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("op_sweep"),
        source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
    });

    // (a, b, fa, fb): integers for the int/uint ops, floats for the float ops. We pair
    // the integer `a` with a float `fa = a*0.1` (the real world_to_brick case) and a
    // couple of awkward fractional values.
    let cases: Vec<(i32, i32, f32, f32)> = vec![
        (-109, 7, -10.9, 7.0),
        (-108, 7, -10.8, 7.0),
        (-30, 7, -3.0, 7.0),
        (-7, 7, -0.7, 7.0),
        (-1, 7, -0.1, 7.0),
        (0, 7, 0.0, 7.0),
        (1, 7, 0.1, 7.0),
        (105, 7, 10.5, 7.0),
        (108, 7, 10.8, 7.0),
        (-5, 7, -10.822, 0.1), // the real ratio: floor(-10.822/0.1) must be -109
        (-50, 7, -5.058, 0.1),
        (3, 7, -0.339, 0.1),
    ];
    let in_data: Vec<OpIn> = cases.iter().map(|&(a, b, fa, fb)| OpIn { a, b, fa, fb }).collect();
    let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ins"),
        contents: bytemuck::cast_slice(&in_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_size = (cases.len() * std::mem::size_of::<OpOut>()) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("outs"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: out_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("op_sweep_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: in_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(cases.len() as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, out_size);
    queue.submit([enc.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let data = slice.get_mapped_range();
    let outs: Vec<OpOut> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback.unmap();

    let mut bad: Vec<String> = Vec::new();
    for ((a, b, fa, fb), o) in cases.iter().zip(&outs) {
        let mut diffs = Vec::new();
        // signed int
        if o.i_rem != a % b { diffs.push(format!("i32 %: gpu={} rust={}", o.i_rem, a % b)); }
        if o.i_div != a / b { diffs.push(format!("i32 /: gpu={} rust={}", o.i_div, a / b)); }
        // unsigned (only when inputs non-negative)
        if *a >= 0 && *b > 0 {
            let (ua, ub) = (*a as u32, *b as u32);
            if o.u_rem != ua % ub { diffs.push(format!("u32 %: gpu={} rust={}", o.u_rem, ua % ub)); }
            if o.u_div != ua / ub { diffs.push(format!("u32 /: gpu={} rust={}", o.u_div, ua / ub)); }
        }
        // float — compare bit-exact
        let r_fdiv = fa / fb;
        let r_ffloor = (fa / fb).floor();
        let r_ffract = fa - fa.floor();
        let r_frem = fa % fb;
        let r_ftrunc = *fa as i32;
        let r_ffloor_i = fa.floor() as i32;
        if o.f_div.to_bits() != r_fdiv.to_bits() { diffs.push(format!("f32 /: gpu={} rust={}", o.f_div, r_fdiv)); }
        if o.f_floor.to_bits() != r_ffloor.to_bits() { diffs.push(format!("floor: gpu={} rust={}", o.f_floor, r_ffloor)); }
        if (o.f_fract - r_ffract).abs() > 1e-6 { diffs.push(format!("fract: gpu={} rust={}", o.f_fract, r_ffract)); }
        if (o.f_rem - r_frem).abs() > 1e-6 { diffs.push(format!("f32 %: gpu={} rust={}", o.f_rem, r_frem)); }
        if o.f_trunc_i != r_ftrunc { diffs.push(format!("i32(f): gpu={} rust={}", o.f_trunc_i, r_ftrunc)); }
        if o.f_floor_i != r_ffloor_i { diffs.push(format!("i32(floor): gpu={} rust={}", o.f_floor_i, r_ffloor_i)); }

        // signedness-isolation: every form should give -4 (correct) if `%` is signed.
        // If a form gives 0 it computed u32(bits)%7 — the value reached `%` as unsigned.
        let u_form = (*a as u32 % 7) as i32;
        println!(
            "a={a} b={b} fa={fa} fb={fb}\n  i%={} i/={} u%={} u/={} | f/={} ffloor={} fract={} f%={} i32(f)={} i32(floor)={}\n  signedness[a%7]: buffer_a={} (u32(a)%7={u_form})  literal(-109%7)={} computed(0-109%7)={} negA={}{}",
            o.i_rem, o.i_div, o.u_rem, o.u_div,
            o.f_div, o.f_floor, o.f_fract, o.f_rem, o.f_trunc_i, o.f_floor_i,
            o.i_rem, o.rem_lit, o.rem_computed, o.rem_neg_a,
            if diffs.is_empty() { String::new() } else { format!("\n  <-- {}", diffs.join("; ")) }
        );
        if !diffs.is_empty() {
            bad.push(format!("a={a},b={b},fa={fa},fb={fb}: {}", diffs.join("; ")));
        }
    }
    println!("\n===== BROKEN OPS ON THIS GPU =====");
    if bad.is_empty() {
        println!("(only the ops already known broken — none beyond what's listed inline)");
    }
    for line in &bad {
        println!("{line}");
    }
    // Diagnostic catalogue, not a gate.
}



// =====================================================================================
// CHUNK-KEY PROBE: run the REAL abs_chunk_key + local_brick_index on the GPU for given
// brick coords and compare against CPU chunk_of + chunk_gpu_key. Isolates whether the
// chunk-addressing math (not world_to_brick_lod) diverges for negative coords — the
// suspect after the offset bug returned when chunking was re-enabled.
// =====================================================================================

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CoordIn { x: i32, y: i32, z: i32, lod: u32 }

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct KeyOut { key_hi: u32, key_lo: u32, local_idx: u32, fd_y: i32, em_y: i32, p0: u32, p1: u32, p2: u32 }

#[test]
fn gpu_chunk_key_matches_cpu() {
    use wgpu::util::DeviceExt;
    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter — skipping");
        return;
    };
    let config = SdfGridConfig { lod_count: 1, ring_bricks: 64, ..Default::default() };

    // Brick coords to probe — the negative cube's coord plus positive/zero controls.
    // These are stride-aligned (multiples of cell_stride=7) like real brick keys.
    let coords = [
        IVec3::new(-112, -7, -56),  // the failing cube
        IVec3::new(-35, -35, -35),
        IVec3::new(105, 0, 49),     // positive control (worked)
        IVec3::new(0, 0, 0),
        IVec3::new(-7, 0, 0),
        IVec3::new(-28, -28, -28),
        // The y values that diverged in the full-lookup probe (li off by 16 = ly off by 4):
        IVec3::new(-112, -14, -49),
        IVec3::new(-105, 0, -49),
        IVec3::new(-119, -7, -49),
    ];

    let module = compose_entry(CHUNK_KEY_PROBE_WGSL, "chunk_key_probe.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chunk_key_probe"),
        source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
    });

    let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera"),
        contents: &camera_uniform_bytes(&config),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let in_data: Vec<CoordIn> = coords.iter().map(|c| CoordIn { x: c.x, y: c.y, z: c.z, lod: 0 }).collect();
    let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("coords"),
        contents: bytemuck::cast_slice(&in_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_size = (coords.len() * std::mem::size_of::<KeyOut>()) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("keys"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: out_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chunk_key_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: in_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(coords.len() as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, out_size);
    queue.submit([enc.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let data = slice.get_mapped_range();
    let outs: Vec<KeyOut> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback.unmap();

    use adventure::sdf_render::atlas::BrickKey;
    use adventure::sdf_render::chunk::{chunk_of, chunk_gpu_key};
    let mut bad = Vec::new();
    for (c, o) in coords.iter().zip(&outs) {
        let (ck, li) = chunk_of(BrickKey::new(0, *c), &config);
        let (cpu_hi, cpu_lo) = chunk_gpu_key(ck);
        let key_ok = o.key_hi == cpu_hi && o.key_lo == cpu_lo;
        let li_ok = o.local_idx == li;
        let s = config.cell_stride();
        let cpu_fd_z = c.z.div_euclid(s);
        let cpu_em_z = cpu_fd_z.rem_euclid(4);
        println!(
            "coord={c:?}\n  CPU key=({cpu_hi:#x},{cpu_lo:#x}) local={li} fd_z={cpu_fd_z} em_z={cpu_em_z}\n  GPU key=({:#x},{:#x}) local={} fd_z={} em_z={}{}",
            o.key_hi, o.key_lo, o.local_idx, o.fd_y, o.em_y,
            if key_ok && li_ok { "" } else { "  <-- DIVERGES" }
        );
        if !key_ok { bad.push(format!("coord={c:?}: key gpu=({:#x},{:#x}) cpu=({cpu_hi:#x},{cpu_lo:#x})", o.key_hi, o.key_lo)); }
        if !li_ok { bad.push(format!("coord={c:?}: local gpu={} cpu={li}", o.local_idx)); }
    }
    assert!(bad.is_empty(), "GPU chunk math diverged:\n{}", bad.join("\n"));
}

// =====================================================================================
// FULL LOOKUP PROBE: run find_brick_lookup on the GPU with REAL chunk tables and compare
// the resolved atlas_base to a CPU reference resolve. The only chunk-path piece not yet
// GPU-verified (brick_in_chunk: occupancy bit + popcount -> tile_run index).
// =====================================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LookupOut { atlas_base: u32, found: u32, local_idx: u32, ci: i32 }

fn chunk_lookup_bytes(chunks: &[adventure::sdf_render::chunk::ChunkLookup]) -> Vec<u8> {
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

fn brick_tile_bytes(tiles: &[adventure::sdf_render::chunk::BrickTile]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tiles.len() * 12);
    for t in tiles {
        out.extend_from_slice(&t.atlas_base.to_le_bytes());
        out.extend_from_slice(&t.pal01.to_le_bytes());
        out.extend_from_slice(&t.pal23.to_le_bytes());
    }
    out
}

#[test]
fn gpu_find_brick_lookup_matches_cpu() {
    use wgpu::util::DeviceExt;
    use adventure::sdf_render::atlas::BrickKey;
    use adventure::sdf_render::chunk::{build_chunk_tables, chunk_of, chunk_gpu_key, BrickTile};

    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter — skipping");
        return;
    };
    let config = SdfGridConfig { lod_count: 1, ring_bricks: 64, ..Default::default() };

    let center = Vec3::new(-10.822, -0.339, -5.058);
    let (edits, bvh) = single_cube(center);
    let mut atlas = SdfAtlas::default();
    atlas.full_bake(&edits, &bvh, &config, center);

    // Deterministic atlas_base per brick so a wrong-tile resolve is detectable.
    let tables = build_chunk_tables(&atlas, &config, |key| {
        let base = ((key.coord.x as u32 & 0xff) << 16)
            | ((key.coord.y as u32 & 0xff) << 8)
            | (key.coord.z as u32 & 0xff);
        BrickTile { atlas_base: base, pal01: 0, pal23: 0 }
    });

    let brick_coords: Vec<IVec3> = atlas.bricks.keys().map(|k| k.coord).collect();
    let in_data: Vec<CoordIn> =
        brick_coords.iter().map(|c| CoordIn { x: c.x, y: c.y, z: c.z, lod: 0 }).collect();

    let module = compose_entry(FULL_LOOKUP_PROBE_WGSL, "full_lookup_probe.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("full_lookup"),
        source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
    });

    // grid_dims.w = resident chunk count (find_chunk's search bound). Layout: 2 mats (32
    // floats) + camera_pos(4) + screen_params(4) + grid_origin(4) = float 44; grid_dims.w
    // = float 47 = byte 188.
    let mut cam = camera_uniform_bytes(&config);
    let n = tables.chunks.len() as f32;
    cam[188..192].copy_from_slice(&n.to_le_bytes());

    let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera"), contents: &cam, usage: wgpu::BufferUsages::UNIFORM,
    });
    let coords_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("coords"), contents: bytemuck::cast_slice(&in_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let chunk_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chunk_buf"), contents: &chunk_lookup_bytes(&tables.chunks),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let tile_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chunk_tile_buf"), contents: &brick_tile_bytes(&tables.tile_run),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_size = (in_data.len() * std::mem::size_of::<LookupOut>()) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("outs"), size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"), size: out_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("full_lookup_pipeline"), layout: None, module: &shader,
        entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    });
    let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg0"), layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: coords_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
        ],
    });
    let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg1"), layout: &pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry { binding: 2, resource: chunk_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: tile_buf.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg0, &[]);
        pass.set_bind_group(1, &bg1, &[]);
        pass.dispatch_workgroups(in_data.len() as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, out_size);
    queue.submit([enc.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let data = slice.get_mapped_range();
    let outs: Vec<LookupOut> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback.unmap();

    let cpu_resolve = |coord: IVec3| -> Option<u32> {
        let (ck, li) = chunk_of(BrickKey::new(0, coord), &config);
        let (kh, kl) = chunk_gpu_key(ck);
        let idx = tables.chunks.binary_search_by(|c| (c.key_hi, c.key_lo).cmp(&(kh, kl))).ok()?;
        let chunk = tables.chunks[idx];
        let occ = (chunk.occ_lo as u64) | ((chunk.occ_hi as u64) << 32);
        if (occ >> li) & 1 == 0 { return None; }
        let off = (occ & ((1u64 << li) - 1)).count_ones();
        Some(tables.tile_run[(chunk.tile_run_base + off) as usize].atlas_base)
    };

    let mut bad = Vec::new();
    for (c, o) in brick_coords.iter().zip(&outs) {
        let cpu = cpu_resolve(*c);
        let gpu = if o.found == 1 { Some(o.atlas_base) } else { None };
        let (ck, cpu_li) = chunk_of(BrickKey::new(0, *c), &config);
        let (kh, kl) = chunk_gpu_key(ck);
        let cpu_ci = tables.chunks.binary_search_by(|x| (x.key_hi, x.key_lo).cmp(&(kh, kl)))
            .map(|i| i as i32).unwrap_or(-1);
        if cpu != gpu {
            bad.push(format!(
                "coord={c:?}: GPU base={gpu:?} li={} ci={} | CPU base={cpu:?} li={cpu_li} ci={cpu_ci}",
                o.local_idx, o.ci
            ));
        }
    }
    println!("probed {} resident bricks, {} divergences", brick_coords.len(), bad.len());
    for line in bad.iter().take(20) {
        println!("  {line}");
    }
    assert!(bad.is_empty(), "find_brick_lookup diverged on {} bricks", bad.len());
}

// =====================================================================================
// VOLUME PROBE: run the REAL `sample_volume` (sdf::volume) on-device against ONE CPU-baked
// clipmap level uploaded as a 3D R16Snorm texture, comparing GPU-decoded distance vs the CPU
// trilinear decode. Isolates the "flat band / values read ~0" symptom: if the GPU returns ~0
// while the CPU varies, the 3D texture upload or sample path is broken (not the bake).
// =====================================================================================

use adventure::sdf_render::volume::{bake_level, snap_origin, VolumeConfig};

const VOLUME_PROBE_WGSL: &str = "
#import sdf::volume::sample_volume

struct PIn { x: f32, y: f32, z: f32, pad: f32 };
@group(0) @binding(2) var<storage, read> pts: array<PIn>;
@group(0) @binding(3) var<storage, read_write> outd: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    outd[i] = sample_volume(vec3<f32>(pts[i].x, pts[i].y, pts[i].z));
}
";

const VOLUME_MODULES: [&str; 2] = [
    "assets/shaders/sdf/bindings.wgsl",
    "assets/shaders/sdf/volume.wgsl",
];

fn compose_volume_probe() -> naga::Module {
    let mut composer = Composer::default();
    for path in VOLUME_MODULES {
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
            source: VOLUME_PROBE_WGSL,
            file_path: "volume_probe.wgsl",
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("compose volume_probe: {e}"))
}

// SdfVolumeUniform bytes for ONE active level (rest zeroed). Layout mirrors
// render.rs::SdfVolumeData: levels[4] vec4, decode[4] vec4, volume_dims vec4 = 36 floats.
fn volume_uniform_bytes(origin_world: Vec3, voxel_size: f32, decode: f32, res: u32) -> Vec<u8> {
    let mut f = [0.0f32; 36];
    f[0] = origin_world.x;
    f[1] = origin_world.y;
    f[2] = origin_world.z;
    f[3] = voxel_size;
    f[16] = decode; // decode[0].x
    f[32] = 1.0; // volume_dims.x = level_count
    f[33] = res as f32; // volume_dims.y = resolution
    bytemuck::cast_slice(&f).to_vec()
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VPIn {
    x: f32,
    y: f32,
    z: f32,
    pad: f32,
}

#[test]
fn gpu_sample_volume_matches_cpu() {
    use wgpu::util::DeviceExt;
    let Some((device, queue)) = device_queue() else {
        eprintln!("no GPU adapter -- skipping");
        return;
    };

    let cfg = VolumeConfig::default();
    let vs = cfg.level_voxel_size(0);
    let decode = cfg.decode_scale(0);
    let res = cfg.resolution;
    let edits = vec![ResolvedEdit {
        prim: SdfPrimitive::Sphere { radius: 3.0 },
        transform: bevy::prelude::Transform::IDENTITY,
        op: SdfOp::default(),
        material_id: 0,
    }];
    let aabbs: Vec<_> = edits
        .iter()
        .map(|e| edit_world_aabb(&e.prim, &e.transform, e.op.smoothing))
        .collect();
    let bvh = Bvh::build(&aabbs);
    let origin = snap_origin(Vec3::ZERO, vs, res);
    let data = bake_level(origin, vs, decode, res, &edits, &bvh);
    let origin_world = Vec3::new(origin.x as f32, origin.y as f32, origin.z as f32) * vs;

    let pts: Vec<Vec3> = (0..16)
        .map(|i| Vec3::new(i as f32 * 0.6, 0.0, 0.0))
        .filter(|p| {
            let rel = (*p - origin_world) / (res as f32 * vs);
            rel.x >= 0.0 && rel.x <= 1.0 && rel.y >= 0.0 && rel.y <= 1.0 && rel.z >= 0.0 && rel.z <= 1.0
        })
        .collect();
    assert!(!pts.is_empty(), "probe points must land inside the level box");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("volume_probe"),
        source: wgpu::ShaderSource::Naga(Cow::Owned(compose_volume_probe())),
    });

    let make_tex = |label: &str, bytes: &[u8], dim: u32| {
        device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: dim, height: dim, depth_or_array_layers: dim },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R16Snorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::default(),
            bytes,
        )
    };
    let level_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tex0 = make_tex("vol_l0", &level_bytes, res);
    let dummy = make_tex("vol_dummy", &[0u8, 0u8], 1);
    let v3 = |t: &wgpu::Texture| {
        t.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        })
    };
    let view0 = v3(&tex0);
    let viewd = v3(&dummy);
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("vol_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let vol_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vol_uniform"),
        contents: &volume_uniform_bytes(origin_world, vs, decode, res),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let in_data: Vec<VPIn> = pts.iter().map(|p| VPIn { x: p.x, y: p.y, z: p.z, pad: 0.0 }).collect();
    let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pts"),
        contents: bytemuck::cast_slice(&in_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_size = (pts.len() * 4) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("outd"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rb"),
        size: out_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("vol_probe_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg0"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 1, resource: vol_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: in_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
        ],
    });
    let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg1"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(&view0) },
            wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(&viewd) },
            wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(&viewd) },
            wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(&viewd) },
            wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::Sampler(&sampler) },
        ],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg0, &[]);
        pass.set_bind_group(1, &bg1, &[]);
        pass.dispatch_workgroups(pts.len() as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, out_size);
    queue.submit([enc.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let gpu: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    readback.unmap();

    let cpu_sample = |p: Vec3| -> f32 {
        let uvw = (p - origin_world) / vs;
        let clampc = |v: f32| v.clamp(0.0, (res - 1) as f32);
        let fx = clampc(uvw.x);
        let fy = clampc(uvw.y);
        let fz = clampc(uvw.z);
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let z0 = fz.floor() as usize;
        let rmax = res as usize - 1;
        let x1 = (x0 + 1).min(rmax);
        let y1 = (y0 + 1).min(rmax);
        let z1 = (z0 + 1).min(rmax);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let tz = fz - z0 as f32;
        let at = |x: usize, y: usize, z: usize| -> f32 {
            data[(z * res as usize + y) * res as usize + x] as f32 / 32767.0 * decode
        };
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let c00 = lerp(at(x0, y0, z0), at(x1, y0, z0), tx);
        let c10 = lerp(at(x0, y1, z0), at(x1, y1, z0), tx);
        let c01 = lerp(at(x0, y0, z1), at(x1, y0, z1), tx);
        let c11 = lerp(at(x0, y1, z1), at(x1, y1, z1), tx);
        lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz)
    };

    let mut max_err = 0.0f32;
    println!("x      GPU      CPU      err");
    for (p, &g) in pts.iter().zip(&gpu) {
        let c = cpu_sample(*p);
        let err = (g - c).abs();
        max_err = max_err.max(err);
        println!("{:.2}    gpu={g:.4}  cpu={c:.4}  err={err:.4}", p.x);
    }
    println!("max_err = {max_err}  (decode scale = {decode})");
    // The GPU sample must TRACK the CPU decode (varying from the inside clamp through the
    // zero-crossing to the outside clamp). A loose tolerance: the CPU reference uses
    // corner-clamped indexing while the GPU does true ClampToEdge trilinear, so a fraction
    // of a voxel of disagreement near the surface is expected. The failure we're guarding
    // against is GPU ≈ constant (texture not sampling the data at all).
    assert!(
        max_err < decode * 0.25,
        "GPU sample_volume diverges from CPU decode (max_err {max_err} vs decode {decode}) -- texture upload/sample broken?"
    );
    // Guard the real symptom directly: the sampled field must actually VARY (inside negative,
    // outside positive), not read a constant.
    let lo = gpu.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = gpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        lo < -0.5 && hi > 0.5,
        "GPU volume field is flat (min {lo}, max {hi}) -- texture not sampling the baked data"
    );
}
