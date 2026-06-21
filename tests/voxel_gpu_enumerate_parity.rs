//! **Phase G "G-c.1" — the GPU-vs-CPU ENUMERATE + face-cull parity gate** (docs/PHASE_G_GC_PLAN.md §6 "G-c.1",
//! §7 risk #1 — the LOAD-BEARING gate of the GPU-driven streaming pivot).
//!
//! The stage's whole point: the GPU clipmap enumeration + 6-face surface cull (`voxel_residency.wgsl` Pass B0
//! `prepare_shell_dispatch` + Pass B `enumerate_shells`) must produce a SURFACE-brick set that EXACTLY equals
//! the CPU oracle [`adventure::voxel::streaming::desired_clipmap_surface_classified`] (the desired clipmap
//! tiling ∩ `classify == Surface` — the same set the live `ResidencyManager` keeps resident). We dispatch the
//! two GPU passes, read back `candidate_list`, and assert SET-equality (no extras, no misses) over a
//! representative static scene + SEVERAL camera positions INCLUDING brick-boundary crossings and
//! negative-coord regions — the cases that matter. This lifts the CPU oracle test
//! `shell_first_resident_set_matches_cube_oracle` (streaming.rs) to GPU-vs-CPU.
//!
//! Skips cleanly when no GPU adapter is present (plain compute — no special features).

use adventure::voxel::brickmap::{BrickMap, MAX_LOD, brick_span};
use adventure::voxel::residency_gpu::SectorOccupancy;
use adventure::voxel::source::StaticVoxSource;
use adventure::voxel::streaming::{
    StreamingConfig, camera_brick_coord_lod, desired_clipmap_surface_classified, level_box_pub,
};
use bevy::math::IVec3;
use bytemuck::Zeroable;
use std::collections::HashSet;
use wgpu::util::DeviceExt;

#[path = "common/mod.rs"]
mod common;

const LODS: usize = (MAX_LOD + 1) as usize; // 8

/// One per-LOD block of `ResidencyParams` — MUST match the WGSL `LevelParams` (48 B, three vec3+scalar rows).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LevelParams {
    cam_brick_coord: [i32; 3],
    _pad_a: i32,
    cell_lo: [i32; 3],
    cell_offset: u32,
    cell_dims: [u32; 3],
    cell_count: u32,
}

/// The clipmap uniform — MUST match the WGSL `ResidencyParams` (8×48 + 32 = 416 B). std140 array stride of the
/// 48-B `LevelParams` is 48 (a multiple of 16), so the layout is exact. The enter-cap fields
/// (`hist_scale`/`cam_world`) are unused by the enumerate passes but MUST be present so the uniform's size
/// matches the shader (else a bind-size validation error).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ResidencyParams {
    levels: [LevelParams; LODS],
    clip_half_bricks: i32,
    total_cells: u32,
    hist_scale: f32,
    _pad1: u32,
    cam_world: [f32; 3],
    _pad2: u32,
}

/// The WG-cell grid edge in bricks — the WGSL Pass B0/B tile 8³-brick cells (`workgroup_size = 512`).
const WG_CELL: i32 = 8;

/// Build the [`ResidencyParams`] for the GPU passes from the camera + config — the CPU mirror of what the live
/// `prepare_gpu_residency` arm would write. For each LOD: `cam_brick_coord` (the SSOT
/// [`camera_brick_coord_lod`]), and the WG-cell tiling that covers the LOD's `level_box` (floored to the 8-brick
/// cell grid), with a prefix-sum `cell_offset` so a flat dispatch index decodes to (lod, local cell). Returns
/// the params + the total WG-cell count (the Pass B0 dispatch size).
fn build_params(cam: [f32; 3], cfg: &StreamingConfig) -> ResidencyParams {
    let half = cfg.clip_half_bricks;
    let mut levels = [LevelParams::zeroed(); LODS];
    let mut offset = 0u32;
    for lod in 0..=MAX_LOD {
        let (lo, hi) = level_box_pub(cam, lod, half);
        let cam_brick = camera_brick_coord_lod(cam, lod);
        // Floor the box lo to the 8-brick WG-cell grid (Euclidean — correct for negatives), and count the cells
        // that cover [lo, hi] inclusive.
        let cell_lo = IVec3::new(
            lo.x.div_euclid(WG_CELL) * WG_CELL,
            lo.y.div_euclid(WG_CELL) * WG_CELL,
            lo.z.div_euclid(WG_CELL) * WG_CELL,
        );
        let dims = IVec3::new(
            (hi.x - cell_lo.x).div_euclid(WG_CELL) + 1,
            (hi.y - cell_lo.y).div_euclid(WG_CELL) + 1,
            (hi.z - cell_lo.z).div_euclid(WG_CELL) + 1,
        );
        let count = (dims.x * dims.y * dims.z) as u32;
        levels[lod as usize] = LevelParams {
            cam_brick_coord: [cam_brick.x, cam_brick.y, cam_brick.z],
            _pad_a: 0,
            cell_lo: [cell_lo.x, cell_lo.y, cell_lo.z],
            cell_offset: offset,
            cell_dims: [dims.x as u32, dims.y as u32, dims.z as u32],
            cell_count: count,
        };
        offset += count;
    }
    ResidencyParams {
        levels,
        clip_half_bricks: half,
        total_cells: offset,
        hist_scale: 0.0, // enter-cap unused by the enumerate passes
        _pad1: MAX_LOD + 1, // 4-S1: this u32 is `backdrop_lod` in the WGSL now — MAX_LOD+1 = backdrop OFF
        cam_world: cam,
        _pad2: 0,
    }
}

/// Dispatch the GPU Pass B0 + Pass B over the uploaded occupancy `occ` with the clipmap `params`, and read back
/// the resulting surface `candidate_list` as a `(coord, lod)` SET. The single GPU round-trip the gate measures.
fn gpu_candidate_set(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    occ: &SectorOccupancy,
    params: &ResidencyParams,
    cap: usize,
) -> HashSet<(IVec3, u32)> {
    let header = occ.header();
    let header_buf = buf_init(device, "header", bytemuck::bytes_of(&header), wgpu::BufferUsages::UNIFORM);
    let entries_buf =
        buf_init(device, "entries", bytemuck::cast_slice(occ.entries()), wgpu::BufferUsages::STORAGE);
    let params_buf = buf_init(device, "params", bytemuck::bytes_of(params), wgpu::BufferUsages::UNIFORM);

    // Pass B0 outputs: solid WG-cell indices + count + the Pass-B indirect dispatch (seed [0,1,1]).
    let total_cells = params.total_cells.max(1) as usize;
    let shell_indices = storage_buf(device, "shell_idx", (total_cells * 4) as u64);
    let shell_count = buf_init(device, "shell_count", bytemuck::bytes_of(&0u32), storage_usage());
    let shell_dispatch =
        buf_init(device, "shell_dispatch", bytemuck::cast_slice(&[0u32, 1u32, 1u32]), dispatch_usage());

    // Pass B outputs: the surface candidate count + list.
    let cand_count = buf_init(device, "cand_count", bytemuck::bytes_of(&0u32), storage_usage());
    let cand_list = storage_buf(device, "cand_list", (cap * 16) as u64); // vec4<i32> = 16 B
    // G-c.2a — Pass B also emits the DESIRED (occupied-in-shell) set at bindings 10/11. This gate only checks the
    // SURFACE `candidate_list`, but `enumerate_shells` now writes both, so bind real buffers for them.
    let desired_count = buf_init(device, "desired_count", bytemuck::bytes_of(&0u32), storage_usage());
    let desired_list = storage_buf(device, "desired_list", (cap * 16) as u64);

    let src = std::fs::read_to_string("assets/shaders/voxel_residency.wgsl")
        .expect("read voxel_residency.wgsl");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_residency_enum"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });

    // All bindings the two enumerate entries declare at @group(0): header@0, entries@1, (keys@2/out@3 unused
    // here — bind dummies), params@4, shell_wg_indices@5, shell_count@6, shell_dispatch@7, candidate_count@8,
    // candidate_list@9. (2/3 are the G-c.0 parity entry's bindings; wgpu's auto-layout for the enumerate
    // entries doesn't require them, but a single shared bind-group layout is simplest — bind small dummies.)
    let dummy_in = buf_init(device, "dummy_in", bytemuck::cast_slice(&[0u32; 4]), wgpu::BufferUsages::STORAGE);
    let dummy_out = storage_buf(device, "dummy_out", 16);

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("enum_bgl"),
        entries: &[
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            uniform_entry(4),
            storage_entry(5, false),
            storage_entry(6, false),
            storage_entry(7, false),
            storage_entry(8, false),
            storage_entry(9, false),
            storage_entry(10, false),
            storage_entry(11, false),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("enum_pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let make_pipeline = |entry: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pl),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let p_b0 = make_pipeline("prepare_shell_dispatch");
    let p_fin = make_pipeline("finalize_shell_dispatch_2d");
    let p_b = make_pipeline("enumerate_shells");

    // Pass B0's bind group binds the REAL `shell_dispatch` at binding 7 (so `finalize_shell_dispatch_2d`, which
    // runs with this same bind group, can write the 2D indirect dims from `shell_count`). Pass B uses
    // `shell_dispatch` ONLY as the INDIRECT source — so its bind group binds a DUMMY at 7 (Pass B's shader does
    // not reference binding 7), avoiding the "same buffer as STORAGE + INDIRECT in one dispatch" conflict.
    let dummy_dispatch = storage_buf(device, "dummy_dispatch", 16);
    let make_bg = |label: &str, slot7: &wgpu::Buffer| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &layout,
            entries: &[
                bind(0, &header_buf),
                bind(1, &entries_buf),
                bind(2, &dummy_in),
                bind(3, &dummy_out),
                bind(4, &params_buf),
                bind(5, &shell_indices),
                bind(6, &shell_count),
                bind(7, slot7),
                bind(8, &cand_count),
                bind(9, &cand_list),
                bind(10, &desired_count),
                bind(11, &desired_list),
            ],
        })
    };
    let bg_b0 = make_bg("enum_bg_b0", &shell_dispatch);
    let bg_b = make_bg("enum_bg_b", &dummy_dispatch);

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enum_enc") });
    {
        let mut pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("b0"), timestamp_writes: None });
        pass.set_pipeline(&p_b0);
        pass.set_bind_group(0, &bg_b0, &[]);
        // prepare_shell_dispatch runs at @workgroup_size(256) (size-agnostic — mirror of the live front end).
        pass.dispatch_workgroups(params.total_cells.div_ceil(256).max(1), 1, 1);
        // Finalize the indirect dims: shell_dispatch [shell_count,1,1] → 2D [x,y,1] (so enumerate_shells is
        // size-agnostic past the 65535 workgroup-per-dim cap). Uses bg_b0 (real shell_dispatch at binding 7).
        pass.set_pipeline(&p_fin);
        pass.dispatch_workgroups(1, 1, 1);
    }
    {
        // Pass B is `record_indirect` over Pass B0's GPU-written 2D dispatch — exactly the design's record_indirect.
        let mut pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("b"), timestamp_writes: None });
        pass.set_pipeline(&p_b);
        pass.set_bind_group(0, &bg_b, &[]);
        pass.dispatch_workgroups_indirect(&shell_dispatch, 0);
    }
    queue.submit(std::iter::once(encoder.finish()));

    let count = readback_u32(device, queue, &cand_count, 1)[0] as usize;
    assert!(count <= cap, "candidate count {count} exceeded the cap {cap} — raise the test cap");
    let raw = readback_i32(device, queue, &cand_list, count * 4);
    let mut set = HashSet::new();
    for chunk in raw.chunks_exact(4) {
        set.insert((IVec3::new(chunk[0], chunk[1], chunk[2]), chunk[3] as u32));
    }
    assert_eq!(set.len(), count, "the GPU candidate_list contained duplicate keys (should be a clean set)");
    set
}

// --- small wgpu helpers ---

fn buf_init(device: &wgpu::Device, label: &str, bytes: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytes, usage })
}
fn storage_buf(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
fn storage_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
}
fn dispatch_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
}
fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
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
fn bind(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }
}

fn readback_u32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, words: usize) -> Vec<u32> {
    bytemuck::cast_slice(&readback_bytes(device, queue, buf, words * 4)).to_vec()
}
fn readback_i32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, words: usize) -> Vec<i32> {
    bytemuck::cast_slice(&readback_bytes(device, queue, buf, words * 4)).to_vec()
}
fn readback_bytes(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, bytes: usize) -> Vec<u8> {
    let bytes = (bytes.max(4)) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("enum_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enum_rb") });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    queue.submit(std::iter::once(encoder.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = staging.slice(..).get_mapped_range().expect("map staging").to_vec();
    staging.unmap();
    data
}

/// A representative static scene: a partly-solid slab + a tall pillar + an isolated cluster, spanning enough
/// bricks that several LOD shells thread through it (so the coarse-LOD enumeration + the cross-LOD hole-clip
/// are exercised) AND spanning NEGATIVE coords (the clipmap reaches both signs). All bricks fully solid except
/// the slab top layer (PARTIAL bricks ⇒ always Surface), so both the `is_full`/Interior cull and the
/// partial-overrides-occlusion path are present. Shared SSOT — see [`common::slab_pillar_cluster_scene`].
fn representative_scene() -> BrickMap {
    common::slab_pillar_cluster_scene()
}

/// **THE GATE — GPU enumerate + face-cull ≡ the CPU surface oracle, EXACTLY**, over several camera positions
/// incl. brick-boundary crossings and negative-coord regions. A small `clip_half` so the cube oracle
/// (`desired_clipmap`) enumerates without bailing at `MAX_CLIP_ENUMERATION`, and so the WG-cell dispatch stays
/// modest. The GPU `candidate_list` SET must EQUAL the CPU `desired_clipmap_surface_classified` SET — no
/// extras, no misses. If they differ the test reports the symmetric difference (the divergent keys).
#[test]
fn gpu_enumerate_matches_cpu_surface_oracle() {
    // The enumerate pass dispatches 8³ = 512-thread workgroups, above wgpu's default 256 — request the raised
    // compute limit (the renderer bumps the same in `wgpu_settings()`).
    // G-c.2a — `enumerate_shells` now also emits the desired list (bindings 10/11), so the layout binds 10
    // storage buffers — over wgpu's default 8. Raise the storage-buffer limit alongside the 512-wide compute.
    let Some((device, queue)) = common::headless_compute_device_with_storage(512, 12) else {
        eprintln!("[skip] no GPU adapter (or limits too low) — voxel GPU enumerate parity skipped");
        return;
    };

    let map = representative_scene();
    let source = StaticVoxSource::new(&map);
    let occ = SectorOccupancy::from_occupied_full(source.occupied_keys_full());

    let cfg = StreamingConfig { clip_half_bricks: 8, max_resident_bricks: usize::MAX, max_bricks_per_frame: 1 };
    let span0 = brick_span(0);

    // Camera positions: the scene centre; offsets that cross LOD0 + coarse brick BOUNDARIES (the case that
    // matters — a crossing shifts the snapped box / hole edge); and a NEGATIVE-coord camera (over the slab's
    // negative region). Mixed sub-brick offsets exercise the even/odd snap.
    let cams: [[f32; 3]; 6] = [
        [0.5 * span0, 1.5 * span0, 0.5 * span0],     // near origin, surface in LOD0
        [0.0, 1.5 * span0, 0.0],                     // exactly ON a LOD0 brick boundary (x=z=0)
        [span0, 2.0 * span0, span0],                 // one brick over (a LOD0 crossing)
        [-2.5 * span0, 1.0 * span0, -2.5 * span0],   // NEGATIVE-coord camera over the slab
        [0.5 * span0, 5.5 * span0, 0.5 * span0],     // up the pillar (coarse shells thread the surface)
        [7.5 * span0, 4.5 * span0, 7.5 * span0],     // off toward the isolated cluster
    ];

    // A generous cap for the candidate buffer (the surface shell at clip_half 8 over this scene is small).
    let cap = 200_000usize;
    let mut total_checked = 0usize;
    for (i, cam) in cams.iter().enumerate() {
        let cpu: HashSet<(IVec3, u32)> = desired_clipmap_surface_classified(*cam, &cfg, &source)
            .into_iter()
            .map(|k| (k.coord, k.lod))
            .collect();
        let params = build_params(*cam, &cfg);
        let gpu = gpu_candidate_set(&device, &queue, &occ, &params, cap);

        // EXACT set-equality. Report the divergence precisely if it fails (never loosen the gate).
        if cpu != gpu {
            let missing: Vec<_> = cpu.difference(&gpu).take(20).collect();
            let extra: Vec<_> = gpu.difference(&cpu).take(20).collect();
            panic!(
                "[cam {i} {cam:?}] GPU surface set != CPU oracle. cpu={} gpu={} \
                 missing(GPU lacks, first 20)={missing:?} extra(GPU has, first 20)={extra:?}",
                cpu.len(),
                gpu.len(),
            );
        }
        assert!(!cpu.is_empty(), "[cam {i}] the surface set must be non-empty (the surface is in the clipmap)");
        // The crossing cases must actually enumerate coarse shells (≥1 LOD>0 surface brick), proving the GPU
        // clipmap math reaches the coarse levels, not just LOD0.
        let has_coarse = cpu.iter().any(|(_, l)| *l > 0);
        assert!(has_coarse, "[cam {i}] the oracle must include coarse-LOD surface (the scene threads the shells)");
        total_checked += cpu.len();
        eprintln!("[gpu-enumerate-parity] cam {i} {cam:?}: {} surface bricks match exactly", cpu.len());
    }
    eprintln!(
        "[gpu-enumerate-parity] OK — {} cameras, {total_checked} surface keys total, table_size {}",
        cams.len(),
        occ.table_size(),
    );
}
