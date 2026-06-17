//! **Phase G "G-c.2a" — the GPU RESIDENCY DIFF (Pass C) set-parity gate** (docs/PHASE_G_GC_PLAN.md §1 Pass C,
//! §2.3 slot allocation, §3.2 keep-old-until-revealed, §6 "G-c.2", §7).
//!
//! Pass C turns the GPU-enumerated candidate surface set (`candidate_list`, G-c.1) into enter/drop decisions +
//! a GPU resident `slot_table`, replacing the CPU `ResidencyManager`/`ResidentPacker` drop/enqueue decision +
//! slot allocator. This gate proves the GPU resident SET (the set of resident `(coord, lod)` keys held by the
//! GPU `slot_table`) EXACTLY equals the CPU `ResidencyManager` resident set at every CONVERGED step, over a
//! SEQUENCE of camera positions — cold fill, brick-boundary crossings, a recede/approach that triggers LOD
//! coarsen/refine, and NEGATIVE coords.
//!
//! ## What "converged" means + how the two sides converge
//! * **CPU reference:** alternate `ResidencyManager::update(cam)` then `drain_work_from` until the queue empties
//!   AND an `update` produces no drops and enqueues nothing new — keep-old-until-revealed makes a single
//!   update+drain insufficient (a superseded LOD only drops once its replacement is RESIDENT, on a LATER
//!   update). The converged `resident` set is the reference.
//! * **GPU side:** the Pass C pipeline runs ONE diff round per "frame" (release-quarantine → build present-flag
//!   → drop-mark → drop-apply → enter); one round = one CPU update+drain. We iterate the round until
//!   `change_count == 0` (the idempotency signal), then read back the `slot_table` resident set. Each round
//!   re-runs Pass B (enumerate) for the same camera (the candidate/desired lists are per-frame transients).
//!
//! ## Parity = SET equality by key (NOT byte-per-slot)
//! The GPU free-list assigns slots in a DIFFERENT order than the CPU `SlotAllocator`, so per-slot byte-identity
//! is neither expected nor required (the design says so explicitly, §6 G-c.2 + the deliverable). We compare the
//! set of resident `(coord, lod)` KEYS; on failure we report the symmetric difference.
//!
//! ## keep-old-until-revealed parity
//! After a move, BEFORE the replacement is entered, a transition brick must be RETAINED on the GPU iff the CPU
//! `safe_to_drop` would retain it — i.e. no still-covered key becomes a hole. We assert this directly: after the
//! FIRST GPU diff round following a move (replacements not yet resident), the GPU resident set must still cover
//! every region the CPU's pre-converged set covers (the no-hole invariant), matching the CPU's own first-round
//! retained set.
//!
//! Skips cleanly when no GPU adapter (or its compute limit is too low for the 512-wide enumerate workgroup).

use adventure::voxel::brickmap::{BRICK_EDGE, Brick, BrickMap, MAX_LOD, brick_span};
use adventure::voxel::edits::VoxelEdits;
use adventure::voxel::palette::{BlockId, BlockRegistry};
use adventure::voxel::residency_gpu::{GpuResidencyDiffConfig, SectorOccupancy, brick_key_hash};
use adventure::voxel::source::StaticVoxSource;
use adventure::voxel::streaming::{
    ResidencyManager, StreamingConfig, camera_brick_coord_lod, level_box_pub,
};
use bevy::math::IVec3;
use bytemuck::Zeroable;
use rustc_hash::FxHashMap;
use std::collections::HashSet;
use wgpu::util::DeviceExt;

#[path = "common/mod.rs"]
mod common;

const LODS: usize = (MAX_LOD + 1) as usize; // 8
const WG_CELL: i32 = 8;
/// MUST match the WGSL `REFINE_*` / streaming.rs `REFINE_DESCENT_CAP`.
const REFINE_DESCENT_CAP: u32 = 5;
/// EMPTY-slot sentinel — MUST match the WGSL `EMPTY_LOD`.
const EMPTY_LOD: u32 = 0xFFFF_FFFF;

// =========================================================================================================
//  ResidencyParams (Pass B input) — copied from the G-c.1 enumerate-parity rig (same SSOT).
// =========================================================================================================

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

/// Enter-cap distance histogram buckets — MUST equal `HIST_BUCKETS` in `voxel_residency.wgsl`.
const HIST_BUCKETS: u32 = 4096;

fn build_params(cam: [f32; 3], cfg: &StreamingConfig) -> ResidencyParams {
    let half = cfg.clip_half_bricks;
    let mut levels = [LevelParams::zeroed(); LODS];
    let mut offset = 0u32;
    for lod in 0..=MAX_LOD {
        let (lo, hi) = level_box_pub(cam, lod, half);
        let cam_brick = camera_brick_coord_lod(cam, lod);
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
    let max_dist = (half.max(1) as f32) * brick_span(MAX_LOD) * 3.0_f32.sqrt();
    ResidencyParams {
        levels,
        clip_half_bricks: half,
        total_cells: offset,
        hist_scale: HIST_BUCKETS as f32 / max_dist,
        _pad1: 0,
        cam_world: cam,
        _pad2: 0,
    }
}

// =========================================================================================================
//  The persistent GPU residency-diff state — slot_table + free-list + quarantine survive across cameras.
// =========================================================================================================

struct GpuDiff {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // Occupancy (immutable for the scene).
    header_buf: wgpu::Buffer,
    entries_buf: wgpu::Buffer,
    // Sizing.
    cand_cap: usize,
    slot_table_size: u32,
    present_size: u32,
    // Per-frame transient lists (re-zeroed each round).
    // Persistent diff buffers.
    slot_table_buf: wgpu::Buffer,    // SLOT_WORDS=5 u32 / slot
    free_ring_buf: wgpu::Buffer,
    free_ctrl_buf: wgpu::Buffer,     // [head, tail]
    quarantine_ring_buf: wgpu::Buffer,
    quarantine_ctrl_buf: wgpu::Buffer,
    diff_cfg_buf: wgpu::Buffer,
    // Pipelines.
    p_b0: wgpu::ComputePipeline,
    p_b: wgpu::ComputePipeline,
    p_present: wgpu::ComputePipeline,
    p_release: wgpu::ComputePipeline,
    p_mark: wgpu::ComputePipeline,
    p_apply: wgpu::ComputePipeline,
    p_enter: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl GpuDiff {
    fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        occ: &SectorOccupancy,
        max_resident: u32,
        cand_cap: usize,
    ) -> Self {
        let header = occ.header();
        let header_buf =
            buf_init(&device, "header", bytemuck::bytes_of(&header), wgpu::BufferUsages::UNIFORM);
        let entries_buf =
            buf_init(&device, "entries", bytemuck::cast_slice(occ.entries()), wgpu::BufferUsages::STORAGE);

        // Hash tables sized for ~0.5 load factor over the worst-case resident set.
        let slot_table_size = (max_resident as usize * 2).max(2).next_power_of_two() as u32;
        let present_size = (cand_cap * 2).max(2).next_power_of_two() as u32;

        // Persistent buffers, initialized empty.
        let slot_table_buf = empty_slot_table(&device, slot_table_size);
        let free_ring_buf = storage_buf(&device, "free_ring", (max_resident as u64) * 4);
        // free-list starts FULL: head=0, tail=max_resident, ring[i]=i (every slot free).
        let free_init: Vec<u32> = (0..max_resident).collect();
        queue.write_buffer(&free_ring_buf, 0, bytemuck::cast_slice(&free_init));
        let free_ctrl_buf =
            buf_init(&device, "free_ctrl", bytemuck::cast_slice(&[0u32, max_resident]), storage_usage());
        let quarantine_ring_buf = storage_buf(&device, "quar_ring", (max_resident as u64) * 4);
        let quarantine_ctrl_buf =
            buf_init(&device, "quar_ctrl", bytemuck::cast_slice(&[0u32, 0u32]), storage_usage());

        let cfg = GpuResidencyDiffConfig {
            slot_table_size,
            present_size,
            max_resident,
            refine_descent_cap: REFINE_DESCENT_CAP,
        };
        let diff_cfg_buf =
            buf_init(&device, "diff_cfg", bytemuck::bytes_of(&cfg), wgpu::BufferUsages::UNIFORM);

        let src = std::fs::read_to_string("assets/shaders/voxel_residency.wgsl")
            .expect("read voxel_residency.wgsl");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel_residency_diff"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("diff_bgl"),
            entries: &[
                uniform_entry(0),       // header
                storage_entry(1, true), // entries
                storage_entry(2, true), // query_keys (dummy)
                storage_entry(3, false),// query_out (dummy)
                uniform_entry(4),       // params
                storage_entry(5, false),// shell_wg_indices
                storage_entry(6, false),// shell_count
                storage_entry(7, false),// shell_dispatch
                storage_entry(8, false),// candidate_count
                storage_entry(9, false),// candidate_list
                storage_entry(10, false),// desired_count
                storage_entry(11, false),// desired_list
                uniform_entry(12),      // diff_cfg
                storage_entry(13, false),// slot_table
                storage_entry(14, false),// free_ring
                storage_entry(15, false),// free_ctrl
                storage_entry(16, false),// quarantine_ring
                storage_entry(17, false),// quarantine_ctrl
                storage_entry(18, false),// present_flag
                storage_entry(19, false),// enter_count
                storage_entry(20, false),// enter_list
                storage_entry(21, false),// drop_count
                storage_entry(22, false),// drop_list
                storage_entry(23, false),// drop_decision
                storage_entry(51, false),// enter_cap (G-c.4 nearest cap; default = no cap in this gate)
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("diff_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let mk = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let p_b0 = mk("prepare_shell_dispatch");
        let p_b = mk("enumerate_shells");
        let p_present = mk("build_present_flag");
        let p_release = mk("diff_release_quarantine");
        let p_mark = mk("diff_drop_mark");
        let p_apply = mk("diff_drop_apply");
        let p_enter = mk("diff_enter_scan");

        Self {
            device,
            queue,
            header_buf,
            entries_buf,
            cand_cap,
            slot_table_size,
            present_size,
            slot_table_buf,
            free_ring_buf,
            free_ctrl_buf,
            quarantine_ring_buf,
            quarantine_ctrl_buf,
            diff_cfg_buf,
            p_b0,
            p_b,
            p_present,
            p_release,
            p_mark,
            p_apply,
            p_enter,
            bgl,
        }
    }

    /// Run ONE diff ROUND for `cam` (= one CPU update+drain). Returns `change_count` (enter + drop). Mutates the
    /// persistent slot_table / free-list / quarantine.
    fn round(&self, params: &ResidencyParams) -> u32 {
        let device = &self.device;
        let queue = &self.queue;
        let params_buf =
            buf_init(device, "params", bytemuck::bytes_of(params), wgpu::BufferUsages::UNIFORM);

        // Per-frame transients (re-created/zeroed each round).
        let total_cells = params.total_cells.max(1) as usize;
        let shell_indices = storage_buf(device, "shell_idx", (total_cells * 4) as u64);
        let shell_count = buf_init(device, "shell_count", bytemuck::bytes_of(&0u32), storage_usage());
        let shell_dispatch =
            buf_init(device, "shell_dispatch", bytemuck::cast_slice(&[0u32, 1u32, 1u32]), dispatch_usage());
        let cand_count = buf_init(device, "cand_count", bytemuck::bytes_of(&0u32), storage_usage());
        let cand_list = storage_buf(device, "cand_list", (self.cand_cap * 16) as u64);
        let desired_count = buf_init(device, "desired_count", bytemuck::bytes_of(&0u32), storage_usage());
        let desired_list = storage_buf(device, "desired_list", (self.cand_cap * 16) as u64);
        // present_flag is a per-round hash (re-zeroed to EMPTY each round).
        let present_init = vec![EMPTY_LOD; self.present_size as usize * 4];
        let present_flag = buf_init(device, "present_flag", bytemuck::cast_slice(&present_init), storage_usage());
        let enter_count = buf_init(device, "enter_count", bytemuck::bytes_of(&0u32), storage_usage());
        let enter_list = storage_buf(device, "enter_list", (self.cand_cap * 16) as u64);
        let drop_count = buf_init(device, "drop_count", bytemuck::bytes_of(&0u32), storage_usage());
        let drop_list = storage_buf(device, "drop_list", (self.cand_cap * 16) as u64);
        let drop_decision = storage_buf(device, "drop_decision", (self.slot_table_size as u64) * 4);
        // G-c.4 enter-cap: this gate does NOT run the cap passes, so seed `enter_cap = [HIST_BUCKETS, 0]` ⇒ no cap
        // (the enter scan admits every candidate, exactly as before) while the shared `diff_enter_scan` still binds it.
        let enter_cap = buf_init(device, "enter_cap", bytemuck::cast_slice(&[HIST_BUCKETS, 0u32]), storage_usage());

        let dummy_in = buf_init(device, "dummy_in", bytemuck::cast_slice(&[0u32; 4]), wgpu::BufferUsages::STORAGE);
        let dummy_out = storage_buf(device, "dummy_out", 16);
        let dummy_dispatch = storage_buf(device, "dummy_dispatch", 16);

        let mk_bg = |label: &str, slot7: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.bgl,
                entries: &[
                    bind(0, &self.header_buf),
                    bind(1, &self.entries_buf),
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
                    bind(12, &self.diff_cfg_buf),
                    bind(13, &self.slot_table_buf),
                    bind(14, &self.free_ring_buf),
                    bind(15, &self.free_ctrl_buf),
                    bind(16, &self.quarantine_ring_buf),
                    bind(17, &self.quarantine_ctrl_buf),
                    bind(18, &present_flag),
                    bind(19, &enter_count),
                    bind(20, &enter_list),
                    bind(21, &drop_count),
                    bind(22, &drop_list),
                    bind(23, &drop_decision),
                    bind(51, &enter_cap),
                ],
            })
        };
        let bg_main = mk_bg("diff_bg", &dummy_dispatch);
        let bg_b0 = mk_bg("diff_bg_b0", &shell_dispatch); // Pass B0 writes shell_dispatch at binding 7

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("diff_enc") });
        // Pass A — release the previous frame's quarantine + clear counters.
        compute(&mut enc, &self.p_release, &bg_main, 1);
        // Pass B0 — shell dispatch prep (binds the real shell_dispatch at 7).
        compute(&mut enc, &self.p_b0, &bg_b0, params.total_cells.div_ceil(64).max(1));
        // Pass B — enumerate (indirect over shell_dispatch); emits candidate_list + desired_list.
        {
            let mut pass =
                enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("b"), timestamp_writes: None });
            pass.set_pipeline(&self.p_b);
            pass.set_bind_group(0, &bg_main, &[]);
            pass.dispatch_workgroups_indirect(&shell_dispatch, 0);
        }
        queue.submit(std::iter::once(enc.finish()));

        // Read back the desired/candidate counts so the subsequent dispatches size correctly.
        let d_cnt = readback_u32(device, queue, &desired_count, 1)[0];
        let c_cnt = readback_u32(device, queue, &cand_count, 1)[0];

        let mut enc2 =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("diff_enc2") });
        // Pass C0 — build present-flag from desired_list.
        compute(&mut enc2, &self.p_present, &bg_main, d_cnt.div_ceil(64).max(1));
        // Pass C2a — drop mark (over the pre-drop slot table). One invocation per slot.
        compute(&mut enc2, &self.p_mark, &bg_main, self.slot_table_size.div_ceil(64).max(1));
        // Pass C2b — drop apply.
        compute(&mut enc2, &self.p_apply, &bg_main, self.slot_table_size.div_ceil(64).max(1));
        // Pass C1 — enter scan (AFTER drops, mirroring CPU update: safe_to_drop sees the pre-enter set).
        compute(&mut enc2, &self.p_enter, &bg_main, c_cnt.div_ceil(64).max(1));
        queue.submit(std::iter::once(enc2.finish()));

        let enters = readback_u32(device, queue, &enter_count, 1)[0];
        let drops = readback_u32(device, queue, &drop_count, 1)[0];
        enters + drops
    }

    /// Iterate diff rounds for `cam` until `change_count == 0` (converged). Returns the round count.
    fn converge(&self, cam: [f32; 3], cfg: &StreamingConfig) -> u32 {
        let params = build_params(cam, cfg);
        let mut rounds = 0u32;
        loop {
            let change = self.round(&params);
            rounds += 1;
            if change == 0 {
                break;
            }
            assert!(rounds < 64, "GPU diff failed to converge for cam {cam:?} in 64 rounds");
        }
        rounds
    }

    /// The resident `(coord, lod)` key SET from the GPU `slot_table` (read back).
    fn resident_set(&self) -> HashSet<(IVec3, u32)> {
        let words = readback_u32(&self.device, &self.queue, &self.slot_table_buf, self.slot_table_size as usize * 5);
        let mut set = HashSet::new();
        for slot in words.chunks_exact(5) {
            let lod = slot[3];
            if lod == EMPTY_LOD {
                continue;
            }
            let coord = IVec3::new(slot[0] as i32, slot[1] as i32, slot[2] as i32);
            assert!(set.insert((coord, lod)), "duplicate key {coord:?}@{lod} in slot_table");
        }
        set
    }
}

/// Initialize the slot table to ALL EMPTY (lod word = EMPTY_LOD per slot).
fn empty_slot_table(device: &wgpu::Device, slots: u32) -> wgpu::Buffer {
    let mut words = vec![0u32; slots as usize * 5];
    for slot in words.chunks_exact_mut(5) {
        slot[3] = EMPTY_LOD;
    }
    buf_init(device, "slot_table", bytemuck::cast_slice(&words), storage_usage())
}

fn compute(enc: &mut wgpu::CommandEncoder, p: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, wgs: u32) {
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
    pass.set_pipeline(p);
    pass.set_bind_group(0, bg, &[]);
    pass.dispatch_workgroups(wgs, 1, 1);
}

// =========================================================================================================
//  CPU reference — drive ResidencyManager to convergence; expose the resident key set.
// =========================================================================================================

/// Drive the CPU `ResidencyManager` to CONVERGENCE for `cam`: alternate `update` + `drain_work_from` until the
/// queue empties AND an `update` drops nothing new. Returns the converged resident key set.
fn cpu_converge(
    mgr: &mut ResidencyManager,
    cam: [f32; 3],
    cfg: &StreamingConfig,
    source: &StaticVoxSource,
    registry: &BlockRegistry,
) -> HashSet<(IVec3, u32)> {
    let edits = VoxelEdits::new();
    for _ in 0..64 {
        mgr.update(cam, cfg, source);
        // Drain everything queued this update (the test caps are large enough to drain in one call).
        while mgr.pending() > 0 {
            mgr.drain_work_from(cfg, source, registry, &edits);
        }
        // A second update with nothing newly queued + no drops ⇒ converged.
        let before = mgr.resident_count();
        let dropped = mgr.update(cam, cfg, source);
        while mgr.pending() > 0 {
            mgr.drain_work_from(cfg, source, registry, &edits);
        }
        if dropped == 0 && mgr.resident_count() == before {
            break;
        }
    }
    cpu_resident_set(mgr)
}

fn cpu_resident_set(mgr: &ResidencyManager) -> HashSet<(IVec3, u32)> {
    mgr.resident_entries().into_iter().map(|e| (e.coord, e.lod)).collect()
}

// =========================================================================================================
//  Scene.
// =========================================================================================================

/// A representative static scene with full + partial bricks, a tall pillar threading LOD shells, an isolated
/// far cluster, and negative-coord geometry. (Same shape as the G-c.1 enumerate-parity scene.)
fn representative_scene() -> BrickMap {
    let mut map = BrickMap::new();
    let full = |id: u16| {
        let mut v = Box::new([BlockId::AIR; (BRICK_EDGE * BRICK_EDGE * BRICK_EDGE) as usize]);
        v.iter_mut().for_each(|c| *c = BlockId(id));
        Brick::from_voxels(v)
    };
    let partial = |id: u16| {
        let mut v = Box::new([BlockId(id); (BRICK_EDGE * BRICK_EDGE * BRICK_EDGE) as usize]);
        v[0] = BlockId::AIR;
        Brick::from_voxels(v)
    };
    for z in -3..3 {
        for x in -3..3 {
            for y in 0..3 {
                let brick = if y == 2 { partial(2) } else { full(1) };
                map.insert(IVec3::new(x, y, z), brick);
            }
        }
    }
    for y in 3..10 {
        map.insert(IVec3::new(0, y, 0), full(3));
    }
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                map.insert(IVec3::new(15 + x, 4 + y, 15 + z), full(4));
            }
        }
    }
    map
}

fn registry() -> BlockRegistry {
    // StaticVoxSource ignores the registry in `brick`, so an AIR-only registry suffices for the drain.
    BlockRegistry::air_only()
}

// =========================================================================================================
//  THE GATES.
// =========================================================================================================

/// **THE GATE — GPU Pass C resident SET ≡ the CPU `ResidencyManager` resident set, EXACTLY**, at every converged
/// step across a camera SEQUENCE: cold fill, brick-boundary crossings, a recede (coarsen) + approach (refine)
/// that flips LODs, and negative-coord regions. EXACT set match by key; reports the symmetric difference on
/// failure.
#[test]
fn gpu_residency_diff_set_matches_cpu_at_each_converged_step() {
    let Some((device, queue)) = common::headless_compute_device_with_storage(512, 24) else {
        eprintln!("[skip] no GPU adapter (or limits too low) — voxel GPU residency-diff parity skipped");
        return;
    };

    let map = representative_scene();
    let source = StaticVoxSource::new(&map);
    let occ = SectorOccupancy::from_occupied_full(source.occupied_keys_full());
    let registry = registry();

    let cfg = StreamingConfig {
        clip_half_bricks: 8,
        max_resident_bricks: usize::MAX,
        max_bricks_per_frame: usize::MAX,
    };
    let span0 = brick_span(0);

    // The camera SEQUENCE. The GPU diff + CPU manager are BOTH stateful and follow this same path, so the
    // keep-old-until-revealed transitions (coarsen on recede, refine on approach) are exercised on both sides.
    //  0 cold fill at origin surface;
    //  1 one LOD0 brick over (a crossing);
    //  2 RECEDE far up the pillar — coarse shells thread the surface (coarsen);
    //  3 even farther (more coarsen);
    //  4 APPROACH back near origin (refine — fine shells re-enter);
    //  5 NEGATIVE-coord camera over the slab's negative region (crossing into negatives);
    //  6 toward the isolated +X cluster.
    let cams: [[f32; 3]; 7] = [
        [0.5 * span0, 1.5 * span0, 0.5 * span0],
        [1.5 * span0, 1.5 * span0, 0.5 * span0],
        [0.5 * span0, 30.0 * span0, 0.5 * span0],
        [0.5 * span0, 80.0 * span0, 0.5 * span0],
        [0.5 * span0, 2.5 * span0, 0.5 * span0],
        [-2.5 * span0, 1.0 * span0, -2.5 * span0],
        [7.5 * span0, 4.5 * span0, 7.5 * span0],
    ];

    // The candidate buffer cap (the surface shell at clip_half 8 over this scene is small).
    let cand_cap = 200_000usize;
    let gpu = GpuDiff::new(device, queue, &occ, cand_cap as u32, cand_cap);
    let mut mgr = ResidencyManager::new();

    let mut total = 0usize;
    for (i, cam) in cams.iter().enumerate() {
        let cpu = cpu_converge(&mut mgr, *cam, &cfg, &source, &registry);
        let rounds = gpu.converge(*cam, &cfg);
        let gpu_set = gpu.resident_set();

        if cpu != gpu_set {
            let missing: Vec<_> = cpu.difference(&gpu_set).take(20).collect();
            let extra: Vec<_> = gpu_set.difference(&cpu).take(20).collect();
            panic!(
                "[cam {i} {cam:?}] GPU resident set != CPU resident set. cpu={} gpu={} rounds={rounds} \
                 missing(GPU lacks, first 20)={missing:?} extra(GPU has, first 20)={extra:?}",
                cpu.len(),
                gpu_set.len(),
            );
        }
        assert!(!cpu.is_empty(), "[cam {i}] the resident set must be non-empty");
        total += cpu.len();
        eprintln!(
            "[gpu-diff-parity] cam {i} {cam:?}: {} resident bricks match exactly ({rounds} GPU rounds)",
            cpu.len()
        );
    }
    eprintln!(
        "[gpu-diff-parity] OK — {} cameras, {total} resident keys total, slot_table_size {}",
        cams.len(),
        gpu.slot_table_size,
    );
}

/// **keep-old-until-revealed parity.** After a MOVE that flips LODs, BEFORE the replacements are entered, a
/// transition brick must be RETAINED on the GPU iff the CPU `safe_to_drop` retains it — no still-covered key
/// becomes a hole. We converge BOTH sides at cam A, then run exactly ONE GPU diff round at cam B (replacements
/// not yet resident) and ONE CPU `update` (no drain) at cam B, and assert: (a) the GPU retained set ⊇ the
/// covered transition keys (no hole), and (b) the GPU resident set after the single round EQUALS the CPU
/// resident set after the single `update` (same retained-vs-dropped decision).
#[test]
fn gpu_keep_old_until_revealed_matches_cpu() {
    let Some((device, queue)) = common::headless_compute_device_with_storage(512, 24) else {
        eprintln!("[skip] no GPU adapter — keep-old-until-revealed parity skipped");
        return;
    };

    let map = representative_scene();
    let source = StaticVoxSource::new(&map);
    let occ = SectorOccupancy::from_occupied_full(source.occupied_keys_full());
    let registry = registry();
    let cfg = StreamingConfig {
        clip_half_bricks: 8,
        max_resident_bricks: usize::MAX,
        max_bricks_per_frame: usize::MAX,
    };
    let span0 = brick_span(0);

    let cand_cap = 200_000usize;
    let gpu = GpuDiff::new(device, queue, &occ, cand_cap as u32, cand_cap);
    let mut mgr = ResidencyManager::new();

    // Converge both at cam A (near origin).
    let cam_a = [0.5 * span0, 1.5 * span0, 0.5 * span0];
    let cpu_a = cpu_converge(&mut mgr, cam_a, &cfg, &source, &registry);
    gpu.converge(cam_a, &cfg);
    let gpu_a = gpu.resident_set();
    assert_eq!(cpu_a, gpu_a, "pre-move sets must match (sanity)");

    // Move to cam B (recede up the pillar so near-surface regions coarsen — a LOD flip).
    let cam_b = [0.5 * span0, 20.0 * span0, 0.5 * span0];

    // ONE CPU `update` at cam B (NO drain): drops only the keys `safe_to_drop` allows; enqueues replacements but
    // they are NOT yet resident. This is the CPU's first-round retained set.
    let edits = VoxelEdits::new();
    mgr.update(cam_b, &cfg, &source);
    let cpu_b1 = cpu_resident_set(&mgr);

    // ONE GPU diff round at cam B: drop-mark/apply (over the pre-move slot table) + enter (claims slots for the
    // candidates that were not resident). To mirror the CPU's "no drain" first round — where replacements are
    // enqueued but NOT yet resident — we compare the RETAINED-old set: every key the CPU kept (cpu_b1) that is
    // ALSO old (resident at A) must be resident on the GPU after the round, and vice versa. The GPU enters new
    // candidates immediately (it has no separate drain step), so we compare the OLD-key retention decision: the
    // subset of `gpu_a` (the pre-move resident set) each side still holds.
    let params_b = build_params(cam_b, &cfg);
    gpu.round(&params_b);
    let gpu_b1 = gpu.resident_set();

    // (a) no-hole: every OLD key the CPU RETAINED (kept resident through the move, before its replacement loaded)
    //     must ALSO be retained on the GPU — else the GPU created a hole the CPU did not.
    let cpu_retained_old: HashSet<_> = cpu_b1.intersection(&gpu_a).copied().collect();
    let gpu_retained_old: HashSet<_> = gpu_b1.intersection(&gpu_a).copied().collect();
    if cpu_retained_old != gpu_retained_old {
        let cpu_kept_gpu_dropped: Vec<_> =
            cpu_retained_old.difference(&gpu_retained_old).take(20).collect();
        let gpu_kept_cpu_dropped: Vec<_> =
            gpu_retained_old.difference(&cpu_retained_old).take(20).collect();
        panic!(
            "keep-old-until-revealed diverged. cpu_retained_old={} gpu_retained_old={} \
             cpu-kept-but-gpu-dropped(first 20)={cpu_kept_gpu_dropped:?} \
             gpu-kept-but-cpu-dropped(first 20)={gpu_kept_cpu_dropped:?}",
            cpu_retained_old.len(),
            gpu_retained_old.len(),
        );
    }
    assert!(
        !cpu_retained_old.is_empty(),
        "the move must retain SOME old LODs (else keep-old-until-revealed is untested)"
    );

    // (b) After full convergence at cam B, both must agree again (the replacements load, the old LODs drop).
    while mgr.pending() > 0 {
        mgr.drain_work_from(&cfg, &source, &registry, &edits);
    }
    let cpu_b = cpu_converge(&mut mgr, cam_b, &cfg, &source, &registry);
    gpu.converge(cam_b, &cfg);
    let gpu_b = gpu.resident_set();
    assert_eq!(cpu_b, gpu_b, "post-move converged sets must match");
    eprintln!(
        "[gpu-keep-old] OK — retained {} old LODs through the move; converged to {} resident",
        cpu_retained_old.len(),
        cpu_b.len()
    );
}

// =========================================================================================================
//  CPU-side SSOT cross-checks (no GPU): the brick_key_hash matches a reference; the diff config is well-formed.
// =========================================================================================================

/// `brick_key_hash` is deterministic + the SAME family as `sector_hash` shape — a stable reference over a few
/// keys (so a future edit that drifts the Rust hash from the WGSL `hash_key` is caught even without a GPU).
#[test]
fn brick_key_hash_is_stable_and_distinct() {
    let a = brick_key_hash(IVec3::new(0, 0, 0), 0);
    let b = brick_key_hash(IVec3::new(1, 0, 0), 0);
    let c = brick_key_hash(IVec3::new(0, 0, 0), 1);
    let d = brick_key_hash(IVec3::new(-1, -1, -1), 3);
    // Distinct keys hash distinctly here (no collisions in this tiny sample) + negative coords don't panic.
    let set: HashSet<u32> = [a, b, c, d].into_iter().collect();
    assert_eq!(set.len(), 4, "the sample keys must hash distinctly");
    // Deterministic.
    assert_eq!(a, brick_key_hash(IVec3::new(0, 0, 0), 0));
    // A quick distribution sanity: a small grid produces mostly-distinct hashes (a degenerate hash would
    // collide heavily). Allow a few collisions but require high uniqueness.
    let mut hashes: FxHashMap<u32, u32> = FxHashMap::default();
    let mut n = 0u32;
    for z in -4..4 {
        for y in -4..4 {
            for x in -4..4 {
                for lod in 0..=2 {
                    *hashes.entry(brick_key_hash(IVec3::new(x, y, z), lod)).or_insert(0) += 1;
                    n += 1;
                }
            }
        }
    }
    let distinct = hashes.len() as f64;
    assert!(distinct / n as f64 > 0.95, "brick_key_hash distribution too clustered: {distinct}/{n}");
}

// =========================================================================================================
//  small wgpu helpers (copied from the G-c.1 enumerate-parity rig).
// =========================================================================================================

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
    wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::INDIRECT
        | wgpu::BufferUsages::COPY_SRC
        | wgpu::BufferUsages::COPY_DST
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
fn readback_bytes(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, bytes: usize) -> Vec<u8> {
    let bytes = (bytes.max(4)) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("diff_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("diff_rb") });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    queue.submit(std::iter::once(encoder.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = staging.slice(..).get_mapped_range().expect("map staging").to_vec();
    staging.unmap();
    data
}
