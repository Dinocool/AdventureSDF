//! **Stage 5 — build/destroy voxel editing correctness (headless).**
//!
//! Proves the edit delta ([`VoxelEdits`]) is the single source of truth shared by the voxelizer, the GPU
//! packer, and the CPU pick — WITHOUT the GUI. Two layers:
//!
//!   * CPU (always runs): place / remove / boundary-halo / pick-correctness against the static Cornell box.
//!     The pick is asserted to agree with an independent ground-truth world-grid DDA for several known rays.
//!   * GPU re-trace (skips cleanly without a ray-query Vulkan adapter): build the EDITED Cornell map, pack
//!     it into the SSOT GPU layout, build the per-brick AABB BLAS + TLAS, and run the real
//!     `voxel_raytrace.wgsl` `trace_one` shader — asserting a PLACED voxel is hit, a REMOVED surface voxel is
//!     passed through, and a BRICK-BOUNDARY edit updates the neighbour brick's halo (no stale voxel from the
//!     side). This closes the loop: the same edit the user makes reaches the hardware ray tracer.

use std::iter;
use std::mem;

use bevy::math::{IVec3, Vec3};
use wgpu::util::DeviceExt;

use adventure::voxel::brickmap::{BRICK_EDGE, BrickMap, VOXEL_SIZE, brick_coord_of_voxel};
use adventure::voxel::cornell::{INTERIOR, WALL, build_cornell, build_cornell_with_edits};
use adventure::voxel::edits::{
    VoxelEdits, apply_edits_to_map, dirty_bricks_for_edit, pick_voxel,
};
use adventure::voxel::gpu::{GpuBrickPatch, pack_brickmap};
use adventure::voxel::palette::{BlockId, BlockRegistry, CornellBlock};

mod common;

/// True iff the world voxel is solid in `map` (the packed/base geometry).
fn solid(map: &BrickMap, v: IVec3) -> bool {
    map.voxel_is_solid(v)
}

/// Ground-truth world-grid DDA over 0.2 m voxels (independent of the production [`pick_voxel`]): returns the
/// first solid voxel + its world-t along the ray. Mirrors the GPU shader's per-voxel stepping. Used to
/// cross-check the production pick.
fn cpu_first_solid(map: &BrickMap, ro: Vec3, rd: Vec3, t_max: f32) -> Option<(IVec3, f32)> {
    let rd = rd.normalize();
    let step = IVec3::new(rd.x.signum() as i32, rd.y.signum() as i32, rd.z.signum() as i32);
    let inv = Vec3::new(1.0 / rd.x, 1.0 / rd.y, 1.0 / rd.z);
    let mut vox = IVec3::new(
        (ro.x / VOXEL_SIZE).floor() as i32,
        (ro.y / VOXEL_SIZE).floor() as i32,
        (ro.z / VOXEL_SIZE).floor() as i32,
    );
    let next_boundary = Vec3::new(
        (vox.x + step.x.max(0)) as f32 * VOXEL_SIZE,
        (vox.y + step.y.max(0)) as f32 * VOXEL_SIZE,
        (vox.z + step.z.max(0)) as f32 * VOXEL_SIZE,
    );
    let big = f32::MAX;
    let pick = |z: bool, v: f32| if z { big } else { v };
    let mut t_max_axis = Vec3::new(
        pick(rd.x.abs() < 1e-12, (next_boundary.x - ro.x) * inv.x),
        pick(rd.y.abs() < 1e-12, (next_boundary.y - ro.y) * inv.y),
        pick(rd.z.abs() < 1e-12, (next_boundary.z - ro.z) * inv.z),
    );
    let t_delta = Vec3::new(
        pick(rd.x.abs() < 1e-12, (VOXEL_SIZE * inv.x).abs()),
        pick(rd.y.abs() < 1e-12, (VOXEL_SIZE * inv.y).abs()),
        pick(rd.z.abs() < 1e-12, (VOXEL_SIZE * inv.z).abs()),
    );
    let mut t_cur = 0.0f32;
    for _ in 0..8192 {
        if t_cur > t_max {
            return None;
        }
        if !map.voxel_block(vox).is_air() {
            return Some((vox, t_cur));
        }
        if t_max_axis.x < t_max_axis.y && t_max_axis.x < t_max_axis.z {
            t_cur = t_max_axis.x;
            t_max_axis.x += t_delta.x;
            vox.x += step.x;
        } else if t_max_axis.y < t_max_axis.z {
            t_cur = t_max_axis.y;
            t_max_axis.y += t_delta.y;
            vox.y += step.y;
        } else {
            t_cur = t_max_axis.z;
            t_max_axis.z += t_delta.z;
            vox.z += step.z;
        }
    }
    None
}

/// Build the EDITED Cornell base map (the SSOT the renderer packs) for a given delta.
fn edited_cornell(edits: &VoxelEdits) -> (BrickMap, BlockRegistry) {
    let reg = BlockRegistry::cornell();
    let map = build_cornell_with_edits(&reg, edits);
    (map, reg)
}

// --- CPU correctness --------------------------------------------------------------------------------

/// PLACE a voxel into the empty interior of the Cornell box → it appears in the edited map AND in the packed
/// GPU bricks, and a CPU pick ray now hits it at the right coord + face.
#[test]
fn place_voxel_appears_in_map_pack_and_pick() {
    // A clearly-empty interior voxel near the room centre, away from the floor boxes.
    let target = IVec3::new(INTERIOR / 2, INTERIOR / 2, 4);
    let base = build_cornell(&BlockRegistry::cornell());
    assert!(!solid(&base, target), "the target interior voxel must start empty");

    let mut edits = VoxelEdits::new();
    edits.place(target, CornellBlock::Green.id());
    let (map, reg) = edited_cornell(&edits);

    // 1. The edited map contains the placed voxel.
    assert_eq!(map.voxel_block(target), CornellBlock::Green.id(), "placed voxel present in the edited map");

    // 2. The packed GPU bricks contain it (find its brick + the haloed core cell).
    let patch = pack_brickmap(&map, &reg);
    assert!(contains_solid_voxel(&patch, target, CornellBlock::Green.id()), "placed voxel present in packed GPU bricks");

    // 3. A CPU pick ray from in front of the box (open −Z front) along +Z hits the placed voxel first.
    //    Origin centred on the target's X/Y, just outside the front; the voxel is the only thing along that
    //    column until the back wall.
    let vox_centre = (target.as_vec3() + Vec3::splat(0.5)) * VOXEL_SIZE;
    let ro = Vec3::new(vox_centre.x, vox_centre.y, -2.0);
    let hit = pick_voxel(&map, &VoxelEdits::new(), ro, Vec3::Z, 1.0e3).expect("pick must hit the placed voxel");
    assert_eq!(hit.voxel, target, "pick resolves the placed voxel");
    assert_eq!(hit.normal, IVec3::new(0, 0, -1), "entry face is the −Z face (toward the camera)");
    assert_eq!(hit.block, CornellBlock::Green.id());
}

/// REMOVE a surface voxel (the back wall) → it's gone from the packed bricks, and a ray that hit it now
/// passes THROUGH to whatever's behind (here: nothing within the box → it exits, hitting the far wall voxel
/// behind the removed one if the wall is >1 voxel thick).
#[test]
fn remove_surface_voxel_passes_through() {
    // The back wall (+Z) is WALL voxels thick at z ∈ [INTERIOR, INTERIOR+WALL). Pick a back-wall column.
    let col_x = INTERIOR / 2;
    let col_y = INTERIOR / 2;
    let front_wall_voxel = IVec3::new(col_x, col_y, INTERIOR); // the −Z face voxel of the back wall

    let base = build_cornell(&BlockRegistry::cornell());
    assert!(solid(&base, front_wall_voxel), "back-wall surface voxel must start solid");

    // A +Z ray down the column hits the front face of the back wall first (voxel z = INTERIOR).
    let vox_centre = (front_wall_voxel.as_vec3() + Vec3::splat(0.5)) * VOXEL_SIZE;
    let ro = Vec3::new(vox_centre.x, vox_centre.y, -2.0);
    let before = pick_voxel(&base, &VoxelEdits::new(), ro, Vec3::Z, 1.0e3).expect("ray hits the back wall");
    assert_eq!(before.voxel, front_wall_voxel, "before removal the ray hits the wall's front face");

    // Remove that surface voxel.
    let mut edits = VoxelEdits::new();
    edits.remove(front_wall_voxel);
    let (map, reg) = edited_cornell(&edits);

    // It's gone from the packed bricks.
    let patch = pack_brickmap(&map, &reg);
    assert!(!contains_solid_voxel_any(&patch, front_wall_voxel), "removed voxel is absent from packed bricks");

    // The ray now passes through to the next wall voxel behind (z = INTERIOR+1, since the wall is >=2 thick).
    const _: () = assert!(WALL >= 2, "this test assumes a >=2-voxel-thick back wall");
    let after = pick_voxel(&map, &VoxelEdits::new(), ro, Vec3::Z, 1.0e3).expect("ray hits the voxel behind");
    assert_eq!(after.voxel, IVec3::new(col_x, col_y, INTERIOR + 1), "ray now reaches the voxel behind the hole");
    assert!(after.t > before.t, "the new hit is farther along the ray (passed through the hole)");
}

/// A boundary edit makes the right neighbour bricks dirty (the halo SSOT), and the neighbour brick's packed
/// HALO actually carries the edited voxel (no stale halo cell).
#[test]
fn boundary_edit_updates_neighbour_halo() {
    // Find a world voxel that sits on a brick FACE inside the box so an edit there touches two bricks. The
    // floor SHELL (y ∈ [-WALL, 0)) is solid white; choose a floor voxel whose X and Z lie on a brick boundary.
    // Brick edge is 8: world voxel x = 8 is the low face of brick (1,..) and high face of brick (0,..) → an
    // edit there dirties both. y = -1 is a solid floor-shell voxel. REMOVE it and check both bricks update.
    let v = IVec3::new(BRICK_EDGE, -1, BRICK_EDGE); // x and z on brick faces, y in the solid floor shell
    let base = build_cornell(&BlockRegistry::cornell());
    assert!(solid(&base, v), "the chosen boundary floor voxel must start solid");

    // Dirty set must include the owner AND the boundary neighbours.
    let dirty = dirty_bricks_for_edit(v);
    let owner = brick_coord_of_voxel(v);
    assert!(dirty.contains(&owner), "owner brick is dirty");
    // x and z are on the LOW face (local 0) → the −X and −Z neighbours are dirty too.
    assert!(dirty.contains(&(owner + IVec3::new(-1, 0, 0))), "−X neighbour dirty");
    assert!(dirty.contains(&(owner + IVec3::new(0, 0, -1))), "−Z neighbour dirty");

    // Remove the voxel; in the re-baked map the neighbour brick's HALO border (which reads this world voxel)
    // must now be AIR — proving the neighbour was re-voxelized, not left stale.
    let mut edits = VoxelEdits::new();
    edits.remove(v);
    let (map, reg) = edited_cornell(&edits);
    assert_eq!(map.voxel_block(v), BlockId::AIR, "the boundary voxel is removed in the edited map");

    // Pack and inspect the −X neighbour brick's halo: its +X halo border cell adjacent to `v` reads world
    // voxel `v`, which must now be air. We verify via the packed buffer: the neighbour brick's slice, at the
    // halo index for the border cell mapping to `v`, must be 0.
    let patch = pack_brickmap(&map, &reg);
    let nbr = owner + IVec3::new(-1, 0, 0);
    assert_eq!(halo_cell_for_world_voxel(&patch, nbr, v), Some(0), "−X neighbour halo cell for the removed voxel is air (re-baked, not stale)");

    // Sanity: before the edit, that same neighbour halo cell was solid (so the test isn't vacuous).
    let base_patch = pack_brickmap(&base, &reg);
    let was = halo_cell_for_world_voxel(&base_patch, nbr, v);
    assert!(matches!(was, Some(id) if id != 0), "before removal the neighbour halo carried the solid voxel, got {was:?}");
}

/// The production [`pick_voxel`] agrees with the independent ground-truth DDA for several known rays into the
/// Cornell box (first-solid voxel + ~t), and the entry face is a unit axis normal.
#[test]
fn pick_matches_ground_truth_for_known_rays() {
    let map = build_cornell(&BlockRegistry::cornell());
    let edits = VoxelEdits::new();

    // A spread of rays from outside the open −Z front, aimed at different interior targets.
    let interior_m = INTERIOR as f32 * VOXEL_SIZE;
    let rays: [(Vec3, Vec3); 4] = [
        // Straight +Z at the back wall.
        (Vec3::new(interior_m * 0.5, interior_m * 0.5, -2.0), Vec3::Z),
        // Down-and-in: hits the floor or a box.
        (Vec3::new(interior_m * 0.5, interior_m * 0.9, -2.0), Vec3::new(0.05, -0.4, 1.0)),
        // Toward the left (red, −X) wall region.
        (Vec3::new(interior_m * 0.5, interior_m * 0.5, -2.0), Vec3::new(-0.5, 0.0, 1.0)),
        // Toward the right (green, +X) wall region.
        (Vec3::new(interior_m * 0.5, interior_m * 0.5, -2.0), Vec3::new(0.5, 0.0, 1.0)),
    ];
    for (i, &(ro, rd)) in rays.iter().enumerate() {
        let rd = rd.normalize();
        let gt = cpu_first_solid(&map, ro, rd, 1.0e3);
        let hit = pick_voxel(&map, &edits, ro, rd, 1.0e3);
        match (gt, hit) {
            (Some((gv, gt_t)), Some(h)) => {
                assert_eq!(h.voxel, gv, "ray {i}: pick voxel must match ground truth");
                assert!((h.t - gt_t).abs() <= VOXEL_SIZE + 1e-3, "ray {i}: pick t {} ~ gt {}", h.t, gt_t);
                // The normal is a unit axis vector.
                let n = h.normal;
                assert_eq!(n.abs().element_sum(), 1, "ray {i}: face normal is a single axis, got {n}");
            }
            (None, None) => {} // both miss — fine
            (g, p) => panic!("ray {i}: pick/ground-truth disagree on hit-vs-miss: gt={g:?} pick={p:?}"),
        }
    }
}

/// `apply_edits_to_map` (the map-wide overlay used as the worldgen-path SSOT) agrees with
/// `build_cornell_with_edits` (the Cornell-path overlay) on the shared voxels — both apply `base unless
/// overridden`, so the two scene paths can't diverge.
#[test]
fn map_overlay_matches_cornell_overlay() {
    let reg = BlockRegistry::cornell();
    let base = build_cornell(&reg);
    let mut edits = VoxelEdits::new();
    edits.place(IVec3::new(INTERIOR / 2, INTERIOR / 2, 5), CornellBlock::Red.id());
    edits.remove(IVec3::new(INTERIOR / 2, INTERIOR / 2, INTERIOR)); // dig the back wall

    let via_map = apply_edits_to_map(&base, &edits);
    let via_cornell = build_cornell_with_edits(&reg, &edits);

    // Both agree on the placed + removed voxels.
    assert_eq!(via_map.voxel_block(IVec3::new(INTERIOR / 2, INTERIOR / 2, 5)), CornellBlock::Red.id());
    assert_eq!(via_cornell.voxel_block(IVec3::new(INTERIOR / 2, INTERIOR / 2, 5)), CornellBlock::Red.id());
    assert_eq!(via_map.voxel_block(IVec3::new(INTERIOR / 2, INTERIOR / 2, INTERIOR)), BlockId::AIR);
    assert_eq!(via_cornell.voxel_block(IVec3::new(INTERIOR / 2, INTERIOR / 2, INTERIOR)), BlockId::AIR);
}

// --- packed-buffer inspection helpers ---------------------------------------------------------------

/// The LOD0 haloed edge (BRICK_EDGE + 2) — the only LOD `pack_brickmap` emits.
const HALO_EDGE: i32 = BRICK_EDGE + 2;

/// True iff the packed patch has a brick containing `world_voxel` as a solid CORE cell with exactly `block`.
fn contains_solid_voxel(patch: &GpuBrickPatch, world_voxel: IVec3, block: BlockId) -> bool {
    core_cell(patch, world_voxel) == Some(block.0 as u32) && !block.is_air()
}

/// True iff the packed patch has `world_voxel` as a solid CORE cell of ANY non-air block.
fn contains_solid_voxel_any(patch: &GpuBrickPatch, world_voxel: IVec3) -> bool {
    matches!(core_cell(patch, world_voxel), Some(id) if id != 0)
}

/// The packed CORE-cell block id at `world_voxel` (the cell in the brick that OWNS the voxel), or `None` if
/// that brick isn't packed.
fn core_cell(patch: &GpuBrickPatch, world_voxel: IVec3) -> Option<u32> {
    let owner = brick_coord_of_voxel(world_voxel);
    let origin = owner * BRICK_EDGE;
    let local = world_voxel - origin; // [0, BRICK_EDGE)
    // Core cells are at haloed index (local + 1).
    let hx = local.x + 1;
    let hy = local.y + 1;
    let hz = local.z + 1;
    let idx = (hx + hy * HALO_EDGE + hz * HALO_EDGE * HALO_EDGE) as usize;
    let meta = patch.metas.iter().find(|m| {
        m.voxel_origin == [origin.x, origin.y, origin.z]
    })?;
    Some(patch.voxels[meta.voxel_offset as usize + idx])
}

/// The packed HALO-border cell of brick `brick_coord` that maps to world voxel `world_voxel` (which must lie
/// OUTSIDE `brick_coord`'s core, in its 1-voxel halo). Returns the stored block id, or `None` if the brick
/// isn't packed / the voxel isn't in this brick's halo range.
fn halo_cell_for_world_voxel(patch: &GpuBrickPatch, brick_coord: IVec3, world_voxel: IVec3) -> Option<u32> {
    let origin = brick_coord * BRICK_EDGE;
    let local = world_voxel - origin; // expected in [-1, BRICK_EDGE] for a halo/core cell
    if local.x < -1 || local.x > BRICK_EDGE || local.y < -1 || local.y > BRICK_EDGE || local.z < -1 || local.z > BRICK_EDGE {
        return None;
    }
    let hx = local.x + 1;
    let hy = local.y + 1;
    let hz = local.z + 1;
    let idx = (hx + hy * HALO_EDGE + hz * HALO_EDGE * HALO_EDGE) as usize;
    let meta = patch.metas.iter().find(|m| m.voxel_origin == [origin.x, origin.y, origin.z])?;
    Some(patch.voxels[meta.voxel_offset as usize + idx])
}

// --- GPU re-trace -----------------------------------------------------------------------------------

/// Mirror of the WGSL `Hit` struct (binding 5) — same layout as `voxel_raytrace_gpu.rs`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
struct GpuHit {
    hit: u32,
    block_id: u32,
    prim: u32,
    t: f32,
    color: [f32; 4],
    normal: [f32; 3],
    shadowed: u32,
    direct: [f32; 3],
    _p0: u32,
    indirect: [f32; 3],
    _p1: u32,
    emissive_out: [f32; 3],
    _p2: u32,
}

/// Mirror of the WGSL `RayUniform` (binding 4).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RayUniform {
    origin: [f32; 3],
    t_min: f32,
    dir: [f32; 3],
    t_max: f32,
}

/// Build the BLAS/TLAS + pipeline for a packed patch and return a closure that traces one ray on the real
/// `trace_one` shader. Kept inside the test so the GPU objects live for the closure's lifetime.
struct GpuTracer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    light_bg: wgpu::BindGroup,
    ray_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    read_buf: wgpu::Buffer,
    // Keep-alive for the accel structures the bind group references.
    _blas: wgpu::Blas,
    _tlas: wgpu::Tlas,
    _bufs: Vec<wgpu::Buffer>,
}

impl GpuTracer {
    fn new(device: wgpu::Device, queue: wgpu::Queue, patch: &GpuBrickPatch) -> Self {
        let n = patch.brick_count() as u32;
        let aabb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("aabbs"),
            contents: bytemuck::cast_slice(&patch.aabbs),
            usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE,
        });
        let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("metas"),
            contents: bytemuck::cast_slice(&patch.metas),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let voxel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxels"),
            contents: bytemuck::cast_slice(&patch.voxels),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let palette_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("palette"),
            contents: bytemuck::cast_slice(&patch.palette),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let size_desc = wgpu::BlasAABBGeometrySizeDescriptor {
            primitive_count: n,
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("blas"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::AABBs { descriptors: vec![size_desc.clone()] },
        );
        let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("tlas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            max_instances: 1,
        });
        tlas[0] = Some(wgpu::TlasInstance::new(
            &blas,
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            0,
            0xff,
        ));

        let ray_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ray"),
            size: mem::size_of::<RayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hit"),
            size: mem::size_of::<GpuHit>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read"),
            size: mem::size_of::<GpuHit>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let src = common::voxel_raytrace_shader_src();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel_raytrace"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("trace_one"),
            layout: None,
            module: &shader,
            entry_point: Some("trace_one"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::AccelerationStructure(&tlas) },
                wgpu::BindGroupEntry { binding: 1, resource: meta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: voxel_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: palette_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: ray_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
            ],
        });
        let light_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lighting"),
            contents: bytemuck::bytes_of(&adventure::voxel::raytrace::LightingUniformData::default()),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let sky_buf = common::sky_uniform_buffer(&device);
        let light_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lighting_bg"),
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[
                wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: sky_buf.as_entire_binding() },
            ],
        });

        let mut build = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("build") });
        build.build_acceleration_structures(
            iter::once(&wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::AabbGeometries(vec![wgpu::BlasAabbGeometry {
                    size: &size_desc,
                    stride: mem::size_of::<adventure::voxel::gpu::GpuBrickAabb>() as wgpu::BufferAddress,
                    aabb_buffer: &aabb_buf,
                    primitive_offset: 0,
                }]),
            }),
            iter::once(&tlas),
        );
        queue.submit(Some(build.finish()));

        Self {
            device,
            queue,
            pipeline,
            bind_group,
            light_bg,
            ray_buf,
            out_buf,
            read_buf,
            _blas: blas,
            _tlas: tlas,
            _bufs: vec![aabb_buf, meta_buf, voxel_buf, palette_buf, light_buf],
        }
    }

    fn trace(&self, ro: Vec3, rd: Vec3) -> GpuHit {
        let ray = RayUniform { origin: ro.into(), t_min: 0.0, dir: rd.normalize().into(), t_max: 1.0e3 };
        self.queue.write_buffer(&self.ray_buf, 0, bytemuck::bytes_of(&ray));
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, Some(&self.bind_group), &[]);
            cpass.set_bind_group(1, Some(&self.light_bg), &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.read_buf, 0, mem::size_of::<GpuHit>() as u64);
        self.queue.submit(Some(encoder.finish()));
        let slice = self.read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
        let data = slice.get_mapped_range().unwrap();
        let gpu: GpuHit = *bytemuck::from_bytes(&data);
        drop(data);
        self.read_buf.unmap();
        gpu
    }
}

/// GPU re-trace: a PLACED voxel is hit by the real shader; a REMOVED surface voxel is passed through; and a
/// BRICK-BOUNDARY edit re-traces cleanly from the side (the neighbour halo is fresh, no stale voxel).
#[test]
fn gpu_retrace_reflects_edits() {
    let Some((device, queue)) = common::headless_ray_query_device() else {
        eprintln!("no ray-query device — skipping gpu_retrace_reflects_edits");
        return;
    };

    let interior_m = INTERIOR as f32 * VOXEL_SIZE;

    // --- 1. PLACE a voxel in the empty interior; the shader must hit it. ---
    let placed = IVec3::new(INTERIOR / 2, INTERIOR / 2, 4);
    let mut edits = VoxelEdits::new();
    edits.place(placed, CornellBlock::Green.id());
    let (map, reg) = edited_cornell(&edits);
    let patch = pack_brickmap(&map, &reg);
    let tracer = GpuTracer::new(device, queue, &patch);

    let vc = (placed.as_vec3() + Vec3::splat(0.5)) * VOXEL_SIZE;
    let ro = Vec3::new(vc.x, vc.y, -2.0);
    let hit = tracer.trace(ro, Vec3::Z);
    eprintln!("placed-voxel GPU hit: {hit:?}");
    assert_eq!(hit.hit, 1, "GPU must hit the placed voxel");
    assert_eq!(hit.block_id, CornellBlock::Green.id().0 as u32, "GPU hit is the placed Green block");
    // The hit is at the placed voxel's −Z face (~ z = placed.z * VOXEL_SIZE; ro at -2 → t ≈ that + 2).
    let expect_t = placed.z as f32 * VOXEL_SIZE + 2.0;
    assert!((hit.t - expect_t).abs() < VOXEL_SIZE + 0.05, "GPU hit-t {} ≈ {}", hit.t, expect_t);

    // The CPU pick agrees with the GPU on the same ray (pick == render).
    let cpu = pick_voxel(&map, &VoxelEdits::new(), ro, Vec3::Z, 1.0e3).expect("CPU pick hits the placed voxel");
    assert_eq!(cpu.voxel, placed, "CPU pick == GPU hit (placed voxel)");

    // --- 2. REMOVE a back-wall surface voxel; the shader now passes through to the voxel behind. ---
    let wall_v = IVec3::new(INTERIOR / 2, INTERIOR / 2, INTERIOR);
    let mut edits2 = VoxelEdits::new();
    edits2.remove(wall_v);
    let (map2, reg2) = edited_cornell(&edits2);
    let patch2 = pack_brickmap(&map2, &reg2);
    let tracer2 = GpuTracer::new(tracer.device.clone(), tracer.queue.clone(), &patch2);

    let wvc = (wall_v.as_vec3() + Vec3::splat(0.5)) * VOXEL_SIZE;
    let ro2 = Vec3::new(wvc.x, wvc.y, -2.0);
    let hit2 = tracer2.trace(ro2, Vec3::Z);
    eprintln!("removed-surface GPU hit: {hit2:?}");
    assert_eq!(hit2.hit, 1, "GPU still hits the wall behind the hole");
    // The hit must be FARTHER than the original wall face (passed through the removed voxel).
    let removed_face_t = wall_v.z as f32 * VOXEL_SIZE + 2.0;
    assert!(hit2.t > removed_face_t + 0.5 * VOXEL_SIZE, "GPU hit-t {} must be past the removed face {}", hit2.t, removed_face_t);

    // --- 3. BRICK-BOUNDARY edit: PLACE a voxel right on a Z brick face inside the box and re-trace +Z
    //        through the open front. The ray crosses the brick boundary (brick z=0 → z=1) exactly at the
    //        placed voxel, so the neighbour brick's halo must carry it (else a stale halo → a miss/seam). ---
    let boundary_v = IVec3::new(INTERIOR / 2, INTERIOR / 2, BRICK_EDGE); // z on a brick face inside the room
    let base = build_cornell(&BlockRegistry::cornell());
    assert!(!solid(&base, boundary_v), "the boundary placement target must start empty interior");
    // The column in front of it (z in [0, BRICK_EDGE)) must be empty so the ray reaches the boundary.
    for z in 0..BRICK_EDGE {
        assert!(!solid(&base, IVec3::new(INTERIOR / 2, INTERIOR / 2, z)), "front column must be clear");
    }
    let mut edits3 = VoxelEdits::new();
    edits3.place(boundary_v, CornellBlock::Red.id());
    let (map3, reg3) = edited_cornell(&edits3);
    let patch3 = pack_brickmap(&map3, &reg3);
    let tracer3 = GpuTracer::new(tracer.device.clone(), tracer.queue.clone(), &patch3);

    let bvc = (boundary_v.as_vec3() + Vec3::splat(0.5)) * VOXEL_SIZE;
    let _ = interior_m;
    let ro3 = Vec3::new(bvc.x, bvc.y, -2.0);
    let hit3 = tracer3.trace(ro3, Vec3::Z);
    eprintln!("boundary-edit GPU hit (+Z across brick face): {hit3:?}");
    assert_eq!(hit3.hit, 1, "GPU must hit the boundary-placed voxel across the brick face (fresh neighbour halo)");
    assert_eq!(hit3.block_id, CornellBlock::Red.id().0 as u32, "boundary hit is the placed Red block");
    // The hit is at the placed voxel's −Z face (its brick boundary), t ≈ z*VOXEL_SIZE + 2.
    let bt = boundary_v.z as f32 * VOXEL_SIZE + 2.0;
    assert!((hit3.t - bt).abs() < VOXEL_SIZE + 0.05, "boundary hit-t {} ≈ {}", hit3.t, bt);
    // CPU pick agrees on this ray too.
    let cpu3 = pick_voxel(&map3, &VoxelEdits::new(), ro3, Vec3::Z, 1.0e3).expect("CPU pick hits boundary voxel");
    assert_eq!(cpu3.voxel, boundary_v, "CPU pick == GPU hit (boundary voxel)");
    assert_eq!(cpu3.normal, IVec3::new(0, 0, -1), "boundary hit face is −Z");
}
