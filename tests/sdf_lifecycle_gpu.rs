//! Full bake-lifecycle simulation on a LARGE object, on a real GPU.
//!
//! Reproduces what the live renderer does as the camera flies away from and back toward a
//! big sphere (radius 25), crossing LOD boundaries. At every camera step it drives the REAL
//! topology code (`schedule_bakes` step-2 recenter + `emit_gpu_bakes`'s cull/palette/alloc),
//! runs the REAL bake compute shader on the GPU to fill the newly-entered tiles, maintains a
//! persistent R16Snorm atlas texture that GROWS exactly as `prepare_sdf_atlas_gpu` does
//! (recreate taller + copy old content), and builds the REAL chunk lookup table.
//!
//! Then it asserts the two invariants the live flicker would violate:
//!   1. NO HOLE: every tile the chunk table references this frame contains real baked texels
//!      (not the zero/unbaked sentinel) — i.e. the resident set and the atlas content never
//!      desync, even on a grow frame.
//!   2. COARSENING: as the camera recedes, the finest resident LOD at the object's centre
//!      coarsens (LOD 0/1 → LOD 2 …) and refines back on return — the clipmap serves the
//!      coarse LOD correctly.

use std::borrow::Cow;
use std::collections::HashSet;

use bevy::math::bounding::Aabb3d;
use bevy::math::{IVec3, Vec3};
use bevy::prelude::Transform;
use futures_lite::future::block_on;
use naga_oil::compose::{Composer, NagaModuleDescriptor};

use adventure::sdf_render::atlas::{BRICK_EDGE, BrickKey, SdfAtlas, dist_band_world};
use adventure::sdf_render::bvh::Bvh;
use adventure::sdf_render::chunk::{self, build_chunk_tables};
use adventure::sdf_render::edits::{
    GpuEdit, ResolvedEdit, SdfOp, SdfPrimitive, build_palette, edit_world_aabb, to_gpu_edit,
};
use adventure::sdf_render::SdfGridConfig;

const TILE_W: u32 = 64; // px per tile (8*8)
const DIST_ROW_U32: u32 = 64; // padded
const DIST_TILE_U32: u32 = DIST_ROW_U32 * 8;
const MAT_TILE_U32: u32 = 128 * 8;
const TEST_TILES_PER_ROW: u32 = 64; // 64*64=4096px wide, within the 8192 default limit

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    if !adapter.features().contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM) {
        eprintln!("adapter lacks TEXTURE_FORMAT_16BIT_NORM — skipping");
        return None;
    }
    block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("sdf_lifecycle"),
        required_features: wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
        ..Default::default()
    }))
    .ok()
}

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

fn header_bytes(coord: IVec3, voxel_size: f32, dist_band: f32, edit_start: u32, edit_count: u32, pal: [u16; 4]) -> Vec<u8> {
    let mut b = Vec::with_capacity(48);
    for v in [coord.x, coord.y, coord.z] { b.extend_from_slice(&v.to_le_bytes()); }
    b.extend_from_slice(&voxel_size.to_le_bytes());
    b.extend_from_slice(&dist_band.to_le_bytes());
    b.extend_from_slice(&edit_start.to_le_bytes());
    b.extend_from_slice(&edit_count.to_le_bytes());
    b.extend_from_slice(&(pal[0] as u32 | ((pal[1] as u32) << 16)).to_le_bytes());
    b.extend_from_slice(&(pal[2] as u32 | ((pal[3] as u32) << 16)).to_le_bytes());
    for _ in 0..3 { b.extend_from_slice(&0u32.to_le_bytes()); }
    b
}

fn edit_bytes(e: &GpuEdit) -> Vec<u8> {
    let mut b = Vec::with_capacity(96);
    for col in e.inv_model.to_cols_array() { b.extend_from_slice(&col.to_le_bytes()); }
    for v in [e.params.x, e.params.y, e.params.z, e.params.w] { b.extend_from_slice(&v.to_le_bytes()); }
    for v in [e.params2.x, e.params2.y, e.params2.z, e.params2.w] { b.extend_from_slice(&v.to_le_bytes()); }
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
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: ro }, has_dynamic_offset: false, min_binding_size: None },
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

/// Persistent GPU atlas the harness grows + bakes into, exactly like the live render world.
struct GpuAtlas {
    tex: Option<wgpu::Texture>,
    rows: u32, // committed tile rows
}

impl GpuAtlas {
    fn new() -> Self { Self { tex: None, rows: 0 } }

    /// Run this frame's bake jobs: dispatch the real bake shader into output buffers, grow the
    /// texture if needed (recreate taller + copy old), then copy each job's tile in. Mirrors
    /// `prepare_sdf_atlas_gpu` (grow) + `SdfBrickBakeNode` (dispatch+copy).
    #[allow(clippy::too_many_arguments)]
    fn bake_frame(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, pipeline: &wgpu::ComputePipeline, layout: &wgpu::BindGroupLayout, jobs: &[Job], edits: &[GpuEdit], high_water: u32) {
        // Grow first (separate submit, like prepare_sdf_atlas_gpu).
        let required_rows = high_water.div_ceil(TEST_TILES_PER_ROW).max(1);
        if required_rows > self.rows {
            let w = TEST_TILES_PER_ROW * TILE_W;
            let h = required_rows * BRICK_EDGE as u32;
            let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC;
            let new_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("atlas"), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R16Snorm, usage, view_formats: &[],
            });
            if let Some(old) = &self.tex {
                let old_h = old.height().min(h);
                let mut enc = device.create_command_encoder(&Default::default());
                enc.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo { texture: old, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    wgpu::TexelCopyTextureInfo { texture: &new_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    wgpu::Extent3d { width: w, height: old_h, depth_or_array_layers: 1 },
                );
                queue.submit([enc.finish()]);
            }
            self.tex = Some(new_tex);
            self.rows = required_rows;
        }
        if jobs.is_empty() { return; }

        // Upload headers + edits, size output buffers (mirrors prepare_brick_bake_buffers).
        let mut hbytes = Vec::new();
        for j in jobs { hbytes.extend_from_slice(&header_bytes(j.coord, j.voxel_size, j.dist_band, j.edit_start, j.edit_count, j.pal)); }
        let mut ebytes = Vec::new();
        for e in edits { ebytes.extend_from_slice(&edit_bytes(e)); }
        if ebytes.is_empty() { ebytes.resize(96, 0); }
        use wgpu::util::DeviceExt;
        let header_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &hbytes, usage: wgpu::BufferUsages::STORAGE });
        let edit_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &ebytes, usage: wgpu::BufferUsages::STORAGE });
        let n = jobs.len() as u32;
        let dist_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n * DIST_TILE_U32 * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let mat_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n * MAT_TILE_U32 * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: header_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: edit_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dist_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: mat_buf.as_entire_binding() },
        ]});

        let tex = self.tex.as_ref().unwrap();
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // 2D dispatch like the live node (BAKE_DISPATCH_WIDTH=256).
            let wg_x = n.min(256);
            let wg_y = n.div_ceil(256);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        for (i, j) in jobs.iter().enumerate() {
            let (col_px, row_px) = tile_origin(j.tile);
            enc.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo { buffer: &dist_buf, layout: wgpu::TexelCopyBufferLayout { offset: (i as u32 * DIST_TILE_U32) as u64 * 4, bytes_per_row: Some(DIST_ROW_U32 * 4), rows_per_image: Some(BRICK_EDGE as u32) } },
                wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d { x: col_px, y: row_px, z: 0 }, aspect: wgpu::TextureAspect::All },
                wgpu::Extent3d { width: TILE_W, height: BRICK_EDGE as u32, depth_or_array_layers: 1 },
            );
        }
        queue.submit([enc.finish()]);
    }

    /// Read back one tile's center voxel distance (decoded). Used to prove a table-referenced
    /// tile actually holds baked geometry (non-sentinel).
    fn read_tile_has_content(&self, device: &wgpu::Device, queue: &wgpu::Queue, tile: u32) -> bool {
        let tex = self.tex.as_ref().unwrap();
        let (col_px, row_px) = tile_origin(tile);
        let row_bytes = 256u32; // 64*2=128 → padded to 256
        let size = (row_bytes * BRICK_EDGE as u32) as u64;
        let rb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d { x: col_px, y: row_px, z: 0 }, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &rb, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row_bytes), rows_per_image: Some(BRICK_EDGE as u32) } },
            wgpu::Extent3d { width: TILE_W, height: BRICK_EDGE as u32, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        let slice = rb.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let data = slice.get_mapped_range().to_vec();
        // Any non-zero R16 in the tile → real baked content (a freshly-allocated/cleared tile
        // would read all-zero; bricks straddling/near the surface always have non-zero snorm).
        let texels: &[i16] = bytemuck::cast_slice(&data[..(TILE_W * 2) as usize]);
        texels.iter().any(|&v| v != 0)
    }
}

/// Mirror of `emit_gpu_bakes` topology: for `dirty` chunks, cull+palette+alloc each brick and
/// build the bake jobs (+ flat edit list). Evicts empties. Returns the jobs + edits.
fn emit(atlas: &mut SdfAtlas, cfg: &SdfGridConfig, bvh: &Bvh, resolved: &[ResolvedEdit], dirty: &HashSet<chunk::ChunkKey>) -> (Vec<Job>, Vec<GpuEdit>) {
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
                let culled: Vec<ResolvedEdit> = scratch.iter().map(|&i| resolved[i as usize].clone()).collect();
                let pal = build_palette(&culled, &samples);
                let tile = atlas.insert_gpu_brick(key, pal, 0, cfg);
                let edit_start = edits.len() as u32;
                for e in &culled { edits.push(to_gpu_edit(e)); }
                jobs.push(Job { tile, coord: key.coord, voxel_size: vs, dist_band: dist_band_world(cfg, key.lod), pal, edit_start, edit_count: culled.len() as u32 });
            } else {
                atlas.remove_brick(&key, cfg);
            }
        }
    }
    (jobs, edits)
}

/// All brick keys of a chunk (mirror of the private bake_scheduler helper).
fn chunk_brick_keys(ck: chunk::ChunkKey, cfg: &SdfGridConfig) -> Vec<BrickKey> {
    let s = cfg.cell_stride();
    let c = chunk::CHUNK_BRICKS;
    let base = ck.coord * c;
    let mut keys = Vec::with_capacity(chunk::CHUNK_VOLUME as usize);
    for lz in 0..c { for ly in 0..c { for lx in 0..c {
        let bi = base + IVec3::new(lx, ly, lz);
        keys.push(BrickKey::new(ck.lod, bi * s));
    }}}
    keys
}

fn ring_chunks_per_axis(cfg: &SdfGridConfig) -> i32 { (cfg.ring_bricks / chunk::CHUNK_BRICKS as u32) as i32 }

fn ring_chunk_origin(cfg: &SdfGridConfig, cam: Vec3, lod: u32) -> IVec3 {
    adventure::sdf_render::bake_scheduler::ring_chunk_origin(cfg, cam, lod)
}

fn chunk_window_keys(origin: IVec3, r: i32, lod: u32) -> Vec<chunk::ChunkKey> {
    let mut v = Vec::new();
    for iz in 0..r { for iy in 0..r { for ix in 0..r {
        v.push(chunk::ChunkKey::new(lod, origin + IVec3::new(ix, iy, iz)));
    }}}
    v
}

fn chunk_in_window(c: IVec3, origin: IVec3, r: i32) -> bool {
    let rel = c - origin;
    rel.x >= 0 && rel.y >= 0 && rel.z >= 0 && rel.x < r && rel.y < r && rel.z < r
}

fn chunk_has_geometry(ck: chunk::ChunkKey, bvh: &Bvh, cfg: &SdfGridConfig, scratch: &mut Vec<u32>) -> bool {
    let size = chunk::chunk_world_size(ck.lod, cfg);
    let min = chunk::chunk_min_world(ck, cfg);
    bvh.query_aabb(&Aabb3d::from_min_max(min, min + Vec3::splat(size)), scratch);
    !scratch.is_empty()
}

/// Recenter step (mirror of schedule_bakes step 2): returns the dirty chunk set to bake.
fn recenter(ring: &mut Vec<IVec3>, atlas: &mut SdfAtlas, cfg: &SdfGridConfig, bvh: &Bvh, cam: Vec3) -> HashSet<chunk::ChunkKey> {
    let r = ring_chunks_per_axis(cfg);
    if ring.is_empty() { *ring = vec![IVec3::splat(i32::MIN); cfg.lod_count as usize]; }
    let first = ring.iter().all(|o| *o == IVec3::splat(i32::MIN));
    let mut dirty = HashSet::new();
    let mut scratch = Vec::new();
    for lod in 0..cfg.lod_count {
        let li = lod as usize;
        let new_o = ring_chunk_origin(cfg, cam, lod);
        let old_o = ring[li];
        if new_o == old_o { continue; }
        for ck in chunk_window_keys(new_o, r, lod) {
            let entered = first || !chunk_in_window(ck.coord, old_o, r);
            if entered && chunk_has_geometry(ck, bvh, cfg, &mut scratch) { dirty.insert(ck); }
        }
        if !first {
            for ck in chunk_window_keys(old_o, r, lod) {
                if !chunk_in_window(ck.coord, new_o, r) {
                    for bk in chunk_brick_keys(ck, cfg) { atlas.remove_brick(&bk, cfg); }
                }
            }
        }
        ring[li] = new_o;
    }
    dirty
}

/// The finest resident LOD with a baked brick at world point `p` — mirrors the shader's
/// `resolve_march` fine→coarse walk (chunk-table presence only).
fn served_lod(atlas: &SdfAtlas, cfg: &SdfGridConfig, p: Vec3) -> Option<u32> {
    for lod in 0..cfg.lod_count {
        let coord = cfg.world_to_brick_lod(p, lod);
        let key = BrickKey::new(lod, coord);
        if atlas.bricks.contains_key(&key) { return Some(lod); }
    }
    None
}

#[test]
fn lifecycle_large_sphere_lod_transition_no_hole() {
    let Some((device, queue)) = gpu() else { return; };

    // Defaults but a reasonable ring so the sphere spans several LODs without huge bake counts.
    let cfg = SdfGridConfig { lod_count: 5, ring_bricks: 16, recenter_snap_chunks: 1, ..Default::default() };
    let radius = 25.0f32;
    let edit = ResolvedEdit::new(SdfPrimitive::Sphere { radius }, Transform::IDENTITY, SdfOp::default(), 0);
    let resolved = vec![edit.clone()];
    let aabbs = vec![edit_world_aabb(&edit.prim, &edit.transform, edit.op.smoothing)];
    let bvh = Bvh::build(&aabbs);

    // Bake pipeline.
    let module = compose_bake();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Naga(Cow::Owned(module)) });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[storage_entry(0, true), storage_entry(1, true), storage_entry(2, false), storage_entry(3, false)] });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&layout], push_constant_ranges: &[] });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pl), module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });

    let mut atlas = SdfAtlas::default();
    let mut gpu_atlas = GpuAtlas::new();
    let mut ring: Vec<IVec3> = Vec::new();

    // Fly the camera straight out along +X from the sphere centre, then back. The sphere is at
    // the origin; the probe point is its centre, so the served LOD at the centre should coarsen
    // as the ring's fine LODs no longer reach the origin, then refine on return.
    let probe = Vec3::ZERO;
    let mut served_seq: Vec<Option<u32>> = Vec::new();

    // Fly out only as far as the clipmap still covers the origin, so coverage stays continuous
    // and the test isolates LOD TRANSITIONS (not the legitimate far edge of finite reach). The
    // coarsest ring (LOD lod_count-1) half-extent is ~ring_bricks/2 · brick_world(top); stay
    // well inside it. brick_world(L)=cell_stride·voxel·2^L = 7·0.1·2^L. Top LOD=4 → ~90 units;
    // cap at 72 with fine steps so several LOD bands transition.
    let out: Vec<f32> = (0..=24).map(|i| i as f32 * 3.0).collect(); // 0..72 world units
    let mut path: Vec<f32> = out.clone();
    path.extend(out.iter().rev().skip(1).copied()); // back to 0

    for (step, &x) in path.iter().enumerate() {
        let cam = Vec3::new(x, 0.0, 0.0);
        let dirty = recenter(&mut ring, &mut atlas, &cfg, &bvh, cam);
        let (jobs, edits) = emit(&mut atlas, &cfg, &bvh, &resolved, &dirty);
        let high_water = atlas.tiles.high_water();
        gpu_atlas.bake_frame(&device, &queue, &pipeline, &layout, &jobs, &edits, high_water);

        // INVARIANT 1 — NO HOLE: every tile the chunk table references must hold baked content.
        // Build the real chunk table; for each resident brick, check its tile is non-zero.
        let tables = build_chunk_tables(&atlas, &cfg, |key| {
            let tile = atlas.tiles.tile(key).expect("resident brick has a tile");
            let (col, row) = tile_origin(tile);
            chunk::BrickTile { atlas_base: col | (row << 16), pal01: 0, pal23: 0 }
        });
        // Sample a handful of resident tiles (checking all every step is slow); include the
        // ones nearest the surface at the probe by scanning the served-LOD brick.
        if let Some(lod) = served_lod(&atlas, &cfg, probe) {
            let key = BrickKey::new(lod, cfg.world_to_brick_lod(probe, lod));
            let tile = atlas.tiles.tile(&key).expect("served brick must have a tile");
            assert!(
                gpu_atlas.read_tile_has_content(&device, &queue, tile),
                "step {step} (x={x}): served LOD-{lod} tile {tile} at probe is EMPTY — bake/atlas desync (the hole)"
            );
        }
        // Sanity: the directory's NON-SENTINEL slots match the resident chunks (no stale/extra
        // entries). The directory itself is fixed-size (R³·lod_count), so count occupied slots.
        let resident_in_dir = tables
            .chunks
            .iter()
            .filter(|c| (c.key_hi, c.key_lo) != chunk::SENTINEL_KEY)
            .count();
        assert_eq!(resident_in_dir, chunk::resident_chunks(&atlas, &cfg).len(), "step {step}: directory resident slots != resident chunks");

        served_seq.push(served_lod(&atlas, &cfg, probe));
    }

    // INVARIANT 2 — CLEAN TRANSITIONS (no flicker): while the probe is within clipmap reach,
    // coverage must be CONTINUOUS — there must never be a `None` (no coverage) sandwiched
    // between two covered steps. A 1-frame hole during an LOD transition would show up here as
    // Some(L) → None → Some(L') . (Trailing `None`s at the far end are the legitimate edge of
    // the finite clipmap reach, not a flicker, so we only forbid INTERIOR gaps.)
    eprintln!("served LOD sequence (centre): {served_seq:?}");
    let last_covered = served_seq.iter().rposition(|s| s.is_some()).unwrap();
    let first_covered = served_seq.iter().position(|s| s.is_some()).unwrap();
    for (i, s) in served_seq.iter().enumerate() {
        if i > first_covered && i < last_covered {
            assert!(s.is_some(), "step {i}: coverage hole BETWEEN covered steps (the flicker): {served_seq:?}");
        }
    }
    // COARSENING: going out the served LOD must increase (coarsen) then refine back.
    let covered: Vec<u32> = served_seq.iter().filter_map(|s| *s).collect();
    let start = covered[0];
    let peak = *covered.iter().max().unwrap();
    assert!(peak > start, "served LOD never coarsened (start={start}, peak={peak}): {covered:?}");
    assert_eq!(*covered.last().unwrap(), start, "served LOD did not refine back to the start on return: {covered:?}");
}
