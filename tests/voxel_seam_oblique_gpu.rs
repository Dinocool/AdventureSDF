//! **The OBLIQUE / GRAZING brick-seam correctness ORACLE.**
//!
//! DEFECT under test: rays slipping through the shared faces / edges / corners between adjacent bricks at
//! GRAZING / OBLIQUE angles — the thin BLACK lines along brick boundaries the user still sees on the Cornell
//! box when viewing the scene AT AN ANGLE (the straight-on `voxel_seam_gpu` rig does NOT reproduce it).
//!
//! Strategy: build a CONTINUOUS solid surface out of adjacent bricks (a flat floor / a thin one-voxel slab,
//! mirroring a Cornell wall whose visible surface lies ON a brick boundary), then fire a faithful CAMERA FAN
//! of primary rays at it from many OBLIQUE orientations — NOT pre-aimed at the surface, but fanned through a
//! pinhole view exactly like the real per-pixel rays. For each ray we compute the ANALYTIC ground truth
//! (does this ray geometrically land on the solid surface?) and require the GPU trace to AGREE: a ray the
//! geometry says hits, but the GPU reports as a MISS (or the wrong colour), is the seam bug.
//!
//! It runs the REAL `voxel_raytrace.wgsl` through the batched `trace_batch` entry on a real ray-query device
//! (one dispatch for the whole fan — `trace_one` round-trips the GPU per ray, far too slow to sweep the dense
//! grazing grid needed to catch a thin line). Skips cleanly without a ray-query adapter.

use std::iter;
use std::mem;

use bevy::math::{IVec3, Vec3};
use wgpu::util::DeviceExt;

use adventure::sdf_render::worldgen::biome::{
    BiomeDef, BiomeId, BiomeLibrary, StrataLayer, TerrainMatId, TerrainSurfaceMaterial,
};
use adventure::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, BRICK_WORLD_SIZE, Brick, BrickMap, VOXEL_SIZE, voxel_index};
use adventure::voxel::gpu::pack_brickmap;
use adventure::voxel::palette::{BlockId, BlockRegistry};

mod common;

// Mirror of the WGSL `Hit` struct (bindings 5/7).
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

// Mirror of the WGSL `RayUniform` (binding 4 / the batch input element). The batch input is a storage array
// of these; the struct layout (vec3 + f32 interleaved) matches the WGSL `RayUniform` std430 layout (each
// vec3<f32> is 16-byte aligned, the trailing f32 fills the pad slot), i.e. 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRay {
    origin: [f32; 3],
    t_min: f32,
    dir: [f32; 3],
    t_max: f32,
}

/// A minimal two-material library so the wall block (id 1) has a distinct, known colour.
fn wall_library() -> BiomeLibrary {
    let mat = |name: &str, c: [f32; 4]| TerrainSurfaceMaterial {
        name: name.into(),
        base_color: c,
        roughness: 0.9,
        blend: 0.0,
        texture: None,
        tiling: 4.0,
    };
    let materials = vec![mat("air_unused", [0.0, 0.0, 0.0, 1.0]), mat("wall", [0.8, 0.2, 0.05, 1.0])];
    let column = |_| BiomeDef {
        name: "b".into(),
        surface: TerrainMatId(0),
        surface_rules: vec![],
        strata: vec![StrataLayer { material: TerrainMatId(0), thickness: 1.0 }],
        bedrock: TerrainMatId(1),
    };
    let biomes = BiomeId::ALL.iter().map(column).collect();
    BiomeLibrary { materials, biomes }
}

/// Build a horizontal SLAB floor: solid in the world-voxel layers `y ∈ [floor_top - thick, floor_top)`, air
/// above and (for a thin slab) below, over a `[0, nx·BRICK_EDGE) × [0, nz·BRICK_EDGE)` footprint. The exposed
/// top surface is the world plane `y = floor_top · VOXEL_SIZE`. With `floor_top` a multiple of `BRICK_EDGE`
/// the top surface coincides with a Y brick boundary (the Cornell-wall case). `thick == 1` is a one-voxel
/// slab: a single-cell rounding error at a grazing brick seam then lands on AIR → a false MISS (the bug).
fn slab_map(nx: i32, nz: i32, floor_top: i32, thick: i32, wall: BlockId) -> BrickMap {
    let lo_y = floor_top - thick;
    let mut map = BrickMap::new();
    let by_lo = lo_y.div_euclid(BRICK_EDGE);
    let by_hi = (floor_top - 1).div_euclid(BRICK_EDGE);
    for bz in 0..nz {
        for by in by_lo..=by_hi {
            for bx in 0..nx {
                let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
                let mut any = false;
                for z in 0..BRICK_EDGE {
                    for y in 0..BRICK_EDGE {
                        for x in 0..BRICK_EDGE {
                            let wy = by * BRICK_EDGE + y;
                            if wy >= lo_y && wy < floor_top {
                                voxels[voxel_index(x, y, z)] = wall;
                                any = true;
                            }
                        }
                    }
                }
                if any {
                    map.insert(IVec3::new(bx, by, bz), Brick::from_voxels(voxels));
                }
            }
        }
    }
    map
}

/// A ready-to-trace GPU scene built from a brick patch + the batched `trace_batch` pipeline. `trace_all`
/// dispatches a whole slice of rays in ONE pass and returns the per-ray hits.
struct SeamScene {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    tlas: wgpu::Tlas,
    meta_buf: wgpu::Buffer,
    voxel_buf: wgpu::Buffer,
    palette_buf: wgpu::Buffer,
    // keep alive
    _aabb_buf: wgpu::Buffer,
    _blas: wgpu::Blas,
}

impl SeamScene {
    fn new(device: wgpu::Device, queue: wgpu::Queue, map: &BrickMap, reg: &BlockRegistry) -> Self {
        let patch = pack_brickmap(map, reg);
        assert!(!patch.is_empty(), "scene must contain bricks");
        let n = patch.brick_count() as u32;

        let aabb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("seam_aabbs"),
            contents: bytemuck::cast_slice(&patch.aabbs),
            usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE,
        });
        let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("seam_metas"),
            contents: bytemuck::cast_slice(&patch.metas),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let voxel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("seam_voxels"),
            contents: bytemuck::cast_slice(&patch.voxels),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let palette_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("seam_palette"),
            contents: bytemuck::cast_slice(&patch.palette),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let size_desc = wgpu::BlasAABBGeometrySizeDescriptor {
            primitive_count: n,
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("seam_blas"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::AABBs { descriptors: vec![size_desc.clone()] },
        );
        let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("seam_tlas"),
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

        let src = common::voxel_raytrace_shader_src();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel_raytrace"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("trace_batch"),
            layout: None,
            module: &shader,
            entry_point: Some("trace_batch"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bgl = pipeline.get_bind_group_layout(0);

        let mut build = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("seam_build") });
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
            bgl,
            tlas,
            meta_buf,
            voxel_buf,
            palette_buf,
            _aabb_buf: aabb_buf,
            _blas: blas,
        }
    }

    /// Trace a whole slice of rays in one dispatch and read back the hits.
    fn trace_all(&self, rays: &[GpuRay]) -> Vec<GpuHit> {
        let ray_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("seam_rays_in"),
            contents: bytemuck::cast_slice(rays),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_size = (rays.len() * mem::size_of::<GpuHit>()) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("seam_hits_out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("seam_read"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("seam_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::AccelerationStructure(&self.tlas) },
                wgpu::BindGroupEntry { binding: 1, resource: self.meta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.voxel_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.palette_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: ray_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: out_buf.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, Some(&bind_group), &[]);
            let groups = rays.len().div_ceil(64) as u32;
            cpass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, out_size);
        self.queue.submit(Some(encoder.finish()));
        let slice = read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
        let data = slice.get_mapped_range().unwrap();
        let out: Vec<GpuHit> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        read_buf.unmap();
        out
    }
}

/// A pinhole camera looking ALONG `view` (a unit direction with a downward Y) from `eye`. `fan_rays` builds
/// a dense grid of primary rays through a `±half_fov` pinhole, exactly like the real per-pixel ray fan.
struct Cam {
    view: Vec3,
    right: Vec3,
    up: Vec3,
    half_fov: f32,
}

impl Cam {
    fn looking_at(eye: Vec3, target: Vec3, half_fov_deg: f32) -> Self {
        let view = (target - eye).normalize();
        let right = view.cross(Vec3::Y).normalize();
        let up = right.cross(view).normalize();
        Self { view, right, up, half_fov: half_fov_deg.to_radians() }
    }
    fn ray(&self, sx: f32, sy: f32) -> Vec3 {
        let tf = self.half_fov.tan();
        (self.view + self.right * (sx * tf) + self.up * (sy * tf)).normalize()
    }
}

/// Analytic ground truth: does a ray `(ro, rd)` land on the solid TOP surface of a slab floor whose top plane
/// is `top_y` over the footprint `[0,span_x]×[0,span_z]`? True iff the ray descends and its crossing point is
/// strictly inside the footprint (a half-voxel margin off the open rim, where the answer is genuinely
/// ambiguous; the internal brick seams we care about are all interior).
fn analytic_surface_hit(ro: Vec3, rd: Vec3, top_y: f32, span_x: f32, span_z: f32) -> bool {
    if rd.y >= -1e-6 {
        return false;
    }
    let t = (top_y - ro.y) / rd.y;
    if t <= 0.0 {
        return false;
    }
    let p = ro + rd * t;
    let m = 0.5 * VOXEL_SIZE;
    p.x > m && p.x < span_x - m && p.z > m && p.z < span_z - m
}

/// THE ORACLE. A faithful camera-fan over many oblique orientations against a one-voxel slab whose top sits
/// on a brick boundary. Every ray the geometry says lands on the floor MUST come back a solid hit of the wall
/// colour; a miss / wrong colour is a brick seam. Runs the real shader; skips without a ray-query adapter.
#[test]
fn gpu_oblique_camera_fan_has_no_brick_seams() {
    let Some((device, queue)) = common::headless_ray_query_device() else {
        eprintln!("no ray-query device — skipping gpu_oblique_camera_fan_has_no_brick_seams");
        return;
    };

    let reg = BlockRegistry::from_biome_library(&wall_library());
    let wall = BlockId(1);
    let wall_color = reg.color(wall);

    // A one-voxel slab whose exposed top surface lies on the y = 2S brick boundary (Cornell-wall case), over
    // a 5×5-brick footprint so the surface crosses many internal x/z brick seams.
    let (nx, nz) = (5, 5);
    let floor_top = 2 * BRICK_EDGE;
    let map = slab_map(nx, nz, floor_top, 1, wall);
    let scene = SeamScene::new(device, queue, &map, &reg);

    let s = BRICK_WORLD_SIZE;
    let span_x = nx as f32 * s;
    let span_z = nz as f32 * s;
    let top_y = floor_top as f32 * VOXEL_SIZE;
    let centre = Vec3::new(span_x * 0.5, top_y, span_z * 0.5);
    let t_max = 1000.0f32;

    // Oblique camera orientations: (azimuth°, elevation° of the view below horizon). Small elevation =
    // grazing. The fan is dense so a thin missed line is sampled.
    let orientations: &[(f32, f32)] = &[
        (0.0, 10.0),
        (0.0, 18.0),
        (0.0, 30.0),
        (35.0, 12.0),
        (35.0, 25.0),
        (35.0, 45.0),
        (90.0, 12.0),
        (90.0, 22.0),
        (90.0, 38.0),
        (135.0, 15.0),
        (180.0, 12.0),
        (180.0, 28.0),
        (225.0, 18.0),
        (270.0, 14.0),
        (315.0, 22.0),
        (20.0, 8.0),
        (160.0, 8.0),
        (250.0, 9.0),
    ];
    let view_dist = 1.6 * span_x;
    let fan = 121usize; // dense fan → catch thin seam lines
    let half_fov_deg = 16.0f32;

    let mut rays: Vec<GpuRay> = Vec::new();
    let mut meta: Vec<(f32, f32, Vec3, Vec3)> = Vec::new(); // (az, el, hit_point, rd) for diagnostics
    for &(az_deg, el_deg) in orientations {
        let az = az_deg.to_radians();
        let el = el_deg.to_radians();
        let view = Vec3::new(az.cos() * el.cos(), -el.sin(), az.sin() * el.cos()).normalize();
        let eye = centre - view * view_dist;
        let cam = Cam::looking_at(eye, centre, half_fov_deg);
        for iy in 0..fan {
            for ix in 0..fan {
                let sx = (ix as f32 / (fan - 1) as f32) * 2.0 - 1.0;
                let sy = (iy as f32 / (fan - 1) as f32) * 2.0 - 1.0;
                let rd = cam.ray(sx, sy);
                if !analytic_surface_hit(eye, rd, top_y, span_x, span_z) {
                    continue;
                }
                let t = (top_y - eye.y) / rd.y;
                rays.push(GpuRay { origin: eye.into(), t_min: 0.0, dir: rd.into(), t_max });
                meta.push((az_deg, el_deg, eye + rd * t, rd));
            }
        }
    }

    eprintln!("oblique camera-fan: {} rays the geometry says hit the floor", rays.len());
    let hits = scene.trace_all(&rays);

    let mut misses = 0usize;
    let mut wrong_color = 0usize;
    let mut shown = 0usize;
    for (h, (az, el, p, rd)) in hits.iter().zip(meta.iter()) {
        if h.hit != 1 {
            misses += 1;
            if shown < 24 {
                shown += 1;
                eprintln!(
                    "MISS az={az} el={el} hit_point=({:.4},{:.4},{:.4}) dir=({:.4},{:.4},{:.4})",
                    p.x, p.y, p.z, rd.x, rd.y, rd.z
                );
            }
        } else if (h.color[0] - wall_color[0]).abs() > 1e-4
            || (h.color[1] - wall_color[1]).abs() > 1e-4
            || (h.color[2] - wall_color[2]).abs() > 1e-4
        {
            wrong_color += 1;
        }
    }
    eprintln!("oblique camera-fan seam: tested={} misses={misses} wrong_color={wrong_color}", rays.len());

    assert_eq!(misses, 0, "{misses}/{} camera-fan rays the geometry says hit the floor came back as MISSES — brick seam(s)!", rays.len());
    assert_eq!(wrong_color, 0, "{wrong_color} rays read the wrong colour at a brick seam");
}

/// A FINE sub-voxel sweep of grazing rays skimming the slab top right ACROSS the internal brick boundaries
/// (x = S and z = S), at many shallow elevations and headings. Each ray is aimed so its analytic landing point
/// sits on the solid surface; a thin band of misses at the seam is the bug. Batched into one dispatch.
#[test]
fn gpu_grazing_seam_fine_sweep() {
    let Some((device, queue)) = common::headless_ray_query_device() else {
        eprintln!("no ray-query device — skipping gpu_grazing_seam_fine_sweep");
        return;
    };

    let reg = BlockRegistry::from_biome_library(&wall_library());
    let wall = BlockId(1);
    let wall_color = reg.color(wall);

    let (nx, nz) = (4, 4);
    let floor_top = BRICK_EDGE; // surface on the y = S brick boundary
    let map = slab_map(nx, nz, floor_top, 1, wall);
    let scene = SeamScene::new(device, queue, &map, &reg);

    let s = BRICK_WORLD_SIZE;
    let _ = nz; // footprint is square; nz only sizes the scene
    let top_y = floor_top as f32 * VOXEL_SIZE;
    let t_max = 1000.0f32;

    // Sub-voxel lateral sweep ±1.5 voxels around the x=S / z=S seams, at many grazing elevations + headings.
    // Each ray's TARGET lies exactly ON the top surface (y = top_y) at the swept point, and the camera is
    // placed ABOVE it (descending dir), so the ray reaches y = top_y FOR THE FIRST TIME at the target — the
    // first solid contact is therefore unambiguously the top voxel's +Y face. Targets are kept well inside the
    // footprint (the swept seam at z/x = S and the mid-brick cross coord are both interior), so no ray enters
    // through the floor's open EDGE (which would legitimately have a sideways normal). Every hit must read +Y.
    let half = 1.5 * VOXEL_SIZE;
    let lat_steps = 121usize;
    let elevations = [3.0f32, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0, 50.0];
    let azimuths = [0.0f32, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 160.0, 180.0, 210.0, 270.0, 315.0];
    let radius = 30.0f32;

    let mut rays: Vec<GpuRay> = Vec::new();
    let mut meta: Vec<(char, f32, f32, f32, Vec3)> = Vec::new();
    for (axis, _seam) in [('z', s), ('x', s)] {
        for li in 0..lat_steps {
            let off = -half + (li as f32 / (lat_steps - 1) as f32) * (2.0 * half);
            // Hold the CROSS axis at a MID-brick value (1.5·S), never on a brick boundary — we sweep ONE seam
            // (the swept axis); a cross-coord exactly on the OTHER boundary plane is a measure-zero pathology
            // the continuous per-pixel ray fan never lands on (the camera-fan test covers genuine boundary
            // crossings from many angles). The swept value `S + off` is well inside the footprint.
            let cross = 1.5 * s;
            let tp = if axis == 'z' {
                Vec3::new(cross, top_y, s + off)
            } else {
                Vec3::new(s + off, top_y, cross)
            };
            for &el_deg in &elevations {
                for &az_deg in &azimuths {
                    let el = el_deg.to_radians();
                    let az = az_deg.to_radians();
                    let dir = Vec3::new(az.cos() * el.cos(), -el.sin(), az.sin() * el.cos()).normalize();
                    let eye = tp - dir * radius;
                    // Guard: the camera must be above the surface and the ray descends (so the target is the
                    // first y=top_y crossing). With dir.y<0 and the target on the plane this always holds.
                    rays.push(GpuRay { origin: eye.into(), t_min: 0.0, dir: dir.into(), t_max });
                    meta.push((axis, off, el_deg, az_deg, tp));
                }
            }
        }
    }

    eprintln!("grazing fine-sweep: {} rays", rays.len());
    let hits = scene.trace_all(&rays);

    let mut misses = 0usize;
    let mut wrong_color = 0usize;
    let mut wrong_normal = 0usize;
    let mut shown = 0usize;
    for (h, (axis, off, el, az, tp)) in hits.iter().zip(meta.iter()) {
        if h.hit != 1 {
            misses += 1;
            if shown < 24 {
                shown += 1;
                eprintln!(
                    "MISS {axis}-seam off={off:+.4} el={el} az={az} target=({:.4},{:.4},{:.4})",
                    tp.x, tp.y, tp.z
                );
            }
            continue;
        }
        if (h.color[0] - wall_color[0]).abs() > 1e-4
            || (h.color[1] - wall_color[1]).abs() > 1e-4
            || (h.color[2] - wall_color[2]).abs() > 1e-4
        {
            wrong_color += 1;
        }
        // The visible face is the slab TOP — its outward normal MUST be +Y. A normal that flips to a lateral
        // axis at a brick boundary is the thin DARK seam line (correct geometry, wrong shading). The user sees
        // it as black because a sideways normal on a floor goes dark under the overhead light.
        if (h.normal[0]).abs() > 1e-3 || (h.normal[1] - 1.0).abs() > 1e-3 || (h.normal[2]).abs() > 1e-3 {
            wrong_normal += 1;
            if shown < 24 {
                shown += 1;
                eprintln!(
                    "WRONG NORMAL {axis}-seam off={off:+.4} el={el} az={az} normal=({:.3},{:.3},{:.3}) target=({:.3},{:.3},{:.3})",
                    h.normal[0], h.normal[1], h.normal[2], tp.x, tp.y, tp.z
                );
            }
        }
    }
    eprintln!(
        "grazing fine-sweep seam: tested={} misses={misses} wrong_color={wrong_color} wrong_normal={wrong_normal}",
        rays.len()
    );

    assert_eq!(misses, 0, "{misses}/{} grazing rays missed the slab top at a brick seam — seam bug!", rays.len());
    assert_eq!(wrong_color, 0, "{wrong_color} grazing rays read the wrong colour at a brick seam");
    assert_eq!(wrong_normal, 0, "{wrong_normal} grazing rays recovered the WRONG face normal at a brick seam (the dark line)");
}
