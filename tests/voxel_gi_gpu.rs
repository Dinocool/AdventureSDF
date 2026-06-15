//! **Real-GPU correctness oracle for HW-RT voxel GLOBAL ILLUMINATION** (`voxel_raytrace.wgsl`,
//! `trace_one`'s GI terms).
//!
//! Stage 4b adds single-bounce diffuse GI + emissive-voxel lights on top of the proven `ray_query` DDA +
//! direct-lighting core. This rig proves the three load-bearing GI behaviours in ISOLATION, WITHOUT a GUI,
//! by reading back the SEPARATED `direct` / `indirect` / `emissive_out` terms the shader writes per hit:
//!
//!   1. **Indirect fills shadow** — a floor point in HARD sun shadow under a floating overhang receives
//!      SOME bounced light: with GI on its `indirect` term is meaningfully > 0 (vs exactly 0 with GI off).
//!   2. **Colour bleed** — a neutral grey floor next to a saturated RED wall picks up a red tint in its
//!      `indirect` term (R rises relative to G/B) near the wall, vs a far-from-the-wall patch.
//!   3. **Emissive illuminates** — an EMISSIVE block makes a neutral floor brighter (larger `indirect`)
//!      near the emitter than far from it, AND the emitter's own `emissive_out` glow is non-zero.
//!
//! `trace_one` runs the SAME `gather_gi` / `direct_lighting` the render path's `shade` uses (a fixed seed
//! keeps the estimate reproducible), so these assertions exercise the real GI math — no oracle drift.
//!
//! Skips cleanly (no failure) on a box without an `EXPERIMENTAL_RAY_QUERY` Vulkan adapter.

use std::iter;
use std::mem;

use bevy::math::{IVec3, Vec3};
use wgpu::util::DeviceExt;

use adventure::sdf_render::worldgen::biome::{
    BiomeDef, BiomeId, BiomeLibrary, StrataLayer, TerrainMatId, TerrainSurfaceMaterial,
};
use adventure::voxel::brickmap::{BRICK_WORLD_SIZE, Brick};
use adventure::voxel::gpu::{ResidentBrick, pack_resident_set};
use adventure::voxel::palette::{BlockId, BlockRegistry};
use adventure::voxel::raytrace::{LightingUniformData, SkyUniformData};

mod common;

// Mirror of the WGSL `Hit` struct (binding 5) — geometry fields + the separated GI oracle terms.
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

// Mirror of the WGSL `RayUniform` (binding 4).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RayUniform {
    origin: [f32; 3],
    t_min: f32,
    dir: [f32; 3],
    t_max: f32,
}

/// Block ids in the test library: 1 = neutral grey floor, 2 = saturated RED, 3 = an (initially neutral)
/// block we mark EMISSIVE via the registry.
const FLOOR: BlockId = BlockId(1);
const RED: BlockId = BlockId(2);
const EMITTER: BlockId = BlockId(3);

/// A library whose three materials map to the floor/red/emitter blocks. Colours are LINEAR.
fn test_library() -> BiomeLibrary {
    let mat = |name: &str, c: [f32; 4]| TerrainSurfaceMaterial {
        name: name.into(),
        base_color: c,
        roughness: 0.9,
        blend: 0.0,
        texture: None,
        tiling: 4.0,
        ..Default::default()
    };
    let materials = vec![
        mat("floor", [0.5, 0.5, 0.5, 1.0]), // neutral grey
        mat("red", [0.9, 0.02, 0.02, 1.0]), // saturated red (the colour-bleed source)
        mat("emit", [0.04, 0.04, 0.04, 1.0]), // dark base; emissive added via the registry
    ];
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

/// A fully-solid brick of `id`.
fn solid(id: BlockId) -> Brick {
    Brick::uniform(id)
}

/// The GPU scene + reusable single-ray runner, parameterised by the packed patch and a lighting uniform.
/// Returns a closure that fires one ray and reads back the `GpuHit` (so a test can sweep lighting/rays).
struct GiRig {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GiRig {
    /// Build the GPU scene from a packed patch and return a runner closure `(light, sky, ro, rd) -> GpuHit`.
    /// The acceleration structure is built once; each call only rewrites the ray + lighting + SKY uniforms.
    /// Sky is an EXPLICIT per-run input (never a hidden default): a transport test passes a dark sky to
    /// isolate voxel-to-voxel bounce, while a sky-fill test passes the real procedural sky.
    fn run_all(
        &self,
        patch: &adventure::voxel::gpu::GpuBrickPatch,
    ) -> impl Fn(&LightingUniformData, &SkyUniformData, Vec3, Vec3) -> GpuHit + '_ {
        let device = &self.device;
        let queue = &self.queue;
        let n = patch.brick_count() as u32;

        let aabb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gi_aabbs"),
            contents: bytemuck::cast_slice(&patch.aabbs),
            usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE,
        });
        let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gi_metas"),
            contents: bytemuck::cast_slice(&patch.metas),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let voxel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gi_voxels"),
            contents: bytemuck::cast_slice(&patch.voxels),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let palette_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gi_palette"),
            contents: bytemuck::cast_slice(&patch.palette),
            usage: wgpu::BufferUsages::STORAGE,
        });
        // Storage plan R2b — the per-brick palettes the bit-packed index stream indirects through.
        let brick_palettes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gi_palette_brick_palettes"),
            contents: bytemuck::cast_slice(&patch.brick_palettes),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let size_desc = wgpu::BlasAABBGeometrySizeDescriptor {
            primitive_count: n,
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("gi_blas"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::AABBs { descriptors: vec![size_desc.clone()] },
        );
        let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("gi_tlas"),
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
            label: Some("gi_ray"),
            size: mem::size_of::<RayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let light_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi_light"),
            size: mem::size_of::<LightingUniformData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi_hit"),
            size: mem::size_of::<GpuHit>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi_read"),
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
                wgpu::BindGroupEntry { binding: 12, resource: brick_palettes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: ray_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
            ],
        });
        let sky_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi_sky"),
            size: mem::size_of::<SkyUniformData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let light_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi_light_bg"),
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[
                wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: sky_buf.as_entire_binding() },
            ],
        });

        let mut build = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gi_build") });
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

        // Move the GPU objects into the closure so they outlive every ray.
        move |light: &LightingUniformData, sky: &SkyUniformData, ro: Vec3, rd: Vec3| -> GpuHit {
            queue.write_buffer(&light_buf, 0, bytemuck::bytes_of(light));
            queue.write_buffer(&sky_buf, 0, bytemuck::bytes_of(sky));
            let ray = RayUniform { origin: ro.into(), t_min: 0.0, dir: rd.normalize().into(), t_max: 1000.0 };
            queue.write_buffer(&ray_buf, 0, bytemuck::bytes_of(&ray));
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&pipeline);
                cpass.set_bind_group(0, Some(&bind_group), &[]);
                cpass.set_bind_group(1, Some(&light_bg), &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, mem::size_of::<GpuHit>() as u64);
            queue.submit(Some(encoder.finish()));
            let slice = read_buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
            device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
            let data = slice.get_mapped_range().unwrap();
            let gpu: GpuHit = *bytemuck::from_bytes(&data);
            drop(data);
            read_buf.unmap();
            // Keep the scene alive across calls.
            let _ = (&aabb_buf, &meta_buf, &voxel_buf, &palette_buf, &blas, &tlas, &sky_buf);
            gpu
        }
    }
}

/// Luma of a linear RGB triple (Rec.709-ish) — the "is there light here" scalar.
fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// A lighting uniform with the sun straight DOWN and GI configured. `gi_rays == 0` disables GI (the
/// GI-off baseline); a high ray count drives down the Monte-Carlo noise for stable assertions.
fn light_with_gi(gi_rays: u32) -> LightingUniformData {
    LightingUniformData {
        sun_direction: [0.0, -1.0, 0.0], // straight down
        gi_rays,
        gi_intensity: 1.0,
        gi_bounce_dist: 20.0,
        ..Default::default()
    }
}

/// A BLACK sky (all-zero radiance) for the voxel-transport scenarios. Phase 1A made a bounce that escapes to
/// open sky return `sky_radiance × gi_sky_intensity`; these open scenes have many such misses, so a non-zero
/// sky would add a uniform fill that dilutes the voxel-to-voxel transport the test isolates. Zeroing the sky
/// keeps the colour-bleed / shadow-fill / emissive assertions about voxel transport ALONE. The
/// `open_scene_bounce_miss_returns_sky` test below covers the new sky-fill path explicitly.
fn dark_sky() -> SkyUniformData {
    SkyUniformData {
        horizon_color: [0.0, 0.0, 0.0],
        zenith_color: [0.0, 0.0, 0.0],
        ground_color: [0.0, 0.0, 0.0],
        sun_size: 0.0, // no sun disk either
        intensity: 0.0,
        gi_sky_intensity: 0.0,
        sun_tint: [0.0, 0.0, 0.0],
        _pad: 0.0,
    }
}

/// **Scenario 1 — indirect light fills a hard sun shadow.** A floor under a floating roof is in hard
/// shadow (its `direct` sun term is killed); with GI on, bounced light from the surrounding lit floor +
/// roof underside gives it a non-zero `indirect`. With GI off, `indirect` is exactly 0.
#[test]
fn gi_indirect_fills_shadow() {
    let Some((device, queue)) = common::headless_ray_query_device() else {
        eprintln!("no ray-query device — skipping gi_indirect_fills_shadow");
        return;
    };
    let reg = BlockRegistry::from_biome_library(&test_library());
    let s = BRICK_WORLD_SIZE;

    // A 5x5 floor of grey bricks at by=0, plus a floating roof brick a few bricks up over the centre column
    // so the centre floor point sits in hard shadow but is surrounded by lit floor to bounce off.
    let floor = solid(FLOOR);
    let roof = solid(FLOOR);
    let mut entries: Vec<ResidentBrick> = Vec::new();
    for bx in -2..=2i32 {
        for bz in -2..=2i32 {
            entries.push(ResidentBrick { coord: IVec3::new(bx, 0, bz), brick: &floor, lod: 0 });
        }
    }
    entries.push(ResidentBrick { coord: IVec3::new(0, 3, 0), brick: &roof, lod: 0 });
    let patch = pack_resident_set(&entries, &reg);

    let rig = GiRig { device, queue };
    let run = rig.run_all(&patch);

    let floor_top = s; // floor brick spans Y in [0, s]
    let centre = Vec3::new(s * 0.5, floor_top + 1.0, s * 0.5); // under the roof
    let down = Vec3::new(0.0, -1.0, 0.0);

    let off = run(&light_with_gi(0), &dark_sky(), centre, down);
    let on = run(&light_with_gi(64), &dark_sky(), centre, down);

    eprintln!(
        "[shadow-fill] hit={} shadowed={} direct={:?} indirect(off)={:?} indirect(on)={:?}",
        on.hit, on.shadowed, on.direct, off.indirect, on.indirect
    );
    assert_eq!(on.hit, 1, "centre ray must hit the floor");
    assert_eq!(on.shadowed, 1, "centre floor must be in hard sun shadow (roof above)");
    // GI off: no indirect at all.
    assert!(luma(off.indirect) < 1e-5, "GI-off indirect must be ~0, got {:?}", off.indirect);
    // GI on: bounced light fills the shadow — meaningfully above zero.
    assert!(
        luma(on.indirect) > 1e-3,
        "GI-on indirect must fill the shadow (luma>0), got {:?} (luma {})",
        on.indirect,
        luma(on.indirect)
    );
}

/// **Scenario 2 — colour bleed.** A neutral grey floor beside a saturated RED wall picks up a red tint in
/// its indirect term NEAR the wall (R clearly exceeds G & B), while a patch FAR from the wall stays nearly
/// neutral. Proves single-bounce colour transport (the bounce carries the red surface's lit colour).
#[test]
fn gi_colour_bleed() {
    let Some((device, queue)) = common::headless_ray_query_device() else {
        eprintln!("no ray-query device — skipping gi_colour_bleed");
        return;
    };
    let reg = BlockRegistry::from_biome_library(&test_library());
    let s = BRICK_WORLD_SIZE;

    // A long grey floor strip along +X (bx = 0..=8) at by=0, and a RED wall standing on the floor at bx=0,
    // by = 1..=3 (so it faces +X and is lit by the down-sun on its top — but its +X face is what bounces
    // onto the floor). Use a tall red wall to give a strong bleed source close to the near floor patch.
    let floor = solid(FLOOR);
    let red = solid(RED);
    let mut entries: Vec<ResidentBrick> = Vec::new();
    for bx in 0..=8i32 {
        entries.push(ResidentBrick { coord: IVec3::new(bx, 0, 0), brick: &floor, lod: 0 });
        // Side floor rows so the bleed source/receiver have neighbours to bounce off too.
        entries.push(ResidentBrick { coord: IVec3::new(bx, 0, 1), brick: &floor, lod: 0 });
        entries.push(ResidentBrick { coord: IVec3::new(bx, 0, -1), brick: &floor, lod: 0 });
    }
    // Red wall at bx=0 rising 3 bricks (by=1..=3) across the z strip.
    for by in 1..=3i32 {
        for bz in -1..=1i32 {
            entries.push(ResidentBrick { coord: IVec3::new(0, by, bz), brick: &red, lod: 0 });
        }
    }
    let patch = pack_resident_set(&entries, &reg);

    let rig = GiRig { device, queue };
    let run = rig.run_all(&patch);

    let floor_top = s;
    let down = Vec3::new(0.0, -1.0, 0.0);
    // NEAR the red wall: floor at bx=1 (world X in [s, 2s]) — just past the wall at bx=0.
    let near = Vec3::new(s * 1.3, floor_top + 1.0, s * 0.5);
    // FAR from the wall: floor at bx=7.
    let far = Vec3::new(s * 7.5, floor_top + 1.0, s * 0.5);

    // Tilt the sun so it travels toward -X/-Y: toward-sun then has a +X component, lighting the red wall's
    // +X face (the face that bounces onto the near floor). A straight-down sun leaves that vertical face
    // unlit, so it would bounce nothing — the bleed needs a LIT red surface.
    let mut l = light_with_gi(64);
    l.sun_direction = Vec3::new(-0.7, -0.7, 0.0).normalize().into();
    let near_hit = run(&l, &dark_sky(), near, down);
    let far_hit = run(&l, &dark_sky(), far, down);

    eprintln!(
        "[colour-bleed] near indirect={:?} far indirect={:?}",
        near_hit.indirect, far_hit.indirect
    );
    assert_eq!(near_hit.hit, 1, "near ray must hit the floor");
    assert_eq!(far_hit.hit, 1, "far ray must hit the floor");
    let nr = near_hit.indirect;
    // Near the red wall the indirect R channel clearly dominates G and B (red bleed).
    assert!(
        nr[0] > nr[1] * 1.5 && nr[0] > nr[2] * 1.5,
        "near-wall indirect must be red-tinted (R≫G,B), got {nr:?}"
    );
    // And the near patch is redder than the far patch (R rises relative to far): the bleed is local.
    assert!(
        nr[0] > far_hit.indirect[0],
        "near R ({}) must exceed far R ({}) — bleed is strongest near the red wall",
        nr[0],
        far_hit.indirect[0]
    );
    // Sanity: the far patch is much less red-biased than the near one.
    let near_red_bias = nr[0] - 0.5 * (nr[1] + nr[2]);
    let fr = far_hit.indirect;
    let far_red_bias = fr[0] - 0.5 * (fr[1] + fr[2]);
    assert!(
        near_red_bias > far_red_bias + 1e-4,
        "near red-bias ({near_red_bias}) must exceed far red-bias ({far_red_bias})"
    );
}

/// **Scenario 3 — emissive voxels illuminate neighbours.** An emissive block beside a neutral floor makes
/// the floor's indirect brighter NEAR the emitter than FAR from it, and the emitter's own `emissive_out`
/// glow is non-zero (it visibly lights up itself too).
#[test]
fn gi_emissive_illuminates() {
    let Some((device, queue)) = common::headless_ray_query_device() else {
        eprintln!("no ray-query device — skipping gi_emissive_illuminates");
        return;
    };
    let mut reg = BlockRegistry::from_biome_library(&test_library());
    // Make the EMITTER block glow bright white (linear radiance). This is the per-block palette emissive
    // the GI bounce returns as a light source.
    reg.set_emissive(EMITTER, [3.0, 3.0, 3.0]);
    let s = BRICK_WORLD_SIZE;

    // Grey floor strip along +X (bx = 0..=8), with an EMISSIVE pillar standing on the floor at bx=0
    // (by = 1..=2). To isolate emissive transport from sun bleed, put the sun BELOW the horizon (no direct
    // sun on the floor), so the only indirect energy is the emitter's glow.
    let floor = solid(FLOOR);
    let emit = solid(EMITTER);
    let mut entries: Vec<ResidentBrick> = Vec::new();
    for bx in 0..=8i32 {
        entries.push(ResidentBrick { coord: IVec3::new(bx, 0, 0), brick: &floor, lod: 0 });
        entries.push(ResidentBrick { coord: IVec3::new(bx, 0, 1), brick: &floor, lod: 0 });
        entries.push(ResidentBrick { coord: IVec3::new(bx, 0, -1), brick: &floor, lod: 0 });
    }
    for by in 1..=2i32 {
        for bz in -1..=1i32 {
            entries.push(ResidentBrick { coord: IVec3::new(0, by, bz), brick: &emit, lod: 0 });
        }
    }
    let patch = pack_resident_set(&entries, &reg);

    let rig = GiRig { device, queue };
    let run = rig.run_all(&patch);

    let floor_top = s;
    let down = Vec3::new(0.0, -1.0, 0.0);

    // Sun BELOW the horizon (pointing UP) and zero ambient, so direct sun + sky contribute nothing to the
    // floor — the entire indirect signal is the emitter's glow.
    let mut l = light_with_gi(64);
    l.sun_direction = [0.0, 1.0, 0.0]; // travels upward ⇒ toward-sun points down ⇒ no light on +Y floor
    l.ambient_color = [0.0, 0.0, 0.0];
    l.emissive_strength = 4.0;

    let near = Vec3::new(s * 1.3, floor_top + 1.0, s * 0.5); // just past the emitter pillar at bx=0
    let far = Vec3::new(s * 7.5, floor_top + 1.0, s * 0.5);
    // Dark sky too: with the sun below the +Y floor and ambient zeroed, the ONLY indirect energy must be the
    // emitter's glow — a non-zero sky would add a uniform fill that breaks that isolation.
    let near_hit = run(&l, &dark_sky(), near, down);
    let far_hit = run(&l, &dark_sky(), far, down);

    // Fire a ray directly at the emitter block to read back its own glow term.
    let emitter_probe = Vec3::new(s * 0.5, s * 4.0, s * 0.5); // above the pillar, looking down at its top
    let emit_hit = run(&l, &dark_sky(), emitter_probe, down);

    eprintln!(
        "[emissive] near indirect={:?} far indirect={:?} emitter glow={:?} (block {})",
        near_hit.indirect, far_hit.indirect, emit_hit.emissive_out, emit_hit.block_id
    );
    assert_eq!(near_hit.hit, 1, "near ray must hit the floor");
    assert_eq!(far_hit.hit, 1, "far ray must hit the floor");
    // The emitter's own glow is non-zero (palette emissive × strength = 3 × 4 = 12 per channel).
    assert_eq!(emit_hit.block_id, EMITTER.0 as u32, "probe must hit the emitter block");
    assert!(
        luma(emit_hit.emissive_out) > 1.0,
        "emitter's own glow must be bright, got {:?}",
        emit_hit.emissive_out
    );
    // The floor near the emitter is brighter (more bounced emissive) than far from it.
    assert!(
        luma(near_hit.indirect) > 1e-3,
        "floor near the emitter must receive emissive light, got {:?}",
        near_hit.indirect
    );
    assert!(
        luma(near_hit.indirect) > luma(far_hit.indirect),
        "floor near the emitter ({}) must be brighter than far ({})",
        luma(near_hit.indirect),
        luma(far_hit.indirect)
    );
}

/// **Scenario 4 — OPEN-SCENE sky fill (Phase 1A).** On a flat open floor with no other geometry, the upward
/// diffuse bounces escape to open sky and must return the SKY radiance — not the old flat `ambient_color`.
/// With a bright RED sky (and a neutral-grey floor + no red voxels, sun below the floor), the floor's indirect
/// term must be RED-tinted; with the sky BLACK it must be ~0. Proves the bounce-miss → `sky_radiance` path.
#[test]
fn open_scene_bounce_miss_returns_sky() {
    let Some((device, queue)) = common::headless_ray_query_device() else {
        eprintln!("no ray-query device — skipping open_scene_bounce_miss_returns_sky");
        return;
    };
    let reg = BlockRegistry::from_biome_library(&test_library());
    let s = BRICK_WORLD_SIZE;

    // A flat 5×5 grey floor at by=0 — nothing above it, so a bounce up escapes to sky.
    let floor = solid(FLOOR);
    let mut entries: Vec<ResidentBrick> = Vec::new();
    for bx in -2..=2i32 {
        for bz in -2..=2i32 {
            entries.push(ResidentBrick { coord: IVec3::new(bx, 0, bz), brick: &floor, lod: 0 });
        }
    }
    let patch = pack_resident_set(&entries, &reg);
    let rig = GiRig { device, queue };
    let run = rig.run_all(&patch);

    let floor_top = s;
    let down = Vec3::new(0.0, -1.0, 0.0);
    let probe = Vec3::new(0.0, floor_top + 1.0, 0.0);

    // Sun BELOW the floor (travels upward) + ambient zeroed, so the only indirect energy is the sky a bounce
    // escapes to (no direct, no voxel-bounce colour — the floor is neutral grey).
    let mut l = light_with_gi(64);
    l.sun_direction = [0.0, 1.0, 0.0];
    l.ambient_color = [0.0, 0.0, 0.0];

    // A bright uniform RED sky (flat gradient: horizon == zenith == ground), no sun disk.
    let red_sky = SkyUniformData {
        horizon_color: [1.0, 0.0, 0.0],
        zenith_color: [1.0, 0.0, 0.0],
        ground_color: [1.0, 0.0, 0.0],
        sun_size: 0.0,
        intensity: 1.0,
        gi_sky_intensity: 1.0,
        sun_tint: [0.0, 0.0, 0.0],
        _pad: 0.0,
    };

    let sky_hit = run(&l, &red_sky, probe, down);
    let dark_hit = run(&l, &dark_sky(), probe, down);
    eprintln!(
        "[sky-fill] red-sky indirect={:?} dark-sky indirect={:?}",
        sky_hit.indirect, dark_hit.indirect
    );
    assert_eq!(sky_hit.hit, 1, "probe must hit the floor");
    let si = sky_hit.indirect;
    // The floor's indirect is RED-tinted (sky transport, × the grey floor albedo).
    assert!(
        si[0] > 1e-3 && si[0] > si[1] * 4.0 && si[0] > si[2] * 4.0,
        "open-floor indirect must be red-tinted from the red SKY (R≫G,B), got {si:?}"
    );
    // And a BLACK sky gives ~0 indirect (the old flat-ambient fill is gone — sky is the only fill now).
    assert!(
        luma(dark_hit.indirect) < 1e-3,
        "with a black sky the open floor's indirect must be ~0, got {:?}",
        dark_hit.indirect
    );
}
