//! CPU march simulation — measures raymarch step counts per ray on a baked scene, so we
//! can see WHERE the cost goes (which LOD, what `d`/`brick_exit`) without GPU round-trips.
//!
//! Ports the empty-space + sphere-trace logic of `assets/shaders/sdf_raymarch.wgsl` to the
//! CPU, reusing the real `SdfGridConfig` coord math + `chunk` tables so it matches the GPU
//! by construction. Measurement only (prints, loose asserts) — a tuning + regression tool.

use std::collections::HashMap;

use bevy::math::bounding::Aabb3d;
use bevy::math::{IVec3, Vec3};
use bevy::prelude::Transform;

use adventure::sdf_render::atlas::{BRICK_EDGE, PackedBrick, SdfAtlas};
use adventure::sdf_render::bvh::Bvh;
use adventure::sdf_render::chunk::{self, ChunkTables};
use adventure::sdf_render::edits::{edit_world_aabb, ResolvedEdit, SdfOp, SdfPrimitive};
use adventure::sdf_render::SdfGridConfig;

// --- Scene + tables -------------------------------------------------------------

fn single_sphere() -> (Vec<ResolvedEdit>, Bvh) {
    let edits = vec![ResolvedEdit {
        prim: SdfPrimitive::Sphere { radius: 1.0 },
        transform: Transform::IDENTITY,
        op: SdfOp::default(),
        material_id: 0,
    }];
    let aabbs: Vec<Aabb3d> = edits
        .iter()
        .map(|e| edit_world_aabb(&e.prim, &e.transform, e.op.smoothing))
        .collect();
    let bvh = Bvh::build(&aabbs);
    (edits, bvh)
}

/// Trilinear-sample one brick's `dist` field at world `p`, lod `lod`. Mirrors
/// `brick.wgsl::sample_brick_sdf`: voxel-space, brick-local corner + fractional, 8-corner mix.
fn sample_brick(brick: &PackedBrick, coord: IVec3, p: Vec3, vs: f32) -> f32 {
    let voxel_f = p / vs;
    let local = voxel_f - Vec3::new(coord.x as f32, coord.y as f32, coord.z as f32);
    let edge = BRICK_EDGE as i32;
    let i0 = IVec3::new(
        (local.x.floor() as i32).clamp(0, edge - 2),
        (local.y.floor() as i32).clamp(0, edge - 2),
        (local.z.floor() as i32).clamp(0, edge - 2),
    );
    let f = Vec3::new(
        local.x - i0.x as f32,
        local.y - i0.y as f32,
        local.z - i0.z as f32,
    );
    let at = |x: i32, y: i32, z: i32| -> f32 {
        let idx = (z as usize * BRICK_EDGE + y as usize) * BRICK_EDGE + x as usize;
        brick.dist[idx] as f32 / 32767.0
    };
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let c000 = at(i0.x, i0.y, i0.z);
    let c100 = at(i0.x + 1, i0.y, i0.z);
    let c010 = at(i0.x, i0.y + 1, i0.z);
    let c110 = at(i0.x + 1, i0.y + 1, i0.z);
    let c001 = at(i0.x, i0.y, i0.z + 1);
    let c101 = at(i0.x + 1, i0.y, i0.z + 1);
    let c011 = at(i0.x, i0.y + 1, i0.z + 1);
    let c111 = at(i0.x + 1, i0.y + 1, i0.z + 1);
    let x00 = lerp(c000, c100, f.x);
    let x10 = lerp(c010, c110, f.x);
    let x01 = lerp(c001, c101, f.x);
    let x11 = lerp(c011, c111, f.x);
    lerp(lerp(x00, x10, f.y), lerp(x01, x11, f.y), f.z)
}

struct Resolve {
    in_brick: bool,
    dist: f32,
    lod: u32,
    window_lod: u32,
}

/// CPU mirror of `resolve_march`: LOD 0→N, first resident chunk sets window_lod, first
/// occupied brick returns in_brick + sampled distance.
fn resolve_march(
    p: Vec3,
    cfg: &SdfGridConfig,
    tables: &ChunkTables,
    bricks: &HashMap<adventure::sdf_render::atlas::BrickKey, PackedBrick>,
) -> Resolve {
    use adventure::sdf_render::atlas::BrickKey;
    let levels = cfg.lod_count;
    let mut window_lod = levels - 1;
    let mut has_window = false;
    for lod in 0..levels {
        let coord = cfg.world_to_brick_lod(p, lod);
        let (ck, li) = chunk_of_local(BrickKey::new(lod, coord), cfg);
        let (kh, kl) = chunk::chunk_gpu_key(ck);
        if let Ok(idx) = tables.chunks.binary_search_by(|c| (c.key_hi, c.key_lo).cmp(&(kh, kl))) {
            if !has_window {
                window_lod = lod;
                has_window = true;
            }
            let chunk = tables.chunks[idx];
            let occ = (chunk.occ_lo as u64) | ((chunk.occ_hi as u64) << 32);
            if (occ >> li) & 1 == 1 {
                let key = BrickKey::new(lod, coord);
                if let Some(brick) = bricks.get(&key) {
                    let d = sample_brick(brick, coord, p, cfg.voxel_size_at(lod));
                    return Resolve { in_brick: true, dist: d, lod, window_lod };
                }
            }
        }
    }
    Resolve { in_brick: false, dist: 1e10, lod: 0, window_lod }
}

fn chunk_of_local(
    brick: adventure::sdf_render::atlas::BrickKey,
    cfg: &SdfGridConfig,
) -> (chunk::ChunkKey, u32) {
    chunk::chunk_of(brick, cfg)
}

/// CPU mirror of `dist_to_brick_exit_lod` / `dist_to_chunk_exit_lod` (slab test).
fn dist_to_box_exit(p: Vec3, dir: Vec3, box_world: f32) -> f32 {
    let box_min = (p / box_world).floor() * box_world;
    let box_max = box_min + Vec3::splat(box_world);
    let mut t = 1e10f32;
    for a in 0..3 {
        let d = dir[a];
        if d.abs() > 1e-6 {
            let bound = if d > 0.0 { box_max[a] } else { box_min[a] };
            let ta = (bound - p[a]) / d;
            if ta > 0.0 {
                t = t.min(ta);
            }
        }
    }
    t
}

/// CPU mirror of `in_ring_chunk`: is p's chunk at `lod` inside that LOD's resident ring?
fn in_ring_chunk(coord: IVec3, lod: u32, cfg: &SdfGridConfig, camera_pos: Vec3) -> bool {
    let s = cfg.cell_stride();
    let c = chunk::CHUNK_BRICKS;
    let r = (cfg.ring_bricks as i32) / c;
    let cam_brick = cfg.world_to_brick_lod(camera_pos, lod);
    let cam_c = IVec3::new(
        cam_brick.x.div_euclid(s).div_euclid(c),
        cam_brick.y.div_euclid(s).div_euclid(c),
        cam_brick.z.div_euclid(s).div_euclid(c),
    );
    let snap = cfg.recenter_snap_chunks.max(1);
    let sc = IVec3::new(
        cam_c.x.div_euclid(snap) * snap,
        cam_c.y.div_euclid(snap) * snap,
        cam_c.z.div_euclid(snap) * snap,
    );
    let half = r / 2;
    let origin = sc - IVec3::splat(half);
    let cc = IVec3::new(
        coord.x.div_euclid(s).div_euclid(c),
        coord.y.div_euclid(s).div_euclid(c),
        coord.z.div_euclid(s).div_euclid(c),
    );
    let rel = cc - origin;
    rel.x >= 0 && rel.y >= 0 && rel.z >= 0 && rel.x < r && rel.y < r && rel.z < r
}

fn chunk_resident(coord: IVec3, lod: u32, cfg: &SdfGridConfig, tables: &ChunkTables) -> bool {
    use adventure::sdf_render::atlas::BrickKey;
    let (ck, _) = chunk::chunk_of(BrickKey::new(lod, coord), cfg);
    let (kh, kl) = chunk::chunk_gpu_key(ck);
    tables
        .chunks
        .binary_search_by(|c| (c.key_hi, c.key_lo).cmp(&(kh, kl)))
        .is_ok()
}

struct MarchResult {
    steps: u32,
    hit: bool,
    /// Per-step diagnostics for the (optional) verbose dump: (lod, d, step, branch).
    trace: Vec<(u32, f32, f32, &'static str)>,
}

/// Port of the raymarch loop's empty-space + sphere-trace step logic (no cubic — we record
/// branch tags instead). `verbose` collects per-step diagnostics.
fn march(
    origin: Vec3,
    dir: Vec3,
    cfg: &SdfGridConfig,
    tables: &ChunkTables,
    bricks: &HashMap<adventure::sdf_render::atlas::BrickKey, PackedBrick>,
    camera_pos: Vec3,
    verbose: bool,
) -> MarchResult {
    const MAX_STEPS: u32 = 192;
    const MAX_DIST: f32 = 2000.0;
    const SDF_EPS: f32 = 0.001;
    const OMEGA: f32 = 1.0;
    let mut t = 0.0f32;
    let mut trace = Vec::new();
    for i in 0..MAX_STEPS {
        if t > MAX_DIST {
            return MarchResult { steps: i, hit: false, trace };
        }
        let p = origin + dir * t;
        let scene = resolve_march(p, cfg, tables, bricks);

        if !scene.in_brick {
            // Chunk-DDA: coarsest→fine, step the largest in-ring + absent chunk box.
            let wl = scene.window_lod;
            let vs_wl = cfg.voxel_size_at(wl);
            let mut adv = dist_to_box_exit(p, dir, cfg.brick_world_size(wl)) + vs_wl * 0.01;
            for l in (0..cfg.lod_count).rev() {
                let coord = cfg.world_to_brick_lod(p, l);
                if !chunk_resident(coord, l, cfg, tables) && in_ring_chunk(coord, l, cfg, camera_pos) {
                    let chunk_world = chunk::CHUNK_BRICKS as f32 * cfg.brick_world_size(l);
                    adv = adv.max(dist_to_box_exit(p, dir, chunk_world) + cfg.voxel_size_at(l) * 0.01);
                    break;
                }
            }
            if verbose {
                trace.push((wl, scene.dist, adv, "empty"));
            }
            t += adv;
            continue;
        }

        let lod = scene.lod;
        let voxel_size = cfg.voxel_size_at(lod);
        let d = scene.dist;
        let cone = 0.0; // cone_scale path omitted in the sim (measure raw sphere trace)

        if d < SDF_EPS.max(cone) {
            return MarchResult { steps: i + 1, hit: true, trace };
        }

        let brick_exit = dist_to_box_exit(p, dir, cfg.brick_world_size(lod));
        let step = (OMEGA * d).clamp(voxel_size * 0.01, brick_exit + voxel_size * 0.01);
        if verbose {
            trace.push((lod, d, step, "trace"));
        }
        t += step;
    }
    MarchResult { steps: MAX_STEPS, hit: false, trace }
}

// --- Over-relaxation (Keinert 2014) measurement ------------------------------------
//
// The cone prepass removed the empty-corridor cost; the only hot pixels left are GRAZING
// rays that skim a surface — `d` shrinks to a small min near closest approach, steps shrink
// to ~`d`, and the march crawls many tiny steps through the near-tangent band. Plain sphere
// tracing converges geometrically there, the inherent slow case.
//
// Over-relaxation steps `ω·d` (ω in [1,2)) instead of `d`, with the Keinert fallback: if the
// new unbounding sphere of radius `d` doesn't reach back over the prior relaxed step, the
// step overshot the surface — undo it and re-take safely. So hits never land at an overshoot
// (no swimming normals); the win is purely on the grazing crawl.
//
// MEASURED (this test): a fixed ω≈1.8 cuts grazing-miss steps ~40% with ZERO hit↔miss flips;
// gains flatten past 1.8 (near the ω<2 safety ceiling). Adaptive ω (Bálint & Valasek 2018,
// ω=1/along-ray-slope) and a "receding boost" were BOTH measured WORSE here — on a true miss
// the predicted surface never arrives, so the big jumps overshoot, fall back, and churn. So
// the shipped change is just the fixed-ω default (sdf_render::SdfRaymarchParams::over_relax).

/// March the real baked field with a fixed over-relaxation factor `omega`. Mirrors the
/// shader's Keinert undo fallback + brick-exit clamp. Returns (steps, hit).
fn march_relax(
    origin: Vec3,
    dir: Vec3,
    cfg: &SdfGridConfig,
    tables: &ChunkTables,
    bricks: &HashMap<adventure::sdf_render::atlas::BrickKey, PackedBrick>,
    camera_pos: Vec3,
    omega: f32,
) -> (u32, bool) {
    const MAX_STEPS: u32 = 192;
    const MAX_DIST: f32 = 2000.0;
    const SDF_EPS: f32 = 0.001;
    let mut t = 0.0f32;
    let mut prev_d = 0.0f32;
    let mut prev_step = 0.0f32;
    for i in 0..MAX_STEPS {
        if t > MAX_DIST {
            return (i, false);
        }
        let p = origin + dir * t;
        let scene = resolve_march(p, cfg, tables, bricks);

        if !scene.in_brick {
            let wl = scene.window_lod;
            let vs_wl = cfg.voxel_size_at(wl);
            let mut adv = dist_to_box_exit(p, dir, cfg.brick_world_size(wl)) + vs_wl * 0.01;
            for l in (0..cfg.lod_count).rev() {
                let coord = cfg.world_to_brick_lod(p, l);
                if !chunk_resident(coord, l, cfg, tables) && in_ring_chunk(coord, l, cfg, camera_pos) {
                    let cw = chunk::CHUNK_BRICKS as f32 * cfg.brick_world_size(l);
                    adv = adv.max(dist_to_box_exit(p, dir, cw) + cfg.voxel_size_at(l) * 0.01);
                    break;
                }
            }
            t += adv;
            prev_d = 0.0;
            prev_step = 0.0;
            continue;
        }

        let lod = scene.lod;
        let voxel_size = cfg.voxel_size_at(lod);
        let d = scene.dist;

        // Keinert overshoot undo: if the prior relaxed step jumped past the surface (the new
        // unbounding sphere doesn't reach back over it), step back and re-take safely.
        if prev_step > 0.0 && d + prev_d < prev_step {
            t += prev_d - prev_step; // negative
            prev_d = 0.0;
            prev_step = 0.0;
            continue;
        }

        if d < SDF_EPS {
            return (i + 1, true);
        }

        let brick_exit = dist_to_box_exit(p, dir, cfg.brick_world_size(lod));
        let step = (omega * d).clamp(voxel_size * 0.01, brick_exit + voxel_size * 0.01);
        t += step;
        prev_d = d;
        prev_step = step;
    }
    (MAX_STEPS, false)
}

#[test]
fn measure_grazing_overrelax() {
    let cfg = SdfGridConfig::default();
    let (edits, bvh) = single_sphere();
    let mut atlas = SdfAtlas::default();
    let camera_pos = Vec3::new(0.0, 2.0, 6.0);
    atlas.full_bake(&edits, &bvh, &cfg, camera_pos);
    let tables = chunk::build_chunk_tables(&atlas, &cfg, |_k| chunk::BrickTile::default());

    // Silhouette of the unit sphere from (0,2,6): centre dist √40≈6.32, half-angle
    // asin(1/6.32)≈9.1°. Sweep the grazing band 7–12° around the to-centre direction.
    let to_centre = (Vec3::ZERO - camera_pos).normalize();
    let right = to_centre.cross(Vec3::Y).normalize();

    println!("grazing fixed-omega sweep — steps + fate (silhouette ≈9.1°, shipped ω=1.8):");
    println!("  angle   w1.0     w1.4     w1.8     w1.9     w1.95");
    let mut flips = 0;
    let baseline_hit: Vec<bool> = (700..=1250)
        .step_by(25)
        .map(|q| {
            let ang = (q as f32 / 100.0).to_radians();
            let dir = (to_centre * ang.cos() + right * ang.sin()).normalize();
            march_relax(camera_pos, dir, &cfg, &tables, &atlas.bricks, camera_pos, 1.0).1
        })
        .collect();
    for (idx, q) in (700..=1250).step_by(25).enumerate() {
        let ang = (q as f32 / 100.0).to_radians();
        let dir = (to_centre * ang.cos() + right * ang.sin()).normalize();
        let run = |w: f32| march_relax(camera_pos, dir, &cfg, &tables, &atlas.bricks, camera_pos, w);
        let f = |(s, h): (u32, bool)| format!("{:3}{}", s, if h { "H" } else { "m" });
        let r18 = run(1.8);
        if r18.1 != baseline_hit[idx] {
            flips += 1;
        }
        println!(
            "  {:>5.2}°   {:>6}   {:>6}   {:>6}   {:>6}   {:>6}",
            ang.to_degrees(), f(run(1.0)), f(run(1.4)), f(r18), f(run(1.9)), f(run(1.95))
        );
    }
    // The shipped ω=1.8 must not change any ray's fate vs plain sphere tracing (ω=1.0).
    assert_eq!(flips, 0, "ω=1.8 flipped {flips} ray fate(s) vs ω=1.0 — quality regression");
}

// --- Per-LOD voxel-unit clamp simulation -------------------------------------------
//
// Model the proposed fix WITHOUT re-baking: the scene is one analytic sphere (r=1 at
// origin), so the true distance is known. We march the SAME loop but the "sampled" brick
// distance is the analytic distance clamped to ±k_voxels·voxel_size_at(lod) — what a per-LOD
// voxel-unit clamp would store — instead of the baked ±1.0-world plateau. Shows whether the
// step count collapses, and lets us pick K, before touching the bake.

fn sphere_sdf(p: Vec3) -> f32 {
    p.length() - 1.0
}

fn march_clamped(
    origin: Vec3,
    dir: Vec3,
    cfg: &SdfGridConfig,
    tables: &ChunkTables,
    bricks: &HashMap<adventure::sdf_render::atlas::BrickKey, PackedBrick>,
    camera_pos: Vec3,
    k_voxels: f32,
) -> u32 {
    const MAX_STEPS: u32 = 192;
    const MAX_DIST: f32 = 2000.0;
    const SDF_EPS: f32 = 0.001;
    let mut t = 0.0f32;
    for i in 0..MAX_STEPS {
        if t > MAX_DIST {
            return i;
        }
        let p = origin + dir * t;
        let scene = resolve_march(p, cfg, tables, bricks);
        if !scene.in_brick {
            let wl = scene.window_lod;
            let vs_wl = cfg.voxel_size_at(wl);
            let mut adv = dist_to_box_exit(p, dir, cfg.brick_world_size(wl)) + vs_wl * 0.01;
            for l in (0..cfg.lod_count).rev() {
                let coord = cfg.world_to_brick_lod(p, l);
                if !chunk_resident(coord, l, cfg, tables) && in_ring_chunk(coord, l, cfg, camera_pos) {
                    let cw = chunk::CHUNK_BRICKS as f32 * cfg.brick_world_size(l);
                    adv = adv.max(dist_to_box_exit(p, dir, cw) + cfg.voxel_size_at(l) * 0.01);
                    break;
                }
            }
            t += adv;
            continue;
        }
        let lod = scene.lod;
        let voxel_size = cfg.voxel_size_at(lod);
        let band = k_voxels * voxel_size; // the fix: per-LOD voxel-unit clamp band
        let d = sphere_sdf(p).clamp(-band, band);
        if d < SDF_EPS {
            return i + 1;
        }
        let brick_exit = dist_to_box_exit(p, dir, cfg.brick_world_size(lod));
        let step = d.clamp(voxel_size * 0.01, brick_exit + voxel_size * 0.01);
        t += step;
    }
    MAX_STEPS
}

#[test]
fn measure_per_lod_clamp_k_sweep() {
    let cfg = SdfGridConfig::default();
    let (edits, bvh) = single_sphere();
    let mut atlas = SdfAtlas::default();
    let camera_pos = Vec3::new(0.0, 2.0, 6.0);
    atlas.full_bake(&edits, &bvh, &cfg, camera_pos);
    let tables = chunk::build_chunk_tables(&atlas, &cfg, |_k| chunk::BrickTile::default());

    let dirs = [
        ("toward-z", Vec3::new(0.0, -0.3, -1.0).normalize()),
        ("sky+y", Vec3::new(0.0, 1.0, 0.0)),
        ("sky-x", Vec3::new(-1.0, 0.0, 0.0)),
        ("grazing", Vec3::new(0.4, -0.1, -1.0).normalize()),
        ("diag---", Vec3::new(-1.0, -1.0, -1.0).normalize()),
    ];

    println!("per-LOD voxel-unit clamp — steps by K (baked plateau gives ~95-173):");
    for k in [2.0f32, 4.0, 8.0, 16.0, 32.0] {
        print!(
            "  K={k:>4} (L0 {:.2}u, L5 {:.0}u): ",
            k * cfg.voxel_size_at(0),
            k * cfg.voxel_size_at(5)
        );
        for (name, dir) in dirs {
            let s = march_clamped(camera_pos, dir, &cfg, &tables, &atlas.bricks, camera_pos, k);
            print!("{name}={s} ");
        }
        println!();
    }
}

#[test]
fn measure_march_steps_single_sphere() {
    let cfg = SdfGridConfig::default();
    let (edits, bvh) = single_sphere();
    let mut atlas = SdfAtlas::default();
    // Camera looking at the sphere from a few units back (matches the orbit default-ish).
    let camera_pos = Vec3::new(0.0, 2.0, 6.0);
    atlas.full_bake(&edits, &bvh, &cfg, camera_pos);
    let tables = chunk::build_chunk_tables(&atlas, &cfg, |_key| chunk::BrickTile::default());

    println!(
        "baked {} bricks across {} chunks; ring_bricks={} lod_count={}",
        atlas.bricks.len(),
        tables.chunks.len(),
        cfg.ring_bricks,
        cfg.lod_count
    );

    // Ray fan: vary direction so we see the "depends on view direction" effect.
    let dirs = [
        ("toward sphere -z", Vec3::new(0.0, -0.3, -1.0).normalize()),
        ("sky +y (up)", Vec3::new(0.0, 1.0, 0.0)),
        ("sky +x", Vec3::new(1.0, 0.0, 0.0)),
        ("sky -x", Vec3::new(-1.0, 0.0, 0.0)),
        ("sky +z (behind)", Vec3::new(0.0, 0.2, 1.0).normalize()),
        ("grazing sphere", Vec3::new(0.4, -0.1, -1.0).normalize()),
        ("diagonal +++", Vec3::new(1.0, 1.0, 1.0).normalize()),
        ("diagonal ---", Vec3::new(-1.0, -1.0, -1.0).normalize()),
    ];

    let mut worst = (0u32, "");
    for (name, dir) in dirs {
        let r = march(camera_pos, dir, &cfg, &tables, &atlas.bricks, camera_pos, false);
        println!("  {name:18}  steps={:3}  hit={}", r.steps, r.hit);
        if r.steps > worst.0 {
            worst = (r.steps, name);
        }
    }
    println!("worst: {} steps ({})", worst.0, worst.1);

    // Verbose dump of the costliest ray's first 40 steps to see where the cost concentrates.
    if let Some((_, dir)) = dirs.iter().find(|(n, _)| *n == worst.1) {
        let r = march(camera_pos, *dir, &cfg, &tables, &atlas.bricks, camera_pos, true);
        println!("--- {} per-step (lod, d, step, branch) ---", worst.1);
        for (i, (lod, d, step, branch)) in r.trace.iter().take(40).enumerate() {
            println!("  {i:3}: lod={lod} d={d:+.3} step={step:.3} {branch}");
        }
    }
}

// --- Cone-marching prototype (Skipping Spheres / Claybook / Gunk) ------------------------
//
// PROTOTYPE ONLY (analytic sphere ground truth, no brick resolve) to measure whether a
// coarse low-res CONE pre-pass cuts the per-ray step count enough to justify a real GPU
// second pass. The expensive grazing rays (measured earlier) crawl ~30-54 small steps in a
// near-surface band; a cone trace amortized over an 8×8 tile should advance all 64 rays to
// the START of that band for ~free, so each full-res ray resumes there.
//
// Cone trace: advance by the SDF distance `d`, but the cone has radius `cone_k · t` (the
// world-width of the tile at distance t). Stop when `d <= cone_radius` — the surface is
// within the cone, so SOME ray in the tile may hit beyond here; we must hand off to per-ray.
// Conservative: the cone encloses all 64 rays, so no ray hit anything before this t.

fn analytic_sphere(p: Vec3) -> f32 {
    p.length() - 1.0 // unit sphere at origin
}

/// Coarse cone trace for a tile (represented by its centre ray). Returns the handoff `t`
/// (where the cone first touches the surface) and the coarse step count. `cone_k` = the
/// tile's angular half-width per unit distance = pixel_cone · tile_pixels.
fn cone_trace(origin: Vec3, dir: Vec3, cone_k: f32, max_dist: f32) -> (f32, u32) {
    const MAX_STEPS: u32 = 192;
    let mut t = 0.0f32;
    for i in 0..MAX_STEPS {
        if t > max_dist {
            return (t, i); // cone escaped — whole tile is sky, full-res rays can skip to here
        }
        let d = analytic_sphere(origin + dir * t);
        let cone_radius = cone_k * t;
        if d <= cone_radius {
            return (t, i + 1); // surface within the cone — hand off to per-ray here
        }
        t += d.max(1e-4);
    }
    (t, MAX_STEPS)
}

/// Per-ray sphere trace of the analytic sphere starting from `t0` (the cone handoff).
/// Returns (steps, hit). Mirrors the production hit-accept (`d < max(eps, pixel_cone·t)`).
fn ray_trace_from(origin: Vec3, dir: Vec3, t0: f32, pixel_cone: f32) -> (u32, bool) {
    const MAX_STEPS: u32 = 192;
    const MAX_DIST: f32 = 2000.0;
    const SDF_EPS: f32 = 0.001;
    let mut t = t0;
    for i in 0..MAX_STEPS {
        if t > MAX_DIST {
            return (i, false);
        }
        let d = analytic_sphere(origin + dir * t);
        if d < SDF_EPS.max(pixel_cone * t) {
            return (i + 1, true);
        }
        t += d.max(1e-4);
    }
    (MAX_STEPS, false)
}

#[test]
fn measure_cone_marching() {
    let camera_pos = Vec3::new(0.0, 2.0, 6.0);
    let to_centre = (Vec3::ZERO - camera_pos).normalize();
    let right = to_centre.cross(Vec3::Y).normalize();

    let fov_y = 60.0f32.to_radians();
    let pixel_cone = (fov_y * 0.5).tan() / 1080.0; // per-pixel half-width per unit t
    const TILE: f32 = 8.0; // 8×8 pixel tile → 64 rays share one cone
    let cone_k = pixel_cone * TILE;

    println!("cone marching: tile={TILE}px  baseline=full ray from t=0 vs cone-seeded");
    println!("angle  baseline_steps  cone_steps  perray_after  amortized  fate");

    // Sweep the silhouette band where the expensive grazing rays live (9-12°).
    for q in (700..1250).step_by(50) {
        let ang = (q as f32 / 100.0).to_radians();
        let dir = (to_centre * ang.cos() + right * ang.sin()).normalize();

        // Baseline: full-res ray from t=0.
        let (base_steps, base_hit) = ray_trace_from(camera_pos, dir, 0.0, pixel_cone);

        // Cone-marched: coarse cone to handoff t0, then per-ray from t0.
        let (t0, cone_steps) = cone_trace(camera_pos, dir, cone_k, 2000.0);
        let (after_steps, hit) = ray_trace_from(camera_pos, dir, t0, pixel_cone);
        // Amortized per-pixel cost: the cone is shared by 64 rays, so its cost / 64 + the
        // per-ray steps after handoff.
        let amortized = cone_steps as f32 / (TILE * TILE) + after_steps as f32;

        let fate = if hit { "Hit" } else { "miss" };
        println!(
            "  {:>5.2}°  base={:3}({})  cone={:3}  after={:3}  amort={:.1}  {fate}",
            ang.to_degrees(), base_steps, if base_hit { "H" } else { "m" },
            cone_steps, after_steps, amortized
        );
    }
}
