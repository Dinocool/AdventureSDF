//! Reproduces (and guards against) the "faces showing through geometry" bug: certain camera rays through
//! the Cornell box commit a voxel hit NEARER than the true first-solid voxel — a phantom near-hit that
//! paints a front face over the geometry that should occlude it (confirmed visually: the through-faces are
//! GREEN front faces at a BRIGHTER/nearer depth).
//!
//! This is CPU-only + deterministic. It ports the REAL WGSL `dda_brick` + `trace` FAITHFULLY (grown-AABB
//! slab, one-cell halo grid marched from `world_min − csize`, entry-cell clamp, commit only CORE cells) and
//! compares its committed world-`t` against the simple correct world-grid DDA ground truth
//! (`cpu_first_solid`). Any ray where the faithful port commits a hit measurably NEARER than the ground
//! truth is the show-through bug. The fix must make this test pass.

use adventure::voxel::brickmap::{
    BRICK_EDGE, BrickMap, VOXEL_SIZE, brick_span, lod_edge, lod_voxel_size,
};
use adventure::voxel::cornell::{INTERIOR, build_cornell};
use adventure::voxel::gpu::{GpuBrickPatch, brick_aabb_epsilon, pack_brickmap};
use adventure::voxel::palette::{BlockId, BlockRegistry};
use bevy::math::{IVec3, Vec3};

/// Correct world-grid 3D-DDA over the brickmap → `(block, t)` of the first solid voxel. The ground truth.
fn cpu_first_solid(map: &BrickMap, ro: Vec3, rd: Vec3, t_max: f32) -> Option<(BlockId, f32)> {
    let rd = rd.normalize();
    let step = IVec3::new(rd.x.signum() as i32, rd.y.signum() as i32, rd.z.signum() as i32);
    let inv = Vec3::new(1.0 / rd.x, 1.0 / rd.y, 1.0 / rd.z);
    let mut vox = IVec3::new(
        (ro.x / VOXEL_SIZE).floor() as i32,
        (ro.y / VOXEL_SIZE).floor() as i32,
        (ro.z / VOXEL_SIZE).floor() as i32,
    );
    let nb = Vec3::new(
        (vox.x + step.x.max(0)) as f32 * VOXEL_SIZE,
        (vox.y + step.y.max(0)) as f32 * VOXEL_SIZE,
        (vox.z + step.z.max(0)) as f32 * VOXEL_SIZE,
    );
    let big = f32::MAX;
    let pick = |z: bool, v: f32| if z { big } else { v };
    let mut tma = Vec3::new(
        pick(rd.x.abs() < 1e-12, (nb.x - ro.x) * inv.x),
        pick(rd.y.abs() < 1e-12, (nb.y - ro.y) * inv.y),
        pick(rd.z.abs() < 1e-12, (nb.z - ro.z) * inv.z),
    );
    let td = Vec3::new(
        pick(rd.x.abs() < 1e-12, (VOXEL_SIZE * inv.x).abs()),
        pick(rd.y.abs() < 1e-12, (VOXEL_SIZE * inv.y).abs()),
        pick(rd.z.abs() < 1e-12, (VOXEL_SIZE * inv.z).abs()),
    );
    let mut t = 0.0f32;
    for _ in 0..8192 {
        if t > t_max {
            return None;
        }
        let b = map.voxel_block(vox);
        if !b.is_air() {
            return Some((b, t));
        }
        if tma.x < tma.y && tma.x < tma.z {
            t = tma.x;
            tma.x += td.x;
            vox.x += step.x;
        } else if tma.y < tma.z {
            t = tma.y;
            tma.y += td.y;
            vox.y += step.y;
        } else {
            t = tma.z;
            tma.z += td.z;
            vox.z += step.z;
        }
    }
    None
}

/// FAITHFUL CPU port of the WGSL `dda_brick`: grown-AABB slab → DDA the haloed grid from `world_min − csize`
/// (commit only CORE cells) → `(block_id, hit_t)` or `None`. Mirrors `assets/shaders/voxel_raytrace.wgsl`.
fn dda_brick_faithful(patch: &GpuBrickPatch, bi: usize, ro: Vec3, rd: Vec3) -> Option<(u32, f32)> {
    let m = &patch.metas[bi];
    let core = lod_edge(m.lod());
    let hedge = core + 2;
    let csize = lod_voxel_size(m.lod());
    let wmin = Vec3::from(m.world_min);

    // Grown-AABB slab (same as the shader's trace candidate test). `brick_span(m.lod())` is the clipmap span
    // (Cornell is all LOD0, so this is BRICK_WORLD_SIZE here — but use the SSOT so it never drifts).
    let eps = brick_aabb_epsilon(m.lod()); // A4.2: relative-per-LOD grow (SSOT)
    let bmin = wmin - Vec3::splat(eps);
    let bmax = wmin + Vec3::splat(brick_span(m.lod()) + eps);
    let inv = Vec3::new(1.0 / rd.x, 1.0 / rd.y, 1.0 / rd.z);
    let ta = (bmin - ro) * inv;
    let tb = (bmax - ro) * inv;
    let t_enter = ta.min(tb).max_element();
    let t_exit = ta.max(tb).min_element();
    if !(t_enter <= t_exit && t_exit >= 0.0) {
        return None;
    }

    let gmin = wmin - Vec3::splat(csize);
    let t0 = t_enter.max(0.0);
    let p_enter = ro + rd * (t0 + 1e-4);
    let local = (p_enter - gmin) / csize;
    let mut vox = IVec3::new(local.x.floor() as i32, local.y.floor() as i32, local.z.floor() as i32);
    vox = vox.clamp(IVec3::ZERO, IVec3::splat(hedge - 1));

    let step = IVec3::new(rd.x.signum() as i32, rd.y.signum() as i32, rd.z.signum() as i32);
    let nb = gmin
        + Vec3::new(
            (vox.x + step.x.max(0)) as f32,
            (vox.y + step.y.max(0)) as f32,
            (vox.z + step.z.max(0)) as f32,
        ) * csize;
    let big = f32::MAX;
    let pick = |z: bool, v: f32| if z { big } else { v };
    let nz = |c: f32| c.abs() <= 1e-12;
    let mut tma = Vec3::new(
        pick(nz(rd.x), (nb.x - ro.x) * inv.x),
        pick(nz(rd.y), (nb.y - ro.y) * inv.y),
        pick(nz(rd.z), (nb.z - ro.z) * inv.z),
    );
    let td = Vec3::new(
        pick(nz(rd.x), (csize * inv.x).abs()),
        pick(nz(rd.y), (csize * inv.y).abs()),
        pick(nz(rd.z), (csize * inv.z).abs()),
    );
    // SSOT decode via `GpuBrickPatch::cell_block` (R2b) — uniform meta id or dense bit-packed index + palette,
    // exactly as the GPU `cell_block` does (oracle can never drift from the shader).
    let cell = |x: i32, y: i32, z: i32| patch.cell_block(m, (x + y * hedge + z * hedge * hedge) as usize).0 as u32;
    let mut t = t0;
    let lim = 3 * (BRICK_EDGE + 2);
    for _ in 0..lim {
        if vox.x < 0 || vox.x >= hedge || vox.y < 0 || vox.y >= hedge || vox.z < 0 || vox.z >= hedge {
            break;
        }
        let id = cell(vox.x, vox.y, vox.z);
        let is_core = vox.x >= 1 && vox.x <= core && vox.y >= 1 && vox.y <= core && vox.z >= 1 && vox.z <= core;
        if id != 0 && is_core {
            return Some((id, t));
        }
        if tma.x < tma.y && tma.x < tma.z {
            t = tma.x;
            tma.x += td.x;
            vox.x += step.x;
        } else if tma.y < tma.z {
            t = tma.y;
            tma.y += td.y;
            vox.y += step.y;
        } else {
            t = tma.z;
            tma.z += td.z;
            vox.z += step.z;
        }
        if t > t_exit {
            break;
        }
    }
    None
}

/// FAITHFUL port of the WGSL `trace`: DDA every brick whose grown AABB the ray crosses, keep the nearest
/// first-solid commit (mirrors the ray query keeping the global-min committed t).
fn trace_faithful(patch: &GpuBrickPatch, ro: Vec3, rd: Vec3) -> Option<(u32, f32)> {
    let rd = rd.normalize();
    let mut best: Option<(u32, f32)> = None;
    for bi in 0..patch.metas.len() {
        if let Some((id, t)) = dda_brick_faithful(patch, bi, ro, rd)
            && best.map(|(_, bt)| t < bt).unwrap_or(true)
        {
            best = Some((id, t));
        }
    }
    best
}

/// Sweep camera rays through the Cornell box and assert the faithful DDA never commits a hit measurably
/// NEARER than the true first-solid voxel (the show-through bug).
#[test]
fn cornell_rays_have_no_near_phantom_hits() {
    let reg = BlockRegistry::cornell();
    let map = build_cornell(&reg);
    let patch = pack_brickmap(&map, &reg);

    // Eye outside the open −Z front, looking at the interior centre (≈ the editor framing).
    let c = INTERIOR as f32 * 0.5 * VOXEL_SIZE; // interior centre per axis (≈4.8 m)
    let target = Vec3::splat(c);
    let eye = Vec3::new(c, c + 1.5, -8.5);
    let fwd = (target - eye).normalize();
    let right = fwd.cross(Vec3::Y).normalize();
    let up = right.cross(fwd);

    let t_max = 1.0e3;
    let n = 96; // 96×96 ray grid
    let half_fov = 0.45_f32; // radians-ish half extent at the image plane
    let mut violations = 0u32;
    let mut worst = 0.0f32;
    let mut worst_desc = String::new();

    for j in 0..n {
        for i in 0..n {
            let sx = (i as f32 + 0.5) / n as f32 * 2.0 - 1.0;
            let sy = (j as f32 + 0.5) / n as f32 * 2.0 - 1.0;
            let rd = (fwd + right * (sx * half_fov) + up * (-sy * half_fov)).normalize();

            let truth = cpu_first_solid(&map, eye, rd, t_max);
            let got = trace_faithful(&patch, eye, rd);

            if let Some((_, gt)) = got {
                let truth_t = truth.map(|(_, tt)| tt).unwrap_or(t_max);
                // A near-phantom: the faithful DDA committed a hit more than one voxel NEARER than the true
                // first solid voxel along this ray. (One voxel of slack absorbs the grown-AABB epsilon + the
                // brick-face entry-t convention.)
                if gt < truth_t - VOXEL_SIZE {
                    violations += 1;
                    let d = truth_t - gt;
                    if d > worst {
                        worst = d;
                        worst_desc = format!("ray ({i},{j}) rd={rd:?}: committed t={gt:.3} but true first-solid t={truth_t:.3} (Δ={d:.3} m nearer)");
                    }
                }
            }
        }
    }

    assert_eq!(
        violations, 0,
        "{violations} of {} rays commit a near-phantom hit (faces showing through). Worst: {worst_desc}",
        n * n
    );
}
