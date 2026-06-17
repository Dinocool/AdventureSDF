//! **Shared faithful CPU DDA oracle** — a line-for-line port of `assets/shaders/voxel_raytrace.wgsl`'s
//! `dda_brick` + `trace` (the most-head-on exposed-axis normal heuristic INCLUDED), operating on a packed
//! [`GpuBrickPatch`] (meta + bit-packed index stream + per-brick palettes) via the SSOT `cell_block` decode.
//! Because it mirrors the shader exactly AND reads the pool through the SAME `cell_block`, the CPU result IS
//! the GPU result for the SAME pool bytes — so it is the render-identity oracle for any two pools holding the
//! same resident bricks (e.g. the GPU-driven pool vs the CPU-`ResidentPacker` pool in
//! `voxel_gpu_residency_pack_parity.rs`).
//!
//! Extracted from `voxel_normal_swap.rs` (which proved it faithful against the live shader) so multiple
//! integration tests share ONE oracle SSOT. `#![allow(dead_code)]` — each consumer uses a subset.
#![allow(dead_code)]

use adventure::voxel::brickmap::{BRICK_EDGE, BRICK_WORLD_SIZE, lod_edge, lod_voxel_size};
use adventure::voxel::gpu::{GpuBrickPatch, brick_aabb_epsilon};
use bevy::math::{IVec3, Vec3};

/// One brick's DDA result (the shader's packed outputs + the recovered hit cell for diagnostics).
#[derive(Clone, Copy, Debug)]
pub struct BrickHit {
    pub hit_t: f32,
    pub block_id: u32,
    pub best_axis: i32,
    pub enter_axis: i32,
    pub hit_vox: IVec3,
    pub exposed_count: u32,
}

/// The outward unit face normal from `best_axis`, exactly as WGSL `dh_normal`.
pub fn normal_from_axis(axis: i32, rd: Vec3) -> Vec3 {
    let mut n = Vec3::ZERO;
    match axis {
        0 => n.x = -rd.x.signum(),
        1 => n.y = -rd.y.signum(),
        _ => n.z = -rd.z.signum(),
    }
    n
}

/// FAITHFUL CPU port of the WGSL `dda_brick` (incl. the most-head-on exposed-axis normal heuristic).
pub fn dda_brick_faithful(patch: &GpuBrickPatch, bi: usize, ro: Vec3, rd: Vec3) -> Option<BrickHit> {
    let m = &patch.metas[bi];
    let core = lod_edge(m.lod());
    let hedge = core + 2;
    let csize = lod_voxel_size(m.lod());
    let wmin = Vec3::from(m.world_min);

    let eps = brick_aabb_epsilon(m.lod());
    let bmin = wmin - Vec3::splat(eps);
    let bmax = wmin + Vec3::splat(BRICK_WORLD_SIZE + eps);
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
    let p_enter = ro + rd * (t0 + csize * 5.0e-4);
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

    let ta2 = (gmin - ro) * inv;
    let tb2 = (gmin + Vec3::splat(csize * hedge as f32) - ro) * inv;
    let t_near = ta2.min(tb2);
    let enter_axis: i32 = if t_near.y >= t_near.x && t_near.y >= t_near.z {
        1
    } else if t_near.z >= t_near.x && t_near.z >= t_near.y {
        2
    } else {
        0
    };

    let cell = |x: i32, y: i32, z: i32| patch.cell_block(m, (x + y * hedge + z * hedge * hedge) as usize).0 as u32;

    let mut found = false;
    let mut hit_t = -1.0f32;
    let mut hit_id = 0u32;
    let mut hit_vox = IVec3::ZERO;

    let mut t = t0;
    let lim = 3 * (BRICK_EDGE + 2);
    for _ in 0..lim {
        let oob = vox.x < 0 || vox.x >= hedge || vox.y < 0 || vox.y >= hedge || vox.z < 0 || vox.z >= hedge;
        if oob || found {
            break;
        }
        let id = cell(vox.x, vox.y, vox.z);
        let is_core = vox.x >= 1 && vox.x <= core && vox.y >= 1 && vox.y <= core && vox.z >= 1 && vox.z <= core;
        if id != 0 && is_core {
            found = true;
            hit_t = t;
            hit_id = id;
            hit_vox = vox;
        } else if tma.x < tma.y && tma.x < tma.z {
            t = tma.x;
            tma.x += td.x;
            vox.x += step.x;
            if t > t_exit {
                break;
            }
        } else if tma.y < tma.z {
            t = tma.y;
            tma.y += td.y;
            vox.y += step.y;
            if t > t_exit {
                break;
            }
        } else {
            t = tma.z;
            tma.z += td.z;
            vox.z += step.z;
            if t > t_exit {
                break;
            }
        }
    }

    if !found {
        return None;
    }

    let mut best_axis = enter_axis;
    let mut best_score = -1.0f32;
    let mut exposed_count = 0u32;
    for a in 0..3 {
        let s = step[a];
        if s == 0 {
            continue;
        }
        let mut nb = hit_vox;
        nb[a] -= s;
        let nb_oob = nb.x < 0 || nb.x >= hedge || nb.y < 0 || nb.y >= hedge || nb.z < 0 || nb.z >= hedge;
        let nb_solid = if nb_oob { false } else { cell(nb.x, nb.y, nb.z) != 0 };
        if !nb_solid {
            exposed_count += 1;
            let score = rd[a].abs();
            if score > best_score {
                best_score = score;
                best_axis = a as i32;
            }
        }
    }

    Some(BrickHit { hit_t, block_id: hit_id, best_axis, enter_axis, hit_vox, exposed_count })
}

/// The faithful trace output: the winning brick + its hit + the world normal.
pub struct TraceOut {
    pub best_bi: usize,
    pub best: BrickHit,
    pub normal: Vec3,
}

/// FAITHFUL port of the WGSL `trace`: DDA every brick whose grown AABB the ray crosses; keep the nearest
/// first-solid voxel t (the `< best_t` arbitration). `tie_eps` is unused here (kept for signature stability
/// with `voxel_normal_swap`'s rival diagnostics).
pub fn trace_faithful(patch: &GpuBrickPatch, ro: Vec3, rd: Vec3, _tie_eps: f32) -> Option<TraceOut> {
    let rd = rd.normalize();
    let mut hits: Vec<(usize, BrickHit)> = Vec::new();
    for bi in 0..patch.metas.len() {
        if let Some(h) = dda_brick_faithful(patch, bi, ro, rd) {
            hits.push((bi, h));
        }
    }
    let (&(best_bi, best), _) =
        hits.iter().map(|x| (x, x.1.hit_t)).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())?;
    let normal = normal_from_axis(best.best_axis, rd);
    Some(TraceOut { best_bi, best, normal })
}
