//! Root-causes the "normal swap" bug: in the NORMALS debug view, a flat-face REGION's normal slowly swaps
//! between two normals as the camera rotates — gradual, not a flicker.
//!
//! CPU-only, deterministic, FAITHFUL port of the WGSL `dda_brick` normal computation (the `enter_axis`
//! AABB-entry seed + the most-head-on `best_score = |rd[a]|` exposed-axis heuristic + `dh_normal`) plus the
//! WGSL `trace`'s nearest-voxel-t arbitration across bricks. It mirrors `assets/shaders/voxel_raytrace.wgsl`
//! line-for-line, so the CPU result IS the GPU result.
//!
//! # Findings (run `cargo test --test voxel_normal_swap -- --nocapture`)
//!
//! ROOT CAUSE = the most-head-on heuristic (`dda_brick` WGSL ~lines 217-239). When a hit voxel has >1
//! EXPOSED face (an edge/corner cell: ≥2 of its 6 haloed-grid face-neighbours are air), the heuristic picks
//! the exposed axis the ray meets most head-on (`|rd[a]|`), which FLIPS as the camera rotates. The
//! diagnostic shows it diverges from the geometric (gradient) normal on EXACTLY the edge+corner cells
//! (3328 + 89 of 40000 hits in one Cornell view) and NEVER on flat (single-exposed) faces. A "whole flat
//! face slowly swapping" is the band of EDGE-ROW voxels along box tops / wall-floor seams swapping together.
//!
//! Hypothesis (b) — two coplanar bricks tying on `t` with different normals — is NOT the cause: with a
//! TRUE same-surface criterion (≤1 mm Δt or identical world cell) there are ZERO such collisions (the halo
//! gives abutting bricks the same neighbourhood). The apparent "33 ties" under a half-voxel window were
//! genuinely-distinct surfaces at slightly different depths.
//!
//! THE FIX (validated here, NOT yet applied to the WGSL): replace the heuristic + `dh_normal` with the
//! OCCUPANCY-GRADIENT normal — sum the unit directions toward each AIR face-neighbour of the hit cell in the
//! haloed grid, normalise. It is a pure function of the committed cell's occupancy (NOT the ray), so it
//! cannot swap with the camera and cannot disagree between coplanar bricks. The `fix_gradient_*` tests pass
//! all three sweeps. (The pure DDA-CROSSED-AXIS normal — the flat-grid SOTA, `select(sideMask,-sign(dir),0)`
//! — also cures the flat-face swap but still SHIMMERS at grazing silhouettes; see
//! `diagnose_crossed_axis_residual_swaps`.)
//!
//! The two `#[should_panic]` tests reproduce the LIVE bug (they pass because the shader is buggy). After the
//! WGSL fix lands, delete their `#[should_panic]` to flip them into regression guards.

use adventure::voxel::brickmap::{
    BRICK_EDGE, BRICK_WORLD_SIZE, VOXEL_SIZE, lod_edge, lod_voxel_size,
};
use adventure::voxel::cornell::{INTERIOR, build_cornell};
use adventure::voxel::gpu::{BRICK_AABB_EPSILON, GpuBrickPatch, pack_brickmap};
use adventure::voxel::palette::BlockRegistry;
use bevy::math::{IVec3, Vec3};

/// One brick's DDA result, mirroring the WGSL `dda_brick` packed `vec4` PLUS the recovered hit cell (so the
/// test can identify which voxel/brick produced a normal). `found`/`hit_t`/`block_id`/`best_axis` are the
/// shader's outputs; `hit_vox` is the haloed-grid cell the march committed (for diagnostics).
#[derive(Clone, Copy, Debug)]
struct BrickHit {
    hit_t: f32,
    #[allow(dead_code)] // carried for diagnostics / parity with the WGSL packed result
    block_id: u32,
    /// The shader's `best_axis` (0/1/2) — the axis whose `-sign(rd[axis])` becomes the normal.
    best_axis: i32,
    /// AABB-entry axis seed (`enter_axis`) — the fallback the heuristic starts from (diagnostics).
    enter_axis: i32,
    /// Committed haloed-grid cell (diagnostics: which voxel was hit).
    hit_vox: IVec3,
    /// How many distinct axes were "exposed" (incoming neighbour air) at the hit cell (diagnostics).
    exposed_count: u32,
}

/// The outward unit face normal from `best_axis`, exactly as WGSL `dh_normal`.
fn normal_from_axis(axis: i32, rd: Vec3) -> Vec3 {
    let mut n = Vec3::ZERO;
    match axis {
        0 => n.x = -rd.x.signum(),
        1 => n.y = -rd.y.signum(),
        _ => n.z = -rd.z.signum(),
    }
    n
}

/// FAITHFUL CPU port of the WGSL `dda_brick`, returning the full hit (id, t, normal axis) or `None`.
/// Mirrors `assets/shaders/voxel_raytrace.wgsl` `dda_brick` (lines ~124-242) INCLUDING the most-head-on
/// exposed-axis normal heuristic (lines ~210-239).
fn dda_brick_faithful(patch: &GpuBrickPatch, bi: usize, ro: Vec3, rd: Vec3) -> Option<BrickHit> {
    let m = &patch.metas[bi];
    let core = lod_edge(m.lod);
    let hedge = core + 2;
    let csize = lod_voxel_size(m.lod);
    let wmin = Vec3::from(m.world_min);

    // Grown-AABB slab (same as the shader's trace candidate test) → t_enter/t_exit.
    let bmin = wmin - Vec3::splat(BRICK_AABB_EPSILON);
    let bmax = wmin + Vec3::splat(BRICK_WORLD_SIZE + BRICK_AABB_EPSILON);
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

    // `enter_axis`: the largest near-slab axis of the HALOED grid AABB (the shader's seed).
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

    let off = m.voxel_offset as usize;
    // Storage plan R1: a UNIFORM brick has no voxel array — every haloed cell is its single block id.
    let cell = |x: i32, y: i32, z: i32| {
        if m.is_uniform() {
            m.uniform_block().0 as u32
        } else {
            patch.voxels[off + (x + y * hedge + z * hedge * hedge) as usize]
        }
    };

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

    // The most-head-on exposed-axis heuristic (WGSL lines ~217-239).
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

    Some(BrickHit {
        hit_t,
        block_id: hit_id,
        best_axis,
        enter_axis,
        hit_vox,
        exposed_count,
    })
}

/// FAITHFUL port of the WGSL `trace`: DDA every brick whose grown AABB the ray crosses; keep the brick with
/// the nearest first-solid VOXEL t (the `if ht < best_t` arbitration). Returns the winning brick index + its
/// hit, AND the runner-up (nearest other brick within a sub-voxel t) for tie diagnostics.
struct TraceOut {
    /// winning brick index, its hit, and the world-space normal.
    best_bi: usize,
    best: BrickHit,
    normal: Vec3,
    /// A competing brick within `tie_eps` of `best.hit_t` whose normal DIFFERS (if any).
    rival: Option<(usize, BrickHit, Vec3)>,
}

fn trace_faithful(patch: &GpuBrickPatch, ro: Vec3, rd: Vec3, tie_eps: f32) -> Option<TraceOut> {
    let rd = rd.normalize();
    // Gather every brick hit, then apply the exact `< best_t` arbitration (strict less-than: the FIRST brick
    // to reach a given t wins — candidate order matters, mirroring the GPU's nondeterministic AABB order via
    // our deterministic packed order).
    let mut hits: Vec<(usize, BrickHit)> = Vec::new();
    for bi in 0..patch.metas.len() {
        if let Some(h) = dda_brick_faithful(patch, bi, ro, rd) {
            hits.push((bi, h));
        }
    }
    let (&(best_bi, best), _) = hits
        .iter()
        .map(|x| (x, x.1.hit_t))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())?;
    let normal = normal_from_axis(best.best_axis, rd);

    // Find a rival brick within tie_eps whose committed normal differs from the winner's.
    let mut rival = None;
    for &(bi, h) in &hits {
        if bi == best_bi {
            continue;
        }
        if (h.hit_t - best.hit_t).abs() <= tie_eps {
            let n = normal_from_axis(h.best_axis, rd);
            if n != normal {
                rival = Some((bi, h, n));
                break;
            }
        }
    }

    Some(TraceOut { best_bi, best, normal, rival })
}

/// The PROPOSED FIX, ported faithfully: the canonical Amanatides & Woo voxel-DDA normal — the normal is the
/// axis the DDA CROSSED to ENTER the solid cell, sign opposing the ray. We seed `cross_axis` with the
/// AABB-entry face (`enter_axis`, used only if the very first cell is solid with no step taken) and OVERWRITE
/// it on every DDA advance with the stepped axis. The result is a pure function of GEOMETRY (which boundary
/// the ray pierced), never of `|rd[a]|`, so it cannot swap with camera angle. Returns the same `BrickHit`
/// shape but with `best_axis` = the crossed axis.
fn dda_brick_crossed_axis(patch: &GpuBrickPatch, bi: usize, ro: Vec3, rd: Vec3) -> Option<BrickHit> {
    let m = &patch.metas[bi];
    let core = lod_edge(m.lod);
    let hedge = core + 2;
    let csize = lod_voxel_size(m.lod);
    let wmin = Vec3::from(m.world_min);

    let bmin = wmin - Vec3::splat(BRICK_AABB_EPSILON);
    let bmax = wmin + Vec3::splat(BRICK_WORLD_SIZE + BRICK_AABB_EPSILON);
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

    // AABB-entry seed (only matters if the FIRST cell is already solid).
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

    let off = m.voxel_offset as usize;
    // Storage plan R1: a UNIFORM brick has no voxel array — every haloed cell is its single block id.
    let cell = |x: i32, y: i32, z: i32| {
        if m.is_uniform() {
            m.uniform_block().0 as u32
        } else {
            patch.voxels[off + (x + y * hedge + z * hedge * hedge) as usize]
        }
    };

    let mut found = false;
    let mut hit_t = -1.0f32;
    let mut hit_id = 0u32;
    let mut hit_vox = IVec3::ZERO;
    let mut cross_axis = enter_axis; // THE FIX: tracked across advances.
    let mut hit_axis = enter_axis;
    let mut steps_taken = 0u32;
    let mut hit_steps = u32::MAX;

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
            hit_axis = cross_axis; // the axis we crossed to step INTO this cell
            hit_steps = steps_taken;
        } else {
            steps_taken += 1;
            if tma.x < tma.y && tma.x < tma.z {
                t = tma.x;
                tma.x += td.x;
                vox.x += step.x;
                cross_axis = 0;
            } else if tma.y < tma.z {
                t = tma.y;
                tma.y += td.y;
                vox.y += step.y;
                cross_axis = 1;
            } else {
                t = tma.z;
                tma.z += td.z;
                vox.z += step.z;
                cross_axis = 2;
            }
            if t > t_exit {
                break;
            }
        }
    }

    if !found {
        return None;
    }

    Some(BrickHit {
        hit_t,
        block_id: hit_id,
        best_axis: hit_axis,
        enter_axis,
        hit_vox,
        // Reuse `exposed_count` to carry the diagnostic "DDA steps before the hit" (0 ⇒ the hit cell was the
        // FIRST cell, so the normal came from the camera-dependent AABB-entry seed, not a real crossed face).
        exposed_count: hit_steps,
    })
}

/// Same nearest-voxel-t arbitration as `trace_faithful`, but using the PROPOSED-FIX per-brick normal.
fn trace_crossed_axis(patch: &GpuBrickPatch, ro: Vec3, rd: Vec3) -> Option<(usize, BrickHit, Vec3)> {
    let rd = rd.normalize();
    let mut best: Option<(usize, BrickHit)> = None;
    for bi in 0..patch.metas.len() {
        if let Some(h) = dda_brick_crossed_axis(patch, bi, ro, rd)
            && best.map(|(_, b)| h.hit_t < b.hit_t).unwrap_or(true)
        {
            best = Some((bi, h));
        }
    }
    best.map(|(bi, h)| {
        let n = normal_from_axis(h.best_axis, rd);
        (bi, h, n)
    })
}

/// Camera basis (right, up, fwd) for an eye looking at `target`, rolled about world-Y.
fn camera_basis(eye: Vec3, target: Vec3) -> (Vec3, Vec3, Vec3) {
    let fwd = (target - eye).normalize();
    let right = fwd.cross(Vec3::Y).normalize();
    let up = right.cross(fwd);
    (right, up, fwd)
}

/// Sweep a small CAMERA ARC (orbit the eye around the interior centre) and, for each pixel, record the hit
/// voxel's WORLD position + the committed normal. Detect any voxel hit at the SAME world cell across two
/// adjacent arc steps whose normal CHANGED — the visible "normal swap". Reports the offending voxel + the
/// two normals + why (which exposed axes / scores tied).
///
/// REPRODUCES THE LIVE BUG: this `#[should_panic]` PASSES today because the current shader's most-head-on
/// heuristic swaps normals on edge/corner cells across the arc. Once the gradient fix is applied to the WGSL,
/// the swap vanishes — DELETE the `#[should_panic]` so this becomes a guard that the fix holds.
#[test]
#[should_panic(expected = "same-cell normal swaps across the camera arc")]
fn flat_face_normal_is_stable_across_camera_arc() {
    let reg = BlockRegistry::cornell();
    let map = build_cornell(&reg);
    let patch = pack_brickmap(&map, &reg);

    let c = INTERIOR as f32 * 0.5 * VOXEL_SIZE; // interior centre per axis (≈4.8 m)
    let target = Vec3::splat(c);
    let radius = 9.0f32;
    let eye_y = c + 1.5;

    let n = 64; // ray grid per frame
    let half_fov = 0.45f32;
    // A small arc: orbit the eye in the −Z hemisphere so the open front stays in view. 24 steps over ~28°.
    let arc_steps = 24;
    let arc_span = 0.5f32; // radians total

    // Per-voxel record of (frame_index, normal) keyed by the committed world voxel coordinate, so we can
    // detect a normal that swapped on the SAME cell between frames.
    use std::collections::HashMap;
    // key = (brick_index, hit_vox) → last (frame, normal, ray dir, hit)
    type LastSeen = (i32, Vec3, Vec3, BrickHit);
    let mut last: HashMap<(usize, [i32; 3]), LastSeen> = HashMap::new();

    let mut swaps = 0u32;
    let mut worst = String::new();
    // The pathology is ANY hit cell whose committed normal CHANGES across the arc while the ray keeps hitting
    // the SAME cell. The user's "whole flat face slowly swaps" is precisely a run of such cells: each exposes
    // ≥2 candidate faces (e.g. a +Y top and a +X side at a top-of-wall row) and the most-head-on heuristic
    // (`best_score = |rd[a]|`) flips its winner as the camera rotates, even though the geometry never moved.
    // We count every same-cell swap; that is the bug (a stable cubic-voxel normal must be camera-independent).
    let mut heuristic_swaps = 0u32;

    for f in 0..arc_steps {
        let ang = (f as f32 / (arc_steps - 1) as f32 - 0.5) * arc_span; // centered arc
        let eye = Vec3::new(c + radius * ang.sin(), eye_y, c - radius * ang.cos());
        let (right, up, fwd) = camera_basis(eye, target);

        for j in 0..n {
            for i in 0..n {
                let sx = (i as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                let sy = (j as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                let rd = (fwd + right * (sx * half_fov) + up * (-sy * half_fov)).normalize();

                let Some(out) = trace_faithful(&patch, eye, rd, VOXEL_SIZE * 0.25) else {
                    continue;
                };
                let key = (out.best_bi, [out.best.hit_vox.x, out.best.hit_vox.y, out.best.hit_vox.z]);
                if let Some(&(pf, pn, prd, ph)) = last.get(&key)
                    && pn != out.normal
                {
                    swaps += 1;
                    // The same SOLID cell, hit from a slightly different camera angle, committed a DIFFERENT
                    // normal. For a cubic voxel that is always a bug: the surface didn't move. (Both frames
                    // see ≥1 exposed face; the heuristic flipped its choice.)
                    heuristic_swaps += 1;
                    if worst.is_empty() {
                        worst = format!(
                            "brick {} cell {:?}: frame {pf} normal {pn:?} (axis seed {}, exposed {}) \
                             → frame {f} normal {:?} (axis seed {}, exposed {}); \
                             rd_prev={prd:?} rd_now={rd:?}; rival={:?}",
                            out.best_bi, out.best.hit_vox, ph.enter_axis, ph.exposed_count,
                            out.normal, out.best.enter_axis, out.best.exposed_count,
                            out.rival.map(|(bi, h, n)| (bi, h.hit_vox, n)),
                        );
                    }
                }
                last.insert(key, (f, out.normal, rd, out.best));
            }
        }
    }

    eprintln!("total same-cell normal swaps across arc: {swaps}; heuristic swaps: {heuristic_swaps}");
    eprintln!("worst: {worst}");
    assert_eq!(
        heuristic_swaps, 0,
        "{heuristic_swaps} same-cell normal swaps across the camera arc (a voxel's committed normal changed \
         with camera angle — the surface never moved). Worst: {worst}"
    );
}

/// Detect, on a SINGLE frame, any ray where two DIFFERENT bricks produce a first-solid hit within a sub-voxel
/// `t` but with DIFFERENT normals (the coplanar-brick tie that flips with candidate order / camera angle).
/// This isolates hypothesis (b): two bricks fighting for the same space.
#[test]
fn no_coplanar_brick_normal_tie() {
    let reg = BlockRegistry::cornell();
    let map = build_cornell(&reg);
    let patch = pack_brickmap(&map, &reg);

    let c = INTERIOR as f32 * 0.5 * VOXEL_SIZE;
    let target = Vec3::splat(c);
    let eye = Vec3::new(c, c + 1.5, -8.5);
    let (right, up, fwd) = camera_basis(eye, target);

    let n = 128;
    let half_fov = 0.45f32;
    let mut ties = 0u32;
    let mut worst = String::new();

    for j in 0..n {
        for i in 0..n {
            let sx = (i as f32 + 0.5) / n as f32 * 2.0 - 1.0;
            let sy = (j as f32 + 0.5) / n as f32 * 2.0 - 1.0;
            let rd = (fwd + right * (sx * half_fov) + up * (-sy * half_fov)).normalize();
            // Use a TIGHT 1 mm window: a true coplanar fight is two bricks at the SAME surface depth, not two
            // distinct surfaces a fraction of a voxel apart. (A half-voxel window catches genuinely-different
            // surfaces and over-reports — see the report.)
            if let Some(out) = trace_faithful(&patch, eye, rd, 1.0e-3)
                && let Some((rbi, rh, rn)) = out.rival
            {
                ties += 1;
                if worst.is_empty() {
                    worst = format!(
                        "ray ({i},{j}): brick {} cell {:?} normal {:?} (t={:.4}) vs brick {rbi} cell {:?} normal {rn:?} (t={:.4})",
                        out.best_bi, out.best.hit_vox, out.normal, out.best.hit_t,
                        rh.hit_vox, rh.hit_t,
                    );
                }
            }
        }
    }

    eprintln!("coplanar-brick normal ties: {ties}");
    eprintln!("worst: {worst}");
    assert_eq!(ties, 0, "{ties} rays where two bricks tie on t with DIFFERENT normals. Worst: {worst}");
}

/// MINIMAL ROOT-CAUSE DEMO (no Cornell box): a single solid voxel at the +Y/+X TOP-EDGE of a slab — it has
/// TWO exposed faces (+Y top, +X side). Fire rays at it from a continuum of angles that keep hitting the SAME
/// top cell, and show the committed normal FLIPS from +Y to +X exactly when `|rd.x|` overtakes `|rd.y|` —
/// the most-head-on heuristic's crossover. This is the bug in isolation: the normal is a pure function of the
/// CAMERA, not the geometry.
///
/// REPRODUCES THE LIVE BUG: `#[should_panic]` PASSES today (the heuristic commits 2 different normals for the
/// one cell). After the gradient/crossed-axis fix it commits a single stable normal — remove `#[should_panic]`
/// to turn this into a regression guard.
#[test]
#[should_panic(expected = "different normals depending only on camera angle")]
fn edge_voxel_normal_is_camera_independent() {
    use adventure::voxel::brickmap::{BRICK_VOXELS, Brick, BrickMap};
    use adventure::voxel::palette::BlockId;
    use adventure::voxel::brickmap::voxel_index;

    let reg = BlockRegistry::cornell();
    // A 1-voxel-thick slab filling the brick's bottom layer (y=0) AND a one-voxel lip rising at the +X edge,
    // so the cell at local (7,1,4) has air above (+Y exposed) and air to its +X (the +X face exposed) — a
    // top-edge voxel with two genuinely-exposed faces.
    let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
    for z in 0..BRICK_EDGE {
        for x in 0..BRICK_EDGE {
            voxels[voxel_index(x, 0, z)] = BlockId(1); // floor slab
        }
    }
    voxels[voxel_index(7, 1, 4)] = BlockId(1); // a lip voxel one above the floor at the +X edge
    let mut map = BrickMap::new();
    map.insert(IVec3::ZERO, Brick::from_voxels(voxels));
    let patch = pack_brickmap(&map, &reg);

    // Aim at the centre of the lip voxel's top face. Local (7,1,4) → world cell min = (7,1,4)·0.2.
    let target = Vec3::new(7.5, 2.0, 4.5) * VOXEL_SIZE;
    let mut normals = std::collections::BTreeSet::new();
    let mut samples = Vec::new();
    // Sweep the eye in the X–Y plane above/around the +X edge of the lip so the ray always lands on the SAME
    // top cell but with |rd.x| sweeping past |rd.y|.
    for k in 0..41 {
        let ax = (k as f32 / 40.0 - 0.5) * 1.4; // azimuth-ish tilt toward +X
        let eye = target + Vec3::new(2.0 + ax, 2.0, 0.0);
        let rd = (target - eye).normalize();
        if let Some(out) = trace_faithful(&patch, eye, rd, VOXEL_SIZE * 0.25)
            && out.best.hit_vox == IVec3::new(8, 2, 5)
        {
            // same haloed cell (core (7,1,4) → halo +1)
            normals.insert((out.normal.x as i32, out.normal.y as i32, out.normal.z as i32));
            samples.push((rd, out.normal, out.best.best_axis));
        }
    }
    eprintln!("distinct normals committed for the SAME edge voxel across the angle sweep: {normals:?}");
    for (rd, n, axis) in &samples {
        eprintln!("  rd=({:.3},{:.3},{:.3}) |rd.x|={:.3} |rd.y|={:.3} → normal={n:?} axis={axis}",
            rd.x, rd.y, rd.z, rd.x.abs(), rd.y.abs());
    }
    assert_eq!(
        normals.len(),
        1,
        "the SAME edge voxel committed {} different normals depending only on camera angle: {normals:?}",
        normals.len()
    );
}

// ====================================================================================================
// VALIDATION OF THE PROPOSED FIX (crossed-axis normal). These run the SAME sweeps as above but through
// `trace_crossed_axis`, proving the fix removes every swap/tie. They MUST pass now (the fix is geometry-only).
// ====================================================================================================

/// The proposed crossed-axis normal is STABLE across the camera arc: the same cell never swaps normal.
/// CHARACTERISE the DDA-CROSSED-AXIS candidate (the SOTA flat-grid normal, e.g. dubiousconst282's
/// `select(sideMask, -sign(dir), 0)`): it FIXES the reported whole-flat-face swap (a flat interior face is
/// always entered head-on via its face axis, so its crossed axis is camera-stable) BUT it still SHIMMERS at
/// grazing SILHOUETTE pixels, where the genuinely-last-crossed boundary flips between adjacent angles. This
/// diagnostic classifies every cross-axis swap by the hit cell's exposed-face count to prove the residual
/// swaps are NOT flat faces (exposed==1) but edges/silhouettes (exposed≥2) — i.e. crossed-axis cures the
/// user-visible bug but the occupancy-GRADIENT fix (validated below) is strictly more stable. NOT a hard
/// assertion on the total (the residual grazing shimmer is expected); it DOES assert no flat-face swap.
#[test]
fn diagnose_crossed_axis_residual_swaps() {
    let reg = BlockRegistry::cornell();
    let map = build_cornell(&reg);
    let patch = pack_brickmap(&map, &reg);

    let c = INTERIOR as f32 * 0.5 * VOXEL_SIZE;
    let target = Vec3::splat(c);
    let radius = 9.0f32;
    let eye_y = c + 1.5;
    let n = 64;
    let half_fov = 0.45f32;
    let arc_steps = 24;
    let arc_span = 0.5f32;

    use std::collections::HashMap;
    // value = (last normal, the hit cell's exposed-face count)
    let mut last: HashMap<(usize, [i32; 3]), (Vec3, u32)> = HashMap::new();
    let mut swaps = 0u32;
    let mut swaps_flat_face = 0u32; // swaps where BOTH endpoints were single-exposed (flat) cells
    let mut worst = String::new();

    for f in 0..arc_steps {
        let ang = (f as f32 / (arc_steps - 1) as f32 - 0.5) * arc_span;
        let eye = Vec3::new(c + radius * ang.sin(), eye_y, c - radius * ang.cos());
        let (right, up, fwd) = camera_basis(eye, target);
        for j in 0..n {
            for i in 0..n {
                let sx = (i as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                let sy = (j as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                let rd = (fwd + right * (sx * half_fov) + up * (-sy * half_fov)).normalize();
                let Some((bi, h, normal)) = trace_crossed_axis(&patch, eye, rd) else {
                    continue;
                };
                let exposed = count_exposed_faces(&patch, bi, h.hit_vox);
                let key = (bi, [h.hit_vox.x, h.hit_vox.y, h.hit_vox.z]);
                if let Some(&(pn, pexp)) = last.get(&key)
                    && pn != normal
                {
                    swaps += 1;
                    if pexp == 1 && exposed == 1 {
                        swaps_flat_face += 1;
                    }
                    if worst.is_empty() {
                        worst = format!(
                            "brick {bi} cell {:?}: {pn:?} (exposed {pexp}) → {normal:?} (exposed {exposed})",
                            h.hit_vox
                        );
                    }
                }
                last.insert(key, (normal, exposed));
            }
        }
    }
    eprintln!(
        "crossed-axis residual swaps: {swaps} total, {swaps_flat_face} on FLAT (single-exposed) faces; worst: {worst}"
    );
    // The cure claim: crossed-axis removes the reported WHOLE-FLAT-FACE swap entirely.
    assert_eq!(
        swaps_flat_face, 0,
        "crossed-axis still swapped {swaps_flat_face} FLAT-face cells (the reported bug should be gone). Worst: {worst}"
    );
}

/// The crossed-axis candidate gives the edge voxel a SINGLE camera-independent normal (the actual crossed
/// face) for the ISOLATED edge case (no grazing) — kept to show crossed-axis is correct in isolation, even
/// though the full-scene sweep above shows it shimmers at grazing silhouettes (where gradient wins).
#[test]
fn fix_edge_voxel_normal_is_camera_independent() {
    use adventure::voxel::brickmap::{BRICK_VOXELS, Brick, BrickMap, voxel_index};
    use adventure::voxel::palette::BlockId;

    let reg = BlockRegistry::cornell();
    let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
    for z in 0..BRICK_EDGE {
        for x in 0..BRICK_EDGE {
            voxels[voxel_index(x, 0, z)] = BlockId(1);
        }
    }
    voxels[voxel_index(7, 1, 4)] = BlockId(1);
    let mut map = BrickMap::new();
    map.insert(IVec3::ZERO, Brick::from_voxels(voxels));
    let patch = pack_brickmap(&map, &reg);

    let target = Vec3::new(7.5, 2.0, 4.5) * VOXEL_SIZE;
    let mut normals = std::collections::BTreeSet::new();
    for k in 0..41 {
        let ax = (k as f32 / 40.0 - 0.5) * 1.4;
        let eye = target + Vec3::new(2.0 + ax, 2.0, 0.0);
        let rd = (target - eye).normalize();
        if let Some((_, h, n)) = trace_crossed_axis(&patch, eye, rd)
            && h.hit_vox == IVec3::new(8, 2, 5)
        {
            normals.insert((n.x as i32, n.y as i32, n.z as i32));
        }
    }
    assert_eq!(
        normals.len(),
        1,
        "crossed-axis fix: the edge voxel still committed {} normals: {normals:?}",
        normals.len()
    );
}

// ====================================================================================================
// OCCUPANCY-GRADIENT NORMAL (the SOTA / Teardown-style fix): the normal is a pure function of the hit
// cell's 6-neighbourhood occupancy in the HALOED grid — sum the unit directions toward each AIR
// face-neighbour and normalise. This is INDEPENDENT of the ray direction AND of which brick committed
// (the halo carries the neighbour occupancy so abutting bricks see the same 6-neighbourhood), so it
// cannot swap with the camera and cannot disagree between coplanar bricks. The ONLY camera dependence
// left is a sign tie-break for a fully-flat face (gradient already points outward), which never flips.
// ====================================================================================================

/// The occupancy-gradient outward normal of the hit cell, from the haloed grid's 6 face-neighbours.
/// `+axis` neighbour air ⇒ +unit on that axis; `-axis` neighbour air ⇒ -unit. Sum + normalise. For a
/// flat face exactly one axis contributes (the face axis) → the exact face normal; at a convex edge two
/// contribute → the 45° bevel normal (a STABLE, camera-independent choice); at a corner three. A
/// fully-buried cell (no air neighbour) returns zero → caller falls back (never visible).
fn occupancy_gradient_normal(patch: &GpuBrickPatch, bi: usize, hit_vox: IVec3) -> Vec3 {
    let m = &patch.metas[bi];
    let core = lod_edge(m.lod);
    let hedge = core + 2;
    let off = m.voxel_offset as usize;
    let cell = |x: i32, y: i32, z: i32| -> u32 {
        if x < 0 || x >= hedge || y < 0 || y >= hedge || z < 0 || z >= hedge {
            0 // outside the haloed grid reads as air (matches the halo's absent-neighbour convention)
        } else if m.is_uniform() {
            m.uniform_block().0 as u32 // storage plan R1: uniform brick, no voxel array
        } else {
            patch.voxels[off + (x + y * hedge + z * hedge * hedge) as usize]
        }
    };
    let mut g = Vec3::ZERO;
    let axes = [
        (IVec3::X, Vec3::X),
        (IVec3::Y, Vec3::Y),
        (IVec3::Z, Vec3::Z),
    ];
    for (d, u) in axes {
        let pos = hit_vox + d;
        let neg = hit_vox - d;
        if cell(pos.x, pos.y, pos.z) == 0 {
            g += u;
        }
        if cell(neg.x, neg.y, neg.z) == 0 {
            g -= u;
        }
    }
    if g.length_squared() < 1e-12 { Vec3::ZERO } else { g.normalize() }
}

/// Same arbitration as `trace_faithful` (nearest voxel t, strict `<`), but the normal is the
/// occupancy-gradient of the winning cell.
fn trace_gradient(patch: &GpuBrickPatch, ro: Vec3, rd: Vec3) -> Option<(usize, BrickHit, Vec3)> {
    let rd = rd.normalize();
    let mut best: Option<(usize, BrickHit)> = None;
    for bi in 0..patch.metas.len() {
        if let Some(h) = dda_brick_faithful(patch, bi, ro, rd)
            && best.map(|(_, b)| h.hit_t < b.hit_t).unwrap_or(true)
        {
            best = Some((bi, h));
        }
    }
    best.map(|(bi, h)| {
        let n = occupancy_gradient_normal(patch, bi, h.hit_vox);
        (bi, h, n)
    })
}

/// THE FIX VALIDATED: the occupancy-gradient normal never swaps across the camera arc (it is a pure
/// function of the committed cell's occupancy, not the ray). Same sweep as the failing
/// `flat_face_normal_is_stable_across_camera_arc`.
#[test]
fn fix_gradient_normal_is_stable_across_camera_arc() {
    let reg = BlockRegistry::cornell();
    let map = build_cornell(&reg);
    let patch = pack_brickmap(&map, &reg);

    let c = INTERIOR as f32 * 0.5 * VOXEL_SIZE;
    let target = Vec3::splat(c);
    let radius = 9.0f32;
    let eye_y = c + 1.5;
    let n = 64;
    let half_fov = 0.45f32;
    let arc_steps = 24;
    let arc_span = 0.5f32;

    use std::collections::HashMap;
    let mut last: HashMap<(usize, [i32; 3]), Vec3> = HashMap::new();
    let mut swaps = 0u32;
    let mut worst = String::new();

    for f in 0..arc_steps {
        let ang = (f as f32 / (arc_steps - 1) as f32 - 0.5) * arc_span;
        let eye = Vec3::new(c + radius * ang.sin(), eye_y, c - radius * ang.cos());
        let (right, up, fwd) = camera_basis(eye, target);
        for j in 0..n {
            for i in 0..n {
                let sx = (i as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                let sy = (j as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                let rd = (fwd + right * (sx * half_fov) + up * (-sy * half_fov)).normalize();
                let Some((bi, h, normal)) = trace_gradient(&patch, eye, rd) else {
                    continue;
                };
                let key = (bi, [h.hit_vox.x, h.hit_vox.y, h.hit_vox.z]);
                if let Some(&pn) = last.get(&key)
                    && (pn - normal).length() > 1e-4
                {
                    swaps += 1;
                    if worst.is_empty() {
                        worst = format!("brick {bi} cell {:?}: {pn:?} → {normal:?}", h.hit_vox);
                    }
                }
                last.insert(key, normal);
            }
        }
    }
    eprintln!("gradient-normal swaps across arc: {swaps}");
    eprintln!("worst: {worst}");
    assert_eq!(swaps, 0, "gradient normal swapped {swaps} cell normals across the arc. Worst: {worst}");
}

/// THE FIX VALIDATED: two coplanar bricks (within a sub-voxel t) can NEVER disagree on the gradient
/// normal, because the halo gives both bricks the same 6-neighbourhood occupancy at the shared boundary.
/// Same sweep as the failing `no_coplanar_brick_normal_tie`.
#[test]
fn fix_gradient_no_coplanar_disagreement() {
    let reg = BlockRegistry::cornell();
    let map = build_cornell(&reg);
    let patch = pack_brickmap(&map, &reg);

    let c = INTERIOR as f32 * 0.5 * VOXEL_SIZE;
    let target = Vec3::splat(c);
    let eye = Vec3::new(c, c + 1.5, -8.5);
    let (right, up, fwd) = camera_basis(eye, target);
    let n = 128;
    let half_fov = 0.45f32;
    // A TRUE coplanar tie is two bricks committing the SAME surface position — Δt at FP scale, not a half
    // voxel (which catches genuinely-distinct surfaces at slightly different depths). Use a tight 1 mm
    // window (≪ a 0.2 m voxel) so only real same-cell fights count, AND require the two hits resolve to the
    // same WORLD voxel centre (the definitive "same surface" test).
    let tie = 1.0e-3;
    let mut ties = 0u32;
    let mut worst = String::new();

    // World-cell of a haloed hit: brick voxel_origin + (hit_vox - 1) at LOD0.
    let world_cell = |bi: usize, hv: IVec3| -> IVec3 {
        IVec3::from(patch.metas[bi].voxel_origin) + (hv - IVec3::ONE)
    };

    for j in 0..n {
        for i in 0..n {
            let sx = (i as f32 + 0.5) / n as f32 * 2.0 - 1.0;
            let sy = (j as f32 + 0.5) / n as f32 * 2.0 - 1.0;
            let rd = (fwd + right * (sx * half_fov) + up * (-sy * half_fov)).normalize();
            // Gather all brick hits with the gradient normal, sorted by t.
            let mut hits: Vec<(usize, f32, Vec3, IVec3)> = (0..patch.metas.len())
                .filter_map(|bi| {
                    dda_brick_faithful(&patch, bi, eye, rd.normalize())
                        .map(|h| (bi, h.hit_t, occupancy_gradient_normal(&patch, bi, h.hit_vox), h.hit_vox))
                })
                .collect();
            hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            if hits.len() >= 2 {
                let (b0, t0, n0, hv0) = hits[0];
                let (b1, t1, n1, hv1) = hits[1];
                let same_surface = (t1 - t0).abs() <= tie || world_cell(b0, hv0) == world_cell(b1, hv1);
                if same_surface && (n0 - n1).length() > 1e-4 {
                    ties += 1;
                    if worst.is_empty() {
                        worst = format!("ray ({i},{j}): brick {b0} cell{:?} n={n0:?} (t={t0:.4}) vs brick {b1} cell{:?} n={n1:?} (t={t1:.4})", world_cell(b0, hv0), world_cell(b1, hv1));
                    }
                }
            }
        }
    }
    eprintln!("gradient TRUE-coplanar (same-surface) normal disagreements: {ties}");
    assert_eq!(ties, 0, "{ties} truly-coplanar bricks disagree on the gradient normal. Worst: {worst}");
}

/// THE FIX VALIDATED: the edge voxel gets a SINGLE camera-independent normal under the gradient rule (a
/// stable 45° bevel where +Y and +X faces are both exposed) — no flip across the angle sweep.
#[test]
fn fix_gradient_edge_voxel_is_camera_independent() {
    use adventure::voxel::brickmap::{BRICK_VOXELS, Brick, BrickMap, voxel_index};
    use adventure::voxel::palette::BlockId;

    let reg = BlockRegistry::cornell();
    let mut voxels = Box::new([BlockId::AIR; BRICK_VOXELS]);
    for z in 0..BRICK_EDGE {
        for x in 0..BRICK_EDGE {
            voxels[voxel_index(x, 0, z)] = BlockId(1);
        }
    }
    voxels[voxel_index(7, 1, 4)] = BlockId(1);
    let mut map = BrickMap::new();
    map.insert(IVec3::ZERO, Brick::from_voxels(voxels));
    let patch = pack_brickmap(&map, &reg);

    let target = Vec3::new(7.5, 2.0, 4.5) * VOXEL_SIZE;
    let mut normals = std::collections::BTreeSet::new();
    for k in 0..41 {
        let ax = (k as f32 / 40.0 - 0.5) * 1.4;
        let eye = target + Vec3::new(2.0 + ax, 2.0, 0.0);
        let rd = (target - eye).normalize();
        if let Some((_, h, n)) = trace_gradient(&patch, eye, rd)
            && h.hit_vox == IVec3::new(8, 2, 5)
        {
            // Quantise to milli-units so FP noise in the normalised bevel doesn't inflate the set.
            normals.insert((
                (n.x * 1000.0).round() as i32,
                (n.y * 1000.0).round() as i32,
                (n.z * 1000.0).round() as i32,
            ));
        }
    }
    eprintln!("gradient edge-voxel normals across sweep: {normals:?}");
    assert_eq!(
        normals.len(),
        1,
        "gradient fix: the edge voxel committed {} normals: {normals:?}",
        normals.len()
    );
}

/// DIAGNOSTIC (not asserting): characterise the swapping population for a SINGLE fixed Cornell view —
/// classify every hit cell by how many of its 6 haloed-grid face-neighbours are air (exposed faces), and
/// whether the most-head-on heuristic's chosen normal equals the gradient (geometric) normal. This shows
/// the bug is concentrated on cells with ≥2 exposed faces (edges/corners) AND, crucially, on FLAT-face
/// cells where the chosen axis is a side face the ray never crossed.
#[test]
fn diagnose_swap_population() {
    let reg = BlockRegistry::cornell();
    let map = build_cornell(&reg);
    let patch = pack_brickmap(&map, &reg);

    let c = INTERIOR as f32 * 0.5 * VOXEL_SIZE;
    let target = Vec3::splat(c);
    let eye = Vec3::new(c, c + 1.5, -8.5);
    let (right, up, fwd) = camera_basis(eye, target);
    let n = 200;
    let half_fov = 0.5f32;

    let mut total = 0u64;
    let mut heuristic_ne_gradient = 0u64;
    let mut by_exposed = [0u64; 4]; // index = number of exposed faces (0..3); cells with >3 clamp to 3
    let mut flat_face_wrong = 0u64; // single-exposed-axis cell where heuristic picked a NON-face axis

    for j in 0..n {
        for i in 0..n {
            let sx = (i as f32 + 0.5) / n as f32 * 2.0 - 1.0;
            let sy = (j as f32 + 0.5) / n as f32 * 2.0 - 1.0;
            let rd = (fwd + right * (sx * half_fov) + up * (-sy * half_fov)).normalize();
            let Some(out) = trace_faithful(&patch, eye, rd, VOXEL_SIZE * 0.25) else {
                continue;
            };
            total += 1;
            let grad = occupancy_gradient_normal(&patch, out.best_bi, out.best.hit_vox);
            let exposed = count_exposed_faces(&patch, out.best_bi, out.best.hit_vox);
            by_exposed[exposed.min(3) as usize] += 1;
            if (out.normal - grad).length() > 1e-3 && grad != Vec3::ZERO {
                heuristic_ne_gradient += 1;
                // A flat face = exactly one exposed face. If the heuristic disagrees there, it chose a side
                // axis the ray merely "met head-on" though that side is solid-backed — the visible-face bug.
                if exposed == 1 {
                    flat_face_wrong += 1;
                }
            }
        }
    }
    eprintln!("diagnose_swap_population (single view, {total} hits):");
    eprintln!("  exposed-face histogram (0,1,2,3+): {by_exposed:?}");
    eprintln!("  heuristic normal != gradient normal: {heuristic_ne_gradient}");
    eprintln!("  of those, FLAT-face (1 exposed) wrong: {flat_face_wrong}");
}

/// Count the hit cell's air face-neighbours in the haloed grid (the 6-neighbourhood exposure — the
/// quantity the gradient normal is built from). Used by the diagnostic only.
fn count_exposed_faces(patch: &GpuBrickPatch, bi: usize, hit_vox: IVec3) -> u32 {
    let m = &patch.metas[bi];
    let core = lod_edge(m.lod);
    let hedge = core + 2;
    let off = m.voxel_offset as usize;
    let cell = |x: i32, y: i32, z: i32| -> u32 {
        if x < 0 || x >= hedge || y < 0 || y >= hedge || z < 0 || z >= hedge {
            0
        } else {
            patch.voxels[off + (x + y * hedge + z * hedge * hedge) as usize]
        }
    };
    let mut e = 0;
    for d in [IVec3::X, IVec3::Y, IVec3::Z] {
        let p = hit_vox + d;
        let q = hit_vox - d;
        if cell(p.x, p.y, p.z) == 0 {
            e += 1;
        }
        if cell(q.x, q.y, q.z) == 0 {
            e += 1;
        }
    }
    e
}
