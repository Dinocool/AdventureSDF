//! Offline `.vox` preprocessor — voxelize a fixed classic mesh scene (Sponza) ONCE into a MagicaVoxel
//! `.vox` the runtime loader (`adventure::voxel::vox::load_vox`) reads as a static GI-measurement scene.
//!
//! This is a STANDALONE HEADLESS CPU tool: no Bevy `App`, no window, no GPU. It is a dev-only example, so
//! its mesh + texture decoders (`gltf`, `image`) are DEV-dependencies that never enter the shipped game —
//! the runtime reads only the baked `.vox` via `dot_vox`.
//!
//! PIPELINE
//! 1. Load `assets/models/src/Sponza.gltf` with `gltf` (positions + indices + UV0 + the base-colour texture
//!    per primitive, textures decoded by `gltf`'s `image` feature). If that asset is absent, fall back to a
//!    small procedural coloured box room so the pipeline + downstream test still build + run (and print a
//!    clear "drop in real Sponza" notice).
//! 2. SURFACE-voxelize into a dense grid at `VOXEL_SIZE` (0.2 m) over the mesh AABB: each triangle is
//!    conservatively rasterized (triangle–box overlap, the Akenine-Möller SAT) into every voxel it touches,
//!    marking it SOLID. Each solid voxel's albedo is the base-colour texture sampled at the
//!    barycentric-interpolated UV of the triangle point nearest the voxel centre (or the material
//!    `base_color_factor` when untextured).
//! 3. QUANTIZE the sampled albedos to a ≤255-colour palette (median-cut). Palette index 0 is reserved so the
//!    written `.vox` voxel indices are 1-based (MagicaVoxel convention; `dot_vox` stores them 0-based).
//! 4. WRITE `assets/models/sponza.vox` with `dot_vox`. A MagicaVoxel model is ≤256 per axis, so if the grid
//!    exceeds 256 on any axis it is SPLIT into a grid of ≤256³ sub-models, each placed by a scene-graph
//!    Transform (the model CENTER convention), reassembling into one contiguous scene at load.
//!
//! RUN: `cargo run --example voxelize_scene` (optionally `-- <out.vox> <voxel_metres>`).
//!
//! NOTE on colour space: glTF base-colour textures/factors are sRGB; MagicaVoxel `.vox` palettes are also
//! sRGB `u8`. So this tool keeps everything in sRGB `u8` end-to-end (no linearization here) — the RUNTIME
//! loader converts the `.vox` sRGB palette to linear when it builds the `BlockRegistry`. One conversion, in
//! one place.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use dot_vox::{Color, DotVoxData, Dict, Frame, Model, SceneNode, ShapeModel, Size, Voxel};
use rayon::prelude::*;

/// World edge of one voxel cell, in metres. MUST match `adventure::voxel::brickmap::VOXEL_SIZE` so the
/// baked grid lines up with the runtime brick grid (0.2 m). Duplicated as a literal because the example is a
/// separate binary that doesn't link the lib's render stack; kept in sync by this comment + the round-trip
/// test, which loads the produced `.vox` through the real `VOXEL_SIZE` path.
const DEFAULT_VOXEL_SIZE: f32 = 0.2;

/// MagicaVoxel model size cap per axis (a `.vox` model is ≤256³). Grids larger than this are split into a
/// scene grid of sub-models.
const VOX_MODEL_MAX: i32 = 256;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let out_path = args.next().map(PathBuf::from).unwrap_or_else(|| PathBuf::from("assets/models/sponza.vox"));
    let voxel_size: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_VOXEL_SIZE);

    // 1. Load the mesh (Sponza glTF, or a procedural fallback room).
    let gltf_path = Path::new("assets/models/src/Sponza.gltf");
    let mesh = if gltf_path.exists() {
        println!("loading Sponza glTF: {}", gltf_path.display());
        load_gltf(gltf_path)?
    } else {
        println!(
            "NOTE: {} not found — using the PROCEDURAL FALLBACK box room. Drop the real Khronos \
             glTF-Sample-Assets Sponza into assets/models/src/ (Sponza.gltf + Sponza.bin + textures) and \
             re-run to bake the real scene.",
            gltf_path.display()
        );
        fallback_room()
    };
    println!("mesh: {} triangles, {} textures", mesh.triangles.len(), mesh.textures.len());

    // 2. Surface-voxelize (rayon-parallel rasterization; the dominant cost at fine voxel sizes — timed so a
    // bake self-reports where the wall-clock goes).
    let t_vox = std::time::Instant::now();
    let mut grid = voxelize(&mesh, voxel_size);
    println!(
        "grid: {}×{}×{} voxels, {} surface (voxelize {:.2}s)",
        grid.dims[0], grid.dims[1], grid.dims[2], grid.solid_count(), t_vox.elapsed().as_secs_f32()
    );
    // 2b. Fill ENCLOSED interiors solid (always-on): a destructible voxel object must be solid inside so a cut
    // reveals interior, not empty space. Open / exterior-reachable space stays air (see `solid_fill`).
    let t_fill = std::time::Instant::now();
    solid_fill(&mut grid);
    println!(
        "  + solid fill: {} total solid (fill {:.2}s)",
        grid.solid_count(), t_fill.elapsed().as_secs_f32()
    );

    // 3. Quantize the sampled albedos to a ≤255 palette.
    let (palette, indices) = quantize(&grid);
    println!("palette: {} colours", palette.len());

    // 4. Write the `.vox` (split into ≤256³ models if needed).
    let data = build_dot_vox(&grid, &palette, &indices);
    let n_models = data.models.len();
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(&out_path)?;
    data.write_vox(&mut file)?;
    println!(
        "wrote {} ({} model{}, dims {}×{}×{}, {} solid voxels, {} palette colours)",
        out_path.display(),
        n_models,
        if n_models == 1 { "" } else { "s" },
        grid.dims[0],
        grid.dims[1],
        grid.dims[2],
        grid.solid_count(),
        palette.len()
    );
    Ok(())
}

// ============================================================================================
// Mesh representation (decoupled from glTF so the fallback room is the same shape)
// ============================================================================================

/// A texture decoded to interleaved 8-bit RGBA (sRGB). Sampled with wrapping + nearest filtering — adequate
/// for per-voxel albedo (a voxel is far coarser than a texel).
struct Texture {
    width: u32,
    height: u32,
    rgba: Vec<u8>, // width*height*4
}

impl Texture {
    /// Nearest-sample sRGB RGBA at UV (wrapping). Returns `[r,g,b,a]` sRGB `u8`.
    fn sample(&self, u: f32, v: f32) -> [u8; 4] {
        if self.width == 0 || self.height == 0 {
            return [255, 255, 255, 255];
        }
        let wrap = |t: f32, n: u32| -> u32 {
            let f = t - t.floor(); // [0,1)
            ((f * n as f32) as u32).min(n - 1)
        };
        let x = wrap(u, self.width);
        let y = wrap(v, self.height);
        let i = ((y * self.width + x) * 4) as usize;
        [self.rgba[i], self.rgba[i + 1], self.rgba[i + 2], self.rgba[i + 3]]
    }
}

/// One triangle: world-space positions, UV0 per vertex, and how to colour it — either a texture index +
/// UVs, or a flat sRGB base colour (the material `base_color_factor`, or the fallback's per-face colour).
struct Triangle {
    p: [[f32; 3]; 3],
    uv: [[f32; 2]; 3],
    /// `Some(texture_index)` to sample `textures[i]` at the interpolated UV; `None` to use `base`.
    texture: Option<usize>,
    /// Flat sRGB albedo used when `texture` is `None` (or as a tint multiplier — here we just use it raw).
    base: [u8; 4],
}

/// The decoded scene mesh: a flat triangle soup + the textures they reference.
struct Mesh {
    triangles: Vec<Triangle>,
    textures: Vec<Texture>,
}

// ============================================================================================
// glTF loading
// ============================================================================================

/// Load a glTF file into a [`Mesh`]: every primitive's positions + indices + UV0, with the material's
/// base-colour texture (decoded via `gltf`'s `image` feature) or its `base_color_factor`. Positions are
/// transformed to WORLD space by walking the scene-node hierarchy and accumulating each node's local
/// transform (CRITICAL: Sponza's single node carries a 0.008 scale, so mesh-local coords of ±1400 become a
/// ~24 m world scene — without this the AABB would be ~3000 units and the dense grid would be astronomically
/// large). glTF and this engine are both Y-up; the Z-up swap for `.vox` happens at write time.
fn load_gltf(path: &Path) -> anyhow::Result<Mesh> {
    let (doc, buffers, images) = gltf::import(path)?;

    // Decode every glTF image to RGBA8 once (indexed by image source index).
    let textures: Vec<Texture> = images.iter().map(decode_image).collect();

    let mut triangles = Vec::new();
    // Walk every scene's node hierarchy with the accumulated world matrix (column-major 4×4 from glTF).
    let scene = doc.default_scene().or_else(|| doc.scenes().next());
    if let Some(scene) = scene {
        for node in scene.nodes() {
            walk_node(&node, IDENTITY4, &buffers, &textures, &mut triangles);
        }
    } else {
        // No scene graph: emit meshes at identity (rare; keeps the loader total).
        for mesh in doc.meshes() {
            emit_mesh_primitives(&mesh, IDENTITY4, &buffers, &textures, &mut triangles);
        }
    }
    Ok(Mesh { triangles, textures })
}

/// Column-major 4×4 identity (glTF transform convention).
const IDENTITY4: [[f32; 4]; 4] =
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

/// Multiply two column-major 4×4 matrices: `a · b` (apply `b` then `a`).
fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut r = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            r[col][row] = a[0][row] * b[col][0] + a[1][row] * b[col][1] + a[2][row] * b[col][2] + a[3][row] * b[col][3];
        }
    }
    r
}

/// Transform a position by a column-major 4×4 (homogeneous w=1, perspective-divide ignored — affine only).
fn transform_point(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

/// Recursively walk a node + its children, accumulating the world transform and emitting each node's mesh
/// primitives (positions baked to world space). Bounded by the node count (a glTF hierarchy is a tree).
fn walk_node(
    node: &gltf::Node,
    parent: [[f32; 4]; 4],
    buffers: &[gltf::buffer::Data],
    textures: &[Texture],
    out: &mut Vec<Triangle>,
) {
    let world = mat4_mul(parent, node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        emit_mesh_primitives(&mesh, world, buffers, textures, out);
    }
    for child in node.children() {
        walk_node(&child, world, buffers, textures, out);
    }
}

/// Emit one mesh's primitives as world-space triangles (positions transformed by `world`), reading UV0 +
/// indices and resolving the base-colour texture / factor per material.
fn emit_mesh_primitives(
    mesh: &gltf::Mesh,
    world: [[f32; 4]; 4],
    buffers: &[gltf::buffer::Data],
    textures: &[Texture],
    out: &mut Vec<Triangle>,
) {
    for prim in mesh.primitives() {
        let reader = prim.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
        let positions: Vec<[f32; 3]> = match reader.read_positions() {
            Some(p) => p.map(|v| transform_point(&world, v)).collect(),
            None => continue,
        };
        let uvs: Vec<[f32; 2]> = reader
            .read_tex_coords(0)
            .map(|tc| tc.into_f32().collect())
            .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

        let mat = prim.material();
        let pbr = mat.pbr_metallic_roughness();
        let factor = pbr.base_color_factor();
        let base = [
            (factor[0].clamp(0.0, 1.0) * 255.0) as u8,
            (factor[1].clamp(0.0, 1.0) * 255.0) as u8,
            (factor[2].clamp(0.0, 1.0) * 255.0) as u8,
            (factor[3].clamp(0.0, 1.0) * 255.0) as u8,
        ];
        let texture = pbr
            .base_color_texture()
            .map(|info| info.texture().source().index())
            .filter(|&i| i < textures.len() && textures[i].width > 0);

        // Index iterator: explicit indices, or implied 0..n for a non-indexed primitive.
        let idx: Vec<u32> = match reader.read_indices() {
            Some(it) => it.into_u32().collect(),
            None => (0..positions.len() as u32).collect(),
        };
        for tri in idx.chunks_exact(3) {
            let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if a >= positions.len() || b >= positions.len() || c >= positions.len() {
                continue;
            }
            out.push(Triangle {
                p: [positions[a], positions[b], positions[c]],
                uv: [
                    *uvs.get(a).unwrap_or(&[0.0, 0.0]),
                    *uvs.get(b).unwrap_or(&[0.0, 0.0]),
                    *uvs.get(c).unwrap_or(&[0.0, 0.0]),
                ],
                texture,
                base,
            });
        }
    }
}

/// Decode one `gltf::image::Data` (already CPU-decoded by the `image` feature) into interleaved RGBA8. Only
/// the 8-bit formats Sponza ships are handled; anything else yields an empty texture (callers fall back to
/// the material `base_color_factor`).
fn decode_image(img: &gltf::image::Data) -> Texture {
    use gltf::image::Format;
    let (w, h) = (img.width, img.height);
    let n = (w as usize) * (h as usize);
    let mut rgba = Vec::with_capacity(n * 4);
    match img.format {
        Format::R8G8B8A8 => return Texture { width: w, height: h, rgba: img.pixels.clone() },
        Format::R8G8B8 => {
            for px in img.pixels.chunks_exact(3) {
                rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
        }
        Format::R8G8 => {
            for px in img.pixels.chunks_exact(2) {
                rgba.extend_from_slice(&[px[0], px[0], px[0], px[1]]);
            }
        }
        Format::R8 => {
            for &g in &img.pixels {
                rgba.extend_from_slice(&[g, g, g, 255]);
            }
        }
        // 16/32-bit formats are rare for base-colour; skip (empty → factor fallback).
        _ => return Texture { width: 0, height: 0, rgba: Vec::new() },
    }
    Texture { width: w, height: h, rgba }
}

/// A procedural fallback: a coloured box room (floor + 4 walls + ceiling), each face a distinct flat colour,
/// ~16 m × 8 m × 16 m. Used only when the real Sponza glTF is missing, so the pipeline + the round-trip test
/// still build + run end-to-end (and produce a non-trivial multi-colour `.vox`).
fn fallback_room() -> Mesh {
    // One axis-aligned quad: four corner positions (CCW) + a flat sRGB colour.
    type Quad = ([f32; 3], [f32; 3], [f32; 3], [f32; 3], [u8; 4]);
    // Y-up; room interior is [-8,8]×[0,8]×[-8,8].
    let (lo, hi, top) = (-8.0f32, 8.0f32, 8.0f32);
    let faces: [Quad; 6] = [
        // floor (y=0) — grey
        ([lo, 0.0, lo], [hi, 0.0, lo], [hi, 0.0, hi], [lo, 0.0, hi], [160, 160, 160, 255]),
        // ceiling (y=top) — white
        ([lo, top, lo], [lo, top, hi], [hi, top, hi], [hi, top, lo], [240, 240, 240, 255]),
        // -X wall — red
        ([lo, 0.0, lo], [lo, 0.0, hi], [lo, top, hi], [lo, top, lo], [200, 40, 40, 255]),
        // +X wall — green
        ([hi, 0.0, lo], [hi, top, lo], [hi, top, hi], [hi, 0.0, hi], [40, 180, 60, 255]),
        // -Z wall (back) — blue
        ([lo, 0.0, lo], [lo, top, lo], [hi, top, lo], [hi, 0.0, lo], [50, 80, 210, 255]),
        // +Z wall (front) — yellow
        ([lo, 0.0, hi], [hi, 0.0, hi], [hi, top, hi], [lo, top, hi], [220, 200, 40, 255]),
    ];
    let mut triangles = Vec::new();
    for (a, b, c, d, col) in faces {
        // Two triangles per quad (a,b,c) + (a,c,d). UVs unused (no texture).
        for tri in [[a, b, c], [a, c, d]] {
            triangles.push(Triangle {
                p: tri,
                uv: [[0.0, 0.0]; 3],
                texture: None,
                base: col,
            });
        }
    }
    Mesh { triangles, textures: Vec::new() }
}

// ============================================================================================
// Voxelization (surface / shell)
// ============================================================================================

/// A dense voxel grid over the mesh AABB at the voxelization's voxel size. `solid[i]` true ⇒ that voxel is
/// on the surface; `albedo[i]` is its sampled sRGB colour. Indexed `x + y*dx + z*dx*dy` (X fastest). Only the
/// dims + the solid/albedo arrays are needed downstream (quantize + `.vox` assembly); the world origin /
/// voxel size are consumed entirely within [`voxelize`], so they aren't retained.
struct Grid {
    dims: [i32; 3],
    solid: Vec<bool>,
    albedo: Vec<[u8; 4]>,
}

impl Grid {
    fn idx(&self, x: i32, y: i32, z: i32) -> usize {
        (x + y * self.dims[0] + z * self.dims[0] * self.dims[1]) as usize
    }
    fn solid_count(&self) -> usize {
        self.solid.iter().filter(|&&s| s).count()
    }
}

/// Surface-voxelize the mesh: for every triangle, conservatively rasterize into each voxel it overlaps
/// (triangle–AABB SAT), marking it solid and recording the albedo of the triangle point nearest the voxel
/// centre. The result is a SHELL (the visible surface), which is what we render + measure GI on.
fn voxelize(mesh: &Mesh, voxel_size: f32) -> Grid {
    // AABB of all triangle vertices.
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for t in &mesh.triangles {
        for v in &t.p {
            for a in 0..3 {
                lo[a] = lo[a].min(v[a]);
                hi[a] = hi[a].max(v[a]);
            }
        }
    }
    if !lo[0].is_finite() {
        // No geometry — return a 1³ empty grid.
        return Grid { dims: [1, 1, 1], solid: vec![false], albedo: vec![[0; 4]] };
    }
    // Pad one voxel so surface triangles on the boundary still have a cell.
    let origin = [lo[0] - voxel_size, lo[1] - voxel_size, lo[2] - voxel_size];
    let dims = [
        (((hi[0] - lo[0]) / voxel_size).ceil() as i32 + 3).max(1),
        (((hi[1] - lo[1]) / voxel_size).ceil() as i32 + 3).max(1),
        (((hi[2] - lo[2]) / voxel_size).ceil() as i32 + 3).max(1),
    ];
    // Guard: a dense grid is allocated for the whole AABB, so an absurd extent (e.g. forgetting the glTF
    // node transform, which once made Sponza ~3000 units) would try a terabyte allocation. Abort with a
    // clear message naming the AABB + dims rather than OOM-crashing.
    let total = (dims[0] as i64) * (dims[1] as i64) * (dims[2] as i64);
    const MAX_VOXELS: i64 = 1_500_000_000; // ~6 GB of (solid+albedo); generous for any real classic scene
    assert!(
        total <= MAX_VOXELS,
        "voxel grid {dims:?} = {total} cells exceeds {MAX_VOXELS} — AABB world span is {:?}..{:?} ({:.1} m \
         on the longest axis at {voxel_size} m/voxel). Are glTF node transforms applied? (Sponza needs the \
         0.008 node scale.) Raise --voxel_metres or check the mesh units.",
        lo,
        hi,
        (hi[0] - lo[0]).max(hi[1] - lo[1]).max(hi[2] - lo[2])
    );
    let total = total as usize;
    let mut grid = Grid { dims, solid: vec![false; total], albedo: vec![[0; 4]; total] };

    let half = voxel_size * 0.5;
    let dims = grid.dims; // Copy [i32;3] — captured by the parallel closures so they don't borrow `grid`.
    // Rasterize triangles into voxels IN PARALLEL: the 13-axis SAT overlap test + the per-voxel albedo sample
    // is the hot path and is independent per triangle, so fan it across all cores with rayon. Each triangle
    // returns its solid (cell-index, albedo) list; the lists are merged below in triangle order so the
    // original "first triangle to claim a cell keeps its albedo" rule still holds deterministically — parallel
    // writes into the shared grid couldn't preserve that ordering (and would race), so we don't try.
    let per_tri: Vec<Vec<(usize, [u8; 4])>> = mesh
        .triangles
        .par_iter()
        .map(|t| {
            // Triangle voxel-AABB (clamped to the grid), expanded by ONE cell each side BEFORE clamping. A
            // triangle lying exactly on a voxel boundary floors to the cell on the +side of its plane, and
            // `tri_box_overlap`'s plane test then rejects that cell (the plane only TOUCHES its min face) —
            // silently dropping every grid-aligned face (floors/walls/ceilings → holes, fatal for a GI
            // reference). The ±1 pad keeps the candidate range conservative so the truly-overlapping cell is
            // always tested; the SAT still rejects genuine non-overlaps, so no spurious voxels are added.
            let mut tlo = [i32::MAX; 3];
            let mut thi = [i32::MIN; 3];
            for v in &t.p {
                for a in 0..3 {
                    let c = ((v[a] - origin[a]) / voxel_size).floor() as i32;
                    tlo[a] = tlo[a].min(c);
                    thi[a] = thi[a].max(c);
                }
            }
            for a in 0..3 {
                tlo[a] = (tlo[a] - 1).clamp(0, dims[a] - 1);
                thi[a] = (thi[a] + 1).clamp(0, dims[a] - 1);
            }
            let mut cells = Vec::new();
            for z in tlo[2]..=thi[2] {
                for y in tlo[1]..=thi[1] {
                    for x in tlo[0]..=thi[0] {
                        let center = [
                            origin[0] + (x as f32 + 0.5) * voxel_size,
                            origin[1] + (y as f32 + 0.5) * voxel_size,
                            origin[2] + (z as f32 + 0.5) * voxel_size,
                        ];
                        if tri_box_overlap(center, half, &t.p) {
                            let i = (x + y * dims[0] + z * dims[0] * dims[1]) as usize;
                            cells.push((i, sample_albedo(mesh, t, center)));
                        }
                    }
                }
            }
            cells
        })
        .collect();
    // First-writer-wins merge in triangle order (matches the original single-threaded semantics).
    for cells in &per_tri {
        for &(i, albedo) in cells {
            if !grid.solid[i] {
                grid.solid[i] = true;
                grid.albedo[i] = albedo;
            }
        }
    }
    grid
}

/// Fill ENCLOSED interiors solid (always-on): a destructible voxel object must be solid inside so a cut reveals
/// interior, not empty space. EXTERIOR flood-fill — everything 6-connected to OUTSIDE the grid stays air; every
/// air voxel NOT reachable from outside is enclosed interior → made solid. So open / exterior-reachable space
/// (Sponza's nave, a doorway) stays air; only enclosed interiors (inside walls/columns) fill — "solid where it
/// should be," not the whole bounding box. Robust for non-watertight meshes: a hole connecting an interior to the
/// outside leaks that region to air (correct). Interior voxels take the NEAREST surface voxel's albedo (a
/// multi-source BFS from the surface) so a freshly-cut interior looks like its material; a strata/material system
/// can reassign them later. Ported from `D:\Projects\asset gen` `_solid_fill` (exterior label → interior = unreached).
fn solid_fill(grid: &mut Grid) {
    let [dx, dy, dz] = grid.dims;
    let total = (dx * dy * dz) as usize;
    if total == 0 {
        return;
    }
    const N6: [(i32, i32, i32); 6] =
        [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)];

    // 1. EXTERIOR: 6-connected flood through AIR, seeded from every AIR cell on the grid boundary (outside the
    //    grid is air, so a boundary air cell is exterior). Reached air = exterior; unreached air = enclosed.
    let mut exterior = vec![false; total];
    let mut q: std::collections::VecDeque<(i32, i32, i32)> = std::collections::VecDeque::new();
    for z in 0..dz {
        for y in 0..dy {
            for x in 0..dx {
                if x == 0 || y == 0 || z == 0 || x == dx - 1 || y == dy - 1 || z == dz - 1 {
                    let i = grid.idx(x, y, z);
                    if !grid.solid[i] && !exterior[i] {
                        exterior[i] = true;
                        q.push_back((x, y, z));
                    }
                }
            }
        }
    }
    while let Some((x, y, z)) = q.pop_front() {
        for (ox, oy, oz) in N6 {
            let (nx, ny, nz) = (x + ox, y + oy, z + oz);
            if nx < 0 || ny < 0 || nz < 0 || nx >= dx || ny >= dy || nz >= dz {
                continue;
            }
            let ni = grid.idx(nx, ny, nz);
            if !grid.solid[ni] && !exterior[ni] {
                exterior[ni] = true;
                q.push_back((nx, ny, nz));
            }
        }
    }

    // 2. Fill the enclosed interior (air && !exterior) solid, colouring each cell with the NEAREST surface
    //    voxel's albedo via a multi-source 6-connected BFS seeded from the surface (pre-fill solid) cells.
    //    `filled` is the visited set — surface seeds start visited so they keep their own colour.
    let mut filled = grid.solid.clone();
    q.clear();
    for z in 0..dz {
        for y in 0..dy {
            for x in 0..dx {
                let i = grid.idx(x, y, z);
                if grid.solid[i] {
                    q.push_back((x, y, z));
                }
            }
        }
    }
    while let Some((x, y, z)) = q.pop_front() {
        let src = grid.albedo[grid.idx(x, y, z)];
        for (ox, oy, oz) in N6 {
            let (nx, ny, nz) = (x + ox, y + oy, z + oz);
            if nx < 0 || ny < 0 || nz < 0 || nx >= dx || ny >= dy || nz >= dz {
                continue;
            }
            let ni = grid.idx(nx, ny, nz);
            if !filled[ni] && !exterior[ni] {
                filled[ni] = true;
                grid.solid[ni] = true;
                grid.albedo[ni] = src;
                q.push_back((nx, ny, nz));
            }
        }
    }
}

/// The sRGB albedo for a voxel: the triangle's texture sampled at the barycentric UV of the triangle point
/// nearest the voxel centre (so the colour is spatially right even when the voxel centre is off the
/// triangle), or the flat `base` colour when the triangle is untextured.
fn sample_albedo(mesh: &Mesh, t: &Triangle, center: [f32; 3]) -> [u8; 4] {
    let Some(tex) = t.texture.and_then(|i| mesh.textures.get(i)) else {
        return t.base;
    };
    let bary = closest_point_barycentric(center, &t.p);
    let u = bary[0] * t.uv[0][0] + bary[1] * t.uv[1][0] + bary[2] * t.uv[2][0];
    let v = bary[0] * t.uv[0][1] + bary[1] * t.uv[1][1] + bary[2] * t.uv[2][1];
    tex.sample(u, v)
}

/// Barycentric coordinates of the point on triangle `p` nearest `q` (Ericson, *Real-Time Collision
/// Detection*, §5.1.5). Always returns weights in `[0,1]` summing to 1, even when the projection of `q`
/// falls outside the triangle (it clamps to the nearest edge/vertex), so the sampled UV stays on the face.
fn closest_point_barycentric(q: [f32; 3], p: &[[f32; 3]; 3]) -> [f32; 3] {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let (a, b, c) = (p[0], p[1], p[2]);
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(q, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return [1.0, 0.0, 0.0];
    }
    let bp = sub(q, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return [0.0, 1.0, 0.0];
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return [1.0 - v, v, 0.0];
    }
    let cp = sub(q, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return [0.0, 0.0, 1.0];
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return [1.0 - w, 0.0, w];
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return [0.0, 1.0 - w, w];
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    [1.0 - v - w, v, w]
}

/// Triangle–AABB overlap test (Akenine-Möller, *Fast 3D Triangle-Box Overlap Testing*), the conservative
/// rasterization primitive: true iff the triangle `tri` intersects the axis-aligned box centred at `center`
/// with half-extent `half` on every axis. The 13-axis separating-axis test: 9 edge×box-axis cross products,
/// 3 box face normals, then the triangle plane. This is the canonical formulation — each edge's 3 axis tests
/// project the SPECIFIC two vertices not shared with that edge (so the projected interval is exact).
fn tri_box_overlap(center: [f32; 3], half: f32, tri: &[[f32; 3]; 3]) -> bool {
    // Move triangle into the box's local space (box centred at origin).
    let v0 = [tri[0][0] - center[0], tri[0][1] - center[1], tri[0][2] - center[2]];
    let v1 = [tri[1][0] - center[0], tri[1][1] - center[1], tri[1][2] - center[2]];
    let v2 = [tri[2][0] - center[0], tri[2][1] - center[1], tri[2][2] - center[2]];
    // Triangle edges.
    let e0 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e1 = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]];
    let e2 = [v0[0] - v2[0], v0[1] - v2[1], v0[2] - v2[2]];

    // 9 edge cross-axis tests. Each macro projects two triangle vertices onto the test axis (e_i × unit_j)
    // and the box onto it (radius = half · (|e_a| + |e_b|)); if the intervals are disjoint, separated.
    macro_rules! axis_x {
        // axis = e × X = (0, -e.z, e.y): projects onto (y,z).
        ($e:expr, $pa:expr, $pb:expr) => {{
            let p0 = $e[2] * $pa[1] - $e[1] * $pa[2];
            let p1 = $e[2] * $pb[1] - $e[1] * $pb[2];
            let rad = ($e[2].abs() + $e[1].abs()) * half;
            let (mn, mx) = if p0 < p1 { (p0, p1) } else { (p1, p0) };
            if mn > rad || mx < -rad {
                return false;
            }
        }};
    }
    macro_rules! axis_y {
        // axis = e × Y = (e.z, 0, -e.x): projects onto (x,z).
        ($e:expr, $pa:expr, $pb:expr) => {{
            let p0 = -$e[2] * $pa[0] + $e[0] * $pa[2];
            let p1 = -$e[2] * $pb[0] + $e[0] * $pb[2];
            let rad = ($e[2].abs() + $e[0].abs()) * half;
            let (mn, mx) = if p0 < p1 { (p0, p1) } else { (p1, p0) };
            if mn > rad || mx < -rad {
                return false;
            }
        }};
    }
    macro_rules! axis_z {
        // axis = e × Z = (-e.y, e.x, 0): projects onto (x,y).
        ($e:expr, $pa:expr, $pb:expr) => {{
            let p0 = $e[1] * $pa[0] - $e[0] * $pa[1];
            let p1 = $e[1] * $pb[0] - $e[0] * $pb[1];
            let rad = ($e[1].abs() + $e[0].abs()) * half;
            let (mn, mx) = if p0 < p1 { (p0, p1) } else { (p1, p0) };
            if mn > rad || mx < -rad {
                return false;
            }
        }};
    }
    // e0: test against v0 & v2; e1: v0 & v2; e2: v0 & v1 (the canonical vertex pairings).
    axis_x!(e0, v0, v2);
    axis_y!(e0, v0, v2);
    axis_z!(e0, v1, v2);
    axis_x!(e1, v0, v2);
    axis_y!(e1, v0, v2);
    axis_z!(e1, v0, v1);
    axis_x!(e2, v0, v1);
    axis_y!(e2, v0, v1);
    axis_z!(e2, v1, v2);

    // 3 box face normals: the triangle's AABB must overlap the box on every axis.
    for a in 0..3 {
        let mn = v0[a].min(v1[a]).min(v2[a]);
        let mx = v0[a].max(v1[a]).max(v2[a]);
        if mn > half || mx < -half {
            return false;
        }
    }

    // Triangle plane vs box (the 13th axis).
    let normal = [
        e0[1] * e1[2] - e0[2] * e1[1],
        e0[2] * e1[0] - e0[0] * e1[2],
        e0[0] * e1[1] - e0[1] * e1[0],
    ];
    plane_box_overlap(normal, v0, [half, half, half])
}

/// Plane–box overlap: true iff the plane through `vert` with `normal` intersects the box `[-half,half]³`
/// (the final axis of the triangle-box SAT).
fn plane_box_overlap(normal: [f32; 3], vert: [f32; 3], half: [f32; 3]) -> bool {
    let mut vmin = [0.0f32; 3];
    let mut vmax = [0.0f32; 3];
    for a in 0..3 {
        if normal[a] > 0.0 {
            vmin[a] = -half[a] - vert[a];
            vmax[a] = half[a] - vert[a];
        } else {
            vmin[a] = half[a] - vert[a];
            vmax[a] = -half[a] - vert[a];
        }
    }
    let dot = |n: [f32; 3], x: [f32; 3]| n[0] * x[0] + n[1] * x[1] + n[2] * x[2];
    if dot(normal, vmin) > 0.0 {
        return false;
    }
    dot(normal, vmax) >= 0.0
}

// ============================================================================================
// Palette quantization (median-cut)
// ============================================================================================

/// Quantize the grid's solid-voxel albedos to a ≤255-colour palette (median-cut) and map each solid voxel to
/// its nearest palette index (1-based; 0 is reserved for empty/air per the `.vox` convention). Returns the
/// palette (sRGB RGBA) and a per-voxel index parallel to `grid.solid` (0 for air voxels).
fn quantize(grid: &Grid) -> (Vec<[u8; 4]>, Vec<u8>) {
    // Gather distinct solid albedos with counts (median-cut works on the distinct set, weighted by count).
    let mut counts: HashMap<[u8; 4], u32> = HashMap::new();
    for (i, &s) in grid.solid.iter().enumerate() {
        if s {
            *counts.entry(grid.albedo[i]).or_insert(0) += 1;
        }
    }
    let pixels: Vec<([u8; 4], u32)> = counts.into_iter().collect();
    let palette = median_cut(&pixels, 255);

    // Map every solid voxel to its nearest palette colour (1-based index).
    let mut indices = vec![0u8; grid.solid.len()];
    // Cache nearest-index per distinct albedo so we don't re-search per voxel.
    let mut nearest_cache: HashMap<[u8; 4], u8> = HashMap::new();
    for (i, &s) in grid.solid.iter().enumerate() {
        if !s {
            continue;
        }
        let c = grid.albedo[i];
        let idx = *nearest_cache.entry(c).or_insert_with(|| nearest_palette(&palette, c));
        indices[i] = idx + 1; // 1-based; 0 = air
    }
    (palette, indices)
}

/// Median-cut colour quantization: recursively split the colour set along its widest channel at the
/// (count-weighted) median until `max_colors` buckets exist, then average each bucket. Returns ≤`max_colors`
/// representative sRGB colours. Robust to fewer distinct colours than `max_colors` (returns them all).
fn median_cut(pixels: &[([u8; 4], u32)], max_colors: usize) -> Vec<[u8; 4]> {
    if pixels.is_empty() {
        return vec![[255, 255, 255, 255]];
    }
    let mut buckets: Vec<Vec<([u8; 4], u32)>> = vec![pixels.to_vec()];
    while buckets.len() < max_colors {
        // Find the bucket with the largest channel range to split.
        let mut best = None;
        let mut best_range = 0i32;
        let mut best_channel = 0usize;
        for (bi, b) in buckets.iter().enumerate() {
            if b.len() < 2 {
                continue;
            }
            for ch in 0..3 {
                let mut mn = 255i32;
                let mut mx = 0i32;
                for (c, _) in b {
                    mn = mn.min(c[ch] as i32);
                    mx = mx.max(c[ch] as i32);
                }
                if mx - mn > best_range {
                    best_range = mx - mn;
                    best = Some(bi);
                    best_channel = ch;
                }
            }
        }
        let Some(bi) = best else { break }; // every bucket is a single colour
        let mut bucket = buckets.swap_remove(bi);
        bucket.sort_by_key(|(c, _)| c[best_channel]);
        // Split at the weighted median.
        let total: u32 = bucket.iter().map(|(_, w)| *w).sum();
        let mut acc = 0u32;
        let mut split = bucket.len() / 2;
        for (i, (_, w)) in bucket.iter().enumerate() {
            acc += *w;
            if acc * 2 >= total {
                split = (i + 1).clamp(1, bucket.len() - 1);
                break;
            }
        }
        let right = bucket.split_off(split);
        buckets.push(bucket);
        buckets.push(right);
    }
    // Average each bucket (count-weighted) into one representative colour.
    buckets
        .iter()
        .filter(|b| !b.is_empty())
        .map(|b| {
            let mut sum = [0u64; 4];
            let mut w = 0u64;
            for (c, cnt) in b {
                let cnt = *cnt as u64;
                for a in 0..4 {
                    sum[a] += c[a] as u64 * cnt;
                }
                w += cnt;
            }
            let w = w.max(1);
            [(sum[0] / w) as u8, (sum[1] / w) as u8, (sum[2] / w) as u8, (sum[3] / w) as u8]
        })
        .collect()
}

/// Index of the palette colour nearest `c` by squared RGB distance (alpha ignored — surface voxels are
/// opaque). Linear scan over ≤255 entries; results are cached per distinct albedo by the caller.
fn nearest_palette(palette: &[[u8; 4]], c: [u8; 4]) -> u8 {
    let mut best = 0usize;
    let mut best_d = i64::MAX;
    for (i, p) in palette.iter().enumerate() {
        let dr = c[0] as i64 - p[0] as i64;
        let dg = c[1] as i64 - p[1] as i64;
        let db = c[2] as i64 - p[2] as i64;
        let d = dr * dr + dg * dg + db * db;
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best as u8
}

// ============================================================================================
// `.vox` assembly (split into ≤256³ models on a scene grid)
// ============================================================================================

/// Build the `DotVoxData`: split the grid into ≤256³ sub-models, place each by a scene Transform at its
/// block CENTER (the MagicaVoxel convention the runtime loader reverses), and attach the 256-entry palette.
/// Z-up: the grid's Y (up) becomes `.vox` Z, the grid's Z becomes `.vox` Y, matching the loader's
/// `.vox (x,y,z) → world (x,z,y)` swap so a round-trip is identity.
fn build_dot_vox(grid: &Grid, palette: &[[u8; 4]], indices: &[u8]) -> DotVoxData {
    // Build the 256-entry `.vox` palette: our quantized colours, padded to 256.
    let mut vox_palette: Vec<Color> =
        palette.iter().map(|c| Color { r: c[0], g: c[1], b: c[2], a: c[3] }).collect();
    vox_palette.resize(256, Color { r: 0, g: 0, b: 0, a: 255 });

    // Tile the grid into ≤VOX_MODEL_MAX³ blocks. `.vox` axes: vx = grid x, vy = grid z, vz = grid y.
    let (dx, dy, dz) = (grid.dims[0], grid.dims[1], grid.dims[2]);
    // Ceil-div for non-negative dims (signed `i32::div_ceil` is still unstable on this toolchain).
    let ceil_div = |n: i32, d: i32| (n + d - 1) / d;
    // Number of tiles along each `.vox` axis. `.vox` X ← grid X, `.vox` Y ← grid Z, `.vox` Z ← grid Y.
    let tiles_x = ceil_div(dx, VOX_MODEL_MAX).max(1);
    let tiles_vy = ceil_div(dz, VOX_MODEL_MAX).max(1); // .vox Y from grid Z
    let tiles_vz = ceil_div(dy, VOX_MODEL_MAX).max(1); // .vox Z from grid Y

    let mut models: Vec<Model> = Vec::new();
    // Each model's `.vox`-space min corner, for the scene Transform (center = corner + size/2).
    let mut model_corners: Vec<[i32; 3]> = Vec::new();

    for tz in 0..tiles_vz {
        for ty in 0..tiles_vy {
            for tx in 0..tiles_x {
                // `.vox`-space tile bounds.
                let vx0 = tx * VOX_MODEL_MAX;
                let vy0 = ty * VOX_MODEL_MAX;
                let vz0 = tz * VOX_MODEL_MAX;
                let sx = (dx - vx0).min(VOX_MODEL_MAX);
                let sy = (dz - vy0).min(VOX_MODEL_MAX); // .vox Y extent ← grid Z
                let sz = (dy - vz0).min(VOX_MODEL_MAX); // .vox Z extent ← grid Y

                let mut voxels = Vec::new();
                for lz in 0..sz {
                    // .vox z ← grid y
                    let gy = vz0 + lz;
                    for ly in 0..sy {
                        // .vox y ← grid z
                        let gz = vy0 + ly;
                        for lx in 0..sx {
                            let gx = vx0 + lx;
                            let i = grid.idx(gx, gy, gz);
                            if !grid.solid[i] {
                                continue;
                            }
                            let pal = indices[i];
                            if pal == 0 {
                                continue; // shouldn't happen for a solid voxel, but stay total
                            }
                            voxels.push(Voxel {
                                x: lx as u8,
                                y: ly as u8,
                                z: lz as u8,
                                i: pal - 1, // dot_vox stores 0-based (file is 1-based)
                            });
                        }
                    }
                }
                if voxels.is_empty() {
                    continue; // drop fully-empty tiles
                }
                model_corners.push([vx0, vy0, vz0]);
                models.push(Model {
                    size: Size { x: sx as u32, y: sy as u32, z: sz as u32 },
                    voxels,
                });
            }
        }
    }

    // If nothing was solid, emit a single empty 1³ model so the file is well-formed.
    if models.is_empty() {
        models.push(Model { size: Size { x: 1, y: 1, z: 1 }, voxels: Vec::new() });
        model_corners.push([0, 0, 0]);
    }

    let scenes = build_scene_graph(&models, &model_corners);

    DotVoxData {
        version: 150,
        index_map: Vec::new(),
        models,
        palette: vox_palette,
        materials: Vec::new(),
        scenes,
        layers: Vec::new(),
    }
}

/// Build the MagicaVoxel scene graph placing each model at its tile position. Layout (MagicaVoxel rule): a
/// root Transform → a Group whose children are one Transform→Shape per model; each model Transform's `_t`
/// translation is the model CENTER (corner + size/2). The runtime loader reverses this exactly. For a single
/// model the same structure is emitted (trivial translation 0), which the loader also handles.
fn build_scene_graph(models: &[Model], corners: &[[i32; 3]]) -> Vec<SceneNode> {
    // Node layout: [0]=root Transform→1, [1]=Group→[2,4,6,...], then per model: Transform(2k)→Shape(2k+1).
    let mut scenes: Vec<SceneNode> = Vec::new();
    // Root transform (node 0) → group (node 1).
    scenes.push(SceneNode::Transform {
        attributes: Dict::new(),
        frames: vec![Frame { attributes: Dict::new() }],
        child: 1,
        layer_id: u32::MAX,
    });
    // Group (node 1).
    let mut group_children = Vec::with_capacity(models.len());
    // Per-model nodes start at index 2.
    let mut node_id = 2u32;
    let mut per_model_nodes: Vec<SceneNode> = Vec::new();
    for (mi, model) in models.iter().enumerate() {
        let corner = corners.get(mi).copied().unwrap_or([0, 0, 0]);
        // `_t` is the model CENTER. `dot_vox`'s Frame.position() reads `_t` from the attributes dict.
        let center = [
            corner[0] + (model.size.x / 2) as i32,
            corner[1] + (model.size.y / 2) as i32,
            corner[2] + (model.size.z / 2) as i32,
        ];
        let mut attrs = Dict::new();
        attrs.insert("_t".to_string(), format!("{} {} {}", center[0], center[1], center[2]));
        let transform_id = node_id;
        let shape_id = node_id + 1;
        group_children.push(transform_id);
        per_model_nodes.push(SceneNode::Transform {
            attributes: Dict::new(),
            frames: vec![Frame { attributes: attrs }],
            child: shape_id,
            layer_id: u32::MAX,
        });
        per_model_nodes.push(SceneNode::Shape {
            attributes: Dict::new(),
            models: vec![ShapeModel { model_id: mi as u32, attributes: Dict::new() }],
        });
        node_id += 2;
    }
    scenes.push(SceneNode::Group { attributes: Dict::new(), children: group_children });
    scenes.extend(per_model_nodes);
    scenes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression for the conservative-rasterization blocker: a grid-aligned planar face must NOT be silently
    /// dropped. The fallback room is six distinctly-coloured axis-aligned faces; before the ±1 candidate-AABB
    /// pad the SAT rejected boundary-aligned faces and only 2 of 6 survived (no floor / no ceiling = a useless
    /// GI reference). Assert the floor + ceiling planes are solid and all six face colours appear.
    #[test]
    fn fallback_room_bakes_all_six_faces() {
        let grid = voxelize(&fallback_room(), 1.0);
        // Every one of the 6 distinctly-coloured faces must contribute voxels. Before the conservative ±1
        // candidate-AABB pad, grid-aligned faces were dropped and only 2 of 6 colours survived.
        let distinct: std::collections::HashSet<[u8; 4]> =
            grid.solid.iter().enumerate().filter(|&(_, &s)| s).map(|(i, _)| grid.albedo[i]).collect();
        for col in [
            [160u8, 160, 160, 255], // floor (grey)
            [240, 240, 240, 255],   // ceiling (white)
            [200, 40, 40, 255],     // -X wall (red)
            [40, 180, 60, 255],     // +X wall (green)
            [50, 80, 210, 255],     // -Z wall (blue)
            [220, 200, 40, 255],    // +Z wall (yellow)
        ] {
            assert!(distinct.contains(&col), "face colour {col:?} dropped — non-conservative rasterization");
        }
    }

    /// `solid_fill` closes ENCLOSED interiors but leaves exterior-reachable air alone (the always-on solid
    /// model): a fully-closed shell gets its hollow filled solid; the same shell with an open face stays hollow
    /// (the cavity reaches outside). Also checks the filled interior takes a nearby surface colour, not transparent.
    #[test]
    fn solid_fill_closes_enclosed_but_keeps_open_air() {
        let dims = [5, 5, 5];
        let total = 125usize;
        // A 3×3×3 box SHELL at [1,3]³ (its 6 faces solid) around a single air cavity at (2,2,2).
        let build = |open_face: bool| -> Grid {
            let mut g = Grid { dims, solid: vec![false; total], albedo: vec![[0u8; 4]; total] };
            for z in 1..=3 {
                for y in 1..=3 {
                    for x in 1..=3 {
                        if x == 1 || x == 3 || y == 1 || y == 3 || z == 1 || z == 3 {
                            let i = g.idx(x, y, z);
                            g.solid[i] = true;
                            g.albedo[i] = [200, 100, 50, 255]; // a surface colour
                        }
                    }
                }
            }
            if open_face {
                let i = g.idx(2, 1, 2); // poke a hole in the -Y face → the cavity reaches outside
                g.solid[i] = false;
            }
            g
        };

        // CLOSED: the enclosed cavity at (2,2,2) fills solid and takes the nearest surface colour.
        let mut closed = build(false);
        let c = closed.idx(2, 2, 2);
        assert!(!closed.solid[c], "cavity starts air");
        solid_fill(&mut closed);
        assert!(closed.solid[c], "closed shell: the enclosed cavity is filled solid");
        assert_eq!(closed.albedo[c], [200, 100, 50, 255], "interior takes the nearest surface colour");

        // OPEN: the hole connects the cavity to the outside → it stays air (we never fill reachable space).
        let mut open = build(true);
        let o = open.idx(2, 2, 2);
        solid_fill(&mut open);
        assert!(!open.solid[o], "open shell: a cavity reachable from outside stays air");
    }
}
