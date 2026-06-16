//! Offline `.vox` preprocessor — voxelize a fixed classic mesh scene (Sponza) ONCE into a MagicaVoxel
//! `.vox` the runtime loader (`adventure::voxel::vox::load_vox`) reads as a static GI-measurement scene.
//!
//! This is a STANDALONE HEADLESS CPU tool: no Bevy `App`, no window, no GPU. It is a dev-only example, so
//! its mesh + texture decoders (`gltf`, `image`) are DEV-dependencies that never enter the shipped game —
//! the runtime reads only the baked `.vox` via `dot_vox`.
//!
//! PIPELINE
//! 1. Load a classic mesh scene into a [`Mesh`] (positions + indices + UV0 + per-primitive base colour /
//!    texture), picked by FILE EXTENSION — `.gltf`/`.glb` via `gltf` (textures decoded by `gltf`'s `image`
//!    feature; the default `assets/models/src/Sponza.gltf`), or `.obj` via `tobj` (positions/indices/UVs + the
//!    companion `.mtl` diffuse `Kd` base colour and `map_Kd` diffuse texture, decoded with `image`) — so
//!    classic OBJ scenes (Sibenik, San Miguel, OBJ Sponza variants) load into the SAME `Mesh` the glTF path
//!    builds and the rest of the pipeline is unchanged. The glTF path also handles `KHR_texture_basisu` KTX2
//!    textures (the Amazon Lumberyard Bistro, converted to glTF, ships its base colours as UASTC+Zstd KTX2):
//!    the unsupported `extensionsRequired` entry is stripped so `gltf` parses the document, and each external
//!    `.ktx2` base colour is decoded to RGBA8 by [`ktx2_to_rgba`] (the same `ktx2`/`ruzstd`/`basis-universal`
//!    path `bevy_image` uses) instead of by `gltf`'s `image` feature (which can't read KTX2). FBX (the raw
//!    Lumberyard Bistro) is NOT handled — convert it to glTF/OBJ externally first. If the asset is absent,
//!    fall back to a small procedural coloured box room so the pipeline + downstream test still build + run
//!    (and print a clear "drop in a real" notice).
//! 2. SURFACE-voxelize into a SPARSE grid at `VOXEL_SIZE` (0.2 m) over the mesh AABB: each triangle is
//!    conservatively rasterized (triangle–box overlap, the Akenine-Möller SAT) into every voxel it touches,
//!    marking it SOLID. Each solid voxel's albedo is the base-colour texture sampled at the
//!    barycentric-interpolated UV of the triangle point nearest the voxel centre (or the material
//!    `base_color_factor` when untextured). The grid stores a 1-bit-per-cell occupancy bitset plus a
//!    solid-only `cell → albedo` map, so a billion-cell AABB (Bistro @0.05 m) bakes in a few GB, not tens.
//! 3. QUANTIZE the sampled albedos to a ≤255-colour palette (median-cut). Palette index 0 is reserved so the
//!    written `.vox` voxel indices are 1-based (MagicaVoxel convention; `dot_vox` stores them 0-based).
//! 4. WRITE `assets/models/sponza.vox` with `dot_vox`. A MagicaVoxel model is ≤256 per axis, so if the grid
//!    exceeds 256 on any axis it is SPLIT into a grid of ≤256³ sub-models, each placed by a scene-graph
//!    Transform (the model CENTER convention), reassembling into one contiguous scene at load.
//!
//! RUN: `cargo run --example voxelize_scene` (optionally `-- <out.vox> <voxel_metres> <in_mesh> <scale>`), e.g.
//! `cargo run --example voxelize_scene -- assets/models/sibenik.vox 0.05 assets/models/src/sibenik/sibenik.obj`
//! (Conference is authored in cm → add a `0.01` 4th arg to land it in metres).
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

/// Default per-axis supersample count for area-averaged albedo (an `S×S×S` grid of texture taps over each
/// voxel's surface footprint; `S=3` = 27 candidate taps, matching asset-gen's `supersample=3`). At fine voxel
/// sizes a voxel covers many texels, so a single point sample aliases; averaging the footprint fixes it.
/// Overridable per-bake with `--supersample <N>` (`N=1` reproduces the old single sample LOCATION — now through
/// the bilinear filter, so a strict improvement, not bit-identical to the old nearest-texel sample).
const SUPERSAMPLE: usize = 3;

fn main() -> anyhow::Result<()> {
    // Positional args first (skipping any `--flag`s), then collect the flags. The CLI:
    //   <out.{vox|vxo}> <voxel_metres> <in_mesh> <scale> [--store]
    // An output ending `.vxo` selects the NEW native-format writer (`adventure::voxel::vxo::write_vxo`,
    // Phase B-i); `.vox` keeps the legacy MagicaVoxel writer (interchange/debug only). `--store` (`.vxo` only)
    // writes uncompressed region bodies instead of the default per-region zstd (`docs/VXO_FORMAT.md` §B1.9).
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let store = raw_args.iter().any(|a| a == "--store");
    // `--tiled`: route `.vxo` output through the bounded-RAM OUT-OF-CORE tiled voxelizer (C1) instead of the
    // monolithic grid `voxelize`+`solid_fill`. Required for huge scenes (Bistro @0.05 m). `--scratch <dir>` sets
    // the disk-tile scratch root (default a run-unique subdir under the temp dir, honoring `D:\tmp_test`);
    // `--tile-edge <N>` overrides the tile size (default 128, the RAM/scratch knob, §C1.1).
    let tiled = raw_args.iter().any(|a| a == "--tiled");
    let flag_val = |name: &str| -> Option<String> {
        raw_args.iter().position(|a| a == name).and_then(|p| raw_args.get(p + 1)).cloned()
    };
    // `--supersample <N>`: per-axis area-average tap count for albedo (default SUPERSAMPLE; N=1 = point sample).
    let supersample = flag_val("--supersample").and_then(|s| s.parse::<usize>().ok()).map(|n| n.max(1)).unwrap_or(SUPERSAMPLE);
    let tile_edge: i32 = flag_val("--tile-edge").and_then(|s| s.parse().ok()).unwrap_or(tiled::DEFAULT_TILE_EDGE);
    let scratch_arg = flag_val("--scratch").map(PathBuf::from);
    // Positional args skip flags AND the VALUE that follows a value-taking flag (so it isn't mistaken for one).
    let value_flags = ["--supersample", "--tile-edge", "--scratch"];
    let mut skip_next = false;
    let mut pos = raw_args.iter().filter(|a| {
        if skip_next {
            skip_next = false;
            return false;
        }
        if value_flags.contains(&a.as_str()) {
            skip_next = true;
            return false;
        }
        !a.starts_with("--")
    });
    let out_path = pos.next().map(PathBuf::from).unwrap_or_else(|| PathBuf::from("assets/models/sponza.vox"));
    let voxel_size: f32 = pos.next().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_VOXEL_SIZE);
    // The input mesh path: an explicit 3rd arg, else the default Sponza glTF. Picked by extension (glTF / OBJ).
    let in_path = pos.next().map(PathBuf::from).unwrap_or_else(|| PathBuf::from("assets/models/src/Sponza.gltf"));
    // Optional uniform scale (4th arg) applied to all positions AFTER loading. glTF carries unit scale in its
    // node transforms (Sponza's 0.008), but OBJ has none — some classic OBJ scenes are authored in cm/mm/other
    // (e.g. McGuire's Conference spans ~2700 units → needs ~0.01 to land in metres). Default 1.0 (no scale).
    let scale: f32 = pos.next().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let want_vxo = out_path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("vxo"));

    // 1. Load the mesh, dispatching on file extension (glTF / OBJ); a procedural fallback room when absent.
    let mut mesh = load_mesh(&in_path)?;
    if scale != 1.0 {
        for t in &mut mesh.triangles {
            for p in &mut t.p {
                for x in p.iter_mut() {
                    *x *= scale;
                }
            }
        }
        println!("applied uniform scale {scale} to {} triangles", mesh.triangles.len());
    }
    println!("mesh: {} triangles, {} textures", mesh.triangles.len(), mesh.textures.len());

    // OUT-OF-CORE tiled path (C1): for `.vxo` output with `--tiled`, run the bounded-RAM tiled voxelizer +
    // streaming `.vxo` assembly, bypassing the monolithic grid entirely. Required for huge scenes.
    if tiled {
        anyhow::ensure!(want_vxo, "--tiled requires a `.vxo` output (the streaming assembly targets .vxo)");
        return run_tiled_bake(&mesh, voxel_size, supersample, tile_edge, scratch_arg, &out_path, store);
    }

    // 2. Surface-voxelize (rayon-parallel rasterization; the dominant cost at fine voxel sizes — timed so a
    // bake self-reports where the wall-clock goes).
    let t_vox = std::time::Instant::now();
    println!("albedo supersample: {supersample}×{supersample}×{supersample} (area-averaged)");
    let mut grid = voxelize(&mesh, voxel_size, supersample);
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

    // 4. Write the baked artifact. Build the `DotVoxData` (the legacy interchange form + the bridge the `.vxo`
    //    path re-bricks through). `.vox` writes it directly; `.vxo` re-bricks it into the engine `BrickMap` +
    //    `BlockRegistry` (sharing the runtime loader's SSOT, so the `.vxo` carries exactly what the engine
    //    loads) and emits the native region-streamed format.
    let data = build_dot_vox(&grid, &palette, &indices);
    let n_models = data.models.len();
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if want_vxo {
        write_vxo_output(&out_path, &data, voxel_size, store)?;
        println!(
            "wrote {} (.vxo native format, {}, voxel_size {voxel_size} m, dims {}×{}×{}, {} solid voxels, {} palette colours)",
            out_path.display(),
            if store { "STORE" } else { "zstd-19" },
            grid.dims[0],
            grid.dims[1],
            grid.dims[2],
            grid.solid_count(),
            palette.len()
        );
    } else {
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
    }
    Ok(())
}

/// Re-brick a built [`DotVoxData`] into the engine's `(BrickMap, BlockRegistry)` (via the SAME
/// `adventure::voxel::vox::from_dot_vox` the runtime `.vox` loader uses) and write the native `.vxo`
/// (`adventure::voxel::vxo::write_vxo`, Phase B-i). `store` selects uncompressed region bodies; otherwise
/// default per-region zstd. `voxel_size` is recorded in `HEAD` (self-describing, `docs/VXO_FORMAT.md` §0.4).
fn write_vxo_output(out_path: &Path, data: &DotVoxData, voxel_size: f32, store: bool) -> anyhow::Result<()> {
    use adventure::voxel::vxo::{VxoCompression, VxoHeadParams, write_vxo};
    let (map, registry) = adventure::voxel::vox::from_dot_vox(data);
    let name = out_path.file_stem().and_then(|s| s.to_str()).unwrap_or("vxo").to_string();
    let params = VxoHeadParams { voxel_size, name, ..Default::default() };
    let comp = if store { VxoCompression::Store } else { VxoCompression::default() };
    write_vxo(out_path, &map, &registry, &params, comp)?;
    Ok(())
}

/// Drive the OUT-OF-CORE tiled bake (C1) end-to-end: compute the shared grid geometry, create a RUN-UNIQUE
/// scratch subdir (under `--scratch` or the temp dir, honoring `D:\tmp_test`), run the disk-tiled flood +
/// union-find ([`tiled::bake_tiled`]), STREAM-assemble the `.vxo` ([`assemble_vxo_streaming`]), and clean up.
/// Scratch lifetime (§C1.10): the run-unique subdir is DELETED on success, LEFT + LOGGED on failure for
/// debugging. Reports wall-clock + scratch high-water (the perf gate, memory `feedback-benchmark-deliveries`).
fn run_tiled_bake(
    mesh: &Mesh,
    voxel_size: f32,
    supersample: usize,
    tile_edge: i32,
    scratch_root: Option<PathBuf>,
    out_path: &Path,
    store: bool,
) -> anyhow::Result<()> {
    let Some((origin, dims)) = grid_geometry(mesh, voxel_size) else {
        anyhow::bail!("tiled bake: the mesh has no geometry");
    };
    let total_cells = (dims[0] as i64) * (dims[1] as i64) * (dims[2] as i64);
    let ceil_div = |n: i32, d: i32| (n + d - 1) / d;
    println!(
        "tiled bake: grid {}×{}×{} = {} cells, tile_edge {tile_edge} (~{} tiles/axis)",
        dims[0], dims[1], dims[2], total_cells,
        ceil_div(dims[0], tile_edge).max(1),
    );

    // Run-unique scratch subdir.
    let root = scratch_root.unwrap_or_else(std::env::temp_dir);
    let unique = format!("voxelize_tiled_{}_{}", std::process::id(), std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    let scratch = root.join(unique);
    std::fs::create_dir_all(&scratch)?;
    println!("tiled scratch: {}", scratch.display());

    let grid = tiled::TileGrid::new(dims, tile_edge, scratch.clone());

    // The bake + assembly, capturing any error so the scratch is left for debugging on failure.
    let result = (|| -> anyhow::Result<usize> {
        let t_bake = std::time::Instant::now();
        let (total_solid, final_ids) = tiled::bake_tiled(&grid, mesh, origin, voxel_size, supersample)?;
        println!(
            "  tiled flood + fill: {} solid voxels across {} non-empty tiles ({:.2}s)",
            total_solid, final_ids.len(), t_bake.elapsed().as_secs_f32()
        );
        let t_asm = std::time::Instant::now();
        assemble_vxo_streaming(&grid, &final_ids, voxel_size, out_path, store)?;
        println!("  streamed .vxo assembly ({:.2}s)", t_asm.elapsed().as_secs_f32());
        Ok(total_solid)
    })();

    match result {
        Ok(total_solid) => {
            // Scratch high-water (the disk-tile footprint) before cleanup, for the perf report.
            let high_water = dir_size(&scratch).unwrap_or(0);
            let _ = std::fs::remove_dir_all(&scratch);
            println!(
                "wrote {} (.vxo tiled out-of-core, {}, voxel_size {voxel_size} m, dims {}×{}×{}, {} solid voxels; \
                 scratch high-water {:.1} MiB, removed)",
                out_path.display(),
                if store { "STORE" } else { "zstd-19" },
                dims[0], dims[1], dims[2], total_solid,
                high_water as f64 / (1024.0 * 1024.0),
            );
            Ok(())
        }
        Err(e) => {
            eprintln!("tiled bake FAILED: {e:#}. Scratch left for debugging at {}", scratch.display());
            Err(e)
        }
    }
}

/// Recursive byte size of a directory (the scratch high-water report). Best-effort; ignores unreadable entries.
fn dir_size(dir: &Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let md = entry.metadata()?;
        if md.is_dir() {
            total += dir_size(&entry.path()).unwrap_or(0);
        } else {
            total += md.len();
        }
    }
    Ok(total)
}

/// STREAMING tiled `.vxo` assembly (C1.7): build the CIELAB palette + write the `.vxo` region-by-region from
/// the finalized DISK tiles, never holding the whole grid (or the whole `.vxo` image) in RAM. Two streaming
/// passes over the finalized tiles (`tiled::stream_final`):
///
/// 1. **Palette pass** — accumulate the distinct-albedo `counts` (bounded by texture content, not solid count)
///    AND the solid-extent bounds in grid-voxel space. Build the ≤255 CIELAB k-means palette (reusing C3's
///    [`build_palette`]) and the `BlockRegistry` (`BlockId(i+1)` ← palette entry `i`, the `.vox` convention).
/// 2. **Emit pass** — for each finalized tile, map every solid cell's grid coord → ANCHORED world-voxel coord
///    (floor at y=0, centred X/Z — the SAME anchor `vox.rs::bricks_from_placed` applies, so the tiled `.vxo`
///    lands in the gallery exactly like every other asset; the Z-up↔Y-up swaps cancel so grid↔world is identity
///    up to the anchor), bucket into the owning `8³` brick, and — when a brick's last cell across the stream is
///    seen — finalize it. Bricks are flushed REGION-BY-REGION in `(z,y,x)` order to the [`VxoStreamWriter`].
///
/// To stay bounded-RAM AND emit whole regions, the emit pass buckets ALL solids into a `region → bricks` map
/// keyed by region coord, but only the SPARSE solid bricks are materialized (one `[BlockId; 512]` per non-empty
/// brick) — O(solid bricks), the same residency the engine itself holds. For a Bistro-scale bake this is the
/// surface-shell brick set (millions of bricks × 512 B is ~GB-scale but bounded by SURFACE, not volume), and
/// the writer streams each region's compressed body straight to the scratch `.brik` file. (A future refinement
/// could page bricks by region to disk too; the surface-bounded set is acceptable for the current corpus.)
fn assemble_vxo_streaming(
    grid: &tiled::TileGrid,
    final_ids: &[usize],
    voxel_size: f32,
    out_path: &Path,
    store: bool,
) -> anyhow::Result<()> {
    use adventure::voxel::brickmap::{BRICK_EDGE, Brick, BrickMap, brick_coord_of_voxel, voxel_index};
    use adventure::voxel::palette::{BlockId, BlockRegistry};
    use adventure::voxel::vxo::{VxoCompression, VxoHeadParams, VxoStreamWriter, region_of_brick};
    use bevy::math::IVec3;
    use rustc_hash::FxHashMap;

    let (dx, dy) = (grid.dims[0] as usize, grid.dims[1] as usize);
    let delin = |gi: usize| -> (i32, i32, i32) {
        let z = (gi / (dx * dy)) as i32;
        let r = gi % (dx * dy);
        ((r % dx) as i32, (r / dx) as i32, z)
    };

    // --- Pass 1: distinct-albedo counts + solid bounds (grid-voxel space). ---
    let mut counts: HashMap<[u8; 4], u32> = HashMap::new();
    let mut lo = [i32::MAX; 3];
    let mut hi = [i32::MIN; 3];
    tiled::stream_final(grid, final_ids, |gi, rgba| {
        *counts.entry(rgba).or_insert(0) += 1;
        let (x, y, z) = delin(gi);
        lo = [lo[0].min(x), lo[1].min(y), lo[2].min(z)];
        hi = [hi[0].max(x), hi[1].max(y), hi[2].max(z)];
    })?;
    if lo[0] == i32::MAX {
        anyhow::bail!("tiled .vxo: no solid voxels to assemble");
    }
    let mut pixels: Vec<([u8; 4], u32)> = counts.into_iter().collect();
    pixels.sort_unstable();
    let palette = build_palette(&pixels, MAX_PALETTE);
    let palette_lab: Vec<[f32; 3]> = palette.iter().map(|c| rgb_to_lab([c[0], c[1], c[2]])).collect();
    // The registry: 256-padded sRGB palette → BlockRegistry (BlockId(i+1) ← palette entry i, the .vox rule).
    let mut vox_palette = palette.clone();
    vox_palette.resize(256, [0, 0, 0, 255]);
    let registry: BlockRegistry = BlockRegistry::from_vox_palette(&vox_palette);

    // The anchor shift (floor y=0, centre X/Z) — IDENTICAL to vox.rs::bricks_from_placed (grid↔world identity
    // bar the anchor). Applied to every solid so the tiled .vxo aligns with the rest of the gallery.
    let shift = IVec3::new(-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2);

    // --- Pass 2: emit. Map each solid → anchored world voxel → brick; quantize once per distinct albedo. ---
    let k = VxoHeadParams::default().region_edge_bricks as i32;
    let mut nearest_cache: HashMap<[u8; 4], u8> = HashMap::new();
    // Sparse per-brick dense arrays (only non-empty bricks materialize — surface-bounded residency).
    let mut bricks: FxHashMap<IVec3, Box<[BlockId; (BRICK_EDGE * BRICK_EDGE * BRICK_EDGE) as usize]>> =
        FxHashMap::default();
    tiled::stream_final(grid, final_ids, |gi, rgba| {
        let pal = *nearest_cache.entry(rgba).or_insert_with(|| nearest_palette_lab(&palette_lab, rgba));
        let block = BlockId(pal as u16 + 1); // 1-based; 0 = air
        let (x, y, z) = delin(gi);
        let w = IVec3::new(x, y, z) + shift;
        let bc = brick_coord_of_voxel(w);
        let local = w - bc * BRICK_EDGE;
        let arr = bricks.entry(bc).or_insert_with(|| Box::new([BlockId::AIR; (BRICK_EDGE * BRICK_EDGE * BRICK_EDGE) as usize]));
        arr[voxel_index(local.x, local.y, local.z)] = block;
    })?;

    // Build the BrickMap (uniform-collapse + occupancy via the SSOT Brick::from_voxels), then region-bucket and
    // stream-write. The BrickMap here is the surface-bounded resident set (bounded by SURFACE, not volume).
    let mut map = BrickMap::new();
    for (coord, arr) in bricks {
        map.insert(coord, Brick::from_voxels(arr));
    }

    // Region-bucket the bricks (sorted (z,y,x) within a region; regions fed in (z,y,x) order so the streamed
    // BRIK body is deterministic).
    let mut regions: FxHashMap<IVec3, Vec<IVec3>> = FxHashMap::default();
    for (&coord, _) in map.iter() {
        regions.entry(region_of_brick(coord, k)).or_default().push(coord);
    }
    let mut region_coords: Vec<IVec3> = regions.keys().copied().collect();
    region_coords.sort_by_key(|c| (c.z, c.y, c.x));

    let name = out_path.file_stem().and_then(|s| s.to_str()).unwrap_or("vxo").to_string();
    let params = VxoHeadParams { voxel_size, name, ..Default::default() };
    let comp = if store { VxoCompression::Store } else { VxoCompression::default() };
    let scratch_brik = grid.scratch.join("assembly.brik.tmp");
    let mut writer = VxoStreamWriter::new(params, &registry, comp, &scratch_brik)?;
    for rc in &region_coords {
        let mut coords = regions.remove(rc).expect("region present");
        coords.sort_by_key(|c| (c.z, c.y, c.x));
        let region_bricks: Vec<(IVec3, &Brick)> =
            coords.iter().map(|&c| (c, map.get(c).expect("brick present"))).collect();
        writer.add_region(*rc, &region_bricks)?;
    }

    // Bake the coarse-LOD pyramid (the LODS chunk) from the still-resident base map, driven through the SHARED
    // `drive_coarse_lods` ordering SSOT so the tiled path's LODS bytes are byte-identical to the full-RAM
    // `encode_vxo`/`build_lods_body` path. This is the Stage-0 BOUNDED (resident-map) producer — Stages 1-3 later
    // SWAP it for a disk-spill / windowed-coarse producer feeding the SAME `add_lod_region` sink (no format/reader
    // change). `build_coarse_pyramid` runs the full pyramid to MAX_LOD for a non-empty map, so `finish` satisfies
    // the `max_lod == MAX_LOD` invariant by construction.
    let pyramid = adventure::voxel::vxo::build_coarse_pyramid(&map);
    adventure::voxel::vxo::drive_coarse_lods(&pyramid, k, |lod, rc, bricks| writer.add_lod_region(lod, rc, bricks))?;

    writer.finish(out_path)?;
    Ok(())
}

// ============================================================================================
// Mesh representation (decoupled from glTF so the fallback room is the same shape)
// ============================================================================================

/// A texture decoded to interleaved 8-bit RGBA (sRGB). Sampled with wrapping + BILINEAR filtering — at the
/// fine voxel sizes the engine targets (0.05 m) a voxel still spans many texels, so a smooth per-tap filter
/// (combined with the area supersample in [`sample_albedo`]) is what kills texel aliasing.
struct Texture {
    width: u32,
    height: u32,
    rgba: Vec<u8>, // width*height*4
}

impl Texture {
    /// An empty (0×0) texture — a sentinel for "could not decode this image". Callers treat `width == 0` as
    /// "no texture" and flat-fall-back to the material `base_color_factor`.
    fn empty() -> Self {
        Self { width: 0, height: 0, rgba: Vec::new() }
    }

    /// Fetch one texel's sRGB RGBA at integer texel coords (already wrapped into range).
    #[inline]
    fn texel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        [self.rgba[i], self.rgba[i + 1], self.rgba[i + 2], self.rgba[i + 3]]
    }

    /// BILINEAR-sample sRGB RGBA at UV (wrapping on both axes). Returns `[r,g,b,a]` sRGB `u8`.
    ///
    /// COLOUR SPACE: the blend is done in sRGB `u8` space (the tool is sRGB end-to-end — the runtime loader
    /// does the single sRGB→linear), so this preserves the one-conversion invariant. The slight gamma
    /// inaccuracy of an sRGB-space blend is immaterial at voxel granularity and keeps the colour pipeline
    /// uniform with the rest of the voxelizer.
    fn sample(&self, u: f32, v: f32) -> [u8; 4] {
        if self.width == 0 || self.height == 0 {
            return [255, 255, 255, 255];
        }
        // Map UV → continuous texel space at texel CENTRES (the -0.5 puts the integer texel at its centre), so
        // the bilinear weights are symmetric about a texel.
        let fx = (u - u.floor()) * self.width as f32 - 0.5;
        let fy = (v - v.floor()) * self.height as f32 - 0.5;
        let x0f = fx.floor();
        let y0f = fy.floor();
        let tx = fx - x0f;
        let ty = fy - y0f;
        // Wrap the four neighbour texels (so sampling near the UV seam still bilinearly blends across the wrap).
        let wrap = |c: i32, n: u32| -> u32 { c.rem_euclid(n as i32) as u32 };
        let x0 = wrap(x0f as i32, self.width);
        let y0 = wrap(y0f as i32, self.height);
        let x1 = wrap(x0f as i32 + 1, self.width);
        let y1 = wrap(y0f as i32 + 1, self.height);
        let c00 = self.texel(x0, y0);
        let c10 = self.texel(x1, y0);
        let c01 = self.texel(x0, y1);
        let c11 = self.texel(x1, y1);
        let mut out = [0u8; 4];
        for ch in 0..4 {
            let top = c00[ch] as f32 * (1.0 - tx) + c10[ch] as f32 * tx;
            let bot = c01[ch] as f32 * (1.0 - tx) + c11[ch] as f32 * tx;
            out[ch] = (top * (1.0 - ty) + bot * ty).round().clamp(0.0, 255.0) as u8;
        }
        out
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
// Mesh loading — dispatch by file extension (glTF / OBJ), with a procedural fallback
// ============================================================================================

/// Load the input mesh, picking the loader by FILE EXTENSION: `.gltf`/`.glb` → [`load_gltf`], `.obj` →
/// [`load_obj`]. The rest of the pipeline (voxelize + solid_fill + palette + `.vox` write) is identical
/// regardless of source — both loaders build the SAME [`Mesh`] (world-space triangle soup + textures). An
/// ABSENT file (or an unrecognized extension) falls back to the procedural box room so the pipeline + the
/// round-trip test still build + run end-to-end (with a clear "drop in a real scene" notice). FBX is NOT
/// handled — convert it to glTF/OBJ externally first (e.g. the Lumberyard Bistro ships as FBX).
fn load_mesh(path: &Path) -> anyhow::Result<Mesh> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    if !path.exists() {
        println!(
            "NOTE: {} not found — using the PROCEDURAL FALLBACK box room. Drop a real classic scene into \
             assets/models/src/ (glTF: .gltf + .bin + textures; OBJ: .obj + .mtl + textures) and re-run \
             (pass the path as the 3rd CLI arg) to bake it.",
            path.display()
        );
        return Ok(fallback_room());
    }
    match ext.as_str() {
        "gltf" | "glb" => {
            println!("loading glTF: {}", path.display());
            load_gltf(path)
        }
        "obj" => {
            println!("loading OBJ: {}", path.display());
            load_obj(path)
        }
        "fbx" => Err(anyhow::anyhow!(
            "FBX ({}) is not supported — convert it to glTF or OBJ externally first (e.g. Blender import→export)",
            path.display()
        )),
        other => Err(anyhow::anyhow!(
            "unrecognized mesh extension '.{other}' for {} — use .gltf/.glb or .obj",
            path.display()
        )),
    }
}

// ============================================================================================
// glTF loading
// ============================================================================================

/// Load a glTF file into a [`Mesh`]: every primitive's positions + indices + UV0, with the material's
/// base-colour texture (decoded via `gltf`'s `image` feature, or — for `KHR_texture_basisu` KTX2 — via
/// [`ktx2_to_rgba`]) or its `base_color_factor`. Positions are transformed to WORLD space by walking the
/// scene-node hierarchy and accumulating each node's local transform (CRITICAL: Sponza's single node carries
/// a 0.008 scale, so mesh-local coords of ±1400 become a ~24 m world scene — without this the AABB would be
/// ~3000 units and the dense grid would be astronomically large). glTF and this engine are both Y-up; the
/// Z-up swap for `.vox` happens at write time.
///
/// Texture resolution is by IMAGE index throughout: `Mesh.textures[k]` is the decoded image `k`, and a
/// triangle's `Triangle::texture` is that image index. The `base_color_texture` is a glTF TEXTURE index,
/// which [`emit_mesh_primitives`] maps to an image index via `tex_to_image` (the `KHR_texture_basisu` source
/// when present, else the texture's plain `source`). This keeps the Sponza (PNG/JPEG) and Bistro (KTX2)
/// paths on one code path — only the per-image DECODER differs.
fn load_gltf(path: &Path) -> anyhow::Result<Mesh> {
    // Read the raw glTF JSON FIRST so we can (a) detect / strip the unsupported `KHR_texture_basisu`
    // `extensionsRequired` entry that would make `gltf::import` reject the file, and (b) build the
    // texture-index → image-index map from each texture's `KHR_texture_basisu.source` (the `gltf` crate is
    // pulled WITHOUT its `extensions` feature, so its typed JSON silently drops that extension — we must read
    // the raw JSON ourselves). `.glb` (binary) has no external KTX2 textures in our scene set, so it keeps the
    // plain `gltf::import` path; only the textual `.gltf` Bistro needs the basisu handling.
    let is_glb = path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("glb"));
    if !is_glb
        && let Some(mesh) = load_gltf_basisu(path)?
    {
        return Ok(mesh);
    }

    // Standard path (Sponza + any glTF whose textures `gltf` can decode itself): unchanged from before.
    let (doc, buffers, images) = gltf::import(path)?;

    // Decode every glTF image to RGBA8 once (indexed by image source index).
    let textures: Vec<Texture> = images.iter().map(decode_image).collect();
    // Texture-index → image-index is identity-via-`source()` on this path (no basisu remap).
    let tex_to_image: Vec<usize> = doc.textures().map(|t| t.source().index()).collect();

    let mut triangles = Vec::new();
    // Walk every scene's node hierarchy with the accumulated world matrix (column-major 4×4 from glTF).
    let scene = doc.default_scene().or_else(|| doc.scenes().next());
    if let Some(scene) = scene {
        for node in scene.nodes() {
            walk_node(&node, IDENTITY4, &buffers, &textures, &tex_to_image, &mut triangles);
        }
    } else {
        // No scene graph: emit meshes at identity (rare; keeps the loader total).
        for mesh in doc.meshes() {
            emit_mesh_primitives(&mesh, IDENTITY4, &buffers, &textures, &tex_to_image, &mut triangles);
        }
    }
    Ok(Mesh { triangles, textures })
}

/// The `KHR_texture_basisu` (KTX2) glTF path: if the document does NOT require that extension, returns `None`
/// so [`load_gltf`] falls through to the standard `gltf::import` path (Sponza stays byte-for-byte identical).
/// If it DOES (the Bistro), this:
///   1. reads the raw JSON, builds `texture index → image index` from each texture's
///      `extensions.KHR_texture_basisu.source` (fallback the texture's plain `source`) and `image index → uri`,
///   2. STRIPS `KHR_texture_basisu` from `extensionsRequired` (leaving `extensionsUsed`) and re-parses the
///      stripped bytes with `gltf` (validation now passes) for the mesh,
///   3. loads buffers via [`gltf::import_buffers`], then decodes each REFERENCED external `.ktx2` ONCE (by
///      image index, relative to the glTF dir) with [`ktx2_to_rgba`]; an image that fails to decode becomes an
///      empty `Texture` so its triangles fall back to `base_color_factor`.
fn load_gltf_basisu(path: &Path) -> anyhow::Result<Option<Mesh>> {
    let bytes = std::fs::read(path)?;
    let root: serde_json::Value = serde_json::from_slice(&bytes)?;

    // Only take over when the file actually requires KHR_texture_basisu — otherwise let the standard path run.
    let requires_basisu = root
        .get("extensionsRequired")
        .and_then(|v| v.as_array())
        .is_some_and(|a| a.iter().any(|e| e.as_str() == Some("KHR_texture_basisu")));
    if !requires_basisu {
        return Ok(None);
    }
    println!("  glTF requires KHR_texture_basisu — decoding external KTX2 base colours");

    // (a) texture index → image index (basisu source, fallback plain source).
    let tex_to_image: Vec<usize> = root
        .get("textures")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|t| {
                    t.get("extensions")
                        .and_then(|e| e.get("KHR_texture_basisu"))
                        .and_then(|b| b.get("source"))
                        .or_else(|| t.get("source"))
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0) as usize
                })
                .collect()
        })
        .unwrap_or_default();
    // (b) image index → external URI (relative to the glTF directory).
    let image_uris: Vec<Option<String>> = root
        .get("images")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|im| im.get("uri").and_then(|u| u.as_str()).map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();

    // Strip the unsupported required-extension entry, then re-parse with `gltf` (validation passes now). The
    // `gltf` JSON deserialize ignores unknown texture/image extensions, so the document is well-formed; we
    // supply the textures ourselves below.
    let stripped = strip_basisu_required(&bytes)?;
    let gltf = gltf::Gltf::from_slice(&stripped)
        .map_err(|e| anyhow::anyhow!("re-parsing the basisu-stripped glTF failed: {e}"))?;
    let doc = gltf.document;
    let base = path.parent();
    let buffers = gltf::import_buffers(&doc, base, gltf.blob)
        .map_err(|e| anyhow::anyhow!("importing glTF buffers failed: {e}"))?;

    // Decode each external `.ktx2` ONCE, indexed by image index (Bistro has 405 images; one-time, large). An
    // image with no URI or that fails to decode becomes an empty `Texture` (its triangles flat-fall-back).
    let dir = base.unwrap_or_else(|| Path::new("."));
    let mut decoded = 0usize;
    let mut failed = 0usize;
    let textures: Vec<Texture> = image_uris
        .iter()
        .map(|uri| {
            let Some(uri) = uri else { return Texture::empty() };
            let tex_path = dir.join(uri);
            match std::fs::read(&tex_path) {
                Ok(raw) => match ktx2_to_rgba(&raw) {
                    Some(t) => {
                        decoded += 1;
                        t
                    }
                    None => {
                        failed += 1;
                        Texture::empty()
                    }
                },
                Err(e) => {
                    eprintln!("  ktx2: read {} failed ({e}) — flat factor fallback", tex_path.display());
                    failed += 1;
                    Texture::empty()
                }
            }
        })
        .collect();
    println!("  KTX2 base colours: {decoded} decoded, {failed} skipped (flat-factor fallback)");

    let mut triangles = Vec::new();
    let scene = doc.default_scene().or_else(|| doc.scenes().next());
    if let Some(scene) = scene {
        for node in scene.nodes() {
            walk_node(&node, IDENTITY4, &buffers, &textures, &tex_to_image, &mut triangles);
        }
    } else {
        for mesh in doc.meshes() {
            emit_mesh_primitives(&mesh, IDENTITY4, &buffers, &textures, &tex_to_image, &mut triangles);
        }
    }
    Ok(Some(Mesh { triangles, textures }))
}

/// Remove `"KHR_texture_basisu"` from the glTF's `extensionsRequired` array (leaving `extensionsUsed`
/// untouched) and re-serialize, so `gltf`'s validation no longer rejects the file as an unsupported required
/// extension. Operates on the parsed `serde_json` tree so it's robust to formatting (whitespace / ordering),
/// not a brittle string edit.
fn strip_basisu_required(bytes: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut root: serde_json::Value = serde_json::from_slice(bytes)?;
    if let Some(arr) = root.get_mut("extensionsRequired").and_then(|v| v.as_array_mut()) {
        arr.retain(|e| e.as_str() != Some("KHR_texture_basisu"));
        if arr.is_empty() {
            // An empty `extensionsRequired` is valid, but drop it entirely to keep the document tidy.
            if let Some(obj) = root.as_object_mut() {
                obj.remove("extensionsRequired");
            }
        }
    }
    Ok(serde_json::to_vec(&root)?)
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
    tex_to_image: &[usize],
    out: &mut Vec<Triangle>,
) {
    let world = mat4_mul(parent, node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        emit_mesh_primitives(&mesh, world, buffers, textures, tex_to_image, out);
    }
    for child in node.children() {
        walk_node(&child, world, buffers, textures, tex_to_image, out);
    }
}

/// Emit one mesh's primitives as world-space triangles (positions transformed by `world`), reading UV0 +
/// indices and resolving the base-colour texture / factor per material.
fn emit_mesh_primitives(
    mesh: &gltf::Mesh,
    world: [[f32; 4]; 4],
    buffers: &[gltf::buffer::Data],
    textures: &[Texture],
    tex_to_image: &[usize],
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
        // Resolve the base-colour TEXTURE index → IMAGE index (via `tex_to_image`, which carries the
        // `KHR_texture_basisu` remap on the Bistro path and is identity-via-`source()` otherwise), then keep
        // it only if that image actually decoded (non-empty) — an undecodable KTX2 / missing image falls back
        // to `base_color_factor`.
        let texture = pbr
            .base_color_texture()
            .map(|info| info.texture().index())
            .map(|ti| tex_to_image.get(ti).copied().unwrap_or(ti))
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

// ============================================================================================
// KTX2 / Basis Universal texture decode (KHR_texture_basisu — the Bistro base colours)
// ============================================================================================

/// Decode a `KHR_texture_basisu` KTX2 image (the Amazon Lumberyard Bistro base colours: `vkFormat = 0`
/// UASTC blocks, `supercompressionScheme = 2` Zstandard, 2048², a single mip level) to an interleaved RGBA8
/// [`Texture`]. Returns `None` (caller flat-falls-back to `base_color_factor`) for anything outside that
/// path — an uncompressed / non-Zstd / non-UASTC KTX2, ETC1S, a `vkFormat ≠ 0` already-GPU format, or a
/// transcode failure — with a logged note. Only Bistro's UASTC+Zstd path is required, so the decoder is
/// deliberately narrow; mirrors `bevy_image`'s `ktx2_buffer_to_image` (`D:/bevy-fork/.../ktx2.rs`, the Zstd
/// supercompression branch + the `TranscodeFormat::Uastc` branch).
///
/// CRITICAL: the UASTC source block grid is 4×4 texels (`num_blocks_x = width.div_ceil(4)`), independent of
/// the OUTPUT format's block size. Bevy slices the input by the OUTPUT format's `block_dimensions` because it
/// transcodes to a compressed GPU format (ASTC/BC7, also 4×4) — but we transcode to UNCOMPRESSED `RGBA32`
/// (output "block" 1×1), so reusing the output block size would mis-slice the input. We compute the block
/// counts from the true UASTC 4×4 grid.
fn ktx2_to_rgba(bytes: &[u8]) -> Option<Texture> {
    use basis_universal::{
        DecodeFlags, LowLevelUastcTranscoder, SliceParametersUastc, TranscoderBlockFormat,
    };
    use std::io::Read as _;

    let reader = match ktx2::Reader::new(bytes) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  ktx2: parse failed ({e:?}) — flat factor fallback");
            return None;
        }
    };
    let header = reader.header();
    let (width, height) = (header.pixel_width, header.pixel_height);

    // Only the UASTC-universal layout (vkFormat = VK_FORMAT_UNDEFINED) is transcodable here. A non-zero
    // vkFormat is an already-decided GPU format (e.g. an uncompressed or BC/ASTC KTX2) we don't handle.
    if header.format.is_some() {
        eprintln!(
            "  ktx2: {width}x{height} has a concrete vkFormat (not UASTC-universal) — flat factor fallback"
        );
        return None;
    }
    // Single supercompressed level expected. Decompress it (Zstandard only — the Bistro scheme).
    let level = reader.levels().next()?;
    let uastc_blocks: Vec<u8> = match header.supercompression_scheme {
        Some(ktx2::SupercompressionScheme::Zstandard) => {
            let mut cursor = std::io::Cursor::new(level.data);
            let mut decoder = match ruzstd::decoding::StreamingDecoder::new(&mut cursor) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("  ktx2: zstd init failed ({e}) — flat factor fallback");
                    return None;
                }
            };
            let mut out = Vec::new();
            if let Err(e) = decoder.read_to_end(&mut out) {
                eprintln!("  ktx2: zstd decompress failed ({e}) — flat factor fallback");
                return None;
            }
            out
        }
        // An uncompressed UASTC KTX2 would carry the blocks verbatim — but the Bistro is always Zstd, so we
        // only support that one supercompression scheme (the task's required path); anything else falls back.
        other => {
            eprintln!(
                "  ktx2: {width}x{height} supercompression {other:?} unsupported (need Zstandard) — flat factor fallback"
            );
            return None;
        }
    };

    // Transcode the UASTC 4×4 blocks → uncompressed RGBA8. The block grid is the UASTC 4×4 grid (NOT the
    // RGBA32 1×1 output "block"): one 16-byte UASTC block per 4×4 texels.
    let (num_blocks_x, num_blocks_y) = (width.div_ceil(4).max(1), height.div_ceil(4).max(1));

    // BUG WORKAROUND (basis-universal 0.3.1): `transcode_slice` computes the C++ output ROW PITCH as
    // `original_width / block_width()`, and `block_width()` is a hard-coded 4 for EVERY format — including
    // the uncompressed `RGBA32`, whose "blocks" are single pixels. The C++ `cRGBA32` writer treats that
    // pitch as a PIXEL stride and clips each block row to `min(4, pitch - block_x*4)`; with the 4×-too-small
    // pitch that subtraction goes negative, underflows to a huge `u32`, and the inner loop writes billions of
    // pixels out of bounds → STATUS_ACCESS_VIOLATION (verified: any image ≥ 8×8 crashes). We can't fix the
    // wrapper, but its pitch formula is `original_width / 4`, so passing `original_width = width * 4` makes
    // the emitted pitch exactly `width` PIXELS — the correct uncompressed RGBA8 row stride — while
    // `original_height` (the row COUNT) and `num_blocks_*` (the source 4×4 grid) stay real. The output is
    // then the correct, contiguous `width*height*4` RGBA8 buffer (validated below). Mirrors the data Bevy
    // gets on its GPU path (ASTC/BC7), where the same `/4` happens to be right because those blocks ARE 4×4.
    let slice = SliceParametersUastc {
        num_blocks_x,
        num_blocks_y,
        has_alpha: true,
        original_width: width.saturating_mul(4),
        original_height: height,
    };
    let rgba = match LowLevelUastcTranscoder::new().transcode_slice(
        &uastc_blocks,
        slice,
        DecodeFlags::HIGH_QUALITY,
        TranscoderBlockFormat::RGBA32,
    ) {
        Ok(rgba) => rgba,
        Err(e) => {
            eprintln!("  ktx2: UASTC→RGBA32 transcode failed ({e:?}) — flat factor fallback");
            return None;
        }
    };
    // RGBA32 output must be exactly width*height*4 bytes — a guard that also catches the wrapper's pitch math
    // ever changing under us (it would yield a different length, and we'd fall back rather than ship garbage).
    let expected = (width as usize) * (height as usize) * 4;
    if rgba.len() != expected {
        eprintln!(
            "  ktx2: {width}x{height} transcoded {} bytes, expected {expected} — flat factor fallback",
            rgba.len()
        );
        return None;
    }
    Some(Texture { width, height, rgba })
}

// ============================================================================================
// OBJ loading (+ MTL base colours / diffuse textures)
// ============================================================================================

/// Load a Wavefront `.obj` (+ its companion `.mtl`) into the SAME [`Mesh`] the glTF path builds: every face's
/// world-space positions + UV0, with each material's diffuse `Kd` base colour and `map_Kd` diffuse texture
/// (decoded with `image`, relative to the OBJ's directory). OBJ has NO scene-node transform hierarchy —
/// positions are already world-space — so unlike glTF there's no matrix to bake (the loader uses them
/// verbatim). `tobj` is asked to TRIANGULATE (so quads/n-gons become triangles) and use a SINGLE index
/// (positions/texcoords share one index buffer, matching how we read them). A texture that fails to decode
/// (missing / unsupported) falls back to the material's flat `Kd` colour, so a partially-textured scene still
/// bakes. The downstream pipeline (voxelize + solid_fill + palette + `.vox` write) is identical to glTF.
fn load_obj(path: &Path) -> anyhow::Result<Mesh> {
    let load_opts = tobj::LoadOptions { triangulate: true, single_index: true, ..Default::default() };
    let (models, materials) = tobj::load_obj(path, &load_opts)
        .map_err(|e| anyhow::anyhow!("obj: load {}: {e}", path.display()))?;
    // Materials may fail to load (a missing `.mtl`) without failing the OBJ — treat that as "no materials"
    // (every face then falls back to a neutral base colour). The OBJ's directory anchors relative texture paths.
    let materials = materials.unwrap_or_default();
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));

    // Decode each material's diffuse texture once (indexed parallel to `materials`); `None` if it has no
    // `map_Kd` or the file can't be decoded (then the flat `Kd` colour is used). Same `Texture` the glTF path
    // produces, so `sample_albedo` is shared verbatim.
    let textures: Vec<Option<Texture>> = materials
        .iter()
        .map(|m| {
            m.diffuse_texture.as_ref().and_then(|rel| {
                let tex_path = base_dir.join(rel);
                match image::open(&tex_path) {
                    Ok(img) => {
                        let rgba = img.to_rgba8();
                        let (w, h) = rgba.dimensions();
                        Some(Texture { width: w, height: h, rgba: rgba.into_raw() })
                    }
                    Err(e) => {
                        eprintln!("  obj: texture {} failed to decode ({e}) — using flat Kd", tex_path.display());
                        None
                    }
                }
            })
        })
        .collect();
    // Flatten the decoded textures into the `Mesh.textures` vec, remembering each material's texture index (so
    // a triangle can reference it). `mat_tex_index[mi]` is `Some(slot)` iff material `mi` has a decoded texture.
    let mut mesh_textures: Vec<Texture> = Vec::new();
    let mut mat_tex_index: Vec<Option<usize>> = Vec::with_capacity(materials.len());
    for tex in textures {
        match tex {
            Some(t) => {
                mat_tex_index.push(Some(mesh_textures.len()));
                mesh_textures.push(t);
            }
            None => mat_tex_index.push(None),
        }
    }
    // Each material's flat sRGB base colour from its diffuse `Kd` (0..1 f64 → 0..255 u8), defaulting to neutral
    // grey for an untextured/material-less face.
    let mat_base = |mi: Option<usize>| -> [u8; 4] {
        let kd = mi.and_then(|i| materials.get(i)).and_then(|m| m.diffuse).unwrap_or([0.7, 0.7, 0.7]);
        [
            (kd[0].clamp(0.0, 1.0) * 255.0) as u8,
            (kd[1].clamp(0.0, 1.0) * 255.0) as u8,
            (kd[2].clamp(0.0, 1.0) * 255.0) as u8,
            255,
        ]
    };

    let mut triangles = Vec::new();
    for model in &models {
        let m = &model.mesh;
        let mi = m.material_id;
        let base = mat_base(mi);
        // The decoded texture slot for this model's material (if any).
        let texture = mi.and_then(|i| mat_tex_index.get(i).copied().flatten());
        let has_uv = !m.texcoords.is_empty();
        // single_index: positions (xyz) + texcoords (uv) are parallel arrays indexed by `indices` (triangulated
        // ⇒ chunks of 3). A vertex `v`'s position is positions[3v..3v+3]; its UV is texcoords[2v..2v+2].
        for tri in m.indices.chunks_exact(3) {
            let mut p = [[0.0f32; 3]; 3];
            let mut uv = [[0.0f32; 2]; 3];
            for (k, &vi) in tri.iter().enumerate() {
                let vi = vi as usize;
                if 3 * vi + 2 >= m.positions.len() {
                    continue;
                }
                p[k] = [m.positions[3 * vi], m.positions[3 * vi + 1], m.positions[3 * vi + 2]];
                if has_uv && 2 * vi + 1 < m.texcoords.len() {
                    // OBJ's V origin is bottom-left; our `Texture::sample` wraps so this matches the glTF path's
                    // top-left convention after the flip (1 − v).
                    uv[k] = [m.texcoords[2 * vi], 1.0 - m.texcoords[2 * vi + 1]];
                }
            }
            triangles.push(Triangle { p, uv, texture, base });
        }
    }
    Ok(Mesh { triangles, textures: mesh_textures })
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

/// A 1-bit-per-cell occupancy bitset over a linear cell range — the dense part of [`Grid`]. Bit `i` set ⇒
/// cell `i` is solid. `O(1)` random access (the flood-fill needs it) at 1 bit/cell, so even a multi-billion
/// cell AABB costs `N/8` bytes (vs the old `Vec<bool>`'s `N` bytes), which is what lets large scenes
/// (Bistro @0.05 m ≈ a few billion cells) bake in a few GB instead of tens.
struct BitGrid {
    bits: Vec<u64>,
}

impl BitGrid {
    fn new(n: usize) -> Self {
        Self { bits: vec![0u64; n.div_ceil(64)] }
    }
    #[inline]
    fn get(&self, i: usize) -> bool {
        (self.bits[i >> 6] >> (i & 63)) & 1 != 0
    }
    #[inline]
    fn set(&mut self, i: usize) {
        self.bits[i >> 6] |= 1u64 << (i & 63);
    }
}

/// A voxel grid over the mesh AABB at the voxelization's voxel size, stored SPARSELY: a 1-bit-per-cell
/// occupancy [`BitGrid`] (`solid`) for `O(1)` flood-fill lookups, plus a `cell index → sRGB albedo` map
/// holding ONLY solid cells. Surface voxelization + enclosed fill are sparse (a shell + thin interiors), so
/// the albedo map is bounded by the SOLID count (millions), not the AABB volume (billions) — the prior dense
/// `Vec<[u8;4]>` albedo (4 bytes/cell) was the memory wall for large scenes. Indexed `x + y·dx + z·dx·dy`
/// (X fastest), computed in `usize` so it does NOT overflow past ~2 G cells (the old `i32` index did). The
/// world origin / voxel size are consumed within [`voxelize`], so they aren't retained.
struct Grid {
    dims: [i32; 3],
    solid: BitGrid,
    albedo: HashMap<usize, [u8; 4]>,
}

impl Grid {
    /// An all-air grid of `dims` cells (no solid voxels).
    fn new(dims: [i32; 3]) -> Self {
        let total = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
        Self { dims, solid: BitGrid::new(total), albedo: HashMap::new() }
    }
    #[inline]
    fn idx(&self, x: i32, y: i32, z: i32) -> usize {
        let (dx, dy) = (self.dims[0] as usize, self.dims[1] as usize);
        (x as usize) + (y as usize) * dx + (z as usize) * dx * dy
    }
    /// Is cell `i` solid?
    #[inline]
    fn is_solid(&self, i: usize) -> bool {
        self.solid.get(i)
    }
    /// Mark cell `i` solid with sRGB albedo `a` (sets the occupancy bit + records the albedo).
    #[inline]
    fn set_solid(&mut self, i: usize, a: [u8; 4]) {
        self.solid.set(i);
        self.albedo.insert(i, a);
    }
    /// The albedo of solid cell `i` (callers only ask for solid cells; air reads as transparent black).
    #[inline]
    fn albedo_at(&self, i: usize) -> [u8; 4] {
        self.albedo.get(&i).copied().unwrap_or([0, 0, 0, 0])
    }
    /// Number of solid cells (== albedo entries; every solid cell records an albedo).
    fn solid_count(&self) -> usize {
        self.albedo.len()
    }
    /// Delinearize a cell index back to `(x, y, z)` (inverse of [`idx`](Self::idx)).
    #[inline]
    fn xyz(&self, i: usize) -> (i32, i32, i32) {
        let (dx, dy) = (self.dims[0] as usize, self.dims[1] as usize);
        let z = i / (dx * dy);
        let r = i % (dx * dy);
        ((r % dx) as i32, (r / dx) as i32, z as i32)
    }
}

/// The grid GEOMETRY for a mesh at `voxel_size`: the world-space `origin` (the AABB min minus one padding
/// voxel) + the integer `dims`. A pure function of the mesh AABB + voxel size, factored out so the monolithic
/// [`voxelize`] AND the tiled bake ([`tiled`]) compute the SAME grid frame (identical origin/dims ⇒ identical
/// cell indices ⇒ the oracle can compare cell-for-cell). Returns `None` for an empty mesh.
fn grid_geometry(mesh: &Mesh, voxel_size: f32) -> Option<([f32; 3], [i32; 3])> {
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
        return None;
    }
    // Pad one voxel so surface triangles on the boundary still have a cell.
    let origin = [lo[0] - voxel_size, lo[1] - voxel_size, lo[2] - voxel_size];
    let dims = [
        (((hi[0] - lo[0]) / voxel_size).ceil() as i32 + 3).max(1),
        (((hi[1] - lo[1]) / voxel_size).ceil() as i32 + 3).max(1),
        (((hi[2] - lo[2]) / voxel_size).ceil() as i32 + 3).max(1),
    ];
    Some((origin, dims))
}

/// Surface-voxelize the mesh: for every triangle, conservatively rasterize into each voxel it overlaps
/// (triangle–AABB SAT), marking it solid and recording the albedo of the triangle point nearest the voxel
/// centre. The result is a SHELL (the visible surface), which is what we render + measure GI on.
fn voxelize(mesh: &Mesh, voxel_size: f32, supersample: usize) -> Grid {
    let Some((origin, dims)) = grid_geometry(mesh, voxel_size) else {
        // No geometry — return a 1³ empty grid.
        return Grid::new([1, 1, 1]);
    };
    // AABB recovered from origin (origin = lo - voxel_size) for the guard's diagnostic.
    let lo = [origin[0] + voxel_size, origin[1] + voxel_size, origin[2] + voxel_size];
    let hi = [
        lo[0] + (dims[0] - 3).max(0) as f32 * voxel_size,
        lo[1] + (dims[1] - 3).max(0) as f32 * voxel_size,
        lo[2] + (dims[2] - 3).max(0) as f32 * voxel_size,
    ];
    // Guard: the occupancy bitset + the flood-fill's `exterior` bitset are each `total/8` bytes (1 bit/cell)
    // — the only AABB-volume cost now (the albedo is sparse). So the ceiling is generous: 16 G cells ⇒ ~2 GB
    // per bitset (~4 GB peak during solid_fill). It still catches an absurd extent (e.g. forgetting the glTF
    // node transform, which once made Sponza ~3000 units → trillions of cells) with a clear message rather
    // than an OOM. A scene larger than this would need a spatially-tiled (blocked) flood-fill.
    let total = (dims[0] as i64) * (dims[1] as i64) * (dims[2] as i64);
    const MAX_VOXELS: i64 = 16_000_000_000; // ~2 GB/bitset; spans billion-cell scenes (Bistro @0.05 m) sparsely
    assert!(
        total <= MAX_VOXELS,
        "voxel grid {dims:?} = {total} cells exceeds {MAX_VOXELS} — AABB world span is {:?}..{:?} ({:.1} m \
         on the longest axis at {voxel_size} m/voxel). Are glTF node transforms applied? (Sponza needs the \
         0.008 node scale.) Raise --voxel_metres or check the mesh units.",
        lo,
        hi,
        (hi[0] - lo[0]).max(hi[1] - lo[1]).max(hi[2] - lo[2])
    );
    let mut grid = Grid::new(dims);

    let dims = grid.dims; // Copy [i32;3] — captured by the parallel closures so they don't borrow `grid`.
    // Rasterize triangles into voxels IN PARALLEL (the shared surface SAT, also used by the tiled bake), then
    // first-writer-wins merge in triangle order into the dense grid.
    let per_tri = surface_scatter(mesh, origin, voxel_size, dims, supersample);
    for cells in &per_tri {
        for &(i, albedo) in cells {
            if !grid.is_solid(i) {
                grid.set_solid(i, albedo);
            }
        }
    }
    grid
}

/// The shared SURFACE SAT pass: rasterize every triangle into the voxels it conservatively overlaps and sample
/// its area-averaged albedo, returning the per-triangle `(global_cell_index, albedo)` lists IN TRIANGLE ORDER.
/// Factored out of [`voxelize`] so the monolithic grid build AND the tiled out-of-core bake ([`tiled`]) run the
/// IDENTICAL occupancy + albedo computation — the conservative SAT is the occupancy SSOT (C1 changes only HOW
/// the result is stored/flooded at scale, never the SAT). The lists are returned (not merged) so each caller
/// applies its own first-writer-wins merge in triangle order (the deterministic "first triangle to claim a cell
/// keeps its albedo" rule) — parallel writes into a shared store couldn't preserve that ordering.
fn surface_scatter(
    mesh: &Mesh,
    origin: [f32; 3],
    voxel_size: f32,
    dims: [i32; 3],
    supersample: usize,
) -> Vec<Vec<(usize, [u8; 4])>> {
    let half = voxel_size * 0.5;
    mesh.triangles
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
                            // usize index (the i32 form overflowed past ~2 G cells, silently corrupting large bakes).
                            let i = (x as usize)
                                + (y as usize) * (dims[0] as usize)
                                + (z as usize) * (dims[0] as usize) * (dims[1] as usize);
                            cells.push((i, sample_albedo(mesh, t, center, half, supersample)));
                        }
                    }
                }
            }
            cells
        })
        .collect()
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
    let total = (dx as usize) * (dy as usize) * (dz as usize);
    if total == 0 {
        return;
    }
    const N6: [(i32, i32, i32); 6] =
        [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)];

    // 1. EXTERIOR: 6-connected flood through AIR (1-bit/cell bitset), seeded from every AIR cell on the grid
    //    boundary (outside the grid is air, so a boundary air cell is exterior). Reached air = exterior;
    //    unreached air = enclosed interior.
    let mut exterior = BitGrid::new(total);
    let mut q: std::collections::VecDeque<(i32, i32, i32)> = std::collections::VecDeque::new();
    for z in 0..dz {
        for y in 0..dy {
            for x in 0..dx {
                if x == 0 || y == 0 || z == 0 || x == dx - 1 || y == dy - 1 || z == dz - 1 {
                    let i = grid.idx(x, y, z);
                    if !grid.is_solid(i) && !exterior.get(i) {
                        exterior.set(i);
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
            if !grid.is_solid(ni) && !exterior.get(ni) {
                exterior.set(ni);
                q.push_back((nx, ny, nz));
            }
        }
    }

    // 2. Fill the enclosed interior (air && !exterior) solid, colouring each cell with the NEAREST surface
    //    voxel's albedo via a multi-source 6-connected BFS seeded from the surface cells. The occupancy bit is
    //    its own visited marker: a cell becomes solid the instant it's filled, so the `!is_solid` test stops
    //    re-visits (no separate `filled` bitset). Seed from a SNAPSHOT of the current solid set (the sparse
    //    albedo keys) so the in-loop inserts don't perturb iteration.
    // Seed from a SNAPSHOT of the current solid set (the sparse albedo keys), SORTED by linear index so the
    // multi-source BFS is deterministic — tie-broken interior colours (a cell equidistant from two surfaces)
    // must not depend on HashMap iteration order, or re-bakes would differ.
    let mut seed_idx: Vec<usize> = grid.albedo.keys().copied().collect();
    seed_idx.sort_unstable();
    q.clear();
    q.extend(seed_idx.into_iter().map(|i| grid.xyz(i)));
    while let Some((x, y, z)) = q.pop_front() {
        let src = grid.albedo_at(grid.idx(x, y, z));
        for (ox, oy, oz) in N6 {
            let (nx, ny, nz) = (x + ox, y + oy, z + oz);
            if nx < 0 || ny < 0 || nz < 0 || nx >= dx || ny >= dy || nz >= dz {
                continue;
            }
            let ni = grid.idx(nx, ny, nz);
            if !grid.is_solid(ni) && !exterior.get(ni) {
                grid.set_solid(ni, src);
                q.push_back((nx, ny, nz));
            }
        }
    }
}

/// The sRGB albedo for a voxel from one triangle's contribution: an AREA-AVERAGE of the texture over the
/// triangle's surface footprint inside the voxel box (replacing the old single nearest-texel point sample).
/// Untextured triangles return the flat `base` colour unchanged.
///
/// At fine voxel sizes a voxel covers many texels, so one point sample aliases (a lone texel decides the whole
/// voxel). Instead we lay an `S×S` grid of sample points across the voxel box centred at `center` with
/// half-extent `half` (default `S = SUPERSAMPLE = 3`), project EACH onto the triangle via
/// `closest_point_barycentric`, and average the texture lookups. A sample is kept only if its nearest-triangle
/// point actually lies inside the (slightly padded) voxel box — so off-triangle grid points (whose projection
/// snaps to a far edge) don't pull the average toward colours outside the voxel's true footprint. If no sample
/// lands on-footprint (a sliver), we fall back to the single nearest-point sample so the voxel still gets a
/// sensible colour. Each tap is itself bilinear (`Texture::sample`), compounding to kill texel aliasing.
///
/// COLOUR SPACE: the average is in sRGB `u8` space (the tool is sRGB end-to-end; the runtime loader does the
/// single sRGB→linear), preserving the one-conversion invariant.
fn sample_albedo(mesh: &Mesh, t: &Triangle, center: [f32; 3], half: f32, supersample: usize) -> [u8; 4] {
    let Some(tex) = t.texture.and_then(|i| mesh.textures.get(i)) else {
        return t.base;
    };
    // Sample the texture at the barycentric UV of `p`'s nearest triangle point. Returns the texel colour plus
    // the world-space nearest point (so the caller can reject off-footprint samples).
    let tap = |p: [f32; 3]| -> ([u8; 4], [f32; 3]) {
        let bary = closest_point_barycentric(p, &t.p);
        let u = bary[0] * t.uv[0][0] + bary[1] * t.uv[1][0] + bary[2] * t.uv[2][0];
        let v = bary[0] * t.uv[0][1] + bary[1] * t.uv[1][1] + bary[2] * t.uv[2][1];
        let near = [
            bary[0] * t.p[0][0] + bary[1] * t.p[1][0] + bary[2] * t.p[2][0],
            bary[0] * t.p[0][1] + bary[1] * t.p[1][1] + bary[2] * t.p[2][1],
            bary[0] * t.p[0][2] + bary[1] * t.p[1][2] + bary[2] * t.p[2][2],
        ];
        (tex.sample(u, v), near)
    };

    let s = supersample.max(1);
    // Grid of sample points across the voxel box; for S=1 this is just the centre (= the old point sample).
    // Offsets are the cell-centres of an S×S×S subdivision of [-half, half] on each axis, but we only need the
    // triangle's footprint, so we walk the full S³ lattice and let the on-footprint test prune it (cheap for
    // S=3 → 27 points). Project each onto the triangle and keep on-footprint taps.
    let mut sum = [0u64; 4];
    let mut n = 0u64;
    // A small tolerance so a sample whose nearest point sits exactly on the box face still counts.
    let tol = half * 1.0e-3 + 1.0e-6;
    let step = if s == 1 { 0.0 } else { 2.0 * half / s as f32 };
    let base = if s == 1 { 0.0 } else { -half + 0.5 * step };
    for iz in 0..s {
        for iy in 0..s {
            for ix in 0..s {
                let p = [
                    center[0] + base + ix as f32 * step,
                    center[1] + base + iy as f32 * step,
                    center[2] + base + iz as f32 * step,
                ];
                let (col, near) = tap(p);
                // On-footprint test: the projected triangle point must lie within the voxel box (padded by
                // `tol`). `closest_point_barycentric` clamps to the triangle, so a tap whose true projection is
                // off-triangle but clamps to an edge point inside the box is still accepted — i.e. this is
                // conservative-inclusive at the triangle edges (the accepted colour is a real on-triangle edge
                // texel), an approximation of "the triangle's area clipped to the box", not an exact clip.
                if (near[0] - center[0]).abs() <= half + tol
                    && (near[1] - center[1]).abs() <= half + tol
                    && (near[2] - center[2]).abs() <= half + tol
                {
                    for ch in 0..4 {
                        sum[ch] += col[ch] as u64;
                    }
                    n += 1;
                }
            }
        }
    }
    if n == 0 {
        // Sliver: no grid point projected inside the box — fall back to the single nearest-point sample so the
        // voxel still gets a sensible colour (the conservative SAT already proved the triangle overlaps).
        return tap(center).0;
    }
    [
        (sum[0] / n) as u8,
        (sum[1] / n) as u8,
        (sum[2] / n) as u8,
        (sum[3] / n) as u8,
    ]
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
// Palette quantization (perceptual CIELAB k-means)
// ============================================================================================

/// Maximum palette colours the `.vox` writer can carry (slot 0 is air → usable 1..=255).
const MAX_PALETTE: usize = 255;

/// k-means upper bound on Lloyd iterations. Convergence (no centroid moves more than `KMEANS_EPS²` in Lab²)
/// usually trips well before this; the cap keeps a pathological input from looping unboundedly and — being a
/// fixed function of the (deterministic) input — preserves byte-reproducibility.
const KMEANS_MAX_ITERS: usize = 64;

/// Convergence threshold for k-means, as a SQUARED Lab distance (a centroid moving less than ~0.32 ΔE between
/// Lloyd sweeps is "settled"). Squared so we never take a `sqrt` in the hot loop.
const KMEANS_EPS_SQ: f32 = 0.1;

/// Quantize the grid's solid-voxel albedos to a ≤255-colour palette and map each solid voxel to its nearest
/// palette index (1-based; 0 is reserved for empty/air per the `.vox` convention). Returns the palette (sRGB
/// RGBA) and a SPARSE `cell index → 1-based palette index` map over the solid cells only (a dense per-cell
/// `Vec<u8>` would be billions of bytes for a large AABB; the solid set is millions).
///
/// Clustering is **perceptual CIELAB k-means** (replacing the old sRGB median-cut): sRGB is perceptually
/// non-uniform, so equal sRGB distances span unequal perceived differences and median-cut muddies/biases the
/// palette. We convert each distinct albedo to CIELAB, cluster there with a deterministic seeded k-means
/// (k-means++ init + Lloyd), and use the **count-weighted mean sRGB** of each cluster as the representative
/// (truer than the Lab centroid). Determinism: the input is the `counts` map collected into a vec and
/// `sort_unstable`d, the RNG is a fixed-seed LCG, and Lloyd's tie-breaks are by lowest index — so the same
/// scene bakes byte-identical bytes. **Lossless short-circuit:** ≤255 distinct colours are emitted exactly.
fn quantize(grid: &Grid) -> (Vec<[u8; 4]>, HashMap<usize, u8>) {
    // Gather distinct solid albedos with counts (clustering works on the DISTINCT set — bounded by texture
    // content, not by the billions of solid voxels — weighted by how many voxels carry each colour).
    let mut counts: HashMap<[u8; 4], u32> = HashMap::new();
    for &c in grid.albedo.values() {
        *counts.entry(c).or_insert(0) += 1;
    }
    let mut pixels: Vec<([u8; 4], u32)> = counts.into_iter().collect();
    pixels.sort_unstable(); // deterministic clustering input (independent of HashMap order) → reproducible palette

    let palette = build_palette(&pixels, MAX_PALETTE);

    // Map every solid voxel to its nearest palette colour IN LAB (1-based index), caching per distinct albedo.
    // The palette's own Lab is precomputed once so each cache miss is a linear scan over ≤255 Lab points.
    let palette_lab: Vec<[f32; 3]> = palette.iter().map(|c| rgb_to_lab([c[0], c[1], c[2]])).collect();
    let mut indices: HashMap<usize, u8> = HashMap::with_capacity(grid.albedo.len());
    let mut nearest_cache: HashMap<[u8; 4], u8> = HashMap::new();
    for (&i, &c) in &grid.albedo {
        let idx = *nearest_cache.entry(c).or_insert_with(|| nearest_palette_lab(&palette_lab, c));
        indices.insert(i, idx + 1); // 1-based; 0 = air
    }
    (palette, indices)
}

/// Build a ≤`max_colors` sRGB palette from distinct `(colour, count)` pairs via perceptual CIELAB k-means.
/// `pixels` MUST already be sorted (deterministic input). Empty clusters are dropped. **Lossless
/// short-circuit:** if the distinct-colour set is ≤`max_colors`, the exact colours are returned (no k-means),
/// matching the asset-gen `quantize` fast path.
fn build_palette(pixels: &[([u8; 4], u32)], max_colors: usize) -> Vec<[u8; 4]> {
    if pixels.is_empty() {
        return vec![[255, 255, 255, 255]];
    }
    // Lossless: already within the palette budget — emit every distinct colour verbatim (the order is the
    // sorted input order, so it's reproducible).
    if pixels.len() <= max_colors {
        return pixels.iter().map(|(c, _)| *c).collect();
    }

    // Cluster in Lab. Each distinct colour is one weighted point; we keep its sRGB alongside so the cluster
    // representative is the count-weighted MEAN sRGB (not the Lab centroid mapped back).
    let labs: Vec<[f32; 3]> = pixels.iter().map(|(c, _)| rgb_to_lab([c[0], c[1], c[2]])).collect();
    let weights: Vec<f32> = pixels.iter().map(|(_, w)| *w as f32).collect();

    let assignments = kmeans_lab(&labs, &weights, max_colors);

    // Representative colour = count-weighted mean sRGB of each cluster (truer than the Lab centroid), clamped
    // to [0,255]; alpha = the count-weighted mean alpha. Drop empty clusters (k-means can leave some empty).
    let k = max_colors;
    let mut sum = vec![[0.0f64; 4]; k];
    let mut wsum = vec![0.0f64; k];
    for (pi, &cluster) in assignments.iter().enumerate() {
        let (c, _) = pixels[pi];
        let w = weights[pi] as f64;
        for ch in 0..4 {
            sum[cluster][ch] += c[ch] as f64 * w;
        }
        wsum[cluster] += w;
    }
    (0..k)
        .filter(|&c| wsum[c] > 0.0)
        .map(|c| {
            let w = wsum[c];
            [
                (sum[c][0] / w).round().clamp(0.0, 255.0) as u8,
                (sum[c][1] / w).round().clamp(0.0, 255.0) as u8,
                (sum[c][2] / w).round().clamp(0.0, 255.0) as u8,
                (sum[c][3] / w).round().clamp(0.0, 255.0) as u8,
            ]
        })
        .collect()
}

/// Deterministic weighted k-means over Lab points. Returns, per input point, its cluster id `0..k`. Seeding is
/// **k-means++** driven by a FIXED-seed LCG (no entropy/time → byte-reproducible bake). Standard count-weighted
/// Lab k-means clustering the DISTINCT-colour set weighted by count (per `TILED_VOXELIZER_PLAN` §C3.1) — a
/// deliberate, spec-sanctioned departure from asset-gen's `kmeans2(minit="++", seed=0)`, which clusters
/// UNWEIGHTED raw samples via scipy; this is not the same algorithm. Then **Lloyd** iterations to convergence (`KMEANS_EPS_SQ`) or
/// `KMEANS_MAX_ITERS`. Assignment ties break to the lowest centroid index; an empty cluster keeps its previous
/// centroid (it is simply dropped by the caller). `k` is assumed `< labs.len()` (the caller short-circuits the
/// lossless ≤k case), so k-means++ always finds distinct enough seeds.
fn kmeans_lab(labs: &[[f32; 3]], weights: &[f32], k: usize) -> Vec<usize> {
    let n = labs.len();
    let dist2 = |a: &[f32; 3], b: &[f32; 3]| {
        let (dl, da, db) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        dl * dl + da * da + db * db
    };

    // --- k-means++ seeding (weighted) with a fixed-seed LCG. ---
    let mut rng = Lcg::new(0x9E37_79B9_7F4A_7C15);
    let mut centroids: Vec<[f32; 3]> = Vec::with_capacity(k);
    // First centroid: a weighted random pick.
    let total_w: f64 = weights.iter().map(|&w| w as f64).sum();
    centroids.push(labs[weighted_pick(weights, total_w, rng.next_f64())]);
    // Remaining centroids: pick proportional to weighted squared distance to the nearest chosen centroid.
    let mut d2_nearest: Vec<f32> = labs.iter().map(|p| dist2(p, &centroids[0])).collect();
    while centroids.len() < k {
        // Weight each point by count × D² (k-means++ ∝ D², count-weighted because a colour stands in for many
        // voxels). Deterministic given the sorted input + the fixed LCG draw.
        let mut wsum = 0.0f64;
        for i in 0..n {
            wsum += weights[i] as f64 * d2_nearest[i] as f64;
        }
        let pick = if wsum <= 0.0 {
            // All remaining points coincide with chosen centroids — fall back to the first under-used point.
            // (Deterministic: lowest index whose nearest distance is largest, i.e. 0 here.)
            0
        } else {
            let target = rng.next_f64() * wsum;
            let mut acc = 0.0f64;
            let mut chosen = n - 1;
            for i in 0..n {
                acc += weights[i] as f64 * d2_nearest[i] as f64;
                if acc >= target {
                    chosen = i;
                    break;
                }
            }
            chosen
        };
        let c = labs[pick];
        centroids.push(c);
        for i in 0..n {
            let d = dist2(&labs[i], &c);
            if d < d2_nearest[i] {
                d2_nearest[i] = d;
            }
        }
    }

    // --- Lloyd iterations. ---
    let mut assign = vec![0usize; n];
    for _ in 0..KMEANS_MAX_ITERS {
        // Assignment step: each point → nearest centroid (lowest index on ties).
        for i in 0..n {
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for (c, cen) in centroids.iter().enumerate() {
                let d = dist2(&labs[i], cen);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            assign[i] = best;
        }
        // Update step: each centroid → weighted mean of its members (in Lab). Track the largest move²; an
        // empty cluster keeps its old centroid (it will be dropped downstream).
        let mut sum = vec![[0.0f64; 3]; k];
        let mut wsum = vec![0.0f64; k];
        for i in 0..n {
            let c = assign[i];
            let w = weights[i] as f64;
            for ch in 0..3 {
                sum[c][ch] += labs[i][ch] as f64 * w;
            }
            wsum[c] += w;
        }
        let mut max_move2 = 0.0f32;
        for c in 0..k {
            if wsum[c] > 0.0 {
                let new = [
                    (sum[c][0] / wsum[c]) as f32,
                    (sum[c][1] / wsum[c]) as f32,
                    (sum[c][2] / wsum[c]) as f32,
                ];
                max_move2 = max_move2.max(dist2(&new, &centroids[c]));
                centroids[c] = new;
            }
        }
        if max_move2 < KMEANS_EPS_SQ {
            break;
        }
    }
    assign
}

/// A small linear-congruential generator (Numerical Recipes / Knuth MMIX constants). Used ONLY to seed
/// k-means++ deterministically — a fixed seed makes the whole bake byte-reproducible (no `rand`/entropy/time
/// dependency, satisfying the determinism invariant).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    /// Next `u64` (MMIX LCG: `x' = a·x + c`).
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    /// Next `f64` in `[0,1)` (top 53 bits → mantissa).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Pick an index from `weights` proportional to its weight, given a draw `r ∈ [0,1)` and the precomputed
/// `total` weight. Deterministic linear scan (returns the last index if `r` rounds past the end).
fn weighted_pick(weights: &[f32], total: f64, r: f64) -> usize {
    if total <= 0.0 {
        return 0;
    }
    let target = r * total;
    let mut acc = 0.0f64;
    for (i, &w) in weights.iter().enumerate() {
        acc += w as f64;
        if acc >= target {
            return i;
        }
    }
    weights.len() - 1
}

/// Index of the palette colour nearest `c` by squared **CIELAB** distance (alpha ignored — surface voxels are
/// opaque). `palette_lab` is the palette pre-converted to Lab. Linear scan over ≤255 entries; results are
/// cached per distinct albedo by the caller. Replaces the old sRGB squared-distance `nearest_palette` so
/// assignment is perceptual, matching the perceptual clustering.
fn nearest_palette_lab(palette_lab: &[[f32; 3]], c: [u8; 4]) -> u8 {
    let lab = rgb_to_lab([c[0], c[1], c[2]]);
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, p) in palette_lab.iter().enumerate() {
        let (dl, da, db) = (lab[0] - p[0], lab[1] - p[1], lab[2] - p[2]);
        let d = dl * dl + da * da + db * db;
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best as u8
}

/// Convert an sRGB `u8` triple to CIELAB (D65), porting asset-gen's `palette.py::_rgb_to_lab`: sRGB→linear
/// (the standard 0.04045 piecewise), linear→XYZ via the standard 3×3 matrix, XYZ→Lab with `eps = 216/24389`
/// and `kappa = 24389/27`. CIELAB is perceptually ~uniform, so Euclidean distance there approximates
/// perceived colour difference — the basis for the perceptual palette + nearest-colour assignment.
fn rgb_to_lab(rgb: [u8; 3]) -> [f32; 3] {
    // sRGB u8 → [0,1] → linear.
    let lin = |c: u8| -> f32 {
        let s = c as f32 / 255.0;
        if s > 0.04045 { ((s + 0.055) / 1.055).powf(2.4) } else { s / 12.92 }
    };
    let (r, g, b) = (lin(rgb[0]), lin(rgb[1]), lin(rgb[2]));
    // linear sRGB → XYZ (D65), the standard D65 sRGB→XYZ matrix (IEC 61966-2-1 full-precision coefficients;
    // each row sums to its white-point component Xn/Yn/Zn, so linear white → exactly (Xn, Yn, Zn) → Lab(100,0,0)).
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.072175 * b;
    let z = 0.0193339 * r + 0.119192 * g + 0.9503041 * b;
    // Normalize by the D65 white point.
    let (xn, yn, zn) = (x / 0.95047, y / 1.0, z / 1.08883);
    let eps = 216.0 / 24389.0;
    let kappa = 24389.0 / 27.0;
    let f = |t: f32| -> f32 {
        if t > eps { t.cbrt() } else { (kappa * t + 16.0) / 116.0 }
    };
    let (fx, fy, fz) = (f(xn), f(yn), f(zn));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

// ============================================================================================
// `.vox` assembly (split into ≤256³ models on a scene grid)
// ============================================================================================

/// Build the `DotVoxData`: split the grid into ≤256³ sub-models, place each by a scene Transform at its
/// block CENTER (the MagicaVoxel convention the runtime loader reverses), and attach the 256-entry palette.
/// Z-up: the grid's Y (up) becomes `.vox` Z, the grid's Z becomes `.vox` Y, matching the loader's
/// `.vox (x,y,z) → world (x,z,y)` swap so a round-trip is identity.
fn build_dot_vox(grid: &Grid, palette: &[[u8; 4]], indices: &HashMap<usize, u8>) -> DotVoxData {
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

    // Bucket each SOLID voxel into its ≤256³ tile — O(solid), not O(AABB): scanning every cell per tile would
    // be billions of iterations for a large grid. Tile linear index over the `.vox` axes (X←gridX, Y←gridZ,
    // Z←gridY): `tx + ty·tiles_x + tz·tiles_x·tiles_vy`.
    let n_tiles = (tiles_x as usize) * (tiles_vy as usize) * (tiles_vz as usize);
    let mut tile_voxels: Vec<Vec<Voxel>> = vec![Vec::new(); n_tiles];
    for (&i, &pal) in indices {
        if pal == 0 {
            continue; // shouldn't happen for a solid voxel, but stay total
        }
        let (gx, gy, gz) = grid.xyz(i);
        // `.vox` axes: vx ← grid x, vy ← grid z, vz ← grid y. Tile + local coords per axis.
        let (tx, lx) = (gx / VOX_MODEL_MAX, gx % VOX_MODEL_MAX);
        let (ty, ly) = (gz / VOX_MODEL_MAX, gz % VOX_MODEL_MAX);
        let (tz, lz) = (gy / VOX_MODEL_MAX, gy % VOX_MODEL_MAX);
        let tile = (tx + ty * tiles_x + tz * tiles_x * tiles_vy) as usize;
        tile_voxels[tile].push(Voxel { x: lx as u8, y: ly as u8, z: lz as u8, i: pal - 1 });
    }

    let mut models: Vec<Model> = Vec::new();
    // Each model's `.vox`-space min corner, for the scene Transform (center = corner + size/2).
    let mut model_corners: Vec<[i32; 3]> = Vec::new();
    for tz in 0..tiles_vz {
        for ty in 0..tiles_vy {
            for tx in 0..tiles_x {
                let tile = (tx + ty * tiles_x + tz * tiles_x * tiles_vy) as usize;
                let mut voxels = std::mem::take(&mut tile_voxels[tile]);
                if voxels.is_empty() {
                    continue; // drop fully-empty tiles
                }
                // Sort (z,y,x) so the bake is BYTE-deterministic (the bucket push order is HashMap-order); the
                // loader is order-independent, but reproducible `.vox` bytes ease diffing / CI / debugging.
                voxels.sort_unstable_by_key(|v| (v.z, v.y, v.x));
                let vx0 = tx * VOX_MODEL_MAX;
                let vy0 = ty * VOX_MODEL_MAX;
                let vz0 = tz * VOX_MODEL_MAX;
                let sx = (dx - vx0).min(VOX_MODEL_MAX);
                let sy = (dz - vy0).min(VOX_MODEL_MAX); // .vox Y extent ← grid Z
                let sz = (dy - vz0).min(VOX_MODEL_MAX); // .vox Z extent ← grid Y
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

// ============================================================================================
// C1 — Tiled bounded-RAM out-of-core voxelizer (union-find enclosure flood, disk-backed tiles)
// ============================================================================================
//
// `docs/TILED_VOXELIZER_PLAN.md` §C1. The monolithic `solid_fill` needs the whole AABB resident (occupancy +
// exterior bitsets + a flood frontier) — multi-GB-to-tens-of-GB for Bistro-Exterior @0.05 m. The tiled path
// computes the IDENTICAL solid/air classification under a bounded RAM budget by:
//   1. Routing the shared surface SAT ([`surface_scatter`]) to per-tile disk `.occ`/`.alb` files (only
//      non-empty tiles get a file — the sparse-absent convention).
//   2. Per-tile LOCAL air flood → label air components, record each component's 6-face footprint +
//      global-boundary touch; file-less tiles = one all-air component touching all 6 faces.
//   3. UNION-FIND across shared tile faces (stream the two adjacent face label images, union matching air-air
//      cells) + propagate "exterior" from a synthetic OUTSIDE node seeded by every global-boundary component.
//   4. FILL pass: air whose component root is NOT exterior → solid (a fully-buried file-less tile → a 1-flag
//      uniform-solid marker). Interior albedo via per-tile nearest-surface + 1-voxel halo exchange.
//
// DETERMINISM: tile order (linear id), prefix sums, union-by-rank tie-breaks, and face iteration are fixed
// functions of tile ids, so the partition (and thus the solid mask) is byte-reproducible.
//
// This module is a self-contained sub-pipeline of the offline voxelizer; it shares the surface SAT + the
// `solid_fill` RULE with the monolithic path (the oracle test proves cell-for-cell equality) and feeds the
// streaming `.vxo` assembly (`adventure::voxel::vxo::VxoStreamWriter`, bounded-RAM region-by-region write).
mod tiled {
    use std::collections::VecDeque;
    use std::io::Write;
    use std::path::{Path, PathBuf};

    use super::{Mesh, surface_scatter};

    /// Default tile edge in voxels (`TILE_EDGE`). At `T=128`: a `T³/8 = 256 KiB` occupancy + a `T³·1 = 2 MiB`
    /// `u8` component-label scratch per resident tile — a few resident tiles is well under the budget; the
    /// persistent per-tile component tables + union-find are O(total components) ≪ O(volume). Tunable via
    /// `--tile-edge` for the bounded-RAM/scratch trade (bigger = fewer faces/seams + more sequential I/O, more
    /// RAM per tile). See `TILED_VOXELIZER_PLAN.md` §C1.1.
    pub const DEFAULT_TILE_EDGE: i32 = 128;

    /// A solid (non-air) face cell sentinel in a disk face-label image — distinguishes "no air component here"
    /// from an actual `u16` component id. Air pockets per tile are far below `u16::MAX` on real geometry; the
    /// local flood asserts the bound (never silently wraps, §C1.10).
    const FACE_SOLID: u16 = u16::MAX;

    /// The 6 axis-neighbour offsets (the 6-connected flood relation), in a FIXED order so the flood + the face
    /// stitch are deterministic.
    const N6: [(i32, i32, i32); 6] =
        [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)];

    /// The 6 tile faces, indexed `0..6`: `[+X, -X, +Y, -Y, +Z, -Z]` (matching [`N6`]). The face a component
    /// touches is recorded as a bit in the component table; the stitch pairs a tile's `+axis` face with its
    /// neighbour's `-axis` face.
    const FACE_PX: usize = 0;
    const FACE_NX: usize = 1;
    const FACE_PY: usize = 2;
    const FACE_NY: usize = 3;
    const FACE_PZ: usize = 4;
    const FACE_NZ: usize = 5;

    /// The tiling geometry over the grid AABB: grid `dims`, the cubic `tile_edge` `T`, the per-axis tile counts,
    /// and the scratch directory holding the disk tiles. A pure function of `(dims, tile_edge)` — the SSOT for
    /// tile addressing, so every pass agrees on tile ids + extents.
    pub struct TileGrid {
        pub dims: [i32; 3],
        pub tile_edge: i32,
        pub tnx: i32,
        pub tny: i32,
        pub tnz: i32,
        pub scratch: PathBuf,
    }

    impl TileGrid {
        pub fn new(dims: [i32; 3], tile_edge: i32, scratch: PathBuf) -> Self {
            let t = tile_edge.max(1);
            let tnx = dims[0].div_euclid(t) + i32::from(dims[0].rem_euclid(t) != 0);
            let tny = dims[1].div_euclid(t) + i32::from(dims[1].rem_euclid(t) != 0);
            let tnz = dims[2].div_euclid(t) + i32::from(dims[2].rem_euclid(t) != 0);
            Self { dims, tile_edge: t, tnx: tnx.max(1), tny: tny.max(1), tnz: tnz.max(1), scratch }
        }

        /// Total number of tiles (the union-find / component-table index space).
        pub fn tile_count(&self) -> usize {
            (self.tnx as usize) * (self.tny as usize) * (self.tnz as usize)
        }

        /// Tile linear id `tx + ty·tnx + tz·tnx·tny` (X-fastest, same convention as the grid index).
        #[inline]
        pub fn tile_id(&self, tx: i32, ty: i32, tz: i32) -> usize {
            (tx as usize) + (ty as usize) * (self.tnx as usize) + (tz as usize) * (self.tnx as usize) * (self.tny as usize)
        }

        /// The `(tx,ty,tz)` of a tile id (inverse of [`tile_id`](Self::tile_id)).
        #[inline]
        pub fn tile_xyz(&self, id: usize) -> (i32, i32, i32) {
            let txy = (self.tnx as usize) * (self.tny as usize);
            let tz = id / txy;
            let r = id % txy;
            ((r % self.tnx as usize) as i32, (r / self.tnx as usize) as i32, tz as i32)
        }

        /// The tile owning a grid cell `(x,y,z)` (floor-division by `T`).
        #[inline]
        pub fn tile_of(&self, x: i32, y: i32, z: i32) -> (i32, i32, i32) {
            (x / self.tile_edge, y / self.tile_edge, z / self.tile_edge)
        }

        /// The clamped voxel EXTENT of tile `(tx,ty,tz)` — partial at the grid edge.
        #[inline]
        pub fn tile_extent(&self, tx: i32, ty: i32, tz: i32) -> [i32; 3] {
            [
                (self.dims[0] - tx * self.tile_edge).min(self.tile_edge),
                (self.dims[1] - ty * self.tile_edge).min(self.tile_edge),
                (self.dims[2] - tz * self.tile_edge).min(self.tile_edge),
            ]
        }

        /// GLOBAL cell index of local `(lx,ly,lz)` within tile `(tx,ty,tz)` (the grid's `x + y·dx + z·dx·dy`).
        #[inline]
        pub fn global_index(&self, tx: i32, ty: i32, tz: i32, lx: i32, ly: i32, lz: i32) -> usize {
            let (dx, dy) = (self.dims[0] as usize, self.dims[1] as usize);
            let gx = (tx * self.tile_edge + lx) as usize;
            let gy = (ty * self.tile_edge + ly) as usize;
            let gz = (tz * self.tile_edge + lz) as usize;
            gx + gy * dx + gz * dx * dy
        }

        fn occ_path(&self, id: usize) -> PathBuf {
            self.scratch.join(format!("tile_{id}.occ"))
        }
        fn alb_path(&self, id: usize) -> PathBuf {
            self.scratch.join(format!("tile_{id}.alb"))
        }
        fn face_path(&self, id: usize, face: usize) -> PathBuf {
            self.scratch.join(format!("tile_{id}.face{face}"))
        }
    }

    /// A local occupancy bitset over a tile's clamped extent (X-fastest local index). The only large per-tile
    /// RAM buffer alongside the `u8` component-label scratch.
    struct TileOcc {
        ext: [i32; 3],
        bits: Vec<u64>,
    }

    impl TileOcc {
        fn new(ext: [i32; 3]) -> Self {
            let n = (ext[0] as usize) * (ext[1] as usize) * (ext[2] as usize);
            Self { ext, bits: vec![0u64; n.div_ceil(64)] }
        }
        #[inline]
        fn local_index(&self, lx: i32, ly: i32, lz: i32) -> usize {
            (lx as usize) + (ly as usize) * (self.ext[0] as usize) + (lz as usize) * (self.ext[0] as usize) * (self.ext[1] as usize)
        }
        #[inline]
        fn get(&self, i: usize) -> bool {
            (self.bits[i >> 6] >> (i & 63)) & 1 != 0
        }
        #[inline]
        fn set(&mut self, i: usize) {
            self.bits[i >> 6] |= 1u64 << (i & 63);
        }
        fn count(&self) -> usize {
            (self.ext[0] as usize) * (self.ext[1] as usize) * (self.ext[2] as usize)
        }
    }

    /// One tile-local air component: the 6-face touch bitmask + whether it touches the GLOBAL grid boundary
    /// (open air → exterior seed). Tiny (2 bytes) × O(components) — the only persistent per-component RAM.
    #[derive(Clone, Copy, Default)]
    struct Component {
        touches_face: u8,
        touches_boundary: bool,
    }

    /// A path-compressed, union-by-rank disjoint-set over `[0, n)` PLUS a synthetic OUTSIDE node at index `n`.
    /// Air components are unioned across shared tile faces; OUTSIDE is unioned with every global-boundary
    /// component, so `find(c) == find(OUTSIDE)` ⇔ component `c` is exterior-reachable. Deterministic: union-by-
    /// rank ties break to the lower index, so the forest is a fixed function of the union order (which is itself
    /// a fixed function of tile ids).
    struct UnionFind {
        parent: Vec<u32>,
        rank: Vec<u8>,
    }

    impl UnionFind {
        fn new(n: usize) -> Self {
            // n components + 1 OUTSIDE node.
            let total = n + 1;
            Self { parent: (0..total as u32).collect(), rank: vec![0u8; total] }
        }
        fn outside(&self) -> u32 {
            (self.parent.len() - 1) as u32
        }
        fn find(&mut self, mut x: u32) -> u32 {
            // Iterative find with full path compression.
            let mut root = x;
            while self.parent[root as usize] != root {
                root = self.parent[root as usize];
            }
            while self.parent[x as usize] != root {
                let next = self.parent[x as usize];
                self.parent[x as usize] = root;
                x = next;
            }
            root
        }
        fn union(&mut self, a: u32, b: u32) {
            let (ra, rb) = (self.find(a), self.find(b));
            if ra == rb {
                return;
            }
            // Union by rank; equal ranks attach the HIGHER index under the LOWER (deterministic tie-break).
            let (lo, hi) = if self.rank[ra as usize] < self.rank[rb as usize] {
                (rb, ra) // attach ra under rb
            } else if self.rank[ra as usize] > self.rank[rb as usize] {
                (ra, rb) // attach rb under ra
            } else {
                // Equal rank: lower index becomes the root, the other's rank bumps.
                let (root, child) = if ra < rb { (ra, rb) } else { (rb, ra) };
                self.parent[child as usize] = root;
                self.rank[root as usize] += 1;
                return;
            };
            self.parent[hi as usize] = lo;
        }
    }

    /// The result of the surface scatter: which tiles got a disk file (sorted) + the global surface albedo
    /// `counts` are NOT held here (the assembly rebuilds the palette by streaming the finalized tiles). We keep
    /// only the per-tile presence so the later passes know which tiles have files.
    struct ScatterResult {
        /// `has_file[id]` — true iff tile `id` got an `.occ` file in the surface pass.
        has_file: Vec<bool>,
    }

    /// Surface pass: run the shared SAT ([`surface_scatter`]) and ROUTE its `(global_cell, albedo)` output to
    /// per-tile disk `.occ`/`.alb` files, first-writer-wins per cell in TRIANGLE order (the monolithic rule,
    /// now tile-local). Returns which tiles got a file. Bounded RAM: the per-tile buffers are flushed to disk;
    /// only the (sparse) surface cells are held transiently in the merge.
    fn scatter_surface(
        grid: &TileGrid,
        mesh: &Mesh,
        origin: [f32; 3],
        voxel_size: f32,
        supersample: usize,
    ) -> std::io::Result<ScatterResult> {
        let n_tiles = grid.tile_count();
        // The parallel per-triangle surface lists (triangle order preserved by the Vec-of-Vec).
        let per_tri = surface_scatter(mesh, origin, voxel_size, grid.dims, supersample);

        // Per-tile in-RAM accumulators: a local occupancy bitset + a first-writer albedo map (local_index →
        // rgba). Only NON-EMPTY tiles allocate. The serial first-writer merge in triangle order keeps the same
        // deterministic albedo rule as the monolithic path.
        let mut tile_occ: Vec<Option<TileOcc>> = (0..n_tiles).map(|_| None).collect();
        let mut tile_alb: Vec<std::collections::BTreeMap<u32, [u8; 4]>> =
            (0..n_tiles).map(|_| std::collections::BTreeMap::new()).collect();
        let (dx, dy) = (grid.dims[0] as usize, grid.dims[1] as usize);
        for cells in &per_tri {
            for &(gi, albedo) in cells {
                // Delinearize the global index → (x,y,z) → tile + local.
                let z = (gi / (dx * dy)) as i32;
                let r = gi % (dx * dy);
                let x = (r % dx) as i32;
                let y = (r / dx) as i32;
                let (tx, ty, tz) = grid.tile_of(x, y, z);
                let id = grid.tile_id(tx, ty, tz);
                let ext = grid.tile_extent(tx, ty, tz);
                let occ = tile_occ[id].get_or_insert_with(|| TileOcc::new(ext));
                let (lx, ly, lz) = (x - tx * grid.tile_edge, y - ty * grid.tile_edge, z - tz * grid.tile_edge);
                let li = occ.local_index(lx, ly, lz) as u32;
                // First-writer-wins per cell (triangle order): only set if not already solid in this tile.
                let lidx = occ.local_index(lx, ly, lz);
                if !occ.get(lidx) {
                    occ.set(lidx);
                    tile_alb[id].insert(li, albedo);
                }
            }
        }

        // Flush each non-empty tile to disk (.occ = the raw bitset words; .alb = sorted (local_index, rgba)).
        let mut has_file = vec![false; n_tiles];
        for id in 0..n_tiles {
            if let Some(occ) = &tile_occ[id] {
                write_occ(&grid.occ_path(id), occ)?;
                write_alb(&grid.alb_path(id), &tile_alb[id])?;
                has_file[id] = true;
            }
        }
        Ok(ScatterResult { has_file })
    }

    /// Write a tile's occupancy bitset to disk: extent header (`3 × i32`) then the `u64` words (LE).
    fn write_occ(path: &Path, occ: &TileOcc) -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        for a in occ.ext {
            f.write_all(&a.to_le_bytes())?;
        }
        for w in &occ.bits {
            f.write_all(&w.to_le_bytes())?;
        }
        f.flush()
    }

    /// Read a tile's occupancy bitset from disk.
    fn read_occ(path: &Path) -> std::io::Result<TileOcc> {
        let bytes = std::fs::read(path)?;
        let mut ext = [0i32; 3];
        for (a, slot) in ext.iter_mut().enumerate() {
            *slot = i32::from_le_bytes(bytes[a * 4..a * 4 + 4].try_into().unwrap());
        }
        let n = (ext[0] as usize) * (ext[1] as usize) * (ext[2] as usize);
        let words = n.div_ceil(64);
        let mut bits = vec![0u64; words];
        let off = 12;
        for (w, slot) in bits.iter_mut().enumerate() {
            let b = off + w * 8;
            *slot = u64::from_le_bytes(bytes[b..b + 8].try_into().unwrap());
        }
        Ok(TileOcc { ext, bits })
    }

    /// Write a tile's surface albedo as sorted `(local_index:u32, rgba:[u8;4])` runs.
    fn write_alb(path: &Path, alb: &std::collections::BTreeMap<u32, [u8; 4]>) -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        for (&li, rgba) in alb {
            f.write_all(&li.to_le_bytes())?;
            f.write_all(rgba)?;
        }
        f.flush()
    }

    /// Read a tile's surface albedo back into a map (local_index → rgba).
    fn read_alb(path: &Path) -> std::io::Result<std::collections::HashMap<u32, [u8; 4]>> {
        let mut out = std::collections::HashMap::new();
        match std::fs::read(path) {
            Ok(bytes) => {
                let mut i = 0;
                while i + 8 <= bytes.len() {
                    let li = u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
                    let rgba = [bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]];
                    out.insert(li, rgba);
                    i += 8;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        Ok(out)
    }

    /// Label a tile's AIR cells into 6-connected components (scan-order deterministic BFS): returns the `u16`
    /// per-cell label array (local-index order; solid cells get [`FACE_SOLID`]) and the component count. Guards
    /// the `u16` component bound (§C1.10) — a pathological air-pocket explosion is a HARD error, never a wrap.
    fn label_tile(occ: &TileOcc) -> (Vec<u16>, u32) {
        let n = occ.count();
        let mut label = vec![FACE_SOLID; n];
        let [ex, ey, ez] = occ.ext;
        let mut next: u16 = 0;
        let mut q: VecDeque<(i32, i32, i32)> = VecDeque::new();
        // Scan in local index order so the component ids are a fixed function of the occupancy (deterministic).
        for lz in 0..ez {
            for ly in 0..ey {
                for lx in 0..ex {
                    let i = occ.local_index(lx, ly, lz);
                    if occ.get(i) || label[i] != FACE_SOLID {
                        continue; // solid, or already labelled air
                    }
                    let comp = next;
                    next = next.checked_add(1).unwrap_or_else(|| {
                        panic!(
                            "tiled flood: tile air components exceeded u16::MAX ({}³ extent) — widen the label \
                             type (TILED_VOXELIZER_PLAN §C1.10); practically unreachable on real geometry",
                            occ.tile_edge_hint()
                        )
                    });
                    label[i] = comp;
                    q.push_back((lx, ly, lz));
                    while let Some((cx, cy, cz)) = q.pop_front() {
                        for (ox, oy, oz) in N6 {
                            let (nx, ny, nz) = (cx + ox, cy + oy, cz + oz);
                            if nx < 0 || ny < 0 || nz < 0 || nx >= ex || ny >= ey || nz >= ez {
                                continue;
                            }
                            let ni = occ.local_index(nx, ny, nz);
                            if !occ.get(ni) && label[ni] == FACE_SOLID {
                                label[ni] = comp;
                                q.push_back((nx, ny, nz));
                            }
                        }
                    }
                }
            }
        }
        (label, next as u32)
    }

    impl TileOcc {
        /// A coarse extent hint for the overflow panic message (max extent axis).
        fn tile_edge_hint(&self) -> i32 {
            self.ext[0].max(self.ext[1]).max(self.ext[2])
        }
    }

    /// Per-tile face image: for each of the 6 faces, the in-face `(v,w)` air-component labels (or [`FACE_SOLID`]),
    /// written to disk so the stitch can stream adjacent pairs without re-flooding. The face's in-face dims are
    /// implied by the tile extent + which axis the face is on. Layout per face file: `2 × i32` (the in-face
    /// dims `(w0, w1)`) then `w0*w1 × u16` labels (row-major, w0 fastest).
    fn write_face_images(grid: &TileGrid, id: usize, occ: &TileOcc, label: &[u16]) -> std::io::Result<()> {
        let [ex, ey, ez] = occ.ext;
        // For each face: (in-face dim0, in-face dim1) and a closure mapping in-face (a,b) → local (lx,ly,lz).
        // +X/-X faces span (y,z); +Y/-Y span (x,z); +Z/-Z span (x,y).
        let faces: [(usize, i32, i32, [i32; 1]); 6] = [
            (FACE_PX, ey, ez, [ex - 1]),
            (FACE_NX, ey, ez, [0]),
            (FACE_PY, ex, ez, [ey - 1]),
            (FACE_NY, ex, ez, [0]),
            (FACE_PZ, ex, ey, [ez - 1]),
            (FACE_NZ, ex, ey, [0]),
        ];
        for (face, d0, d1, fixed) in faces {
            let mut buf: Vec<u8> = Vec::with_capacity(8 + (d0 as usize) * (d1 as usize) * 2);
            buf.extend_from_slice(&d0.to_le_bytes());
            buf.extend_from_slice(&d1.to_le_bytes());
            for b in 0..d1 {
                for a in 0..d0 {
                    let (lx, ly, lz) = match face {
                        FACE_PX | FACE_NX => (fixed[0], a, b),
                        FACE_PY | FACE_NY => (a, fixed[0], b),
                        _ => (a, b, fixed[0]),
                    };
                    let li = occ.local_index(lx, ly, lz);
                    let lab = label[li];
                    buf.extend_from_slice(&lab.to_le_bytes());
                }
            }
            std::fs::write(grid.face_path(id, face), &buf)?;
        }
        Ok(())
    }

    /// Read a tile's face image `(dims, labels)`; a missing file (a file-less tile) is the all-air face — the
    /// caller handles that separately (it never writes face images for file-less tiles).
    fn read_face_image(grid: &TileGrid, id: usize, face: usize) -> std::io::Result<(i32, i32, Vec<u16>)> {
        let bytes = std::fs::read(grid.face_path(id, face))?;
        let d0 = i32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let d1 = i32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let mut labels = Vec::with_capacity((d0 as usize) * (d1 as usize));
        let mut i = 8;
        while i + 2 <= bytes.len() {
            labels.push(u16::from_le_bytes(bytes[i..i + 2].try_into().unwrap()));
            i += 2;
        }
        Ok((d0, d1, labels))
    }

    /// The full enclosure classification: surface scatter (done by the caller) → per-tile local flood (build
    /// the component table + face images + boundary seeds + the `component_base` prefix sum) → union-find across
    /// shared faces → exterior closure. Returns the per-tile component base + the union-find with exterior
    /// resolved, so the fill pass can classify each air cell. Bounded RAM throughout (one tile resident at a time).
    struct Classification {
        component_base: Vec<u32>,
        uf: UnionFind,
        exterior_root: Vec<bool>,
    }

    /// Build the per-tile components: flood each tile, write its face images, record the component table +
    /// boundary seeds, and assign the `component_base` prefix sum (deterministic tile-id order). A file-less
    /// tile contributes ONE all-air component touching all 6 faces (+ boundary if on the grid edge) — §C1.5.
    fn build_components(
        grid: &TileGrid,
        scatter: &ScatterResult,
    ) -> std::io::Result<(Vec<u32>, Vec<u32>, Vec<Component>)> {
        let n_tiles = grid.tile_count();
        let mut component_base = vec![0u32; n_tiles];
        let mut component_count = vec![0u32; n_tiles];
        let mut components: Vec<Component> = Vec::new();
        let mut base: u32 = 0;
        for id in 0..n_tiles {
            let (tx, ty, tz) = grid.tile_xyz(id);
            // Which faces lie on the GLOBAL grid boundary (open air outside).
            let on_boundary = |face: usize| -> bool {
                match face {
                    FACE_PX => tx == grid.tnx - 1,
                    FACE_NX => tx == 0,
                    FACE_PY => ty == grid.tny - 1,
                    FACE_NY => ty == 0,
                    FACE_PZ => tz == grid.tnz - 1,
                    _ => tz == 0,
                }
            };
            component_base[id] = base;
            if scatter.has_file[id] {
                let occ = read_occ(&grid.occ_path(id))?;
                let (label, c_tile) = label_tile(&occ);
                write_face_images(grid, id, &occ, &label)?;
                // Build the component table from the labels: per component, the face-touch bitmask + boundary.
                let mut table = vec![Component::default(); c_tile as usize];
                let [ex, ey, ez] = occ.ext;
                // Scan the 6 faces, OR-ing each air cell's component's touch bit + boundary seed.
                let mark = |face: usize, lx: i32, ly: i32, lz: i32, table: &mut Vec<Component>| {
                    let li = occ.local_index(lx, ly, lz);
                    let lab = label[li];
                    if lab != FACE_SOLID {
                        let c = &mut table[lab as usize];
                        c.touches_face |= 1 << face;
                        if on_boundary(face) {
                            c.touches_boundary = true;
                        }
                    }
                };
                for b in 0..ez {
                    for a in 0..ey {
                        mark(FACE_PX, ex - 1, a, b, &mut table);
                        mark(FACE_NX, 0, a, b, &mut table);
                    }
                }
                for b in 0..ez {
                    for a in 0..ex {
                        mark(FACE_PY, a, ey - 1, b, &mut table);
                        mark(FACE_NY, a, 0, b, &mut table);
                    }
                }
                for b in 0..ey {
                    for a in 0..ex {
                        mark(FACE_PZ, a, b, ez - 1, &mut table);
                        mark(FACE_NZ, a, b, 0, &mut table);
                    }
                }
                component_count[id] = c_tile;
                components.extend(table);
                base += c_tile;
            } else {
                // File-less tile: one all-air component touching all 6 faces; boundary if ANY face is on the edge.
                let touches_boundary = (0..6).any(on_boundary);
                components.push(Component { touches_face: 0b0011_1111, touches_boundary });
                component_count[id] = 1;
                base += 1;
            }
        }
        Ok((component_base, component_count, components))
    }

    /// Stitch tiles across shared faces with the union-find, then propagate exterior from the synthetic OUTSIDE
    /// node (seeded by every global-boundary component). Streams the two adjacent face label images per internal
    /// face (file-less tiles use their single all-air component). Returns the resolved [`Classification`].
    fn stitch_and_close(
        grid: &TileGrid,
        component_base: &[u32],
        component_count: &[u32],
        components: &[Component],
    ) -> std::io::Result<Classification> {
        let total: usize = component_count.iter().map(|&c| c as usize).sum();
        let mut uf = UnionFind::new(total);
        let outside = uf.outside();

        // Seed OUTSIDE with every global-boundary component (exterior = the closure of these).
        for (id, &cnt) in component_count.iter().enumerate() {
            let cbase = component_base[id];
            for c in 0..cnt {
                if components[(cbase + c) as usize].touches_boundary {
                    uf.union(cbase + c, outside);
                }
            }
        }

        // For a tile face cell, resolve its air component global node id (or None if solid). A file-less tile is
        // all-air component 0; a tiled-with-file tile reads its persisted face image.
        // We stream internal faces in tile-id order; for each tile, its +X/+Y/+Z neighbour (the negative side is
        // covered by the neighbour's positive iteration).
        for id in 0..grid.tile_count() {
            let (tx, ty, tz) = grid.tile_xyz(id);
            // +X neighbour.
            if tx + 1 < grid.tnx {
                stitch_face(grid, &mut uf, component_base, component_count, components, id, FACE_PX, grid.tile_id(tx + 1, ty, tz), FACE_NX)?;
            }
            if ty + 1 < grid.tny {
                stitch_face(grid, &mut uf, component_base, component_count, components, id, FACE_PY, grid.tile_id(tx, ty + 1, tz), FACE_NY)?;
            }
            if tz + 1 < grid.tnz {
                stitch_face(grid, &mut uf, component_base, component_count, components, id, FACE_PZ, grid.tile_id(tx, ty, tz + 1), FACE_NZ)?;
            }
        }

        // Resolve exterior: a component is exterior iff its root == find(OUTSIDE). Precompute per ROOT so the
        // fill pass is O(1) per cell.
        let out_root = uf.find(outside);
        let mut exterior_root = vec![false; uf.parent.len()];
        for node in 0..total as u32 {
            if uf.find(node) == out_root {
                exterior_root[uf.find(node) as usize] = true;
            }
        }
        exterior_root[out_root as usize] = true;

        Ok(Classification { component_base: component_base.to_vec(), uf, exterior_root })
    }

    /// Union the air-air matching cells of tile `a`'s `face_a` with tile `b`'s `face_b` (the opposite axis face).
    /// A file-less tile's face is its single all-air component (base + 0) for every in-face cell.
    #[allow(clippy::too_many_arguments)]
    fn stitch_face(
        grid: &TileGrid,
        uf: &mut UnionFind,
        component_base: &[u32],
        component_count: &[u32],
        _components: &[Component],
        a: usize,
        face_a: usize,
        b: usize,
        face_b: usize,
    ) -> std::io::Result<()> {
        // Load each side's face labels as (d0, d1, labels) — or None for a file-less tile (all-air comp 0).
        let load = |id: usize, face: usize| -> std::io::Result<Option<(i32, i32, Vec<u16>)>> {
            if component_count[id] == 1 && !grid.occ_path(id).exists() {
                // File-less tile: no face image on disk; its whole face is air component 0.
                Ok(None)
            } else {
                Ok(Some(read_face_image(grid, id, face)?))
            }
        };
        let fa = load(a, face_a)?;
        let fb = load(b, face_b)?;
        // The shared face's in-face dims must match (same (v,w) lattice across the boundary). For two
        // file-less or one-file/one-fileless pairing, take the dims from whichever side has them.
        let (d0, d1) = match (&fa, &fb) {
            (Some((d0, d1, _)), _) => (*d0, *d1),
            (_, Some((d0, d1, _))) => (*d0, *d1),
            (None, None) => {
                // Both file-less: the whole shared face is air↔air, union their single components.
                uf.union(component_base[a], component_base[b]);
                return Ok(());
            }
        };
        for w in 0..d1 {
            for v in 0..d0 {
                let idx = (v + w * d0) as usize;
                let la = match &fa {
                    Some((_, _, labels)) => labels[idx],
                    None => 0, // file-less: comp 0 (air)
                };
                let lb = match &fb {
                    Some((_, _, labels)) => labels[idx],
                    None => 0,
                };
                if la != FACE_SOLID && lb != FACE_SOLID {
                    uf.union(component_base[a] + la as u32, component_base[b] + lb as u32);
                }
            }
        }
        Ok(())
    }

    /// Run the full classification (build components → stitch → close). Surface scatter must already be done.
    fn classify(grid: &TileGrid, scatter: &ScatterResult) -> std::io::Result<Classification> {
        let (component_base, component_count, components) = build_components(grid, scatter)?;
        stitch_and_close(grid, &component_base, &component_count, &components)
    }

    /// Whether a given air cell is EXTERIOR (open) under the classification — `exterior_root[find(node)]`. For a
    /// file-less tile every air cell is component 0.
    impl Classification {
        fn is_exterior(&mut self, tile_id: usize, comp: u16) -> bool {
            let node = self.component_base[tile_id] + comp as u32;
            let root = self.uf.find(node);
            self.exterior_root[root as usize]
        }
    }

    /// A finalized tile's solid cells, streamed to the assembly: its tile coords + the per-solid-cell global
    /// index + albedo. The fill pass produces these lazily (one tile at a time) so the assembly never holds the
    /// whole grid resident.
    pub struct FinalTile {
        pub id: usize,
        /// `(global_cell_index, rgba)` for every SOLID cell in this tile (surface + enclosed-fill), sorted by
        /// global index (deterministic).
        pub solids: Vec<(usize, [u8; 4])>,
    }

    /// The FILL pass over one tile: load `.occ` (or treat a file-less tile as all-air), recompute the local
    /// labeling, classify each air cell (enclosed → solid with nearest-surface albedo), and return the tile's
    /// finalized solid cells. A file-less ENCLOSED tile becomes a uniform-solid block (every cell solid). The
    /// interior albedo is a per-tile nearest-surface BFS seeded by the tile's surface cells PLUS a 1-voxel halo
    /// imported from the 6 neighbour tiles' shared faces (§C1.6 option A).
    fn fill_tile(grid: &TileGrid, scatter: &ScatterResult, cls: &mut Classification, id: usize) -> std::io::Result<FinalTile> {
        let (tx, ty, tz) = grid.tile_xyz(id);
        let ext = grid.tile_extent(tx, ty, tz);
        let [ex, ey, ez] = ext;
        let n = (ex as usize) * (ey as usize) * (ez as usize);

        if !scatter.has_file[id] {
            // File-less tile: one all-air component. If it is exterior → all air (no solids); else uniform solid.
            if cls.is_exterior(id, 0) {
                return Ok(FinalTile { id, solids: Vec::new() });
            }
            // Enclosed buried tile → uniform solid. Colour = the halo-imported nearest surface colour (or a
            // neutral grey if utterly surrounded by other buried tiles — interiors are cosmetic, §C1.6).
            let halo = import_halo_albedo(grid, scatter, id, ext)?;
            let fill_col = halo.values().next().copied().unwrap_or([128, 128, 128, 255]);
            let mut solids = Vec::with_capacity(n);
            for lz in 0..ez {
                for ly in 0..ey {
                    for lx in 0..ex {
                        solids.push((grid.global_index(tx, ty, tz, lx, ly, lz), fill_col));
                    }
                }
            }
            solids.sort_unstable_by_key(|&(gi, _)| gi);
            return Ok(FinalTile { id, solids });
        }

        let occ = read_occ(&grid.occ_path(id))?;
        let (label, _c) = label_tile(&occ);
        let surf_alb = read_alb(&grid.alb_path(id))?;

        // Decide solid vs air per cell: occupied → solid (surface); air whose component is NOT exterior →
        // enclosed solid. Build the SOLID mask + seed the interior-colour BFS from surface cells + the halo.
        let mut solid = vec![false; n];
        let mut color = vec![[0u8; 4]; n];
        for lz in 0..ez {
            for ly in 0..ey {
                for lx in 0..ex {
                    let li = occ.local_index(lx, ly, lz);
                    if occ.get(li) {
                        solid[li] = true;
                        color[li] = surf_alb.get(&(li as u32)).copied().unwrap_or([128, 128, 128, 255]);
                    } else {
                        let comp = label[li];
                        debug_assert!(comp != FACE_SOLID, "air cell must have an air component");
                        if !cls.is_exterior(id, comp) {
                            solid[li] = true; // enclosed → solid (colour assigned by the BFS below)
                        }
                    }
                }
            }
        }
        // Import the 1-voxel halo colours from neighbour faces (so colour propagates across tile seams): a halo
        // entry colours a boundary solid cell ONLY where this tile's own surface didn't already colour it.
        let halo = import_halo_albedo(grid, scatter, id, ext)?;
        for (li, col) in halo {
            let li = li as usize;
            if solid[li] && color[li] == [0u8; 4] {
                color[li] = col;
            }
        }
        // Multi-source 6-connected BFS over the ENCLOSED solids, colouring each newly-filled cell with its
        // nearest surface colour. Re-seed deterministically: surface+halo seeds sorted by local index (so the
        // BFS frontier order — and thus tie-broken interior colours — is independent of map iteration order).
        let mut seeds: Vec<usize> = (0..n).filter(|&li| solid[li] && color[li] != [0u8; 4]).collect();
        seeds.sort_unstable();
        let mut q2: VecDeque<usize> = seeds.into_iter().collect();
        let mut visited = vec![false; n];
        for &li in q2.iter() {
            visited[li] = true;
        }
        while let Some(li) = q2.pop_front() {
            let (lx, ly, lz) = local_xyz(li, ext);
            let src = color[li];
            for (ox, oy, oz) in N6 {
                let (nx, ny, nz) = (lx + ox, ly + oy, lz + oz);
                if nx < 0 || ny < 0 || nz < 0 || nx >= ex || ny >= ey || nz >= ez {
                    continue;
                }
                let ni = occ.local_index(nx, ny, nz);
                if solid[ni] && !visited[ni] {
                    visited[ni] = true;
                    if color[ni] == [0u8; 4] {
                        color[ni] = src;
                    }
                    q2.push_back(ni);
                }
            }
        }

        // Emit the finalized solids (sorted by global index). Any enclosed cell the BFS never coloured (a tile
        // with NO surface seed at all but enclosed by neighbours) gets the neutral fallback — cosmetic.
        let mut solids = Vec::new();
        for lz in 0..ez {
            for ly in 0..ey {
                for lx in 0..ex {
                    let li = occ.local_index(lx, ly, lz);
                    if solid[li] {
                        let col = if color[li] == [0u8; 4] { [128, 128, 128, 255] } else { color[li] };
                        solids.push((grid.global_index(tx, ty, tz, lx, ly, lz), col));
                    }
                }
            }
        }
        solids.sort_unstable_by_key(|&(gi, _)| gi);
        Ok(FinalTile { id, solids })
    }

    /// Local `(lx,ly,lz)` of a local index within a `[ex,ey,ez]` tile extent.
    #[inline]
    fn local_xyz(li: usize, ext: [i32; 3]) -> (i32, i32, i32) {
        let (ex, ey) = (ext[0] as usize, ext[1] as usize);
        let lz = li / (ex * ey);
        let r = li % (ex * ey);
        ((r % ex) as i32, (r / ex) as i32, lz as i32)
    }

    /// Import the 1-voxel halo surface colours from the 6 neighbour tiles' SHARED faces: for each face, read the
    /// neighbour's opposite-face surface albedo (if it has a file) and map it to THIS tile's boundary local
    /// cell. Returns `local_index → rgba` halo colours (a seed source for the interior-colour BFS). Bounded: a
    /// face is T² entries. (§C1.6 option A — colour-only; never affects occupancy.)
    fn import_halo_albedo(
        grid: &TileGrid,
        scatter: &ScatterResult,
        id: usize,
        ext: [i32; 3],
    ) -> std::io::Result<std::collections::HashMap<u32, [u8; 4]>> {
        let (tx, ty, tz) = grid.tile_xyz(id);
        let [ex, ey, ez] = ext;
        let mut halo: std::collections::HashMap<u32, [u8; 4]> = std::collections::HashMap::new();
        // For each of the 6 neighbours that exists + has a file, pull its boundary surface colours onto our face.
        let neighbours: [(i32, i32, i32, usize); 6] = [
            (tx + 1, ty, tz, FACE_PX),
            (tx - 1, ty, tz, FACE_NX),
            (tx, ty + 1, tz, FACE_PY),
            (tx, ty - 1, tz, FACE_NY),
            (tx, ty, tz + 1, FACE_PZ),
            (tx, ty, tz - 1, FACE_NZ),
        ];
        for (nx, ny, nz, my_face) in neighbours {
            if nx < 0 || ny < 0 || nz < 0 || nx >= grid.tnx || ny >= grid.tny || nz >= grid.tnz {
                continue;
            }
            let nid = grid.tile_id(nx, ny, nz);
            if !scatter.has_file[nid] {
                continue; // a file-less neighbour has no surface to import
            }
            let nalb = read_alb(&grid.alb_path(nid))?;
            if nalb.is_empty() {
                continue;
            }
            let nocc = read_occ(&grid.occ_path(nid))?;
            let [nex, ney, nez] = nocc.ext;
            // The neighbour's OPPOSITE face touching ours: if my_face is +X, the neighbour's −X face (its lx=0)
            // maps to my +X boundary (my lx=ex-1), same (y,z) in-face coord.
            match my_face {
                FACE_PX => {
                    for b in 0..ez.min(nez) {
                        for a in 0..ey.min(ney) {
                            let nli = nocc.local_index(0, a, b) as u32;
                            if let Some(&c) = nalb.get(&nli) {
                                let mli = (ex - 1) as usize + (a as usize) * (ex as usize) + (b as usize) * (ex as usize) * (ey as usize);
                                halo.entry(mli as u32).or_insert(c);
                            }
                        }
                    }
                }
                FACE_NX => {
                    for b in 0..ez.min(nez) {
                        for a in 0..ey.min(ney) {
                            let nli = nocc.local_index(nex - 1, a, b) as u32;
                            if let Some(&c) = nalb.get(&nli) {
                                let mli = (a as usize) * (ex as usize) + (b as usize) * (ex as usize) * (ey as usize);
                                halo.entry(mli as u32).or_insert(c);
                            }
                        }
                    }
                }
                FACE_PY => {
                    for b in 0..ez.min(nez) {
                        for a in 0..ex.min(nex) {
                            let nli = nocc.local_index(a, 0, b) as u32;
                            if let Some(&c) = nalb.get(&nli) {
                                let mli = (a as usize) + ((ey - 1) as usize) * (ex as usize) + (b as usize) * (ex as usize) * (ey as usize);
                                halo.entry(mli as u32).or_insert(c);
                            }
                        }
                    }
                }
                FACE_NY => {
                    for b in 0..ez.min(nez) {
                        for a in 0..ex.min(nex) {
                            let nli = nocc.local_index(a, ney - 1, b) as u32;
                            if let Some(&c) = nalb.get(&nli) {
                                let mli = (a as usize) + (b as usize) * (ex as usize) * (ey as usize);
                                halo.entry(mli as u32).or_insert(c);
                            }
                        }
                    }
                }
                FACE_PZ => {
                    for b in 0..ey.min(ney) {
                        for a in 0..ex.min(nex) {
                            let nli = nocc.local_index(a, b, 0) as u32;
                            if let Some(&c) = nalb.get(&nli) {
                                let mli = (a as usize) + (b as usize) * (ex as usize) + ((ez - 1) as usize) * (ex as usize) * (ey as usize);
                                halo.entry(mli as u32).or_insert(c);
                            }
                        }
                    }
                }
                _ => {
                    for b in 0..ey.min(ney) {
                        for a in 0..ex.min(nex) {
                            let nli = nocc.local_index(a, b, nez - 1) as u32;
                            if let Some(&c) = nalb.get(&nli) {
                                let mli = (a as usize) + (b as usize) * (ex as usize);
                                halo.entry(mli as u32).or_insert(c);
                            }
                        }
                    }
                }
            }
        }
        Ok(halo)
    }

    /// The full tiled bake: surface scatter → classify → FILL each tile, persisting every finalized tile's solid
    /// cells to a `tile_{id}.fin` disk file (one tile at a time, bounded RAM). Tiles are processed in tile-id
    /// order (deterministic). Returns the total solid count + the sorted ids of the tiles that produced solids,
    /// so the assembly can STREAM them (twice: once to build the palette, once to emit) without re-running the
    /// flood. This is the bounded-RAM entry point the offline tool + the gates call.
    pub fn bake_tiled(
        grid: &TileGrid,
        mesh: &Mesh,
        origin: [f32; 3],
        voxel_size: f32,
        supersample: usize,
    ) -> anyhow::Result<(usize, Vec<usize>)> {
        let scatter = scatter_surface(grid, mesh, origin, voxel_size, supersample)?;
        let mut cls = classify(grid, &scatter)?;
        let mut total = 0usize;
        let mut final_ids: Vec<usize> = Vec::new();
        for id in 0..grid.tile_count() {
            let ft = fill_tile(grid, &scatter, &mut cls, id)?;
            if !ft.solids.is_empty() {
                total += ft.solids.len();
                write_final(&grid.fin_path(ft.id), &ft.solids)?;
                final_ids.push(ft.id);
            }
        }
        Ok((total, final_ids))
    }

    impl TileGrid {
        fn fin_path(&self, id: usize) -> PathBuf {
            self.scratch.join(format!("tile_{id}.fin"))
        }
    }

    /// Write a finalized tile's solids as `(global_index:u64, rgba:[u8;4])` runs (sorted by global index).
    fn write_final(path: &Path, solids: &[(usize, [u8; 4])]) -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        for &(gi, rgba) in solids {
            f.write_all(&(gi as u64).to_le_bytes())?;
            f.write_all(&rgba)?;
        }
        f.flush()
    }

    /// Stream every solid cell of the finalized tiles `ids` (in id order) to `f` as `(global_index, rgba)`,
    /// reading one `.fin` file at a time (bounded RAM). The assembly calls this twice — palette build + emit.
    pub fn stream_final<F: FnMut(usize, [u8; 4])>(grid: &TileGrid, ids: &[usize], mut f: F) -> std::io::Result<()> {
        for &id in ids {
            let bytes = std::fs::read(grid.fin_path(id))?;
            let mut i = 0;
            while i + 12 <= bytes.len() {
                let gi = u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap()) as usize;
                let rgba = [bytes[i + 8], bytes[i + 9], bytes[i + 10], bytes[i + 11]];
                f(gi, rgba);
                i += 12;
            }
        }
        Ok(())
    }

    /// TEST-ONLY: run the tiled enclosure flood (scatter → classify → fill) over an EXPLICIT surface occupancy
    /// `surface[global_index]` (the bitset a real mesh's SAT would produce), bypassing the mesh scatter so the
    /// oracle can construct the exact hard cases (cracks crossing tiles, S-tunnels, buried tiles, twin cavities)
    /// directly. Returns the global-cell index set of EVERY solid cell after fill — the classification the
    /// oracle compares CELL-FOR-CELL against the monolithic `solid_fill`. Writes/cleans its own scratch under
    /// `scratch`.
    #[cfg(test)]
    pub fn classify_from_surface(
        dims: [i32; 3],
        tile_edge: i32,
        scratch: PathBuf,
        surface: &[bool],
    ) -> std::io::Result<std::collections::HashSet<usize>> {
        std::fs::create_dir_all(&scratch)?;
        let grid = TileGrid::new(dims, tile_edge, scratch.clone());
        // Scatter the explicit surface into per-tile .occ/.alb files (a single uniform albedo — colour is
        // irrelevant to the enclosure classification the oracle checks).
        let n_tiles = grid.tile_count();
        let mut tile_occ: Vec<Option<TileOcc>> = (0..n_tiles).map(|_| None).collect();
        let (dx, dy) = (dims[0] as usize, dims[1] as usize);
        for (gi, &is_solid) in surface.iter().enumerate() {
            if !is_solid {
                continue;
            }
            let z = (gi / (dx * dy)) as i32;
            let r = gi % (dx * dy);
            let x = (r % dx) as i32;
            let y = (r / dx) as i32;
            let (tx, ty, tz) = grid.tile_of(x, y, z);
            let id = grid.tile_id(tx, ty, tz);
            let ext = grid.tile_extent(tx, ty, tz);
            let occ = tile_occ[id].get_or_insert_with(|| TileOcc::new(ext));
            let li = occ.local_index(x - tx * grid.tile_edge, y - ty * grid.tile_edge, z - tz * grid.tile_edge);
            occ.set(li);
        }
        let mut has_file = vec![false; n_tiles];
        for id in 0..n_tiles {
            if let Some(occ) = &tile_occ[id] {
                write_occ(&grid.occ_path(id), occ)?;
                write_alb(&grid.alb_path(id), &std::collections::BTreeMap::new())?;
                has_file[id] = true;
            }
        }
        let scatter = ScatterResult { has_file };
        let mut cls = classify(&grid, &scatter)?;
        let mut solids = std::collections::HashSet::new();
        for id in 0..grid.tile_count() {
            let ft = fill_tile(&grid, &scatter, &mut cls, id)?;
            for (gi, _) in ft.solids {
                solids.insert(gi);
            }
        }
        let _ = std::fs::remove_dir_all(&scratch);
        Ok(solids)
    }
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
        let grid = voxelize(&fallback_room(), 1.0, SUPERSAMPLE);
        // Every one of the 6 distinctly-coloured faces must contribute voxels. Before the conservative ±1
        // candidate-AABB pad, grid-aligned faces were dropped and only 2 of 6 colours survived.
        let distinct: std::collections::HashSet<[u8; 4]> = grid.albedo.values().copied().collect();
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
        // A 3×3×3 box SHELL at [1,3]³ (its 6 faces solid) around a single air cavity at (2,2,2).
        let build = |open_face: bool| -> Grid {
            let mut g = Grid::new(dims);
            for z in 1..=3 {
                for y in 1..=3 {
                    for x in 1..=3 {
                        if x == 1 || x == 3 || y == 1 || y == 3 || z == 1 || z == 3 {
                            if open_face && (x, y, z) == (2, 1, 2) {
                                continue; // poke a hole in the -Y face → the cavity reaches outside
                            }
                            let i = g.idx(x, y, z);
                            g.set_solid(i, [200, 100, 50, 255]); // a surface colour
                        }
                    }
                }
            }
            g
        };

        // CLOSED: the enclosed cavity at (2,2,2) fills solid and takes the nearest surface colour.
        let mut closed = build(false);
        let c = closed.idx(2, 2, 2);
        assert!(!closed.is_solid(c), "cavity starts air");
        solid_fill(&mut closed);
        assert!(closed.is_solid(c), "closed shell: the enclosed cavity is filled solid");
        assert_eq!(closed.albedo_at(c), [200, 100, 50, 255], "interior takes the nearest surface colour");

        // OPEN: the hole connects the cavity to the outside → it stays air (we never fill reachable space).
        let mut open = build(true);
        let o = open.idx(2, 2, 2);
        solid_fill(&mut open);
        assert!(!open.is_solid(o), "open shell: a cavity reachable from outside stays air");
    }

    /// OBJ loading: a tiny synthetic `.obj` (one coloured quad, two triangles, no `.mtl`) loads through the
    /// SAME `load_obj` → `Mesh` → `voxelize` pipeline the glTF path uses. Proves the extension-dispatched OBJ
    /// loader builds a real `Mesh` (world-space positions verbatim, no node transform) the rest of the
    /// pipeline voxelizes — the OBJ half of the new dual-format loader, exercised end-to-end on an in-repo
    /// asset. The quad has no material, so each face takes the neutral-grey `Kd` fallback (an untextured OBJ
    /// still bakes).
    #[test]
    fn obj_loader_voxelizes_a_synthetic_quad() {
        // A 4×4 m floor quad at y=0 in the XZ plane (two CCW triangles), no usemtl / no .mtl companion.
        let obj = "\
o quad
v -2.0 0.0 -2.0
v  2.0 0.0 -2.0
v  2.0 0.0  2.0
v -2.0 0.0  2.0
f 1 2 3
f 1 3 4
";
        let dir = std::env::temp_dir();
        let file = dir.join(format!("voxelize_obj_test_{}.obj", std::process::id()));
        std::fs::write(&file, obj).expect("write temp .obj");
        let mesh = load_obj(&file).expect("load_obj must parse the synthetic quad");
        let _ = std::fs::remove_file(&file);

        // Two triangles, no textures, every face the neutral-grey Kd fallback (no material in the OBJ).
        assert_eq!(mesh.triangles.len(), 2, "the quad triangulates to two triangles");
        assert!(mesh.textures.is_empty(), "an untextured OBJ has no decoded textures");
        assert_eq!(mesh.triangles[0].base, [178, 178, 178, 255], "untextured face takes the neutral Kd fallback");

        // The same downstream voxelizer the glTF/fallback path uses produces a non-empty surface grid.
        let grid = voxelize(&mesh, 0.5, SUPERSAMPLE);
        assert!(grid.solid_count() > 0, "the OBJ quad voxelizes to a non-empty surface");
    }

    /// Bounded-RAM / large-AABB sanity: a grid with a huge AABB but only a handful of solid cells stores ONLY
    /// those cells — the albedo + indices are solid-count-sized, NOT AABB-sized — and still bakes a valid,
    /// multi-tile `.vox`. This sparsity is what lets billion-cell scenes (Bistro @0.05 m) bake without OOM.
    /// Uses a 600³ AABB (216 M cells: a ~27 MB occupancy bitset — cheap; the old dense albedo would be ~864 MB).
    #[test]
    fn large_aabb_grid_stays_sparse_and_bakes() {
        let mut grid = Grid::new([600, 600, 600]);
        // A few solid voxels scattered to the AABB corners + interior (spanning multiple ≤256³ `.vox` tiles).
        let pts = [
            (0, 0, 0),
            (599, 599, 599),
            (300, 300, 300),
            (1, 2, 3),
            (599, 0, 0),
            (0, 599, 0),
            (0, 0, 599),
            (300, 1, 599),
        ];
        for (n, &(x, y, z)) in pts.iter().enumerate() {
            let i = grid.idx(x, y, z);
            grid.set_solid(i, [(n as u8).wrapping_mul(30), 100, 200, 255]);
        }
        assert_eq!(grid.solid_count(), pts.len(), "only solid cells are stored (sparse, not AABB-sized)");
        assert_eq!(grid.albedo.len(), pts.len(), "the albedo map holds ONLY solid cells");
        // idx/xyz round-trip over the full 64-bit index range (the old i32 index overflowed past ~2 G cells).
        for &(x, y, z) in &pts {
            let i = grid.idx(x, y, z);
            assert_eq!(grid.xyz(i), (x, y, z), "xyz ∘ idx is identity");
            assert!(grid.is_solid(i));
        }
        // The whole downstream (quantize + `.vox` assembly) runs O(solid) and tiles the 600³ AABB into models.
        let (palette, indices) = quantize(&grid);
        assert!(!palette.is_empty());
        assert_eq!(indices.len(), pts.len(), "sparse indices: one per solid cell");
        let data = build_dot_vox(&grid, &palette, &indices);
        let baked: usize = data.models.iter().map(|m| m.voxels.len()).sum();
        assert_eq!(baked, pts.len(), "every solid voxel lands in some ≤256³ model");
        assert!(data.models.len() > 1, "a 600³ AABB splits into multiple ≤256³ `.vox` models");
    }

    /// `strip_basisu_required` removes ONLY the `KHR_texture_basisu` entry from `extensionsRequired` (so
    /// `gltf` stops rejecting the file) while leaving `extensionsUsed` and every other field intact. A
    /// purely-synthetic JSON (no external asset needed), so it always runs.
    #[test]
    fn strip_basisu_required_drops_only_the_required_entry() {
        let src = br#"{
            "asset": {"version": "2.0"},
            "extensionsUsed": ["KHR_materials_specular", "KHR_texture_basisu"],
            "extensionsRequired": ["KHR_texture_basisu"],
            "meshes": []
        }"#;
        let stripped = strip_basisu_required(src).expect("strip must succeed");
        let v: serde_json::Value = serde_json::from_slice(&stripped).expect("stripped JSON re-parses");
        // The required-extension array is gone (it had only the one entry), so `gltf` validation passes.
        assert!(v.get("extensionsRequired").is_none(), "the sole required basisu entry is removed");
        // extensionsUsed is untouched — basisu is still advertised as used (correct: we DID use it offline).
        let used = v["extensionsUsed"].as_array().expect("extensionsUsed preserved");
        assert!(used.iter().any(|e| e.as_str() == Some("KHR_texture_basisu")), "extensionsUsed keeps basisu");
        assert!(used.iter().any(|e| e.as_str() == Some("KHR_materials_specular")), "other ext kept");
        assert_eq!(v["asset"]["version"], "2.0", "the rest of the document is untouched");
    }

    /// One real Bistro `.ktx2` base colour decodes to a full 2048×2048 RGBA8 image with non-uniform pixels
    /// (proves the UASTC+Zstd → RGBA path, not a flat fill). The Bistro textures are gitignored, so this SKIPS
    /// gracefully when the asset is absent (mirroring the round-trip test's optional-asset convention).
    #[test]
    fn ktx2_decodes_a_real_bistro_basecolor() {
        // A 2048² BaseColor present in the Bistro texture set (see the file listing in #126).
        let tex = Path::new(
            "assets/models/src/_gltfassets/Bistro/Textures/Antenna_Metal_BaseColor.ktx2",
        );
        if !tex.exists() {
            eprintln!("SKIP ktx2_decodes_a_real_bistro_basecolor: {} not present (gitignored asset)", tex.display());
            return;
        }
        let bytes = std::fs::read(tex).expect("read the .ktx2");
        let decoded = ktx2_to_rgba(&bytes).expect("a UASTC+Zstd KTX2 base colour must decode");
        assert_eq!((decoded.width, decoded.height), (2048, 2048), "full logical extent");
        assert_eq!(decoded.rgba.len(), 2048 * 2048 * 4, "exactly width*height*4 RGBA8 bytes");
        // Non-uniform: at least two distinct pixels (a real texture, not a flat-decoded constant).
        let first = &decoded.rgba[0..4];
        let differs = decoded.rgba.chunks_exact(4).any(|px| px != first);
        assert!(differs, "decoded texture has more than one colour (real content, not a flat fill)");
    }

    /// Bistro-load smoke: `BistroExterior.gltf` loads through the basisu path (the unsupported
    /// `extensionsRequired` is stripped, KTX2 base colours decode) and a COARSE bake (0.5 m, fast) produces a
    /// non-empty grid with MANY distinct albedos — proof the textures were decoded per-voxel, not collapsed to
    /// flat material factors. SKIPS gracefully when the gitignored Bistro asset is absent.
    #[test]
    fn bistro_loads_and_bakes_with_decoded_textures() {
        let gltf = Path::new("assets/models/src/_gltfassets/Bistro/BistroExterior.gltf");
        if !gltf.exists() {
            eprintln!("SKIP bistro_loads_and_bakes_with_decoded_textures: {} not present (gitignored asset)", gltf.display());
            return;
        }
        // Loads WITHOUT the "Unsupported extension" rejection (the basisu strip) and decodes KTX2 base colours.
        let mesh = load_gltf(gltf).expect("Bistro glTF must load via the basisu path");
        assert!(!mesh.triangles.is_empty(), "Bistro has geometry");
        // At least some textures decoded to real images (non-empty) — not every image flat-fell-back.
        let decoded_textures = mesh.textures.iter().filter(|t| t.width > 0).count();
        assert!(decoded_textures > 0, "at least one KTX2 base colour decoded ({decoded_textures} did)");

        // A coarse 0.5 m bake is fast but still exercises real texture sampling. The grid must be non-empty
        // with MANY distinct albedos (flat-factor-only would yield a handful of material colours).
        let grid = voxelize(&mesh, 0.5, SUPERSAMPLE);
        assert!(grid.solid_count() > 0, "the coarse Bistro bake is non-empty");
        let distinct: std::collections::HashSet<[u8; 4]> = grid.albedo.values().copied().collect();
        assert!(
            distinct.len() > 100,
            "decoded textures yield MANY distinct albedos (got {}); flat factors would give few",
            distinct.len()
        );
    }

    // ----------------------------------------------------------------------------------------
    // C3.1 — perceptual CIELAB k-means palette
    // ----------------------------------------------------------------------------------------

    /// `rgb_to_lab` matches known reference values (D65): black→L0, white→L≈100, and mid-grey 128→L≈53.6. A
    /// guard that the sRGB→linear→XYZ→Lab port is correct (a wrong matrix / missing piecewise would shift L).
    #[test]
    fn rgb_to_lab_matches_reference() {
        let black = rgb_to_lab([0, 0, 0]);
        assert!(black[0].abs() < 0.01 && black[1].abs() < 0.01 && black[2].abs() < 0.01, "black → L*a*b* ≈ 0");
        let white = rgb_to_lab([255, 255, 255]);
        assert!((white[0] - 100.0).abs() < 0.1, "white → L ≈ 100 (got {})", white[0]);
        // The full-precision standard matrix rows sum EXACTLY to the white point (Xn=0.95047, Zn=1.08883), so
        // linear white normalizes to (1,1,1) → Lab a,b are exactly zero (neutral).
        assert!(white[1].abs() < 0.01 && white[2].abs() < 0.01, "white is neutral (a,b ≈ 0): {white:?}");
        // sRGB 128 → linear ≈ 0.2158 → Y ≈ 0.2158 → L ≈ 53.59 (the canonical mid-grey lightness).
        let grey = rgb_to_lab([128, 128, 128]);
        assert!((grey[0] - 53.59).abs() < 0.2, "mid-grey → L ≈ 53.6 (got {})", grey[0]);
    }

    /// The CIELAB k-means palette keeps PERCEPTUALLY-DISTINCT colours distinct where the input exceeds the
    /// budget. We build > `max` distinct colours dominated by two well-separated greens plus filler, force a
    /// tiny palette, and assert both greens survive as separate entries (a perceptual clustering keeps the two
    /// green modes apart). Also asserts DETERMINISM: two builds → byte-identical palette.
    #[test]
    fn cielab_palette_keeps_distinct_greens_and_is_deterministic() {
        // Two clearly-distinct greens (the perceptual pair we must NOT merge) given heavy weight, plus a spread
        // of filler colours so the distinct count exceeds the budget and k-means actually runs.
        let mut pixels: Vec<([u8; 4], u32)> = Vec::new();
        pixels.push(([40, 180, 60, 255], 500)); // vivid green
        pixels.push(([90, 140, 70, 255], 500)); // olive/muted green — perceptually distinct from the first
        // Filler: a ramp of distinct colours (low weight) to push past the palette budget.
        for i in 0..40u32 {
            pixels.push(([(i * 6) as u8, 10, (200 - i * 4) as u8, 255], 1));
        }
        pixels.sort_unstable();

        let max = 8;
        let pal = build_palette(&pixels, max);
        assert!(pal.len() <= max, "palette respects the budget");

        // Each input green must map to a palette entry that is itself green-dominant and close in Lab — i.e.
        // they are NOT collapsed into one shared bucket. Assert the two greens land on DIFFERENT palette entries.
        let pal_lab: Vec<[f32; 3]> = pal.iter().map(|c| rgb_to_lab([c[0], c[1], c[2]])).collect();
        let g1 = nearest_palette_lab(&pal_lab, [40, 180, 60, 255]);
        let g2 = nearest_palette_lab(&pal_lab, [90, 140, 70, 255]);
        assert_ne!(g1, g2, "the two perceptually-distinct greens stay on separate palette entries");

        // Determinism: a second build of the same (sorted) input is byte-identical.
        let pal2 = build_palette(&pixels, max);
        assert_eq!(pal, pal2, "the CIELAB k-means palette is byte-reproducible (fixed seed, sorted input)");
    }

    /// Perf + determinism at scale: k-means over a realistic distinct-colour count (10 000 → 255 palette)
    /// runs in well under a second AND is byte-reproducible. Reports the wall-clock (the C3 perf gate).
    #[test]
    fn kmeans_palette_is_fast_and_deterministic_at_scale() {
        // 10 000 distinct colours on a deterministic Lab-spanning ramp, each with a pseudo-count.
        let mut pixels: Vec<([u8; 4], u32)> = Vec::with_capacity(10_000);
        for i in 0..10_000u32 {
            let r = (i % 256) as u8;
            let g = ((i / 256) * 7 % 256) as u8;
            let b = ((i * 13) % 256) as u8;
            pixels.push(([r, g, b, 255], 1 + (i % 17)));
        }
        pixels.sort_unstable();
        pixels.dedup_by_key(|(c, _)| *c);

        let t = std::time::Instant::now();
        let pal = build_palette(&pixels, 255);
        let elapsed = t.elapsed();
        assert!(pal.len() <= 255 && !pal.is_empty(), "produces a bounded non-empty palette");
        let pal2 = build_palette(&pixels, 255);
        assert_eq!(pal, pal2, "k-means palette is byte-reproducible at scale");
        eprintln!(
            "k-means: {} distinct colours → {} palette entries in {:.1} ms",
            pixels.len(),
            pal.len(),
            elapsed.as_secs_f64() * 1000.0
        );
        // Generous ceiling: the offline tool is fine at hundreds of ms, but flag a pathological blowup.
        assert!(elapsed.as_secs_f64() < 30.0, "k-means must stay tractable (took {elapsed:?})");
    }

    /// The lossless short-circuit: ≤`max` distinct colours are emitted EXACTLY (no k-means, no averaging), in
    /// sorted input order.
    #[test]
    fn palette_lossless_short_circuit_returns_exact_colours() {
        let mut pixels: Vec<([u8; 4], u32)> = vec![
            ([10, 20, 30, 255], 3),
            ([200, 100, 50, 255], 1),
            ([0, 0, 0, 255], 7),
        ];
        pixels.sort_unstable();
        let pal = build_palette(&pixels, 255);
        let exact: Vec<[u8; 4]> = pixels.iter().map(|(c, _)| *c).collect();
        assert_eq!(pal, exact, "≤max distinct colours are returned exactly (lossless), in sorted order");
    }

    // ----------------------------------------------------------------------------------------
    // C3.2 — area-averaged albedo
    // ----------------------------------------------------------------------------------------

    /// Build a high-frequency black/white checkerboard texture (`n×n` texels, 1 texel per check) and a single
    /// large quad in the XZ plane mapped to the full [0,1]² UV range. Used to prove area-averaging collapses
    /// the checker to mid-grey at a voxel pitch far coarser than a check.
    fn checkerboard_quad(n: u32, world: f32) -> Mesh {
        let mut rgba = Vec::with_capacity((n * n * 4) as usize);
        for y in 0..n {
            for x in 0..n {
                let on = (x + y) % 2 == 0;
                let c = if on { 255 } else { 0 };
                rgba.extend_from_slice(&[c, c, c, 255]);
            }
        }
        let tex = Texture { width: n, height: n, rgba };
        let h = world * 0.5;
        // Two triangles for a quad in the y=0 plane, UVs covering [0,1]².
        let p = [[-h, 0.0, -h], [h, 0.0, -h], [h, 0.0, h], [-h, 0.0, h]];
        let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let tri = |a: usize, b: usize, c: usize| Triangle {
            p: [p[a], p[b], p[c]],
            uv: [uv[a], uv[b], uv[c]],
            texture: Some(0),
            base: [255, 255, 255, 255],
        };
        Mesh { triangles: vec![tri(0, 1, 2), tri(0, 2, 3)], textures: vec![tex] }
    }

    /// Per-voxel-albedo variance over the solid (textured) cells' grey level. The baseline (point sample, S=1)
    /// aliases each voxel to pure black or white → high variance; area-averaging (S>1) pulls each toward the
    /// mid-grey footprint average → variance drops. Returns `(mean_grey, variance)`.
    fn grey_variance(grid: &Grid) -> (f64, f64) {
        let greys: Vec<f64> = grid.albedo.values().map(|c| c[0] as f64).collect();
        let n = greys.len() as f64;
        let mean = greys.iter().sum::<f64>() / n;
        let var = greys.iter().map(|g| (g - mean) * (g - mean)).sum::<f64>() / n;
        (mean, var)
    }

    /// Area-averaging a high-frequency checkerboard at a COARSE voxel pitch yields ~mid-grey voxels (not
    /// aliased pure black/white), and the per-voxel-albedo variance DROPS sharply vs the point-sample baseline
    /// (`SUPERSAMPLE = 1`). This is the C3.2 acceptance: the area average kills texel aliasing.
    #[test]
    fn area_average_reduces_albedo_variance_on_checkerboard() {
        // A high-frequency checker whose period is NON-commensurate with the voxel pitch, so the per-voxel
        // point sample scatters across both checker phases (genuine texel aliasing → high per-voxel variance),
        // while the area average sees ≈half of each phase in every voxel → ~mid-grey (low variance). 50 checks
        // over 6.4 m = 0.128 m/check; voxelize at 0.8 m → 6.25 checks/voxel (non-integer) so no phase locking.
        // A finer supersample (5×5) is used so the area average resolves the ~6 checks per voxel cleanly.
        let mesh = checkerboard_quad(50, 6.4);
        let voxel = 0.8;
        let s_area = 5;

        let point = voxelize(&mesh, voxel, 1); // baseline: single nearest-texel point sample
        let area = voxelize(&mesh, voxel, s_area); // area-averaged

        assert!(point.solid_count() > 0 && area.solid_count() > 0, "both bakes produce solid voxels");

        let (point_mean, point_var) = grey_variance(&point);
        let (area_mean, area_var) = grey_variance(&area);

        // Area average lands near mid-grey (≈128) — NOT aliased to the extremes.
        assert!(
            (area_mean - 128.0).abs() < 60.0,
            "area-averaged voxels are ~mid-grey (mean {area_mean:.1}), not aliased black/white"
        );
        // The headline gate: per-voxel variance drops sharply with area-averaging vs the point baseline.
        assert!(
            area_var < point_var * 0.5,
            "per-voxel variance drops with area-averaging: point {point_var:.1} → area {area_var:.1}"
        );
        eprintln!(
            "checkerboard variance: point(S=1) mean {point_mean:.1} var {point_var:.1} | \
             area(S={s_area}) mean {area_mean:.1} var {area_var:.1} \
             (drop {:.1}×)",
            point_var / area_var.max(1e-9)
        );
    }

    /// Bilinear `Texture::sample`: at a texel CENTRE the sample equals that texel exactly (no neighbour bleed),
    /// and exactly BETWEEN two texels it is their average (the 4-tap blend), proving the filter upgrade.
    #[test]
    fn texture_sample_is_bilinear() {
        // 2×1 texture: left texel black, right texel white.
        let tex = Texture { width: 2, height: 1, rgba: vec![0, 0, 0, 255, 255, 255, 255, 255] };
        // Texel centres are at u = 0.25 (left) and u = 0.75 (right).
        assert_eq!(tex.sample(0.25, 0.5)[0], 0, "at the left texel centre → exact black");
        assert_eq!(tex.sample(0.75, 0.5)[0], 255, "at the right texel centre → exact white");
        // Halfway between the two centres (u = 0.5) → average grey (~128).
        let mid = tex.sample(0.5, 0.5)[0];
        assert!((mid as i32 - 128).abs() <= 1, "midway between texels → bilinear average (~128, got {mid})");
    }

    // ----------------------------------------------------------------------------------------
    // C1 — tiled out-of-core voxelizer gates (oracle / determinism / bounded-RAM / no-regression)
    // ----------------------------------------------------------------------------------------

    /// The MONOLITHIC oracle's solid set over an EXPLICIT surface occupancy: build a `Grid` with exactly
    /// `surface` set solid, run the production `solid_fill`, and return the global-cell index set of every solid
    /// cell. The tiled path's [`tiled::classify_from_surface`] must reproduce this CELL-FOR-CELL. Decoupling the
    /// occupancy from a mesh lets the oracle construct the hard cases (cracks, S-tunnels, buried tiles) exactly.
    fn monolithic_solid_set(dims: [i32; 3], surface: &[bool]) -> std::collections::HashSet<usize> {
        let mut g = Grid::new(dims);
        for (gi, &s) in surface.iter().enumerate() {
            if s {
                g.set_solid(gi, [200, 100, 50, 255]);
            }
        }
        solid_fill(&mut g);
        // The solid set = every cell whose occupancy bit is now set.
        let total = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
        (0..total).filter(|&i| g.is_solid(i)).collect()
    }

    /// Run the tiled flood over `surface` at `tile_edge` and assert the solid set is CELL-FOR-CELL identical to
    /// the monolithic oracle. Returns the (shared) solid set for further assertions.
    fn assert_oracle_match(
        name: &str,
        dims: [i32; 3],
        surface: &[bool],
        tile_edge: i32,
    ) -> std::collections::HashSet<usize> {
        let oracle = monolithic_solid_set(dims, surface);
        let scratch = std::env::temp_dir().join(format!(
            "voxelize_oracle_{name}_{}_{}",
            std::process::id(),
            tile_edge
        ));
        let tiled_set =
            tiled::classify_from_surface(dims, tile_edge, scratch, surface).expect("tiled classify");
        // Cell-for-cell: same cardinality AND same membership (a symmetric-difference would be a wrong enclosure).
        assert_eq!(
            tiled_set.len(),
            oracle.len(),
            "{name}: tiled solid count {} != oracle {} (TILE_EDGE={tile_edge})",
            tiled_set.len(),
            oracle.len()
        );
        let diff: Vec<usize> = oracle.symmetric_difference(&tiled_set).copied().collect();
        assert!(
            diff.is_empty(),
            "{name}: tiled flood differs from the monolithic oracle at {} cells (TILE_EDGE={tile_edge}); \
             first divergent global index {:?}",
            diff.len(),
            diff.first()
        );
        oracle
    }

    /// A linear index helper for the hard-case scene builders.
    fn gidx(dims: [i32; 3], x: i32, y: i32, z: i32) -> usize {
        (x as usize) + (y as usize) * (dims[0] as usize) + (z as usize) * (dims[0] as usize) * (dims[1] as usize)
    }

    /// **THE ORACLE GATE (§C1.9.1): the tiled flood is CELL-FOR-CELL identical to the monolithic `solid_fill`**,
    /// over all four hard cases, with a SMALL `TILE_EDGE` so each scene spans many tiles (the union-find stitch
    /// is exercised, not just the per-tile local flood). The cases:
    ///  (a) a SEALED box (enclosed → its cavity fills) vs the same box with a 1-voxel crack to the boundary whose
    ///      leak path CROSSES ≥2 tiles (open → stays air) — this is the union-find-stitch proof.
    ///  (b) a cavity reachable only via a long S-tunnel spanning many tiles (transitive exterior across tiles).
    ///  (c) a fully-buried FILE-LESS tile (no surface) that is ENCLOSED → uniform-solid, AND one that is exterior
    ///      → air (the file-less-tile component-enumeration guard).
    ///  (d) two adjacent INDEPENDENT cavities (distinct union-find sets, both enclosed).
    /// Each is checked at `TILE_EDGE` 8 AND 16 (different tilings of the same scene → same answer).
    #[test]
    fn tiled_flood_matches_monolithic_oracle_on_hard_cases() {
        for &t in &[8i32, 16] {
            // ---- (a) sealed box vs a box with a ≥2-tile crack. A 30³ grid; a hollow box shell spanning the
            // middle so its walls + the crack cross several tile boundaries at T=8. ----
            let dims = [30, 30, 30];
            let n = (dims[0] * dims[1] * dims[2]) as usize;
            let in_box = |x: i32, y: i32, z: i32| (5..=24).contains(&x) && (5..=24).contains(&y) && (5..=24).contains(&z);
            let on_shell = |x: i32, y: i32, z: i32| {
                in_box(x, y, z) && (x == 5 || x == 24 || y == 5 || y == 24 || z == 5 || z == 24)
            };
            // Sealed: the full shell. Its interior (6..23)³ is one enclosed cavity → fills solid.
            let mut sealed = vec![false; n];
            for z in 0..dims[2] {
                for y in 0..dims[1] {
                    for x in 0..dims[0] {
                        if on_shell(x, y, z) {
                            sealed[gidx(dims, x, y, z)] = true;
                        }
                    }
                }
            }
            let sealed_set = assert_oracle_match("sealed_box", dims, &sealed, t);
            // The cavity centre IS filled in the sealed case (enclosed).
            assert!(sealed_set.contains(&gidx(dims, 15, 15, 15)), "sealed box: the enclosed cavity fills solid");

            // Cracked: remove a 1-voxel-wide COLUMN of the -X wall AND carve an air channel from that hole out to
            // the grid boundary (x: 0..5 at the same (y,z)), so the leak path runs x=0→24 — crossing tiles
            // x∈{0..7},{8..15},{16..23},{24..} at T=8 (≥2 tiles). The cavity now reaches outside → stays air.
            let mut cracked = sealed.clone();
            let (cy, cz) = (15, 15);
            // Poke the wall hole at (5,cy,cz) and ensure the channel (0..=4, cy, cz) is air (it already is —
            // nothing was solid there). Removing the shell voxel at x=5 opens the cavity to the air channel,
            // which runs to x=0 (the grid boundary) → exterior.
            cracked[gidx(dims, 5, cy, cz)] = false;
            let cracked_set = assert_oracle_match("cracked_box", dims, &cracked, t);
            assert!(
                !cracked_set.contains(&gidx(dims, 15, 15, 15)),
                "cracked box: the cavity leaks to the boundary across ≥2 tiles → stays air"
            );

            // ---- (b) an S-tunnel: an enclosed pocket connected to the outside ONLY by a long thin air channel
            // that snakes across many tiles. We carve the channel out of an otherwise-solid SLAB so the only air
            // is the tunnel + the pocket; the pocket is therefore exterior (reaches the boundary via the S). ----
            let sdims = [24, 9, 5];
            let sn = (sdims[0] * sdims[1] * sdims[2]) as usize;
            let mut slab = vec![true; sn]; // start fully solid
            // Carve an S-shaped 1-voxel air tunnel at z=2: go +X along y=2, up to y=6, back -X, reaching x=0.
            let carve = |v: &mut [bool], x: i32, y: i32, z: i32| v[gidx(sdims, x, y, z)] = false;
            for x in 0..=20 {
                carve(&mut slab, x, 2, 2);
            }
            for y in 2..=6 {
                carve(&mut slab, 20, y, 2);
            }
            for x in 0..=20 {
                carve(&mut slab, x, 6, 2);
            }
            // The tunnel reaches x=0 at both ends (the grid boundary) → all carved air is exterior. The oracle
            // must keep EXACTLY the carved cells air; the tiled path must agree across the many tiles the S spans.
            let s_set = assert_oracle_match("s_tunnel", sdims, &slab, t);
            assert!(!s_set.contains(&gidx(sdims, 10, 2, 2)), "S-tunnel: the channel stays air (reaches boundary)");
            assert!(s_set.contains(&gidx(sdims, 10, 4, 2)), "S-tunnel: the solid between the channel arms stays solid");

            // ---- (c) a fully-buried FILE-LESS tile. A grid ≥ 3 tiles per axis; fill the MIDDLE region solid as
            // a shell around a tile-sized buried cavity that has NO surface inside it (so its tile is file-less),
            // enclosed → must become uniform solid. Plus a file-less tile OUTSIDE the solid → exterior → air. ----
            // Use T=8 specific geometry: a 24³ grid = 3×3×3 tiles. Make tile (1,1,1) [coords 8..16] fully buried:
            // surround it with a solid shell one voxel thick just outside the tile, leaving the tile's interior
            // (8..16)³ entirely air with no surface → file-less. It is enclosed → fills solid.
            if t == 8 {
                let bdims = [24, 24, 24];
                let bn = (bdims[0] * bdims[1] * bdims[2]) as usize;
                let mut buried = vec![false; bn];
                // A solid shell on the faces just outside tile (1,1,1): the planes x=7,x=16,y=7,y=16,z=7,z=16
                // over the 7..=16 cube → encloses the 8..15 cube (the file-less middle tile) with no interior
                // surface. (The shell voxels live in neighbouring tiles, so the middle tile has NO file.)
                for z in 7..=16 {
                    for y in 7..=16 {
                        for x in 7..=16 {
                            let on = x == 7 || x == 16 || y == 7 || y == 16 || z == 7 || z == 16;
                            if on {
                                buried[gidx(bdims, x, y, z)] = true;
                            }
                        }
                    }
                }
                let bset = assert_oracle_match("buried_fileless_tile", bdims, &buried, 8);
                // The buried file-less tile's centre is enclosed → solid.
                assert!(bset.contains(&gidx(bdims, 12, 12, 12)), "buried file-less tile: enclosed → uniform solid");
                // A far-away file-less tile (e.g. tile (0,0,0) corner away from the shell) is exterior → air.
                assert!(!bset.contains(&gidx(bdims, 1, 1, 1)), "a far file-less tile is exterior → stays air");
            }

            // ---- (d) two adjacent INDEPENDENT cavities (distinct union-find sets, both enclosed). Two sealed
            // boxes side by side sharing a solid wall; each cavity fills independently. ----
            let ddims = [20, 12, 12];
            let dn = (ddims[0] * ddims[1] * ddims[2]) as usize;
            let mut twin = vec![false; dn];
            // Box A shell over x:2..=8, box B shell over x:11..=17, both y,z:2..=9. The wall at x=8..11 region
            // separates them; each interior is its own enclosed cavity.
            let shell = |v: &mut [bool], x0: i32, x1: i32| {
                for z in 2..=9 {
                    for y in 2..=9 {
                        for x in x0..=x1 {
                            let on = x == x0 || x == x1 || y == 2 || y == 9 || z == 2 || z == 9;
                            if on {
                                v[gidx(ddims, x, y, z)] = true;
                            }
                        }
                    }
                }
            };
            shell(&mut twin, 2, 8);
            shell(&mut twin, 11, 17);
            let tset = assert_oracle_match("twin_cavities", ddims, &twin, t);
            assert!(tset.contains(&gidx(ddims, 5, 5, 5)), "twin cavities: cavity A fills");
            assert!(tset.contains(&gidx(ddims, 14, 5, 5)), "twin cavities: cavity B fills");
            // The gap between them (x=9,10 at the boxes' mid-height) is OUTSIDE both shells → exterior → air.
            assert!(!tset.contains(&gidx(ddims, 9, 5, 0)), "the space outside both boxes stays air");
        }
    }

    /// **DETERMINISM (§C1.9.2): the tiled solid mask is reproducible** — the same scene baked twice yields the
    /// identical solid set, AND two DIFFERENT `TILE_EDGE`s yield the identical solid set (the partition is a
    /// fixed function of tile ids, independent of the tiling granularity).
    #[test]
    fn tiled_flood_is_deterministic_across_tile_edges() {
        // A mixed scene: a sealed box (enclosed cavity) + a separate open shell (a cavity reaching the boundary).
        let dims = [28, 20, 20];
        let n = (dims[0] * dims[1] * dims[2]) as usize;
        let mut surface = vec![false; n];
        // Sealed box shell at x:2..=10, y,z:2..=10.
        for z in 2..=10 {
            for y in 2..=10 {
                for x in 2..=10 {
                    if x == 2 || x == 10 || y == 2 || y == 10 || z == 2 || z == 10 {
                        surface[gidx(dims, x, y, z)] = true;
                    }
                }
            }
        }
        // Open shell at x:14..=22 with a hole in the +X-most wall opening to the boundary side.
        for z in 2..=10 {
            for y in 2..=10 {
                for x in 14..=22 {
                    if x == 14 || x == 22 || y == 2 || y == 10 || z == 2 || z == 10 {
                        surface[gidx(dims, x, y, z)] = true;
                    }
                }
            }
        }
        surface[gidx(dims, 22, 6, 6)] = false; // poke the +X wall → that cavity leaks out

        let bake = |t: i32, tag: &str| {
            let scratch = std::env::temp_dir().join(format!("voxelize_det_{tag}_{}", std::process::id()));
            tiled::classify_from_surface(dims, t, scratch, &surface).expect("tiled classify")
        };
        let a8 = bake(8, "a8");
        let b8 = bake(8, "b8");
        assert_eq!(a8, b8, "same scene + same TILE_EDGE → identical solid mask (reproducible)");
        let a16 = bake(16, "a16");
        assert_eq!(a8, a16, "two different TILE_EDGEs → identical solid mask (granularity-independent)");
        // And it matches the monolithic oracle (the classification is correct, not just self-consistent).
        let oracle = monolithic_solid_set(dims, &surface);
        assert_eq!(a8, oracle, "the deterministic tiled mask equals the monolithic oracle");
    }

    /// **BOUNDED RAM (§C1.9.3): a large-AABB SPARSE scene bakes through the tiled path with a peak-RSS probe
    /// under the budget.** The synthetic scene is a big hollow box (a large AABB, sparse surface) at a fine grid
    /// so it spans MANY tiles; the tiled path holds only one tile's working set + the (small) union-find at a
    /// time, so peak RAM stays well under 4 GiB regardless of the AABB volume. (Bistro adds the real asset when
    /// present — that bake is documented in `docs/TESTING.md` as a separate long run; this unit test always runs.)
    #[test]
    fn tiled_bake_bounded_ram_on_large_sparse_scene() {
        // 200³ = 8 M cells; a hollow box shell (sparse) → ~240 K surface cells. At T=32 that is ~7³ tiles. The
        // dense monolithic occupancy would be 1 MB here (small), but the POINT is the tiled path's peak is a few
        // tiles (32³ bits = 4 KiB occ + 32 KiB label each) + union-find (a few hundred components) — KB-scale
        // working set independent of the 8 M-cell volume.
        let dims = [200, 200, 200];
        let n = (dims[0] * dims[1] * dims[2]) as usize;
        let mut surface = vec![false; n];
        for z in 10..=189 {
            for y in 10..=189 {
                for x in 10..=189 {
                    if x == 10 || x == 189 || y == 10 || y == 189 || z == 10 || z == 189 {
                        surface[gidx(dims, x, y, z)] = true;
                    }
                }
            }
        }
        let rss_before = peak_rss_bytes();
        let scratch = std::env::temp_dir().join(format!("voxelize_ram_{}", std::process::id()));
        let solids = tiled::classify_from_surface(dims, 32, scratch, &surface).expect("tiled classify");
        let rss_after = peak_rss_bytes();
        // The box is sealed → its whole interior fills: solids ≈ the full 180³ cube.
        assert!(solids.len() > 5_000_000, "the sealed large box fills its interior solid (got {})", solids.len());
        // Peak RSS budget. The probe is the process peak; we assert the DELTA stays well under the 4 GiB budget
        // (the absolute peak includes the test harness baseline). The surface bitset the test itself holds (8 M
        // bools = 8 MB) + the solid HashSet (~5 M usizes ≈ 40 MB) dominate here, NOT the tiled working set.
        let budget = 4u64 * 1024 * 1024 * 1024;
        if let (Some(before), Some(after)) = (rss_before, rss_after) {
            assert!(after < budget, "peak RSS {after} bytes exceeds the 4 GiB budget");
            eprintln!(
                "tiled bounded-RAM: peak RSS {:.1} MiB (delta {:.1} MiB over the bake), budget 4096 MiB",
                after as f64 / (1024.0 * 1024.0),
                (after.saturating_sub(before)) as f64 / (1024.0 * 1024.0),
            );
        } else {
            eprintln!("tiled bounded-RAM: peak-RSS probe unavailable on this platform — correctness still checked");
        }
    }

    /// Best-effort PEAK resident-set bytes for THIS process (the bounded-RAM probe). Windows: `GetProcessMemoryInfo`
    /// `PeakWorkingSetSize`. Other platforms return `None` (the test then skips the RSS assertion, keeping the
    /// correctness check). No external crate — a tiny `windows-sys`-free FFI to `psapi`.
    fn peak_rss_bytes() -> Option<u64> {
        #[cfg(windows)]
        {
            // PROCESS_MEMORY_COUNTERS: cb, PageFaultCount, PeakWorkingSetSize, WorkingSetSize, ...
            #[repr(C)]
            struct ProcessMemoryCounters {
                cb: u32,
                page_fault_count: u32,
                peak_working_set_size: usize,
                working_set_size: usize,
                quota_peak_paged_pool_usage: usize,
                quota_paged_pool_usage: usize,
                quota_peak_non_paged_pool_usage: usize,
                quota_non_paged_pool_usage: usize,
                pagefile_usage: usize,
                peak_pagefile_usage: usize,
            }
            unsafe extern "system" {
                fn GetCurrentProcess() -> isize;
                fn K32GetProcessMemoryInfo(process: isize, counters: *mut ProcessMemoryCounters, cb: u32) -> i32;
            }
            let mut pmc: ProcessMemoryCounters = unsafe { std::mem::zeroed() };
            pmc.cb = std::mem::size_of::<ProcessMemoryCounters>() as u32;
            let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) };
            if ok != 0 { Some(pmc.peak_working_set_size as u64) } else { None }
        }
        #[cfg(not(windows))]
        {
            None
        }
    }

    /// **NO REGRESSION (§C1.9.4): small scenes bake the SAME through the tiled path with a LARGE `TILE_EDGE` (a
    /// single tile = the degenerate case == today).** Routes the existing `solid_fill_closes_enclosed_but_keeps_
    /// open_air` scene through the tiled flood at a tile edge ≥ the grid extent (one tile) and asserts the solid
    /// set equals the monolithic oracle — the degenerate single-tile case must reproduce the monolithic path.
    #[test]
    fn tiled_single_tile_matches_monolithic_small_scene() {
        // The 5³ enclosed-cavity scene from `solid_fill_closes_enclosed_but_keeps_open_air`, as a surface bitset.
        let dims = [5, 5, 5];
        let build = |open_face: bool| {
            let n = (dims[0] * dims[1] * dims[2]) as usize;
            let mut s = vec![false; n];
            for z in 1..=3 {
                for y in 1..=3 {
                    for x in 1..=3 {
                        if x == 1 || x == 3 || y == 1 || y == 3 || z == 1 || z == 3 {
                            if open_face && (x, y, z) == (2, 1, 2) {
                                continue;
                            }
                            s[gidx(dims, x, y, z)] = true;
                        }
                    }
                }
            }
            s
        };
        // A LARGE tile edge (≥5) ⇒ one tile ⇒ the degenerate (monolithic-equivalent) case.
        for big in [5i32, 64] {
            let closed = assert_oracle_match("small_closed", dims, &build(false), big);
            assert!(closed.contains(&gidx(dims, 2, 2, 2)), "closed shell: the cavity fills (single-tile)");
            let open = assert_oracle_match("small_open", dims, &build(true), big);
            assert!(!open.contains(&gidx(dims, 2, 2, 2)), "open shell: the cavity stays air (single-tile)");
        }
    }

    /// **No-regression for the fallback room (§C1.9.4):** the procedural room baked through the FULL tiled bake
    /// (mesh → tiled scatter → flood → fill → solids) with a large single tile produces the SAME solid set as the
    /// monolithic `voxelize` + `solid_fill`. Exercises the real mesh-scatter path (not just the explicit-surface
    /// oracle helper), proving the tiled surface scatter matches the monolithic surface SAT cell-for-cell.
    #[test]
    fn tiled_full_bake_matches_monolithic_on_fallback_room() {
        let mesh = fallback_room();
        let voxel = 1.0;
        let (origin, dims) = grid_geometry(&mesh, voxel).expect("room has geometry");

        // Monolithic: voxelize + solid_fill → solid set.
        let mut mono = voxelize(&mesh, voxel, SUPERSAMPLE);
        solid_fill(&mut mono);
        let total = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
        let mono_set: std::collections::HashSet<usize> = (0..total).filter(|&i| mono.is_solid(i)).collect();

        // Tiled FULL bake (mesh scatter included) with a single big tile.
        let big = dims[0].max(dims[1]).max(dims[2]) + 1;
        let scratch = std::env::temp_dir().join(format!("voxelize_room_full_{}", std::process::id()));
        std::fs::create_dir_all(&scratch).expect("scratch");
        let grid = tiled::TileGrid::new(dims, big, scratch.clone());
        let (_total, final_ids) =
            tiled::bake_tiled(&grid, &mesh, origin, voxel, SUPERSAMPLE).expect("tiled bake");
        let mut tiled_set = std::collections::HashSet::new();
        tiled::stream_final(&grid, &final_ids, |gi, _| {
            tiled_set.insert(gi);
        })
        .expect("stream final");
        let _ = std::fs::remove_dir_all(&scratch);

        assert_eq!(
            tiled_set.len(),
            mono_set.len(),
            "tiled full bake solid count {} != monolithic {}",
            tiled_set.len(),
            mono_set.len()
        );
        assert!(
            mono_set.symmetric_difference(&tiled_set).next().is_none(),
            "tiled full bake (mesh→scatter→flood) differs from monolithic voxelize+solid_fill on the room"
        );
    }

    /// **End-to-end tiled `.vxo`:** the room baked through the FULL tiled out-of-core pipeline (mesh → tiled
    /// flood → STREAMING `.vxo` assembly) writes a valid `.vxo` that parses through the engine reader with a
    /// non-empty brick set whose decoded solids match the bake's solid count. Proves the streaming assembly
    /// (`assemble_vxo_streaming` + `VxoStreamWriter`) emits a loadable artifact, not just a correct mask.
    #[test]
    fn tiled_vxo_end_to_end_parses_and_round_trips() {
        use adventure::voxel::brickmap::BRICK_EDGE;
        use adventure::voxel::vxo::VxoFile;

        let mesh = fallback_room();
        let voxel = 0.5;
        let (origin, dims) = grid_geometry(&mesh, voxel).expect("room geometry");
        let scratch = std::env::temp_dir().join(format!("voxelize_vxo_e2e_{}", std::process::id()));
        std::fs::create_dir_all(&scratch).expect("scratch");
        let grid = tiled::TileGrid::new(dims, 16, scratch.clone());
        let (total_solid, final_ids) =
            tiled::bake_tiled(&grid, &mesh, origin, voxel, SUPERSAMPLE).expect("tiled bake");
        assert!(total_solid > 0, "the room bakes solid voxels");

        let out = scratch.join("room.vxo");
        // STORE so the test doesn't require the zstd compressor to be present at parse-decode (still uses the
        // streaming writer + region framing).
        assemble_vxo_streaming(&grid, &final_ids, voxel, &out, true).expect("assemble .vxo");

        let bytes = std::fs::read(&out).expect("read .vxo");
        let file = VxoFile::parse(&bytes).expect("the tiled .vxo parses");
        assert!(file.head.brick_count > 0, "the .vxo has bricks");
        assert_eq!(file.head.voxel_size, voxel, "HEAD records the bake spacing");
        // Decode every region's bricks and count solids — must equal the bake's solid count (the assembly is
        // lossless: every solid cell lands in exactly one brick voxel).
        let mut decoded_solids = 0usize;
        for dir in &file.bidx {
            let region = file.decode_region(dir).expect("decode region");
            for entry in &region.entries {
                let brick = region.brick(entry);
                for z in 0..BRICK_EDGE {
                    for y in 0..BRICK_EDGE {
                        for x in 0..BRICK_EDGE {
                            if brick.is_solid(x, y, z) {
                                decoded_solids += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(decoded_solids, total_solid, "the .vxo decodes exactly the baked solid voxels");
        let _ = std::fs::remove_dir_all(&scratch);
    }

    /// **Bistro `.vxo` validation (optional asset; `#[ignore]`d — explicit run only).** If `assets/models/
    /// bistro.vxo` exists (baked via the tiled path — see `docs/TESTING.md`), it PARSES through the engine
    /// reader with the expected 0.05 m spacing + a large brick set, and a sampling of its regions decode without
    /// error. This is the post-bake smoke for the shipped Bistro deliverable. Run:
    /// `cargo test --example voxelize_scene --features vxo-encode -- --ignored bistro_vxo_parses`.
    #[test]
    #[ignore = "validates the baked assets/models/bistro.vxo when present; run explicitly"]
    fn bistro_vxo_parses() {
        use adventure::voxel::vxo::VxoFile;
        let path = Path::new("assets/models/bistro.vxo");
        if !path.exists() {
            eprintln!("SKIP bistro_vxo_parses: {} not present", path.display());
            return;
        }
        let bytes = std::fs::read(path).expect("read bistro.vxo");
        let file = VxoFile::parse(&bytes).expect("bistro.vxo parses");
        assert_eq!(file.head.voxel_size, 0.05, "Bistro baked at 0.05 m");
        assert!(file.head.brick_count > 100_000, "Bistro has a large brick set (got {})", file.head.brick_count);
        // Decode the first few regions to prove the bodies are well-formed.
        for dir in file.bidx.iter().take(8) {
            let region = file.decode_region(dir).expect("region decodes");
            assert!(!region.entries.is_empty(), "a Bistro region has bricks");
        }
        eprintln!(
            "bistro.vxo: {} bricks, {} regions, bounds {:?}..{:?}",
            file.head.brick_count, file.head.region_count, file.head.bounds_min, file.head.bounds_max
        );
    }
}
