//! Runtime loader for baked MagicaVoxel `.vox` static scenes — the read side of the offline `.vox`
//! pipeline whose write side is `examples/voxelize_scene.rs`.
//!
//! A `.vox` file is a fixed, pre-voxelized scene (e.g. Sponza, voxelized once offline) we load as a static
//! GI-measurement scene. [`load_vox`] is a PURE function `path -> (BrickMap, BlockRegistry)` — no Bevy
//! resources, no GPU, no `gltf`/`image` (those live ONLY in the offline example; the shipped game's loader
//! reads the baked `.vox` with `dot_vox` alone). It builds:
//!
//! * a [`BlockRegistry`] from the `.vox` 256-entry palette — palette entry `i` (sRGB `u8` RGBA) becomes
//!   `BlockId(i+1)` with the colour converted to LINEAR RGBA (matching every other registry colour), via
//!   [`BlockRegistry::from_vox_palette`]; block `0` stays AIR.
//! * a [`BrickMap`] of the SAME `8³`/0.2 m bricks the rest of the engine uses — each solid `.vox` voxel
//!   writes its `BlockId` at the right brick coord + local voxel ([`voxel_index`]).
//!
//! ## Coordinate convention
//! MagicaVoxel is **Z-up**; this engine is **Y-up**. So a `.vox` voxel `(vx, vy, vz)` maps to the world
//! voxel `(vx, vz, vy)` — the `.vox` `z` axis becomes world `+Y`. Multiple models are placed by their
//! scene-graph transform (a `.vox` `_t` translation positions a model's CENTER, the MagicaVoxel rule), so a
//! scene split into a grid of `≤256³` models (the offline writer does this when an axis exceeds 256)
//! reassembles into one contiguous world. After assembly the whole scene is shifted so its lowest world
//! voxel sits at `y = 0` (floor at the origin) and it is centred on the X/Z origin — a sensible anchor for a
//! GI-measurement scene regardless of where the authoring tool put it.

use bevy::math::IVec3;
use dot_vox::{DotVoxData, SceneNode};

use super::brickmap::{BRICK_EDGE, Brick, BrickMap, BRICK_VOXELS, brick_coord_of_voxel, voxel_index};
use super::palette::{BlockId, BlockRegistry, srgb_u8_to_linear};

/// Calibration scale tying MagicaVoxel's `_emit · 2^_flux` emission to this engine's lumen-scale linear
/// emissive radiance. Chosen `= 1.0` so the canonical default MagicaVoxel emitter — a pure-white voxel with
/// `_emit = 1.0` and `_flux = 0` (strength `1.0`) — lands at linear emissive `[1,1,1]`, i.e. the SAME radiance
/// as the Cornell light panel (`palette.rs::cornell` sets the light block to `[1,1,1]`). The engine's GI/NEE
/// treats `BlockRegistry::emissive` as lumen-scale linear radiance (memory `solari-gi`), so a unit white
/// emitter matching the reference Cornell light is the natural anchor; brighter MagicaVoxel emitters (higher
/// `_flux`) scale up proportionally above it. One named const — never a scattered magic number.
const EMISSIVE_SCALE: f32 = 1.0;

/// One placed `.vox` voxel in WORLD-voxel space (Y-up, after the Z-up→Y-up swap + model placement), with
/// its 1-based [`BlockId`] (palette index `+1`). The intermediate the loader collects before anchoring +
/// bricking, so the anchor shift is computed over the whole multi-model scene at once.
struct PlacedVoxel {
    pos: IVec3,
    block: BlockId,
}

/// Load a baked `.vox` file into a [`BrickMap`] + a [`BlockRegistry`]. Pure: reads only the file at `path`
/// and returns the two CPU data structures — no Bevy resources, no GPU, no mesh/texture decoders.
///
/// The palette becomes the registry (`BlockId(i+1)` ← palette entry `i`, sRGB→linear). Every solid voxel of
/// every model is placed by its scene transform (Z-up→Y-up swapped), the assembled scene is anchored with
/// its floor at `y = 0` and centred on X/Z, then written into the sparse `8³`/0.2 m bricks. Returns an error
/// if the file cannot be read or parsed (`dot_vox::load` yields a `String` error we wrap), or if it has no
/// models. An all-empty model set yields an empty `BrickMap` (no solid voxels) but a populated registry.
pub fn load_vox(path: impl AsRef<std::path::Path>) -> anyhow::Result<(BrickMap, BlockRegistry)> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("vox: read {}: {e}", path.display()))?;
    let data = dot_vox::load_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("vox: parse {}: {e}", path.display()))?;
    Ok(from_dot_vox(&data))
}

/// Build the `(BrickMap, BlockRegistry)` from already-parsed [`DotVoxData`] — the core, file-IO-free path
/// shared by [`load_vox`] and the round-trip test (which builds a `DotVoxData` in memory). Splitting it out
/// keeps the parse/IO at the edge and the data transform pure + directly testable.
pub fn from_dot_vox(data: &DotVoxData) -> (BrickMap, BlockRegistry) {
    let registry = registry_from_palette(data);
    let placed = place_voxels(data);
    let map = bricks_from_placed(&placed);
    (map, registry)
}

/// Build the [`BlockRegistry`] from the `.vox` palette: each `Color` (sRGB `u8` RGBA) → `BlockId(i+1)` with
/// a LINEAR-RGBA colour. Falls back to the MagicaVoxel default palette only if the file carried none (a
/// well-formed `.vox` always has 256 entries). The palette is index-aligned with voxel `i` (`Voxel.i` is the
/// 0-based palette index after `dot_vox`'s 1→0 adjustment), so `BlockId(i+1)` matches voxel `i`.
///
/// MagicaVoxel stores emissive in the `MATL` chunk (`dot_vox` parses it into `data.materials`), keyed by the
/// SAME palette index as the colour. After the base (colour-only) registry is built, [`apply_matl_emissive`]
/// overlays each emissive material's radiance onto its block (`BlockId(mat.id + 1)`), so imported lamps light
/// the GI/NEE stack (which already consumes `BlockRegistry::emissive`/`has_emitters`) with zero GI changes.
/// `dot_vox` stays confined to this module — `palette.rs` never sees the offline crate (layering invariant).
fn registry_from_palette(data: &DotVoxData) -> BlockRegistry {
    let colors: Vec<[u8; 4]> = data.palette.iter().map(|c| [c.r, c.g, c.b, c.a]).collect();
    let mut registry = BlockRegistry::from_vox_palette(&colors);
    apply_matl_emissive(&mut registry, data);
    registry
}

/// Overlay the `.vox` `MATL` emissive table onto an already-built [`BlockRegistry`] (the C2 step). For each
/// material that is an emitter — `material_type() == Some("_emit")` OR `emission()` returns some `e > 0` —
/// the emissive radiance is the voxel's OWN colour (an emitter glows in its hue, matching MagicaVoxel's
/// `_emit` rendering) times the emission strength times [`EMISSIVE_SCALE`]:
///
/// ```text
/// strength   = emission · 2^radiant_flux            // MagicaVoxel emission scale (its renderer's _emit·2^_flux)
/// albedo_lin = srgb_u8_to_linear(palette[mat.id])   // the block's linear RGB (the loader's existing decode)
/// emissive   = albedo_lin.rgb · strength · EMISSIVE_SCALE
/// ```
///
/// `mat.id` is the palette index — the SAME index the colour map uses — so it lands on `BlockId(mat.id + 1)`
/// (the `+1` of `palette` entry → block, [`from_vox_palette`](BlockRegistry::from_vox_palette)). Out-of-range
/// ids (`>= 256`) are skipped here, and [`BlockRegistry::set_emissive`] additionally no-ops AIR / out-of-range
/// blocks, so a malformed material id can never panic. The `_ldr` (low-dynamic-range) field is IGNORED: it is
/// a display-dampening hack, not physical radiance, and GI wants the true emitted radiance for the bounce.
fn apply_matl_emissive(registry: &mut BlockRegistry, data: &DotVoxData) {
    for mat in &data.materials {
        // Skip ids that can't index the 256-entry palette (defensive — set_emissive also guards).
        if mat.id >= 256 {
            continue;
        }
        // An emitter is flagged either by an explicit `_type == "_emit"` or a positive `_emit` strength.
        let e = mat.emission().unwrap_or(0.0);
        let is_emit = mat.material_type() == Some("_emit") || e > 0.0;
        if !is_emit || e <= 0.0 {
            continue;
        }
        // MagicaVoxel scales emission by 2^flux (flux is an integer-ish power exponent, default 0).
        let strength = e * 2f32.powf(mat.radiant_flux().unwrap_or(0.0));
        // The emitter's own colour (sRGB→linear, RGB only) — reuse the loader's single sRGB decode.
        let c = data.palette[mat.id as usize];
        let albedo = srgb_u8_to_linear([c.r, c.g, c.b, c.a]);
        let emissive = [
            albedo[0] * strength * EMISSIVE_SCALE,
            albedo[1] * strength * EMISSIVE_SCALE,
            albedo[2] * strength * EMISSIVE_SCALE,
        ];
        // Palette index → block id is the same `+1` as the colour map (palette entry i → BlockId(i+1)).
        registry.set_emissive(BlockId(mat.id as u16 + 1), emissive);
    }
}

/// The translation (corner offset) of each model, indexed by model id, derived from the scene graph. A
/// `.vox` Transform frame's `_t` positions the model's CENTER, so the corner offset is `t − floor(size/2)`
/// (the MagicaVoxel convention). Models not referenced by any Shape node (or files with no scene graph at
/// all — a single bare model) default to a zero offset, so a minimal one-model `.vox` still loads. Returns
/// a `Vec` parallel to `data.models` (length == model count), each entry the model's `.vox`-space corner.
fn model_offsets(data: &DotVoxData) -> Vec<IVec3> {
    let mut offsets = vec![IVec3::ZERO; data.models.len()];
    // Walk the scene graph from the root, accumulating Transform translations down to each Shape's models.
    // The graph is a flat `Vec<SceneNode>` indexed by node id; root is node 0 (MagicaVoxel's convention).
    if data.scenes.is_empty() {
        return offsets; // no scene graph: a bare single model at the origin
    }
    walk_scene(data, 0, IVec3::ZERO, &mut offsets);
    offsets
}

/// Recursive scene-graph walk: accumulate the world translation down each Transform → Group → Shape chain,
/// recording every referenced model's CORNER offset (center translation minus half its size). Bounded by the
/// node count (each node visited via its parent edge); a malformed cyclic/over-range index is simply ignored
/// (no panic) so a hand-built or corrupt graph can't crash the loader.
fn walk_scene(data: &DotVoxData, node: u32, translation: IVec3, offsets: &mut [IVec3]) {
    let Some(node) = data.scenes.get(node as usize) else { return };
    match node {
        SceneNode::Transform { frames, child, .. } => {
            // Use the first frame's translation (static scene — no animation).
            let t = frames
                .first()
                .and_then(|f| f.position())
                .map(|p| IVec3::new(p.x, p.y, p.z))
                .unwrap_or(IVec3::ZERO);
            walk_scene(data, *child, translation + t, offsets);
        }
        SceneNode::Group { children, .. } => {
            for &c in children {
                walk_scene(data, c, translation, offsets);
            }
        }
        SceneNode::Shape { models, .. } => {
            for m in models {
                if let (Some(slot), Some(model)) =
                    (offsets.get_mut(m.model_id as usize), data.models.get(m.model_id as usize))
                {
                    // `_t` is the model CENTER; convert to the min-corner offset.
                    let half = IVec3::new(
                        (model.size.x / 2) as i32,
                        (model.size.y / 2) as i32,
                        (model.size.z / 2) as i32,
                    );
                    *slot = translation - half;
                }
            }
        }
    }
}

/// Collect every solid voxel of every model into WORLD-voxel space: apply the model's `.vox`-space corner
/// offset, then swap Z-up→Y-up (`.vox (x,y,z)` → world `(x, z, y)`). The voxel's palette index `i` becomes
/// `BlockId(i+1)`. Coordinates are still in raw `.vox` units here — the floor/centre anchor is applied
/// afterward in [`bricks_from_placed`] so it sees the whole multi-model extent.
fn place_voxels(data: &DotVoxData) -> Vec<PlacedVoxel> {
    let offsets = model_offsets(data);
    let mut placed = Vec::new();
    for (mi, model) in data.models.iter().enumerate() {
        let off = offsets.get(mi).copied().unwrap_or(IVec3::ZERO);
        for v in &model.voxels {
            // `.vox` corner-relative voxel coord, plus the model's placement, in .vox space (Z-up).
            let vx = off.x + v.x as i32;
            let vy = off.y + v.y as i32;
            let vz = off.z + v.z as i32;
            // Z-up (.vox) → Y-up (world): world.y = vox.z, world.z = vox.y.
            let world = IVec3::new(vx, vz, vy);
            // `Voxel.i` is the 0-based palette index; BlockId(i+1) (block 0 is AIR).
            let block = BlockId(v.i as u16 + 1);
            placed.push(PlacedVoxel { pos: world, block });
        }
    }
    placed
}

/// Anchor the assembled scene (floor at `y = 0`, centred on X/Z), then write every solid voxel into the
/// sparse `8³` [`BrickMap`]. Bricks are built per-coord from a dense `BRICK_VOXELS` block array so the
/// uniform-collapse + occupancy invariants hold (via [`Brick::from_voxels`]); empty bricks are never stored
/// ([`BrickMap::insert`] drops them). An empty `placed` (no solid voxels) yields an empty map.
fn bricks_from_placed(placed: &[PlacedVoxel]) -> BrickMap {
    let mut map = BrickMap::new();
    if placed.is_empty() {
        return map;
    }
    // Bounds of the assembled scene in world-voxel space.
    let mut lo = IVec3::splat(i32::MAX);
    let mut hi = IVec3::splat(i32::MIN);
    for p in placed {
        lo = lo.min(p.pos);
        hi = hi.max(p.pos);
    }
    // Anchor: floor (min Y) to 0, centre X and Z on the origin. Integer-centred so voxels stay on the grid.
    let shift = IVec3::new(
        -(lo.x + hi.x) / 2, // centre X
        -lo.y,              // floor at y = 0
        -(lo.z + hi.z) / 2, // centre Z
    );

    // Group voxels into bricks. We accumulate per-brick dense arrays, then commit each brick once. Using a
    // map of dense arrays keeps `Brick::from_voxels` (and its uniform/occupancy SSOT) the single brick path.
    use rustc_hash::FxHashMap;
    let mut dense: FxHashMap<IVec3, Box<[BlockId; BRICK_VOXELS]>> = FxHashMap::default();
    for p in placed {
        let w = p.pos + shift;
        let bc = brick_coord_of_voxel(w);
        let local = w - bc * BRICK_EDGE;
        let arr = dense.entry(bc).or_insert_with(|| Box::new([BlockId::AIR; BRICK_VOXELS]));
        arr[voxel_index(local.x, local.y, local.z)] = p.block;
    }
    for (coord, arr) in dense {
        map.insert(coord, Brick::from_voxels(arr));
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use dot_vox::{Color, Material, Model, Size, Voxel};

    /// Build a `dot_vox::Material` for the in-memory `MATL` oracle: an `_emit` material at palette index `id`
    /// with the given emission strength and flux exponent (the `_type`/`_emit`/`_flux` string properties
    /// MagicaVoxel writes). Mirrors what `dot_vox` parses from a real `MATL` chunk.
    fn emit_material(id: u32, emit: f32, flux: f32) -> Material {
        let properties = [
            ("_type".to_string(), "_emit".to_string()),
            ("_emit".to_string(), emit.to_string()),
            ("_flux".to_string(), flux.to_string()),
        ]
        .into_iter()
        .collect();
        Material { id, properties }
    }

    /// Like [`make_vox`] but with an explicit `MATL` material table — the C2 emissive oracle.
    fn make_vox_with_materials(
        size: (u32, u32, u32),
        voxels: Vec<Voxel>,
        colors: &[[u8; 4]],
        materials: Vec<Material>,
    ) -> DotVoxData {
        let mut data = make_vox(size, voxels, colors);
        data.materials = materials;
        data
    }

    /// A minimal in-memory `DotVoxData`: one model of `size`, the given voxels, and a palette built from
    /// `colors` (sRGB `u8` RGBA), padded to 256 with black so it's a well-formed `.vox`. No scene graph (the
    /// single model loads at the origin via the bare-model fallback). The round-trip oracle.
    fn make_vox(size: (u32, u32, u32), voxels: Vec<Voxel>, colors: &[[u8; 4]]) -> DotVoxData {
        let mut palette: Vec<Color> = colors.iter().map(|c| Color { r: c[0], g: c[1], b: c[2], a: c[3] }).collect();
        palette.resize(256, Color { r: 0, g: 0, b: 0, a: 255 });
        DotVoxData {
            version: 150,
            index_map: Vec::new(),
            models: vec![Model { size: Size { x: size.0, y: size.1, z: size.2 }, voxels }],
            palette,
            materials: Vec::new(),
            scenes: Vec::new(),
            layers: Vec::new(),
        }
    }

    /// sRGB→linear: pure black and pure white map to 0.0 and 1.0 exactly; a mid-grey decodes BELOW its
    /// normalized sRGB value (the curve is concave) — i.e. linear < srgb/255 for mid tones. Guards the
    /// `.vox` colour-space conversion the registry depends on.
    #[test]
    fn srgb_decode_endpoints_and_midtone() {
        use crate::voxel::palette::{srgb_channel_to_linear, srgb_u8_to_linear};
        assert_eq!(srgb_channel_to_linear(0), 0.0);
        assert!((srgb_channel_to_linear(255) - 1.0).abs() < 1e-6);
        let mid = srgb_channel_to_linear(128);
        assert!(mid < 128.0 / 255.0, "sRGB mid-grey decodes darker than its raw value: {mid}");
        // Alpha is linear (passes straight through, normalized).
        assert!((srgb_u8_to_linear([0, 0, 0, 128])[3] - 128.0 / 255.0).abs() < 1e-6);
    }

    /// ROUND-TRIP: a tiny 2-colour cube `DotVoxData` → `from_dot_vox` → the `BrickMap` has exactly the
    /// expected solid voxels at the right world coords with the right palette `BlockId`s, and the registry
    /// colours are the sRGB→linear of the two palette entries. Exercises the Z-up→Y-up swap + floor anchor.
    #[test]
    fn round_trip_two_colour_cube() {
        use crate::voxel::palette::srgb_u8_to_linear;
        // Two coloured voxels in a 2×1×2 model: red at .vox (0,0,0) palette idx 0, green at .vox (1,0,1)
        // palette idx 1. The Z-up→Y-up swap (.vox (x,y,z) → world (x,z,y)) maps them to world (0,0,0) and
        // (1,1,0): the green voxel's .vox z=1 (up) becomes world y=1, so the two span world y ∈ {0,1}.
        let red = [200u8, 30, 20, 255];
        let green = [40u8, 180, 50, 255];
        let voxels = vec![
            Voxel { x: 0, y: 0, z: 0, i: 0 },
            Voxel { x: 1, y: 0, z: 1, i: 1 },
        ];
        let data = make_vox((2, 1, 2), voxels, &[red, green]);
        let (map, reg) = from_dot_vox(&data);

        // Two distinct solid voxels → at least one brick, and exactly two solid cells.
        assert!(!map.is_empty(), "two solid voxels must produce a brick");
        let mut solid = Vec::new();
        for (bc, brick) in map.iter() {
            for z in 0..BRICK_EDGE {
                for y in 0..BRICK_EDGE {
                    for x in 0..BRICK_EDGE {
                        if brick.is_solid(x, y, z) {
                            let world = *bc * BRICK_EDGE + IVec3::new(x, y, z);
                            solid.push((world, brick.get(x, y, z)));
                        }
                    }
                }
            }
        }
        assert_eq!(solid.len(), 2, "exactly two solid voxels round-trip: {solid:?}");

        // Palette idx 0 → BlockId(1), idx 1 → BlockId(2).
        let blocks: std::collections::HashSet<u16> = solid.iter().map(|(_, b)| b.0).collect();
        assert_eq!(blocks, [1u16, 2].into_iter().collect(), "palette idx i → BlockId(i+1)");

        // Floor anchor: the LOWEST world voxel sits at y = 0. The two voxels span world y ∈ {0,1} (the
        // green voxel's .vox z=1 became world y=1 via the Z-up→Y-up swap), so the minimum y is 0.
        let min_y = solid.iter().map(|(w, _)| w.y).min().unwrap();
        assert_eq!(min_y, 0, "floor anchored to y=0: {solid:?}");
        // Block colours land at the swapped coords: red (BlockId 1) at world y=0, green (2) at world y=1.
        let red_v = solid.iter().find(|(_, b)| b.0 == 1).unwrap().0;
        let green_v = solid.iter().find(|(_, b)| b.0 == 2).unwrap().0;
        assert_eq!(red_v.y, 0, "red (.vox z=0) → world y=0: {red_v:?}");
        assert_eq!(green_v.y, 1, "green (.vox z=1) → world y=1 (Z-up→Y-up): {green_v:?}");

        // Registry colours are sRGB→linear of the source palette (block 0 = AIR, transparent).
        assert!(reg.color(BlockId::AIR)[3] == 0.0, "AIR transparent");
        assert_eq!(reg.color(BlockId(1)), srgb_u8_to_linear(red), "BlockId(1) = sRGB→linear(red)");
        assert_eq!(reg.color(BlockId(2)), srgb_u8_to_linear(green), "BlockId(2) = sRGB→linear(green)");
        // Registry has AIR + 256 palette entries (the padded .vox palette is always 256).
        assert_eq!(reg.len(), 257, "AIR + 256 palette blocks");
    }

    /// write_vox → load_vox: the SAME bytes the offline writer emits parse back to the same scene. Writes the
    /// cube to a temp `.vox`, loads it via the public file path, and re-checks the solid-voxel count — proving
    /// the loader reads real `dot_vox`-serialized bytes, not just an in-memory struct.
    #[test]
    fn write_then_load_file_round_trip() {
        let voxels = vec![
            Voxel { x: 0, y: 0, z: 0, i: 0 },
            Voxel { x: 1, y: 0, z: 1, i: 1 },
        ];
        let data = make_vox((2, 1, 2), voxels, &[[200, 30, 20, 255], [40, 180, 50, 255]]);
        let mut buf = Vec::new();
        data.write_vox(&mut buf).expect("write_vox");
        let dir = std::env::temp_dir();
        let file = dir.join(format!("voxel_rt_roundtrip_{}.vox", std::process::id()));
        std::fs::write(&file, &buf).expect("write temp .vox");
        let (map, reg) = load_vox(&file).expect("load_vox");
        let _ = std::fs::remove_file(&file);

        let solid: usize = map.iter().map(|(_, b)| {
            let mut n = 0;
            for z in 0..BRICK_EDGE {
                for y in 0..BRICK_EDGE {
                    for x in 0..BRICK_EDGE {
                        if b.is_solid(x, y, z) {
                            n += 1;
                        }
                    }
                }
            }
            n
        }).sum();
        assert_eq!(solid, 2, "two voxels survive a real write→read .vox round-trip");
        assert!(reg.len() > 1, "palette populated from the .vox file");
    }

    /// An empty model (no voxels) yields an empty `BrickMap` but a populated registry — robust, no panic.
    #[test]
    fn empty_model_empty_map() {
        let data = make_vox((1, 1, 1), Vec::new(), &[[255, 255, 255, 255]]);
        let (map, reg) = from_dot_vox(&data);
        assert!(map.is_empty(), "no solid voxels → empty map");
        assert!(!reg.is_empty(), "registry still has the palette blocks");
    }

    /// If the baked Sponza asset exists, it loads non-empty with a bounded brick count + a populated palette.
    /// Skipped (passes vacuously) when `assets/models/sponza.vox` hasn't been generated yet.
    #[test]
    fn sponza_loads_if_present() {
        let path = std::path::Path::new("assets/models/sponza.vox");
        if !path.exists() {
            return; // asset not baked in this checkout — nothing to assert
        }
        let (map, reg) = load_vox(path).expect("sponza.vox must load");
        assert!(!map.is_empty(), "sponza must have solid bricks");
        assert!(map.len() < 5_000_000, "sponza brick count must be bounded: {}", map.len());
        assert!(reg.len() > 1, "sponza palette must be populated");
    }

    /// C2: a `MATL` `_emit` material lights its block. Palette entry 0 = an orange-ish colour, with an emissive
    /// material `{ _type:_emit, _emit:0.8, _flux:2 }` → strength `0.8 · 2^2 = 3.2`. The loaded
    /// `BlockRegistry::emissive(BlockId(1))` must be non-zero, PROPORTIONAL to `albedo_lin · 3.2 · EMISSIVE_SCALE`,
    /// and TINTED by the palette colour (the emissive hue tracks the entry). A second non-emissive block stays
    /// `[0,0,0]`, and `has_emitters()` flips true.
    #[test]
    fn matl_emissive_lights_block_proportional_and_tinted() {
        let orange = [220u8, 120, 30, 255]; // palette idx 0 → BlockId(1), the emitter
        let plain = [40u8, 40, 200, 255]; // palette idx 1 → BlockId(2), non-emissive
        let voxels = vec![Voxel { x: 0, y: 0, z: 0, i: 0 }, Voxel { x: 1, y: 0, z: 0, i: 1 }];
        let data = make_vox_with_materials(
            (2, 1, 1),
            voxels,
            &[orange, plain],
            vec![emit_material(0, 0.8, 2.0)],
        );
        let (_map, reg) = from_dot_vox(&data);

        // The emitter's emissive equals its own linear colour × strength (0.8·2^2 = 3.2) × EMISSIVE_SCALE.
        let strength = 0.8f32 * 2f32.powi(2); // 3.2
        let albedo = srgb_u8_to_linear(orange);
        let expected = [
            albedo[0] * strength * EMISSIVE_SCALE,
            albedo[1] * strength * EMISSIVE_SCALE,
            albedo[2] * strength * EMISSIVE_SCALE,
        ];
        let got = reg.emissive(BlockId(1));
        assert!(got != [0.0, 0.0, 0.0], "emitter block must be emissive: {got:?}");
        for k in 0..3 {
            assert!((got[k] - expected[k]).abs() < 1e-5, "emissive[{k}] {} != expected {} ", got[k], expected[k]);
        }
        // Tinted by the palette colour: the orange entry is red-dominant, so emissive red > green > blue.
        assert!(got[0] > got[1] && got[1] > got[2], "emissive must track the orange palette hue: {got:?}");

        // The non-emissive block stays dark, and the scene now reports emitters present.
        assert_eq!(reg.emissive(BlockId(2)), [0.0, 0.0, 0.0], "non-emissive block stays dark");
        assert!(reg.has_emitters(), "has_emitters flips true with an emitter present");
    }

    /// C2: a material with no `_emit` (or `_emit == 0`) leaves its block non-emissive, and a scene with only
    /// such materials reports no emitters.
    #[test]
    fn matl_zero_or_absent_emit_is_not_emissive() {
        let color = [200u8, 30, 20, 255];
        let voxels = vec![Voxel { x: 0, y: 0, z: 0, i: 0 }];

        // `_emit == 0` → not an emitter.
        let zero = make_vox_with_materials((1, 1, 1), voxels.clone(), &[color], vec![emit_material(0, 0.0, 2.0)]);
        let (_m, reg) = from_dot_vox(&zero);
        assert_eq!(reg.emissive(BlockId(1)), [0.0, 0.0, 0.0], "_emit==0 is not emissive");
        assert!(!reg.has_emitters(), "no emitters when _emit==0");

        // A material with NO `_emit` property at all (a plain diffuse) → not an emitter.
        let props = [("_type".to_string(), "_diffuse".to_string())].into_iter().collect();
        let diffuse = make_vox_with_materials(
            (1, 1, 1),
            voxels,
            &[color],
            vec![Material { id: 0, properties: props }],
        );
        let (_m2, reg2) = from_dot_vox(&diffuse);
        assert_eq!(reg2.emissive(BlockId(1)), [0.0, 0.0, 0.0], "diffuse (no _emit) is not emissive");
        assert!(!reg2.has_emitters(), "no emitters for a diffuse-only scene");
    }

    /// C2: an out-of-range / malformed `mat.id` does NOT panic. An id `>= 256` (beyond the palette) and an id
    /// referencing the AIR-mapped slot are both handled by the loader's `< 256` guard + `set_emissive`'s
    /// no-op-on-out-of-range guard — the load completes and emits no spurious emitter.
    #[test]
    fn matl_out_of_range_id_does_not_panic() {
        let color = [255u8, 255, 255, 255];
        let voxels = vec![Voxel { x: 0, y: 0, z: 0, i: 0 }];
        // id 999 is well past the 256-entry palette; id 300 too. Both must be skipped, no panic.
        let data = make_vox_with_materials(
            (1, 1, 1),
            voxels,
            &[color],
            vec![emit_material(999, 1.0, 1.0), emit_material(300, 0.5, 0.0)],
        );
        let (_map, reg) = from_dot_vox(&data); // must not panic
        // The real block stays non-emissive (the bogus ids never touched it), and no emitters were created.
        assert_eq!(reg.emissive(BlockId(1)), [0.0, 0.0, 0.0], "out-of-range material must not light a block");
        assert!(!reg.has_emitters(), "out-of-range emitter ids produce no emitters");
    }
}
