//! The voxel block registry — the SSOT mapping a [`BlockId`] to its display/PBR appearance, and the
//! single bridge from a worldgen [`TerrainMatId`] to a [`BlockId`].
//!
//! Stage 1 is a thin proof of voxelization, so the registry is built DIRECTLY from the worldgen
//! [`BiomeLibrary`] palette: block `i+1` mirrors `TerrainMatId(i)` and takes its
//! [`preview_color`](crate::sdf_render::worldgen::biome::TerrainSurfaceMaterial::preview_color)
//! (linear RGBA), roughness, and texture-presence. Block `0` is reserved for AIR. Because the registry
//! is generated from the worldgen library, there is exactly ONE source of truth for the
//! `TerrainMatId → BlockId → colour` chain — they cannot drift.
//!
//! AUTHORING CHOICE (Stage 1): the registry is CODE-BUILT from the live worldgen palette rather than a
//! standalone `blocks.ron`. A RON asset would duplicate the colours already authored in
//! `assets/worldgen/biomes.ron`; deriving them keeps the single source of truth and dodges async-load
//! ordering for the first visual proof. `assets/voxel/blocks.ron` ships as documentation of the intended
//! schema; swap to loading it in a later stage if blocks need to diverge from terrain materials.

use bevy::prelude::*;
use rustc_hash::FxHashMap;

use crate::sdf_render::worldgen::biome::{BiomeLibrary, TerrainMatId};

/// Convert ONE sRGB-encoded channel byte (`0..=255`) to a LINEAR `[0,1]` value via the standard sRGB EOTF
/// (the IEC 61966-2-1 piecewise curve Bevy's `Color::srgb` uses). MagicaVoxel `.vox` palettes store sRGB
/// `u8`; the voxel registry stores LINEAR colour, so the loader must decode here. The single SSOT for the
/// `.vox` sRGB→linear conversion (shared by the loader + its tests).
#[inline]
pub fn srgb_channel_to_linear(byte: u8) -> f32 {
    let c = byte as f32 / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// Convert an sRGB-`u8` RGBA colour (the `.vox` palette convention) to LINEAR RGBA. RGB go through the sRGB
/// EOTF ([`srgb_channel_to_linear`]); ALPHA is linear by convention (sRGB encodes only the colour channels),
/// so it is just normalized to `[0,1]`. Matches the linear-RGBA colours the rest of the registry stores.
#[inline]
pub fn srgb_u8_to_linear(c: [u8; 4]) -> [f32; 4] {
    [
        srgb_channel_to_linear(c[0]),
        srgb_channel_to_linear(c[1]),
        srgb_channel_to_linear(c[2]),
        c[3] as f32 / 255.0,
    ]
}

/// A voxel block id. `0` is always AIR (empty space); `1..=N` are solid blocks. A `u16` so a world can
/// hold many distinct block types and a brick can store one per voxel cheaply.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u16);

impl BlockId {
    /// The empty/air block — the absence of a voxel. Bricks store `AIR` for every empty voxel.
    pub const AIR: BlockId = BlockId(0);

    /// True iff this is the air block.
    #[inline]
    pub fn is_air(self) -> bool {
        self == BlockId::AIR
    }
}

/// The fixed block ids of the CORNELL-BOX palette ([`BlockRegistry::cornell`]) — the SSOT shared by the
/// palette builder and the geometry builder ([`super::cornell`]) so a colour and the voxel that uses it can
/// never drift. Block `0` is AIR; these start at `1`. `as u16` gives each its [`BlockId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum CornellBlock {
    /// Floor / ceiling / back wall — neutral white.
    White = 1,
    /// Left wall (−X) — red.
    Red = 2,
    /// Right wall (+X) — green.
    Green = 3,
    /// Ceiling light panel — emissive white (the area light).
    Light = 4,
}

impl CornellBlock {
    /// Number of Cornell blocks (excluding AIR).
    pub const COUNT: usize = 4;

    /// This Cornell block's [`BlockId`].
    #[inline]
    pub fn id(self) -> BlockId {
        BlockId(self as u16)
    }
}

/// One block type's appearance + simple material params. `color` is **linear** RGBA (matching the
/// worldgen material `base_color`). `emissive` is linear RGB radiance; `tintable` flags whether a future
/// per-instance tint may modulate the base colour (unused visually in Stage 1, carried for the schema).
#[derive(Clone, Debug, PartialEq)]
pub struct BlockDef {
    pub name: String,
    pub color: [f32; 4],
    pub roughness: f32,
    pub metal: f32,
    pub emissive: [f32; 3],
    pub tintable: bool,
}

impl BlockDef {
    /// The AIR block definition — fully transparent, inert. Never rendered.
    fn air() -> Self {
        Self {
            name: "air".into(),
            color: [0.0, 0.0, 0.0, 0.0],
            roughness: 1.0,
            metal: 0.0,
            emissive: [0.0, 0.0, 0.0],
            tintable: false,
        }
    }
}

/// The block registry resource: `BlockId(i)` indexes `blocks[i]`, with `blocks[0]` == AIR. Built once
/// from the worldgen [`BiomeLibrary`] via [`BlockRegistry::from_biome_library`], which also records the
/// `TerrainMatId → BlockId` map so voxelization can translate worldgen materials to blocks.
#[derive(Resource, Clone, Debug)]
pub struct BlockRegistry {
    /// `BlockId(i)` → its definition. `blocks[0]` is always AIR.
    blocks: Vec<BlockDef>,
    /// `TerrainMatId(i)` → the `BlockId` mirroring it. Length == the worldgen palette length.
    mat_to_block: FxHashMap<TerrainMatId, BlockId>,
}

impl BlockRegistry {
    /// Build the registry from a worldgen [`BiomeLibrary`]: block `0` = AIR, then one block per palette
    /// material (id `i+1`), copying its `preview_color` (linear RGBA), roughness, metal=0 (terrain materials
    /// are dielectric this stage), and its `emissive_radiance` (`emissive_color * emissive_intensity`, the
    /// material SSOT) — so an emissive terrain material (lava, glowing crystal) makes its voxels GI light
    /// sources, exactly as the Cornell light panel does via [`set_emissive`]. Records the `TerrainMatId(i) →
    /// BlockId(i+1)` mapping — the SSOT the voxelizer uses. Robust to an EMPTY library (no materials yet →
    /// only the AIR block); the voxelizer then maps every material to AIR, producing an all-air (empty) world
    /// rather than panicking.
    pub fn from_biome_library(lib: &BiomeLibrary) -> Self {
        let mut blocks = Vec::with_capacity(lib.materials.len() + 1);
        let mut mat_to_block = FxHashMap::default();
        blocks.push(BlockDef::air());
        for (i, m) in lib.materials.iter().enumerate() {
            let id = BlockId(blocks.len() as u16);
            blocks.push(BlockDef {
                name: m.name.clone(),
                color: m.preview_color(),
                roughness: m.roughness,
                metal: 0.0,
                // The material's emissive radiance (0 for the common non-emitter) — a non-zero value makes
                // every voxel of this block an emitter the GI bounce gathers. Same SSOT the preview/shader use.
                emissive: m.emissive_radiance(),
                tintable: true,
            });
            mat_to_block.insert(TerrainMatId(i as u16), id);
        }
        Self { blocks, mat_to_block }
    }

    /// Build the CORNELL-BOX palette — the SSOT block set for the static Cornell scene
    /// ([`super::cornell`]). Independent of worldgen (no [`BiomeLibrary`] / [`TerrainMatId`]): the Cornell
    /// path never touches the terrain chain. Block ids are fixed by [`CornellBlock`] so the geometry builder
    /// and the palette agree by construction. Colours are LINEAR (the classic Cornell-box reflectances), and
    /// the light panel block carries a bright emissive so `emissive_strength × emissive` lights the room.
    pub fn cornell() -> Self {
        let mut blocks = Vec::with_capacity(CornellBlock::COUNT + 1);
        blocks.push(BlockDef::air()); // block 0 = AIR
        let opaque = |name: &str, c: [f32; 3]| BlockDef {
            name: name.into(),
            color: [c[0], c[1], c[2], 1.0],
            roughness: 1.0,
            metal: 0.0,
            emissive: [0.0, 0.0, 0.0],
            tintable: false,
        };
        // The order MUST match `CornellBlock`'s discriminants (white=1, red=2, green=3, light=4).
        blocks.push(opaque("cornell_white", [0.73, 0.73, 0.73]));
        blocks.push(opaque("cornell_red", [0.63, 0.065, 0.05]));
        blocks.push(opaque("cornell_green", [0.14, 0.45, 0.091]));
        // The ceiling light: white albedo + a bright emissive (the GI area light).
        blocks.push(BlockDef {
            name: "cornell_light".into(),
            color: [0.78, 0.78, 0.78, 1.0],
            roughness: 1.0,
            metal: 0.0,
            emissive: [1.0, 1.0, 1.0],
            tintable: false,
        });
        debug_assert_eq!(blocks.len(), CornellBlock::COUNT + 1, "cornell palette length must match CornellBlock");
        // The Cornell palette has no worldgen-material bridge (it isn't built from a library).
        Self { blocks, mat_to_block: FxHashMap::default() }
    }

    /// Build a registry DIRECTLY from a raw `.vox`-style palette of sRGB-`u8` RGBA colours — the SSOT for a
    /// baked static scene loaded by [`super::vox::load_vox`] (Sponza et al.). Block `0` = AIR, then one
    /// opaque dielectric block per palette entry (`BlockId(i+1)` mirrors `colors[i]`). Each colour is
    /// converted from sRGB-`u8` (the `.vox`/MagicaVoxel convention) to LINEAR RGBA via [`srgb_u8_to_linear`]
    /// so it matches every other registry colour (which are linear). No worldgen-material bridge (a baked
    /// scene has no [`TerrainMatId`] chain), so `mat_to_block` is empty. Robust to an empty palette (only the
    /// AIR block). The alpha byte is carried through (sRGB→linear is applied to RGB only; alpha is linear).
    pub fn from_vox_palette(colors: &[[u8; 4]]) -> Self {
        let mut blocks = Vec::with_capacity(colors.len() + 1);
        blocks.push(BlockDef::air()); // block 0 = AIR
        for (i, &c) in colors.iter().enumerate() {
            let lin = srgb_u8_to_linear(c);
            blocks.push(BlockDef {
                name: format!("vox_{i}"),
                color: lin,
                roughness: 1.0,
                metal: 0.0,
                emissive: [0.0, 0.0, 0.0],
                tintable: false,
            });
        }
        Self { blocks, mat_to_block: FxHashMap::default() }
    }

    /// Build a registry from a `.vxo` `MATL` table (`super::vxo::format::VxoMaterial`) — the SSOT for a baked
    /// `.vxo` static scene loaded by [`super::vxo`] (sibling to [`from_vox_palette`](Self::from_vox_palette),
    /// which builds from a raw `.vox` sRGB palette). Entry `i` → `BlockId(i)` DIRECTLY (NO `+1` offset: the
    /// `.vxo` table already includes entry 0 = AIR), with colours taken VERBATIM (the `.vxo` stores LINEAR
    /// RGBA + linear emissive, unlike `.vox`'s sRGB, so no conversion). `roughness`/`metallic`/`tintable`
    /// round-trip from the table. No worldgen-material bridge (a baked asset has no [`TerrainMatId`] chain).
    /// Robust to an empty table (no AIR entry ⇒ falls back to an AIR-only registry so callers never panic).
    ///
    /// **v1 write-reserved-but-not-rebuilt fields (§B1.2, an honest round-trip gap).** [`BlockDef`] has no field
    /// for these, so three `VxoMaterial` members are NOT reconstructed into the rebuilt registry:
    /// * `emissive.w` (the emissive_strength multiplier) — the writer stores `1.0` and `BlockDef::emissive` is the
    ///   already-multiplied radiance, so a non-`1.0` strength is not separately recoverable. (v1 always writes
    ///   `1.0`, so there is no loss today; a future per-block strength needs a `BlockDef` field + a `head_version`
    ///   bump.)
    /// * `MATL_FLAG_EMITTER` — recomputed from `emissive != 0` ([`Self::has_emitters`]/[`Self::emissive`]), the
    ///   SAME SSOT the writer set it from, so reading the stored bit would be redundant. It stays on disk as a
    ///   cheap pre-baked hint for a future fast-path loader.
    /// * the per-block `name` — `MATL` has no name field, so `BlockDef::name` is rebuilt EMPTY (debug-only, not
    ///   load-bearing for trace/GI).
    pub fn from_vxo_matl(mats: &[super::vxo::format::VxoMaterial]) -> Self {
        if mats.is_empty() {
            return Self::air_only();
        }
        let blocks = mats
            .iter()
            .map(|m| BlockDef {
                name: String::new(),
                color: m.albedo,
                roughness: m.roughness,
                metal: m.metallic,
                emissive: [m.emissive[0], m.emissive[1], m.emissive[2]],
                tintable: m.flags & super::vxo::format::MATL_FLAG_TINTABLE != 0,
            })
            .collect();
        Self { blocks, mat_to_block: FxHashMap::default() }
    }

    /// An AIR-only registry — just the AIR block (id 0), no solid blocks. The empty starting point the GALLERY
    /// merge ([`super::gallery::load_gallery`]) extends one scene's palette at a time onto via
    /// [`extend_blocks_from`](Self::extend_blocks_from). No worldgen-material bridge (a merged static scene has
    /// no [`TerrainMatId`] chain). Equivalent to [`from_vox_palette`](Self::from_vox_palette)`(&[])` but named
    /// for intent.
    pub fn air_only() -> Self {
        Self { blocks: vec![BlockDef::air()], mat_to_block: FxHashMap::default() }
    }

    /// APPEND every SOLID block of `other` (its blocks `1..`, i.e. excluding AIR) onto `self`, preserving each
    /// block's full appearance (colour, roughness, metal, emissive, tintable). The SSOT for the GALLERY palette
    /// concat: with `palette_base = self.len()` BEFORE the call, `other`'s first solid block (local
    /// [`BlockId`] `1`) is appended at merged index `palette_base`, so after this call `other`'s local block
    /// `b` (`b >= 1`) is `self`'s block at merged id `palette_base - 1 + b`. AIR is shared (block 0) and never
    /// duplicated. The caller is responsible for keeping the total `<= u16::MAX` (a [`BlockId`] is a `u16`).
    pub fn extend_blocks_from(&mut self, other: &BlockRegistry) {
        // Skip `other`'s AIR (index 0); append its solid blocks verbatim so colours/materials are preserved.
        self.blocks.extend(other.blocks.iter().skip(1).cloned());
    }

    /// The block a worldgen [`TerrainMatId`] maps to (the single bridge). An unknown id (e.g. a library
    /// that grew after the registry was built) maps to AIR — a robust default that simply leaves that
    /// voxel empty rather than indexing out of range.
    #[inline]
    pub fn block_for_material(&self, mat: TerrainMatId) -> BlockId {
        self.mat_to_block.get(&mat).copied().unwrap_or(BlockId::AIR)
    }

    /// The definition for `id` (AIR for `BlockId(0)`). An out-of-range id (shouldn't occur — ids come
    /// from this registry) resolves to AIR so callers never panic.
    #[inline]
    pub fn block(&self, id: BlockId) -> &BlockDef {
        self.blocks.get(id.0 as usize).unwrap_or(&self.blocks[0])
    }

    /// The linear-RGBA colour for `id` (AIR is fully transparent black).
    #[inline]
    pub fn color(&self, id: BlockId) -> [f32; 4] {
        self.block(id).color
    }

    /// The linear-RGB emissive radiance for `id` (`[0,0,0]` = non-emitter). A non-zero emissive makes the
    /// block a GI light source (the bounce returns this as out-radiance). AIR is never emissive.
    #[inline]
    pub fn emissive(&self, id: BlockId) -> [f32; 3] {
        self.block(id).emissive
    }

    /// Set the linear-RGB emissive radiance for `id` (the SSOT mutator for making a block glow). No-op for
    /// AIR or an out-of-range id. Used to author emitter blocks (and by the GI test oracle).
    #[inline]
    pub fn set_emissive(&mut self, id: BlockId, emissive: [f32; 3]) {
        if !id.is_air()
            && let Some(b) = self.blocks.get_mut(id.0 as usize)
        {
            b.emissive = emissive;
        }
    }

    /// Number of registered blocks (including AIR).
    #[inline]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// True iff ANY registered block has non-zero emissive radiance — i.e. the scene CAN have NEE lights. When
    /// false (the common worldgen-terrain case: no glowing blocks) the per-brick air-exposed-emissive gather
    /// can be SKIPPED entirely (it would find nothing), turning the O(resident) light pass into an O(1) check.
    /// The incremental `snapshot_patch` uses this so a non-emissive scene's per-move re-pack stays O(changed).
    #[inline]
    pub fn has_emitters(&self) -> bool {
        self.blocks.iter().any(|b| b.emissive != [0.0, 0.0, 0.0])
    }

    /// True iff only the AIR block is registered (an empty world palette).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.blocks.len() <= 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_render::worldgen::biome::{BiomeDef, BiomeId, BiomeLibrary, StrataLayer, TerrainSurfaceMaterial};

    /// A small worldgen library: a few flat materials + one trivial biome per `BiomeId` (enough to build
    /// a registry; the biome columns aren't read by the palette).
    fn test_library() -> BiomeLibrary {
        let mat = |name: &str, color: [f32; 4]| TerrainSurfaceMaterial {
            name: name.into(),
            base_color: color,
            roughness: 0.9,
            blend: 0.0,
            texture: None,
            tiling: 4.0,
            ..Default::default()
        };
        let materials = vec![
            mat("grass", [0.05, 0.22, 0.04, 1.0]),
            mat("dirt", [0.12, 0.07, 0.03, 1.0]),
            mat("stone", [0.18, 0.18, 0.19, 1.0]),
        ];
        let biome = |surface: u16| BiomeDef {
            name: "b".into(),
            surface: TerrainMatId(surface),
            surface_rules: vec![],
            strata: vec![StrataLayer { material: TerrainMatId(surface), thickness: 1.0 }],
            bedrock: TerrainMatId(2),
        };
        let biomes = BiomeId::ALL.iter().map(|_| biome(0)).collect();
        BiomeLibrary { materials, biomes }
    }

    /// Air is block 0 and reports as air; the first real material maps to block 1.
    #[test]
    fn air_is_block_zero() {
        let reg = BlockRegistry::from_biome_library(&test_library());
        assert!(BlockId::AIR.is_air());
        assert_eq!(BlockId::AIR.0, 0);
        assert!(reg.block(BlockId::AIR).color[3] == 0.0, "air is transparent");
        assert_eq!(reg.block_for_material(TerrainMatId(0)), BlockId(1));
    }

    /// BlockId ↔ colour round-trips through the registry, and the colour equals the worldgen material's
    /// `preview_color` (the single source of truth — no divergence).
    #[test]
    fn blockid_color_round_trip() {
        let lib = test_library();
        let reg = BlockRegistry::from_biome_library(&lib);
        for (i, m) in lib.materials.iter().enumerate() {
            let id = reg.block_for_material(TerrainMatId(i as u16));
            assert!(!id.is_air(), "material {i} must map to a solid block");
            assert_eq!(reg.color(id), m.preview_color(), "block colour must equal worldgen preview_color");
        }
    }

    /// An unknown material id (beyond the palette) maps to AIR — robust, no panic / out-of-range.
    #[test]
    fn unknown_material_maps_to_air() {
        let reg = BlockRegistry::from_biome_library(&test_library());
        assert!(reg.block_for_material(TerrainMatId(9999)).is_air());
    }

    /// An empty library yields a registry with only AIR (every material would map to air).
    #[test]
    fn empty_library_only_air() {
        let lib = BiomeLibrary::default();
        let reg = BlockRegistry::from_biome_library(&lib);
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 1);
    }

    /// The Cornell palette has AIR + the four Cornell blocks, the `CornellBlock` ids index the right
    /// colours, and ONLY the light block is emissive (the area-light SSOT).
    #[test]
    fn cornell_palette_blocks_and_emissive() {
        let reg = BlockRegistry::cornell();
        assert_eq!(reg.len(), CornellBlock::COUNT + 1, "AIR + 4 Cornell blocks");
        // Left/right walls are saturated red/green; white floor/ceiling is neutral and brighter.
        let red = reg.color(CornellBlock::Red.id());
        assert!(red[0] > red[1] && red[0] > red[2], "red wall is red-dominant: {red:?}");
        let green = reg.color(CornellBlock::Green.id());
        assert!(green[1] > green[0] && green[1] > green[2], "green wall is green-dominant: {green:?}");
        let white = reg.color(CornellBlock::White.id());
        assert!(white[0] > 0.5 && (white[0] - white[1]).abs() < 0.01 && (white[1] - white[2]).abs() < 0.01,
            "white is bright neutral: {white:?}");
        // Only the light panel glows.
        assert_eq!(reg.emissive(CornellBlock::White.id()), [0.0, 0.0, 0.0]);
        assert_eq!(reg.emissive(CornellBlock::Red.id()), [0.0, 0.0, 0.0]);
        assert_eq!(reg.emissive(CornellBlock::Green.id()), [0.0, 0.0, 0.0]);
        let e = reg.emissive(CornellBlock::Light.id());
        assert!(e[0] > 0.0 && e[1] > 0.0 && e[2] > 0.0, "light panel is emissive: {e:?}");
    }
}
