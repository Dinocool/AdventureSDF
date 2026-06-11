//! TERRAIN SURFACE material — **per-chunk baked volumetric biome strata + detail-normal PBR** for
//! TERRAIN-ONLY chunks (Stages 2+3 of the terrain-materials feature; see `docs/TERRAIN_MATERIALS_PLAN.md`).
//! A dedicated `ExtendedMaterial<StandardMaterial, TerrainSurfaceExt>`, ONE per terrain-only chunk, whose
//! fragment ([`terrain_surface.wgsl`]) shades the surface by the biome's volumetric strata column:
//!
//!   `depth = original_surface_height(world.xz) − world.y` → biome strata lookup → flat base colour
//!   (grass → dirt → stone → bedrock by depth; biome-dependent), + surface treatment (snow/sand/rock) for
//!   the top layer, + the baked hi-fi detail normal, + Bevy PBR.
//!
//! The per-chunk bake ([`TerrainSurfaceBake`]) produces, over the chunk's world-XZ footprint:
//! - **detail-normal** (`Rg16Float`, `detail_res²`): the fine band-limited surface slope `(dh/dx, dh/dz)` —
//!   reconstructs the hi-fi normal per pixel. GATED to COARSE chunks (near chunks already carry the relief).
//! - **surface-height** (`R32Float`, `detail_res²`): the PRISTINE `sample_world` height `h(x,z)` — the depth
//!   reference (`depth = surf_h − world.y`). Baked on EVERY terrain-only chunk (essentially free — the same
//!   `sample_world` eval that yields the slope also yields the height).
//! - **biome** (`Rgba16Float`, `biome_res²`, low-res — biome is low-frequency): primary id, secondary id,
//!   blend — the Stage-1 Whittaker classifier per texel (no Whittaker logic in WGSL).
//!
//! The flattened per-biome **strata GPU table** ([`biome::StrataTableStd`]) is carried in the material's
//! uniform — the SAME shared flatten the editor biome/slice preview uploads (one SSOT, no duplication).
//!
//! Lifecycle: the per-chunk `Image`s + `TerrainMaterial` are created on COMMIT and held by STRONG handles on
//! the chunk entity (`MeshMaterial3d` + [`TerrainDetailAssets`]); despawn drops them → Bevy frees the assets.

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat};
use bevy::shader::ShaderRef;
use half::f16;

use super::worldgen::biome::{MaterialPaletteStd, StrataTableStd};

/// The terrain surface material: StandardMaterial (PBR lighting) extended with the per-chunk baked
/// strata/biome/height/detail-normal maps + the biome strata table.
pub type TerrainMaterial = ExtendedMaterial<StandardMaterial, TerrainSurfaceExt>;

/// Terrain PBR roughness fallback (near-fully-rough ground). The per-fragment base colour comes from the
/// biome strata table; roughness is per-material in the strata RON (Stage 5 will sample it per-fragment),
/// so this is just the StandardMaterial base default.
pub const TERRAIN_ROUGHNESS: f32 = 0.95;

/// Per-draw uniform for the terrain-surface extension. `{chunk_min(vec2), chunk_size, strength}` then
/// `flags(vec4<u32>)` then surface-treatment knobs as `vec4`s — every member 16-aligned (a trailing
/// `vec4<u32>` not a `u32`+`vec3` pad avoids encase's alignment panic on a misaligned `vec3<u32>`).
#[derive(ShaderType, Clone, Copy, Debug, Reflect)]
pub struct TerrainSurfaceParams {
    /// World-XZ minimum corner of the chunk's footprint (all maps cover `[chunk_min, chunk_min + size]`).
    pub chunk_min: Vec2,
    /// World-XZ edge length of the chunk's (square) footprint, in metres.
    pub chunk_size: f32,
    /// Detail-normal blend strength in `[0, 1]`: 0 = pure geometry normal, 1 = pure baked hi-fi detail normal.
    pub strength: f32,
    /// `.x` = 1 for "View normals" debug (unlit, the applied world-normal as RGB), else lit PBR; `.y` = 1 to
    /// FORCE the height/detail-normal-only legacy look (debug, no strata); `.zw` pad.
    pub flags: UVec4,
    /// Surface-treatment thresholds (the top, undug layer):
    /// `.x` = rock slope-start (cos of the angle past which rock shows on steep ground; smaller = steeper),
    /// `.y` = rock slope-full (cos at which it's fully rock),
    /// `.z` = snow height-start (world Y above which snow accumulates in cold biomes),
    /// `.w` = snow height-full (world Y at which it's fully snow).
    pub surf_a: Vec4,
    /// More surface-treatment knobs:
    /// `.x` = sand height-band half-width below sea level (near-sea-level sand), `.y` = sea level (world Y),
    /// `.z` = surface-treatment master strength `[0,1]` (0 = pure strata surface colour, 1 = full treatment),
    /// `.w` = layer/biome boundary blend softness in metres (cross-fade band across strata boundaries).
    pub surf_b: Vec4,
}

impl Default for TerrainSurfaceParams {
    fn default() -> Self {
        Self {
            chunk_min: Vec2::ZERO,
            chunk_size: 1.0,
            strength: 1.0,
            flags: UVec4::ZERO,
            // Rock from ~60° (cos 0.5) to ~80° (cos 0.17); snow from y=900 to y=1300 (cold biomes only);
            // sand within 6 m of sea level; treatment full strength; 0.5 m boundary blend.
            surf_a: Vec4::new(0.5, 0.17, 900.0, 1300.0),
            surf_b: Vec4::new(6.0, 0.0, 1.0, 0.5),
        }
    }
}

/// The terrain-surface extension: the per-chunk maps + samplers + params + the shared strata table.
/// `TypePath` (not full `Reflect`) satisfies `Asset` without forcing `Reflect` on the bound handles.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
pub struct TerrainSurfaceExt {
    #[uniform(100)]
    pub params: TerrainSurfaceParams,
    /// The baked `Rg16Float` surface-slope map; `.rg = (dh/dx, dh/dz)`. Zero-filled on fine chunks (the
    /// detail-normal LOD gate) so the shader falls back to the geometry normal.
    #[texture(101)]
    #[sampler(102)]
    pub detail_normal: Handle<Image>,
    /// The baked `R32Float` PRISTINE surface height `h(x,z)` (depth reference). Unfilterable float → manual
    /// bilinear in the shader (`textureLoad`).
    #[texture(103, sample_type = "float", filterable = false)]
    pub surface_height: Handle<Image>,
    /// The baked low-res `Rgba16Float` biome map: `R = primary id, G = secondary id, B = blend`. Nearest
    /// (`textureLoad`). Used for the VOLUMETRIC strata (dug walls) — the undug surface uses `surface_mat`.
    #[texture(104, sample_type = "float", filterable = false)]
    pub biome: Handle<Image>,
    /// The shared per-biome strata table (the SAME std140 flatten the editor preview uploads).
    #[uniform(105)]
    pub strata: StrataTableStd,
    /// The baked low-res `Rgba16Float` SURFACE-MATERIAL map: `R = mat_a id, G = mat_b id, B = weight` — the
    /// worldgen-resolved undug surface (biome base + altitude caps + cliffs + patches, all resolved at BAKE
    /// time by [`super::worldgen::biome::resolve_surface`]). Nearest (`textureLoad`); the shader resolves each
    /// texel to a palette colour then bilinear-interpolates the COLOUR (ids can't interpolate).
    #[texture(106, sample_type = "float", filterable = false)]
    pub surface_mat: Handle<Image>,
    /// The flat material palette (`TerrainMatId` → colour + roughness) the `surface_mat` ids index. Same SSOT
    /// flatten as the strata table; re-synced live on a `biomes.ron` edit.
    #[uniform(107)]
    pub palette: MaterialPaletteStd,
    /// Shared PBR texture arrays (one `texture_2d_array` layer per material, layer == `TerrainMatId`). The
    /// shader triplanar-samples the baked `mat_a`/`mat_b` layers; a material without a texture (`has_tex == 0`)
    /// uses its flat palette colour instead. All three share the sampler at binding 111.
    #[texture(108, dimension = "2d_array")]
    #[sampler(111)]
    pub diffuse_array: Handle<Image>,
    #[texture(109, dimension = "2d_array")]
    pub normal_array: Handle<Image>,
    #[texture(110, dimension = "2d_array")]
    pub mra_array: Handle<Image>,
}

impl MaterialExtension for TerrainSurfaceExt {
    fn fragment_shader() -> ShaderRef {
        "shaders/terrain_surface.wgsl".into()
    }
    // Forward-only fragment (no PREPASS branch) → keep it out of the depth/normal prepass pipeline, matching
    // the shared mesh material.
    fn enable_prepass() -> bool {
        false
    }
}

/// Strong handles to a terrain chunk's per-chunk surface assets, parked on the chunk ENTITY so they are
/// freed when the entity despawns. The `Image`s are ALSO referenced by the material's bind group, but
/// holding them here makes the ownership explicit + robust-by-construction: despawn drops this component,
/// dropping the only strong handles to the material + every per-chunk image → Bevy frees them. No leak.
#[derive(Component)]
pub struct TerrainDetailAssets {
    pub material: Handle<TerrainMaterial>,
    pub detail_normal: Handle<Image>,
    pub surface_height: Handle<Image>,
    pub biome: Handle<Image>,
    pub surface_mat: Handle<Image>,
}

/// The CPU-side baked terrain-surface payload a terrain-only chunk's mesh task produces (off-thread),
/// carried in `ChunkMeshData` to the main-thread commit. The commit turns it into the per-chunk `Image`s +
/// `TerrainMaterial`.
///
/// `detail_texels`: row-major `detail_res²` packed `[f16(dh/dx), f16(dh/dz)]` LE (`Rg16Float`); zero-filled
/// (all-zero slope → geometry normal) when the chunk is FINE (the detail-normal LOD gate). `height_texels`:
/// row-major `detail_res²` `f32` LE (`R32Float`) PRISTINE surface height. `biome_texels`: row-major
/// `biome_res²` × 4 `f16` LE (`Rgba16Float`) `(primary, secondary, blend, 0)`.
pub struct TerrainSurfaceBake {
    /// `true` iff the detail-normal slope was actually baked (a COARSE chunk); `false` ⇒ the detail texels
    /// are zero-filled (a FINE chunk, the LOD gate) and the shader must use the GEOMETRY normal, not the
    /// flattened detail normal. Drives `flags.z`.
    pub detail_present: bool,
    /// Detail-normal + surface-height texel resolution per axis.
    pub detail_res: u32,
    /// Biome map resolution per axis (low — biome is low-frequency).
    pub biome_res: u32,
    /// World-XZ minimum corner of the chunk's footprint.
    pub chunk_min: Vec2,
    /// World-XZ edge length of the chunk's (square) footprint.
    pub chunk_size: f32,
    /// `Rg16Float` detail-normal bytes (`detail_res² × 4`).
    pub detail_texels: Vec<u8>,
    /// `R32Float` surface-height bytes (`detail_res² × 4`).
    pub height_texels: Vec<u8>,
    /// `Rgba16Float` biome bytes (`biome_res² × 8`).
    pub biome_texels: Vec<u8>,
    /// `Rgba16Float` SURFACE-MATERIAL bytes (`biome_res² × 8`): `(mat_a, mat_b, weight, 0)` per texel, the
    /// worldgen-resolved undug surface (the data-driven replacement for the hardcoded shader treatment).
    pub surface_mat_texels: Vec<u8>,
}

impl TerrainSurfaceBake {
    /// Pack one texel's surface slope `(dh/dx, dh/dz)` into the `Rg16Float` little-endian byte pair. SSOT for
    /// both the bake and the test so the on-GPU layout can't drift.
    #[inline]
    pub fn pack_slope(dhdx: f32, dhdz: f32) -> [u8; 4] {
        let r = f16::from_f32(dhdx).to_bits().to_le_bytes();
        let g = f16::from_f32(dhdz).to_bits().to_le_bytes();
        [r[0], r[1], g[0], g[1]]
    }

    /// Pack one biome texel `(primary, secondary, blend, temperature)` into the `Rgba16Float` 8-byte group.
    /// Ids are small integers (0..=4) that f16 stores exactly; the 4th lane is the CONTINUOUS temperature
    /// `[0,1]` — bilinear-sampled in the shader so cold-driven SNOW blends smoothly everywhere (the discrete
    /// biome-id `cold` stepped with the climate gradient → hard snow edges on steep sides).
    #[inline]
    pub fn pack_biome(primary: u8, secondary: u8, blend: f32, temperature: f32) -> [u8; 8] {
        let p = f16::from_f32(primary as f32).to_bits().to_le_bytes();
        let s = f16::from_f32(secondary as f32).to_bits().to_le_bytes();
        let b = f16::from_f32(blend).to_bits().to_le_bytes();
        let t = f16::from_f32(temperature).to_bits().to_le_bytes();
        [p[0], p[1], s[0], s[1], b[0], b[1], t[0], t[1]]
    }

    /// Pack one surface-material texel `(mat_a, mat_b, weight)` into the `Rgba16Float` 8-byte group. Ids are
    /// small integers f16 stores exactly; `weight ∈ [0, 0.5]` is the fraction toward `mat_b`. SSOT for the
    /// bake + test so the on-GPU layout can't drift from the shader's `textureLoad`.
    #[inline]
    pub fn pack_surface(mat_a: u16, mat_b: u16, weight: f32) -> [u8; 8] {
        let a = f16::from_f32(mat_a as f32).to_bits().to_le_bytes();
        let b = f16::from_f32(mat_b as f32).to_bits().to_le_bytes();
        let w = f16::from_f32(weight).to_bits().to_le_bytes();
        let z = f16::from_f32(0.0).to_bits().to_le_bytes();
        [a[0], a[1], b[0], b[1], w[0], w[1], z[0], z[1]]
    }
}

/// A clamp-to-edge, LINEAR sampler for the detail-normal map (smooth per-texel slope interpolation).
fn detail_sampler() -> ImageSampler {
    ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        address_mode_w: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    })
}

/// Build the per-chunk `Rg16Float` detail-normal `Image` (clamp/linear sampler).
pub fn make_detail_image(bake: &TerrainSurfaceBake) -> Image {
    let mut img = Image::new(
        Extent3d { width: bake.detail_res, height: bake.detail_res, depth_or_array_layers: 1 },
        TextureDimension::D2,
        bake.detail_texels.clone(),
        TextureFormat::Rg16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    img.sampler = detail_sampler();
    img
}

/// Build the per-chunk `R32Float` surface-height `Image`. Unfilterable (the shader does manual bilinear via
/// `textureLoad`), so a nearest sampler suffices.
pub fn make_height_image(bake: &TerrainSurfaceBake) -> Image {
    let mut img = Image::new(
        Extent3d { width: bake.detail_res, height: bake.detail_res, depth_or_array_layers: 1 },
        TextureDimension::D2,
        bake.height_texels.clone(),
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    img.sampler = ImageSampler::nearest();
    img
}

/// Build the per-chunk low-res `Rgba16Float` biome `Image` (nearest — biome ids must not interpolate; the
/// shader blends primary↔secondary by the stored blend weight instead).
pub fn make_biome_image(bake: &TerrainSurfaceBake) -> Image {
    let mut img = Image::new(
        Extent3d { width: bake.biome_res, height: bake.biome_res, depth_or_array_layers: 1 },
        TextureDimension::D2,
        bake.biome_texels.clone(),
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    img.sampler = ImageSampler::nearest();
    img
}

/// Build the per-chunk low-res `Rgba16Float` SURFACE-MATERIAL `Image` (nearest — material ids must not
/// interpolate; the shader resolves each texel to a palette colour then bilinear-blends the colours).
pub fn make_surface_mat_image(bake: &TerrainSurfaceBake) -> Image {
    let mut img = Image::new(
        Extent3d { width: bake.biome_res, height: bake.biome_res, depth_or_array_layers: 1 },
        TextureDimension::D2,
        bake.surface_mat_texels.clone(),
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    img.sampler = ImageSampler::nearest();
    img
}

/// Build a per-chunk `TerrainMaterial` from the chunk's baked images + footprint + the live strength + the
/// shared strata table + material palette.
#[allow(clippy::too_many_arguments)]
pub fn make_terrain_material(
    detail_normal: Handle<Image>,
    surface_height: Handle<Image>,
    biome: Handle<Image>,
    surface_mat: Handle<Image>,
    bake: &TerrainSurfaceBake,
    strength: f32,
    debug_normals: bool,
    strata: StrataTableStd,
    palette: MaterialPaletteStd,
    arrays: (Handle<Image>, Handle<Image>, Handle<Image>),
) -> TerrainMaterial {
    TerrainMaterial {
        base: StandardMaterial {
            // The per-fragment base colour is the biome strata colour (set in-shader); the StandardMaterial
            // base colour is left white so PBR multiplies it through unchanged.
            base_color: Color::WHITE,
            perceptual_roughness: TERRAIN_ROUGHNESS,
            ..default()
        },
        extension: TerrainSurfaceExt {
            params: TerrainSurfaceParams {
                chunk_min: bake.chunk_min,
                chunk_size: bake.chunk_size,
                strength,
                // .x = debug normals, .z = detail-normal present (else use geometry normal).
                flags: UVec4::new(debug_normals as u32, 0, bake.detail_present as u32, 0),
                ..Default::default()
            },
            detail_normal,
            surface_height,
            biome,
            strata,
            surface_mat,
            palette,
            diffuse_array: arrays.0,
            normal_array: arrays.1,
            mra_array: arrays.2,
        },
    }
}

/// Plugin: registers the terrain surface material pipeline. Added by `MeshBakePlugin`.
pub struct TerrainMaterialPlugin;

impl Plugin for TerrainMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<TerrainMaterial>::default())
            .register_type::<TerrainSurfaceParams>()
            .init_resource::<super::terrain_textures::TerrainTextureArrays>()
            .add_systems(
                Update,
                (
                    super::terrain_textures::build_terrain_texture_arrays,
                    sync_terrain_texture_arrays.after(super::terrain_textures::build_terrain_texture_arrays),
                ),
            );
    }
}

/// Push the shared terrain texture-array handles into every live `TerrainMaterial` whenever the arrays
/// resource changes (the assembled arrays swapped in once their PNGs loaded, or a `biomes.ron` retexture
/// rebuilt them). New chunks already bake with the current handles (via the commit's `SpawnAssets`); this
/// keeps already-spawned chunks in sync — the texture analogue of `sync_terrain_detail_params`' strata push.
fn sync_terrain_texture_arrays(
    arrays: Res<super::terrain_textures::TerrainTextureArrays>,
    mut mats: ResMut<Assets<TerrainMaterial>>,
) {
    if !arrays.is_changed() {
        return;
    }
    let (d, n, m) = (arrays.diffuse.clone(), arrays.normal.clone(), arrays.mra.clone());
    let ids: Vec<_> = mats.iter().map(|(id, _)| id).collect();
    for id in ids {
        if let Some(mat) = mats.get_mut(id) {
            mat.extension.diffuse_array = d.clone();
            mat.extension.normal_array = n.clone();
            mat.extension.mra_array = m.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `pack_slope` round-trips through f16 (`[r_lo, r_hi, g_lo, g_hi]`, Rg16Float) — pins the on-GPU layout.
    #[test]
    fn pack_slope_roundtrips_through_f16() {
        for &(dx, dz) in &[(0.0f32, 0.0f32), (0.5, -0.25), (-2.0, 3.0), (0.001_5, -0.002_5)] {
            let b = TerrainSurfaceBake::pack_slope(dx, dz);
            let r = f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
            let g = f16::from_bits(u16::from_le_bytes([b[2], b[3]])).to_f32();
            assert_eq!(r, f16::from_f32(dx).to_f32(), "r lane mismatch for {dx}");
            assert_eq!(g, f16::from_f32(dz).to_f32(), "g lane mismatch for {dz}");
        }
    }

    /// `pack_biome` round-trips the (primary, secondary, blend) lanes through f16; ids store exactly.
    #[test]
    fn pack_biome_roundtrips_ids_exactly() {
        for &(p, s, blend, temp) in &[(0u8, 0u8, 0.0f32, 0.3f32), (2, 3, 0.5, 0.7), (4, 1, 1.0, 0.1)] {
            let b = TerrainSurfaceBake::pack_biome(p, s, blend, temp);
            let rp = f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
            let rs = f16::from_bits(u16::from_le_bytes([b[2], b[3]])).to_f32();
            assert_eq!(rp, p as f32, "primary id must store exactly");
            assert_eq!(rs, s as f32, "secondary id must store exactly");
            assert_eq!(f16::from_bits(u16::from_le_bytes([b[4], b[5]])).to_f32(), f16::from_f32(blend).to_f32());
            assert_eq!(f16::from_bits(u16::from_le_bytes([b[6], b[7]])).to_f32(), f16::from_f32(temp).to_f32());
        }
    }

    /// `TerrainSurfaceParams` must be a VALID std140 uniform (the `[u32; N]`-pad gotcha — fires only at
    /// encode/assert time, not in `--lib` type checks).
    #[test]
    fn surface_params_is_valid_std140_uniform() {
        TerrainSurfaceParams::assert_uniform_compat();
    }

    /// The terrain-surface shader's strata-table dimension consts MUST match the Rust/biome SSOT — the
    /// `strata` uniform is `[StrataColumn; BIOME_COUNT]` with `[Vec4; STRATA_MAX_LAYERS]` colours. A mismatch
    /// silently corrupts the in-world strata lookup; catch it at build time (like the preview shader's test).
    #[test]
    fn shader_strata_dims_match_rust() {
        use super::super::worldgen::biome::{BIOME_COUNT, GPU_MAX_MATERIALS, GPU_STRATA_MAX_LAYERS};
        let src = include_str!("../../assets/shaders/terrain_surface.wgsl");
        let uint_const = |name: &str| -> usize {
            let line = src
                .lines()
                .find(|l| l.contains(name) && l.contains("const"))
                .unwrap_or_else(|| panic!("terrain_surface.wgsl declares `const {name}`"));
            let rhs = line.split('=').nth(1).expect("const has an `= value`").trim();
            let digits: String = rhs.chars().take_while(|c| c.is_ascii_digit()).collect();
            digits.parse().unwrap_or_else(|_| panic!("`{name}` value is numeric, got `{rhs}`"))
        };
        assert_eq!(
            uint_const("STRATA_MAX_LAYERS"),
            GPU_STRATA_MAX_LAYERS,
            "STRATA_MAX_LAYERS in terrain_surface.wgsl != GPU_STRATA_MAX_LAYERS in biome.rs"
        );
        assert_eq!(
            uint_const("BIOME_COUNT"),
            BIOME_COUNT,
            "BIOME_COUNT in terrain_surface.wgsl != BiomeId::ALL.len()"
        );
        assert_eq!(
            uint_const("MAX_MATERIALS"),
            GPU_MAX_MATERIALS,
            "MAX_MATERIALS in terrain_surface.wgsl != GPU_MAX_MATERIALS in biome.rs"
        );
    }

    /// The baked images have the expected extents + formats.
    #[test]
    fn images_have_expected_extent_and_format() {
        let dres = 8u32;
        let bres = 4u32;
        let bake = TerrainSurfaceBake {
            detail_present: true,
            detail_res: dres,
            biome_res: bres,
            chunk_min: Vec2::ZERO,
            chunk_size: 256.0,
            detail_texels: vec![0u8; (dres * dres * 4) as usize],
            height_texels: vec![0u8; (dres * dres * 4) as usize],
            biome_texels: vec![0u8; (bres * bres * 8) as usize],
            surface_mat_texels: vec![0u8; (bres * bres * 8) as usize],
        };
        let dn = make_detail_image(&bake);
        assert_eq!(dn.texture_descriptor.format, TextureFormat::Rg16Float);
        assert_eq!((dn.width(), dn.height()), (dres, dres));
        let h = make_height_image(&bake);
        assert_eq!(h.texture_descriptor.format, TextureFormat::R32Float);
        assert_eq!((h.width(), h.height()), (dres, dres));
        let bi = make_biome_image(&bake);
        assert_eq!(bi.texture_descriptor.format, TextureFormat::Rgba16Float);
        assert_eq!((bi.width(), bi.height()), (bres, bres));
        let sm = make_surface_mat_image(&bake);
        assert_eq!(sm.texture_descriptor.format, TextureFormat::Rgba16Float);
        assert_eq!((sm.width(), sm.height()), (bres, bres));
    }

    /// `pack_surface` round-trips the ids exactly + the weight through f16 — pins the on-GPU layout.
    #[test]
    fn pack_surface_roundtrips() {
        for &(a, b, w) in &[(0u16, 0u16, 0.0f32), (3, 7, 0.5), (11, 2, 0.25)] {
            let p = TerrainSurfaceBake::pack_surface(a, b, w);
            let ra = f16::from_bits(u16::from_le_bytes([p[0], p[1]])).to_f32();
            let rb = f16::from_bits(u16::from_le_bytes([p[2], p[3]])).to_f32();
            let rw = f16::from_bits(u16::from_le_bytes([p[4], p[5]])).to_f32();
            assert_eq!(ra, a as f32, "mat_a id must store exactly");
            assert_eq!(rb, b as f32, "mat_b id must store exactly");
            assert_eq!(rw, f16::from_f32(w).to_f32(), "weight roundtrip");
        }
    }
}
