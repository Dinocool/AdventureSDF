//! DETAIL-NORMAL terrain material — **per-chunk baked normal-map PBR** for TERRAIN-ONLY coarse-LOD chunks
//! (Zylann-style "detail rendering"). A dedicated `ExtendedMaterial<StandardMaterial, TerrainDetailExt>`,
//! ONE per chunk: the bake samples the FINE (mip-0-scale) band-limited surface slope `(dh/dx, dh/dz)` at a
//! dense `N×N` grid over the chunk's world-XZ footprint (far finer than the coarse mesh's vertices) and
//! stores it in an `Rg16Float` texture. The fragment ([`terrain_detail.wgsl`]) reconstructs the hi-fi
//! surface normal `N = normalize(-dh/dx, 1, -dh/dz)` PER PIXEL and feeds Bevy PBR, so a low-poly distant
//! chunk SHADES as if it had the fine relief its averaged geometry lacks. Terrain is a HEIGHTFIELD
//! (`y - h(x,z)`), so a top-down PLANAR projection (world XZ → UV) is exact — no triplanar/atlas needed.
//!
//! Lifecycle: the per-chunk `Image` + `TerrainMaterial` are created on COMMIT ([`spawn_terrain_chunk`]) and
//! held by STRONG handles on the chunk entity (`MeshMaterial3d` + a [`TerrainDetailAssets`] component). When
//! the chunk entity despawns (evict/rebuild) those components drop, dropping the only strong handles, so
//! Bevy frees both assets — no leak, the same ref-counted lifecycle the shared mesh material relies on.
//!
//! GATING + cost is the bake's job (`mesh_bake`): only terrain-only chunks coarser than the clipmap's finest
//! node spacing get a baked map; near chunks already have full geometric detail and keep the shared material.

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat};
use bevy::shader::ShaderRef;
use half::f16;

/// The terrain detail-normal material: StandardMaterial (PBR lighting) extended with the per-chunk baked
/// surface-slope normal map.
pub type TerrainMaterial = ExtendedMaterial<StandardMaterial, TerrainDetailExt>;

/// Linear base colour for terrain (a muted green); roughness is set on the StandardMaterial base.
pub const TERRAIN_GREEN: Vec3 = Vec3::new(0.16, 0.27, 0.10);
/// Terrain PBR roughness (near-fully-rough ground).
pub const TERRAIN_ROUGHNESS: f32 = 0.95;

/// Per-draw uniform for the detail-normal extension. 32 B, two cleanly 16-aligned blocks: `{chunk_min(vec2),
/// chunk_size, strength}` then `flags(vec4<u32>)` (`.x` = debug-normals). A trailing `vec4<u32>` (not a
/// `u32` + `vec3` pad) avoids encase's 16-byte-alignment panic on a misaligned `vec3<u32>` member.
#[derive(ShaderType, Clone, Copy, Default, Debug, Reflect)]
pub struct TerrainDetailParams {
    /// World-XZ minimum corner of the chunk's footprint (the detail map covers `[chunk_min, chunk_min + size]`).
    pub chunk_min: Vec2,
    /// World-XZ edge length of the chunk's (square) footprint, in metres.
    pub chunk_size: f32,
    /// Detail-normal blend strength in `[0, 1]`: 0 = pure geometry normal, 1 = pure baked hi-fi detail normal.
    pub strength: f32,
    /// `.x` = 1 for "View normals" debug (unlit, the applied world-normal as RGB), else 0 (lit PBR); `.yzw`
    /// pad the block to a clean 16-byte tail.
    pub flags: UVec4,
}

/// The detail-normal extension: the per-chunk `Rg16Float` slope map + a clamp/linear sampler + the params.
/// `TypePath` (not full `Reflect`) satisfies `Asset` without forcing `Reflect` on the bound handle.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
pub struct TerrainDetailExt {
    #[uniform(100)]
    pub params: TerrainDetailParams,
    /// The baked `Rg16Float` surface-slope map; `.rg = (dh/dx, dh/dz)` at the texel's world-XZ position.
    #[texture(101)]
    #[sampler(102)]
    pub detail_normal: Handle<Image>,
}

impl MaterialExtension for TerrainDetailExt {
    fn fragment_shader() -> ShaderRef {
        "shaders/terrain_detail.wgsl".into()
    }
    // Forward-only fragment (no PREPASS branch) → keep it out of the depth/normal prepass pipeline, matching
    // the shared mesh material.
    fn enable_prepass() -> bool {
        false
    }
}

/// Strong handles to a terrain chunk's per-chunk detail-normal assets, parked on the chunk ENTITY so they
/// are freed when the entity despawns. The `Image` is ALSO referenced by the material's bind group, but
/// holding it here makes the ownership explicit and robust-by-construction: despawn drops this component,
/// dropping the only strong handles to BOTH the per-chunk material and its image → Bevy frees them. No leak.
#[derive(Component)]
pub struct TerrainDetailAssets {
    pub material: Handle<TerrainMaterial>,
    pub image: Handle<Image>,
}

/// The CPU-side baked detail-normal payload a terrain-only chunk's mesh task produces (off-thread), carried
/// in `ChunkMeshData` to the main-thread commit. The commit turns it into the `Image` + per-chunk
/// `TerrainMaterial`. `texels` is row-major `N×N` packed `[r=f16(dh/dx), g=f16(dh/dz)]` little-endian bytes
/// (an `Rg16Float` texture), exactly `N·N·4` bytes.
pub struct DetailNormalBake {
    /// Texel resolution per axis (`N`).
    pub res: u32,
    /// World-XZ minimum corner of the chunk's footprint.
    pub chunk_min: Vec2,
    /// World-XZ edge length of the chunk's (square) footprint.
    pub chunk_size: f32,
    /// Row-major `Rg16Float` bytes: `N·N` texels × `(f16, f16)` little-endian = `N·N·4` bytes.
    pub texels: Vec<u8>,
}

impl DetailNormalBake {
    /// Pack one texel's surface slope `(dh/dx, dh/dz)` into the `Rg16Float` little-endian byte pair the
    /// texture expects. The single SSOT both the bake and the unit test use, so the on-GPU layout can't
    /// drift from what the bake writes.
    #[inline]
    pub fn pack_texel(dhdx: f32, dhdz: f32) -> [u8; 4] {
        let r = f16::from_f32(dhdx).to_bits().to_le_bytes();
        let g = f16::from_f32(dhdz).to_bits().to_le_bytes();
        [r[0], r[1], g[0], g[1]]
    }
}

/// A clamp-to-edge, linear-filtered sampler for the detail-normal map: clamp so a fragment at the chunk
/// border reads the edge texel (never wraps); linear so the per-texel slope interpolates smoothly across
/// the dense grid (no blocky normal facets between texels).
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

/// Build the per-chunk `Rg16Float` detail-normal `Image` from a baked payload (clamp/linear sampler).
pub fn make_detail_image(bake: &DetailNormalBake) -> Image {
    let mut img = Image::new(
        Extent3d { width: bake.res, height: bake.res, depth_or_array_layers: 1 },
        TextureDimension::D2,
        bake.texels.clone(),
        TextureFormat::Rg16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    img.sampler = detail_sampler();
    img
}

/// Build a per-chunk `TerrainMaterial` from the chunk's detail-normal image + footprint + the live strength.
pub fn make_terrain_material(
    image: Handle<Image>,
    bake: &DetailNormalBake,
    strength: f32,
    debug_normals: bool,
) -> TerrainMaterial {
    TerrainMaterial {
        base: StandardMaterial {
            base_color: Color::linear_rgb(TERRAIN_GREEN.x, TERRAIN_GREEN.y, TERRAIN_GREEN.z),
            perceptual_roughness: TERRAIN_ROUGHNESS,
            ..default()
        },
        extension: TerrainDetailExt {
            params: TerrainDetailParams {
                chunk_min: bake.chunk_min,
                chunk_size: bake.chunk_size,
                strength,
                flags: UVec4::new(debug_normals as u32, 0, 0, 0),
            },
            detail_normal: image,
        },
    }
}

/// Plugin: registers the terrain detail-normal material pipeline. Added by `MeshBakePlugin` (the material is
/// gameplay-independent, baked by the mesh bake).
pub struct TerrainMaterialPlugin;

impl Plugin for TerrainMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<TerrainMaterial>::default())
            .register_type::<TerrainDetailParams>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `pack_texel` round-trips through f16 (the lossy step is f16's mantissa, not our packing): unpacking
    /// the little-endian byte pair back to f32 reproduces `f16::from_f32(x)` for both lanes, and the byte
    /// layout is `[r_lo, r_hi, g_lo, g_hi]` (Rg16Float). Pins the on-GPU layout the shader reads.
    #[test]
    fn pack_texel_roundtrips_through_f16() {
        for &(dx, dz) in &[(0.0f32, 0.0f32), (0.5, -0.25), (-2.0, 3.0), (0.001_5, -0.002_5)] {
            let b = DetailNormalBake::pack_texel(dx, dz);
            let r = f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
            let g = f16::from_bits(u16::from_le_bytes([b[2], b[3]])).to_f32();
            assert_eq!(r, f16::from_f32(dx).to_f32(), "r lane mismatch for {dx}");
            assert_eq!(g, f16::from_f32(dz).to_f32(), "g lane mismatch for {dz}");
        }
    }

    /// `make_detail_image` produces an `Rg16Float` texture of exactly `res×res` with `res·res·4` bytes.
    #[test]
    fn detail_image_has_expected_extent_and_format() {
        let res = 8u32;
        let texels = vec![0u8; (res * res * 4) as usize];
        let bake = DetailNormalBake { res, chunk_min: Vec2::ZERO, chunk_size: 256.0, texels };
        let img = make_detail_image(&bake);
        assert_eq!(img.texture_descriptor.format, TextureFormat::Rg16Float);
        assert_eq!(img.width(), res);
        assert_eq!(img.height(), res);
        assert_eq!(img.data.as_ref().map(|d| d.len()), Some((res * res * 4) as usize));
    }
}
