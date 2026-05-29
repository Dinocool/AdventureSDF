//! PBR texture-library loading: parse the `material.ron` manifests, build the
//! global [`MaterialRegistry`], and decode the role-named PNG maps into GPU texture
//! arrays.
//!
//! The library lives at `assets/textures/<slug>/<variant>/{diffuse,normal,metallic,
//! roughness,ao,height,edge}.png` with one `material.ron` per slug listing variants.
//! Each (slug, variant) pair becomes one registry material AND one layer in every
//! texture array — the layer index is the variant's position in [`TextureLibrary`],
//! so the main-world registry and the render-world arrays agree by construction.

use bevy::prelude::*;
use serde::Deserialize;

use super::edits::MATERIAL_TEX_MAPS;

/// Texture-array layer edge length. The importer emits 1024²; we resize on decode
/// so the arrays are uniform regardless of on-disk size. Uploaded as BC7 with a full
/// mip chain (see `encode_library_bc7`), so full 1024² res costs ~⅙ the VRAM of
/// uncompressed RGBA8.
pub const TEXTURE_SIZE: u32 = 1024;

/// Map-array order. Metallic, roughness, and ao are packed into one RGBA "MRA"
/// texture (`.r=metallic, .g=roughness, .b=ao`), so 7 source maps → 5 arrays. This
/// order is the layout the shader's `sample_material_map` map enum mirrors.
#[derive(Clone, Copy)]
pub enum MapArray {
    Diffuse = 0,
    Normal = 1,
    Mra = 2,
    Height = 3,
    Edge = 4,
}

pub const TEXTURE_ROOT: &str = "assets/textures";

/// One library material variant — the unit that becomes a registry entry + a layer.
#[derive(Clone, Debug)]
pub struct LibraryVariant {
    pub slug: String,
    pub dir: String,
    pub display_name: String,
}

// --- Manifest parsing (used by the Resource Inspector to list available textures) ---

/// Serde struct for a variant entry inside a `material.ron` manifest.
#[derive(Deserialize)]
pub struct ManifestVariant {
    #[allow(dead_code)]
    pub id: u32,
    pub dir: String,
}

/// Serde struct for a `assets/textures/<slug>/material.ron` file.
#[derive(Deserialize)]
pub struct Manifest {
    pub name: String,
    pub slug: String,
    pub variants: Vec<ManifestVariant>,
}

/// Parse the manifest at `assets/textures/<slug>/material.ron` and return its
/// variants as [`LibraryVariant`]s. Returns an empty vec and logs on error.
pub fn read_manifest(slug: &str) -> Vec<LibraryVariant> {
    let path = format!("{TEXTURE_ROOT}/{slug}/material.ron");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            warn!("SDF texture manifest: cannot read {path}: {e}");
            return Vec::new();
        }
    };
    let manifest: Manifest = match ron::from_str(&text) {
        Ok(m) => m,
        Err(e) => {
            warn!("SDF texture manifest: cannot parse {path}: {e}");
            return Vec::new();
        }
    };
    manifest
        .variants
        .iter()
        .map(|v| LibraryVariant {
            slug: manifest.slug.clone(),
            dir: v.dir.clone(),
            display_name: format!("{} {}", manifest.name, v.dir),
        })
        .collect()
}

/// BC7-compressed maps for ONE variant (one array layer): `[diffuse, normal, mra,
/// height, edge]`, each a single-layer full mip chain ready to `write_texture` into
/// the corresponding array's layer.
pub type VariantBc7 = [super::bc7::Bc7Array; MATERIAL_TEX_MAPS];

/// Encode (or load from cache) one variant's 5 BC7 maps. The cache is a single blob
/// per variant (`<slug>/<variant>/pbr.bc7`) holding all 5 maps' mip chains, keyed by
/// a content hash of that variant's decoded RGBA — so editing one texture only
/// re-encodes that one variant. Pure CPU + filesystem: safe to run on a background
/// task pool (no GPU, no ECS access). `slug`/`dir` are owned so a task can capture
/// them.
pub fn encode_variant_bc7(slug: &str, dir: &str) -> VariantBc7 {
    let var_dir = format!("{TEXTURE_ROOT}/{slug}/{dir}");

    // Decode all 5 maps to RGBA8 for this one variant.
    let rgba: [Vec<u8>; MATERIAL_TEX_MAPS] = std::array::from_fn(|map| {
        let mut buf = vec![0u8; (TEXTURE_SIZE * TEXTURE_SIZE * 4) as usize];
        if map == MapArray::Mra as usize {
            write_mra_map(
                &format!("{var_dir}/metallic.png"),
                &format!("{var_dir}/roughness.png"),
                &format!("{var_dir}/ao.png"),
                &mut buf,
            );
        } else {
            let file = match map {
                x if x == MapArray::Diffuse as usize => "diffuse",
                x if x == MapArray::Normal as usize => "normal",
                x if x == MapArray::Height as usize => "height",
                _ => "edge",
            };
            write_rgba_map(&format!("{var_dir}/{file}.png"), &mut buf);
        }
        buf
    });

    // Cache key = hash over all 5 maps' source RGBA, so any map change re-encodes.
    let mut source_key = Vec::with_capacity(rgba.iter().map(|r| r.len()).sum());
    for r in &rgba {
        source_key.extend_from_slice(r);
    }

    super::bc7::Bc7Cache::new(format!("{var_dir}/pbr.bc7")).load_or_encode_multi(
        &source_key,
        || {
            std::array::from_fn(|map| {
                super::bc7::encode_layers_bc7(&rgba[map], TEXTURE_SIZE, 1, false)
            })
        },
    )
}

/// Decode a PNG, resize to `TEXTURE_SIZE²`, and write RGBA8 into `dst`. On failure
/// leaves `dst` as-is (zeroed), logging a warning.
fn write_rgba_map(path: &str, dst: &mut [u8]) {
    match image::open(path) {
        Ok(img) => {
            let rgba = img
                .resize_exact(
                    TEXTURE_SIZE,
                    TEXTURE_SIZE,
                    image::imageops::FilterType::Triangle,
                )
                .to_rgba8();
            dst.copy_from_slice(&rgba);
        }
        Err(e) => warn!("SDF texture: cannot load {path}: {e}"),
    }
}

/// Decode three single-channel maps and pack into RGBA (R=metallic, G=roughness,
/// B=ao, A=255).
fn write_mra_map(metallic: &str, roughness: &str, ao: &str, dst: &mut [u8]) {
    let load_r = |path: &str| -> Option<Vec<u8>> {
        match image::open(path) {
            Ok(img) => Some(
                img.resize_exact(
                    TEXTURE_SIZE,
                    TEXTURE_SIZE,
                    image::imageops::FilterType::Triangle,
                )
                .to_luma8()
                .into_raw(),
            ),
            Err(e) => {
                warn!("SDF texture: cannot load {path}: {e}");
                None
            }
        }
    };
    let m = load_r(metallic);
    let r = load_r(roughness);
    let a = load_r(ao);
    let px = (TEXTURE_SIZE * TEXTURE_SIZE) as usize;
    for i in 0..px {
        dst[i * 4] = m.as_ref().map(|v| v[i]).unwrap_or(0);
        dst[i * 4 + 1] = r.as_ref().map(|v| v[i]).unwrap_or(255); // default rough
        dst[i * 4 + 2] = a.as_ref().map(|v| v[i]).unwrap_or(255); // default unoccluded
        dst[i * 4 + 3] = 255;
    }
}
