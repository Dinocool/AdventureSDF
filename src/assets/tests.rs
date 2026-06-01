//! Round-trip + resolution tests for the asset framework.

use std::path::PathBuf;

use super::material::MaterialAsset;
use super::pbr_texture::PbrTextureAsset;
use super::texture_lib::{MAX_TEXTURE_LAYERS, MaterialTextureLibrary};
use super::Asset;

fn temp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "soul_asset_test_{}_{}.material.ron",
        name,
        std::process::id()
    ));
    p
}

#[test]
fn material_asset_round_trip_is_stable() {
    let asset = MaterialAsset {
        base_color: [0.2, 0.4, 0.6, 1.0],
        blend_softness: 0.3,
        metallic: 0.8,
        roughness: 0.25,
        parallax_scale: 0.04,
        emissive_color: [0.0, 0.0, 0.0],
        emissive_intensity: 0.0,
        texture: Some("pbrtextures/cobble_stone_1.pbrtex.ron".into()),
        overrides: PbrTextureAsset {
            height: Some("textures/cobble_stone/2/height.png".into()),
            ..Default::default()
        },
    };

    let path = temp_path("roundtrip");
    asset.save(&path).expect("save");

    let text = std::fs::read_to_string(&path).unwrap();
    let loaded: MaterialAsset = ron::de::from_str(&text).expect("load");

    assert_eq!(loaded.base_color, asset.base_color);
    assert_eq!(loaded.texture, asset.texture);
    assert_eq!(loaded.overrides, asset.overrides);

    // Re-save must be byte-identical.
    let path2 = temp_path("roundtrip2");
    loaded.save(&path2).expect("re-save");
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        std::fs::read_to_string(&path2).unwrap(),
        "save->load->save must be byte-stable"
    );

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(&path2).ok();
}

#[test]
fn material_missing_texture_fields_default() {
    // Older RON without `texture`/`overrides` must still parse (back-compat).
    let ron = r#"MaterialAsset(base_color:(1.0,1.0,1.0,1.0),blend_softness:0.0)"#;
    let m: MaterialAsset = ron::de::from_str(ron).expect("load legacy material");
    assert_eq!(m.texture, None);
    assert!(m.overrides.is_empty());
}

#[test]
fn map_set_resolves_to_stable_layer() {
    let mut lib = MaterialTextureLibrary::default();
    let a = PbrTextureAsset {
        diffuse: Some("textures/cobble_stone/1/diffuse.png".into()),
        ..Default::default()
    }
    .to_map_set();
    let b = PbrTextureAsset {
        diffuse: Some("textures/sand/2/diffuse.png".into()),
        ..Default::default()
    }
    .to_map_set();

    let la = lib.resolve_layer(&a);
    let lb = lib.resolve_layer(&b);
    let la_again = lib.resolve_layer(&a);

    assert_eq!(la, 0, "first map-set gets layer 0");
    assert_eq!(lb, 1, "second distinct map-set gets layer 1");
    assert_eq!(la_again, la, "same map-set resolves to the same layer");
    assert_eq!(lib.variants.len(), 2);
    assert!(lib.dirty);
}

#[test]
fn empty_map_set_is_fallback_layer() {
    let mut lib = MaterialTextureLibrary::default();
    assert_eq!(lib.resolve_layer(&super::MapSet::default()), u32::MAX);
    assert_eq!(lib.variants.len(), 0, "empty set consumes no layer");
}

#[test]
fn texture_library_respects_cap() {
    let mut lib = MaterialTextureLibrary::default();
    for i in 0..MAX_TEXTURE_LAYERS {
        let set = PbrTextureAsset {
            diffuse: Some(format!("t/{i}/diffuse.png").into()),
            ..Default::default()
        }
        .to_map_set();
        assert_eq!(lib.resolve_layer(&set), i);
    }
    // One past the cap falls back rather than panicking.
    let over = PbrTextureAsset {
        diffuse: Some("t/overflow/diffuse.png".into()),
        ..Default::default()
    }
    .to_map_set();
    assert_eq!(lib.resolve_layer(&over), u32::MAX);
}

// === One-off migration generators (run explicitly with `-- --ignored`) ==============

/// Generate one `.pbrtex.ron` per existing texture variant directory, pointing at that
/// dir's role PNGs. Run with:
/// `cargo test --features editor export_variant_pbrtextures -- --ignored --nocapture`.
#[test]
#[ignore]
fn export_variant_pbrtextures() {
    use crate::sdf_render::textures::{TEXTURE_ROOT, read_manifest};

    let role = |slug: &str, dir: &str, file: &str| -> Option<PathBuf> {
        let rel = PathBuf::from(format!("textures/{slug}/{dir}/{file}.png"));
        if std::path::Path::new("assets").join(&rel).is_file() {
            Some(rel)
        } else {
            None
        }
    };

    let Ok(entries) = std::fs::read_dir(TEXTURE_ROOT) else {
        panic!("no {TEXTURE_ROOT} dir");
    };
    let slugs: Vec<String> = entries
        .flatten()
        .filter(|e| e.path().join("material.ron").is_file())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();

    for slug in slugs {
        for v in read_manifest(&slug) {
            let tex = PbrTextureAsset {
                diffuse: role(&v.slug, &v.dir, "diffuse"),
                normal: role(&v.slug, &v.dir, "normal"),
                metallic: role(&v.slug, &v.dir, "metallic"),
                roughness: role(&v.slug, &v.dir, "roughness"),
                ao: role(&v.slug, &v.dir, "ao"),
                height: role(&v.slug, &v.dir, "height"),
                edge: role(&v.slug, &v.dir, "edge"),
            };
            let path = PathBuf::from(format!(
                "assets/pbrtextures/{}_{}.pbrtex.ron",
                v.slug, v.dir
            ));
            tex.save(&path).expect("save pbrtex");
            println!("wrote {}", path.display());
        }
    }
}

/// Regenerate the demo `.material.ron` resources to reference the migrated bundles.
/// Run after `export_variant_pbrtextures`. Run with:
/// `cargo test --features editor export_demo_materials -- --ignored --nocapture`.
#[test]
#[ignore]
fn export_demo_materials() {
    let textured = |slug: &str| MaterialAsset {
        base_color: [1.0, 1.0, 1.0, 1.0],
        blend_softness: 0.0,
        metallic: 0.0,
        roughness: 1.0,
        parallax_scale: 0.15,
        emissive_color: [0.0, 0.0, 0.0],
        emissive_intensity: 0.0,
        texture: Some(PathBuf::from(format!("pbrtextures/{slug}_1.pbrtex.ron"))),
        overrides: PbrTextureAsset::default(),
    };
    let exemplar = |color: [f32; 4], metallic: f32, roughness: f32| MaterialAsset {
        base_color: color,
        blend_softness: 0.0,
        metallic,
        roughness,
        parallax_scale: 0.06,
        emissive_color: [0.0, 0.0, 0.0],
        emissive_intensity: 0.0,
        texture: None,
        overrides: PbrTextureAsset::default(),
    };

    let items: [(&str, MaterialAsset); 6] = [
        ("sand", textured("sand")),
        ("cobble", textured("cobble_stone")),
        ("ground", textured("ground")),
        ("red_metal", exemplar([0.55, 0.04, 0.03, 1.0], 1.0, 0.18)),
        ("gold_rough", exemplar([0.83, 0.62, 0.18, 1.0], 1.0, 0.45)),
        ("white_gloss", exemplar([0.9, 0.9, 0.92, 1.0], 0.0, 0.08)),
    ];
    for (name, asset) in &items {
        let path = PathBuf::from(format!("assets/materials/{name}.material.ron"));
        asset.save(&path).expect("save demo material");
        println!("wrote {}", path.display());
    }
}
