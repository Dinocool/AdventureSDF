//! Round-trip + resolution tests for the asset framework.

use std::path::PathBuf;

use super::material::MaterialAsset;
use super::texture_lib::{MAX_TEXTURE_LAYERS, MaterialTextureLibrary};
use super::{Asset, TexRef};

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
    let mut asset = MaterialAsset {
        base_color: [0.2, 0.4, 0.6, 1.0],
        blend_softness: 0.3,
        maps: std::array::from_fn(|_| None),
    };
    asset.maps[0] = Some(TexRef {
        slug: "cobble_stone".into(),
        dir: "1".into(),
    });

    let path = temp_path("roundtrip");
    asset.save(&path).expect("save");

    let text = std::fs::read_to_string(&path).unwrap();
    let loaded: MaterialAsset = ron::de::from_str(&text).expect("load");

    assert_eq!(loaded.base_color, asset.base_color);
    assert_eq!(loaded.blend_softness, asset.blend_softness);
    assert_eq!(loaded.maps[0], asset.maps[0]);

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
fn texref_resolves_to_stable_layer() {
    let mut lib = MaterialTextureLibrary::default();
    let a = lib.resolve_layer("cobble_stone", "1");
    let b = lib.resolve_layer("sand", "2");
    let a_again = lib.resolve_layer("cobble_stone", "1");

    assert_eq!(a, 0, "first variant gets layer 0");
    assert_eq!(b, 1, "second distinct variant gets layer 1");
    assert_eq!(a_again, a, "same (slug,dir) resolves to the same layer");
    assert_eq!(lib.variants.len(), 2);
    assert!(lib.dirty);
}

#[test]
fn texture_library_respects_cap() {
    let mut lib = MaterialTextureLibrary::default();
    for i in 0..MAX_TEXTURE_LAYERS {
        let layer = lib.resolve_layer("slug", &i.to_string());
        assert_eq!(layer, i);
    }
    // One past the cap falls back rather than panicking.
    let over = lib.resolve_layer("slug", "overflow");
    assert_eq!(over, u32::MAX);
}
