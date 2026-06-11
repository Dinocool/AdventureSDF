//! Stage-1 biome/climate/strata tests (headless, CPU-only). See `biome.rs` for the module contract.

use super::*;

/// The shipped biome library — parsed once from the on-disk RON for the asset/strata tests. Panics
/// loudly if `biomes.ron` is malformed (the round-trip guard).
fn shipped_library() -> BiomeLibrary {
    let ron = include_str!("../../../../assets/worldgen/biomes.ron");
    let asset: BiomeLibraryAsset =
        ron::de::from_str(ron).expect("parse assets/worldgen/biomes.ron");
    BiomeLibrary::compile(&asset).expect("compile shipped biome library")
}

// ---------------------------------------------------------------------------------------------
// Climate fields
// ---------------------------------------------------------------------------------------------

/// Temperature + humidity are deterministic (same input → byte-identical output) and stay in `[0,1]`
/// over a wide spread of world coordinates (incl. negatives + large magnitudes).
#[test]
fn climate_deterministic_and_normalized() {
    let seed = 0xA15E_C0DE_2026u64;
    let pts = [
        (0.0, 0.0),
        (1234.5, -6789.0),
        (-50_000.0, 50_000.0),
        (131_072.0, -131_072.0),
        (-1_000_000.25, 2_000_000.5),
    ];
    for &(wx, wz) in &pts {
        let t = temperature(wx, wz, seed);
        let h = humidity(wx, wz, seed);
        assert!((0.0..=1.0).contains(&t), "temperature {t} out of [0,1] at ({wx},{wz})");
        assert!((0.0..=1.0).contains(&h), "humidity {h} out of [0,1] at ({wx},{wz})");
        // Byte-identical on recompute.
        assert_eq!(t.to_bits(), temperature(wx, wz, seed).to_bits(), "temperature not deterministic");
        assert_eq!(h.to_bits(), humidity(wx, wz, seed).to_bits(), "humidity not deterministic");
    }
}

/// Temperature and humidity are INDEPENDENT streams — they don't return the same value everywhere (the
/// salts decorrelate them). Guards a copy-paste / shared-seed bug.
#[test]
fn climate_temperature_and_humidity_decorrelated() {
    let seed = 7;
    let mut differ = 0;
    let mut x = -20_000.0;
    while x < 20_000.0 {
        let t = temperature(x, x * 0.5, seed);
        let h = humidity(x, x * 0.5, seed);
        if (t - h).abs() > 1e-6 {
            differ += 1;
        }
        x += 1731.0;
    }
    assert!(differ > 10, "temperature and humidity look correlated ({differ} differing samples)");
}

/// Climate is low-frequency: two points 100 m apart have nearly-equal climate (biomes are large). Guards
/// against an accidental high-frequency field (which would shatter biomes into noise).
#[test]
fn climate_is_low_frequency() {
    let seed = 42;
    for &(wx, wz) in &[(0.0, 0.0), (5000.0, -3000.0), (-12_345.0, 6789.0)] {
        let t0 = temperature(wx, wz, seed);
        let t1 = temperature(wx + 100.0, wz, seed);
        assert!((t0 - t1).abs() < 0.05, "temperature varies too fast: {t0} vs {t1}");
    }
}

// ---------------------------------------------------------------------------------------------
// Classifier
// ---------------------------------------------------------------------------------------------

/// `classify` is total: it returns a biome for every `(t,h)` over a dense scan of the unit square (and
/// for out-of-range inputs, which it clamps).
#[test]
fn classify_is_total() {
    let mut t = -0.2;
    while t <= 1.2 {
        let mut h = -0.2;
        while h <= 1.2 {
            let s = classify(t, h);
            assert!(BiomeId::ALL.contains(&s.primary), "primary not a known biome at ({t},{h})");
            assert!(BiomeId::ALL.contains(&s.secondary), "secondary not a known biome at ({t},{h})");
            assert!((0.0..=1.0).contains(&s.blend), "blend {} out of [0,1] at ({t},{h})", s.blend);
            h += 0.03;
        }
        t += 0.03;
    }
}

/// Each demo biome is REACHABLE: a representative `(t,h)` per the plan's table maps to it.
#[test]
fn classify_reaches_every_demo_biome() {
    // T mid / H mid → Plains.
    assert_eq!(classify(0.5, 0.3).primary, BiomeId::Plains);
    // T mid / H high → Forest.
    assert_eq!(classify(0.5, 0.8).primary, BiomeId::Forest);
    // T high / H low → Desert.
    assert_eq!(classify(0.9, 0.2).primary, BiomeId::Desert);
    // T high / H high → still Desert (hot tier ignores humidity, per the table).
    assert_eq!(classify(0.9, 0.9).primary, BiomeId::Desert);
    // T low / H low → Tundra.
    assert_eq!(classify(0.1, 0.2).primary, BiomeId::Tundra);
    // T low / H high → Snowy.
    assert_eq!(classify(0.1, 0.9).primary, BiomeId::Snowy);
}

/// At a cell CENTRE (deep interior, far from every border) blend → 0 and secondary == primary.
#[test]
fn classify_blend_zero_at_cell_center() {
    // Centre of the Plains cell: T in (T_COLD, T_WARM), H well below H_MID_WET, both far from borders.
    let s = classify(0.5, 0.2);
    assert_eq!(s.primary, BiomeId::Plains);
    assert_eq!(s.secondary, BiomeId::Plains, "deep interior should have no secondary");
    assert!(s.blend.abs() < 1e-6, "blend {} should be ~0 deep in a cell", s.blend);
}

/// Right at a partition border, blend → 1 and secondary is the biome across it. At the T_WARM border
/// between mid (Plains, low H) and hot (Desert).
#[test]
fn classify_blend_high_at_border() {
    let s = classify(T_WARM - 1e-4, 0.2);
    assert_eq!(s.primary, BiomeId::Plains);
    assert_eq!(s.secondary, BiomeId::Desert, "across T_WARM (low H) should be Desert");
    assert!(s.blend > 0.99, "blend {} should be ~1 at the border", s.blend);
}

/// Blend ramps monotonically toward a border (sanity on the distance→blend map).
#[test]
fn classify_blend_increases_toward_border() {
    // Approach the T_WARM border from the Plains side at fixed low H.
    let far = classify(T_WARM - BLEND_BAND - 0.01, 0.2).blend;
    let mid = classify(T_WARM - BLEND_BAND * 0.5, 0.2).blend;
    let near = classify(T_WARM - 0.005, 0.2).blend;
    assert!(far <= mid && mid <= near, "blend not monotone toward border: {far} {mid} {near}");
}

// ---------------------------------------------------------------------------------------------
// Data model + RON
// ---------------------------------------------------------------------------------------------

/// `BiomeLibraryAsset` round-trips through RON byte-for-byte (structurally).
#[test]
fn biome_asset_ron_round_trips() {
    let ron = include_str!("../../../../assets/worldgen/biomes.ron");
    let asset: BiomeLibraryAsset = ron::de::from_str(ron).expect("parse biomes.ron");
    let s = ron::ser::to_string(&asset).expect("serialize");
    let back: BiomeLibraryAsset = ron::de::from_str(&s).expect("re-parse");
    assert_eq!(asset, back, "biome library asset did not round-trip");
}

/// The shipped `biomes.ron` compiles, defines exactly the 5 demo biomes in id order, and every
/// `TerrainMatId` it references exists in the palette (compile validates this; we also assert names).
#[test]
fn shipped_biomes_ron_compiles_and_is_consistent() {
    let lib = shipped_library();
    assert_eq!(lib.biomes.len(), BiomeId::ALL.len());
    // Biome `i` is the def for `BiomeId::ALL[i]`; check the names match the variant intent.
    let expected = ["Plains", "Forest", "Desert", "Tundra", "Snowy"];
    for (i, id) in BiomeId::ALL.iter().enumerate() {
        assert_eq!(*id as usize, i, "BiomeId ordering must match array index");
        assert_eq!(lib.biome(*id).name, expected[i], "biome {i} name");
    }
    // Every referenced material id is in range (compile would have errored otherwise; double-check).
    for b in &lib.biomes {
        for id in std::iter::once(b.surface)
            .chain(b.strata.iter().map(|s| s.material))
            .chain(std::iter::once(b.bedrock))
        {
            assert!((id.0 as usize) < lib.materials.len(), "material {id:?} out of range");
        }
    }
}

/// `compile` rejects a library that references a missing material.
#[test]
fn compile_rejects_missing_material() {
    let asset = BiomeLibraryAsset {
        materials: vec![TerrainSurfaceMaterial {
            name: "only".into(),
            base_color: [0.5, 0.5, 0.5, 1.0],
            roughness: 1.0,
        }],
        biomes: BiomeId::ALL
            .iter()
            .map(|id| BiomeDef {
                name: format!("{id:?}"),
                surface: TerrainMatId(0),
                strata: vec![StrataLayer { material: TerrainMatId(99), thickness: 1.0 }],
                bedrock: TerrainMatId(0),
            })
            .collect(),
    };
    assert!(matches!(
        BiomeLibrary::compile(&asset),
        Err(BiomeCompileError::MissingMaterial { .. })
    ));
}

/// `compile` rejects the wrong biome count.
#[test]
fn compile_rejects_biome_count_mismatch() {
    let asset = BiomeLibraryAsset {
        materials: vec![TerrainSurfaceMaterial {
            name: "m".into(),
            base_color: [0.0; 4],
            roughness: 1.0,
        }],
        biomes: vec![],
    };
    assert!(matches!(
        BiomeLibrary::compile(&asset),
        Err(BiomeCompileError::BiomeCountMismatch { .. })
    ));
}

// ---------------------------------------------------------------------------------------------
// Strata lookup + color compose
// ---------------------------------------------------------------------------------------------

/// `strata_material`: depth 0 = surface; mid-depths walk the layers in order; very deep = bedrock.
#[test]
fn strata_walks_column_then_bedrock() {
    let lib = shipped_library();
    let biome = BiomeId::Plains;
    let def = lib.biome(biome).clone();

    // Depth 0 → surface.
    assert_eq!(strata_material(biome, 0.0, &lib), def.surface);
    // Negative depth (above surface) also → surface (total).
    assert_eq!(strata_material(biome, -5.0, &lib), def.surface);

    // Walk each stratum: a depth just inside band `k` returns `strata[k].material`.
    let mut top = 0.0_f64;
    for layer in &def.strata {
        let mid = top + layer.thickness as f64 * 0.5;
        assert_eq!(
            strata_material(biome, mid, &lib),
            layer.material,
            "depth {mid} should be in stratum {:?}",
            layer.material
        );
        top += layer.thickness as f64;
    }

    // Below the last stratum → bedrock.
    assert_eq!(strata_material(biome, top + 1000.0, &lib), def.bedrock);
}

/// `strata_material` is total across a depth sweep for every biome (never panics, always a palette id).
#[test]
fn strata_is_total_for_all_biomes() {
    let lib = shipped_library();
    for &biome in &BiomeId::ALL {
        let mut depth = -2.0;
        while depth < 5000.0 {
            let id = strata_material(biome, depth, &lib);
            assert!((id.0 as usize) < lib.materials.len(), "bad material id at depth {depth}");
            depth += 0.5;
        }
    }
}

/// `terrain_color` at depth 0 returns the surface biome's SURFACE material colour.
#[test]
fn terrain_color_surface_matches_biome_surface() {
    let lib = shipped_library();
    let seed = 0xA15E_C0DE_2026u64;
    for &(wx, wz) in &[(0.0, 0.0), (12_000.0, -8_000.0), (-40_000.0, 40_000.0)] {
        let sample = surface_biome(wx, wz, seed);
        let surf_mat = lib.biome(sample.primary).surface;
        let expected = lib.material(surf_mat).base_color;
        assert_eq!(terrain_color(wx, wz, 0.0, seed, &lib), expected, "surface color at ({wx},{wz})");
    }
}

/// `surface_biome` is deterministic (same world point + seed → same sample).
#[test]
fn surface_biome_deterministic() {
    let seed = 123;
    for &(wx, wz) in &[(0.0, 0.0), (9999.0, -1234.0), (-55_555.0, 66_666.0)] {
        assert_eq!(surface_biome(wx, wz, seed), surface_biome(wx, wz, seed));
    }
}
