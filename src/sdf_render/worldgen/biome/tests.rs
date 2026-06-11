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
            blend: 4.0,
        }],
        biomes: BiomeId::ALL
            .iter()
            .map(|id| BiomeDef {
                name: format!("{id:?}"),
                surface: TerrainMatId(0),
                surface_rules: vec![],
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
            blend: 4.0,
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

// ---------------------------------------------------------------------------------------------
// preview_color SSOT + GPU strata table flatten (preview slice / Stage-3 shader)
// ---------------------------------------------------------------------------------------------

/// `preview_color` is the SSOT for a material's flat colour — for a flat-colour material it equals
/// `base_color` exactly (Stage 5 will redefine it to the texture average without touching callers).
#[test]
fn preview_color_is_base_color_for_flat_materials() {
    let lib = shipped_library();
    for m in &lib.materials {
        assert_eq!(m.preview_color(), m.base_color, "{} preview_color != base_color", m.name);
    }
}

/// The flattened GPU strata table matches `BiomeLibrary` row-for-row: one column per biome in id order,
/// surface/bedrock/layer colours resolved through `preview_color`, and cumulative layer bottoms equal to
/// the running sum of `StrataLayer::thickness` — i.e. a depth probe of the table reproduces
/// An EMPTY/unloaded `BiomeLibrary` (its `Default`, before `biomes.ron` finishes loading) must flatten to
/// a zeroed table of the right length — NOT panic indexing the empty `biomes` Vec. This was a launch crash:
/// `sync_terrain_detail_params` + the preview flatten the library every frame from startup.
#[test]
fn gpu_strata_table_on_empty_library_is_zeroed_not_panic() {
    let table = BiomeLibrary::default().gpu_strata_table();
    assert_eq!(table.len(), BiomeId::ALL.len());
    assert!(table.iter().all(|c| *c == super::GpuStrataColumn::default()));
}

/// `strata_material → preview_color` exactly.
#[test]
fn gpu_strata_table_matches_library() {
    let lib = shipped_library();
    let table = lib.gpu_strata_table();
    assert_eq!(table.len(), BiomeId::ALL.len());

    for (i, &biome) in BiomeId::ALL.iter().enumerate() {
        let def = lib.biome(biome);
        let col = &table[i];
        assert_eq!(col.surface_color, lib.material(def.surface).preview_color());
        assert_eq!(col.bedrock_color, lib.material(def.bedrock).preview_color());
        assert_eq!(col.layer_count as usize, def.strata.len());

        // Per-layer colour + cumulative bottom depth.
        let mut cum = 0.0_f32;
        for (k, layer) in def.strata.iter().enumerate() {
            cum += layer.thickness;
            assert_eq!(col.layer_color[k], lib.material(layer.material).preview_color());
            assert_eq!(col.layer_bottom[k], cum, "biome {biome:?} layer {k} bottom");
        }

        // A depth probe of the GPU column == the CPU strata walk's colour, across the column.
        for depth in [-1.0_f32, 0.0, 0.5, 2.5, 6.0, 500.0, 5000.0] {
            let cpu = lib.material(strata_material(biome, depth as f64, &lib)).preview_color();
            assert_eq!(gpu_probe(col, depth), cpu, "biome {biome:?} depth {depth}");
        }
    }
}

/// `StrataTableStd` (the SHARED std140 flatten consumed by BOTH the editor preview AND the in-world
/// terrain-surface material) must be a VALID std140 uniform. encase's "array stride must be a multiple of
/// 16" assert fires only at encode/assert time — NOT in the layout-mirror tests — which is how a `[u32; 3]`
/// pad (stride 4) passes `--lib` + naga validation yet panics `prepare_erased_assets` at launch. Keep every
/// uniform field a `Vec*`/`UVec*` (never `[scalar; N]`).
#[test]
fn strata_table_std_is_valid_std140_uniform() {
    StrataTableStd::assert_uniform_compat();
}

/// The material palette uniform must be a VALID std140 uniform too (same `[scalar; N]` stride hazard — the
/// `_pad` is a `UVec3`, the arrays are `Vec4`). Fires the encase assert at test time, not launch.
#[test]
fn material_palette_std_is_valid_std140_uniform() {
    MaterialPaletteStd::assert_uniform_compat();
}

/// `MaterialPaletteStd::from_library` flattens colour + roughness by `TerrainMatId`, sets `count`, and never
/// panics on an empty (unloaded) library.
#[test]
fn material_palette_from_library_flattens_and_clamps() {
    let empty = BiomeLibrary::default();
    let p = MaterialPaletteStd::from_library(&empty);
    assert_eq!(p.count, 0, "empty library ⇒ zeroed palette, no panic");

    let lib = BiomeLibrary {
        materials: vec![
            TerrainSurfaceMaterial { name: "a".into(), base_color: [0.1, 0.2, 0.3, 1.0], roughness: 0.7, blend: 4.0 },
            TerrainSurfaceMaterial { name: "b".into(), base_color: [0.4, 0.5, 0.6, 1.0], roughness: 0.2, blend: 8.0 },
        ],
        biomes: vec![],
    };
    let p = MaterialPaletteStd::from_library(&lib);
    assert_eq!(p.count, 2);
    assert_eq!(p.materials[1].color, bevy::math::Vec4::new(0.4, 0.5, 0.6, 1.0));
    assert_eq!(p.materials[1].props.x, 0.2, "props.x = roughness");
}

/// The packed `layer_bottom: [Vec4; 2]` (8 floats) must hold all `GPU_STRATA_MAX_LAYERS` layer bottoms.
#[test]
fn packed_layer_bottom_fits_all_layers() {
    const { assert!(GPU_STRATA_MAX_LAYERS <= 8, "layer_bottom packs into 2 vec4 (8 floats)") };
}

/// `GpuStrataColumnStd` (the std140 flatten) mirrors the CPU `GpuStrataColumn`: a depth probe of the
/// packed/unpacked layout the shaders read reproduces the source column's colours.
#[test]
fn gpu_strata_column_std_mirrors_cpu() {
    let cpu = GpuStrataColumn {
        surface_color: [0.1, 0.2, 0.3, 1.0],
        layer_color: {
            let mut a = [[0.0; 4]; GPU_STRATA_MAX_LAYERS];
            a[0] = [0.4, 0.0, 0.0, 1.0];
            a[1] = [0.0, 0.5, 0.0, 1.0];
            a[2] = [0.0, 0.0, 0.6, 1.0];
            a
        },
        layer_bottom: {
            let mut a = [0.0; GPU_STRATA_MAX_LAYERS];
            a[0] = 1.0;
            a[1] = 5.0;
            a[2] = 1005.0;
            a
        },
        bedrock_color: [0.01, 0.01, 0.02, 1.0],
        layer_count: 3,
        _pad: [0; 3],
    };
    let std = GpuStrataColumnStd::from(&cpu);
    assert_eq!(std.surface_color, Vec4::from_array(cpu.surface_color));
    assert_eq!(std.bedrock_color, Vec4::from_array(cpu.bedrock_color));
    assert_eq!(std.layer_count, cpu.layer_count);
    for i in 0..3 {
        assert_eq!(std.layer_bottom[i / 4][i % 4], cpu.layer_bottom[i], "bottom {i}");
        assert_eq!(std.layer_color[i], Vec4::from_array(cpu.layer_color[i]), "colour {i}");
    }
}

/// `StrataTableStd::from_library` flattens the shipped library row-for-row (one column per biome in id
/// order), so the SHARED GPU table the preview + in-world material upload matches the CPU SSOT.
#[test]
fn strata_table_std_from_library_matches_cpu() {
    let lib = shipped_library();
    let table = StrataTableStd::from_library(&lib);
    let cpu = lib.gpu_strata_table();
    for (i, c) in cpu.iter().enumerate() {
        let std = &table.columns[i];
        assert_eq!(std.surface_color, Vec4::from_array(c.surface_color), "biome {i} surface");
        assert_eq!(std.bedrock_color, Vec4::from_array(c.bedrock_color), "biome {i} bedrock");
        assert_eq!(std.layer_count, c.layer_count, "biome {i} layer_count");
    }
}

/// CPU mirror of the WGSL strata-column walk: find the first layer whose cumulative bottom exceeds
/// `depth`; surface at/above 0, bedrock past the last layer. Used by the table test to assert the GPU
/// layout reproduces `strata_material`.
fn gpu_probe(col: &GpuStrataColumn, depth: f32) -> [f32; 4] {
    if depth <= 0.0 {
        return col.surface_color;
    }
    for k in 0..col.layer_count as usize {
        if depth < col.layer_bottom[k] {
            return col.layer_color[k];
        }
    }
    col.bedrock_color
}

// ---------------------------------------------------------------------------------------------
// Surface material resolver (Stage 2) — the data-driven replacement for the hardcoded shader treatment
// ---------------------------------------------------------------------------------------------

/// A tiny hand-built library for the resolver tests: materials 0=grass 1=snow 2=rock 3=flower; Plains (id 0)
/// carries a snow→rock altitude cap + cliff-rock + a flower patch; Snowy (id 4) surfaces snow; the rest are
/// plain grass. Built directly (not via `compile`) so the rules are explicit + isolated from `biomes.ron`.
fn surf_lib() -> BiomeLibrary {
    let mat = |name: &str, c: [f32; 4]| TerrainSurfaceMaterial {
        name: name.into(),
        base_color: c,
        roughness: 1.0,
        blend: 4.0,
    };
    let materials = vec![
        mat("grass", [0.0, 0.5, 0.0, 1.0]),  // 0
        mat("snow", [1.0, 1.0, 1.0, 1.0]),   // 1
        mat("rock", [0.3, 0.3, 0.3, 1.0]),   // 2
        mat("flower", [0.9, 0.1, 0.9, 1.0]), // 3
    ];
    let base = |n: &str, s: u16| BiomeDef {
        name: n.into(),
        surface: TerrainMatId(s),
        surface_rules: vec![],
        strata: vec![],
        bedrock: TerrainMatId(s),
    };
    let plains = BiomeDef {
        name: "Plains".into(),
        surface: TerrainMatId(0),
        surface_rules: vec![
            SurfaceLayer { material: TerrainMatId(1), when: vec![SurfaceCond::AboveY { start: 100.0, full: 140.0 }] },
            SurfaceLayer { material: TerrainMatId(2), when: vec![SurfaceCond::AboveY { start: 160.0, full: 200.0 }] },
            SurfaceLayer { material: TerrainMatId(2), when: vec![SurfaceCond::Slope { gentle: 0.9, steep: 0.6 }] },
        ],
        strata: vec![],
        bedrock: TerrainMatId(0),
    };
    BiomeLibrary {
        materials,
        biomes: vec![plains, base("Forest", 0), base("Desert", 0), base("Tundra", 0), base("Snowy", 1)],
    }
}

/// Like [`surf_lib`] but Plains carries ONLY a flower [`SurfaceCond::Patch`] (isolated from the cap/cliff
/// rules so the patch test isn't confounded by them, and vice-versa).
fn patch_lib() -> BiomeLibrary {
    let mut lib = surf_lib();
    lib.biomes[0].surface_rules = vec![SurfaceLayer {
        material: TerrainMatId(3),
        when: vec![SurfaceCond::Patch { wavelength: 600.0, threshold: 0.5, softness: 0.05, seed: 1 }],
    }];
    lib
}

fn plains_only(blend: f32) -> BiomeSample {
    BiomeSample { primary: BiomeId::Plains, secondary: BiomeId::Plains, blend }
}

/// A base-only sample (flat, low altitude) resolves to just the biome's surface material (`mat_a == mat_b`,
/// weight 0) — no rule fires.
#[test]
fn resolve_surface_base_only() {
    let lib = surf_lib();
    let s = resolve_surface(0.0, 0.0, 0.0, 1.0, plains_only(0.0), 7, &lib);
    assert_eq!(s.mat_a, 0, "flat low Plains = grass");
    assert_eq!(s.mat_b, 0);
    assert_eq!(s.weight, 0.0);
}

/// An altitude cap: high above the rock band the dominant surface is rock (2); in the snow band snow (1) is
/// present. The hardcoded shader snow/rock treatment is now this data.
#[test]
fn resolve_surface_altitude_cap() {
    let lib = surf_lib();
    // Well above the rock full-altitude (200) → rock dominates.
    let high = resolve_surface(0.0, 0.0, 300.0, 1.0, plains_only(0.0), 7, &lib);
    assert_eq!(high.mat_a, 2, "peak caps to rock");
    // Mid snow band (120, between 100 and 140) → snow is one of the two materials.
    let mid = resolve_surface(0.0, 0.0, 120.0, 1.0, plains_only(0.0), 7, &lib);
    assert!(mid.mat_a == 1 || mid.mat_b == 1, "snow present in the snow band: {mid:?}");
}

/// Cliff rock: a steep surface normal (low `n_y`) at low altitude resolves to rock; a flat one stays grass.
#[test]
fn resolve_surface_slope_cliff() {
    let lib = surf_lib();
    let flat = resolve_surface(5000.0, -3000.0, 0.0, 1.0, plains_only(0.0), 7, &lib);
    assert_eq!(flat.mat_a, 0, "flat = grass");
    let steep = resolve_surface(5000.0, -3000.0, 0.0, 0.4, plains_only(0.0), 7, &lib);
    assert_eq!(steep.mat_a, 2, "steep = cliff rock");
}

/// A biome border (blend 1.0 → 50/50) mixes the two biomes' surface materials: Plains(grass 0) ↔ Snowy(snow
/// 1), weight ≈ 0.5.
#[test]
fn resolve_surface_biome_border() {
    let lib = surf_lib();
    let sample = BiomeSample { primary: BiomeId::Plains, secondary: BiomeId::Snowy, blend: 1.0 };
    let s = resolve_surface(0.0, 0.0, 0.0, 1.0, sample, 7, &lib);
    let pair = [s.mat_a, s.mat_b];
    assert!(pair.contains(&0) && pair.contains(&1), "grass↔snow border: {s:?}");
    assert!((s.weight - 0.5).abs() < 1e-5, "even border ⇒ weight 0.5, got {}", s.weight);
}

/// A noise patch produces its material somewhere over a sampled region (the flower field), and the resolver
/// is deterministic (byte-identical weight on recompute).
#[test]
fn resolve_surface_patch_and_deterministic() {
    let lib = patch_lib();
    let mut saw_flower = false;
    let mut x = -3000.0;
    while x < 3000.0 {
        let s = resolve_surface(x, x * 0.7, 0.0, 1.0, plains_only(0.0), 7, &lib);
        if s.mat_a == 3 || s.mat_b == 3 {
            saw_flower = true;
        }
        // Determinism: recompute is byte-identical.
        let s2 = resolve_surface(x, x * 0.7, 0.0, 1.0, plains_only(0.0), 7, &lib);
        assert_eq!(s.weight.to_bits(), s2.weight.to_bits(), "resolver not deterministic at {x}");
        x += 137.0;
    }
    assert!(saw_flower, "the flower patch never appeared over a 6 km sweep");
}

/// AND-combined conditions: a flower rule requiring BOTH a noise patch AND flat ground appears on flat
/// terrain but NEVER on steep terrain (the mountain-flowers bug) even where the patch noise fires.
#[test]
fn resolve_surface_and_conditions_gate_steep() {
    let mut lib = surf_lib();
    lib.biomes[0].surface_rules = vec![SurfaceLayer {
        material: TerrainMatId(3),
        when: vec![
            SurfaceCond::Patch { wavelength: 600.0, threshold: 0.5, softness: 0.05, seed: 1 },
            SurfaceCond::Slope { gentle: 0.7, steep: 0.95 }, // flat-only: weight → 1 as n_y → 1
        ],
    }];
    let mut saw_flat_flower = false;
    let mut x = -3000.0;
    while x < 3000.0 {
        let flat = resolve_surface(x, x * 0.7, 0.0, 1.0, plains_only(0.0), 7, &lib);
        if flat.mat_a == 3 || flat.mat_b == 3 {
            saw_flat_flower = true;
        }
        let steep = resolve_surface(x, x * 0.7, 0.0, 0.3, plains_only(0.0), 7, &lib);
        assert!(steep.mat_a != 3 && steep.mat_b != 3, "flower must NOT appear on steep ground at {x}");
        x += 137.0;
    }
    assert!(saw_flat_flower, "flower should still appear on flat ground");
}
