//! Climate + biome classification + volumetric strata — **Stage 1** of the terrain-materials feature
//! (see `docs/TERRAIN_MATERIALS_PLAN.md`). CPU/data only: no shader or per-chunk bake wiring yet (that
//! is Stage 2/3). This module supplies the SSOT the later stages call:
//!
//! 1. **Climate fields** — [`temperature`] / [`humidity`]: low-frequency, bit-portable value-noise
//!    fields over world XZ, each normalized to `[0,1]`, on two distinct seed streams. Climate is
//!    INDEPENDENT of terrain height (altitude-snow is a later surface treatment, not baked here), so it
//!    does **not** touch `sample_world` / the height graph ⇒ `worldgen_parity` is unaffected (NO
//!    `HEIGHT_GEN_VERSION` bump). It still reuses [`super::noise`]'s integer-hash + IEEE-basic-op value
//!    noise so biome choice agrees bit-for-bit across machines (shared-seed multiplayer).
//!
//! 2. **Classifier** — [`classify`]: a Whittaker-style partition of the unit `T×H` climate square into
//!    [`BiomeId`]s, returning a [`BiomeSample`] (`primary`, `secondary`, `blend`) for smooth
//!    boundaries. Total (every `(t,h)` maps somewhere) and deterministic.
//!
//! 3. **Data model + RON** — [`BiomeDef`] / [`StrataLayer`] / [`TerrainSurfaceMaterial`] compiled into a
//!    [`BiomeLibrary`] resource, authored in `assets/worldgen/biomes.ron` ([`BiomeLibraryAsset`]).
//!
//! 4. **Query API** — [`surface_biome`] (climate → biome), [`strata_material`] (depth → material id),
//!    and [`terrain_color`] (the full climate→biome→strata→color compose, for tests/CPU preview).
//!
//! Stage 2/3 will consume [`BiomeLibrary`] to build a GPU strata table and bake the surface-height +
//! biome channels per chunk; this module is the authoritative CPU reference for that.

use bevy::asset::{AssetLoader, LoadContext, io::Reader};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use super::noise::{FbmParams, fbm_height};

// ============================================================================================
// Climate fields
// ============================================================================================

/// Base wavelength of the climate fields, in world metres. Multiple km so biomes are *large* regions
/// (the demo world is ±131 km — see `WORLDGEN_TERRAIN_HALF_XZ`). Low frequency ⇒ smooth, slowly-varying
/// climate so a biome spans many chunks. `base_freq = 1 / wavelength`.
pub const CLIMATE_WAVELENGTH_M: f64 = 8192.0;

/// Octave count for the climate fBm — a few octaves so the field has some structure (coast/interior
/// variation) without becoming high-frequency noise.
pub const CLIMATE_OCTAVES: u32 = 4;

/// Per-field salts folded into the world seed so temperature and humidity are INDEPENDENT noise streams
/// (decorrelated — otherwise hot would always imply dry). Distinct large odd constants.
const TEMPERATURE_SALT: u64 = 0x5417_1AB7_C0FF_EE01;
const HUMIDITY_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// fBm params for a climate field at `(base_freq, seed)`. Gain 0.5 / lacunarity 2.0 (the standard fBm
/// rolloff); amplitude 1.0 since we normalize the raw `[-1,1]`-ish sum to `[0,1]` afterwards.
fn climate_fbm(seed: u32) -> FbmParams {
    FbmParams {
        octaves: CLIMATE_OCTAVES,
        base_freq: 1.0 / CLIMATE_WAVELENGTH_M,
        lacunarity: 2.0,
        gain: 0.5,
        amplitude: 1.0,
        seed,
    }
}

/// Map a raw fBm sum (≈`[-amp_sum, amp_sum]`) into `[0,1]`. The geometric octave-amplitude sum bounds
/// `|fbm|`; we divide by it (so the field reaches the rails only at extreme noise) and affine-map
/// `[-1,1] → [0,1]`, then clamp for total safety. Pure basic ops ⇒ bit-portable.
fn normalize_climate(raw: f64) -> f64 {
    // Geometric sum of octave amplitudes for amplitude 1.0, gain 0.5: Σ 0.5^k, k=0..octaves.
    let mut bound = 0.0;
    let mut amp = 1.0;
    for _ in 0..CLIMATE_OCTAVES {
        bound += amp;
        amp *= 0.5;
    }
    let unit = raw / bound; // ≈ [-1, 1]
    let mapped = unit * 0.5 + 0.5; // → [0, 1]
    mapped.clamp(0.0, 1.0)
}

/// Fold a 64-bit per-field salt into the world `seed` and narrow to the `u32` the noise basis hashes
/// with. XOR the high/low halves of `seed ^ salt` so both 32-bit halves of the world seed matter (the
/// noise basis is `u32`-keyed). Pure integer ops ⇒ deterministic + portable.
fn climate_seed(seed: u64, salt: u64) -> u32 {
    let mixed = seed ^ salt;
    ((mixed >> 32) as u32) ^ (mixed as u32)
}

/// Temperature climate field at world `(wx, wz)` for `seed`, normalized to `[0,1]` (0 = coldest,
/// 1 = hottest). Low-frequency, deterministic, bit-portable (value noise + basic ops). Independent of
/// height — altitude/latitude treatments are applied later as surface overrides, not here.
#[inline]
pub fn temperature(wx: f64, wz: f64, seed: u64) -> f64 {
    let raw = fbm_height(wx, wz, &climate_fbm(climate_seed(seed, TEMPERATURE_SALT)));
    normalize_climate(raw)
}

/// Humidity climate field at world `(wx, wz)` for `seed`, normalized to `[0,1]` (0 = driest, 1 =
/// wettest). Independent seed stream from [`temperature`]; same portability guarantees.
#[inline]
pub fn humidity(wx: f64, wz: f64, seed: u64) -> f64 {
    let raw = fbm_height(wx, wz, &climate_fbm(climate_seed(seed, HUMIDITY_SALT)));
    normalize_climate(raw)
}

// ============================================================================================
// Biome classification (Whittaker-style T×H partition)
// ============================================================================================

/// The demo biomes. A Whittaker-style partition of the unit temperature×humidity square (see
/// [`classify`]). `u8`-tagged so it round-trips cleanly through RON and can later index a GPU table.
#[derive(
    Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord,
)]
#[repr(u8)]
pub enum BiomeId {
    /// T mid, H mid — grass over dirt over stone.
    Plains = 0,
    /// T mid, H high — grass (darker) over dirt over stone.
    Forest = 1,
    /// T high, H low — sand over sandstone over stone.
    Desert = 2,
    /// T low, H low–mid — tundra over permafrost over stone.
    Tundra = 3,
    /// T low, H high (cold + wet) — snow over rock over stone.
    Snowy = 4,
}

impl BiomeId {
    /// All demo biomes, in id order. Used to size/validate the library and by tests.
    pub const ALL: [BiomeId; 5] = [
        BiomeId::Plains,
        BiomeId::Forest,
        BiomeId::Desert,
        BiomeId::Tundra,
        BiomeId::Snowy,
    ];
}

/// Result of climate → biome classification. `primary` is the dominant biome at the sample; `secondary`
/// is the nearest neighbouring biome across the closest climate-cell border; `blend` ∈ `[0,1]` is how
/// far the sample sits into the transition toward `secondary` (0 at a cell centre / deep interior,
/// approaching 1 right at the border). Later stages cross-fade strata/colours by `blend` for seamless
/// biome boundaries.
#[derive(Reflect, Clone, Copy, Debug, PartialEq)]
pub struct BiomeSample {
    pub primary: BiomeId,
    pub secondary: BiomeId,
    pub blend: f32,
}

/// Temperature partition thresholds (`[0,1]`): `< COLD` = cold tier (Tundra/Snowy), `< WARM` = mid tier
/// (Plains/Forest), `≥ WARM` = hot tier (Desert). Used by [`classify`] and the blend distances.
pub const T_COLD: f64 = 0.33;
pub const T_WARM: f64 = 0.66;

/// Humidity split inside the mid temperature tier: `< MID_WET` = Plains, `≥` = Forest.
pub const H_MID_WET: f64 = 0.55;

/// Humidity split inside the cold temperature tier: `< COLD_WET` = Tundra (dry-ish), `≥` = Snowy
/// (cold + wet). Matches the plan: Tundra at low–mid H, Snowy at high H / very cold.
pub const H_COLD_WET: f64 = 0.5;

/// Classify a climate sample `(t, h)` (both expected in `[0,1]`, clamped for safety) into a
/// [`BiomeSample`]. A Whittaker-style partition matching the plan's "Demo biomes" table:
///
/// | T tier | H | biome |
/// |---|---|---|
/// | hot (≥ [`T_WARM`]) | any | Desert |
/// | mid ([`T_COLD`]‥[`T_WARM`]) | < [`H_MID_WET`] | Plains |
/// | mid | ≥ [`H_MID_WET`] | Forest |
/// | cold (< [`T_COLD`]) | < [`H_COLD_WET`] | Tundra |
/// | cold | ≥ [`H_COLD_WET`] | Snowy |
///
/// Total (every `(t,h)` maps somewhere) and deterministic. `secondary`/`blend` come from the distance to
/// the nearest partition border crossed (in climate space): we measure the signed distance to each
/// boundary adjacent to the primary cell, take the closest, and map "near a border" → `blend → 1` via a
/// transition band [`BLEND_BAND`] wide. At a cell interior `blend → 0` and `secondary == primary`.
pub fn classify(t: f64, h: f64) -> BiomeSample {
    let t = t.clamp(0.0, 1.0);
    let h = h.clamp(0.0, 1.0);

    let primary = primary_biome(t, h);

    // Find the nearest cell border adjacent to `primary` and the biome on its far side. Each candidate
    // is `(distance_to_border, neighbour_biome)`; we keep the minimum distance.
    let mut best_dist = f64::INFINITY;
    let mut secondary = primary;
    for (dist, neigh) in neighbour_borders(t, h, primary) {
        if dist < best_dist {
            best_dist = dist;
            secondary = neigh;
        }
    }

    // Map border distance → blend: 1 at the border (dist 0), 0 once `dist ≥ BLEND_BAND`. Linear ramp.
    let blend = if best_dist.is_finite() {
        (1.0 - best_dist / BLEND_BAND).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Outside the transition band (blend == 0) the sample is unambiguously `primary` — collapse
    // `secondary` to `primary` so a deep-interior sample has no spurious neighbour (the blend would
    // weight it 0 anyway, but this keeps the sample self-consistent for consumers that read secondary
    // directly).
    if blend <= 0.0 {
        secondary = primary;
    }

    BiomeSample {
        primary,
        secondary,
        blend: blend as f32,
    }
}

/// Width (in climate units) of the transition band over which [`classify`]'s `blend` ramps from 0
/// (interior) to 1 (border). Small relative to the cell sizes so blending is confined near boundaries.
pub const BLEND_BAND: f64 = 0.08;

/// The total partition function: `(t,h) → BiomeId`. Factored out so [`classify`] and
/// [`neighbour_borders`] share one SSOT for the cell assignment (no divergence between the primary pick
/// and the border math).
fn primary_biome(t: f64, h: f64) -> BiomeId {
    if t >= T_WARM {
        BiomeId::Desert
    } else if t >= T_COLD {
        // mid temperature
        if h >= H_MID_WET {
            BiomeId::Forest
        } else {
            BiomeId::Plains
        }
    } else {
        // cold
        if h >= H_COLD_WET {
            BiomeId::Snowy
        } else {
            BiomeId::Tundra
        }
    }
}

/// For the cell `primary` at `(t,h)`, the borders to its neighbouring cells, each as
/// `(distance_to_border, neighbour_biome)`. Only borders that actually separate `primary` from a
/// DIFFERENT biome are returned (so `secondary != primary` whenever a finite distance exists). Pure
/// basic ops.
fn neighbour_borders(t: f64, h: f64, primary: BiomeId) -> Vec<(f64, BiomeId)> {
    let mut out = Vec::new();
    // Helper: the biome just across a temperature/humidity boundary, evaluated by nudging the sample
    // across it and re-partitioning (keeps the neighbour identity consistent with `primary_biome`).
    let mut push = |dist: f64, neigh: BiomeId| {
        if neigh != primary {
            out.push((dist.abs(), neigh));
        }
    };

    match primary {
        BiomeId::Desert => {
            // Border to the mid tier at T_WARM; neighbour is Plains or Forest by current humidity.
            let neigh = if h >= H_MID_WET { BiomeId::Forest } else { BiomeId::Plains };
            push(t - T_WARM, neigh);
        }
        BiomeId::Forest => {
            // Hot border (→ Desert) and the humidity border (→ Plains).
            push(t - T_WARM, BiomeId::Desert);
            push(h - H_MID_WET, BiomeId::Plains);
            // Cold border (→ Snowy, since H is high here).
            push(t - T_COLD, BiomeId::Snowy);
        }
        BiomeId::Plains => {
            push(t - T_WARM, BiomeId::Desert);
            push(h - H_MID_WET, BiomeId::Forest);
            // Cold border (→ Tundra, since H is low here).
            push(t - T_COLD, BiomeId::Tundra);
        }
        BiomeId::Tundra => {
            // Warm border (→ Plains, low H) and the humidity border (→ Snowy).
            push(t - T_COLD, BiomeId::Plains);
            push(h - H_COLD_WET, BiomeId::Snowy);
        }
        BiomeId::Snowy => {
            // Warm border (→ Forest, high H) and the humidity border (→ Tundra).
            push(t - T_COLD, BiomeId::Forest);
            push(h - H_COLD_WET, BiomeId::Tundra);
        }
    }
    out
}

// ============================================================================================
// Data model: materials + strata + biome columns (RON)
// ============================================================================================

/// Index of a terrain surface material in the [`BiomeLibrary`]'s palette. A small newtype so the RON is
/// readable (`TerrainMatId(2)`) and later stages can index a GPU material table directly. Authored RON
/// references materials by id into the parallel `materials` array.
#[derive(
    Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord,
)]
pub struct TerrainMatId(pub u16);

/// A flat-colour terrain material (no textures this stage — Stage 5 adds PBR maps). `base_color` is
/// **linear** RGBA so the RON is engine-version-stable and trivially serde-able (mirrors
/// `MaterialAsset`).
#[derive(Asset, Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TerrainSurfaceMaterial {
    /// Human-readable name (grass, dirt, stone, …) — for the editor / debugging.
    pub name: String,
    /// Linear RGBA flat colour.
    pub base_color: [f32; 4],
    /// PBR roughness fallback (0 = mirror, 1 = fully diffuse).
    pub roughness: f32,
}

impl TerrainSurfaceMaterial {
    /// The **single source of truth** for "what flat colour represents this material" — used by every
    /// preview (biome map, strata slice) AND any flat-colour fallback so they always agree. For a
    /// flat-colour material this IS its `base_color` (linear RGBA). When Stage 5 adds PBR textures this
    /// becomes the average colour of the diffuse texture (computed once on load) WITHOUT changing any
    /// caller — they all read `preview_color()`, never `base_color` directly for preview purposes.
    #[inline]
    pub fn preview_color(&self) -> [f32; 4] {
        self.base_color
    }
}

/// One stratum in a biome's vertical column: a material occupying `thickness` metres BELOW the layer
/// above it (so the column is read top-down by accumulating thicknesses). The surface material sits
/// above the first `StrataLayer`; bedrock fills everything below the last.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct StrataLayer {
    pub material: TerrainMatId,
    /// Thickness in metres of this layer (the band of `depth` it covers).
    pub thickness: f32,
}

/// A biome's full definition: its surface material (top, undug), the ordered sub-surface strata, and the
/// bedrock material that fills everything below the strata (down to the world bedrock floor). Referenced
/// by [`BiomeId`] in the library.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct BiomeDef {
    /// Human-readable biome name.
    pub name: String,
    /// The top (depth 0) material — the undug surface (before surface treatment, a later stage).
    pub surface: TerrainMatId,
    /// Sub-surface strata, top→down (each `thickness` metres below the previous). The `surface`
    /// material's band is the first entry's `thickness` (i.e. `strata[0]` IS the surface band); see
    /// [`strata_material`] for the exact depth walk.
    pub strata: Vec<StrataLayer>,
    /// Fills all depth below the last stratum (down to the bedrock floor).
    pub bedrock: TerrainMatId,
}

// ============================================================================================
// Asset (RON) + compiled library resource
// ============================================================================================

/// The on-disk biome/strata/material set — `assets/worldgen/biomes.ron`. Authored RON loaded via
/// [`BiomeLibraryAssetLoader`], then compiled into the [`BiomeLibrary`] resource. Mirrors the
/// graph/material RON asset pattern (`graph::GraphAsset`, `assets::MaterialAsset`).
#[derive(Asset, Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct BiomeLibraryAsset {
    /// The flat-colour material palette. `TerrainMatId(i)` indexes this array.
    pub materials: Vec<TerrainSurfaceMaterial>,
    /// One [`BiomeDef`] per demo biome. Parallel to [`BiomeId`] (entry `i` defines biome with `id == i`);
    /// validated at compile time ([`BiomeLibrary::compile`]).
    pub biomes: Vec<BiomeDef>,
}

impl crate::assets::Asset for BiomeLibraryAsset {
    const EXTENSION: &'static str = "biomes.ron";
}

/// The compiled, runtime-queried biome library — a `Resource` holding the validated palette + per-biome
/// columns. Built from a [`BiomeLibraryAsset`] via [`BiomeLibrary::compile`]; the query API
/// ([`strata_material`], [`terrain_color`]) reads it. Later stages flatten this into a GPU storage table.
#[derive(Resource, Reflect, Clone, Debug, Default)]
pub struct BiomeLibrary {
    /// `TerrainMatId(i)` → material. (Owned copy of the asset's palette.)
    pub materials: Vec<TerrainSurfaceMaterial>,
    /// `BiomeId as usize` → its column definition (indexed; always length [`BiomeId::ALL`]).
    pub biomes: Vec<BiomeDef>,
}

/// Errors compiling a [`BiomeLibraryAsset`] into a [`BiomeLibrary`] — every referenced material must
/// exist and every demo biome must be defined exactly once, in id order.
#[derive(Debug, PartialEq)]
pub enum BiomeCompileError {
    /// `biomes.len()` did not equal the number of [`BiomeId`] variants.
    BiomeCountMismatch { expected: usize, found: usize },
    /// A [`TerrainMatId`] referenced by a biome is out of range of the palette.
    MissingMaterial { biome: usize, id: TerrainMatId },
}

impl std::fmt::Display for BiomeCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BiomeCompileError::BiomeCountMismatch { expected, found } => {
                write!(f, "biome count mismatch: expected {expected}, found {found}")
            }
            BiomeCompileError::MissingMaterial { biome, id } => {
                write!(f, "biome {biome} references missing material {id:?}")
            }
        }
    }
}

impl std::error::Error for BiomeCompileError {}

impl BiomeLibrary {
    /// Compile + validate an authored [`BiomeLibraryAsset`] into a query-ready library: every
    /// `TerrainMatId` referenced by a biome (surface, strata, bedrock) must index the palette, and there
    /// must be exactly one biome per [`BiomeId`] (in id order). Total + deterministic.
    pub fn compile(asset: &BiomeLibraryAsset) -> Result<BiomeLibrary, BiomeCompileError> {
        let n_mats = asset.materials.len();
        if asset.biomes.len() != BiomeId::ALL.len() {
            return Err(BiomeCompileError::BiomeCountMismatch {
                expected: BiomeId::ALL.len(),
                found: asset.biomes.len(),
            });
        }
        let valid = |id: TerrainMatId| (id.0 as usize) < n_mats;
        for (bi, b) in asset.biomes.iter().enumerate() {
            for id in std::iter::once(b.surface)
                .chain(b.strata.iter().map(|s| s.material))
                .chain(std::iter::once(b.bedrock))
            {
                if !valid(id) {
                    return Err(BiomeCompileError::MissingMaterial { biome: bi, id });
                }
            }
        }
        Ok(BiomeLibrary {
            materials: asset.materials.clone(),
            biomes: asset.biomes.clone(),
        })
    }

    /// The [`BiomeDef`] for `id` (indexed; the library is validated to hold one per [`BiomeId`]).
    #[inline]
    pub fn biome(&self, id: BiomeId) -> &BiomeDef {
        &self.biomes[id as usize]
    }

    /// The material for `id`. Panics if out of range — the library is validated at compile time, so any
    /// id obtained from a [`BiomeDef`] in THIS library is in range by construction.
    #[inline]
    pub fn material(&self, id: TerrainMatId) -> &TerrainSurfaceMaterial {
        &self.materials[id.0 as usize]
    }
}

// ============================================================================================
// GPU strata table (reusable: preview slice AND the Stage-3 in-world surface shader)
// ============================================================================================

/// Max strata layers (excluding bedrock) the flattened GPU table stores per biome. The demo biomes use 3
/// (surface band + sub-surface + stone); a fixed cap keeps the table a simple dense array the shader can
/// index by `(biome * GPU_STRATA_MAX_LAYERS + layer)`. Extra authored layers beyond this are clamped into
/// the last slot at flatten time (and a debug-assert fires in tests).
pub const GPU_STRATA_MAX_LAYERS: usize = 6;

/// One biome's flattened strata column for the GPU: each layer's resolved `preview_color` (linear RGBA)
/// and its cumulative depth *bottom* (metres below the original surface), plus the surface colour and the
/// bedrock colour. The shader walks `cum_bottom` to find the first layer whose bottom exceeds `depth`,
/// taking its colour; past the last real layer it uses `bedrock`. This is exactly the CPU
/// [`strata_material`] walk, pre-resolved to colours so no Whittaker/strata logic is ported to WGSL.
///
/// `bytemuck`-able + `std430`-friendly layout (all `[f32;4]` / `f32` / `u32`, 16-byte aligned by padding)
/// so it uploads straight into a storage/uniform buffer in both the preview and Stage-3 pipelines.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct GpuStrataColumn {
    /// `surface` material colour (depth ≤ 0), linear RGBA.
    pub surface_color: [f32; 4],
    /// Per-layer colour (linear RGBA), `[GPU_STRATA_MAX_LAYERS]`; slots past `layer_count` are unused.
    pub layer_color: [[f32; 4]; GPU_STRATA_MAX_LAYERS],
    /// Per-layer cumulative BOTTOM depth (metres). `layer_bottom[i]` = Σ thickness up to & incl. layer i.
    /// Stored as `[f32;4]` groups so the whole struct stays 16-byte aligned for std140/std430.
    pub layer_bottom: [f32; GPU_STRATA_MAX_LAYERS],
    /// Bedrock colour (linear RGBA) — fills depth below the last layer.
    pub bedrock_color: [f32; 4],
    /// Number of real layers (≤ [`GPU_STRATA_MAX_LAYERS`]).
    pub layer_count: u32,
    /// Padding to keep the trailing scalars 16-byte aligned (3 × u32).
    pub _pad: [u32; 3],
}

impl Default for GpuStrataColumn {
    fn default() -> Self {
        Self {
            surface_color: [0.0; 4],
            layer_color: [[0.0; 4]; GPU_STRATA_MAX_LAYERS],
            layer_bottom: [0.0; GPU_STRATA_MAX_LAYERS],
            bedrock_color: [0.0; 4],
            layer_count: 0,
            _pad: [0; 3],
        }
    }
}

impl BiomeLibrary {
    /// Flatten this library into the per-biome GPU strata table (one [`GpuStrataColumn`] per [`BiomeId`],
    /// in id order). Resolves every `TerrainMatId` to its material's [`preview_color`](TerrainSurfaceMaterial::preview_color)
    /// and accumulates layer thicknesses into cumulative bottom-depths — the exact data the preview-slice
    /// shader AND the Stage-3 in-world surface shader index by `(biome, depth)`. Built REUSABLY (this is
    /// the one flatten both consumers call), not throwaway. A biome with more than [`GPU_STRATA_MAX_LAYERS`]
    /// strata clamps the overflow into the last slot (its bottom extended to the last layer's bottom).
    pub fn gpu_strata_table(&self) -> Vec<GpuStrataColumn> {
        BiomeId::ALL
            .iter()
            .map(|&id| {
                let def = self.biome(id);
                let mut col = GpuStrataColumn {
                    surface_color: self.material(def.surface).preview_color(),
                    bedrock_color: self.material(def.bedrock).preview_color(),
                    ..Default::default()
                };
                let mut cum = 0.0_f32;
                let mut n = 0usize;
                for layer in &def.strata {
                    cum += layer.thickness;
                    let slot = n.min(GPU_STRATA_MAX_LAYERS - 1);
                    col.layer_color[slot] = self.material(layer.material).preview_color();
                    col.layer_bottom[slot] = cum;
                    if n < GPU_STRATA_MAX_LAYERS {
                        n += 1;
                    }
                }
                debug_assert!(
                    def.strata.len() <= GPU_STRATA_MAX_LAYERS,
                    "biome {:?} has {} strata > GPU_STRATA_MAX_LAYERS {GPU_STRATA_MAX_LAYERS}",
                    id,
                    def.strata.len()
                );
                col.layer_count = n as u32;
                col
            })
            .collect()
    }
}

/// Loads `assets/worldgen/biomes.ron` into a [`BiomeLibraryAsset`] — plain RON deserialization (mirrors
/// `GraphAssetLoader` / `MaterialAssetLoader`).
#[derive(Default, bevy::reflect::TypePath)]
pub struct BiomeLibraryAssetLoader;

/// Errors surfaced while loading the biome library asset.
#[derive(Debug)]
pub enum BiomeLoadError {
    Io(std::io::Error),
    Ron(ron::error::SpannedError),
}

impl std::fmt::Display for BiomeLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BiomeLoadError::Io(e) => write!(f, "biomes io: {e}"),
            BiomeLoadError::Ron(e) => write!(f, "biomes ron: {e}"),
        }
    }
}

impl std::error::Error for BiomeLoadError {}

impl From<std::io::Error> for BiomeLoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<ron::error::SpannedError> for BiomeLoadError {
    fn from(e: ron::error::SpannedError) -> Self {
        Self::Ron(e)
    }
}

impl AssetLoader for BiomeLibraryAssetLoader {
    type Asset = BiomeLibraryAsset;
    type Settings = ();
    type Error = BiomeLoadError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _ctx: &mut LoadContext<'_>,
    ) -> Result<BiomeLibraryAsset, BiomeLoadError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let asset = ron::de::from_bytes::<BiomeLibraryAsset>(&bytes)?;
        Ok(asset)
    }

    fn extensions(&self) -> &[&str] {
        // Bevy matches on the final extension; `.biomes.ron` ends in `ron` (same convention as graphs).
        &["biomes.ron", "ron"]
    }
}

// ============================================================================================
// Query API — the SSOT later stages call
// ============================================================================================

/// The surface biome at world `(wx, wz)` for `seed` — composes the climate fields ([`temperature`] /
/// [`humidity`]) and [`classify`]. The Stage-2 bake will call this per chunk-texel to write the biome
/// channel. Deterministic + bit-portable.
#[inline]
pub fn surface_biome(wx: f64, wz: f64, seed: u64) -> BiomeSample {
    let t = temperature(wx, wz, seed);
    let h = humidity(wx, wz, seed);
    classify(t, h)
}

/// The material at `depth` metres below the original surface for `biome`, walking its strata column:
/// `depth ≤ 0` (or within the first band) → surface material; accumulating each [`StrataLayer`]'s
/// `thickness`; below the last stratum (≥ the stone bottom) → bedrock. Total (any `depth` maps to a
/// material) and deterministic.
///
/// The column is read as: `strata[0]` covers `[0, t0)`, `strata[1]` covers `[t0, t0+t1)`, … The biome's
/// `surface` material IS the material a Stage-3 shader paints at depth 0 (the top of `strata[0]`); we
/// keep `surface` as a distinct field (rather than assuming `strata[0].material`) so the surface can
/// later diverge from the first stratum (e.g. a thin grass skin over a thicker "topsoil" stratum). Here
/// depth 0 returns `surface`; the moment `depth > 0` we walk `strata`.
pub fn strata_material(biome: BiomeId, depth: f64, lib: &BiomeLibrary) -> TerrainMatId {
    let def = lib.biome(biome);
    if depth <= 0.0 {
        return def.surface;
    }
    let mut top = 0.0_f64;
    for layer in &def.strata {
        let bottom = top + layer.thickness as f64;
        if depth < bottom {
            return layer.material;
        }
        top = bottom;
    }
    def.bedrock
}

/// Convenience: the full climate→biome→strata→colour compose at world `(wx, wz)` and `depth` metres
/// below the original surface. Returns the linear RGBA of the material that occupies that point. For
/// tests / CPU picking / preview (Stage 3 does this per-fragment on the GPU). At `depth == 0` this is the
/// **surface** biome's surface-material colour.
pub fn terrain_color(wx: f64, wz: f64, depth: f64, seed: u64, lib: &BiomeLibrary) -> [f32; 4] {
    let sample = surface_biome(wx, wz, seed);
    let mat = strata_material(sample.primary, depth, lib);
    lib.material(mat).preview_color()
}

#[cfg(test)]
mod tests;
