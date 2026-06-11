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
use bevy::render::render_resource::ShaderType;
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
    for (dist, neigh, _axis) in neighbour_borders(t, h, primary) {
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
/// NOTE: this is CLIMATE-space, so its WORLD width varies with the local climate gradient (steep gradient →
/// sharp border, gentle → soft) — widening it can't fix that AND eats the ~0.33-wide cells. The real fix for
/// uniformly-soft borders is a WORLD-space blend (divide by the climate gradient at bake time).
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

/// Which climate axis a partition border lies on. Every Whittaker border in this partition is a pure
/// temperature- or humidity-threshold line, so the world-space distance to it (used by the bake-time
/// [`surface_biome_world`] blend) is the climate distance divided by that ONE axis's world gradient.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ClimateAxis {
    Temperature,
    Humidity,
}

/// For the cell `primary` at `(t,h)`, the borders to its neighbouring cells, each as
/// `(distance_to_border, neighbour_biome, axis)`. Only borders that actually separate `primary` from a
/// DIFFERENT biome are returned (so `secondary != primary` whenever a finite distance exists). The axis
/// names which climate field the border threshold is on (so the bake can normalise by the right
/// gradient). Pure basic ops.
fn neighbour_borders(t: f64, h: f64, primary: BiomeId) -> Vec<(f64, BiomeId, ClimateAxis)> {
    let mut out = Vec::new();
    // Helper: the biome just across a temperature/humidity boundary, evaluated by nudging the sample
    // across it and re-partitioning (keeps the neighbour identity consistent with `primary_biome`).
    let mut push = |dist: f64, neigh: BiomeId, axis: ClimateAxis| {
        if neigh != primary {
            out.push((dist.abs(), neigh, axis));
        }
    };
    use ClimateAxis::{Humidity, Temperature};

    match primary {
        BiomeId::Desert => {
            // Border to the mid tier at T_WARM; neighbour is Plains or Forest by current humidity.
            let neigh = if h >= H_MID_WET { BiomeId::Forest } else { BiomeId::Plains };
            push(t - T_WARM, neigh, Temperature);
        }
        BiomeId::Forest => {
            // Hot border (→ Desert) and the humidity border (→ Plains).
            push(t - T_WARM, BiomeId::Desert, Temperature);
            push(h - H_MID_WET, BiomeId::Plains, Humidity);
            // Cold border (→ Snowy, since H is high here).
            push(t - T_COLD, BiomeId::Snowy, Temperature);
        }
        BiomeId::Plains => {
            push(t - T_WARM, BiomeId::Desert, Temperature);
            push(h - H_MID_WET, BiomeId::Forest, Humidity);
            // Cold border (→ Tundra, since H is low here).
            push(t - T_COLD, BiomeId::Tundra, Temperature);
        }
        BiomeId::Tundra => {
            // Warm border (→ Plains, low H) and the humidity border (→ Snowy).
            push(t - T_COLD, BiomeId::Plains, Temperature);
            push(h - H_COLD_WET, BiomeId::Snowy, Humidity);
        }
        BiomeId::Snowy => {
            // Warm border (→ Forest, high H) and the humidity border (→ Tundra).
            push(t - T_COLD, BiomeId::Forest, Temperature);
            push(h - H_COLD_WET, BiomeId::Tundra, Humidity);
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
    /// Transition SOFTNESS in WORLD metres: how far THIS material cross-fades into an adjacent material at
    /// a surface boundary (a biome border, an altitude cap, a slope/patch edge — see [`SurfaceCond`]). The
    /// worldgen surface resolver ([`resolve_surface`]) uses it to turn a hard rule edge into a metres-wide
    /// blend; the shader just renders the baked `(mat_a, mat_b, weight)`. `#[serde(default)]` so older RON
    /// without the field still loads (0 ⇒ a hard edge). A RENDER attribute (no `HEIGHT_GEN_VERSION` tie).
    #[serde(default = "default_material_blend")]
    pub blend: f32,
}

/// Default [`TerrainSurfaceMaterial::blend`] (metres) when the RON omits it — a gentle few-metre fade so an
/// unauthored material still blends rather than hard-edging.
fn default_material_blend() -> f32 {
    4.0
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

/// A condition under which a [`SurfaceLayer`]'s material appears, as a pure function of world position /
/// surface geometry / noise — evaluated by the worldgen at BAKE time (NOT the shader), yielding a smooth
/// presence weight in `[0,1]`. This is the extensible "where is this material on the surface" vocabulary
/// (leverages x/y/z, slope, noise…); add variants without touching the shader. Bit-portable (value noise +
/// basic f64 ops); a RENDER attribute (no `HEIGHT_GEN_VERSION` tie).
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub enum SurfaceCond {
    /// Always present (weight 1). The biome's base surface — the bottom of the layer stack.
    Base,
    /// Above a world-Y altitude: weight ramps `0→1` from `start` m to `full` m (a snow line / the lower
    /// edge of a rock cap). `full < start` inverts it (present BELOW an altitude).
    AboveY { start: f32, full: f32 },
    /// On steep ground: weight ramps `0→1` as the surface-normal `.y` (cos of the slope from vertical)
    /// drops from `gentle` (cos, ~1 = flat, weight 0) to `steep` (cos, smaller = steeper, weight 1). Cliffs.
    Slope { gentle: f32, steep: f32 },
    /// A low-frequency value-noise PATCH: weight ramps across `threshold ± softness` of the `[0,1]` noise at
    /// `wavelength` m with a `seed` salt — sub-areas like a flower field in plains. Soft-edged sub-biomes.
    Patch { wavelength: f32, threshold: f32, softness: f32, seed: u32 },
}

/// One layer in a biome's SURFACE stack (the undug top, bottom→top): its `material` appears where the `when`
/// conditions fire, over-blending the layers below it (see [`resolve_surface`]). `when` is AND-combined — the
/// layer's weight is the PRODUCT of its conditions' weights — so a rule can require several things at once
/// (e.g. a flower patch = noise AND flat ground, keeping flowers off steep mountainsides). An EMPTY `when`
/// means "always" (weight 1). The data-driven replacement for the old hardcoded shader surface treatment.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SurfaceLayer {
    pub material: TerrainMatId,
    #[serde(default)]
    pub when: Vec<SurfaceCond>,
}

/// A biome's full definition: its surface material (top, undug), the ordered sub-surface strata, and the
/// bedrock material that fills everything below the strata (down to the world bedrock floor). Referenced
/// by [`BiomeId`] in the library.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct BiomeDef {
    /// Human-readable biome name.
    pub name: String,
    /// The top (depth 0) material for the VOLUMETRIC strata column (dug walls). The undug RENDER surface is
    /// chosen by [`surface_rules`] (which default to just this material). Kept for the depth walk + as the
    /// surface-stack base.
    pub surface: TerrainMatId,
    /// SURFACE material stack (bottom→top) chosen by world position / slope / noise — the data-driven undug
    /// surface (snow caps, cliff rock, patches). Bottom layer is the base; higher layers over-blend it where
    /// their [`SurfaceCond`] fires. `#[serde(default)]` ⇒ EMPTY means "just the `surface` material" (old
    /// behaviour); the resolver treats an empty stack as a single `Base` layer of `surface`.
    #[serde(default)]
    pub surface_rules: Vec<SurfaceLayer>,
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
        // `biomes.ron` is an async asset, so the library can be EMPTY (its `Default`) for the first frames
        // while it loads — but `sync_terrain_detail_params` + the preview flatten it EVERY frame. Return a
        // zeroed table instead of indexing an empty `Vec` (this was a launch panic: `biome()`/`material()`
        // index out of bounds at biome.rs while the asset was still loading). Once compiled the library has
        // exactly one validated entry per `BiomeId`.
        if self.biomes.len() != BiomeId::ALL.len() {
            return vec![GpuStrataColumn::default(); BiomeId::ALL.len()];
        }
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

// ============================================================================================
// GPU strata table — std140 UNIFORM layout (the ONE flatten shared by the editor preview AND the
// in-world Stage-3 terrain-surface material; do NOT duplicate this for either consumer).
// ============================================================================================

/// Demo biome count — `BiomeId::ALL.len()`, the fixed length of the GPU strata table. The shaders that
/// index the table (`worldgen_preview.wgsl`, `terrain_surface.wgsl`) declare a `BIOME_COUNT` const that
/// MUST equal this (asserted by the shader-dims build tests).
pub const BIOME_COUNT: usize = BiomeId::ALL.len();

/// GPU mirror of [`GpuStrataColumn`] (one biome's flattened strata column) laid out for **std140**: the
/// [`GPU_STRATA_MAX_LAYERS`] cumulative layer bottoms are packed into 2 `Vec4` lanes so the array stays
/// 16-byte aligned. Built from the CPU SSOT [`BiomeLibrary::gpu_strata_table`]. This is the SHARED flatten:
/// the editor biome/slice preview AND the in-world terrain surface material both upload [`StrataTableStd`]
/// from it, so the two NEVER diverge.
///
/// `_pad` is a `UVec3` (= `vec3<u32>`), NOT `[u32; 3]`: a uniform array element's stride must be 16-aligned,
/// which a `u32` array (stride 4) can't satisfy — encase panics at encode time (the `[u32; N]` gotcha — see
/// `strata_table_is_valid_std140_uniform`). Keep every uniform field a `Vec*`/`UVec*`.
#[derive(ShaderType, Clone, Copy, Default)]
pub struct GpuStrataColumnStd {
    pub surface_color: Vec4,
    pub layer_color: [Vec4; GPU_STRATA_MAX_LAYERS],
    /// `GPU_STRATA_MAX_LAYERS` (= 6) floats packed: `lane0.xyzw + lane1.xy`.
    pub layer_bottom: [Vec4; 2],
    pub bedrock_color: Vec4,
    pub layer_count: u32,
    pub _pad: UVec3,
}

impl From<&GpuStrataColumn> for GpuStrataColumnStd {
    fn from(c: &GpuStrataColumn) -> Self {
        let mut layer_color = [Vec4::ZERO; GPU_STRATA_MAX_LAYERS];
        for (i, col) in c.layer_color.iter().enumerate() {
            layer_color[i] = Vec4::from_array(*col);
        }
        // Pack the layer bottoms into 2 vec4 lanes (xyzw, then xy…). `GPU_STRATA_MAX_LAYERS <= 8` (asserted
        // in tests) so they always fit.
        let mut bottom = [Vec4::ZERO; 2];
        for (i, &b) in c.layer_bottom.iter().enumerate() {
            bottom[i / 4][i % 4] = b;
        }
        Self {
            surface_color: Vec4::from_array(c.surface_color),
            layer_color,
            layer_bottom: bottom,
            bedrock_color: Vec4::from_array(c.bedrock_color),
            layer_count: c.layer_count,
            _pad: UVec3::ZERO,
        }
    }
}

/// The full per-biome strata table uniform (one column per [`BiomeId`], id order) — the GPU side of
/// [`BiomeLibrary::gpu_strata_table`]. A fixed-size array sized to [`BIOME_COUNT`]. SHARED by the preview
/// and the in-world surface material; built once via [`StrataTableStd::from_library`].
#[derive(ShaderType, Clone, Copy)]
pub struct StrataTableStd {
    pub columns: [GpuStrataColumnStd; BIOME_COUNT],
}

impl Default for StrataTableStd {
    fn default() -> Self {
        Self { columns: [GpuStrataColumnStd::default(); BIOME_COUNT] }
    }
}

impl StrataTableStd {
    /// Flatten a [`BiomeLibrary`] into the GPU table (via the CPU SSOT [`BiomeLibrary::gpu_strata_table`]),
    /// clamped/padded to [`BIOME_COUNT`]. The single flatten both the preview and the in-world material call.
    pub fn from_library(lib: &BiomeLibrary) -> Self {
        let table = lib.gpu_strata_table();
        let mut columns = [GpuStrataColumnStd::default(); BIOME_COUNT];
        for (i, c) in table.iter().take(BIOME_COUNT).enumerate() {
            columns[i] = GpuStrataColumnStd::from(c);
        }
        Self { columns }
    }
}

/// Max materials in the GPU surface palette ([`MaterialPaletteStd`]) — the `TerrainMatId(i)` the baked
/// surface map (`mat_a`, `mat_b`) indexes. A fixed cap so the uniform is a sized array; the WGSL palette
/// declares the SAME `GPU_MAX_MATERIALS`. Bump (here + the shader const) if a world needs more than this.
pub const GPU_MAX_MATERIALS: usize = 32;

/// One palette material laid out for **std140**: `color` (linear RGBA) and `props` (`x` = roughness, rest
/// reserved). The worldgen bakes a surface map of material IDs plus a blend weight; the shader looks the
/// colour and roughness up here and mixes, so all the "which material is on the surface" logic lives in the
/// bake, not the shader. The per-material `blend` softness is consumed CPU-side by [`resolve_surface`] to
/// compute the baked weight, so it is not uploaded. `Vec4`s only — the `scalar`-array std140 stride gotcha.
#[derive(ShaderType, Clone, Copy, Default)]
pub struct GpuMaterialStd {
    pub color: Vec4,
    /// `x` = roughness; `y,z,w` reserved (metallic / future PBR scalars).
    pub props: Vec4,
}

/// The flat material palette uniform — `TerrainMatId(i)` → colour + roughness, sized to [`GPU_MAX_MATERIALS`].
/// Built from the live [`BiomeLibrary`] palette by [`MaterialPaletteStd::from_library`]; uploaded alongside
/// the strata table for the terrain-surface material. `count` is the number of REAL materials (the rest are
/// zeroed); the shader clamps `mat_a`/`mat_b` into `[0, count)`.
#[derive(ShaderType, Clone, Copy)]
pub struct MaterialPaletteStd {
    pub materials: [GpuMaterialStd; GPU_MAX_MATERIALS],
    pub count: u32,
    pub _pad: UVec3,
}

impl Default for MaterialPaletteStd {
    fn default() -> Self {
        Self { materials: [GpuMaterialStd::default(); GPU_MAX_MATERIALS], count: 0, _pad: UVec3::ZERO }
    }
}

impl MaterialPaletteStd {
    /// Flatten the library's material palette into the GPU uniform (clamped to [`GPU_MAX_MATERIALS`]). The
    /// SSOT the bake's `mat_a`/`mat_b` indices resolve against in the shader. Robust to an unloaded library
    /// (empty `materials` ⇒ a zeroed palette, `count = 0` — never panics, mirrors [`gpu_strata_table`]).
    pub fn from_library(lib: &BiomeLibrary) -> Self {
        let mut out = Self::default();
        let n = lib.materials.len().min(GPU_MAX_MATERIALS);
        for (i, m) in lib.materials.iter().take(n).enumerate() {
            out.materials[i] = GpuMaterialStd {
                color: Vec4::from_array(m.base_color),
                props: Vec4::new(m.roughness, 0.0, 0.0, 0.0),
            };
        }
        out.count = n as u32;
        out
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

/// Like [`surface_biome`] but the `blend` ramps over a fixed WORLD distance (`transition_m` metres)
/// instead of [`classify`]'s fixed climate-space [`BLEND_BAND`] — so biome borders are uniformly soft
/// regardless of the local climate gradient (a steep-gradient border is no longer a hard line, a gentle
/// one no longer a smear). Every Whittaker border is one axis-threshold, so the world distance to it is
/// the climate distance divided by that axis's world gradient (a central difference on the true f64
/// field — smooth, computed once at BAKE time, NOT per-fragment on bilinear-sampled texels, which stepped
/// at the texel grid). `primary`/`secondary` are unchanged; only the ramp width/shape differs. The bake
/// SSOT for the in-world surface colour. Determinism note: a RENDER attribute (not keyed by
/// `HEIGHT_GEN_VERSION`); uses only basic f64 ops on the portable climate fields.
pub fn surface_biome_world(wx: f64, wz: f64, seed: u64, transition_m: f64) -> BiomeSample {
    let t = temperature(wx, wz, seed).clamp(0.0, 1.0);
    let h = humidity(wx, wz, seed).clamp(0.0, 1.0);
    let primary = primary_biome(t, h);

    // Nearest cell border + the biome across it + which axis its threshold is on.
    let mut best = f64::INFINITY;
    let mut secondary = primary;
    let mut best_axis = ClimateAxis::Temperature;
    for (dist, neigh, axis) in neighbour_borders(t, h, primary) {
        if dist < best {
            best = dist;
            secondary = neigh;
            best_axis = axis;
        }
    }

    let blend = if best.is_finite() {
        // climate distance to the iso-threshold → WORLD distance: divide by |∇axis| (the perpendicular
        // distance to an axis-aligned iso-line is |value − threshold| / |∇value|). Gradient by central
        // difference on the true field at a 16 m delta (smooth at the ≥1 km climate wavelength).
        let g = climate_axis_gradient_mag(wx, wz, seed, best_axis).max(1e-9);
        let world_dist = best / g;
        // smoothstep ramp: 1 at the border (dist 0), 0 once `world_dist ≥ transition_m`. C1 in distance;
        // across the border `best → 0` on BOTH sides (primary/secondary swap) ⇒ blend → 1 both sides ⇒
        // the colour mix meets at 50/50 — continuous, no seam.
        let x = (world_dist / transition_m.max(1e-3)).clamp(0.0, 1.0);
        (1.0 - x * x * (3.0 - 2.0 * x)) as f32
    } else {
        0.0
    };

    let secondary = if blend <= 0.0 { primary } else { secondary };
    BiomeSample { primary, secondary, blend }
}

/// |∇(climate axis)| at world `(wx, wz)` by central difference (delta `D`). The world-space gradient
/// magnitude of the temperature/humidity field; divides the climate border distance into a world
/// distance in [`surface_biome_world`]. `D = 16 m` is small vs the ≥1 km climate wavelength, so the
/// estimate is the smooth local slope.
fn climate_axis_gradient_mag(wx: f64, wz: f64, seed: u64, axis: ClimateAxis) -> f64 {
    const D: f64 = 16.0;
    let f = |x: f64, z: f64| match axis {
        ClimateAxis::Temperature => temperature(x, z, seed),
        ClimateAxis::Humidity => humidity(x, z, seed),
    };
    let gx = (f(wx + D, wz) - f(wx - D, wz)) / (2.0 * D);
    let gz = (f(wx, wz + D) - f(wx, wz - D)) / (2.0 * D);
    (gx * gx + gz * gz).sqrt()
}

/// Per-field salt for [`SurfaceCond::Patch`] noise so it decorrelates from the climate streams.
const PATCH_SALT: u64 = 0xA24B_AED4_EAF7_1B9D;

/// C1 smoothstep on `[edge0, edge1]` (handles `edge0 == edge1` and reversed edges). f64, bit-portable.
fn smoothstep_f64(edge0: f64, edge1: f64, x: f64) -> f64 {
    if (edge1 - edge0).abs() < 1e-12 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Low-frequency value-noise patch field at world `(wx,wz)`, normalised to `[0,1]` — drives
/// [`SurfaceCond::Patch`] (flower fields etc.). One fBm octave at `wavelength` m, salted by `seed`.
fn patch_noise(wx: f64, wz: f64, wavelength: f64, salt: u32, world_seed: u64) -> f64 {
    let p = FbmParams {
        octaves: 1,
        base_freq: 1.0 / wavelength.max(1.0),
        lacunarity: 2.0,
        gain: 0.5,
        amplitude: 1.0,
        seed: climate_seed(world_seed, PATCH_SALT ^ (salt as u64)),
    };
    (fbm_height(wx, wz, &p) * 0.5 + 0.5).clamp(0.0, 1.0)
}

/// The presence weight `[0,1]` of one [`SurfaceCond`] at a point (`surf_y` = surface altitude, `n_y` = cos
/// of the surface slope). The smooth bake-time evaluation of "is this material's rule firing here".
fn cond_weight(cond: SurfaceCond, wx: f64, wz: f64, surf_y: f64, n_y: f64, seed: u64) -> f64 {
    match cond {
        SurfaceCond::Base => 1.0,
        SurfaceCond::AboveY { start, full } => smoothstep_f64(start as f64, full as f64, surf_y),
        // n_y drops from `gentle` (flat, 0) to `steep` (steep, 1): ramp on the REVERSED edges.
        SurfaceCond::Slope { gentle, steep } => smoothstep_f64(gentle as f64, steep as f64, n_y),
        SurfaceCond::Patch { wavelength, threshold, softness, seed: salt } => {
            let v = patch_noise(wx, wz, wavelength as f64, salt, seed);
            let s = (softness as f64).max(1e-4);
            smoothstep_f64(threshold as f64 - s, threshold as f64 + s, v)
        }
    }
}

/// The resolved undug SURFACE at a point: the two dominant materials + a blend `weight` (`0` = all `mat_a`,
/// `0.5` = an even 50/50). The bake writes this per texel; the shader looks the palette colours up and mixes
/// (and bilinear-interpolates the resolved COLOUR across texels, so material-pair boundaries don't step).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceBlend {
    pub mat_a: u16,
    pub mat_b: u16,
    /// Fraction toward `mat_b` in `[0, 0.5]` (mirrors the biome-blend convention the shader already mixes).
    pub weight: f32,
}

/// Worldgen SSOT for the undug RENDER surface at world `(wx,wz)` (surface altitude `surf_y`, surface-normal
/// cos `n_y`), for the baked `biome` sample. Composes (1) each biome's SURFACE STACK — base material
/// over-blended by its altitude-cap / cliff / patch [`SurfaceLayer`]s — and (2) the biome-border cross-fade
/// (primary↔secondary by the baked `biome.blend`). Reduces the weighted material set to its top two. Shared
/// by the bake AND the editor preview so they never diverge. Bit-portable (RENDER attribute, no version tie).
pub fn resolve_surface(
    wx: f64,
    wz: f64,
    surf_y: f64,
    n_y: f64,
    biome: BiomeSample,
    seed: u64,
    lib: &BiomeLibrary,
) -> SurfaceBlend {
    let mut acc: Vec<(u16, f64)> = Vec::new();
    let border = (biome.blend.clamp(0.0, 1.0) as f64) * 0.5; // fraction toward the neighbour biome
    accumulate_biome_surface(&mut acc, biome.primary, wx, wz, surf_y, n_y, seed, lib, 1.0 - border);
    if border > 0.0 && biome.secondary != biome.primary {
        accumulate_biome_surface(&mut acc, biome.secondary, wx, wz, surf_y, n_y, seed, lib, border);
    }
    top_two(&acc)
}

/// Evaluate one biome's surface STACK (base + over-blended rule layers) at a point and merge its per-material
/// weights (scaled by `scale`) into `acc`. The stack is seeded with the biome's `surface` material (weight 1)
/// then each rule layer that fires scales everything below by `(1-w)` and adds its material with weight `w`
/// (a standard back-to-front over-blend), so higher layers (caps, cliffs, patches) sit on top.
#[allow(clippy::too_many_arguments)]
fn accumulate_biome_surface(
    acc: &mut Vec<(u16, f64)>,
    biome: BiomeId,
    wx: f64,
    wz: f64,
    surf_y: f64,
    n_y: f64,
    seed: u64,
    lib: &BiomeLibrary,
    scale: f64,
) {
    let def = lib.biome(biome);
    let mut stack: Vec<(u16, f64)> = vec![(def.surface.0, 1.0)];
    for layer in &def.surface_rules {
        // AND-combine the layer's conditions: its weight is the product (empty ⇒ 1 = always).
        let w: f64 = layer.when.iter().map(|c| cond_weight(*c, wx, wz, surf_y, n_y, seed)).product();
        if w <= 0.0 {
            continue;
        }
        for e in stack.iter_mut() {
            e.1 *= 1.0 - w;
        }
        merge_weight(&mut stack, layer.material.0, w);
    }
    for (m, w) in stack {
        merge_weight(acc, m, w * scale);
    }
}

/// Accumulate `w` onto material `id` in a `(id, weight)` list (collapsing duplicates).
fn merge_weight(list: &mut Vec<(u16, f64)>, id: u16, w: f64) {
    if let Some(e) = list.iter_mut().find(|e| e.0 == id) {
        e.1 += w;
    } else {
        list.push((id, w));
    }
}

/// Reduce a weighted material set to its two heaviest → a [`SurfaceBlend`] (`weight` = `wb/(wa+wb)` ∈ `[0,
/// 0.5]`). One material (or empty) → `mat_a == mat_b`, weight 0.
fn top_two(acc: &[(u16, f64)]) -> SurfaceBlend {
    let mut a: Option<(u16, f64)> = None;
    let mut b: Option<(u16, f64)> = None;
    for &(id, w) in acc {
        if a.is_none() || w > a.unwrap().1 {
            b = a;
            a = Some((id, w));
        } else if b.is_none() || w > b.unwrap().1 {
            b = Some((id, w));
        }
    }
    match (a, b) {
        (Some((ma, wa)), Some((mb, wb))) if wa + wb > 0.0 => {
            SurfaceBlend { mat_a: ma, mat_b: mb, weight: (wb / (wa + wb)) as f32 }
        }
        (Some((ma, _)), _) => SurfaceBlend { mat_a: ma, mat_b: ma, weight: 0.0 },
        _ => SurfaceBlend { mat_a: 0, mat_b: 0, weight: 0.0 },
    }
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
