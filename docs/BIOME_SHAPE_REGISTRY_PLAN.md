# Biome shape registry — biomes own their terrain shape (node-graph Phase 2)

## Goal

Each biome owns a **terrain-shape graph**; a point's height is the **climate-weighted blend** of the
biomes present there. Unifies shape with materials: the SAME climate classification (`temperature`/
`humidity` → `BiomeSample`) that picks a biome's surface materials also places its shape. Replaces the
single hand-wired `world.graph.ron` (one graph with a manual plains↔mountains `Mix`) with per-biome graphs
auto-blended by the classifier.

## Architecture

- **`BiomeShapeSet`** (`Arc`, held by `HeightLayer`): `BiomeId → Arc<Graph>`. A biome without its own graph
  falls back to a shared default graph. Built from the `WorldGraph`/registry resource, threaded through
  `LayerManager` exactly like the current single `graph` (`set_graph` → `BiomeShapeSet`).
- **`sample_world`** becomes: `classify(temperature, humidity) → (primary, secondary, blend)` → eval
  `shapes[primary]` + `shapes[secondary]` as `Field`s → blend by `w = blend·0.5` with `Field::mix` (the
  autodiff carries the climate-weight gradient via the product rule, so normals stay analytic). **Parity
  guard:** if the two biomes resolve to the SAME `Arc<Graph>` (`Arc::ptr_eq`) — always true while every
  biome shares the default — eval ONCE and return it bit-identically (no blend math) ⇒ Stage B1 is
  behaviour-preserving, no `HEIGHT_GEN_VERSION` bump.
- **Climate as `Field`** (needed for `∇w`): `temperature_grad`/`humidity_grad` in `biome.rs` — `fbm_height_grad`
  on the climate `FbmParams` through the affine `normalize_climate` (gradient ×= `0.5/bound`, zeroed at the
  clamp rails). The value EQUALS `temperature()`/`humidity()` (the material-classifier SSOT), so shape +
  materials use one climate field.
- **Grid hot path** (`sample_world_grid`, gen): classify the column; the COMMON case (all points one biome,
  `blend≈0`) evals that one graph columnar (today's speed). A border column evals only the distinct biome
  graphs present (≤ a few) columnar, then blends per point. Never per-point-scalar in the common case.

## Stages (each builds + tests green)

- **B1 — registry + blend infra, BEHAVIOUR-PRESERVING.** `BiomeShapeSet` (all biomes → the current default
  graph), `sample_world`/`_grid` blend with the same-`Arc` parity guard, climate `Field` fns + grad-vs-CD
  tests. Every biome shares the default ⇒ output bit-identical ⇒ `worldgen_parity` green, NO version bump.
  Lands all the machinery invisibly.
- **B2 — distinct per-biome shape graphs.** Author a shape graph per climate biome (Plains gentle, Forest
  hilly, Desert dunes, Tundra rolling, Snowy peaks) in `assets/worldgen/biomes/<slug>.graph.ron`; registry
  loads them. Blends now fire ⇒ behaviour changes ⇒ bump `HEIGHT_GEN_VERSION`, re-pin `worldgen_parity`
  (+ border/plains/peak hazard points). The visible "biomes own shape".
- **B3 — editor multi-graph authoring.** A biome selector in the worldgen node panel to edit each biome's
  shape graph (+ the placement/classify view); RON save/load per biome. Reuse the existing snarl panel.

## Invariants / risks

- **Determinism + analytic gradient**: the blend is `Field` ops (autodiff) — grad-vs-CD test gates it; the
  same-`Arc` guard keeps B1 bit-exact. f64 + portable noise only (no transcendentals) — bit-portability holds.
- **Cross-tier agreement**: every clipmap tier evals the SAME `BiomeShapeSet` (like today's single graph) ⇒
  no LOD seam. The blend is a pure `f(wx,wz,seed)`.
- **Perf**: ≤2 graph evals per border point (climate is km-scale, so most chunks are single-biome → 1 eval).
  Gen-perf rig is the tripwire; cap per-biome graph node counts.
- **Vertical band**: `terrain_band_graph` must cover the MAX over all biome graphs (scan each).

## Critical files
- `src/sdf_render/worldgen/layers/height.rs` — `sample_world`/`_grid` blend; hold `BiomeShapeSet`; `HEIGHT_GEN_VERSION` (B2).
- `src/sdf_render/worldgen/biome.rs` — `temperature_grad`/`humidity_grad` (climate as `Field`).
- `src/sdf_render/worldgen/{mod.rs,manager.rs}` — `WorldGraph`→`BiomeShapeSet`, thread through; band-aware.
- `assets/worldgen/biomes/<slug>.graph.ron` — per-biome shape graphs (B2).
- `src/editor/worldgen_graph/` — biome selector (B3).
- `tests/worldgen_parity.rs` — green at B1 (no change), re-pin at B2.
