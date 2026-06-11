# Terrain Materials — volumetric biome strata, destruction-aware

## Vision (decided)

Terrain material is **volumetric**, not a 2D surface property: it's a function of **depth below the
original surface**, so **digging/deformation exposes layers** (Minecraft-style: grass → dirt → stone →
bedrock). The column is **biome-dependent** (desert: sand → sandstone → stone; tundra: snow → permafrost →
stone; …), and the topmost (undug) layer also gets **surface treatment** by slope/height/biome (snow on
peaks, sand low, rock on cliffs). **Flat colors** for the initial demo (no textures). Evaluated
**per-fragment**.

## Why per-fragment (the key architectural choice)

Destruction exposes **vertical** strata on pit walls (grass at the rim → dirt → stone → bedrock going down).
- A 2D top-down **splat map cannot represent vertical strata** at all.
- **Per-vertex** material can't resolve layers thinner than a voxel (~2.8 m at LOD0).
- **Per-fragment depth lookup** handles any layer thickness, any dug wall, at any LOD — and "just works"
  through destruction.

```
depth = original_surface_height(world.xz) − world.y          // pristine surface, NOT the carved geometry
biome = biome_at(world.xz)                                    // climate classification
material = biome.column.lookup(depth)                         // surface layer … bedrock
if depth ≈ 0 (top, undug): material = surface_treatment(biome, slope, height)   // snow/sand/rock/grass
```

`depth` uses the **original (pristine) surface height** — carving changes the GEOMETRY but not this
reference, so exposed faces read their true stratum automatically. (Building terrain *up* / overhangs is out
of scope for v1; subtract-only digging.)

## Architecture

**Per-chunk baked terrain data** (extend the existing detail-normal per-chunk bake — already covers the
chunk's world-XZ footprint with a planar map):
- `Rg16Float` detail-normal slope (exists).
- **+ original surface height** `h(x,z)` (R32Float, or pack a channel) — the depth reference.
- **+ biome id / blend weights** (a small map) — Phase 2.

**Terrain surface shader** (extend `terrain_detail.wgsl` → `terrain_surface.wgsl`): per-fragment —
1. sample surface height (planar UV) → `depth = surf_h − world.y`;
2. sample biome → strata column;
3. look up the material by depth (blend across layer boundaries + biome boundaries);
4. surface treatment for the top layer (slope/height/biome → grass/snow/sand/rock);
5. apply the detail normal (already wired);
6. PBR with the material's flat `base_color` (textures later).

**Strata data** — RON-defined, data-driven: per biome a column `{ surface_material, [ {material, thickness} … ],
bedrock_material }`; each material a flat linear `base_color` (+ roughness). Compiled to a GPU table (storage
buffer) the shader indexes by `(biome, depth)`.

**Biome classification** (Phase 2) — climate axes (temperature, humidity — low-freq fields, the node-graph
direction in [soft-coalescing-dolphin] / WORLD_GEN_PLAN §9) → biome id + transition weights, baked per-chunk.

**Destruction** (in scope) — runtime carve = a CSG-subtract edit on the terrain (the edit system exists;
mesh-bake already re-bakes "mixed" terrain+edit chunks). Needed changes:
- **Terrain surface shader must cover CARVED terrain chunks**, not just pristine terrain-only ones (today a
  carved chunk goes "mixed" → shared material). Route terrain-origin surfaces on mixed chunks through the
  strata shader.
- A basic **dig interaction** (cursor/brush → add a sphere subtract edit → re-bake the touched chunks) so
  layers can be seen exposed live.
- **Bedrock floor**: a world-Y floor (or max depth) below which it's all bedrock.

## Phasing (BIOME-FIRST — decided)

**Stage 1 — climate + biome classification + biome/strata RON (CPU/data, headless-testable).**
- **Climate fields**: `temperature(wx,wz,seed)` + `humidity(wx,wz,seed)` — low-freq bit-portable value-noise
  (reuse `noise.rs`), deterministic, normalized. Independent of `sample_world` height ⇒ parity unaffected.
- **Classifier**: `classify(temp, humidity) -> (primary, secondary, blend)` — a Whittaker-style table over
  the demo biomes, with a blend weight for smooth transitions.
- **RON**: `assets/worldgen/biomes.ron` — the demo biomes below, each with a climate cell, a flat-colour
  surface material, and a strata column. Loaded as an asset → compiled GPU table.
- Tests: deterministic climate, classifier covers each demo biome, RON round-trip, strata lookup.

**Stage 2 — per-chunk bake channels.** Extend the per-chunk terrain bake with **original surface height**
(R32Float, the depth reference) + **biome id/weights** (small map) so the fragment can sample both by the
existing planar UV.

**Stage 3 — terrain surface shader.** `terrain_surface.wgsl`: per-fragment `depth = surf_h − world.y` →
biome strata column → flat-colour material (+ layer/biome boundary blend); surface treatment for the top
layer; + the detail normal; PBR. The biome/strata GPU table.

**Stage 4 — destruction.** Route carved (mixed terrain) chunks through the surface shader; a dig brush
(sphere subtract → re-bake). Bedrock floor.

**Stage 5 — polish.** PBR textures per material, resources/ores in strata, brush/transition tuning.

## Demo biomes (Stage 1 RON, flat colours; tune later)

Climate axes normalized `[0,1]`: temperature T, humidity H. Strata are `surface → … → bedrock`.

| Biome | Climate cell | Surface | Strata column (top→down) |
|---|---|---|---|
| **Plains/Grassland** | T mid, H mid | grass (green) | grass → dirt (brown) → stone (grey) → bedrock (near-black) |
| **Forest** | T mid, H high | grass (darker) | grass → dirt → stone → bedrock |
| **Desert** | T high, H low | sand (tan) | sand → sandstone (pale orange) → stone → bedrock |
| **Tundra** | T low, H low–mid | tundra (pale khaki) | tundra → permafrost (blue-grey) → stone → bedrock |
| **Snowy peaks** | T low (or high-altitude override) | snow (white) | snow → rock (grey) → stone → bedrock |

Surface treatment override (any biome): high+cold → snow cap; very steep → exposed rock; near sea level →
sand. Layer thickness defaults: surface ~1 m, sub-surface (dirt/sandstone/permafrost) ~4 m, stone to the
bedrock floor (fixed world-Y).

## Biome preview (editor 2D/3D previews)

The worldgen node-graph previews must also visualize the **biome material classification** so you can author
against it (see where deserts/tundra/forests land). A top-down 2D preview colored by surface material **is**
a climate/biome map.

- **Per-material PREVIEW COLOR = the single source of truth for "what flat color represents this material."**
  Flat-color materials (now): `preview_color = base_color`. Textured materials (Stage 5): `preview_color =
  average color of the diffuse texture` (computed once on load). The SAME `preview_color` is used by the
  preview AND any flat-color fallback, so they always agree (SSOT — don't hardcode preview colors separately).
- **Preview = the SURFACE material** (depth 0): climate → biome (+ blend) → surface treatment → its
  `preview_color`. Strata are only seen when dug, so the preview shows the undug surface map.
- **Implementation**: CPU-evaluate `surface_biome` → surface `preview_color` over the preview region (small,
  bounded) into a color buffer, sample it in the existing GPU preview shader — reuses Stage 1's CPU
  classifier directly (no WGSL port of the Whittaker logic). A preview MODE toggle (height/field ↔ biome).
- **Timing**: after Stage 1 (classifier exists); share the climate→biome→surface-color path with the real
  Stage-3 surface shader so preview and in-world terrain match.

### 3D preview SLICE / cutaway (reveals the strata)

A clip plane in the 3D preview, positioned by a **percentage along an axis** (e.g. slice across X at 20%),
that hides the near half and renders the exposed **cross-section** — so you see the volumetric strata
(grass → dirt → stone → bedrock) as colored bands in the cut face, plus the biome surface on top. The best
way to inspect the columns without digging.

- **Controls**: axis (X / Z / optionally Y) + a 0–100% position slider; toggle on/off.
- **Rendering**: the 3D preview raymarches the graph's height field. At the slice plane, where the sampled
  point is BELOW the surface (`depth = surf_h − y > 0`), shade by `strata_material(biome, depth)` →
  `preview_color`; above/beyond the plane render the normal surface. The cut face = the depth→strata column
  evaluated per pixel — exactly the Stage-3 strata logic, just on a vertical plane instead of dug geometry.
- **SSOT**: same climate→biome→strata→`preview_color` path as the in-world shader + the biome map preview —
  one definition, three consumers (in-world terrain, biome map, slice).

## Standalone-then-integrate note
The Stage-1 climate/biome classifier is a pragmatic **standalone** module (fast path to visible biomes +
strata); it does NOT modify the height node-graph or `sample_world` (so parity holds). It can later feed /
be fed by the biome node-graph engine ([soft-coalescing-dolphin], [[worldgen-biome-node-graph]]) rather than
diverging from it.

## Reuses / touches
- Reuses: the per-chunk terrain bake + planar-UV sampling (detail-normal infra), the terrain `ExtendedMaterial`
  path, the CSG edit + mesh re-bake pipeline, the analytic slope, the height clipmap.
- New: surface-height (+ biome) channels in the per-chunk bake; `terrain_surface.wgsl` strata logic; the
  strata/biome RON + GPU table; carved-chunk material routing; the dig interaction.

## Open / to confirm
- Phasing: global-strata-first (recommended — fast, de-risks destruction) vs biome from the start.
- Layer thicknesses / bedrock floor depth (gameplay numbers).
