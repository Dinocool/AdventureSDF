# Tiled Bounded-RAM Voxelizer + Import Fidelity — Phase C Design Plan

Status: DESIGN (implementation-ready). Worktree: `voxel-rt`. This is the owning doc for **Phase C** of
`docs/VOXEL_PROGRAM.md` (C1 = tiled bake, C2 = `.vox` MATL emissive, C3 = import fidelity). It is referenced
by `VOXEL_PROGRAM.md` C1 as the previously-missing `docs/TILED_VOXELIZER_PLAN.md` and supersedes the bare
"#125 tiled voxelizer" stub. A future implementation agent should execute directly from this doc.

Scope boundaries (do not re-litigate):
- **Keep** the conservative triangle–box-SAT occupancy (`tri_box_overlap`/`plane_box_overlap` in
  `examples/voxelize_scene.rs`) verbatim. It is correct for first-hit RT — a voxel a triangle merely *touches*
  must be solid or the surface gets holes. Do **not** switch to a coverage-threshold/≥50% occupancy (that is
  the asset-gen char path; wrong for architectural first-hit geometry).
- **Keep** the always-on `solid_fill` semantics (exterior-reachable air stays air; enclosed cavities fill).
  C1 changes only *how* that classification is computed at scale, not the rule.
- The offline voxelizer (`examples/voxelize_scene.rs`) stays a standalone headless CPU dev-binary; the shipped
  runtime never links `gltf`/`image`/`dot_vox` (they remain dev/offline deps).

Three independently-shippable work items, in recommended order:

| Item | What | Blocker for | Size |
|---|---|---|---|
| **C2** | `.vox` MATL emissive reader (`registry_from_palette`) | GI on imported assets | Small (~1 file) |
| **C3** | CIELAB palette + area-averaged albedo (voxelizer) | Import fidelity / 0.05 m aliasing | Medium |
| **C1** | Tiled out-of-core voxelizer (the hard part) | Bistro-Exterior @0.05 m bake | Large (the real algorithm work) |

C2 and C3 are small/medium ports with no out-of-core algorithm; ship them first. C1 is the genuine
engineering item and the rest of this doc is mostly about it.

---

## Current state (what exists, verified in code)

`examples/voxelize_scene.rs` today:
- `voxelize(&mesh, voxel_size) -> Grid`: parallel conservative SAT rasterization over the mesh AABB into a
  `Grid` = `{ dims:[i32;3], solid: BitGrid (1 bit/cell), albedo: HashMap<usize,[u8;4]> (solid-only) }`.
- `solid_fill(&mut grid)`: a **single global** exterior 6-flood from the grid boundary through air
  (`exterior: BitGrid`, `total/8` bytes), then a second multi-source BFS that fills `air && !exterior`
  enclosed cells solid with the nearest surface albedo.
- `quantize(&grid)`: distinct-albedo median-cut to ≤255 (sRGB squared-distance `nearest_palette`).
- `build_dot_vox`: tiles solids into ≤256³ `.vox` models on a scene graph; writes via `dot_vox`.
- `MAX_VOXELS = 16e9` guard: each of the occupancy + exterior bitsets is `total/8` bytes, so peak is
  ~2 GB/bitset at the cap → **~4 GB during `solid_fill`** plus the albedo `HashMap` (solid-count-sized).

The wall: Bistro-Exterior @0.05 m is **>1.5 B dense cells in the AABB** (`SOTA_REFERENCE.md §6`). The two
in-RAM bitsets + a `VecDeque` flood frontier + the albedo map is multi-GB and grows with the AABB *volume*,
not the solid *surface*. The `solid_fill` global flood is the part that *requires* the whole AABB resident at
once — it is globally coupled (next section). `voxelize` and `quantize` are already sparse/streamable; only the
flood forces the volume into RAM.

`src/voxel/vox.rs::registry_from_palette` reads only `data.palette` (256 RGBA) and **drops `data.materials`**,
so imported emissive is lost (C2).

---

## C1 — Tiled bounded-RAM voxelizer

### C1.0 The hard sub-problem, stated precisely

Interior classification is **globally coupled**. A cell is *enclosed* (→ solid) iff a 6-connected air flood
starting at that cell **never reaches the grid boundary**; it is *exterior/open* (→ stays air) iff it does.
"Reaches the boundary" is a transitive, whole-grid property: a cavity 500 tiles deep is still open air if a
single 1-voxel crack connects it, tile by tile, out to the edge. So a naive per-tile flood **cannot** decide a
tile-boundary-touching air cell on its own — it doesn't know whether the air leaving its face eventually
escapes. This is exactly the classic out-of-core / blocked connected-components problem.

The solution is the standard two-level approach: **flood locally inside each tile to find which boundary-face
air cells are mutually connected within the tile, then stitch tiles together with a union-find over shared
faces, and finally propagate the single "exterior" label inward** from the global grid boundary across the
stitched graph. A cell is exterior iff its tile-local air component's union-find root is in the same set as
any global-boundary air cell. Everything else that is air is enclosed → fill solid.

This is provably identical to the global flood's answer (C1.7 acceptance), because air-connectivity is an
equivalence relation and union-find computes exactly its connected components; seeding "exterior" from the
global boundary and taking its closure under that relation is the global flood by definition.

### C1.1 Tiling, coordinate system, RAM budget

Tile the AABB into cubic **TxTxT-voxel tiles** (`T = TILE_EDGE`). With `dims = [dx,dy,dz]` (the existing
`Grid::dims`, padded as today), the tile grid is `tnx = dx.div_ceil(T)`, etc. A tile is addressed by
`(tx,ty,tz)`; its origin voxel is `(tx*T, ty*T, tz*T)`; its actual extent is clamped at the grid edge
(`min(T, dx - tx*T)` etc.) so boundary tiles are partial. Tile linear id `tx + ty*tnx + tz*tnx*tny`
(deterministic, X-fastest, same convention as `Grid::idx`).

**Choosing `T` vs RAM.** The peak RAM during C1 is dominated by *one resident tile's working set* plus the
*persistent boundary-face + union-find metadata for all tiles*. We size each so the sum is well under a stated
budget.

- One tile's occupancy = `T³` bits = `T³/8` bytes. For `T = 256`: `256³/8 = 2 MiB`. For `T = 128`: `256 KiB`.
- One tile's local-flood scratch (a `T³` component-label array, `u16` or `u32` per cell — components per tile
  are far fewer than `T³`, so `u16` suffices with a guard, see C1.3) = `T³·2` bytes. `T=256` → 32 MiB;
  `T=128` → 4 MiB. **This dominates the per-tile working set** and argues for `T=128`.
- Persistent across all tiles (held the whole run): the **boundary-face labels** (only the 6 faces of each
  tile, `6·T²` cells × a `u32` component id) and the **union-find** parent array over all (tile, local-comp)
  pairs. Faces: `6·T²·4` bytes/tile. Number of tiles for Bistro-Exterior @0.05 m: AABB longest axis ~ (scene
  ~ tens of m / 0.05) ≈ a few thousand voxels per axis → with `T=128`, ~ (≈24)³ ≈ 14 k tiles for a cube-ish
  AABB; Bistro is wide/flat so realistically a few thousand. Face metadata at `T=128`: `6·128²·4 = 384 KiB`
  **per tile of face storage** is too much × thousands of tiles. → **Do not keep full per-tile face arrays in
  RAM.** Keep only the *compressed* boundary description: per tile face, the run-length/label image is written
  to disk; in RAM we keep only the **per-tile component table** (a handful of components per tile) and the
  **union-find**. See C1.4 — the face *label images* live on disk; only the cross-tile *adjacency* (which
  component touches which neighbour component) is reduced in RAM.

**Stated budget & default.** Target **peak RAM < 4 GiB** for Bistro-Exterior @0.05 m (matches the order of the
current `MAX_VOXELS` guard but now independent of AABB volume). Default **`TILE_EDGE = 128`** (4 MiB label
scratch + 256 KiB occupancy per resident tile; with a small in-flight tile cache of, say, 8 tiles for the
stitch pass that is < 40 MiB of tile working set). The persistent union-find + component tables are
O(total_components) which is O(tiles × components/tile) ≈ thousands × tens = well under 100 MB. The albedo
store stays solid-count-sized (sparse, unchanged from today). So the budget is dominated by the **disk-backed
tile occupancy** (next), not RAM. `TILE_EDGE` is a `const` with a comment tying it to the budget; expose it as
an optional CLI override for tuning.

### C1.2 Disk-backed tile storage + the surface pass

The surface voxelization (`voxelize`) is already sparse and parallel and does **not** need the full volume in
RAM — it emits `(cell_index, albedo)` per solid surface cell. C1 keeps that pass intact but **routes its
output to disk-backed tiles** instead of one giant `BitGrid`:

1. **Tile store on disk.** A scratch directory (`--scratch <dir>`, default a temp dir via `std::env::temp_dir`
   honoring the project's `D:\tmp_test` redirect convention; see memory `test-temp-dir-split`). Each tile's
   occupancy is a file `tile_{id}.occ` of exactly `extent_x*extent_y*extent_z` bits (`ceil/8` bytes), created
   lazily (a tile with no surface cell never gets a file → it is "all air" by absence, mirroring `BrickMap`'s
   sparse-absent convention). The surface albedo stays in the sparse in-RAM `HashMap<usize,[u8;4]>` keyed by
   *global* cell index (it is solid-count-bounded — millions, fine; Bistro ~660 M solid @0.05 m is the worry
   → store albedo **per-tile on disk too**: `tile_{id}.alb` as a sorted `(local_index:u32, rgba:[u8;4])`
   run, written when the tile is finalized. See C1.6.).
2. **Surface scatter.** Run `voxelize`'s parallel SAT exactly as today, but the per-triangle output
   `(global_cell, albedo)` is bucketed by owning tile (cheap: `tile_of(cell)`), and each tile's contributions
   are appended to an in-RAM per-tile buffer; buffers flush to the tile's `.occ`/`.alb` files when they exceed
   a cap (bounded RAM) or at end of pass. First-writer-wins albedo is preserved per tile by processing
   triangles in order within a tile and skipping a cell already set (same rule as today, now tile-local). To
   keep determinism with rayon, collect `per_tri` lists as today, then do the **serial** first-writer merge
   into tile buffers in triangle order (the merge is already serial today).
3. After the surface pass, every tile file holds that tile's surface occupancy + albedo. Empty interior tiles
   (fully buried, no surface) have **no file** yet — they are resolved in the fill pass (C1.5): a tile with no
   surface and no exterior-reachable air is entirely enclosed → entirely solid.

Disk volume: occupancy is `total/8` bytes spread across tile files, but **only non-empty tiles exist**. A
fully-enclosed interior tile is *absent on disk until the fill pass marks it solid* (and even then it is
stored as a 1-bit "uniform solid" flag, not `T³` bits — C1.5). So disk ≈ surface-shell tiles × tile size +
metadata, far below the dense `total/8`. This is the disk analog of the engine's surface-only residency.

### C1.3 Per-tile local flood (within-tile connected components of air)

For each tile (processed one or a few at a time, streaming its `.occ` from disk into a `T³` bit buffer):

1. Build the tile's air mask = `!occupancy` over the tile's actual extent.
2. **Label air connected components within the tile** via 6-connected flood (BFS/DFS, or two-pass
   union-find labeling). Each air cell gets a tile-local component id `0..C_tile`. `C_tile` is typically a few
   (the tile's air is a few pockets); guard `C_tile <= u16::MAX` and widen the label type if exceeded (a
   degenerate checkerboard could blow this — assert with a clear message, practically never hit on real
   geometry).
3. **Record each component's 6-face footprint.** For each of the tile's 6 faces, for each face cell that is
   air, write `(face_cell_local_coord → local_component_id)` to that face's **disk label image**
   (`tile_{id}.face{f}`), and note in the in-RAM **component table** that component `c` *touches face f*. We do
   NOT keep the full face image in RAM — only the boolean "component c touches face f" and, for the stitch, the
   neighbour matching is done by **streaming the two adjacent face images** (C1.4). The component table per
   tile is `C_tile` rows of `{ touches_face: u8 bitmask, touches_global_boundary: bool }` — tiny.
4. **Global-boundary seed.** If the tile lies on the grid boundary (e.g. `tx==0`), any air component that
   touches that outward face is connected to *outside the grid* = exterior. Mark `touches_global_boundary` on
   those components (the grid edge is open air, exactly as today's `solid_fill` seeds boundary air cells).
5. Assign each tile-local component a **global node id** in the union-find = `component_base[tile] + c`
   (`component_base` is a prefix sum over `C_tile`, assigned deterministically in tile-id order). Append the
   component to the union-find (`parent[node] = node`).

The label scratch (`T³`) and the tile occupancy (`T³/8`) are the only large per-tile buffers; with `T=128`
that is ~4.25 MiB resident per tile, and the pass touches each tile once. Components, face-touch flags, and
the union-find grow O(total components) ≪ O(volume).

### C1.4 Union-find across tile faces (the crux)

Two tiles share an internal face: tile `A` at `(tx,ty,tz)` and tile `B` at `(tx+1,ty,tz)` share A's +X face
with B's −X face (and analogously for ±Y, ±Z). Air is 6-connected, so an air cell on A's +X face is connected
to the air cell directly across on B's −X face **iff both are air** (same `(v,w)` in-face coordinate). To
stitch:

1. **Iterate every internal face exactly once** (e.g. for each tile, its +X/+Y/+Z neighbours — the negative
   side is covered by the neighbour's positive iteration; deterministic order = tile-id order).
2. Stream the two adjacent face label images from disk: A's +X face image (`(v,w) → comp_a or AIR-absent`) and
   B's −X face image (`(v,w) → comp_b`). For each in-face cell `(v,w)` where **both** are air:
   `union(component_base[A] + comp_a, component_base[B] + comp_b)`.
3. Union-find with **path compression + union by rank**, so the whole stitch is near-linear in the number of
   matched face cells (which is bounded by the total internal face area = O(tiles · T²), streamed, not
   resident).

The face images are read in pairs and discarded; only the union-find parent array (O(total components),
resident) accumulates the merges. After all internal faces are processed, the union-find encodes the global
6-connected air components across the whole grid — **identical** to what the monolithic flood would compute,
because every adjacency the monolithic flood would traverse across a tile boundary is exactly one of these
matched face-cell pairs, and every within-tile adjacency was captured by the local flood.

**Propagate the exterior label.** After all unions: a component's root is **exterior** iff *any* component in
its set had `touches_global_boundary`. Compute this by a single pass: for each component with
`touches_global_boundary`, mark `exterior_root[find(node)] = true`. (Optionally union all boundary components
into one synthetic "OUTSIDE" node first, then exterior == `find(node) == find(OUTSIDE)` — cleaner and
O(components).)

Determinism: tile order, component_base prefix-sums, face iteration order, and union-by-rank tie-breaks are
all fixed functions of tile ids, so the partition (and thus the fill) is byte-reproducible across runs.

### C1.5 The fill pass (resolve every cell to solid/air)

Now classify and finalize each tile, streaming one (or a few) at a time:

For each tile:
- If the tile **has a face/occupancy file**: load `.occ`, and for each **air** cell, look up its tile-local
  component (recompute the local labeling on the fly — cheaper than persisting the full `T³` label image to
  disk; the local flood is fast and the occupancy is already loaded) and test `exterior_root[find(component_base[tile]+c)]`:
  - exterior → leave air.
  - not exterior → **enclosed**: set the cell solid. Its albedo is the nearest surface voxel's colour. For the
    interior-colour BFS (today a global multi-source BFS from all surface cells), see C1.6 — it becomes a
    bounded per-tile + halo propagation.
  Write the finalized tile occupancy + albedo back to disk (it is now the full solid mask for that tile).
- If the tile **has no file** (no surface cell at all): it is either entirely exterior air or entirely
  enclosed solid — a fully-buried interior tile. Decide by **one probe**: the tile contributed at least one
  component to the union-find only if it had air on a face touching a neighbour... but a file-less tile never
  ran the local flood. Handle file-less tiles explicitly: a file-less tile's air is one single component (the
  whole tile is air). During C1.3 we **still create a component for every tile** (including file-less ones: a
  file-less tile = one all-air component touching all 6 faces). Then the union-find naturally connects it to
  neighbours, and the boundary/exterior propagation classifies it: exterior → emit nothing (stays absent/air);
  enclosed → emit a **uniform-solid tile** (a 1-flag marker + the inherited interior albedo, NOT a `T³` bit
  array) so a fully-buried interior region costs O(1) on disk. This is the disk analog of `Brick::uniform`.
  *(So C1.3 must enumerate a component for file-less tiles too; cheap — one component, all-faces-touch.)*

The output of the fill pass is the complete solid mask, tile by tile on disk, plus the solid-only albedo.

### C1.6 Interior albedo at scale (replacing the global multi-source BFS)

Today `solid_fill` colours enclosed cells with the nearest *surface* voxel's albedo via one global 6-connected
multi-source BFS from all surface cells (deterministic, sorted seeds). At billions of cells that BFS is itself
unbounded-RAM. Out-of-core replacement, two options — pick **(A)** for simplicity/robustness, escalate to (B)
only if a bake shows visible interior-colour seams (cosmetic, interiors are revealed only on destruction):

- **(A) Per-tile nearest-surface fill with a 1-voxel halo exchange (recommended).** Within a tile, run the
  multi-source BFS seeded from that tile's surface cells **plus** the surface/colour values imported from the
  6 neighbour tiles' shared faces (a 1-voxel halo, streamed from neighbour `.alb`/face images). For tiles with
  no interior surface within reach, do a second relaxation sweep in tile-id order then reverse order
  (Gauss-Seidel-style) so colour propagates across tile boundaries over ≤2 passes. This bounds RAM to a tile +
  its 6 halos and is deterministic. Exact match to the global BFS only up to ties near tile seams; interiors
  are cosmetic so this is acceptable.
- **(B) Exact global wavefront on disk.** A disk-backed multi-source BFS (bucketed by distance, tiles paged in
  on demand). Exact but more I/O and complexity. Defer unless (A) seams are visible.

Because interiors are revealed only when cut and a future strata/material system reassigns them anyway
(`solid_fill` doc comment), (A) is the plan of record; note (B) as the exact fallback.

### C1.7 Assembly to `.vxo`/`.vox` (output)

The existing `build_dot_vox` tiles solids into ≤256³ `.vox` models reading from the sparse `indices` map. For
the tiled path, feed it from the **disk tile store** instead of the in-RAM `Grid`:
- Stream each finalized tile's solids, quantize (C3 CIELAB; the palette is built from a *sampled* subset of
  albedos to bound the quantizer's input — see C3.1), and emit voxels into the `.vox`/`.vxo` writer's tiles.
  The `.vox` 256³ model split and the `.vxo` `BRIK` chunk both consume a stream of `(world_voxel, block_id)`,
  so neither needs the whole grid resident.
- **Preferred output is `.vxo`** (`VOXEL_INSTANCING_PLAN §1.5`, Phase B `B1`): its `BRIK` chunk stores the
  sparse `BrickMap` directly (no re-bricking on load) and its `MATL` chunk carries emissive — so the tiled
  bake should target `.vxo` once Phase B lands. Until then it can still emit `.vox` (≤255 palette) for parity
  with the current corpus. Keep both writers behind one `Grid`/tile-stream reader trait so the algorithm body
  is format-agnostic.

### C1.8 Composition with the existing passes (summary of the pipeline)

```
load_mesh ──► surface SAT (parallel, unchanged) ──► scatter to DISK tiles (.occ/.alb)   [bounded RAM]
                                                          │
                                  per-tile LOCAL air flood (+ file-less = 1 all-air comp)
                                                          │  component table + boundary seeds
                                  UNION-FIND across shared tile faces (stream face images)
                                                          │  exterior closure from global boundary
                                  FILL pass: enclosed→solid (uniform-solid marker for buried tiles)
                                                          │  per-tile nearest-surface albedo (+halo)
                                  STREAM tiles ──► CIELAB quantize ──► .vxo (BRIK+MATL) / .vox
```

Only the surface SAT and quantize are reused verbatim; the flood is replaced by the tiled flood + union-find;
storage moves to disk-backed tiles.

### C1.9 Acceptance gates (C1)

1. **Correctness vs the oracle.** A headless test bakes a synthetic scene small enough for the **monolithic
   `solid_fill`** to run, then bakes the **same scene through the tiled path with `TILE_EDGE` set small**
   (e.g. 8 or 16, so the scene spans many tiles) and asserts the **enclosed/air classification is identical
   cell-for-cell**. Cover the hard cases explicitly:
   - a sealed box (enclosed) vs a box with a 1-voxel crack to the boundary (open) — the crack must cross a tile
     boundary (place it so the leak path traverses ≥2 tiles), proving the union-find stitch, not just local flood.
   - a cavity reachable only via a long thin S-shaped tunnel spanning many tiles (transitive exterior).
   - a fully-buried interior tile (file-less) that is enclosed → uniform-solid; and one that is exterior → air.
   - two adjacent independent cavities (distinct union-find sets, both enclosed).
2. **Determinism.** Bake the same scene twice (and at two different `TILE_EDGE`s) → identical solid mask
   (byte-identical `.occ` aggregate / identical solid-cell set). Tile order, prefix sums, union-by-rank, and
   face iteration are fixed functions of tile ids.
3. **Bounded RAM (the headline).** Bake **Bistro-Exterior @0.05 m** with a resident-set / peak-RSS probe and
   assert **peak RAM < 4 GiB** (stated budget) — and that it *completes* (today it cannot without the
   ~tens-of-GB dense flood). Gate in the perf harness (memory `mesh-bake-perf-rig` style: a headless bench that
   bakes and reports peak RSS + wall-clock + scratch-disk high-water). Because the Bistro asset is gitignored,
   the test SKIPS gracefully when absent (mirroring `bistro_loads_and_bakes_with_decoded_textures`), but the
   bounded-RAM *unit* test uses a synthetic large-AABB-sparse scene that always runs.
4. **No regression on small scenes.** Cornell/Sponza/the fallback room bake to the **same** `.vox` solids as
   the monolithic path (the existing `fallback_room_bakes_all_six_faces` and
   `solid_fill_closes_enclosed_but_keeps_open_air` tests must still pass, routed through the tiled path with a
   large `TILE_EDGE` that yields a single tile = the degenerate case == today).

### C1.10 Risks / open questions (C1)

- **Disk I/O dominates wall-clock** at 0.05 m. Mitigate: write tiles compressed (RLE/zstd the `.occ` — runs of
  solid/air are long; reuse the `.vxo` BRIK compressor), keep a small LRU of decompressed tiles for the stitch
  + fill passes, and process tiles in spatially-coherent (Z-order/Morton) order so neighbour faces are likely
  cached. Measure with the perf harness; tune `TILE_EDGE` (bigger = fewer faces/seams + more sequential I/O,
  but more RAM per tile).
- **`u16` component-id overflow** on pathological geometry (per-tile air pockets > 65535). Guard + widen; never
  silently wrap. Practically unreachable at `T=128` on architectural scenes.
- **File-less-tile component enumeration** must not be forgotten — a buried tile with no surface still needs a
  component node so the union-find can classify it (C1.5). The oracle test's "fully-buried enclosed tile" case
  guards this.
- **Scratch-dir lifetime / cleanup**: create under a run-unique subdir, delete on success, leave on failure for
  debugging (log the path). Respect `D:\tmp_test` redirect (memory `test-temp-dir-split`).

---

## C2 — `.vox` MATL emissive reader (pull forward, small)

### C2.1 Goal & where it goes

`src/voxel/vox.rs::registry_from_palette` currently builds the registry from only the 256-RGBA palette and
emits `emissive: [0,0,0]` for every block (`palette.rs::from_vox_palette` hardcodes it). MagicaVoxel stores
emissive in the `MATL` chunk, which `dot_vox` already parses into `data.materials: Vec<Material>` — we just
never read it. The GI/NEE stack already consumes `BlockRegistry::emissive` (the air-exposed-emissive light
gather; `has_emitters`), so wiring MATL → registry emissive lights imported lamps with zero GI changes.

### C2.2 The `dot_vox` Material API (verified, dot_vox 5.2)

`dot_vox::Material { pub id: u32 /* palette index */, pub properties: Dict }` with accessors returning
`Option<f32>` (parsed from the string-keyed `properties`):
- `material_type() -> Option<&str>` — `"_diffuse"`, `"_emit"`, `"_glass"`, `"_metal"`, etc.
- `emission() -> Option<f32>` — the `_emit` field, **0..1 emission strength**.
- `radiant_flux() -> Option<f32>` — the `_flux` field, an **integer power exponent** (MagicaVoxel multiplies
  emission by `2^flux`; flux is typically 0..4).
- `roughness()`, `metalness()`, `specular()`, `refractive_index()` — for the reserved `.vxo` MATL fields later.
- `low_dynamic_range_scale() -> Option<f32>` — `_ldr`, a visual dampening hack; **ignore for GI** (it darkens
  the *display*, not the physical emission — we want physical radiance for the bounce).

`Material.id` is the **palette index** (1-based in MagicaVoxel's UI but the struct id corresponds to the
palette slot; `dot_vox`'s `DEFAULT_MATERIALS` are indexed `0..256`). It aligns with `data.palette[id]` — the
same index the loader already maps to `BlockId(id+1)` via `from_vox_palette` (palette entry `i` → `BlockId(i+1)`).

### C2.3 Emissive radiance formula

For each material with `emission() == Some(e)` and `e > 0` (or `material_type() == Some("_emit")`):
```
flux        = radiant_flux().unwrap_or(0.0)          // _flux exponent
strength    = e * 2f32.powf(flux)                    // MagicaVoxel emission scale (matches its renderer)
albedo_lin  = srgb_u8_to_linear(palette[id])         // the voxel's own colour (sRGB→linear), RGB only
emissive    = [albedo_lin.rgb * strength * EMISSIVE_SCALE]
```
where `EMISSIVE_SCALE` is a single **runtime/uniform-style constant** (a `const` in the loader, documented;
per the project's "knobs as uniforms" rule it should ultimately be a tunable, but the loader is offline-ish so
a documented const with a comment is acceptable here — keep it ONE named constant, not a magic number) tying
MagicaVoxel's 0..1·2^flux scale to the engine's lumen-scale emissive (the Cornell light panel is `[1,1,1]`
linear; calibrate `EMISSIVE_SCALE` so a default MagicaVoxel emitter lands in a comparable range — note
`solari-gi` memory: emissive is lumen-scale). Emissive is **the voxel colour times strength** (an emitter
glows in its own hue), matching how MagicaVoxel renders `_emit` materials.

### C2.4 Implementation

Add an overload/extension to the registry build that takes the materials:
```rust
// palette.rs — extend the .vox builder to accept the MATL table.
pub fn from_vox_palette_with_materials(colors: &[[u8;4]], materials: &[(u32 /*id*/, f32 /*emissive_strength*/)]) -> Self
// or: from_dot_vox_materials(colors, &data.materials) reading emission()/radiant_flux() directly.
```
Cleanest: keep `palette.rs` free of `dot_vox` (it must not depend on the offline crate). So **read MATL in
`vox.rs`** (which already depends on `dot_vox`) and pass a plain `&[(u16 block_id, [f32;3] emissive)]` into a
new `BlockRegistry::set_emissive`-based step, OR add `from_vox_palette` a second param of pre-computed
emissives. Recommended:
1. In `vox.rs::registry_from_palette`, after building the base registry from `data.palette`, iterate
   `data.materials`; for each emissive material compute `emissive` (C2.3) and call
   `registry.set_emissive(BlockId(mat.id as u16 + 1), emissive)` — `set_emissive` already exists and is the
   SSOT mutator (`palette.rs`), no-op for AIR/out-of-range, so a malformed material id can't panic.
2. Map `mat.id` (palette index) → `BlockId(mat.id + 1)` — the SAME `+1` the palette→block mapping uses, so the
   emissive lands on the right block. (Guard: `mat.id < 256`; ignore others.)

`set_emissive` taking `[f32;3]` and the loop in `vox.rs` keeps `dot_vox` out of `palette.rs` — clean layering.

### C2.5 `.vxo` MATL forward-compat

The `.vxo` `MATL` chunk (`VOXEL_INSTANCING_PLAN §1.5`) carries emissive (linear RGB × strength) per `u16`
`BlockId` with reserved roughness/metallic/ior fields. C2's offline reader is the **source** that populates
that chunk at import time: `.vox` MATL → `BlockDef.emissive` (+ roughness/metal from `material.roughness()`/
`metalness()` into the reserved fields when `.vxo` lands). So C2 is the first consumer-side step of the `.vxo`
material table; ship it now reading into the live `BlockRegistry`, and the `.vxo` writer later serializes the
same `BlockDef` fields.

### C2.6 Acceptance (C2)

- A unit test builds an in-memory `DotVoxData` with a palette + a `Material { id, properties: {"_type":"_emit",
  "_emit":"0.8","_flux":"2"} }` and asserts the loaded `BlockRegistry.emissive(BlockId(id+1))` is non-zero,
  proportional to `0.8 * 2^2`, and tinted by the palette colour; non-emissive blocks stay `[0,0,0]`;
  `has_emitters()` flips true. (Mirror the existing `round_trip_two_colour_cube` style.)
- A material with no `_emit` (or `_emit == 0`) leaves the block non-emissive.
- Out-of-range / malformed `mat.id` does not panic (relies on `set_emissive`'s guard).

---

## C3 — Import fidelity (CIELAB palette + area-averaged albedo)

Two independent quality ports from `D:\Projects\asset gen` (`palette.py`, `voxelize.py`), both in
`examples/voxelize_scene.rs`. **Skip the char-art stages** (`_cleanup` despeckle/smooth, `_symmetrize_x`,
`_hair_mask`) — they damage architectural scenes (would erode thin railings, fuse columns, mirror Sponza).

### C3.1 CIELAB-space palette clustering (replace median-cut)

Today `quantize` uses `median_cut` (sRGB widest-channel split) + `nearest_palette` (sRGB squared distance).
Both operate in sRGB, which is perceptually non-uniform → muddy/biased clustering. Port the asset-gen CIELAB
path (`palette.py::build_palette` + `_rgb_to_lab`):

- **Convert to CIELAB** (D65) before clustering. Port `_rgb_to_lab` exactly: sRGB→linear (already have
  `srgb_channel_to_linear` semantics), linear→XYZ via the 3×3 matrix, XYZ→Lab with the `eps=216/24389`,
  `kappa=24389/27` piecewise `f`. Implement as a small `rgb_to_lab([u8;3]) -> [f32;3]` in the voxelizer.
- **k-means in Lab** to ≤`max_colors` (255 for `.vox`; the `u16` cap-lift rides the `.vxo` MATL chunk later —
  C3 stays within 255 for the current corpus). Use **k-means++ seeding with a FIXED seed (0)** for
  determinism (asset-gen uses `kmeans2(minit="++", seed=0)`). No scipy in Rust → implement a compact
  deterministic k-means: k-means++ init (fixed RNG, e.g. a seeded `SmallRng` or a simple LCG so the bake is
  reproducible) + Lloyd iterations to convergence or a fixed iteration cap. Cluster the **distinct** albedos
  weighted by voxel count (as median-cut does today via `counts`), or a capped sample (asset-gen samples
  ≤50 k for speed) — at billions of voxels, **cluster the distinct-colour set** (bounded by texture content,
  typically thousands–tens of thousands of distinct sRGB triples), weighted by count. This keeps the input
  bounded regardless of solid count (it is the same `counts: HashMap<[u8;4],u32>` `quantize` already builds).
- **Representative colour = count-weighted mean RGB of each cluster** (asset-gen: "mean RGB of each cluster,
  truer than the LAB centroid"), clamped to `[0,255]` sRGB — drop empty clusters (k-means can emit empty).
- **Nearest-palette assignment in Lab** (replace `nearest_palette`'s sRGB distance with Lab distance), cached
  per distinct albedo as today.

Keep the existing **count-weighting** and **sorted-deterministic input** (`pixels.sort_unstable()` before
clustering) so the palette is reproducible. Fallback: if the distinct-colour set is ≤255, emit it exactly
(lossless — asset-gen's `quantize` short-circuits when `len(uniq) <= max_colors`); only run k-means when it
exceeds the cap.

### C3.2 Area-averaged albedo (replace nearest-texel point sample)

Today `sample_albedo` takes ONE texel: the texture sampled at the barycentric UV of the triangle point
nearest the voxel centre (`closest_point_barycentric` → `Texture::sample` nearest). At 0.05 m a voxel still
covers many texels of a 2K texture → point-sampling aliases (a single texel decides the whole voxel's colour).
Port the asset-gen supersampling idea (box-filter the voxel's texel footprint), adapted to our
triangle/SAT pipeline:

- For each `(triangle, voxel)` pair that `tri_box_overlap` accepts, instead of one sample, **supersample the
  triangle's surface within the voxel's box** and average the texture lookups:
  - Take an `S×S×S` (or `S×S` over the dominant triangle plane) grid of sample points across the voxel's
    extent (default `SUPERSAMPLE = 3`, matching asset-gen's `supersample=3`), project each onto the triangle
    via `closest_point_barycentric`, sample the texture at each barycentric UV, and **average in linear space**
    (convert sRGB texel → linear, average, store back; or average sRGB directly for parity with the current
    end-to-end-sRGB pipeline — the voxelizer keeps everything sRGB `u8` today per its colour-space note, so
    average in sRGB to avoid changing the one-conversion-in-the-loader invariant, with a comment). Weight by
    whether the sample's nearest-triangle-point actually lies near the voxel (so off-triangle samples don't
    pull the average) — simplest: sample points = a jittered/regular grid over the *triangle area clipped to
    the voxel box*, all of which are on the triangle by construction.
  - Cleaner formulation that matches our geometry: sample the **triangle's area inside the voxel box**. Take
    the voxel-box∩triangle region (the triangle is conservative-rasterized, so it overlaps the box); pick `K`
    points across that region (barycentric grid on the triangle clipped to the box), look up each UV, average.
    This is a true area average of the texture over the voxel's surface footprint.
- Upgrade `Texture::sample` to support **bilinear** (4-tap) filtering as the per-tap filter (current is
  nearest), so each of the `K` taps is itself smooth — compounds with the supersample to kill texel aliasing.
- First-writer-wins per cell is unchanged (the merge still keeps the first triangle's albedo per cell); the
  averaging is *within* a single triangle's contribution to a voxel.

Keep it **off the conservative-occupancy path**: occupancy is still pure `tri_box_overlap` (a touched voxel is
solid). Only the *colour* of a solid voxel is now area-averaged. Performance: `K`-tap × parallel SAT — fold
into the existing rayon `par_iter`; `K=9` (3×3) is a modest constant. Expose `SUPERSAMPLE` as a CLI/const knob.

### C3.3 What to SKIP (and why)

- `_cleanup` (despeckle / majority-smooth / fill) — erodes thin architectural detail (railings, grilles),
  fuses nearby surfaces. Wrong for scenes; it exists to clean character extremities.
- `_symmetrize_x` / `_hair_mask` — character-specific; would mirror/destroy asymmetric architecture.
- `_view_projected_colors` orthographic ray casting — an alternative to surface-sampling for *characters*; our
  conservative SAT + barycentric texture sample is already correct for scenes. Do not adopt.
- The `_coverage_occupancy` ≥50% downsample — this is the coverage-threshold occupancy the scope explicitly
  forbids (it would drop sub-voxel-thin geometry that conservative SAT correctly keeps).

### C3.4 Acceptance (C3)

- **CIELAB palette:** a unit test with a known gradient/textured patch asserts the CIELAB palette places more
  entries in perceptually-distinct regions than sRGB median-cut on the same input (e.g. a green-vs-similar-green
  pair stays distinct), and is **deterministic** (two runs → identical palette). The ≤255-distinct lossless
  short-circuit returns the exact colours.
- **Area-averaged albedo:** a synthetic high-frequency checkerboard texture on a quad, voxelized at a pitch
  coarser than the checker → each voxel's albedo is the **average grey** (≈ mid), not an aliased pure
  black/white per voxel (which the current nearest-texel sample produces). Assert the variance of per-voxel
  albedo drops vs the point-sample baseline.
- **No regression:** the existing `fallback_room_bakes_all_six_faces`, `obj_loader_voxelizes_a_synthetic_quad`,
  and Bistro decode tests still pass (untextured flat-`base` faces are unaffected — `sample_albedo` falls back
  to `base` when there's no texture, so area-averaging only changes textured voxels).

---

## Sequencing & dependencies (Phase C)

1. **C2 first** (smallest, unblocks GI on imports; ~1 file in `vox.rs` + a `set_emissive` loop + 1 test).
   Independent of C1/C3.
2. **C3 next** (medium; CIELAB + supersample in `voxelize_scene.rs`). Independent of C1; improves every bake
   immediately, including the current 0.2 m corpus.
3. **C1 last** (the large item). C1 *uses* C3's quantizer in its assembly stage (C1.7) but the tiled flood +
   union-find is independent of both — implement the disk-tile pipeline and the union-find, then wire C3's
   CIELAB quantize into the streaming assembly. C1 should target `.vxo` output once Phase B (`B1` `.vxo`
   writer) lands; until then emit `.vox` for corpus parity.

Each item is gated by its own acceptance tests (C1.9 / C2.6 / C3.4) and folds a perf measurement into the QA
gate (memory `feedback-benchmark-deliveries`): C1 reports peak-RSS + scratch high-water + wall-clock for the
Bistro bake; C3 reports the per-voxel-albedo variance drop and the k-means wall-clock; C2 is correctness-only.

---

## Appendix — invariants this plan must not break

- **Conservative SAT occupancy** (`tri_box_overlap`) is the occupancy SSOT — first-hit RT correctness. C1/C3
  never touch it (C3 changes only colour; C1 changes only storage + flood).
- **`solid_fill` rule**: exterior-reachable air stays air; enclosed fills. C1's tiled flood computes the
  *identical* partition (C1.7 acceptance oracle).
- **One colour-space conversion** (sRGB `u8` end-to-end in the offline tool; the runtime loader does the single
  sRGB→linear). C3 averages in sRGB to preserve this; C2's emissive uses the loader's existing
  `srgb_u8_to_linear`.
- **Determinism** (reproducible `.vox`/`.vxo` bytes for diffing/CI): every new stage (tile order, union-find,
  k-means seed, supersample grid) is a fixed function of inputs.
- **Runtime never links offline deps**: MATL read is in `vox.rs` (already `dot_vox`-dependent for `.vox`
  loading); CIELAB/supersample are in the offline `examples/voxelize_scene.rs`. `palette.rs` stays free of
  `dot_vox`/`gltf`/`image`.
```
