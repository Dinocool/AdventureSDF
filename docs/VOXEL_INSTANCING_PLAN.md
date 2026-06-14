# Voxel Instancing, Scene-Graph & Nested Sub-Grids — Design Plan

Status: DESIGN (no engine code changed by this doc). Worktree: `voxel-rt`. Target: Bevy 0.19 + forked
wgpu-trunk, RTX 4090 / Vulkan. This plan extends the existing HW-ray-traced brickmap path
(`src/voxel/{brickmap,gpu,raytrace,streaming,vox,edits,palette,source}.rs` +
`assets/shaders/voxel_raytrace.wgsl`).

## 0. The vision, restated as requirements

> "Seamlessly import `.vox` models into scenes and merge them together — this is how we sub-instance and
> spawn trees, props, etc. Eventually support NESTED voxel scenes that have their own trees etc., so we can
> have OFF-AXIS sub-voxel worlds — e.g. a tree on its side that we cut into."

Decomposed:

1. **`.vox` as reusable importable assets** — one parse → a shareable, object-local geometry blob.
2. **Instance / merge into a scene** — many placements of the same asset, shared geometry (true instancing).
3. **Nested voxel scenes** — a voxel object whose contents are themselves a placed sub-scene of instances.
4. **Off-axis** — an arbitrary rotation (and scale) per instance; "a tree on its side."
5. **Per-instance destruction** — cut into ONE instance, independent of the world and of sibling instances.

The non-negotiable engine constraint that shapes everything below: today the renderer builds **one global
BLAS** over every resident brick AABB and **one TLAS instance** with the identity transform
(`raytrace.rs::prepare_voxel_rt`, `max_instances: 1`, the `[1,0,0,0, 0,1,0,0, 0,0,1,0]` instance). The GI is
a **world-space** ReSTIR + SHARC-style cache keyed on **world position + world normal**
(`query_world_cache` in the shader). Both of those facts must change to support per-instance transforms —
this doc specifies how, and shapes Phase 3 of `soft-coalescing-dolphin.md` (per-chunk BLAS + multi-instance
TLAS) so the TLAS carries **arbitrary per-instance 3×4 transforms**, not just chunk translations.

---

## 1. Data model

### 1.1 The split: shared asset vs. placed instance

The whole architecture is one idea borrowed from Teardown and from standard HW-RT instancing: **geometry is
object-local and shared; placement is per-instance and cheap.**

- Teardown represents the world as *thousands of independent voxel volumes*, each its own grid + palette +
  transform, rasterized as a bounding box and raymarched in object-local space, with a per-object palette
  lookup. That is exactly a BLAS-per-object + TLAS-instance-per-placement in HW-RT terms.
- HW-RT 2-level instancing: the TLAS leaf stores a world→object transform; on entering a leaf the ray is
  transformed into the BLAS's local space and traversal continues there. **One BLAS can back many TLAS
  instances**, so repeated geometry costs memory once.

So we introduce two types.

```text
VoxelObject  (the ASSET — parsed once, shared, immutable geometry)
    object_local BrickMap         // bricks keyed on the OBJECT's own LOD0 grid, origin at object (0,0,0)
    palette: BlockRegistry        // the object's OWN palette (its .vox colours), see §1.4 on merging
    bounds_voxels: (IVec3, IVec3) // tight solid AABB in object-local voxel units (for the BLAS extent + LOD)
    lod_pyramid (optional)        // pre-downsampled coarse BrickMaps for distant instances (§6)
    handle: VoxelObjectId         // dense index; the SSOT key instances reference

VoxelInstance  (the PLACEMENT — many per object, cheap)
    object: VoxelObjectId         // which asset
    transform: Affine3A           // world_from_object: rotation + translation + (uniform) scale  ← OFF-AXIS
    edits: Option<VoxelEdits>     // per-instance override layer (None until first cut; §4)
    blas: BlasRef                 // SHARED asset BLAS, or a forked private BLAS once edited (copy-on-write)
    mask: u8                      // ray instance mask (optional: separate static world / props / debris)
    children: Vec<VoxelInstance>  // NESTED sub-instances (§3); empty for a leaf object
```

Key invariants:
- A `VoxelObject` is **immutable** after import. Editing never mutates the asset; it forks the instance (§4).
- A `VoxelInstance`'s `transform` is `world_from_object` (an `Affine3A`: rotation matrix + translation;
  uniform scale folded into the matrix). The TLAS stores its inverse-or-forward per the wgpu API; the shader
  receives **object-local** ray data automatically (§2).
- The **world itself is the root instance.** `VoxelScene::Worldgen` / `Cornell` / `Sponza` become the
  identity-transform instance(s) of the scene graph; nothing about the streaming clipmap changes for them.
  This is the unification: the existing streamed world is "instance 0 with `transform == IDENTITY` and a
  streamed `BrickSource`", and props are additional instances. (See §1.3, §7 risks on composing the streamed
  world with a multi-instance TLAS.)

### 1.2 Reusing `load_vox` — what changes and what doesn't

`vox::load_vox` is already a pure `path -> (BrickMap, BlockRegistry)` (§`vox.rs`). It already:
- builds the object-local `BrickMap` (`8³` / 0.2 m bricks — the SAME bricks the engine traces);
- builds a `BlockRegistry` from the `.vox` palette (sRGB→linear, `BlockId(i+1)`);
- walks the `.vox` scene graph (`nTRN`/`nGRP`/`nSHP`) to place multi-model files (`model_offsets`/`walk_scene`).

What we keep: the parse, the palette build, the brick build. **What we change for asset import:**

1. **Stop the floor/centre anchor for assets.** `bricks_from_placed` currently shifts the whole scene so its
   floor sits at `y=0` and it is X/Z-centred (a GI-measurement convenience for Sponza-as-world). For a
   reusable *asset* we instead anchor on a **declared pivot** (default: footprint centre on X/Z, min-Y on
   floor — the same numbers, but recorded as the object pivot, not silently baked into voxel coords). This
   keeps `transform.translation` meaningful ("place the tree's base here"). Refactor `bricks_from_placed`
   to return `(BrickMap, pivot_shift)` so both the Sponza-as-world path (apply shift) and the asset path
   (record pivot, keep object-local coords) share one SSOT. Robust-by-construction: one anchoring function,
   two callers, no divergence.
2. **Honour the `.vox` `_r` rotation byte in the scene graph (currently dropped).** `walk_scene` only reads
   `frames.first().position()` (the `_t`). The `.vox` `nTRN` frame also carries `_r` (an int8 rotation/
   reflection matrix code) and the file groups (`nGRP`) define sub-trees. For a *single merged object* we
   bake `_r` into the object-local voxels at import (it is a fixed authoring rotation, not a runtime one).
   For a file authored as a **scene of distinct named sub-models** (a `.vox` with several `nSHP`s under
   `nGRP`s) we have the OPTION to import each `nSHP` subtree as its OWN `VoxelObject` and emit a parent
   `VoxelInstance` with `children` — i.e. the file's own scene graph becomes our scene graph. This is the
   natural authoring path for "a tree `.vox` that already contains leaf-clusters as sub-shapes." Phase-gated
   (§7): merged-object import first; per-subtree import is the nested-import upgrade.

So `.vox` import has two modes, both built on `load_vox`'s primitives:
- **Merge mode** (default, ships first): the whole `.vox` (all models, `_t` + `_r` baked) → ONE `VoxelObject`.
- **Scene mode** (nested upgrade): each top-level `nSHP`/`nGRP` subtree → a `VoxelObject` + a child
  `VoxelInstance` carrying that subtree's `_t`/`_r` as a runtime transform.

### 1.3 Bevy ECS shape

```rust
// Resource: the asset arena. Dense Vec indexed by VoxelObjectId; shared, immutable geometry.
#[derive(Resource, Default)]
struct VoxelObjects { objects: Vec<VoxelObject> }      // + a path→id cache so re-import is free

// Component on a scene entity = one placement. Uses Bevy's own Transform/GlobalTransform for the
// hierarchy (so off-axis rotation + nesting reuse Bevy's parent/child propagation for free).
#[derive(Component)]
struct VoxelInstance { object: VoxelObjectId, mask: u8 }

#[derive(Component)]                                     // present only once a cut happens (COW; §4)
struct VoxelInstanceEdits(VoxelEdits);

// Nesting = Bevy entity hierarchy: a VoxelInstance entity with ChildOf children that are also
// VoxelInstance entities. GlobalTransform already composes the chain → world_from_object for the leaf.
```

Why lean on Bevy `Transform`/`GlobalTransform` + `ChildOf`:
- Off-axis rotation is just `Transform.rotation` — no new math.
- Nested sub-scenes are just child entities — `GlobalTransform` already gives the composed
  `world_from_object` per leaf, which is exactly the TLAS instance transform we need (§3).
- The editor/`soul_scene` `.scene` format and gizmos already manipulate `Transform`, so placement,
  rotation, and the move-gizmo come for free.

The **extract** step (main world → render world) gathers, per frame, the flat list of
`(VoxelObjectId, world_from_object: [f32; 12], mask, blas_ref, edit_generation)` for every leaf instance in
the camera region. That flat list is the TLAS build input (§2). Composing this with the streamed world:
instance 0 is the streamed clipmap patch (identity transform, the existing `VoxelRtPatch`); instances 1..N
are props. The TLAS holds **both** (§7 covers the BLAS-memory and per-frame-rebuild interaction).

### 1.4 Palettes: per-object vs. global

Each `VoxelObject` carries its own palette (the `.vox`'s 256 colours). Two voxel objects can both use
`BlockId(5)` to mean different colours. The shader indexes ONE global `palette` storage buffer today, so we
need per-instance palette addressing. Two options, we pick (B):

- **(A) Global palette merge + remap.** At import, merge the object's palette into a global registry and
  remap its brick `BlockId`s to global ids. Simple shader (one palette), but: ids drift per import order;
  the global palette can blow past `u16`; two identical assets imported twice duplicate colours; and a
  per-instance recolour/tint is impossible. Rejected as the primary model.
- **(B) Per-object palette + a `palette_base` offset** (chosen). Concatenate all object palettes into one
  global buffer; each `VoxelObject` records `palette_base` (its slice start). A brick's stored id stays the
  **object-local** id; the shader resolves `palette[instance.palette_base + local_id]`. The instance
  descriptor (§2.2) carries `palette_base`. This keeps object ids stable, supports identical-asset
  de-dup (same object → same base), and leaves room for a future per-instance tint (multiply `palette_base`
  lookup by an instance colour). Cost: one extra add per shade — negligible.

`BlockRegistry`/`palette.rs` already supports building from a raw `.vox` palette (`from_vox_palette`); the
new piece is the concatenation + per-object `palette_base`, which lives in `VoxelObjects` (one SSOT).

### 1.5 Native asset format (`.vxo`) — engine-owned, `.vox`-as-import-only

`.vox` becomes an **import/interchange** format (MagicaVoxel authoring, our offline voxelizer); the **native
runtime asset is our own format**. Rationale + design:

- **`.vox` already stores emissive** — via the `MATL` chunk (`_type:_emit`, `_emit` 0–1, `_flux` power exp), and
  `dot_vox` parses it into `data.materials`. But `from_dot_vox` reads ONLY the 256-colour `RGBA` palette and
  ignores `data.materials`, so we drop emissive (and roughness/metal/ior/etc.) today. So even staying on `.vox`
  we'd have to read `MATL`; and `.vox`'s string-keyed material dicts + 256-colour cap + scene-graph cruft are more
  than we want. We own it instead.
- **Keep `.vox`'s ONE good idea: tagged, length-prefixed, skippable CHUNKS** (RIFF-style). That structure is
  precisely what makes "store additional detail as we need + strip what we don't" robust-by-construction: a reader
  skips unknown chunks (forward-compat), and dropping an optional chunk doesn't break old assets (back-compat). Our
  own magic + version + chunk set, no `.vox` baggage.
- **The `.vxo` is the on-disk serialization of a `VoxelObject` (§1.1)** — and, with the `INST` chunk, of a whole
  scene of `VoxelInstance`s (§3.2 nesting). Chunks (each tagged + versioned; required unless noted *optional*):
  - `HEAD` — magic, format version, `voxel_size`, object bounds/anchor, name/tags.
  - `MATL` — the material table, per `BlockId` (`u16`, NO 256 cap): albedo (linear RGBA) + **emissive (linear RGB ×
    strength)** + reserved fields the renderer can grow into (roughness/metallic/ior/translucency) without a format
    break (new fields = a `MATL` version bump, old readers default them).
  - `BRIK` — the sparse `BrickMap` serialized **directly** (bricks @ coords + per-voxel `BlockId`), so the runtime
    loads with NO re-bricking (unlike `.vox`'s flat voxel list → our `from_dot_vox` re-buckets every load).
    Occupancy-mask + RLE/palette-compressed.
  - *optional* `LITE` — the pre-extracted air-exposed emissive-voxel light list (the NEE list, §GI) baked at import
    so the runtime skips the gather.
  - *optional* `LODS` — a pre-baked coarse-mip pyramid (the §6 per-object LOD), so far instances don't downsample
    at load.
  - *optional* `INST` — a scene: a list of `{ object_ref, transform (3×4), per-instance edits }` → this is how a
    `.vxo` carries nested sub-scenes (§3.2) and authored prop/tree placements.
  - *optional* `SOCK` / `PHYS` — attach/socket points for procedural placement; collision/physics hints.
- **Encoding:** binary for the runtime asset (compact, mmap-friendly; `postcard`/`bincode` per chunk body, or a
  hand-rolled layout). Optionally a sibling **RON "scene"** file for human-authored composition (instances +
  transforms) that references binary `.vxo` objects by path — authoring-friendly, compiles to an `INST` chunk.
- **Pipeline:** `.vox` / glTF → **offline import** (extend `examples/voxelize_scene.rs`; add a `.vox` `MATL`
  emissive reader so MagicaVoxel emission survives the round-trip) → `.vxo` (canonical). The shipped runtime depends
  on the `.vxo` reader only — never `dot_vox`/`gltf`/`image` (those stay DEV/offline deps, as today). Versioned so
  the schema evolves with the engine.
- **Import sampling quality** (planned ports from `D:\Projects\asset gen`, detail in memory
  `assetgen-voxelizer-improvements`): keep our conservative triangle-box-SAT occupancy + speed, but adopt
  **area-averaged albedo** (fixes the nearest-texel aliasing in `sample_albedo`), an **ALWAYS-ON exterior-floodfill
  interior SOLID fill** (user directive — interiors solid *always*, no flag; the destructible vision needs solid
  interiors so a cut reveals strata not empty space — open/exterior-reachable space stays air, only enclosed
  interiors fill), and a
  **CIELAB-space palette** (replaces sRGB median-cut). These close the import's albedo-aliasing / no-interior /
  palette-fidelity gaps and feed the `MATL`/`BRIK` chunks.

---

## 2. Render integration — the crux

### 2.1 BLAS per object, TLAS instance per placement

Replace the single global BLAS + single identity TLAS instance with:

- **One BLAS per `VoxelObject`**, built from that object's brick AABBs **in object-local space** (origin at
  the object pivot; the existing `gpu::brick_aabb` / `pack_*` produce exactly these, just fed the object's
  own bricks instead of world-resident bricks). Built **once at import**, retained for the object's lifetime,
  shared by every instance of that object.
- **The streamed world** keeps its current BLAS (it is just "the root object", whose geometry changes via
  streaming, so it rebuilds per the existing keep-old-until-revealed lifecycle and Phase 3's per-chunk BLAS).
- **One TLAS instance per leaf `VoxelInstance`**, carrying the leaf's `world_from_object` 3×4 transform and
  `instance_custom_index = descriptor_index` (§2.2). Many instances of the same object → many TLAS instances
  pointing at the **same BLAS** = true instancing, geometry stored once.

The wgpu API already exposes this: `TlasInstance::new(&blas, transform_3x4, custom_index, mask)`
(`raytrace.rs` uses it today with identity). The ONLY blockers are (a) `max_instances: 1` and (b) the
identity transform — both removed in Phase 3A below.

### 2.2 How a hit resolves to the right object's data (object-local DDA)

This is the subtle part. The shader's per-brick DDA (`dda_brick`) currently reconstructs cells from the
brick meta's **world** `world_min` and marches in **world** space. With instances, a brick's geometry is in
**object-local** space and the instance applies an arbitrary rotation — so the world-space DDA assumption
breaks. The fix is exactly what the hardware gives us for free:

**Run the DDA in object-local space, using the ray-query's object-local ray.** On a procedural-AABB
candidate, `ray_query` exposes the **object-space ray origin/direction** for the instance
(`rayQueryGetCandidate*` object-ray accessors on the wgpu-trunk fork; equivalently we transform `ro/rd` by
the instance's stored `object_from_world` we keep in the descriptor). The DDA then walks the brick's voxels
in the brick's own local grid — *identical math to today*, because today's "world" IS the root object's
local space (identity transform). So:

1. Candidate AABB hit → `instance_custom_index` (the descriptor index) + `primitive_index` (the brick index
   WITHIN that object's BLAS).
2. Look up `InstanceDescriptor[instance_custom_index]` → `{ object_id, object_from_world: mat3x4,
   meta_base, voxel_base, palette_base, edit_overlay_ref }`. `meta_base`/`voxel_base` offset into the global
   `metas`/`voxels` buffers so `metas[meta_base + primitive_index]` is the brick (object-local `world_min`).
3. Transform the world ray into object space: `ro_l = object_from_world * ro`, `rd_l = mat3(object_from_world)
   * rd` (rotation only on the direction). For a uniform-scaled instance the `t` along the local ray relates
   to world `t` by the (constant) scale factor — store `1/scale` in the descriptor and convert committed `t`
   back to world before comparing across instances. (Pure rotation + translation ⇒ scale 1 ⇒ no conversion;
   off-axis trees are this case.)
4. `dda_brick` runs unchanged in local space → first-solid local voxel, local entry-face normal, block id.
5. Commit the **world** `t` (local `t · scale`) via `rayQueryGenerateIntersection` so the TLAS resolves the
   nearest hit ACROSS instances + the streamed world correctly.
6. Shade: colour `= palette[palette_base + block_id]`; **rotate the local normal to world**
   `n_world = mat3(world_from_object) * n_local` (so lighting/GI use the correct world normal even for an
   off-axis tree). Emissive likewise from `palette[palette_base + id].emissive`.

The per-instance addressing therefore needs a new **`InstanceDescriptor` storage buffer** indexed by
`instance_custom_index`, holding the per-instance offsets + transform. This is the standard HW-RT pattern
("instance custom index → a per-instance descriptor"); it is cleaner than SBT records and portable on the
fork. The `metas`/`voxels`/`palette` buffers become **global concatenations** of all loaded objects + the
streamed patch, addressed by the descriptor's base offsets. (The streamed world is descriptor 0 with
identity transform and base offsets 0 — the existing code path falls out as the degenerate case.)

Concretely, the descriptor:

```rust
#[repr(C)] struct GpuInstanceDescriptor {
    object_from_world: [f32; 12], // 3x4 inverse transform (world→object) for the local ray
    world_from_object_rot: [f32; 12], // 3x4 (rotation+trans) to push the local normal back to world
    meta_base: u32,   // offset into global `metas`/`aabbs` for this object's bricks
    voxel_base: u32,  // offset into global `voxels`
    palette_base: u32,// offset into global `palette`
    inv_scale: f32,   // local-t → world-t factor (1.0 for rigid instances)
    edit_base: u32,   // offset into the per-instance edit-overlay voxel region, or SENTINEL if none (§4)
    mask: u32, _pad: [u32;2],
}
```

### 2.3 Shared BLAS (instancing) vs per-instance BLAS (unique destruction) — copy-on-write

- **Un-edited instances share the asset BLAS.** 1000 identical trees = 1 BLAS + 1000 TLAS instances + 1000
  descriptors. Geometry stored once; placement is 64 B (3×4) + a descriptor each.
- **The first cut into an instance forks it (COW).** When `VoxelInstanceEdits` first appears on an instance,
  we (a) clone the object's bricks affected by the edit overlay into a **private per-instance brick region**
  (only the dirty bricks; the rest still reference the shared object via the descriptor's `meta_base`), and
  (b) build a **private BLAS** for that instance (or, cheaper, a per-instance BLAS that is the object BLAS
  with the dirty chunk's AABBs replaced — see §4 refit). The descriptor's `meta_base`/`voxel_base`/`blas`
  swap to the private copies. Sibling instances are untouched (independence — the Teardown property).

When to use each:
- Static decoration that is never destroyed → shared BLAS forever (the common, cheapest case).
- Anything destructible → shared until first hit, then private. Most props are never actually cut, so the
  COW keeps memory near the shared-best-case in practice.

### 2.4 What this requires of `soft-coalescing-dolphin.md` Phase 3

Phase 3A as written says "one instance per **chunk** with a **translation** transform." This plan **widens
Phase 3A's contract**: the TLAS must carry **arbitrary per-instance 3×4 transforms** (rotation + translation
+ scale), not only chunk translations, and the hit path must resolve `instance_custom_index → descriptor`
(not just `instance_offset + primitive_index → global brick`). Doing this in Phase 3A is strictly more
general and costs nothing extra for the chunk case (a chunk is an instance whose transform happens to be a
translation and whose descriptor points at the streamed buffers). So: **build Phase 3A with the
`InstanceDescriptor` indirection and full 3×4 transforms from day one** — props/nesting then need no second
TLAS refactor. This is the single most important cross-cutting decision in this doc.

---

## 3. Nested / off-axis sub-grids

### 3.1 Off-axis is just the instance rotation

"A tree on its side" = a `VoxelInstance` whose `Transform.rotation` is a 90° (or arbitrary) rotation. Because
the ray is transformed into object-local space at the TLAS leaf (§2.2), the DDA walks the un-rotated brick
grid and is **correct by construction** — there is no special "rotated voxel" code. The only rotation-aware
steps are: transform the ray in (rotation on dir, full affine on origin) and rotate the hit normal out. Both
are 3×3 mat-vec ops already specified in §2.2. **No re-voxelization, no axis-aligned restriction.** This is
the central payoff of the BLAS/TLAS instancing model over a "merge voxels into the world grid" approach
(which could only ever do 90° rotations and would re-pack the world).

### 3.2 Nested voxel scenes

Two legitimate realizations; we support the first now and keep the door open for the second:

- **(Chosen) 2-level via Bevy hierarchy + a flattened TLAS.** A nested scene is a `VoxelInstance` with child
  `VoxelInstance`s (Bevy `ChildOf`). We **flatten** the hierarchy at extract time: each leaf's
  `GlobalTransform` is its composed `world_from_object`, and we emit ONE TLAS instance per leaf. So an
  N-deep nest of M leaves → M TLAS instances in the single TLAS. This needs no hardware multi-level support,
  works on any RT backend, and the GI/DDA are unchanged (every leaf is just another descriptor). Recursion is
  resolved on the CPU by `GlobalTransform` propagation; the GPU sees a flat instance list.
- **(Future) True hardware 2-level TLAS (TLAS-in-TLAS).** HW vendors support a BLAS that is itself an
  instance of a sub-TLAS (chained world→alt-world→object transforms). This would let a sub-scene be a single
  instance whose internals are not re-flattened when the parent moves (cheaper for huge, rarely-changing
  nests that move rigidly). Deferred — flattening is simpler, portable, and adequate until a nest has
  thousands of leaves that move together every frame. If we adopt it, the descriptor model generalizes (each
  level adds a transform compose), so it is not a throwaway.

**Depth / recursion limits.** Flattening makes depth a CPU concern only. Cap nesting depth (e.g. 8) to bound
`GlobalTransform` propagation cost and to catch authoring cycles; cap total flattened leaf instances by the
TLAS `max_instances` (allocate generously, e.g. 4096–16384, and cull by camera region — distant nested leaves
collapse to a single proxy instance, §6). A nest deeper than the cap clamps + logs (never panics), mirroring
the engine's existing robust-by-construction posture (the `.vox` walk already ignores malformed graphs).

### 3.3 Worked example — "a tree on its side that you cut into," end to end

1. **Import.** `import_vox("tree.vox")` → `load_vox` → object-local `BrickMap` + palette. Anchor on the
   tree's base pivot (record, don't bake). Build the tree's BLAS from its object-local brick AABBs
   (`gpu::pack` on the object's bricks). Concatenate its palette → `palette_base`. Store as
   `VoxelObject { id: 7 }`. **Done once**; re-import returns id 7 from the path cache.
2. **Place rotated.** Spawn an entity `VoxelInstance { object: 7 }` with
   `Transform::from_translation(p).with_rotation(Quat::from_rotation_z(PI/2))` (lay it on its side). Extract
   emits a TLAS instance: `world_from_object = GlobalTransform.affine()`, `custom_index = descriptor 42`,
   pointing at BLAS 7, descriptor 42 = `{ object 7 offsets, object_from_world = inverse(world_from_object),
   palette_base, inv_scale=1, edit_base=NONE }`.
3. **Ray hit.** Primary ray enters the tree's world AABB (rotated). TLAS transforms the ray into the tree's
   **upright** local space; `dda_brick` walks the upright brick grid → first-solid local voxel at, say,
   local `(3, 40, 2)` (high up the trunk in local Y = sideways in world). Commit world `t`. Shade: colour
   from `palette[palette_base + id]`; world normal `= mat3(world_from_object) * local_normal` (points
   sideways in world — correct for the toppled trunk). GI sees the correct world normal.
4. **Cut a voxel.** Click → CPU pick. The pick ray must hit the SAME voxel the GPU shaded, so the CPU pick
   transforms the world ray into the instance's object space (the inverse of step 2's transform) and runs the
   existing `pick_voxel` **in object-local voxel space** against the object's bricks ∪ this instance's edits.
   It returns the local voxel `(3,40,2)` + local face normal. `edits.remove(local_voxel)` on this instance's
   `VoxelInstanceEdits` (created on first cut — COW). The edit delta is **object-local** (so it travels with
   the instance and is independent of the world and of sibling trees).
5. **BLAS refit.** The cut dirties one brick (`dirty_bricks_for_edit` in local coords). On first edit, COW
   the instance: clone the dirty brick(s) to a private region, build/clone a private BLAS. Re-voxelize the
   dirty brick with `apply_edit_overlay` (already the SSOT), update its AABB if occupancy changed, and
   **refit** the private BLAS (§4) instead of rebuilding. Descriptor 42's `meta_base`/`voxel_base`/`blas`
   now point at the private copies. The world, and every other tree, are untouched. Next frame the cut is
   visible.

---

## 4. Per-instance destruction

### 4.1 Reuse the existing edit primitives, in object-local space

`edits.rs` is already the SSOT: a sparse `VoxelEdits` keyed by voxel coord; `apply_edit_overlay` resolves
`base unless overridden` per voxel; `dirty_bricks_for_edit` names the bricks (owner + halo neighbours) one
edit touches; `pick_voxel` DDA-marches `base ∪ edits`. **All of this works unchanged — the only change is the
coordinate frame: per-instance edits are keyed by the OBJECT's local voxel grid, not world voxels.** That is
the whole trick that makes destruction per-instance and rotation-independent: edits live in the un-rotated
object frame, so they travel rigidly with the instance and never collide with the world grid or a sibling.

### 4.2 Refit only that instance's BLAS — independent of the world

Mirror the SDF incremental-refit precedent (`sdf-bvh-incremental-refit`: O(depth) per-edit refit, ~7000× vs
rebuild) and Phase 3C's `PreferUpdate` plan:

- **AABB-stable edit (the common cut: occupancy/block change, brick still has the same AABB list)** → wgpu
  `update`/refit the private BLAS in place. No topology change, O(changed-brick) not O(object).
- **Topology change (a brick becomes empty and its AABB leaves, or empty space becomes solid and an AABB
  appears)** → rebuild ONLY the affected chunk's BLAS (chunk = the §3 Phase-3 8×8×8 brick meta-chunk),
  keeping the rest. For a small object the whole-object rebuild is still cheap (a tree is a few hundred
  bricks ≈ sub-ms — the same budget the static Cornell re-bake already pays per edit).
- **Build private BLASes with `PREFER_FAST_TRACE | ALLOW_UPDATE`** so refit is available; confirm the
  wgpu-trunk fork's `AccelerationStructureUpdateMode::PreferUpdate` semantics first (Phase 3C already flags
  this; reuse the finding).

Crucially this is **per instance**: the cut re-voxelizes the object's dirty bricks overlaid with THIS
instance's edits, refits THIS instance's private BLAS, bumps THIS descriptor's generation. The streamed
world's BLAS, and sibling instances' BLASes, never rebuild. Cost scales with *what you cut*, not world size —
the same principle as the streaming "adapt locally, never full-clear" rule.

### 4.3 The CPU pick across instances

The click pick must choose among many instances + the world. Cleanest: a CPU TLAS-lite — test the world ray
against each instance's world AABB (cheap, few hundred props), for each candidate transform the ray to object
space and run `pick_voxel` against that object's bricks ∪ instance edits, keep the nearest world-`t` hit
(convert local `t` by `inv_scale`). The world (streamed clipmap) is one more candidate using the existing
`pick_voxel` path. This mirrors how the GPU resolves nearest-across-instances, so CPU pick == GPU render
stays true by construction (the engine's standing invariant).

---

## 5. GI interaction — be honest about the hard part

The GI is **world-space**: ReSTIR reservoirs are per-screen-pixel but reproject through world motion, and the
SHARC/Solari world cache is a hash grid keyed on **quantized world position + world normal**
(`query_world_cache`). This has two regimes.

### 5.1 Static instances — easy, basically free

A non-moving instance (a placed tree that never moves) occupies fixed world-space voxels. Its surfaces
generate world-cache cells at their world positions exactly like terrain does; ReSTIR reservoirs reproject
normally. **The world cache and ReSTIR need essentially no change for static instances** — they already key
on world pos/normal, and the shader now produces correct **world** normals + world hit positions for
instances (§2.2 step 6). Per-object materials/emissive flow through the per-object palette (`palette_base`):
an emissive block in a `.vox` lamp lights the scene via the same NEE light-list + cache path as Cornell's
ceiling panel. The light-list builder (`gpu.rs::build_light_list`) must run over instance surfaces too — for
static instances it can be built once at placement (their emissive world positions are fixed) and merged into
the resident light list; this is additive and bounded by `MAX_VOXEL_LIGHTS`.

### 5.2 Moving / rotating instances — the genuinely hard case

A moving or rotating instance is fundamentally at odds with a **world-space** cache: a cell that was valid
last frame (when the tree was here) is wrong this frame (the tree moved), and the tree's surfaces now want
cells at NEW world positions. We must **adapt, not reset** (the standing GI rule — never full-clear the cache
or DLSS history on a move). Plan, in increasing order of fidelity (ship the cheap ones, escalate only if
needed):

1. **Decay handles it for slow/occasional movement (free).** The cache already has per-cell life/decay and
   lazy re-insert on query. A moved instance simply stops querying its old cells (they decay out over a few
   frames) and lazily fills new cells at its new world positions. For props that move rarely (a safe dragged
   once, debris settling), this is adequate — the same mechanism that already handles terrain edits.
2. **Targeted decay of the swept volume (cheap, robust-by-construction).** When an instance moves, mark the
   cache cells in its **swept world AABB** (union of last-frame and this-frame world bounds) for accelerated
   decay / invalidation — NOT a global clear. This bounds the stale-cell lifetime to the mover's footprint
   without touching the rest of the world's converged cache. Implement as a small per-frame "dirty world AABB"
   list the decay pass consults (additive to the existing decay pass). This is the recommended default for
   moving props.
3. **Per-instance (object-local) GI cache for fast continuous motion (hard, deferred).** For something that
   spins/translates every frame (a rotating platform, the "off-axis world you walk inside"), a world-space
   cache can never converge on its surfaces. The principled fix is a **second cache keyed in the instance's
   OBJECT space** (quantized object pos + object normal), so the cell identity is invariant under the
   instance's motion — the surface "carries its GI with it." Querying then blends: world-cache for static
   surroundings, object-cache for the mover. This is real new infrastructure (a second hash grid, per-object
   lifetime, a blend rule, and direct-light that must be re-evaluated in world space each frame because the
   *incoming* light changes as the object rotates even if the surface cell is stable). **Honest assessment:
   this is the single hardest part of the whole vision and should be deferred until a concrete moving-nested-
   world use case exists.** Static and rarely-moving instances (the actual "trees and props" ask) are fully
   served by 5.1 + 5.2.1/5.2.2.

**Summary of honesty:** trees/props placed (even rotated) and occasionally destroyed → GI is easy (world
cache + correct world normals). Continuously rotating sub-worlds with full GI → genuinely hard; object-space
cache is the known answer but is deferred, and until then such a mover gets decay-based approximate GI (still
correct direct light + DLSS-RR temporal, just a softer/laggier indirect term on its own surfaces).

---

## 6. LOD — instances at distance

Two LOD systems must coexist: the world's **clipmap** (continuous, camera-following, `(coord,lod)` keyed) and
**discrete instances** (each an independent object).

- **Per-object LOD pyramid.** At import, pre-downsample the object's `BrickMap` into a few coarser
  `BrickMap`s (the existing `StaticVoxSource` downsample — solid-if-any + dominant block — is exactly this,
  already deterministic). Build one BLAS per LOD level (small; a tree's coarse levels are tiny). At extract,
  pick the object LOD by the instance's **projected screen size** (world distance / object radius), and emit
  the TLAS instance pointing at that level's BLAS + descriptor (the per-level `meta_base`). One instance,
  one LOD, swapped by distance — no clipmap shells per prop.
- **Far proxy / impostor.** Beyond a threshold, collapse an instance (or a whole nested sub-scene) to a
  single coarse proxy: its lowest-LOD BLAS, or even a single emissive-averaged AABB, so a forest of thousands
  of distant trees costs a handful of instances. Cull instances outside the camera region entirely (they
  contribute nothing to the TLAS) — the same region the clipmap uses.
- **Coexistence.** The clipmap and instances are independent TLAS members; the nearest-`t` ray-query merge
  already resolves "a prop in front of terrain" correctly (it is just another candidate). No cross-system
  LOD seam exists because they are separate BLASes resolved by depth, not stitched grids. The one care point:
  an instance's coarse LOD must not pop visibly against nearby terrain — pick the LOD transition distance by
  screen-space voxel size (match the clipmap's sub-pixel target), reusing the clipmap's existing LOD-distance
  math.

---

## 7. Phasing

Each phase is independently shippable, testable (headless brick/DDA tests + the GPU ground-truth harness +
the perf rig), and leaves the engine in a working state. The ordering front-loads the load-bearing render
refactor (multi-instance TLAS) because everything else depends on it, then layers capability.

**Dependency gate (Phase 0):** the Sponza-unification `BrickSource` (already landed) + Phase-3A multi-
instance TLAS. The instancing work cannot begin until the world is *one instance among many* rather than a
hardwired identity BLAS. So Phase 1 below IS a re-scoped, widened Phase 3A of `soft-coalescing-dolphin.md`.

- **Phase 1 — Multi-instance TLAS + `InstanceDescriptor` indirection (widened Phase 3A).** Replace
  `max_instances:1` + identity with a TLAS of N instances, each a 3×4 transform + `instance_custom_index →
  GpuInstanceDescriptor`. Move `metas`/`voxels`/`palette` to global concatenations addressed by descriptor
  base offsets. The streamed world becomes descriptor 0 (identity, base 0) — **zero visual change** is the
  acceptance test. Object-local DDA path in the shader (degenerate to identity for the world). *Test:* extend
  `voxel_raytrace_gpu` to a 2-instance scene (world + one translated cube object) and assert GPU == CPU hit;
  perf rig shows no regression on the pure-world scene. *Risk:* shader hit-handler rewrite; mitigate by
  keeping descriptor 0 identical to today and diffing the rendered Cornell/Sponza frame.

- **Phase 2 — `.vox` asset import (merge mode) + axis-aligned placement.** `import_vox → VoxelObject`
  (object-local BrickMap + palette + BLAS + `palette_base`), refactor `bricks_from_placed` to record a pivot
  instead of baking the floor anchor, `VoxelObjects` resource, `VoxelInstance` component, extract → TLAS
  instances with **translation-only** transforms first. Place several trees in a worldgen scene. *Test:* a
  headless test that an imported `.vox` instance's bricks hit at the placed translation; a 1000-instance
  scene shares one BLAS (memory assertion). *Ship:* "spawn trees/props as shared instances."

- **Phase 3 — Off-axis (arbitrary rotation + uniform scale).** Add rotation/scale to the instance transform;
  shader transforms the ray in + rotates the normal out (§2.2); CPU pick transforms into object space.
  *Test:* a rotated cube instance — GPU == CPU hit + correct world normal vs an analytic oracle; the
  "tree on its side" visual. *Ship:* off-axis props.

- **Phase 4 — Nested sub-scenes (flattened).** Bevy `ChildOf` hierarchy of `VoxelInstance`s; extract flattens
  via `GlobalTransform` to leaf TLAS instances; depth/instance caps. `.vox` "scene mode" import (each
  `nSHP`/`nGRP` subtree → object + child instance, honouring `_t`/`_r`). *Test:* a 2-level nest (a parent
  instance with child instances) flattens to the right world transforms; a `.vox` with a sub-shape graph
  imports as nested. *Ship:* "a voxel object whose contents are a sub-scene."

- **Phase 5 — Per-instance destruction (COW + refit).** `VoxelInstanceEdits` (object-local `VoxelEdits`),
  copy-on-write fork on first cut, per-instance BLAS refit (mirror SDF refit / Phase 3C), per-instance CPU
  pick. Static-instance light-list contribution for emissive props. *Test:* cut one of two identical tree
  instances; assert the sibling + world are byte-unchanged and only the cut instance's BLAS rebuilt; the cut
  is visible. *Ship:* "cut into a specific instance, independent of the world." This is the worked example §3.3.

- **Phase 6 — LOD + far proxies + moving-instance GI decay.** Per-object LOD pyramid + screen-size LOD
  select, far-proxy collapse, region cull, and the §5.2.2 swept-AABB targeted cache decay for moving
  instances. *Test:* a forest of N distant trees collapses to ≤K instances; a moved instance's stale cache
  cells decay locally without a global clear (the world cache convergence test elsewhere stays green).
  *Defer:* §5.2.3 object-space GI cache for continuously-rotating sub-worlds — only if a use case lands.

### Risks & open questions

- **Phase-1 shader rewrite is the highest-risk single change** (the hit handler is load-bearing for primary,
  shadow, AO, and GI rays — `trace`/`trace_occluded`/`brick_hit_at` all re-walk the committed brick). Mitigate
  by making descriptor 0 bit-identical to today and gating the multi-instance path so Cornell/Sponza render
  pixel-identical before any prop exists.
- **BLAS memory for many distinct objects.** Shared BLAS makes *instances* cheap, but each distinct *object*
  is a BLAS. A world with thousands of *unique* assets is a different cost than thousands of *copies*. Bound
  by de-duplicating identical imports (path cache), an LOD pyramid that keeps coarse BLASes tiny, and far-
  proxy collapse. Open question: a BLAS budget + LRU eviction of off-screen objects' BLASes (rebuild on
  re-entry) if VRAM binds — measure first (the perf rig is the gate).
- **Per-frame TLAS rebuild cost with many instances + a streaming world.** The TLAS must include the
  streamed world (which re-packs on stream-in) AND all props. Building/refitting one TLAS per frame over
  thousands of instances must stay cheap; wgpu TLAS build is fast but measure at the instance cap. Open
  question: incremental TLAS instance update (touch only moved/added/removed instances) vs full TLAS rebuild
  per frame — likely needed at scale; sequence after Phase 1 once the cost is measured.
- **Palette `u16` headroom.** Per-object palettes + a global concatenation could exceed a single buffer's
  practical size with very many large-palette assets. The `palette_base` model keeps *object-local* ids at
  `u16`; the global offset is `u32`. Fine; flagged for completeness.
- **The world-space cache vs continuously-moving instances (§5.2.3)** is the deepest open problem and is
  explicitly deferred with a known answer (object-space cache). Do not gate the trees/props vision on it.
- **`.vox` `_r` rotation/reflection decoding** (bake at import) must be validated against MagicaVoxel ground
  truth — the existing loader drops `_r` today; add round-trip tests when Phase 2/4 land it.

---

## 8. Why this composes with the existing engine (summary)

- **Bricks, halo, palette, NEE, the DDA** are all reused unchanged — instancing only adds a coordinate
  transform at the TLAS leaf and a descriptor indirection for buffer offsets. The brick is still `8³`/0.2 m;
  `gpu::pack_*` still produces AABBs/metas/voxels; `dda_brick` still walks `8³`.
- **`load_vox` and `edits.rs` are the import + destruction primitives** — reused as-is, only re-framed into
  object-local space.
- **The streamed world is the degenerate root instance** (identity transform, descriptor 0), so the clipmap,
  `BrickSource`, streaming lifecycle, and keep-old-until-revealed are untouched.
- **Phase 3 of `soft-coalescing-dolphin.md` is shaped by this doc** to carry full per-instance 3×4 transforms
  + a descriptor table from the start, so no second AS refactor is ever needed.
- **GI adapts, never resets** — static instances are free; moving instances decay their swept volume locally;
  the world cache is never globally cleared.

### Prior art this design draws on
- **Teardown** — many independent voxel volumes, each its own grid + palette + transform, raymarched in
  object-local space from a bounding box; the per-object-volume + per-object-palette model is exactly our
  VoxelObject/VoxelInstance split.
  ([gamedeveloper.com](https://www.gamedeveloper.com/design/how-beautiful-voxels-laid-the-way-for-i-teardown-s-i-heist-y-framework),
  [acko.net Teardown frame teardown](https://acko.net/blog/teardown-frame-teardown/))
- **HW-RT 2-level instancing** — TLAS leaf stores a world→object transform, ray transformed into BLAS-local
  space, one BLAS shared by many instances; nested instancing = chained transforms (our deferred true 2-level
  option). ([USPTO 11,282,261 — alternative world-space transforms](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11282261))
- **MagicaVoxel `.vox` scene graph** — `nTRN` (with `_t` translation + `_r` rotation byte) / `nGRP` / `nSHP`
  nested transform nodes; we already parse `_t` and now read `_r` + the group hierarchy for nested import.
  ([ephtracy voxel-model .vox extension spec](https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox-extension.txt))
- **Brickmap/SVO instancing & LOD** — object-local sparse bricks + per-object downsample pyramids
  (already in `StaticVoxSource`) for distant instances.
