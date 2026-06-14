# Voxel Large-Scene / Far-View Residency — Design Plan

Status: DESIGN (no engine code changed by this doc). Worktree: `voxel-rt`. Read-only survey of `src/voxel/**`
+ a phased plan. Target: Bevy 0.19 + forked wgpu-trunk, RTX 4090 / Vulkan. Extends the HW-ray-traced brickmap
path (`src/voxel/{brickmap,gpu,streaming,raytrace}.rs` + `assets/shaders/voxel_raytrace.wgsl`).

**Scope.** This is the RESIDENCY / STREAMING / SPARSE-HIERARCHY architecture — *how much is kept GPU-resident
for a large view*. It is the orthogonal axis to **per-brick compression**, which `docs/VOXEL_STORAGE_PLAN.md`
already covers (R1 uniform-collapse-into-VRAM **[landed]**, R2 per-brick palette, R3 brick dedup). This doc
builds **on** R1–R3; it does not re-derive them. The new question: *how do SOTA engines make the view/scene
huge at bounded VRAM?*

---

## 0. The hard constraint (state it, design around it)

Same line in the sand as the storage plan, restated for the residency axis: **the live render path is hardware
ray tracing.** A ray hits a brick's procedural AABB (a BLAS primitive), then the shader DDA (`dda_brick`,
`voxel_raytrace.wgsl`) walks the brick's `8³` cells. So the **live, traced structure must stay O(1)-DDA-friendly
and flat** — a brick is addressed purely by `metas[prim].voxel_offset`; no pointer-chasing SVDAG/octree descent
on the hot path. Any deep tree is a **transport / storage** form (Tier B) that is *decoded* to the flat
brick-pool form (Tier A) before the trace ever touches it. The storage plan's §0 Tier-A/Tier-B split is the same
discipline; residency adds a second axis on top:

- **Resident (Tier A, GPU):** in the BLAS + the meta/voxel/AABB/palette buffers the ray query reads.
- **Non-resident, CPU-only or generatable:** in the sparse CPU brickmap (`brickmap.rs`) or *re-derivable on
  demand* by the worldgen source. Costs ~nothing in VRAM; can be promoted to resident in O(1) when needed.

The whole game of this doc is **shrinking the resident (Tier A) set from O(view-volume) to O(visible-surface)**
while the *addressable* world stays unbounded.

---

## 1. Where we are today (ground truth from the code)

### Residency = a dense nested clipmap of bricks (`streaming.rs`)
`desired_clipmap(cam_world, cfg)` enumerates, per LOD `L ∈ 0..=MAX_LOD`, a `(2·clip_half+1)³` cube on the LOD-`L`
grid (LOD0 fills the inner cube; each coarser level is the shell `clip_half/2−1 < cheby ≤ clip_half`). With the
shipping `StreamingConfig` (`clip_half_bricks = 8`, `MAX_LOD = 7`) that is **8 nested shells reaching
`8·1.6·2⁷ ≈ 1640 m`**, capped at `max_resident_bricks = 60_000`. Empty (all-air) bricks are dropped + memoized
(`ResidencyManager::empty`), so *sky* is already free.

### Every SOLID brick is GPU-resident — including buried interiors
`ResidencyManager::resident` stores every non-empty `(coord, lod)` brick; `resident_entries()` hands them all to
`pack_resident_set` (`gpu.rs`), which emits **one AABB + one `GpuBrickMeta` per brick** into the patch. The
render path (`prepare_voxel_rt`, `raytrace.rs:2199`) builds **one BLAS with `primitive_count = brick_count`** and
a single-instance TLAS. So **resident-brick count == BLAS primitive count == AABBs the ray query must consider.**

### What R1 (landed) does — and what it does NOT do
The uniform-incl-halo collapse (`gpu.rs`, `BRICK_UNIFORM_FLAG`) already detects a **fully-buried** brick (core +
entire `10³` halo one solid block) and drops its **voxel-buffer bytes** (4000 B → ~8 B in the meta). This is the
storage-plan win and it is real. **But the buried brick STILL emits an AABB and STILL consumes a BLAS primitive
+ a meta slot.** R1 shrinks *bytes per brick*; it does **not** shrink the *brick count*, the *BLAS size*, or the
*TLAS candidate set*. That is exactly the gap this doc closes.

### The cost model that blocks a 10× farther view
The reported worldgen slice cold-fills to a 60k-class resident set; the perf rig
(`tests/voxel_worldgen_perf.rs`) reports `pack_resident_set` + the BLAS rebuild as O(resident-bricks), and the
voxel buffer at ~240 MB pre-R1. The resident-brick count grows **cubically** with `clip_half` (each LOD shell is
`O(clip_half²)` but LOD0 is a full `O(clip_half³)` cube, and the dominant near-field is volumetric once interiors
are solid). Push `clip_half` for a 10× finer near view and the resident set — hence BLAS build time, TLAS
traversal candidacy, and meta/AABB VRAM — blows up cubically even with R1 zeroing the voxel bytes. **R1 fixed the
bytes; the brick COUNT is the next wall.**

### The DDA already terminates at the first surface (the key enabler)
`trace` (`voxel_raytrace.wgsl:299`) walks all candidate AABBs the ray pierces and keeps the **nearest** per-voxel
first-solid hit (`best_t`), feeding `rayQueryGenerateIntersection(ht)` so the TLAS culls farther candidates. A
ray that hits a solid surface **never reaches the bricks behind it**. Therefore a brick whose **all six face
neighbours are solid** (an *enclosed* brick) can never be the first-solid hit for *any* primary, shadow, or GI
ray — every ray that could reach it must first commit a hit on the solid neighbour in front. **Enclosed bricks
are pure dead weight in the BLAS.** This is the structural fact the whole plan rests on.

### What we already have to build on
- A sparse CPU brickmap with `Uniform | Dense` storage + per-brick `occupancy: [u64; 8]` (`brickmap.rs`) — the
  cheap CPU home an evicted/never-resident brick lives in (for destruction).
- `solid_fill` (`examples/voxelize_scene.rs:484`) makes imported-model interiors always-solid — *creating* the
  enclosed-brick mass this plan culls from VRAM. (Worldgen bedrock is likewise solid below the surface.)
- AABB-BLAS-from-a-GPU-buffer works on the fork (`raytrace.rs:2255`); the GPU-worldgen pivot
  (`docs/GPU_VOXEL_WORLDGEN_PLAN.md`) plans a fixed-cap GPU brick pool with degenerate AABBs for free slots — the
  exact substrate ray-guided residency needs.
- A device-free `StorageReport` + a perf rig (`tests/voxel_worldgen_perf.rs`) to measure every delivery.

---

## 2. How SOTA systems bound memory for a LARGE view

Each row: the **residency idea**, and whether it is **Tier A (lives in the traced structure)** or a
**transport/decode** form that never touches the hot path.

| System | How it bounds VRAM for a huge view | Tier / our fit |
|---|---|---|
| **Surface-only / shell residency** (Teardown-class + every "ray stops at first hit" tracer) | A ray commits at the first surface, so any voxel/brick with all neighbours solid is **never hit** → keep only the **air-exposed surface shell** in the traced set; enclosed interiors live in cheap CPU/disk storage only. Resident set → **O(surface) not O(volume)**. | **Tier A residency policy** — pure cull of the BLAS set. Our DDA already terminates at first-solid (§1), so this is *correct by construction*. **Biggest, lowest-risk win.** |
| **GigaVoxels** (Crassin, INRIA 2009) | SVO of nodes → pointers into a **fixed-size GPU brick pool**; rendering emits a **ray-guided** request buffer (each ray reports the node/brick LOD it needed but found missing); an **LRU** pool evicts the least-recently-used bricks. View distance is **decoupled from VRAM** — "walk through very large and detailed worlds in real time in bounded GPU memory." | **Demand residency model.** The octree is GigaVoxels' *index*; we already have a flat index (clipmap + BLAS). Adopt the **ray-guided request + LRU-pool** idea on our brick pool; keep our flat addressing (no SVO descent on the hot path). |
| **SVDAG / SSVDAG** (Kämpe 2013 / Villanueva 2016) | Merge **identical empty AND uniform subtrees** into a DAG → mostly-empty/uniform worlds collapse to a tiny graph (SSVDAG ≈ 0.12 bits/voxel). | **Tier B transport only.** A DAG is pointer-chasing — wrong for the in-shader DDA. It is a *stored/streamed* form, **decoded to a brick pool** before tracing (see Aokana). Useful for static `.vox` assets (storage-plan R5), not the live structure. |
| **Aokana** (arXiv 2505.02017, ACM CGIT 2025) | World stored as **per-chunk SVDAGs** + an **LOD aggregation** mechanism + a **streaming system** for seamless open-world traversal; the SVDAG is the compact *stored/streamed* form, **decoded to a GPU-traceable representation** per resident chunk. Reported: **tens of billions of voxels**, **up to 9× less memory** and **up to 4.8× faster** than prior SOTA. *(The "~10 B voxels / ~400 MB / ~5% resident / 6 ms" figures in our internal survey are a paraphrase; the paper's published headline numbers are the 9× / 4.8× ones.)* | **The full target shape**: chunked SVDAG storage → LOD-aggregated → demand-streamed → **decoded to a traceable pool**. We are SVDAG-free on the trace path by design; we adopt the *streaming + LOD-aggregation + decode-to-pool* spine, with our flat brick as the decoded form. |
| **Nanite-style virtualized geometry / virtual textures** (UE5) | Stream **only what the screen needs at the LOD it needs**: LOD chosen by **projected screen size**; clusters below a pixel threshold are culled; geometry paged from disk on a GPU feedback request, hierarchy metadata kept resident. | **The selection principle.** Our clipmap already approximates "coarser LOD farther away" by distance; the Nanite refinement is **screen-error / pixel-projected LOD** + **feedback-driven paging** — i.e. don't make a brick resident until a ray/pixel asks for it at that LOD (this is GigaVoxels' ray-guidance, re-derived for triangles). |
| **dubiousconst282 sparse-64-tree** (2024) | "**Many small trees**" (a top-level grid of small `4³`-node trees) rather than one monolith; brick **occupancy masks** for two-level DDA space-skipping; DAG-based **instancing** of repeated regions. | **Chunking + occupancy-skip.** Confirms: chunk the world (we do — clipmap cells), keep **flat brick leaves** for the DDA, and use **instancing** for repeats (our storage-plan R3 dedup). A deep 64-tree is *not* wanted on our trace path (BLAS already does empty-space skipping). |
| **NVIDIA / OptiX large-scene RT** (general HW-RT practice) | Many **small BLASes** under a TLAS so only *changed/visible* BLASes rebuild; instancing of repeated geometry; compaction. | **TLAS/BLAS granularity.** Motivates the GPU-worldgen pivot's per-chunk BLAS (Stage 3 there): rebuild only changed chunks, not one monolithic BLAS. Composes with surface-only (fewer primitives per chunk BLAS). |

**The synthesis for us:** the headline-ratio methods (SVDAG/SSVDAG/Aokana-chunk) are **storage/transport** forms,
decoded to a flat traceable pool — they are *not* what bounds the *resident* set; they bound *disk/stream* size.
What bounds the **resident GPU** set for a large view is the same three ideas, all Tier-A-policy:
1. **Surface-only residency** — never make an enclosed brick resident (correctness-free given first-hit DDA).
2. **Demand / ray-guided residency** — make a brick resident only when a ray actually needs it at that LOD;
   bound the pool with an LRU; a far view requests coarse LOD.
3. **LOD aggregation** — coarse bricks cover `2ᴸ×` more world at fixed `8³` resolution (our clipmap already does
   this; surface-only + demand let it grow far).

---

## 3. The brick-budget math: cubic → quadratic (the core quantification)

Let `H = clip_half_bricks`, and treat the near-field as the dominant cost (LOD0 + the fine shells, where the
solid-interior mass lives). Define the resident-brick count for the worst case (camera at/under a solid surface,
e.g. terrain or a solid building):

### Today (every solid brick resident)
LOD0 is a full cube: `(2H+1)³` candidate bricks; once interiors are solid_fill'd, a large fraction are
**solid** (sky is dropped, but buried stone/bedrock is kept). The resident solid count scales as:

```
N_today  ≈  k_solid · (2H+1)³            →   Θ(H³)   (CUBIC in clip_half)
```

For `H = 8`: `(17)³ = 4913` LOD0 bricks alone; ×8 LOD shells; the rig observes the 60k-class cap binding. To push
the near view **10× finer reach** you raise `H` ~10× (to keep the same coarse LOD count) → `N` grows **~10³ = 1000×**.
Infeasible at a fixed budget — the BLAS build, the TLAS candidate set, and meta+AABB VRAM all scale with `N`.

### After surface-only residency (only air-exposed bricks resident)
A solid mass of `m³` bricks has an enclosed interior of `(m−2)³` (all-neighbours-solid) and a surface shell of
`m³ − (m−2)³ ≈ 6m²`. Keeping only the shell:

```
N_surface  ≈  c · (2H+1)²                →   Θ(H²)   (QUADRATIC in clip_half)
```

because the resident set is the **2D surface manifold** threading the 3D clip volume (terrain is a height-field
→ one surface layer per XZ column; a building is hollow shells). The cubic interior term **vanishes from VRAM**
(it stays in the cheap CPU brickmap for destruction).

**The change is Θ(H³) → Θ(H²).** Pushing the view 10× farther now costs **~10² = 100×** more resident bricks
instead of **1000×** — a **10× structural reduction in the growth rate**, on top of the constant-factor win of
dropping today's buried mass. Concretely, for a terrain slice where the buried interior is the bulk:

| | resident bricks | BLAS primitives | meta+AABB VRAM | voxel VRAM (post-R1) |
|---|---|---|---|---|
| **Today** (`H=8`) | ~60k (cap binds) | ~60k | ~3.8 MB (60k·64 B) | shell-only already (R1) |
| **Surface-only** (`H=8`) | ~ shell only (∼ few k–15k, content-dep.) | same drop | ~0.6–1 MB | strictly ≤ today |
| **Surface-only** (`H≈26`, ~10× reach at same coarse-LOD budget) | ~Θ(H²) ≈ same order as today's H=8 cube-corner, **not** 1000× | quadratic | quadratic | quadratic |

So **surface-only residency is what lets `clip_half` (or `MAX_LOD`) grow ~10× at a bounded budget** — it converts
the dominant cost from the view *volume* to the view *surface*. Demand/ray-guided residency (§5 Phase B) then caps
the *remaining* surface set with an LRU so even a pathological all-surface view (a voxel forest) can't exceed the
pool. Numbers above are predictions to be **measured**, not trusted (§7).

---

## 4. Surface-only GPU residency — the design (Phase A, the big win)

### 4.1 Definition (correct by construction)
A resident brick is **enclosed** iff every one of its 6 face-neighbour bricks is **fully solid across the shared
face** (the 64-voxel face plane of the neighbour adjoining this brick is all solid). An enclosed brick can never
be the nearest first-solid hit (§1), so **it need not be in the BLAS / Tier-A set at all** — only in the CPU
brickmap. The trace result is **identical** with or without it. This is not an approximation; it is exact for a
first-hit DDA. (Contrast with the old SDF conservative-occupancy mask, `[[sdf-conservative-occupancy]]`, which had
to be *conservative* because the SDF empty-space DDA could skip past thin features; here we are culling
**provably-unreachable** bricks, so it is exact, not conservative.)

### 4.2 Detection — reuse what exists, add one face test
We already compute per-brick `occupancy: [u64; 8]` (`brickmap.rs`) and the packer already resolves same-LOD
neighbours (`neighbour_border_cell`, `by_key` in `pack_resident_set`). Add a **face-solid predicate** and an
**enclosed predicate**:

- `Brick::face_full(face) -> bool` — the 64 occupancy bits of one face plane are all set (6 cheap masks). For a
  `Uniform(solid)` brick this is trivially `true`; for `Dense` it is a masked-popcount per face.
- A brick is **enclosed** iff, for all 6 faces, the **adjacent neighbour brick exists, is same-LOD, and its
  opposing face plane is full**. This subsumes R1's uniform-incl-halo test (a uniform-incl-halo brick is a
  *special case* of enclosed) — so **R1's collapse and surface-only cull share one SSOT predicate** rather than
  two parallel checks. An enclosed brick whose own core has interior air (a Dense brick with a sealed cavity) is
  *also* cullable from the BLAS, because no ray can reach the cavity either — a strict generalization of R1.

The predicate runs in `pack_resident_set` / the GPU pool fill, exactly where R1's collapse runs today.

### 4.3 What changes in the pipeline
- **`ResidencyManager`** keeps storing every solid brick in `resident` (the CPU home for destruction — unchanged).
- **`resident_entries()`** (or a new `surface_entries()`) **filters out enclosed bricks** before handing the list
  to `pack_resident_set`. The CPU set is the SSOT for "what exists"; the *packed* set is the SSOT for "what the
  GPU traces." This keep-the-CPU-set / shrink-the-GPU-set split is the robust-by-construction shape.
- **`pack_resident_set`** emits AABB + meta + voxels only for **surface** bricks → BLAS primitive count, TLAS
  candidacy, and meta/AABB VRAM all drop to O(surface). R1's voxel-byte collapse is unchanged (and now rarely
  fires, because enclosed-uniform bricks are culled entirely before R1 would collapse them — a *uniform surface*
  brick on an exposed face still keeps its dense halo, as today).

### 4.4 Incremental maintenance under edits + streaming (the only real risk)
The correctness hazard: an edit that **exposes a previously-enclosed brick's face** must **promote** that brick
into the resident/BLAS set the same frame, or a hole appears.

- **Edit (dig/place).** The edit path already re-queues the owner LOD0 brick **+ its halo neighbours**
  (`affected_resident_keys` → `dirty_bricks_for_edit`, `raytrace.rs:520`). Extend it so the re-pack **re-evaluates
  the enclosed predicate** for the owner *and its 6 neighbours*: digging a voxel turns the owner from enclosed →
  surface (it now has an air-exposed face) → it must enter the BLAS; and its newly-exposed neighbour across the
  cut likewise. Because the predicate is a pure function of the (post-edit) occupancy of a brick + its neighbours,
  the dirty set (owner + neighbours, already computed) is *exactly* the set whose enclosed-ness can change — **no
  wider invalidation, robust by construction.** A place that *seals* a face demotes a brick surface → enclosed →
  it leaves the BLAS (a pure win; never a hole).
- **Streaming (brick enters/leaves a shell).** When a brick is voxelized, evaluating its enclosed-ness needs its 6
  neighbours present. At a **shell boundary** a neighbour may be a different LOD or not-yet-drained → treat a
  **missing/different-LOD neighbour as NOT-full** (i.e. the face is *potentially* exposed) → the brick stays
  resident (conservative *toward keeping*, never *toward dropping*). This is the same fallback the halo already
  uses (`neighbour_border_cell` returns AIR at a shell boundary). So a brick is culled **only when all 6
  neighbours are present, same-LOD, and full** — never on incomplete information. As the shell fills in, a brick
  can transition enclosed and drop on the next re-pack; the keep-old-until-revealed cadence covers the interim.
- **Interaction with solid_fill + R1.** `solid_fill` is what *creates* the enclosed mass; surface-only is what
  stops us paying for it in VRAM while keeping it in the CPU brickmap for destruction. R1 becomes a *subset* of
  this cull (uniform-incl-halo ⊂ enclosed). The two share the §4.2 SSOT predicate.

### 4.5 Why this is the biggest, lowest-risk win
It is a **pure filter on the packed set** — no shader change, no new buffer, no hot-loop cost (fewer primitives =
*faster* traversal + *faster* BLAS build). It degrades gracefully (cull nothing → today's behaviour). The only
new logic is the enclosed predicate + extending the already-existing edit-dirty re-pack to re-evaluate it. It
turns the dominant cubic interior term into zero VRAM **before** any of the harder demand-streaming machinery.

---

## 5. Demand / ray-guided residency (Phase B — caps the remaining surface set)

Surface-only makes the resident set O(surface); for a *very* far / *very* detailed view even the surface can
exceed a fixed pool (a forest, a city). GigaVoxels' answer — **ray-guided demand paging + LRU pool** — caps it.

### 5.1 The model, mapped to our fixed-cap GPU pool
The GPU-worldgen pivot (`docs/GPU_VOXEL_WORLDGEN_PLAN.md`) already specifies a **fixed-capacity GPU brick pool**
with **degenerate AABBs for free slots** and a GPU free-list. Ray-guided residency layers onto it:

1. **Feedback buffer.** During the trace, when a ray would refine into a brick/LOD that is **not resident** (a
   coarse brick was hit but a finer one is wanted, or a frustum brick is missing), append its `(coord, lod)` to a
   GPU **request buffer** (append via an atomic counter — standard Nanite/GigaVoxels feedback). No readback on the
   hot path; the request buffer is consumed by the next streaming step.
2. **Residency decision = requests, not a fixed cube.** The CPU/GPU coarse-residency step reads the (compacted)
   request buffer and fills the requested slots from the GPU voxelizer (worldgen) or the CPU brickmap (static
   `.vox`). The clipmap becomes a **seed/prefetch** (what to have ready before the first ray asks), not the
   *definition* of residency.
3. **LRU eviction.** Each resident slot carries a `last_used_frame` (written by the trace when a ray reads it).
   When the pool is full, evict the least-recently-used **surface** slots. Eviction just frees the slot + writes a
   degenerate AABB for it (no BLAS realloc; the per-chunk BLAS rebuild of the GPU-pivot Stage 3 handles it).
4. **LOD by aggregation for the far field.** A far ray requests a **coarse** brick (large `lod`), which covers
   `2ᴸ×` more world for one slot — so the far field is bounded by *angular* resolution, not world distance. This
   is the Aokana/Nanite "LOD by screen size" principle; choose `want_lod` from the **projected voxel footprint**
   (one voxel ≈ one pixel) rather than the current pure-distance shell test — the Nanite refinement.

### 5.2 Honesty about HW-RT + latency
- **The live structure stays flat.** Demand residency changes *which* bricks fill the pool, not *how* a brick is
  traced — the BLAS + flat `voxel_offset` addressing is unchanged. No SVDAG descent enters the hot path. This is
  the non-negotiable §0 constraint and it is preserved.
- **Latency / holes.** A just-requested brick is not resident the frame it is first needed → a 1-frame hole or a
  fall-back to the coarser parent. Mitigations (all standard): (a) the clipmap **prefetch** keeps the near field
  always-resident so demand only governs the *far/detail* fringe; (b) **fall back to the coarser resident
  ancestor** for a missing fine brick (the clipmap guarantees a coarse shell exists behind every fine one — our
  cede-one-ring overlap already ensures coarse coverage) so a miss is a *blurrier* pixel, never a *black* one;
  (c) keep-old-until-revealed (already in `ResidencyManager`) so the old set stays bound until the new fills.
- **When to do it.** Phase B is only needed if surface-only (Phase A) + the clipmap prefetch still overflow the
  pool at the target view distance. The GPU-worldgen plan correctly lists ray-guided residency as the *optional
  endgame* (its Stage 5). Order: ship Phase A, measure, then Phase B **only if** the surface set still exceeds
  budget at 10× reach.

---

## 6. How this composes with everything else (no rework)

- **Storage plan R1–R3 (per-brick compression).** Orthogonal and complementary. Surface-only reduces the *count*
  of resident bricks; R1/R2/R3 reduce the *bytes per resident brick*. They multiply: a surface set that is mostly
  R2-palette strata bricks + R3-dedup'd repeats, with zero enclosed bricks, is the minimal Tier-A footprint. R1's
  uniform-incl-halo collapse becomes a strict subset of the enclosed cull (§4.2) — unify them on one predicate.
- **GPU-worldgen pivot.** Surface-only is a *filter the GPU voxelizer applies at pool-fill time*: the voxelizer
  knows each brick's content + neighbours, so it can write a **degenerate AABB** for an enclosed brick's slot (or
  not allocate one) — no CPU readback, consistent with the pivot's fixed-cap-pool / degenerate-AABB-for-free-slots
  design. Demand residency *is* the pivot's Stage 5; this doc is its detailed spec. Per-chunk BLAS (pivot Stage 3)
  composes: each chunk BLAS now has O(surface) primitives, so dirty-chunk rebuilds are cheaper too.
- **GI (world cache + ReSTIR + DLSS-RR).** Unchanged. GI rays terminate at first-solid like primary rays, so an
  enclosed brick is invisible to GI as well — culling it cannot change any bounce. Edits ADAPT locally
  (`[[feedback-gi-adapt-not-reset]]`): the promote/demote of a brick is a *local* re-pack, never a reservoir/cache
  clear. (A subtlety to verify, §7: an interior that is *opened* by a dig becomes a new GI surface — covered
  because the edit promotes the exposed brick into the traced set, so the next-frame GI sees it.)
- **`.vxo` on disk (storage-plan R5).** Disk stores the *full* world (incl. interiors, SVDAG-packed for static
  assets); surface-only is a *runtime residency* decision applied after load/generate. Disk = unbounded; resident
  = O(surface). They are the two ends of the Tier-B → Tier-A decode (the Aokana shape).

---

## 7. Phasing, risks, measurement

### Phasing (least-risk / biggest-win first)
- **Phase A — Surface-only GPU residency.** The §4 enclosed-brick cull. Independently shippable; pure filter on
  the packed set; no shader change. **Do this first** — it is the cubic→quadratic win and the prerequisite for a
  far view at bounded VRAM. *Acceptance:* worldgen slice + a solid building render **pixel-identical** to today
  (GPU oracle green) while BLAS primitive count + meta/AABB VRAM drop by the enclosed fraction; a dig that exposes
  an interior brick shows it the next frame (promote test); a place that seals a face drops it (demote test).
  *Sub-steps:* A1 = the `face_full`/enclosed predicate + tests (CPU, on CI); A2 = filter in `pack_resident_set` +
  the storage/residency report numbers; A3 = the edit promote/demote re-pack path + a GPU edit-exposure test.
- **Phase B — Demand / ray-guided residency + LRU pool.** The §5 GigaVoxels model on the GPU pool. **Only if**
  Phase A + clipmap prefetch still overflow at the target reach. Larger, riskier (feedback buffer, LRU, the
  coarse-ancestor fallback). Ship behind a flag, A/B vs. the clipmap-only path. This is the GPU-worldgen pivot's
  Stage 5 — defer until its Stages 2–4 land.
- **Phase C — Screen-error LOD selection (Nanite refinement).** Replace the pure-distance shell test with a
  **projected-voxel-footprint** `want_lod` (one voxel ≈ one pixel), feeding Phase B's request LOD. Optional polish
  once B is live; lets the far view spend its budget where screen-error is largest.

### Top risks (and the mitigation)
1. **Edit-exposure correctness (Phase A).** A dig that exposes an enclosed brick must promote it same-frame. *Mit:*
   the enclosed predicate is a pure function of a brick + its 6 neighbours; the edit-dirty set (owner + neighbours)
   is *exactly* the set whose enclosed-ness can change → re-evaluate only those. A dedicated GPU test digs into a
   solid mass and asserts the revealed interior is traced (no hole). Robust-by-construction: the CPU set always
   has the brick; only its *resident* flag flips.
2. **Shell-boundary false-cull (Phase A).** Culling a brick on incomplete neighbour info → a black crack at a LOD
   seam. *Mit:* cull **only** when all 6 neighbours are present, same-LOD, and full; a missing/different-LOD
   neighbour ⇒ keep (conservative toward keeping). Mirrors the halo's existing AIR-at-boundary fallback.
3. **Ray-guided latency / holes (Phase B).** A just-requested brick is a 1-frame hole. *Mit:* clipmap prefetch
   keeps the near field always-resident; fall back to the coarser resident ancestor (never black);
   keep-old-until-revealed. Bound the request rate per frame (like `max_bricks_per_frame`).
4. **The HW-RT flat-structure invariant.** No deep tree on the hot path. *Mit:* both phases only change *which*
   bricks fill the flat pool; the BLAS + `voxel_offset` DDA is untouched. SVDAG stays Tier-B (disk/asset) only,
   decoded to bricks before tracing. Every reviewer checks this explicitly.
5. **Concave-but-sealed cavities.** A Dense brick with a fully interior air pocket is enclosed (no ray reaches the
   pocket) and *should* be cullable — but only if its 6 outer faces are also sealed by neighbours. *Mit:* the
   §4.2 predicate is over *neighbour* faces, so a sealed pocket inside a brick whose outer faces are exposed keeps
   the brick resident (correct — rays do reach its surface). No special-casing needed; the predicate is total.

### Measurement (extend `tests/voxel_worldgen_perf.rs` — benchmark every delivery)
The rig already reports resident count, pack cost, BLAS build cost, and the R1 `StorageReport`. Extend it with a
**residency-vs-view-distance** sweep so the cubic→quadratic claim is *measured*, not asserted:

| Metric | How | What it proves |
|---|---|---|
| **resident-brick count vs `clip_half`** | sweep `clip_half ∈ {4,8,16,26}`, cold-fill, report `brick_count()` before/after the enclosed filter | the **Θ(H³) → Θ(H²)** growth-rate change (fit the exponent) |
| **enclosed fraction** | `% of resident bricks where enclosed(brick, neighbours)` | Phase A's win predictor (high for terrain/solid buildings) |
| **surface-set VRAM** | meta+AABB+voxel bytes over the *filtered* set | the headline number each phase must reduce |
| **BLAS primitive count + build ms** | the §1 BLAS bench over the filtered set | traversal + build both scale with surface, not volume |
| **max view distance at a fixed VRAM budget** | binary-search the largest `clip_half`/`MAX_LOD` whose filtered VRAM ≤ a budget (e.g. 256 MB) | the deliverable: *how much farther can we see at bounded VRAM* (target ~10×) |
| **edit promote/demote latency** | frames from a dig to the exposed brick being in the packed set | Phase A correctness (must be 1) |
| **(Phase B) request count + pool occupancy + LRU evictions/frame** | feedback-buffer counter, pool high-water, eviction count | demand residency bounds the pool under a pathological all-surface view |

Acceptance gate per phase: the **filtered surface-set VRAM** for the worldgen slice + a solid building drops by
the enclosed fraction; the GPU oracle (`tests/voxel_raytrace_gpu.rs` + the seam/show-through/GI GPU tests) stays
pixel-identical; and the `clip_half` sweep shows the exponent dropping from ~3 toward ~2. Per the standing QA
mandate, each phase: specialist implements → ≥2 adversarial reviewers vs. the GPU ground truth → benchmark gate.

---

## 8. Summary

| Idea | Tier | Effort | Effect on a large view |
|---|---|---|---|
| **Surface-only residency** (cull enclosed bricks from BLAS; keep in CPU brickmap) | A (residency policy) | S–M | **resident set Θ(H³) → Θ(H²)** — the cubic interior term → 0 VRAM; pushes `clip_half`/`MAX_LOD` ~10× at bounded budget; pixel-identical, faster trace + BLAS build |
| **Demand / ray-guided residency + LRU pool** (GigaVoxels on the GPU pool) | A (pool policy) | L | caps the *remaining* surface set; view distance decoupled from VRAM; coarse-ancestor fallback hides latency |
| **Screen-error LOD selection** (Nanite refinement) | A (selection) | M | spend the far-field budget where screen-error is largest |

**How SOTA bounds memory for large scenes (a):** ray stops at the first surface ⇒ **only the visible surface need
be traced** (surface-only / Teardown-class); **ray-guided demand paging + LRU pool** decouples view distance from
VRAM (GigaVoxels); **LOD aggregation** makes coarse bricks cover exponentially more world (Aokana/Nanite); the
big-ratio **SVDAG/SSVDAG/Aokana-chunk** forms are **storage/transport** that is *decoded to a flat traceable pool*
— they bound *disk/stream*, not the *resident* set.

**The ranked path for us (b):** **Phase A surface-only residency** is the dominant, lowest-risk win — it is
*correct by construction* given our first-hit DDA, requires no shader change, unifies with R1 on one
enclosed-predicate SSOT, and changes the resident-brick growth from **cubic to quadratic** in `clip_half`
(~1000× → ~100× for a 10× reach). **Phase B demand/ray-guided residency** caps the remaining surface set with an
LRU pool, mapped onto the GPU-worldgen pivot's fixed-cap pool — deferred to that pivot's Stage 5. **Phase C
screen-error LOD** is optional polish.

**Phasing + top risks (c):** A (ship first) → B (only if A overflows) → C (polish). Top risks: edit-exposure
correctness (mitigated: re-evaluate the already-computed edit-dirty owner+neighbour set), shell-boundary
false-cull (cull only on complete same-LOD neighbour info), ray-guided latency/holes (clipmap prefetch +
coarse-ancestor fallback + keep-old-until-revealed), and the HW-RT flat-structure invariant (both phases change
only *which* bricks fill the flat pool — no tree on the hot path; SVDAG stays Tier-B). Measure the
**resident-count-vs-view-distance** exponent on the extended perf rig to prove the cubic→quadratic change, not
assert it.

---

## 9. References
Aokana ([arXiv 2505.02017](https://arxiv.org/abs/2505.02017), [ACM CGIT 2025](https://dl.acm.org/doi/10.1145/3728299)
— "tens of billions of voxels … up to ninefold less memory … up to 4.8× faster"); GigaVoxels
([Crassin et al., INRIA I3D 2009](https://maverick.inria.fr/Publications/2009/CNLE09/CNLE09.pdf) — ray-guided
streaming, fixed brick pool, LRU, bounded GPU memory); SVDAG ([Kämpe et al. 2013](https://www.researchgate.net/publication/262367808_High_resolution_sparse_voxel_DAGs))
/ SSVDAG ([Villanueva et al., I3D 2016](https://www.crs4.it/vic/data/papers/i3d2016-symmetry-dags.pdf));
Nanite virtualized geometry ([UE5 streaming/LOD-by-screen-size](https://cs418.cs.illinois.edu/website/text/nanite.html));
Teardown voxel RT ([breakdown](https://juandiegomontoya.github.io/teardown_breakdown.html) — first-hit / early-z
culling); dubiousconst282 sparse-64-tree ([guide 2024-10](https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/),
[VoxelRT](https://github.com/dubiousconst282/VoxelRT) — many small trees, brickmap occupancy, instancing).
Our code: `src/voxel/{streaming,gpu,raytrace,brickmap}.rs`, `assets/shaders/voxel_raytrace.wgsl`,
`examples/voxelize_scene.rs` (`solid_fill`), `tests/voxel_worldgen_perf.rs`.
Companion docs: `docs/VOXEL_STORAGE_PLAN.md` (R1–R5 per-brick compression), `docs/GPU_VOXEL_WORLDGEN_PLAN.md`
(GPU-driven streaming + fixed-cap pool; ray-guided residency = its Stage 5).
