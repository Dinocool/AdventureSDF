# Unified All-GPU Voxel Residency / Streaming — Design of Record

> ## ⟶ FINAL TARGET (4+1 directives — these SUPERSEDE/EXTEND the body below)
> The body was written before these directives; where it talks about "routing in-RAM scenes through
> the pager" or preserving a CPU pack, read it through this lens — the target is *simpler* than the body assumes.
>
> 1. **One GPU-driven residency path** for every scene (enumerate → diff → pack → AABB → BLAS, readback-free, fixed-cap pool + GPU page/slot table).
> 2. **`.vxo`-only** — delete `.vox` loading + the legacy full-RAM `.vox` merge + the "no `.vxo` → fall back" branch. Every *file-backed* scene loads `.vxo` (corpus converted in #139; verify a `.vxo` exists for every shipped scene, esp. Sponza, BEFORE deleting `.vox`).
> 3. **No CPU residency/pack pipeline** — delete `ResidencyManager`, `ResidentPacker` (`pack_one`/`update`/`update_gpu`/`apply_delta` + the SlabArena CPU side), the CPU classify, and the StreamSnapshot CPU cold-fill. CPU keeps ONLY: `.vxo` byte IO, command submission, and the one-time pool BUFFER allocation (empty + degenerate AABBs — the GPU cold-fills it).
> 4. **No CPU NEE/light bake** — build the emissive-voxel light list GPU-side from the resident emissive voxels (GPU compaction → light buffer; Solari-style light tiles, GPU-built). NO CPU emitter enumeration / no per-epoch CPU light bake. (Net-new GPU work.)
>
> **+1 — MUST SCALE TO WORLDGEN (design constraint, do NOT build worldgen here):** the same path must serve a full procedural world with every clipmap shell filled, at full perf. Therefore: **(a)** keep the `BrickSource`/producer abstraction clean — `.vxo`-only means file-backed scenes; worldgen is a *procedural* producer (no file) feeding the SAME pool/residency, NOT a special case; the pool brick format is the SSOT both feed. **(b)** Leave room for a **GPU producer** — `.vxo` = disk memcpy, worldgen = GPU generation into pool slots (the approved GPU-worldgen pivot; not built here, but the residency design must not preclude it). **(c)** Be robust under **sustained motion / never-converging residency** — worldgen never settles to idle, so the BLAS-rebuild sweep + `change_count` convergence gating must handle residency changing every frame (the windowed-cursor stale-BLAS hazard the audit flagged must not leave far chunks stale under continuous flight). Two axes are separate: residency throughput (this doc) vs raymarch trace perf (occupancy-bound; out of scope here).
>
> **Sequencing (critical):** the GPU path is the only thing that can be deleted-toward, but it is NOT yet working for all scenes (paged-blank on non-zero offset / the BLAS sweep). The CPU pack is the only currently-rendering path. So: **fix the GPU drive → make it universal (all `.vxo`, lights GPU-side, robust under motion) → ONLY THEN delete `.vox` + the CPU pipeline + the CPU light bake + eager + env gate.** Delete last, never first.

> **Mandate (non-negotiable, from the user):** there must be exactly **ONE** residency/streaming
> path — fully GPU-driven, readback-free, scalable — for **EVERY** scene. NO dual store. Today there
> are two: an **EAGER** in-RAM store (built whole at scene-switch; the only currently-*working* GPU
> path, used for Sponza and the `.vox` Gallery) and a **DEMAND-PAGED** store (for out-of-RAM `.vxo`
> Bistro; exists, parity-tested, but its live drive renders BLANK so it is gated off behind
> `ADVENTURE_GPU_PAGED_DRIVE`, and streamed `.vxo` falls back to a CPU pack). The eager store, the
> CPU-pack fallback, and the env gate must ALL be deleted: small/in-RAM scenes must feed the SAME
> paged GPU pipeline as huge ones, mirroring what others do. No shortcuts, no quality/scope reduction.

> **Status of this doc:** read-only architecture pass over the worktree `voxel-rt` (branch `voxel-rt`).
> Extends and supersedes `docs/PHASE_G_GC_PLAN.md` §8 (G-c.4-paging) for the *unification* question.
> Every file:line cite is this repo unless prefixed `re-flora` (`D:/tmp_test/re-flora`). The §2
> root-cause was validated by *running* the existing paged front-end gate (it PASSES — see §2.4),
> which sharply narrows the blame.

---

## §1 SOTA survey — how others run ONE GPU-driven streaming path

The recurring, canonical structure across every mature out-of-core voxel/virtual-content system is:

> **a fixed-capacity *physical* pool of uniform slots (bricks/pages/tiles) + a GPU-resident *page/slot
> table* (key → slot) + GPU-driven *demand* streaming into that pool, bounded by what the *camera/view*
> needs — not by the total scene size.** A small scene is not a special case: it is simply the case
> where "everything the view needs" ≤ pool capacity, so the demand loop fills the pool once and never
> evicts. There is no second code path for "fits in RAM."

### 1.1 GigaVoxels / GigaSpace (Crassin et al. 2009) — the founding pattern
A dynamic sparse octree of nodes, each node pointing into ONE large GPU **brick pool** (a 3D texture
of fixed M³ bricks). "Usually only a subset of the brick pool can be stored in GPU memory, so the
renderer requests bricks on demand at the appropriate LOD and handles missing data." Bricks are
"loaded or generated only once proven necessary for the image and only at the necessary resolution,
then kept in an LRU cache on the GPU." The crucial unification: the *same* node-pool + brick-pool +
LRU services a 32³ test volume and a billion-voxel dataset; the dataset size only changes how often the
LRU evicts. There is no "small" path. (https://maverick.inria.fr/Publications/2009/CNLE09/CNLE09.pdf,
http://gigavoxels.inria.fr/WhatIsGigaVoxels.html) — our `RegionCache` LRU (`vxo/source.rs:76`) and
`PagedBrickCoreStore` free-list (`residency_gpu.rs:668`) are direct descendants of this brick-cache.

### 1.2 Virtual texturing / Nanite (feedback → page table → fixed physical pool)
The general GPU-driven virtual-content pattern, identical in shape for textures (Sampler Feedback
Streaming), geometry (Nanite), and voxels: the GPU writes a **feedback buffer** of what it *accessed
but was not resident*; a **page table** maps virtual address → physical pool slot (lowest LODs always
resident as a fallback); a **fixed-cap physical pool** that ALL content streams into "just like virtual
memory on a CPU." Nanite explicitly "renders to an array of views in a single pass… different views
can have different LOD priorities and stop at different steps" — one pipeline, many sizes/views, no
fork. (https://trickybitsblog.github.io/2024/04/20/nanite.html,
https://github.com/GameTechDev/SamplerFeedbackStreaming) — note: these systems use *GPU-written
feedback + CPU demand-load*; the readback is a small, amortised, out-of-band request stream, NOT a
per-frame stall. Our design replaces even that feedback readback with a **camera-driven CPU prefetch
of the page set** (§1.5), which is *strictly* less CPU↔GPU coupling.

### 1.3 Teardown / Voxagon (Dennis Gustafsson)
Teardown uses an 8-bit palette (one byte/voxel) and a proprietary engine built around dense small voxel
volumes streamed/instanced into a world. The publicly-documented invariant relevant here: a uniform,
byte-cheap voxel representation packed into bounded GPU volumes that the renderer treats uniformly
regardless of object size (destruction edits the *same* resident volumes the renderer reads). We already
mirror this: 8-bit palette material (`mesh-bake-materials` memory note) + one persistent pool the front
end and renderer share (`residency_front_end.rs:7-10`). (https://blog.voxagon.se/,
https://softwareengineeringdaily.com/2025/01/02/teardown-and-voxel-based-rendering-with-dennis-gustafsson/)

### 1.4 gvox / Aokana — GPU-driven, prefix-sum compaction, readback-free build
Aokana ("A GPU-Driven Voxel Rendering Framework for Open World Games", SVDAG, tens of billions of
voxels) and the gvox engine establish the *build* half of the single path: the **whole** active-set
enumeration, compaction (parallel prefix-sum to turn a sparse marked array into a dense indirection
list), and acceleration-structure build run on the GPU with no per-frame readback. "Compaction can be
parallelized by filling an array marking differences between nodes, then using a parallel prefix sum to
generate indirection lists" — exactly the `atomicAdd`-append + indirect-dispatch our `voxel_residency.wgsl`
Pass B/C/D already does. (https://arxiv.org/html/2505.02017v1,
https://github.com/GabeRundlett/gvox_engine)

### 1.5 re-flora (`tr-nc/re-flora`, in-repo at `D:/tmp_test/re-flora`) — the closest cited reference
re-flora is the reference our `PHASE_G_GC_PLAN.md` was authored against. Its build loop is a clean
single-path example:

- **Camera-first paging, no GPU→CPU request readback.** `ChunkWorkQueue::pop_nearest_to(player_pos, …)`
  (re-flora `src/util/chunk_work_queue.rs:39`, driven from `src/app/core/terrain_rebuild.rs:225-228`)
  pops the chunk *nearest the camera* (with an age bonus so a starved far chunk eventually wins,
  `chunk_work_queue.rs:5,61-74`). The dealloc/alloc signal is **camera-driven on the CPU**, NOT a GPU
  readback of "what did the ray miss" (VoxelNotes leaves GPU-request-readback an open TODO; re-flora
  sidesteps it — and so do we, `PHASE_G_GC_PLAN.md:212-215`).
- **GPU occupancy → instances → accel, readback-free.** `src/builder/surface/mod.rs` runs
  `clear_occupancy` → `instances_to_occupancy` / `edit_occupancy_sphere` → `make_surface` →
  `occupancy_to_instances` (the shaders under `shader/builder/surface/`), then the `contree` build
  (`shader/builder/contree/*.comp`) — all compute, all driven by GPU-written counts + indirect dispatch.
  This is what our `voxel_residency.wgsl` Pass A0/A/B0/B/C/D + `voxel_pack.wgsl` already port.
- **One structure, every size.** The chunk pool and the occupancy/instance buffers are fixed-shape; a
  tiny edited region and a freshly-revealed distant chunk go through the identical
  surface→occupancy→instance→contree pipeline. The chunk *queue* meters work to bound per-frame cost,
  but there is no "this chunk is small, take the other path."

### 1.6 The canonical single-path structure (the synthesis we adopt)
1. **Source** (any `BrickSource` — `.vxo` MergedSource, `.vox`, worldgen) provides, *on demand*, the
   occupancy bits + the 8³ cores for a `(coord,lod)` key. It is read camera-first; nothing is built whole.
2. **GPU occupancy** (the sparse sector-mask hash, `residency_gpu.rs:198` / `voxel_residency.wgsl:164`)
   is the face-cull input — uploaded for the *resident* region set only.
3. **GPU core store** (`(coord,lod) → 8³ core`, `residency_gpu.rs:668`) holds the cores of the resident
   surface shell + 26-halo only — the Θ(H²) footprint, capped (constant-RAM).
4. **GPU front end** (`residency_front_end.rs`) enumerates the clipmap shell, face-culls against (2),
   diffs against a persistent **GPU slot table + free-list** (the page table), enter-caps to pool room,
   packs into the **fixed-cap pool** (the scene's `meta/voxel/brick_palettes/aabb`), and writes
   degenerate AABBs for free slots — all readback-free, indirect-dispatch-gated.
5. **AS build:** full BLAS rebuild of dirty chunks over the fixed-cap pool (no indirect AS on the fork).
6. A **small scene** is the case where the enter-cap never trips and the LRU/free-list never evicts:
   the demand loop fills once, `change_count → 0`, the pipeline idles. *Same path.*

The only piece (4)+(5) lack today, relative to this ideal, is that the *source/occupancy/core feed*
(1)–(3) is provided by **two** different producers: the eager whole-scene build (`raytrace.rs:704-765`)
and the `StreamedResidencyPager` (`residency_pager.rs`). The unification is to make the pager (or a
trivially-resident wrapper) the *sole* producer for all scenes — exactly GigaVoxels' "the small volume
is just the cache that never evicts."

---

## §2 Root-cause of the paged-drive BLANK (the prerequisite blocker)

### 2.1 The prior audit's hypothesis (a world-coord vs clipmap-brick-frame mismatch)
The hypothesis: the pager keys occupancy/cores in **world** coords (`residency_pager.rs:279-290`
`rebuild_occupancy` / `283` `for_each_region_brick_occ`, `from_occupied_full`) while the front end's
enumerate works in a clipmap brick-coord space derived from `cam_world`
(`residency_front_end.rs:93-119` `build_params`; `streaming.rs:126/182` `camera_brick_coord_lod`/
`level_box_pub`); for a `MergedSource` with non-zero `offset_at_lod` every `is_occupied` probe would
miss → face-cull finds nothing → enter-everything → all-origin/degenerate AABBs → blank.

### 2.2 Verdict on the literal hypothesis: **REFUTED at the occupancy probe level — both sides are WORLD coords.**
Tracing the exact frames:

- **Front end enumerate is in WORLD brick coords.** `build_params` computes each LOD's `cell_lo`/
  `cam_brick_coord` from `level_box_pub(cam, lod, half)` and `camera_brick_coord_lod(cam, lod)` —
  pure functions of `cam_world` (`residency_front_end.rs:97-103,110-111`). In WGSL,
  `enumerate_shells` reconstructs `coord = key.cell + (lx,ly,lz)` (`voxel_residency.wgsl:444`) and probes
  `is_occupied(coord, lod)` (`:454`) — i.e. with that **world** brick coord.
- **Pager occupancy is ALSO in WORLD brick coords.** `rebuild_occupancy` iterates each decoded region
  via `for_each_region_brick_occ` (`residency_pager.rs:283`), which yields `local + offset_at_lod(lod)`
  = **world** coords (`vxo/source.rs:595-601`), and feeds those straight into
  `SectorOccupancy::from_occupied_full` (`residency_pager.rs:287`). So the sector hash is keyed by the
  *same world coord* the front end probes with.

Therefore `is_occupied(world_coord)` on the pager-built occupancy is *frame-consistent* with the front
end's enumerate. The pager only uses `offset_at_lod` *internally* — to map the world clipmap AABB into
each asset's local directory for `present_regions_in` (`residency_pager.rs:188-192`) and to translate
local entry coords back to world before storing (`source.rs:596`). The world↔local round-trip is closed
*before* anything reaches the GPU. **The "occupancy keyed in local while front end probes world" failure
mode does not exist in the current code.**

### 2.3 The decisive experiment: the existing paged front-end gate PASSES
`tests/voxel_paged_front_end_render.rs` drives the **production** `GpuResidencyFrontEnd` over the
**production** `StreamedResidencyPager` **exactly as `drive_gpu_residency_front_end` does** (poll mirror
→ `update` pager → rebind → `record_frame` → `advance_ring`, lines 262-281), to convergence, then
asserts the paged resident brick **set equals the eager set** (`:301-304`) with **zero origin-collapsed
AABBs** (`:296`). Running it on the worktree (NVIDIA RTX 4090, Vulkan):

```
[eager]  converged in 32 frames — 676 live AABBs (0 origin-collapsed)
[paged]  converged in 32 frames — 676 live AABBs (0 origin-collapsed, 15708 degenerate); eager had 676
[diff] only in eager: []   only in paged: []
[scale] probed 1000000 keys: 0 false-positive, 0 false-negative
test result: ok. 2 passed
```

So the pager + front-end *algorithm* converges correctly and produces the identical pool to the eager
path — **the over-enumerate-to-origin-AABB mechanism the audit feared is NOT firing in the tested
configuration.** The blank therefore lives in *what the test does NOT exercise* relative to the live
`drive_gpu_residency_front_end`. There are exactly three such gaps, in priority order:

### 2.4 The actual gaps (what differs between the passing test and the live blank)

**(A) Non-zero per-asset `offset_at_lod` is NEVER exercised — the most likely real culprit.**
Both render scenes in the gate build a `MergedSource` with a SINGLE asset at **`IVec3::ZERO`**
(`voxel_paged_front_end_render.rs:252,395`). But the live multi-scene Gallery places each `.vxo` at a
**cumulative +X brick offset** (`gallery.rs:vxo_gallery_placements:196-203`; even the *first* asset gets
`offset.x = x_cursor - lo_x`, non-zero whenever its baked `lo_x ≠ 0`). The Bistro *bench* harness
(`ADVENTURE_BENCH_BISTRO`) places at `IVec3::ZERO` (`gallery.rs:213-215`) — so *if* the blank reproduces
under the bench harness, offset is exonerated and gap (C) is the cause; if it reproduces only in the
multi-scene Gallery, offset is the cause. **This is the single most important thing to disambiguate
first** and it is NOT determinable read-only (needs a launch or a GPU capture of each configuration).

   Why offset is the prime suspect despite §2.2: §2.2 proves the occupancy *probe* is frame-consistent.
   But the offset path has three independent consumers that must ALL agree, and only the occupancy round
   trip is gate-covered:
   - `desired_regions` maps world→local via `asset.source.offset_at_lod_pub(lod)`
     (`residency_pager.rs:188`) and clips to `asset_lod_bounds` (`:182`).
   - `flush_cores` re-derives the owning region by `world - offset_at_lod_pub(lod)` then
     `div_euclid(k)` (`residency_pager.rs:402-405`) — an independent inverse of the same transform; a
     sign/rounding disagreement with `present_regions_in`'s `div_euclid` padding (`source.rs:564-567`)
     for a non-zero, possibly-negative offset would page a core for the *wrong* region → `core_at_world`
     returns `None` (`source.rs:623-626`) → the brick enters with an absent core → packs degenerate.
   - The eager path has NO such transform (see (B)), so it is structurally immune.

   A focused parity assertion at non-zero offset (gap-fix in §3 Stage 1) is the robust way to settle
   this without a GPU; the current gate's `IVec3::ZERO` is exactly why the bug slipped.

**(B) The eager path's structural immunity (why it never blanks) — and what the unified path must
inherit.** The eager in-RAM Gallery merges every asset into ONE `BrickMap` at *baked-in world coords*
(`gallery.rs:merge_brickmap_into` writes shifted coords directly, `:441-444`), and the eager occupancy/
core build runs `StaticVoxSource::occupied_keys_full()` / `occupied_keys()` over that single map
(`raytrace.rs:717,735`). There is **no per-asset, per-LOD offset transform at query time** — the world
coord IS the stored key, at every LOD, with no `div_euclid(2^lod)` offset round-trip. So the eager path
cannot hit gap (A) by construction. **The unified path must preserve this property:** the offset
transform must be applied *once*, consistently, with a single SSOT for world↔local that all of
`desired_regions`, `present_regions_in`, occupancy build, and core fetch share — see §3 Stage 1's
"single offset SSOT" requirement.

**(C) The live BLAS/TLAS rebuild path is NEVER exercised by the gate.** The render gate reads back the
`meta`/`aabb` pool and stops (`voxel_paged_front_end_render.rs:286-311`); it builds **no BLAS, no TLAS,
and never ray-traces.** The live driver, after the front end writes the pool, does a *windowed,
full-rebuild* BLAS sweep over a persistent cursor + a TLAS-over-all build (`raytrace.rs:3801-3877`),
gated by the 1-frame-late `change_count` mirror. This path has its own documented hazards that a correct
pool can still trip into a blank trace:
   - **`rebuild_as` may never fire.** The mirror returns `Some(0)` spuriously on the first bound frames
     (the staging ring starts zeroed — noted in the gate itself, `:189-190`). The live driver maps
     `prev_change == Some(0) → rebuild_as = false` after the very first `None`
     (`raytrace.rs:3775-3780`). If the front end is still cold-filling while the mirror reads a stale 0,
     the windowed sweep can *stop advancing the cursor* before the pool is fully streamed → a partially-
     built TLAS → large blank regions. The gate sidesteps this entirely (it never builds AS and drives a
     fixed 32 frames).
   - **Whole-set build overrun.** The comment at `raytrace.rs:3811-3818` documents that rebuilding all
     chunk BLASes at once silently yields a non-tracing TLAS past ~60-100k live prims (clip_half-
     dependent blank, no validation error). The windowed sweep (`BLAS_REBUILD_WINDOW=48`) is the
     mitigation — but it is unverified for the *streamed* convergence profile (the cursor resets to 0 on
     every `change_count>0`, `:3823-3824`; if a streamed scene never fully converges, the sweep restarts
     forever and may never complete a full TLAS).
   - **Refit corruption** (`voxel-rt-blas-refit-corruption` memory) is avoided by `create_chunk_blas`
     recreation (`:3838`), so that specific trap is handled — but only confirms the sweep is *built*
     correctly, not that it ever *completes* under streaming.

### 2.5 Precise conclusion + the fix
- **Refute** the literal world-vs-local occupancy-frame hypothesis (§2.2): both probe and store are
  world coords; the gate confirms convergence parity at zero offset (§2.3).
- **The real blocker is one (or both) of:** (A) a non-zero `offset_at_lod` inconsistency among the
  pager's three independent offset consumers (occupancy/desired/core-fetch), which the gate's
  `IVec3::ZERO` scenes hide; and (C) the **live BLAS/TLAS sweep** (mirror-gating + windowed-rebuild
  convergence) which the gate does not exercise at all.
- **The fix is staged in §3:** Stage 1 adds a *live-enumerate* parity gate at **non-zero offset**
  (settling A) AND a BLAS-inclusive headless render-to-trace gate (settling C), then fixes whichever
  fails. **Do not delete the eager path until both gates are green driving every scene** (§3 sequencing).
- **Read-only flags (need a launch / GPU capture, cannot determine here):** which of (A)/(C) fires in
  the live Bistro bench vs. the multi-scene Gallery; whether `rebuild_as` stalls mid-cold-fill; the
  actual live `clip_half` at which the TLAS overrun bites for the streamed pool.

---

## §3 The unified-path migration plan (staged, QA-gated, safe sequencing)

### 3.0 The ONE path, concretely (data flow, all readback-free, into a fixed-cap pool)
```
                       (CPU, camera-driven, per region-crossing only — re-flora pop_nearest pattern)
 source: BrickSource  ─► ResidencyProducer ──────────────────────────────────────────────────┐
 (.vxo MergedSource,     • desired region/brick set from level_box_pub(cam,lod) ∩ asset bounds │
  .vox, worldgen)        • decode newly-covered, drop uncovered (region LRU bounds RAM)         │
                         • rebuild GPU occupancy (sector-mask hash) for resident set            │
                         • incrementally page GPU core store (surface+26-halo, capped)          ▼
                                                                              GPU occupancy + GPU cores
 (GPU, per frame, one encoder, readback-free — voxel_residency.wgsl)                            │
   Pass A0/A/A2  clear counts, seed indirect dispatch, drain quarantine, clear hashes           │
   Pass B0/B     enumerate clipmap shell, coarse occ test, 6-face cull ◄──────────────── occupancy
   Pass C        diff vs persistent GPU slot table + free-list; enter-cap to pool room          │
   Pass D        build pack/aabb/classify commands; 26-halo neighbour table ◄───────────── cores
   classify/pack/write_aabb (INDIRECT, self-gating) ─► fixed-cap POOL (meta/voxel/palette/aabb) │
   write_change_count ─► 1-frame-late staging-ring mirror ─────────────────────────────────────┘
 (GPU, conditional on change_count>0) full-rebuild dirty-chunk BLAS over fixed-cap pool + TLAS-over-all
```
"Eager" becomes "the LRU/free-list never evicts and the enter-cap never trips" — no branch.

### 3.1 The `ResidencyProducer` abstraction (the unification primitive)
Introduce ONE trait the front end is driven by, replacing the `have_eager`/`have_paged` fork:
```rust
trait ResidencyProducer {
    fn update(&mut self, queue, cam_world) -> bool;        // page in/out; true on a crossing
    fn occupancy(&self) -> &GpuResidencyBuffers;
    fn core_buffers(&self) -> GpuBrickCoreBuffers;
    fn take_needs_rebind(&mut self) -> bool;
}
```
`StreamedResidencyPager` already has exactly this shape (`residency_pager.rs:137-158,206`). The
unification is: **every scene gets a `StreamedResidencyPager` driving a `MergedSource`.** In-RAM scenes
become a `MergedSource` over their (already in-RAM) `.vox`/worldgen data exposed as a trivially-resident
`BrickSource` — see Stage 2 for HOW. There is then no eager store and no `have_eager` branch.

### Stage 1 — Fix the §2 blank + add the gates that would have caught it
**What changes:** (i) Establish a **single world↔local offset SSOT** used by *all* pager consumers:
`desired_regions` (`residency_pager.rs:188`), `present_regions_in` pad/bucket (`source.rs:564-567`),
`for_each_region_brick_occ` (`source.rs:596`), and `flush_cores`' inverse (`residency_pager.rs:402-405`)
must call ONE function (e.g. `offset_at_lod_pub`) and ONE `div_euclid` bucketing helper — so a non-zero
or negative offset cannot disagree between them (robust-by-construction; the recurring offset bug class
becomes structurally impossible). (ii) Fix whichever of §2.4 (A)/(C) the new gates expose.

**What's deleted:** nothing yet (the eager path stays as the reference oracle for the new gates).

**Tests/gates (the missing coverage that let the bug slip):**
- **Gate 1a — paged-ENUMERATE parity at NON-ZERO offset.** Extend `voxel_paged_front_end_render.rs` (or
  a sibling) to build the `MergedSource` with a non-zero, and a *negative*, asset offset (and ≥2 assets,
  mirroring the live Gallery), then assert the **live front-end's enumerated candidate/desired set**
  (read back `candidate_list`/`desired_list` after `record_frame`) equals `desired_clipmap_surface` over
  the same `MergedSource` (`streaming.rs:410`). The *existing* parity gates validate **store contents**
  (`voxel_paged_source_parity.rs`) and the **converged pool set** at zero offset
  (`voxel_paged_front_end_render.rs`) — neither asserts the **live enumerate** at non-zero offset; **that
  is exactly why the bug slipped.** This gate settles §2.4(A).
- **Gate 1b — BLAS-inclusive headless render-to-trace.** A headless gate that drives the front end AND
  the live windowed BLAS/TLAS sweep (`raytrace.rs:3801-3877`) to convergence, then traces a handful of
  primary rays (or reads a tiny render target) and asserts non-blank coverage matching the eager path.
  This settles §2.4(C) — the gate that the pool-only render gate cannot. (Mine `tests/common/` for the
  existing GPU/AS scaffolding; the `voxel_gpu_residency_converge.rs` rig is the closest starting point.)
- **Gate 1c — mirror/cold-fill convergence.** Assert the `change_count` mirror does not gate `rebuild_as`
  off *before* the front end has finished cold-filling (drive a streamed scene from cold; assert the BLAS
  sweep cursor completes a full pass after `change_count → 0`).

**Risk:** the live BLAS overrun (§2.4(C)) may need the GPU `chunk_dirty_mask` (PHASE_G_GC_PLAN §4 / G-c.4+)
to bound per-frame build cost for the streamed convergence profile; if Gate 1b shows the windowed sweep
never completes under streaming, promote that optimisation into this stage. **Respect the fork
constraint: NO indirect AS build** — keep the fixed-cap pool with degenerate AABBs for free slots + full
BLAS rebuild of dirty chunks (`raytrace.rs:3838` recreation), never an indirect AS build.

### Stage 2 — Route in-RAM scenes (Sponza, `.vox`) through the SAME paged pipeline
**The HOW (the decision the mandate asks for):** the pager must drive **any `BrickSource` uniformly** —
do NOT write a `.vox`-specific paged wrapper. Two viable shapes; **adopt (b):**

- (a) *Paged-source wrapper:* wrap the in-RAM `BrickMap`/`StaticVoxSource` in a `BrickSource` that
  exposes a single all-resident "region" (the whole map) so `present_regions_in`/`decode_region_pub`/
  `for_each_region_brick_occ` work unchanged. Simple, but bolts a fake region directory onto in-RAM data.
- **(b) *Generalize the producer over `BrickSource` directly* (preferred, robust-by-construction):**
  the pager already only needs, per source: (1) a present-region enumerator over a clipmap AABB,
  (2) per-region brick occupancy/full bits, (3) per-key core fetch. `StaticVoxSource` can implement the
  same three accessors `VxoSource` exposes (`present_regions_in`/`for_each_region_brick_occ`/
  `core_at_world` — or their `OccupancyOracle`/`occupied_keys_full` equivalents it already has,
  `residency_gpu.rs:168-190`). Then `StreamedResidencyPager::new` takes `Arc<dyn BrickSource + …>` (or a
  small `ResidencySource` trait), and Sponza/`.vox` flow through `update` → occupancy rebuild → core
  paging identically. For an in-RAM source the "region LRU" simply never evicts and occupancy rebuild is
  the whole (small) map — i.e. *the GigaVoxels "small volume = cache that never evicts" reduction*, §1.6.

Either way, the in-RAM data stays in RAM (no re-bake, no `.vxo` round-trip); it is just *fed through the
same producer → front end → pool* pipeline. The front end is already store-agnostic (`rebind_pool` takes
the same `GpuResidencyBuffers`/`GpuBrickCoreBuffers` shape for both, `residency_front_end.rs:612-661`).

**What changes:** `drive_gpu_residency_front_end` builds a `StreamedResidencyPager` for *every*
`gpu_residency`-eligible scene (in-RAM included), not just streamed `.vxo`; the per-frame loop is the
single `update → (rebind if needed) → record_frame → BLAS sweep` block.

**What's deleted (in Stage 3, after this is green):** see Stage 3.

**Tests/gates:** Sponza + the `.vox` Gallery, driven through the paged producer, must produce a pool
**byte-/set-identical** to the current eager front-end path (extend `voxel_sponza_residency.rs` /
`voxel_gallery_residency.rs`), and render non-blank (Gate 1b extended to these scenes). Static-hold
converges to idle. Bench: Sponza `max_frame_ms` must NOT regress vs the eager path (it is the same front
end + a now-trivial producer; the only new cost is the first occupancy rebuild, which for a small map is
sub-ms).

**Risk:** the in-RAM occupancy rebuild on the producer's first `update` must be a one-time cost (the LRU
never evicts → no per-crossing rebuild for a static in-RAM scene). Confirm via the bench (no per-frame
`rebuild_occupancy` for a held camera).

### Stage 3 — Delete the eager store, the CPU-pack fallback, the env gate
**What's deleted (file-level — see §4):** the eager `gpu_residency`/`gpu_core_store` build
(`raytrace.rs:704-765`), the CPU-pack fallback arm (`raytrace.rs:3265-3271` + the
`front_end_will_drive`/`front_end_drives` gating ~871/969/992), the `have_eager` branch +
`gpu_residency`/`gpu_core_store` resource fields + their epoch fields, the `ADVENTURE_GPU_PAGED_DRIVE`
env gate (`raytrace.rs:887,3622,3663`), and the `paged_drive_enabled`/`have_paged` naming (it is now
*the* path). `would_overflow`'s "for the in-RAM scenes this never trips" comment
(`residency_front_end.rs:670-676`) becomes the universal guard.

**Sequencing — CRITICAL:** this stage runs ONLY after Stage 1 (blank fixed) AND Stage 2 (in-RAM scenes
green through the paged path) are merged. Until then the eager path is the working reference oracle and
the user-facing correctness guarantee. Deleting it earlier would leave NO working GPU path if Stage 1/2
regressed.

**Tests/gates:** all suites green; `gpu_residency` toggle-OFF still byte-unchanged (the CPU
`ResidencyManager` path is retained as the toggle-off / non-RT fallback — the mandate deletes the GPU
*eager store* and the *CPU-pack-as-GPU-fallback*, not the CPU residency that backs toggle-off). Confirm
no remaining reference to `gpu_residency`/`gpu_core_store` resource fields or `ADVENTURE_GPU_PAGED_DRIVE`.

### Stage 4 — Verify scale + perf (the 165-FPS goal; no Sponza regression)
**What changes:** nothing structural — this is the measurement gate.
**Tests/gates:**
- **Sponza no-regression:** `max_frame_ms` and steady-state FPS within noise of the pre-unification eager
  path (the `voxel-rt-165fps-bistro` config is the SOTA reference).
- **Bistro streams on GPU (the headline):** `ADVENTURE_BENCH_BISTRO=1`, moving fly-through, clip_half at
  the shipping value — no 317 ms classify freeze, converges to idle on a static hold, **correct COMPLETE
  screenshot (no holes = the invariant proof)**, ceiling FPS toward the 165-FPS target.
- **Constant-RAM:** bounded resident region count over a fly-through (the pager's
  `resident_region_count`/`resident_core_count` diagnostics, `residency_pager.rs:162,169`).
- **Per-stage QA:** specialist → ≥2 adversarial reviewers (validating against re-flora / GigaVoxels /
  Aokana, not just our own tests) → parity + perf + 3 zero-warning builds, per
  `feedback-agent-team-qa-per-stage` + `PHASE_G_GALLERY_PLAN.md:69`.

---

## §4 What gets deleted / unified (file-level)

`src/voxel/raytrace.rs`:
- **Eager store build (~704-765):** the `SectorOccupancy::from_occupied_full(src.occupied_keys_full())`
  occupancy build (`:716-725`) and the `BrickCoreStore::from_cores(...)` core build (`:732-758`) — both
  collapse into the `ResidencyProducer`/`StreamedResidencyPager` over a `BrickSource` (Stage 2). The
  `sponza_source`/`gallery_source` `StaticVoxSource` fields (`:691-703`) feed the producer instead of an
  eager store.
- **`have_eager`/`have_paged` branching in `drive_gpu_residency_front_end` (3651-3664, 3731-3757):** the
  whole `if have_paged { pager } else { eager }` store-selection (`:3731-3757`) collapses to a single
  `let producer = resources.producer.as_…` — the front end binds the producer's `occupancy()`/
  `core_buffers()` unconditionally. `have_eager` and its 4-field epoch guard (`:3651-3654`) are deleted.
- **CPU-pack fallback (3258-3271):** the `front_end_active` check that, when the front end declined,
  falls back to `apply_gpu_pack(...)` for streamed scenes (`:3265-3271` + the
  `front_end_will_drive`/`front_end_drives` plumbing at `:874-888`, `:981-992`) is deleted — the producer
  always drives, so there is no "front end declined a streamed scene" case. (`apply_gpu_pack` itself,
  `:3965`, is retained only insofar as the front end reuses its pack passes; the *CPU-command-built*
  invocation goes away.)
- **`ADVENTURE_GPU_PAGED_DRIVE` env gate (887, 3622, 3663):** deleted; `paged_drive_enabled`/the
  `known-failing` comment block (`:3655-3664`) removed.
- **`VoxelRtResources` fields:** `gpu_residency` + `gpu_residency_epoch`, `gpu_core_store` +
  `gpu_core_store_epoch` (the eager stores) are removed; `streamed_pager` (`:1531`) generalises to the
  single `producer`. The `streamed_source` upload field (`:774`, set only for `Gallery`) generalises to
  carry the producer's source for every scene.

`src/voxel/residency_gpu.rs`:
- The immutable `BrickCoreStore` (`:553-629`) is used only by the eager build + the gates' oracle; once
  Stage 3 lands it is retained ONLY as a test oracle (or deleted if `PagedBrickCoreStore` + the producer
  cover the gate needs). `PagedBrickCoreStore` (`:668`) is the live store for all scenes.

`src/voxel/residency_pager.rs`:
- Generalised from "streamed `.vxo` only" to drive any `BrickSource` (Stage 2); the doc comment's
  "streamed Bistro" framing updates to "every scene's residency producer."

`docs/PHASE_G_GC_PLAN.md` §8.4: the line "REMOVE the `gpu_residency && gpu_core_store.is_some()`
fallback → the prefetcher provides the stores for streamed scenes" (`:250-251`) is *extended* by this doc
to "the producer provides the stores for **every** scene; delete the eager store too."

**Net result:** one producer (`StreamedResidencyPager` over a `BrickSource`), one front end
(`GpuResidencyFrontEnd`), one fixed-cap pool, one BLAS sweep — no `have_eager`/`have_paged`, no env gate,
no CPU-pack fallback. A small scene is the producer whose LRU never evicts and whose front-end enter-cap
never trips: the SOTA "small = the cache that never evicts" single path (§1.6).

---

## Appendix — read-only items flagged for a launch / GPU capture
1. **Which gap fires the live blank (§2.4):** run the Bistro bench (`ADVENTURE_BENCH_BISTRO=1`, offset 0)
   with `ADVENTURE_GPU_PAGED_DRIVE=1` + `ADVENTURE_PAGED_DIAG=1` (`raytrace.rs:3882`) — if it blanks at
   offset 0, the cause is (C) BLAS/TLAS, not (A) offset. Then run the multi-scene Gallery (non-zero
   offsets) — if THAT blanks while the bench does not, (A) offset is confirmed.
2. **Mirror cold-fill stall (§2.4(C)):** capture whether `rebuild_as` gates off (`raytrace.rs:3780`)
   before the BLAS sweep cursor completes a full pass during cold-fill.
3. **TLAS overrun threshold for the streamed pool:** the live `clip_half` at which the windowed sweep
   still yields a non-tracing TLAS (the `:3811-3818` overrun) for streamed convergence — needs a GPU
   capture; may force the GPU `chunk_dirty_mask` (G-c.4+) into Stage 1.

### Sources
- GigaVoxels (Crassin et al. 2009): https://maverick.inria.fr/Publications/2009/CNLE09/CNLE09.pdf ,
  http://gigavoxels.inria.fr/WhatIsGigaVoxels.html
- Nanite deep dive (virtual geometry, page table, single-pass multi-view):
  https://trickybitsblog.github.io/2024/04/20/nanite.html
- Sampler Feedback Streaming (feedback buffer + fixed physical pool):
  https://github.com/GameTechDev/SamplerFeedbackStreaming
- Teardown / Voxagon (palette voxels, uniform volumes): https://blog.voxagon.se/ ,
  https://softwareengineeringdaily.com/2025/01/02/teardown-and-voxel-based-rendering-with-dennis-gustafsson/
- Aokana GPU-driven voxel framework (prefix-sum compaction, readback-free build):
  https://arxiv.org/html/2505.02017v1
- gvox engine: https://github.com/GabeRundlett/gvox_engine
- re-flora (in-repo `D:/tmp_test/re-flora`): `src/util/chunk_work_queue.rs:39` (`pop_nearest_to`),
  `src/app/core/terrain_rebuild.rs:225-228`, `src/builder/surface/mod.rs`, `shader/builder/surface/*.comp`,
  `shader/builder/contree/*.comp`
