# Voxel-RT — SOTA Reference (consolidated prior-art)

A single, scannable reference for the state-of-the-art techniques this project has surveyed across its design
docs. **One place to consult before designing the next subsystem** — so we don't re-survey the same papers.

This doc *gathers and organizes*; it does not re-derive. Each design doc remains the authority for its own
plan. Here, per item, you get:

- **What it is** — one line.
- **Key numbers / claims** — the headline figures (paraphrased where noted).
- **Tier** — the load-bearing distinction for this engine:
  - **Tier A** = GPU-DDA-traceable, *live* in the buffers the `ray_query` reads. Decode ≤ a couple of ALU ops
    + one fetch per voxel step. No pointer chasing on the hot path.
  - **Tier B** = on-disk / compression / transport. Aggressively packed (DAG, zstd, RLE), **decoded to a
    Tier-A layout before the trace ever touches it.**
  - Confusing the two is the classic mistake — a full SVDAG is a great Tier-B form and a terrible Tier-A one.
- **Decision** — ADOPTED / REJECTED / DEFERRED + the concrete reason, with the engine constraint that drove it.
- **Source** — link preserved.
- **Used by** — cross-link to the design doc that specs it.

**The one hard constraint that classifies everything (state it, design around it):** the live render path is
hardware ray tracing. A ray hits a brick's procedural AABB (a BLAS primitive), then an in-shader 3D-DDA
(`dda_brick` in `assets/shaders/voxel_raytrace.wgsl`) does, per cell it steps:
`id = voxels[meta.voxel_offset + cell_index(x,y,z)]` — an O(1) flat fetch. **Any VRAM-resident format must answer
"block id at local (x,y,z)" in ≈O(1) from a storage buffer, in the hot per-step loop, with no pointer chasing and
no per-ray whole-brick decompression.** That line is what makes a method Tier A or Tier B.

Engine baseline being optimized: brick = `8³` voxels @ `0.2 m` LOD0 (`VOXEL_SIZE`, `brickmap.rs`), world span
scales `2^L` with the clipmap LOD (`MAX_LOD = 7`); per resident brick today ≈ **4064 B** (haloed `10³` `u32`
voxels = 4000 B + 32 B meta + 32 B AABB); ~60k-brick resident cap ≈ **~240 MB** voxel buffer.

---

## Contents

1. [Storage / Compression](#1-storage--compression)
2. [Global Illumination](#2-global-illumination)
3. [Streaming / Residency / LOD](#3-streaming--residency--lod)
4. [Acceleration Structure / Instancing](#4-acceleration-structure--instancing)
5. [DDA / Traversal Primitives](#5-dda--traversal-primitives)
6. [Scene Sources (voxelization corpus)](#6-scene-sources-voxelization-corpus)
7. [Local checkouts & saved references](#7-local-checkouts--saved-references)
8. [Quick decision table](#8-quick-decision-table)

---

## 1. Storage / Compression

> Authority: [`docs/VOXEL_STORAGE_PLAN.md`](VOXEL_STORAGE_PLAN.md) (§2 survey table, §5 what-NOT-to-adopt).
> The methods that **stay Tier A** are the *brick-local* ones (uniform collapse, per-brick palette + bit-pack,
> occupancy+compaction, brick dedup — the GigaVoxels/Teardown/BrickMap family). The headline-ratio methods
> (SVDAG/SSVDAG/Aokana-chunk) are **tree/DAG** forms — Tier B transport only, decoded to bricks before the trace.

### 1.1 Uniform-brick collapse  — **Tier A · ADOPTED (R1, LANDED)**
- **What:** an all-same brick stores *one* block id (+ a flag) instead of 1000 identical `u32`s.
- **Numbers:** uniform interior brick **4000 B → ~8 B** (≈500×) per brick; *for that brick* 512–1000×. DDA reads
  the single id, **skips the per-cell fetch** ⇒ uniform bricks trace *faster*, not slower.
- **Caveat (built in):** only collapse when **all 6 neighbours are solid** (a fully-buried brick has no
  air-exposed face, so it needs no stored halo) — "uniform-incl-halo." This makes collapse exactly the
  fully-buried case, robust by construction.
- **Decision:** **ADOPTED, landed.** The single highest value/effort change; pure VRAM/packing, DDA math unmoved.
- **Source:** Teardown palette idea ([acko.net](https://acko.net/blog/teardown-frame-teardown/)); ours.
- **Used by:** VOXEL_STORAGE_PLAN R1; subset-of by VOXEL_LARGE_SCENE surface-only (§3.3 below).

### 1.2 Per-brick palette + bit-packed indices  — **Tier A · ADOPTED (R2, NEXT / S1)**
- **What:** a brick with ≤k distinct ids stores a tiny palette + `ceil(log2 k)`-bit index per voxel.
- **Numbers:** strata (≤4 ids) **4000 B → ~258 B** (15.5×); ≤16 ids → ~532 B (7.5×); 32-id surface → ~690 B
  (5.8×). DDA cost: 1–2 fetches + shift + mask + 1 palette indirection per step — handful of ALU ops.
- **Decision:** **ADOPTED** (the strata win). This is the proven Teardown / `voxel.wiki` palette-compression path,
  shipped at scale in production GPU voxel renderers. Most *invasive* shader change (hottest loop) — gate behind
  R1, A/B-measure per-step cost. Shared as the `.vxo` BRIK on-disk format (disk↔VRAM transcode-free).
- **Source:** [voxel.wiki palette-compression](https://voxel.wiki/wiki/palette-compression/),
  [Teardown/acko.net](https://acko.net/blog/teardown-frame-teardown/), [voxagon](https://blog.voxagon.se/),
  MagicaVoxel palette.
- **Used by:** VOXEL_STORAGE_PLAN R2; VOXEL_FINE_RESOLUTION S1 (prerequisite for 0.05 m VRAM).

### 1.3 Occupancy mask + compacted solid list  — **Tier A · ADOPTED (R4, later phase)**
- **What:** store a `512`-bit occupancy mask (already CPU-side) + only the ids of *solid* cells; popcount-prefix
  gives the dense index. Air cells cost 1 bit, not a full id.
- **Numbers:** ~2–8× (only solids carry an id); helps *surface* bricks (lots of air), not interiors.
- **Decision:** **ADOPTED but lowest priority** — only matters once R1+R3 make interiors ~free and the surface
  shell is the budget. Slightly more ALU than R2. (The BrickMap / sparse-64-tree leaf trick.)
- **Source:** [stijnherfst/BrickMap](https://github.com/stijnherfst/BrickMap); sparse-64-tree (1.5 below).
- **Used by:** VOXEL_STORAGE_PLAN R4.

### 1.4 Brick-level dedup (1-level DAG at brick granularity)  — **Tier A · ADOPTED (R3, LANDED)**
- **What:** content-hash each packed brick's voxel slice; intern identical slices; many metas share one
  `voxel_offset`. **Dedup the leaf payload, keep the spatial index flat.**
- **Numbers:** identical interior bricks → one shared slice (∞ per duplicate); interior fill → amortized ~0.
- **Why Tier A:** the DDA addresses voxels purely through `meta.voxel_offset` — pointing two metas at the same
  offset is **invisible to the shader, zero shader change.** A cut into a shared brick **copy-on-write** forks it
  (same COW the instancing plan uses).
- **Decision:** **ADOPTED, landed** (`719fab4`). This is the *cheap, safe slice of the DAG idea* — voxel-level
  DAG would break Tier A; brick-level dedup keeps it.
- **Source:** SVDAG dedup idea ([Kämpe](https://www.researchgate.net/publication/262367808_High_resolution_sparse_voxel_DAGs)),
  applied at brick granularity; dubiousconst282 instancing.
- **Used by:** VOXEL_STORAGE_PLAN R3; instancing COW (VOXEL_INSTANCING §2.3).

### 1.5 Sparse 64-tree (`tree64` / dubiousconst282, used by re-flora)  — **Tier A (shallow) / B · PARTIAL**
- **What:** `4³` nodes, 64-bit child+leaf masks, omit empty; "many small trees" over one monolith; optional leaf
  palette + tile hashing; brick occupancy masks for two-level DDA space-skipping.
- **Numbers:** 2–10× + good empty-space skipping.
- **Decision:** **REJECTED as the per-voxel store** (the *tree* is pointer-ish; its leaves still want flat bricks
  for the DDA, and the BLAS already does the spatial-index / empty-skip job). **ADOPT its lessons:** chunk the
  world (we do — clipmap cells), keep flat brick leaves, instance repeats (= our R3). re-flora confirmation: it's
  *software* 64-tree, 1-bounce GI, no LOD — we're ahead; transferable = its readback-free GPU build pipeline.
- **Source:** [dubiousconst282 guide 2024-10](https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/),
  [tree64](https://github.com/expenses/tree64), [VoxelRT](https://github.com/dubiousconst282/VoxelRT).
- **Used by:** GPU_VOXEL_WORLDGEN (chunking); VOXEL_STORAGE_PLAN §5 (rejected as live store).

### 1.6 GigaVoxels brick pool  — **Tier A · ADOPTED (the GPU pool shape)**
- **What:** fixed-size GPU pool of `N³` bricks + an octree of pointers into it; ray-guided paging; LRU eviction.
- **Numbers:** bounded VRAM regardless of view distance ("walk through very large detailed worlds in bounded GPU
  memory").
- **Decision:** **ADOPTED as the fixed-capacity GPU brick pool** (GPU-worldgen pivot). We keep our *flat* index
  (clipmap + BLAS) instead of GigaVoxels' SVO descent on the hot path; we borrow the **pool + ray-guided request +
  LRU** model. (Streaming/residency detail in §3.2 below.)
- **Source:** [Crassin et al., INRIA I3D 2009](https://maverick.inria.fr/Publications/2009/CNLE09/CNLE09.pdf).
- **Used by:** GPU_VOXEL_WORLDGEN (pool); VOXEL_LARGE_SCENE Phase B.

### 1.7 NanoVDB / OpenVDB (Museth)  — **Tier A (values) / B · REJECTED (as our resident format)**
- **What:** linear, pointer-free hierarchical sparse grid (`5,4,3` tree) + HDDA; optional 2/4/8/16-bit per-block
  quantization.
- **Numbers:** **4–6×** from quantization; sparse from the tree.
- **Decision:** **REJECTED as the resident format** — its compression is **float/scalar quantization** (density,
  SDF), *not discrete palette ids*; we'd gain nothing on block ids and lose our brick/halo/clipmap machinery.
  **ADOPT its philosophy:** pointer-free, memcpy-able, GPU-random-access — the discipline every Tier-A format
  follows.
- **Source:** [dl.acm.org NanoVDB](https://dl.acm.org/doi/fullHtml/10.1145/3450623.3464653),
  [OpenVDB NanoVDB FAQ](https://academysoftwarefoundation.github.io/openvdb/NanoVDB_FAQ.html).
- **Used by:** VOXEL_STORAGE_PLAN §5 (rejected, philosophy adopted).

### 1.8 SVDAG (Kämpe 2013)  — **Tier B · REJECTED live; ADOPTED as asset transport**
- **What:** sparse voxel octree with **identical subtrees merged** into a DAG.
- **Numbers:** 10–100×+ on binary geometry.
- **Decision:** **REJECTED as the live (Tier-A) structure** — the per-voxel query becomes pointer-chasing
  octree/DAG descent, the opposite of the flat O(1) fetch; it can't be addressed by `meta.voxel_offset` and
  doesn't compose with the AABB-per-brick BLAS. **ADOPTED Tier-B only:** an offline DAG pass on a *static
  imported asset's* `.vxo` is a legit 10–100× win, decoded to bricks on import (the Aokana shape).
- **Source:** [Kämpe et al. 2013](https://www.researchgate.net/publication/262367808_High_resolution_sparse_voxel_DAGs).
- **Used by:** VOXEL_STORAGE_PLAN R5 (optional asset DAG), §5 (rejected live).

### 1.9 SSVDAG (Villanueva 2016)  — **Tier B · REJECTED live**
- **What:** SVDAG + **similarity-transform (symmetry) merging** + variable-bit pointers.
- **Numbers:** up to **~2× over SVDAG**; **~0.12 bits/voxel** (6 GB-class voxel scenes in <86 MB).
- **Decision:** **REJECTED live** — even more indirection; pure on-disk/compression-domain form. Same Tier-B
  asset-only verdict as SVDAG.
- **Source:** [SSVDAGs PDF (I3D 2016)](https://www.crs4.it/vic/data/papers/i3d2016-symmetry-dags.pdf).
- **Used by:** VOXEL_STORAGE_PLAN §5; VOXEL_LARGE_SCENE §2 (transport only).

### 1.10 Aokana — chunked SVDAG + LOD aggregation + streaming  — **Tier B chunks → Tier A pool · ADOPTED (the spine)**
- **What:** world in **chunks**, each chunk an **SVDAG**, **streamed**, with an **LOD-aggregation** mechanism;
  the SVDAG is the *stored/streamed* form, **decoded to a GPU-traceable representation per resident chunk**.
- **Numbers:** **tens of billions of voxels**; **up to 9× less memory**, **up to 4.8× faster** than prior SOTA.
  *(The "~10 B vox / ~400 MB / ~5% resident / 6 ms" figures in our internal survey are a paraphrase; the paper's
  published headlines are the 9× / 4.8×.)*
- **Decision:** **ADOPTED the streaming + LOD-aggregation + decode-to-pool spine** (the full target shape), with
  our flat brick as the decoded form. We stay SVDAG-free *on the trace path* by design — the DAG is the transport,
  not the trace structure.
- **Source:** [arXiv 2505.02017](https://arxiv.org/abs/2505.02017),
  [ACM CGIT 2025](https://dl.acm.org/doi/10.1145/3728299).
- **Used by:** GPU_VOXEL_WORLDGEN (chunked streaming); VOXEL_LARGE_SCENE §2 (LOD aggregation); VOXEL_STORAGE R5.

### 1.11 RLE / zstd / Brotli  — **Tier B · ADOPTED (`.vxo` BRIK / R5)**
- **What:** byte-stream compression of serialized bricks; RLE the strata bands (strata is run-length-friendly
  along Y).
- **Numbers:** 3–20× on uniform/strata data; Brotli for max ratio if load time allows.
- **Decision:** **ADOPTED for disk only.** On-disk order: sparse brickmap → per-brick palette+bitpack (R2 format,
  disk↔VRAM shared) → brick dedup (R3) → RLE strata → zstd wrapper. Never traced; expanded to Tier-A on load.
- **Source:** general (zstd / Brotli).
- **Used by:** VOXEL_STORAGE_PLAN R5; VOXEL_INSTANCING §1.5 (`.vxo` BRIK); VOXEL_FINE_RESOLUTION S2.

### 1.12 Per-voxel `u8`/`u16` narrowing  — **Tier A · REJECTED (in favor of R1–R3)**
- **What:** narrow the resident id from `u32` to `u16`/`u8` for a flat 2–4×.
- **Decision:** **REJECTED as a primary lever** — real but dominated by R1+R3 (which make most interior bricks
  cost ~0 *regardless* of per-voxel width); it also caps the palette. A `u16` index buffer falls out of R2 anyway
  (R2 stores indices, not ids). Do the structural wins first.
- **Used by:** VOXEL_STORAGE_PLAN §5.

> **STORAGE headline adopt/reject:** ADOPT brick-local Tier-A (R1 uniform-collapse ✓landed, R3 dedup ✓landed, R2
> palette = next, R4 occupancy = later) — together they turn the **2–4× cost of solid interiors into ≈0**
> (~240 MB → low tens of MB). DAG/tree forms (SVDAG/SSVDAG/Aokana-chunk) are **Tier B disk/asset only**, decoded
> to flat bricks before tracing. REJECT NanoVDB (float quant, wrong for palette ids), deep-64-tree as live store,
> and `u8` narrowing as a primary lever.

---

## 2. Global Illumination

> Authority: [`docs/REFERENCES.md`](REFERENCES.md) §5–6; GI stack named in
> [`docs/GPU_VOXEL_WORLDGEN_PLAN.md`](GPU_VOXEL_WORLDGEN_PLAN.md). The whole GI plumbing is reused **near-verbatim**
> from the vendored `bevy_solari` fork. Standing rule: **GI is world-space and agnostic to how bricks arrive — do
> NOT change it to buy streaming speed; edits ADAPT locally, never full-clear the cache or DLSS history.**

### 2.1 ReSTIR DI (Bitterli 2020)  — **ADOPTED (reused from bevy_solari)**
- **What:** spatiotemporal reservoir resampling for real-time direct lighting with many dynamic lights.
- **Decision:** **ADOPTED** — reservoirs per screen pixel, reproject through world motion; emissive voxels act as
  the area-light set sampled. Impl ref: `bevy_solari/src/realtime/restir_di.wgsl`.
- **Source:** [NVIDIA 2020](https://research.nvidia.com/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct).

### 2.2 ReSTIR GI (Ouyang 2021)  — **ADOPTED (reused from bevy_solari)**
- **What:** path resampling for real-time indirect (single-bounce GI informed by this).
- **Decision:** **ADOPTED** — our own single-bounce GI uses the ReSTIR GI plumbing; emissive voxels are area
  lights; results temporally accumulated/resampled. Impl ref: `restir_gi.wgsl`.
- **Source:** [NVIDIA 2021](https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing).
- **Known noise ladder** (memory `voxel-rt-gi-noise`): surface "boiling" = emitter-catch count variance RR can't
  converge → fix ladder more-rays/DLAA → low-discrepancy sampling (landed) → firefly-clamp knob (landed) → NEE →
  ReSTIR. Plan-to-best-practice: discard firefly clamping in favor of ReSTIR + cache + DLSS-RR + NEE.

### 2.3 SHARC / RTXGI 2.0 world cache  — **ADOPTED (reused from bevy_solari world-cache)**
- **What:** a **world-space** radiance cache — a hash grid keyed on **quantized world position + world normal**;
  per-cell life/decay + lazy re-insert on query (NVIDIA SHARC / RTXGI 2.0 lineage).
- **Decision:** **ADOPTED** as `query_world_cache` (`world_cache_{query,update,compact}.wgsl`). The world-space key
  is *why* static instances are GI-free (their surfaces hash to fixed cells like terrain) and *why* continuously
  moving instances are the hard case (a moved cell is stale) — see VOXEL_INSTANCING §5. **Adapt, not reset:** edits
  trigger a *local* re-pack (or swept-AABB targeted decay), never a global clear.
- **Source:** NVIDIA SHARC / RTXGI 2.0; impl ref `bevy_solari` world-cache.
- **Used by:** VOXEL_INSTANCING §5 (static = free; moving = swept-AABB decay; per-object cache deferred);
  GPU_VOXEL_WORLDGEN (GI unchanged invariant).

### 2.4 NEE / emissive-voxel light sampling  — **ADOPTED**
- **What:** next-event estimation against a light list of air-exposed emissive voxels (the `LITE` `.vxo` chunk
  bakes this at import); `gpu.rs::build_light_list`, bounded by `MAX_VOXEL_LIGHTS`.
- **Decision:** **ADOPTED** — emissive `.vox`/`.vxo` blocks (read from the `MATL` chunk) light the scene via the
  NEE list + cache path; static-instance emissive contribution built once at placement and merged into the
  resident light list.
- **Used by:** VOXEL_INSTANCING §1.5 (`MATL`/`LITE`), §5.1; GI noise ladder (NEE rung).

### 2.5 DLSS Ray Reconstruction  — **ADOPTED (reused from bevy_solari resolve)**
- **What:** feed noisy single-bounce GI + G-buffer guides (albedo, normal, depth, motion vectors) to DLSS-RR for
  denoise + super-resolution.
- **Decision:** **ADOPTED, default feature.** Reuse the resolve pass `resolve_dlss_rr_textures.wgsl` (+
  `gbuffer_utils.wgsl`) — it shows exactly which guide buffers RR expects. Contract: depth-jittered /
  motion-unjittered; **never full-clear DLSS history on an edit** (adapt locally). No-admin SDK setup; needs the
  `bevy/dlss` umbrella feature; forces Vulkan; `build.rs` copies DLLs (memory `solari-gi-worktree`).
- **Source:** [NVIDIA Streamline](https://github.com/NVIDIA-RTX/Streamline),
  [DLSS](https://developer.nvidia.com/rtx/dlss).

### 2.6 DDGI / Radiance Cascades  — **REFERENCE / PRIOR (not the live GI path)**
- **What:** DDGI (Majercik et al. JCGT 2019) ray-traced irradiance probe fields; Radiance Cascades (Sannikov)
  angular-hierarchy GI.
- **Decision:** **NOT the live path** for the voxel-RT pivot (which is ReSTIR + world-cache + DLSS-RR). Retained as
  *prior-art reference* — RC merge math saved at `docs/reference/*.txt` (canonical projective-visibility merge),
  RC paper at `D:/refs/RadianceCascadesPaper`; DDGI lived in the SDF era (`docs/DDGI_*`, memory `ddgi-*`). Consult
  if revisiting probe/cascade GI.
- **Source:** [DDGI JCGT 8(2) 2019](https://jcgt.org/published/0008/02/01/);
  [Raikiri/RadianceCascadesPaper](https://github.com/Raikiri/RadianceCascadesPaper).

> **GI headline adopt/reject:** ADOPT the bevy_solari stack near-verbatim — **ReSTIR DI + ReSTIR GI + SHARC/RTXGI
> world-space cache + NEE emissive lights + DLSS-RR** (default). World-space cache key is load-bearing: static
> instances are free, moving instances get swept-AABB *local* decay (never a global clear), continuously-rotating
> sub-worlds need an object-space cache (DEFERRED, known answer). DDGI / Radiance Cascades are prior-era reference,
> not the live path.

---

## 3. Streaming / Residency / LOD

> Authority: [`docs/VOXEL_LARGE_SCENE_PLAN.md`](VOXEL_LARGE_SCENE_PLAN.md) (residency axis) +
> [`docs/GPU_VOXEL_WORLDGEN_PLAN.md`](GPU_VOXEL_WORLDGEN_PLAN.md) (GPU-driven streaming). The headline-ratio storage
> forms bound *disk/stream* size; what bounds the **resident GPU set** for a large view is three Tier-A *policies*:
> surface-only residency, demand/ray-guided residency, LOD aggregation.

### 3.1 Geometry clipmaps (nested camera-following shells)  — **ADOPTED, LANDED (the residency baseline)**
- **What:** per LOD `L`, a `(2·clip_half+1)³` cube on the LOD-`L` grid; LOD0 fills the inner cube, each coarser
  level a shell. Empty (all-air) bricks dropped + memoized (sky is free).
- **Numbers:** `clip_half=8`, `MAX_LOD=7` → 8 nested shells reaching **~1640 m**, capped 60k bricks. Per-single-
  brick-move streaming cost is **O(shell)** (~6.7 ms / 786 bricks vs 138 ms dense cold-fill — 21×).
- **Decision:** **ADOPTED, landed** (GPU_VOXEL_WORLDGEN Stage 4). It *is* our LOD-aggregation substrate (a coarse
  brick is always `8³` but spans `2^L` more world). The remaining wall: resident-brick **count** grows Θ(H³) — see
  3.3.
- **Source:** Hoppe geometry clipmaps lineage; our `streaming.rs` `desired_clipmap`.

### 3.2 GigaVoxels ray-guided residency + LRU pool  — **ADOPTED (DEFERRED to endgame)**
- **What:** rendering emits a **ray-guided request buffer** (each ray reports the brick/LOD it needed but found
  missing, appended via an atomic counter — no readback on the hot path); an **LRU** pool evicts least-recently-
  used bricks; view distance decoupled from VRAM.
- **Decision:** **ADOPTED as the model, DEFERRED** to the GPU-worldgen pivot's *Stage 5 / VOXEL_LARGE_SCENE Phase
  B* — **only if** surface-only (3.3) + clipmap prefetch still overflow the pool at target reach. Layers onto the
  fixed-cap pool (degenerate AABBs for free slots). Keep the flat brick (no SVO descent on the hot path). Latency
  mitigations: clipmap prefetch keeps near-field resident; fall back to the coarser resident ancestor (a miss is a
  blurrier pixel, never black); keep-old-until-revealed.
- **Source:** [Crassin INRIA 2009](https://maverick.inria.fr/Publications/2009/CNLE09/CNLE09.pdf).
- **Used by:** VOXEL_LARGE_SCENE Phase B; GPU_VOXEL_WORLDGEN Stage 5.

### 3.3 Surface-only / shell residency (Teardown-class first-hit)  — **ADOPTED (Phase A, the big win)**
- **What:** a ray commits at the **first surface**, so any brick with all 6 neighbours solid (an *enclosed* brick)
  can **never be the nearest first-solid hit** → keep only the **air-exposed surface shell** in the BLAS/Tier-A
  set; enclosed interiors live in the cheap CPU brickmap only.
- **Numbers:** resident set **Θ(H³) → Θ(H²)** in `clip_half` — converts view *volume* to view *surface*. A solid
  `m³` mass has shell `≈ 6m²`. Pushing the view ~10× farther costs ~**100×** instead of ~**1000×**. Meta+AABB VRAM
  ~3.8 MB → ~0.6–1 MB.
- **Why exact (not conservative):** culling *provably-unreachable* bricks given a first-hit DDA — pixel-identical,
  *faster* trace + BLAS build. (Contrast the old SDF conservative-occupancy mask, which *had* to be conservative
  because the SDF empty-space DDA could skip past thin features.)
- **Decision:** **ADOPTED — the dominant, lowest-risk win.** Pure filter on the packed set (no shader change, no
  new buffer). Unifies with R1 on **one enclosed-predicate SSOT** (uniform-incl-halo ⊂ enclosed). Risk: edit that
  *exposes* a buried brick must promote it same-frame — mitigated because the predicate is a pure function of a
  brick + its 6 neighbours, and the edit-dirty set is exactly that. Shell-boundary: cull only on complete same-LOD
  neighbour info (missing/different-LOD ⇒ keep, conservative toward keeping).
- **Source:** Teardown first-hit / early-z ([breakdown](https://juandiegomontoya.github.io/teardown_breakdown.html)).
- **Used by:** VOXEL_LARGE_SCENE Phase A (the headline of that doc).

### 3.4 Nanite-style screen-error LOD selection  — **ADOPTED (Phase C polish)**
- **What:** stream only what the screen needs at the LOD it needs; LOD by **projected screen size** (one voxel ≈
  one pixel); clusters below a pixel threshold culled; feedback-driven paging.
- **Decision:** **ADOPTED as optional polish (Phase C)** — replace the pure-distance shell test with a projected-
  voxel-footprint `want_lod`, feeding the Phase-B request LOD; spend the far-field budget where screen-error is
  largest. (GigaVoxels ray-guidance re-derived for triangles.)
- **Source:** [UE5 Nanite virtualized geometry](https://cs418.cs.illinois.edu/website/text/nanite.html).
- **Used by:** VOXEL_LARGE_SCENE Phase C.

### 3.5 Distant Horizons / LOD-by-aggregation  — **ADOPTED (principle, via clipmap)**
- **What:** aggregate finer levels into coarser chunks → huge view distance at bounded VRAM (Distant Horizons
  Minecraft mod; Aokana LOD aggregation).
- **Decision:** **ADOPTED** — our clipmap already does this (coarse brick covers `2^L×` more world at fixed `8³`
  resolution); surface-only + demand let it grow far. The GPU downsamples per `want_lod` at pack time.
- **Source:** Distant Horizons; Aokana ([arXiv 2505.02017](https://arxiv.org/abs/2505.02017)).
- **Used by:** GPU_VOXEL_WORLDGEN Stage 4; VOXEL_LARGE_SCENE §2.

### 3.6 Many small BLASes under a TLAS (OptiX large-scene practice)  — **ADOPTED (per-chunk BLAS)**
- **What:** many small BLASes so only *changed/visible* BLASes rebuild; instancing of repeats; compaction.
- **Decision:** **ADOPTED** — per-chunk BLAS (chunk = KxKxK bricks) so only changed chunks rebuild (GPU_VOXEL_
  WORLDGEN Stage 3), replacing the single full-rebuild BLAS. Composes with surface-only (fewer prims per chunk).
- **Engine fact:** AABB BLAS *from a GPU buffer* works on our fork (`raytrace.rs`); **no indirect AS build** ⇒
  fixed-capacity pool, always build `N_capacity` AABBs, degenerate/zero-extent AABBs for free slots, no CPU
  readback of voxel data in the hot path.
- **Used by:** GPU_VOXEL_WORLDGEN Stage 3; VOXEL_LARGE_SCENE §2.

> **STREAMING/LOD headline adopt/reject:** the dominant resident-set win is **surface-only residency** (Θ(H³)→Θ(H²),
> exact given first-hit DDA, ~10× farther at bounded VRAM, ship first). **Demand/ray-guided + LRU** (GigaVoxels)
> caps the remaining surface set — deferred endgame. **LOD aggregation** (clipmap, landed) + **screen-error LOD**
> (Nanite, polish) round it out. SVDAG/Aokana-chunk bound *disk/stream*, not the resident set — decoded to a flat
> pool before tracing.

---

## 4. Acceleration Structure / Instancing

> Authority: [`docs/VOXEL_INSTANCING_PLAN.md`](VOXEL_INSTANCING_PLAN.md) ("Prior art this design draws on").
> Core idea borrowed from Teardown + standard HW-RT instancing: **geometry is object-local and shared; placement
> is per-instance and cheap.**

### 4.1 Teardown — per-object voxel volumes  — **ADOPTED (the VoxelObject/VoxelInstance split)**
- **What:** the world as *thousands of independent voxel volumes*, each its own grid + palette + transform,
  rasterized as a bounding box and raymarched in **object-local space**, with a per-object palette lookup.
- **Decision:** **ADOPTED** — this is exactly a BLAS-per-object + TLAS-instance-per-placement in HW-RT terms; it is
  our `VoxelObject` (immutable shared asset) / `VoxelInstance` (cheap per-placement transform) split, per-object
  palette via a `palette_base` offset, and per-instance destruction independence (cut one tree, siblings untouched).
- **Source:** [gamedeveloper.com Teardown](https://www.gamedeveloper.com/design/how-beautiful-voxels-laid-the-way-for-i-teardown-s-i-heist-y-framework),
  [acko.net Teardown frame teardown](https://acko.net/blog/teardown-frame-teardown/).

### 4.2 HW-RT 2-level instancing  — **ADOPTED (TLAS instance per placement)**
- **What:** the TLAS leaf stores a world→object transform; on entering a leaf the ray is transformed into the
  BLAS's local space and traversal continues there. **One BLAS backs many TLAS instances** ⇒ repeated geometry
  costs memory once. Nested instancing = chained transforms.
- **Decision:** **ADOPTED** — replace the single global BLAS + identity TLAS with one BLAS per `VoxelObject` + one
  TLAS instance per leaf carrying an arbitrary 3×4 transform + `instance_custom_index → GpuInstanceDescriptor`
  (meta/voxel/palette base offsets + inverse transform). Off-axis ("tree on its side") is *just the instance
  rotation* — DDA walks the un-rotated brick grid, correct by construction. **Key cross-cutting decision: build
  the Phase-3 TLAS with full per-instance 3×4 transforms + the descriptor indirection from day one** (a chunk is
  the degenerate translation-only case) — so props/nesting need no second AS refactor. True hardware TLAS-in-TLAS
  is **DEFERRED** (flattening via Bevy `GlobalTransform` is portable and adequate until thousands of leaves move
  rigidly per frame).
- **Source:** [USPTO 11,282,261 — alternative world-space transforms](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11282261);
  wgpu `TlasInstance::new(&blas, transform_3x4, custom_index, mask)`.

### 4.3 MagicaVoxel `.vox` scene graph  — **ADOPTED (import only)**
- **What:** `nTRN` (with `_t` translation + `_r` int8 rotation/reflection byte) / `nGRP` / `nSHP` nested transform
  nodes; `MATL` chunk carries emissive (`_emit`/`_flux`) + material dicts.
- **Decision:** **ADOPTED as import/interchange only** — we already parse `_t`; now read `_r` + the group
  hierarchy (merge mode bakes `_r` into object-local voxels; scene mode imports each subtree as its own object +
  child instance). **Add a `MATL` emissive reader** (currently dropped). The *native runtime asset is our own
  `.vxo`* (RIFF-style tagged/length-prefixed/skippable chunks — `.vox`'s one good idea), not `.vox` (256-colour
  cap + string-keyed dicts + scene-graph cruft rejected). `.vxo` chunks: HEAD / MATL / BRIK / [LITE / LODS / INST /
  SOCK / PHYS].
- **Source:** [ephtracy voxel-model .vox extension spec](https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox-extension.txt).
- **Used by:** VOXEL_INSTANCING §1.2, §1.5.

### 4.4 Brickmap/SVO instancing & per-object LOD pyramid  — **ADOPTED**
- **What:** object-local sparse bricks + per-object downsample pyramids (solid-if-any + dominant block) for distant
  instances; far proxy/impostor collapse.
- **Decision:** **ADOPTED** — per-object LOD pyramid (one tiny BLAS per level), picked by projected screen size;
  far-proxy collapse so a forest of thousands of distant trees costs a handful of instances. Reuses
  `StaticVoxSource` downsample. Coexists with the world clipmap (independent TLAS members, nearest-`t` resolves).
- **Used by:** VOXEL_INSTANCING §6.

> **INSTANCING headline adopt/reject:** ADOPT Teardown's object-local-volume model + HW-RT 2-level instancing
> (BLAS-per-object, TLAS-instance-per-placement, full 3×4 transforms + descriptor table **from day one**). `.vox`
> is **import-only** (read `_r` + `MATL` emissive); the **`.vxo`** chunked format is the native runtime asset. True
> hardware TLAS-in-TLAS and continuously-moving-instance object-space GI are DEFERRED.

---

## 5. DDA / Traversal Primitives

> Authority: [`docs/REFERENCES.md`](REFERENCES.md) §1–4. These are the foundational, non-controversial primitives
> the whole Tier-A path stands on.

### 5.1 Brickmap layout (8³ palette voxels)  — **ADOPTED (the core structure)**
- **What:** sparse top-level grid of bricks, each an `8³` block; linear index-into-arena (no pointers); per-brick
  occupancy + LOD; palette/indexed colors per brick; GPU-resident with CPU streaming of surface bricks on miss.
- **Source:** [stijnherfst/BrickMap](https://github.com/stijnherfst/BrickMap) (CUDA path tracer, 16³ superchunks,
  12-bit indices, streaming request buffer, 3 LODs) — **local `D:/refs/BrickMap`**; UU thesis (link-only);
  [dubiousconst282/VoxelRT](https://github.com/dubiousconst282/VoxelRT) `eXtendedBrickMap` (sectors→bricks→voxels,
  4³ occupancy bitmasks) — **local `D:/refs/VoxelRT`**.

### 5.2 Per-brick procedural AABB BLAS + HW traversal  — **ADOPTED (the Teardown approach)**
- **What:** register one procedural AABB per brick as custom-geometry BLAS, let the HW BVH cull/sort brick hits,
  then run a fine 3D-DDA inside the intersection shader.
- **Source:** Teardown ([blog.voxagon.se](https://blog.voxagon.se/)); `wgpu` `ray_query` + procedural-AABB AS
  (**vendored `D:/wgpu-fork`** — `examples/features/src/ray_aabb_compute/`,
  `tests/.../ray_tracing/as_aabb.rs`, `naga/src/back/spv/ray/query.rs`); VoxelRT `StdBVH` (BVH+DDA over 8³ leaves).

### 5.3 In-shader 3D-DDA (intra-brick)  — **ADOPTED (Amanatides & Woo)**
- **What:** the incremental DDA loop (compare per-axis `tMax`, step the least-progressed axis) in WGSL inside the
  brick AABB intersection.
- **Source:** [Amanatides & Woo, EG 1987](http://www.cse.yorku.ca/~amana/research/grid.pdf) (canonical grid-DDA);
  [dubiousconst282 guide](https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/) (DDA pitfalls at scale,
  parametric vs incremental, mirrored at `D:/refs/VoxelRT/docs/VoxelNotes.md`);
  [Branchless Voxel Raycasting Shadertoy](https://www.shadertoy.com/view/4dX3zl) (WGSL loop template); shocovox
  `viewport_render.wgsl` (**local `D:/refs/shocovox`**, archived → successor VoxelHex).
- **WGSL gotcha** (memory `wgsl-integer-ops-gpu`): signed `%` returns unsigned + float `/` has 1-ULP error for
  negatives — use the safe `floor_div`.

### 5.4 Voxel surface normals  — **ADOPTED (branchless step-mask normal)**
- **What:** cube-face normal from the **crossed axis** of the last DDA step — `mask * -sign(rayDir)`.
- **Source:** Amanatides & Woo; [Branchless Voxel Raycasting Shadertoy](https://www.shadertoy.com/view/4dX3zl).

---

## 6. Scene Sources (voxelization corpus)

> The side-by-side Gallery corpus. Authority: memories `classic-scene-sources`, `assetgen-voxelizer-improvements`;
> voxelization plan in [`docs/VOXEL_INSTANCING_PLAN.md`](VOXEL_INSTANCING_PLAN.md) §1.5;
> density/scale targets in [`docs/VOXEL_FINE_RESOLUTION_PLAN.md`](VOXEL_FINE_RESOLUTION_PLAN.md).
> Voxelizer accepts glTF (PNG textures only) + OBJ; **1.5B-dense guard**; conservative triangle-box-SAT occupancy.

| Scene | Origin / format | Voxelization notes | Status |
|---|---|---|---|
| **Sponza** | Crytek/Khronos glTF (PBR variant) | **Already oversampled at 0.05 m** (prior experiment) — needs NO re-bake at the fine-resolution flip; currently loads ~4× oversized vs the still-0.2 m engine until S3. | corpus, baked |
| **Sibenik** | McGuire archive OBJ | ~8–13 MB @0.2 m → **~0.5–0.8 GB @0.05 m**; re-bake at S3. | corpus |
| **Conference** | McGuire archive OBJ | ~8–13 MB @0.2 m → **~0.5–0.8 GB @0.05 m**; re-bake at S3. | corpus |
| **Bistro (Exterior)** | Amazon Lumberyard; qian-o **pre-converted glTF** (KTX2 → flat colours) | 41 MB / 10.3 M vox @0.2 m → **~2.6 GB / ~660 M vox @0.05 m**; Exterior @0.05 m is **>1.5B dense** ⇒ needs the **TILED bounded-RAM voxelizer** (#125). KTX2 textures resolve to flat colours. | corpus, heavy |
| **San Miguel** | McGuire archive | **SKIP** — too heavy. | excluded |

**Units / scale:** scenes load **into the world brick map** (the merge path, not per-object instances — user-
confirmed), so a scene's detail *is* the world's detail; this is why finer scenes force a finer global
`VOXEL_SIZE` (the 0.2 m → 0.05 m flip = 64× more voxels per scene). At 0.05 m **every** scene's `.vox` is ~1 GB+
and its in-RAM `BrickMap` is multiple GB ⇒ the re-bake **cannot land until R5 `.vxo`** (compact disk + streamed
load, no full-RAM dense expand) and **R2** (resident VRAM at 4× density) exist. The flip + re-bakes are one atomic
step (scenes load wrong-scaled between flip and re-bake).

**Voxelizer improvements to port** (from `D:\Projects\asset gen`, memory `assetgen-voxelizer-improvements`), keeping
our conservative SAT occupancy + speed:
- **Area-averaged albedo** — fixes nearest-texel aliasing in `sample_albedo`.
- **ALWAYS-ON exterior-floodfill interior SOLID fill** — user directive, interiors solid *always* (open/exterior-
  reachable space stays air; only enclosed interiors fill). This is what *creates* the enclosed-brick mass that
  storage R1 collapses and large-scene surface-only residency culls — the destructible vision needs solid interiors
  so a cut reveals strata, not empty space.
- **CIELAB-space palette** — replaces sRGB median-cut for fidelity.
- Skip the char-art stages.

---

## 7. Local checkouts & saved references

> Mirror of [`docs/REFERENCES.md`](REFERENCES.md) "Local checkouts" — kept here for one-stop lookup.

| Path | Repo | Role |
|---|---|---|
| `D:/refs/BrickMap` | [stijnherfst/BrickMap](https://github.com/stijnherfst/BrickMap) (MIT) | Brickmap structure + GPU streaming (CUDA) |
| `D:/refs/VoxelRT` | [dubiousconst282/VoxelRT](https://github.com/dubiousconst282/VoxelRT) | Accel-structure benchmark suite (MultiDDA / brickmap / sparse-64-tree) + `docs/VoxelNotes.md` |
| `D:/refs/shocovox` | [davids91/shocovox](https://github.com/davids91/shocovox) (MIT/Apache, archived → VoxelHex) | WGSL+Rust sparse-voxel ray-marcher (closest stack match) |
| `D:/refs/RadianceCascadesPaper` | [Raikiri/RadianceCascadesPaper](https://github.com/Raikiri/RadianceCascadesPaper) (CC-BY-ND) | RC paper source (prior-era GI) |
| `D:/bevy-fork/crates/bevy_solari` | vendored bevy_solari (Bevy 0.19) | ReSTIR DI/GI + world-cache + DLSS-RR resolve reference |
| `D:/wgpu-fork` | vendored wgpu (trunk) | `ray_query` / procedural-AABB BLAS API + examples/tests |
| `docs/reference/*.txt` | saved Shadertoy 3D Radiance Cascades (BufferA–D, CubeA, Image, common) | canonical projective-visibility merge port (prior-era GI) |

---

## 8. Quick decision table

| Area | Technique | Tier | Decision | One-line reason |
|---|---|---|---|---|
| Storage | Uniform-brick collapse (R1) | A | **ADOPTED ✓landed** | 4 KB → ~8 B for buried bricks; faster trace |
| Storage | Per-brick palette + bitpack (R2) | A | **ADOPTED (next)** | strata 4 KB → ~258 B; proven Teardown path |
| Storage | Brick dedup / COW (R3) | A | **ADOPTED ✓landed** | identical interiors → one shared slice; zero shader change |
| Storage | Occupancy + compacted solids (R4) | A | **ADOPTED (later)** | surface air cells → 1 bit; low priority |
| Storage | `.vxo` zstd/RLE/palette (R5) | B | **ADOPTED (disk)** | 3–20×; expanded to Tier-A on load |
| Storage | Sparse 64-tree as live store | A/B | **REJECTED live** | leaves still want flat bricks; BLAS already space-skips |
| Storage | NanoVDB as resident format | A/B | **REJECTED** | float quant, not palette ids (borrow its pointer-free philosophy) |
| Storage | SVDAG / SSVDAG as live store | B | **REJECTED live / asset-only** | pointer-chasing breaks O(1) DDA; OK as offline asset transport |
| Storage | Aokana chunked-SVDAG + LOD-agg | B→A | **ADOPTED (spine)** | decode-to-pool streaming shape; 9× mem / 4.8× faster |
| Storage | `u8`/`u16` narrowing | A | **REJECTED (primary)** | dominated by R1+R3; falls out of R2 |
| GI | ReSTIR DI / GI | — | **ADOPTED** | reused near-verbatim from bevy_solari |
| GI | SHARC / RTXGI world cache | — | **ADOPTED** | world-space key; adapt-not-reset on edits |
| GI | NEE emissive-voxel lights | — | **ADOPTED** | `MATL`/`LITE`; bounded light list |
| GI | DLSS-RR | — | **ADOPTED (default)** | denoise+upscale noisy single-bounce; never clear history |
| GI | DDGI / Radiance Cascades | — | **REFERENCE only** | prior SDF era, not the live voxel-RT path |
| Streaming | Geometry clipmap | A | **ADOPTED ✓landed** | nested camera shells; LOD-aggregation substrate |
| Streaming | Surface-only residency | A | **ADOPTED (Phase A, big win)** | Θ(H³)→Θ(H²), exact given first-hit DDA, ~10× reach |
| Streaming | GigaVoxels ray-guided + LRU | A | **ADOPTED (deferred endgame)** | caps remaining surface set; only if Phase A overflows |
| Streaming | Nanite screen-error LOD | A | **ADOPTED (polish)** | spend far budget where screen-error is largest |
| Streaming | Distant Horizons LOD-agg | A | **ADOPTED (via clipmap)** | coarse brick covers 2^L× more world |
| Streaming | Many small BLASes / per-chunk | A | **ADOPTED** | rebuild only changed chunks; fixed-cap pool (no indirect AS build) |
| Instancing | Teardown object-volumes | A | **ADOPTED** | VoxelObject/VoxelInstance split, per-object palette |
| Instancing | HW-RT 2-level instancing | A | **ADOPTED** | BLAS-per-object + 3×4 TLAS transforms from day one |
| Instancing | `.vox` scene graph | — | **ADOPTED (import only)** | read `_r` + `MATL` emissive; `.vxo` is the runtime asset |
| Instancing | TLAS-in-TLAS (true nesting) | A | **DEFERRED** | flattening via GlobalTransform adequate + portable |
| DDA | Brickmap 8³ + AABB BLAS + Amanatides-Woo DDA | A | **ADOPTED (core)** | foundational traversal primitives |
```
