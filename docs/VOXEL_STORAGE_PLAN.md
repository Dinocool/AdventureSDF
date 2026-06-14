# Voxel Storage & Compression — Design Plan

Status: DESIGN (no engine code changed by this doc). Worktree: `voxel-rt`. Read-only survey + phased plan.
Target: Bevy 0.19 + forked wgpu-trunk, RTX 4090 / Vulkan. Extends the HW-ray-traced brickmap path
(`src/voxel/{brickmap,gpu,streaming,palette}.rs` + `assets/shaders/voxel_raytrace.wgsl`) and the planned
GPU brick pool (`docs/GPU_VOXEL_WORLDGEN_PLAN.md`) + `.vxo` BRIK chunk (`docs/VOXEL_INSTANCING_PLAN.md §1.5`).

The motivating change: we now store **solid interiors** (always-on exterior-floodfill interior fill —
`VOXEL_INSTANCING_PLAN.md §1.5`), so the world is dominated by the *most compressible* voxels — large
uniform regions and horizontally strata-layered bands, with many **identical** interior bricks. Raw, that is
a 2–4× memory blow-up over a hollow shell. This doc says how to claw essentially all of it back.

---

## 0. The hard constraint (state it, design around it)

The **live render path is hardware ray tracing**. A ray hits a brick's procedural AABB (the BLAS primitive),
then the shader runs an **in-shader 3D-DDA** (`dda_brick` in `voxel_raytrace.wgsl`) that, for every cell it
steps through, does:

```wgsl
let id = voxels[m.voxel_offset + cell_index(vox.x, vox.y, vox.z, hedge)];   // O(1) flat fetch
```

So **any VRAM-resident format must answer "block id at local (x,y,z)" in ≈O(1) from a GPU storage buffer**,
inside a hot per-step loop, with no pointer chasing and no per-ray decompression of a whole brick. This is the
line that divides every method below into:

- **Tier A — VRAM-resident (must be GPU-DDA-traceable).** Lives in the storage buffers the ray query reads.
  Decode must be a couple of ALU ops + one extra fetch at most, per voxel step.
- **Tier B — on-disk `.vxo` (CPU/GPU decompress on load, never traced directly).** Can be aggressively packed
  (DAG, zstd, RLE) and is expanded to a Tier-A layout at load. The trace never touches it.

Every recommendation is tagged A or B. Confusing the two is the classic mistake — a full SVDAG is a great
Tier-B form and a terrible Tier-A one (§7).

---

## 1. Where we are today (ground truth from the code)

### CPU side (`brickmap.rs`)
- A brick is `8³ = 512` voxels, `0.2 m` at LOD0, world span scales `2^L` with the clipmap LOD.
- `BrickStorage` is **already** `Uniform(BlockId) | Dense(Box<[BlockId; 512]>)` — a uniform-brick fast path
  that stores *one* `BlockId` (`u16`) for a fully-buried interior brick, plus an occupancy bitmask
  (`[u64; 8]` = 64 B). `from_voxels` collapses an all-identical array to `Uniform` automatically.
- **So the CPU side already exploits uniform interiors.** The gap is everything downstream.

### VRAM side (`gpu.rs` → `voxel_raytrace.wgsl`) — the actual resident cost
Per resident brick, regardless of content, `pack_resident_set` emits:

| Buffer | Per brick | Notes |
|---|---|---|
| `voxels: array<u32>` | `halo_cells(0)` = `10³` = **1000 × u32 = 4000 B** | one `BlockId` per `u32` (zero-extended `u16`), **haloed** `10³` |
| `metas: GpuBrickMeta` | **32 B** | `voxel_origin`, `voxel_offset`, `world_min`, `lod` |
| `aabbs: GpuBrickAabb` | **32 B** | BLAS primitive, `min/max` + pad |
| **Total** | **≈ 4064 B/brick** | + a global palette (`GpuPaletteColor` 32 B × ≤256) |

**The critical fact: the CPU uniform-brick collapse is *thrown away* at pack time.** A `Uniform(stone)`
interior brick is expanded into 1000 identical `u32`s in VRAM — 4 KB to say "all stone." At the 60k-brick
resident cap that is **~240 MB of voxel buffer**, the bulk of which, once interiors are solid, is
duplicated-constant data. The `voxel_worldgen_perf` rig already notes "~240 MB scene buffers."

This is the single biggest, lowest-risk win on the table, and it is purely a VRAM/packing change — the DDA
math does not have to move.

---

## 2. Survey of SOTA storage/compression methods

Each row: **what it stores → compression vs. raw `u8`/voxel → Tier A (GPU-DDA-traceable) or B (on-disk/CPU)**.
"Raw" = one block id per voxel (our `u32`-per-voxel VRAM baseline, or `u8`-per-voxel disk baseline).

| Method | What it stores for geometry | Ratio vs. raw | Tier | GPU-DDA-traceable? |
|---|---|---|---|---|
| **Uniform-brick collapse** (ours, CPU only today) | 1 block id for an all-same brick | 512–1000× *for that brick* | **A** (if a flag is added) | **Yes** — DDA reads the single id, no per-cell fetch |
| **Per-brick palette + bit-packed indices** (Teardown/MagicaVoxel/`voxel.wiki`) | tiny per-brick palette of the ≤k distinct ids + `ceil(log2 k)`-bit index per voxel | 8/`bits` (e.g. **8–32×** for 1–4-bit strata) | **A** | **Yes** — 1 extra fetch + bit-extract + 1 palette indirection per step |
| **Occupancy bitmask + compacted solid list** (BrickMap, sparse-64-tree) | 64-bit-per-`4³` (or 512-bit-per-`8³`) mask + prefix-sum → dense solid array | ~2–8× (only solids carry an id) | **A** | **Yes** — popcount-prefix gives the index; common in GPU voxel tracers |
| **Brick-level dedup / instancing** (DAG idea at brick granularity) | each *distinct* brick stored once; bricks reference it by index | **huge** for identical interiors (∞ per duplicate) | **A** | **Yes** — meta's `voxel_offset` just points at the shared slice |
| **Sparse 64-tree** (dubiousconst282 / `tree64`, used by re-flora) | `4³` nodes, 64-bit child+leaf masks, omit empty; optional leaf palette + tile hashing | 2–10×, great space-skipping | **A** (shallow) / B | **Partially** — the *tree* is pointer-ish; works as a top-level space-skip, but a flat brick leaf is what the DDA wants |
| **GigaVoxels brick pool** | fixed GPU pool of `N³` bricks + an octree of pointers into it; ray-guided paging | bounded VRAM, not a per-brick ratio | **A** | **Yes** — this is essentially our planned GPU pool |
| **NanoVDB / OpenVDB** (Museth) | linear, pointer-free hierarchical sparse grid (`5,4,3` tree) + HDDA; optional 2/4/8/16-bit per-block quantization | **4–6×** from quantization; sparse from the tree | **A** (HDDA) / B | **Yes** for *values*; but it quantizes scalar fields, not palette ids — fits float density, not discrete block ids |
| **SVDAG** (Kämpe) | sparse voxel octree with **identical subtrees merged** into a DAG | 10–100×+ (binary geometry) | **B** | **No** for our path — pointer-chasing octree descent per voxel; wrong shape for an AABB-hit DDA |
| **SSVDAG** (Villanueva) | SVDAG + **similarity-transform (symmetry) merging** + variable-bit pointers | up to **~2× over SVDAG**; ~0.12 bits/voxel (6 B voxels in <86 MB) | **B** | **No** — even more indirection; pure on-disk/compression-domain form |
| **Aokana** (arXiv 2505.02017) | world in **chunks**, each chunk an **SVDAG**, streamed; LOD aggregation | up to **9× less memory** vs prior SOTA | **B** chunks → **A** decoded pool | SVDAG is the *streamed/stored* form; decoded to a traceable pool per chunk |
| **RLE / zstd / Brotli** (general) | byte-stream compression of the serialized bricks | 3–20× on uniform/strata data | **B** | **No** — must decompress to a buffer first |

Key reading of the table for **our** problem:
- The methods that **stay traceable (Tier A)** are exactly the *brick-local* ones: uniform collapse, per-brick
  palette + bit-pack, occupancy+compaction, and brick dedup. These are the GigaVoxels/Teardown/BrickMap family.
- The methods with the *headline* ratios (SVDAG/SSVDAG/Aokana-chunk-SVDAG) are **tree/DAG** forms — fantastic
  for **Tier B** (disk + streaming), wrong for the in-shader DDA. Aokana itself **decodes** its SVDAG chunks
  into a GPU-traceable representation; the DAG is the *transport*, not the trace structure.
- NanoVDB's quantization is about *float* fields (density/SDF), not discrete block ids, so its compression knob
  doesn't map onto our palette ids — but its **pointer-free, memcpy-able, GPU-random-access** philosophy is
  exactly the design discipline our Tier-A formats must follow.

Sources: Aokana ([arXiv 2505.02017](https://arxiv.org/abs/2505.02017),
[html](https://arxiv.org/html/2505.02017)); sparse 64-tree
([dubiousconst282](https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/),
[tree64](https://github.com/expenses/tree64), [VoxelRT](https://github.com/dubiousconst282/VoxelRT));
NanoVDB ([dl.acm.org](https://dl.acm.org/doi/fullHtml/10.1145/3450623.3464653),
[OpenVDB FAQ](https://academysoftwarefoundation.github.io/openvdb/NanoVDB_FAQ.html)); SVDAG/SSVDAG
([SSVDAGs PDF](https://www.crs4.it/vic/data/papers/i3d2016-symmetry-dags.pdf)); Teardown palette
([acko.net](https://acko.net/blog/teardown-frame-teardown/), [voxagon](https://blog.voxagon.se/),
[voxel.wiki palette compression](https://voxel.wiki/wiki/palette-compression/)); BrickMap
([stijnherfst/BrickMap](https://github.com/stijnherfst/BrickMap)).

---

## 3. The solid-interior collapse, quantified

Take the dominant case after the always-on interior fill: a worldgen region of mostly-buried bricks. Three
populations:

1. **Fully uniform interior bricks** (all one block — deep stone/bedrock). The common case.
2. **Strata-layered bricks** (a few horizontal bands — e.g. grass/dirt/stone over 2–4 ids).
3. **Surface bricks** (genuinely varied — the air-exposed shell).

Per-brick VRAM, today vs. with the Tier-A wins, using the haloed `10³ = 1000`-cell grid (`hedge=10`):

| Brick type | Today (raw `u32`/voxel) | + uniform collapse | + per-brick palette (bit-pack) | + brick dedup |
|---|---|---|---|---|
| **Uniform interior** | 4000 B | **~8 B** (1 id + flag) | ~8 B | **~0 B** (Nth copy is free; one shared slice) |
| **Strata (≤4 ids)** | 4000 B | 4000 B (not uniform) | palette `4×2 B` + `1000 × 2-bit` = **~258 B** (15.5×) | ~258 B (or shared if identical) |
| **Strata (≤16 ids)** | 4000 B | 4000 B | `16×2 B` + `1000 × 4-bit` = **~532 B** (7.5×) | ~532 B |
| **Surface (varied)** | 4000 B | 4000 B | `k×2 B` + `1000 × ceil(log2 k)-bit` (e.g. 32 ids → 5-bit ≈ **~690 B**, 5.8×) | ~690 B |

**The "2–4× raw cost of solid interiors" collapses to roughly nothing.** Concretely, for a region where
interiors dominate (say 80% uniform, 15% strata, 5% surface — typical once you fill below the surface):

- **Today:** `0.80·4000 + 0.15·4000 + 0.05·4000 = 4000 B/brick` average (content-blind).
- **+ uniform collapse + per-brick palette:** `0.80·8 + 0.15·258 + 0.05·690 ≈ 6.4 + 38.7 + 34.5 ≈ 80 B/brick`
  average voxel data — a **~50× reduction** of the voxel buffer for that mix, *before* dedup.
- **+ brick dedup:** the 80% uniform bricks share a handful of slices (one per (block,lod) combo), so their
  amortized cost → ~0; strata bricks dedup heavily too (the same band pattern repeats across a flat region).
  The voxel buffer for interiors approaches the cost of the **surface shell alone** — i.e. solid-fill becomes
  nearly *free* in VRAM, which is the whole point: we keep destructible solid interiors without paying for them
  until something is actually cut.

Net: the resident voxel buffer drops from the **~240 MB** (60k bricks × 4 KB) baseline toward the
**low tens of MB**, dominated by the surface shell and strata variety, not by buried constant fill.
(The exact factor is content-dependent — §8 says how to *measure* it rather than trust this estimate.)

Note: the **occupancy bitmask** already lives CPU-side (`[u64;8]`, 64 B/brick); promoting a compact occupancy
mask to VRAM (the "+ compacted solid list" row) is an *additional* axis that helps the *surface* bricks
(store ids only for solid cells) but matters far less than the three interior wins, so it is later-phase.

---

## 4. Ranked recommendations (memory impact / effort), mapped to our engine

Ranked by `VRAM reduction ÷ effort`. Each tagged with tier, the buffer/struct it touches, and the
shader-side decode cost.

### R1 — Uniform-brick collapse INTO VRAM  ·  Tier A  ·  effort: S  ·  impact: very high
**The single highest-value change.** Make the GPU layout represent a uniform brick the way the CPU already
does — one block id, not 1000 copies.

- **Layout:** add a `kind`/`flags` field to `GpuBrickMeta` (it has room — currently 32 B, well-aligned). A
  `UNIFORM` brick stores its block id directly in the meta (reuse a field, e.g. pack into `voxel_offset`'s high
  bits or add a `uniform_id: u32` and set `voxel_offset = SENTINEL`). `pack_resident_set` emits **no voxel-buffer
  entries** for uniform bricks.
- **Shader decode (`dda_brick`):** at brick entry, branch on `kind == UNIFORM`. If uniform, every core cell's
  id is `m.uniform_id` (skip the `voxels[...]` fetch entirely); the DDA still steps geometrically for the
  normal/first-solid-face logic but never touches the voxel buffer. **Cost: one branch per brick, *fewer*
  fetches** (a uniform brick is faster to trace, not slower) — and uniform solid bricks are the buried ones a
  ray usually stops at immediately, so this is a net perf win too.
- **Halo caveat (must handle):** the seam fix relies on a brick reading its neighbour's boundary voxel from its
  own haloed grid. A uniform brick has no stored halo. Resolution: a uniform interior brick is fully buried, so
  its halo neighbours are *also* solid — the seam/normal logic only fires at air-exposed faces, which a fully
  buried uniform brick has none of. Keep a uniform brick uniform **only when all 6 neighbours are solid**
  (cheap check the packer already can do via the resident map); otherwise store it dense (it's a surface brick
  anyway). This makes uniform-collapse *exactly* the "fully buried" case — robust by construction.
- **Maps onto:** `GpuBrickMeta` (`gpu.rs`), `pack_resident_set`/`pack_brickmap` (skip emit), `dda_brick`
  (`voxel_raytrace.wgsl`). The CPU `Brick::is_uniform_solid()` + neighbour test already exists.
- **GPU-pool fit:** the planned fixed-capacity pool (`GPU_VOXEL_WORLDGEN_PLAN.md`) gets *more* effective
  capacity — uniform bricks consume a meta+AABB slot but ~0 voxel-pool bytes.

### R2 — Per-brick palette + bit-packed indices  ·  Tier A  ·  effort: M  ·  impact: high
**The strata win.** A brick with ≤k distinct ids stores a tiny palette + `ceil(log2 k)`-bit indices.

- **Layout:** per brick, a small palette (the ≤k `BlockId`s, `u16` each) + a bit-packed index stream. Two
  sub-options:
  - **(a) Fixed tiers** (1/2/4/8-bit) chosen per brick by `k` — simplest to decode (k≤2→1-bit, ≤4→2-bit,
    ≤16→4-bit, ≤256→8-bit). The meta stores `index_bits` + a `palette_offset` into a global palette-slice
    buffer + the usual `voxel_offset` into a (now bit-packed) index buffer.
  - **(b) Always store the brick's local palette inline** before its index stream (one slice).
  Recommend **(a)** with a separate `brick_palettes: array<u16>` buffer (slice per brick) so the index buffer
  stays word-aligned per brick.
- **Shader decode (`dda_brick`):** per cell step: compute bit offset `= cell_index · index_bits`; fetch the
  `u32` word(s) covering it; shift+mask to get the local index; then `id = brick_palettes[palette_base + idx]`.
  **Cost: 1–2 fetches + a shift + a mask + 1 palette indirection per step** — a handful of ALU ops, fully
  DDA-friendly (this is exactly the Teardown / `voxel.wiki` palette-compression path, proven in production
  GPU voxel renderers). Sub-byte indices that straddle a `u32` boundary need a 2-word read; pad each brick's
  index stream to a `u32` boundary so a cell never straddles a brick.
- **Interaction with R1:** uniform = the degenerate k=1 case (0-bit indices); implement R1 first, then R2 makes
  the k=2..16 strata bricks cheap. Together they cover populations 1 and 2 of §3.
- **Maps onto:** new `brick_palettes` + bit-packed `voxel_indices` buffers in `GpuBrickPatch`; `GpuBrickMeta`
  gains `index_bits` + `palette_base`; `pack_resident_set` computes per-brick palettes (a 512-entry histogram
  per brick — cheap); `dda_brick` bit-extract path.
- **Honesty:** this is the most *invasive* shader change (the per-cell fetch is in the hottest loop). Gate it
  behind R1 and measure the per-step cost on the perf rig before committing — but the literature (Teardown ships
  this at scale) says it's a clear win for strata-heavy worlds.

### R3 — Brick-level dedup (identical bricks stored once)  ·  Tier A  ·  effort: M  ·  impact: high (interiors)
**The "many identical interior bricks" win** — a 1-level DAG at brick granularity (not voxel granularity, so it
stays traceable).

- **Layout:** content-hash each packed brick's voxel/index slice; intern identical slices in a `HashMap<hash,
  voxel_offset>`. Multiple `GpuBrickMeta`s then share one `voxel_offset`. The AABB/meta stay per-brick (each
  brick has its own world position); only the *voxel data* is shared. This is the cheap, safe slice of the DAG
  idea — **dedup the leaf payload, keep the spatial index flat.**
- **Why it stays Tier A:** the DDA already addresses voxels purely through `meta.voxel_offset`; pointing two
  metas at the same offset is invisible to the shader — **zero shader change**. The trace never knows.
- **Synergy:** with R1, all uniform bricks of the same `(block, neighbours-solid)` collapse to *one* shared
  zero-byte entry (or a single shared slice if you keep a degenerate slice). With R2, identical strata patterns
  (very common across a flat region) share one index slice.
- **Edit caveat:** a cut into a shared brick must **copy-on-write** (fork its slice) — exactly the COW the
  instancing plan already specifies (`VOXEL_INSTANCING_PLAN.md §2.3`). Dedup is read-only sharing; first write
  un-shares. Robust by construction.
- **Maps onto:** `pack_resident_set` interning pass (CPU) / the GPU pool's allocator can dedup at fill time;
  `GpuBrickMeta.voxel_offset` semantics unchanged.

### R4 — Occupancy bitmask + compacted solid list (surface bricks)  ·  Tier A  ·  effort: M  ·  impact: medium
For the **varied surface** bricks (population 3), store the `512`-bit occupancy mask (already computed CPU-side)
+ only the ids of *solid* cells (prefix-sum/popcount → dense index). Air cells cost 1 bit, not a full id.

- **Decode:** per cell, read the mask word, `popcount` the bits below the cell, index the compacted id array.
  This is the sparse-64-tree / BrickMap leaf trick. Slightly more ALU than R2 and only helps bricks with lots
  of air (surface), so it's **lower priority than R1–R3** which target the dominant interior mass. Worth it once
  interiors are nearly free and the surface shell becomes the budget.
- **Maps onto:** `occupancy` is already on `Brick`; promote a compact form to VRAM + a compacted id buffer.

### R5 — `.vxo` on-disk compression (RLE → palette → zstd; optional DAG)  ·  Tier B  ·  effort: M  ·  impact: high (disk)
The on-disk `BRIK` chunk (`VOXEL_INSTANCING_PLAN.md §1.5`) is **not traced**, so compress it hard and expand to
the Tier-A layout (R1–R4) on load.

- **Order of operations on disk:** (1) the sparse brickmap (only non-empty bricks); (2) each brick as
  **per-brick palette + bit-packed indices** (R2's format — disk and VRAM can share it, minimizing load-time
  transcode); (3) **brick dedup** (R3) so identical bricks serialize once; (4) **RLE** the strata bands (a
  strata brick is literally run-length-friendly along Y); (5) wrap the whole chunk in **zstd** (or Brotli for
  max ratio if load time allows). Uniform bricks serialize as one id (R1).
- **Optional DAG for the asset:** for a *static* `.vox`-imported object (a tree, a building), an offline **SVDAG
  / SSVDAG** pass on the `.vxo` is a legitimate Tier-B win (10–100×; the headline numbers in the table) because
  it's decoded to bricks on import. **Reserve the DAG for immutable imported assets**, not the streamed world
  (which is GPU-generated and never round-trips through disk anyway). This matches Aokana's "store as SVDAG,
  decode to a traceable pool" exactly.
- **Maps onto:** `BRIK` chunk encoder/decoder; reuse R2's per-brick format so disk↔VRAM transcode is trivial.

---

## 5. What NOT to adopt (and why)

- **A full SVDAG/SSVDAG as the LIVE (Tier-A) structure — NO.** It would wreck the HW-RT DDA: the per-voxel
  query becomes a pointer-chasing octree/DAG descent (variable-bit child pointers, symmetry transforms),
  which is the opposite of the flat O(1) `voxels[offset + index]` fetch the in-shader DDA needs. It also can't
  be addressed by `meta.voxel_offset`, and it doesn't compose with the AABB-per-brick BLAS (the BLAS hands us a
  brick + a local cell range; a DAG wants to *be* the whole spatial index). **Viable as Tier-B only** (R5's
  optional asset DAG, and as Aokana's streamed/stored chunk form that is *decoded* to bricks). Don't put it on
  the trace path.
- **NanoVDB as the resident format — NO (for our data).** Its compression is float/scalar quantization (density,
  SDF), not discrete palette ids; we'd gain nothing on block ids and lose our brick/halo/clipmap machinery. Borrow
  its *philosophy* (pointer-free, memcpy-able, GPU-random-access) for the Tier-A layout, not the structure.
- **A deep sparse 64-tree as the per-voxel store — NO.** The 64-tree is excellent as a **top-level space-skip
  acceleration** (and our clipmap + BLAS already provide that role), but its *leaves* still want to be flat
  bricks for the DDA. Adopting it wholesale would duplicate the spatial-index job the BLAS already does. Keep
  flat bricks at the leaf; if we ever want better empty-space skipping than the BLAS gives, revisit a shallow
  64-tree *above* the bricks, not inside them.
- **Per-voxel `u8` in VRAM to halve the buffer — TEMPTING BUT SKIP in favor of R1–R3.** Narrowing the resident
  id from `u32` to `u16`/`u8` is a flat 2–4× and is real, but it (a) caps the palette and (b) is dominated by
  R1+R3 which make most interior bricks cost ~0 *regardless* of per-voxel width. Do the structural wins first;
  a `u16` index buffer falls out of R2 anyway (indices, not ids, are stored).

---

## 6. How this composes with the GPU pool + `.vxo` (no rework)

- **GPU brick pool (`GPU_VOXEL_WORLDGEN_PLAN.md`):** R1/R2/R3 *increase effective pool capacity* without
  changing the pool's fixed-capacity, degenerate-AABB-for-free-slots design. A pool "slot" still has a meta +
  AABB; the *voxel* sub-pool it draws from shrinks dramatically (uniform = 0 bytes, dedup = shared, strata =
  bit-packed). The GPU voxelizer writes the **packed** form directly into the pool (it already knows each
  brick's content, so it can emit uniform/palette/dense at write time) — no CPU readback, consistent with the
  pivot. The haloed-brick seam fix is preserved (a dense brick still stores its halo; a uniform buried brick
  needs none, per R1's caveat).
- **`.vxo` BRIK chunk (`VOXEL_INSTANCING_PLAN.md §1.5`):** R5 *is* the BRIK chunk's compression spec. Sharing
  R2's per-brick palette format between disk and VRAM means the loader expands a `.vxo` brick to a resident
  brick with minimal transcode. The optional DAG sits at the asset-import boundary, decoded once.
- **Instancing COW (`§2.3`):** R3 (dedup) is the same mechanism as instance sharing — identical bricks share a
  slice; a cut forks it. One COW rule serves both.

---

## 7. Phased adoption plan

Each phase: independently shippable, zero-warning across all three feature builds, benchmarked before/after on
the extended perf/residency rig (§8), reviewed by ≥2 adversarial reviewers vs. the GPU ground-truth harness
(the standing per-stage QA mandate). Ordered by impact/effort and by risk (least-risky, biggest-win first).

- **Phase 1 — Uniform-brick collapse into VRAM (R1).** `GpuBrickMeta` gains a `kind`/`uniform_id`; packers skip
  voxel emit for fully-buried uniform bricks; `dda_brick` branches to the no-fetch uniform path. *Acceptance:*
  Cornell + a worldgen region render pixel-identical (GPU==CPU oracle green); the resident voxel buffer for a
  solid-interior region drops by the uniform fraction (report MB before/after); no perf regression (likely a
  small *win*). *Risk:* low — additive flag, degenerate to today when no brick is uniform.

- **Phase 2 — Brick-level dedup (R3).** CPU/GPU intern identical voxel slices; metas share offsets; COW on edit.
  *Acceptance:* a flat strata region shows N_distinct ≪ N_bricks; cutting one shared brick forks only that
  slice and leaves siblings byte-identical; render unchanged. *Risk:* low-medium — shader unchanged; the work is
  the interning pass + the COW edit path (reuses the instancing COW design).

- **Phase 3 — Per-brick palette + bit-packed indices (R2).** New `brick_palettes` + bit-packed index buffer;
  meta `index_bits`/`palette_base`; `dda_brick` bit-extract. *Acceptance:* strata bricks shrink to the §3
  numbers; GPU==CPU oracle green at every bit width (1/2/4/8); per-step cost measured on the rig within budget.
  *Risk:* medium-high — hottest-loop shader change; gate behind a flag, A/B vs. the dense path, diff frames.

- **Phase 4 — Occupancy + compacted solid list for surface bricks (R4).** Only after 1–3 make interiors cheap
  and the surface shell is the budget. *Acceptance:* surface-brick voxel bytes drop by the air fraction; oracle
  green. *Risk:* medium — second hot-loop decode path; lowest priority.

- **Phase 5 — `.vxo` BRIK compression + optional asset DAG (R5).** Tier-B only; share R2's format disk↔VRAM;
  zstd wrapper; offline SVDAG for static imported assets. *Acceptance:* `.vxo` size vs. raw dump (report ratio);
  load expands to a byte-identical resident brick set vs. generating it live. *Risk:* low — off the trace path.

---

## 8. How to MEASURE it (extend the perf/residency harness — benchmark-every-delivery)

The harness exists: `tests/voxel_worldgen_perf.rs` (streamed worldgen fly-through + a PACK-cost stage) and
`tests/voxel_sponza_pack.rs` / `tests/voxel_sponza_residency.rs` (pack/residency asserts). Extend with a
**storage-bytes report**, run before/after each phase, numbers in the commit message.

Add a `report_storage_bytes(patch: &GpuBrickPatch)` helper (CPU-only, runs anywhere — no GPU device needed)
that prints, for a representative resident set (cold-fill of the shipping worldgen region at the 60k cap):

| Metric | How |
|---|---|
| **resident brick count** | `patch.brick_count()` |
| **voxel buffer bytes** | `patch.voxels.len() * 4` (today) / new packed buffer sizes |
| **meta + AABB bytes** | `brick_count * (size_of::<GpuBrickMeta>() + size_of::<GpuBrickAabb>())` |
| **bytes / brick (mean)** | total ÷ brick_count — the headline number |
| **uniform-brick fraction** | count bricks where `is_uniform_solid()` (R1's win predictor) |
| **distinct-slice fraction** | after R3, `distinct_slices ÷ brick_count` (dedup ratio) |
| **mean index_bits** | after R2, average `ceil(log2 k)` over bricks (strata-compression predictor) |
| **palette buffer bytes** | `brick_palettes.len() * 2` (R2) |
| **total VRAM est.** | sum of all GPU buffers — the single number each phase must reduce |

Acceptance gate per phase: the **total VRAM est.** for the solid-interior worldgen region drops by at least the
predicted fraction (§3), and the GPU ground-truth oracle (`voxel_raytrace_gpu` + the GI/seam GPU tests) stays
byte/pixel-identical. Also report the **per-step DDA cost** delta via the existing Nsight path
(`nsight-shader-profiling` memory) for R2/R4, since those add hot-loop work — confirm the *fewer fetches* of R1
and the *shared slices* of R3 offset the *bit-extract* of R2, so the net trace time is flat-or-better while VRAM
falls. Measure, don't guess.

---

## 9. Summary

| # | Recommendation | Tier | Effort | Expected VRAM reduction |
|---|---|---|---|---|
| R1 | Uniform-brick collapse into VRAM | A | S | uniform bricks: 4 KB → ~8 B (≈500×); dominant interior mass → ~0 |
| R3 | Brick-level dedup (COW on edit) | A | M | identical bricks → one shared slice; interior fill → amortized ~0 |
| R2 | Per-brick palette + bit-packed indices | A | M | strata bricks: 4 KB → ~258 B (≤4 ids) … ~690 B (32 ids), 6–15× |
| R4 | Occupancy + compacted solid list | A | M | surface bricks: air cells → 1 bit (medium, later) |
| R5 | `.vxo` zstd + palette/RLE + optional asset DAG | B | M | disk: 3–20× (RLE/zstd), 10–100× (asset DAG) — off-trace |

Together, R1+R2+R3 turn the **2–4× cost of solid interiors into ≈0** — the resident voxel buffer drops from
~240 MB (60k × 4 KB, content-blind) toward the low tens of MB, dominated by the surface shell, not buried fill.
All three are **Tier A (GPU-DDA-traceable)** and keep the hot DDA loop O(1) — R1 makes it *faster* (no fetch),
R3 is *invisible* to the shader, R2 adds a few ALU ops per step (the proven Teardown/`voxel.wiki` path). The
big-ratio DAG methods (SVDAG/SSVDAG/Aokana-chunk) are **Tier B only** — great for `.vxo`/streaming, decoded to
bricks before the trace, never on it.

**Skip:** SVDAG/SSVDAG/NanoVDB/deep-64-tree as the *live* structure — they break the O(1) GPU-DDA fetch (§5).
