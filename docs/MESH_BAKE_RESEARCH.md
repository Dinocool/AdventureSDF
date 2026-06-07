# SDF → Chunked Mesh Bake — Research, Tradeoffs & Decisions

> **Status:** decision document for the render-pipeline pivot (SDF raymarch → baked chunked
> meshes). Generated 2026-06-07 from a fan-out of 5 research agents + 3 adversarial challenge
> agents (each required to read real sources and cite them). Every non-obvious claim below is
> tagged with a source URL. **Verified** = read in a fetched page; **inferred** = reasoned from
> verified facts.

## DECISION (locked 2026-06-07)

After the research + adversarial round, the user locked the v1 direction:

- **Algorithm: Surface Nets via off-the-shelf [`fast-surface-nets`](https://github.com/bonsairobo/fast-surface-nets-rs)**
  (+ `ndshape`). Rationale: use a mature, proven crate rather than building DC/CMS ourselves; our
  atlas (quantized distance + palette) maps onto it 1:1.
- **Sharp edges: NOT required.** SN's edge-rounding is accepted — so no QEF/Hermite vertex placement
  in v1 (kept only as an optional future lever; the gradient atlas stays useful for shading normals).
- **Cross-LOD crack-free via SKIRTS.** The user accepts skirts as satisfying "true crack-free."
  This resolves SN's one weak area (skirts are algorithm-agnostic and compose trivially with SN) —
  *with the known caveat* that skirts are a visual cover, not a true stitch: they can leak if too
  short, add overdraw, and produce a shading discontinuity at the fold
  ([godot_voxel #63](https://github.com/Zylann/godot_voxel/issues/63)). Mitigate by tuning skirt
  length per LOD and keeping the 2:1-ring / per-face-coarser-neighbor invariants (§2) so skirts only
  go where a face borders a coarser ring.
- **Compute: CPU-async** meshing on `AsyncComputeTaskPool` (§4); re-mesh only edited chunks (+ pad
  neighbours); upload vertex/index buffers and rasterize.

**v1 shape:** per resident brick (8³ + 1-voxel neighbour pad) → read back baked distance+material →
`surface_nets()` → positions/normals/indices; carry our 4-id palette + per-corner material weights
onto vertices and shade via triplanar splat (bonsairobo's approach,
[Smooth Voxel Mapping](https://bonsairobo.medium.com/smooth-voxel-mapping-a-technical-deep-dive-on-real-time-surface-nets-and-texturing-ef06d0f8ca14));
same-LOD seams are free (1-voxel pad), cross-LOD seams hidden by skirts.

*Note:* depend on `fast-surface-nets` + `ndshape` directly; `bevy-sculpter` (the Bevy 0.18 SN crate)
is a useful integration reference but is "early development, expect breaking changes," so don't
take a hard dependency on it. Still run the §0 premise profile before scaling up.

The sections below are retained as the substantiating research.

---

## 0. Why this pivot, and the one thing we must check first

We render an SDF brick atlas by live raymarching (deferred G-buffer + screen-space RC / DDGI GI).
The pivot: **bake the SDF into chunked meshes that rasterize normally**, because mesh
rasterization is expected to scale better than per-pixel raymarch.

**The premise is industry-plausible but unmeasured for our scene — validate before committing weeks.**
- Real-world validation it *can* work: **ALICE-SDF** bakes an SDF into Nanite-style hierarchical
  cluster meshes and renders "at polygon speed… no need to run raymarching every frame"
  ([ALICE-SDF](https://github.com/ext-sakamoro/ALICE-SDF)). The general consensus is that direct
  SDF rendering is "less performant and less widely adopted than mesh-based renderers"
  ([HN 27818857](https://news.ycombinator.com/item?id=27818857)).
- **Counter-evidence:** raymarching has **zero overdraw and zero geometry memory**; meshing *adds*
  overdraw, vertex/index memory, a bake step, and the cross-LOD crack problem. "Scales better" is
  asserted, not benchmarked for our content. *(inferred)*
- **Action item P0:** before building the full pipeline, profile primary-visibility raymarch cost
  vs. projected mesh+overdraw cost on `gallery.scene` and `stress.scene`. If raymarch primary
  visibility is *not* a measured bottleneck at target scale, the pivot's payoff is unproven.

## 1. Hard constraint from our platform: wgpu rules out two whole families

This gates the entire option space and was the biggest blind spot in the first-pass research.

- **wgpu has no mesh shaders** ([gfx-rs/wgpu #3018](https://github.com/gfx-rs/wgpu/issues/3018)) — Nanite-style
  *mesh-shader* geometry is off the table.
- **wgpu has no 64-bit texture atomics** — Bevy's own virtual-geometry author hit exactly this wall
  ("wgpu once again lacks support for a needed feature, this time 64-bit texture atomics")
  ([jms55, Virtual Geometry in Bevy 0.14](https://jms55.github.io/posts/2024-06-09-virtual-geometry-bevy-0-14/)).
  This blocks **both**:
  - **Compute-rasterized point splatting** (our roadmap's R10 / Dreams-style) — the fast technique
    (Schütz, ~50 B points/s) *requires* 64-bit atomics ([arXiv 1908.02681](https://arxiv.org/abs/1908.02681)).
  - **Nanite-style software rasterization** (the >90% path of real Nanite).

**Conclusion:** on current wgpu the only viable "rasterized" representation is **hardware-rasterized
polygon meshes**. This *validates choosing meshing over splatting* for now. (Aside: shipped *Dreams*
actually **raymarches** its solid hulls and only splats painterly "fluff" — pure splatting could not
opaquely cover hard surfaces, producing see-through buildings — so R10-as-primary is counter-evidenced
by its own precedent: [aras-p](https://aras-p.info/blog/2023/09/05/Gaussian-Splatting-is-pretty-cool/),
[andrewkchan](https://andrewkchan.dev/posts/lit-splat.html).)

## 2. Decision LOCKED by the user

**Full cross-LOD crack-free seams from the start** (not deferred). The adversarial round argued for
deferring the seam *mesher* while keeping the data-model invariants; the user has chosen to keep full
crack-free in v1. This document therefore treats crack-free cross-LOD as a v1 requirement — which, as
§5 shows, **substantially constrains the algorithm choice**.

Regardless of the rest, design these **invariants in now** (expensive to retrofit — Gildea: mesh gen
"only works when the nodes are of uniform size, this is often not the case when generating seams"
[OpenCL DC](http://ngildea.blogspot.com/2015/06/dual-contouring-with-opencl.html)):
- **Strict 2:1 ring ratios** between adjacent clipmap LODs (every seam algorithm assumes ≤1-level /
  factor-2 neighbor deltas).
- **Per-face coarser-neighbor flags** (6 bits/chunk): which faces border a coarser ring.
- **Neighbor-aware dirty propagation** on edit (extend the existing BVH-refit incremental path to
  seam adjacency).
- **Retain Hermite/gradient data at chunk borders** so the seam mesher has its input without a re-bake.

---

## 3. Algorithm tradeoffs

All methods consume our brick atlas (per-voxel quantized signed distance + a gradient/normal =
**Hermite** atlas + 4-id material palette). Our content is **sharp CSG/primitive geometry** (boxes,
booleans, hard 90° edges).

### 3.1 Summary table

| Method | Sharp features | Cross-LOD crack-free | Rust maturity | Perf | Fit for us |
|---|---|---|---|---|---|
| **Surface Nets** (naive) | ✗ rounds/bevels 90° edges | same-res only; **no** cross-LOD natively | **High** — `fast-surface-nets` mature, `bevy-sculpter` on Bevy 0.18 | ~20 M tris/s/core (SIMD) | 5/10 |
| **Surface Nets + constrained QEF** | ~ partial (QEF vertex relocation) | inherits SN's weak cross-LOD story | Med — QEF is a hand-add on the SN base | ≈ SN + QEF cost | 6/10 |
| **Dual Contouring** (Hermite/QEF) | ✓ best | ✓ native mixed-res via Gildea seam-octree — **but never shipped robustly** | **None production** (`isosurface` alpha; `ALICE-SDF` no cross-LOD seams) | QEF per cell; heaviest | 6/10 (quality 9, risk high) |
| **Manifold DC / Dual Marching Cubes** | ✓ | ✓ adaptive-octree crack-free (not clipmap-streaming) | None in Rust | + topology pass | research-grade |
| **Cubical Marching Squares** | ✓ + manifold | adaptive within fixed-res field; **clipmap continuous-LOD unsolved** | None (academic) | "very tedious"; no benchmarks | research-grade |
| **Transvoxel** (MC + transition cells) | ✗ MC rounds edges | ✓ **purpose-built, shipped** (C4, Godot Voxel Tools) | Med — `transvoxel` 2.0 crate, "experimental", Bevy 0.17 examples | MC-speed; transition cells "significantly more expensive" | 5/10 |

### 3.2 Surface Nets (`fast-surface-nets-rs`)
- **What it is:** a *dual* mesher on a uniform grid — one vertex per sign-changing cell, placed by
  **averaging** edge crossings (it deliberately discards Hermite/gradient data)
  ([docs.rs/fast-surface-nets](https://docs.rs/fast-surface-nets)).
- **Perf:** "~20 million triangles per second on a **single core** of a 2.5 GHz Core i7", SIMD via
  `glam`, tiny LUTs ([github](https://github.com/bonsairobo/fast-surface-nets-rs)). *(Single-core CPU
  number on old hardware — you parallelize per chunk; not a GPU throughput figure.)*
- **Maturity:** used by `block-mesh`, `transvoxel`, `bevy_voxel_world`; **`bevy-sculpter` v0.18 pairs
  it with Bevy 0.18** and is the closest drop-in reference ([lib.rs/bevy-sculpter](https://lib.rs/crates/bevy-sculpter)).
- **Quality cost (decisive for us):** structurally **rounds/bevels every 90° edge** — bonsairobo
  himself points at DC as the fix and notes he never implemented it
  ([Medium](https://bonsairobo.medium.com/smooth-voxel-mapping-a-technical-deep-dive-on-real-time-surface-nets-and-texturing-ef06d0f8ca14)).
  We'd be paying to store a Hermite atlas and then throwing the exact data away. We found **no
  shipped sharp-edged/CSG game on plain SN**.
- **Cross-LOD:** same-res chunks tile seamlessly (1-voxel pad; "faces not generated on positive
  boundaries"). **Different-res neighbors are unsolved in SN itself** — every watertight add-on is
  either non-dual (Transvoxel, fights SN topology) or a DC technique (seam-octree). Skirts are a
  visual hack (§5).

### 3.3 Surface Nets + constrained QEF
- The adversarial counter-proposal: keep SN's robust uniform-grid topology + same-res stitching, and
  borrow **only** DC's QEF vertex placement (constrained + center-biased) to recover sharp edges —
  *without* the octree/seam-octree/Manifold-DC stack ([boristhebrave](https://www.boristhebrave.com/2018/04/15/dual-contouring-tutorial/)).
- **Caveat:** QEF on our **quantized** normals is ill-conditioned — near-colinear normals push the
  vertex outside the cell and it gets clamped to the mass-point, i.e. **back toward the SN average**
  ([Gildea, Implementing DC](https://ngildea.blogspot.com/2014/11/implementing-dual-contouring.html)).
  Every clamp/bias erodes the sharp-edge gain. Sharp recovery is partial, not DC-grade.
- **Cross-LOD:** still SN's weak spot — this option does **not** by itself satisfy the locked
  full-crack-free requirement; it needs a seam strategy bolted on (§5), and Transvoxel transition
  cells do **not** co-vertex with dual topology.

### 3.4 Dual Contouring (Hermite/QEF + Gildea seam-octree)
- **Quality ceiling for us:** consumes our Hermite atlas directly; reconstructs sharp CSG edges/corners
  (one vertex/cell at the QEF minimum) — a perfect box is 2 polys/face ([Ju 2002](https://www.cs.rice.edu/~jwarren/papers/dualcontour.pdf)).
- **Cross-LOD is elegant in theory:** Gildea's seam-octree gathers seam nodes from up to 7 neighbors
  into a temporary octree and contours it; mixed-resolution leaves "just work" with no special
  transition cells ([Seams & LOD](http://ngildea.blogspot.com/2014/09/dual-contouring-chunked-terrain.html)).
- **But the risk is severe and well-documented (the adversarial round's strongest finding):**
  - **Never shipped robustly.** Gildea's engine `leven` shipped with **holes *and* overlapping
    geometry at the LOD seams** and was "never fully polished or completed"; the `java-leven` fork
    exists *specifically* to fix those seam holes
    ([java-leven](https://github.com/proton2/java-leven/blob/master/README.md)); Gildea's *next*
    project `fast_dual_contouring` **deleted the octree and LOD seams entirely**
    ([fast_dual_contouring](https://github.com/nickgildea/fast_dual_contouring)).
  - **No production Rust DC.** `isosurface` is alpha, uniform-grid only, no octree/LOD, and warns
    feature placement is "very sensitive to input data quality"
    ([isosurface](https://github.com/swiftcoder/isosurface)). `ALICE-SDF` does DC with a
    manifold-via-repair pass but **decimation LOD, not cross-LOD seams**.
  - **Structural defects on our content.** Non-manifold vertices + self-intersection occur exactly
    on thin walls / near-touching primitives — common in dense CSG
    ([boristhebrave](https://www.boristhebrave.com/2018/04/15/dual-contouring-tutorial/)). Manifold
    DC fixes the manifold half (extra cell-splitting pass) but not self-intersection.
  - **Editor-hostile.** The seam method couples a chunk's mesh to up to 7 neighbors → **non-local
    invalidation** on every edit, fighting the local-edit architecture (BVH refit) we deliberately built.

### 3.5 Transvoxel (Marching Cubes + transition cells)
- **The shipped answer to crack-free cross-LOD.** Transition cells stitch a full-res block face to a
  half-res neighbor with shared vertices on both sides — crack-free by construction; 9 hi-res samples
  → 512 cases → 73 classes ([transvoxel.org](https://transvoxel.org/)). Shipped in C4 Engine and the
  leading editable-LOD open-source engine **Godot Voxel Tools**
  ([voxel-tools](https://voxel-tools.readthedocs.io/en/latest/smooth_terrain/)).
- **Rust crate:** `transvoxel` 2.0 (MIT/Apache, "experimental", ~80 dl/mo), supports multiple
  transition faces per block, Bevy 0.17 examples ([lib.rs](https://lib.rs/crates/transvoxel),
  [Gnurfos/transvoxel_rs](https://github.com/Gnurfos/transvoxel_rs)).
- **Constraints:** adjacent blocks may differ by **at most one LOD** (factor-2) — our clipmap
  satisfies this *by design*. Cannot "flip" a transition face without re-extracting the block.
- **Quality cost (decisive against, for us):** it is **Marching Cubes** — vertices live on cell edges,
  so **sharp CSG edges round to fillets ≈ one voxel** and our Hermite atlas is unused. Godot's docs
  call the result inherently "smooth/rounded" with residual seam "little steps"
  ([voxel-tools](https://voxel-tools.readthedocs.io/en/latest/smooth_terrain/)).

### 3.6 Honorable mentions (research-grade, not v1)
- **Cubical Marching Squares (CMS):** the only method claiming manifold + sharp + adaptive from
  Hermite data — but it "assumes the underlying Hermite data is fixed resolution", and clipmap
  continuous-LOD is "challenging and unresolved"; academic only
  ([tammearu](https://blog.tammearu.eu/posts/cms/), [Ho 2005](https://www.csie.ntu.edu.tw/~cyy/publications/papers/Ho2005CMS.pdf)).
- **Dual Marching Cubes / Manifold DC:** crack-free adaptive within an octree, not clipmap-streaming;
  no Rust impl ([Schaefer/Ju/Warren](https://people.engr.tamu.edu/schaefer/research/dualsimp_tvcg.pdf)).

---

## 4. Compute: where meshing runs (CPU vs GPU)

### 4.1 Summary

| | CPU async meshing | GPU compute meshing |
|---|---|---|
| Readback | reads baked field back per (re)bake — but **amortized** (bake once, reuse many frames), overlaps GPU work | none — field already GPU-resident |
| Portability | none at risk | **stream compaction = least-portable wgpu corner** |
| Debuggability | high (Rust debugger, deterministic, unit-testable) | low (no tooling; nondeterministic compaction) |
| Proven pattern | **yes** — `AsyncComputeTaskPool` (vx_bevy, bevy_voxel_world) | bespoke; GPU Gems 3 MC; UnityVoxelMeshGPU |
| Seams | stay here regardless | impractical on GPU (see below) |
| Effort | low (crate + async pattern) | multi-week R&D |

### 4.2 Why CPU-first is recommended
- **GPU variable-length output needs stream compaction / prefix-sum**, and the efficient form
  (decoupled look-back) **cannot run on WebGPU** (missing device-scope atomic barrier); atomic
  variants have **documented hardware failures** (AMD 5700 XT) and forward-progress hangs on
  Apple/ARM ([Raph Levien, portable prefix sum](https://raphlinus.github.io/gpu/2021/11/17/prefix-sum-portable.html)).
- **The readback objection is weak for a clipmap:** chunks bake once and are reused for many frames,
  so transfer is amortized and overlaps other GPU work
  ([tillcode wgpu readback](https://tillcode.com/rust-wgpu-compute-minimal-example-buffer-readback-and-performance-tips/)).
- **The proven Bevy pattern** is CPU meshing on `AsyncComputeTaskPool`, polled across frames to avoid
  stutter, re-meshing only edited chunks — shipped at interactive rates
  ([vx_bevy meshing.rs](https://github.com/Game4all/vx_bevy/blob/master/src/voxel/world/meshing.rs),
  [bevy_voxel_world](https://github.com/splashdust/bevy_voxel_world)).
- **Decisive for our locked cross-LOD requirement:** the canonical GPU-DC author (Gildea) moved DC to
  the GPU (OpenCL) but **kept seam generation on the CPU on purpose** — non-uniform seam nodes,
  per-LOD lookup explosion, tiny per-seam workloads
  ([OpenCL DC](http://ngildea.blogspot.com/2015/06/dual-contouring-with-opencl.html)). On wgpu
  (weaker than OpenCL) full GPU seams are worse. **Seams must run on CPU.**

### 4.3 When GPU meshing wins (the steelman)
- Per-frame churn (destruction/fluids) or huge streamed worlds where the readback round-trip is
  per-frame, not per-edit. A *later, localized* GPU mesh of a single edited chunk's **interior**
  (no seams, multi-pass scan for portability) is a legitimate optimization once profiling demands it
  — never a v1 architecture, and never for seams.

**Recommendation:** **CPU meshing on `AsyncComputeTaskPool` for v1** (the "reference" path *is* the
shipping path). Indirect-draw the resulting GPU buffers. Revisit GPU interior meshing only if a
dynamic brush-preview profile demands it.

---

## 5. Cross-LOD seam strategies (required in v1)

| Strategy | Pairs with | True watertight? | Notes |
|---|---|---|---|
| **Transvoxel transition cells** | Marching Cubes only | ✓ | purpose-built, shipped; ≤1-LOD; doesn't compose with *dual* (SN/DC) topology |
| **DC seam-octree (Gildea)** | Dual Contouring / SN | ✓ in theory | native mixed-res; bug-prone, never shipped clean, 7-neighbor coupling |
| **Skirts / flanges** | any | ✗ visual hack | godot_voxel still leaks LOD-border holes even with skirts ([#63](https://github.com/Zylann/godot_voxel/issues/63)) — **disqualified by "true crack-free"** |
| **Geomorphing** | DC/SN | ~ | residual corner gaps + "pop" even when mature ([dexyfex](https://dexyfex.com/2016/07/14/voxels-and-seamless-lod-transitions/)) |
| **Geometry-clipmap transition region** (Losasso–Hoppe) | heightfield (2.5D) | ✓ for 2.5D | degenerate strips + α-morph; concept transfers, math doesn't generalize to full 3D iso-surfaces ([GPU Gems 2 ch.2](https://developer.nvidia.com/gpugems/gpugems2/part-i-geometric-complexity/chapter-2-terrain-rendering-using-gpu-based-geometry)) |

**Key consequence:** *the locked "full crack-free from the start" requirement effectively forces the
algorithm into one of two coherent end-to-end families* (skirts are out; mixing topologies
reintroduces seams):
- **Transvoxel ↔ Marching Cubes** (rounded edges, lowest risk, shipped), **or**
- **Seam-octree ↔ Dual Contouring** (sharp edges, highest risk, never shipped clean).

Our clipmap's fixed factor-2 ring adjacency is a gift to **either** family (the ≤1-LOD invariant is
free; a DC seam only ever spans two adjacent ring resolutions).

---

## 6. The core decision (yours to make — tradeoff framed)

Because cross-LOD-crack-free-now is locked, the choice collapses to a single value judgment:

> **Are sharp CSG edges non-negotiable, or is robust crack-free-now non-negotiable?**

- **If sharp edges win → Dual Contouring family**, accepting it is a from-scratch build with no Rust
  prior art and the worst shipped track record. De-risk by: uniform-grid DC interiors (lean on the
  clipmap for LOD rather than an adaptive octree), **constrained/center-biased QEF** for quantized
  normals, a **CPU** seam pass (seam-octree), and a manifold post-pass only where thin walls demand it.
- **If robust-now wins → Transvoxel**, accepting rounded edges (our Hermite atlas goes unused);
  lowest risk, shipped, Rust crate, clipmap satisfies its ≤1-LOD rule for free. Sharp-edge recovery
  becomes a *later* research graft (CMS-style Hermite vertices), not a v1 dependency.
- **Middle (SN + constrained QEF)** gives partial sharpness with a mature base, but its cross-LOD
  seam story is the weakest and pulls you back toward one of the two families above to actually be
  crack-free — so it is **not** a clean fit for the locked requirement.

**Author's recommendation:** given the project's stated bias toward the *correct, best-in-class*
result and the fact that we *already store the Hermite data*, the quality-maximizing target is the
**Dual Contouring family** — but **only after** (a) the §0 premise profile confirms the pivot pays,
and (b) a 1–2 day spike meshing one `gallery.scene` brick with `fast-surface-nets` on our **actual
quantized field** to see how bad edge-rounding really is on our content. If the spike shows SN
rounding is acceptable *or* DC's seam risk proves intractable, fall back to **Transvoxel**. Do not
commit the multi-week DC build before both checks. Meshing runs **CPU-async**; seams **always CPU**.

---

## 7. Open decisions
1. ~~Algorithm family~~ — **RESOLVED: Surface Nets (`fast-surface-nets`) + skirts, CPU-async.** See the
   DECISION block at the top.
2. **Run the §0 premise profile before scaling up?** (still recommended — confirm the mesh path
   actually beats raymarch primary visibility on `gallery`/`stress` before building out LOD + skirts).

## 8. Sources
*(every URL below was fetched and read by a research/challenge agent; a few secondary pages returned
403 and are flagged inline where used as corroboration only.)*

Surface Nets / general: https://github.com/bonsairobo/fast-surface-nets-rs ·
https://docs.rs/fast-surface-nets · https://lib.rs/crates/bevy-sculpter ·
https://bonsairobo.medium.com/smooth-voxel-mapping-a-technical-deep-dive-on-real-time-surface-nets-and-texturing-ef06d0f8ca14 ·
http://lionelpigou.com/meshing · https://cerbion.net/blog/understanding-surface-nets/

Dual Contouring: https://www.cs.rice.edu/~jwarren/papers/dualcontour.pdf ·
https://www.boristhebrave.com/2018/04/15/dual-contouring-tutorial/ ·
https://ngildea.blogspot.com/2014/11/implementing-dual-contouring.html ·
http://ngildea.blogspot.com/2014/09/dual-contouring-chunked-terrain.html ·
http://ngildea.blogspot.com/2015/06/dual-contouring-with-opencl.html ·
https://github.com/nickgildea/leven · https://github.com/proton2/java-leven ·
https://github.com/nickgildea/fast_dual_contouring · https://github.com/Lin20/BinaryMeshFitting ·
https://github.com/swiftcoder/isosurface · https://github.com/ext-sakamoro/ALICE-SDF ·
https://people.engr.tamu.edu/schaefer/research/dualsimp_tvcg.pdf · https://www.cs.rice.edu/~jwarren/papers/dmc.pdf

Transvoxel: https://transvoxel.org/ · https://docs.rs/transvoxel · https://lib.rs/crates/transvoxel ·
https://github.com/Gnurfos/transvoxel_rs · https://voxel-tools.readthedocs.io/en/latest/smooth_terrain/ ·
https://deepwiki.com/Zylann/godot_voxel/6.2-smooth-meshing-with-transvoxel ·
https://www.binaryconstruct.com/posts/transvoxel-xna/

CMS / other meshers: https://blog.tammearu.eu/posts/cms/ ·
https://www.csie.ntu.edu.tw/~cyy/publications/papers/Ho2005CMS.pdf

CPU vs GPU / wgpu: https://raphlinus.github.io/gpu/2021/11/17/prefix-sum-portable.html ·
https://tillcode.com/rust-wgpu-compute-minimal-example-buffer-readback-and-performance-tips/ ·
https://github.com/Game4all/vx_bevy/blob/master/src/voxel/world/meshing.rs ·
https://github.com/splashdust/bevy_voxel_world ·
https://developer.nvidia.com/gpugems/gpugems3/part-i-geometry/chapter-1-generating-complex-procedural-terrains-using-gpu ·
https://github.com/artnas/UnityVoxelMeshGPU · https://github.com/gfx-rs/wgpu/issues/3018 ·
https://jms55.github.io/posts/2024-06-09-virtual-geometry-bevy-0-14/ ·
https://docs.rs/wgpu/latest/wgpu/struct.Features.html · https://toji.dev/webgpu-best-practices/indirect-draws.html

Seams / LOD: https://github.com/Zylann/godot_voxel/issues/63 ·
https://github.com/Zylann/godot_voxel/issues/24 ·
https://dexyfex.com/2016/07/14/voxels-and-seamless-lod-transitions/ ·
https://developer.nvidia.com/gpugems/gpugems2/part-i-geometric-complexity/chapter-2-terrain-rendering-using-gpu-based-geometry ·
https://hhoppe.com/proj/geomclipmap/

Premise / splatting / Nanite: https://github.com/ext-sakamoro/ALICE-SDF ·
https://aras-p.info/blog/2023/09/05/Gaussian-Splatting-is-pretty-cool/ ·
https://andrewkchan.dev/posts/lit-splat.html · https://arxiv.org/abs/1908.02681 ·
https://github.com/Scthe/nanite-webgpu · https://news.ycombinator.com/item?id=27818857
