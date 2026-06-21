# Voxel-RT Raytrace Path Optimization Plan

Worktree `rt-optimize`. Goal: optimize the voxel-RT raytrace path (the GI/DI megakernel that is ~93% of
the GPU frame). Driven by Nsight per-dispatch profiling + a 4-agent SOTA-comparison audit (ReSTIR, HW
traversal, world-cache, re-flora/OSS). Measurement harness: `rdoc/scripts/rt/` (bench_rt / capture_bistro /
perf_median). **All A/B uses median-of-N** (single Nsight captures swing ±15% from stochastic GI + convergence
state). See [[voxel-rt-optimize-harness]], [[voxel-rt-raymarch-occupancy-bound]].

## Baseline (Nsight, Bistro interior pin)

The raytrace megakernel `voxel_rt_restir_dlss` = 93% of GPU frame; everything else (DLSS resolve/upscale,
tonemap, clustering) < 0.4 ms total. Per-dispatch (clip_half=64, pre-optimization):

| dispatch | time | %GPU | SM% | note |
|---|---|---|---|---|
| gi_restir_p1 | 12.96 ms | 52% | 21.7 | primary trace + GI bounce trace + world-cache query + temporal — worst occupancy |
| gi_restir_p2 | 7.19 ms | 29% | 38.5 | shade: primary RE-trace + sun shadow + DI resolve + GI resolve |
| gi_restir_spatial | 0.19 ms | <1% | | |
| gi_di_p1 | 0.02 ms | <1% | | |

**Bound: SM-bound at low SM throughput (~32%) + ~33% warp occupancy + ~45% warps-inactive = OCCUPANCY/
REGISTER-LIMITED**, not ALU- or memory-bound (DRAM ~3%, L1 hit ~70%). The lever is REDUCING LIVE REGISTER
STATE / SPLITTING KERNELS, not cutting FLOPs (instruction micro-cuts regressed 25-35% in the past).

## THE scaling fix: closest-hit prune (committed 90fa270f + 4954676e)

**Root cause of "perf degrades as brick count grows":** the HW AABB-BVH hands back brick candidates
OUT OF nearest-first order, and `trace()`/`trace_occluded()` DDA-marched EVERY candidate the ray pierced —
including bricks in the rooms BEHIND a wall the ray already hit. Diagnosed with a per-ray candidate-count
heatmap (`TraceResult.cand`, `debug_view 10`): it looked like **x-ray vision** (you could see brick structure
through walls). The wasted behind-wall marches grow with resident brick density ⇒ frame time degrades as the
scene fills. **Fix:** skip a candidate's DDA when its AABB ENTRY is already at/beyond the nearest hit so far
(`t_enter ≥ best_t`); for occlusion, stop after the first occluder. Bit-identical. Heatmap collapsed broad
20-40 (+64 hotspots) → ~5-15. **Nsight median @ clip_half=160 FULLY LOADED (840k bricks): gi_restir_p1
40.9→4.5 ms (~9×), gi_gbuffer 16.0→1.3 ms (~12×).** The 40.9 ms p1 is the original ~49 ms GPU-bound frame.

GOTCHA learned: do NOT commit `generateIntersection` only on new-nearest — the fork's BVH culling needs frequent
commits; restricting it MEASURED ~10× regression. And measurement: Bistro's `.vxo` streaming (256 bricks/frame
default) never converges in a capture window, so captures hit random streaming hitches. Added
`ADVENTURE_STREAM_BUDGET` to load the whole clipmap fast → measure the converged 840k-brick state. At that high
budget the WHOLE-FRAME bench is dominated by per-frame CPU streaming/residency maintenance (a SEPARATE, possibly
real secondary scaling cost — the residency re-scan over all resident bricks); isolate the GPU GI via Nsight
per-dispatch, not whole-frame.

## Done (all committed on `rt-optimize`)

- **Deferred G-buffer for shade (f1e5e79a).** Primary ray was traced TWICE/frame (p1 seeds receiver, p2
  refetches albedo/emissive). `PixelSurface` 32→64 B (+albedo+emissive); shade reads it instead of re-tracing.
  Nsight: gi_restir_p2 7.19→4.19 ms (−42%), whole raytrace −17%. All 4 SOTA agents validated this as the correct
  Solari/kajiya "G-buffer resolve" structure.
- **Dedicated primary G-buffer pass + per-candidate normal elimination (T1) (44359dfe).** (a) standalone
  `gbuffer`/`gbuffer_dlss` kernel traces the primary once → `surfaces_cur`; p1 now carries ONE ray-query (the GI
  bounce); occupancy 21.7→37.1%. (b) `dda_brick` split into a lean `dda_brick_march` (found+t) for the candidate
  loops, with the face-normal reconstruction kept ONLY in the winner re-walk (it was computed per-candidate,
  used once). Bit-identical. **Nsight min-of-3 (clip_half=48): gi_restir_p1 5.52→3.33 ms (−40%), gi_gbuffer
  3.79→3.15 ms (−17%), GI total 12.4→9.83 ms (−21%).**
- **Measurement note:** Bistro's rate-limited `.vxo` streaming never converges in a capturable window, so
  single Nsight frames randomly hit streaming hitches (28× spread). Hitches only ADD time → **MIN-of-N is the
  clean steady-state estimator** (perf_median.ps1 reports it). Median is corrupted by hitches.

## Rejected after analysis

- **T2 (fold winner attrs into candidate loop, kill `brick_hit_at` re-walk).** The re-walk runs AFTER the
  ray-query loop closes — at FULL occupancy (no ray state live) — so its 2nd brick-DDA is already cheap. T2 would
  either move the normal work INTO the low-occupancy traversal loop or hold ~8 extra live registers across it,
  likely REGRESSING the register-bound kernel. Skipped (agent 2 had flagged it as Nsight-gated).

## Validated ALREADY-CORRECT (do NOT redo / regress)

M=1 initial GI candidate; cache-fed bounce radiance (the boil fix); split DI + GI-spatial dispatches; screen
probes (octa atlas); world-space radiance cache (Solari-aligned: PCG+IQ hash, 3-step probe, 2 atomics, EMA+
decay, stochastic 40k cap, NEE+alias-MIS, camera-relative quantization — the last is AHEAD of stock Solari);
unbiased store-before-visibility reservoirs; balance-heuristic NaN-safe MIS. re-flora is behind us on every
rendering axis (software DDA, 1-bounce sun-only, no ReSTIR/cache/LOD); its only transferable asset is the
readback-free GPU BUILD pipeline (→ GPU-worldgen track, orthogonal to GI perf).

## Backlog (ranked by occupancy-impact × confidence) — T1 done, T2 rejected (above)

### T3 — Wavefront split of `restir_p1`: bounce-trace kernel → hit-sample buffer → temporal-merge kernel [agents 1,3,4 — CONSENSUS biggest lever]
The reservoir + `merge_reservoirs` + world-cache-query RNG are live ACROSS the heavy bounce `trace()` in p1.
Split: (1a) ray-gen + bounce trace + cache resolve → write a compact per-pixel sample (pos/normal/radiance/ucw
~32 B); (1b) read sample + reprojected/permuted temporal merge. Each kernel carries only its own register class.
Solari 0.19 split DI/GI world-cache sampling into separate dispatches; kajiya/gvox structure trace as its own
kernel (Megakernels Considered Harmful, HPG 2013). The deferred G-buffer is step 0 of this. Medium effort
(extra buffer + dispatch + barrier); favorable since we're occupancy- not bandwidth-bound. Image-preserving.

### T4 — Skip GI for near-zero-diffuse (smooth-metal / dark-albedo) receivers [agent 1]
Early-out in the 52% kernel for receivers with albedo·(1−metallic) ≈ 0 (Solari 0.19). Needs a diffuse-weight
in the G-buffer. Cheap, image-preserving; value depends on how much specular/dark material the scenes have.

### T5 — Dedupe the two `wc_get_cell_size` calls in `query_world_cache` [agent 3]
The post-jitter LOD recompute (:2744) almost never flips a bin within a ±0.5-cell jitter; compute cell_size
once. Saves a distance/log2/exp2/rand_next chain inside p1. Near-zero risk.

### T6 — Pack the in-flight `Reservoir` (48 B → smaller) [agent 1]
Octahedral-pack `sample_point_world_normal` (3×f32→1×u32), RGB9E5-pack `radiance` (~−40% struct). Fewer live
vector registers in `merge_reservoirs`. ALU for pack/unpack is cheap (we're ALU-headroom-rich). Keep positions
full-precision (Jacobian). Validate within quantization tolerance.

### Lower priority / gated
- T7 GI spatial K=4 pairwise-MIS → 1 neighbour + fuse resolve into shade [agent 1] — changes converged image,
  gate on boil-meter (the K-MIS was added to fight boil).
- T8 DI light-tile/ReGIR presampling + DI spatial reuse [agents 1,4] — di_p1 is <1% of frame; mostly upstream
  alignment + light-heavy-scene divergence; low frame-time priority.
- T9 Fused/packed world-cache hash (12 dependent rounds → ~2) [agent 3] — guard collision rate.
- T10 Ray-coherence sorting before the bounce [agent 4] — screen probes already buy most coherence; gate on
  benchmark (prior cuts regressed). T11 cascaded far-probe LOD [agent 4] — large-scene quality, low priority.
- T12 Track the wgpu-trunk committed-t fix; then drop manual `best_t` from `trace()` [agent 2] — blocked upstream.

## Method note

Each landed step: median-of-N Nsight (converged clip), verify render correctness (screenshot), keep only on a
clear win, commit on the `rt-optimize` branch. Build both feature configs + zero-warnings before any main merge.
