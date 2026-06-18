# Screen-Space Radiance Probes (Lumen-style GI) — Plan of Record

Status: **PLANNING** (gi-boil branch, after checkpoint `edb0637`).
Goal: replace the per-pixel ReSTIR diffuse-GI gather with a **downsampled screen-probe** gather so we get the
boil-killing effect of high M (many *independent* samples) at a *fraction* of the ray cost — the one thing the
whole boil investigation proved actually works (see `docs/GI_BOIL_PLAN.md` §7, 2026-06-18).

## Why this and not more per-pixel ReSTIR

Measured: only **fresh independent samples (M)** kill the boil; spatial/temporal reuse is *correlation-limited*
(blotch plateaus ~0.052 vs M4's 0.036 no matter how many neighbours / how high the cap — raising either makes
it WORSE). Screen probes break the correlation wall: each probe shoots its OWN octahedral set of rays
(independent samples), at ~1 probe / 8×8 px, so we can afford ~64 independent dirs/probe for less than 1
ray/pixel — then interpolate to full res with the *material* kept full-res. This is exactly Lumen's "downsampled
indirect lighting integrated with full-res material data."

## Architecture (mirrors UE5 Lumen ScreenProbeGather; CORRECTED per 2-reviewer design review)

Naming: call these **`ScreenProbe`** — the existing test-only `restir_probe`/`ProbePoint` (a per-surface
estimator test, `voxel_raytrace.wgsl` ~1465, group-0 bindings 8–11) is a DIFFERENT concept; do not collide.

Resolution: probes on a **PROBE_SIZE×PROBE_SIZE** pixel grid — **start at 16** (Lumen default; 8 is 4× the
trace cost and was only proposed to paper over the missing SH low-pass — fix that and 16 is affordable). Each
probe = an **OCT_RES×OCT_RES = 8×8 FULL-SPHERE octahedral** radiance map in a **FIXED SHARED frame** (world or
view — NOT normal-aligned), so neighbouring probes / a probe across frames share texel directions and can be
filtered/reprojected by a plain texel-aligned blend. Plus an **order-2/3 spherical-harmonic (SH)** projection
per probe (9–16 coeffs × RGB) — this is the per-pixel integration representation (see pass 4).

Passes (probe grid sized to **full**-derived dims, dispatched over the **render_res** subrect; world cache stays
world-space & unchanged and serves as the level-2 far-field cache — the probe ray is ONE bounce → cache read):
1. **Probe placement** — for each probe cell, snap to the center pixel's primary-hit surface; store world-pos +
   normal + validity. **Adaptive edge probes are NOT deferrable** (uniform-only leaks across every depth/normal
   discontinuity; Sponza is wall-to-wall thin geo): include ≥1 edge-refinement placement pass by P2, OR scope
   P2/P3 validation explicitly to flat interior regions and flag edges as known-broken.
2. **Probe trace** — each probe traces OCT_RES² FULL-SPHERE directions (fixed frame); each is ONE bounce reading
   the world radiance cache (`reservoir_from_bounce_cached`) → short rays, multi-bounce via cache. Specify the
   miss path (→ sky/cache, never black). Writes the octahedral radiance **storage BUFFER** (not a storage
   texture — those cap at 8 and DLSS already uses 6; buffers have 48 headroom). These are the independent samples.
3. **Probe filter + temporal** — spatial filter between neighbour probes in the fixed-frame octa atlas + temporal
   reproject/accumulate. Reprojection is **world-position-based** (probe world-pos → prev-frame screen → prev
   cell) with a depth/normal validity reject + confidence cap + disocclusion reset — reuse the surface-reservoir
   reprojection machinery (`restir_p1`). Keep temporal **LIGHT** (short history) so it doesn't fight DLSS-RR's
   own temporal accumulation (double-temporal = ghosting); let RR do the heavy temporal lift.
4. **SH projection + per-pixel integration** — project each filtered probe's octa radiance to order-2/3 SH (the
   SH low-pass is the VARIANCE-REDUCTION mechanism, not an optimization — it discards the angular noise the boil
   meter measures). Per pixel: bilinear 2×2 probe gather, bilateral depth/normal reject (with a coarser-probe /
   world-cache fallback when all 4 are rejected → never black), blend the probes' SH coeffs, then a single
   **SH · cosine-lobe dot product** → indirect irradiance. × full-res albedo at shade time (demodulated → crisp).

## Integration with the current pipeline (file:line from the feasibility review)

- New probe compute passes inserted in BOTH paths: DLSS `voxel_rt_dlss_pass` after the world-cache dispatch
  (`raytrace.rs:5332`, before the restir p1 at `:5347`) AND the non-DLSS `voxel_rt_pass` mirror (`raytrace.rs`
  ~4714 region) — ~2× integration surface (two alloc blocks, two bind-group builds, two dispatch sites). The
  non-DLSS path is what the headless harness exercises, so it must be correct independently.
- New GPU resources as `Option<...>` fields on `VoxelRtResources` (two sets, non-DLSS + DLSS, mirroring
  `reservoirs`/`dlss_reservoirs`): probe header buffer, probe octa-radiance **storage buffer** (current +
  history), probe SH buffer, and a **separate** probe-grid uniform (NOT folded into `RestirParamsData` — avoids
  std140 padding churn). Allocate at **full**-derived size (`raytrace.rs:5013` gate keys off `full`), dispatch
  over the `render_res` subrect. History resets on realloc + a `reset` flag in the probe uniform (mirror `:5186`).
- Probe pipelines built ONCE (non-DLSS layout, no DLSS guide group) and reused in both paths; kept OUTSIDE the
  `#[cfg(feature="dlss")]` `init_dlss_pipelines` so the headless harness can dispatch them. New
  `dispatch_probe_passes(...)` sibling to `dispatch_world_cache_passes` (`raytrace.rs:4335`). Probe trace binds
  groups 0 (scene) + 1 (view) + its own probe group + 3 (world cache); re-set group 3 after binding the probe
  group (the `:5342` "rebinding drops higher groups" gotcha).
- `shade_restir_p2` (`voxel_raytrace.wgsl:2134`) indirect term: replace `restir_p2_core(...) * albedo` with
  `screen_probe_integrate(...) * albedo` — IDENTICAL contract (both return albedo-factored irradiance), a clean
  drop-in behind a `gi_mode` knob. BUT the probe SH buffer must be bound in the shade pass, so this 1-line WGSL
  swap drags the probe binding through BOTH `restir_pl` (`:2581`) and `dlss_restir_pl` (`:2924`) layouts + both
  p2 bind-group build sites — 4 wiring sites. **DI, guides, depth, motion stay full-res and unchanged.**
- Device limits already OK: `max_storage_buffers_per_shader_stage` is 48 (`main.rs:113`). Storage TEXTURES cap
  at 8 with DLSS using 6 (`main.rs:149`) — hence the atlas-as-buffer rule above.
- The pairwise-MIS spatial filter is reusable for probe spatial filtering, BUT only because probes store in a
  fixed angular frame (C2) — verify the MIS reuse is sound for octa-texel radiance records.

## Phasing (each phase: specialist → ≥2 adversarial reviewers → QA gate; harness-gated)

- **P0** — this doc + resource/uniform scaffolding (atlas+SH+header as storage buffers, full-size alloc) + the
  `gi_mode` A/B knob (restir | probes) + the fresh probe bind group. No visual change.
- **P1** — probe placement (uniform grid) + full-sphere octa trace → atlas buffer; debug view of a probe's
  octa-map. Validate: probe radiance mean ≈ per-pixel GI mean (energy), boil-meter luma.
- **P2** — **SH projection (C1, MUST be here, it's the variance mechanism)** + per-pixel bilinear+bilateral SH
  gather + SH·cosine integration → irradiance; wire into shade behind the knob; **≥1 adaptive edge-probe pass
  (H1)** or flat-region-only validation. Validate over a CONVERGED window (single-frame probe variance ≥ M1):
  Sponza interior blotch ≤ M4 (0.036); luma matches; **user live check** under DLSS-RR.
- **P3** — probe temporal accumulation (LIGHT history, world-pos reprojection + disocclusion reset — H2) + octa
  spatial filter. Validate: blotch keeps falling; **no double-temporal ghosting vs M4+RR under motion (user)**.
- **P4** — full adaptive-probe placement + perf pass (probe trace is the cost; measure vs M4; PROBE_SIZE 16 vs 8
  as a measured tradeoff). Retire per-pixel diffuse ReSTIR (drop reservoir bindings 0/1, keep DI 5/6 + surfaces
  3/4) if probes win on quality AND perf.

## Validation harness
Extend `tests/voxel_gi_boil_gpu.rs`: a `gi_probe_*` group measuring blotch/luma with `gi_mode=probes` vs the M4
reference. Headless can't see DLSS-RR — the user eyeballs each phase on Sponza (the meter gates energy +
fine/blotch CoV only). Perf: a headless probe-trace timing vs the per-pixel M4 path.

## RESULTS — P0–P4 IMPLEMENTED + VALIDATED (2026-06-18, headless Sponza meter)

Measured on the captured worst-boil Sponza viewpoint (`gi_sponza_blotch`), probes vs the per-pixel reference:

| config | luma | fine_CoV | blotch_CoV |
|---|---|---|---|
| M1 (unbiased per-pixel ref) | 56.5 | 0.62 | 0.052 |
| probes 16px oct8, no temporal | 66.8 | 0.16 | 0.10 |
| **probes 16px oct8 +temporal** | 67.2 | **0.055** | **0.037** |
| **probes 8px oct8 +temporal** | 70.1 | **0.041** | **0.020** |

- **Energy correct/unbiased:** probe luma ~67–70 matches the converged high-M value (M1's 57 is biased LOW;
  M-merge's 68 was biased high). The direct MC+SH probe is the unbiased reference.
- **SH low-pass crushes per-pixel grain:** fine_CoV 0.62 → 0.04–0.05 (~15×).
- **Temporal crushes the blotch (the boil):** 8px = **0.020**, 16px = **0.037** — at/below M4's 0.036 (the
  previous best), at CORRECT brightness.
- **Perf:** probe trace = (res/probe_size)² × oct_res² rays. 16px/oct8 = **0.25 ray/px** (16× fewer than M4's
  4 ray/px) for ≈M4 blotch; 8px/oct8 = 1 ray/px for 2× better blotch. The per-pixel ReSTIR GI is RETIRED when
  probes drive the diffuse (`restir_p1_core` gated off; DI still runs), so the wasted M-bounces are skipped.

**Implemented:** P0 scaffolding (group-4 probe layout/buffers/uniform, `gi_mode` knob, editor sliders) · P1
placement + equal-area **Fibonacci-sphere** trace (octahedral was area-biased) · P2 order-2 SH projection +
bilateral 2×2 integration · P3 light temporal (pos/normal-validity reject, no-reproject → no smear, packed in SH
`.w` lanes) · P4 edge fallback (nearest valid probe in 5×5) + retire per-pixel GI. Knob: `RestirSettings`
`screen_probes`/`probe_size`/`probe_oct_res`/`probe_temporal` (default OFF — A/B).

**Deferred refinements (not blocking):** full *adaptive* edge-probe placement (the fallback substitutes); true
world-position reprojection for temporal accumulation UNDER motion (currently rejects→fresh on motion, safe but
noisier — DLSS-RR cleans); octa spatial filtering between probes; SH negative-lobe clamp. **User live check
pending:** boil + ghosting under motion with DLSS-RR (headless can't see RR); dark-edge quality.

## Resolved by review (were open questions)
- Octahedral: **full-sphere, fixed shared frame** (not normal-aligned hemisphere) — cosine applied at SH
  integration, not storage. (C2)
- Per-pixel integration: **SH·cosine-lobe dot product** off an order-2/3 SH projection — NOT a raw octa loop.
  The SH low-pass is the variance killer. (C1)
- Reprojection: **world-position-based** + depth/normal reject + confidence cap + disocclusion reset; reuse the
  surface-reservoir machinery. (H2)
- Far-field/multi-bounce: the **world radiance cache is the level-2 cache** (probe ray = 1 bounce → cache). (M1)
- Atlas storage: **storage buffer** (storage-texture budget is 8, DLSS uses 6). Alloc at full, dispatch render_res.

## Remaining risks
- **Double-temporal vs DLSS-RR** (H3): keep probe history light; validate ghosting under motion against M4+RR
  (headless can't see it — user check).
- **Edge leaking** (H1): uniform-only probes leak across silhouettes; adaptive placement is required by P2, not
  P4 — or P2/P3 must be validated on flat regions only and not green-light edges.
- **Perf reality** (C3/M2): 1 probe/16px × 64 dirs = 0.25 ray/px trace; the quality comes from SH + temporal +
  spatial, NOT the trace. The M4 comparison must be over a converged window. Measure PROBE_SIZE 16 vs 8.
- **~2× path duplication** (non-DLSS + DLSS) and the 4-site bind-group wiring behind the "1-line shade swap".
- Pre-existing `restir_probe` (group-0 8..11) collision — use a fresh group, rename to `ScreenProbe`.
