# GI "boil" elimination — SOTA-grounded diagnosis & plan

Status: **in progress** (worktree `gi-boil`, branched off `voxel-rt`).
Owner: GI. Goal: kill the residual temporal variance ("boil"/shimmer) in the GI output **at full quality**,
by aligning to the SOTA reference (bevy_solari), never by turning quality knobs down.

> "Boil" = temporal variance in the GI that never converges — surfaces shimmer/crawl frame-to-frame,
> worst in shadowed / indirectly-lit areas and under camera motion. The ReSTIR math is reference-correct
> (prior audit); this is **unconverged variance** the cache + ReSTIR reuse + DLSS-RR aren't damping.

---

## 1. Current GI stack (as built — confirmed by reading the code)

Per-frame compute sequence (`src/voxel/raytrace.rs`, `encode_*`):

```
wc_decay → wc_compact_single_block → wc_compact_blocks → wc_compact_write_active
        → wc_update (indirect dispatch) → wc_blend
        → restir_p1 (initial RIS + temporal)    → reservoirs_b + surfaces_cur
        → restir_p2 (1 spatial neighbour + shade) → reservoirs_a + out_tex (+ DLSS guides)
```

- **ReSTIR GI** (`restir_p1_core`/`restir_p2_core`, `merge_reservoirs`) — near-verbatim Solari `restir_gi.wgsl`:
  pairwise balance-heuristic MIS merge, Jacobian guard `>1.2`, `RESTIR_CONFIDENCE_CAP = 8` (== Solari GI),
  LD (Hammersley + Cranley–Patterson rotation) hemisphere direction, permuted reprojected temporal tap,
  **one** spatial neighbour merged per frame (search budget `spatial_samples`).
- **World-space radiance cache** (`query_world_cache`, `world_cache_update`, `world_cache_blend`) — Solari
  `world_cache_*` hash grid; `max_temporal_samples = 32`, `cell_lifetime = 10`, adaptive luminance-delta blend.
  Cache-side **NEE** over an emissive-voxel light list (`wc_sample_light_nee` + `voxel_lights`/`voxel_light_alias`),
  MIS'd against the cosine bounce. Multi-bounce feed-forward (`gi_multibounce`).
- **DLSS Ray Reconstruction** (Quality) final denoise/upscale; guides written in `restir_dlss_p2`
  (diffuse albedo, specular albedo `0.04`, normal+roughness `1.0`, jittered depth, un-jittered motion).
- **No firefly clamp** (intentionally removed — best practice; do not re-add a biased clamp).

### Knob parity vs Solari (as shipped)

| Param | Ours | Solari GI | Solari DI | Note |
|---|---|---|---|---|
| GI confidence cap | **8** | 8 | 20 (DI) | matches GI |
| world-cache temporal samples | **32** | 32 | — | matches |
| world-cache cell lifetime | **10** | 10 | — | matches |
| GI ray distance | **50** | 50 | — | matches |
| spatial radius (px) | **16** | 30 | 30 | **weaker than ref** |
| spatial search taps | **4** | 5 | 5 | **weaker than ref** |
| screen-space ReSTIR **DI** | **none** | — | full pass | **missing — principal gap** |
| light-tile presampling | **none** | (shared) | 128×1024 | **missing** |

---

## 2. The decisive architectural divergence from Solari (the boil mechanism)

Solari splits illumination into **two** screen-space estimators plus the cache:

- **ReSTIR DI** handles *direct* light on the **primary** surface (lights/emitters via light-tile presampling
  + RIS + temporal/spatial reuse + a visibility ray). Low variance by construction.
- **ReSTIR GI** handles *indirect* only. Its initial sample traces **one** bounce and reads the bounce-hit's
  radiance **entirely from the world cache** (`reservoir.radiance = query_world_cache(hit) · base_color/π`),
  and **excludes** emitters hit by the bounce (`if emissive != 0 { return empty }` — they're DI's job).
  So the GI candidate radiance is a **smooth, temporally-averaged cache lookup** — RIS selection barely
  perturbs it frame-to-frame → it converges.

**Our GI initial sample is different** (`reservoir_from_bounce_cached`):

```
L_o(hit) = direct_lighting(hit)            // FRESH sun shadow ray at the bounce hit — per-frame, HIGH variance
         + emissive(hit) · strength
         + albedo(hit) · query_world_cache(hit)   // indirect only — smooth
```

The cache stores **indirect-incoming only** (the cell's own *direct sun* is NOT in it — `world_cache_update`
adds emitter-NEE at the cell + a cosine bounce, but no sun-direct-at-cell). So we **recompute the bounce-hit's
direct sun term fresh every frame**. A single LD bounce lands on a different surface point each frame; whether
that point is **sunlit or shadowed** is a large radiance jump, and it is **baked into the candidate's stored
radiance**. RIS then keeps switching the held reservoir between "bright sunlit-bounce" and "dark shadow-bounce"
samples → the resolved irradiance shimmers. **This is the dominant boil on sun/sky-lit scenes (e.g. Sponza,
which has no emitters).** It is exactly the high-variance term Solari moved into the smoothed cache.

For **emitter-lit scenes (Cornell, the Gallery emissive objects)** the dominant boil is the **missing screen-space
ReSTIR DI**: emitter light reaches a surface only when a random bounce happens to hit a small emissive voxel
(plus the cache's NEE feeding the cache). Finding emitters by random bounce is the classic ReSTIR-DI motivating
variance source (Bitterli 2020 / RTXDI).

> **Teardown corroboration** (Gustafsson, blog.voxagon.se; RenderDoc teardowns by juandiegomontoya & acko.net):
> a shipping voxel renderer keeps GI stable by **terminating bounce rays into a low-variance reprojected lookup**
> (last-frame lit color), **neighborhood-clamping** the reprojected history, and an **adaptive blend** (slower on
> noisier surfaces) — i.e. *stabilize the mean first; the denoiser only cleans high-freq residual.* Their explicit
> caveat on DLSS-RR: it is "a bit aggressive," and **RR will not fix a wandering low-frequency mean** — which is
> precisely what the fresh-direct-at-bounce term produces. This validates "fold direct into the smoothed cache."

---

## 3. SOTA references (what each does about variance / temporal stability)

### bevy_solari (in-repo ground truth — `D:\bevy-fork\crates\bevy_solari\src\realtime\`)
- `presample_light_tiles.wgsl`: **128 tiles × 1024 presampled light samples** per frame (RIS source shared by
  DI **and** the world cache's `sample_di`). A workgroup picks one random tile; each pixel draws its initial
  candidates from that tile → coherent memory + massively reduced light-sampling variance.
- `restir_di.wgsl`: 8 initial samples from the tile, RIS, temporal (cap **20**) + 1 spatial (radius **30**, up
  to **5** taps), **visibility ray** before shade, MIS balance heuristic, previous-frame light-id translation.
- `restir_gi.wgsl`: indirect via cache lookup at the bounce hit; cap **8**, spatial radius **30**/5 taps,
  Jacobian guard **1.2**, excludes emitters from the GI bounce.
- `world_cache_update.wgsl`: per active cell, `sample_di` (RIS over the light tiles) **+** `sample_gi` (one
  cosine bounce → `base_color · query_world_cache(hit)`). Adaptive temporal blend with a luminance-delta-driven
  responsiveness, cap **32**. **Cache stores total incoming (direct + indirect).**

### Teardown (Dennis Gustafsson) — closest production voxel-GI analogue
See §2 corroboration. Transferable: neighborhood history clamp, adaptive blend, terminate into a low-variance
lookup, blue-noise sample placement. *Not* a ReSTIR reference (no ReSTIR/NEE in the shipped engine).
Sources: `blog.voxagon.se/2018/01/03/...diffuse.html`, `.../2024/12/29/year-summary.html`,
`juandiegomontoya.github.io/teardown_breakdown.html`, `acko.net/blog/teardown-frame-teardown/`.

### re-flora (tr-nc/re-flora)
Software 64-tree, 1-bounce GI, no LOD, no ReSTIR — **ruled out as a boil reference** (we are ahead). Its only
transferable idea (readback-free GPU build) is unrelated to boil. (Prior in-repo analysis; confirm in §research.)

### General SOTA (papers / NVIDIA docs — cited)
- **ReSTIR DI course** (Wyman, SIGGRAPH 2023, `intro-to-restir.cwyman.org/.../2023ReSTIR_Course_Notes.pdf`):
  BSDF/bounce-only direct lighting is *"terrible"* variance; emitters must be a RIS candidate with proper MIS.
  This is the **#1 cited boil cause** and the textbook justification for Stage 2 (screen-space DI). Confidence
  cap (DI) **5–30, start 20**; "capping confidence weights is vital to combat correlations." Original ReSTIR DI
  draws **32** initial light candidates; ReGIR default **M = 8** (diminishing returns above).
- **ReSTIR GI** (Ouyang 2021): temporal **M-cap 30**, spatial **M-cap 500** ("so reservoirs don't get stuck
  with a particular sample"); spatial search radius = **10 % of image resolution**, halved on failure to a
  3 px floor; neighbour similarity normals < **25°**, depth < **0.05**; **Jacobian (Eq. 11) is mandatory**
  (omitting → lighting discontinuities + overestimation); **sample validation every ~6 frames** discards stale
  bright samples (else bias persists >12 frames after a lighting change). *Note: Solari — our direct port —
  chose cap **8** for GI; the paper's 30 is higher. The cap is a noise-vs-lag trade → tune by the boil-meter,
  do not blind-raise.*
- **GRIS / ReSTIR PT** (Lin 2022): constant `1/M` MIS weights **do not converge** (only Talbot/pairwise MIS
  do). We already use the balance-heuristic pairwise merge (Solari) — correct. A *too-high* M-cap eventually
  **increases** noise via correlation; "some cap is mandatory" but no universal number.
- **SHARC** (NVIDIA spatial-hash radiance cache, the cache family we ported): terminate paths into a
  **sample-count-weighted running average**; a **max-accumulated-frames** cap is the explicit stability-vs-lag
  knob (higher = stabler/laggier); a **responsive-lighting** mode trades smoothing for adaptation. Capacity
  baseline 2²², ~10–20 % occupancy. → validates our `max_temporal_samples = 32` + adaptive luminance-delta blend.
- **NRD / SVGF vs DLSS-RR — TOPOLOGY (confirmed):** DLSS Ray Reconstruction **replaces** the spatial denoiser.
  The correct pipeline is **`noisy RT GI → DLSS-RR`** — do **NOT** add NRD/SVGF upstream of RR. (So Stage 3's
  "screen-space denoiser" option is *ruled out* while RR is on.) RR guide contract (Streamline DLSS-RR guide):
  motion vectors (pixel-space jitter, matching scale), **LINEAR depth** (RR wants linear, unlike DLSS-SR),
  normals, roughness, **separate diffuse + specular albedo guides**, HDR color (`colorBuffersHDR = eTrue`), and
  **specular hit distance** (or specular MVs) for reflections. Wrong/missing MVs or albedo guides defeat RR.
  ⚠ **Flag for Stage 3:** we currently write **clip-space NDC depth** (`depth_clip.z/depth_clip.w`) to the RR
  depth guide, and a flat specular albedo `0.04` with no specular hit distance — verify Bevy's DLSS layer
  expects NDC (it may), else this is a real guide error contributing to residual instability.
- **Blue noise / STBN** (Heitz–Belcour 2019; EA SEED + NVIDIA STBN): per-frame-**rotated** LD (our current
  `rot = hash(pixel, frame)`) is **white over time** → "slower convergence + unstable when filtered
  temporally." **Spatiotemporal blue noise** (3D mask, one z-slice/frame, R2-offset) is blue in space *and*
  time → rapid convergence + temporal stability; it **redistributes** residual error to high frequencies RR
  removes (magnitude ~unchanged). This is the **last-mile residual** fix (Stage 3), layered on ReSTIR + RR.

---

## 4. Quantifying boil (so before/after is a number, not a vibe)

**Boil-meter** (`tests/voxel_gi_boil_gpu.rs`, to build): drive the full ReSTIR GI pipeline on a **static camera**
for `W` warm-up frames + `N` measurement frames; read the GI output each measurement frame; compute the
**per-pixel temporal coefficient of variation** `CoV = stddev_t(luma) / mean_t(luma)` over surfaces with
non-trivial GI, then report the **mean CoV** (and 95th-percentile) as the boil score. Lower = less boil.
Run on two scenes to separate the two boil axes:
- **Cornell** (emitter-lit) — exercises the emitter / DI-gap boil.
- a **sun-lit synthetic** scene (sun on, an occluder casting indirect shadow) — exercises the fresh-direct boil.

Attribution sweeps (toggling uniforms): world-cache on/off, fresh-direct vs cache-fold, spatial radius/taps,
NEE on/off. Each Stage records before/after numbers here.

> Runs only with an `EXPERIMENTAL_RAY_QUERY` Vulkan adapter (skips cleanly otherwise). If the dev box has no
> such adapter, the harness ships green-skipping and the **user runs it** and reports the numbers.

---

## 5. Plan (staged, each stage: specialist → ≥2 adversarial reviewers vs the references → QA gate)

**Stage 0 — Boil-meter** (§4). Land the harness + baseline numbers. *No render change.*

**Stage 1 — Fold bounce-hit direct into the world cache (Solari alignment) + spatial dials.**
Make the cache store **total incoming radiance** per cell (add a sun-direct-at-cell term to `world_cache_update`,
mirroring Solari `sample_di`), then change the GI initial sample to read the bounce-hit radiance **entirely from
the cache** (`emissive(hit) + albedo·cache(hit)`), dropping the fresh `direct_lighting(hit)`. Restructure
`world_cache_update`'s bounce term to read the hit's outgoing from the cache too (Solari `sample_gi`). This
removes the per-frame fresh-direct variance — the dominant sun/sky boil — with zero quality loss (primary direct
stays sharp & screen-space; only the *indirect* gets the cache's intended low-pass). Then bump
`spatial_radius 16→30`, `spatial_samples 4→5` toward Solari, **validated by the boil-meter** (not blind).
Add a **sun-on** world-cache test (existing energy tests are sun-off and don't cover this).

**Stage 2 — Screen-space ReSTIR DI for emissive voxels** (principal emitter-scene gap). Light-tile presampling
(reuse `voxel_lights`/alias) + a DI reservoir pass pair (initial+temporal / spatial+shade+visibility), in a
**separate UBO** (the 80B `LightingUniform` must not widen). Primary-surface emitter direct moves out of the GI
bounce into DI; the GI bounce then **excludes** emitters (Solari parity). Validate vs `restir_di.wgsl`.

**Stage 3 — Residual** (only if the boil-meter still shows variance RR can't damp), each gated on a measured win:
- **Spatiotemporal blue noise (STBN)** for the LD rotation. Our current `rot = hash(pixel, frame)` is *white over
  time* (the cited instability source); a 3D STBN mask (one z-slice/frame, R2-offset) redistributes residual to
  high frequencies RR removes. Highest-value residual fix.
- **DLSS-RR guide-correctness pass:** verify the depth guide (we write **NDC** `z/w`; RR wants **linear** — check
  Bevy's DLSS layer expectation), add **specular hit distance**, confirm motion-vector scale/jitter contract.
- **NOT** a screen-space denoiser before RR — SOTA topology is `noisy → RR` only (NRD/SVGF would be the wrong
  topology while RR is on). Ruled out.
- M-cap tuning (temporal toward 8→20–30 per the ReSTIR-GI paper, **only if the meter shows it helps** — GRIS
  warns a too-high cap re-introduces correlated noise) and spatial radius as %-of-resolution (paper) vs fixed.

## 6. Constraints (non-negotiable)
- SOTA-aligned, **no quality-knob-downs** to fake convergence. Full GI maintained.
- Zero warnings; `cargo build` **and** `cargo build --features editor`; `cargo clippy --features editor
  --all-targets` clean.
- `LightingUniformData` / WGSL `LightingUniform` stays **exactly 80 bytes**; new uniforms → a separate UBO
  (mirror `SkyUniformData` / `WorldCacheSettings`).
- Every tunable = runtime uniform + editor slider (no WGSL `const`). Register every new `Reflect` type.
- No self-launch; quantify via the boil-meter / probe oracle; hand the user precise eyeball steps.
- Keep changes scoped to the GI path (rebase onto `voxel-rt` cleanly; the legacy A/B `restir` toggle +
  `raymarch`/`raymarch_dlss` path are being removed there).

## 6a. Boil-meter & attribution results (2026-06-17, `tests/voxel_gi_boil_gpu.rs`)

Per-pixel temporal CoV of the **debug-view-5 (GI-only, pre-accumulation)** buffer, static camera, Cornell,
192², 90-frame warmup + 24-frame measure.

| Config (Cornell, emitter-lit, sun off) | mean CoV | p95 CoV | note |
|---|---|---|---|
| **A — defaults** (cache on, spatial 4/16, cap 8) | **0.232** | 0.457 | baseline |
| B — spatial 8 taps / radius 30 | 0.273 | 0.517 | **worse** |
| C — + confidence cap 24 (B held) | 0.344 | 0.647 | **worse** |
| D — + cache temporal 128 (B,C held) | 0.346 | 0.643 | ~same |
| E — **world cache OFF** | 0.455 | 0.906 | **worst (≈2×)** |

**Conclusions (this refutes part of the task's cheap-dial hypothesis #3):**
1. The **world cache is the dominant variance damper** (off ⇒ ~2× boil). Do not break it; *improve what it
   captures.*
2. **Cranking spatial reuse and the confidence cap toward the reference made boil WORSE**, matching GRIS
   (too-high M-cap re-introduces correlated noise) and the codebase's own "more spatial → more boil" note.
   So Stage-1's "bump spatial to Solari values" is **dropped** — measured, not assumed. The reuse path is
   already at a good operating point.
3. Residual boil = **candidate-radiance variance** the cache damps. The high-variance per-candidate term is a
   *direct* light term computed fresh at the GI bounce hit: **emissive-via-bounce** for emitter scenes
   (Cornell) and **fresh `direct_lighting` (sun shadow)-via-bounce** for sun-lit scenes (Sponza). Both are the
   classic "find direct light by random bounce" variance — fixed by moving that term to a low-variance
   estimator (the smoothed cache for sun → Stage 1; a ReSTIR DI reservoir for emitters → Stage 2).

## 6b. Stage 1 detailed design (cache-folds-direct — Solari `sample_di`+`sample_gi`)
The cell cache must store **total incoming** `E_total/π = (E_sun + E_emitter + E_indirect)/π`, all in the
**same cosine-pre-divided measure** the bounce term already produces (`E[L_o sample] = E_indirect/π`) and the
NEE term already uses (`·WC_INV_PI`):
- `world_cache_update` per cell adds `sample_di = wc_sun_direct(cell) + Σ wc_sample_light_nee(cell)` where
  `wc_sun_direct = sun_color·sun_intensity·max(dot(n,toSun),0)·shadow` (a sun shadow ray at the cell). **NOTE
  (corrected from the original plan):** `wc_sun_direct` carries **NO 1/π** — it reproduces `direct_lighting`'s
  convention exactly (the engine's whole direct path omits 1/π), so the consumer's `albedo·cache` reproduces
  the dropped `direct_lighting(hp)` term **byte-for-byte** → energy preserved, brightness unchanged. (A 1/π
  would dim GI sun ≈π× below the unchanged primary sun. The pre-existing sun-no-1/π vs emitter-with-1/π
  asymmetry in the cache is an engine-wide latent convention issue, NOT introduced here — out of scope.)
- `world_cache_update`'s bounce (`sample_gi`) drops `direct_lighting(bounce_hit)` and reads the hit's outgoing
  from the cache: `emissive(bounce_hit)·mis + albedo(bounce_hit)·cache(bounce_hit)` (Solari `sample_gi`). The
  cache feed-forward must therefore be **always on** (the sun's first indirect bounce now flows ONLY through
  the cache), so the `gi_multibounce` A/B gate is retired/forced-on for this path.
- The consumer `reservoir_from_bounce_cached` drops the fresh `direct_lighting(hit)`:
  `radiance = emissive(hit)·str + albedo(hit)·query_world_cache(hit)`.
- **Double-count guard (the correctness risk):** adding an explicit sun delta-NEE means a GI bounce that
  MISSES toward the sky must NOT also see the sun **disk** (that would count the sun twice). GI-bounce sky must
  return the sky **gradient only** (sun disk excluded), or MIS the sun NEE against the bounce-to-sky. Validate
  with the energy tests + a sun-on world-cache test (existing tests are sun-off, so they neither break nor
  cover this — a new sun-on test is required).

This keeps full GI (primary direct stays sharp + screen-space in `shade_restir_p2`; only the *indirect* sun
bounce gets the cache's intended temporal/spatial low-pass) and is pure alignment to Solari.

## 6c. Empirical fix results (measured, 2026-06-17)

All numbers are mean CoV of the debug-5 (raw, pre-accumulation, pre-DLSS) GI buffer; run-to-run noise ≈ ±2%.

| Change | Cornell (emitter) | Cornell (sun-lit) | verdict |
|---|---|---|---|
| baseline | 0.232 | 0.219 | — |
| **Stage 1** cache-folds-direct (Solari `sample_di`+`sample_gi`) | 0.235 | 0.221 | **boil-neutral** (refutes "fresh direct-at-bounce is the dominant boil") AND it made the GI bounce read its radiance from the cache, which is **empty on first sight** → a just-disoccluded surface has **0 indirect for ~1 frame** (a dark-flash). Solari tolerates this only because its separate **DI pass** lights first-sight surfaces; we don't have DI yet, so the flash would be black. **REVERTED** — zero boil benefit + a real disocclusion cost without the DI that justifies it. The cache-folds-direct form is correct and lands **together with Stage 2 DI** (it broke `voxel_gi_gpu::gi_indirect_fills_shadow`, which measures single-shot indirect before the cache is pumped — exactly the first-sight=black case). |
| **+ GI 3.2** LD-over-time R2 rotation (was white-over-time) | 0.229 | 0.215 | **−3%**, consistent on both axes, free. Kept. (True STBN mask = further upgrade.) |
| **+ GI 3.3** boil-tuned defaults (cap 8→5, radius 16→12; samples kept 4) | ~0.20 | **0.189** | **−13.5%** vs baseline (combined), measured §6d. Knobs stay editor sliders. |

## 6d. Knob attribution — isolated sweeps (recorded, `gi_boil_attribution_sweep` / `_radius_resolution_check`)

Each point sets ONE knob from default + long re-warm + reset between (no cumulative confound). Cornell, debug-5.

**Confidence cap (192², res-independent — lower = smoother temporal evolution, less "stuck-then-jump"):**
cap3 0.219 · cap4 0.219 · **cap5 0.221** · cap6 0.225 · cap8 (old default) 0.229 · cap16 0.253 · cap32 0.291.
→ chose **5** (near the floor, still enough accumulation; within the ReSTIR-DI course's 5–30).

**Spatial radius (px) — tighter reuses more similar surfaces:** at 192² (cap5): r10 best, r12 0.202, r16 0.221,
r32 0.279. At **384²** (`gi_boil_radius_resolution_check`, cap5): r10 0.146 · r12 ~0.15 · r16 0.163 · r20 0.173 ·
r32 0.202 · r48 0.229. The optimum is the SAME small px radius at BOTH resolutions (it does **not** scale with
res → a fixed small px radius, not a fraction-of-resolution). → chose **12** (a touch above the ~10 optimum to
avoid spatial under-smoothing the temporal-only metric can't see). `spatial_samples` (the SEARCH budget) left at
**4**: raising it measured no isolated boil benefit and costs a merge ("more spatial" was the direction that hurt).

**Cache:** temporal-samples 32→128 ≈ no change (0.229); cache **OFF 0.356** (≈2× — confirms the cache is the
dominant damper). Higher res already lowers per-pixel boil (0.146@384² vs ~0.19@192²) → live render-res boil is
better than these small-res numbers.

**Caveat (honest):** the meter is the RAW pre-accumulation, pre-DLSS GI buffer on a STATIC camera — it cannot see
disocclusion lag (no motion) or DLSS-RR's residual handling. The cap/radius choices are validated for static
input variance; the user should eyeball motion/disocclusion + the post-DLSS result. All three remain live sliders.

### Adversarial review outcomes (2 reviewers, validated vs Solari + the SOTA papers)
- **No critical correctness bug** in any landed change. The R2 rotation is a valid low-discrepancy-over-time
  sequence that doesn't touch the probe oracle; the cap/radius defaults are evidence-backed and remain sliders.
- **Applied fixes (to the LANDED changes):** reverted `spatial_samples 4→5` (unsupported, perf cost — held at 4);
  masked the R2 frame counter to 16 bits (f32 precision past 2²⁴ frames). The reviewers also confirmed the
  knob-default citations needed the recorded numbers (now §6d above) — done.
- **Stage 1 (cache-folds-direct) reviewed favourably** (energy byte-preserving, double-count guarded) but
  **reverted** for the reason in the §6c table — it was boil-neutral and would dark-flash disoccluded surfaces
  on first sight without a DI pass. Its review fixes (`wc_sun_direct` t_min=0; the sun-no-1/π convention; the
  `gi_sun_reaches_cache` guard) and the `gi_multibounce=0` weakening note travel WITH the change when it
  re-lands alongside Stage 2 DI.

**Interpretation:** the raw per-frame GI CoV (~0.22) is dominated not by the candidate's direct *magnitude* (Stage 1
showed that) but by the **single random-hemisphere bounce's directional + RIS-selection variance** — the inherent
1-spp ReSTIR-GI input the temporal/spatial reuse + DLSS-RR are meant to finish. Cranking reuse hurts (correlation).
The remaining big lever is the task's stated **principal gap #4 — screen-space ReSTIR DI** for the *emitter* term
(the `emissive(hit)`-via-random-bounce variance that dominates emitter scenes), and the **DLSS-RR guide
correctness** (NDC-vs-linear depth, specular hit distance) for the *post*-RR residual the headless meter can't see.

## 6e. Stage 2 — screen-space ReSTIR DI (GI 4.0), LANDED + validated

Implemented a Solari-`restir_di`-aligned screen-space DI for the emissive-voxel light list: per-pixel DI
reservoir (16 B, stores light_index+seed), RIS over `di_initial_samples` power-weighted alias candidates,
**initial-visibility folded into the reservoir** (the load-bearing fix), **temporal-only** reuse, a per-receiver
visibility ray at shade. New group-2 buffers (bindings 5/6), `RestirParams` widened 32→48 B, both render paths
wired, editor sliders. DI default-ON; cache emitter-NEE still carries indirect; the GI bounce drops its raw
`emissive(hit)` only when DI owns it (no double-count).

**Measured (DI-only buffer, debug view 8, Cornell emitter, static camera):**
| DI variant | mean CoV | note |
|---|---|---|
| no visibility fold + spatial | **1.25** | emitter flickers as the selected light's per-frame visibility flips |
| + initial-visibility fold | 0.30 | reservoir drops occluded lights (Solari `generate_initial`) |
| + same-pixel temporal | 0.26 | (permute swaps discrete lights) |
| **+ temporal-only (final)** | **0.05** | **−96 % vs the start**; emitter direct is now a stable, low-variance estimate |

So the emitter's direct light — previously found only by a random GI bounce hitting it (the dominant emitter-scene
boil) — is now a CoV-0.05 screen-space reservoir estimate. Two adversarial reviewers found **no defects** (RIS
unbiased, energy counted exactly once, plumbing/UBO-layout correct). Builds + clippy clean both configs; the
boil-meter `gi_di_emitter_direct_is_low_variance` guard (<0.15) + all GI tests pass (the one failing
`voxel_gi_gpu::gi_indirect_fills_shadow` is **pre-existing on voxel-rt** — the flat-ambient removal — confirmed
by stashing). Key engineering lesson: **spatial reuse of a CONTINUOUS GI radiance averages; spatial reuse of a
DISCRETE light choice oscillates** → DI is temporal-only (light-tile presampling is the discrete-light-aware
spatial upgrade, deferred).

## 6f. DLSS-RR boil (user report: "stabilises WITHOUT RR after some frames, tons of boil WITH RR")

This is a SEPARATE axis from input variance, and a sharp diagnostic: if the non-RR temporal accumulator converges
a static frame, the input MEAN is stable (the per-frame noise averages out) — so **RR boiling means RR is not
accumulating history** (if it were, it would converge too). Audited the RR wiring: the DLSS primary ray **is**
jittered (`temporal_jitter.jitter_projection`, raytrace.rs:5010-5012), depth is jittered reverse-Z (matches
Bevy's prepass), motion is un-jittered NDC·(0.5,−0.5) (matches Bevy's convention), guides are populated
(`voxel_dlss_guides_gpu`). No obvious wiring bug found — but the RR path **cannot be tested headless** (the
harness forces DLSS off; `voxel_dlss_guides_gpu` only checks guides are non-zero on the legacy `raymarch_dlss`
path, not the live `restir_dlss_p2` motion *convention*). The DI + cap/radius/LD work lowers the input variance
RR must handle (helps), but the "RR rejects history" symptom needs runtime diagnosis (see §7 handoff).

## 7. Decisions & log
- **2026-06-17** Diagnosis: dominant sun/sky boil = fresh `direct_lighting` at the GI bounce hit (un-cached,
  per-frame sun-shadow variance frozen into candidate radiance); Solari avoids it by reading the bounce radiance
  from the smoothed cache. Emitter-scene boil = missing screen-space ReSTIR DI. Spatial reuse is weaker than the
  reference. Sponza (no emitters) isolates the fresh-direct axis; Cornell isolates the DI axis.
- **2026-06-17 (measured)** The fresh-direct hypothesis was **refuted by the meter** (Stage 1 boil-neutral) —
  but Stage 1 is kept as the Solari-correct foundation + Stage 2 prerequisite. The boil that the cheap levers
  *can* reduce: LD-over-time sampling (−3%) + tighter spatial radius + lower confidence cap (−10% more). Landed:
  Stage 1 + GI 3.2 + GI 3.3, all reviewed (2 adversarial reviewers, no critical bug, 4 fixes applied) and
  validated (boil-meter −13.5% sun-lit; world-cache 6/6, Cornell, probe-oracle 2/2 green; new `gi_sun_reaches_cache`).
- **Status after this session — LANDED:** R2 low-discrepancy-over-time rotation + tuned defaults (cap5/radius12).
  Raw pre-DLSS GI CoV **−11% (emitter) / −14% (sun-lit)**, full GI maintained, energy unchanged, builds+clippy
  clean both configs, all relevant GI tests green. Stage 1 (cache-folds-direct) implemented + reviewed but
  **reverted** (boil-neutral + first-sight dark-flash without DI — re-lands with Stage 2). The boil-meter
  (`tests/voxel_gi_boil_gpu.rs`) is now a CI regression guard (mean CoV < 0.30).
- **Remaining (principal) lever:** **Stage 2 — screen-space ReSTIR DI** for the emitter term
  (`emissive(hit)`-via-random-bounce is the dominant emitter-scene variance). Large (light-tile presampling +
  a DI reservoir pass pair + a separate UBO + bind groups); own specialist→review→QA cycle. Re-land Stage 1's
  cache-folds-direct WITH it (DI lights first-sight surfaces, removing the dark-flash). See §5 Stage 2 +
  §3 `restir_di.wgsl`.
- **User-side check the headless meter cannot do:** the meter is pre-DLSS + static-camera, so it cannot see the
  post-DLSS-RR residual or disocclusion-under-motion — the user must eyeball those on Sponza (see handoff).
- **Pre-existing, NOT from this work:** `voxel_gi_gpu::gi_indirect_fills_shadow` fails on the clean `voxel-rt`
  base too (the recent flat-ambient removal makes that single-shot scene's indirect ≈0). Belongs to the
  ambient-removal change to update, not the boil work.
- **2026-06-18 — Stage 2 DI + the live-debugging arc (all LANDED, checkpoint-committed):**
  - **ReSTIR DI (GI 4.0)** landed (temporal-only, visibility-folded; DI-only CoV ~1.25→0.05). Stage 1
    cache-folds-direct re-landed with it.
  - **M initial GI samples (GI 6.0)** — `gi_initial_samples` (M) RIS-merges M fresh bounces/frame via a CHEAP
    same-receiver confidence-weighted streaming merge (algebraically identical to `merge_reservoirs` for the
    same receiver — Jacobian≡1 — but no per-candidate GRIS overhead / register bloat; bit-exact vs the heavy
    merge, validated). Sponza blotch: M1 0.054 → M4 0.036 → M8 0.029. **Default M=4.** M is the ONE lever that
    reliably kills the boil (fresh *independent* samples); cost is M× the bounce trace.
  - **Demodulation fix** — the specular-albedo guide was a constant `0.04` on a diffuse-only renderer, making
    DLSS-RR mis-demodulate dark surfaces (÷(albedo+0.04) ⇒ half the lighting on albedo 0.04). Set to 0.
    **User-confirmed: cut the visible boil a lot at M=4 + correct dark surfaces.** (We already follow RR's
    native demod contract: modulated colour + matching *linear* albedo guide; no rewrite needed.)
  - **STBN** — GI sample rotation now spatiotemporal blue noise (Bevy `stbn.ktx2`, `bluenoise_texture` feature,
    reservoir bind group 7, white-noise dummy fallback). Headless-neutral; **live: marginal** (the meter is
    DLSS-off so blind to it, like demod).
  - **Defensive/pairwise MIS** spatial reuse (RTXDI/Wyman Algo 7) — replaces the iterated single-neighbour
    balance merge (which AMPLIFIED variance, hence the 1-neighbour cap) with a proper MIS partition over
    {canonical + K neighbours}. **Correctness win** (fine-grain CoV now *falls* with more neighbours: 1.07→0.60
    at M1) but **does NOT close the boil gap** — blotch plateaus ~0.052 vs M4's 0.036, because spatial
    neighbours are *correlated* (not independent like fresh M). Oracle 2/2 still green.
  - **Root-cause measurement (Sponza, M1):** raising the temporal cap (16→128) or spatial count MONOTONICALLY
    WORSENS blotch (0.053→0.069) and darkens luma — reuse is correlation-limited; only fresh M helps.
- **DECISION (2026-06-18):** the cheap per-pixel levers (cap, radius, STBN, MIS) are exhausted; reuse cannot
  manufacture the independent samples that kill boil. **Next: half-resolution GI via Lumen-style screen-space
  radiance probes** (user-chosen) — downsampled probe grid + octahedral per-probe trace (independent samples
  per probe) + depth/normal bilateral integration to full res. Cuts ray cost AND adds independent samples in
  the low-res probe domain. The pairwise-MIS spatial filter is reusable groundwork.
