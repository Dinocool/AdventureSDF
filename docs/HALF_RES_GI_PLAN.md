# Half-Resolution ReSTIR GI — Plan of Record

Status: **IN PROGRESS** (gi-boil branch). Screen probes (docs/SCREEN_PROBE_PLAN.md) are SHELVED (default off):
they kill the boil but go FLAT — order-2 SH + one probe per 16px loses per-pixel occlusion (contact shadows
lifted, ~20% over-bright; confirmed by the `gi_probe_spatial_diag` region grid). They'd need an AO/detail layer.

## Why half-res reservoirs instead

The investigation's one robust win is M (fresh independent bounce samples) — it's SHARP and looked correct to
the user at M4, just 4× the trace cost. Half-res ReSTIR affords high M at ¼ the pixels:
- trace + run ReSTIR GI at **half render-res** (¼ the bounce traces),
- the full-res resolve gathers the neighbouring **half-res reservoirs** and re-evaluates each against the
  FULL-res pixel's own normal (the real sample direction's cosθ) — so it stays sharp + normal-correct (NOT
  SH-smoothed like the probes). kajiya / RTXDI pattern.

## Architecture

- `gi_half_res` knob (RestirSettings). GI res = render_res / 2 (knobbable later). DI + direct + shade FULL-res.
- **restir_p1 (GI) at half-res** → half-res reservoirs_b + surfaces (the M-candidate trace + temporal reuse).
- **restir_p2 (GI spatial) at half-res** → half-res reservoirs_a (final), NO shade.
- **Full-res shade** (the existing p2 entry, restructured): primary trace + sun direct + DI (full-res) +
  **bilateral gather of the 2×2 half-res GI reservoirs**, each resolved with the full-res normal, depth/normal
  weighted → indirect irradiance × albedo. Guides/depth/motion full-res + unchanged.
- First cut: run DI at half-res too (simpler; DI is already temporal/low-variance) — promote DI to full-res
  later if its shadows soften too much.

## Upscale technique — SOTA decision (researched 2026-06-18; do NOT use plain bilateral-on-color)

Verdict: bilateral-on-finished-GI-color is the shipping *baseline*, not the ceiling, and a full SVGF/RELAX pass
before DLSS-RR is wrong (double-filtering — RR is the terminal learned denoiser; pre-denoising removes detail RR
needs). SOTA for a half-res-reservoirs → render-res → DLSS-RR pipeline (kajiya / RTXDI / NRD+RR guidance):

1. **Reservoir-aware resolve, not color blur** — gather ~4–8 nearby half-res reservoirs per full-res pixel and
   re-evaluate each against THAT pixel's own normal+position, weighted by normal/depth/AO similarity (kajiya
   two-pass spiral resolve). Reconstructs thin features from the samples. THE #1 lever.
2. **Rotating half-res + temporal** — trace a different pixel of each 2×2 each frame + reproject reservoirs →
   recover TRUE full-res detail over ~4 frames, not interpolation. Nearly free (we keep history). kajiya's trick.
   (NRD prefers regular pixelPos/2; RR is tolerant — rotating is fine + beneficial with RR downstream.)
3. **Keep the spatial step LIGHT + demodulated** — output render-res, ALBEDO-DEMODULATED (irradiance/albedo)
   noisy GI + firefly clamp, hand to DLSS-RR. No SVGF/RELAX/strong-temporal pre-RR. (We already feed RR a linear
   albedo guide; ensure the GI path is the ratio.)
4. **Near-field-raw / far-field-reservoir split** (kajiya) — keep per-frame raw ray data for CONTACT GI so the
   resolve doesn't wash out corners (the exact failure that flattened the probes).

Sources: kajiya gi-overview, NRD README (run-before-DLSS, demodulate, fireflies, pixelPos/2), DLSS-RR
Programming Guide (render-res noisy input, demodulated reflectance), SVGF (Schied 2017), Lumen perf guide.

## Validation
The aggregate CoV metric was MISLEADING for probes (blind to flatness). Gate on the **`gi_probe_spatial_diag`
region grid** (reused/renamed): half-res GI region luma must MATCH the full-res restir reference (sharp, same
contrast — NOT lifted darks), AND blotch ≤ the M-per-pixel reference at the reduced trace cost. Then user live.

## RESULTS (2026-06-18, headless Sponza) — IMPLEMENTED, honest findings

Implemented (non-DLSS path; `gi_half_res` knob, default off): restir_p1 + a new `restir_gi_spatial` pass at
render_res/2 → half-res final reservoirs; full-res `restir_gi_gather` bilinearly gathers the 2×2 half-res
reservoirs and RE-RESOLVES each per the full-res normal + per-pixel visibility. RestirParams grew `gi_half`/
`gi_half_x/y`; `max_bind_groups` already raised for probes.

- **SHARP — the win vs probes (region-grid diag):** half-res tracks the full-res reference AND PRESERVES dark
  contact shadows (dark center 18–27 vs ref 27–34 — *more* contrast, not the probe's lifted 45). The
  reservoir-resolve (re-evaluate per full-res normal, not SH/colour blur) keeps it sharp. Flatness SOLVED.
- **But BOILIER, not cleaner (the honest result):** blotch_CoV half-res M4 **0.074** (no jitter) / 0.088
  (rotating) vs full-res M4 **0.036**; half-res M8 0.058. Fundamental: ¼ the pixels = ¼ the samples = noisier;
  the gather interpolates but cannot manufacture samples. The boil is sample-count-limited, so reducing samples
  raises it. Perf win (¼ GI bounce traces) is real, but it trades MORE pre-RR boil for it.
- **Rotating jitter:** recovers spatial detail over frames but adds ~20% blotch → defaulted OFF (centre sample);
  kept as a future knob. DLSS-RR is expected to recover detail downstream instead.

**Open question only the live DLSS-RR test answers:** does RR clean the half-res boil to acceptable? The meter
is DLSS-off (blind to RR), like demod/STBN/probes were. If RR cleans it, half-res = M4 quality at ¼ trace cost.
If not, full-res M4 (clean, 4× cost) stands. **Needs the user's live A/B (toggle "Half-res GI") on Sponza.**

## FULL-RES SPATIAL-AVERAGE FILTER (2026-06-18) — the boil-killer, LANDED

Origin: the user observed half-res GI is **temporally STABLE when the camera is static** but breaks up in motion,
while full-res is NOT stable even static. The stability comes from the full-res shade **averaging the 2×2
neighbour reservoirs** in the half-res gather — a spatial filter. Averaging N independent per-pixel estimates cuts
the per-pixel temporal *switching* (the boil) ~1/N. So: do that averaging at **full res** (no half-res sample
deficit, no motion break-up, no sharpness loss).

Implementation (`gi_filter_radius` knob, RestirParams + RestirSettings, default **1**):
- A `restir_gi_spatial` pass runs at full res (gated `gi_half_res || gi_filter_radius>0`) → writes the POST-SPATIAL
  final reservoirs to `reservoirs_a` (via `restir_p2_core`).
- The full-res shade/debug calls `restir_gi_spatial_average`: averages the (2r+1)² neighbour `reservoirs_a`, each
  **re-resolved against THIS pixel's own normal/position** (sharp, not a colour blur), depth/normal-bilateral
  weighted (`exp(-plane*16)`, `nw²`). Averaging the POST-SPATIAL finals (bounded ucw) is stable; averaging the
  post-TEMPORAL `reservoirs_b` spread ucw fireflies (measured worse — reverted).
- **CRITICAL BUG fixed:** non-DLSS `restir_p2` gated its `reservoirs_a` clear on **only `gi_half==0`** (missed
  `gi_filter_radius`), so it wiped the gi_spatial output every frame → the average read empty → black. Now gated
  `gi_half==0 && gi_filter_radius==0` (matching the DLSS variant). This was the entire "black output" red herring.

RESULTS (headless Sponza, sun-lit, debug_view=5 raw GI):

| Config | luma | fine_CoV (per-px boil) | blotch_CoV |
|---|---|---|---|
| M4 filter OFF (prior baseline) | 68.0 | 0.473 | 0.0353 |
| M4 r1 (3×3) | 79.7 | **0.096** | **0.0225** |
| M4 r2 (5×5) | 80.0 | **0.067** | 0.0218 |
| M1 filter OFF | 56.4 | 0.628 | 0.0531 |
| **M1 r1 (3×3)** | 72.3 | **0.111** | **0.0297** |
| M1 r2 (5×5) | 72.7 | 0.082 | 0.0297 |

- **fine_CoV (the boil) collapses ~5–6×** (0.47→0.096 at M4 r1). blotch also drops.
- **M1 r1 beats unfiltered M4** on both axes (0.030 vs 0.035 blotch, 0.11 vs 0.47 fine) at **¼ the trace cost** —
  i.e. the spatial average at 1 sample/px is cleaner AND cheaper than 4 fresh samples/px. The boil was per-pixel
  estimator switching; averaging neighbours fixes it directly, where raising M (more independent samples) only
  diluted it. Stays sharp (per-pixel re-resolve) + motion-correct (full-res, no reprojection seam).
- Wired into BOTH render paths (non-DLSS + DLSS) so the user can live-A/B with DLSS-RR. Editor slider added.

**LIVE RESULT (user, with DLSS-RR): WORSE at any radius>0 — the meter was inverted.** The expectation above was
wrong. A spatial pre-filter converts the **RR-removable high-frequency per-pixel noise into RR-UNremovable
low-frequency blotches**: RR's spatial denoiser crushes HF noise easily but reads spatially-smooth LF variance as
real large-scale lighting and keeps it — and LF blotch is exactly what the eye perceives as boil. This is the
"don't pre-denoise before DLSS-RR" rule (RR is the terminal learned denoiser), now confirmed empirically here.
**Default → `gi_filter_radius = 0`** (kept as a knob for the non-RR path / experiments).

**Consequence for the whole effort:** every "reduce pre-RR input variance" lever (this filter, half-res, screen
probes, STBN, spatial MIS) passed the DLSS-OFF meter and then lost or regressed live — the meter is the wrong
instrument for the RR pipeline. The only real RR wins were variance reduced **at the source** without adding
spatial correlation: the demod-guide fix and ReSTIR DI. The remaining real lever is **RR's temporal
accumulation** (the user's "stable WITHOUT RR, boils WITH RR" ⇒ RR isn't converging its history) — needs a
runtime diagnosis (does a STATIC camera converge WITH RR on?), not more input-side denoising.

## Phases
- H0: knob + half-res reservoir/surface buffers (reuse existing, sized full, dispatched/indexed at half-res).
- H1: dispatch restir_p1/p2 GI at half-res; full-res shade bilateral-gathers half-res reservoirs_a. Validate
  region grid matches reference + blotch.
- H2: temporal/motion correctness (reproject at half-res), edge/disocclusion fallback, perf.
- H3: promote DI to full-res if needed; retire / pick the default.
