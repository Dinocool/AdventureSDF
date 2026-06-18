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

## Phases
- H0: knob + half-res reservoir/surface buffers (reuse existing, sized full, dispatched/indexed at half-res).
- H1: dispatch restir_p1/p2 GI at half-res; full-res shade bilateral-gathers half-res reservoirs_a. Validate
  region grid matches reference + blotch.
- H2: temporal/motion correctness (reproject at half-res), edge/disocclusion fallback, perf.
- H3: promote DI to full-res if needed; retire / pick the default.
