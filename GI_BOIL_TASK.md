# Task: eliminate GI "boil" (residual temporal variance) — SOTA-grounded

You are a fresh Claude Code session in the `gi-boil` worktree (branched off `voxel-rt`). Your mission: **eliminate the boil/shimmer in the global-illumination output while maintaining FULL GI quality at the full config** — by aligning to the SOTA, not by turning quality down. Drive this end to end: research → plan → implement → verify. The user reviews at the end.

## What "boil" means here
Temporal variance/noise in the GI that doesn't converge — surfaces shimmer/crawl frame to frame, especially in shadowed / indirectly-lit areas and under camera motion. It is NOT a broken estimator (the ReSTIR math is reference-correct per a prior audit) — it is **unconverged variance** that the cache + ReSTIR reuse + DLSS-RR aren't fully damping.

## The engine + current GI stack (read the code to confirm all of this)
HW-ray-traced cubic-voxel brickmap engine on a wgpu-trunk Bevy fork (`D:\bevy-fork`). The GI path, ported from `bevy_solari`:
- **ReSTIR GI** (screen-space reservoirs, two passes `restir_p1`/`restir_p2`, temporal + 1 spatial neighbour reuse).
- **World-space radiance cache** (SHARC/Solari-style hash grid) feeding ReSTIR's *initial* reservoir (the main variance-collapse mechanism; temporal blend cap 32). Cache-side **NEE** (`wc_sample_light_nee`, MIS'd against the bounce).
- **DLSS Ray Reconstruction** (Quality mode) as the final denoiser/upscaler.
- Knobs live in `RestirSettings` / `WorldCacheSettings` (`src/voxel/raytrace.rs`).

## What a prior 4-agent SOTA-alignment audit found (variance-relevant)
1. The ReSTIR RIS/reservoir core + world-cache hash/probe/temporal blend + DLSS jitter contract are **near-verbatim Solari ports** — reference-correct. There is **no firefly clamp** (intentionally removed — best practice; do NOT re-add a biased clamp).
2. A flat `ambient_color · AO` double-count was just **removed** from the lit path. This **unmasks** boil: shadowed areas are now lit *only* by GI, so GI variance there is fully exposed (this is correct — dark/occluded is intended). Expect boil to look worse in shadow than before the fix; that's the variance you must now actually reduce.
3. **Spatial reuse is weaker than Solari**: `spatial_radius = 16px` (Solari 30), `spatial_samples = 4` (Solari 5), one neighbour merged. Cheap dial to test, but validate against SOTA rather than blindly matching.
4. **We have cache-side NEE but NO screen-space ReSTIR DI / light-tile presampling.** Finding emitters only via random bounce is the classic boil source. Solari has a full `restir_di.wgsl` + `presample_light_tiles`. **This is the most likely principal gap.**
5. The world cache IS wired as the initial-reservoir source — but VERIFY it's genuinely damping (queried + accumulating, not starved). A separate streaming bug has been seen to partially-load scenes; a half-loaded scene starves the cache. Use Sponza (loads fully) as your test scene, not the Gallery, to isolate GI boil from streaming.

## Research mandate (the user explicitly wants breadth, not just Solari)
Before designing, study CONTEMPORARY + SOTA real-time GI and write down what each does about variance/temporal stability:
- **bevy_solari** (in-repo reference): `D:\bevy-fork\crates\bevy_solari\src\realtime\` — `restir_gi.wgsl`, `restir_di.wgsl`, `world_cache_*.wgsl`, `sampling.wgsl` / `presample_light_tiles`, `node.rs`, `prepare.rs`. This is the ground truth for ReSTIR DI + light tiles.
- **Teardown** (Dennis Gustafsson) — a shipping VOXEL engine with real-time GI. Study his GDC/blog talks on voxel GI, path tracing, temporal accumulation + denoising, and how he keeps it stable. Closest production analogue to our engine. (WebSearch/WebFetch his writeups + talks.)
- **re-flora** (tr-nc/re-flora) — check whether it has a GI solution worth porting. Per prior analysis it's a SOFTWARE 64-tree, 1-bounce GI, no LOD — likely NOT a boil reference, but inspect its temporal/denoise approach and confirm or rule it out. (Search for the repo / its notes.)
- **General SOTA**: ReSTIR DI/GI original papers + NVIDIA RTXDI, ReSTIR PT, **SHARC** (spatial hash radiance cache), **NRD** (NVIDIA Real-Time Denoisers — REBLUR/RELAX), DLSS-RR integration best practices, SVGF/à-trous screen-space denoising, blue-noise / low-discrepancy sampling for the residual.

Validate any proposed fix against these references (cite them). The user's standing rule: design to the SOTA TARGET, align to the reference; if we're slower/noisier than the reference it's OUR divergence — fix by aligning, never by a biased hack.

## Likely fix ladder (a HYPOTHESIS — confirm/replace it with your research)
1. **Verify cache damping** (is the world cache actually feeding + accumulating into the initial reservoir, or under-sampled/starved? quantify).
2. **Screen-space ReSTIR DI** for direct/emitter light (build an emissive-voxel light list + light-tile presampling + a DI reservoir, à la Solari `restir_di` + `presample_light_tiles`) — the principal suspected gap.
3. **Stronger spatial reuse** (radius/samples toward the reference; or a better spatial scheme).
4. **DLSS-RR guide correctness** (are the guides — depth/motion/albedo/roughness/specular-hit-distance — all correct and complete? a wrong/missing guide defeats RR's denoising).
5. Only if input variance is still too high for RR: a screen-space denoiser stage (NRD/SVGF) before/around RR.

## Constraints (NON-NEGOTIABLE)
- SOTA-aligned, NO shortcuts / NO quality-knob-downs to fake convergence. Maintain FULL GI.
- Zero warnings. Build BOTH configs: `cargo build` AND `cargo build --features editor` (run cargo IN this worktree). `cargo clippy --features editor --all-targets` clean.
- The `LightingUniformData` / WGSL `LightingUniform` is exactly **80 bytes, fully packed** — do NOT widen it; new uniforms go in a SEPARATE UBO (mirror how `SkyUniformData` / `WorldCacheSettings` were added).
- Every tunable is a **runtime uniform + editor slider**, never a WGSL `const`.
- Register every new `Reflect` type in its plugin.
- **No self-launch** — you cannot run the app; the user does the runtime/visual verification. Reason rigorously, use the headless GI probe oracle / any GPU test harness (`tests/voxel_gi_gpu`, `voxel_restir_gi_gpu`, the `restir_probe` oracle) to validate energy/convergence where possible, and tell the user exactly what to eyeball.
- **Measure**: find or build a way to QUANTIFY boil (e.g. temporal variance of a static-camera GI buffer across N frames) so "before/after" is a number, not a vibe.
- Work as specialist → ≥2 adversarial reviewers (validate against the references above, catch drift from the canonical approach) → QA gate, per stage. Document the design + decisions in `docs/` so it isn't re-derived.

## Coordination
A concurrent change on `voxel-rt` is removing the legacy ReSTIR A/B toggle (`RestirSettings.restir`) + the dead legacy `raymarch`/`raymarch_dlss` path. The LIVE GI is the ReSTIR path (`restir_p1`/`restir_p2`/`restir_dlss_p2` + the world cache). Focus there. Expect to rebase/merge onto `voxel-rt` when done — keep your changes scoped to the GI path to minimize conflicts. Do NOT `git restore` scenes or `world.graph.ron`.

## Key files
- `assets/shaders/voxel_raytrace.wgsl` — `restir_p1_core`, `restir_p2_core`, `merge_reservoirs`, `query_world_cache`, `world_cache_update`, `wc_sample_light_nee`, `reservoir_from_bounce(_cached)`, `restir_p2` / `restir_dlss_p2`, `sky_radiance`.
- `src/voxel/raytrace.rs` — `RestirSettings`, `WorldCacheSettings`, `LightingUniformData`, the GI dispatch (`restir_p1`/`p2` pipelines), the world-cache passes, the DLSS-RR wiring/guides.
- `docs/soft-coalescing-dolphin.md` — the GI plan (Phase 2.5 = NEE / emissive-voxel light sampling / ReSTIR DI, currently deferred — this task likely promotes it).
- `src/editor/render_panel.rs` — the GI/render knob sliders.

## Start here
1. Read this file's referenced code + the Solari reference to ground yourself.
2. Do the SOTA research (Solari, Teardown, re-flora, general) and write findings into `docs/GI_BOIL_PLAN.md`.
3. Diagnose the dominant boil source (instrument/quantify on Sponza).
4. Propose the SOTA-aligned plan, then implement stage by stage with the QA discipline above.
