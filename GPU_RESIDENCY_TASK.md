# Task: BUILD the one all-GPU voxel residency path

You are an INTERACTIVE Claude Code session in the `gpu-residency` worktree (off `voxel-rt`). The user steers you. **Your job is to IMPLEMENT — write the code. Do NOT get stuck debugging or building test gates/diagnostics. Build the correct path; the user runs it to verify.**

## What to build (the spec)
Read `docs/UNIFIED_GPU_RESIDENCY_PLAN.md` — the **"⟶ FINAL TARGET (4+1 directives)"** block at the top is the spec (the older body is background). In one breath:

> ONE GPU-driven, readback-free residency path (enumerate → diff → pack → AABB → BLAS into a fixed-cap GPU pool + page/slot table) for EVERY scene. `.vxo`-only (delete `.vox`). NO CPU residency/pack pipeline (delete `ResidencyManager`/`ResidentPacker`/`pack_one`/classify/`apply_delta`/StreamSnapshot). NO CPU NEE bake — build the emissive light list GPU-side. CPU keeps only `.vxo` IO + command submission + one-time pool buffer alloc. Must not preclude procedural worldgen (keep the `BrickSource` producer abstraction; allow a GPU producer; robust under continuous motion) — but don't build worldgen.

## How to approach it — BUILD, don't investigate
- **The "paged-blank" is fixed by building the path correctly, not by debugging the old one.** The eager path never blanks because every asset is placed at baked-in world coords with ONE transform. Build the unified path with a single world↔local offset SSOT so multi-asset offsets are correct by construction. **Do NOT spin on `ADVENTURE_PAGED_DIAG` / A-vs-C disambiguation / RenderDoc** — just implement it right and let the user run it.
- **Make the paged GPU drive the one true path and route every scene through it** (in-RAM scene = "fits resident, never evicts"). Then move the NEE light list GPU-side. Then delete the CPU pipeline + `.vox` + eager + the env gate. Build it, don't gate it to death.
- Lean on the **existing** tests to not regress; don't author elaborate new diagnostic harnesses. The user's live run is the render verification.

## Stages (implement, commit each, check in with the user between)
1. Single world↔local offset SSOT + route the paged GPU drive as the live path for `.vxo` scenes (no env gate) — get the Gallery (multi-asset, offsets) + Sponza rendering on the GPU path. Commit; user verifies render.
2. GPU emissive light-list build (replace the CPU NEE bake). Commit; user verifies lighting.
3. Delete `.vox` + the eager store + the CPU residency/pack pipeline + the CPU light bake + `ADVENTURE_GPU_PAGED_DRIVE`. Commit; user verifies nothing regressed.
(Verify a `.vxo` exists for every shipped scene before Stage 3 deletes `.vox`.)

## Constraints (brief)
- GPU-only, SOTA-aligned, no shortcuts/scope cuts. Fork has NO indirect AS build → fixed-cap pool + degenerate AABBs + full BLAS rebuild of dirty chunks.
- Zero warnings; build BOTH `cargo build` and `cargo build --features editor` (run cargo IN this worktree). `LF→CRLF` git notices are cosmetic.
- Don't widen the 80B `LightingUniformData` (new uniforms → separate UBO). Knobs = runtime uniforms. Register new `Reflect` types.
- No self-launch — the USER runs + verifies the render; tell them what to check.
- READ-ONLY research sub-agents are fine if you need a SOTA pattern (GPU light-list, sustained-motion streaming); implementation stays here.
- Don't `git restore` `*.scene` / `world.graph.ron`. You own residency/streaming (`residency_*.rs`, `voxel_residency.wgsl`, the streaming + AS-build in `raytrace.rs`, `vxo/source.rs`, `incremental.rs`); a separate `gi-boil` session owns the GI shader path — stay out of it.

## Start
Read the plan's directives block + the residency files, then go straight to Stage 1. Propose the Stage-1 approach in 3-4 lines, then implement it. Don't over-analyze.
