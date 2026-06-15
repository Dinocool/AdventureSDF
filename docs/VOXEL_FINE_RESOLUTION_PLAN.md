# Fine-Resolution Migration — global LOD0 voxel size 0.2 m → 0.05 m — plan of record

Status: APPROVED + LOCKED (user-directed 2026-06: "0.05 m, make the change then build everything we need
for it"). Worktree `voxel-rt`. This is the program that ties the storage stack
(`docs/VOXEL_STORAGE_PLAN.md`) to a concrete target. Read this FIRST before touching `VOXEL_SIZE` — the
flip is the LAST step, not the first, and the reason is below.

## Decision
The world's LOD0 voxel edge becomes **0.05 m** (4× finer than today's 0.2 m). Because the demo scenes load
**into the world brick map** (the merge path — NOT per-object instances; user-confirmed), a scene's detail is
the world's detail, so finer scenes ⇒ a finer global `VOXEL_SIZE`. Driver: architectural detail (Bistro
signage/railings, Sponza relief) needs ~5 cm voxels; 0.2 m is visibly coarse.

## The cost — and why storage MUST come first (the forced order)
Refinement scales voxel count by the CUBE. 0.2 m → 0.05 m = **64×** more voxels per scene:

Note: `VOXEL_SIZE` is **0.2 m today** (verified: `brickmap.rs` `VOXEL_SIZE = 0.2`, `BRICK_WORLD_SIZE = 1.6`,
`MAX_LOD = 7`). Sponza was already **oversampled at 0.05 m** (a prior experiment) — so its asset is already at
the target and needs NO re-bake at S3; the discrepancy people see now is that the 0.05 m Sponza loads ~4×
oversized against the still-0.2 m engine, which the S3 flip resolves.

| scene | @0.2 m (today) | @0.05 m (est.) |
|---|---|---|
| Sponza | 15 MB .vox | **~1 GB** (already baked at 0.05 m) |
| Sibenik / Conference | ~8–13 MB | **~0.5–0.8 GB each** |
| Bistro Exterior | 41 MB / 10.3 M vox | **~2.6 GB / ~660 M vox** |

So at 0.05 m **every** scene's `.vox` is ~1 GB+ and its in-RAM `BrickMap` is multiple GB. A 2.6 GB `.vox` +
a full-RAM brick expansion per scene is impractical (disk, load time, RAM). Therefore the re-bake **cannot
land** until the storage exists to (a) store a fine scene compactly on disk and (b) load it WITHOUT a
full-RAM dense expand. That makes R5 (`.vxo`) a hard prerequisite of the flip, and R2 a prerequisite of the
resident VRAM at 4× density. VRAM-resident cost itself stays bounded by the camera-following clipmap +
surface-only residency (#135) + R1 (landed) — the blow-up is DISK + RAM + load, which R5 targets.

LOD0 fine-detail reach also QUARTERS: `brick_span(0)` 1.6 m → 0.4 m, so `clip_half · 0.4 · 2^MAX_LOD`. To keep
the current ~13 m fine reach, `clip_half` (or `MAX_LOD`) must rise (measure on the perf rig).

## Dependency-ordered program (each stage shippable, zero-warning, benchmarked, A/B-gated)
The storage stages keep the build GREEN (additive, the live path unchanged until each is proven); the flip +
re-bakes land together LAST (scenes load wrong-scaled between flip and re-bake, so they are one atomic step).

- **S1 — R2: per-brick palette + bit-packed indices (VRAM).** `docs/VOXEL_STORAGE_PLAN.md §R2`. Strata/surface
  bricks → tiny palette + `ceil(log2 k)`-bit indices; `dda_brick` bit-extract. Shared as the `.vxo` BRIK
  on-disk format too (disk↔VRAM transcode-free). Gate: GPU oracle byte-identical at every bit width; per-step
  DDA cost measured within budget. (R1 ✓ + R3 ✓ already landed; R2 is the remaining Tier-A in-memory win.)
- **S2 — R5: native `.vxo` format (disk + STREAMED load).** `docs/VOXEL_STORAGE_PLAN.md §R5` +
  `docs/VOXEL_INSTANCING_PLAN.md §1.5` (HEAD/MATL/BRIK[/LITE/LODS] chunks; `voxel_size` in HEAD so a `.vxo`
  is self-describing). Encoder (offline, in `examples/voxelize_scene.rs` → emit `.vxo`) + runtime decoder that
  **loads bricks directly without re-bricking** and can **stream by region** (so a 2.6 GB scene never fully
  expands in RAM). BRIK reuses R2's per-brick palette + RLE strata + brick dedup (R3) + zstd wrapper. `.vox`
  becomes import-only; the runtime depends on the `.vxo` reader. Gate: a `.vxo` round-trips byte-identical to
  the live-generated brick set; size ratio reported.
- **S3 — the FLIP + re-bake (atomic).** `VOXEL_SIZE` 0.2 → 0.05 in `brickmap.rs` (`BRICK_WORLD_SIZE` derives
  0.4). Re-pin EVERY test asserting `0.2`/`1.6`/`brick_span`/reach/scene dims (worldgen perf counts shift; the
  clipmap reach ratio shifts). Raise `clip_half`/`MAX_LOD` to restore fine reach (measure). Re-bake
  **Sibenik/Conference/Bistro** at 0.05 m → `.vxo` (Sponza is ALREADY at 0.05 m — no re-bake; Bistro needs the bounded-RAM tiled
  path #125 at this density). Gate: both feature builds green, re-pinned tests pass, the gallery loads all four
  at 0.05 m (user visual check), perf/VRAM measured before/after.
- **S4 — reach + LOD tuning.** With 0.05 m landed, re-tune `clip_half`/`MAX_LOD`/per-object LOD so the fine
  band reaches far enough and coarse LODs cover distance at bounded VRAM (the LOD-reach work #132 folds in
  here, now on the fine base).

## Status
- R1 (uniform collapse) ✓ landed. R3 (brick dedup) ✓ landed (`719fab4`).
- NEXT = S1 (R2 palette) → S2 (R5 `.vxo`) → S3 (flip + re-bake) → S4 (reach).
- `VOXEL_SIZE` stays 0.2 until S3 (flipping earlier breaks the build + makes scenes un-re-bakeable).

## Invariants (every stage)
Tier-A formats keep the in-shader DDA O(1) (no pointer chasing on the trace — see VOXEL_STORAGE_PLAN §5);
GPU==CPU oracle byte/pixel-identical; zero warnings + both feature builds; GI untouched; measure every
delivery (perf/residency/storage harness, numbers in the commit).
