# Phase D — the 0.05 m flip (D1) + screen-error LOD reach (D2) — implementation-ready

Status: DESIGN (no engine code changed by this doc). Worktree: `voxel-rt`. This is the executable spec for
**Phase D** of `docs/VOXEL_PROGRAM.md` — the last phase of the voxel-RT program. A future implementation agent
runs from this doc. It assumes **Phases A (GPU execution), B (`.vxo` disk), and C (tiled 0.05 m bake)** have
landed; D1 is gated on all three (the re-bakes are un-doable without B's `.vxo` and C's tiled voxelizer). This
doc specs the **flip mechanics** (D1) and the **reach mechanism** (D2), not those prerequisites.

Read first: `docs/VOXEL_PROGRAM.md` (Phase D), `docs/VOXEL_FINE_RESOLUTION_PLAN.md` (the migration program — D1
is its S3+S4), `docs/VOXEL_LARGE_SCENE_PLAN.md` §3/Phase C (the screen-error notes D2 promotes),
`docs/SOTA_REFERENCE.md` §3.4 (Nanite screen-error / projected-footprint LOD).

Ground-truth code surveyed: `src/voxel/brickmap.rs` (`VOXEL_SIZE`, `BRICK_WORLD_SIZE`, `MAX_LOD`,
`brick_span`/`lod_voxel_size`/`lod_edge`), `src/voxel/streaming.rs` (the distance-shell clipmap D2
augments), `src/voxel/{gpu,voxelize,source,raytrace,physics,cornell,edits,mod}.rs`,
`assets/shaders/voxel_raytrace.wgsl`, and the `tests/voxel_*.rs` + `examples/voxelize_scene.rs` corpus.

---

## 0. The two changes in one sentence

D1 is a **one-constant edit** (`VOXEL_SIZE: 0.2 → 0.05`) plus a **mechanical re-pin** of everything that
hardcoded the old number, plus three scene re-bakes. D2 is the **reach mechanism that makes D1 usable**: the
flip quarters LOD0's metric reach (`brick_span(0)` 1.6 m → 0.4 m), so the fine band that reached ~13 m now
reaches ~3.2 m; a brute `clip_half`/`MAX_LOD` bump to compensate is FOV- and resolution-blind, so we instead
pick each brick's LOD from its **projected pixel footprint** (Nanite/GigaVoxels screen-error), keeping the
exact-tiling clipmap as the residency filler.

These compose but are **separable**: D1 can land with the existing distance-shell `brick_lod` (just a coarser
`clip_half`) and D2 layered on after. Recommended order: **D1 first (flip + re-bake + re-pin, green build),
then D2 (screen-error)** — so the flip's correctness is provable against the unchanged LOD policy before the
LOD policy itself changes.

---

# D1 — the flip

## D1.0 The edit itself

In `src/voxel/brickmap.rs`:

```rust
pub const VOXEL_SIZE: f32 = 0.05;                                  // was 0.2
pub const BRICK_WORLD_SIZE: f32 = BRICK_EDGE as f32 * VOXEL_SIZE;  // derives 0.40 (was 1.6) — DO NOT hardcode
```

`BRICK_WORLD_SIZE` is already a derived `const` (`BRICK_EDGE * VOXEL_SIZE`), so it follows automatically — **do
not** write `0.4`. `lod_voxel_size`, `brick_span`, `lod_edge` are all functions of these two — they need **no
edit** and are the SSOT every other module reads through. The whole point of D1 being small is that the
codebase already routes through these functions; the work is finding the few places that **bypassed** them with
a literal.

Mirror in WGSL — `assets/shaders/voxel_raytrace.wgsl`:

```wgsl
const VOXEL_SIZE: f32 = 0.05;                              // was 0.2  (line 24)
const BRICK_WORLD_SIZE: f32 = f32(BRICK_EDGE) * VOXEL_SIZE; // derives 0.40 (line 27) — already derived, no edit
```

`BRICK_AABB_EPSILON = VOXEL_SIZE * 1.0e-3` (both Rust `gpu.rs:80` and WGSL line 33) auto-scales — **this is
correct and intended**: the seam epsilon is *relative to a voxel*, so it shrinks 4× with the voxel and stays
1e-3 of a cell. (Phase A robustness item A4 already flagged making it relative-per-LOD; if A4 landed, confirm
it still derives from `VOXEL_SIZE`.) `MAX_LOD` stays `7` for D1 — D2 may revisit it, but the flip itself does
not change the LOD count; it changes what each LOD *means* metrically.

## D1.1 The re-pin inventory (grep-derived — the exact list of what breaks)

Two failure classes:

1. **Asserts that pin a metric value** — `BRICK_WORLD_SIZE == 1.6`, `brick_span(L) == 1.6·2^L`, world-distance
   tolerances written as `VOXEL_SIZE`-multiples, comments saying "1.6 m" / "0.2 m". Most are **already written
   through the SSOT** (`VOXEL_SIZE` / `BRICK_WORLD_SIZE` / `brick_span(lod)`), so they **re-pin automatically**
   and only the *doc comment* drifts. A minority hardcode the number — those **break the test** and must change.
2. **Reach / scene-dimension assumptions** — view-radius numbers (`~1640 m`), the perf-rig reach ratio, the
   Cornell `INTERIOR=48` (9.6 m → 2.4 m after the flip), physics step heights.

Below, every grep hit, classified `[AUTO]` (re-pins through the SSOT — fix the comment only) or `[BREAK]`
(hardcoded literal — must edit) or `[REVIEW]` (numeric value whose *meaning* shifts even though it compiles).

### A brick is **always 8³ voxels** — so voxel-LOCAL math is flip-invariant
`BRICK_EDGE`, `BRICK_VOXELS`, `voxel_index`, occupancy, palette, the DDA cell-walk, `cornell_voxel`'s voxel
layout, the `.vox` voxel grid — all in **voxel units** — are **completely unaffected**. Only **world-metre**
conversions move. This halves the blast radius: anything that never multiplies by `VOXEL_SIZE`/`brick_span` is
untouched.

### `src/voxel/` production

| File:loc | What | Class | Re-pin |
|---|---|---|---|
| `brickmap.rs:31,34` | `VOXEL_SIZE`, `BRICK_WORLD_SIZE` | **BREAK** | the edit itself (D1.0) |
| `brickmap.rs:8-9,37-45` (doc) | "1.6 m brick of 0.2 m voxels", "~1640 m half-extent" | REVIEW | update comments to 0.4 m / 0.05 m / the new reach |
| `brickmap.rs:323-335` `clipmap_span_scales_with_lod` | asserts `brick_span(lod) == BRICK_WORLD_SIZE·2^lod` and `lod_voxel_size == VOXEL_SIZE·2^lod` | **AUTO** | written through the SSOT — passes unchanged; **verify** |
| `gpu.rs:80` `BRICK_AABB_EPSILON` | `VOXEL_SIZE·1e-3` | AUTO | auto-scales (intended) |
| `gpu.rs:514-520` `pack_brickmap` world_min | `coord·BRICK_WORLD_SIZE` | AUTO | SSOT |
| `gpu.rs:1357` test | `brick_aabb([BRICK_WORLD_SIZE,0,0],0)` | AUTO | SSOT |
| `gpu.rs:1438` test | `assert (span - 4·BRICK_WORLD_SIZE)` (LOD2=4× LOD0) | AUTO | ratio, SSOT |
| `gpu.rs:1508-1513` light-area test | `VOXEL_SIZE²` face area | AUTO | SSOT (area is now 16× smaller — the assert tracks it) |
| `gpu.rs:1615-1617` coarse light area | `(VOXEL_SIZE·2^lod)²` | AUTO | SSOT |
| `gpu.rs:1631` test | `GpuBrickMeta::uniform(..,[1.6,0,-1.6],3)` | **BREAK** | the `[1.6,0,-1.6]` world_min is a **literal** — re-derive as `coord·brick_span(3)` or update to `[0.4,0,-0.4]·…`; confirm against the coord it pairs with |
| `voxelize.rs:22-37` (doc + `SURFACE_SKIN_DEPTH=VOXEL_SIZE`) | skin depth = one voxel | AUTO | SSOT (skin is now 0.05 m — intended: the emissive shell is one voxel) |
| `voxelize.rs:299` (doc) | "cell = 0.2·4 = 0.8 m, span = 1.6·4 = 6.4 m" | REVIEW | comment only (cell 0.05·4=0.2, span 0.4·4=1.6) |
| `source.rs:18,183,202,630` (doc + test) | "0.2 m Sponza voxels"; the `~600×250×370` test slab built in voxel units | REVIEW/AUTO | comments; the slab is **voxel-unit** so geometry is invariant, but its **world size** quarters — if a test frames a camera in metres off it, re-check (see `source.rs` tests) |
| `physics.rs:251-262` `walk_controller` | autostep `max_height: 0.25`, snap `0.1`, comment "one voxel is 0.2 m" | **BREAK** | a one-voxel ledge is now 0.05 m. **Decision (see D1.4):** keep step height at a *gameplay* value (a human steps ~0.25–0.3 m = 5–6 voxels now), do **not** scale it to 0.05 — but update the comment and make it `VOXEL_SIZE`-relative if a multiple is wanted (`5.0·VOXEL_SIZE`). |
| `physics.rs:55` `EYE_HEIGHT=1.6` | camera height (m) | AUTO-keep | **coincidentally** 1.6, but it is a *human eye height in metres*, NOT `BRICK_WORLD_SIZE` — it must STAY 1.6 m. Add a comment so a future agent doesn't "fix" it. |
| `physics.rs:186-187` collider min/max | `vmin·VOXEL_SIZE` | AUTO | SSOT |
| `physics.rs:317` orbit target | `INTERIOR·VOXEL_SIZE·0.5` | AUTO | SSOT (Cornell centre quarters with INTERIOR rescale — see D1.3) |
| `cornell.rs:15-42` | `INTERIOR=48` → `interior_m = 48·VOXEL_SIZE` = 9.6 m → **2.4 m** | **REVIEW → rescale** | see D1.3 — the Cornell box must be **rescaled** (raise `INTERIOR`) or it becomes a 2.4 m dollhouse |
| `mod.rs:1,9,146,218-250,323,449,483` (doc + patch consts) | "0.2 m cubes", `PATCH_*` depths in metres ÷ `VOXEL_SIZE` | AUTO | the worldgen patch (`min_y`/`max_y`/`half_v`) is `metres/VOXEL_SIZE` → 4× more voxels (the 64× cost is real; gated on B/C); comments to 0.05 |
| `edits.rs:236-286` world-space DDA | all `VOXEL_SIZE`-stepped | AUTO | SSOT (the pick DDA walks the finer grid automatically) |
| `raytrace.rs:619` (doc) | "0.2 m LOD0 edit" | REVIEW | comment |
| `raytrace.rs:1434,1462,1492,1530` `gi_bounce_dist` presets | 64 / 24 / 96 / 48 m | **REVIEW** | these are **world-metre** GI reach knobs tied to **scene size**; Cornell's (24) must rescale with the Cornell box (D1.3); Sibenik/Conference (48/96) are tied to the **real scene's** metric size which is **unchanged** by the flip (a re-baked Sibenik is the same metres, just finer) — so **keep** those, **rescale** Cornell's |

### `tests/` (the assert corpus — the bulk of the re-pin)

| File | What pins the old value | Class | Re-pin |
|---|---|---|---|
| `voxel_raytrace_gpu.rs` | CPU-oracle DDA over `VOXEL_SIZE` grid; tolerances `±VOXEL_SIZE`; `s0/s1/s2 = BRICK_WORLD_SIZE / brick_span(1) / brick_span(2)` with comments "1.6/3.2/6.4"; clipmap layout comment "LOD0 coords {0,1} → X [0,3.2)" | AUTO (asserts) / REVIEW (comments + the `[0,3.2)` literal in a comment) | oracle + tolerances re-pin via SSOT; **fix the `// span 1.6` / `[0,3.2)` comments**; the geometry is built in brick coords so it's invariant |
| `voxel_streaming.rs` | `brick_span(0)` nudges, `brick_span(MAX_LOD)` reach, `4·brick_span(0)` ratios, comments "span 1.6/3.2/6.4" | AUTO | all SSOT-relative; **comments only** |
| `voxel_worldgen_perf.rs` | `OLD_DENSE_RADIUS_BRICKS·BRICK_WORLD_SIZE`; the reach-ratio print; `span0=brick_span(0)` step script "1.6 m" | AUTO + **REVIEW** | the reach-ratio bench (`new_view ≥ 15× old`) **still holds numerically** (both sides scale by ¼) BUT the *absolute* metres in the printout quarter — update the doc-comment "~44.8 m" / "1.6 m" strings; the step-count perf assertions are in **bricks** (invariant) |
| `voxel_normal_swap.rs` | `c = INTERIOR·0.5·VOXEL_SIZE` (≈4.8 m comment), targets `(7.5,2,4.5)·VOXEL_SIZE`, ray windows `VOXEL_SIZE·0.25`, comment "≪ a 0.2 m voxel" | AUTO | all SSOT-relative; **fix the "≈4.8 m" / "0.2 m" comments**; INTERIOR-derived metres quarter (or change with the Cornell rescale — keep these tests using their **local** INTERIOR const so they stay self-consistent) |
| `voxel_edit.rs` | oracle DDA over `VOXEL_SIZE`; `±VOXEL_SIZE` tolerances; `expect_t = z·VOXEL_SIZE+2`; comment "0.2 m grid", "[10,11)·0.2", "[2.0,2.2)" | AUTO + REVIEW | asserts re-pin; **fix the literal-metre comments** |
| `voxel_show_through.rs` | oracle DDA; `brick_span(m.lod())`; `c=INTERIOR·0.5·VOXEL_SIZE` "≈4.8 m" | AUTO | comments |
| `voxel_seam_oblique_gpu.rs` | `s=BRICK_WORLD_SIZE`; `top_y=floor_top·VOXEL_SIZE`; `m=0.5·VOXEL_SIZE`; `half=1.5·VOXEL_SIZE`; `view_dist=1.6·span_x` | AUTO | SSOT (note `1.6` here is a *multiplier on span_x*, **not** BRICK_WORLD_SIZE — invariant) |
| `voxel_seam_gpu.rs` | `s=BRICK_WORLD_SIZE` | AUTO | SSOT |
| `voxel_lighting_gpu.rs` | `s=BRICK_WORLD_SIZE`; brick-coord scene; comment "S=BRICK_WORLD_SIZE" | AUTO | SSOT — geometry in brick coords, frame in `s` |
| `voxel_gi_gpu.rs` | `s=BRICK_WORLD_SIZE` (×4 sites) | AUTO | SSOT |
| `voxel_restir_gi_gpu.rs` | `s=BRICK_WORLD_SIZE`; `gi_bounce_dist=40` with comment "reach the ceiling (~1.6 m up)" | AUTO + **REVIEW** | `s` re-pins; the ceiling is now ~0.4 m up — `gi_bounce_dist=40` still **reaches** (it's a max), but update the comment; consider a smaller value to keep the test's intent crisp |
| `voxel_render_headless.rs` | `surf/VOXEL_SIZE`, `wx/VOXEL_SIZE` voxel addressing | AUTO | SSOT |
| `voxel_sponza_residency.rs` | comment "0.2 m voxels"; bounds "to cover the floor + column" | **REVIEW** | Sponza re-bake is **already 0.05 m** (no re-bake) but currently loads 4× oversized vs the 0.2 m engine; after the flip it loads **correct-scaled** — the residency-bounds AABB this test uses may shift 4× → **re-derive the bounds against the post-flip world size** |

### `examples/voxelize_scene.rs`

| Loc | What | Class | Re-pin |
|---|---|---|---|
| `:22,49-53` | `DEFAULT_VOXEL_SIZE = 0.2` (duplicated literal, doc "MUST match brickmap::VOXEL_SIZE") | **BREAK** | change to `0.05`; it's a CLI default (`:62` lets an arg override). **Robustness:** ideally `pub use adventure::voxel::brickmap::VOXEL_SIZE` instead of duplicating — kill the drift class entirely (the comment already admits it's a duplicate). Do this. |

### `assets/shaders/voxel_raytrace.wgsl`

| Loc | What | Class | Re-pin |
|---|---|---|---|
| `:24` | `VOXEL_SIZE = 0.2` | **BREAK** | → `0.05` (D1.0) |
| `:22,25-26,36` (comments) | "0.2 m → 1.6 m brick", "~1.6 km half-extent at clip_half 8" | REVIEW | comments to 0.05 / 0.4 m / the new reach |
| `:27,33,123,130` | `BRICK_WORLD_SIZE`, `BRICK_AABB_EPSILON`, `lod_cell_size`, `brick_span` — all derived | AUTO | follow `VOXEL_SIZE` |
| `:2259-2277` NEE area | `VOXEL_SIZE²` emitter face area | AUTO | SSOT — face area is 16× smaller; the alias-pdf surrogate tracks it (equal-power LOD0 emitters stay consistent) |

### The mechanical procedure
1. Edit the two consts (Rust + WGSL).
2. Fix the **BREAK** rows (5: the two consts, `gpu.rs:1631` literal world_min, `physics.rs` step comment/value,
   `examples` default → ideally re-export the const).
3. `cargo test --lib` + the GPU oracle tests (`voxel_raytrace_gpu`, `voxel_edit`, `voxel_seam*`, `voxel_show_through`,
   `voxel_lighting_gpu`, `voxel_gi_gpu`, `voxel_restir_gi_gpu`): the **AUTO** asserts must pass unchanged (they
   are the proof the SSOT routing is complete). Any AUTO that fails is a hidden hardcode — promote it to BREAK.
4. Sweep the **REVIEW** comments/numbers (mostly doc strings) — these don't fail the build but rot if left.
5. Rescale Cornell + worldgen scene framing (D1.3) and re-bake (D1.2).

## D1.2 Re-bake order

The re-bake is the expensive, B/C-gated part. Order (from `VOXEL_PROGRAM` + `SOTA_REFERENCE` §6):
1. **Sponza** — **already baked at 0.05 m** (prior oversample experiment). NO re-bake. After the flip it loads
   correct-scaled (today it loads 4× oversized against the 0.2 m engine). Just re-pin `voxel_sponza_residency`.
2. **Sibenik, Conference** — McGuire OBJ, ~8–13 MB @0.2 m → ~0.5–0.8 GB @0.05 m each. Re-bake via the
   voxelizer → **`.vxo`** (Phase B format). These fit the in-RAM voxelizer (under the 1.5 B-dense guard).
3. **Bistro Exterior** — 41 MB / 10.3 M vox @0.2 m → ~2.6 GB / ~660 M vox @0.05 m; **>1.5 B dense** ⇒ requires
   **Phase C's tiled bounded-RAM voxelizer** (`docs/TILED_VOXELIZER_PLAN.md`, the out-of-core floodfill). Re-bake
   **last**, gated on C1 landing. Also gated on the `.vxo` MATL emissive reader (C2) for its lamps to light.

Each re-bake: `cargo run --example voxelize_scene <mesh> 0.05 → <scene>.vxo`, then load through the Phase-B
region-streamed `BrickSource`. **The flip + the Sibenik/Conference/Bistro re-bakes are ONE atomic landing** —
between the const edit and the re-bake, a still-0.2 m-baked scene loads 4× wrong-scaled (per `VOXEL_FINE_RESOLUTION_PLAN`
S3). Sponza is the exception (already fine). Acceptance: the gallery loads all four correct-scaled (user visual).

## D1.3 Scene rescaling (Cornell + worldgen)

The flip shrinks every **synthetically-sized-in-voxels** scene 4× in world metres. Two cases:

- **Cornell** (`cornell.rs`): `INTERIOR = 48` voxels was a 9.6 m room (a believable Cornell box). At 0.05 m it
  becomes a **2.4 m dollhouse** — too small to fly a camera in, and every Cornell GPU test frames its camera off
  `interior_m`. **Fix: raise `INTERIOR` to keep ~9.6 m** → `INTERIOR = 192` (192·0.05 = 9.6 m), a clean ×4. This
  keeps the box metric-identical and 4× finer-walled (a thicker `WALL` in voxels too — scale `WALL` ×4 to keep
  the same metric wall thickness). **Cascade:** Cornell's `gi_bounce_dist` preset (`raytrace.rs:1462`, 24 m) and
  `physics.rs:317` orbit target re-derive from `interior_m` automatically once `INTERIOR` is set — they read the
  SSOT. The Cornell GPU tests (`voxel_lighting_gpu`, `voxel_gi_gpu`, `voxel_restir_gi_gpu`, `voxel_normal_swap`,
  `voxel_show_through`) build geometry in **voxel/brick coords** and frame in `BRICK_WORLD_SIZE`/`INTERIOR`, so
  they re-pin automatically — but **the brick COUNT quadruples per axis** (a 192-voxel room is 24 bricks vs 6),
  so any test asserting a specific brick count must update. **Grep `INTERIOR` in tests after the rescale.**
- **Worldgen**: the patch consts (`mod.rs` `PATCH_*`) are **metres ÷ VOXEL_SIZE** → they already produce 4× more
  voxels at the same world coverage (the 64× cost). The worldgen world is **metrically anchored** (heights in
  metres), so it does **not** rescale — it just gets finer. No geometry change; only the cost (gated on B/C).

> Open question (D1.3): is Cornell's box a *test fixture* (keep at exactly 9.6 m for oracle continuity) or a
> *gallery showpiece* (could grow)? Recommend: keep 9.6 m (`INTERIOR=192`) for continuity; the GPU oracle tests
> are the regression net and a metric-identical box keeps them honest.

## D1.4 The knobs that scale (and the ones that DON'T)

Three categories — get these right or the flip silently breaks gameplay/GI:

- **Scales with VOXEL_SIZE (auto, via SSOT):** `BRICK_AABB_EPSILON`, `SURFACE_SKIN_DEPTH`, the NEE voxel-face
  area, every `coord·brick_span`/`v·VOXEL_SIZE` world conversion, the edit/pick world-DDA. **No action** — they
  read the SSOT. Verify they still derive (not re-hardcoded by Phase A).
- **Scales with the SCENE, not the voxel:** `gi_bounce_dist` presets are **world-metre GI reach** tied to scene
  size. Cornell's rescales **with the Cornell box** (auto, if read off `interior_m`; today it's a literal `24` —
  **make it `~2.5·interior_m()` or similar so it tracks the rescale**). Sibenik/Conference/Bistro's are tied to
  the **real metric scene size**, which the flip does **not** change — **keep** them.
- **Does NOT scale (gameplay/physics in real metres):** `EYE_HEIGHT = 1.6 m` (human height — coincidentally the
  old `BRICK_WORLD_SIZE`, must **stay** 1.6 m; comment it loudly). The player **autostep** `max_height = 0.25 m`:
  a human steps ~a 25–30 cm ledge regardless of voxel size; **keep it a gameplay value** (now 5 voxels, not 1) —
  update the comment from "one voxel is 0.2 m" and optionally express as `5.0·VOXEL_SIZE` if a voxel-multiple is
  wanted, but **do not** shrink it to 0.05 (a 5 cm max step is unwalkable terrain). `snap_to_ground = 0.1 m`
  likewise stays a gameplay value (2 voxels now). The character **collider radius/height** (grep
  `KinematicCharacter`/capsule in `physics.rs`) — if any is voxel-derived, it must stay a human ~0.3 m radius /
  ~1.8 m height. The Rapier **physics timestep** is voxel-independent — no change.

> **The trap (call it out for the agent):** the flip makes "one voxel" 4× smaller, so any constant that *meant*
> "one voxel" but represents a *physical/gameplay* quantity (eye height, step height) must be **un-coupled** from
> the voxel, not scaled with it. The grep distinguishes them: `VOXEL_SIZE`-relative = scales; bare metre literal
> with a gameplay meaning = stays.

## D1.5 D1 acceptance gate
- Both feature builds green (`cargo build` + `cargo build --features editor`), **zero warnings**.
- All AUTO asserts pass **unchanged** (proves SSOT routing complete); all BREAK rows edited; REVIEW comments swept.
- GPU oracle suite pixel/byte-identical **after** the Cornell `INTERIOR` rescale (the rescale keeps the box
  metric-identical, so the oracle hits are at the same world positions → identical).
- Gallery loads Sponza/Sibenik/Conference/Bistro correct-scaled at 0.05 m (user visual).
- Physics: the player walks the worldgen terrain (no 5 cm-step paralysis), eye at 1.6 m.
- Perf/VRAM measured before/after on the rig (the 64× voxel-count cost is expected and bounded by B/C + surface-only).

---

# D2 — screen-error LOD (the reach mechanism)

## D2.0 Why the flip forces this

The clipmap's fine band reaches `clip_half · brick_span(0)` in metres before stepping to LOD1. At 0.2 m,
`brick_span(0)=1.6 m`, `clip_half=8` → ~12.8 m of LOD0 before the first coarsening. At 0.05 m,
`brick_span(0)=0.4 m` → **~3.2 m**. The fine detail you flipped *for* now evaporates at arm's length.

The naive fix — quadruple `clip_half` (8→32) or raise `MAX_LOD` — is **wrong** because the *correct* LOD at a
world point depends on **how big a voxel projects on screen**, which depends on **camera FOV and viewport
resolution**, neither of which `clip_half` knows. A 4K viewport at 60° FOV needs finer LOD at the same distance
than a 720p viewport at 90°. A fixed `clip_half` over-resolves (wastes VRAM) on a wide/low-res view and
under-resolves (visible blockiness) on a narrow/high-res one. This is exactly the Nanite / GigaVoxels
**screen-error** insight: pick LOD by **projected footprint**, not raw distance.

## D2.1 The `want_lod` formula

A voxel at LOD `lod` has world edge `lod_voxel_size(lod) = VOXEL_SIZE · 2^lod`. At distance `d` from the camera,
a feature of world size `w` projects to a pixel height of (standard perspective projection):

```
pixels(w, d) = (w / d) · (viewport_h / (2 · tan(fov_y / 2)))
```

Let `K = viewport_h / (2 · tan(fov_y / 2))` — the **camera's pixels-per-radian-ish constant** (one scalar,
computed once per frame from the live `Projection` + `Camera` viewport). Then a LOD-`lod` voxel at distance `d`
covers:

```
pixels_per_voxel(lod, d) = lod_voxel_size(lod) · K / d
        = VOXEL_SIZE · 2^lod · K / d
```

We want the **finest** `lod` whose voxel does not exceed a target on-screen size `target_px` (the screen-error
budget — e.g. `target_px = 1.0` for "one voxel ≈ one pixel", a tunable; `>1` trades sharpness for VRAM/perf):

```
want_lod(d) = clamp( ceil( log2( target_px · d / (VOXEL_SIZE · K) ) ), 0, MAX_LOD )
```

Derivation: solve `VOXEL_SIZE · 2^lod · K / d ≤ target_px` for `lod` → `2^lod ≤ target_px · d / (VOXEL_SIZE · K)`
→ `lod ≤ log2(...)`. The finest LOD whose footprint is ≤ target is the **ceil of the closest LOD that hits the
budget** (ceil so we never pick a finer-than-needed level; `floor` would over-resolve by one). Clamp to
`[0, MAX_LOD]`. `d` is the world distance from `cam_world` to the **brick's world centre** (`(coord+0.5)·brick_span(lod)`
— but since `want_lod` is queried per LOD0-coord in `brick_lod`, use the LOD0-coord centre `(coord+0.5)·brick_span(0)`,
consistent with the existing `brick_lod` signature).

Implementation as a pure function next to `brick_lod` in `streaming.rs`:

```rust
/// Screen-error target LOD for a world point: the FINEST lod whose voxel projects to ≤ target_px pixels.
/// `k = viewport_h / (2·tan(fov_y/2))` is the per-frame camera constant (computed once, passed in cfg/ScreenError).
#[inline]
pub fn want_lod(world: [f32; 3], cam_world: [f32; 3], se: &ScreenError) -> u32 {
    let dx = world[0] - cam_world[0];
    let dy = world[1] - cam_world[1];
    let dz = world[2] - cam_world[2];
    let d = (dx*dx + dy*dy + dz*dz).sqrt().max(se.min_dist); // clamp near-zero distance
    let ratio = se.target_px * d / (VOXEL_SIZE * se.k);
    // log2(ratio), ceil, clamp. ratio<=1 ⇒ lod 0.
    let lod = ratio.max(1.0).log2().ceil() as i32;
    lod.clamp(0, MAX_LOD as i32) as u32
}
```

`ScreenError { k: f32, target_px: f32, min_dist: f32 }` — a small struct (a Bevy resource or a field on
`StreamingConfig`). `target_px` and `min_dist` are **editor sliders** (knobs-as-uniforms discipline); `k` is
recomputed each frame from the camera.

Computing `k` (in `stream_voxel_rt_residency`, where the camera is already queried): add `&Projection` and the
`Camera` viewport to the query (the system today queries only `&GlobalTransform With<SdfCamera>`). For a
`Projection::Perspective(p)`: `fov_y = p.fov` (Bevy's vertical FOV, radians); `viewport_h =
camera.physical_viewport_size().y`. Then `k = viewport_h as f32 / (2.0 * (fov_y * 0.5).tan())`. (For an
orthographic projection — not the current camera — `k` degenerates; gate to perspective, fall back to the
distance-shell policy otherwise.)

> Sanity check the formula against the flip: to restore the old 0.2 m LOD0 reach (~12.8 m of LOD0), pick
> `target_px` so `want_lod(12.8) == 1` boundary, i.e. `target_px ≈ VOXEL_SIZE·K/12.8 · 2`. At 1080p/45° FOV,
> `K = 1080/(2·tan(22.5°)) ≈ 1303`; `VOXEL_SIZE·K = 0.05·1303 ≈ 65`; `want_lod=1` boundary at
> `d = VOXEL_SIZE·K/target_px`. For `target_px=2`, LOD0 reaches `d ≈ 65/2 ≈ 32 m` — **farther** than the old
> 12.8 m, and now *correctly* FOV/res-aware. The point: `target_px` is the single dial for the quality/VRAM
> tradeoff, replacing a hand-tuned `clip_half`.

## D2.2 How it composes with the exact-tiling clipmap (the invariant to preserve)

The clipmap's **exact nested tiling** (`desired_clipmap`/`level_box`/`level_hole` in `streaming.rs` — every
world point covered by exactly one level, no overlap/gap, levels snap to the 2×-coarser grid) is **load-bearing
and must not break**. Screen-error does **not** replace the tiling; it **replaces the per-level half-extent
choice**. Two valid integrations, in increasing scope:

**(A) Per-level `clip_half` derived from `want_lod` (minimal, recommended first).** Today every level uses the
same `clip_half_bricks`. Instead, derive the LOD-`L`→LOD-`(L+1)` **transition distance** from the screen-error
target: the distance at which `want_lod` crosses from `L` to `L+1` is `d_L = VOXEL_SIZE · K · 2^L / target_px`.
Convert to a per-level half-extent in LOD-`L` bricks: `half_L = ceil(d_L / brick_span(L))` (then `snap_even_odd`
as today). Feed `half_L` into `level_box(cam, L, half_L)`. The tiling machinery is **unchanged** — it still
snaps each level's box to the coarser grid and carves the finer level's hole, so **no-overlap/no-gap still holds
by construction** (the proof in `clipmap_tiles_exactly_no_overlap_no_gap` depends only on the even/odd snap, not
on `half` being constant across levels — **verify this**: the snap and hole logic are per-level already, so a
per-level `half` is a drop-in *provided each `half_L ≥ 2`* for a proper annulus). This keeps the entire residency
diff/reconcile/cap path intact and makes the reach FOV/res-aware with a ~10-line change to how `half` is chosen.

> Caveat for (A): `level_hole(L)` is computed from `level_box(L-1, half_{L-1})` — so the per-level halves must be
> **monotonic** (`half_{L-1}` in LOD-(L-1) bricks must downsample to ≤ `half_L` region in LOD-L bricks) or the
> hole could exceed the box. Screen-error gives monotonic transition distances (`d_L = d_0·2^L`), and
> `half_L = ceil(d_L/brick_span(L)) = ceil(d_0/brick_span(0))` is **constant in brick units** across levels (the
> `2^L` cancels!) — so per-level `half` is actually **the same brick count every level**, differing only if
> `target_px`/rounding nudges it. This means **(A) reduces to: pick the single `clip_half` from the screen-error
> target** `clip_half = ceil(VOXEL_SIZE·K/(target_px·brick_span(0)))`, and the existing constant-half tiling is
> already correct. The screen-error work is then *just deriving `clip_half` from the camera each frame* instead
> of a hand-set slider — minimal, and it preserves the tiling trivially.

**(B) Per-point `want_lod` overriding `brick_lod` (fuller Nanite, defer).** `brick_lod(coord, cam, cfg)`
currently returns the finest *resident* level covering a point. A screen-error variant would return
`max(brick_lod(...), want_lod(world, cam, se))` — i.e. never finer than the clipmap holds, but coarsen further
out where screen-error allows, even *inside* the clip volume (e.g. a grazing-angle floor far across a large
room is sub-pixel before it leaves the LOD0 box). This needs `want_lod` mirrored in **WGSL** (the shader's LOD
selection reads the same `K`/`target_px` from a uniform) and a matching CPU residency policy so the resident set
matches what the shader asks for. **Defer (B) to a Phase-B/-C follow-up** — it couples to the ray-guided request
path; (A) captures the headline reach win without touching the shader.

**Recommendation:** ship **(A)** — derive `clip_half` (and optionally a per-level cap) from the per-frame camera
`K` + `target_px`, leaving `desired_clipmap`/tiling untouched. It is the screen-error reach mechanism with the
exact-tiling invariant preserved **by construction** (constant `clip_half` is already proven correct), and it
removes the hand-tuned `clip_half` slider in favor of `target_px` (a perceptual, FOV/res-independent dial).

## D2.3 Residency-test + perf-rig changes
- **`streaming.rs` tests:** add `want_lod` unit tests — monotonic in `d` (farther ⇒ coarser-or-equal),
  clamps `[0,MAX_LOD]`, and the **FOV/resolution sensitivity** test: same `d`, larger `K` (higher res / narrower
  FOV) ⇒ finer (lower) lod. Add a test that the screen-error-derived `clip_half` reproduces the existing
  `clipmap_tiles_exactly_no_overlap_no_gap` (the tiling is invariant under choosing `half` from `K`). Add a
  **reach-at-target-pixel-error** test: at `target_px=1`, the LOD0→LOD1 boundary distance equals
  `VOXEL_SIZE·K/target_px` within one brick.
- **`tests/voxel_worldgen_perf.rs`:** the existing reach bench prints absolute metres — extend it to a
  **screen-error sweep**: for a grid of `(fov_y, viewport_h, target_px)`, report the derived `clip_half`, the
  resident-brick count (post surface-only), the meta+AABB+voxel VRAM, and the **LOD0 reach in metres**. The
  acceptance figure: **at a fixed VRAM budget, the LOD0 reach (metres) at the gallery's nominal FOV/res is ≥ the
  old 0.2 m engine's ~12.8 m**, while VRAM stays bounded — i.e. the flip's quartered reach is recovered *without*
  a brute `clip_half` bump, and it tracks FOV/res. Reuse the surface-only residency from Phase A
  (`VOXEL_LARGE_SCENE` Phase A) so the resident set is Θ(H²): the screen-error `clip_half` can grow without the
  cubic blow-up.
- **Decoupling check:** assert that flipping resolution 1080p→4K *automatically* refines the LOD distribution
  (resident_lod_counts shifts finer) with **no code/slider change** — the proof that reach is no longer tied to a
  hand-tuned `clip_half`.

## D2.4 D2 acceptance gate
- `want_lod` unit tests green (monotonic, clamped, FOV/res-sensitive); the tiling-invariance test green.
- Both feature builds, zero warnings.
- Perf rig: LOD0 reach at gallery FOV/res ≥ old engine's, VRAM bounded (with Phase-A surface-only), exponent
  stays Θ(H²); the resolution-sweep shows automatic refinement.
- Visual (user): the fine band reaches plausibly far at the gallery camera; no LOD-seam cracks (the exact tiling
  held); changing FOV/res visibly re-balances detail without a slider.

---

## Summary for the implementer
- **D1 = one const edit + a mechanical re-pin.** The blast radius is small because nearly everything routes
  through `VOXEL_SIZE`/`BRICK_WORLD_SIZE`/`brick_span(lod)` (the **AUTO** rows re-pin for free; they're the
  *proof* the SSOT is intact). The real edits are 5 **BREAK** sites (the two consts, `gpu.rs:1631` literal
  world_min, `physics.rs` step height/comment, `examples` duplicated default → re-export it), the Cornell
  `INTERIOR` rescale (48→192 to stay 9.6 m), the gameplay knobs that must **NOT** scale (eye height, step
  height — physical metres, not voxels), and the Sibenik/Conference/Bistro re-bakes (gated on Phases B+C;
  Sponza already 0.05 m).
- **D2 = `want_lod` from projected footprint.** `want_lod(d) = clamp(ceil(log2(target_px·d/(VOXEL_SIZE·K))),0,MAX_LOD)`
  with `K = viewport_h/(2·tan(fov_y/2))`. The neat result: under exact tiling the screen-error half-extent is a
  **constant brick count across LODs** (the `2^L` cancels), so D2 collapses to *deriving `clip_half` from the
  per-frame camera* instead of hand-tuning it — preserving the no-overlap/no-gap tiling **by construction** and
  removing the FOV/res-blind slider. Defer the fuller per-point/WGSL `want_lod` (option B) to the ray-guided path.
- **Acceptance:** D1 — both builds green, AUTO asserts unchanged, GPU oracle pixel-identical after the metric-
  preserving Cornell rescale, gallery correct-scaled, player walks. D2 — `want_lod` tests green, LOD0 reach ≥ old
  engine at bounded VRAM (with Phase-A surface-only), reach tracks FOV/res automatically.
- **Open questions:** (1) Cornell box — keep at 9.6 m for oracle continuity (recommended) vs grow? (2) confirm
  `clipmap_tiles_exactly_no_overlap_no_gap` holds under a per-frame-derived `clip_half` (expected yes — the proof
  is snap-based, half-independent); (3) does Phase A's A4 already make `BRICK_AABB_EPSILON` per-LOD, or does it
  still derive from `VOXEL_SIZE` (either is fine post-flip, just confirm)?
```
