# SDF Renderer + Editor -- Structure & Maintainability Roadmap

> **Status:** living backlog. Generated 2026-06-02 from a multi-agent code audit (5 survey agents
> -- modularization, refactor/dead-code, API boundaries, editor structure, cross-cutting -- then
> adversarial verification of every high-impact finding, 14 agents total). This is the *structure*
> companion to `PERF_ROADMAP.md`: it targets navigability, duplication, and encapsulation, not
> frame time. Items are mostly independent -- pick by impact/effort, tick the box.

## How to read this

Every item was checked against the actual code; the survey impacts were then **corrected in
adversarial verification** -- several `high`s dropped to `medium` because the payoff is real but
the churn is large or the code is test-only. Where a verdict landed, its corrected impact and the
key caveat are recorded inline (**Verified:**).

The three load-bearing constraints that gate *how* these are done:

1. **The CPU↔GPU byte contract is sacred.** `ChunkLookup`/`BrickTile` layout, `dir_index`,
   `chunk_gpu_key`, the std430 encoders, and the delta-upload protocol are mirrored between Rust
   and WGSL and guarded by the differential tests (`chunk.rs` churn, `bake_scheduler.rs` lifecycle,
   `tests/sdf_gpu_rig.rs`). Any dedup here must keep those tests green at every step.
2. **`chunk.rs` is deliberately pure** (imports only `bevy::math`). Don't pull render types into it
   without a conscious call.
3. **Two build configs, zero warnings.** `cargo build` AND `cargo build --features editor`; the
   whole `editor/` tree is feature-gated, so a plain build won't catch editor breakage.

Legend: **Impact** is verification-corrected. **Effort** S/M/L. Checkbox = done. IDs are stable
handles (D/C/M/A/E/T/X) for referencing an item in a future session.

> **Note on `cargo fmt`:** this codebase is **not** rustfmt-styled -- running fmt reformats 50+
> files. Every refactor below is a manual, surgical move. Never run `cargo fmt`.

---

## Tier 0 -- Ready now: dead code, stale docs, isolated fixes

Small, isolated, near-zero-risk. Each is a few lines and unblocks nothing else -- do them whenever.
Knocking these out first also shrinks the surface the bigger splits have to move.

### [x] D1. Delete dead atlas API + the never-read `last_bake_was_full`
`impact: medium` * `effort: small` * `source: refactor-deadcode + api-boundaries`

- *Now:* `atlas.rs:396 ring_brick_keys` and `atlas.rs:425 bricks_in_aabb_lod` have **no callers**
  (superseded by `bake_scheduler::chunks_in_aabb_windowed`); both are `pub`, so they read as live
  API. `atlas.rs:147 last_bake_was_full` is **set once** (`bake_scheduler.rs:948`) and **read
  nowhere** -- its doc claims "render world reads this" but the real grow signal flows through
  `tiles.high_water()` (`render.rs:1225`).
- *Approach:* Delete `ring_brick_keys` and `bricks_in_aabb_lod`. Remove `last_bake_was_full` at all
  three sites: field (`atlas.rs:147`), `Default` init (`atlas.rs:169`), and the write
  (`bake_scheduler.rs:948`). Fix the stale doc comment. (`ring_window_coords`/`coord_in_window` are
  still used by tests -- keep, consider `#[cfg(test)]`-gating `coord_in_window`.)
- *Verified:* confirmed. `last_bake_was_full` is genuinely dead; this is the clean carve-out from
  the (rejected) larger SdfAtlas split -- see X1. `SdfAtlas` is a plain `Resource`, not Reflect, so
  no `register_type` concern.

### [x] D2. Remove the no-op `upload_sdf_buffers` system + unused single-resolution addressing
`impact: low` * `effort: small` * `source: refactor-deadcode`

- *Now:* `mod.rs:1217 upload_sdf_buffers(_atlas)` has an **empty body** but is still added to the
  Update schedule (`mod.rs:563`) -- a wasted system registration + schedule edge.
  `mod.rs:435 brick_id` / `mod.rs:364 bricks_per_axis` are the non-LOD single-resolution path the
  clipmap replaced; `brick_id`'s own comment says "Kept for the non-LOD path" which no longer exists.
- *Approach:* Remove `upload_sdf_buffers` + its entry in the Update tuple. Remove
  `brick_id`/`bricks_per_axis` from `SdfGridConfig` (grep-confirm no `feature=editor`/test caller first).

### [x] D3. Rewrite stale toroidal-swap comments (they describe the deleted sorted array)
`impact: medium` * `effort: small` * `source: refactor-deadcode`

- *Now:* `chunk.rs:858-877` (churn test) comments reference "row shifting", "sentinel tail", and a
  `sentinel_tail_from` floor that **no longer exists** -- the directory is fixed-size now.
  `chunk.rs:259` doc still says `LiveChunkTables` mirrors "sorted `chunks` row order".
  `docs/TOROIDAL_MIGRATION_PLAN.md:21` still describes the OLD "sparse sorted array".
- *Approach:* Rewrite the churn-test block to describe the fixed-size directory + in-place
  dirty-slot delta (no shift; sentinel is just an empty-slot tag). Fix the `LiveChunkTables` doc to
  "dense toroidal directory". Mark the migration-plan doc completed/historical. Comments only.

### [x] D4. `draw_lod_rings` debug overlay uses stale ring math (it lies about residency)
`impact: medium` * `effort: small` * `source: refactor-deadcode`

- *Now:* `mod.rs:1180` claims the boxes use "the same ring_origin math the bake centres each ring
  on", but the bake now centres on `bake_scheduler::ring_chunk_origin` (chunk-space, **with**
  `recenter_snap_chunks` hysteresis) while `draw_lod_rings` calls `config.ring_origin` (brick-space,
  **no** snap). The boxes recenter every chunk crossing instead of on the snapped lattice, so the
  overlay drifts from the actual resident ring -- the exact thing it exists to verify.
- *Approach:* Derive the box origin from `bake_scheduler::ring_chunk_origin` (convert chunk origin →
  world via `chunk::chunk_min_world`). Update the now-correct comment. Afterwards `config.ring_origin`
  has only test callers and could move behind `#[cfg(test)]`. Debug-overlay only -- no render impact.

### [x] D5. Fix the misplaced doc fragment + cross-reference the twin CSG folds in `edits.rs`
`impact: low` * `effort: small` * `source: refactor-deadcode`

- *Now:* `edits.rs:741 fold_csg` (tracks material) and `edits.rs:835 fold_csg_dist_indexed`
  (distance-only) share identical fold control flow and must stay in sign-agreement (a regression
  test exists because they can diverge). `edits.rs:789-790` has an orphaned doc fragment "Same fold
  rules as" that got attached to `bake_content_hash` instead of `fold_csg_dist_indexed`.
- *Approach:* Lowest-churn: move the misplaced doc fragment above `fold_csg_dist_indexed`, add a
  comment pinning the two as deliberate mirrors. (Unifying into one index-iterator + optional
  material-sink core is optional and higher-churn -- the material branch must be preserved exactly.)

### [x] D6. `GizmoMesh` derives `Reflect` but is never registered
`impact: low` * `effort: small` * `source: cross-cutting`

- *Now:* `gizmo_render/mod.rs:27 GizmoMesh` derives `Reflect` but is never `register_type`'d (it's a
  field of `GizmoDraw`, which isn't Reflect). It's the lone derive-without-register in the crate (audit
  found 55 `register_type` calls cover all `#[reflect(Component/Resource)]` types). Invariant #4.
- *Approach:* Decide intent -- if it's a per-frame render buffer (it is), **drop `Reflect`**. If it
  should be inspectable, `register_type` it in `GizmoRenderPlugin::build`.

### [x] D7. Update the stale `CLAUDE.md` architecture map
`impact: low` * `effort: small` * `source: cross-cutting`

- *Now:* `lib.rs:1-13` declares `pub mod assets, gizmo_render, node` but `CLAUDE.md:9`'s Modules list
  and Architecture tree omit all three. The doc also says only `SdfRenderPlugin` is registered
  separately in `main.rs`, while `GizmoRenderPlugin` is actually installed implicitly from
  `sdf_render` (see X2).
- *Approach:* Add `node/` (NodePlugin), `gizmo_render/` (GizmoRenderPlugin), `assets/` (AssetsPlugin)
  to the module map; note where each is registered. Docs only. (Pairs with X2.)

---

## Tier 1 -- Dedup the CPU↔GPU contract (highest-value structural wins)

These are the *dangerous* duplications: the brick-resolve math and the upload protocol are
hand-copied across production + tests, so a contract change must be edited in 4-6 places or a test
silently stops exercising the real path. Consolidating **reduces** drift risk. Do C1 first (it's the
verified `high`). Keep the differential tests green at every step.

### [x] C1. Collapse the triplicated upload decision behind a `LiveChunkTables::upload()` accessor
`impact: high` * `effort: medium` * `source: api-boundaries`

- *Now:* The full-rebuild-vs-delta decision + the `+50%` tile-run headroom formula
  (`(needed_slots + needed_slots/2).max(needed_slots + TILE_RUN_SLOT)`) + the dirty-set paging is
  **verbatim in three places**: production `render.rs:1240-1289`, the `chunk.rs:938-959` churn test,
  and `bake_scheduler.rs:1308 apply_table_delta`. `dirty_rows`/`dirty_slots` are `pub BTreeSet<u32>`
  (`chunk.rs:280-282`) **solely** so these external sites can iterate them -- a leaked internal
  representation.
- *Approach:* Give `LiveChunkTables` a `ChunkUpload` accessor returning `Full{...}` or
  `Delta{row_updates, region_updates}` computed against a caller-supplied current capacity,
  encapsulating the headroom policy. `render.rs` extract, the churn test, and `apply_table_delta` all
  call it -- deleting two copies; `dirty_rows`/`dirty_slots` drop to private/`pub(crate)`.
- *Verified:* **worth-it, impact stays high.** Caveats: (1) the accessor must **fill caller-owned
  Vecs** (clear+extend), not allocate fresh per frame -- preserve the perf work that just landed.
  (2) The three copies differ in their *tail*: `render.rs` sizes `tile_run_data` to slot high-water
  and pads later in `prepare` (`render.rs:1591-1595`); the tests `resize(cap_slots, default)` inline.
  Expose the *decision + delta data* and leave final buffer sizing to each caller. (3) Don't fold
  `to_gpu_lookup`/`to_gpu_tile` into it -- those belong to the deferred GPU write.

### [x] C2. Delete `render.rs`'s duplicate GPU record structs + collapse to one encoder
`impact: medium` * `effort: medium` * `source: api-boundaries`

- *Now:* `render.rs:32-47 GpuChunkLookup`/`GpuBrickTile` are field-for-field copies of
  `chunk.rs:111-128 ChunkLookup`/`BrickTile`; `to_gpu_lookup`/`to_gpu_tile` (1169-1182) convert,
  `encode_lookup`/`encode_tile` (1538-1551) hand-serialize, and `tile_origin` (1160-1166) is a
  verbatim copy of `chunk.rs:133 tile_atlas_base`. One logical field add = **four edit sites** with no
  compiler link.
- *Approach:* Collapse the converters/encoders to one encoder living next to the struct it serializes;
  replace `tile_origin` with `tile_atlas_base` (unpacking its `col|row<<16`).
- *Verified:* **worth-it-with-caveats, medium.** (1) Do **not** swap manual encoding for
  encase/`StorageBuffer` in the same pass -- that changes the bytes-on-the-wire (the GPU-rig-tested
  std430 contract); separate step if pursued. (2) The `ShaderType` derive is **still load-bearing** at
  `render.rs:1798/1812` (min-binding-size) -- don't delete as "unused". (3) `tile_origin`
  (`render.rs:2457`) needs **unpacked** `(col,row)` for `copy_buffer_to_texture`. (4) Deriving
  `ShaderType` on `chunk.rs` types breaks its documented purity (constraint #2) -- prefer keeping the
  GPU derive in `render.rs` and just collapsing the converters/encoder.

### [x] C3. Extract the 4x `shader_resolve` unpack + 2x table-delta into shared test-support helpers
`impact: medium` * `effort: medium` * `source: refactor-deadcode`

- *Now:* The `dir_index → tag-check → occupancy-bit → popcount-offset` unpack is hand-rewritten at
  `chunk.rs:606`, `chunk.rs:885`, `bake_scheduler.rs:1282`, and `tests/sdf_gpu_rig.rs:895-916` (and
  likely a 5th in `tests/sdf_lifecycle_gpu.rs` -- grep before consolidating). It mirrors
  `brick.wgsl::find_chunk`; a packing change must touch all copies or a test silently stops testing
  the real path.
- *Approach:* Add a `resolve_via_tables(rows, tiles, r, ck, local) -> Option<BrickTile>` in `chunk.rs`
  as **THE** documented shader mirror; every test calls it. The `ChunkTables` variant becomes a thin
  wrapper; the gpu-rig caller reads `.atlas_base` off the result.
- *Verified:* **worth-it-with-caveats, medium.** Must be `#[doc(hidden)] pub` (NOT `#[cfg(test)]`) so
  the `tests/` integration crate can reach it -- `dir_index`/`chunk_gpu_key`/`full_tables` are already
  pub for exactly this reason. Watch the non-test `dead_code` lint (`#[cfg_attr(not(test),
  allow(dead_code))]` if needed). The four copies differ in signature/return -- standardize on
  `(rows, tiles, r, ck, local) -> Option<BrickTile>`, don't force one shape blindly.

### [x] C4. Make `R = ring_bricks / CHUNK_BRICKS` a single `SdfGridConfig` method
`impact: medium` * `effort: small` * `source: api-boundaries`

- *Now:* The ring-chunks-per-axis derivation is recomputed at `chunk.rs:180/314/779/837/872`,
  `bake_scheduler.rs:241` (a free `ring_chunks_per_axis`), and the half-window
  `ring_bricks/CHUNK_BRICKS/2` at `bake_scheduler.rs:236` -- with `as i32`/`as u32` casts varying. Three
  caches of one derived constant (`LiveChunkTables.r`, `bake_scheduler`'s fn, `ChunkTables.r`) that must
  agree for `dir_index` to resolve.
- *Approach:* Add `SdfGridConfig::ring_chunks_per_axis(&self) -> i32` (+ `directory_len`,
  `ring_half_chunks`) as the single source; route all sites through it. `SdfGridConfig` is already the
  config SSOT threaded everywhere. Watch the i32/u32 cast at the half-window stays floor-correct;
  covered by the `ring_chunks_per_axis` test + dir_index parity tests.

### [~] C5. Factor the std430 byte-encoders + the duplicated 12-entry atlas bind-group
`impact: medium` * `effort: medium` * `source: refactor-deadcode + cross-cutting`
> **Folded into M1.** The high-value part (the `atlas_bind_group_1` dedup across the two render
> nodes) only makes sense once `render.rs` is split into `gbuffer.rs`/`cone.rs`, so it lands there.
> The per-struct `encode()` part is not cross-struct duplication (each serializes distinct fields on
> a load-bearing std430 layout M1 will relocate anyway), so it's done in the same pass — not as a
> standalone churn commit.

- *Now:* `render.rs` hand-rolls LE byte encoding in 4 places (`GpuSdfMaterial` 1313-1338,
  `GpuJobHeader` 2278-2292, `GpuEdit` 2301-2316, plus `encode_lookup`/`encode_tile`). The 12-binding
  atlas bind group 1 is built **identically** in `SdfGBufferNode` (`render.rs:535-564`) and
  `SdfConeNode` (`render.rs:2157-2186`) -- ~30 lines each, must stay in sync by hand.
- *Approach:* Factor `fn atlas_bind_group_1(device, layout, gpu_atlas) -> BindGroup` called by both
  nodes (the high-confidence win -- fold this into M1). For the encoders, prefer `encase`'s
  `write_into` if the structs already derive `ShaderType`, else give each an `encode(&self, &mut
  Vec<u8>)`. *Risk:* the std430 layout (emissive pad at offset 64; dist-row pad to 256B) is
  load-bearing -- verify byte-for-byte parity with a test before/after.

---

## Tier 2 -- Modularization (the big file splits)

Large, mechanical, one-time churn. Each improves navigability but threads shared state, so expect a
wide `pub(super)`/`use super::{...}` surface. **Verification corrected all of these from high to
medium** -- the payoff is real but dampened by shared-state plumbing. Do them when a file becomes a
genuine friction point, not speculatively. **Don't split `chunk.rs`** (X3).

### [x] M1. Split `render.rs` (2511 LoC) into a `render/` directory by render-pass concern
`impact: medium` * `effort: large` * `source: modularization`
> **Done (landed incrementally).** `render.rs` → `render/mod.rs` + four cohesive submodules:
> `bake.rs` (GPU brick-bake compute), `pbr_textures.rs` (BC7 array streaming), `cone.rs` (cone
> prepass), `atlas_upload.rs` (chunk-table GPU mirror + encode/upload). **C5/A4 folded in**: the
> 12-entry atlas bind group is now one shared `atlas_bind_group_1()` helper (was copy-pasted in the
> G-buffer + cone nodes). `render/mod.rs`: **2398 → 1259 LoC (−47%)**. The remaining core —
> `SdfCameraData` (the camera uniform, referenced ~12× across both view nodes, pipeline init, and
> `register_type`/`ExtractComponentPlugin`), `SdfPipeline`/`SdfGpuAtlas`, the G-buffer + combine view
> nodes, camera/material prepare, and pipeline init — is **kept central by design**: it threads the
> shared state across every pass, so per the verification verdict ("the state doesn't partition along
> pass lines") splitting it would create wide `super::`/re-export plumbing for the most-shared type
> with minimal cohesion gain. Submodules reach shared types via `use super::*`; the `sdf_render`
> siblings via `super::super`. Each peel was its own verified commit (GPU rigs green throughout).

- *Now:* One flat file mixes ~6 independent subsystems (camera prepare, atlas/chunk-table upload, PBR
  texture streaming, cone prepass, brick bake, deferred G-buffer + combine). To read the bake path you
  scroll past 1500 lines of unrelated upload/pipeline-init.
- *Approach:* `render/mod.rs` keeps `SdfRenderPlugin` + shared GPU mirror types + `SdfPipeline`/
  `SdfGpuAtlas`. Split into `camera.rs` (847-986), `gbuffer.rs` (G-buffer + combine + node), `atlas_upload.rs`
  (1155-1762), `textures.rs` (PBR streaming), `cone.rs` (cone pipeline + node), `bake.rs` (GPU
  brick-bake half + `BAKE_*` byte-contract consts, which co-locate with their consumers). All
  `pub(super)`; only `SdfRenderPlugin` (+ `SdfShaderDefs`/`SdfMaterialTable`) stay `pub`.
- *Verified:* **medium, worth-it-with-caveats.** (1) The clean per-pass boundary is undercut by
  shared state: `SdfGpuAtlas` threads all 7 units, `SdfPipeline.layout_0/1` is reused by
  cone+combine+gbuffer -- the `use super::{...}` plumbing IS the dominant cost, not a footnote.
  (2) **Fold in C5's `atlas_bind_group_1` helper as part of this change**, not after. (3) `build()`
  wires node/label types (`SdfBrickBakeLabel`, `SdfConeLabel`, ...) that move into submodules -- mod.rs
  re-imports them. (4) Confirmed safe: no test module to move; `main.rs:143`'s
  `sdf_render::render::SdfRenderPlugin` path is preserved by `render/mod.rs`; only `sync_sdf_shader_defs`
  is editor-gated and stays put.

### [x] M2. Split `bake_scheduler.rs` (2587 LoC) -- `window.rs` + `classify.rs` are the clean wins
`impact: medium` * `effort: large` * `source: modularization + refactor-deadcode`
> **Done (the two clean wins).** `bake_scheduler.rs` → `bake_scheduler/mod.rs` + `window.rs` (pure
> chunk-ring window geometry — `ring_chunk_origin`, entered/exited diffs, `chunks_in_aabb_windowed`,
> the BVH occupancy probe) + `classify.rs` (the Send read-only core — `Verdict`, `narrow_band_keep`,
> `classify_chunk`, `classify_candidates[_serial]`, `snapshot_hash_peek`). `mod.rs`: **2561 → 2173
> LoC**. The apply/dispatch split + A2 (`sync_emit`/`recenter_window` test-mirror extraction) are
> **left in `mod.rs` by design** — the verdict flagged them "murkier, optional": they reach
> `BakeScheduler`'s private fields (`pending`, `ring_chunk_origin`, `bvh`, `emit_scratch`, …), so a
> sibling module would just widen those to `pub(super)` for little cohesion gain. The 1400-line test
> module stays in `mod.rs` (in-file convention; it drives the whole lifecycle). Submodules reach
> shared types via `use super::*`; names re-imported so production + tests call them unqualified.

- *Now:* 54% of the file is `mod tests` (1194-2587). Production mashes five concerns: window-diff
  geometry (207-415), `schedule_bakes` (417-620), classify (640-863), apply (703-1107), and the
  sync/async dispatch state machine (1109-1192).
- *Approach:* Make `bake_scheduler/` a directory. `window.rs` = pure integer geometry (`ring_chunk_origin`,
  `chunk_in_window`, `chunk_window_keys`, `for_each_entered/exited_chunk`, `chunks_in_aabb_windowed`,
  `chunk_has_geometry_with`) + its own focused test mod. `classify.rs` = `Verdict`, `narrow_band_keep`,
  `classify_chunk`, `classify_candidates[_serial]` (already `pub(crate)`, snapshot-only/Send). `mod.rs`
  keeps the `BakeScheduler` resource, `schedule_bakes`, `dispatch_bake`, and the integration lifecycle
  tests.
- *Verified:* **medium, worth-it-with-caveats.** Do `window.rs` + `classify.rs` first -- truly
  zero-coupling, clean. The **apply/dispatch split is murkier**: `dispatch_bake` and the test harness
  reach `BakeScheduler`'s *private fields* (`pending`, `ring_chunk_origin`, `bvh`, `edits`,
  `emit_scratch`, `edit_gen`), so a sibling `dispatch.rs` forces those fields to `pub(super)` --
  trading file length for a looser encapsulation surface; treat as optional. **`emit_gpu_bakes:972`
  and `chunk_brick_keys:352` are `#[cfg(test)]`** -- they must travel with the tests, not into
  production submodules (the survey misassigned them). Verify zero clippy warnings after the
  visibility widening.

### [x] M3. Extract the editor camera + gizmo overlays out of `sdf_render/mod.rs` (1264 LoC)
`impact: medium` * `effort: medium` * `source: modularization`

- *Now:* `mod.rs` is the module front door (plugin, components, `SdfGridConfig`, re-exports) but also
  carries ~480 lines of editor camera control (orbit + free-fly + focus easing) and ~110 lines of
  immediate-mode gizmo overlay drawing -- burying the plugin wiring and the data-flow doc.
- *Approach:* `editor_camera.rs` ← `SdfOrbitCamera`, `SdfCameraMode`, `OrbitFocus`, `CameraInput`,
  `orbit_camera`, `fps_camera`, `ease_orbit_focus`, `spawn_editor_camera`, the sync systems.
  `overlays.rs` ← the gizmo groups, `configure_overlay_gizmos`, `draw_ground_grid`, `line_color`,
  `draw_lod_rings`, `LodRingsVisible`. `mod.rs` keeps the plugin, components/resources,
  `SdfGridConfig`, re-exports, and core bake/pick glue.
- *Risk:* Low-medium. Gizmo groups must keep their `register_type`/`init_gizmo_group` in `build()`
  (invariant #4). Picking (`sdf_picking`/`pick_sdf_volume`) is borderline -- could go to a
  `selection.rs` or stay. Combine with D4 (it touches `draw_lod_rings`).

### [x] M4. Move the tower/scatter stress-scene generator out of `edits.rs`
`impact: medium` * `effort: small` * `source: modularization`

- *Now:* `edits.rs:365-517` (`TowerRole`, `TowerEdit`, `TowerFieldParams`, `tower_field_edits`,
  `random_rotation`) is a scene-content generator calling `super::scatter` -- wedged between the CSG
  primitive eval and the AABB code in the core field module every bake/pick path imports.
- *Approach:* Move into a new `tower_field.rs` (or fold into the existing `stress.rs`, its only
  production consumer). Keep `tower_field_edits` `pub`. `hash2` is shared with `height_sample` -- keep
  it in `edits.rs` (or a tiny `noise.rs`). Single production caller + a test caller; only `use` paths
  change. Lowest-effort modularization win.

### [ ] M5. (optional) `scene_tabs.rs` (741 LoC) -- split model / io / swap
`impact: medium` * `effort: medium` * `source: editor-structure`

- *Now:* Mixes the pure document model (`SceneDoc`, `OpenScenes`, `CameraState`, dirty re-check), the
  save/load-to-disk orchestration (`drain_requests`, `save_active_to` + fs::write + notifications +
  thumbnail side effects), and the close-with-confirm egui flow -- three different change cadences.
- *Approach:* `scene_tabs/` dir: `model.rs` (the no-egui testable core + existing tests), `io.rs`
  (disk + request side), `mod.rs` thin for activate/handle_close/`confirm_close_dialog` + center-leaf
  splicing. *Risk:* low-medium; dirty/baseline invariants (`snapshot_active`, `load_doc_into_world`)
  must stay with the model. Pairs with E4 (the center-leaf splicing belongs with `dock`).

---

## Tier 3 -- API boundaries & encapsulation

Tighten visibility and relocate misplaced helpers. **Do A3 (the `pub` → `pub(crate)` sweep) LAST**,
after the structural moves, so it doesn't fight them -- and so it turns the encapsulation wins above
into compiler-enforced invariants.

### [~] A1. Relocate ring/window geometry + stateless `impl SdfAtlas` helpers into one geometry module
`impact: medium` * `effort: medium` * `source: api-boundaries`
> **Superseded by M2 (primary scope) + residual left optional.** The ring/window geometry the finding
> wanted consolidated now lives in `bake_scheduler/window.rs` (chunk-space) alongside the brick-space
> helpers in `atlas.rs` — a sensible split, so a *third* `clipmap_geometry` module would re-fragment
> rather than clarify. The only residual is renaming the 3 stateless `SdfAtlas::cull_edit_indices*` /
> `brick_palette_samples` static methods to free functions (so `classify` names the geometry, not the
> storage type). That's a cosmetic naming change on the hot cull path, and a clean de-indent of the
> ~100-line block out of `impl SdfAtlas` isn't worth the churn/risk there — left optional.

- *Now:* Brick-space ring geometry lives in `atlas.rs:364-451`; the parallel chunk-space geometry lives
  in `bake_scheduler.rs:271-415` -- two coordinate conventions with no shared home, free to drift.
  `cull_edit_indices*`, `brick_palette_samples`, `voxel_world_pos` are `impl SdfAtlas` static methods
  that use **zero** atlas state -- pure brick-geometry/BVH helpers parked on the storage struct,
  coupling the read-only classify path to the `SdfAtlas` type.
- *Approach:* Introduce a `clipmap_geometry` module (or fold into `chunk`) owning **all** ring/window/
  brick-lattice math, brick-space and chunk-space side by side. Move the stateless helpers off
  `impl SdfAtlas` into it as free functions -- classify then depends on geometry, not `SdfAtlas`.
  Confirm `ring_window_coords`/`ring_brick_keys` survival vs D1 (some are test-only). Mechanical moves;
  unit tests travel with the functions.

### [~] A2. Extract the production sync-emit + recenter loop so tests stop mirroring them
> **Deferred (verdict-optional, same call as the M2 apply/dispatch split).** Extracting
> `sync_emit`/`recenter_window` means pulling logic out of the `schedule_bakes`/`dispatch_bake`
> systems, which read `BakeScheduler`'s private fields (`pending`, `ring_chunk_origin`, `bvh`,
> `emit_scratch`, `edit_gen`) — the verdict flagged this murky and optional. The test mirrors it
> dedups (`emit_gpu_bakes`, `recenter_step`) are low drift-risk (the lifecycle differential catches
> divergence), so the modest dedup isn't worth refactoring the critical bake-scheduler system. Left
> with the apply/dispatch cluster in `mod.rs` (see M2).
`impact: medium` * `effort: medium` * `source: refactor-deadcode`

- *Now:* `#[cfg(test)] emit_gpu_bakes` (`bake_scheduler.rs:972`) re-implements the production sync path
  (drain → `sort_drained` → `gather_candidates` → `classify_candidates` → `apply_verdicts` → re-queue)
  from `dispatch_bake:1158-1173`. Test `recenter_step:1215` re-implements `schedule_bakes` step 2's
  per-LOD entered/exited loop (`556-592`). When production changes, the mirror must follow in lockstep
  or it stops exercising real code.
- *Approach:* Extract `fn sync_emit(...)` called by both `dispatch_bake`'s sync branch and the test
  settle loop; extract `fn recenter_window(...)` called by both `schedule_bakes` and `recenter_step`.
  ~60 lines of mirrored logic removed. *Risk:* the extracted fns take plain `&mut` refs (not ResMut);
  the lifecycle tests guard behavior. Natural follow-on to M2.

### [~] A3. Demote the `sdf_render` submodule tree + items from blanket `pub` to `pub(crate)`
> **Module-level sweep done (7 modules); item-level + dead-code cleanup deferred.** Only 6 submodules
> are reached externally (the `tests/` crate + the binary: `atlas`, `bake_scheduler`, `bvh`, `chunk`,
> `edits`, `render`) — those stay `pub mod`. Demoted the 7 cleanly-internal ones to `pub(crate) mod`
> (`bc7`, `editor_camera`, `height`, `overlays`, `picking`, `scatter`, `tower_field`), making their
> encapsulation a compiler invariant; cascade-gated `picking::debug_capture_march` (editor-only)
> behind `feature = "editor"`. The other 6 (`gallery`, `gizmo`, `node_gizmos`, `debug`, `stress`,
> `textures`) ALSO demote correctly but each surfaced genuinely-dead `pub` items (`spawn_gallery`,
> `spawn_stress`, the gizmo `TRANSLATE/ROTATE/SCALE` consts, the `textures` manifest structs, …) —
> exactly the dead API the sweep is meant to expose. Cleaning those is a focused delete-vs-keep pass
> (some may be intended-but-unwired API); deferred so it isn't rushed. Item-level `pub`→`pub(crate)`
> tightening within the kept-`pub` modules is the larger remaining piece.
`impact: medium` * `effort: small` * `source: api-boundaries`

- *Now:* `mod.rs:39-55` declares every submodule `pub mod`; many items are `pub` where `pub(crate)` is
  correct. Almost nothing here is a real public API -- it's internal to `adventure` + the in-crate
  `tests/` rig. Blanket `pub` means the encapsulation wins above (e.g. private `LiveChunkTables` fields)
  provide no guarantee and refactors can't lean on the compiler to prove a fn is unused.
- *Approach:* Demote submodule decls to `pub(crate) mod` (except where `main.rs`/siblings need them);
  tighten item visibility to `pub(crate)` except the handful the `tests/` integration crate imports
  (`chunk_gpu_key`, `dir_index`, `ring_chunk_origin`, the C3 helper, ...). The compiler reports every
  site that breaks. **Do this last.**

### [x] A4. Replace the duplicated `.as_ref().unwrap()` bind-group wall with one resolver
`impact: medium` * `effort: medium` * `source: cross-cutting`
> **Done (folded into M1 as C5).** The core finding — the ~11-unwrap atlas bind-group 1 wall
> copy-pasted in `SdfGBufferNode` + `SdfConeNode` — is eliminated: both nodes now call the single
> `render/mod.rs::atlas_bind_group_1(device, layout, gpu_atlas, label)` helper (one place, exact
> binding order preserved, GPU lifecycle rigs green). The remaining nuance (a `views() -> Result`
> with a descriptive message instead of `unwrap`) is minor polish on that one helper — the nodes
> already early-out before it when resources are missing — and is left as optional.

- *Now:* `render.rs:534-563` (bind_group_1 build) is duplicated near-verbatim at `render.rs:2156-2183`
  -- ~11 `gpu_atlas.dist_view.as_ref().unwrap()`-style unwraps each. A partially-initialized
  `SdfGpuAtlas` panics deep in render with no context, and the two copies can drift.
- *Approach:* Add `SdfGpuAtlas::views(&self) -> Result<AtlasViews<'_>, &'static str>` resolving all
  Options once with a single descriptive failure; both nodes construct from it. **This is the same
  duplication as C5's bind-group helper** -- unify them; do both inside M1. *Risk:* preserve the exact
  binding order (CPU↔GPU contract); run the GPU lifecycle tests.

---

## Tier 4 -- Editor structure

Improves the editor's discoverability and wiring consistency. Independent of the SDF-core tiers.
The whole `editor/` tree is feature-gated -- **build `--features editor` to catch breakage**.

### [x] E1. Break `EditorPlugin::build` (107-line grab-bag) into per-concern sub-plugins
`impact: medium` * `effort: medium` * `source: editor-structure`

- *Now:* `editor/mod.rs:42-147` adds 7 plugins, inits ~10 resources, registers 4 Reflect types, and
  hand-seeds two dispatch registries **inline** (thumbnails 78-85, asset inspectors 98-104) whose
  definitions live elsewhere. Three wiring idioms coexist (`XPlugin` structs, free `plugin(app)` fns,
  raw inline chains).
- *Approach:* Give each concern a `Plugin`: `ThumbnailRegistryPlugin` (next to the providers),
  `AssetInspectorPlugin`, `SelectionPlugin` (a `selection.rs` already exists with `EditorSelection`,
  no plugin), `DockPlugin` (the PostStartup init_dock_state/phosphor-font systems + the
  `EguiGlobalSettings` poke). `build` shrinks to an `add_plugins((...))` tuple + the import_settings
  Reflect registrations.
- *Verified:* **medium, worth-it-with-caveats.** Pure wiring move; panel order is already decoupled
  (registry consumed via `remove_resource` at PostStartup, `dock.rs:283-292`). **Must preserve:** the
  4 import_settings Reflect registrations (62-65), the `enable_absorb_bevy_input_system = false` poke
  (112-114, load-bearing comment), the phosphor font's `.after(EguiStartupSet::InitContexts)` (143),
  the `sync_selection` run_if gate (96), `register_component_editor::<Transform>` (72), and the
  renderdoc-gated plugin add (68-69). Build BOTH configs.

### [ ] E2. Generic `PathDispatchRegistry` for the 4 hand-rolled first-match-wins registries
`impact: medium` * `effort: medium` * `source: editor-structure`

- *Now:* `ThumbnailRegistry`, `AssetInspectorRegistry` (+ traits) are structurally identical -- a
  `Vec<Box<dyn T>>`, a `register`, and a `for p {... if p.matches(path) { return p.handle() }}` loop via
  the shared `with_registry` helper (`fs_util.rs:114`). `asset_inspector.rs:3`'s own doc says it
  "mirrors the ThumbnailProvider pattern."
- *Approach:* A `PathDispatchRegistry<T: PathMatcher>` in `fs_util.rs`/`registry_kit.rs` with `register`
  + `dispatch(world, path) -> Option<Out>`. `ThumbnailProvider`/`AssetInspector` become impls; their
  registry resources become thin newtypes. Leave `InspectorOverrides` (type-path keyed) and
  `DebugPanelRegistry` (id-keyed, multi-result) as-is -- different access patterns. *Risk:* medium --
  generic-over-trait-object ergonomics can get fiddly; skip if the generic ends up more complex than
  the two loops it replaces.

### [ ] E3. Extract a shared file-picker body from the Open/Save-As modals
`impact: medium` * `effort: medium` * `source: editor-structure`

- *Now:* `scene_browser.rs:74-145` (open) vs `:173-279` (save) are ~70% the same: window scaffolding,
  up-button + breadcrumb, `read_sorted` → `.scene` filter, dir/scene row loop, post-UI navigate/close
  epilogue. ~210 lines, ~70% duplicated.
- *Approach:* Extract `file_browser_body(ui, dir, files, &mut nav_to) -> Option<PickedFile>` +
  `scene_listing(dir)`. Each `*_dialog_ui` shrinks to window chrome + the shared body + its distinct
  footer. Optionally fold both behind one `enum FileDialog { Open, SaveAs }` resource (never open
  simultaneously). Good existing test coverage (`scene_browser.rs:281-332`); UI-only.

### [ ] E4. Concentrate `DockState<EditorTab>` topology surgery into `dock`
`impact: medium` * `effort: medium` * `source: editor-structure`

- *Now:* Three modules reach into dock-leaf internals: `scene_tabs.rs:305-328` exports center-leaf
  finders `pub(crate)` so `dock`/`layout` can use them; `layout.rs:63-86` re-derives the center leaf by
  scanning `is_center_tab`; both `scene_tabs` and `layout` do split/append leaf surgery on the same
  `DockState`.
- *Approach:* Move all topology helpers (`center_leaf`, `is_center_tab`, `set_scene_box_tabs`,
  `inject_live_scenes`, `side_anchor_leaf`, `add_panel_tab`, find/remove/activate) into a `dock::topology`
  submodule (or `EditorDockState` methods). `scene_tabs` keeps only document/swap/dirty logic; `layout`
  calls the same API; the `pub(crate)` center-leaf exports become private to `dock`. *Risk:* medium --
  subtle ordering invariants (e.g. `close_doc.rs:680-687` adds the placeholder BEFORE removing the
  scene tab so the leaf survives); move methods without changing call order. Lightly unit-tested area.

### [x] E5. Adopt one editor wiring convention (`Plugin` structs everywhere)
`impact: medium` * `effort: small` * `source: editor-structure`

- *Now:* Three coexisting conventions: `XPlugin` structs (config, panels, registry, ...), free
  `pub fn plugin(app)` (`keybinds.rs:13`, `status_bar.rs:34`), and inline-only (selection, chrome_trace
  toggle, the two registry seedings -- invisible from their own module).
- *Approach:* Every editor sub-module that needs wiring exposes a `Plugin` struct (matches the project's
  stated architecture). Convert `keybinds`/`status_bar` `plugin(app)` to structs; this ties into E1.
  Mechanical; preserve the `EguiGlobalSettings` poke + `register_component_editor::<Transform>`.

### [ ] E6. (low) `ui_temp` helper + a documented defer-apply pattern for panels
`impact: low` * `effort: small` * `source: editor-structure`

- *Now:* The egui-with-exclusive-World idiom (clone state out before the closure, collect mutations
  into locals, apply after) is hand-written at every panel with different variable names
  (`hierarchy/mod.rs:31-57`, `layout.rs:300/380`, `assets_browser/mod.rs:96-204`, `scene_browser.rs`).
- *Approach:* A `ui_temp<T: Clone + Default>(ui, id, |&mut T|)` wrapper for the `get_temp`/`insert_temp`
  dance (3x in `hierarchy/mod.rs` alone), and document the defer-apply pattern once. Don't over-abstract
  the defer-apply itself -- the explicit local vars stay readable. Only worth doing alongside other
  hierarchy work.

---

## Tier 5 -- Test infrastructure & cross-cutting

Mostly test-only scaffolding (zero runtime risk) plus a couple of project-wide consistency passes.

### [x] T1. Add `tests/common/` for the copy-pasted GPU device + naga_oil composer bring-up
`impact: medium` * `effort: medium` * `source: cross-cutting`

- *Now:* `device_queue()` is duplicated at `sdf_gpu_rig.rs:49`, `sdf_bake_gpu.rs:22` **and** an inline
  second copy at `sdf_bake_gpu.rs:266`, plus `gpu()` at `sdf_lifecycle_gpu.rs:41` -- four headless-wgpu
  bring-ups that have **drifted** (one logs adapter info + intersects features; another hard-skips;
  bake_gpu's first copy requests no features while its own second copy hard-requires 16BIT_NORM).
- *Approach:* `tests/common/mod.rs` (plain `mod common;`, no `#[path]` hack) with `headless_device() ->
  Option<(Device,Queue)>` doing the 16-bit-norm/BC feature negotiation once.
- *Verified:* **medium, worth-it-with-caveats.** The composer is **not** uniformly factorable --
  `SDF_MODULES` differs per file (see T2). Provide a generic `compose_entry(entry, &[module_paths])`,
  not one canonical set. Keep the feature-negotiating form so it returns `Some` on adapters lacking
  16-bit-norm and callers decide whether to skip; preserve bake_gpu's first test deliberately needing
  no features.

### [x] T2. Single source of truth for the `SDF_MODULES` shader dependency list
`impact: medium` * `effort: small` * `source: cross-cutting`

- *Now:* The import graph is encoded three times: `shader_validation.rs:31` (9 modules),
  `sdf_gpu_rig.rs:195` (2 modules), and the real pipeline in `render.rs`. A new `sdf/*.wgsl` module
  silently fails to validate until each list is hand-updated.
- *Approach:* Define the canonical ordered list once (a `const` in `src` usable by pipeline + tests, or
  in `tests/common`). Rigs needing a subset pass an explicit slice referencing the same source. Keep the
  intentional-subset cases explicit. Pairs with T1.

### [x] T3. Promote the reusable test-app builders out of `#[cfg(test)]`
`impact: medium` * `effort: medium` * `source: cross-cutting`

- *Now:* `tests/integration.rs:15 integration_app()` is byte-for-byte `src/test_utils.rs:9
  test_app_with_input()`, re-implemented because `test_utils` is `#[cfg(test)] pub mod` (invisible to
  the `tests/` crate). New shared spawn helpers are also unreachable.
- *Approach:* Gate `test_utils` behind a `test-support` feature exported normally (or a `pub mod testkit`
  compiled always), then both unit + integration tests share one definition; delete `integration_app()`.
  *Risk:* medium -- un-gating must avoid shipping in release (hence the `test-support` feature); touches
  `lib.rs` visibility.

### [x] T4. Standardize the inline-vs-external `mod tests` convention
`impact: medium` * `effort: medium` * `source: cross-cutting`

- *Now:* Two conventions with no documented trigger: external `mod tests;` files (`assets/tests.rs`,
  `soul_scene/tests.rs`) vs huge inline bodies (`bake_scheduler.rs` ~700 test lines, `chunk.rs` ~300,
  `edits.rs` ~247). The biggest test bodies stay inline, inflating already-large modules.
- *Approach:* Adopt the external-file convention once a test body crosses a threshold (e.g. >150 lines
  or >40% of the file): move `bake_scheduler`/`chunk`/`edits` test mods to sibling `_tests.rs` files.
  **Update `CLAUDE.md` File Conventions** (which currently mandates inline) so the move doesn't violate
  a stated invariant. Naturally folds into M1/M2. Keep shared rig fixtures with the moved tests.

### [x] X2. Register `GizmoRenderPlugin` explicitly in `main.rs`
`impact: medium` * `effort: small` * `source: cross-cutting`

- *Now:* `sdf_render/mod.rs:575-577` self-installs `GizmoRenderPlugin` as a hidden side effect behind a
  runtime `Assets<GizmoAsset>` probe + `is_plugin_added` guard. Every other plugin is added explicitly
  in `main.rs:138-150`; the central manifest doesn't reflect this one's lifecycle.
- *Approach:* Register it explicitly in `main.rs` next to the render plugins; drop the lazy add + the
  `is_plugin_added` guard. Keep the `MinimalPlugins`/`Assets<GizmoAsset>` inner guard only around the
  gizmo *group* init that genuinely needs `GizmoPlugin` in headless tests. Pairs with D7.

### [x] N1. (low) Rename `...Event` message types to `...Message` (Bevy 0.18)
`impact: low` * `effort: medium` * `source: cross-cutting`

- *Now:* `combat`/`inventory`/`camera` define `#[derive(Message)] DamageEvent`/`LootEvent`/
  `RightClickEvent` etc. -- using the new 0.18 APIs (`derive(Message)`, `add_message`, `MessageWriter`)
  but carrying the old `Event` suffix. Inconsistent with `ChatMessage`.
- *Approach:* Pure IDE rename to `...Message`; update the `MessageWriter<T>` sites + `tests/integration.rs`
  imports. Establish the convention in `CLAUDE.md`. Wide but mechanical -- keep to one PR.

---

## Explicitly NOT worth doing (verified)

Recorded so a future session doesn't misapply effort here.

### X1. Do **not** split the `SdfAtlas` god-resource
`verdict: not-worth-the-churn` * corrected to `low`

The survey flagged `SdfAtlas` (bricks + tiles + live_chunks + 3 generation counters + scratch, all
`pub`) for an encapsulation split. **Verification rejected it:** the generation invariant is *already*
behind methods (`insert_gpu_brick`/`remove_brick` co-locate both bumps; `bump_generation` is the only
public single-bump path), so the bug class the eviction test guards is already closed. The survey's
claims that `gpu_baked_tiles` is "read by render.rs" and the invariant is "convention scattered across
three files" are both false. The full split would touch ~80 direct `atlas.<field>` accesses (mostly
test boilerplate) to encapsulate something already encapsulated. **Only the dead-field carve-out
(`last_bake_was_full`) is worth it -- captured as D1.**

### X3. Do **not** split `chunk.rs`
`verdict: leave-whole` * `low`

At 984 lines `chunk.rs` is over the rough threshold, **but every symbol participates in one
invariant**: "the CPU-built directory + tile-run resolves each brick to exactly the tile the GPU shader
reads, byte-for-byte." `dir_index`/`chunk_gpu_key`/`KEY_BIAS` are mirrored in `bindings.wgsl` and
guarded by `wgsl_chunk_constants_match_rust`; `set_brick`/`clear_brick`/`dense_region` are validated by
the churn differential. Splitting would scatter a single tested contract across files and force the
test's GPU mirror to reach across module boundaries -- **lowering cohesion, not raising it.** This is
the file where splitting hurts. (See constraint #2.)

---

## Suggested sequencing

1. **Tier 0** anytime -- isolated, unblocks nothing, shrinks the surface for the splits.
2. **C1** (the verified `high`) -- biggest contract-dedup win; do before M1 touches the upload path.
3. **C4 → C2 → C3 → C5** -- the rest of the contract dedup, smallest-first.
4. **M4 → M3 → M1 → M2** -- modularization, smallest-first; fold C5/A4's bind-group helper into M1,
   A2 into M2, T4 into M1/M2.
5. **Editor (E1 → E5 → E2/E3/E4)** -- independent track, can run in parallel with the SDF tiers.
6. **A1 → A3 last** -- the visibility sweep turns every encapsulation win above into a
   compiler-enforced invariant; doing it last avoids fighting the moves.
7. **Test infra (T1/T2/T3/T4) + X2 + N1** -- low-risk cleanup, fit between larger items.
