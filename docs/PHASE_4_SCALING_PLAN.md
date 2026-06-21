# Phase 4 — Ray-Guided Demand Residency + LRU (the unbounded-scenes answer)

**Goal.** See far detail along sight lines at a *bounded* VRAM budget, with no black holes, and
destruction-safe — so a scene far larger than the resident pool renders correctly. This is the foundation the
Phase 3 pool-collapse later rides on (a smaller pool is safe once the demand/refill + graceful-degradation net
exists). Plan of record; supersedes the terse Phase 4 stub in `DYNAMIC_LARGE_SCENE_PLAN.md`.

## Research findings that shape the design (2026-06-21, two read-only mapping passes)

1. **The clipmap is EXACT-TILING — one resident LOD per world location.** `level_resident` =
   `level_box(L) \ level_hole(L)`; the finer level punches a hole in the coarser shell, so coarse and fine
   AABBs never overlap (`src/voxel/streaming.rs:233`, `assets/shaders/voxel_residency.wgsl:314`). ⇒ there is **no
   coarse brick resident behind a fine one** to fall back to today.
2. **A non-resident brick is invisible to the trace.** Free slots carry a **degenerate AABB**
   (`incremental.rs:566`) that the ray query rejects, so the BLAS-walk trace never sees "I wanted a brick here
   but it's absent" — there is no natural miss-observation point (`voxel_raytrace.wgsl` `trace`). ⇒ ray-guided
   requests must come from a brick the ray **does** hit (a resident coarse proxy), not from a miss.
3. **The trace scene bindings are READ-ONLY** (`voxel_raytrace.wgsl:196`: metas/voxels/palette are `read`).
   ⇒ LRU (`last_used_frame`) and the request buffer need NEW writable group-0 bindings.
4. **keep-old-until-revealed already exists + is robust.** `safe_to_drop` drops a brick only once its
   replacement (coarse ancestor OR fine descendants) is resident (`streaming.rs:632`, GPU port
   `voxel_residency.wgsl:748`). Budget eviction (4a/4b) is **nearest-first**, so it drops far/out-of-view
   bricks, not mid-view ones. ⇒ the no-black-hole machinery for transitions is in place; Phase 4 extends it.
5. **The readback-free 1-frame-late mirror pattern** (`change_count` / `dirty_chunk` staging rings,
   `residency_front_end.rs:872`) is the template the ray-request buffer follows — never a blocking GPU→CPU read.

## Architecture (SOTA: GigaVoxels demand + Nanite screen-error, adapted to HW-RT + our resident occupancy)

Keep the **flat HW-RT invariant**: the BLAS holds **one LOD per location** (the DDA never walks a tree). But
*which* LOD is **demand-chosen**, not a fixed clipmap ring:

- **Always-resident COARSE BACKDROP.** The coarsest LODs are kept resident over a LARGE box (cheap — few
  bricks). Every location is covered by at least the coarse backdrop ⇒ a ray ALWAYS hits something ⇒ no black
  holes, and you see far (coarse) instead of a hard clip boundary.
- **Demand-filled FINE OVERRIDE.** A ray that hits a coarse brick which screen-error says should be finer emits
  a **refinement request** (the child `(coord,lod)` — available at the hit). The requested fine brick pages in
  next frame; when it enters, its coarse ancestor's AABB at that location is **degenerated** (fine overrides —
  still one LOD per location). On LRU eviction of the fine brick, the coarse ancestor's AABB is **restored** (no
  hole). This is the coarse↔fine swap done by demand, not by a distance ring.
- **LRU residency.** The trace writes `last_used_frame` to each brick it hits; eviction prefers
  least-recently-hit fine bricks. Distance remains the *prefetch seed* (the near clipmap); LRU governs the
  demand-filled fringe. The coarse backdrop is pinned (never LRU-evicted).
- **Readback-free throughout.** Requests are appended GPU-side (atomic counter) and consumed via the 1-frame
  -late staging-ring mirror; the pager pages the requested regions/cores on demand.

Destruction: an edit re-sources affected bricks (existing path; GI adapts, never resets). A dig that promotes
many cores stays safe because (a) the coarse backdrop is always behind everything (no black hole if a fine core
is briefly missing), and (b) graceful halo degradation (below) derives the halo solid/air from the resident
occupancy mask so a missing neighbour core can never make a wrong-normal black cube.

## Phased sub-steps (each independently shippable + gated; order by dependency)

- **4-S0 — Graceful halo degradation (the safety floor; do FIRST).** In the pack halo-fill, when a neighbour
  is occupied per the occupancy mask but its core is absent, fill the halo SOLID from occupancy (correct
  exposed-face normal = no black cube) + best-effort material. Robust-by-construction under *any* pool pressure;
  the prerequisite that makes every later step (and the Phase 3 collapse) edit-safe. **Gate:** `pack_parity`
  green; a harness that drops a neighbour core and asserts no wrong-normal/black cube.
- **4-S1 — Coarse backdrop + demand-LOD override (the foundation).** Pin the coarse LODs resident over a large
  box; allow a fine brick's entry to degenerate its coarse ancestor's AABB (and restore on drop). Replace the
  exact-tiling `level_resident` with backdrop-coarse + demand-fine. **Gate:** far view shows coarse (not a hard
  clip edge); near shows fine; no black holes under motion; VRAM bounded (coarse backdrop is cheap).
- **4-S2 — Ray-guided refinement requests.** Add the writable request buffer (group-0) + `last_used_frame`;
  the trace, on hitting a coarse brick the screen-error wants finer, appends the child request + stamps
  last_used. A new residency pass consumes the 1-frame-late requests into the enter path (feeding the existing
  histogram/cut). **Gate:** rays refine along sight lines within budget; request count / frame bounded; no
  thrash on a static camera (deterministic convergence).
- **4-S3 — LRU eviction.** `diff_drop_mark` evicts least-recently-hit fine bricks (last_used) when the budget
  is full, complementing the distance cut; coarse backdrop pinned. **Gate:** a pathological all-surface fly
  (forest/city) never exceeds the VRAM budget; evictions/frame reported; no black holes under continuous motion.
- **4-S4 — Demand paging beyond the clipmap.** The pager pages requested regions/cores OUTSIDE the clip box
  (the cores for ray-requested far-detail), readback-free. **Gate:** a scene far larger than the pool renders
  far detail along sight lines at a fixed VRAM budget; pool occupancy / page rate reported. Peak-VRAM gate (8GB).

Phase 5 (screen-error `want_lod` = projected voxel footprint) then *feeds* 4-S2's request LOD — the natural
follow-on once demand is live.

## Then: Phase 3 pool-collapse rides on this

With 4-S0 (graceful degradation) + the coarse backdrop + demand/LRU in place, the uniform-core collapse
(`residency_gpu.rs` core store: uniform id inline in the table, dense cores fixed-512, `cores_buf` sized to
dense + headroom) is edit-safe — a big dig that overflows headroom degrades gracefully (drop far cores; coarse
backdrop + occupancy-halo prevent black cubes) instead of corrupting. See [[voxel-storage-plan]],
[[voxel-large-scene-plan]].

## Invariants held throughout
- Flat HW-RT: one LOD per BLAS location; DDA + `voxel_offset` decode unchanged (no tree on the hot path).
- Readback-free: GPU-driven; all CPU mirrors 1-frame-late + non-blocking.
- GI adapts locally, never resets ([[feedback-gi-adapt-not-reset]]).
- Edit-robust: fixed-per-slot pools; demand/LRU manages WHICH bricks fill the flat pool, never variable per-slot
  sizing.
- 8 GB VRAM gate on every step that touches pool sizing ([[feedback-vram-budget-8gb]]).
