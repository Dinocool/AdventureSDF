# Phase A — GPU Execution Layer: implementation-ready spec

Status: EXECUTION SPEC (no engine code changed by this doc). Worktree: `voxel-rt`. This is the
detailed, code-verified build guide for **Phase A of `docs/VOXEL_PROGRAM.md`** (A1–A4). Read
`VOXEL_PROGRAM.md` §"Phase A" first for the *why* and the sequencing rationale; read
`VOXEL_LARGE_SCENE_PLAN.md` (surface-only residency → A2) and `VOXEL_INSTANCING_PLAN.md` §2
(InstanceDescriptor + object-local DDA → A3) for the design these sub-stages execute. This doc is the
*how*: exact struct fields, buffer shapes, control-flow, the file:functions to change, the cross-world
plumbing, and the acceptance test + verification per sub-stage.

The hot-path code this touches:
- `src/voxel/incremental.rs` — the fixed-cap slot packer (`ResidentPacker`, `RepackDelta`/`ChangedSlot`,
  `SlotAllocator`, the voxel arena + quarantine). It already produces O(changed) deltas; today the render
  path throws them away and consumes the contiguous `snapshot_patch` instead.
- `src/voxel/raytrace.rs` — `VoxelRtPatch` (the main→render extracted resource), `stream_voxel_rt_residency`
  (the main-world re-pack), `prepare_voxel_rt` (the render-world buffer/BLAS/TLAS build + scene bind group),
  `VoxelRtResources`/`SceneKeepAlive`.
- `src/voxel/gpu.rs` — `GpuBrickMeta` (48 B), `GpuBrickAabb` (32 B), `GpuBrickPatch`, `BRICK_UNIFORM_FLAG`,
  `pack_resident_set`/`pack_one`/`finalize_patch_palette_and_lights`.
- `src/voxel/streaming.rs` — `desired_clipmap` (the cap) + `ResidencyManager::update` (the classify).
- `src/voxel/source.rs` — `BrickSource::classify` / `BrickClass`.
- `assets/shaders/voxel_raytrace.wgsl` — the scene bind group (group 0), `trace`/`trace_occluded` candidate
  loop, `dda_brick`, `brick_hit_at`, `cell_block`, `BrickMeta`.

> R2b note: a parallel agent (R2b) is mid-edit landing the per-brick palette + bit-packed index stream
> (`brick_palettes`, `index_bits`, `palette_base`, `voxel_indices` rename, the +group(12) binding). This
> spec is written **against the post-R2b architecture** (the code shown above already reflects it). If a
> field name here doesn't match the tree, reconcile to the R2b form — the *structure* is what matters.

## Dependency order (build in this order)

1. **A1** — wire the O(changed) GPU upload. Allocates the fixed-capacity buffers ONCE; everything downstream
   builds on those buffers. **BLOCKER, ~90% built.** No dependency.
2. **A2** — cap-after-classify. Small, independent of A1 but cheap to land alongside it; do it second so the
   surface-shell cap is correct before A3 grows the view. **Depends on nothing**, but is most useful once
   A1's fixed-cap pool exists (the cap == the pool capacity).
3. **A3** — per-chunk BLAS + multi-instance TLAS + `InstanceDescriptor` from day one. **The big rewrite.**
   **Depends on A1** (it reuses the fixed-cap slot buffers as the streamed world's "descriptor 0" arena and
   the per-generation `queue_write_buffer` upload).
4. **A4** — robustness (retire bit-31 flag, relative AABB epsilon, persist the interner). Independent
   cleanups; do last so they don't churn the A1/A3 surface area mid-flight. The bit-31 retirement (A4.1) is
   **easiest to land immediately after A1** because A1 already touches every meta read/write site.

Phase A gate (all four): the worldgen-perf rig shows per-move re-pack is O(changed) (no full rebuild); the
GPU oracle (`tests/voxel_raytrace_gpu.rs` + the seam/show-through/GI GPU tests) is pixel-identical; the
3-scene gallery streams smooth at 0.2 m.

---

## A1 — Wire the O(changed) GPU upload

### Goal

Stop re-creating every GPU buffer + the whole BLAS on every camera move. Allocate the meta/aabb/voxel/
palette/brick_palette buffers **once** at fixed capacity, and per generation `queue_write_buffer` **only the
slots `RepackDelta.changed` touched**. The BLAS is rebuilt per generation that changed topology (kept as a
single-instance build until A3 makes it per-chunk).

### Ground truth (what exists today)

- `incremental.rs::ResidentPacker::update(&entries) -> RepackDelta` already diffs the resident set and emits
  `RepackDelta { changed: Vec<ChangedSlot>, freed: Vec<u32>, topology_changed: bool }`. Each `ChangedSlot`
  carries `{ slot, meta, aabb, voxels: Option<Vec<u32>>, voxel_word_offset }`. A freed slot ships a
  `degenerate_aabb()` + `GpuBrickMeta::zeroed()`. `slot == primitive_index` for life.
- **CRITICAL — the arena stores RAW haloed cells, not R2b-encoded indices.** `emit_changed_slot` writes a
  `ChangedSlot.meta` whose `voxel_offset = block · dense_block_u32()` (the arena word offset) with
  `index_bits = 0`/`palette_base = 0` as the "raw-arena" marker (see `incremental.rs` line ~404-410). The
  R2b paletted encode happens only at `snapshot_patch` time (the contiguous path). **So `RepackDelta` as it
  stands is NOT directly shader-consumable** — the shader expects R2b-encoded `voxel_indices` + `palette_base`
  + `index_bits`, but the arena holds raw `u32`-per-cell. **This mismatch is the central A1 design decision
  (see "The encoding question" below).**
- `raytrace.rs::stream_voxel_rt_residency` already calls `packer.update(&entries)` then
  `packer.snapshot_patch(active_registry)` — it throws the delta away and ships the contiguous patch. The
  contiguous patch becomes `VoxelRtPatch.patch` + a bumped `generation`.
- `prepare_voxel_rt` re-creates `aabb_buf`/`meta_buf`/`voxel_buf`/`palette_buf`/`brick_palettes_buf` from
  scratch via `create_buffer_init`, builds a fresh single-instance BLAS/TLAS, and a fresh scene bind group,
  on every generation. This is the per-move hitch.

### The encoding question (resolve this first — it shapes A1)

The arena holds **raw cells**; the shader decodes **R2b paletted**. Two viable designs:

- **(A1-α) Ship the contiguous R2b patch via the delta-driven path, but allocate buffers once and write only
  the changed *range*.** Reject — the R2b encode re-bases every brick's `voxel_offset`/`palette_base` on
  every snapshot (the interner runs over the whole resident set), so the contiguous buffer is a different
  layout each generation. You cannot `queue_write_buffer` "only the changed slots" into a buffer whose
  every offset moved. This is why `snapshot_patch` is O(resident) memcpy, not O(changed).

- **(A1-β, CHOSEN) Make the GPU consume the RAW-ARENA dense form directly.** The fixed-cap arena IS already
  O(changed)-addressable: slot `s`'s meta lives at `meta_buf[s]`, its AABB at `aabb_buf[s]`, its dense block
  at `voxel_buf[block · dense_block_u32() .. +dense_block_u32()]`. Ship that. The shader's `cell_block` then
  reads a **raw `u32`-per-cell** dense brick (the pre-R2b decode: `voxel_indices[voxel_offset + cell_index]`)
  instead of the bit-packed paletted decode. **This trades R2b's voxel-VRAM win for the O(changed) upload on
  the streamed path.** Given Phase A's headline is "kill the per-move hitch" and the fixed-cap arena is
  already sized for raw blocks (`arena_capacity_u32 = max_resident_bricks · halo_cells(0)` ≈ 60k·1000·4 B ≈
  240 MB — the pre-R1 size), this is the honest, correct-by-construction choice for A1. R2b's compression is
  reclaimed later (B/storage) on the *disk*/transport side; the *resident* arena being raw is acceptable at
  0.2 m. **Document this tradeoff in the commit.**

  Concretely A1-β needs a shader path that, given a dense (`index_bits == 0` marker) meta, decodes
  `voxel_indices[voxel_offset + cell_index(x,y,z,hedge)]` directly as the block id (no palette indirection).
  Keep the R2b path for the *static* Cornell/Sponza scenes (which still go through `pack_brickmap`/
  `pack_resident_set` + `snapshot_patch`'s R2b encode) by branching on `index_bits == 0` ⇒ raw,
  `index_bits != 0` ⇒ paletted. Both are already in the meta; the marker is total.

> **Open question flagged for the implementer:** an alternative to A1-β is to push the R2b encode INTO the
> packer's per-slot shadow (persist the encoded indices + per-brick palette per slot, so the arena holds
> R2b bytes and the brick_palettes buffer is itself a fixed-cap slot-addressed arena). That is strictly A4.4
> ("persist the interner per slot so `snapshot_*` is truly O(changed)") — it makes the GPU arena R2b AND
> O(changed). If A4.4 lands *with* A1, prefer it (no raw-vs-paletted shader branch, full R2b VRAM win,
> O(changed) upload). The minimal-A1 path is A1-β (raw arena, shader branch); the maximal path folds A4.4 in.
> **Recommendation: land A1-β first (smallest correct delta, unblocks A3), then A4.4 to recover R2b VRAM.**

### Data-structure changes

**`incremental.rs` — add `snapshot_buffers()` and a scene-epoch reset.**

```rust
/// The fixed-capacity initial buffer contents the render path allocates ONCE (capacity-sized, zero/degenerate
/// for unused slots). Built from the packer's current shadow state. After this, the render path applies each
/// generation's `RepackDelta` via `queue_write_buffer` — never re-creating these buffers.
pub struct SnapshotBuffers {
    pub aabbs: Vec<GpuBrickAabb>,   // length == capacity; unused slots = degenerate_aabb()
    pub metas: Vec<GpuBrickMeta>,   // length == capacity; unused slots = GpuBrickMeta::zeroed()
    pub voxels: Vec<u32>,           // length == arena_capacity_u32(); raw dense blocks at block·stride
    pub palette: Vec<GpuPaletteColor>, // the registry palette (length == registry.len())
}

impl ResidentPacker {
    /// Build the FULL capacity-sized initial buffers (called ONCE per scene epoch, right after the packer is
    /// (re-)created and the first `update` ran). `aabbs`/`metas` are capacity-length with degenerate/zeroed
    /// unused slots; `voxels` is the arena_capacity_u32() raw block pool with each resident dense brick's
    /// block written at its arena offset. O(capacity) — paid ONCE per scene switch, not per move.
    pub fn snapshot_buffers(&self) -> SnapshotBuffers { /* fill from self.last_aabb/last_meta/last_voxels */ }
}
```

`SnapshotBuffers.metas`/`aabbs` are filled by iterating `0..capacity()`: slot `s` gets `last_meta[&s]`/
`last_aabb[&s]` if present, else `zeroed()`/`degenerate_aabb()`. `voxels` is `vec![0u32; arena_capacity_u32()]`
then, for each resident dense slot, its `last_voxels[&s]` raw block is copied to `arena_block · dense_block_u32()`.
(The packer already tracks `arena_block` per slot in `resident: FxHashMap<BrickKey, SlotState>`; expose the
slot→block map or thread it through `SnapshotBuffers`.)

**Important:** the **palette + NEE light list** are NOT per-slot — they are a function of the whole resident
set + the registry. `snapshot_buffers` can carry the palette (it's the registry, fixed per scene). The light
list is rebuilt per generation (see "NEE + palette plumbing" below) — it does not need O(changed).

**`raytrace.rs::VoxelRtPatch` — carry a `Snapshot | Delta` enum, not just the contiguous patch.**

```rust
/// What the main world ships to the render world this generation.
#[derive(Clone)]
pub enum VoxelRtUpload {
    /// A fresh scene epoch: allocate the fixed-cap buffers from these capacity-sized contents, then build the
    /// BLAS/TLAS over `brick_count` primitives. Ships on a scene switch / first pack.
    Snapshot { buffers: SnapshotBuffers, brick_count: u32, lights: Vec<GpuVoxelLight>, alias: Vec<GpuAliasEntry> },
    /// An incremental generation: queue_write only these changed slots into the already-allocated buffers.
    /// Rebuild the BLAS iff `topology_changed`. Carries the FULL light list for the generation (NEE is not
    /// per-slot — see plumbing).
    Delta { delta: RepackDelta, brick_count: u32, lights: Vec<GpuVoxelLight>, alias: Vec<GpuAliasEntry> },
}

#[derive(Resource, Clone, ExtractResource)]
pub struct VoxelRtPatch {
    pub upload: VoxelRtUpload,   // replaces `patch: GpuBrickPatch`
    pub generation: u64,
    /// The scene epoch id — incremented on every scene switch (fresh packer). The render world reallocates
    /// the fixed-cap buffers when this changes (a Snapshot also carries this implicitly).
    pub epoch: u64,
}
```

Keep `GpuBrickPatch`/`pack_brickmap` for the **static Cornell** fast-path (the `static_map_missing` arm and
the Cornell-with-edits arm still produce a contiguous `GpuBrickPatch`). Wrap that in a `Snapshot` whose
`SnapshotBuffers` is the contiguous patch padded to capacity — OR keep a tiny `Snapshot::Static(GpuBrickPatch)`
variant for the static scenes (they don't need a fixed-cap arena because they don't stream). The cleanest:
**`Snapshot` always carries capacity-sized buffers**; the static path builds a `ResidentPacker`-free
`SnapshotBuffers` directly from the `GpuBrickPatch` (pad metas/aabbs to its own brick_count, no spare slots —
static scenes never re-pack incrementally, so a generation bump always re-snapshots). This keeps `prepare_voxel_rt`
single-pathed (Snapshot ⇒ allocate, Delta ⇒ write).

### Control-flow

**`raytrace.rs::stream_voxel_rt_residency`** (the streamed arm, ~line 584-609):

```text
on scene switch:
    streaming.packer = Some(ResidentPacker::new(cfg.max_resident_bricks))   // already done, line 422
    streaming.epoch += 1                                                    // NEW: bump epoch
    (the first re-pack below will ship a Snapshot for this epoch)

each re-pack (the `if worldgen_dirty_pending && (settled || interval)` block):
    let delta = packer.update(&entries);                    // O(changed) — already happens
    let (lights, alias) = build_lights_for(&entries, active_registry);  // NEE — see plumbing
    if first re-pack of this epoch (a flag on streaming, reset on switch):
        patch_res.upload = Snapshot { buffers: packer.snapshot_buffers(registry-palette),
                                      brick_count: packer.resident_count() as u32, lights, alias };
        mark epoch-snapshotted
    else if !delta.is_empty():
        patch_res.upload = Delta { delta, brick_count: packer.resident_count() as u32, lights, alias };
    else:
        return;   // nothing changed — don't bump generation
    patch_res.generation += 1;
    patch_res.epoch = streaming.epoch;
```

`brick_count` for the BLAS = `packer.resident_count()` — but the fixed-cap BLAS is built over **`capacity`**
primitives (degenerate AABBs for free slots are skipped by the build). Decide: build the BLAS over `capacity`
primitives once (free slots degenerate, so they cost nothing in traversal — this is the whole point of the
degenerate-AABB design) and **never resize it**, OR rebuild over the live `resident_count` each topology
change. **CHOSEN: build over `capacity` once at the Snapshot, refit/rebuild in place on Delta.** A
degenerate AABB is a guaranteed non-candidate, so a capacity-sized BLAS with mostly-degenerate boxes is
correct and lets the BLAS be **built once per epoch** (or refit per topology change) instead of resized.
(See A3 for the per-chunk version; for A1 the single capacity-sized BLAS is the stepping stone.)

**`prepare_voxel_rt`** (~line 2287):

```text
if upload.epoch != resources.built_epoch  OR  upload is Snapshot:
    // (re)allocate the fixed-cap buffers from Snapshot.buffers (create_buffer_init, capacity-sized)
    aabb_buf  = create_buffer_init(cap·32 B, BLAS_INPUT|STORAGE|COPY_DST)   // COPY_DST is NEW (was missing)
    meta_buf  = create_buffer_init(cap·48 B, STORAGE|COPY_DST)
    voxel_buf = create_buffer_init(arena_u32·4 B, STORAGE|COPY_DST)
    palette_buf, brick_palettes_buf likewise
    build BLAS over `capacity` primitives (degenerate free slots) ; build single-instance TLAS
    build scene bind group
    resources.built_epoch = upload.epoch
else (Delta):
    for cs in delta.changed:
        queue.write_buffer(meta_buf,  cs.slot·48,  bytes_of(&cs.meta))
        queue.write_buffer(aabb_buf,  cs.slot·32,  bytes_of(&cs.aabb))
        if let Some(v) = cs.voxels:
            queue.write_buffer(voxel_buf, cs.voxel_word_offset·4, cast_slice(&v))
    if delta.topology_changed:
        rebuild (or refit) the BLAS over the SAME aabb_buf (capacity primitives) ; the bind group is unchanged
        (the TLAS references the same BLAS handle — only the BLAS geometry changed)
    rebuild the NEE light buffers from upload.lights/alias (WorldCacheLights::new equivalent)
```

**The AABB buffer needs `BufferUsages::COPY_DST`** (today it is `BLAS_INPUT | STORAGE` only — add `COPY_DST`)
so `queue_write_buffer` can patch a freed/changed slot's AABB. Same for meta/voxel/palette buffers (today
`STORAGE` only). This is a one-line-per-buffer change and is required for the delta path to write at all.

### Scene-epoch reset (re-zero the buffers)

A scene switch creates a **fresh** `ResidentPacker` (already done, line 422) and bumps `epoch`. The first
re-pack of the new epoch ships a `Snapshot`, which `prepare_voxel_rt` handles by **reallocating** the
buffers from the capacity-sized `SnapshotBuffers` (every unused slot is `zeroed()`/`degenerate_aabb()`).
So the buffers are re-zeroed by construction at epoch boundaries — no stale slot from the old scene survives,
because the buffers are new allocations. **Do NOT reuse the old buffers across an epoch** (a fresh packer's
slot 5 is a different brick than the old packer's slot 5; reallocating is the robust choice and only costs
the one-time create_buffer_init the old code already paid every move).

### NEE light list + palette plumbing (the only-a-delta-ships problem)

Today `prepare_voxel_rt` builds the light list from the **contiguous** `patch` via `WorldCacheLights::new(device,
patch)`, and `finalize_patch_palette_and_lights` derives palette+lights from the assembled buffers. When only
a `RepackDelta` ships, there is no contiguous patch to gather from. Two facts make this clean:

- The **palette** is the registry (fixed per scene). Ship it in `Snapshot.buffers.palette` (built from the
  registry, not from bricks). On a `Delta`, the palette buffer is unchanged — never re-upload it. (Emissive
  edits to the registry are a scene-config change, not a streamed delta; if they ever happen, treat them as
  an epoch bump.)
- The **light list** is a function of the whole resident set's air-exposed emissive voxels — it is genuinely
  not per-slot. **Build it CPU-side in the main world** from `manager.resident_entries()` each re-pack and
  ship it whole in `VoxelRtUpload::{Snapshot,Delta}.lights/alias`. There is already a fast path:
  `finalize_patch_palette_and_lights` **skips the O(resident) gather entirely when the registry has no
  emitters** (`registry.has_emitters()`), which is the common worldgen/Sponza case ⇒ the light list is empty
  and free. For emissive scenes (Cornell's ceiling panel, lava worldgen) the per-generation O(resident)
  light gather is acceptable (it already ran every re-pack in the old code) and is bounded by
  `MAX_VOXEL_LIGHTS`. Refactor: add a free function `build_lights_from_entries(&[ResidentBrick], &BlockRegistry)
  -> (Vec<GpuVoxelLight>, Vec<GpuAliasEntry>)` that runs `pack_one` only to gather faces (or reuses the
  packer's shadow `last_voxels` raw cells directly — they're already in memory). Call it in the main world,
  ship the result in the upload, drop the render-world `WorldCacheLights::new(device, patch)` in favour of a
  `WorldCacheLights::from_lists(device, &lights, &alias)`.

  > **Recommended:** gather lights from the packer's shadow (`last_voxels` raw cells per resident dense
  > slot + `last_meta` for uniform) so the light build reuses the already-resident bytes and never
  > re-`pack_one`s. This keeps the per-generation light cost ∝ resident emissive surface, which is what it
  > already was.

### Files/functions to change (A1)

- `incremental.rs`: add `SnapshotBuffers` + `ResidentPacker::snapshot_buffers()`; expose the slot→arena_block
  map (or build `voxels` inside `snapshot_buffers`); add `build_lights_from_entries`/shadow-based light
  gather (or put it in `gpu.rs`).
- `gpu.rs`: factor the light gather so it can run from `ResidentBrick` slices OR packer shadow without a
  contiguous patch (split `gather_lights_into` from the patch dependency, or add the entries-based wrapper).
- `raytrace.rs`: change `VoxelRtPatch` to `{ upload: VoxelRtUpload, generation, epoch }`; bump `epoch` on
  scene switch + a `epoch_snapshotted` flag; ship Snapshot/Delta in `stream_voxel_rt_residency`; rewrite
  `prepare_voxel_rt` to allocate-once + queue_write the delta; add `COPY_DST` to the buffer usages; replace
  the static-scene arms (Cornell/`static_map_missing`) to ship a `Snapshot`. Update `SceneKeepAlive` (the
  buffers now persist across generations — keep them in `VoxelRtResources` directly, not re-created in
  `_keep` each gen).
- `voxel_raytrace.wgsl`: add the **raw-arena dense decode** branch in `cell_block` (`index_bits == 0` ⇒
  `voxel_indices[voxel_offset + cell_index(x,y,z,hedge)]` as the block id directly; else the existing R2b
  paletted decode). This is the A1-β shader change.

### Acceptance test + verification (A1)

1. **GPU oracle pixel-identical.** `tests/voxel_raytrace_gpu.rs` (`gpu_ray_query_hit_matches_cpu_ground_truth`,
   `gpu_mixed_lod_matches_cpu_ground_truth`) + `voxel_seam_gpu`, `voxel_show_through`, `voxel_gi_gpu`,
   `voxel_temporal_gpu` must stay green. Add a test that drives `stream_voxel_rt_residency` over a short
   camera sequence and asserts the **delta-uploaded buffer state byte-equals** the buffer state a
   from-scratch `snapshot_buffers()` would produce at the same generation (the GPU-side analogue of the
   existing CPU `incremental::tests::snapshot_patch_matches_full_pack` / `incremental_matches_full_pack_over_camera_sequence`).
2. **Per-move re-pack is O(changed).** Extend `tests/voxel_worldgen_perf.rs::bench_incremental_repack_vs_full`
   (already compares `packer.update` vs full `pack_resident_set`) to also time the **GPU upload** path:
   assert the bytes written per delta-generation ∝ `delta.changed.len() · (48 + 32 + dense_block·4)` and that
   a steady-state move writes ≪ the capacity-sized buffer. Print bytes/gen; the gate is "no full-buffer
   re-upload on a per-brick move."
3. **Manual:** hand the user `cargo run` with the gallery; the per-move hitch (58–103 ms) should be gone and
   streaming smooth at 0.2 m. (Per the no-auto-run rule, the agent verifies via tests; the user confirms the
   runtime feel.)

---

## A2 — Cap after classify

### Goal

Make `max_resident_bricks` bound the **surface SHELL** (Θ(H²)), not the **clip VOLUME** (Θ(H³)). Today the
cap is applied to the full clipmap volume *before* the surface `classify` prunes the buried interior + sky —
so at a large `clip_half` the cap throws away near-surface bricks to make room for far interior bricks that
`classify` would have pruned anyway. The surface-only residency win (`VOXEL_LARGE_SCENE_PLAN`) is undone at
the larger `clip_half` the flip (Phase D) forces.

### Ground truth

- `streaming.rs::desired_clipmap` (line 254) enumerates the full `(2·clip_half+1)³`-ish clip volume and, at
  line 274-296, **caps it to `max_resident_bricks` by world distance** — this is the clip VOLUME cap, applied
  before any classify.
- `ResidencyManager::update` (line 417) calls `desired_clipmap`, drops bricks that left, then **enqueues only
  `classify(coord,lod) == Surface` keys** (line 444-462), pruning `Air`/`Interior` into the `pruned` memo.
  So the SURFACE filter runs *after* the cap. The cap and the classify are on opposite sides of the volume→
  shell reduction.

### The change

Move the cap to **after** classify so it bounds the surface set:

- **Option A2-i (minimal, CHOSEN):** make `desired_clipmap` **not cap** (return the full uncapped tiling),
  and cap inside `ResidencyManager::update` **after** the classify split. Concretely: in `update`, enumerate
  `desired` (uncapped), classify each not-yet-resident key, collect the `Surface` keys, and if
  `resident.len() + queued.len() + surface_candidates.len() > max_resident_bricks`, drop the **farthest
  surface candidates** (same `world_d` ranking `desired_clipmap` used). The `Air`/`Interior` keys never count
  against the cap (they're pruned, not resident). This makes the cap bound exactly the surface shell.

  - Keep a separate, MUCH larger safety ceiling on the raw clip volume enumeration (e.g. a hard
    `MAX_CLIP_ENUMERATION` guard) so a pathological `clip_half` can't OOM the enumeration itself — but that
    ceiling is not `max_resident_bricks`; it's a defensive enumeration bound.

- **Option A2-ii:** push `classify` *into* `desired_clipmap` (pass the `source`), so the cap sees only
  surface keys. Rejected — `desired_clipmap` is a pure geometric function used by `clipmap_uncapped_len` and
  the perf rig; coupling it to a `BrickSource` muddies the SSOT. A2-i keeps `desired_clipmap` pure and puts
  the surface-aware cap where the classify already lives.

### Residency-test updates

- `desired_clipmap`'s cap test (any unit test asserting `out.len() <= max_resident_bricks` for a tight cfg)
  moves to a `ResidencyManager::update` test: after a cold fill at a small `max_resident_bricks`, assert
  `resident_count() + pending() <= max_resident_bricks` and that the **kept** bricks are the nearest surface
  ones (a far surface brick is dropped before a near one; an interior brick never occupies a slot).
- `clipmap_uncapped_len` stays the SSOT for "the full tiling size" (the cap-drop log). The `capped_total`
  accounting moves to count surface-candidates dropped by the new cap.
- Add a regression: at a large `clip_half` (e.g. 26) over the worldgen surface, assert the resident count
  scales ~Θ(H²) not Θ(H³) (this is the `VOXEL_LARGE_SCENE_PLAN` §7 measurement — fit the exponent), proving
  the cap now bounds the shell.

### Files/functions to change (A2)

- `streaming.rs`: remove the cap block from `desired_clipmap` (or gate it behind a much larger enumeration
  guard); add the surface-aware cap to `ResidencyManager::update` after the classify split; update
  `capped_total`.
- `streaming.rs` tests + any `voxel_streaming.rs` integration test asserting the old volume cap.

### Acceptance test + verification (A2)

- GPU oracle pixel-identical (the cap only changes *which* far bricks are dropped, never the near surface a
  ray sees — at the shipping `clip_half=8` the cap rarely binds, so the gallery render is unchanged).
- The new `update` cap test: surface-bounded resident count; nearest-kept ordering.
- The Θ(H²) exponent regression at large `clip_half`.

---

## A3 — Per-chunk BLAS + multi-instance TLAS + InstanceDescriptor (from day one)

### Goal

Replace the single global BLAS + single identity TLAS instance with **N per-chunk BLASes** under a
**multi-instance TLAS**, each instance carrying a **3×4 transform** + an **`instance_custom_index →
GpuInstanceDescriptor`** indirection, **built from day one** so the streamed world is just "descriptor 0,
identity transform, base offsets 0" — the degenerate case — and multi-instance FEATURE work (`.vox`-as-
instances, off-axis props, per-instance destruction) later needs **no second AS rewrite**. This is
`VOXEL_INSTANCING_PLAN` §2.2/§2.4 ("widened Phase 3A") executed.

### Ground truth

- `prepare_voxel_rt` builds **one** BLAS (`primitive_count = brick_count`), **one** TLAS
  (`max_instances: 1`), one identity `TlasInstance::new(&blas, IDENTITY_3x4, custom_index=0, mask=0xff)`.
- The shader's `trace`/`trace_occluded` loop reads `metas[c.primitive_index]` directly (one global meta
  array), `dda_brick` marches in **world** space using `m.world_min`. There is no transform, no descriptor.
- `metas`/`voxel_indices`/`palette`/`brick_palettes` are global single buffers (group 0, bindings 1/2/3/12).

### Data-structure changes

**The descriptor (from `VOXEL_INSTANCING_PLAN` §2.2), 80 B, `bytemuck`-uploadable:**

```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuInstanceDescriptor {
    object_from_world: [f32; 12],     // 3x4 world→object (transforms the ray INTO object space)
    world_from_object_rot: [f32; 12], // 3x4 object→world (rotates the hit normal BACK to world)
    meta_base: u32,    // offset into the global `metas`/`aabbs` for this object's first brick
    voxel_base: u32,   // offset into the global `voxels` arena
    palette_base: u32, // offset into the global `brick_palettes` (per-object palette slice; 0 for the world)
    inv_scale: f32,    // local-t → world-t factor (1.0 for rigid/identity)
    edit_base: u32,    // per-instance edit overlay base, or SENTINEL (unused in A3; reserved)
    mask: u32,
    _pad: [u32; 2],
}
```

> **Descriptor 0 MUST be bit-identical-in-effect to today.** `object_from_world = world_from_object_rot =
> IDENTITY_3x4`, `meta_base = voxel_base = palette_base = 0`, `inv_scale = 1.0`, `edit_base = SENTINEL`,
> `mask = 0xff`. With descriptor 0 the new hit path reduces to `metas[c.primitive_index]` marched in world
> space — **exactly the A1 code**. This is the single most important correctness property of A3.

**Chunk scheme.** Chunk = **KxKxK bricks** (start with K=8 → an 8³-brick chunk = a 1.6 m³ region at 0.2 m,
~512 brick slots). The streamed world's `capacity` slots are partitioned into chunks by brick coord; each
chunk owns a contiguous slot range and one BLAS over its (≤K³) brick AABBs. One TLAS instance per chunk, with
**translation-only** transform (the chunk's world origin) OR identity + per-brick `world_min` in the meta
(simpler — the brick AABBs are already in world space, so a chunk's instance can be **identity** and the
descriptor's `meta_base` selects the chunk's slot range). **CHOSEN for the streamed world: identity-transform
chunk instances, descriptor `meta_base` = chunk's slot base, AABBs stay world-space** — this keeps the world
render bit-identical (no ray transform for the world) while still exercising the multi-instance TLAS + per-
chunk BLAS + descriptor indirection. A future `.vox` prop is the *non-trivial* transform case the same path
serves.

**Global concatenated buffers.** `metas`/`aabbs`/`voxels`/`brick_palettes` become global concatenations
addressed by `descriptor.meta_base`/`voxel_base`/`palette_base`. For A3 the streamed world is descriptor 0
with all bases 0, so the existing fixed-cap A1 buffers ARE the global buffers (one object). A prop appended
later gets a descriptor with non-zero bases pointing past the world's slot range. The shader resolves
`metas[descriptor.meta_base + primitive_index]`.

### Shader hit-path rewrite (the highest-risk step)

In `trace`/`trace_occluded`, on an AABB candidate:

```text
let inst = rayQueryGetCandidate...instance_custom_index   // descriptor index
let d    = descriptors[inst]
// transform the world ray into object space (identity for the world ⇒ ro_l==ro, rd_l==rd):
let ro_l = apply_3x4(d.object_from_world, ro)             // affine
let rd_l = apply_3x4_rot(d.object_from_world, rd)         // rotation only
let prim = d.meta_base + c.primitive_index                // global brick index
let m    = metas[prim]
// slab + dda_brick run in OBJECT space using m.world_min (object-local for a prop; world for descriptor 0)
let bh   = dda_brick_indexed(prim, d.voxel_base, d.palette_base, ro_l, rd_l, t_enter, t_exit)
// commit WORLD t: local_t · inv_scale  (== local_t for the world / rigid props)
rayQueryGenerateIntersection(&rq, local_t * d.inv_scale)
```

`dda_brick`/`cell_block`/`brick_hit_at` gain a `voxel_base`/`palette_base` param so they address
`voxel_indices[voxel_base + m.voxel_offset + ...]` and `brick_palettes[palette_base + m.palette_base + ...]`
(or fold the base into a single `prim`/offset add). For descriptor 0 the bases are 0 ⇒ identical addressing.
The hit **normal** is rotated back to world via `world_from_object_rot` (identity for the world ⇒ unchanged).

Add the descriptors buffer to group 0 (new binding, e.g. `@binding(13)`). The TLAS instance's
`instance_custom_index` carries the descriptor index (chunk index for the world; per-prop index for props).

### CPU side (`prepare_voxel_rt`)

- Build a `Vec<GpuInstanceDescriptor>` — for A3-world: one descriptor PER CHUNK, all identity, `meta_base =
  chunk_slot_base`, other bases 0. Upload as a storage buffer (group 0, new binding).
- Build **one BLAS per chunk** over that chunk's slot-range AABBs (a slice of the capacity-sized aabb_buf via
  `primitive_offset` + `primitive_count`). `wgpu::BlasAabbGeometry` takes a `primitive_offset` — use it so
  each chunk BLAS reads its slice of the single aabb_buf (no per-chunk buffer).
- TLAS: `max_instances = chunk_count` (allocate generously — `capacity / K³` rounded up, plus headroom for
  future props). One `TlasInstance::new(&chunk_blas[i], IDENTITY_3x4, custom_index = i, 0xff)` per non-empty
  chunk.
- **Dirty-chunk rebuild:** track which chunks a `RepackDelta` touched (a `ChangedSlot.slot` maps to its chunk
  by `slot / K³`). On a `Delta` with `topology_changed`, rebuild **only the dirty chunks' BLASes**; refit (or
  skip) stable chunks. Rebuild the TLAS over all chunk instances each topology change (TLAS build is cheap;
  per-instance incremental TLAS update is a later optimization, flagged in `VOXEL_INSTANCING_PLAN` §7).
  **Refit-on-stable-topology:** a `Delta` whose changed slots are pure meta/voxel edits (no enter/drop in a
  chunk, `!topology_changed` for that chunk) needs no BLAS rebuild for that chunk at all (the AABBs didn't
  move) — only the `queue_write_buffer` from A1.

### Highest-risk steps + mitigation

1. **The shader hit-handler rewrite** is load-bearing for primary, shadow, AO, and GI rays (`trace`,
   `trace_occluded`, `brick_hit_at` all re-walk the committed brick). *Mitigation:* make descriptor 0
   **bit-identical-in-effect** to today (identity transform, zero bases) and **diff Cornell/Sponza frames
   pixel-for-pixel** before any prop exists. Land the descriptor indirection with a SINGLE descriptor 0 first
   (world = one instance, one chunk = whole world) — prove pixel-identical — THEN split into K³ chunks, prove
   pixel-identical again. Two small steps, each oracle-gated.
2. **`primitive_offset` semantics on the wgpu-trunk fork** — confirm a chunk BLAS over a slice of the shared
   aabb_buf via `primitive_offset` reports `primitive_index` **relative to the BLAS** (so `meta_base` must add
   the chunk's slot base) vs **absolute**. *Mitigation:* a 2-chunk headless test asserting `meta_base +
   primitive_index` lands on the right brick; pin the convention before the world split.
3. **TLAS `max_instances` + per-frame rebuild cost** with many chunks. *Mitigation:* the perf rig times the
   per-chunk BLAS rebuild over dirty chunks only; assert it stays ≪ the old monolithic rebuild. Start K large
   (fewer, bigger chunks) and tune down.

### Acceptance test + verification (A3)

- **Extend `voxel_raytrace_gpu` to a 2-instance scene:** world (descriptor 0, identity) + one **translated
  cube object** (descriptor 1, translation transform, non-zero `meta_base`/`voxel_base`). Assert GPU hit ==
  CPU ground truth for rays hitting each. This proves the descriptor indirection + object-local DDA + world-t
  commit across instances. (`VOXEL_INSTANCING_PLAN` §7 Phase 1 acceptance.)
- **Cornell/Sponza pixel-identical** (the GPU oracle) with the world rendered as descriptor 0 — first as a
  single chunk, then split into K³ chunks. This is the load-bearing "zero visual change" gate.
- Perf rig: per-chunk dirty rebuild ≪ monolithic rebuild; gallery streams smooth.

---

## A4 — Robustness

Three independent cleanups. Land **A4.1 right after A1** (A1 touches every meta read/write). A4.2/A4.3/A4.4
are independent and can land in any order after A1.

### A4.1 — Retire the bit-31 uniform-flag invariant

**Problem.** `BRICK_UNIFORM_FLAG = 1<<31` steals bit 31 of `voxel_offset` to mark a uniform brick (gpu.rs
line 118). It is release-unchecked (`debug_assert!` only, line 170/808-809) and pinned to "real offsets are
`< 2^31` (≤ ~60k bricks × halo_cells ≈ 60 M u32s)". When the flip (Phase D) or a larger budget raises the
arena past 2^31 u32s, `voxel_offset` collides with the flag ⇒ **silent corruption**.

**Fix.** Move the uniform flag to a **free `GpuBrickMeta` field**, freeing the full u32 for `voxel_offset`/
`palette_base`. `GpuBrickMeta` (gpu.rs line 130) is 48 B with a **3-u32 pad** (`_pad: [u32; 3]`, line 155).
**Use `_pad[0]` as a `flags` field.** Define `const META_FLAG_UNIFORM: u32 = 1` in the `flags` word.

- `GpuBrickMeta::uniform`: set `flags |= META_FLAG_UNIFORM`; store the block id in `voxel_offset` (full u32,
  no flag bit). `GpuBrickMeta::dense`: `flags = 0`; `voxel_offset` is the full-range offset.
- `is_uniform()` reads `flags & META_FLAG_UNIFORM`; `uniform_block()` reads `voxel_offset & 0xFFFF` (or the
  full low bits — block ids are u16); `dense_offset()` returns `voxel_offset` unmasked.
- Drop the `debug_assert!(voxel_offset & BRICK_UNIFORM_FLAG == 0, ...)` guards (line 170, 808-809) — the
  offset is now unconstrained.
- **WGSL mirror** (`voxel_raytrace.wgsl` line 50-54 `_pad0/_pad1/_pad2`, line 62 `BRICK_UNIFORM_FLAG`, line
  76 `meta_is_uniform`): rename `_pad0` → `flags`; `meta_is_uniform` reads `(m.flags & 1u) != 0u`;
  `meta_uniform_block` reads `m.voxel_offset & 0xFFFFu`; `cell_block`'s dense branch uses `m.voxel_offset`
  unmasked. Delete `BRICK_UNIFORM_FLAG`.
- `incremental.rs` `degenerate`/`zeroed` paths: `zeroed()` already sets pad to 0 ⇒ `flags = 0` ⇒ not-uniform,
  correct for a freed slot.
- **Test:** `meta_uniform_flag_roundtrips_without_growing` (gpu.rs line 1627) must still pass (struct stays
  48 B, 4-align) — update it to assert the flag lives in `flags`, and add a case with `voxel_offset >= 2^31`
  that round-trips a dense offset (the corruption regression).

### A4.2 — `BRICK_AABB_EPSILON` relative-per-LOD

**Problem.** `BRICK_AABB_EPSILON = VOXEL_SIZE * 1.0e-3` (gpu.rs line 80) is an **absolute** world distance.
It is the seam-overlap fudge — `1e-3` of a *0.2 m voxel*. At 0.05 m (Phase D) `VOXEL_SIZE` quarters, so the
epsilon shrinks 4× — and a coarse-LOD brick (span `2^lod×` larger) overlaps by a *relatively* tinier fraction,
risking the seam miss the epsilon exists to prevent.

**Fix.** Make the epsilon **relative to the brick's per-LOD span** (or per-LOD cell size). Replace the
constant with `fn brick_aabb_epsilon(lod: u32) -> f32 { brick_span(lod) * REL_EPS }` where
`REL_EPS = 1.0e-4` (tune so the absolute grow at LOD0/0.2 m matches today's `VOXEL_SIZE*1e-3`). `brick_aabb`
(line 88) takes `lod` already — use `brick_aabb_epsilon(lod)`. **The WGSL `BRICK_AABB_EPSILON` (used in
`trace`/`trace_occluded`/`brick_hit_at` slab tests, line 319/355/421) MUST mirror this** — make it a
`fn brick_aabb_epsilon(lod)` in WGSL too and pass `meta_lod(m)`. This is a shared SSOT change: the slab grow
in the shader must equal the AABB grow on the CPU or the seam fix breaks. **Test:** `voxel_seam_gpu` /
`voxel_seam_oblique_gpu` stay green at LOD0 AND add a coarse-LOD (lod≥2) oblique-grazing case (the per-LOD
relative epsilon's reason for being).

### A4.3 — (covered by A4.1; the bit-31 retirement IS the storage robustness fix)

### A4.4 — Persist the interner + encoded form per slot (truly O(changed) snapshot)

**Problem.** `ResidentPacker::snapshot_patch` (incremental.rs line 444) re-runs `VoxelInterner` over **every**
resident slot's `last_voxels` raw cells (`intern_paletted`, line 472) on every snapshot — an O(resident)
re-encode, defeating the "O(changed)" goal for the contiguous path. (A1-β sidesteps this for the *streamed
GPU* path by shipping raw deltas; A4.4 makes the encoded form itself O(changed), enabling the R2b-VRAM-
preserving A1 variant flagged above.)

**Fix.** Persist per slot the **encoded form** (its `DenseLayout` = `{voxel_offset, palette_base, index_bits}`
into a slot-addressed R2b arena) instead of re-encoding from raw cells each snapshot:

- Give the packer a **second fixed-cap arena for the R2b index stream** + a **per-object/global brick-palette
  arena**, slot-addressed like the raw arena. `emit_changed_slot` (line 370) encodes the brick **once** when
  its cells change (R2b `encode_paletted`) and writes the encoded indices + per-brick palette into the slot's
  R2b arena region, caching the `DenseLayout`. A slot whose raw cells didn't change keeps its cached encoding
  (no re-encode).
- `snapshot_buffers` (A1) then ships the **R2b arena** directly (not raw), recovering R2b's VRAM win while
  staying O(changed) on the GPU upload — this is the "maximal A1" path. The shader keeps the R2b paletted
  decode (no raw branch needed).
- **Interner persistence (R3 dedup across snapshots):** keep the `VoxelInterner` (`seen: FxHashMap<Box<[u32]>,
  DenseLayout>`) as a **packer field** (not a per-snapshot local), so an identical brick that re-enters reuses
  its existing slice. Eviction: when a slot's encoding changes/drops, decrement a refcount on its interned
  slice; free the R2b arena region when the refcount hits 0 (quarantine like the raw arena). This is more
  bookkeeping than A1 needs — **only do A4.4 if you want the R2b VRAM win on the streamed path**; otherwise
  A1-β (raw arena, shader branch) is the simpler shippable.
- **Test:** extend `incremental::tests::snapshot_patch_matches_full_pack` to assert the persisted-encoding
  snapshot byte-equals the from-scratch `pack_resident_set` R2b layout (mapping by `key → bytes`, not slot
  order), AND that a no-op `update` (no cells changed) re-encodes **zero** bricks (the O(changed) gate —
  instrument the encode count).

### Files/functions to change (A4)

- `gpu.rs`: `GpuBrickMeta` `_pad[0]`→`flags`; `uniform`/`dense`/`is_uniform`/`uniform_block`/`dense_offset`;
  remove `BRICK_UNIFORM_FLAG` + its `debug_assert`s; `brick_aabb` → `brick_aabb_epsilon(lod)`; the meta
  round-trip test.
- `incremental.rs`: (A4.4) second R2b arena + persisted interner + refcounted slice eviction; `snapshot_buffers`
  ships R2b.
- `voxel_raytrace.wgsl`: `flags` field + `meta_is_uniform`/`cell_block` updates (A4.1); `brick_aabb_epsilon(lod)`
  fn + per-LOD slab grow (A4.2).

### Acceptance (A4)

- All GPU-oracle + seam tests green (A4.1/A4.2 are layout/SSOT changes — pixel-identical by construction).
- The `>= 2^31` dense-offset round-trip (A4.1 corruption regression).
- The coarse-LOD oblique seam case (A4.2).
- The zero-re-encode no-op `update` (A4.4 O(changed) gate).

---

## Cross-cutting: the SSOT discipline to preserve

- **`pack_one` is the per-brick byte SSOT.** Both `pack_resident_set` and `ResidentPacker` build every brick
  through it; A1/A4 must keep that (the incremental-vs-full A/B test, `incremental::tests`, is the gate).
- **The shader mirrors the CPU layout exactly.** Every meta-layout change (A4.1 flags, A4.2 epsilon) is a
  *paired* CPU+WGSL edit; `tests/shader_validation.rs` + the GPU oracle catch drift.
- **Descriptor 0 = the degenerate world.** A3's correctness rests entirely on descriptor 0 being a no-op;
  build it that way and diff Cornell/Sponza before any non-trivial instance exists.
- **Adapt, never reset.** Edits/streaming bump generations and patch slots locally; never full-clear the
  reservoirs / DLSS history / world cache (the standing GI rule).

## Build/verify per sub-stage (mirrors CI)

`cargo build` AND `cargo build --features editor` (zero warnings) + `cargo test` + the GPU tests under
`tests/` (TMP/TEMP redirect per the test-temp-dir memory for GPU tests). Per the standing QA mandate, each
sub-stage: specialist implements → ≥2 independent adversarial reviewers vs the GPU ground truth → benchmark
gate (the perf rig). Do NOT `cargo run` to verify — verify via tests; hand the user the run for runtime feel.
