# Brick Loading And Transition Audit

Date: 2026-06-18

Scope: current streamed `.vxo` brick loading and the GPU residency front end, primarily:

- `src/voxel/residency_pager.rs`
- `src/voxel/residency_gpu.rs`
- `src/voxel/residency_front_end.rs`
- `src/voxel/raytrace.rs`
- `assets/shaders/voxel_residency.wgsl`
- `assets/shaders/voxel_pack.wgsl`

## Executive Summary

The most likely sources of corrupted or stale bricks during transitions are not in a single "load brick" function. They come from the interaction between the pager, the persistent GPU residency diff, the pack tail, and the one-frame-late BLAS rebuild.

The highest-risk issues are:

1. Core availability changes are not a dirty signal. If a brick packs while its center or halo core is absent, it can remain packed as air or with wrong halo data after the pager later supplies the core, because the GPU dirty set is driven only by enter/drop events.
2. Current-frame drops zero metadata and write degenerate AABBs before the corresponding chunk BLAS is rebuilt. The stale BLAS can still trace the old primitive for the current frame while the shader reads the newly-zeroed or changed slot data.
3. The pager's "+1 brick" coverage invariant is documented but not enforced in `desired_regions`. Missing halo regions silently become `NEIGHBOUR_ABSENT`, which feeds issue 1.
4. Several shader append lists and command buffers rely on `LIST_CAP` but do not check bounds before writing. If the actual desired/candidate/dirty/pack/aabb counts exceed the fixed capacity, the shader can write out of bounds.
5. Core-store capacity and fetch misses are silent. The visual fallback is often "pack as air", but there is no counter, admission guard, or forced repack when the missing core appears later.

## Pipeline Notes

The streamed path is:

1. `drive_gpu_residency_front_end` updates the `StreamedResidencyPager` before recording the GPU front-end frame (`src/voxel/raytrace.rs:4058-4076`).
2. The pager computes desired resident regions, decodes newly-covered regions, rebuilds occupancy, derives surface-plus-halo core keys, and syncs the paged core store (`src/voxel/residency_pager.rs:207-267`).
3. The GPU front end enumerates desired/candidate bricks from occupancy, diffs against the persistent slot table, then packs entered/dirty bricks into the live scene pool (`src/voxel/residency_front_end.rs:823-875`).
4. The CPU reads back the previous frame's dirty chunk mask before recording the current frame, then rebuilds those chunk BLASes after the current compute submit (`src/voxel/raytrace.rs:4155-4263`).

That ordering is important: pool writes are current-frame, but BLAS rebuild selection is previous-frame.

## Likely Transition Corruption Causes

### 1. Core availability changes do not force a repack

Severity: High

`pack_build_dirty` only builds dirty work from entered keys, dropped keys, and their same-LOD neighbors (`assets/shaders/voxel_residency.wgsl:1368-1453`). A brick whose slot is already resident is not repacked just because its core or one of its halo cores appears, disappears, or changes in the paged core store.

This is dangerous because missing cores are explicitly converted into air during packing:

- `core_lookup` returns `NEIGHBOUR_ABSENT` when a key is not in the paged core table (`assets/shaders/voxel_residency.wgsl:1183-1205`).
- The front-end neighbor build writes that result directly into `neighbour_indices` (`assets/shaders/voxel_residency.wgsl:1458-1487`).
- `fill_halo` treats an absent center core as air to avoid an out-of-bounds core read (`assets/shaders/voxel_pack.wgsl:220-260`).

That avoids GPU memory garbage, but it creates a persistence bug:

- Frame N: a surface brick enters while its center or halo core is absent, so it packs as air, with exposed faces, or with a degenerate AABB.
- Frame N+1 or later: the pager finally supplies that core, or frees enough core-store capacity for it.
- The slot is already resident, so no enter event occurs. If no neighbor enter/drop happens, `dirty_count` stays zero and the brick is not repacked.

The result can look like corrupted, missing, or wrong-normal bricks left behind after a transition.

Recommended fix:

- Add a pager-to-front-end dirty signal for changed core keys. Any resident brick whose center core changes, plus resident bricks whose 26-neighbor halo includes that key, must be marked dirty.
- At minimum, maintain a CPU-side set of newly inserted/evicted core keys in `StreamedResidencyPager::update`, upload it as a transient dirty-core list, and feed it into Pass D dirty expansion.
- Add debug counters for "resident slot packed with absent center core" and "resident dirty candidate has missing center core". These should be zero outside intentionally unloaded far detail.

### 2. Current-frame drops can be traced by stale BLAS

Severity: High

The current frame records the GPU front end first, then rebuilds BLASes for chunks reported dirty by the previous frame:

- Previous dirty chunks are polled before `record_frame` (`src/voxel/raytrace.rs:4155-4162`).
- The front end writes current-frame metadata/AABBs in submit 1 (`src/voxel/raytrace.rs:4166-4180`).
- Submit 2 rebuilds only the accumulated dirty chunks, which came from previous readbacks (`src/voxel/raytrace.rs:4183-4263`).
- The dirty mask for current writes is copied to staging at the end of `record_frame` and read next frame (`src/voxel/residency_front_end.rs:870-873`).

For drops, Pass D0 immediately zeroes metadata and emits a degenerate AABB command (`assets/shaders/voxel_residency.wgsl:1333-1365`). `write_aabb_dirty` marks the affected chunk dirty, but that dirty bit is only visible to the CPU next frame (`assets/shaders/voxel_pack.wgsl:403-418`).

So for the current render frame:

- The scene pool may contain zeroed metadata and degenerate AABB data for a dropped slot.
- The TLAS/BLAS can still contain the old primitive for that slot's chunk if that chunk was not already in the previous dirty set.
- A ray can hit the stale BLAS primitive and then read the new zeroed or repurposed slot data.

This can produce a black/corrupt transition brick for at least one frame. Under continuous camera motion there can be fresh current-frame drops every frame, so the artifact can look persistent.

Recommended fix:

- Do not destructively zero dropped slot metadata before the chunk BLAS containing that slot is rebuilt, or keep old metadata/AABB live until the BLAS rebuild catches up.
- Alternatively, rebuild a conservative current-frame superset when drops occur. A crude fallback is "any drop this frame -> rebuild all chunks" until a better GPU-to-CPU dirty path exists.
- Extend slot/drop quarantine from "one frame" to "until dirty chunk rebuild acknowledged" if the goal is to prevent both stale-BLAS/zero-meta hits and stale-BLAS/reused-meta hits.

### 3. Pager region coverage does not locally enforce the documented +1 brick pad

Severity: High

`desired_regions` says it collects present regions "PADDED +1 brick", but it passes the clipped `lo..hi` box directly to `present_world_regions_in` (`src/voxel/residency_pager.rs:174-200`). There is no local `lo - 1` / `hi + 1` expansion in the pager.

The rest of the system relies on that invariant:

- The paged core store documentation says the prefetcher pages clipmap regions padded by one brick, so every enterable brick and 26-neighbor halo has a resident core (`src/voxel/residency_gpu.rs:653-667`).
- `collect_surface_halo_keys` inserts each surface brick plus occupied 26-neighbor halo keys (`src/voxel/residency_pager.rs:285-313`).
- The lazy core fetch then looks up the owning region from the already-decoded resident region map and silently returns `None` when that region was not decoded (`src/voxel/residency_pager.rs:243-252`).

If `present_world_regions_in` does not itself pad by one brick in every edge case, a surface brick near a clip/region boundary can enter while its halo core is absent. That feeds issue 1.

Recommended fix:

- Make padding explicit in `desired_regions` or rename/document `present_world_regions_in` as the sole padding authority and add tests for boundary bricks.
- Add a debug assertion or counter after `collect_surface_halo_keys`: every desired core key must either fetch successfully or be intentionally all-air/outside-scene.
- Add a movement test that positions the camera so a surface brick sits exactly on a region boundary.

### 4. Silent core-store misses and capacity drops can leave resident bricks with missing cores

Severity: High

The paged core store intentionally drops inserts when the core buffer is full:

- Core capacity is clamped in the pager (`src/voxel/residency_pager.rs:97-106`).
- `insert_brick` returns `None` when `free_cores` is empty (`src/voxel/residency_gpu.rs:807-837`).
- `sync_to_keys` ignores both `fetch == None` and failed insert results (`src/voxel/residency_gpu.rs:918-936`).

The front end still makes residency decisions from occupancy, not from "core is available". Therefore a slot can become resident even though its center core cannot be packed correctly.

Recommended fix:

- Track and expose `core_fetch_miss_count` and `core_capacity_miss_count` per pager update.
- Refuse to enter a candidate whose center core is absent, or enter it as a known pending state that must be repacked when the core appears.
- Consider making the core store admission priority match resident-slot admission priority, so near visible bricks do not lose cores to farther halo keys.

### 5. Shader append lists are not bounds-checked against `LIST_CAP`

Severity: High

The front end allocates fixed-capacity list and command buffers based on `LIST_CAP = 1_000_000` (`src/voxel/residency_front_end.rs:143-158`, `src/voxel/residency_front_end.rs:343-365`).

Several shader paths append without checking the destination array length:

- `prepare_shell_dispatch` writes `shell_wg_indices[slot]` (`assets/shaders/voxel_residency.wgsl:356-414`).
- `enumerate_shells` writes `desired_list[d_slot]` and `candidate_list[slot]` (`assets/shaders/voxel_residency.wgsl:434-472`).
- `diff_enter_scan` writes `enter_list[e]` (`assets/shaders/voxel_residency.wgsl:889-937`).
- `diff_drop_apply` writes `drop_list[d]` (`assets/shaders/voxel_residency.wgsl:991-1012`).
- `try_mark_dirty` writes `dirty_list[d]` and `dirty_slot[d]` (`assets/shaders/voxel_residency.wgsl:1382-1408`).
- `pack_build_drops` and `pack_build_commands` append AABB/pack commands (`assets/shaders/voxel_residency.wgsl:1340-1365`, `assets/shaders/voxel_residency.wgsl:1502-1585`).

`would_overflow` only checks `params.total_cells > LIST_CAP` and the B0 workgroup count (`src/voxel/residency_front_end.rs:704-712`). It does not know the actual `desired_count`, `candidate_count`, `dirty_count`, `pack_count`, or `aabb_count`. A dense scene can have valid shell cell count but still overflow one of the append lists.

Recommended fix:

- Add a `list_cap` uniform and guard every append. Increment an overflow counter when clamped.
- If any overflow counter is non-zero, skip destructive drops and fall back to CPU or a conservative coarser admission policy for that frame.
- Add a stress test that forces `desired_count > LIST_CAP` with a reduced test cap.

### 6. Slab allocator high-water has no shader-side capacity guard

Severity: Medium/High

The GPU slab allocators bump shared high-water counters and return offsets without checking the actual pool capacity:

- `alloc_index_slab` bumps `index_slab_ctrl[0]` (`assets/shaders/voxel_residency.wgsl:1240-1260`).
- `alloc_palette_slab` bumps `palette_slab_ctrl[0]` (`assets/shaders/voxel_residency.wgsl:1262-1281`).
- Dense pack commands use those offsets directly (`assets/shaders/voxel_residency.wgsl:1552-1585`).

There are diagnostics for high-water readback in Rust, but no GPU-side guard before the write. If the arena is exhausted, dense pack can overlap or write outside the intended pool.

Recommended fix:

- Pass index/palette pool word capacities into `PackConfig`.
- If an allocation would exceed capacity, emit an overflow flag and avoid writing dense data for that slot. Prefer leaving the old brick live or writing a safe degenerate state over corrupting shared arenas.
- Add an automated assertion around the existing high-water diagnostics.

### 7. `region_replacement_resident` stack overflow is treated as success

Severity: Medium

The WGSL replacement-coverage check uses a fixed 512-entry stack (`assets/shaders/voxel_residency.wgsl:670-723`). When the stack is full, extra children are skipped. The function can then return `true` without checking all descendants.

The comment says the pruned descent should stay under the cap. If a dense transition or teleport-scale refinement violates that assumption, a coarse brick can be dropped before all fine replacements are resident.

Recommended fix:

- Add an overflow flag and treat overflow as "not safe to drop".
- Add a parity/stress test where refinement depth and desired coverage approach the stack limit.

## Performance Issues

### 1. Default residency settings reserve very large VRAM up front

The default `StreamingConfig` uses `clip_half_bricks = 160` and `max_resident_bricks = 900_000` (`src/voxel/streaming.rs:108-125`). Comments note that the index arena and paged core store alone reserve about 3.7 GiB.

The front end also allocates large fixed transient buffers for lists, commands, neighbors, staging, slab state, and hash tables (`src/voxel/residency_front_end.rs:312-381`).

This is acceptable for the target dGPU only if the residency path never exceeds those fixed capacities. It is risky for lower-memory GPUs and makes scene switches/rebinds expensive.

Recommended work:

- Add a startup/runtime VRAM budget summary for this path.
- Gate the default settings by adapter limits or expose a lower-memory profile.
- Move from fixed global `LIST_CAP` to per-scene sizing plus guarded fallback.

### 2. Region crossings rebuild occupancy and surface+halo keys from scratch

Every region crossing does a full occupancy rebuild and full surface+halo key derivation:

- `StreamedResidencyPager::update` calls `rebuild_occupancy` and `collect_surface_halo_keys` on every desired-region change (`src/voxel/residency_pager.rs:234-253`).
- `rebuild_occupancy` allocates and fills a new `Vec` of occupied bricks (`src/voxel/residency_pager.rs:270-282`).
- `collect_surface_halo_keys` scans decoded resident regions and builds hash sets (`src/voxel/residency_pager.rs:285-313`).

This is simpler and probably correct when it fits, but it can hitch on boundary crossings in dense scenes.

Recommended work:

- Reuse scratch buffers and hash sets across updates.
- Track per-region occupancy/core deltas once correctness counters exist.
- Keep the full rebuild path as a debug/reference mode.

### 3. The "non-blocking" readbacks block the device

`poll_change_count` and `poll_dirty_chunks` both call `device.poll(wgpu::PollType::wait_indefinitely())` (`src/voxel/residency_front_end.rs:885-954`). That can stall the CPU on the GPU timeline, especially during cold fill or heavy transition frames.

Recommended work:

- Maintain persistent map state per staging slot and use a true try-poll path.
- Do not call `wait_indefinitely` in the hot frame path.
- Keep blocking readback only for explicit debug dumps.

### 4. Dirty BLAS rebuild drains all pending chunks in one frame

The current code intentionally drains the full pending dirty chunk set every frame, batching only per `build_acceleration_structures` call (`src/voxel/raytrace.rs:4183-4263`). This avoids stale backlog, but it can produce large transition spikes.

Recommended work:

- After fixing current-frame drop safety, consider a bounded rebuild budget with a no-reuse/no-zero policy for chunks still awaiting rebuild.
- Track chunk rebuild count and time in telemetry so transitions can be profiled.

### 5. Per-frame and per-rebind zeroing allocates large temporary vectors

`record_frame` allocates a fresh zero vector for the dirty chunk mask every frame (`src/voxel/residency_front_end.rs:831-833`). `reset_state` also allocates large vectors for slot tables, hashes, slab state, and free lists on every rebind (`src/voxel/residency_front_end.rs:719-743`).

Recommended work:

- Keep reusable zero buffers or use GPU clear/fill paths where available.
- Reuse CPU scratch vectors inside `GpuResidencyFrontEnd`.

## Diagnostic And Documentation Issues

### 1. `live meta + degen AABB` is not always a hole

The debug dump labels live metadata plus degenerate AABB as "resident but no BLAS prim - a HOLE; should be ~0" (`src/voxel/raytrace.rs:3639-3642`).

However, the current shader intentionally keeps buried/uniform bricks resident while writing a degenerate AABB so they are not traced (`assets/shaders/voxel_residency.wgsl:1513-1528`). That means the diagnostic can report expected hidden residents as holes.

Recommended fix:

- Split this metric into "buried resident with degenerate AABB" and "unexpected visible candidate with degenerate AABB".
- Use the classify result or a recomputed `has_air`/surface test to distinguish them.

### 2. `NEIGHBOUR_SOLID` comments are inconsistent with the live front end

`voxel_pack.wgsl` still documents `NEIGHBOUR_SOLID` as the occupied-but-no-core fallback (`assets/shaders/voxel_pack.wgsl:79-86`, `assets/shaders/voxel_pack.wgsl:241-255`).

The live GPU front end now writes `core_lookup` results directly and documents missing cores as air (`assets/shaders/voxel_residency.wgsl:1472-1482`). `core_lookup` does not appear to emit `NEIGHBOUR_SOLID`.

Recommended fix:

- Either remove the dead fallback from the live path or reintroduce it deliberately with tests.
- Make the comments in both shaders describe the same policy.

### 3. Several high-level comments describe older behavior

Examples:

- `drive_gpu_residency_front_end` says the pipeline records into one encoder and rebuilds BLAS in the same submit, but the implementation now uses split submits (`src/voxel/raytrace.rs:3961-3974`, implementation at `src/voxel/raytrace.rs:4166-4263`).
- `pending_blas_chunks` says a bounded window is rebuilt, but the implementation drains the whole set each frame (`src/voxel/raytrace.rs:1606-1615`, `src/voxel/raytrace.rs:4214-4263`).
- `empty_snapshot_for_cold_fill` says `brick_count = 0`, but the code preserves or forces a non-zero count so the pool is allocated (`src/voxel/raytrace.rs:3331-3350`).

These are not direct corruption causes, but they make the transition path hard to reason about and can cause future fixes to target the wrong model.

## Existing Mitigations Worth Keeping

- Slot-table and core-table deletion now uses tombstones, preserving open-addressing probe chains (`assets/shaders/voxel_residency.wgsl:26-28`, `src/voxel/residency_gpu.rs:646-651`).
- `write_aabb_dirty` gates on the actual AABB command count, avoiding stale command tail writes (`assets/shaders/voxel_pack.wgsl:403-418`).
- Dirty chunk BLAS rebuild recreates BLAS handles instead of relying on refit, avoiding known degenerate-to-real refit corruption (`src/voxel/raytrace.rs:4187-4191`).
- The CPU-packed snapshot is stripped before GPU cold fill, avoiding CPU-seeded bricks that the GPU front end would never repack (`src/voxel/raytrace.rs:3331-3350`).

## Suggested Fix Order

1. Add core-miss/change telemetry first. Count missing center cores, missing halo cores, core fetch misses, and core capacity misses per frame.
2. Make core availability a dirty source. Repack resident bricks affected by newly inserted/evicted core keys.
3. Fix drop/BLAS timing. Do not allow a current-frame dropped slot to be traced by stale BLAS while its metadata has already been zeroed.
4. Enforce the pager +1 halo-region invariant with tests at region and clip boundaries.
5. Add shader-side append bounds and overflow counters for all `LIST_CAP`-backed buffers.
6. Add slab allocator capacity guards before dense writes.
7. Clean up diagnostics and stale comments so future debugging reflects the actual pipeline.

## Useful Repro/Debug Signals

- If F10 full BLAS rebuild clears the artifact, the issue is likely stale BLAS or dirty chunk timing.
- If F9 reports source-content mismatches or dense bricks with empty cores, suspect pager/core-store misses or the missing core-dirty signal.
- If artifacts appear at region boundaries, inspect whether the pager decoded the +1 halo region and whether the center/halo core was present when the brick packed.
- If artifacts only occur in very dense/wide scenes, check list-cap, core-cap, and slab high-water overflow before chasing shader math.
