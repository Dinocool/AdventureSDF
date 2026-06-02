# Chunk lookup → per-LOD toroidal directory (migration plan)

## Why

The GPU chunk lookup is a **sparse sorted array** (`chunk.rs` `LiveChunkTables`: `sorted_keys` +
`key_to_row`; GPU `chunk_buf` binary-searched in `brick.wgsl::find_chunk`). Every chunk insert/remove
is **O(resident)** — a `Vec` splice + a `key_to_row` re-stamp — so a coarse LOD-ring snap that evicts a
shell of thousands of chunks costs O(shell × resident). Measured: the recenter spikes to **448 ms**
while flying. The GPU lookup is also an O(log N) binary search done many times per ray.

A clipmap is a *bounded* per-LOD window (`R = ring_bricks/CHUNK_BRICKS = 32` chunks/axis), so the right
structure is a **dense per-LOD toroidal directory**: chunk coord `c` lives at the fixed slot
`c mod R` (component-wise `rem_euclid`). Lookup = one indexed read + a tag compare (O(1), no search);
insert/update = one slot write (O(1)); the delta upload writes fixed-position slots (no tail shift).

### Validated by the profiling rig (`tests/chunk_lookup_bench.rs`)

Run: `CARGO_INCREMENTAL=0 cargo test --test chunk_lookup_bench --release -- --ignored --nocapture`

| structure | max mutate/frame | lookup | memory |
|---|---|---|---|
| SortedArray (today) | **2729 ms** | 99 ns/op | 30 MB |
| Toroidal **Hybrid** | **8.9 ms (~300×)** | 33 ns/op (3×) | 24 MB |
| Toroidal inline-dense | 3.2 ms | 37 ns/op | 196 MB ← why tile-runs stay sparse |

`structures_agree_on_lookups` proves identical resolution. The **hybrid** is the pick: a dense 20 B
directory for *residency* + the existing `ChunkSlotAllocator` free-list for the *sparse* tile-runs.

## Decisions (locked by the adversarial tests)

Three `adversarial_*` tests in the rig pin the correctness model:

1. **Migrate in EXPLICIT-CLEAR mode**, not pure free-eviction. Keep the recenter loop calling
   `clear_brick` on each exited chunk — but `clear_brick` is now **O(1)** (a slot write), so the spike
   is gone *and* it stays correct. The adversarial leak test showed the audit's "free-on-overwrite"
   idea **still leaks** the tile-run buffer (it grows with fly distance, because departed chunks whose
   slot is never reused are never reclaimed); **explicit O(1) clear-on-exit is bounded**. Explicit
   clear also bumps `generation` and frees the tile-run immediately, so it **moots blockers 2 and 3**.
2. **The one hard rule — atomic apply-time publish (blocker 1).** Publish a directory slot (the whole
   20 B record: tag + occ + tile_run_base) **only at bake-APPLY** (`insert_gpu_brick`), never at
   chunk-enter, and as one delta write. A tag-valid slot pointing at unbaked/departed texels makes an
   entering chunk resolve to a *departed chunk's geometry* — a wrong-geometry class the sorted array
   can't produce. This is just the existing `insert_gpu_brick` discipline.
3. **`c mod R` via `euclid_mod` on BOTH sides; don't assume R is pow2** (`ring_bricks=96 → R=24`). Pass
   `R` + per-LOD window origin `O_lod` as derived uniforms. Use `bindings.wgsl::euclid_mod`, never raw `%`.

## What stays unchanged

`brick_in_chunk` (occ popcount → `chunk_tile_buf[tile_run_base + popcount(below)]`), the atlas, the
tile-run sparse allocation, `abs_chunk_key`/`chunk_gpu_key`, the march's fine→coarse fallback. Only the
*chunk → row* step changes (binary-search → toroidal index + tag), and the delta-upload machinery.

## Critical files

- `src/sdf_render/chunk.rs` — `LiveChunkTables` → toroidal directory (the core change).
- `assets/shaders/sdf/brick.wgsl` — `find_chunk` → `find_chunk_toroidal`; `bindings.wgsl` — `R` + `O_lod`.
- `src/sdf_render/render.rs` — `extract_sdf_atlas` / `prepare_sdf_atlas_gpu` dense-slot upload; camera uniform `O_lod`/`R`.
- `src/sdf_render/atlas.rs` — `insert_gpu_brick` (atomic publish) / `remove_brick` (O(1) clear).
- `src/sdf_render/bake_scheduler.rs` — recenter already evicts on exit (no-deferral); confirm generation/`O_lod` bump on window-advance.
- Tests: `tests/sdf_gpu_rig.rs` (GPU parity), `tests/chunk_lookup_bench.rs` (rig — already landed), `chunk.rs`/`bake_scheduler.rs` unit + lifecycle tests.

---

## Phases (each builds + passes its tests before the next)

### Phase 1 — CPU directory in `chunk.rs` (behind the existing GPU contract)
Replace `LiveChunkTables`' sorted array with the toroidal hybrid, but keep producing the **same
`chunk_buf` bytes** the current shader expects (a *sorted snapshot* derived from the directory at
upload time), so the GPU is untouched this phase.
- New: per-LOD dense `Vec<DirEntry>` (`R³ × lod_count`), `slot = euclid_mod(c, R)` flattened + `lod·R³`.
  `set_brick`/`clear_brick` become O(1) slot writes (whole-record). Tile-runs keep the existing
  `ChunkSlotAllocator`; `clear_brick` frees the run; `set_brick` overwrite frees the old run (belt-and-
  suspenders even though explicit-clear is primary).
- Keep `dirty_slots` (now directory slots). `set_window_origins(O_lod)` each frame.
- **Tests:** extend the `shader_resolve`/`build_chunk_tables` mirror + the
  `live_delta_upload_matches_ground_truth_under_churn` pattern: after each scripted camera move/edit,
  `directory.lookup(c)` == the sorted-array ground truth for every brick (incl. negative coords,
  non-pow2 R, wrap). Atomic-publish + free-on-overwrite already covered by the rig's adversarial tests.

### Phase 2 — Generation + `O_lod` on window-advance
- Ensure `atlas.generation` bumps and the camera uniform re-uploads per-LOD `O_lod` + `R` on **every
  frame the window origin advances for any LOD**, independent of whether a bake applied (relocate /
  augment the `remove_brick` generation guard onto the recenter path — `bake_scheduler.rs`).
- **Test:** the `adversarial_flyaway_must_bump_generation` invariant, ported to the real types.

### Phase 3 — GPU lookup: `find_chunk_toroidal`
- `brick.wgsl`: `find_chunk(coord,lod)` → `in_ring_chunk(coord,lod)` (against uniform `O_lod`) →
  `slot = euclid_mod(chunk_coord, R)` flatten → read `chunk_buf[slot]` → compare `key` tag to
  `abs_chunk_key(coord,lod)`. Drop the binary search + `arrayLength` self-defense (the tag replaces it).
- `bindings.wgsl`: add `R` and per-LOD `O_lod` to the camera uniform; `euclid_mod` on the slot math.
- **Tests:** extend `tests/sdf_gpu_rig.rs` (the `gpu_find_brick_lookup_matches_cpu` pattern): pin
  `slot = euclid_mod(c,R)` CPU == GPU over negative coords + non-pow2 R, and the tag-compare resolve.
  Keep the existing `camera_uniform_bytes` mirror in sync (it just grew for `O_lod`/`R`).

### Phase 4 — Dense-slot upload in `render.rs`
- `chunk_buf` becomes the fixed-size dense directory (`R³ × lod_count × 20 B`), uploaded once; per frame
  `write_buffer` only the `dirty_slots`. Tile-run buffer upload unchanged (sparse, dirty runs).
- Delete `sentinel_tail_from` / `structure_changed` / the dirty-row re-stamp path (no tail shift now).

### Phase 5 — Delete dead code + verify end-to-end
- Remove `sorted_keys` / `key_to_row` / binary-search remnants and any now-unused delta machinery.
- Zero warnings on `cargo build` AND `cargo build --features editor`; clippy both; full `cargo test`
  (incl. gpu rig + lifecycle + the chunk_lookup_bench adversarials).
- **Live confirm (ask user to run):** re-capture a fly-through (F6); `sched_recenter` must no longer
  spike (was 448 ms) and stay a flat few-ms; no holes/ghosts; brick count bounded.

## Verification harness (already in place)
- `tests/chunk_lookup_bench.rs` — structure benchmark + the 3 adversarial regression tests.
- `tests/sdf_gpu_rig.rs` — CPU↔GPU parity (extend for slot+tag).
- `bake_scheduler.rs::live_table_resolves_correct_tile_through_recenter_lifecycle` — real recenter drain.
- Chrome trace (F6) — the `sched_*` spans confirm the spike is gone.
