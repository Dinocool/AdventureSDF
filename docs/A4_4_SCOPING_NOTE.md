# A4.4 — variable paletted streamed arena: scoped implementation note

Companion to `PHASE_A_GPU_EXECUTION.md` §A4.4. Scoped against the post-A3 code (commit `43af75fa`). This is the
**clean variable-arena (option B)** — the real VRAM win. Execute in fresh context (it's R2b/A1-scale).

## Goal
Replace A1-β's RAW fixed-block voxel arena on the STREAMED path with a PALETTED variable arena, recovering R2's
VRAM (the ~240 MB capacity reservation → ~tens of MB actual) while KEEPING A1's O(changed) `queue_write_buffer`
upload. The STATIC path (`pack_resident_set`/`snapshot_patch`) is already paletted — don't touch it.

## Design — SIZE-CLASS SLABS keyed by `index_bits` (the tractable variable allocator)
The R2 index stream is `ceil(1000·index_bits/32)` words: **{1,2,4,8,16}-bit → {32, 63, 125, 250, 500} words** = 5
size classes. Each class = a free-list of fixed-size blocks (no coalescing/fragmentation — fixed size within a
class). A dense brick's index block is allocated from ITS `index_bits` class. On a brick's `index_bits` CHANGE
(edit/neighbour change), free the old-class block (→ quarantine, keep-old) + allocate from the new class. The
slab buffers are allocated once per epoch (generous per-class capacity) + grow-on-overflow (rare).

**Palette:** same size-class-slab approach for `brick_palettes` (classes by k's rounded size), OR a simpler
**Checkpoint-1**: fixed `MAX_PAL=512` per slot (~60 MB) for the palette while the INDEX uses slabs (the big
win), then **Checkpoint-2** makes the palette variable too. Land Checkpoint-1 first (commit) — it's the safe
~3× and a coherent working state; Checkpoint-2 is the remaining palette win.

## Reuse (don't rewrite the encode)
`gpu.rs` `encode_paletted` / `VoxelInterner::intern_paletted` already turn raw haloed cells → `(palette,
index_bits, indices)` — `snapshot_patch` (incremental.rs:523) uses them for the CONTIGUOUS path. The slab arena
uses the SAME encode but writes into slot-stable slab blocks instead of a contiguous buffer.

## Change sites (confirmed by reading)
- `src/voxel/incremental.rs`:
  - The arena allocator: `arena_high_water`/`arena_free`/`arena_capacity` (fixed-block free-list, `dense_block_u32()`=1000) → **per-`index_bits`-class slab free-lists**.
  - `SnapshotBuffers` + `snapshot_buffers()` (≈180/289): build the SLAB index arena (paletted, per-class blocks) + a per-slot/slab palette buffer — NOT the raw 1000-block arena.
  - `emit_changed_slot()` dense branch (≈474): `encode_paletted(cells)`; allocate from the `index_bits` class; set the meta's REAL `index_bits`/`palette_base` (drop the `index_bits==0` raw marker); on class change, quarantine the old block; push the paletted `ChangedSlot`.
  - `ChangedSlot` (≈127): carry the paletted index block + the palette block + their offsets (not the raw 1000 block).
  - `last_voxels` stays RAW cells (encode at emit/snapshot, exactly like `snapshot_patch`) — so the byte-identity A/B gate still holds.
- `src/voxel/raytrace.rs`: `prepare_voxel_rt` StreamSnapshot/Delta consumption — allocate the slab index buffer + the palette buffer (add `COPY_DST`); BIND `brick_palettes` for the streamed path (group0/binding12, as the static path does — A1-β bound a dummy); `queue_write_buffer` only changed slots' index + palette blocks.
- `assets/shaders/voxel_raytrace.wgsl`: the streamed path now uses the EXISTING paletted `cell_block` decode (the static path already does); the raw `index_bits==0` branch becomes streamed-unused — keep it (harmless) or remove once nothing references it.

## Gate
`voxel_raytrace_gpu` GPU-vs-CPU byte/pixel-identical (THE gate) + the full GPU suite + the incremental A/B
byte-identity tests (the slab snapshot/delta must equal the paletted content) + `voxel_render_headless` + both
feature builds zero-warning + clippy. Report streamed-path resident VRAM before/after on the worldgen slice.
Commit-or-revert per checkpoint; never commit a half-applied arena (black screen).
