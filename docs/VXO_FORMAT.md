# `.vxo` Native Asset Format & Region-Streamed Loader — Implementation Spec

Status: SPEC (no engine code changed by this doc). Worktree: `voxel-rt`. This is the **implementation-ready**
spec for **Phase B** of the voxel-RT program (`docs/VOXEL_PROGRAM.md` §"Phase B"): B1 the on-disk `.vxo`
format, B2 the region-streamed loader, B3 the R6 SVDAG asset transport. A future implementation agent
executes from this doc alone. Read `docs/VOXEL_PROGRAM.md` (the sequencing), `docs/VOXEL_INSTANCING_PLAN.md`
§1.5 (the original chunk sketch), `docs/VOXEL_STORAGE_PLAN.md` §R5/§2 (Tier-A/B split + SVDAG notes), and
`docs/SOTA_REFERENCE.md` §1 + §6 (storage methods + scene corpus) for the surrounding context.

The format is **engine-owned**; `.vox` becomes **import-only** (offline). The shipped runtime depends only on
the `.vxo` reader (`src/voxel/vxo/`); `dot_vox`/`gltf`/`image` stay DEV/offline deps (`examples/`).

---

## 0. Design constraints (the non-negotiables this spec is built around)

These are read straight from the code and the storage plan; every layout decision below honours them.

1. **`BRIK` stores the R2b encoding so a read-back brick is bit-identical + its packed `GpuBrickPatch` is
   byte-identical** — NOT a literal memcpy of brick bodies into the GPU arena. The resident VRAM form
   (`src/voxel/gpu.rs`) is the **R2b per-brick triple**: a tiny per-brick palette (`brick_palettes: Vec<u32>`,
   the `k` distinct `u16` ids zero-extended), a bit-packed index stream (`voxels: Vec<u32>`,
   `index_bits ∈ {1,2,4,8,16}`), plus a 48-byte `GpuBrickMeta` and a 32-byte `GpuBrickAabb`. A uniform brick
   (R1) emits NO index/palette bytes — its single id rides in the meta's low 16 bits with the dedicated
   `META_FLAG_UNIFORM` (A4.1; see constraint 1a below). **What B-i actually delivers (the honest property):** the
   disk stores each brick's **8³ CORE** R2b-encoded; the loader decodes it to a `Brick`, and the packer
   (`pack_one`) re-halos + re-encodes it from the resident set. So a streamed brick is a **bit-identical
   `Brick`** and its **packed `GpuBrickPatch` is byte-identical** to a live one — the CORRECT `BrickSource`
   contract — but loading is a decode-then-pack, NOT a raw memcpy of stored bodies into the arena (the running
   `voxel_offset`/`palette_base`/`world_min`/AABB are arena-relative, recomputed at pack time, never stored).
   This still honours `VOXEL_STORAGE_PLAN.md` R5 ("share R2's per-brick palette format between disk and VRAM so
   the loader expands a brick with minimal transcode"): the disk and resident encodings are the same R2b form,
   so the transcode is just the re-halo, not a format conversion.

   **1a. Uniform discriminant — a DEDICATED flag bit, not bit-31 of an offset (A4.1).** `gpu.rs` RETIRED the
   old `BRICK_UNIFORM_FLAG = 1<<31` of `voxel_offset` — it silently capped a real dense offset at `< 2^31` (a
   corruption trap once the arena passes `2^31` `u32`s). The uniform marker now lives in a dedicated
   `GpuBrickMeta.flags` word (`META_FLAG_UNIFORM = 1`). The `.vxo` disk form MIRRORS this: the uniform marker is
   a dedicated `VxoBrickEntry.flags` bit (`BRICK_FLAG_UNIFORM = 1<<2`; bit0=surface, bit1=full are taken), so
   `VxoBrickEntry.index_off` uses the FULL `u32` range for a dense brick's region-local offset. The encoder also
   guards that each region's blob lengths/offsets fit `u32` (region-local ⇒ always true today, a robust-by-
   construction backstop).

2. **Pointer-free / fixed-layout / mmap-able.** Every chunk body is a flat, self-describing run of
   POD records addressed by byte offset from the chunk start — no in-file pointers, no `Box`, no relocation.
   A region-chunk decompresses into one `Vec<u8>` (or, for an uncompressed `STORE` chunk, an mmap slice) and
   is read with `bytemuck::cast_slice`. This is the NanoVDB discipline (`VOXEL_STORAGE_PLAN.md` §2).

3. **Stream BY REGION.** The loader must seek + decompress ONLY the region-chunks a clipmap shell needs, never
   the whole file. This is **impossible without a spatial directory** — the gap the audit flagged
   (`VOXEL_PROGRAM.md` B1). The `BIDX` chunk (§B1.5) is that directory. The region granularity is aligned to
   the residency clipmap (`brick_coord_of_voxel` / `BrickKey`).

4. **Self-describing `voxel_size`.** `HEAD.voxel_size` is stored as an `f32` so a `0.05 m` asset is
   unambiguous regardless of the engine's current `VOXEL_SIZE` (`brickmap.rs` is `0.2` today, flips to `0.05`
   at D1). The loader asserts/rescales against the live `VOXEL_SIZE` (§B2.6).

5. **NO 256-colour cap.** `MATL` is keyed by `u16` `BlockId` with NO 256 limit (the `.vox` cap is exactly
   what we shed). `BlockRegistry` (`palette.rs`) already stores a `Vec<BlockDef>` of arbitrary length.

6. **Feed the EXISTING `ResidencyManager` demand path.** A `.vxo` scene is a `BrickSource`
   (`src/voxel/source.rs`) — the SAME trait `WorldgenSource`/`StaticVoxSource` implement — so static-scene
   load and worldgen streaming share ONE residency SSOT (`VOXEL_PROGRAM.md` B2).

---

## B1 — The `.vxo` on-disk format

### B1.0 File skeleton (RIFF-style, tagged + length-prefixed + skippable)

```
+----------------+
| FILE HEADER    |  magic "VXO1" (4B) + format_version u16 + flags u16
+----------------+
| CHUNK: HEAD    |  tag 'HEAD' | u64 body_len | body…   (REQUIRED, first)
| CHUNK: MATL    |  tag 'MATL' | u64 body_len | body…   (REQUIRED)
| CHUNK: BIDX    |  tag 'BIDX' | u64 body_len | body…   (REQUIRED — the region directory)
| CHUNK: BRIK    |  tag 'BRIK' | u64 body_len | body…   (REQUIRED — the region-chunk blob, BIDX indexes into it)
| CHUNK: LITE    |  …                                   (OPTIONAL — baked NEE light list)
| CHUNK: LODS    |  …                                   (OPTIONAL — coarse mip pyramid)
| CHUNK: INST    |  …                                   (OPTIONAL — scene of instances)
| CHUNK: END     |  tag 'END ' | u64 0                  (sentinel, optional)
+----------------+
```

**File header (16 B, fixed):**

| Field | Type | Notes |
|---|---|---|
| `magic` | `[u8;4]` | `b"VXO1"` |
| `format_version` | `u16` | starts at `1`; bump only on a breaking framing change |
| `flags` | `u16` | bit0 = whole-file is little-endian (always 1 — we target LE x86/Vulkan); bit1 = `BRIK` bodies are SVDAG-encoded (B3); other bits reserved 0 |
| `header_crc32` | `u32` | CRC32 of the 8 bytes above (cheap integrity check; loader logs + rejects on mismatch) |
| `_reserved` | `u32` | 0 |

**Chunk framing (every chunk):**

| Field | Type | Notes |
|---|---|---|
| `tag` | `[u8;4]` | ASCII, e.g. `b"BRIK"` |
| `body_len` | `u64` | byte length of `body` (NOT incl. this 16-B header) |
| `body_crc32` | `u32` | CRC32 of `body` (optional verify; 0 = skip) |
| `_pad` | `u32` | 0 → 16-B aligned header so bodies start 16-aligned (mmap + `bytemuck` happy) |
| `body` | `[u8; body_len]` | chunk-specific; padded to a 16-B multiple with zeros (the pad is OUTSIDE `body_len`, so `body_len` is exact) |

**Reader rule (forward/back compat):** read the file header, then loop: read a chunk header, dispatch on `tag`
to a known parser or **skip `body_len` (rounded up to 16)** if unknown. A missing OPTIONAL chunk is fine
(defaults). A missing REQUIRED chunk (`HEAD`/`MATL`/`BIDX`/`BRIK`) is a hard error. This is the one good `.vox`
idea kept (`VOXEL_INSTANCING_PLAN.md` §1.5): unknown chunks skip, dropping an optional chunk doesn't break old
assets.

### B1.1 `HEAD` — self-describing geometry + identity (REQUIRED, first)

Flat POD, fixed layout:

| Field | Type | Bytes | Notes |
|---|---|---|---|
| `head_version` | `u16` | 2 | per-chunk schema version (independent of file `format_version`) |
| `_pad0` | `u16` | 2 | 0 |
| `voxel_size` | `f32` | 4 | **metres per LOD0 voxel** (e.g. `0.05`). SELF-DESCRIBING (§0.4). |
| `brick_edge` | `u32` | 4 | voxels per brick edge — MUST equal `brickmap::BRICK_EDGE` (8); loader asserts |
| `max_lod` | `u32` | 4 | the LODS-chunk pyramid depth (0 if no `LODS`); ≤ `brickmap::MAX_LOD` |
| `bounds_min` | `[i32;3]` | 12 | inclusive LOD0 world-VOXEL min corner of the asset's solid extent |
| `bounds_max` | `[i32;3]` | 12 | exclusive LOD0 world-VOXEL max corner |
| `anchor_voxel` | `[i32;3]` | 12 | the asset PIVOT in LOD0 world-voxel coords (`bricks_from_placed`'s floor/centre anchor recorded, NOT baked — `VOXEL_INSTANCING_PLAN.md §1.2`). For a merge-into-world scene this is `(0,0,0)`. |
| `region_edge_bricks` | `u32` | 4 | **K** — the region-chunk granularity: a region is `K×K×K` bricks (§B1.4). MUST be a power of two and align to the clipmap residency (default **K = 8**). |
| `brick_count` | `u64` | 8 | total non-empty LOD0 bricks (for load-budget pre-allocation) |
| `region_count` | `u32` | 4 | number of `BIDX` entries (non-empty regions) |
| `_pad1` | `u32` | 4 | 0 |
| `name_len` | `u32` | 4 | UTF-8 name byte length |
| `name` | `[u8; name_len]` | … | asset name / tags (debug + path-cache key); padded to 4-B in-body |

`voxel_size`, `bounds`, `anchor`, `region_edge_bricks`, `brick_count` are everything the loader needs to size
buffers and locate a brick's region BEFORE touching `BRIK`.

### B1.2 `MATL` — material table per `u16` BlockId (REQUIRED, NO 256 cap)

The on-disk form of `BlockRegistry` (`palette.rs` `BlockDef`) — but a flat POD table indexed by `BlockId`
(block 0 = AIR). Loader rebuilds a `BlockRegistry` from it (a new `BlockRegistry::from_vxo_matl(&[VxoMaterial])`
constructor, sibling to `from_vox_palette`). No `mat_to_block` bridge (a baked asset has no `TerrainMatId`
chain — same as `from_vox_palette` today).

**Body:** `material_count: u32` then `material_count × VxoMaterial`:

```rust
#[repr(C)] #[derive(Pod, Zeroable)]
struct VxoMaterial {           // 48 bytes, 16-aligned
    albedo:   [f32; 4],        // LINEAR RGBA (already decoded — disk stores linear, unlike .vox sRGB)
    emissive: [f32; 4],        // LINEAR RGB radiance in .xyz; .w = emissive_strength multiplier (default 1.0)
    roughness: f32,            // reserved-but-present (renderer grows into it; default 1.0)
    metallic:  f32,            // reserved-but-present (default 0.0)
    flags:     u32,            // bit0 = tintable; bit1 = emitter (precomputed = any(emissive>0)); rest reserved 0
    _pad:      u32,            // 0
}
```

Index `i` → `BlockId(i)`. Entry 0 is AIR (transparent black, the `BlockDef::air()` values). `albedo`/`emissive`
mirror `GpuPaletteColor` (`gpu.rs`) so the resident palette buffer is a near-direct copy. `roughness`/`metallic`
are the "reserved fields the renderer can grow into without a format break" (`VOXEL_INSTANCING_PLAN.md §1.5`) —
present from v1 so adding PBR later is a value-population, not a schema break. New fields ⇒ `head_version`/`MATL`
version bump, old readers default them (the forward-compat rule).

> **C3 hook (import fidelity):** the `u16` cap-lift (`VOXEL_PROGRAM.md` C3) rides this chunk — the offline
> CIELAB clustering may emit > 256 materials; `MATL` has no cap, so it just works.

### B1.3 `BRIK` — the sparse bricks, R2b body, pointer-free (REQUIRED)

`BRIK` is the **concatenation of all region-chunks**; `BIDX` (§B1.5) gives each region's byte slice within it.
Each region-chunk is **independently zstd-compressed** (or `STORE`d) so a single region decompresses without
touching its neighbours. The loader NEVER decompresses the whole `BRIK` body.

**One region-chunk, decompressed, is a flat blob:**

```
REGION CHUNK (decompressed) =
  region_header: VxoRegionHeader      // 32 B
  brick_dir:     [VxoBrickEntry; N]   // N = region_header.brick_count, the bricks in this region
  palette_blob:  [u32; P]             // all dense bricks' palettes, concatenated (per-brick slices)
  index_blob:    [u32; I]             // all dense bricks' bit-packed index streams, concatenated
```

```rust
#[repr(C)] #[derive(Pod, Zeroable)]
struct VxoRegionHeader {        // 32 B
    region_coord: [i32; 3],     // the region's coord on the K-brick grid (redundant w/ BIDX key; verify)
    brick_count:  u32,          // N bricks in this region (all LOD0; coarse LODs live in LODS)
    palette_u32:  u32,          // P — length of palette_blob (region-local base = 0)
    index_u32:    u32,          // I — length of index_blob  (region-local base = 0)
    lod:          u32,          // 0 for the base BRIK; LODS regions carry their level (§B1.7)
    _pad:         u32,
}

#[repr(C)] #[derive(Pod, Zeroable)]
struct VxoBrickEntry {          // 32 B — one resident brick, decode-ready
    brick_coord:  [i32; 3],     // LOD0 brick coord (absolute, world grid)
    // R1 uniform OR R2b dense, distinguished by the dedicated BRICK_FLAG_UNIFORM bit in `flags` (A4.1):
    index_off:    u32,          // if UNIFORM (flags & BRICK_FLAG_UNIFORM): low 16 bits = the uniform BlockId.
                                // else: REGION-LOCAL u32 offset into index_blob — the FULL u32 range, no
                                //   reserved high bit.
    palette_off:  u32,          // REGION-LOCAL u32 offset into palette_blob (dense only; 0 for uniform)
    index_bits:   u8,           // ∈ {1,2,4,8,16} (dense only; 0 for uniform)
    palette_len:  u8,           // k distinct ids (dense; ≤ 255 — a 10³ brick can't exceed that meaningfully;
                                //   if a pathological brick needs >255, split index_bits=16 path, store k in
                                //   a u16 via `_pad` — see note). 0 for uniform.
    flags:        u8,           // bit0 = surface (LITE/light-gather hint); bit1 = fully-solid (classify cull);
                                //   bit2 = BRICK_FLAG_UNIFORM (R1 collapse — the uniform discriminant)
    _pad0:        u8,
    _pad1:        u32,          // 0 → 32-B aligned
}
```

Critical: `index_off`/`palette_off` are **region-local** (base 0 within this region's blobs). **The uniform
discriminant is a DEDICATED `flags` bit (`BRICK_FLAG_UNIFORM = 1<<2`), mirroring `gpu.rs`'s `META_FLAG_UNIFORM`
(A4.1)** — NOT the old `1<<31`-of-`index_off` convention, which `gpu.rs` retired to free `voxel_offset` for the
full `u32` range (a silent-corruption trap past `2^31`). So `index_off` uses the FULL `u32` for a dense brick's
region-local offset; the encoder guards the blob lengths fit `u32` (region-local ⇒ always true, a backstop).
`index_bits`/`palette_len`/the index stream are **byte-identical to `encode_paletted(cells)`** (`gpu.rs`) — the
disk stores the R2b core output verbatim; the read-back `Brick` and its re-packed `GpuBrickPatch` are
bit-/byte-identical to a live one (a decode-then-pack, see §0.1, not a raw memcpy).

The cells stored per brick are the **haloed `10³` grid** (`halo_cells(0)`), exactly as `pack_one` produces
(core + 1-cell neighbour border, AIR where absent). Storing the halo means the loaded brick is trace-ready with
NO neighbour re-read at load (the seam fix travels with the asset). **Trade-off vs. recompute:** storing the
halo costs ~`10³/8³ ≈ 1.95×` index entries but makes a region-chunk decode purely local (no cross-region
neighbour fetch). For a streamed scene this locality is worth it; the zstd wrapper reclaims most of the halo
redundancy (a buried brick's halo is constant). (If profiling later shows the halo blob dominates, an
alternative is to store only the `8³` core and re-halo at load from the resident set — but that re-introduces
cross-region coupling, so default to storing the halo.)

> **R3 dedup on disk:** identical haloed bricks within a region share ONE `(palette_off, index_off)` slice
> (the encoder interns with `VoxelInterner`, `gpu.rs`). Two `VxoBrickEntry` rows then point at the same
> region-local offsets. Cross-region dedup is NOT done in v1 (regions are independently decompressible — a
> shared slice across regions would break that); intra-region dedup captures the common case (strata bands,
> repeated columns) within the K³-brick window.

> **`palette_len > 255` note:** a `10³`-cell brick has ≤ 1000 cells, so ≤ 1000 distinct ids is theoretically
> possible but practically absurd (it would mean ~every cell distinct). `index_bits=16` already covers up to
> 65535 ids; `palette_len` only sizes the palette slice and the decode reads `palette_blob[palette_off ..]`
> bounded by the index width, so a `u8` `palette_len` is a load-time PREALLOC hint, not a correctness bound.
> If `k > 255`, store `palette_len = 0` as a sentinel meaning "derive k from `index_bits` / the next entry's
> `palette_off`". Keep it simple: assert `k ≤ 255` in the encoder for v1 (no shipping brick approaches it).

### B1.4 Region granularity (the chunking)

- A **region** is `K×K×K` LOD0 bricks, `K = HEAD.region_edge_bricks` (default **8**). Region coord =
  `brick_coord.div_euclid(K)` per axis (Euclidean, correct for negative coords — mirror
  `brick_coord_of_voxel`). A region spans `K · BRICK_WORLD_SIZE` metres = `8 · 8 · voxel_size`; at 0.05 m that
  is `8·8·0.05 = 3.2 m` per axis — a tight streaming granule, a few hundred KB compressed.
- **Alignment to the clipmap:** the residency clipmap (`streaming.rs` `desired_clipmap`) requests bricks by
  `(coord, lod)`. A region groups bricks the clipmap tends to need together (spatial locality), so demanding
  one brick warms its `K³` neighbours in one decompress — amortizing IO over the shell. `K = 8` ⇒ a region is
  `512` bricks max; a LOD0 clipmap shell of `clip_half_bricks = 8` touches ~`(2·8)² = 256` surface bricks
  spread over a handful of regions.
- Only **non-empty** regions get a `BIDX` entry + a `BRIK` slice (sparsity — empty space costs nothing, same
  as `BrickMap`). A demanded brick in a region with no `BIDX` entry is AIR (the clipmap bound, exactly like
  `StaticVoxSource::wholly_outside`).

### B1.5 `BIDX` — the spatial region DIRECTORY (REQUIRED — the missing piece)

This is the chunk that makes "stream by region" possible (`VOXEL_PROGRAM.md` B1: "unimplementable without
it"). A **sorted** table mapping a region coord to its byte slice in `BRIK`, so the loader binary-searches a
region, seeks, and decompresses ONLY it.

**Body:** `entry_count: u32` (= `HEAD.region_count`), `_pad: u32`, then `entry_count × VxoRegionDirEntry`,
**sorted by `(region_coord.z, .y, .x)`** (so a coord → entry is an O(log n) binary search; the sort key
mirrors `pack_brickmap`'s deterministic order):

```rust
#[repr(C)] #[derive(Pod, Zeroable)]
struct VxoRegionDirEntry {      // 48 B (32 + the explicit compression byte, padded to a 16-multiple)
    region_coord:   [i32; 3],   // the K-brick-grid region coord (the search key)
    brick_count:    u32,        // bricks in this region (preallocate the decode)
    brik_offset:    u64,        // byte offset of this region's chunk WITHIN the BRIK chunk body
    brik_comp_len:  u32,        // COMPRESSED byte length of the region chunk (the seek+read span)
    brik_raw_len:   u32,        // DECOMPRESSED byte length (preallocate the decode output)
    compression:    u8,         // EXPLICIT: 0 = STORE (uncompressed), 1 = zstd. The reader BRANCHES ON THIS,
                                //   never on `comp_len == raw_len` (a zstd body can compress to exactly raw len).
    _pad:           [u8; 15],   // 0 → 48-B (16-multiple) stride (the u64 forces 8-align; pad to a clean 16x).
}
```

- `brik_offset` + `brik_comp_len` = the exact `pread` span; `brik_raw_len` sizes the decompress buffer (or, if
  `compression == STORE`, lets the loader mmap the slice directly — `brik_raw_len == brik_comp_len`). The
  STORE-vs-zstd choice is the EXPLICIT `compression` byte, NOT length equality (FIX 3 — length equality silently
  corrupts when a zstd body happens to compress to exactly its raw length). Mmap-ability (§0.2):
  with a `STORE` (uncompressed) `BRIK` the whole file is `bytemuck`-castable and a region needs zero copy; with
  zstd, the region decompresses into a cached `Vec<u8>` (§B2).
- The directory is small (`region_count × 32 B`; even a Bistro-scale scene with ~50k non-empty regions is
  ~1.6 MB) and is loaded **eagerly** at open (it is the index). `BRIK` region bodies are loaded **lazily**.
- A coarse-LOD region directory (the `LODS` pyramid, §B1.7) is a SEPARATE sub-table per level, or `lod` is
  folded into the search key — v1 keeps base-LOD `BIDX` here and puts coarse-LOD directories inside `LODS`.

### B1.6 `LITE` — baked NEE light list (OPTIONAL, sketch — defer)

The pre-extracted air-exposed emissive-voxel light list (`gpu.rs` `GpuVoxelLight` + the power-weighted
`GpuAliasEntry` table), baked offline so the runtime skips `build_light_list`/`gather_lights_into` on load.
Body = `light_count: u32` + `light_count × VxoLight` (mirror `GpuVoxelLight` 32 B: `pos[3]`, `area`,
`radiance[3]`, `inv_pdf`) + the alias table. **Per-region** light sub-lists keyed like `BIDX`, so streaming a
region brings in its lights (merged into the resident `lights`/`alias`, capped at `MAX_VOXEL_LIGHTS`).
Deferred: v1 loader rebuilds lights from the resident bricks (the existing path), correctness-first; `LITE` is
a later load-time optimization. (`VOXEL_INSTANCING_PLAN.md §1.5`.)

### B1.7 `LODS` — coarse mip pyramid (OPTIONAL, sketch — defer)

The pre-baked coarse-LOD bricks so far shells don't downsample at load (today `StaticVoxSource::new` builds the
`MAX_LOD+1` pyramid in RAM at load — fine for one Sponza, too heavy for streamed Bistro). Body = `max_lod: u32`
then, per level `L ∈ 1..=max_lod`: a `(BIDX_L, BRIK_L)` pair in the SAME region+entry layout as §B1.3/B1.5 but
on level-`L`'s coarser brick grid (each coarse brick = the `solid-if-any + dominant-block` downsample,
`source.rs` `downsample_brickmap` — bake it offline). The loader's `BrickSource::brick(coord, L>0)` then reads
the `LODS` region for level `L` instead of downsampling. Deferred: v1 loader either (a) only serves LOD0 from
`.vxo` and downsamples coarse LODs in RAM from the loaded LOD0 regions (bounded because residency is
surface-only), or (b) for very large scenes, ships `LODS` so coarse shells stream too. Decide at B2 from the
gallery-worst-case RAM measurement.

### B1.8 `INST` — scene of instances (OPTIONAL, sketch — defer to the instancing track)

A list of `{ object_ref: path or BIDX-to-another-.vxo, transform: [f32;12] (3×4 world_from_object),
per_instance_edits_ref }` — the on-disk form of the `VoxelInstance` tree (`VOXEL_INSTANCING_PLAN.md §3.2`).
This is how a `.vxo` carries nested sub-scenes / authored prop placements. Body sketch: `instance_count: u32` +
`instance_count × VxoInstance { object_path_off: u32, transform: [f32;12], mask: u32, edit_ref: u32 }` + a
string blob for paths. Deferred entirely — the merge-into-world scenes (the gallery corpus) need only
`HEAD/MATL/BIDX/BRIK`. The instancing track (`VOXEL_INSTANCING_PLAN.md` Phases 2-6) consumes this.

### B1.9 zstd wrapping + mmap

- **Per-region zstd.** Each region-chunk in `BRIK` is compressed independently; `BIDX` carries comp/raw lengths
  + the explicit `compression` code. Level ~`19` offline (size) is fine — decode speed is what matters at
  runtime and zstd decode is ~GB/s.
  - **DECODE = pure-Rust `ruzstd`, a RUNTIME dep** (matches the project's ktx2/`ruzstd` "no C toolchain"
    discipline). The shipped library/runtime reader (`voxel::vxo::reader`) decodes with `ruzstd`, so the default
    build pulls NO C `zstd-sys`/`cc`/`pkg-config` (FIX 2).
  - **COMPRESS = C `zstd`, OFFLINE-ENCODE ONLY, behind the `vxo-encode` cargo feature.** Only the offline encoder
    (`write_vxo`'s `VxoCompression::Zstd`, used by `examples/voxelize_scene.rs --features vxo-encode`) links the
    C `zstd` crate. `ruzstd` decodes standard zstd frames, so what the C encoder produces the runtime reader
    reads back. A default-feature `write_vxo(.., Zstd)` errors clearly (use `Store` or enable `vxo-encode`).
- **`STORE` mode** (the explicit `compression == 0` per region): no compression ⇒ the region body is
  `bytemuck`-castable in place from an mmap of the file. Use for assets where load latency trumps disk size (or
  for the round-trip test, which wants byte-identity without a zstd-compress dep).
- **Whole-file mmap:** `HEAD`/`MATL`/`BIDX` are read once eagerly (small); `BRIK` is read lazily per region.
  With `memmap2` (offline+runtime) the loader mmaps the file and either casts (`STORE`) or `zstd::decode` a
  region slice. The file is the durable store; the only RAM held is `HEAD`+`MATL`+`BIDX` + the decoded-region
  LRU cache (§B2.2) + the resident `ResidencyManager` set — NEVER the whole expanded scene (§0 + the B2
  acceptance gate).

---

## B2 — The region-streamed loader

### B2.1 Shape: a `.vxo` scene is a `BrickSource`

A new `src/voxel/vxo/` module (`mod.rs` + `reader.rs` + `source.rs`). The public type:

```rust
/// A memory-mapped .vxo file exposed as a streamed BrickSource — the read side feeding the SAME
/// ResidencyManager demand path as WorldgenSource/StaticVoxSource.
pub struct VxoSource {
    mmap:    memmap2::Mmap,                 // the whole file (durable; regions read lazily)
    head:    VxoHead,                       // parsed HEAD (voxel_size, bounds, K, anchor…)
    bidx:    Vec<VxoRegionDirEntry>,        // sorted region directory (eager)
    cache:   Mutex<RegionCache>,            // decoded-region LRU (§B2.2)
    offset_voxels: IVec3,                   // merge offset (§B2.4): added to incoming brick coords
    // registry rebuilt from MATL is returned alongside, like load_vox's (BrickMap, BlockRegistry)
}

impl BrickSource for VxoSource {
    fn brick(&self, coord: IVec3, lod: u32, _registry: &BlockRegistry) -> Brick { … }   // §B2.3
    fn classify(&self, coord: IVec3, lod: u32) -> BrickClass { … }                       // §B2.5
}
```

`VxoSource::open(path) -> anyhow::Result<(VxoSource, BlockRegistry)>` — mirror `load_vox`'s `(map, registry)`
return so the scene-load call site swaps `load_vox` + `StaticVoxSource::new` for `VxoSource::open` with no
shape change. The `ResidencyManager::update`/`drain_work_from` loop is **unchanged** — it sources bricks
through `&dyn BrickSource`, applies the shared `edits` overlay, stores non-empty results (`source.rs` module
doc). One residency SSOT for worldgen + static-`.vox`(legacy) + `.vxo` (`VOXEL_PROGRAM.md` B2).

### B2.2 `brick(coord, lod)` — seek → decompress → cache → return

For `lod == 0` (and `lod > 0` once `LODS` is wired; §B1.7):

1. Apply the merge offset: `local = coord - offset_bricks` (§B2.4).
2. `region = local.div_euclid(K)` (Euclidean per axis).
3. **Binary-search `bidx`** for `region` (sorted by z,y,x). Absent ⇒ return `Brick::uniform(AIR)` (the clipmap
   bound — `wholly_outside`-equivalent; the residency memoizes empties, so no re-source). This makes a `.vxo`
   scene self-bounding exactly like `StaticVoxSource`.
4. **Region cache lookup** (`RegionCache`, an LRU keyed by `region`). Miss ⇒ `pread`
   `bidx[i].brik_offset .. +brik_comp_len` from the mmap, `zstd::decode` into a `brik_raw_len` buffer (or use
   the mmap slice directly if `STORE`), parse `VxoRegionHeader` + the `VxoBrickEntry` table + the
   `palette_blob`/`index_blob` into a `DecodedRegion { entries: Vec<VxoBrickEntry>, palette: Vec<u32>,
   index: Vec<u32> }`. Insert into the LRU (evict the least-recently-used past a byte budget, §B2.2 budget).
5. **Find the brick** within the region: the region's `entries` are sorted by `brick_coord`, binary-search for
   `local`. Absent ⇒ `Brick::uniform(AIR)`.
6. **Decode the brick to a `Brick`** (the in-RAM CPU brick the residency stores): if the entry's `flags` has
   `BRICK_FLAG_UNIFORM` (A4.1 dedicated bit) ⇒ `Brick::uniform(BlockId(index_off & 0xFFFF))`. Else decode the
   `8³` core cells via
   `decode_paletted_cell(&region.palette[palette_off..], index_bits, &region.index[index_off..], cell)` for
   each core cell (the `8³` core — strip the halo when building the CPU `Brick`, since the packer re-halos from
   the resident set; OR, optimization, keep the halo and feed it through — see §B2.7) → `Brick::from_voxels`.
   This reuses the EXACT `gpu.rs` decode SSOT (`cell_block`/`decode_paletted_cell`), so the loaded brick is
   bit-identical to the live-generated one (the round-trip acceptance gate, §B2.8).

> **Why this never fully-expands the scene:** only demanded regions decode; the LRU caps decoded RAM; the
> `ResidencyManager` caps resident bricks (`max_resident_bricks`, surface-only after `classify`). The 2.6 GB
> Bistro `.vxo` on disk never materializes in RAM — only the surface shell the camera sees, region by region.

**Cache / LRU.** `RegionCache` is an LRU `FxHashMap<IVec3, Arc<DecodedRegion>>` + a usage list, bounded by a
**byte budget** (`decoded_region_budget`, default ~`256 MB`). Eviction drops the least-recently-touched region.
A region is `Arc`'d so an in-flight parallel drain (`drain_work_from` runs the batch in parallel) can hold a
decoded region across the lock release. Because `brick()` MUST be `Sync` + a pure function of its inputs
(`source.rs` trait contract), the cache is behind a `Mutex` (or a sharded lock / `dashmap`) and is a pure
*memoization* — two threads decoding the same region get the same bytes, so determinism holds (the cache is
observationally transparent).

### B2.3 Async vs sync decode

- **v1: synchronous decode inside `brick()`**, parallelized by the EXISTING `drain_work_from` (it already runs
  the per-frame brick batch in parallel across the rayon pool — `source.rs` module doc: "the parallel drain
  voxelizes the per-frame batch IN PARALLEL"). zstd decode of one ~few-hundred-KB region is sub-ms; the
  `max_bricks_per_frame` budget (`streaming.rs`) already bounds per-frame work, so a region miss costs one
  decode amortized over the K³ bricks it serves. This is the smallest change and reuses all the streaming
  back-pressure machinery.
- **Deferred (only if the gate fails): async prefetch.** If a region miss on the critical shell causes a
  visible hitch, add a background decode thread that prefetches the regions the clipmap is *about to* need
  (the shell one ring out), feeding the cache ahead of demand — the same "keep-old-until-revealed" lifecycle
  the streamer already has. Gate this behind a measured hitch on the perf rig; don't build it speculatively
  (`VOXEL_PROGRAM.md` defers demand/LRU "behind a concrete measurement gate").

### B2.4 Merged-gallery offset composition

The gallery loads several scenes "into the world brick map" (the merge path — `SOTA_REFERENCE.md` §6: "scenes
load into the world brick map, not per-object instances — user-confirmed"). Composition is a pure **coordinate
offset**, computed at open and added to every incoming `coord`:

- Each `VxoSource` gets an `offset_bricks: IVec3` placement in the world (e.g. Sponza at origin, Sibenik
  shifted +X by its width + a gap). `brick(coord, lod)` maps the world `coord` back to asset-local via
  `coord - offset_bricks` (step 1 of §B2.2). `classify` likewise.
- A merged scene is then either (a) **one `BrickSource` per asset** with the residency querying each and taking
  the non-air result (a thin `MergedSource { sources: Vec<(IVec3 offset, VxoSource)> }` that dispatches by
  which asset's bounds contain the coord — bounds from each `HEAD`), or (b) a single rebased `.vxo` baked
  offline with all assets already placed (the offline encoder merges). v1 ships (a) `MergedSource` — it needs
  no re-bake and composes N independent `.vxo` files; each region read still hits exactly one asset's mmap.
  `MATL` palettes concatenate with a per-asset `block_base` offset (the `palette_base` idea from
  `VOXEL_INSTANCING_PLAN.md §1.4`, applied to the merged registry) so two assets' `BlockId(5)` don't collide —
  the merged `BlockRegistry` is the concatenation, each asset's brick ids remapped by its base at decode (a
  cheap add in step 6). Keep this in `MergedSource`, one SSOT for the offset+rebase.

### B2.5 `classify` for the surface-only residency

`VxoSource::classify(coord, lod)` must implement the SAME conservative enclosed-cull `StaticVoxSource` does
(`source.rs`) so the surface-only residency (the Θ(H²) win, `VOXEL_PROGRAM.md` A2) applies to `.vxo` scenes:
a brick is `Interior` (prunable) iff it is fully solid AND all 6 face-neighbours are fully solid. To answer
this WITHOUT decoding the brick's voxels, the encoder bakes a **per-brick `is_full` bit** into `VxoBrickEntry.
flags` (bit1 = "fully solid"), and `classify` reads the region directory + brick entries (cheap: a region
decode is already cached when its bricks are demanded; for classify-before-demand, the entry table alone — not
the palette/index blobs — answers `is_full`). If the region isn't cached, classify can (a) decode just the
region header + entry table (a small prefix read, since `palette_blob`/`index_blob` come AFTER the entries in
the region layout — store the entry count in `VxoRegionDirEntry` so classify reads only the header+entries
prefix), or (b) conservatively return `Surface` (never prune) — correct but loses the cull. Default: bake the
`is_full` bit + read the entry-table prefix; absent region ⇒ `Air`. (Mirror `StaticVoxSource::classify`'s
clamped-coarse-LOD guard: return `Surface` for a coarse `lod` not present as a baked `LODS` level.)

### B2.6 `voxel_size` reconciliation (self-describing)

At open, compare `HEAD.voxel_size` to the live `brickmap::VOXEL_SIZE`. If equal ⇒ load directly. If the asset
was baked at a different size (e.g. a 0.05 m asset loaded by a still-0.2 m engine before the D1 flip), the
loader MUST either reject with a clear error ("asset baked at 0.05 m; engine is 0.2 m — rebake or flip") or
rescale. v1: **assert-equal + clear error** (the flip + re-bake is one atomic step, `SOTA_REFERENCE.md` §6 —
scenes load wrong-scaled between flip and re-bake, so a mismatch is a build error, not a silent rescale). This
is the "self-describing so a 0.05 m asset is unambiguous" guarantee (§0.4): the size is in the file, the loader
checks it, no guessing.

### B2.7 Halo: store-and-feed vs strip-and-re-halo

Two valid loader paths for the haloed `10³` body:

- **Strip-and-re-halo (default, simplest):** decode only the `8³` core into a `Brick`; let the existing
  `pack_one`/`pack_resident_set` re-build the halo from the resident set (the SAME path worldgen + static
  bricks take). Pro: the loaded `Brick` is exactly what every other source yields ⇒ the round-trip test
  compares `Brick`s directly; the packer's R1 uniform-incl-halo collapse + R3 dedup run identically. Con: the
  stored halo is redundant with the re-halo (wasted disk). **Mitigation:** the encoder MAY store only the `8³`
  core (`region body = core cells`, not haloed) and re-halo at load — but then a core brick on a region
  boundary needs its neighbour's core, which may be in a different (uncached) region. Default: **store the
  `8³` core only** (smaller disk, no cross-region coupling at decode because the re-halo happens in `pack_one`
  over the RESIDENT set, not at region-decode time — the resident neighbours are already streamed in by the
  clipmap). This is the cleanest: `.vxo` stores cores, the packer halos from residency exactly as today.
  Update §B1.3 accordingly: the per-brick body is the `8³` core (`BRICK_VOXELS = 512` cells), `encode_paletted`
  over the core. (The haloed-storage variant is the optimization to revisit only if `pack_one`'s re-halo shows
  up on the perf rig.)

> **Resolution:** v1 stores the **`8³` core** per brick (not the halo). The R2b encode is over the 512 core
> cells; the loader decodes the core → `Brick::from_voxels` → the residency + packer halo it identically to a
> worldgen/static brick. This keeps regions independently decodable AND the loaded brick bit-identical to a
> live brick (the round-trip gate). `decode_paletted_cell` is reused verbatim (the cell count differs, the
> decode math doesn't).

### B2.8 Acceptance gates

1. **Peak-RAM gate (the headline).** During a Bistro-Exterior `.vxo` load + a representative camera
   fly-through, **peak process RAM stays under a stated budget** — NOT the 2.6 GB the dense scene would be.
   Concretely: `RAM_peak < HEAD/MATL/BIDX (≈ tens of MB) + decoded_region_budget (≈256 MB) + resident-set VRAM
   mirror (≈ tens of MB after R1/R2/R3) ≈ < 512 MB`. Measured by extending `tests/voxel_worldgen_perf.rs` /
   `voxel_sponza_residency.rs` with a `.vxo` streaming stage that tracks the decoded-region cache bytes + the
   resident count, asserting the cache never exceeds its budget and the loader never holds all regions at once.
   "A 2.6 GB scene must never fully expand in RAM" (`VOXEL_PROGRAM.md` B2).
2. **Round-trip byte-identity.** A `.vxo` written by the offline encoder from a known `BrickMap`, then streamed
   back through `VxoSource`, yields **bit-identical `Brick`s** vs the live-generated set: for every `(coord)`,
   `VxoSource::brick(coord, 0)` `== StaticVoxSource::new(&original_map).brick(coord, 0)` (both `Brick`
   `PartialEq`). And the packed `GpuBrickPatch` from the streamed set is byte-identical to one packed from the
   original map (reuse the `incremental` A/B fingerprint helper, `incremental/tests.rs`). This proves the
   delivered property — the disk R2b core decodes to a bit-identical `Brick` whose re-packed layout is
   byte-identical (a decode-then-pack, §0.1; not a raw memcpy).
3. **`classify` parity.** `VxoSource::classify == StaticVoxSource::classify` for the same geometry (the
   surface-only cull is preserved), so the Θ(H²) residency win holds for `.vxo` scenes.
4. **Zero warnings, both feature builds** (the standing invariant) + the per-stage adversarial-review QA gate
   (`feedback-agent-team-qa-per-stage`).

---

## B3 — R6 SVDAG asset transport (the `BRIK` variant for immutable imports)

For a **static, never-edited** imported asset (a `.vox`/glTF scene baked once), the `BRIK` region bodies MAY
be DAG-encoded for a far bigger disk win (`VOXEL_STORAGE_PLAN.md` R5/§1.8, `SOTA_REFERENCE.md` §1.8 "ADOPTED as
asset transport"; Aokana's "store as SVDAG, decode to a traceable pool"). This is **Tier B only** — decoded per
region-chunk to the R2b brick form before any trace (`VOXEL_STORAGE_PLAN.md` §5: a DAG on the trace path is
forbidden). Gate: **non-edited assets only** (no COW hazard — `VOXEL_PROGRAM.md` B3).

### B3.1 Encoding (offline, in the encoder)

- The file `flags` bit1 marks "BRIK bodies are SVDAG-encoded." A region-chunk's body becomes a **small SVDAG**
  over that region's `K³` bricks' voxels (a sparse voxel octree with identical subtrees merged into a DAG —
  Kämpe 2013). Because a region is small (`K=8` ⇒ `8·8 = 64³` voxels max), the per-region DAG is shallow and
  fast to build + decode.
- The encoder builds the region's dense occupancy octree, **interns identical subtrees** (hash subtree → node
  id, the classic SVDAG subtree-merge), and serializes the node array + the leaf-brick palettes pointer-free
  (node children as region-local `u32` indices). The headline ratio (~0.12 bits/voxel for binary geometry,
  `VOXEL_STORAGE_PLAN.md` §1.9) lands here for the buried-interior mass.
- **Materials:** the SVDAG leaves still reference the per-region palette (the R2b `(palette, index)` form at
  the leaf), so block ids survive — the DAG merges GEOMETRY (occupancy + the dominant-block leaf), the palette
  rides alongside. (A region whose interior is one uniform block DAG-collapses to a single shared subtree —
  the same R1 win, expressed as a DAG node.)

### B3.2 Decoding (per region-chunk, at load)

- On a region miss, if `flags` bit1 is set: decompress (zstd over the DAG bytes) → **decode the region's
  SVDAG into the R2b `(VxoBrickEntry[], palette_blob, index_blob)` form** (the §B1.3 `DecodedRegion`), then
  proceed identically to §B2.2 step 5+. The DAG → bricks decode is a tree walk emitting each leaf brick's
  haloed/core cells; it runs ONCE per region miss and is cached (the LRU holds the DECODED `DecodedRegion`,
  not the DAG, so the DAG cost is paid once per residency).
- The trace NEVER sees the DAG — it sees the decoded R2b bricks in VRAM, exactly as for a non-DAG `.vxo`
  (`VOXEL_STORAGE_PLAN.md` §5).

### B3.3 Gating

- DAG encoding is opt-in per asset (an encoder flag `--svdag`), applied only to assets marked immutable (no
  in-scene destruction). An EDITED `.vxo` (one with per-instance edits / a destructible world) uses the plain
  R2b `BRIK` (§B1.3) so a cut re-packs a region without re-DAGing (COW-friendly). The runtime reader handles
  both via `flags` bit1 — same `DecodedRegion` output, different decode front-end. v1 ships the plain R2b
  `BRIK`; B3 adds the DAG front-end behind the flag once B1/B2 land (`VOXEL_PROGRAM.md`: B3 "after B1/B2").

---

## The offline encoder (extend `examples/voxelize_scene.rs`)

`.vox` becomes **import-only**; the canonical baked artifact is `.vxo`.

- **Today:** `voxelize_scene.rs` voxelizes a mesh → a `Grid` → `build_dot_vox` → `data.write_vox` →
  `assets/models/*.vox`. The runtime then `load_vox` → re-brick → `StaticVoxSource`.
- **Change:** after voxelizing to the in-RAM `BrickMap` (the same `bricks_from_placed` path, or directly from
  the `Grid`), add a `write_vxo(path, &brick_map, &registry, head_params)` step that emits the `.vxo`:
  1. Walk the `BrickMap` bricks, bucket them by region (`brick_coord.div_euclid(K)`).
  2. Per region: for each brick, `encode_paletted` its `8³` core (R1 uniform-collapse via
     `Brick::uniform_block`, else R2b dense), intern identical slices within the region (`VoxelInterner`),
     emit the `VxoBrickEntry` table + region-local `palette_blob`/`index_blob` + `VxoRegionHeader`. Bake the
     `is_full` bit (§B2.5).
  3. zstd-compress each region body (or `STORE`); record `(brik_offset, brik_comp_len, brik_raw_len,
     brick_count)` into the `BIDX` table; sort `BIDX` by `(z,y,x)`.
  4. Emit `HEAD` (voxel_size from the bake config, bounds/anchor from the `BrickMap`, K, counts), `MATL` from
     the `BlockRegistry` (`BlockDef` → `VxoMaterial`, linear colours straight through), `BIDX`, then the
     concatenated `BRIK` region bodies. Optionally `LITE`/`LODS`/`INST` (deferred).
  5. (B3) If `--svdag`, encode each region body as a DAG and set `flags` bit1.
- **Reuse the import improvements** (`VOXEL_PROGRAM.md` C2/C3, already in flight): the `.vox` MATL emissive
  reader, area-averaged albedo, always-on interior floodfill, CIELAB palette — these feed the `BrickMap` +
  `BlockRegistry` the encoder serializes, so a `.vxo` carries emissive + solid interiors + a >256 palette for
  free (the `MATL` `u16` cap-lift).
- CLI: `cargo run --example voxelize_scene -- <out.vxo> <voxel_metres> <in_mesh> <scale> [--svdag] [--store]`.
  Output extension `.vxo` selects the new writer; `.vox` keeps the legacy writer (interchange/debug only).

### Migration

- **Runtime depends only on the `.vxo` reader** (`src/voxel/vxo/`). `dot_vox` stays a DEV/offline dep
  (`examples/`, the import side) — drop it from the runtime `vox.rs` once all corpus scenes are re-baked to
  `.vxo`. `vox::load_vox` stays as an OFFLINE import primitive (it builds the `BrickMap` the encoder
  serializes) but is no longer on the shipped load path.
- The scene-load call site swaps `load_vox(path) + StaticVoxSource::new(&map)` for `VxoSource::open(path)` —
  both return `(impl BrickSource, BlockRegistry)`, so the `ResidencyManager` wiring is unchanged.
- Re-bake the corpus (`SOTA_REFERENCE.md` §6) to `.vxo`: Sponza/Sibenik/Conference/Bistro. Bistro-Exterior
  @0.05 m needs the tiled voxelizer (`VOXEL_PROGRAM.md` C1) to PRODUCE the `BrickMap` the encoder writes — the
  `.vxo` write side is bounded-RAM by construction (region-by-region streaming write), so the encoder can emit
  Bistro region-by-region from the tiled voxelizer's disk-backed tiles without a full-RAM expand.

---

## Open questions

1. **Region granularity K.** Default `K = 8` (a `512`-brick, ~3.2 m³ @0.05 m region). Too small ⇒ a fat `BIDX`
   + many tiny zstd frames (poor ratio); too large ⇒ a region decode over-reads vs. the shell's demand. Tune
   from the gallery fly-through (decoded-bytes-per-demanded-brick on the perf rig). Open: should K vary with
   LOD (coarse-LOD regions cover more world per brick)?
2. **Halo: core-only (chosen) vs stored-halo.** v1 stores the `8³` core and re-halos in `pack_one` (§B2.7). If
   the re-halo shows on the perf rig as a meaningful cost, revisit storing the haloed `10³` (bigger disk,
   region-local decode). Measure first.
3. **Cross-region dedup (R3).** v1 dedups WITHIN a region only (to keep regions independently decompressible).
   A global brick dictionary (shared slices across regions, decompressed once into a "dictionary region")
   would capture more interior repetition — at the cost of region independence. Defer; measure the intra-region
   dedup ratio first.
4. **`LITE`/`LODS` ship-vs-rebuild.** v1 rebuilds lights + coarse LODs in RAM at load (bounded by surface-only
   residency). Decide from the gallery-worst-case RAM measurement (B2 gate) whether streamed Bistro needs the
   pre-baked `LODS`/`LITE` chunks to stay under budget.
5. **zstd dependency split (RESOLVED, FIX 2).** Runtime DECODE uses pure-Rust `ruzstd` (no C toolchain); the C
   `zstd` crate (compress) is offline-encode-only behind the `vxo-encode` feature. The default library/runtime
   build pulls no `zstd-sys`. `STORE` mode remains the zstd-free fallback. (Open only: whether very-large-scene
   bake times want a faster compressor — irrelevant to the runtime.)
6. **B3 SVDAG node format.** The exact per-region DAG node encoding (child-mask + variable vs fixed `u32`
   pointers) is sketched, not pinned — pin it when B3 starts, validated against a region round-trip
   (DAG-encode → decode → bit-identical `DecodedRegion`).

---

## Summary

- **Chunks:** `magic "VXO1"` + RIFF-style tagged/length-prefixed/skippable chunks: **HEAD** (self-describing
  `voxel_size`/bounds/anchor/K/counts), **MATL** (`u16`-keyed linear-RGBA + emissive material table, NO 256
  cap), **BIDX** (the sorted region→`(brik_offset, comp_len, raw_len, compression, brick_count)` directory — the
  piece that makes region streaming possible), **BRIK** (per-region STORE/zstd blobs; each region =
  `VxoRegionHeader` + `VxoBrickEntry[]` + region-local `palette_blob`/`index_blob`, the **R2b core triple
  verbatim** so a read-back brick is bit-identical and its re-packed `GpuBrickPatch` byte-identical — a
  decode-then-pack, NOT a raw memcpy); optional **LITE**/**LODS**/**INST** (sketched, deferred).
- **BIDX design:** small eager-loaded sorted table (`48 B/region`, binary-searched by `(z,y,x)`), giving the
  exact `pread` span + decompress size per region — lazy per-region `BRIK` reads, never a whole-file expand.
- **Streamed loader:** `VxoSource` = mmap'd file + `BIDX` + a byte-budgeted decoded-region LRU, implementing
  `BrickSource` (`brick`/`classify`) so it feeds the EXISTING `ResidencyManager` demand path — ONE residency
  SSOT for worldgen + static + `.vxo`. Sync decode parallelized by the existing `drain_work_from`; async
  prefetch deferred behind a hitch gate. Merged gallery = per-asset coord offset + a concatenated remapped
  `MATL` (`MergedSource`).
- **B3:** an offline DAG-merged `BRIK` variant (`flags` bit1) for immutable assets, decoded per region-chunk
  to the same R2b `DecodedRegion` (Tier-B transport, never on the trace), gated to non-edited assets.
- **Encoder:** extend `voxelize_scene.rs` to `write_vxo` (region-bucketed, R1/R2b/R3-encoded, zstd'd); `.vox`
  becomes import-only; runtime depends only on the `.vxo` reader, `dot_vox` stays an offline dep.
- **Acceptance gates:** (1) peak RAM during a Bistro-Exterior load `< ~512 MB` (never the 2.6 GB dense
  expand); (2) `.vxo` round-trips **bit-identical `Brick`s** + a byte-identical packed `GpuBrickPatch` vs the
  live set; (3) `classify` parity preserves the surface-only Θ(H²) residency; (4) zero warnings + both feature
  builds + the per-stage adversarial QA gate.
