# Testing — fast default suite + the explicit perf/validation harnesses

This is the map of what `cargo test` runs by default, what it deliberately does **not**, and the
exact command to invoke every perf / validation / long-running harness that lives behind
`#[ignore]`.

The guiding rule (project invariant + the agent mandate): **the default `cargo test` is FAST and
covers correctness completely.** Anything that only *measures* (a perf/timing/throughput/storage
bench) or that is *long-running without being a unique correctness gate* is `#[ignore]`d and run
explicitly. Every such harness carries an `#[ignore = "…"]` reason that includes its run command,
and is also tabulated below.

---

## 1. The fast default suite — `cargo test`

```sh
cargo test          # everything not #[ignore]: lib unit tests + integration correctness gates
cargo test --lib    # just the in-crate (src/**) unit tests — the quickest green signal (~7 s)
```

- `cargo test --lib` runs **298 unit tests in ~7 s** (4 ignored — see the lib-ignored list below).
- The integration crate (`tests/*.rs`) adds the cross-plugin correctness gates: the WGSL/naga
  validation rigs, the worldgen determinism-parity gates, the streaming/residency CPU bookkeeping
  tests, the `chunk_lookup_bench` *adversarial property* tests, and the GPU oracle rigs. The GPU
  rigs **skip cleanly** (pass, no failure) on a box without an `EXPERIMENTAL_RAY_QUERY` Vulkan
  adapter, so the default suite is green on CPU-only machines too.

### Env caveat for GPU tests — `TMP` / `TEMP`

The headless GPU rigs require the temp dir to be on `D:` (a C:-temp issue on this box). Set both
before running anything that touches the GPU:

```powershell
$env:TMP = "D:\tmp_test"; $env:TEMP = "D:\tmp_test"
cargo test
```

```sh
# bash (this is what the agent uses):
TMP=D:/tmp_test TEMP=D:/tmp_test cargo test
```

**Exception — do NOT redirect temp for these:** the `soul_scene` and `assets` file-round-trip lib
tests need the **default** temp dir (pointing them at `D:\tmp_test` makes them fail with OS error
123). In practice this means: run the GPU/integration tests with `TMP/TEMP=D:\tmp_test`, but run the
plain `cargo test --lib` (which includes the soul_scene/assets round-trip tests) with the default
temp. If you must run the whole thing in one shot, the soul_scene/assets round-trip tests are the
ones to watch.

### Issue #134 (DLSS default feature) — FIXED

`dlss` is a *default* feature (`default = ["fast", "physics", "dlss"]` in `Cargo.toml`). With it on,
`bevy/dlss`'s `DlssInitPlugin` (inside `DefaultPlugins`) panics at device-init time —
`"DlssProjectId was not added to the App before DlssInitPlugin"` — unless a `DlssProjectId` resource
is inserted first (the real app does this in `src/main.rs`). Previously the full-App GPU rigs that did
NOT insert it (`voxel_cornell_headless`, `voxel_temporal_gpu`) failed the default `cargo test`.

**Fixed:** all three full-App rigs (`voxel_cornell_headless`, `voxel_temporal_gpu`,
`voxel_render_headless`) now boot through the shared `common::HeadlessRender` harness, whose
constructor (`HeadlessRender::new`) ALWAYS inserts the test `DlssProjectId` before `DefaultPlugins`
and forces the readback-capturable non-DLSS path. Because that is the only way to build the headless
render app, the panic is structurally impossible to reintroduce. **`cargo test` now passes under the
default feature set with no special flags.**

---

## 2. Ignored harnesses — what each measures + how to run it

All commands assume the repo root as cwd. The GPU ones need `TMP/TEMP=D:\tmp_test` (shown inline)
and a ray-query adapter; they skip cleanly with no device. Add `--release` only where noted (the
bench wants optimized timings).

### Integration crate (`tests/*.rs`)

| File::fn | What it measures | Run command | Ballpark / output |
|---|---|---|---|
| `voxel_worldgen_perf.rs::bench_voxelize_brick_cost` | per-brick `voxelize_brick` cost over the real shipping worldgen surface (the inner grain of `drain_work`) | `cargo test --test voxel_worldgen_perf bench_voxelize_brick_cost -- --ignored --nocapture` | prints µs/brick + dense/uniform/empty split |
| `voxel_worldgen_perf.rs::bench_initial_fill_cold` | cold-stream the whole clipmap from empty (the V-toggle-into-worldgen hitch): `update` + bounded `drain_work` + amortized `pack` | `cargo test --test voxel_worldgen_perf bench_initial_fill_cold -- --ignored --nocapture` | prints frames-to-settle, per-frame drain p50/p95/max, wall fill time |
| `voxel_worldgen_perf.rs::bench_steady_state_moving` | per-step cost as the camera walks/jumps a warm clipmap (the in-flight hitch) | `cargo test --test voxel_worldgen_perf bench_steady_state_moving -- --ignored --nocapture` | per-step update/drain/pack stats |
| `voxel_worldgen_perf.rs::bench_pack_at_resident_count` | `pack_resident_set` cost at the 60k-class resident count (the SSOT GPU-buffer rebuild) | `cargo test --test voxel_worldgen_perf bench_pack_at_resident_count -- --ignored --nocapture` | ms/pack + per-buffer MB |
| `voxel_worldgen_perf.rs::bench_blas_build_at_resident_count` | **GPU** — full BLAS rebuild from scratch at resident count: buffer uploads + `create_blas`/`create_tlas` + `build_acceleration_structures` | `TMP=D:\tmp_test TEMP=D:\tmp_test cargo test --test voxel_worldgen_perf bench_blas_build_at_resident_count -- --ignored --nocapture` | upload/create/build ms (skips w/o ray-query device) |
| `voxel_worldgen_perf.rs::bench_per_chunk_blas_rebuild_vs_monolithic` | **GPU** — per-chunk (banded) BLAS rebuild vs the monolithic rebuild; proves O(changed chunks) | `TMP=D:\tmp_test TEMP=D:\tmp_test cargo test --test voxel_worldgen_perf bench_per_chunk_blas_rebuild_vs_monolithic -- --ignored --nocapture` | mono vs per-chunk ms + speedup× |
| `voxel_worldgen_perf.rs::bench_incremental_repack_vs_full` | incremental `ResidentPacker::update` (O(changed) delta) vs full `pack_resident_set` (O(resident)); voxelizes the shipping clipmap | `cargo test --test voxel_worldgen_perf bench_incremental_repack_vs_full -- --ignored --nocapture` | incr vs full ms, changed-slot count, delta KB. **Asserts** the O(changed) win. |
| `voxel_worldgen_perf.rs::clipmap_per_move_cost` | per-single-brick-move streaming stutter (update+drain, O(shell)) vs the cold fill; voxelizes the shipping clipmap | `cargo test --test voxel_worldgen_perf clipmap_per_move_cost -- --ignored --nocapture` | stream per-move ms + reduction×. **Asserts** stutter < ½ cold fill. |
| `voxel_worldgen_perf.rs::report_storage_bytes_sponza` | storage-plan R1 VRAM BEFORE/AFTER on the baked Sponza (needs `assets/models/sponza.vox`) | `cargo test --test voxel_worldgen_perf report_storage_bytes_sponza -- --ignored --nocapture` | resident VRAM before/after + reduction× (skips if asset absent) |
| `voxel_worldgen_perf.rs::report_storage_bytes_worldgen_slice` | storage-plan R1 VRAM on the full shipping worldgen clipmap (60k cap); voxelizes the real clipmap | `cargo test --test voxel_worldgen_perf report_storage_bytes_worldgen_slice -- --ignored --nocapture` | resident VRAM before/after + reduction× |
| `worldgen_bench.rs::bench_layer_manager_cold_fill` | `LayerManager::update` cold-filling the resident height window (focus-jump / scene-load hot path) | `cargo test --features editor --test worldgen_bench bench_layer_manager_cold_fill -- --ignored --nocapture` | ms + chunks + ms/chunk |
| `worldgen_bench.rs::bench_build_height_ring` | `build_height_ring` over a full resident store (rebuilt every worldgen delta) | `cargo test --features editor --test worldgen_bench bench_build_height_ring -- --ignored --nocapture` | ms/build |
| `worldgen_bench.rs::bench_height_layer_generate_per_chunk` | per-chunk `HeightLayer::generate` (the fBm fill grain) | `cargo test --features editor --test worldgen_bench bench_height_layer_generate_per_chunk -- --ignored --nocapture` | ms/chunk |
| `chunk_lookup_bench.rs::bench_chunk_lookup_structures` | head-to-head profiling of 3 GPU chunk-lookup structures over a production fly-path (the 448 ms-spike investigation) | `CARGO_INCREMENTAL=0 cargo test --test chunk_lookup_bench --release bench_chunk_lookup_structures -- --ignored --nocapture` | per-structure max-mutate/lookup ns/mem table |
| `chunk_lookup_bench.rs::structures_agree_on_lookups` | cross-check the 3 candidate structures resolve identically (a correctness check for the profiling rig; `#[ignore]` because it pairs with the bench) | `cargo test --test chunk_lookup_bench structures_agree_on_lookups -- --ignored --nocapture` | prints agreement note |
| `voxel_sponza_pack.rs::sponza_loads_and_packs_non_empty` | **long-running (~115 s)** asset-integrity check: load + `pack_brickmap` the full 16 MB shipped `sponza.vox` (33k bricks, 7.5M cells) | `TMP=D:\tmp_test TEMP=D:\tmp_test cargo test --test voxel_sponza_pack sponza_loads_and_packs_non_empty -- --ignored --nocapture` | `sponza pack: 33591 bricks, 7472628 voxel cells, 257 palette entries` |
| `worldgen_parity.rs::print_reference_vectors` | **generator, not a bench** — prints the pinned determinism reference literals for paste when the height layer's output intentionally changes | `cargo test --features editor --test worldgen_parity print_reference_vectors -- --ignored --nocapture` | prints Rust literal tables (never writes a file) |

### Standalone example (`examples/*.rs`)

| Example | What it measures | Run command | Ballpark / output |
|---|---|---|---|
| `d1c_scaling.rs` | **D1c de-risk + D1d shell-first re-measure** — 64 m@0.05 m reach/perf scaling at the PRODUCTION `StreamingConfig::default()` (clip_half 160). A FAST single-pass version of the `voxel_worldgen_perf` benches. Reports BOTH the OLD cube `desired_clipmap` (section A: enumeration-ceiling truncation, LOD0-only, the 38 s classify) AND the NEW D1d shell-first `desired_clipmap_surface` (section A': every LOD enumerates, ms enumerate) so the fix is visible side-by-side; plus the per-LOD distribution, classify split, single-`update` wall (now D1d shell-first), steady-state resident count, full-pack wall, and A4.4 resident VRAM. NB: it still runs the slow OLD cube baseline + full pack for the side-by-side, so it takes minutes — use `d1d_shellfirst.rs` for a quick shell-first-only number. | `TMP=D:/tmp_test TEMP=D:/tmp_test cargo run --release --no-default-features --features fast,physics --example d1c_scaling` | prints the D1c/D1d table (cube path still hits-ceiling 8 M / LOD0-only; D1d shell-first enumerates all 8 LODs, update drops 38 s → ~3 s; ≈ 143 k resident / 40.5 MB VRAM) |
| `d1d_shellfirst.rs` | **D1d shell-first re-measure (fast)** — times ONLY the NEW `desired_clipmap_surface` + a single cold `ResidencyManager::update` at clip_half 160, SKIPPING the slow cube baseline + full pack `d1c_scaling` runs, so the shell-first number is seconds not minutes. Reports the per-LOD candidate distribution (all 8 LODs now present) + the cold-update wall. | `cargo run --release --no-default-features --features fast,physics --example d1d_shellfirst` | ≈ 6.7 M candidates / all 8 LODs / cold update ~3 s (was 38 s on the cube path) |

### Offline voxelizer — the tiled out-of-core bake (C1) + the Bistro deliverable

The `voxelize_scene` example bakes a classic mesh scene into a `.vox`/`.vxo`. The default (monolithic) path
needs the whole AABB resident; the **`--tiled`** path (Phase C1, `docs/TILED_VOXELIZER_PLAN.md`) is the
bounded-RAM out-of-core voxelizer — disk-backed tiles + a union-find enclosure flood + a streaming `.vxo`
write — required for huge scenes (Bistro-Exterior @0.05 m, a ~13 B-cell AABB). Its CPU gates run in the
default suite (the example's `#[cfg(test)]`): the cell-for-cell oracle vs the monolithic `solid_fill`,
determinism across `TILE_EDGE`, the bounded-RAM probe, and the no-regression single-tile cases. They are fast
and need no asset.

**The Bistro `.vxo` bake (the long, asset-gated deliverable run).** `assets/models/bistro.vxo` is baked from
the gitignored qian-o Bistro-Exterior glTF via the tiled path. It is a multi-minute, multi-GB-disk run (NOT a
default-suite test — it produces a shipped asset). The exact step:

```sh
# Bistro-Exterior @0.05 m → assets/models/bistro.vxo via the tiled out-of-core voxelizer.
# ~5 min wall-clock; scratch high-water ~5.2 GiB on D:\ (deleted on success); peak RAM < 4 GiB.
TMP=D:/tmp_test TEMP=D:/tmp_test cargo run --release --example voxelize_scene --features vxo-encode -- \
  assets/models/bistro.vxo 0.05 \
  assets/models/src/_gltfassets/Bistro/BistroExterior.gltf 1.0 \
  --tiled --tile-edge 128 --scratch D:/tmp_test
```

Reported on the reference run: grid `3479×1025×3692` = 13.17 B AABB cells, **354 M solid voxels**, ~159 s
flood+fill + ~124 s streamed assembly, 83 MB `.vxo` (1.21 M bricks / 8061 regions, bounds anchored y=0). The
`--scratch` dir holds the disk tiles + the streamed BRIK temp; it is deleted on success, LEFT + logged on
failure. After baking, validate the artifact (asset-gated, `#[ignore]`d):

```sh
cargo test --example voxelize_scene --features vxo-encode -- --ignored bistro_vxo_parses --nocapture
```

(The gallery's `Bistro` row in `GALLERY_SCENES` prefers this region-streamed `.vxo`; the legacy full-RAM
`.vox` is too large to load.) The same `--tiled` path bakes any scene — drop `--tiled --scratch …` onto the
normal positional CLI; without `--tiled` the monolithic path is unchanged.

### Lib crate (`src/**`, `#[cfg(test)]`)

These four are `#[ignore]`d in the lib build (`cargo test --lib` reports `4 ignored`):

| Module::fn | What it is | Run command |
|---|---|---|
| `assets/tests.rs::export_demo_materials` | one-shot exporter (writes demo material assets), not a test gate | `cargo test --lib export_demo_materials -- --ignored --nocapture` |
| `sdf_render/worldgen/layers/height.rs::bench_analytic_vs_fd_gradient` | gen-perf microbench: analytic stored-gradient normals vs the old 5-tap finite-difference path | `cargo test --release --lib bench_analytic_vs_fd_gradient -- --ignored --nocapture` |
| `sdf_render/worldgen/layers/height.rs::bench_generate_chunk` | gen-perf bench: per-chunk `generate` cost + `sample_world` call count (band-limit not a regression) | `cargo test --release --lib bench_generate_chunk -- --ignored --nocapture` |
| `sdf_render/worldgen/graph/asset.rs::print_preset_graphs_ron` | prints the shipped preset graphs as RON (asset-authoring helper) | `cargo test --lib print_preset_graphs_ron -- --ignored --nocapture` |

> Note: `src/editor/worldgen_graph/tests.rs::write_world_biome_assets` is also `#[ignore]`d but only
> compiles under `--features editor`; it is an asset-writing helper, not a bench. Run with
> `cargo test --features editor write_world_biome_assets -- --ignored --nocapture`.

---

## 3. What deliberately stays in the default path (correctness gates — NOT ignored)

For the record, so a future pass doesn't mistakenly ignore them:

- **WGSL/naga validation** — `shader_validation.rs`, `worldgen_codegen.rs` (compose every shipped
  graph + the library) — fast, no GPU.
- **Worldgen determinism parity** — `worldgen_parity.rs` (the bit-identity hash/height reference
  vectors; the CI gate for shared-seed multiplayer). The `print_reference_vectors` *generator* in
  the same file is the only ignored member.
- **GPU oracle rigs** — `voxel_raytrace_gpu`, `voxel_render_headless`, `voxel_cornell_headless`
  (the three full-App rigs share `common::HeadlessRender`; #134 fixed), `voxel_gi_gpu`,
  `voxel_restir_gi_gpu`, `voxel_world_cache_gpu`, `voxel_lighting_gpu`,
  `voxel_temporal_gpu`, `voxel_seam_gpu`, `voxel_seam_oblique_gpu`, `voxel_dlss_guides_gpu`,
  `voxel_normal_swap`, `voxel_show_through`, `voxel_primitive_offset`, `worldgen_gpu_parity`,
  `voxel_edit`. These build small **synthetic** scenes and assert GPU-vs-CPU ground truth; they skip
  cleanly without a ray-query device and run in well under a second each when one is present.
- **CPU streaming / residency bookkeeping** — `voxel_streaming.rs`, `voxel_sponza_residency.rs`,
  `voxel_gallery_residency.rs`, and the non-ignored members of `voxel_worldgen_perf.rs`
  (`clipmap_view_distance`, `solid_building_storage_collapses`, `worldgen_stack_is_non_empty`) —
  small synthetic clipmaps, fast.
- **Adversarial property tests** — `chunk_lookup_bench.rs`'s `adversarial_*` tests (prove each design
  blocker is real and that the fix closes it). Fast; only the two profiling members are ignored.
- **Sponza SSOT** — `voxel_sponza_pack.rs::sponza_is_the_default_scene` (instant; pins the default
  boot scene). Only the slow `sponza_loads_and_packs_non_empty` sibling is ignored.

---

## 4. Pre-handoff gate (mirrors CI)

```sh
cargo build --tests                              # all tests compile (default features)
cargo clippy --all-targets                       # zero warnings (project invariant)
cargo build --features editor                    # the editor feature config builds
TMP=D:/tmp_test TEMP=D:/tmp_test cargo test --lib  # fast green correctness signal (~7 s)
```

The full GPU integration suite needs a ray-query adapter and `TMP/TEMP=D:\tmp_test`; it is not part
of the quick gate but **must compile** (`cargo build --tests` covers that).
