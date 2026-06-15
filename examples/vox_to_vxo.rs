//! Offline `.vox` → `.vxo` TRANSCODER — **Phase C / Migration** (`docs/VXO_FORMAT.md` §Migration).
//!
//! This does NOT re-voxelize from a mesh (that is `examples/voxelize_scene.rs`). It TRANSCODES an existing
//! baked MagicaVoxel `.vox` grid into the engine-native region-streamed `.vxo`:
//!
//! ```text
//!   load_vox(in.vox) → (BrickMap, BlockRegistry) → write_vxo(out.vxo, &map, &registry, VxoHeadParams { .. })
//! ```
//!
//! Both halves are the engine's own SSOT: [`adventure::voxel::vox::load_vox`] is the runtime `.vox` loader,
//! and [`adventure::voxel::vxo::write_vxo`] is the Phase B-i encoder (region-bucket + R1/R2b/R3 + per-region
//! STORE/zstd). So the `.vxo` carries EXACTLY what the engine loads from the `.vox` today — a pure transcode,
//! no C2/C3 import improvements (emissive / CIELAB palette / flood-fill) — those are a later RE-BAKE.
//!
//! CLI: `vox_to_vxo <in.vox> <out.vxo> [voxel_metres=0.05] [--store]`
//!   - `voxel_metres` (default **0.05**) is stamped into `HEAD.voxel_size` so the `.vxo` is self-describing
//!     (`docs/VXO_FORMAT.md` §0.4). The corpus's `.vox` grids ARE 0.05 m-dense, so 0.05 is the true sampling
//!     density — the transcoder does not resample, it only RECORDS the spacing the grid was baked at.
//!   - `--store` writes uncompressed region bodies; the default is per-region zstd-19 (`docs/VXO_FORMAT.md`
//!     §B1.9). zstd needs the C compressor, so this example is gated `required-features = ["vxo-encode"]`.
//!
//! RUN (the corpus, the THREE small scenes — Bistro is DEFERRED, see below):
//! ```sh
//!   cargo run --release --example vox_to_vxo --features vxo-encode -- assets/models/sponza.vox     assets/models/sponza.vxo
//!   cargo run --release --example vox_to_vxo --features vxo-encode -- assets/models/sibenik.vox    assets/models/sibenik.vxo
//!   cargo run --release --example vox_to_vxo --features vxo-encode -- assets/models/conference.vox assets/models/conference.vxo
//! ```
//!
//! **Bistro is intentionally NOT transcoded by this tool.** `assets/models/bistro.vox` is ~1.4 GB; its
//! `load_vox` → in-RAM `BrickMap` → `write_vxo` (which builds the WHOLE `.vxo` byte image in RAM, see
//! `writer::encode_vxo`) would very likely OOM. Bistro is deferred to Phase C1's BOUNDED-RAM, region-by-region
//! streaming write. This transcoder will happily attempt any file you give it, but the corpus conversion
//! deliberately skips Bistro.
//!
//! **Loadability note (the D1 flip):** the produced `.vxo` are stamped 0.05 m, but `brickmap::VOXEL_SIZE` is
//! still 0.2 m today, and `VxoSource::open` asserts `head.voxel_size == VOXEL_SIZE`. So these `.vxo` will only
//! LOAD through the streamed `VxoSource` AFTER the D1 `VOXEL_SIZE` 0.2 → 0.05 flip. That is EXPECTED — do not
//! change `VOXEL_SIZE` or the assert here. The full-file `VxoFile` reader has no such assert (the validation
//! test round-trips through `VxoFile`), and the legacy `.vox` load path stays live until D1.

use std::path::PathBuf;
use std::time::Instant;

use adventure::voxel::vox::load_vox;
use adventure::voxel::vxo::{VxoCompression, VxoHeadParams, write_vxo};

/// The corpus's true LOD0 voxel spacing in metres (every `.vox` in `assets/models/` is baked at 0.05 m). Used
/// when the CLI omits the 3rd arg; recorded verbatim in `HEAD.voxel_size` (the transcoder does NOT resample).
const DEFAULT_VOXEL_METRES: f32 = 0.05;

fn main() -> anyhow::Result<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let store = raw.iter().any(|a| a == "--store");
    let mut pos = raw.iter().filter(|a| !a.starts_with("--"));

    let in_path = pos
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: vox_to_vxo <in.vox> <out.vxo> [voxel_metres=0.05] [--store]"))?;
    let out_path = pos
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: vox_to_vxo <in.vox> <out.vxo> [voxel_metres=0.05] [--store]"))?;
    let voxel_metres: f32 = pos.next().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_VOXEL_METRES);

    anyhow::ensure!(voxel_metres > 0.0, "voxel_metres must be positive (got {voxel_metres})");

    // 1. Load the existing baked `.vox` grid into the engine's `(BrickMap, BlockRegistry)` SSOT.
    let t_load = Instant::now();
    let (map, registry) = load_vox(&in_path)?;
    let in_size = std::fs::metadata(&in_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "loaded {} ({:.1} MB): {} bricks, {} blocks ({:.2}s)",
        in_path.display(),
        in_size as f64 / 1_048_576.0,
        map.len(),
        registry.len(),
        t_load.elapsed().as_secs_f32()
    );

    // 2. Transcode → `.vxo` (reuses the B-i region-bucket / R1 / R2b / R3 encode — NO re-encoding here). The
    //    `voxel_metres` is RECORDED in HEAD (self-describing); the grid itself is copied brick-for-brick.
    let comp = if store { VxoCompression::Store } else { VxoCompression::default() };
    let name = out_path.file_stem().and_then(|s| s.to_str()).unwrap_or("vxo").to_string();
    let params = VxoHeadParams { voxel_size: voxel_metres, name, ..Default::default() };

    let t_write = Instant::now();
    write_vxo(&out_path, &map, &registry, &params, comp)?;
    let out_size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);

    let pct = if in_size > 0 { 100.0 * out_size as f64 / in_size as f64 } else { 0.0 };
    println!(
        "wrote {} ({:.1} MB, {}, voxel_size {voxel_metres} m) — {:.1}% of the {:.1} MB .vox ({:.2}s)",
        out_path.display(),
        out_size as f64 / 1_048_576.0,
        if store { "STORE" } else { "zstd-19" },
        pct,
        in_size as f64 / 1_048_576.0,
        t_write.elapsed().as_secs_f32()
    );
    Ok(())
}
