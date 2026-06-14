//! **Headless load + pack oracle for the static Sponza GI-measurement scene** (no GPU required).
//!
//! Sponza is a baked `.vox` scene loaded + packed EXACTLY like the Cornell box (fully resident, NOT streamed):
//! `load_vox(SPONZA_VOX_PATH) -> (BrickMap, BlockRegistry)`, then `pack_brickmap(&map, &registry)` — the same
//! packer the Cornell static path uses — into the SSOT `GpuBrickPatch` the renderer's BLAS is built from.
//!
//! This rig proves the CPU half of that path with no GPU: that the baked asset loads, packs into a non-empty
//! patch whose per-brick buffers line up, and carries a POPULATED palette (so the renderer has real colours to
//! shade + bounce). The GPU rigs (`voxel_raytrace_gpu`, `voxel_gi_gpu`, `voxel_render_headless`,
//! `voxel_cornell_headless`) cover the trace/GI/composite; this one is a fast, device-free build assert that
//! the Sponza scene is wired correctly. It ALSO pins Sponza as the default boot scene (the user requirement).
//!
//! If the baked `assets/models/sponza.vox` is absent (a checkout that hasn't run the offline bake) the
//! load/pack body is skipped (the test passes vacuously on that part), but the default-scene assert always
//! runs — mirroring `vox::tests::sponza_loads_if_present`.

use adventure::voxel::VoxelScene;
use adventure::voxel::gpu::pack_brickmap;
use adventure::voxel::raytrace::SPONZA_VOX_PATH;
use adventure::voxel::vox::load_vox;

/// Sponza is the DEFAULT boot scene (the user: "Sponza is the default on load"). The `#[default]` on the
/// `VoxelScene` enum is the single SSOT for the boot scene; assert it here so a future re-default is caught.
#[test]
fn sponza_is_the_default_scene() {
    assert_eq!(
        VoxelScene::default(),
        VoxelScene::Sponza,
        "Sponza must be the default boot scene"
    );
    // It is a STATIC scene (not the streaming worldgen path) — the residency packs it once like Cornell.
    assert!(!VoxelScene::Sponza.is_worldgen(), "Sponza must NOT route through the streaming worldgen path");
    assert!(!VoxelScene::Sponza.is_cornell(), "Sponza is its own static scene, distinct from Cornell");
}

/// The baked Sponza `.vox` loads and packs into a NON-EMPTY `GpuBrickPatch` with a POPULATED palette, via the
/// SAME `load_vox` → `pack_brickmap` path the live `stream_voxel_rt_residency` Sponza branch runs (mirroring
/// the Cornell fully-resident pack, NOT streaming). No GPU needed — this is the CPU load/pack assert.
#[test]
fn sponza_loads_and_packs_non_empty() {
    let path = std::path::Path::new(SPONZA_VOX_PATH);
    if !path.exists() {
        eprintln!("{SPONZA_VOX_PATH} not baked in this checkout — skipping sponza_loads_and_packs_non_empty");
        return; // asset not produced yet (run `cargo run --example voxelize_scene`)
    }

    // 1. Load the baked scene: the pure `path -> (BrickMap, BlockRegistry)` loader.
    let (map, registry) = load_vox(path).expect("sponza.vox must load");
    assert!(!map.is_empty(), "Sponza must have solid bricks");
    assert!(registry.len() > 1, "Sponza registry must carry the .vox palette (more than just AIR)");

    // 2. Pack EXACTLY as the static Cornell path does — `pack_brickmap`, no streaming.
    let patch = pack_brickmap(&map, &registry);

    // 3. The packed patch is non-empty and internally consistent (the SSOT GPU layout).
    assert!(!patch.is_empty(), "the packed Sponza patch must have resident bricks");
    let bricks = patch.brick_count();
    assert!(bricks > 0, "Sponza must pack at least one brick");
    assert_eq!(bricks, patch.metas.len(), "one meta per brick (parallel buffers)");
    assert_eq!(bricks, patch.aabbs.len(), "one AABB per brick (the BLAS primitive count)");
    assert!(!patch.voxels.is_empty(), "the packed voxel buffer must be non-empty");

    // 4. The PALETTE is populated (AIR + the .vox colours) AND carries at least one non-AIR, non-black colour
    //    — the renderer needs real albedo to shade + bounce. (`from_vox_palette` linearizes the sRGB palette;
    //    a populated atrium has plenty of non-black stone/drape colours.)
    assert_eq!(patch.palette.len(), registry.len(), "palette length must match the registry");
    let lit_color = patch
        .palette
        .iter()
        .skip(1) // skip AIR (block 0)
        .any(|c| c.rgba[0] + c.rgba[1] + c.rgba[2] > 0.0);
    assert!(lit_color, "Sponza palette must have at least one non-black colour for shading/GI");

    eprintln!(
        "sponza pack: {bricks} bricks, {} voxel cells, {} palette entries",
        patch.voxels.len(),
        patch.palette.len()
    );
}
