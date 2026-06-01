//! Chunk addressing for the clipmap atlas.
//!
//! A **chunk** groups `CHUNK_BRICKS³ = 64` bricks into the clipmap's addressing,
//! bake-batch, and debug unit. Brick GPU lookup is done per *chunk* (a ~64× smaller
//! table than per-brick) keyed by an **absolute** world-lattice chunk coord that never
//! references the camera/ring origin — so the CPU-built lookup and the GPU shader agree
//! by construction, regardless of where the camera is. This is what fixes the
//! "objects shift / world disappears" bugs: those came from per-brick ids computed
//! relative to a camera-moving ring origin.
//!
//! Within a chunk, bricks are **sparse**: only non-empty bricks get atlas tiles. A
//! 64-bit occupancy mask records which of the 64 local slots are present; the GPU tests
//! one bit and `countOneBits` gives the offset into that chunk's packed tile run.
//!
//! A LOD-`L` chunk holds the same 64 bricks as a LOD-0 chunk but each brick is `2^L`
//! larger, so it covers `2^L`× the world — the nested-shell clipmap structure.
//!
//! This module owns ONLY the coordinate math + table layout (pure, unit-tested). The
//! per-brick texel storage (`atlas::TileAllocator`) and incremental upload are unchanged.

use bevy::math::IVec3;

use super::SdfGridConfig;
use super::atlas::{BrickKey, SdfAtlas};

/// Bricks per axis in one chunk. 64 = `4³` fits a single u64 occupancy mask.
pub const CHUNK_BRICKS: i32 = 4;
/// Brick slots in one chunk (`CHUNK_BRICKS³`).
pub const CHUNK_VOLUME: u32 = (CHUNK_BRICKS * CHUNK_BRICKS * CHUNK_BRICKS) as u32; // 64
/// Bias added to each signed chunk-axis index so it fits an unsigned 16-bit key field.
/// ±32768 chunks/axis — at LOD0 (chunk ≈ 2.8 m) that's ±90 km, ample for a several-km
/// world; coarser LODs reach exponentially further. Mirrored verbatim in
/// `bindings.wgsl::abs_chunk_key`; the `wgsl_chunk_constants_match_rust` test guards
/// against silent drift (a mismatch reintroduces the camera-shift / blank-world bug).
pub const KEY_BIAS: i32 = 1 << 15;

/// Absolute chunk identity: LOD level + chunk coord on that level's chunk lattice
/// (anchored at world 0, independent of the camera). The GPU key is derived from this.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ChunkKey {
    pub lod: u32,
    /// Chunk coord = brick_index.div_euclid(CHUNK_BRICKS), per axis.
    pub coord: IVec3,
}

impl ChunkKey {
    pub fn new(lod: u32, coord: IVec3) -> Self {
        Self { lod, coord }
    }
}

/// The chunk a brick belongs to, and the brick's local slot (0..63) within it.
pub fn chunk_of(brick: BrickKey, config: &SdfGridConfig) -> (ChunkKey, u32) {
    let s = config.cell_stride();
    // Brick index on the LOD lattice (stride-aligned coord → contiguous index).
    let bi = IVec3::new(
        brick.coord.x.div_euclid(s),
        brick.coord.y.div_euclid(s),
        brick.coord.z.div_euclid(s),
    );
    let cc = IVec3::new(
        bi.x.div_euclid(CHUNK_BRICKS),
        bi.y.div_euclid(CHUNK_BRICKS),
        bi.z.div_euclid(CHUNK_BRICKS),
    );
    let local = IVec3::new(
        bi.x.rem_euclid(CHUNK_BRICKS),
        bi.y.rem_euclid(CHUNK_BRICKS),
        bi.z.rem_euclid(CHUNK_BRICKS),
    );
    let idx = (local.z * CHUNK_BRICKS * CHUNK_BRICKS + local.y * CHUNK_BRICKS + local.x) as u32;
    (ChunkKey::new(brick.lod, cc), idx)
}

/// The absolute 64-bit GPU key for a chunk, packed lexicographically so a sort /
/// binary-search by `(key_hi, key_lo)` orders by lod, then x, y, z. Mirrored exactly by
/// `abs_chunk_key` in `bindings.wgsl`.
pub fn chunk_gpu_key(key: ChunkKey) -> (u32, u32) {
    let cx = ((key.coord.x + KEY_BIAS) as u32) & 0xffff;
    let cy = ((key.coord.y + KEY_BIAS) as u32) & 0xffff;
    let cz = ((key.coord.z + KEY_BIAS) as u32) & 0xffff;
    let key_hi = (key.lod << 16) | cx;
    let key_lo = (cy << 16) | cz;
    (key_hi, key_lo)
}

/// World-space minimum corner of a chunk (its brick-(0,0,0) corner).
pub fn chunk_min_world(key: ChunkKey, config: &SdfGridConfig) -> bevy::math::Vec3 {
    let vs = config.voxel_size_at(key.lod);
    let bricks_per_chunk_world = config.cell_stride() as f32 * vs * CHUNK_BRICKS as f32;
    bevy::math::Vec3::new(
        key.coord.x as f32,
        key.coord.y as f32,
        key.coord.z as f32,
    ) * bricks_per_chunk_world
}

/// World-space edge length of a whole chunk at `lod`.
pub fn chunk_world_size(lod: u32, config: &SdfGridConfig) -> f32 {
    config.cell_stride() as f32 * config.voxel_size_at(lod) * CHUNK_BRICKS as f32
}

/// One entry in the GPU chunk lookup table (sorted by `(key_hi, key_lo)`, binary-
/// searched by the shader). 5×u32 = 20 bytes. `occ_lo|occ_hi` is the 64-bit occupancy
/// mask (bit `i` set ⇒ local brick `i` is resident); `tile_run_base` indexes the packed
/// `tile_run` table where this chunk's `popcount(mask)` brick `atlas_base`s live in
/// ascending local-index order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChunkLookup {
    pub key_hi: u32,
    pub key_lo: u32,
    pub occ_lo: u32,
    pub occ_hi: u32,
    pub tile_run_base: u32,
}

/// One resident brick's GPU record inside a chunk's tile run: its atlas tile origin plus
/// its packed 4-entry material palette (`pal01 = id0|id1<<16`, `pal23 = id2|id3<<16`).
/// 3×u32 = 12 bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BrickTile {
    pub atlas_base: u32,
    pub pal01: u32,
    pub pal23: u32,
}

/// The two GPU buffers the shader needs to resolve a brick: the sorted chunk table and
/// the packed per-chunk brick runs. Built from the resident brick set each upload.
#[derive(Default)]
pub struct ChunkTables {
    pub chunks: Vec<ChunkLookup>,
    /// Per resident brick, grouped by chunk (chunk `c` occupies
    /// `tile_run[c.tile_run_base .. + popcount(c.occ)]`), in ascending local-index order.
    pub tile_run: Vec<BrickTile>,
}

/// Group an atlas's resident bricks into the sorted chunk table + packed tile-run table.
/// `tile_of(key)` returns the brick's [`BrickTile`] (atlas origin + packed palette).
/// Pure aside from the closure; lives here so addressing + table layout are one unit and
/// independently testable. Cost is O(bricks log bricks), same order as the old per-brick
/// lookup build, just grouped.
pub fn build_chunk_tables(
    atlas: &SdfAtlas,
    config: &SdfGridConfig,
    mut tile_of: impl FnMut(&BrickKey) -> BrickTile,
) -> ChunkTables {
    use std::collections::HashMap;

    // Gather per chunk: (local_index, brick tile) for each resident brick.
    let mut by_chunk: HashMap<ChunkKey, Vec<(u32, BrickTile)>> = HashMap::new();
    for key in atlas.bricks.keys() {
        let (ck, local) = chunk_of(*key, config);
        by_chunk.entry(ck).or_default().push((local, tile_of(key)));
    }

    // Stable order: sort chunks by GPU key so the shader can binary-search.
    let mut chunk_keys: Vec<ChunkKey> = by_chunk.keys().copied().collect();
    chunk_keys.sort_by_key(|k| chunk_gpu_key(*k));

    let mut tables = ChunkTables::default();
    for ck in chunk_keys {
        let mut bricks = by_chunk.remove(&ck).unwrap();
        bricks.sort_by_key(|(local, _)| *local);

        let mut occ: u64 = 0;
        let tile_run_base = tables.tile_run.len() as u32;
        for (local, tile) in &bricks {
            occ |= 1u64 << *local;
            tables.tile_run.push(*tile);
        }
        let (key_hi, key_lo) = chunk_gpu_key(ck);
        tables.chunks.push(ChunkLookup {
            key_hi,
            key_lo,
            occ_lo: occ as u32,
            occ_hi: (occ >> 32) as u32,
            tile_run_base,
        });
    }
    tables
}

/// The distinct non-empty chunks an atlas currently has resident — for the debug
/// overlay (one wireframe box per chunk).
pub fn resident_chunks(atlas: &SdfAtlas, config: &SdfGridConfig) -> Vec<ChunkKey> {
    use std::collections::HashSet;
    let mut set: HashSet<ChunkKey> = HashSet::new();
    for key in atlas.bricks.keys() {
        set.insert(chunk_of(*key, config).0);
    }
    set.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SdfGridConfig {
        SdfGridConfig::default()
    }

    /// chunk_of maps a brick to a chunk coord + local slot, and local round-trips into
    /// the 0..63 range with the documented packing.
    #[test]
    fn chunk_of_local_index_in_range_and_roundtrips() {
        let cfg = config();
        let s = cfg.cell_stride();
        for bz in 0..CHUNK_BRICKS {
            for by in 0..CHUNK_BRICKS {
                for bx in 0..CHUNK_BRICKS {
                    // Brick at chunk (0,0,0), local (bx,by,bz).
                    let coord = IVec3::new(bx * s, by * s, bz * s);
                    let (ck, local) = chunk_of(BrickKey::new(0, coord), &cfg);
                    assert_eq!(ck.coord, IVec3::ZERO);
                    let expect = (bz * CHUNK_BRICKS * CHUNK_BRICKS + by * CHUNK_BRICKS + bx) as u32;
                    assert_eq!(local, expect);
                    assert!(local < CHUNK_VOLUME);
                }
            }
        }
    }

    /// Negative brick coords land in the chunk below (div_euclid), not chunk 0.
    #[test]
    fn negative_coords_use_euclidean_chunk() {
        let cfg = config();
        let s = cfg.cell_stride();
        // One brick left of the origin → brick index -1 → chunk -1, local CHUNK_BRICKS-1.
        let (ck, local) = chunk_of(BrickKey::new(0, IVec3::new(-s, 0, 0)), &cfg);
        assert_eq!(ck.coord.x, -1);
        assert_eq!(local % CHUNK_BRICKS as u32, (CHUNK_BRICKS - 1) as u32);
    }

    /// The GPU key is order-preserving: sorting by (key_hi,key_lo) orders by lod, x, y, z
    /// — required for the shader's binary search.
    #[test]
    fn gpu_key_is_order_preserving() {
        let mut keys = vec![
            ChunkKey::new(0, IVec3::new(0, 0, 0)),
            ChunkKey::new(0, IVec3::new(0, 0, 1)),
            ChunkKey::new(0, IVec3::new(0, 1, 0)),
            ChunkKey::new(0, IVec3::new(1, 0, 0)),
            ChunkKey::new(0, IVec3::new(-1, 0, 0)),
            ChunkKey::new(1, IVec3::new(-5, -5, -5)),
        ];
        let mut by_packed = keys.clone();
        by_packed.sort_by_key(|k| chunk_gpu_key(*k));
        // Expected lexicographic order on (lod, x, y, z), with x,y,z biased ascending.
        keys.sort_by_key(|k| (k.lod, k.coord.x, k.coord.y, k.coord.z));
        assert_eq!(by_packed, keys);
    }

    /// Distinct (lod,coord) within range never collide on the packed key.
    #[test]
    fn gpu_key_no_collision_in_range() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for lod in 0..4u32 {
            for x in -3..=3 {
                for y in -3..=3 {
                    for z in -3..=3 {
                        let k = chunk_gpu_key(ChunkKey::new(lod, IVec3::new(x, y, z)));
                        assert!(seen.insert(k), "collision at lod={lod} ({x},{y},{z})");
                    }
                }
            }
        }
    }

    /// The chunk-addressing constants are hand-duplicated in `bindings.wgsl`
    /// (`abs_chunk_key` / `local_brick_index`) because WGSL can't import Rust consts.
    /// A silent mismatch there makes the GPU search a different key than the CPU stored
    /// → the camera-shift / blank-world bug class this clipmap rework fixed. This test
    /// parses the shader and pins both constants to the Rust source of truth, so any
    /// future edit to one side without the other fails CI instead of shipping a
    /// hard-to-trace visual corruption.
    #[test]
    fn wgsl_chunk_constants_match_rust() {
        let src = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/shaders/sdf/bindings.wgsl"
        ))
        .expect("read bindings.wgsl");

        // Helper: find `pat` and parse the integer literal that follows it.
        let int_after = |pat: &str| -> i64 {
            let i = src
                .find(pat)
                .unwrap_or_else(|| panic!("bindings.wgsl missing `{pat}`"));
            let tail = &src[i + pat.len()..];
            let digits: String = tail
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            digits
                .parse()
                .unwrap_or_else(|_| panic!("no integer after `{pat}` in bindings.wgsl"))
        };

        // `const CHUNK_BRICKS: i32 = 4;`
        let wgsl_chunk_bricks = int_after("const CHUNK_BRICKS: i32 =");
        assert_eq!(
            wgsl_chunk_bricks, CHUNK_BRICKS as i64,
            "WGSL CHUNK_BRICKS ({wgsl_chunk_bricks}) != Rust chunk::CHUNK_BRICKS ({CHUNK_BRICKS})"
        );

        // `let bias = 32768;` inside abs_chunk_key — must equal Rust KEY_BIAS.
        let wgsl_bias = int_after("let bias =");
        assert_eq!(
            wgsl_bias, KEY_BIAS as i64,
            "WGSL chunk key bias ({wgsl_bias}) != Rust chunk::KEY_BIAS ({KEY_BIAS})"
        );
    }

    // --- Chunk-table build ↔ shader-resolve round-trip ------------------------------

    use super::super::atlas::PackedBrick;

    /// Mirror EXACTLY what `brick.wgsl::find_brick_lookup` does on the GPU: binary-search
    /// the sorted chunk table by absolute key, test the occupancy bit for the brick's
    /// local slot, and (if set) index the tile run at `tile_run_base + popcount(bits
    /// strictly below the slot)`. Returns the resolved `BrickTile`, or `None` if not
    /// resident. Keeping this in lockstep with the shader is the point of the test below.
    fn shader_resolve(
        tables: &ChunkTables,
        config: &SdfGridConfig,
        brick: BrickKey,
    ) -> Option<BrickTile> {
        let (ck, li) = chunk_of(brick, config); // li = local slot 0..63
        let (key_hi, key_lo) = chunk_gpu_key(ck);
        let idx = tables
            .chunks
            .binary_search_by(|c| (c.key_hi, c.key_lo).cmp(&(key_hi, key_lo)))
            .ok()?;
        let chunk = tables.chunks[idx];
        let occ = (chunk.occ_lo as u64) | ((chunk.occ_hi as u64) << 32);
        if (occ >> li) & 1 == 0 {
            return None; // brick not resident in this chunk
        }
        let below = occ & ((1u64 << li) - 1); // bits strictly below the slot
        let off = below.count_ones();
        Some(tables.tile_run[(chunk.tile_run_base + off) as usize])
    }

    fn dummy_brick() -> PackedBrick {
        use crate::sdf_render::edits::{PALETTE_EMPTY, PALETTE_K};
        PackedBrick {
            palette: [PALETTE_EMPTY; PALETTE_K],
            baked_hash: 0,
        }
    }

    /// End-to-end CPU↔GPU contract: bricks scattered across several chunks and LODs must
    /// each resolve — via the shader's occupancy-mask + popcount-offset unpack — back to
    /// the exact tile `build_chunk_tables` assigned them, and a brick that isn't resident
    /// must miss. A packing bug here silently maps a brick to the wrong tile (the visual
    /// corruption class the chunked rework fixed), so this is the key regression guard.
    #[test]
    fn build_chunk_tables_resolves_each_brick_to_its_tile() {
        let cfg = config();
        let s = cfg.cell_stride();
        let c = CHUNK_BRICKS;

        // Encode each brick's identity into a unique tile so a wrong-tile mapping shows.
        let tile_of = |k: &BrickKey| -> BrickTile {
            let base = (k.lod << 28)
                ^ ((k.coord.x as u32) << 16)
                ^ ((k.coord.y as u32) << 8)
                ^ (k.coord.z as u32);
            BrickTile { atlas_base: base, pal01: base ^ 0x1111, pal23: base ^ 0x2222 }
        };

        // Bricks across: a sparse subset of slots in chunk (0,0,0), a neighbouring chunk,
        // and a negative-coord chunk at lod 1.
        let mut atlas = SdfAtlas::default();
        let mut keys = Vec::new();
        for (lx, ly, lz) in [(0, 0, 0), (1, 0, 0), (3, 2, 1)] {
            keys.push(BrickKey::new(0, IVec3::new(lx * s, ly * s, lz * s)));
        }
        keys.push(BrickKey::new(0, IVec3::new(c * s, 0, 0))); // chunk (+x), local 0
        keys.push(BrickKey::new(1, IVec3::new(-s, -s, -s))); // lod1, chunk (-1,-1,-1)
        for k in &keys {
            atlas.bricks.insert(*k, dummy_brick());
        }

        let tables = build_chunk_tables(&atlas, &cfg, tile_of);

        assert!(
            tables
                .chunks
                .windows(2)
                .all(|w| (w[0].key_hi, w[0].key_lo) <= (w[1].key_hi, w[1].key_lo)),
            "chunk table must be sorted by gpu key (binary-searchable)"
        );
        assert_eq!(
            tables.tile_run.len(),
            keys.len(),
            "tile_run holds exactly one entry per resident brick"
        );

        for k in &keys {
            let got = shader_resolve(&tables, &cfg, *k)
                .unwrap_or_else(|| panic!("brick {k:?} failed to resolve"));
            assert_eq!(got, tile_of(k), "brick {k:?} resolved to the wrong tile");
        }

        // Unoccupied slot in a resident chunk must miss (not alias a neighbour's tile).
        let absent = BrickKey::new(0, IVec3::new(2 * s, 2 * s, 2 * s));
        assert!(
            shader_resolve(&tables, &cfg, absent).is_none(),
            "an unoccupied slot in a resident chunk must not resolve"
        );

        // A brick in a chunk that isn't resident at all must miss.
        let no_chunk = BrickKey::new(0, IVec3::new(50 * c * s, 0, 0));
        assert!(
            shader_resolve(&tables, &cfg, no_chunk).is_none(),
            "a brick in an absent chunk must not resolve"
        );
    }

    // --- Chunk world geometry (debug-viz boxes + LOD-shell convention) --------------

    /// A LOD-`L` chunk covers exactly 2× the world extent of LOD `L-1` — the nested
    /// "twice as coarse / twice the area" shell property the clipmap is built on.
    #[test]
    fn chunk_world_size_doubles_per_lod() {
        let cfg = config();
        for lod in 1..cfg.lod_count {
            let coarse = chunk_world_size(lod, &cfg);
            let fine = chunk_world_size(lod - 1, &cfg);
            assert!(
                (coarse - 2.0 * fine).abs() < 1e-4,
                "lod {lod} chunk ({coarse}) must be 2x lod {} ({fine})",
                lod - 1
            );
        }
        // Anchor the absolute scale: a LOD-0 chunk spans cell_stride·voxel·CHUNK_BRICKS.
        let expect0 = cfg.cell_stride() as f32 * cfg.voxel_size_at(0) * CHUNK_BRICKS as f32;
        assert!((chunk_world_size(0, &cfg) - expect0).abs() < 1e-6);
    }

    /// The world point → chunk mapping is geometrically self-consistent: the chunk a
    /// point resolves to (`chunk_of(world_to_brick_lod(p))`) has a world box that
    /// actually encloses `p` on every axis: `min ≤ p < min + size`. A drift between the
    /// addressing math and the debug-viz geometry would break this.
    #[test]
    fn chunk_box_contains_its_world_point() {
        use bevy::math::Vec3;
        let cfg = config();
        for lod in 0..cfg.lod_count {
            let size = chunk_world_size(lod, &cfg);
            for p in [
                Vec3::ZERO,
                Vec3::new(0.05, 0.05, 0.05),
                Vec3::new(13.7, -4.2, 88.1),
                Vec3::new(-260.0, 30.0, -9.0),
            ] {
                let brick = cfg.world_to_brick_lod(p, lod);
                let (ck, _) = chunk_of(BrickKey::new(lod, brick), &cfg);
                let min = chunk_min_world(ck, &cfg);
                let max = min + Vec3::splat(size);
                assert!(
                    p.x >= min.x && p.x < max.x
                        && p.y >= min.y && p.y < max.y
                        && p.z >= min.z && p.z < max.z,
                    "lod {lod}: point {p:?} not in its chunk box [{min:?}, {max:?})"
                );
            }
        }
    }

    /// Adjacent chunks tile exactly — the next chunk's min corner is one full chunk
    /// further on, with no gap or overlap (so the debug overlay reads as a clean grid).
    #[test]
    fn adjacent_chunks_tile_without_gaps() {
        let cfg = config();
        for lod in 0..cfg.lod_count {
            let size = chunk_world_size(lod, &cfg);
            let base = ChunkKey::new(lod, IVec3::new(2, -1, 0));
            let min = chunk_min_world(base, &cfg);
            for (axis, delta) in [
                (0, IVec3::X),
                (1, IVec3::Y),
                (2, IVec3::Z),
            ] {
                let next = chunk_min_world(ChunkKey::new(lod, base.coord + delta), &cfg);
                let step = next - min;
                // Only the stepped axis advances, by exactly one chunk world size.
                for a in 0..3 {
                    let want = if a == axis { size } else { 0.0 };
                    assert!(
                        (step[a] - want).abs() < 1e-4,
                        "lod {lod} axis {axis}: neighbour offset[{a}]={} want {want}",
                        step[a]
                    );
                }
            }
        }
    }
}
