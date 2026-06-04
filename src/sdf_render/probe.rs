//! DDGI probe addressing — the CPU data model for the sparse, surface-anchored probe set.
//!
//! A **probe** is anchored one-per-occupied-SDF-brick (optionally a `subdiv³` sub-lattice
//! within a brick — see P5). Its identity is the brick's ABSOLUTE `(lod, brick_coord)` key,
//! anchored at world 0 and independent of the camera — exactly like [`super::chunk::ChunkKey`].
//! That is what makes DDGI here boil-free by construction: a surface cell maps to the same
//! probe every frame no matter where the camera is, so temporal history always aligns.
//!
//! This module owns ONLY the pure addressing math + slot allocation (unit-tested, no GPU).
//! The probe payload atlases, trace/blend passes, and the parallel probe-run buffer in
//! [`super::chunk::LiveChunkTables`] are wired in later phases.

use bevy::math::{IVec3, Vec3};
use rustc_hash::FxHashMap;

use super::atlas::BrickKey;
use super::SdfGridConfig;

/// Per-probe octahedral irradiance map resolution (`PROBE_OCT_RES²` texels stored flat in the
/// irradiance buffer). The apply samples it by the surface normal → directional GI. Mirrored in
/// `sdf/probe.wgsl`; guarded by `wgsl_probe_constants_match_rust`.
pub const PROBE_OCT_RES: u32 = 8;
pub const PROBE_OCT_TEXELS: u32 = PROBE_OCT_RES * PROBE_OCT_RES;

/// Octahedral irradiance tile edge in texels, INCLUDING the 1px wrap border (interior is
/// `PROBE_IRR_TILE - 2`). Mirrored in `sdf/probe.wgsl`; guarded by `wgsl_probe_constants_match_rust`.
pub const PROBE_IRR_TILE: u32 = 8;
/// Octahedral depth/visibility (Chebyshev moments) tile edge in texels, including border.
pub const PROBE_DEPTH_TILE: u32 = 16;
/// Interior (sampled) octahedral resolution of an irradiance tile.
pub const PROBE_IRR_INTERIOR: u32 = PROBE_IRR_TILE - 2;
/// Interior octahedral resolution of a depth tile.
pub const PROBE_DEPTH_INTERIOR: u32 = PROBE_DEPTH_TILE - 2;

/// Absolute probe identity: the occupied brick it lives in. (`subdiv` sub-probes within a brick
/// are addressed by the per-brick run block, not by a distinct key — see P5.) Equivalent data to
/// a [`BrickKey`]; a distinct type documents intent and keeps probe maps from mixing with brick maps.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ProbeKey {
    pub lod: u32,
    /// Stride-aligned brick coord (multiple of `cell_stride`), anchored at world 0.
    pub brick_coord: IVec3,
}

impl ProbeKey {
    pub fn new(lod: u32, brick_coord: IVec3) -> Self {
        Self { lod, brick_coord }
    }

    pub fn from_brick(b: BrickKey) -> Self {
        Self { lod: b.lod, brick_coord: b.coord }
    }

    pub fn to_brick(self) -> BrickKey {
        BrickKey::new(self.lod, self.brick_coord)
    }
}

/// World-space center of a brick (`subdiv == 1`: one probe per brick). The brick's stride-aligned
/// voxel coord maps to its world min via [`SdfGridConfig::brick_min_world`]; the center is half a
/// brick further on each axis.
pub fn probe_world_pos(key: ProbeKey, config: &SdfGridConfig) -> Vec3 {
    let min = config.brick_min_world(key.brick_coord, key.lod);
    min + Vec3::splat(0.5 * config.brick_world_size(key.lod))
}

/// World-space center of sub-probe `sub` (each component in `0..subdiv`) of a brick subdivided
/// `subdiv³`. `subdiv == 1` collapses to [`probe_world_pos`]. Used by the adaptive-subdivision
/// path (P5) so creases / small features get a denser probe lattice within a single brick.
pub fn subprobe_world_pos(key: ProbeKey, sub: IVec3, subdiv: u32, config: &SdfGridConfig) -> Vec3 {
    debug_assert!(subdiv >= 1);
    let bw = config.brick_world_size(key.lod);
    let cell = bw / subdiv as f32;
    let min = config.brick_min_world(key.brick_coord, key.lod);
    min + (sub.as_vec3() + Vec3::splat(0.5)) * cell
}

/// Number of probe slots a brick subdivided `subdiv³` occupies (contiguous block).
pub fn probes_per_brick(subdiv: u32) -> u32 {
    subdiv * subdiv * subdiv
}

/// Stable probe-block → compacted-slot allocator with a free-list, mirroring
/// [`super::chunk::ChunkSlotAllocator`]. A brick is assigned a contiguous block of `count`
/// slots (`count = probes_per_brick(subdiv)`); reusing freed blocks before growing `next`
/// keeps the probe atlas densely packed (bounded by peak resident probe count). For the common
/// `subdiv == 1` case `count == 1`, so this degenerates to the chunk allocator exactly.
///
/// Blocks are allocated from a per-size free-list so a freed block is only reused by a request of
/// the same size (no fragmentation bookkeeping). `subdiv` rarely exceeds a couple of values, so
/// the per-size lists stay tiny.
#[derive(Default)]
pub struct ProbeSlotAllocator {
    base_of: FxHashMap<ProbeKey, (u32, u32)>, // key -> (base_slot, count)
    free: FxHashMap<u32, Vec<u32>>,           // count -> freed base slots
    next: u32,
}

impl ProbeSlotAllocator {
    /// Assign (or return the existing) base slot for `key`'s `count`-slot block. Reuses a freed
    /// same-size block first. Returns the base slot; the block spans `base..base+count`.
    pub fn alloc(&mut self, key: ProbeKey, count: u32) -> u32 {
        debug_assert!(count >= 1);
        if let Some(&(base, c)) = self.base_of.get(&key) {
            debug_assert_eq!(c, count, "probe block resized without free (count {c} != {count})");
            return base;
        }
        let base = self
            .free
            .get_mut(&count)
            .and_then(|v| v.pop())
            .unwrap_or_else(|| {
                let b = self.next;
                self.next += count;
                b
            });
        self.base_of.insert(key, (base, count));
        base
    }

    /// Return `key`'s block to the free pool (brick evicted). Stale atlas tiles are harmless —
    /// no live probe references them once the block is free.
    pub fn release(&mut self, key: &ProbeKey) {
        if let Some((base, count)) = self.base_of.remove(key) {
            self.free.entry(count).or_default().push(base);
        }
    }

    /// The block `(base, count)` currently assigned to `key`, if resident.
    pub fn block(&self, key: &ProbeKey) -> Option<(u32, u32)> {
        self.base_of.get(key).copied()
    }

    /// One past the largest slot ever handed out → how many probe slots the atlas must span.
    pub fn high_water(&self) -> u32 {
        self.next
    }

    /// Count of resident probe blocks (bricks with probes).
    pub fn resident_blocks(&self) -> usize {
        self.base_of.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SdfGridConfig {
        SdfGridConfig::default()
    }

    /// A probe's world center round-trips back to its own brick at every LOD: this is the
    /// world-anchoring contract — the probe sits squarely inside the brick it identifies, so
    /// `world_to_brick_lod(center)` returns the brick coord regardless of camera/frame.
    #[test]
    fn probe_center_roundtrips_to_brick() {
        let cfg = config();
        let s = cfg.cell_stride();
        for lod in 0..cfg.lod_count {
            for (bx, by, bz) in [(0, 0, 0), (1, 0, 0), (3, 2, 1), (-1, -1, -1), (-5, 2, -9)] {
                let coord = IVec3::new(bx * s, by * s, bz * s);
                let key = ProbeKey::new(lod, coord);
                let c = probe_world_pos(key, &cfg);
                assert_eq!(
                    cfg.world_to_brick_lod(c, lod),
                    coord,
                    "lod {lod} brick {coord:?}: center {c:?} fell outside its brick"
                );
            }
        }
    }

    /// The center is exactly half a brick from the min corner on every axis (so sub-probe math
    /// with `subdiv == 1` collapses to it).
    #[test]
    fn probe_center_is_brick_midpoint() {
        let cfg = config();
        let key = ProbeKey::new(2, IVec3::new(14, 0, -7));
        let min = cfg.brick_min_world(key.brick_coord, key.lod);
        let half = 0.5 * cfg.brick_world_size(key.lod);
        let c = probe_world_pos(key, &cfg);
        assert!((c - (min + Vec3::splat(half))).length() < 1e-6);
        // And subprobe(subdiv=1, sub=0) is the same point.
        let sp = subprobe_world_pos(key, IVec3::ZERO, 1, &cfg);
        assert!((sp - c).length() < 1e-6);
    }

    /// A `subdiv²`³ lattice stays inside the brick and is symmetric about the center.
    #[test]
    fn subprobes_fill_brick_symmetrically() {
        let cfg = config();
        let key = ProbeKey::new(0, IVec3::new(0, 0, 0));
        let bw = cfg.brick_world_size(0);
        let min = cfg.brick_min_world(key.brick_coord, key.lod);
        let subdiv = 2u32;
        let mut centroid = Vec3::ZERO;
        let n = probes_per_brick(subdiv);
        assert_eq!(n, 8);
        for z in 0..subdiv {
            for y in 0..subdiv {
                for x in 0..subdiv {
                    let p = subprobe_world_pos(key, IVec3::new(x as i32, y as i32, z as i32), subdiv, &cfg);
                    for a in 0..3 {
                        assert!(p[a] >= min[a] - 1e-5 && p[a] <= min[a] + bw + 1e-5);
                    }
                    centroid += p;
                }
            }
        }
        centroid /= n as f32;
        let expect = min + Vec3::splat(0.5 * bw);
        assert!((centroid - expect).length() < 1e-5, "sub-probe centroid {centroid:?} != brick center {expect:?}");
    }

    /// The allocator hands out a contiguous block per brick, reuses a freed same-size block before
    /// growing, and keeps the high-water bounded by peak residency — the sparsity guarantee.
    #[test]
    fn allocator_blocks_and_reuses() {
        let mut a = ProbeSlotAllocator::default();
        let k0 = ProbeKey::new(0, IVec3::new(0, 0, 0));
        let k1 = ProbeKey::new(0, IVec3::new(7, 0, 0));
        let k2 = ProbeKey::new(0, IVec3::new(14, 0, 0));

        // subdiv=1 → 1 slot each.
        assert_eq!(a.alloc(k0, 1), 0);
        assert_eq!(a.alloc(k1, 1), 1);
        assert_eq!(a.high_water(), 2);
        // Idempotent.
        assert_eq!(a.alloc(k0, 1), 0);

        // Free k0, then a new key reuses slot 0 (no growth).
        a.release(&k0);
        assert_eq!(a.alloc(k2, 1), 0, "freed slot should be reused");
        assert_eq!(a.high_water(), 2, "reuse must not grow the water mark");

        // A subdiv=2 block (8 slots) is contiguous and grows past the water mark.
        let kb = ProbeKey::new(1, IVec3::new(0, 0, 0));
        let base = a.alloc(kb, 8);
        assert_eq!(base, 2);
        assert_eq!(a.high_water(), 10);
        assert_eq!(a.block(&kb), Some((2, 8)));

        // Freeing the 8-block and re-requesting 8 reuses it; requesting 1 does NOT (different size).
        a.release(&kb);
        let kc = ProbeKey::new(1, IVec3::new(7, 0, 0));
        assert_eq!(a.alloc(kc, 8), 2, "same-size block reused");
        let kd = ProbeKey::new(1, IVec3::new(14, 0, 0));
        assert_eq!(a.alloc(kd, 1), 10, "different-size request grows rather than splitting the 8-block");
    }

    /// Probe identity is purely a function of `(lod, brick_coord)` — never of any camera state.
    /// Two `ProbeKey`s built for the same brick are equal and hash equal, so the same world cell
    /// always maps to the same probe slot across frames (the no-boil invariant at the data level).
    #[test]
    fn probe_key_is_camera_independent() {
        use std::collections::HashSet;
        let s = config().cell_stride();
        let coord = IVec3::new(-3 * s, 2 * s, 5 * s);
        let a = ProbeKey::new(1, coord);
        let b = ProbeKey::from_brick(BrickKey::new(1, coord));
        assert_eq!(a, b);
        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b), "equal probe keys must hash-collide (stable identity)");
        assert_eq!(a.to_brick(), BrickKey::new(1, coord));
    }

    /// The octahedral tile constants are hand-mirrored in `sdf/probe.wgsl` (WGSL can't import Rust
    /// consts). A silent mismatch mis-sizes the probe atlas vs. the shader's sampling footprint —
    /// corrupt irradiance. Pin both to the Rust source of truth, like `wgsl_chunk_constants_match_rust`.
    #[test]
    fn wgsl_probe_constants_match_rust() {
        let src = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/shaders/sdf/probe.wgsl"
        ))
        .expect("read sdf/probe.wgsl");

        let int_after = |pat: &str| -> i64 {
            let i = src.find(pat).unwrap_or_else(|| panic!("probe.wgsl missing `{pat}`"));
            let tail = &src[i + pat.len()..];
            let digits: String = tail
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            digits.parse().unwrap_or_else(|_| panic!("no integer after `{pat}` in probe.wgsl"))
        };

        assert_eq!(
            int_after("const PROBE_OCT_RES: u32 ="),
            PROBE_OCT_RES as i64,
            "WGSL PROBE_OCT_RES != Rust"
        );
        // PROBE_OCT_TEXELS is a hand-written literal in the WGSL (not computed), so pin it too — a
        // stale value here mis-indexes every probe tile.
        assert_eq!(
            int_after("const PROBE_OCT_TEXELS: u32 ="),
            PROBE_OCT_TEXELS as i64,
            "WGSL PROBE_OCT_TEXELS != Rust"
        );
        assert_eq!(
            int_after("const PROBE_IRR_TILE: u32 ="),
            PROBE_IRR_TILE as i64,
            "WGSL PROBE_IRR_TILE != Rust"
        );
        assert_eq!(
            int_after("const PROBE_DEPTH_TILE: u32 ="),
            PROBE_DEPTH_TILE as i64,
            "WGSL PROBE_DEPTH_TILE != Rust"
        );
        // The probe-trace key decode mirrors `chunk::KEY_BIAS`; pin the WGSL copy to it.
        assert_eq!(
            int_after("const PROBE_KEY_BIAS: i32 ="),
            crate::sdf_render::chunk::KEY_BIAS as i64,
            "WGSL PROBE_KEY_BIAS != Rust chunk::KEY_BIAS"
        );
    }
}
