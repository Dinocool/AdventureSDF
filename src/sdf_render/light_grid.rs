//! Sparse world-wide light grid for clustered point-light culling.
//!
//! The Stage-1 G-buffer loop iterates EVERY uploaded point light per pixel, which doesn't scale to
//! the thousands of lights a per-object stress scene wants. This bins each light's range-sphere
//! into a coarse, world-anchored grid so a surface point (or, later, a DDGI probe-ray hit) only
//! iterates the lights in its own world cell.
//!
//! Unlike the clipmap chunk grid (`super::chunk`), this is rebuilt from scratch every frame (point
//! lights are dynamic — worst case all move each frame, so there is no slowly-churning resident set
//! to delta-upload). It is **sparse + world-wide**: only cells that actually contain a light get a
//! record, keyed by an order-preserving 64-bit cell key and **sorted** so the GPU binary-searches
//! it — the same pattern as the chunk directory (`chunk::chunk_gpu_key` + the sorted lookup). This
//! covers the whole world (no camera window) at a cost proportional to occupied cells, so a small
//! scene uploads almost nothing while a 3000-light field still scales.
//!
//! Key packing + the cell-size constant are mirrored in `assets/shaders/sdf/lights.wgsl` and pinned
//! by `wgsl_light_grid_constants_match_rust` (a CPU/GPU desync silently mis-bins lights).

use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;
use rustc_hash::FxHashMap;

use super::render::GpuPointLight;

/// World size (metres) of one light-grid cell. With no camera window the dominant cost is the
/// binning fan-out + directory size, so this is coarser than a windowed grid would need: a
/// range-10 light fans into ~3³ cells, keeping the directory + per-cell runs small. Pinned to WGSL.
pub const LIGHT_CELL_SIZE: f32 = 8.0;
/// Bias added to each signed cell axis so it packs into an unsigned 16-bit field — same convention
/// as `chunk::KEY_BIAS`. Reach: `±32768 cells × LIGHT_CELL_SIZE = ±262 km`. Pinned to WGSL.
pub const LIGHT_KEY_BIAS: i32 = 1 << 15;
/// Hard cap on lights binned into a single cell; the importance sort keeps the brightest, so the
/// dropped tail is the least significant. Bounds both the index buffer and per-pixel loop length.
pub const MAX_LIGHTS_PER_CELL: u32 = 256;
/// Defensive bound on a single light's cell-AABB span per axis, so a pathologically huge `range`
/// can't bin into millions of cells and stall the rebuild. 32 cells × 8 m = 256 m radius — far past
/// any sane point light; a larger light is simply truncated (it was never going to be performant).
const MAX_CELL_SPAN: i32 = 32;

/// One occupied cell's directory record: its world-cell key (for the GPU binary search) + where its
/// light-index run lives in the flat index buffer. 16 bytes (std430 stride 16, 4×u32 — naturally
/// aligned, like `GpuChunkLookup`). Mirrored as `LightCell` in `lights.wgsl`.
#[derive(ShaderType, Clone, Copy, Default, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuLightCell {
    /// `light_gpu_key().0` — biased x in the low 16 bits (high 16 bits unused, kept 0).
    pub key_hi: u32,
    /// `light_gpu_key().1` — `(biased_y << 16) | biased_z`.
    pub key_lo: u32,
    /// Start offset into the flat light-index buffer.
    pub base: u32,
    /// Number of light indices for this cell.
    pub count: u32,
}

/// The built sparse light grid: a key-sorted directory (`cells`) + a flat index buffer of light
/// indices (into the uploaded `GpuPointLight` array). Buffers + scratch are reused across frames
/// (`rebuild` clears + refills) to avoid per-frame churn.
#[derive(Default)]
pub struct LightGrid {
    /// Occupied cells, SORTED ascending by `(key_hi, key_lo)` so the shader can binary-search.
    pub cells: Vec<GpuLightCell>,
    /// Packed per-cell runs of light indices (cell `i`'s run is `cells[i].base .. +count`).
    pub index_buf: Vec<u32>,
    // --- reused scratch (per-frame, cleared in rebuild) ---
    /// Discovery slot for each occupied cell key (insertion order, pre-sort).
    key_to_slot: FxHashMap<(u32, u32), u32>,
    /// Cell keys in discovery order (index = discovery slot).
    keys: Vec<(u32, u32)>,
    /// Per-discovery-slot light count (capped at `MAX_LIGHTS_PER_CELL`).
    counts: Vec<u32>,
    /// discovery slot → sorted position in `cells` (built during prefix-sum).
    sorted_slot_of: Vec<u32>,
    /// Sort permutation of discovery slots by key.
    order: Vec<u32>,
    /// Per-sorted-cell write cursor for the scatter pass.
    cursors: Vec<u32>,
}

/// Absolute world cell of a point — `floor(p / LIGHT_CELL_SIZE)`. MUST match the WGSL
/// `floor(world_pos / LIGHT_CELL_SIZE)` (same float floor) so CPU binning and GPU lookup agree.
#[inline]
fn cell_of(p: Vec3) -> IVec3 {
    (p / LIGHT_CELL_SIZE).floor().as_ivec3()
}

/// Pack a world cell coord into the order-preserving 64-bit key — mirrors `chunk::chunk_gpu_key`
/// (biased 16-bit axis fields, no LOD). Sorting `(key_hi, key_lo)` orders by `(x, y, z)`. MUST
/// byte-match the WGSL `light_cell_key`.
#[inline]
fn light_gpu_key(cell: IVec3) -> (u32, u32) {
    let cx = ((cell.x + LIGHT_KEY_BIAS) as u32) & 0xffff;
    let cy = ((cell.y + LIGHT_KEY_BIAS) as u32) & 0xffff;
    let cz = ((cell.z + LIGHT_KEY_BIAS) as u32) & 0xffff;
    (cx, (cy << 16) | cz)
}

/// True if the sphere `(center, radius)` intersects the world AABB `[min, max]` — tighter than a
/// box-vs-box test (avoids binning a light into corner cells its sphere doesn't actually reach).
#[inline]
fn sphere_hits_aabb(center: Vec3, radius: f32, min: Vec3, max: Vec3) -> bool {
    let closest = center.clamp(min, max);
    closest.distance_squared(center) <= radius * radius
}

/// Perceptual brightness proxy for the importance sort (Rec. 709 luma of the candela radiance).
#[inline]
fn brightness(light: &GpuPointLight) -> f32 {
    let c = light.color_radius.truncate();
    c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722
}

impl LightGrid {
    /// Rebuild the sparse grid for `lights`, world-wide (no camera window). Two-pass counting sort
    /// over the OCCUPIED cell set: discover cells + per-cell counts, sort the occupied keys,
    /// prefix-sum into a key-sorted directory, scatter light indices, then importance-sort each run
    /// (brightest first, so the shader can shadow just the brightest few).
    pub(crate) fn rebuild(&mut self, lights: &[GpuPointLight]) {
        self.cells.clear();
        self.index_buf.clear();
        self.key_to_slot.clear();
        self.keys.clear();
        self.counts.clear();

        // --- Pass 0: discover occupied cells + per-cell counts (capped) ---
        for light in lights {
            self.for_overlapped_cells(light, |grid, cell| {
                let key = light_gpu_key(cell);
                let slot = if let Some(&s) = grid.key_to_slot.get(&key) {
                    s
                } else {
                    let s = grid.keys.len() as u32;
                    grid.keys.push(key);
                    grid.counts.push(0);
                    grid.key_to_slot.insert(key, s);
                    s
                };
                let c = &mut grid.counts[slot as usize];
                if *c < MAX_LIGHTS_PER_CELL {
                    *c += 1;
                }
            });
        }

        // --- Sort occupied cells by key, prefix-sum into the directory ---
        self.order.clear();
        self.order.extend(0..self.keys.len() as u32);
        let keys = &self.keys;
        self.order.sort_unstable_by_key(|&i| keys[i as usize]);
        self.sorted_slot_of.clear();
        self.sorted_slot_of.resize(self.keys.len(), 0);
        let mut total = 0u32;
        for (sorted_pos, &old_slot) in self.order.iter().enumerate() {
            let (key_hi, key_lo) = self.keys[old_slot as usize];
            let count = self.counts[old_slot as usize];
            self.cells.push(GpuLightCell { key_hi, key_lo, base: total, count });
            self.sorted_slot_of[old_slot as usize] = sorted_pos as u32;
            total += count;
        }
        self.index_buf.resize(total as usize, 0);

        // --- Pass 1: scatter light indices into each cell's run ---
        self.cursors.clear();
        self.cursors.extend(self.cells.iter().map(|c| c.base));
        for (li, light) in lights.iter().enumerate() {
            self.for_overlapped_cells(light, |grid, cell| {
                let key = light_gpu_key(cell);
                let sorted = grid.sorted_slot_of[grid.key_to_slot[&key] as usize] as usize;
                let end = grid.cells[sorted].base + grid.cells[sorted].count;
                let cur = &mut grid.cursors[sorted];
                if *cur < end {
                    grid.index_buf[*cur as usize] = li as u32;
                    *cur += 1;
                }
            });
        }

        // --- Importance sort: brightest first within each cell run (for the shadow cap) ---
        for cell in &self.cells {
            if cell.count > 1 {
                let run = &mut self.index_buf[cell.base as usize..(cell.base + cell.count) as usize];
                run.sort_unstable_by(|&a, &b| {
                    brightness(&lights[b as usize]).total_cmp(&brightness(&lights[a as usize]))
                });
            }
        }

        // Empty scene → one sentinel record whose key can never match a real probe, so the buffer
        // is never zero-sized and the GPU binary search always has ≥1 element (cf. chunk SENTINEL).
        if self.cells.is_empty() {
            self.cells.push(GpuLightCell {
                key_hi: u32::MAX,
                key_lo: u32::MAX,
                base: 0,
                count: 0,
            });
        }
    }

    /// Invoke `f(self, cell)` for every world cell the light's range-sphere overlaps (absolute cell
    /// coords, no window). Walks the light's cell-AABB (axis-span-clamped against a pathological
    /// `range`) and applies the tighter sphere-vs-cell test. Skips sentinel lights (`range <= 0`).
    fn for_overlapped_cells(&mut self, light: &GpuPointLight, mut f: impl FnMut(&mut Self, IVec3)) {
        let range = light.pos_range.w;
        if range <= 0.0 {
            return;
        }
        let center = light.pos_range.truncate();
        let lo = cell_of(center - Vec3::splat(range));
        let hi = cell_of(center + Vec3::splat(range)).min(lo + IVec3::splat(MAX_CELL_SPAN));
        for cz in lo.z..=hi.z {
            for cy in lo.y..=hi.y {
                for cx in lo.x..=hi.x {
                    let cell = IVec3::new(cx, cy, cz);
                    let cell_min = cell.as_vec3() * LIGHT_CELL_SIZE;
                    let cell_max = cell_min + Vec3::splat(LIGHT_CELL_SIZE);
                    if sphere_hits_aabb(center, range, cell_min, cell_max) {
                        f(self, cell);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn light(pos: Vec3, range: f32, radiance: f32) -> GpuPointLight {
        GpuPointLight {
            pos_range: pos.extend(range),
            color_radius: Vec3::splat(radiance).extend(0.0),
        }
    }

    /// The CPU mirror of the WGSL binary-search lookup: `(base, count)` for the cell at `p`, or
    /// `(0, 0)` on a miss. Guards that the sorted directory is searchable exactly as the shader will.
    fn lights_in_cell(grid: &LightGrid, p: Vec3) -> (u32, u32) {
        let want = light_gpu_key(cell_of(p));
        // Lower-bound binary search by (key_hi, key_lo) — mirrors lights.wgsl.
        let mut lo = 0usize;
        let mut hi = grid.cells.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let e = grid.cells[mid];
            if (e.key_hi, e.key_lo) < want {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo < grid.cells.len() {
            let e = grid.cells[lo];
            if (e.key_hi, e.key_lo) == want {
                return (e.base, e.count);
            }
        }
        (0, 0)
    }

    /// A single light is found (via the binary search) at its own position, referencing light 0.
    #[test]
    fn single_light_is_found_at_its_position() {
        let mut grid = LightGrid::default();
        let pos = Vec3::new(10.0, 2.0, -5.0);
        grid.rebuild(&[light(pos, 5.0, 1.0)]);

        let (base, count) = lights_in_cell(&grid, pos);
        assert!(count >= 1, "the light's own cell must list it");
        let run = &grid.index_buf[base as usize..(base + count) as usize];
        assert!(run.contains(&0), "cell run must reference light 0");
    }

    /// A FAR light is now binned (no camera window); only a sentinel (range <= 0) is skipped.
    #[test]
    fn far_light_is_binned_sentinel_is_not() {
        let mut grid = LightGrid::default();
        grid.rebuild(&[light(Vec3::ZERO, 0.0, 1.0)]); // sentinel only
        assert!(grid.index_buf.is_empty(), "a range-0 light binds nothing");
        assert_eq!(grid.cells.len(), 1, "empty scene gets one sentinel cell");
        assert_eq!(grid.cells[0].key_hi, u32::MAX, "sentinel key");

        let far = Vec3::new(10_000.0, 0.0, -8_000.0); // far from origin — no window now
        grid.rebuild(&[light(far, 5.0, 1.0)]);
        assert!(!grid.index_buf.is_empty(), "a far light is binned world-wide");
        let (_, count) = lights_in_cell(&grid, far);
        assert!(count >= 1, "the far light is found at its position");
    }

    /// A light spanning several cells appears in each (sphere-vs-cell binning).
    #[test]
    fn wide_light_bins_into_many_cells() {
        let mut grid = LightGrid::default();
        // Range 20 at cell 8 → ~5 cells/axis footprint → many distinct cells.
        grid.rebuild(&[light(Vec3::ZERO, 20.0, 1.0)]);
        assert!(grid.cells.len() > 1, "a 20 m light spans multiple 8 m cells (got {})", grid.cells.len());
        assert!(grid.index_buf.len() > 1);
    }

    /// Within a cell, the brightest light sorts first (so the shader shadows the most important).
    #[test]
    fn cell_run_is_sorted_brightest_first() {
        let mut grid = LightGrid::default();
        let p = Vec3::new(1.0, 1.0, 1.0);
        grid.rebuild(&[light(p, 5.0, 0.1), light(p, 5.0, 10.0)]); // both in one cell
        let (base, count) = lights_in_cell(&grid, p);
        assert!(count >= 2);
        assert_eq!(grid.index_buf[base as usize], 1, "brightest light (index 1) sorts first");
    }

    /// Packing is order-preserving: sorting `(key_hi, key_lo)` orders by `(x, y, z)` — the binary
    /// search precondition (mirror of `chunk::gpu_key_is_order_preserving`).
    #[test]
    fn gpu_key_is_order_preserving() {
        let mut prev: Option<(u32, u32)> = None;
        for x in -3..=3 {
            for y in -3..=3 {
                for z in -3..=3 {
                    let k = light_gpu_key(IVec3::new(x, y, z));
                    if let Some(p) = prev {
                        assert!(p < k, "keys must strictly increase in (x,y,z) order");
                    }
                    prev = Some(k);
                }
            }
        }
    }

    /// Distinct cells in range never collide.
    #[test]
    fn gpu_key_no_collision_in_range() {
        let mut seen = std::collections::HashSet::new();
        for x in -40..=40 {
            for y in -40..=40 {
                for z in -40..=40 {
                    assert!(seen.insert(light_gpu_key(IVec3::new(x, y, z))), "key collision");
                }
            }
        }
    }

    /// Resolve parity: for a grid of several lights, the binary-search lookup at each light's
    /// position returns a run that actually contains that light — i.e. the sorted directory + the
    /// binning agree (the CPU analogue of the GPU lookup the shader does).
    #[test]
    fn binary_search_resolves_binning() {
        let mut grid = LightGrid::default();
        let lights: Vec<_> = (0..50)
            .map(|i| {
                let f = i as f32;
                light(Vec3::new(f * 7.0 - 100.0, (f * 3.0) % 40.0, f * -5.0 + 60.0), 6.0, 1.0 + f)
            })
            .collect();
        grid.rebuild(&lights);
        // cells must be sorted by key.
        for w in grid.cells.windows(2) {
            assert!((w[0].key_hi, w[0].key_lo) < (w[1].key_hi, w[1].key_lo), "cells must be key-sorted");
        }
        for (i, l) in lights.iter().enumerate() {
            let (base, count) = lights_in_cell(&grid, l.pos_range.truncate());
            let run = &grid.index_buf[base as usize..(base + count) as usize];
            assert!(run.contains(&(i as u32)), "light {i} must be in the run at its own cell");
        }
    }

    /// The Rust constants must match the WGSL mirror, or CPU binning and GPU lookup disagree and
    /// lights flicker/vanish as the camera moves (the camera-shift bug class).
    #[test]
    fn wgsl_light_grid_constants_match_rust() {
        let src = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/shaders/sdf/lights.wgsl"
        ))
        .expect("read lights.wgsl");

        let lit_after = |pat: &str| -> String {
            let i = src.find(pat).unwrap_or_else(|| panic!("lights.wgsl missing `{pat}`"));
            src[i + pat.len()..]
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                .collect()
        };
        assert_eq!(
            lit_after("const LIGHT_CELL_SIZE: f32 =").parse::<f32>().unwrap(),
            LIGHT_CELL_SIZE
        );
        assert_eq!(
            lit_after("let bias =").parse::<i32>().unwrap(),
            LIGHT_KEY_BIAS,
            "WGSL light_cell_key bias must match LIGHT_KEY_BIAS"
        );
    }
}
