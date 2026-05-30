//! Flat, GPU-upload-ready bounding-volume hierarchy over per-edit influence AABBs.
//!
//! Used CPU-side to cull candidate edits per brick during baking and to accelerate
//! picking/raycasting, and uploaded to the GPU (as raw [`GpuBvhNode`] bytes) to
//! speed empty-space skipping in the raymarch. The node array is a flat `Vec`
//! (no pointers) so the same layout serves both CPU traversal and std430 upload.

use bevy::math::bounding::{Aabb3d, BoundingVolume};
use bevy::prelude::*;

/// One BVH node. 32 bytes, std430-compatible (two `vec3 + u32` rows).
///
/// `count == 0` → internal node: `left_or_first` is the left child index and
/// `count_or_right` is the right child index.
/// `count > 0` → leaf: `left_or_first` is the first index into [`Bvh::edit_indices`]
/// and `count_or_right` is how many edits the leaf holds.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BvhNode {
    pub aabb_min: [f32; 3],
    pub left_or_first: u32,
    pub aabb_max: [f32; 3],
    pub count_or_right: u32,
}

// Internal vs leaf is encoded by the high bit of `count_or_right` (see
// `INTERNAL_FLAG`): set => internal node (the field is the right-child index);
// clear => leaf (the field is the edit count). This keeps the node a tight 32
// bytes with no extra flag word and stays unambiguous even for child index 0.

/// Maximum edits stored in a single leaf before we stop trying to split.
const LEAF_SIZE: usize = 4;
/// Sentinel high bit on `count_or_right` marking an internal node (so a right-child
/// index of 0 is unambiguous).
const INTERNAL_FLAG: u32 = 0x8000_0000;

/// A built BVH: the flat node array plus the leaf-referenced edit index list.
#[derive(Resource, Default, Clone)]
pub struct Bvh {
    pub nodes: Vec<BvhNode>,
    /// Edit indices (into the caller's edit slice) grouped by leaf.
    pub edit_indices: Vec<u32>,
}

struct BuildItem {
    aabb: Aabb3d,
    center: Vec3,
    edit: u32,
}

impl Bvh {
    /// Build a BVH over the given per-edit world AABBs. Median split on the longest
    /// axis; leaves hold up to [`LEAF_SIZE`] edits.
    pub fn build(edit_aabbs: &[Aabb3d]) -> Self {
        let mut bvh = Bvh::default();
        if edit_aabbs.is_empty() {
            return bvh;
        }

        let mut items: Vec<BuildItem> = edit_aabbs
            .iter()
            .enumerate()
            .map(|(i, a)| BuildItem {
                aabb: *a,
                center: Vec3::from(a.center()),
                edit: i as u32,
            })
            .collect();

        // Reserve node 0 as the root; build recursively into `nodes`.
        bvh.nodes.push(BvhNode {
            aabb_min: [0.0; 3],
            left_or_first: 0,
            aabb_max: [0.0; 3],
            count_or_right: 0,
        });
        let range = 0..items.len();
        build_node(&mut bvh, &mut items, range, 0);
        bvh
    }

    /// True if the BVH holds no edits.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Collect edit indices whose leaf AABB-subtree overlaps `query`. Iterative
    /// stack walk (no recursion), so it mirrors the GPU traversal shape.
    pub fn query_aabb(&self, query: &Aabb3d, out: &mut Vec<u32>) {
        out.clear();
        if self.nodes.is_empty() {
            return;
        }
        let mut stack: Vec<u32> = vec![0];
        while let Some(ni) = stack.pop() {
            let node = &self.nodes[ni as usize];
            if !aabb_overlap(node, query) {
                continue;
            }
            if is_internal(node) {
                stack.push(node.left_or_first);
                stack.push(node.count_or_right & !INTERNAL_FLAG);
            } else {
                let first = node.left_or_first as usize;
                let count = node.count_or_right as usize;
                out.extend_from_slice(&self.edit_indices[first..first + count]);
            }
        }
    }

    /// Visit edit indices whose subtree the ray enters. Order is not sorted by t;
    /// callers that need the nearest hit evaluate all visited edits. Returns all
    /// candidate edit indices (deduplication left to the caller if needed).
    pub fn raycast_candidates(&self, origin: Vec3, dir: Vec3, max_t: f32, out: &mut Vec<u32>) {
        out.clear();
        if self.nodes.is_empty() {
            return;
        }
        let inv = Vec3::new(
            1.0 / safe_dir(dir.x),
            1.0 / safe_dir(dir.y),
            1.0 / safe_dir(dir.z),
        );
        let mut stack: Vec<u32> = vec![0];
        while let Some(ni) = stack.pop() {
            let node = &self.nodes[ni as usize];
            if ray_aabb(node, origin, inv, max_t).is_none() {
                continue;
            }
            if is_internal(node) {
                stack.push(node.left_or_first);
                stack.push(node.count_or_right & !INTERNAL_FLAG);
            } else {
                let first = node.left_or_first as usize;
                let count = node.count_or_right as usize;
                out.extend_from_slice(&self.edit_indices[first..first + count]);
            }
        }
    }

}

fn is_internal(node: &BvhNode) -> bool {
    node.count_or_right & INTERNAL_FLAG != 0
}

fn safe_dir(d: f32) -> f32 {
    if d.abs() < 1e-8 {
        1e-8_f32.copysign(d).max(1e-8)
    } else {
        d
    }
}

fn aabb_overlap(node: &BvhNode, q: &Aabb3d) -> bool {
    let nmin = node.aabb_min;
    let nmax = node.aabb_max;
    nmin[0] <= q.max.x
        && nmax[0] >= q.min.x
        && nmin[1] <= q.max.y
        && nmax[1] >= q.min.y
        && nmin[2] <= q.max.z
        && nmax[2] >= q.min.z
}

/// Slab test; returns entry `t` if the ray hits the node's box within `max_t`.
fn ray_aabb(node: &BvhNode, origin: Vec3, inv_dir: Vec3, max_t: f32) -> Option<f32> {
    let mut tmin = 0.0_f32;
    let mut tmax = max_t;
    let lo = node.aabb_min;
    let hi = node.aabb_max;
    for a in 0..3 {
        let o = origin[a];
        let inv = inv_dir[a];
        let mut t0 = (lo[a] - o) * inv;
        let mut t1 = (hi[a] - o) * inv;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        tmin = tmin.max(t0);
        tmax = tmax.min(t1);
        if tmax < tmin {
            return None;
        }
    }
    Some(tmin)
}

/// Recursively populate node `node_idx` for `items[range]`.
fn build_node(
    bvh: &mut Bvh,
    items: &mut [BuildItem],
    range: std::ops::Range<usize>,
    node_idx: usize,
) {
    let slice = &items[range.clone()];
    let bounds = bounds_of(slice);
    let (bmin, bmax) = bounds;

    let count = range.len();
    if count <= LEAF_SIZE {
        let first = bvh.edit_indices.len() as u32;
        for it in &items[range] {
            bvh.edit_indices.push(it.edit);
        }
        bvh.nodes[node_idx] = BvhNode {
            aabb_min: bmin.into(),
            left_or_first: first,
            aabb_max: bmax.into(),
            count_or_right: count as u32, // leaf: high bit clear
        };
        return;
    }

    // Split on the longest axis at the median centroid.
    let extent = bmax - bmin;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };

    let start = range.start;
    let sub = &mut items[range.clone()];
    sub.sort_by(|a, b| {
        a.center[axis]
            .partial_cmp(&b.center[axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mid = start + count / 2;

    // Allocate child node slots.
    let left_idx = bvh.nodes.len();
    bvh.nodes.push(placeholder());
    let right_idx = bvh.nodes.len();
    bvh.nodes.push(placeholder());

    bvh.nodes[node_idx] = BvhNode {
        aabb_min: bmin.into(),
        left_or_first: left_idx as u32,
        aabb_max: bmax.into(),
        count_or_right: (right_idx as u32) | INTERNAL_FLAG,
    };

    build_node(bvh, items, start..mid, left_idx);
    build_node(bvh, items, mid..range.end, right_idx);
}

fn placeholder() -> BvhNode {
    BvhNode {
        aabb_min: [0.0; 3],
        left_or_first: 0,
        aabb_max: [0.0; 3],
        count_or_right: 0,
    }
}

fn bounds_of(items: &[BuildItem]) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for it in items {
        min = min.min(Vec3::from(it.aabb.min));
        max = max.max(Vec3::from(it.aabb.max));
    }
    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aabb(cx: f32, cy: f32, cz: f32, h: f32) -> Aabb3d {
        Aabb3d::new(Vec3::new(cx, cy, cz), Vec3::splat(h))
    }

    #[test]
    fn empty_build() {
        let bvh = Bvh::build(&[]);
        assert!(bvh.is_empty());
        let mut out = Vec::new();
        bvh.query_aabb(&aabb(0.0, 0.0, 0.0, 1.0), &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn root_bounds_contain_all() {
        let boxes = vec![
            aabb(-5.0, 0.0, 0.0, 1.0),
            aabb(5.0, 0.0, 0.0, 1.0),
            aabb(0.0, 5.0, 0.0, 1.0),
            aabb(0.0, -5.0, 0.0, 1.0),
            aabb(0.0, 0.0, 5.0, 1.0),
        ];
        let bvh = Bvh::build(&boxes);
        let root = &bvh.nodes[0];
        assert!(root.aabb_min[0] <= -6.0 + 1e-3);
        assert!(root.aabb_max[0] >= 6.0 - 1e-3);
        assert!(root.aabb_max[1] >= 6.0 - 1e-3);
        assert!(root.aabb_max[2] >= 6.0 - 1e-3);
    }

    /// `query_aabb` is a broad phase: it returns every edit in any leaf whose
    /// subtree AABB overlaps the query (a superset — extra far edits are harmless
    /// for the bake, which evaluates them and lets them lose the min/argmin). The
    /// guarantee is that an overlapping edit is always included, and edits in a
    /// non-overlapping subtree are excluded. With enough spatial separation the
    /// tree splits so the far edit is pruned.
    #[test]
    fn query_includes_overlapping_and_prunes_distant() {
        // Two far-apart clusters of 3; >LEAF_SIZE total forces a split so the
        // median plane separates left (0,1,2) from right (3,4,5).
        let boxes = vec![
            aabb(-50.0, 0.0, 0.0, 1.0), // edit 0
            aabb(-49.0, 0.0, 0.0, 1.0), // edit 1
            aabb(-48.0, 0.0, 0.0, 1.0), // edit 2
            aabb(48.0, 0.0, 0.0, 1.0),  // edit 3
            aabb(49.0, 0.0, 0.0, 1.0),  // edit 4
            aabb(50.0, 0.0, 0.0, 1.0),  // edit 5
        ];
        let bvh = Bvh::build(&boxes);

        let mut out = Vec::new();
        bvh.query_aabb(&aabb(50.0, 0.0, 0.0, 0.5), &mut out);
        out.sort();
        // Must include the overlapping edit 5, and must NOT include the far-left
        // cluster (0,1,2) which lives in the other subtree.
        assert!(out.contains(&5), "overlapping edit must be returned");
        assert!(
            !out.contains(&0) && !out.contains(&1) && !out.contains(&2),
            "distant subtree must be pruned, got {out:?}"
        );
    }

    #[test]
    fn query_collects_all_when_overlapping_everything() {
        let boxes = vec![
            aabb(-2.0, 0.0, 0.0, 1.0),
            aabb(0.0, 0.0, 0.0, 1.0),
            aabb(2.0, 0.0, 0.0, 1.0),
        ];
        let bvh = Bvh::build(&boxes);
        let mut out = Vec::new();
        bvh.query_aabb(&aabb(0.0, 0.0, 0.0, 10.0), &mut out);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn raycast_hits_box_on_axis() {
        let boxes = vec![aabb(0.0, 0.0, 5.0, 1.0)];
        let bvh = Bvh::build(&boxes);
        let mut out = Vec::new();
        bvh.raycast_candidates(Vec3::ZERO, Vec3::Z, 100.0, &mut out);
        assert_eq!(out, vec![0]);

        // Ray pointing away misses.
        bvh.raycast_candidates(Vec3::ZERO, -Vec3::Z, 100.0, &mut out);
        assert!(out.is_empty());
    }
}
