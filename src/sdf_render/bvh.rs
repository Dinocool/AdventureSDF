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

/// `parent` sentinel for the root node (it has no parent). `u32::MAX` so a real index never collides.
const NO_PARENT: u32 = u32::MAX;

/// A built BVH: the flat node array plus the leaf-referenced edit index list.
#[derive(Resource, Default, Clone)]
pub struct Bvh {
    pub nodes: Vec<BvhNode>,
    /// Edit indices (into the caller's edit slice) grouped by leaf.
    pub edit_indices: Vec<u32>,
    /// Per-edit world AABB, indexed by the ORIGINAL edit index (the value stored in
    /// `edit_indices`), not by leaf order. CPU-only (never uploaded). Lets a cull refine the
    /// over-returned leaf set down to edits whose own box overlaps the query — see
    /// [`crate::sdf_render::atlas::SdfAtlas::cull_edit_indices_with`].
    pub edit_aabbs: Vec<Aabb3d>,
    /// For each ORIGINAL edit index, the leaf node holding it. Lets [`Self::refit_edit`] update a
    /// moved edit's leaf + ancestors in O(depth) instead of a full O(n log n) rebuild. Built by
    /// `build`; same length as `edit_aabbs`.
    edit_to_leaf: Vec<u32>,
    /// For each node, its parent node index ([`NO_PARENT`] for the root). The bottom-up walk in
    /// [`Self::refit_edit`] follows it. Built by `build`; same length as `nodes`.
    parent: Vec<u32>,
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
        bvh.edit_aabbs = edit_aabbs.to_vec();
        // Filled per leaf in `build_node`; sized up front, indexed by ORIGINAL edit index.
        bvh.edit_to_leaf = vec![0u32; edit_aabbs.len()];

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
        bvh.parent.push(NO_PARENT); // root
        let range = 0..items.len();
        build_node(&mut bvh, &mut items, range, 0);
        bvh
    }

    /// Refit a single moved edit's AABB in O(depth): update `edit_aabbs[edit_idx]`, recompute its
    /// leaf node's box from the leaf's edits, then walk parents to the root recomputing each as the
    /// union of its two children. The tree TOPOLOGY is unchanged (same nodes, same `edit_indices`),
    /// so every consumer — the cull (`query_aabb_with` + `edit_aabb`), the recenter overlap test,
    /// picking — stays correct; only the bounds tighten/loosen. This is the localized-edit fast path
    /// vs the O(n log n) `build`. A big move (an edit teleporting across the scene) keeps the result
    /// CORRECT but loosens ancestor bounds (slower queries), so the caller should occasionally
    /// `build` to restore split quality (e.g. once a drag ends).
    pub fn refit_edit(&mut self, edit_idx: u32, new_aabb: Aabb3d) {
        let ei = edit_idx as usize;
        if ei >= self.edit_aabbs.len() {
            return;
        }
        self.edit_aabbs[ei] = new_aabb;
        let leaf = self.edit_to_leaf[ei] as usize;

        // Recompute the leaf box as the union of its (now-updated) edits' AABBs.
        let first = self.nodes[leaf].left_or_first as usize;
        let count = self.nodes[leaf].count_or_right as usize; // leaf ⇒ high bit clear
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for k in first..first + count {
            let a = self.edit_aabbs[self.edit_indices[k] as usize];
            min = min.min(Vec3::from(a.min));
            max = max.max(Vec3::from(a.max));
        }
        self.nodes[leaf].aabb_min = min.into();
        self.nodes[leaf].aabb_max = max.into();

        // Walk to the root, recomputing each ancestor from its two children (so bounds can both
        // grow AND shrink — a true refit, not just an expand).
        let mut node = leaf;
        while self.parent[node] != NO_PARENT {
            let p = self.parent[node] as usize;
            let l = self.nodes[p].left_or_first as usize;
            let r = (self.nodes[p].count_or_right & !INTERNAL_FLAG) as usize;
            let (lmin, lmax) = (self.nodes[l].aabb_min, self.nodes[l].aabb_max);
            let (rmin, rmax) = (self.nodes[r].aabb_min, self.nodes[r].aabb_max);
            self.nodes[p].aabb_min = [lmin[0].min(rmin[0]), lmin[1].min(rmin[1]), lmin[2].min(rmin[2])];
            self.nodes[p].aabb_max = [lmax[0].max(rmax[0]), lmax[1].max(rmax[1]), lmax[2].max(rmax[2])];
            node = p;
        }
    }

    /// True if the BVH holds no edits.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// World AABB of edit `idx` (the original edit index stored in leaves), if recorded.
    pub fn edit_aabb(&self, idx: u32) -> Option<&Aabb3d> {
        self.edit_aabbs.get(idx as usize)
    }

    /// Collect edit indices whose leaf AABB-subtree overlaps `query`. Iterative
    /// stack walk (no recursion), so it mirrors the GPU traversal shape. Allocates a fresh
    /// traversal stack each call — for hot per-brick loops use [`Self::query_aabb_with`].
    pub fn query_aabb(&self, query: &Aabb3d, out: &mut Vec<u32>) {
        let mut stack: Vec<u32> = Vec::new();
        self.query_aabb_with(query, out, &mut stack);
    }

    /// True if ANY edit's leaf subtree overlaps `query`, stopping at the first hit. For callers
    /// that only need an occupancy boolean (e.g. the recenter's `chunk_has_geometry` and the emit
    /// phase-1 pre-cull), this avoids collecting every overlapping edit index into a Vec — a big
    /// win on dense chunks that overlap hundreds of edits. Reuses a caller-owned traversal `stack`.
    pub fn any_overlap_with(&self, query: &Aabb3d, stack: &mut Vec<u32>) -> bool {
        stack.clear();
        if self.nodes.is_empty() {
            return false;
        }
        stack.push(0);
        while let Some(ni) = stack.pop() {
            let node = &self.nodes[ni as usize];
            if !aabb_overlap(node, query) {
                continue;
            }
            if is_internal(node) {
                stack.push(node.left_or_first);
                stack.push(node.count_or_right & !INTERNAL_FLAG);
            } else {
                // A leaf overlaps the query box → at least one candidate edit reaches it. (Leaf
                // bounds are the union of its edits' AABBs; a leaf-level overlap is the same
                // occupancy signal the full collect would yield a non-empty `out` for.)
                return true;
            }
        }
        false
    }

    /// As [`Self::query_aabb`] but reuses a caller-owned traversal `stack` (cleared on entry), so
    /// a tight per-brick cull loop does zero heap allocation per query.
    pub fn query_aabb_with(&self, query: &Aabb3d, out: &mut Vec<u32>, stack: &mut Vec<u32>) {
        out.clear();
        stack.clear();
        if self.nodes.is_empty() {
            return;
        }
        stack.push(0);
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
            bvh.edit_to_leaf[it.edit as usize] = node_idx as u32;
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

    // Allocate child node slots (keep `parent` parallel to `nodes`).
    let left_idx = bvh.nodes.len();
    bvh.nodes.push(placeholder());
    bvh.parent.push(node_idx as u32);
    let right_idx = bvh.nodes.len();
    bvh.nodes.push(placeholder());
    bvh.parent.push(node_idx as u32);

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

    fn aabbs_overlap(a: &Aabb3d, b: &Aabb3d) -> bool {
        a.min.x <= b.max.x && a.max.x >= b.min.x
            && a.min.y <= b.max.y && a.max.y >= b.min.y
            && a.min.z <= b.max.z && a.max.z >= b.min.z
    }

    /// `refit_edit` must keep the BVH a correct CONSERVATIVE cull: after moving an edit, every edit
    /// whose AABB overlaps a query must still be returned (a superset is fine; a MISS is a bake hole).
    #[test]
    fn refit_keeps_queries_a_correct_superset() {
        let mut boxes: Vec<Aabb3d> = (0..40)
            .map(|i| aabb((i % 8) as f32 * 3.0, (i / 8) as f32 * 3.0, 0.0, 1.0))
            .collect();
        let mut bvh = Bvh::build(&boxes);

        // Move several edits to new locations and refit each.
        for (idx, c) in [(13u32, 100.0f32), (0, -50.0), (27, 60.0)] {
            let m = aabb(c, c, c, 1.0);
            boxes[idx as usize] = m;
            bvh.refit_edit(idx, m);
        }

        let queries = [
            aabb(100.0, 100.0, 100.0, 0.5),
            aabb(-50.0, -50.0, -50.0, 0.5),
            aabb(0.0, 0.0, 0.0, 2.0),
            aabb(12.0, 9.0, 0.0, 1.5),
            aabb(60.0, 60.0, 60.0, 3.0),
        ];
        let mut stack = Vec::new();
        let mut out = Vec::new();
        for q in queries {
            bvh.query_aabb_with(&q, &mut out, &mut stack);
            let got: std::collections::HashSet<u32> = out.iter().copied().collect();
            for (i, b) in boxes.iter().enumerate() {
                if aabbs_overlap(b, &q) {
                    assert!(got.contains(&(i as u32)), "refit DROPPED overlapping edit {i} for query {q:?}");
                }
            }
            // The refined per-edit AABB must also be the moved one (the cull's refine step reads it).
            assert!(aabbs_overlap(bvh.edit_aabb(13).unwrap(), &aabb(100.0, 100.0, 100.0, 1.1)));
        }
        // Root must still bound every (moved) edit.
        let root = &bvh.nodes[0];
        for b in &boxes {
            assert!(root.aabb_min[0] <= b.min.x + 1e-3 && root.aabb_max[0] >= b.max.x - 1e-3);
        }
    }

    // --- BVH performance harness ----------------------------------------------------
    //
    // Measures the localized-edit cost the production drag path pays: a FULL `build` (the old
    // per-drag-frame cost) vs a `refit_edit` (the fix), over the REAL ~14.6k-edit stress scene,
    // plus query cost/candidates before vs after a long drag (the refit-quality degradation) and
    // after a restoring rebuild. Run:
    //   cargo test --release --lib sdf_render::bvh::perf_localized_edit -- --ignored --nocapture

    /// CPU memory of the BVH's backing Vecs (KB).
    #[cfg(test)]
    fn bvh_mem_kb(b: &Bvh) -> usize {
        use std::mem::size_of;
        (b.nodes.len() * size_of::<BvhNode>()
            + b.edit_indices.len() * 4
            + b.edit_aabbs.len() * size_of::<Aabb3d>()
            + b.parent.len() * 4
            + b.edit_to_leaf.len() * 4)
            / 1024
    }

    /// Shift an AABB by `dx` along +X (mimics a drag step).
    #[cfg(test)]
    fn shift(a: Aabb3d, dx: f32) -> Aabb3d {
        let d = bevy::math::Vec3A::new(dx, 0.0, 0.0);
        Aabb3d { min: a.min + d, max: a.max + d }
    }

    /// Cull a fixed grid of brick-sized query boxes over the stress field; return (total µs, total
    /// candidate count). A proxy for the bake's per-brick cull cost — sensitive to BVH bounds
    /// quality (a degraded tree returns more candidates and takes longer).
    #[cfg(test)]
    fn query_sweep(bvh: &Bvh) -> (u128, usize) {
        let mut stack = Vec::new();
        let mut out = Vec::new();
        let mut cands = 0usize;
        let t = std::time::Instant::now();
        // ~12×12×6 grid of 0.5m boxes over the ±270m field near the tower band (y∈[-35,-5]).
        let mut x = -270.0;
        while x <= 270.0 {
            let mut z = -270.0;
            while z <= 270.0 {
                let mut y = -35.0;
                while y <= -5.0 {
                    let q = Aabb3d::new(Vec3::new(x, y, z), Vec3::splat(0.5));
                    bvh.query_aabb_with(&q, &mut out, &mut stack);
                    cands += out.len();
                    y += 6.0;
                }
                z += 45.0;
            }
            x += 45.0;
        }
        (t.elapsed().as_micros(), cands)
    }

    #[test]
    #[ignore = "perf rig; run with --release --ignored --nocapture"]
    fn perf_localized_edit() {
        use crate::sdf_render::{edits, tower_field};
        let tf = tower_field::tower_field_edits(&tower_field::TowerFieldParams::default());
        let aabbs: Vec<Aabb3d> = tf.iter().map(|(_o, t, p, _r)| edits::edit_world_aabb(p, t, 0.0)).collect();
        let n = aabbs.len();

        // 1. Full build (the old per-drag-frame cost) — average of 5.
        let mut build_us = u128::MAX;
        let mut bvh = Bvh::build(&aabbs);
        for _ in 0..5 {
            let t = std::time::Instant::now();
            bvh = Bvh::build(&aabbs);
            build_us = build_us.min(t.elapsed().as_micros());
        }
        eprintln!("BVH-PERF: {n} edits | full build = {build_us}us (best of 5) | nodes={} mem~{}KB", bvh.nodes.len(), bvh_mem_kb(&bvh));

        let (q_us0, cand0) = query_sweep(&bvh);
        eprintln!("BVH-PERF: query sweep (fresh build) = {q_us0}us, {cand0} candidates");

        // 2. ONE localized refit (the drag fix) vs the full build.
        let mid = n / 2;
        let moved = shift(aabbs[mid], 0.3);
        let t = std::time::Instant::now();
        bvh.refit_edit(mid as u32, moved);
        let refit_ns = t.elapsed().as_nanos().max(1);
        eprintln!(
            "BVH-PERF: refit ONE edit = {refit_ns}ns  →  {:.0}x faster than the {build_us}us full rebuild",
            (build_us as f64 * 1000.0) / refit_ns as f64
        );

        // 3. A 200-frame drag: refit each frame; measure per-frame cost + query degradation.
        let mut acc = aabbs[mid];
        let t = std::time::Instant::now();
        for _ in 0..200 {
            acc = shift(acc, 0.3);
            bvh.refit_edit(mid as u32, acc);
        }
        let drag_us = t.elapsed().as_micros();
        let (q_us1, cand1) = query_sweep(&bvh);
        eprintln!("BVH-PERF: 200-frame drag = {drag_us}us total ({:.1}us/frame refit)", drag_us as f64 / 200.0);
        eprintln!("BVH-PERF: query sweep (after 60m drag, degraded tree) = {q_us1}us, {cand1} candidates (was {q_us0}us/{cand0})");

        // 4. Rebuild-on-idle restores split quality.
        let mut final_aabbs = aabbs.clone();
        final_aabbs[mid] = acc;
        let bvh2 = Bvh::build(&final_aabbs);
        let (q_us2, cand2) = query_sweep(&bvh2);
        eprintln!("BVH-PERF: query sweep (after restoring rebuild) = {q_us2}us, {cand2} candidates");
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
