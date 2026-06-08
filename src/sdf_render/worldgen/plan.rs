//! `GenerationPlan` — which chunks of which layers must exist for the current focus, and in what
//! order (dependencies first).
//!
//! This is the heart of the LayerProcGen port (WORLD_GEN_PLAN §2.7). Top-level requirements (the
//! rendered/consumed layers, within a radius of each focus point) pull their dependencies through
//! **padded read-windows**: a dependent chunk forces every dependency chunk overlapping
//! `(its world bounds + the dependency's padding)` to exist first. Walking this DAG yields the exact
//! required-chunk set per layer, dependency-ordered, which the `LayerManager` diffs against residency
//! to roll generation.
//!
//! Pure integer/`f64` lattice math (deterministic, focus-rounded to each tier) — no Bevy, fully
//! unit-tested. The Phase-1 slice has a single root layer (height), so the plan is just "height
//! chunks within radius"; the padding/DAG machinery is exercised by the synthetic multi-tier tests
//! and is ready for climate/biome/cave layers.

use std::collections::BTreeSet;

use bevy::math::{DVec2, IVec3};
use rustc_hash::FxHashMap;

use super::coord::{ChunkCoord, ChunkSize, LayerId};
use super::layer::LayerDependency;

/// The planner's view of one layer: identity, tier size, dependencies, and (if it is directly
/// consumed by rendering) the world radius around each focus point its chunks are required within.
#[derive(Clone, Debug)]
pub struct LayerMeta {
    pub id: LayerId,
    pub size: ChunkSize,
    pub deps: Vec<LayerDependency>,
    /// `Some(r)` ⇒ the focus directly requires this layer's chunks within radius `r` metres (a
    /// rendered/consumed layer, e.g. the slice's height layer). `None` ⇒ purely a dependency, pulled
    /// only through other layers' padding.
    pub direct_radius: Option<f64>,
}

/// The required chunks per layer, dependency-ordered (every layer appears after all layers it
/// depends on, so generating in this order means a chunk's dependencies are already resident).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerationPlan {
    pub required: Vec<(LayerId, Vec<ChunkCoord>)>,
}

impl GenerationPlan {
    /// All `(layer, coord)` pairs in dependency order — flattened convenience for the manager.
    pub fn iter(&self) -> impl Iterator<Item = ChunkCoord> + '_ {
        self.required.iter().flat_map(|(_, cs)| cs.iter().copied())
    }

    /// Total required chunks across all layers.
    pub fn len(&self) -> usize {
        self.required.iter().map(|(_, cs)| cs.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.required.iter().all(|(_, cs)| cs.is_empty())
    }

    /// Build the plan for `metas` around `focus` points (XZ; 2D lattices — Y is 0 for now).
    ///
    /// 1. Topologically order layers (dependencies first).
    /// 2. Seed direct requirements: each `direct_radius` layer gets the chunks overlapping each
    ///    focus's radius box.
    /// 3. Walk dependents → dependencies (reverse topo): a layer's required set is final before it
    ///    pulls its own deps, so transitive padding accumulates correctly across tiers.
    /// 4. Emit in topo order (dependencies first).
    pub fn build(metas: &[LayerMeta], focus: &[DVec2]) -> GenerationPlan {
        let by_id: FxHashMap<LayerId, usize> = metas.iter().enumerate().map(|(i, m)| (m.id, i)).collect();
        let topo = topo_order(metas, &by_id);

        // Required coord set per layer (BTreeSet → sorted, deterministic).
        let mut req: FxHashMap<LayerId, BTreeSet<ChunkCoord>> =
            metas.iter().map(|m| (m.id, BTreeSet::new())).collect();

        // (2) Direct requirements.
        for m in metas {
            if let Some(r) = m.direct_radius {
                let set = req.get_mut(&m.id).unwrap();
                for f in focus {
                    for c in chunks_overlapping_xz(m.id, m.size, *f - DVec2::splat(r), *f + DVec2::splat(r)) {
                        set.insert(c);
                    }
                }
            }
        }

        // (3) Pull dependencies, dependents first (reverse topo). A layer's required set is complete
        // (direct + pulled by higher dependents already processed) before we expand it onto its deps.
        for &mi in topo.iter().rev() {
            let m = &metas[mi];
            if m.deps.is_empty() {
                continue;
            }
            let dependent_chunks: Vec<ChunkCoord> = req[&m.id].iter().copied().collect();
            for dep in &m.deps {
                let Some(&di) = by_id.get(&dep.on) else { continue };
                let dep_size = metas[di].size;
                let s = m.size.world_size();
                let mut pulled = BTreeSet::new();
                for c in &dependent_chunks {
                    // The dependent chunk's world XZ bounds, expanded by the dependency's padding.
                    let min = DVec2::new(c.xyz.x as f64 * s, c.xyz.z as f64 * s)
                        - DVec2::splat(dep.padding);
                    let max = DVec2::new((c.xyz.x as f64 + 1.0) * s, (c.xyz.z as f64 + 1.0) * s)
                        + DVec2::splat(dep.padding);
                    for dc in chunks_overlapping_xz(dep.on, dep_size, min, max) {
                        pulled.insert(dc);
                    }
                }
                req.get_mut(&dep.on).unwrap().extend(pulled);
            }
        }

        // (4) Emit in topo (dependency-first) order.
        let required = topo
            .iter()
            .map(|&mi| {
                let id = metas[mi].id;
                (id, req[&id].iter().copied().collect::<Vec<_>>())
            })
            .collect();
        GenerationPlan { required }
    }
}

/// Chunks of `layer` (size `size`) overlapping the world XZ box `[min, max]`. Inclusive of the chunk
/// containing `max` (a boundary point belongs to the upper chunk). Uses `floor` to match the
/// CPU/GPU cell-mapping convention (§10).
fn chunks_overlapping_xz(layer: LayerId, size: ChunkSize, min: DVec2, max: DVec2) -> Vec<ChunkCoord> {
    let s = size.world_size();
    let cx0 = (min.x / s).floor() as i32;
    let cx1 = (max.x / s).floor() as i32;
    let cz0 = (min.y / s).floor() as i32;
    let cz1 = (max.y / s).floor() as i32;
    let mut out = Vec::with_capacity(((cx1 - cx0 + 1).max(0) * (cz1 - cz0 + 1).max(0)) as usize);
    for cz in cz0..=cz1 {
        for cx in cx0..=cx1 {
            out.push(ChunkCoord::new(layer, IVec3::new(cx, 0, cz)));
        }
    }
    out
}

/// Topological order (dependency indices before dependent indices) via DFS post-order. Assumes the
/// dependency graph is acyclic (validated at recipe-instantiation time elsewhere); a cycle would
/// simply break the "deps first" guarantee, not loop forever (visited-guarded).
fn topo_order(metas: &[LayerMeta], by_id: &FxHashMap<LayerId, usize>) -> Vec<usize> {
    let mut visited = vec![false; metas.len()];
    let mut order = Vec::with_capacity(metas.len());
    fn visit(
        i: usize,
        metas: &[LayerMeta],
        by_id: &FxHashMap<LayerId, usize>,
        visited: &mut [bool],
        order: &mut Vec<usize>,
    ) {
        if visited[i] {
            return;
        }
        visited[i] = true;
        for dep in &metas[i].deps {
            if let Some(&di) = by_id.get(&dep.on) {
                visit(di, metas, by_id, visited, order);
            }
        }
        order.push(i); // post-order: deps pushed before self
    }
    for i in 0..metas.len() {
        visit(i, metas, by_id, &mut visited, &mut order);
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(id: u32, cells: u32, deps: Vec<LayerDependency>, radius: Option<f64>) -> LayerMeta {
        LayerMeta { id: LayerId(id), size: ChunkSize::new(cells), deps, direct_radius: radius }
    }

    fn coords_for(plan: &GenerationPlan, id: u32) -> Vec<ChunkCoord> {
        plan.required.iter().find(|(l, _)| *l == LayerId(id)).map(|(_, c)| c.clone()).unwrap_or_default()
    }

    /// A single root layer with a direct radius → exactly the chunks overlapping the radius box.
    #[test]
    fn single_layer_radius_window() {
        // 100 m chunks, radius 150 m around the origin → chunks cx,cz ∈ {-2,-1,0,1} (box [-150,150]).
        let metas = [meta(0, 100, vec![], Some(150.0))];
        let plan = GenerationPlan::build(&metas, &[DVec2::ZERO]);
        let got = coords_for(&plan, 0);
        // floor(-150/100)=-2 .. floor(150/100)=1 → 4 indices per axis → 16 chunks.
        assert_eq!(got.len(), 16, "expected a 4x4 chunk window, got {}", got.len());
        assert!(got.contains(&ChunkCoord::new(LayerId(0), IVec3::new(-2, 0, -2))));
        assert!(got.contains(&ChunkCoord::new(LayerId(0), IVec3::new(1, 0, 1))));
        assert!(!got.contains(&ChunkCoord::new(LayerId(0), IVec3::new(2, 0, 0))));
    }

    /// A dependent layer pulls exactly the padded window of its (coarser) dependency — the §2.7
    /// contextual mechanism. Detail layer (size 50) depends on coarse layer (size 200) with padding
    /// 30; one detail chunk at origin pulls the coarse chunks overlapping [-30, 50+30] = [-30, 80].
    #[test]
    fn padded_dependency_window() {
        let detail = meta(1, 50, vec![LayerDependency { on: LayerId(0), padding: 30.0 }], Some(1.0));
        let coarse = meta(0, 200, vec![], None);
        // Focus at origin, tiny radius → just the detail chunk (0,0).
        let plan = GenerationPlan::build(&[coarse, detail], &[DVec2::new(1.0, 1.0)]);
        let detail_chunks = coords_for(&plan, 1);
        assert_eq!(detail_chunks, vec![ChunkCoord::new(LayerId(1), IVec3::ZERO)]);
        // Detail chunk (0,0) covers [0,50]; +pad 30 → [-30,80]. Coarse size 200:
        // floor(-30/200)=-1 .. floor(80/200)=0 → coarse chunks {-1,0} per axis = 4.
        let coarse_chunks = coords_for(&plan, 0);
        assert_eq!(coarse_chunks.len(), 4, "padded window should pull a 2x2 coarse block: {coarse_chunks:?}");
        assert!(coarse_chunks.contains(&ChunkCoord::new(LayerId(0), IVec3::new(-1, 0, -1))));
        assert!(coarse_chunks.contains(&ChunkCoord::new(LayerId(0), IVec3::new(0, 0, 0))));
    }

    /// Zero padding reduces to the bounds-only window.
    #[test]
    fn zero_padding_is_bounds_only() {
        let detail = meta(1, 100, vec![LayerDependency { on: LayerId(0), padding: 0.0 }], Some(1.0));
        let coarse = meta(0, 100, vec![], None);
        let plan = GenerationPlan::build(&[coarse, detail], &[DVec2::new(50.0, 50.0)]);
        // Detail chunk (0,0) covers [0,100]; same-size coarse → just coarse (0,0) (boundary at 100
        // includes chunk 1 too, since floor(100/100)=1). Accept the inclusive-boundary 2x2.
        let coarse_chunks = coords_for(&plan, 0);
        assert!(coarse_chunks.contains(&ChunkCoord::new(LayerId(0), IVec3::ZERO)));
        assert!(coarse_chunks.iter().all(|c| (0..=1).contains(&c.xyz.x) && (0..=1).contains(&c.xyz.z)));
    }

    /// Dependencies are emitted before dependents (topological order).
    #[test]
    fn dependencies_emitted_before_dependents() {
        let biome = meta(2, 50, vec![LayerDependency { on: LayerId(1), padding: 10.0 }], Some(1.0));
        let region = meta(1, 200, vec![LayerDependency { on: LayerId(0), padding: 20.0 }], None);
        let continent = meta(0, 1000, vec![], None);
        // Deliberately pass out of order; topo sort must still order deps first.
        let plan = GenerationPlan::build(&[biome, continent, region], &[DVec2::ZERO]);
        let order: Vec<u32> = plan.required.iter().map(|(l, _)| l.0).collect();
        let pos = |id: u32| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(0) < pos(1), "continent before region");
        assert!(pos(1) < pos(2), "region before biome");
    }

    /// Three-tier transitive padding: a biome chunk pulls padded region chunks, which pull padded
    /// continent chunks — the accumulation isn't dropped at depth.
    #[test]
    fn three_tier_cascade_pulls_all_levels() {
        let biome = meta(2, 50, vec![LayerDependency { on: LayerId(1), padding: 10.0 }], Some(1.0));
        let region = meta(1, 200, vec![LayerDependency { on: LayerId(0), padding: 20.0 }], None);
        let continent = meta(0, 1000, vec![], None);
        let plan = GenerationPlan::build(&[continent, region, biome], &[DVec2::new(5.0, 5.0)]);
        assert!(!coords_for(&plan, 0).is_empty(), "continent chunks must be pulled");
        assert!(!coords_for(&plan, 1).is_empty(), "region chunks must be pulled");
        assert!(!coords_for(&plan, 2).is_empty(), "biome chunks are the direct requirement");
        // The origin chunk of every tier is required (the focus sits in it).
        assert!(coords_for(&plan, 0).contains(&ChunkCoord::new(LayerId(0), IVec3::ZERO)));
        assert!(coords_for(&plan, 1).contains(&ChunkCoord::new(LayerId(1), IVec3::ZERO)));
        assert!(coords_for(&plan, 2).contains(&ChunkCoord::new(LayerId(2), IVec3::ZERO)));
    }

    /// The plan is a pure function of (metas, focus): rebuilding with the same inputs is identical,
    /// and it doesn't depend on the order layers are passed in.
    #[test]
    fn plan_is_deterministic_and_input_order_independent() {
        let a = meta(0, 256, vec![], Some(300.0));
        let b = meta(1, 64, vec![LayerDependency { on: LayerId(0), padding: 16.0 }], Some(100.0));
        let p1 = GenerationPlan::build(&[a.clone(), b.clone()], &[DVec2::new(10.0, -20.0)]);
        let p2 = GenerationPlan::build(&[b, a], &[DVec2::new(10.0, -20.0)]);
        assert_eq!(p1, p2, "plan must not depend on input layer order");
    }
}
