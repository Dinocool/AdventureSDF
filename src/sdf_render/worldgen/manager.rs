//! `LayerManager` — owns the layer stack + artifact stores and rolls generation around the focus.
//!
//! The core is a synchronous, deterministic [`update`](LayerManager::update): build the
//! [`GenerationPlan`] for the focus, evict residency that left the plan, and generate up to a budget
//! of newly-required chunks (in dependency order). It is a pure function of `(focus, seed, params)`
//! and the store's prior residency, so it unit-tests without a Bevy app; the thin `WorldGenPlugin`
//! systems just feed it the camera focus and forward the store delta to the GPU upload.
//!
//! Generation runs sequentially here — each chunk is cheap (fBm over a 65² grid) and the per-update
//! budget bounds the cost. Because layers are *pure* `f(coord, seed)`, parallelizing a batch over
//! `ComputeTaskPool` later is a drop-in change (no ordering/determinism impact); the slice keeps it
//! sequential for testability and simplicity.

use std::collections::BTreeSet;
use std::sync::Arc;

use bevy::math::DVec2;
use bevy::prelude::Resource;

use super::artifact::ScalarField2D;
use super::coord::{ChunkCoord, LayerId};
use super::layer::{GenCtx, GenOutput, Layer};
use super::layers::height::{HeightLayer, HeightParams};
use super::plan::{GenerationPlan, LayerMeta};
use super::store::ArtifactStore;

/// Default newly-required chunks generated per [`update`](LayerManager::update) — bounds a focus-jump
/// spike; the rest stream in over subsequent updates (coarser resident LODs cover the gap meanwhile).
pub const DEFAULT_GEN_BUDGET: usize = 8;

/// Owns the worldgen layer stack + artifact stores. A Bevy `Resource`; the slice has one layer
/// (height) and one store, built so adding layers/stores is additive.
#[derive(Resource)]
pub struct LayerManager {
    layers: Vec<Box<dyn Layer>>,
    metas: Vec<LayerMeta>,
    /// The height-field store (slice's only artifact kind). Future kinds get sibling stores.
    height: ArtifactStore<ScalarField2D>,
    seed: u64,
    /// Newly-required chunks generated per update.
    pub budget: usize,
}

impl LayerManager {
    /// Build the Phase-1 slice stack: a single height layer required within `radius` metres of the
    /// focus.
    pub fn new_slice(seed: u64, params: HeightParams, radius: f64) -> Self {
        let layer = HeightLayer::new(LayerId(0), params);
        let metas = vec![LayerMeta {
            id: layer.id(),
            size: layer.chunk_size(),
            deps: layer.dependencies().to_vec(),
            direct_radius: Some(radius),
        }];
        Self { layers: vec![Box::new(layer)], metas, height: ArtifactStore::new(), seed, budget: DEFAULT_GEN_BUDGET }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Read access to the height store (debug/stats).
    pub fn height_store(&self) -> &ArtifactStore<ScalarField2D> {
        &self.height
    }

    /// Mutable access to the height store — the GPU upload drains its dirty/dropped delta here.
    pub fn height_store_mut(&mut self) -> &mut ArtifactStore<ScalarField2D> {
        &mut self.height
    }

    /// Whether all currently-required chunks for `focus` are resident (no generation pending) — lets
    /// the driving system skip work once the ring is full and the focus hasn't moved.
    pub fn is_settled(&self, focus_xz: DVec2) -> bool {
        let plan = GenerationPlan::build(&self.metas, &[focus_xz]);
        plan.iter().all(|c| self.height.contains(c))
    }

    /// Replace the height layer's params (editor tweak) and clear residency so the world regenerates
    /// with the new params. Returns the dropped count (for logging).
    pub fn set_height_params(&mut self, params: HeightParams) -> usize {
        // Rebuild the layer in place (keep its id/tier).
        let id = self.layers[0].id();
        self.layers[0] = Box::new(HeightLayer::new(id, params));
        // Evict everything → next update regenerates from the new params (queues GPU drops).
        self.height.retain(|_| false)
    }

    /// Roll generation around `focus_xz`: evict residency outside the plan, then generate up to
    /// `budget` newly-required chunks (dependency order). Returns `true` if the store has a pending
    /// delta to upload (anything generated or dropped this call).
    pub fn update(&mut self, focus_xz: DVec2) -> bool {
        let plan = GenerationPlan::build(&self.metas, &[focus_xz]);

        // Evict height residency that left the plan (rolling destroy). Multi-store later: one retain
        // per store keyed by that layer's required set.
        let required: BTreeSet<ChunkCoord> = plan
            .required
            .iter()
            .filter(|(l, _)| *l == LayerId(0))
            .flat_map(|(_, cs)| cs.iter().copied())
            .collect();
        self.height.retain(|c| required.contains(&c));

        // Generate missing chunks in dependency order, up to budget.
        let mut made = 0usize;
        'outer: for (layer_id, coords) in &plan.required {
            for &c in coords {
                if made >= self.budget {
                    break 'outer;
                }
                if self.store_contains(*layer_id, c) {
                    continue;
                }
                if let Some(field) = self.generate_height_chunk(c) {
                    self.height.insert(c, Arc::new(field));
                    made += 1;
                }
            }
        }
        self.height.has_delta()
    }

    /// True if the relevant store already holds `c`. (Slice: only the height store.)
    fn store_contains(&self, layer: LayerId, c: ChunkCoord) -> bool {
        layer == LayerId(0) && self.height.contains(c)
    }

    /// Generate one height chunk via its layer (pure). `None` if the layer didn't produce the
    /// expected artifact (shouldn't happen for a correct layer).
    fn generate_height_chunk(&self, c: ChunkCoord) -> Option<ScalarField2D> {
        let layer = self.layers.iter().find(|l| l.id() == c.layer)?;
        let ctx = GenCtx { coord: c, seed: self.seed, size: layer.chunk_size() };
        let mut out = GenOutput::default();
        layer.generate(&ctx, &mut out);
        Arc::try_unwrap(out.take::<ScalarField2D>(HeightLayer::OUTPUT)?).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::layers::height::HEIGHT_CHUNK_CELLS;

    fn mgr() -> LayerManager {
        // Radius ≈ 1.5 chunks so the focus pulls a small, predictable window.
        let r = HEIGHT_CHUNK_CELLS as f64 * 1.5;
        let mut m = LayerManager::new_slice(123, HeightParams::default(), r);
        m.budget = 1000; // generate all required per update in tests
        m
    }

    #[test]
    fn update_populates_window_and_settles() {
        let mut m = mgr();
        assert!(!m.is_settled(DVec2::ZERO));
        let delta = m.update(DVec2::ZERO);
        assert!(delta, "first update produces a store delta");
        assert!(!m.height_store().is_empty());
        assert!(m.is_settled(DVec2::ZERO), "with a high budget the window fills in one update");
        // The origin chunk is resident.
        assert!(m.height_store().contains(ChunkCoord::new(LayerId(0), bevy::math::IVec3::ZERO)));
    }

    #[test]
    fn moving_focus_evicts_old_and_creates_new() {
        let mut m = mgr();
        m.update(DVec2::ZERO);
        let n0 = m.height_store().len();
        assert!(n0 > 0);
        // Jump far away (many chunks over) — the old window must evict, a new one generate.
        let far = DVec2::splat(HEIGHT_CHUNK_CELLS as f64 * 50.0);
        m.update(far);
        assert!(!m.height_store().contains(ChunkCoord::new(LayerId(0), bevy::math::IVec3::ZERO)),
            "origin chunk evicted after focus jumped far away");
        assert!(m.is_settled(far), "new window fully resident");
        // Residency count is stable (same-size window).
        assert_eq!(m.height_store().len(), n0, "window size is focus-independent");
    }

    #[test]
    fn budget_limits_generation_per_update() {
        let r = HEIGHT_CHUNK_CELLS as f64 * 3.0;
        let mut m = LayerManager::new_slice(1, HeightParams::default(), r);
        m.budget = 2;
        m.update(DVec2::ZERO);
        // Only `budget` chunks generated in the first tick.
        assert_eq!(m.height_store().len(), 2);
        assert!(!m.is_settled(DVec2::ZERO));
        // Subsequent updates stream the rest in.
        for _ in 0..100 {
            if m.is_settled(DVec2::ZERO) {
                break;
            }
            m.update(DVec2::ZERO);
        }
        assert!(m.is_settled(DVec2::ZERO), "repeated budgeted updates eventually fill the window");
    }

    #[test]
    fn set_params_clears_and_regenerates_different_terrain() {
        let mut m = mgr();
        m.update(DVec2::ZERO);
        let origin = ChunkCoord::new(LayerId(0), bevy::math::IVec3::ZERO);
        let before = m.height_store().get(origin).unwrap().node(10, 10).height;
        // Drain the delta so we can observe the regen cleanly.
        m.height_store_mut().drain_dirty();

        let p = HeightParams { amplitude: 200.0, ..Default::default() }; // very different terrain
        let dropped = m.set_height_params(p);
        assert!(dropped > 0, "changing params evicts the old residency");
        assert!(m.height_store().is_empty());

        m.update(DVec2::ZERO);
        let after = m.height_store().get(origin).unwrap().node(10, 10).height;
        assert_ne!(before.to_bits(), after.to_bits(), "new params must change the terrain");
    }

    #[test]
    fn two_managers_same_seed_are_identical() {
        // Determinism across instances (the multiplayer property at the manager level).
        let mut a = mgr();
        let mut b = mgr();
        a.update(DVec2::new(40.0, -90.0));
        b.update(DVec2::new(40.0, -90.0));
        let origin = ChunkCoord::new(LayerId(0), bevy::math::IVec3::ZERO);
        let fa = a.height_store().get(origin).unwrap();
        let fb = b.height_store().get(origin).unwrap();
        for (na, nb) in fa.nodes.iter().zip(fb.nodes.iter()) {
            assert_eq!(na.height.to_bits(), nb.height.to_bits());
        }
    }
}
