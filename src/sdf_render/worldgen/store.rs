//! `ArtifactStore` — a world-anchored, sparse, per-kind store of produced chunk artifacts.
//!
//! One store per (layer, artifact) — keyed by [`ChunkCoord`], holding `Arc<A>` so a produced artifact
//! is shared cheaply into both the GPU upload and the CPU `eval_primitive` surface query without
//! cloning the field data. Residency churns as the focus moves (insert on generate, evict on leaving
//! the ring), and the store tracks a **delta** since the last drain — which chunks were (re)produced
//! (`dirty`) and which were evicted (`dropped`) — so the GPU side uploads/drops only what changed,
//! exactly the incremental model `chunk::LiveChunkTables` uses for the brick directory.
//!
//! Generic over the artifact type, so 3D fields, classification, and instance streams each get a
//! store with zero new code (WORLD_GEN_PLAN §2.1 extensibility).

use std::collections::BTreeSet;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::artifact::Artifact;
use super::coord::ChunkCoord;

/// Sparse world-anchored store of one artifact kind. See module docs.
pub struct ArtifactStore<A: Artifact> {
    resident: FxHashMap<ChunkCoord, Arc<A>>,
    /// Coords (re)produced since the last [`drain_dirty`](Self::drain_dirty) — to upload. `BTreeSet`
    /// for deterministic (sorted) drain order, so uploads are reproducible run-to-run.
    dirty: BTreeSet<ChunkCoord>,
    /// Coords evicted since the last [`drain_dropped`](Self::drain_dropped) — to drop on the GPU.
    dropped: BTreeSet<ChunkCoord>,
}

// Manual `Default` so we don't require `A: Default` (the artifact type has no default).
impl<A: Artifact> Default for ArtifactStore<A> {
    fn default() -> Self {
        Self {
            resident: FxHashMap::default(),
            dirty: BTreeSet::new(),
            dropped: BTreeSet::new(),
        }
    }
}

impl<A: Artifact> ArtifactStore<A> {
    pub fn new() -> Self {
        Self::default()
    }

    /// The resident artifact at `coord`, if any.
    #[inline]
    pub fn get(&self, coord: ChunkCoord) -> Option<&Arc<A>> {
        self.resident.get(&coord)
    }

    #[inline]
    pub fn contains(&self, coord: ChunkCoord) -> bool {
        self.resident.contains_key(&coord)
    }

    /// Number of resident chunks.
    #[inline]
    pub fn len(&self) -> usize {
        self.resident.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.resident.is_empty()
    }

    /// Insert (or replace) the artifact at `coord`, marking it dirty (to be uploaded). A coord that
    /// was pending-dropped this cycle is un-dropped (re-produced before the drop was applied).
    pub fn insert(&mut self, coord: ChunkCoord, artifact: Arc<A>) {
        self.resident.insert(coord, artifact);
        self.dropped.remove(&coord);
        self.dirty.insert(coord);
    }

    /// Evict `coord` if resident, marking it dropped (to be removed on the GPU). A pending-dirty
    /// upload for it is cancelled (it's leaving the ring; no point uploading then dropping).
    pub fn evict(&mut self, coord: ChunkCoord) -> bool {
        let was = self.resident.remove(&coord).is_some();
        if was {
            self.dirty.remove(&coord);
            self.dropped.insert(coord);
        }
        was
    }

    /// Evict every resident chunk for which `keep` returns `false` (rolling residency around a moving
    /// focus). Returns the number evicted.
    pub fn retain(&mut self, mut keep: impl FnMut(ChunkCoord) -> bool) -> usize {
        let to_evict: Vec<ChunkCoord> = self
            .resident
            .keys()
            .copied()
            .filter(|&c| !keep(c))
            .collect();
        for c in &to_evict {
            self.evict(*c);
        }
        to_evict.len()
    }

    /// Iterate resident coords (unordered) — for debug overlays / stats.
    pub fn resident_coords(&self) -> impl Iterator<Item = ChunkCoord> + '_ {
        self.resident.keys().copied()
    }

    /// Take the (re)produced-since-last-drain set, in sorted order, with their artifacts — the upload
    /// delta. Clears the dirty set. (Coords are re-fetched from `resident` so a coord that was dirtied
    /// then evicted in the same cycle is skipped — eviction already removed it from `dirty`.)
    pub fn drain_dirty(&mut self) -> Vec<(ChunkCoord, Arc<A>)> {
        let out: Vec<(ChunkCoord, Arc<A>)> = self
            .dirty
            .iter()
            .filter_map(|c| self.resident.get(c).map(|a| (*c, Arc::clone(a))))
            .collect();
        self.dirty.clear();
        out
    }

    /// Take the evicted-since-last-drain set, in sorted order — the GPU drop delta. Clears it.
    pub fn drain_dropped(&mut self) -> Vec<ChunkCoord> {
        let out: Vec<ChunkCoord> = self.dropped.iter().copied().collect();
        self.dropped.clear();
        out
    }

    /// Whether any delta is pending (dirty or dropped) — to gate an upload extract.
    pub fn has_delta(&self) -> bool {
        !self.dirty.is_empty() || !self.dropped.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::super::artifact::{ArtifactKind, ScalarField2D};
    use super::super::coord::{ChunkCoord, ChunkSize, LayerId};
    use super::*;
    use bevy::math::IVec3;

    fn c(x: i32, z: i32) -> ChunkCoord {
        ChunkCoord::new(LayerId(0), IVec3::new(x, 0, z))
    }

    fn field(x: i32, z: i32) -> Arc<ScalarField2D> {
        Arc::new(ScalarField2D::zeroed(c(x, z), ChunkSize::new(16), 4))
    }

    #[test]
    fn insert_get_len() {
        let mut s: ArtifactStore<ScalarField2D> = ArtifactStore::new();
        assert!(s.is_empty());
        s.insert(c(0, 0), field(0, 0));
        s.insert(c(1, 0), field(1, 0));
        assert_eq!(s.len(), 2);
        assert!(s.contains(c(0, 0)));
        assert!(s.get(c(1, 0)).is_some());
        assert!(s.get(c(9, 9)).is_none());
        // The artifact kind is reachable generically.
        assert_eq!(ScalarField2D::kind(), ArtifactKind::ScalarField2D);
    }

    #[test]
    fn dirty_delta_drains_once_in_sorted_order() {
        let mut s: ArtifactStore<ScalarField2D> = ArtifactStore::new();
        s.insert(c(2, 0), field(2, 0));
        s.insert(c(-1, 0), field(-1, 0));
        s.insert(c(0, 5), field(0, 5));
        assert!(s.has_delta());
        let drained: Vec<ChunkCoord> = s.drain_dirty().into_iter().map(|(k, _)| k).collect();
        // Sorted: (-1,0,0) < (0,0,5) < (2,0,0) by the ChunkCoord total order.
        assert_eq!(drained, vec![c(-1, 0), c(0, 5), c(2, 0)]);
        // Drained once — a second drain is empty until something new is inserted.
        assert!(s.drain_dirty().is_empty());
        assert!(!s.has_delta());
    }

    #[test]
    fn evict_marks_dropped_and_cancels_pending_dirty() {
        let mut s: ArtifactStore<ScalarField2D> = ArtifactStore::new();
        s.insert(c(0, 0), field(0, 0));
        // Evict before draining dirty → the dirty upload is cancelled, a drop is queued instead.
        assert!(s.evict(c(0, 0)));
        assert!(!s.contains(c(0, 0)));
        assert!(s.drain_dirty().is_empty(), "evicted coord must not appear in the upload delta");
        assert_eq!(s.drain_dropped(), vec![c(0, 0)]);
        // Evicting an absent coord is a no-op.
        assert!(!s.evict(c(0, 0)));
        assert!(s.drain_dropped().is_empty());
    }

    #[test]
    fn reinsert_after_pending_drop_undrops() {
        let mut s: ArtifactStore<ScalarField2D> = ArtifactStore::new();
        s.insert(c(3, 3), field(3, 3));
        s.drain_dirty(); // clear initial dirty
        s.evict(c(3, 3)); // queues a drop
        s.insert(c(3, 3), field(3, 3)); // re-produced before the drop was applied
        assert!(s.contains(c(3, 3)));
        assert!(s.drain_dropped().is_empty(), "re-inserted coord must cancel its pending drop");
        assert_eq!(s.drain_dirty().len(), 1, "re-inserted coord is dirty again");
    }

    #[test]
    fn retain_evicts_outside_focus() {
        let mut s: ArtifactStore<ScalarField2D> = ArtifactStore::new();
        for x in -2..=2 {
            s.insert(c(x, 0), field(x, 0));
        }
        s.drain_dirty();
        // Keep |x| <= 1 (focus shrank).
        let evicted = s.retain(|coord| coord.xyz.x.abs() <= 1);
        assert_eq!(evicted, 2);
        assert_eq!(s.len(), 3);
        assert!(!s.contains(c(2, 0)) && !s.contains(c(-2, 0)));
        // The two evictions are queued as drops, in sorted order.
        assert_eq!(s.drain_dropped(), vec![c(-2, 0), c(2, 0)]);
    }
}
