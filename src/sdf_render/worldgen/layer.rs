//! The `Layer` trait and its generation context — the core of the LayerProcGen port.
//!
//! A layer is a pure, deterministic producer: given a chunk coord and the world seed it generates
//! that chunk's artifact(s), reading only its declared (padded) dependencies. Authoritative layers
//! must be cross-platform bit-deterministic (WORLD_GEN_PLAN §2.8); the [`generate`](Layer::generate)
//! contract forbids any frame/order/clock/global state so parallel dispatch is sound.
//!
//! Outputs are type-erased ([`GenOutput`]) so heterogeneous layers (height field, biome
//! classification, instance streams) share one trait; the `LayerManager` routes each produced
//! artifact to its typed [`super::store::ArtifactStore`] by name + downcast. The Phase-1 slice has a
//! single layer (`layers::height`) producing one `ScalarField2D`.

use std::any::Any;
use std::sync::Arc;

use super::artifact::{Artifact, ArtifactKind};
use super::coord::{Authority, ChunkCoord, ChunkSize, Dim, LayerId};

/// A read-dependency on a (coarser) layer with a padded read window (WORLD_GEN_PLAN §2.7).
/// Generating one chunk of the dependent forces every dependency chunk overlapping
/// `(this chunk's world bounds + padding)` to exist first — the contextual-generation mechanism.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerDependency {
    pub on: LayerId,
    /// Extra world metres read beyond the dependent chunk's bounds on every side.
    pub padding: f64,
}

/// A declared output of a layer: a stable name + its artifact kind. Consumers/visualizers and the
/// stores key on `(LayerId, name)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArtifactDecl {
    pub name: &'static str,
    pub kind: ArtifactKind,
}

/// Context handed to [`Layer::generate`] for one chunk. A struct (not bare params) so dependency
/// views and other inputs can be added later without changing the trait method signature.
pub struct GenCtx {
    /// The chunk being generated (on this layer's lattice).
    pub coord: ChunkCoord,
    /// The world seed (folded with each layer's salt inside the layer).
    pub seed: u64,
    /// This layer's chunk size (its tier).
    pub size: ChunkSize,
    // Future: `deps: DependencyView` exposing the padded, already-resident dependency artifacts.
}

/// Type-erased sink for a layer's produced artifacts. Each is stored as `Arc<dyn Any + Send + Sync>`
/// tagged by its declared name; the manager downcasts to the concrete type for its store.
#[derive(Default)]
pub struct GenOutput {
    pub artifacts: Vec<(&'static str, Arc<dyn Any + Send + Sync>)>,
}

impl GenOutput {
    /// Produce artifact `a` under `name` (must match one of the layer's [`ArtifactDecl`]s).
    pub fn produce<A: Artifact>(&mut self, name: &'static str, a: A) {
        self.artifacts.push((name, Arc::new(a)));
    }

    /// Take the produced artifact named `name`, downcast to `A`, if present. Used by the manager to
    /// route an output into its typed store.
    pub fn take<A: Artifact>(&mut self, name: &'static str) -> Option<Arc<A>> {
        let pos = self.artifacts.iter().position(|(n, _)| *n == name)?;
        let (_, any) = self.artifacts.remove(pos);
        any.downcast::<A>().ok()
    }
}

/// One generation layer. Object-safe so layers are boxed in the manager's stack.
pub trait Layer: Send + Sync + 'static {
    /// Stable identity within the recipe's stack.
    fn id(&self) -> LayerId;
    /// Chunk size = tier (bigger = higher abstraction).
    fn chunk_size(&self) -> ChunkSize;
    /// Lattice dimensionality.
    fn dimensionality(&self) -> Dim;
    /// CPU authority class (authoritative ⇒ must be bit-deterministic).
    fn authority(&self) -> Authority;
    /// Padded dependencies on coarser layers (empty for a root layer like height).
    fn dependencies(&self) -> &[LayerDependency] {
        &[]
    }
    /// Declared named outputs + kinds.
    fn produces(&self) -> &[ArtifactDecl];
    /// Generate `ctx.coord`'s artifact(s) into `out`. MUST be a pure function of
    /// `(ctx.coord, ctx.seed)` and the (future) dependency reads — no frame/order/clock/global state.
    fn generate(&self, ctx: &GenCtx, out: &mut GenOutput);
}

#[cfg(test)]
mod tests {
    use super::super::artifact::ScalarField2D;
    use super::*;
    use bevy::math::IVec3;

    /// A trivial layer for trait-mechanics tests: produces a zeroed height field, no deps.
    struct DummyLayer;
    impl Layer for DummyLayer {
        fn id(&self) -> LayerId {
            LayerId(0)
        }
        fn chunk_size(&self) -> ChunkSize {
            ChunkSize::new(32)
        }
        fn dimensionality(&self) -> Dim {
            Dim::D2
        }
        fn authority(&self) -> Authority {
            Authority::Authoritative
        }
        fn produces(&self) -> &[ArtifactDecl] {
            &[ArtifactDecl { name: "height", kind: ArtifactKind::ScalarField2D }]
        }
        fn generate(&self, ctx: &GenCtx, out: &mut GenOutput) {
            out.produce("height", ScalarField2D::zeroed(ctx.coord, ctx.size, 4));
        }
    }

    #[test]
    fn generate_produces_downcastable_artifact() {
        let layer = DummyLayer;
        assert!(layer.dependencies().is_empty());
        let ctx = GenCtx {
            coord: ChunkCoord::new(layer.id(), IVec3::ZERO),
            seed: 42,
            size: layer.chunk_size(),
        };
        let mut out = GenOutput::default();
        layer.generate(&ctx, &mut out);
        assert_eq!(out.artifacts.len(), 1);
        // Routes by name + downcast to the concrete type.
        let field = out.take::<ScalarField2D>("height").expect("height artifact present + correct type");
        assert_eq!(field.res, 4);
        // Taken once.
        assert!(out.take::<ScalarField2D>("height").is_none());
    }
}
