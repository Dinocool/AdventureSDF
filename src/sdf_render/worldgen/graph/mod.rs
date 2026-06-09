//! The worldgen **field node-graph engine**: a DAG of nodes that each compute a [`Field`] (value +
//! analytic world-XZ gradient) at a point, so the whole graph forward-mode-autodiffs to the terrain
//! `(height, dh_dx, dh_dz)`. Biomes (climate axes, placement, per-biome shape) are authored as these
//! graphs and serialized to RON; the editor composes them visually. See the plan + the
//! `worldgen-biome-node-graph` memory.
//!
//! Bit-portable + deterministic like [`super::noise`] (f64 basic ops only). The engine is built up in
//! staged increments: [`field`] (the dual-number value) first; nodes / graph eval / RON next.

pub mod field;
pub mod node;

pub use field::Field;
pub use node::{FbmAxis, Graph, GraphError, Node, NodeKind};
