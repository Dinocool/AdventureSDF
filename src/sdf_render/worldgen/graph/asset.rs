//! `GraphAsset` — a terrain biome [`Graph`] as a first-class soul-engine resource, loaded/hot-reloaded/
//! saved through the SAME pipeline as materials (`crate::assets::material`): a `bevy::asset::Asset` with
//! an [`AssetLoader`] (RON via `ron::de::from_bytes`) plus the custom [`crate::assets::Asset`] trait
//! (gives `.save()` to RON for free, for the editor). On-disk under `assets/worldgen/*.graph.ron`.
//!
//! The asset carries the authored [`Graph`]; a system republishes the active graph into the
//! `WorldGraph` resource (an `Arc<Graph>`) that the pure `sample_world` path samples — mirroring how
//! materials compile into the `MaterialAssetTable` the bake consumes (the GPU/CPU split: the asset is
//! the editable source, the `Arc<Graph>` is the hot-path form).

use bevy::asset::{AssetLoader, LoadContext, io::Reader};
use bevy::prelude::*;

use super::node::Graph;

/// A biome terrain graph resource. Editable/savable RON; loaded via [`GraphAssetLoader`].
#[derive(Asset, Reflect, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GraphAsset {
    /// The authored field node-graph (produces `(height, dh_dx, dh_dz)` at a world point).
    pub graph: Graph,
}

impl crate::assets::Asset for GraphAsset {
    const EXTENSION: &'static str = "graph.ron";
}

/// Loads `*.graph.ron` into a [`GraphAsset`] — plain RON deserialization (a concrete serde type, like
/// `MaterialAssetLoader`; no reflection registry needed at load time).
#[derive(Default, bevy::reflect::TypePath)]
pub struct GraphAssetLoader;

/// Errors surfaced while loading a graph resource.
#[derive(Debug)]
pub enum GraphLoadError {
    Io(std::io::Error),
    Ron(ron::error::SpannedError),
}

impl std::fmt::Display for GraphLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphLoadError::Io(e) => write!(f, "graph io: {e}"),
            GraphLoadError::Ron(e) => write!(f, "graph ron: {e}"),
        }
    }
}

impl std::error::Error for GraphLoadError {}

impl From<std::io::Error> for GraphLoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<ron::error::SpannedError> for GraphLoadError {
    fn from(e: ron::error::SpannedError) -> Self {
        Self::Ron(e)
    }
}

impl AssetLoader for GraphAssetLoader {
    type Asset = GraphAsset;
    type Settings = ();
    type Error = GraphLoadError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _ctx: &mut LoadContext<'_>,
    ) -> Result<GraphAsset, GraphLoadError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let asset = ron::de::from_bytes::<GraphAsset>(&bytes)?;
        Ok(asset)
    }

    fn extensions(&self) -> &[&str] {
        // Bevy matches on the final extension; `.graph.ron` ends in `ron`, so claim `ron` + rely on the
        // `worldgen/` directory (same convention as materials).
        &["graph.ron", "ron"]
    }
}

#[cfg(test)]
mod tests {
    use super::super::preset::mountains_plains_graph;
    use super::*;

    /// The shipped `assets/worldgen/*.graph.ron` files must parse AND equal their Rust presets — guards
    /// the on-disk assets against drift from the preset builders (and that they're valid RON).
    #[test]
    fn shipped_graph_ron_files_match_presets() {
        use super::super::node::FbmAxis;
        use super::super::preset::{default_terrain_graph, mountains_plains_graph};

        let default_ron = include_str!("../../../../assets/worldgen/default.graph.ron");
        let default: GraphAsset = ron::de::from_str(default_ron).expect("parse default.graph.ron");
        let carrier = FbmAxis { octaves: 6, base_freq: 1.0 / 1536.0, lacunarity: 2.0, gain: 0.5, amplitude: 280.0, seed_salt: 0 };
        assert_eq!(default.graph, default_terrain_graph(carrier, 0.5, 280.0 * 1.96875, 0.0), "default.graph.ron drifted from preset");

        let mtn_ron = include_str!("../../../../assets/worldgen/mountains_plains.graph.ron");
        let mtn: GraphAsset = ron::de::from_str(mtn_ron).expect("parse mountains_plains.graph.ron");
        assert_eq!(
            mtn.graph,
            mountains_plains_graph(super::super::preset::MOUNTAINS_PLAINS_AMPLITUDE),
            "mountains_plains.graph.ron drifted from preset"
        );
        mtn.graph.validate().expect("shipped graph valid");
    }

    #[test]
    fn graph_asset_ron_round_trips() {
        let a = GraphAsset { graph: mountains_plains_graph(280.0) };
        let s = ron::ser::to_string(&a).expect("serialize");
        let back: GraphAsset = ron::de::from_str(&s).expect("deserialize");
        assert_eq!(a, back);
    }

    /// Prints the shipped preset graphs as RON — run to (re)generate `assets/worldgen/*.graph.ron`:
    /// `cargo test --lib print_preset_graphs_ron -- --ignored --nocapture`.
    #[test]
    #[ignore = "prints RON for the shipped asset files"]
    fn print_preset_graphs_ron() {
        use super::super::preset::{default_terrain_graph, mountains_plains_graph};
        use super::super::node::FbmAxis;
        let pretty = ron::ser::PrettyConfig::new();
        let carrier = FbmAxis { octaves: 6, base_freq: 1.0 / 1536.0, lacunarity: 2.0, gain: 0.5, amplitude: 280.0, seed_salt: 0 };
        let default = GraphAsset { graph: default_terrain_graph(carrier, 0.5, 280.0 * 1.96875, 0.0) };
        let mtn = GraphAsset { graph: mountains_plains_graph(super::super::preset::MOUNTAINS_PLAINS_AMPLITUDE) };
        eprintln!("// === default.graph.ron ===\n{}", ron::ser::to_string_pretty(&default, pretty.clone()).unwrap());
        eprintln!("// === mountains_plains.graph.ron ===\n{}", ron::ser::to_string_pretty(&mtn, pretty).unwrap());
    }
}
