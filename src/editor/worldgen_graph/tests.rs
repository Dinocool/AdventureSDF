use super::convert::world_biome_snarl;
use super::*;
use egui_snarl::{InPinId, OutPinId};

use crate::assets::Asset as _;
use crate::sdf_render::worldgen::graph::GraphAsset;
use crate::sdf_render::worldgen::graph::node::FbmAxis;
use crate::sdf_render::worldgen::graph::preset::{default_terrain_graph, mountains_plains_graph};

/// engine Graph → Snarl → engine Graph must round-trip to an EVALUATION-equivalent graph. (Not
/// structurally equal: `snarl_to_graph` topologically re-orders nodes, so indices differ — but the
/// DAG + params are preserved, so it evaluates bit-for-bit identically.)
#[test]
fn graph_snarl_round_trip() {
    for g in [
        mountains_plains_graph(700.0),
        default_terrain_graph(
            FbmAxis { octaves: 6, base_freq: 1.0 / 1536.0, lacunarity: 2.0, gain: 0.5, amplitude: 280.0, seed_salt: 0 },
            0.5,
            551.25,
            0.0,
        ),
    ] {
        let snarl = graph_to_snarl(&g);
        let back = snarl_to_graph(&snarl).expect("convert back");
        for &(x, z) in &[(0.0, 0.0), (123.0, -456.0), (2000.0, 1000.0), (-800.0, 300.0)] {
            let (a, b) = (g.eval(x, z, 7), back.eval(x, z, 7));
            assert_eq!(a.v.to_bits(), b.v.to_bits(), "value at ({x},{z}) after Snarl round-trip");
            assert_eq!(a.dx.to_bits(), b.dx.to_bits(), "∂x at ({x},{z}) after Snarl round-trip");
            assert_eq!(a.dz.to_bits(), b.dz.to_bits(), "∂z at ({x},{z}) after Snarl round-trip");
        }
    }
}

#[test]
fn missing_output_is_an_error() {
    let mut snarl = Snarl::new();
    snarl.insert_node(egui::pos2(0.0, 0.0), EdNode::Op(NodeKind::Const(1.0)));
    assert!(snarl_to_graph(&snarl).is_err());
}

// -- biome (nested sub-graph) inlining ----------------------------------------------------------
fn p() -> egui::Pos2 {
    egui::pos2(0.0, 0.0)
}
fn out(n: NodeId) -> OutPinId {
    OutPinId { node: n, output: 0 }
}
fn inn(n: NodeId, i: usize) -> InPinId {
    InPinId { node: n, input: i }
}

/// A biome wrapping a sub-graph must inline to a graph that evaluates bit-for-bit like the same
/// sub-graph placed flat (no biome) — biomes are pure authoring grouping.
#[test]
fn biome_inlines_to_flat_equivalent() {
    let axis = FbmAxis { octaves: 3, base_freq: 1.0 / 512.0, lacunarity: 2.0, gain: 0.5, amplitude: 100.0, seed_salt: 2 };

    let mut flat = Snarl::new();
    let f = flat.insert_node(p(), EdNode::Op(NodeKind::Fbm(axis)));
    let o = flat.insert_node(p(), EdNode::Output);
    flat.connect(out(f), inn(o, 0));
    let flat = snarl_to_graph(&flat).expect("flat");

    let mut sub = Snarl::new();
    let sf = sub.insert_node(p(), EdNode::Op(NodeKind::Fbm(axis)));
    let so = sub.insert_node(p(), EdNode::Output);
    sub.connect(out(sf), inn(so, 0));
    let mut top = Snarl::new();
    let b = top.insert_node(p(), EdNode::Biome { name: "Mountains".into(), graph: Box::new(sub) });
    let o = top.insert_node(p(), EdNode::Output);
    top.connect(out(b), inn(o, 0));
    let nested = snarl_to_graph(&top).expect("nested");

    for &(x, z) in &[(0.0, 0.0), (123.0, -456.0), (2000.0, 1000.0)] {
        let (a, c) = (flat.eval(x, z, 7), nested.eval(x, z, 7));
        assert_eq!(a.v.to_bits(), c.v.to_bits(), "value at ({x},{z})");
        assert_eq!(a.dx.to_bits(), c.dx.to_bits(), "∂x at ({x},{z})");
        assert_eq!(a.dz.to_bits(), c.dz.to_bits(), "∂z at ({x},{z})");
    }
}

/// A climate value wired into a biome pin must reach the biome's `Input` sentinel through inlining.
#[test]
fn biome_climate_input_is_piped() {
    let mut sub = Snarl::new();
    let inp = sub.insert_node(p(), EdNode::Input(0)); // continentalness
    let c5 = sub.insert_node(p(), EdNode::Op(NodeKind::Const(5.0)));
    let add = sub.insert_node(p(), EdNode::Op(NodeKind::Add));
    sub.connect(out(inp), inn(add, 0));
    sub.connect(out(c5), inn(add, 1));
    let so = sub.insert_node(p(), EdNode::Output);
    sub.connect(out(add), inn(so, 0));

    let mut top = Snarl::new();
    let c10 = top.insert_node(p(), EdNode::Op(NodeKind::Const(10.0)));
    let b = top.insert_node(p(), EdNode::Biome { name: "B".into(), graph: Box::new(sub) });
    top.connect(out(c10), inn(b, 0)); // feed continentalness pin
    let o = top.insert_node(p(), EdNode::Output);
    top.connect(out(b), inn(o, 0));

    let g = snarl_to_graph(&top).expect("piped");
    assert_eq!(g.eval(0.0, 0.0, 7).v, 15.0); // Input(0)=10 + Const 5
}

/// A hierarchical snarl (with a biome) must survive a RON round-trip and compile identically — this
/// is what persists the biome hierarchy across save/reload (`.worldgraph.ron`).
#[test]
fn biome_hierarchy_ron_round_trips() {
    let mut sub = Snarl::new();
    let inp = sub.insert_node(p(), EdNode::Input(0));
    let c = sub.insert_node(p(), EdNode::Op(NodeKind::Const(3.0)));
    let add = sub.insert_node(p(), EdNode::Op(NodeKind::Add));
    sub.connect(out(inp), inn(add, 0));
    sub.connect(out(c), inn(add, 1));
    let so = sub.insert_node(p(), EdNode::Output);
    sub.connect(out(add), inn(so, 0));
    let mut top = Snarl::new();
    let c10 = top.insert_node(p(), EdNode::Op(NodeKind::Const(10.0)));
    let b = top.insert_node(p(), EdNode::Biome { name: "Hills".into(), graph: Box::new(sub) });
    top.connect(out(c10), inn(b, 0));
    let o = top.insert_node(p(), EdNode::Output);
    top.connect(out(b), inn(o, 0));

    let s = ron::ser::to_string(&top).expect("serialize");
    let back: Snarl<EdNode> = ron::de::from_str(&s).expect("deserialize");
    let (g1, g2) = (snarl_to_graph(&top).unwrap(), snarl_to_graph(&back).unwrap());
    for &(x, z) in &[(0.0, 0.0), (123.0, -456.0)] {
        assert_eq!(g1.eval(x, z, 7).v.to_bits(), g2.eval(x, z, 7).v.to_bits());
    }
    assert_eq!(g2.eval(0.0, 0.0, 7).v, 13.0); // Input(0)=10 + Const 3
}

/// The default multi-biome world graph compiles, evaluates finite, and shows BOTH gentle (plains) and
/// tall (mountains) terrain over a region — the climate classifier actually places both biomes.
#[test]
fn biome_world_has_plains_and_mountains() {
    let g = snarl_to_graph(&world_biome_snarl()).expect("compile biome world");
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..40 {
        for j in 0..40 {
            let x = i as f64 * 450.0 - 9000.0;
            let z = j as f64 * 450.0 - 9000.0;
            let v = g.eval(x, z, 7).v;
            assert!(v.is_finite(), "non-finite at ({x},{z})");
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    assert!(hi - lo > 300.0, "expected plains+mountains spread, got {lo}..{hi}");
}

/// A biome `Input` that is used but its parent pin is unconnected is a hard error (no silent 0).
#[test]
fn unconnected_used_biome_input_errors() {
    let mut sub = Snarl::new();
    let inp = sub.insert_node(p(), EdNode::Input(0));
    let so = sub.insert_node(p(), EdNode::Output);
    sub.connect(out(inp), inn(so, 0));
    let mut top = Snarl::new();
    let b = top.insert_node(p(), EdNode::Biome { name: "B".into(), graph: Box::new(sub) });
    let o = top.insert_node(p(), EdNode::Output);
    top.connect(out(b), inn(o, 0)); // continentalness pin left unconnected
    assert!(snarl_to_graph(&top).is_err());
}

/// Regenerate the shipped default world assets from `world_biome_snarl` (run on purpose):
/// `cargo test --features editor write_world_biome_assets -- --ignored`.
#[test]
#[ignore]
fn write_world_biome_assets() {
    let snarl = world_biome_snarl();
    let g = snarl_to_graph(&snarl).expect("compile");
    (GraphAsset { graph: g })
        .save(std::path::Path::new("assets/worldgen/world.graph.ron"))
        .expect("save flat");
    let s = ron::ser::to_string_pretty(&snarl, ron::ser::PrettyConfig::default()).expect("ser hierarchy");
    std::fs::write("assets/worldgen/world.worldgraph.ron", s).expect("write hierarchy");
}

/// The shipped `world.graph.ron` must match the compiled `world_biome_snarl` (catches drift after the
/// default graph changes — re-run `write_world_biome_assets`).
#[test]
fn shipped_world_graph_matches_snarl() {
    let s = std::fs::read_to_string("assets/worldgen/world.graph.ron").expect("read shipped world graph");
    let shipped: GraphAsset = ron::de::from_str(&s).expect("parse shipped");
    let built = snarl_to_graph(&world_biome_snarl()).unwrap();
    for &(x, z) in &[(0.0, 0.0), (1234.0, -987.0), (5000.0, 5000.0)] {
        assert_eq!(
            shipped.graph.eval(x, z, 7).v.to_bits(),
            built.eval(x, z, 7).v.to_bits(),
            "shipped world.graph.ron is stale — re-run write_world_biome_assets"
        );
    }
}
