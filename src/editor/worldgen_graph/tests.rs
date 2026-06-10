use super::arrange::auto_arrange;
use super::compile::output_root;
use super::convert::{
    breadcrumb_names, load_editor_snarl, new_biome_subgraph, resolve_snarl, valid_depth, world_biome_snarl,
    worldgraph_path,
};
use super::preview::{PANEL_GPU_KEY, gpu_inline_key};
use super::*;
use egui::pos2;
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

// -- worldgraph_path -----------------------------------------------------------------------------

#[test]
fn worldgraph_path_strips_known_suffixes() {
    // `.graph.ron` stripped (the common case).
    assert_eq!(worldgraph_path("a/b/world.graph.ron"), "a/b/world.worldgraph.ron");
    // Plain `.ron` stripped.
    assert_eq!(worldgraph_path("a/b/world.ron"), "a/b/world.worldgraph.ron");
    // Neither suffix → appended whole.
    assert_eq!(worldgraph_path("a/b/world"), "a/b/world.worldgraph.ron");
    // `.graph.ron` takes precedence over the bare `.ron` (no double-strip).
    assert_eq!(worldgraph_path("x.graph.ron"), "x.worldgraph.ron");
}

// -- load_editor_snarl fallback chain ------------------------------------------------------------

/// A unique temp path under the test temp dir (TMP/TEMP), removed on drop, for the file-based tests.
struct TmpFile(std::path::PathBuf);
impl TmpFile {
    fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let pid = std::process::id();
        let n = N.fetch_add(1, Ordering::Relaxed);
        Self(std::env::temp_dir().join(format!("wg_test_{tag}_{pid}_{n}")))
    }
    fn path(&self) -> &std::path::Path {
        &self.0
    }
    fn str(&self) -> String {
        self.0.to_string_lossy().into_owned()
    }
    fn write(&self, contents: &str) {
        std::fs::write(&self.0, contents).expect("write temp file");
    }
    fn wg(&self) -> std::path::PathBuf {
        std::path::PathBuf::from(worldgraph_path(&self.str()))
    }
}
impl Drop for TmpFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
        let _ = std::fs::remove_file(self.wg());
    }
}

#[test]
fn load_editor_snarl_prefers_hierarchical() {
    // A valid hierarchical `.worldgraph.ron` (with a biome) is loaded in preference to the flat graph.
    let f = TmpFile::new("hier");
    let mut top = Snarl::new();
    let b = top.insert_node(p(), EdNode::Biome { name: "Hills".into(), graph: Box::new(new_biome_subgraph()) });
    let o = top.insert_node(p(), EdNode::Output);
    top.connect(out(b), inn(o, 0));
    let s = ron::ser::to_string(&top).unwrap();
    std::fs::write(f.wg(), s).expect("write hierarchical");
    let loaded = load_editor_snarl(&f.str());
    assert!(loaded.node_ids().any(|(_, n)| matches!(n, EdNode::Biome { .. })), "kept the biome hierarchy");
}

#[test]
fn load_editor_snarl_falls_back_to_flat() {
    // No `.worldgraph.ron`, but a valid flat `.graph.ron` → loaded (and re-snarled, so no biomes).
    let f = TmpFile::new("flat");
    let g = mountains_plains_graph(700.0);
    (GraphAsset { graph: g }).save(f.path()).expect("save flat");
    let loaded = load_editor_snarl(&f.str());
    // Round-trips to a compilable graph and has no biomes (flat).
    assert!(snarl_to_graph(&loaded).is_ok());
    assert!(!loaded.node_ids().any(|(_, n)| matches!(n, EdNode::Biome { .. })));
}

#[test]
fn load_editor_snarl_corrupt_falls_through_to_default() {
    // Both files present but CORRUPT → falls through to the built-in default world (which has biomes).
    let f = TmpFile::new("corrupt");
    std::fs::write(f.wg(), "this is not valid ron {{{").expect("write corrupt hier");
    f.write("also not ron )))");
    let loaded = load_editor_snarl(&f.str());
    let default = world_biome_snarl();
    // Same structure as the default: identical node count + both biomes present.
    assert_eq!(loaded.node_ids().count(), default.node_ids().count());
    assert_eq!(loaded.node_ids().filter(|(_, n)| matches!(n, EdNode::Biome { .. })).count(), 2);
}

#[test]
fn load_editor_snarl_missing_uses_default() {
    // Neither file exists → the built-in default world.
    let f = TmpFile::new("missing");
    let loaded = load_editor_snarl(&f.str());
    assert_eq!(loaded.node_ids().count(), world_biome_snarl().node_ids().count());
}

// -- nav helpers (valid_depth / resolve_snarl / breadcrumb_names) --------------------------------

/// A top snarl with one biome named `name`; returns (snarl, biome id).
fn top_with_biome(name: &str) -> (Snarl<EdNode>, NodeId) {
    let mut top = Snarl::new();
    let b = top.insert_node(p(), EdNode::Biome { name: name.into(), graph: Box::new(new_biome_subgraph()) });
    let o = top.insert_node(p(), EdNode::Output);
    top.connect(out(b), inn(o, 0));
    (top, b)
}

#[test]
fn valid_depth_truncates_stale_and_non_biome_paths() {
    let (mut top, b) = top_with_biome("B");
    // Empty path is always fully valid.
    assert_eq!(valid_depth(&top, &[]), 0);
    // A path through the real biome is valid to depth 1.
    assert_eq!(valid_depth(&top, &[b]), 1);
    // A stale id (never in the snarl) resolves to depth 0.
    let stale = top.insert_node(pos2(0.0, 0.0), EdNode::Op(NodeKind::Const(0.0)));
    assert_eq!(valid_depth(&top, &[stale]), 0, "a non-biome node isn't a navigable level");
    // A valid biome followed by a stale tail truncates after the biome.
    assert_eq!(valid_depth(&top, &[b, stale]), 1);
}

#[test]
fn resolve_snarl_none_on_stale_or_non_biome() {
    let (mut top, b) = top_with_biome("B");
    assert!(resolve_snarl(&top, &[]).is_some(), "empty nav resolves to the root");
    assert!(resolve_snarl(&top, &[b]).is_some(), "into the biome resolves");
    let other = top.insert_node(pos2(0.0, 0.0), EdNode::Op(NodeKind::Const(0.0)));
    assert!(resolve_snarl(&top, &[other]).is_none(), "non-biome node → None");
    assert!(resolve_snarl(&top, &[b, other]).is_none(), "stale tail → None");
}

#[test]
fn breadcrumb_names_reflects_rename() {
    let (mut top, b) = top_with_biome("Old");
    assert_eq!(breadcrumb_names(&top, &[b]), vec!["Old".to_string()]);
    // Rename the biome → the breadcrumb tracks it.
    if let EdNode::Biome { name, .. } = &mut top[b] {
        *name = "New".into();
    }
    assert_eq!(breadcrumb_names(&top, &[b]), vec!["New".to_string()]);
    // A non-biome / stale step stops the breadcrumb.
    assert!(breadcrumb_names(&top, &[]).is_empty());
}

// -- auto_arrange --------------------------------------------------------------------------------

/// A small chain `Const → Scale → Output` plus a stray `WorldX`, to exercise depth columns.
fn chain_snarl() -> Snarl<EdNode> {
    let mut s = Snarl::new();
    let c = s.insert_node(pos2(0.0, 0.0), EdNode::Op(NodeKind::Const(1.0)));
    let sc = s.insert_node(pos2(0.0, 0.0), EdNode::Op(NodeKind::Scale(2.0)));
    s.connect(out(c), inn(sc, 0));
    let o = s.insert_node(pos2(0.0, 0.0), EdNode::Output);
    s.connect(out(sc), inn(o, 0));
    s.insert_node(pos2(0.0, 0.0), EdNode::Op(NodeKind::WorldX)); // a leaf at depth 0
    s
}

fn positions(s: &Snarl<EdNode>) -> Vec<(NodeId, egui::Pos2)> {
    let mut v: Vec<_> = s.node_ids().map(|(id, _)| (id, s.get_node_info(id).unwrap().pos)).collect();
    v.sort_by_key(|(id, _)| *id);
    v
}

#[test]
fn auto_arrange_is_deterministic() {
    let body = std::collections::HashMap::new();
    let mut a = chain_snarl();
    let mut b = chain_snarl();
    auto_arrange(&mut a, &body);
    auto_arrange(&mut b, &body);
    assert_eq!(positions(&a), positions(&b), "same wiring + sizes ⇒ identical layout");
    // Idempotent: a second arrange doesn't move anything.
    let once = positions(&a);
    auto_arrange(&mut a, &body);
    assert_eq!(once, positions(&a));
}

#[test]
fn auto_arrange_columns_increase_with_depth() {
    let body = std::collections::HashMap::new();
    let mut s = chain_snarl();
    // Capture ids in chain order.
    let ids: Vec<NodeId> = s.node_ids().map(|(id, _)| id).collect();
    auto_arrange(&mut s, &body);
    let x = |id: NodeId| s.get_node_info(id).unwrap().pos.x;
    // ids[0]=Const(d0), ids[1]=Scale(d1), ids[2]=Output(d2), ids[3]=WorldX(d0).
    assert!(x(ids[0]) < x(ids[1]), "Scale is a column right of Const");
    assert!(x(ids[1]) < x(ids[2]), "Output is a column right of Scale");
    assert_eq!(x(ids[0]), x(ids[3]), "two depth-0 leaves share a column x");
}

#[test]
fn auto_arrange_tolerates_a_cycle() {
    // A 2-node cycle (A→B→A) must not hang the depth recursion (the cycle guard returns 0).
    let body = std::collections::HashMap::new();
    let mut s = Snarl::new();
    let a = s.insert_node(pos2(0.0, 0.0), EdNode::Op(NodeKind::Abs));
    let b = s.insert_node(pos2(0.0, 0.0), EdNode::Op(NodeKind::Neg));
    s.connect(out(a), inn(b, 0));
    s.connect(out(b), inn(a, 0));
    auto_arrange(&mut s, &body); // must return (no infinite loop)
    assert_eq!(s.node_ids().count(), 2);
}

// -- gpu_inline_key ------------------------------------------------------------------------------

#[test]
fn gpu_inline_key_is_stable_and_non_colliding() {
    let mut s = Snarl::new();
    let n = s.insert_node(pos2(0.0, 0.0), EdNode::Op(NodeKind::Const(0.0)));
    // Stable for the same (salt, node).
    assert_eq!(gpu_inline_key(42, n), gpu_inline_key(42, n));
    // The top bit is set on every inline key.
    let k = gpu_inline_key(42, n);
    assert!(k & (1u64 << 63) != 0, "inline keys have the high bit set");
    // …so they can never collide with the small pop-out ids (start at 1000) or the PANEL key (7).
    assert!(k > 1000);
    assert_ne!(k, PANEL_GPU_KEY);
    assert_ne!(k & (1u64 << 63), 0);
    // Different salt (nav level) ⇒ (almost surely) different key — and certainly not equal here.
    assert_ne!(gpu_inline_key(42, n), gpu_inline_key(43, n));
}

// -- output_root errors --------------------------------------------------------------------------

#[test]
fn output_root_distinguishes_duplicate_and_unwired() {
    // Duplicate Output → a specific "more than one Output" error.
    let mut dup = Snarl::new();
    dup.insert_node(p(), EdNode::Output);
    dup.insert_node(p(), EdNode::Output);
    let e = output_root(&dup).expect_err("duplicate Output is an error");
    assert!(e.contains("more than one Output"), "got: {e}");

    // Single Output but nothing wired into it → a distinct "no input wired" error.
    let mut unwired = Snarl::new();
    unwired.insert_node(p(), EdNode::Output);
    let e = output_root(&unwired).expect_err("unwired Output is an error");
    assert!(e.contains("no input wired"), "got: {e}");
    assert!(!e.contains("more than one"), "distinct from the duplicate error");

    // No Output at all → yet another distinct error.
    let mut none = Snarl::new();
    none.insert_node(p(), EdNode::Op(NodeKind::Const(0.0)));
    let e = output_root(&none).expect_err("missing Output is an error");
    assert!(e.contains("no Output node"), "got: {e}");
}
