//! Built-in terrain [`Graph`] presets — the default surface and the "mountains placed in plains"
//! biome graph that delivers the peaks+plains look. These are the graphs the editor will later load/
//! edit as RON; for now they're constructed in Rust and serialize round-trip (see tests). The graph
//! produces `(height, dh_dx, dh_dz)` DIRECTLY (the final node's [`Field`](super::Field)); erosion is a
//! future node (the per-point erosion stage needs the Hessian, which the first-order graph doesn't
//! carry — deferred per the roadmap).

use super::super::spline::Spline;
use super::node::{FbmAxis, Graph, Node, NodeKind};

/// Max nodes a terrain graph may have — bounds the per-sample stack scratch buffer (no heap alloc in
/// the bake hot path). Phase-1 presets are well under this.
pub const MAX_GRAPH_NODES: usize = 64;

/// The legacy-equivalent surface: carrier fBm → ridge fold → sea-level offset. Reproduces the pre-graph
/// `HeightLayer` surface (minus the separate erosion stage) — VALUE bit-for-bit (same op order via
/// commutativity), gradient to f64 round-off (autodiff vs the hand-derived `k`-scaling). Used as the
/// safe baseline + the `router`-off equivalent.
pub fn default_terrain_graph(carrier: FbmAxis, ridge: f64, amp_sum: f64, sea_level: f64) -> Graph {
    let nodes = vec![
        Node::source(NodeKind::Fbm(carrier)),                          // 0 carrier fBm
        Node::unary(NodeKind::Ridge { ridge, amp_sum }, 0),            // 1 ridge fold
        Node::unary(NodeKind::Offset(sea_level), 1),                   // 2 + sea level
    ];
    Graph { nodes, output: 2 }
}

/// "Broad plains + isolated towering mountains": a low-frequency CONTINENTALNESS axis gates (via a
/// smoothstep) a blend between a gentle plains surface and a tall ridged mountain surface — biome
/// PLACEMENT expressed as nodes. `amplitude` scales the mountain relief; the continentalness threshold
/// controls how much of the world is mountainous.
pub fn mountains_plains_graph(seed_amplitude: f64) -> Graph {
    // Continentalness: very low frequency (≈8 km), normalized ≈[-1,1] (amplitude 1).
    let cont = FbmAxis { octaves: 3, base_freq: 1.0 / 8192.0, lacunarity: 2.0, gain: 0.5, amplitude: 1.0, seed_salt: 0x00C0_0001 };
    // Plains carrier: gentle, low amplitude.
    let plains = FbmAxis { octaves: 4, base_freq: 1.0 / 768.0, lacunarity: 2.0, gain: 0.5, amplitude: 8.0, seed_salt: 0x00B1_0002 };
    // Mountain carrier: large amplitude, ridge-folded into sharp crests.
    let mtn = FbmAxis { octaves: 6, base_freq: 1.0 / 1536.0, lacunarity: 2.0, gain: 0.5, amplitude: seed_amplitude, seed_salt: 0x00B2_0003 };
    let mtn_amp_sum = seed_amplitude * (1.0 + 0.5 + 0.25 + 0.125 + 0.0625 + 0.03125);

    let nodes = vec![
        // Placement gate from continentalness: a curve shapes it, smoothstep selects high-continent land.
        Node::source(NodeKind::Fbm(cont)),                                                       // 0
        Node::unary(NodeKind::Curve(Spline::new(&[(-1.0, -1.0), (0.1, -0.6), (0.5, 0.4), (1.0, 1.0)])), 0), // 1
        Node::unary(NodeKind::Smoothstep { edge0: 0.1, edge1: 0.7 }, 1),                         // 2 gate∈[0,1]
        // Plains surface (gentle), low base elevation.
        Node::source(NodeKind::Fbm(plains)),                                                     // 3
        Node::unary(NodeKind::Offset(4.0), 3),                                                   // 4 plains
        // Mountain surface: ridged carrier, high base elevation.
        Node::source(NodeKind::Fbm(mtn)),                                                        // 5
        Node::unary(NodeKind::Ridge { ridge: 0.85, amp_sum: mtn_amp_sum }, 5),                   // 6
        Node::unary(NodeKind::Offset(60.0), 6),                                                  // 7 mountains
        // Placement: blend plains→mountains by the continentalness gate.
        Node::ternary(NodeKind::Mix, 4, 7, 2),                                                   // 8 output
    ];
    Graph { nodes, output: 8 }
}

#[cfg(test)]
mod tests {
    use super::super::Field;
    use super::*;
    use crate::sdf_render::worldgen::noise::{FbmParams, fbm_height_grad};

    fn carrier() -> FbmAxis {
        FbmAxis { octaves: 6, base_freq: 1.0 / 1536.0, lacunarity: 2.0, gain: 0.5, amplitude: 280.0, seed_salt: 0 }
    }

    #[test]
    fn presets_validate_and_fit_scratch() {
        let g = default_terrain_graph(carrier(), 0.5, 280.0 * 1.96875, 0.0);
        g.validate().unwrap();
        assert!(g.nodes.len() <= MAX_GRAPH_NODES);
        let m = mountains_plains_graph(280.0);
        m.validate().unwrap();
        assert!(m.nodes.len() <= MAX_GRAPH_NODES);
    }

    #[test]
    fn default_graph_value_matches_legacy_fbm_ridge() {
        // The default graph's VALUE must equal the legacy carrier-fBm + ridge-fold composition bit-for-bit
        // (the no-op baseline). Gradient is checked separately (autodiff) by the graph grad-vs-CD tests.
        let seed = 7u64;
        let ax = carrier();
        let ridge = 0.5f64;
        let amp_sum = ax.amplitude * (1.0 + 0.5 + 0.25 + 0.125 + 0.0625 + 0.03125);
        let sea = 12.0;
        let g = default_terrain_graph(ax, ridge, amp_sum, sea);
        let p = FbmParams {
            octaves: ax.octaves,
            base_freq: ax.base_freq,
            lacunarity: ax.lacunarity,
            gain: ax.gain,
            amplitude: ax.amplitude,
            seed: (seed as u32) ^ ((seed >> 32) as u32) ^ ax.seed_salt,
        };
        for &(wx, wz) in &[(0.0, 0.0), (123.5, -456.25), (-789.0, 1011.0), (5000.0, -3000.0)] {
            let (h, _, _) = fbm_height_grad(wx, wz, &p);
            // Legacy ridge fold (height.rs:carved_grad): h_base = h + ridge·((amp_sum − 2|h|) − h).
            let ah = if h < 0.0 { -h } else { h };
            let ridged = amp_sum - 2.0 * ah;
            let legacy = h + ridge * (ridged - h) + sea;
            let got = g.eval(wx, wz, seed).v;
            assert_eq!(got.to_bits(), legacy.to_bits(), "default graph value vs legacy at ({wx},{wz})");
        }
    }

    #[test]
    fn mountains_plains_grad_matches_central_difference() {
        let g = mountains_plains_graph(280.0);
        let e = 1e-3;
        for &(wx, wz) in &[(100.0, 200.0), (4000.0, -1000.0), (-2500.0, 3000.0)] {
            let f = g.eval(wx, wz, 9);
            let cdx = (g.eval(wx + e, wz, 9).v - g.eval(wx - e, wz, 9).v) / (2.0 * e);
            let cdz = (g.eval(wx, wz + e, 9).v - g.eval(wx, wz - e, 9).v) / (2.0 * e);
            assert!((f.dx - cdx).abs() < 1e-2, "∂x {} vs CD {cdx}", f.dx);
            assert!((f.dz - cdz).abs() < 1e-2, "∂z {} vs CD {cdz}", f.dz);
        }
    }

    #[test]
    fn mountains_plains_has_plains_and_peaks() {
        // Over a wide region: flat low plains AND tall peaks both appear (the look).
        let g = mountains_plains_graph(280.0);
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut flat = 0usize;
        let mut n = 0usize;
        let step = 256.0;
        for iz in -40..40 {
            for ix in -40..40 {
                let f = g.eval(ix as f64 * step, iz as f64 * step, 9);
                lo = lo.min(f.v);
                hi = hi.max(f.v);
                let slope = (f.dx * f.dx + f.dz * f.dz).sqrt();
                if slope < 0.05 {
                    flat += 1;
                }
                n += 1;
            }
        }
        assert!(hi > 200.0, "expected tall peaks, max height {hi}");
        assert!(lo < 30.0, "expected low plains, min height {lo}");
        assert!(flat * 100 / n >= 5, "expected ≥5% near-flat plains columns, got {}%", flat * 100 / n);
    }

    #[test]
    fn graph_ron_round_trips() {
        let g = mountains_plains_graph(280.0);
        let s = ron::ser::to_string(&g).expect("serialize");
        let back: Graph = ron::de::from_str(&s).expect("deserialize");
        assert_eq!(g, back, "graph must survive a RON round-trip");
        // And the deserialized graph evaluates identically.
        let a = g.eval(1234.0, -567.0, 3);
        let b = back.eval(1234.0, -567.0, 3);
        assert_eq!(a.v.to_bits(), b.v.to_bits());
        let _ = Field::constant(0.0);
    }
}
