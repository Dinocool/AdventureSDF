//! Auto-arrange: lay the shown snarl out left→right by dependency depth, packing columns/rows by each
//! node's real measured size so preview-laden nodes don't overlap.

use bevy_egui::egui;
use egui_snarl::{NodeId, Snarl};

use super::node::input_label;
use super::preview::DEFAULT_PREVIEW_PX;
use super::{CLIMATE_INPUTS, EdNode};

/// Lay the graph out left→right by dependency depth: each node's column = the longest input-chain to a
/// leaf, rows stack within a column. Columns are spaced by their widest node and rows by each node's real
/// height (measured last frame in `body_size`), so preview-laden nodes don't overlap. Pure function of
/// the wiring + measured sizes ⇒ stable + readable.
pub(super) fn auto_arrange(snarl: &mut Snarl<EdNode>, body_size: &std::collections::HashMap<NodeId, egui::Vec2>) {
    use std::collections::{HashMap, HashSet};
    const GAP_X: f32 = 80.0;
    const GAP_Y: f32 = 44.0;
    const HEADER: f32 = 40.0; // title bar
    const PIN_ROW: f32 = 26.0; // per input/output pin row
    const FRAME: f32 = 34.0; // node frame margins (top+bottom / left+right)
    const OUT_PIN: f32 = 18.0; // output-pin column on the right

    // Upstream nodes feeding each node (over all input slots).
    let mut up: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for (out, inp) in snarl.wires() {
        up.entry(inp.node).or_default().push(out.node);
    }

    fn depth(
        id: NodeId,
        up: &HashMap<NodeId, Vec<NodeId>>,
        memo: &mut HashMap<NodeId, i32>,
        on_stack: &mut HashSet<NodeId>,
    ) -> i32 {
        if let Some(&d) = memo.get(&id) {
            return d;
        }
        if !on_stack.insert(id) {
            return 0; // cycle guard (validation rejects cycles elsewhere)
        }
        let d = match up.get(&id) {
            Some(parents) if !parents.is_empty() => {
                parents.iter().map(|&p| depth(p, up, memo, on_stack)).max().unwrap_or(-1) + 1
            }
            _ => 0,
        };
        on_stack.remove(&id);
        memo.insert(id, d);
        d
    }

    let mut ids: Vec<NodeId> = snarl.node_ids().map(|(id, _)| id).collect();
    ids.sort(); // stable order within a column
    let mut memo = HashMap::new();
    let mut on_stack = HashSet::new();

    // Depth + estimated full node size per node (body measured last frame + header/pin/frame allowance).
    let mut depth_of: HashMap<NodeId, i32> = HashMap::new();
    let mut size_of: HashMap<NodeId, (f32, f32)> = HashMap::new();
    let mut max_depth = 0i32;
    for &id in &ids {
        let d = depth(id, &up, &mut memo, &mut on_stack);
        depth_of.insert(id, d);
        max_depth = max_depth.max(d);
        let node = snarl.get_node(id);
        let arity = node
            .map(|n| match n {
                EdNode::Output => 1,
                EdNode::Op { kind, .. } => kind.arity().max(1),
                EdNode::Biome { .. } => CLIMATE_INPUTS.len(),
                EdNode::Input(_) => 1,
            })
            .unwrap_or(1);
        // Op + Biome nodes carry a (default-on) preview, so they're tall; Input/Output are tiny. Use a
        // realistic per-kind default for any node not yet measured this session (e.g. off-screen at load),
        // so a single Auto-arrange already clears them instead of collapsing to a tiny default.
        let has_preview = matches!(node, Some(EdNode::Op { .. }) | Some(EdNode::Biome { .. }));
        let default_body =
            if has_preview { egui::vec2(210.0, DEFAULT_PREVIEW_PX + 96.0) } else { egui::vec2(70.0, 6.0) };
        let body = body_size.get(&id).copied().unwrap_or(default_body);
        // Input pins flank the body on the LEFT (header on top, output pin on the right), so the node's
        // height is `header + max(body, pin-rows)` — NOT the sum — and its width includes the input-label
        // column (biome climate labels are long) + the output pin. Summing body+pins over-estimated height,
        // spreading nodes (and shrinking the fit-to-view zoom).
        let max_label = node.map(|n| (0..arity).map(|s| input_label(n, s).len()).max().unwrap_or(0)).unwrap_or(0);
        let in_col = if max_label > 0 { 20.0 + max_label as f32 * 7.0 } else { 14.0 };
        let w = in_col + body.x.max(96.0) + OUT_PIN + FRAME;
        let h = HEADER + body.y.max(arity as f32 * PIN_ROW) + FRAME;
        size_of.insert(id, (w, h));
    }

    // Column x = prefix sum of each column's widest node + gap.
    let cols = (max_depth + 1) as usize;
    let mut col_w = vec![0.0f32; cols];
    for &id in &ids {
        let w = size_of[&id].0;
        let c = depth_of[&id] as usize;
        if w > col_w[c] {
            col_w[c] = w;
        }
    }
    let mut col_x = vec![0.0f32; cols];
    let mut acc = 0.0;
    for c in 0..cols {
        col_x[c] = acc;
        acc += col_w[c] + GAP_X;
    }

    // Stack rows within each column by real height.
    let mut col_y: HashMap<i32, f32> = HashMap::new();
    for &id in &ids {
        let d = depth_of[&id];
        let h = size_of[&id].1;
        let y = col_y.entry(d).or_insert(0.0);
        if let Some(node) = snarl.get_node_info_mut(id) {
            node.pos = egui::pos2(col_x[d as usize], *y);
        }
        *y += h + GAP_Y;
    }
}
