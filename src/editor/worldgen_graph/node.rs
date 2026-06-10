//! Presentation + metadata for the editor node kinds: per-kind param widgets, the add-node catalogue,
//! display names, and input-pin labels. Pure functions of an [`EdNode`]/[`NodeKind`].

use bevy_egui::egui;

use crate::sdf_render::worldgen::graph::node::{FbmAxis, NodeKind};
use crate::sdf_render::worldgen::spline::Spline;

use super::{EdNode, climate_name};

/// Editable params for a node kind (drawn on its output row).
pub(super) fn node_params_ui(ui: &mut egui::Ui, kind: &mut NodeKind) {
    match kind {
        NodeKind::Const(v) => {
            ui.add(egui::DragValue::new(v).speed(1.0));
        }
        NodeKind::Scale(k) | NodeKind::Offset(k) => {
            ui.add(egui::DragValue::new(k).speed(1.0));
        }
        NodeKind::Ridge { ridge, amp_sum } => {
            ui.add(egui::DragValue::new(ridge).speed(0.01).range(0.0..=1.0).prefix("ridge "))
                .on_hover_text("Ridge fold strength: 0 = smooth fBm, 1 = sharp ridged peaks (folds toward 1−|n|). Lower this to calm over-prominent ridgelines.");
            ui.add(egui::DragValue::new(amp_sum).speed(1.0).prefix("amp_sum "))
                .on_hover_text("Expected swing of the input (the fBm's amplitude·Σgain^o). Sets where the fold reflects.");
        }
        NodeKind::Smoothstep { edge0, edge1 } => {
            ui.add(egui::DragValue::new(edge0).speed(0.01).prefix("e0 "));
            ui.add(egui::DragValue::new(edge1).speed(0.01).prefix("e1 "));
        }
        NodeKind::Clamp { lo, hi } => {
            ui.add(egui::DragValue::new(lo).speed(1.0).prefix("lo "));
            ui.add(egui::DragValue::new(hi).speed(1.0).prefix("hi "));
        }
        NodeKind::Fbm(ax) => {
            ui.add(egui::DragValue::new(&mut ax.amplitude).speed(1.0).prefix("amp "))
                .on_hover_text("Height of the biggest (octave-0) wave, in metres — the overall vertical scale.");
            let mut wavelength = if ax.base_freq != 0.0 { 1.0 / ax.base_freq } else { 0.0 };
            if ui
                .add(egui::DragValue::new(&mut wavelength).speed(8.0).prefix("λ "))
                .on_hover_text("Wavelength (m) of the biggest feature: larger = broader, gentler shapes.")
                .changed()
                && wavelength > 0.0
            {
                ax.base_freq = 1.0 / wavelength;
            }
            ui.add(egui::DragValue::new(&mut ax.octaves).range(1..=8).prefix("oct "))
                .on_hover_text("How many noise layers to sum — more octaves = finer detail (each half as tall, twice as fine).");
        }
        NodeKind::Curve(_) => {
            ui.label("curve");
        }
        // WorldX/WorldZ/Abs/Neg/Add/Sub/Mul/Min/Max/Mix — no scalar params.
        _ => {}
    }
}

/// The palette of node kinds offered by the add-node menu (sensible defaults).
pub(super) fn node_catalog() -> Vec<NodeKind> {
    vec![
        NodeKind::Fbm(FbmAxis { octaves: 4, base_freq: 1.0 / 1024.0, lacunarity: 2.0, gain: 0.5, amplitude: 100.0, seed_salt: 1 }),
        NodeKind::Ridge { ridge: 0.85, amp_sum: 200.0 },
        NodeKind::Curve(Spline::new(&[(-1.0, 0.0), (0.0, 0.5), (1.0, 1.0)])),
        NodeKind::Smoothstep { edge0: 0.0, edge1: 1.0 },
        NodeKind::Mix,
        NodeKind::Add,
        NodeKind::Sub,
        NodeKind::Mul,
        NodeKind::Min,
        NodeKind::Max,
        NodeKind::Clamp { lo: -1000.0, hi: 1000.0 },
        NodeKind::Scale(1.0),
        NodeKind::Offset(0.0),
        NodeKind::Abs,
        NodeKind::Neg,
        NodeKind::Const(0.0),
        NodeKind::WorldX,
        NodeKind::WorldZ,
    ]
}

pub(super) fn node_kind_name(k: &NodeKind) -> &'static str {
    match k {
        NodeKind::WorldX => "WorldX",
        NodeKind::WorldZ => "WorldZ",
        NodeKind::Const(_) => "Const",
        NodeKind::Fbm(_) => "Fbm",
        NodeKind::Curve(_) => "Curve",
        NodeKind::Ridge { .. } => "Ridge",
        NodeKind::Clamp { .. } => "Clamp",
        NodeKind::Smoothstep { .. } => "Smoothstep",
        NodeKind::Scale(_) => "Scale",
        NodeKind::Offset(_) => "Offset",
        NodeKind::Abs => "Abs",
        NodeKind::Neg => "Neg",
        NodeKind::Add => "Add",
        NodeKind::Sub => "Sub",
        NodeKind::Mul => "Mul",
        NodeKind::Min => "Min",
        NodeKind::Max => "Max",
        NodeKind::Mix => "Mix",
    }
}

/// Human label for a node's input pin `slot` (shown beside the pin).
pub(super) fn input_label(node: &EdNode, slot: usize) -> &'static str {
    match node {
        EdNode::Output => "height",
        EdNode::Input(_) => "",
        EdNode::Biome { .. } => climate_name(slot),
        EdNode::Op(k) => match k {
            NodeKind::Add | NodeKind::Sub | NodeKind::Mul | NodeKind::Min | NodeKind::Max => {
                if slot == 0 { "a" } else { "b" }
            }
            NodeKind::Mix => match slot {
                0 => "a",
                1 => "b",
                _ => "t",
            },
            NodeKind::Ridge { .. }
            | NodeKind::Curve(_)
            | NodeKind::Smoothstep { .. }
            | NodeKind::Clamp { .. }
            | NodeKind::Scale(_)
            | NodeKind::Offset(_)
            | NodeKind::Abs
            | NodeKind::Neg => "x",
            _ => "in",
        },
    }
}
