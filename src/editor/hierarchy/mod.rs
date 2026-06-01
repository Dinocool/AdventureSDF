//! Scene panel (jackdaw `hierarchy`): the scene **node tree**. Walks Bevy's
//! `ChildOf`/`Children` hierarchy and lists every [`SceneNode`](crate::node::SceneNode)
//! (SDF volumes, the directional light, the camera, plain nodes) as a collapsible tree.
//! Click selects (drives [`SdfSelection`]); double-click focuses the orbit camera; F2 /
//! right-click renames inline; drag a row onto another to reparent, or onto the unparent
//! strip to make it a root. The `+` button opens the Create Node dialog (a searchable,
//! category-nested catalog of node types).
//!
//! Submodules: [`catalog`] (Create-Node catalog + dialog), [`tree`] (ECS → arena
//! snapshot + filter), [`reparent`] (action application: rename/reparent/focus/select).

mod catalog;
mod reparent;
mod tree;

use bevy::prelude::*;
use bevy_egui::egui;

use crate::sdf_render::SdfSelection;

use catalog::CreateDialog;
use reparent::{Actions, RenameState, apply_actions};
use tree::{DragCtx, NodeRow, collect_tree, descendant_set};

/// Render the scene node tree.
pub fn hierarchy_ui(world: &mut World, ui: &mut egui::Ui) {
    let selected = world.resource::<SdfSelection>().entity;

    let filter_id = ui.make_persistent_id("scene_filter");
    let rename_id = ui.make_persistent_id("scene_rename");
    let mut filter: String =
        ui.memory_mut(|m| m.data.get_temp::<String>(filter_id).unwrap_or_default());
    let mut rename: RenameState =
        ui.memory_mut(|m| m.data.get_temp::<RenameState>(rename_id).unwrap_or_default());

    let dialog_id = ui.make_persistent_id("create_node_dialog");
    let mut dialog: CreateDialog =
        ui.memory_mut(|m| m.data.get_temp::<CreateDialog>(dialog_id).unwrap_or_default());

    // Toolbar: add-node button (opens the Create Node dialog) + filter box.
    ui.horizontal(|ui| {
        if ui.button("+").on_hover_text("Add a node…").clicked() {
            // Default the new node's parent to the current selection.
            dialog = CreateDialog::opened_under(selected);
        }
        ui.add(
            egui::TextEdit::singleline(&mut filter)
                .hint_text("Filter")
                .desired_width(f32::INFINITY),
        );
    });
    ui.memory_mut(|m| m.data.insert_temp(filter_id, filter.clone()));

    // The Create Node dialog spawns + selects a node; render it before the tree so the
    // new node shows this frame.
    catalog::show_create_dialog(world, ui, &mut dialog);
    ui.memory_mut(|m| m.data.insert_temp(dialog_id, dialog));

    let needle = filter.trim().to_lowercase();
    let (arena, roots) = collect_tree(world, &needle);

    // F2 starts renaming the current selection (if not already renaming).
    let f2 = ui.input(|i| i.key_pressed(egui::Key::F2));
    if f2
        && rename.entity.is_none()
        && let Some(sel) = selected
        && let Some(row) = arena.iter().find(|r| r.entity == sel)
    {
        rename.entity = Some(sel);
        rename.buf = row.name.clone();
    }

    let mut actions = Actions::default();
    let mut commit_rename = false;

    // Peek the in-flight drag payload (if any) so rows can show where the node will
    // land. A node can't be dropped onto itself or one of its descendants, so collect
    // that forbidden set once from the arena.
    let dragged = egui::DragAndDrop::payload::<Entity>(ui.ctx()).map(|e| *e);
    let forbidden = dragged
        .and_then(|d| arena.iter().position(|r| r.entity == d))
        .map(|i| descendant_set(&arena, i))
        .unwrap_or_default();
    let drag = DragCtx { dragged, forbidden };

    egui::ScrollArea::vertical().show(ui, |ui| {
        // Full-panel background target. Created BEFORE the rows so the rows (drawn on
        // top) win where they are; any click/drop landing on empty space — below OR to
        // the right of a row — falls through here. Click → deselect; drop a dragged
        // node here → unparent it to the scene root.
        let bg = ui.interact(
            ui.max_rect(),
            ui.id().with("tree_bg"),
            egui::Sense::click_and_drag(),
        );
        if bg.clicked() {
            actions.deselect = true;
        }
        // While dragging, hint that releasing on empty space unparents to the root.
        if drag.dragged.is_some() && bg.contains_pointer() {
            ui.painter().rect_stroke(
                bg.rect.shrink(1.0),
                3.0,
                ui.visuals().selection.stroke,
                egui::StrokeKind::Inside,
            );
        }

        if arena.is_empty() {
            ui.weak("No nodes in scene");
        }
        for &root in &roots {
            render_node(
                ui,
                &arena,
                root,
                selected,
                &drag,
                &mut rename,
                &mut commit_rename,
                &mut actions,
            );
        }

        // Unparent drop is resolved AFTER the rows so a row drop-target consumes the
        // payload first; the background only claims it when released over empty space.
        // (`dnd_release_payload` *takes* the payload, so order matters.)
        if actions.reparent.is_none()
            && let Some(child) = bg.dnd_release_payload::<Entity>()
        {
            actions.reparent = Some((*child, None));
        }
    });

    apply_actions(world, &mut rename, commit_rename, actions);
    ui.memory_mut(|m| m.data.insert_temp(rename_id, rename));
}

/// Recursively draw one node and its node-children. Each row is a drag source (grab
/// to reparent) and a drop target (release another node on it to nest it here).
/// Parents use a `CollapsingState` so the disclosure triangle works while the header
/// row itself stays draggable.
#[allow(clippy::too_many_arguments)]
fn render_node(
    ui: &mut egui::Ui,
    arena: &[NodeRow],
    idx: usize,
    selected: Option<Entity>,
    drag: &DragCtx,
    rename: &mut RenameState,
    commit_rename: &mut bool,
    actions: &mut Actions,
) {
    let row = &arena[idx];
    if !row.matches_filter {
        return;
    }

    // Inline rename field replaces the row label while active.
    if rename.entity == Some(row.entity) {
        let resp =
            ui.add(egui::TextEdit::singleline(&mut rename.buf).desired_width(f32::INFINITY));
        resp.request_focus();
        let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
        if enter || resp.lost_focus() {
            *commit_rename = true;
        }
        return;
    }

    let has_children = row.children.iter().any(|&c| arena[c].matches_filter);
    let state_id = ui.make_persistent_id(("node", row.entity));
    let mut collapse = egui::collapsing_header::CollapsingState::load_with_default_open(
        ui.ctx(),
        state_id,
        true,
    );

    let header_resp = ui
        .horizontal(|ui| {
            if has_children {
                collapse.show_toggle_button(ui, egui::collapsing_header::paint_default_icon);
            } else {
                // Indent leaves to align with siblings that have a toggle.
                ui.add_space(18.0);
            }
            draggable_row(ui, row, selected == Some(row.entity), actions)
        })
        .inner;

    wire_row(ui, &header_resp, row, drag, actions);

    if has_children {
        collapse.show_body_indented(&header_resp, ui, |ui| {
            for &child in &row.children {
                render_node(
                    ui,
                    arena,
                    child,
                    selected,
                    drag,
                    rename,
                    commit_rename,
                    actions,
                );
            }
        });
    }
    collapse.store(ui.ctx());
}

/// Draw the node's label as a click-and-drag row and return its response. The
/// response is both the drag source (sets an `Entity` payload while dragged) and,
/// via [`wire_row`], the drop target. Selected rows are tinted.
fn draggable_row(
    ui: &mut egui::Ui,
    row: &NodeRow,
    is_sel: bool,
    _actions: &mut Actions,
) -> egui::Response {
    let label = if row.named {
        format!("{} {}", row.kind.icon(), row.name)
    } else {
        format!("{} {}  #{}", row.kind.icon(), row.name, row.entity.index())
    };
    let mut text = egui::RichText::new(label);
    if is_sel {
        text = text.color(ui.visuals().selection.stroke.color).strong();
    }
    // `click_and_drag` so the same row both selects (click) and reparents (drag).
    let resp = ui.add(
        egui::Label::new(text)
            .selectable(false)
            .sense(egui::Sense::click_and_drag()),
    );

    if resp.dragged() {
        resp.dnd_set_drag_payload(row.entity);

        // Live preview: paint a floating copy of the row to a Tooltip-order layer and
        // translate that layer so it tracks the cursor (mirrors egui's own
        // `dnd_drag_source`, which we can't use here because it suppresses clicks).
        let layer_id = egui::LayerId::new(egui::Order::Tooltip, resp.id.with("drag_preview"));
        let preview = ui
            .scope_builder(egui::UiBuilder::new().layer_id(layer_id), |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.label(format!("{} {}", row.kind.icon(), row.name));
                });
            })
            .response;
        if let Some(pointer) = ui.ctx().pointer_interact_pos() {
            let delta = pointer - preview.rect.left_center() + egui::vec2(12.0, 0.0);
            ui.ctx().transform_layer_shapes(
                layer_id,
                egui::emath::TSTransform::from_translation(delta),
            );
        }
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
    }
    resp
}

/// Attach selection / rename / drop-reparent behavior to a row's response, and draw
/// the drop-target indicator (highlight + insertion line) when a valid drag hovers it.
fn wire_row(
    ui: &egui::Ui,
    resp: &egui::Response,
    row: &NodeRow,
    drag: &DragCtx,
    actions: &mut Actions,
) {
    if resp.clicked() {
        actions.clicked = Some(row.entity);
    }
    if resp.double_clicked() {
        actions.double_clicked = Some(row.entity);
    }
    resp.context_menu(|ui| {
        if ui.button("Rename").clicked() {
            actions.start_rename = Some((row.entity, row.name.clone()));
            ui.close();
        }
    });

    // Drop preview: when a droppable node hovers this row, outline the prospective
    // parent and draw an insertion line at the indent where the child will be appended.
    if resp.contains_pointer() && drag.can_drop_on(row.entity) {
        let painter = ui.painter();
        let stroke = ui.visuals().selection.stroke;
        painter.rect_stroke(
            resp.rect.expand2(egui::vec2(2.0, 1.0)),
            3.0,
            stroke,
            egui::StrokeKind::Outside,
        );
        // Insertion line: bottom edge of the row, indented one level (child position).
        let y = resp.rect.bottom() + 1.0;
        let x0 = resp.rect.left() + 16.0;
        painter.hline(x0..=resp.rect.right(), y, stroke);
    }

    // Drop target: releasing node X on this row reparents X under it (cycle rejected).
    if let Some(child) = resp.dnd_release_payload::<Entity>()
        && drag.can_drop_on(row.entity)
        && *child != row.entity
    {
        actions.reparent = Some((*child, Some(row.entity)));
    }
}
