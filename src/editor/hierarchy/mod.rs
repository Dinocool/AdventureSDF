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

use crate::node::SceneNode;
use crate::sdf_render::{SdfPrimitive, SdfSelection};

use catalog::CreateDialog;
use reparent::{Actions, RenameState, apply_actions};
use tree::{DragCtx, NodeRow, collect_tree, descendant_set};

/// Cached scene-tree snapshot reused across frames. [`hierarchy_ui`] rebuilds it only when the
/// hierarchy actually changes ([`mark_hierarchy_dirty`] clears `valid`) or the filter string
/// changes — `collect_tree` is an O(all-nodes) ECS walk + per-node string allocation, so running
/// it every frame was the panel's whole residual cost once the render was virtualized.
#[derive(Resource, Default)]
pub struct HierarchyCache {
    arena: Vec<NodeRow>,
    roots: Vec<usize>,
    /// Filter string the cached `matches_filter` flags were computed for.
    filter: String,
    /// Cleared on any structural change; forces a rebuild the next time the panel draws.
    valid: bool,
}

/// Query filter matching a scene node whose tree-relevant state changed this tick: a new node, a
/// rename, a reparent (`ChildOf`/`Children`), or a primitive-kind change (drives the row icon).
type HierarchyChanged = (
    With<SceneNode>,
    Or<(
        Added<SceneNode>,
        Changed<Name>,
        Changed<ChildOf>,
        Changed<Children>,
        Changed<SdfPrimitive>,
    )>,
);

/// Clear [`HierarchyCache::valid`] when the scene tree changes — spawn / despawn / reparent /
/// rename / primitive-kind change. O(changed entities), so idle frames cost ~nothing and the
/// hierarchy panel reuses last frame's snapshot until something actually moves.
pub fn mark_hierarchy_dirty(
    mut cache: ResMut<HierarchyCache>,
    changed: Query<(), HierarchyChanged>,
    mut removed: RemovedComponents<SceneNode>,
) {
    if !changed.is_empty() || removed.read().count() > 0 {
        cache.valid = false;
    }
}

/// Register the hierarchy snapshot cache + its change-detection invalidator.
pub struct HierarchyPlugin;

impl Plugin for HierarchyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HierarchyCache>()
            .add_systems(Update, mark_hierarchy_dirty);
    }
}

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

    // The Create Node dialog spawns + selects a node. With the cached tree (below) the new
    // node appears the next frame, once `mark_hierarchy_dirty` sees the `Added<SceneNode>`.
    catalog::show_create_dialog(world, ui, &mut dialog);
    ui.memory_mut(|m| m.data.insert_temp(dialog_id, dialog));

    let needle = filter.trim().to_lowercase();

    // Reuse last frame's snapshot unless the hierarchy changed (valid cleared) or the filter
    // changed. Take the cached `Vec`s out so the render below owns them (no long `World` borrow,
    // and the rest of the function is unchanged); they're stashed back at the end.
    let rebuild = {
        let cache = world.resource::<HierarchyCache>();
        !cache.valid || cache.filter != needle
    };
    let (arena, roots) = if rebuild {
        let built = collect_tree(world, &needle);
        let mut cache = world.resource_mut::<HierarchyCache>();
        cache.filter = needle.clone();
        cache.valid = true;
        built
    } else {
        let mut cache = world.resource_mut::<HierarchyCache>();
        (
            std::mem::take(&mut cache.arena),
            std::mem::take(&mut cache.roots),
        )
    };

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

    // Flatten the VISIBLE tree (filter-matching rows, descending only into expanded
    // subtrees) to a linear list, then render ONLY the rows intersecting the viewport.
    // This makes the panel O(visible rows) instead of O(all entities): on the stress scene
    // (thousands of nodes) `ScrollArea::show` was laying out every row every frame even when
    // scrolled off-screen — ~25 ms/frame. Reading per-node expand state during the flatten is
    // a cheap memory lookup, so the flatten stays O(nodes) while the RENDER becomes O(visible).
    let flat = flatten_visible(&arena, &roots, ui.ctx());
    let row_h = ui.spacing().interact_size.y;
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show_viewport(ui, |ui, viewport| {
            let total_h = flat.len() as f32 * row_h;
            ui.set_min_height(total_h);
            let origin = ui.min_rect().min;

            // Full-content background: click empty space to deselect, drop a node here to
            // unparent it to the scene root. Interacted BEFORE the rows so the rows (added
            // after) win where they sit; only genuine empty space falls through here.
            let bg = ui.interact(
                egui::Rect::from_min_size(
                    origin,
                    egui::vec2(ui.available_width(), total_h.max(ui.available_height())),
                ),
                ui.id().with("tree_bg"),
                egui::Sense::click_and_drag(),
            );
            if bg.clicked() {
                actions.deselect = true;
            }
            if drag.dragged.is_some() && bg.contains_pointer() {
                ui.painter().rect_stroke(
                    bg.rect.shrink(1.0),
                    3.0,
                    ui.visuals().selection.stroke,
                    egui::StrokeKind::Inside,
                );
            }

            if flat.is_empty() {
                let mut c = ui.new_child(egui::UiBuilder::new().max_rect(
                    egui::Rect::from_min_size(origin, egui::vec2(ui.available_width(), row_h)),
                ));
                c.weak("No nodes in scene");
            }

            // Only the rows whose band [vi*row_h, (vi+1)*row_h) intersects the viewport.
            let first = (viewport.min.y / row_h).floor().max(0.0) as usize;
            let last = ((viewport.max.y / row_h).ceil() as usize).min(flat.len());
            for (offset, &(idx, depth)) in flat[first..last].iter().enumerate() {
                let vi = first + offset;
                let row_rect = egui::Rect::from_min_size(
                    origin + egui::vec2(0.0, vi as f32 * row_h),
                    egui::vec2(ui.available_width(), row_h),
                );
                // id_salt by entity so each row's widget ids are stable + unique across the
                // virtualized window (rows reuse source locations otherwise → id collisions).
                let mut row_ui = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(row_rect)
                        .layout(egui::Layout::left_to_right(egui::Align::Center))
                        .id_salt(arena[idx].entity),
                );
                render_flat_row(
                    &mut row_ui,
                    &arena,
                    idx,
                    depth,
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

    // Stash the snapshot back into the cache for next frame's reuse.
    {
        let mut cache = world.resource_mut::<HierarchyCache>();
        cache.arena = arena;
        cache.roots = roots;
    }

    apply_actions(world, &mut rename, commit_rename, actions);
    ui.memory_mut(|m| m.data.insert_temp(rename_id, rename));
}

/// Flatten the visible tree into `(arena_idx, depth)` rows in display order: filter-matching
/// nodes only, descending into a node's children only while it's expanded. Reading the expand
/// state is a cheap memory lookup, so this stays O(nodes) while the render becomes O(visible).
fn flatten_visible(arena: &[NodeRow], roots: &[usize], ctx: &egui::Context) -> Vec<(usize, u32)> {
    fn push(
        arena: &[NodeRow],
        idx: usize,
        depth: u32,
        ctx: &egui::Context,
        out: &mut Vec<(usize, u32)>,
    ) {
        if !arena[idx].matches_filter {
            return;
        }
        out.push((idx, depth));
        let has_children = arena[idx].children.iter().any(|&c| arena[c].matches_filter);
        if has_children && is_expanded(ctx, arena[idx].entity) {
            for &child in &arena[idx].children {
                push(arena, child, depth + 1, ctx, out);
            }
        }
    }
    let mut out = Vec::new();
    for &root in roots {
        push(arena, root, 0, ctx, &mut out);
    }
    out
}

/// Per-node expand state, persisted in egui memory keyed by entity (default expanded, matching
/// the old `default_open = true`). Stored as a plain bool so the flatten can read it cheaply
/// without instantiating a `CollapsingState` per node.
fn expand_id(entity: Entity) -> egui::Id {
    egui::Id::new(("hier_open", entity))
}
fn is_expanded(ctx: &egui::Context, entity: Entity) -> bool {
    ctx.memory(|m| m.data.get_temp::<bool>(expand_id(entity)).unwrap_or(true))
}
fn set_expanded(ctx: &egui::Context, entity: Entity, open: bool) {
    ctx.memory_mut(|m| m.data.insert_temp(expand_id(entity), open));
}

/// Draw one already-positioned (and already filter-/visibility-checked) row: indent,
/// disclosure toggle (parents only), then the draggable label — or the inline rename field.
/// Non-recursive: the tree shape comes from [`flatten_visible`], so off-screen rows cost nothing.
#[allow(clippy::too_many_arguments)]
fn render_flat_row(
    ui: &mut egui::Ui,
    arena: &[NodeRow],
    idx: usize,
    depth: u32,
    selected: Option<Entity>,
    drag: &DragCtx,
    rename: &mut RenameState,
    commit_rename: &mut bool,
    actions: &mut Actions,
) {
    let row = &arena[idx];
    ui.add_space(depth as f32 * 16.0);

    // Inline rename field replaces the row content while active.
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
    if has_children {
        let open = is_expanded(ui.ctx(), row.entity);
        let icon = if open {
            egui_phosphor::regular::CARET_DOWN
        } else {
            egui_phosphor::regular::CARET_RIGHT
        };
        if ui
            .add(egui::Button::new(icon).frame(false).min_size(egui::vec2(18.0, 0.0)))
            .clicked()
        {
            set_expanded(ui.ctx(), row.entity, !open);
        }
    } else {
        // Indent leaves to align with siblings that have a toggle.
        ui.add_space(18.0);
    }

    let resp = draggable_row(ui, row, selected == Some(row.entity), actions);
    wire_row(ui, &resp, row, drag, actions);
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
        // Snap the editor viewport to look through this node (its forward/position).
        // Most useful on a scene Camera node, but works on any node to frame it.
        if ui.button("Look through").clicked() {
            actions.look_through = Some(row.entity);
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
