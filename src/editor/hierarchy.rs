//! Scene panel (jackdaw `hierarchy`): the scene **node tree**. Walks Bevy's
//! `ChildOf`/`Children` hierarchy and lists every [`SceneNode`] (SDF volumes, the
//! directional light, the camera, plain nodes) as a collapsible tree. Click selects
//! (drives [`SdfSelection`]); double-click focuses the orbit camera; F2 / right-click
//! renames inline; drag a row onto another to reparent, or onto the unparent strip to
//! make it a root. The `+` button opens the Create Node dialog (a searchable,
//! category-nested catalog of node types).

use bevy::prelude::*;
use bevy_egui::egui;

use crate::node::SceneNode;
use crate::sdf_render::debug::{
    spawn_directional_light, spawn_empty_node, spawn_sdf_primitive,
};
use crate::sdf_render::{OrbitFocus, SdfPrimitive, SdfSelection};
use crate::soul_scene::{EditorHidden, SkipSerialization};

// --- Create Node catalog -------------------------------------------------------------

/// A single creatable node type: how it's labelled in the Create-Node dialog and the
/// closure that spawns it (returning the new entity, which the dialog reparents).
struct NodeType {
    label: &'static str,
    icon: &'static str,
    /// One-line description shown under the dialog's selection.
    hint: &'static str,
    spawn: fn(&mut World) -> Entity,
}

/// A named group of node types in the Create-Node dialog (Godot-style nesting).
struct NodeCategory {
    label: &'static str,
    types: &'static [NodeType],
}

/// The full node-type tree offered by the Create-Node dialog. Adding a new node type
/// is a one-line entry here plus a `spawn_*` helper — no dialog code changes.
const NODE_CATALOG: &[NodeCategory] = &[
    NodeCategory {
        label: "Node3D",
        types: &[NodeType {
            label: "Node3D",
            icon: "✦",
            hint: "Empty spatial node — a transform-only group / locator.",
            spawn: spawn_empty_node,
        }],
    },
    NodeCategory {
        label: "SDF Primitives",
        types: &[
            NodeType {
                label: "Sphere",
                icon: "◆",
                hint: "SDF sphere volume.",
                spawn: |w| spawn_sdf_primitive(w, SdfPrimitive::Sphere { radius: 0.5 }),
            },
            NodeType {
                label: "Box",
                icon: "◆",
                hint: "SDF box volume.",
                spawn: |w| {
                    spawn_sdf_primitive(w, SdfPrimitive::Box {
                        half_extents: Vec3::splat(0.5),
                    })
                },
            },
            NodeType {
                label: "Torus",
                icon: "◆",
                hint: "SDF torus volume.",
                spawn: |w| {
                    spawn_sdf_primitive(w, SdfPrimitive::Torus {
                        major: 0.5,
                        minor: 0.18,
                    })
                },
            },
            NodeType {
                label: "Capsule",
                icon: "◆",
                hint: "SDF capsule volume.",
                spawn: |w| {
                    spawn_sdf_primitive(w, SdfPrimitive::Capsule {
                        half_height: 0.4,
                        radius: 0.28,
                    })
                },
            },
            NodeType {
                label: "Cylinder",
                icon: "◆",
                hint: "SDF cylinder volume.",
                spawn: |w| {
                    spawn_sdf_primitive(w, SdfPrimitive::Cylinder {
                        radius: 0.4,
                        half_height: 0.5,
                    })
                },
            },
        ],
    },
    NodeCategory {
        label: "Lights",
        types: &[NodeType {
            label: "Directional Light",
            icon: "☀",
            hint: "Sun-style directional light (editor gizmo shows its direction).",
            spawn: spawn_directional_light,
        }],
    },
];

/// Create-Node dialog state, stashed in egui temp memory between frames.
#[derive(Clone, Default)]
struct CreateDialog {
    open: bool,
    /// Filter text matched against type labels.
    filter: String,
    /// `(category, type)` indices of the highlighted entry, if any.
    selected: Option<(usize, usize)>,
    /// Entity the new node is parented under (the selection when the dialog opened);
    /// `None` spawns at the scene root.
    parent: Option<Entity>,
}

/// In-progress inline rename, stashed in egui temp memory between frames.
#[derive(Clone, Default)]
struct RenameState {
    entity: Option<Entity>,
    buf: String,
}

/// Derived display kind for a node — probed from marker components (single source of
/// truth; no stored enum to drift). Drives the row's leading icon glyph.
#[derive(Clone, Copy, PartialEq)]
enum NodeKind {
    Sdf,
    Light,
    Camera,
    /// Spatial node with a transform but no recognized concrete type.
    Spatial,
    /// Bare node (no transform).
    Node,
}

impl NodeKind {
    fn icon(self) -> &'static str {
        match self {
            NodeKind::Sdf => "◆",
            NodeKind::Light => "☀",
            NodeKind::Camera => "▣",
            NodeKind::Spatial => "✦",
            NodeKind::Node => "•",
        }
    }
}

/// One node in the collected tree snapshot. `children` indexes back into the flat
/// arena so the egui pass can recurse without holding a query borrow.
struct NodeRow {
    entity: Entity,
    name: String,
    /// True if the node has a `Name` component (vs. a derived kind label). When named,
    /// the row hides the `#index` identifier.
    named: bool,
    kind: NodeKind,
    children: Vec<usize>,
    /// Lowercased text the filter matches against (this node only).
    filter_key: String,
    /// True if this node or any descendant matches the active filter.
    matches_filter: bool,
}

/// In-flight drag context for the current frame: the node being dragged and the set
/// of entities it may NOT be dropped onto (itself + its descendants). Drives the
/// drop-target highlight + insertion line.
struct DragCtx {
    dragged: Option<Entity>,
    forbidden: std::collections::HashSet<Entity>,
}

impl DragCtx {
    /// True if `target` is a valid drop target for the active drag (a drag is active,
    /// and target isn't the dragged node or one of its descendants).
    fn can_drop_on(&self, target: Entity) -> bool {
        self.dragged.is_some() && !self.forbidden.contains(&target)
    }
}

/// The dragged node (arena index `idx`) plus all its descendants, as an entity set —
/// the nodes a reparent must refuse to avoid creating a cycle.
fn descendant_set(arena: &[NodeRow], idx: usize) -> std::collections::HashSet<Entity> {
    let mut set = std::collections::HashSet::new();
    let mut stack = vec![idx];
    while let Some(i) = stack.pop() {
        if set.insert(arena[i].entity) {
            stack.extend(arena[i].children.iter().copied());
        }
    }
    set
}

/// Actions accumulated during the egui pass, applied to the world afterward (so we
/// never mutate while a query/tree borrow is live).
#[derive(Default)]
struct Actions {
    clicked: Option<Entity>,
    double_clicked: Option<Entity>,
    start_rename: Option<(Entity, String)>,
    /// `(child, new_parent)` — `None` parent unparents to a root.
    reparent: Option<(Entity, Option<Entity>)>,
    /// Clicked empty space in the tree → clear the selection.
    deselect: bool,
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
        if ui
            .button("+")
            .on_hover_text("Add a node…")
            .clicked()
        {
            dialog = CreateDialog {
                open: true,
                // Default the new node's parent to the current selection.
                parent: selected,
                ..Default::default()
            };
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
    show_create_dialog(world, ui, &mut dialog);
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
    let drag = DragCtx {
        dragged,
        forbidden,
    };

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

/// The Godot-style Create Node dialog: a searchable, category-nested tree of node
/// types. Picking one (double-click or the Create button) spawns it via its catalog
/// closure, parents it under `dialog.parent` (preserving world position), selects it,
/// and closes. State persists in `dialog` between frames.
fn show_create_dialog(world: &mut World, ui: &mut egui::Ui, dialog: &mut CreateDialog) {
    if !dialog.open {
        return;
    }

    let mut open = true;
    let mut spawn_choice: Option<(usize, usize)> = None;

    egui::Window::new("Create Node")
        .id(ui.make_persistent_id("create_node_window"))
        .collapsible(false)
        .resizable(true)
        .default_size([320.0, 420.0])
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .open(&mut open)
        .show(ui.ctx(), |ui| {
            ui.add(
                egui::TextEdit::singleline(&mut dialog.filter)
                    .hint_text("Search nodes…")
                    .desired_width(f32::INFINITY),
            );
            ui.separator();

            let needle = dialog.filter.trim().to_lowercase();
            egui::ScrollArea::vertical()
                .max_height(300.0)
                .show(ui, |ui| {
                    for (ci, cat) in NODE_CATALOG.iter().enumerate() {
                        // Hide a category whose every type is filtered out.
                        let any = cat.types.iter().any(|t| {
                            needle.is_empty() || t.label.to_lowercase().contains(&needle)
                        });
                        if !any {
                            continue;
                        }
                        egui::CollapsingHeader::new(cat.label)
                            .default_open(true)
                            .id_salt(("cat", ci))
                            .show(ui, |ui| {
                                for (ti, ty) in cat.types.iter().enumerate() {
                                    if !needle.is_empty()
                                        && !ty.label.to_lowercase().contains(&needle)
                                    {
                                        continue;
                                    }
                                    let is_sel = dialog.selected == Some((ci, ti));
                                    let resp = ui.selectable_label(
                                        is_sel,
                                        format!("{}  {}", ty.icon, ty.label),
                                    );
                                    if resp.clicked() {
                                        dialog.selected = Some((ci, ti));
                                    }
                                    if resp.double_clicked() {
                                        spawn_choice = Some((ci, ti));
                                    }
                                }
                            });
                    }
                });

            ui.separator();
            // Hint line for the highlighted type.
            if let Some((ci, ti)) = dialog.selected {
                ui.weak(NODE_CATALOG[ci].types[ti].hint);
            } else {
                ui.weak("Select a node type to create.");
            }

            ui.horizontal(|ui| {
                let can_create = dialog.selected.is_some();
                if ui
                    .add_enabled(can_create, egui::Button::new("Create"))
                    .clicked()
                {
                    spawn_choice = dialog.selected;
                }
                if ui.button("Cancel").clicked() {
                    dialog.open = false;
                }
            });
        });

    // Window close button (the `open` bool) also dismisses the dialog.
    if !open {
        dialog.open = false;
    }

    if let Some((ci, ti)) = spawn_choice {
        let entity = (NODE_CATALOG[ci].types[ti].spawn)(world);
        if let Some(parent) = dialog.parent
            && world.get_entity(parent).is_ok()
            && parent != entity
        {
            reparent_preserving_world(world, entity, parent);
        }
        world.resource_mut::<SdfSelection>().entity = Some(entity);
        dialog.open = false;
    }
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
        let resp = ui
            .add(egui::TextEdit::singleline(&mut rename.buf).desired_width(f32::INFINITY));
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
            ui.ctx()
                .transform_layer_shapes(layer_id, egui::emath::TSTransform::from_translation(delta));
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

/// Collect a flat arena of node rows + the root indices, walking `ChildOf`/`Children`.
/// `needle` (lowercased, may be empty) flags `matches_filter` on each row and its
/// ancestors so a filtered tree keeps the path to every match. Skips editor-only /
/// hidden entities so gizmos and overlays never pollute the tree.
fn collect_tree(world: &mut World, needle: &str) -> (Vec<NodeRow>, Vec<usize>) {
    struct Snap {
        entity: Entity,
        name: String,
        named: bool,
        kind: NodeKind,
        parent: Option<Entity>,
        children: Vec<Entity>,
    }

    let snaps: Vec<Snap> = world
        .query_filtered::<(
            Entity,
            Option<&Name>,
            Option<&ChildOf>,
            Option<&Children>,
            Option<&SdfPrimitive>,
            Has<DirectionalLight>,
            Has<Camera3d>,
            Has<Transform>,
        ), (With<SceneNode>, Without<EditorHidden>, Without<SkipSerialization>)>()
        .iter(world)
        .map(|(e, name, parent, children, prim, is_light, is_cam, has_xf)| {
            let kind = if prim.is_some() {
                NodeKind::Sdf
            } else if is_light {
                NodeKind::Light
            } else if is_cam {
                NodeKind::Camera
            } else if has_xf {
                NodeKind::Spatial
            } else {
                NodeKind::Node
            };
            let named = name.is_some();
            let name = name
                .map(|n| n.as_str().to_string())
                .unwrap_or_else(|| prim.map(primitive_label).unwrap_or("Node").to_string());
            Snap {
                entity: e,
                name,
                named,
                kind,
                parent: parent.map(|p| p.parent()),
                children: children.map(|c| c.iter().collect()).unwrap_or_default(),
            }
        })
        .collect();

    // Index entities so we can resolve parent/child references within the node set.
    let index: std::collections::HashMap<Entity, usize> =
        snaps.iter().enumerate().map(|(i, s)| (s.entity, i)).collect();

    let mut arena: Vec<NodeRow> = snaps
        .iter()
        .map(|s| {
            let filter_key = format!("{} #{}", s.name, s.entity.index()).to_lowercase();
            // Keep only children that are themselves nodes in our set.
            let children = s
                .children
                .iter()
                .filter_map(|c| index.get(c).copied())
                .collect();
            NodeRow {
                entity: s.entity,
                name: s.name.clone(),
                named: s.named,
                kind: s.kind,
                children,
                filter_key,
                matches_filter: true,
            }
        })
        .collect();

    // Roots: nodes whose parent is absent or not part of the node set.
    let mut roots: Vec<usize> = snaps
        .iter()
        .enumerate()
        .filter(|(_, s)| s.parent.map(|p| !index.contains_key(&p)).unwrap_or(true))
        .map(|(i, _)| i)
        .collect();
    roots.sort_by_key(|&i| arena[i].entity.index());

    // Filter: mark a node visible if it OR any descendant matches the needle. Walk
    // bottom-up isn't trivial with the arena, so recurse from roots.
    if !needle.is_empty() {
        for r in &mut arena {
            r.matches_filter = false;
        }
        for &root in &roots {
            mark_filter(&mut arena, root, needle);
        }
    }

    (arena, roots)
}

/// Returns true if `idx` or any descendant matches `needle`; sets `matches_filter`
/// along every surviving path.
fn mark_filter(arena: &mut [NodeRow], idx: usize, needle: &str) -> bool {
    let self_match = arena[idx].filter_key.contains(needle);
    let children = arena[idx].children.clone();
    let mut any_child = false;
    for c in children {
        any_child |= mark_filter(arena, c, needle);
    }
    let visible = self_match || any_child;
    arena[idx].matches_filter = visible;
    visible
}

/// Apply the egui pass's accumulated actions to the world.
fn apply_actions(
    world: &mut World,
    rename: &mut RenameState,
    commit_rename: bool,
    actions: Actions,
) {
    if let Some((entity, name)) = actions.start_rename {
        rename.entity = Some(entity);
        rename.buf = name;
    }

    if commit_rename {
        if let Some(entity) = rename.entity {
            let new = rename.buf.trim();
            if !new.is_empty()
                && let Ok(mut e) = world.get_entity_mut(entity)
            {
                e.insert(Name::new(new.to_string()));
            }
        }
        *rename = RenameState::default();
    }

    // Reparent (cycle-guarded): never parent a node under itself or a descendant.
    if let Some((child, new_parent)) = actions.reparent {
        match new_parent {
            Some(parent) if parent != child && !is_descendant(world, parent, child) => {
                reparent_preserving_world(world, child, parent);
            }
            Some(_) => {} // self or cycle — ignore
            None => {
                // Unparented: local transform becomes the former world transform.
                let cg = world.get::<GlobalTransform>(child).copied();
                if let Ok(mut e) = world.get_entity_mut(child) {
                    if let Some(cg) = cg {
                        e.insert(cg.compute_transform());
                    }
                    e.remove::<ChildOf>();
                }
            }
        }
    }

    // Double-click focuses the orbit camera on the node (if it has a Transform).
    if let Some(entity) = actions.double_clicked {
        let pos = world.get::<Transform>(entity).map(|t| t.translation);
        if let Some(pos) = pos {
            world.resource_mut::<OrbitFocus>().target = Some(pos);
        }
    }

    if let Some(entity) = actions.clicked.or(actions.double_clicked) {
        world.resource_mut::<SdfSelection>().entity = Some(entity);
    } else if actions.deselect {
        // Clicked empty tree space → select nothing.
        world.resource_mut::<SdfSelection>().entity = None;
    }
}

/// True if `candidate` is `ancestor` or appears in `ancestor`'s `Children` subtree.
/// Used to reject reparent operations that would create a cycle.
fn is_descendant(world: &World, candidate: Entity, ancestor: Entity) -> bool {
    if candidate == ancestor {
        return true;
    }
    let mut stack = vec![ancestor];
    while let Some(e) = stack.pop() {
        if let Some(children) = world.get::<Children>(e) {
            for child in children.iter() {
                if child == candidate {
                    return true;
                }
                stack.push(child);
            }
        }
    }
    false
}

/// Parent `child` under `parent`, preserving the child's world transform. Bevy keeps
/// the child's *local* Transform across a reparent, so under a non-identity parent the
/// node would visually jump; recompute the local transform via `reparented_to`.
/// Caller must have already rejected cycles (`is_descendant`).
fn reparent_preserving_world(world: &mut World, child: Entity, parent: Entity) {
    let cg = world.get::<GlobalTransform>(child).copied();
    let pg = world.get::<GlobalTransform>(parent).copied();
    if let (Some(cg), Some(pg)) = (cg, pg) {
        let local = cg.reparented_to(&pg);
        if let Ok(mut e) = world.get_entity_mut(child) {
            e.insert((ChildOf(parent), local));
        }
    } else if let Ok(mut e) = world.get_entity_mut(child) {
        e.insert(ChildOf(parent));
    }
}

fn primitive_label(prim: &SdfPrimitive) -> &'static str {
    match prim {
        SdfPrimitive::Sphere { .. } => "Sphere",
        SdfPrimitive::Box { .. } => "Box",
        SdfPrimitive::Torus { .. } => "Torus",
        SdfPrimitive::Capsule { .. } => "Capsule",
        SdfPrimitive::Cylinder { .. } => "Cylinder",
        SdfPrimitive::Heightmap { .. } => "Heightmap",
    }
}
