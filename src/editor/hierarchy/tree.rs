//! Scene-tree snapshot: walk Bevy's `ChildOf`/`Children` hierarchy into a flat arena of
//! [`NodeRow`]s the egui pass can recurse over without holding a query borrow, plus the
//! filter marking that keeps the path to every match.

use bevy::prelude::*;

use crate::node::SceneNode;
use crate::scene_manager::EditorEntity;
use crate::sdf_render::SdfPrimitive;
use crate::soul_scene::{EditorHidden, SkipSerialization};

/// Derived display kind for a node — probed from marker components (single source of
/// truth; no stored enum to drift). Drives the row's leading icon glyph.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum NodeKind {
    Sdf,
    Light,
    Camera,
    /// Spatial node with a transform but no recognized concrete type.
    Spatial,
    /// Bare node (no transform).
    Node,
}

impl NodeKind {
    pub(super) fn icon(self) -> &'static str {
        use egui_phosphor::regular as icon;
        match self {
            NodeKind::Sdf => icon::CUBE,
            NodeKind::Light => icon::LIGHTBULB,
            NodeKind::Camera => icon::VIDEO_CAMERA,
            NodeKind::Spatial => icon::CUBE_TRANSPARENT,
            NodeKind::Node => icon::DOT_OUTLINE,
        }
    }
}

/// One node in the collected tree snapshot. `children` indexes back into the flat
/// arena so the egui pass can recurse without holding a query borrow.
pub(super) struct NodeRow {
    pub(super) entity: Entity,
    pub(super) name: String,
    /// True if the node has a `Name` component (vs. a derived kind label). When named,
    /// the row hides the `#index` identifier.
    pub(super) named: bool,
    pub(super) kind: NodeKind,
    pub(super) children: Vec<usize>,
    /// Lowercased text the filter matches against (this node only).
    filter_key: String,
    /// True if this node or any descendant matches the active filter.
    pub(super) matches_filter: bool,
}

/// In-flight drag context for the current frame: the node being dragged and the set
/// of entities it may NOT be dropped onto (itself + its descendants). Drives the
/// drop-target highlight + insertion line.
pub(super) struct DragCtx {
    pub(super) dragged: Option<Entity>,
    pub(super) forbidden: std::collections::HashSet<Entity>,
}

impl DragCtx {
    /// True if `target` is a valid drop target for the active drag (a drag is active,
    /// and target isn't the dragged node or one of its descendants).
    pub(super) fn can_drop_on(&self, target: Entity) -> bool {
        self.dragged.is_some() && !self.forbidden.contains(&target)
    }
}

/// The dragged node (arena index `idx`) plus all its descendants, as an entity set —
/// the nodes a reparent must refuse to avoid creating a cycle.
pub(super) fn descendant_set(arena: &[NodeRow], idx: usize) -> std::collections::HashSet<Entity> {
    let mut set = std::collections::HashSet::new();
    let mut stack = vec![idx];
    while let Some(i) = stack.pop() {
        if set.insert(arena[i].entity) {
            stack.extend(arena[i].children.iter().copied());
        }
    }
    set
}

/// Collect a flat arena of node rows + the root indices, walking `ChildOf`/`Children`.
/// `needle` (lowercased, may be empty) flags `matches_filter` on each row and its
/// ancestors so a filtered tree keeps the path to every match. Skips editor-only /
/// hidden entities so gizmos and overlays never pollute the tree.
pub(super) fn collect_tree(world: &mut World, needle: &str) -> (Vec<NodeRow>, Vec<usize>) {
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
        ), (
            With<SceneNode>,
            Without<EditorEntity>,
            Without<EditorHidden>,
            Without<SkipSerialization>,
        )>()
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
