use bevy::prelude::*;

/// Which dock region a panel renders into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DockSide {
    Left,
    Right,
    Bottom,
}

/// A debug panel contributed by any plugin. The `render` closure is called once
/// per frame with exclusive `World` access inside a collapsing section.
pub struct DebugPanel {
    pub id: String,
    pub title: String,
    pub dock: DockSide,
    pub order: i32,
    pub render: Box<dyn Fn(&mut World, &mut bevy_egui::egui::Ui) + Send + Sync>,
}

/// Write-once catalog of debug panels. Plugins register at build time; the dock
/// layout reads it every frame.
#[derive(Resource, Default)]
pub struct DebugPanelRegistry {
    panels: Vec<DebugPanel>,
}

impl DebugPanelRegistry {
    pub fn register(&mut self, panel: DebugPanel) {
        self.panels.push(panel);
    }

    /// Panels for one side, sorted by `order` then `id`. The dock layout owns the
    /// registry locally while rendering (see `mod.rs`), so handing back refs is
    /// fine — the closures are invoked with `&mut World` separately.
    pub fn panels_for(&self, side: DockSide) -> Vec<&DebugPanel> {
        let mut matching: Vec<&DebugPanel> =
            self.panels.iter().filter(|p| p.dock == side).collect();
        matching.sort_by(|a, b| a.order.cmp(&b.order).then(a.id.cmp(&b.id)));
        matching
    }
}

pub struct DebugPanelRegistryPlugin;

impl Plugin for DebugPanelRegistryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DebugPanelRegistry>();
    }
}

/// Helper for registering a panel from a plugin `build`.
///
/// Inits the registry if absent so plugins can register panels regardless of
/// whether they build before or after [`DebugToolkitPlugin`] — the toolkit's own
/// `init_resource` is then a no-op that preserves these entries.
pub fn register_panel(
    app: &mut App,
    id: impl Into<String>,
    title: impl Into<String>,
    dock: DockSide,
    order: i32,
    render: impl Fn(&mut World, &mut bevy_egui::egui::Ui) + Send + Sync + 'static,
) {
    app.init_resource::<DebugPanelRegistry>();
    let mut registry = app.world_mut().resource_mut::<DebugPanelRegistry>();
    registry.register(DebugPanel {
        id: id.into(),
        title: title.into(),
        dock,
        order,
        render: Box::new(render),
    });
}
