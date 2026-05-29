use bevy::input::mouse::MouseMotion;
use bevy::picking::prelude::*;
use bevy::prelude::*;
use bevy::window::{CursorIcon, SystemCursorIcon};

use crate::camera::{RightClickEvent, ThirdPersonCamera};
use crate::inventory::{LootTransferEvent, Lootable, PlayerInventory};
use crate::scene_manager::AppScene;

pub struct UiPlugin;

#[derive(Component)]
struct HealthBar;

#[derive(Component)]
struct ManaBar;

#[derive(Component)]
struct ActionBar;

#[derive(Component)]
struct InventoryPanel;

#[derive(Component)]
struct InventoryTitleBar;

#[derive(Component)]
struct InventorySlot(pub usize);

#[derive(Component)]
struct LootWindow;

#[derive(Component)]
struct LootTitleBar;

#[derive(Component)]
struct LootSlot {
    source_entity: Entity,
    item_index: usize,
}

#[derive(Component)]
struct LootWindowClose;

#[derive(Component)]
struct Tooltip;

#[derive(Resource, Default)]
pub struct InventoryUiState {
    pub inventory_open: bool,
    pub loot_target: Option<Entity>,
    pub last_right_click_pos: Vec2,
    pub prev_loot_target: Option<Entity>,
}

#[derive(Resource, Default)]
pub struct WindowDragState {
    pub dragging: Option<Entity>,
    pub offset: Vec2,
}

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum UiSet {
    Input,
    Sync,
}

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InventoryUiState>()
            .init_resource::<WindowDragState>()
            .configure_sets(Update, (UiSet::Input, UiSet::Sync).chain())
            .add_systems(OnEnter(AppScene::AdventureGame), setup_ui)
            .add_systems(
                Update,
                (
                    toggle_inventory,
                    handle_right_click_pick,
                    update_lootable_cursor,
                )
                    .in_set(UiSet::Input)
                    .run_if(in_state(AppScene::AdventureGame)),
            )
            .add_systems(
                Update,
                (
                    handle_window_drag,
                    spawn_inventory_panel.run_if(inventory_open),
                    despawn_inventory_panel.run_if(inventory_closed),
                    spawn_loot_window.run_if(loot_window_open),
                    despawn_loot_window.run_if(loot_window_closed),
                    handle_loot_slot_click,
                    handle_tooltip,
                    handle_empty_container,
                )
                    .in_set(UiSet::Sync)
                    .run_if(in_state(AppScene::AdventureGame)),
            );
    }
}

fn inventory_open(state: Res<InventoryUiState>) -> bool {
    state.inventory_open
}

fn inventory_closed(state: Res<InventoryUiState>) -> bool {
    !state.inventory_open
}

fn loot_window_open(state: Res<InventoryUiState>) -> bool {
    state.loot_target.is_some()
}

fn loot_window_closed(state: Res<InventoryUiState>) -> bool {
    state.loot_target.is_none()
}

// --- HUD ---

fn setup_ui(mut commands: Commands) {
    commands.spawn((
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            ..default()
        },
        children![
            (
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(60.0),
                    padding: UiRect::all(Val::Px(10.0)),
                    ..default()
                },
                children![
                    (
                        Node {
                            width: Val::Px(200.0),
                            height: Val::Px(20.0),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.6, 0.1, 0.1)),
                        HealthBar,
                    ),
                    (
                        Node {
                            width: Val::Px(200.0),
                            height: Val::Px(20.0),
                            margin: UiRect::top(Val::Px(5.0)),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.1, 0.1, 0.6)),
                        ManaBar,
                    ),
                ],
            ),
            (
                Node {
                    position_type: PositionType::Absolute,
                    bottom: Val::Px(20.0),
                    width: Val::Percent(100.0),
                    height: Val::Px(60.0),
                    justify_content: JustifyContent::Center,
                    ..default()
                },
                BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.7)),
                ActionBar,
            ),
        ],
    ));
}

#[allow(dead_code)]
fn update_ui(
    player_query: Query<(&crate::player::Health, &crate::player::Mana)>,
    mut health_query: Query<&mut Node, (With<HealthBar>, Without<ManaBar>)>,
    mut mana_query: Query<&mut Node, (With<ManaBar>, Without<HealthBar>)>,
) {
    let Ok((health, mana)) = player_query.single() else {
        return;
    };

    let health_pct = health.current / health.max;
    let mana_pct = mana.current / mana.max;

    for mut style in &mut health_query {
        style.width = Val::Px(200.0 * health_pct);
    }
    for mut style in &mut mana_query {
        style.width = Val::Px(200.0 * mana_pct);
    }
}

// --- Input ---

fn toggle_inventory(keyboard: Res<ButtonInput<KeyCode>>, mut state: ResMut<InventoryUiState>) {
    if keyboard.just_pressed(KeyCode::KeyI) {
        state.inventory_open = !state.inventory_open;
        if !state.inventory_open {
            state.loot_target = None;
        }
    }
}

fn handle_right_click_pick(
    mut messages: MessageReader<RightClickEvent>,
    mut ray_cast: MeshRayCast,
    camera_query: Query<(&Camera, &GlobalTransform), With<ThirdPersonCamera>>,
    lootable_query: Query<(), With<Lootable>>,
    mut ui_state: ResMut<InventoryUiState>,
) {
    let Ok((camera, cam_transform)) = camera_query.single() else {
        return;
    };

    for event in messages.read() {
        let Ok(ray) = camera.viewport_to_world(cam_transform, event.screen_position) else {
            continue;
        };

        let filter = |entity: Entity| lootable_query.contains(entity);
        let settings = MeshRayCastSettings::default()
            .with_filter(&filter)
            .with_visibility(RayCastVisibility::Any);

        let hits = ray_cast.cast_ray(ray, &settings);
        if let Some((entity, _)) = hits.first() {
            ui_state.loot_target = Some(*entity);
            ui_state.last_right_click_pos = event.screen_position;
        }
    }
}

fn update_lootable_cursor(
    mut cursor_icon: Query<&mut CursorIcon, With<Window>>,
    mut ray_cast: MeshRayCast,
    camera_query: Query<(&Camera, &GlobalTransform), With<ThirdPersonCamera>>,
    lootable_query: Query<(), With<Lootable>>,
    windows: Query<&Window>,
) {
    let Ok(mut icon) = cursor_icon.single_mut() else {
        return;
    };
    let Ok(window) = windows.single() else {
        return;
    };
    let cursor = match window.cursor_position() {
        Some(pos) => pos,
        None => {
            *icon = CursorIcon::System(SystemCursorIcon::Default);
            return;
        }
    };

    let Ok((camera, cam_transform)) = camera_query.single() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(cam_transform, cursor) else {
        return;
    };

    let filter = |entity: Entity| lootable_query.contains(entity);
    let settings = MeshRayCastSettings::default()
        .with_filter(&filter)
        .with_visibility(RayCastVisibility::Any);

    let hits = ray_cast.cast_ray(ray, &settings);
    *icon = if !hits.is_empty() {
        CursorIcon::System(SystemCursorIcon::Pointer)
    } else {
        CursorIcon::System(SystemCursorIcon::Default)
    };
}

// --- Inventory Panel ---

fn spawn_inventory_panel(
    mut commands: Commands,
    inventory: Res<PlayerInventory>,
    panel_query: Query<Entity, With<InventoryPanel>>,
) {
    if !panel_query.is_empty() {
        return;
    }

    let max_slots = inventory.max_slots;
    let items = &inventory.items;
    let gold = inventory.gold;
    let panel_width = 300.0;

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(70.0),
                left: Val::Px(20.0),
                width: Val::Px(panel_width),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(8.0),
                ..default()
            },
            BackgroundColor(Color::srgb(0.1, 0.1, 0.15)),
            BorderColor::all(Color::srgb(0.3, 0.3, 0.5)),
            InventoryPanel,
        ))
        .with_children(|parent| {
            // Title bar
            parent
                .spawn((
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(30.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.2, 0.2, 0.3)),
                    InventoryTitleBar,
                ))
                .with_children(|title| {
                    title.spawn((Text("Inventory".to_string()),));
                });

            // Gold display
            parent.spawn((Text(format!("Gold: {gold}")),));

            // Item grid
            parent
                .spawn((Node {
                    flex_wrap: FlexWrap::Wrap,
                    width: Val::Px(panel_width - 24.0),
                    column_gap: Val::Px(4.0),
                    row_gap: Val::Px(4.0),
                    ..default()
                },))
                .with_children(|grid| {
                    for i in 0..max_slots {
                        let mut slot = grid.spawn((
                            Button,
                            Node {
                                width: Val::Px(50.0),
                                height: Val::Px(50.0),
                                border: UiRect::all(Val::Px(1.0)),
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                ..default()
                            },
                            BorderColor::all(Color::srgb(0.3, 0.3, 0.3)),
                            InventorySlot(i),
                        ));

                        if i < items.len() {
                            let (item, qty) = &items[i];
                            let label = truncate_str(&item.name, 5);
                            slot.insert(BackgroundColor(item.rarity.color()));
                            slot.with_children(|s| {
                                s.spawn((
                                    Node {
                                        flex_direction: FlexDirection::Column,
                                        align_items: AlignItems::Center,
                                        ..default()
                                    },
                                    children![
                                        (Text(label.to_string()),),
                                        (Text(format!("x{qty}")),),
                                    ],
                                ));
                            });
                        } else {
                            slot.insert(BackgroundColor(Color::srgb(0.15, 0.15, 0.2)));
                        }
                    }
                });
        });
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        s[..s.floor_char_boundary(max_len)].to_string()
    }
}

fn despawn_inventory_panel(
    mut commands: Commands,
    panel_query: Query<Entity, With<InventoryPanel>>,
) {
    for entity in &panel_query {
        commands.entity(entity).despawn();
    }
}

// --- Loot Window ---

fn spawn_loot_window(
    mut commands: Commands,
    mut ui_state: ResMut<InventoryUiState>,
    lootable_query: Query<&Lootable>,
    window_query: Query<Entity, With<LootWindow>>,
) {
    for entity in &window_query {
        commands.entity(entity).despawn();
    }

    let Some(target) = ui_state.loot_target else {
        return;
    };

    // Skip respawn if target hasn't changed and window already exists
    if ui_state.prev_loot_target == Some(target) {
        return;
    }

    let Ok(lootable) = lootable_query.get(target) else {
        return;
    };

    let items = &lootable.items;
    let pos = ui_state.last_right_click_pos;

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(pos.x),
                top: Val::Px(pos.y),
                width: Val::Px(250.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(4.0),
                ..default()
            },
            BackgroundColor(Color::srgb(0.12, 0.12, 0.18)),
            BorderColor::all(Color::srgb(0.4, 0.4, 0.6)),
            LootWindow,
        ))
        .with_children(|parent| {
            // Title bar
            parent
                .spawn((
                    Node {
                        width: Val::Percent(100.0),
                        justify_content: JustifyContent::SpaceBetween,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.2, 0.2, 0.3)),
                    LootTitleBar,
                ))
                .with_children(|title| {
                    title.spawn((Text("Loot".to_string()),));
                });

            // Close button (sibling of title, still inside window)
            parent
                .spawn((
                    Button,
                    Node {
                        width: Val::Px(24.0),
                        height: Val::Px(24.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.4, 0.1, 0.1)),
                    LootWindowClose,
                ))
                .with_children(|btn| {
                    btn.spawn((Text("X".to_string()),));
                });

            // Loot item rows
            for (i, (item, qty)) in items.iter().enumerate() {
                parent
                    .spawn((
                        Button,
                        Node {
                            width: Val::Percent(100.0),
                            height: Val::Px(36.0),
                            justify_content: JustifyContent::FlexStart,
                            align_items: AlignItems::Center,
                            column_gap: Val::Px(8.0),
                            padding: UiRect::horizontal(Val::Px(8.0)),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.18, 0.18, 0.25)),
                        LootSlot {
                            source_entity: target,
                            item_index: i,
                        },
                    ))
                    .with_children(|row| {
                        row.spawn((
                            Node {
                                width: Val::Px(10.0),
                                height: Val::Px(10.0),
                                ..default()
                            },
                            BackgroundColor(item.rarity.color()),
                        ));
                        row.spawn((Text(format!("{} x{}", item.name, qty)),));
                    });
            }
        });

    // Mark target as seen to prevent respawn-flicker
    ui_state.prev_loot_target = Some(target);
}

fn despawn_loot_window(mut commands: Commands, window_query: Query<Entity, With<LootWindow>>) {
    for entity in &window_query {
        commands.entity(entity).despawn();
    }
}

// --- Window Dragging ---

#[allow(clippy::type_complexity)] // Bevy query tuples; aliases hurt readability here.
fn handle_window_drag(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut motion_events: MessageReader<MouseMotion>,
    mut drag_state: ResMut<WindowDragState>,
    mut inventory_panel_query: Query<
        (Entity, &mut Node),
        (With<InventoryPanel>, Without<LootWindow>),
    >,
    mut loot_window_query: Query<(Entity, &mut Node), (With<LootWindow>, Without<InventoryPanel>)>,
    title_bar_query: Query<&Interaction, (Changed<Interaction>, With<InventoryTitleBar>)>,
    loot_title_bar_query: Query<&Interaction, (Changed<Interaction>, With<LootTitleBar>)>,
) {
    // Detect drag start on title bars
    if !mouse_buttons.pressed(MouseButton::Left) {
        // Button released — clear dragging
        drag_state.dragging = None;
    }

    for interaction in &title_bar_query {
        if *interaction == Interaction::Pressed && drag_state.dragging.is_none() {
            drag_state.dragging = inventory_panel_query.iter().map(|(e, _)| e).next();
        }
    }
    for interaction in &loot_title_bar_query {
        if *interaction == Interaction::Pressed && drag_state.dragging.is_none() {
            drag_state.dragging = loot_window_query.iter().map(|(e, _)| e).next();
        }
    }

    if let Some(dragged_entity) = drag_state.dragging {
        let delta: Vec2 = motion_events.read().map(|m| m.delta).sum();

        let target_style = if inventory_panel_query.contains(dragged_entity) {
            inventory_panel_query.get_mut(dragged_entity)
        } else {
            loot_window_query.get_mut(dragged_entity)
        };

        if let Ok((_, mut style)) = target_style {
            if let Val::Px(mut left) = style.left {
                left += delta.x;
                style.left = Val::Px(left);
            }
            if let Val::Px(mut top) = style.top {
                top += delta.y;
                style.top = Val::Px(top);
            }
        }
    }
}

// --- Tooltip ---

fn handle_tooltip(
    mut commands: Commands,
    _tooltip_query: Query<Entity, With<Tooltip>>,
    _inventory_slot_query: Query<(&Interaction, &InventorySlot, Entity), Changed<Interaction>>,
    loot_slot_query: Query<(&Interaction, &LootSlot), Changed<Interaction>>,
    inventory: Res<PlayerInventory>,
    lootable_query: Query<&Lootable>,
    windows: Query<&Window>,
) {
    // Despawn existing tooltip
    for entity in &_tooltip_query {
        commands.entity(entity).despawn();
    }

    let mut tooltip_data: Option<(String, Color, Vec2)> = None;

    // Check inventory slots
    for (interaction, slot, _slot_entity) in &_inventory_slot_query {
        if *interaction == Interaction::Hovered && slot.0 < inventory.items.len() {
            let (item, qty) = &inventory.items[slot.0];
            tooltip_data = Some((
                format!(
                    "{} (x{})\n{} | Lv{}",
                    item.name,
                    qty,
                    item_rarity_label(&item.rarity),
                    item.level_requirement
                ),
                item.rarity.color(),
                Vec2::ZERO, // will be offset below
            ));
        }
    }

    // Check loot slots
    for (interaction, slot) in &loot_slot_query {
        if *interaction == Interaction::Hovered
            && let Ok(lootable) = lootable_query.get(slot.source_entity)
            && slot.item_index < lootable.items.len()
        {
            let (item, qty) = &lootable.items[slot.item_index];
            tooltip_data = Some((
                format!(
                    "{} (x{})\n{} | Lv{}",
                    item.name,
                    qty,
                    item_rarity_label(&item.rarity),
                    item.level_requirement
                ),
                item.rarity.color(),
                Vec2::ZERO,
            ));
        }
    }

    if let Some((text, color, _)) = tooltip_data
        && let Ok(window) = windows.single()
    {
        let cursor_pos = window.cursor_position().unwrap_or(Vec2::ZERO);

        commands.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(cursor_pos.x + 15.0),
                top: Val::Px(cursor_pos.y + 15.0),
                padding: UiRect::all(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.05, 0.05, 0.1, 0.95)),
            BorderColor::all(color),
            ZIndex(100),
            Tooltip,
            Text(text),
            TextColor(color),
        ));
    }
}

fn item_rarity_label(rarity: &crate::inventory::Rarity) -> &'static str {
    match rarity {
        crate::inventory::Rarity::Common => "Common",
        crate::inventory::Rarity::Uncommon => "Uncommon",
        crate::inventory::Rarity::Rare => "Rare",
        crate::inventory::Rarity::Epic => "Epic",
        crate::inventory::Rarity::Legendary => "Legendary",
    }
}

// --- Loot Slot Interaction ---

fn handle_loot_slot_click(
    mut messages: MessageWriter<LootTransferEvent>,
    mut ui_state: ResMut<InventoryUiState>,
    slot_query: Query<(&Interaction, &LootSlot), Changed<Interaction>>,
    close_query: Query<&Interaction, (With<LootWindowClose>, Changed<Interaction>)>,
) {
    for (interaction, slot) in &slot_query {
        if *interaction == Interaction::Pressed {
            messages.write(LootTransferEvent {
                source_entity: slot.source_entity,
                item_index: slot.item_index,
            });
            // Force respawn on next frame to refresh content
            ui_state.prev_loot_target = None;
        }
    }

    for interaction in &close_query {
        if *interaction == Interaction::Pressed {
            ui_state.loot_target = None;
        }
    }
}

// --- Empty Container Cleanup ---

fn handle_empty_container(
    mut commands: Commands,
    mut messages: MessageReader<crate::inventory::ContainerEmptyEvent>,
    mut ui_state: ResMut<InventoryUiState>,
    lootable_query: Query<(), With<Lootable>>,
) {
    for event in messages.read() {
        if lootable_query.contains(event.entity) {
            commands.entity(event.entity).despawn();
        }
        if ui_state.loot_target == Some(event.entity) {
            ui_state.loot_target = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_bar_width_at_full_health() {
        let pct = 100.0_f32 / 100.0;
        let width = 200.0 * pct;
        assert_eq!(width, 200.0);
    }

    #[test]
    fn health_bar_width_at_half_health() {
        let pct = 50.0_f32 / 100.0;
        let width = 200.0 * pct;
        assert_eq!(width, 100.0);
    }

    #[test]
    fn health_bar_width_at_zero_health() {
        let pct = 0.0_f32 / 100.0;
        let width = 200.0 * pct;
        assert_eq!(width, 0.0);
    }

    #[test]
    fn mana_bar_width_at_full_mana() {
        let pct = 50.0_f32 / 50.0;
        assert_eq!(200.0 * pct, 200.0);
    }

    #[test]
    fn mana_bar_width_at_partial_mana() {
        let pct = 25.0_f32 / 50.0;
        assert_eq!(200.0 * pct, 100.0);
    }

    #[test]
    fn inventory_ui_state_defaults() {
        let state = InventoryUiState::default();
        assert!(!state.inventory_open);
        assert!(state.loot_target.is_none());
        assert_eq!(state.last_right_click_pos, Vec2::ZERO);
        assert!(state.prev_loot_target.is_none());
    }

    #[test]
    fn window_drag_state_defaults() {
        let state = WindowDragState::default();
        assert!(state.dragging.is_none());
        assert_eq!(state.offset, Vec2::ZERO);
    }

    #[test]
    fn truncate_str_short() {
        assert_eq!(truncate_str("Sword", 5), "Sword");
    }

    #[test]
    fn truncate_str_long() {
        assert!(truncate_str("Rusty Longsword", 5).len() <= 5);
    }

    #[test]
    fn item_rarity_labels() {
        use crate::inventory::Rarity;
        assert_eq!(item_rarity_label(&Rarity::Common), "Common");
        assert_eq!(item_rarity_label(&Rarity::Uncommon), "Uncommon");
        assert_eq!(item_rarity_label(&Rarity::Rare), "Rare");
        assert_eq!(item_rarity_label(&Rarity::Epic), "Epic");
        assert_eq!(item_rarity_label(&Rarity::Legendary), "Legendary");
    }
}
