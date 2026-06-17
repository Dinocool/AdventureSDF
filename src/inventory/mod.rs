use bevy::prelude::*;

use crate::scene_manager::AppScene;

pub struct InventoryPlugin;

#[derive(Clone, Reflect)]
pub struct Item {
    pub name: String,
    pub item_type: ItemType,
    pub rarity: Rarity,
    pub level_requirement: u32,
}

#[derive(Clone, Reflect)]
pub enum ItemType {
    Weapon,
    Armor,
    Consumable,
    QuestItem,
    Material,
}

#[derive(Clone, Reflect)]
pub enum Rarity {
    Common,
    Uncommon,
    Rare,
    Epic,
    Legendary,
}

impl Rarity {
    pub fn color(&self) -> Color {
        match self {
            Rarity::Common => Color::srgb(0.6, 0.6, 0.6),
            Rarity::Uncommon => Color::srgb(0.1, 0.8, 0.1),
            Rarity::Rare => Color::srgb(0.2, 0.4, 1.0),
            Rarity::Epic => Color::srgb(0.6, 0.2, 0.8),
            Rarity::Legendary => Color::srgb(1.0, 0.5, 0.0),
        }
    }
}

#[derive(Resource)]
pub struct PlayerInventory {
    pub items: Vec<(Item, u32)>,
    pub gold: u32,
    pub max_slots: usize,
}

impl Default for PlayerInventory {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            gold: 0,
            max_slots: 20,
        }
    }
}

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct Lootable {
    pub items: Vec<(Item, u32)>,
}

#[derive(Message)]
pub struct LootMessage {
    pub item: Item,
    pub quantity: u32,
}

#[derive(Message)]
pub struct LootTransferMessage {
    pub source_entity: Entity,
    pub item_index: usize,
}

#[derive(Message)]
pub struct ContainerEmptyMessage {
    pub entity: Entity,
}

#[derive(Message)]
pub struct EquipMessage {
    pub slot: EquipSlot,
    pub item_index: usize,
}

#[derive(Clone, Reflect)]
pub enum EquipSlot {
    Head,
    Chest,
    Legs,
    Feet,
    Weapon,
    OffHand,
}

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct InventorySet;

impl Plugin for InventoryPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Item>()
            .register_type::<ItemType>()
            .register_type::<Rarity>()
            .register_type::<EquipSlot>()
            .register_type::<Lootable>()
            .init_resource::<PlayerInventory>()
            .add_message::<LootMessage>()
            .add_message::<LootTransferMessage>()
            .add_message::<ContainerEmptyMessage>()
            .add_message::<EquipMessage>()
            .configure_sets(Update, InventorySet)
            .add_systems(
                Update,
                (handle_loot, handle_loot_transfer, handle_equip)
                    .in_set(InventorySet)
                    .run_if(in_state(AppScene::AdventureGame)),
            );
    }
}

fn handle_loot(mut messages: MessageReader<LootMessage>, mut inventory: ResMut<PlayerInventory>) {
    for event in messages.read() {
        if inventory.items.len() < inventory.max_slots {
            inventory.items.push((event.item.clone(), event.quantity));
        }
    }
}

fn handle_loot_transfer(
    mut messages: MessageReader<LootTransferMessage>,
    mut empty_messages: MessageWriter<ContainerEmptyMessage>,
    mut lootable_query: Query<&mut Lootable>,
    mut inventory: ResMut<PlayerInventory>,
) {
    for event in messages.read() {
        if inventory.items.len() >= inventory.max_slots {
            continue;
        }
        let Ok(mut lootable) = lootable_query.get_mut(event.source_entity) else {
            continue;
        };
        if event.item_index >= lootable.items.len() {
            continue;
        }
        let (item, qty) = lootable.items.remove(event.item_index);
        inventory.items.push((item, qty));
        if lootable.items.is_empty() {
            empty_messages.write(ContainerEmptyMessage {
                entity: event.source_entity,
            });
        }
    }
}

fn handle_equip(mut messages: MessageReader<EquipMessage>) {
    for _event in messages.read() {
        // Handle equipping items
    }
}

#[cfg(test)]
mod tests;
