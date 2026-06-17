//! Inventory loot / equip / container tests (split from mod.rs per the test-module convention).

use super::*;
use crate::test_utils::*;

fn test_item(name: &str) -> Item {
    Item {
        name: name.into(),
        item_type: ItemType::Weapon,
        rarity: Rarity::Common,
        level_requirement: 1,
    }
}

#[test]
fn loot_adds_item_to_inventory() {
    let mut app = test_app();
    app.add_message::<LootMessage>();
    app.insert_resource(PlayerInventory {
        max_slots: 20,
        ..default()
    });
    app.add_systems(Update, handle_loot);

    app.world_mut()
        .resource_mut::<Messages<LootMessage>>()
        .write(LootMessage {
            item: test_item("Sword"),
            quantity: 1,
        });

    app.update();

    let inv = app.world().resource::<PlayerInventory>();
    assert_eq!(inv.items.len(), 1);
    assert_eq!(inv.items[0].0.name, "Sword");
    assert_eq!(inv.items[0].1, 1);
}

#[test]
fn loot_rejected_when_bag_full() {
    let mut app = test_app();
    app.add_message::<LootMessage>();
    app.insert_resource(PlayerInventory {
        items: (0..20)
            .map(|i| (test_item(&format!("Item {i}")), 1))
            .collect(),
        gold: 0,
        max_slots: 20,
    });
    app.add_systems(Update, handle_loot);

    app.world_mut()
        .resource_mut::<Messages<LootMessage>>()
        .write(LootMessage {
            item: test_item("Overflow"),
            quantity: 1,
        });

    app.update();

    let inv = app.world().resource::<PlayerInventory>();
    assert_eq!(inv.items.len(), 20);
}

#[test]
fn multiple_loot_events_same_frame() {
    let mut app = test_app();
    app.add_message::<LootMessage>();
    app.insert_resource(PlayerInventory {
        max_slots: 20,
        ..default()
    });
    app.add_systems(Update, handle_loot);

    let mut msgs = app.world_mut().resource_mut::<Messages<LootMessage>>();
    msgs.write(LootMessage {
        item: test_item("Axe"),
        quantity: 1,
    });
    msgs.write(LootMessage {
        item: test_item("Shield"),
        quantity: 2,
    });

    app.update();

    let inv = app.world().resource::<PlayerInventory>();
    assert_eq!(inv.items.len(), 2);
}

#[test]
fn inventory_defaults() {
    let inv = PlayerInventory::default();
    assert!(inv.items.is_empty());
    assert_eq!(inv.gold, 0);
    assert_eq!(inv.max_slots, 20);
}

#[test]
fn loot_transfer_moves_item_from_lootable() {
    let mut app = test_app();
    app.add_message::<LootTransferMessage>();
    app.add_message::<ContainerEmptyMessage>();
    app.insert_resource(PlayerInventory::default());
    app.add_systems(Update, handle_loot_transfer);

    let entity = app
        .world_mut()
        .spawn(Lootable {
            items: vec![(test_item("Sword"), 1), (test_item("Potion"), 3)],
        })
        .id();

    app.world_mut()
        .resource_mut::<Messages<LootTransferMessage>>()
        .write(LootTransferMessage {
            source_entity: entity,
            item_index: 0,
        });

    app.update();

    let inv = app.world().resource::<PlayerInventory>();
    assert_eq!(inv.items.len(), 1);
    assert_eq!(inv.items[0].0.name, "Sword");

    let lootable = app.world().get::<Lootable>(entity).unwrap();
    assert_eq!(lootable.items.len(), 1);
    assert_eq!(lootable.items[0].0.name, "Potion");
}

#[test]
fn loot_transfer_rejected_when_full() {
    let mut app = test_app();
    app.add_message::<LootTransferMessage>();
    app.add_message::<ContainerEmptyMessage>();
    app.insert_resource(PlayerInventory {
        items: (0..20)
            .map(|i| (test_item(&format!("Item {i}")), 1))
            .collect(),
        gold: 0,
        max_slots: 20,
    });
    app.add_systems(Update, handle_loot_transfer);

    let entity = app
        .world_mut()
        .spawn(Lootable {
            items: vec![(test_item("Overflow"), 1)],
        })
        .id();

    app.world_mut()
        .resource_mut::<Messages<LootTransferMessage>>()
        .write(LootTransferMessage {
            source_entity: entity,
            item_index: 0,
        });

    app.update();

    let inv = app.world().resource::<PlayerInventory>();
    assert_eq!(inv.items.len(), 20);

    let lootable = app.world().get::<Lootable>(entity).unwrap();
    assert_eq!(lootable.items.len(), 1);
}

#[test]
fn rarity_colors() {
    assert_eq!(Rarity::Common.color(), Color::srgb(0.6, 0.6, 0.6));
    assert_eq!(Rarity::Uncommon.color(), Color::srgb(0.1, 0.8, 0.1));
    assert_eq!(Rarity::Rare.color(), Color::srgb(0.2, 0.4, 1.0));
    assert_eq!(Rarity::Epic.color(), Color::srgb(0.6, 0.2, 0.8));
    assert_eq!(Rarity::Legendary.color(), Color::srgb(1.0, 0.5, 0.0));
}
