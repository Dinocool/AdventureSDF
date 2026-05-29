use bevy::prelude::*;

use crate::player::Health;
use crate::scene_manager::AppScene;
use crate::world::Npc;

pub struct CombatPlugin;

#[derive(Component)]
pub struct CombatTarget;

#[derive(Resource, Default)]
pub struct CombatState {
    pub in_combat: bool,
    pub target_name: Option<String>,
}

#[derive(Message)]
pub struct DamageEvent {
    pub target: Entity,
    pub amount: f32,
    pub damage_type: DamageType,
}

#[derive(Message)]
pub struct AbilityCastEvent {
    pub caster: Entity,
    pub ability: Ability,
}

#[derive(Clone, Reflect)]
pub enum DamageType {
    Physical,
    Magical,
    Fire,
    Frost,
}

#[derive(Clone, Reflect)]
pub enum Ability {
    MeleeAttack,
    Fireball,
    Heal,
}

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct CombatSet;

impl Plugin for CombatPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<DamageType>()
            .register_type::<Ability>()
            .insert_resource(CombatState::default())
            .add_message::<DamageEvent>()
            .add_message::<AbilityCastEvent>()
            .configure_sets(Update, CombatSet)
            .add_systems(
                Update,
                (handle_damage, handle_ability_cast)
                    .in_set(CombatSet)
                    .run_if(in_state(AppScene::AdventureGame)),
            );
    }
}

fn handle_damage(
    mut messages: MessageReader<DamageEvent>,
    mut health_query: Query<&mut Health>,
    mut npc_query: Query<&mut Npc>,
) {
    for event in messages.read() {
        if let Ok(mut health) = health_query.get_mut(event.target) {
            health.current = (health.current - event.amount).max(0.0);
        }
        if let Ok(_npc) = npc_query.get_mut(event.target) {
            // Apply damage to NPC
        }
    }
}

fn handle_ability_cast(mut messages: MessageReader<AbilityCastEvent>) {
    for _event in messages.read() {
        // Handle ability casting
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn damage_reduces_player_health() {
        let mut app = test_app();
        app.add_message::<DamageEvent>();
        let player = spawn_test_player(app.world_mut());
        app.add_systems(Update, handle_damage);

        app.world_mut()
            .resource_mut::<Messages<DamageEvent>>()
            .write(DamageEvent {
                target: player,
                amount: 30.0,
                damage_type: DamageType::Physical,
            });

        app.update();

        let health = app.world().get::<Health>(player).unwrap().current;
        assert_eq!(health, 70.0);
    }

    #[test]
    fn damage_clamps_health_at_zero() {
        let mut app = test_app();
        app.add_message::<DamageEvent>();
        let player = spawn_test_player(app.world_mut());
        app.add_systems(Update, handle_damage);

        app.world_mut()
            .resource_mut::<Messages<DamageEvent>>()
            .write(DamageEvent {
                target: player,
                amount: 999.0,
                damage_type: DamageType::Fire,
            });

        app.update();

        assert_eq!(app.world().get::<Health>(player).unwrap().current, 0.0);
    }

    #[test]
    fn multiple_damage_events_accumulate() {
        let mut app = test_app();
        app.add_message::<DamageEvent>();
        let player = spawn_test_player(app.world_mut());
        app.add_systems(Update, handle_damage);

        let mut msgs = app.world_mut().resource_mut::<Messages<DamageEvent>>();
        msgs.write(DamageEvent {
            target: player,
            amount: 20.0,
            damage_type: DamageType::Physical,
        });
        msgs.write(DamageEvent {
            target: player,
            amount: 30.0,
            damage_type: DamageType::Magical,
        });

        app.update();

        assert_eq!(app.world().get::<Health>(player).unwrap().current, 50.0);
    }

    #[test]
    fn damage_to_nonexistent_entity_does_not_panic() {
        let mut app = test_app();
        app.add_message::<DamageEvent>();
        app.add_systems(Update, handle_damage);

        app.world_mut()
            .resource_mut::<Messages<DamageEvent>>()
            .write(DamageEvent {
                target: Entity::from_bits(9999),
                amount: 10.0,
                damage_type: DamageType::Frost,
            });

        app.update();
    }

    #[test]
    fn combat_state_defaults() {
        let state = CombatState::default();
        assert!(!state.in_combat);
        assert!(state.target_name.is_none());
    }
}
