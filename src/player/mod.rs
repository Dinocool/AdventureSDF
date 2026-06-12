//! Player stats + the legacy on-foot character marker.
//!
//! The voxel engine's "walk the cubes" mode now lives in [`crate::voxel::physics`] (a first-person
//! kinematic controller on the engine-agnostic `rapier3d` solver). The old AdventureGame third-person
//! controller — which was driven by `bevy_rapier` — has been removed; what remains here are the gameplay
//! stat components ([`Health`]/[`Mana`]/…) and a plain [`CharacterController`] marker the (currently
//! dormant) AdventureGame camera still follows.

use bevy::prelude::*;

pub struct PlayerPlugin;

#[derive(Component)]
#[require(Health, Mana, MovementSpeed, PlayerName, PlayerLevel)]
pub struct Player;

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct Health {
    pub current: f32,
    pub max: f32,
}

impl Health {
    pub fn full(max: f32) -> Self {
        Self { current: max, max }
    }
}

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct Mana {
    pub current: f32,
    pub max: f32,
}

impl Mana {
    pub fn full(max: f32) -> Self {
        Self { current: max, max }
    }
}

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct MovementSpeed(pub f32);

impl Default for MovementSpeed {
    fn default() -> Self {
        Self(5.0)
    }
}

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct PlayerName(pub String);

impl Default for PlayerName {
    fn default() -> Self {
        Self("Adventurer".into())
    }
}

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct PlayerLevel(pub u32);

impl Default for PlayerLevel {
    fn default() -> Self {
        Self(1)
    }
}

/// Legacy on-foot character state (vertical velocity). The AdventureGame third-person movement that used
/// this was `bevy_rapier`-driven and has been removed in the voxel-RT rebuild; the component stays as a
/// plain marker the AdventureGame camera's follow logic still queries. The live "walk the cubes" path is
/// [`crate::voxel::physics`].
#[derive(Component)]
pub struct CharacterController {
    pub vertical_velocity: f32,
}

#[derive(Message)]
pub struct PlayerLevelUp {
    pub new_level: u32,
}

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Health>()
            .register_type::<Mana>()
            .register_type::<MovementSpeed>()
            .register_type::<PlayerName>()
            .register_type::<PlayerLevel>()
            .add_message::<PlayerLevelUp>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_default_full() {
        let health = Health {
            current: 100.0,
            max: 100.0,
        };
        assert_eq!(health.current, health.max);
    }

    #[test]
    fn mana_default_full() {
        let mana = Mana {
            current: 50.0,
            max: 50.0,
        };
        assert_eq!(mana.current, mana.max);
    }

    #[test]
    fn character_controller_default() {
        let cc = CharacterController {
            vertical_velocity: 0.0,
        };
        assert_eq!(cc.vertical_velocity, 0.0);
    }

    #[test]
    fn movement_speed_default() {
        assert_eq!(MovementSpeed::default().0, 5.0);
    }
}
