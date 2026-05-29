use bevy::prelude::*;

use crate::scene_manager::AppScene;

pub struct NetworkingPlugin;

#[derive(Resource, Default)]
pub struct NetworkState {
    pub connected: bool,
    pub player_count: u32,
    pub server_address: String,
}

#[derive(Message)]
pub struct ChatMessage {
    pub sender: String,
    pub message: String,
    pub channel: ChatChannel,
}

#[derive(Clone, Debug, Reflect)]
pub enum ChatChannel {
    Say,
    Yell,
    General,
    Trade,
    Guild,
    Party,
    Whisper,
}

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct NetworkingSet;

impl Plugin for NetworkingPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<ChatChannel>()
            .insert_resource(NetworkState::default())
            .add_message::<ChatMessage>()
            .configure_sets(Update, NetworkingSet)
            .add_systems(
                Update,
                handle_chat
                    .in_set(NetworkingSet)
                    .run_if(in_state(AppScene::AdventureGame)),
            );
    }
}

fn handle_chat(mut messages: MessageReader<ChatMessage>) {
    for msg in messages.read() {
        info!("[{:?}] {}: {}", msg.channel, msg.sender, msg.message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn network_state_defaults() {
        let state = NetworkState::default();
        assert!(!state.connected);
        assert_eq!(state.player_count, 0);
        assert!(state.server_address.is_empty());
    }

    #[test]
    fn chat_message_processing_does_not_panic() {
        let mut app = test_app();
        app.add_message::<ChatMessage>();
        app.add_systems(Update, handle_chat);

        let mut msgs = app.world_mut().resource_mut::<Messages<ChatMessage>>();
        msgs.write(ChatMessage {
            sender: "Player1".into(),
            message: "Hello!".into(),
            channel: ChatChannel::Say,
        });
        msgs.write(ChatMessage {
            sender: "Player2".into(),
            message: "TRADE WTS".into(),
            channel: ChatChannel::Trade,
        });

        app.update();
    }

    #[test]
    fn chat_channel_variants() {
        let channels = [
            ChatChannel::Say,
            ChatChannel::Yell,
            ChatChannel::General,
            ChatChannel::Trade,
            ChatChannel::Guild,
            ChatChannel::Party,
            ChatChannel::Whisper,
        ];
        assert_eq!(channels.len(), 7);
    }
}
