use bevy::pbr::wireframe::{Wireframe, WireframeColor};
use bevy::prelude::*;

use crate::soul_scene::ReflectSerializeSkip;

#[derive(States, Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum AppScene {
    #[default]
    SdfEditor,
    WireframeTest,
    AdventureGame,
}

#[derive(Component, Reflect, Default)]
#[reflect(Component, SerializeSkip)]
pub struct SceneEntity;

/// Marks EDITOR-owned infrastructure (e.g. the viewport camera) — distinct from
/// [`SceneEntity`] scene content. Editor entities are never serialized into a `.scene` and
/// are never despawned by scene load/clear/switch; they persist across scene transitions.
#[derive(Component, Reflect, Default)]
#[reflect(Component, SerializeSkip)]
pub struct EditorEntity;

#[derive(Resource, Default)]
pub struct MenuOpen(pub bool);

#[derive(Component)]
struct MenuRoot;

#[derive(Component)]
struct SdfEditorButton;

#[derive(Component)]
struct AdventureGameButton;

#[derive(Component)]
struct WireframeTestButton;

pub struct SceneManagerPlugin;

impl Plugin for SceneManagerPlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<AppScene>()
            .init_resource::<MenuOpen>()
            .register_type::<SceneEntity>()
            .register_type::<EditorEntity>()
            .add_systems(Update, toggle_menu)
            .add_systems(Update, handle_menu_buttons)
            .add_systems(OnEnter(AppScene::WireframeTest), setup_wireframe_test)
            .add_systems(OnExit(AppScene::WireframeTest), cleanup_scene_entities)
            .add_systems(OnExit(AppScene::SdfEditor), cleanup_scene_entities)
            .add_systems(OnExit(AppScene::AdventureGame), cleanup_scene_entities);
    }
}

fn setup_wireframe_test(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 2.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        SceneEntity,
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_x(-0.5)),
        SceneEntity,
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.8, 0.4),
            unlit: true,
            ..default()
        })),
        Wireframe,
        WireframeColor {
            color: Color::srgb(0.3, 0.8, 0.4),
        },
        Transform::from_xyz(0.0, 0.5, 0.0),
        SceneEntity,
    ));
}

fn toggle_menu(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut menu_open: ResMut<MenuOpen>,
    mut commands: Commands,
    menu_query: Query<Entity, With<MenuRoot>>,
    current_scene: Res<State<AppScene>>,
) {
    if !keyboard.just_pressed(KeyCode::Escape) {
        return;
    }
    menu_open.0 = !menu_open.0;

    if menu_open.0 {
        spawn_menu(&mut commands, current_scene.get());
    } else {
        for entity in &menu_query {
            commands.entity(entity).despawn();
        }
    }
}

fn spawn_menu(commands: &mut Commands, current_scene: &AppScene) {
    commands
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                position_type: PositionType::Absolute,
                top: Val::Px(0.0),
                left: Val::Px(0.0),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.7)),
            MenuRoot,
        ))
        .with_children(|parent| {
            parent
                .spawn((
                    Node {
                        flex_direction: FlexDirection::Column,
                        padding: UiRect::all(Val::Px(20.0)),
                        row_gap: Val::Px(10.0),
                        column_gap: Val::Px(10.0),
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.15, 0.15, 0.2)),
                ))
                .with_children(|panel| {
                    panel.spawn((
                        Text::new("Scene Selector"),
                        TextFont {
                            font_size: 24.0,
                            ..default()
                        },
                        TextColor(Color::WHITE),
                        Node {
                            margin: UiRect::bottom(Val::Px(10.0)),
                            ..default()
                        },
                    ));

                    let wf_color = match current_scene {
                        AppScene::WireframeTest => Color::srgb(0.2, 0.5, 0.8),
                        _ => Color::srgb(0.3, 0.3, 0.4),
                    };
                    panel
                        .spawn((
                            Button,
                            Node {
                                width: Val::Px(200.0),
                                height: Val::Px(40.0),
                                align_items: AlignItems::Center,
                                justify_content: JustifyContent::Center,
                                border: UiRect::all(Val::Px(4.0)),
                                ..default()
                            },
                            BackgroundColor(wf_color),
                            WireframeTestButton,
                        ))
                        .with_child((
                            Text::new("Wireframe Test"),
                            TextFont {
                                font_size: 16.0,
                                ..default()
                            },
                            TextColor(Color::WHITE),
                        ));

                    let sdf_color = match current_scene {
                        AppScene::SdfEditor => Color::srgb(0.2, 0.5, 0.8),
                        _ => Color::srgb(0.3, 0.3, 0.4),
                    };
                    panel
                        .spawn((
                            Button,
                            Node {
                                width: Val::Px(200.0),
                                height: Val::Px(40.0),
                                align_items: AlignItems::Center,
                                justify_content: JustifyContent::Center,
                                border: UiRect::all(Val::Px(4.0)),
                                ..default()
                            },
                            BackgroundColor(sdf_color),
                            SdfEditorButton,
                        ))
                        .with_child((
                            Text::new("SDF Editor"),
                            TextFont {
                                font_size: 16.0,
                                ..default()
                            },
                            TextColor(Color::WHITE),
                        ));

                    let adventure_color = match current_scene {
                        AppScene::AdventureGame => Color::srgb(0.2, 0.5, 0.8),
                        _ => Color::srgb(0.3, 0.3, 0.4),
                    };
                    panel
                        .spawn((
                            Button,
                            Node {
                                width: Val::Px(200.0),
                                height: Val::Px(40.0),
                                align_items: AlignItems::Center,
                                justify_content: JustifyContent::Center,
                                border: UiRect::all(Val::Px(4.0)),
                                ..default()
                            },
                            BackgroundColor(adventure_color),
                            AdventureGameButton,
                        ))
                        .with_child((
                            Text::new("Adventure Game"),
                            TextFont {
                                font_size: 16.0,
                                ..default()
                            },
                            TextColor(Color::WHITE),
                        ));
                });
        });
}

#[allow(clippy::too_many_arguments)] // Bevy system params; splitting is artificial.
fn handle_menu_buttons(
    mut commands: Commands,
    query_wf: Query<&Interaction, (With<WireframeTestButton>, Changed<Interaction>)>,
    query_sdf: Query<&Interaction, (With<SdfEditorButton>, Changed<Interaction>)>,
    query_adventure: Query<&Interaction, (With<AdventureGameButton>, Changed<Interaction>)>,
    menu_query: Query<Entity, With<MenuRoot>>,
    mut menu_open: ResMut<MenuOpen>,
    current_scene: Res<State<AppScene>>,
    mut next_scene: ResMut<NextState<AppScene>>,
) {
    for interaction in &query_wf {
        if *interaction == Interaction::Pressed && *current_scene.get() != AppScene::WireframeTest {
            next_scene.set(AppScene::WireframeTest);
            close_menu(&mut commands, &menu_query, &mut menu_open);
            return;
        }
    }
    for interaction in &query_sdf {
        if *interaction == Interaction::Pressed && *current_scene.get() != AppScene::SdfEditor {
            next_scene.set(AppScene::SdfEditor);
            close_menu(&mut commands, &menu_query, &mut menu_open);
            return;
        }
    }
    for interaction in &query_adventure {
        if *interaction == Interaction::Pressed && *current_scene.get() != AppScene::AdventureGame {
            next_scene.set(AppScene::AdventureGame);
            close_menu(&mut commands, &menu_query, &mut menu_open);
            return;
        }
    }
}

fn close_menu(
    commands: &mut Commands,
    menu_query: &Query<Entity, With<MenuRoot>>,
    menu_open: &mut MenuOpen,
) {
    menu_open.0 = false;
    for entity in menu_query {
        commands.entity(entity).despawn();
    }
}

fn cleanup_scene_entities(mut commands: Commands, entities: Query<Entity, With<SceneEntity>>) {
    for entity in &entities {
        commands.entity(entity).despawn();
    }
}
