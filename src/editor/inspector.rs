//! Component Inspector (Godot / jackdaw `inspector`): edit the selected entity's
//! components. Generic reflection UI by default (every reflected component shows
//! editable fields), with an override registry so specific components — or
//! specific use cases — can supply a hand-built editor for fine-grained control.

use bevy::prelude::*;
use bevy_egui::egui;

use crate::editor::selection::EditorSelection;

/// Renders a custom editor for one component on `entity`, replacing the generic
/// reflection UI for that component's type. Exclusive `World` access.
pub type ComponentEditorFn = dyn Fn(&mut World, Entity, &mut egui::Ui) + Send + Sync;

/// Registry of per-component-type custom inspector editors, keyed by Bevy type
/// path. A plugin registers an override via [`register_component_editor`]; the
/// Inspector consults it before falling back to the generic reflection UI.
#[derive(Resource, Default)]
pub struct InspectorOverrides {
    editors: std::collections::HashMap<String, Box<ComponentEditorFn>>,
}

/// Register a custom inspector editor for component type `T`. Called from a
/// plugin `build`. The closure runs with exclusive `World` access and should draw
/// (and write back) `T` on the given entity.
pub fn register_component_editor<T: 'static>(
    app: &mut App,
    editor: impl Fn(&mut World, Entity, &mut egui::Ui) + Send + Sync + 'static,
) {
    app.init_resource::<InspectorOverrides>();
    let type_path = std::any::type_name::<T>().to_string();
    app.world_mut()
        .resource_mut::<InspectorOverrides>()
        .editors
        .insert(type_path, Box::new(editor));
}

/// The Inspector tab: shows the selected entity and its components. For each
/// component type with a registered override, the override draws it; otherwise the
/// generic reflection inspector renders all remaining components.
pub fn inspector_ui(world: &mut World, ui: &mut egui::Ui) {
    // The inspector follows the unified selection: a scene entity, an asset file, or
    // nothing.
    match world.resource::<EditorSelection>().clone() {
        EditorSelection::None => {
            ui.weak("No selection. Click an entity in the viewport/Hierarchy, or an asset.");
        }
        EditorSelection::Asset(path) => {
            crate::editor::asset_inspector::asset_inspector_ui(world, &path, ui);
        }
        EditorSelection::Entity(entity) => entity_inspector_ui(world, entity, ui),
    }
}

/// Inspect one scene entity: its name + components (custom overrides then generic
/// reflection).
fn entity_inspector_ui(world: &mut World, entity: Entity, ui: &mut egui::Ui) {
    if world.get_entity(entity).is_err() {
        ui.weak("Selected entity no longer exists.");
        return;
    }

    ui.heading(format!("Entity #{}", entity.index()));

    // Name field at the top: edit (or add) the entity's display name. Rendered here
    // explicitly — and skipped in the generic section below — so it reads as the
    // entity's title rather than just another reflected component.
    name_field_ui(world, entity, ui);

    ui.separator();

    // Run any registered per-component overrides first (curated editors), taking
    // the registry out so the closures get exclusive `&mut World`.
    let (rendered_custom, mut custom_paths) = crate::editor::fs_util::with_registry(
        world,
        |world, overrides: &InspectorOverrides| {
            let mut rendered = false;
            for (type_path, editor) in overrides.editors.iter() {
                // Only show an override if the component is actually present on the entity.
                if entity_has_type_path(world, entity, type_path) {
                    egui::CollapsingHeader::new(short_name(type_path))
                        .default_open(true)
                        .show(ui, |ui| {
                            editor(world, entity, ui);
                        });
                    rendered = true;
                }
            }
            let paths: Vec<String> = overrides.editors.keys().cloned().collect();
            (rendered, paths)
        },
    );
    // Name is rendered by `name_field_ui` above; keep it out of the generic section.
    custom_paths.push(std::any::type_name::<Name>().to_string());
    // GlobalTransform is derived from Transform (which has a custom euler editor); showing
    // its raw affine matrix is noise, so hide it.
    custom_paths.push(std::any::type_name::<GlobalTransform>().to_string());
    // The child hierarchy belongs in the Hierarchy panel, not here. Reflecting `Children`
    // renders a row per child entity, so selecting a node with thousands of children (the
    // stress tower-field root) cost ~200 ms/FRAME in the inspector. Skip both relationship
    // components — they're navigation infrastructure, not editable data.
    custom_paths.push(std::any::type_name::<Children>().to_string());
    custom_paths.push(std::any::type_name::<ChildOf>().to_string());

    if rendered_custom {
        ui.separator();
    }

    // Generic reflection editor for the remaining components, with control the
    // stock `ui_for_entity` doesn't give us: skip zero-field marker components
    // (SceneEntity, SdfVolume…) and default-expand each component's fields.
    generic_components_ui(world, entity, &custom_paths, ui);
}

/// Render every reflected component on `entity` (except those handled by an
/// override or that have no editable fields) as a default-open section. Each
/// component is cloned out, edited in place, and written back via reflection.
fn generic_components_ui(
    world: &mut World,
    entity: Entity,
    skip_paths: &[String],
    ui: &mut egui::Ui,
) {
    // Snapshot the component type paths present on the entity (those with a
    // ReflectComponent and at least one editable field).
    let targets: Vec<String> = {
        let registry = world.resource::<AppTypeRegistry>().read();
        let Ok(entity_ref) = world.get_entity(entity) else {
            return;
        };
        registry
            .iter()
            .filter_map(|reg| {
                let type_path = reg.type_info().type_path();
                if skip_paths.iter().any(|p| p == type_path) {
                    return None;
                }
                let rc = reg.data::<bevy::ecs::reflect::ReflectComponent>()?;
                if !rc.contains(entity_ref) {
                    return None;
                }
                // Skip components with no editable fields (ZST markers like
                // SceneEntity / SdfVolume render as empty headers otherwise).
                if is_zero_field(reg.type_info()) {
                    return None;
                }
                // Skip components tagged `#[reflect(@HideFromInspector)]` (editor-only
                // markers like EditorGizmo that shouldn't be user-editable).
                if is_hidden(reg.type_info()) {
                    return None;
                }
                Some(type_path.to_string())
            })
            .collect()
    };

    if targets.is_empty() {
        ui.weak("No editable components.");
        return;
    }

    for type_path in targets {
        egui::CollapsingHeader::new(short_name(&type_path))
            .default_open(true)
            .show(ui, |ui| {
                edit_component(world, entity, &type_path, ui);
            });
    }
}

/// Whether a reflected type has no fields (a marker): unit struct or a struct/
/// tuple-struct with zero fields. Such components have nothing to edit.
fn is_zero_field(info: &bevy::reflect::TypeInfo) -> bool {
    use bevy::reflect::TypeInfo;
    match info {
        TypeInfo::Struct(s) => s.field_len() == 0,
        TypeInfo::TupleStruct(s) => s.field_len() == 0,
        TypeInfo::Opaque(_) => false,
        _ => false,
    }
}

/// Whether a component is tagged `#[reflect(@HideFromInspector)]` at the container level
/// and so should be skipped by the generic inspector.
fn is_hidden(info: &bevy::reflect::TypeInfo) -> bool {
    use bevy::reflect::TypeInfo;
    use crate::node::HideFromInspector;
    match info {
        TypeInfo::Struct(s) => s.has_attribute::<HideFromInspector>(),
        TypeInfo::TupleStruct(s) => s.has_attribute::<HideFromInspector>(),
        TypeInfo::Enum(s) => s.has_attribute::<HideFromInspector>(),
        _ => false,
    }
}

/// Clone the component value out, edit it with the reflection inspector, and write
/// it back if changed. Cloning sidesteps the borrow conflict between the inspector
/// (which needs `&mut World` for asset lookups) and the live component.
fn edit_component(world: &mut World, entity: Entity, type_path: &str, ui: &mut egui::Ui) {
    // Extract a clone of the current value.
    let mut value: Box<dyn bevy::reflect::Reflect> = {
        let registry = world.resource::<AppTypeRegistry>().read();
        let Some(reg) = registry.get_with_type_path(type_path) else {
            return;
        };
        let Some(rc) = reg.data::<bevy::ecs::reflect::ReflectComponent>() else {
            return;
        };
        let Ok(entity_ref) = world.get_entity(entity) else {
            return;
        };
        let Some(reflected) = rc.reflect(entity_ref) else {
            return;
        };
        match reflected.reflect_clone() {
            Ok(v) => v,
            Err(_) => return,
        }
    };

    let changed = bevy_inspector_egui::bevy_inspector::ui_for_value(value.as_mut(), ui, world);

    if changed {
        let registry = world.resource::<AppTypeRegistry>().clone();
        let registry = registry.read();
        if let Some(reg) = registry.get_with_type_path(type_path)
            && let Some(rc) = reg.data::<bevy::ecs::reflect::ReflectComponent>()
            && let Ok(mut entity_mut) = world.get_entity_mut(entity)
        {
            rc.apply(&mut entity_mut, value.as_partial_reflect());
        }
    }
}

/// Whether `entity` has a component whose registered type path equals `type_path`.
fn entity_has_type_path(world: &World, entity: Entity, type_path: &str) -> bool {
    let registry = world.resource::<AppTypeRegistry>().read();
    let Some(registration) = registry.get_with_type_path(type_path) else {
        return false;
    };
    let Some(reflect_component) = registration.data::<bevy::ecs::reflect::ReflectComponent>()
    else {
        return false;
    };
    let Ok(entity_ref) = world.get_entity(entity) else {
        return false;
    };
    reflect_component.contains(entity_ref)
}

/// Last path segment of a type path (`a::b::Foo` -> `Foo`).
fn short_name(type_path: &str) -> &str {
    type_path.rsplit("::").next().unwrap_or(type_path)
}

/// Editable Name field for `entity`. Edits the existing [`Name`], or inserts one on
/// first edit if the entity has none. Mirrors the Scene panel's inline rename.
fn name_field_ui(world: &mut World, entity: Entity, ui: &mut egui::Ui) {
    let mut name = world
        .get::<Name>(entity)
        .map(|n| n.as_str().to_string())
        .unwrap_or_default();
    ui.horizontal(|ui| {
        ui.label("Name");
        let resp = ui.add(
            egui::TextEdit::singleline(&mut name)
                .hint_text("(unnamed)")
                .desired_width(f32::INFINITY),
        );
        if resp.changed()
            && let Ok(mut e) = world.get_entity_mut(entity)
        {
            e.insert(Name::new(name.clone()));
        }
    });
}
