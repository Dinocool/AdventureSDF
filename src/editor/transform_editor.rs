//! Custom inspector editor for `Transform`: position / euler-angle rotation (degrees) /
//! scale, instead of the generic `Vec3 + Quat(xyzw) + Vec3` reflection UI. Edits the
//! node's own `Transform` (for a root this is its world pose; for a child it's
//! parent-relative, matching the move/rotate gizmo).
//!
//! Euler angles are stored in egui temp memory keyed by entity so repeated quat→euler→quat
//! conversions don't drift the rotation (and so two equivalent eulers don't fight). The
//! buffer reseeds from the live `Transform` whenever the selection changes or the rotation
//! was altered elsewhere (gizmo drag).

use bevy::math::EulerRot;
use bevy::prelude::*;
use bevy_egui::egui;

/// Euler order for display/edit. YXZ (yaw, pitch, roll) reads naturally for scene nodes.
const EULER: EulerRot = EulerRot::YXZ;

/// egui temp-memory buffer: the working euler angles (degrees) plus the entity + the quat
/// they were derived from, so we only reseed when the live rotation actually changed.
#[derive(Clone, Copy)]
struct EulerBuf {
    entity: Entity,
    /// Quat the displayed angles were last synced from (detect external changes).
    from: Quat,
    /// Working angles in degrees: (yaw, pitch, roll) for [`EULER`].
    deg: [f32; 3],
}

/// Inspector override for `Transform`. Registered via
/// `inspector::register_component_editor::<Transform>`.
pub fn transform_editor(world: &mut World, entity: Entity, ui: &mut egui::Ui) {
    let Ok(mut entity_mut) = world.get_entity_mut(entity) else {
        return;
    };
    let Some(mut transform) = entity_mut.get_mut::<Transform>() else {
        return;
    };
    let mut t = *transform;

    let buf_id = ui.make_persistent_id(("transform_euler", entity));
    let mut buf: Option<EulerBuf> = ui.memory(|m| m.data.get_temp(buf_id));

    // Reseed the euler buffer from the live rotation when the entity changed or the
    // rotation was edited outside this widget (e.g. the viewport gizmo).
    let needs_reseed = match buf {
        Some(b) => b.entity != entity || b.from != t.rotation,
        None => true,
    };
    if needs_reseed {
        let (y, x, z) = t.rotation.to_euler(EULER);
        buf = Some(EulerBuf {
            entity,
            from: t.rotation,
            deg: [y.to_degrees(), x.to_degrees(), z.to_degrees()],
        });
    }
    let mut b = buf.unwrap();

    let mut changed = false;

    ui.horizontal(|ui| {
        ui.label("Position");
        changed |= ui.add(egui::DragValue::new(&mut t.translation.x).speed(0.05).prefix("x ")).changed();
        changed |= ui.add(egui::DragValue::new(&mut t.translation.y).speed(0.05).prefix("y ")).changed();
        changed |= ui.add(egui::DragValue::new(&mut t.translation.z).speed(0.05).prefix("z ")).changed();
    });

    let mut rot_changed = false;
    ui.horizontal(|ui| {
        ui.label("Rotation°");
        rot_changed |= ui.add(egui::DragValue::new(&mut b.deg[0]).speed(0.5).prefix("y ")).changed();
        rot_changed |= ui.add(egui::DragValue::new(&mut b.deg[1]).speed(0.5).prefix("x ")).changed();
        rot_changed |= ui.add(egui::DragValue::new(&mut b.deg[2]).speed(0.5).prefix("z ")).changed();
    });
    if rot_changed {
        t.rotation = Quat::from_euler(
            EULER,
            b.deg[0].to_radians(),
            b.deg[1].to_radians(),
            b.deg[2].to_radians(),
        );
        // Record the quat we just produced so the reseed check doesn't fire on it.
        b.from = t.rotation;
        changed = true;
    }

    ui.horizontal(|ui| {
        ui.label("Scale");
        changed |= ui.add(egui::DragValue::new(&mut t.scale.x).speed(0.05).prefix("x ")).changed();
        changed |= ui.add(egui::DragValue::new(&mut t.scale.y).speed(0.05).prefix("y ")).changed();
        changed |= ui.add(egui::DragValue::new(&mut t.scale.z).speed(0.05).prefix("z ")).changed();
    });

    if changed {
        *transform = t;
    }
    ui.memory_mut(|m| m.data.insert_temp(buf_id, b));
}
