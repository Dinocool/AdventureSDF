//! 2D shape tessellation for the gizmo overlay. Projects world-space handle
//! geometry to screen space and tessellates filled, antialiased triangles via
//! egui's `epaint`, producing a [`GizmoMesh`] (NDC + linear-RGBA + indices).
//!
//! Ported from `transform-gizmo`'s `shape.rs` + `math::world_to_screen`, using
//! Bevy's `glam` (`Mat4`/`Vec3`) for the projection math.

use bevy::prelude::*;
use ecolor::Color32;
use emath::Pos2;
use epaint::{Mesh as EMesh, Shape, Stroke, TessellationOptions, Tessellator};

use super::GizmoMesh;

const STEPS_PER_RAD: f32 = 20.0;

/// Tessellates world-space gizmo geometry into screen-space [`GizmoMesh`]es.
pub struct ShapeBuilder {
    view_proj: Mat4,
    /// Full window size in physical-independent (logical) pixels.
    window: Vec2,
    ppp: f32,
}

impl ShapeBuilder {
    pub fn new(view_proj: Mat4, window: Vec2, pixels_per_point: f32) -> Self {
        Self {
            view_proj,
            window,
            ppp: pixels_per_point,
        }
    }

    /// Project a world point to screen pixels (origin top-left). `None` if behind
    /// the camera. Mirrors transform-gizmo's `world_to_screen`.
    fn world_to_screen(&self, p: Vec3) -> Option<Pos2> {
        let mut c = self.view_proj * p.extend(1.0);
        // Cull points at/behind the camera. A tiny positive `w` projects to huge
        // screen coords; a small near threshold avoids the stretched-line artifact.
        if c.w < 1e-3 {
            return None;
        }
        c /= c.w;
        let x = self.window.x * 0.5 + c.x * self.window.x * 0.5;
        // NDC y is up; screen y is down — flip here (the shader flips again).
        let y = self.window.y * 0.5 - c.y * self.window.y * 0.5;
        Some(Pos2::new(x, y))
    }

    /// Tessellate an `epaint` shape and convert it to a [`GizmoMesh`] (px → NDC,
    /// `Color32` → linear RGBA).
    fn finish(&self, shape: Shape) -> GizmoMesh {
        let mut tess = Tessellator::new(
            self.ppp,
            TessellationOptions {
                feathering: true,
                ..Default::default()
            },
            [0, 0],
            Vec::new(),
        );
        let mut emesh = EMesh::default();
        tess.tessellate_shape(shape, &mut emesh);

        let mut out = GizmoMesh {
            vertices: Vec::with_capacity(emesh.vertices.len()),
            colors: Vec::with_capacity(emesh.vertices.len()),
            indices: emesh.indices,
        };
        for v in &emesh.vertices {
            out.vertices.push([
                v.pos.x / self.window.x * 2.0 - 1.0,
                v.pos.y / self.window.y * 2.0 - 1.0,
            ]);
            out.colors.push(color32_to_linear(v.color));
        }
        out
    }

    /// A filled arrow: a quad shaft plus a triangle head (matches the plugin's
    /// `arrow`, which is a single convex polygon from base to tip).
    pub fn arrow(&self, from: Vec3, to: Vec3, width: f32, color: Color) -> GizmoMesh {
        // Default proportions: shaft half-width = width/2, head base half-width = width
        // (twice the shaft, the flare), head length ~5x the shaft half-width.
        let shaft_half = px_width(width, self.ppp) * 0.5;
        self.arrow_ex(from, to, shaft_half, shaft_half * 2.0, shaft_half * 5.0, color)
    }

    /// Arrow with every dimension specified independently in **screen pixels**:
    /// `shaft_half` (shaft half-width), `head_half` (head base half-width), `head_len`
    /// (head length). Lets a caller outset the whole arrow uniformly (a hover glow) by
    /// adding the same margin to each — `arrow`'s single `width` couples shaft and head,
    /// so widening it flares the head twice as much as the shaft (distortion).
    pub fn arrow_ex(
        &self,
        from: Vec3,
        to: Vec3,
        shaft_half: f32,
        head_half: f32,
        head_len: f32,
        color: Color,
    ) -> GizmoMesh {
        let (Some(start), Some(end)) = (self.world_to_screen(from), self.world_to_screen(to))
        else {
            return GizmoMesh::default();
        };
        let dir = end - start;
        let len = dir.length();
        if len < 1e-3 {
            return GizmoMesh::default();
        }
        let n = dir / len;
        let perp = Pos2::new(-n.y, n.x); // rot90
        let head = head_len.min(len);
        let shoulder = Pos2::new(end.x - n.x * head, end.y - n.y * head);

        let c = to_c32(color);
        let mut mesh = self.finish(Shape::convex_polygon(
            vec![
                offset(start, perp, shaft_half),
                offset(start, perp, -shaft_half),
                offset(shoulder, perp, -shaft_half),
                offset(shoulder, perp, shaft_half),
            ],
            c,
            Stroke::NONE,
        ));
        // arrow head triangle
        mesh.append(&self.finish(Shape::convex_polygon(
            vec![
                offset(shoulder, perp, head_half),
                offset(shoulder, perp, -head_half),
                end,
            ],
            c,
            Stroke::NONE,
        )));
        mesh
    }

    /// A stroked line segment of the given pixel width.
    pub fn line(&self, from: Vec3, to: Vec3, width: f32, color: Color) -> GizmoMesh {
        let (Some(a), Some(b)) = (self.world_to_screen(from), self.world_to_screen(to)) else {
            return GizmoMesh::default();
        };
        self.finish(Shape::line(
            vec![a, b],
            Stroke::new(px_width(width, self.ppp), to_c32(color)),
        ))
    }

    /// A stroked ring (closed circle outline) in the plane through `center` with the
    /// given `normal` and `radius` (world units), stroked at `width` pixels.
    pub fn ring(
        &self,
        center: Vec3,
        normal: Vec3,
        radius: f32,
        width: f32,
        color: Color,
    ) -> GizmoMesh {
        let pts = self.arc_world_points(center, normal, radius, std::f32::consts::TAU);
        let stroke = Stroke::new(px_width(width, self.ppp), to_c32(color));

        // Project, marking points that are behind/grazing the camera as gaps. Drawing
        // a single closed_line across a gap produces long stretched lines, so instead
        // we emit each CONTIGUOUS visible run as an open polyline.
        let projected: Vec<Option<Pos2>> = pts.iter().map(|p| self.world_to_screen(*p)).collect();
        let all_visible = projected.iter().all(|p| p.is_some());
        if all_visible {
            // Fast path: full ring → one closed line (drop the duplicate last point).
            let mut screen: Vec<Pos2> = projected.into_iter().flatten().collect();
            screen.pop();
            if screen.len() < 3 {
                return GizmoMesh::default();
            }
            return self.finish(Shape::closed_line(screen, stroke));
        }

        let mut mesh = GizmoMesh::default();
        let mut run: Vec<Pos2> = Vec::new();
        let flush = |run: &mut Vec<Pos2>, mesh: &mut GizmoMesh, sb: &Self| {
            if run.len() >= 2 {
                mesh.append(&sb.finish(Shape::line(std::mem::take(run), stroke)));
            } else {
                run.clear();
            }
        };
        for p in projected {
            match p {
                Some(pos) => run.push(pos),
                None => flush(&mut run, &mut mesh, self),
            }
        }
        flush(&mut run, &mut mesh, self);
        mesh
    }

    /// A filled disc in the plane through `center` with `normal` and `radius`.
    pub fn filled_circle(
        &self,
        center: Vec3,
        normal: Vec3,
        radius: f32,
        color: Color,
    ) -> GizmoMesh {
        let pts = self.arc_world_points(center, normal, radius, std::f32::consts::TAU);
        let screen: Vec<Pos2> = pts
            .iter()
            .filter_map(|p| self.world_to_screen(*p))
            .collect();
        if screen.len() < 3 {
            return GizmoMesh::default();
        }
        self.finish(Shape::convex_polygon(screen, to_c32(color), Stroke::NONE))
    }

    /// A filled quad from four world-space corners (e.g. a planar-scale square).
    pub fn quad(&self, corners: [Vec3; 4], color: Color) -> GizmoMesh {
        let screen: Vec<Pos2> = corners
            .iter()
            .filter_map(|p| self.world_to_screen(*p))
            .collect();
        if screen.len() < 3 {
            return GizmoMesh::default();
        }
        self.finish(Shape::convex_polygon(screen, to_c32(color), Stroke::NONE))
    }

    /// A small filled box centred at `center`, oriented by `basis`, half-extent
    /// `half` (world units) — drawn as a screen-space filled square (the scale-axis
    /// handle cap). Cheap: project the centre + an in-plane offset.
    pub fn box_(&self, center: Vec3, basis: [Vec3; 3], half: f32, color: Color) -> GizmoMesh {
        let (u, v) = (basis[0] * half, basis[1] * half);
        self.quad(
            [
                center - u - v,
                center + u - v,
                center + u + v,
                center - u + v,
            ],
            color,
        )
    }

    /// A filled circular sector (pie slice) in the plane through `center` with
    /// `normal`, spanning `[start_angle, start_angle + sweep]` (radians) at `radius`.
    /// Used to visualise the swept angle during a rotation drag.
    pub fn sector(
        &self,
        center: Vec3,
        normal: Vec3,
        radius: f32,
        start_angle: f32,
        sweep: f32,
        color: Color,
    ) -> GizmoMesh {
        if sweep.abs() < 1e-4 {
            return GizmoMesh::default();
        }
        let steps = ((STEPS_PER_RAD * sweep.abs()).ceil() as usize).max(2);
        let (t, b) = plane_basis(normal.normalize_or_zero());
        // Centre + arc rim points → triangle fan.
        let mut pts = Vec::with_capacity(steps + 2);
        pts.push(center);
        for i in 0..=steps {
            let a = start_angle + sweep * i as f32 / steps as f32;
            pts.push(center + (t * a.cos() + b * a.sin()) * radius);
        }
        let screen: Vec<Pos2> = pts
            .iter()
            .filter_map(|p| self.world_to_screen(*p))
            .collect();
        if screen.len() < 3 {
            return GizmoMesh::default();
        }
        self.finish(Shape::convex_polygon(screen, to_c32(color), Stroke::NONE))
    }

    /// Sample `count` world points around a circle (center, normal, radius).
    fn arc_world_points(&self, center: Vec3, normal: Vec3, radius: f32, angle: f32) -> Vec<Vec3> {
        let steps = ((STEPS_PER_RAD * angle.abs()).ceil() as usize).max(8);
        let (tangent, bitangent) = plane_basis(normal.normalize_or_zero());
        (0..=steps)
            .map(|i| {
                let a = angle * i as f32 / steps as f32;
                center + (tangent * a.cos() + bitangent * a.sin()) * radius
            })
            .collect()
    }
}

/// Two orthonormal in-plane basis vectors for a plane normal.
fn plane_basis(n: Vec3) -> (Vec3, Vec3) {
    let a = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let t = a.cross(n).normalize();
    (t, n.cross(t).normalize())
}

/// Convert a logical stroke/handle width to the pixel width epaint expects.
fn px_width(width: f32, ppp: f32) -> f32 {
    width * ppp
}

fn offset(p: Pos2, dir: Pos2, amount: f32) -> Pos2 {
    Pos2::new(p.x + dir.x * amount, p.y + dir.y * amount)
}

/// Bevy `Color` → epaint `Color32` (sRGB bytes, unmultiplied alpha).
fn to_c32(color: Color) -> Color32 {
    let s = color.to_srgba();
    Color32::from_rgba_unmultiplied(
        (s.red * 255.0) as u8,
        (s.green * 255.0) as u8,
        (s.blue * 255.0) as u8,
        (s.alpha * 255.0) as u8,
    )
}

/// epaint `Color32` (sRGB, premultiplied after tessellation) → linear-RGBA f32.
fn color32_to_linear(c: Color32) -> [f32; 4] {
    let lin = Color::srgba_u8(c.r(), c.g(), c.b(), c.a()).to_linear();
    [lin.red, lin.green, lin.blue, lin.alpha]
}
