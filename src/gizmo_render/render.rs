//! Render-world pipeline for the gizmo overlay: builds GPU buffers from the
//! extracted [`GizmoDraw`], queues a `Transparent3d` item on each [`GizmoCamera`]
//! view, and draws the indexed triangle mesh with a flat 2D shader. Ported from
//! `transform-gizmo-bevy`'s render layer, simplified to a single resource-held mesh
//! (no per-gizmo `RenderAsset`).

use bevy::asset::{load_internal_asset, uuid_handle};
use bevy::core_pipeline::core_3d::{CORE_3D_DEPTH_FORMAT, Transparent3d, TransparentSortingInfo3d};
use bevy::core_pipeline::prepass::{
    DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass,
};
use bevy::ecs::query::ROQueryItem;
use bevy::ecs::system::SystemParamItem;
use bevy::ecs::system::lifetimeless::SRes;
use bevy::mesh::{PrimitiveTopology, VertexBufferLayout};
use bevy::pbr::{MeshPipeline, MeshPipelineKey, SetMeshViewBindGroup};
use bevy::prelude::*;
use bevy::render::render_phase::{
    AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
    RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
};
use bevy::render::render_resource::{
    BlendState, Buffer, BufferInitDescriptor, BufferUsages, ColorTargetState, ColorWrites,
    CompareFunction, DepthBiasState, DepthStencilState, FragmentState, IndexFormat,
    MultisampleState, PipelineCache, PrimitiveState, RenderPipelineDescriptor,
    SpecializedRenderPipeline, SpecializedRenderPipelines, StencilState, TextureFormat,
    VertexAttribute, VertexFormat, VertexState, VertexStepMode,
};
use bevy::render::renderer::RenderDevice;
use bevy::render::sync_world::MainEntity;
use bevy::render::view::ExtractedView;
use bevy::render::{Render, RenderApp, RenderSystems};

use super::{GizmoCamera, GizmoDraw};

const GIZMO_SHADER_HANDLE: Handle<Shader> = uuid_handle!("6f1d3c2a-9b84-4e57-a0d2-7c1155ab3e90");

/// Wire the overlay into the render sub-app.
pub(crate) fn build_render(app: &mut App) {
    load_internal_asset!(app, GIZMO_SHADER_HANDLE, "shader.wgsl", Shader::from_wgsl);

    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .add_render_command::<Transparent3d, DrawGizmo>()
        .init_resource::<SpecializedRenderPipelines<GizmoPipeline>>()
        .init_resource::<GpuGizmoBuffers>()
        .add_systems(
            Render,
            (prepare_gizmo_buffers, queue_gizmos)
                .chain()
                .in_set(RenderSystems::Queue),
        );
}

/// Init the pipeline after the render app's `MeshPipeline` exists.
pub(crate) fn finish_render(app: &mut App) {
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.init_resource::<GizmoPipeline>();
    }
}

/// GPU buffers for the current frame's overlay mesh. Rebuilt each frame from the
/// extracted [`GizmoDraw`]; `index_count == 0` means nothing to draw.
#[derive(Resource, Default)]
struct GpuGizmoBuffers {
    buffers: Option<GizmoBuffers>,
}

struct GizmoBuffers {
    position: Buffer,
    color: Buffer,
    index: Buffer,
    index_count: u32,
}

fn prepare_gizmo_buffers(
    draw: Option<Res<GizmoDraw>>,
    device: Res<RenderDevice>,
    mut gpu: ResMut<GpuGizmoBuffers>,
) {
    let Some(draw) = draw else {
        gpu.buffers = None;
        return;
    };
    let mesh = &draw.0;
    if mesh.indices.is_empty() {
        gpu.buffers = None;
        return;
    }

    let position = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("gizmo_overlay_position"),
        usage: BufferUsages::VERTEX,
        contents: bytemuck::cast_slice(&mesh.vertices),
    });
    let color = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("gizmo_overlay_color"),
        usage: BufferUsages::VERTEX,
        contents: bytemuck::cast_slice(&mesh.colors),
    });
    let index = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("gizmo_overlay_index"),
        usage: BufferUsages::INDEX,
        contents: bytemuck::cast_slice(&mesh.indices),
    });
    gpu.buffers = Some(GizmoBuffers {
        position,
        color,
        index,
        index_count: mesh.indices.len() as u32,
    });
}

#[derive(Resource)]
struct GizmoPipeline {
    mesh_pipeline: MeshPipeline,
}

impl FromWorld for GizmoPipeline {
    fn from_world(world: &mut World) -> Self {
        Self {
            mesh_pipeline: world.resource::<MeshPipeline>().clone(),
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
struct GizmoPipelineKey {
    view_key: MeshPipelineKey,
    /// The view's target texture format (HDR `Rgba16Float` or the swapchain format). In Bevy 0.19
    /// HDR is no longer a `MeshPipelineKey` flag — the colour-target format comes straight from
    /// [`ExtractedView::target_format`].
    target_format: TextureFormat,
}

impl SpecializedRenderPipeline for GizmoPipeline {
    type Key = GizmoPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let format = key.target_format;
        let view_layout = self.mesh_pipeline.get_view_layout(key.view_key.into());

        RenderPipelineDescriptor {
            label: Some("gizmo_overlay_pipeline".into()),
            zero_initialize_workgroup_memory: true,
            layout: vec![view_layout.main_layout.clone()],
            immediate_size: 0,
            vertex: VertexState {
                shader: GIZMO_SHADER_HANDLE,
                entry_point: Some("vertex".into()),
                shader_defs: vec![],
                buffers: vec![
                    VertexBufferLayout {
                        array_stride: VertexFormat::Float32x2.size(),
                        step_mode: VertexStepMode::Vertex,
                        attributes: vec![VertexAttribute {
                            format: VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        }],
                    },
                    VertexBufferLayout {
                        array_stride: VertexFormat::Float32x4.size(),
                        step_mode: VertexStepMode::Vertex,
                        attributes: vec![VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 1,
                        }],
                    },
                ],
            },
            fragment: Some(FragmentState {
                shader: GIZMO_SHADER_HANDLE,
                entry_point: Some("fragment".into()),
                shader_defs: vec![],
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..default()
            },
            // Always-pass depth so the overlay sits on top of the 3D scene.
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                // 0.19: `depth_write_enabled` + `depth_compare` are now `Option`.
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::Always),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: key.view_key.msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        }
    }
}

type DrawGizmo = (SetItemPipeline, SetMeshViewBindGroup<0>, DrawGizmoMesh);

struct DrawGizmoMesh;

impl<P: PhaseItem> RenderCommand<P> for DrawGizmoMesh {
    type Param = SRes<GpuGizmoBuffers>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        _view: ROQueryItem<'w, '_, Self::ViewQuery>,
        _entity: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        gpu: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(b) = &gpu.into_inner().buffers else {
            return RenderCommandResult::Success;
        };
        pass.set_index_buffer(b.index.slice(..), IndexFormat::Uint32);
        pass.set_vertex_buffer(0, b.position.slice(..));
        pass.set_vertex_buffer(1, b.color.slice(..));
        pass.draw_indexed(0..b.index_count, 0, 0..1);
        RenderCommandResult::Success
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn queue_gizmos(
    draw_functions: Res<DrawFunctions<Transparent3d>>,
    pipeline: Res<GizmoPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<GizmoPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    gpu: Res<GpuGizmoBuffers>,
    views: Query<
        (
            &ExtractedView,
            Option<&Msaa>,
            (
                Has<NormalPrepass>,
                Has<DepthPrepass>,
                Has<MotionVectorPrepass>,
                Has<DeferredPrepass>,
            ),
        ),
        With<GizmoCamera>,
    >,
    mut phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
) {
    if gpu.buffers.is_none() {
        return;
    }
    let draw_function = draw_functions.read().get_id::<DrawGizmo>().unwrap();

    for (view, msaa, (normal_prepass, depth_prepass, motion_vector_prepass, deferred_prepass)) in
        &views
    {
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        // The view layout must match the camera's actual `mesh_view_bind_group`,
        // which depends on MSAA + which prepasses the camera runs (e.g. DepthPrepass
        // adds binding 20). OR those flags into the key or the bind group is
        // incompatible at draw time.
        let msaa = msaa.copied().unwrap_or_default();
        let mut view_key = MeshPipelineKey::from_msaa_samples(msaa.samples());
        if normal_prepass {
            view_key |= MeshPipelineKey::NORMAL_PREPASS;
        }
        if depth_prepass {
            view_key |= MeshPipelineKey::DEPTH_PREPASS;
        }
        if motion_vector_prepass {
            view_key |= MeshPipelineKey::MOTION_VECTOR_PREPASS;
        }
        if deferred_prepass {
            view_key |= MeshPipelineKey::DEFERRED_PREPASS;
        }
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            GizmoPipelineKey {
                view_key,
                target_format: view.target_format,
            },
        );

        // The overlay is drawn last with an always-pass depth test, so it sits on top of the 3D
        // scene — `AlwaysOnTop` sorts it after every distance-sorted item. `add_transient` (was
        // `add` in 0.18) since this item isn't retained across frames.
        phase.add_transient(Transparent3d {
            sorting_info: TransparentSortingInfo3d::AlwaysOnTop,
            entity: (Entity::PLACEHOLDER, MainEntity::from(Entity::PLACEHOLDER)),
            draw_function,
            pipeline: pipeline_id,
            distance: 0.0,
            batch_range: 0..1,
            extra_index: PhaseItemExtraIndex::None,
            indexed: true,
        });
    }
}
