//! The GPU brick-bake compute path: extract this frame's bake jobs from the scheduler, upload the
//! job headers + flat edit list into storage buffers, dispatch the bake compute shader (one
//! workgroup per brick), and blit each baked tile from the output buffers into the persistent atlas
//! textures. Self-contained except for the shared `SdfGpuAtlas` it writes into (from [`super`]).

use super::super::{bake_scheduler, chunk, edits};
use super::*;

pub(super) const SDF_BAKE_SHADER_PATH: &str = "shaders/sdf_brick_bake.wgsl";

#[derive(Resource)]
pub(super) struct SdfBakeShaderHandle(pub(super) Handle<Shader>);

/// Width of the 2D bake dispatch grid in workgroups. The compute dispatch uses one workgroup
/// per brick job; a single dimension caps at 65535 (wgpu/Vulkan limit), which a large edit can
/// blow past (a big sphere dirties 70k+ bricks). So we lay the jobs out in a 2D grid of this
/// width and reconstruct the linear job index in the shader as `wg.y * DISPATCH_WIDTH + wg.x`.
/// 256² = 65536 jobs per "page"; the Y extent then carries the rest, well under the limit.
/// Must match `DISPATCH_WIDTH` in sdf_brick_bake.wgsl.
const BAKE_DISPATCH_WIDTH: u32 = 256;

/// u32s per distance tile in the bake output buffer. Each tile is 64×8 R16 texels = 512
/// texels = 256 u32 (two R16 packed per u32), but rows are padded to 64 u32 so each tile row
/// is 256 bytes — `copy_buffer_to_texture` requires `bytes_per_row` to be a multiple of 256.
/// 64 u32/row × 8 rows = 512 u32 per tile (32 real + 32 pad per row). Must match the bake
/// shader's `DIST_ROW_U32`/`DIST_TILE_U32`.
const BAKE_DIST_ROW_U32: u32 = 64;
const BAKE_DIST_TILE_U32: u32 = BAKE_DIST_ROW_U32 * 8;
/// u32s per material tile: 64×8 Rgba16 texels, 2 u32 per texel, 128 u32/row × 8 = 1024.
/// Row stride = 128 u32 = 512 bytes (already a multiple of 256). Matches `MAT_TILE_U32`.
const BAKE_MAT_ROW_U32: u32 = 128;
const BAKE_MAT_TILE_U32: u32 = BAKE_MAT_ROW_U32 * 8;

/// One brick bake job's header, std430. Mirror of the WGSL `JobHeader` in
/// `sdf_brick_bake.wgsl` and built from `bake_scheduler::GpuBakeJob`.
#[derive(ShaderType, Clone, Copy, Default)]
struct GpuJobHeader {
    coord: IVec3,
    voxel_size: f32,
    dist_band: f32,
    edit_start: u32,
    edit_count: u32,
    pal01: u32,
    pal23: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Render-world copy of this frame's GPU bake jobs (extracted from
/// `bake_scheduler::PendingGpuBakes`). `tiles` parallels `headers`: job i writes the atlas
/// tile `tiles[i]`. Empty on frames with no bake work (the node early-outs).
#[derive(Resource, Default)]
pub(super) struct ExtractedBrickBakes {
    headers: Vec<GpuJobHeader>,
    edits: Vec<edits::GpuEdit>,
    /// Destination atlas tile index per job (drives the `copy_buffer_to_texture` origin).
    tiles: Vec<u32>,
}

/// The bake compute pipeline + the storage buffers the dispatch writes (sized to the job
/// count each frame). The buffers are re-created when the job count grows; the per-tile
/// `copy_buffer_to_texture` into the persistent atlas happens in the bake node.
#[derive(Resource)]
struct SdfBakePipeline {
    pipeline_id: CachedComputePipelineId,
    layout: BindGroupLayoutDescriptor,
}

#[derive(Resource, Default)]
pub(super) struct SdfBakeBuffers {
    header_buffer: Option<Buffer>,
    edit_buffer: Option<Buffer>,
    dist_buffer: Option<Buffer>,
    mat_buffer: Option<Buffer>,
    /// Number of jobs prepared this frame (workgroup dispatch count + copy loop bound).
    job_count: u32,
    /// Destination atlas tiles for this frame's jobs (parallels the dispatch order).
    tiles: Vec<u32>,
}

/// Extract this frame's GPU bake jobs from the main world into the render world, converting
/// each `GpuBakeJob` into its `GpuJobHeader`. The flat `GpuEdit` list is shared by all jobs
/// (each job's `edit_start..edit_start+edit_count` indexes it). Empty when not in GPU mode.
pub(super) fn extract_brick_bakes(
    pending: Extract<Res<bake_scheduler::PendingGpuBakes>>,
    mut commands: Commands,
) {
    if pending.jobs.is_empty() {
        commands.insert_resource(ExtractedBrickBakes::default());
        return;
    }
    let mut headers = Vec::with_capacity(pending.jobs.len());
    let mut tiles = Vec::with_capacity(pending.jobs.len());
    for j in &pending.jobs {
        headers.push(GpuJobHeader {
            coord: j.coord,
            voxel_size: j.voxel_size,
            dist_band: j.dist_band,
            edit_start: j.edit_start,
            edit_count: j.edit_count,
            pal01: j.palette[0] as u32 | ((j.palette[1] as u32) << 16),
            pal23: j.palette[2] as u32 | ((j.palette[3] as u32) << 16),
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        });
        tiles.push(j.tile);
    }
    commands.insert_resource(ExtractedBrickBakes {
        headers,
        edits: pending.edits.clone(),
        tiles,
    });
}

/// Upload this frame's bake job headers + edits into storage buffers and (re)size the
/// dist/mat output buffers to the job count. The actual dispatch + per-tile copy into the
/// atlas happens in `SdfBrickBakeNode`. Runs before `prepare_sdf_atlas_gpu` so a realloc that
/// recreates the atlas texture this frame is followed by our bake filling it.
pub(super) fn prepare_brick_bake_buffers(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    extracted: Option<Res<ExtractedBrickBakes>>,
    mut buffers: ResMut<SdfBakeBuffers>,
) {
    let Some(extracted) = extracted else { return };
    let n = extracted.headers.len() as u32;
    buffers.job_count = n;
    buffers.tiles = extracted.tiles.clone();
    if n == 0 {
        return;
    }
    let _span = info_span!("sdf_prepare_bake_buffers", jobs = n).entered();

    // Headers (std430, GpuJobHeader = 48 bytes).
    let mut header_bytes: Vec<u8> = Vec::with_capacity(extracted.headers.len() * 48);
    for h in &extracted.headers {
        header_bytes.extend_from_slice(&h.coord.x.to_le_bytes());
        header_bytes.extend_from_slice(&h.coord.y.to_le_bytes());
        header_bytes.extend_from_slice(&h.coord.z.to_le_bytes());
        header_bytes.extend_from_slice(&h.voxel_size.to_le_bytes());
        header_bytes.extend_from_slice(&h.dist_band.to_le_bytes());
        header_bytes.extend_from_slice(&h.edit_start.to_le_bytes());
        header_bytes.extend_from_slice(&h.edit_count.to_le_bytes());
        header_bytes.extend_from_slice(&h.pal01.to_le_bytes());
        header_bytes.extend_from_slice(&h.pal23.to_le_bytes());
        header_bytes.extend_from_slice(&0u32.to_le_bytes());
        header_bytes.extend_from_slice(&0u32.to_le_bytes());
        header_bytes.extend_from_slice(&0u32.to_le_bytes());
    }
    buffers.header_buffer = Some(device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_bake_headers"),
        contents: &header_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    }));

    // Edits (std430, GpuEdit = 96 bytes: mat4 + 2×vec4 + 4×u32). Always ≥1 row so the
    // storage binding is never zero-sized.
    let mut edit_bytes: Vec<u8> = Vec::with_capacity(extracted.edits.len().max(1) * 96);
    for e in &extracted.edits {
        for col in e.inv_model.to_cols_array() {
            edit_bytes.extend_from_slice(&col.to_le_bytes());
        }
        for v in [e.params.x, e.params.y, e.params.z, e.params.w] {
            edit_bytes.extend_from_slice(&v.to_le_bytes());
        }
        for v in [e.params2.x, e.params2.y, e.params2.z, e.params2.w] {
            edit_bytes.extend_from_slice(&v.to_le_bytes());
        }
        edit_bytes.extend_from_slice(&e.tag.to_le_bytes());
        edit_bytes.extend_from_slice(&e.op_kind.to_le_bytes());
        edit_bytes.extend_from_slice(&e.smoothing.to_le_bytes());
        edit_bytes.extend_from_slice(&e.material_id.to_le_bytes());
    }
    if edit_bytes.is_empty() {
        edit_bytes.resize(96, 0);
    }
    buffers.edit_buffer = Some(device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf_bake_edits"),
        contents: &edit_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    }));

    // Output buffers (STORAGE write target + COPY_SRC for the per-tile blit into the atlas).
    let dist_size = (n * BAKE_DIST_TILE_U32) as u64 * 4;
    let mat_size = (n * BAKE_MAT_TILE_U32) as u64 * 4;
    let needs_dist = buffers.dist_buffer.as_ref().is_none_or(|b| b.size() < dist_size);
    let needs_mat = buffers.mat_buffer.as_ref().is_none_or(|b| b.size() < mat_size);
    if needs_dist {
        buffers.dist_buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("sdf_bake_dist_out"),
            size: dist_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }
    if needs_mat {
        buffers.mat_buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("sdf_bake_mat_out"),
            size: mat_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }
    let _ = &queue; // (kept for parity with sibling prepare systems; no immediate write here)
}

/// Queue the brick-bake compute pipeline. Standalone bind group (no camera/atlas-read): two
/// read-only storage buffers (headers, edits) + two read-write storage buffers (dist, mat
/// output). Runs at `RenderStartup` after `init_sdf_pipeline` (no dependency, just ordering).
pub(super) fn init_bake_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    bake_shader: Res<SdfBakeShaderHandle>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "sdf_bake_bind_group",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only::<GpuJobHeader>(false),
                storage_buffer_read_only::<edits::GpuEdit>(false),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    );
    let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("sdf_bake_pipeline".into()),
        layout: vec![layout.clone()],
        shader: bake_shader.0.clone(),
        ..default()
    });
    commands.insert_resource(SdfBakePipeline {
        pipeline_id,
        layout,
    });
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(super) struct SdfBrickBakeLabel;

#[derive(Default)]
pub(super) struct SdfBrickBakeNode;

impl Node for SdfBrickBakeNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let buffers = world.resource::<SdfBakeBuffers>();
        if buffers.job_count == 0 {
            return Ok(());
        }
        let _span = info_span!("sdf_brick_bake_node", jobs = buffers.job_count).entered();
        let bake = world.resource::<SdfBakePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(bake.pipeline_id) else {
            return Ok(());
        };
        let (Some(header_buf), Some(edit_buf), Some(dist_buf), Some(mat_buf)) = (
            buffers.header_buffer.as_ref(),
            buffers.edit_buffer.as_ref(),
            buffers.dist_buffer.as_ref(),
            buffers.mat_buffer.as_ref(),
        ) else {
            return Ok(());
        };
        // The atlas textures must already exist (a prior bake/realloc created them). If not,
        // there's nothing to copy into yet — skip this frame.
        let gpu_atlas = world.resource::<SdfGpuAtlas>();
        let (Some(dist_tex), Some(mat_tex)) = (&gpu_atlas.dist_tex, &gpu_atlas.mat_tex) else {
            return Ok(());
        };

        let device = render_context.render_device();
        let layout = pipeline_cache.get_bind_group_layout(&bake.layout);
        let bind_group = device.create_bind_group(
            "sdf_bake_bind_group",
            &layout,
            &BindGroupEntries::sequential((
                header_buf.as_entire_buffer_binding(),
                edit_buf.as_entire_buffer_binding(),
                dist_buf.as_entire_buffer_binding(),
                mat_buf.as_entire_buffer_binding(),
            )),
        );

        {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("sdf_brick_bake"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per brick job, laid out in a 2D grid so the count can exceed the
            // 65535 single-dimension dispatch limit (a large edit dirties 70k+ bricks). The
            // shader reconstructs the linear job index from (wg.x, wg.y).
            let wg_x = buffers.job_count.min(BAKE_DISPATCH_WIDTH);
            let wg_y = buffers.job_count.div_ceil(BAKE_DISPATCH_WIDTH);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Blit each job's tile from the output buffers into the persistent atlas textures.
        // The buffer layout matches the texture sub-rect (dist rows padded to 256 bytes,
        // mat rows already 512). `copy_buffer_to_texture` requires bytes_per_row % 256 == 0.
        let edge = BRICK_EDGE as u32;
        let tile_width = edge * edge; // 64
        let encoder = render_context.command_encoder();
        for (i, &tile) in buffers.tiles.iter().enumerate() {
            // Same packing as the lookup rows (single source: `chunk::tile_atlas_base`); unpack the
            // `col_px | row_px<<16` it returns into the sub-rect origin for the texture blit.
            let base = chunk::tile_atlas_base(tile);
            let (col_px, row_px) = (base & 0xFFFF, base >> 16);
            let tile_extent = Extent3d {
                width: tile_width,
                height: edge,
                depth_or_array_layers: 1,
            };
            let dist_offset = (i as u32 * BAKE_DIST_TILE_U32) as u64 * 4;
            encoder.copy_buffer_to_texture(
                TexelCopyBufferInfo {
                    buffer: dist_buf,
                    layout: TexelCopyBufferLayout {
                        offset: dist_offset,
                        bytes_per_row: Some(BAKE_DIST_ROW_U32 * 4), // 256 bytes
                        rows_per_image: Some(edge),
                    },
                },
                TexelCopyTextureInfo {
                    texture: dist_tex,
                    mip_level: 0,
                    origin: Origin3d {
                        x: col_px,
                        y: row_px,
                        z: 0,
                    },
                    aspect: TextureAspect::All,
                },
                tile_extent,
            );
            let mat_offset = (i as u32 * BAKE_MAT_TILE_U32) as u64 * 4;
            encoder.copy_buffer_to_texture(
                TexelCopyBufferInfo {
                    buffer: mat_buf,
                    layout: TexelCopyBufferLayout {
                        offset: mat_offset,
                        bytes_per_row: Some(BAKE_MAT_ROW_U32 * 4), // 512 bytes
                        rows_per_image: Some(edge),
                    },
                },
                TexelCopyTextureInfo {
                    texture: mat_tex,
                    mip_level: 0,
                    origin: Origin3d {
                        x: col_px,
                        y: row_px,
                        z: 0,
                    },
                    aspect: TextureAspect::All,
                },
                tile_extent,
            );
        }

        Ok(())
    }
}
