//! Paged (bindless) atlas pool — the distance + material brick atlases as a runtime-grown array
//! of fixed-size PAGE textures instead of one tall texture.
//!
//! The old single-texture atlas grew by allocating a taller texture and `copy_texture_to_texture`-ing
//! the whole old atlas in (old + new alive at once ≈ 2× the resident bricks), repeated on every
//! row-boundary crossing during a fill → O(N²) copy + a ~2× VRAM spike (see `atlas_upload`'s former
//! realloc path). Here growth allocates ONE new page and copies NOTHING: existing bricks stay put in
//! their pages. A brick's global tile row splits into `(page = row / PAGE_ROWS, local row)`, matching
//! `sdf::brick::voxel_loc` in the shader (which indexes `binding_array<texture_2d, ATLAS_MAX_PAGES>`).
//!
//! The pages bind as a SIZED `binding_array` (count = [`ATLAS_MAX_PAGES`]); live pages fill the front,
//! a 1×1 dummy fills the rest (so no `PARTIALLY_BOUND` feature is needed). Indexing the array with the
//! per-fragment page needs `SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING` (requested
//! in `main.rs`).

use super::*;
use crate::sdf_render::atlas::{ATLAS_TILES_PER_ROW, BRICK_EDGE};

/// Height of one page texture, in PIXELS. MUST match `sdf::bindings::ATLAS_PAGE_HEIGHT_PX` and be a
/// multiple of `BRICK_EDGE` (8) so a tile never straddles two pages. 2048 px = 256 tile-rows.
pub const ATLAS_PAGE_HEIGHT_PX: u32 = 2048;

/// Compile-time max page count = the shader's `binding_array<texture_2d, ATLAS_MAX_PAGES>` size. The
/// bind group always binds exactly this many views (live pages + dummy fill). 64 pages × 256 tile-rows
/// × `ATLAS_TILES_PER_ROW` ≈ 4.2 M bricks of capacity.
pub const ATLAS_MAX_PAGES: u32 = 64;

/// Tile-rows per page (`ATLAS_PAGE_HEIGHT_PX / BRICK_EDGE`).
pub const PAGE_ROWS: u32 = ATLAS_PAGE_HEIGHT_PX / BRICK_EDGE as u32;

/// Atlas texture width in pixels (`ATLAS_TILES_PER_ROW` tiles × tile_width). One page is this wide ×
/// `ATLAS_PAGE_HEIGHT_PX` tall.
pub fn atlas_width_px() -> u32 {
    let tile_width = (BRICK_EDGE * BRICK_EDGE) as u32; // 64
    ATLAS_TILES_PER_ROW * tile_width
}

/// (page index, y-pixel WITHIN that page) for a global atlas y-pixel. A tile is `BRICK_EDGE` px tall
/// and `ATLAS_PAGE_HEIGHT_PX` is a multiple of it, so a whole tile lives in one page.
pub fn split_row(global_y: u32) -> (u32, u32) {
    (global_y / ATLAS_PAGE_HEIGHT_PX, global_y % ATLAS_PAGE_HEIGHT_PX)
}

/// The two atlas formats grow in lockstep (a brick writes both), so one pool owns both.
pub struct AtlasPages {
    dist: Vec<Texture>,
    dist_views: Vec<TextureView>,
    mat: Vec<Texture>,
    mat_views: Vec<TextureView>,
    /// 1×1 fills for the unused binding-array slots (one per format).
    dummy_dist_view: TextureView,
    dummy_mat_view: TextureView,
}

impl AtlasPages {
    pub fn new(device: &RenderDevice) -> Self {
        let dummy_dist = make_dummy(device, TextureFormat::R16Snorm, "sdf_atlas_page_dummy_dist");
        let dummy_mat = make_dummy(device, TextureFormat::Rgba8Snorm, "sdf_atlas_page_dummy_mat");
        Self {
            dist: Vec::new(),
            dist_views: Vec::new(),
            mat: Vec::new(),
            mat_views: Vec::new(),
            dummy_dist_view: dummy_dist.create_view(&TextureViewDescriptor::default()),
            dummy_mat_view: dummy_mat.create_view(&TextureViewDescriptor::default()),
        }
    }

    pub fn page_count(&self) -> usize {
        self.dist.len()
    }

    /// Ensure at least `rows_needed` tile-rows of capacity exist, allocating whole pages as needed.
    /// NO copy — existing pages are untouched. Returns true if any page was added (⇒ the bind group
    /// must be rebuilt). Panics past [`ATLAS_MAX_PAGES`] (the shader's binding-array ceiling) — a hard
    /// cap the working set should never reach; if it does, raise both constants in lockstep.
    pub fn ensure(&mut self, device: &RenderDevice, rows_needed: u32) -> bool {
        let pages_needed = rows_needed.div_ceil(PAGE_ROWS).max(1);
        assert!(
            pages_needed <= ATLAS_MAX_PAGES,
            "SDF atlas needs {pages_needed} pages > ATLAS_MAX_PAGES ({ATLAS_MAX_PAGES}); raise it + the shader's binding_array size"
        );
        let grew = self.dist.len() < pages_needed as usize;
        while (self.dist.len() as u32) < pages_needed {
            let idx = self.dist.len();
            let dist = make_page(device, TextureFormat::R16Snorm, idx, "dist");
            let mat = make_page(device, TextureFormat::Rgba8Snorm, idx, "mat");
            self.dist_views.push(dist.create_view(&TextureViewDescriptor::default()));
            self.mat_views.push(mat.create_view(&TextureViewDescriptor::default()));
            self.dist.push(dist);
            self.mat.push(mat);
        }
        grew
    }

    /// Live distance page (for the bake node's tile blit).
    pub fn dist_page(&self, page: u32) -> &Texture {
        &self.dist[page as usize]
    }
    pub fn mat_page(&self, page: u32) -> &Texture {
        &self.mat[page as usize]
    }

    /// True before the first page is allocated (no bake has run) — the bake node skips, and the bind
    /// group binds all-dummy.
    pub fn is_empty(&self) -> bool {
        self.dist.is_empty()
    }

    /// `ATLAS_MAX_PAGES` distance views for the binding array: live pages first, dummy-filled to the
    /// fixed count. Returns RAW `wgpu::TextureView` refs — that's what `BindingResource::
    /// TextureViewArray` / `IntoBinding for &[&wgpu::TextureView]` takes (the single-view `IntoBinding`
    /// takes bevy's wrapper, but the array impl takes wgpu's; bevy's `TextureView` derefs to it).
    pub fn dist_refs(&self) -> Vec<&wgpu::TextureView> {
        fill_refs(&self.dist_views, &self.dummy_dist_view)
    }
    pub fn mat_refs(&self) -> Vec<&wgpu::TextureView> {
        fill_refs(&self.mat_views, &self.dummy_mat_view)
    }
}

fn fill_refs<'a>(views: &'a [TextureView], dummy: &'a TextureView) -> Vec<&'a wgpu::TextureView> {
    let mut refs: Vec<&wgpu::TextureView> = Vec::with_capacity(ATLAS_MAX_PAGES as usize);
    refs.extend(views.iter().map(|v| &**v));
    while refs.len() < ATLAS_MAX_PAGES as usize {
        refs.push(&**dummy);
    }
    refs
}

fn make_page(device: &RenderDevice, format: TextureFormat, idx: usize, kind: &str) -> Texture {
    let label = format!("sdf_atlas_{kind}_page{idx}");
    device.create_texture(&TextureDescriptor {
        label: Some(&label),
        size: Extent3d {
            width: atlas_width_px(),
            height: ATLAS_PAGE_HEIGHT_PX,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        // No COPY_SRC: pages are never copied out (the whole point — no grow-copy).
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn make_dummy(device: &RenderDevice, format: TextureFormat, label: &str) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    })
}
