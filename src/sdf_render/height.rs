//! CPU height-field cache for bake-time SDF displacement.
//!
//! Height-map relief is folded into the baked distance field (not a per-pixel GPU march):
//! at each voxel the bake subtracts `(h - 0.5)·depth` from the `fold_csg` distance, so coarse
//! relief lives in the field itself and shadows/reflections see it for free. This module owns
//! the CPU-side decoded height images and the triplanar sampler the bake calls.
//!
//! Resolution note: the field is voxel-limited (~0.1 world units). Detail finer than a voxel
//! is averaged away by the trilinear atlas — that's the design trade (fine detail stays in the
//! normal map). Keep `depth` ≲ ~1 voxel so the displaced field stays close to a true SDF
//! (IQ's `opDisplace` guidance — a large offset makes the field non-Euclidean and the
//! conservative march can overstep).

use std::sync::Arc;

use bevy::prelude::*;

use super::edits::MaterialRegistry;
use crate::assets::MaterialTextureLibrary;

/// World units per texture tile reciprocal — MUST match `TEXTURE_WORLD_SCALE` in
/// `bindings.wgsl` (0.5 → a tile spans 2 world units) so the baked carve lines up with the
/// triplanar diffuse the shader samples.
const TEXTURE_WORLD_SCALE: f32 = 0.5;

/// One material's decoded height map + its relief depth.
struct MaterialHeight {
    /// Square height image side length (px).
    size: usize,
    /// Row-major grayscale height in `[0,1]`, length `size*size`.
    data: Vec<f32>,
    /// Relief depth in world units (`MaterialDef::parallax_scale`).
    depth: f32,
}

impl MaterialHeight {
    /// Bilinear height sample at tile UV (wrapped to `[0,1)`), matching the GPU's linear
    /// filtering + repeat addressing.
    fn sample(&self, u: f32, v: f32) -> f32 {
        let s = self.size as f32;
        // Wrap into [0,1) then to texel space.
        let uu = (u.fract() + 1.0).fract() * s - 0.5;
        let vv = (v.fract() + 1.0).fract() * s - 0.5;
        let x0 = uu.floor();
        let y0 = vv.floor();
        let fx = uu - x0;
        let fy = vv - y0;
        let wrap = |i: i32| -> usize { (((i % self.size as i32) + self.size as i32) % self.size as i32) as usize };
        let xi0 = wrap(x0 as i32);
        let xi1 = wrap(x0 as i32 + 1);
        let yi0 = wrap(y0 as i32);
        let yi1 = wrap(y0 as i32 + 1);
        let h00 = self.data[yi0 * self.size + xi0];
        let h10 = self.data[yi0 * self.size + xi1];
        let h01 = self.data[yi1 * self.size + xi0];
        let h11 = self.data[yi1 * self.size + xi1];
        let h0 = h00 + (h10 - h00) * fx;
        let h1 = h01 + (h11 - h01) * fx;
        h0 + (h1 - h0) * fy
    }
}

/// Decoded height maps indexed by global material id. Built from the material registry +
/// texture library; threaded into the bake (as `Arc`) so async bake tasks can sample it.
#[derive(Resource, Default)]
pub struct HeightField {
    /// `mats[id]` is `Some` when material `id` has a height map and non-zero relief depth.
    mats: Vec<Option<MaterialHeight>>,
    /// Fingerprint of the (tex_height layer, parallax_scale) columns the cache was built
    /// from, so the bake trigger only fires a rebake when the displacement actually changes
    /// (not on a colour-only registry edit).
    pub fingerprint: u64,
}

impl HeightField {
    /// Signed displacement offset (world units) to SUBTRACT from the envelope distance at
    /// `world_pos` for material `id`: `(h - 0.5)·depth`. Height is TRIPLANAR-blended by the
    /// normal weights — IDENTICAL projection to the shader's `sample_material_map` (same
    /// uv↔world pairings, same `TEXTURE_WORLD_SCALE`, same `pow(|n|,4)` blend) so the baked
    /// carve lines up with the visible diffuse/normal detail on curved faces, not just on
    /// axis-aligned ones. 0 if the material has no relief.
    pub fn displacement(&self, id: u16, world_pos: Vec3, normal: Vec3) -> f32 {
        let Some(Some(mh)) = self.mats.get(id as usize) else {
            return 0.0;
        };
        let p = world_pos * TEXTURE_WORLD_SCALE;
        // Triplanar UV per plane (mirrors sample_material_map's uv_x/uv_y/uv_z).
        let hx = mh.sample(p.z, p.y); // X plane (uv = zy)
        let hy = mh.sample(p.x, p.z); // Y plane (uv = xz)
        let hz = mh.sample(p.x, p.y); // Z plane (uv = xy)
        // Blend weights: pow(|n|,4) normalized — same as `triplanar_weights` in the shader.
        let an = normal.abs();
        let mut w = Vec3::new(an.x.powi(4), an.y.powi(4), an.z.powi(4));
        let sum = w.x + w.y + w.z;
        w /= sum.max(1e-5);
        let h = hx * w.x + hy * w.y + hz * w.z;
        (h - 0.5) * mh.depth
    }

    /// True if any material carries relief — lets the bake skip the per-voxel gradient/sample
    /// work entirely when no height maps are in play.
    pub fn any_relief(&self) -> bool {
        self.mats.iter().any(|m| m.is_some())
    }
}

/// Decode a material's height PNG into a grayscale `[0,1]` buffer. Reuses the same load +
/// resize the BC7 encoder uses (`image::open` → resize → luma) so CPU and GPU see the same
/// pixels. `None` on a missing/unreadable file.
fn load_height_image(slug: &str, dir: &str) -> Option<MaterialHeight> {
    let path = format!("assets/textures/{slug}/{dir}/height.png");
    let img = match image::open(&path) {
        Ok(img) => img,
        Err(_) => return None,
    };
    // Modest resolution: the bake only samples at voxel scale, so a full 1024² is wasted
    // memory per material. 256² preserves the coarse relief the voxel grid can resolve.
    const BAKE_HEIGHT_SIZE: u32 = 256;
    let luma = img
        .resize_exact(
            BAKE_HEIGHT_SIZE,
            BAKE_HEIGHT_SIZE,
            image::imageops::FilterType::Triangle,
        )
        .to_luma8();
    let data: Vec<f32> = luma.iter().map(|&p| p as f32 / 255.0).collect();
    Some(MaterialHeight {
        size: BAKE_HEIGHT_SIZE as usize,
        data,
        depth: 0.0, // filled by the caller (depends on the material's parallax_scale)
    })
}

/// Fingerprint the displacement-relevant columns so we only rebuild + rebake when they change.
fn fingerprint(registry: &MaterialRegistry) -> u64 {
    let mut h: u64 = 1469598103934665603; // FNV-1a offset basis
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(1099511628211);
    };
    for def in &registry.defs {
        mix(def.tex_layers[3] as u64);
        mix(def.parallax_scale.to_bits() as u64);
    }
    h
}

/// Rebuild the height cache from the current registry + texture library. Returns `None` if
/// the fingerprint is unchanged (nothing displacement-relevant changed → no rebuild/rebake).
pub fn build(
    registry: &MaterialRegistry,
    library: &MaterialTextureLibrary,
    prev_fingerprint: u64,
) -> Option<HeightField> {
    let fp = fingerprint(registry);
    if fp == prev_fingerprint {
        return None;
    }
    let mut mats: Vec<Option<MaterialHeight>> = Vec::with_capacity(registry.defs.len());
    for def in &registry.defs {
        let layer = def.tex_layers[3];
        if layer == u32::MAX || def.parallax_scale <= 0.0 {
            mats.push(None);
            continue;
        }
        let variant = library.variants.get(layer as usize);
        let loaded = variant.and_then(|v| load_height_image(&v.slug, &v.dir));
        mats.push(loaded.map(|mut mh| {
            mh.depth = def.parallax_scale;
            mh
        }));
    }
    Some(HeightField {
        mats,
        fingerprint: fp,
    })
}

/// Shared handle the scheduler clones into async bake tasks.
pub type SharedHeightField = Arc<HeightField>;

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a HeightField with one material id carrying a constant-height image.
    fn const_field(id: usize, h: f32, depth: f32) -> HeightField {
        let mut mats: Vec<Option<MaterialHeight>> = (0..=id).map(|_| None).collect();
        mats[id] = Some(MaterialHeight {
            size: 4,
            data: vec![h; 16],
            depth,
        });
        HeightField {
            mats,
            fingerprint: 0,
        }
    }

    #[test]
    fn flat_mid_height_is_no_op() {
        // h = 0.5 everywhere → (h-0.5)·depth = 0, so the baked distance is unchanged.
        let hf = const_field(2, 0.5, 0.2);
        let d = hf.displacement(2, Vec3::new(1.3, 0.4, -2.1), Vec3::Y);
        assert!(d.abs() < 1e-6, "flat mid-height must not displace, got {d}");
    }

    #[test]
    fn peak_and_valley_signs() {
        // h = 1 (peak) → +dep/2 carve OUT (subtracted from dist → surface moves toward camera);
        // h = 0 (valley) → -dep/2.
        let peak = const_field(0, 1.0, 0.2);
        let valley = const_field(0, 0.0, 0.2);
        let p = Vec3::new(0.5, 0.5, 0.5);
        assert!((peak.displacement(0, p, Vec3::Z) - 0.1).abs() < 1e-6);
        assert!((valley.displacement(0, p, Vec3::Z) + 0.1).abs() < 1e-6);
    }

    #[test]
    fn no_relief_for_absent_material() {
        let hf = const_field(1, 1.0, 0.2);
        // id 0 has no height map → zero displacement regardless of position.
        assert_eq!(hf.displacement(0, Vec3::ONE, Vec3::Y), 0.0);
        // out-of-range id → zero (safe).
        assert_eq!(hf.displacement(99, Vec3::ONE, Vec3::Y), 0.0);
    }

    #[test]
    fn triplanar_axis_matches_normal() {
        // A horizontal ramp on an 8x8 image; sampling on the Z plane (uv = x,y) must vary
        // with world x. Use a wide x span (tile = 2 world units at scale 0.5) so the wrapped
        // UVs land on clearly different columns.
        let size = 8usize;
        let mut data = vec![0.0f32; size * size];
        for y in 0..size {
            for x in 0..size {
                data[y * size + x] = x as f32 / (size - 1) as f32; // 0 → 1 across columns
            }
        }
        let mut mats = vec![None];
        mats[0] = Some(MaterialHeight {
            size,
            data,
            depth: 1.0,
        });
        let hf = HeightField {
            mats,
            fingerprint: 0,
        };
        // Z-plane (normal = Z): uv = (x,y)*0.5. world x 0.2 → uv.x 0.1, x 1.8 → uv.x 0.9.
        let a = hf.displacement(0, Vec3::new(0.2, 1.0, 0.0), Vec3::Z);
        let b = hf.displacement(0, Vec3::new(1.8, 1.0, 0.0), Vec3::Z);
        assert!(b > a, "z-plane sample must increase with world x ({a} vs {b})");
    }
}
