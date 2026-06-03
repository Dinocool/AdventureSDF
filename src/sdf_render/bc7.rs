//! Reusable BC7 texture-array compression with a content-hashed disk cache.
//!
//! Generic over content: give it raw RGBA8 layers + dimensions and it produces a
//! BC7 block stream with a full mip chain (layer-major: `layer0[mip0..mipN],
//! layer1[...], ...`), ready for `create_texture_with_data` with a `Bc7Rgba*`
//! format. The result is cached to disk keyed by a hash of the *source* bytes plus
//! an encoder-version byte, so it re-encodes automatically when the inputs change
//! and is otherwise a cheap `fs::read`.
//!
//! Not SDF-specific — any texture pipeline can reuse `encode_layers_bc7` and
//! [`Bc7Cache`].

use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use bevy::prelude::*;

/// Bump when the encoder settings or block/mip layout change, to invalidate all
/// caches. v2: mip chain now stops at the 4×4 block minimum (9 levels at 1024², not
/// 11) — v1 blobs claim 11 mips and would over-run the texture's mip count.
const ENCODER_VERSION: u32 = 2;

/// Magic + version header prefixing every cached blob, so a stale/foreign/short
/// file is rejected rather than mis-decoded.
const CACHE_MAGIC: u32 = 0x42433701; // "BC7\x01"

/// A BC7-encoded texture array ready for GPU upload.
pub struct Bc7Array {
    /// BC7 blocks, layer-major with a full mip chain per layer.
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub mip_levels: u32,
}

/// Number of mip levels for a square BC7 texture of `size`, stopping at the 4×4
/// block minimum. BC7 copies require block-aligned dimensions, so levels below 4×4
/// (2×2, 1×1) are NOT generated — `Queue::write_texture` rejects a sub-block copy
/// width. For 1024² this yields 9 levels (1024…4), not 11.
pub fn mip_count(size: u32) -> u32 {
    // log2(size)+1 levels down to 1×1, minus the 2×2 and 1×1 tail.
    (32 - size.max(4).leading_zeros()).saturating_sub(2)
}

/// BC7 byte size of one `w×h` mip level (4×4 blocks, 16 bytes each).
fn bc7_level_bytes(w: u32, h: u32) -> usize {
    (w.div_ceil(4) * h.div_ceil(4) * 16) as usize
}

/// Total BC7 bytes for one layer's mip chain at square `size` (stops at 4×4).
pub(crate) fn bc7_layer_bytes(size: u32) -> usize {
    let mut total = 0;
    let mut s = size;
    for _ in 0..mip_count(size) {
        total += bc7_level_bytes(s, s);
        s = (s / 2).max(4);
    }
    total
}

/// Box-downsample an RGBA8 image to half size (each axis), clamping at 1.
fn downsample_rgba(src: &[u8], w: u32, h: u32) -> (Vec<u8>, u32, u32) {
    let nw = (w / 2).max(1);
    let nh = (h / 2).max(1);
    let mut dst = vec![0u8; (nw * nh * 4) as usize];
    for y in 0..nh {
        for x in 0..nw {
            // Average the 2×2 source block per channel.
            let mut acc = [0u32; 4];
            for dy in 0..2u32 {
                for dx in 0..2u32 {
                    let sx = (x * 2 + dx).min(w - 1);
                    let sy = (y * 2 + dy).min(h - 1);
                    let si = ((sy * w + sx) * 4) as usize;
                    for c in 0..4 {
                        acc[c] += src[si + c] as u32;
                    }
                }
            }
            let di = ((y * nw + x) * 4) as usize;
            for c in 0..4 {
                dst[di + c] = (acc[c] / 4) as u8;
            }
        }
    }
    (dst, nw, nh)
}

/// Encode RGBA8 layers (each `size×size`, tightly packed, layer-major) into a BC7
/// array with a full mip chain. `has_alpha` picks the alpha-aware encoder preset.
pub fn encode_layers_bc7(rgba_layers: &[u8], size: u32, layers: u32, has_alpha: bool) -> Bc7Array {
    use intel_tex_2::{RgbaSurface, bc7};

    let settings = if has_alpha {
        bc7::alpha_basic_settings()
    } else {
        bc7::opaque_basic_settings()
    };
    let layer_rgba = (size * size * 4) as usize;
    let mips = mip_count(size);
    let mut data = Vec::with_capacity(bc7_layer_bytes(size) * layers as usize);

    for layer in 0..layers as usize {
        let base = layer * layer_rgba;
        let mut level = rgba_layers[base..base + layer_rgba].to_vec();
        let mut w = size;
        let mut h = size;
        for _ in 0..mips {
            let surface = RgbaSurface {
                width: w,
                height: h,
                stride: w * 4,
                data: &level,
            };
            data.extend_from_slice(&bc7::compress_blocks(&settings, &surface));
            // Next level (clamped at the 4×4 block minimum; the loop count already
            // excludes sub-4×4 levels, so the final downsample is just unused).
            let (next, nw, nh) = downsample_rgba(&level, w, h);
            level = next;
            w = nw.max(4);
            h = nh.max(4);
        }
    }

    Bc7Array {
        data,
        width: size,
        height: size,
        layers,
        mip_levels: mips,
    }
}

/// A solid-color BC7 array (all layers, full mip chain) — cheap fallback fill while
/// real layers stream in. A solid colour compresses to one identical 16-byte block
/// everywhere, so we encode a single 4×4 block once and tile its bytes across the
/// whole array. `color` is RGBA8.
pub fn solid_fill_bc7(color: [u8; 4], size: u32, layers: u32) -> Bc7Array {
    use intel_tex_2::{RgbaSurface, bc7};

    // Encode one 4×4 solid block → 16 bytes.
    let block_rgba: Vec<u8> = color.iter().copied().cycle().take(4 * 4 * 4).collect();
    let block = bc7::compress_blocks(
        &bc7::opaque_basic_settings(),
        &RgbaSurface {
            width: 4,
            height: 4,
            stride: 16,
            data: &block_rgba,
        },
    );
    debug_assert_eq!(block.len(), 16);

    // Tile that block across every block of every mip of every layer.
    let total = bc7_layer_bytes(size) * layers as usize;
    let data = block
        .iter()
        .copied()
        .cycle()
        .take(total)
        .collect::<Vec<u8>>();

    Bc7Array {
        data,
        width: size,
        height: size,
        layers,
        mip_levels: mip_count(size),
    }
}

/// 64-bit content hash of the source bytes (FNV-1a via DefaultHasher), folded with
/// the encoder version so a settings change also invalidates the cache.
fn content_key(source: &[u8]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    ENCODER_VERSION.hash(&mut hasher);
    source.hash(&mut hasher);
    hasher.finish()
}

/// Disk-backed cache for a *set* of BC7 arrays in one blob (e.g. a variant's 5 PBR
/// maps). Layout: `[magic u32][key u64][count u32]` then per array `[width u32]
/// [layers u32][mips u32][len u32][BC7 bytes...]`. A missing / short / wrong-magic /
/// stale-key / wrong-count file is treated as a miss and re-encoded.
pub struct Bc7Cache {
    path: PathBuf,
}

impl Bc7Cache {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    /// Load `N` cached BC7 arrays if present and fresh for `source`; otherwise encode
    /// via `encode`, write the blob, and return the fresh arrays. `encode` runs only
    /// on a miss. Pure CPU + filesystem — safe to call from a background task.
    pub fn load_or_encode_multi<const N: usize>(
        &self,
        source: &[u8],
        encode: impl FnOnce() -> [Bc7Array; N],
    ) -> [Bc7Array; N] {
        let key = content_key(source);
        if let Some(cached) = self.try_read_multi::<N>(key) {
            return cached;
        }
        let arrays = encode();
        if let Err(e) = self.write_multi(key, &arrays) {
            warn!("BC7 cache: failed to write {}: {e}", self.path.display());
        }
        arrays
    }

    fn try_read_multi<const N: usize>(&self, key: u64) -> Option<[Bc7Array; N]> {
        let bytes = std::fs::read(&self.path).ok()?;
        // Header: magic(4) key(8) count(4) = 16 bytes.
        if bytes.len() < 16 {
            return None;
        }
        let rd_u32 = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let rd_u64 = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        if rd_u32(0) != CACHE_MAGIC || rd_u64(4) != key || rd_u32(12) as usize != N {
            return None; // foreign magic, stale source, or different array count → miss
        }
        let mut off = 16;
        let mut out: [Option<Bc7Array>; N] = std::array::from_fn(|_| None);
        for slot in out.iter_mut() {
            if off + 16 > bytes.len() {
                return None; // truncated
            }
            let width = rd_u32(off);
            let layers = rd_u32(off + 4);
            let mip_levels = rd_u32(off + 8);
            let len = rd_u32(off + 12) as usize;
            off += 16;
            if off + len > bytes.len() {
                return None; // truncated
            }
            *slot = Some(Bc7Array {
                data: bytes[off..off + len].to_vec(),
                width,
                height: width,
                layers,
                mip_levels,
            });
            off += len;
        }
        Some(out.map(|o| o.unwrap()))
    }

    fn write_multi<const N: usize>(&self, key: u64, arrays: &[Bc7Array; N]) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let total: usize = arrays.iter().map(|a| a.data.len() + 16).sum();
        let mut out = Vec::with_capacity(16 + total);
        out.extend_from_slice(&CACHE_MAGIC.to_le_bytes());
        out.extend_from_slice(&key.to_le_bytes());
        out.extend_from_slice(&(N as u32).to_le_bytes());
        for a in arrays {
            out.extend_from_slice(&a.width.to_le_bytes());
            out.extend_from_slice(&a.layers.to_le_bytes());
            out.extend_from_slice(&a.mip_levels.to_le_bytes());
            out.extend_from_slice(&(a.data.len() as u32).to_le_bytes());
            out.extend_from_slice(&a.data);
        }
        std::fs::write(&self.path, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_count_stops_at_block_min() {
        // Chain stops at 4×4 (BC7 block min): no 2×2/1×1 levels.
        assert_eq!(mip_count(4), 1); // just 4×4
        assert_eq!(mip_count(8), 2); // 8, 4
        assert_eq!(mip_count(1024), 9); // 1024…4
    }

    #[test]
    fn bc7_layer_bytes_chain() {
        // 4×4 = one block, 16B, single level.
        assert_eq!(bc7_layer_bytes(4), 16);
        // 8×8: 8 (2×2 blocks=64B) + 4 (1 block=16B) = 80B.
        assert_eq!(bc7_layer_bytes(8), 80);
        // 1024²: mip0 = 256×256 blocks ×16 = 1_048_576, plus the tail.
        assert!(bc7_layer_bytes(1024) > 1_048_576);
    }

    #[test]
    fn encode_roundtrip_size() {
        // Two solid 4×4 layers → one mip level each (4×4 is the block min).
        let layer = vec![128u8; 4 * 4 * 4];
        let mut src = layer.clone();
        src.extend_from_slice(&layer);
        let arr = encode_layers_bc7(&src, 4, 2, false);
        assert_eq!(arr.layers, 2);
        assert_eq!(arr.mip_levels, 1);
        assert_eq!(arr.data.len(), bc7_layer_bytes(4) * 2);
    }

    #[test]
    fn content_key_changes_with_source() {
        assert_ne!(content_key(&[1, 2, 3]), content_key(&[1, 2, 4]));
    }
}
