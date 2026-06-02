//! Shared filesystem/path + world helpers for the editor. Consolidates path predicates,
//! directory traversal, and the "take a registry out of the world, dispatch against it,
//! put it back" pattern that several panels need for exclusive `&mut World` access.

use std::path::{Path, PathBuf};

use bevy::prelude::*;

use super::assets_browser::ASSETS_ROOT;

/// Supported raster image extensions (lowercase, no dot).
pub const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "bmp", "tga"];

/// Whether `path` has a supported image extension (case-insensitive).
pub fn is_image_file(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => {
            let ext = ext.to_lowercase();
            IMAGE_EXTENSIONS.contains(&ext.as_str())
        }
        None => false,
    }
}

/// Whether `path` is a `*.material.ron` asset (case-insensitive).
pub fn is_material_ron(path: &Path) -> bool {
    path.to_string_lossy().to_lowercase().ends_with(".material.ron")
}

/// Convert a working-dir path under `assets/` to its root-relative form (e.g.
/// `assets/materials/x.ron` → `materials/x.ron`). `None` if not under the assets root.
pub fn relative_to_assets(path: &Path) -> Option<PathBuf> {
    path.strip_prefix(ASSETS_ROOT).ok().map(Path::to_path_buf)
}

/// Final path component as an owned `String` (lossy), or empty if none.
pub fn file_name_str(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Read the direct children of `dir`, partitioned into (dirs, files), each sorted
/// alphabetically (case-insensitive). Returns empty vecs if `dir` can't be read.
pub fn read_sorted(dir: &Path) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                dirs.push(p);
            } else {
                files.push(p);
            }
        }
    }
    let by_name = |p: &PathBuf| file_name_str(p).to_lowercase();
    dirs.sort_by_key(by_name);
    files.sort_by_key(by_name);
    (dirs, files)
}

/// How long a cached directory listing stays valid before [`read_sorted_cached`] re-reads.
/// 5s is the target responsiveness for new/removed files appearing in the file panels.
const DIR_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(5);

/// Per-path cache entry: when it was read, plus the sorted (dirs, files).
type DirCacheMap =
    std::collections::HashMap<PathBuf, (std::time::Instant, Vec<PathBuf>, Vec<PathBuf>)>;

thread_local! {
    /// Main-thread cache of [`read_sorted`] results. The editor's file panels walk the same
    /// directories every frame; this avoids a blocking `read_dir` syscall per frame (a real
    /// chunk of the egui-pass cost — see the perf roadmap E1).
    static DIR_CACHE: std::cell::RefCell<DirCacheMap> =
        std::cell::RefCell::new(DirCacheMap::new());
}

/// Like [`read_sorted`] but memoized per path for [`DIR_CACHE_TTL`] on the calling (main)
/// thread. Returns owned clones. New/removed files appear within the TTL. Call only from the
/// main thread (the editor egui pass) — the cache is thread-local.
pub fn read_sorted_cached(dir: &Path) -> (Vec<PathBuf>, Vec<PathBuf>) {
    DIR_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let fresh = cache
            .get(dir)
            .is_some_and(|(read_at, ..)| read_at.elapsed() < DIR_CACHE_TTL);
        if !fresh {
            let (dirs, files) = read_sorted(dir);
            cache.insert(dir.to_path_buf(), (std::time::Instant::now(), dirs, files));
        }
        let (_, dirs, files) = &cache[dir];
        (dirs.clone(), files.clone())
    })
}

/// Recursively collect every file (not directory) under `dir`.
pub fn walk_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let (dirs, files) = read_sorted(&d);
        stack.extend(dirs);
        out.extend(files);
    }
    out
}

/// Take resource `R` out of the world so `f` gets exclusive `&mut World` plus a `&R`,
/// then restore it. The dispatch registries (inspector overrides, asset inspectors,
/// thumbnail providers, debug panels) all need this: their callbacks want `&mut World`,
/// which they can't have while the registry is borrowed from the same world.
pub fn with_registry<R, T>(world: &mut World, f: impl FnOnce(&mut World, &R) -> T) -> T
where
    R: Resource + Default,
{
    let registry = world.remove_resource::<R>().unwrap_or_default();
    let out = f(world, &registry);
    world.insert_resource(registry);
    out
}

/// Edit the asset behind `handle` against a clone, writing back via `get_mut` only if it
/// actually changed. A bare `get_mut` every frame fires `AssetEvent::Modified` even with
/// no edit, which needlessly invalidates dependents (e.g. thumbnails). No-op while the
/// asset is still loading. Returns whether a write-back happened.
pub fn edit_asset<A>(
    world: &mut World,
    handle: &Handle<A>,
    f: impl FnOnce(&mut A, &mut World),
) -> bool
where
    A: Asset + Clone + PartialEq,
{
    let Some(mut edited) = world.resource::<Assets<A>>().get(handle).cloned() else {
        return false;
    };
    let before = edited.clone();
    f(&mut edited, world);
    if edited != before
        && let Some(slot) = world.resource_mut::<Assets<A>>().get_mut(handle)
    {
        *slot = edited;
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_predicate_is_case_insensitive() {
        assert!(is_image_file(Path::new("a.png")));
        assert!(is_image_file(Path::new("a.JPG")));
        assert!(is_image_file(Path::new("dir/b.tga")));
        assert!(!is_image_file(Path::new("a.ron")));
        assert!(!is_image_file(Path::new("noext")));
    }

    #[test]
    fn material_ron_predicate() {
        assert!(is_material_ron(Path::new("materials/x.material.ron")));
        assert!(is_material_ron(Path::new("X.MATERIAL.RON")));
        assert!(!is_material_ron(Path::new("x.pbrtex.ron")));
        assert!(!is_material_ron(Path::new("x.ron")));
    }

    #[test]
    fn relative_to_assets_strips_root() {
        assert_eq!(
            relative_to_assets(Path::new("assets/materials/x.ron")),
            Some(PathBuf::from("materials/x.ron"))
        );
        assert_eq!(relative_to_assets(Path::new("elsewhere/x.ron")), None);
    }

    #[test]
    fn file_name_str_handles_missing() {
        assert_eq!(file_name_str(Path::new("a/b/c.png")), "c.png");
        assert_eq!(file_name_str(Path::new("")), "");
    }
}
