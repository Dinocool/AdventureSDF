//! # SDF clipmap renderer
//!
//! Renders an editable signed-distance-field world by raymarching a sparse brick atlas,
//! with camera-centred LOD shells so it can reach vast distances. The data flow, in order,
//! and where each stage lives:
//!
//! 1. **Edits → analytic CSG field** (`edits`). Each [`SdfVolume`] is a primitive + CSG op
//!    (`fold_csg`). This field is *resolution-independent*: callable at any point and any
//!    scale. Everything downstream samples it.
//! 2. **Per-LOD bake** (`atlas`). For each resident brick, every voxel stores the analytic
//!    CSG distance (`edits::fold_csg`) sampled at the voxel centre — a true trilinear SDF.
//!    A coarse brick samples the analytic field at its own (larger) voxel scale, so far
//!    geometry bakes correctly without any LOD-0 data, and the surface sits at the same
//!    place at every LOD (no inter-LOD seam). Trade-off: a feature thinner than a voxel can
//!    be missed at coarse LOD (its zero-crossing falls between samples) — accepted as the
//!    cost of a clean, un-inflated field.
//! 3. **Sparse storage + GPU lookup** (`chunk`, `render`, `bindings.wgsl`). Bricks group
//!    into 4³=64-brick **chunks** addressed by an *absolute* world-lattice key (independent
//!    of the camera, so CPU and GPU agree by construction). Resident chunks live in a per-LOD
//!    **toroidal directory** — a dense `R³` array per LOD where chunk `c` sits at the fixed slot
//!    `c mod R`, so the GPU resolves it by a direct index + key-tag compare (no sort, no binary
//!    search) and the CPU inserts/evicts in O(1). Each slot carries a 64-bit occupancy mask +
//!    popcount index into a packed (sparse) tile-run buffer. Brick texels live in a 2D-tiled
//!    atlas texture.
//! 4. **Async incremental bake** (`bake_scheduler`). The camera-centred chunk ring recenters
//!    as the camera moves; entered chunks bake on a task pool, exited chunks evict IMMEDIATELY
//!    (the march falls back to a coarser resident LOD during the brief handoff) — never blocking
//!    the main thread.
//! 5. **Unified raymarch** (`sdf_raymarch.wgsl`, helpers in `brick`). One loop:
//!    resolve the finest resident LOD at `p`; skip empty space by brick-DDA; otherwise
//!    sphere-trace the trilinear field and accept the hit once the surface is within the
//!    pixel cone (screen-space termination — the vast-distance speed win). There is **no GPU
//!    BVH** in the march; the field + brick-geometry DDA drive all skipping. The `bvh` module
//!    is CPU-only, used solely as the bake's edit-culling acceleration structure.
//!
//! Editor-only pieces (`debug`, `gizmo`, `picking`, overlays) sit alongside but are not on
//! the render hot path.

pub mod atlas;
pub mod bake_scheduler;
pub(crate) mod bc7;
pub mod bvh;
pub mod chunk;
#[cfg(feature = "editor")]
pub(crate) mod debug;
pub(crate) mod editor_camera;
pub mod edits;
// The gallery + cornell modules are purely scene GENERATORS (the runtime loads the serialized
// `assets/scenes/*.scene`); only the regen tests use them, so they're test-only.
#[cfg(test)]
mod cornell;
#[cfg(test)]
mod gallery;
// Mesh-bake migration test scene (sharp cube / sphere / smooth blend / subtraction) — see
// `mesh_test.rs`. Test-only generator like `gallery`; runtime loads the serialized `.scene`.
#[cfg(test)]
mod mesh_test;
/// Phase-0 SDF→mesh bake spike (Surface Nets via `fast_surface_nets`). Added as `MeshBakePlugin` in
/// `main.rs`; see `docs/MESH_BAKE_PLAN.md`.
pub mod mesh_bake;
pub mod gizmo;
pub(crate) mod height;
pub mod light_grid;
pub(crate) mod node_gizmos;
pub(crate) mod overlays;
pub(crate) mod picking;
pub mod probe;
pub mod render;
pub(crate) mod scatter;
pub(crate) mod stress;
pub mod textures;
pub(crate) mod tower_field;

use bevy::prelude::*;

use crate::scene_manager::AppScene;

// The editor viewport cameras (orbit + free-fly) live in `editor_camera`, and the gizmo overlays in
// `overlays`. Their public types are re-exported here so cross-module consumers keep the stable
// `sdf_render::` path.
pub use editor_camera::{
    CameraInput, OrbitFocus, SdfCameraMode, SdfOrbitCamera, sync_orbit_camera_transform,
};
pub use overlays::{LodRingsVisible, SdfGridGizmos, SdfNodeGizmos, SdfOverlayGizmos};

// --- Components ---

// Edit primitives, CSG ops, ordering, and material live in `edits`. Re-exported
// here so the rest of the module (and external callers) keep a stable
// `sdf_render::` path.
pub use edits::{CsgKind, MaterialFields, SdfMaterial, SdfMaterialSource, SdfOp, SdfOrder, SdfPrimitive};

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
#[require(crate::node::Node3D)]
pub struct SdfVolume;

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct SdfCamera;

/// Whether the debug toolkit is currently drawing SDF bounds wireframes. Owned by
/// the core module so the gizmo-draw system can live behind the feature gate
/// without the resource type vanishing from the core build.
#[derive(Resource, Default)]
pub struct WireframeBoundsVisible(pub bool);

/// Per-[`GizmoKind`](crate::node::GizmoKind) viewport visibility. A kind absent from the map is
/// VISIBLE (default-on); the editor's "View" toolbar writes entries to hide/show a type. Owned by
/// the core module (not editor-gated) so the always-compiled `node_gizmos::draw_node_gizmos` can
/// read it; the mutating UI is editor-only. Driven entirely by `GizmoKind::ALL`, so a new gizmo
/// type gets a toggle for free.
#[derive(Resource, Default)]
pub struct GizmoVisibility(
    pub bevy::platform::collections::HashMap<crate::node::GizmoKind, bool>,
);

impl GizmoVisibility {
    /// Whether gizmos of `kind` should draw (absent ⇒ visible).
    pub fn is_visible(&self, kind: crate::node::GizmoKind) -> bool {
        self.0.get(&kind).copied().unwrap_or(true)
    }
}

/// Diagnostic: world-space center + size of recently-baked bricks, each tagged with the time it
/// was baked so the editor can FADE the wire box out over a few seconds. Lets you SEE which
/// bricks an edit move dirties (e.g. confirm a far small object doesn't touch the heightmap) AND
/// how rapidly — a continuous drag leaves a bright cloud, an idle frame fades to nothing.
/// Entries accumulate across frames (NOT cleared each frame); the draw system drops expired ones.
/// `enabled` gates collection so it costs nothing when off. Owned by the core module so the
/// scheduler can fill it without the editor feature; the draw system is editor-gated.
#[derive(Resource, Default)]
pub struct BakedBrickDebug {
    pub enabled: bool,
    /// (center, edge_size, baked_at_secs) per recently-baked brick.
    pub bricks: Vec<(Vec3, f32, f32)>,
}

/// How long (seconds) a baked-brick marker stays visible, fading to transparent over its life.
pub const BAKED_BRICK_FADE_SECS: f32 = 2.0;

/// Last CPU ray-step capture from the debug ray inspector. Empty until a capture
/// is requested.
#[derive(Resource, Default)]
pub struct RayStepCapture {
    pub steps: Vec<picking::RayStep>,
}

/// Toggle for the SDF fullscreen raymarch pass. F1 flips this. Must be `ExtractResource` (synced to
/// the render world) — the render nodes read it there to short-circuit; without the sync F1 has no
/// effect (the render world would never see the flipped value).
#[derive(Resource, Clone, bevy::render::extract_resource::ExtractResource)]
pub struct SdfRenderEnabled(pub bool);

impl Default for SdfRenderEnabled {
    fn default() -> Self {
        Self(true)
    }
}

/// Whether viewport input (orbit/pick/gizmo-drag) is allowed this frame. The
/// editor sets this from the pointer-in-viewport test so clicks on dock panels
/// don't fall through to the 3D scene. Defaults to `true` so the non-editor build
/// (full-window viewport, no panels) keeps working unchanged.
#[derive(Resource)]
pub struct ViewportInputAllowed(pub bool);

impl Default for ViewportInputAllowed {
    fn default() -> Self {
        Self(true)
    }
}

/// Live raymarch tuning, fed to the shader each frame via the camera uniform's
/// `debug_params`. Always present (defaults match the historical shader constants)
/// so the render path never depends on the debug toolkit feature.
#[derive(Resource, Reflect)]
#[reflect(Resource)]
pub struct SdfRaymarchParams {
    pub max_steps: u32,
    pub max_dist: f32,
    pub sdf_eps: f32,
    /// Multiplier on the per-pixel cone half-width used for screen-space march
    /// termination. The march stops when the conservative field drops below
    /// `pixel_cone · t` (surface within ~`cone_scale` pixels), so far geometry resolves
    /// at coarse LOD instead of marching down to LOD 0 — the vast-distance speed win.
    /// 1.0 = exactly one pixel; larger = coarser/cheaper, smaller = sharper/costlier.
    pub cone_scale: f32,
    /// Sphere-trace over-relaxation factor (Keinert 2014). The march steps `over_relax · d`
    /// with a safe fallback when consecutive unbounding spheres separate, converging on
    /// grazing rays in fewer steps. 1.0 = plain sphere tracing; (1,2) accelerates. Default
    /// 1.6: measured (tests/sdf_march_sim.rs) big step cut on grazing-MISS rays (the slow
    /// tangent-band crawl) with zero hit↔miss flips — the fallback undoes any overshoot on
    /// hits, and the cross-fade shell forces ω=1 where the blended field is non-eikonal.
    /// (1.8 cut more in the sim but showed visual artifacts on the real scene, so backed off
    /// to 1.6 for margin below the ω<2 overlapping-sphere safety ceiling.)
    pub over_relax: f32,
    /// LOD cross-fade band width, as a fraction of each clipmap ring's half-extent. In the
    /// outer `lod_blend_band` shell of a ring the marched field is `mix`-faded from the
    /// serving LOD toward its coarser neighbour, so the surface morphs smoothly across the
    /// ring boundary instead of snapping (removes the visible LOD pop/seam). 0 = disabled
    /// (hard LOD seams, the original behaviour). Tunable live via the editor raymarch panel.
    ///
    /// Default 0 (OFF): with bake-time curvature compensation + the 256 ring, the per-LOD surface
    /// shrink is small enough that the morph isn't needed — and the distance-driven morph itself
    /// coarsened the surface INSIDE a level's ring (it shrank MORE than the raw LOD). Re-enable via
    /// the slider if a hard LOD seam shows up at a transition.
    pub lod_blend_band: f32,
    /// Soft-shadow penumbra hardness `k` (the IQ `min(k·d/t)` factor in `sdf::shadows`). LOWER =
    /// softer/wider penumbra, which blurs the coarse-LOD brick faceting AND softens the
    /// penumbra→umbra edge (both quantified on the harness tradeoff curve in
    /// `tests/sdf_shadow_harness`); HIGHER = sharper/tighter but boxier + harder-edged. Tunable
    /// live via the editor raymarch panel ("Shadow Softness").
    pub shadow_softness: f32,
    /// How many point lights (brightest-first, of those reaching a surface) cast an SDF shadow per
    /// pixel; the rest add unshadowed. Bounds the per-pixel shadow-march cost. Uploaded into the
    /// camera uniform so it's live-tunable from the editor raymarch panel ("Shadow lights") with no
    /// shader rebuild.
    pub shadow_light_cap: u32,
}

impl Default for SdfRaymarchParams {
    fn default() -> Self {
        Self {
            // Raised for vast-distance marching: cone termination keeps the step count
            // bounded even though the reach is far larger than the old 100-unit cap.
            max_steps: 192,
            max_dist: 5000.0,
            sdf_eps: 0.001,
            cone_scale: 1.0,
            over_relax: 1.6,
            lod_blend_band: 0.0, // OFF — LODs are good enough without the morph (see field doc)
            // Soft-shadow penumbra hardness `k` (`sdf::shadows`): 0 = HARD shadow (binary
            // occlusion, no penumbra — artifact-free); >0 = cone-traced soft, HIGHER = tighter
            // (less near-miss darkening). A tight default (64) stays clean; the soft end (low k)
            // re-introduces the penumbra near-miss/field artifacts.
            shadow_softness: 64.0,
            // Safety ceiling on shadowed lights per pixel (live "Shadow lights" slider). The real
            // cull is distance-based (shadows only within a fraction of each light's range), so this
            // rarely binds — it just bounds pathological clusters. 0 = no point-light shadows.
            shadow_light_cap: 8,
        }
    }
}

/// Live DDGI (probe-based global illumination) tuning. Extracted to the render world and packed into
/// the probe-trace params uniform each frame; exposed as editor sliders (the knobs-as-uniforms
/// invariant). `enabled` gates the whole trace + apply so GI can be toggled with no cost when off.
#[derive(Resource, Reflect, Clone, bevy::render::extract_resource::ExtractResource)]
#[reflect(Resource)]
pub struct DdgiParams {
    /// Rays traced per probe per frame (Fibonacci sphere). Higher = smoother irradiance, costlier.
    pub ray_count: u32,
    /// Caps the progressive-average sample count: `N_max = 1/(1-hysteresis)`. Higher = longer
    /// steady-state window = more accumulated samples = smoother / less boil, but slower to react to
    /// lighting changes (more lag). 0.95 ≈ 20 samples. Trades stability against responsiveness; the
    /// history clamp (`change_thresh`) keeps boil bounded so this can sit lower than a plain EMA would.
    pub hysteresis: f32,
    /// Multiplies the gathered irradiance before it is added to the lit result.
    pub intensity: f32,
    /// Per-brick probe sub-lattice factor: each occupied brick emits `subdiv³` probes, so LOD-0
    /// probe spacing is `brick_size / subdiv` (subdiv 1 = one probe/brick ≈ 0.7 m; 2 ≈ 0.35 m).
    /// Costs `subdiv³`× probe rays + irradiance memory.
    pub subdiv: u32,
    /// Round-robin amortization: only `1/update_stride` of probes re-trace each frame (the rest carry
    /// forward and converge over `update_stride` frames via temporal blend). Bounds per-frame trace
    /// cost ~`1/update_stride`. 1 = trace every probe every frame (most expensive, no latency).
    pub update_stride: u32,
    /// Max distance (world units) a probe ray marches. Indirect bounce is local, so a short range
    /// keeps per-ray cost bounded regardless of the multi-km clipmap reach (the dominant cost at high
    /// LOD count). Geometry beyond this contributes via coarser/farther probes, not these rays.
    pub gi_range: f32,
    /// Apply-side surface bias along the normal, as a fraction of the probe cell: pushes the shading
    /// point off the surface toward the lit side so trilinear favours front-facing probes (anti-leak).
    pub normal_bias: f32,
    /// Apply-side surface bias toward the camera, as a fraction of the probe cell: reduces self-shadow
    /// artifacts at grazing angles (RTXGI view bias).
    pub view_bias: f32,
    /// Screen-space GI denoise: depth edge-stop tolerance (relative to camera distance). Lower = the
    /// blur respects depth discontinuities more strictly (less bleed across surfaces, but can keep more
    /// of the probe-lattice blocks); higher = wider blur across depth.
    pub gi_blur_depth_sigma: f32,
    /// Screen-space GI denoise: normal edge-stop sharpness. Higher = the blur stops harder at normal
    /// discontinuities (creases, silhouettes); lower = smoother across them.
    pub gi_blur_normal_power: f32,
    /// Scales the analytic sky's contribution to the GI bounce (escaped probe rays). 1.0 = the full
    /// physical sky (`sdf::sky`) lights the scene indirectly; 0.0 isolates GI to scene emitters + sun
    /// only (used by the harness gates, and useful for interiors where the sky shouldn't bleed in).
    pub gi_sky_intensity: f32,
    /// Shadow the direct lighting gathered at each probe-ray bounce hit: a secondary SDF march toward
    /// the sun (bounded to `gi_range`) + a sphere-shadow for the brightest point light reaching the
    /// hit. Prevents direct light leaking through walls into GI, at the cost of a shadow march per ray
    /// hit — the dominant trace cost, so it's a toggle. Off = the bounce uses unshadowed direct light.
    pub gi_bounce_shadows: bool,
    /// Hard ceiling (bytes) on the probe irradiance buffer — a safety net so a large scene can't size
    /// it past `max_storage_buffer_binding_size` (the wgpu binding limit). When the LOD-aware probe
    /// count still exceeds this, the buffer is clamped and the over-budget probe slots go inactive
    /// (the trace/apply already bounds-check `arrayLength`), with a one-shot warning. The effective cap
    /// is `min(this, device max_storage_buffer_binding_size)`.
    pub probe_budget_bytes: u32,
    /// Probe CLASSIFICATION: once a probe has converged (and nothing nearby changed), drop it to a much
    /// lower re-trace rate (`dormant_stride`) instead of the active `update_stride` — it keeps its value
    /// and skips the expensive ray-march. A global `gi_epoch` (bumped on topology / lighting change)
    /// wakes all probes for a fast re-converge; the dormant re-trace rate also bounds staleness from any
    /// change the epoch didn't catch. Off = every finest probe traces at `update_stride` (no pruning).
    pub classify_enabled: bool,
    /// Re-trace rate for DORMANT (converged + unchanged) probes: `1/dormant_stride` of them per frame.
    /// Larger = bigger steady-state savings on static scenes + slower revalidation of undetected changes.
    /// Only used when `classify_enabled`. Typical 16–64.
    pub dormant_stride: u32,
    /// LOD at/above which DISTANT probes trace fewer rays (`distant_ray_count`) — the far field needs far
    /// less angular detail, so this cuts the dominant ray-march cost without touching near quality. Set
    /// high (≥ lod_count) to disable.
    pub ray_falloff_lod: u32,
    /// Rays/probe for probes at LOD ≥ `ray_falloff_lod` (the distant field). Lower than `ray_count`.
    pub distant_ray_count: u32,
    /// LOD at/above which probe DENSITY is halved (a checkerboard decimation of bricks): distant probes
    /// are sparser, cutting probe COUNT (memory + ray work) where the GI is low-frequency anyway. The
    /// apply's coverage-weighted trilinear interpolates the gaps. Set high (≥ lod_count) to disable.
    pub probe_halve_lod: u32,
    /// Sphere-trace step budget per GI ray (the slim GI march's `MAX_STEPS`). GI is low-frequency and the
    /// temporal average hides the rare under-converged ray, so this can sit well below the primary pass's
    /// budget. Lower = cheaper per ray (the dominant trace cost) but blockier far hits. Clamped 1..=64.
    pub gi_march_steps: u32,
    /// RELEVANCE CULL: finest probes whose chunk is outside the view cone AND beyond `cull_near_radius`
    /// re-trace only `1/cull_off_stride` of the time (they keep their irradiance) — a moving camera then
    /// spends its probe budget on what's on-screen instead of fully converging probes behind it. Off =
    /// every finest probe traces at the normal (active/dormant) rate regardless of view.
    pub relevance_cull: bool,
    /// Re-trace rate for off-screen (culled) probes: `1/cull_off_stride` per frame. Larger = bigger
    /// moving-camera savings + slower catch-up when the camera turns toward a culled region (the wake
    /// system also re-activates edited regions). Only used when `relevance_cull`.
    pub cull_off_stride: u32,
    /// Probes within this world-space radius of the camera are ALWAYS relevant (never culled), regardless
    /// of view direction — nearby surfaces behind/beside the camera can still bounce light onto on-screen
    /// geometry, so a generous near shell avoids darkening contact GI when the camera turns.
    pub cull_near_radius: f32,
    /// View-cone cull threshold: a chunk is relevant if `dot(dir_to_chunk, camera_forward) > this`. −1 =
    /// never cull (whole sphere relevant); 0 = cull the rear hemisphere; −0.2 ≈ keep a ~101° half-cone
    /// (FOV + margin). Lower (toward −1) = more conservative (fewer culled); higher = more aggressive.
    pub cull_cone_dot: f32,
    /// PER-FRAME DISPATCH CAP: at most this many finest probe-chunks trace per frame (0 = unlimited).
    /// On a scene load / fast camera, far more chunks become turn-eligible at once than a frame can
    /// afford — that flood is the scene-start GPU dip. The cap bounds it, and the eligible chunks are
    /// dispatched NEAREST-TO-CAMERA first, so visible GI converges first and the rest fill in over the
    /// next frames. At steady state (few eligible) the cap never binds. Tune up for faster whole-scene
    /// convergence, down for a smoother load.
    pub max_probe_chunks_per_frame: u32,
}

impl Default for DdgiParams {
    fn default() -> Self {
        Self {
            ray_count: 128,
            // Progressive-average window N_max = 1/(1-h) ≈ 20: accumulates per-frame-rotated ray sets
            // (smoothness/low boil) while staying reasonably responsive; the history clamp bounds boil.
            hysteresis: 0.95,
            // DDGI DISABLED for the mesh-bake pivot — Bevy Solari (mesh-native raytraced GI) will
            // replace our custom DDGI/RC/surfel stack once meshes land. 0 = no GI contribution (the
            // lit pass adds `albedo × probe_irradiance × intensity`). NOTE: this only zeroes the
            // OUTPUT — the probe-trace/resolve/blur passes still run; fully removing them is a plan
            // step (see docs/MESH_BAKE_PLAN.md). Re-enable by restoring intensity to 1.0.
            intensity: 0.0,
            subdiv: 2,
            update_stride: 4,
            gi_range: 24.0,
            normal_bias: 0.6,
            view_bias: 0.1,
            gi_blur_depth_sigma: 0.15,
            gi_blur_normal_power: 16.0,
            gi_sky_intensity: 1.0,
            gi_bounce_shadows: true,
            // 1 GiB — comfortably below the common 2 GiB binding limit, headroom for a big scene.
            probe_budget_bytes: 1 << 30,
            classify_enabled: true,
            dormant_stride: 32,
            // Distant LODs get cheaper probes by default (the user can dial these): half the rays beyond
            // LOD 4, and half the probe density beyond LOD 5. Near LODs (the visible field) keep full
            // quality. Raise the thresholds to lod_count to disable.
            ray_falloff_lod: 4,
            distant_ray_count: 32,
            probe_halve_lod: 5,
            // GI rays take a short step budget (the slim march): far cheaper than the primary pass, and
            // the temporal average hides the occasional under-converged ray.
            gi_march_steps: 16,
            // Relevance cull ON by default: off-screen finest probes throttle to 1/64. Generous near shell
            // (6 m) + a ~101° half-cone keep contact GI correct when the camera turns.
            relevance_cull: true,
            cull_off_stride: 64,
            cull_near_radius: 6.0,
            cull_cone_dot: -0.2,
            // Cap the per-frame probe-chunk dispatch so a scene load / fast camera spreads convergence
            // over frames (nearest-first) instead of one huge stall. 256 chunks × up to CHUNK_VOLUME
            // bricks is ample for fast convergence yet tames the 1024-room flood; never binds on small scenes.
            max_probe_chunks_per_frame: 256,
        }
    }
}

/// How many consecutive frames the GI inputs (geometry topology, sun, GI lighting knobs) have been
/// UNCHANGED. Probe classification only lets probes go dormant once this exceeds the convergence window
/// — so a moving camera / changing light keeps every finest probe at full re-trace rate (no dormancy, no
/// stale GI, and slot-churn re-converges immediately), and only a genuinely settled scene saves ray
/// work. Maintained by [`track_gi_settle`], extracted to the render world for `prepare_sdf_probe`.
#[derive(Resource, Clone, Copy, Default, bevy::render::extract_resource::ExtractResource)]
pub struct GiSettle {
    pub frames_unchanged: u32,
}

/// Reset [`GiSettle`] to 0 whenever the GI inputs change (topology generation, sun direction/illuminance,
/// or the GI lighting knobs), else increment. Point lights are intentionally NOT hashed here (O(lights)
/// every frame would be costly on a 1000s-of-lights scene); a moved point light is instead picked up by
/// the dormant re-trace rate within `dormant_stride` frames — bounded, never permanently stale.
#[allow(clippy::type_complexity)] // a Bevy change-detection query filter — alias would obscure it
fn track_gi_settle(
    ddgi: Res<DdgiParams>,
    sun: Query<(&GlobalTransform, &DirectionalLight), With<crate::scene_manager::SceneEntity>>,
    // Cheap O(changed) detection of a moved/edited point light (NOT O(all lights) — Bevy change ticks).
    // A changing light must wake the probes; otherwise dormant probes update only every `dormant_stride`
    // frames → visible STEPPING instead of a smooth blend (the static case already converges then stays).
    lights_changed: Query<
        (),
        (With<PointLight>, Or<(Changed<GlobalTransform>, Changed<PointLight>)>),
    >,
    mut settle: ResMut<GiSettle>,
    mut last: Local<u64>,
) {
    // NOTE: geometry/topology changes do NOT reset the global settle — they're handled by the LOCAL
    // wake set (`update_probe_wake`), so one moving cube doesn't drop the whole scene out of dormancy.
    // LIGHTING changes (sun, point lights, GI knobs) globally wake every probe (they affect all of them),
    // so the GI re-converges at the active rate (smoothly) instead of stepping at the dormant rate.
    let mut h = 0u64;
    if let Ok((t, dl)) = sun.single() {
        let f = t.forward();
        h ^= (f.x.to_bits() as u64)
            ^ ((f.y.to_bits() as u64) << 1)
            ^ ((f.z.to_bits() as u64) << 2)
            ^ ((dl.illuminance.to_bits() as u64) << 3);
    }
    for v in [ddgi.intensity, ddgi.gi_sky_intensity, ddgi.gi_range] {
        h = h.wrapping_mul(0x0100_0000_01b3).wrapping_add(v.to_bits() as u64);
    }
    h = h.wrapping_add(u64::from(ddgi.gi_bounce_shadows));
    let hash_changed = *last != h;
    *last = h;
    if hash_changed || !lights_changed.is_empty() {
        settle.frames_unchanged = 0;
    } else {
        settle.frames_unchanged = settle.frames_unchanged.saturating_add(1);
    }
}

/// The set of DDGI probe chunk-SLOTS currently "awake" — recently changed (or adjacent to a change), so
/// they re-converge at the ACTIVE re-trace rate while the rest of a settled scene stays dormant. Built
/// by [`update_probe_wake`], extracted to the render world's `prepare_sdf_probe` (localized edit wake —
/// no global FPS cliff when one cube moves).
#[derive(Resource, Clone, Default, bevy::render::extract_resource::ExtractResource)]
pub struct ProbeWakeSet {
    pub slots: Vec<u32>,
}

/// Internal main-world bookkeeping: chunk slot → frames remaining awake.
#[derive(Resource, Default)]
struct ProbeWake {
    frames: std::collections::HashMap<u32, u32>,
}

/// Frames a changed region stays at the active re-trace rate before returning to dormant — long enough
/// to re-converge the temporal blend (≈ `n_max` traces at the active stride).
const PROBE_WAKE_FRAMES: u32 = 90;

/// Maintain the DDGI wake set: each changed chunk (+ its 3×3×3 same-LOD neighbourhood, so contact
/// shadows / colour bleed on adjacent surfaces re-converge too) is woken for [`PROBE_WAKE_FRAMES`];
/// expired entries age out. Cheap when nothing changed (the common case drains an empty set).
fn update_probe_wake(
    mut atlas: ResMut<atlas::SdfAtlas>,
    mut wake: ResMut<ProbeWake>,
    mut set: ResMut<ProbeWakeSet>,
) {
    let changed = atlas.live_chunks.drain_wake_keys();
    for ck in changed {
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nb = chunk::ChunkKey::new(ck.lod, ck.coord + IVec3::new(dx, dy, dz));
                    if let Some(slot) = atlas.live_chunks.slot_of(nb) {
                        wake.frames.insert(slot, PROBE_WAKE_FRAMES);
                    }
                }
            }
        }
    }
    if !wake.frames.is_empty() {
        wake.frames.retain(|_, f| {
            *f = f.saturating_sub(1);
            *f > 0
        });
    }
    set.slots.clear();
    set.slots.extend(wake.frames.keys().copied());
}

/// The finest probe chunk-SLOTS currently CULLED by the relevance test — off-screen (outside the view
/// cone) AND beyond the near shell. The render world's `prepare_sdf_probe` re-traces these only
/// `1/cull_off_stride` of the time, so a moving camera spends its probe budget on what's visible. Built
/// by [`update_probe_relevance`] (main world, where the camera lives) + extracted. `culled`/`total` ride
/// along for the DDGI debug panel.
#[derive(Resource, Clone, Default, bevy::render::extract_resource::ExtractResource)]
pub struct ProbeRelevanceSet {
    pub culled_slots: Vec<u32>,
    pub culled: u32,
    pub total: u32,
    /// Finest chunk SLOT → squared distance from its world center to the camera. Filled for EVERY finest
    /// chunk (independent of the cull toggle) so the render world's per-frame probe-dispatch CAP can
    /// prioritise the nearest chunks (visible GI converges first during a load/fast-camera flood).
    pub dist2_by_slot: std::collections::HashMap<u32, f32>,
}

/// Maintain the DDGI relevance-cull set: a finest-resident chunk whose world center is outside the
/// camera's view cone (`dot(dir_to_chunk, forward) <= cull_cone_dot`) AND beyond `cull_near_radius` is
/// CULLED — the render world throttles its probes to `1/cull_off_stride`. Cheap: O(finest chunks), one
/// normalize + dot each. Disabled (or no SDF camera) → empty set (nothing culled). The wake set still
/// overrides this in the render world, so an edited off-screen region re-converges promptly.
fn update_probe_relevance(
    atlas: Res<atlas::SdfAtlas>,
    ddgi: Res<DdgiParams>,
    config: Res<SdfGridConfig>,
    cameras: Query<&GlobalTransform, With<SdfCamera>>,
    mut set: ResMut<ProbeRelevanceSet>,
) {
    set.culled_slots.clear();
    set.culled = 0;
    set.total = 0;
    set.dist2_by_slot.clear();
    let Some(cam) = cameras.iter().next() else {
        return; // no camera (headless / test) → nothing culled, no distances
    };
    let cam_pos = cam.translation();
    let fwd = cam.forward().as_vec3();
    let near2 = ddgi.cull_near_radius * ddgi.cull_near_radius;
    for (ck, slot) in atlas.live_chunks.finest_slots_keyed() {
        set.total += 1;
        let center = chunk::chunk_min_world(ck, &config)
            + Vec3::splat(0.5 * chunk::chunk_world_size(ck.lod, &config));
        let to = center - cam_pos;
        let d2 = to.length_squared();
        // Distance is recorded for EVERY finest chunk (the dispatch cap uses it even with the cull off).
        set.dist2_by_slot.insert(slot, d2);
        if ddgi.relevance_cull {
            let relevant = d2 <= near2 || to.normalize_or_zero().dot(fwd) > ddgi.cull_cone_dot;
            if !relevant {
                set.culled_slots.push(slot);
                set.culled += 1;
            }
        }
    }
}

/// Monotonic counter bumped on every scene switch ([`SceneSwitched`]) — the render-world SDF cache-reset
/// signal. The render world compares it to detect a switch and start the new scene clean: `prepare_sdf_
/// probe` ZEROES the probe irradiance buffer (the grow-with-headroom buffer otherwise reuses old slots'
/// converged GI), and `prepare_sdf_atlas_gpu` reallocates fresh (zeroed) brick atlas PAGES (the texel
/// pages otherwise persist in VRAM, so a reused tile could show the previous scene's geometry).
/// Extracted to the render world.
#[derive(Resource, Clone, Copy, Default, bevy::render::extract_resource::ExtractResource)]
pub struct ProbeReset(pub u32);

/// On a [`SceneSwitched`], EVICT all per-scene SDF state so the new scene starts from a clean slate:
///  - DDGI probes: bump [`ProbeReset`] (→ the render world zeroes the irradiance buffer), reset the
///    settle counter + wake set;
///  - chunk/atlas data: [`SdfAtlas::reset`](atlas::SdfAtlas::reset) clears bricks/tiles/chunk tables
///    (and thus the probe slot allocator) + forces a full rebuild;
///  - bake scheduler: [`BakeScheduler::reset`](bake_scheduler::BakeScheduler::reset) drops queued/
///    in-flight work so the window re-bakes from scratch.
///
/// Central — fires for both editor tab swaps and in-game scene transitions (both routed through
/// `scene_manager::SceneSwitched`). Not state-gated: a switch fires as the state leaves the editor, so
/// this must run regardless of the current `AppScene`.
fn evict_on_scene_switch(
    mut ev: MessageReader<crate::scene_manager::SceneSwitched>,
    mut settle: ResMut<GiSettle>,
    mut wake: ResMut<ProbeWake>,
    mut wake_set: ResMut<ProbeWakeSet>,
    mut reset: ResMut<ProbeReset>,
    mut atlas: ResMut<atlas::SdfAtlas>,
    mut sched: ResMut<bake_scheduler::BakeScheduler>,
) {
    let n = ev.read().count();
    if n == 0 {
        return;
    }
    // Probe state.
    settle.frames_unchanged = 0;
    wake.frames.clear();
    wake_set.slots.clear();
    reset.0 = reset.0.wrapping_add(1);
    // Chunk/atlas + scheduler state (geometry starts clean too — no stale bricks or queued bakes).
    let bricks_before = atlas.bricks.len();
    atlas.reset();
    sched.reset();
    info!(
        "SDF scene switch: evicted {bricks_before} bricks + probes/atlas/scheduler (reset #{}, {n} event(s))",
        reset.0
    );
}

// --- Selection ---

/// The currently-selected SDF volume. Click-picking sets `entity`; the transform
/// manipulator (transform-gizmo-bevy) is attached to this entity via `GizmoTarget`.
#[derive(Resource, Default)]
pub struct SdfSelection {
    pub entity: Option<Entity>,
}

// --- Grid Config ---

/// Number of LOD levels the clipmap generates by default. Level 0 is the base
/// resolution; each coarser level doubles `voxel_size` (and so covers 2× the linear
/// extent / 8× the volume) of the one below it.
pub const DEFAULT_LOD_COUNT: u32 = 8;
/// Bricks per axis in each LOD ring window centred on the camera. The ring at level
/// `L` covers `ring_bricks · cell_stride · voxel_size · 2^L` world units per axis, so
/// the same count reaches twice as far each coarser level (the clipmap nesting). Must be
/// a multiple of [`chunk::CHUNK_BRICKS`] (= 4; the ring is enumerated in whole chunks).
/// 256 = 4·64: each LOD window spans twice as many bricks per axis as before, so every level
/// reaches 2x further at the SAME voxel resolution — distant geometry is served a finer LOD
/// (eases the far-LOD shrink), at the cost of a larger resident shell. The sparse cull still
/// keeps only non-empty bricks, so resident bricks grow ~with surface AREA (≈4×), but the dense
/// per-LOD chunk directory grows ~with VOLUME (≈8×). Must be a multiple of `CHUNK_BRICKS` (= 4).
pub const DEFAULT_RING_BRICKS: u32 = 256;
/// Default ring-recenter hysteresis, in whole chunks (see
/// [`SdfGridConfig::recenter_snap_chunks`]). With `CHUNK_BRICKS = 4` and a 256-brick ring
/// (64 chunks/axis), snapping to 2 chunks means the window recenters every ~5.6 m at LOD
/// 0 instead of every brick crossing, while still keeping the camera 30+ chunks from any
/// window edge.
pub const DEFAULT_RECENTER_SNAP_CHUNKS: i32 = 2;
/// How many COARSER LOD levels each region keeps resident beyond its native (finest-covering) LOD —
/// the `+N` in "hold `{native .. native+N}`". `1` (the default) holds the native LOD plus one coarser
/// fallback (the cone-LOD floor / secondary-ray coarsening read it, and it gives a hole-free streaming
/// handoff), and drops LOD `native+2..` which the renderer never samples there — so a near surface
/// triggers ~2 LOD bakes instead of the full ~8-deep stack. Larger keeps more of the stack resident
/// (more redundant bake work); `0` would keep only the native level but loses the fallback the LOD
/// floor relies on. NOT a shader uniform — it only changes which bricks are resident.
pub const DEFAULT_OVERLAP_DEPTH: u32 = 1;
/// Frustum bake-PRIORITY margin in world units: chunks within this slack of the view frustum still
/// rank as "in view" (so they bake a touch earlier), smoothing pop-in when the camera turns. Priority
/// only — it never changes residency (off-screen geometry stays resident for shadows/GI).
pub const DEFAULT_FRUSTUM_PRIORITY_MARGIN: f32 = 4.0;

#[derive(Resource, Clone)]
pub struct SdfGridConfig {
    pub grid_size: u32,
    pub brick_size: u32,
    /// Base (level-0) voxel size in world units. Level `L` uses `voxel_size · 2^L`.
    pub voxel_size: f32,
    /// How many LOD levels the clipmap bakes (level `0..lod_count`).
    pub lod_count: u32,
    /// Bricks per axis in each LOD ring window centred on the camera.
    pub ring_bricks: u32,
    /// Hysteresis: the ring window only recenters when the camera crosses this many
    /// whole chunks, so the per-LOD origin snaps to a coarse `recenter_snap_chunks`
    /// lattice instead of moving every brick crossing (~0.7 m at LOD 0). `1` = recenter
    /// on every chunk crossing (no hysteresis). Must stay well below
    /// `ring_bricks / CHUNK_BRICKS` so the camera never leaves its own window.
    pub recenter_snap_chunks: i32,
    /// Coarser LOD levels kept resident beyond each region's native LOD — `{native .. native+N}`
    /// residency (the hollow-shell clipmap). See [`DEFAULT_OVERLAP_DEPTH`].
    pub overlap_depth: u32,
    /// World-space slack for the in-frustum bake-priority test. See [`DEFAULT_FRUSTUM_PRIORITY_MARGIN`].
    pub frustum_priority_margin: f32,
}

impl Default for SdfGridConfig {
    fn default() -> Self {
        Self {
            grid_size: 1024,
            brick_size: 8,
            voxel_size: 0.1,
            lod_count: DEFAULT_LOD_COUNT,
            ring_bricks: DEFAULT_RING_BRICKS,
            recenter_snap_chunks: DEFAULT_RECENTER_SNAP_CHUNKS,
            overlap_depth: DEFAULT_OVERLAP_DEPTH,
            frustum_priority_margin: DEFAULT_FRUSTUM_PRIORITY_MARGIN,
        }
    }
}

impl SdfGridConfig {
    /// Spatial stride between bricks, in voxels. A brick stores `brick_size`
    /// samples (8) but covers `brick_size - 1` cells (7); adjacent bricks share
    /// (duplicate) the boundary sample plane so trilinear interpolation never
    /// needs to read a neighbouring brick. This "apron" is what the paper's
    /// Sparse Brick Set uses to avoid cracks at brick seams.
    pub fn cell_stride(&self) -> i32 {
        (self.brick_size - 1) as i32
    }

    /// Ring chunks per axis: `R = ring_bricks / CHUNK_BRICKS`. The edge of each per-LOD toroidal
    /// directory window and the SINGLE source for this derivation (CPU mirror of `ring_chunks() /
    /// CHUNK_BRICKS` in `bindings.wgsl`). `LiveChunkTables`/`ChunkTables` cache it and `dir_index`
    /// resolves against it, so every site MUST agree — route through here, never recompute ad hoc.
    pub fn ring_chunks_per_axis(&self) -> i32 {
        self.ring_bricks as i32 / chunk::CHUNK_BRICKS
    }

    /// Half the ring window in chunks (`R / 2`) — the camera-centred window's reach from its origin.
    pub fn ring_half_chunks(&self) -> i32 {
        self.ring_chunks_per_axis() / 2
    }

    /// Total per-LOD toroidal directory length: `R³ × lod_count` fixed `ChunkLookup` slots.
    pub fn directory_len(&self) -> usize {
        let r = self.ring_chunks_per_axis() as usize;
        r * r * r * self.lod_count as usize
    }

    pub fn world_extent(&self) -> f32 {
        self.grid_size as f32 * self.voxel_size
    }
    pub fn world_origin(&self) -> Vec3 {
        Vec3::splat(-self.world_extent() * 0.5)
    }

    /// Voxel size (world units) at LOD level `lod`: `base · 2^lod`.
    pub fn voxel_size_at(&self, lod: u32) -> f32 {
        self.voxel_size * (1u32 << lod) as f32
    }

    /// World-space edge length of one brick at LOD `lod` (`cell_stride · voxel_size`).
    pub fn brick_world_size(&self, lod: u32) -> f32 {
        self.cell_stride() as f32 * self.voxel_size_at(lod)
    }

    /// Convert world position to brick origin (grid-relative voxel coords,
    /// snapped down to the brick stride). Single-resolution (level-0, centred grid);
    /// kept for the non-LOD bake/test paths. LOD bakes use [`Self::world_to_brick_lod`].
    pub fn world_to_brick(&self, world_pos: Vec3) -> IVec3 {
        let s = self.cell_stride();
        let relative = world_pos - self.world_origin();
        let vox_x = (relative.x / self.voxel_size) as i32;
        let vox_y = (relative.y / self.voxel_size) as i32;
        let vox_z = (relative.z / self.voxel_size) as i32;
        IVec3::new((vox_x / s) * s, (vox_y / s) * s, (vox_z / s) * s)
    }

    /// Brick origin (stride-aligned voxel coords at LOD `lod`) containing `world_pos`.
    /// Each LOD lattice is anchored at world 0 (not the centred grid origin), so coords
    /// are signed and a ring can sit anywhere around the camera. `div_euclid` floors
    /// toward negative infinity so the lattice is continuous across the origin.
    pub fn world_to_brick_lod(&self, world_pos: Vec3, lod: u32) -> IVec3 {
        let s = self.cell_stride();
        let vs = self.voxel_size_at(lod);
        let vox = IVec3::new(
            (world_pos.x / vs).floor() as i32,
            (world_pos.y / vs).floor() as i32,
            (world_pos.z / vs).floor() as i32,
        );
        IVec3::new(
            vox.x.div_euclid(s) * s,
            vox.y.div_euclid(s) * s,
            vox.z.div_euclid(s) * s,
        )
    }

    /// World-space minimum corner of the brick at LOD `lod` with origin coord `coord`.
    pub fn brick_min_world(&self, coord: IVec3, lod: u32) -> Vec3 {
        let vs = self.voxel_size_at(lod);
        Vec3::new(coord.x as f32, coord.y as f32, coord.z as f32) * vs
    }

    /// The ring window's corner brick coord at LOD `lod` for a camera at `camera_pos`:
    /// the camera's brick minus half the ring on each axis, so the ring is centred on
    /// the camera. Coords are multiples of `cell_stride`.
    pub fn ring_origin(&self, camera_pos: Vec3, lod: u32) -> IVec3 {
        let s = self.cell_stride();
        let center = self.world_to_brick_lod(camera_pos, lod);
        let half = (self.ring_bricks / 2) as i32 * s;
        center - IVec3::splat(half)
    }

    // Chunk addressing (absolute keys, sparse occupancy) lives in `super::chunk`.
}

// --- Plugin ---

pub struct SdfScenePlugin;

impl Plugin for SdfScenePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SdfGridConfig>()
            .init_resource::<SdfSelection>()
            .init_resource::<OrbitFocus>()
            .init_resource::<SdfOrbitCamera>()
            .init_resource::<SdfCameraMode>()
            .init_resource::<edits::MaterialRegistry>()
            .init_resource::<atlas::SdfAtlas>()
            .init_resource::<bake_scheduler::PrevEditAabbs>()
            .init_resource::<bake_scheduler::BakeScheduler>()
            .init_resource::<bake_scheduler::PendingGpuBakes>()
            .init_resource::<LodRingsVisible>()
            .init_resource::<bvh::Bvh>()
            .init_resource::<SdfRenderEnabled>()
            .init_resource::<SdfRaymarchParams>()
            .init_resource::<DdgiParams>()
            .init_resource::<GiSettle>()
            .init_resource::<ProbeWake>()
            .init_resource::<ProbeWakeSet>()
            .init_resource::<ProbeRelevanceSet>()
            .init_resource::<ProbeReset>()
            // `evict_on_scene_switch` reads this message; register it here too (idempotent) so the SDF
            // plugin is self-sufficient and doesn't depend on `SceneManagerPlugin` being added first.
            .add_message::<crate::scene_manager::SceneSwitched>()
            .init_resource::<WireframeBoundsVisible>()
            .init_resource::<GizmoVisibility>()
            .init_resource::<BakedBrickDebug>()
            .init_resource::<RayStepCapture>()
            .init_resource::<ViewportInputAllowed>()
            .init_resource::<gizmo::GizmoState>()
            .register_type::<SdfVolume>()
            .register_type::<SdfCamera>()
            .register_type::<SdfPrimitive>()
            .register_type::<SdfOp>()
            .register_type::<SdfOrder>()
            .register_type::<SdfMaterial>()
            .register_type::<edits::SdfMaterialSource>()
            .register_type::<edits::MaterialFields>()
            .register_type::<CsgKind>()
            .register_type::<SdfRaymarchParams>()
            .register_type::<DdgiParams>()
            .register_type::<stress::TowerSpawner>()
            // Spawn the scene. Material ids come from the demand-driven asset table
            // (loaded MaterialAssets get stable registry ids); the compile step in
            // `assets::compile` fills the registry once assets resolve, and the GPU
            // table re-uploads via change detection.
            // The viewport camera persists across scene-state transitions (editor infra),
            // spawned once at startup and activated only while in the SDF editor.
            .add_systems(Startup, editor_camera::spawn_editor_camera)
            .add_systems(Update, editor_camera::sync_editor_camera_active)
            .add_systems(
                OnEnter(AppScene::SdfEditor),
                (setup_sdf_scene, load_default_gallery).chain(),
            )
            // Camera control: skipped when the pointer is over a dock panel (editor
            // sets ViewportInputAllowed). Non-editor build leaves it true.
            .add_systems(
                Update,
                (
                    editor_camera::orbit_camera.run_if(|m: Res<SdfCameraMode>| !m.fps),
                    editor_camera::fps_camera.run_if(|m: Res<SdfCameraMode>| m.fps),
                )
                    .run_if(in_state(AppScene::SdfEditor))
                    .run_if(|allowed: Res<ViewportInputAllowed>| allowed.0),
            )
            // Focus easing runs even while the pointer is over a dock panel, so a
            // Hierarchy double-click animates the camera without re-entering the
            // viewport. NOT gated on ViewportInputAllowed (unlike orbit_camera).
            .add_systems(
                Update,
                editor_camera::ease_orbit_focus
                    .run_if(in_state(AppScene::SdfEditor))
                    .run_if(|m: Res<SdfCameraMode>| !m.fps),
            )
            // Gizmo interaction THEN click-selection, both in `Last`, chained so the
            // gizmo claims a handle click before `sdf_picking` would reselect the
            // volume underneath (`sdf_picking` bails when `GizmoState.claimed_click`).
            .add_systems(
                Last,
                (gizmo::gizmo_update, sdf_picking, focus_on_double_click)
                    .chain()
                    .run_if(in_state(AppScene::SdfEditor))
                    .run_if(|allowed: Res<ViewportInputAllowed>| allowed.0),
            )
            // Ungated: end any active gizmo drag on mouse release even when the pointer
            // is over a dock panel, so a stale drag never carries into the next click.
            .add_systems(
                Last,
                gizmo::clear_gizmo_drag_on_release.run_if(in_state(AppScene::SdfEditor)),
            )
            // Bake/upload/render-toggle always run in the editor scene — property
            // edits in the inspector (and gizmo drags) must still re-bake. The GPU bake is
            // the only path: `schedule_bakes` does topology (edit detection + camera
            // recenter) and emits GPU compute jobs.
            // Rebuild the bake-time height cache when materials change, BEFORE the baker, so a
            // displacement edit triggers a rebake the same frame.
            // Expand any loaded TowerSpawner node into its tower-field volumes (the stress scene).
            // Before the baker so the spawned volumes bake the same frame they appear.
            // Clear last frame's incremental chunk-table delta record at the START of the frame —
            // AFTER the render world extracted it (extract runs at the end of the prior frame) and
            // BEFORE `schedule_bakes` accumulates this frame's. See `clear_chunk_table_dirty`.
            .add_systems(
                First,
                clear_chunk_table_dirty.run_if(in_state(AppScene::SdfEditor)),
            )
            .add_systems(
                Update,
                stress::expand_tower_spawners
                    .run_if(in_state(AppScene::SdfEditor))
                    .before(bake_scheduler::schedule_bakes),
            )
            .add_systems(
                Update,
                update_height_field
                    .run_if(in_state(AppScene::SdfEditor))
                    .before(bake_scheduler::schedule_bakes),
            )
            .add_systems(
                Update,
                bake_scheduler::schedule_bakes.run_if(in_state(AppScene::SdfEditor)),
            )
            .add_systems(
                Update,
                refresh_probe_lod
                    .run_if(in_state(AppScene::SdfEditor))
                    .after(bake_scheduler::schedule_bakes),
            )
            .add_systems(Update, track_gi_settle.after(refresh_probe_lod))
            .add_systems(Update, update_probe_wake.after(track_gi_settle))
            .add_systems(Update, update_probe_relevance.after(track_gi_settle))
            // Ungated: a scene switch fires as the state leaves the editor, so probe eviction must run
            // regardless of the current `AppScene`.
            .add_systems(Update, evict_on_scene_switch)
            .add_systems(
                Update,
                toggle_sdf_render.run_if(in_state(AppScene::SdfEditor)),
            );

        // Overlay gizmos (ground grid + bounds) need GizmoPlugin (Assets<GizmoAsset>).
        // Present in the real app (DefaultPlugins) but not in MinimalPlugins test
        // harnesses, so register the group + drawing only when present.
        if app.world().is_resource_added::<Assets<GizmoAsset>>()
            || app.world().get_resource::<Assets<GizmoAsset>>().is_some()
        {
            // `GizmoRenderPlugin` (the filled-overlay renderer consumed by `draw_gizmo`) is added
            // explicitly in `main.rs`; here we only wire the gizmo groups, which need `GizmoPlugin`
            // (`Assets<GizmoAsset>`) — present under DefaultPlugins, absent in MinimalPlugins tests.
            app.init_gizmo_group::<SdfOverlayGizmos>()
                .init_gizmo_group::<SdfGridGizmos>()
                .init_gizmo_group::<SdfNodeGizmos>()
                .add_systems(OnEnter(AppScene::SdfEditor), overlays::configure_overlay_gizmos)
                .add_systems(
                    Update,
                    (overlays::draw_ground_grid, gizmo::draw_gizmo)
                        .run_if(in_state(AppScene::SdfEditor)),
                )
                // LOD ring overlay: only while the toggle is on (LodRingsVisible, F8),
                // so it doesn't clutter the normal view.
                .add_systems(
                    Update,
                    overlays::draw_lod_rings
                        .run_if(in_state(AppScene::SdfEditor))
                        .run_if(|v: Res<LodRingsVisible>| v.0),
                );

            // Per-node-type gizmos (light glyphs, point-light ring + radius drag, axes)
            // own their draw/pick/interaction in `node_gizmos`.
            node_gizmos::register(app);
        }

        #[cfg(feature = "editor")]
        app.add_plugins(debug::SdfDebugPlugin);
    }
}

// --- Scene Setup ---

fn setup_sdf_scene(mut asset_table: ResMut<crate::assets::MaterialAssetTable>) {
    asset_table.ensure_fallback();

    // Materials are no longer hardcoded here: each volume in the loaded scene carries an
    // `SdfMaterialSource` (a file path and/or inline overrides), and `resolve_materials`
    // loads + derives the GPU registry dynamically from whatever the scene contains.
    //
    // The viewport camera is EDITOR infrastructure (see `spawn_editor_camera`), not scene
    // content — it persists across scene loads/switches and is never serialized. The gallery
    // geometry + light come from `assets/scenes/gallery.scene` via `load_default_gallery`.
    //
    // Initial bake happens on the first `schedule_bakes` tick (atlas starts dirty), once the
    // loaded edit entities exist and the BVH can be built from them.
}

/// Path to the editor's default scene: the **mesh-bake test scene** (`mesh_test.rs`) — a small,
/// legible CSG set (sharp cube / sphere / smooth blend / subtraction) for evaluating Surface Nets
/// meshing during the SDF→mesh migration. The PBR gallery (`assets/scenes/gallery.scene`), the
/// Cornell GI box (`assets/scenes/cornell.scene`), and the stress tower-field
/// (`assets/scenes/stress.scene`) remain loadable via the scene browser.
pub const DEFAULT_SCENE_PATH: &str = "assets/scenes/mesh_test.scene";

/// Load the default scene into the world on editor enter. Exclusive (scene load
/// needs `&mut World` + the type registry). Runs after `setup_sdf_scene` so the materials
/// it registers exist before the volumes that reference them appear — though the load only
/// needs the registry, since `registry_id`s are baked into the file.
fn load_default_gallery(world: &mut World) {
    let registry = world.resource::<AppTypeRegistry>().clone();
    // Profiling/headless-capture aid: `ADVENTURE_STARTUP_SCENE=<path>` (project-root-relative) loads
    // that scene instead of the default, so a Nsight capture can target a specific scene (e.g.
    // `assets/scenes/cornell8.scene`) without any editor interaction. Mirrors `ADVENTURE_EXIT_AFTER_FRAMES`.
    let startup = std::env::var("ADVENTURE_STARTUP_SCENE").ok();
    let path_str = startup.as_deref().unwrap_or(DEFAULT_SCENE_PATH);
    let path = std::path::Path::new(path_str);
    match crate::soul_scene::load_scene(world, path, &registry.read()) {
        Ok(roots) => info!("loaded scene '{path_str}' ({} roots)", roots.len()),
        Err(e) => error!("failed to load scene '{path_str}': {e}"),
    }
    // Restore the editor camera saved with the scene (if any), so launching frames the
    // gallery the way it was last saved.
    if let Some(cam) = world.resource::<crate::soul_scene::LoadedEditorCamera>().0 {
        {
            let mut orbit = world.resource_mut::<SdfOrbitCamera>();
            orbit.target = Vec3::from_array(cam.target);
            orbit.distance = cam.distance;
            orbit.yaw = cam.yaw;
            orbit.pitch = cam.pitch;
        }
        sync_orbit_camera_transform(world);
    }
}

// --- Picking ---

/// A volume entity paired with its resolved edit + world AABB, sorted by `SdfOrder`
/// (ties by entity index) so CSG evaluation order is deterministic. Each edit's
/// material id is its `SdfMaterial.registry_id` — a global id into the material
/// registry, independent of spawn/sort order.
pub struct GatheredEdit {
    pub entity: Entity,
    pub edit: edits::ResolvedEdit,
    pub aabb: bevy::math::bounding::Aabb3d,
}

/// Query data for reading an SDF volume edit's full definition. Aliased so the same
/// (6-field) query reads identically across the bake, picking, and debug systems
/// without tripping the type-complexity lint.
pub type VolumeQueryData = (
    Entity,
    // World transform, so a volume parented under another node inherits its parent's
    // motion (Bevy propagates `Transform` → `GlobalTransform`). Baking/picking operate
    // in world space, so this is the value they need.
    &'static GlobalTransform,
    &'static SdfPrimitive,
    &'static SdfOp,
    &'static SdfOrder,
    &'static SdfMaterial,
);

/// Collect all SDF volume edits from the world, sorted by `SdfOrder` (ties broken by
/// entity index for determinism). The material id comes from each edit's
/// `SdfMaterial` registry reference.
pub fn gather_sorted_edits(volumes: &Query<VolumeQueryData, With<SdfVolume>>) -> Vec<GatheredEdit> {
    let mut rows: Vec<(
        SdfOrder,
        Entity,
        Transform,
        SdfPrimitive,
        SdfOp,
        SdfMaterial,
    )> = volumes
        .iter()
        .map(|(e, t, p, op, order, m)| (*order, e, t.compute_transform(), p.clone(), *op, *m))
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.index().cmp(&b.1.index())));

    rows.into_iter()
        .map(|(_, entity, transform, prim, op, material)| {
            let aabb = edits::edit_world_aabb(&prim, &transform, op.smoothing);
            GatheredEdit {
                entity,
                edit: edits::ResolvedEdit::new(prim, transform, op, material.registry_id as u16),
                aabb,
            }
        })
        .collect()
}

/// Left-click selects the SDF volume under the cursor (CPU raymarch pick). Runs
/// after `gizmo_update` in `Last`; if the gizmo claimed the click (a handle was
/// grabbed), it bails so grabbing a handle doesn't reselect the volume underneath.
/// Query filter for non-SDF spatial nodes pickable via their gizmo bounds (lights/empties).
type GizmoNodeFilter = (Without<SdfVolume>, Without<SdfCamera>);

#[allow(clippy::too_many_arguments)]
fn sdf_picking(
    mouse: Res<ButtonInput<MouseButton>>,
    mut selection: ResMut<SdfSelection>,
    gizmo_state: Res<gizmo::GizmoState>,
    cameras: Query<(&Camera, &GlobalTransform, &Transform), With<SdfCamera>>,
    windows: Query<&Window>,
    volumes: Query<VolumeQueryData, With<SdfVolume>>,
    // Non-SDF spatial nodes (lights, empties) have no raymarchable geometry, so they're
    // picked by ray-testing the oriented bounding box of their drawn editor gizmo.
    gizmo_nodes: Query<(Entity, &GlobalTransform, &crate::node::EditorGizmo), GizmoNodeFilter>,
    // Point lights are also pickable by clicking their drawn range sphere (a large target).
    point_lights: Query<&PointLight>,
    bvh: Res<bvh::Bvh>,
) {
    let _span = crate::instrument::span("picking");
    if !mouse.just_pressed(MouseButton::Left) || gizmo_state.claimed_click {
        return;
    }

    let Ok(window) = windows.single() else {
        return;
    };
    let Some(mouse_pos) = window.cursor_position() else {
        return;
    };
    let Ok((camera, _cam_global, cam_transform)) = cameras.single() else {
        return;
    };
    let Some(ray) = picking::mouse_to_ray(camera, cam_transform, window, mouse_pos) else {
        return;
    };

    // 1. Raymarch the SDF volumes (the geometric pick), keeping the hit depth `t` so a
    //    node gizmo in front of the surface can win the click.
    let gathered = gather_sorted_edits(&volumes);
    let sdf_hit = picking::pick_entity(&bvh, &ray, &gathered);

    // 2. Ray-test each node gizmo's oriented bounding box (matching the drawn glyph),
    //    keeping the nearest entry distance — directly comparable to the SDF hit's `t`.
    let mut best_node: Option<(f32, Entity)> = None; // (ray_depth, entity)
    let consider = |t: f32, e: Entity, best: &mut Option<(f32, Entity)>| {
        if best.is_none_or(|(bt, _)| t < bt) {
            *best = Some((t, e));
        }
    };
    for (entity, xf, gizmo) in &gizmo_nodes {
        let (center, half) = node_gizmos::pick_bounds(gizmo);
        let obb = picking::Obb::from_local(center, half, xf);
        if let Some(t) = obb.ray_hit(&ray) {
            consider(t, entity, &mut best_node);
        }
        // A point light is also pickable by clicking its drawn range sphere (its two great
        // circles), a much larger target than the central bulb. Tolerance scales with
        // distance so the line stays ~8px thick on screen.
        if let Ok(light) = point_lights.get(entity) {
            let origin = xf.translation();
            let tol = (8.0 * (origin - cam_transform.translation).length()
                / camera.clip_from_view().y_axis.y)
                / window.height();
            for normal in node_gizmos::draw::SPHERE_CIRCLE_NORMALS {
                if let Some(t) = picking::ray_circle(&ray, origin, normal, light.range, tol) {
                    consider(t, entity, &mut best_node);
                }
            }
        }
    }

    // 3. Depth arbitration: a node in front of the SDF surface (or when the ray missed
    //    the SDF entirely) wins; otherwise the SDF hit wins. A click on truly empty space
    //    deselects (matching the prior raymarch-miss behaviour).
    selection.entity = match (sdf_hit, best_node) {
        (Some((sdf_e, sdf_t)), Some((node_t, node_e))) => {
            if node_t <= sdf_t {
                Some(node_e)
            } else {
                Some(sdf_e)
            }
        }
        (Some((sdf_e, _)), None) => Some(sdf_e),
        (None, Some((_, node_e))) => Some(node_e),
        (None, None) => None,
    };
}

/// CPU-pick the nearest SDF volume under a window-space cursor position, for callers
/// outside the `sdf_picking` system (e.g. the dock's material drag-drop handler, which runs
/// with `&mut World`). Returns the hit `SdfVolume` entity, or `None` on a miss. Reuses the
/// same ray + raymarch as `sdf_picking`; only SDF volumes are considered (gizmo nodes don't
/// accept a material).
pub fn pick_sdf_volume(world: &mut World, cursor: Vec2) -> Option<Entity> {
    let (camera, cam_transform) = {
        let mut q = world.query_filtered::<(&Camera, &Transform), With<SdfCamera>>();
        let (c, t) = q.single(world).ok()?;
        (c.clone(), *t)
    };
    let window = {
        let mut q = world.query::<&Window>();
        q.single(world).ok()?.clone()
    };
    let ray = picking::mouse_to_ray(&camera, &cam_transform, &window, cursor)?;

    let gathered = {
        let mut q = world.query_filtered::<VolumeQueryData, With<SdfVolume>>();
        gather_sorted_edits(&q.query(world))
    };
    let bvh = world.resource::<bvh::Bvh>();
    picking::pick_entity(bvh, &ray, &gathered).map(|(e, _t)| e)
}

/// Double-click (within 300ms) on the selected volume eases the orbit camera onto
/// it. Runs right after `sdf_picking` so `SdfSelection.entity` is already current;
/// kept separate so picking stays a single-responsibility pick. Orbit-mode only —
/// the FPS camera flies freely and ignores the orbit target.
fn focus_on_double_click(
    mouse: Res<ButtonInput<MouseButton>>,
    time: Res<Time>,
    mode: Res<SdfCameraMode>,
    selection: Res<SdfSelection>,
    mut focus: ResMut<OrbitFocus>,
    mut gizmo_state: ResMut<gizmo::GizmoState>,
    volumes: Query<&Transform, With<SdfVolume>>,
) {
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    let now = time.elapsed_secs();
    let double_click = now - focus.last_click < 0.3;
    focus.last_click = now;
    if !double_click {
        return;
    }

    // The first click of a double-click selects the object, which makes the transform
    // gizmo appear centered on it — so the SECOND click lands on the view-plane translate
    // handle and `gizmo_update` (earlier in this chain) just started a drag. Cancel it so
    // a double-click focuses without dragging the object.
    gizmo_state.drag = None;
    gizmo_state.claimed_click = false;

    if !mode.fps
        && let Some(entity) = selection.entity
        && let Ok(transform) = volumes.get(entity)
    {
        focus.target = Some(transform.translation);
    }
}

/// Clear the incremental chunk-table delta record (dirty rows/slots/sentinel) accumulated last
/// frame. Runs in `First`, AFTER the render world extracted the delta (extract runs at the end of
/// the previous frame) and BEFORE `schedule_bakes` accumulates this frame's changes — so each
/// frame's `dirty_*` sets carry exactly that frame's topology mutations. `schedule_bakes` only
/// APPENDS to these sets (never reads them), so a start-of-frame clear can't drop pending work.
fn clear_chunk_table_dirty(mut atlas: ResMut<atlas::SdfAtlas>) {
    atlas.live_chunks.clear_dirty();
}

/// Recompute the finest-resident DDGI probe FLAGS (`probe_base` 0/`u32::MAX`) when the chunk set
/// changed (a finer chunk entering/leaving flips its coarse parent's finest status, which is non-local,
/// so the recompute scans the resident set — O(resident)). Runs after `schedule_bakes`, before the
/// end-of-frame extract, so the directory delta carries the updated rows. Gated on
/// `topology_generation` so it scans the resident set only when the chunk set actually changed; idle
/// frames + texel-only re-bakes are free.
fn refresh_probe_lod(
    mut atlas: ResMut<atlas::SdfAtlas>,
    ddgi: Res<DdgiParams>,
    mut last_topo: Local<u64>,
    mut last_halve: Local<u32>,
) {
    // Recompute on a chunk-set change OR when the density-halving LOD knob changed (it re-decimates the
    // distant probe set). `u32::MAX` Local sentinel forces the first run.
    let halve = ddgi.probe_halve_lod;
    if atlas.topology_generation == *last_topo && *last_halve == halve {
        return;
    }
    *last_topo = atlas.topology_generation;
    *last_halve = halve;
    atlas.live_chunks.refresh_probe_bases(halve);
    // `refresh_probe_bases` marks the changed `probe_base` directory rows dirty; the render-world
    // extract is gated on `generation`. A topology change already bumped `generation` (set/clear_brick),
    // but bump again to be certain the refreshed finest-flag rows upload this frame — otherwise a stale
    // `u32::MAX` placeholder on the GPU makes `probe_slot_at` return -1 and GI renders as zero there.
    atlas.generation = atlas.generation.wrapping_add(1);
}

/// Rebuild the bake-time height cache when the material registry's displacement columns
/// (`tex_layers[3]`, `parallax_scale`) change, snapshot it into the scheduler for async tasks,
/// and force a rebake so the new relief is folded into the field. A no-op when nothing
/// displacement-relevant changed (fingerprint match) — colour-only edits don't rebake.
fn update_height_field(
    registry: Res<edits::MaterialRegistry>,
    library: Res<crate::assets::MaterialTextureLibrary>,
    mut sched: ResMut<bake_scheduler::BakeScheduler>,
    mut atlas: ResMut<atlas::SdfAtlas>,
    mut last_fingerprint: Local<u64>,
) {
    let _span = crate::instrument::span("height field");
    if let Some(rebuilt) = height::build(&registry, &library, *last_fingerprint) {
        *last_fingerprint = rebuilt.fingerprint;
        // The scheduler owns the canonical Arc snapshot (async bake tasks clone it; sync_bake
        // reads it via `height_field`). A registry change that alters displacement forces a
        // full rebake so the relief is folded into the field.
        sched.set_height(std::sync::Arc::new(rebuilt));
        atlas.rebake_all = true;
    }
}

fn toggle_sdf_render(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut enabled: ResMut<SdfRenderEnabled>,
    mut lod_rings: ResMut<LodRingsVisible>,
) {
    if keyboard.just_pressed(KeyCode::F1) {
        enabled.0 = !enabled.0;
        info!("SDF render pass: {}", if enabled.0 { "ON" } else { "OFF" });
    }
    if keyboard.just_pressed(KeyCode::F8) {
        lod_rings.0 = !lod_rings.0;
        info!("LOD ring overlay: {}", if lod_rings.0 { "ON" } else { "OFF" });
    }
}
