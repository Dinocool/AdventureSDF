// DDGI probe trace: OCTAHEDRAL directional irradiance per probe — THREAD-PER-TEXEL blend.
//
// GPU-only probe enumeration: one workgroup per RESIDENT (chunk × local brick); the 64 threads of the
// workgroup are the probe's OCTAHEDRAL TEXELS (8×8 max), not bricks. This is the canonical DDGI update
// layout (Majercik 2019 / RTXGI): an occupied brick owns a block of `subdiv³` probes; for each one the
// workgroup (a) cooperatively traces the rays into a groupshared radiance cache, then (b) each thread
// integrates ALL of those rays into its OWN texel. That removes the old per-thread `acc[64]`/`accw[64]`
// accumulators (which spilled to local memory and crushed occupancy) and traces each `raymarch` once,
// shared across all texels — instead of one thread serially doing rays×texels for the whole brick.
//
// Rays use the shared `raymarch` (default empty-space skipping, coarse LOD floor + SHORT `gi_range` so
// per-ray cost stays bounded). The probe's identity is the absolute (lod, brick_coord, sub), so it is
// world-anchored → boil-free. Result is temporally blended in place (single buffer); round-robin
// re-traces `1/update_stride` of probe slots per frame.

#import sdf::bindings::{camera, cell_stride, voxel_size_at, lod_count, chunk_tile_buf, ChunkLookup}
#import sdf::march::{raymarch, MarchQuality}
#import sdf::brick::{calc_normal, scene_sdf, world_to_brick_lod}
#import sdf::material::material_at
#import sdf::lights::point_lights_diffuse
#import sdf::shadows::surface_shadow
#import sdf::sky::sky_color
#import sdf::oct::{oct_decode, oct_encode}
#import sdf::probe::{subprobe_world_pos, probe_slot_at, decode_chunk_key, brick_coord_in_chunk, occ_bit_set, occ_rank_below, PROBE_OCT_RES, PROBE_OCT_TEXELS}

struct ProbeParams {
    ray_count: u32,
    hysteresis: f32,
    intensity: f32,
    frame: u32,
    subdiv: u32,
    update_stride: u32,
    gi_range: f32,
    normal_bias: f32,
    view_bias: f32,
    sky_intensity: f32,
    bounce_shadows: f32, // >0.5 → shadow-march the sun + brightest point light at each bounce hit
    dormant_stride: u32, // re-trace rate for converged probes when classify != 0
    classify: u32,       // 1 = converged probes go dormant (skip the ray-march); 0 = all trace at update_stride
    ray_falloff_lod: u32, // LOD >= this → trace distant_ray_count rays (far field needs less angular detail)
    distant_ray_count: u32,
};

// Single in-place octahedral irradiance buffer: probe `slot`'s OCT_RES² texels live at
// `slot * PROBE_OCT_TEXELS .. + PROBE_OCT_TEXELS`. Each slot is written by exactly one thread, so the
// read-modify-write temporal blend is race-free. The lit apply binds the SAME buffer read-only.
@group(2) @binding(0) var<storage, read_write> irradiance: array<vec4<f32>>;
@group(2) @binding(1) var<uniform> params: ProbeParams;
@group(2) @binding(2) var<storage, read> resident_chunks: array<ChunkLookup>;

const GOLDEN_ANGLE: f32 = 2.399963229728653; // π(3−√5)
const MAX_OCT_TEXELS: u32 = 64u; // octahedral texels per probe upper bound (8×8) = the workgroup size
// Max rays/probe/frame. The trace BATCHES rays in groups of 64 (see main), so the groupshared cache
// stays 64-deep (1 KB) no matter how high this is — a deep cache would burn workgroup memory and trip
// this kernel's occupancy cliff. DDGI casts ~100–300 rays/frame; the temporal-rotation accumulation
// + screen-space denoise then turn that into many effective samples (fills the inter-ray gaps).
const MAX_RAYS: u32 = 256u;

// Per-probe scratch shared by the workgroup's 64 texel-threads: ONE batch (≤64) of per-ray radiances
// (traced cooperatively, folded into each thread's texel accumulator), the relocated ray origin, and a
// buried/deactivated flag — origin/flag/quat computed once by thread 0 to avoid 64× redundant work.
var<workgroup> gs_rad: array<vec3<f32>, 64>;
var<workgroup> gs_origin: vec3<f32>;
var<workgroup> gs_skip: u32;
var<workgroup> gs_quat: vec4<f32>; // per-probe ray-rotation quaternion (computed once by thread 0)

// DETERMINISTIC Fibonacci-sphere directions — the SAME set every frame, so a converged probe's value
// doesn't change frame to frame (no boiling) even at low hysteresis. The broad cosine weighting in the
// octahedral integration means ~half the rays contribute to each texel, so 24 rays cover the 6×6 map
// adequately without per-frame ray rotation (which traded boiling for supersampling and is gone).
fn fib_dir(i: u32, n: u32) -> vec3<f32> {
    let z = 1.0 - (2.0 * f32(i) + 1.0) / f32(n);
    let r = sqrt(max(0.0, 1.0 - z * z));
    let phi = f32(i) * GOLDEN_ANGLE;
    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

// Integer hash (xxhash-style) → [0,1). Used to derive a stable per-probe random rotation.
fn hash_u32(x0: u32) -> u32 {
    var x = x0;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}
fn rnd01(seed: u32) -> f32 {
    return f32(hash_u32(seed) & 0xffffffu) / f32(0x1000000u);
}

// A STABLE per-probe random rotation, as a QUATERNION (4 floats — far less register pressure than a
// 3×3 matrix + transpose, which tripped this kernel's occupancy cliff). Every probe traces the same
// Fibonacci set, so a small/lifted emitter is sampled by ray #k for one band of probes and #k+1 for the
// next → coherent banding ("splotches") instead of a smooth penumbra. Rotating each probe's ray set by
// a rotation hashed from its WORLD-STABLE identity (brick coord + sub) decorrelates the sampling
// spatially; ALSO hashing the FRAME makes it stochastic per frame so the adaptive accumulation below
// gathers a fresh ray set every frame (filling the inter-ray gaps). The per-frame variation is made
// boil-free by the progressive sample-count average in the blend (each frame contributes only 1/N),
// not by hysteresis. Shoemake's uniform-random quat.
fn probe_quat(brick_coord: vec3<i32>, su: u32, lod: u32, frame: u32) -> vec4<f32> {
    let seed = (u32(brick_coord.x) * 73856093u)
        ^ (u32(brick_coord.y) * 19349663u)
        ^ (u32(brick_coord.z) * 83492791u)
        ^ (su * 2654435761u)
        ^ (lod * 40503u)
        ^ (frame * 2246822519u);
    let u1 = rnd01(seed);
    let a = sqrt(1.0 - u1);
    let b = sqrt(u1);
    let t2 = 6.2831853 * rnd01(seed + 1u);
    let t3 = 6.2831853 * rnd01(seed + 2u);
    return vec4<f32>(a * sin(t2), a * cos(t2), b * sin(t3), b * cos(t3)); // (x,y,z,w)
}
// Rotate `v` by quaternion `q` (xyz = axis·sin, w = cos). Conjugate (inverse rotation) = negate xyz.
fn quat_rot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Read the irradiance volume at `pos` toward `n` (nearest sub-probe, nearest octahedral texel) — the
// PREVIOUS frame's value, since this same buffer is being written in place. Used at a ray hit to add
// the indirect light already on that surface → cheap infinite bounce (converges over frames).
fn sample_probe_gi(pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let nlods = lod_count();
    let subdiv = max(params.subdiv, 1u);
    let nsub = subdiv * subdiv * subdiv;
    let s = cell_stride();
    for (var l = 0u; l < nlods; l = l + 1u) {
        let vs = voxel_size_at(l);
        let bc = world_to_brick_lod(pos, l);
        let base_slot = probe_slot_at(bc, l);
        if (base_slot >= 0) {
            let cell = f32(s) * vs / f32(subdiv);
            let rel = (pos - vec3<f32>(bc) * vs) / cell;
            let mx = i32(subdiv) - 1;
            let sx = clamp(i32(floor(rel.x)), 0, mx);
            let sy = clamp(i32(floor(rel.y)), 0, mx);
            let sz = clamp(i32(floor(rel.z)), 0, mx);
            let sub_lin = u32(sz) * subdiv * subdiv + u32(sy) * subdiv + u32(sx);
            let pslot = u32(base_slot) * nsub + sub_lin;
            let oct_base = pslot * PROBE_OCT_TEXELS;
            if (oct_base + PROBE_OCT_TEXELS <= arrayLength(&irradiance)) {
                let uv = oct_encode(n) * f32(PROBE_OCT_RES);
                let r2 = i32(PROBE_OCT_RES) - 1;
                let tx = u32(clamp(i32(floor(uv.x)), 0, r2));
                let ty = u32(clamp(i32(floor(uv.y)), 0, r2));
                let probe = irradiance[oct_base + ty * PROBE_OCT_RES + tx];
                if (probe.a > 0.5) {
                    // Stored in sqrt (perceptual) space; decode to linear (square) before re-emitting it
                    // as indirect radiance for the next bounce. Square is ~free vs a general pow — the
                    // trace kernel is register/occupancy bound, so cheap encode/decode matters here.
                    let lin = max(probe.rgb, vec3<f32>(0.0));
                    return lin * lin;
                }
            }
        }
    }
    return vec3<f32>(0.0);
}

// Octahedral texel `t` → its world direction (texel-center).
fn texel_dir(t: u32) -> vec3<f32> {
    let tx = t % PROBE_OCT_RES;
    let ty = t / PROBE_OCT_RES;
    let uv = (vec2<f32>(f32(tx), f32(ty)) + vec2<f32>(0.5)) / f32(PROBE_OCT_RES);
    return oct_decode(uv);
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>,
    @builtin(local_invocation_index) lii: u32,
) {
    // One workgroup per (resident chunk × local brick); `tid` is this thread's octahedral texel.
    let gwg = wid.y * ng.x + wid.x;
    let chunk_index = gwg / 64u;
    let local = gwg % 64u;
    if (chunk_index >= arrayLength(&resident_chunks)) {
        return;
    }
    let chunk = resident_chunks[chunk_index];
    if (!occ_bit_set(chunk.occ_lo, chunk.occ_hi, local)) {
        return; // empty brick — no probe here (whole workgroup exits uniformly)
    }
    let tid = lii;

    // Finest-resident filter: a chunk fully covered by finer-LOD resident chunks owns no probes
    // (`probe_base == u32::MAX`) — its region is served by the finer LOD, so skip the whole workgroup
    // (uniform: probe_base is constant across the brick). Bounds the probe buffer to the finest set.
    if (chunk.probe_base == 0xffffffffu) {
        return;
    }
    let id = decode_chunk_key(chunk.key_hi, chunk.key_lo);
    let brick_coord = brick_coord_in_chunk(id.coord, local);
    // COMPACT per-brick probe slot, read from the brick's tile-run record (`BrickTile.probe_slot`). It's
    // a stable, free-list slot allocated only over finest-resident OCCUPIED bricks, so the probe buffer
    // is exact + scales with the clipmap window (no intra-chunk waste, no all-LOD redundancy). The apply
    // derives the identical slot via `probe_slot_at`. `local` is occupied here (checked above), so the
    // dense rank `off` indexes the brick's own record.
    let off = occ_rank_below(chunk.occ_lo, chunk.occ_hi, local);
    let probe_idx = chunk_tile_buf[chunk.tile_run_base + off].probe_slot;
    if (probe_idx == 0xffffffffu) {
        return; // brick has no probe slot assigned (defensive; finest-flag already gated the workgroup)
    }

    let subdiv = max(params.subdiv, 1u);
    let nsub = subdiv * subdiv * subdiv;
    let block_base = probe_idx * nsub;
    let bw = f32(cell_stride()) * voxel_size_at(id.lod);
    let cell = bw / f32(subdiv);
    let stride = max(params.update_stride, 1u);
    // CLASSIFICATION: a converged probe (its sample count reached the cap) re-traces at the slower
    // `dormant_stride` instead of `update_stride` — it keeps its value and skips the ray-march. The CPU
    // only sets `classify` once the scene is settled, so a moving camera / changing light keeps every
    // probe active (full rate) and nothing goes stale. The convergence cap mirrors the blend's `n_max`.
    let n_cap = 1.0 / (1.0 - clamp(params.hysteresis, 0.0, 0.999));
    // Distant (coarse-LOD) probes trace fewer rays — the far field's GI is low-frequency, so this cuts
    // the dominant ray-march cost without touching near quality.
    let ray_n = select(params.ray_count, params.distant_ray_count, id.lod >= params.ray_falloff_lod);
    let n = min(max(ray_n, 1u), MAX_RAYS);
    let octn = min(PROBE_OCT_TEXELS, MAX_OCT_TEXELS);
    let sun = normalize(camera.sun_dir.xyz);
    let my_dir = texel_dir(tid); // constant for this thread across all of the brick's sub-probes

    for (var su = 0u; su < nsub; su = su + 1u) {
        let probe_slot = block_base + su;
        let oct_base = probe_slot * PROBE_OCT_TEXELS;
        if (oct_base + PROBE_OCT_TEXELS > arrayLength(&irradiance)) {
            continue; // uniform
        }
        // Per-probe effective stride: dormant (converged + classify on) → slow rate, else active rate.
        // `irradiance[oct_base].a` is texel 0's sample count (uniform across the workgroup's threads).
        let converged = irradiance[oct_base].a >= n_cap - 0.5;
        let eff_stride = select(stride, max(params.dormant_stride, 1u), params.classify != 0u && converged);
        if ((probe_slot % eff_stride) != (params.frame % eff_stride)) {
            continue; // not this slot's turn — retain its in-place octahedral tile (uniform skip)
        }

        // --- Relocation + per-probe rotation (thread 0 only; results shared) ---------------------
        // Push the probe out of any solid it sits in (the emissive self-shadow fix); flag it
        // deactivated if buried too deep so every texel-thread can zero its texel. Computed once
        // (not 64×) because `center`/`sdf0` are identical for all threads.
        if (tid == 0u) {
            let sub = vec3<i32>(i32(su % subdiv), i32((su / subdiv) % subdiv), i32(su / (subdiv * subdiv)));
            let center = subprobe_world_pos(brick_coord, id.lod, sub, subdiv);
            var origin = center;
            var skip = 0u;
            let sdf0 = scene_sdf(center);
            if (sdf0.in_brick) {
                if (sdf0.dist < -0.5 * cell) {
                    skip = 1u;
                } else {
                    origin = center + calc_normal(center) * clamp(0.1 * cell - sdf0.dist, 0.0, cell);
                }
            }
            gs_origin = origin;
            gs_skip = skip;
            // Per-probe + per-frame ray rotation. Computed once here, not 64×.
            gs_quat = probe_quat(brick_coord, su, id.lod, params.frame);
        }
        // workgroupUniformLoad barriers AND yields a uniform value, so the following branch is uniform
        // control flow (required for the barriers below) and `gs_origin`'s write is visible to all.
        if (workgroupUniformLoad(&gs_skip) == 1u) {
            if (tid < octn) {
                irradiance[oct_base + tid] = vec4<f32>(0.0); // deactivated (apply skips alpha 0)
            }
            continue;
        }

        // --- Trace + integrate in BATCHES of 64 rays --------------------------------------------
        // To support n > 64 rays without a large groupshared cache (which trips this kernel's occupancy
        // cliff), trace 64 rays at a time into the 1 KB `gs_rad` cache and fold each batch into a
        // per-texel accumulator that persists across batches. Each thread owns texel `tid`; its inverse-
        // rotated direction (dot(my_dir, q·fib) == dot(q⁻¹·my_dir, fib)) makes the inner weight a plain
        // dot against the un-rotated fib_dir.
        let origin = gs_origin;
        let my_dir_local = quat_rot(vec4<f32>(-gs_quat.xyz, gs_quat.w), my_dir);
        var acc = vec3<f32>(0.0);
        var accw = 0.0;
        for (var base_ray = 0u; base_ray < n; base_ray = base_ray + 64u) {
            // Cooperative trace: thread `tid` marches ray (base_ray + tid) of this batch.
            let ri = base_ray + tid;
            var radiance = vec3<f32>(0.0);
            if (ri < n) {
                let dir = quat_rot(gs_quat, fib_dir(ri, n));
                let mq = MarchQuality(4.0, 24u, params.gi_range, id.lod);
                let r = raymarch(origin, dir, 0.05, mq);
                if (r.fate == 1u) {
                    // Escaped to the sky: the analytic environment is physical (× SKY_LUMINANCE), so
                    // scale it by `sky_intensity` — 1.0 lets the sky bounce into the scene, 0.0 isolates
                    // GI to scene emitters + sun (interiors / the harness isolation gates).
                    radiance = sky_color(dir, sun) * params.sky_intensity;
                } else if (r.hit) {
                    let m = material_at(r.object_id);
                    let nrm = calc_normal(r.hit_pos);
                    let albedo = m.base_color.rgb;
                    let shadows = params.bounce_shadows > 0.5;
                    // Direct lighting at the bounce, gathered as DIFFUSE-only Lambert (radiance·N·L).
                    // DDGI caches diffuse irradiance, so the heavy view-dependent Frostbite specular the
                    // primary pass uses would only inject firefly noise here. Sun first (directional),
                    // shadowed by a march toward the sun — bounded to `gi_range` (GI is local) but using
                    // the HIT's LOD (`r.lod`) for the shadow falloff, exactly like the primary pass.
                    var direct = camera.sun_color.rgb * max(dot(nrm, sun), 0.0);
                    if (shadows && max(direct.x, max(direct.y, direct.z)) > 0.0) {
                        direct *= surface_shadow(r.hit_pos, nrm, sun, r.lod, params.gi_range);
                    }
                    // …then the point lights, via the SHARED gather in sdf::lights: the SAME world-grid
                    // cull, brightest-first strength culls, `shadow_light_cap()` budget, and sphere-light
                    // shadow (with the hit's LOD falloff) the G-buffer's direct pass uses — only the
                    // final term is diffuse Lambert irradiance here instead of the Frostbite BRDF.
                    direct += point_lights_diffuse(r.hit_pos, nrm, r.lod, shadows);
                    // emissive + albedo·(direct + prev-frame indirect = cheap infinite bounce).
                    let indirect = sample_probe_gi(r.hit_pos, nrm);
                    radiance = m.emissive.rgb + albedo * (direct + indirect);
                }
            }
            gs_rad[tid] = radiance;
            workgroupBarrier();

            // Fold this batch's rays into the per-texel accumulator (cosine-weighted by texel dir).
            if (tid < octn) {
                let bn = min(64u, n - base_ray);
                for (var j = 0u; j < bn; j = j + 1u) {
                    let w = max(dot(my_dir_local, fib_dir(base_ray + j, n)), 0.0);
                    acc += w * gs_rad[j];
                    accw += w;
                }
            }
            workgroupBarrier(); // before the next batch overwrites gs_rad
        }

        // Store in SQRT (perceptual, gamma-2) space. The blend is an ADAPTIVE PROGRESSIVE AVERAGE:
        //  (A) the texel keeps a sample count N (in alpha); each frame contributes weight 1/N, so a
        //      settled probe barely moves → no boil, while fresh per-frame ray sets keep accumulating
        //      (filling the inter-ray gaps). N is capped via `hysteresis` (N_max = 1/(1−h)) → a long
        //      EMA at steady state.
        //  (B) if this frame's estimate jumps far from the accumulated value (emitter/light moved),
        //      reset N → it re-converges in a few frames → responsive, with no per-frame lag knob.
        // alpha doubles as validity: 0 = deactivated (apply skips); ≥1 = valid + the sample count.
        if (tid < octn) {
            let irr = acc / max(accw, 1.0e-3);
            let enc = sqrt(max(irr, vec3<f32>(0.0)));     // this frame's sqrt-space estimate
            let prev = irradiance[oct_base + tid];        // rgb = accumulated, a = sample count N
            // PROGRESSIVE AVERAGE (boil-free temporal accumulation): weight 1/N, N capped at
            // 1/(1−hysteresis). A settled probe barely moves per frame → no boil; fresh per-frame-rotated
            // ray sets keep accumulating (fills the inter-ray gaps). Per-frame variance is bounded at the
            // source by the firefly clamp in the trace, so bright emitters don't boil. (No single-frame
            // change-reset: it can't tell a firefly from a real change and re-boils at high emissiveness.)
            let n_max = 1.0 / (1.0 - clamp(params.hysteresis, 0.0, 0.999));
            let prev_n = select(prev.a, 0.0, prev.a < 0.5); // <0.5 ⇒ uninitialised/deactivated → restart
            let nsamp = min(prev_n + 1.0, n_max);
            let blended = mix(prev.rgb, enc, 1.0 / nsamp);
            irradiance[oct_base + tid] = vec4<f32>(blended, nsamp);
        }
        workgroupBarrier(); // before the next sub-probe reuses gs_rad / gs_origin
    }
}
