// GPU worldgen height-field library — the WGSL port of the CPU node-graph terrain SSOT.
//
// This is Stage 1a of the GPU-voxel-worldgen pivot (docs/GPU_VOXEL_WORLDGEN_PLAN.md): a faithful
// translation of the CPU height-field evaluation so a compute shader can voxelize bricks on the GPU.
// It mirrors, function-for-function:
//   - src/sdf_render/worldgen/graph/field.rs   → WgField + its dual-number ops (autodiff gradient)
//   - src/sdf_render/worldgen/noise.rs          → hash2 / value_noise_grad / fbm_height_grad (noise basis)
//   - src/sdf_render/worldgen/spline.rs         → wg_spline_eval (monotone cubic Hermite — Curve node)
//   - src/sdf_render/worldgen/layers/erosion.rs → wg_erode_with_grad (analytic-gradient erosion)
//
// A `NodeKind -> WGSL` codegen pass (src/sdf_render/worldgen/graph/wgsl_codegen.rs) emits the per-graph
// `wg_eval_graph(wx, wz, world_seed) -> WgField` that calls the per-op helpers below in topological
// order. This module is `#import`ed by that generated function.
//
// ## f64 vs f32 (THE divergence the CPU SSOT does NOT have)
// The CPU evaluates in f64 (bit-portable, shared-seed-multiplayer authoritative). WGSL has only f32, so
// the FLOATING parts (noise interpolation, fBm octave sum, spline, field arithmetic, erosion) compute in
// f32 and WILL differ from the CPU in the low mantissa bits. This is acceptable here: Stage 1a is the
// GPU voxelize CORRECTNESS port (matched to a pinned tolerance, not bit-for-bit) — the f64 CPU path
// remains the authoritative one. The INTEGER hash (hash2/fmix32) is ported with EXACT u32 wrapping
// semantics (WGSL u32 arithmetic wraps per spec, identical to Rust `wrapping_*`/`^`/`>>`), so the
// lattice ENTROPY matches the CPU bit-for-bit; only the float interpolation of that entropy diverges.
//
// ## Knobs stay knobs
// Erosion parameters are a uniform-shaped struct (WgErosionParams) passed as a function arg, never a
// const — the editor sliders drive them exactly as the CPU ErosionParams resource does.

#define_import_path worldgen::gpu

// =====================================================================================================
// WgField — the dual number flowing along graph edges: value + analytic world-XZ gradient.
// Mirror of `graph::field::Field { v, dx, dz }`. Every op is the SAME forward-mode-autodiff rule the
// CPU uses (e.g. `wg_mul` is the product rule, NOT scalar multiply). f32 (CPU is f64 — see header).
// =====================================================================================================

struct WgField {
    v: f32,
    dx: f32,
    dz: f32,
}

// --- sources (arity 0) ---

// Field::constant — spatially-constant field (gradient zero).
fn wg_const(v: f32) -> WgField {
    return WgField(v, 0.0, 0.0);
}

// Field::world_x — the world-X coordinate as a field: value wx, gradient (1, 0).
fn wg_world_x(wx: f32) -> WgField {
    return WgField(wx, 1.0, 0.0);
}

// Field::world_z — the world-Z coordinate as a field: value wz, gradient (0, 1).
fn wg_world_z(wz: f32) -> WgField {
    return WgField(wz, 0.0, 1.0);
}

// Field::new — a field from a value + its already-known analytic gradient (e.g. a noise sample).
fn wg_field(v: f32, dx: f32, dz: f32) -> WgField {
    return WgField(v, dx, dz);
}

// --- arithmetic (the dual-number ops) ---

// Field::add.
fn wg_add(a: WgField, b: WgField) -> WgField {
    return WgField(a.v + b.v, a.dx + b.dx, a.dz + b.dz);
}

// Field::sub.
fn wg_sub(a: WgField, b: WgField) -> WgField {
    return WgField(a.v - b.v, a.dx - b.dx, a.dz - b.dz);
}

// Field::neg.
fn wg_neg(a: WgField) -> WgField {
    return WgField(-a.v, -a.dx, -a.dz);
}

// Field::mul — product rule: (ab)' = a'b + ab'.
fn wg_mul(a: WgField, b: WgField) -> WgField {
    return WgField(
        a.v * b.v,
        a.dx * b.v + a.v * b.dx,
        a.dz * b.v + a.v * b.dz,
    );
}

// Field::scale — multiply by a spatial constant.
fn wg_scale(a: WgField, k: f32) -> WgField {
    return WgField(a.v * k, a.dx * k, a.dz * k);
}

// Field::offset — add a spatial constant.
fn wg_offset(a: WgField, k: f32) -> WgField {
    return WgField(a.v + k, a.dx, a.dz);
}

// Field::abs — gradient flips sign with the value (kink at v=0, measure-zero; ties → self, like CPU `< 0`).
fn wg_abs(a: WgField) -> WgField {
    if (a.v < 0.0) {
        return wg_neg(a);
    }
    return a;
}

// Field::min — smaller value, carrying ITS gradient. Ties → self (CPU: `if b.v < self.v { b } else { self }`).
fn wg_min(a: WgField, b: WgField) -> WgField {
    if (b.v < a.v) {
        return b;
    }
    return a;
}

// Field::max — larger value, carrying ITS gradient. Ties → self (CPU: `if b.v > self.v { b } else { self }`).
fn wg_max(a: WgField, b: WgField) -> WgField {
    if (b.v > a.v) {
        return b;
    }
    return a;
}

// Field::clamp — value clamped to [lo, hi]; gradient passes through inside, flat (zero) when saturated.
// Matches CPU branch order exactly (`< lo` then `> hi`).
fn wg_clamp_field(a: WgField, lo: f32, hi: f32) -> WgField {
    if (a.v < lo) {
        return wg_const(lo);
    }
    if (a.v > hi) {
        return wg_const(hi);
    }
    return a;
}

// Field::mix — linear interpolation self + (b - self)·t, t a field (varies in space). Full product/sum rule.
// CPU: `let d = b.sub(self); self.add(d.mul(t))`.
fn wg_mix(a: WgField, b: WgField, t: WgField) -> WgField {
    let d = wg_sub(b, a);
    return wg_add(a, wg_mul(d, t));
}

// Field::smoothstep — C¹ Hermite s = t²(3 − 2t), t = clamp((v − e0)/(e1 − e0), 0, 1). Saturated ends → flat.
// Mirrors the CPU branch structure (`raw <= 0` → 0, `raw >= 1` → 1) and chain rule (`ds_dv = ds_dt · inv`).
fn wg_smoothstep_field(a: WgField, edge0: f32, edge1: f32) -> WgField {
    let inv = 1.0 / (edge1 - edge0);
    let raw = (a.v - edge0) * inv;
    if (raw <= 0.0) {
        return wg_const(0.0);
    }
    if (raw >= 1.0) {
        return wg_const(1.0);
    }
    let t = raw;
    let s = t * t * (3.0 - 2.0 * t);
    let ds_dt = 6.0 * t * (1.0 - t);
    let ds_dv = ds_dt * inv;
    return WgField(s, ds_dv * a.dx, ds_dv * a.dz);
}

// NodeKind::Ridge — in + ridge·((amp_sum − 2|in|) − in). Gradient via the WgField ops (autodiff), exactly
// like the CPU: `let ridged = constant(amp_sum).sub(a.abs().scale(2.0)); a.add(ridged.sub(a).scale(ridge))`.
fn wg_ridge(a: WgField, ridge: f32, amp_sum: f32) -> WgField {
    let ridged = wg_sub(wg_const(amp_sum), wg_scale(wg_abs(a), 2.0));
    return wg_add(a, wg_scale(wg_sub(ridged, a), ridge));
}

// NodeKind::Curve — chain the spline's (y, dy) through the input field: Field::new(y, dy·a.dx, dy·a.dz).
// The spline control points (xs/ys, len) are codegen'd into the arrays passed here (see wgsl_codegen.rs).
fn wg_curve(a: WgField, xs: ptr<function, array<f32, 8>>, ys: ptr<function, array<f32, 8>>, len: u32) -> WgField {
    let yd = wg_spline_eval(xs, ys, len, a.v);
    return WgField(yd.x, yd.y * a.dx, yd.y * a.dz);
}

// =====================================================================================================
// Spline — monotone cubic Hermite, mirror of spline.rs `Spline::eval` → (y, dy/dx). Returns vec2(y, dy).
// Catmull-Rom tangents + Fritsch–Carlson monotonicity clamp, Hermite basis in Horner form. Flat-clamped
// outside [x0, x_{n-1}]. f32 (CPU f64). Control points arrive as fixed-size arrays (SPLINE_MAX_POINTS=8).
// =====================================================================================================

const WG_SPLINE_MAX_POINTS: u32 = 8u;

// spline.rs `clamp_to_secant`: clamp tangent m so it doesn't exceed 3·d in d's direction (0 if d flat).
fn wg_clamp_to_secant(m: f32, d: f32) -> f32 {
    if (d == 0.0) {
        return 0.0;
    }
    if (m * d < 0.0) {
        return 0.0;
    }
    let lim = 3.0 * d;
    if (d > 0.0) {
        return min(m, lim);
    }
    return max(m, lim);
}

// spline.rs `Spline::tangent` — Catmull-Rom tangent at knot i (one-sided at the ends) + monotonicity clamp.
fn wg_spline_tangent(xs: ptr<function, array<f32, 8>>, ys: ptr<function, array<f32, 8>>, n: u32, i: u32) -> f32 {
    var raw: f32;
    if (i == 0u) {
        raw = ((*ys)[1] - (*ys)[0]) / ((*xs)[1] - (*xs)[0]);
    } else if (i == n - 1u) {
        raw = ((*ys)[n - 1u] - (*ys)[n - 2u]) / ((*xs)[n - 1u] - (*xs)[n - 2u]);
    } else {
        raw = ((*ys)[i + 1u] - (*ys)[i - 1u]) / ((*xs)[i + 1u] - (*xs)[i - 1u]);
    }
    var m = raw;
    if (i > 0u) {
        let d = ((*ys)[i] - (*ys)[i - 1u]) / ((*xs)[i] - (*xs)[i - 1u]);
        m = wg_clamp_to_secant(m, d);
    }
    if (i + 1u < n) {
        let d = ((*ys)[i + 1u] - (*ys)[i]) / ((*xs)[i + 1u] - (*xs)[i]);
        m = wg_clamp_to_secant(m, d);
    }
    return m;
}

// spline.rs `Spline::eval` — returns (y, dy/dx). Flat-clamped outside the knot domain.
fn wg_spline_eval(xs: ptr<function, array<f32, 8>>, ys: ptr<function, array<f32, 8>>, len: u32, x: f32) -> vec2<f32> {
    let n = len;
    if (n == 1u || x <= (*xs)[0]) {
        return vec2<f32>((*ys)[0], 0.0);
    }
    if (x >= (*xs)[n - 1u]) {
        return vec2<f32>((*ys)[n - 1u], 0.0);
    }
    // Segment i with xs[i] <= x < xs[i+1] (n ≤ 8 → linear scan).
    var i = 0u;
    loop {
        if (!(i + 1u < n && x >= (*xs)[i + 1u])) {
            break;
        }
        i = i + 1u;
    }
    let h = (*xs)[i + 1u] - (*xs)[i];
    let mi = wg_spline_tangent(xs, ys, n, i);
    let mi1 = wg_spline_tangent(xs, ys, n, i + 1u);
    let t = (x - (*xs)[i]) / h;
    let t2 = t * t;
    let h00 = 2.0 * t2 * t - 3.0 * t2 + 1.0;
    let h10 = t2 * t - 2.0 * t2 + t;
    let h01 = -2.0 * t2 * t + 3.0 * t2;
    let h11 = t2 * t - t2;
    let h00d = 6.0 * t2 - 6.0 * t;
    let h10d = 3.0 * t2 - 4.0 * t + 1.0;
    let h01d = -6.0 * t2 + 6.0 * t;
    let h11d = 3.0 * t2 - 2.0 * t;
    let y0 = (*ys)[i];
    let y1 = (*ys)[i + 1u];
    let y = h00 * y0 + h10 * h * mi + h01 * y1 + h11 * h * mi1;
    let dy_dt = h00d * y0 + h10d * h * mi + h01d * y1 + h11d * h * mi1;
    return vec2<f32>(y, dy_dt / h);
}

// =====================================================================================================
// Noise basis — mirror of noise.rs. The integer hash is EXACT (u32 wrapping = bit-for-bit CPU). The
// float interpolation is f32 (CPU f64 — the documented divergence).
// =====================================================================================================

// noise.rs `fmix32` — Murmur3 finalizer. Pure u32 wrapping ops (WGSL u32 wraps per spec) ⇒ bit-exact.
fn wg_fmix32(h_in: u32) -> u32 {
    var h = h_in;
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    return h;
}

// noise.rs `hash2` — 2D integer-lattice hash → u32. Golden-ratio + distinct-stream multipliers, then
// fmix32. ix/iz are i32 lattice coords; `bitcast<u32>` mirrors the CPU `as u32` (a pure bit reinterpret).
fn wg_hash2(ix: i32, iz: i32, seed: u32) -> u32 {
    var h = seed;
    h = h + bitcast<u32>(ix) * 0x9E3779B1u;
    h = h + bitcast<u32>(iz) * 0x85EBCA77u;
    return wg_fmix32(h);
}

// noise.rs `value_at_lattice` — lattice value in [-1, 1) from the integer hash. The only int→float step
// is a divide by 2^31 (exact power of two on the CPU; f32 here — the magnitude fits f32 exactly enough,
// but the reinterpreted i32 may exceed f32's 24-bit mantissa ⇒ rounded to f32 — a documented divergence).
fn wg_value_at_lattice(ix: i32, iz: i32, seed: u32) -> f32 {
    let hi = bitcast<i32>(wg_hash2(ix, iz, seed));
    return f32(hi) * (1.0 / 2147483648.0);
}

// noise.rs `fade` — Perlin quintic fade 6t⁵ − 15t⁴ + 10t³, Horner form (basic ops).
fn wg_fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// noise.rs `fade_deriv` — 30t²(t−1)² = 30t²(t² − 2t + 1). Same association as the CPU (`30·t·t·…`).
fn wg_fade_deriv(t: f32) -> f32 {
    return 30.0 * t * t * (t * (t - 2.0) + 1.0);
}

// noise.rs `value_noise_grad` — one octave of bilinear value noise + analytic gradient at the (already
// frequency-scaled) coord (x, z). Returns vec3(value, ∂value/∂x, ∂value/∂z). Faded-bilinear blend + the
// product/chain-rule gradient through the fades — the SAME expression tree as the CPU.
fn wg_value_noise_grad(x: f32, z: f32, seed: u32) -> vec3<f32> {
    let xi = floor(x);
    let zi = floor(z);
    let ix = i32(xi);
    let iz = i32(zi);
    let fx = x - xi;
    let fz = z - zi;

    let v00 = wg_value_at_lattice(ix, iz, seed);
    let v10 = wg_value_at_lattice(ix + 1, iz, seed);
    let v01 = wg_value_at_lattice(ix, iz + 1, seed);
    let v11 = wg_value_at_lattice(ix + 1, iz + 1, seed);

    let u = wg_fade(fx);
    let v = wg_fade(fz);
    let du = wg_fade_deriv(fx);
    let dv = wg_fade_deriv(fz);

    let a = v00 + (v10 - v00) * u;
    let b = v01 + (v11 - v01) * u;
    let value = a + (b - a) * v;

    let da_dx = (v10 - v00) * du;
    let db_dx = (v11 - v01) * du;
    let dval_dx = da_dx + (db_dx - da_dx) * v;
    let dval_dz = (b - a) * dv;

    return vec3<f32>(value, dval_dx, dval_dz);
}

// One fBm parameter block — mirror of noise.rs `FbmParams` (the per-axis knobs the codegen passes in).
struct WgFbmParams {
    octaves: u32,
    base_freq: f32,
    lacunarity: f32,
    gain: f32,
    amplitude: f32,
    seed: u32,
}

// noise.rs `fbm_height_grad` — fBm height + analytic world-space XZ gradient. Identical octave loop:
// per-octave distinct seed (wrapping-mul, bit-exact), value_noise at the frequency-scaled coord, then
// `h += v·amp`, `dh += g·amp·freq` (SAME `(g·amp)·freq` association as the CPU). Returns vec3(h, ∂x, ∂z).
fn wg_fbm_height_grad(wx: f32, wz: f32, p: WgFbmParams) -> vec3<f32> {
    var freq = p.base_freq;
    var amp = p.amplitude;
    var h = 0.0;
    var dh_dx = 0.0;
    var dh_dz = 0.0;
    for (var o: u32 = 0u; o < p.octaves; o = o + 1u) {
        let oseed = p.seed + o * 0x9E3779B9u; // wrapping (u32) — matches CPU `wrapping_add(o.wrapping_mul(..))`
        let g = wg_value_noise_grad(wx * freq, wz * freq, oseed);
        h = h + g.x * amp;
        dh_dx = dh_dx + g.y * amp * freq;
        dh_dz = dh_dz + g.z * amp * freq;
        freq = freq * p.lacunarity;
        amp = amp * p.gain;
    }
    return vec3<f32>(h, dh_dx, dh_dz);
}

// NodeKind::Fbm — the source node: fold the world seed with the axis salt (matches the CPU
// `(world_seed_lo) ^ (world_seed_hi) ^ seed_salt`; here `world_seed` is the already-u32-collapsed seed,
// so the fold is `world_seed ^ seed_salt` — see header divergence note), then evaluate fBm.
fn wg_fbm_node(wx: f32, wz: f32, world_seed: u32, axis: WgFbmParams) -> WgField {
    var p = axis;
    p.seed = world_seed ^ axis.seed;
    let g = wg_fbm_height_grad(wx, wz, p);
    return WgField(g.x, g.y, g.z);
}

// =====================================================================================================
// Erosion — mirror of erosion.rs `erode_with_grad` (analytic-gradient ridged-erosion filter). All knobs
// arrive in WgErosionParams (uniform-shaped) — NEVER consts. Returns vec3(eroded_h, ∂H/∂wx, ∂H/∂wz).
// f32 (CPU f64). The value lane mirrors `erode_height` op-for-op; the gradient lane the closed-form
// differentiation (needs the base Hessian — passed in as hxx/hxz/hzz).
// =====================================================================================================

// Editor-tweakable erosion params — mirror of erosion.rs `ErosionParams`. `enabled` is a u32 flag
// (0 = off → exact identity, 1 = on). Knobs as a struct (uniform), never consts.
struct WgErosionParams {
    enabled: u32,
    strength: f32,
    octaves: u32,
    base_cell_size: f32,
    lacunarity: f32,
    gain: f32,
    gully_weight: f32,
    peak_valley_fade: f32,
    seed_salt: u32,
}

// noise.rs `fade_deriv2` — 60t(2t³ − 3t² + t) (Horner: 60·t·(t·(2t−3)+1)). The noise Hessian ingredient.
fn wg_fade_deriv2(t: f32) -> f32 {
    return 60.0 * t * (t * (2.0 * t - 3.0) + 1.0);
}

// noise.rs `value_noise_grad_hess` — one octave value noise + gradient AND Hessian at (scaled) (x, z).
// Returns (v, ∂x, ∂z, ∂xx, ∂xz, ∂zz) packed into two vec3s as (v, dx, dz) and (dxx, dxz, dzz).
struct WgNoiseHess {
    g: vec3<f32>,  // (value, ∂v/∂x, ∂v/∂z)
    hess: vec3<f32>, // (∂²v/∂x², ∂²v/∂x∂z, ∂²v/∂z²)
}

fn wg_value_noise_grad_hess(x: f32, z: f32, seed: u32) -> WgNoiseHess {
    let xi = floor(x);
    let zi = floor(z);
    let ix = i32(xi);
    let iz = i32(zi);
    let fx = x - xi;
    let fz = z - zi;

    let v00 = wg_value_at_lattice(ix, iz, seed);
    let v10 = wg_value_at_lattice(ix + 1, iz, seed);
    let v01 = wg_value_at_lattice(ix, iz + 1, seed);
    let v11 = wg_value_at_lattice(ix + 1, iz + 1, seed);

    let u = wg_fade(fx);
    let vv = wg_fade(fz);
    let du = wg_fade_deriv(fx);
    let dv = wg_fade_deriv(fz);
    let ddu = wg_fade_deriv2(fx);
    let ddv = wg_fade_deriv2(fz);

    let a = v00 + (v10 - v00) * u;
    let b = v01 + (v11 - v01) * u;
    let value = a + (b - a) * vv;

    let da_dx = (v10 - v00) * du;
    let db_dx = (v11 - v01) * du;
    let dval_dx = da_dx + (db_dx - da_dx) * vv;
    let dval_dz = (b - a) * dv;

    let dxx = ddu * ((v10 - v00) + ((v11 - v01) - (v10 - v00)) * vv);
    let dxz = du * dv * ((v11 - v01) - (v10 - v00));
    let dzz = (b - a) * ddv;

    var out: WgNoiseHess;
    out.g = vec3<f32>(value, dval_dx, dval_dz);
    out.hess = vec3<f32>(dxx, dxz, dzz);
    return out;
}

// erosion.rs `smooth_bump` — (1 − clamp(t)²) clamped to [0,1].
fn wg_smooth_bump(t: f32) -> f32 {
    let tc = clamp(t, -1.0, 1.0);
    let b = 1.0 - tc * tc;
    if (b < 0.0) {
        return 0.0;
    }
    return b;
}

// erosion.rs `smooth_bump_deriv` — −2t inside (−1,1), 0 outside.
fn wg_smooth_bump_deriv(t: f32) -> f32 {
    if (t > -1.0 && t < 1.0) {
        return -2.0 * t;
    }
    return 0.0;
}

// erosion.rs `erosion_seed` — fold the (u32-collapsed) world seed with the erosion salt.
fn wg_erosion_seed(world_seed: u32, p: WgErosionParams) -> u32 {
    return world_seed ^ p.seed_salt;
}

// erosion.rs `octave_salt` — per-octave salt (wrapping mul, bit-exact).
fn wg_octave_salt(o: u32) -> u32 {
    return o * 0x9E3779B9u;
}

// erosion.rs `erode_with_grad` — carve the base height with the ridged-erosion filter AND return the
// carved surface's analytic XZ gradient. Inputs: base height + gradient (gx_base/gz_base) + Hessian
// (hxx/hxz/hzz), world (wx,wz), the u32-collapsed world seed, and the knob struct. `enabled == 0` ⇒
// exact identity (h_base, gx_base, gz_base). Returns vec3(H, ∂H/∂wx, ∂H/∂wz). Mirrors the CPU term by term.
fn wg_erode_with_grad(
    h_base: f32,
    gx_base: f32,
    gz_base: f32,
    hxx: f32,
    hxz: f32,
    hzz: f32,
    wx: f32,
    wz: f32,
    world_seed: u32,
    p: WgErosionParams,
) -> vec3<f32> {
    if (p.enabled == 0u) {
        return vec3<f32>(h_base, gx_base, gz_base);
    }
    let seed = wg_erosion_seed(world_seed, p);
    let base_cell = p.base_cell_size;
    var inv_cell = 1.0;
    if (base_cell > 1e-6) {
        inv_cell = 1.0 / base_cell;
    }
    let lacunarity = p.lacunarity;
    let gain = p.gain;
    let gully = p.gully_weight;

    var freq = inv_cell;
    var amp = 1.0;
    // Running slope (gradient) + its derivative (accumulated Hessian), seeded by the base values.
    var gx = gx_base;
    var gz = gz_base;
    var g_xx = hxx;
    var g_xz = hxz;
    var g_zz = hzz;
    var detail = 0.0;
    var ddetail_dx = 0.0;
    var ddetail_dz = 0.0;
    var norm = 0.0;

    for (var o: u32 = 0u; o < p.octaves; o = o + 1u) {
        let nh = wg_value_noise_grad_hess(wx * freq, wz * freq, seed ^ wg_octave_salt(o));
        let vval = nh.g.x;
        let dvx = nh.g.y;
        let dvz = nh.g.z;
        let dxx = nh.hess.x;
        let dxz = nh.hess.y;
        let dzz = nh.hess.z;

        // --- value path (mirror of erode_height) ---
        var av = vval;
        if (vval < 0.0) {
            av = -vval;
        }
        var r = 1.0 - av;
        r = r * r * (1.0 + gully);
        let slope2 = gx * gx + gz * gz;
        let damp = 1.0 / (1.0 + gully * slope2);
        detail = detail + amp * r * damp;
        norm = norm + amp;

        // --- gradient path ---
        let dv_dx = dvx * freq;
        let dv_dz = dvz * freq;
        var sgn = 1.0;
        if (vval < 0.0) {
            sgn = -1.0;
        }
        let one_minus_av = 1.0 - av;
        let coef = (1.0 + gully) * 2.0 * one_minus_av;
        let dr_dx = coef * (-sgn * dv_dx);
        let dr_dz = coef * (-sgn * dv_dz);
        let dslope2_dx = 2.0 * gx * g_xx + 2.0 * gz * g_xz;
        let dslope2_dz = 2.0 * gx * g_xz + 2.0 * gz * g_zz;
        let damp2 = damp * damp;
        let ddamp_dx = -gully * dslope2_dx * damp2;
        let ddamp_dz = -gully * dslope2_dz * damp2;
        ddetail_dx = ddetail_dx + amp * (dr_dx * damp + r * ddamp_dx);
        ddetail_dz = ddetail_dz + amp * (dr_dz * damp + r * ddamp_dz);

        // Feed this octave's gradient into the running slope, Hessian into its derivative.
        let f2 = freq * freq;
        gx = gx + dvx * freq * amp;
        gz = gz + dvz * freq * amp;
        g_xx = g_xx + dxx * f2 * amp;
        g_xz = g_xz + dxz * f2 * amp;
        g_zz = g_zz + dzz * f2 * amp;

        freq = freq * lacunarity;
        amp = amp * gain;
    }

    let norm_clamped = max(norm, 1e-6);
    let detail_n = detail / norm_clamped;
    let inv_norm = 1.0 / norm_clamped;
    let ddet_dx = ddetail_dx * inv_norm;
    let ddet_dz = ddetail_dz * inv_norm;

    let str1 = p.strength + 1.0;
    let h_norm = h_base / str1;
    let pvf = p.peak_valley_fade;
    let bump = wg_smooth_bump(h_norm);
    let fade = 1.0 - pvf * bump;
    let bump_d = wg_smooth_bump_deriv(h_norm);
    let dfade_dx = -pvf * bump_d * gx_base / str1;
    let dfade_dz = -pvf * bump_d * gz_base / str1;

    let strength = p.strength;
    let h = h_base - strength * fade * detail_n;
    let dh_dx = gx_base - strength * (dfade_dx * detail_n + fade * ddet_dx);
    let dh_dz = gz_base - strength * (dfade_dz * detail_n + fade * ddet_dz);
    return vec3<f32>(h, dh_dx, dh_dz);
}
