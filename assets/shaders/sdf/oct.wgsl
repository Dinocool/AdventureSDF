#define_import_path sdf::oct

// Octahedral encode/decode for a unit normal ↔ a 2-component [0,1] pair (jcgt 2014,
// "A Survey of Efficient Representations for Independent Unit Vectors"). Packs a
// world-space normal into two G-buffer channels; the composite/cascade passes decode it.
// No bindings — safe to import from any shader (including standalone ones).

fn oct_sign_not_zero(v: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        select(-1.0, 1.0, v.x >= 0.0),
        select(-1.0, 1.0, v.y >= 0.0),
    );
}

// Unit normal → [0,1]² octahedral coords.
fn oct_encode(n: vec3<f32>) -> vec2<f32> {
    let denom = abs(n.x) + abs(n.y) + abs(n.z);
    var p = n.xy * (1.0 / max(denom, 1e-8));
    if (n.z <= 0.0) {
        p = (vec2<f32>(1.0) - abs(p.yx)) * oct_sign_not_zero(p);
    }
    return p * 0.5 + 0.5;
}

// [0,1]² octahedral coords → unit normal.
fn oct_decode(f_in: vec2<f32>) -> vec3<f32> {
    let f = f_in * 2.0 - 1.0;
    var n = vec3<f32>(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0) {
        let xy = (vec2<f32>(1.0) - abs(vec2<f32>(n.y, n.x))) * oct_sign_not_zero(n.xy);
        n.x = xy.x;
        n.y = xy.y;
    }
    return normalize(n);
}
