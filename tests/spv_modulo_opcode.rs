//! Root-cause probe for the negative-signed-`%` divergence (RTX 4090 / Vulkan / NVIDIA
//! 595.79): does naga's SPIR-V writer lower a SIGNED `%` on a runtime value to the
//! correct signed remainder (`OpSRem`/`OpSMod`), or to an UNSIGNED `OpUMod`?
//!
//! If it emits `OpUMod` for a signed operand → naga codegen bug (reportable upstream).
//! If it emits `OpSRem`/`OpSMod` → naga is correct and the unsigned result observed on
//! the GPU is a driver bug. Either way this pins the layer.
//!
//! SPIR-V arithmetic opcodes (word = (wordcount<<16)|opcode), the low 16 bits:
//!   OpSDiv = 135, OpUDiv = 134, OpUMod = 137, OpSRem = 138, OpSMod = 139, OpFMod = 141.

use naga::back::spv;
use naga::valid::{Capabilities, ValidationFlags, Validator};

const SRC: &str = r#"
@group(0) @binding(0) var<storage, read> ins: array<i32>;
@group(0) @binding(1) var<storage, read_write> outs: array<i32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let a = ins[i * 2u];
    let b = ins[i * 2u + 1u];
    outs[i * 2u] = a % b;   // SIGNED modulo on runtime values — the op under test
    outs[i * 2u + 1u] = a / b;
}
"#;

fn opcode_name(op: u16) -> Option<&'static str> {
    Some(match op {
        134 => "OpUDiv",
        135 => "OpSDiv",
        136 => "OpFDiv",
        137 => "OpUMod",
        138 => "OpSRem",
        139 => "OpSMod",
        140 => "OpFRem",
        141 => "OpFMod",
        _ => return None,
    })
}

#[test]
fn naga_signed_modulo_spirv_opcode() {
    let module = naga::front::wgsl::parse_str(SRC).expect("parse wgsl");
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("validate");

    let mut words = Vec::new();
    let opts = spv::Options::default();
    let mut writer = spv::Writer::new(&opts).expect("writer");
    writer
        .write(&module, &info, None, &None, &mut words)
        .expect("emit spirv");

    // Walk the SPIR-V instruction stream: each instruction's first word packs
    // wordcount in the high 16 bits and the opcode in the low 16. Header is 5 words.
    let mut found: Vec<&'static str> = Vec::new();
    let mut idx = 5;
    while idx < words.len() {
        let w = words[idx];
        let wc = (w >> 16) as usize;
        let op = (w & 0xffff) as u16;
        if wc == 0 {
            break;
        }
        if let Some(name) = opcode_name(op) {
            found.push(name);
        }
        idx += wc;
    }

    println!("arithmetic div/mod opcodes emitted by naga spv-out: {found:?}");

    let has_umod = found.contains(&"OpUMod");
    let has_signed_rem = found.contains(&"OpSRem") || found.contains(&"OpSMod");
    let has_sdiv = found.contains(&"OpSDiv");

    println!(
        "  signed `/` -> OpSDiv present: {has_sdiv}\n  signed `%` -> OpUMod present: {has_umod} | OpSRem/OpSMod present: {has_signed_rem}"
    );

    // Verdict (printed, not asserted — this is a diagnostic):
    if has_umod && !has_signed_rem {
        println!("VERDICT: naga lowered SIGNED `%` to UNSIGNED OpUMod — naga codegen bug.");
    } else if has_signed_rem {
        println!("VERDICT: naga emitted signed remainder (OpSRem/OpSMod) — naga correct; GPU/driver lowers it wrong.");
    } else {
        println!("VERDICT: inconclusive — no recognised modulo opcode (maybe polyfilled via mul/sub).");
    }
}
