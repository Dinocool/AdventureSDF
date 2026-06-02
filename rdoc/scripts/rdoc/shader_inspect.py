"""Inspect the SDF fragment shader bound at the heavy draw: what's actually compiled in.

Static attribution of the raymarch's cost — pulls the COMPILED fragment shader at the
sdf_pass draw and reports: which shader-defs are live (the #ifdef branches that survived),
how many texture-sample ops the disassembly contains (the triplanar tap count), and the
march-loop bounds from the bound uniform (max_steps etc). Concrete evidence of which
shader elements exist in THIS capture, before any A/B timing.

    RDOC_CAPTURE=<cap.rdc> qrenderdoc.exe --python rdoc/scripts/rdoc/shader_inspect.py
    -> rdoc/shader_inspect_out.txt
"""

import sys
import os

# qrenderdoc launches from repo root; cwd gives the toolkit dir (no __file__, no sys.argv).
sys.path.insert(0, os.path.join(os.getcwd(), "rdoc", "scripts", "rdoc"))
import renderdoc as rd
from _lib import Tee, open_capture, capture_arg, find_action, walk, finish


def heaviest_draw(ctrl):
    """The Drawcall action with the largest GPU duration — for the SDF scene this is the
    fullscreen raymarch. A marker/pass event has no bound pipeline, so we must time-rank
    the actual draws and inspect the winner."""
    secs = {}
    try:
        for r in ctrl.FetchCounters([rd.GPUCounter.EventGPUDuration]):
            v = r.value
            secs[r.eventId] = getattr(v, "d", 0.0) or getattr(v, "f", 0.0)
    except Exception:
        pass
    best = None
    best_t = -1.0
    for a in walk(ctrl.GetRootActions()):
        if a.flags & rd.ActionFlags.Drawcall:
            t = secs.get(a.eventId, 0.0)
            if t > best_t:
                best, best_t = a, t
    return best, best_t

log = Tee("shader_inspect")
try:
    path = capture_arg()
    if not path:
        log("no .rdc found")
        finish()
    log(f"=== shader inspect: {os.path.basename(path)} ===")
    cap, ctrl = open_capture(path, log)
    if not ctrl:
        finish(cap)

    # The heaviest DRAW (not the sdf_pass marker — markers have no bound pipeline, which is
    # why GetShaderReflection came back empty when we used the marker's eid).
    sdf, t = heaviest_draw(ctrl)
    if not sdf:
        log("no draw action found")
        finish(ctrl, cap)
    ctrl.SetFrameEvent(sdf.eventId, True)
    sf = ctrl.GetStructuredFile()
    log(f"  heaviest draw: eid={sdf.eventId} gpu={t*1e3:.1f}ms  {sdf.GetName(sf)[:60]}")

    pipe = ctrl.GetPipelineState()
    stage = rd.ShaderStage.Fragment
    refl = pipe.GetShaderReflection(stage)
    if not refl:
        log("  no fragment reflection")
        finish(ctrl, cap)

    # Resources the fragment shader actually uses: texture bindings (tap targets) + UBOs.
    ro = refl.readOnlyResources
    samplers = refl.samplers
    log(f"  read-only resources (textures/buffers): {len(ro)}")
    for r in ro:
        log(f"    - {r.name}")
    log(f"  samplers: {len(samplers)}")

    # Disassemble the compiled shader and COUNT texture-sample ops — the real per-pixel
    # tap count after all #ifdef/inlining. Targets vary by API; pick the first available.
    targets = ctrl.GetDisassemblyTargets(True)
    log(f"  disassembly targets: {[t for t in targets]}")
    disasm = ""
    if targets:
        disasm = ctrl.DisassembleShader(pipe.GetGraphicsPipelineObject(), refl, targets[0])
    if disasm:
        low = disasm.lower()
        # SPIR-V: OpImageSampleExplicitLod / ImplicitLod. GLSL-ish: textureLod/texture(.
        n_sample = (
            low.count("opimagesample")
            + low.count("imagesampl")
        )
        n_texturelod = low.count("texturelod") + low.count("texturesample")
        # Live feature branches: grep the disassembly for our function/define footprints.
        feats = {
            "trace_reflection (SDF_REFLECTIONS)": "trace_reflection" in low or "reflection" in low,
            "surface_shadow (SDF_SHADOWS)": "shadow" in low,
            "edge sample (SDF_EDGE_WEAR)": "edge" in low,
        }
        log(f"\n  === compiled fragment shader ({len(disasm)} chars disasm) ===")
        log(f"  texture-sample ops (OpImageSample*): {n_sample}")
        log(f"  textureLod/Sample tokens: {n_texturelod}")
        for k, v in feats.items():
            log(f"  live: {k} = {v}")
        # Save full disassembly for manual reading.
        dp = os.path.join(os.getcwd(), "rdoc", "shader_inspect_disasm.txt")
        with open(dp, "w") as f:
            f.write(disasm)
        log(f"  full disassembly -> {dp}")
    else:
        log("  (no disassembly available)")

    # March bounds from the fragment UBO (max_steps is debug_params/march tuning; the loop
    # iteration cap directly scales cost).
    log("\n  (march uniforms: see dump_camera_ubo.py for camera/lod/debug_params)")

    finish(ctrl, cap)
except Exception as e:
    import traceback
    log("ERROR:", e, "\n", traceback.format_exc())
    finish()
