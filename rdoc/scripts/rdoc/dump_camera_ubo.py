"""Decode the SdfCameraData uniform bound to the SDF fragment shader.

Verifies what the GPU actually saw — camera_pos, num_chunks, lod_params, debug_params —
when a render bug looks like a CPU/GPU data mismatch (e.g. wrong LOD, ghost geometry).
Generalised from the old camdump.py/analyze.py CB dumps.

    qrenderdoc.exe --python rdoc/scripts/rdoc/dump_camera_ubo.py [capture.rdc]
    -> rdoc/dump_camera_ubo_out.txt

SdfCameraData layout (see sdf/bindings.wgsl): 2x mat4 (32f) then vec4s — camera_pos =
f[32..35], grid_dims = f[44..47] (w = num_chunks), debug_params = f[48..51],
lod_params = f[56..59] (z = voxel_size).
"""

import sys
import os
import struct

# qrenderdoc launches from repo root; cwd gives the toolkit dir (no __file__, no hardcode).
sys.path.insert(0, os.path.join(os.getcwd(), "rdoc", "scripts", "rdoc"))
import renderdoc as rd
from _lib import Tee, open_capture, capture_arg, find_action, finish

log = Tee("dump_camera_ubo")
try:
    path = capture_arg()
    if not path:
        log("no .rdc found")
        finish()
    log(f"=== camera UBO: {os.path.basename(path)} ===")
    cap, ctrl = open_capture(path, log)
    if not ctrl:
        finish(cap)

    sdf = find_action(ctrl, "sdf_pass") or find_action(ctrl, "sdf")
    if not sdf:
        log("no sdf_pass action found")
        finish(ctrl, cap)
    ctrl.SetFrameEvent(sdf.eventId, True)

    pipe = ctrl.GetPipelineState()
    cbs = pipe.GetConstantBlocks(rd.ShaderStage.Fragment, False)
    found = False
    for i, cb in enumerate(cbs):
        d = cb.descriptor
        try:
            data = bytes(ctrl.GetBufferData(d.resource, d.byteOffset, d.byteSize))
        except Exception:
            continue
        if len(data) < 240:
            continue
        f = struct.unpack_from("<60f", data[:240])
        # Heuristic guard: grid_dims.x ~ 1024 (grid size) marks the SDF camera UBO.
        if not (1000.0 < f[44] < 1100.0):
            continue
        found = True
        log(
            f"  CB{i} size={len(data)}\n"
            f"    camera_pos = ({f[32]:.3f}, {f[33]:.3f}, {f[34]:.3f})\n"
            f"    grid_dims  = ({f[44]:.1f}, {f[45]:.1f}, {f[46]:.1f}, num_chunks={f[47]:.0f})\n"
            f"    debug_params = ({f[48]:.3f}, {f[49]:.3f}, {f[50]:.5f}, {f[51]:.5f})\n"
            f"    lod_params = (count={f[56]:.0f}, {f[57]:.0f}, voxel={f[58]:.4f}, {f[59]:.2f})"
        )
    if not found:
        log("  no SDF camera UBO matched in fragment constant blocks")

    finish(ctrl, cap)
except Exception as e:
    import traceback
    log("ERROR:", e, "\n", traceback.format_exc())
    finish()
