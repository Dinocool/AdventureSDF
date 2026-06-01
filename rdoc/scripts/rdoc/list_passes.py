"""List the render passes / draws / dispatches in a capture, with event IDs.

The orientation tool: run this first to see the frame's structure and get the eventId of
the pass you care about (sdf_pass, cone_prepass, etc.) to feed other scripts.

    qrenderdoc.exe --python rdoc/scripts/rdoc/list_passes.py [capture.rdc]
    -> rdoc/list_passes_out.txt
"""

import sys
import os

# qrenderdoc launches from repo root; cwd gives the toolkit dir (no __file__, no hardcode).
sys.path.insert(0, os.path.join(os.getcwd(), "rdoc", "scripts", "rdoc"))
import renderdoc as rd
from _lib import Tee, open_capture, capture_arg, walk, finish

log = Tee("list_passes")
try:
    path = capture_arg()
    if not path:
        log("no .rdc found")
        finish()
    log(f"=== passes: {os.path.basename(path)} ===")
    cap, ctrl = open_capture(path, log)
    if not ctrl:
        finish(cap)

    sf = ctrl.GetStructuredFile()
    for a in walk(ctrl.GetRootActions()):
        flags = a.flags
        # Only the "interesting" actions: draws, dispatches, and named markers/passes.
        is_draw = bool(flags & rd.ActionFlags.Drawcall)
        is_disp = bool(flags & rd.ActionFlags.Dispatch)
        is_pass = bool(flags & (rd.ActionFlags.PushMarker | rd.ActionFlags.SetMarker))
        if not (is_draw or is_disp or is_pass):
            continue
        kind = "DRAW " if is_draw else "DISP " if is_disp else "MARK "
        log(f"  eid={a.eventId:5d} {kind} {a.GetName(sf)[:72]}")

    finish(ctrl, cap)
except Exception as e:
    import traceback
    log("ERROR:", e, "\n", traceback.format_exc())
    finish()
