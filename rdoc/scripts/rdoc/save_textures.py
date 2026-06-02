"""Save named GPU textures from a capture to PNG for visual diff.

Dumps the SDF atlas + cone-seed (or any named resources you pass) at the sdf_pass event,
so you can eyeball atlas contents / a bake bug across two captures. Generalised from the
old analyze.py texture-save block.

    RDOC_CAPTURE=<cap.rdc> RDOC_TEXTURES="name1 name2" \
      qrenderdoc.exe --python rdoc/scripts/rdoc/save_textures.py
    -> rdoc/<basename>_<resourcename>.png  (+ rdoc/save_textures_out.txt)

Default resource names: sdf_dist_atlas, sdf_cone_seed. Override via $RDOC_TEXTURES
(space-separated) since qrenderdoc's Python has no sys.argv.
"""

import sys
import os

# qrenderdoc launches from repo root; cwd gives the toolkit dir (no __file__, no hardcode).
sys.path.insert(0, os.path.join(os.getcwd(), "rdoc", "scripts", "rdoc"))
import renderdoc as rd
from _lib import Tee, open_capture, capture_arg, find_action, finish

_REPO = os.getcwd()

log = Tee("save_textures")
try:
    path = capture_arg()
    if not path:
        log("no .rdc found")
        finish()
    # Resource names from $RDOC_TEXTURES (space-separated); else argv (system python);
    # else defaults. qrenderdoc's Python has no sys.argv, so the env var is primary.
    env_tex = os.environ.get("RDOC_TEXTURES", "").split()
    argv_tex = [a for a in getattr(sys, "argv", [])[1:] if not a.lower().endswith(".rdc")]
    wanted = env_tex or argv_tex or ["sdf_dist_atlas", "sdf_cone_seed"]
    base = os.path.splitext(os.path.basename(path))[0]
    log(f"=== save textures {wanted} from {base} ===")

    cap, ctrl = open_capture(path, log)
    if not ctrl:
        finish(cap)

    sdf = find_action(ctrl, "sdf_pass") or find_action(ctrl, "sdf")
    if sdf:
        ctrl.SetFrameEvent(sdf.eventId, True)

    for res in ctrl.GetResources():
        if res.name in wanted:
            ts = rd.TextureSave()
            ts.resourceId = res.resourceId
            ts.destType = rd.FileType.PNG
            ts.mip = 0
            ts.slice.sliceIndex = 0
            fn = os.path.join(_REPO, "rdoc", f"{base}_{res.name}.png")
            ok = ctrl.SaveTexture(ts, fn)
            log(f"  {'saved' if ok else 'FAILED'}: {res.name} -> {fn}")

    finish(ctrl, cap)
except Exception as e:
    import traceback
    log("ERROR:", e, "\n", traceback.format_exc())
    finish()
