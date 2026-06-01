"""Per-draw / per-dispatch GPU timing — the FIRST tool to reach for on a perf capture.

This is the ONLY way to attribute frame cost to specific GPU work: a Bevy chrome-trace
shows fragment cost only as a longer vsync wait (`prepare_windows` self-time), never which
draw, and `renderdoccmd convert` gives CPU command-record times (every draw reads 0us), not
GPU execution. RenderDoc's EventGPUDuration counter times each command on the GPU itself —
but it needs a replay device, so this must run under qrenderdoc --python (window flashes,
auto-closes).

    "C:/Program Files/RenderDoc/qrenderdoc.exe" --python rdoc/scripts/rdoc/gpu_timings.py [capture.rdc]
    -> read rdoc/gpu_timings_out.txt

Prints every draw/dispatch/copy sorted by GPU microseconds, descending. The top row is
almost always the bottleneck. For the SDF renderer the fullscreen `vkCmdDraw` IS the
raymarch — if it dominates, the cost is in the fragment shader (taps, march steps,
reflections), not the CPU.
"""

import sys
import os

# qrenderdoc's embedded Python 3.6 defines neither __file__ nor sys.argv. It IS launched
# from the repo root, so cwd gives the toolkit dir portably (no hardcoded worktree path).
sys.path.insert(0, os.path.join(os.getcwd(), "rdoc", "scripts", "rdoc"))
from _lib import Tee, open_capture, action_names, capture_arg, counter_seconds, finish

log = Tee("gpu_timings")
try:
    path = capture_arg()
    if not path:
        log("no .rdc found (pass one as argv, or capture with F5 in the editor)")
        finish()
    log(f"=== GPU timings: {os.path.basename(path)} ===")

    cap, ctrl = open_capture(path, log)
    if not ctrl:
        finish(cap)

    names = action_names(ctrl)
    secs = counter_seconds(ctrl)  # {eventId: gpu seconds}, robust decode

    rows = sorted(
        ((s * 1e6, eid, names.get(eid, "?")) for eid, s in secs.items()),
        reverse=True,
    )
    total = sum(us for us, _, _ in rows)
    log(f"  {len(rows)} timed events, summed GPU = {total / 1000:.2f} ms\n")
    log(f"  {'us':>10}  {'%':>5}  {'eid':>5}  name")
    for us, eid, nm in rows[:25]:
        pct = 100.0 * us / total if total else 0.0
        log(f"  {us:10.1f}  {pct:5.1f}  {eid:5d}  {nm[:64]}")

    finish(ctrl, cap)
except Exception as e:
    import traceback
    log("ERROR:", e)
    log(traceback.format_exc())
    finish()
