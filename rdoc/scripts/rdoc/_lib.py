"""Shared helpers for RenderDoc capture-analysis scripts.

Run any script in this folder with RenderDoc's bundled Python (it ships the `renderdoc`
module; the system Python does NOT have it):

    "C:/Program Files/RenderDoc/qrenderdoc.exe" --python rdoc/scripts/rdoc/<script>.py <args>

`qrenderdoc --python` briefly flashes the UI window (the only path with a replay device,
which GPU counters require — headless `renderdoccmd convert` gives CPU record times only,
and the offscreen Qt platform deadlocks at init). Each script ends in `finish()`, which
force-exits via os._exit so the window closes immediately and nothing hangs. Output is
printed to stdout AND mirrored to rdoc/<script>_out.txt, because qrenderdoc's stdout is
swallowed on Windows — ALWAYS read the _out.txt file, the console will be empty.

Captures live in the system temp dir by default; our editor build now writes them to
rdoc/ (see editor/renderdoc_capture.rs set_capture_file_path_template). Pass a capture
path as argv[1]; if omitted, the newest *.rdc under rdoc/ then the temp dir is used.
"""

import os
import sys
import glob
import renderdoc as rd

# qrenderdoc is always launched from the repo root, so cwd IS the repo — portable, no
# hardcoded worktree path. (qrenderdoc's embedded Python 3.6 defines neither __file__ nor
# sys.argv, so cwd + an env var are the only reliable inputs — see scripts_dir/capture_arg.)
_REPO = os.getcwd()
_TEMP_RD = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Temp", "RenderDoc")


def scripts_dir():
    """Toolkit dir, derived from cwd (repo root) — used for sys.path bootstrap."""
    return os.path.join(_REPO, "rdoc", "scripts", "rdoc")


def newest_capture():
    """Newest .rdc: prefer rdoc/, fall back to the RenderDoc temp dir."""
    pools = [os.path.join(_REPO, "rdoc", "*.rdc"), os.path.join(_TEMP_RD, "*.rdc")]
    cands = [f for p in pools for f in glob.glob(p)]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def capture_arg(default=None):
    """Resolve the capture path, in order:
      1. $RDOC_CAPTURE env var (how the qrenderdoc runner passes it — no sys.argv there)
      2. argv[1] ending in .rdc (system-python use)
      3. newest *.rdc under rdoc/ then the temp dir
    qrenderdoc's embedded Python 3.6 has NO sys.argv, so the env var is the primary path."""
    env = os.environ.get("RDOC_CAPTURE")
    if env and env.lower().endswith(".rdc"):
        return env
    argv = getattr(sys, "argv", [])
    if len(argv) > 1 and argv[1].lower().endswith(".rdc"):
        return argv[1]
    return newest_capture() or default


class Tee:
    """Mirror writes to stdout + a per-script text file (qrenderdoc swallows stdout)."""

    def __init__(self, name):
        self.f = open(os.path.join(_REPO, "rdoc", f"{name}_out.txt"), "w")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        print(s, file=self.f)
        self.f.flush()

    def close(self):
        self.f.close()


def walk(acts):
    """Depth-first iterate every action (draw/dispatch/copy/marker) in the capture."""
    for a in acts:
        yield a
        yield from walk(a.children)


def open_capture(path, log):
    """Open + replay a capture. Returns (cap, ctrl) or (None, None) on failure."""
    cap = rd.OpenCaptureFile()
    if cap.OpenFile(path, "", None) != rd.ResultCode.Succeeded:
        log("open FAIL:", path)
        return None, None
    st, ctrl = cap.OpenCapture(rd.ReplayOptions(), None)
    if st != rd.ResultCode.Succeeded:
        log("replay FAIL:", path)
        cap.Shutdown()
        return None, None
    return cap, ctrl


def action_names(ctrl):
    """{eventId: name} for every action — used to label GPU-counter rows by eid."""
    sf = ctrl.GetStructuredFile()
    return {a.eventId: a.GetName(sf) for a in walk(ctrl.GetRootActions())}


def find_action(ctrl, substr):
    """First action whose name contains `substr` (e.g. 'sdf_pass'), or None."""
    sf = ctrl.GetStructuredFile()
    for a in walk(ctrl.GetRootActions()):
        if substr in a.GetName(sf):
            return a
    return None


def counter_seconds(ctrl, counter=None):
    """Fetch a duration counter and return {eventId: seconds}, decode robust to the
    result's byte width / float-vs-double union member (the silent-empty-output trap:
    reading the wrong union field throws, so we try both)."""
    if counter is None:
        counter = rd.GPUCounter.EventGPUDuration
    out = {}
    for r in ctrl.FetchCounters([counter]):
        v = r.value
        # The union exposes .d (double) and .f (float); pick whichever is finite & nonzero.
        secs = 0.0
        for attr in ("d", "f"):
            try:
                x = getattr(v, attr)
            except Exception:
                continue
            if x == x and x >= 0.0:  # not NaN, not negative
                secs = x
                if x > 0.0:
                    break
        out[r.eventId] = secs
    return out


def finish(*shutdowns):
    """Shut down replay handles and HARD-exit so qrenderdoc doesn't hang on its window."""
    for s in shutdowns:
        try:
            s.Shutdown()
        except Exception:
            pass
    try:
        import qrenderdoc  # only bound in the UI context
        qrenderdoc.GetMainWindow().Close()
    except Exception:
        pass
    os._exit(0)
