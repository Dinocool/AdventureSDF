"""Shared helpers for Bevy chrome-trace (perfetto) analysis.

These run with the SYSTEM python (needs `pip install perfetto`), NOT qrenderdoc — they
parse the trace-<ts>.json the editor writes on exit (feature "editor" -> bevy/trace_chrome).
A chrome trace shows CPU system + render-graph spans; it canNOT see GPU fragment cost (that
shows only as longer prepare_windows / vsync wait — use the rdoc/ GPU-timing tools for that).

    python rdoc/scripts/trace/<script>.py <trace.json>
"""

import sys
import glob
import os

try:
    from perfetto.trace_processor import TraceProcessor
except ImportError:
    print("perfetto not installed: python -m pip install perfetto")
    raise


_REPO = r"D:\Projects\bevy-setup\.claude\worktrees\gpu-sdf-bake"


def newest_trace():
    cands = glob.glob(os.path.join(_REPO, "trace-*.json"))
    return max(cands, key=os.path.getmtime) if cands else None


def trace_arg():
    """argv[1] if a .json, else newest trace-*.json in the repo root."""
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        return sys.argv[1]
    return newest_trace()


def processor(path):
    return TraceProcessor(trace=path)


# Reusable SQL fragment: self-time = a slice's dur minus its children's dur.
SELF_TIME_CTE = """
WITH self AS (
  SELECT s.name AS name, s.dur AS dur,
    s.dur - IFNULL((SELECT sum(c.dur) FROM slice c WHERE c.parent_id = s.id), 0) AS self_dur
  FROM slice s {where}
)
"""
