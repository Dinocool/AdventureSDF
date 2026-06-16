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


# Repo root derived from this file's location (rdoc/scripts/trace/_lib.py -> up 3), so
# `newest_trace()` finds traces in the CURRENT worktree, not a hardcoded one.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))


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


# Bevy's per-system / per-function spans are all named the generic literal `function_scope`;
# the REAL system/function name lives in the slice's `args.message` arg (the editor builds the
# layer with `.include_args(true)`). So a meaningful name is: if the span is `function_scope`,
# read its `args.message`, else use the span name. perfetto exposes args via EXTRACT_ARG on the
# slice's arg_set_id. `{s}` is the slice-table alias to qualify (default `s`).
def resolved_name(s="s"):
    return (
        f"CASE WHEN {s}.name='function_scope' "
        f"THEN IFNULL(EXTRACT_ARG({s}.arg_set_id,'args.message'), {s}.name) "
        f"ELSE {s}.name END"
    )


def frame_span_name(tp):
    """The top-level per-frame span name, detected robustly.

    Current Bevy traces name it `update` (older ones used `update: ` with a trailing
    colon-space). We accept either, preferring whichever actually appears, and fall back to
    the most-frequent depth-0 root span so this keeps working if the literal drifts again.
    """
    for cand in ("update", "update: "):
        rows = list(tp.query(f"SELECT count(*) n FROM slice WHERE name='{cand}'"))
        if rows and rows[0].n:
            return cand
    # Fallback: most frequent root (depth 0) span — frames repeat once per tick.
    rows = list(
        tp.query(
            "SELECT name, count(*) n FROM slice WHERE depth=0 "
            "GROUP BY name ORDER BY n DESC LIMIT 1"
        )
    )
    return rows[0].name if rows else "update"


# Reusable SQL fragment: self-time = a slice's dur minus its children's dur, grouped by the
# RESOLVED name (so `function_scope` rolls up under the real system/function name). `{where}`
# is an optional `WHERE ...` clause on `slice s`.
SELF_TIME_CTE = """
WITH self AS (
  SELECT {resolved} AS name, s.dur AS dur,
    s.dur - IFNULL((SELECT sum(c.dur) FROM slice c WHERE c.parent_id = s.id), 0) AS self_dur
  FROM slice s {where}
)
""".replace("{resolved}", resolved_name("s"))
