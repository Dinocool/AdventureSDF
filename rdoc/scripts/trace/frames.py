"""Per-frame duration distribution + breakdown of the slowest frame(s).

For hitches: finds the worst per-frame spans and shows what ran inside them BY SELF-TIME,
resolving Bevy's generic `function_scope` spans to their real system/function name (the span's
`args.message`). Pass --steady to ignore the first ~2.2s of startup allocation spikes.

    python rdoc/scripts/trace/frames.py [trace.json] [--steady]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from _lib import trace_arg, processor, SELF_TIME_CTE, frame_span_name

steady = "--steady" in sys.argv
path = trace_arg()
if not path:
    print("no trace-*.json found")
    sys.exit(0)
print(f"=== frames{' (steady, ts>2.2e9)' if steady else ''}: {os.path.basename(path)} ===")
tp = processor(path)
def q(s): return list(tp.query(s))

frame = frame_span_name(tp)
fesc = frame.replace("'", "''")
ts_filter = "AND ts > 2200000000" if steady else ""

print(f"=== frame duration distribution (span '{frame}') ===")
for r in q(f"""
SELECT count(*) n, min(dur)/1e3 min_us, max(dur)/1e3 max_us, avg(dur)/1e3 avg_us,
  sum(CASE WHEN dur>20000000 THEN 1 ELSE 0 END) over20ms
FROM slice WHERE name='{fesc}' {ts_filter}
"""):
    if not r.n:
        print("  no frames found")
        sys.exit(0)
    print(f"  n={r.n} min={r.min_us:.0f}us avg={r.avg_us:.0f}us MAX={r.max_us:.0f}us (>20ms: {r.over20ms})")

print("\n=== 10 SLOWEST frames ===")
slow = q(f"""
SELECT id, ts, dur FROM slice WHERE name='{fesc}' {ts_filter}
ORDER BY dur DESC LIMIT 10
""")
for r in slow:
    print(f"  ts={r.ts} dur={r.dur/1e3:.0f}us")

if slow:
    f = slow[0]
    lo, hi = f.ts, f.ts + f.dur
    print(f"\n=== slowest frame breakdown by self-time (dur={f.dur/1e3:.0f}us) ===")
    where = f"WHERE s.ts>={lo} AND s.ts<{hi}"
    for r in q(SELF_TIME_CTE.format(where=where) + """
    SELECT name, count(*) n, sum(self_dur)/1e3 self_us
    FROM self GROUP BY name ORDER BY self_us DESC LIMIT 20
    """):
        nm = r.name if r.name is not None else "?"
        print(f"  {r.self_us:9.0f}us self | n={r.n:6d} | {nm[:70]}")
