"""Stats for one named span across the whole trace — count, total, max, and occurrences.

For when hotspots.py points at a suspect (e.g. init_texture_streaming, schedule_bakes) and
you want its distribution and worst instances. Replaces the bespoke trace_hitch.py.

    python rdoc/scripts/trace/span.py <name-substring> [trace.json]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from _lib import trace_arg, processor, resolved_name

args = [a for a in sys.argv[1:] if not a.endswith(".json")]
if not args:
    print("usage: span.py <name-substring> [trace.json]")
    sys.exit(0)
needle = args[0]
path = trace_arg()
if not path:
    print("no trace-*.json found")
    sys.exit(0)
print(f"=== span '{needle}': {os.path.basename(path)} ===")
tp = processor(path)
def q(s): return list(tp.query(s))

# Match on the RESOLVED name so a needle finds real system/function names that live in
# `function_scope` spans' args.message (e.g. `prepare_voxel_rt`), not just raw span names.
esc = needle.replace("'", "''")
rn = resolved_name("slice")
for r in q(f"""
SELECT count(*) n, sum(dur)/1e6 tot_ms, max(dur)/1e3 max_us, avg(dur)/1e3 avg_us
FROM slice WHERE {rn} LIKE '%{esc}%'
"""):
    if not r.n:
        print("  no matching spans")
        sys.exit(0)
    print(f"  count={r.n} total={r.tot_ms:.1f}ms avg={r.avg_us:.1f}us max={r.max_us:.0f}us")

print("\n=== 10 longest occurrences ===")
for r in q(f"""
SELECT ts, dur/1e3 us, {rn} nm FROM slice WHERE {rn} LIKE '%{esc}%'
ORDER BY dur DESC LIMIT 10
"""):
    print(f"  ts={r.ts} {r.us:.0f}us | {(r.nm or '?')[:50]}")
