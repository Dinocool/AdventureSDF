"""Top CPU spans by self-time and total-time across a whole chrome trace.

The first trace tool: ranks which Bevy systems / render-graph nodes cost the most CPU.
Replaces trace_analyze.py. Self-time (children subtracted) finds the true leaf cost.

    python rdoc/scripts/trace/hotspots.py [trace.json]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from _lib import trace_arg, processor, SELF_TIME_CTE

path = trace_arg()
if not path:
    print("no trace-*.json found (pass one as argv)")
    sys.exit(0)
print(f"=== hotspots: {os.path.basename(path)} ===")
tp = processor(path)
def q(s): return list(tp.query(s))

for r in q("SELECT min(ts) a, max(ts+dur) b FROM slice"):
    print(f"  wall span = {(r.b - r.a) / 1e9:.3f} s")

print("\n=== TOP 25 by SELF time (children subtracted) ===")
for r in q(SELF_TIME_CTE.format(where="") + """
SELECT name, count(*) n, sum(self_dur)/1e6 self_ms, sum(dur)/1e6 total_ms
FROM self GROUP BY name ORDER BY self_ms DESC LIMIT 25
"""):
    print(f"  {r.self_ms:10.1f}ms self | {r.total_ms:10.1f}ms tot | n={r.n:7d} | {r.name[:70]}")

print("\n=== TOP 20 by TOTAL time ===")
for r in q("""
SELECT name, count(*) n, sum(dur)/1e6 total_ms, avg(dur)/1e3 avg_us
FROM slice GROUP BY name ORDER BY total_ms DESC LIMIT 20
"""):
    print(f"  {r.total_ms:10.1f}ms tot | avg {r.avg_us:9.1f}us | n={r.n:7d} | {r.name[:70]}")
