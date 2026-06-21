#!/usr/bin/env python3
# Median per-dispatch GPU time across N Nsight actions.json files (each = one capture's top_20_slowest_actions).
# Usage: python median_actions.py <label> <run1.actions.json> <run2...> ...
import json, sys, statistics
label = sys.argv[1]
files = sys.argv[2:]
per = {}
for f in files:
    d = json.load(open(f))
    for a in d['top_20_slowest_actions']:
        ms = a['total_duration_ns'] / 1e6
        if ms < 0.05:
            continue
        per.setdefault(a['name'], []).append((ms, a.get('headline', {}).get('sm_throughput', 0)))
# NOTE: streaming hitches on Bistro inflate occasional single-frame captures (only ever ADD time), so the
# MIN across runs is the clean steady-state GPU cost — use MIN for A/B, not median. Sorted by min.
print(f"=== over {len(files)} runs ({label}) -- use MIN (clean) for A/B ===")
print(f"{'marker':20s} {'MIN_ms':>8s} {'med':>7s} {'max':>7s} {'n':>3s}")
rows = []
for name, vals in per.items():
    mss = sorted(v[0] for v in vals)
    rows.append((mss[0], name, statistics.median(mss), mss[-1], len(vals)))
rows.sort(reverse=True)
tot = 0.0
for mn, name, med, mx, n in rows:
    print(f"{name[:20]:20s} {mn:8.2f} {med:7.2f} {mx:7.2f} {n:3d}")
    if name.startswith('gi_'):
        tot += mn
print(f"{'GI-dispatch MIN total':20s} {tot:8.2f}")
