#!/usr/bin/env bash
# Like perf_median.sh but extracts ALL GI passes from EACH capture (one set of N launches → every kernel's
# median), so a 5-kernel split-pipeline A/B costs N launches, not N×markers. Prints per-pass median time+occ.
#
# Usage:  rdoc/scripts/ngfx/perf_allmarkers.sh <label> [N=4] [Frames=2400] [Cam="-0.48,9.59,0.33,0.29,8.96,0.36"]
set -u
LABEL="${1:-run}"; N="${2:-4}"; FRAMES="${3:-2400}"; CAM="${4:--0.48,9.59,0.33,0.29,8.96,0.36}"
OUT=$(mktemp -d)
for i in $(seq 1 "$N"); do
  powershell -NoProfile -Command "Get-Process | Where-Object { \$_.Name -eq 'adventure' } | Stop-Process -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
  rm -rf .soul/ngfx/BASE
  powershell -ExecutionPolicy Bypass -File rdoc/scripts/ngfx/capture.ps1 -Light -Frames "$FRAMES" -Cam "$CAM" > "$OUT/log$i.txt" 2>&1
  if grep -qiE "Failed to connect|TARGET ERROR" "$OUT/log$i.txt"; then
    echo "  $LABEL run $i/$N: CAPTURE FAILED (rebuild static: cargo build --no-default-features --features editor,shader-debug)"; rm -rf "$OUT"; exit 2
  fi
  python rdoc/scripts/ngfx/parse.py .soul/ngfx >/dev/null 2>&1
  R=$(grep -oE "resident_bricks=[0-9]+" "$OUT/log$i.txt" | tail -1 | grep -oE "[0-9]+"); R="${R:-0}"
  cp .soul/ngfx/perf.json "$OUT/perf$i.json"
  echo "$R" >> "$OUT/resid.txt"
  printf "  %s run %d/%d: resident_bricks=%s\n" "$LABEL" "$i" "$N" "$R"
done
python - "$LABEL" "$OUT" <<'PY'
import sys, json, glob, os, statistics as st
lab, out = sys.argv[1], sys.argv[2]
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out,'perf*.json')))]
resid = [int(x) for x in open(os.path.join(out,'resid.txt')).read().split()]
rmax = max(resid) if resid else 0
# collect every pass name (lowercased) seen across runs
names = {}
for d in runs:
    for p in d['passes']:
        names.setdefault(p['pass'].lower(), p['pass'])
print(f"[{lab}] resident_max={rmax}  n={len(runs)}")
order = ['gi_world_cache','gi_restir_p1','gi_di_p1','gi_restir_spatial','gi_restir_p2','gi_restir_debug']
def key(n):
    for i,k in enumerate(order):
        if n.endswith(k): return (i, n)
    return (99, n)
total_med = 0.0
for low in sorted(names, key=key):
    ts=[]; os_=[]
    for d in runs:
        m=[x for x in d['passes'] if x['pass'].lower()==low]
        if m:
            ts.append(m[0]['gpu_time_us']); os_.append(m[0]['cs_warp_occupancy_pct'])
    if not ts: continue
    mt = st.median(ts); mo = st.median(os_)
    if any(low.endswith(k) for k in order[1:]): total_med += mt  # sum GI kernels (skip world_cache)
    print(f"  {names[low]:<28} time={mt:7.0f}us  occ={mo:5.1f}%  (n={len(ts)})")
print(f"  {'== GI kernels total ==':<28} time={total_med:7.0f}us")
bad=[i+1 for i,v in enumerate(resid) if v<rmax]
if bad: print(f"  WARNING: runs {bad} captured pre-load (resident<{rmax}) — raise Frames")
PY
rm -rf "$OUT"
