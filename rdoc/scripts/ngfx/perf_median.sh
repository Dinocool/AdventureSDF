#!/usr/bin/env bash
# Autonomous, noise-controlled perf A/B for the voxel-RT GI raymarch.
#
# WHY: a single Nsight capture is too noisy for fine A/B — the per-line shader sampler overflows the timestamp
# buffer on a heavy pass (use capture.ps1 -Light to drop it), AND the stochastic GI does different per-frame work
# even on a static, fully-loaded scene. So we (a) capture PAST world-load (resident_bricks must converge) and
# (b) take the MEDIAN of N independent captures. Each capture is a fresh app launch → independent GI state.
#
# Usage:  rdoc/scripts/ngfx/perf_median.sh <label> [N=5] [Frames=2400] [Cam="-0.48,9.59,0.33,0.29,8.96,0.36"]
# Prints per-run (raymarch gpu_time_us, occupancy%, resident_bricks) + the median time/occupancy. Reports any
# run whose resident_bricks didn't reach the converged max (capture fired before the world finished loading).
set -u
LABEL="${1:-run}"; N="${2:-5}"; FRAMES="${3:-2400}"; CAM="${4:--0.48,9.59,0.33,0.29,8.96,0.36}"
TMP=$(mktemp -d); TIMES=(); OCCS=(); RESIDS=()
for i in $(seq 1 "$N"); do
  powershell -NoProfile -Command "Get-Process | Where-Object { \$_.Name -eq 'adventure' } | Stop-Process -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
  powershell -ExecutionPolicy Bypass -File rdoc/scripts/ngfx/capture.ps1 -Light -Frames "$FRAMES" -Cam "$CAM" > "$TMP/log$i.txt" 2>&1
  python rdoc/scripts/ngfx/parse.py .soul/ngfx >/dev/null 2>&1
  read -r T O < <(python -c "import json;d=json.load(open('.soul/ngfx/perf.json'));p=[x for x in d['passes'] if 'raymarch' in x['pass'].lower()][0];print(p['gpu_time_us'],p['cs_warp_occupancy_pct'])")
  R=$(grep -oE "resident_bricks=[0-9]+" "$TMP/log$i.txt" | tail -1 | grep -oE "[0-9]+"); R="${R:-0}"
  TIMES+=("$T"); OCCS+=("$O"); RESIDS+=("$R")
  # %s for the floats (avoids printf locale 'invalid number' on some shells); python rounds in the summary.
  printf "  %s run %d/%d: time=%s us  occ=%s%%  resident_bricks=%s\n" "$LABEL" "$i" "$N" "$T" "$O" "$R"
done
python - "$LABEL" "${TIMES[@]}" "--occ" "${OCCS[@]}" "--res" "${RESIDS[@]}" <<'PY'
import sys, statistics as st
a=sys.argv[1:]; lab=a[0]; rest=a[1:]
t=[float(x) for x in rest[:rest.index('--occ')]]
o=[float(x) for x in rest[rest.index('--occ')+1:rest.index('--res')]]
r=[int(x) for x in rest[rest.index('--res')+1:]]
rmax=max(r) if r else 0
bad=[i+1 for i,v in enumerate(r) if v < rmax]
print(f"[{lab}] MEDIAN time={st.median(t):.0f}us (min {min(t):.0f}, max {max(t):.0f}, n={len(t)})  "
      f"MEDIAN occ={st.median(o):.1f}%  resident_max={rmax}")
if bad: print(f"  WARNING: runs {bad} captured before world-load complete (resident<{rmax}) — raise Frames")
PY
rm -rf "$TMP"
