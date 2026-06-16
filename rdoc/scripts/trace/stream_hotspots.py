"""Streaming chrome-trace hotspot analyzer — for HUGE traces perfetto can't load.

bevy/trace_chrome writes one JSON object PER LINE (after the opening `[`), so we parse line
by line with the stdlib — RAM stays tiny no matter the file size (6-25GB traces that OOM
perfetto's trace_processor stream through in one pass). Aggregates B/E span durations by name
via a per-thread (pid,tid) stack, computing self-time (a span's dur minus its children's).

    python rdoc/scripts/trace/stream_hotspots.py <trace.json> [--frames] [--skip-us=N]

--frames also reports the per-frame (`update`) duration distribution + slowest frames.
--skip-us=N ignores all events with ts < N (drop startup/loading frames; bevy ts is µs,
so --skip-us=5000000 skips the first 5 seconds). Also reports each span's MAX single
occurrence + occurrences-after-skip, so a steady per-frame cost is distinguishable from a
few transient heavy frames.
"""

import sys
import os
import json
from collections import defaultdict


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    skip_us = 0.0
    for fl in flags:
        if fl.startswith("--skip-us="):
            skip_us = float(fl.split("=", 1)[1])
    if not args:
        # newest trace-*.json in the CURRENT worktree (shared with the perfetto scripts).
        sys.path.insert(0, os.path.dirname(__file__))
        from _lib import newest_trace
        path = newest_trace()
        if not path:
            print("no trace-*.json found (pass one as argv)")
            return
    else:
        path = args[0]
    print(f"=== streaming hotspots: {os.path.basename(path)} ===")

    # Per (pid,tid) stack of open B events: (name, ts, child_dur_accum).
    stacks = defaultdict(list)
    # name -> [self_us_total, total_us, count, max_self_us]
    agg = defaultdict(lambda: [0.0, 0.0, 0, 0.0])
    # frame durations (update: spans)
    frames = []
    if skip_us > 0:
        print(f"  (skipping events with ts < {skip_us/1e6:.1f}s — startup/loading frames)")

    want_frames = "--frames" in flags
    n_events = 0
    n_bad = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()  # the bare "[" line
        for line in f:
            line = line.strip()
            if not line or line == "]":
                continue
            if line.endswith(","):
                line = line[:-1]
            try:
                e = json.loads(line)
            except Exception:
                n_bad += 1
                continue
            ph = e.get("ph")
            if ph == "B":
                key = (e.get("pid"), e.get("tid"))
                name = e.get("name", "?")
                # Bevy names every system/function span `function_scope`; the real name is in
                # args.message. Resolve it here so the aggregate is per-system, not one giant
                # function_scope bucket.
                if name == "function_scope":
                    msg = (e.get("args") or {}).get("message")
                    if msg:
                        name = msg
                stacks[key].append([name, e.get("ts", 0.0), 0.0])
                n_events += 1
            elif ph == "E":
                key = (e.get("pid"), e.get("tid"))
                st = stacks[key]
                if not st:
                    continue
                name, ts0, child = st.pop()
                dur = e.get("ts", 0.0) - ts0
                self_dur = dur - child
                # add full dur to the new parent's child accumulator (always — keeps parent
                # self-time correct regardless of the skip window)
                if st:
                    st[-1][2] += dur
                n_events += 1
                if ts0 < skip_us:
                    continue  # span started in the skipped startup window
                a = agg[name]
                a[0] += self_dur
                a[1] += dur
                a[2] += 1
                if self_dur > a[3]:
                    a[3] = self_dur
                if want_frames and name in ("update", "update: "):
                    frames.append(dur)

    print(f"  parsed {n_events} B/E events ({n_bad} unparseable lines)\n")

    # us values: bevy trace ts is in microseconds already.
    rows = sorted(agg.items(), key=lambda kv: kv[1][0], reverse=True)
    print("=== TOP 25 by SELF time (us) ===")
    print(f"  {'self_ms':>10} {'total_ms':>10} {'count':>8} {'avg_us':>9} {'max_us':>9}  name")
    for name, (self_us, tot_us, cnt, max_us) in rows[:25]:
        avg = self_us / cnt if cnt else 0.0
        print(f"  {self_us/1e3:10.1f} {tot_us/1e3:10.1f} {cnt:8d} {avg:9.0f} {max_us:9.0f}  {name[:56]}")

    if want_frames and frames:
        frames.sort()
        n = len(frames)
        avg = sum(frames) / n
        print(f"\n=== frames (update): n={n} avg={avg:.0f}us "
              f"p50={frames[n//2]:.0f}us p99={frames[min(n-1, n*99//100)]:.0f}us "
              f"max={frames[-1]:.0f}us ===")


if __name__ == "__main__":
    main()
