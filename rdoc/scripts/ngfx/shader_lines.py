#!/usr/bin/env python3
"""Rank the hot WGSL lines in an Nsight shader-profiler CSV export.

Nsight Graphics' Shader Profiler view (open the `.ngfx-gputrace` in the GUI, pick the slow
shader, right-click the source/SASS table -> Export to CSV) emits one row per SPIR-V instruction
with `Samples`, up to three `Top Stall` reasons, `Instruction Mix`, and `Live Registers`. When the
app is built with `--features shader-debug` (which now also pulls `bevy_render/decoupled_naga`, so
naga emits `OpSource`/`OpLine` — see Cargo.toml), the export embeds the COMPOSED WGSL source and
`OpLine` line mappings. This script reconstructs that source, walks the instructions tracking the
current `OpLine`, and aggregates GPU samples + peak live registers back onto WGSL lines.

  python rdoc/scripts/ngfx/shader_lines.py .soul/ngfx/sample.csv [--top 25]

Use it after `parse.py` / the analyzer say a pass is shader/occupancy bound and you need to know
WHICH lines hold the registers (occupancy limiter) or burn the samples (time). `Samples` is 0 on a
settled/idle frame — capture an actively-converging frame for time, but `Live Registers` is present
regardless and pinpoints the occupancy limiter.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

OPLINE = re.compile(r"^OpLine %\d+ (\d+) (\d+)")
# SPIR-V ops that mark the end of the OpSource embedded-source text block.
SRC_END = re.compile(r"^(OpName|OpMemberName|OpModuleProcessed|OpDecorate|OpMemberDecorate|%\d|OpTypeVoid|OpExtInst)")


def num(s: str) -> float:
    s = (s or "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="Nsight shader-profiler CSV export")
    ap.add_argument("--top", type=int, default=25, help="rows to print per ranking")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        sys.stderr.write(f"not found: {path}\n")
        return 2
    rows = list(csv.reader(open(path, encoding="utf-8", errors="replace")))
    if not rows:
        sys.stderr.write("empty CSV\n")
        return 2
    hdr = rows[0]
    ci = {name: i for i, name in enumerate(hdr)}
    SRC = ci.get("Source", 1)
    SAMP = ci.get("Samples", 2)
    LR = ci.get("Live Registers", len(hdr) - 1)
    ST1 = ci.get("Top Stall #1 (Type)", 3)

    def col(r, i):
        return r[i] if len(r) > i else ""

    # 1. reconstruct the embedded composed-WGSL source from the OpSource block
    wgsl: list[str] = []
    in_src = False
    for r in rows[1:]:
        s = col(r, SRC)
        if s.startswith("OpSource WGSL"):
            in_src = True
            m = re.search(r'"(.*)$', s)
            wgsl.append(m.group(1) if m else "")
            continue
        if in_src:
            if SRC_END.match(s):
                in_src = False
            else:
                wgsl.append(s)

    # 2. walk instructions, attributing samples + max live-registers to the current OpLine
    cur = None
    samp_by, lr_by, stall_by = {}, {}, {}
    total_samp = 0.0
    peak_lr = 0.0
    peak_line = None
    for r in rows[1:]:
        s = col(r, SRC)
        m = OPLINE.match(s)
        if m:
            cur = int(m.group(1))
            continue
        sv, lr = num(col(r, SAMP)), num(col(r, LR))
        total_samp += sv
        if lr > peak_lr:
            peak_lr, peak_line = lr, cur
        if cur is None:
            continue
        if sv > 0:
            samp_by[cur] = samp_by.get(cur, 0.0) + sv
            st = col(r, ST1).strip()
            if st:
                stall_by.setdefault(cur, {})
                stall_by[cur][st] = stall_by[cur].get(st, 0.0) + sv
        if lr > 0:
            lr_by[cur] = max(lr_by.get(cur, 0.0), lr)

    def wline(n):
        return wgsl[n - 1].strip()[:88] if wgsl and 1 <= n <= len(wgsl) else "?"

    if not wgsl:
        sys.stderr.write(
            "no embedded WGSL source found — was the app built with `--features shader-debug` "
            "(which pulls bevy_render/decoupled_naga)? Without it naga emits no OpSource/OpLine.\n"
        )
    print(f"embedded WGSL lines: {len(wgsl)}")
    print(f"total samples: {total_samp:.0f}   peak live registers: {peak_lr:.0f} (WGSL L{peak_line})")

    print(f"\n=== top {args.top} WGSL lines by GPU samples ===")
    if total_samp == 0:
        print("  (no samples - settled/idle frame; capture an actively-converging frame for time)")
    for ln, sv in sorted(samp_by.items(), key=lambda x: -x[1])[: args.top]:
        st = stall_by.get(ln, {})
        top_st = max(st.items(), key=lambda x: x[1])[0] if st else "-"
        print(f"  L{ln:<5} {sv / total_samp * 100:5.1f}%  lr<={lr_by.get(ln, 0):3.0f}  {top_st:<22} | {wline(ln)}")

    print(f"\n=== top {args.top} WGSL lines by LIVE REGISTERS (occupancy limiters) ===")
    for ln, lr in sorted(lr_by.items(), key=lambda x: -x[1])[: args.top]:
        sp = samp_by.get(ln, 0.0) / max(total_samp, 1) * 100
        print(f"  L{ln:<5} lr={lr:3.0f}  samp {sp:4.1f}% | {wline(ln)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
