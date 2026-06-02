#!/usr/bin/env python3
"""Parse an Nsight Graphics GPU-Trace auto-export (the BASE/GPUTRACE_REGIMES.xls TSV) into a
compact per-pass perf JSON an AI can diff across runs.

Each row of GPUTRACE_REGIMES.xls is a named GPU "regime" (== one of our render-graph passes,
e.g. `sdf_gbuffer_pass`); the ~120 columns are hardware counters. We pull a curated set: the
unit throughputs (SM / L1TEX / L2 / DRAM — what the pass is BOUND on), the SM instruction mix
(ALU / FMA / transcendental — where compute goes), texture-cache hit rate, warp occupancy,
draw/dispatch counts, and a derived GPU time from elapsed cycles.

Usage:
    python rdoc/scripts/ngfx/parse.py [capture_dir]        # default: .soul/ngfx
    -> writes <dir>/perf.json and prints a summary of the SDF passes.
"""
import json
import sys
from pathlib import Path

# friendly_key -> exact column header in GPUTRACE_REGIMES.xls
METRICS = {
    "sm_throughput_pct": "TriageAC.sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex_throughput_pct": "SM_A.TriageAC.l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_throughput_pct": "LTS.TriageAC.lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_throughput_pct": "FBSP.TriageAC.dramc__throughput.avg.pct_of_peak_sustained_elapsed",
    "inst_executed": "SM_A.TriageAC.sm__inst_executed_realtime.sum",
    "inst_alu": "SM_A.TriageAC.sm__inst_executed_pipe_alu_realtime.sum",
    "inst_fma": "SM_C.TriageAC.smsp__inst_executed_pipe_fma.sum",
    "inst_transcendental": "SM_C.TriageAC.smsp__inst_executed_pipe_xu.sum",
    "tex_hit_rate_pct": "SM_B.TriageAC.l1tex__t_sector_hit_rate.pct",
    "ps_warp_occupancy_pct": "TPC.TriageAC.tpc__warps_active_shader_ps_realtime.avg.pct_of_peak_sustained_elapsed",
    "cs_warp_occupancy_pct": "TPC.TriageAC.tpc__warps_active_shader_cs_realtime.avg.pct_of_peak_sustained_elapsed",
    "draws": "FE_B.TriageAC.fe__draw_count.sum",
    "dispatches": "FE_A.TriageAC.gr__dispatch_count.sum",
    "_cycles": "gpc__cycles_elapsed.avg",
    "_cycles_per_sec": "gpc__cycles_elapsed.avg.per_second",
}

# Passes worth surfacing in the printed summary (substring match).
HOT = ("sdf_", "main_opaque", "main_transparent", "upscaling")


def find_regimes(d: Path) -> Path:
    for c in (d / "BASE" / "GPUTRACE_REGIMES.xls", d / "GPUTRACE_REGIMES.xls"):
        if c.exists():
            return c
    hits = list(d.rglob("GPUTRACE_REGIMES.xls"))
    if not hits:
        sys.exit(f"GPUTRACE_REGIMES.xls not found under {d}")
    return hits[0]


def fnum(s: str):
    try:
        return float(s)
    except ValueError:
        return None


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else ".soul/ngfx")
    f = find_regimes(d)
    lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
    header = lines[0].split("\t")
    # header[0] is "flattened_event_name"; map each wanted column to its index.
    idx = {k: header.index(col) for k, col in METRICS.items() if col in header}
    missing = [k for k in METRICS if k not in idx]

    passes = []
    for line in lines[1:]:
        cols = line.split("\t")
        if not cols or not cols[0]:
            continue
        name = cols[0]
        row = {}
        for k, i in idx.items():
            if i < len(cols):
                row[k] = fnum(cols[i])
        # `gpc__cycles_elapsed.avg` is cycles for the regime; its `.per_second` sibling is the
        # graphics clock in GHz. time = cycles / (GHz * 1e9) -> us = cycles / GHz / 1e3.
        cyc, ghz = row.pop("_cycles", None), row.pop("_cycles_per_sec", None)
        row["gpu_time_us"] = round(cyc / ghz / 1e3, 2) if cyc and ghz else None
        # bottleneck = the highest unit throughput (what the pass is limited by).
        units = {
            "SM": row.get("sm_throughput_pct"),
            "L1TEX": row.get("l1tex_throughput_pct"),
            "L2": row.get("l2_throughput_pct"),
            "DRAM": row.get("dram_throughput_pct"),
        }
        units = {u: v for u, v in units.items() if v is not None}
        row["bottleneck"] = max(units, key=units.get) if units else None
        passes.append({"pass": name, **row})

    out = {"source": str(f), "missing_metrics": missing, "passes": passes}
    (d / "perf.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Console summary of the hot passes.
    print(f"parsed {len(passes)} regimes from {f}")
    if missing:
        print(f"  (columns not found, skipped: {', '.join(missing)})")
    hdr = f"{'pass':28} {'time_us':>8} {'bound':>6} {'SM%':>6} {'L1%':>6} {'DRAM%':>6} {'texhit%':>7} {'inst':>12}"
    print(hdr)
    print("-" * len(hdr))
    for p in passes:
        if not any(h in p["pass"] for h in HOT):
            continue
        print(
            f"{p['pass'][:28]:28} {p.get('gpu_time_us') or 0:8.2f} {str(p.get('bottleneck')):>6} "
            f"{p.get('sm_throughput_pct') or 0:6.1f} {p.get('l1tex_throughput_pct') or 0:6.1f} "
            f"{p.get('dram_throughput_pct') or 0:6.1f} {p.get('tex_hit_rate_pct') or 0:7.1f} "
            f"{int(p.get('inst_executed') or 0):12d}"
        )
    print(f"\nwrote {d / 'perf.json'}")


if __name__ == "__main__":
    main()
