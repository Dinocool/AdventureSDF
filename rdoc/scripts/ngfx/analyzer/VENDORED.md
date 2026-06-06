# Vendored: nsight-graphics-analyzer

This directory is a **vendored copy** of the runtime of
[`FrostyLeaves/nsight-graphics-analyzer`](https://github.com/FrostyLeaves/nsight-graphics-analyzer)
(MIT, see `LICENSE`) — a stdlib-only Python CLI that turns the Nsight Graphics GPU-Trace
auto-export bundle (`BASE/*.xls`) into compact JSON with per-marker bottleneck diagnostics.

- **Upstream commit:** `67a9e5a3f00d5e96687985beb6d0983f830231c9` (2026-05-12)
- **What was copied:** `scripts/nsight.py` + the `scripts/nsight/` package only. Tests and the
  Codex/Claude plugin manifests were dropped — we drive it from the `profile-shaders` skill.
- **No dependencies:** pure Python 3.10+ standard library; nothing to `pip install`.
- **Why vendored, not submoduled:** it's a self-contained analysis tool and the `profile-shaders`
  skill must work in any checkout without a network fetch.

## How it fits our flow

It reads the **same `BASE/*.xls` bundle** our `rdoc/scripts/ngfx/capture.ps1` already auto-exports
(it can't parse the `.ngfx-gputrace` binary — NVIDIA dropped the offline parser in Nsight 2026.1).
So it's a richer sibling of `rdoc/scripts/ngfx/parse.py`: where `parse.py` prints a per-pass table,
this exposes 120+ metrics with per-marker scoping and bottleneck verdicts.

```sh
# after capture.ps1 has produced .soul/ngfx/BASE/*.xls:
NS=rdoc/scripts/ngfx/analyzer/nsight.py
TR=.soul/ngfx/<capture>.ngfx-gputrace
python $NS gputrace            "$TR"                              # → summary/stages/actions JSON
python $NS gputrace-shader-bound "$TR" --in-marker sdf_probe_trace  # SM/occupancy diagnosis for one pass
python $NS gputrace-stalls     "$TR"                              # frame-level idle / pipeline efficiency
python $NS gputrace-metric     "$TR" --name <regex> --in-marker <pass>   # drill any metric, scoped
```

See `.claude/skills/profile-shaders.md` for the full capture→analyze workflow.

## Updating

Re-copy `scripts/nsight.py` + `scripts/nsight/` from a newer upstream commit and bump the SHA above.
Do not edit the vendored files locally — keep the diff to upstream empty so updates stay trivial.
