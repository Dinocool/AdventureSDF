---
name: verify-build
description: The pre-done verification gate for the adventure project — mirrors CI so local passing means CI passes. Trigger before reporting any code change complete, before committing, or when asked to check/verify the build.
---

# Verify Build

Run this before claiming a change is done. It mirrors `.github/workflows/ci.yaml`
(check, test, clippy, fmt) plus the both-feature-configs invariant. If any step fails,
the change is NOT done.

## The checks

```sh
# 1. Builds clean in BOTH feature configs (zero warnings — project invariant)
cargo build
cargo build --features debug_toolkit

# 2. All tests green
cargo test

# 3. Clippy with warnings as errors (CI uses -D warnings)
cargo clippy --all-features -- -D warnings

# 4. Formatting matches
cargo fmt --all -- --check
```

## Notes

- **Zero warnings is a hard invariant.** A warning is a failure — fix it, don't report
  done. (Memory: `no-warnings`.)
- **Both feature configs must build.** A feature-gated change can compile with the
  feature on and break with it off (or vice versa). Always build both.
- Hooks (`.claude/settings.json` → `.claude/hooks/`) help, but don't replace this:
  after each `.rs` edit, `post-edit-rs.sh` runs `cargo fmt` and records the touched
  file; at end of turn `stop-gate.sh` runs only if Rust was edited — it hard-fails on
  any `cargo check` error, and fails on clippy warnings **only in the files you
  touched** (pre-existing debt elsewhere won't block you). This skill is the **fuller**
  gate (whole-crate clippy + `cargo test` + both-config builds) — run it when finishing
  real work; the scoped hook is a guardrail, not the full check.
- Shaders: if you touched any `.wgsl`, `cargo test` already runs the
  `shader_validation` rig that parses every shader. Watch for its failures.

## Related

- `/add-feature` Step 8 — where this gate fits in the workflow.
