#!/usr/bin/env bash
# Stop hook. Runs the build/clippy gate only when Claude edited .rs files this
# turn, and scopes clippy warnings to just the touched files. A compile error
# always fails (a broken crate is non-negotiable); clippy lints fail only if they
# point at a file Claude actually touched.
set -u
touched=target/.claude-rs-touched
[ -f "$touched" ] || exit 0   # no Rust edited this turn → nothing to gate

# 1. Compile gate — hard fail on any error, regardless of file.
if ! check_out=$(cargo check --all-features 2>&1); then
  printf '%s\n' "$check_out" | tail -40 >&2
  echo 'cargo check failed — fix the build before finishing.' >&2
  exit 2
fi

# 2. Clippy gate — warnings as errors, but only for touched files.
# Normalize backslashes to forward slashes on both sides so Windows-style clippy
# paths (src\foo\bar.rs) match the forward-slash touched list.
clippy_out=$(cargo clippy --all-features --message-format=short 2>&1 | tr '\\' '/')
# Unique, sorted list of files Claude touched (repo-relative, forward slashes).
mapfile -t files < <(sort -u "$touched")

hits=""
for f in "${files[@]}"; do
  [ -n "$f" ] || continue
  # Match "<...>file.rs:line:col: warning|error" anywhere the path ends with $f.
  match=$(printf '%s\n' "$clippy_out" | grep -E "(^|/)${f}:[0-9]+:[0-9]+: (warning|error)" || true)
  [ -n "$match" ] && hits+="$match"$'\n'
done

if [ -n "$hits" ]; then
  printf '%s' "$hits" | tail -40 >&2
  echo 'Clippy issues in files you edited — fix before finishing (zero-warning invariant).' >&2
  exit 2
fi

rm -f "$touched"
exit 0
