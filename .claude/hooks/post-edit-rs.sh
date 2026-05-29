#!/usr/bin/env bash
# PostToolUse hook (Edit|Write|MultiEdit). Reads the tool-call JSON on stdin.
# If a .rs file was edited: format it, and record the path so the Stop hook can
# scope its clippy gate to only the files Claude touched this turn.
set -u
input=$(cat)

# Extract the edited file path from the tool input JSON.
path=$(printf '%s' "$input" | grep -oE '"file_path"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed -E 's/.*:[[:space:]]*"(.*)"/\1/')

case "$path" in
  *.rs)
    cargo fmt 2>/dev/null
    mkdir -p target
    # Normalize to a repo-relative, forward-slash path for later matching.
    # tr handles backslashes (Windows-sed chokes on '\\' in a regex); then sed
    # collapses doubled slashes and strips everything before src/ or tests/.
    rel=$(printf '%s' "$path" | tr '\\' '/' | sed -E 's#/+#/#g; s#^.*/(src/|tests/)#\1#')
    printf '%s\n' "$rel" >> target/.claude-rs-touched
    ;;
esac
exit 0
