#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

abort() { echo "[post-session][ERR] $*" >&2; exit 1; }

echo "[post-session] Validating tooling..."
command -v jq >/dev/null || abort "jq is required"
command -v python3 >/dev/null || abort "python3 is required"
git rev-parse --is-inside-work-tree >/dev/null || abort "Not a git repo"
git fetch --all --quiet || true

echo "[post-session] Running repo scan + validation..."
python3 scripts/agent_scan.py --refresh-index --validate || abort "scan/validate failed"

echo "[post-session] Snapshotting ROADMAP (OVERVIEW.md pointers)..."
# (Optional) generate a machine snapshot for CI consumption
jq '.' .agents/index.json > .agents/index.pretty.json

echo "[post-session] Git add/commit..."
git add -A

if ! git diff --cached --quiet; then
  TS="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  git commit -m "agent: post-session update @ ${TS}" -m "Runbook: scripts/agent_post_session_update.sh"
else
  echo "[post-session] No changes to commit."
fi

# Push with basic retry
RETRIES=3
for i in $(seq 1 $RETRIES); do
  if git push; then
    echo "[post-session] Pushed successfully."
    exit 0
  fi
  echo "[post-session] Push failed (attempt $i/$RETRIES)."

  STATUS_OUTPUT="$(git status -sb || true)"
  if printf '%s\n' "$STATUS_OUTPUT" | grep -q "\[behind"; then
    echo "[post-session] Remote has new commits. Attempting 'git pull --rebase --autostash'..."
    if git pull --rebase --autostash; then
      echo "[post-session] Rebase completed. Retrying push immediately..."
      continue
    else
      abort "Automatic rebase failed. Resolve conflicts and rerun the script."
    fi
  fi

  echo "[post-session] Retrying in 3s..."
  sleep 3
done

abort "Push failed after ${RETRIES} attempts."
