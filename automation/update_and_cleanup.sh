#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

log() {
  printf '[update_and_cleanup] %s\n' "$1"
}

fail() {
  log "ERROR: $1"
  exit 1
}

if ! command -v python3 >/dev/null 2>&1; then
  fail "Python 3 is required to run the session synchronisation workflow."
fi

SESSION_SYNC_ARGS=("--repo-root" "${REPO_ROOT}")

if [[ "${SESSION_SYNC_CHECK:-0}" == "1" ]]; then
  SESSION_SYNC_ARGS+=("--check")
fi
if [[ "${SESSION_SYNC_SKIP_FORMATTING:-0}" == "1" ]]; then
  SESSION_SYNC_ARGS+=("--skip-formatting")
fi
if [[ "${SESSION_SYNC_SKIP_AGENT_SYNC:-0}" == "1" ]]; then
  SESSION_SYNC_ARGS+=("--skip-agent-sync")
fi
if [[ "${SESSION_SYNC_SKIP_CLEANUP:-0}" == "1" ]]; then
  SESSION_SYNC_ARGS+=("--skip-cleanup")
fi
if [[ "${SESSION_SYNC_RUN_NPM_LINT:-0}" == "1" ]]; then
  SESSION_SYNC_ARGS+=("--run-npm-lint")
fi
if [[ "${SESSION_SYNC_RUN_PYTEST:-0}" == "1" ]]; then
  SESSION_SYNC_ARGS+=("--run-pytest")
fi
if [[ "${SESSION_SYNC_ENFORCE_MANIFEST:-0}" == "1" ]]; then
  SESSION_SYNC_ARGS+=("--enforce-manifest")
fi
if [[ "${SESSION_SYNC_VERBOSE:-0}" == "1" ]]; then
  SESSION_SYNC_ARGS+=("--verbose")
fi

SESSION_SYNC_ARGS+=("$@")

log "Executing automation.session_sync with arguments: ${SESSION_SYNC_ARGS[*]}"
python3 -m automation.session_sync "${SESSION_SYNC_ARGS[@]}"
log "Repository refresh complete."
