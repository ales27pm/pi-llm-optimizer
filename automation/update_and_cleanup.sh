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

CURRENT_STEP="initialisation"

on_error() {
  local exit_code=$1
  local line_no=$2
  log "ERROR: Step '${CURRENT_STEP}' failed at line ${line_no} with exit code ${exit_code}."
  exit ${exit_code}
}

trap 'on_error $? ${LINENO}' ERR

declare -a FLAG_ENV_MAP=(
  "SESSION_SYNC_CHECK=--check"
  "SESSION_SYNC_SKIP_FORMATTING=--skip-formatting"
  "SESSION_SYNC_SKIP_AGENT_SYNC=--skip-agent-sync"
  "SESSION_SYNC_SKIP_CLEANUP=--skip-cleanup"
  "SESSION_SYNC_RUN_NPM_LINT=--run-npm-lint"
  "SESSION_SYNC_RUN_PYTEST=--run-pytest"
  "SESSION_SYNC_ENFORCE_MANIFEST=--enforce-manifest"
  "SESSION_SYNC_VERBOSE=--verbose"
)

for mapping in "${FLAG_ENV_MAP[@]}"; do
  IFS="=" read -r env_var flag <<<"${mapping}"
  if [[ "${!env_var:-0}" == "1" ]]; then
    SESSION_SYNC_ARGS+=("${flag}")
  fi
done

SESSION_SYNC_ARGS+=("$@")

run_step() {
  local step_name=$1
  shift
  CURRENT_STEP=${step_name}
  log "Starting ${step_name}..."
  "$@"
  log "Completed ${step_name}."
}

log "automation.session_sync arguments: ${SESSION_SYNC_ARGS[*]}"
run_step "automation.session_sync" python3 -m automation.session_sync "${SESSION_SYNC_ARGS[@]}"

log "Repository refresh complete."
