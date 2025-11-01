#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/.session_sync_venv"
VENV_PYTHON=""
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
  "SESSION_SYNC_SKIP_ROADMAP=--skip-roadmap"
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

resolve_venv_python() {
  if [[ -x "${VENV_DIR}/bin/python3" ]]; then
    VENV_PYTHON="${VENV_DIR}/bin/python3"
  elif [[ -x "${VENV_DIR}/bin/python" ]]; then
    VENV_PYTHON="${VENV_DIR}/bin/python"
  elif [[ -x "${VENV_DIR}/Scripts/python.exe" ]]; then
    VENV_PYTHON="${VENV_DIR}/Scripts/python.exe"
  else
    VENV_PYTHON=""
  fi
}

ensure_session_sync_python() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    run_step "create session_sync virtualenv" python3 -m venv "${VENV_DIR}"
  else
    log "Reusing session sync virtual environment at ${VENV_DIR}."
  fi

  resolve_venv_python
  if [[ -z "${VENV_PYTHON}" ]]; then
    fail "Unable to locate Python executable in ${VENV_DIR}. Remove the directory and re-run the script."
  fi

  run_step "install session_sync dependencies" "${VENV_PYTHON}" -m pip install --disable-pip-version-check --requirement "${SCRIPT_DIR}/requirements.txt"
}

ensure_session_sync_python

run_step "automation.session_sync" "${VENV_PYTHON}" -m automation.session_sync "${SESSION_SYNC_ARGS[@]}"

log "Repository refresh complete."
