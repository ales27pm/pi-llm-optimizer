#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CURRENT_STEP="initialisation"

log() {
  printf '[update_and_cleanup] %s\n' "$1"
}

fail() {
  log "ERROR: $1"
  exit 1
}

trap 'log "ERROR: Script failed during ${CURRENT_STEP:-unknown step}"' ERR

CURRENT_STEP="markdown discovery"
log "Synchronizing markdown documentation (including roadmap and agent protocols)..."
MARKDOWN_FILES=()
while IFS= read -r file; do
  MARKDOWN_FILES+=("$file")
done < <(git ls-files '*.md' | sort)

if [[ ${#MARKDOWN_FILES[@]} -gt 0 ]]; then
  CURRENT_STEP="prettier availability"
  if ! command -v npx >/dev/null 2>&1; then
    fail "Node.js tooling (npx) is required to format markdown files. Install Node.js or run via CI tooling."
  fi
  if ! npx --yes prettier --version >/dev/null 2>&1; then
    fail "Prettier is required to format markdown files. Install it globally or add it to the project dependencies."
  fi

  CURRENT_STEP="markdown formatting"
  npx --yes prettier --log-level warn --write "${MARKDOWN_FILES[@]}" >/dev/null
fi

CURRENT_STEP="python cache cleanup"
log "Clearing Python cache directories..."
find . -type d -name '__pycache__' -delete

CURRENT_STEP="temporary file cleanup"
log "Removing stray temporary files..."
find . -type f \( -name '*.tmp' -o -name '*~' -o -name '.DS_Store' \) -delete

CURRENT_STEP="completed"
log "Completed repository refresh."
