#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

log() {
  printf '[update_and_cleanup] %s\n' "$1"
}

log "Synchronizing markdown documentation (including roadmap and agent protocols)..."
mapfile -t MARKDOWN_FILES < <(git ls-files '*.md' | sort)
if [[ ${#MARKDOWN_FILES[@]} -gt 0 ]]; then
  npx --yes prettier --log-level warn --write "${MARKDOWN_FILES[@]}" >/dev/null
fi

log "Clearing Python cache directories..."
find . -type d -name '__pycache__' -prune -exec rm -rf {} +

log "Removing stray temporary files..."
find . -type f \( -name '*.tmp' -o -name '*~' -o -name '.DS_Store' \) -delete

log "Completed repository refresh."
