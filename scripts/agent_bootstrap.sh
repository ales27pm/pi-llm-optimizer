#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[bootstrap] Ensuring dependencies..."
if ! command -v jq >/dev/null 2>&1; then
  echo "[ERR] 'jq' is not installed. Please install it to continue." >&2
  echo "       Debian/Ubuntu: sudo apt-get install jq" >&2
  echo "       macOS (Homebrew): brew install jq" >&2
  exit 1
fi

if ! python3 -c "import jsonschema" >/dev/null 2>&1; then
  echo "[ERR] Python package 'jsonschema' is required. Install it with 'pip install jsonschema'." >&2
  exit 1
fi

mkdir -p ".agents/schemas" ".agents/modules/core" "scripts"

# Ensure schemas exist (idempotent write if missing)
write_if_missing() {
  local path="$1"; shift
  if [ ! -f "$path" ]; then
    printf "%s" "$*" > "$path"
    echo "[bootstrap] created $path"
  fi
}

# Touch machine plane if missing
write_if_missing ".agents/index.json" '{"$schema":".agents/schemas/index.schema.json","version":1,"generated_at":"bootstrap","modules":[{"name":"core","path":"src/core","tasks_file":".agents/modules/core/tasks.json","docs":[{"file":"OVERVIEW.md"},{"file":"VISION.md"}]}],"docs":[{"file":"VISION.md"},{"file":"OVERVIEW.md"}]}'
write_if_missing ".agents/priorities.json" '{"$schema":".agents/schemas/priorities.schema.json","version":1,"updated_at":"bootstrap","policy":{"strategy":"critical_path_first","tie_breakers":["dependency_depth","risk","value"]},"queue":[{"task_id":"core:bootstrap-structure","title":"Establish validated agent control plane and schemas","file":".agents/modules/core/tasks.json","priority":100,"status":"todo"}]}'
write_if_missing ".agents/modules/core/tasks.json" '{"$schema":".agents/schemas/tasks.schema.json","module":"core","updated_at":"bootstrap","tasks":[]}'

# Seed schemas if missing (content already provided in repo files)
# Validate current JSONs
python3 scripts/agent_scan.py --validate-only

echo "[bootstrap] OK"
