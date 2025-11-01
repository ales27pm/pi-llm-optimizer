#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

jq -r '
  .queue
  | map(select(.status=="todo"))
  | sort_by(-.priority)
  | .[0]
  | {task_id, file} | @json
' .agents/priorities.json
