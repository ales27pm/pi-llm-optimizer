#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

jq -e -r '
    .queue
    | map(select(.status=="todo"))
    | sort_by(-.priority)
    | if length == 0 then error("No todo tasks in queue") else .[0] end
    | {task_id, file} | @json
' .agents/priorities.json
