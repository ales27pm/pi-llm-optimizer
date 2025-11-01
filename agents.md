# agents.md — Root Contract for Coding Agents

> **Purpose.** This repository is agent-operated. This file is the single source of truth for how agents (e.g., Codex) must behave.

## Non-Negotiables (Immutable)
1. **No placeholders or partial code.** Do not produce examples, stubs, TODOs-instead-of-code, mock functions, or “left as an exercise”. Implement fully, robustly, and safely.
2. **End-of-session update is mandatory.** Before opening any PR or pushing any branch, you **must** run:
   ```bash
   bash scripts/agent_post_session_update.sh
   ```
   The push/PR must be aborted if this script fails.
3. Machine-readable control plane. Internal state lives under /.agents/ (JSON only). Human docs live in README.md, VISION.md, OVERVIEW.md, and module READMEs.
4. Link everything. When a task needs context elsewhere, include precise cross-file references: {"file":"path", "line":N, "end_line":M, "anchor":"#heading"}.
5. Idempotence. All scripts and generated artifacts must be repeatable. No brittle, one-shot steps.

## First Run (Bootstrap)

On first contact in a repo:
1.Read VISION.md (top intent) and OVERVIEW.md (structure).
2.Execute:

```bash
bash scripts/agent_bootstrap.sh
```

This performs a deep scan (scripts/agent_scan.py), builds .agents/index.json, initialises .agents/priorities.json, seeds a modules/*/tasks.json for discovered modules, and validates all JSON against schemas.

## Session Loop (Every Time You Code)
1.Plan: Read .agents/index.json and .agents/priorities.json. Select next task using:

```bash
bash scripts/agent_next_step.sh
```

This returns the canonical {task_id, file} to execute next.

2.Execute: Implement the task fully (no placeholders). Update or create supporting docs as needed.
3.Write back context:
•Append decisions and rationale to the relevant task’s notes[].
•If new cross-module knowledge emerges, add refs[] with exact file+line spans.
•If new sub-work appears, split it into atomic tasks with clear acceptance criteria.
4.Housekeeping (MANDATORY):

```bash
bash scripts/agent_post_session_update.sh
```

This validates JSON, refreshes indices, updates ROADMAP snapshots, commits, and pushes. If it fails, fix and rerun. Do not open a PR until it passes.

## File Hierarchy
•VISION.md — immutable high-level goals, guardrails, and anti-drift constraints.
•OVERVIEW.md — current shape of the system, module map, and links to task files.
•.agents/index.json — machine index of modules, docs, and link targets.
•.agents/priorities.json — dynamic, ordered list of tasks (critical-path first).
•.agents/modules/*/tasks.json — atomic, verifiable tasks per module.
•scripts/ — the only allowed operational scripts. They must remain robust and idempotent.

## Quality Gates (before marking a task done)
•Deterministic acceptance tests (unit/integration) exist and pass.
•Performance/complexity is appropriate for expected scale.
•Security considerations addressed (inputs validated, secrets not hard-coded, least privilege).
•Observability (logs/metrics) and failure handling implemented where relevant.

## Notes for Future You (Self-Memory)
•At the end of each task, append an actionable next_notes[] with hints you would have wanted before starting the next task.
•If priorities changed during your session, update .agents/priorities.json using the scripts. Never leave the repo in a state that requires implicit memory.

Compliance with this contract is required for any PR authored by an agent.
