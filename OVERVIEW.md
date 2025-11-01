# Overview

This repository is organised into modules. The authoritative machine index lives at `.agents/index.json`.

- Start here for tasks: `.agents/priorities.json`
- Module task lists: `.agents/modules/**/tasks.json`

When you need deeper context, follow the `refs` inside each task entry; they include exact file paths and line spans.

> To choose the next task automatically:
> ```bash
> bash scripts/agent_next_step.sh
> ```
