# AGENT_PROTOCOL_V1

version: 1.0.0

## Scope

This protocol governs the `automation/` toolchain including Textual UI code, pipeline helpers and shell utilities.

## Python Modules

- Target Python 3.10+ syntax and typing features.
- Prefer `logging` over `print` for runtime diagnostics; structure messages for CI readability.
- Every CLI must guard entrypoints with `if __name__ == "__main__"` and surface actionable error messages.
- Keep shell-outs sandboxed: validate inputs, check return codes, and document external tool dependencies in module docstrings.
- Keep Textual UI controls for QLoRA presets aligned with `desktop_distill.train_student.QLORA_PRESETS`, ensuring selection logs describe the applied quantisation parameters.

## Shell Scripts

- Use `set -euo pipefail` and trap fatal exits.
- Accept configuration through environment variables or flags so automation remains composable.

## Tests

- Add unit coverage under `tests/automation/` when feasible.
- Mock filesystem and subprocess interactions to keep tests deterministic.
