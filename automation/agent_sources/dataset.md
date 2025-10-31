# AGENT_PROTOCOL_V1

version: 1.0.0

## Scope

Covers dataset blueprints, preprocessing scripts and corpus documentation in `dataset/`.

## Data Integrity

- Treat JSONL and YAML schemas as source of truth; validate before writing derived artifacts.
- Never commit raw proprietary data—only synthetic or open datasets with accompanying licenses.

## Tooling

- Keep scripts idempotent and safe for repeated execution on the same workspace.
- Document dataset generation steps in the relevant README or dataset card whenever behaviour changes.

## Testing

- Add schema regression tests in `tests/dataset/` to catch format drift.
