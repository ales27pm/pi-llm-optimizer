# AGENT_PROTOCOL_V1

version: 1.0.0

## Scope

Applies to the desktop distillation pipeline (training, export and UI glue) under `desktop_distill/`.

## Coding Standards

- Annotate public functions with precise type hints and document expected tensor/device shapes.
- Maintain feature parity between CLI modules and the Textual UI; update usage docs alongside behavioural changes.
- Route long-running operations through cancellable abstractions so the UI remains responsive.

## Error Handling

- Convert exceptions from third-party libraries into domain-specific errors with remediation steps.
- Emit structured logs (JSON or key=value) for progress reporting to integrate with dashboards.

## Testing

- Back new logic with pytest cases in `tests/desktop_distill/`, using fixtures for model artifacts.
- Provide synthetic datasets for deterministic regression coverage.
