# AGENT_PROTOCOL_V1

version: 1.0.0

## Scope

Covers training utilities and experiment configs inside `pi_llm_training/`.

## Coding Standards

- Prefer composable configuration objects over ad-hoc dictionaries.
- Keep trainer entrypoints deterministic by seeding RNGs and documenting reproducibility controls.
- Guard long-running experiments with timeout-aware retry logic and checkpointing.

## Documentation

- Mirror new training flags in `README.md` and sample configs.

## Testing

- Provide smoke tests that exercise end-to-end flows on synthetic data under `tests/training/`.
