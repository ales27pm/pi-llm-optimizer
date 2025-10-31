# AGENT_PROTOCOL_V1

version: 1.0.0

## Scope

Governs all assets under `tests/`.

## Guidelines

- Structure modules to mirror the source tree hierarchy.
- Use pytest style with fixtures for shared setup; keep tests deterministic and hermetic.
- Mark integration tests with `@pytest.mark.integration` and document required resources.
- Favour realistic assertions that validate behaviour rather than implementation details.
