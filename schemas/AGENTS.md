# AGENT_PROTOCOL_V1

version: 1.0.0

## Scope

Applies to JSON Schema definitions and validation helpers in `schemas/`.

## Guidelines

- Ensure schemas remain backward compatible unless the major version changes.
- Include `$id`, `$schema` and description metadata for discoverability.
- Provide example payloads demonstrating valid and invalid cases.
- Update associated validator utilities and tests whenever schema fields change.
