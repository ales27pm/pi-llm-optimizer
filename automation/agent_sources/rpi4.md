# AGENT_PROTOCOL_V1

version: 1.0.0

## Scope

Applies to Raspberry Pi runtime, benchmarking and packaging assets under `rpi4/`.

## Coding Standards

- Keep scripts compatible with Debian Bullseye and Bookworm environments.
- Avoid hard-coding board revisions; detect capabilities at runtime and document fallbacks.
- Shell automation must check for required kernel modules and exit with actionable guidance.

## Performance

- Capture benchmark metadata (device model, kernel, commit hash) in logs to enable historical comparisons.

## Documentation

- Update the Raspberry Pi setup guides when tooling or dependencies change.
