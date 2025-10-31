# AGENT_PROTOCOL_V1
version: 1.0.0

## 1. Scope Matrix
1.1 This file governs the entire repository except where a nested `AGENTS.md` overrides these directives.
1.2 When a subsystem requires bespoke rules, create a nested `AGENTS.md` alongside the code it governs and keep its instructions synchronized with this root contract.

## 2. Delivery Contract
2.1 Ship complete, production-grade solutions. Partial implementations, placeholders, scaffolds, mocks, TODO markers, or simplified logic are categorically forbidden.
2.2 Preserve behavioural accuracy: align code, documentation, configuration, and tests with the current repository state in the same change set.
2.3 Every code path must include robust error handling, telemetry/logging hooks, and deterministic fallbacks where practical.
2.4 Maintain cross-platform compatibility already supported by the project. If a change is platform-specific, document and guard it explicitly.

## 3. Documentation & Roadmap Synchronization
3.1 Whenever behaviour, interfaces, or processes change, update all impacted documentation immediately (`README.md`, `ROADMAP.md`, dataset cards, runbooks, and any nested guides).
3.2 Keep this root `AGENTS.md` current. Amend it whenever new constraints, workflows, or tooling emerge.
3.3 Expand the roadmap into actionable items that reflect shipped and planned work. Remove or revise stale entries.
3.4 If a documentation source becomes obsolete, either delete it or rewrite it to match reality within the same commit.

## 4. Session Workflow
4.1 Before proposing or merging code:
    a. Implement the change completely.
    b. Update documentation and roadmap artifacts to mirror the new state.
    c. Run relevant unit/integration tests and linters defined for the touched areas.
    d. Execute `./automation/update_and_cleanup.sh` to format documentation, sweep caches, and validate synchronization.
4.2 Record the executed commands and their outcomes in the session summary.
4.3 Never rely on manual follow-up—each session must leave the repository consistent and releasable.

## 5. Quality Gates
5.1 Honour repository-specific tooling (formatters, lint configurations, test harnesses). Add or update checks if coverage is missing.
5.2 Reject work that introduces flaky behaviour, nondeterministic ordering, or hidden side effects.
5.3 For any new capability, include automated tests or clearly state and justify the absence of feasible coverage.
5.4 Enforce secure defaults and least-privilege configurations. Document security considerations alongside the implementation.

## 6. Change Control
6.1 Keep commit histories clean, scoped, and well described. Each commit must be self-contained and pass the quality gates.
6.2 When refactoring, provide migration notes and ensure backwards compatibility or document breaking changes explicitly.
6.3 Proactively remove dead code, redundant configuration, and unused assets encountered while working on a task. This mandate
    applies to all changes submitted after adopting this protocol.

## 7. Automation Maintenance
7.1 Ensure `automation/update_and_cleanup.sh` remains accurate. Extend it whenever new documentation or cache locations appear.
7.2 Validate that automation can run non-interactively in CI environments.
7.3 When adding new automation, document invocation patterns and integrate them into this protocol.
7.4 Declare nested agent protocols in `automation/agents_manifest.json`. The update-and-cleanup workflow keeps the generated
    `AGENTS.md` files in sync; modify the manifest instead of editing nested files directly.

## 8. Enforcement
8.1 Any contribution that violates these directives must be revised before it can be accepted.
8.2 Automated or human reviewers should block changes that do not meet the delivery contract, documentation synchronization requirements, or session workflow steps.
