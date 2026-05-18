# Agent Workflow Playbook

This playbook turns AI coding agents into repeatable engineering workers.

## Standard Flow

```text
intake -> research -> plan -> implement -> verify -> review -> docs -> commit/PR
```

## Phase Rules

### Intake

- Restate the goal.
- Identify non-goals and constraints.
- Ask only when a missing fact would make execution risky.

### Research

- Use `codebase-research`.
- Find entry points with fast search.
- Read only enough code to explain the behavior.
- Return facts, inferences, and unknowns separately.

### Plan

- Use `architecture-plan` or `fullstack-architect`.
- List files to edit.
- Define acceptance criteria and verification.
- Prefer small slices that can ship independently.

### Implement

- Use `implementation-agent`.
- Preserve user changes.
- Follow existing patterns.
- Avoid unrelated refactors.

### Verify

- Use `test-and-verify`.
- Run the smallest meaningful check first.
- Expand verification when blast radius is broad.

### Review

- Use `code-review` and `security-review` for risky changes.
- Findings should lead.
- Include file and line references when possible.

### Docs

- Use `docs-sync`.
- Update README, runbook, ADRs, or API docs when behavior/setup changes.

### Finalize

- Use `pr-finalization`.
- Commit only intended files.
- Report changed files, tests, and residual risk.

## Good Agent Prompts

```text
Use @skills:codebase-research. Inspect first and report root cause with file
references. Do not edit yet.
```

```text
Use @skills:architecture-plan and @skills:implementation-agent. Implement only
the approved plan, then use @skills:test-and-verify.
```

```text
Use @skills:code-review and @skills:security-review. Review the latest diff for
runtime bugs, unsafe behavior, and missing tests.
```
