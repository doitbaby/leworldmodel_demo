# Agent Prompt Library

Reusable prompts for high-signal AI coding work.

## Research Only

```text
Use @skills:codebase-research.
Inspect the codebase and report exact files, current behavior, likely root
cause, risks, and recommended next steps. Do not edit files.
```

## Plan Then Wait

```text
Use @skills:architecture-plan and @skills:fullstack-architect.
Create a concise implementation plan with files, contracts, verification
commands, risks, and fallback. Do not edit yet.
```

## Implement Approved Slice

```text
Use @skills:implementation-agent and @skills:test-and-verify.
Implement only the approved slice, preserve user changes, run focused
verification, and update docs if behavior or setup changes.
```

## Debug Production-Like Bug

```text
Use @skills:systematic-debugging.
Reproduce or reason from evidence, identify the root cause, patch the smallest
cause, and verify the failing scenario.
```

## Fullstack Feature

```text
Use @skills:fullstack-architect, @skills:backend-engineer,
@skills:frontend-engineer, and @skills:test-and-verify.
Design contracts first, implement in small slices, and verify UI/API behavior.
```

## Database Change

```text
Use @skills:database-migrations and @skills:security-review.
Plan expand-migrate-contract, define rollback, implement migration safely, and
verify with representative queries.
```

## Final Review

```text
Use @skills:code-review and @skills:security-review.
Review the latest diff. Lead with findings ordered by severity, then summarize
tests and residual risks.
```
