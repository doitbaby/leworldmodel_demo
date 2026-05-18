---
name: architecture-plan
description: Turn a feature or research goal into a practical implementation plan with files, risks, tests, and fallback path.
argument-hint: "<feature or technical goal>"
allowed-tools:
  - Read
  - Grep
  - ListDir
  - read
  - grep
  - glob
triggers:
  - user
  - model
---

# Architecture Plan

Use this skill before medium or large implementation work.

## Goal

Design a practical plan for: `$ARGUMENTS`

Prefer the repository's existing architecture over new frameworks. Optimize for a working, demonstrable implementation.

## Steps

1. Confirm the intended user outcome and demo acceptance criteria.
2. Inspect existing modules, boundaries, and extension points.
3. Propose a minimal architecture using current patterns.
4. Define data contracts, file ownership, and integration points.
5. Identify risks and fallback options.
6. Define verification steps.

## Output

Return:

- One-paragraph summary.
- Architecture/data-flow diagram in text or Mermaid if useful.
- File list to create or modify.
- Implementation order.
- Test/verification plan.
- Fallback plan if the ideal path fails.

Avoid theory unless it changes the implementation decision.
