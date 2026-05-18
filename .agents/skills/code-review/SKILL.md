---
name: code-review
description: Review changed code for correctness, regressions, missing tests, maintainability, and risky assumptions.
argument-hint: "<diff, branch, PR, or file list>"
allowed-tools:
  - Read
  - Grep
  - ListDir
  - read
  - grep
  - glob
  - exec
triggers:
  - user
---

# Code Review

Use this skill when the user asks for review, audit, or risk assessment.

## Review Stance

Find bugs first. Do not lead with praise or summary.

## Checklist

Evaluate:

1. Correctness and edge cases.
2. State transitions and lifecycle behavior.
3. Error handling and recovery.
4. Security, secrets, and unsafe inputs.
5. Performance and unbounded work.
6. Test coverage and missing verification.
7. Documentation and operational gaps.

## Output Format

Return:

1. Findings ordered by severity.
2. Each finding must include file path and line reference.
3. Open questions or assumptions.
4. Brief change summary only after findings.
5. Test gaps or residual risks.

If no issues are found, say so clearly and name remaining risks.
