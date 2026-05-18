---
name: systematic-debugging
description: Debug failures by reproducing, tracing root cause, patching narrowly, and verifying the fix.
argument-hint: "<bug, symptom, log, or screenshot>"
triggers:
  - user
  - model
---

# Systematic Debugging

Use this skill for bugs, regressions, hangs, crashes, failed tests, or confusing runtime behavior.

## Rule

Do not guess-patch. Find evidence first.

## Process

1. Capture the symptom precisely: expected vs actual behavior.
2. Reproduce or inspect logs/screenshots if reproduction is not available.
3. Trace the path from user action to failing state.
4. Form 2-4 hypotheses and eliminate them with evidence.
5. Identify the smallest root cause.
6. Patch at the root cause, not only the visible symptom.
7. Add a guard/test/log only when it prevents recurrence.
8. Verify the original scenario and a nearby edge case.

## Output

Return:

- Root cause in plain language.
- Evidence with file paths/lines/logs.
- Patch summary.
- Verification result.

If a live app/editor is open, explain when the user must restart, reload, or re-enter Play mode.
