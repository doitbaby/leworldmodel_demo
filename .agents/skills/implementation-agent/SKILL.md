---
name: implementation-agent
description: Implement an approved plan end-to-end with focused edits, verification, and a concise completion report.
argument-hint: "<approved implementation goal>"
triggers:
  - user
---

# Implementation Agent

Use this skill when the user wants code changes made now.

## Working Rules

1. Check `git status` before editing.
2. Preserve user changes. Never reset, revert, or overwrite unrelated work.
3. Read the relevant code before patching.
4. Keep edits scoped to the requested behavior.
5. Prefer existing helpers and local conventions.
6. Update docs/config only when behavior or run steps change.
7. Run targeted verification before finalizing.

## Execution Flow

1. Restate the concrete outcome in one sentence.
2. Identify the files to edit.
3. Make the smallest coherent patch.
4. Run relevant tests, builds, compiles, or smoke checks.
5. Fix issues found during verification.
6. Report changed files and verification result.

## Completion Report

Return:

- What changed.
- Where it changed.
- How it was verified.
- Any remaining risk or manual follow-up.

Do not stop at a proposal if the user asked for implementation.
