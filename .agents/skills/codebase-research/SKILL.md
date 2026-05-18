---
name: codebase-research
description: Read-only codebase investigation with file references, call flow, risks, and recommended next steps.
argument-hint: "<area, bug, feature, or question>"
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

# Codebase Research

Use this skill when the task requires understanding an unfamiliar area before changing code.

## Scope

Investigate: `$ARGUMENTS`

Do not edit files. Do not run destructive commands. Prefer fast search first, then read the most relevant files.

## Method

1. Identify likely entry points using filename, symbol, and text search.
2. Read the smallest set of files that explains the behavior.
3. Trace control flow, data flow, configuration, and runtime dependencies.
4. Check tests, docs, build scripts, and scene/config files when relevant.
5. Separate facts from inferences.

## Output

Return:

- Key findings with exact file paths and line references.
- Current behavior and why it happens.
- Important risks, edge cases, and unknowns.
- Recommended next action.

If the user asked for implementation, end with a concise plan and proceed only after the investigation is sufficient.
