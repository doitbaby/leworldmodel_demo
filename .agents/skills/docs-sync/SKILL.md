---
name: docs-sync
description: Update README, runbooks, ADRs, or demo docs so they match the current implementation.
argument-hint: "<changed feature or doc target>"
triggers:
  - user
  - model
---

# Docs Sync

Use this skill after behavior, setup, architecture, commands, or demo flow changes.

## Process

1. Identify what changed in code/config.
2. Locate the relevant docs.
3. Update only docs that users need.
4. Keep commands copy-pasteable.
5. Include exact file paths, scene names, script names, and expected outputs.
6. Remove stale claims.

## Output

Return:

- Docs changed.
- Important new/changed instructions.
- Any docs intentionally left unchanged.

Prefer concise, operational docs over broad marketing language.
