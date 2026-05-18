---
name: pr-finalization
description: Prepare a branch for push or PR with status check, diff summary, verification notes, and PR description.
argument-hint: "<target branch or PR goal>"
triggers:
  - user
---

# PR Finalization

Use this skill before pushing, opening a PR, or handing work to a reviewer.

## Steps

1. Run `git status`.
2. Review staged and unstaged changes.
3. Confirm ignored build/cache artifacts are not included.
4. Run appropriate verification or summarize why not possible.
5. Prepare a concise commit message if committing.
6. Prepare PR body:
   - Summary.
   - Key files.
   - Verification.
   - Risks/follow-up.

## Output

Return:

- Branch and commit SHA.
- Push/PR status.
- Verification status.
- PR-ready summary.

Do not force-push unless explicitly asked.
