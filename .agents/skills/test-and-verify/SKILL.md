---
name: test-and-verify
description: Discover and run the right verification steps for the current change, then report pass/fail evidence.
argument-hint: "<change or area to verify>"
triggers:
  - user
  - model
---

# Test And Verify

Use this skill before marking work complete.

## Process

1. Inspect package/build/test files to discover the correct commands.
2. Prefer targeted checks first, then broader checks when risk warrants.
3. For Unity projects, use batchmode compile when available and inspect Editor logs if the project is open.
4. For Python scripts, run `python -m py_compile` and any smoke commands.
5. For UI/game changes, include a manual smoke checklist if automated checks are insufficient.

## Output

Return:

- Commands run.
- Pass/fail result.
- Important log lines or error summaries.
- What was not verified and why.
- Recommended next verification if needed.

Never claim success without evidence.
