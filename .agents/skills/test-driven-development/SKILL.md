---
name: test-driven-development
description: Applies test-first development for risky feature work, bug fixes, domain logic, APIs, migrations, and regressions. Use when correctness matters, behavior is unclear, or a previous bug must stay fixed.
---

# Test-Driven Development

Use this when a failing test can define the intended behavior.

## Loop

1. Write the smallest failing test that captures the behavior.
2. Run it and confirm it fails for the expected reason.
3. Implement the smallest code change that passes the test.
4. Run the focused test.
5. Refactor only after green.
6. Run the broader relevant test set.

## Good Tests

- Name the behavior, not the implementation detail.
- Cover one reason to fail.
- Assert externally visible behavior.
- Use realistic fixtures but keep them minimal.
- Include regression tests for bug fixes.

## When Not To Force TDD

- Pure documentation changes.
- Mechanical rename or formatting-only changes.
- Exploratory spikes where the desired behavior is not known yet.

## Output

Report failing test added, implementation files, focused and broader test
results, and any behavior left untested.
