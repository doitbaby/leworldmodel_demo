---
name: frontend-engineer
description: Builds production frontend features with accessible UI, state handling, responsive layout, error/loading states, and tests. Use when editing UI components, screens, design systems, client state, or browser behavior.
---

# Frontend Engineer

Use this for user-facing UI work.

## Method

1. Inspect existing components, routes, styling tokens, and interaction patterns.
2. Define user states: empty, loading, error, partial data, success, disabled,
   permission denied, mobile, and keyboard-only.
3. Implement the smallest component/page slice that matches the existing design
   system.
4. Keep business rules out of purely visual components unless the repo already
   follows that pattern.
5. Add or update tests for behavior with real user flows where possible.
6. Verify responsive layout and text overflow for realistic data.

## Checklist

- Accessible labels, focus behavior, and keyboard navigation are covered.
- Controls use appropriate UI affordances.
- Errors are actionable and not swallowed.
- Loading state does not shift layout unnecessarily.
- No hardcoded sample data leaks into production paths.
- Screenshots or browser smoke checks are run for visible UI changes.

## Output

Report changed files, verification, screenshots if relevant, and remaining UI
risks.
