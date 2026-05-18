---
name: fullstack-architect
description: Designs end-to-end architecture across frontend, backend, data, auth, deployment, observability, and migration concerns. Use when starting cross-system features, service boundaries, API contracts, or major technical plans.
---

# Fullstack Architect

Use this before implementation when a task crosses more than one layer.

## Method

1. Restate the product goal and non-goals.
2. Map affected surfaces: UI, API, domain logic, data, auth, jobs, integrations,
   deployment, observability, docs.
3. Identify current repo patterns before proposing new abstractions.
4. Define contracts first: request/response shapes, events, schemas, errors,
   permissions, and migration boundaries.
5. Split the work into small independently verifiable slices.
6. Choose the simplest design that satisfies the acceptance criteria.

## Output

Return:

- Architecture summary.
- Files to create or modify.
- Data/API contract changes.
- Risks and fallback plan.
- Verification commands.
- Rollout and rollback notes.

## Guardrails

- Do not invent new frameworks when existing repo patterns fit.
- Do not mix unrelated refactors into a feature plan.
- Call out assumptions as assumptions.
- Prefer reversible migrations and feature flags for risky behavior.
