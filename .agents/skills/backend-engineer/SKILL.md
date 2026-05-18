---
name: backend-engineer
description: Implements backend services, APIs, jobs, validation, authz, data access, and integration logic. Use when changing server behavior, API contracts, domain logic, queues, or service boundaries.
---

# Backend Engineer

Use this for server-side or domain behavior.

## Method

1. Trace the current request/job flow before editing.
2. Define contract changes explicitly: inputs, outputs, errors, auth, idempotency,
   rate limits, and compatibility.
3. Validate at boundaries. Keep domain invariants close to domain logic.
4. Use existing repository data-access and error-handling patterns.
5. Add focused tests for success, validation failure, permission failure, and
   important edge cases.
6. Update docs or API examples when behavior changes.

## Guardrails

- Never trust client-provided authorization fields.
- Avoid ad hoc string parsing when structured parsers exist.
- Do not log secrets, tokens, PII, or full credentials.
- Make retries idempotent or explicitly unsafe.
- Treat background jobs as production APIs with observability and retry behavior.

## Output

Report contracts changed, tests run, migration or deployment notes, and residual
risks.
