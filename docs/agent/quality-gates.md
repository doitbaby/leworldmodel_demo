# Agent Quality Gates

Use these gates before accepting AI-generated code.

## Gate 1: Scope

- The change matches the user request.
- No unrelated refactors.
- No generated/cache files committed accidentally.
- User edits are preserved.

## Gate 2: Correctness

- The implementation follows current repo patterns.
- Edge cases from the task are covered.
- Errors are handled at the boundary.
- Data/API contracts are explicit.

## Gate 3: Tests

- Focused test or smoke check was run.
- Broader checks were run when shared behavior changed.
- Any skipped tests are explained.

## Gate 4: Security

- No secrets in code, logs, configs, or docs.
- Auth and authorization are checked server-side.
- Shell/file operations avoid unsafe path handling.
- New dependencies are justified.

## Gate 5: Operations

- Env vars and setup changes are documented.
- Migrations have rollback or forward-fix notes.
- CI/deploy changes are observable.

## Gate 6: Final Report

The final response includes:

- Files changed.
- Commands run and results.
- Risks or test gaps.
- Next action, if needed.
