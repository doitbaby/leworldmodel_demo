---
name: database-migrations
description: Plans and reviews database schema changes, indexes, data migrations, backfills, rollbacks, and compatibility windows. Use when changing persistence, queries, migrations, or production data shape.
---

# Database Migrations

Use this before modifying schema or production data shape.

## Method

1. Identify current schema, query paths, indexes, and data volume assumptions.
2. Choose an expand-migrate-contract plan for risky changes.
3. Keep application code backward-compatible during rollout windows.
4. Separate schema migration from large backfills when possible.
5. Define rollback behavior before writing migration code.
6. Verify with tests, query plans, and representative data.

## Safety Checklist

- Migration is deterministic and re-runnable where the framework expects it.
- Locks, table scans, and index creation cost are considered.
- Data backfill can resume after interruption.
- Rollback or forward-fix plan is documented.
- Application code tolerates old and new schemas during deployment.
- Sensitive data is not copied into logs or fixtures.

## Output

Report migration files, data risks, rollback plan, and verification commands.
