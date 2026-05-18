---
name: devops-release
description: Handles CI, deployment, environment configuration, release checks, rollback, observability, and production readiness. Use when changing pipelines, deployment files, secrets, runtime config, or release process.
---

# DevOps Release

Use this for release and operations work.

## Method

1. Inspect existing CI/CD, environment variables, secrets references, and runtime
   topology.
2. Identify what changes at build time, deploy time, and run time.
3. Keep changes reversible and observable.
4. Add validation steps to CI when they prevent repeat failures.
5. Document required manual actions, owner, and rollback.

## Checklist

- No secrets are committed.
- Required env vars are documented with safe example values.
- CI commands match local verification commands where possible.
- Deployment has a rollback path.
- Health checks, logs, or metrics reveal whether the change works.
- Generated artifacts are ignored unless intentionally versioned.

## Output

Report pipeline changes, verification, deployment order, rollback, and open
operational risks.
