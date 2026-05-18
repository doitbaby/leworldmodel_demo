---
name: security-review
description: Security-focused review for secrets, unsafe permissions, injection risks, supply-chain risk, and agent prompt-injection hazards.
argument-hint: "<diff, repo, feature, or deployment path>"
triggers:
  - user
  - model
---

# Security Review

Use this skill for security-sensitive changes, public repos, deployments, auth, scripts, CI, or AI-agent configuration.

## Review Areas

1. Secrets, tokens, keys, credentials, and generated artifacts.
2. Unsafe shell commands and path handling.
3. Dependency and package supply-chain risk.
4. Authn/authz and access control.
5. Input validation, injection, deserialization, and file upload paths.
6. Logging of sensitive data.
7. CI/CD permissions and GitHub token scope.
8. Agent skills, prompts, MCP tools, and prompt-injection surfaces.

## Output

Return:

- High-risk findings first with file/line references.
- Exploit scenario in one sentence when useful.
- Recommended fix.
- Whether the issue blocks release.

Do not ask for secrets. Do not print secrets if found; redact them.
