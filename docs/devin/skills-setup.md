# Devin Skills Setup

This repository includes a repo-local skill pack for Devin and other coding
agents that understand the open `SKILL.md` format.

## Why This Exists

Devin works best when recurring engineering workflows are written down in the
repo. The official Devin docs say repo skills live at:

```text
.agents/skills/<skill-name>/SKILL.md
```

Devin automatically discovers these files across connected repositories. This
repo follows that layout so a new Devin session starts with project-specific
procedures instead of rediscovering them from scratch.

## Installed Skills

```text
.agents/skills/
  architecture-plan/
  code-review/
  codebase-research/
  docs-sync/
  implementation-agent/
  pr-finalization/
  security-review/
  systematic-debugging/
  test-and-verify/
  unity-ml-agents/
```

Use these as composable work modes:

- `codebase-research`: read-only investigation with exact file references.
- `architecture-plan`: turn a feature request into a scoped plan.
- `implementation-agent`: make focused edits while preserving user changes.
- `systematic-debugging`: reproduce, isolate root cause, patch, verify.
- `test-and-verify`: choose and run the right build/test checks.
- `code-review`: findings-first review for bugs and regressions.
- `security-review`: check secrets, injection, dangerous scripts, and auth risk.
- `docs-sync`: keep README, runbooks, and architecture notes in sync.
- `pr-finalization`: prepare clean commits or pull requests.
- `unity-ml-agents`: Unity and ML-Agents workflow for this project.

## Source Selection

This setup uses high-signal public skill ecosystems as design references, then
keeps this repo's skills as clean-room project-specific procedures instead of
copying third-party prompt packs.

Useful reference ecosystems:

- [Devin Skills docs](https://docs.devin.ai/product-guides/skills): canonical
  `.agents/skills/<name>/SKILL.md` location.
- [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills):
  lifecycle pattern from spec to plan to build, verify, review, and ship.
- [Awesome Agent Skills](https://www.awesomeskills.dev/en): public index used
  to identify large installable skill collections such as `vercel-labs`,
  `openai`, and `antfu`.
- `alirezarezvani/claude-skills`, `chriscox/agent-skills`,
  `tech-leads-club/agent-skills`, and `gotalab/cc-sdd`: broad coding-agent
  workflow packs worth reviewing before manual adoption.

Security rule: do not blindly vendor third-party skills into this repo. Skills
can instruct an agent to run commands, inspect secrets, or change files. Read
them first, then copy only the safe workflow ideas that match this codebase.

## How To Use In Devin

1. Connect or refresh this GitHub repo in Devin.
2. Start a new Devin session on the repo.
3. Ask Devin to use one or more skills by name.

Recommended first prompt for hard tasks:

```text
Use @skills:codebase-research and @skills:architecture-plan.
Inspect the codebase first, then propose a concise plan with files to edit,
verification commands, and risks. Do not edit yet.
```

Implementation prompt:

```text
Use @skills:implementation-agent, @skills:test-and-verify, and @skills:docs-sync.
Implement the approved plan, run the smallest meaningful verification, update
docs if behavior changes, then prepare a commit summary.
```

Debugging prompt:

```text
Use @skills:systematic-debugging.
Reproduce the bug, collect evidence, identify the root cause, patch only the
root cause, and verify the exact scenario.
```

Review prompt:

```text
Use @skills:code-review and @skills:security-review.
Review the latest diff for runtime bugs, Unity asset churn, unsafe scripts,
secret leaks, and missing verification.
```

## Best Operating Pattern

For maximum Devin throughput, split work by phase:

1. Research session: ask for findings only.
2. Planning session: ask for a small plan and acceptance criteria.
3. Implementation session: approve one slice at a time.
4. Verification session: ask Devin to run checks and summarize logs.
5. Finalization session: ask for commit or PR notes.

For this Unity project, do not ask Devin to train forever. Prefer bounded jobs:

```text
Train smoke model for 2 epochs.
Run synthetic model for 60 epochs.
Evaluate 5 fixed seeds.
Record steps per level and food remaining.
```

## Repo-Specific Devin Recipes

AI toggle bug:

```text
Use @skills:systematic-debugging and @skills:unity-ml-agents.
Investigate why pressing M enables AI for a while, then the agent stops after
level transitions. Verify with WorldModelPlannerRoom and report exact root cause.
```

Improve AI performance:

```text
Use @skills:architecture-plan and @skills:implementation-agent.
Add a finite benchmark mode for 10 levels, track steps per level, food spent,
combat count, and success rate. Keep the live demo stable.
```

World-model training:

```text
Use @skills:unity-ml-agents and @skills:test-and-verify.
Train or smoke-test tools/world_model/train_world_model.py, export weights and
metrics to Assets/StreamingAssets, and explain whether Unity loaded them.
```

Research documentation:

```text
Use @skills:docs-sync.
Update the README and docs/demo runbook so the project clearly states
LeWorldModel-lite uses vector observations and short-horizon planning.
```

## When External Skills Are Worth Installing

Keep this repo-local skill pack as the default. Add external packs only when a
task needs a domain this repo does not cover:

- Frontend or Vercel deployment: consider `vercel-labs/agent-skills`.
- General coding lifecycle: consider `addyosmani/agent-skills`.
- OpenAI API work: consider `openai/skills`.
- Security-heavy work: review security skills manually before enabling them.

Do not install a large public skill pack into this repo just because it has many
stars. Large packs increase prompt surface area and can conflict with the
project-specific workflow in `AGENTS.md`.
