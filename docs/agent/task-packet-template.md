# Agent Task Packet Template

Use this when handing work to Devin, Claude Code, Codex, Cursor, Copilot, or a
human engineer.

```markdown
# Task: <short title>

## Goal
<one sentence user/business outcome>

## Context
- Repo:
- Branch:
- Relevant docs:
- Relevant files:

## Constraints
- Preserve:
- Do not touch:
- Performance/security/product constraints:

## Required Skills
- @skills:codebase-research
- @skills:architecture-plan
- @skills:implementation-agent
- @skills:test-and-verify

## Acceptance Criteria
- [ ] Behavior 1
- [ ] Behavior 2
- [ ] Docs updated if behavior/setup changes
- [ ] Verification command passes

## Verification
Run:

```text
<exact command>
```

Expected:

```text
<expected result>
```

## Final Response
Report:

- Files changed
- Verification run
- Risks
- Follow-up
```
