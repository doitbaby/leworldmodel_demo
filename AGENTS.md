# Agent Operating Guide

This repository is a Unity research demo. Agents should optimize for a live,
explainable demo rather than a theoretical rewrite.

## Default Workflow

Use the skills in `.agents/skills` when they match the task.

Recommended task flow:

1. `codebase-research` - understand the relevant files and behavior.
2. `architecture-plan` or `fullstack-architect` - produce a small plan.
3. `implementation-agent` - make focused code changes.
4. `systematic-debugging` - use when behavior is broken or intermittent.
5. `test-driven-development` - use when a failing test can define the change.
6. `test-and-verify` - run the smallest meaningful verification.
7. `code-review` and `security-review` - review risky changes before finalizing.
8. `docs-sync` - update README or runbooks when behavior or setup changes.
9. `pr-finalization` - prepare commit or pull request notes.

Do not skip investigation for Unity scene, asset, or generated-file behavior.

## Project Facts

- Unity version: `6000.4.6f1`.
- Main demo scene: `Assets/Scenes/WorldModelPlannerRoom.unity`.
- Training scene: `Assets/Scenes/TrainingRoom.unity`.
- Core AI files live in `Assets/Scripts/ML/`.
- Brain HUD UI lives in `Assets/UI/GameUI.uxml`.
- World model trainer: `tools/world_model/train_world_model.py`.
- Runtime model outputs:
  - `Assets/StreamingAssets/world_model_weights.json`
  - `Assets/StreamingAssets/world_model_metrics.json`
- Generated metrics and training artifacts can appear in `results/`.

## Demo Goal

The demo should show LeWorldModel-lite:

- Vector observation world model, not full pixel-JEPA.
- AI toggle between manual and AI control.
- Action ranking for available moves.
- Imagined rollouts over 3 to 5 steps.
- Learning metrics in the HUD.
- Plain-language explanation for the selected action.
- Stable fallback if learned weights are missing or weak.

## Safety Rules

- Preserve user changes. Check `git status --short` before editing.
- Do not use destructive git commands unless the user explicitly asks.
- Do not force-push.
- Do not commit Unity cache or local machine folders.
- Avoid editing:
  - `Library/`
  - `Temp/`
  - `Logs/`
  - `UserSettings/`
  - `.venv_mla/`
  - `results/`, unless the task is specifically about generated results.
- Use focused edits. Avoid unrelated refactors.
- Do not rewrite scene files unless necessary for the task.
- Prefer code and UXML changes over fragile manual scene edits when possible.
- For reusable workflow improvements, update `.agents/skills`, `docs/agent`,
  and tool adapters together.

## Unity Verification

When Unity compile or smoke verification is requested, prefer:

```powershell
& "C:\Program Files\Unity\Hub\Editor\6000.4.6f1\Editor\Unity.exe" -batchmode -quit -projectPath . -logFile Logs\agent-compile.log
```

If the editor is already open, avoid launching competing Unity instances unless
the user asks for it. Inspect `Logs/agent-compile.log` after batch runs.

## Python Verification

For world-model script changes, run the smallest useful check first:

```powershell
python -m py_compile tools/world_model/train_world_model.py
python tools/world_model/train_world_model.py --smoke --epochs 2
```

Use real training data only when the task asks for training quality, metrics, or
model export.

## Good Devin Prompts

Use explicit skill names in Devin to reduce wandering:

```text
Use @skills:codebase-research to inspect why AI toggle stops after level changes.
Return exact files and root cause before editing.
```

```text
Use @skills:systematic-debugging and fix the smallest cause.
Verify AI toggle, HUD update, and one restart flow.
```

```text
Use @skills:implementation-agent and @skills:test-and-verify.
Implement a finite benchmark mode for 10 levels, then update docs.
```

```text
Use @skills:code-review.
Review the latest diff for Unity runtime bugs, generated-file churn, and missing tests.
```

## Portable Agent Kit

This repo includes a reusable agent engineering kit:

- `.agents/skills/` - portable skills for Devin and compatible agents.
- `docs/agent/` - task packets, workflow playbook, quality gates, MCP policy,
  and prompt library.
- `.github/copilot-instructions.md` - GitHub Copilot adapter.
- `.cursor/rules/` - Cursor adapter.
- `CLAUDE.md` - Claude Code adapter.
- `scripts/validate-agent-kit.ps1` - validates skill frontmatter and required docs.
- `scripts/install-agent-kit.ps1` - copies the kit into another repository.

## Final Response Standard

Every completed agent task should report:

- Files changed.
- Verification commands and results.
- Any residual risk.
- Exact follow-up needed, if any.
