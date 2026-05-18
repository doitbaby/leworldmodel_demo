# Repository Instructions For AI Coding Agents

Follow `AGENTS.md` first. It is the canonical operating manual for this
repository.

Use the repo-local skills in `.agents/skills/` as reusable workflows:

- Research before editing: `codebase-research`.
- Plan cross-system work: `architecture-plan` or `fullstack-architect`.
- Implement focused slices: `implementation-agent`.
- Debug with evidence: `systematic-debugging`.
- Verify before final response: `test-and-verify`.
- Review risky diffs: `code-review` and `security-review`.
- Keep docs updated: `docs-sync`.

For this Unity project:

- Main scene: `Assets/Scenes/WorldModelPlannerRoom.unity`.
- Core AI scripts: `Assets/Scripts/ML/`.
- Python trainer: `tools/world_model/train_world_model.py`.
- Do not edit or commit Unity cache folders such as `Library/`, `Temp/`,
  `Logs/`, or `UserSettings/`.

Before finishing, report changed files, commands run, results, and remaining
risks.
