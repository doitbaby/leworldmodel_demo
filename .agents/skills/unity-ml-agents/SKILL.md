---
name: unity-ml-agents
description: Work on this Unity ML-Agents/world-model demo with correct scenes, scripts, training commands, and verification habits.
argument-hint: "<Unity AI task>"
triggers:
  - user
  - model
---

# Unity ML-Agents World-Model Demo

Use this skill for tasks involving Unity, ML-Agents, world-model training, AI Brain HUD, scenes, or gameplay behavior.

## Project Facts

- Unity project root: current repository root.
- Main demo scene: `Assets/Scenes/WorldModelPlannerRoom.unity`.
- Training scene: `Assets/Scenes/TrainingRoom.unity`.
- UI file: `Assets/UI/GameUI.uxml`.
- Core AI scripts:
  - `Assets/Scripts/ML/WorldModelPlannerAgent.cs`
  - `Assets/Scripts/ML/BrainPlanner.cs`
  - `Assets/Scripts/ML/BrainHUDController.cs`
  - `Assets/Scripts/ML/RogueObservationBuilder.cs`
  - `Assets/Scripts/ML/RogueTransitionRecorder.cs`
  - `Assets/Scripts/ML/WorldModelWeights.cs`
- Python training script: `tools/world_model/train_world_model.py`.
- Exported model files:
  - `Assets/StreamingAssets/world_model_weights.json`
  - `Assets/StreamingAssets/world_model_metrics.json`

## Engineering Rules

1. Do not edit Unity cache folders: `Library`, `Temp`, `Logs`, `UserSettings`, `results`, `.venv_mla`.
2. Keep model demo stable before making it more research-pure.
3. Use vector observations first; do not introduce pixel encoders unless explicitly requested.
4. For AI behavior bugs, inspect planner score, map state, movement rules, and food/tick lifecycle.
5. If Play mode is running, remind the user to Stop/Play again after script/UI changes.

## Verification

Use relevant checks:

```powershell
python -m py_compile tools/world_model/train_world_model.py
python tools/world_model/train_world_model.py --smoke --epochs 2
```

Unity batchmode compile when the Editor is not locking the project:

```powershell
& "C:\Program Files\Unity\Hub\Editor\6000.4.6f1\Editor\Unity.exe" -batchmode -quit -projectPath "." -logFile "Logs\unity-batch.log"
```

If batchmode cannot run because the Editor is open, inspect:

```powershell
$env:LOCALAPPDATA\Unity\Editor\Editor.log
```

## Demo Goal

The live demo should show:

- AI toggle works.
- Agent moves through levels.
- HUD shows selected action, ranking, imagined futures, metrics, and explanation.
- Planner optimizes safe exit progress, food efficiency, and risk avoidance.
