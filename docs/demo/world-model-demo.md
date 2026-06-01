# World-Model Demo Runbook

This project implements the Unity demo path from the research report:

1. run the original `2DRogueTest` game,
2. train a PPO baseline with Unity ML-Agents,
3. record transitions from the PPO/random agent,
4. train a small next-state/reward predictor,
5. run a live Unity planner that scores candidate actions with the exported predictor,
6. show an AI Coach HUD with action ranking, imagined futures, learning metrics, compliance, risk warning, and explanation text.

## Tooling

- Unity Editor: `6000.4.6f1` on this machine.
- ML-Agents Unity package: local package at `../ml-agents/com.unity.ml-agents`, checked out at `release_23_tag` for Unity `6000.4.x` compatibility.
- Python: use `3.10.12` for ML-Agents.

Create the Python environment from the Unity project root:

```powershell
uv venv --python 3.10.12 .venv_mla
.\.venv_mla\Scripts\Activate.ps1
python -m pip install --upgrade pip
uv pip install --python .\.venv_mla\Scripts\python.exe ..\ml-agents\ml-agents-envs ..\ml-agents\ml-agents tensorboard torch
```

## Unity Scene Setup

Open the project in Unity, then run:

```text
World Model Demo > Create All Demo Scenes
```

This creates:

- `Assets/Scenes/TrainingRoom.unity` for PPO training and transition logging.
- `Assets/Scenes/WorldModelPlannerRoom.unity` for live predictor-driven planning.

The generated training scene adds `RogueAgent`, `RogueTransitionRecorder`, `Behavior Parameters`, and `Decision Requester` to `PlayerCharacter`.

The planner scene starts in `HUMAN` mode. Press the top-right mode button, or press `M`, to cycle `HUMAN -> COACH -> AI -> HUMAN`. After game over, click `NEW RUN` or press `R` to restart without leaving Play mode.

The left-side `AI BRAIN HUD` is the presentation cockpit:

- `Action Ranking` shows all four actions sorted by predicted future score.
- `Imagined Futures` shows the top short rollouts, usually 3 steps.
- `metrics` shows the last exported dynamics loss, reward loss, transition count, and success proxy.
- The explanation label states why the selected action won.
- In `COACH` mode the panel also shows the suggested move, compliance tracker, and first-step risk warning.

## PPO Baseline

From the project root:

```powershell
.\.venv_mla\Scripts\Activate.ps1
mlagents-learn config/rogue_ppo.yaml --run-id=rogue_baseline --time-scale=20
```

Then press Play in Unity with `TrainingRoom.unity` open.

Open TensorBoard:

```powershell
tensorboard --logdir results
```

## Trajectory Recording

`RogueTransitionRecorder` writes JSONL to:

```text
%USERPROFILE%\AppData\LocalLow\<CompanyName>\<ProductName>\WorldModelDemo\rogue_transitions.jsonl
```

Each row contains:

- `obs`: 31-float observation vector,
- `action`: discrete action id,
- `reward`: scalar reward,
- `next_obs`: 31-float next observation vector,
- `done`: episode terminal flag.

`WorldModelPlannerRoom` records planner transitions in both `COACH` and `AI` modes, and `COACH` mode also writes separate coach-session JSONL analytics for compliance review.

## Train Predictor

Copy or reference the JSONL path, then run:

```powershell
.\.venv_mla\Scripts\Activate.ps1
python tools/world_model/train_world_model.py --input "PATH\TO\rogue_transitions.jsonl" --epochs 30
```

The script writes:

- `Assets/StreamingAssets/world_model_weights.json`
- `results/world_model_metrics.csv`
- `Assets/StreamingAssets/world_model_metrics.json`

For a quick tooling smoke test without Unity data:

```powershell
python tools/world_model/train_world_model.py --smoke --epochs 2
```

For a self-contained demo model before collecting Unity trajectories:

```powershell
python tools/world_model/train_world_model.py --synthetic-game --synthetic-game-states 6000 --epochs 60 --hidden-size 96 --batch-size 256 --learning-rate 0.002 --reward-loss-weight 8
```

## Live Planner

Open `Assets/Scenes/WorldModelPlannerRoom.unity` and press Play.

`WorldModelPlannerAgent` loads `Assets/StreamingAssets/world_model_weights.json` and `Assets/StreamingAssets/world_model_metrics.json`. If weights are not present or invalid, it falls back to the mission planner so the scene remains demonstrable while you debug the learned model.

The live objective is:

```text
maximize survival-adjusted progress =
  shortest safe path to exit
  + food detours when food is low or the detour is cheap
  - enemy/obstacle/adjacent-enemy risk
  + small world-model predicted reward check
```

This means the agent is trying to reach the exit with fewer turns, preserve food, avoid combat, and collect useful food. The game does not currently have a finite final boss or ending; success is measured by how many levels it clears before food reaches zero.

Demo flow:

1. Start the scene and move manually with WASD or arrow keys in `HUMAN`.
2. Click the mode button or press `M` once to enter `COACH`.
3. Keep playing manually while the HUD recommends moves, ranks alternatives, and updates compliance.
4. Click the mode button or press `M` again to enter `AI` autonomous mode.
5. Press `M` once more to return to `HUMAN`.
6. If the run ends, click `NEW RUN` or press `R`.

The current shipped demo weights were trained from generated Rogue-like transitions as a stable fallback. To make the demo stronger, record real Unity transitions in `TrainingRoom.unity`, then retrain with `--input`.

## Learning Loop

The AI does not update neural weights automatically every time you press Play. During Play, Unity can record transitions:

```text
obs, action, reward, next_obs, done
```

The model becomes smarter only after you run the offline training script again and export new JSON weights:

```powershell
python tools/world_model/train_world_model.py --input "PATH\TO\rogue_transitions.jsonl" --epochs 30
```

For the live demo, the safest workflow is:

1. use the mission planner for stable one-run gameplay,
2. record AI/human/PPO trajectories,
3. retrain the world model offline,
4. reload the planner scene so the HUD uses the new metrics and weights.

You do not need to leave the game open for hours for this version. A few minutes of recorded runs plus synthetic data is enough for a demo. Longer training helps the world-model metrics, but live reliability comes mainly from the mission planner and safety costs.

## Presentation Framing

Use this wording:

```text
LeWorldModel is the research inspiration. This Unity implementation is LeWorldModel-lite: it uses vector observations instead of pixel-JEPA so the demo can run live, but it preserves the core idea that the agent predicts future states/rewards before choosing an action.
```

Short talk track:

```text
This is LeWorldModel-lite. The original LeWorldModel learns a pixel-based latent world model. This demo keeps the same model-based idea but uses vector observations for live stability. In Coach Mode the player still moves, while the model ranks all four actions, imagines short futures, warns about immediate risk, and explains which move it would choose. The Brain HUD exposes that internal loop so we can see why one direction is recommended over another.
```
