# LeWorldModel port (`tools/lewm/`)

This directory hosts the in-tree port of the upstream **LeWorldModel** paper
into the rogue demo. The full motivation and architecture is in
[`docs/research/lewm-port-analysis.md`](../../docs/research/lewm-port-analysis.md);
this README is the day-to-day usage guide.

> Upstream source: <https://github.com/lucas-maes/le-wm>
> Upstream commit pinned: `c8a44170b22dfe15b0b1924e2845da97af926c42`
> Upstream license: MIT (see [`LICENSE`](https://github.com/lucas-maes/le-wm/blob/main/LICENSE)).

## Status

| Milestone | PR | Status |
| --- | --- | --- |
| M0 — research / port analysis | #1 | landed (see `docs/research/lewm-port-analysis.md`). |
| M1 — vendor modules + smoke training | #1 | landed. |
| M2 — Unity pixel observation builder + JSONL v3 | #2 | landed. |
| M3 — Python inference sidecar | #4 | landed (sidecar + CLI + Unity client). |
| M4 — training pipeline (LR schedule + val split + metrics CSV + best-ckpt) | #5 | landed. |
| M5 — Dreamer-style actor on imagined rollouts | #6 | landed. |
| M6 — benchmark harness (5-mode comparison + CSV metrics) | this PR | landed. |
| M7 — CI + pre-commit | TBD | not started. |

## Layout

```
tools/lewm/
├── __init__.py
├── module.py     # vendored upstream layers (SIGReg, ARPredictor, ...)
├── encoder.py    # TinyConvEncoder for 32x32 board + VectorEncoder for 31-d obs
├── jepa.py       # JEPA wrapper with reward / done heads
├── data.py       # synthetic + v2/v3 JSONL datasets (vector + board-pixel)
├── train.py      # PyTorch-only training entry point
├── schedule.py   # cosine + warmup LR schedules (M4)
├── planner.py    # random-shooting actor over JEPA rollouts (M5)
├── sidecar.py    # FastAPI app wrapping JEPA inference + planning (M3+M5)
├── serve.py      # CLI: load checkpoint and boot uvicorn (M3)
├── env.py        # python-side rogue env mirroring Unity rules (M6)
├── benchmark.py  # multi-mode evaluation CLI emitting per-episode + summary CSVs (M6)
├── tests/
│   ├── test_board_jsonl.py        # round-trip check for the v3 board JSONL path
│   ├── test_sidecar.py            # in-process TestClient integration (incl. /plan_actions)
│   ├── test_training_pipeline.py  # M4 end-to-end (train → metrics CSV → best.pt → sidecar)
│   ├── test_planner.py            # M5 random-shooting actor unit test
│   ├── test_env.py                # M6 RogueSimEnv step-dynamics smoke (9 cases)
│   └── test_benchmark.py          # M6 benchmark CLI smoke across all 5 modes (6 cases)
├── requirements.txt
└── README.md     # you are here
```

## Setup

```bash
python -m venv .venv_lewm
source .venv_lewm/bin/activate    # PowerShell: .\.venv_lewm\Scripts\Activate.ps1
pip install -r tools/lewm/requirements.txt
```

The vendored modules only depend on `torch`, `einops` and `numpy`; the full
upstream Lightning / Hydra / `stable-pretraining` / `stable-worldmodel` stack
is **not** required.

## Smoke training (CPU, synthetic data)

The default mode renders fake 32×32 rogue boards and trains the small CNN
encoder + AR predictor + reward/done heads for two epochs:

```bash
python -m tools.lewm.train --smoke
```

Expected output (timings vary):

```
epoch=001 lr=3.00e-04 train_loss=... train_pred=... train_reward=... train_done=... epoch_time_s=...
epoch=002 lr=3.00e-04 train_loss=... train_pred=... train_reward=... train_done=... epoch_time_s=...
saved metrics csv: results/lewm/metrics.csv
saved checkpoint (final): results/lewm/checkpoint.pt
saved metrics:           results/lewm/checkpoint.metrics.json
```

The checkpoint, `metrics.json` and per-epoch `metrics.csv` are written under
`results/lewm/` (already ignored by `.gitignore`).

## Vector-mode training (existing JSONL transitions)

The existing `RogueTransitionRecorder` writes vector transitions. The JEPA can
be trained over those files via `VectorEncoder`, regardless of schema version:

```bash
python -m tools.lewm.train \
    --observation-mode vector \
    --jsonl-path "$HOME/.config/unity3d/.../rogue_transitions.jsonl" \
    --epochs 5 \
    --batch-size 64
```

Vector mode is a useful sanity check while iterating on the LeWM losses.

## Pixel-mode training from Unity gameplay (v3 board JSONL)

From M2 onward, `RogueTransitionRecorder` emits schema `rogue.transition.v3`
with the new `board_state` / `next_board_state` integer arrays produced by
`Assets/Scripts/ML/PixelObservationBuilder.cs`. The Python side renders these
to a deterministic RGB tile image on the fly, so JEPA training can use real
Unity gameplay data without ever capturing a `RenderTexture`:

```bash
python -m tools.lewm.train \
    --observation-mode board-jsonl \
    --jsonl-path "$HOME/.config/unity3d/.../rogue_transitions.jsonl" \
    --epochs 5 \
    --batch-size 64
```

For a longer non-smoke run with proper validation and the cosine
schedule M4 ships (see [Training with validation and metrics (M4)](#training-with-validation-and-metrics-m4)
below) use:

```bash
python -m tools.lewm.train \
    --observation-mode board-jsonl \
    --jsonl-path "$HOME/.config/unity3d/.../rogue_transitions.jsonl" \
    --epochs 30 --batch-size 64 \
    --val-split 0.1 \
    --lr-schedule cosine --warmup-steps 200
```

Cell-code conventions (must match `PixelObservationBuilder.cs`):

| Code | Meaning | Colour (`ROGUE_CELL_COLORS`) |
| --- | --- | --- |
| `-1` | wall | dark grey |
| `0` | empty passable | near-white |
| `1` | exit | green |
| `2` | enemy | red |
| `3` | obstacle | brown |
| `4` | food | yellow |
| `5` | player (overlaid) | blue |

To sanity-check the v3 schema without Unity:

```bash
python -m tools.lewm.tests.test_board_jsonl
# OK: BoardJsonlDataset round-trip passed.
```

## Architecture summary

```
obs (B,T,3,H,W)
   │ TinyConvEncoder (or VectorEncoder)        ── projector ──► emb (B,T,D)
                                                                       │
                  action (B,T-1,A) ── Embedder ─────────► act_emb (B,T-1,D)
                                                                       │
                                                          ARPredictor (T-1 steps)
                                                                       │
                                                          pred_emb (B,T-1,D)
                                                            │           │
                                                            ▼           ▼
                                                       RewardHead   DoneHead
                                                       (B,T-1)      (B,T-1 logits)
```

Loss = `pred_mse + λ·SIGReg(emb) + α·reward_mse + β·done_bce` with the same
`λ = 0.09` default as upstream and `α = β = 1.0` defaults from the existing
MLP trainer.

## Configuration knobs that matter

| Flag | Meaning | Sensible range |
| --- | --- | --- |
| `--embed-dim` | hidden width of the JEPA latent | 64 (smoke) → 192 (full) |
| `--predictor-depth` | AR transformer layers | 1 → 6 |
| `--sequence-length` | frames per training sample | 3 → 5 |
| `--sigreg-weight` | regularizer strength | 0.05 → 0.2 |
| `--reward-loss-weight` | reward MSE weight | 0.5 → 2.0 |
| `--done-loss-weight` | termination BCE weight | 0.5 → 2.0 |
| `--items-per-epoch` | synthetic episodes per epoch | 32 (smoke) → 2048 |
| `--lr-schedule` | `none` or `cosine` (with warmup) | `cosine` for non-smoke runs |
| `--warmup-steps` | linear warmup steps before cosine decay | 0 (smoke) → ~5 % of total |
| `--min-lr-ratio` | floor of the cosine schedule (× base LR) | 0.0 → 0.1 |
| `--val-split` | fraction of JSONL episodes held out for val | 0.0 → 0.2 |
| `--val-items-per-epoch` | synthetic-mode val budget | 0 → 256 |

## Training with validation and metrics (M4)

M4 layers a real training loop on top of the M1 scaffold:

* **Cosine LR schedule with linear warmup** (`--lr-schedule cosine
  --warmup-steps N --min-lr-ratio R`) matches the upstream recipe. Use
  `--lr-schedule none` to keep a constant LR (the default for `--smoke`).
* **Train/val split** over JSONL episodes. `--val-split 0.1` holds out
  10 % of the *episodes* (not transitions) for validation, so windows
  from a training episode cannot leak into the val set. Synthetic-mode
  validation uses `--val-items-per-epoch` instead.
* **Per-epoch metrics CSV** at `--metrics-csv results/lewm/metrics.csv`
  with columns `epoch, lr, epoch_time_s, train_loss, train_pred_loss,
  train_sigreg_loss, train_reward_loss, train_done_loss` (+ matching
  `val_*` columns when validation is enabled). The M6 benchmark harness
  consumes this CSV directly.
* **Best-checkpoint tracking**. Whenever `val_loss` improves, the
  current weights are written to `--best-output` (default
  `results/lewm/best.pt`). The final-epoch checkpoint always goes to
  `--output` (default `results/lewm/checkpoint.pt`). Both are loadable
  by the M3 sidecar; the sidecar uses
  `train.load_config_from_payload` to stay backward-compatible with
  M1/M3 checkpoints that lack the new config fields.

Full example against a real Unity gameplay JSONL:

```bash
python -m tools.lewm.train \
    --observation-mode board-jsonl \
    --jsonl-path "$HOME/.config/unity3d/.../rogue_transitions.jsonl" \
    --epochs 30 --batch-size 64 \
    --val-split 0.1 --lr-schedule cosine \
    --warmup-steps 200 --min-lr-ratio 0.05 \
    --metrics-csv results/lewm/metrics.csv \
    --output results/lewm/checkpoint.pt \
    --best-output results/lewm/best.pt
```

## Inference sidecar (M3 + M5)

Unity cannot run PyTorch / JEPA in-engine. M3 added a thin Python
sidecar that wraps a trained checkpoint, and M5 added the
`/plan_actions` endpoint that lets Unity outsource the action search
entirely:

```
GET  /healthz          → { "status": "ok" | "no_model", "service": ... }
GET  /info             → { embed_dim, action_dim, image_size,
                          sequence_length, max_horizon, service }
POST /score_actions    → { scores: [float, ...], horizon, service }
POST /plan_actions     → { best_actions, best_score,
                          top_k_actions_flat, top_k_scores,
                          num_candidates, top_k, horizon, service }   (M5)
```

Boot it like this once `python -m tools.lewm.train` has produced a
checkpoint:

```bash
python -m tools.lewm.serve \
    --checkpoint results/lewm/checkpoint.pt \
    --host 127.0.0.1 --port 5555
```

The Unity side talks to the sidecar via
[`Assets/Scripts/ML/LewmClient.cs`](../../Assets/Scripts/ML/LewmClient.cs)
— a small `MonoBehaviour` wrapping `UnityWebRequest`. From M5 onward,
`WorldModelPlannerAgent` queries `POST /plan_actions` as the **primary**
scorer (see [Model-first action selection (M5)](#model-first-action-selection-m5))
and only falls back to the mission heuristic when the sidecar is
unreachable.

The flat action layouts (`action_sequences_flat` in `/score_actions`,
`top_k_actions_flat` in `/plan_actions`) are a JsonUtility compatibility
quirk: Unity's JsonUtility does not deserialize nested arrays. The
sidecar reshapes internally before calling
`JEPA.score_action_sequences`, and the Unity client
`LewmClient.PlanResponse.GetTopKAction(k, t)` inflates the wire format
back into a 2-D view.

## Model-first action selection (M5)

From M5 onward, the world model is the **primary** decision maker, not
a tiebreaker. The flow is:

1. `WorldModelPlannerAgent.Update` snapshots the current board state
   (via `PixelObservationBuilder.BuildCellCodes`).
2. It sends `POST /plan_actions` to the sidecar with
   `{ board_state, horizon, num_candidates, top_k, ... }`.
3. The sidecar samples `num_candidates` random action sequences,
   scores each through `JEPA.score_action_sequences`, and returns the
   best plan plus the top-k for the HUD.
4. `BrainPlanner.DecideWithSidecarPlan` translates the plan into a
   per-action ranking: the model score dominates, the map heuristic
   only hard-rejects blocked moves, and a small first-step safety bias
   (`SidecarSafetyWeight = 0.5`) keeps the agent from walking adjacent
   into an enemy when an equally-scored alternative exists.
5. The brain HUD renders the top-k plans as "imagined futures".

When the sidecar is offline or unreachable, the agent automatically
falls back to the existing `BrainPlanner.Decide` (mission heuristic +
local LeWM-lite tiebreaker) so the demo keeps working without Python.

### Trying it locally

1. Train (or smoke-train) a checkpoint:
   ```bash
   python -m tools.lewm.train --smoke
   ```
2. Start the sidecar:
   ```bash
   python -m tools.lewm.serve --checkpoint results/lewm/checkpoint.pt
   ```
3. Open Unity, select the `WorldModelPlannerAgent` GameObject, and in
   the inspector toggle:
   - **Use Sidecar** = on
   - **Sidecar Base Url** = `http://127.0.0.1:5555` (default)
   - **Sidecar Candidates** = 64 (raise for stronger planning at the
     cost of latency)
   - **Sidecar Top K** = 3 (drives the HUD's imagined-futures panel)
4. Enter Play mode and press **M** to give control to the agent. The
   BrainHUD mode label switches to **WORLD MODEL SIDECAR (Nx H)**.

### Direct planner usage (Python)

The planner is also callable directly without HTTP, useful for
batched evaluation or benchmarks:

```python
from tools.lewm.planner import random_shooting
plan = random_shooting(
    model, current_obs,
    horizon=horizon, num_candidates=128,
    action_dim=4, top_k=5, seed=0,
)
print(plan.best_actions, plan.best_score)
```

## Benchmark harness (M6)

M6 ships a CLI that evaluates several planners against a small
rogue-style env that mirrors the Unity game's cell-code grid and step
dynamics (see [`env.py`](env.py)). The benchmark writes a per-episode
CSV plus a `<name>.summary.csv` companion with per-mode aggregates.

Five modes are exposed via `--modes`:

| Mode | Picks actions via | Notes |
| --- | --- | --- |
| `random` | uniform random | baseline floor. |
| `mission` | BFS toward the exit + enemy-adjacency avoidance | mirrors the in-engine `BrainPlanner.ChooseMissionTarget` heuristic with no learned model. |
| `mlp_lite` | `mission` + adjacent-food preference when `Assets/StreamingAssets/world_model_weights.json` exists | surrogate for the in-engine LeWM-lite path. The real MLP needs the 31-d vector observation only Unity builds, so this surrogate uses the same scaffolding but rule-based action selection. |
| `lewm_no_planner` | 1-step JEPA argmax (no rollout) | requires `--checkpoint`. |
| `lewm_dreamer` | M5 random-shooting actor over JEPA rollouts | requires `--checkpoint`. |

Smoke run (one episode of every mode, ~10s on CPU):

```bash
python -m tools.lewm.train --smoke                              # produces results/lewm/checkpoint.pt
python -m tools.lewm.benchmark --smoke \
    --checkpoint results/lewm/checkpoint.pt \
    --output-csv results/lewm/benchmark.csv
```

Longer run (5 episodes per mode, full max-steps budget):

```bash
python -m tools.lewm.benchmark \
    --episodes 5 --max-steps 200 --seed 0 \
    --checkpoint results/lewm/best.pt \
    --modes random mission mlp_lite lewm_no_planner lewm_dreamer \
    --output-csv results/lewm/benchmark.csv
```

Per-episode CSV columns (`results/lewm/benchmark.csv`):

```
seed, mode, episode_idx, levels_cleared, food_left, steps,
episode_return, dynamics_loss, reward_loss, died
```

Summary CSV columns (`results/lewm/benchmark.summary.csv`):

```
seed, mode, episodes, mean_levels_cleared, mean_food_left, mean_steps,
mean_episode_return, mean_dynamics_loss, mean_reward_loss,
death_rate, std_episode_return
```

`dynamics_loss` and `reward_loss` are recorded whenever a JEPA
checkpoint is loaded (so even `random` / `mission` rows get a finite
model-prediction error against the trajectories *those* policies
traced out). When the benchmark runs without `--checkpoint`, only the
three non-JEPA modes can be requested and the loss columns are `nan`.

## Verification

Eight cheap checks cover the LeWM port end-to-end without GPU or Unity:

```bash
python -m tools.lewm.train --smoke                       # synthetic smoke training
python -m tools.lewm.tests.test_board_jsonl              # v3 JSONL round-trip
python -m tools.lewm.tests.test_sidecar                  # FastAPI sidecar integration (incl. /plan_actions)
python -m tools.lewm.tests.test_training_pipeline        # M4 full pipeline
python -m tools.lewm.tests.test_planner                  # M5 random-shooting actor unit test
python -m tools.lewm.tests.test_env                      # M6 RogueSimEnv step-dynamics smoke
python -m tools.lewm.tests.test_benchmark                # M6 benchmark CLI smoke
python tools/world_model/train_world_model.py --smoke --epochs 2  # legacy MLP path stays untouched
```

The smoke train exercises encoder, Embedder, ARPredictor, RewardHead,
DoneHead, SIGReg, optimiser, checkpoint save. The round-trip test confirms
that the JSONL written by `RogueTransitionRecorder.cs` (schema v3) can be
read back and rendered into the exact `(T, 3, H, W)` pixel tensors the
training loop expects. The sidecar test trains a tiny checkpoint, loads
it via `load_checkpoint`, then drives `/healthz`, `/info`, `/score_actions`
and `/plan_actions` (M5) through Starlette's `TestClient` — including
bad-payload error paths. The M4 pipeline test generates a synthetic v3
JSONL, drives `tools.lewm.train` through its CLI with `--val-split` +
cosine schedule + metrics CSV, then re-opens the resulting `best.pt`
through the sidecar's `load_checkpoint`. The M5 planner test trains a
smoke checkpoint and asserts that random shooting (1) returns a sorted
top-k with the best plan at index 0, (2) is deterministic at a fixed
seed, and (3) honours the optional `include_sequences` forcing path.
The M6 env test covers the cell-code conventions, wall bumps, food
pickup, exit transitions (which regenerate the board), starvation,
and max-step termination. The M6 benchmark test trains a smoke
checkpoint, runs all five modes for one episode each, and asserts the
per-episode CSV header, the summary CSV header, finite `dynamics_loss`
when a checkpoint is loaded, and the expected error paths
(`ValueError` when JEPA mode is requested without a checkpoint,
`FileNotFoundError` for a missing checkpoint path).

The existing `tools/world_model/train_world_model.py` MLP path is untouched
and continues to support the live Unity demo.
