# LeWorldModel Upstream Port Analysis (M0)

> Source upstream: <https://github.com/lucas-maes/le-wm>
> Upstream commit pinned: `c8a44170b22dfe15b0b1924e2845da97af926c42`
> License: MIT — vendoring permitted with attribution.

This document is the deliverable for **Milestone 0** of the LeWorldModel
integration: a feasibility analysis of porting the upstream **LeWorldModel**
(JEPA pixel world model) into the `doitbaby/leworldmodel_demo` Unity game,
plus a concrete decomposition of the work into milestones M1–M7.

It supersedes the design assumption in
[`docs/architecture/adr-001-world-model-demo-boundary.md`](../architecture/adr-001-world-model-demo-boundary.md):
the original ADR-001 explicitly stayed inside "LeWorldModel-lite" (vector MLP)
to keep the demo stable. After explicit user direction we are now porting the
real JEPA pixel architecture, with the constraints described below.

---

## 1. Upstream LeWorldModel anatomy

The upstream repo is small — five Python files plus Hydra configs:

| File | Purpose |
| --- | --- |
| `jepa.py` | `JEPA` module: encoder + AR predictor + projector, plus `encode`, `predict`, `rollout`, `criterion`, `get_cost`. |
| `module.py` | `SIGReg`, `Attention`, `ConditionalBlock`, `Block`, `Transformer`, `Embedder`, `MLP`, `ARPredictor`. Pure PyTorch. |
| `train.py` | Hydra + Lightning training loop, `lejepa_forward`, calls `stable_pretraining` + `stable_worldmodel`. |
| `eval.py` | Hydra + `stable_worldmodel.World` evaluation harness with CEM / Adam solvers. |
| `utils.py` | Image preprocessing, z-score normalizer, checkpoint callback. |

### 1.1 Model architecture

- **Encoder**: HuggingFace ViT-tiny via `stable_pretraining.backbone.utils.vit_hf`,
  `patch_size=14`, `image_size=224`. Returns CLS token of dimension `embed_dim=192`.
- **Predictor**: `module.ARPredictor` — 6-layer Transformer with
  `ConditionalBlock` (AdaLN-zero conditioning on actions), 16 heads,
  `mlp_dim=2048`, `dim_head=64`, `dropout=0.1`, hidden=192.
- **Action encoder**: `module.Embedder` — `Conv1d(input_dim → smoothed_dim)`
  followed by a 2-layer MLP producing an action embedding of `embed_dim`.
  `input_dim = frameskip × dataset_action_dim`. Continuous actions.
- **Projector / Pred-proj**: 2-layer MLPs with `BatchNorm1d`.

Parameter count: ~15M (paper Section 4 / README).

### 1.2 Loss

LeWM has a single tunable hyperparameter (the headline contribution).

```
loss = pred_loss + λ * sigreg_loss
pred_loss = MSE(predicted_next_emb, target_next_emb)
sigreg_loss = SIGReg(emb)           # Sketch Isotropic Gaussian Regularizer
λ = 0.09 (config/train/lewm.yaml)
```

`SIGReg` is the key novel piece — it sketches random projections of the
embedding distribution and matches their characteristic function against the
isotropic Gaussian via the Epps–Pulley statistic. It prevents representation
collapse without exponential moving averages or pretrained encoders.

### 1.3 Training stack

- PyTorch + PyTorch Lightning + Hydra + OmegaConf + WandB + einops
- Heavy deps: `stable-pretraining`, `stable-worldmodel` (galilai-group)
- Data: HDF5 datasets (`pusht_expert_train.h5`, etc.) hosted on HuggingFace.
- Optimizer: AdamW lr=5e-5, weight_decay=1e-3, cosine warmup
- Hardware: GPU (`accelerator: gpu`, `precision: bf16`), "a few hours on a single GPU"

### 1.4 Planning / evaluation

`eval.py` runs Model-Predictive Control over the learned latent. The world
exposes `pixels` (current) and `goal_pixels` (target image); two solvers are
provided:

- `swm.solver.CEMSolver` — 300 samples, 30 steps, topk=30. Continuous actions.
- `swm.solver.GradientSolver` — Adam over actions, 30 steps, lr=0.1.

The cost is `MSE(predicted_last_emb, goal_emb)` — i.e. goal-conditioned MPC.
This is the formulation used in PushT, Cube, TwoRooms, Reacher: every episode
has a **fixed goal image**.

---

## 2. Gap analysis vs the Unity rogue game

| Aspect | Upstream LeWM | `leworldmodel_demo` (rogue) | Gap |
| --- | --- | --- | --- |
| Observation | Raw pixels (3×224×224) | 31-d hand-engineered vector | New pixel pipeline needed (M2). |
| Action space | Continuous `R^A` | Discrete {Up,Down,Left,Right} (+possibly Wait/Attack) | Adapt `Embedder` to one-hot, swap CEM for discrete-action sampling. |
| Goal | Fixed image per episode | Procedural levels, no fixed goal image | **Replace goal MSE cost** with reward+done prediction. |
| Inference compute | GPU (`.to('cuda')`) | Unity in-engine, every tick | **Unity cannot run PyTorch/JEPA in-engine.** Need Python sidecar or ONNX export. |
| Training data | HDF5 expert trajectories | JSONL `(s,a,r,s')` from gameplay | Add JSONL → HDF5 converter (or PyTorch `Dataset` reading JSONL directly). |
| Dependencies | Lightning + Hydra + stable-pretraining + stable-worldmodel | `torch`, `tensorboard` only | Heavy upstream stack is overkill for CI; **vendor the model code only**. |
| Encoder size | ViT-tiny, ~5M params | n/a | ViT-tiny on 32×32 board is overkill. Provide a small CNN encoder. |
| Latency budget | Offline planning | <50 ms per decision | Need cached latent + cheap rollout. |

### Decision: vendor, do not embed full upstream

The upstream repo's framework dependencies (`stable-pretraining`,
`stable-worldmodel`, Lightning, Hydra) total hundreds of megabytes and need
HuggingFace dataset access plus a GPU. They are designed for offline research,
not embedded use.

We therefore **vendor the core LeWM modules** (`module.py`, `jepa.py`)
verbatim under `tools/lewm/` with the MIT notice, and replace the heavy
training/eval scaffolding with a small PyTorch-only loop. This is a faithful
port of the **architecture and loss** (JEPA + SIGReg + AR predictor + action
embedder), not a copy of the framework.

---

## 3. Architecture for the port

```
┌─────────────────────────── Unity ──────────────────────────────┐
│ BoardManager → PixelObservationBuilder (M2) ── 32×32×3 RGB     │
│        │                                                       │
│        ▼                                                       │
│ WorldModelPlannerAgent (M5)                                    │
│   ├─ HTTP/JSON ──► Python sidecar (M3)                         │
│   │                  ├─ LeWM encoder (CNN, M1)                 │
│   │                  ├─ AR predictor + action embedder (M1)    │
│   │                  ├─ reward head + done head (M1, added)    │
│   │                  └─ Dreamer-style actor (M5)               │
│   ├─ RogueTransitionRecorder (JSONL v3 w/ pixels, M2)          │
│   └─ BrainHUDController (existing, displays returned plan)     │
└────────────────────────────────────────────────────────────────┘

Offline training (M4):
    JSONL v3 ── dataset_loader.py ──► PyTorch Dataset
                                        │
                                        ▼
                                tools/lewm/train.py
                                  (LeWM loss: pred + λ·SIGReg
                                   + reward MSE + done BCE)
                                        │
                                        ▼
                            checkpoints/{episode}.pt  ─► loaded by sidecar
```

### 3.1 Why a Python sidecar instead of ONNX Sentis

`SIGReg` is training-only and would not appear in inference graph, so it is
not a problem. But:

- The full JEPA forward (ViT or CNN + AR transformer + action embedder)
  contains operations (`F.scaled_dot_product_attention`, AdaLN zero-init,
  arbitrary `einops.rearrange`) that round-trip through ONNX poorly without
  custom shape-tracing. Unity Sentis supports a subset of ONNX ops.
- A Python sidecar lets us iterate on the model with full PyTorch (any new
  module added in upstream is one `pip install` away from working).
- The sidecar runs over `127.0.0.1` (or a Unix socket) — latency stays sub-5ms
  per tick on CPU once the encoded latent is cached.

Sidecar trade-off: the user must launch the sidecar before opening the demo,
or the Unity client falls back to the existing MLP/mission planner. Fallback
is required for the demo's stability rule from AGENTS.md (*"Stable fallback if
learned weights are missing or weak"*).

### 3.2 Replacing the goal-conditioned cost

The original `JEPA.criterion` uses `MSE(predicted_last_emb, goal_emb)`. We
keep the rollout machinery but score candidate action sequences with:

```
cost(action_seq) = − Σ_t γ^t · r̂(emb_t, action_t)
                  + α · donê(emb_T)               # avoid early termination
                  + β · SIGReg(emb_seq)            # surprise regulariser (eval-time, optional)
```

`r̂` and `donê` are small heads added to `JEPA` (M1). They are trained
alongside the JEPA losses with extra MSE / BCE-with-logits terms. This adds
two hyperparameters (`reward_loss_weight`, `done_loss_weight`); both default
to `1.0` to match the current MLP trainer.

### 3.3 Encoder for 32×32 board

ViT-tiny on a 32×32 rendered board is wasteful. We provide
`tools/lewm/encoder.py::TinyConvEncoder`:

```
Conv2d(3, 32, k=3, s=2)   → 32×16×16
GELU
Conv2d(32, 64, k=3, s=2)  → 64×8×8
GELU
Conv2d(64, 128, k=3, s=2) → 128×4×4
GroupNorm
Linear(128*4*4, embed_dim=128)
```

~280 K params, sub-millisecond on CPU. Still produces a CLS-equivalent token
of shape `(B, embed_dim)` so the rest of `JEPA` is unchanged.

---

## 4. Milestone breakdown

Each milestone is a single PR. Quality gates per milestone:

- Python: `python -m py_compile` on every new module **and** a smoke run.
- Linting: `ruff check tools/lewm/` once introduced (M7).
- Unit smoke for the model: `python tools/lewm/train.py --smoke --epochs 2`.
- Each PR must keep the existing `tools/world_model/train_world_model.py`
  smoke working.

| # | PR title | Scope |
| -- | --- | --- |
| ~~M1~~ | `LeWM port — vendor upstream modules + scaffold` | **landed (PR #1).** Vendor `module.py`, adapt `jepa.py`, add `TinyConvEncoder`, synthetic dataset + smoke training script. No Unity changes. |
| ~~M2~~ | `LeWM port — Unity pixel observations + JSONL v3` | **landed (PR #2).** New `PixelObservationBuilder.cs` emits a row-major cell-code grid; `RogueTransitionRecorder.cs` schema bumped to `rogue.transition.v3` with `board_state` / `next_board_state` / `board_width` / `board_height`; `WorldModelPlannerAgent.cs` captures pre/post-step board snapshots; new `BoardJsonlDataset` renders cell codes to RGB on the Python side. Vector observations preserved; v2 readers keep working. |
| ~~M3~~ | `LeWM port — Python inference sidecar` | **landed (PR #4).** `tools/lewm/sidecar.py` (FastAPI app with `/healthz`, `/info`, `/score_actions`), `tools/lewm/serve.py` (uvicorn CLI), `Assets/Scripts/ML/LewmClient.cs` (`UnityWebRequest` MonoBehaviour). M3 only lays the wire — `BrainPlanner` / `WorldModelPlannerAgent` are **not yet** rerouted through the sidecar; that swap happens in M5 with a sidecar-down fallback to the existing path. Includes `tests/test_sidecar.py` in-process TestClient integration covering load → healthz → info → score_actions → error paths. |
| ~~M4~~ | `LeWM port — training pipeline (LR schedule + val split + metrics CSV + best-ckpt)` | **landed (PR #5).** `tools/lewm/schedule.py` (cosine + warmup `LambdaLR`), `train.py` refactored with `build_dataloaders`/`evaluate`/per-epoch metrics CSV/best-checkpoint tracking, `TrainConfig` extended with `lr_schedule`/`warmup_steps`/`min_lr_ratio`/`val_split`/`val_items_per_epoch` (all defaulted so M1/M3 checkpoints keep loading). Sidecar uses new `load_config_from_payload` helper that filters unknown payload fields for backward compat. JSONL train/val split is **episode-level** (not transition-level) so val windows cannot leak from train. New `tests/test_training_pipeline.py` drives the full pipeline (synthetic v3 JSONL → train → metrics CSV → best.pt → sidecar score) end-to-end on CPU. |
| ~~M5~~ | `LeWM port — Dreamer-style actor on imagined rollouts` | **landed (PR #6).** `tools/lewm/planner.py` adds `random_shooting` (samples `num_candidates` discrete action sequences, scores each via `JEPA.score_action_sequences`, returns best + top-k). Sidecar gains `POST /plan_actions` (and `/info` now reports `max_horizon = sequence_length − 1`). Unity-side: `LewmClient` extended with a `PlanActions` coroutine wrapping a generic `SendPost<TReq, TResp>`; `BrainPlanner` exposes `DecideWithSidecarPlan(...)` so the **model is the primary scorer** (`SidecarSafetyWeight = 0.5` first-step safety bias, map heuristic only hard-rejects blocked moves); `WorldModelPlannerAgent` polls `/info` for liveness, fires one async `/plan_actions` per decision tick, and gracefully falls back to `BrainPlanner.Decide` when the sidecar is unreachable. JSONL recording (M2) keeps emitting v3 transitions for both the sidecar and mission-heuristic paths. New `tests/test_planner.py` covers determinism + `include_sequences`; `tests/test_sidecar.py` covers the new `/plan_actions` happy path and 400 error paths. |
| ~~M6~~ | `LeWM port — benchmark harness (5-mode comparison + CSV metrics)` | **landed (PR #7).** `tools/lewm/env.py` is a small python-side rogue env that mirrors the Unity cell-code grid (-1 wall / 0 empty / 1 exit / 2 enemy / 3 obstacle / 4 food / 5 player) and the `SyntheticRogueDataset` step dynamics (step cost, bump penalty, exit reward + auto-regenerated next level, enemy food loss, food pickup, starvation). `tools/lewm/benchmark.py` exposes five modes (`random`, `mission` BFS-to-exit + enemy-avoidance, `mlp_lite` surrogate triggered by the on-disk `world_model_weights.json` marker, `lewm_no_planner` 1-step JEPA argmax, `lewm_dreamer` reusing the M5 `random_shooting`) and writes both a per-episode CSV (`seed, mode, episode_idx, levels_cleared, food_left, steps, episode_return, dynamics_loss, reward_loss, died`) and a per-mode summary CSV (`seed, mode, episodes, mean_levels_cleared, mean_food_left, mean_steps, mean_episode_return, mean_dynamics_loss, mean_reward_loss, death_rate, std_episode_return`). `dynamics_loss` / `reward_loss` are recorded whenever a JEPA checkpoint is loaded (including for random / mission rows, as an off-policy predictor-quality diagnostic). Two new smokes: `tests/test_env.py` covers cell-code conventions, wall bump, food pickup, exit/level transition, starvation, max-step termination; `tests/test_benchmark.py` runs every mode for one episode each, asserts the CSV/summary headers, finite dynamics loss with a checkpoint, and the JEPA-without-checkpoint / missing-checkpoint error paths. |
| ~~M7~~ | `LeWM port — GitHub Actions Python CI + pre-commit + ruff` | **landed (this PR).** `.github/workflows/python.yml` runs three jobs on every push to `main` / PR touching the Python tree: (1) `lint` runs `ruff check tools/` + `ruff format --check tools/` against the pinned `ruff==0.6.9`; (2) `smoke` runs `py_compile` on every `.py` under `tools/` and the eight verification commands from `tools/lewm/README.md` (`tools.lewm.train --smoke`, the five `tools/lewm/tests/test_*.py` smokes, `tools/world_model/train_world_model.py --smoke --epochs 2`); (3) `benchmark-smoke` re-trains a fresh checkpoint and drives `tools/lewm/benchmark.py` over all five modes for one episode each, then uploads the resulting `benchmark.csv` + `benchmark.summary.csv` as a CI artifact. `pyproject.toml` ships a shared ruff config (`line-length = 100`, `target-version = "py310"`, the standard pyflakes / pycodestyle / isort / bugbear / pyupgrade / pylint-subset / ruff-specific rule set, vendored `tools/lewm/module.py` exempt from both lint and format to keep it byte-identical to upstream `c8a4417`, `tools/world_model/` excluded since it predates the lint config). `.pre-commit-config.yaml` wires up `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-merge-conflict`, `check-added-large-files` (max 2 MB), `ruff` and `ruff-format`. The hygiene hooks are explicitly scoped to `tools/lewm/*.py`, `tools/lewm/requirements*.txt`, `pyproject.toml`, `.pre-commit-config.yaml` and `.github/workflows/*.yml` so they cannot rewrite Unity-managed assets. `tools/lewm/requirements-dev.txt` pins `ruff==0.6.9` and `pre-commit>=3.7`. |

### 4.1 Tier B/C metrics this targets

Per user clarification, the optimisation target is the joint of:

1. `dynamicsLoss` / `rewardLoss` (predictor quality)
2. `levels_cleared` per episode (game progress)
3. `food_left` at game-over (survival)

`benchmark.py` (M6) writes both a per-episode CSV (`seed, mode,
episode_idx, levels_cleared, food_left, steps, episode_return,
dynamics_loss, reward_loss, died`) and a per-mode summary CSV
(`seed, mode, episodes, mean_levels_cleared, mean_food_left, mean_steps,
mean_episode_return, mean_dynamics_loss, mean_reward_loss,
death_rate, std_episode_return`) for every mode in
`{random, mission, mlp_lite, lewm_no_planner, lewm_dreamer}`.

---

## 5. Risks and mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Sidecar latency > 50 ms breaks demo | Med | Cache encoded latent; pre-warm sidecar; benchmark on CI smoke. |
| Pixel rendering hides game state (e.g. unseen enemy) | Low | M2 renders **every** cell type to a distinct RGB triple including invisible objects. |
| User has no GPU → cannot retrain | High | Document CPU smoke schedule (slow but feasible at 8×8 board, ~1 hr to converge). |
| Vendored upstream code drifts | Low | Pin upstream commit in this file; copy unchanged with header. |
| Existing `BrainPlanner` regresses while M5 lands | Med | Keep `--mode mission_only` path identical until M5 ships. |
| Unity Sentis ONNX path requested later | Low | M1 keeps the encoder small and ONNX-friendly so a Sentis fork is possible. |

---

## 6. Acceptance criteria for M0 (this document)

- [x] Upstream repo cloned and inspected (commit `c8a4417`).
- [x] Architecture, loss, training stack, planning method documented.
- [x] Gap analysis vs the rogue game written.
- [x] Sidecar vs ONNX trade-off decided (sidecar; see §3.1).
- [x] Replacement cost formulation specified (§3.2).
- [x] Encoder replacement specified (§3.3).
- [x] 7 milestones (M1–M7) sequenced with exit criteria.
- [x] Risks and mitigations enumerated.

Next: M1 implements `tools/lewm/` with the vendored upstream modules and a
synthetic-data smoke training run on CPU. M1 introduces **no Unity changes**.
