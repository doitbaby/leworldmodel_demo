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
| M4 — training pipeline (LR schedule + val split + metrics CSV + best-ckpt) | this PR | landed. |
| M5 — Dreamer-style actor on imagined rollouts | TBD | not started. |
| M6 — benchmark harness | TBD | not started. |
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
├── sidecar.py    # FastAPI app wrapping JEPA inference (M3)
├── serve.py      # CLI: load checkpoint and boot uvicorn (M3)
├── tests/
│   ├── test_board_jsonl.py        # round-trip check for the v3 board JSONL path
│   ├── test_sidecar.py            # in-process TestClient integration for the sidecar
│   └── test_training_pipeline.py  # M4 end-to-end (train → metrics CSV → best.pt → sidecar)
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

## Inference sidecar (M3)

Unity cannot run PyTorch / JEPA in-engine. M3 adds a thin Python sidecar
that wraps a trained checkpoint behind three HTTP endpoints:

```
GET  /healthz          → { "status": "ok" | "no_model", "service": ... }
GET  /info             → { embed_dim, action_dim, image_size, sequence_length, service }
POST /score_actions    → { scores: [float, ...], horizon, service }
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
— a small `MonoBehaviour` wrapping `UnityWebRequest`. M3 only lands the
wire (`LewmClient` is **not yet** referenced by `BrainPlanner` /
`WorldModelPlannerAgent`); M5 plugs it into the live decision loop with
a graceful fallback to the existing mission heuristic + LeWM-lite path
when the sidecar is unreachable.

The flat action layout (`action_sequences_flat = num_sequences ×
horizon` row-major ints) is a JsonUtility compatibility quirk: Unity's
JsonUtility does not handle nested arrays. The sidecar reshapes
internally before calling `JEPA.score_action_sequences`.

## Verification

Four cheap checks cover the LeWM port end-to-end without GPU or Unity:

```bash
python -m tools.lewm.train --smoke                       # synthetic smoke training
python -m tools.lewm.tests.test_board_jsonl              # v3 JSONL round-trip
python -m tools.lewm.tests.test_sidecar                  # FastAPI sidecar integration
python -m tools.lewm.tests.test_training_pipeline        # M4 full pipeline (NEW)
```

The smoke train exercises encoder, Embedder, ARPredictor, RewardHead,
DoneHead, SIGReg, optimiser, checkpoint save. The round-trip test confirms
that the JSONL written by `RogueTransitionRecorder.cs` (schema v3) can be
read back and rendered into the exact `(T, 3, H, W)` pixel tensors the
training loop expects. The sidecar test trains a tiny checkpoint, loads
it via `load_checkpoint`, then drives `/healthz`, `/info`, and
`/score_actions` through Starlette's `TestClient`. The M4 pipeline test
generates a synthetic v3 JSONL, drives `tools.lewm.train` through its
CLI with `--val-split` + cosine schedule + metrics CSV, then re-opens
the resulting `best.pt` through the sidecar's `load_checkpoint`.

The existing `tools/world_model/train_world_model.py` MLP path is untouched
and continues to support the live Unity demo.
