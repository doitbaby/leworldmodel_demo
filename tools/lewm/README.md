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
| M0 — research / port analysis | this PR | landed (see `docs/research/lewm-port-analysis.md`). |
| M1 — vendor modules + smoke training | this PR | landed. |
| M2 — Unity pixel observation builder | TBD | not started. |
| M3 — Python inference sidecar | TBD | not started. |
| M4 — full training pipeline + checkpoints | TBD | not started. |
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
├── data.py       # synthetic rogue dataset + v2 JSONL adapter
├── train.py      # PyTorch-only training entry point
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
epoch=001 loss=... pred_loss=... sigreg_loss=... reward_loss=... done_loss=...
epoch=002 loss=... pred_loss=... sigreg_loss=... reward_loss=... done_loss=...
saved checkpoint: results/lewm/checkpoint.pt
saved metrics:    results/lewm/checkpoint.metrics.json
```

The checkpoint and `metrics.json` are written under `results/lewm/` (already
ignored by `.gitignore`).

## Vector-mode training (existing JSONL transitions)

The existing `RogueTransitionRecorder` writes v2 vector transitions. M1 can
train the JEPA over those files via `VectorEncoder`:

```bash
python -m tools.lewm.train \
    --observation-mode vector \
    --jsonl-path "$HOME/.config/unity3d/.../rogue_transitions.jsonl" \
    --epochs 5 \
    --batch-size 64
```

Vector mode is a pre-pixel-pipeline stop-gap useful for iterating on the
LeWM losses without M2 in place.

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

## Verification

The smallest meaningful check is the smoke run shown above; it exercises:

- Encoder forward (`TinyConvEncoder`)
- `Embedder` action embedding
- `ARPredictor` autoregressive predictor
- Reward + done heads
- `SIGReg` regulariser
- Optimiser step + checkpoint save

The existing `tools/world_model/train_world_model.py` MLP path is untouched
and continues to support the live Unity demo.
