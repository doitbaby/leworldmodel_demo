"""LeWorldModel port -- PyTorch-only training entry point.

Replaces the upstream ``train.py`` (which depended on PyTorch Lightning,
Hydra, ``stable-pretraining`` and ``stable-worldmodel``). The vendored
:mod:`tools.lewm.module` and :mod:`tools.lewm.jepa` are the same building
blocks; we just drive them with a plain ``argparse`` + ``DataLoader``
loop.

Usage (smoke training on synthetic data, CPU-friendly)::

    python -m tools.lewm.train --smoke

Pixel-mode training from Unity gameplay (v3 board JSONL)::

    python -m tools.lewm.train \\
        --observation-mode board-jsonl \\
        --jsonl-path %APPDATA%/.../rogue_transitions.jsonl \\
        --epochs 30 --batch-size 64 \\
        --val-split 0.1 --lr-schedule cosine --warmup-steps 200

M4 additions on top of the M1 scaffold:

* Cosine learning-rate schedule with linear warmup (matches upstream).
* Train/val split over JSONL episodes (the validation set holds out
  whole episodes so windows from train cannot leak into val).
* Per-epoch metrics CSV (``results/lewm/metrics.csv`` by default) for
  the M6 benchmark harness to consume.
* Best-checkpoint tracking — whenever ``val_loss`` improves, a copy is
  written to ``--best-output`` (default ``results/lewm/best.pt``).

The output checkpoint is a single ``.pt`` file consumed by the Python
inference sidecar (M3) and any future planner integration (M5).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import (
    BoardJsonlDataset,
    SequenceCollator,
    SyntheticRogueDataset,
    VectorJsonlDataset,
)
from .encoder import TinyConvEncoder, VectorEncoder
from .jepa import JEPA
from .module import MLP, ARPredictor, Embedder, SIGReg
from .schedule import constant_lambda, cosine_warmup_lambda

_CHECKPOINT_SCHEMA = "lewm.port.checkpoint.v1"
_METRICS_JSON_SCHEMA = "lewm.port.metrics.v1"
_PIXEL_MODES = {"pixel", "board-jsonl"}


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    embed_dim: int
    predictor_depth: int
    predictor_heads: int
    predictor_mlp_dim: int
    predictor_dim_head: int
    sequence_length: int
    sigreg_weight: float
    sigreg_knots: int
    sigreg_num_proj: int
    reward_loss_weight: float
    done_loss_weight: float
    observation_mode: str
    image_size: int
    board_size: int
    action_dim: int
    observation_size: int
    items_per_epoch: int
    seed: int
    # ---- M4 additions (all defaulted for backward compat with M1/M3 ckpts) ----
    lr_schedule: str = "none"
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0
    val_split: float = 0.0
    val_items_per_epoch: int = 0
    # ---- M9 ablation knobs (defaulted off for backward compat) ----
    zero_actions: bool = False


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def load_config_from_payload(payload: dict[str, Any]) -> TrainConfig:
    """Reconstruct a :class:`TrainConfig` from a checkpoint payload.

    Forward and backward compatible: any payload fields not present on
    the current ``TrainConfig`` are silently dropped; any current fields
    missing from the payload fall back to their dataclass defaults.
    Sidecar code uses this so an M1 / M3 checkpoint keeps loading after
    M4 adds new optional fields.
    """
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    raw = payload.get("config", {})
    if not isinstance(raw, dict):
        raise ValueError(f"checkpoint config must be a dict, got {type(raw).__name__}")
    filtered = {k: v for k, v in raw.items() if k in known}
    return TrainConfig(**filtered)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model(cfg: TrainConfig) -> JEPA:
    if cfg.observation_mode in _PIXEL_MODES:
        encoder: torch.nn.Module = TinyConvEncoder(
            in_channels=3,
            image_size=cfg.image_size,
            embed_dim=cfg.embed_dim,
        )
    elif cfg.observation_mode == "vector":
        encoder = VectorEncoder(
            input_dim=cfg.observation_size,
            embed_dim=cfg.embed_dim,
        )
    else:
        raise ValueError(f"unknown observation_mode={cfg.observation_mode}")

    predictor = ARPredictor(
        num_frames=cfg.sequence_length - 1,
        depth=cfg.predictor_depth,
        heads=cfg.predictor_heads,
        mlp_dim=cfg.predictor_mlp_dim,
        input_dim=cfg.embed_dim,
        hidden_dim=cfg.embed_dim,
        output_dim=cfg.embed_dim,
        dim_head=cfg.predictor_dim_head,
        dropout=0.0,
        emb_dropout=0.0,
    )

    action_encoder = Embedder(
        input_dim=cfg.action_dim,
        smoothed_dim=cfg.action_dim,
        emb_dim=cfg.embed_dim,
        mlp_scale=4,
    )

    projector = MLP(
        input_dim=cfg.embed_dim,
        hidden_dim=cfg.embed_dim * 4,
        output_dim=cfg.embed_dim,
        norm_fn=None,
    )
    pred_proj = MLP(
        input_dim=cfg.embed_dim,
        hidden_dim=cfg.embed_dim * 4,
        output_dim=cfg.embed_dim,
        norm_fn=None,
    )

    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        action_dim=cfg.action_dim,
        embed_dim=cfg.embed_dim,
        projector=projector,
        pred_proj=pred_proj,
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _obs_key(mode: str) -> str:
    return "pixels" if mode in _PIXEL_MODES else "obs"


def _episode_split_indices(
    n_episodes: int,
    val_split: float,
    seed: int,
) -> tuple[set[int], set[int]]:
    """Pick ``ceil(n_episodes * val_split)`` distinct episode ids for val.

    Episodes are split at the boundary (not transitions) so a window
    from a training episode cannot leak into the validation set.
    """
    if n_episodes <= 0:
        return set(), set()
    if val_split <= 0.0:
        return set(range(n_episodes)), set()

    n_val = max(1, int(math.ceil(n_episodes * val_split)))
    if n_episodes > 1:
        n_val = min(n_val, n_episodes - 1)
    else:
        n_val = 0
    rng = np.random.default_rng(seed)
    order = np.arange(n_episodes)
    rng.shuffle(order)
    val_eps = {int(e) for e in order[:n_val]}
    train_eps = {int(e) for e in order[n_val:]}
    return train_eps, val_eps


def _subset_by_episode(
    dataset: BoardJsonlDataset | VectorJsonlDataset,
    eps: set[int],
) -> Subset:
    indices = [i for i, (ep, _) in enumerate(dataset._windows) if ep in eps]
    return Subset(dataset, indices)


def build_dataloaders(
    cfg: TrainConfig,
    jsonl_path: str | None,
    num_workers: int,
) -> tuple[DataLoader, DataLoader | None, str]:
    """Construct train + (optional) val dataloaders for the active mode.

    Returns ``(train_loader, val_loader_or_none, obs_key)``. The val
    loader is ``None`` when ``cfg.val_split == 0`` (JSONL modes) or
    when ``cfg.val_items_per_epoch == 0`` (synthetic mode).
    """
    if cfg.observation_mode == "pixel":
        train_dataset = SyntheticRogueDataset(
            sequence_length=cfg.sequence_length,
            board_size=cfg.board_size,
            image_size=cfg.image_size,
            action_dim=cfg.action_dim,
            seed=cfg.seed,
            items_per_epoch=cfg.items_per_epoch,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            num_workers=num_workers,
            collate_fn=SequenceCollator(),
        )

        val_loader: DataLoader | None = None
        if cfg.val_items_per_epoch > 0:
            val_dataset = SyntheticRogueDataset(
                sequence_length=cfg.sequence_length,
                board_size=cfg.board_size,
                image_size=cfg.image_size,
                action_dim=cfg.action_dim,
                # Separate seed range so val episodes never coincide
                # with the training stream.
                seed=cfg.seed + 991,
                items_per_epoch=cfg.val_items_per_epoch,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.batch_size,
                num_workers=num_workers,
                collate_fn=SequenceCollator(),
            )
        return train_loader, val_loader, "pixels"

    if cfg.observation_mode == "board-jsonl":
        if not jsonl_path:
            raise ValueError("board-jsonl mode requires --jsonl-path")
        board_dataset = BoardJsonlDataset(
            path=jsonl_path,
            sequence_length=cfg.sequence_length,
            image_size=cfg.image_size,
            action_dim=cfg.action_dim,
        )
        if len(board_dataset) == 0:
            raise RuntimeError(
                f"no v3 board transitions found in {jsonl_path} "
                "(missing board_state/next_board_state fields?)"
            )
        return _build_jsonl_loaders(board_dataset, cfg, num_workers, obs_key="pixels")

    if not jsonl_path:
        raise ValueError("vector mode requires --jsonl-path")
    vec_dataset = VectorJsonlDataset(
        path=jsonl_path,
        sequence_length=cfg.sequence_length,
        observation_size=cfg.observation_size,
        action_dim=cfg.action_dim,
    )
    if len(vec_dataset) == 0:
        raise RuntimeError(f"no transitions found in {jsonl_path}")
    return _build_jsonl_loaders(vec_dataset, cfg, num_workers, obs_key="obs")


def _build_jsonl_loaders(
    dataset: BoardJsonlDataset | VectorJsonlDataset,
    cfg: TrainConfig,
    num_workers: int,
    *,
    obs_key: str,
) -> tuple[DataLoader, DataLoader | None, str]:
    n_eps = len(dataset._episodes)
    train_subset: Subset | BoardJsonlDataset | VectorJsonlDataset
    val_loader: DataLoader | None = None
    if cfg.val_split > 0 and n_eps >= 2:
        train_eps, val_eps = _episode_split_indices(n_eps, cfg.val_split, cfg.seed)
        train_subset = _subset_by_episode(dataset, train_eps)
        val_subset = _subset_by_episode(dataset, val_eps)
        if len(val_subset) > 0:
            val_loader = DataLoader(
                val_subset,
                batch_size=cfg.batch_size,
                num_workers=num_workers,
                shuffle=False,
                drop_last=False,
                collate_fn=SequenceCollator(obs_key=obs_key),
            )
    else:
        train_subset = dataset

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        collate_fn=SequenceCollator(obs_key=obs_key),
    )
    return train_loader, val_loader, obs_key


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def _accumulate(d: dict[str, float], **kv: float) -> None:
    for k, v in kv.items():
        d[k] = d.get(k, 0.0) + v


def _normalise(d: dict[str, float], n: int) -> dict[str, float]:
    return {k: v / max(1, n) for k, v in d.items()}


def _forward_losses(
    model: JEPA,
    sigreg: SIGReg,
    batch: dict[str, torch.Tensor],
    cfg: TrainConfig,
    device: torch.device,
    obs_key: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    obs = batch[obs_key].to(device)
    action = batch["action"].to(device)
    reward = batch["reward"].to(device)
    done = batch["done"].to(device)

    if cfg.zero_actions:
        # Action-conditioning ablation: replace one-hot actions with
        # all-zeros so the predictor cannot use action information.
        # The action tensor must keep the same shape so the predictor
        # still receives a valid action-embedding sequence.
        action = torch.zeros_like(action)

    output = model.compute_losses(obs, action, reward, done)
    sigreg_loss = sigreg(output.embedding.transpose(0, 1))

    total = (
        output.pred_loss
        + cfg.sigreg_weight * sigreg_loss
        + cfg.reward_loss_weight * output.reward_loss
        + cfg.done_loss_weight * output.done_loss
    )
    return total, {
        "loss": total.item(),
        "pred_loss": output.pred_loss.item(),
        "sigreg_loss": sigreg_loss.item(),
        "reward_loss": output.reward_loss.item(),
        "done_loss": output.done_loss.item(),
    }


@torch.no_grad()
def evaluate(
    model: JEPA,
    sigreg: SIGReg,
    loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    obs_key: str,
) -> dict[str, float]:
    model.eval()
    sigreg.eval()
    accum: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        _, metrics = _forward_losses(model, sigreg, batch, cfg, device, obs_key)
        _accumulate(accum, **metrics)
        n_batches += 1
    model.train()
    sigreg.train()
    return _normalise(accum, n_batches)


def train_loop(
    model: JEPA,
    sigreg: SIGReg,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: TrainConfig,
    device: torch.device,
    obs_key: str,
    *,
    metrics_csv: Path | None = None,
    best_output: Path | None = None,
    final_output: Path | None = None,
) -> list[dict[str, float]]:
    """Run training with optional LR schedule, validation and CSV logging.

    Returns the per-epoch metrics list. Each entry is a dict with at
    least ``epoch``, ``train_loss``, ``epoch_time_s`` and (when a val
    loader is supplied) ``val_loss`` keys, plus the four constituent
    losses for both splits.
    """
    optim = torch.optim.AdamW(
        list(model.parameters()) + list(sigreg.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Estimate total optimisation steps for the cosine schedule.
    # ``len(train_loader)`` is well-defined for map-style datasets
    # (JSONL). The synthetic IterableDataset has implicit length, so
    # we fall back to ``items_per_epoch / batch_size``.
    try:
        steps_per_epoch = len(train_loader)
    except TypeError:
        steps_per_epoch = max(1, cfg.items_per_epoch // max(1, cfg.batch_size))

    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warmup = max(0, min(cfg.warmup_steps, total_steps - 1))
    if cfg.lr_schedule == "cosine":
        lr_fn = cosine_warmup_lambda(
            warmup_steps=warmup,
            total_steps=total_steps,
            min_lr_ratio=cfg.min_lr_ratio,
        )
    else:
        lr_fn = constant_lambda()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_fn)

    history: list[dict[str, float]] = []
    best_val: float | None = None

    for epoch in range(1, cfg.epochs + 1):
        epoch_started = time.time()
        accum: dict[str, float] = {}
        n_batches = 0
        last_lr = cfg.learning_rate

        for batch in train_loader:
            loss, metrics = _forward_losses(model, sigreg, batch, cfg, device, obs_key)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            last_lr = optim.param_groups[0]["lr"]
            _accumulate(accum, **metrics)
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("dataloader yielded zero batches")
        train_metrics = _normalise(accum, n_batches)

        val_metrics: dict[str, float] | None = None
        if val_loader is not None:
            val_metrics = evaluate(model, sigreg, val_loader, cfg, device, obs_key)

        epoch_record: dict[str, float] = {
            "epoch": float(epoch),
            "lr": float(last_lr),
            "epoch_time_s": time.time() - epoch_started,
            "train_loss": train_metrics["loss"],
            "train_pred_loss": train_metrics["pred_loss"],
            "train_sigreg_loss": train_metrics["sigreg_loss"],
            "train_reward_loss": train_metrics["reward_loss"],
            "train_done_loss": train_metrics["done_loss"],
        }
        if val_metrics is not None:
            epoch_record.update(
                val_loss=val_metrics["loss"],
                val_pred_loss=val_metrics["pred_loss"],
                val_sigreg_loss=val_metrics["sigreg_loss"],
                val_reward_loss=val_metrics["reward_loss"],
                val_done_loss=val_metrics["done_loss"],
            )
        history.append(epoch_record)

        msg_parts = [
            f"epoch={epoch:03d}",
            f"lr={last_lr:.2e}",
            f"train_loss={train_metrics['loss']:.6f}",
            f"train_pred={train_metrics['pred_loss']:.6f}",
            f"train_reward={train_metrics['reward_loss']:.6f}",
            f"train_done={train_metrics['done_loss']:.6f}",
        ]
        if val_metrics is not None:
            msg_parts.append(f"val_loss={val_metrics['loss']:.6f}")
            msg_parts.append(f"val_pred={val_metrics['pred_loss']:.6f}")
        msg_parts.append(f"epoch_time_s={epoch_record['epoch_time_s']:.3f}")
        print(" ".join(msg_parts))

        if val_metrics is not None and best_output is not None:
            if best_val is None or val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(
                    model,
                    sigreg,
                    cfg,
                    history,
                    best_output,
                    label="best",
                )

    if metrics_csv is not None:
        write_metrics_csv(history, metrics_csv, has_val=val_loader is not None)

    if final_output is not None:
        save_checkpoint(model, sigreg, cfg, history, final_output, label="final")

    return history


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_metrics_csv(
    history: Iterable[dict[str, float]],
    metrics_csv: Path,
    *,
    has_val: bool,
) -> None:
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "lr",
        "epoch_time_s",
        "train_loss",
        "train_pred_loss",
        "train_sigreg_loss",
        "train_reward_loss",
        "train_done_loss",
    ]
    if has_val:
        fieldnames.extend(
            [
                "val_loss",
                "val_pred_loss",
                "val_sigreg_loss",
                "val_reward_loss",
                "val_done_loss",
            ]
        )
    with metrics_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    print(f"saved metrics csv: {metrics_csv}")


def save_checkpoint(
    model: JEPA,
    sigreg: SIGReg,
    cfg: TrainConfig,
    history: list[dict[str, float]] | dict[str, float],
    output: Path,
    *,
    label: str = "final",
) -> None:
    """Persist a JEPA checkpoint compatible with the M3 sidecar.

    ``history`` may be either the M4 per-epoch list of dicts or the M1
    "last metrics" dict; both are accepted so older callers keep
    working.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(history, list):
        final_metrics = history[-1] if history else {}
        history_list = history
    else:
        final_metrics = history
        history_list = [history] if history else []

    payload = {
        "schema": _CHECKPOINT_SCHEMA,
        "config": asdict(cfg),
        "metrics": final_metrics,
        "history": history_list,
        "model_state": model.state_dict(),
        "sigreg_state": sigreg.state_dict(),
        "label": label,
    }
    torch.save(payload, output)
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "schema": _METRICS_JSON_SCHEMA,
                "label": label,
                "config": asdict(cfg),
                "metrics": final_metrics,
                "history": history_list,
            },
            indent=2,
        )
    )
    print(f"saved checkpoint ({label}): {output}")
    print(f"saved metrics:           {metrics_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the LeWM rogue port.")
    p.add_argument("--smoke", action="store_true", help="tiny preset for CI.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--predictor-depth", type=int, default=2)
    p.add_argument("--predictor-heads", type=int, default=4)
    p.add_argument("--predictor-mlp-dim", type=int, default=256)
    p.add_argument("--predictor-dim-head", type=int, default=32)
    p.add_argument("--sequence-length", type=int, default=4)
    p.add_argument("--sigreg-weight", type=float, default=0.09)
    p.add_argument("--sigreg-knots", type=int, default=9)
    p.add_argument("--sigreg-num-proj", type=int, default=128)
    p.add_argument("--reward-loss-weight", type=float, default=1.0)
    p.add_argument("--done-loss-weight", type=float, default=1.0)
    p.add_argument(
        "--zero-actions",
        action="store_true",
        help=(
            "Action-conditioning ablation: replace one-hot actions with "
            "zeros at the dataloader boundary so the predictor cannot use "
            "action information. Recorded in TrainConfig so downstream "
            "tooling can identify the run."
        ),
    )
    p.add_argument(
        "--observation-mode",
        choices=["pixel", "vector", "board-jsonl"],
        default="pixel",
        help=(
            "'pixel' = synthetic CNN smoke; 'vector' = v2/v3 JSONL with 31-d "
            "obs; 'board-jsonl' = v3 JSONL with board_state cell codes."
        ),
    )
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--board-size", type=int, default=8)
    p.add_argument("--action-dim", type=int, default=4)
    p.add_argument("--observation-size", type=int, default=31)
    p.add_argument("--items-per-epoch", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--jsonl-path", default=None)
    # ---- M4 additions ----
    p.add_argument(
        "--lr-schedule",
        choices=["none", "cosine"],
        default=None,
        help=("LR schedule. Defaults to 'cosine' for non-smoke runs and 'none' " "for --smoke."),
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Linear warmup steps for the cosine schedule (default: 5%% of total).",
    )
    p.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.0,
        help="Final LR multiplier reached by the cosine schedule.",
    )
    p.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help=(
            "Fraction of episodes to hold out for validation in JSONL modes "
            "(0 = no validation). Ignored by synthetic mode; use "
            "--val-items-per-epoch there."
        ),
    )
    p.add_argument(
        "--val-items-per-epoch",
        type=int,
        default=0,
        help="Synthetic-mode validation budget (0 = no validation).",
    )
    p.add_argument(
        "--metrics-csv",
        default="results/lewm/metrics.csv",
        help="Per-epoch metrics CSV (set to empty string to disable).",
    )
    p.add_argument(
        "--output",
        default="results/lewm/checkpoint.pt",
        help="Path to write the final-epoch checkpoint.",
    )
    p.add_argument(
        "--best-output",
        default="results/lewm/best.pt",
        help=(
            "Path to write the best-val checkpoint (only used when "
            "--val-split > 0 or --val-items-per-epoch > 0)."
        ),
    )
    return p.parse_args(argv)


def _resolve_defaults(args: argparse.Namespace) -> None:
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.batch_size = min(args.batch_size, 16)
        args.items_per_epoch = min(args.items_per_epoch, 32)
        args.predictor_depth = 1
        args.predictor_heads = 2
        args.predictor_mlp_dim = 128
        args.embed_dim = 64
        args.sigreg_num_proj = 32
        args.sequence_length = 3
        if args.lr_schedule is None:
            args.lr_schedule = "none"
        if args.warmup_steps is None:
            args.warmup_steps = 0
    if args.lr_schedule is None:
        args.lr_schedule = "cosine"
    if args.warmup_steps is None:
        # 5 % of total steps as a safe default.
        approx_total = args.epochs * max(1, args.items_per_epoch // max(1, args.batch_size))
        args.warmup_steps = max(0, int(0.05 * approx_total))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _resolve_defaults(args)

    torch.manual_seed(args.seed)
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        embed_dim=args.embed_dim,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        predictor_mlp_dim=args.predictor_mlp_dim,
        predictor_dim_head=args.predictor_dim_head,
        sequence_length=args.sequence_length,
        sigreg_weight=args.sigreg_weight,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
        reward_loss_weight=args.reward_loss_weight,
        done_loss_weight=args.done_loss_weight,
        observation_mode=args.observation_mode,
        image_size=args.image_size,
        board_size=args.board_size,
        action_dim=args.action_dim,
        observation_size=args.observation_size,
        items_per_epoch=args.items_per_epoch,
        seed=args.seed,
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        val_split=args.val_split,
        val_items_per_epoch=args.val_items_per_epoch,
        zero_actions=args.zero_actions,
    )

    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    sigreg = SIGReg(knots=cfg.sigreg_knots, num_proj=cfg.sigreg_num_proj).to(device)

    train_loader, val_loader, obs_key = build_dataloaders(cfg, args.jsonl_path, args.num_workers)

    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else None
    best_output = Path(args.best_output) if (val_loader is not None and args.best_output) else None
    final_output = Path(args.output)

    train_loop(
        model,
        sigreg,
        train_loader,
        val_loader,
        cfg,
        device,
        obs_key,
        metrics_csv=metrics_csv,
        best_output=best_output,
        final_output=final_output,
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
