"""LeWorldModel port -- PyTorch-only training entry point.

Replaces the upstream ``train.py`` (which depended on PyTorch Lightning,
Hydra, ``stable-pretraining`` and ``stable-worldmodel``). The vendored
:mod:`tools.lewm.module` and :mod:`tools.lewm.jepa` are the same building
blocks; we just drive them with a plain ``argparse`` + ``DataLoader`` loop.

Usage (smoke training on synthetic data, CPU-friendly)::

    python tools/lewm/train.py --smoke

Vector-mode training against an existing transition JSONL::

    python tools/lewm/train.py \\
        --observation-mode vector \\
        --jsonl-path %APPDATA%/.../rogue_transitions.jsonl \\
        --epochs 5

The output checkpoint is a single ``.pt`` file consumed by the Python
inference sidecar (M3) and any future training loop iteration (M4).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import (
    SequenceCollator,
    SyntheticRogueDataset,
    VectorJsonlDataset,
)
from .encoder import TinyConvEncoder, VectorEncoder
from .jepa import JEPA
from .module import MLP, ARPredictor, Embedder, SIGReg


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


def build_model(cfg: TrainConfig) -> JEPA:
    if cfg.observation_mode == "pixel":
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


def _obs_key(mode: str) -> str:
    return "pixels" if mode == "pixel" else "obs"


def build_dataloader(
    cfg: TrainConfig,
    jsonl_path: str | None,
    num_workers: int,
) -> tuple[DataLoader, str]:
    if cfg.observation_mode == "pixel":
        dataset = SyntheticRogueDataset(
            sequence_length=cfg.sequence_length,
            board_size=cfg.board_size,
            image_size=cfg.image_size,
            action_dim=cfg.action_dim,
            seed=cfg.seed,
            items_per_epoch=cfg.items_per_epoch,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=num_workers,
            collate_fn=SequenceCollator(),
        )
        return loader, "pixels"

    if not jsonl_path:
        raise ValueError("vector mode requires --jsonl-path")
    dataset = VectorJsonlDataset(
        path=jsonl_path,
        sequence_length=cfg.sequence_length,
        observation_size=cfg.observation_size,
        action_dim=cfg.action_dim,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"no transitions found in {jsonl_path}")
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        collate_fn=SequenceCollator(obs_key="obs"),
    )
    return loader, "obs"


def train_loop(
    model: JEPA,
    sigreg: SIGReg,
    loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    obs_key: str,
) -> dict[str, float]:
    optim = torch.optim.AdamW(
        list(model.parameters()) + list(sigreg.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    last_metrics: dict[str, float] = {}
    for epoch in range(1, cfg.epochs + 1):
        epoch_started = time.time()
        n_batches = 0
        accum: dict[str, float] = {}
        for batch in loader:
            obs = batch[obs_key].to(device)
            action = batch["action"].to(device)
            reward = batch["reward"].to(device)
            done = batch["done"].to(device)

            output = model.compute_losses(obs, action, reward, done)
            sigreg_loss = sigreg(output.embedding.transpose(0, 1))

            loss = (
                output.pred_loss
                + cfg.sigreg_weight * sigreg_loss
                + cfg.reward_loss_weight * output.reward_loss
                + cfg.done_loss_weight * output.done_loss
            )

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            for k, v in {
                "loss": loss.item(),
                "pred_loss": output.pred_loss.item(),
                "sigreg_loss": sigreg_loss.item(),
                "reward_loss": output.reward_loss.item(),
                "done_loss": output.done_loss.item(),
            }.items():
                accum[k] = accum.get(k, 0.0) + v
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("dataloader yielded zero batches")

        epoch_metrics = {k: v / n_batches for k, v in accum.items()}
        epoch_metrics["epoch_time_s"] = time.time() - epoch_started
        last_metrics = epoch_metrics
        print(
            f"epoch={epoch:03d} "
            + " ".join(f"{k}={v:.6f}" for k, v in epoch_metrics.items())
        )
    return last_metrics


def save_checkpoint(
    model: JEPA,
    sigreg: SIGReg,
    cfg: TrainConfig,
    metrics: dict[str, float],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "lewm.port.checkpoint.v1",
        "config": asdict(cfg),
        "metrics": metrics,
        "model_state": model.state_dict(),
        "sigreg_state": sigreg.state_dict(),
    }
    torch.save(payload, output)
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "schema": "lewm.port.metrics.v1",
                "config": asdict(cfg),
                "metrics": metrics,
            },
            indent=2,
        )
    )
    print(f"saved checkpoint: {output}")
    print(f"saved metrics:    {metrics_path}")


def parse_args() -> argparse.Namespace:
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
        "--observation-mode",
        choices=["pixel", "vector"],
        default="pixel",
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
    p.add_argument(
        "--output",
        default="results/lewm/checkpoint.pt",
        help="Path to write the trained checkpoint.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
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
    )

    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    sigreg = SIGReg(knots=cfg.sigreg_knots, num_proj=cfg.sigreg_num_proj).to(device)

    loader, obs_key = build_dataloader(cfg, args.jsonl_path, args.num_workers)
    metrics = train_loop(model, sigreg, loader, cfg, device, obs_key)

    save_checkpoint(model, sigreg, cfg, metrics, Path(args.output))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
