"""Integration smoke for the M4 training pipeline.

Generates a tiny synthetic v3 board JSONL, drives ``tools.lewm.train``
through its CLI with ``--val-split`` + cosine schedule + metrics CSV,
then re-opens the resulting checkpoint through the sidecar's loader and
scores a fake plan.

This does **not** prove training quality (the dataset is random); it
just proves that every piece of the M4 plumbing — episode-level val
split, LR scheduler, CSV writer, best-checkpoint tracking, sidecar
backward-compat loader — stays wired together end-to-end.

Usage:
    python -m tools.lewm.tests.test_training_pipeline
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .. import sidecar as sidecar_mod
from ..sidecar import load_checkpoint
from ..train import (
    TrainConfig,
    _episode_split_indices,
    load_config_from_payload,
    main,
)


_SCHEMA_V3 = "rogue.transition.v3"


def _emit_synthetic_v3(path: Path, *, episodes: int, steps: int, board: int) -> None:
    """Write ``episodes * steps`` synthetic v3 transitions to ``path``."""
    rng = np.random.default_rng(0)
    with path.open("w", encoding="utf-8") as fh:
        for ep in range(episodes):
            for step in range(steps):
                board_state = rng.integers(low=-1, high=6, size=board * board).tolist()
                next_board = rng.integers(low=-1, high=6, size=board * board).tolist()
                obs = rng.standard_normal(31).astype(np.float32).tolist()
                next_obs = rng.standard_normal(31).astype(np.float32).tolist()
                action = int(rng.integers(low=0, high=4))
                reward = float(rng.standard_normal())
                done = bool(step == steps - 1)
                record = {
                    "schema": _SCHEMA_V3,
                    "episode": ep,
                    "step": step,
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "next_obs": next_obs,
                    "board_width": board,
                    "board_height": board,
                    "board_state": board_state,
                    "next_board_state": next_board,
                }
                fh.write(json.dumps(record))
                fh.write("\n")


def _assert_episode_split() -> None:
    """The val/train partition must be disjoint and stable for a seed."""
    train_eps, val_eps = _episode_split_indices(n_episodes=10, val_split=0.3, seed=7)
    assert train_eps.isdisjoint(val_eps), "train and val episodes overlap"
    assert len(val_eps) == 3, f"expected 3 val episodes, got {len(val_eps)}"
    assert len(train_eps) == 7, f"expected 7 train episodes, got {len(train_eps)}"

    # Edge: single-episode dataset should not produce a val set.
    train_only, val_only = _episode_split_indices(n_episodes=1, val_split=0.5, seed=7)
    assert val_only == set()
    assert train_only == {0}


def _assert_metrics_csv(metrics_csv: Path, *, has_val: bool) -> list[dict[str, str]]:
    with metrics_csv.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, f"metrics CSV {metrics_csv} is empty"
    must_have = {
        "epoch",
        "lr",
        "epoch_time_s",
        "train_loss",
        "train_pred_loss",
        "train_sigreg_loss",
        "train_reward_loss",
        "train_done_loss",
    }
    if has_val:
        must_have.update({"val_loss", "val_pred_loss"})
    missing = must_have - set(rows[0].keys())
    assert not missing, f"metrics CSV missing columns: {sorted(missing)}"
    return rows


def _assert_checkpoint_payload(ckpt_path: Path, *, expect_label: str) -> TrainConfig:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert payload.get("schema") == "lewm.port.checkpoint.v1", payload.get("schema")
    assert payload.get("label") == expect_label, payload.get("label")
    assert "history" in payload and isinstance(payload["history"], list)
    cfg = load_config_from_payload(payload)
    # New M4 fields must round-trip.
    assert cfg.lr_schedule in {"none", "cosine"}
    assert cfg.val_split >= 0.0
    return cfg


def _assert_legacy_payload_still_loads() -> None:
    """Old M1/M3 checkpoints (no M4 fields) must keep loading."""
    legacy = {
        "config": {
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "embed_dim": 32,
            "predictor_depth": 1,
            "predictor_heads": 1,
            "predictor_mlp_dim": 32,
            "predictor_dim_head": 16,
            "sequence_length": 3,
            "sigreg_weight": 0.0,
            "sigreg_knots": 4,
            "sigreg_num_proj": 8,
            "reward_loss_weight": 1.0,
            "done_loss_weight": 1.0,
            "observation_mode": "pixel",
            "image_size": 32,
            "board_size": 8,
            "action_dim": 4,
            "observation_size": 31,
            "items_per_epoch": 4,
            "seed": 1,
            # NOTE: no lr_schedule / warmup_steps / val_split here.
        }
    }
    cfg = load_config_from_payload(legacy)
    assert cfg.lr_schedule == "none"
    assert cfg.warmup_steps == 0
    assert cfg.val_split == 0.0


def _drive_sidecar(ckpt_path: Path, *, board: int, image_size: int) -> None:
    # Reset module-global state before loading, otherwise an earlier
    # test in the same process would leak.
    sidecar_mod._state = sidecar_mod.SidecarState()
    load_checkpoint(str(ckpt_path), device="cpu")

    rng = np.random.default_rng(0)
    board_state = rng.integers(low=-1, high=6, size=board * board).tolist()
    info = sidecar_mod.info()
    assert info.image_size == image_size
    assert info.action_dim == 4
    assert info.max_horizon >= 1

    horizon = info.max_horizon
    n_sequences = 3
    actions = rng.integers(low=0, high=4, size=n_sequences * horizon).tolist()
    request = sidecar_mod.ScoreRequest(
        board_width=board,
        board_height=board,
        board_state=board_state,
        horizon=horizon,
        num_sequences=n_sequences,
        action_sequences_flat=actions,
    )
    response = sidecar_mod.score_actions(request)
    assert len(response.scores) == n_sequences
    assert response.horizon == horizon


def main_test() -> int:
    _assert_episode_split()
    _assert_legacy_payload_still_loads()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        jsonl_path = tmp / "rogue_transitions_v3.jsonl"
        _emit_synthetic_v3(jsonl_path, episodes=6, steps=8, board=8)

        ckpt = tmp / "final.pt"
        best = tmp / "best.pt"
        metrics_csv = tmp / "metrics.csv"

        argv = [
            "--observation-mode",
            "board-jsonl",
            "--jsonl-path",
            str(jsonl_path),
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--sequence-length",
            "3",
            "--embed-dim",
            "32",
            "--predictor-depth",
            "1",
            "--predictor-heads",
            "2",
            "--predictor-mlp-dim",
            "64",
            "--sigreg-num-proj",
            "16",
            "--items-per-epoch",
            "16",
            "--val-split",
            "0.34",
            "--lr-schedule",
            "cosine",
            "--warmup-steps",
            "2",
            "--metrics-csv",
            str(metrics_csv),
            "--output",
            str(ckpt),
            "--best-output",
            str(best),
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ]
        rc = main(argv)
        assert rc == 0

        rows = _assert_metrics_csv(metrics_csv, has_val=True)
        assert len(rows) == 2, f"expected 2 epoch rows, got {len(rows)}"
        # Cosine schedule must change the LR across epochs.
        lrs = [float(r["lr"]) for r in rows]
        assert lrs[0] != lrs[-1], f"LR did not change across epochs: {lrs}"
        # Reasonable bounds for the small dataset.
        for row in rows:
            assert float(row["train_loss"]) > 0.0
            assert float(row["val_loss"]) > 0.0

        assert ckpt.exists(), f"final checkpoint missing: {ckpt}"
        assert best.exists(), f"best checkpoint missing: {best}"
        _assert_checkpoint_payload(ckpt, expect_label="final")
        _assert_checkpoint_payload(best, expect_label="best")

        _drive_sidecar(best, board=8, image_size=32)

    print("OK: M4 training pipeline integration passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main_test())
