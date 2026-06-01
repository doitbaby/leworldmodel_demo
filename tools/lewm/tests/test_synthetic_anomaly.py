"""Smoke test for ``tools.lewm.scripts.synthetic_anomaly``.

Builds a tiny CPU JEPA checkpoint, runs the synthetic-anomaly
evaluator against it (with ``--no-plots`` so matplotlib stays
optional in CI), and asserts:

* the summary JSON contains an entry for every anomaly kind,
* AUROC is in [0, 1],
* anomaly injection on a controlled board actually changes the
  board (i.e. ``inject_anomaly`` does not silently return its input),
* the NumPy AUROC implementation matches a hand-computed value
  on a fixed score vector.

Run with::

    python -m tools.lewm.tests.test_synthetic_anomaly

Exits 0 on success, 1 on assertion failure.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from tools.lewm.scripts.generate_jsonl_v3 import generate
from tools.lewm.scripts.synthetic_anomaly import (
    ANOMALY_TYPES,
    auroc,
    inject_anomaly,
)
from tools.lewm.scripts.synthetic_anomaly import (
    main as anomaly_main,
)
from tools.lewm.train import main as train_main


def _train_smoke_checkpoint(jsonl_path: Path, checkpoint_path: Path) -> None:
    argv = [
        "--observation-mode",
        "board-jsonl",
        "--jsonl-path",
        str(jsonl_path),
        "--epochs",
        "2",
        "--batch-size",
        "8",
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
        "--device",
        "cpu",
        "--output",
        str(checkpoint_path),
        "--best-output",
        str(checkpoint_path.with_name(checkpoint_path.stem + "_best.pt")),
        "--metrics-csv",
        str(checkpoint_path.with_suffix(".csv")),
    ]
    rc = train_main(argv)
    assert rc == 0, f"train_main exited with {rc}"


def _make_test_board() -> np.ndarray:
    """8x8 board with one player, one exit, one enemy, food and walls."""
    b = np.zeros((8, 8), dtype=np.int32)
    b[0, :] = -1
    b[-1, :] = -1
    b[:, 0] = -1
    b[:, -1] = -1
    b[1, 1] = 5  # player
    b[6, 6] = 1  # exit
    b[3, 4] = 2  # enemy
    b[2, 3] = 4  # food
    b[5, 2] = 4
    return b


def _test_auroc() -> None:
    # Two well-separated populations.
    normal = np.array([0.1, 0.2, 0.15, 0.18, 0.12])
    anomaly = np.array([0.9, 0.85, 0.95, 0.88, 0.92])
    scores = np.concatenate([normal, anomaly])
    labels = np.concatenate([np.zeros_like(normal), np.ones_like(anomaly)]).astype(int)
    val = auroc(scores, labels)
    assert abs(val - 1.0) < 1e-9, f"perfect separation should give AUROC=1.0, got {val}"

    # Overlapping populations.
    val2 = auroc(np.array([0.0, 1.0]), np.array([0, 1]))
    assert abs(val2 - 1.0) < 1e-9, val2

    val3 = auroc(np.array([1.0, 0.0]), np.array([0, 1]))
    assert abs(val3 - 0.0) < 1e-9, val3

    # Empty class -> 0.5 fallback.
    val4 = auroc(np.array([0.0, 1.0]), np.array([0, 0]))
    assert abs(val4 - 0.5) < 1e-9, val4


def _test_inject_anomaly() -> None:
    board = _make_test_board()
    rng = np.random.default_rng(0)
    for kind in ANOMALY_TYPES:
        out = inject_anomaly(board, kind, rng)
        assert out.shape == board.shape, (kind, out.shape, board.shape)
        # Anomaly should change at least one cell on every supported kind.
        assert not np.array_equal(out, board), f"{kind} did not change the board"
        if kind == "food_vanish":
            assert not np.any(out == 4), "food_vanish should remove all food cells"
        if kind == "spawn_enemy":
            # Player tile must now be an enemy.
            assert (out == 5).sum() == 0, "spawn_enemy should remove player cell"
            assert (out == 2).sum() >= 1, "spawn_enemy should keep/add at least one enemy"


def main() -> int:
    _test_auroc()
    _test_inject_anomaly()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        jsonl_path = tmp / "rogue_smoke.jsonl"
        checkpoint = tmp / "lewm_smoke.pt"
        out_dir = tmp / "anomaly"

        generate(
            output=jsonl_path,
            num_transitions=300,
            mission_ratio=0.7,
            epsilon=0.1,
            seed=0,
            board_size=8,
            image_size=32,
            food_start=100,
            food_per_level=50,
            max_steps=200,
            include_obs_stub=True,
        )
        _train_smoke_checkpoint(jsonl_path, checkpoint)

        rc = anomaly_main(
            [
                "--checkpoint",
                str(checkpoint),
                "--jsonl",
                str(jsonl_path),
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
                "--max-windows",
                "40",
                "--batch-size",
                "16",
                "--sequence-length",
                "3",
                "--no-plots",
            ]
        )
        assert rc == 0, f"anomaly_main exited with {rc}"

        summary = json.loads((out_dir / "anomaly_summary.json").read_text())
        kinds = {e["kind"] for e in summary["anomalies"]}
        assert kinds == set(ANOMALY_TYPES), (kinds, ANOMALY_TYPES)
        for entry in summary["anomalies"]:
            assert 0.0 <= entry["auroc"] <= 1.0, entry
            assert entry["n_normal"] > 0, entry
            assert entry["n_anomaly"] > 0, entry

    print("test_synthetic_anomaly: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
