"""In-process integration test for the LeWM inference sidecar.

Drives :mod:`tools.lewm.train` for a smoke checkpoint, points
:mod:`tools.lewm.sidecar` at it, and exercises ``/healthz``, ``/info``,
and ``/score_actions`` through Starlette's ``TestClient`` (no real
network). Mirrors the JSON shape the Unity ``LewmClient.cs`` will send.

Run with::

    python -m tools.lewm.tests.test_sidecar

Exits 0 on success, 1 on assertion failure.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[3]


def _check(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


def _train_smoke_checkpoint(output: Path) -> None:
    """Run ``python -m tools.lewm.train --smoke`` to produce a checkpoint."""
    rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.lewm.train",
            "--smoke",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    _check(rc.returncode == 0, f"smoke training exited {rc.returncode}")
    _check(output.exists(), f"checkpoint file not created at {output}")


def _make_board(size: int = 8) -> np.ndarray:
    board = np.zeros((size, size), dtype=np.int32)
    board[0, :] = -1
    board[-1, :] = -1
    board[:, 0] = -1
    board[:, -1] = -1
    board[size - 2, size - 2] = 1  # exit
    board[3, 4] = 4  # food
    board[2, 5] = 2  # enemy
    board[1, 1] = 5  # player
    return board


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "checkpoint.pt"
        _train_smoke_checkpoint(ckpt)
        print(f"trained smoke checkpoint: {ckpt}")

        # Late import so the train module has finished writing its files.
        from tools.lewm.sidecar import app, get_state, load_checkpoint

        load_checkpoint(str(ckpt))
        _check(get_state().ready, "sidecar state not ready after load")

        with TestClient(app) as client:
            r = client.get("/healthz")
            _check(r.status_code == 200, f"/healthz status {r.status_code}")
            body = r.json()
            _check(body["status"] == "ok", f"/healthz body {body}")
            _check(body["service"].startswith("lewm.sidecar"), f"/healthz body {body}")

            r = client.get("/info")
            _check(r.status_code == 200, f"/info status {r.status_code}: {r.text}")
            info = r.json()
            _check(info["action_dim"] == 4, f"/info action_dim {info}")
            _check(info["image_size"] == 32, f"/info image_size {info}")
            _check(info["embed_dim"] > 0, f"/info embed_dim {info}")
            _check(info["max_horizon"] >= 1, f"/info max_horizon {info}")

            horizon = info["max_horizon"]
            num_sequences = 4
            # 4 sequences, ``horizon`` steps each, one per cardinal direction.
            actions_flat: list[int] = []
            for a in range(num_sequences):
                actions_flat.extend([a] * horizon)

            board = _make_board()
            payload = {
                "board_width": 8,
                "board_height": 8,
                "board_state": board.flatten().tolist(),
                "horizon": horizon,
                "num_sequences": num_sequences,
                "action_sequences_flat": actions_flat,
                "discount": 0.95,
                "done_penalty": 1.0,
            }
            r = client.post("/score_actions", json=payload)
            _check(
                r.status_code == 200,
                f"/score_actions status {r.status_code}: {r.text}",
            )
            data = r.json()
            _check(
                len(data["scores"]) == num_sequences,
                f"/score_actions scores length {data}",
            )
            _check(data["horizon"] == horizon, f"/score_actions horizon {data}")
            for s in data["scores"]:
                _check(
                    isinstance(s, float) and abs(s) < 1e6,
                    f"score out of plausible range: {s}",
                )

            # Bad payload: board_state length mismatch.
            bad = {**payload, "board_state": [0] * 30}
            r = client.post("/score_actions", json=bad)
            _check(
                r.status_code == 400,
                f"bad board_state expected 400, got {r.status_code}: {r.text}",
            )

            # Bad payload: action_sequences_flat length mismatch.
            bad = {**payload, "action_sequences_flat": [0, 0, 0]}
            r = client.post("/score_actions", json=bad)
            _check(
                r.status_code == 400,
                f"bad action length expected 400, got {r.status_code}: {r.text}",
            )

            # Bad payload: out-of-range action index.
            bad_actions = list(actions_flat)
            bad_actions[-1] = 99
            bad = {**payload, "action_sequences_flat": bad_actions}
            r = client.post("/score_actions", json=bad)
            _check(
                r.status_code == 400,
                f"bad action index expected 400, got {r.status_code}: {r.text}",
            )

            # Bad payload: horizon longer than predictor was trained for.
            too_long = horizon + 5
            bad = {
                **payload,
                "horizon": too_long,
                "action_sequences_flat": [0] * (num_sequences * too_long),
            }
            r = client.post("/score_actions", json=bad)
            _check(
                r.status_code == 400,
                f"oversized horizon expected 400, got {r.status_code}: {r.text}",
            )

    print("OK: sidecar integration test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
