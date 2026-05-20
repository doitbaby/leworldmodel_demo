"""In-process integration test for the LeWM inference sidecar.

Drives :mod:`tools.lewm.train` for a smoke checkpoint, points
:mod:`tools.lewm.sidecar` at it, and exercises ``/healthz``, ``/info``,
``/score_actions``, and ``/plan_actions`` through Starlette's
``TestClient`` (no real network). Mirrors the JSON shape the Unity
``LewmClient.cs`` will send.

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

            # ----------------------------------------------------------
            # /plan_actions (M5)
            # ----------------------------------------------------------
            plan_payload = {
                "board_width": 8,
                "board_height": 8,
                "board_state": board.flatten().tolist(),
                "horizon": horizon,
                "num_candidates": 16,
                "top_k": 3,
                "discount": 0.95,
                "done_penalty": 1.0,
                "seed": 4242,
            }
            r = client.post("/plan_actions", json=plan_payload)
            _check(
                r.status_code == 200,
                f"/plan_actions status {r.status_code}: {r.text}",
            )
            plan = r.json()
            _check(
                len(plan["best_actions"]) == horizon,
                f"/plan_actions best_actions length {plan}",
            )
            _check(
                plan["top_k"] == 3, f"/plan_actions top_k {plan}",
            )
            _check(
                len(plan["top_k_scores"]) == plan["top_k"],
                f"/plan_actions top_k_scores length {plan}",
            )
            _check(
                len(plan["top_k_actions_flat"]) == plan["top_k"] * horizon,
                f"/plan_actions top_k_actions_flat length {plan}",
            )
            _check(
                plan["num_candidates"] == 16,
                f"/plan_actions num_candidates {plan}",
            )
            # top-k must be sorted best-first
            for i in range(1, len(plan["top_k_scores"])):
                _check(
                    plan["top_k_scores"][i - 1] >= plan["top_k_scores"][i] - 1e-6,
                    f"/plan_actions top_k not sorted at i={i}: {plan['top_k_scores']}",
                )
            # best_actions equals the first row of top_k_actions_flat
            _check(
                plan["best_actions"]
                == plan["top_k_actions_flat"][:horizon],
                f"/plan_actions best_actions vs top_k_actions_flat row 0 mismatch: {plan}",
            )

            # Determinism: same seed produces identical plan via HTTP.
            r2 = client.post("/plan_actions", json=plan_payload)
            _check(r2.status_code == 200, f"/plan_actions retry status {r2.status_code}")
            plan2 = r2.json()
            _check(
                plan2["best_actions"] == plan["best_actions"],
                f"/plan_actions determinism: best_actions differ\n  a={plan['best_actions']}\n  b={plan2['best_actions']}",
            )
            _check(
                abs(plan2["best_score"] - plan["best_score"]) < 1e-5,
                f"/plan_actions determinism: best_score {plan['best_score']} vs {plan2['best_score']}",
            )

            # Bad payload: board_state length mismatch.
            bad = {**plan_payload, "board_state": [0] * 17}
            r = client.post("/plan_actions", json=bad)
            _check(
                r.status_code == 400,
                f"/plan_actions bad board expected 400, got {r.status_code}: {r.text}",
            )

            # Bad payload: oversized horizon.
            bad = {**plan_payload, "horizon": horizon + 5}
            r = client.post("/plan_actions", json=bad)
            _check(
                r.status_code == 400,
                f"/plan_actions oversized horizon expected 400, got {r.status_code}: {r.text}",
            )

            # Bad payload: num_candidates out of bounds (pydantic 422).
            bad = {**plan_payload, "num_candidates": 0}
            r = client.post("/plan_actions", json=bad)
            _check(
                r.status_code in (400, 422),
                f"/plan_actions zero candidates expected 4xx, got {r.status_code}: {r.text}",
            )

    print("OK: sidecar integration test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
