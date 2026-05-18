"""Round-trip check for the v3 board-state JSONL pipeline.

Generates a small JSONL file in the exact shape that
``Assets/Scripts/ML/RogueTransitionRecorder.cs`` writes for schema
``rogue.transition.v3`` (using the rolling ``SyntheticRogueDataset``
dynamics as a stand-in for Unity), then loads it with
:class:`tools.lewm.data.BoardJsonlDataset` and asserts the pixel /
action / reward / done tensor shapes line up with what the train loop
expects.

Run with::

    python -m tools.lewm.tests.test_board_jsonl

Exits 0 on success, 1 on assertion failure.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from tools.lewm.data import BoardJsonlDataset, SyntheticRogueDataset


BOARD_SIZE = 8
IMAGE_SIZE = 32
ACTION_DIM = 4
SEQUENCE_LENGTH = 4
NUM_EPISODES = 3
STEPS_PER_EPISODE = 8


def _write_fixture(path: Path) -> int:
    """Emit a JSONL file matching ``rogue.transition.v3`` from the recorder.

    Returns the number of transitions written.
    """
    rng = np.random.default_rng(0)
    builder = SyntheticRogueDataset(
        sequence_length=2,  # only used for episode bootstrap, not the loop
        board_size=BOARD_SIZE,
        image_size=IMAGE_SIZE,
        action_dim=ACTION_DIM,
        seed=0,
    )

    n_transitions = 0
    with path.open("w") as fh:
        for episode_idx in range(1, NUM_EPISODES + 1):
            state = builder._make_state(rng)
            for step_idx in range(STEPS_PER_EPISODE):
                action = int(rng.integers(0, ACTION_DIM))
                prev_board = state.board.flatten().tolist()
                next_state, reward, done = builder._step(state, action, rng)
                next_board = next_state.board.flatten().tolist()
                record = {
                    "schema": "rogue.transition.v3",
                    "timestamp": "1970-01-01T00:00:00Z",
                    "episode": episode_idx,
                    "step": step_idx,
                    "obs": [0.0] * 31,
                    "action": action,
                    "action_name": ["up", "down", "left", "right"][action],
                    "reward": reward,
                    "next_obs": [0.0] * 31,
                    "done": done,
                    "outcome": "running",
                    "level": next_state.level,
                    "food": next_state.food,
                    "board_width": BOARD_SIZE,
                    "board_height": BOARD_SIZE,
                    "board_state": prev_board,
                    "next_board_state": next_board,
                }
                fh.write(json.dumps(record) + "\n")
                n_transitions += 1
                if done:
                    break
                state = next_state
    return n_transitions


def _check(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        fixture = Path(tmp) / "rogue_transitions_v3.jsonl"
        n = _write_fixture(fixture)
        print(f"wrote {n} transitions to {fixture}")

        ds = BoardJsonlDataset(
            path=fixture,
            sequence_length=SEQUENCE_LENGTH,
            image_size=IMAGE_SIZE,
            action_dim=ACTION_DIM,
        )
        _check(len(ds) > 0, "BoardJsonlDataset windowed zero items")
        print(f"got {len(ds)} sliding windows of length {SEQUENCE_LENGTH}")

        sample = ds[0]
        _check(
            sample["pixels"].shape == (SEQUENCE_LENGTH, 3, IMAGE_SIZE, IMAGE_SIZE),
            f"pixels shape {tuple(sample['pixels'].shape)} != "
            f"({SEQUENCE_LENGTH},3,{IMAGE_SIZE},{IMAGE_SIZE})",
        )
        _check(
            sample["action"].shape == (SEQUENCE_LENGTH - 1, ACTION_DIM),
            f"action shape {tuple(sample['action'].shape)}",
        )
        _check(
            sample["reward"].shape == (SEQUENCE_LENGTH - 1,),
            f"reward shape {tuple(sample['reward'].shape)}",
        )
        _check(
            sample["done"].shape == (SEQUENCE_LENGTH - 1,),
            f"done shape {tuple(sample['done'].shape)}",
        )
        _check(
            sample["pixels"].min().item() >= 0.0
            and sample["pixels"].max().item() <= 1.0,
            "pixels not in [0, 1]",
        )
        _check(
            float(sample["action"].sum(dim=-1).max().item()) == 1.0,
            "action rows must be one-hot",
        )

    print("OK: BoardJsonlDataset round-trip passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
