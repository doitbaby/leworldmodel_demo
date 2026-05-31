"""Smoke test for the offline v3 JSONL generator.

Runs :mod:`tools.lewm.scripts.generate_jsonl_v3` against a small target
count, then immediately reloads the output with
:class:`tools.lewm.data.BoardJsonlDataset` to confirm the schema is
exactly what the train loop consumes. Mirrors the round-trip check in
:mod:`tools.lewm.tests.test_board_jsonl` but exercises the generator
CLI path rather than a hand-written fixture so a regression in the
generator schema gets caught in CI.

Run with::

    python -m tools.lewm.tests.test_generate_jsonl_v3

Exits 0 on success, 1 on assertion failure.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from tools.lewm.data import BoardJsonlDataset
from tools.lewm.scripts.generate_jsonl_v3 import generate

TARGET_TRANSITIONS = 400
SEQUENCE_LENGTH = 4
BOARD_SIZE = 8
IMAGE_SIZE = 32
ACTION_DIM = 4


def _required_v3_fields() -> set[str]:
    """Fields BoardJsonlDataset / downstream tooling rely on."""
    return {
        "schema",
        "episode",
        "step",
        "action",
        "reward",
        "done",
        "board_width",
        "board_height",
        "board_state",
        "next_board_state",
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "rogue_smoke.jsonl"
        summary = generate(
            output=jsonl_path,
            num_transitions=TARGET_TRANSITIONS,
            mission_ratio=0.7,
            epsilon=0.1,
            seed=0,
            board_size=BOARD_SIZE,
            image_size=IMAGE_SIZE,
            food_start=100,
            food_per_level=50,
            max_steps=200,
            include_obs_stub=True,
        )

        assert summary["transitions"] == TARGET_TRANSITIONS, summary
        assert summary["episodes"] >= 1, summary
        assert (
            summary["policy_mix"]["mission"] + summary["policy_mix"]["random"]
            == (summary["episodes"])
        ), summary

        # Spot-check the first record's shape against the recorder schema.
        with jsonl_path.open() as fh:
            first = json.loads(fh.readline())
        missing = _required_v3_fields() - set(first.keys())
        assert not missing, f"generator dropped v3 fields: {missing}"
        assert first["schema"] == "rogue.transition.v3", first["schema"]
        assert len(first["board_state"]) == BOARD_SIZE * BOARD_SIZE, first
        assert len(first["next_board_state"]) == BOARD_SIZE * BOARD_SIZE, first
        assert isinstance(first["action"], int) and 0 <= first["action"] < ACTION_DIM
        assert isinstance(first["reward"], int | float)
        assert isinstance(first["done"], bool)

        # Reload via the production loader and confirm tensor shapes.
        dataset = BoardJsonlDataset(
            path=jsonl_path,
            sequence_length=SEQUENCE_LENGTH,
            image_size=IMAGE_SIZE,
            action_dim=ACTION_DIM,
        )
        assert len(dataset) > 0, "BoardJsonlDataset loaded 0 windows from generator output"
        sample = dataset[0]
        assert tuple(sample["pixels"].shape) == (SEQUENCE_LENGTH, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert tuple(sample["action"].shape) == (SEQUENCE_LENGTH - 1, ACTION_DIM)
        assert tuple(sample["reward"].shape) == (SEQUENCE_LENGTH - 1,)
        assert tuple(sample["done"].shape) == (SEQUENCE_LENGTH - 1,)

    print("test_generate_jsonl_v3: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
