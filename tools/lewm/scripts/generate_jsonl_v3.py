"""Offline generator for ``rogue.transition.v3`` JSONL training data.

The Unity demo's :class:`RogueTransitionRecorder` writes one v3 JSONL
line per (state, action, next_state) tuple as the agent plays the
in-engine game. When Unity is not available (cloud GPU notebooks,
headless CI, fast iteration during port work), this script produces
the **same** JSONL shape from the Python surrogate
:class:`tools.lewm.env.RogueSimEnv` so the JEPA training and benchmark
pipelines under :mod:`tools.lewm.train` /
:mod:`tools.lewm.benchmark` can run end-to-end on cell-code grids
without a running Unity Editor.

Two rollout policies are mixed per episode:

* ``mission`` -- BFS toward the exit cell with an adjacent-food magnet
  and a hard mask on moves that step directly into an enemy. Mirrors
  the in-engine ``BrainPlanner.ChooseMissionTarget`` /
  ``BrainPlanner.BuildMissionFutureActions`` heuristic the M6
  benchmark already implements (see
  :func:`tools.lewm.benchmark._bfs_next_step`,
  :func:`tools.lewm.benchmark._adjacent_food_action`,
  :func:`tools.lewm.benchmark._adjacent_enemy_mask`).
* ``random`` -- uniform random over the 4 discrete moves. Keeps the
  data distribution from collapsing onto one near-optimal trajectory,
  which is what JEPA needs to learn dynamics across the whole
  state-action grid.

The mix is controlled by ``--mission-ratio`` (default 0.7 mission /
0.3 random episodes) and an in-episode ``--epsilon`` (default 0.1) that
replaces individual mission moves with a uniform random action so the
mission rollouts still explore.

Output schema matches ``Assets/Scripts/ML/RogueTransitionRecorder.cs``
v3 line-for-line: ``schema``, ``timestamp``, ``episode``, ``step``,
``obs``, ``action``, ``action_name``, ``reward``, ``next_obs``,
``done``, ``outcome``, ``level``, ``food``, ``board_width``,
``board_height``, ``board_state``, ``next_board_state``. The 31-d
``obs``/``next_obs`` fields are zero-padded because :class:`BoardJsonlDataset`
ignores them and the C# 31-d feature is non-trivial to reproduce
without GameManager. Set ``--include-obs-stub False`` to omit them.

Usage::

    python -m tools.lewm.scripts.generate_jsonl_v3 \\
        --output data/rogue_transitions.jsonl \\
        --num-transitions 6000 \\
        --mission-ratio 0.7 \\
        --epsilon 0.1 \\
        --seed 0

Verify it loads with :class:`BoardJsonlDataset`::

    python -m tools.lewm.scripts.generate_jsonl_v3 \\
        --output /tmp/rogue.jsonl --num-transitions 200 --verify
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from tools.lewm.benchmark import (
    _adjacent_enemy_mask,
    _adjacent_food_action,
    _bfs_next_step,
)
from tools.lewm.env import RogueSimEnv

_ACTION_NAMES: tuple[str, ...] = ("up", "down", "left", "right")
_SCHEMA = "rogue.transition.v3"


def _mission_action(
    board: np.ndarray,
    player: tuple[int, int],
    rng: np.random.Generator,
    epsilon: float,
) -> int:
    """Pick an action using the same heuristic as the M6 ``mission`` mode.

    Priority order (matches ``BrainPlanner`` semantics):

    1. With probability ``epsilon`` take a uniform random non-enemy
       move so the dataset still explores around the BFS path.
    2. Prefer stepping onto adjacent food when available -- this
       mirrors the in-engine ``ChooseMissionTarget`` food magnet and
       keeps the agent alive longer (more transitions / episode).
    3. Otherwise BFS toward the exit, masking moves that walk into
       enemies. Falls back to a random non-enemy direction when BFS
       cannot find a path (closed-off start cell).
    """
    enemy_mask = _adjacent_enemy_mask(board, player)
    safe_actions = [i for i, blocked in enumerate(enemy_mask) if not blocked]
    if not safe_actions:
        # Every direction lands on an enemy; pick any move (the env
        # will resolve damage). Keeps the recorder running rather than
        # stalling on an empty action choice.
        return int(rng.integers(0, len(_ACTION_NAMES)))

    if rng.random() < epsilon:
        return int(rng.choice(safe_actions))

    food_action = _adjacent_food_action(board, player)
    if food_action is not None and food_action in safe_actions:
        return food_action

    bfs_action = _bfs_next_step(board, player)
    if bfs_action is not None and bfs_action in safe_actions:
        return bfs_action

    return int(rng.choice(safe_actions))


def _episode_policy(
    name: str,
    rng: np.random.Generator,
    epsilon: float,
):
    """Return a ``(env, obs) -> action`` callable for the given policy name."""
    if name == "random":

        def pick(env: RogueSimEnv, obs) -> int:
            del env  # unused
            del obs
            return int(rng.integers(0, len(_ACTION_NAMES)))

        return pick

    if name == "mission":

        def pick(env: RogueSimEnv, obs) -> int:
            del env
            return _mission_action(obs.board, obs.player, rng, epsilon)

        return pick

    raise ValueError(f"unknown policy name: {name!r}")


def _outcome_label(reward: float, done: bool, info: dict) -> str:
    """Human-readable outcome tag mirroring the Unity recorder."""
    if info.get("died"):
        return "died"
    if info.get("level_cleared"):
        return "exit"
    event = str(info.get("event", "move"))
    if event in {"wall", "obstacle"}:
        return "bump"
    if event == "enemy":
        return "enemy_hit"
    if event == "food":
        return "food_pickup"
    return "running" if not done else "timeout"


def _emit_record(
    *,
    episode: int,
    step: int,
    action: int,
    reward: float,
    done: bool,
    outcome: str,
    level: int,
    food: int,
    board_width: int,
    board_height: int,
    board_state: Iterable[int],
    next_board_state: Iterable[int],
    include_obs_stub: bool,
) -> dict:
    """Build a single v3 JSONL record as a dict."""
    record: dict = {
        "schema": _SCHEMA,
        "timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "episode": episode,
        "step": step,
        "action": int(action),
        "action_name": _ACTION_NAMES[int(action)],
        "reward": float(reward),
        "done": bool(done),
        "outcome": outcome,
        "level": int(level),
        "food": int(food),
        "board_width": int(board_width),
        "board_height": int(board_height),
        "board_state": list(int(c) for c in board_state),
        "next_board_state": list(int(c) for c in next_board_state),
    }
    if include_obs_stub:
        # 31-d zero vectors so VectorJsonlDataset / legacy v2 readers
        # don't blow up if someone points them at this file. The real
        # 31-d feature is computed from GameManager state inside
        # Unity; reproducing it here would duplicate gameplay logic
        # for fields BoardJsonlDataset never touches.
        record["obs"] = [0.0] * 31
        record["next_obs"] = [0.0] * 31
    return record


def generate(
    *,
    output: Path,
    num_transitions: int,
    mission_ratio: float,
    epsilon: float,
    seed: int,
    board_size: int,
    image_size: int,
    food_start: int,
    food_per_level: int,
    max_steps: int,
    include_obs_stub: bool,
) -> dict:
    """Write ``num_transitions`` v3 JSONL lines to ``output``.

    Returns a small summary dict with episode / transition counts and
    policy mix for logging.
    """
    if num_transitions <= 0:
        raise ValueError("num_transitions must be positive")
    if not 0.0 <= mission_ratio <= 1.0:
        raise ValueError("mission_ratio must be in [0, 1]")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")

    rng = np.random.default_rng(seed)
    policy_rng = np.random.default_rng(seed + 1)
    env = RogueSimEnv(
        board_size=board_size,
        image_size=image_size,
        food_start=food_start,
        food_per_level=food_per_level,
        max_steps=max_steps,
        seed=seed,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    transitions_written = 0
    episodes_written = 0
    counts = {"mission": 0, "random": 0}

    with output.open("w", encoding="utf-8") as fh:
        episode_idx = 0
        while transitions_written < num_transitions:
            episode_idx += 1
            policy_name = "mission" if rng.random() < mission_ratio else "random"
            counts[policy_name] += 1
            pick_action = _episode_policy(policy_name, policy_rng, epsilon)

            obs = env.reset(seed=seed + episode_idx * 13)
            step_idx = 0
            while not env.is_done and transitions_written < num_transitions:
                prev_board = env.board.flatten().tolist()
                board_w = int(env.board.shape[1])
                board_h = int(env.board.shape[0])
                action = pick_action(env, obs)
                next_obs, reward, done, info = env.step(action)
                next_board = env.board.flatten().tolist()
                outcome = _outcome_label(reward, done, info)
                record = _emit_record(
                    episode=episode_idx,
                    step=step_idx,
                    action=action,
                    reward=reward,
                    done=done,
                    outcome=outcome,
                    level=int(info.get("level", next_obs.level)),
                    food=int(info.get("food", next_obs.food)),
                    board_width=board_w,
                    board_height=board_h,
                    board_state=prev_board,
                    next_board_state=next_board,
                    include_obs_stub=include_obs_stub,
                )
                fh.write(json.dumps(record) + "\n")
                transitions_written += 1
                step_idx += 1
                obs = next_obs

            episodes_written = episode_idx

    return {
        "transitions": transitions_written,
        "episodes": episodes_written,
        "policy_mix": counts,
        "output": str(output),
    }


def _verify(path: Path, sequence_length: int = 4) -> dict:
    """Load ``path`` with :class:`BoardJsonlDataset` and return a summary."""
    # Imported lazily so callers that just want to generate don't pull
    # torch into the import graph.
    import torch  # noqa: F401

    from tools.lewm.data import BoardJsonlDataset

    dataset = BoardJsonlDataset(
        path=path,
        sequence_length=sequence_length,
        image_size=32,
        action_dim=4,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"BoardJsonlDataset loaded 0 windows from {path}; "
            "schema or sequence_length probably mismatched"
        )
    sample = dataset[0]
    return {
        "windows": len(dataset),
        "pixels_shape": tuple(sample["pixels"].shape),
        "action_shape": tuple(sample["action"].shape),
        "reward_shape": tuple(sample["reward"].shape),
        "done_shape": tuple(sample["done"].shape),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_jsonl_v3",
        description=(
            "Generate rogue.transition.v3 JSONL training data from "
            "RogueSimEnv (no Unity required)."
        ),
    )
    p.add_argument("--output", type=Path, required=True, help="output .jsonl path")
    p.add_argument(
        "--num-transitions",
        type=int,
        default=6000,
        help="target number of (state, action, next_state) lines",
    )
    p.add_argument(
        "--mission-ratio",
        type=float,
        default=0.7,
        help="fraction of episodes that use the BFS mission policy",
    )
    p.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="within mission episodes, prob. of substituting a random action",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--board-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--food-start", type=int, default=100)
    p.add_argument("--food-per-level", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument(
        "--no-obs-stub",
        action="store_true",
        help="omit the 31-d obs/next_obs zero-padded stub fields",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="after writing, load with BoardJsonlDataset and print shapes",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    # Make Python's stdlib `random` deterministic too, for any library
    # that picks up the global RNG.
    random.seed(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

    summary = generate(
        output=args.output,
        num_transitions=args.num_transitions,
        mission_ratio=args.mission_ratio,
        epsilon=args.epsilon,
        seed=args.seed,
        board_size=args.board_size,
        image_size=args.image_size,
        food_start=args.food_start,
        food_per_level=args.food_per_level,
        max_steps=args.max_steps,
        include_obs_stub=not args.no_obs_stub,
    )
    print(json.dumps({"generate": summary}, indent=2))

    if args.verify:
        verify_summary = _verify(args.output)
        print(json.dumps({"verify": verify_summary}, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
