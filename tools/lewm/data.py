"""Dataset adapters for training :class:`tools.lewm.jepa.JEPA`.

Two data sources are supported:

1. :class:`SyntheticRogueDataset` -- generates fake rogue-style episodes on
   the fly (8x8 board, 32x32x3 pixel renders). Used by M1's smoke training
   when the Unity pixel pipeline (M2) is not yet in place.

2. :class:`VectorJsonlDataset` -- reads the existing v2 transition JSONL
   files emitted by ``RogueTransitionRecorder.cs`` (vector observations
   only). Lets the new training loop reuse the data the demo already has.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

__all__ = [
    "ROGUE_CELL_COLORS",
    "render_board_to_pixels",
    "SyntheticRogueDataset",
    "VectorJsonlDataset",
    "SequenceCollator",
]


# RGB triples for each cell code, chosen to be visually distinct.
# Matches the GameManager cell code convention:
#  -1 wall, 0 empty, 1 exit, 2 enemy, 3 obstacle, 4 food, 5 player.
ROGUE_CELL_COLORS: dict[int, tuple[int, int, int]] = {
    -1: (40, 40, 40),     # wall: dark grey
    0:  (220, 220, 220),  # empty: near-white
    1:  (0, 200, 80),     # exit: green
    2:  (220, 30, 30),    # enemy: red
    3:  (120, 60, 0),     # obstacle: brown
    4:  (255, 210, 0),    # food: yellow
    5:  (40, 110, 240),   # player: blue
}


def render_board_to_pixels(
    board: np.ndarray,
    image_size: int = 32,
) -> np.ndarray:
    """Render an integer ``board`` into a ``(3, image_size, image_size)`` array.

    Args:
        board: ``(H, W)`` int array with cell codes in
            :data:`ROGUE_CELL_COLORS`.
        image_size: Output spatial size; must be a multiple of ``board.shape``.
    """
    h, w = board.shape
    if image_size % h != 0 or image_size % w != 0:
        raise ValueError(f"image_size {image_size} not divisible by board {h}x{w}")
    cell = image_size // h
    img = np.zeros((image_size, image_size, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            color = ROGUE_CELL_COLORS.get(int(board[y, x]), (0, 0, 0))
            img[y * cell : (y + 1) * cell, x * cell : (x + 1) * cell] = color
    return (img / 255.0).transpose(2, 0, 1)  # to (C, H, W)


@dataclass
class _SyntheticState:
    board: np.ndarray
    player: tuple[int, int]
    food: int
    level: int


class SyntheticRogueDataset(IterableDataset):
    """Generate fake rogue episodes for smoke testing the JEPA pipeline.

    Each item is a dict with:

    * ``"pixels"`` shape ``(T, 3, image_size, image_size)``, float32 in [0,1]
    * ``"action"`` shape ``(T-1, action_dim)``, one-hot
    * ``"reward"`` shape ``(T-1,)``, float32
    * ``"done"`` shape ``(T-1,)``, float32 (0/1)
    """

    def __init__(
        self,
        sequence_length: int = 5,
        board_size: int = 8,
        image_size: int = 32,
        action_dim: int = 4,
        food_start: int = 100,
        seed: int = 0,
        items_per_epoch: int = 256,
    ):
        super().__init__()
        if sequence_length < 2:
            raise ValueError("sequence_length must be >= 2")
        self.sequence_length = sequence_length
        self.board_size = board_size
        self.image_size = image_size
        self.action_dim = action_dim
        self.food_start = food_start
        self.seed = seed
        self.items_per_epoch = items_per_epoch

    # Direction deltas (dx, dy) in (Up, Down, Left, Right) order.
    _DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def _make_state(self, rng: np.random.Generator) -> _SyntheticState:
        n = self.board_size
        board = np.zeros((n, n), dtype=np.int32)
        board[0, :] = -1
        board[-1, :] = -1
        board[:, 0] = -1
        board[:, -1] = -1

        free: list[tuple[int, int]] = [(x, y) for x in range(1, n - 1) for y in range(1, n - 1)]
        rng.shuffle(free)

        def take() -> tuple[int, int]:
            return free.pop()

        exit_cell = take()
        board[exit_cell[1], exit_cell[0]] = 1

        for _ in range(rng.integers(1, 3)):
            x, y = take()
            board[y, x] = 2  # enemy
        for _ in range(rng.integers(2, 5)):
            x, y = take()
            board[y, x] = 3  # obstacle
        for _ in range(rng.integers(2, 5)):
            x, y = take()
            board[y, x] = 4  # food

        px, py = take()
        board[py, px] = 5
        return _SyntheticState(board=board, player=(px, py), food=self.food_start, level=1)

    def _step(
        self,
        state: _SyntheticState,
        action: int,
        rng: np.random.Generator,
    ) -> tuple[_SyntheticState, float, bool]:
        dx, dy = self._DIRS[action]
        px, py = state.player
        nx, ny = px + dx, py + dy
        board = state.board.copy()
        board[py, px] = 0

        cell = int(state.board[ny, nx])
        reward = -0.01  # step cost
        done = False
        new_food = state.food - 1
        new_level = state.level

        if cell == -1 or cell == 3:
            # bumped a wall or obstacle: stay in place
            board[py, px] = 5
            nx, ny = px, py
            reward -= 0.05
        elif cell == 1:
            # exit
            reward += 1.0
            new_level += 1
        elif cell == 2:
            # enemy: take damage but survive (combat)
            new_food -= 25
            reward -= 0.5
            board[ny, nx] = 5
        elif cell == 4:
            # food
            new_food += 30
            reward += 0.3
            board[ny, nx] = 5
        else:
            board[ny, nx] = 5

        if new_food <= 0:
            done = True
            reward -= 1.0

        return (
            _SyntheticState(
                board=board,
                player=(nx, ny),
                food=max(0, new_food),
                level=new_level,
            ),
            reward,
            done,
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = np.random.default_rng(self.seed + worker_id * 7919)

        for _ in range(self.items_per_epoch):
            state = self._make_state(rng)
            pixels = [render_board_to_pixels(state.board, self.image_size)]
            actions: list[np.ndarray] = []
            rewards: list[float] = []
            dones: list[float] = []

            for _ in range(self.sequence_length - 1):
                act = int(rng.integers(0, self.action_dim))
                one_hot = np.zeros(self.action_dim, dtype=np.float32)
                one_hot[act] = 1.0
                state, r, d = self._step(state, act, rng)
                pixels.append(render_board_to_pixels(state.board, self.image_size))
                actions.append(one_hot)
                rewards.append(r)
                dones.append(float(d))
                if d:
                    # pad remaining steps with last frame so tensors stay aligned
                    while len(pixels) < self.sequence_length:
                        pixels.append(pixels[-1])
                        zero = np.zeros(self.action_dim, dtype=np.float32)
                        actions.append(zero)
                        rewards.append(0.0)
                        dones.append(1.0)
                    break

            yield {
                "pixels": torch.from_numpy(np.stack(pixels)).float(),
                "action": torch.from_numpy(np.stack(actions)).float(),
                "reward": torch.tensor(rewards, dtype=torch.float32),
                "done": torch.tensor(dones, dtype=torch.float32),
            }


class VectorJsonlDataset(Dataset):
    """Read v2 transition JSONL into fixed-length ``(obs, action, reward, done)`` windows.

    The Unity recorder writes individual transitions, not sequences. We
    chunk per ``(episode, step)`` ordering and slide a window of size
    ``sequence_length`` so the JEPA gets multi-step context.
    """

    def __init__(
        self,
        path: str | Path,
        sequence_length: int = 5,
        observation_size: int = 31,
        action_dim: int = 4,
    ):
        if sequence_length < 2:
            raise ValueError("sequence_length must be >= 2")
        self.path = Path(path)
        self.sequence_length = sequence_length
        self.observation_size = observation_size
        self.action_dim = action_dim
        self._windows: list[tuple[int, int]] = []

        episodes: dict[int, list[dict]] = {}
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ep = int(rec.get("episode", 0))
                episodes.setdefault(ep, []).append(rec)

        self._episodes: list[list[dict]] = []
        for ep in sorted(episodes):
            steps = sorted(episodes[ep], key=lambda r: int(r.get("step", 0)))
            if len(steps) < sequence_length:
                continue
            ep_idx = len(self._episodes)
            self._episodes.append(steps)
            for start in range(len(steps) - sequence_length + 1):
                self._windows.append((ep_idx, start))

    def __len__(self) -> int:
        return len(self._windows)

    def _pull_obs(self, rec: dict, key: str) -> np.ndarray:
        arr = np.asarray(rec.get(key, []), dtype=np.float32)
        if arr.shape != (self.observation_size,):
            raise ValueError(
                f"{key} expected length {self.observation_size}, got {arr.shape}"
            )
        return arr

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, start = self._windows[idx]
        steps = self._episodes[ep_idx]
        window = steps[start : start + self.sequence_length]

        obs = [self._pull_obs(window[0], "observation")]
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        dones: list[float] = []

        for rec in window[:-1]:
            obs.append(self._pull_obs(rec, "next_observation"))
            act_idx = int(rec.get("action", 0))
            one_hot = np.zeros(self.action_dim, dtype=np.float32)
            if 0 <= act_idx < self.action_dim:
                one_hot[act_idx] = 1.0
            actions.append(one_hot)
            rewards.append(float(rec.get("reward", 0.0)))
            dones.append(float(rec.get("done", False)))

        return {
            "obs": torch.from_numpy(np.stack(obs)).float(),
            "action": torch.from_numpy(np.stack(actions)).float(),
            "reward": torch.tensor(rewards, dtype=torch.float32),
            "done": torch.tensor(dones, dtype=torch.float32),
        }


@dataclass
class SequenceCollator:
    """Stack a list of dict samples into a batched dict of tensors."""

    obs_key: str = "pixels"

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        keys = batch[0].keys()
        out: dict[str, torch.Tensor] = {}
        for k in keys:
            out[k] = torch.stack([item[k] for item in batch])
        # alias `pixels` <-> `obs` so the train loop can be agnostic
        if "pixels" in out and "obs" not in out:
            out["obs"] = out["pixels"]
        if "obs" in out and "pixels" not in out:
            out["pixels"] = out["obs"]
        return out


def num_workers_for_smoke(default: int = 0) -> int:
    """Pick a safe ``num_workers`` for synthetic data on CI / small VMs."""
    cpus = max(1, math.floor((__import__("os").cpu_count() or 2) / 2))
    return min(cpus - 1, default) if default else 0
