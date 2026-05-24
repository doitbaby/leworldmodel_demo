"""Python-side rogue-style env used by the M6 benchmark harness.

The Unity demo is the production environment; this module is a small
companion env that mirrors its dynamics closely enough to score
different planners against each other on the dev box (no Unity, no
GPU). It is intentionally cheap and deterministic:

* board shape and cell codes match
  :class:`tools.lewm.data.SyntheticRogueDataset` and the v3 JSONL
  schema produced by ``PixelObservationBuilder.cs``
  (``-1`` wall, ``0`` empty, ``1`` exit, ``2`` enemy, ``3`` obstacle,
  ``4`` food, ``5`` player);
* step dynamics (rewards, food deltas, level transitions) match the
  ``_step`` rules in ``SyntheticRogueDataset`` so the JEPA models
  trained on the smoke pipeline can be evaluated here without a
  distribution shift bigger than what we already accept;
* clearing the exit regenerates a fresh board with the food counter
  topped up. This is what makes ``mean_levels_cleared`` a meaningful
  metric in the M6 benchmark CSV.

This module is *not* the Unity game environment. It is a surrogate for
the benchmark harness. The Unity-side trajectories collected through
``RogueTransitionRecorder.cs`` remain the ground truth for real
training; the env here is only for the offline comparison sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import render_board_to_pixels

__all__ = ["EnvObservation", "RogueSimEnv"]


# Direction deltas (dx, dy) in (Up, Down, Left, Right) order.
# Must stay consistent with :class:`RogueObservationBuilder` /
# :class:`tools.lewm.data.SyntheticRogueDataset`.
_DIRS: tuple[tuple[int, int], ...] = ((0, -1), (0, 1), (-1, 0), (1, 0))


@dataclass
class EnvObservation:
    """One time-step observation returned by :class:`RogueSimEnv`."""

    board: np.ndarray  # (H, W) int32 cell codes
    pixels: np.ndarray  # (3, image_size, image_size) float32 in [0, 1]
    player: tuple[int, int]
    food: int
    level: int
    step: int
    last_event: str = "init"  # 'move' | 'wall' | 'obstacle' | 'enemy' | 'food' | 'exit'

    def copy(self) -> EnvObservation:
        return EnvObservation(
            board=self.board.copy(),
            pixels=self.pixels.copy(),
            player=self.player,
            food=self.food,
            level=self.level,
            step=self.step,
            last_event=self.last_event,
        )


@dataclass
class _EnvConfig:
    board_size: int = 8
    image_size: int = 32
    food_start: int = 100
    food_per_level: int = 50
    max_steps: int = 200
    # Rewards (mirror SyntheticRogueDataset._step exactly).
    step_cost: float = -0.01
    bump_penalty: float = -0.05
    exit_reward: float = 1.0
    enemy_reward: float = -0.5
    enemy_food_loss: int = 25
    food_reward: float = 0.3
    food_gain: int = 30
    starvation_penalty: float = -1.0


class RogueSimEnv:
    """Lightweight rogue env used by ``tools/lewm/benchmark.py``.

    Action space is :data:`tools.lewm.data.SyntheticRogueDataset._DIRS`
    (4 discrete moves). Episode terminates when the player runs out of
    food or when ``max_steps`` is reached.

    Notes:
        * The board is regenerated when the player steps onto an exit
          cell. ``info["level_cleared"] == True`` flags that transition
          so the benchmark can exclude it from dynamics-loss
          computation (a JEPA model cannot be expected to predict a
          fresh, unseen board).
        * Reset is deterministic at a fixed seed; pass ``seed=None`` to
          :meth:`step` ``reset`` to keep the previous PRNG stream.
    """

    DIRS = _DIRS

    def __init__(
        self,
        *,
        board_size: int = 8,
        image_size: int = 32,
        food_start: int = 100,
        food_per_level: int = 50,
        max_steps: int = 200,
        seed: int = 0,
    ):
        if board_size < 4:
            raise ValueError("board_size must be >= 4 (need walls + interior)")
        if image_size % board_size != 0:
            raise ValueError(
                f"image_size {image_size} must be divisible by board_size {board_size}"
            )
        self.cfg = _EnvConfig(
            board_size=board_size,
            image_size=image_size,
            food_start=food_start,
            food_per_level=food_per_level,
            max_steps=max_steps,
        )
        self._rng = np.random.default_rng(seed)
        self._board: np.ndarray
        self._player: tuple[int, int] = (0, 0)
        self._food: int = food_start
        self._level: int = 1
        self._step: int = 0
        self._done: bool = True  # forces reset before first step
        self._last_event: str = "init"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return len(_DIRS)

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def board(self) -> np.ndarray:
        return self._board.copy()

    def reset(self, seed: int | None = None) -> EnvObservation:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._spawn_board()
        self._food = self.cfg.food_start
        self._level = 1
        self._step = 0
        self._done = False
        self._last_event = "init"
        return self._observation()

    def step(self, action: int) -> tuple[EnvObservation, float, bool, dict]:
        if self._done:
            raise RuntimeError("step() called on terminated env; call reset() first")
        if not 0 <= action < len(_DIRS):
            raise ValueError(f"action must be in [0, {len(_DIRS)}), got {action}")

        dx, dy = _DIRS[action]
        px, py = self._player
        nx, ny = px + dx, py + dy

        reward = self.cfg.step_cost
        info: dict = {"level_cleared": False, "died": False}
        event = "move"

        # Clear current player overlay; we'll set it back at the new spot.
        self._board[py, px] = 0

        cell = int(self._board[ny, nx])

        if cell in (-1, 3):
            # Wall or obstacle: bump and stay.
            self._board[py, px] = 5
            reward += self.cfg.bump_penalty
            event = "wall" if cell == -1 else "obstacle"
        elif cell == 1:
            # Exit: regenerate board, increment level, top food up.
            reward += self.cfg.exit_reward
            self._level += 1
            self._food = min(self.cfg.food_start, self._food + self.cfg.food_per_level)
            info["level_cleared"] = True
            event = "exit"
            # Spawning a new board also overwrites _player and the
            # board layout, so we return immediately after step bookkeeping.
            self._spawn_board()
        elif cell == 2:
            # Enemy: cross the cell but take damage.
            self._food -= self.cfg.enemy_food_loss
            reward += self.cfg.enemy_reward
            self._board[ny, nx] = 5
            self._player = (nx, ny)
            event = "enemy"
        elif cell == 4:
            # Food pickup.
            self._food += self.cfg.food_gain
            reward += self.cfg.food_reward
            self._board[ny, nx] = 5
            self._player = (nx, ny)
            event = "food"
        else:
            # Empty cell.
            self._board[ny, nx] = 5
            self._player = (nx, ny)
            event = "move"

        # Step counter + termination.
        self._step += 1
        self._food -= 1  # base hunger drain on every step (matches Unity demo)
        if self._food <= 0:
            self._food = 0
            self._done = True
            reward += self.cfg.starvation_penalty
            info["died"] = True
            event = event if event != "move" else "starve"
        if self._step >= self.cfg.max_steps:
            self._done = True

        self._last_event = event
        info["event"] = event
        info["food"] = self._food
        info["level"] = self._level
        return self._observation(), float(reward), self._done, info

    def pixel_obs(self) -> np.ndarray:
        return render_board_to_pixels(self._board, self.cfg.image_size)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _observation(self) -> EnvObservation:
        return EnvObservation(
            board=self._board.copy(),
            pixels=self.pixel_obs(),
            player=self._player,
            food=int(self._food),
            level=int(self._level),
            step=int(self._step),
            last_event=self._last_event,
        )

    def _spawn_board(self) -> None:
        n = self.cfg.board_size
        board = np.zeros((n, n), dtype=np.int32)
        board[0, :] = -1
        board[-1, :] = -1
        board[:, 0] = -1
        board[:, -1] = -1

        interior = [(x, y) for x in range(1, n - 1) for y in range(1, n - 1)]
        self._rng.shuffle(interior)

        def take() -> tuple[int, int]:
            return interior.pop()

        exit_cell = take()
        board[exit_cell[1], exit_cell[0]] = 1

        n_enemies = int(self._rng.integers(1, 3))
        for _ in range(n_enemies):
            if not interior:
                break
            x, y = take()
            board[y, x] = 2

        n_obstacles = int(self._rng.integers(2, 5))
        for _ in range(n_obstacles):
            if not interior:
                break
            x, y = take()
            board[y, x] = 3

        n_food = int(self._rng.integers(2, 5))
        for _ in range(n_food):
            if not interior:
                break
            x, y = take()
            board[y, x] = 4

        if not interior:
            raise RuntimeError("board too small: no free cells left for the player spawn")
        px, py = take()
        board[py, px] = 5
        self._board = board
        self._player = (px, py)
