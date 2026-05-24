"""Unit checks for :class:`tools.lewm.env.RogueSimEnv`.

Run with::

    python -m tools.lewm.tests.test_env
"""

from __future__ import annotations

import sys

import numpy as np

from tools.lewm.env import RogueSimEnv


def _player_count(board: np.ndarray) -> int:
    return int(np.count_nonzero(board == 5))


def _exit_count(board: np.ndarray) -> int:
    return int(np.count_nonzero(board == 1))


def test_reset_returns_valid_observation() -> None:
    env = RogueSimEnv(board_size=8, image_size=32, seed=0)
    obs = env.reset()
    assert obs.board.shape == (8, 8)
    assert obs.pixels.shape == (3, 32, 32)
    assert obs.food == env.cfg.food_start
    assert obs.level == 1
    assert obs.step == 0
    assert _player_count(obs.board) == 1
    assert _exit_count(obs.board) == 1
    assert obs.player == tuple(map(int, np.argwhere(obs.board == 5)[0][::-1]))


def test_step_advances_player_on_empty_cell() -> None:
    env = RogueSimEnv(board_size=8, seed=1)
    obs = env.reset()
    # Try every direction and find one that lands on an empty cell.
    moved = False
    for action, (dx, dy) in enumerate(RogueSimEnv.DIRS):
        nx, ny = obs.player[0] + dx, obs.player[1] + dy
        if not (0 <= nx < 8 and 0 <= ny < 8):
            continue
        if int(obs.board[ny, nx]) != 0:
            continue
        next_obs, reward, done, info = env.step(action)
        assert next_obs.player == (nx, ny)
        assert reward < 0  # step cost dominates an empty move
        assert _player_count(next_obs.board) == 1
        assert next_obs.step == 1
        assert next_obs.food == env.cfg.food_start - 1
        moved = True
        break
    assert moved, "RNG should produce at least one walkable neighbour"


def test_wall_bump_keeps_player_in_place() -> None:
    env = RogueSimEnv(board_size=8, seed=2)
    obs = env.reset()
    # Find a wall neighbour.
    bump_action = None
    for action, (dx, dy) in enumerate(RogueSimEnv.DIRS):
        nx, ny = obs.player[0] + dx, obs.player[1] + dy
        if 0 <= nx < 8 and 0 <= ny < 8 and int(obs.board[ny, nx]) == -1:
            bump_action = action
            break
    if bump_action is None:
        # The interior spawn does not always touch the border; manually move
        # the player onto a border-adjacent cell then bump.
        # Find the wall cell at (0, 1) and place the player adjacent to it.
        env._board[obs.player[1], obs.player[0]] = 0
        env._board[1, 1] = 5
        env._player = (1, 1)
        bump_action = 2  # Left (toward x=0 wall)
    pre_player = env._player
    pre_food = env._food
    obs_next, reward, done, info = env.step(bump_action)
    assert obs_next.player == pre_player, "bump should not move player"
    assert obs_next.food == pre_food - 1, "hunger drain still applies on a bump"
    assert reward <= env.cfg.bump_penalty + env.cfg.step_cost + 1e-6


def test_food_pickup_increases_food() -> None:
    env = RogueSimEnv(board_size=8, seed=3)
    obs = env.reset()
    # Place a food cell directly to the right of the player for determinism.
    px, py = obs.player
    if px + 1 >= 8 - 1:
        # Avoid the border wall: move spawn left.
        env._board[py, px] = 0
        env._board[py, 2] = 5
        env._player = (2, py)
        px = 2
    env._board[py, px + 1] = 4
    pre_food = env._food
    obs_next, reward, done, info = env.step(3)  # Right
    assert obs_next.player == (px + 1, py)
    assert obs_next.food == pre_food + env.cfg.food_gain - 1  # +30 minus hunger
    assert reward > 0


def test_exit_regenerates_board_and_increments_level() -> None:
    env = RogueSimEnv(board_size=8, seed=4)
    obs = env.reset()
    # Drop an exit one step to the right and clear the existing exit.
    env._board[env._board == 1] = 0
    px, py = obs.player
    if px + 1 >= 7:
        env._board[py, px] = 0
        env._board[py, 2] = 5
        env._player = (2, py)
        px = 2
    env._board[py, px + 1] = 1
    pre_level = env._level
    obs_next, reward, done, info = env.step(3)  # Right onto exit
    assert info["level_cleared"] is True
    assert obs_next.level == pre_level + 1
    assert _exit_count(obs_next.board) == 1, "fresh board still has an exit"
    assert _player_count(obs_next.board) == 1
    assert reward >= env.cfg.exit_reward + env.cfg.step_cost - 1e-6


def test_food_depletion_terminates_episode() -> None:
    env = RogueSimEnv(board_size=8, food_start=2, max_steps=100, seed=5)
    obs = env.reset()
    # Keep moving in directions until food drains.
    rng = np.random.default_rng(0)
    steps = 0
    while not env.is_done and steps < 50:
        action = int(rng.integers(0, env.action_dim))
        obs, reward, done, info = env.step(action)
        steps += 1
    assert env.is_done
    assert obs.food == 0
    assert info.get("died", False) is True


def test_step_after_done_raises() -> None:
    env = RogueSimEnv(board_size=8, food_start=1, seed=6)
    obs = env.reset()
    env.step(0)
    # Food was 1, now 0, env should be done.
    assert env.is_done
    try:
        env.step(0)
    except RuntimeError as e:
        assert "terminated" in str(e)
        return
    raise AssertionError("expected RuntimeError when stepping a done env")


def test_max_steps_terminates_episode() -> None:
    env = RogueSimEnv(board_size=8, food_start=10_000, max_steps=10, seed=7)
    env.reset()
    rng = np.random.default_rng(0)
    while not env.is_done:
        env.step(int(rng.integers(0, env.action_dim)))
    assert env._step == env.cfg.max_steps


def test_image_divisibility_validation() -> None:
    try:
        RogueSimEnv(board_size=8, image_size=33, seed=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-divisible image size")


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print("env tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
