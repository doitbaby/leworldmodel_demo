"""M6 benchmark harness: compare planners on the rogue surrogate env.

Runs one or more "modes" against :class:`tools.lewm.env.RogueSimEnv` and
writes a CSV row per ``(seed, mode, episode_idx)`` plus a summary row
per ``(seed, mode)``. The target metrics requested up front are:

* ``levels_cleared`` (how often the agent steps through the exit cell);
* ``food_left`` at episode end (proxy for "stayed alive");
* ``dynamics_loss`` (mean MSE of the JEPA next-embedding prediction
  against the observed next embedding -- only defined for JEPA-based
  modes; ``nan`` otherwise).

Five modes are shipped:

* ``random`` -- uniform random actions. Baseline floor.
* ``mission`` -- shortest path to the exit via BFS over the cell-code
  grid. Mirrors the in-engine ``BrainPlanner.ChooseMissionTarget`` /
  ``BuildMissionFutureActions`` heuristic with no learned model.
* ``mlp_lite`` -- alias for ``mission`` plus a tiny food/enemy avoidance
  rule from ``Assets/StreamingAssets/world_model_weights.json`` if the
  file is present; falls back to ``mission`` otherwise. This is a
  surrogate for the in-engine "LeWM-lite" path (the real MLP needs the
  31-d vector observation that only Unity builds).
* ``lewm_no_planner`` -- score each candidate single action through
  the JEPA reward+done heads and pick the argmax. No multi-step
  rollout.
* ``lewm_dreamer`` -- :func:`tools.lewm.planner.random_shooting`. This
  is the M5 actor-critic style planner.

Usage (smoke, all modes, 1 episode each)::

    python -m tools.lewm.benchmark --smoke \
        --checkpoint results/lewm/checkpoint.pt \
        --output-csv results/lewm/benchmark.csv

Outputs:

* ``<output-csv>``: per-episode rows ``seed,mode,episode_idx,
  levels_cleared,food_left,steps,episode_return,dynamics_loss,
  reward_loss,died``
* ``<output-csv>.summary.csv``: per-(seed, mode) means and stds across
  the episodes for the same columns. The summary is intentionally
  separate so plotting tools can read either granularity directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .env import RogueSimEnv
from .planner import random_shooting
from .train import build_model, load_config_from_payload

__all__ = ["BenchmarkConfig", "EpisodeResult", "ModeRunner", "run_benchmark", "main"]


_ALL_MODES = ("random", "mission", "mlp_lite", "lewm_no_planner", "lewm_dreamer")
_JEPA_MODES = {"lewm_no_planner", "lewm_dreamer"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bfs_next_step(board: np.ndarray, start: tuple[int, int]) -> int | None:
    """Return the action index that moves ``start`` one step toward the exit.

    Returns ``None`` when the exit is unreachable from ``start`` (closed in
    by walls / obstacles / enemies). Treats enemies and obstacles as
    impassable; only walks across empty / food cells. Matches the
    in-engine ``BrainPlanner.IsCellWalkable`` semantics closely enough
    for the surrogate.
    """
    h, w = board.shape
    sx, sy = start
    target: tuple[int, int] | None = None
    for y in range(h):
        for x in range(w):
            if board[y, x] == 1:
                target = (x, y)
                break
        if target is not None:
            break
    if target is None:
        return None

    queue: deque[tuple[int, int]] = deque([target])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {target: None}
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in RogueSimEnv.DIRS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if (nx, ny) in prev:
                continue
            cell = int(board[ny, nx])
            # Walkable into: empty, food, player (start). Block walls /
            # obstacles / enemies / the exit itself (we already came
            # from there). The start tile may carry the player code 5.
            if cell == -1 or cell == 2 or cell == 3:
                continue
            prev[(nx, ny)] = (cx, cy)
            if (nx, ny) == (sx, sy):
                # Walk back from start to identify the first move.
                step = prev[(nx, ny)]
                if step is None:
                    return None
                first_dx = step[0] - sx
                first_dy = step[1] - sy
                for idx, (ddx, ddy) in enumerate(RogueSimEnv.DIRS):
                    if ddx == first_dx and ddy == first_dy:
                        return idx
                return None
            queue.append((nx, ny))
    return None


def _adjacent_food_action(board: np.ndarray, start: tuple[int, int]) -> int | None:
    """Return an action that walks onto an adjacent food cell, if any."""
    sx, sy = start
    h, w = board.shape
    for idx, (dx, dy) in enumerate(RogueSimEnv.DIRS):
        nx, ny = sx + dx, sy + dy
        if 0 <= nx < w and 0 <= ny < h and int(board[ny, nx]) == 4:
            return idx
    return None


def _adjacent_enemy_mask(board: np.ndarray, start: tuple[int, int]) -> list[bool]:
    """Mask actions that walk directly into an enemy."""
    sx, sy = start
    h, w = board.shape
    out: list[bool] = []
    for dx, dy in RogueSimEnv.DIRS:
        nx, ny = sx + dx, sy + dy
        if 0 <= nx < w and 0 <= ny < h and int(board[ny, nx]) == 2:
            out.append(True)
        else:
            out.append(False)
    return out


# ---------------------------------------------------------------------------
# Mode runner protocol
# ---------------------------------------------------------------------------


@dataclass
class ModeRunner:
    """One row in the benchmark mode matrix."""

    name: str
    pick_action: callable  # (env, obs, rng) -> int
    requires_jepa: bool = False

    def __call__(self, env: RogueSimEnv, obs, rng: np.random.Generator) -> int:
        return self.pick_action(env, obs, rng)


def _make_random_mode() -> ModeRunner:
    def pick(env: RogueSimEnv, obs, rng: np.random.Generator) -> int:
        return int(rng.integers(0, env.action_dim))

    return ModeRunner(name="random", pick_action=pick)


def _make_mission_mode(*, prefer_adjacent_food: bool = False) -> ModeRunner:
    def pick(env: RogueSimEnv, obs, rng: np.random.Generator) -> int:
        board = obs.board
        # Heuristic 1: grab adjacent food when available (optional).
        if prefer_adjacent_food:
            food_action = _adjacent_food_action(board, obs.player)
            if food_action is not None:
                return food_action
        # Heuristic 2: BFS toward the exit.
        bfs_action = _bfs_next_step(board, obs.player)
        if bfs_action is not None:
            # Avoid walking directly into an enemy if alternatives exist.
            enemy_mask = _adjacent_enemy_mask(board, obs.player)
            if enemy_mask[bfs_action]:
                # Try any safe direction.
                for idx, dangerous in enumerate(enemy_mask):
                    if not dangerous:
                        return idx
            return bfs_action
        # Fallback: random.
        return int(rng.integers(0, env.action_dim))

    return ModeRunner(name="mission", pick_action=pick)


def _make_mlp_lite_mode(
    weights_path: Path | None,
) -> ModeRunner:
    """Surrogate for the in-engine LeWM-lite path.

    The Python env does not build the 31-d vector observation the MLP
    expects, so we cannot evaluate the actual MLP forward pass here.
    We instead apply a behavioural-cloned heuristic: BFS to exit with
    explicit food preference + enemy avoidance, plus a tiny food
    pickup bias when the on-disk MLP exists (used as a marker that the
    user has trained the demo path). When the weights file is absent
    the mode degrades silently to ``mission``.
    """
    weights_available = weights_path is not None and weights_path.exists()
    mode_name = "mlp_lite"
    base = _make_mission_mode(prefer_adjacent_food=weights_available)

    def pick(env: RogueSimEnv, obs, rng: np.random.Generator) -> int:
        return base.pick_action(env, obs, rng)

    return ModeRunner(name=mode_name, pick_action=pick)


def _make_lewm_no_planner_mode(model, device, *, discount, done_penalty) -> ModeRunner:
    action_dim = model.action_dim

    def pick(env: RogueSimEnv, obs, rng: np.random.Generator) -> int:
        pixels = torch.from_numpy(obs.pixels).float().to(device)
        # One candidate per discrete action, horizon = 1.
        eye = torch.eye(action_dim, device=device).float()
        action_seq = eye.view(1, action_dim, 1, action_dim)  # (B=1, S=A, H=1, A)
        scores = model.score_action_sequences(
            pixels.unsqueeze(0),
            action_seq,
            discount=discount,
            done_penalty=done_penalty,
        )  # (1, A)
        return int(torch.argmax(scores[0]).item())

    return ModeRunner(name="lewm_no_planner", pick_action=pick, requires_jepa=True)


def _make_lewm_dreamer_mode(
    model,
    device,
    *,
    horizon: int,
    num_candidates: int,
    top_k: int,
    discount: float,
    done_penalty: float,
) -> ModeRunner:
    action_dim = model.action_dim
    # ARPredictor does not store num_frames; recover it from the
    # positional embedding shape (matches the M3 sidecar's max_horizon).
    max_h = int(model.predictor.pos_embedding.size(1))
    horizon = max(1, min(horizon, max_h))

    def pick(env: RogueSimEnv, obs, rng: np.random.Generator) -> int:
        pixels = torch.from_numpy(obs.pixels).float().to(device)
        seed = int(rng.integers(0, 2**31 - 1))
        plan = random_shooting(
            model,
            pixels,
            horizon=horizon,
            num_candidates=num_candidates,
            action_dim=action_dim,
            top_k=top_k,
            discount=discount,
            done_penalty=done_penalty,
            seed=seed,
            device=device,
        )
        return int(plan.best_actions[0])

    return ModeRunner(name="lewm_dreamer", pick_action=pick, requires_jepa=True)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    seed: int
    mode: str
    episode_idx: int
    levels_cleared: int
    food_left: int
    steps: int
    episode_return: float
    dynamics_loss: float  # nan when not applicable
    reward_loss: float  # nan when not applicable
    died: bool


def _run_episode(
    env: RogueSimEnv,
    mode: ModeRunner,
    model,
    device,
    rng: np.random.Generator,
    *,
    seed: int,
    episode_idx: int,
    measure_dynamics: bool,
) -> EpisodeResult:
    obs = env.reset(seed=seed * 7919 + episode_idx)
    levels_cleared = 0
    episode_return = 0.0
    dyn_losses: list[float] = []
    reward_losses: list[float] = []
    died = False

    while not env.is_done:
        action = mode(env, obs, rng)

        prev_pixels = obs.pixels
        prev_board = obs.board
        next_obs, reward, done, info = env.step(action)
        episode_return += float(reward)
        if info.get("level_cleared", False):
            levels_cleared += 1
        if info.get("died", False):
            died = True

        if measure_dynamics and model is not None:
            # Skip the step where the board was regenerated -- the
            # JEPA cannot predict an unseen board.
            if not info.get("level_cleared", False):
                with torch.no_grad():
                    p0 = (
                        torch.from_numpy(prev_pixels)
                        .float()
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device)
                    )
                    p1 = (
                        torch.from_numpy(next_obs.pixels)
                        .float()
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device)
                    )
                    one_hot = torch.zeros(
                        1, 1, model.action_dim, device=device, dtype=torch.float32
                    )
                    one_hot[0, 0, action] = 1.0

                    emb_ctx = model.encode_obs(p0)
                    emb_next = model.encode_obs(p1)
                    act_emb = model.encode_actions(one_hot)
                    pred_emb = model.predict_next(emb_ctx, act_emb)
                    dyn_mse = float(
                        torch.mean((pred_emb[:, -1:] - emb_next[:, -1:]) ** 2).item()
                    )
                    reward_pred = model.reward_head(pred_emb, act_emb)
                    rew_mse = float(
                        torch.mean((reward_pred - torch.tensor(reward, device=device)) ** 2).item()
                    )
                    dyn_losses.append(dyn_mse)
                    reward_losses.append(rew_mse)

        obs = next_obs

    dyn_loss = float(np.mean(dyn_losses)) if dyn_losses else float("nan")
    rew_loss = float(np.mean(reward_losses)) if reward_losses else float("nan")

    return EpisodeResult(
        seed=seed,
        mode=mode.name,
        episode_idx=episode_idx,
        levels_cleared=levels_cleared,
        food_left=int(obs.food),
        steps=int(obs.step),
        episode_return=float(episode_return),
        dynamics_loss=dyn_loss,
        reward_loss=rew_loss,
        died=died,
    )


# ---------------------------------------------------------------------------
# Top-level config + driver
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    modes: list[str]
    episodes: int = 5
    max_steps: int = 200
    seed: int = 0
    horizon: int = 3
    num_candidates: int = 64
    top_k: int = 3
    discount: float = 0.95
    done_penalty: float = 1.0
    board_size: int = 8
    image_size: int = 32
    food_start: int = 100
    food_per_level: int = 50
    checkpoint: Path | None = None
    mlp_weights: Path | None = None
    device: str = "cpu"
    output_csv: Path = field(default_factory=lambda: Path("results/lewm/benchmark.csv"))


def _load_jepa(checkpoint: Path, device: str) -> tuple[object, int]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = load_config_from_payload(payload)
    model = build_model(cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, int(cfg.sequence_length)


def _build_modes(cfg: BenchmarkConfig, model, device) -> list[ModeRunner]:
    runners: list[ModeRunner] = []
    for name in cfg.modes:
        if name == "random":
            runners.append(_make_random_mode())
        elif name == "mission":
            runners.append(_make_mission_mode())
        elif name == "mlp_lite":
            runners.append(_make_mlp_lite_mode(cfg.mlp_weights))
        elif name == "lewm_no_planner":
            if model is None:
                print(f"warn: skipping mode '{name}' (no JEPA checkpoint provided)")
                continue
            runners.append(
                _make_lewm_no_planner_mode(
                    model,
                    device,
                    discount=cfg.discount,
                    done_penalty=cfg.done_penalty,
                )
            )
        elif name == "lewm_dreamer":
            if model is None:
                print(f"warn: skipping mode '{name}' (no JEPA checkpoint provided)")
                continue
            runners.append(
                _make_lewm_dreamer_mode(
                    model,
                    device,
                    horizon=cfg.horizon,
                    num_candidates=cfg.num_candidates,
                    top_k=cfg.top_k,
                    discount=cfg.discount,
                    done_penalty=cfg.done_penalty,
                )
            )
        else:
            raise ValueError(f"unknown mode '{name}'")
    return runners


def run_benchmark(cfg: BenchmarkConfig) -> list[EpisodeResult]:
    """Run the benchmark matrix and return per-episode results.

    Side effects: writes ``cfg.output_csv`` (per-episode) and a
    ``cfg.output_csv.with_suffix('.summary.csv')`` companion file.
    """
    device = torch.device(cfg.device)
    needs_jepa = any(m in _JEPA_MODES for m in cfg.modes)

    model = None
    if needs_jepa:
        if cfg.checkpoint is None:
            raise ValueError(
                "JEPA mode requested but no --checkpoint provided. Train a "
                "checkpoint with 'python -m tools.lewm.train' first."
            )
        model, _ = _load_jepa(cfg.checkpoint, cfg.device)

    runners = _build_modes(cfg, model, device)
    if not runners:
        raise ValueError("no benchmark modes resolved (all skipped)")

    rng_master = np.random.default_rng(cfg.seed)
    results: list[EpisodeResult] = []

    for runner in runners:
        env = RogueSimEnv(
            board_size=cfg.board_size,
            image_size=cfg.image_size,
            food_start=cfg.food_start,
            food_per_level=cfg.food_per_level,
            max_steps=cfg.max_steps,
            seed=cfg.seed,
        )
        # Each mode gets its own RNG stream so the per-mode comparison
        # at a fixed --seed is reproducible (independent of which other
        # modes were enabled in the same run).
        mode_rng = np.random.default_rng(
            int(rng_master.integers(0, 2**31 - 1))
        )
        t0 = time.time()
        for ep in range(cfg.episodes):
            result = _run_episode(
                env,
                runner,
                model,
                device,
                mode_rng,
                seed=cfg.seed,
                episode_idx=ep,
                measure_dynamics=(runner.name in _JEPA_MODES) or (model is not None),
            )
            results.append(result)
        dt = time.time() - t0
        print(
            f"mode={runner.name:<16s} episodes={cfg.episodes:>3d} "
            f"elapsed={dt:.2f}s"
        )

    _write_per_episode_csv(cfg.output_csv, results)
    _write_summary_csv(cfg.output_csv.with_suffix(".summary.csv"), results)
    return results


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


_PER_EPISODE_FIELDS = (
    "seed",
    "mode",
    "episode_idx",
    "levels_cleared",
    "food_left",
    "steps",
    "episode_return",
    "dynamics_loss",
    "reward_loss",
    "died",
)


def _write_per_episode_csv(path: Path, results: list[EpisodeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_PER_EPISODE_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "seed": r.seed,
                    "mode": r.mode,
                    "episode_idx": r.episode_idx,
                    "levels_cleared": r.levels_cleared,
                    "food_left": r.food_left,
                    "steps": r.steps,
                    "episode_return": f"{r.episode_return:.6f}",
                    "dynamics_loss": _fmt_float(r.dynamics_loss),
                    "reward_loss": _fmt_float(r.reward_loss),
                    "died": int(r.died),
                }
            )
    print(f"saved per-episode csv: {path}")


_SUMMARY_FIELDS = (
    "seed",
    "mode",
    "episodes",
    "mean_levels_cleared",
    "mean_food_left",
    "mean_steps",
    "mean_episode_return",
    "mean_dynamics_loss",
    "mean_reward_loss",
    "death_rate",
    "std_episode_return",
)


def _write_summary_csv(path: Path, results: list[EpisodeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[int, str], list[EpisodeResult]] = {}
    for r in results:
        grouped.setdefault((r.seed, r.mode), []).append(r)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        for (seed, mode), rs in grouped.items():
            writer.writerow(
                {
                    "seed": seed,
                    "mode": mode,
                    "episodes": len(rs),
                    "mean_levels_cleared": f"{np.mean([r.levels_cleared for r in rs]):.4f}",
                    "mean_food_left": f"{np.mean([r.food_left for r in rs]):.4f}",
                    "mean_steps": f"{np.mean([r.steps for r in rs]):.4f}",
                    "mean_episode_return": f"{np.mean([r.episode_return for r in rs]):.6f}",
                    "mean_dynamics_loss": _fmt_float(_nanmean([r.dynamics_loss for r in rs])),
                    "mean_reward_loss": _fmt_float(_nanmean([r.reward_loss for r in rs])),
                    "death_rate": f"{np.mean([float(r.died) for r in rs]):.4f}",
                    "std_episode_return": f"{np.std([r.episode_return for r in rs]):.6f}",
                }
            )
    print(f"saved summary csv:     {path}")


def _fmt_float(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    return f"{v:.6f}"


def _nanmean(values: list[float]) -> float:
    finite = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LeWM port: M6 benchmark harness")
    p.add_argument("--smoke", action="store_true", help="tiny preset for CI")
    p.add_argument(
        "--modes",
        nargs="+",
        default=list(_ALL_MODES),
        choices=list(_ALL_MODES),
    )
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--num-candidates", type=int, default=64)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--discount", type=float, default=0.95)
    p.add_argument("--done-penalty", type=float, default=1.0)
    p.add_argument("--board-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--food-start", type=int, default=100)
    p.add_argument("--food-per-level", type=int, default=50)
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a JEPA checkpoint (required for lewm_* modes).",
    )
    p.add_argument(
        "--mlp-weights",
        default="Assets/StreamingAssets/world_model_weights.json",
        help=(
            "Path to the in-engine MLP weights JSON. Used only as a "
            "presence marker for the mlp_lite surrogate."
        ),
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--output-csv",
        default="results/lewm/benchmark.csv",
        help="Per-episode CSV path. A '.summary.csv' companion is also written.",
    )
    return p.parse_args(argv)


def _resolve_defaults(args: argparse.Namespace) -> None:
    if args.smoke:
        args.episodes = min(args.episodes, 1)
        args.max_steps = min(args.max_steps, 30)
        args.num_candidates = min(args.num_candidates, 16)
        args.top_k = min(args.top_k, 3)
        args.horizon = min(args.horizon, 2)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _resolve_defaults(args)

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint is not None and not checkpoint.exists():
        raise FileNotFoundError(f"--checkpoint not found: {checkpoint}")

    mlp_weights = Path(args.mlp_weights) if args.mlp_weights else None

    cfg = BenchmarkConfig(
        modes=list(args.modes),
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        horizon=args.horizon,
        num_candidates=args.num_candidates,
        top_k=args.top_k,
        discount=args.discount,
        done_penalty=args.done_penalty,
        board_size=args.board_size,
        image_size=args.image_size,
        food_start=args.food_start,
        food_per_level=args.food_per_level,
        checkpoint=checkpoint,
        mlp_weights=mlp_weights,
        device=args.device,
        output_csv=Path(args.output_csv),
    )
    run_benchmark(cfg)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
