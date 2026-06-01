"""Synthetic-anomaly surprise-score evaluator for a LeWorldModel checkpoint.

This is the rogue case-study analogue of "latent surprise detects
physically implausible events" from the LeWorldModel paper
(arXiv 2603.19312, Maes et al. 2026). For each window of normal
gameplay we compute the **latent prediction error** between the
predicted next embedding and the encoded next observation::

    surprise(t) = || predict_next(encode(o_{<t}), encode(a_{<t})) - encode(o_t) ||^2_2

We then construct an **anomalous variant** of every window by
surgically corrupting the next ``board_state`` and recomputing
``surprise``. Four bug types are supported (chosen to mirror common
in-engine glitches the Unity demo can demonstrate):

* ``teleport``     -- relocate the player to a far valid cell.
* ``spawn_enemy``  -- place an enemy on the player's current cell
                       (sudden hostile spawn on the player tile).
* ``wall_pass``    -- move the player diagonally through a wall, a
                       move the rogue dynamics never produces.
* ``food_vanish``  -- delete every food cell on the board.

For each variant we report:

* mean / std normal vs anomaly surprise,
* AUROC of the surprise score as a binary classifier
  (normal vs anomalous), computed in NumPy via the Mann-Whitney /
  rank-sum formula -- no sklearn dependency,
* per-anomaly AUROC.

Outputs ``{output_dir}/anomaly_summary.json`` and (optionally) a
``surprise_histogram.png`` showing per-class score distributions.

Why this matters for the FISAT paper: surprise-based anomaly
detection is the headline application of pixel-JEPA world models
(Wilkins & Stathis 2022, World of Bugs; Maes et al. 2026,
LeWorldModel). The rogue case study lets us measure it on a tiny
controlled environment first, before scaling to World of Bugs.

Usage::

    python -m tools.lewm.scripts.synthetic_anomaly \\
        --checkpoint results/lewm/lewm_best.pt \\
        --jsonl data/rogue_transitions_v3.jsonl \\
        --output-dir results/lewm/anomaly/ \\
        --max-windows 1000

Exit 0 on success.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data import BoardJsonlDataset, render_board_to_pixels
from ..train import build_model, load_config_from_payload

__all__ = [
    "ANOMALY_TYPES",
    "AnomalyResult",
    "auroc",
    "inject_anomaly",
    "compute_window_surprise",
    "run_anomaly_eval",
    "main",
]


ANOMALY_TYPES: tuple[str, ...] = ("teleport", "spawn_enemy", "wall_pass", "food_vanish")


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _load_model(checkpoint_path: Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    schema = payload.get("schema")
    if schema != "lewm.port.checkpoint.v1":
        raise ValueError(
            f"unsupported checkpoint schema {schema!r}, expected 'lewm.port.checkpoint.v1'"
        )
    cfg = load_config_from_payload(payload)
    model = build_model(cfg)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model.to(device), cfg


# ---------------------------------------------------------------------------
# AUROC (no sklearn)
# ---------------------------------------------------------------------------


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Binary AUROC via the Mann-Whitney U statistic.

    ``scores`` and ``labels`` are 1-D arrays of equal length. ``labels``
    must contain only 0/1 values. Returns 0.5 when either class is
    empty (uninformative).
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if scores.shape != labels.shape:
        raise ValueError(f"shape mismatch: {scores.shape} vs {labels.shape}")
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks across ties.
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    sum_ranks_pos = float(ranks[labels == 1].sum())
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# Anomaly injection
# ---------------------------------------------------------------------------


def _to_board(flat: list[int], width: int, height: int) -> np.ndarray:
    return np.asarray(flat, dtype=np.int32).reshape(height, width)


def _free_cells(board: np.ndarray) -> list[tuple[int, int]]:
    """Cells with code 0 (empty) -- safe to overwrite without nuking the player."""
    ys, xs = np.where(board == 0)
    return [(int(x), int(y)) for x, y in zip(xs, ys, strict=False)]


def _find_code(board: np.ndarray, code: int) -> list[tuple[int, int]]:
    ys, xs = np.where(board == code)
    return [(int(x), int(y)) for x, y in zip(xs, ys, strict=False)]


def _walls(board: np.ndarray) -> list[tuple[int, int]]:
    return _find_code(board, -1)


def inject_anomaly(
    board: np.ndarray,
    kind: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a corrupted copy of ``board`` matching ``kind``.

    Raises ``ValueError`` for unknown ``kind`` or boards lacking the
    structural feature the anomaly needs (e.g. no player on board).
    """
    if kind not in ANOMALY_TYPES:
        raise ValueError(f"unknown anomaly kind {kind!r}; expected one of {ANOMALY_TYPES}")
    out = board.copy()

    players = _find_code(out, 5)
    if not players:
        # Some terminal frames lose the player code; skip these cleanly.
        raise ValueError("no player on board, cannot inject anomaly")
    px, py = players[0]

    if kind == "teleport":
        free = _free_cells(out)
        if not free:
            raise ValueError("no free cells available for teleport anomaly")
        nx, ny = free[int(rng.integers(0, len(free)))]
        out[py, px] = 0
        out[ny, nx] = 5
        return out

    if kind == "spawn_enemy":
        # Overwrite the player cell with an enemy. The agent should be
        # *very* surprised that an enemy materialised on its tile.
        out[py, px] = 2
        return out

    if kind == "wall_pass":
        # Move the player two cells through a wall (Unity dynamics
        # never produce this; the player cell ends up on a wall tile).
        walls = _walls(out)
        if not walls:
            raise ValueError("no walls on board, cannot inject wall_pass anomaly")
        wx, wy = walls[int(rng.integers(0, len(walls)))]
        out[py, px] = 0
        out[wy, wx] = 5  # player stuck inside the wall tile
        return out

    if kind == "food_vanish":
        out[out == 4] = 0
        return out

    raise AssertionError(f"unreachable: kind={kind}")


# ---------------------------------------------------------------------------
# Surprise score
# ---------------------------------------------------------------------------


@dataclass
class AnomalyResult:
    kind: str
    normal_scores: np.ndarray
    anomaly_scores: np.ndarray
    auroc: float

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "kind": self.kind,
            "auroc": float(self.auroc),
            "normal_mean": float(self.normal_scores.mean()) if self.normal_scores.size else 0.0,
            "normal_std": float(self.normal_scores.std()) if self.normal_scores.size else 0.0,
            "anomaly_mean": (
                float(self.anomaly_scores.mean()) if self.anomaly_scores.size else 0.0
            ),
            "anomaly_std": float(self.anomaly_scores.std()) if self.anomaly_scores.size else 0.0,
            "n_normal": int(self.normal_scores.size),
            "n_anomaly": int(self.anomaly_scores.size),
        }


@torch.no_grad()
def compute_window_surprise(
    model,
    pixels: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Latent prediction error per window.

    ``pixels`` is ``(B, T, 3, H, W)``, ``actions`` is ``(B, T-1, A)``.
    Returns the mean squared error between predicted and encoded next
    embeddings as a ``(B,)`` tensor (averaged over the T-1 prediction
    steps and the embedding dim).
    """
    emb = model.encode_obs(pixels)  # (B, T, D)
    act_emb = model.encode_actions(actions)  # (B, T-1, D)
    pred = model.predict_next(emb[:, :-1], act_emb)  # (B, T-1, D)
    diff = (pred - emb[:, 1:]) ** 2  # (B, T-1, D)
    return diff.mean(dim=(1, 2))


def _render_window(
    boards: list[np.ndarray],
    image_size: int,
) -> torch.Tensor:
    """Render a list of ``H x W`` boards into a ``(T, 3, image_size, image_size)`` tensor."""
    return torch.from_numpy(
        np.stack(
            [render_board_to_pixels(b, image_size=image_size) for b in boards],
            axis=0,
        )
    ).float()


def run_anomaly_eval(
    model,
    cfg,
    jsonl_path: Path,
    *,
    device: torch.device,
    max_windows: int = 1000,
    sequence_length: int = 4,
    image_size: int | None = None,
    action_dim: int | None = None,
    batch_size: int = 64,
    seed: int = 0,
) -> list[AnomalyResult]:
    """Compute surprise on normal windows vs the four anomaly variants."""
    img_size = image_size or cfg.image_size
    act_dim = action_dim or cfg.action_dim
    seq_len = sequence_length

    dataset = BoardJsonlDataset(
        path=jsonl_path,
        sequence_length=seq_len,
        image_size=img_size,
        action_dim=act_dim,
    )
    if len(dataset) == 0:
        raise ValueError(f"no usable v3 windows in {jsonl_path}")

    rng = np.random.default_rng(seed)
    # We need the raw board grids for anomaly surgery, so re-walk the
    # underlying episode list directly (BoardJsonlDataset already
    # parsed it). Each window is a list of ``seq_len`` records.
    windows: list[list[dict]] = []
    for ep_idx, start in dataset._windows:
        steps = dataset._episodes[ep_idx]
        window = steps[start : start + seq_len]
        if len(window) == seq_len:
            windows.append(window)
        if len(windows) >= max_windows:
            break

    if not windows:
        raise ValueError("no usable windows extracted from dataset")

    # Pre-extract board grids and actions so we can render them many times.
    normal_boards_per_window: list[list[np.ndarray]] = []
    actions_per_window: list[np.ndarray] = []
    for window in windows:
        boards = [
            _to_board(window[0]["board_state"], window[0]["board_width"], window[0]["board_height"])
        ]
        for rec in window[:-1]:
            boards.append(
                _to_board(rec["next_board_state"], rec["board_width"], rec["board_height"])
            )
        one_hots = np.zeros((seq_len - 1, act_dim), dtype=np.float32)
        for i, rec in enumerate(window[:-1]):
            a = int(rec.get("action", 0))
            if 0 <= a < act_dim:
                one_hots[i, a] = 1.0
        normal_boards_per_window.append(boards)
        actions_per_window.append(one_hots)

    # ---- normal surprise (one pass) ----
    normal_scores: list[float] = []
    for start in range(0, len(normal_boards_per_window), batch_size):
        chunk_boards = normal_boards_per_window[start : start + batch_size]
        chunk_actions = actions_per_window[start : start + batch_size]
        pixels = torch.stack([_render_window(b, img_size) for b in chunk_boards], dim=0).to(device)
        actions = torch.from_numpy(np.stack(chunk_actions, axis=0)).to(device)
        scores = compute_window_surprise(model, pixels, actions).cpu().numpy()
        normal_scores.extend(scores.tolist())
    normal_scores_np = np.asarray(normal_scores, dtype=np.float64)

    results: list[AnomalyResult] = []
    for kind in ANOMALY_TYPES:
        anomaly_scores: list[float] = []
        anomaly_boards_per_window: list[list[np.ndarray]] = []
        anomaly_actions_per_window: list[np.ndarray] = []
        for boards, actions in zip(normal_boards_per_window, actions_per_window, strict=False):
            corrupted_boards = list(boards)
            try:
                corrupted_boards[-1] = inject_anomaly(boards[-1], kind, rng)
            except ValueError:
                continue
            anomaly_boards_per_window.append(corrupted_boards)
            anomaly_actions_per_window.append(actions)

        if not anomaly_boards_per_window:
            results.append(
                AnomalyResult(
                    kind=kind,
                    normal_scores=normal_scores_np,
                    anomaly_scores=np.array([], dtype=np.float64),
                    auroc=0.5,
                )
            )
            continue

        for start in range(0, len(anomaly_boards_per_window), batch_size):
            chunk_boards = anomaly_boards_per_window[start : start + batch_size]
            chunk_actions = anomaly_actions_per_window[start : start + batch_size]
            pixels = torch.stack([_render_window(b, img_size) for b in chunk_boards], dim=0).to(
                device
            )
            actions = torch.from_numpy(np.stack(chunk_actions, axis=0)).to(device)
            scores = compute_window_surprise(model, pixels, actions).cpu().numpy()
            anomaly_scores.extend(scores.tolist())
        anomaly_np = np.asarray(anomaly_scores, dtype=np.float64)
        # AUROC against the corresponding-length normal pool.
        normal_pool = normal_scores_np[: anomaly_np.size]
        scores_all = np.concatenate([normal_pool, anomaly_np])
        labels = np.concatenate([np.zeros_like(normal_pool), np.ones_like(anomaly_np)]).astype(int)
        results.append(
            AnomalyResult(
                kind=kind,
                normal_scores=normal_pool,
                anomaly_scores=anomaly_np,
                auroc=auroc(scores_all, labels),
            )
        )
    return results


def _plot_hist(
    results: list[AnomalyResult],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), dpi=120)
    for ax, res in zip(axes.flatten(), results, strict=False):
        if res.normal_scores.size == 0 and res.anomaly_scores.size == 0:
            ax.set_title(f"{res.kind} (no samples)")
            continue
        ax.hist(
            res.normal_scores,
            bins=30,
            alpha=0.5,
            label=f"normal (n={res.normal_scores.size})",
            color="#3a78b8",
        )
        ax.hist(
            res.anomaly_scores,
            bins=30,
            alpha=0.5,
            label=f"anomaly (n={res.anomaly_scores.size})",
            color="#d4694a",
        )
        ax.set_title(f"{res.kind}  AUROC={res.auroc:.3f}")
        ax.set_xlabel("latent surprise")
        ax.set_ylabel("count")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inject synthetic glitches into v3 rogue boards and "
        "report latent-surprise AUROC for a LeWM checkpoint.",
    )
    p.add_argument("--checkpoint", required=True, help="Path to *.pt checkpoint.")
    p.add_argument("--jsonl", required=True, help="rogue.transition.v3 JSONL.")
    p.add_argument("--output-dir", required=True, help="Where to write summary + histogram.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--sequence-length", type=int, default=4)
    p.add_argument("--max-windows", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating the histogram PNG (useful for CI smoke).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = _load_model(Path(args.checkpoint), device)
    results = run_anomaly_eval(
        model,
        cfg,
        Path(args.jsonl),
        device=device,
        max_windows=args.max_windows,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "jsonl": str(args.jsonl),
        "sequence_length": args.sequence_length,
        "max_windows": args.max_windows,
        "anomalies": [r.to_dict() for r in results],
    }
    (output_dir / "anomaly_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if not args.no_plots:
        _plot_hist(results, output_dir / "surprise_histogram.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
