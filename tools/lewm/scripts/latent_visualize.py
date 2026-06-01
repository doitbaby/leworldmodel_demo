"""Visualize the latent space learned by a LeWorldModel checkpoint.

Loads a checkpoint produced by :func:`tools.lewm.train.save_checkpoint`
and a ``rogue.transition.v3`` JSONL file, encodes every transition's
``board_state`` through the JEPA encoder, projects the resulting
embeddings to 2D via PCA (NumPy-only -- no sklearn dependency), and
writes a coloured scatter plot for each available label channel:

* ``level``       -- procedural level index (proxy for "where in the
                     game" the frame came from).
* ``food``        -- remaining food (proxy for episode progress).
* ``near_exit``   -- 1 if the player is adjacent to the exit cell.
* ``near_enemy``  -- 1 if the player is adjacent to any enemy.
* ``action``      -- index of the action taken from this state.

The script also dumps the raw embeddings to a ``.npz`` so downstream
analysis (t-SNE, UMAP, classification probes) can be done offline
without re-encoding the entire JSONL.

Why PCA and not t-SNE/UMAP? PCA is deterministic, dependency-free
(``np.linalg.svd``), and good enough to demonstrate cluster structure
for the paper case study. Anyone who wants UMAP can load the ``.npz``
and run it themselves.

Usage::

    python -m tools.lewm.scripts.latent_visualize \\
        --checkpoint results/lewm/lewm_best.pt \\
        --jsonl data/rogue_transitions_v3.jsonl \\
        --output-dir results/lewm/latent_viz/ \\
        --max-samples 4000

Exits 0 on success. The output directory contains:

* ``embeddings.npz``      -- raw embeddings + per-frame labels.
* ``pca_<label>.png``     -- 2D PCA scatter coloured by ``<label>``.
* ``summary.json``        -- counts, PCA explained-variance ratios.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from ..data import render_board_to_pixels
from ..env import RogueSimEnv
from ..train import build_model, load_config_from_payload

__all__ = ["encode_jsonl", "pca_2d", "main"]


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _load_model(checkpoint_path: Path, device: torch.device):
    """Load a JEPA checkpoint into eval mode. Returns ``(model, cfg)``."""
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
# Label extraction
# ---------------------------------------------------------------------------


def _board_from_record(rec: dict) -> np.ndarray:
    w = int(rec.get("board_width", 0))
    h = int(rec.get("board_height", 0))
    flat = rec.get("board_state", [])
    if w <= 0 or h <= 0 or len(flat) != w * h:
        raise ValueError(f"bad board on record step={rec.get('step')}: {w}x{h} vs {len(flat)}")
    return np.asarray(flat, dtype=np.int32).reshape(h, w)


def _player_xy(board: np.ndarray) -> tuple[int, int] | None:
    """Return (x, y) of cell code 5 (player), or None if missing."""
    ys, xs = np.where(board == 5)
    if len(xs) == 0:
        return None
    return int(xs[0]), int(ys[0])


def _adjacent_codes(board: np.ndarray, px: int, py: int) -> set[int]:
    """Codes of the four orthogonally adjacent cells to ``(px, py)``."""
    h, w = board.shape
    codes: set[int] = set()
    for dx, dy in RogueSimEnv.DIRS:
        nx, ny = px + dx, py + dy
        if 0 <= nx < w and 0 <= ny < h:
            codes.add(int(board[ny, nx]))
    return codes


def _record_labels(rec: dict) -> dict[str, int | float]:
    """Extract scalar labels for plot colouring from a v3 record."""
    board = _board_from_record(rec)
    labels: dict[str, int | float] = {
        "level": int(rec.get("level", 0)),
        "food": float(rec.get("food", 0)),
        "action": int(rec.get("action", -1)),
    }
    player = _player_xy(board)
    if player is None:
        labels["near_exit"] = 0
        labels["near_enemy"] = 0
    else:
        adj = _adjacent_codes(board, *player)
        labels["near_exit"] = int(1 in adj)
        labels["near_enemy"] = int(2 in adj)
    return labels


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _iter_records(jsonl_path: Path) -> Iterable[dict]:
    with jsonl_path.open() as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if "board_state" in rec and "board_width" in rec and "board_height" in rec:
                yield rec


def encode_jsonl(
    model,
    cfg,
    jsonl_path: Path,
    *,
    device: torch.device,
    max_samples: int = 4000,
    batch_size: int = 64,
    image_size: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Encode every ``board_state`` in ``jsonl_path`` through ``model.encode_obs``.

    Returns ``(embeddings, labels)`` where ``embeddings`` is shape
    ``(N, embed_dim)`` and ``labels`` maps label-name -> ``(N,)`` array.
    Caps at ``max_samples`` records to keep memory bounded for big files.
    """
    img_size = image_size or cfg.image_size
    records: list[dict] = []
    for rec in _iter_records(jsonl_path):
        records.append(rec)
        if len(records) >= max_samples:
            break

    if not records:
        raise ValueError(f"no usable v3 records in {jsonl_path}")

    label_keys = ("level", "food", "near_exit", "near_enemy", "action")
    labels: dict[str, list[float]] = {k: [] for k in label_keys}
    embeddings: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            pixels = np.stack(
                [render_board_to_pixels(_board_from_record(r), image_size=img_size) for r in batch],
                axis=0,
            )
            # encode_obs expects (B, T, ...); we use T=1 so we get one
            # embedding per frame.
            obs = torch.from_numpy(pixels).float().unsqueeze(1).to(device)
            emb = model.encode_obs(obs)  # (B, 1, D)
            embeddings.append(emb.squeeze(1).cpu().numpy())
            for r in batch:
                lab = _record_labels(r)
                for k in label_keys:
                    labels[k].append(lab[k])

    emb_np = np.concatenate(embeddings, axis=0)
    label_np = {k: np.asarray(v) for k, v in labels.items()}
    return emb_np, label_np


# ---------------------------------------------------------------------------
# PCA + plotting
# ---------------------------------------------------------------------------


def pca_2d(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project ``(N, D)`` embeddings to ``(N, 2)`` via PCA.

    Returns ``(projection, explained_variance_ratio[:2])``. Uses
    ``numpy.linalg.svd`` so no sklearn dependency. Center the data first
    so the projection passes through the embedding centroid.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"expected (N, D), got {embeddings.shape}")
    n, d = embeddings.shape
    if n < 2 or d < 2:
        raise ValueError(f"need >=2 samples and >=2 dims, got {embeddings.shape}")
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    # SVD on centered data: rows are samples. U has the projections, S^2/(N-1)
    # gives the explained variance.
    u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    projection = u[:, :2] * s[:2]
    total = float(np.sum(s**2))
    if total <= 0:
        ratio = np.zeros(2, dtype=np.float64)
    else:
        ratio = (s[:2] ** 2) / total
    return projection.astype(np.float32), ratio.astype(np.float64)


def _plot_scatter(
    proj: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    output_path: Path,
    cmap: str = "viridis",
) -> None:
    """Save a 2D scatter of ``proj`` coloured by ``values`` to ``output_path``."""
    # Import here so a missing matplotlib only breaks plotting, not import.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=120)
    sc = ax.scatter(
        proj[:, 0],
        proj[:, 1],
        c=values,
        cmap=cmap,
        s=8,
        alpha=0.7,
        linewidths=0,
    )
    fig.colorbar(sc, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize a LeWM latent space via PCA.")
    p.add_argument("--checkpoint", required=True, help="Path to *.pt checkpoint.")
    p.add_argument("--jsonl", required=True, help="rogue.transition.v3 JSONL to encode.")
    p.add_argument("--output-dir", required=True, help="Directory to write npz + plots.")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--max-samples",
        type=int,
        default=4000,
        help="Cap on number of frames to encode (default 4000).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used while encoding (default 64).",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating .png plots (useful for CI smoke tests).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = _load_model(Path(args.checkpoint), device)
    embeddings, labels = encode_jsonl(
        model,
        cfg,
        Path(args.jsonl),
        device=device,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )

    npz_path = output_dir / "embeddings.npz"
    np.savez_compressed(npz_path, embeddings=embeddings, **labels)

    projection, evr = pca_2d(embeddings)

    summary = {
        "checkpoint": str(args.checkpoint),
        "jsonl": str(args.jsonl),
        "num_samples": int(embeddings.shape[0]),
        "embed_dim": int(embeddings.shape[1]),
        "pca_explained_variance_ratio": [float(x) for x in evr.tolist()],
        "label_counts": {
            k: {
                str(int(v)): int(c)
                for v, c in zip(*np.unique(arr, return_counts=True), strict=False)
            }
            for k, arr in labels.items()
            if k in ("near_exit", "near_enemy", "action")
        },
    }

    plot_files: list[str] = []
    if not args.no_plots:
        for key, cmap in (
            ("level", "viridis"),
            ("food", "plasma"),
            ("near_exit", "coolwarm"),
            ("near_enemy", "coolwarm"),
            ("action", "tab10"),
        ):
            out = output_dir / f"pca_{key}.png"
            _plot_scatter(
                projection,
                labels[key].astype(np.float32),
                title=f"LeWM latent PCA - {key}",
                output_path=out,
                cmap=cmap,
            )
            plot_files.append(str(out))
        # Joint projection scatter (uncoloured) for the npz reference.
        proj_npz = output_dir / "projection.npz"
        np.savez_compressed(proj_npz, projection=projection)
        summary["projection_npz"] = str(proj_npz)

    summary["plots"] = plot_files
    summary["embeddings_npz"] = str(npz_path)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
