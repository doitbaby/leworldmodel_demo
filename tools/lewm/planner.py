"""Sample-based action planner driven by the JEPA world model.

Random-shooting / Dreamer-style planner: sample N candidate action
sequences from a uniform distribution over discrete actions, score each
sequence by rolling it through the world model (using
:meth:`tools.lewm.jepa.JEPA.score_action_sequences`), and return the
best plan plus a small top-k for the brain HUD.

We use random shooting (RS) rather than full CEM / MPPI because the
rogue demo's action space is tiny (4 discrete actions, horizon <= 7 in
practice) and the decision interval is ~0.45s. RS with 64-128 samples
already covers a large fraction of the search space without iterative
refits, and the resulting top-k naturally feeds the HUD's "imagined
futures" panel without any extra work on the Unity side.

The :func:`random_shooting` entry point is exposed to Unity through the
``POST /plan_actions`` endpoint added in :mod:`tools.lewm.sidecar`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .jepa import JEPA

__all__ = ["PlanResult", "random_shooting"]


@dataclass
class PlanResult:
    """Outcome of one random-shooting pass.

    Attributes:
        best_actions: Highest-scoring action sequence (length ``horizon``).
        best_score: Score of :attr:`best_actions`.
        top_k_actions: Top-k sequences ordered best-first, including
            :attr:`best_actions` at index 0.
        top_k_scores: Scores parallel to :attr:`top_k_actions`.
        num_candidates: How many sequences were sampled in this pass.
        horizon: Plan length used.
    """

    best_actions: list[int]
    best_score: float
    top_k_actions: list[list[int]] = field(default_factory=list)
    top_k_scores: list[float] = field(default_factory=list)
    num_candidates: int = 0
    horizon: int = 0


@torch.no_grad()
def random_shooting(
    model: JEPA,
    current_obs: torch.Tensor,
    *,
    horizon: int,
    num_candidates: int,
    action_dim: int,
    top_k: int = 3,
    discount: float = 0.95,
    done_penalty: float = 1.0,
    seed: int | None = None,
    device: torch.device | None = None,
    include_sequences: Sequence[Sequence[int]] | None = None,
) -> PlanResult:
    """Sample random action sequences and pick the highest-scoring one.

    Args:
        model: Trained :class:`tools.lewm.jepa.JEPA` (should already be
            in ``eval`` mode; this function will call ``torch.no_grad``).
        current_obs: Single observation tensor matching what the model's
            encoder expects (e.g. ``(3, image_size, image_size)`` for
            the in-tree :class:`TinyConvEncoder`).
        horizon: Plan length. The caller must clamp this to
            ``model.predictor.num_frames - 1`` (or whatever
            ``InfoResponse.max_horizon`` reports).
        num_candidates: How many sequences to sample.
        action_dim: Number of discrete actions. Must match
            ``model.action_dim``.
        top_k: How many top sequences to return for HUD display.
        discount, done_penalty: Forwarded to
            :meth:`tools.lewm.jepa.JEPA.score_action_sequences`.
        seed: Optional RNG seed for reproducible sampling. Useful for
            tests; in production the sidecar reseeds per request.
        device: Optional override for the tensor placement (defaults to
            ``current_obs.device``).
        include_sequences: Optional fixed sequences (each of length
            ``horizon``) to score alongside the random samples. Used by
            tests to guarantee deterministic action coverage.

    Returns:
        :class:`PlanResult`. ``best_actions`` and ``top_k_actions[0]``
        are always the same sequence.

    Raises:
        ValueError: On invalid arguments.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if num_candidates < 1:
        raise ValueError(f"num_candidates must be >= 1, got {num_candidates}")
    if action_dim < 1:
        raise ValueError(f"action_dim must be >= 1, got {action_dim}")
    if model.action_dim != action_dim:
        raise ValueError(
            f"action_dim={action_dim} does not match model.action_dim={model.action_dim}"
        )

    device = device or current_obs.device
    top_k = max(1, min(top_k, num_candidates))

    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(int(seed))

    # Sample (S, H) discrete actions on CPU then move to device once.
    action_idx = torch.randint(
        0, action_dim, (num_candidates, horizon), generator=g
    )

    if include_sequences is not None:
        extras = []
        for seq in include_sequences:
            if len(seq) != horizon:
                raise ValueError(
                    f"include_sequences entry has length {len(seq)}, expected {horizon}"
                )
            extras.append(torch.tensor(list(seq), dtype=torch.long))
        if extras:
            forced = torch.stack(extras, dim=0)
            action_idx = torch.cat([action_idx, forced], dim=0)

    # One-hot to (S, H, A) then add the batch dim expected by JEPA.rollout.
    action_seq = (
        F.one_hot(action_idx, num_classes=action_dim).float().unsqueeze(0).to(device)
    )

    obs = current_obs.unsqueeze(0).to(device)  # (1, ...)
    scores = model.score_action_sequences(
        obs,
        action_seq,
        discount=discount,
        done_penalty=done_penalty,
    )  # (1, S)
    scores_np = scores[0].detach().cpu().numpy()

    # Stable sort so ties resolve to the lowest sample index — useful
    # for the test that wants reproducible top-k order at a fixed seed.
    order = np.argsort(-scores_np, kind="stable")
    top_indices = order[:top_k].tolist()

    best_actions = action_idx[order[0]].tolist()
    return PlanResult(
        best_actions=best_actions,
        best_score=float(scores_np[order[0]]),
        top_k_actions=[action_idx[i].tolist() for i in top_indices],
        top_k_scores=[float(scores_np[i]) for i in top_indices],
        num_candidates=action_idx.size(0),
        horizon=horizon,
    )
