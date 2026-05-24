"""Learning-rate schedule helpers for the LeWM port.

Matches the upstream training recipe (``config/optim/cosine_warmup.yaml``):
linear warmup from 0 to ``base_lr`` over ``warmup_steps``, then cosine
decay to ``min_lr_ratio * base_lr`` over the remaining steps.

The helpers return a multiplier in ``[0, 1]`` given an integer step.
Plug into :class:`torch.optim.lr_scheduler.LambdaLR`::

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(opt, lr_lambda=cosine_warmup_lambda(
        warmup_steps=200, total_steps=1000,
    ))

Kept separate from :mod:`tools.lewm.train` so other entry points (e.g.
the M4 unit tests or a future Dreamer-style actor in M5) can re-use it
without importing the full training loop.
"""

from __future__ import annotations

import math
from collections.abc import Callable


def cosine_warmup_lambda(
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> Callable[[int], float]:
    """Build a step → multiplier function for :class:`LambdaLR`.

    Args:
        warmup_steps: Number of linear warmup steps (multiplier rises
            linearly from ``0`` to ``1`` over this many steps). Use
            ``0`` to start straight at the base lr.
        total_steps: Total optimisation steps including warmup. After
            ``total_steps`` the multiplier stays at ``min_lr_ratio``.
        min_lr_ratio: Floor multiplier reached at ``total_steps``.
            Default ``0`` matches the upstream config; bump to e.g.
            ``0.1`` to keep training alive in the last few steps.
    """
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    if warmup_steps >= total_steps:
        raise ValueError(f"warmup_steps={warmup_steps} must be < total_steps={total_steps}")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

    decay_steps = max(1, total_steps - warmup_steps)

    def f(step: int) -> float:
        if step < warmup_steps:
            # Avoid zero on step 0 — start at 1/warmup_steps so the
            # first batch sees a non-zero LR.
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(decay_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return f


def constant_lambda() -> Callable[[int], float]:
    """No-op schedule for backward-compatible defaults (smoke / vector mode)."""

    def f(_step: int) -> float:
        return 1.0

    return f
