"""LeWorldModel port for the leworldmodel_demo Unity game.

This package vendors the core architecture from
https://github.com/lucas-maes/le-wm (MIT, commit c8a4417), adapted for the
rogue game's discrete actions and procedural levels.

Submodules:
- :mod:`tools.lewm.module`    -- vendored upstream layers (SIGReg, ARPredictor, ...)
- :mod:`tools.lewm.encoder`   -- small CNN encoder replacing the upstream ViT
- :mod:`tools.lewm.jepa`      -- JEPA wrapper with reward / done heads
- :mod:`tools.lewm.data`      -- synthetic + JSONL datasets for training
- :mod:`tools.lewm.train`     -- PyTorch-only training loop (CLI entry point)
- :mod:`tools.lewm.schedule`  -- LR schedule helpers (cosine + warmup)
- :mod:`tools.lewm.planner`   -- random-shooting actor over JEPA rollouts (M5)
- :mod:`tools.lewm.sidecar`   -- FastAPI inference + planning service
- :mod:`tools.lewm.env`       -- python-side rogue env used by the M6 benchmark
- :mod:`tools.lewm.benchmark` -- multi-mode evaluation CLI emitting CSVs (M6)
"""

__all__ = [
    "module",
    "encoder",
    "jepa",
    "data",
    "train",
    "schedule",
    "planner",
    "env",
    "benchmark",
]
