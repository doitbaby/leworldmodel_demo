"""LeWorldModel port for the leworldmodel_demo Unity game.

This package vendors the core architecture from
https://github.com/lucas-maes/le-wm (MIT, commit c8a4417), adapted for the
rogue game's discrete actions and procedural levels.

Submodules:
- :mod:`tools.lewm.module`  -- vendored upstream layers (SIGReg, ARPredictor, ...)
- :mod:`tools.lewm.encoder` -- small CNN encoder replacing the upstream ViT
- :mod:`tools.lewm.jepa`    -- JEPA wrapper with reward / done heads
- :mod:`tools.lewm.data`    -- synthetic + JSONL datasets for training
- :mod:`tools.lewm.train`   -- PyTorch-only training loop (CLI entry point)
"""

__all__ = ["module", "encoder", "jepa", "data", "train"]
