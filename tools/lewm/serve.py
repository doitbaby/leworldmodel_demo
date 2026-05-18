"""CLI entry point for the LeWorldModel inference sidecar.

Loads a checkpoint produced by ``tools.lewm.train.save_checkpoint`` and
boots the FastAPI app from :mod:`tools.lewm.sidecar` on the requested
host/port.

Usage::

    python -m tools.lewm.serve \\
        --checkpoint results/lewm/checkpoint.pt \\
        --host 127.0.0.1 \\
        --port 5555

The Unity side talks to this via :class:`LewmClient` in
``Assets/Scripts/ML/LewmClient.cs``. Until M5 the demo does not actually
consume the sidecar; the sidecar is an M3 wire that M5 will plug into
``BrainPlanner``.
"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from .sidecar import app, load_checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve the LeWM inference sidecar.")
    p.add_argument("--checkpoint", required=True, help="Path to *.pt checkpoint.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-level", default="info")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper())
    load_checkpoint(args.checkpoint, device=args.device)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
