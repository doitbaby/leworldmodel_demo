"""LeWorldModel inference sidecar.

A small FastAPI app that wraps a trained JEPA checkpoint and exposes
inference over HTTP. Unity (or any other client) sends a board-state
cell-code array plus candidate action sequences and gets back a value
estimate per sequence — high values mean "the model thinks this plan
leads to good outcomes from the current board".

The sidecar is single-process / single-threaded by design. Multi-worker
uvicorn would duplicate the model in memory and is not needed for the
expected demo load (one Unity client at ~10 Hz).

Endpoints
---------
- ``GET /healthz`` — liveness probe.
- ``GET /info`` — model metadata (embed_dim, action_dim, image_size,
  sequence_length, service).
- ``POST /score_actions`` — score candidate action sequences.

Request payload for ``/score_actions``::

    {
        "board_width": 8,
        "board_height": 8,
        "board_state": [int, ...],            # length = w * h
        "horizon": 4,
        "num_sequences": 5,
        "action_sequences_flat": [int, ...],  # length = num_sequences * horizon
        "discount": 0.95,                     # optional
        "done_penalty": 1.0                   # optional
    }

The flat action layout is a JsonUtility-compatibility quirk: Unity's
``JsonUtility`` does not handle nested ``int[][]`` so the client
flattens to a single row-major array of length ``num_sequences *
horizon`` and we reshape on this side.

Response::

    {
        "scores": [float, ...],   # length = num_sequences, higher = better
        "horizon": 4,
        "service": "lewm.sidecar.v1"
    }
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import render_board_to_pixels


SERVICE_VERSION = "lewm.sidecar.v1"

_LOG = logging.getLogger("lewm.sidecar")


class ScoreRequest(BaseModel):
    board_width: int = Field(..., gt=0)
    board_height: int = Field(..., gt=0)
    board_state: list[int]
    horizon: int = Field(..., ge=1)
    num_sequences: int = Field(..., ge=1)
    action_sequences_flat: list[int]
    discount: float = 0.95
    done_penalty: float = 1.0


class ScoreResponse(BaseModel):
    scores: list[float]
    horizon: int
    service: str = SERVICE_VERSION


class InfoResponse(BaseModel):
    service: str = SERVICE_VERSION
    embed_dim: int
    action_dim: int
    image_size: int
    sequence_length: int
    # Maximum supported rollout horizon. The AR predictor uses a learned
    # positional embedding sized at ``sequence_length - 1`` (it was only
    # trained over that many action steps), so longer plans would overflow
    # the position table. Clients should clamp ``horizon`` to this value.
    max_horizon: int


class SidecarState:
    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self.config: Any = None
        self.device: torch.device = torch.device("cpu")

    @property
    def ready(self) -> bool:
        return self.model is not None and self.config is not None


_state = SidecarState()
app = FastAPI(title="LeWorldModel sidecar", version=SERVICE_VERSION)


def load_checkpoint(path: str, device: str = "cpu") -> None:
    """Load a checkpoint produced by ``tools.lewm.train.save_checkpoint``."""
    # Lazy import so ``import tools.lewm.sidecar`` is cheap and free of
    # circular-import risk during testing.
    from .train import build_model, load_config_from_payload

    payload = torch.load(path, map_location=device, weights_only=False)
    schema = payload.get("schema")
    if schema != "lewm.port.checkpoint.v1":
        raise ValueError(
            f"unsupported checkpoint schema {schema!r}, "
            "expected 'lewm.port.checkpoint.v1'"
        )

    # ``load_config_from_payload`` filters unknown fields, so M1/M3
    # checkpoints keep loading after M4 adds new TrainConfig fields.
    cfg = load_config_from_payload(payload)
    model = build_model(cfg)
    model.load_state_dict(payload["model_state"])
    model.eval()
    _state.model = model.to(device)
    _state.config = cfg
    _state.device = torch.device(device)
    _LOG.info(
        "loaded checkpoint from %s (embed_dim=%d action_dim=%d image_size=%d)",
        path,
        cfg.embed_dim,
        cfg.action_dim,
        cfg.image_size,
    )


def get_state() -> SidecarState:
    return _state


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok" if _state.ready else "no_model", "service": SERVICE_VERSION}


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    if not _state.ready:
        raise HTTPException(status_code=503, detail="no model loaded")
    cfg = _state.config
    return InfoResponse(
        embed_dim=cfg.embed_dim,
        action_dim=cfg.action_dim,
        image_size=cfg.image_size,
        sequence_length=cfg.sequence_length,
        max_horizon=max(1, cfg.sequence_length - 1),
    )


@app.post("/score_actions", response_model=ScoreResponse)
def score_actions(req: ScoreRequest) -> ScoreResponse:
    if not _state.ready:
        raise HTTPException(status_code=503, detail="no model loaded")

    cfg = _state.config
    max_horizon = max(1, cfg.sequence_length - 1)
    if req.horizon > max_horizon:
        raise HTTPException(
            status_code=400,
            detail=(
                f"horizon {req.horizon} exceeds max_horizon={max_horizon} "
                f"(sequence_length={cfg.sequence_length}); train with a longer "
                "sequence_length or shorten the plan"
            ),
        )

    expected_board = req.board_width * req.board_height
    if len(req.board_state) != expected_board:
        raise HTTPException(
            status_code=400,
            detail=(
                f"board_state has {len(req.board_state)} cells, "
                f"expected board_width({req.board_width}) * "
                f"board_height({req.board_height}) = {expected_board}"
            ),
        )

    expected_actions = req.num_sequences * req.horizon
    if len(req.action_sequences_flat) != expected_actions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"action_sequences_flat has {len(req.action_sequences_flat)} "
                f"entries, expected num_sequences({req.num_sequences}) * "
                f"horizon({req.horizon}) = {expected_actions}"
            ),
        )

    board = np.asarray(req.board_state, dtype=np.int32).reshape(
        req.board_height, req.board_width
    )
    pixels = render_board_to_pixels(board, cfg.image_size)
    obs = torch.from_numpy(pixels).float().unsqueeze(0).to(_state.device)

    flat = np.asarray(req.action_sequences_flat, dtype=np.int64).reshape(
        req.num_sequences, req.horizon
    )
    if (flat < 0).any() or (flat >= cfg.action_dim).any():
        raise HTTPException(
            status_code=400,
            detail=f"action indices must be in [0, {cfg.action_dim})",
        )

    action_idx = torch.from_numpy(flat).long().to(_state.device)
    action_seq = F.one_hot(action_idx, num_classes=cfg.action_dim).float()
    # action_seq: (S, H, A) -> add batch dim expected by JEPA.rollout
    action_seq = action_seq.unsqueeze(0)

    with torch.no_grad():
        scores = _state.model.score_action_sequences(
            obs,
            action_seq,
            discount=req.discount,
            done_penalty=req.done_penalty,
        )

    # scores: (1, num_sequences)
    return ScoreResponse(
        scores=scores[0].cpu().tolist(),
        horizon=req.horizon,
    )
