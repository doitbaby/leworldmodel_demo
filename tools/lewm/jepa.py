"""LeWorldModel JEPA wrapper for the rogue demo.

Adapted from https://github.com/lucas-maes/le-wm/blob/c8a4417/jepa.py.

Differences from upstream:

1. The encoder is decoupled from the HuggingFace ViT API. We just call
   ``self.encoder(pixels)`` and expect it to return a ``(B*T, embed_dim)``
   embedding (the upstream code took the CLS token of a ViT). See
   :mod:`tools.lewm.encoder` for the in-tree replacement.

2. We add :class:`RewardHead` and :class:`DoneHead`. The rogue game has no
   fixed goal image (procedural levels), so goal-conditioned MSE planning
   does not apply. Instead we score candidate action sequences by their
   predicted reward sum and done probability.

3. :meth:`JEPA.compute_losses` returns the four LeWM-style loss components in
   one pass so the training loop in :mod:`tools.lewm.train` stays small.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .module import MLP

__all__ = ["RewardHead", "DoneHead", "JEPA", "JEPAOutput"]


class RewardHead(nn.Module):
    """Maps an embedding (and the action that produced it) to a scalar reward."""

    def __init__(self, embed_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = MLP(embed_dim + action_dim, hidden_dim, output_dim=1)

    def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """``emb`` and ``act_emb`` share leading dims; output drops the last dim."""
        x = torch.cat([emb, act_emb], dim=-1)
        return self.net(x).squeeze(-1)


class DoneHead(nn.Module):
    """Maps an embedding to a logit predicting episode termination."""

    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = MLP(embed_dim, hidden_dim, output_dim=1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb).squeeze(-1)


@dataclass
class JEPAOutput:
    """Container returned by :meth:`JEPA.compute_losses`."""

    pred_loss: torch.Tensor
    reward_loss: torch.Tensor
    done_loss: torch.Tensor
    embedding: torch.Tensor  # (B, T, D)
    predicted_embedding: torch.Tensor  # (B, T-1, D)


class JEPA(nn.Module):
    """LeWorldModel-style JEPA with reward + done auxiliary heads.

    Args:
        encoder: Module mapping ``(B*T, ...)`` pixels (or vectors) to
            ``(B*T, embed_dim)`` embeddings.
        predictor: Autoregressive predictor matching the upstream
            :class:`tools.lewm.module.ARPredictor` API.
        action_encoder: Module mapping ``(B, T, action_dim)`` actions to
            ``(B, T, embed_dim)``.
        projector: Optional projector applied to the encoder output.
        pred_proj: Optional projector applied to the predictor output.
        action_dim: Width of the raw action tensor (one-hot length for
            discrete actions, ``A`` for continuous).
        embed_dim: Embedding dim shared by encoder, predictor, projector.
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        action_encoder: nn.Module,
        *,
        action_dim: int,
        embed_dim: int,
        projector: nn.Module | None = None,
        pred_proj: nn.Module | None = None,
        reward_head: nn.Module | None = None,
        done_head: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.reward_head = reward_head or RewardHead(embed_dim, embed_dim)
        self.done_head = done_head or DoneHead(embed_dim)
        self.action_dim = action_dim
        self.embed_dim = embed_dim

    # ------------------------------------------------------------------
    # Encoding & prediction
    # ------------------------------------------------------------------

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a ``(B, T, ...)`` observation tensor into ``(B, T, D)``."""
        if obs.ndim < 3:
            raise ValueError(f"obs must have leading (B, T) dims, got {obs.shape}")
        b = obs.size(0)
        flat = rearrange(obs, "b t ... -> (b t) ...")
        emb = self.projector(self.encoder(flat))
        return rearrange(emb, "(b t) d -> b t d", b=b)

    def encode_actions(self, action: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, T, action_dim)`` actions to ``(B, T, embed_dim)``."""
        if action.size(-1) != self.action_dim:
            raise ValueError(
                f"expected action_dim={self.action_dim}, got {action.size(-1)}"
            )
        return self.action_encoder(action)

    def predict_next(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """Predict next embeddings from history. ``emb``/``act_emb`` are ``(B, T, D)``."""
        preds = self.predictor(emb, act_emb)
        b = emb.size(0)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=b)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_losses(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> JEPAOutput:
        """Compute LeWM-style losses for a batch of sequences.

        Args:
            obs: ``(B, T, ...)`` observation sequence (``T >= 2``).
            action: ``(B, T-1, action_dim)`` actions applied at each step.
            reward: ``(B, T-1)`` reward received per action.
            done: ``(B, T-1)`` 0/1 episode termination after each action.

        Returns:
            :class:`JEPAOutput` with prediction MSE, reward MSE, done BCE,
            and the embedding tensors used by downstream regularisers.
        """
        if obs.size(1) < 2:
            raise ValueError("need T >= 2 observation frames per sequence")
        if action.size(1) != obs.size(1) - 1:
            raise ValueError(
                f"action length {action.size(1)} must equal obs length-1 = {obs.size(1)-1}"
            )

        emb = self.encode_obs(obs)  # (B, T, D)
        act_emb = self.encode_actions(action)  # (B, T-1, D)

        ctx_emb = emb[:, :-1]
        target_emb = emb[:, 1:]

        pred_emb = self.predict_next(ctx_emb, act_emb)
        pred_loss = F.mse_loss(pred_emb, target_emb.detach())

        # Reward & done heads consume the predicted embedding so they remain
        # useful at planning time when ground-truth next obs are unavailable.
        reward_pred = self.reward_head(pred_emb, act_emb)
        reward_loss = F.mse_loss(reward_pred, reward.float())

        done_logits = self.done_head(pred_emb)
        done_loss = F.binary_cross_entropy_with_logits(done_logits, done.float())

        return JEPAOutput(
            pred_loss=pred_loss,
            reward_loss=reward_loss,
            done_loss=done_loss,
            embedding=emb,
            predicted_embedding=pred_emb,
        )

    # ------------------------------------------------------------------
    # Inference / planning
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        current_obs: torch.Tensor,
        action_sequence: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Imagine ``H`` steps into the future for candidate action plans.

        Args:
            current_obs: ``(B, ...)`` latest observation.
            action_sequence: ``(B, S, H, action_dim)`` candidate plans
                (``S`` candidates per batch element).
            history: Optional ``(B, T_h, ...)`` past observations to prepend.

        Returns:
            ``(predicted_emb, predicted_reward, predicted_done_prob)``
            with shapes ``(B, S, H, D)``, ``(B, S, H)``, ``(B, S, H)``.
        """
        b = current_obs.size(0)
        s = action_sequence.size(1)
        h = action_sequence.size(2)

        if history is None:
            obs_seq = current_obs.unsqueeze(1)  # (B, 1, ...)
        else:
            obs_seq = torch.cat([history, current_obs.unsqueeze(1)], dim=1)

        emb = self.encode_obs(obs_seq)  # (B, T_h, D)
        emb = emb.unsqueeze(1).expand(b, s, -1, -1).clone()  # (B, S, T_h, D)
        emb = rearrange(emb, "b s t d -> (b s) t d")

        act = rearrange(action_sequence, "b s h a -> (b s) h a")
        act_emb_full = self.encode_actions(act)  # (B*S, H, D)

        pred_embs = []
        for t in range(h):
            act_emb_step = act_emb_full[:, : t + 1]
            ctx_emb = emb[:, -act_emb_step.size(1) :]
            pred = self.predict_next(ctx_emb, act_emb_step)[:, -1:]  # (B*S, 1, D)
            emb = torch.cat([emb, pred], dim=1)
            pred_embs.append(pred)

        pred_emb = torch.cat(pred_embs, dim=1)  # (B*S, H, D)

        reward_pred = self.reward_head(pred_emb, act_emb_full)  # (B*S, H)
        done_prob = torch.sigmoid(self.done_head(pred_emb))  # (B*S, H)

        pred_emb = rearrange(pred_emb, "(b s) h d -> b s h d", b=b, s=s)
        reward_pred = rearrange(reward_pred, "(b s) h -> b s h", b=b, s=s)
        done_prob = rearrange(done_prob, "(b s) h -> b s h", b=b, s=s)
        return pred_emb, reward_pred, done_prob

    @torch.no_grad()
    def score_action_sequences(
        self,
        current_obs: torch.Tensor,
        action_sequence: torch.Tensor,
        *,
        discount: float = 0.95,
        done_penalty: float = 1.0,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a ``(B, S)`` value estimate for candidate action sequences.

        Higher is better. Used by :mod:`tools.lewm.train` to evaluate
        rollouts and (after M5) by ``BrainPlanner`` to pick actions.
        """
        _, reward_pred, done_prob = self.rollout(
            current_obs, action_sequence, history=history
        )
        h = reward_pred.size(-1)
        discounts = torch.tensor(
            [discount**t for t in range(h)],
            device=reward_pred.device,
            dtype=reward_pred.dtype,
        )
        # weight rewards by chance of still being alive at that step
        survive = torch.cumprod(1.0 - done_prob, dim=-1)
        discounted = reward_pred * discounts * survive
        value = discounted.sum(dim=-1)
        value = value - done_penalty * done_prob[..., -1]
        return value
