"""Behavior Cloning baseline — MLP mapping obs → action.

This is the simplest baseline: direct supervised regression from
observations to actions, with no flow matching or action chunking.
"""

import torch
import torch.nn as nn
import numpy as np


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1)]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BCPolicy:
    """Behavior cloning policy.

    Trained with MSE loss on (obs, action) pairs from demonstrations.
    At inference, directly outputs the predicted action chunk.
    """

    def __init__(self, dim_obs: int, dim_action: int, hidden: int = 256, lr: float = 1e-3):
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self._net = _MLP(dim_obs, dim_action * 6, hidden=hidden)  # predict H=6 chunk
        self._opt = torch.optim.Adam(self._net.parameters(), lr=lr)
        self._H = 6  # action chunk size

    def train_step(self, obs_batch: torch.Tensor, act_batch: torch.Tensor) -> float:
        """Train one step. obs: (B, obs_dim), act: (B, H, act_dim)."""
        B, T, A = act_batch.shape
        # flatten chunk to supervised target
        target = act_batch[:, :self._H].reshape(B, self._H * A)
        pred = self._net(obs_batch)
        loss = nn.functional.mse_loss(pred, target)
        self._opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
        self._opt.step()
        return float(loss.item())

    @torch.no_grad()
    def act(self, obs: np.ndarray, subgoal: np.ndarray = None) -> np.ndarray:
        """Return action chunk (H, act_dim)."""
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        pred = self._net(x).squeeze(0).numpy()
        return pred.reshape(self._H, self.dim_action).astype(np.float32)

    def state_dict(self):
        return {"net": self._net.state_dict()}

    def load_state_dict(self, d):
        self._net.load_state_dict(d["net"])
