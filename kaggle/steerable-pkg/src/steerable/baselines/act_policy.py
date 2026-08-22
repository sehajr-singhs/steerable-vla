"""Action Chunking Transformer (ACT) baseline.

Simplified version of Zhao et al. 2023: encoder-processes obs, decoder
autoregressively generates an action chunk. For our comparison, we use
a lightweight transformer (2 layers, 4 heads) to match the flow policy's
parameter count.
"""

import torch
import torch.nn as nn
import numpy as np
import math


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class _ACTNet(nn.Module):
    def __init__(self, dim_obs: int, dim_action: int, H: int = 6,
                 d_model: int = 128, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.H = H
        self.d_model = d_model
        self.obs_proj = nn.Linear(dim_obs, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.action_head = nn.Linear(d_model, dim_action)
        # learnable action query tokens
        self.action_queries = nn.Parameter(torch.randn(1, H, d_model) * 0.02)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim) → actions: (B, H, act_dim)"""
        B = obs.shape[0]
        memory = self.obs_proj(obs).unsqueeze(1)  # (B, 1, d_model)
        memory = self.pos_enc(memory)
        queries = self.action_queries.expand(B, -1, -1)  # (B, H, d_model)
        queries = self.pos_enc(queries)
        decoded = self.decoder(queries, memory)  # (B, H, d_model)
        return self.action_head(decoded)  # (B, H, act_dim)


class ACTPolicy:
    """Action Chunking Transformer policy.

    Trained with MSE loss on action chunks. At inference, generates
    the full chunk flattened as (H*3,) to match the eval harness.
    """

    ACT_DIM = 3  # per-step action dimension (dx, dy, grab)

    def __init__(self, dim_obs: int, dim_action: int = 3, hidden: int = 128, lr: float = 3e-4):
        self.dim_obs = dim_obs
        self._H = 6
        # ACT predicts (H, 3) internally, flattened on output
        self._net = _ACTNet(dim_obs, self.ACT_DIM, H=self._H,
                            d_model=hidden, n_layers=2, n_heads=4)
        self._opt = torch.optim.AdamW(self._net.parameters(), lr=lr, weight_decay=1e-4)

    def train_step(self, obs_batch: torch.Tensor, act_batch: torch.Tensor) -> float:
        """obs: (B, obs_dim), act: (B, H, 3) or (B, H*3) → loss."""
        if act_batch.dim() == 3:
            target = act_batch[:, :self._H, :self.ACT_DIM]  # (B, H, 3)
        else:
            target = act_batch[:, :self._H * self.ACT_DIM].reshape(-1, self._H, self.ACT_DIM)
        pred = self._net(obs_batch)
        loss = nn.functional.mse_loss(pred, target)
        self._opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
        self._opt.step()
        return float(loss.item())

    @torch.no_grad()
    def act(self, obs: np.ndarray, subgoal: np.ndarray = None) -> np.ndarray:
        """Return flat (H*3,) action chunk for the eval harness."""
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        # model returns (1, H, 3); flatten to (H*3,)
        return self._net(x).squeeze(0).reshape(-1).numpy().astype(np.float32)

    def state_dict(self):
        return {"net": self._net.state_dict()}

    def load_state_dict(self, d):
        self._net.load_state_dict(d["net"])
