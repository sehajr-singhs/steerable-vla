"""Diffusion Policy baseline (Chi et al. 2023).

Simplified DDPM for action generation: condition on obs, denoise a
random action sequence into a coherent chunk. Uses a small 1D U-Net
for the denoising network.

For fair comparison with our flow-matching policy, we use the same
number of denoising steps (N=5 for training, 10 for inference) and
the same parameter budget.
"""

import torch
import torch.nn as nn
import numpy as np


class _SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _Denoiser1D(nn.Module):
    """1D U-Net denoiser for action sequences."""

    def __init__(self, act_dim: int, obs_dim: int, H: int = 6, d: int = 128):
        super().__init__()
        self.H = H
        total_dim = act_dim * H

        self.time_emb = nn.Sequential(
            _SinusoidalEmbedding(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)
        )
        self.obs_proj = nn.Linear(obs_dim, d)

        self.in_proj = nn.Linear(total_dim + d + d, d * 2)
        self.down1 = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d))
        self.mid = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.up1 = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d))
        self.out_proj = nn.Linear(d, total_dim)

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor,
                obs: torch.Tensor) -> torch.Tensor:
        """x_noisy: (B, H*act_dim), t: (B,), obs: (B, obs_dim) → eps: same shape"""
        B = x_noisy.shape[0]
        t_emb = self.time_emb(t)
        o_emb = self.obs_proj(obs)

        inp = torch.cat([x_noisy, t_emb, o_emb], dim=-1)
        h = self.in_proj(inp)
        h1 = self.down1(h)
        h_mid = self.mid(h1)
        h_cat = torch.cat([h_mid, h1], dim=-1)
        h_out = self.up1(h_cat)
        return self.out_proj(h_out)


class DiffusionPolicy:
    """DDPM-based action generation policy.

    Training: sample noise, denoise for 1 step, predict noise.
    Inference: 10-step denoising from pure noise.
    """

    def __init__(self, dim_obs: int, dim_action: int, hidden: int = 128,
                 lr: float = 2e-4, n_train_steps: int = 5, n_infer_steps: int = 10):
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self._H = 6
        self._n_train = n_train_steps
        self._n_infer = n_infer_steps
        self._total_dim = dim_action * self._H

        self._denoiser = _Denoiser1D(dim_action, dim_obs, H=self._H, d=hidden)
        self._opt = torch.optim.AdamW(self._denoiser.parameters(), lr=lr)

        # DDPM schedule
        N = n_infer_steps
        self._betas = torch.linspace(1e-4, 0.02, N)
        self._alphas = 1.0 - self._betas
        self._alpha_bar = torch.cumprod(self._alphas, dim=0)

    def train_step(self, obs_batch: torch.Tensor, act_batch: torch.Tensor) -> float:
        """obs: (B, obs_dim), act: (B, T, act_dim) → loss."""
        B = obs_batch.shape[0]
        target = act_batch[:, :self._H].reshape(B, self._total_dim)

        # sample random timestep and noise
        t = torch.randint(0, self._n_train, (B,))
        ab = self._alpha_bar.to(obs_batch.device)[t]
        noise = torch.randn_like(target)
        noisy = ab.sqrt().unsqueeze(1) * target + (1 - ab).sqrt().unsqueeze(1) * noise

        eps_pred = self._denoiser(noisy, t.float(), obs_batch)
        loss = nn.functional.mse_loss(eps_pred, noise)
        self._opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._denoiser.parameters(), 1.0)
        self._opt.step()
        return float(loss.item())

    @torch.no_grad()
    def act(self, obs: np.ndarray, subgoal: np.ndarray = None) -> np.ndarray:
        """Return (H, act_dim) action chunk via DDPM denoising."""
        device = next(self._denoiser.parameters()).device
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        # start from noise
        x = torch.randn(1, self._total_dim, device=device)

        for i in range(self._n_infer):
            t = torch.full((1,), i, dtype=torch.float32, device=device)
            eps = self._denoiser(x, t, obs_t)
            alpha = self._alphas[i].to(device)
            alpha_bar = self._alpha_bar[i].to(device)
            x = (1 / alpha.sqrt()) * (x - (1 - alpha) / (1 - alpha_bar).sqrt() * eps)
            if i < self._n_infer - 1:
                x += torch.randn_like(x) * (self._betas[i] ** 0.5).to(device)

        return x.reshape(self._H, self.dim_action).cpu().numpy().astype(np.float32)

    def state_dict(self):
        return {"denoiser": self._denoiser.state_dict()}

    def load_state_dict(self, d):
        self._denoiser.load_state_dict(d["denoiser"])
