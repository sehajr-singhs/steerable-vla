"""RT-2-style baseline: single-network VLA, no hierarchy.

At miniature scale, this is a direct MLP: (obs + language) → action chunk.
No flow matching, no subgoals, no safety filter.

Reference: Brohan et al., "RT-2: Vision-Language-Action Models
Transfer Web Knowledge to Robotic Control" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RT2Policy(nn.Module):
    """RT-2-style policy: single network, no hierarchy.
    
    At miniature scale, this is a direct MLP: obs → action chunk.
    No flow matching, no subgoals, no safety filter.
    """
    
    def __init__(self, dim_obs=70, dim_subgoal=70, dim_action=12,
                 embed_dim=128, **kwargs):
        super().__init__()
        self.dim_action = dim_action
        self.embed_dim = embed_dim
        self._built = False
        self._dim_obs = dim_obs
        self._dim_sub = dim_subgoal
    
    def _ensure_built(self, total_dim):
        if self._built:
            return
        h = self.embed_dim
        self.net = nn.Sequential(
            nn.Linear(total_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, self.dim_action),
        )
        self._built = True
    
    def forward(self, obs, subgoal=None, actions=None):
        if subgoal is None:
            subgoal = torch.zeros_like(obs)
        x = torch.cat([obs, subgoal], dim=-1)
        self._ensure_built(x.shape[-1])
        out = self.net(x)
        if actions is not None:
            return F.mse_loss(out, actions)
        return out
    
    def act(self, obs, subgoal=None, **kw):
        self.eval()
        dev = next(self.parameters()).device if self._built else 'cpu'
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=dev)
        if subgoal is None:
            subgoal = torch.zeros_like(obs)
        elif isinstance(subgoal, np.ndarray):
            subgoal = torch.tensor(subgoal, dtype=torch.float32, device=dev)
        with torch.no_grad():
            return self.forward(obs.unsqueeze(0), subgoal.unsqueeze(0))[0].cpu().numpy()
