"""Diffusion Policy baseline: flat diffusion over action chunks.

This is the standard diffusion policy (Chi et al., 2023) adapted to our
cable untangling task. It denoises action chunks directly, without
subgoal conditioning or safety filtering.

Reference: Chi et al., "Diffusion Policy: Visuomotor Policy Learning
via Action Diffusion" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConditionalDenoiser(nn.Module):
    """Simple 1D U-Net denoiser for action chunks."""
    
    def __init__(self, action_dim, cond_dim, hidden=128, n_steps=4):
        super().__init__()
        self.action_dim = action_dim
        self.hidden = hidden
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(32, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        
        # Condition embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        
        # Denoising network (simple MLP for miniature scale)
        self.net = nn.Sequential(
            nn.Linear(action_dim + hidden + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )
    
    def _timestep_embedding(self, t, dim=32):
        """Sinusoidal timestep embedding."""
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    def forward(self, x_noisy, t, condition):
        """
        Args:
            x_noisy: (B, action_dim) noisy action
            t: (B,) diffusion timestep
            condition: (B, cond_dim) conditioning vector
        Returns:
            noise_pred: (B, action_dim) predicted noise
        """
        t_emb = self.time_embed(self._timestep_embedding(t))
        c_emb = self.cond_embed(condition)
        
        h = torch.cat([x_noisy, t_emb, c_emb], dim=-1)
        return self.net(h)


class DiffusionPolicy(nn.Module):
    """Diffusion Policy: flat denoising over action chunks.
    
    At inference, iteratively denoises from Gaussian noise to an action,
    conditioned on the current observation.
    """
    
    def __init__(self, dim_obs, dim_subgoal, dim_action,
                 n_diffusion_steps=100, hidden=128):
        super().__init__()
        self.dim_action = dim_action
        self.n_steps = n_diffusion_steps
        
        # Condition = observation + subgoal concatenated
        cond_dim = dim_obs + dim_subgoal
        
        self.denoiser = ConditionalDenoiser(dim_action, cond_dim, hidden)
        
        # Beta schedule
        betas = torch.linspace(1e-4, 0.02, n_diffusion_steps)
        alphas = 1 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)
    
    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar = self.alphas_bar[t].reshape(-1, 1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
    
    def forward(self, obs, subgoal=None, actions=None):
        """Unified forward: returns loss when actions provided, else sampled actions.
        
        This is the interface the shared train_policy loop expects.
        """
        if subgoal is None:
            subgoal = torch.zeros_like(obs)
        if actions is not None:
            return self.training_loss(obs, subgoal, actions)
        return self.sample(obs, subgoal)
    
    def training_loss(self, obs, subgoal, chunk):
        """Compute the diffusion training loss.
        
        Args:
            obs: (B, dim_obs) observations
            subgoal: (B, dim_subgoal) subgoals
            chunk: (B, dim_action) action chunks
        Returns:
            loss: scalar
        """
        B = chunk.shape[0]
        dev = chunk.device
        
        # Sample random timesteps
        t = torch.randint(0, self.n_steps, (B,), device=dev)
        
        # Sample noise
        noise = torch.randn_like(chunk)
        
        # Add noise
        x_noisy = self.q_sample(chunk, t, noise)
        
        # Condition
        condition = torch.cat([obs, subgoal], dim=-1)
        
        # Predict noise
        noise_pred = self.denoiser(x_noisy, t, condition)
        
        return F.mse_loss(noise_pred, noise)
    
    @torch.no_grad()
    def sample(self, obs, subgoal, n_samples=1):
        """DDPM sampling: denoise from Gaussian to action.
        
        Args:
            obs: (1, dim_obs) or (B, dim_obs)
            subgoal: (1, dim_subgoal) or (B, dim_subgoal)
            n_samples: number of samples to average
        Returns:
            actions: (dim_action,) or (B, dim_action)
        """
        self.eval()
        dev = next(self.parameters()).device
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).to(dev)
            subgoal = subgoal.unsqueeze(0).to(dev)
        
        B = obs.shape[0]
        condition = torch.cat([obs, subgoal], dim=-1)
        
        # Start from noise
        x = torch.randn(B, self.dim_action, device=dev)
        
        # Denoise
        for t in reversed(range(self.n_steps)):
            t_tensor = torch.full((B,), t, device=dev, dtype=torch.long)
            noise_pred = self.denoiser(x, t_tensor, condition)
            
            alpha = self.alphas[t]
            alpha_bar = self.alphas_bar[t]
            beta = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_bar)) * noise_pred
            ) + torch.sqrt(beta) * noise
        
        # Average multiple samples
        all_samples = []
        for _ in range(n_samples):
            x = torch.randn(B, self.dim_action, device=dev)
            for t in reversed(range(self.n_steps)):
                t_tensor = torch.full((B,), t, device=dev, dtype=torch.long)
                noise_pred = self.denoiser(x, t_tensor, condition)
                alpha = self.alphas[t]
                alpha_bar = self.alphas_bar[t]
                beta = self.betas[t]
                noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
                x = (1 / torch.sqrt(alpha)) * (
                    x - (beta / torch.sqrt(1 - alpha_bar)) * noise_pred
                ) + torch.sqrt(beta) * noise
            all_samples.append(x)
        
        return torch.stack(all_samples).mean(0)
    
    def act(self, obs, subgoal=None, **kw):
        """Compatibility interface."""
        self.eval()
        dev = next(self.parameters()).device
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=dev)
        if subgoal is None:
            subgoal = torch.zeros_like(obs)
        elif isinstance(subgoal, np.ndarray):
            subgoal = torch.tensor(subgoal, dtype=torch.float32, device=dev)
        
        with torch.no_grad():
            return self.sample(obs.unsqueeze(0), subgoal.unsqueeze(0))[0].cpu().numpy()


import numpy as np
