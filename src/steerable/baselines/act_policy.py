"""ACT (Action Chunking Transformer) baseline.

Reference: Zhao et al., "Learning Fine-Grained Bimanual Manipulation
with Low-Cost Hardware" (2023)

ACT uses a CVAE (Conditional Variational Autoencoder) with a transformer
encoder to predict action chunks from observations. It's the standard
baseline for bimanual manipulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ACTEncoder(nn.Module):
    """Transformer encoder for ACT: processes observation tokens."""
    
    def __init__(self, embed_dim=128, n_heads=4, n_layers=3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    
    def forward(self, tokens):
        """Encode observation tokens."""
        return self.encoder(tokens)


class ACTDecoder(nn.Module):
    """Transformer decoder for ACT: generates action chunks."""
    
    def __init__(self, embed_dim=128, n_heads=4, n_layers=3, max_action_len=8):
        super().__init__()
        self.max_action_len = max_action_len
        
        # Learnable action queries
        self.action_queries = nn.Parameter(
            torch.randn(1, max_action_len, embed_dim) * 0.02
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
    
    def forward(self, memory, n_actions=None):
        """Decode action queries using encoded observations as memory."""
        B = memory.shape[0]
        K = n_actions or self.max_action_len
        queries = self.action_queries[:, :K, :].expand(B, -1, -1)
        return self.decoder(queries, memory)


class ACTPolicy(nn.Module):
    """Action Chunking Transformer with CVAE.
    
    Architecture:
        1. Observation encoder: tokenize obs → transformer
        2. CVAE: latent z encodes action diversity
        3. Action decoder: cross-attend to obs + z → action chunk
    
    Training: reconstruction loss + KL divergence
    Inference: sample z from prior, decode action chunk
    """
    
    def __init__(self, dim_obs, dim_subgoal, dim_action,
                 embed_dim=128, n_heads=4, n_layers=3,
                 max_action_len=8, latent_dim=32):
        super().__init__()
        self.dim_action = dim_action
        self.max_action_len = max_action_len
        self.latent_dim = latent_dim
        
        # Observation projection
        self.obs_proj = nn.Linear(dim_obs + dim_subgoal, embed_dim)
        
        # Encoder
        self.encoder = ACTEncoder(embed_dim, n_heads, n_layers)
        
        # CVAE: encoder (obs → latent) and decoder (obs + z → action)
        self.cvae_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, latent_dim * 2),  # mean + logvar
        )
        self.cvae_decoder = nn.Linear(latent_dim, embed_dim)
        
        # Action decoder
        self.action_decoder = ACTDecoder(embed_dim, n_heads, n_layers, max_action_len)
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, dim_action),
        )
    
    def encode(self, obs, subgoal):
        """Encode observation into tokens."""
        x = self.obs_proj(torch.cat([obs, subgoal], dim=-1))
        return self.encoder(x.unsqueeze(1))  # (B, 1, embed_dim)
    
    def reparameterize(self, mu, logvar):
        """CVAE reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def forward(self, obs, subgoal, actions=None):
        """Forward pass.
        
        Args:
            obs: (B, dim_obs)
            subgoal: (B, dim_subgoal)
            actions: (B, T, dim_action) for training, None for inference
        Returns:
            If training: (loss, kl_loss)
            If inference: (B, T, dim_action)
        """
        B = obs.shape[0]
        dev = obs.device
        
        # Encode observation
        memory = self.encode(obs, subgoal)  # (B, 1, embed_dim)
        
        # CVAE: encode to latent
        mu_logvar = self.cvae_encoder(memory.mean(dim=1))  # (B, latent_dim*2)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        
        if actions is not None:
            # Training: use ground truth actions to encode latent
            action_tokens = actions.reshape(B, -1, self.dim_action)
            # Simple: use mean of actions as latent input
            action_mean = actions.mean(dim=1)
            z_input = self.obs_proj(torch.cat([obs, subgoal], dim=-1))
            z_mu_logvar = self.cvae_encoder(z_input)
            z_mu, z_logvar = z_mu_logvar.chunk(2, dim=-1)
            z = self.reparameterize(z_mu, z_logvar)
            
            # Decode actions
            z_emb = self.cvae_decoder(z).unsqueeze(1)
            decoded = self.action_decoder(memory + z_emb)
            action_pred = self.action_head(decoded)
            
            # Losses
            recon_loss = F.mse_loss(action_pred, actions.unsqueeze(1))
            kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
            
            return recon_loss + 0.01 * kl_loss
        else:
            # Inference: sample from prior
            z = torch.randn(B, self.latent_dim, device=dev)
            z_emb = self.cvae_decoder(z).unsqueeze(1)
            decoded = self.action_decoder(memory + z_emb)
            return self.action_head(decoded)
    
    def act(self, obs, subgoal=None, n_samples=1, **kw):
        """Compatibility interface."""
        self.eval()
        dev = next(self.parameters()).device
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
        if subgoal is None:
            subgoal = torch.zeros_like(obs)
        elif isinstance(subgoal, np.ndarray):
            subgoal = torch.tensor(subgoal, dtype=torch.float32, device=dev).unsqueeze(0)
        
        with torch.no_grad():
            all_actions = []
            for _ in range(n_samples):
                actions = self.forward(obs, subgoal)
                all_actions.append(actions[0])
            # Average multiple samples
            return torch.stack(all_actions).mean(0).cpu().numpy().flatten()
