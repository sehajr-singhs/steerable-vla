"""Transformer backbone policy for cable untangling.

A proper sequence-to-sequence transformer that:
1. Encodes the cable node observations as a sequence of tokens
2. Attends over them with self-attention to capture spatial relationships
3. Produces action chunks through cross-attention with subgoal tokens

This should learn much better than the MLP because it can represent
the spatial structure of the cable and the relationship between nodes.
"""

import torch
import torch.nn as nn
import numpy as np
import math

from ..config import TrainConfig


def _d(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class CableTransformer(nn.Module):
    """Transformer that encodes cable node positions and produces actions.
    
    The key insight: cable untangling requires understanding SPATIAL
    RELATIONSHIPS between nodes (which ones cross, which are near each
    other). A transformer with self-attention can represent this; an MLP
    on flattened nodes cannot.
    """
    
    def __init__(self, n_nodes, n_actions, hidden=256, n_heads=8,
                 n_layers=4, dropout=0.1):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_actions = n_actions  # H * 3
        self.H = n_actions // 3     # chunk length
        
        # Node encoder: each node (x, y) -> token
        self.node_proj = nn.Linear(2, hidden)
        self.node_pos = nn.Parameter(torch.randn(1, n_nodes, hidden) * 0.02)
        
        # Gripper encoding
        self.gripper_proj = nn.Linear(2, hidden)
        
        # Global conditioning: crossings_ratio, step_fraction
        self.global_proj = nn.Linear(2, hidden)
        
        # Self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Subgoal conditioning
        self.subgoal_proj = nn.Linear(n_nodes * 2 + 4, hidden)
        
        # Cross-attention: actions attend to encoded nodes
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        # Action query tokens (learnable)
        self.action_queries = nn.Parameter(torch.randn(1, self.H, hidden) * 0.02)
        
        # Action heads: dx, dy (continuous) + grab (discrete)
        self.action_head = nn.Linear(hidden, 3)  # per-step: dx, dy, grab_logit
        
        # Independent grab head (like the flow expert)
        self.grab_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1)
        )
    
    def encode(self, obs):
        """Encode observation into node tokens + global token.
        
        obs layout: [nodes (n_nodes*2), gripper (2), crossings_ratio, step_frac]
        """
        B = obs.shape[0]
        n = self.n_nodes
        
        # Split obs
        nodes_flat = obs[:, :n * 2]  # (B, n*2)
        gripper = obs[:, n * 2:n * 2 + 2]  # (B, 2)
        global_feat = obs[:, n * 2 + 2:]  # (B, 2+)
        
        # Project nodes to tokens: (B, n, hidden)
        node_xy = nodes_flat.reshape(B, n, 2)
        node_tokens = self.node_proj(node_xy) + self.node_pos[:, :n]
        
        # Add gripper as a special token
        gripper_token = self.gripper_proj(gripper).unsqueeze(1)  # (B, 1, hidden)
        global_token = self.global_proj(global_feat[:, :2]).unsqueeze(1)  # (B, 1, hidden)
        
        # Concatenate: [nodes, gripper, global]
        tokens = torch.cat([node_tokens, gripper_token, global_token], dim=1)
        
        # Self-attention
        encoded = self.encoder(tokens)
        
        return encoded
    
    def forward(self, obs, subgoal=None):
        """Produce action chunk from observation.
        
        Returns: (B, H*3) flat action chunk
        """
        B = obs.shape[0]
        
        # Encode observation
        memory = self.encode(obs)  # (B, n+2, hidden)
        
        # Subgoal conditioning
        if subgoal is not None:
            sub_tokens = self.subgoal_proj(subgoal).unsqueeze(1)  # (B, 1, hidden)
            memory = torch.cat([memory, sub_tokens], dim=1)
        
        # Action queries
        queries = self.action_queries.expand(B, -1, -1)  # (B, H, hidden)
        
        # Cross-attention: actions attend to encoded nodes
        decoded = self.decoder(queries, memory)  # (B, H, hidden)
        
        # Action heads
        actions = self.action_head(decoded)  # (B, H, 3)
        
        # Flatten: (B, H*3)
        return actions.reshape(B, -1)
    
    def act(self, obs, subgoal=None, **kw):
        """Inference: return flat (H*3,) action chunk."""
        self.eval()
        with torch.no_grad():
            dev = next(self.parameters()).device
            obs_t = _d(obs, dev).unsqueeze(0)
            sg_t = _d(subgoal, dev).unsqueeze(0) if subgoal is not None else None
            out = self.forward(obs_t, sg_t)  # (1, H*3)
            return out[0].cpu().numpy().astype(np.float32)


class TransformerPolicy:
    """Wrapper for training + inference of CableTransformer.
    
    Exposes train()/eval()/parameters() so it works with train_policy().
    """
    
    def __init__(self, n_nodes, obs_dim, hidden=256, lr=3e-4):
        self.n_nodes = n_nodes
        self.obs_dim = obs_dim
        self.H = 4  # action chunk length
        self.n_actions = self.H * 3
        
        self._net = CableTransformer(
            n_nodes=n_nodes, n_actions=self.n_actions,
            hidden=hidden, n_heads=8, n_layers=4
        )
        self._opt = torch.optim.AdamW(
            self._net.parameters(), lr=lr, weight_decay=1e-4
        )
        self.use_subgoal = True
    
    def train(self, mode=True):
        self._net.train(mode)
        return self
    
    def eval(self):
        self._net.eval()
        return self
    
    def parameters(self):
        return self._net.parameters()

    def train_step(self, obs_batch, act_batch):
        """Train on one batch. obs: (B, obs_dim), act: (B, H*3) or (B, H, 3)."""
        if act_batch.dim() == 3:
            target = act_batch[:, :self.H, :3].reshape(-1, self.n_actions)
        else:
            target = act_batch[:, :self.n_actions]

        self._net.train()
        pred = self._net(obs_batch)
        loss = nn.functional.mse_loss(pred, target)
        self._opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
        self._opt.step()
        return float(loss.item())
    
    def act(self, obs, subgoal=None, **kw):
        """Flat (H*3,) action chunk for the eval harness."""
        return self._net.act(obs, subgoal)
    
    def state_dict(self):
        return {"net": self._net.state_dict()}
    
    def load_state_dict(self, d):
        self._net.load_state_dict(d["net"])
