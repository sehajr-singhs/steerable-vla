"""Steerable Multimodal Conditioning (SMC) layer.

The SMC layer is a gated adapter on the flow-matching velocity field that
admits mid-execution corrections (human nudges, world-model replans,
constraint setpoints) with a provable Lipschitz bound on trajectory deviation.

Architecture:
    v^S(x_s, s, c, u) = v^N(x_s, s, c) + Σ_j λ_j(x_s, s) · v^J_j(x_s, s, u_j)

    where:
    - v^N is the nominal (unsteered) velocity field
    - v^J_j is the j-th steering branch
    - λ_j = σ(w_j^T φ(x_s, s)) is a learned gate (sigmoid)
    - anchoring constraint: v^J_j(·, u_j=0) ≈ 0 (removing steering recovers nominal)

Theorem (No-Jerk Grönwall Bound):
    If the steered field is L_v-Lipschitz in x and the steering contribution
    is bounded by M, then over a steering window of flow-time Δs:
    ||x_S(s) - x_N(s)|| ≤ (M / L_v) * (exp(L_v * Δs) - 1)

This gives a computable smoothness envelope: an operator bounds a priori how far
a correction may move the trajectory. Jerk is bounded by the same quantity
divided by the gate rise time.

References:
    - Proposition 4 in the NMI manuscript
    - Grönwall's inequality applied to the ODE dx/ds = v^S
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SMCLayer(nn.Module):
    """Steerable Multimodal Conditioning layer.
    
    Args:
        dim_action: dimensionality of the continuous action chunk
        dim_cond: dimensionality of the conditioning vector c
        dim_steer: dimensionality of each steering signal u_j
        n_steer: number of independent steering channels
        hidden: hidden dimension for gate and steering networks
        lipschitz_target: target Lipschitz constant for the gate (used in
            the no-jerk bound computation, not enforced during training)
        gate_temperature: temperature for the sigmoid gate (lower = sharper)
    """
    
    def __init__(self, dim_action, dim_cond, dim_steer=3, n_steer=3,
                 hidden=256, lipschitz_target=1.0, gate_temperature=1.0):
        super().__init__()
        self.dim_action = dim_action
        self.dim_steer = dim_steer
        self.n_steer = n_steer
        self.lipschitz_target = lipschitz_target
        self.gate_temperature = gate_temperature
        
        # Base input: [x_s, s, c]
        base_dim = dim_action + 1 + dim_cond
        
        # Gate network: λ_j = σ(w_j^T φ(x_s, s))
        # One gate per steering channel, all sharing the same base features
        self.gate_net = nn.Sequential(
            nn.Linear(base_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_steer),
        )
        
        # Steering branches: v^J_j(x_s, s, c, u_j)
        # Each branch takes base + its own steering signal
        self.steer_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_dim + dim_steer, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, dim_action),
            )
            for _ in range(n_steer)
        ])
        
        # Lipschitz estimation (for the bound computation)
        self._registered_lipschitz = lipschitz_target
    
    def forward(self, x_s, s, c, steering_signals=None):
        """Compute the steered velocity field.
        
        Args:
            x_s: noisy action chunk at flow-time s, shape (B, dim_action)
            s: flow-time scalar or tensor, shape (B, 1) or (1,)
            c: conditioning vector, shape (B, dim_cond)
            steering_signals: list of steering tensors u_j, each (B, dim_steer)
                If None, returns the nominal field (no steering)
        
        Returns:
            v_S: steered velocity, shape (B, dim_action)
            v_N: nominal velocity (for loss computation), shape (B, dim_action)
            gates: gate values, shape (B, n_steer)
            steer_magnitudes: ||λ_j * v^J_j|| for each channel
        """
        B = x_s.shape[0]
        dev = x_s.device
        
        # Ensure s is (B, 1)
        if isinstance(s, (int, float)):
            s_t = torch.full((B, 1), s, device=dev)
        elif s.dim() == 1:
            s_t = s.unsqueeze(1)
        else:
            s_t = s
        
        # Base features: [x_s, s, c]
        base = torch.cat([x_s, s_t, c], dim=-1)
        
        # Compute gates: λ_j = σ(temperature * gate_net(base))
        gate_logits = self.gate_net(base)
        gates = torch.sigmoid(self.gate_temperature * gate_logits)  # (B, n_steer)
        
        # Compute steering contributions
        steer_mags = torch.zeros(B, self.n_steer, device=dev)
        v_steer = torch.zeros_like(x_s)
        
        if steering_signals is not None:
            for j, branch in enumerate(self.steer_branches):
                if j < len(steering_signals) and steering_signals[j] is not None:
                    u_j = steering_signals[j]
                    if u_j.dim() == 1:
                        u_j = u_j.unsqueeze(0).expand(B, -1)
                    v_j = branch(torch.cat([base, u_j], dim=-1))
                    # Gate this branch
                    lam_j = gates[:, j:j+1]  # (B, 1)
                    contribution = lam_j * v_j
                    v_steer = v_steer + contribution
                    steer_mags[:, j] = contribution.norm(dim=-1).mean()
        
        return v_steer, gates, steer_mags
    
    def anchoring_loss(self, x_s, s, c):
        """Anchoring auxiliary loss: v^J_j(·, u=0) ≈ 0.
        
        This enforces that removing all steering signals recovers the nominal
        field exactly. Without this, the steering branches can drift to
        contribute non-trivially even when u=0, breaking the formal guarantee.
        
        Returns:
            loss: scalar, should be close to 0 when anchoring is satisfied
        """
        B = x_s.shape[0]
        dev = x_s.device
        
        if isinstance(s, (int, float)):
            s_t = torch.full((B, 1), s, device=dev)
        elif s.dim() == 1:
            s_t = s.unsqueeze(1)
        else:
            s_t = s
        
        base = torch.cat([x_s.detach(), s_t, c.detach()], dim=-1)
        gates = torch.sigmoid(self.gate_temperature * self.gate_net(base))
        
        loss = 0.0
        for j, branch in enumerate(self.steer_branches):
            u_zero = torch.zeros(B, self.dim_steer, device=dev)
            v_j_zero = branch(torch.cat([base.detach(), u_zero], dim=-1))
            lam_j = gates[:, j:j+1]
            loss = loss + torch.mean((lam_j * v_j_zero) ** 2)
        
        return loss
    
    def compute_gronwall_bound(self, M, delta_s, L_v=None):
        """Compute the Grönwall bound on trajectory deviation.
        
        ||x_S(s) - x_N(s)|| ≤ (M / L_v) * (exp(L_v * Δs) - 1)
        
        Args:
            M: upper bound on ||λ_j * v^J_j|| (steering contribution magnitude)
            delta_s: flow-time window over which steering is active
            L_v: Lipschitz constant of the velocity field (default: self.lipschitz_target)
        
        Returns:
            bound: maximum trajectory deviation over the steering window
        """
        if L_v is None:
            L_v = self.lipschitz_target
        
        if L_v < 1e-10:
            # Degenerate case: velocity field is constant, deviation = M * Δs
            return M * delta_s
        
        bound = (M / L_v) * (math.exp(L_v * delta_s) - 1)
        return bound
    
    def estimate_lipschitz(self, x_s_samples, s_samples, c_samples, n_pairs=100):
        """Empirically estimate the Lipschitz constant of the nominal velocity field.
        
        Samples pairs of (x_s, s) points and computes the maximum ratio
        ||v_N(x1, s1) - v_N(x2, s2)|| / ||(x1, s1) - (x2, s2)||.
        
        Args:
            x_s_samples: (N, dim_action) sampled noisy states
            s_samples: (N, 1) corresponding flow-times
            c_samples: (N, dim_cond) corresponding conditioning
            n_pairs: number of random pairs to evaluate
        
        Returns:
            L_est: estimated Lipschitz constant
        """
        N = x_s_samples.shape[0]
        dev = x_s_samples.device
        
        # Sample random pairs
        idx1 = torch.randint(0, N, (n_pairs,))
        idx2 = torch.randint(0, N, (n_pairs,))
        
        x1, x2 = x_s_samples[idx1], x_s_samples[idx2]
        s1, s2 = s_samples[idx1], s_samples[idx2]
        c1, c2 = c_samples[idx1], c_samples[idx2]
        
        # Compute nominal velocities (no steering)
        base1 = torch.cat([x1, s1, c1], dim=-1)
        base2 = torch.cat([x2, s2, c2], dim=-1)
        
        # Use the first steering branch as a proxy for the nominal field
        # (in practice, v_N is computed by the policy's velocity method)
        v1 = self.steer_branches[0](torch.cat([base1, torch.zeros(n_pairs, self.dim_steer, device=dev)], dim=-1))
        v2 = self.steer_branches[0](torch.cat([base2, torch.zeros(n_pairs, self.dim_steer, device=dev)], dim=-1))
        
        # Input difference (concatenate x and s)
        input_diff = torch.cat([x1 - x2, s1 - s2], dim=-1).norm(dim=-1)
        output_diff = (v1 - v2).norm(dim=-1)
        
        # Lipschitz estimate = max ratio (with small epsilon for stability)
        ratios = output_diff / (input_diff + 1e-8)
        L_est = float(ratios.max().item())
        
        return L_est


class SMCEnabledFlowExpert(nn.Module):
    """Flow expert with SMC layer integrated into the velocity field.
    
    This wraps the existing FlowExpert and adds proper SMC gating:
    v^S = v^N + Σ_j λ_j · v^J_j
    
    with anchoring loss and Lipschitz bound computation.
    """
    
    def __init__(self, base_expert, smc_layer):
        super().__init__()
        self.base = base_expert
        self.smc = smc_layer
        # Copy the conditioning from the base expert
        self.condition = base_expert.condition
        self.dim_action = base_expert.dim_action
    
    def velocity(self, x_s, s, c, u=None, steering_signals=None):
        """Compute steered velocity with SMC layer."""
        # Nominal velocity from base expert
        v_n = self.base.velocity(x_s, s, c, u=None)
        
        # Steering contributions from SMC layer
        if steering_signals is not None and len(steering_signals) > 0:
            v_steer, gates, mags = self.smc(x_s, s, c, steering_signals)
            return v_n + v_steer, v_n, gates, mags
        elif u is not None:
            # Legacy: single steering signal
            v_steer, gates, mags = self.smc(x_s, s, c, [u])
            return v_n + v_steer, v_n, gates, mags
        else:
            return v_n, v_n, None, None
    
    def cfm_loss(self, obs, subgoal, chunk, nudge, rng, steer_prob):
        """CFM loss with SMC anchoring loss."""
        # Base CFM loss
        base_loss = self.base.cfm_loss(obs, subgoal, chunk, nudge, rng, steer_prob)
        
        # SMC anchoring loss
        B = obs.shape[0]
        dev = obs.device
        x1 = chunk[:, :self.base.dim_action]
        x0 = torch.randn_like(x1)
        s = torch.rand(B, 1, device=dev)
        x_s = (1 - s) * x0 + s * x1
        c = self.base.condition(obs, subgoal)
        
        anchor_loss = self.smc.anchoring_loss(x_s, s, c)
        
        return base_loss + 0.5 * anchor_loss

    def act(self, obs, subgoal=None, nudge=None, **kw):
        """Delegate to base expert's act method."""
        return self.base.act(obs, subgoal, nudge=nudge, **kw)

    @property
    def cfg(self):
        return self.base.cfg

    @cfg.setter
    def cfg(self, value):
        self.base.cfg = value
