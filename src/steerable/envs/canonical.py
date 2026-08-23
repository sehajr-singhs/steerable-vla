"""Canonical cross-embodiment action frame.

Actions are normalized end-effector deltas in a universal coordinate system:
    (Δp_ee, Δψ, gripper) where:
    - Δp_ee: position delta in canonical frame (x, y, z normalized to [-1, 1])
    - Δψ: rotation delta (normalized)
    - gripper: open/close (binary)

This enables transfer across morphologies (different robots, different cable
configurations) without reparameterization.

Reference: Section 2.1 of the NMI manuscript — "Canonical actions. The
cross-embodiment frame transfers learned segments across morphologies
without reparameterization."
"""

import numpy as np


class CanonicalActionFrame:
    """Convert between embodiment-specific and canonical action frames.
    
    The canonical frame normalizes all actions to:
    - Position: [-1, 1]^D (normalized by workspace bounds)
    - Rotation: [-1, 1] (normalized by max rotation)
    - Gripper: {0, 1} (binary)
    
    This allows the same policy to work across:
    - Different cable configurations (different number of nodes, lengths)
    - Different robot embodiments (different workspace sizes)
    - Different tasks (cable, textile, tool use) with unified action space
    """
    
    def __init__(self, workspace_bounds=None, max_rotation=2*np.pi):
        """
        Args:
            workspace_bounds: (x_lo, x_hi, y_lo, y_hi, [z_lo, z_hi])
                If None, uses default cable workspace
            max_rotation: maximum rotation angle for normalization
        """
        if workspace_bounds is None:
            self.bounds = np.array([-0.95, 0.95, -0.30, 0.95])
        else:
            self.bounds = np.array(workspace_bounds)
        
        self.max_rotation = max_rotation
        self.workspace_size = np.array([
            self.bounds[1] - self.bounds[0],
            self.bounds[3] - self.bounds[2],
        ])
        self.workspace_center = np.array([
            (self.bounds[0] + self.bounds[1]) / 2,
            (self.bounds[2] + self.bounds[3]) / 2,
        ])
    
    def to_canonical(self, action, embodiment='cable'):
        """Convert embodiment-specific action to canonical frame.
        
        Args:
            action: (dx, dy, [dz], [dpsi], grab) embodiment-specific action
            embodiment: 'cable', 'ur5', 'franka', etc.
        
        Returns:
            canonical: normalized action in [-1, 1]^D
        """
        action = np.asarray(action, dtype=np.float64)
        
        if embodiment == 'cable':
            # Cable: (dx, dy, grab) -> normalize dx, dy by workspace size
            dx, dy = action[0], action[1]
            grab = action[2] if len(action) > 2 else 0.0
            
            norm_dx = np.clip(dx / (self.workspace_size[0] / 2), -1, 1)
            norm_dy = np.clip(dy / (self.workspace_size[1] / 2), -1, 1)
            
            return np.array([norm_dx, norm_dy, grab])
        
        elif embodiment == 'ur5':
            # UR5: (dx, dy, dz, dpsi, gripper) -> normalize all
            normalized = action[:5].copy()
            for i in range(3):
                normalized[i] = np.clip(
                    normalized[i] / (self.workspace_size[i % 2] / 2), -1, 1
                )
            if len(action) > 3:
                normalized[3] = np.clip(
                    normalized[3] / self.max_rotation, -1, 1
                )
            if len(action) > 4:
                normalized[4] = 1.0 if action[4] > 0.5 else 0.0
            
            return normalized
        
        else:
            # Generic: normalize by workspace
            n_dims = min(len(action), 5)
            normalized = action[:n_dims].copy()
            for i in range(min(n_dims, 2)):
                normalized[i] = np.clip(
                    normalized[i] / (self.workspace_size[i] / 2), -1, 1
                )
            return normalized
    
    def from_canonical(self, canonical, embodiment='cable'):
        """Convert canonical action back to embodiment-specific frame.
        
        Args:
            canonical: normalized action in [-1, 1]^D
            embodiment: target embodiment
        
        Returns:
            action: embodiment-specific action
        """
        canonical = np.asarray(canonical, dtype=np.float64)
        
        if embodiment == 'cable':
            dx = canonical[0] * (self.workspace_size[0] / 2)
            dy = canonical[1] * (self.workspace_size[1] / 2)
            grab = canonical[2] if len(canonical) > 2 else 0.0
            return np.array([dx, dy, grab])
        
        elif embodiment == 'ur5':
            action = canonical[:5].copy()
            for i in range(3):
                action[i] = canonical[i] * (self.workspace_size[i % 2] / 2)
            if len(canonical) > 3:
                action[3] = canonical[3] * self.max_rotation
            if len(canonical) > 4:
                action[4] = 1.0 if canonical[4] > 0.5 else 0.0
            return action
        
        else:
            action = canonical.copy()
            for i in range(min(len(action), 2)):
                action[i] = canonical[i] * (self.workspace_size[i] / 2)
            return action
    
    def normalize_obs(self, obs, obs_type='cable'):
        """Normalize observation to canonical frame.
        
        Args:
            obs: raw observation
            obs_type: 'cable', 'textile', 'tool_use'
        
        Returns:
            normalized_obs: observation in canonical frame
        """
        obs = np.asarray(obs, dtype=np.float64)
        
        if obs_type == 'cable':
            # Cable obs: [nodes (N*2), gripper (2), subgoals (2)]
            n_nodes = (len(obs) - 4) // 2
            nodes = obs[:n_nodes*2].reshape(-1, 2)
            gripper = obs[n_nodes*2:n_nodes*2+2]
            subgoals = obs[n_nodes*2+2:]
            
            # Normalize by workspace
            nodes_norm = (nodes - self.workspace_center) / (self.workspace_size / 2)
            gripper_norm = (gripper - self.workspace_center) / (self.workspace_size / 2)
            
            return np.concatenate([
                nodes_norm.flatten(),
                gripper_norm,
                subgoals,
            ]).astype(np.float32)
        
        else:
            # Generic normalization
            return obs
    
    def denormalize_obs(self, obs_norm, obs_type='cable'):
        """Convert normalized observation back to raw frame."""
        obs_norm = np.asarray(obs_norm, dtype=np.float64)
        
        if obs_type == 'cable':
            n_nodes = (len(obs_norm) - 4) // 2
            nodes_norm = obs_norm[:n_nodes*2].reshape(-1, 2)
            gripper_norm = obs_norm[n_nodes*2:n_nodes*2+2]
            subgoals = obs_norm[n_nodes*2+2:]
            
            nodes = nodes_norm * (self.workspace_size / 2) + self.workspace_center
            gripper = gripper_norm * (self.workspace_size / 2) + self.workspace_center
            
            return np.concatenate([
                nodes.flatten(),
                gripper,
                subgoals,
            ]).astype(np.float32)
        
        return obs_norm


def make_canonical_converter(env_type='cable'):
    """Factory for canonical converters per task."""
    if env_type == 'cable':
        return CanonicalActionFrame(
            workspace_bounds=(-0.95, 0.95, -0.30, 0.95)
        )
    elif env_type == 'textile':
        return CanonicalActionFrame(
            workspace_bounds=(-0.25, 0.25, -0.2, 0.25)
        )
    elif env_type == 'tool_use':
        return CanonicalActionFrame(
            workspace_bounds=(-0.25, 0.25, -0.2, 0.25)
        )
    else:
        return CanonicalActionFrame()
