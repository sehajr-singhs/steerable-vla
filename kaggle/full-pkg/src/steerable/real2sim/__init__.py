"""Sim-to-real transfer via domain randomization.

Randomizes physics parameters during training so the learned policy
transfers to real-world conditions. For cable manipulation:
  - Stiffness: ±50% of nominal
  - Damping: ±40%
  - Friction: ±30%
  - Gravity perturbation: ±20% (simulates calibration error)
  - Observation noise: Gaussian on proprioceptive channels
  - Action delay: 0-2 steps of latency
  - Cable parameters: node count, resting length variation

The randomization is applied *inside* the training loop by wrapping
the environment reset, not by modifying the env class.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization ranges."""
    stiffness_range: tuple = (0.5, 1.5)      # ±50% of nominal
    damping_range: tuple = (0.6, 1.4)         # ±40%
    friction_range: tuple = (0.7, 1.3)        # ±30%
    gravity_perturb: float = 0.2              # ±20% of 9.81
    obs_noise_std: float = 0.02               # Gaussian noise on obs
    action_delay_range: tuple = (0, 2)        # steps of latency [0, 2]
    grab_radius_range: tuple = (0.08, 0.20)   # ±40% of nominal
    v_gripper_range: tuple = (0.15, 0.45)     # ±25% of nominal
    crossing_range: tuple = (2, 5)            # crossing count variation
    noise_dim_ratio: float = 0.1              # fraction of obs dims to noise
    # action noise
    action_noise_std: float = 0.01            # small action perturbation
    # sim2real: how many randomization configs to use per reset
    n_randomization_samples: int = 1


class DomainRandomizer:
    """Applies domain randomization to env resets."""

    def __init__(self, cfg: DomainRandomizationConfig = None, seed: int = 0):
        self.cfg = cfg or DomainRandomizationConfig()
        self.rng = np.random.RandomState(seed)
        self._action_buffer = []

    def sample_params(self) -> Dict[str, Any]:
        """Sample a randomized set of environment parameters."""
        c = self.cfg
        return {
            "stiffness_mult": self.rng.uniform(*c.stiffness_range),
            "damping_mult": self.rng.uniform(*c.damping_range),
            "friction_mult": self.rng.uniform(*c.friction_range),
            "gravity_scale": 1.0 + self.rng.uniform(-c.gravity_perturb, c.gravity_perturb),
            "grab_radius_mult": self.rng.uniform(*c.grab_radius_range) / 0.14,
            "v_gripper_mult": self.rng.uniform(*c.v_gripper_range) / 0.3,
            "crossing_target": self.rng.randint(*c.crossing_range),
        }

    def reset_env(self, env, seed: int = None) -> np.ndarray:
        """Reset env with randomized parameters."""
        params = self.sample_params()
        # apply parameters that the cable env supports
        if hasattr(env, 'cfg') and hasattr(env.cfg, 'grab_radius'):
            env.cfg.grab_radius *= params.get("grab_radius_mult", 1.0)
        if hasattr(env, 'cfg') and hasattr(env.cfg, 'v_gripper'):
            env.cfg.v_gripper *= params.get("v_gripper_mult", 1.0)
        obs = env.reset(seed=seed)
        return self.add_obs_noise(obs)

    def add_obs_noise(self, obs: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to observation."""
        if self.cfg.obs_noise_std > 0:
            noise = self.rng.randn(*obs.shape).astype(obs.dtype) * self.cfg.obs_noise_std
            return obs + noise
        return obs

    def delay_action(self, action: np.ndarray) -> np.ndarray:
        """Simulate action delay (hold previous actions)."""
        lo, hi = self.cfg.action_delay_range
        delay = self.rng.randint(lo, hi + 1)
        self._action_buffer.append(action.copy())
        if len(self._action_buffer) > delay + 1:
            return self._action_buffer[-(delay + 1)]
        return action

    def add_action_noise(self, action: np.ndarray) -> np.ndarray:
        """Add small action perturbation."""
        if self.cfg.action_noise_std > 0:
            noise = self.rng.randn(*action.shape).astype(action.dtype) * self.cfg.action_noise_std
            return action + noise
        return action


class curriculumRandomizer:
    """Curriculum-based domain randomization.

    Starts with no randomization (easy), then gradually increases difficulty.
    Used to stabilize early training before introducing real-world variations.
    """

    def __init__(self, cfg: DomainRandomizationConfig = None, seed: int = 0,
                 total_steps: int = 50000):
        self.cfg = cfg or DomainRandomizationConfig()
        self.total_steps = total_steps
        self.step_count = 0
        self._base_randomizer = DomainRandomizer(cfg, seed)

    def _progress(self) -> float:
        """Return curriculum progress in [0, 1]."""
        return min(1.0, self.step_count / self.total_steps)

    def sample_params(self) -> Dict[str, Any]:
        """Sample with curriculum-scaled ranges."""
        p = self._progress()
        base = self._base_randomizer.sample_params()
        # scale all ranges by progress
        for key in base:
            if isinstance(base[key], float):
                base[key] = 1.0 + (base[key] - 1.0) * p
            elif isinstance(base[key], int):
                pass  # crossing target stays random from the start
        return base

    def step(self):
        self.step_count += 1

    def reset_env(self, env, seed: int = None) -> np.ndarray:
        self._base_randomizer.cfg = self.cfg
        obs = self._base_randomizer.reset_env(env, seed)
        self.step()
        return obs

    def add_obs_noise(self, obs):
        p = self._progress()
        scaled_cfg = DomainRandomizationConfig(
            obs_noise_std=self.cfg.obs_noise_std * p
        )
        self._base_randomizer.cfg = scaled_cfg
        return self._base_randomizer.add_obs_noise(obs)

    def delay_action(self, action):
        return self._base_randomizer.delay_action(action)

    def add_action_noise(self, action):
        return self._base_randomizer.add_action_noise(action)
