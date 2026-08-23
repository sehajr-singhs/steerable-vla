"""Baseline policies for comparison.

Implements:
  - BC: standard behavior cloning (MLP, no flow matching)
  - RT2Policy: RT-2 style VLM → discretized actions (Brohan et al. 2023)
  - ACT: Action Chunking Transformer with CVAE (Zhao et al. 2023)
  - DiffusionPolicy: diffusion-based action generation (Chi et al. 2023)
"""

from .bc_policy import BCPolicy
from .act_policy import ACTPolicy
from .diffusion_policy import DiffusionPolicy
from .rt2_policy import RT2Policy

__all__ = ["BCPolicy", "RT2Policy", "ACTPolicy", "DiffusionPolicy"]
